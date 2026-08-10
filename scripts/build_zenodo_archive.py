#!/usr/bin/env python3
"""Build the DNPS-Hybrid Zenodo bundle from declarative include specs."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from zenodo_archive import (
    collect_candidates,
    default_specs,
    find_repo_root,
    load_spec_file,
    nine_species_staged_include_specs,
    parse_include,
    prune_unselected,
    resolve_roots,
    stage_archive_readme,
    stage_candidates,
    stage_nine_species_mgfs,
    write_manifests,
)

def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage the final, reproducible DNPS-Hybrid Zenodo bundle. Existing "
            "matching files are resumed; differing files require --overwrite."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="archive staging directory")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_env_path("DNPS_SOURCE_DATA_PATH"),
        help="original project data root (or DNPS_SOURCE_DATA_PATH)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="source repository (default: auto-detected from this script)",
    )
    parser.add_argument(
        "--kol-root", type=Path, default=_env_path("DNPS_SOURCE_KOL_PATH")
    )
    parser.add_argument(
        "--mutations-root",
        type=Path,
        default=_env_path("DNPS_SOURCE_MUTATIONS_PATH"),
    )
    parser.add_argument(
        "--smsnet-root",
        type=Path,
        default=_env_path("DNPS_SOURCE_SMSNET_PATH"),
        help="SMSNet checkout/output root (or DNPS_SOURCE_SMSNET_PATH)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "extra ROOT:GLOB:DEST:CATEGORY[:required|optional] include; ROOT may "
            "be data, repo, kol, mutations, or an absolute path. DEST supports "
            "{relative}, {name}, {stem}, {run}, {parent}, {tail}, and "
            "{payload_tail}; {index} is a zero-padded match index. A JSON "
            "IncludeSpec object is also accepted"
        ),
    )
    parser.add_argument(
        "--spec-file",
        action="append",
        type=Path,
        default=[],
        help="JSON file containing an include-spec list",
    )
    parser.add_argument(
        "--only-explicit",
        action="store_true",
        help="omit built-in canonical specs and use only --include/--spec-file",
    )
    parser.add_argument("--dry-run", action="store_true", help="plan and hash without writing")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace differing destination files",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="remove output files not selected by the current manifest plan",
    )
    symlinks = parser.add_mutually_exclusive_group()
    symlinks.add_argument(
        "--dereference-symlinks",
        dest="dereference",
        action="store_true",
        default=True,
        help="copy symlink targets into the archive (default)",
    )
    symlinks.add_argument(
        "--preserve-symlinks",
        dest="dereference",
        action="store_false",
        help="preserve source symlinks; validation rejects broken links",
    )
    parser.add_argument(
        "--allow-missing-required",
        action="store_true",
        help="build a partial archive despite missing required spec groups",
    )
    parser.add_argument(
        "--skip-nine-species",
        action="store_true",
        help="do not stage external/nine_species/*.mgf benchmark inputs",
    )
    return parser


def _describe_missing(title: str, specs) -> None:
    if not specs:
        return
    print(f"{title} ({len(specs)}):", file=sys.stderr)
    for spec in specs:
        print(
            f"  - {spec.name}: root={spec.root!r}, glob={spec.glob!r}, "
            f"destination={spec.destination!r}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root is None:
        print(
            "error: set --data-root or DNPS_SOURCE_DATA_PATH",
            file=sys.stderr,
        )
        return 2
    if args.kol_root is None:
        args.kol_root = args.data_root
    if args.mutations_root is None:
        args.mutations_root = args.data_root
    repo_root = args.repo_root or find_repo_root(Path(__file__).parent)
    roots = resolve_roots(
        args.data_root,
        repo_root,
        args.kol_root,
        args.mutations_root,
        args.smsnet_root,
    )

    specs = [] if args.only_explicit else list(default_specs())
    for spec_file in args.spec_file:
        specs.extend(load_spec_file(spec_file))
    output = args.output.expanduser().resolve()
    try:
        specs.extend(parse_include(value, i) for i, value in enumerate(args.include, 1))
        if not args.skip_nine_species:
            if args.dry_run:
                print(
                    f"Would stage nine-species MGFs under {output / 'external/nine_species'}",
                    file=sys.stderr,
                )
                if (output / "external/nine_species").is_dir():
                    specs.extend(nine_species_staged_include_specs(output))
                else:
                    print(
                        "  (nine-species MGFs absent; dry-run manifest omits them)",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"Staging nine-species MGFs under {output / 'external/nine_species'}...",
                    file=sys.stderr,
                )
                stage_nine_species_mgfs(
                    roots["data"],
                    output,
                    overwrite=args.overwrite,
                    dereference=args.dereference,
                )
                specs.extend(nine_species_staged_include_specs(output))
        candidates, missing_required, missing_optional = collect_candidates(specs, roots)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _describe_missing("Missing required groups", missing_required)
    _describe_missing("Missing optional groups", missing_optional)
    if missing_required and not args.allow_missing_required:
        print(
            "Refusing to stage an incomplete archive. Supply the missing files, "
            "correct a source root, or use --allow-missing-required.",
            file=sys.stderr,
        )
        return 2
    if not candidates:
        print("error: no files matched the manifest specs", file=sys.stderr)
        return 2

    for name, root in roots.items():
        if output == root:
            print(f"error: --output cannot equal the {name} source root", file=sys.stderr)
            return 2

    try:
        records, counts = stage_candidates(
            candidates,
            output,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            dereference=args.dereference,
            progress=True,
        )
        if args.prune and not args.dry_run:
            removed = prune_unselected(
                output, (candidate.destination for candidate in candidates)
            )
            print(f"Pruned {removed} unselected file(s).")
        readme_record, readme_action = stage_archive_readme(
            output, dry_run=args.dry_run, overwrite=args.overwrite
        )
        records.append(readme_record)
        records.sort(key=lambda record: record.destination)
        counts[readme_action] += 1
        if not args.dry_run:
            write_manifests(output, records)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total_size = sum(record.size for record in records)
    mode = "Dry run" if args.dry_run else "Archive staged"
    print(
        f"{mode}: {len(records)} files, {total_size:,} bytes; "
        f"copied={counts['copied']}, resumed={counts['resumed']}, "
        f"planned={counts['planned']}"
    )
    if not args.dry_run:
        print(f"Manifest: {output / 'MANIFEST.tsv'}")
        print(f"Manifest: {output / 'MANIFEST.json'}")
        print(
            "Next: python scripts/build_zenodo_release_tarball.py "
            f"--staging {output}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
