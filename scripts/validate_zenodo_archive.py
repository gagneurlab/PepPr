#!/usr/bin/env python3
"""Validate a staged DNPS-Hybrid Zenodo archive."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zenodo_archive import (
    load_manifest,
    validate_archive_paths,
    validate_external_benchmark_scope,
    validate_manifest,
    validate_mgfs,
    validate_mztabs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="staged archive directory")
    parser.add_argument(
        "--checksums",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify SHA-256 checksums (default: enabled; use --no-checksums for speed)",
    )
    parser.add_argument(
        "--proforma-sample-size",
        type=int,
        default=100,
        help="maximum annotated sequences parsed per MGF when pyteomics is available",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive = args.archive.expanduser().resolve()
    if not archive.is_dir():
        print(f"error: archive directory does not exist: {archive}", file=sys.stderr)
        return 2
    if args.proforma_sample_size < 0:
        print("error: --proforma-sample-size must be non-negative", file=sys.stderr)
        return 2

    try:
        records = load_manifest(archive)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: invalid or missing MANIFEST.json: {exc}", file=sys.stderr)
        return 2

    errors = []
    errors.extend(validate_manifest(archive, records, args.checksums))
    errors.extend(validate_archive_paths(archive))
    errors.extend(validate_external_benchmark_scope(archive))
    mgf_errors, parser_status = validate_mgfs(archive, args.proforma_sample_size)
    errors.extend(mgf_errors)
    errors.extend(validate_mztabs(archive))

    print(
        f"Validated {len(records)} manifest entries "
        f"({'with' if args.checksums else 'without'} checksums); {parser_status}."
    )
    if errors:
        print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
