#!/usr/bin/env python3
"""Create the single-file Zenodo release tarball from a staged archive directory."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from zenodo_archive import sha256_file  # noqa: E402


def build_release_tarball(
    staging_dir: Path,
    output: Path,
    *,
    compresslevel: int = 6,
) -> tuple[Path, str, int]:
    staging_dir = staging_dir.expanduser().resolve()
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"staging directory not found: {staging_dir}")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_name = staging_dir.name

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".partial",
        dir=output.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    file_count = 0
    # Manifest-driven: ship exactly the files listed in MANIFEST.json (plus the
    # manifests + README), never a raw directory walk. The staging dir may hold
    # local-only material not meant for release — e.g. the 28 GB external
    # nine-species tree (README: obtain separately) or derived *.npz curves.
    manifest_path = staging_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MANIFEST.json not found in {staging_dir}")
    allowed = {f["destination"] for f in json.loads(manifest_path.read_text())["files"]}
    allowed |= {"MANIFEST.json", "MANIFEST.tsv", "ARCHIVE_README.md"}
    missing = sorted(d for d in allowed if not (staging_dir / d).is_file())
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest entries absent from staging dir, e.g. {missing[:5]}"
        )
    try:
        with tarfile.open(tmp_path, "w:gz", compresslevel=compresslevel) as archive:
            for rel in sorted(allowed):
                path = staging_dir / rel
                relative = path.relative_to(staging_dir.parent)
                archive.add(path, arcname=relative.as_posix(), recursive=False)
                file_count += 1
        os.replace(tmp_path, output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    checksum = sha256_file(output)
    return output, checksum, file_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging",
        type=Path,
        required=True,
        help="staged archive directory (for example dnps_hybrid_zenodo/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="release tarball path (default: <parent>/<staging-name>.tar.gz)",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=6,
        choices=range(1, 10),
        help="gzip compression level (default: 6)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    staging = args.staging.expanduser().resolve()
    output = args.output or staging.with_suffix(".tar.gz")
    try:
        tarball, checksum, file_count = build_release_tarball(
            staging,
            output,
            compresslevel=args.compresslevel,
        )
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    size = tarball.stat().st_size
    print(f"Wrote {tarball}")
    print(f"Files: {file_count:,}")
    print(f"Size:  {size:,} bytes ({size / 1e9:.2f} GB)")
    print(f"SHA-256: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
