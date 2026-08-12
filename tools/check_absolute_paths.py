#!/usr/bin/env python3
"""Reject machine-specific paths in maintained paper workflows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MACHINE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])/(?:data|s|scratch|opt|mnt|home)/[^\s\"'`]+"
)

MAINTAINED_ROOT_FILES = (
    "README.md",
    "casanovo/casanovo/config.yaml",
    "casanovo_integration.patch",
    "contranovo_integration.patch",
    "experiments/train.sh",
    "experiments/fig2/run_nine_species_inference.slurm",
    "experiments/fig2/run_baseline.sh",
    "experiments/fig2/run_benchmark.py",
    "experiments/fig3/run_inference.slurm",
    "experiments/fig3/run_kol_eval.sh",
    "experiments/fig3/eval_kingdoms_pp.py",
    "experiments/fig4/run_mab.sh",
    "experiments/fig4/mab_prep.py",
    "experiments/fig4/nontryp_registry.py",
)

MAINTAINED_SCRIPT_GLOBS = (
    "experiments/**/plot_*.py",
    "experiments/plot_*.py",
)


def maintained_paths() -> list[Path]:
    paths = list((REPO_ROOT / "dnps_hybrid").glob("*.py"))
    paths.extend(REPO_ROOT / name for name in MAINTAINED_ROOT_FILES)
    for pattern in MAINTAINED_SCRIPT_GLOBS:
        paths.extend(REPO_ROOT.glob(pattern))
    return sorted({path for path in paths if path.is_file()})


def violations(paths: list[Path]) -> list[tuple[Path, int, str]]:
    found = []
    for path in paths:
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if MACHINE_PATH.search(line):
                found.append((path, line_number, line.strip()))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Paths to scan instead of the maintained-workflow default set.",
    )
    args = parser.parse_args()
    paths = [
        path if path.is_absolute() else REPO_ROOT / path
        for path in args.paths
    ] or maintained_paths()
    found = violations(paths)
    for path, line_number, line in found:
        print(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line}")
    if found:
        print(f"\nFound {len(found)} machine-specific absolute path(s).")
        return 1
    print(f"Checked {len(paths)} maintained files: no machine-specific paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
