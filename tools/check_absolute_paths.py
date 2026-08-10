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
    "eval_kingdoms_pp.py",
    "nontryp_registry.py",
    "plot_benchmark.py",
    "plot_kol_overlap_vs_pp_gain.py",
    "plot_scatter_precision.py",
    "run_benchmark.py",
    "run_proteometools_saav.py",
    "infer_instanovo.sh",
    "infer_contranovo.sh",
    "run_on_slurm.sh",
    "run_kol_pipeline.sh",
    "run_eval_kingdoms_pp.sh",
    "run_plm_human_iso.sh",
    "run_plm_yeast.sh",
    "train_antibody_prior_human.sh",
    "train_antibody_prior_mouse.sh",
    "train_fusion_contranovo_swap50.sh",
    "scripts/zenodo_archive.py",
    "scripts/build_zenodo_archive.py",
    "scripts/validate_zenodo_archive.py",
)

MAINTAINED_SCRIPT_GLOBS = (
    "scripts/rerun_9s_inference_*asymbnln.slurm",
    "scripts/rerun_kol_mskb_swap50_eval*.slurm",
    "scripts/train_fusion_human_massivekb*.slurm",
    "scripts/train_pepLM_*.slurm",
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
