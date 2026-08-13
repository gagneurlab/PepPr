#!/usr/bin/env python3
"""Run a de novo benchmark baseline (PowerNovo) on a nine-species dataset.

Usage:
    python run_benchmark.py <species>     # e.g. human, mouse, yeast, ...

Species names and their result run-names are defined in peptide_priors.const.SPECIES.
"""
import argparse

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from peptide_priors.const import (PROJECT_ROOT, SPECIES, nine_species_benchmark_dir,
                               result_run_path)
from powernovo.run import run_inference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("species", choices=sorted(SPECIES),
                    help="nine-species dataset to run PowerNovo on")
    args = ap.parse_args()
    run_inference(
        inputs=nine_species_benchmark_dir(args.species),
        working_folder=f"{PROJECT_ROOT}/powernovo",
        output_folder=result_run_path(SPECIES[args.species]["run_name"]),
        use_assembler=False,
        protein_inference=False,
        use_bert=True,
        batch_size=16,
    )


if __name__ == "__main__":
    main()
