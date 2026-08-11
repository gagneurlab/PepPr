#!/bin/bash
#SBATCH --job-name=build_annot_nt
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_build_annot_nt.out
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=noninterruptive
# Build annotated MGFs for every nontryp_registry.RUNS entry after FragPipe finishes.

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
cd /data/nasif12/home_if12/khsam/dnps_hybrid

conda activate khsam
echo "=== Build annotated MGFs ==="
python build_annotated_mgf_nontryp.py
echo "=== Done $(date -Is) ==="
