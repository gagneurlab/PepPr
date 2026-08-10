#!/bin/bash
#SBATCH --job-name=benchmark_9s
#SBATCH --output=data/slurm/%j_powernovo.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#
# PowerNovo baseline launcher. Usage: sbatch slurm_powernovo.sh <species>  (default human)
source /data/nasif12/home_if12/khsam/.bashrc
conda activate khsam
set -euo pipefail
set -x
cd /data/nasif12/home_if12/khsam/dnps_hybrid
python run_benchmark.py "${1:-human}"
