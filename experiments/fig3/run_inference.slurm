#!/bin/bash
#SBATCH --job-name=run_inference
#SBATCH --output=data/slurm/%A_%a_run_inference.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# General Casanovo +/- PepPr inference on a registered benchmark dataset, via
# dnps_hybrid/inference.py auto. One driver for every DatasetPaths defined in
# const.py (e.g. PROTEOMETOOLS_SAAV_DATASET). The nine-species and Kingdoms-of-
# Life benchmarks have their own drivers (run_nine_species_inference.slurm,
# run_kol_eval.sh) because of their species/model-path specialisation.
#
# Usage:
#   sbatch experiments/fig3/run_inference.slurm <DATASET_NAME> [--use-plm true|false|both]
#
# Example (ProteomeTools SAAV, both arms):
#   sbatch experiments/fig3/run_inference.slurm PROTEOMETOOLS_SAAV_DATASET
#
# Large datasets can be sharded across a Slurm array (round-robin over MGFs):
#   sbatch --array=0-3 experiments/fig3/run_inference.slurm PROTEOMETOOLS_SAAV_DATASET

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

export TMUX="${TMUX:-noninteractive}"
set -euo pipefail
cd "$REPO_ROOT"

DATASET="${1:?usage: run_inference.slurm <DATASET_NAME> [--use-plm true|false|both]}"
shift || true
USE_PLM="both"
if [[ "${1:-}" == "--use-plm" ]]; then
  USE_PLM="${2:?--use-plm needs a value}"; shift 2
fi

# Per-job isolated Lance cache (avoids commit conflicts between parallel jobs).
export DNPS_LANCE_DIR="${TMPDIR:-/tmp}/casanovo_lance_${SLURM_JOB_ID:-$$}"
# Optional array sharding of the dataset's MGF set.
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  export DNPS_MGF_SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
fi

echo "=== Pre-flight ==="
echo "dataset:   $DATASET"
echo "use_plm:   $USE_PLM"
echo "lance_dir: $DNPS_LANCE_DIR  (per-job, isolated)"
[[ -n "${DNPS_MGF_SHARD:-}" ]] && echo "shard:     $DNPS_MGF_SHARD"
echo

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi

"$PYTHON_BIN" dnps_hybrid/inference.py auto --dataset "$DATASET" --use-plm "$USE_PLM"

echo "=== Done ==="
