#!/bin/bash
#SBATCH --job-name=9s_yeast
#SBATCH --output=data/slurm/%A_%a_9s_yeast.out
#SBATCH --array=0-3
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
# External nine-species ProForma tree; intentionally excluded from Zenodo.
: "${DNPS_NINE_SPECIES_PATH:?Set DNPS_NINE_SPECIES_PATH to the external nine-species benchmark root}"
export DNPS_NINE_SPECIES_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi
#
# Same-species +PP arm for the canonical S288c yeast proteome (UP000002311).
# The 27 nine-species yeast benchmark ProForma MGFs are split across a
# four-task array. The Casanovo-only baseline is proteome-independent and
# does not need to be rerun.

export TMUX="${TMUX:-noninteractive}"
set -euo pipefail
cd "$REPO_ROOT"

DATA_ROOT="$DNPS_DATA_PATH"
export DNPS_FUSION_MODEL_PATH="$DATA_ROOT/models/casanovo/fusion_model_asymbnln.pth"
export DNPS_NULL_MODEL_PATH="$DATA_ROOT/models/casanovo/null_model_asymbnln.pth"
export DNPS_LANCE_DIR="${TMPDIR:-/tmp}/casanovo_lance_${SLURM_JOB_ID}"
export DNPS_MGF_SHARD="${SLURM_ARRAY_TASK_ID}/4"

if [[ ! -s "$DNPS_FUSION_MODEL_PATH" ]]; then
  echo "[fatal] missing $DNPS_FUSION_MODEL_PATH" >&2
  exit 2
fi

echo "=== Pre-flight (array task ${SLURM_ARRAY_TASK_ID}/4) ==="
echo "fusion:    $DNPS_FUSION_MODEL_PATH ($(stat -c %y $DNPS_FUSION_MODEL_PATH))"
echo "null:      $DNPS_NULL_MODEL_PATH"
echo "shard:     $DNPS_MGF_SHARD"
echo "lance_dir: $DNPS_LANCE_DIR  (per-job, isolated)"
echo

DNPS_SPECIES=yeast DNPS_PLM_SPECIES=yeast \
  "$PYTHON_BIN" dnps_hybrid/inference.py auto \
    --dataset NINE_SPECIES_DATASET --use-plm true
