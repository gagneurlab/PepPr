#!/bin/bash
#SBATCH --job-name=9s_infer_mouse_asymbnln
#SBATCH --output=data/slurm/%j_9s_inference_mouse_asymbnln.out
#SBATCH --time=24:00:00
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
# Apply the asymbnln (BN-cas + LN-plm) MassiveKB fusion head to mouse 9s
# same-species inference (mouse pepLM). Counterpart to the LN/LN run that
# produced 9s_mouse_hybrid_19137422.mztab. Output filename gets the new
# SLURM_JOB_ID so the LN/LN and BN/BN files survive untouched.

export TMUX="${TMUX:-noninteractive}"
set -euo pipefail
cd "$REPO_ROOT"

DATA_ROOT="$DNPS_DATA_PATH"
export DNPS_FUSION_MODEL_PATH="$DATA_ROOT/models/casanovo/fusion_model_asymbnln.pth"
export DNPS_NULL_MODEL_PATH="$DATA_ROOT/models/casanovo/null_model_asymbnln.pth"
export DNPS_LANCE_DIR="${TMPDIR:-/tmp}/casanovo_lance_${SLURM_JOB_ID}"

if [[ ! -s "$DNPS_FUSION_MODEL_PATH" ]]; then
  echo "[fatal] missing $DNPS_FUSION_MODEL_PATH" >&2
  exit 2
fi

echo "=== Pre-flight ==="
echo "fusion:    $DNPS_FUSION_MODEL_PATH ($(stat -c %y $DNPS_FUSION_MODEL_PATH))"
echo "null:      $DNPS_NULL_MODEL_PATH"
echo "lance_dir: $DNPS_LANCE_DIR  (per-job, isolated)"
echo

DNPS_SPECIES=mouse DNPS_PLM_SPECIES=mouse \
  "$PYTHON_BIN" dnps_hybrid/inference.py auto \
    --dataset NINE_SPECIES_DATASET --use-plm true
