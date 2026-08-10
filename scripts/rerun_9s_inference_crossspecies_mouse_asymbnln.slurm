#!/bin/bash
#SBATCH --job-name=9s_mouse_humaniso_asymbnln
#SBATCH --output=data/slurm/%j_9s_mouse_humaniso_asymbnln.out
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
# Cross-species +PP arm for mouse using the asymbnln (BN-cas + LN-plm) MassiveKB
# fusion head with the human_iso PLM. Split out from
# rerun_9s_inference_crossspecies_asymbnln.slurm (job 19142933 bundles the 5
# fast species only) because mouse is too large to fit in that 24h window.
#
# Isolation: per-job DNPS_LANCE_DIR avoids the Lance commit conflicts that
# crashed 19088271/19088272 when sharing the casanovo default lance_dir.

export TMUX="${TMUX:-noninteractive}"
set -euo pipefail
cd "$REPO_ROOT"

DATA_ROOT="$DNPS_DATA_PATH"
export DNPS_FUSION_MODEL_PATH="$DATA_ROOT/models/human_iso_asymbnln/fusion_model.pth"
export DNPS_NULL_MODEL_PATH="$DATA_ROOT/models/human_iso_asymbnln/null_model.pth"
export DNPS_LANCE_DIR="${TMPDIR:-/tmp}/casanovo_lance_${SLURM_JOB_ID}"

if [[ ! -s "$DNPS_FUSION_MODEL_PATH" ]]; then
  echo "[fatal] missing $DNPS_FUSION_MODEL_PATH" >&2
  exit 2
fi

echo "=== Pre-flight ==="
echo "fusion:    $DNPS_FUSION_MODEL_PATH ($(stat -c %y $DNPS_FUSION_MODEL_PATH))"
echo "null:      $DNPS_NULL_MODEL_PATH"
echo "lance_dir: $DNPS_LANCE_DIR  (per-job, isolated)"
echo "PLM:       human_iso (cross-species)"
echo

DNPS_SPECIES=mouse DNPS_PLM_SPECIES=human_iso \
  "$PYTHON_BIN" dnps_hybrid/inference.py auto \
    --dataset NINE_SPECIES_DATASET --use-plm true
