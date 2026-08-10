#!/bin/bash
#SBATCH --job-name=9s_yeast_humaniso_asymbnln
#SBATCH --output=data/slurm/%A_%a_9s_yeast_humaniso_asymbnln.out
#SBATCH --array=0-3
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga05,ouga06,ouga07,ouga08,ouga09,ouga12,ouga26

#SBATCH --chdir=/data/nasif12/home_if12/khsam/dnps_hybrid

REPO_ROOT="${SLURM_SUBMIT_DIR:-/data/nasif12/home_if12/khsam/dnps_hybrid}"
source "${SLURM_SUBMIT_DIR:-/data/nasif12/home_if12/khsam/dnps_hybrid}/scripts/dnps_data_path.sh"
export DNPS_NINE_SPECIES_PATH="${DNPS_NINE_SPECIES_PATH:-$DNPS_DATA_PATH/external/nine_species}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi
#
# Cross-species +PP arm for yeast using the asymbnln (BN-cas + LN-plm) MassiveKB
# fusion head with the human_iso PLM. Split into its own job (cf. the same-species
# rerun_9s_inference_yeast_asymbnln.slurm) because the yeast benchmark
# previously ran ~1-2 days and would not fit in the 24h crossspecies bundle
# (job 19142933).
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
export DNPS_MGF_SHARD="${SLURM_ARRAY_TASK_ID}/4"

if [[ ! -s "$DNPS_FUSION_MODEL_PATH" ]]; then
  echo "[fatal] missing $DNPS_FUSION_MODEL_PATH" >&2
  exit 2
fi

echo "=== Pre-flight (array task ${SLURM_ARRAY_TASK_ID}/4) ==="
echo "fusion:    $DNPS_FUSION_MODEL_PATH ($(stat -c %y $DNPS_FUSION_MODEL_PATH))"
echo "null:      $DNPS_NULL_MODEL_PATH"
echo "lance_dir: $DNPS_LANCE_DIR  (per-job, isolated)"
echo "shard:     $DNPS_MGF_SHARD"
echo "PLM:       human_iso (cross-species)"
echo

DNPS_SPECIES=yeast DNPS_PLM_SPECIES=human_iso \
  "$PYTHON_BIN" dnps_hybrid/inference.py auto \
    --dataset NINE_SPECIES_DATASET --use-plm true
