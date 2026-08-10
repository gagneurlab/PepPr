#!/bin/bash
#SBATCH --job-name=dnps_fus_hmskb_swap50_asymbnln
#SBATCH --output=data/slurm/%j_train_fusion_human_mskb_swap50_asymbnln.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi
#
# Asymmetric BN(cas) + LN(plm) fusion-head variant of
# train_fusion_human_massivekb_swap50.slurm. FusionModel now uses
# BatchNorm1d on the casanovo stream and LayerNorm on the pepLM stream
# (dnps_hybrid/model.py FusionModel). Writes to *_asymbnln.pth so it
# doesn't clobber the LN/LN or original BN/BN baseline checkpoints.

export TMUX="${TMUX:-noninteractive}"
set -euo pipefail
cd "$REPO_ROOT"

export DNPS_SPECIES=human
export DNPS_PLM_SPECIES=human
export DNPS_FUSION_PLM_TOP2_SWAP_FRAC=0.50

DATA_ROOT="$DNPS_DATA_PATH"
export DNPS_FUSION_MODEL_PATH="$DATA_ROOT/casanovo/fusion_model_asymbnln.pth"
export DNPS_NULL_MODEL_PATH="$DATA_ROOT/casanovo/null_model_asymbnln.pth"

for f in casanovo_teacher_scores_train.pt casanovo_teacher_scores_test.pt \
         plm_psm_teacher_scores_train.pt plm_psm_teacher_scores_test.pt \
         fusion_y_train.pt fusion_y_test.pt; do
  if [[ ! -s "$DATA_ROOT/casanovo/$f" ]]; then
    echo "[fatal] missing teacher artifact: $DATA_ROOT/casanovo/$f" >&2
    exit 2
  fi
done

export DNPS_FUSION_EPOCHS="${DNPS_FUSION_EPOCHS:-16}"

echo "=== Pre-flight ==="
echo "swap_frac:  $DNPS_FUSION_PLM_TOP2_SWAP_FRAC"
echo "epochs:     $DNPS_FUSION_EPOCHS"
echo "fusion out: $DNPS_FUSION_MODEL_PATH"
echo "null out:   $DNPS_NULL_MODEL_PATH"

echo
echo "=== Train fusion head ==="
"$PYTHON_BIN" dnps_hybrid/train_fusion_head.py
echo "Done -- fusion: $DNPS_FUSION_MODEL_PATH"
ls -lh "$DNPS_FUSION_MODEL_PATH" "$DNPS_NULL_MODEL_PATH"
