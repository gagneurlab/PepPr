#!/bin/bash
#SBATCH --job-name=train_fusion_contranovo_swap50
#SBATCH --output=data/slurm/%j_train_fusion_contranovo_swap50.out
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga25,ouga26

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi
#
# Retrain the ContraNovo fusion head with the current asymbnln + swap50
# methodology (matches the Casanovo asymbnln recipe). Reuses the existing
# Apr-27 ContraNovo teacher artifacts under casanovo/ (no teacher pass needed).
#
# Outputs:
#   casanovo/contranovo_fusion_asymbnln_swap50.pth
#   casanovo/contranovo_null_asymbnln_swap50.pth

set -euo pipefail
set -x

SHARED_DIR=$DNPS_DATA_PATH/casanovo

export DNPS_FUSION_BACKBONE=contranovo
export DNPS_CONTRANOVO_FUSION_MODEL_PATH=$SHARED_DIR/contranovo_fusion_asymbnln_swap50.pth
export DNPS_CONTRANOVO_NULL_MODEL_PATH=$SHARED_DIR/contranovo_null_asymbnln_swap50.pth
export DNPS_FUSION_PLM_TOP2_SWAP_FRAC=0.5
export DNPS_FUSION_PLM_RAND_SWAP_FRAC=0.0
export DNPS_FUSION_EPOCHS=16
unset DNPS_FUSION_OUTPUT_IL || true

for f in \
  contranovo_teacher_scores_train.pt \
  contranovo_teacher_scores_test.pt \
  contranovo_fusion_y_train.pt \
  contranovo_fusion_y_test.pt \
  contranovo_plm_psm_teacher_scores_train.pt \
  contranovo_plm_psm_teacher_scores_test.pt
do
  if [ ! -s "$SHARED_DIR/$f" ]; then
    echo "[fatal] missing $SHARED_DIR/$f" >&2
    exit 2
  fi
done

ls -lh "$SHARED_DIR"/contranovo_*.pt 2>/dev/null | head

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
cd "$REPO_ROOT"

echo "=== Train ContraNovo+pepLM fusion (asymbnln, swap50, 16 epochs) ==="
"$PYTHON_BIN" -m dnps_hybrid.train_fusion_head

ls -lh "$DNPS_CONTRANOVO_FUSION_MODEL_PATH" "$DNPS_CONTRANOVO_NULL_MODEL_PATH"
echo "=== Done ==="
