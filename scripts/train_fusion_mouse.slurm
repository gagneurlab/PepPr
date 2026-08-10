#!/bin/bash
#SBATCH --job-name=dnps_fusion_mouse
#SBATCH --output=data/slurm/%j_train_fusion_mouse.out
#SBATCH --time=04:00:00
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
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi
#
# Train the Casanovo+pepLM fusion head for mouse pepLM on **MassiveKB** PSMs.
# Uses PLM teacher scores under mus_musculus/ (inference.py plm_teacher) plus
# shared Casanovo teacher tensors in casanovo/.
#
# Prereqs (run once per refreshed teacher data):
#   export DNPS_SPECIES=mouse
#   python dnps_hybrid/inference.py plm_teacher
#
# Submit:
#   sbatch $REPO_ROOT/scripts/train_fusion_mouse.slurm

export TMUX="${TMUX:-noninteractive}"
set -euo pipefail
cd "$REPO_ROOT"

export DNPS_SPECIES=mouse
export DNPS_PLM_SPECIES=mouse

TRAIN="$DNPS_DATA_PATH/mus_musculus/plm_psm_teacher_scores_train.pt"
if [[ ! -f "$TRAIN" ]]; then
  echo "Missing $TRAIN — run first: DNPS_SPECIES=mouse python dnps_hybrid/inference.py plm_teacher" >&2
  exit 2
fi

echo "Training fusion (DNPS_SPECIES=$DNPS_SPECIES) ..."
"$PYTHON_BIN" dnps_hybrid/train_fusion_head.py
echo "Done."
