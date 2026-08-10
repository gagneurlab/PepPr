#!/bin/bash
#SBATCH --job-name=9s_infer_same_asymbnln
#SBATCH --output=data/slurm/%j_9s_inference_samespecies_asymbnln.out
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
# Same-species +PP arm for the remaining nine-species benchmark members using
# the asymbnln (BN-cas + LN-plm) MassiveKB fusion head.
#
# Human and mouse asymbnln runs already exist (19138512, 19138084). Yeast and
# bacillus are split into their own 72h jobs (rerun_9s_inference_{yeast,bacillus}_asymbnln.slurm)
# because they each take ~1-2 days. This job bundles the five remaining species.
#
# Isolation: per-job DNPS_LANCE_DIR avoids the Lance commit conflicts that
# crashed 19088271/19088272 when sharing the casanovo default lance_dir.

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

SPECIES_LIST=( ${DNPS_SPECIES_LIST:-honeybee tomato cowpea archaeon endoloripes} )

echo "=== Pre-flight ==="
echo "fusion:    $DNPS_FUSION_MODEL_PATH ($(stat -c %y $DNPS_FUSION_MODEL_PATH))"
echo "null:      $DNPS_NULL_MODEL_PATH"
echo "lance_dir: $DNPS_LANCE_DIR  (per-job, isolated)"
echo "species:   ${SPECIES_LIST[*]}  (each w/ its own PLM, asymbnln fusion head)"
echo

for sp in "${SPECIES_LIST[@]}"; do
  echo "================================================"
  echo "[$sp] DNPS_SPECIES=$sp  DNPS_PLM_SPECIES=$sp"
  echo "================================================"
  DNPS_SPECIES="$sp" DNPS_PLM_SPECIES="$sp" \
    "$PYTHON_BIN" dnps_hybrid/inference.py auto \
      --dataset NINE_SPECIES_DATASET --use-plm true
  echo "[$sp] done"
  echo
done

echo "=== All species done ==="
