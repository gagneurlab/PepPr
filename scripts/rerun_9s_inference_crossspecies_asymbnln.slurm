#!/bin/bash
#SBATCH --job-name=9s_cross_humaniso_asymbnln
#SBATCH --output=data/slurm/%j_9s_cross_humaniso_asymbnln.out
#SBATCH --time=24:00:00
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
# Cross-species +PP arm for the nine-species benchmark using the asymbnln
# (BN-cas + LN-plm) MassiveKB fusion head with the human_iso PLM applied to
# every non-human species. Counterpart to rerun_9s_inference_samespecies_asymbnln.slurm
# (job 19138634), which runs each species against its own PLM.
#
# Human is excluded (PLM=human + SPECIES=human collapses to the same-species
# arm). Mouse asymbnln cross-species + yeast/bacillus take their own jobs
# because of the 24h wall limit. This job bundles the 5 fast species.
#
# Isolation: per-job DNPS_LANCE_DIR avoids Lance commit conflicts.

export TMUX="${TMUX:-noninteractive}"
set -euo pipefail
cd "$REPO_ROOT"

DATA_ROOT="$DNPS_DATA_PATH"
export DNPS_FUSION_MODEL_PATH="$DATA_ROOT/models/human_iso_asymbnln/fusion_model.pth"
export DNPS_NULL_MODEL_PATH="$DATA_ROOT/models/human_iso_asymbnln/null_model.pth"
export DNPS_LANCE_DIR="${TMPDIR:-/tmp}/casanovo_lance_${SLURM_JOB_ID}"
export DNPS_PLM_SPECIES=human_iso

if [[ ! -s "$DNPS_FUSION_MODEL_PATH" ]]; then
  echo "[fatal] missing $DNPS_FUSION_MODEL_PATH" >&2
  exit 2
fi

SPECIES_LIST=( ${DNPS_SPECIES_LIST:-tomato cowpea archaeon endoloripes} )

echo "=== Pre-flight ==="
echo "fusion:    $DNPS_FUSION_MODEL_PATH ($(stat -c %y $DNPS_FUSION_MODEL_PATH))"
echo "null:      $DNPS_NULL_MODEL_PATH"
echo "lance_dir: $DNPS_LANCE_DIR  (per-job, isolated)"
echo "PLM:       human_iso (cross-species)"
echo "species:   ${SPECIES_LIST[*]}  (each w/ human_iso PLM, asymbnln fusion head)"
echo

for sp in "${SPECIES_LIST[@]}"; do
  echo "================================================"
  echo "[$sp] DNPS_SPECIES=$sp  DNPS_PLM_SPECIES=$DNPS_PLM_SPECIES"
  echo "================================================"
  DNPS_SPECIES="$sp" \
    "$PYTHON_BIN" dnps_hybrid/inference.py auto \
      --dataset NINE_SPECIES_DATASET --use-plm true
  echo "[$sp] done"
  echo
done

echo "=== All species done ==="
