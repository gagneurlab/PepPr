#!/bin/bash
#SBATCH --job-name=9s_infer_cross_arc_endo
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_9s_inference_crossspecies_arc_endo.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Fill in the cross-species PLM (human) + human-trained fusion head arms for
# archaeon and endoloripes. The companion job rerun_9s_inference_human_swap50.slurm
# (19087317) timed out mid-archaeon and never reached endoloripes; the bare
# 9s_<sp>_plmhuman_hybrid.mztab symlinks for these two species were removed
# so the plot would not mix fusion heads.
#
# Both invocations use the same human fusion head as the same-species runs.

export TMUX="${TMUX:-noninteractive}"
source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
cd /data/nasif12/home_if12/khsam/dnps_hybrid

DATA_ROOT="${DATA_PATH:-/s/project/denovo-prosit/SamKhan/dnps_hybrid}"
export DNPS_FUSION_MODEL_PATH="$DATA_ROOT/casanovo/fusion_model.pth"
export DNPS_NULL_MODEL_PATH="$DATA_ROOT/casanovo/null_model.pth"
export DNPS_PLM_SPECIES=human

if [[ ! -s "$DNPS_FUSION_MODEL_PATH" ]]; then
  echo "[fatal] missing $DNPS_FUSION_MODEL_PATH" >&2
  exit 2
fi

# Run endoloripes first (smaller, faster) so archaeon's longer run has the
# remaining wall time.
SPECIES_LIST=( ${DNPS_SPECIES_LIST:-endoloripes archaeon} )

echo "=== Pre-flight ==="
echo "fusion: $DNPS_FUSION_MODEL_PATH ($(stat -c %y $DNPS_FUSION_MODEL_PATH))"
echo "null:   $DNPS_NULL_MODEL_PATH"
echo "species: ${SPECIES_LIST[*]}  (each with human PLM, shared human fusion head)"
echo

for sp in "${SPECIES_LIST[@]}"; do
  echo "================================================"
  echo "[$sp] DNPS_SPECIES=$sp  DNPS_PLM_SPECIES=$DNPS_PLM_SPECIES"
  echo "================================================"
  DNPS_SPECIES="$sp" python dnps_hybrid/inference.py auto \
    --dataset NINE_SPECIES_DATASET --use-plm true
  echo "[$sp] done"
  echo
done

echo "=== All species done ==="
