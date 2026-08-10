#!/bin/bash
#SBATCH --job-name=9s_infer_cross_yb
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_9s_inference_crossspecies_yeast_bacillus.out
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Cross-species PLM (human) + human-trained fusion head for yeast and bacillus.
# Companion to rerun_9s_inference_crossspecies_archaeon_endoloripes.slurm; the
# old _plmhuman_hybrid_18933956 / _18933957 files predate the swap50 fusion
# head and would mix fusion-head generations in plot_scatter_precision.py.
#
# Yeast first (smaller, ~9h previously) so bacillus (~1.5x larger) gets the
# remaining wall time.
#
# Isolation: per-job DNPS_LANCE_DIR avoids the Lance commit conflicts that
# crashed 19088271/19088272 when sharing the casanovo default lance_dir with
# concurrent inference jobs.

export TMUX="${TMUX:-noninteractive}"
source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
cd /data/nasif12/home_if12/khsam/dnps_hybrid

DATA_ROOT="${DATA_PATH:-/s/project/denovo-prosit/SamKhan/dnps_hybrid}"
export DNPS_FUSION_MODEL_PATH="$DATA_ROOT/casanovo/fusion_model.pth"
export DNPS_NULL_MODEL_PATH="$DATA_ROOT/casanovo/null_model.pth"
export DNPS_PLM_SPECIES=human
export DNPS_LANCE_DIR="${TMPDIR:-/tmp}/casanovo_lance_${SLURM_JOB_ID}"

if [[ ! -s "$DNPS_FUSION_MODEL_PATH" ]]; then
  echo "[fatal] missing $DNPS_FUSION_MODEL_PATH" >&2
  exit 2
fi

SPECIES_LIST=( ${DNPS_SPECIES_LIST:-yeast bacillus} )

echo "=== Pre-flight ==="
echo "fusion:    $DNPS_FUSION_MODEL_PATH ($(stat -c %y $DNPS_FUSION_MODEL_PATH))"
echo "null:      $DNPS_NULL_MODEL_PATH"
echo "lance_dir: $DNPS_LANCE_DIR  (per-job, isolated)"
echo "species:   ${SPECIES_LIST[*]}  (each with human PLM, shared human fusion head)"
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
