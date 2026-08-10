#!/bin/bash
#SBATCH --job-name=mut_infer_human_iso
#SBATCH --output=data/slurm/%j_mut_infer_human_iso.out
#SBATCH --chdir=/data/nasif12/home_if12/khsam/dnps_hybrid
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga05,ouga06,ouga07,ouga08,ouga09,ouga12,ouga26
#
# Rerun ONLY the pepLM (use_plm=true / fusion) arm of MUTATIONS_DATASET
# with the human_iso pepLM. DNPS arm reused (Apr 28 mztab on disk).
# Output namespaced so the April canonical mutations_hybrid.mztab is
# preserved.

source /data/nasif12/home_if12/khsam/.bashrc
set -e
set -x

REPO_ROOT=/data/nasif12/home_if12/khsam/dnps_hybrid
source "$(dirname "$0")/scripts/dnps_data_path.sh"
RUN_PATH=/s/project/denovo-prosit/SamKhan/dnps_hybrid/casanovo
MGF_GLOB="$RUN_PATH/mutations_proforma/*.mgf"
RESULTS_DIR=$RUN_PATH/results
OUT_BASENAME=mutations_hybrid_human_iso

export DNPS_PLM_SPECIES=human_iso
export DNPS_FUSION_MODEL_PATH="${DNPS_FUSION_MODEL_PATH:-$DNPS_DATA_PATH/models/human_iso_asymbnln/fusion_model.pth}"
export DNPS_NULL_MODEL_PATH="${DNPS_NULL_MODEL_PATH:-$DNPS_DATA_PATH/models/human_iso_asymbnln/null_model.pth}"
export DNPS_PLM_CKPT_PATH="${DNPS_PLM_CKPT_PATH:-$DNPS_DATA_PATH/models/human_iso/plm_ckpt.pt}"

# Sanity
[ -s "$RESULTS_DIR/mutations_dnps.mztab" ] || { echo "[fatal] DNPS arm mztab missing"; exit 2; }
ls $MGF_GLOB >/dev/null || { echo "[fatal] no mutations MGFs at $MGF_GLOB"; exit 2; }

conda activate khsam
mkdir -p "$RESULTS_DIR"

casanovo sequence \
    -m /s/project/denovo-prosit/CASANOVO/casanovo_5.0.0_model_weights/casanovo_v5_0_0.ckpt \
    -c $REPO_ROOT/casanovo/casanovo/config.yaml \
    -d "$RESULTS_DIR" \
    -o "$OUT_BASENAME" \
    --teacher_forcing false \
    --use_plm true \
    -f $MGF_GLOB

ls -lh "$RESULTS_DIR/${OUT_BASENAME}.mztab"
echo "=== Done ==="
