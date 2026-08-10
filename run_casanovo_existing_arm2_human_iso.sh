#!/bin/bash
#SBATCH --job-name=casanovo_arm2hi_existing
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_casanovo_%x.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Arm 2 with the human_iso pepLM for the two pre-existing mAb runs
# (Herceptin/PXD023419 and Trastuzumab/PXD057525). Arm 1 (no pepLM) mztab is reused.
#
# Usage:  sbatch run_casanovo_existing_arm2_human_iso.sh <herceptin|trastuzumab>

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
cd /data/nasif12/home_if12/khsam/dnps_hybrid

WHICH="${1:?'herceptin' or 'trastuzumab'}"
case "$WHICH" in
  herceptin)
    RES_DIR=/s/project/denovo-prosit/SamKhan/dnps_hybrid/PXD023419/casanovo_results
    EVAL_MGF=$RES_DIR/Peng2021_Herceptin_tryp_annotated_v5norm.mgf
    OUT_TAG_BASE=casanovo_herceptin_humaniso_plm
    ;;
  trastuzumab)
    RES_DIR=/s/project/denovo-prosit/SamKhan/dnps_hybrid/PXD057525/casanovo_results
    EVAL_MGF=$RES_DIR/PXD057525_trastuzumab_annotated.mgf
    if [[ ! -f "$EVAL_MGF" ]]; then
      ln /s/project/denovo-prosit/SamKhan/dnps_hybrid/PXD057525/PXD057525_trastuzumab_annotated.mgf "$EVAL_MGF"
    fi
    OUT_TAG_BASE=casanovo_pxd057525_humaniso_plm
    ;;
  *) echo "unknown: $WHICH"; exit 2 ;;
esac

CKPT=/s/project/denovo-prosit/CASANOVO/casanovo_5.0.0_model_weights/casanovo_v5_0_0.ckpt
CONFIG_TPL=/data/nasif12/home_if12/khsam/dnps_hybrid/casanovo/casanovo/config.yaml
JOB_TAG="${WHICH}_${SLURM_JOB_ID:-$$}"
CONFIG="$RES_DIR/casanovo_config_humaniso_${JOB_TAG}.yaml"
sed "s|^lance_dir:.*|lance_dir: $RES_DIR/lance_humaniso_${JOB_TAG}|" "$CONFIG_TPL" > "$CONFIG"
grep -E "^(lance_dir|n_beams|top_match):" "$CONFIG"

export DNPS_PLM_CKPT_PATH=/s/project/denovo-prosit/SamKhan/dnps_hybrid/human_iso/plm_ckpt.pt
export DNPS_FUSION_MODEL_PATH=/s/project/denovo-prosit/SamKhan/dnps_hybrid/casanovo/fusion_model.pth
export DNPS_NULL_MODEL_PATH=/s/project/denovo-prosit/SamKhan/dnps_hybrid/casanovo/null_model.pth

echo "=== Pre-flight ==="
echo "plm ckpt: $DNPS_PLM_CKPT_PATH ($(stat -c %y $DNPS_PLM_CKPT_PATH))"
echo "eval:     $EVAL_MGF ($(grep -c '^BEGIN IONS' $EVAL_MGF) spectra)"

conda activate khsam
cd "$RES_DIR"
base=$(basename "$EVAL_MGF" .mgf)
rm -rf "${base}__lance.lance" "${base}__lance" "lance_humaniso_${JOB_TAG}" 2>/dev/null || true

OUT_TAG="${OUT_TAG_BASE}_${SLURM_JOB_ID:-$$}"
casanovo sequence \
  -m "$CKPT" -c "$CONFIG" \
  -d "$RES_DIR" \
  -o "$OUT_TAG" \
  --teacher_forcing false --use_plm true -f \
  "$EVAL_MGF" 2>&1 | tee "$RES_DIR/${OUT_TAG}.log"

echo
echo "=== Done ==="
ls -lh "$RES_DIR/${OUT_TAG}".mztab 2>/dev/null
