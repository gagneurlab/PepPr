#!/bin/bash
#SBATCH --job-name=casanovo_mab_arm2hi
#SBATCH --output=data/slurm/%j_casanovo_%x.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26

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
# Arm 2 only: human_iso pepLM (human_iso/plm_ckpt.pt). Output named
# `casanovo_<MAB>_humaniso_plm_<jobid>.mztab` to keep it distinct from the
# other pepLM arms' mztabs.
# Usage:  sbatch --job-name=casanovo_arm2hi_$MAB run_casanovo_beslic_mab_arm2_human_iso.sh <MAB_ID>

set -euo pipefail
cd "$REPO_ROOT"

MAB="${1:?mab id required}"
BESLIC=$DNPS_DATA_PATH/beslic_mab
DIR="$BESLIC/$MAB"

case "$MAB" in
  IgG1_Human_H)  SAMPLE="Heavy-Chain-Trypsin-1" ;;
  IgG1_Human_L)  SAMPLE="Light-Chain-Trypsin-1" ;;
  *) echo "unknown mab '$MAB'" >&2; exit 2 ;;
esac

RES_DIR="$DIR/casanovo_results"
EVAL_MGF="$RES_DIR/${SAMPLE}_v5norm.mgf"
if [[ ! -f "$EVAL_MGF" ]]; then
  echo "missing eval MGF $EVAL_MGF -- run the full two-arm script first" >&2
  exit 1
fi

CKPT=${CASANOVO_CKPT:-$DNPS_DATA_PATH/models/casanovo_v5_0_0.ckpt}
CONFIG_TPL=$REPO_ROOT/casanovo/casanovo/config.yaml
JOB_TAG="${MAB}_${SLURM_JOB_ID:-$$}"
CONFIG="$RES_DIR/casanovo_config_humaniso_${JOB_TAG}.yaml"
sed "s|^lance_dir:.*|lance_dir: $RES_DIR/lance_humaniso_${JOB_TAG}|" "$CONFIG_TPL" > "$CONFIG"
grep -E "^(lance_dir|n_beams|top_match):" "$CONFIG"

export DNPS_PLM_CKPT_PATH=$DNPS_DATA_PATH/human_iso/plm_ckpt.pt
export DNPS_FUSION_MODEL_PATH=$DNPS_DATA_PATH/casanovo/fusion_model.pth
export DNPS_NULL_MODEL_PATH=$DNPS_DATA_PATH/casanovo/null_model.pth

echo "=== Pre-flight ==="
echo "Host:      $(hostname)"
echo "Start:     $(date -Is)"
echo "MAB:       $MAB"
echo "plm ckpt:  $DNPS_PLM_CKPT_PATH ($(stat -c %y $DNPS_PLM_CKPT_PATH))"
echo "eval MGF:  $EVAL_MGF ($(grep -c '^BEGIN IONS' $EVAL_MGF) spectra)"

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
command -v "$CASANOVO_BIN"
cd "$RES_DIR"
base=$(basename "$EVAL_MGF" .mgf)
rm -rf "${base}__lance.lance" "${base}__lance" "lance_humaniso_${JOB_TAG}" 2>/dev/null || true

OUT_TAG="casanovo_${MAB}_humaniso_plm_${JOB_TAG}"
"$CASANOVO_BIN" sequence \
  -m "$CKPT" -c "$CONFIG" \
  -d "$RES_DIR" \
  -o "$OUT_TAG" \
  --teacher_forcing false --use_plm true -f \
  "$EVAL_MGF" 2>&1 | tee "$RES_DIR/${OUT_TAG}.log"

echo
echo "=== Done ==="
ls -lh "$RES_DIR/${OUT_TAG}".mztab 2>/dev/null
echo "End: $(date -Is)"
