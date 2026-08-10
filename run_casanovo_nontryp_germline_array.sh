#!/bin/bash
#SBATCH --job-name=casanovo_nt_germline
#SBATCH --output=data/slurm/%A_%a_casanovo_nt_germline.out
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
# Casanovo + germline_{species}_clean pepLM (asymbnln fusion head) on
# non-tryptic mAb runs. Species is auto-selected per mAb: IgG1_Human_* and
# Herceptin -> human; WIgG1_* and anti-FLAG-M2 -> mouse.
#
# Submit:
#   sbatch --array=<idx-spec>%5 run_casanovo_nontryp_germline_array.sh

set -eo pipefail
cd "$REPO_ROOT"

IDX="${SLURM_ARRAY_TASK_ID:?array task id required}"

read TAG EVAL_MGF RES_DIR SPECIES < <("$PYTHON_BIN" - <<PYEOF
from nontryp_registry import RUNS
r = RUNS[$IDX]
# Species mapping: human for IgG1_Human_H/L and Herceptin, mouse otherwise.
sp = "human" if r.mab_id.startswith("IgG1_Human") or r.mab_id == "Herceptin" else "mouse"
print(f"{r.mab_id}_{r.protease}", r.annotated_mgf, r.res_dir, sp)
PYEOF
)

[[ -f "$EVAL_MGF" ]] || { echo "missing annotated MGF $EVAL_MGF" >&2; exit 1; }
mkdir -p "$RES_DIR"

CKPT=${CASANOVO_CKPT:-$DNPS_DATA_PATH/models/casanovo_v5_0_0.ckpt}
CONFIG_TPL=$REPO_ROOT/casanovo/casanovo/config.yaml
JOB_TAG="${TAG}_${SLURM_JOB_ID:-$$}"
CONFIG="$RES_DIR/casanovo_config_germline_${SPECIES}_sw_clean_${JOB_TAG}.yaml"
sed "s|^lance_dir:.*|lance_dir: $RES_DIR/lance_germline_${SPECIES}_sw_clean_${JOB_TAG}|" \
    "$CONFIG_TPL" > "$CONFIG"

export DNPS_PLM_DISTINGUISH_IL=0
export DNPS_FUSION_OUTPUT_IL=0
export DNPS_PLM_CKPT_PATH=$DNPS_DATA_PATH/germline_${SPECIES}_clean/plm_ckpt.pt
export DNPS_FUSION_MODEL_PATH=$DNPS_DATA_PATH/casanovo/fusion_model_asymbnln.pth
export DNPS_NULL_MODEL_PATH=$DNPS_DATA_PATH/casanovo/null_model_asymbnln.pth

echo "=== Pre-flight ==="
echo "Host:      $(hostname)"
echo "Start:     $(date -Is)"
echo "Array idx: $IDX  (TAG=$TAG, species=$SPECIES)"
echo "pepLM:     $DNPS_PLM_CKPT_PATH"
echo "eval MGF:  $EVAL_MGF ($(grep -c '^BEGIN IONS' $EVAL_MGF) spectra)"

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
cd "$RES_DIR"
base=$(basename "$EVAL_MGF" .mgf)
rm -rf "${base}__lance.lance" "${base}__lance" "lance_germline_${SPECIES}_sw_clean_${JOB_TAG}" 2>/dev/null || true

OUT_TAG="casanovo_${TAG}_germline_${SPECIES}_sw_clean_${JOB_TAG}"
"$CASANOVO_BIN" sequence \
  -m "$CKPT" -c "$CONFIG" \
  -d "$RES_DIR" \
  -o "$OUT_TAG" \
  --teacher_forcing false --use_plm true -f \
  "$EVAL_MGF" 2>&1 | tee "$RES_DIR/${OUT_TAG}.log"

echo "=== Done ==="
ls -lh "$RES_DIR/${OUT_TAG}".mztab 2>/dev/null
echo "End: $(date -Is)"
