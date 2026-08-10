#!/bin/bash
#SBATCH --job-name=casanovo_nt_vanilla
#SBATCH --output=data/slurm/%A_%a_casanovo_nt_vanilla.out
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
# Vanilla Casanovo v5 (no pepLM) inference on non-tryptic mAb runs.
# Array idx -> RUNS[idx] in nontryp_registry.
#
# Submit:
#   sbatch --array=<idx-spec>%5 run_casanovo_nontryp_array.sh

set -eo pipefail
cd "$REPO_ROOT"

IDX="${SLURM_ARRAY_TASK_ID:?array task id required}"

read TAG EVAL_MGF RES_DIR < <("$PYTHON_BIN" - <<PYEOF
from nontryp_registry import RUNS
r = RUNS[$IDX]
print(f"{r.mab_id}_{r.protease}", r.annotated_mgf, r.res_dir)
PYEOF
)

[[ -f "$EVAL_MGF" ]] || { echo "missing annotated MGF $EVAL_MGF" >&2; exit 1; }
mkdir -p "$RES_DIR"

CKPT=${CASANOVO_CKPT:-$DNPS_DATA_PATH/models/casanovo_v5_0_0.ckpt}
CONFIG_TPL=$REPO_ROOT/casanovo/casanovo/config.yaml
JOB_TAG="${TAG}_${SLURM_JOB_ID:-$$}"
CONFIG="$RES_DIR/casanovo_config_noplm_${JOB_TAG}.yaml"
sed "s|^lance_dir:.*|lance_dir: $RES_DIR/lance_noplm_${JOB_TAG}|" "$CONFIG_TPL" > "$CONFIG"

echo "=== Pre-flight ==="
echo "Host:      $(hostname)"
echo "Start:     $(date -Is)"
echo "Array idx: $IDX  (TAG=$TAG)"
echo "eval MGF:  $EVAL_MGF ($(grep -c '^BEGIN IONS' $EVAL_MGF) spectra)"

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
cd "$RES_DIR"
base=$(basename "$EVAL_MGF" .mgf)
rm -rf "${base}__lance.lance" "${base}__lance" "lance_noplm_${JOB_TAG}" 2>/dev/null || true

OUT_TAG="casanovo_${TAG}_noplm_${JOB_TAG}"
"$CASANOVO_BIN" sequence \
  -m "$CKPT" -c "$CONFIG" \
  -d "$RES_DIR" \
  -o "$OUT_TAG" \
  --teacher_forcing false --use_plm false -f \
  "$EVAL_MGF" 2>&1 | tee "$RES_DIR/${OUT_TAG}.log"

echo "=== Done ==="
ls -lh "$RES_DIR/${OUT_TAG}".mztab 2>/dev/null
echo "End: $(date -Is)"
