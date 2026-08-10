#!/bin/bash
#SBATCH --job-name=casanovo_mab
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
# Vanilla Casanovo v5 (no pepLM) on a beslic_mab mAb. The PLM arm lives in
# run_casanovo_beslic_mab_arm2_multi.sh (multi-protease pepLM).
# Usage:  sbatch --job-name=casanovo_$MAB run_casanovo_beslic_mab.sh <MAB_ID>
#
# Cys mods (MSV / Tran 2016 IgG1-Human): IAM = Carbamidomethyl +57.021 ->
# Casanovo native vocab; no SEQ rewrite needed.

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

SRC_MGF="$DIR/${SAMPLE}_annotated.mgf"
RES_DIR="$DIR/casanovo_results"
EVAL_MGF="$RES_DIR/${SAMPLE}_v5norm.mgf"
mkdir -p "$RES_DIR"

CKPT=${CASANOVO_CKPT:-$DNPS_DATA_PATH/models/casanovo_v5_0_0.ckpt}
CONFIG_TPL=$REPO_ROOT/casanovo/casanovo/config.yaml
# Per-job config with a task-unique lance_dir so parallel casanovo runs don't
# collide on a shared Lance dataset (see memory: feedback_casanovo_lance_dir).
JOB_TAG="${MAB}_${SLURM_JOB_ID:-$$}"
CONFIG="$RES_DIR/casanovo_config_${JOB_TAG}.yaml"
mkdir -p "$RES_DIR"
sed "s|^lance_dir:.*|lance_dir: $RES_DIR/lance_${JOB_TAG}|" "$CONFIG_TPL" > "$CONFIG"
grep "^lance_dir:" "$CONFIG"

echo "=== Pre-flight ==="
echo "Host:      $(hostname)"
echo "Start:     $(date -Is)"
echo "MAB:       $MAB"
echo "ckpt:      $CKPT"
echo "input:     $SRC_MGF ($(grep -c '^BEGIN IONS' $SRC_MGF) spectra)"
echo "eval out:  $EVAL_MGF"
echo

# Normalize SEQ= mods to Casanovo's vocab.
# All mAbs: also Pyro-Glu Q at res-1 -> [Ammonia-loss]-Q; M[+15.995] -> M[Oxidation]; N/Q[+0.984] -> Deamidated;
# strip W[+15.995] since Casanovo can't predict W oxidation.
"$PYTHON_BIN" - <<PYEOF
import re
src = "$SRC_MGF"; out = "$EVAL_MGF"
with open(src) as f, open(out, "w") as o:
    for line in f:
        if line.startswith("SEQ="):
            s = line[4:].rstrip()
            s = re.sub(r'^Q\[\-17\.027\]', '[Ammonia-loss]-Q', s)
            s = s.replace('C[+57.021]', 'C[Carbamidomethyl]')
            s = s.replace('M[+15.995]', 'M[Oxidation]')
            s = s.replace('N[+0.984]', 'N[Deamidated]')
            s = s.replace('Q[+0.984]', 'Q[Deamidated]')
            s = s.replace('W[+15.995]', 'W')
            o.write(f"SEQ={s}\n")
        else:
            o.write(line)
PYEOF

grep -m3 '^SEQ=' "$EVAL_MGF" || true
echo

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
command -v "$CASANOVO_BIN"
"$CASANOVO_BIN" --version 2>&1 | head -3 || true

JOBID="${SLURM_JOB_ID:-$$}"

# ---- Arm 1: no pepLM ----
echo "=== Arm 1: no pepLM ==="
cd "$RES_DIR"
base=$(basename "$EVAL_MGF" .mgf)
rm -rf "${base}__lance.lance" "${base}__lance" 2>/dev/null || true
"$CASANOVO_BIN" sequence \
  -m "$CKPT" -c "$CONFIG" \
  -d "$RES_DIR" \
  -o "casanovo_${MAB}_noplm_${JOBID}" \
  --teacher_forcing false --use_plm false -f \
  "$EVAL_MGF" 2>&1 | tee "$RES_DIR/casanovo_${MAB}_noplm_${JOBID}.log"

echo
echo "=== Done ==="
ls -lh "$RES_DIR"/casanovo_${MAB}_*_${JOBID}.mztab 2>/dev/null
echo "End: $(date -Is)"
