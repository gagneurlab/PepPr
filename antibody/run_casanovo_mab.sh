#!/bin/bash
#SBATCH --job-name=casanovo_mab
#SBATCH --output=data/slurm/%A_%a_casanovo_mab.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Casanovo v5 de novo sequencing on a mAb benchmark, one dataset x one arm.
#
# Usage:
#   sbatch antibody/run_casanovo_mab.sh <dataset> <arm>
#
#   dataset: beslic_H | beslic_L | herceptin | trastuzumab | nontryp
#            (nontryp resolves the (mAb, protease) run from SLURM_ARRAY_TASK_ID,
#             so submit it as an array:  sbatch --array=0-13%5 ... nontryp <arm>)
#   arm:     noplm      -- vanilla Casanovo v5, no pepLM
#            human_iso  -- human_iso pepLM + LN/LN MassiveKB fusion head
#            germline   -- germline_{species}_clean pepLM + asymbnln fusion head
#                          (species auto-selected per mAb: human for IgG1_Human*
#                           / Herceptin / Trastuzumab, mouse otherwise)
#
# Cys-alkylation mod normalisation is per-dataset (IAM +57.021, IAA +58.005, or
# already-native), so the ground-truth SEQ= vocab matches Casanovo's.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

set -euo pipefail
cd "$REPO_ROOT"

DATASET="${1:?usage: run_casanovo_mab.sh <beslic_H|beslic_L|herceptin|trastuzumab|nontryp> <noplm|human_iso|germline>}"
ARM="${2:?arm required: noplm | human_iso | germline}"

DATA=$DNPS_DATA_PATH

# ── Resolve dataset → SRC_MGF, RES_DIR, TAG, MAB_ID, cys-normalisation mass ──
# NORM_CYS empty  -> ground-truth SEQ= already uses Casanovo's vocab (SRC used as-is).
# NORM_CYS +57.02 / +58.00 -> rewrite that Cys mass to C[Carbamidomethyl] (+ the
# shared M-ox / N,Q-deamidation / W-ox / N-term pyro-Glu rewrites) into a v5norm MGF.
case "$DATASET" in
  beslic_H|beslic_L)
    if [[ "$DATASET" == beslic_H ]]; then MAB_ID=IgG1_Human_H; SAMPLE=Heavy-Chain-Trypsin-1
    else                                  MAB_ID=IgG1_Human_L; SAMPLE=Light-Chain-Trypsin-1; fi
    DIR="$DATA/beslic_mab/$MAB_ID"
    SRC_MGF="$DIR/${SAMPLE}_annotated.mgf"
    RES_DIR="$DIR/casanovo_results"
    TAG="$MAB_ID"
    NORM_CYS="+57.021"
    ;;
  herceptin)
    MAB_ID=Herceptin
    DIR="$DATA/PXD023419"
    SRC_MGF="$DIR/Peng2021_Herceptin_tryp_annotated.mgf"
    RES_DIR="$DIR/casanovo_results"
    TAG="herceptin"
    NORM_CYS="+58.005"   # Peng 2021 used iodoacetic acid (Carboxymethyl, +58)
    ;;
  trastuzumab)
    MAB_ID=Trastuzumab
    DIR="$DATA/PXD057525"
    SRC_MGF="$DIR/PXD057525_trastuzumab_annotated.mgf"
    RES_DIR="$DIR/casanovo_results"
    TAG="pxd057525"
    NORM_CYS=""          # chloroacetamide -> Carbamidomethyl (+57.021), native vocab
    ;;
  nontryp)
    IDX="${SLURM_ARRAY_TASK_ID:?nontryp needs an --array task id (RUNS index)}"
    read MAB_ID PROTEASE SRC_MGF RES_DIR < <("$PYTHON_BIN" - <<PYEOF
from antibody.nontryp_registry import RUNS
r = RUNS[$IDX]
print(r.mab_id, r.protease, r.annotated_mgf, r.res_dir)
PYEOF
)
    TAG="${MAB_ID}_${PROTEASE}"
    NORM_CYS=""          # registry annotated MGFs are already v5-vocab
    ;;
  *)
    echo "unknown dataset '$DATASET'" >&2; exit 2 ;;
esac

[[ -f "$SRC_MGF" ]] || { echo "[fatal] missing source MGF $SRC_MGF" >&2; exit 1; }
mkdir -p "$RES_DIR"

# ── Normalise ground-truth mods to Casanovo's vocab (if required) ───────────
if [[ -n "$NORM_CYS" ]]; then
  EVAL_MGF="$RES_DIR/$(basename "${SRC_MGF%.mgf}")_v5norm.mgf"
  NORM_CYS="$NORM_CYS" SRC_MGF="$SRC_MGF" EVAL_MGF="$EVAL_MGF" "$PYTHON_BIN" - <<'PYEOF'
import os, re
src, out, cys = os.environ["SRC_MGF"], os.environ["EVAL_MGF"], os.environ["NORM_CYS"]
with open(src) as f, open(out, "w") as o:
    for line in f:
        if line.startswith("SEQ="):
            s = line[4:].rstrip()
            s = re.sub(r'^Q\[\-17\.027\]', '[Ammonia-loss]-Q', s)   # N-term pyro-Glu
            s = s.replace(f'C[{cys}]', 'C[Carbamidomethyl]')
            s = s.replace('M[+15.995]', 'M[Oxidation]')
            s = s.replace('N[+0.984]', 'N[Deamidated]')
            s = s.replace('Q[+0.984]', 'Q[Deamidated]')
            s = s.replace('W[+15.995]', 'W')                        # Casanovo can't predict W-ox
            o.write(f"SEQ={s}\n")
        else:
            o.write(line)
PYEOF
else
  EVAL_MGF="$SRC_MGF"
fi

# ── Per-job Casanovo config with a task-unique lance_dir (avoids Lance commit
#    conflicts between parallel runs; see memory: feedback_casanovo_lance_dir) ──
CKPT=${CASANOVO_CKPT:-$DATA/models/casanovo_v5_0_0.ckpt}
CONFIG_TPL="$REPO_ROOT/casanovo/casanovo/config.yaml"
JOB_TAG="${TAG}_${SLURM_JOB_ID:-$$}"
CONFIG="$RES_DIR/casanovo_config_${ARM}_${JOB_TAG}.yaml"
sed "s|^lance_dir:.*|lance_dir: $RES_DIR/lance_${ARM}_${JOB_TAG}|" "$CONFIG_TPL" > "$CONFIG"

# ── Resolve arm → pepLM env + output tag ───────────────────────────────────
USE_PLM=false
case "$ARM" in
  noplm)
    OUT_TAG="casanovo_${TAG}_noplm_${JOB_TAG}"
    ;;
  human_iso)
    USE_PLM=true
    export DNPS_PLM_CKPT_PATH="$DATA/human_iso/plm_ckpt.pt"
    export DNPS_FUSION_MODEL_PATH="$DATA/casanovo/fusion_model.pth"
    export DNPS_NULL_MODEL_PATH="$DATA/casanovo/null_model.pth"
    OUT_TAG="casanovo_${TAG}_humaniso_plm_${JOB_TAG}"
    ;;
  germline)
    USE_PLM=true
    case "$MAB_ID" in
      IgG1_Human*|Herceptin|Trastuzumab) SP=human ;;
      *)                                 SP=mouse ;;
    esac
    export DNPS_PLM_DISTINGUISH_IL=0
    export DNPS_FUSION_OUTPUT_IL=0
    export DNPS_PLM_CKPT_PATH="$DATA/germline_${SP}_clean/plm_ckpt.pt"
    export DNPS_FUSION_MODEL_PATH="$DATA/casanovo/fusion_model_asymbnln.pth"
    export DNPS_NULL_MODEL_PATH="$DATA/casanovo/null_model_asymbnln.pth"
    OUT_TAG="casanovo_${TAG}_germline_${SP}_sw_clean_${JOB_TAG}"
    ;;
  *)
    echo "unknown arm '$ARM'" >&2; exit 2 ;;
esac

echo "=== Pre-flight ==="
echo "Host:     $(hostname)"
echo "Start:    $(date -Is)"
echo "dataset:  $DATASET  (MAB=$MAB_ID  TAG=$TAG)"
echo "arm:      $ARM  (use_plm=$USE_PLM)"
echo "ckpt:     $CKPT"
echo "eval MGF: $EVAL_MGF ($(grep -c '^BEGIN IONS' "$EVAL_MGF") spectra)"
[[ "$USE_PLM" == true ]] && echo "pepLM:    ${DNPS_PLM_CKPT_PATH}"
echo

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
command -v "$CASANOVO_BIN"
"$CASANOVO_BIN" --version 2>&1 | head -3 || true

cd "$RES_DIR"
base=$(basename "$EVAL_MGF" .mgf)
rm -rf "${base}__lance.lance" "${base}__lance" "lance_${ARM}_${JOB_TAG}" 2>/dev/null || true

"$CASANOVO_BIN" sequence \
  -m "$CKPT" -c "$CONFIG" \
  -d "$RES_DIR" \
  -o "$OUT_TAG" \
  --teacher_forcing false --use_plm "$USE_PLM" -f \
  "$EVAL_MGF" 2>&1 | tee "$RES_DIR/${OUT_TAG}.log"

echo
echo "=== Done ==="
ls -lh "$RES_DIR/${OUT_TAG}".mztab 2>/dev/null
echo "End: $(date -Is)"
