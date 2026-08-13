#!/bin/bash
#SBATCH --job-name=dnps_kol_eval
#SBATCH --output=data/slurm/%j_kol_eval.out
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Casanovo vs Casanovo + PepPr (pepLM) on the archived, fixed 10k-spectrum
# Kingdoms-of-Life subsets. Runs eval_kingdoms_pp.py per species, tolerating
# per-species failures (logs and continues). The full KoL raw / intermediate
# MGF trees are NOT needed — only the archived <species>.mgf subsets.
#
# Usage:
#   sbatch run_kol_eval.sh                       # default: human
#   sbatch run_kol_eval.sh mouse dog             # explicit species
#
# Reproduce the human_iso asymbnln 15-species generalization sweep:
#   DNPS_FUSION_MODEL_PATH=$DNPS_DATA_PATH/models/human_iso_asymbnln/fusion_model.pth \
#   DNPS_NULL_MODEL_PATH=$DNPS_DATA_PATH/models/human_iso_asymbnln/null_model.pth \
#   PLM_SPECIES=human_iso \
#   KOL_OUT_DIR=$DNPS_DATA_PATH/kingdoms_of_life_eval_asymbnln \
#   sbatch run_kol_eval.sh human mouse dog bos_tauros canaerohabidis_elegans \
#     cricetulus_griseus danio_rerio didelphis_didelphinae drosophila_melanogaster \
#     gallus_gallus oryctolagus_cuniculus oryzias_melastigma rattus_norvegicus \
#     sus_scrofa tardigrade
#
# Optional env:
#   PLM_SPECIES      pepLM weights for the +PP arm            (default: human)
#   MODES            passed to --modes (e.g. casanovo_pp)     (default: all)
#   REGEN=1          overwrite existing Casanovo results (--no-skip-existing)
#   KOL_SUBSETS_DIR  archived fixed subsets                   (default:
#                    $DNPS_DATA_PATH/benchmarks/kingdoms_of_life/subsets)
#   KOL_OUT_DIR      results root passed to --out-dir         (default: eval_kingdoms_pp.py default)
#   DNPS_FUSION_MODEL_PATH / DNPS_NULL_MODEL_PATH  fusion head (default: const)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

export TMUX="${TMUX:-noninteractive}"
set -u  # NOT -e: we want per-species fault tolerance.
cd "$REPO_ROOT"

PLM_SPECIES=${PLM_SPECIES:-human}
KOL_SUBSETS_DIR=${KOL_SUBSETS_DIR:-"$DNPS_DATA_PATH/benchmarks/kingdoms_of_life/subsets"}
REGEN=${REGEN:-0}

SPECIES_LIST=("$@")
if [ "${#SPECIES_LIST[@]}" -eq 0 ]; then
  SPECIES_LIST=(human)
fi

if [ ! -d "$KOL_SUBSETS_DIR" ]; then
  echo "[fatal] missing archived KoL subsets: $KOL_SUBSETS_DIR" >&2
  exit 2
fi

# Optional fusion-head override (e.g. the human_iso asymbnln head).
if [ -n "${DNPS_FUSION_MODEL_PATH:-}" ]; then
  if [ ! -s "$DNPS_FUSION_MODEL_PATH" ] || [ ! -s "${DNPS_NULL_MODEL_PATH:-}" ]; then
    echo "[fatal] DNPS_FUSION_MODEL_PATH set but head(s) missing" >&2
    exit 2
  fi
  export DNPS_FUSION_MODEL_PATH DNPS_NULL_MODEL_PATH
fi

EXTRA_FLAGS=()
[ "$REGEN" = "1" ] && EXTRA_FLAGS+=(--no-skip-existing)
[ -n "${MODES:-}" ] && EXTRA_FLAGS+=(--modes ${MODES})
[ -n "${KOL_OUT_DIR:-}" ] && EXTRA_FLAGS+=(--out-dir "$KOL_OUT_DIR")

JOB_LOG_DIR=data/slurm/${SLURM_JOB_ID:-local}_kol
mkdir -p "$JOB_LOG_DIR"

echo "=================================================================="
echo "KoL eval"
echo "  species:       ${SPECIES_LIST[*]}"
echo "  plm-species:   $PLM_SPECIES"
echo "  subsets:       $KOL_SUBSETS_DIR/<species>.mgf"
echo "  regen:         $REGEN"
echo "  out-dir:       ${KOL_OUT_DIR:-<eval_kingdoms_pp default>}"
echo "  fusion head:   ${DNPS_FUSION_MODEL_PATH:-<const default>}"
echo "  per-species logs: $JOB_LOG_DIR/<species>.log"
echo "=================================================================="

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi

declare -a FAILED=()
for sp in "${SPECIES_LIST[@]}"; do
  echo
  echo "----- [$sp] -----"
  subset="$KOL_SUBSETS_DIR/$sp.mgf"
  if [ ! -f "$subset" ]; then
    echo "[$sp] missing archived subset: $subset (continuing)"
    FAILED+=("$sp:missing-subset"); continue
  fi
  logfile="$JOB_LOG_DIR/$sp.log"
  (
    set -o pipefail
    "$PYTHON_BIN" "$SCRIPT_DIR/eval_kingdoms_pp.py" \
      --species "$sp" \
      --plm-species "$PLM_SPECIES" \
      --mgf-root "$KOL_SUBSETS_DIR" \
      "${EXTRA_FLAGS[@]}" 2>&1 | tee "$logfile"
  )
  if [ $? -ne 0 ]; then
    echo "[$sp] FAILED (see $logfile)"
    FAILED+=("$sp:eval")
  else
    echo "[$sp] OK"
  fi
done

echo
echo "=================================================================="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "KoL eval DONE with failures: ${FAILED[*]}"
  exit 1
fi
echo "KoL eval DONE — all species succeeded."
echo "=================================================================="
