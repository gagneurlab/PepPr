#!/bin/bash
#SBATCH --job-name=kol_humaniso_asymbnln
#SBATCH --output=data/slurm/%j_kol_humaniso_asymbnln.out
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi
#
# KoL eval for the human_iso pepLM using its matching asymbnln MassiveKB
# fusion head and the archived fixed 10k subsets. Existing Casanovo-only and
# mouse-prior results are reused; full KoL raw/MGF trees are not required.

export TMUX="${TMUX:-noninteractive}"
set -euo pipefail
cd "$REPO_ROOT"

DATA_ROOT="$DNPS_DATA_PATH"
if [[ ! -s "$DNPS_FUSION_MODEL_PATH" || ! -s "$DNPS_NULL_MODEL_PATH" ]]; then
  echo "[fatal] missing asymbnln head(s)" >&2
  exit 2
fi

KOL_ROOT=${KOL_ROOT:-"/s/project/denovo-prosit/SamKhan/dnps_hybrid/kingdoms_of_life_eval_asymbnln"}
OUT_DIR=${KOL_OUT_DIR:-"$KOL_ROOT"}
KOL_SUBSETS_DIR=${KOL_SUBSETS_DIR:-"$KOL_ROOT/subsets"}
export DNPS_FUSION_MODEL_PATH="${DNPS_FUSION_MODEL_PATH:-/s/project/denovo-prosit/SamKhan/dnps_hybrid/human_iso_asymbnln/fusion_model.pth}"
export DNPS_NULL_MODEL_PATH="${DNPS_NULL_MODEL_PATH:-/s/project/denovo-prosit/SamKhan/dnps_hybrid/human_iso_asymbnln/null_model.pth}"
mkdir -p "$OUT_DIR/runs"

if [[ ! -d "$KOL_SUBSETS_DIR" ]]; then
  echo "[fatal] missing archived KoL subsets: $KOL_SUBSETS_DIR" >&2
  exit 2
fi

# 15-species set from rerun_kol_mskb_swap50_eval.slurm.
SPECIES_LIST=(
  human mouse dog
  bos_tauros canaerohabidis_elegans cricetulus_griseus danio_rerio
  didelphis_didelphinae drosophila_melanogaster gallus_gallus
  oryctolagus_cuniculus oryzias_melastigma rattus_norvegicus sus_scrofa
  tardigrade
)

echo "=== Pre-flight ==="
echo "fusion head: $DNPS_FUSION_MODEL_PATH ($(stat -c %y $DNPS_FUSION_MODEL_PATH))"
echo "null head:   $DNPS_NULL_MODEL_PATH"
echo "out-dir:     $OUT_DIR"
echo "subsets:     $KOL_SUBSETS_DIR/<species>.mgf"
echo "species:     ${SPECIES_LIST[*]}"
echo

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi

for PLM in human_iso; do
  echo "================================================"
  echo "[asymbnln] $PLM pepLM eval"
  echo "================================================"
  for sp in "${SPECIES_LIST[@]}"; do
    echo "----- $sp / PLM=$PLM -----"
    if [[ ! -f "$KOL_SUBSETS_DIR/$sp.mgf" ]]; then
      echo "[$sp/$PLM] missing archived subset (continuing)"
      continue
    fi
    "$PYTHON_BIN" eval_kingdoms_pp.py \
      --species "$sp" \
      --plm-species "$PLM" \
      --mgf-root "$KOL_SUBSETS_DIR" \
      --out-dir "$OUT_DIR" \
      || echo "[$sp/$PLM] FAILED (continuing)"
  done
done

echo
echo "=== Done ==="
echo "Regenerate the plot vs LN/LN with:"
echo "  python plot_kol_overlap_vs_pp_gain.py --runs-dir $OUT_DIR/runs --out kol_overlap_vs_pp_gain_asymbnln.png"
