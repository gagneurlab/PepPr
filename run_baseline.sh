#!/bin/bash
#SBATCH --job-name=dnps_baseline
#SBATCH --output=data/slurm/%j_baseline.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# De novo baseline sequencing on a nine-species dataset. One driver for the
# three external baselines compared against Casanovo+PepPr in the benchmark.
#
# Usage:
#   sbatch run_baseline.sh <instanovo|contranovo|powernovo> <species>
#
#   instanovo   -- InstaNovo v1.1.0 transformer (no diffusion refinement),
#                  vanilla arm only. Conda env: $INSTANOVO_ENV (default ka-instanovo).
#   contranovo  -- ContraNovo +/- pepLM (PepPr) with the asymbnln + swap50
#                  ContraNovo fusion head (pepLM/species-independent).
#   powernovo   -- PowerNovo (transformer + BERT rescoring), vanilla arm.
#
# Species names and result run-names are defined in dnps_hybrid.const.SPECIES.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

set -euo pipefail
cd "$REPO_ROOT"

TOOL="${1:?usage: run_baseline.sh <instanovo|contranovo|powernovo> <species>}"
SP="${2:?species required (e.g. human, mouse, yeast, ...)}"

activate_env() {  # activate_env <conda-env-name>
  if declare -F conda >/dev/null; then conda activate "$1"; fi
}

case "$TOOL" in
  instanovo)
    activate_env "${INSTANOVO_ENV:-ka-instanovo}"
    MGF_DIR=$("$PYTHON_BIN" -c "from dnps_hybrid.const import nine_species_benchmark_dir as d; print(d('$SP'))")
    OUT_DIR=$("$PYTHON_BIN" -c "from dnps_hybrid.const import SPECIES, result_run_path as r; print(r(SPECIES['$SP']['run_name']))")/instanovo
    mkdir -p "$OUT_DIR"
    for mgf in "$MGF_DIR"/*.mgf; do
      base=$(basename "$mgf" .mgf)
      out="$OUT_DIR/${base}.csv"
      if [ -s "$out" ]; then echo "Skipping $base (already done)"; continue; fi
      echo "=== InstaNovo on $base ==="
      instanovo transformer predict \
        --data-path "$mgf" \
        --output-path "$out" \
        --instanovo-model instanovo-v1.1.0 \
        --denovo \
        'index_columns=[precursor_mz,precursor_charge,spectrum_id]'
    done
    ls -lh "$OUT_DIR"
    ;;

  contranovo)
    SHARED_DIR=$DNPS_DATA_PATH/casanovo
    export DNPS_SPECIES=$SP
    export DNPS_CONTRANOVO_FUSION_MODEL_PATH=$SHARED_DIR/contranovo_fusion_asymbnln_swap50.pth
    export DNPS_CONTRANOVO_NULL_MODEL_PATH=$SHARED_DIR/contranovo_null_asymbnln_swap50.pth
    ls -lh "$DNPS_CONTRANOVO_FUSION_MODEL_PATH" "$DNPS_CONTRANOVO_NULL_MODEL_PATH"
    activate_env "${DNPS_CONDA_ENV:-khsam}"
    echo "=== ContraNovo +/- pepLM for species: $SP (asymbnln+swap50 fusion head) ==="
    "$PYTHON_BIN" dnps_hybrid/inference.py contranovo
    ;;

  powernovo)
    activate_env "${DNPS_CONDA_ENV:-khsam}"
    echo "=== PowerNovo for species: $SP ==="
    "$PYTHON_BIN" run_benchmark.py "$SP"
    ;;

  *)
    echo "unknown tool '$TOOL' (expected instanovo | contranovo | powernovo)" >&2
    exit 2 ;;
esac

echo "=== Done ==="
