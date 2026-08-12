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
#   sbatch run_baseline.sh <instanovo|contranovo|powernovo|smsnet> <species>
#
#   instanovo   -- InstaNovo v1.1.0 transformer (no diffusion refinement),
#                  vanilla arm only. Conda env: $INSTANOVO_ENV (default ka-instanovo).
#   contranovo  -- ContraNovo +/- pepLM (PepPr) with the asymbnln + swap50
#                  ContraNovo fusion head (pepLM/species-independent).
#   powernovo   -- PowerNovo (transformer + BERT rescoring), vanilla arm.
#
# Species names and result run-names are defined in peptide_priors.const.SPECIES.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

set -euo pipefail
cd "$REPO_ROOT"

TOOL="${1:?usage: run_baseline.sh <instanovo|contranovo|powernovo|smsnet> <species>}"
SP="${2:?species required (e.g. human, mouse, yeast, ...)}"

activate_env() {  # activate_env <conda-env-name>
  if declare -F conda >/dev/null; then conda activate "$1"; fi
}

case "$TOOL" in
  instanovo)
    activate_env "${INSTANOVO_ENV:-ka-instanovo}"
    MGF_DIR=$("$PYTHON_BIN" -c "from peptide_priors.const import nine_species_benchmark_dir as d; print(d('$SP'))")
    OUT_DIR=$("$PYTHON_BIN" -c "from peptide_priors.const import SPECIES, result_run_path as r; print(r(SPECIES['$SP']['run_name']))")/instanovo
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
    SHARED_DIR=$DNPS_DATA_PATH/models/casanovo
    export DNPS_SPECIES=$SP
    export DNPS_CONTRANOVO_FUSION_MODEL_PATH=$SHARED_DIR/contranovo_fusion_model.pth
    export DNPS_CONTRANOVO_NULL_MODEL_PATH=$SHARED_DIR/contranovo_null_model.pth
    ls -lh "$DNPS_CONTRANOVO_FUSION_MODEL_PATH" "$DNPS_CONTRANOVO_NULL_MODEL_PATH"
    activate_env "${DNPS_CONDA_ENV:-khsam}"
    echo "=== ContraNovo +/- pepLM for species: $SP (asymbnln+swap50 fusion head) ==="
    "$PYTHON_BIN" peptide_priors/inference.py contranovo
    ;;

  powernovo)
    activate_env "${DNPS_CONDA_ENV:-khsam}"
    echo "=== PowerNovo for species: $SP ==="
    "$PYTHON_BIN" "$SCRIPT_DIR/run_benchmark.py" "$SP"
    ;;

  smsnet)
    # SMSNet is an external TensorFlow tool (not pip-installable like the others,
    # and not a submodule). Run it from its own repository, then place its
    # per-species output where plot_benchmark reads it (const.SMSNET_ROOT):
    #   ${DNPS_SMSNET_ROOT:-<archive>/results/baselines/smsnet}/<species>_inputs_output
    root=$("$PYTHON_BIN" -c "from peptide_priors.const import SMSNET_ROOT; print(SMSNET_ROOT)")
    echo "SMSNet is an external tool and is not driven from this repo." >&2
    echo "Run it from its own repository, then place its output at:" >&2
    echo "  $root/${SP}_inputs_output" >&2
    echo "plot_benchmark.py reads it from there (override with DNPS_SMSNET_ROOT)." >&2
    exit 3 ;;

  *)
    echo "unknown tool '$TOOL' (expected instanovo | contranovo | powernovo | smsnet)" >&2
    exit 2 ;;
esac

echo "=== Done ==="
