#!/bin/bash
#SBATCH --job-name=dnps_9s
#SBATCH --output=data/slurm/%j.out
#SBATCH --time=120:00:00
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

ALL_SPECIES="cowpea archaeon endoloripes"
# ALL_SPECIES="mouse honeybee tomato"


set -e
set -x

if [ "${1:-}" = "all" ]; then
    # Nine-species MGFs are included loose in the extracted Zenodo archive.
    : "${DNPS_NINE_SPECIES_PATH:=${DNPS_DATA_PATH}/external/nine_species}"
    export DNPS_NINE_SPECIES_PATH
    for species in $ALL_SPECIES; do
        export DNPS_PLM_SPECIES=${2:-$species}
        echo "=== Running species: $species (PLM from: $DNPS_PLM_SPECIES) ==="
        DNPS_SPECIES=$species "$PYTHON_BIN" dnps_hybrid/inference.py auto
    done
else
    export DNPS_SPECIES=${1:-human}
    export DNPS_PLM_SPECIES=${2:-$DNPS_SPECIES}
    echo "=== Running pipeline for species: $DNPS_SPECIES (PLM from: $DNPS_PLM_SPECIES) ==="
    # python dnps_hybrid/inference.py auto
    "$PYTHON_BIN" run_proteometools_saav.py
fi
