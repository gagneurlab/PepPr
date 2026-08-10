#!/bin/bash
#SBATCH --job-name=dnps_kol_pp
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

# Casanovo vs Casanovo + pepLM (PP) on the archived, fixed 10k-spectrum
# Kingdoms-of-Life subsets. No full KoL raw or intermediate MGF tree is needed.
# Runs casanovo -e twice per species
# (--use_plm false / true; PP from --plm-species / DNPS_PLM_SPECIES, default human),
# beams=5 from casanovo/config.yaml.
#
# Usage:
#   sbatch run_eval_kingdoms_pp.sh
#       → default --species human only (see eval_kingdoms_pp.py).
#   sbatch run_eval_kingdoms_pp.sh mouse yeast
#       → --species mouse yeast
#   sbatch --export=ALL,REGEN=1 run_eval_kingdoms_pp.sh mouse
#       → rerun on the same archived subset (overwrite mztab/log).
#
# Optional env:
#   MODES=...       — passed to --modes (whitespace-separated), e.g. MODES=casanovo_pp
#   PLM_SPECIES=x   — passed to --plm-species (default in Python is human).
#   REGEN=1         — add --no-skip-existing.
#   KOL_SUBSETS_DIR — archived subset directory (default:
#                     $DNPS_DATA_PATH/benchmarks/kingdoms_of_life/subsets).
#

export TMUX="${TMUX:-noninteractive}"

set -e
set -x

cd "$REPO_ROOT"

KOL_SUBSETS_DIR=${KOL_SUBSETS_DIR:-"$DNPS_DATA_PATH/benchmarks/kingdoms_of_life/subsets"}
if [ ! -d "$KOL_SUBSETS_DIR" ]; then
    echo "missing archived KoL subsets: $KOL_SUBSETS_DIR" >&2
    exit 2
fi

# Re-run Casanovo on the fixed subsets when requested.
REGEN_FLAG=()
if [ "${REGEN:-0}" = "1" ]; then
    REGEN_FLAG=(--no-skip-existing)
fi

MODES_FLAG=()
if [ -n "${MODES:-}" ]; then
    # shellcheck disable=SC2206
    MODES_FLAG=(--modes ${MODES})
fi

PLM_FLAG=()
if [ -n "${PLM_SPECIES:-}" ]; then
    PLM_FLAG=(--plm-species "${PLM_SPECIES}")
fi

SPECIES_ARGS=("$@")
if [ "${#SPECIES_ARGS[@]}" -eq 0 ]; then
    SPECIES_ARGS=(human)
fi

for sp in "${SPECIES_ARGS[@]}"; do
    subset="$KOL_SUBSETS_DIR/$sp.mgf"
    if [ ! -f "$subset" ]; then
        echo "missing archived KoL subset: $subset" >&2
        exit 2
    fi
done

"$PYTHON_BIN" eval_kingdoms_pp.py --mgf-root "$KOL_SUBSETS_DIR" \
    "${REGEN_FLAG[@]}" "${MODES_FLAG[@]}" "${PLM_FLAG[@]}" \
    --species "${SPECIES_ARGS[@]}"
