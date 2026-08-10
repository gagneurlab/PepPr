#!/bin/bash
#SBATCH --job-name=dnps_kol_pipeline
#SBATCH --output=data/slurm/%j_kol_pipeline.out
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
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

# Evaluate the archived, fixed 10k-spectrum Kingdoms-of-Life subsets for each
# species passed as a positional argument, sequentially. The full KoL raw and
# intermediate MGF trees are not part of the maintained paper workflow.
#
# Usage:
#   sbatch run_kol_pipeline.sh <species> [<species> ...]
#
# Examples:
#   sbatch run_kol_pipeline.sh dog
#   sbatch run_kol_pipeline.sh mouse dog
#
# The script does NOT abort the whole job if one species fails — it logs the
# failure and moves on. Per-species stdout+stderr are tee'd to
# data/slurm/$SLURM_JOB_ID/<species>/eval.log.
#
# Optional env-var overrides (set via `sbatch --export=ALL,VAR=val ...`):
#   PLM_SPECIES      pepLM weights for inference                (default: human)
#   KOL_SUBSETS_DIR   archived fixed subsets                     (default:
#                    $DNPS_DATA_PATH/benchmarks/kingdoms_of_life/subsets)
#   REGEN            if =1, overwrite existing Casanovo results

set -u  # fail on unset vars; we explicitly do NOT set -e (we want per-species
        # fault tolerance — see retval handling below).


cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
    echo "ERROR: pass one or more species as positional args. Known species:" >&2
    "$PYTHON_BIN" -c "from dnps_hybrid.const import KOL_SPECIES_DIRS; print(' '.join(sorted(KOL_SPECIES_DIRS)))" >&2
    exit 2
fi

PLM_SPECIES=${PLM_SPECIES:-human}
KOL_SUBSETS_DIR=${KOL_SUBSETS_DIR:-"$DNPS_DATA_PATH/benchmarks/kingdoms_of_life/subsets"}
REGEN=${REGEN:-0}

REGEN_FLAG=()
if [ "$REGEN" = "1" ]; then
    REGEN_FLAG=(--no-skip-existing)
fi

JOB_LOG_DIR=data/slurm/${SLURM_JOB_ID:-local}_kol
mkdir -p "$JOB_LOG_DIR"

echo "=================================================================="
echo "KoL pipeline run"
echo "  species:         $*"
echo "  plm-species:     $PLM_SPECIES"
echo "  fixed subsets:   $KOL_SUBSETS_DIR/<species>.mgf"
echo "  regen:           $REGEN"
echo "  per-species logs: $JOB_LOG_DIR/<species>/eval.log"
echo "=================================================================="

run_stage() {
    # run_stage <species> <stage_name> <cmd...>
    local sp=$1; shift
    local stage=$1; shift
    local logdir="$JOB_LOG_DIR/$sp"
    mkdir -p "$logdir"
    local logfile="$logdir/${stage}.log"
    echo
    echo "----- [$sp] $stage -----"
    echo "$ $*" | tee -a "$logfile"
    # Use `set -o pipefail` for this command so we get the python exit code
    # rather than tee's success.
    ( set -o pipefail; "$@" 2>&1 | tee -a "$logfile" )
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$sp] $stage FAILED (exit $rc) — see $logfile"
    else
        echo "[$sp] $stage OK"
    fi
    return $rc
}

OVERALL_RC=0
declare -a FAILED_SPECIES=()

for sp in "$@"; do
    echo
    echo "=================================================================="
    echo "===  SPECIES: $sp"
    echo "=================================================================="

    subset_path="$KOL_SUBSETS_DIR/$sp.mgf"
    if [ ! -f "$subset_path" ]; then
        echo "[$sp] missing archived subset: $subset_path"
        FAILED_SPECIES+=("$sp:missing-subset"); OVERALL_RC=1; continue
    fi

    if ! run_stage "$sp" "eval" \
        "$PYTHON_BIN" eval_kingdoms_pp.py \
            --species "$sp" \
            --plm-species "$PLM_SPECIES" \
            --mgf-root "$KOL_SUBSETS_DIR" \
            "${REGEN_FLAG[@]}"
    then
        FAILED_SPECIES+=("$sp:eval"); OVERALL_RC=1; continue
    fi

    echo "[$sp] all stages OK"
done

echo
echo "=================================================================="
echo "KoL pipeline DONE"
if [ ${#FAILED_SPECIES[@]} -gt 0 ]; then
    echo "Failures: ${FAILED_SPECIES[*]}"
else
    echo "All species succeeded."
fi
echo "=================================================================="
exit $OVERALL_RC
