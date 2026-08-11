#!/bin/bash
#SBATCH --job-name=fragpipe_mab
#SBATCH --output=data/slurm/%A_%a_fragpipe_mab.out
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=noninterruptive
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# FragPipe 23.0 headless (MSFragger -> MSBooster -> Percolator -> Philosopher)
# to a 1% FDR psm.tsv, for a mAb benchmark run.
#
# Usage:
#   sbatch antibody/run_fragpipe_mab.sh beslic <IgG1_Human_H|IgG1_Human_L>
#   sbatch --array=0-13%5 antibody/run_fragpipe_mab.sh nontryp
#     (nontryp resolves RUNS[SLURM_ARRAY_TASK_ID] from the non-tryptic registry)
#
# Tool locations are overridable via env (FRAGPIPE, FRAGPIPE_JAVA_BIN).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

set -euo pipefail
cd "$REPO_ROOT"

MODE="${1:?usage: run_fragpipe_mab.sh <beslic|nontryp> [MAB_ID]}"

case "$MODE" in
  beslic)
    MAB="${2:?mab id required (IgG1_Human_H | IgG1_Human_L)}"
    WORKDIR="$DNPS_DATA_PATH/beslic_mab/$MAB/fragpipe_workdir"
    LABEL="$MAB"
    ;;
  nontryp)
    IDX="${SLURM_ARRAY_TASK_ID:?nontryp needs an --array task id (RUNS index)}"
    # Stagger array startup so concurrent FragPipe instances don't race on the
    # shared ~/.config/FragPipe workflows cache during GUI-tab init.
    STAGGER=30
    echo "Staggering startup by $((IDX * STAGGER))s (anti-race for FragPipe config dir)"
    sleep $((IDX * STAGGER))
    read MAB PROTEASE WORKDIR < <("$PYTHON_BIN" - <<PYEOF
from antibody.nontryp_registry import RUNS
r = RUNS[$IDX]
print(r.mab_id, r.protease, r.workdir)
PYEOF
)
    LABEL="${MAB}_${PROTEASE}"
    ;;
  *)
    echo "unknown mode '$MODE' (expected beslic | nontryp)" >&2; exit 2 ;;
esac

WORKFLOW="$WORKDIR/fragpipe.workflow"
MANIFEST="$WORKDIR/fragpipe-files.fp-manifest"

FRAGPIPE="${FRAGPIPE:-$HOME/local/fragpipe-23.0/bin/fragpipe}"
export PATH="${FRAGPIPE_JAVA_BIN:-/opt/modules/i12g/anaconda/envs/java11/bin}:$PATH"

echo "=== Pre-flight ==="
echo "Host:     $(hostname)"
echo "Start:    $(date -Is)"
echo "run:      $LABEL"
echo "workdir:  $WORKDIR"
echo "manifest: $(cat "$MANIFEST")"
echo "java:     $(java -version 2>&1 | head -1)"
echo

cd "$WORKDIR"
"$FRAGPIPE" --headless \
  --workflow "$WORKFLOW" \
  --manifest "$MANIFEST" \
  --workdir "$WORKDIR" \
  --ram 24 --threads 8 \
  2>&1 | tee "$WORKDIR/fragpipe_run_${SLURM_JOB_ID:-$$}.log"

echo
echo "=== psm.tsv summary ==="
PSM=$(find "$WORKDIR" -maxdepth 2 -name psm.tsv | head -1)
if [[ -n "$PSM" ]]; then
  echo "PSM file: $PSM"
  wc -l "$PSM"
else
  echo "No psm.tsv found!" >&2
  exit 1
fi
echo "End: $(date -Is)"
