#!/bin/bash
#SBATCH --job-name=fragpipe_nt
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%A_%a_fragpipe_nt.out
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=noninterruptive
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
# Submit with:
#   sbatch --array=0-13%5 run_fragpipe_nontryp_array.sh
# Per-task index → entry in nontryp_registry.RUNS; cap at 5 concurrent jobs.

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail

cd /data/nasif12/home_if12/khsam/dnps_hybrid

IDX="${SLURM_ARRAY_TASK_ID:?array task id required}"

# Stagger startup to avoid concurrent FragPipe instances racing on the shared
# ~/.config/FragPipe/fragpipe/workflows/ cache during GUI tab init.
# Within each %5-concurrent slot the offset is bounded by 5*STAGGER seconds.
STAGGER=30
echo "Staggering startup by $((IDX * STAGGER)) seconds (anti-race for FragPipe config dir)"
sleep $((IDX * STAGGER))

# Resolve registry entry → workdir + workflow + manifest paths.
read MAB PROTEASE WORKDIR < <(python - <<PYEOF
from nontryp_registry import RUNS
r = RUNS[$IDX]
print(r.mab_id, r.protease, r.workdir)
PYEOF
)

WORKFLOW="$WORKDIR/fragpipe.workflow"
MANIFEST="$WORKDIR/fragpipe-files.fp-manifest"

FRAGPIPE=/data/nasif12/home_if12/khsam/local/fragpipe-23.0/bin/fragpipe
export PATH=/opt/modules/i12g/anaconda/envs/java11/bin:$PATH

echo "=== Pre-flight ==="
echo "Host:     $(hostname)"
echo "Start:    $(date -Is)"
echo "Array idx: $IDX  (MAB=$MAB  PROTEASE=$PROTEASE)"
echo "workdir:  $WORKDIR"
echo "manifest: $(cat $MANIFEST)"
echo "java:     $(java -version 2>&1 | head -1)"
echo

cd "$WORKDIR"
"$FRAGPIPE" --headless \
  --workflow "$WORKFLOW" \
  --manifest "$MANIFEST" \
  --workdir "$WORKDIR" \
  --ram 24 --threads 8 \
  2>&1 | tee "$WORKDIR/fragpipe_run_${SLURM_JOB_ID}.log"

echo
echo "=== psm.tsv summary ==="
PSM=$(find "$WORKDIR" -maxdepth 2 -name psm.tsv | head -1)
if [[ -n "$PSM" ]]; then
  echo "PSM file: $PSM"
  wc -l "$PSM"
else
  echo "No psm.tsv found!"
  exit 1
fi
echo
echo "End: $(date -Is)"
