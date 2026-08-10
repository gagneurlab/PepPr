#!/bin/bash
#SBATCH --job-name=fragpipe_mab
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_fragpipe_%x.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=noninterruptive
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# FragPipe 23.0 headless: MSFragger -> MSBooster -> Percolator -> Philosopher -> 1% FDR psm.tsv
# Usage:  sbatch --job-name=fragpipe_$MAB run_fragpipe_beslic_mab.sh <MAB_ID>
# MAB_ID one of: IgG1_Human_H IgG1_Human_L

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail

MAB="${1:?mab id required}"
BESLIC=/s/project/denovo-prosit/SamKhan/dnps_hybrid/beslic_mab
DIR="$BESLIC/$MAB"
WORKDIR="$DIR/fragpipe_workdir"
WORKFLOW="$WORKDIR/fragpipe.workflow"
MANIFEST="$WORKDIR/fragpipe-files.fp-manifest"

FRAGPIPE=/data/nasif12/home_if12/khsam/local/fragpipe-23.0/bin/fragpipe
export PATH=/opt/modules/i12g/anaconda/envs/java11/bin:$PATH

echo "=== Pre-flight ==="
echo "Host:     $(hostname)"
echo "Start:    $(date -Is)"
echo "MAB:      $MAB"
echo "workdir:  $WORKDIR"
echo "manifest: $(cat $MANIFEST)"
echo "java:     $(java -version 2>&1 | h

ead -1)"
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
  head -1 "$PSM" | tr '\t' '\n' | head -20
else
  echo "No psm.tsv found!"
  exit 1
fi
echo
echo "End: $(date -Is)"
