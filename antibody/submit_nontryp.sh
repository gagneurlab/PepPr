#!/bin/bash
# Submit the full non-tryptic pipeline: FragPipe array -> annotate MGF -> Casanovo array.
# Caps concurrency at 5 within each array (--array=0-13%5).
#
# Usage:
#   bash submit_nontryp.sh                    # all 14 entries
#   bash submit_nontryp.sh 0 9                # only entries 0 and 9
#   bash submit_nontryp.sh --skip-fragpipe    # re-run only Casanovo (FragPipe already done)
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data/slurm

SKIP_FRAGPIPE=0
IDX_LIST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-fragpipe) SKIP_FRAGPIPE=1; shift ;;
    --) shift; IDX_LIST="$*"; break ;;
    *) IDX_LIST="$IDX_LIST $1"; shift ;;
  esac
done

# Build the --array range from IDX_LIST (or default 0-13)
if [[ -z "${IDX_LIST// }" ]]; then
  ARRAY_SPEC="0-13%5"
else
  # join with comma, then append %5
  ARR=$(echo "$IDX_LIST" | tr -s ' ' | sed 's/^ //; s/ /,/g')
  ARRAY_SPEC="${ARR}%5"
fi
echo "Using --array=$ARRAY_SPEC"

DEP=""
if [[ "$SKIP_FRAGPIPE" -eq 0 ]]; then
  FRAGPIPE_JID=$(sbatch --parsable --array="$ARRAY_SPEC" run_fragpipe_nontryp_array.sh)
  echo "FragPipe array job: $FRAGPIPE_JID"

  BUILD_JID=$(sbatch --parsable --dependency=afterok:$FRAGPIPE_JID build_annot_nontryp.slurm)
  echo "Build-annotated MGF job: $BUILD_JID (depends on $FRAGPIPE_JID)"

  DEP="--dependency=afterok:$BUILD_JID"
fi

CASANOVO_JID=$(sbatch --parsable $DEP --array="$ARRAY_SPEC" run_casanovo_nontryp_array.sh)
echo "Casanovo array job: $CASANOVO_JID${DEP:+ ($DEP)}"

echo
echo "Submitted. Monitor with:"
echo "  squeue -u $USER -o '%.12i %.30j %.2t %.10M %R'"
