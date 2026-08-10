#!/bin/bash
#SBATCH --job-name=assemble_pc_mouse
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_assemble_pc_mouse.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=noninterruptive
#SBATCH --exclude=ouga03,ouga08,ouga09,ouga05,ouga06,ouga26

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
cd /data/nasif12/home_if12/khsam/dnps_hybrid

conda activate khsam

MOUSE_MABS="2B4 36H6 85F7"

echo "=== Assembly (mouse mAbs) ==="
echo "Host:  $(hostname)"
echo "Start: $(date -Is)"

python -m dnps_hybrid.assembly --mabs $MOUSE_MABS

echo ""
echo "=== PC curves (mouse mAbs) ==="
python xa_novo/plot_pc_3arm_mabs.py \
    --mabs $MOUSE_MABS \
    --out xa_novo/pc_3arm_mouse_mabs.png

echo "=== Done ==="
echo "End: $(date -Is)"
ls -lh xa_novo/alps_2arm_all/2arm_summary.tsv xa_novo/pc_3arm_mouse_mabs.png
