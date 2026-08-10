#!/bin/bash
#SBATCH --job-name=assemble_pc_human
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_assemble_pc_human.out
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

HUMAN_MABS="S2P6 IgG1_Human_H IgG1_Human_L"

echo "=== Assembly (human mAbs) ==="
echo "Host:  $(hostname)"
echo "Start: $(date -Is)"

python -m dnps_hybrid.assembly --mabs $HUMAN_MABS

echo ""
echo "=== PC curves (human mAbs) ==="
python xa_novo/plot_pc_3arm_mabs.py \
    --mabs $HUMAN_MABS \
    --out xa_novo/pc_3arm_human_mabs.png

echo "=== Done ==="
echo "End: $(date -Is)"
ls -lh xa_novo/alps_2arm_all/2arm_summary.tsv xa_novo/pc_3arm_human_mabs.png
