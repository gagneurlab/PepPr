#!/bin/bash
#SBATCH --job-name=casanovo_pxd057525
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_casanovo_pxd057525.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Casanovo v5 on PXD057525 Trastuzumab tryptic (Q Exactive HF, DDA HCD).
# Cys alkylation is chloroacetamide -> Carbamidomethyl C (+57.021), so the
# truth SEQs match Casanovo's vocab natively (no vocab-normalize step).
# Vanilla Casanovo v5 (no pepLM). The PLM arm lives in the matching
# run_casanovo_pxd057525_trastuzumab_arm2*.sh scripts.

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
cd /data/nasif12/home_if12/khsam/dnps_hybrid

CKPT=/s/project/denovo-prosit/CASANOVO/casanovo_5.0.0_model_weights/casanovo_v5_0_0.ckpt
CONFIG=/data/nasif12/home_if12/khsam/dnps_hybrid/casanovo/casanovo/config.yaml
MGF=/s/project/denovo-prosit/SamKhan/dnps_hybrid/PXD057525/PXD057525_trastuzumab_annotated.mgf
RES=/s/project/denovo-prosit/SamKhan/dnps_hybrid/PXD057525/casanovo_results
mkdir -p "$RES"

echo "=== Pre-flight ==="
echo "ckpt:   $CKPT"
echo "input:  $MGF ($(grep -c '^BEGIN IONS' $MGF) spectra)"
echo

conda activate khsam

# --- Arm 1: no PLM ---
cd "$RES"
base=$(basename "$MGF" .mgf)
rm -rf "${base}__lance.lance" "${base}__lance" 2>/dev/null || true
echo "=== Arm 1: no pepLM ==="
casanovo sequence \
  -m "$CKPT" -c "$CONFIG" \
  -d "$RES" \
  -o "casanovo_pxd057525_noplm_${SLURM_JOB_ID}" \
  --teacher_forcing false --use_plm false -f \
  "$MGF" 2>&1 | tee "$RES/casanovo_pxd057525_noplm_${SLURM_JOB_ID}.log"

echo
echo "=== Done ==="
ls -lh "$RES"/casanovo_pxd057525_*${SLURM_JOB_ID}.mztab
