#!/bin/bash
#SBATCH --job-name=casanovo_herceptin
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_casanovo_herceptin.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Vanilla Casanovo v5 (no pepLM) on the PXD023419 Herceptin tryptic MGF,
# which carries SEQ= ground truths from a FragPipe (MSFragger+MSBooster+
# Percolator+Philosopher) 1% FDR pass. Then compute peptide/AA accuracy.
#
# Notes on the mod mismatch:
#   The Peng 2021 sample protocol uses iodoacetic acid (Carboxymethyl C, +58),
#   not iodoacetamide (Carbamidomethyl C, +57). Casanovo's vocabulary only
#   knows C[Carbamidomethyl] (+57). We rewrite C[+58.005] -> C[Carbamidomethyl]
#   in a normalized eval-MGF below so Casanovo's predictions and the ground
#   truth use the same vocab. The 1-Da Cys mass mismatch on the spectra is
#   unavoidable without re-deriving SEQ= from a +57-assumption search.

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
cd /data/nasif12/home_if12/khsam/dnps_hybrid

CKPT=/s/project/denovo-prosit/CASANOVO/casanovo_5.0.0_model_weights/casanovo_v5_0_0.ckpt
CONFIG=/data/nasif12/home_if12/khsam/dnps_hybrid/casanovo/casanovo/config.yaml
SRC_MGF=/s/project/denovo-prosit/SamKhan/dnps_hybrid/PXD023419/Peng2021_Herceptin_tryp_annotated.mgf
RES_DIR=/s/project/denovo-prosit/SamKhan/dnps_hybrid/PXD023419/casanovo_results
EVAL_MGF=$RES_DIR/Peng2021_Herceptin_tryp_annotated_v5norm.mgf

mkdir -p "$RES_DIR"

echo "=== Pre-flight ==="
echo "Host:   $(hostname)"
echo "Start:  $(date -Is)"
echo "ckpt:   $CKPT ($(stat -c %y $CKPT))"
echo "config: $CONFIG"
echo "input:  $SRC_MGF ($(grep -c '^BEGIN IONS' $SRC_MGF) spectra)"

# Rewrite C[+58.005] -> C[Carbamidomethyl] in a copy; pass that to casanovo.
# Also strip W[+15.995] (Casanovo can't predict W oxidation) and Q[-17.027]
# at peptide position 1 -> [Ammonia-loss]- N-term token.
python - <<PYEOF
import re
src = "$SRC_MGF"
out = "$EVAL_MGF"
with open(src) as f, open(out, "w") as o:
    for line in f:
        if line.startswith("SEQ="):
            s = line[4:].rstrip()
            # Pyro-Glu Q at residue 1 -> [Ammonia-loss]-Q
            s = re.sub(r'^Q\[\-17\.027\]', '[Ammonia-loss]-Q', s)
            # Standard mods
            s = s.replace('C[+58.005]', 'C[Carbamidomethyl]')
            s = s.replace('M[+15.995]', 'M[Oxidation]')
            s = s.replace('N[+0.984]', 'N[Deamidated]')
            s = s.replace('Q[+0.984]', 'Q[Deamidated]')
            # W oxidation: Casanovo can't predict; drop the annotation so the
            # peptide stays evaluable (we'll exclude W-ox PSMs from accuracy
            # since the residue mass will be off by 16 Da).
            s = s.replace('W[+15.995]', 'W')
            o.write(f"SEQ={s}\n")
        else:
            o.write(line)
PYEOF
echo "eval:   $EVAL_MGF"
grep -m3 '^SEQ=' "$EVAL_MGF"
echo

conda activate khsam
which casanovo
casanovo --version 2>&1 | head -3 || true

echo "=== Running Casanovo v5 (no PLM) ==="
cd "$RES_DIR"
# Drop any prior lance dir so depthcharge rebuilds cleanly.
base=$(basename "$EVAL_MGF" .mgf)
rm -rf "${base}__lance.lance" "${base}__lance" 2>/dev/null || true

casanovo sequence \
  -m "$CKPT" \
  -c "$CONFIG" \
  -d "$RES_DIR" \
  -o "casanovo_herceptin_${SLURM_JOB_ID}" \
  --teacher_forcing false \
  --use_plm false \
  -f \
  "$EVAL_MGF" 2>&1 | tee "$RES_DIR/casanovo_herceptin_${SLURM_JOB_ID}.log"

echo
echo "=== Accuracy ==="
# Pass the same mztab twice (the script expects two arms; we only have one)
python /data/nasif12/home_if12/khsam/dnps_hybrid/scripts/compute_accuracy_casanovo.py \
  --mgf "$EVAL_MGF" \
  --mztab-fusion "$RES_DIR/casanovo_herceptin_${SLURM_JOB_ID}.mztab" \
  --mztab-dnps   "$RES_DIR/casanovo_herceptin_${SLURM_JOB_ID}.mztab" \
  2>&1 | tee "$RES_DIR/accuracy_${SLURM_JOB_ID}.txt"

echo
echo "=== Done ==="
echo "End: $(date -Is)"
ls -lh "$RES_DIR/casanovo_herceptin_${SLURM_JOB_ID}".{mztab,log} 2>/dev/null
