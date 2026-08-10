#!/bin/bash
#SBATCH --job-name=infer_instanovo
#SBATCH --output=data/slurm/%j_infer_instanovo.out
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=urgent
#SBATCH --gres=gpu:1
#
# InstaNovo baseline (transformer only, no diffusion refinement) on a nine-species
# subset. Vanilla arm only (no PepPr fusion). Paths resolve from dnps_hybrid.const.
#
# Usage: sbatch infer_instanovo.sh <species>     # e.g. human, mouse, ...

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
set -x

SP="${1:?usage: infer_instanovo.sh <species>}"
conda activate ka-instanovo

MGF_DIR=$(python -c "from dnps_hybrid.const import nine_species_benchmark_dir as d; print(d('$SP'))")
OUT_DIR=$(python -c "from dnps_hybrid.const import SPECIES,result_run_path as r; print(r(SPECIES['$SP']['run_name']))")/instanovo
mkdir -p "$OUT_DIR"

for mgf in "$MGF_DIR"/*.mgf; do
    base=$(basename "$mgf" .mgf)
    out="$OUT_DIR/${base}.csv"
    if [ -s "$out" ]; then
        echo "Skipping $base (already done)"
        continue
    fi
    echo "=== InstaNovo on $base ==="
    instanovo transformer predict \
        --data-path "$mgf" \
        --output-path "$out" \
        --instanovo-model instanovo-v1.1.0 \
        --denovo \
        'index_columns=[precursor_mz,precursor_charge,spectrum_id]'
done

ls -lh "$OUT_DIR"
echo "=== Done ==="
