#!/bin/bash
#SBATCH --job-name=prepare_yeast_proforma
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_prepare_yeast_proforma.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=noninterruptive

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail

REPO_ROOT=/data/nasif12/home_if12/khsam/dnps_hybrid
SOURCE_DIR=/s/project/denovo-prosit/data/benchmark_data/nine_species_benchmark_noble/nine-species-main/Saccharomyces-cerevisiae
OUTPUT_DIR=/s/project/denovo-prosit/SamKhan/dnps_hybrid/saccharomyces_cerevisiae/nine_species_proforma

conda activate khsam
cd "$REPO_ROOT"

SOURCE_DIR="$SOURCE_DIR" OUTPUT_DIR="$OUTPUT_DIR" python - <<'PY'
import os
from pathlib import Path

from dnps_hybrid.prepare_data import convert_mgf_to_proforma

source_dir = Path(os.environ["SOURCE_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
source_files = sorted(source_dir.glob("*.mgf"))
if not source_files:
    raise FileNotFoundError(f"No source MGFs found in {source_dir}")

output_dir.mkdir(parents=True, exist_ok=True)
for path in output_dir.glob("*.mgf"):
    if path.is_symlink():
        path.unlink()

convert_mgf_to_proforma(
    str(source_dir / "*.mgf"),
    str(output_dir),
    validate_proforma=True,
)

outputs = sorted(output_dir.glob("*.mgf"))
if len(outputs) != len(source_files):
    raise RuntimeError(
        f"Expected {len(source_files)} converted MGFs, found {len(outputs)}"
    )
if any(path.is_symlink() for path in outputs):
    raise RuntimeError("Converted output unexpectedly contains symlinks")
print(f"Prepared {len(outputs)} yeast ProForma MGFs")
PY
