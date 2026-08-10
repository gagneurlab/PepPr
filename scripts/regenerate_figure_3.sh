#!/usr/bin/env bash
# Regenerate figure 3 after KoL human_iso eval and mutations inference finish.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$(dirname "$0")/dnps_data_path.sh"
KOL_RUNS="/s/project/denovo-prosit/SamKhan/dnps_hybrid/kingdoms_of_life_eval_asymbnln/runs"
MUT_MZTAB="$DNPS_DATA_PATH/results/casanovo/mutations_hybrid_human_iso.mztab"

n_iso=$(find "$KOL_RUNS" -name 'casanovo_pp_human_iso.log' -exec grep -l 'Peptide Precision' {} + 2>/dev/null | wc -l)
echo "human_iso KoL runs with precision: $n_iso / 15 (expect 14 non-self + human self optional)"
if [[ "$n_iso" -lt 10 ]]; then
  echo "[warn] fewer than 10 human_iso KoL logs — panel D may be incomplete" >&2
fi

if [[ ! -s "$MUT_MZTAB" ]]; then
  echo "[fatal] missing $MUT_MZTAB (submit run_mutations_inference_human_iso.sh)" >&2
  exit 2
fi

python plot_kol_overlap_vs_pp_gain.py \
  --runs-dir "$KOL_RUNS" \
  --plm-species human_iso mouse \
  --csv-out kol_overlap_vs_pp_gain.csv \
  --out kol_overlap_vs_pp_gain.png

python dnps_hybrid/plot_figure_3.py
echo "Wrote $REPO_ROOT/figure_3.png"
