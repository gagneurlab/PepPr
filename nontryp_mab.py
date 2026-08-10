"""Build MABS-style entries for non-tryptic (mAb, protease) runs.

Bridges nontryp_registry.RUNS (which encodes inference + ground truth paths)
to the MAbSpec dataclass used by plot_pc_curves_mab.py / plot_acc_bars_mab.py.

Each non-tryptic run has 2 Casanovo arms (no pepLM, multi-protease pepLM). The
antibody germline pepLM is intentionally NOT wired -- it's tuned for tryptic Ig
peptides and would be off-distribution here.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from dnps_hybrid.const import MABS_BENCHMARK_DIR
from plot_pc_curves_mab import MAbSpec
from nontryp_registry import RUNS, PROTEASE_ORDER, PROTEASE_LABEL, PXD023419_RAW

# Beslic's combined-fraction PEAKS CSVs live under one dir per chain. Each row
# is prefixed F<n>:<scan>; the F-number identifies the protease run. F-numbers
# verified empirically from C/N-terminal residue distributions:
#   F1: AspN (~72% peptides start with D)
#   F2: Chymotrypsin (~87% C-term in L/Y/W/F/M)
#   F3: Trypsin (already wired in plot_pc_curves_mab.MABS for the tryptic specs)
# GluC, LysC, ProteinaseK are NOT in these CSVs (Beslic only published PEAKS
# for 3 proteases per chain), so those (mAb, protease) combos stay peaks=None.
_PEAKS_DIR_BASE = os.path.join(
    MABS_BENCHMARK_DIR,
    "figshare_21394143",
    "01-RawData",
    "MSV000079801_CompleteAssembly",
    "ALPS_Data",
)
_PEAKS_CHAIN_DIR = {
    "IgG1_Human_H": "Data_Human_Heavy",
    "IgG1_Human_L": "Data_Human_Light",
}
_PEAKS_F_NUM = {"aspn": 1, "chymo": 2}  # trypsin (F3) is wired in MABS, not here


def _peaks_for(mab_id: str, protease: str) -> Optional[Tuple[str, str, str, int]]:
    """Return (db_csv, denovo_csv, spider_csv, f_num) for a (mAb, protease) pair,
    or None if Beslic didn't publish PEAKS for that combination."""
    chain_dir = _PEAKS_CHAIN_DIR.get(mab_id)
    f_num = _PEAKS_F_NUM.get(protease)
    if chain_dir is None or f_num is None:
        return None
    base = f"{_PEAKS_DIR_BASE}/{chain_dir}"
    return (f"{base}/peptides_db.csv",
            f"{base}/peptides_0.csv",
            f"{base}/peptides_spider.csv",
            f_num)


def _spec_for_run(run) -> MAbSpec:
    short_title = f"{run.short_mab} ({run.short_protease})"
    title = f"{run.short_mab} ({run.short_protease}, FragPipe 1% FDR)"
    # Beslic's Herceptin summary.csv uses a re-indexed 0-based scan counter into
    # the original raw MS2 order, NOT the Thermo scan numbers our annotated MGF
    # carries. Pass the per-protease source MGF (which has SCANS=<Thermo>) so
    # parse_summary_csv can remap. The Beslic IgG1 summary.csv already keeps
    # Thermo scans in 'Old scan: N', so no remap needed there.
    remap_mgf = (
        f"{PXD023419_RAW}/{run.sample}.mgf" if run.mab_id == "Herceptin" else None
    )
    peaks = _peaks_for(run.mab_id, run.protease)
    if peaks is not None:
        db_csv, denovo_csv, spider_csv, f_num = peaks
        peaks_db = (db_csv, f_num)
        peaks_denovo = (denovo_csv, f_num)
        peaks_spider = (spider_csv, f_num)
    else:
        peaks_db = peaks_denovo = peaks_spider = None
    return MAbSpec(
        mab_id=f"{run.mab_id}_{run.protease}",
        short_title=short_title,
        title=title,
        annotated_mgf=run.annotated_mgf,
        casanovo_results_dir=run.res_dir,
        noplm_glob=f"casanovo_{run.mab_id}_{run.protease}_noplm_*.mztab",
        note="",
        multiplm_glob=f"casanovo_{run.mab_id}_{run.protease}_multiplm_*.mztab",
        peaks_db=peaks_db,
        peaks_denovo=peaks_denovo,
        peaks_spider=peaks_spider,
        # Supernovo CSV only has Herceptin _tryp rows (verified empirically);
        # non-tryptic substrings return 0 hits and break evaluate(). Always None.
        supernovo=None,
        summary_csv=run.summary_csv,
        summary_scan_remap_mgf=remap_mgf,
    )


def specs_for_protease(protease: str) -> List[MAbSpec]:
    return [_spec_for_run(r) for r in RUNS if r.protease == protease]


def all_protease_groups() -> List[tuple]:
    """Return [(protease_key, label, [MAbSpec, ...]), ...] in display order."""
    groups = []
    for prot in PROTEASE_ORDER:
        specs = specs_for_protease(prot)
        if not specs:
            continue
        groups.append((prot, PROTEASE_LABEL[prot], specs))
    return groups
