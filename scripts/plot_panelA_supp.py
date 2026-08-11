#!/usr/bin/env python3
"""Supplementary figure: the three MSFragger DMO database-search curves for the
ProteomeTools SAAV benchmark in one panel, isolating how the DB call is scored and
ranked (de novo curves live in the main Figure 3 Panel A).

  - site-localized (Percolator-ranked): the main-panel DB curve. Correct = the SAAV
    offset, placed at its best-available localized residue (PTM-Prophet where present,
    else MSFragger localize_delta_mass), reconstructs the ground-truth variant.
  - un-localized / peptide-level (Percolator-ranked): localization ignored — correct if
    ANY valid placement of the offset on the WT backbone yields the GT peptide. The gap
    above the site-localized curve is the pure localization penalty.
  - site-localized (localization-score-ranked): the same site-localized correctness but
    ordered by PTM-Prophet best-localization probability instead of Percolator, showing
    the result's sensitivity to how the DB calls are ranked.

Reads the precomputed panelA_curves.npz (build_panelA_data.py).
Run: python scripts/plot_panelA_supp.py
"""
import os
os.environ.setdefault("DNPS_DATA_PATH", "/s/project/denovo-prosit/SamKhan/dnps_hybrid_zenodo")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from dnps_hybrid.const import COLOR_GREEN, PROTEOMETOOLS_SAAV_DIR, PROJECT_ROOT

NPZ = os.path.join(PROTEOMETOOLS_SAAV_DIR, "panelA_curves.npz")


def plot_panelA_supp(ax):
    d = np.load(NPZ)
    ap = lambda x, y: auc(x, y)
    ax.plot(d["dmo_pep_cov"], d["dmo_pep_prec"], color=COLOR_GREEN, lw=2.1, ls="--",
            label=f"DMO, un-localized · Percolator-ranked (AP={ap(d['dmo_pep_cov'], d['dmo_pep_prec']):.3f})")
    ax.plot(d["dmo_site_cov"], d["dmo_site_prec"], color=COLOR_GREEN, lw=2.3, ls="-",
            label=f"DMO, site-localized · Percolator-ranked (AP={ap(d['dmo_site_cov'], d['dmo_site_prec']):.3f})")
    ax.plot(d["dmo_site_ptm_cov"], d["dmo_site_ptm_prec"], color=COLOR_GREEN, lw=1.8, ls=":",
            label=f"DMO, site-localized · loc-score-ranked (AP={ap(d['dmo_site_ptm_cov'], d['dmo_site_ptm_prec']):.3f})")
    if "fdr_cov" in d.files:                              # 1% PSM-FDR point (Percolator-ranked)
        cf, sf = float(d["fdr_cov"]), float(d["fdr_site_prec"])
        pf = float(d["fdr_pep_prec"]) if "fdr_pep_prec" in d.files else None
        ys = [sf] + ([pf] if pf is not None else [])
        if pf is not None:
            ax.plot([cf, cf], [sf, pf], color=COLOR_GREEN, lw=0.9, alpha=0.6, zorder=5)
        for yv in ys:
            ax.plot([cf], [yv], "o", color=COLOR_GREEN, ms=8, mec="black", mew=1.0, zorder=6)
        ax.annotate("1% FDR", xy=(cf, sf), xytext=(cf + 0.03, sf - 0.11), fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage", fontsize=11)
    ax.set_ylabel("Peptide precision", fontsize=11)
    ax.set_title("MSFragger DMO on ProteomeTools SAAVs — localization & ranking", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.85, handlelength=3.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def main():
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    plot_panelA_supp(ax)
    out = os.path.join(PROJECT_ROOT, "figure_s_msfragger_curves.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
