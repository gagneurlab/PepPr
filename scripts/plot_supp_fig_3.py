#!/usr/bin/env python3
"""Supplementary Figure 3 — Same-species PLM precision-coverage curves.

3×3 grid (one panel per nine-species dataset) showing Casanovo vs Casanovo+PepPr
PC curves where the pepLM is matched to the species (so e.g. mouse pepLM on
mouse data, human pepLM on human data, ...).  Reads cached curves from
``pc_curves_cache.pkl`` (built by plot_scatter_precision.py).

Usage:
    python scripts/plot_supp_fig_3.py
"""
import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from dnps_hybrid.metrics import (
    NINE_SPECIES_ORDER, species_label,
    NINE_SPECIES_NCOLS, NINE_SPECIES_NROWS, NINE_SPECIES_PANEL_INCHES,
    NINE_SPECIES_LINE_WIDTH, NINE_SPECIES_TICK_FONTSIZE,
    NINE_SPECIES_LABEL_FONTSIZE, NINE_SPECIES_TITLE_FONTSIZE,
    NINE_SPECIES_COLOR_DNPS, NINE_SPECIES_COLOR_PP,
    count_mgf_spectra,
)
from dnps_hybrid.const import PROJECT_ROOT, result_run_path
from sklearn.metrics import auc

# Map 9-species keys to their results directory (for the mztab → MGF spectra
# count used as the coverage denominator).
_SPECIES_DATA_DIRS = {
    "human":       "casanovo",
    "mouse":       "mus_musculus",
    "yeast":       "saccharomyces_cerevisiae",
    "bacillus":    "bacillus_subtilis",
    "honeybee":    "apis_mellifera",
    "tomato":      "solanum_lycopersicum",
    "cowpea":      "vigna_mungo",
    "archaeon":    "methanosarcina_mazei",
    "endoloripes": "candidatus_endoloripes",
}
def _n_total_for(species):
    """Total MGF spectra in the species' nine-species results dataset."""
    dirname = _SPECIES_DATA_DIRS.get(species)
    if dirname is None:
        return None
    mztab = os.path.join(result_run_path(dirname), f"9s_{species}_dnps.mztab")
    if not os.path.exists(mztab):
        return None
    return count_mgf_spectra(mztab) or None


def _rescale(curve, n_total):
    """Renormalize cached cov from i/len → i/n_total and recompute AP."""
    cov, prec, _ = curve
    if n_total is None or len(cov) == 0:
        return curve
    new_cov = cov * (len(cov) / n_total)
    return new_cov, prec, float(auc(new_cov, prec))

plt.rcParams.update({
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
})

CURVES_CACHE = os.path.join(PROJECT_ROOT, "pc_curves_cache.pkl")
OUT_PATH     = os.path.join(PROJECT_ROOT, "supp_fig_3.png")
CONDITION    = "same"     # human pepLM on human, mouse pepLM on mouse, ...


def _plot_panel(ax, cov_d, prec_d, auc_d, cov_p, prec_p, auc_p, title, n):
    final_d = float(prec_d[-1]) if len(prec_d) else 0.0
    final_p = float(prec_p[-1]) if len(prec_p) else 0.0
    ax.plot(cov_d, prec_d, color=NINE_SPECIES_COLOR_DNPS,
            lw=NINE_SPECIES_LINE_WIDTH,
            label=f"Casanovo (AP={auc_d:.3f}, pep recall={final_d:.2f})")
    ax.plot(cov_p, prec_p, color=NINE_SPECIES_COLOR_PP,
            lw=NINE_SPECIES_LINE_WIDTH,
            label=f"Casanovo+PepPr (AP={auc_p:.3f}, pep recall={final_p:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} (n={n:,})", fontsize=NINE_SPECIES_TITLE_FONTSIZE)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)
    ax.grid(False)
    ax.tick_params(labelsize=NINE_SPECIES_TICK_FONTSIZE)


def main():
    if not os.path.exists(CURVES_CACHE):
        sys.exit(f"missing cache: {CURVES_CACHE} "
                 f"— run plot_scatter_precision.py first to build it")
    with open(CURVES_CACHE, "rb") as f:
        curves = pickle.load(f)

    species_available = [sp for sp in NINE_SPECIES_ORDER
                         if (CONDITION, sp) in curves]
    if not species_available:
        sys.exit(f"no curves with condition={CONDITION!r}")

    ncols = NINE_SPECIES_NCOLS
    nrows = NINE_SPECIES_NROWS
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(NINE_SPECIES_PANEL_INCHES[0] * ncols,
                 NINE_SPECIES_PANEL_INCHES[1] * nrows),
    )
    axes = np.atleast_2d(axes)
    for idx, sp in enumerate(species_available):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        cv = curves[(CONDITION, sp)]
        n_total = _n_total_for(sp)
        dnps = _rescale(cv["dnps"], n_total)
        pp   = _rescale(cv["pp"],   n_total)
        _plot_panel(ax, *dnps, *pp, species_label(sp), cv["n"])
        if r == nrows - 1:
            ax.set_xlabel("Coverage", fontsize=NINE_SPECIES_LABEL_FONTSIZE)
        if c == 0:
            ax.set_ylabel("Peptide precision", fontsize=NINE_SPECIES_LABEL_FONTSIZE)
    for idx in range(len(species_available), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Same-species PLM precision–coverage curves",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
