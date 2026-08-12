#!/usr/bin/env python3
"""Supplementary Figure 4 — Cross-species (human-pepLM) PC curves.

Same 3×3 layout as supp_fig_3, but the +PP arm uses the **human** pepLM on
every non-human nine-species dataset (cross-species transfer).  Tests whether
the human prior generalises when the species differs.

Reads cached curves from ``pc_curves_cache.pkl`` (built by
plot_scatter_precision.py).

Usage:
    python experiments/supp/plot_supp_fig_4.py
"""
import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from experiments.utils.evaluation import (
    NINE_SPECIES_ORDER, species_label,
    NINE_SPECIES_NCOLS, NINE_SPECIES_NROWS, NINE_SPECIES_PANEL_INCHES,
    NINE_SPECIES_LINE_WIDTH, NINE_SPECIES_TICK_FONTSIZE,
    NINE_SPECIES_LABEL_FONTSIZE, NINE_SPECIES_TITLE_FONTSIZE,
    NINE_SPECIES_COLOR_DNPS, NINE_SPECIES_COLOR_PP,
)
# Reuse the panel renderer so supp_fig_3 and supp_fig_4 are identical in style.
from plot_supp_fig_3 import _plot_panel, _n_total_for, _rescale
from peptide_priors.const import PROJECT_ROOT

plt.rcParams.update({
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
})

CURVES_CACHE = os.path.join(PROJECT_ROOT, "pc_curves_cache.pkl")
OUT_PATH     = os.path.join(PROJECT_ROOT, "supp_fig_4.png")
CONDITION    = "cross"    # human pepLM applied to every non-human dataset


def main():
    if not os.path.exists(CURVES_CACHE):
        sys.exit(f"missing cache: {CURVES_CACHE}")
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

    fig.suptitle("Cross-species PLM (human prior) precision–coverage curves",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
