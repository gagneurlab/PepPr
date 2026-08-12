#!/usr/bin/env python3
"""Beslic mAb per-(mAb, protease) precision-coverage grid.

Rows × cols = mAbs × non-tryptic proteases. Each panel shows PC curves for
two arms: vanilla Casanovo v5 vs Casanovo+PepPr (germline_{species}_sw_clean).
Empty panels indicate proteases that weren't run for that mAb.

Usage:
    python experiments/supp/plot_supp_fig_beslic.py
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from peptide_priors.metrics import (
    MASSIVEKB_MASSES, _normalize_to_massivekb, _expand_masses,
    load_mztab_with_mgf, evaluate, average_precision,
)
from peptide_priors.const import (
    COLOR_CASANOVO as COLOR_VAN,
    COLOR_PP,
    PROJECT_ROOT,
)
from experiments.fig4.nontryp_registry import RUNS

# ── Layout: 6 mAbs × 8 proteases ────────────────────────────────────────
MABS = ["IgG1_Human_H", "IgG1_Human_L", "Herceptin",
        "anti-FLAG-M2", "WIgG1_H", "WIgG1_L"]
MAB_LABELS = {
    "IgG1_Human_H": "IgG1_Human HC",
    "IgG1_Human_L": "IgG1_Human LC",
    "Herceptin":    "Herceptin",
    "anti-FLAG-M2": "anti-FLAG-M2",
    "WIgG1_H":      "WIgG1 HC",
    "WIgG1_L":      "WIgG1 LC",
}
PROTEASES = ["aspn", "chymo", "elastase", "gluc",
             "lysc", "lysn", "proteinasek", "thermolysin"]
PROTEASE_LABELS = {
    "aspn": "AspN", "chymo": "Chymo", "elastase": "Elastase", "gluc": "GluC",
    "lysc": "LysC", "lysn": "LysN", "proteinasek": "ProteinaseK",
    "thermolysin": "Thermolysin",
}

# Species per mAb for +PP arm selection
_SPECIES = {"IgG1_Human_H": "human", "IgG1_Human_L": "human",
            "Herceptin": "human", "anti-FLAG-M2": "mouse",
            "WIgG1_H": "mouse", "WIgG1_L": "mouse"}

OUT_PATH = os.path.join(PROJECT_ROOT, "supp_fig_beslic.png")


def _latest_mztab(res_dir, pattern):
    files = glob.glob(os.path.join(res_dir, pattern))
    return max(files, key=os.path.getmtime) if files else None


def _find_run(mab, protease):
    for r in RUNS:
        if r.mab_id == mab and r.protease == protease:
            return r
    return None


def _pc_from_mztab(path):
    if path is None or not os.path.exists(path):
        return None
    df = load_mztab_with_mgf(path)
    if len(df) == 0:
        return None
    df = df.dropna(subset=["true_seq"]).copy()
    if len(df) == 0:
        return None
    df["pred"] = df["pred"].apply(_normalize_to_massivekb)
    df["true_seq"] = df["true_seq"].apply(_normalize_to_massivekb)
    df["score"] = df["score"].astype(float)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    masses = _expand_masses(df["pred"].tolist() + df["true_seq"].tolist(),
                            MASSIVEKB_MASSES)
    matches = evaluate.aa_match_batch(df["true_seq"].tolist(),
                                       df["pred"].tolist(), masses)
    pep = np.asarray([m[1] for m in matches[0]])
    if len(pep) == 0:
        return None
    n_total = len(pep)
    cov = np.arange(1, n_total + 1) / n_total
    prec = np.cumsum(pep) / np.arange(1, n_total + 1)
    ap = average_precision(cov, prec, n_total=n_total)
    return cov, prec, ap, n_total


def main():
    plt.style.use("ggplot")
    plt.rcParams.update({
        "axes.facecolor": "white", "axes.edgecolor": "black",
        "axes.linewidth": 0.8, "axes.grid": False,
        "axes.labelsize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 7, "font.family": "sans-serif",
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })

    nrows, ncols = len(MABS), len(PROTEASES)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 2.8 * nrows),
                             squeeze=False)

    n_plotted = 0
    for ri, mab in enumerate(MABS):
        for ci, protease in enumerate(PROTEASES):
            ax = axes[ri, ci]
            run = _find_run(mab, protease)
            if run is None:
                ax.set_axis_off()
                continue
            sp = _SPECIES[mab]
            van_path = _latest_mztab(run.res_dir,
                                     f"casanovo_{mab}_{protease}_noplm_*.mztab")
            pp_path  = _latest_mztab(
                run.res_dir,
                f"casanovo_{mab}_{protease}_germline_{sp}_sw_clean_*.mztab")

            plotted = False
            for label, color, path in [
                ("Casanovo",       COLOR_VAN, van_path),
                ("Casanovo + PepPr",  COLOR_PP,  pp_path),
            ]:
                r = _pc_from_mztab(path)
                if r is None:
                    continue
                cov, prec, ap_v, n = r
                ax.plot(cov, prec, lw=1.8, color=color,
                        label=f"{label} (AP={ap_v:.2f}, pep recall={float(prec[-1]):.2f})",
                        alpha=0.9)
                plotted = True
            if plotted:
                n_plotted += 1
                ax.legend(loc="lower left", framealpha=0.7)
            else:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray",
                        fontstyle="italic")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            title = f"{MAB_LABELS[mab]} — {PROTEASE_LABELS[protease]}"
            ax.set_title(title, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == nrows - 1:
                ax.set_xlabel("Coverage", fontsize=9)
            if ci == 0:
                ax.set_ylabel("Peptide precision", fontsize=9)

    fig.suptitle("Beslic mAbs — per-antibody precision-coverage curves, "
                 "by protease", fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({n_plotted} panels populated)")
    plt.close(fig)


if __name__ == "__main__":
    main()
