#!/usr/bin/env python3
"""Peptide precision by length for Casanovo and Casanovo+PepPr (same-species PLM).

Reads the asymbnln (BN-cas + LN-plm) MassiveKB fusion head outputs
(`9s_<sp>_asymbnln.mztab`). One panel per species, arranged in a 4×2 grid.
Results are cached to precision_by_length_cache.pkl so re-runs only recompute
missing species.
"""

import os
import sys
import re
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dnps_hybrid.metrics import (
    _normalize_to_massivekb,
    _expand_masses,
    evaluate,
    MASSIVEKB_MASSES,
    load_mztab_with_mgf,
)
from dnps_hybrid.const import (
    COLOR_CASANOVO as COLOR_DNPS,
    COLOR_HISTOGRAM_GRAY,
    COLOR_PP as COLOR_HYBRID,
    COLOR_TEXT_GRAY,
    SPECIES,
    result_run_path,
)

plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
})

LENGTH_BINS = [
    ("≤9",    0,   9),
    ("10–13", 10, 13),
    ("14–17", 14, 17),
    ("≥18",  18, 999),
]

SPECIES_ORDER = [
    "endoloripes", "mouse", "archaeon", "human",
    "honeybee", "cowpea", "tomato", "yeast",
    "bacillus",
]

# Display-name overrides to match plot_scatter_precision.py
SPECIES_LABELS = {
    "cowpea":      "ricebean",
    "archaeon":    "mmazei",
    "endoloripes": "clambacteria",
}
def species_label(sp: str) -> str:
    return SPECIES_LABELS.get(sp, sp)

CACHE_PATH = "precision_by_length_cache.pkl"

_MOD_RE = re.compile(r"[^A-Z]")

def _aa_len(seq):
    return len(_MOD_RE.sub("", seq)) if isinstance(seq, str) else 0


def load_and_evaluate(path):
    """Load mztab, evaluate correctness, return df with pep_match and true_len."""
    df = load_mztab_with_mgf(path)
    df["pred"]     = df["pred"].apply(_normalize_to_massivekb)
    df["true_seq"] = df["true_seq"].apply(_normalize_to_massivekb)
    df["score"]    = df["score"].astype(float)
    masses = _expand_masses(
        df["pred"].tolist() + df["true_seq"].tolist(), MASSIVEKB_MASSES
    )
    res = evaluate.aa_match_batch(df["true_seq"].tolist(), df["pred"].tolist(), masses)
    df["pep_match"] = [m[1] for m in res[0]]
    df["true_len"]  = df["true_seq"].apply(_aa_len)
    return df


def precision_by_length(df):
    """Return list of (label, precision) for each length bin."""
    out = []
    for label, lo, hi in LENGTH_BINS:
        mask = (df["true_len"] >= lo) & (df["true_len"] <= hi)
        sub  = df[mask]
        prec = float(sub["pep_match"].values.astype(float).mean()) if len(sub) > 0 else np.nan
        n    = len(sub)
        out.append((label, prec, n))
    return out


def main():
    # ── Load or build cache ──────────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as fh:
            cache = pickle.load(fh)
        print(f"Loaded cache: {list(cache.keys())}", flush=True)
    else:
        cache = {}

    for species in SPECIES_ORDER:
        if species in cache:
            print(f"  {species}: cached", flush=True)
            continue
        sp_cfg = SPECIES.get(species)
        if sp_cfg is None:
            continue
        base = result_run_path(sp_cfg["run_name"])
        dnps_path  = os.path.join(base, f"9s_{species}_dnps.mztab")
        pp_path    = os.path.join(base, f"9s_{species}_asymbnln.mztab")
        if not os.path.exists(dnps_path) or not os.path.exists(pp_path):
            print(f"  {species}: mztab(s) missing, skipping", flush=True)
            continue
        print(f"  {species} loading...", end=" ", flush=True)
        dnps_df = load_and_evaluate(dnps_path)
        pp_df   = load_and_evaluate(pp_path)
        print("done", flush=True)
        cache[species] = {
            "dnps": precision_by_length(dnps_df),
            "pp":   precision_by_length(pp_df),
            "n":    len(dnps_df),
        }
        with open(CACHE_PATH, "wb") as fh:
            pickle.dump(cache, fh)

    # ── Plot ─────────────────────────────────────────────────────────────
    species_available = [sp for sp in SPECIES_ORDER if sp in cache]
    ncols = 3
    nspec = len(species_available)
    nrows = (nspec + ncols - 1) // ncols

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # Outer grid: one cell per species, with generous spacing between species
    fig = plt.figure(figsize=(3.8 * ncols, 4.2 * nrows))
    outer = GridSpec(nrows, ncols, hspace=0.45, wspace=0.35, figure=fig,
                     top=0.93, bottom=0.09)

    x       = np.arange(len(LENGTH_BINS))
    xlabels = [b[0] for b in LENGTH_BINS]

    for idx, species in enumerate(species_available):
        grow, gcol = divmod(idx, ncols)

        # Inner grid: line chart (height 3) + histogram (height 1), tight gap
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[grow, gcol],
            height_ratios=[3, 1], hspace=0.05,
        )
        ax_line = fig.add_subplot(inner[0])
        ax_hist = fig.add_subplot(inner[1])

        cv     = cache[species]
        prec_d = [v[1] for v in cv["dnps"]]
        prec_p = [v[1] for v in cv["pp"]]
        ns     = [v[2] for v in cv["dnps"]]

        # ── Line chart ───────────────────────────────────────────────
        ax_line.plot(x, prec_d, marker="o", lw=2.0, color=COLOR_DNPS,
                     markersize=5, label="Casanovo")
        ax_line.plot(x, prec_p, marker="o", lw=2.0, color=COLOR_HYBRID,
                     markersize=5, label="Casanovo+PepPr")
        ax_line.set_xlim(-0.5, len(x) - 0.5)
        ax_line.set_ylim(0, 1)
        ax_line.set_xticks(x)
        ax_line.set_xticklabels([])
        ax_line.set_title(f"{species_label(species)}  (n={cv['n']:,})", fontsize=10,
                          fontweight="bold")
        ax_line.spines["top"].set_visible(False)
        ax_line.spines["right"].set_visible(False)
        ax_line.tick_params(labelsize=9)
        if gcol == 0:
            ax_line.set_ylabel("Peptide precision", fontsize=9)

        # ── Histogram ────────────────────────────────────────────────
        ax_hist.bar(x, ns, width=0.6, color=COLOR_HISTOGRAM_GRAY, linewidth=0)
        for i, n in enumerate(ns):
            ax_hist.text(i, n, f"{n:,}", ha="center", va="bottom",
                         fontsize=8, color=COLOR_TEXT_GRAY)
        ax_hist.set_xlim(-0.5, len(x) - 0.5)
        ax_hist.set_xticks(x)
        ax_hist.set_xticklabels(xlabels, fontsize=8)
        ax_hist.set_yticks([])
        ax_hist.set_ylim(0, max(ns) * 1.4 if max(ns) > 0 else 1)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.spines["left"].set_visible(False)
        ax_hist.tick_params(labelsize=8)
        if gcol == 0:
            ax_hist.set_ylabel("n PSMs", fontsize=8)

    # Hide unused outer cells
    for idx in range(nspec, nrows * ncols):
        grow, gcol = divmod(idx, ncols)
        fig.add_subplot(outer[grow, gcol]).set_visible(False)

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color=COLOR_DNPS,   lw=2, marker="o", ms=5,
                   label="Casanovo"),
        plt.Line2D([0], [0], color=COLOR_HYBRID,  lw=2, marker="o", ms=5,
                   label="Casanovo+PepPr"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Peptide Precision by Length", fontsize=14)

    out = "precision_by_length.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
