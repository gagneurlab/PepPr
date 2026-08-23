#!/usr/bin/env python3
"""Supplementary Figure 1 — peptide precision by length, per species.

3×3 grid (one panel per nine-species dataset); each panel shows Casanovo vs
Casanovo+PepPr precision in four length bins with a PSM-count histogram below.
Reads the asymbnln MassiveKB fusion-head outputs (`9s_<sp>_asymbnln.mztab`).
Results are cached to ``precision_by_length_cache.pkl`` so re-runs only
recompute missing species.

Usage:
    python experiments/supp/plot_supp_fig_1.py
"""

import os
import sys
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from experiments.utils.evaluation import (
    _normalize_to_massivekb,
    _expand_masses,
    evaluate,
    MASSIVEKB_MASSES,
    load_mztab_with_mgf,
    NINE_SPECIES_ORDER, species_label,
    NINE_SPECIES_NCOLS, NINE_SPECIES_PANEL_INCHES,
    NINE_SPECIES_LINE_WIDTH, NINE_SPECIES_MARKER_SIZE,
    NINE_SPECIES_TICK_FONTSIZE, NINE_SPECIES_LABEL_FONTSIZE,
    NINE_SPECIES_TITLE_FONTSIZE,
    NINE_SPECIES_COLOR_DNPS, NINE_SPECIES_COLOR_PP,
)
from peppr.const import (
    COLOR_HISTOGRAM_GRAY,
    COLOR_TEXT_GRAY,
    PROJECT_ROOT,
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

# Shared with the other three nine-species supp figs (see metrics.py).
SPECIES_ORDER = NINE_SPECIES_ORDER
COLOR_DNPS    = NINE_SPECIES_COLOR_DNPS
COLOR_HYBRID  = NINE_SPECIES_COLOR_PP
CACHE_PATH    = os.path.join(PROJECT_ROOT, "precision_by_length_cache.pkl")

_MOD_RE = re.compile(r"[^A-Z]")

def _aa_len(seq):
    return len(_MOD_RE.sub("", seq)) if isinstance(seq, str) else 0


def load_and_evaluate(path):
    """Load mztab, evaluate correctness, return df with pep_match and true_len."""
    paths = [path] if isinstance(path, (str, os.PathLike)) else path
    df = pd.concat(
        [load_mztab_with_mgf(input_path) for input_path in paths],
        ignore_index=True,
    )
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
        sp_cfg = SPECIES.get(species)
        if sp_cfg is None:
            continue
        base = result_run_path(sp_cfg["run_name"])
        dnps_path  = os.path.join(base, f"9s_{species}_dnps.mztab")
        if species == "yeast":
            pp_path = sorted(
                str(path) for path in Path(base).glob("9s_yeast_hybrid_*.mztab")
            )
            source_files = tuple(pp_path)
        elif species == "human":
            pp_path = os.path.join(base, "9s_human_plmhuman_iso_hybrid.mztab")
            source_files = (pp_path,)
        else:
            pp_path = os.path.join(base, f"9s_{species}_asymbnln.mztab")
            source_files = None
        cache_is_stale = (
            source_files is not None
            and cache.get(species, {}).get("source_files") != source_files
        )
        if species in cache and not cache_is_stale:
            print(f"  {species}: cached", flush=True)
            continue
        if not os.path.exists(dnps_path) or not pp_path or (
            isinstance(pp_path, str) and not os.path.exists(pp_path)
        ):
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
            "source_files": source_files,
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
    fig = plt.figure(figsize=(NINE_SPECIES_PANEL_INCHES[0] * ncols,
                              NINE_SPECIES_PANEL_INCHES[1] * nrows))
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
        ax_line.plot(x, prec_d, marker="o", lw=NINE_SPECIES_LINE_WIDTH,
                     color=COLOR_DNPS, markersize=NINE_SPECIES_MARKER_SIZE,
                     label="Casanovo")
        ax_line.plot(x, prec_p, marker="o", lw=NINE_SPECIES_LINE_WIDTH,
                     color=COLOR_HYBRID, markersize=NINE_SPECIES_MARKER_SIZE,
                     label="Casanovo+PepPr")
        ax_line.set_xlim(-0.5, len(x) - 0.5)
        ax_line.set_ylim(0, 1)
        ax_line.set_xticks(x)
        ax_line.set_xticklabels([])
        ax_line.set_title(f"{species_label(species)} (n={cv['n']:,})",
                          fontsize=NINE_SPECIES_TITLE_FONTSIZE)
        ax_line.spines["top"].set_visible(False)
        ax_line.spines["right"].set_visible(False)
        ax_line.tick_params(labelsize=9)
        if gcol == 0:
            ax_line.set_ylabel("Peptide precision", fontsize=NINE_SPECIES_LABEL_FONTSIZE)

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

    out = os.path.join(PROJECT_ROOT, "supp_fig_1.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
