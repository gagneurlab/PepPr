#!/usr/bin/env python3
"""Supplementary Figure 5 — Per-antibody PC curves, by region category.

Rows × columns = mAbs × region categories (xa_novo, PXD060500).  Each panel
pools PSMs from all five proteases for that (mAb, region) and shows three
precision–coverage curves: Casanovo, XA-Novo, Casanovo+PepPr.

Region categories (matching figure-4 Panel A ordering, by descending
Casanovo precision):
    CDR-only | FR-only | C-only | FR<->CDR | V<->C

Usage:
    python scripts/plot_supp_fig_5.py
"""
import sys

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from dnps_hybrid import const
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from dnps_hybrid.metrics import (
    MASSIVEKB_MASSES, _normalize_to_massivekb, _expand_masses,
    load_mztab_with_mgf, evaluate,
)
from dnps_hybrid.antibody_utils import (
    ARM_STYLE,
    MAB_SPECS,
    XANOVO_PROTEASES,
    categorize_gt,
    find_mztab,
    read_fasta_with_regions,
)

MABS = list(MAB_SPECS)
REGIONS = ["CDR-only", "FR-only", "C-only", "FR<->CDR", "V<->C"]
OUT_PATH = const.SUPP_FIGURE_5_PATH

# ARM_STYLE from xa_novo tags labels as "Casanovo v5" / "XA-Novo v3". Drop
# the version suffixes for the supp fig legend. Beslic mAbs use `noplm` for
# vanilla and `germline_{sp}_sw_clean` for +PP — style them like their
# xa_novo counterparts.
_LABEL_OVERRIDES = {
    "vanilla":                   "Casanovo",
    "noplm":                     "Casanovo",
    "xanovo_v3":                 "XA-Novo",
    "antibody_mouse":   "Casanovo+PepPr (mouse)",
    "antibody_human":   "Casanovo+PepPr (human)",
}
_COLOR_OVERRIDES = {
    "noplm":                   ARM_STYLE["vanilla"]["color"],
    "antibody_mouse": ARM_STYLE["germline_mouse_sw"]["color"],
    "antibody_human": ARM_STYLE["germline_human_sw"]["color"],
}


def _load_with_cat(path, ref_chains):
    df = load_mztab_with_mgf(path)
    if "true_seq" not in df.columns or len(df) == 0:
        return None
    df = df.dropna(subset=["true_seq"]).copy()
    if len(df) == 0:
        return None
    df["category"] = df["true_seq"].apply(
        lambda s: categorize_gt(str(s), ref_chains)
    )
    df = df.dropna(subset=["category"]).copy()
    if len(df) == 0:
        return None
    df["pred"] = df["pred"].apply(_normalize_to_massivekb)
    df["true_seq"] = df["true_seq"].apply(_normalize_to_massivekb)
    df["score"] = df["score"].astype(float)
    return df


def _pc_from_df(df):
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    masses = _expand_masses(df["pred"].tolist() + df["true_seq"].tolist(),
                            MASSIVEKB_MASSES)
    matches = evaluate.aa_match_batch(df["true_seq"].tolist(),
                                       df["pred"].tolist(), masses)
    pep = np.asarray([m[1] for m in matches[0]])
    if len(pep) == 0:
        return None
    cov = np.arange(1, len(pep) + 1) / len(pep)
    prec = np.cumsum(pep) / np.arange(1, len(pep) + 1)
    auc_val = float(auc(cov, prec)) if len(pep) >= 2 else float(pep.mean())
    return cov, prec, float(pep.mean()), auc_val, len(pep)


def main():
    plt.style.use("ggplot")
    plt.rcParams.update({
        "axes.facecolor": "white", "axes.edgecolor": "black",
        "axes.linewidth": 0.8, "axes.grid": False,
        "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 8, "font.family": "sans-serif",
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })

    # Preload references for region classification
    ref_by_mab = {}
    for mab in MABS:
        chains = read_fasta_with_regions(MAB_SPECS[mab]["ref"])
        chains = {k: v for k, v in chains.items()
                  if v["cdrs"] is not None and v["vc"] is not None}
        if not chains:
            sys.exit(f"{mab}: no annotated chains in {MAB_SPECS[mab]['ref']}")
        ref_by_mab[mab] = chains

    nrows = len(MABS)
    ncols = len(REGIONS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows),
                             squeeze=False)

    n_plotted = 0
    for ri, mab in enumerate(MABS):
        spec = MAB_SPECS[mab]
        pp_arm  = spec["pp_arm"]
        van_arm = spec.get("vanilla_arm", "vanilla")
        # Beslic mAbs don't have XA-Novo runs; drop that arm from the panel.
        arms = ([van_arm, "xanovo_v3", pp_arm] if spec.get("source") != "beslic"
                else [van_arm, pp_arm])
        proteases = spec.get("proteases", XANOVO_PROTEASES)

        # Pre-load all (protease, arm) dfs for this mAb so we can pool across proteases
        all_dfs = {}                         # (protease, arm) -> df
        for protease in proteases:
            for arm in arms:
                path = find_mztab(mab, protease, arm)
                if path is None:
                    continue
                df = _load_with_cat(path, ref_by_mab[mab])
                if df is not None:
                    all_dfs[(protease, arm)] = df

        for ci, region in enumerate(REGIONS):
            ax = axes[ri, ci]

            # Intersect spectra per protease so all arms see the same set,
            # then pool across proteases for this region.
            pooled = {arm: [] for arm in arms}
            for protease in proteases:
                arm_dfs = {arm: all_dfs[(protease, arm)]
                           for arm in arms if (protease, arm) in all_dfs}
                # Filter to region first
                arm_dfs = {arm: df[df["category"] == region]
                           for arm, df in arm_dfs.items()}
                arm_dfs = {arm: df for arm, df in arm_dfs.items() if len(df) > 0}
                if not arm_dfs:
                    continue
                shared = None
                for df in arm_dfs.values():
                    refs = set(df["spectra_ref"])
                    shared = refs if shared is None else shared & refs
                if not shared:
                    continue
                for arm, df in arm_dfs.items():
                    sub = df[df["spectra_ref"].isin(shared)].copy()
                    pooled[arm].append(sub)

            panel_n = 0
            any_plotted = False
            for arm in arms:
                if not pooled[arm]:
                    continue
                import pandas as _pd
                df = _pd.concat(pooled[arm], ignore_index=True)
                if panel_n == 0:
                    panel_n = len(df)
                result = _pc_from_df(df)
                if result is None:
                    continue
                cov, prec, acc, auc_val, n = result
                style = ARM_STYLE.get(arm, {"label": arm, "color": "gray"})
                lbl = _LABEL_OVERRIDES.get(arm, style["label"])
                color = _COLOR_OVERRIDES.get(arm, style["color"])
                pep_recall = float(prec[-1]) if len(prec) else 0.0
                ax.plot(cov, prec, lw=2.0, color=color,
                        label=f"{lbl} (AP={auc_val:.3f}, pep recall={pep_recall:.2f})",
                        alpha=0.9)
                any_plotted = True
                n_plotted += 1

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            title = f"{mab} ({spec['species']}) — {region}"
            if panel_n:
                title += f", n={panel_n:,}"
            ax.set_title(title, fontsize=11, color="black")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_linewidth(1.2)
            ax.spines["left"].set_linewidth(1.2)
            if any_plotted:
                ax.legend(loc="lower left", framealpha=0.7)
            else:
                ax.text(0.5, 0.5, "(no PSMs)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray",
                        fontstyle="italic")
            if ri == nrows - 1:
                ax.set_xlabel("Coverage")
            if ci == 0:
                ax.set_ylabel("Peptide precision")

    fig.suptitle("Per-antibody precision–coverage curves, by region "
                 "(pooled across proteases)",
                 fontsize=15, color="black", y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({n_plotted} curves)")
    plt.close(fig)


if __name__ == "__main__":
    main()
