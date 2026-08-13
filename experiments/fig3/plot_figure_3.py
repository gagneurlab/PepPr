#!/usr/bin/env python3
"""Figure 3 — Cross-species and out-of-distribution generalization of
Casanovo+PepPr.

Four panels:
  A. ProteomeTools SAAV (Mutations) precision-coverage curves — shows the
     human-isoform pepLM lifts Casanovo on single-amino-acid-variant peptides.
  B. Representative ProteomeTools SAAV example (KRAS G12D peptide
     LVVVGAGDVGK) — Casanovo swaps L↔V at positions 1-2; Casanovo+PepPr recovers.
  C. Cross-species scatter — human isoform pepLM used on every non-human 9-species
     dataset.  Tests transfer of the human prior to unrelated proteomes.
  D. KoL animal-proteome overlap vs Δ peptide-precision lift — uses both
     human_iso and mouse pepLMs against each KoL kingdom species; pooled linear
     fit shows lift correlates with proteome overlap.

Reads cached outputs from:
  * plot_scatter_precision.py  → scatter_precision_cache.csv (Panel C)
  * plot_kol_overlap_vs_pp_gain.py → kol_overlap_vs_pp_gain.csv (Panel D)
Panel A re-loads the mutations mztabs and re-computes PC inline.  Panel B
uses the spectrum + sequence-diff helpers from xa_novo/plot_figure_4_vc.py.

Usage:
    python experiments/fig3/plot_figure_3.py
"""
import os
import sys
import re
import glob
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.gridspec import GridSpec
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from sklearn.metrics import auc
from experiments.utils.evaluation import (
    _normalize_to_massivekb, evaluate, MASSIVEKB_MASSES, load_mztab_with_mgf,
    _parse_ms_run_locations, count_mgf_spectra,
    plot_sequence_diff, plot_spectrum_on_ax, paired_median_delta_stats,
    annotate_missing_ion_gap, annotate_precursor_readout,
)
from peptide_priors.const import (
    COLOR_CASANOVO as COLOR_DNPS,
    COLOR_DARK_GRAY,
    COLOR_GREEN,
    COLOR_MID_GRAY,
    COLOR_PP,
    COLOR_PURPLE,
    PROTEOMETOOLS_SAAV_DIR,
    PROJECT_ROOT,
    result_run_path,
)


SPECIES_LABELS = {
    "cowpea":      "ricebean",
    "archaeon":    "mmazei",
    "endoloripes": "clambacteria",
}
def _sp_label(s):
    return SPECIES_LABELS.get(s, s)


# ── Panel A: ProteomeTools SAAV PC curves ────────────────────
PANELA_NPZ = os.path.join(PROTEOMETOOLS_SAAV_DIR, "panelA_curves.npz")


def plot_panel_A(ax):
    """ProteomeTools SAAV PC curves on the GT-search 1%-FDR spectra. De novo (Casanovo,
    Casanovo+PepPr) vs the MSFragger DMO database search (site-localized, ranked by
    Percolator; localization = PTMProphet where present, else MSFragger
    localize_delta_mass). Curves are precomputed (build_panelA_data.py ->
    panelA_curves.npz); this is a thin renderer."""
    d = np.load(PANELA_NPZ)
    ap = lambda x, y: auc(x, y)
    ax.plot(d["pp_cov"], d["pp_prec"], color=COLOR_PP, lw=2.2,
            label=f"Casanovo + PepPr (AP={ap(d['pp_cov'], d['pp_prec']):.3f})")
    ax.plot(d["cas_cov"], d["cas_prec"], color=COLOR_DNPS, lw=2.2,
            label=f"Casanovo (AP={ap(d['cas_cov'], d['cas_prec']):.3f})")
    ax.plot(d["dmo_site_cov"], d["dmo_site_prec"], color=COLOR_GREEN, lw=2.2, ls="--",
            label=f"MSFragger DMO, site-localized (AP={ap(d['dmo_site_cov'], d['dmo_site_prec']):.3f})")
    if "fdr_cov" in d.files:                              # 1% PSM-FDR operating point
        cf, sf = float(d["fdr_cov"]), float(d["fdr_site_prec"])
        ax.plot([cf], [sf], "o", color=COLOR_GREEN, ms=8, mec="black", mew=1.0, zorder=6)
        ax.annotate("1% FDR", xy=(cf, sf), xytext=(cf + 0.03, sf - 0.11),
                    fontsize=8.5, color="black",
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage", fontsize=11)
    ax.set_ylabel("Peptide precision", fontsize=11)
    ax.set_title("A. Reference human PepPr on ProteomeTools variant data", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.85, handlelength=3.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


# ── Panel C: cross-species scatter (human_iso PLM, non-human data) ───────
def plot_panel_C(ax):
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "scatter_precision_cache.csv"))
    sc = df[df["condition"] == "cross"].copy()

    for _, row in sc.iterrows():
        ax.plot([row["casanovo"], row["casanovo"]],
                [row["casanovo"], row["casanovo_pp"]],
                color=COLOR_PP, lw=1.5, alpha=0.5, zorder=1)
    stats = paired_median_delta_stats(sc["casanovo"].to_numpy(),
                                       sc["casanovo_pp"].to_numpy())
    med, ci_lo, ci_hi = stats["median_delta"], stats["ci_low"], stats["ci_high"]
    ax.scatter(sc["casanovo"], sc["casanovo_pp"], s=50, zorder=3,
               marker="o", color=COLOR_PP, edgecolors="white", linewidths=0.4,
               label=(f"Casanovo + PepPr (median Δ = {med*100:+.1f} pp "
                      f"[{ci_lo*100:+.1f}, {ci_hi*100:+.1f}])"))

    glow = [patheffects.withStroke(linewidth=3.0, foreground="white")]
    try:
        from adjustText import adjust_text
        # clambacteria sits in the lower-left corner, right where the legend
        # box lands in the half-width layout — nudge its label up so
        # adjust_text starts clear of the legend instead of settling on it.
        _label_y_offset = {"endoloripes": 0.025}
        texts = [
            ax.text(row["casanovo"],
                    row["casanovo_pp"] + _label_y_offset.get(row["species"], 0.0),
                    f"{_sp_label(row['species'])}\n({(row['casanovo_pp']-row['casanovo'])*100:+.1f} pp)",
                    fontsize=9, color="black", path_effects=glow, zorder=5,
                    ha="center", va="center", linespacing=1.05)
            for _, row in sc.iterrows()
        ]
        adjust_text(texts, ax=ax, seed=42,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="gray",
                                    alpha=0.6, zorder=4),
                    force_points=(8, 8), force_text=(1.5, 1.5), min_arrow_len=5,
                    expand=(1.8, 1.8))
    except ImportError:
        for _, row in sc.iterrows():
            ax.text(row["casanovo"], row["casanovo_pp"] + 0.012,
                    _sp_label(row["species"]), fontsize=9, ha="center")

    all_vals = pd.concat([sc["casanovo"], sc["casanovo_pp"]])
    lo, hi = float(all_vals.min()) - 0.03, float(all_vals.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], ls="--", color=COLOR_DNPS, lw=1.5, alpha=1.0,
            zorder=0, label="Casanovo")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.set_xlabel("Casanovo peptide recall", fontsize=11)
    ax.set_ylabel("Casanovo + PepPr peptide recall", fontsize=11)
    ax.set_title("C. Human PepPr on non-human nine-species data",
                 fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.tick_params(labelsize=10)


# ── Panel D: KoL proteome overlap vs PP precision lift ───────────────────
_PLM_STYLE = {
    "human_iso": dict(color=COLOR_PURPLE, marker="o"),
    "mouse": dict(color=COLOR_GREEN, marker="^"),
}
_PLM_LEGEND = {
    "human_iso": "human iso prior",
    "mouse": "mouse prior",
}


def plot_panel_D(ax):
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "kol_overlap_vs_pp_gain.csv"))

    ax.axhline(0.0, color=COLOR_DARK_GRAY, lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Pooled linear fit over all (plm, kol) points, with 95% CI on the
    # mean-prediction line + scipy p-value for slope ≠ 0.
    x_all = df["pct_plm_in_kol"].to_numpy()
    y_all = df["delta_pep_precision"].to_numpy()
    if len(x_all) >= 3:
        from scipy import stats
        reg = stats.linregress(x_all, y_all)
        slope, intercept, r, pval = reg.slope, reg.intercept, reg.rvalue, reg.pvalue
        xs = np.linspace(x_all.min(), x_all.max(), 100)
        ys = slope * xs + intercept

        # 95% CI band around the mean prediction.
        n = len(x_all)
        resid = y_all - (slope * x_all + intercept)
        sigma = np.sqrt(np.sum(resid ** 2) / (n - 2))
        x_mean = x_all.mean()
        Sxx = np.sum((x_all - x_mean) ** 2)
        se_pred = sigma * np.sqrt(1.0 / n + (xs - x_mean) ** 2 / Sxx)
        tcrit = stats.t.ppf(0.975, df=n - 2)
        lo, hi = ys - tcrit * se_pred, ys + tcrit * se_pred
        ax.fill_between(xs, lo, hi, color=COLOR_MID_GRAY, alpha=0.18, zorder=1,
                        linewidth=0)

        # P-value formatting: "<0.001" if very small, else scientific or 3-decimal.
        if pval < 1e-3:
            pstr = "p<0.001"
        elif pval < 0.01:
            pstr = f"p={pval:.1e}"
        else:
            pstr = f"p={pval:.3f}"
        ax.plot(xs, ys, color=COLOR_MID_GRAY, lw=1.2, ls="-", alpha=0.95, zorder=2,
                label=f"Linear fit (n={n}, r={r:+.2f}, {pstr})")

    glow = [patheffects.withStroke(linewidth=3.0, foreground="white")]
    try:
        from adjustText import adjust_text
        _have_adjust = True
    except ImportError:
        _have_adjust = False

    all_texts = []
    for plm in ["human_iso", "mouse"]:
        sub = df[df["plm_species"] == plm]
        if sub.empty:
            continue
        style = _PLM_STYLE.get(plm, dict(color="black", marker="s"))
        ax.scatter(sub["pct_plm_in_kol"], sub["delta_pep_precision"],
                   s=110, edgecolors="white", linewidths=0.8, zorder=3,
                   label=_PLM_LEGEND.get(plm, f"{plm} prior"), **style)
        if _have_adjust:
            for xi, yi, name in zip(sub["pct_plm_in_kol"],
                                     sub["delta_pep_precision"],
                                     sub["kol_label"]):
                all_texts.append(
                    ax.text(xi, yi, name, fontsize=8, color="black",
                            path_effects=glow, zorder=5)
                )

    if _have_adjust and all_texts:
        from adjustText import adjust_text
        adjust_text(all_texts, ax=ax, seed=42,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="gray",
                                    alpha=0.6),
                    force_points=(8, 8), force_text=(1.5, 1.5), min_arrow_len=5,
                    expand=(1.8, 1.8))

    ax.set_xlabel("% of prior peptides in target proteome", fontsize=11)
    ax.set_ylabel("Δ peptide recall (pp)", fontsize=11)
    ax.set_title("D. Human/mouse PepPr on Kingdoms of Life animal data", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


# ── Panel B: representative ProteomeTools SAAV spectrum ─────────────────
EXAMPLE = dict(
    mgf=os.path.join(PROTEOMETOOLS_SAAV_DIR, "mgf_gt",
                     "03512a_BA1-TUM_mutations_1_01_01-3xHCD-1h-R1.mgf"),
    scan=25728,
    # KRAS G12D oncogenic mutation: wild-type "LVVVGAGGVGK" → mutant
    # "LVVVGAGDVGK" (G→D at the peptide's position 8 = KRAS residue 12).
    # Casanovo predicts the right G→D substitution but swaps L↔V at the
    # N-terminus; Casanovo+PepPr recovers the correct ordering.
    true_seq="LVVVGAGDVGK",
    v_pred  ="VLVVGAGDVGK", v_score=0.498,
    pp_pred ="LVVVGAGDVGK", pp_score=0.685,
    boundary_pos=None,
    precursor_mz=507.3039, precursor_charge=2,
    true_sa=0.6925,   # Prosit_2020_intensity_HCD SA (see experiments.utils.sa_prosit)
)


def plot_panel_B(ax_diff, ax_spec):
    plot_sequence_diff(
        ax_diff,
        true_seq=EXAMPLE["true_seq"],
        preds=[("Casanovo",      EXAMPLE["v_pred"],  EXAMPLE["v_score"]),
               ("Casanovo + PepPr", EXAMPLE["pp_pred"], EXAMPLE["pp_score"])],
        boundary_pos=EXAMPLE["boundary_pos"] or 0,
        highlight_positions=[8],          # G12D variant residue (peptide pos 8)
        highlight_label="G12D",
        true_sa=EXAMPLE.get("true_sa"),
    )
    if EXAMPLE["boundary_pos"] is None:
        for ln in list(ax_diff.lines):
            ln.remove()
        for txt in list(ax_diff.texts):
            if txt.get_text() == "V/C boundary":
                txt.remove()
    try:
        plot_spectrum_on_ax(ax_spec, mgf_path=EXAMPLE["mgf"],
                             scan=EXAMPLE["scan"], seq=EXAMPLE["true_seq"])
        # Casanovo swaps positions 1–2 (L↔V). Both b1 (114.09) and y10
        # (900.52) — the diagnostic ions for residue 2 — are missing.
        annotate_missing_ion_gap(ax_spec, EXAMPLE["true_seq"], 'b',
                                  missing_idx=1, observed_idx=2)
        annotate_missing_ion_gap(ax_spec, EXAMPLE["true_seq"], 'y',
                                  missing_idx=10, observed_idx=9)
        # Pedagogical readout: in a complete spectrum, [M+H]+ − y10 would
        # reveal the N-terminal L (gap = 113.08 Da on the singly-charged
        # m/z axis). Dashed to flag that it depends on the missing y10.
        annotate_precursor_readout(ax_spec, EXAMPLE["true_seq"],
                                   precursor_mz=EXAMPLE["precursor_mz"],
                                   precursor_charge=EXAMPLE["precursor_charge"],
                                   from_y_idx=10)
    except Exception as e:
        ax_spec.text(0.5, 0.5, f"(spectrum not available: {e})",
                     ha="center", va="center", transform=ax_spec.transAxes,
                     color="gray", fontsize=10)
        ax_spec.axis("off")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    out_path = os.path.join(PROJECT_ROOT, "figure_3.png")
    plt.style.use("ggplot")
    plt.rcParams.update({
        "axes.facecolor": "white", "axes.edgecolor": "black",
        "axes.linewidth": 0.8, "axes.grid": False,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "font.family": "sans-serif",
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })

    fig = plt.figure(figsize=(15.5, 13.5))
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs = GridSpec(2, 1, height_ratios=[1.4, 1.15], hspace=0.32, figure=fig)

    # Row 0 — Panel A (narrow) + Panel B: nested rows for sequence diff
    # (top) + spectrum (bottom).
    gsTop = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                    width_ratios=[1.25, 1.4], wspace=0.12)
    axA = fig.add_subplot(gsTop[0])
    gsB = GridSpecFromSubplotSpec(2, 1, subplot_spec=gsTop[1],
                                  height_ratios=[1.6, 3.0], hspace=0.15)
    axB_diff = fig.add_subplot(gsB[0])
    axB_spec = fig.add_subplot(gsB[1])

    # Row 1 — Panel C + Panel D side by side.
    gsBot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],
                                    width_ratios=[1, 1], wspace=0.25)
    axC = fig.add_subplot(gsBot[0])
    axD = fig.add_subplot(gsBot[1])

    print("[A] Mutations PC ...")
    plot_panel_A(axA)
    print("[B] SAAV worked-example spectrum ...")
    plot_panel_B(axB_diff, axB_spec)
    pos_b = gsTop[1].get_position(fig)
    fig.text(
        pos_b.x0 + pos_b.width / 2,
        pos_b.y1 + 0.006,
        "B. Example ProteomeTools variant (oncogenic mutation KRAS G12D)",
        ha="center", va="bottom", fontsize=12,
    )
    print("[C] cross-species scatter ...")
    plot_panel_C(axC)
    print("[D] KoL overlap vs PP gain ...")
    plot_panel_D(axD)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
