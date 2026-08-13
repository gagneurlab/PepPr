#!/usr/bin/env python3
"""Figure 4.

Main figure (paper Figure 4):
  figure_4.png                  — A: peptide recall per region category × arm;
                                  B: V<->C recall vs peptide length per protease;
                                  C: worked example (36H6 pepsin, scan 18064);
                                  D: deamidation calls on true N/Q[+0.984];
                                  E: single-mAb assembly contiguity

Usage:
    python experiments/fig4/plot_figure_4.py
"""
import argparse, csv, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from peptide_priors import const

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Region helpers remain in experiments/; Figure 4 helpers are package modules.

from experiments.utils.evaluation import (
    _normalize_to_massivekb, _expand_masses, evaluate, MASSIVEKB_MASSES,
    load_mztab_with_mgf, plot_sequence_diff, plot_spectrum_on_ax, _AA_RE,
    annotate_missing_ion_gap, _ion_mz, _B_ION_COLOR,
)
from peptide_priors.const import COLOR_LIGHT_GRAY, COLOR_PP, COLOR_RED
from experiments.fig4.references import (
    ARM_STYLE,
    MAB_SPECS,
    XANOVO_PROTEASES,
    categorize_gt,
    find_all_matches,
    find_mztab,
    read_fasta_with_regions,
)

# ── Deamidation panel helpers ─────────────────────────────────────────
_DEAM_MABS = ["2B4", "36H6", "85F7", "S2P6",                    # XA-Novo singles
        "IgG1_Human_H", "IgG1_Human_L", "Herceptin",      # Beslic
        "anti-FLAG-M2", "WIgG1_H", "WIgG1_L"]


_DEAM_TOKEN_RE = re.compile(r"([A-Z])((?:\[[^\]]+\])?(?:[+\-]\d+(?:\.\d+)?)?)")


def _canonical_deam_mod(mod):
    if not mod: return ""
    if "Deamidated" in mod or "+0.984" in mod: return "+0.984"
    if "Carbamidomethyl" in mod or "+57.021" in mod: return "+57.021"
    if "Oxidation" in mod or "+15.99" in mod: return "+15.995"
    if "Carbamyl" in mod or "+43.006" in mod: return "+43.006"
    if "Acetyl" in mod or "+42.01" in mod: return "+42.011"
    return mod


def _tokenize_deam(seq):
    if not isinstance(seq, str): return None
    s = re.sub(r"^[\[+\-][^A-Z]*?-", "", seq)
    s = re.sub(r"^[+\-]\d+(\.\d+)?", "", s)
    return [(m.group(1), _canonical_deam_mod(m.group(2))) for m in _DEAM_TOKEN_RE.finditer(s)]


_DEAM_ARMS = ["vanilla", "pp"]


_DEAM_ARM_LABEL = {"vanilla": "Casanovo",
             "pp": "Casanovo + PepPr"}


_DEAM_CATEGORIES = ["correct", "missed_mod", "mis_substituted", "other"]


_DEAM_CAT_COLOR = {
    "correct":         COLOR_PP,   # correct deam call (deep blue — PP color)
    "missed_mod":      "#9eb9c4",  # right AA, missed mod (light blue/grey)
    "mis_substituted": COLOR_RED,  # called as the mass-equivalent native AA
    "other":           "#b0b0b0",  # completely different (grey)
}


def _collect_deam_counts(mabs=_DEAM_MABS, verbose=False,
                        residue="N", mis_sub_aa="D"):
    """Per-arm counts of deam-call outcomes at ground-truth ``residue``[+0.984]
    positions. ``residue``/``mis_sub_aa`` are the deamidation pair — ("N","D")
    for Asn or ("Q","E") for Gln.
    Returns a DataFrame with one row per arm and the four CATEGORIES columns.
    """
    counts = {a: Counter() for a in _DEAM_ARMS}
    for mab in mabs:
        spec    = MAB_SPECS[mab]
        pp_arm  = spec["pp_arm"]
        van_arm = spec.get("vanilla_arm", "vanilla")
        proteases = spec.get("proteases", XANOVO_PROTEASES)
        for protease in proteases:
            paths = {
                "vanilla": find_mztab(mab, protease, van_arm),
                "pp":      find_mztab(mab, protease, pp_arm),
            }
            if not all(paths.values()):
                if verbose:
                    missing = [a for a, p in paths.items() if not p]
                    print(f"  skip {mab}/{protease} — missing {missing}")
                continue
            dfs = {a: load_mztab_with_mgf(p).dropna(subset=["true_seq"]).copy()
                   for a, p in paths.items()}
            # Inner-join vanilla + PP on spectra_ref so positions align
            merged = dfs["vanilla"][["spectra_ref", "true_seq", "pred"]].rename(
                columns={"pred": "vanilla_pred"})
            for a in ("pp",):
                merged = merged.merge(
                    dfs[a][["spectra_ref", "pred"]].rename(columns={"pred": f"{a}_pred"}),
                    on="spectra_ref")
            if verbose:
                print(f"  {mab}/{protease}: {len(merged):,} 2-arm-matched PSMs")
            for _, r in merged.iterrows():
                gt = _tokenize_deam(r["true_seq"])
                if not gt: continue
                preds = {a: _tokenize_deam(r[f"{a}_pred"]) for a in _DEAM_ARMS}
                if not all(p and len(p) == len(gt) for p in preds.values()):
                    continue
                for i, (g_aa, g_mod) in enumerate(gt):
                    if not (g_aa == residue and g_mod == "+0.984"):
                        continue
                    for a in _DEAM_ARMS:
                        pa, pm = preds[a][i]
                        if pa == residue and pm == "+0.984":  k = "correct"
                        elif pa == residue and pm == "":      k = "missed_mod"
                        elif pa == mis_sub_aa and pm == "":   k = "mis_substituted"
                        else:                                  k = "other"
                        counts[a][k] += 1
    rows = []
    for a in _DEAM_ARMS:
        rows.append({"arm": a, **{c: counts[a].get(c, 0) for c in _DEAM_CATEGORIES}})
    df = pd.DataFrame(rows)
    df["total"] = df[_DEAM_CATEGORIES].sum(axis=1)
    df.attrs["residue"]   = residue
    df.attrs["mis_sub_aa"] = mis_sub_aa
    return df


def _plot_deam_panel(ax, df=None, title=None, show_xlabel=True, show_legend=True,
               residue=None, mis_sub_aa=None, show_yticks=True):
    """Render one residue's deamidation panel onto ``ax``.

    ``residue`` / ``mis_sub_aa`` default to whatever the DataFrame was built
    with (read from ``df.attrs``); pass explicitly to override the display
    text. Display labels are formatted with the residue letter so the same
    code renders both the N→D and Q→E variants.
    """
    if df is None:
        df = _collect_deam_counts(residue="N", mis_sub_aa="D")
    if residue is None:
        residue = df.attrs.get("residue", "N")
    if mis_sub_aa is None:
        mis_sub_aa = df.attrs.get("mis_sub_aa", "D")
    cat_label = {
        "correct":         f"{residue}[+0.984] (correct)",
        "missed_mod":      f"{residue} (missed mod)",
        "mis_substituted": f"{mis_sub_aa} (mis-substituted, mass-eq)",
        "other":           "other AA",
    }
    if title is None:
        title = (f"Deamidation calls on true {residue}[+0.984] "
                 f"(→ {mis_sub_aa} when mis-substituted)")
    n = int(df["total"].iloc[0])
    arms = df["arm"].tolist()
    y_pos = np.arange(len(arms))
    left = np.zeros(len(arms))
    fracs = df[_DEAM_CATEGORIES].div(df[_DEAM_CATEGORIES].sum(axis=1), axis=0)
    for c in _DEAM_CATEGORIES:
        w = fracs[c].to_numpy() * 100   # to percent
        ax.barh(y_pos, w, left=left, color=_DEAM_CAT_COLOR[c],
                edgecolor="white", linewidth=0.5,
                label=cat_label[c])
        for i, (val, lf) in enumerate(zip(w, left)):
            if val >= 8:  # only label segments wide enough
                ax.text(lf + val / 2, y_pos[i], f"{val:.0f}%",
                        ha="center", va="center", fontsize=8.5,
                        color=("white" if c in ("correct", "mis_substituted")
                               else "black"),
                        fontweight="bold")
        left += w
    ax.set_yticks(y_pos)
    if show_yticks:
        ax.set_yticklabels([_DEAM_ARM_LABEL[a] for a in arms], fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    if show_xlabel:
        ax.set_xlabel(f"% of true-{residue}[+0.984] positions (n={n:,})",
                      fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title(title, fontsize=11)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32),
                  ncol=4, fontsize=8.5, frameon=False, handlelength=1.2,
                  columnspacing=1.4, handletextpad=0.5)
    return df


_ASSEMBLY_MABS = [
    ("WIgG1", "noplm", "antibody_mouse",
     [("HC", "WIgG1_H", "WIgG1_HeavyChain"),
      ("LC", "WIgG1_L", "WIgG1_LightChain")]),
    ("α-FLAG", "noplm", "antibody_mouse",
     [("HC", "anti-FLAG-M2", "HC"), ("LC", "anti-FLAG-M2", "LC")]),
    ("Herceptin", "noplm", "antibody_human",
     [("HC", "Herceptin", "HC"), ("LC", "Herceptin", "LC")]),
    ("IgG1", "noplm", "antibody_human",
     [("HC", "IgG1_Human_H", "HC"), ("LC", "IgG1_Human_L", "LC")]),
    ("36H6", "vanilla", "antibody_mouse",
     [("HC", "36H6", "HC"), ("LC", "36H6", "LC")]),
    ("2B4", "vanilla", "antibody_mouse",
     [("HC", "2B4", "HC"), ("LC", "2B4", "LC")]),
    ("85F7", "vanilla", "antibody_mouse",
     [("HC", "85F7", "HC"), ("LC", "85F7", "LC")]),
    ("S2P6", "vanilla", "antibody_human",
     [("HC", "S2P6", "HC"), ("LC", "S2P6", "LC")]),
]


def _load_assembly_rows():
    rows = []
    for path in const.FIGURE_4_ASSEMBLY_TSV_PATHS:
        with open(path) as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    return rows


def _assembly_value(rows, mab, arm, chain, column):
    for row in rows:
        if row["mab"] == mab and row["arm"] == arm and row["chain"] == chain:
            return float(row[column])
    return 0.0


def _plot_assembly_panel(ax, rows):
    groups = []
    for display, baseline_arm, pp_arm, chains in _ASSEMBLY_MABS:
        for chain_label, mab, chain in chains:
            groups.append(
                (f"{display}\n{chain_label}", mab, baseline_arm, pp_arm, chain)
            )

    baseline_lengths = []
    pp_lengths = []
    baseline_cdr = []
    pp_cdr = []
    reference_lengths = []
    for _, mab, baseline_arm, pp_arm, chain in groups:
        baseline_lengths.append(
            _assembly_value(rows, mab, baseline_arm, chain, "longest_aligned")
        )
        pp_lengths.append(
            _assembly_value(rows, mab, pp_arm, chain, "longest_aligned")
        )
        baseline_cdr.append(
            _assembly_value(rows, mab, baseline_arm, chain, "cdr_coverage")
        )
        pp_cdr.append(_assembly_value(rows, mab, pp_arm, chain, "cdr_coverage"))
        reference_lengths.append(
            _assembly_value(rows, mab, baseline_arm, chain, "ref_len")
        )

    x = np.arange(len(groups))
    bar_width = 0.46
    ax.bar(
        x - bar_width / 2,
        baseline_lengths,
        bar_width,
        color=ARM_SERIES_COLOR["vanilla"],
        edgecolor="black",
        linewidth=0.4,
        label="Casanovo",
    )
    ax.bar(
        x + bar_width / 2,
        pp_lengths,
        bar_width,
        color=ARM_SERIES_COLOR["pp"],
        edgecolor="black",
        linewidth=0.4,
        label="Casanovo + PepPr",
    )
    for index, reference_length in enumerate(reference_lengths):
        ax.plot(
            [index - bar_width, index + bar_width],
            [reference_length, reference_length],
            color="black",
            linestyle="--",
            linewidth=1.2,
            zorder=5,
        )
    ax.plot(
        [], [], color="black", linestyle="--", linewidth=1.2,
        label="Reference chain length"
    )

    ymax = max(reference_lengths + baseline_lengths + pp_lengths) * 1.18
    ax.set_ylim(0, ymax)
    for offset, lengths, recalls in (
        (-bar_width / 2, baseline_lengths, baseline_cdr),
        (bar_width / 2, pp_lengths, pp_cdr),
    ):
        for index, (length, recall) in enumerate(zip(lengths, recalls)):
            if not length:
                continue
            ax.text(
                index + offset, length + ymax * 0.015, f"{length:.0f}",
                ha="center", va="bottom", fontsize=8,
            )
            ax.text(
                index + offset, length / 2, f"{recall * 100:.0f}%",
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold",
            )

    labels = [
        f"{group[0]}\n({reference_length:.0f} aa)"
        for group, reference_length in zip(groups, reference_lengths)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Longest aligned contig (aa)")
    ax.set_title(
        "E. Single-mAb assembly contiguity "
        "(CDR recall inside bars)",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    for index in range(2, len(groups), 2):
        ax.axvline(index - 0.5, color=COLOR_LIGHT_GRAY, linewidth=0.6, zorder=0)
    ax.legend(
        loc="lower center", ncol=3, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.30),
    )


# All panels of figure 4 use the same mAb scope: 4 XA-Novo singles
# (2B4/36H6/85F7/S2P6) + 4 Beslic mAbs (IgG1_Human, Herceptin, anti-FLAG-M2,
# WIgG1). Beslic mAbs with separately-searched H and L data appear as two
# entries (…_H / …_L).
FIGURE_MABS_PSM = ["S2P6", "2B4", "85F7", "36H6",
                   "IgG1_Human_H", "IgG1_Human_L", "Herceptin",
                   "anti-FLAG-M2", "WIgG1_H", "WIgG1_L"]

# Ordered by descending Casanovo v5 precision so the bar chart reads as a
# monotone decline ending at V<->C (the failure case).
CATEGORIES = ["CDR-only", "FR-only", "C-only", "FR<->CDR", "V<->C"]
# XA-Novo dropped from the figure — keep only the two arms that compete on
# the final paper claim (Casanovo vs Casanovo+PepPr).
ARM_SERIES = ["vanilla", "pp"]
ARM_SERIES_LABEL = {
    "vanilla":   "Casanovo",
    "pp":        "Casanovo + PepPr",
}
ARM_SERIES_COLOR = {
    "vanilla":   ARM_STYLE["vanilla"]["color"],
    "pp":        ARM_STYLE["germline_human_sw"]["color"],
}

def _aa_len(seq):
    return len(_AA_RE.sub("", str(seq)))

def _load_arm_df(path, ref_chains):
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
    return df

def _pep_matches(df):
    if df is None or len(df) == 0:
        return np.array([], dtype=bool)
    masses = _expand_masses(df["pred"].tolist() + df["true_seq"].tolist(),
                            MASSIVEKB_MASSES)
    aa_matches = evaluate.aa_match_batch(df["true_seq"].tolist(),
                                          df["pred"].tolist(), masses)
    return np.asarray([m[1] for m in aa_matches[0]], dtype=bool)

def collect_psm_rows(mabs, ref_by_mab):
    """Return a long-form DataFrame with one row per scored PSM:
       columns: series, mab, protease, category, true_seq, pred, correct, true_len.
       Within each (mab, protease) panel, only spectra shared by all arms are kept.
    """
    rows = []
    for mab in mabs:
        spec = MAB_SPECS[mab]
        pp_arm = spec["pp_arm"]
        # Beslic mAbs use "noplm" as their vanilla arm name in the mztab
        # filename; XA-Novo mAbs use "vanilla".
        van_arm = spec.get("vanilla_arm", "vanilla")
        arm_to_series = {
            van_arm: "vanilla",
            pp_arm:  "pp",
        }
        arms = list(arm_to_series.keys())
        # Use each mAb's own protease set (XA-Novo vs Beslic differ).
        proteases = spec.get("proteases", XANOVO_PROTEASES)
        for protease in proteases:
            arm_dfs = {}
            for arm in arms:
                path = find_mztab(mab, protease, arm)
                if path is None:
                    print(f"  [{mab}/{protease}/{arm}] no mztab", file=sys.stderr)
                    continue
                df = _load_arm_df(path, ref_by_mab[mab])
                if df is not None:
                    arm_dfs[arm] = df

            shared = None
            for df in arm_dfs.values():
                refs = set(df["spectra_ref"])
                shared = refs if shared is None else shared & refs
            if not shared:
                continue

            for arm, df in arm_dfs.items():
                series = arm_to_series[arm]
                df_s = df[df["spectra_ref"].isin(shared)].copy()
                if len(df_s) == 0:
                    continue
                matches = _pep_matches(df_s)
                df_s["correct"] = matches
                df_s["series"] = series
                df_s["mab"] = mab
                df_s["protease"] = protease
                df_s["true_len"] = df_s["true_seq"].apply(_aa_len)
                rows.append(df_s[[
                    "series", "mab", "protease", "category",
                    "true_seq", "pred", "correct", "true_len",
                ]])
    if not rows:
        sys.exit("No PSMs collected — all mztabs missing?")
    return pd.concat(rows, ignore_index=True)

# plot_sequence_diff and plot_spectrum_on_ax now live in experiments.utils.evaluation
# — imported at the top of this file.

def _v_portion_for_row(row, ref_by_mab):
    """Number of V-region residues in this peptide (including the boundary K/R)."""
    chains = ref_by_mab[row["mab"]]
    plain = _AA_RE.sub("", str(row["true_seq"]))
    if len(plain) < 2:
        return np.nan
    for info in chains.values():
        matches = list(find_all_matches(plain, info["seq"]))
        if not matches:
            continue
        s, _ = matches[0]
        vc = info["vc"]
        if s <= vc <= s + len(plain) - 1:
            return vc - s + 1
        return np.nan
    return np.nan

# ── Plot 4: V<->C protease-driven mechanism figure ────────────────────────
def plot_all_panels(dat, ref_by_mab, out_path):
    """Five-panel mechanism figure built around the protease story:
        A. peptide recall per region category × arm (the puzzle)
        B. V↔C precision per protease, 2 arms — yield + length encoded in
           xticks (combines the old separate yield + length-cliff panels)
        C. worked-example sequence diff + annotated MS/MS spectrum
        D. deamidation correction on true N[+0.984]
        E. single-mAb assembly contiguity and CDR recall
    """
    vc = dat[dat["category"] == "V<->C"].copy()
    has_v_residues = vc.apply(lambda row: _v_portion_for_row(row, ref_by_mab), axis=1).notna()
    vc = vc[has_v_residues]

    # Four-row layout:
    #   Row 0: A region precision         | B V↔C precision per protease
    #   Row 1: C worked-example sequence diff + annotated spectrum
    #   Row 2: D deamidation (full width, short)
    #   Row 3: E single-mAb assembly contiguity
    fig = plt.figure(figsize=(15.5, 18))
    gs = GridSpec(4, 2, height_ratios=[1, 1.1, 0.45, 1],
                  width_ratios=[1.15, 1],
                  hspace=0.45, wspace=0.22, figure=fig)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    # Panel C: worked example (sequence diff + annotated spectrum), full width
    gsF = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :],
                                  height_ratios=[1.6, 3.0], hspace=0.40)
    axE_diff = fig.add_subplot(gsF[0])
    axE_spec = fig.add_subplot(gsF[1])
    # Panel D: deamidation correction — N (left) and Q (right) sub-axes.
    # Q-deam truth was added to the upstream FragPipe search (variable mod
    # `Deamidated (Q)`), then propagated into the annotated MGFs.
    gsD = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], wspace=0.18)
    axD_N = fig.add_subplot(gsD[0])
    axD_Q = fig.add_subplot(gsD[1])
    axE = fig.add_subplot(gs[3, :])

    # ── Panel A: final precision per region category × arm (the puzzle) ──
    n_cat, n_arm = len(CATEGORIES), len(ARM_SERIES)
    bar_w = 0.8 / n_arm
    xa = np.arange(n_cat)
    n_per_cat = []
    for cat in CATEGORIES:
        ns = [len(dat[(dat["series"] == s) & (dat["category"] == cat)])
              for s in ARM_SERIES]
        n_per_cat.append(max(ns) if ns else 0)
    from experiments.utils.evaluation import wilson_ci
    for si, series in enumerate(ARM_SERIES):
        precs, err_lo, err_hi = [], [], []
        for cat in CATEGORIES:
            sub = dat[(dat["series"] == series) & (dat["category"] == cat)]
            c, t = int(sub["correct"].sum()), len(sub)
            p, lo, hi = wilson_ci(c, t)
            precs.append(p if p == p else 0.0)
            err_lo.append((p - lo) if p == p else 0.0)
            err_hi.append((hi - p) if p == p else 0.0)
        bars = axA.bar(xa + (si - (n_arm - 1) / 2) * bar_w, precs,
                       width=bar_w, color=ARM_SERIES_COLOR[series],
                       label=ARM_SERIES_LABEL[series],
                       edgecolor="black", linewidth=0.4,
                       yerr=[err_lo, err_hi],
                       error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#333"))
        for b, p in zip(bars, precs):
            axA.text(b.get_x() + b.get_width() / 2, p + 0.04, f"{p:.2f}",
                     ha="center", va="bottom", fontsize=8, color="black")
    axA.set_xticks(xa)
    # Render the bidirectional arrow in CATEGORIES display labels with the
    # Unicode glyph (the internal CATEGORIES strings keep "<->" since they
    # are used as dict keys / category match values across multiple files).
    _disp = [c.replace("<->", "↔") for c in CATEGORIES]
    axA.set_xticklabels([f"{c}\n(n={n_per_cat[i]:,})"
                         for i, c in enumerate(_disp)], fontsize=9)
    axA.set_ylabel("Peptide recall")
    axA.set_ylim(0, 1.15)
    axA.set_title("A. Peptide recall per region category",
                  fontsize=11)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # ── Panel B: V↔C precision vs peptide length (bubble per protease, arm)
    # Each point is one (protease × arm) combo: x = median V↔C peptide length,
    # y = V↔C precision, bubble area ∝ # V↔C PSMs, color = arm. Lines connect
    # same-arm points across proteases to make the length-cliff story visible:
    # vanilla/XA-Novo precision collapses for long peptides (pepsin L=20),
    # PP holds ≥0.66 across the whole length range.
    vc_van = vc[vc["series"] == "vanilla"].copy()
    proteases_by_yield = (vc_van.groupby("protease").size()
                          .sort_values(ascending=False).index.tolist())
    med_len = {p: float(vc_van[vc_van["protease"] == p]["true_len"].median())
               for p in proteases_by_yield}
    n_psm   = {p: int(len(vc_van[vc_van["protease"] == p]))
               for p in proteases_by_yield}
    # x-positions sorted by median length so the cliff trend reads left-to-right
    proteases_by_length = sorted(proteases_by_yield, key=lambda p: med_len[p])
    # When multiple proteases share the same median length, nudge them apart so
    # bubbles and labels do not pile up. Anchor x = median length + offset.
    x_by_protease = {}
    same_len_groups = {}
    for p in proteases_by_length:
        same_len_groups.setdefault(med_len[p], []).append(p)
    for L, group in same_len_groups.items():
        if len(group) == 1:
            x_by_protease[group[0]] = L
        else:
            # Spread N proteases over a 0.8-wide span centred on L.
            offsets = np.linspace(-0.45, 0.45, len(group))
            for p, off in zip(group, offsets):
                x_by_protease[p] = L + off

    from experiments.utils.evaluation import wilson_ci
    # Bubble area scales sqrt with N so the visual area is roughly linear in N
    # without letting pepsin's PSMs dwarf trypsin's too aggressively.
    def _bubble_size(n):
        return 25 + n * 0.18

    for series in ARM_SERIES:
        xs, ys, ylo, yhi, sizes = [], [], [], [], []
        for p in proteases_by_length:
            sub = vc[(vc["series"] == series) & (vc["protease"] == p)]
            c, t = int(sub["correct"].sum()), len(sub)
            pp, lo, hi = wilson_ci(c, t)
            xs.append(x_by_protease[p]); ys.append(pp)
            ylo.append(pp - lo);   yhi.append(hi - pp)
            sizes.append(_bubble_size(t))
        # Light connecting line (helps eye follow per-arm trend across proteases)
        axB.plot(xs, ys, color=ARM_SERIES_COLOR[series], lw=1.6, alpha=0.45,
                 zorder=2)
        # Wilson 95% error bars (thin, behind the marker)
        axB.errorbar(xs, ys, yerr=[ylo, yhi], fmt="none",
                     ecolor=ARM_SERIES_COLOR[series], elinewidth=0.8,
                     capsize=2.5, capthick=0.8, alpha=0.7, zorder=2.5)
        # Bubbles
        axB.scatter(xs, ys, s=sizes, color=ARM_SERIES_COLOR[series],
                    edgecolor="black", linewidth=0.5,
                    label=ARM_SERIES_LABEL[series], zorder=3)

    # Annotate each protease (name + N) once, just below the lowest-arm
    # bubble at that x position. Per-protease x lookup so jittered positions
    # are honoured.
    y_min_at = {p: 1.0 for p in proteases_by_length}
    for series in ARM_SERIES:
        for p in proteases_by_length:
            sub = vc[(vc["series"] == series) & (vc["protease"] == p)]
            c, t = int(sub["correct"].sum()), len(sub)
            pp, _, _ = wilson_ci(c, t)
            y_min_at[p] = min(y_min_at[p], pp if pp == pp else 1.0)
    # Place labels below each protease's lowest bubble, then let adjustText
    # nudge them apart to eliminate collisions in dense clusters.  Draw a
    # leader line so the reader can associate a moved label back to its point.
    _labels = []
    _pts_x, _pts_y = [], []
    for p in proteases_by_length:
        _labels.append(axB.text(x_by_protease[p], y_min_at[p] - 0.06,
                                f"{p}\nn={n_psm[p]:,}",
                                ha="center", va="top", fontsize=7.5, color="#333"))
        _pts_x.append(x_by_protease[p]); _pts_y.append(y_min_at[p])
    try:
        from adjustText import adjust_text
        adjust_text(_labels, x=_pts_x, y=_pts_y, ax=axB,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#888",
                                    alpha=0.7, zorder=1),
                    expand_text=(1.3, 1.8), expand_points=(1.5, 2.0),
                    force_text=(0.8, 2.0), force_points=(0.6, 1.6),
                    lim=500)
    except ImportError:
        pass

    axB.set_xlabel("Median V↔C peptide length (aa)")
    axB.set_ylabel("V↔C peptide recall")
    axB.set_ylim(-0.28, 1.05)
    axB.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Pad x-axis so the leftmost / rightmost annotations don't clip
    xs_all = list(x_by_protease.values())
    axB.set_xlim(min(xs_all) - 1.5, max(xs_all) + 1.5)
    axB.set_title("B. V↔C recall vs peptide length per protease "
                  "(bubble area ∝ # V↔C PSMs)", fontsize=11)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.grid(axis="y", alpha=0.25, linewidth=0.4)
    axB.set_axisbelow(True)
    axB.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # ── Panel E: single-mAb assembly contiguity ──────────────────────────
    _plot_assembly_panel(axE, _load_assembly_rows())

    # ── Panel C: worked example — 36H6 pepsin scan 18064 (v=4, L=21)
    # Long V↔C peptide (21 aa, representative of the modal length where v5
    # collapses per Panel C).  Vanilla rearranges positions 2-5 across the
    # V/C boundary: L↔K↔A 3-cycle, R stays put, mass preserved.  PP correct.
    # Displayed with "…" truncation so the boundary motif is visible at the
    # N-terminus and the C-terminal anchor (EQL) at the C-terminus.
    EXAMPLE = dict(
        mab="36H6", protease="pepsin", scan=18064,
        mgf=const.FIGURE_4_EXAMPLE_MGF,
        true_seq="ELKRADAAPTVSIFPPSSEQL",
        v5_pred  ="EKARLDAAPTVSLFPPSSEQL", v5_score=0.11,
        pp_pred  ="ELKRADAAPTVSLFPPSSEQL", pp_score=0.56,
        boundary_pos=4,
        true_sa=0.8081,   # Prosit_2020_intensity_HCD SA (see experiments.utils.sa_prosit)
    )

    # ── Panel D: deamidation correction (N→D on left, Q→E on right) ───────
    print("\nCollecting deamidation counts — N:")
    _deam_N = _collect_deam_counts(mabs=FIGURE_MABS_PSM, residue="N", mis_sub_aa="D")
    print(_deam_N.to_string(index=False))
    print("\nCollecting deamidation counts — Q:")
    _deam_Q = _collect_deam_counts(mabs=FIGURE_MABS_PSM, residue="Q", mis_sub_aa="E")
    print(_deam_Q.to_string(index=False))
    _plot_deam_panel(axD_N, df=_deam_N, show_legend=True,
                     title="D. Deamidation calls on true N[+0.984]")
    _plot_deam_panel(axD_Q, df=_deam_Q, show_legend=True,
                     show_yticks=False,
                     title="    Deamidation calls on true Q[+0.984]")

    def _trunc(s, head=8, tail=3, ellipsis="…"):
        return s if len(s) <= head + 1 + tail else s[:head] + ellipsis + s[-tail:]

    plot_sequence_diff(
        axE_diff,
        true_seq=_trunc(EXAMPLE["true_seq"]),
        preds=[("Casanovo",      _trunc(EXAMPLE["v5_pred"]), EXAMPLE["v5_score"]),
               ("Casanovo + PepPr", _trunc(EXAMPLE["pp_pred"]), EXAMPLE["pp_score"])],
        boundary_pos=EXAMPLE["boundary_pos"],
        title=f"C. 36H6 pepsin, scan {EXAMPLE['scan']}, L={len(EXAMPLE['true_seq'])}",
        true_sa=EXAMPLE.get("true_sa"),
    )
    try:
        plot_spectrum_on_ax(axE_spec,
                             mgf_path=EXAMPLE["mgf"],
                             scan=EXAMPLE["scan"],
                             seq=EXAMPLE["true_seq"])
        # V↔C boundary error: position 5 (true A vs Casanovo L) is genuinely
        # ambiguous — b5 is mass-degenerate for either ordering and the y-series
        # has a large missing window (y11–y18 all absent in both 1+ and 2+
        # charge). Mark y17 and the adjacent y18 as missing so the figure shows
        # that y16/y17/y18 are all absent.
        annotate_missing_ion_gap(axE_spec, EXAMPLE["true_seq"], 'y',
                                  missing_idx=17, observed_idx=16)
        annotate_missing_ion_gap(axE_spec, EXAMPLE["true_seq"], 'y',
                                  missing_idx=18, observed_idx=17)
        # b3 is absent in both 1+ (371.22) and 2+ (186.11), and its complement
        # y18 is missing too — so while b2/b4 pin the {K,R} composition at
        # positions 3-4, no observed ion (in either charge state) resolves the
        # K-vs-R order. Mark b3 as missing. Drawn after the y-annotations so the
        # xlim expansion is reflected in the label nudge.
        residues = re.findall(r"[A-Z]", EXAMPLE["true_seq"])
        b3 = _ion_mz(residues, 'b', 3)
        axE_spec.axvline(b3, color=_B_ION_COLOR, ls="--", lw=1.0, alpha=0.75,
                          zorder=2)
        xlo, xhi = axE_spec.get_xlim()
        axE_spec.text(b3 + (xhi - xlo) * 0.006, 0.97, "b₃ (missing)",
                       ha="left", va="top",
                       fontsize=8, color=_B_ION_COLOR, alpha=0.9,
                       fontstyle="italic")
    except Exception as e:
        print(f"  [Panel C spectrum skipped: {e}]")
        axE_spec.text(0.5, 0.5, f"(spectrum not available: {e})",
                      ha="center", va="center", transform=axE_spec.transAxes,
                      color="gray", fontsize=10)
        axE_spec.axis("off")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out",
                   default=const.FIGURE_4_PATH)
    p.add_argument("--mabs", nargs="*", default=None)
    args = p.parse_args()

    mabs = args.mabs if args.mabs else FIGURE_MABS_PSM

    ref_by_mab = {}
    for mab in mabs:
        chains = read_fasta_with_regions(MAB_SPECS[mab]["ref"])
        chains = {k: v for k, v in chains.items()
                  if v["cdrs"] is not None and v["vc"] is not None}
        if not chains:
            sys.exit(f"{mab}: no annotated chains in {MAB_SPECS[mab]['ref']}")
        ref_by_mab[mab] = chains

    dat = collect_psm_rows(mabs, ref_by_mab)
    print(f"\nCollected {len(dat):,} (series, PSM) rows "
          f"across {dat['mab'].nunique()} mAbs × {dat['protease'].nunique()} proteases")

    plt.style.use("ggplot")
    plt.rcParams.update({
        "axes.facecolor": "white", "axes.edgecolor": "black",
        "axes.linewidth": 0.8, "axes.grid": False,
        "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "font.family": "sans-serif",
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })

    plot_all_panels(dat, ref_by_mab, args.out)

if __name__ == "__main__":
    main()
