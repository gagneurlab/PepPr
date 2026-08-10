#!/usr/bin/env python3
"""Supplementary Figure 2 — Peptide precision by spectral angle, per species.

Hypothesis: species with lower spectral angle (SA) between the observed
spectrum and the theoretical ions of the true peptide benefit more from
the Casanovo+PepPr sequence prior — the spectrum alone provides weaker
evidence.

Reads the asymbnln MassiveKB fusion-head outputs (`9s_<sp>_asymbnln.mztab`)
for the +PP arm.  The script still produces several diagnostic plots as
side effects; the canonical supp-fig-2 output is ``supp_fig_2.png``
(precision-by-SA grid, matching the layout of supp_fig_1).

Usage:
    python -m dnps_hybrid.plot_supp_fig_2
"""

import os
import re
import sys
import pickle
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from dnps_hybrid.metrics import (
    MASSIVEKB_MASSES, _normalize_to_massivekb, load_mztab_with_mgf,
    NINE_SPECIES_ORDER, species_label,
    NINE_SPECIES_NCOLS, NINE_SPECIES_PANEL_INCHES,
    NINE_SPECIES_LINE_WIDTH, NINE_SPECIES_MARKER_SIZE,
    NINE_SPECIES_TICK_FONTSIZE, NINE_SPECIES_LABEL_FONTSIZE,
    NINE_SPECIES_TITLE_FONTSIZE,
    NINE_SPECIES_COLOR_DNPS, NINE_SPECIES_COLOR_PP,
)
from dnps_hybrid.const import (
    COLOR_CASANOVO,
    COLOR_HISTOGRAM_GRAY,
    COLOR_PEACH,
    COLOR_PP,
    COLOR_TEXT_GRAY,
    COLOR_XANOVO,
    PROJECT_ROOT,
    result_run_path,
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "casanovo"))
from casanovo.denovo import evaluate

plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.grid": False,
})

# ── Constants ─────────────────────────────────────────────────────────────────
SA_CACHE = os.path.join(PROJECT_ROOT, "sa_correction_cache_prosit.pkl")
EXISTING_CACHE_CSV = os.path.join(PROJECT_ROOT, "scatter_precision_cache.csv")

SPECIES_DIRS = {
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

# Historical MAX_PSMS cap removed — with batched Prosit predictions (cached to
# disk) the per-species SA compute is fast enough to use every merged PSM.

H2O = 18.010564684
H   =  1.007276
MATCH_TOL_DA = 0.5   # fragment ion matching tolerance (Da)

COLOR_CORRECTED    = COLOR_PP
COLOR_STABLE_CORR  = COLOR_XANOVO
COLOR_STABLE_WRONG = "#e07050"
COLOR_WORSENED     = COLOR_CASANOVO

STATUS_COLORS = {
    "corrected":      COLOR_CORRECTED,
    "stable_correct": COLOR_STABLE_CORR,
    "stable_wrong":   COLOR_STABLE_WRONG,
    "worsened":       COLOR_WORSENED,
}
STATUS_LABELS = {
    "corrected":      "Corrected (cas wrong → PepPr right)",
    "stable_correct": "Both correct",
    "stable_wrong":   "Both wrong",
    "worsened":       "Worsened (cas right → PepPr wrong)",
}

# Shared with the other three nine-species supp figs (see metrics.py).
SPECIES_ORDER = NINE_SPECIES_ORDER


# ── 1. Extended MGF parser (returns peaks alongside title / SEQ) ──────────
def parse_mgf_with_peaks(mgf_path):
    """Return a list of dicts: {title, seq, charge, mz, intensity}.

    ``charge`` is the precursor charge parsed from ``CHARGE=N+`` (None if
    absent). ``mz`` / ``intensity`` are numpy arrays sorted by m/z.
    """
    spectra = []
    cur = None
    with open(mgf_path) as fh:
        for line in fh:
            line = line.strip()
            if line == "BEGIN IONS":
                cur = {"title": None, "seq": None, "charge": None,
                       "mz": [], "intensity": []}
            elif line == "END IONS":
                if cur is not None:
                    mz = np.array(cur["mz"], dtype=np.float32)
                    intensity = np.array(cur["intensity"], dtype=np.float32)
                    order = np.argsort(mz)
                    spectra.append({
                        "title":     cur["title"],
                        "seq":       cur["seq"],
                        "charge":    cur["charge"],
                        "mz":        mz[order],
                        "intensity": intensity[order],
                    })
                cur = None
            elif cur is not None:
                if line.startswith("TITLE="):
                    cur["title"] = line[6:]
                elif line.startswith("SEQ="):
                    cur["seq"] = line[4:]
                elif line.startswith("CHARGE="):
                    try:
                        cur["charge"] = int(line[7:].rstrip("+"))
                    except ValueError:
                        pass
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            cur["mz"].append(float(parts[0]))
                            cur["intensity"].append(float(parts[1]))
                        except ValueError:
                            pass
    return spectra


def parse_ms_run_locations(mztab_path):
    """Extract ms_run[N] → file path mapping from mztab metadata."""
    locations = {}
    with open(mztab_path) as fh:
        for line in fh:
            if line.startswith("MTD\tms_run["):
                parts = line.strip().split("\t")
                m = re.match(r"ms_run\[(\d+)\]-location", parts[1])
                if m:
                    locations[int(m.group(1))] = parts[2].replace("file://", "")
            elif line.startswith("PSH\t"):
                break
    return locations


# ── 2. Fragment mass calculator (spectrum_utils b/y generator) ───────────
# MassIVE-KB uses inline offsets like "C+57.021" or "+42.011-M+15.995AC…".
# spectrum_utils speaks ProForma, which brackets the offset ("C[+57.021]",
# "[+42.011]-M[+15.995]AC…"), so we convert token-by-token before handing
# the sequence to sfa.get_theoretical_fragments.
import spectrum_utils.fragment_annotation as _sfa
import spectrum_utils.proforma as _pf

_TOKEN_RE = re.compile(r"(?<=.)(?=[A-Z])")
_NTERM_MOD_RE = re.compile(r"^([+\-]\d+(?:\.\d+)?)-(.*)")
_RESIDUE_MOD_RE = re.compile(r"^([A-Z])([+\-]\d+(?:\.\d+)?)?$")


def _tokenize(seq):
    return _TOKEN_RE.split(seq)


def _massivekb_to_proforma(seq):
    m = _NTERM_MOD_RE.match(seq)
    if m:
        head, body = f"[{m.group(1)}]-", m.group(2)
    else:
        head, body = "", seq

    def _wrap(tok):
        m = _RESIDUE_MOD_RE.match(tok)
        if not m:
            return tok
        aa, mod = m.group(1), m.group(2)
        return aa if not mod else f"{aa}[{mod}]"

    return head + "".join(_wrap(t) for t in _tokenize(body))


def theoretical_ions(seq, max_charge=2):
    """Compute b/y ion m/z values for a MassIVE-KB format sequence.

    Delegates to spectrum_utils.fragment_annotation.get_theoretical_fragments
    after translating the MassIVE-KB mod syntax to ProForma. Returns a sorted
    numpy array of theoretical m/z values across singly- and doubly-charged
    b/y ions (max_charge default 2).
    """
    try:
        proteoform = _pf.parse(_massivekb_to_proforma(seq))[0]
    except Exception:
        return np.empty(0, dtype=np.float32)
    frags = _sfa.get_theoretical_fragments(
        proteoform, ion_types="by", max_charge=max_charge,
    )
    if not frags:
        return np.empty(0, dtype=np.float32)
    return np.sort(np.array([mz for _, mz in frags], dtype=np.float32))


# ── 3. Spectral angle computation ────────────────────────────────────────
def spectral_angle(mz_obs, int_obs, theo_mz, tol=MATCH_TOL_DA):
    """Normalized spectral angle between observed and theoretical ions.

    SA = 1 - (2/π) * arccos(cosine similarity of sqrt-intensity vectors).
    SA=1 → perfect match; SA=0 → orthogonal (no matching ions).

    For each theoretical ion we record the best-matching observed intensity
    (or 0 if none within tol), so unmatched ions penalise the score.
    Theoretical intensities are all equal (1.0 = unit height).
    """
    if len(mz_obs) == 0 or len(theo_mz) == 0:
        return np.nan

    int_obs = int_obs.astype(np.float64)
    mz_obs_list = mz_obs.tolist()

    # obs_matched[i] = observed intensity at theo_mz[i], or 0 if unmatched
    obs_matched = np.zeros(len(theo_mz), dtype=np.float64)
    for i, tmz in enumerate(theo_mz):
        idx = bisect_left(mz_obs_list, tmz - tol)
        best_dist, best_int = tol + 1, 0.0
        while idx < len(mz_obs_list) and mz_obs_list[idx] <= tmz + tol:
            d = abs(mz_obs_list[idx] - tmz)
            if d < best_dist:
                best_dist = d
                best_int = int_obs[idx]
            idx += 1
        if best_dist <= tol:
            obs_matched[i] = best_int

    # Theoretical vector: unit height for each ion
    theo_vec = np.ones(len(theo_mz), dtype=np.float64)
    obs_vec  = np.sqrt(obs_matched)

    norm_o = np.linalg.norm(obs_vec)
    norm_t = np.linalg.norm(theo_vec)
    if norm_o == 0:
        return 0.0

    cosine = np.dot(obs_vec, theo_vec) / (norm_o * norm_t)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return 1.0 - 2.0 * np.arccos(cosine) / np.pi


# ── 4 & 5. Load species data, compute SA, classify PSMs, cache ──────────
_SPECTRA_REF_RE = re.compile(r"ms_run\[(\d+)\]:index=(\d+)")


def _pp_result_paths(species, base):
    if species == "yeast":
        return sorted(
            str(path) for path in Path(base).glob("9s_yeast_hybrid_*.mztab")
        )
    if species == "human":
        return [os.path.join(base, "9s_human_plmhuman_iso_hybrid.mztab")]
    return [os.path.join(base, f"9s_{species}_asymbnln.mztab")]


def _load_result_paths(paths):
    frames = []
    for path in paths:
        df = load_mztab_with_mgf(path)
        locations = parse_ms_run_locations(path)

        def _spectrum_key(ref):
            match = _SPECTRA_REF_RE.match(ref or "")
            if match is None:
                return None
            run_num, index = int(match.group(1)), int(match.group(2))
            return os.path.basename(locations[run_num]), index

        df["spectrum_key"] = df["spectra_ref"].apply(_spectrum_key)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_species(species, dirname, rng):
    """Load dnps + asymbnln mztabs, compute SA, classify PSMs.

    Returns a DataFrame with columns:
      spectra_ref, true_seq, sa, status, cas_ok, pp_ok
    where status ∈ {corrected, stable_correct, stable_wrong, worsened}.
    """
    base = result_run_path(dirname)
    dnps_path = os.path.join(base, f"9s_{species}_dnps.mztab")
    pp_paths = _pp_result_paths(species, base)
    if not os.path.exists(dnps_path) or not pp_paths or not all(
        os.path.exists(path) for path in pp_paths
    ):
        print(f"  Skipping {species}: missing mztab(s).", file=sys.stderr)
        return None

    print(f"  Loading {species} mztabs…", file=sys.stderr)
    dnps_df = _load_result_paths([dnps_path])
    pp_df = _load_result_paths(pp_paths)

    for df in (dnps_df, pp_df):
        df["pred"]     = df["pred"].apply(_normalize_to_massivekb)
        df["true_seq"] = df["true_seq"].apply(_normalize_to_massivekb)

    merged = pp_df.merge(dnps_df, on="spectrum_key", suffixes=("_pp", "_dnps"))
    merged = merged[merged["true_seq_pp"] == merged["true_seq_dnps"]].copy()
    merged = merged.rename(columns={"true_seq_pp": "true_seq"})
    merged["spectra_ref"] = merged["spectra_ref_dnps"]
    merged = merged.dropna(subset=["true_seq"])

    print(f"  {species}: {len(merged):,} matched PSMs", file=sys.stderr)
    merged = merged.reset_index(drop=True)

    # Classify correctness
    truths    = merged["true_seq"].tolist()
    cas_preds = merged["pred_dnps"].tolist()
    pp_preds  = merged["pred_pp"].tolist()
    cas_ok = np.array(
        [m[1] for m in evaluate.aa_match_batch(truths, cas_preds, MASSIVEKB_MASSES)[0]]
    )
    pp_ok = np.array(
        [m[1] for m in evaluate.aa_match_batch(truths, pp_preds,  MASSIVEKB_MASSES)[0]]
    )

    status = np.where(
        ~cas_ok &  pp_ok,  "corrected",
        np.where(
         cas_ok &  pp_ok,  "stable_correct",
        np.where(
        ~cas_ok & ~pp_ok,  "stable_wrong",
                           "worsened"
        ))
    )

    # Load MGF peak data
    print(f"  Loading MGF peaks for {species}…", file=sys.stderr)
    ms_run_locs = parse_ms_run_locations(dnps_path)
    mgf_peak_cache = {}
    for run_num, mgf_path_loc in ms_run_locs.items():
        if mgf_path_loc not in mgf_peak_cache:
            print(f"    parsing {os.path.basename(mgf_path_loc)}", file=sys.stderr)
            mgf_peak_cache[mgf_path_loc] = parse_mgf_with_peaks(mgf_path_loc)
        mgf_peak_cache[run_num] = mgf_peak_cache[mgf_path_loc]

    # Compute SA for each PSM via Prosit_2020_intensity_HCD (batched, cached).
    # Peptides carrying mods Prosit doesn't recognise (deamidation, TMT, …)
    # return NaN and are excluded downstream.
    print(f"  Computing Prosit SA for {len(merged):,} PSMs…", file=sys.stderr)
    from dnps_hybrid.sa_prosit import compute_sa_for_psms

    peptides = merged["true_seq"].tolist()
    charges, mz_obs_list, int_obs_list = [], [], []
    for ref in merged["spectra_ref"]:
        m = _SPECTRA_REF_RE.match(ref or "")
        if m is None:
            charges.append(None); mz_obs_list.append(None); int_obs_list.append(None)
            continue
        run_num, idx = int(m.group(1)), int(m.group(2))
        spectra = mgf_peak_cache.get(run_num, [])
        if idx >= len(spectra):
            charges.append(None); mz_obs_list.append(None); int_obs_list.append(None)
            continue
        spec = spectra[idx]
        charges.append(spec.get("charge"))
        mz_obs_list.append(spec["mz"])
        int_obs_list.append(spec["intensity"])
    sa_vals = compute_sa_for_psms(peptides, charges, mz_obs_list, int_obs_list)
    sa_vals = sa_vals.astype(np.float32)

    result = pd.DataFrame({
        "spectra_ref": merged["spectra_ref"].values,
        "true_seq":    merged["true_seq"].values,
        "sa":          sa_vals,
        "status":      status,
        "cas_ok":      cas_ok,
        "pp_ok":       pp_ok,
    })
    result.attrs["source_files"] = tuple(pp_paths)
    return result


# ── Plot 1: Split-violin per species ──────────────────────────────────────
# Each species has two split violins:
#   pos 0 – Casanovo:    left half = correct, right half = wrong
#   pos 1 – Casanovo+PepPr: left half = correct, right half = wrong
# Left (correct) is lighter, right (wrong) is darker within each model color.

COLOR_CAS_RIGHT  = COLOR_PEACH     # light orange – Casanovo correct
COLOR_CAS_WRONG  = COLOR_CASANOVO  # dark orange – Casanovo wrong
COLOR_PP_RIGHT   = COLOR_XANOVO    # light blue – PP correct
COLOR_PP_WRONG   = COLOR_PP        # dark blue – PP wrong

from scipy.stats import gaussian_kde


def _half_violin(ax, data, pos, side, color, half_width=0.38, bw=0.05):
    """Draw one half of a violin at x=pos.

    side: 'left'  → density extends leftward  (x decreases)
          'right' → density extends rightward (x increases)
    """
    if len(data) < 5:
        return
    y_min, y_max = data.min(), data.max()
    if y_min == y_max:
        return
    y_pts = np.linspace(y_min, y_max, 300)
    kde = gaussian_kde(data, bw_method=bw)
    density = kde(y_pts)
    density = density / density.max() * half_width   # scale to half_width

    if side == "left":
        x_edge = pos - density
        ax.fill_betweenx(y_pts, pos, x_edge, color=color, alpha=0.80, lw=0)
        ax.plot(x_edge, y_pts, color="black", lw=0.6)
    else:
        x_edge = pos + density
        ax.fill_betweenx(y_pts, pos, x_edge, color=color, alpha=0.80, lw=0)
        ax.plot(x_edge, y_pts, color="black", lw=0.6)

    # Median tick
    med = float(np.median(data))
    med_hw = float(kde(med)[0]) / density.max() * half_width
    if side == "left":
        ax.plot([pos - med_hw, pos], [med, med], color="black", lw=1.8, solid_capstyle="round")
    else:
        ax.plot([pos, pos + med_hw], [med, med], color="black", lw=1.8, solid_capstyle="round")


def plot_violin(all_df, sa_data, out_path):
    species_with_data = [sp for sp in SPECIES_ORDER if sp in sa_data
                         and len(sa_data[sp].dropna(subset=["sa"])) > 0]
    n_sp = len(species_with_data)
    fig, axes = plt.subplots(1, n_sp, figsize=(2.4 * n_sp, 4.8), sharey=True)
    if n_sp == 1:
        axes = [axes]

    for ax, sp in zip(axes, species_with_data):
        df_sp = all_df[all_df["species"] == sp].dropna(subset=["sa"])
        cas_right = df_sp[df_sp["cas_ok"] == True]["sa"].values
        cas_wrong = df_sp[df_sp["cas_ok"] == False]["sa"].values
        pp_right  = df_sp[df_sp["pp_ok"]  == True]["sa"].values
        pp_wrong  = df_sp[df_sp["pp_ok"]  == False]["sa"].values

        # pos 0: Casanovo — left=correct, right=wrong
        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        _half_violin(ax, cas_right, pos=0, side="left",  color=COLOR_CAS_RIGHT)
        _half_violin(ax, cas_wrong, pos=0, side="right", color=COLOR_CAS_WRONG)

        # pos 1: Casanovo+PepPr — left=correct, right=wrong
        ax.axvline(1, color="gray", lw=0.5, ls="--", alpha=0.4)
        _half_violin(ax, pp_right, pos=1, side="left",  color=COLOR_PP_RIGHT)
        _half_violin(ax, pp_wrong, pos=1, side="right", color=COLOR_PP_WRONG)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Casanovo", "Casanovo\n+PepPr"], fontsize=8)
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(species_label(sp), fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)

        # n annotations
        for xpos, n in [(0, len(df_sp)), (1, len(df_sp))]:
            ax.text(xpos, 0.06, f"n={n:,}", ha="center", va="bottom",
                    fontsize=5, color="gray")

    axes[0].set_ylabel("Spectral Angle (true peptide)", fontsize=11)
    axes[0].set_ylim(0.05, 0.70)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_CAS_RIGHT, alpha=0.8, edgecolor="black", lw=0.5,
              label="Casanovo correct"),
        Patch(facecolor=COLOR_CAS_WRONG, alpha=0.8, edgecolor="black", lw=0.5,
              label="Casanovo wrong"),
        Patch(facecolor=COLOR_PP_RIGHT,  alpha=0.8, edgecolor="black", lw=0.5,
              label="Casanovo+PepPr correct"),
        Patch(facecolor=COLOR_PP_WRONG,  alpha=0.8, edgecolor="black", lw=0.5,
              label="Casanovo+PepPr wrong"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Spectral Angle: correct (left) vs wrong (right) for each model",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Plot 2: Scatter — median SA (cas-wrong PSMs) vs Δprecision ───────────
def plot_sa_vs_delta(all_df, prec_df, out_path):
    if prec_df is None:
        print("No precision cache CSV found; skipping scatter.", file=sys.stderr)
        return

    rows = []
    for sp, grp in all_df.groupby("species"):
        wrong = grp[grp["cas_ok"] == False].dropna(subset=["sa"])
        if len(wrong) == 0:
            continue
        row = prec_df[prec_df["species"] == sp]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        rows.append({
            "species":    sp,
            "median_sa":  float(wrong["sa"].median()),
            "delta_prec": float(row["casanovo_pp"] - row["casanovo"]),
            "rel_change": float(row["rel_change"]),
            "n_wrong":    len(wrong),
        })
    sc = pd.DataFrame(rows)
    if sc.empty:
        print("No scatter data; skipping.", file=sys.stderr)
        return

    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    glow = [patheffects.withStroke(linewidth=3.0, foreground="white")]

    ax.scatter(sc["median_sa"], sc["delta_prec"], s=80, color=COLOR_PP,
               edgecolors="white", linewidths=0.5, zorder=3)

    if len(sc) >= 3:
        coeffs = np.polyfit(sc["median_sa"], sc["delta_prec"], 1)
        x_line = np.linspace(sc["median_sa"].min() - 0.02,
                             sc["median_sa"].max() + 0.02, 100)
        ax.plot(x_line, np.polyval(coeffs, x_line),
                ls="--", color=COLOR_CASANOVO, lw=1.5, alpha=0.8, zorder=1,
                label=f"Linear fit (slope={coeffs[0]:.3f})")
        r = np.corrcoef(sc["median_sa"], sc["delta_prec"])[0, 1]
        ax.text(0.97, 0.97, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=COLOR_CASANOVO)

    texts = [
        ax.text(row["median_sa"], row["delta_prec"], species_label(row["species"]),
                fontsize=9, color="black", path_effects=glow, zorder=5,
                ha="center", va="bottom")
        for _, row in sc.iterrows()
    ]
    adjust_text(texts, ax=ax, seed=42,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="gray",
                                alpha=0.6, zorder=4),
                force_text=(2, 2), min_arrow_len=5, expand=(1.5, 1.5))

    ax.set_xlabel("Median SA of Casanovo-wrong PSMs (true peptide)", fontsize=12)
    ax.set_ylabel("Δ Peptide Precision (Casanovo+PepPr − Casanovo)", fontsize=12)
    ax.set_title("Spectral Evidence Strength vs PepPr gain", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(sc) >= 3:
        ax.legend(fontsize=10, framealpha=0.8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"+{y:.0%}" if y >= 0 else f"{y:.0%}")
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# Fixed SA bins tuned for the Prosit_2020_intensity_HCD SA distribution
# (median ~0.75–0.87 across the nine species). Fixed edges keep panels
# directly comparable — the per-species bin populations, shown by the
# histogram row, then carry real information about spectrum quality.
SA_BINS = [
    ("<0.70",     0.00, 0.70),
    ("0.70–0.80", 0.70, 0.80),
    ("0.80–0.90", 0.80, 0.90),
    ("≥0.90",     0.90, 2.00),
]

COLOR_DNPS   = NINE_SPECIES_COLOR_DNPS
COLOR_HYBRID = NINE_SPECIES_COLOR_PP


def _precision_by_sa(df, bins=SA_BINS):
    """Return [(label, prec_cas, prec_pp, n), …] for each SA bin."""
    out = []
    for label, lo, hi in bins:
        mask = (df["sa"] >= lo) & (df["sa"] < hi)
        sub  = df[mask]
        prec_cas = float(sub["cas_ok"].mean()) if len(sub) > 0 else np.nan
        prec_pp  = float(sub["pp_ok"].mean())  if len(sub) > 0 else np.nan
        out.append((label, prec_cas, prec_pp, len(sub)))
    return out


def plot_precision_by_sa(all_df, out_path):
    """Precision-by-SA plot mirroring the peptide-precision-by-length layout.

    One panel per species: line chart (top) shows Casanovo and Casanovo+PepPr
    precision in each SA bin; bar chart (bottom) shows PSM counts per bin.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    species_available = [
        sp for sp in SPECIES_ORDER
        if sp in all_df["species"].unique()
        and len(all_df[all_df["species"] == sp].dropna(subset=["sa"])) > 0
    ]
    if not species_available:
        print("No SA data for precision-by-SA plot; skipping.", file=sys.stderr)
        return

    ncols  = 3
    nspec  = len(species_available)
    nrows  = (nspec + ncols - 1) // ncols
    x      = np.arange(len(SA_BINS))
    xlbls  = [b[0] for b in SA_BINS]

    fig = plt.figure(figsize=(NINE_SPECIES_PANEL_INCHES[0] * ncols,
                              NINE_SPECIES_PANEL_INCHES[1] * nrows))
    outer = GridSpec(nrows, ncols, hspace=0.45, wspace=0.35, figure=fig,
                     top=0.93, bottom=0.09)

    for idx, sp in enumerate(species_available):
        grow, gcol = divmod(idx, ncols)
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[grow, gcol],
            height_ratios=[3, 1], hspace=0.05,
        )
        ax_line = fig.add_subplot(inner[0])
        ax_hist = fig.add_subplot(inner[1])

        sp_df  = all_df[all_df["species"] == sp].dropna(subset=["sa"])
        bins   = _precision_by_sa(sp_df)
        prec_c = [b[1] for b in bins]
        prec_p = [b[2] for b in bins]
        ns     = [b[3] for b in bins]

        ax_line.plot(x, prec_c, marker="o", lw=NINE_SPECIES_LINE_WIDTH,
                     color=COLOR_DNPS, markersize=NINE_SPECIES_MARKER_SIZE,
                     label="Casanovo")
        ax_line.plot(x, prec_p, marker="o", lw=NINE_SPECIES_LINE_WIDTH,
                     color=COLOR_HYBRID,
                     markersize=5, label="Casanovo+PepPr")
        ax_line.set_xlim(-0.5, len(x) - 0.5)
        ax_line.set_ylim(0, 1)
        ax_line.set_xticks(x)
        ax_line.set_xticklabels([])
        ax_line.set_title(f"{species_label(sp)} (n={len(sp_df):,})",
                          fontsize=NINE_SPECIES_TITLE_FONTSIZE)
        ax_line.spines["top"].set_visible(False)
        ax_line.spines["right"].set_visible(False)
        ax_line.tick_params(labelsize=9)
        if gcol == 0:
            ax_line.set_ylabel("Peptide precision", fontsize=NINE_SPECIES_LABEL_FONTSIZE)

        ax_hist.bar(x, ns, width=0.6, color=COLOR_HISTOGRAM_GRAY, linewidth=0)
        for i, n in enumerate(ns):
            ax_hist.text(i, n, f"{n:,}", ha="center", va="bottom",
                         fontsize=8, color=COLOR_TEXT_GRAY)
        ax_hist.set_xlim(-0.5, len(x) - 0.5)
        ax_hist.set_xticks(x)
        ax_hist.set_xticklabels(xlbls, fontsize=8)
        ax_hist.set_yticks([])
        ax_hist.set_ylim(0, max(ns) * 1.4 if max(ns) > 0 else 1)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.spines["left"].set_visible(False)
        ax_hist.tick_params(labelsize=8)
        if gcol == 0:
            ax_hist.set_ylabel("n PSMs", fontsize=8)

    for idx in range(nspec, nrows * ncols):
        grow, gcol = divmod(idx, ncols)
        fig.add_subplot(outer[grow, gcol]).set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=COLOR_DNPS,   lw=2, marker="o", ms=5,
                   label="Casanovo"),
        plt.Line2D([0], [0], color=COLOR_HYBRID,  lw=2, marker="o", ms=5,
                   label="Casanovo+PepPr"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Peptide Precision by Spectral Angle", fontsize=14)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Plot 3: P(corrected | SA bin) per species ─────────────────────────────
def plot_correction_rate(all_df, sa_data, out_path, n_bins=10):
    species_with_data = [sp for sp in SPECIES_ORDER if sp in sa_data
                         and len(sa_data[sp].dropna(subset=["sa"])) > 0]
    wrong_df = all_df[all_df["cas_ok"] == False].dropna(subset=["sa"])
    if wrong_df.empty:
        print("No wrong predictions; skipping correction rate plot.", file=sys.stderr)
        return

    bin_edges   = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cmap = plt.cm.get_cmap("tab10", len(species_with_data))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, sp in enumerate(species_with_data):
        sp_df = wrong_df[wrong_df["species"] == sp]
        if len(sp_df) < 20:
            continue
        bin_idx = np.digitize(sp_df["sa"].values, bin_edges[1:-1])
        rates, centers = [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() >= 5:
                rates.append(float(sp_df["pp_ok"].values[mask].mean()))
                centers.append(bin_centers[b])
        if not rates:
            continue
        ax.plot(centers, rates, "o-", color=cmap(i), lw=1.8, ms=5,
                label=species_label(sp), alpha=0.85)

    overall_mean = float(wrong_df["pp_ok"].mean())
    ax.axhline(overall_mean, ls=":", color="gray", lw=1.2,
               label=f"Overall mean ({overall_mean:.1%})")
    ax.set_xlabel("Spectral Angle (true peptide)", fontsize=12)
    ax.set_ylabel("P(Corrected by PP | Casanovo wrong)", fontsize=12)
    ax.set_title(
        "Correction Probability vs Spectral Angle\n(Casanovo-wrong PSMs only)",
        fontsize=12,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load or build the SA cache
    if os.path.exists(SA_CACHE):
        with open(SA_CACHE, "rb") as _f:
            sa_data = pickle.load(_f)
        print(f"Loaded SA cache: {list(sa_data.keys())}", file=sys.stderr)
    else:
        sa_data = {}

    rng = np.random.default_rng(42)
    changed = False
    for species, dirname in SPECIES_DIRS.items():
        expected_sources = tuple(_pp_result_paths(
            species, result_run_path(dirname)
        ))
        cache_is_stale = (
            species == "yeast"
            and sa_data.get(species, pd.DataFrame()).attrs.get("source_files")
            != expected_sources
        )
        if species in sa_data and not cache_is_stale:
            print(f"  {species}: already cached ({len(sa_data[species]):,} rows).",
                  file=sys.stderr)
            continue
        df = load_species(species, dirname, rng)
        if df is not None:
            sa_data[species] = df
            changed = True
            # Save incrementally after each species
            with open(SA_CACHE, "wb") as _f:
                pickle.dump(sa_data, _f)
            print(f"  Saved incremental cache ({list(sa_data.keys())})",
                  file=sys.stderr)

    # Combine all species into one DataFrame
    all_df = pd.concat(
        [df.assign(species=sp) for sp, df in sa_data.items() if df is not None],
        ignore_index=True,
    ).dropna(subset=["sa"])

    # Load existing Δprecision data (from plot_scatter_precision.py output)
    prec_df = None
    if os.path.exists(EXISTING_CACHE_CSV):
        prec_df = pd.read_csv(EXISTING_CACHE_CSV)
        prec_df = prec_df[prec_df["condition"] == "same"]

    # Print summary statistics
    print("\nSA statistics per species (same-species condition):")
    for sp in sorted(sa_data):
        d = sa_data[sp].dropna(subset=["sa"])
        cts = d["status"].value_counts().to_dict()
        print(f"  {sp:12s}  n={len(d):6,}  "
              f"median_sa={d['sa'].median():.3f}  {cts}")

    plot_violin(all_df, sa_data, os.path.join(PROJECT_ROOT, "sa_distribution_per_species.png"))
    plot_sa_vs_delta(all_df, prec_df, os.path.join(PROJECT_ROOT, "sa_vs_delta_precision.png"))
    plot_correction_rate(all_df, sa_data, os.path.join(PROJECT_ROOT, "correction_rate_vs_sa.png"))
    plot_precision_by_sa(all_df, os.path.join(PROJECT_ROOT, "supp_fig_2.png"))

    print("\nDone.")
