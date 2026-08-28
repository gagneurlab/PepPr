#!/usr/bin/env python3
"""Figure 2 — Casanovo+PepPr improves de novo precision across nine species.

Five panels:
  A. Same-species scatter (Casanovo vs Casanovo+PepPr final precision, per species)
  B. Nine-species mouse benchmark — Casanovo+PepPr vs PowerNovo / Casanovo /
     SMSNet / ContraNovo
  C. Peptide recall vs spectral angle (mouse panel of plot_supp_fig_2;
     Prosit_2020_intensity_HCD spectral angle)
  D. Peptide recall vs peptide length (mouse panel of plot_supp_fig_1)
  E. Representative mouse spectrum where Casanovo misorders an adjacent residue
     pair (L↔A) and Casanovo+PepPr recovers the correct ordering.

Reuses caches written by:
  * plot_scatter_precision.py  → scatter_precision_cache.csv
  * plot_supp_fig_2.py         → sa_correction_cache_prosit.pkl (Prosit SA)
  * plot_supp_fig_1.py         → precision_by_length_cache.pkl
and builds benchmark_mouse_cache.pkl in-script (baseline PC curves; first run).
Panel E uses the shared spectrum-rendering and sequence-diff helpers.

Usage:
    python experiments/fig2/plot_figure_2.py
"""
import os
import sys
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.metrics import auc

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from experiments.utils.evaluation import (
    plot_sequence_diff, plot_spectrum_on_ax, paired_median_delta_stats,
    annotate_missing_ion_gap, annotate_precursor_readout,
    _normalize_to_massivekb, _expand_masses, evaluate, MASSIVEKB_MASSES,
    load_mztab_with_mgf, _parse_mgf_spectra, _parse_ms_run_locations, _SPECTRA_REF_RE,
)
from experiments.paths import (
    COLOR_CASANOVO as COLOR_DNPS,
    COLOR_HISTOGRAM_GRAY,
    COLOR_PP,
    PROJECT_ROOT,
    SMSNET_ROOT,
    nine_species_benchmark_dir,
    result_run_path,
)


def _fmt_p(p):
    if not (p == p):  # NaN
        return "p=n/a"
    if p < 1e-3:
        return "p<0.001"
    return f"p={p:.3f}"

# ── Panel A: same-species scatter ─────────────────────────────────────────
SPECIES_LABELS = {
    "cowpea":      "ricebean",
    "archaeon":    "mmazei",
    "endoloripes": "clambacteria",
}
def _sp_label(s):
    return SPECIES_LABELS.get(s, s)


def plot_panel_A(ax):
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "scatter_precision_cache.csv"))
    sc = df[df["condition"] == "same"].copy()

    # Connect (Casanovo, Casanovo) → (Casanovo, Casanovo+PepPr) per species
    for _, row in sc.iterrows():
        ax.plot([row["casanovo"], row["casanovo"]],
                [row["casanovo"], row["casanovo_pp"]],
                color=COLOR_PP, lw=1.5, alpha=0.5, zorder=1)
    stats = paired_median_delta_stats(sc["casanovo"].to_numpy(),
                                       sc["casanovo_pp"].to_numpy())
    med_d, ci_lo, ci_hi = stats["median_delta"], stats["ci_low"], stats["ci_high"]
    pstr = _fmt_p(stats["pvalue"])
    ax.scatter(sc["casanovo"], sc["casanovo_pp"], s=50, zorder=3,
               marker="o", color=COLOR_PP, edgecolors="white", linewidths=0.4,
               label=(f"Casanovo + PepPr (median Δ = {med_d*100:+.1f} pp "
                      f"[{ci_lo*100:+.1f}, {ci_hi*100:+.1f}], {pstr})"))

    glow = [patheffects.withStroke(linewidth=3.0, foreground="white")]
    try:
        from adjustText import adjust_text
        texts = [
            ax.text(row["casanovo"], row["casanovo_pp"],
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
    ax.set_xlim(lo, 0.8)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.set_xlabel("Casanovo peptide recall", fontsize=11)
    ax.set_ylabel("Casanovo + PepPr peptide recall", fontsize=11)
    ax.set_title("A. Nine-species benchmark", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.tick_params(labelsize=10)


# ── Panel B: 9s[mouse] benchmark vs other tools ───────────────────────────
BENCH_CACHE = os.path.join(PROJECT_ROOT, "benchmark_mouse_cache.pkl")


# benchmark_mouse_cache.pkl may predate the PepPr rename.
_LEGACY_PP_CURVE_KEYS = {
    "Casanovo + PP": "Casanovo + PepPr",
    "ContraNovo + PP": "ContraNovo + PepPr",
}


def _migrate_pp_curve_keys(cache: dict) -> dict:
    for old, new in _LEGACY_PP_CURVE_KEYS.items():
        if old in cache and new not in cache:
            cache[new] = cache.pop(old)
    return cache




# ── Benchmark baseline loaders + PC helpers (merged from the former
# plot_benchmark.py; its standalone __main__ figure was dropped). ──
_SMSNET_BASE = SMSNET_ROOT

SPECIES_CFG = {
    "mouse": {
        "mgf_dir":         nine_species_benchmark_dir("mouse"),
        "results_dir":     result_run_path("mus_musculus"),
        "label":           "Mouse",
        "mztab_dnps":      "9s_mouse_dnps.mztab",
        "mztab_hybrid":    "9s_mouse_asymbnln.mztab",
        "smsnet_out":      (os.path.join(_SMSNET_BASE, "mouse_inputs_output")
                            if _SMSNET_BASE else None),
        "contranovo_out":  os.path.join(
            result_run_path("mus_musculus"), "contranovo", "mouse_dnps"
        ),
        "contranovo_pp_out": os.path.join(
            result_run_path("mus_musculus"), "contranovo", "mouse_fusion"
        ),
        "instanovo_out":   os.path.join(result_run_path("mus_musculus"), "instanovo"),
    },
    "human": {
        "mgf_dir":         nine_species_benchmark_dir("human"),
        "results_dir":     result_run_path("casanovo"),
        "label":           "Human",
        "mztab_dnps":      "9s_human_dnps.mztab",
        "mztab_hybrid":    "9s_human_plmhuman_iso_hybrid.mztab",
        "smsnet_out":      (os.path.join(_SMSNET_BASE, "human_inputs_output")
                            if _SMSNET_BASE else None),
        "contranovo_out":  os.path.join(
            result_run_path("casanovo"), "contranovo", "human_dnps"
        ),
        "contranovo_pp_out": None,   # not trained yet for human
        "instanovo_out":   None,
    },
}

plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
})


# PowerNovo prints mod masses rounded to 2 decimals; MassIVE-KB uses the
# canonical 3-decimal monoisotopic shifts.  Map 1:1 to the MKB tokens so
# downstream ``_expand_masses``/``aa_match`` sees the same vocabulary as
# the ground-truth sequences.
_POWERNOVO_MOD_MAP = {
    "+15.99": "+15.995",   # Oxidation (M)
    "+42.01": "+42.011",   # Acetyl (N-term)
    "+57.02": "+57.021",   # Carbamidomethyl (C)
    "43.01":  "+43.006",   # Carbamyl (N-term) — PowerNovo drops the '+'
    "+43.01": "+43.006",
    "-17.03": "-17.027",   # Ammonia-loss (N-term pyro-Glu)
    "+0.98":  "+0.984",    # Deamidation (N/Q)
}


def _canonical_mod(mass_str):
    """Map a PowerNovo 2-dp mass string to its MassIVE-KB 3-dp token."""
    if mass_str in _POWERNOVO_MOD_MAP:
        return _POWERNOVO_MOD_MAP[mass_str]
    if not mass_str.startswith(("+", "-")):
        return "+" + mass_str
    return mass_str


def _normalize_powernovo(seq):
    """Normalize PowerNovo peptide notation to MassIVE-KB compatible format.

    PowerNovo emits three flavours of parenthetical modifications:

    * ``X(+mass)`` in the middle of a peptide → side-chain mod on AA ``X``
      (e.g. ``M(+15.99)`` → ``M+15.995``).
    * ``(+mass)X...`` at the start → N-terminal mass-shift token
      (e.g. ``(43.01)PEP`` → ``+43.006PEP``).  Note PowerNovo omits the
      leading ``+`` for Carbamyl, which we restore here.
    * ``...X(+mass)`` at the end → PowerNovo artefact where an N-terminal
      Acetyl was placed at the C-term because the model generates
      C→N.  We migrate the token back to the N-terminus.

    Additionally, Carbamidomethyl on Cys is a *fixed* experimental modification
    that PowerNovo does not always annotate; instead the +57.021 Da mass is
    absorbed as an extra trailing glycine.  We add ``+57.021`` to each bare C
    and strip one trailing ``G`` per bare C so the total peptide mass is
    preserved.
    """
    if not isinstance(seq, str):
        return seq

    m = re.match(r'^([A-Z][^()]*)\(([+-]?\d+\.\d+)\)$', seq)
    if m:
        seq = f"({m.group(2)}){m.group(1)}"

    m = re.match(r'^\(([+-]?\d+\.\d+)\)(.*)$', seq)
    if m:
        seq = _canonical_mod(m.group(1)) + m.group(2)

    seq = re.sub(
        r'\(([+-]?\d+\.\d+)\)',
        lambda mm: _canonical_mod(mm.group(1)),
        seq,
    )

    n_bare_c = len(re.findall(r'C(?![+-]?\d)', seq))
    seq = re.sub(r'C(?![+-]?\d)', 'C+57.021', seq)
    for _ in range(n_bare_c):
        if seq.endswith('G'):
            seq = seq[:-1]
    return seq


_SMSNET_AA = {
    "A": "A", "C": "C+57.021", "D": "D", "E": "E", "F": "F", "G": "G",
    "H": "H", "I": "I", "K": "K", "L": "L", "M": "M", "m": "M+15.995",
    "N": "N", "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T", "V": "V",
    "W": "W", "Y": "Y",
}


def _smsnet_tokens_to_massivekb(tokens):
    """Join SMSNet AA tokens into a MassIVE-KB sequence, or None if any token is unknown."""
    out = []
    for t in tokens:
        aa = _SMSNET_AA.get(t)
        if aa is None:
            return None
        out.append(aa)
    return "".join(out)


def load_mztab_with_keys(mztab_path):
    """Like ``load_mztab_with_mgf`` but also attaches ``raw_name`` and
    ``title`` columns for unambiguous joins across tools.

    MGF spectrum TITLE strings (e.g. ``controllerType=0 controllerNumber=1
    scan=N``) are NOT globally unique across files — the same scan number
    appears in multiple raw files.  We therefore key every join on the
    composite ``(raw_name, title)`` pair instead of ``title`` alone.
    """
    df = load_mztab_with_mgf(mztab_path)

    ms_run_locs = _parse_ms_run_locations(mztab_path)
    run_to_raw = {
        r: os.path.splitext(os.path.basename(p))[0]
        for r, p in ms_run_locs.items()
    }
    raw_names = []
    with open(mztab_path) as f:
        header = None
        for line in f:
            if line.startswith("PSH\t"):
                header = line.strip().split("\t")
            elif line.startswith("PSM\t"):
                cols = line.strip().split("\t")
                row = dict(zip(header, cols))
                m = _SPECTRA_REF_RE.match(row.get("spectra_ref", ""))
                raw_names.append(run_to_raw.get(int(m.group(1))) if m else None)
    df = df.copy()
    df["raw_name"] = raw_names
    df["title"] = df["scans"]
    return df


def load_smsnet(mgf_dir, smsnet_output_dir):
    """Load SMSNet outputs for all MGFs and join with ground truth."""
    rows = []
    for mgf_name in sorted(os.listdir(mgf_dir)):
        if not mgf_name.endswith(".mgf"):
            continue
        base = mgf_name[:-4]
        seq_path = os.path.join(smsnet_output_dir, base)
        score_path = os.path.join(smsnet_output_dir, base + "_rescore")
        if not os.path.exists(seq_path) or not os.path.exists(score_path):
            continue

        spectra = _parse_mgf_spectra(os.path.join(mgf_dir, mgf_name))
        with open(seq_path) as sf, open(score_path) as pf:
            for idx, (sline, pline) in enumerate(zip(sf, pf)):
                tokens = sline.split()
                scores = pline.split()
                if not tokens or len(tokens) != len(scores):
                    continue
                if "<s>" in tokens or "<unk>" in tokens or "</s>" in tokens:
                    continue
                pred = _smsnet_tokens_to_massivekb(tokens)
                if pred is None or idx >= len(spectra):
                    continue
                probs = np.exp(np.asarray(scores, dtype=float))
                score = float(probs.mean())
                rows.append({
                    "raw_name": base,
                    "pred": pred,
                    "score": score,
                    "true_seq": spectra[idx][1],
                    "title": spectra[idx][0],
                })
    return pd.DataFrame(rows)


def load_contranovo(mgf_dir, contranovo_output_dir):
    """Load ContraNovo denovo CSVs and join with MGF ground truth."""
    rows = []
    for mgf_name in sorted(os.listdir(mgf_dir)):
        if not mgf_name.endswith(".mgf"):
            continue
        base = mgf_name[:-4]
        csv_path = os.path.join(contranovo_output_dir, base + ".csv")
        if not os.path.exists(csv_path):
            continue
        spectra = _parse_mgf_spectra(os.path.join(mgf_dir, mgf_name))
        title_to_true = {title: true_seq for title, true_seq in spectra}
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            title = row["title"]
            true_seq = title_to_true.get(title)
            if true_seq is None:
                continue
            pred = row["peptide"]
            if not isinstance(pred, str) or not pred:
                continue
            rows.append({
                "raw_name": base,
                "pred": pred,
                "score": float(row["score"]),
                "true_seq": true_seq,
                "title": title,
            })
    return pd.DataFrame(rows)


def load_instanovo(mgf_dir, instanovo_output_dir):
    """Load InstaNovo predict CSVs (one per MGF) and join with MGF ground truth.

    Each CSV has columns ``preds``, ``log_probs``, ``spectrum_id`` (the last
    set via ``index_columns=[...,spectrum_id]`` in the predict invocation).
    ``spectrum_id`` is ``"<source_file>:<idx>"`` where ``idx`` is the 0-based
    row index in the input SpectrumDataFrame (one MGF per call → MGF order).
    """
    rows = []
    for mgf_name in sorted(os.listdir(mgf_dir)):
        if not mgf_name.endswith(".mgf"):
            continue
        base = mgf_name[:-4]
        csv_path = os.path.join(instanovo_output_dir, base + ".csv")
        if not os.path.exists(csv_path):
            continue
        spectra = _parse_mgf_spectra(os.path.join(mgf_dir, mgf_name))
        df = pd.read_csv(csv_path)
        has_sid = "spectrum_id" in df.columns
        pred_col = "preds" if "preds" in df.columns else "predictions"
        score_col = "log_probs" if "log_probs" in df.columns else "log_probabilities"
        for row_idx, row in df.iterrows():
            pred = row.get(pred_col)
            if not isinstance(pred, str) or not pred:
                continue
            if has_sid:
                sid = str(row.get("spectrum_id", ""))
                try:
                    idx = int(sid.rsplit(":", 1)[-1])
                except (ValueError, AttributeError):
                    continue
            else:
                idx = int(row_idx)
            if idx < 0 or idx >= len(spectra):
                continue
            title, true_seq = spectra[idx]
            if true_seq is None:
                continue
            try:
                score = float(np.exp(float(row.get(score_col))))
            except (TypeError, ValueError):
                continue
            rows.append({
                "raw_name": base,
                "pred": pred,
                "score": score,
                "true_seq": true_seq,
                "title": title,
            })
    return pd.DataFrame(rows)


def load_powernovo(results_dir, mgf_dir, powernovo_files):
    """Load all PowerNovo pw_score.csv files and join with MGF ground truth.

    Each row's ``Spectrum Name`` has the form ``<file>:index=N`` where N is
    the 0-based MGF row index of the source spectrum.  This requires the
    patched ``powernovo/models/peptide_bert/peptide_bert.py`` whose
    ``score_hypotheses`` preserves 1:1 alignment with the input batch;
    without the patch N drifts by the number of BERT-filtered hypotheses.
    """
    rows = []
    for pw_file in powernovo_files:
        raw_name = pw_file.replace("_pw_score.csv", "")
        mgf_path = os.path.join(mgf_dir, raw_name + ".mgf")
        if not os.path.exists(mgf_path):
            continue

        spectra = _parse_mgf_spectra(mgf_path)
        n_spec = len(spectra)

        df = pd.read_csv(os.path.join(results_dir, pw_file))
        for _, row in df.iterrows():
            idx = int(row["Spectrum Name"].split(":index=")[1])
            if idx >= n_spec:
                continue
            raw_pred = row["PowerNovo Peptides"]
            rows.append({
                "raw_name": raw_name,
                "idx": idx,
                "pred_raw": raw_pred,
                "pred": _normalize_powernovo(raw_pred),
                "score": float(row["PowerNovo Score"]),
                "true_seq": spectra[idx][1],
                "title": spectra[idx][0],
            })
    return pd.DataFrame(rows)


def compute_pc(df, n_total, return_df=False):
    """Given a df with pred/score/true_seq, return (coverage, precision) arrays.

    Coverage is computed as rank / n_total so that tools predicting on fewer
    spectra top out at a coverage < 1, giving a fair cross-tool comparison.

    When ``return_df`` is True, also return a copy of the sorted input df
    augmented with a boolean ``pep_match`` column so callers can dump the
    exact rows (true/pred/score) that feed into the PC curve.
    """
    df = df.copy()
    df["pred"] = df["pred"].apply(_normalize_to_massivekb)
    df["true_seq"] = df["true_seq"].apply(_normalize_to_massivekb)
    df["score"] = df["score"].astype(float)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    masses = _expand_masses(
        df["pred"].tolist() + df["true_seq"].tolist(), MASSIVEKB_MASSES
    )
    aa_matches_batch = evaluate.aa_match_batch(
        df["true_seq"].tolist(), df["pred"].tolist(), masses,
    )
    pep_match = np.asarray([m[1] for m in aa_matches_batch[0]])
    precision = np.cumsum(pep_match) / np.arange(1, len(pep_match) + 1)
    coverage = np.arange(1, len(pep_match) + 1) / n_total
    accuracy = float(pep_match.sum()) / len(pep_match) if len(pep_match) else 0.0
    if return_df:
        df = df.copy()
        df["pep_match"] = pep_match.astype(bool)
        return coverage, precision, accuracy, df
    return coverage, precision, accuracy



def _load_or_build_benchmark_mouse():
    cfg = SPECIES_CFG["mouse"]
    results_dir = cfg["results_dir"]
    mgf_dir = cfg["mgf_dir"]

    cache = None
    if os.path.exists(BENCH_CACHE):
        with open(BENCH_CACHE, "rb") as f:
            cache = pickle.load(f)
        cache = _migrate_pp_curve_keys(cache)

    def _n_total():
        return sum(
            sum(1 for line in open(os.path.join(mgf_dir, f)) if line.startswith("BEGIN IONS"))
            for f in os.listdir(mgf_dir) if f.endswith(".mgf")
        )

    # If we already have a non-empty cache, top it up with any newly-added
    # tools (currently InstaNovo) and return. Only fall through to a full
    # rebuild when the cache file is missing or empty.
    if cache:
        if "InstaNovo" not in cache:
            in_dir = cfg.get("instanovo_out")
            if in_dir and os.path.isdir(in_dir):
                in_df = load_instanovo(mgf_dir, in_dir)
                if len(in_df):
                    cov, prec, acc = compute_pc(in_df, n_total=_n_total())
                    cache["InstaNovo"] = (cov, prec, acc, len(cov))
                    with open(BENCH_CACHE, "wb") as f:
                        pickle.dump(cache, f)
        return cache

    pw_files = sorted(
        f for f in os.listdir(results_dir)
        if f.endswith("_pw_score.csv") and not any(
            f.endswith(s) for s in (".k7.csv", ".k8.csv", ".k10.csv"))
    )
    pw_df = load_powernovo(results_dir, mgf_dir, pw_files)
    hy_df = load_mztab_with_keys(os.path.join(results_dir, cfg["mztab_hybrid"]))
    ca_df = load_mztab_with_keys(os.path.join(results_dir, cfg["mztab_dnps"]))
    if not cfg["smsnet_out"]:
        raise RuntimeError(
            "Figure 2 SMSNet panel requires DNPS_SMSNET_ROOT."
        )
    sm_df = load_smsnet(mgf_dir, cfg["smsnet_out"])
    cn_df = load_contranovo(mgf_dir, cfg["contranovo_out"])
    cn_pp_dir = cfg.get("contranovo_pp_out")
    cn_pp_df = (
        load_contranovo(mgf_dir, cn_pp_dir)
        if cn_pp_dir and os.path.isdir(cn_pp_dir) else None
    )
    in_dir = cfg.get("instanovo_out")
    in_df = (
        load_instanovo(mgf_dir, in_dir)
        if in_dir and os.path.isdir(in_dir) else None
    )
    pc_kwargs = dict(n_total=_n_total())
    curves = {
        "PowerNovo":      compute_pc(pw_df, **pc_kwargs),
        "Casanovo":       compute_pc(ca_df, **pc_kwargs),
        "SMSNet":         compute_pc(sm_df, **pc_kwargs),
        "ContraNovo":     compute_pc(cn_df, **pc_kwargs),
        "Casanovo + PepPr":  compute_pc(hy_df, **pc_kwargs),
    }
    if cn_pp_df is not None and len(cn_pp_df):
        curves["ContraNovo + PepPr"] = compute_pc(cn_pp_df, **pc_kwargs)
    if in_df is not None and len(in_df):
        curves["InstaNovo"] = compute_pc(in_df, **pc_kwargs)
    # compute_pc returns (cov, prec, acc); cache that + n
    out = {k: (cov, prec, acc, len(cov)) for k, (cov, prec, acc) in curves.items()}
    with open(BENCH_CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def plot_panel_B(ax):
    curves = _load_or_build_benchmark_mouse()
    # PP arms are navy; every non-PP de novo baseline gets its own shade of
    # orange (light→dark ramp) so they read as one family and are told apart
    # by hue rather than by line style alone. Line styles are kept as a
    # secondary cue where curves overlap.
    style = {
        "Casanovo + PepPr":   dict(color=COLOR_PP,   ls="-"),
        "ContraNovo + PepPr": dict(color=COLOR_PP,   ls="-."),
        "Casanovo":        dict(color=COLOR_DNPS,  ls="-"),   # signature orange
        "PowerNovo":       dict(color="#fdae6b",  ls="--"),  # light orange
        "SMSNet":          dict(color="#f16913",  ls=":"),   # mid orange
        "ContraNovo":      dict(color="#d94801",  ls="-."),  # dark orange
        "InstaNovo":       dict(color="#a63603",              # darkest orange
                                 ls=(0, (3, 1, 1, 1, 1, 1))), # dash-dot-dot
    }
    order = ["Casanovo + PepPr", "Casanovo", "PowerNovo", "SMSNet",
             "ContraNovo", "ContraNovo + PepPr", "InstaNovo"]
    order = [k for k in order if k in curves]
    for name in order:
        cov, prec, _acc, _n = curves[name]
        ap_v = auc(cov, prec)
        pep_recall = float(prec[-1]) if len(prec) else 0.0
        ax.plot(cov, prec, lw=2.2,
                label=f"{name} (AP={ap_v:.3f}, pep recall={pep_recall:.2f})",
                **style[name])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage", fontsize=11)
    ax.set_ylabel("Peptide precision", fontsize=11)
    ax.set_title("B. DNPS benchmark on mouse data", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.85, handlelength=3.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


# ── Panel C: precision vs spectral angle (mouse) ──────────────────────────
# Matches Supp Fig 2 methodology: spectral angle is computed via
# Prosit_2020_intensity_HCD (cached to sa_correction_cache_prosit.pkl), with
# fixed bins tuned for the Prosit SA distribution (median ~0.75–0.87).
SA_BINS = [
    ("<0.70",     0.00, 0.70),
    ("0.70–0.80", 0.70, 0.80),
    ("0.80–0.90", 0.80, 0.90),
    ("≥0.90",     0.90, 2.00),
]


def plot_panel_C(ax_line, ax_hist):
    with open(os.path.join(PROJECT_ROOT, "sa_correction_cache_prosit.pkl"), "rb") as f:
        sa_cache = pickle.load(f)
    mouse_df = sa_cache["mouse"].dropna(subset=["sa"]) if "mouse" in sa_cache else None
    if mouse_df is None or len(mouse_df) == 0:
        ax_line.text(0.5, 0.5, "(no SA data)", ha="center", va="center",
                     transform=ax_line.transAxes); return
    from experiments.utils.evaluation import wilson_ci
    bins, prec_c, prec_p, ns = [], [], [], []
    cas_err_lo, cas_err_hi, pp_err_lo, pp_err_hi = [], [], [], []
    for label, lo, hi in SA_BINS:
        m = (mouse_df["sa"] >= lo) & (mouse_df["sa"] < hi)
        sub = mouse_df[m]
        bins.append(label)
        n = len(sub)
        ns.append(n)
        pc, lo_c, hi_c = wilson_ci(int(sub["cas_ok"].sum()), n) if n else (np.nan,) * 3
        pp, lo_p, hi_p = wilson_ci(int(sub["pp_ok"].sum()), n)  if n else (np.nan,) * 3
        prec_c.append(pc if pc == pc else np.nan)
        prec_p.append(pp if pp == pp else np.nan)
        cas_err_lo.append((pc - lo_c) if pc == pc else 0.0)
        cas_err_hi.append((hi_c - pc) if pc == pc else 0.0)
        pp_err_lo.append((pp - lo_p)  if pp == pp else 0.0)
        pp_err_hi.append((hi_p - pp)  if pp == pp else 0.0)

    x = np.arange(len(SA_BINS))
    ax_line.errorbar(x, prec_c, yerr=[cas_err_lo, cas_err_hi], marker="o",
                     lw=2.0, ms=6, color=COLOR_DNPS, label="Casanovo",
                     elinewidth=0.9, capsize=2.5, capthick=0.9)
    ax_line.errorbar(x, prec_p, yerr=[pp_err_lo, pp_err_hi], marker="o",
                     lw=2.0, ms=6, color=COLOR_PP, label="Casanovo + PepPr",
                     elinewidth=0.9, capsize=2.5, capthick=0.9)
    ax_line.set_xlim(-0.4, len(x) - 0.6)
    ax_line.set_ylim(0, 1)
    ax_line.set_xticks(x)
    ax_line.set_xticklabels([])
    ax_line.set_ylabel("Peptide recall", fontsize=11)
    ax_line.set_title("C. Stratification of mouse data by Spectral Angle", fontsize=12)
    ax_line.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.tick_params(labelsize=10)

    ax_hist.bar(x, ns, width=0.6, color=COLOR_HISTOGRAM_GRAY, linewidth=0)
    for i, n in enumerate(ns):
        ax_hist.text(i, n, f"{n:,}", ha="center", va="bottom",
                     fontsize=8, color="#444")
    ax_hist.set_xlim(-0.4, len(x) - 0.6)
    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels(bins, fontsize=9)
    ax_hist.set_yticks([])
    ax_hist.set_ylim(0, max(ns) * 1.4 if max(ns) > 0 else 1)
    ax_hist.set_xlabel("Spectral angle", fontsize=10)
    ax_hist.set_ylabel("n PSMs", fontsize=9)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.spines["left"].set_visible(False)


# ── Panel D: precision vs peptide length (mouse) ──────────────────────────
def plot_panel_D(ax_line, ax_hist):
    with open(os.path.join(PROJECT_ROOT, "precision_by_length_cache.pkl"), "rb") as f:
        L = pickle.load(f)
    from experiments.utils.evaluation import wilson_ci
    mouse = L["mouse"]
    bins = [b[0] for b in mouse["dnps"]]
    prec_c = [b[1] for b in mouse["dnps"]]
    prec_p = [b[1] for b in mouse["pp"]]
    ns_c = [b[2] for b in mouse["dnps"]]
    ns_p = [b[2] for b in mouse["pp"]]
    ns = ns_c
    cas_err_lo, cas_err_hi, pp_err_lo, pp_err_hi = [], [], [], []
    for p, n in zip(prec_c, ns_c):
        _, lo, hi = wilson_ci(int(round(p * n)), n) if n else (np.nan,) * 3
        cas_err_lo.append((p - lo) if lo == lo else 0.0)
        cas_err_hi.append((hi - p) if hi == hi else 0.0)
    for p, n in zip(prec_p, ns_p):
        _, lo, hi = wilson_ci(int(round(p * n)), n) if n else (np.nan,) * 3
        pp_err_lo.append((p - lo) if lo == lo else 0.0)
        pp_err_hi.append((hi - p) if hi == hi else 0.0)
    x = np.arange(len(bins))
    ax_line.errorbar(x, prec_c, yerr=[cas_err_lo, cas_err_hi], marker="o",
                     lw=2.0, ms=6, color=COLOR_DNPS, label="Casanovo",
                     elinewidth=0.9, capsize=2.5, capthick=0.9)
    ax_line.errorbar(x, prec_p, yerr=[pp_err_lo, pp_err_hi], marker="o",
                     lw=2.0, ms=6, color=COLOR_PP, label="Casanovo + PepPr",
                     elinewidth=0.9, capsize=2.5, capthick=0.9)
    ax_line.set_xlim(-0.4, len(x) - 0.6)
    ax_line.set_ylim(0, 1)
    ax_line.set_xticks(x)
    ax_line.set_xticklabels([])
    ax_line.set_ylabel("Peptide recall", fontsize=11)
    ax_line.set_title("D. Stratification of mouse data by peptide length", fontsize=12)
    ax_line.legend(fontsize=9, loc="lower left", framealpha=0.85)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.tick_params(labelsize=10)

    ax_hist.bar(x, ns, width=0.6, color=COLOR_HISTOGRAM_GRAY, linewidth=0)
    for i, n in enumerate(ns):
        ax_hist.text(i, n, f"{n:,}", ha="center", va="bottom",
                     fontsize=8, color="#444")
    ax_hist.set_xlim(-0.4, len(x) - 0.6)
    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels(bins, fontsize=9)
    ax_hist.set_yticks([])
    ax_hist.set_ylim(0, max(ns) * 1.4 if max(ns) > 0 else 1)
    ax_hist.set_xlabel("Peptide length (aa)", fontsize=10)
    ax_hist.set_ylabel("n PSMs", fontsize=9)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.spines["left"].set_visible(False)


# ── Panel E: representative spectrum ──────────────────────────────────────
EXAMPLE = dict(
    # Mouse 9-species, L=13 example with mass-preserving L↔A swap at positions
    # 1-2 of the peptide. PP recovers the correct order with high confidence.
    mgf=os.path.join(nine_species_benchmark_dir("mouse"),
                     "20160323_CoAN_CTRL2_3373.mgf"),
    scan=22758,
    true_seq="LAPDYDALDVANK",
    v_pred  ="ALPDYDALDVANK", v_score=0.42,   # Casanovo — L↔A swap at pos 1-2
    pp_pred ="LAPDYDALDVANK", pp_score=0.99,  # Casanovo + PepPr — recovers order
    boundary_pos=None,
    precursor_mz=702.853393554688, precursor_charge=2,
    true_sa=0.8014,   # Prosit_2020_intensity_HCD SA (see experiments.utils.sa_prosit)
)


def plot_panel_E(ax_diff, ax_spec):
    # Sequence diff
    plot_sequence_diff(
        ax_diff,
        true_seq=EXAMPLE["true_seq"],
        preds=[("Casanovo",      EXAMPLE["v_pred"],  EXAMPLE["v_score"]),
               ("Casanovo + PepPr", EXAMPLE["pp_pred"], EXAMPLE["pp_score"])],
        boundary_pos=EXAMPLE["boundary_pos"] or 0,  # 0 = no visible line
        true_sa=EXAMPLE.get("true_sa"),
    )
    # If no boundary, remove the dashed line drawn by the helper
    if EXAMPLE["boundary_pos"] is None:
        for ln in list(ax_diff.lines):
            ln.remove()
        for txt in list(ax_diff.texts):
            if txt.get_text() == "V/C boundary":
                txt.remove()

    try:
        plot_spectrum_on_ax(ax_spec, mgf_path=EXAMPLE["mgf"],
                             scan=EXAMPLE["scan"], seq=EXAMPLE["true_seq"])
        # Casanovo swaps positions 1–2 (L↔A). The b1↔b2 and y11↔y12 gaps
        # would have pinned residue 2 from either end, but b1 (114.09) and
        # y12 (1291.62) are both missing — leaving the order unconstrained.
        annotate_missing_ion_gap(ax_spec, EXAMPLE["true_seq"], 'b',
                                  missing_idx=1, observed_idx=2)
        annotate_missing_ion_gap(ax_spec, EXAMPLE["true_seq"], 'y',
                                  missing_idx=12, observed_idx=11)
        # Pedagogical readout: in a complete spectrum, precursor − y12 would
        # reveal the N-terminal L (gap = 113.08 Da on the singly-charged
        # m/z axis). Dashed to flag that it depends on the missing y12.
        annotate_precursor_readout(ax_spec, EXAMPLE["true_seq"],
                                   precursor_mz=EXAMPLE["precursor_mz"],
                                   precursor_charge=EXAMPLE["precursor_charge"],
                                   from_y_idx=12)
    except Exception as e:
        ax_spec.text(0.5, 0.5, f"(spectrum not available: {e})",
                     ha="center", va="center", transform=ax_spec.transAxes,
                     color="gray", fontsize=10)
        ax_spec.axis("off")


# ── Main: assemble the 3-row figure ───────────────────────────────────────
def main():
    out_path = os.path.join(PROJECT_ROOT, "figure_2.png")
    plt.style.use("ggplot")
    plt.rcParams.update({
        "axes.facecolor": "white", "axes.edgecolor": "black",
        "axes.linewidth": 0.8, "axes.grid": False,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "font.family": "sans-serif",
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })

    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.32, wspace=0.24,
                  figure=fig)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    gsC = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0],
                                  height_ratios=[3, 1], hspace=0.05)
    axC_line = fig.add_subplot(gsC[0])
    axC_hist = fig.add_subplot(gsC[1])

    gsD = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1],
                                  height_ratios=[3, 1], hspace=0.05)
    axD_line = fig.add_subplot(gsD[0])
    axD_hist = fig.add_subplot(gsD[1])

    gsE = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, :],
                                  height_ratios=[1.6, 3.0], hspace=0.15)
    axE_diff = fig.add_subplot(gsE[0])
    axE_spec = fig.add_subplot(gsE[1])

    print("[A] same-species scatter ...")
    plot_panel_A(axA)
    print("[B] mouse benchmark ...")
    plot_panel_B(axB)
    print("[C] precision-by-SA (mouse) ...")
    plot_panel_C(axC_line, axC_hist)
    print("[D] precision-by-length (mouse) ...")
    plot_panel_D(axD_line, axD_hist)
    print("[E] worked-example spectrum ...")
    plot_panel_E(axE_diff, axE_spec)
    pos_e = gs[2, :].get_position(fig)
    fig.text(
        pos_e.x0 + pos_e.width / 2,
        pos_e.y1 + 0.004,
        "E. Example: PepPr resolves ambiguity at N-terminus",
        ha="center", va="bottom", fontsize=12,
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
