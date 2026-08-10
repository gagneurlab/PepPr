#!/usr/bin/env python3

import argparse
import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnps_hybrid.metrics import (
    _normalize_to_massivekb,
    _expand_masses,
    evaluate,
    MASSIVEKB_MASSES,
    load_mztab_with_mgf,
    _parse_mgf_spectra,
    _parse_ms_run_locations,
    _SPECTRA_REF_RE,
)
from dnps_hybrid.const import (
    COLOR_CASANOVO,
    COLOR_PP,
    SMSNET_ROOT,
    nine_species_benchmark_dir,
    result_run_path,
)
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


def main():
    parser = argparse.ArgumentParser(
        description="Precision–coverage: PowerNovo vs Casanovo vs DNPS-Hybrid "
                    "vs SMSNet on nine-species benchmark.",
    )
    parser.add_argument(
        "--species", choices=list(SPECIES_CFG), default="mouse",
        help="Species to evaluate (default: mouse).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output PNG path (default: bechmark_<species>.png).",
    )
    args = parser.parse_args()
    cfg = SPECIES_CFG[args.species]
    if args.out is None:
        args.out = f"benchmark_{args.species}.png"

    results_dir = cfg["results_dir"]
    mgf_dir = cfg["mgf_dir"]
    powernovo_files = sorted(
        f for f in os.listdir(results_dir)
        if f.endswith("_pw_score.csv") and not f.endswith(".k7.csv")
        and not f.endswith(".k8.csv") and not f.endswith(".k10.csv")
    )

    print("Loading PowerNovo ...")
    pw_df = load_powernovo(results_dir, mgf_dir, powernovo_files)
    print(f"  {len(pw_df)} PSMs from {len(powernovo_files)} files")

    print("Loading Casanovo+PepPr ...")
    hybrid_df = load_mztab_with_keys(os.path.join(results_dir, cfg["mztab_hybrid"]))
    print(f"  {len(hybrid_df)} PSMs")

    print("Loading Casanovo ...")
    casanovo_df = load_mztab_with_keys(os.path.join(results_dir, cfg["mztab_dnps"]))
    print(f"  {len(casanovo_df)} PSMs")

    n_total = sum(
        sum(1 for line in open(os.path.join(mgf_dir, f)) if line.startswith("BEGIN IONS"))
        for f in os.listdir(mgf_dir) if f.endswith(".mgf")
    )
    print(f"  {n_total} total spectra in dataset")

    print("Loading SMSNet ...")
    if not cfg["smsnet_out"]:
        raise RuntimeError(
            "SMSNet comparison requires DNPS_SMSNET_ROOT to point to the "
            "SMSNet checkout/output root."
        )
    smsnet_df = load_smsnet(mgf_dir, cfg["smsnet_out"])
    print(f"  {len(smsnet_df)} PSMs")

    print("Loading ContraNovo ...")
    contranovo_df = load_contranovo(mgf_dir, cfg["contranovo_out"])
    print(f"  {len(contranovo_df)} PSMs")

    contranovo_pp_df = None
    cn_pp_dir = cfg.get("contranovo_pp_out")
    if cn_pp_dir and os.path.isdir(cn_pp_dir):
        print("Loading ContraNovo+PepPr ...")
        contranovo_pp_df = load_contranovo(mgf_dir, cn_pp_dir)
        print(f"  {len(contranovo_pp_df)} PSMs")

    instanovo_df = None
    in_dir = cfg.get("instanovo_out")
    if in_dir and os.path.isdir(in_dir):
        print("Loading InstaNovo ...")
        instanovo_df = load_instanovo(mgf_dir, in_dir)
        print(f"  {len(instanovo_df)} PSMs")

    print("Computing precision–coverage ...")
    pc_kwargs = dict(n_total=n_total)
    cov_pw, prec_pw, acc_pw, pw_scored = compute_pc(pw_df, return_df=True, **pc_kwargs)

    pw_csv_out = "pw_pc_inputs.csv"
    pw_scored_out = pw_scored[
        ["raw_name", "idx", "true_seq", "pred_raw", "pred", "score", "pep_match"]
    ].rename(columns={"pred": "pred_norm"})
    pw_scored_out.to_csv(pw_csv_out, index=False)
    print(f"  wrote {pw_csv_out} ({len(pw_scored_out)} rows, "
          f"{pw_scored_out['pep_match'].sum()} matches)")

    cov_hy, prec_hy, acc_hy = compute_pc(hybrid_df, **pc_kwargs)
    cov_ca, prec_ca, acc_ca = compute_pc(casanovo_df, **pc_kwargs)
    cov_sm, prec_sm, acc_sm = compute_pc(smsnet_df, **pc_kwargs)
    cov_cn, prec_cn, acc_cn = compute_pc(contranovo_df, **pc_kwargs)
    if contranovo_pp_df is not None and len(contranovo_pp_df):
        cov_cnpp, prec_cnpp, acc_cnpp = compute_pc(contranovo_pp_df, **pc_kwargs)
    else:
        cov_cnpp = prec_cnpp = None; acc_cnpp = None
    if instanovo_df is not None and len(instanovo_df):
        cov_in, prec_in, acc_in = compute_pc(instanovo_df, **pc_kwargs)
    else:
        cov_in = prec_in = None; acc_in = None


    ap_pw = auc(cov_pw, prec_pw)
    ap_hy = auc(cov_hy, prec_hy)
    ap_ca = auc(cov_ca, prec_ca)
    ap_sm = auc(cov_sm, prec_sm)
    ap_cn = auc(cov_cn, prec_cn)
    ap_cnpp = auc(cov_cnpp, prec_cnpp) if cov_cnpp is not None else None
    ap_in = auc(cov_in, prec_in) if cov_in is not None else None
    n_pw, n_hy, n_ca, n_sm, n_cn = (
        len(cov_pw), len(cov_hy), len(cov_ca), len(cov_sm), len(cov_cn)
    )
    n_cnpp = len(cov_cnpp) if cov_cnpp is not None else None
    n_in = len(cov_in) if cov_in is not None else None
    print(f"  PowerNovo   AP={ap_pw:.4f}  precision={acc_pw:.4f}  n={n_pw}")
    print(f"  Casanovo    AP={ap_ca:.4f}  precision={acc_ca:.4f}  n={n_ca}")
    print(f"  SMSNet      AP={ap_sm:.4f}  precision={acc_sm:.4f}  n={n_sm}")
    print(f"  ContraNovo  AP={ap_cn:.4f}  precision={acc_cn:.4f}  n={n_cn}")
    if ap_cnpp is not None:
        print(f"  ContraNovo+PepPr AP={ap_cnpp:.4f}  precision={acc_cnpp:.4f}  n={n_cnpp}")
    if ap_in is not None:
        print(f"  InstaNovo   AP={ap_in:.4f}  precision={acc_in:.4f}  n={n_in}")
    print(f"  Casanovo+PepPr AP={ap_hy:.4f}  precision={acc_hy:.4f}  n={n_hy}")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(cov_hy, prec_hy, lw=2.2, color=COLOR_PP, ls="-",
            label=f"Casanovo+PepPr (AP={ap_hy:.3f}, n={n_hy})")
    ax.plot(cov_ca, prec_ca, lw=2.2, color=COLOR_CASANOVO, ls="-",
            label=f"Casanovo (AP={ap_ca:.3f}, n={n_ca})")
    ax.plot(cov_pw, prec_pw, lw=2.2, color=COLOR_CASANOVO, ls="--",
            label=f"PowerNovo (AP={ap_pw:.3f}, n={n_pw})")
    ax.plot(cov_sm, prec_sm, lw=2.2, color=COLOR_CASANOVO, ls=":",
            label=f"SMSNet (AP={ap_sm:.3f}, n={n_sm})")
    ax.plot(cov_cn, prec_cn, lw=2.2, color=COLOR_CASANOVO, ls="-.",
            label=f"ContraNovo (AP={ap_cn:.3f}, n={n_cn})")
    if cov_cnpp is not None:
        ax.plot(cov_cnpp, prec_cnpp, lw=2.2, color=COLOR_PP, ls="-.",
                label=f"ContraNovo+PepPr (AP={ap_cnpp:.3f}, n={n_cnpp})")
    if cov_in is not None:
        ax.plot(cov_in, prec_in, lw=2.2, color="#8c5a00", ls="-",
                label=f"InstaNovo (AP={ap_in:.3f}, n={n_in})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage", fontsize=13)
    ax.set_ylabel("Peptide precision", fontsize=13)
    ax.legend(fontsize=11, loc="lower left", framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)
    ax.grid(False)
    ax.tick_params(labelsize=12)
    fig.tight_layout()

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
