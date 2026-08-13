#!/usr/bin/env python3
"""ProteomeTools-SAAV Panel A curve arrays.

`load_panelA_curves()` returns the precomputed `panelA_curves.npz` (shipped in the
Zenodo archive under `benchmarks/proteometools_saav/`), building it from the
canonical Casanovo/Casanovo+PepPr mzTabs and the MSFragger DMO PSMs if absent.
Keeping it in an npz lets the figure functions be thin, fast renderers with no
pepXML/mgf parsing at plot time.

Consumers:
  * plot_figure_3.plot_panel_A — main panel: Casanovo, Casanovo+PepPr, and the
    MSFragger DMO site-localized curve (Percolator-ranked) + the 1% PSM-FDR point.
  * supp/plot_panelA_supp.py — DMO localization/ranking supplement: the same DMO
    correctness ranked by PTM-Prophet localization score, the DMO peptide-level
    (localization-agnostic) curve, and Casanovo with adjacent-swap leniency.

Run `python experiments/fig3/panelA_curves.py` to (re)build the npz explicitly.
"""
import os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from experiments.utils.evaluation import (load_mztab_with_mgf, _normalize_to_massivekb, evaluate,
                                 MASSIVEKB_MASSES, count_mgf_spectra)
from peptide_priors.const import PROTEOMETOOLS_SAAV_DIR, result_run_path
import experiments.fig3.build_dmo_curves as bd

PANELA_NPZ = os.path.join(PROTEOMETOOLS_SAAV_DIR, "panelA_curves.npz")
_tok = lambda s: re.split(r"(?<=.)(?=[A-Z])", s)


def _denovo_curve(mz, n_total, lenient=False):
    df = load_mztab_with_mgf(mz)
    T = [_normalize_to_massivekb(r["true_seq"]) for _, r in df.iterrows() if r["true_seq"]]
    P = [_normalize_to_massivekb(r["pred"]) for _, r in df.iterrows() if r["true_seq"]]
    S = np.array([float(r["score"]) for _, r in df.iterrows() if r["true_seq"]])
    ok = np.array([1 if m[1] else 0 for m in evaluate.aa_match_batch(T, P, MASSIVEKB_MASSES)[0]])
    if lenient:
        ft, fp, ow = [], [], []
        for i in range(len(P)):
            if ok[i]:
                continue
            tk = _tok(P[i])
            for j in range(len(tk) - 1):
                sw = tk[:]; sw[j], sw[j + 1] = sw[j + 1], sw[j]
                ft.append(T[i]); fp.append("".join(sw)); ow.append(i)
        if ft:
            fok = [m[1] for m in evaluate.aa_match_batch(ft, fp, MASSIVEKB_MASSES)[0]]
            for k, i in enumerate(ow):
                if fok[k]:
                    ok[i] = 1
    o = np.argsort(-S, kind="mergesort")
    return np.arange(1, len(ok) + 1) / n_total, np.cumsum(ok[o]) / np.arange(1, len(ok) + 1)


def _dmo_curves(n_total):
    """MSFragger DMO curves. Main figure uses SITE-LOCALIZED ranked by Percolator
    (localization = best available: PTMProphet where present, else MSFragger
    localize_delta_mass, via psm['site_loc']). Also stored for the supplement:
    site-localized ranked by PTM-Prophet best-loc probability, and PEPTIDE-LEVEL
    (localization ignored — offset placed at any valid residue that yields GT)."""
    truth = bd.load_truth(); psms = list(bd.parse_psms(truth))
    sp = [bd.build_pred(p, p["site_loc"] if abs(p["massdiff"]) > 0.01 else None) for p in psms]
    site_ok = np.array([1 if m[1] else 0 for m in
                        evaluate.aa_match_batch([p["true"] for p in psms], sp, MASSIVEKB_MASSES)[0]])
    # peptide-level (localization-agnostic): correct if ANY valid placement matches GT
    ft, fp, ow = [], [], []
    for idx, p in enumerate(psms):
        if abs(p["massdiff"]) <= 0.01:
            ft.append(p["true"]); fp.append(bd.build_pred(p, None)); ow.append(idx); continue
        wm = bd.wt_masses(p)
        cand = [i for i in range(len(wm))
                if abs(MASSIVEKB_MASSES[bd.resolve_tok(wm[i] + p["massdiff"])] - (wm[i] + p["massdiff"])) < 0.03] \
            or [p["site_loc"] if p["site_loc"] is not None else 0]
        for i in cand:
            ft.append(p["true"]); fp.append(bd.build_pred(p, i)); ow.append(idx)
    fok = np.array([1 if m[1] else 0 for m in evaluate.aa_match_batch(ft, fp, MASSIVEKB_MASSES)[0]])
    pep_ok = np.zeros(len(psms), int)
    for k, idx in enumerate(ow):
        if fok[k]:
            pep_ok[idx] = 1
    perc = np.array([p["perc"] for p in psms]); ptm = np.array([p["ptm_prob"] for p in psms])
    rank = np.arange(1, len(perc) + 1)
    op = np.argsort(-perc, kind="mergesort"); ot = np.argsort(-ptm, kind="mergesort")
    out = dict(
        dmo_site_cov=rank / n_total, dmo_site_prec=np.cumsum(site_ok[op]) / rank,          # perc-ranked (main)
        dmo_site_ptm_cov=rank / n_total, dmo_site_ptm_prec=np.cumsum(site_ok[ot]) / rank,  # loc-score-ranked (suppl.)
        dmo_pep_cov=rank / n_total, dmo_pep_prec=np.cumsum(pep_ok[op]) / rank,             # peptide-level (suppl.)
    )
    pep = 1.0 - perc[op]; q = np.cumsum(pep) / rank
    below = np.where(q <= 0.01)[0]
    if len(below):
        k = below[-1] + 1
        out["fdr_cov"] = k / n_total
        out["fdr_site_prec"] = site_ok[op][:k].mean()
        out["fdr_pep_prec"] = pep_ok[op][:k].mean()
        out["fdr_thr"] = float(perc[op][k - 1])
    return out


def build_panelA_curves():
    """Recompute every Panel A curve from the canonical mzTabs + DMO PSMs and write
    the npz. Returns the npz path."""
    res = result_run_path("casanovo")
    dnps = os.path.join(res, "proteometools_saav_dnps.mztab")
    hyb = os.path.join(res, "proteometools_saav_hybrid.mztab")
    n = count_mgf_spectra(dnps)
    d = {"n_total": n}
    d["cas_cov"], d["cas_prec"] = _denovo_curve(dnps, n, lenient=False)
    d["casswap_cov"], d["casswap_prec"] = _denovo_curve(dnps, n, lenient=True)
    d["pp_cov"], d["pp_prec"] = _denovo_curve(hyb, n, lenient=False)
    d.update(_dmo_curves(n))
    os.makedirs(os.path.dirname(PANELA_NPZ), exist_ok=True)
    np.savez(PANELA_NPZ, **d)
    return PANELA_NPZ


def load_panelA_curves(rebuild=False):
    """Return the Panel A curves as an npz mapping, building it if missing."""
    if rebuild or not os.path.exists(PANELA_NPZ):
        build_panelA_curves()
    return np.load(PANELA_NPZ)


if __name__ == "__main__":
    build_panelA_curves()
    d = np.load(PANELA_NPZ)
    print("saved", PANELA_NPZ)
    print(f"n_total={int(d['n_total'])}  fdr_cov={d['fdr_cov'] if 'fdr_cov' in d.files else None}"
          f" fdr_site={float(d['fdr_site_prec']):.3f}" if "fdr_site_prec" in d.files else "")
