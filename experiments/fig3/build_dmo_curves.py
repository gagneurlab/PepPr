#!/usr/bin/env python3
"""ProteomeTools SAAV DMO curves: human_iso DMO + PTMProphet + Percolator search
(proteometools_saav DMO/interact-*.mod.pep.xml) vs GT-search truth (mgf_gt
SEQ), on the same GT-passing spectra the de novo models run on.

Two curves, both scored by Casanovo's evaluate.aa_match_batch (as de novo curves):
  SITE-AWARE    offset at PTMProphet best-localized residue; rank by best-loc prob.
  PEPTIDE-LEVEL any placement matches (localization ignored); rank by Percolator.
Coverage denom = # GT spectra.
"""
import os, re, glob, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from experiments.utils.evaluation import _normalize_to_massivekb, MASSIVEKB_MASSES
from experiments.utils.evaluation import peptide_match_mass  # noqa
from experiments.paths import PROTEOMETOOLS_SAAV_DIR
from casanovo.denovo import evaluate  # noqa

DMO = os.path.join(PROTEOMETOOLS_SAAV_DIR, "msfragger_dmo")
MGF_GT = os.path.join(PROTEOMETOOLS_SAAV_DIR, "mgf_gt")
_C_CAM = MASSIVEKB_MASSES["C+57.021"]
_SQ = re.compile(r"<spectrum_query\b.*?</spectrum_query>", re.S)
_PTMRES = re.compile(r'<ptmprophet_result ptm="[^:]+:([-\d.]+)"(.*?)</ptmprophet_result>', re.S)
_MODPROB = re.compile(r'<mod_aminoacid_probability position="(\d+)" probability="([-+\d.eE]+)"')
_PTMR = re.compile(r'<ptm_result\b([^>]*)>')


def load_truth():
    d = {}
    for f in glob.glob(os.path.join(MGF_GT, "*.mgf")):
        ba = os.path.basename(f)[:-4]
        seq = scan = None
        for line in open(f):
            if line.startswith("BEGIN IONS"):
                seq = scan = None
            elif line.startswith("SEQ="):
                seq = line[4:].strip()
            elif line.startswith("SCANS="):
                scan = int(line[6:])
            elif line.startswith("END IONS") and seq is not None and scan is not None:
                d[(ba, scan)] = _normalize_to_massivekb(seq)
    return d


def resolve_tok(m):
    return min(MASSIVEKB_MASSES, key=lambda t: abs(MASSIVEKB_MASSES[t] - m))


def _ptm_bestloc(block, massdiff):
    best = None
    for m in _PTMRES.finditer(block):
        if abs(float(m.group(1)) - massdiff) > 0.05:
            continue
        probs = [(int(p) - 1, float(pr)) for p, pr in _MODPROB.findall(m.group(2))]
        if not probs:
            continue
        loc, pr = max(probs, key=lambda x: x[1])
        if best is None or pr > best[0]:
            best = (pr, loc)
    return best if best else (0.0, None)


def _msfrag_loc(hit):
    """MSFragger localize_delta_mass site (0-indexed) from the ptm_result element:
    argmax position_score among best_positions. Fallback used when PTMProphet, which
    only localizes the param-converted offset subset (~71%), has no result."""
    m = _PTMR.search(hit)
    if not m:
        return None
    attrs = m.group(1)
    bp = re.search(r'best_positions="([^"]*)"', attrs)
    ps = re.search(r'position_scores="([^"]*)"', attrs)
    if not bp:
        return None
    cands = [int(x) - 1 for x in re.findall(r'[A-Za-z](\d+)', bp.group(1))]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    scores = [float(s) for s in re.findall(r'\(([\d.]+)\)', ps.group(1))] if ps else []
    return max(cands, key=lambda i: scores[i]) if len(scores) > max(cands) else cands[0]


def parse_psms(truth):
    for f in sorted(glob.glob(os.path.join(DMO, "interact-*.mod.pep.xml"))):
        ba = os.path.basename(f)[len("interact-"):-len(".mod.pep.xml")]
        txt = open(f, encoding="utf-8", errors="replace").read()
        for mq in _SQ.finditer(txt):
            b = mq.group(0)
            ms = re.search(r'spectrum="[^"]*\.(\d+)\.\d+\.\d+"', b)
            if not ms:
                continue
            scan = int(ms.group(1))
            if (ba, scan) not in truth:
                continue
            sh = re.search(r"<search_hit\b.*?</search_hit>", b, re.S)
            if not sh:
                continue
            hit = sh.group(0)
            mp = re.search(r'peptide="([A-Z]+)"', hit)
            mm = re.search(r'massdiff="([-\d.eE]+)"', hit)
            pr = re.search(r'<peptideprophet_result probability="([-+\d.eE]+)"', b)
            if not (mp and mm and pr):
                continue
            pep, massdiff, perc = mp.group(1), float(mm.group(1)), float(pr.group(1))
            mods, saav_pos, best_var = {}, None, 0.05
            for tag in re.finditer(r'<mod_aminoacid_mass\b[^>]*>', hit):
                pm = re.search(r'position="(\d+)"', tag.group(0))
                m2 = re.search(r'mass="([-\d.eE]+)"', tag.group(0))
                vv = re.search(r'variable="([-\d.eE]+)"', tag.group(0))
                if not (pm and m2):
                    continue
                i = int(pm.group(1)) - 1
                mods[i] = float(m2.group(1))
                if vv and abs(massdiff) > 0.01 and abs(float(vv.group(1)) - massdiff) < best_var:
                    best_var, saav_pos = abs(float(vv.group(1)) - massdiff), i
            best_prob, best_loc = (1.0, None)
            site_loc = None
            if abs(massdiff) > 0.01:
                best_prob, best_loc = _ptm_bestloc(b, massdiff)
                # best-available localization: PTMProphet (param-mod subset, ~71%),
                # else MSFragger localize_delta_mass, else MSFragger's assigned pos.
                site_loc = best_loc
                if site_loc is None:
                    site_loc = _msfrag_loc(hit)
                if site_loc is None:
                    site_loc = saav_pos
            yield dict(ba=ba, scan=scan, pep=pep, massdiff=massdiff, perc=perc,
                       mods=mods, saav_pos=saav_pos, ptm_prob=best_prob,
                       ptm_loc=best_loc, site_loc=site_loc, true=truth[(ba, scan)])


def wt_masses(psm):
    masses = [(_C_CAM if c == "C" else MASSIVEKB_MASSES.get(c, 0.0)) for c in psm["pep"]]
    for i, m in psm["mods"].items():
        if i != psm["saav_pos"] and 0 <= i < len(masses):
            masses[i] = m
    return masses


def build_pred(psm, pos):
    masses = wt_masses(psm)
    if pos is not None and 0 <= pos < len(masses) and abs(psm["massdiff"]) > 0.01:
        masses[pos] += psm["massdiff"]
    return "".join(resolve_tok(m) for m in masses)


def auc(x, y):
    return float(np.trapz(y, x)) if len(x) else 0.0


def main():
    truth = load_truth()
    n_total = len(truth)
    print(f"GT eval spectra: {n_total}", flush=True)
    psms = list(parse_psms(truth))
    print(f"DMO rank-1 PSMs on GT spectra: {len(psms)} (coverage {len(psms)/n_total:.3f})", flush=True)

    # SITE-AWARE. Place the offset at PTMProphet's best-localized residue; rank by its
    # best-loc probability. When PTMProphet emits no localization (offset not converted to a
    # var mod, e.g. -57 CAM loss on Cys) we predict the WT backbone (pos=None) — correct for
    # the net-zero cases where the DB peptide already equals GT, and a genuine (~1%) site
    # miss for the handful of real-but-unlocalized SAAVs (a fallback that force-places the
    # offset was tested and is net-negative: it breaks more net-zero cases than it recovers).
    site_preds = [build_pred(p, p["ptm_loc"] if abs(p["massdiff"]) > 0.01 else None) for p in psms]
    site_trues = [p["true"] for p in psms]
    site_scores = np.array([p["ptm_prob"] for p in psms])
    site_ok = np.array([1 if m[1] else 0 for m in
                        evaluate.aa_match_batch(site_trues, site_preds, MASSIVEKB_MASSES)[0]])
    o = np.argsort(-site_scores, kind="mergesort")
    site_prec = np.cumsum(site_ok[o]) / np.arange(1, len(o) + 1)
    site_cov = np.arange(1, len(o) + 1) / n_total

    # PEPTIDE-LEVEL (any valid placement)
    flat_trues, flat_preds, owner = [], [], []
    for idx, p in enumerate(psms):
        if abs(p["massdiff"]) <= 0.01:
            flat_trues.append(p["true"]); flat_preds.append(build_pred(p, None)); owner.append(idx); continue
        wm = wt_masses(p)
        cand = [i for i in range(len(wm))
                if abs(MASSIVEKB_MASSES[resolve_tok(wm[i] + p["massdiff"])] - (wm[i] + p["massdiff"])) < 0.03]
        if not cand:
            cand = [p["ptm_loc"]] if p["ptm_loc"] is not None else [0]
        for i in cand:
            flat_trues.append(p["true"]); flat_preds.append(build_pred(p, i)); owner.append(idx)
    flat_ok = np.array([1 if m[1] else 0 for m in
                        evaluate.aa_match_batch(flat_trues, flat_preds, MASSIVEKB_MASSES)[0]])
    pep_ok = np.zeros(len(psms), dtype=int)
    for k, idx in enumerate(owner):
        if flat_ok[k]:
            pep_ok[idx] = 1
    pep_scores = np.array([p["perc"] for p in psms])
    o2 = np.argsort(-pep_scores, kind="mergesort")
    pep_prec = np.cumsum(pep_ok[o2]) / np.arange(1, len(psms) + 1)
    pep_cov = np.arange(1, len(psms) + 1) / n_total

    print(f"\nSITE-AWARE   (rank PTMProphet): correct {site_ok.sum()}/{len(psms)} "
          f"prec {site_ok.mean():.4f} maxcov {site_cov[-1]:.4f} AP {auc(site_cov,site_prec):.4f}")
    print(f"PEPTIDE-LEVEL(rank Percolator): correct {pep_ok.sum()}/{len(psms)} "
          f"prec {pep_ok.mean():.4f} maxcov {pep_cov[-1]:.4f} AP {auc(pep_cov,pep_prec):.4f}")
    for c in (0.2, 0.4, 0.5, 0.6, 0.7):
        def at(cov, prec):
            k = np.searchsorted(cov, c); return prec[min(k, len(prec) - 1)] if len(prec) else float("nan")
        print(f"  cov{c}: site {at(site_cov,site_prec):.3f}  pep {at(pep_cov,pep_prec):.3f}")

    np.savez("dmo_curves.npz",
             site_cov=site_cov, site_prec=site_prec, pep_cov=pep_cov, pep_prec=pep_prec,
             site_scores=site_scores[o], pep_scores=pep_scores[o2])
    print("saved dmo_curves.npz")


if __name__ == "__main__":
    main()
