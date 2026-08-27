"""Prosit-based spectral angle.

Replaces the hand-rolled unit-height b/y theoretical spectrum with per-fragment
intensities predicted by Prosit_2020_intensity_HCD (via Koina). The SA formula
is unchanged — `1 - (2/π) * arccos(cos_sim(sqrt(int)))` — but the "theoretical"
vector is now the Prosit-predicted intensity at each fragment m/z.

Design notes
------------
- Prosit_2020_intensity_HCD supports only Carbamidomethyl(C) and Oxidation(M).
  Peptides carrying any other modification (deamidation, TMT, phospho, …) are
  intentionally *not* predicted here (SA → NaN); callers filter them out.
- NCE: no MGF in this project carries a per-scan collision energy, so we default
  to 30 unless the caller supplies one.
- Batching: predictions are memoised on disk (parquet) keyed by
  (proforma, charge, ce). First call warms the cache; subsequent runs are cheap.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from peppr.const import PROJECT_ROOT

CACHE_PATH = os.path.join(PROJECT_ROOT, "prosit_sa_cache.parquet")

KOINA_URL   = "koina.proteomicsdb.org:443"
KOINA_MODEL = "Prosit_2020_intensity_HCD"
DEFAULT_NCE = 30

MATCH_TOL_DA = 0.5

_NTERM_MOD_RE = re.compile(r"^([+\-]\d+(?:\.\d+)?)-(.*)")
_PROSIT_SEQUENCE_RE = re.compile(
    r"^(?:C\[Carbamidomethyl\]|M\[Oxidation\]|[ACDEFGHIKLMNPQRSTVWY])+$"
)
_PROSIT_MOD_NAMES = {
    "C[Carbamidomethyl]": "C[UNIMOD:4]",
    "M[Oxidation]": "M[UNIMOD:35]",
}
_MOD_RE = re.compile(r"\[[^\]]+\]")

# Vendored from depthcharge.primitives.Peptide.massivekb_to_proforma (depthcharge
# 0.4.8). DNPS_benchmark_casa pins casanovo, which requires
# `depthcharge-ms <0.3.0`, whose `primitives` module doesn't exist yet — so this
# small, self-contained conversion is copied in rather than upgrading the shared
# env's depthcharge and risking casanovo compatibility.
_MSKB_TO_UNIMOD = {
    "+42.011": "[Acetyl]-",
    "+43.006": "[Carbamyl]-",
    "-17.027": "[Ammonia-loss]-",
    "+43.006-17.027": "[+25.980265]-",  # Not in Unimod
    "M+15.995": "M[Oxidation]",
    "N+0.984": "N[Deamidated]",
    "Q+0.984": "Q[Deamidated]",
    "C+57.021": "C[Carbamidomethyl]",
}


def _massivekb_to_proforma(sequence: str) -> str:
    """Convert a MassIVE-KB peptide sequence to ProForma."""
    return "".join(
        _MSKB_TO_UNIMOD.get(aa, aa)
        for aa in re.split(r"(?<=.)(?=[A-Z])", sequence)
    )


def to_prosit_proforma(seq: str) -> Optional[str]:
    """Translate a MassIVE-KB peptide (e.g. `C+57.021ASGYTFTNYWIC+57.021WVK`)
    into the ProForma variant Prosit_2020_intensity_HCD accepts.

    Returns ``None`` if the peptide (a) carries an unsupported modification
    (anything besides Carbamidomethyl C / Oxidation M / N-terminal mods) or
    (b) is outside Prosit_2020_intensity_HCD's length window of 7–30 residues.
    """
    if not isinstance(seq, str) or not seq:
        return None
    if _NTERM_MOD_RE.match(seq):
        return None
    try:
        converted = _massivekb_to_proforma(seq)
    except (TypeError, ValueError):
        return None
    if not _PROSIT_SEQUENCE_RE.fullmatch(converted):
        return None
    if not (7 <= len(_MOD_RE.sub("", converted)) <= 30):
        return None
    for named_mod, unimod in _PROSIT_MOD_NAMES.items():
        converted = converted.replace(named_mod, unimod)
    return converted


def _load_cache() -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame(columns=[
        "peptide_sequences", "precursor_charges", "collision_energies",
        "mz", "intensities",
    ])


def _save_cache(df: pd.DataFrame) -> None:
    tmp = CACHE_PATH + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, CACHE_PATH)


def predict_batch(triples: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fetch Prosit predictions for every unique (peptide, charge, ce) in
    ``triples``. Results are memoised in ``CACHE_PATH``.

    ``triples`` must have columns ``peptide_sequences``, ``precursor_charges``,
    ``collision_energies``. Returns a DataFrame with the same keys plus ``mz``
    (np.ndarray[float32]) and ``intensities`` (np.ndarray[float32]).
    """
    triples = triples[["peptide_sequences", "precursor_charges", "collision_energies"]].drop_duplicates()
    triples = triples.reset_index(drop=True)

    cache = _load_cache()
    if len(cache):
        key_cols = ["peptide_sequences", "precursor_charges", "collision_energies"]
        merged = triples.merge(cache, on=key_cols, how="left", indicator=True)
        missing = merged[merged["_merge"] == "left_only"][key_cols].reset_index(drop=True)
    else:
        missing = triples
    if verbose:
        print(f"  Prosit cache: {len(triples) - len(missing):,}/{len(triples):,} hits, "
              f"{len(missing):,} misses", flush=True)

    if len(missing):
        from koinapy import Koina
        model = Koina(KOINA_MODEL, KOINA_URL)
        preds_long = model.predict(missing)
        preds = (preds_long
                 .groupby(["peptide_sequences", "precursor_charges", "collision_energies"])
                 .agg({"mz": list, "intensities": list})
                 .reset_index())
        preds["mz"] = preds["mz"].apply(lambda x: np.asarray(x, dtype=np.float32))
        preds["intensities"] = preds["intensities"].apply(lambda x: np.asarray(x, dtype=np.float32))
        cache = pd.concat([cache, preds], ignore_index=True)
        _save_cache(cache)

    # Return only the requested rows.
    return triples.merge(cache, on=["peptide_sequences", "precursor_charges",
                                     "collision_energies"], how="left")


def spectral_angle(mz_obs: np.ndarray, int_obs: np.ndarray,
                   mz_theo: np.ndarray, int_theo: np.ndarray,
                   tol: float = MATCH_TOL_DA) -> float:
    """Normalised spectral angle between observed and Prosit-predicted peaks.

    ``SA = 1 - (2/π) * arccos(cos_sim(sqrt(int_obs_matched), sqrt(int_theo)))``.
    For each predicted ion we take the intensity of the best-matching observed
    peak within ``tol`` (0 if no match). Prosit intensities are used directly
    for the theoretical vector — no re-normalisation (the cosine is scale
    invariant). Returns NaN if the observed spectrum or Prosit prediction is
    empty.
    """
    if (mz_obs is None or int_obs is None or mz_theo is None or int_theo is None
            or len(mz_obs) == 0 or len(mz_theo) == 0):
        return float("nan")
    # Drop any negative Prosit intensities (Prosit sometimes emits -1 for ions
    # it deems non-existent).
    keep = int_theo > 0
    if not keep.any():
        return float("nan")
    mz_theo = mz_theo[keep].astype(np.float64)
    int_theo = int_theo[keep].astype(np.float64)

    order = np.argsort(mz_obs)
    mz_obs_sorted = np.asarray(mz_obs, dtype=np.float64)[order]
    int_obs_sorted = np.asarray(int_obs, dtype=np.float64)[order]

    idx_lo = np.searchsorted(mz_obs_sorted, mz_theo - tol, side="left")
    idx_hi = np.searchsorted(mz_obs_sorted, mz_theo + tol, side="right")
    matched = np.zeros(len(mz_theo), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(idx_lo, idx_hi)):
        if lo >= hi:
            continue
        seg = int_obs_sorted[lo:hi]
        dists = np.abs(mz_obs_sorted[lo:hi] - mz_theo[i])
        matched[i] = seg[dists.argmin()]

    theo_vec = np.sqrt(int_theo)
    obs_vec  = np.sqrt(matched)
    n_o, n_t = np.linalg.norm(obs_vec), np.linalg.norm(theo_vec)
    if n_o == 0 or n_t == 0:
        return 0.0
    cos = float(np.clip(np.dot(obs_vec, theo_vec) / (n_o * n_t), -1.0, 1.0))
    return 1.0 - (2.0 / np.pi) * np.arccos(cos)


def compute_sa_for_psms(
    peptides: Iterable[str],
    charges: Iterable[int],
    mz_obs_list: Iterable[np.ndarray],
    int_obs_list: Iterable[np.ndarray],
    ces: Optional[Iterable[float]] = None,
    tol: float = MATCH_TOL_DA,
    verbose: bool = True,
) -> np.ndarray:
    """Batched SA over a set of PSMs. Peptides with unsupported mods return NaN.

    Parameters mirror the per-PSM inputs of the previous hand-rolled path.
    Predictions for unique (proforma, charge, ce) triples are fetched from
    Prosit once (with an on-disk cache), then applied to every PSM.
    """
    peptides = list(peptides)
    charges  = list(charges)
    mz_obs_list  = list(mz_obs_list)
    int_obs_list = list(int_obs_list)
    n = len(peptides)
    if ces is None:
        ces = [DEFAULT_NCE] * n
    else:
        ces = list(ces)

    proforma = [to_prosit_proforma(p) for p in peptides]
    # Prosit_2020_intensity_HCD supports precursor charges 1-6.
    keep = [(p is not None and c is not None and 1 <= int(c) <= 6)
            for p, c in zip(proforma, charges)]

    triples = pd.DataFrame({
        "peptide_sequences":  [proforma[i]   for i in range(n) if keep[i]],
        "precursor_charges":  [int(charges[i]) for i in range(n) if keep[i]],
        "collision_energies": [float(ces[i]) for i in range(n) if keep[i]],
    })
    preds = predict_batch(triples, verbose=verbose) if len(triples) else \
            pd.DataFrame(columns=["peptide_sequences", "precursor_charges",
                                   "collision_energies", "mz", "intensities"])
    pred_lookup = {
        (r.peptide_sequences, r.precursor_charges, r.collision_energies):
            (r.mz, r.intensities)
        for r in preds.itertuples(index=False)
    }

    sa_vals = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not keep[i]:
            continue
        key = (proforma[i], int(charges[i]), float(ces[i]))
        pred = pred_lookup.get(key)
        if pred is None:
            continue
        mz_theo, int_theo = pred
        sa_vals[i] = spectral_angle(mz_obs_list[i], int_obs_list[i],
                                     mz_theo, int_theo, tol=tol)
        if verbose and (i + 1) % 10_000 == 0:
            print(f"    SA {i + 1:,}/{n:,}", flush=True)
    return sa_vals
