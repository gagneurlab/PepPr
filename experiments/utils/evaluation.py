import os
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from experiments import paths
from pyteomics import mztab

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'casanovo'))
from casanovo.denovo import evaluate

# Exact monoisotopic residue masses from depthcharge 0.2.3.
CANONICAL_AA_MASSES = {
    "G": 57.021463735, "A": 71.037113805, "S": 87.032028435,
    "P": 97.052763875, "V": 99.068413945, "T": 101.047678505,
    "C": 103.009184505, "L": 113.084064015, "I": 113.084064015,
    "N": 114.04292747, "D": 115.026943065, "Q": 128.05857754,
    "K": 128.09496305, "E": 129.042593135, "M": 131.040484645,
    "H": 137.058911875, "F": 147.068413945, "R": 156.10111105,
    "Y": 163.063328575, "W": 186.07931298,
}

# MassIVE-KB tokens use carbamidomethylated cysteine rather than bare cysteine.
# Keys match the tokenization regex in evaluate.aa_match_batch.
MASSIVEKB_MASSES = {
    **{aa: mass for aa, mass in CANONICAL_AA_MASSES.items() if aa != "C"},
    "C+57.021": 160.030644505,
    "+42.011": 42.010565, "+43.006": 43.005814, "-17.027": -17.026549,
    "+43.006-17.027": 25.980265,
    "M+15.995": 147.03539964499998,
    "N+0.984": 115.02694346999999, "Q+0.984": 129.04259353999998,
    # TMT6plex (UniMod:737). N-term carries it as a standalone +229.163 token
    # after _normalize_to_massivekb; K+229.163 is added on demand by
    # _expand_masses. Required for the LUAD-finetuned Casanovo v5 model whose
    # vocab is TMT6plex on every lysine + N-term.
    "+229.163": 229.162932,
}

_PROFORMA_TO_MASSIVEKB = {
    '[Carbamidomethyl]': '+57.021',
    '[Oxidation]': '+15.995',
    '[Deamidated]': '+0.984',
    '[Acetyl]-': '+42.011',
    '[Carbamyl]-': '+43.006',
    '[Ammonia-loss]-': '-17.027',
    '[+25.980265]-': '+43.006-17.027',
    # N-term variant must precede the residue variant so [TMT6plex]- isn't
    # eaten by the substring [TMT6plex] rule.
    '[TMT6plex]-': '+229.163',
    '[TMT6plex]': '+229.163',
}

import re

_UNIMOD_TO_MASS_SHIFT = {
    1: '+42.011',    # Acetyl
    4: '+57.021',    # Carbamidomethyl
    5: '+43.006',    # Carbamyl (InstaNovo N-term vocab)
    7: '+0.984',     # Deamidated
    21: '+79.966',   # Phospho
    27: '-18.011',   # Pyro-glu from E
    28: '-17.027',   # Pyro-glu from Q
    34: '+14.016',   # Methyl
    35: '+15.995',   # Oxidation
    43: '+203.079',  # HexNAc
    121: '+114.043', # GlyGly (ubiquitinylation remnant)
    385: '-17.027',  # Ammonia loss (N-term)
}

def _unimod_replace(m):
    code = int(m.group(1))
    shift = _UNIMOD_TO_MASS_SHIFT.get(code)
    if shift is None:
        raise ValueError(f"Unknown UNIMOD code: {code}")
    return shift

_UNIMOD_RE = re.compile(r'\[UNIMOD:(\d+)\]-?')
_BRACKET_MOD_RE = re.compile(r'([A-Z]?)\[([+-]?\d+\.\d+)\](-?)')
# Casanovo v3 paren-style mod notation: C(+57.02), M(+15.99), N(+.98), Q(-17.03).
# Note the leading '.' in '+.98' — the digits-before-decimal are optional.
_PAREN_MOD_RE = re.compile(r'\(([+-]?\d*\.\d+)\)')

# Casanovo v3 emits 2-decimal mass shifts (+57.02), but MASSIVEKB_MASSES is
# keyed on the 3-decimal canonical form (+57.021). The +57.02 token would then
# fall back to a bare 'C' lookup that misses (Cys is never bare in casanovo's
# vocab), yielding mass = 57.02 instead of 160.03. Map the v3 shifts to the
# canonical MassIVE-KB form here so token lookup hits.
_V3_SHIFT_CANON = {
    '+57.02':  '+57.021',
    '+15.99':  '+15.995',
    '+.98':    '+0.984',
    '+0.98':   '+0.984',
    '-17.03':  '-17.027',
    '+42.01':  '+42.011',
    '+43.01':  '+43.006',
    '-18.01':  '-18.011',
    '+229.16': '+229.163',
}

def _normalize_to_massivekb(seq):
    """Normalize a peptide sequence (ProForma, UNIMOD, or mass-shift) to MassIVE-KB format.

    Handles five notations:
    - ProForma names:  C[Carbamidomethyl] -> C+57.021
    - UNIMOD codes:    C[UNIMOD:4]        -> C+57.021
    - Bracket shifts:  C[+57.021]         -> C+57.021  (strip brackets)
    - Paren shifts:    C(+57.02)          -> C+57.02   (Casanovo v3 / xanovo_v3 output)
    - Already MKB:     C+57.021           -> C+57.021  (no-op)
    N-terminal mods with trailing '-' are consumed: [UNIMOD:1]-M... -> +42.011M...
    Mass shifts that differ in decimal precision (e.g. +57.02 vs +57.021) are
    reconciled downstream by aa_match_batch's mass tolerance.
    """
    if seq is None or not isinstance(seq, str):
        return seq
    for proforma, mkb in _PROFORMA_TO_MASSIVEKB.items():
        seq = seq.replace(proforma, mkb)
    seq = _UNIMOD_RE.sub(_unimod_replace, seq)
    seq = _PAREN_MOD_RE.sub(
        lambda m: _V3_SHIFT_CANON.get(m.group(1), m.group(1)), seq)
    # Drop the trailing '-' on N-term mass shifts so the resulting token splits
    # cleanly via aa_match_batch's `(?<=.)(?=[A-Z])` rule into a leading
    # mass-shift token (e.g. '+229.163') followed by the first residue.
    # ProForma-name N-term mods like [Acetyl]- already drop the dash via the
    # _PROFORMA_TO_MASSIVEKB table; this brings bracket-mass N-term mods (the
    # form Krug's truth uses for TMT) into line.
    seq = _BRACKET_MOD_RE.sub(r'\1\2', seq)
    return seq

def _expand_masses(sequences, base_masses):
    """Return a mass dict extended with any AA+mass tokens found in sequences."""
    extra = {}
    for seq in sequences:
        if seq is None:
            continue
        for tok in re.split(r"(?<=.)(?=[A-Z])", seq):
            if tok not in base_masses and tok not in extra:
                m = re.match(r'^([A-Z])([+-]\d+\.\d+)$', tok)
                if m:
                    aa, shift = m.group(1), float(m.group(2))
                    extra[tok] = base_masses.get(aa, 0) + shift
    if not extra:
        return base_masses
    merged = dict(base_masses)
    merged.update(extra)
    return merged

def peptide_match_mass(preds, trues):
    """Mass-based peptide matching using casanovo's evaluate.aa_match_batch."""
    preds_norm = [_normalize_to_massivekb(p) if isinstance(p, str) else None for p in preds]
    trues_norm = [_normalize_to_massivekb(t) if isinstance(t, str) else None for t in trues]
    masses = _expand_masses(preds_norm + trues_norm, MASSIVEKB_MASSES)
    aa_matches_batch = evaluate.aa_match_batch(
        trues_norm, preds_norm, masses,
    )
    return np.array([aa_match[1] for aa_match in aa_matches_batch[0]])


def _parse_mgf_spectra(mgf_path):
    """Parse an MGF file and return a list of (title, seq) tuples."""
    spectra = []
    title, seq = None, None
    with open(mgf_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("TITLE="):
                title = line[6:]
            elif line.startswith("SEQ="):
                seq = line[4:]
            elif line == "END IONS":
                spectra.append((title, seq))
                title, seq = None, None
    return spectra


@lru_cache(maxsize=None)
def _resolve_archived_mgf(path: str) -> str:
    """Resolve stale mzTab locations against portable benchmark roots."""
    original = Path(path)
    if original.is_file():
        return str(original)
    roots = (Path(paths.NINE_SPECIES_PATH), Path(paths.BENCHMARKS_DIR))
    matches = []
    for root in roots:
        if root.is_dir():
            matches.extend(candidate for candidate in root.rglob(original.name)
                           if candidate.is_file())
    unique = sorted({candidate.resolve() for candidate in matches})
    if len(unique) == 1:
        return str(unique[0])
    if not unique:
        raise FileNotFoundError(
            f"MGF referenced by mzTab is unavailable: {path}. Set "
            "DNPS_NINE_SPECIES_PATH or verify DNPS_DATA_PATH."
        )
    raise RuntimeError(
        f"Ambiguous archived MGF basename {original.name!r}: "
        + ", ".join(map(str, unique))
    )


def _parse_ms_run_locations(mztab_path):
    """Extract ms_run[N] -> file path mapping from mztab metadata."""
    locations = {}
    with open(mztab_path) as f:
        for line in f:
            if line.startswith("MTD\tms_run["):
                parts = line.strip().split("\t")
                m = re.match(r'ms_run\[(\d+)\]-location', parts[1])
                if m:
                    run_num = int(m.group(1))
                    path = parts[2].replace("file://", "")
                    locations[run_num] = _resolve_archived_mgf(path)
            elif line.startswith("PSH\t"):
                break
    return locations


_SPECTRA_REF_RE = re.compile(r'ms_run\[(\d+)\]:index=(\d+)')


def average_precision(cov, prec, n_total=None):
    """Rectangular average precision matching ``sklearn.metrics.average_precision_score``.

    AP = Σ (R_n − R_{n−1}) · P_n, i.e. only precision at TP steps contributes,
    with R normalized by ``n_total`` (dataset size) rather than by the number
    of positives so cross-tool comparisons stay honest. If ``n_total`` is None,
    it is inferred from ``len(cov) / cov[-1]``.

    Drop-in replacement for ``sklearn.metrics.auc(cov, prec)`` when the caller
    wants the standard AP definition instead of trapezoidal AUC.
    """
    if len(prec) == 0:
        return 0.0
    cov = np.asarray(cov, dtype=np.float64)
    prec = np.asarray(prec, dtype=np.float64)
    if n_total is None:
        n_total = len(cov) / cov[-1] if cov[-1] > 0 else len(cov)
    N = len(prec)
    tp = prec * np.arange(1, N + 1)
    pep_match = np.round(np.diff(tp, prepend=0.0)).astype(np.int64)
    return float((pep_match * prec).sum() / n_total)


def count_mgf_spectra(mztab_path):
    """Return the total number of BEGIN IONS records across MGFs referenced
    by ``mztab_path``'s ms_run metadata. Used as the coverage denominator so
    PC-curve coverage reflects the full input dataset, not just the PSMs
    the tool emitted.
    """
    total = 0
    for mgf_path in _parse_ms_run_locations(mztab_path).values():
        if not os.path.exists(mgf_path):
            continue
        with open(mgf_path) as f:
            for line in f:
                if line.startswith("BEGIN IONS"):
                    total += 1
    return total

def load_mztab_with_mgf(mztab_path):
    """Load an mztab and join predictions with ground truth from MGF files.

    Works with two mztab formats:
    - Casanovo: spectra_ref = ms_run[N]:index=M → ground truth from MGF SEQ=
    - Modanovo: spectra_ref = filename → uses opt_correct_seq / opt_title columns

    Returns a DataFrame with columns: pred, score, true_seq, aa_scores,
    scans, spectra_ref.
    """
    ms_run_locs = _parse_ms_run_locations(mztab_path)
    mgf_cache = {}
    for run_num, mgf_path in ms_run_locs.items():
        if mgf_path not in mgf_cache:
            mgf_cache[mgf_path] = _parse_mgf_spectra(mgf_path)
        mgf_cache[run_num] = mgf_cache[mgf_path]

    rows = []
    with open(mztab_path) as f:
        header = None
        for line in f:
            if line.startswith("PSH\t"):
                header = line.strip().split("\t")
            elif line.startswith("PSM\t"):
                cols = line.strip().split("\t")
                row = dict(zip(header, cols))

                spectra_ref = row.get("spectra_ref")
                ref_match = _SPECTRA_REF_RE.match(spectra_ref or "")
                if ref_match:
                    run_num, idx = int(ref_match.group(1)), int(ref_match.group(2))
                    spectra = mgf_cache.get(run_num, [])
                    title, gt = spectra[idx] if idx < len(spectra) else (None, None)
                else:
                    gt = row.get("opt_correct_seq")
                    title = row.get("opt_title")

                pred = row.get("opt_ms_run[1]_proforma") or row["sequence"]
                score = float(row["search_engine_score[1]"])
                aa_scores = row.get("opt_ms_run[1]_aa_scores")
                rows.append({"pred": pred, "score": score, "true_seq": gt,
                             "aa_scores": aa_scores, "scans": title,
                             "spectra_ref": spectra_ref})

    return pd.DataFrame(rows)

COLORS = [paths.COLOR_CASANOVO, paths.COLOR_PP, paths.COLOR_XANOVO]

plt.style.use('ggplot')
plt.rcParams.update({
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'font.family': 'sans-serif',
})


def plot_position_accuracy(logits, labels, Y, title='Accuracy by Position in Sequence (C->N)'):
    """
    logits: (m, b, t, c)
    labels: string[m]
    Y: (b, t)
    Plots accuracy vs position for each model
    """
    m, _b, t, _c = logits.shape
    positions = range(t)
    
    plt.figure(figsize=(12, 6))
    _, predicted_classes = torch.max(logits, dim=3)
    ignore_mask = (Y != paths.VOCAB.index('-'))
    correct_predictions = (predicted_classes == Y.unsqueeze(0)).float()  # Shape: (m, b, t)
    masked_correct = correct_predictions * ignore_mask.unsqueeze(0)  # Shape: (m, b, t)
    position_accuracies = masked_correct.sum(dim=1) / ignore_mask.sum(dim=0).unsqueeze(0)  # Shape: (m, t)
    
    for model in range(m):
        plt.plot(positions, position_accuracies[model].numpy(), marker='o', linewidth=2, label=labels[model])
    
    plt.xlabel('Position', fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    plt.title(title, fontsize=13)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.8,
               edgecolor='gray', fancybox=False)
    plt.tick_params(labelsize=12)
    plt.grid(True, alpha=0.1)
    plt.ylim(-0.02, 1.02)
    plt.xlim(0, t-1)
    return plt


def load_mztab(path):
    df = load_mztab_with_mgf(path)
    df['idx'] = range(len(df))
    df['pred'] = df['pred'].apply(_normalize_to_massivekb)
    df['true_seq'] = df['true_seq'].apply(_normalize_to_massivekb)
    df['correct'] = peptide_match_mass(df['pred'].tolist(), df['true_seq'].tolist())
    df.set_index('idx', inplace=True)
    return df

def plot_peptide_pc(dfs, labels, mutation, title="Precision-Coverage"):
    if len(dfs) != len(labels):
        raise ValueError("Number of dataframes must match number of labels")
    
    for df, label in zip(dfs, labels):
        df['tool'] = label
 
    df = pd.concat(dfs, axis=0)
    _count_re = re.match(r'^(\d+)(\+?)\[(.+)\]$', mutation or '')
    if mutation is None:
        mutation = paths.ACTIVE_SPECIES
    elif mutation == "unmodified":
        df = df[~df['true_seq'].str.contains('+', regex=False)]
    elif _count_re:
        count_val = int(_count_re.group(1))
        at_least = _count_re.group(2) == '+'
        mod_str = _count_re.group(3)
        occurrences = df['true_seq'].str.count(re.escape(mod_str))
        if at_least:
            df = df[occurrences >= count_val]
            mutation = f"{count_val}+ × {mod_str}"
        else:
            df = df[occurrences == count_val]
            mutation = f"{count_val} × {mod_str}"
        if len(df) == 0:
            return None
    elif mutation is not None:
        df = df[df['true_seq'].str.contains(mutation, regex=False)]
        if len(df) == 0:
            return None
    
    n = len(df[df['tool'] == labels[0]])

    # Per-tool: sort by score, run aa_match_batch, compute precision/coverage (reference code pattern)
    tool_curves = {}
    auc_values = {}
    for label in labels:
        tdf = df[df['tool'] == label].sort_values('score', ascending=False).reset_index(drop=True)
        aa_matches_batch = evaluate.aa_match_batch(
            tdf['true_seq'].tolist(), tdf['pred'].tolist(), MASSIVEKB_MASSES,
        )
        peptide_matches = np.asarray([m[1] for m in aa_matches_batch[0]])
        precision = np.cumsum(peptide_matches) / np.arange(1, len(peptide_matches) + 1)
        coverage = np.arange(1, len(peptide_matches) + 1) / len(peptide_matches)
        tool_curves[label] = (coverage, precision)
        auc_values[label] = auc(coverage, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color in zip(labels, COLORS[:len(labels)]):
        coverage, precision = tool_curves[label]
        ax.plot(coverage, precision, color=color, lw=2.0, alpha=1.0,
                label=f"{label} (AUC={auc_values[label]:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage", fontsize=13, color="black")
    ax.set_ylabel("Peptide precision", fontsize=13, color="black")
    ax.set_title(f"{title} (n={n:,})", fontsize=13, color="black")
    ax.legend(fontsize=11, loc="lower left", framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.grid(False)
    ax.tick_params(labelsize=12, colors="black")
    fig.tight_layout()
    return fig

# ── Shared plotting helpers ───────────────────────────────────────────────
# Used by the figure-2/3/4 generator scripts (plot_figure_*.py) for the
# sequence-diff + annotated MS/MS spectrum panels.

import re as _re
import matplotlib.patches as _mpatches


_AA_RE = _re.compile(r"[^A-Z]")


def plot_sequence_diff(ax, true_seq, preds, boundary_pos=0, title=None,
                       highlight_positions=None, highlight_label=None,
                       true_sa=None, right_align=False, row_gap=0.0,
                       cell_notes=None):
    """Draw a sequence-diff visual onto `ax`: a "True" row above prediction
    rows, each residue as a boxed letter, with mismatches highlighted in red.

    Args:
        ax: matplotlib axes (will be cleared / configured)
        true_seq: ground-truth peptide (string of single-letter AAs)
        preds: list of (label, pred_seq) or (label, pred_seq, score) tuples.
               When score is provided it is drawn to the right of the row.
        boundary_pos: 1-based position of last V-residue (V/C boundary line is
                      drawn AFTER it).  Pass 0 to suppress the boundary line.
        title: optional panel title (drawn with ax.set_title).
        highlight_positions: list of 1-indexed positions to highlight across
                             every row with a thicker gold border (e.g.,
                             SAAV variant residues).
        highlight_label: optional text drawn above the highlighted column
                         (e.g., "G12D"). Only the first highlighted position
                         is labelled.
        right_align: when True, align every row to a common max width on the
                     RIGHT and compare residues from the C-terminus (use when
                     predictions differ in length but share the C-terminal
                     suffix). Default False (classic left-aligned, per-position).

    I↔L substitutions are treated as a match (isobaric, Casanovo folds I→L).
    """
    L = len(true_seq)
    hl_set = set(highlight_positions or [])

    def _norm(p):
        return (p[0], p[1], "compare", p[2] if len(p) > 2 else None)

    rows = [("True", true_seq, "neutral", None)] + [_norm(p) for p in preds]
    n_rows = len(rows)
    box_w, box_h = 1.0, 1.0
    pitch = box_h + row_gap        # vertical distance between row baselines
    cell_notes = cell_notes or {}
    # Grid width spans the longest row so longer predictions aren't clipped.
    W = max([len(r[1]) for r in rows])
    true_off = (W - L) if right_align else 0
    hl_cols = {true_off + (p - 1) for p in hl_set}

    for ri, (label, seq, mode, score) in enumerate(rows):
        y = (n_rows - 1 - ri) * pitch
        seq_off = (W - len(seq)) if right_align else 0
        ax.text(-0.4, y + box_h / 2, label, ha="right", va="center",
                fontsize=12, fontweight="bold")
        if ri == 0 and true_sa is not None:
            ax.text(W * box_w + 0.4, y + box_h / 2,
                    f"SA = {true_sa:.2f}",
                    ha="left", va="center", fontsize=10, color="#444")
        for c in range(W):
            si = c - seq_off
            ch = seq[si] if 0 <= si < len(seq) else " "
            if ch == " ":
                continue
            x = c * box_w
            if mode == "neutral":
                fc, tc = "#e8e8e8", "#222"
            else:
                ti = c - true_off
                t = true_seq[ti] if 0 <= ti < L else " "
                is_match = (ch == t) or ({ch, t} <= {"I", "L"})
                fc, tc = ("#e0f0e2", paths.COLOR_GREEN) if is_match else ("#fde0e1", "#a32a2c")
            is_hl = c in hl_cols
            ec = "#c69214" if is_hl else "black"
            lw = 2.4 if is_hl else 0.6
            ax.add_patch(_mpatches.Rectangle((x, y), box_w, box_h,
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         zorder=3 if is_hl else 1))
            ax.text(x + box_w / 2, y + box_h / 2, ch,
                    ha="center", va="center_baseline",
                    fontsize=11, fontweight="bold", color=tc,
                    zorder=4 if is_hl else 2)
        if score is not None:
            ax.text(W * box_w + 0.4, y + box_h / 2,
                    f"score = {score:.2f}",
                    ha="left", va="center", fontsize=10, color="#444")
        for (nri, npos), ntxt in cell_notes.items():
            if nri == ri:
                nx = (seq_off + npos) * box_w + box_w / 2
                ax.text(nx, y - 0.10, ntxt, ha="center", va="top",
                        fontsize=8.5, fontweight="bold", color="#b5651d")

    if hl_set and highlight_label:
        x_hl = (true_off + min(hl_set) - 1) * box_w + box_w / 2
        ax.text(x_hl, -0.25, highlight_label, ha="center", va="top",
                fontsize=10, fontweight="bold", color="#c69214")

    if boundary_pos:
        bx = (true_off + boundary_pos) * box_w
        ax.plot([bx, bx], [-0.05, (n_rows - 1) * pitch + box_h + 0.05],
                color="black", lw=2.2, ls="--")
        ax.text(bx, -0.25, "V/C boundary",
                ha="center", va="top", fontsize=10, fontstyle="italic")

    if title:
        ax.set_title(title, fontsize=11, loc="center", pad=8)

    has_scores = (
        any(len(r) > 3 and r[3] is not None for r in rows[1:])
        or true_sa is not None
    )
    right_pad = 4.5 if has_scores else 0.5
    ax.set_xlim(-3.5, W * box_w + right_pad)
    ax.set_ylim(-0.85, (n_rows - 1) * pitch + box_h + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_spectrum_on_ax(ax, mgf_path, scan, seq, fragment_tol_ppm=20):
    """Render an annotated MS/MS spectrum onto matplotlib `ax`.

    Annotates b/y ions (singly + doubly charged) at every matched peak,
    labelled with rotated text (90°) above each peak so labels don't collide
    horizontally.  Spectrum_utils' built-in labelling is suppressed.

    Args:
        ax: matplotlib axes
        mgf_path: path to an MGF file containing the spectrum
        scan: SCANS= value of the target spectrum
        seq: ProForma sequence used for fragment annotation
        fragment_tol_ppm: match tolerance for fragment ions
    """
    from pyteomics import mgf as _mgf_mod
    import spectrum_utils.spectrum as sus
    import spectrum_utils.plot as sup

    with _mgf_mod.read(mgf_path) as reader:
        spec = None
        for s in reader:
            if int(s["params"].get("scans", -1)) == scan:
                spec = s
                break
    if spec is None:
        raise ValueError(f"No spectrum with SCANS={scan} in {mgf_path}")

    params = spec["params"]
    spectrum = sus.MsmsSpectrum(
        identifier=f"scan={scan}",
        precursor_mz=float(params["pepmass"][0]),
        precursor_charge=int(params["charge"][0]),
        mz=spec["m/z array"],
        intensity=spec["intensity array"],
    )
    sup.colors["y"] = paths.COLOR_RED
    sup.colors["?"] = paths.COLOR_LIGHT_GRAY
    spectrum.annotate_proforma(
        seq, fragment_tol_mass=fragment_tol_ppm,
        fragment_tol_mode="ppm", ion_types="yb",
    )
    sup.spectrum(spectrum, ax=ax, grid=False, annot_fmt=lambda a: None)
    for line in ax.get_lines():
        line.set_linewidth(2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0%", "100%"])

    max_int = float(spectrum.intensity.max())
    for mz, intensity, anno in zip(spectrum.mz, spectrum.intensity,
                                    spectrum.annotation):
        if anno is None or str(anno) == "?":
            continue
        label = str(anno).split("/")[0]
        label = label.replace("^2", "²").replace("^3", "³")
        ion_char = label[0] if label and label[0] in "by" else "?"
        color = {
            "b": paths.COLOR_BLUE,
            "y": paths.COLOR_RED,
        }.get(ion_char, "#888")
        rel_int = intensity / max_int if max_int > 0 else 0
        ax.text(mz, rel_int + 0.01, label,
                ha="center", va="bottom", fontsize=7, color=color,
                rotation=90)


_AA_MONO_MASS = CANONICAL_AA_MASSES
_PROTON_MZ = 1.00728
_H2O_MASS = 18.01056

_B_ION_COLOR = paths.COLOR_BLUE
_Y_ION_COLOR = paths.COLOR_RED


def _ion_mz(residues, ion_type, idx):
    """Singly-charged b- or y-ion m/z for 1-indexed `idx` over `residues`."""
    L = len(residues)
    if ion_type == 'b':
        return sum(_AA_MONO_MASS[r] for r in residues[:idx]) + _PROTON_MZ
    elif ion_type == 'y':
        return (sum(_AA_MONO_MASS[r] for r in residues[L - idx:])
                + _H2O_MASS + _PROTON_MZ)
    raise ValueError(f"ion_type must be 'b' or 'y', got {ion_type!r}")


def annotate_missing_ion_gap(ax_spec, true_seq, ion_type, missing_idx,
                              observed_idx, color=None, y_arrow=0.6,
                              missing_label=None, expand_xlim=True):
    """Mark a missing diagnostic b- or y-ion that explains an ordering error.

    Draws a dashed vertical line at the theoretical m/z of
    ``{ion_type}_{missing_idx}``, an arrow to ``{ion_type}_{observed_idx}``,
    and labels the gap with the residues that fall between the two ions in
    ``true_seq``. Indices are 1-based.
    """
    import re
    residues = re.findall(r"[A-Z]", true_seq)
    L = len(residues)
    missing_mz = _ion_mz(residues, ion_type, missing_idx)
    observed_mz = _ion_mz(residues, ion_type, observed_idx)
    lo, hi = min(missing_idx, observed_idx), max(missing_idx, observed_idx)
    if ion_type == 'b':
        gap_residues = "".join(residues[lo:hi])
        default_color = _B_ION_COLOR
    else:
        gap_residues = "".join(residues[L - hi:L - lo])
        default_color = _Y_ION_COLOR
    color = color or default_color

    if expand_xlim:
        cur_lo, cur_hi = ax_spec.get_xlim()
        ax_spec.set_xlim(min(cur_lo, missing_mz - 25),
                         max(cur_hi, missing_mz + 25))

    ax_spec.axvline(missing_mz, color=color, ls="--", lw=1.0, alpha=0.75,
                    zorder=2)
    # Always draw the arrow in the direction of increasing fragment size
    # (smaller-fragment → larger-fragment), so it reads left-to-right in m/z.
    # For b, that's missing→observed; for y, observed→missing.
    lo_mz, hi_mz = sorted((missing_mz, observed_mz))
    ax_spec.annotate("", xy=(hi_mz, y_arrow), xytext=(lo_mz, y_arrow),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.4,
                                     shrinkA=3, shrinkB=3))
    ax_spec.text((missing_mz + observed_mz) / 2, y_arrow + 0.02, gap_residues,
                 ha="center", va="bottom", fontsize=11, color=color,
                 fontweight="bold")
    # The amino acid that the missing peak itself would have revealed —
    # for b_n, b_n - b_{n-1} = residue n; for y_n, y_n - y_{n-1} = residue
    # at position L-n+1. We only draw this label for terminal missing ions
    # (b1 reveals residue 1; y1 reveals residue L) because for non-terminal
    # positions the adjacent gap arrow already names the same residue.
    own_aa = (residues[missing_idx - 1] if ion_type == 'b'
              else residues[L - missing_idx])
    if missing_idx == 1:
        ax_spec.text(missing_mz, y_arrow + 0.02, own_aa,
                     ha="center", va="bottom", fontsize=11, color=color,
                     fontweight="bold",
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    if missing_label is None:
        sub = str(missing_idx).translate(str.maketrans("0123456789",
                                                         "₀₁₂₃₄₅₆₇₈₉"))
        missing_label = f"{ion_type}{sub} (missing)"
    ha = "left" if ion_type == 'b' else "right"
    xlo, xhi = ax_spec.get_xlim()
    nudge = (xhi - xlo) * 0.006 * (1 if ion_type == 'b' else -1)
    ax_spec.text(missing_mz + nudge, 0.97, missing_label, ha=ha, va="top",
                 fontsize=8, color=color, alpha=0.9, fontstyle="italic")


def annotate_missing_b_gap(ax_spec, true_seq, missing_position, observed_position,
                            **kw):
    """Backward-compatible alias — see ``annotate_missing_ion_gap``."""
    return annotate_missing_ion_gap(ax_spec, true_seq, 'b',
                                     missing_position, observed_position, **kw)


def annotate_precursor_readout(ax_spec, true_seq, precursor_mz, precursor_charge,
                                from_y_idx, y_arrow=0.6, color=None,
                                peak_color=paths.COLOR_PURPLE):
    """Pedagogical annotation: in a *complete* spectrum the N-terminal residue
    could be read as ``[M+H]⁺ − y_{from_y_idx}``.

    Draws a dashed 'phantom' precursor peak at the singly-protonated
    ``[M+H]⁺`` m/z (derived from the measured ``precursor_mz`` / charge) and a
    dashed arrow from the (missing) ``y_{from_y_idx}`` ion up to it, labelled
    with the residue that the gap reveals. Because both ions are singly
    charged, the m/z gap ``[M+H]⁺ − y_{from_y_idx}`` equals that residue's mass
    exactly. ``from_y_idx`` should be ``len(seq) - 1`` so the gap spans a single
    residue. The dashed styling marks this as the hypothetical readout that the
    missing peak would have enabled.
    """
    import re
    residues = re.findall(r"[A-Z]", true_seq)
    L = len(residues)
    color = color or _Y_ION_COLOR
    y_mz = _ion_mz(residues, 'y', from_y_idx)
    neutral = precursor_mz * precursor_charge - precursor_charge * _PROTON_MZ
    mh_plus = neutral + _PROTON_MZ
    revealed = "".join(residues[0:L - from_y_idx])

    cur_lo, cur_hi = ax_spec.get_xlim()
    ax_spec.set_xlim(min(cur_lo, y_mz - 25), max(cur_hi, mh_plus + 45))

    # Phantom (reference) precursor peak — dashed so it isn't mistaken for an
    # observed fragment. The real precursor is multiply charged; this is its
    # singly-protonated position, placed so the gap to y reads one residue.
    ax_spec.plot([mh_plus, mh_plus], [0, 1.0], color=peak_color, lw=2.0,
                 ls="--", alpha=0.85, zorder=2)
    ax_spec.text(mh_plus, 1.0, "precursor", ha="center", va="bottom",
                 fontsize=7, color=peak_color, fontweight="bold",
                 linespacing=0.95)

    # Dashed arrow y_{from_y_idx} → precursor, labelled with the revealed residue.
    ax_spec.annotate("", xy=(mh_plus, y_arrow), xytext=(y_mz, y_arrow),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.4,
                                     ls="--", shrinkA=3, shrinkB=3))
    ax_spec.text((y_mz + mh_plus) / 2, y_arrow + 0.02, revealed,
                 ha="center", va="bottom", fontsize=11, color=color,
                 fontweight="bold")
    return mh_plus


# ── Nine-species benchmark — shared layout constants ──────────────────────
# Canonical ordering used across the four supplementary figures
# (precision_by_length / precision_by_sa / pc_same / pc_cross).
NINE_SPECIES_ORDER = [
    "endoloripes", "mouse", "archaeon",
    "human", "honeybee", "cowpea",
    "tomato", "yeast", "bacillus",
]

# Display labels — botanically-and-shortened common names for plots.
NINE_SPECIES_LABELS = {
    "cowpea":      "ricebean",
    "archaeon":    "mmazei",
    "endoloripes": "clambacteria",
}


def species_label(sp: str) -> str:
    """Return the plot-friendly display name for a 9-species key."""
    return NINE_SPECIES_LABELS.get(sp, sp)


def wilson_ci(k, n, ci=0.95):
    """Two-sided Wilson 95% CI for a binomial proportion k/n.

    Wilson is better-behaved than normal approximation when k is near 0 or n.
    Returns (p_hat, lo, hi).  For n == 0, returns (nan, nan, nan).
    """
    import numpy as _np
    from scipy import stats as _stats
    if n is None or n == 0:
        return float("nan"), float("nan"), float("nan")
    z = float(_stats.norm.ppf(0.5 + ci / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * _np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return float(p), float(centre - half), float(centre + half)


def paired_median_delta_stats(x, y, n_boot=10000, ci=0.95, seed=42):
    """Stats for paired (x, y) data — typically Casanovo (x) and Casanovo+PP (y)
    final precisions across species.

    Returns dict with:
      median_delta  : median(y - x)
      ci_low, ci_high : bootstrap CI on the median Δ at the given level
      pvalue        : two-sided paired Wilcoxon signed-rank p-value of Δ != 0
    """
    import numpy as _np
    from scipy import stats as _stats
    x = _np.asarray(x, dtype=float)
    y = _np.asarray(y, dtype=float)
    delta = y - x
    n = len(delta)
    if n < 2:
        return dict(median_delta=float(_np.median(delta)) if n else float("nan"),
                    ci_low=float("nan"), ci_high=float("nan"), pvalue=float("nan"))
    rng = _np.random.default_rng(seed)
    boot = _np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = _np.median(delta[idx])
    alpha = 1.0 - ci
    ci_low, ci_high = _np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    try:
        pvalue = float(_stats.wilcoxon(delta).pvalue)
    except ValueError:
        # All-zero diffs etc.; fall back to one-sample t on delta vs 0.
        pvalue = float(_stats.ttest_1samp(delta, 0.0).pvalue)
    return dict(median_delta=float(_np.median(delta)),
                ci_low=float(ci_low), ci_high=float(ci_high),
                pvalue=pvalue)


# Shared layout for the 3×3 nine-species supplementary grids.  Same panel
# size + line width + marker size + tick fontsize across all four supp figs.
NINE_SPECIES_NCOLS = 3
NINE_SPECIES_NROWS = 3
NINE_SPECIES_PANEL_INCHES = (4.0, 4.0)   # (width, height) per panel
NINE_SPECIES_LINE_WIDTH = 2.0
NINE_SPECIES_MARKER_SIZE = 5
NINE_SPECIES_TICK_FONTSIZE = 10
NINE_SPECIES_LABEL_FONTSIZE = 11
NINE_SPECIES_TITLE_FONTSIZE = 12

# Casanovo vs Casanovo+PP colours used in all four supp figs.
NINE_SPECIES_COLOR_DNPS = paths.COLOR_CASANOVO
NINE_SPECIES_COLOR_PP = paths.COLOR_PP
