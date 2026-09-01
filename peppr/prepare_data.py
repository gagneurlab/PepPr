import argparse
import os
import hashlib
import json
import re
import glob
import random
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from peppr import const
from typing import Optional, Tuple, TextIO
from collections import defaultdict
from tqdm import tqdm
import pygtrie
import pickle
from peppr.proforma import massivekb_to_proforma

try:
    from pyteomics.proforma import parse as _proforma_parse
    from pyteomics.proforma import ProFormaError as _ProFormaError
except ImportError:
    _proforma_parse = None
    _ProFormaError = None



_MASS_TO_PROFORMA = {
    "C[+57.021]": "C[Carbamidomethyl]",
    "M[+15.995]": "M[Oxidation]",
    "N[+0.984]": "N[Deamidated]",
    "Q[+0.984]": "Q[Deamidated]",
    "[+42.011]-": "[Acetyl]-",
    "K[+229.163]": "K[TMT6plex]",
    "[+229.163]-": "[TMT6plex]-",
    "K[+304.207]": "K[TMTpro]",
    "[+304.207]-": "[TMTpro]-",
}

# Casanovo v5 default vocabulary + TMT6plex. Used by prepare_luad_casanovo_ft
# to drop PSMs whose ProForma SEQ contains any modification outside this set
# (e.g. Phospho, HexNAc, GlyGly, Methyl, Pro-hydroxylation, ...).
_CASANOVO_TMT_ALLOWED_MODS = frozenset({
    "Carbamidomethyl",
    "Oxidation",
    "Deamidated",
    "Acetyl",
    "Carbamyl",
    "Ammonia-loss",
    "+25.980265",
    "TMT6plex",
    # TMTpro16plex (UNIMOD:2016) — PXD033643 fine-tune.
    "TMTpro",
})

def _normalize_mass_shifts(seq: str) -> str:
    for mass, name in _MASS_TO_PROFORMA.items():
        seq = seq.replace(mass, name)
    return seq

def _convert_seq_to_proforma(seq_value: str, validate: bool = True) -> str:
    """
    Convert SEQ= value to ProForma. If validate=True and the result fails to parse
    (e.g. ProFormaError 'Missing Closing Tag'), fall back to plain amino acid sequence.
    """
    converted = massivekb_to_proforma(seq_value)
    converted = _normalize_mass_shifts(converted)
    if validate:
        try:
            _proforma_parse(converted)
        except _ProFormaError:
            print(f"Failed to parse {seq_value} as ProForma, falling back to plain sequence")
            plain = re.sub(r"\[.*?\]", "", converted)
            plain = re.sub(r"^[+-]?[\d.]+-?", "", plain)  # n-term mass mods
            plain = re.sub(r"/\d+$", "", plain)  # charge suffix
            plain = "".join(c for c in plain if c in "ACDEFGHIKLMNPQRSTVWY")
            if plain:
                converted = plain
    return converted

itos = { i:ch for i,ch in enumerate(const.VOCAB) }
# Start token isn't part of the vocabulary
itos[const.START_TOKEN] = '^'

stoi = { ch:i for i,ch in enumerate(const.VOCAB) }
# Treat Selenocysteine as Cysteine
stoi['U'] = stoi['C']
# Treat Isoleucine as Leucine (isobaric).
stoi['I'] = stoi['L']

def encode(s):
    return np.array([stoi[c] for c in s], dtype=np.uint16) # encoder: take a string, output a list of integers

def digest_protein(sequence, max_missed_cleavages, methionine_excision, reverse=True):
    """Generate tryptic peptides, cleaving after K/R with optional missed cleavages."""
    max_len = int(os.environ.get("PEPPR_PRIOR_MAX_PEP_LEN", const.PRIOR_BLOCK_SIZE))
    sites = sorted({0, len(sequence), *(m.start() + 1 for m in re.finditer("[KR]", sequence))})
    seen = set()
    peptides = []
    for i in range(len(sites) - 1):
        for j in range(i + 1, min(i + 2 + max_missed_cleavages, len(sites))):
            start, end = sites[i], sites[j]
            peptide = sequence[start:end]
            if reverse:
                peptide = peptide[::-1]
            if not (3 < len(peptide) <= max_len):
                continue
            tok = peptide + "$"
            if tok not in seen:
                seen.add(tok)
                peptides.append(tok)
            if methionine_excision:
                if len(peptide) > 5 and peptide[-2] == "M" and peptide[-1] == ".":
                    tok2 = peptide[:-2] + ".$"
                    if tok2 not in seen:
                        seen.add(tok2)
                        peptides.append(tok2)
                elif len(peptide) > 4 and peptide[-1] == "M":
                    tok2 = peptide[:-1] + "$"
                    if tok2 not in seen:
                        seen.add(tok2)
                        peptides.append(tok2)
    return peptides

def sliding_window_peptides(sequence: str, lmin: int = 6, lmax: int = 25,
                             reverse: bool = True,
                             cdr3_range: tuple[int, int] | None = None) -> list:
    """Enumerate every sliding L-mer (L in [lmin, lmax]) in `sequence`.
    Caller is responsible for dedup + corpus-wide budget control — see
    `generate_plm_training_data` which shuffles records and stops when the
    dedup set reaches the buffer cap.

    Honors PEPPR_PRIOR_MIN_PEP_LEN and PEPPR_PRIOR_MAX_PEP_LEN as bounds.
    Returns peptides with `$` terminator, reverse-C-to-N to match `digest_protein`.

    If ``cdr3_range`` is given (1-based inclusive positions in ``sequence``),
    also returns a parallel list of CDR3 masks: per-peptide list of booleans
    of the same length as the encoded peptide (with the `$` terminator slot
    set to False), True where the residue lies inside the CDR3 span. The
    mask is in the same orientation as the returned peptide (reversed too,
    if ``reverse=True``). When ``cdr3_range`` is None, returns peptides only
    (backwards-compatible).
    """
    cap = int(os.environ.get("PEPPR_PRIOR_MAX_PEP_LEN", lmax))
    floor = int(os.environ.get("PEPPR_PRIOR_MIN_PEP_LEN", lmin))
    lmax = min(lmax, cap)
    lmin = max(lmin, floor)
    cdr3_lo, cdr3_hi = (cdr3_range if cdr3_range is not None else (-1, -1))
    return_masks = cdr3_range is not None
    out = []
    out_masks = []
    n = len(sequence)
    for L in range(lmin, lmax + 1):
        for s in range(n - L + 1):
            peptide = sequence[s:s + L]
            if return_masks:
                # forward-orientation CDR3 mask: True where (1-based position
                # s+i+1) lies in [cdr3_lo, cdr3_hi]
                mask = [(s + i + 1) >= cdr3_lo and (s + i + 1) <= cdr3_hi
                        for i in range(L)]
            if reverse:
                peptide = peptide[::-1]
                if return_masks:
                    mask = mask[::-1]
            out.append(peptide + "$")
            if return_masks:
                out_masks.append(mask + [False])   # `$` position is never CDR3
    if return_masks:
        return out, out_masks
    return out


# Conserved motifs at the FR3/CDR3 and CDR3/FR4 boundaries of antibody V-regions.
# Used by ``_find_cdr3_range`` to locate the CDR3 within a germline corpus record
# without needing a per-record annotation file.
#
# FR3-end Cys-104 anchor: a conserved Cys preceded by Y, F, or another residue
# (variants `YYC`, `YFC`, `HYC`, `IYFC`, `MYFC` etc. across HC + LC germlines).
# We just require `C` preceded by some residue; the FR3-end Cys is uniquely
# the LAST `[A-Z]C` before the FR4 anchor (V-region has at most two Cys, at
# IMGT positions 23 and 104; the latter is what we want).
# FR4 starts with WG (HC) or FG (kappa/lambda LC) followed by X-GT.
_CDR3_FR3_END_RE = re.compile(r"[A-Z]C")
_CDR3_FR4_START_RE = re.compile(r"[WF]G[A-Z]GT")


def _find_cdr3_range(seq: str) -> tuple[int, int] | None:
    """Locate CDR3 in an antibody record. Returns (start, end) 1-based
    inclusive positions, or None if either boundary motif isn't found."""
    m_fr4 = _CDR3_FR4_START_RE.search(seq)
    if m_fr4 is None:
        return None
    # Last `[A-Z]C` strictly before the FR4 anchor is the FR3-end Cys-104.
    last_yc = None
    for m in _CDR3_FR3_END_RE.finditer(seq, 0, m_fr4.start()):
        last_yc = m
    if last_yc is None:
        return None
    cdr3_start_0b = last_yc.end()       # first residue AFTER the conserved Cys
    cdr3_end_0b = m_fr4.start() - 1     # last residue BEFORE the FR4 anchor
    # Sanity: real CDR3 is 3-30 aa. Outside this we caught a spurious motif
    # (e.g. FR1 Cys-23 + a false-positive FR4 anchor in CDR2/3 noise).
    cdr3_len = cdr3_end_0b - cdr3_start_0b + 1
    if cdr3_len < 3 or cdr3_len > 30:
        return None
    return cdr3_start_0b + 1, cdr3_end_0b + 1   # convert to 1-based inclusive


def generate_plm_training_data():
    const.require_data_path("pepLM training data")
    os.makedirs(const.RUN_PATH, exist_ok=True)
    # PEPPR_PRIOR_PROTEASES supports only tryptic digestion or non-specific
    # sliding-window generation.
    digestion_mode = os.environ.get("PEPPR_PRIOR_PROTEASES", "trypsin").strip().lower()
    if digestion_mode not in {"trypsin", "sliding"}:
        raise ValueError("PEPPR_PRIOR_PROTEASES must be 'trypsin' or 'sliding'")
    use_sliding = digestion_mode == "sliding"
    if use_sliding:
        print(f"prior training corpus: sliding-window non-specific peptides "
              f"(lmin={os.environ.get('PEPPR_PRIOR_MIN_PEP_LEN', 6)}, "
              f"lmax={os.environ.get('PEPPR_PRIOR_MAX_PEP_LEN', 25)})")
    else:
        print("prior training corpus: tryptic peptides")

    PRIOR_BUFFER_SIZE = int(os.environ.get("PEPPR_PRIOR_BUFFER_SIZE", 4_000_000))
    cdr3_ood = os.environ.get("PEPPR_PRIOR_CDR3_OOD", "0").lower() in ("1", "true", "yes")
    if cdr3_ood:
        print("prior CDR3-OOD training: target positions inside CDR3 will be "
              "written as -1 (triggers OOD satisficing loss; encourages flat "
              "logits at CDR3).")
    trie = pygtrie.Trie()
    all_encoded = []
    all_cdr3_masks = []     # parallel to all_encoded; None where no mask
    if use_sliding:
        # Two-pass ordering: unique protein sequences first, naug variants second.
        # A random early-stop over a flat-shuffled corpus risks missing rare V-gene
        # alleles when common alleles fill the budget first.  Processing all unique
        # sequences (one entry per distinct protein string) before any naug variant
        # guarantees full allele coverage regardless of buffer size.
        print("Loading FASTA into memory (sliding mode)...")
        records = list(SeqIO.parse(const.FASTA_PATH, "fasta"))
        rng = np.random.default_rng(0)
        seen_seqs: set = set()
        unique_records, naug_records = [], []
        for rec in records:
            s = str(rec.seq)
            if s not in seen_seqs:
                seen_seqs.add(s)
                unique_records.append(rec)
            else:
                naug_records.append(rec)
        rng.shuffle(unique_records)
        rng.shuffle(naug_records)
        ordered_records = unique_records + naug_records
        print(f"FASTA: {len(records):,} total records "
              f"({len(unique_records):,} unique, {len(naug_records):,} naug); "
              f"unique first, then naug, until {PRIOR_BUFFER_SIZE:,} uniques reached...")
        record_iter = tqdm(ordered_records, total=len(ordered_records),
                           desc="Generating prior training data")
    else:
        total_records = len(SeqIO.index(const.FASTA_PATH, "fasta"))
        record_iter = tqdm(SeqIO.parse(const.FASTA_PATH, "fasta"), total=total_records, desc="Generating prior training data")
    full_break = False
    per_protein_cap = int(os.environ.get("PEPPR_PRIOR_PER_PROTEIN_CAP", "0"))
    sampling_rng = np.random.default_rng(0)
    for record in record_iter:
        if full_break:
            break
        protein = '.' + str(record.seq)  # Add N-terminus
        # CDR3 location in `protein` (1-based, accounting for the leading '.').
        cdr3_in_protein = None
        if use_sliding and cdr3_ood:
            cdr3_in_rec = _find_cdr3_range(str(record.seq))
            if cdr3_in_rec is not None:
                cdr3_in_protein = (cdr3_in_rec[0] + 1, cdr3_in_rec[1] + 1)
        if use_sliding:
            res = sliding_window_peptides(protein, lmin=6, lmax=25, reverse=True,
                                          cdr3_range=cdr3_in_protein)
            if cdr3_in_protein is not None:
                peptides, peptide_masks = res
            else:
                peptides = res
                peptide_masks = [None] * len(peptides)
            # Optional per-protein cap: when set, every protein contributes at
            # most `cap` randomly-sampled L-mers. This guarantees proteome-wide
            # coverage when the corpus is large enough that the global buffer
            # would otherwise fill from a small fraction of proteins (e.g.,
            # HLA-length sliding window over 20k human proteins → 6.8% of
            # proteome reached without a cap).
            if per_protein_cap and len(peptides) > per_protein_cap:
                idx = sampling_rng.choice(len(peptides), per_protein_cap,
                                            replace=False)
                idx.sort()
                peptides = [peptides[i] for i in idx]
                peptide_masks = [peptide_masks[i] for i in idx]
        else:
            peptides = digest_protein(protein, 1, False)
            peptide_masks = [None] * len(peptides)
        for peptide, pep_mask in zip(peptides, peptide_masks):
            try:
                encoded = encode(peptide)
            except KeyError:
                # Some peptides contain "X" which is not in the vocabulary
                continue
            if encoded in trie:
                continue
            # Skip peptides that don't fit the pepLM block. The row layout is
            # [START] + peptide + terminator, and get_batch() needs a trailing
            # pad slot to detect peptide length, so len(encoded) (peptide + $)
            # must be < PRIOR_BLOCK_SIZE. Ultra-long tryptic peptides from
            # K/R-poor regions are not realistic MS/MS targets; most proteomes
            # contain none, but a few (e.g. S. cerevisiae S288c) do.
            if len(encoded) >= const.PRIOR_BLOCK_SIZE:
                continue
            # CDR3-OOD filter: drop peptides whose first emitted residue would
            # be OOD (corrupts the model's initial-context behavior) and
            # peptides where every residue is OOD (zero CE signal).
            if pep_mask is not None:
                # pep_mask is parallel to `encoded`; mask[0] = True means the
                # first predicted token is CDR3-OOD.
                if pep_mask[0]:
                    continue
                # All-OOD: every non-terminator position is CDR3.
                n_residues = len(encoded) - 1   # exclude the terminator $
                if all(pep_mask[i] for i in range(n_residues)):
                    continue
            length = len(encoded)
            for i in range(1, length+1):
                prefix = encoded[:i]
                try:
                    trie[prefix] += 1
                except KeyError:
                    trie[prefix] = 1
            all_encoded.append(encoded)
            all_cdr3_masks.append(pep_mask)
            # Stop once the budget is full. Unique-first ordering above
            # ensures all V-gene alleles appear before naug variants fill
            # the remaining budget.
            if use_sliding and len(all_encoded) >= PRIOR_BUFFER_SIZE:
                full_break = True
                break

    total_unique = len(all_encoded)
    print(f"Total unique peptides after digestion: {total_unique}")
    if total_unique > PRIOR_BUFFER_SIZE:
        print(f"Downsampling from {total_unique} to {PRIOR_BUFFER_SIZE} sequences (seed=0)")
        rng = np.random.default_rng(0)
        chosen = rng.choice(total_unique, size=PRIOR_BUFFER_SIZE, replace=False)
        chosen.sort()
        all_encoded = [all_encoded[i] for i in chosen]
        all_cdr3_masks = [all_cdr3_masks[i] for i in chosen]

    index = len(all_encoded)
    X_prior_seq = torch.full(size=(index, const.PRIOR_BLOCK_SIZE), fill_value=const.VOCAB.index('-'), dtype=torch.int64)
    Y_prior_seq = torch.full(size=(index, const.PRIOR_BLOCK_SIZE), fill_value=const.VOCAB.index('-'), dtype=torch.int64)
    n_cdr3_pos_total = 0
    for i, (encoded, mask) in enumerate(zip(all_encoded, all_cdr3_masks)):
        length = len(encoded)
        #  ^PEPTIDE
        X_prior_seq[i, 0] = const.START_TOKEN
        X_prior_seq[i, 1:length] = torch.as_tensor(encoded[:length-1], dtype=torch.int64)
        #  PEPTIDE$
        Y_prior_seq[i, :length] = torch.as_tensor(encoded, dtype=torch.int64)
        if mask is not None:
            # CDR3 OOD: replace Y at CDR3 positions with -1 (model treats as OOD).
            # `mask` is parallel to `encoded` and includes the $ slot (always False).
            for j, m_flag in enumerate(mask):
                if m_flag and j < length:
                    Y_prior_seq[i, j] = -1
                    n_cdr3_pos_total += 1
    print("Generated X_plm_seq and Y_plm_seq with ", index, " sequences")
    if cdr3_ood:
        print(f"  CDR3-OOD positions marked: {n_cdr3_pos_total:,} "
              f"(~{100*n_cdr3_pos_total/((index)*const.PRIOR_BLOCK_SIZE):.2f}% of "
              f"all target positions)")

    torch.save(X_prior_seq, const.PRIOR_SEQ_X_PATH)
    torch.save(Y_prior_seq, const.PRIOR_SEQ_Y_PATH)
    with open(const.PRIOR_SEQ_COUNTS_PATH, 'wb') as f:
        pickle.dump(trie, f)
    print("Saved prior training data to", const.PRIOR_SEQ_X_PATH, const.PRIOR_SEQ_Y_PATH)
    
def split_mgf_file(
    input_file: str,
    train_file: Optional[TextIO] = None,
    test_file: Optional[TextIO] = None,
    assignments: Optional[dict] = None,
    train_ratio: float = 0.8,
    skip: int = 0,
    output_file: Optional[str] = None,
    validate_proforma: bool = False,
) -> Tuple[int, int]:
    """
    Process an MGF file: convert SEQ= to ProForma (MassIVE-KB -> depthcharge format).

    Two modes:
    - With output_file: write all spectra to a single file (no train/test split).
    - With train_file and test_file: split into train/test by train_ratio.

    When skip > 0, only every (skip+1)-th spectrum is written.
    When validate_proforma=True, SEQ= lines that fail to parse (e.g. "Missing Closing Tag")
    are fixed by stripping modification annotations to plain amino acids.
    Returns (total_spectra, written_spectra).
    """
    current_spectrum = []
    in_spectrum = False
    force_train = None
    spectrum_index = 0
    written_count = 0
    no_split = output_file is not None
    assignments = assignments or {}
    out_handle = None
    if no_split:
        out_handle = open(output_file, 'w')

    try:
        with open(input_file, 'r') as infile, tqdm() as pbar:
            for line in infile:
                line = line.rstrip('\n')
                if line == 'BEGIN IONS':
                    in_spectrum = True
                    current_spectrum = [line]
                    seq = None
                elif line == 'END IONS':
                    current_spectrum.append(line)
                    keep = skip == 0 or spectrum_index % (skip + 1) == 0
                    if keep:
                        lines = '\n'.join(current_spectrum) + '\n\n'
                        if no_split:
                            out_handle.write(lines)
                            written_count += 1
                        elif force_train is not None:
                            if force_train:
                                train_file.write(lines)
                            else:
                                test_file.write(lines)
                            if seq is not None:
                                assignments[seq] = force_train
                            written_count += 1
                        elif random.random() < train_ratio:
                            train_file.write(lines)
                            if seq is not None:
                                assignments[seq] = True
                            written_count += 1
                        else:
                            test_file.write(lines)
                            if seq is not None:
                                assignments[seq] = False
                            written_count += 1
                    spectrum_index += 1
                    in_spectrum = False
                    force_train = None
                    current_spectrum = None
                    pbar.update(1)
                elif in_spectrum:
                    if line.startswith('SEQ='):
                        seq_value = line.split('=', 1)[1]
                        line = 'SEQ=' + _convert_seq_to_proforma(seq_value, validate=validate_proforma)
                        seq = re.sub(r'\[.*?\]', '', line.split('=', 1)[1])
                        if not no_split and seq in assignments:
                            force_train = assignments[seq]
                    current_spectrum.append(line)
                else:
                    pass
    finally:
        if out_handle is not None:
            out_handle.close()

    return (spectrum_index, written_count)


def convert_mgf_to_proforma(
    input_glob: str,
    output_dir: str,
    skip: int = 0,
    validate_proforma: bool = True,
) -> None:
    """
    Convert MGF file(s) to ProForma (MassIVE-KB -> depthcharge format) without splitting.
    Writes each input file to output_dir with the same basename.

    When validate_proforma=True (default), SEQ= lines that fail to parse
    (e.g. ProFormaError "Missing Closing Tag") are fixed by stripping
    modification annotations to plain amino acids.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file in glob.glob(input_glob):
        output_path = os.path.join(output_dir, os.path.basename(file))
        print(f"Converting {file} -> {output_path}...")
        total, written = split_mgf_file(
            file,
            output_file=output_path,
            skip=skip,
            validate_proforma=validate_proforma,
        )
        print(f"  Wrote {written} / {total} spectra")


_BRACKET_TOKEN_RE = re.compile(r"\[([^\[\]]+)\]")
_BRACKET_SPAN_RE = re.compile(r"\[[^\[\]]+\]-?")
_STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _has_nonstandard_aa(proforma: str) -> bool:
    backbone = _BRACKET_SPAN_RE.sub("", proforma)
    return any(c not in _STANDARD_AAS for c in backbone)


def _prepare_casanovo_ft(
    mgf_in: str,
    out_dir: str,
    output_prefix: str,
    description: str,
    ratios: Tuple[float, float, float],
) -> dict:
    """Normalize an MGF to Casanovo ProForma and split by peptide backbone."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    train_ratio, val_ratio, _ = ratios
    train_cut = round(train_ratio * 1000)
    val_cut = round((train_ratio + val_ratio) * 1000)
    split_names = ("train", "val", "test")
    os.makedirs(out_dir, exist_ok=True)
    out_paths = {
        split: os.path.join(out_dir, f"{output_prefix}_{split}.mgf")
        for split in split_names
    }
    counts = {
        "total": 0, "train": 0, "val": 0, "test": 0,
        "dropped_vocab": 0, "dropped_no_seq": 0,
        "dropped_nonstandard_aa": 0,
    }
    backbones = {split: set() for split in split_names}
    dropped_tokens: dict[str, int] = defaultdict(int)

    def parse_sequence(line: str) -> tuple[str, str, Optional[str]]:
        proforma = _normalize_mass_shifts(line.rstrip("\n").split("=", 1)[1])
        bad_token = next(
            (token for token in _BRACKET_TOKEN_RE.findall(proforma)
             if token not in _CASANOVO_TMT_ALLOWED_MODS),
            None,
        )
        backbone = _BRACKET_SPAN_RE.sub("", proforma)
        if bad_token is None and _has_nonstandard_aa(proforma):
            bad_token = "<nonstandard_aa>"
        return proforma, backbone, bad_token

    def choose_split(backbone: str) -> str:
        bucket = int(hashlib.md5(backbone.encode("utf-8")).hexdigest()[:8], 16) % 1000
        if bucket < train_cut:
            return "train"
        return "val" if bucket < val_cut else "test"

    file_size = os.path.getsize(mgf_in)
    with open(mgf_in) as source, \
         open(out_paths["train"], "w") as train_out, \
         open(out_paths["val"], "w") as val_out, \
         open(out_paths["test"], "w") as test_out, \
         tqdm(total=file_size, unit="B", unit_scale=True, desc=description) as pbar:
        outputs = {"train": train_out, "val": val_out, "test": test_out}
        spectrum: list[str] = []
        backbone: Optional[str] = None
        rejection: Optional[str] = None

        for line in source:
            pbar.update(len(line))
            if line.startswith("BEGIN IONS"):
                spectrum = [line]
                backbone = rejection = None
            elif not spectrum:
                continue
            elif line.startswith("SEQ="):
                proforma, backbone, rejection = parse_sequence(line)
                if rejection is not None:
                    dropped_tokens[rejection] += 1
                spectrum.append(f"SEQ={proforma}\n")
            elif line.startswith("END IONS"):
                spectrum.append(line)
                counts["total"] += 1
                if backbone is None:
                    counts["dropped_no_seq"] += 1
                elif rejection == "<nonstandard_aa>":
                    counts["dropped_nonstandard_aa"] += 1
                elif rejection is not None:
                    counts["dropped_vocab"] += 1
                else:
                    split = choose_split(backbone)
                    outputs[split].writelines(spectrum)
                    outputs[split].write("\n")
                    counts[split] += 1
                    backbones[split].add(backbone)
                spectrum = []
            else:
                spectrum.append(line)

    stats = {
        "input_mgf": mgf_in,
        "ratios": list(ratios),
        "spectra": counts,
        "unique_backbones": {split: len(values) for split, values in backbones.items()},
        "overlap_check": {
            "train_val": len(backbones["train"] & backbones["val"]),
            "train_test": len(backbones["train"] & backbones["test"]),
            "val_test": len(backbones["val"] & backbones["test"]),
        },
        "dropped_tokens": dict(sorted(dropped_tokens.items(), key=lambda item: -item[1])),
    }
    with open(os.path.join(out_dir, "split_stats.json"), "w") as stats_file:
        json.dump(stats, stats_file, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


def prepare_luad_casanovo_ft(
    mgf_in: str,
    out_dir: str,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    return _prepare_casanovo_ft(
        mgf_in, out_dir, "luad", "LUAD MGF -> ProForma + split", ratios
    )


def prepare_pxd033643_casanovo_ft(
    mgf_in: str,
    out_dir: str,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    return _prepare_casanovo_ft(
        mgf_in, out_dir, "pxd033643", "PXD033643 MGF -> ProForma + split", ratios
    )


def translate_for_prior(input, output_path, translation_table=None):
    if translation_table is None:
        translation_table = const.CASANOVO_TRANSLATION
    translated_train = translation_table[input]
    translated_train[translated_train == const.VOCAB.index('$')] = const.VOCAB.index('-')
    output = torch.zeros_like(input)
    output[:, 1:] = translated_train[:, :-1]
    output[:, 0] = const.START_TOKEN
    torch.save(output, output_path)

def generate_fusion_files(translation_table=None):
    """Build the fusion-training tensors from `casanovo_teacher_{train,test}_torch_data.pt`.

    Parameters
    ----------
    translation_table : Optional[torch.Tensor]
        Per-token mapping from the teacher's vocab index to the pepLM ``VOCAB``
        index, used by ``translate_for_prior`` to materialise ``PRIOR_PSM_X_*``.
        When ``None`` (default) the built-in ``const.CASANOVO_TRANSLATION``
        (Casanovo v5 default residues) is used. Pass a custom tensor when the
        teacher checkpoint was trained with a non-default residue dict
        (e.g. Casanovo v5 + TMT6plex for the LUAD fine-tune), in which case the
        bundled translation table doesn't cover the trailing TMT token ids.
    """
    torch_file_train = torch.load(const.CASANOVO_TEACHER_TRAIN_MZTAB_PATH.replace('.mztab', '_torch_data.pt'))

    scores_casanovo_train = torch_file_train['emb']
    scores_padded_train = torch.zeros(scores_casanovo_train.size(0), const.PRIOR_BLOCK_SIZE, scores_casanovo_train.size(2))
    scores_padded_train[:, :scores_casanovo_train.size(1), :] = scores_casanovo_train[:, :scores_casanovo_train.size(1), :]
    torch.save(scores_padded_train, const.CASANOVO_TEACHER_SCORES_TRAIN_PATH)
    print("Saved CASANOVO scores for train")

    true_tokenized_train = torch_file_train['true_tokenized']
    true_tokenized_padded_train = torch.zeros(true_tokenized_train.size(0), const.PRIOR_BLOCK_SIZE, dtype=torch.long)
    true_tokenized_padded_train[:, :true_tokenized_train.size(1)] = true_tokenized_train[:, :true_tokenized_train.size(1)]
    torch.save(true_tokenized_padded_train, const.FUSION_Y_TRAIN_PATH)
    translate_for_prior(true_tokenized_padded_train, const.PRIOR_PSM_X_TRAIN_PATH, translation_table=translation_table)
    print("Fusion Y train saved to", const.FUSION_Y_TRAIN_PATH)

    torch_file_test = torch.load(const.CASANOVO_TEACHER_TEST_MZTAB_PATH.replace('.mztab', '_torch_data.pt'))

    scores_casanovo_test = torch_file_test['emb']
    scores_padded_test = torch.zeros(scores_casanovo_test.size(0), const.PRIOR_BLOCK_SIZE, scores_casanovo_test.size(2))
    scores_padded_test[:, :scores_casanovo_test.size(1), :] = scores_casanovo_test[:, :scores_casanovo_test.size(1), :]
    torch.save(scores_padded_test, const.CASANOVO_TEACHER_SCORES_TEST_PATH)
    print("Saved CASANOVO scores for test")

    true_tokenized_test = torch_file_test['true_tokenized']
    true_tokenized_padded_test = torch.zeros(true_tokenized_test.size(0), const.PRIOR_BLOCK_SIZE, dtype=torch.long)
    true_tokenized_padded_test[:, :true_tokenized_test.size(1)] = true_tokenized_test[:, :true_tokenized_test.size(1)]
    torch.save(true_tokenized_padded_test, const.FUSION_Y_TEST_PATH)
    translate_for_prior(true_tokenized_padded_test, const.PRIOR_PSM_X_TEST_PATH, translation_table=translation_table)
    print("Fusion Y test saved to", const.FUSION_Y_TEST_PATH)


def run_casanovo_teacher_for_mgfs(
    mgf_paths: list[str],
    out_dir: str,
    mztab_basename: str,
    *,
    force: bool = False,
) -> None:
    """Teacher-forcing Casanovo on the given MGFs; writes ``{basename}.mztab`` and
    ``{basename}_torch_data.pt`` under ``out_dir``."""
    import subprocess

    if not mgf_paths:
        raise ValueError("run_casanovo_teacher_for_mgfs: empty mgf_paths")
    os.makedirs(out_dir, exist_ok=True)
    torch_path = os.path.join(out_dir, f"{mztab_basename}_torch_data.pt")
    if force and os.path.exists(torch_path):
        os.remove(torch_path)
    if os.path.exists(torch_path) and os.path.getsize(torch_path) > 0:
        print(f"[skip] teacher tensor exists: {torch_path}")
        return
    if not os.path.isfile(const.CASANOVO_CONFIG_YAML):
        raise FileNotFoundError(f"Casanovo config not found: {const.CASANOVO_CONFIG_YAML}")
    ckpt = os.environ.get(
        "PEPPR_CASANOVO_CKPT",
        const.CASANOVO_DEFAULT_CHECKPOINT,
    )
    cmd = [
        "casanovo",
        "sequence",
        "-m",
        ckpt,
        "-c",
        const.CASANOVO_CONFIG_YAML,
        "-d",
        out_dir,
        "-o",
        mztab_basename,
        "--teacher_forcing",
        "true",
        "--use_peppr",
        "false",
        "-e",
        *mgf_paths,
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def generate_contranovo_fusion_files():
    """Build fusion training inputs from ContraNovo teacher-forced output.

    Reads the {emb, true_tokenized} dicts written by ContraNovo's run_teacher.py
    (matches casanovo's _torch_data.pt schema), pads to PRIOR_BLOCK_SIZE, and
    derives the per-position pepLM input by translating ContraNovo's vocab to
    pepLM's via build_contranovo_translation. ContraNovo's residues come from
    its config.yaml; we read them here so the translation matches what's in
    the checkpoint at inference time.
    """
    import yaml

    with open(const.CONTRANOVO_CONFIG_YAML) as f:
        residues = yaml.safe_load(f)["residues"]
    residues = {str(aa): float(m) for aa, m in residues.items()}
    translation = const.build_contranovo_translation(residues).cpu()

    for tag, in_pt, out_scores_path, out_y_path, out_x_path in (
        (
            "train",
            const.CONTRANOVO_TEACHER_TRAIN_PT_PATH,
            const.CONTRANOVO_TEACHER_SCORES_TRAIN_PATH,
            const.CONTRANOVO_FUSION_Y_TRAIN_PATH,
            const.CONTRANOVO_PRIOR_PSM_X_TRAIN_PATH,
        ),
        (
            "test",
            const.CONTRANOVO_TEACHER_TEST_PT_PATH,
            const.CONTRANOVO_TEACHER_SCORES_TEST_PATH,
            const.CONTRANOVO_FUSION_Y_TEST_PATH,
            const.CONTRANOVO_PRIOR_PSM_X_TEST_PATH,
        ),
    ):
        if not os.path.exists(in_pt):
            raise FileNotFoundError(
                f"ContraNovo teacher pt file not found: {in_pt}. "
                "Run ContraNovo/run_teacher.py first."
            )
        torch_file = torch.load(in_pt)
        scores = torch_file["emb"]
        true_tokenized = torch_file["true_tokenized"].long()

        scores_padded = torch.zeros(scores.size(0), const.PRIOR_BLOCK_SIZE, scores.size(2))
        L = min(scores.size(1), const.PRIOR_BLOCK_SIZE)
        scores_padded[:, :L, :] = scores[:, :L, :]
        torch.save(scores_padded, out_scores_path)
        print(f"Saved ContraNovo scores ({tag}) -> {out_scores_path}  shape={tuple(scores_padded.shape)}")

        y_padded = torch.zeros(true_tokenized.size(0), const.PRIOR_BLOCK_SIZE, dtype=torch.long)
        Lt = min(true_tokenized.size(1), const.PRIOR_BLOCK_SIZE)
        y_padded[:, :Lt] = true_tokenized[:, :Lt]
        torch.save(y_padded, out_y_path)
        print(f"Fusion Y ({tag}) saved to {out_y_path}")

        translate_for_prior(y_padded, out_x_path, translation_table=translation)
        print(f"prior X ({tag}) saved to {out_x_path}")




def prepare_training_data() -> None:
    """Create prior and fusion training tensors when they do not already exist."""
    print(
        f"=== Training data: species={const.ACTIVE_SPECIES}, "
        f"prior species={const.PRIOR_SPECIES}, RUN_PATH={const.RUN_PATH} ==="
    )
    prior_data_exists = os.path.exists(const.PRIOR_SEQ_X_PATH) and os.path.exists(const.PRIOR_SEQ_Y_PATH)
    if const.PRIOR_SPECIES != const.ACTIVE_SPECIES:
        if not prior_data_exists:
            raise RuntimeError(
                f"Cross-species prior data not found at {const.PRIOR_RUN_PATH}; "
                f"train {const.PRIOR_SPECIES!r} first."
            )
        print(f"Using cross-species prior data from {const.PRIOR_RUN_PATH}.")
    elif prior_data_exists:
        print("prior training data already exists, skipping.")
    else:
        print(f"Generating prior training data from {const.FASTA_PATH}...")
        generate_plm_training_data()

    if os.path.exists(const.PRIOR_PSM_X_TRAIN_PATH):
        print("Fusion files already exist, skipping.")
    else:
        print("Generating fusion files...")
        generate_fusion_files()


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prepare_data.py")
    subparsers = parser.add_subparsers(dest="command")
    convert = subparsers.add_parser(
        "convert_proforma",
        help="convert MGF SEQ mass-shift annotations to valid ProForma",
    )
    convert.add_argument("input_glob")
    convert.add_argument("output_dir")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_cli().parse_args(argv)
    if args.command == "convert_proforma":
        convert_mgf_to_proforma(
            args.input_glob,
            args.output_dir,
            validate_proforma=True,
        )
    else:
        prepare_training_data()


if __name__ == "__main__":
    main()
