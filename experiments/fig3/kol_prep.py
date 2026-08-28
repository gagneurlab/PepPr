"""Kingdoms-of-Life benchmark preparation.

Turns the raw PXD014877 / PXD019483 acquisitions and their MaxQuant msms.txt
search results into the per-species ProForma MGF subsets the KoL panel of
figure 3 is scored on.

This is paper-reproduction tooling and is only needed to regenerate the KoL
inputs from raw: the portable data archive already ships the final subset MGFs
under benchmarks/kingdoms_of_life/subsets/. It lives here rather than in peppr
because nothing about it generalises beyond this benchmark.
"""

import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments import paths
from peppr.prepare_data import convert_mgf_to_proforma


def _normalize_kol_raw_name(raw_field: str) -> str:
    """Map a 'Raw file' value from msms.txt to the on-disk basename (no .raw).

    PRIDE strips '+' and replaces 'ü' with 'u' when storing files, and
    additionally renames a handful of species at upload (e.g. 'Maus' ->
    'Musmusculus'). Per-species renames live in paths.KOL_NAME_REPLACEMENTS.
    """
    s = raw_field.replace('+', '').replace('ü', 'u')
    for old, new in paths.KOL_NAME_REPLACEMENTS.items():
        s = s.replace(old, new)
    return s


def extract_psms_from_msms(
    msms_path: str,
    drop_multi_secpep: bool = True,
    drop_reverse: bool = True,
) -> dict[str, dict[str, str]]:
    """Read a MaxQuant msms.txt and return {raw_basename: {scan_str: modified_seq}}.

    The Modified sequence is returned in MaxQuant form (e.g. _C(ox)PEPTIDE_).
    Raw-file keys are normalized via _normalize_kol_raw_name so they match the
    on-disk .raw basenames.
    """
    # MaxQuant >=2.x renamed the 'Reverse' column to 'Decoy'; older MQ versions
    # (the Muller KoL search used MQ 1.5.3) still ship 'Reverse'. Detect which
    # name the file uses so this works across both vintages.
    msms_cols = set(pd.read_csv(msms_path, sep='\t', nrows=0).columns)
    decoy_col = 'Reverse' if 'Reverse' in msms_cols else 'Decoy'
    df = pd.read_csv(
        msms_path,
        sep='\t',
        dtype=str,
        usecols=['Raw file', 'Scan number', 'Modified sequence', 'Type', decoy_col],
    )
    if drop_multi_secpep:
        df = df[df['Type'].ne('MULTI-SECPEP')]
    if drop_reverse:
        df = df[df[decoy_col].isna()]
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for raw, scan, seq in zip(df['Raw file'], df['Scan number'], df['Modified sequence']):
        result[_normalize_kol_raw_name(raw)][str(scan)] = seq
    return result


def _maxquant_modseq_to_massshift(modseq: str) -> str:
    """Convert MaxQuant _(ac)M(ox)PEPTIDE_ -> [+42.011]-M[+15.995]PEPTIDE.

    Strips the underscore wrappers, expands (ac)/(ox), and adds Carbamidomethyl
    on every unmodified C (it's the fixed mod in this study).
    """
    s = modseq.strip('_')
    s = re.sub(r'\(ac\)', '[+42.011]-', s)
    s = re.sub(r'M\(ox\)', 'M[+15.995]', s)
    s = re.sub(r'C(?!\[)', 'C[+57.021]', s)
    return s


def _raw_to_mzml(raw_path: str, output_dir: str, log_path: str | None = None) -> str:
    """Convert .raw to centroided MS2-only mzML via ThermoRawFileParser.

    Returns the output mzML path. If ``log_path`` is given, TRFP's stdout/stderr
    are redirected there (avoids interleaving when multiple TRFPs run in
    parallel). The MGF output path in TRFP 2.0 has a serious perf bug on these
    large QExactive .raws (~12 MB/min); the mzML path is ~80x faster and still
    gives us peak-picked spectra.
    """
    if not paths.THERMO_RAW_FILE_PARSER:
        raise RuntimeError(
            "RAW conversion requires DNPS_THERMO_RAW_FILE_PARSER to point "
            "to the ThermoRawFileParser executable."
        )
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        f"{paths.THERMO_RAW_FILE_PARSER} -i={raw_path} -o={output_dir} "
        f"-f=1 -L=2 -l=2"
    )
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cmd += f" >>{log_path} 2>&1"
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"ThermoRawFileParser failed (rc={rc}) on {raw_path}")
    return os.path.join(output_dir, os.path.basename(raw_path).replace('.raw', '.mzML'))


def annotate_mzml_to_mgf(
    mzml_path: str,
    scan_to_modseq: dict,
    mgf_out_path: str,
    only_with_psm: bool = True,
) -> Tuple[int, int]:
    """Read MS2 spectra from mzML and write a SEQ-annotated MGF.

    SEQ values are written in mass-shift form (MassIVE-KB style) so the existing
    convert_mgf_to_proforma helper can finalize them. scan_to_modseq is the
    MaxQuant 'Modified sequence' for each scan number (e.g. '_(ac)PEP_').

    Returns (ms2_total, written) counts.
    """
    from pyteomics import mzml as _mzml
    total = written = 0
    with open(mgf_out_path, 'w') as out, _mzml.read(mzml_path) as reader:
        for spec in reader:
            if spec.get('ms level') != 2:
                continue
            total += 1
            scan_id = spec.get('id', '')
            m = re.search(r'\bscan=(\d+)\b', scan_id)
            if not m:
                continue
            scan = m.group(1)
            modseq = scan_to_modseq.get(scan)
            if only_with_psm and modseq is None:
                continue
            try:
                pl = spec['precursorList']['precursor'][0]
                sil = pl['selectedIonList']['selectedIon'][0]
                pepmass = sil.get('selected ion m/z')
                charge = sil.get('charge state')
            except (KeyError, IndexError):
                continue
            try:
                rt_min = spec['scanList']['scan'][0].get('scan start time')
                rt_sec = float(rt_min) * 60.0 if rt_min is not None else None
            except (KeyError, IndexError):
                rt_sec = None
            mz_arr = spec['m/z array']
            intensity_arr = spec['intensity array']

            out.write("BEGIN IONS\n")
            out.write(f"TITLE={scan_id}\n")
            if pepmass is not None:
                out.write(f"PEPMASS={pepmass}\n")
            if rt_sec is not None:
                out.write(f"RTINSECONDS={rt_sec}\n")
            if charge is not None:
                out.write(f"CHARGE={int(charge)}+\n")
            out.write(f"SCANS={scan}\n")
            if modseq is not None:
                out.write(f"SEQ={_maxquant_modseq_to_massshift(modseq)}\n")
            for mz_val, i_val in zip(mz_arr, intensity_arr):
                out.write(f"{mz_val} {i_val}\n")
            out.write("END IONS\n\n")
            written += 1
    return total, written


def _sample_psm_map(
    psm_map: dict,
    target_n: int,
    seed: int,
) -> dict:
    """Uniformly sample ``target_n`` PSMs across all raws in ``psm_map``.

    Input/output schema is ``{raw_basename: {scan_str: modified_seq}}``.
    Returns a new dict with the same shape; raws that end up with zero sampled
    PSMs are dropped from the result so downstream code can skip their raws
    entirely (the main win of sampling-before-TRFP).
    """
    flat: list[tuple[str, str, str]] = [
        (raw, scan, seq)
        for raw, scans in psm_map.items()
        for scan, seq in scans.items()
    ]
    if target_n >= len(flat):
        return psm_map
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(flat), size=target_n, replace=False)
    sampled: dict = defaultdict(dict)
    for i in chosen_idx:
        raw, scan, seq = flat[i]
        sampled[raw][scan] = seq
    return dict(sampled)


def _kol_process_one_raw(
    basename: str,
    raw_path: str,
    scan_to_seq: dict,
    annotated_path: str,
    staging_dir: str,
    keep_mzml: bool,
) -> tuple[str, int, int, int, str | None]:
    """Worker: raw -> mzML -> annotated MGF for a single .raw, with the TRFP
    output redirected to a per-raw log file (parallel-friendly).

    Failures are caught and returned via the last tuple element so the parent
    can log the bad raw and keep processing the rest (some PRIDE-uploaded
    .raws are missing scan indices and TRFP rejects them as "Empty RAW file").

    Returns (basename, ms2_total, written, n_psms, error_or_None).
    """
    mzml_path = os.path.join(staging_dir, basename + ".mzML")
    log_path = os.path.join(staging_dir, "logs", basename + ".trfp.log")
    try:
        if not os.path.exists(mzml_path):
            _raw_to_mzml(raw_path, staging_dir, log_path=log_path)
        total, written = annotate_mzml_to_mgf(mzml_path, scan_to_seq, annotated_path)
        if not keep_mzml:
            try:
                os.remove(mzml_path)
            except FileNotFoundError:
                pass
        return basename, total, written, len(scan_to_seq), None
    except Exception as e:
        return basename, 0, 0, len(scan_to_seq), f"{type(e).__name__}: {e}"


def prepare_kol_species_dataset(
    species: str,
    target_n: int | None = None,
    seed: int = 0,
    n_workers: int = 4,
    staging_dir: str | None = None,
    keep_mzml: bool = False,
):
    """End-to-end Kingdoms of Life prep for one species.

    Pipeline:
      1. Read per-spectrum PSMs from
         ``PXD014877/search_results/<SearchDir>/msms.txt``.
      2. If ``target_n`` is set, uniformly sample that many PSMs across raws
         BEFORE any expensive work — raws with 0 sampled PSMs are skipped
         entirely (no TRFP conversion).
      3. For each surviving raw, in a process pool of ``n_workers``:
           a. raw -> mzML (centroided, MS2-only) via TRFP, log redirected.
           b. mzML -> SEQ-annotated MGF (mass-shift form).
           c. delete staging mzML (unless ``keep_mzml=True``).
      4. Convert mass-shift SEQs to ProForma into the final mgf/ dir.

    Parallelism note: TRFP is single-threaded per process, so the dominant
    win for multi-species runs is parallel TRFP across raws. ProcessPool
    avoids GIL contention with pyteomics.
    """
    if staging_dir is None:
        staging_dir = paths.KOL_STAGING_DIR
    if species not in paths.KOL_SPECIES_DIRS:
        raise ValueError(
            f"unknown KoL species {species!r}; known: {sorted(paths.KOL_SPECIES_DIRS)}"
        )
    msms_path = paths.kol_species_msms_path(species)
    out_root = paths.kol_species_output_root(species)
    annotated_dir = os.path.join(out_root, "mgf_annotated")
    final_dir = os.path.join(out_root, "mgf")

    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    print(f"=== KoL species prep: {species} ({paths.KOL_SPECIES_DIRS[species]}) ===")
    print(f"Reading PSMs from {msms_path}")
    psm_map = extract_psms_from_msms(msms_path)
    total_psms = sum(len(v) for v in psm_map.values())
    print(f"  Loaded {total_psms:,} PSMs across {len(psm_map)} raws")

    if target_n is not None and target_n > 0:
        psm_map = _sample_psm_map(psm_map, target_n, seed)
        sampled_psms = sum(len(v) for v in psm_map.values())
        print(f"  Sampled to {sampled_psms:,} PSMs across {len(psm_map)} raws "
              f"(seed={seed}, target_n={target_n})")

    raw_index: dict[str, str] = {}
    for raw_dir in paths.KOL_RAW_DIRS:
        if not os.path.isdir(raw_dir):
            continue
        for p in sorted(glob.glob(os.path.join(raw_dir, "*.raw"))):
            raw_index.setdefault(os.path.basename(p).replace(".raw", ""), p)

    tasks: list[tuple] = []
    for basename, scan_to_seq in psm_map.items():
        if not scan_to_seq:
            continue
        annotated_path = os.path.join(annotated_dir, basename + ".mgf")
        if os.path.exists(annotated_path):
            print(f"  Skip (annotated exists): {basename}")
            continue
        raw_path = raw_index.get(basename)
        if raw_path is None:
            print(f"  WARN: missing .raw for {basename} (searched {paths.KOL_RAW_DIRS})")
            continue
        tasks.append((basename, raw_path, scan_to_seq, annotated_path,
                      staging_dir, keep_mzml))

    failures: list[tuple[str, str]] = []
    if not tasks:
        print("  Nothing to do (all annotated MGFs already present).")
    else:
        print(f"Processing {len(tasks)} raws with {n_workers} parallel worker(s)")

        def _report(result):
            basename, total, written, n_psms, err = result
            if err:
                failures.append((basename, err))
                print(f"  FAILED {basename}: {err}")
            else:
                print(f"  {basename}: {written}/{total} MS2 annotated (PSMs={n_psms})")

        if n_workers <= 1:
            for t in tqdm(tasks, desc="Raws"):
                _report(_kol_process_one_raw(*t))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_kol_process_one_raw, *t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Raws"):
                    _report(fut.result())

    print(f"Converting mass-shift SEQs to ProForma -> {final_dir}")
    convert_mgf_to_proforma(
        input_glob=os.path.join(annotated_dir, "*.mgf"),
        output_dir=final_dir,
        skip=0,
        validate_proforma=True,
    )

    if failures:
        print(f"\n!!! {len(failures)} raw(s) failed and were skipped:")
        for basename, err in failures:
            print(f"  - {basename}: {err}")
