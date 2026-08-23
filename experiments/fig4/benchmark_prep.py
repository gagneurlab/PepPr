"""mAb-benchmark input prep and output annotation (one CLI for the whole
antibody data pipeline).

Subcommands:
  stage-beslic              Stage the beslic tryptic chains: per-mAb dirs, TD
                            FASTA, spectra hardlinks, FragPipe workflow+manifest.
  prepare-nontryp           Build per-(mAb, protease) FragPipe workdirs from the
                            non-tryptic registry (MGF fixup + workflow + manifest).
  annotate {beslic,nontryp} Insert SEQ=<ProForma> into spectra from a FragPipe
                            1%-FDR psm.tsv.

Run from the repo root, e.g.:
  python experiments/fig4/benchmark_prep.py stage-beslic
  python experiments/fig4/benchmark_prep.py prepare-nontryp --idx 3 5
  python experiments/fig4/benchmark_prep.py annotate nontryp
  python experiments/fig4/benchmark_prep.py annotate beslic IgG1_Human_H
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from peppr.const import DATA_PATH
from experiments.fig4.benchmark_registry import RUNS, NonTrypRun

BESLIC_DIR = os.path.join(DATA_PATH, "beslic_mab")


# ══════════════════════════════════════════════════════════════════════════
# Shared: FragPipe psm.tsv → ProForma, and SEQ= MGF annotation
# ══════════════════════════════════════════════════════════════════════════
def parse_psm_tsv(path):
    """Yield (raw_basename, scan, charge, peptide, modifications) from a psm.tsv."""
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_spec = header.index("Spectrum")
        i_pep = header.index("Peptide")
        i_z = header.index("Charge")
        i_mods = header.index("Assigned Modifications")
        for line in f:
            row = line.rstrip("\n").split("\t")
            parts = row[i_spec].rsplit(".", 3)
            if len(parts) < 4:
                continue
            raw_base, scan_str, _scan_hi, _z = parts
            try:
                scan = int(scan_str)
                z = int(row[i_z])
            except ValueError:
                continue
            yield raw_base, scan, z, row[i_pep], row[i_mods]


def mods_to_proforma(plain_peptide: str, mods_str: str) -> str:
    out = list(plain_peptide)
    if not mods_str:
        return "".join(out)
    pat = re.compile(r"(\d+)([A-Za-z])\(([+-]?[\d.]+)\)")
    for m in pat.finditer(mods_str):
        pos = int(m.group(1))
        mass = float(m.group(3))
        if not 1 <= pos <= len(out):
            continue
        sign = "+" if mass >= 0 else "-"
        out[pos - 1] = f"{out[pos - 1]}[{sign}{abs(mass):.3f}]"
    return "".join(out)


def find_psm_tsv(workdir, required=True):
    """psm.tsv lives under fragpipe_workdir/<experiment>_<bioreplicate>/psm.tsv."""
    for entry in sorted(os.listdir(workdir)):
        full = os.path.join(workdir, entry)
        if os.path.isdir(full):
            p = os.path.join(full, "psm.tsv")
            if os.path.exists(p):
                return p
    if required:
        raise FileNotFoundError(f"no psm.tsv under {workdir}")
    return None


def scan_to_psm_map(psm_tsv):
    """Collapse a psm.tsv to {scan: (ProForma, charge)}, first PSM per scan."""
    scan_to_psm, n_rows = {}, 0
    for _raw_base, scan, z, pep, mods in parse_psm_tsv(psm_tsv):
        n_rows += 1
        if scan in scan_to_psm:
            continue
        scan_to_psm[scan] = (mods_to_proforma(pep, mods), z)
    return scan_to_psm, n_rows


def _insert_seq(block, seq):
    """Insert a `SEQ=<seq>` line after SCANS= (or CHARGE= as fallback) in a block."""
    out_lines, inserted = [], False
    for bl in block:
        out_lines.append(bl)
        if not inserted and bl.startswith("SCANS="):
            out_lines.append(f"SEQ={seq}\n")
            inserted = True
    if not inserted:
        out_lines, inserted = [], False
        for bl in block:
            out_lines.append(bl)
            if not inserted and bl.startswith("CHARGE="):
                out_lines.append(f"SEQ={seq}\n")
                inserted = True
    return out_lines


def annotate_mgf_from_mgf(mgf_in, scan_to_psm, out_mgf):
    """Source MGF already carries SCANS=<scan>; emit a copy with SEQ= for matches."""
    n_in = n_written = 0
    with open(mgf_in) as fi, open(out_mgf, "w") as fo:
        block, scan = [], None
        for line in fi:
            if line.startswith("BEGIN IONS"):
                block, scan = [line], None
                continue
            if line.startswith("END IONS"):
                n_in += 1
                if scan is not None and scan in scan_to_psm:
                    seq, _charge = scan_to_psm[scan]
                    fo.writelines(_insert_seq(block, seq))
                    fo.write("END IONS\n\n")
                    n_written += 1
                block = []
                continue
            block.append(line)
            if line.startswith("SCANS="):
                try:
                    scan = int(line.strip().split("=", 1)[1])
                except (ValueError, IndexError):
                    scan = None
    return n_in, n_written


def annotate_mzml(mzml_path, scan_to_psm, out_mgf):
    from pyteomics import mzml

    n_ms2 = n_written = 0
    with open(out_mgf, "w") as out, mzml.read(mzml_path) as reader:
        for spec in reader:
            if spec.get("ms level") != 2:
                continue
            n_ms2 += 1
            scan_id = spec.get("id", "")
            m = re.search(r"\bscan=(\d+)\b", scan_id)
            if not m:
                continue
            psm = scan_to_psm.get(int(m.group(1)))
            if psm is None:
                continue
            seq, charge = psm
            try:
                pl = spec["precursorList"]["precursor"][0]
                pepmass = pl["selectedIonList"]["selectedIon"][0].get("selected ion m/z")
            except (KeyError, IndexError):
                continue
            try:
                rt_min = spec["scanList"]["scan"][0].get("scan start time")
                rt_sec = float(rt_min) * 60.0 if rt_min is not None else None
            except (KeyError, IndexError):
                rt_sec = None
            out.write("BEGIN IONS\n")
            out.write(f"TITLE={scan_id}\n")
            out.write(f"PEPMASS={pepmass}\n")
            if rt_sec is not None:
                out.write(f"RTINSECONDS={rt_sec}\n")
            out.write(f"CHARGE={charge}+\n")
            out.write(f"SCANS={int(m.group(1))}\n")
            out.write(f"SEQ={seq}\n")
            for mz, it in zip(spec["m/z array"], spec["intensity array"]):
                out.write(f"{mz} {it}\n")
            out.write("END IONS\n\n")
            n_written += 1
    return n_ms2, n_written


def annotate_mzxml(mzxml_path, scan_to_psm, out_mgf):
    from pyteomics import mzxml

    n_ms2 = n_written = 0
    with open(out_mgf, "w") as out, mzxml.read(mzxml_path) as reader:
        for spec in reader:
            if spec.get("msLevel") != 2:
                continue
            n_ms2 += 1
            scan = int(spec.get("num"))
            psm = scan_to_psm.get(scan)
            if psm is None:
                continue
            seq, charge = psm
            try:
                pepmass = spec["precursorMz"][0].get("precursorMz")
            except (KeyError, IndexError):
                continue
            rt = spec.get("retentionTime")  # pyteomics returns seconds (float)
            out.write("BEGIN IONS\n")
            out.write(f"TITLE=scan={scan}\n")
            out.write(f"PEPMASS={pepmass}\n")
            if rt is not None:
                out.write(f"RTINSECONDS={float(rt)}\n")
            out.write(f"CHARGE={charge}+\n")
            out.write(f"SCANS={scan}\n")
            out.write(f"SEQ={seq}\n")
            for mz, it in zip(spec["m/z array"], spec["intensity array"]):
                out.write(f"{mz} {it}\n")
            out.write("END IONS\n\n")
            n_written += 1
    return n_ms2, n_written


_ANNOTATE_BY_EXT = {"mzml": annotate_mzml, "mzxml": annotate_mzxml, "mgf": annotate_mgf_from_mgf}

# beslic tryptic chains: (mab_id, sample_basename, ext).
BESLIC_MABS = [
    ("IgG1_Human_H", "Heavy-Chain-Trypsin-1", "mgf"),
    ("IgG1_Human_L", "Light-Chain-Trypsin-1", "mgf"),
]


def _annotate_beslic(args):
    only = set(args.mabs) or None
    for mab_id, basename, ext in BESLIC_MABS:
        if only is not None and mab_id not in only:
            continue
        d = f"{BESLIC_DIR}/{mab_id}"
        spec_path = f"{d}/mzml/{basename}.{ext}"
        out_mgf = f"{d}/{basename}_annotated.mgf"
        print(f"\n=== {mab_id} ===")
        try:
            psm_tsv = find_psm_tsv(f"{d}/fragpipe_workdir")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        scan_to_psm, n_rows = scan_to_psm_map(psm_tsv)
        print(f"  PSMs at 1% FDR: {n_rows}; unique scans: {len(scan_to_psm)}")
        if not os.path.exists(spec_path):
            print(f"  SKIP: spectra file missing {spec_path}")
            continue
        annotate = _ANNOTATE_BY_EXT.get(ext.lower())
        if annotate is None:
            print(f"  SKIP: unknown extension {ext}")
            continue
        n_ms2, n_written = annotate(spec_path, scan_to_psm, out_mgf)
        print(f"  MS2 in spectra: {n_ms2}; MGF blocks written: {n_written}")
        print(f"  -> {out_mgf}")


def _annotate_nontryp(args):
    idx_set = set(args.idx) if args.idx else set(range(len(RUNS)))
    for i, run in enumerate(RUNS):
        if i not in idx_set:
            continue
        print(f"\n[{i:2d}] {run.mab_id:14s} {run.protease:12s}")
        psm_tsv = find_psm_tsv(run.workdir, required=False)
        if psm_tsv is None:
            print(f"     SKIP: no psm.tsv under {run.workdir}")
            continue
        scan_to_psm, n_rows = scan_to_psm_map(psm_tsv)
        print(f"     PSMs 1% FDR: {n_rows}; unique scans: {len(scan_to_psm)}")
        if not os.path.exists(run.prepared_mgf):
            print(f"     SKIP: missing prepared MGF {run.prepared_mgf}")
            continue
        n_in, n_out = annotate_mgf_from_mgf(run.prepared_mgf, scan_to_psm, run.annotated_mgf)
        print(f"     MS2 in:  {n_in}  annotated: {n_out}")
        print(f"     -> {run.annotated_mgf}")


# ══════════════════════════════════════════════════════════════════════════
# prepare-nontryp: per-(mAb, protease) FragPipe workdirs from the registry
# ══════════════════════════════════════════════════════════════════════════
# Tryptic FragPipe workflow used as the enzyme/DB template for every non-tryptic run.
TRYP_WORKFLOW_TEMPLATE = os.path.join(
    DATA_PATH, "beslic_mab", "IgG1_Human_H", "fragpipe_workdir", "fragpipe.workflow"
)
_TITLE_SCAN_RE = re.compile(r"Scan:\s*(\d+)")


def _fixup_mgf(src: str, dst: str, sample: str) -> tuple[int, int]:
    """Rewrite TITLE to `<sample>.<scan>.<scan>.<charge>` and ensure SCANS=<scan>.

    Source variants: Beslic Centroided (TITLE has 'Scan: N', no SCANS=) and
    PXD023419 RAW (TITLE is raw filename, SCANS= already present).
    """
    n_in = n_out = 0
    with open(src) as fi, open(dst, "w") as fo:
        block, scan, charge = [], None, None
        for line in fi:
            if line.startswith("BEGIN IONS"):
                block, scan, charge = [line], None, None
                continue
            if line.startswith("END IONS"):
                n_in += 1
                if scan is not None and charge is not None:
                    new_block, title_done, scans_done = [], False, False
                    for bl in block:
                        if bl.startswith("TITLE="):
                            if not title_done:
                                new_block.append(f"TITLE={sample}.{scan}.{scan}.{charge}\n")
                                title_done = True
                            continue
                        if bl.startswith("SCANS="):
                            continue  # re-emit canonical SCANS below
                        new_block.append(bl)
                    out_lines = []
                    for bl in new_block:
                        out_lines.append(bl)
                        if not scans_done and bl.startswith("CHARGE="):
                            out_lines.append(f"SCANS={scan}\n")
                            scans_done = True
                    if not title_done:
                        out_lines.insert(1, f"TITLE={sample}.{scan}.{scan}.{charge}\n")
                    if not scans_done:
                        out_lines.append(f"SCANS={scan}\n")
                    fo.writelines(out_lines)
                    fo.write("END IONS\n")
                    n_out += 1
                block = []
                continue
            block.append(line)
            if line.startswith("TITLE="):
                m = _TITLE_SCAN_RE.search(line)
                if m:
                    scan = int(m.group(1))
            elif line.startswith("SCANS=") and scan is None:
                try:
                    scan = int(line.strip().split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("CHARGE="):
                m = re.match(r"^CHARGE=(\d+)", line)
                if m:
                    charge = int(m.group(1))
    return n_in, n_out


def _patch_workflow(template_text: str, run: NonTrypRun) -> str:
    """Swap database/enzyme params in the tryptic template for this run."""
    enz = run.enzyme
    replacements = {
        "database.db-path":                    run.fasta_td,
        "msfragger.search_enzyme_name_1":      enz.name,
        "msfragger.search_enzyme_cut_1":       enz.cut if enz.cut != "-" else "",
        "msfragger.search_enzyme_nocut_1":     enz.nocut,
        "msfragger.search_enzyme_sense_1":     enz.sense,
        "msfragger.allowed_missed_cleavage_1": str(enz.missed_cleavage),
        # Semi-specific: one peptide terminus matches the enzyme rule, the other
        # can be anywhere. Captures ragged mAb termini (signal-peptide remnants,
        # pyroglutamate, C-terminal K clipping) and the looseness of elastase /
        # thermolysin / lysN that fully-specific (=2) would miss.
        "msfragger.num_enzyme_termini":        "1",
        # Single-mAb databases have only 2-4 proteins (target+decoy of H/L
        # chains), so Philosopher's protein-FDR filter can't converge even at
        # --prot 1.00 under semi-specific. PSM-level FDR (1%) is what we need for
        # the ground truth, so disable protein-level inference entirely.
        "protein-prophet.run-protein-prophet": "false",
    }
    if enz.name == "nonspecific":
        replacements["msfragger.num_enzyme_termini"] = "0"
        replacements["msfragger.allowed_missed_cleavage_1"] = "0"
    out = []
    for line in template_text.splitlines(keepends=True):
        key = line.split("=", 1)[0] if "=" in line else line
        out.append(f"{key}={replacements[key]}\n" if key in replacements else line)
    return "".join(out)


def _prepare_one(run: NonTrypRun, force: bool = False) -> dict:
    os.makedirs(os.path.dirname(run.prepared_mgf), exist_ok=True)
    os.makedirs(run.workdir, exist_ok=True)
    os.makedirs(run.res_dir, exist_ok=True)

    if force or not os.path.exists(run.prepared_mgf):
        n_in, n_out = _fixup_mgf(run.src_mgf, run.prepared_mgf, run.sample)
        status_mgf = f"rewritten {n_in}->{n_out}"
    else:
        status_mgf = "exists"

    workflow_path = os.path.join(run.workdir, "fragpipe.workflow")
    with open(TRYP_WORKFLOW_TEMPLATE) as fi:
        patched = _patch_workflow(fi.read(), run)
    with open(workflow_path, "w") as fo:
        fo.write(patched)

    exp = re.sub(r"[^A-Za-z0-9_]+", "_", run.sample).strip("_").lower()
    manifest_path = os.path.join(run.workdir, "fragpipe-files.fp-manifest")
    with open(manifest_path, "w") as fo:
        fo.write(f"{run.prepared_mgf}\t{exp}\t1\tDDA\n")

    return {"mgf": status_mgf, "workflow": workflow_path,
            "manifest": manifest_path, "exp_name": exp}


def _prepare_nontryp(args):
    idx_set = set(args.idx) if args.idx else set(range(len(RUNS)))
    for i, run in enumerate(RUNS):
        if i not in idx_set:
            continue
        print(f"[{i:2d}] {run.mab_id:14s} {run.protease:12s}", flush=True)
        if args.check:
            ok = "OK" if os.path.exists(run.src_mgf) else "MISSING"
            print(f"     src    : {run.src_mgf} ({ok})")
            print(f"     prepd  : {run.prepared_mgf}")
            print(f"     workdir: {run.workdir}")
            continue
        r = _prepare_one(run, force=args.force)
        print(f"     mgf      : {r['mgf']}")
        print(f"     workflow : {r['workflow']}")
        print(f"     manifest : {r['manifest']}")
        print(f"     exp_name : {r['exp_name']}")


# ══════════════════════════════════════════════════════════════════════════
# stage-beslic: per-mAb dirs, TD FASTA, spectra hardlinks, FragPipe inputs
# ══════════════════════════════════════════════════════════════════════════
_FIG = os.path.join(DATA_PATH, "figshare_21394143", "01-RawData")
_TPL_57 = os.path.join(DATA_PATH, "PXD057525", "fragpipe_workdir")  # IAM, +57.02146
_TPL_58 = os.path.join(DATA_PATH, "PXD023419", "fragpipe_workdir")  # IAA, +58.005478

# (mab_id, fasta_name, sample_basename, ext, cys_mass, template_dir)
STAGE_MABS = [
    ("IgG1_Human_H", "Human_HeavyChain", "Heavy-Chain-Trypsin-1", "mzxml", 57.02146, _TPL_57),
    ("IgG1_Human_L", "Human_LightChain", "Light-Chain-Trypsin-1", "mzxml", 57.02146, _TPL_57),
]


def _read_fasta(path):
    entries, name, seq_lines = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    entries.append((name, "".join(seq_lines)))
                name, seq_lines = line[1:].split()[0], []
            elif line:
                seq_lines.append(line)
        if name is not None:
            entries.append((name, "".join(seq_lines)))
    return entries


def _build_td_fasta(target_fa, out_td, decoy_prefix="DECOY_"):
    entries = _read_fasta(target_fa)
    with open(out_td, "w") as f:
        for name, seq in entries:
            f.write(f">{name}\n{seq}\n")
        for name, seq in entries:
            f.write(f">{decoy_prefix}{name}\n{seq[::-1]}\n")
    return [n for n, _ in entries]


def _stage_spectra(src, dst):
    """Hardlink if possible (same filesystem), else symlink."""
    if os.path.lexists(dst):
        os.unlink(dst)
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def _adapt_workflow(src, dst_workflow, mab_dir, td_fasta_path, expected_cys):
    """Copy the FragPipe template, rewriting db-path + workdir; sanity-check Cys mod."""
    with open(src) as f:
        lines = f.read().splitlines()
    out = []
    for line in lines:
        if line.startswith("database.db-path="):
            out.append(f"database.db-path={td_fasta_path}")
        elif line.startswith("workdir="):
            out.append(f"workdir={mab_dir}/fragpipe_workdir")
        else:
            out.append(line)
    text = "\n".join(out) + "\n"
    cys_token = f"{expected_cys},C (cysteine),true,-1"
    if cys_token not in text:
        raise SystemExit(f"template {src} lacks expected Cys fix-mod token {cys_token!r}")
    with open(dst_workflow, "w") as f:
        f.write(text)


def _stage_beslic(args):
    only = set(args.mabs) or None
    os.makedirs(BESLIC_DIR, exist_ok=True)
    for mab_id, fasta_name, sample, ext, cys, tpl in STAGE_MABS:
        if only is not None and mab_id not in only:
            continue
        d = os.path.join(BESLIC_DIR, mab_id)
        for sub in ("", "mzml", "fragpipe_workdir", "casanovo_results"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)

        # 1. Target+decoy FASTA.
        target = f"{_FIG}/FASTAs/{fasta_name}.fasta"
        if not os.path.exists(target):
            raise FileNotFoundError(f"target FASTA not found: {target}")
        td_path = os.path.join(d, f"{fasta_name}_td.fasta")
        names = _build_td_fasta(target, td_path)
        print(f"[{mab_id}] TD FASTA {td_path}  ({len(names)}->{2*len(names)} entries)")

        # 2. Spectra hardlink.
        src = f"{_FIG}/MSV000079801_CompleteAssembly/{sample}.{ext}"
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        dst = os.path.join(d, "mzml", f"{sample}.{ext}")
        _stage_spectra(src, dst)
        print(f"[{mab_id}] spectra {dst} ({os.path.getsize(src)/1e6:.1f} MB)")

        # 3. FragPipe workflow + manifest (+ msbooster params).
        _adapt_workflow(f"{tpl}/fragpipe.workflow",
                        f"{d}/fragpipe_workdir/fragpipe.workflow", d, td_path, cys)
        exp = mab_id.lower()
        with open(f"{d}/fragpipe_workdir/fragpipe-files.fp-manifest", "w") as f:
            f.write(f"{d}/mzml/{sample}.{ext}\t{exp}\t1\tDDA\n")
        aux = f"{tpl}/msbooster_params.txt"
        if os.path.exists(aux):
            shutil.copy(aux, f"{d}/fragpipe_workdir/msbooster_params.txt")
        print(f"[{mab_id}] workflow + manifest written (Cys=+{cys})")


# ══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("stage-beslic", help="stage beslic tryptic chains")
    ps.add_argument("mabs", nargs="*", help="mab ids (default: all)")
    ps.set_defaults(func=_stage_beslic)

    pp = sub.add_parser("prepare-nontryp", help="build per-protease FragPipe workdirs")
    pp.add_argument("--idx", type=int, nargs="*", help="indices into RUNS (default: all)")
    pp.add_argument("--force", action="store_true", help="re-rewrite prepared MGFs")
    pp.add_argument("--check", action="store_true", help="report planned actions only")
    pp.set_defaults(func=_prepare_nontryp)

    pa = sub.add_parser("annotate", help="insert SEQ= from FragPipe psm.tsv")
    asub = pa.add_subparsers(dest="which", required=True)
    ab = asub.add_parser("beslic", help="beslic tryptic chains")
    ab.add_argument("mabs", nargs="*", help="mab ids (default: all)")
    ab.set_defaults(func=_annotate_beslic)
    an = asub.add_parser("nontryp", help="non-tryptic registry runs")
    an.add_argument("--idx", type=int, nargs="*", help="indices into RUNS (default: all)")
    an.set_defaults(func=_annotate_nontryp)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
