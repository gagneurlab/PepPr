"""Build SEQ-annotated MGFs for each beslic_mab mAb from FragPipe psm.tsv + mzML/mzXML.

Handles both mzML (scan id like 'controllerType=0 controllerNumber=1 scan=1234')
and mzXML (scan `num="1234"`).

Outputs <beslic_mab>/<mab_id>/<sample_basename>_annotated.mgf  with SEQ= per PSM scan.
"""

import os
import re
import sys
from collections import defaultdict

from pyteomics import mgf, mzml, mzxml

BESLIC = "/s/project/denovo-prosit/SamKhan/dnps_hybrid/beslic_mab"

# (mab_id, sample_basename, ext)
# After fixup_msv_mgf.py the MSV samples ship as MGF (centroided) with
# TITLE=<sample>.<scan>.<scan>.<charge> + SCANS=<scan>.
MABS = [
    ("IgG1_Human_H",  "Heavy-Chain-Trypsin-1",        "mgf"),
    ("IgG1_Human_L",  "Light-Chain-Trypsin-1",        "mgf"),
]


def parse_psm_tsv(path):
    """Yield (raw_basename, scan, charge, peptide, modifications)."""
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_spec = header.index("Spectrum")
        i_pep = header.index("Peptide")
        i_z = header.index("Charge")
        i_mods = header.index("Assigned Modifications")
        for line in f:
            row = line.rstrip("\n").split("\t")
            spec = row[i_spec]
            parts = spec.rsplit(".", 3)
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


def find_psm_tsv(workdir):
    """psm.tsv is under fragpipe_workdir/<experiment>_<bioreplicate>/psm.tsv."""
    for entry in os.listdir(workdir):
        full = os.path.join(workdir, entry)
        if os.path.isdir(full):
            p = os.path.join(full, "psm.tsv")
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"no psm.tsv under {workdir}")


def write_annotated_mzml(mzml_path, scan_to_psm, out_mgf):
    """mzML spectra read with pyteomics.mzml."""
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
            scan = int(m.group(1))
            psm = scan_to_psm.get(scan)
            if psm is None:
                continue
            seq, charge = psm
            try:
                pl = spec["precursorList"]["precursor"][0]
                sil = pl["selectedIonList"]["selectedIon"][0]
                pepmass = sil.get("selected ion m/z")
            except (KeyError, IndexError):
                continue
            try:
                rt_min = spec["scanList"]["scan"][0].get("scan start time")
                rt_sec = float(rt_min) * 60.0 if rt_min is not None else None
            except (KeyError, IndexError):
                rt_sec = None
            mz_arr = spec["m/z array"]
            int_arr = spec["intensity array"]
            out.write("BEGIN IONS\n")
            out.write(f"TITLE={scan_id}\n")
            out.write(f"PEPMASS={pepmass}\n")
            if rt_sec is not None:
                out.write(f"RTINSECONDS={rt_sec}\n")
            out.write(f"CHARGE={charge}+\n")
            out.write(f"SCANS={scan}\n")
            out.write(f"SEQ={seq}\n")
            for mz, it in zip(mz_arr, int_arr):
                out.write(f"{mz} {it}\n")
            out.write("END IONS\n\n")
            n_written += 1
    return n_ms2, n_written


def write_annotated_mgf_from_mgf(mgf_in, scan_to_psm, out_mgf):
    """Source MGF already has SCANS=<scan>. Emit a new MGF with SEQ= for PSMs that match."""
    n_in = n_written = 0
    with open(mgf_in) as fi, open(out_mgf, "w") as fo:
        block = []
        scan = None
        for line in fi:
            if line.startswith("BEGIN IONS"):
                block = [line]
                scan = None
                continue
            if line.startswith("END IONS"):
                n_in += 1
                if scan is not None and scan in scan_to_psm:
                    seq, _charge = scan_to_psm[scan]
                    # Insert SEQ= line just before the m/z lines start.
                    # The header lines come first (TITLE, PEPMASS, CHARGE, SCANS, RTINSECONDS).
                    # We add SEQ= right after SCANS=<scan>.
                    out_lines = []
                    inserted = False
                    for bl in block:
                        out_lines.append(bl)
                        if not inserted and bl.startswith("SCANS="):
                            out_lines.append(f"SEQ={seq}\n")
                            inserted = True
                    if not inserted:
                        # fall back: insert after CHARGE
                        out_lines = []
                        inserted = False
                        for bl in block:
                            out_lines.append(bl)
                            if not inserted and bl.startswith("CHARGE="):
                                out_lines.append(f"SEQ={seq}\n")
                                inserted = True
                    fo.writelines(out_lines)
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


def write_annotated_mzxml(mzxml_path, scan_to_psm, out_mgf):
    """mzXML spectra: spec['num'] is the scan number (string in pyteomics)."""
    n_ms2 = n_written = 0
    with open(out_mgf, "w") as out, mzxml.read(mzxml_path) as reader:
        for spec in reader:
            ms_level = spec.get("msLevel")
            if ms_level != 2:
                continue
            n_ms2 += 1
            scan = int(spec.get("num"))
            psm = scan_to_psm.get(scan)
            if psm is None:
                continue
            seq, charge = psm
            # mzXML: precursor info under 'precursorMz' (list of dicts)
            try:
                prec = spec["precursorMz"][0]
                pepmass = prec.get("precursorMz")
            except (KeyError, IndexError):
                continue
            rt = spec.get("retentionTime")  # already in seconds (pyteomics returns float)
            mz_arr = spec["m/z array"]
            int_arr = spec["intensity array"]
            out.write("BEGIN IONS\n")
            out.write(f"TITLE=scan={scan}\n")
            out.write(f"PEPMASS={pepmass}\n")
            if rt is not None:
                out.write(f"RTINSECONDS={float(rt)}\n")
            out.write(f"CHARGE={charge}+\n")
            out.write(f"SCANS={scan}\n")
            out.write(f"SEQ={seq}\n")
            for mz, it in zip(mz_arr, int_arr):
                out.write(f"{mz} {it}\n")
            out.write("END IONS\n\n")
            n_written += 1
    return n_ms2, n_written


def main():
    only = set(sys.argv[1:]) or None
    for mab_id, basename, ext in MABS:
        if only is not None and mab_id not in only:
            continue
        d = f"{BESLIC}/{mab_id}"
        workdir = f"{d}/fragpipe_workdir"
        spec_path = f"{d}/mzml/{basename}.{ext}"
        out_mgf = f"{d}/{basename}_annotated.mgf"
        try:
            psm_tsv = find_psm_tsv(workdir)
        except FileNotFoundError as e:
            print(f"[{mab_id}] SKIP: {e}")
            continue
        scan_to_psm = {}
        n_rows = 0
        for raw_base, scan, z, pep, mods in parse_psm_tsv(psm_tsv):
            n_rows += 1
            if scan in scan_to_psm:
                continue
            seq = mods_to_proforma(pep, mods)
            scan_to_psm[scan] = (seq, z)
        print(f"\n=== {mab_id} ===")
        print(f"  PSMs at 1% FDR: {n_rows}; unique scans: {len(scan_to_psm)}")
        if not os.path.exists(spec_path):
            print(f"  SKIP: spectra file missing {spec_path}")
            continue
        ext_l = ext.lower()
        if ext_l == "mzml":
            n_ms2, n_written = write_annotated_mzml(spec_path, scan_to_psm, out_mgf)
        elif ext_l == "mzxml":
            n_ms2, n_written = write_annotated_mzxml(spec_path, scan_to_psm, out_mgf)
        elif ext_l == "mgf":
            n_ms2, n_written = write_annotated_mgf_from_mgf(spec_path, scan_to_psm, out_mgf)
        else:
            print(f"  SKIP: unknown extension {ext}")
            continue
        print(f"  MS2 in spectra: {n_ms2}; MGF blocks written: {n_written}")
        print(f"  -> {out_mgf}")


if __name__ == "__main__":
    main()
