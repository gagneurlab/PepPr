"""Stage per-mAb dirs under /s/.../beslic_mab/ for FragPipe + Casanovo two-arm runs.

Each mAb gets:
  <dir>/<chain>_td.fasta              -- target+decoy with DECOY_ prefix (matches existing PXD023419/PXD057525 setup)
  <dir>/mzml/<sample>.mzML or .mzXML  -- hardlink to figshare bundle (no copy)
  <dir>/fragpipe_workdir/             -- pre-seeded with workflow + manifest + fragger.params
  <dir>/casanovo_results/             -- empty, populated by run_casanovo_mab.sh
"""

import os
import shutil

FIG = "/s/project/denovo-prosit/SamKhan/dnps_hybrid/figshare_21394143/01-RawData"
BESLIC = "/s/project/denovo-prosit/SamKhan/dnps_hybrid/beslic_mab"

# (mab_id, target_fasta_name, prefix_for_fasta_entry, src_dir, spectra_basename, spectra_ext, cys_mass)
MABS = [
    ("IgG1_Human_H",   "Human_HeavyChain",   "Human_HeavyChain",
        f"{FIG}/MSV000079801_CompleteAssembly", "Heavy-Chain-Trypsin-1",  "mzxml", 57.02146),
    ("IgG1_Human_L",   "Human_LightChain",   "Human_LightChain",
        f"{FIG}/MSV000079801_CompleteAssembly", "Light-Chain-Trypsin-1",  "mzxml", 57.02146),
]


def read_fasta(path):
    entries = []
    with open(path) as f:
        name = None
        seq_lines = []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    entries.append((name, "".join(seq_lines)))
                name = line[1:].split()[0]
                seq_lines = []
            elif line:
                seq_lines.append(line)
        if name is not None:
            entries.append((name, "".join(seq_lines)))
    return entries


def build_td_fasta(target_fa, out_td, decoy_prefix="DECOY_"):
    entries = read_fasta(target_fa)
    with open(out_td, "w") as f:
        for name, seq in entries:
            f.write(f">{name}\n{seq}\n")
        for name, seq in entries:
            f.write(f">{decoy_prefix}{name}\n{seq[::-1]}\n")
    return [n for n, _ in entries]


def stage_spectra(src, dst):
    """Hardlink if possible (same filesystem), else symlink."""
    if os.path.lexists(dst):
        os.unlink(dst)
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def find_target_fasta(name, mab_id):
    cand = f"{FIG}/FASTAs/{name}.fasta"
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"target FASTA for {name} not found")


def main():
    os.makedirs(BESLIC, exist_ok=True)
    for mab_id, fasta_name, _prefix, src_dir, basename, ext, _cys in MABS:
        d = os.path.join(BESLIC, mab_id)
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "mzml"), exist_ok=True)
        os.makedirs(os.path.join(d, "fragpipe_workdir"), exist_ok=True)
        os.makedirs(os.path.join(d, "casanovo_results"), exist_ok=True)

        # TD FASTA
        target = find_target_fasta(fasta_name, mab_id)
        td_path = os.path.join(d, f"{fasta_name}_td.fasta")
        names = build_td_fasta(target, td_path)
        print(f"[{mab_id}] target {target}  ->  TD {td_path}  ({len(names)} target -> {2*len(names)} entries)")

        # Spectra
        src = os.path.join(src_dir, f"{basename}.{ext}")
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        dst = os.path.join(d, "mzml", f"{basename}.{ext}")
        stage_spectra(src, dst)
        print(f"[{mab_id}] spectra {src}  ->  {dst} ({os.path.getsize(src)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
