"""Generate FragPipe inputs (fragpipe.workflow, manifest, fragger.params, msbooster_params.txt)
for each mAb under /s/.../beslic_mab/<mab_id>/fragpipe_workdir/.

Templates from /s/.../PXD057525/fragpipe_workdir/ (IAM +57) and PXD023419 (IAA +58).
"""

import os
import shutil

BESLIC = "/s/project/denovo-prosit/SamKhan/dnps_hybrid/beslic_mab"
TPL_57 = "/s/project/denovo-prosit/SamKhan/dnps_hybrid/PXD057525/fragpipe_workdir"  # IAM, +57.02146
TPL_58 = "/s/project/denovo-prosit/SamKhan/dnps_hybrid/PXD023419/fragpipe_workdir"  # IAA, +58.005478

# (mab_id, td_fasta_basename, sample_basename, ext, cys_mass, template_dir)
MABS = [
    ("IgG1_Human_H",   "Human_HeavyChain_td.fasta", "Heavy-Chain-Trypsin-1",  "mzxml", 57.02146,  TPL_57),
    ("IgG1_Human_L",   "Human_LightChain_td.fasta", "Light-Chain-Trypsin-1",  "mzxml", 57.02146,  TPL_57),
]

OLD_FIX57 = "57.02146,C (cysteine),true,-1"
OLD_FIX58 = "58.005478,C (cysteine),true,-1"


def adapt_workflow(src, dst_workflow, mab_dir, td_fasta_path, expected_cys):
    """Copy workflow, rewriting database.db-path, workdir, and (sanity) the Cys fix-mod."""
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
    # Sanity check the templated Cys fix mod matches the expected mass for this template.
    cys_token = f"{expected_cys},C (cysteine),true,-1"
    if cys_token not in text:
        raise SystemExit(f"template {src} doesn't contain expected Cys fix-mod token {cys_token!r}")
    with open(dst_workflow, "w") as f:
        f.write(text)


def write_manifest(mab_dir, sample_basename, ext):
    """FragPipe manifest is tab-separated: file<TAB>experiment<TAB>bioreplicate<TAB>data_type."""
    sample_path = f"{mab_dir}/mzml/{sample_basename}.{ext}"
    exp = os.path.basename(mab_dir).lower()
    manifest = f"{sample_path}\t{exp}\t1\tDDA\n"
    with open(f"{mab_dir}/fragpipe_workdir/fragpipe-files.fp-manifest", "w") as f:
        f.write(manifest)


def main():
    for mab_id, td_name, sample_basename, ext, cys, tpl in MABS:
        mab_dir = f"{BESLIC}/{mab_id}"
        td_path = f"{mab_dir}/{td_name}"
        if not os.path.exists(td_path):
            raise SystemExit(f"missing TD fasta {td_path} -- run stage_beslic_mabs.py first")

        # 1) fragpipe.workflow
        src_workflow = f"{tpl}/fragpipe.workflow"
        dst_workflow = f"{mab_dir}/fragpipe_workdir/fragpipe.workflow"
        adapt_workflow(src_workflow, dst_workflow, mab_dir, td_path, cys)
        # 2) manifest
        write_manifest(mab_dir, sample_basename, ext)
        # 3) msbooster_params.txt: copy verbatim if present
        for aux in ("msbooster_params.txt",):
            src = f"{tpl}/{aux}"
            if os.path.exists(src):
                shutil.copy(src, f"{mab_dir}/fragpipe_workdir/{aux}")

        print(f"[{mab_id}] workflow + manifest written  (Cys=+{cys}, tpl={os.path.basename(os.path.dirname(tpl))})")


if __name__ == "__main__":
    main()
