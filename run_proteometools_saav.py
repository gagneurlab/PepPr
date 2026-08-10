#!/usr/bin/env python3
"""De novo inference for the ProteomeTools SAAV benchmark: run Casanovo (use_plm=false)
and Casanovo+PepPr (use_plm=true) on the GT-search 1%-FDR spectra (mgf_gt, already in
ProForma form) -> proteometools_saav_{dnps,hybrid}.mztab. Paths from const."""
import os, glob
import dnps_hybrid.const as const

CKPT = "/s/project/denovo-prosit/CASANOVO/casanovo_5.0.0_model_weights/casanovo_v5_0_0.ckpt"
CONFIG = "/data/nasif12/home_if12/khsam/dnps_hybrid/casanovo/casanovo/config.yaml"

ds = const.PROTEOMETOOLS_SAAV_DATASET
mgf_files = ' '.join(sorted(glob.glob(ds.final_mgf_glob)))
assert mgf_files, f"no MGFs at {ds.final_mgf_glob!r}"

for use_plm, mztab_path in [("true", ds.mztab_path_fusion), ("false", ds.mztab_path_dnps)]:
    out_dir = os.path.dirname(mztab_path)
    base = os.path.basename(mztab_path).replace(".mztab", "")
    os.makedirs(out_dir, exist_ok=True)
    for ext in (".log", ".mztab"):            # casanovo refuses to overwrite
        p = os.path.join(out_dir, base + ext)
        if os.path.exists(p):
            os.remove(p)
    cmd = (f"casanovo sequence -m {CKPT} -c {CONFIG} -d {out_dir} -o {base} "
           f"--teacher_forcing false --use_plm {use_plm} -e {mgf_files}")
    print(cmd, flush=True)
    if os.system(cmd) != 0:
        raise RuntimeError(f"casanovo failed (use_plm={use_plm})")
