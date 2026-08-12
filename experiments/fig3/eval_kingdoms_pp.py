#!/usr/bin/env python3
"""Evaluate Casanovo and Casanovo + a chosen pepLM (PP) on Kingdoms-of-Life
species, using the ProForma-annotated MGFs produced by
``peptide_priors.prepare_data.prepare_kol_species_dataset()``.

For each ``--species`` (e.g. human, mouse, …):
  1. Read the archived, fixed 10k-spectrum subset from
     ``<mgf-root>/<species>.mgf``.
  2. Run ``casanovo sequence -e`` twice — once with ``--use_plm false``
     (Casanovo) and once with ``--use_plm true`` while
     ``DNPS_PLM_SPECIES=<plm-species>`` (Casanovo + that species' PP).
     Beam width is taken from config.yaml (n_beams=5).
  3. Tee casanovo's stdout/stderr to a per-run log; mztab lands next to it.
  4. Summarize peptide precision (percent, 0–100) from the mztab plus the
     subset MGF (``load_mztab_with_mgf`` + Casanovo's ``aa_match_batch`` /
     ``aa_match_metrics``).

The MGFs already carry ProForma SEQ= lines, so no annotation conversion or
species filtering is needed here — that all happened in prepare_data.py.

Note that ``--species`` (eval data) and ``--plm-species`` (PP weights) are
independent: the canonical "cross-species transfer" experiment is
``--species mouse --plm-species human`` — i.e. evaluate the human PP on
mouse spectra.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Resolve per-pepLM fusion weights (see peptide_priors.const.FUSION_MODEL_PATH).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from peptide_priors.const import (  # noqa: E402
    CASANOVO_CONFIG_YAML,
    KOL_RESULTS_DIR,
    KOL_SUBSETS_DIR,
    MODELS_DIR,
    SPECIES as _PLM_SPECIES_CFG,
)
from experiments.utils.evaluation import (  # noqa: E402
    MASSIVEKB_MASSES,
    _expand_masses,
    _normalize_to_massivekb,
    evaluate,
    load_mztab_with_mgf,
)

DEFAULT_KOL_ROOT = Path(KOL_SUBSETS_DIR)

DEFAULT_OUT = KOL_RESULTS_DIR
DEFAULT_CKPT = "https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt"
DEFAULT_CONFIG = CASANOVO_CONFIG_YAML

# casanovo/casanovo/config.yaml ships with `max_charge: 4`. The charge encoder
# is nn.Embedding(max_charge, d_model) indexed by `charge - 1`, so spectra
# with charge > 4 trip a CUDA OOB.
DEFAULT_MAX_CHARGE = 4


def _iter_mgf_blocks(path: Path):
    """Yield (charge, block_lines) for each spectrum in an MGF file.

    ``charge`` is parsed from the ``CHARGE=`` header (None if missing/bad).
    ``block_lines`` is the raw line list from BEGIN IONS through END IONS,
    suitable for re-emitting verbatim.
    """
    block: list[str] = []
    in_block = False
    charge: int | None = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("BEGIN IONS"):
                block = [line]
                charge = None
                in_block = True
                continue
            if not in_block:
                continue
            block.append(line)
            if line.startswith("CHARGE="):
                try:
                    charge = int(line.split("=", 1)[1].strip().rstrip("+"))
                except ValueError:
                    charge = None
            elif line.startswith("END IONS"):
                yield charge, block
                in_block = False
                block = []


def sample_mgfs(
    mgf_dir: Path,
    n: int,
    seed: int,
    out_path: Path,
    max_charge: int = DEFAULT_MAX_CHARGE,
) -> int:
    """Draw ``n`` spectra (charge in [1, max_charge]) from all .mgf files in
    ``mgf_dir`` and write them to ``out_path`` as a single concatenated MGF.

    Sampling is uniform across the eligible global pool with a fixed seed, so
    repeated calls produce identical subsets. Returns the number of spectra
    actually written (≤ ``n``; equals total eligible if pool is smaller).
    """
    files = sorted(mgf_dir.glob("*.mgf"))
    if not files:
        raise RuntimeError(f"no .mgf files in {mgf_dir}")

    eligible_per_file: list[int] = []
    for f in files:
        c = sum(
            1 for ch, _ in _iter_mgf_blocks(f)
            if ch is not None and 1 <= ch <= max_charge
        )
        eligible_per_file.append(c)
    total = sum(eligible_per_file)
    print(f"  eligible spectra (charge ≤ {max_charge}): {total:,} across {len(files)} files")
    if total == 0:
        raise RuntimeError(f"no eligible spectra in {mgf_dir}")

    rng = np.random.default_rng(seed)
    if n >= total:
        keep = np.ones(total, dtype=bool)
    else:
        chosen = rng.choice(total, size=n, replace=False)
        keep = np.zeros(total, dtype=bool)
        keep[chosen] = True

    written = 0
    idx = 0
    with open(out_path, "w") as out:
        for f in files:
            for ch, block in _iter_mgf_blocks(f):
                if ch is None or not (1 <= ch <= max_charge):
                    continue
                if keep[idx]:
                    out.writelines(block)
                    written += 1
                idx += 1
    return written


def count_mgf_spectra(path: Path) -> int:
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("BEGIN IONS"):
                n += 1
    return n


def run_casanovo(
    input_path: Path,
    out_dir: Path,
    out_root: str,
    use_plm: bool,
    plm_species: str,
    ckpt: str,
    config: str,
    log_path: Path,
) -> int:
    """Invoke ``casanovo sequence -e`` and tee output to ``log_path``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "casanovo",
        "sequence",
        "-m", ckpt,
        "-c", config,
        "-d", str(out_dir),
        "-o", out_root,
        "-e",
        "--teacher_forcing", "false",
        "--use_plm", "true" if use_plm else "false",
        "--force_overwrite",
        str(input_path),
    ]
    env = os.environ.copy()
    # PLM checkpoint: PLM_RUN_PATH in const (pepLM weights for plm_species).
    env["DNPS_PLM_SPECIES"] = plm_species
    # Null / RUN_PATH follow ACTIVE_SPECIES; align with pepLM so the null
    # baseline matches the fusion head's training regime.
    env.setdefault("DNPS_SPECIES", plm_species)
    if use_plm:
        run_name = _PLM_SPECIES_CFG[plm_species]["run_name"]
        fusion_p = os.path.join(MODELS_DIR, run_name, "fusion_model.pth")
        env.setdefault("DNPS_FUSION_MODEL_PATH", fusion_p)

    print(f"  $ {' '.join(cmd)}")
    print(
        f"    DNPS_PLM_SPECIES={env['DNPS_PLM_SPECIES']}  "
        f"DNPS_SPECIES={env['DNPS_SPECIES']}"
        + (
            f"  DNPS_FUSION_MODEL_PATH={env.get('DNPS_FUSION_MODEL_PATH', '')}"
            f"  DNPS_NULL_MODEL_PATH={env.get('DNPS_NULL_MODEL_PATH', '')}"
            if use_plm
            else ""
        )
    )
    with open(log_path, "wb") as fh:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            fh.write(line)
        return proc.wait()


def metrics_from_mztab(mztab_path: Path) -> dict[str, float | None]:
    """Peptide precision (percent, 0–100) from mztab + MGF.

    Uses the same mass-based matching as Casanovo's evaluator
    (``evaluate.aa_match_batch`` / ``aa_match_metrics``) after normalizing
    sequences to MassIVE-KB token form.
    """
    empty = {"pep_precision": None}
    if not mztab_path.is_file():
        return empty
    df = load_mztab_with_mgf(str(mztab_path))
    df = df.dropna(subset=["true_seq", "pred"])
    if df.empty:
        return empty
    preds = df["pred"].apply(_normalize_to_massivekb).tolist()
    trues = df["true_seq"].apply(_normalize_to_massivekb).tolist()
    masses = _expand_masses(preds + trues, MASSIVEKB_MASSES)
    aa_batch, n_aa_true, n_aa_pred = evaluate.aa_match_batch(trues, preds, masses)
    _, _, pep_prec = evaluate.aa_match_metrics(aa_batch, n_aa_true, n_aa_pred)
    return {"pep_precision": 100.0 * float(pep_prec)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", default=["human"],
                        help="One or more species keys (matching subdirs under "
                             "--mgf-root, e.g. 'human', 'mouse'). Default: human.")
    parser.add_argument("--mgf-root", type=Path, default=DEFAULT_KOL_ROOT,
                        help=f"Directory containing fixed <species>.mgf subsets "
                             f"(default: {DEFAULT_KOL_ROOT}).")
    parser.add_argument("--n-spectra", type=int, default=10_000,
                        help="Deprecated; archived subsets are already fixed at 10k.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--plm-species", default="human_iso",
                        help="DNPS_PLM_SPECIES for the +PP run (default: human_iso). "
                             "Independent of --species; the canonical "
                             "transfer experiment runs --species X with "
                             "--plm-species human_iso.")
    parser.add_argument("--max-charge", type=int, default=DEFAULT_MAX_CHARGE,
                        help="Drop spectra with charge > this (default: 4 to fit "
                             "casanovo's charge embedding).")
    parser.add_argument("--modes", nargs="+", default=["casanovo", "casanovo_pp"],
                        choices=["casanovo", "casanovo_pp"])
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for species in args.species:
        print(f"\n=== {species} (PP={args.plm_species}) ===")
        subset_path = args.mgf_root / f"{species}.mgf"
        if not subset_path.is_file():
            print(f"  ! missing archived subset: {subset_path} — skipping")
            continue

        n_used = count_mgf_spectra(subset_path)
        print(f"  archived subset: {subset_path}")
        print(f"  subset spectra: {n_used:,}")

        sp_run_dir = runs_dir / species
        sp_run_dir.mkdir(parents=True, exist_ok=True)

        for mode in args.modes:
            use_plm = mode == "casanovo_pp"
            # "casanovo_only" rather than "casanovo" — click rejects bare
            # "casanovo" because there's a directory of that name in CWD.
            # The +PP filename includes the PLM species so cross-species
            # transfer runs (e.g. mouse data, human PP) don't collide.
            out_root = (
                f"casanovo_pp_{args.plm_species}" if use_plm else "casanovo_only"
            )
            mztab = sp_run_dir / f"{out_root}.mztab"
            log_path = sp_run_dir / f"{out_root}.log"
            if mztab.exists() and args.skip_existing:
                print(f"  reuse {mode}: {mztab}")
            else:
                # Casanovo calls check_dir_file_exists before writing; existing
                # *.mztab is not removed by --force_overwrite.
                for stale in sp_run_dir.glob(f"{out_root}*.mztab"):
                    stale.unlink()
                rc = run_casanovo(
                    subset_path, sp_run_dir, out_root, use_plm,
                    args.plm_species, args.ckpt, args.config, log_path,
                )
                if rc != 0:
                    print(f"  ! casanovo failed (rc={rc}) for {species} {mode}")
                    continue
            metrics = metrics_from_mztab(mztab)
            print(f"  {mode}: pep_precision={metrics['pep_precision']}")
            rows.append({
                "species": species,
                "plm_species": args.plm_species,
                "mode": mode,
                "n_spectra": n_used,
                **metrics,
            })

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w") as fh:
        fh.write("species,plm_species,mode,n_spectra,pep_precision\n")
        for r in rows:
            fh.write(
                f"{r['species']},{r['plm_species']},{r['mode']},{r['n_spectra']},"
                f"{r['pep_precision']}\n"
            )
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
