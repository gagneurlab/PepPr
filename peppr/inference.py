import argparse
import torch
import sys
import os
import re
import tempfile
from peppr import const
from tqdm import tqdm
from peppr.model import load_prior_model, load_fusion_model
import glob


CASANOVO_CONFIG_DEFAULT = const.CASANOVO_CONFIG_YAML


def resolve_casanovo_config():
    lance_dir = os.environ.get(
        "PEPPR_LANCE_DIR", os.path.join(const.WORK_DIR, "lance")
    )
    os.makedirs(lance_dir, exist_ok=True)
    with open(CASANOVO_CONFIG_DEFAULT) as f:
        cfg = f.read()
    cfg = re.sub(r"(?m)^lance_dir:.*$", f"lance_dir: {lance_dir}", cfg)
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="casanovo_cfg_")
    with os.fdopen(fd, "w") as f:
        f.write(cfg)
    return tmp




# Benchmark datasets are defined by whoever is driving inference, not by peppr:
# the paper analyses register theirs in experiments.paths, and a downstream user
# can point PEPPR_DATASETS_MODULE at any module exposing DatasetPaths attributes.
DATASETS_MODULE = os.environ.get("PEPPR_DATASETS_MODULE", "peppr.const")


def resolve_dataset(name: str, module_name: str | None = None):
    """Look up a DatasetPaths instance by attribute name in a datasets module."""
    import importlib

    module_name = module_name or DATASETS_MODULE
    module = importlib.import_module(module_name)
    if not hasattr(module, name):
        raise ValueError(
            f"No dataset named {name!r} in {module_name}. Set "
            "PEPPR_DATASETS_MODULE to the module that defines it "
            "(the paper analyses use experiments.paths)."
        )
    return getattr(module, name)



def predict_prior_teacher(model, X, batch_size):
    scores = torch.zeros(size=(X.shape[0], const.PRIOR_BLOCK_SIZE, len(const.VOCAB)), device=const.DEVICE)
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), total=len(X)//batch_size):
            end = min(i+batch_size, len(X))
            logits, *_ = model(X[i:end])
            scores[i:end] = logits
    return scores


MODES = ("prior_teacher", "fusion_teacher", "contranovo", "auto")


def main(argv: list[str] | None = None) -> int:
    """Run one inference mode. Importing this module has no side effects."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in MODES:
        given = argv[0] if argv else "<none>"
        print(
            "usage: inference.py {" + "|".join(MODES) + "} [options]",
            file=sys.stderr,
        )
        print(f"error: unknown mode {given!r}", file=sys.stderr)
        return 2
    model_type = argv[0]

    torch.manual_seed(const.SEED)
    torch.cuda.manual_seed(const.SEED)
    torch.set_float32_matmul_precision("high")

    if model_type == 'prior_teacher':
        prior_model = load_prior_model()
        batch_size = 4096

        X_prior_train = torch.load(const.PRIOR_PSM_X_TRAIN_PATH, map_location=const.DEVICE)
        scores_prior_train = predict_prior_teacher(prior_model, X_prior_train, batch_size)
        torch.save(scores_prior_train, const.PRIOR_PSM_TEACHER_SCORES_TRAIN_PATH)
        print("Saved prior scores for train")

        X_prior_test = torch.load(const.PRIOR_PSM_X_TEST_PATH, map_location=const.DEVICE)
        scores_prior_test = predict_prior_teacher(prior_model, X_prior_test, batch_size)
        torch.save(scores_prior_test, const.PRIOR_PSM_TEACHER_SCORES_TEST_PATH)
        print("Saved prior scores for test")

        # X_plm_plm = torch.load(const.PRIOR_SEQ_X_PATH, map_location=const.DEVICE)
        # scores_plm_plm = predict_prior_teacher(prior_model, X_plm_plm, batch_size)
        # torch.save(scores_plm_plm, const.PRIOR_SEQ_TEACHER_SCORES_PATH)
    elif model_type == 'fusion_teacher':
        scores_prior_test = torch.load(const.PRIOR_PSM_TEACHER_SCORES_TEST_PATH)
        scores_casanovo_test = torch.load(const.CASANOVO_TEACHER_SCORES_TEST_PATH, map_location=const.DEVICE)
        print("Loaded prior and casanovo scores")
        X_fusion = torch.cat([scores_casanovo_test, scores_prior_test], dim=2)

        vocab_size = scores_casanovo_test.shape[2]
        fusion_model = load_fusion_model(vocab_size=vocab_size)
        batch_size = 4096
        scores_fusion = torch.zeros(size=(X_fusion.shape[0], const.PRIOR_BLOCK_SIZE, vocab_size), device=const.DEVICE)
        with torch.no_grad():
            for i in range(0, len(X_fusion), batch_size):
                end = min(i+batch_size, len(X_fusion))
                scores_fusion[i:end] = fusion_model(X_fusion[i:end].view(-1, X_fusion.size(2))).view(-1, const.PRIOR_BLOCK_SIZE, vocab_size)
                print(f"Processed {i} of {len(X_fusion)}")

        torch.save(scores_fusion, const.FUSION_SCORES_TEACHER_TEST_PATH)
    elif model_type == 'contranovo':
        print(f"=== ContraNovo +/- pepLM on species: {const.ACTIVE_SPECIES} ===")
        contranovo_root = os.path.join(const.PROJECT_ROOT, "ContraNovo")
        contranovo_python = const.CONTRANOVO_PYTHON
        if not contranovo_python:
            raise RuntimeError(
                "ContraNovo inference requires PEPPR_CONTRANOVO_PYTHON to point "
                "to the ContraNovo environment's Python executable."
            )
        contranovo_ckpt = os.path.join(contranovo_root, "ContraNovo", "ContraNovo.ckpt")
        contranovo_config = os.path.join(contranovo_root, "ContraNovo", "config.yaml")
        run_denovo = os.path.join(contranovo_root, "run_denovo.py")
        out_root = os.path.join(const.RESULT_RUN_PATH, "contranovo")
        os.makedirs(out_root, exist_ok=True)

        cn_dataset = resolve_dataset(
            os.environ.get("PEPPR_CONTRANOVO_DATASET", "NINE_SPECIES_DATASET")
        )
        mgf_files = sorted(glob.glob(cn_dataset.final_mgf_glob))
        if not mgf_files:
            raise FileNotFoundError(
                f"No MGFs matched {cn_dataset.final_mgf_glob!r}; "
                "run prepare_data first or set PEPPR_SPECIES correctly."
            )

        # ContraNovo's model.py imports peppr for the prior/fusion loaders, but
        # the khsam_contranovo env doesn't have peppr installed. Expose this
        # checkout on PYTHONPATH for the subprocess so the import resolves.
        peppr_root = const.PROJECT_ROOT
        existing_pp = os.environ.get("PYTHONPATH", "")
        sub_pythonpath = (
            f"{peppr_root}:{existing_pp}" if existing_pp else peppr_root
        )

        for use_peppr in ("false", "true"):
            suffix = "fusion" if use_peppr == "true" else "dnps"
            out_dir = os.path.join(out_root, f"{const.ACTIVE_SPECIES}_{suffix}")
            os.makedirs(out_dir, exist_ok=True)
            for mgf in mgf_files:
                base = os.path.splitext(os.path.basename(mgf))[0]
                out_csv = os.path.join(out_dir, f"{base}.csv")
                if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
                    print(f"[skip] {use_peppr=} {base}: already exists")
                    continue
                command = (
                    f"PYTHONPATH={sub_pythonpath} "
                    f"{contranovo_python} {run_denovo} "
                    f"--peak_path={mgf} --model={contranovo_ckpt} "
                    f"--config={contranovo_config} --out={out_csv} "
                    f"--use_peppr={use_peppr}"
                )
                print(command)
                ret = os.system(command)
                if ret != 0:
                    raise RuntimeError(
                        f"ContraNovo run failed (exit={ret}) on {mgf} (use_peppr={use_peppr})"
                    )
    elif model_type == 'auto':
        parser = argparse.ArgumentParser(prog="inference.py auto")
        parser.add_argument(
            "--dataset", nargs="+", default=["NINE_SPECIES_DATASET"],
            help="Names of DatasetPaths attributes to run inference on, resolved "
                 "from --datasets-module.",
        )
        parser.add_argument(
            "--datasets-module", default=DATASETS_MODULE,
            help="Module defining the named datasets (default: %(default)s; the "
                 "paper analyses use experiments.paths).",
        )
        parser.add_argument(
            "--use-peppr", choices=["true", "false", "both"], default="both",
            help="Run with prior fusion (true), without (false), or both (default).",
        )
        args = parser.parse_args(argv[1:])

        datasets = [resolve_dataset(name, args.datasets_module)
                    for name in args.dataset]

        if args.use_peppr == "both":
            use_peppr_modes = ["true", "false"]
        else:
            use_peppr_modes = [args.use_peppr]

        print(f"=== Auto inference for species: {const.ACTIVE_SPECIES} ===")
        print(f"Datasets: {[d.name for d in datasets]}  use_peppr={use_peppr_modes}")

        casanovo_config = resolve_casanovo_config()
        if casanovo_config != CASANOVO_CONFIG_DEFAULT:
            print(f"[lance-isolation] casanovo config: {casanovo_config}  (lance_dir={os.environ.get('PEPPR_LANCE_DIR')})")

        for dataset in datasets:
            shard_mgfs = sorted(glob.glob(dataset.final_mgf_glob))
            # Optional MGF sharding for parallel SLURM workers: PEPPR_MGF_SHARD="i/n"
            # (0-indexed) runs only files [i::n] (round-robin, balances file sizes).
            shard = os.environ.get("PEPPR_MGF_SHARD")
            if shard:
                shard_i, shard_n = (int(x) for x in shard.split("/"))
                shard_mgfs = shard_mgfs[shard_i::shard_n]
                print(f"[shard {shard_i}/{shard_n}] running {len(shard_mgfs)} of the matched MGFs")
            mgf_files = ' '.join(shard_mgfs)
            if not mgf_files:
                raise FileNotFoundError(f"No MGFs matched {dataset.final_mgf_glob!r}")
            for use_peppr in use_peppr_modes:
                mztab_path = dataset.mztab_path_fusion if use_peppr == "true" else dataset.mztab_path_dnps
                mztab_dir = os.path.dirname(mztab_path)
                os.makedirs(mztab_dir, exist_ok=True)
                slurm_id = os.environ.get('SLURM_JOB_ID', '')
                mztab_basename = os.path.basename(mztab_path).replace('.mztab', '')
                if slurm_id:
                    mztab_basename += f'_{slurm_id}'
                command = f"casanovo sequence -m https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt -c {casanovo_config} -d {mztab_dir} -o {mztab_basename} --teacher_forcing false --use_peppr {use_peppr} -e {mgf_files}"
                print(command)
                ret = os.system(command)
                if ret != 0:
                    raise RuntimeError(
                        f"casanovo failed (exit={ret}) on dataset={dataset.name}, use_peppr={use_peppr}"
                    )


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
