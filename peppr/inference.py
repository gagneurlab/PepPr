import argparse
import torch
import sys
import os
import re
import tempfile
from peppr import const
from tqdm import tqdm
from peppr.model import load_plm_model, load_fusion_model
import glob


CASANOVO_CONFIG_DEFAULT = const.CASANOVO_CONFIG_YAML


def resolve_casanovo_config():
    lance_dir = os.environ.get(
        "DNPS_LANCE_DIR", os.path.join(const.WORK_DIR, "lance")
    )
    os.makedirs(lance_dir, exist_ok=True)
    with open(CASANOVO_CONFIG_DEFAULT) as f:
        cfg = f.read()
    cfg = re.sub(r"(?m)^lance_dir:.*$", f"lance_dir: {lance_dir}", cfg)
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="casanovo_cfg_")
    with os.fdopen(fd, "w") as f:
        f.write(cfg)
    return tmp


torch.manual_seed(const.SEED)
torch.cuda.manual_seed(const.SEED)
torch.set_float32_matmul_precision('high')


model_type = sys.argv[1]

def predict_plm_teacher(model, X, batch_size):
    scores = torch.zeros(size=(X.shape[0], const.PLM_BLOCK_SIZE, len(const.VOCAB)), device=const.DEVICE)
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), total=len(X)//batch_size):
            end = min(i+batch_size, len(X))
            logits, *_ = model(X[i:end])
            scores[i:end] = logits
    return scores

if model_type == 'plm_teacher':
    plm_model = load_plm_model()
    batch_size = 4096

    X_plm_train = torch.load(const.PLM_PSM_X_TRAIN_PATH, map_location=const.DEVICE)
    scores_plm_train = predict_plm_teacher(plm_model, X_plm_train, batch_size)
    torch.save(scores_plm_train, const.PLM_PSM_TEACHER_SCORES_TRAIN_PATH)
    print("Saved PLM scores for train")

    X_plm_test = torch.load(const.PLM_PSM_X_TEST_PATH, map_location=const.DEVICE)
    scores_plm_test = predict_plm_teacher(plm_model, X_plm_test, batch_size)
    torch.save(scores_plm_test, const.PLM_PSM_TEACHER_SCORES_TEST_PATH)
    print("Saved PLM scores for test")

    # X_plm_plm = torch.load(const.PLM_SEQ_X_PATH, map_location=const.DEVICE)
    # scores_plm_plm = predict_plm_teacher(plm_model, X_plm_plm, batch_size)
    # torch.save(scores_plm_plm, const.PLM_SEQ_TEACHER_SCORES_PATH)
elif model_type == 'fusion_teacher':
    scores_plm_test = torch.load(const.PLM_PSM_TEACHER_SCORES_TEST_PATH)
    scores_casanovo_test = torch.load(const.CASANOVO_TEACHER_SCORES_TEST_PATH, map_location=const.DEVICE)
    print("Loaded PLM and casanovo scores")
    X_fusion = torch.cat([scores_casanovo_test, scores_plm_test], dim=2)

    vocab_size = scores_casanovo_test.shape[2]
    fusion_model = load_fusion_model(null_model=False, vocab_size=vocab_size)
    null_model = load_fusion_model(null_model=True, vocab_size=vocab_size)
    batch_size = 4096
    scores_fusion = torch.zeros(size=(X_fusion.shape[0], const.PLM_BLOCK_SIZE, vocab_size), device=const.DEVICE)
    scores_null = torch.zeros_like(scores_fusion)
    with torch.no_grad():
        for i in range(0, len(X_fusion), batch_size):
            end = min(i+batch_size, len(X_fusion))
            scores_fusion[i:end] = fusion_model(X_fusion[i:end].view(-1, X_fusion.size(2))).view(-1, const.PLM_BLOCK_SIZE, vocab_size)
            scores_null[i:end] = null_model(scores_casanovo_test[i:end].view(-1, scores_casanovo_test.size(2))).view(-1, const.PLM_BLOCK_SIZE, vocab_size)
            print(f"Processed {i} of {len(X_fusion)}")

    torch.save(scores_fusion, const.FUSION_SCORES_TEACHER_TEST_PATH)
    torch.save(scores_null, const.NULL_SCORES_TEACHER_TEST_PATH)
elif model_type == 'contranovo':
    print(f"=== ContraNovo +/- pepLM on species: {const.ACTIVE_SPECIES} ===")
    contranovo_root = os.path.join(const.PROJECT_ROOT, "ContraNovo")
    contranovo_python = const.CONTRANOVO_PYTHON
    if not contranovo_python:
        raise RuntimeError(
            "ContraNovo inference requires DNPS_CONTRANOVO_PYTHON to point "
            "to the ContraNovo environment's Python executable."
        )
    contranovo_ckpt = os.path.join(contranovo_root, "ContraNovo", "ContraNovo.ckpt")
    contranovo_config = os.path.join(contranovo_root, "ContraNovo", "config.yaml")
    run_denovo = os.path.join(contranovo_root, "run_denovo.py")
    out_root = os.path.join(const.RESULT_RUN_PATH, "contranovo")
    os.makedirs(out_root, exist_ok=True)

    mgf_files = sorted(glob.glob(const.NINE_SPECIES_DATASET.final_mgf_glob))
    if not mgf_files:
        raise FileNotFoundError(
            f"No MGFs matched {const.NINE_SPECIES_DATASET.final_mgf_glob!r}; "
            "run prepare_data first or set DNPS_SPECIES correctly."
        )

    # ContraNovo's model.py imports peppr for the PLM/fusion loaders, but
    # the khsam_contranovo env doesn't have peppr installed. Expose this
    # checkout on PYTHONPATH for the subprocess so the import resolves.
    peppr_root = const.PROJECT_ROOT
    existing_pp = os.environ.get("PYTHONPATH", "")
    sub_pythonpath = (
        f"{peppr_root}:{existing_pp}" if existing_pp else peppr_root
    )

    for use_plm in ("false", "true"):
        suffix = "fusion" if use_plm == "true" else "dnps"
        out_dir = os.path.join(out_root, f"{const.ACTIVE_SPECIES}_{suffix}")
        os.makedirs(out_dir, exist_ok=True)
        for mgf in mgf_files:
            base = os.path.splitext(os.path.basename(mgf))[0]
            out_csv = os.path.join(out_dir, f"{base}.csv")
            if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
                print(f"[skip] {use_plm=} {base}: already exists")
                continue
            command = (
                f"PYTHONPATH={sub_pythonpath} "
                f"{contranovo_python} {run_denovo} "
                f"--peak_path={mgf} --model={contranovo_ckpt} "
                f"--config={contranovo_config} --out={out_csv} "
                f"--use_plm={use_plm}"
            )
            print(command)
            ret = os.system(command)
            if ret != 0:
                raise RuntimeError(
                    f"ContraNovo run failed (exit={ret}) on {mgf} (use_plm={use_plm})"
                )
elif model_type == 'auto':
    parser = argparse.ArgumentParser(prog="inference.py auto")
    parser.add_argument(
        "--dataset", nargs="+", default=["NINE_SPECIES_DATASET"],
        help="Names of DatasetPaths attributes in const.py to run inference on.",
    )
    parser.add_argument(
        "--use-plm", choices=["true", "false", "both"], default="both",
        help="Run with PLM fusion (true), without (false), or both (default).",
    )
    args = parser.parse_args(sys.argv[2:])

    datasets = []
    for name in args.dataset:
        if not hasattr(const, name):
            raise ValueError(f"No dataset named {name!r} in const.py")
        datasets.append(getattr(const, name))

    if args.use_plm == "both":
        use_plm_modes = ["true", "false"]
    else:
        use_plm_modes = [args.use_plm]

    print(f"=== Auto inference for species: {const.ACTIVE_SPECIES} ===")
    print(f"Datasets: {[d.name for d in datasets]}  use_plm={use_plm_modes}")

    casanovo_config = resolve_casanovo_config()
    if casanovo_config != CASANOVO_CONFIG_DEFAULT:
        print(f"[lance-isolation] casanovo config: {casanovo_config}  (lance_dir={os.environ.get('DNPS_LANCE_DIR')})")

    for dataset in datasets:
        shard_mgfs = sorted(glob.glob(dataset.final_mgf_glob))
        # Optional MGF sharding for parallel SLURM workers: DNPS_MGF_SHARD="i/n"
        # (0-indexed) runs only files [i::n] (round-robin, balances file sizes).
        shard = os.environ.get("DNPS_MGF_SHARD")
        if shard:
            shard_i, shard_n = (int(x) for x in shard.split("/"))
            shard_mgfs = shard_mgfs[shard_i::shard_n]
            print(f"[shard {shard_i}/{shard_n}] running {len(shard_mgfs)} of the matched MGFs")
        mgf_files = ' '.join(shard_mgfs)
        if not mgf_files:
            raise FileNotFoundError(f"No MGFs matched {dataset.final_mgf_glob!r}")
        for use_plm in use_plm_modes:
            mztab_path = dataset.mztab_path_fusion if use_plm == "true" else dataset.mztab_path_dnps
            mztab_dir = os.path.dirname(mztab_path)
            os.makedirs(mztab_dir, exist_ok=True)
            slurm_id = os.environ.get('SLURM_JOB_ID', '')
            mztab_basename = os.path.basename(mztab_path).replace('.mztab', '')
            if slurm_id:
                mztab_basename += f'_{slurm_id}'
            command = f"casanovo sequence -m https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt -c {casanovo_config} -d {mztab_dir} -o {mztab_basename} --teacher_forcing false --use_plm {use_plm} -e {mgf_files}"
            print(command)
            ret = os.system(command)
            if ret != 0:
                raise RuntimeError(
                    f"casanovo failed (exit={ret}) on dataset={dataset.name}, use_plm={use_plm}"
                )
