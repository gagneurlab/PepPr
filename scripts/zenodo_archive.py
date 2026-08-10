#!/usr/bin/env python3
"""Shared implementation for building and validating the Zenodo archive.

The archive is selected by declarative ``IncludeSpec`` records.  The defaults
capture the final inputs and outputs used by this repository; additional specs
can be supplied on the command line without changing this file.
"""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

MANIFEST_FIELDS = (
    "category",
    "source",
    "destination",
    "size",
    "sha256",
    "provenance",
)

# These are checked both while selecting files and by the validator.  Includes
# never override them: an explicit opt-in should not accidentally publish RAW
# data, large derived tensors, work directories, or machine-local artifacts.
EXCLUDED_PATTERNS = (
    "*.raw",
    "*.mzml",
    "*.mzml.gz",
    "benchmarks/nine_species/*.mgf",
    "benchmarks/nine_species/**/*.mgf",
    "*/nine_species_proforma/*.mgf",
    "*/nine_species_proforma/**/*.mgf",
    "*unannotated*.mgf",
    "*_torch_data.pt",
    "*teacher_scores*.pt",
    "*plm_seq*.pt",
    "*plm_psm*.pt",
    "*fusion_y_*.pt",
    "*.lance",
    "*.lance/**",
    "*/lance/*",
    "*/fragpipe_workdir/*",
    "*/fragpipe_workdir/**",
    "*/lightning_logs/*",
    "*/lightning_logs/**",
    "*/__pycache__/*",
    "*/__pycache__/**",
    "*/logs/*",
    "*/logs/**",
    "*.slurm.out",
    "slurm-*.out",
    "*.cache",
    "*.pkl",
    "*.npz",
    "*generated_config*.yaml",
    "*generated_config*.yml",
)


@dataclass(frozen=True)
class IncludeSpec:
    """One source glob and its archive destination template."""

    name: str
    category: str
    root: str
    glob: str
    destination: str
    required: bool = False
    provenance: str = ""
    latest_valid_mztab: bool = False


@dataclass(frozen=True)
class Candidate:
    category: str
    source: Path
    source_label: str
    destination: Path
    provenance: str
    spec_name: str


@dataclass(frozen=True)
class ManifestRecord:
    category: str
    source: str
    destination: str
    size: int
    sha256: str
    provenance: str


ARCHIVE_README = """# DNPS-Hybrid Zenodo archive

This directory contains the processed inputs, final checkpoints, and final
outputs needed to reproduce the DNPS-Hybrid paper analyses.

Zenodo distributes this tree as a single gzip-compressed tarball
(`dnps_hybrid_zenodo.tar.gz`). After downloading, extract it and configure the
checkout with:

```
tar xzf dnps_hybrid_zenodo.tar.gz
export DNPS_DATA_PATH=/path/to/dnps_hybrid_zenodo
```

The top-level directories are:

- `fastas/`: maintained proteome and antibody training FASTAs
- `training/massivekb/`: final fusion train/validation MGFs
- `benchmarks/`: final mutation, mAb, and fixed KoL inputs
- `external/nine_species/`: Noble nine-species ProForma MGF benchmark
- `models/`: canonical pepLM, fusion, null, antibody, and baseline checkpoints
- `results/`: canonical predictions, summaries, logs, and baseline outputs
- `metadata/`: mAb references, regions, and final assembly summaries

## Nine-species benchmark

The Noble nine-species ProForma MGF tree (MassIVE `MSV000090982`) is included
under `external/nine_species/` using the original benchmark directory names
(for example `H.-sapiens`, `Mus-musculus`, and `Saccharomyces-cerevisiae`).
By default `DNPS_NINE_SPECIES_PATH` resolves to that directory.

The MGFs contain valid ProForma `SEQ=` annotations. Some third-party
distributions use leading numeric mass shifts instead; convert those files with
`dnps_hybrid.prepare_data.convert_mgf_to_proforma` before inference.

## Kingdoms of Life benchmark

Only the canonical fixed 10,000-spectrum subset for each maintained species is
included, at `benchmarks/kingdoms_of_life/subsets/<species>.mgf`. The full
per-species MGF trees are intentionally excluded.

`MANIFEST.tsv` and `MANIFEST.json` record every archived file, its original
logical source, archive destination, byte size, SHA-256 checksum, and
provenance. Validate the extracted payload with:

```
python scripts/validate_zenodo_archive.py "$DNPS_DATA_PATH"
```
"""


# benchmark_dir -> relative path under the project data root.
NINE_SPECIES_SOURCE_DIRS: tuple[tuple[str, str], ...] = (
    ("H.-sapiens", "casanovo/nine_species_proforma"),
    ("Mus-musculus", "mus_musculus/nine_species_proforma"),
    ("Saccharomyces-cerevisiae", "saccharomyces_cerevisiae/nine_species_proforma"),
    ("Bacillus-subtilis", "bacillus_subtilis/nine_species_proforma"),
    ("Apis-mellifera", "apis_mellifera/nine_species_proforma"),
    ("Solanum-lycopersicum", "solanum_lycopersicum/nine_species_proforma"),
    ("Vigna-mungo", "vigna_mungo/nine_species_proforma"),
    ("Methanosarcina-mazei", "methanosarcina_mazei/nine_species_proforma"),
    ("Candidatus-endoloripes", "candidatus_endoloripes/nine_species_proforma"),
)


def find_repo_root(start: Path) -> Path:
    """Return the nearest parent containing .git."""
    current = start.resolve()
    for path in (current, *current.parents):
        if (path / ".git").exists():
            return path
    raise RuntimeError(f"cannot auto-detect repository root from {start}")


def default_specs() -> tuple[IncludeSpec, ...]:
    """Canonical paper-bundle selections.

    Exact checkpoint names are intentional: variant/sweep checkpoints are not
    silently swept into the deposit.  Optional groups report absence but do not
    make a build fail.
    """
    specs = [
        IncludeSpec(
            "imgt_reference", "mab_metadata", "repo", "antibody/imgt/*",
            "metadata/mabs/references/imgt/{name}", True,
            "IMGT/GENE-DB amino-acid reference used to rebuild antibody FASTAs",
        ),
        IncludeSpec(
            "mab_reference_fastas", "mab_metadata", "repo", "xa_novo/**/*_ref.fasta",
            "metadata/mabs/references/{name}", False,
            "mature antibody references and CDR coordinates",
        ),
        IncludeSpec(
            "massivekb_train", "training", "data",
            "casanovo/fusion_train_set/*.mgf",
            "training/massivekb/fusion_train_set/{name}", True,
            "canonical annotated MassIVE-KB fusion training spectra",
        ),
        IncludeSpec(
            "massivekb_validation", "training", "data",
            "casanovo/fusion_val_set/*.mgf",
            "training/massivekb/fusion_val_set/{name}", True,
            "canonical annotated MassIVE-KB fusion validation spectra",
        ),
        IncludeSpec(
            "mutations_proforma", "benchmark", "data",
            "casanovo/mutations_proforma/*.mgf",
            "benchmarks/mutations/{name}", True,
            "final ProForma-annotated mutation benchmark spectra",
        ),
        IncludeSpec(
            "kol_fixed_subsets", "benchmark", "data",
            "kingdoms_of_life_eval_asymbnln/subsets/*.mgf",
            "benchmarks/kingdoms_of_life/subsets/{name}", True,
            "canonical final fixed 10,000-spectrum Kingdoms of Life subset",
        ),
    ]
    for fasta in (
        "UP000005640_9606.fasta",
        "mus_musculus.fasta",
        "saccharomyces_cerevisiae.fasta",
        "bacillus_subtilis.fasta",
        "apis_mellifera.fasta",
        "solanum_lycopersicum.fasta",
        "vigna_mungo.fasta",
        "methanosarcina_mazei.fasta",
        "candidatus_endoloripes.fasta",
        "human_iso.fasta",
        "antibody_human.fasta",
        "antibody_mouse.fasta",
    ):
        specs.append(
            IncludeSpec(
                f"fasta_{Path(fasta).stem}", "fasta", "data", fasta,
                "fastas/{name}", True,
                "reference FASTA used by a maintained peptide-prior model",
            )
        )

    peptide_prior_runs = (
        "mus_musculus",
        "saccharomyces_cerevisiae",
        "bacillus_subtilis",
        "apis_mellifera",
        "solanum_lycopersicum",
        "vigna_mungo",
        "methanosarcina_mazei",
        "candidatus_endoloripes",
        "human_iso",
        "antibody_human",
        "antibody_mouse",
    )
    for run_name in peptide_prior_runs:
        specs.append(
            IncludeSpec(
                f"model_{run_name}_plm", "model", "data",
                f"{run_name}/plm_ckpt.pt", "models/{run}/plm_ckpt.pt",
                True, "canonical peptide language-model checkpoint",
            )
        )
    for run_name, filename, provenance in (
        ("casanovo", "fusion_model.pth", "canonical fusion checkpoint"),
        ("casanovo", "null_model.pth", "canonical spectrum-only control"),
        ("casanovo", "fusion_model_asymbnln.pth", "canonical asymmetric fusion checkpoint"),
        ("casanovo", "null_model_asymbnln.pth", "canonical asymmetric null checkpoint"),
        ("human_iso_asymbnln", "fusion_model.pth", "human-isoform asymmetric fusion checkpoint"),
        ("human_iso_asymbnln", "null_model.pth", "human-isoform asymmetric null checkpoint"),
    ):
        specs.append(
            IncludeSpec(
                f"model_{run_name}_{filename}", "model", "data",
                f"{run_name}/{filename}", "models/{run}/{name}",
                True, provenance,
            )
        )
    specs.extend(
        (
            IncludeSpec(
                "smsnet_outputs", "result", "smsnet", "*_inputs_output/*",
                "results/baselines/smsnet/{relative}", True,
                "final SMSNet baseline predictions and rescoring outputs",
            ),
            IncludeSpec(
                "contranovo_fusion_model", "model", "data",
                "casanovo/contranovo_fusion_model.pth",
                "models/casanovo/{name}", False,
                "canonical ContraNovo fusion checkpoint",
            ),
            IncludeSpec(
                "contranovo_null_model", "model", "data",
                "casanovo/contranovo_null_model.pth",
                "models/casanovo/{name}", False,
                "canonical ContraNovo null checkpoint",
            ),
            IncludeSpec(
                "xanovo_model", "model", "repo", "xa_novo/epoch=9-step=550000.ckpt",
                "models/xa_novo/{name}", False,
                "XA-Novo v3 checkpoint used by the supplementary benchmark",
            ),
        )
    )

    # Canonical nine-species and mutation outputs. Sharded jobs are renamed to
    # stable part numbers rather than preserving scheduler job IDs.
    species_runs = {
        "human": "casanovo",
        "mouse": "mus_musculus",
        "yeast": "saccharomyces_cerevisiae",
        "bacillus": "bacillus_subtilis",
        "honeybee": "apis_mellifera",
        "tomato": "solanum_lycopersicum",
        "cowpea": "vigna_mungo",
        "archaeon": "methanosarcina_mazei",
        "endoloripes": "candidatus_endoloripes",
    }
    for species, run_name in species_runs.items():
        specs.append(
            IncludeSpec(
                f"nine_species_{species}_dnps", "result", "data",
                f"{run_name}/results/9s_{species}_dnps.mztab",
                f"results/{run_name}/9s_{species}_dnps.mztab", True,
                "canonical Casanovo nine-species prediction",
            )
        )
        if species == "human":
            specs.append(
                IncludeSpec(
                    "nine_species_human_same", "result", "data",
                    "casanovo/results/9s_human_plmhuman_iso_hybrid*.mztab",
                    "results/casanovo/9s_human_plmhuman_iso_hybrid.mztab",
                    True, "canonical human-isoform-prior prediction",
                )
            )
        elif species == "yeast":
            specs.append(
                IncludeSpec(
                    "nine_species_yeast_same", "result", "data",
                    f"{run_name}/results/9s_yeast_hybrid_*.mztab",
                    f"results/{run_name}/9s_yeast_hybrid_part-{{index}}.mztab",
                    True, "canonical sharded yeast same-species prediction",
                )
            )
        else:
            specs.append(
                IncludeSpec(
                    f"nine_species_{species}_same", "result", "data",
                    f"{run_name}/results/9s_{species}_asymbnln.mztab",
                    f"results/{run_name}/9s_{species}_asymbnln.mztab",
                    True, "canonical same-species-prior prediction",
                )
            )
        if species != "human":
            specs.append(
                IncludeSpec(
                    f"nine_species_{species}_cross", "result", "data",
                    f"{run_name}/results/9s_{species}_plmhuman_iso_hybrid*.mztab",
                    f"results/{run_name}/9s_{species}_plmhuman_iso_hybrid"
                    f"_part-{{index}}.mztab",
                    True, "canonical human-isoform cross-species prediction",
                )
            )
    for filename in (
        "mutations_dnps.mztab",
        "mutations_hybrid_human_iso.mztab",
    ):
        specs.append(
            IncludeSpec(
                f"mutation_{Path(filename).stem}", "result", "data",
                f"casanovo/results/{filename}", f"results/casanovo/{filename}",
                True, "canonical mutation benchmark prediction",
            )
        )
    # Final baseline outputs consumed by plot_benchmark.py.
    for pattern in (
        "mus_musculus/results/*_pw_score.csv",
        "mus_musculus/results/contranovo/**/*",
        "mus_musculus/results/instanovo/**/*",
        "casanovo/results/contranovo/**/*",
    ):
        specs.append(
            IncludeSpec(
                f"baseline_{len(specs)}", "result", "data", pattern,
                "results/{run}/{payload_tail}/{name}", False,
                "final external-baseline prediction",
            )
        )
    # KoL final outputs; avoid scheduler and training logs.
    for extension in ("mztab", "log", "csv"):
        specs.extend(
            (
                IncludeSpec(
                    f"kol_human_iso_results_{extension}", "kol_result", "data",
                    f"kingdoms_of_life_eval_asymbnln/**/*.{extension}",
                    "results/kingdoms_of_life/{tail}/{name}", False,
                    "final human-isoform-prior Kingdoms of Life results",
                ),
            )
        )

    # mAb inputs, outputs, external baselines, references, and final assemblies.
    xanovo_mabs = {
        "2B4": "germline_mouse_sw",
        "36H6": "germline_mouse_sw",
        "85F7": "germline_mouse_sw",
        "S2P6": "germline_human_sw",
    }
    for mab in xanovo_mabs:
        specs.append(
            IncludeSpec(
                f"mab_mgf_{mab}", "mab_benchmark", "repo",
                f"xa_novo/PXD060500_{mab}/mgf/*.mgf",
                "benchmarks/mabs/{relative}", False,
                "final annotated monoclonal-antibody benchmark spectra",
            )
        )
    specs.append(
        IncludeSpec(
            "mab_nontryp_mgfs", "mab_benchmark", "data",
            "nontryp/*/*/*_annotated.mgf",
            "benchmarks/mabs/{relative}", False,
            "final annotated non-tryptic mAb benchmark spectra",
        )
    )
    xanovo_proteases = ("aspn", "chymo", "elastase", "pepsin", "trypsin")
    for mab, pp_arm in xanovo_mabs.items():
        for protease in xanovo_proteases:
            for arm in ("vanilla", pp_arm):
                basename = f"casanovo_{mab}_{protease}_{arm}_{mab}_canonical.mztab"
                specs.append(
                    IncludeSpec(
                        f"mab_result_{mab}_{protease}_{arm}", "mab_result", "repo",
                        f"xa_novo/PXD060500_{mab}/casanovo_results/"
                        f"casanovo_{mab}_{protease}_{arm}_{mab}_*.mztab",
                        f"results/mabs/xa_novo/PXD060500_{mab}/"
                        f"casanovo_results/{basename}",
                        False, "latest complete canonical mAb prediction",
                        latest_valid_mztab=True,
                    )
                )
    nontryp_mabs = {
        "IgG1_Human_H": (("aspn", "chymo", "gluc", "lysc", "proteinasek"), "antibody_human"),
        "IgG1_Human_L": (("aspn", "chymo", "gluc", "lysc", "proteinasek"), "antibody_human"),
        "Herceptin": (("elastase", "gluc", "lysc", "lysn", "thermolysin"), "antibody_human"),
        "anti-FLAG-M2": (("aspn", "chymo", "elastase", "gluc", "lysc", "lysn", "thermolysin"), "antibody_mouse"),
        "WIgG1_H": (("aspn", "chymo"), "antibody_mouse"),
        "WIgG1_L": (("aspn", "chymo"), "antibody_mouse"),
    }
    for mab, (proteases, pp_arm) in nontryp_mabs.items():
        for protease in proteases:
            for arm in ("noplm", pp_arm):
                basename = f"casanovo_{mab}_{protease}_{arm}_{mab}_canonical.mztab"
                source_basename = (
                    f"casanovo_{mab}_{protease}_{arm}*.mztab"
                    if arm == "noplm"
                    else f"casanovo_{mab}_{protease}_{arm}_{mab}_*.mztab"
                )
                specs.append(
                    IncludeSpec(
                        f"mab_result_{mab}_{protease}_{arm}", "mab_result", "data",
                        f"nontryp/{mab}/{protease}/casanovo_results/"
                        f"{source_basename}",
                        f"results/mabs/nontryp/{mab}/{protease}/"
                        f"casanovo_results/{basename}",
                        False, "latest complete canonical mAb prediction",
                        latest_valid_mztab=True,
                    )
                )
    for pattern in (
        "figshare_21394143/02-Results/**/summary.csv",
        "figshare_21394143/01-RawData/PXD023419_anti-FLAG-M2_antibody/"
        "RAW/*SupernovoPeptideOverview.csv",
    ):
        specs.append(
            IncludeSpec(
                f"mab_external_{len(specs)}", "mab_result", "data", pattern,
                "results/mabs/external/{relative}", False,
                "external mAb baseline summary used in supplementary analyses",
            )
        )
    specs.extend(
        (
            IncludeSpec(
                "mab_assemblies", "mab_metadata", "repo",
                "xa_novo/alps_3arm_all/3arm_summary_*.tsv",
                "metadata/mabs/assembly/{name}", False,
                "final antibody assembly summary",
            ),
            IncludeSpec(
                "human_igg_target_decoy", "mab_metadata", "data",
                "beslic_mab/IgG1_Human_H/Human_HL_td.fasta",
                "metadata/mabs/references/{name}", False,
                "target-decoy FASTA used for human IgG database search",
            ),
            IncludeSpec(
                "mouse_igg_target_decoy", "mab_metadata", "data",
                "beslic_mab/WIgG1/WIgG1_HL_td.fasta",
                "metadata/mabs/references/{name}", False,
                "target-decoy FASTA used for mouse IgG database search",
            ),
            IncludeSpec(
                "mab_regions", "mab_metadata", "repo", "mab_db_regions.csv",
                "metadata/mabs/assembly/{name}", False,
                "mAb database-region annotations",
            ),
        )
    )
    return tuple(specs)


def path_is_excluded(path: Path | str) -> str | None:
    normalized = str(path).replace(os.sep, "/").lower().lstrip("./")
    basename = normalized.rsplit("/", 1)[-1]
    for pattern in EXCLUDED_PATTERNS:
        p = pattern.lower()
        if fnmatch.fnmatch(normalized, p) or fnmatch.fnmatch(basename, p):
            return pattern
    return None


def resolve_roots(
    data_root: Path,
    repo_root: Path,
    kol_root: Path,
    mutations_root: Path,
    smsnet_root: Path | None = None,
) -> dict[str, Path]:
    roots = {
        "data": data_root.expanduser().resolve(),
        "repo": repo_root.expanduser().resolve(),
        "kol": kol_root.expanduser().resolve(),
        "mutations": mutations_root.expanduser().resolve(),
    }
    if smsnet_root is not None:
        roots["smsnet"] = smsnet_root.expanduser().resolve()
    return roots


def parse_include(value: str, index: int) -> IncludeSpec:
    """Parse ROOT:GLOB:DEST:CATEGORY[:required|optional].

    JSON objects are also accepted, which is useful when a path contains a
    colon or when setting name/provenance explicitly.
    """
    if value.lstrip().startswith("{"):
        obj = json.loads(value)
        obj.setdefault("name", f"cli_include_{index}")
        obj.setdefault("required", False)
        obj.setdefault("provenance", "explicit command-line include")
        return IncludeSpec(**obj)
    parts = value.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(
            "--include must be ROOT:GLOB:DEST:CATEGORY[:required|optional] "
            "or a JSON IncludeSpec object"
        )
    root, pattern, destination, category = parts[:4]
    required = len(parts) == 5 and parts[4].lower() == "required"
    if len(parts) == 5 and parts[4].lower() not in {"required", "optional"}:
        raise ValueError("fifth --include field must be required or optional")
    return IncludeSpec(
        f"cli_include_{index}", category, root, pattern, destination, required,
        "explicit command-line include",
    )


def load_spec_file(path: Path) -> list[IncludeSpec]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("includes")
    if not isinstance(payload, list):
        raise ValueError("spec file must contain a JSON list or {'includes': [...]}")
    return [IncludeSpec(**item) for item in payload]


def _safe_relative(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive destination: {path}")
    return path


def _render_destination(template: str, relative: Path, index: int = 0) -> Path:
    parts = relative.parts
    run = parts[0] if parts else ""
    parent = relative.parent.name if relative.parent != Path(".") else ""
    # tail removes both the run/root directory and the filename.  It is useful
    # for results/<run_name>/... layouts without duplicating either.
    tail = Path(*parts[1:-1]).as_posix() if len(parts) > 2 else ""
    payload_tail = (
        Path(*parts[2:-1]).as_posix()
        if len(parts) > 3 and parts[1] == "results"
        else ""
    )
    rendered = template.format(
        relative=relative.as_posix(),
        name=relative.name,
        stem=relative.stem,
        run=run,
        parent=parent,
        tail=tail,
        payload_tail=payload_tail,
        index=f"{index:03d}",
    )
    return _safe_relative(Path(rendered))


def iter_spec_matches(spec: IncludeSpec, roots: dict[str, Path]) -> Iterator[Candidate]:
    root = roots.get(spec.root, Path(spec.root).expanduser().resolve())
    source_root_label = spec.root if spec.root in roots else "external"
    if not root.exists():
        return
    sources = sorted(path for path in root.glob(spec.glob) if not path.is_dir())
    if spec.latest_valid_mztab:
        sources = [path for path in sources if _mztab_has_psms(path, minimum=101)]
        sources = [max(sources, key=lambda path: (path.stat().st_mtime_ns, str(path)))] if sources else []
    for index, source in enumerate(sources):
        if source.is_dir():
            continue
        relative = source.relative_to(root)
        if path_is_excluded(relative):
            continue
        yield Candidate(
            spec.category,
            source,
            f"{source_root_label}:{relative.as_posix()}",
            _render_destination(spec.destination, relative, index),
            spec.provenance or spec.name,
            spec.name,
        )


def _mztab_has_psms(path: Path, minimum: int) -> bool:
    count = 0
    with path.open("rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("PSM\t"):
                count += 1
                if count >= minimum:
                    return True
    return False


def collect_candidates(
    specs: Sequence[IncludeSpec], roots: dict[str, Path]
) -> tuple[list[Candidate], list[IncludeSpec], list[IncludeSpec]]:
    candidates: list[Candidate] = []
    missing_required: list[IncludeSpec] = []
    missing_optional: list[IncludeSpec] = []
    destinations: dict[Path, Path] = {}
    for spec in specs:
        matches = list(iter_spec_matches(spec, roots))
        if not matches:
            (missing_required if spec.required else missing_optional).append(spec)
        for candidate in matches:
            prior = destinations.get(candidate.destination)
            if prior is not None:
                try:
                    same = prior.resolve() == candidate.source.resolve()
                except OSError:
                    same = False
                if not same:
                    raise ValueError(
                        f"destination collision: {candidate.destination}: "
                        f"{prior} and {candidate.source}"
                    )
                continue
            destinations[candidate.destination] = candidate.source
            candidates.append(candidate)
    candidates.sort(key=lambda item: item.destination.as_posix())
    return candidates, missing_required, missing_optional


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_for_copy(path: Path, dereference: bool) -> Path:
    if path.is_symlink():
        if not path.exists():
            raise FileNotFoundError(f"broken source symlink: {path}")
        if dereference:
            resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f"source symlink does not resolve to a file: {path}")
            return resolved
    return path


def _copy_atomic(source: Path, destination: Path, dereference: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if source.is_symlink() and not dereference:
            tmp.unlink()
            os.symlink(os.readlink(source), tmp)
        else:
            shutil.copy2(source, tmp, follow_symlinks=True)
        os.replace(tmp, destination)
    finally:
        if tmp.lexists() if hasattr(tmp, "lexists") else os.path.lexists(tmp):
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def stage_nine_species_mgfs(
    data_root: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    dereference: bool = True,
) -> list[Path]:
    """Copy Noble nine-species ProForma MGFs into ``external/nine_species/``."""
    written: list[Path] = []
    nine_species_root = output_dir / "external" / "nine_species"
    for benchmark_dir, relative_source in NINE_SPECIES_SOURCE_DIRS:
        source_dir = (data_root / relative_source).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"missing nine-species source directory: {source_dir}"
            )
        dest_dir = nine_species_root / benchmark_dir
        mgfs = sorted(source_dir.glob("*.mgf"))
        if not mgfs:
            raise FileNotFoundError(f"no MGF files under {source_dir}")
        for mgf in mgfs:
            destination = dest_dir / mgf.name
            source = _source_for_copy(mgf, dereference)
            if destination.exists() and not overwrite:
                if (
                    destination.stat().st_size == source.stat().st_size
                    and sha256_file(destination) == sha256_file(source)
                ):
                    written.append(destination)
                    continue
            _copy_atomic(source, destination, dereference=False)
            written.append(destination)
    return written


def nine_species_staged_include_specs(staging_root: Path) -> tuple[IncludeSpec, ...]:
    """IncludeSpec entries for staged nine-species MGF directories."""
    specs: list[IncludeSpec] = []
    for benchmark_dir, _ in NINE_SPECIES_SOURCE_DIRS:
        species_dir = staging_root / "external" / "nine_species" / benchmark_dir
        specs.append(
            IncludeSpec(
                f"nine_species_{benchmark_dir.replace('.', '_').replace('-', '_')}",
                "benchmark",
                str(species_dir.resolve()),
                "*.mgf",
                f"external/nine_species/{benchmark_dir}/{{name}}",
                True,
                "Noble nine-species ProForma MGF",
            )
        )
    return tuple(specs)


def stage_candidates(
    candidates: Sequence[Candidate],
    output: Path,
    *,
    dry_run: bool,
    overwrite: bool,
    dereference: bool,
    progress: bool = False,
) -> tuple[list[ManifestRecord], dict[str, int]]:
    records: list[ManifestRecord] = []
    counts = {"copied": 0, "resumed": 0, "planned": 0}
    total = len(candidates)
    for position, candidate in enumerate(candidates, 1):
        if progress:
            print(
                f"STAGING {position}/{total} {candidate.destination.as_posix()}",
                flush=True,
            )
        source_for_data = _source_for_copy(candidate.source, dereference)
        size = source_for_data.stat().st_size
        checksum = sha256_file(source_for_data)
        destination = output / candidate.destination
        action = "planned"
        if not dry_run:
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() and not destination.exists():
                    if not overwrite:
                        raise FileExistsError(
                            f"broken destination symlink exists (use --overwrite): {destination}"
                        )
                elif destination.is_file():
                    existing_size = destination.stat().st_size
                    if existing_size == size and sha256_file(destination) == checksum:
                        action = "resumed"
                    elif not overwrite:
                        raise FileExistsError(
                            f"destination differs (use --overwrite): {destination}"
                        )
                elif not overwrite:
                    raise FileExistsError(
                        f"non-file destination exists (use --overwrite): {destination}"
                    )
            if action != "resumed":
                _copy_atomic(candidate.source, destination, dereference)
                action = "copied"
        counts[action] += 1
        records.append(
            ManifestRecord(
                candidate.category,
                candidate.source_label,
                candidate.destination.as_posix(),
                size,
                checksum,
                candidate.provenance,
            )
        )
        if progress:
            print(
                f"STAGED {position}/{total} {action} "
                f"{candidate.destination.as_posix()}",
                flush=True,
            )
    return records, counts


def stage_archive_readme(
    output: Path, *, dry_run: bool, overwrite: bool
) -> tuple[ManifestRecord, str]:
    """Stage the generated archive README with normal resume semantics."""
    data = ARCHIVE_README.encode("utf-8")
    checksum = hashlib.sha256(data).hexdigest()
    destination = output / "ARCHIVE_README.md"
    action = "planned"
    if not dry_run:
        if destination.exists():
            if destination.is_file() and destination.read_bytes() == data:
                action = "resumed"
            elif not overwrite:
                raise FileExistsError(
                    f"destination differs (use --overwrite): {destination}"
                )
        if action != "resumed":
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=".ARCHIVE_README.md.", suffix=".partial", dir=output
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(tmp_name, destination)
            finally:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
            action = "copied"
    return (
        ManifestRecord(
            "metadata",
            "generated:scripts/build_zenodo_archive.py",
            "ARCHIVE_README.md",
            len(data),
            checksum,
            "documents Zenodo release layout, nine-species MGFs, and KoL subsets",
        ),
        action,
    )


def write_manifests(output: Path, records: Sequence[ManifestRecord]) -> None:
    """Atomically write deterministic TSV and JSON manifests."""
    output.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "files": [asdict(record) for record in records],
    }
    for name, writer in (
        (
            "MANIFEST.json",
            lambda handle: json.dump(json_payload, handle, indent=2, sort_keys=True),
        ),
        (
            "MANIFEST.tsv",
            lambda handle: _write_tsv(handle, records),
        ),
    ):
        fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", dir=output, text=True)
        try:
            with os.fdopen(fd, "w", newline="") as handle:
                writer(handle)
                handle.write("\n")
            os.replace(tmp_name, output / name)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def prune_unselected(output: Path, destinations: Iterable[Path]) -> int:
    """Remove files not selected by the current manifest plan."""
    if not output.exists():
        return 0
    keep = {Path(destination) for destination in destinations}
    keep.update({
        Path("ARCHIVE_README.md"),
        Path("MANIFEST.tsv"),
        Path("MANIFEST.json"),
    })
    removed = 0
    paths = sorted(
        archive_files(output),
        key=lambda path: len(path.relative_to(output).parts),
        reverse=True,
    )
    for path in paths:
        relative = path.relative_to(output)
        if path.is_file() or path.is_symlink():
            if relative not in keep:
                path.unlink()
                removed += 1
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return removed


def _write_tsv(handle, records: Sequence[ManifestRecord]) -> None:
    writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t")
    writer.writeheader()
    for record in records:
        writer.writerow(asdict(record))


def load_manifest(archive: Path) -> list[ManifestRecord]:
    path = archive / "MANIFEST.json"
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("hash_algorithm") != "sha256":
        raise ValueError("unsupported manifest schema or hash algorithm")
    records = payload.get("files")
    if not isinstance(records, list):
        raise ValueError("MANIFEST.json has no files list")
    return [ManifestRecord(**record) for record in records]


def archive_files(archive: Path) -> Iterator[Path]:
    for root, dirs, files in os.walk(archive, followlinks=False):
        root_path = Path(root)
        for name in dirs + files:
            yield root_path / name


def validate_manifest(
    archive: Path, records: Sequence[ManifestRecord], checksums: bool
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for record in records:
        if Path(record.source).is_absolute():
            errors.append(f"manifest source exposes an absolute path: {record.source}")
        try:
            relative = _safe_relative(Path(record.destination))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if record.destination in seen:
            errors.append(f"duplicate manifest destination: {record.destination}")
        seen.add(record.destination)
        path = archive / relative
        if not path.exists():
            errors.append(f"missing manifested file: {record.destination}")
            continue
        if not path.is_file():
            errors.append(f"manifest entry is not a file: {record.destination}")
            continue
        if path.stat().st_size != record.size:
            errors.append(f"size mismatch: {record.destination}")
        if checksums and sha256_file(path) != record.sha256:
            errors.append(f"checksum mismatch: {record.destination}")
    disk_files = {
        path.relative_to(archive).as_posix()
        for path in archive_files(archive)
        if path.is_file() and path.name not in {"MANIFEST.json", "MANIFEST.tsv"}
    }
    for extra in sorted(disk_files - seen):
        errors.append(f"unmanifested file: {extra}")
    return errors


def validate_archive_paths(archive: Path) -> list[str]:
    errors: list[str] = []
    for path in archive_files(archive):
        relative = path.relative_to(archive)
        if path.is_symlink() and not path.exists():
            errors.append(f"broken symlink: {relative}")
        excluded = path_is_excluded(relative)
        if excluded:
            errors.append(f"excluded path ({excluded}): {relative}")
    return errors


def validate_external_benchmark_scope(archive: Path) -> list[str]:
    """Enforce loose nine-species MGFs and fixed-subset KoL policy."""
    errors: list[str] = []
    legacy_nine_species = archive / "benchmarks" / "nine_species"
    if legacy_nine_species.exists():
        errors.append(
            "benchmarks/nine_species must not be archived; use "
            "external/nine_species/<Species>/*.mgf"
        )

    nine_species_root = archive / "external" / "nine_species"
    if not nine_species_root.is_dir():
        errors.append("missing external/nine_species/")
    else:
        legacy_bundles = sorted(path.name for path in nine_species_root.glob("*.tar.gz"))
        if legacy_bundles:
            errors.append(
                "external/nine_species must contain loose MGF directories, "
                f"not bundle archives: {', '.join(legacy_bundles)}"
            )
        for benchmark_dir, _ in NINE_SPECIES_SOURCE_DIRS:
            species_dir = nine_species_root / benchmark_dir
            if not any(species_dir.glob("*.mgf")):
                errors.append(
                    f"missing nine-species MGFs under external/nine_species/{benchmark_dir}/"
                )

    readme = archive / "ARCHIVE_README.md"
    if not readme.is_file():
        errors.append("missing ARCHIVE_README.md")
    else:
        text = readme.read_text(errors="replace")
        for required_text in (
            "external/nine_species/",
            "dnps_hybrid_zenodo.tar.gz",
            "DNPS_NINE_SPECIES_PATH",
            "benchmarks/kingdoms_of_life/subsets/<species>.mgf",
        ):
            if required_text not in text:
                errors.append(
                    f"ARCHIVE_README.md does not document {required_text!r}"
                )

    kol_root = archive / "benchmarks" / "kingdoms_of_life" / "subsets"
    if not kol_root.is_dir():
        errors.append("missing benchmarks/kingdoms_of_life/subsets")
        return errors
    species_seen: set[str] = set()
    for mgf in sorted(kol_root.rglob("*.mgf")):
        relative = mgf.relative_to(kol_root)
        if len(relative.parts) != 1:
            errors.append(f"non-canonical KoL subset path: {mgf.relative_to(archive)}")
            continue
        species = relative.stem
        if relative.name != f"{species}.mgf":
            errors.append(
                f"KoL subset filename must match species: {mgf.relative_to(archive)}"
            )
        if species in species_seen:
            errors.append(f"multiple KoL subset MGFs for species: {species}")
        species_seen.add(species)
        _, spectra, _ = _mgf_sequences(mgf, 0)
        if spectra != 10_000:
            errors.append(
                f"KoL subset must contain exactly 10000 spectra: "
                f"{mgf.relative_to(archive)} has {spectra}"
            )
    if not species_seen:
        errors.append("no canonical Kingdoms of Life subset MGFs found")
    return errors


def _mgf_sequences(path: Path, limit: int) -> tuple[list[str], int, int]:
    sequences: list[str] = []
    spectra = 0
    missing = 0
    in_block = False
    seq: str | None = None
    with path.open("rt", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                in_block = True
                seq = None
            elif in_block and line.startswith("SEQ="):
                seq = line[4:].strip()
            elif line == "END IONS" and in_block:
                spectra += 1
                if not seq:
                    missing += 1
                elif len(sequences) < limit:
                    sequences.append(seq)
                in_block = False
    return sequences, spectra, missing


def validate_mgfs(archive: Path, sample_size: int) -> tuple[list[str], str]:
    errors: list[str] = []
    parser = None
    parser_status = "pyteomics unavailable; ProForma syntax parsing skipped"
    try:
        from pyteomics import proforma  # type: ignore

        parser = proforma.parse
        parser_status = "pyteomics ProForma parser enabled"
    except (ImportError, AttributeError):
        pass
    for path in sorted(archive.rglob("*.mgf")):
        sequences, spectra, missing = _mgf_sequences(path, sample_size)
        relative = path.relative_to(archive)
        if spectra == 0:
            errors.append(f"MGF contains no complete spectra: {relative}")
        partially_annotated_mab = relative.parts[:2] == ("benchmarks", "mabs")
        if missing and not partially_annotated_mab:
            errors.append(f"MGF has {missing}/{spectra} spectra without SEQ: {relative}")
        if parser:
            for sequence in sequences:
                try:
                    parser(sequence)
                except Exception as exc:
                    errors.append(
                        f"invalid sampled ProForma sequence in {relative}: "
                        f"{sequence!r} ({exc})"
                    )
    return errors, parser_status


def validate_mztabs(archive: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted((*archive.rglob("*.mztab"), *archive.rglob("*.mzTab"))):
        has_mtd = has_psh = has_psm = False
        with path.open("rt", errors="replace") as handle:
            for line in handle:
                prefix = line[:3]
                has_mtd |= prefix == "MTD"
                has_psh |= prefix == "PSH"
                has_psm |= prefix == "PSM"
        missing = [
            name
            for name, present in (("MTD", has_mtd), ("PSH", has_psh), ("PSM", has_psm))
            if not present
        ]
        if missing:
            errors.append(
                f"incomplete mzTab ({','.join(missing)} missing): "
                f"{path.relative_to(archive)}"
            )
    return errors
