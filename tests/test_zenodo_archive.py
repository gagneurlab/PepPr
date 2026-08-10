"""Focused tests for the manifest-driven Zenodo archive tools."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from zenodo_archive import (  # noqa: E402
    IncludeSpec,
    NINE_SPECIES_SOURCE_DIRS,
    collect_candidates,
    default_specs,
    load_manifest,
    prune_unselected,
    resolve_roots,
    stage_archive_readme,
    stage_candidates,
    stage_nine_species_mgfs,
    validate_archive_paths,
    validate_external_benchmark_scope,
    validate_manifest,
    validate_mgfs,
    validate_mztabs,
    write_manifests,
)


VALID_MGF = """BEGIN IONS
TITLE=test
PEPMASS=500
CHARGE=2+
SEQ=PEPTIDE
100 200
END IONS
"""

VALID_MZTAB = """MTD\tmzTab-version\t1.0.0
PSH\tsequence\tPSM_ID
PSM\tPEPTIDE\t1
"""


class ZenodoArchiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source = self.root / "source"
        self.archive = self.root / "archive"
        self.source.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _roots(self):
        return resolve_roots(self.source, self.source, self.source, self.source)

    def test_selection_excludes_large_intermediates(self) -> None:
        (self.source / "final.mgf").write_text(VALID_MGF)
        (self.source / "sample.raw").write_bytes(b"raw")
        (self.source / "plm_seq_x.pt").write_bytes(b"tensor")
        spec = IncludeSpec(
            "all", "test", "data", "*", "benchmarks/{name}", required=True
        )
        candidates, missing_required, _ = collect_candidates([spec], self._roots())
        self.assertFalse(missing_required)
        self.assertEqual([item.source.name for item in candidates], ["final.mgf"])
        defaults = default_specs()
        self.assertFalse(any("nine_species" in spec.glob for spec in defaults))
        kol = next(spec for spec in defaults if spec.name == "kol_fixed_subsets")
        self.assertEqual(
            kol.glob, "kingdoms_of_life_eval_asymbnln/subsets/*.mgf"
        )

    def test_stage_resume_and_validate(self) -> None:
        (self.source / "final.mgf").write_text(VALID_MGF)
        (self.source / "final.mztab").write_text(VALID_MZTAB)
        specs = [
            IncludeSpec("mgf", "benchmark", "data", "*.mgf", "benchmarks/{name}", True),
            IncludeSpec("mztab", "result", "data", "*.mztab", "results/{name}", True),
        ]
        candidates, missing_required, _ = collect_candidates(specs, self._roots())
        self.assertFalse(missing_required)
        records, counts = stage_candidates(
            candidates,
            self.archive,
            dry_run=False,
            overwrite=False,
            dereference=True,
        )
        self.assertEqual(counts["copied"], 2)
        write_manifests(self.archive, records)

        resumed, counts = stage_candidates(
            candidates,
            self.archive,
            dry_run=False,
            overwrite=False,
            dereference=True,
        )
        self.assertEqual(records, resumed)
        self.assertEqual(counts["resumed"], 2)

        loaded = load_manifest(self.archive)
        self.assertEqual(validate_manifest(self.archive, loaded, checksums=True), [])
        self.assertEqual(validate_archive_paths(self.archive), [])
        mgf_errors, _ = validate_mgfs(self.archive, sample_size=10)
        self.assertEqual(mgf_errors, [])
        self.assertEqual(validate_mztabs(self.archive), [])

    def test_differing_destination_requires_overwrite(self) -> None:
        (self.source / "file.txt").write_text("source")
        spec = IncludeSpec("text", "test", "data", "*.txt", "results/{name}", True)
        candidates, _, _ = collect_candidates([spec], self._roots())
        destination = self.archive / "results/file.txt"
        destination.parent.mkdir(parents=True)
        destination.write_text("different")
        with self.assertRaises(FileExistsError):
            stage_candidates(
                candidates,
                self.archive,
                dry_run=False,
                overwrite=False,
                dereference=True,
            )
        records, _ = stage_candidates(
            candidates,
            self.archive,
            dry_run=False,
            overwrite=True,
            dereference=True,
        )
        self.assertEqual(destination.read_text(), "source")
        self.assertEqual(len(records), 1)

    def test_prune_unselected(self) -> None:
        keep = self.archive / "results/keep.txt"
        stale = self.archive / "models/stale.pt"
        keep.parent.mkdir(parents=True)
        stale.parent.mkdir(parents=True)
        keep.write_text("keep")
        stale.write_text("stale")
        removed = prune_unselected(self.archive, [Path("results/keep.txt")])
        self.assertEqual(removed, 1)
        self.assertTrue(keep.is_file())
        self.assertFalse(stale.exists())

    def test_stage_nine_species_mgfs(self) -> None:
        for _benchmark_dir, relative in NINE_SPECIES_SOURCE_DIRS:
            src = self.source / relative
            src.mkdir(parents=True, exist_ok=True)
            (src / "sample.mgf").write_text(VALID_MGF)
        written = stage_nine_species_mgfs(self.source, self.archive, overwrite=True)
        self.assertEqual(len(written), len(NINE_SPECIES_SOURCE_DIRS))
        human_mgf = self.archive / "external/nine_species/H.-sapiens/sample.mgf"
        self.assertTrue(human_mgf.is_file())

    def test_external_benchmark_scope(self) -> None:
        kol = self.archive / "benchmarks/kingdoms_of_life/subsets"
        kol.mkdir(parents=True)
        (kol / "human.mgf").write_text(VALID_MGF * 10_000)
        for _benchmark_dir, relative in NINE_SPECIES_SOURCE_DIRS:
            src = self.source / relative
            src.mkdir(parents=True, exist_ok=True)
            (src / "sample.mgf").write_text(VALID_MGF)
        stage_nine_species_mgfs(self.source, self.archive, overwrite=True)
        stage_archive_readme(self.archive, dry_run=False, overwrite=False)
        self.assertEqual(validate_external_benchmark_scope(self.archive), [])

        legacy = self.archive / "benchmarks/nine_species/human"
        legacy.mkdir(parents=True)
        (legacy / "external.mgf").write_text(VALID_MGF)
        errors = validate_external_benchmark_scope(self.archive)
        self.assertTrue(any("must not be archived" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
