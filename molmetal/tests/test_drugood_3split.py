"""Tests for DrugOOD (Ji 2023 ICLR) group-key splits.

DrugOOD enforces zero overlap of three group keys between train and test:

* ``assay_id``       — ChEMBL bioassay identifier (33,179 unique in train).
* ``size_split_id``  — molecular-size bucket (277 unique in train).
* ``scaffold_split_id`` — Bemis-Murcko scaffold hash (51,296 unique in train).

These tests cover:

1. ``test_assay_split_no_overlap``     — ``AssayGroupSplitter`` produces
   pairwise-disjoint ``assay_id`` sets across train/val/test (DrugOOD
   Ji 2023 assay-split invariant).
2. ``test_size_split_no_overlap``      — same invariant for ``SizeGroupSplitter``
   (``size_split_id`` column).
3. ``test_scaffold_split_no_overlap``  — regression for the existing
   Bemis-Murcko ``ScaffoldSplitter`` (re-asserts the canonical behaviour
   with the new splitter family).
4. ``test_random_split_baseline``      — ``RandomSplitter`` baseline
   regression (80/10/10 with seed=42).

Plus integration tests:

* ``test_cli_split_choices_include_assay_size`` — argparse ``--split``
  choices include the new ``assay`` and ``size`` keys.
* ``test_splitter_factory_dispatch`` — ``SPLITTER_FACTORIES[args.split]``
  returns the right class for each name.
* ``test_assay_split_on_drugood_rows`` — end-to-end: load the real
  DrugOOD CSV, apply ``AssayGroupSplitter``, verify zero ``assay_id``
  overlap between splits.  Skipped if the DrugOOD CSV is missing.

Backward-compat tests:

* ``test_existing_split_choices_unchanged`` — the five legacy
  ``--split`` choices (chemical, ligand_dedup, random, scaffold,
  temporal) still resolve to the same splitter classes they did
  before this change.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from molmetal.baselines.dmpnn import SPLITTER_FACTORIES
from molmetal.data.splits import (
    AssayGroupSplitter,
    GroupIdSplitter,
    LigandDeduplicatedSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    SizeGroupSplitter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dict_dataset(rows: List[Dict[str, object]]) -> "_DictDataset":
    """Tiny dataset adapter that yields per-row dicts on ``__getitem__``.

    Mirrors the contract of ``molmetal.data.four_domain_combined._DrugOODRows``
    for testing without loading the 1.58M-row DrugOOD CSV.
    """

    class _DictDataset:
        def __init__(self, rows):
            self._rows = list(rows)

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, i):
            return self._rows[int(i)]

    return _DictDataset(rows)


def _groups_for(idx: np.ndarray, dataset) -> set:
    """Return the unique assay_id / size_id / etc. for ``idx``."""
    out = set()
    for i in idx:
        # We always read from the dict-style row (test helper).
        out.add(int(dataset[int(i)]["assay_id"]))
    return out


def _groups_for_attr(idx: np.ndarray, dataset, attr: str) -> set:
    out = set()
    for i in idx:
        out.add(int(dataset[int(i)][attr]))
    return out


# ---------------------------------------------------------------------------
# (1) Assay split — group-by-assay_id
# ---------------------------------------------------------------------------
class TestAssaySplit:
    def test_assay_split_no_overlap(self) -> None:
        """DrugOOD Ji 2023 assay-split invariant: no shared assay_id across splits."""
        # Build a toy dataset with 5 distinct assays, each appearing
        # 6 times (so the largest-first strategy has enough rows to
        # populate all three buckets).
        rows: List[Dict[str, object]] = []
        for assay in range(5):
            for k in range(6):
                rows.append({
                    "smiles": f"c1ccccc{k}",
                    "y": 1,
                    "assay_id": assay,
                    "scaffold_split_id": 100 + assay,
                    "size_split_id": 10 + assay,
                })
        ds = _make_dict_dataset(rows)
        sp = AssayGroupSplitter(strategy="largest_first", seed=42)
        result = sp(ds)

        # All 30 indices assigned exactly once
        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == len(ds)

        # Group disjointness — the Ji 2023 invariant.
        train_g = _groups_for(result.train_idx, ds)
        val_g = _groups_for(result.val_idx, ds)
        test_g = _groups_for(result.test_idx, ds)
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)
        assert val_g.isdisjoint(test_g)

        # Union covers all 5 assays
        assert train_g | val_g | test_g == {0, 1, 2, 3, 4}

    def test_assay_split_seed_reproducible(self) -> None:
        rows = [
            {"smiles": f"c{k}", "y": 0, "assay_id": k // 2, "scaffold_split_id": k, "size_split_id": k % 3}
            for k in range(20)
        ]
        ds = _make_dict_dataset(rows)
        r1 = AssayGroupSplitter(strategy="largest_first", seed=7)(ds)
        r2 = AssayGroupSplitter(strategy="largest_first", seed=7)(ds)
        np.testing.assert_array_equal(r1.train_idx, r2.train_idx)
        np.testing.assert_array_equal(r1.val_idx, r2.val_idx)
        np.testing.assert_array_equal(r1.test_idx, r2.test_idx)


# ---------------------------------------------------------------------------
# (2) Size split — group-by-size_split_id
# ---------------------------------------------------------------------------
class TestSizeSplit:
    def test_size_split_no_overlap(self) -> None:
        """DrugOOD Ji 2023 size-split invariant: no shared size_split_id across splits."""
        rows: List[Dict[str, object]] = []
        # 4 distinct size bins, 7 rows each — same semantics as the
        # assay test, exercises the largest-first assignment.
        for size_bin in range(4):
            for k in range(7):
                rows.append({
                    "smiles": f"c{k}",
                    "y": 1,
                    "assay_id": size_bin + 100,  # different from size_bin
                    "scaffold_split_id": size_bin + 200,
                    "size_split_id": size_bin,
                })
        ds = _make_dict_dataset(rows)
        sp = SizeGroupSplitter(strategy="largest_first", seed=42)
        result = sp(ds)

        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == len(ds)

        train_g = _groups_for_attr(result.train_idx, ds, "size_split_id")
        val_g = _groups_for_attr(result.val_idx, ds, "size_split_id")
        test_g = _groups_for_attr(result.test_idx, ds, "size_split_id")
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)
        assert val_g.isdisjoint(test_g)
        assert train_g | val_g | test_g == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# (3) Scaffold split — regression for existing Bemis-Murcko behavior
# ---------------------------------------------------------------------------
class TestScaffoldSplitRegression:
    def test_scaffold_split_no_overlap(self, tmp_path) -> None:
        """``ScaffoldSplitter`` still groups by Bemis-Murcko and produces
        a disjoint partition — regression guard against the GroupIdSplitter
        refactor accidentally altering the legacy class."""
        from molmetal.data.splits import ScaffoldSplitter

        # Build a MetalCytotoxDataset-shaped stand-in via the
        # scaffold splitter's own __call__ pipeline.
        import pandas as pd

        smiles_list = (
            ["c1ccccc1"] * 6
            + ["c1ccc2ccccc2c1"] * 6
            + ["c1ccsc1"] * 6
            + ["c1cnc[nH]1"] * 6
        )
        n = len(smiles_list)
        df = pd.DataFrame(
            {
                "SMILES_Ligands": smiles_list,
                "Metal": ["Ru"] * n,
                "Cell_line": ["HeLa"] * n,
                "IC50_Dark_value": [5.0] * n,
                "Time(h)": [24] * n,
                "Year": [2023] * n,
                "Charge_complex": [0] * n,
                "Oxidation_state": [2] * n,
            }
        )
        from molmetal.data.cytotox import MetalCytotoxDataset

        ds = MetalCytotoxDataset(df=df, cache_dir=tmp_path / "cache")
        sp = ScaffoldSplitter(strategy="largest_first", seed=42)
        result = sp(ds)

        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == n

        # Validate scaffold disjointness by replaying the Bemis-Murcko
        # computation.
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")

        def _key(smi: str) -> str:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return f"__empty__{smi}"
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            if sc is None:
                return f"__empty__{smi}"
            return Chem.MolToSmiles(sc)

        train_sc = {_key(s) for s in ds.smiles[result.train_idx]}
        val_sc = {_key(s) for s in ds.smiles[result.val_idx]}
        test_sc = {_key(s) for s in ds.smiles[result.test_idx]}
        assert train_sc.isdisjoint(val_sc)
        assert train_sc.isdisjoint(test_sc)
        assert val_sc.isdisjoint(test_sc)


# ---------------------------------------------------------------------------
# (4) Random split — baseline regression
# ---------------------------------------------------------------------------
class TestRandomSplitBaseline:
    def test_random_split_baseline(self) -> None:
        """RandomSplitter still returns an 80/10/10 split with seed=42."""
        from molmetal.data.splits import RandomSplitter

        # Build a synthetic dataset-like object.
        n = 1000

        class _NumpyDS:
            def __init__(self):
                self._smiles = np.array([f"c{k}" for k in range(n)], dtype=object)

            def __len__(self):
                return n

            @property
            def smiles(self):
                return self._smiles

        ds = _NumpyDS()
        sp = RandomSplitter(seed=42)
        result = sp(ds)
        assert result.n_train() == 800
        assert result.n_val() == 100
        assert result.n_test() == 100
        # No overlap
        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == n


# ---------------------------------------------------------------------------
# (5) CLI integration — argparse choices + factory dispatch
# ---------------------------------------------------------------------------
class TestCLIIntegration:
    def test_cli_split_choices_include_assay_size(self) -> None:
        """``--split`` choices include the DrugOOD Ji 2023 splits."""
        from molmetal.scripts.train_dmpnn_multitask import _parse_args

        ns = _parse_args(["--split", "assay", "--epochs", "1"])
        assert ns.split == "assay"
        ns = _parse_args(["--split", "size", "--epochs", "1"])
        assert ns.split == "size"

    def test_splitter_factory_dispatch(self) -> None:
        """``SPLITTER_FACTORIES`` maps the new keys to the right classes."""
        assert SPLITTER_FACTORIES["assay"](42).__class__.__name__ == "AssayGroupSplitter"
        assert SPLITTER_FACTORIES["size"](42).__class__.__name__ == "SizeGroupSplitter"

    def test_existing_split_choices_unchanged(self) -> None:
        """The five legacy ``--split`` choices still resolve to the same
        splitter classes they did before this change (backward-compat)."""
        # Each tuple: (key, expected_class_name)
        legacy = [
            ("random", "RandomSplitter"),
            ("ligand_dedup", "LigandDeduplicatedSplitter"),
            ("scaffold", "ScaffoldSplitter"),
            ("temporal", "TemporalSplitter"),
            ("chemical", "ChemicalSplitter"),
        ]
        for key, cls_name in legacy:
            assert SPLITTER_FACTORIES[key](42).__class__.__name__ == cls_name, (
                f"--split {key} now resolves to a different class"
            )

    def test_cli_rejects_unknown_split(self) -> None:
        """argparse rejects ``--split bogus`` (no new choices leak through)."""
        from molmetal.scripts.train_dmpnn_multitask import _parse_args

        with pytest.raises(SystemExit):
            _parse_args(["--split", "bogus_split_choice"])


# ---------------------------------------------------------------------------
# (6) End-to-end on real DrugOOD CSV (if available)
# ---------------------------------------------------------------------------
DRUGOOD_TRAIN_CSV = "/home/hugo/codes/mol-metal/.data/molmetal/drugood/ic50/data/train.csv"


@pytest.mark.skipif(
    not __import__("os").path.exists(DRUGOOD_TRAIN_CSV),
    reason=f"DrugOOD CSV not found at {DRUGOOD_TRAIN_CSV}",
)
class TestDrugOODRealCSV:
    """End-to-end smoke tests against the real DrugOOD ChEMBL_29 IC50 CSV."""

    def _load_drugood(self) -> "_RealDrugOOD":
        """Load just the assay_id / size_split_id / scaffold_split_id columns
        of the DrugOOD CSV into a tiny dict-style dataset adapter.  Reading
        only the three id columns (not the smiles) keeps the test under
        ~150 MB RAM even on the 1.58M-row file."""
        import csv

        class _RealDrugOOD:
            def __init__(self, rows):
                self._rows = rows

            def __len__(self):
                return len(self._rows)

            def __getitem__(self, i):
                return self._rows[int(i)]

        rows = []
        with open(DRUGOOD_TRAIN_CSV, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({
                    "assay_id": int(row["assay_id"]),
                    "size_split_id": int(row["size_split_id"]),
                    "scaffold_split_id": int(row["scaffold_split_id"]),
                })
                # Cap at 30k rows to keep the test fast.
                if len(rows) >= 30000:
                    break
        return _RealDrugOOD(rows)

    def test_assay_split_on_drugood_rows(self) -> None:
        ds = self._load_drugood()
        sp = AssayGroupSplitter(strategy="largest_first", seed=42)
        result = sp(ds)

        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == len(ds)

        train_g = _groups_for(result.train_idx, ds)
        val_g = _groups_for(result.val_idx, ds)
        test_g = _groups_for(result.test_idx, ds)
        # DrugOOD Ji 2023 assay invariant.
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)
        assert val_g.isdisjoint(test_g)

    def test_size_split_on_drugood_rows(self) -> None:
        ds = self._load_drugood()
        sp = SizeGroupSplitter(strategy="largest_first", seed=42)
        result = sp(ds)

        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == len(ds)

        train_g = _groups_for_attr(result.train_idx, ds, "size_split_id")
        val_g = _groups_for_attr(result.val_idx, ds, "size_split_id")
        test_g = _groups_for_attr(result.test_idx, ds, "size_split_id")
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)
        assert val_g.isdisjoint(test_g)


# ---------------------------------------------------------------------------
# (7) Error-path tests — missing column raises a clear error
# ---------------------------------------------------------------------------
class TestErrorPaths:
    def test_assay_split_raises_when_assay_id_missing(self, tmp_path) -> None:
        """``AssayGroupSplitter`` raises a clear error when the dataset
        does not expose an ``assay_id`` column (legacy MetalCytoToxDB
        rows in single-domain mode)."""
        from molmetal.data.cytotox import MetalCytotoxDataset
        import pandas as pd

        # MetalCytoToxDB-shaped dataset (no assay_id column).
        df = pd.DataFrame({
            "SMILES_Ligands": ["c1ccccc1"] * 5,
            "Metal": ["Ru"] * 5,
            "Cell_line": ["HeLa"] * 5,
            "IC50_Dark_value": [5.0] * 5,
            "Time(h)": [24] * 5,
            "Year": [2023] * 5,
            "Charge_complex": [0] * 5,
            "Oxidation_state": [2] * 5,
        })
        # Use a tmp_path cache_dir because the default
        # ``/mnt/storage/data/molmetal/3d_cache`` is unwritable on most
        # dev boxes.
        ds = MetalCytotoxDataset(df=df, cache_dir=tmp_path / "cache")
        sp = AssayGroupSplitter()
        with pytest.raises(ValueError, match="assay_id"):
            sp(ds)

    def test_size_split_raises_when_size_id_missing(self, tmp_path) -> None:
        from molmetal.data.cytotox import MetalCytotoxDataset
        import pandas as pd

        df = pd.DataFrame({
            "SMILES_Ligands": ["c1ccccc1"] * 5,
            "Metal": ["Ru"] * 5,
            "Cell_line": ["HeLa"] * 5,
            "IC50_Dark_value": [5.0] * 5,
            "Time(h)": [24] * 5,
            "Year": [2023] * 5,
            "Charge_complex": [0] * 5,
            "Oxidation_state": [2] * 5,
        })
        ds = MetalCytotoxDataset(df=df, cache_dir=tmp_path / "cache")
        sp = SizeGroupSplitter()
        with pytest.raises(ValueError, match="size_id"):
            sp(ds)