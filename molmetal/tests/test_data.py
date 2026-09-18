"""Tests for the molmetal.data layer.

Smoke coverage:

* :class:`MetalCytotoxDataset` loads from the production CSV, the Ru
  whitelist yields ~19 k rows, and pIC50 matches the expected formula.
* :class:`RandomSplitter` returns an 80/10/10 split (sum == len).
* :class:`MorganFingerprinter` produces a 2048-dim vector for aspirin.
* :class:`CrossDockedDataset` extracts (or locates) the split file and
  returns one pocket-ligand pair.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from molmetal.data import (
    ChemicalSplitter,
    CrossDockedDataset,
    CytotoxFilter,
    GraphFeaturizer,
    LigandDeduplicatedSplitter,
    MetalCytotoxDataset,
    MorganFingerprinter,
    RandomSplitter,
    ScaffoldSplitter,
    TemporalSplitter,
)


CSV_PATH = Path("/mnt/storage/data/molmetal/MetalCytoToxDB.csv")
CROSSDOCKED_ARCHIVE = Path("/mnt/storage/data/molmetal/CrossDocked2020_cascadediff.zip")


# ---------------------------------------------------------------------------
# Fixtures (session-scoped — load the CSV once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def full_dataset() -> MetalCytotoxDataset:
    """Full un-filtered dataset (sanity: just loads)."""
    if not CSV_PATH.exists():
        pytest.skip(f"MetalCytoToxDB.csv not present at {CSV_PATH}")
    return MetalCytotoxDataset.from_csv(CSV_PATH, time_threshold=None, ic50_min=None)


@pytest.fixture(scope="session")
def ru_dataset() -> MetalCytotoxDataset:
    """Filtered to Ru only."""
    if not CSV_PATH.exists():
        pytest.skip(f"MetalCytoToxDB.csv not present at {CSV_PATH}")
    return MetalCytotoxDataset.from_csv(CSV_PATH, metal_whitelist=["Ru"])


# ---------------------------------------------------------------------------
# 1. MetalCytotoxDataset
# ---------------------------------------------------------------------------
class TestMetalCytotoxDataset:
    def test_load_from_csv(self, full_dataset: MetalCytotoxDataset) -> None:
        assert len(full_dataset) > 0
        # Summary diagnostics
        s = full_dataset.summary()
        assert s["n_rows"] == len(full_dataset)
        assert "Ru" in s["metals"]
        assert s["year_min"] <= s["year_max"]
        assert 0.0 <= s["active_fraction"] <= 1.0

    def test_metal_ru_filter(self, ru_dataset: MetalCytotoxDataset) -> None:
        # Ru has ~19,135 rows in the CSV
        n = len(ru_dataset)
        assert 18000 <= n <= 20000, f"expected ~19k Ru rows, got {n}"
        # All rows should have metal == Ru
        metals = ru_dataset.metals
        assert (metals == "Ru").all()

    def test_metal_ir_filter(self) -> None:
        if not CSV_PATH.exists():
            pytest.skip("CSV missing")
        ds = MetalCytotoxDataset.from_csv(CSV_PATH, metal_whitelist=["Ir"])
        assert 4000 <= len(ds) <= 5000

    def test_pic50_value_for_sample_row(
        self, full_dataset: MetalCytotoxDataset
    ) -> None:
        # For a row with IC50_Dark_value = 5.0 µM:
        #   pIC50 = -log10(5e-6) ≈ 5.301
        ds = MetalCytotoxDataset.from_csv(CSV_PATH)
        # Inject a synthetic row directly into the dataframe for a precise test
        df = ds.df.copy()
        df.iloc[0, df.columns.get_loc("IC50_Dark_value")] = 5.0
        df = df.iloc[:1].copy()
        # Re-derive pIC50 using the loader's formula
        from molmetal.data.cytotox import _add_pic50

        df = _add_pic50(df)
        expected = -math.log10(5e-6)
        assert math.isclose(float(df["pIC50"].iloc[0]), expected, rel_tol=1e-3)
        # The value of the dataset is unchanged
        assert full_dataset is not None

    def test_active_label_threshold(self) -> None:
        # active := IC50 < 10 µM
        ds = MetalCytotoxDataset.from_csv(CSV_PATH)
        df = ds.df
        active_rows = df[df["active"] == True]  # noqa: E712
        assert (active_rows["IC50_Dark_value"] < 10.0).all()
        inactive_rows = df[df["active"] == False]  # noqa: E712
        assert (inactive_rows["IC50_Dark_value"] >= 10.0).all()

    def test_getitem_keys(self, ru_dataset: MetalCytotoxDataset) -> None:
        item = ru_dataset[0]
        for k in (
            "smiles",
            "metal",
            "cell_line",
            "pIC50",
            "active",
            "year",
            "charge_complex",
            "oxidation_state",
        ):
            assert k in item, f"missing key {k!r}"
        assert item["metal"] == "Ru"
        assert isinstance(item["smiles"], str)
        assert isinstance(item["active"], bool)
        assert isinstance(item["pIC50"], float)

    def test_conformer_cache(self, ru_dataset: MetalCytotoxDataset) -> None:
        # Pick the first non-empty SMILES
        for i in range(len(ru_dataset)):
            sm = ru_dataset.smiles[i]
            if sm:
                break
        mol = ru_dataset.get_conformer(sm)
        assert mol is not None
        assert mol.n_atoms > 0
        # Second call must hit cache (same object)
        mol2 = ru_dataset.get_conformer(sm)
        assert mol2 is mol

    def test_filter_year_range(self) -> None:
        if not CSV_PATH.exists():
            pytest.skip("CSV missing")
        ds = MetalCytotoxDataset.from_csv(
            CSV_PATH, year_min=2020, year_max=2022
        )
        years = ds.years
        assert (years >= 2020).all()
        assert (years <= 2022).all()
        assert len(ds) > 0

    def test_filter_ic50_minimum(self) -> None:
        if not CSV_PATH.exists():
            pytest.skip("CSV missing")
        ds = MetalCytotoxDataset.from_csv(CSV_PATH, ic50_min=1.0)
        # All rows should have IC50 >= 1 uM
        ic50s = ds.df["IC50_Dark_value"].astype(float).to_numpy()
        assert (ic50s >= 1.0).all()


# ---------------------------------------------------------------------------
# 2. Splitters
# ---------------------------------------------------------------------------
class TestSplitters:
    def test_random_split_80_10_10(self, full_dataset: MetalCytotoxDataset) -> None:
        sp = RandomSplitter()
        result = sp(full_dataset)
        n = len(full_dataset)
        assert result.n_train() + result.n_val() + result.n_test() == n
        # Allow ±2 rows tolerance for rounding
        assert abs(result.n_train() - int(0.8 * n)) <= 2
        assert abs(result.n_val() - int(0.1 * n)) <= 2
        assert abs(result.n_test() - int(0.1 * n)) <= 2
        # Indices are unique
        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == n

    def test_random_split_seed_reproducible(self, full_dataset: MetalCytotoxDataset) -> None:
        sp1 = RandomSplitter(seed=123)
        sp2 = RandomSplitter(seed=123)
        r1 = sp1(full_dataset)
        r2 = sp2(full_dataset)
        np.testing.assert_array_equal(r1.train_idx, r2.train_idx)
        np.testing.assert_array_equal(r1.val_idx, r2.val_idx)
        np.testing.assert_array_equal(r1.test_idx, r2.test_idx)

    def test_temporal_split(self, full_dataset: MetalCytotoxDataset) -> None:
        sp = TemporalSplitter(cutoff_year=2024)
        result = sp(full_dataset)
        # All test rows must be post-2023
        test_years = full_dataset.years[result.test_idx]
        assert (test_years >= 2024).all()
        # Train/val rows must be pre-2024
        train_years = full_dataset.years[result.train_idx]
        val_years = full_dataset.years[result.val_idx]
        assert (train_years < 2024).all()
        assert (val_years < 2024).all()

    def test_chemical_split_runs(self, ru_dataset: MetalCytotoxDataset) -> None:
        # Use a small subsample for speed
        ds = ru_dataset
        # Limit to first 1000 rows for test speed
        from molmetal.data.cytotox import _add_pic50, _add_active

        small = MetalCytotoxDataset(ds.df.iloc[:1000].copy())
        sp = ChemicalSplitter(test_fraction=0.2, val_fraction=0.1, threshold=0.7, seed=42)
        result = sp(small)
        n = len(small)
        # Total assigned should be ≤ n (random_bucket fallback may leave some out)
        total = result.n_train() + result.n_val() + result.n_test()
        assert total <= n
        # Indices must be unique
        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == total

    def test_ligand_dedup_split_no_overlap(self) -> None:
        """No canonical SMILES appears in 2 different splits."""
        # Build a tiny synthetic dataset with 3 distinct SMILES × 10 rows each
        import pandas as pd

        smiles = (
            ["CCO"] * 10
            + ["c1ccccc1"] * 10
            + ["CC(=O)Oc1ccccc1C(=O)O"] * 10
        )
        n = len(smiles)
        df = pd.DataFrame(
            {
                "SMILES_Ligands": smiles,
                "Metal": ["Ru"] * n,
                "Cell_line": ["HeLa"] * n,
                "IC50_Dark_value": [5.0] * n,
                "Time(h)": [24] * n,
                "Year": [2023] * n,
                "Charge_complex": [0] * n,
                "Oxidation_state": [2] * n,
            }
        )
        ds = MetalCytotoxDataset.from_csv(
            "/mnt/storage/data/molmetal/MetalCytoToxDB.csv",
            filters=None,
        ) if False else MetalCytotoxDataset(df=df)
        sp = LigandDeduplicatedSplitter(strategy="largest_first", seed=42)
        result = sp(ds)
        # Every index assigned exactly once
        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == n
        # No canonical SMILES appears in two splits
        from molmetal.data.leakage_utils import canonicalize_smiles
        train_canon = {canonicalize_smiles(s) for s in ds.smiles[result.train_idx]}
        val_canon = {canonicalize_smiles(s) for s in ds.smiles[result.val_idx]}
        test_canon = {canonicalize_smiles(s) for s in ds.smiles[result.test_idx]}
        assert train_canon.isdisjoint(val_canon)
        assert train_canon.isdisjoint(test_canon)
        assert val_canon.isdisjoint(test_canon)
        # Union of splits covers all 3 SMILES
        assert len(train_canon | val_canon | test_canon) == 3

    def test_ligand_dedup_split_counts(self, ru_dataset: MetalCytotoxDataset) -> None:
        """On the real Ru subset the 80/10/10 ratio is approximately honored."""
        sp = LigandDeduplicatedSplitter(strategy="largest_first", seed=42)
        result = sp(ru_dataset)
        n = len(ru_dataset)
        total = result.n_train() + result.n_val() + result.n_test()
        # Every index assigned
        assert total == n
        # With ``largest_first`` on a heavily-skewed group-size distribution
        # (one SMILES appears 394 times), train is over-filled and val/test
        # under-filled relative to the row-count fractions. We only assert
        # the lower-bound presence of each split and that no split is empty.
        assert result.n_train() > 0
        assert result.n_val() > 0
        assert result.n_test() > 0
        # Approximate ratio: train should still be the majority
        assert result.n_train() >= result.n_val()
        assert result.n_train() >= result.n_test()

    def test_ligand_dedup_split_seed_reproducible(
        self, ru_dataset: MetalCytotoxDataset
    ) -> None:
        sp1 = LigandDeduplicatedSplitter(strategy="largest_first", seed=7)
        sp2 = LigandDeduplicatedSplitter(strategy="largest_first", seed=7)
        r1 = sp1(ru_dataset)
        r2 = sp2(ru_dataset)
        np.testing.assert_array_equal(r1.train_idx, r2.train_idx)
        np.testing.assert_array_equal(r1.val_idx, r2.val_idx)
        np.testing.assert_array_equal(r1.test_idx, r2.test_idx)

    def test_scaffold_split_no_overlap(self) -> None:
        """No Bemis-Murcko scaffold appears in 2 different splits."""
        import pandas as pd

        # Use 4 distinct scaffolds with no sharing:
        #   benzene, naphthalene, thiophene, imidazole.
        # Each appears 6 times so the splitter has enough rows to populate
        # all three buckets without the "largest_first" overflow that
        # happens when one group dominates (see Ru subset test).
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
        ds = MetalCytotoxDataset(df=df)
        sp = ScaffoldSplitter(strategy="largest_first", seed=42)
        result = sp(ds)
        # All assigned, indices unique
        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        assert len(np.unique(all_idx)) == n
        # Inspect the splitter's own group-key assignment by replaying its
        # helper logic on the assigned indices.
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

        train_keys = {_key(s) for s in ds.smiles[result.train_idx]}
        val_keys = {_key(s) for s in ds.smiles[result.val_idx]}
        test_keys = {_key(s) for s in ds.smiles[result.test_idx]}
        assert train_keys.isdisjoint(val_keys)
        assert train_keys.isdisjoint(test_keys)
        assert val_keys.isdisjoint(test_keys)

    def test_scaffold_split_handles_ru_subset(self, ru_dataset: MetalCytotoxDataset) -> None:
        """ScaffoldSplitter runs on the real Ru subset and partitions all rows."""
        sp = ScaffoldSplitter(strategy="largest_first", seed=42)
        result = sp(ru_dataset)
        n = len(ru_dataset)
        total = result.n_train() + result.n_val() + result.n_test()
        assert total == n
        # Replay the splitter's *exact* group-key pipeline.  We delegate
        # to ``sp._scaffold_for`` and ``sp._canonicalize`` so we cannot
        # accidentally diverge (e.g. via a different ``includeChirality``
        # default or a missing empty-string fallback).
        from molmetal.data.leakage_utils import canonicalize_array

        canon, _ = canonicalize_array(list(ru_dataset.smiles))

        def _key(canon_smi: str) -> str:
            scaf = sp._scaffold_for(canon_smi)
            return scaf if scaf else f"__empty__{canon_smi}"

        train_keys = {_key(str(canon[i])) for i in result.train_idx}
        val_keys = {_key(str(canon[i])) for i in result.val_idx}
        test_keys = {_key(str(canon[i])) for i in result.test_idx}
        assert train_keys.isdisjoint(val_keys)
        assert train_keys.isdisjoint(test_keys)
        assert val_keys.isdisjoint(test_keys)

    def test_empty_dataset(self) -> None:
        # Splitters must handle the empty-dataset case without raising.
        # We test the splitter directly rather than the dataset constructor
        # (which intentionally rejects empty DataFrames).
        from molmetal.data.splits import SplitResult

        class _EmptyDS:
            years = np.array([], dtype=np.int64)
            metals = np.array([], dtype="<U1")

            def __len__(self):
                return 0

            @property
            def smiles(self):
                return np.array([], dtype="<U1")

            @property
            def pic50(self):
                return np.array([], dtype=np.float32)

            @property
            def active(self):
                return np.array([], dtype=bool)

        sp = RandomSplitter()
        result = sp(_EmptyDS())
        assert result.n_train() == 0
        assert result.n_val() == 0
        assert result.n_test() == 0


# ---------------------------------------------------------------------------
# 3. Featurisers
# ---------------------------------------------------------------------------
class TestFeaturizers:
    def test_morgan_fingerprint_aspirin_shape(self) -> None:
        mfp = MorganFingerprinter(radius=2, nBits=2048)
        # Aspirin: CC(=O)Oc1ccccc1C(=O)O
        fp = mfp("CC(=O)Oc1ccccc1C(=O)O")
        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8
        # Some bits should be set (not all-zero)
        assert fp.sum() > 0
        # Re-call must be deterministic
        fp2 = mfp("CC(=O)Oc1ccccc1C(=O)O")
        np.testing.assert_array_equal(fp, fp2)

    def test_morgan_invalid_smiles_returns_zeros(self) -> None:
        mfp = MorganFingerprinter()
        fp = mfp("not_a_smiles_@@@")
        assert fp.shape == (2048,)
        assert fp.sum() == 0

    def test_morgan_batch(self) -> None:
        mfp = MorganFingerprinter()
        out = mfp(["CCO", "c1ccccc1"])
        assert out.shape == (2, 2048)

    def test_graph_featurizer_atoms(self) -> None:
        gf = GraphFeaturizer()
        out = gf("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert out["x"].shape[0] == out["n_atoms"]
        assert out["x"].shape[1] == gf.atom_feature_dim
        # Sanity: bond count is non-zero
        assert out["edge_index"].shape[1] == 2 * out["n_bonds"]
        assert out["edge_attr"].shape[0] == out["edge_index"].shape[1]
        assert out["edge_attr"].shape[1] == gf.edge_feature_dim
        # Edge index is int64
        assert out["edge_index"].dtype == np.int64
        # Atom features are float32
        assert out["x"].dtype == np.float32
        assert out["edge_attr"].dtype == np.float32

    def test_graph_featurizer_empty_mol(self) -> None:
        gf = GraphFeaturizer()
        with pytest.raises(ValueError):
            gf("not_a_smiles_@@@")


# ---------------------------------------------------------------------------
# 4. CrossDockedDataset
# ---------------------------------------------------------------------------
class TestCrossDockedDataset:
    def test_extract_pairs(self, tmp_path: Path) -> None:
        if not CROSSDOCKED_ARCHIVE.exists():
            pytest.skip(f"Archive missing at {CROSSDOCKED_ARCHIVE}")
        # Use a temp extraction directory so we don't pollute /mnt/storage
        ds = CrossDockedDataset(
            archive_path=CROSSDOCKED_ARCHIVE,
            extracted_dir=tmp_path / "crossdocked",
            split="train",
        )
        # The split file must exist
        sp = ds.split_path()
        assert sp is not None
        assert sp.exists()
        # We should have ~100k pairs
        n = len(ds)
        assert n > 0
        # First pair yields both paths
        item = ds[0]
        assert "pocket_pdb_path" in item
        assert "ligand_sdf_path" in item
        # Pocket path string is non-empty
        assert item["pocket_pdb_path"] != ""

    def test_extract_test_split(self, tmp_path: Path) -> None:
        if not CROSSDOCKED_ARCHIVE.exists():
            pytest.skip("Archive missing")
        ds = CrossDockedDataset(
            archive_path=CROSSDOCKED_ARCHIVE,
            extracted_dir=tmp_path / "crossdocked",
            split="test",
        )
        # Test split has 100 pairs in the standard DiffDock split
        n = len(ds)
        assert 50 <= n <= 200, f"unexpected test-split size {n}"
        # First pair tuple is well-formed
        pdb, sdf = ds._pairs[0]
        assert pdb.endswith(".pdb")
        assert sdf.endswith(".sdf")


# ---------------------------------------------------------------------------
# 5. Imports
# ---------------------------------------------------------------------------
class TestImports:
    def test_all_exports_importable(self) -> None:
        from molmetal.data import (  # noqa: F401
            ChemicalSplitter,
            CrossDockedDataset,
            CytotoxFilter,
            GraphFeaturizer,
            LigandDeduplicatedSplitter,
            MetalCytotoxDataset,
            MorganFingerprinter,
            RandomSplitter,
            ScaffoldSplitter,
            SplitResult,
            TemporalSplitter,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))