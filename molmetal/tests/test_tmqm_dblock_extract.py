"""Tests for the WF-28 Phase 1 d-block metal complex extraction pipeline.

The extraction pipeline is implemented in
``molmetal/scripts/extract_tmqm_dblock.py`` and produces two artefacts:

* ``molmetal/data/tmqm_dblock_subset.csv``  (5,000-row stratified subset)
* ``molmetal/reports/wf28_wetlab_data/tmqm_dblock_summary.md``
* ``molmetal/reports/wf28_wetlab_data/tmqm_dblock_stats.json``

These tests pin down the **invariants** the wet-lab validation tier
relies on so future schema changes or sampling regressions are caught
immediately:

1. CSV shape and column contract (~5,000 rows, 14 expected columns).
2. The 30 d-block metals from PRIMITIVE_ATOMS are all represented.
3. Every row satisfies the heavy-atom window [8, 60].
4. Every row satisfies the coordination-number window [2, 9].
5. Every SMILES parses with RDKit and yields a mol with a metal atom.
6. RDKit descriptors (MW / LogP / QED / SA / n_atoms / n_rings) are finite.
7. Anchor metals (Pt / Ru / Ir / Pd / Au / Rh) each have >= 50 samples
   so wet-lab procurement batches can target the metallodrug core.
8. The stratified sampler returns ``n_target`` rows total.
9. The per-molecule featuriser returns the documented schema.
10. The summary markdown reports the same per-metal counts as the CSV.

The tests are skipped (not failed) when the raw tmQM release is not
mounted under ``$MOLMETAL_TMQM_DIR`` — that mirrors the contract in
``tests/test_tmqm_loading.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.tmqm import TRANSITION_METALS, tmqm_dir  # noqa: E402
from molmetal.molmetal_lam.atoms.combinators import (  # noqa: E402
    METAL_SYMBOLS,
    PRIMITIVE_ATOMS,
)

# ---------------------------------------------------------------------------
# Paths & skips
# ---------------------------------------------------------------------------
CSV_PATH = PROJECT_ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv"
SUMMARY_MD = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf28_wetlab_data" / "tmqm_dblock_summary.md"
)
STATS_JSON = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf28_wetlab_data" / "tmqm_dblock_stats.json"
)

pytestmark = pytest.mark.skipif(
    not (tmqm_dir() / "tmQM_y.csv").exists(),
    reason=f"tmQM release not available at {tmqm_dir()}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def csv_df() -> pd.DataFrame:
    """Load the on-disk extraction result (5,000 rows)."""
    return pd.read_csv(CSV_PATH)


@pytest.fixture(scope="module")
def stats() -> dict:
    """Load the machine-readable companion stats JSON."""
    return json.loads(STATS_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Shape & columns
# ---------------------------------------------------------------------------
def test_csv_exists_and_has_expected_shape(csv_df: pd.DataFrame) -> None:
    assert len(csv_df) == 5000, f"expected 5000 rows, got {len(csv_df)}"
    expected_cols = {
        "csd_code",
        "smiles",
        "metal_center",
        "formal_charge",
        "spin",
        "coord_number",
        "n_atoms",
        "mw",
        "logp",
        "qed_score",
        "sa_score",
        "sa_score_unit",
        "has_aromatic",
        "n_rings",
    }
    missing = expected_cols - set(csv_df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    # CSD codes are the corpus primary key — no duplicates.
    assert csv_df["csd_code"].is_unique, "duplicate CSD codes in subset"


# ---------------------------------------------------------------------------
# 2. Metal coverage contract — all 30 d-block metals present
# ---------------------------------------------------------------------------
def test_covers_all_30_dblock_metals(csv_df: pd.DataFrame, stats: dict) -> None:
    present = set(csv_df["metal_center"].unique().tolist())
    expected = set(TRANSITION_METALS)
    missing = expected - present
    extra = present - expected
    assert not missing, f"metals missing from subset: {sorted(missing)}"
    assert not extra, f"non-d-block metals in subset: {sorted(extra)}"
    assert csv_df["metal_center"].nunique() == 30
    assert set(stats["per_metal"].keys()) == expected


def test_metal_center_matches_primitive_atoms(csv_df: pd.DataFrame) -> None:
    """Every metal_center value must be in PRIMITIVE_ATOMS.METAL_SYMBOLS."""
    bad = set(csv_df["metal_center"].unique()) - set(METAL_SYMBOLS)
    assert not bad, f"metal_center values not in METAL_SYMBOLS: {sorted(bad)}"
    assert len(METAL_SYMBOLS) == 30
    assert len(PRIMITIVE_ATOMS) == 42


# ---------------------------------------------------------------------------
# 3. Heavy-atom window [8, 60]
# ---------------------------------------------------------------------------
def test_heavy_atom_window(csv_df: pd.DataFrame) -> None:
    n = csv_df["n_atoms"].astype(int)
    assert (n >= 8).all(), f"rows below heavy-atom floor: {n[n < 8].head().tolist()}"
    assert (n <= 60).all(), f"rows above heavy-atom ceiling: {n[n > 60].head().tolist()}"


# ---------------------------------------------------------------------------
# 4. Coordination-number window [2, 9]
# ---------------------------------------------------------------------------
def test_coordination_number_window(csv_df: pd.DataFrame) -> None:
    cn = csv_df["coord_number"].astype(int)
    assert (cn >= 2).all(), f"coord_number below floor: {cn[cn < 2].head().tolist()}"
    assert (cn <= 9).all(), f"coord_number above ceiling: {cn[cn > 9].head().tolist()}"


# ---------------------------------------------------------------------------
# 5. SMILES parseability + 6. RDKit descriptors finite
# ---------------------------------------------------------------------------
def test_smiles_parseable_and_descriptors_finite(csv_df: pd.DataFrame) -> None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    parse_failures: list[tuple[str, str]] = []
    n_fail = 0
    for _, row in csv_df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            n_fail += 1
            parse_failures.append((str(row["csd_code"]), str(row["smiles"])[:60]))
            continue
        # Must contain a metal atom matching metal_center.
        syms = {a.GetSymbol() for a in mol.GetAtoms()}
        assert str(row["metal_center"]) in syms, (
            f"metal {row['metal_center']} not present in {row['csd_code']}: {syms}"
        )
    assert n_fail == 0, f"RDKit failed on {n_fail} SMILES; first: {parse_failures[:3]}"


def test_descriptor_columns_finite(csv_df: pd.DataFrame) -> None:
    """The five RDKit-computed numerics must all be finite in [0, ~].  SA is
    mapped via ``sa_score_to_unit`` so values in [0.0, 1.0] are valid; MW
    / n_atoms must be positive; QED / LogP are RDKit-native ranges."""
    assert csv_df["n_atoms"].astype(float).between(8, 60).all()
    assert csv_df["mw"].astype(float).gt(50).all()  # MW > 50 Da (organic fragment floor)
    assert csv_df["qed_score"].astype(float).between(0.0, 1.0).all()
    assert csv_df["sa_score"].astype(float).between(1.0, 10.0).all()
    assert csv_df["sa_score_unit"].astype(float).between(0.0, 1.0).all()
    # n_rings is a non-negative int.
    assert csv_df["n_rings"].astype(int).ge(0).all()


# ---------------------------------------------------------------------------
# 7. Anchor metals (Pt/Ru/Ir/Pd/Au/Rh) have enough samples
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metal", ["Pt", "Ru", "Ir", "Pd", "Au", "Rh"])
def test_anchor_metals_have_min_50_samples(
    csv_df: pd.DataFrame, stats: dict, metal: str
) -> None:
    n = int((csv_df["metal_center"] == metal).sum())
    assert n >= 50, f"anchor metal {metal} has only {n} samples (need >=50)"
    assert stats["per_metal"].get(metal, 0) == n


# ---------------------------------------------------------------------------
# 8. Stratified sampler returns the requested count
# ---------------------------------------------------------------------------
def test_stratified_sample_indices_respects_target() -> None:
    from molmetal.scripts.extract_tmqm_dblock import stratified_sample_indices

    # Four groups with non-overlapping upstream index ranges — mirrors the
    # production use case where each group's `idxs` are positions in the
    # shuffled, featurised table.
    groups = [
        ("A", list(range(0, 30))),
        ("B", list(range(30, 100))),
        ("C", list(range(100, 200))),
        ("D", list(range(200, 400))),
    ]
    chosen = stratified_sample_indices(groups, 100)
    assert len(chosen) == 100
    # All chosen indices fall inside the union of available indices.
    pool = set().union(*(idxs for _m, idxs in groups))
    assert set(chosen).issubset(pool)
    # No duplicates (production table has globally-unique upstream indices).
    assert len(set(chosen)) == len(chosen)

    # The split should respect the proportional allocation roughly.
    # Group D (size 200 of 400) should get ~half of n_target=100.
    d_count = sum(1 for i in chosen if 200 <= i < 400)
    assert 40 <= d_count <= 60, f"D group got {d_count} (expected ~50)"

    # Empty corpus -> empty result (defensive guard).
    assert stratified_sample_indices([], 100) == []
    assert stratified_sample_indices([("A", [])], 100) == []

    # n_target larger than total — must return at most total without crashing.
    chosen2 = stratified_sample_indices(groups, 10_000)
    assert len(chosen2) == 30 + 70 + 100 + 200  # all available rows


# ---------------------------------------------------------------------------
# 9. Per-molecule featuriser contract
# ---------------------------------------------------------------------------
def test_featurise_one_schema_and_parse_fail_fallback() -> None:
    from molmetal.scripts.extract_tmqm_dblock import featurise_one

    out = featurise_one("CCO")  # ethanol — no metal, but should still parse
    assert isinstance(out, dict)
    assert set(out.keys()) == {
        "n_atoms",
        "mw",
        "logp",
        "qed_score",
        "sa_score",
        "sa_score_unit",
        "has_aromatic",
        "n_rings",
    }
    assert out["n_atoms"] == 3  # ethanol has 3 heavy atoms (well below window)
    assert out["n_rings"] == 0
    assert out["has_aromatic"] is False
    assert 0.0 <= out["qed_score"] <= 1.0
    assert 1.0 <= out["sa_score"] <= 10.0

    # Pathological SMILES — must not raise, must return the fallback dict.
    bad = featurise_one("not_a_smiles!!!")
    assert bad["n_atoms"] is None
    assert bad["has_aromatic"] is False


# ---------------------------------------------------------------------------
# 10. Summary markdown stays in sync with the CSV
# ---------------------------------------------------------------------------
def test_summary_md_present_and_in_sync(csv_df: pd.DataFrame, stats: dict) -> None:
    assert SUMMARY_MD.exists(), f"summary markdown missing at {SUMMARY_MD}"
    text = SUMMARY_MD.read_text(encoding="utf-8")
    # Must mention every d-block metal so the wet-lab operator can see coverage.
    for metal in sorted(stats["per_metal"].keys()):
        assert f"| {metal} |" in text, f"summary missing row for {metal}"
    # Header sanity.
    assert "WF-28 Phase 1" in text
    assert "tmQM 108K" in text
    assert "Stratified random sample" in text or "stratified" in text.lower()


def test_stats_json_present_and_consistent(stats: dict, csv_df: pd.DataFrame) -> None:
    assert STATS_JSON.exists(), f"stats JSON missing at {STATS_JSON}"
    # The stats JSON must report the same per-metal totals as the CSV.
    csv_counts = csv_df["metal_center"].value_counts().to_dict()
    json_counts = stats["per_metal"]
    assert csv_counts == json_counts, (
        f"CSV vs stats JSON mismatch: diff={set(csv_counts.items()) ^ set(json_counts.items())}"
    )
    # The window and seed fields must be populated.
    assert stats["heavy_atom_window"] == [8, 60]
    assert stats["coord_window"] == [2, 9]
    assert isinstance(stats["seed"], int)
    assert stats["n_sampled"] == len(csv_df)
    assert stats["metal_symbols_count"] == 30
    assert stats["primitive_atoms_count"] == 42
    assert stats["transition_metals_in_tmqm"] == 30
