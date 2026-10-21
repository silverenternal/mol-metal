"""The NCI-60 9-cell split: shipped protocol vs the coverage-stratified one.

Pins three things that were measured by
``molmetal/scripts/wf_a62_nci60_split_coverage_audit.py``:

* the default split is the shipped m21 protocol, bit-for-bit
  (7,320 / 160 / 343 rows; 4,860 / 115 / 232 molecules);
* that protocol starves small cell lines (SKOV3 3 test rows, DU145 7);
* the opt-in coverage split gives every cell >= 20 test rows while keeping the
  one-scaffold-one-split invariant, and does not depend on CSV row order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molmetal.data.splits import _bemis_murcko_key
from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
    NCI60_COVERAGE_SALT,
    load_nci60_9cell_split_table,
)

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"


@pytest.fixture(scope="module")
def shipped() -> pd.DataFrame:
    return load_nci60_9cell_split_table(CSV, "scaffold_70_10_10")


@pytest.fixture(scope="module")
def coverage() -> pd.DataFrame:
    return load_nci60_9cell_split_table(CSV, "scaffold_coverage_70_10_20")


def _counts(df: pd.DataFrame) -> dict:
    return {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}


def _per_cell(df: pd.DataFrame, split: str) -> dict:
    sub = df[df["split"] == split]
    return {c: int(n) for c, n in sub.groupby("cell_line").size().items()}


def test_shipped_split_is_bit_identical(shipped: pd.DataFrame) -> None:
    assert _counts(shipped) == {"train": 7320, "val": 160, "test": 343}
    mols = shipped.groupby("split")["smiles"].nunique().to_dict()
    assert mols == {"train": 4860, "val": 115, "test": 232}


def test_shipped_split_starves_small_cells(shipped: pd.DataFrame) -> None:
    test = _per_cell(shipped, "test")
    assert test["SKOV3"] == 3
    assert test["DU145"] == 7
    assert sum(1 for n in test.values() if n < 30) == 5
    # SKOV3 has no val rows at all under the shipped split.
    assert "SKOV3" not in _per_cell(shipped, "val")


def test_coverage_split_lifts_every_cell(coverage: pd.DataFrame) -> None:
    test = _per_cell(coverage, "test")
    val = _per_cell(coverage, "val")
    assert min(test.values()) >= 20
    assert min(val.values()) >= 15
    # 8 of 9 clear the stricter 30-row bar used by the m3 line.
    assert sum(1 for n in test.values() if n >= 30) == 8


def test_coverage_split_row_fractions(coverage: pd.DataFrame) -> None:
    counts = _counts(coverage)
    total = sum(counts.values())
    assert total == len(coverage) == 7823
    assert 0.15 <= counts["test"] / total <= 0.25
    assert 0.08 <= counts["val"] / total <= 0.14


def test_coverage_split_keeps_scaffolds_whole(coverage: pd.DataFrame) -> None:
    scaffold = {
        s: (_bemis_murcko_key(s) or f"__empty__{s}")
        for s in coverage["smiles"].unique()
    }
    spans = (
        coverage.assign(scaffold=coverage["smiles"].map(scaffold))
        .groupby("scaffold")["split"]
        .nunique()
    )
    assert int((spans > 1).sum()) == 0


def test_coverage_split_partitions_every_molecule(coverage: pd.DataFrame) -> None:
    per_mol = coverage.groupby("smiles")["split"].nunique()
    assert int((per_mol > 1).sum()) == 0
    assert coverage["split"].notna().all()


def test_coverage_split_is_row_order_invariant(
    coverage: pd.DataFrame, tmp_path: Path
) -> None:
    shuffled = pd.read_csv(CSV).sample(frac=1.0, random_state=7)
    path = tmp_path / "nci60_shuffled.csv"
    shuffled.to_csv(path, index=False)
    again = load_nci60_9cell_split_table(path, "scaffold_coverage_70_10_20")
    left = coverage.set_index("smiles")["split"].to_dict()
    right = again.set_index("smiles")["split"].to_dict()
    assert left == right


def test_coverage_salt_is_recorded_and_nonzero() -> None:
    # The salt was chosen on coverage only; it must be explicit, not implicit.
    assert isinstance(NCI60_COVERAGE_SALT, int)
    assert NCI60_COVERAGE_SALT != 0


def test_default_is_now_the_coverage_split() -> None:
    """The loader's default flipped on 2026-10-21; the legacy path is opt-in."""
    default = load_nci60_9cell_split_table(CSV)
    assert _counts(default) == {"train": 5371, "val": 920, "test": 1532}
    assert min(_per_cell(default, "test").values()) >= 20


def test_audit_artifact_agrees_with_the_loader() -> None:
    path = ROOT / "metrics" / "by_metric" / "nci60_split_coverage_audit.json"
    audit = json.loads(path.read_text())
    assert audit["as_is_scaffold_70_10_10"]["rows"] == {
        "train": 7320,
        "val": 160,
        "test": 343,
    }
    applied = audit["applied_coverage_split"]
    assert applied["rows"] == {"train": 5371, "val": 920, "test": 1532}
    assert applied["scaffolds_spanning_splits"] == 0
