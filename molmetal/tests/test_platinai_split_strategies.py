"""Pin the two PlatinAI 4-cell split strategies apart.

Wf-A53 (2026-10-19) — ``load_platinai_4cell_split_table`` had exactly one
strategy, ``scaffold_70_10_10``, whose name promised a Bemis-Murcko scaffold
split but whose code assigned ``_stable_hash_int(smiles)`` to a 70/10/10
bucket.  MEASURED: 41.9% of test rows share a desalted stereo-blind scaffold
with train.

``bemis_murcko_70_10_10`` is the real thing, added as an opt-in so existing
checkpoints stay loadable.  These tests hold the two apart: the legacy
strategy must keep producing the documented 41.9% (if it silently becomes a
real scaffold split, every shipped Pt checkpoint is invalid), and the new one
must produce 0.0%.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLATINAI_CSV = PROJECT_ROOT / ".data" / "molmetal" / "platinai_4cell.csv"

pytestmark = pytest.mark.skipif(
    not PLATINAI_CSV.exists(), reason=f"missing cohort: {PLATINAI_CSV}"
)


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_dmpnn_split_under_test",
        PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


@pytest.fixture(scope="module")
def key_fn():
    from molmetal.data.splits import _bemis_murcko_key

    return lambda s: _bemis_murcko_key(str(s), include_chirality=False) or f"__empty__{s}"


@pytest.fixture(scope="module")
def legacy(trainer):
    return trainer.load_platinai_4cell_split_table(
        PLATINAI_CSV, split_strategy="scaffold_70_10_10"
    )


@pytest.fixture(scope="module")
def scaffold(trainer):
    return trainer.load_platinai_4cell_split_table(
        PLATINAI_CSV, split_strategy="bemis_murcko_70_10_10"
    )


def _overlap_pct(df, key_fn):
    train_keys = set(df.loc[df["split"] == "train", "smiles"].map(key_fn))
    test = df[df["split"] == "test"]
    if test.empty:
        return 0.0
    return 100.0 * test["smiles"].map(lambda s: key_fn(s) in train_keys).mean()


def test_legacy_strategy_keeps_its_measured_leak(legacy, key_fn):
    """The documented 41.9% is the legacy strategy's fingerprint.

    If this moves, either the data changed or the legacy path was silently
    turned into a real scaffold split — in which case every shipped Pt
    checkpoint was trained on a different cohort than the one now produced.
    """
    assert _overlap_pct(legacy, key_fn) == pytest.approx(41.9, abs=0.1)


def test_scaffold_strategy_has_zero_scaffold_overlap(scaffold, key_fn):
    assert _overlap_pct(scaffold, key_fn) == 0.0


def test_scaffold_strategy_is_deterministic(trainer, scaffold, key_fn):
    again = trainer.load_platinai_4cell_split_table(
        PLATINAI_CSV, split_strategy="bemis_murcko_70_10_10"
    )
    assert again["split"].tolist() == scaffold["split"].tolist()


def test_both_strategies_keep_every_row(legacy, scaffold):
    assert len(legacy) == len(scaffold)
    assert set(legacy["split"]) == {"train", "val", "test"}
    assert set(scaffold["split"]) == {"train", "val", "test"}


def test_scaffold_strategy_keeps_a_molecule_in_one_split(scaffold):
    """All four cell-line rows of a SMILES must follow the same assignment."""
    n_splits = scaffold.groupby("smiles")["split"].nunique()
    assert int(n_splits.max()) == 1, "a molecule straddles two splits"


def test_scaffold_id_names_a_scaffold_not_a_smiles_hash(scaffold, key_fn):
    """``scaffold_id`` must be a function of the scaffold key alone."""
    pairs = scaffold[["smiles", "scaffold_id"]].drop_duplicates()
    by_key = {}
    for smi, sid in pairs.itertuples(index=False):
        by_key.setdefault(key_fn(smi), set()).add(int(sid))
    assert all(len(ids) == 1 for ids in by_key.values()), (
        "two molecules with the same scaffold got different scaffold_id values"
    )


def test_unknown_strategy_names_both_options(trainer):
    with pytest.raises(ValueError) as excinfo:
        trainer.load_platinai_4cell_split_table(PLATINAI_CSV, split_strategy="nope")
    msg = str(excinfo.value)
    assert "scaffold_70_10_10" in msg
    assert "bemis_murcko_70_10_10" in msg
