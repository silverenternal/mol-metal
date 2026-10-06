"""Wf-C1 — the synthetic-row routing guard must be GENERAL, not Rh-specific.

Wf-A3 fixed a real bug in
``molmetal/scripts/train_metal_lomo_wasserstein.py``: the synthetic Rh
cohort is tagged ``Cell_line == "SYNTHETIC_RH"`` but carries
``Metal == "Rh"``, so a naive ``metals == held_out_metal`` test mask swept
all ~500 fabricated rows into the Rh TEST fold.  Wf-A3 fixed it by testing
the exact string ``"SYNTHETIC_RH"``.

That fix was fragile: a second synthetic cohort tagged ``SYNTHETIC_Os`` or
``SYNTHETIC_Pt`` would have re-introduced the identical bug with no test
failing.  Wf-C1 replaces the equality test with a ``SYNTHETIC*`` prefix
predicate.  These tests pin that generalisation shut.

Covers:

* ``synthetic_row_mask`` recognises every ``SYNTHETIC_*`` tag (and is
  case/whitespace tolerant), and returns all-False for real cell lines or a
  dataframe with no ``Cell_line`` column.
* ``append_synthetic_rh_to_dataset`` tags rows with the canonical
  ``synthetic_cell_line_tag("Rh")`` value, so writer and reader cannot drift.
* The LOMO splitter routes synthetic rows to TRAIN for *any* synthetic
  cohort / held-out metal combination — the bug cannot return silently.
* Val is drawn only from real rows.
* A real cell line that merely contains the substring ``SYNTHETIC`` is not
  misrouted (the predicate is a prefix test, not a substring test).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from molmetal.data.cytotox import MetalCytotoxDataset
from molmetal.scripts.train_metal_lomo_wasserstein import (
    SYNTHETIC_CELL_LINE_PREFIX,
    append_synthetic_rh_to_dataset,
    synthetic_cell_line_tag,
    synthetic_row_mask,
    WassersteinLOMOSplitter,
)

_REAL_CELL_LINES = ["A549", "HeLa", "MCF-7", "HEK293", "NCI-H460"]


def _make_ds(rows, tmp_path: Path) -> MetalCytotoxDataset:
    """Build a minimal dataset from (smiles, metal, cell_line, active) rows."""
    df = pd.DataFrame(
        {
            "SMILES_Ligands": [r[0] for r in rows],
            "Metal": [r[1] for r in rows],
            "Cell_line": [r[2] for r in rows],
            "active": [r[3] for r in rows],
            "Year": [2020] * len(rows),
        }
    )
    return MetalCytotoxDataset(df=df, cache_dir=tmp_path)


def _fake_cohort(metal: str, n: int, tag: str, pos_rate: float = 0.5, seed: int = 0):
    """n synthetic rows for ``metal``, tagged ``tag``."""
    rng = np.random.default_rng(seed)
    smis = [f"C{metal}{'C' * 3}{i % 10}O" for i in range(n)]
    labels = (rng.random(n) < pos_rate).astype(int)
    return [(s, metal, tag, int(a)) for s, a in zip(smis, labels)]


# ---------------------------------------------------------------------------
# synthetic_row_mask
# ---------------------------------------------------------------------------


def test_prefix_constant_is_bare_synthetic():
    """The prefix must be the bare word so any element suffix matches."""
    assert SYNTHETIC_CELL_LINE_PREFIX == "SYNTHETIC"


def test_mask_detects_arbitrary_synthetic_tags():
    df = pd.DataFrame(
        {
            "Cell_line": [
                "A549",
                "SYNTHETIC_RH",
                "SYNTHETIC_Os",
                "SYNTHETIC_Pt",
                "SYNTHETIC_FOO",
                "SYNTHETIC",
                "MCF-7",
            ]
        }
    )
    mask = synthetic_row_mask(df)
    assert mask.tolist() == [False, True, True, True, True, True, False]


def test_mask_is_case_and_whitespace_tolerant():
    df = pd.DataFrame({"Cell_line": [" synthetic_rh ", "Synthetic_Os", "A549"]})
    assert synthetic_row_mask(df).tolist() == [True, True, False]


def test_mask_is_prefix_not_substring():
    """A real cell line that merely CONTAINS SYNTHETIC is not synthetic."""
    df = pd.DataFrame({"Cell_line": ["HeLa-SYNTHETIC-DERIVED", "MCF7 SYNTHETIC"]})
    assert synthetic_row_mask(df).tolist() == [False, False]


def test_mask_all_false_without_cell_line_column():
    df = pd.DataFrame({"Metal": ["Rh"] * 3})
    assert synthetic_row_mask(df).tolist() == [False, False, False]


def test_mask_all_false_on_pure_real_data():
    df = pd.DataFrame({"Cell_line": _REAL_CELL_LINES * 4})
    assert not synthetic_row_mask(df).any()


def test_mask_length_matches_dataframe():
    df = pd.DataFrame({"Cell_line": ["A549", "SYNTHETIC_RH"]})
    assert len(synthetic_row_mask(df)) == len(df)


# ---------------------------------------------------------------------------
# writer / reader cannot drift
# ---------------------------------------------------------------------------


def test_append_tags_rows_with_canonical_tag(tmp_path):
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "source": "unit-test",
                "per_compound": [
                    {"smiles": s, "catherine_6of6_pass": True, "is_active": False}
                    for s in ("CCO", "CCCO", "CC(C)O", "CCN", "CCCl", "CC#N")
                ],
            }
        ),
        encoding="utf-8",
    )
    base = _make_ds(
        [(f"C{'C' * i}O", "Ru", "A549", 0) for i in range(5)], tmp_path
    )
    ds, n_appended = append_synthetic_rh_to_dataset(
        base, cohort, n_target=10, active_cohort_json=None
    )
    assert n_appended > 0
    mask = synthetic_row_mask(ds.df)
    assert int(mask.sum()) == n_appended
    assert set(ds.df["Cell_line"].astype(str)[mask]) == {"SYNTHETIC_RH"}
    assert synthetic_cell_line_tag("Rh") == "SYNTHETIC_RH"


# ---------------------------------------------------------------------------
# the bug cannot return
# ---------------------------------------------------------------------------


def test_splitter_routes_both_synthetic_tags_to_train(tmp_path):
    """Rh held out; synthetic cohorts tagged SYNTHETIC_RH *and* SYNTHETIC_FOO.

    Wf-A3's hard-coded equality only caught the first.  Both must be excluded
    from test and present in train.
    """
    rh_real = [("C" + "C" * i + "O", "Rh", "A549", i % 2) for i in range(50)]
    rh_synth_a = _fake_cohort("Rh", 100, "SYNTHETIC_RH", pos_rate=0.5, seed=1)
    rh_synth_b = _fake_cohort("Rh", 40, "SYNTHETIC_FOO", pos_rate=0.5, seed=2)
    ru_real = [("N" + "C" * i + "O", "Ru", "HeLa", i % 2) for i in range(100)]

    ds = _make_ds(rh_real + rh_synth_a + rh_synth_b + ru_real, tmp_path)
    split = WassersteinLOMOSplitter(held_out_metal="Rh", val_fraction=0.10, seed=42)(
        ds
    )

    # test == the 50 real Rh rows only
    assert len(split.test_idx) == 50
    test_metal = ds.df["Metal"].astype(str).to_numpy()[split.test_idx]
    assert set(test_metal) == {"Rh"}
    assert not synthetic_row_mask(ds.df)[split.test_idx].any()

    # val > 0, and every val row is a real (non-synthetic) row
    assert len(split.val_idx) > 0
    assert not synthetic_row_mask(ds.df)[split.val_idx].any()

    # train carries ALL 140 synthetic rows (100 + 40), unheld-out
    train_synth = synthetic_row_mask(ds.df)[split.train_idx]
    assert int(train_synth.sum()) == 140

    # partitions are disjoint and cover the dataset
    allidx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert len(np.unique(allidx)) == len(ds)
    assert len(allidx) == len(ds)


def test_splitter_routes_synthetic_os_when_os_is_held_out(tmp_path):
    """The regression is symmetric: a future SYNTHETIC_Os cohort must behave
    the same way when Os (not Rh) is the held-out metal."""
    rh_real = [("C" + "C" * i + "O", "Rh", "A549", i % 2) for i in range(80)]
    os_real = [("O" + "C" * i + "N", "Os", "A549", i % 2) for i in range(30)]
    os_synth = _fake_cohort("Os", 120, "SYNTHETIC_Os", pos_rate=0.5, seed=3)

    ds = _make_ds(rh_real + os_real + os_synth, tmp_path)
    split = WassersteinLOMOSplitter(held_out_metal="Os", val_fraction=0.10, seed=7)(
        ds
    )

    assert len(split.test_idx) == 30
    assert not synthetic_row_mask(ds.df)[split.test_idx].any()
    assert int(synthetic_row_mask(ds.df)[split.train_idx].sum()) == 120
    assert len(split.val_idx) > 0
    assert not synthetic_row_mask(ds.df)[split.val_idx].any()


def test_splitter_synthetic_mask_is_all_true_for_synthetic_only_frame(tmp_path):
    """Degenerate case: every row of the held-out metal is fabricated.

    Then the test fold is empty — the guard must not silently re-inject the
    synthetic rows into it to make it non-empty.
    """
    rows = _fake_cohort("Rh", 30, "SYNTHETIC_RH", pos_rate=0.5, seed=4)
    rows += [("Z" + "C" * i, "Ru", "HeLa", 0) for i in range(30)]
    ds = _make_ds(rows, tmp_path)
    split = WassersteinLOMOSplitter(held_out_metal="Rh", val_fraction=0.10, seed=1)(ds)
    assert len(split.test_idx) == 0


def test_splitter_without_synthetic_rows_unchanged(tmp_path):
    """Regression guard: with no synthetic cohort the split is the plain LOMO
    split (all held-out-metal rows to test)."""
    rows = [("C" + "C" * i + "O", "Rh", "A549", i % 2) for i in range(50)]
    rows += [("N" + "C" * i + "O", "Ru", "HeLa", i % 2) for i in range(100)]
    ds = _make_ds(rows, tmp_path)
    split = WassersteinLOMOSplitter(held_out_metal="Rh", val_fraction=0.10, seed=42)(ds)
    assert len(split.test_idx) == 50
    assert set(ds.df["Metal"].astype(str).to_numpy()[split.test_idx]) == {"Rh"}
    assert len(split.val_idx) > 0
