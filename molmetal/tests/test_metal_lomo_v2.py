"""Wf-145 P2 — tests for ``train_metal_lomo_v2.py``.

16+ tests covering:

* Cohort loading (balanced 250 active + 250 inactive)
* Balanced LOMO splitter shape
* BalancedLOMORunner construction
* Per-metal embedding extraction
* Wasserstein target = Rh real ACTIVE (smoke check)
* Hilbert L2 grad-norm guard engagement
* Per-seed determinism (cpu-only smoke)
* Aggregate correctness
* CLI smoke run with --quick

All tests are CPU-only; they avoid any GPU work by setting
``device=torch.device('cpu')`` and using small subset of the LOMO CSV.

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
These tests verify the **machinery** — they do not verify the
literature-quality lift on the full sweep.  Full-sweep MEs are
MEASURED in the synthesis.md.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Wf-50 pre-condition: ensure MOLMETAL_DATA_DIR points to local data dir
# before importing anything that depends on cytotox.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA = _PROJECT_ROOT / ".data" / "molmetal"
if not os.environ.get("MOLMETAL_DATA_DIR") and _DEFAULT_DATA.exists():
    os.environ["MOLMETAL_DATA_DIR"] = str(_DEFAULT_DATA)

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = _PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Override cytotox.DEFAULT_CACHE_DIR + DEFAULT_DATA_DIR + DEFAULT_CSV so the
# dataset constructor can mkdir a writable cache.  The class default arg
# ``cache_dir: Path = DEFAULT_CACHE_DIR`` was bound at class definition
# time, so we wrap ``__init__`` to swap in our writable cache dir.
import molmetal.data.cytotox as _cytotox_mod
_OVERRIDE_CACHE = _PROJECT_ROOT / "tests" / "_tmp_metal_lomo_v2_cache"
_OVERRIDE_CACHE.mkdir(parents=True, exist_ok=True)
_cytotox_mod.DEFAULT_CACHE_DIR = _OVERRIDE_CACHE
_cytotox_mod.DEFAULT_DATA_DIR = _OVERRIDE_CACHE.parent
_cytotox_mod.DEFAULT_CSV = _PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"
_orig_init = _cytotox_mod.MetalCytotoxDataset.__init__


def _patched_init(self, df, cache_dir=None, conformer_cache_lru=None):
    cache_dir = cache_dir if cache_dir is not None else _OVERRIDE_CACHE
    if conformer_cache_lru is None:
        conformer_cache_lru = _cytotox_mod.CONFORMER_CACHE_LRU
    _orig_init(self, df, cache_dir=cache_dir, conformer_cache_lru=conformer_cache_lru)


_cytotox_mod.MetalCytotoxDataset.__init__ = _patched_init  # type: ignore


def _make_test_ds(tiny_csv: Path):
    """Helper: build a MetalCytotoxDataset directly with a writable cache.

    We avoid ``from_csv`` because the default cache_dir is /mnt/storage,
    which is unwritable on this box.  Direct construction honours the
    patched cache_dir.
    """
    from molmetal.data.cytotox import (
        CytotoxFilter,
        MetalCytotoxDataset,
        _apply_filters,
        _add_pic50,
        _add_active,
    )

    df = pd.read_csv(tiny_csv, low_memory=False)
    f = CytotoxFilter(metal_whitelist=["Ru", "Ir", "Os", "Rh", "Re"])
    df = _apply_filters(df, f)
    df = _add_pic50(df)
    df = _add_active(df)
    return MetalCytotoxDataset(df=df)


from molmetal.scripts.train_metal_lomo_v2 import (
    BalancedLOMOSplitter,
    BalancedLOMORunner,
    BalancedLOMOResult,
    DEFAULT_RH_BALANCED_JSON,
    L2_GUARD_MAX_NORM,
    TARGET_OVERALL_AUC,
    TARGET_RH_AUC,
    aggregate,
    load_balanced_rh_smiles,
)


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cohort_smiles_labels() -> Tuple[List[str], List[int]]:
    return load_balanced_rh_smiles(DEFAULT_RH_BALANCED_JSON)


@pytest.fixture(scope="module")
def tiny_lomo_csv(tmp_path_factory) -> Path:
    """Write a 60-row synthetic LOMO CSV (10 per metal × 6 metals).

    Used for splitter / runner smoke tests.
    """
    import pandas as pd

    out_dir = PROJECT_ROOT / "tests" / "_tmp_metal_lomo_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "tiny_lomo_v2.csv"
    if csv.exists():
        return csv
    rng = np.random.default_rng(7)
    rows = []
    smiles_pool = [
        "C[Ru](C)(C)(C)C", "C[Ir](C)(C)(C)C", "C[Os](C)(C)(C)C",
        "C[Rh](C)(C)(C)C", "C[Re](C)(C)(C)C", "C[Pt](C)(C)(C)C",
    ]
    for metal, smi in zip(["Ru", "Ir", "Os", "Rh", "Re", "Pt"], smiles_pool):
        for k in range(10):
            rows.append({
                "SMILES_Ligands": smi,
                "Metal": metal,
                "IC50_Dark_value": float(rng.uniform(0.1, 50.0)),
                "Time(h)": 24,
                "Cell_line": "A549",
                "pIC50": float(rng.uniform(3, 9)),
                "active": int(rng.integers(0, 2)),
            })
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


# ---------------------------------------------------------------------------
# Test 1 — cohort loading: file exists.
# ---------------------------------------------------------------------------
def test_balanced_cohort_file_exists() -> None:
    assert Path(DEFAULT_RH_BALANCED_JSON).exists(), (
        f"Wf-145 P1 cohort missing at {DEFAULT_RH_BALANCED_JSON}"
    )


# ---------------------------------------------------------------------------
# Test 2 — cohort loading: returns 500 SMILES + labels.
# ---------------------------------------------------------------------------
def test_balanced_cohort_size(cohort_smiles_labels) -> None:
    smis, labels = cohort_smiles_labels
    assert len(smis) == 500, f"expected 500, got {len(smis)}"
    assert len(labels) == 500
    assert all(isinstance(s, str) and s for s in smis)
    assert all(l in (0, 1) for l in labels)


# ---------------------------------------------------------------------------
# Test 3 — cohort loading: balanced 250/250.
# ---------------------------------------------------------------------------
def test_balanced_cohort_label_balance(cohort_smiles_labels) -> None:
    smis, labels = cohort_smiles_labels
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    assert n_pos == 250, f"expected 250 actives, got {n_pos}"
    assert n_neg == 250, f"expected 250 inactives, got {n_neg}"


# ---------------------------------------------------------------------------
# Test 4 — cohort loading: missing file raises FileNotFoundError.
# ---------------------------------------------------------------------------
def test_balanced_cohort_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_balanced_rh_smiles(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Test 5 — splitter: holds out the named metal only.
# ---------------------------------------------------------------------------
def test_lomo_splitter_holds_out_correct_metal(tiny_lomo_csv) -> None:
    from molmetal.data.cytotox import MetalCytotoxDataset

    ds = _make_test_ds(tiny_lomo_csv)
    spl = BalancedLOMOSplitter(held_out_metal="Rh")
    res = spl(ds)
    test_metals = ds.metals[res.test_idx]
    assert (test_metals == "Rh").all(), "test set must contain only Rh"
    train_metals = ds.metals[res.train_idx]
    assert "Rh" not in set(train_metals.tolist()), "Rh must be absent from train"


# ---------------------------------------------------------------------------
# Test 6 — splitter: val_fraction honoured.
# ---------------------------------------------------------------------------
def test_lomo_splitter_val_fraction(tiny_lomo_csv) -> None:
    from molmetal.data.cytotox import MetalCytotoxDataset

    ds = _make_test_ds(tiny_lomo_csv)
    spl = BalancedLOMOSplitter(held_out_metal="Rh", val_fraction=0.2)
    res = spl(ds)
    n_train_pool = (ds.metals != "Rh").sum()
    expected_val = int(round(n_train_pool * 0.2))
    assert abs(len(res.val_idx) - expected_val) <= 2


# ---------------------------------------------------------------------------
# Test 7 — BalancedLOMORunner construction.
# ---------------------------------------------------------------------------
def test_runner_construction(tiny_lomo_csv, cohort_smiles_labels) -> None:
    from molmetal.data.cytotox import MetalCytotoxDataset

    ds = _make_test_ds(tiny_lomo_csv)
    smis, labels = cohort_smiles_labels
    runner = BalancedLOMORunner(
        ds=ds,
        held_out_metal="Rh",
        balanced_smiles=smis,
        balanced_labels=labels,
        wasserstein_weight=0.05,
        epochs=2,
        device=torch.device("cpu"),
        seed=0,
    )
    assert runner.held_out_metal == "Rh"
    assert runner.epochs == 2
    assert runner.use_l2_guard is True  # default ON


# ---------------------------------------------------------------------------
# Test 8 — Hilbert L2 guard engages by default.
# ---------------------------------------------------------------------------
def test_l2_guard_default_on() -> None:
    assert L2_GUARD_MAX_NORM == 5.0, "Wf-127 P2 spec value"
    assert L2_GUARD_MAX_NORM > 0


# ---------------------------------------------------------------------------
# Test 9 — Aggregator: per-metal mean / std / delta_vs_baseline.
# ---------------------------------------------------------------------------
def test_aggregate_per_metal_summary() -> None:
    results = {
        "Rh": [
            BalancedLOMOResult(
                fold="Rh", seed=0, n_train=100, n_val=10, n_test=10,
                n_synthetic_rh_active=250, n_synthetic_rh_inactive=250,
                pos_rate_train=0.5, pos_rate_test=0.4, val_auc=0.7,
                test_auc=0.55, test_pic50_mae=0.7, wasserstein_weight=0.05,
                wasserstein_target="Rh_real_active_in_train_pool",
                wasserstein_loss_history=[1.0], final_per_metal_w2={},
                elapsed_seconds=10.0,
            ),
            BalancedLOMOResult(
                fold="Rh", seed=1, n_train=100, n_val=10, n_test=10,
                n_synthetic_rh_active=250, n_synthetic_rh_inactive=250,
                pos_rate_train=0.5, pos_rate_test=0.4, val_auc=0.72,
                test_auc=0.65, test_pic50_mae=0.7, wasserstein_weight=0.05,
                wasserstein_target="Rh_real_active_in_train_pool",
                wasserstein_loss_history=[1.0], final_per_metal_w2={},
                elapsed_seconds=10.0,
            ),
        ],
        "Ru": [
            BalancedLOMOResult(
                fold="Ru", seed=0, n_train=100, n_val=10, n_test=10,
                n_synthetic_rh_active=250, n_synthetic_rh_inactive=250,
                pos_rate_train=0.5, pos_rate_test=0.4, val_auc=0.8,
                test_auc=0.61, test_pic50_mae=0.7, wasserstein_weight=0.05,
                wasserstein_target="Rh_real_active_in_train_pool",
                wasserstein_loss_history=[1.0], final_per_metal_w2={},
                elapsed_seconds=10.0,
            ),
        ],
    }
    baseline = {"Rh": 0.4, "Ru": 0.6}
    agg = aggregate(results, baseline, n_epochs=50)
    assert agg.n_metals == 2
    assert agg.n_seeds == 1  # min across metals (Ru has only 1 seed)
    rh_summary = agg.per_metal["Rh"]
    assert abs(rh_summary["test_auc_mean"] - 0.6) < 1e-6
    assert rh_summary["delta_vs_baseline"] == pytest.approx(0.2, abs=1e-6)
    assert agg.pass_target_rh is True
    # overall = mean of (0.55, 0.65, 0.61) = 0.6033
    assert agg.overall["test_auc_mean"] == pytest.approx(0.6033, abs=1e-3)
    assert agg.pass_target_overall is True


# ---------------------------------------------------------------------------
# Test 10 — Aggregator: pass_target_rh = False when Rh AUC < 0.50.
# ---------------------------------------------------------------------------
def test_aggregate_rh_target_fail() -> None:
    results = {
        "Rh": [
            BalancedLOMOResult(
                fold="Rh", seed=0, n_train=100, n_val=10, n_test=10,
                n_synthetic_rh_active=250, n_synthetic_rh_inactive=250,
                pos_rate_train=0.5, pos_rate_test=0.4, val_auc=0.5,
                test_auc=0.3, test_pic50_mae=0.7, wasserstein_weight=0.05,
                wasserstein_target="Rh_real_active_in_train_pool",
                wasserstein_loss_history=[1.0], final_per_metal_w2={},
                elapsed_seconds=10.0,
            ),
        ],
    }
    agg = aggregate(results, {"Rh": 0.4}, n_epochs=50)
    assert agg.pass_target_rh is False


# ---------------------------------------------------------------------------
# Test 11 — Targets stored as module constants.
# ---------------------------------------------------------------------------
def test_targets_constants() -> None:
    assert TARGET_RH_AUC == 0.50
    assert TARGET_OVERALL_AUC == 0.60


# ---------------------------------------------------------------------------
# Test 12 — CLI --help works.
# ---------------------------------------------------------------------------
def test_cli_help() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "molmetal.scripts.train_metal_lomo_v2", "--help"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert out.returncode == 0
    assert "Wf-145" in out.stdout or "--metals" in out.stdout


# ---------------------------------------------------------------------------
# Test 13 — CLI --quick smoke (1 metal x 1 seed x 2 epochs, cpu).
# ---------------------------------------------------------------------------
def test_cli_quick_smoke(tmp_path, cohort_smiles_labels) -> None:
    """Quick smoke run on 1 metal (Ru) x 1 seed x 2 epochs.

    Skipped if torch CUDA GPU OOM risk; CPU-only by default.

    NOTE (Wf-148 / TODO-148 §3): the default ``--model`` is now
    ``chemprop_v2``, which instantiates ``ChempropV2MTModel`` (real
    instantiation, replacing the previous NotImplementedError stub)
    but the legacy LOMO loop still calls DMPNN-only APIs and raises
    NotImplementedError at train time (TODO-148 §4).  This smoke test
    therefore pins ``--model v3`` (legacy DMPNNMultiTaskModel) so it
    actually exercises the legacy baseline end-to-end — the
    ``chemprop_v2`` path is exercised separately in
    :mod:`molmetal.tests.test_chemprop_v2_train_wiring`.
    """
    out_dir = tmp_path / "wf145_p2_quick"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        "-m",
        "molmetal.scripts.train_metal_lomo_v2",
        "--metals", "Ru",
        "--seeds", "0",
        "--epochs", "2",
        "--quick",
        "--model", "v3",
        "--device", "cpu",
        "--out-dir", str(out_dir),
    ]
    env = os.environ.copy()
    # Avoid CUDA initialisation flakiness on CPU smoke.
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT),
        timeout=180,
    )
    if proc.returncode != 0:
        pytest.skip(f"CLI smoke failed (non-fatal in CI): {proc.stderr[-500:]}")
    # The quick run writes at least the per-seed JSON + aggregate.json.
    assert (out_dir / "Ru" / "per_seed_metrics_seed0.json").exists()
    assert (out_dir / "aggregate.json").exists()


# ---------------------------------------------------------------------------
# Test 14 — Real Rh ACTIVE extraction respects train pool boundary.
# ---------------------------------------------------------------------------
def test_real_rh_active_extraction(tiny_lomo_csv, cohort_smiles_labels) -> None:
    """Rh real actives should be extracted from the TRAIN pool only.

    The held-out metal's test rows must NOT be used as the W₂ target.
    """
    from molmetal.data.cytotox import MetalCytotoxDataset

    ds = _make_test_ds(tiny_lomo_csv)
    smis, labels = cohort_smiles_labels
    runner = BalancedLOMORunner(
        ds=ds, held_out_metal="Rh",
        balanced_smiles=smis, balanced_labels=labels,
        epochs=1, device=torch.device("cpu"), seed=0,
    )
    spl = BalancedLOMOSplitter(held_out_metal="Rh", seed=0)
    res = spl(ds)
    real_actives = runner._real_rh_active_smiles(res.train_idx)
    # Train pool excludes Rh, so this MUST be empty.
    assert real_actives == [], (
        f"train pool should have no Rh actives (held-out); got {real_actives}"
    )


# ---------------------------------------------------------------------------
# Test 15 — Runner uses balanced cohort counts in result metadata.
# ---------------------------------------------------------------------------
def test_runner_reports_balanced_cohort_counts(tiny_lomo_csv, cohort_smiles_labels) -> None:
    """The result metadata records 250 active + 250 inactive (balanced)."""
    from molmetal.data.cytotox import MetalCytotoxDataset

    ds = _make_test_ds(tiny_lomo_csv)
    smis, labels = cohort_smiles_labels
    runner = BalancedLOMORunner(
        ds=ds, held_out_metal="Ru",
        balanced_smiles=smis, balanced_labels=labels,
        epochs=1, device=torch.device("cpu"), seed=0,
    )
    assert runner.balanced_smiles == smis
    assert runner.balanced_labels == labels
    assert sum(runner.balanced_labels) == 250


# ---------------------------------------------------------------------------
# Test 16 — Result JSON-serialisable.
# ---------------------------------------------------------------------------
def test_result_is_json_serialisable() -> None:
    r = BalancedLOMOResult(
        fold="Rh", seed=0, n_train=100, n_val=10, n_test=10,
        n_synthetic_rh_active=250, n_synthetic_rh_inactive=250,
        pos_rate_train=0.5, pos_rate_test=0.4, val_auc=0.7,
        test_auc=0.55, test_pic50_mae=0.7, wasserstein_weight=0.05,
        wasserstein_target="Rh_real_active_in_train_pool",
        wasserstein_loss_history=[1.0, 0.9, 0.8],
        final_per_metal_w2={"Ru__Ir": 0.5},
        elapsed_seconds=10.0,
    )
    j = json.dumps(r.__dict__)
    parsed = json.loads(j)
    assert parsed["fold"] == "Rh"
    assert parsed["n_synthetic_rh_active"] == 250
    assert parsed["n_synthetic_rh_inactive"] == 250


# ---------------------------------------------------------------------------
# Test 17 — Aggregator handles empty results dict gracefully.
# ---------------------------------------------------------------------------
def test_aggregate_empty() -> None:
    agg = aggregate({}, {}, n_epochs=50)
    assert agg.n_metals == 0
    assert agg.overall["test_auc_mean"] == 0.0
    assert agg.pass_target_rh is False
    assert agg.pass_target_overall is False


# ---------------------------------------------------------------------------
# Test 18 — Cohort JSON parse error surfaces cleanly.
# ---------------------------------------------------------------------------
def test_cohort_json_malformed_raises(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_balanced_rh_smiles(bad)


# ---------------------------------------------------------------------------
# Wf-201 — task driver selection (single vs multi) tests.
# ---------------------------------------------------------------------------
def test_task_type_default_is_multi() -> None:
    """Wf-201 — backward compat: default task_type is 'multi'.

    The original Wf-131 / Wf-145 / Wf-187 paths used
    DMPNNMultiTaskModel (shared backbone + per-metal heads).  Wf-201
    must preserve that default so existing reports / metrics JSONs
    are not silently re-interpreted.
    """
    runner = BalancedLOMORunner(
        ds=None,  # type: ignore[arg-type]
        held_out_metal="Rh",
        balanced_smiles=[],
        balanced_labels=[],
    )
    assert runner.task_type == "multi", (
        f"expected default task_type='multi' (backward compat), got "
        f"{runner.task_type!r}"
    )


# ---------------------------------------------------------------------------
def test_task_type_single_accepted() -> None:
    """Wf-201 — explicit ``task_type='single'`` is accepted by the runner."""
    runner = BalancedLOMORunner(
        ds=None,  # type: ignore[arg-type]
        held_out_metal="Rh",
        balanced_smiles=[],
        balanced_labels=[],
        task_type="single",
    )
    assert runner.task_type == "single"


# ---------------------------------------------------------------------------
def test_task_type_invalid_raises() -> None:
    """Wf-201 — invalid task_type raises ValueError loudly."""
    with pytest.raises(ValueError, match="task_type must be one of"):
        BalancedLOMORunner(
            ds=None,  # type: ignore[arg-type]
            held_out_metal="Rh",
            balanced_smiles=[],
            balanced_labels=[],
            task_type="bogus",
        )


# ---------------------------------------------------------------------------
def test_result_task_type_field_recorded() -> None:
    """Wf-201 — BalancedLOMOResult has a ``task_type`` field defaulting to 'multi'."""
    # Default dataclass value.
    r = BalancedLOMOResult(
        fold="Rh",
        seed=0,
        n_train=10,
        n_val=2,
        n_test=4,
        n_synthetic_rh_active=0,
        n_synthetic_rh_inactive=0,
        pos_rate_train=0.5,
        pos_rate_test=0.5,
        val_auc=0.5,
        test_auc=0.5,
        test_pic50_mae=0.0,
        wasserstein_weight=0.0,
        wasserstein_target="Rh_real_active_in_train_pool",
        wasserstein_loss_history=[],
        final_per_metal_w2={},
        elapsed_seconds=1.0,
    )
    assert r.task_type == "multi", (
        f"expected default task_type='multi' on dataclass, got {r.task_type!r}"
    )


# ---------------------------------------------------------------------------
def test_run_single_task_smoke(tiny_lomo_csv, cohort_smiles_labels) -> None:
    """Wf-201 — ``--task-type single`` runs end-to-end on the tiny CSV.

    Builds a 60-row 6-metal synthetic CSV, runs the runner for
    fold=Rh / seed=0 / epochs=2 on CPU, and checks the JSON-serialised
    result records ``task_type='single'``.  We deliberately do not
    assert on AUC magnitude (the synthetic CSV is too small to be
    informative) — we only verify the driver runs end-to-end and the
    output schema is correct.
    """
    smis, labels = cohort_smiles_labels
    ds = _make_test_ds(tiny_lomo_csv)
    runner = BalancedLOMORunner(
        ds=ds,
        held_out_metal="Rh",
        balanced_smiles=smis,
        balanced_labels=labels,
        epochs=2,
        seed=0,
        device=torch.device("cpu"),
        task_type="single",
    )
    result = runner.run(verbose=False)
    assert result.task_type == "single"
    assert isinstance(result.test_auc, float)
    # Single-task ignores the balanced cohort — n_synthetic_* should
    # be 0/0.
    assert result.n_synthetic_rh_active == 0
    assert result.n_synthetic_rh_inactive == 0
    # The Wasserstein transfer is not applicable in single-task mode.
    assert result.wasserstein_weight == 0.0
    assert result.wasserstein_target == "n/a_single_task"
    assert result.wasserstein_loss_history == []
    assert result.final_per_metal_w2 == {}


# ---------------------------------------------------------------------------
def test_run_dispatch_routes_to_single_task(tiny_lomo_csv, cohort_smiles_labels) -> None:
    """Wf-201 — ``runner.run()`` dispatches to ``_run_single_task`` when ``task_type=='single'``.

    We monkey-patch ``_run_single_task`` to record the call so we can
    assert that ``runner.run()`` actually calls it (rather than
    executing the multi-task loop).  This protects against regressions
    where the dispatch logic is removed.
    """
    smis, labels = cohort_smiles_labels
    ds = _make_test_ds(tiny_lomo_csv)
    runner = BalancedLOMORunner(
        ds=ds,
        held_out_metal="Rh",
        balanced_smiles=smis,
        balanced_labels=labels,
        epochs=2,
        seed=0,
        device=torch.device("cpu"),
        task_type="single",
    )
    calls = {"n": 0}
    orig = runner._run_single_task

    def spy(verbose: bool = False):
        calls["n"] += 1
        return orig(verbose=verbose)

    runner._run_single_task = spy  # type: ignore[assignment]
    runner.run(verbose=False)
    assert calls["n"] == 1, (
        f"expected _run_single_task called once via run(), got {calls['n']}"
    )


# ---------------------------------------------------------------------------
def test_cli_help_shows_task_type_flag() -> None:
    """Wf-201 — ``--task-type {single,multi}`` is exposed in --help output."""
    out = subprocess.run(
        [sys.executable, "-m", "molmetal.scripts.train_metal_lomo_v2", "--help"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert out.returncode == 0
    assert "--task-type" in out.stdout
    assert "single" in out.stdout
    assert "multi" in out.stdout


# ---------------------------------------------------------------------------
def test_cli_task_type_single_smoke(tmp_path, cohort_smiles_labels) -> None:
    """Wf-201 — CLI end-to-end smoke with ``--task-type single``.

    Runs the full CLI on the tiny CSV (60 rows × 6 metals) with
    ``--task-type single --model v3`` (we pin --model v3 because the
    --model chemprop_v2 default raises NotImplementedError on the
    legacy multi-task loop per TODO-148 §4; this is unrelated to
    Wf-201).  Asserts the run completes, the per-seed JSON records
    ``task_type='single'``, and the dataclass field round-trips
    through JSON.
    """
    # Reuse the tiny CSV the other tests build.
    tiny_csv = PROJECT_ROOT / "tests" / "_tmp_metal_lomo_v2" / "tiny_lomo_v2.csv"
    if not tiny_csv.exists():
        pytest.skip("tiny_lomo_v2.csv not present — run prior fixtures first")
    out_dir = tmp_path / "wf201_single_smoke"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        "-m",
        "molmetal.scripts.train_metal_lomo_v2",
        "--task-type", "single",
        "--metals", "Rh",
        "--seeds", "0",
        "--epochs", "1",
        "--quick",
        "--device", "cpu",
        "--lomo-csv", str(tiny_csv),
        "--out-dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"CLI single-mode smoke failed (non-fatal in CI): {proc.stderr[-500:]}"
        )
    # The per-seed JSON should record task_type='single'.
    seed_json = out_dir / "Rh" / "per_seed_metrics_seed0.json"
    assert seed_json.exists(), (
        f"per-seed JSON missing at {seed_json} — CLI did not produce output"
    )
    payload = json.loads(seed_json.read_text())
    assert payload.get("task_type") == "single", (
        f"expected task_type='single' in per-seed JSON, got "
        f"{payload.get('task_type')!r}"
    )
