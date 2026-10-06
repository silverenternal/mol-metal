"""E2 fix — the Rh val fold must come from the held-out metal, not from the
other metals.

The structural bug this pins shut: with Rh held out, the legacy
``WassersteinLOMOSplitter`` drew ``val`` from ``train_pool`` (the OTHER
metals only) and put all 108 real Rh rows in ``test``.  So val_auc=0.785
was an in-distribution Ru/Ir/Re/Os number while test_auc=0.386 was the true
LOMO transfer number.  The two metrics measured DIFFERENT distributions,
which made ``--model-select best_val`` select on the wrong signal.

With ``rh_loso_validation=True`` the 108 real Rh rows are halved: one half
val, one half test.  Both folds are then drawn from the target domain, so
val_auc is an honest early-stopping proxy for test_auc.

Scope is deliberately Rh-only ("fix only Rh"): for any other held-out metal
the flag is a no-op, and the default (``False``) leaves every existing
checkpoint's split byte-identical.

Covers:

* val + test partition the target metal's real rows, disjointly.
* Every val row is Rh and non-synthetic.
* The default is unchanged (legacy val has 0 Rh rows).
* The 50/50 split is stratified on the activity label.
* A non-Rh held-out metal is unaffected by the flag.
* A real 2-epoch CPU training run honours the flag end-to-end (AGENTS.md
  §5.7 trainer-flag wire-in smoke).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from molmetal.data.cytotox import MetalCytotoxDataset
from molmetal.scripts.train_metal_lomo_wasserstein import (
    SYNTHETIC_CELL_LINE_PREFIX,
    WassersteinLOMOSplitter,
)

CSV = PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv"
N_RH_REAL = 108


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def five_metal_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    assert len(df) == 5460, f"expected 5460 rows, got {len(df)}"
    return df


@pytest.fixture(scope="module")
def rh_with_synth_ds(tmp_path_factory, five_metal_df) -> MetalCytotoxDataset:
    """The real 5-metal CSV plus a synthetic Rh cohort appended after it.

    The synthetic rows carry ``Metal == "Rh"`` (that is the point of the
    oversample) and are tagged ``SYNTHETIC_RH``, so a splitter that forgets
    the synthetic mask will sweep them into the folds.
    """
    synth = five_metal_df.head(40).copy()
    synth["Metal"] = "Rh"
    synth["Cell_line"] = f"{SYNTHETIC_CELL_LINE_PREFIX}Rh"
    synth["active"] = 0
    df = pd.concat([five_metal_df, synth], ignore_index=True)
    return MetalCytotoxDataset(df=df, cache_dir=tmp_path_factory.mktemp("e2"))


def _split(ds: MetalCytotoxDataset, *, rh_loso: bool, metal: str = "Rh", seed: int = 0):
    splitter = WassersteinLOMOSplitter(
        held_out_metal=metal, rh_loso_validation=rh_loso, seed=seed
    )
    return splitter(ds)


def _synth_mask(ds: MetalCytotoxDataset) -> np.ndarray:
    cell = ds.df["Cell_line"].astype(str).str.strip().str.upper()
    return cell.str.startswith(SYNTHETIC_CELL_LINE_PREFIX).to_numpy()


# ---------------------------------------------------------------------------
# 1. val and test both come from the target metal's real rows
# ---------------------------------------------------------------------------
def test_rh_loso_val_test_both_from_target_real(rh_with_synth_ds):
    ds = rh_with_synth_ds
    sp = _split(ds, rh_loso=True)

    assert len(sp.val_idx) > 0, "val fold is empty"
    assert len(sp.test_idx) > 0, "test fold is empty"

    # The 108 real Rh rows are exactly partitioned between val and test.
    assert len(sp.val_idx) + len(sp.test_idx) == N_RH_REAL, (
        f"val+test should cover all {N_RH_REAL} real Rh rows, got "
        f"{len(sp.val_idx)}+{len(sp.test_idx)}"
    )

    # ...and disjointly.
    assert set(sp.val_idx) & set(sp.test_idx) == set(), "val/test overlap"

    # Every val row is Rh and real.
    metals = ds.df["Metal"].astype(str).to_numpy()
    synth = _synth_mask(ds)
    assert set(metals[sp.val_idx]) == {"Rh"}, "val fold contains a non-Rh row"
    assert not synth[sp.val_idx].any(), "val fold contains a synthetic row"

    # The train fold must not contain any real Rh row — that is what makes
    # the test fold a genuine leave-one-metal-out fold.
    assert not (metals[sp.train_idx] == "Rh")[~synth[sp.train_idx]].any(), (
        "a real Rh row leaked into train"
    )


def test_rh_loso_deterministic_across_seeds(rh_with_synth_ds):
    """Same seed -> same folds; different seed -> different folds."""
    a = _split(rh_with_synth_ds, rh_loso=True, seed=0)
    b = _split(rh_with_synth_ds, rh_loso=True, seed=0)
    c = _split(rh_with_synth_ds, rh_loso=True, seed=7)
    assert np.array_equal(a.val_idx, b.val_idx)
    assert np.array_equal(a.test_idx, b.test_idx)
    assert not np.array_equal(a.val_idx, c.val_idx), (
        "seed 7 produced the same val fold as seed 0 — the seed is ignored"
    )


# ---------------------------------------------------------------------------
# 2. Default off is byte-identical to the legacy behaviour
# ---------------------------------------------------------------------------
def test_rh_loso_default_off_unchanged(rh_with_synth_ds):
    """The default must not perturb any existing checkpoint's split."""
    ds = rh_with_synth_ds
    metals = ds.df["Metal"].astype(str).to_numpy()
    synth = _synth_mask(ds)

    sp = _split(ds, rh_loso=False)
    assert not (metals[sp.val_idx] == "Rh").any(), (
        "legacy val fold unexpectedly contains Rh rows"
    )
    # The legacy test fold is ALL real Rh rows.
    assert len(sp.test_idx) == N_RH_REAL
    assert set(metals[sp.test_idx]) == {"Rh"}
    assert not synth[sp.test_idx].any()

    # And the dataclass default really is False.
    assert WassersteinLOMOSplitter(held_out_metal="Rh").rh_loso_validation is False


# ---------------------------------------------------------------------------
# 3. The 50/50 split is stratified on the activity label
# ---------------------------------------------------------------------------
def test_rh_loso_stratify_works(rh_with_synth_ds):
    ds = rh_with_synth_ds
    labels = ds.df["active"].to_numpy().astype(int)

    sp = _split(ds, rh_loso=True)
    target_rows = np.concatenate([sp.val_idx, sp.test_idx])
    overall_rate = labels[target_rows].mean()
    val_rate = labels[sp.val_idx].mean()
    test_rate = labels[sp.test_idx].mean()

    assert abs(val_rate - overall_rate) <= 0.10, (
        f"val pos_rate {val_rate:.3f} drifted from the target rate "
        f"{overall_rate:.3f} — stratify did not take"
    )
    assert abs(test_rate - overall_rate) <= 0.10, (
        f"test pos_rate {test_rate:.3f} drifted from the target rate "
        f"{overall_rate:.3f}"
    )
    assert abs(val_rate - test_rate) <= 0.10, (
        f"val pos_rate {val_rate:.3f} vs test {test_rate:.3f} — >0.10 apart"
    )


# ---------------------------------------------------------------------------
# 4. A non-Rh held-out metal is untouched by the flag
# ---------------------------------------------------------------------------
def test_rh_loso_non_rh_metal_unchanged(rh_with_synth_ds):
    """Scope is Rh-only: for e.g. Os the flag must be a strict no-op."""
    ds = rh_with_synth_ds
    off = _split(ds, rh_loso=False, metal="Os")
    on = _split(ds, rh_loso=True, metal="Os")
    assert np.array_equal(off.train_idx, on.train_idx)
    assert np.array_equal(off.val_idx, on.val_idx)
    assert np.array_equal(off.test_idx, on.test_idx)

    # Ir, Re, Ru too — every non-Rh fold is unaffected.
    for metal in ("Ir", "Re", "Ru"):
        a = _split(ds, rh_loso=False, metal=metal)
        b = _split(ds, rh_loso=True, metal=metal)
        assert np.array_equal(a.val_idx, b.val_idx), metal
        assert np.array_equal(a.test_idx, b.test_idx), metal


# ---------------------------------------------------------------------------
# 5. AGENTS.md §5.7 — the flag is not FLAG_ONLY: a real 2-epoch CPU run
#    must carry Rh rows in the val fold.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_rh_loso_smoke_train_2_epoch_cpu(tmp_path):
    out_metric = tmp_path / "metrics.json"
    out_dir = tmp_path / "out"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py"),
            "--metals", "Rh",
            "--seeds", "0",
            "--epochs", "2",
            "--rh-loso-validation",
            "--device", "cpu",
            "--rh-oversample", "20",
            "--wasserstein-weight", "0",
            "--out-metric", str(out_metric),
            "--out-dir", str(out_dir),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    # 0 = PASS, 2 = BELOW_GATE, 3 = guard error. A crash is anything else.
    assert proc.returncode in (0, 2, 3), (
        f"unexpected exit {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr ---\n{proc.stderr[-3000:]}"
    )

    seed_json = out_dir / "Rh" / "per_seed_metrics_seed0.json"
    assert seed_json.exists(), (
        f"no per-seed JSON written; stderr tail:\n{proc.stderr[-2000:]}"
    )
    payload = json.loads(seed_json.read_text())
    assert "error" not in payload, f"fold errored: {payload.get('error')}"

    # The wire-in proof: val + test must together cover all 108 real Rh rows.
    assert payload["n_val"] > 0, "val fold is empty in the real run"
    assert payload["n_val"] + payload["n_test"] == N_RH_REAL, (
        f"n_val={payload['n_val']} n_test={payload['n_test']} — expected "
        f"them to sum to {N_RH_REAL} real Rh rows"
    )
    assert payload["rh_loso_validation"] is not None

    if out_metric.exists():
        metric_payload = json.loads(out_metric.read_text())
        assert metric_payload.get("rh_loso_validation") is True
