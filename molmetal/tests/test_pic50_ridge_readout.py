"""Regression guard for the pIC50 ridge readout on frozen encoder embeddings.

Root cause (WF diagnostic 2026-10-08, molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py):
on the Ru cohort the multi-task D-MPNN's neural pIC50 head reaches test
pearson ~0.08-0.31, while a Ridge probe on the SAME encoder's pooled embedding
reaches ~0.38. The encoder carries more signal than the head extracts.

Tests 1-3 are CPU-only unit tests of the readout helpers.  Test 4 is the
AGENTS.md §5.7 end-to-end smoke: the ``--pic50-ridge-readout`` trainer flag
must actually reach the eval path (not be FLAG_ONLY), and must be silent when
off.  It runs a 1-epoch tiny-CPU trainer invocation and asserts the emitted
``RIDGE-READOUT`` line appears iff the flag is set.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from molmetal.training.pic50_ridge_readout import fit_ridge_readout

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
DATA_DIR = PROJECT_ROOT / ".data" / "molmetal"


def test_fit_ridge_readout_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 24)).astype(np.float32)
    w = rng.normal(size=24)
    y = X @ w + rng.normal(size=400) * 0.2
    m = fit_ridge_readout(X[:300], y[:300], X[300:], y[300:])
    assert m["pearson"] > 0.9, m
    assert m["n"] == 100


def test_fit_ridge_readout_handles_tiny_split():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1, 8)).astype(np.float32)
    m = fit_ridge_readout(X, np.array([1.0]), X, np.array([1.0]))
    assert np.isnan(m["pearson"])
    assert m["n"] == 1


def test_fit_ridge_readout_selects_from_alpha_grid():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 16)).astype(np.float32)
    y = X @ rng.normal(size=16) + rng.normal(size=300) * 0.5
    m = fit_ridge_readout(X[:200], y[:200], X[200:], y[200:], alphas=(0.1, 1.0, 10.0))
    assert "ridge_alpha" in m
    assert m["ridge_alpha"] in (0.1, 1.0, 10.0)
    assert "r2" in m


def test_fit_ridge_readout_alpha_not_selected_on_eval():
    """Guards the 2026-10-11 leak fix: alpha must come from the val fold.

    Property checked directly: relabelling the EVAL fold must not move the
    chosen alpha. Under the old select-on-eval loop the argmax over the grid
    was a function of exactly those labels, so at least one permutation below
    flips it.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(600, 8)).astype(np.float32)
    y = (X @ rng.normal(size=8) + rng.normal(size=600) * 1.5).astype(np.float32)
    Xtr, ytr = X[:400], y[:400]
    Xval, yval = X[400:500], y[400:500]
    Xte, yte = X[500:], y[500:]

    base = fit_ridge_readout(Xtr, ytr, Xte, yte, X_val=Xval, y_val=yval)
    assert base["ridge_alpha"] in (0.1, 1.0, 10.0, 100.0)
    assert base["n_val"] == 100 and base["n_train"] == 400

    for s in range(5):
        perm = np.random.default_rng(100 + s).permutation(len(yte))
        other = fit_ridge_readout(Xtr, ytr, Xte, yte[perm], X_val=Xval, y_val=yval)
        assert other["ridge_alpha"] == base["ridge_alpha"], (
            f"seed {s}: alpha moved {base['ridge_alpha']} -> "
            f"{other['ridge_alpha']} when only the eval labels changed"
        )


def test_fit_ridge_readout_internal_val_split_is_deterministic():
    """No-val callers must carve the same split on every call (reproducible)."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 6)).astype(np.float32)
    y = (X @ rng.normal(size=6)).astype(np.float32)
    a = fit_ridge_readout(X[:150], y[:150], X[150:], y[150:])
    b = fit_ridge_readout(X[:150], y[:150], X[150:], y[150:])
    assert a["ridge_alpha"] == b["ridge_alpha"]
    assert a["pearson"] == b["pearson"]
    assert a["n_val"] == 30 and a["n_train"] == 150


# ===========================================================================
# 4. AGENTS.md §5.7 — trainer flag wire-in smoke (end-to-end)
# ===========================================================================
def _run_trainer(tmp_path: Path, ridge_flag: str) -> str:
    env = dict(os.environ)
    env["MOLMETAL_DATA_DIR"] = str(DATA_DIR)
    proc = subprocess.run(
        [
            str(VENV_PY),
            str(TRAINER),
            "--metals", "Ru",
            "--seeds", "0",
            "--epochs", "1",
            "--max-joint-rows", "128",
            "--no-bf16", "--no-tf32", "--no-torch-compile",
            "--cpu-num-workers", "0",
            "--device", "cpu",
            ridge_flag,
            "--out-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout + proc.stderr


def test_pic50_ridge_readout_flag_reaches_eval_path(tmp_path: Path):
    """ON emits the RIDGE-READOUT line; OFF is silent (§5.7 gate)."""
    on = _run_trainer(tmp_path / "on", "--pic50-ridge-readout")
    assert "RIDGE-READOUT" in on, on[-2000:]
    off = _run_trainer(tmp_path / "off", "--no-pic50-ridge-readout")
    assert "RIDGE-READOUT" not in off, off[-2000:]


def test_pic50_ridge_readout_is_yaml_bindable_and_typed():
    """§5.11: the flag must be settable from `model:` in the launch YAML.

    Asserted BY TYPE, not by value: ``apply_yaml_overrides`` coerces through
    the action's ``type=`` and ``argparse.BooleanOptionalAction`` declares
    none, so a QUOTED "true" in the YAML survives as ``str 'true'`` and
    ``bool('false')``-style truthiness downstream would silently enable it.
    """
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    cfg = PROJECT_ROOT / "configs" / "dmpnn.yaml"
    args = parse_args(["--config", str(cfg)])
    assert isinstance(args.pic50_ridge_readout, bool), type(args.pic50_ridge_readout)
    # Default stays OFF: the report-only contract is unchanged by being bindable.
    assert args.pic50_ridge_readout is False
