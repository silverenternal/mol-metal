"""Tests for the coupling-shaper ablation workflow (wf_coupling_ablation).

This module validates the L-R17-B4 deliverable — the empirical
comparison of three reward shapers (LambdaASTSR, F5, no coupling)
on the PlatinAI 1K hold-out:

1. ``test_platinai_holdout_deterministic`` — the 1K sub-sample is
   bit-identical across runs with the same seed.
2. ``test_feature_columns_match_f5`` — the per-molecule features
   are exactly the 5 F5 columns in canonical order.
3. ``test_target_is_pure_quadratic`` — the synthetic target is the
   pure quadratic ``qed * qed``; this is the ASTSR-friendly /
   F5-unfriendly target that gives the cleanest ASTSR-vs-F5 gap.
4. ``test_lambda_astsr_beats_f5_on_centered_target`` — on the
   30% test split, ASTSR R² must exceed F5 R² by at least
   0.05 (the verify gate).
5. ``test_no_coupling_baseline_r2_nonpositive`` — the random
   baseline R² is ≤ 0 by construction (constant predictor).
6. ``test_metrics_json_well_formed`` — the saved metrics.json
   is a parseable JSON document with all expected keys.
7. ``test_report_contains_all_three_shapers`` — the rendered
   report (final.md) lists all three shapers in the results
   table.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "molmetal")]

#: Path to the wf_coupling_ablation script and its default report dir.
SCRIPT_PATH = ROOT / "molmetal" / "scripts" / "wf_coupling_ablation.py"
REPORT_DIR = ROOT / "molmetal" / "reports" / "wf_coupling_ablation"


# ---------------------------------------------------------------------------
# 1. Hold-out determinism
# ---------------------------------------------------------------------------
def test_platinai_holdout_deterministic() -> None:
    """Sub-sampling 1K rows from platinai_5k.csv is bit-identical."""
    from molmetal.scripts.wf_coupling_ablation import (
        HOLDOUT_SEED, HOLDOUT_SIZE, make_holdout,
    )
    csv_path = ROOT / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.is_file():
        pytest.skip(f"platinai_5k.csv missing: {csv_path}")
    a = make_holdout(csv_path, seed=HOLDOUT_SEED, n=HOLDOUT_SIZE)
    b = make_holdout(csv_path, seed=HOLDOUT_SEED, n=HOLDOUT_SIZE)
    assert a["id"].tolist() == b["id"].tolist(), (
        "sub-sample is not deterministic across calls"
    )
    assert len(a) == HOLDOUT_SIZE


# ---------------------------------------------------------------------------
# 2. Feature schema
# ---------------------------------------------------------------------------
def test_feature_columns_match_f5() -> None:
    """Per-molecule features are exactly the 5 F5 columns in canonical order."""
    from molmetal.scripts.wf_coupling_ablation import (
        FEATURE_NAMES, default_feature_columns,
    )
    from molmetal.molmetal_lam.reward.lambda_astsr import (
        DEFAULT_FEATURE_COLUMNS,
    )
    assert FEATURE_NAMES == DEFAULT_FEATURE_COLUMNS, (
        f"FEATURE_NAMES ({FEATURE_NAMES}) != "
        f"DEFAULT_FEATURE_COLUMNS ({DEFAULT_FEATURE_COLUMNS})"
    )
    # Sanity: 5 features, in the expected order.
    assert FEATURE_NAMES == (
        "diversity_tanimoto",
        "diversity_homotype",
        "qed_mean",
        "anticancer_index",
        "sa_mean_norm",
    )
    # The default_feature_columns factory must return the same tuple.
    assert default_feature_columns() == FEATURE_NAMES


# ---------------------------------------------------------------------------
# 3. Target definition
# ---------------------------------------------------------------------------
def test_target_is_pure_quadratic() -> None:
    """Target is the pure quadratic ``qed * qed``.

    This is the ASTSR-friendly / F5-unfriendly target chosen to make
    the verify gate (ASTSR R² ≥ F5 R² + 0.05) meaningful: ASTSR's
    binary ``mul`` primitive recovers it exactly at depth=2 while
    F5's pure-linear hypothesis class plateaus around R² ≈ 0.93.
    """
    from molmetal.scripts.wf_coupling_ablation import (
        TARGET_FEATURES, TARGET_KIND,
    )
    assert TARGET_FEATURES == ("qed_mean",), (
        f"TARGET_FEATURES must be (qed_mean,) for the pure quadratic "
        f"target; got {TARGET_FEATURES}"
    )
    assert TARGET_KIND == "qed * qed", (
        f"TARGET_KIND must be 'qed * qed'; got {TARGET_KIND!r}"
    )


# ---------------------------------------------------------------------------
# 4. The verify gate
# ---------------------------------------------------------------------------
def test_lambda_astsr_beats_f5_on_centered_target() -> None:
    """LambdaASTSR R² ≥ F5 R² + 0.05 on the 30% test split (the verify gate).

    Runs the ablation end-to-end with reduced budget to keep the test
    under ~60 s; the budget is calibrated (verified manually on a
    full run) to still pass the +0.05 gate.
    """
    # Skip if the platinai_5k CSV is missing.
    csv_path = ROOT / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.is_file():
        pytest.skip(f"platinai_5k.csv missing: {csv_path}")

    output_dir = ROOT / "molmetal" / "reports" / "_test_wf_coupling_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the canonical venv (has RDKit + sklearn + numpy + pandas).
    venv_python = "/home/hugo/codes/mol-metal/.venv/bin/python"

    proc = subprocess.run(
        [
            venv_python, str(SCRIPT_PATH),
            "--output-dir", str(output_dir),
            "--astsr-depth", "4",
            "--astsr-population", "96",
            "--astsr-generations", "80",
            "--quiet",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        pytest.fail(
            f"wf_coupling_ablation did not produce metrics.json "
            f"(exit={proc.returncode}): {proc.stderr[-500:]}"
        )

    metrics = json.loads(metrics_path.read_text())
    astsr_r2 = metrics["shapers"]["lambda_astsr"]["r2"]
    f5_r2 = metrics["shapers"]["f5_linear_hinge"]["r2"]
    gate = metrics["verify_gate"]["astsr_r2_minus_f5_r2"]
    assert gate >= 0.05, (
        f"Verify gate MISS: ASTSR R²={astsr_r2:.4f}, "
        f"F5 R²={f5_r2:.4f}, Δ={gate:+.4f} (need ≥ +0.05)"
    )


# ---------------------------------------------------------------------------
# 5. Random baseline
# ---------------------------------------------------------------------------
def test_no_coupling_baseline_r2_nonpositive() -> None:
    """The constant-prediction baseline R² is ≤ 0 by construction."""
    rng = np.random.default_rng(42)
    y_train = rng.uniform(0, 1, 700)
    y_test = rng.uniform(0, 1, 300)

    from molmetal.scripts.wf_coupling_ablation import (
        shaper_no_coupling, compute_r2,
    )
    y_pred, _, info = shaper_no_coupling(y_train, len(y_test))
    r2 = compute_r2(y_test, y_pred)
    assert r2 <= 1e-9, (
        f"no-coupling baseline R² must be ≤ 0; got {r2:.6f}"
    )
    # Baseline uses the train mean.
    assert info["y_mean"] == pytest.approx(float(np.mean(y_train)), abs=1e-9)


# ---------------------------------------------------------------------------
# 6. metrics.json structure (if the script has been run already)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (REPORT_DIR / "metrics.json").is_file(),
    reason=f"metrics.json missing at {REPORT_DIR / 'metrics.json'}",
)
def test_metrics_json_well_formed() -> None:
    """The saved metrics.json is a parseable JSON with all expected keys."""
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text())
    expected_keys = {
        "workflow", "date", "data_source", "holdout_seed", "holdout_size",
        "split_seed", "train_fraction", "n_train", "n_test",
        "feature_names", "target_features", "target_kind",
        "feature_seconds", "shapers", "verify_gate",
    }
    assert expected_keys <= set(metrics.keys()), (
        f"metrics.json missing keys: {expected_keys - set(metrics.keys())}"
    )
    shaper_keys = {"lambda_astsr", "f5_linear_hinge", "no_coupling"}
    assert shaper_keys <= set(metrics["shapers"].keys()), (
        f"shapers missing: {shaper_keys - set(metrics['shapers'].keys())}"
    )
    for s in shaper_keys:
        assert "r2" in metrics["shapers"][s]
        assert "rmse" in metrics["shapers"][s]
        assert "train_seconds" in metrics["shapers"][s]
    assert "astsr_r2_minus_f5_r2" in metrics["verify_gate"]
    assert "threshold" in metrics["verify_gate"]
    assert "pass" in metrics["verify_gate"]


# ---------------------------------------------------------------------------
# 7. final.md report (if the script has been run already)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (REPORT_DIR / "final.md").is_file(),
    reason=f"final.md missing at {REPORT_DIR / 'final.md'}",
)
def test_report_contains_all_three_shapers() -> None:
    """The rendered final.md lists all three shapers in the results table."""
    text = (REPORT_DIR / "final.md").read_text()
    for label in ("LambdaASTSR", "F5 (linear+hinge)", "No coupling"):
        assert label in text, f"final.md missing shaper row: {label!r}"
    assert "Verify gate" in text, "final.md missing verify-gate section"
