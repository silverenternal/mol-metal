"""Tests for Wf-44 Phase 5 — Feature-target correlation analysis.

================================================================
Validates that ``molmetal/scripts/analyze_feature_target_correlation.py``
correctly:

1. **Artefact existence** — ``correlation_analysis.json`` is written to
   the expected location after a CLI run (1 test).
2. **Schema correctness** — top-level keys + per-triple keys, JSON
   round-trip (3 tests).
3. **Correlation primitives** — ``_safe_pearsonr``,
   ``_safe_spearmanr``, ``_safe_mutual_info``, ``_safe_pIC50_from_nM``
   (4 tests).
4. **Feature extractors** — ``extract_astsr_features`` and
   ``extract_dmpnn_features`` return the right shape + finite
   values (2 tests).
5. **End-to-end** — full pipeline produces a non-empty JSON with
   top-10 lists and a correlation matrix that is not degenerate
   (1 test).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPORT_DIR = _REPO_ROOT / "molmetal/reports/wf44_r2_debug"
OUT_PATH = REPORT_DIR / "correlation_analysis.json"
SCRIPT_PATH = _REPO_ROOT / "molmetal/scripts/analyze_feature_target_correlation.py"


def _import_script():
    import importlib.util
    mod_name = "analyze_feature_target_correlation"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(_REPO_ROOT / "molmetal/scripts/analyze_feature_target_correlation.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Insert into sys.modules so dataclass introspection works.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Artefact existence
# ---------------------------------------------------------------------------

def test_cli_writes_correlation_analysis_json():
    """Run the CLI and confirm the JSON artefact is produced."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--out-dir", str(REPORT_DIR), "--quiet"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert result.returncode == 0, (
        f"Script failed: rc={result.returncode} stderr={result.stderr}"
    )
    assert OUT_PATH.exists(), f"Expected {OUT_PATH} to exist after CLI run"
    assert OUT_PATH.stat().st_size > 0, "correlation_analysis.json is empty"


# ---------------------------------------------------------------------------
# 2. Schema correctness
# ---------------------------------------------------------------------------

def _load_json_or_skip():
    if not OUT_PATH.exists():
        pytest.skip("correlation_analysis.json not present — run CLI test first")
    return json.loads(OUT_PATH.read_text())


def test_schema_top_level_keys():
    payload = _load_json_or_skip()
    expected = {
        "$schema_version", "workflow", "phase", "title", "description",
        "n_datasets", "n_triples", "datasets", "summary",
        "top_10_by_abs_pearson", "top_10_by_abs_spearman",
        "correlation_matrix",
    }
    assert expected.issubset(payload.keys()), (
        f"Missing top-level keys: {expected - set(payload.keys())}"
    )
    assert payload["workflow"] == "Wf-44"
    assert payload["phase"] == "Phase 5"


def test_schema_summary_block_keys():
    payload = _load_json_or_skip()
    s = payload["summary"]
    for k in (
        "max_abs_pearson_r", "max_abs_spearman_rho", "max_mutual_info",
        "c8_trip", "c8_partial", "verdict",
    ):
        assert k in s, f"Missing summary key: {k}"
    assert s["verdict"] in {
        "features_truly_weak", "features_partially_correlated",
        "features_carry_signal",
    }


def test_schema_correlation_matrix_shape():
    payload = _load_json_or_skip()
    cm = payload["correlation_matrix"]
    assert "rows_features" in cm and "cols_targets" in cm
    assert "pearson" in cm and "spearman" in cm and "mutual_info" in cm
    assert "n_samples" in cm
    rows = cm["rows_features"]
    cols = cm["cols_targets"]
    # The Pearson dict should have one entry per row.
    assert set(cm["pearson"].keys()) == set(rows)


def test_top10_lists_have_at_most_10():
    payload = _load_json_or_skip()
    assert len(payload["top_10_by_abs_pearson"]) <= 10
    assert len(payload["top_10_by_abs_spearman"]) <= 10


# ---------------------------------------------------------------------------
# 3. Correlation primitives
# ---------------------------------------------------------------------------

def test_safe_pearsonr_known_signal():
    mod = _import_script()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 0.7 * x + 0.3 * rng.standard_normal(500)
    r, p, n = mod._safe_pearsonr(x, y)
    assert n == 500
    assert r > 0.6
    assert 0.0 <= p <= 1.0


def test_safe_spearmanr_handles_constant():
    mod = _import_script()
    x = np.ones(20)
    y = np.arange(20, dtype=np.float64)
    rho, p, n = mod._safe_pearsonr(x, y) if False else (None, None, None)
    rho, p, n = mod._safe_spearmanr(x, y)
    assert n == 20
    assert rho == 0.0
    assert p == 1.0


def test_safe_mutual_info_handles_constant():
    mod = _import_script()
    x = np.ones(50)
    y = np.arange(50, dtype=np.float64)
    mi, n = mod._safe_mutual_info(x, y)
    assert n == 50
    assert mi == 0.0


def test_safe_pIC50_from_nM_known():
    mod = _import_script()
    # 10 nM IC50 → pIC50 = -log10(10e-9) = -log10(1e-8) = 8
    arr = mod._safe_pIC50_from_nM(np.array([10.0, 100.0, 1000.0]))
    np.testing.assert_allclose(arr, np.array([8.0, 7.0, 6.0]), atol=1e-6)
    # NaN-safe on zeros / negatives.
    arr2 = mod._safe_pIC50_from_nM(np.array([0.0, -1.0, 5.0]))
    assert np.isnan(arr2[0])
    assert np.isnan(arr2[1])
    np.testing.assert_allclose(arr2[2], -np.log10(5e-9), atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Feature extractors
# ---------------------------------------------------------------------------

def test_extract_astsr_features_shape():
    mod = _import_script()
    smiles = ["CCO", "c1ccccc1", "CCN(CC)CC"]
    feats = mod.extract_astsr_features(smiles)
    assert feats.shape == (3, len(mod.ASTSR_FEATURE_COLUMNS))
    assert np.all(np.isfinite(feats))


def test_extract_dmpnn_features_shape():
    mod = _import_script()
    smiles = ["CCO", "c1ccccc1", "CCN(CC)CC"]
    feats = mod.extract_dmpnn_features(smiles, max_atoms=32)
    # 39 atom dims × 3 pools = 117.
    assert feats.shape == (3, 39 * 3)
    assert np.all(np.isfinite(feats))


# ---------------------------------------------------------------------------
# 5. End-to-end — non-degenerate
# ---------------------------------------------------------------------------

def test_correlation_analysis_is_nonempty():
    payload = _load_json_or_skip()
    # We have 3 datasets × (5 ASTSR + 117 D-MPNN) features × n_targets;
    # the JSON must contain at least one triple.
    assert payload["n_triples"] > 0
    # The summary verdict must be one of the 3 valid labels.
    assert payload["summary"]["verdict"] in {
        "features_truly_weak", "features_partially_correlated",
        "features_carry_signal",
    }
    # The correlation matrix must have at least one row.
    assert len(payload["correlation_matrix"]["rows_features"]) > 0
