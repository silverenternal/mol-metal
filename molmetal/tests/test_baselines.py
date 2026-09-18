"""Tests for the classical ML baselines (Morgan FP + XGBoost / RF).

We use a tiny subset (``max_rows=200``) for the smoke test so pytest
stays fast.  The full run is exercised by
``molmetal/scripts/baselines.py``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.baselines.eval_utils import (
    compute_metrics,
    morgan_features,
    per_cell_line_auc,
)
from molmetal.baselines.morgan_xgb import MorganXGBBaseline, quick_smoke
from molmetal.baselines.rf_baseline import RFBaseline


# ---------------------------------------------------------------------------
# Import + API sanity
# ---------------------------------------------------------------------------
def test_imports() -> None:
    """All baseline modules import cleanly."""
    from molmetal.baselines import eval_utils, morgan_xgb, rf_baseline  # noqa: F401


def test_morgan_features_shape() -> None:
    """``morgan_features`` returns (N, n_bits) uint8 array."""
    fps = morgan_features(["CCO", "c1ccccc1", "INVALID_SMILES"], n_bits=512)
    assert fps.shape == (3, 512)
    assert fps.dtype == "uint8"
    # Invalid SMILES → zero vector
    assert fps[2].sum() == 0
    # Ethanol + benzene → at least some bits set
    assert fps[0].sum() > 0
    assert fps[1].sum() > 0


def test_compute_metrics_basic() -> None:
    """``compute_metrics`` returns sane ROC-AUC, AP and Hit@5%."""
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]
    out = compute_metrics(y_true, y_score)
    assert 0.5 <= out["roc_auc"] <= 1.0
    assert 0.5 <= out["pr_auc"] <= 1.0
    assert 0.0 <= out["hit_rate_top5pct"] <= 1.0
    assert out["n"] == 4


def test_per_cell_line_auc() -> None:
    """Per-cell-line AUC respects min_count threshold."""
    cell_lines = ["A549"] * 50 + ["HeLa"] * 5
    y_true = [0] * 27 + [1] * 23 + [0] * 3 + [1] * 2
    y_score = [0.1] * 27 + [0.9] * 23 + [0.1] * 3 + [0.9] * 2
    out = per_cell_line_auc(cell_lines, y_true, y_score, min_count=10)
    assert "A549" in out
    # HeLa has only 5 rows → skipped
    assert "HeLa" not in out


# ---------------------------------------------------------------------------
# Smoke test: full pipeline on a tiny subset
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_xgb_smoke_ru() -> None:
    """XGBoost baseline on a 200-row Ru subset completes and AUC > 0.5."""
    res = quick_smoke(metal="Ru", max_rows=200)
    assert res.model == "xgb"
    assert res.metal == "Ru"
    auc = res.test_metrics["roc_auc"]
    # We don't expect a strong AUC on 200 rows, but the classifier must do
    # better than chance.
    assert auc > 0.5, f"XGB smoke AUC={auc:.4f} not above chance"


@pytest.mark.slow
def test_xgb_serializable() -> None:
    """Result object serialises to JSON cleanly."""
    res = quick_smoke(metal="Ru", max_rows=200)
    blob = res.to_json()
    parsed = json.loads(blob)
    assert parsed["metal"] == "Ru"
    assert parsed["model"] == "xgb"
    assert "test_metrics" in parsed
    assert "val_metrics" in parsed


@pytest.mark.slow
def test_rf_smoke_ru() -> None:
    """RandomForest baseline runs end-to-end and emits a JSON-able result."""
    runner = RFBaseline(metal="Ru")
    # Monkey-patch the dataset load to use a tiny head of the Ru data
    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    flt = CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=["Ru"])
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    small = MetalCytotoxDataset(df=ds.df.head(200).copy())

    # Re-use the dataset in run() by intercepting the from_csv call
    orig_from_csv = MetalCytotoxDataset.from_csv
    MetalCytotoxDataset.from_csv = classmethod(lambda cls, **kw: small)  # type: ignore[assignment]
    try:
        res = runner.run()
    finally:
        MetalCytotoxDataset.from_csv = orig_from_csv  # type: ignore[assignment]
    assert res.model == "rf"
    # JSON-serialisable
    json.loads(res.to_json())
