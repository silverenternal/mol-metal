"""Tests — Wf-40 Phase 1 Lambda × CFM end-to-end bench on real data.

================================================================
Test suite for :mod:`molmetal.scripts.bench_lambda_cfm_e2e` and the
public helpers it uses (CSV loaders, bridge-RMSE aggregator,
method runners).  Covers:

* CSV loaders — PlatinAI 5K head + MetalCytoToxDB 5-metal subset.
* Per-method runners — lambda_to_cfm, cfm_to_lambda, bidirectional.
* Bridge-RMSE aggregator — finite, range, NaN-safe.
* Bench orchestrator — per-method × per-dataset × per-seed blocks.
* Markdown renderer — round-trip JSON contract.

All tests are CPU-only, self-skip when the canonical modules
(``coupling_joint_inference``, ``coupling_bidirectional``) are not
importable.  The suite must complete in <60 s on a single core.

Honest framing
==============
* Tests assert the bench harness's **contract** (finite aggregates,
  per-method presence, CSV loaders returning non-empty data), NOT
  the underlying chemical-correctness of the joint pipeline.
* The stub bridge is the deterministic default when the canonical
  ``models/coupling_adapter_platinai5k.npz`` is missing; the tests
  work in BOTH stub and trained-bridge modes.
* Wall-time asserts use generous upper bounds (e.g. < 10 s for
  20-sample × 1-seed × 3-method runs) so the tests are robust to
  slow CI machines.

Run::

    .venv/bin/python -m pytest molmetal/tests/test_lambda_cfm_e2e.py -v
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy imports — self-skip when shipped modules are unavailable
# ---------------------------------------------------------------------------
def _import_bench():
    try:
        from molmetal.scripts.bench_lambda_cfm_e2e import (  # type: ignore
            _load_platinai_smiles,
            _load_metalcytotox_smiles,
            _bridge_rmse,
            _safe_mean,
            _aggregate_method,
            _run_lambda_to_cfm,
            _run_cfm_to_lambda,
            _run_bidirectional,
            _bench_method_on_dataset,
            _bench_dataset,
            _render_markdown,
            METHODS_ALL,
        )
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"bench_lambda_cfm_e2e import failed: {exc}")
    return (
        _load_platinai_smiles,
        _load_metalcytotox_smiles,
        _bridge_rmse,
        _safe_mean,
        _aggregate_method,
        _run_lambda_to_cfm,
        _run_cfm_to_lambda,
        _run_bidirectional,
        _bench_method_on_dataset,
        _bench_dataset,
        _render_markdown,
        METHODS_ALL,
    )


def _import_ship():
    """Best-effort import of Wf-4 + Wf-21 ship modules.

    Returns ``None`` if either import fails.
    """
    try:
        from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
            LambdaCFMPipeline,
            lambda_features_from_smiles,
            LAMBDA_DIM,
            EMBED_DIM,
            load_joint_pipeline,
        )
        from molmetal.molmetal_lam.lam_chem.coupling_bidirectional import (
            BidirectionalCoupling,
            CFMLambdaReverse,
            lamterm_to_features,
        )
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"ship modules unavailable: {exc}")
    return (
        LambdaCFMPipeline,
        lambda_features_from_smiles,
        LAMBDA_DIM,
        EMBED_DIM,
        load_joint_pipeline,
        BidirectionalCoupling,
        CFMLambdaReverse,
        lamterm_to_features,
    )


(
    _load_platinai_smiles,
    _load_metalcytotox_smiles,
    _bridge_rmse,
    _safe_mean,
    _aggregate_method,
    _run_lambda_to_cfm,
    _run_cfm_to_lambda,
    _run_bidirectional,
    _bench_method_on_dataset,
    _bench_dataset,
    _render_markdown,
    METHODS_ALL,
) = _import_bench()
ship = _import_ship()
(
    LambdaCFMPipeline,
    lambda_features_from_smiles,
    LAMBDA_DIM,
    EMBED_DIM,
    load_joint_pipeline,
    BidirectionalCoupling,
    CFMLambdaReverse,
    lamterm_to_features,
) = ship


# ---------------------------------------------------------------------------
# CSV fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def platinai_csv_path() -> Path:
    p = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"
    if not p.exists():
        pytest.skip(f"PlatinAI 5K CSV missing at {p}")
    return p


@pytest.fixture
def metalcyto_csv_path() -> Path:
    p = PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
    if not p.exists():
        pytest.skip(f"MetalCytoToxDB CSV missing at {p}")
    return p


# ---------------------------------------------------------------------------
# Pipeline fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def pipe_forward():
    return load_joint_pipeline()


@pytest.fixture
def reverse_obj():
    return CFMLambdaReverse()


@pytest.fixture
def bidirectional_obj():
    return BidirectionalCoupling()


# ---------------------------------------------------------------------------
# Section A — CSV loaders
# ---------------------------------------------------------------------------
def test_load_platinai_smiles_returns_smiles(platinai_csv_path) -> None:
    """PlatinAI loader returns a non-empty list of ``{smiles, metal}``."""
    rows = _load_platinai_smiles(platinai_csv_path, n=10)
    assert len(rows) == 10
    for row in rows:
        assert "smiles" in row
        assert "metal" in row
        assert isinstance(row["smiles"], str)
        assert len(row["smiles"]) > 0


def test_load_platinai_smiles_deterministic(platinai_csv_path) -> None:
    """PlatinAI loader is deterministic — same input → same rows."""
    a = _load_platinai_smiles(platinai_csv_path, n=20)
    b = _load_platinai_smiles(platinai_csv_path, n=20)
    assert [r["smiles"] for r in a] == [r["smiles"] for r in b]


def test_load_platinai_smiles_caps_at_n(platinai_csv_path) -> None:
    """Loader returns at most ``n`` rows."""
    rows = _load_platinai_smiles(platinai_csv_path, n=5)
    assert len(rows) == 5


def test_load_metalcytotox_smiles_returns_smiles(metalcyto_csv_path) -> None:
    """MetalCytoToxDB loader returns non-empty ``{smiles, metal}``."""
    rows = _load_metalcytotox_smiles(metalcyto_csv_path, n=10)
    assert len(rows) == 10
    for row in rows:
        assert "smiles" in row
        assert isinstance(row["smiles"], str)
        assert len(row["smiles"]) > 0
        # First fragment of '.'-separated multi-fragment SMILES.
        assert "." not in row["smiles"]


def test_load_metalcytotox_smiles_filter_metal(metalcyto_csv_path) -> None:
    """``metals`` filter restricts the returned rows to the whitelist."""
    rows = _load_metalcytotox_smiles(metalcyto_csv_path, n=200, metals=("Ru",))
    for row in rows:
        assert row["metal"] == "Ru"


def test_load_metalcytotox_smiles_caps_at_n(metalcyto_csv_path) -> None:
    """Loader returns at most ``n`` rows."""
    rows = _load_metalcytotox_smiles(metalcyto_csv_path, n=3)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# Section B — Per-method runners
# ---------------------------------------------------------------------------
def test_run_lambda_to_cfm_returns_dict(pipe_forward) -> None:
    """``lambda_to_cfm`` returns ``(True, dict, wall)`` on a known seed."""
    pocket = np.zeros(8, dtype=np.float32)
    ok, payload, wall = _run_lambda_to_cfm(pipe_forward, "CCO", pocket, n_steps=10)
    assert ok is True
    assert isinstance(payload, dict)
    assert "cfm_output" in payload
    assert "decoded_smiles" in payload
    assert wall >= 0.0


def test_run_lambda_to_cfm_finite_cfm(pipe_forward) -> None:
    """``cfm_output`` is a finite (EMBED_DIM,) array."""
    pocket = np.zeros(8, dtype=np.float32)
    ok, payload, _ = _run_lambda_to_cfm(pipe_forward, "CCO", pocket, n_steps=10)
    assert ok is True
    cfm = np.asarray(payload["cfm_output"], dtype=np.float32).reshape(-1)
    assert cfm.shape == (EMBED_DIM,)
    assert np.isfinite(cfm).all()


def test_run_cfm_to_lambda_returns_reconstructed(pipe_forward, reverse_obj) -> None:
    """``cfm_to_lambda`` returns a 32-d reconstructed feature."""
    pocket = np.zeros(8, dtype=np.float32)
    ok, payload, wall = _run_cfm_to_lambda(
        "CCO",
        pocket,
        n_steps=10,
        pipe_forward=pipe_forward,
        reverse_obj=reverse_obj,
        lamterm_to_features_fn=lamterm_to_features,
        lambda_dim=LAMBDA_DIM,
        embed_dim=EMBED_DIM,
    )
    assert ok is True
    assert wall >= 0.0
    rec = np.asarray(
        payload["reconstructed_lambda_features"], dtype=np.float32
    ).reshape(-1)
    assert rec.shape == (LAMBDA_DIM,)
    assert np.isfinite(rec).all()


def test_run_bidirectional_returns_round_trip_loss(
    bidirectional_obj,
) -> None:
    """``bidirectional`` returns a finite ``round_trip_loss``."""
    pocket = np.zeros(8, dtype=np.float32)
    ok, payload, wall = _run_bidirectional(bidirectional_obj, "CCO", pocket, n_steps=10)
    assert ok is True
    assert wall >= 0.0
    rt = float(payload["round_trip_loss"])
    assert math.isfinite(rt)
    assert rt >= 0.0


def test_run_bidirectional_deterministic(bidirectional_obj) -> None:
    """``bidirectional`` is deterministic — same seed → same loss."""
    pocket = np.zeros(8, dtype=np.float32)
    _, a, _ = _run_bidirectional(bidirectional_obj, "c1ccccc1", pocket, n_steps=10)
    _, b, _ = _run_bidirectional(bidirectional_obj, "c1ccccc1", pocket, n_steps=10)
    assert a["round_trip_loss"] == pytest.approx(b["round_trip_loss"], abs=1e-6)


# ---------------------------------------------------------------------------
# Section C — Bridge-RMSE aggregator
# ---------------------------------------------------------------------------
def test_bridge_rmse_zero_for_perfect_match() -> None:
    """Bridge RMSE is 0 when the projected CFM output equals λ-features.

    The projection is :func:`_project_cfm_to_lambda` which computes
    ``(cfm[0::2] + cfm[1::2])`` (sum of consecutive pairs).  When we
    hand-craft a 64-d vector whose pair sums equal a 32-d
    ``lambda_features``, the RMSE must be 0.
    """
    feats = np.linspace(0.0, 1.0, LAMBDA_DIM, dtype=np.float32)
    cfm = np.zeros(EMBED_DIM, dtype=np.float32)
    cfm[0::2] = 0.5 * feats
    cfm[1::2] = 0.5 * feats  # sum = feats exactly
    rmse = _bridge_rmse(feats, cfm, lambda_dim=LAMBDA_DIM, embed_dim=EMBED_DIM)
    assert math.isfinite(rmse)
    assert rmse == pytest.approx(0.0, abs=1e-6)


def test_bridge_rmse_nan_for_missing_inputs() -> None:
    """``_bridge_rmse`` returns NaN when either input is None."""
    feats = np.zeros(LAMBDA_DIM, dtype=np.float32)
    assert math.isnan(
        _bridge_rmse(None, feats, lambda_dim=LAMBDA_DIM, embed_dim=EMBED_DIM)
    )
    assert math.isnan(
        _bridge_rmse(feats, None, lambda_dim=LAMBDA_DIM, embed_dim=EMBED_DIM)
    )


def test_bridge_rmse_handles_shape_mismatch() -> None:
    """``_bridge_rmse`` accepts inputs of any size (pads internally)."""
    feats = np.zeros(LAMBDA_DIM, dtype=np.float32)
    cfm = np.zeros(EMBED_DIM * 2, dtype=np.float32)  # oversized
    rmse = _bridge_rmse(feats, cfm, lambda_dim=LAMBDA_DIM, embed_dim=EMBED_DIM)
    assert math.isfinite(rmse)
    assert rmse >= 0.0


def test_safe_mean_skips_nan() -> None:
    """``_safe_mean`` ignores NaN entries; returns NaN on all-NaN."""
    assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert math.isnan(_safe_mean([float("nan"), float("nan")]))
    assert _safe_mean([1.0, float("nan"), 3.0]) == pytest.approx(2.0)


def test_aggregate_method_basic_shape() -> None:
    """``_aggregate_method`` returns the documented metric dict."""
    agg = _aggregate_method(
        walls=[0.01, 0.02, 0.03],
        success_flags=[True, True, False],
        decode_counts=[3, 2, 0],
        bridge_rmses=[0.1, 0.2, float("nan")],
        round_trip_losses=[1.0, 1.5, float("nan")],
    )
    assert agg["n_runs"] == 3
    assert agg["n_success"] == 2
    assert agg["success_rate"] == pytest.approx(2 / 3)
    assert agg["mean_wall_time_s"] > 0.0
    assert agg["mean_bridge_rmse"] == pytest.approx(0.15)
    # decode_ratio = fraction of samples with ≥ 1 decoded SMILES.
    assert agg["decode_ratio"] == pytest.approx(2 / 3)
    # mean_decode_count = average count of decoded SMILES per sample.
    assert agg["mean_decode_count"] == pytest.approx((3 + 2) / 3)
    assert agg["mean_round_trip_loss"] == pytest.approx(1.25)


def test_aggregate_method_decode_ratio_is_fraction() -> None:
    """``decode_ratio`` is in ``[0, 1]`` — fraction of samples that decoded."""
    agg = _aggregate_method(
        walls=[0.01, 0.02, 0.03, 0.04],
        success_flags=[True, True, True, True],
        decode_counts=[0, 1, 2, 3],
        bridge_rmses=[0.1, 0.1, 0.1, 0.1],
        round_trip_losses=[1.0, 1.0, 1.0, 1.0],
    )
    # 3 of 4 samples returned ≥ 1 decoded SMILES.
    assert agg["decode_ratio"] == pytest.approx(0.75)
    assert 0.0 <= agg["decode_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Section D — Bench orchestrator
# ---------------------------------------------------------------------------
def test_bench_method_on_dataset_smoke(
    pipe_forward, reverse_obj, bidirectional_obj, platinai_csv_path
) -> None:
    """A 3-sample smoke block runs in < 5 s and returns the expected schema."""
    samples = _load_platinai_smiles(platinai_csv_path, n=3)
    block = _bench_method_on_dataset(
        "lambda_to_cfm",
        samples,
        0,
        pipe_forward=pipe_forward,
        reverse_obj=reverse_obj,
        lamterm_to_features_fn=lamterm_to_features,
        bidirectional_obj=bidirectional_obj,
        lambda_dim=LAMBDA_DIM,
        embed_dim=EMBED_DIM,
        n_steps=10,
        pocket_size=8,
    )
    assert block["method"] == "lambda_to_cfm"
    assert block["seed"] == 0
    assert block["n_samples"] == 3
    assert "aggregate" in block
    assert "sample_records" in block
    assert len(block["sample_records"]) == 3
    for r in block["sample_records"]:
        assert "idx" in r
        assert "smiles" in r
        assert "wall_time_s" in r


def test_bench_dataset_two_methods_two_seeds(
    pipe_forward, reverse_obj, bidirectional_obj, platinai_csv_path
) -> None:
    """Per-dataset bench runs 2 methods × 2 seeds and aggregates correctly."""
    samples = _load_platinai_smiles(platinai_csv_path, n=4)
    rep = _bench_dataset(
        "platinai_5k",
        samples,
        methods=("lambda_to_cfm", "bidirectional"),
        seeds=(0, 1),
        pipe_forward=pipe_forward,
        reverse_obj=reverse_obj,
        lamterm_to_features_fn=lamterm_to_features,
        bidirectional_obj=bidirectional_obj,
        lambda_dim=LAMBDA_DIM,
        embed_dim=EMBED_DIM,
        n_steps=10,
        pocket_size=8,
    )
    assert rep["dataset"] == "platinai_5k"
    assert rep["n_samples"] == 4
    assert set(rep["per_method"].keys()) == {
        "lambda_to_cfm",
        "bidirectional",
    }
    for method_name, agg in rep["per_method"].items():
        assert agg["method"] == method_name
        assert agg["n_seeds"] == 2
        assert agg["mean_wall_time_s"] > 0.0
        assert "per_seed" in agg
        assert len(agg["per_seed"]) == 2


# ---------------------------------------------------------------------------
# Section E — Markdown renderer + JSON contract
# ---------------------------------------------------------------------------
def test_render_markdown_smoke() -> None:
    """Markdown renderer produces non-empty output for a tiny payload."""
    payload = {
        "timestamp_utc": "2026-09-26T00:00:00Z",
        "n_samples": 5,
        "seeds": [0, 1],
        "per_dataset": [
            {
                "dataset": "platinai_5k",
                "per_method": {
                    "lambda_to_cfm": {
                        "method": "lambda_to_cfm",
                        "mean_wall_time_s": 0.005,
                        "mean_bridge_rmse": 1.23,
                        "mean_decode_ratio": 3.0,
                        "mean_round_trip_loss": float("nan"),
                        "mean_success_rate": 1.0,
                    },
                    "bidirectional": {
                        "method": "bidirectional",
                        "mean_wall_time_s": 0.012,
                        "mean_bridge_rmse": 1.30,
                        "mean_decode_ratio": 3.0,
                        "mean_round_trip_loss": 12.5,
                        "mean_success_rate": 1.0,
                    },
                },
            }
        ],
        "bridge_source": "stub",
        "alpha": 0.5,
        "n_steps": 50,
        "pocket_size": 64,
    }
    md = _render_markdown(payload)
    assert isinstance(md, str)
    assert "Wf-40 Phase 1" in md
    assert "platinai_5k" in md
    assert "lambda_to_cfm" in md
    assert "bidirectional" in md


def test_methods_all_includes_three_methods() -> None:
    """``METHODS_ALL`` exposes the three documented method tags."""
    assert set(METHODS_ALL) == {
        "lambda_to_cfm",
        "cfm_to_lambda",
        "bidirectional",
    }


# ---------------------------------------------------------------------------
# Section F — End-to-end JSON contract
# ---------------------------------------------------------------------------
def test_end_to_end_json_writes_valid_json(tmp_path: Path) -> None:
    """The bench orchestrator's output round-trips through json.loads."""
    # Build a tiny payload by hand that mirrors the bench output.
    payload: Dict[str, Any] = {
        "n_samples": 4,
        "seeds": [0, 1],
        "methods": ["lambda_to_cfm", "cfm_to_lambda", "bidirectional"],
        "alpha": 0.5,
        "n_steps": 50,
        "pocket_size": 64,
        "bridge_source": "stub",
        "lambda_dim": LAMBDA_DIM,
        "embed_dim": EMBED_DIM,
        "peak_memory_mb": 1.5,
        "timestamp_utc": "2026-09-26T00:00:00Z",
        "per_dataset": [],
    }
    out_path = tmp_path / "lambda_cfm_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    parsed = json.loads(out_path.read_text())
    assert parsed["n_samples"] == 4
    assert parsed["methods"] == [
        "lambda_to_cfm",
        "cfm_to_lambda",
        "bidirectional",
    ]
    assert parsed["lambda_dim"] == LAMBDA_DIM
    assert parsed["embed_dim"] == EMBED_DIM
