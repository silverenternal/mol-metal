"""Integration tests — Lambda × CFM CPU joint inference skeleton (TODO-21).

================================================================
This module extends the 25 unit tests in
:mod:`molmetal.tests.test_lambda_cfm_joint_inference` with 15
**integration** tests that exercise the joint pipeline against:

* the real PlatinAI 5K CPU pretraining checkpoint,
* the legacy :class:`molmetal.molmetal_lam.lam_chem.coupling_adapter.CouplingAdapter`
  (9-d → 64-d internal stub),
* the :class:`molmetal.molmetal_lam.reward.lambda_astsr.LambdaASTSR`
  Deflexpressor-style symbolic regressor (R² contract),
* the :class:`CascadedRefiner` Strategy 3 refinement,
* the actual PlatinAI 5K CSV (``molmetal/data/platinai_5k.csv``).

All 15 tests are CPU-only and run in < 60 s on a single core.

Honest framing
==============
The joint inference skeleton is a **stand-in** for the full
GPU-retrained Lambda × CFM loop (TODO-21 Strategy 2).  These
integration tests assert:

* the 32-d → 64-d bridge is consistent with the 9-d → 64-d legacy
  adapter (both live in the 64-d embedding space),
* the ASTSR-formulated reward flag is honored at the pipeline layer
  (Strategy 1, Lambda-as-reward),
* the CascadedRefiner composes correctly on top of the bridge output
  (Strategy 3),
* the real PlatinAI 5K pretrain checkpoint produces finite output
  for real PlatinAI SMILES,
* the pipeline is deterministic and memory-bounded across many
  sequential runs (no leak).
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipe_real():
    """Real pipeline loaded from the canonical PlatinAI 5K checkpoint."""
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        DEFAULT_PRETRAIN_NPZ,
        LambdaCFMPipeline,
        load_platinai5k_bridge,
    )

    if not DEFAULT_PRETRAIN_NPZ.exists():
        pytest.skip(f"PlatinAI 5K npz missing: {DEFAULT_PRETRAIN_NPZ}")
    os.environ["LAMBDA_ASTSR_ENABLED"] = "1"
    return LambdaCFMPipeline(bridge=load_platinai5k_bridge())


@pytest.fixture(scope="module")
def pipe_stub():
    """Pipeline with the deterministic stub bridge (no checkpoint)."""
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        LambdaCFMPipeline,
    )

    os.environ["LAMBDA_ASTSR_ENABLED"] = "1"
    return LambdaCFMPipeline(
        bridge=LambdaCFMPipeline().bridge,
        use_lambda_astsr=True,
        alpha=0.5,
        seed=0,
    )


@pytest.fixture(scope="module")
def platinai_smiles() -> List[str]:
    """First 100 PlatinAI SMILES from the on-disk 5K CSV."""
    csv_path = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.exists():
        pytest.skip(f"platinai_5k.csv missing: {csv_path}")
    out: List[str] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= 100:
                break
            smi = (row.get("smiles") or "").strip()
            if smi:
                out.append(smi)
    if not out:
        pytest.skip("platinai_5k.csv contains no usable SMILES")
    return out


# ---------------------------------------------------------------------------
# 1 — LambdaASTSR reward shaper integration (Strategy 1)
# ---------------------------------------------------------------------------
def test_integration_with_lambda_astsr(pipe_stub) -> None:
    """Pipeline honors the ASTSR opt-in flag from the env-var gate."""
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        LambdaCFMPipeline,
    )

    os.environ["LAMBDA_ASTSR_ENABLED"] = "1"
    p_on = LambdaCFMPipeline(
        bridge=pipe_stub.bridge, use_lambda_astsr=True, seed=0
    )
    os.environ["LAMBDA_ASTSR_ENABLED"] = "0"
    p_off = LambdaCFMPipeline(
        bridge=pipe_stub.bridge, use_lambda_astsr=False, seed=0
    )
    pocket = np.zeros(8, dtype=np.float32)
    on_out = p_on.run_full_pipeline("CCO", pocket, n_steps=5)
    off_out = p_off.run_full_pipeline("CCO", pocket, n_steps=5)
    assert on_out["use_lambda_astsr"] is True
    assert off_out["use_lambda_astsr"] is False
    # Numerical output is identical (the flag is metadata-only).
    np.testing.assert_array_equal(on_out["cfm_output"], off_out["cfm_output"])


def test_integration_with_lambda_astsr_channel_returns_finite(pipe_stub) -> None:
    """The shipped ASTSR reward channel returns a finite value on a row dict."""
    from molmetal.molmetal_lam.reward.lambda_astsr import (
        DEFAULT_FEATURE_COLUMNS,
        LambdaASTSR,
        make_lambda_astsr_reward_channel,
    )

    os.environ["LAMBDA_ASTSR_ENABLED"] = "1"
    astsr = LambdaASTSR(
        depth=2, population=8, generations=2, seed=0,
        feature_names=DEFAULT_FEATURE_COLUMNS,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, len(DEFAULT_FEATURE_COLUMNS))).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1]  # linear target
    astsr.fit(X, y)
    ch = make_lambda_astsr_reward_channel(astsr=astsr, enabled=True)
    row = {fn: float(X[0, i]) for i, fn in enumerate(DEFAULT_FEATURE_COLUMNS)}
    val = ch(row)
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# 2 — CouplingAdapter (9-d → 64-d legacy) integration
# ---------------------------------------------------------------------------
def test_integration_with_coupling_adapter(pipe_stub) -> None:
    """The 9-d → 64-d CouplingAdapter.embed_features is dimension-compatible
    with the 32-d → 64-d PlatinAI 5K bridge (both live in the 64-d space).
    """
    from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
        EMBED_DIM as LEGACY_EMBED_DIM,
        CouplingAdapter,
        load_coupling_adapter,
    )

    adapter = load_coupling_adapter(prefer_stub=True)
    x9 = np.random.default_rng(0).standard_normal(9).astype(np.float32)
    legacy_out = adapter.embed_features(x9)
    assert legacy_out.shape == (LEGACY_EMBED_DIM,) == (64,)

    # Bridge (joint inference) operates in the same 64-d space.
    x32 = np.random.default_rng(1).standard_normal(32).astype(np.float32)
    bridge_out = pipe_stub.embed_lambda_features(x32)
    assert bridge_out.shape == (64,)
    assert isinstance(adapter, CouplingAdapter)
    assert np.isfinite(legacy_out).all()
    assert np.isfinite(bridge_out).all()


def test_integration_coupling_adapter_and_joint_pipeline_share_embed_dim(
    pipe_stub,
) -> None:
    """Both adapters share ``EMBED_DIM = 64`` (the joint pipeline contract)."""
    from molmetal.molmetal_lam.lam_chem.coupling_adapter import (
        EMBED_DIM as LEGACY_EMBED_DIM,
    )
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        EMBED_DIM,
    )

    assert EMBED_DIM == 64
    assert LEGACY_EMBED_DIM == EMBED_DIM


# ---------------------------------------------------------------------------
# 3 — Deflexpressor R² integration (LambdaASTSR fit + predict)
# ---------------------------------------------------------------------------
def test_integration_with_deflexpressor(pipe_stub) -> None:
    """LambdaASTSR fits a linear target and predicts finite values.

    This is the "R² integration" smoke: an ASTSR with depth=2 +
    generations=2 must learn enough to predict on held-out X without
    NaN.  We do NOT require R² > 0.5 here — that gate belongs in
    ``test_lambda_astsr.py``.  This test only asserts the contract that
    the joint pipeline integration does not break ASTSR.
    """
    from molmetal.molmetal_lam.reward.lambda_astsr import (
        DEFAULT_FEATURE_COLUMNS,
        LambdaASTSR,
        compute_r2,
    )

    rng = np.random.default_rng(0)
    F = len(DEFAULT_FEATURE_COLUMNS)
    X = rng.standard_normal((60, F)).astype(np.float32)
    y = (X[:, 0] + 0.25 * X[:, 1] - 0.1 * X[:, 2]).astype(np.float32)

    astsr = LambdaASTSR(
        depth=3, population=16, generations=4, seed=0,
        feature_names=DEFAULT_FEATURE_COLUMNS,
    )
    astsr.fit(X[:40], y[:40])
    y_hat = astsr.predict(X[40:])
    r2 = compute_r2(y[40:], y_hat)
    # Honest bound: even random search on depth=3 / 16-pop / 4-gen
    # should keep the predictions in a sane range (R² > -10 guards
    # against catastrophic collapse, not over-fitting).
    assert r2 > -10.0
    assert np.isfinite(y_hat).all()


def test_integration_deflexpressor_channel_after_pipeline(pipe_stub) -> None:
    """ASTSR reward channel accepts the row dict that the pipeline emits.

    The pipeline's ``lambda_features`` (32-d) is *not* the same as the
    ASTSR feature columns (5 columns from ``DEFAULT_FEATURE_COLUMNS``);
    the integration wiring is via :func:`make_lambda_astsr_reward_channel`
    on the *measured reward* row, not on the 32-d lambda vector.  This
    test pins that contract.
    """
    from molmetal.molmetal_lam.reward.lambda_astsr import (
        DEFAULT_FEATURE_COLUMNS,
        LambdaASTSR,
        make_lambda_astsr_reward_channel,
    )

    astsr = LambdaASTSR(
        depth=2, population=4, generations=1, seed=0,
        feature_names=DEFAULT_FEATURE_COLUMNS,
    )
    ch = make_lambda_astsr_reward_channel(astsr=astsr, enabled=True)
    # A measured-reward row (5 fields) → scalar.
    row = {fn: 0.5 for fn in DEFAULT_FEATURE_COLUMNS}
    val = ch(row)
    assert isinstance(val, float)
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# 4 — End-to-end smoke (CPU, ≤ 30 s)
# ---------------------------------------------------------------------------
def test_full_pipeline_smoke(pipe_real) -> None:
    """End-to-end smoke on the real PlatinAI 5K checkpoint (CPU, ≤ 30 s)."""
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        smoke_test_full_pipeline,
    )

    t0 = time.perf_counter()
    ok = smoke_test_full_pipeline(seed_smiles="CCO", pocket_size=8)
    elapsed = time.perf_counter() - t0
    assert ok is True
    assert elapsed < 30.0, f"smoke took {elapsed:.2f}s (>30s)"


# ---------------------------------------------------------------------------
# 5 — Determinism (same seed → same output, bit-for-bit)
# ---------------------------------------------------------------------------
def test_pipeline_deterministic(pipe_stub) -> None:
    """Two sequential ``run_full_pipeline`` calls return identical dicts."""
    pocket = np.random.default_rng(0).standard_normal(8).astype(np.float32)
    a = pipe_stub.run_full_pipeline("c1ccccc1", pocket, n_steps=20)
    b = pipe_stub.run_full_pipeline("c1ccccc1", pocket, n_steps=20)
    np.testing.assert_array_equal(a["lambda_features"], b["lambda_features"])
    np.testing.assert_array_equal(a["lambda_embed"], b["lambda_embed"])
    np.testing.assert_array_equal(a["cfm_output"], b["cfm_output"])
    assert a["decoded_smiles"] == b["decoded_smiles"]
    # Even z0 must be identical because the seed is fixed.
    np.testing.assert_array_equal(a["z0"], b["z0"])


# ---------------------------------------------------------------------------
# 6 — Batch (batch_size > 1) — embed_lambda_features 2-d input
# ---------------------------------------------------------------------------
def test_pipeline_batch(pipe_stub) -> None:
    """Batch embed of (N, 32) returns (N, 64)."""
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        EMBED_DIM,
        LAMBDA_DIM,
    )

    N = 8
    X = np.random.default_rng(0).standard_normal((N, LAMBDA_DIM)).astype(np.float32)
    out = pipe_stub.embed_lambda_features(X)
    assert out.shape == (N, EMBED_DIM)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # Each row is independently L2-normalised.
    norms = np.linalg.norm(out, axis=1)
    assert np.all((norms >= 0.99) & (norms <= 1.01))


# ---------------------------------------------------------------------------
# 7 — Real PlatinAI 5K CSV smoke (first 5 samples)
# ---------------------------------------------------------------------------
def test_pipeline_real_platinai_csv(pipe_real, platinai_smiles) -> None:
    """Run the full pipeline on real PlatinAI 5K samples (CPU)."""
    pocket = np.zeros(8, dtype=np.float32)
    n_atoms_total = 0
    n_ok = 0
    for smi in platinai_smiles[:5]:
        out = pipe_real.run_full_pipeline(smi, pocket, n_steps=10)
        assert np.isfinite(out["lambda_features"]).all()
        assert np.isfinite(out["cfm_output"]).all()
        assert 0.99 <= float(np.linalg.norm(out["cfm_output"])) <= 1.01
        assert out["decoded_smiles"], f"no decode for {smi}"
        n_atoms_total += int(out["lambda_features"].shape[0])
        n_ok += 1
    assert n_ok == 5
    assert n_atoms_total == 5 * 32  # 32-d λ-features per sample


# ---------------------------------------------------------------------------
# 8 — Real pretrain checkpoint embedding contract (32-d → 64-d)
# ---------------------------------------------------------------------------
def test_pipeline_with_real_pretrain_checkpoint(pipe_real) -> None:
    """Real PlatinAI 5K checkpoint → finite, L2-normalised 64-d output."""
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        EMBED_DIM,
        PlatinAI5KBridge,
    )

    assert isinstance(pipe_real.bridge, PlatinAI5KBridge)
    assert pipe_real.bridge.source_path is not None
    x = np.random.default_rng(0).standard_normal(32).astype(np.float32)
    out = pipe_real.bridge.embed_raw(x)
    assert out.shape == (EMBED_DIM,)
    assert np.isfinite(out).all()
    norm = float(np.linalg.norm(out))
    assert 0.99 <= norm <= 1.01, f"norm={norm:.4f}"


# ---------------------------------------------------------------------------
# 9 — Metallo SMILES (Pt/Pd/Au/etc.) produce finite pipeline output
# ---------------------------------------------------------------------------
def test_pipeline_handles_metallo_smiles(pipe_stub) -> None:
    """Metallo SMILES with Pt/Pd/Au brackets produce finite pipeline output.

    These SMILES exercise the ``metal_indicator`` slot in the 32-d
    lambda feature vector.  We use small hand-crafted examples (real
    PlatinAI 5K SMILES are too large for a unit test) and verify the
    pipeline does not raise and emits finite values.
    """
    pocket = np.zeros(8, dtype=np.float32)
    metal_smiles = [
        "C1->[Pt](<-Cl)(<-Cl)<-N",  # Pt with two Cl ligands
        "[Pd](<-[Cl-])(<-[Cl-])",    # Pd square planar
        "[Au]<-[Cl-]",               # Au(I) chloride
    ]
    for smi in metal_smiles:
        out = pipe_stub.run_full_pipeline(smi, pocket, n_steps=5)
        assert np.isfinite(out["lambda_features"]).all(), smi
        assert np.isfinite(out["cfm_output"]).all(), smi
        assert out["decoded_smiles"], smi


# ---------------------------------------------------------------------------
# 10 — top_k=1 is the first entry of top_k=3
# ---------------------------------------------------------------------------
def test_pipeline_top_k_decode_consistent(pipe_stub) -> None:
    """``top_k=1`` returns the closest tile; ``top_k=3`` is its prefix."""
    cfm = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    cfm /= max(np.linalg.norm(cfm), 1e-8)
    one = pipe_stub.lambda_decode(cfm, top_k=1)
    three = pipe_stub.lambda_decode(cfm, top_k=3)
    assert len(one) == 1
    assert len(three) == 3
    assert one[0] == three[0], "top_k=1 must be top_k=3[0]"


# ---------------------------------------------------------------------------
# 11 — CascadedRefiner wired after the pipeline (Strategy 3)
# ---------------------------------------------------------------------------
def test_cascaded_refiner_after_pipeline(pipe_stub) -> None:
    """``CascadedRefiner.refine`` composes on the pipeline output."""
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        CascadedRefiner,
    )

    pocket = np.zeros(8, dtype=np.float32)
    out = pipe_stub.run_full_pipeline("c1ccccc1", pocket, n_steps=10)
    refiner = CascadedRefiner(n_refine_steps=10, blend=0.3)
    refined = refiner.refine(out["lambda_embed"], out["cfm_output"])
    assert refined.shape == (64,)
    assert np.isfinite(refined).all()
    norm = float(np.linalg.norm(refined))
    assert 0.99 <= norm <= 1.01, f"norm={norm:.4f}"


# ---------------------------------------------------------------------------
# 12 — Memory boundedness across N sequential runs (no leak)
# ---------------------------------------------------------------------------
def test_pipeline_memory_bounded(pipe_stub) -> None:
    """50 sequential runs of the full pipeline stay within a 50 MiB delta."""
    tracemalloc.start()
    pocket = np.zeros(8, dtype=np.float32)
    base = tracemalloc.get_traced_memory()[0]
    for _ in range(50):
        pipe_stub.run_full_pipeline("c1ccccc1", pocket, n_steps=10)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    delta_mb = (peak - base) / (1024 * 1024)
    # 50 MiB is generous — pure numpy allocations do not leak.
    assert delta_mb < 50.0, f"peak grew by {delta_mb:.1f} MiB"


# ---------------------------------------------------------------------------
# 13 — Round-trip: same SMILES yields the same 32-d features twice
# ---------------------------------------------------------------------------
def test_pipeline_smiles_round_trip_preserves_features(pipe_stub) -> None:
    """The same SMILES yields identical 32-d lambda features (round-trip)."""
    from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
        LAMBDA_DIM,
        lambda_features_from_smiles,
    )

    a = lambda_features_from_smiles("CCN=[N+]=[N-]")
    b = lambda_features_from_smiles("CCN=[N+]=[N-]")
    assert a.shape == (LAMBDA_DIM,)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 14 — JSON export of the pipeline output dict
# ---------------------------------------------------------------------------
def test_pipeline_writes_summary_json(pipe_stub, tmp_path: Path) -> None:
    """The pipeline output dict serialises to JSON (numpy-safe)."""
    pocket = np.zeros(8, dtype=np.float32)
    out = pipe_stub.run_full_pipeline("CCO", pocket, n_steps=10)

    # Drop the numpy arrays; keep scalars + string lists.
    summary: Dict[str, object] = {
        "smiles_seed": out["smiles_seed"],
        "n_steps": int(out["n_steps"]),  # type: ignore[arg-type]
        "pocket_size": int(out["pocket_size"]),  # type: ignore[arg-type]
        "use_lambda_astsr": bool(out["use_lambda_astsr"]),
        "decoded_smiles": list(out["decoded_smiles"]),  # type: ignore[arg-type]
        "lambda_features_mean": float(np.mean(out["lambda_features"])),  # type: ignore[arg-type]
        "cfm_output_mean": float(np.mean(out["cfm_output"])),  # type: ignore[arg-type]
        "cfm_output_norm": float(np.linalg.norm(out["cfm_output"])),  # type: ignore[arg-type]
    }
    target = tmp_path / "summary.json"
    target.write_text(json.dumps(summary))
    loaded = json.loads(target.read_text())
    assert loaded["smiles_seed"] == "CCO"
    assert loaded["use_lambda_astsr"] is True
    assert loaded["n_steps"] == 10
    assert isinstance(loaded["decoded_smiles"], list)
    assert 0.99 <= loaded["cfm_output_norm"] <= 1.01


# ---------------------------------------------------------------------------
# 15 — alpha endpoints (alpha=0 → pure pocket; alpha=1 → pure lambda)
# ---------------------------------------------------------------------------
def test_pipeline_alpha_extremes(pipe_stub) -> None:
    """``alpha=0`` reduces the CFM target to ``pocket``; ``alpha=1``
    reduces it to the bridge-embedded lambda vector (up to ``t=0``).
    """
    rng = np.random.default_rng(0)
    z0 = rng.standard_normal(64).astype(np.float32) * 0.1
    lam = rng.standard_normal(32).astype(np.float32)
    pocket = rng.standard_normal(64).astype(np.float32)

    # alpha=1: target is the lambda embed only.
    pipe_lam = type(pipe_stub)(bridge=pipe_stub.bridge, use_lambda_astsr=True,
                                alpha=1.0, seed=0)
    # alpha=0: target is the pocket only.
    pipe_pkt = type(pipe_stub)(bridge=pipe_stub.bridge, use_lambda_astsr=True,
                                alpha=0.0, seed=0)
    out_lam = pipe_lam.cfm_inference(z0, lam, pocket, n_steps=200)
    out_pkt = pipe_pkt.cfm_inference(z0, lam, pocket, n_steps=200)
    assert np.isfinite(out_lam).all()
    assert np.isfinite(out_pkt).all()
    # Both endpoints lie on the unit sphere.
    assert 0.99 <= float(np.linalg.norm(out_lam)) <= 1.01
    assert 0.99 <= float(np.linalg.norm(out_pkt)) <= 1.01
    # And the two endpoints are *distinct* (the lambda and pocket
    # vectors point in different directions).
    cos = float(
        np.dot(out_lam, out_pkt)
        / (float(np.linalg.norm(out_lam)) * float(np.linalg.norm(out_pkt)) + 1e-8)
    )
    assert cos < 0.999, f"alpha endpoints converged unexpectedly (cos={cos:.4f})"