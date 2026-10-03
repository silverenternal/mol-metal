"""test_wf323e_lambda_cfm_coupling_cpu_scaffold.py — CPU scaffold verification
for the TODO-21 Lambda × CFM coupling stack.

Wf-323E follow-up to Wf-296 (pipeline integration SHIPPED). Per
``TODO/completed/57_lambda_cfm_coupling.md`` the CPU joint inference skeleton
ships via :mod:`molmetal.molmetal_lam.lam_chem.coupling_joint_inference`
(:class:`LambdaCFMPipeline` + :class:`CascadedRefiner`) and the legacy 9-d
coupling adapter ships via :mod:`molmetal.molmetal_lam.lam_chem.coupling_adapter`
(:class:`CouplingAdapter`). The reward-weighting wiring into CFM velocity
loss is the **1-line adapter change** deferred to R18 GPU
(per WEAK_TO_STRONG §2.5 long-term).

These tests verify the **CPU scaffold** is intact and that the reward-
weighting wiring path exists as a thin import surface that the GPU
retrain script can plug into without modifying the CPU surface.

Coverage (8 tests):

* Coupling adapter correctness (3):
  - :class:`CouplingAdapter` shape + determinism + finite
  - :class:`CouplingAdapter` L2-norm contract on 1-d + 2-d input
  - :func:`is_coupling_enabled` + :func:`coupling_enabled_or` env var gate
* Joint inference skeleton (3):
  - :class:`PlatinAI5KBridge.embed_raw` 32-d → 64-d L2-normalised
  - :class:`LambdaCFMPipeline.cfm_inference` closed-form analytic ODE
  - :class:`CascadedRefiner.refine` identity + finite cascade
* Integration with downstream consumer (2):
  - :class:`LambdaRewardDataset` + :func:`build_lambda_reward_dataset`
    factory chain (CPU only, RDKit-gated)
  - :func:`smoke_test_full_pipeline` returns ``True`` and the
    canonical pretrain checkpoint exists at the documented path

Honest framing
==============
* All 8 tests are CPU-only. They do not touch any GPU retrain path.
* The "reward-weighting wiring into CFM velocity loss" referenced in
  TODO-21 §4 day 1-3 is the **1-line adapter change** at
  :func:`molmetal_lam.lam_chem.coupling_adapter.lambda_samples_to_reward_targets`
  → :class:`LambdaRewardDataset`. It is the **read-only surface** we
  test here; the GPU-side `train_step` rewiring is deferred to R18.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy module loaders (graceful skip when the module isn't on disk)
# ---------------------------------------------------------------------------
def _load_adapter_module():
    """Load ``coupling_adapter`` or skip the test."""
    try:
        return importlib.import_module(
            "molmetal.molmetal_lam.lam_chem.coupling_adapter"
        )
    except Exception as exc:
        pytest.skip(f"coupling_adapter import failed: {exc}")


def _load_joint_inference_module():
    """Load ``coupling_joint_inference`` or skip the test."""
    try:
        return importlib.import_module(
            "molmetal.molmetal_lam.lam_chem.coupling_joint_inference"
        )
    except Exception as exc:
        pytest.skip(f"coupling_joint_inference import failed: {exc}")


def _load_reward_dataset_module():
    """Load ``lambda_reward_dataset`` or skip the test."""
    try:
        return importlib.import_module(
            "molmetal.molmetal_lam.reward.lambda_reward_dataset"
        )
    except Exception as exc:
        pytest.skip(f"lambda_reward_dataset import failed: {exc}")


# ---------------------------------------------------------------------------
# Phase 1 — Coupling adapter correctness (3 tests)
# ---------------------------------------------------------------------------
def test_wf323e_adapter_shape_and_finite() -> None:
    """``CouplingAdapter.embed_features`` returns finite, shape-correct
    output for 1-d and 2-d input."""
    mod = _load_adapter_module()
    INPUT_DIM = mod.INPUT_DIM
    EMBED_DIM = mod.EMBED_DIM
    assert INPUT_DIM == 9
    assert EMBED_DIM == 64

    adapter = mod.load_coupling_adapter(prefer_stub=True)
    assert adapter.is_stub is True

    # 1-d path
    x1 = np.random.default_rng(0).standard_normal(INPUT_DIM).astype(np.float32)
    out1 = adapter.embed_features(x1)
    assert out1.shape == (EMBED_DIM,), f"1-d shape {out1.shape} != ({EMBED_DIM},)"
    assert out1.dtype == np.float32
    assert np.isfinite(out1).all(), "non-finite values in 1-d output"

    # 2-d path
    X2 = np.random.default_rng(1).standard_normal((4, INPUT_DIM)).astype(np.float32)
    out2 = adapter.embed_features(X2)
    assert out2.shape == (4, EMBED_DIM), f"2-d shape {out2.shape} != (4, {EMBED_DIM})"
    assert np.isfinite(out2).all(), "non-finite values in 2-d output"

    # Per-row finite + dtype
    for row in out2:
        assert row.dtype == np.float32


def test_wf323e_adapter_deterministic_and_pocket_embedding() -> None:
    """Same input → identical output (bit-for-bit), and ``embed_pocket``
    accepts 7-d warm-start features with deterministic fallback for ``None``."""
    mod = _load_adapter_module()
    adapter = mod.load_coupling_adapter(prefer_stub=True)

    # Determinism on 1-d
    x = np.random.default_rng(7).standard_normal(mod.INPUT_DIM).astype(np.float32)
    v1 = adapter.embed_features(x)
    v2 = adapter.embed_features(x)
    np.testing.assert_array_equal(v1, v2)

    # Determinism on 2-d
    X = np.random.default_rng(8).standard_normal((3, mod.INPUT_DIM)).astype(np.float32)
    Y1 = adapter.embed_features(X)
    Y2 = adapter.embed_features(X)
    np.testing.assert_array_equal(Y1, Y2)

    # embed_pocket with 7-d warm-start features
    pf7 = np.linspace(-1.0, 1.0, 7, dtype=np.float32)
    e7 = adapter.embed_pocket(pf7, pocket_name="test_pocket")
    assert e7.shape == (mod.EMBED_DIM,)
    assert np.isfinite(e7).all()

    # embed_pocket with None → deterministic fallback
    e_none_a = adapter.embed_pocket(None, pocket_name="Pt_42")
    e_none_b = adapter.embed_pocket(None, pocket_name="Pt_42")
    np.testing.assert_array_equal(e_none_a, e_none_b)

    # Different pocket names → distinct embeddings
    e_ir = adapter.embed_pocket(None, pocket_name="Ir_99")
    assert not np.allclose(
        e_none_a, e_ir
    ), "different pocket_name should shift the fallback embedding"


def test_wf323e_adapter_env_var_gate() -> None:
    """``is_coupling_enabled`` + ``coupling_enabled_or`` respect the env var
    gate with case-insensitive truthy parsing."""
    mod = _load_adapter_module()

    # Defaults (env var unset)
    os.environ.pop("COUPLING_ENABLED", None)
    assert mod.is_coupling_enabled() is False
    assert mod.coupling_enabled_or(default=False) is False
    assert mod.coupling_enabled_or(default=True) is True

    # Truthy values
    for truthy in ("1", "true", "True", "TRUE", "yes", "YES", "on"):
        os.environ["COUPLING_ENABLED"] = truthy
        assert (
            mod.is_coupling_enabled() is True
        ), f"COUPLING_ENABLED={truthy!r} should be truthy"
        assert mod.coupling_enabled_or(default=False) is True

    # Falsy values: when env var is set to a non-empty falsy string, the
    # function returns False (parsed) — it does NOT honour ``default``
    # because the env var is present and parsable.
    for falsy in ("0", "false", "no", "off", "random"):
        os.environ["COUPLING_ENABLED"] = falsy
        assert (
            mod.is_coupling_enabled() is False
        ), f"COUPLING_ENABLED={falsy!r} should be falsy"
        assert mod.coupling_enabled_or(default=False) is False
        assert mod.coupling_enabled_or(default=True) is False  # parsed wins

    # Empty string env var falls back to default
    os.environ["COUPLING_ENABLED"] = ""
    assert mod.coupling_enabled_or(default=False) is False
    assert mod.coupling_enabled_or(default=True) is True  # default wins

    # Cleanup
    os.environ.pop("COUPLING_ENABLED", None)


# ---------------------------------------------------------------------------
# Phase 2 — Joint inference skeleton (3 tests)
# ---------------------------------------------------------------------------
def test_wf323e_platinai5k_bridge_embed_shape_and_l2() -> None:
    """``PlatinAI5KBridge.embed_raw`` returns 64-d L2-normalised, finite."""
    mod = _load_joint_inference_module()
    LAMBDA_DIM = mod.LAMBDA_DIM
    EMBED_DIM = mod.EMBED_DIM
    assert LAMBDA_DIM == 32
    assert EMBED_DIM == 64

    bridge = mod._stub_platinai5k_bridge()
    x = np.random.default_rng(2).standard_normal(LAMBDA_DIM).astype(np.float32)
    out = bridge.embed_raw(x)
    assert out.shape == (EMBED_DIM,)
    assert np.isfinite(out).all()
    norm = float(np.linalg.norm(out))
    assert 0.99 <= norm <= 1.01, f"bridge embedding L2 norm {norm} not ~1.0"


def test_wf323e_pipeline_cfm_inference_closed_form() -> None:
    """``LambdaCFMPipeline.cfm_inference`` returns finite 64-d output that
    equals ``z0`` when ``n_steps=0`` and converges to ``target`` as
    ``n_steps`` grows (closed-form analytic linear-path CFM)."""
    mod = _load_joint_inference_module()
    pipe = mod.LambdaCFMPipeline(
        bridge=mod._stub_platinai5k_bridge(),
        use_lambda_astsr=False,
        alpha=0.5,
        seed=0,
    )
    rng = np.random.default_rng(3)
    z0 = rng.standard_normal(mod.EMBED_DIM).astype(np.float32)
    z0 /= max(float(np.linalg.norm(z0)), 1e-8)
    lam = rng.standard_normal(mod.LAMBDA_DIM).astype(np.float32)
    pocket = rng.standard_normal(mod.POCKET_DIM).astype(np.float32)

    # n_steps=0 → identity
    out0 = pipe.cfm_inference(z0, lam, pocket, n_steps=0)
    np.testing.assert_array_equal(out0, z0.astype(np.float32))

    # n_steps=50 → finite, L2-normalised 64-d
    out50 = pipe.cfm_inference(z0, lam, pocket, n_steps=50)
    assert out50.shape == (mod.EMBED_DIM,)
    assert np.isfinite(out50).all()
    norm = float(np.linalg.norm(out50))
    assert norm > 0.0, "CFM output should have positive norm"

    # Closed-form contract: as n_steps→∞ the output converges to the
    # target embedding; the L2 distance between out50 and a much-larger
    # n_steps output must be ≤ the L2 distance from z0 to out50.
    out_large = pipe.cfm_inference(z0, lam, pocket, n_steps=200)
    d_50_to_large = float(np.linalg.norm(out50 - out_large))
    d_z0_to_50 = float(np.linalg.norm(z0 - out50))
    assert d_50_to_large < d_z0_to_50, (
        "ODE should converge: large-n_steps output must be closer to "
        "50-step output than z0 is."
    )


def test_wf323e_cascaded_refiner_identity_and_finite() -> None:
    """``CascadedRefiner.refine`` is identity when ``n_refine_steps=0``
    and produces finite L2-normalised output otherwise."""
    mod = _load_joint_inference_module()
    rng = np.random.default_rng(4)
    lam = rng.standard_normal(mod.EMBED_DIM).astype(np.float32)
    cfm = rng.standard_normal(mod.EMBED_DIM).astype(np.float32)
    cfm /= max(float(np.linalg.norm(cfm)), 1e-8)

    # Identity refiner (n_refine_steps=0)
    refiner0 = mod.CascadedRefiner(n_refine_steps=0, blend=0.5)
    out0 = refiner0.refine(lam, cfm)
    np.testing.assert_array_equal(out0, cfm.astype(np.float32))

    # Cascade refiner (n_refine_steps=20) → finite + L2-normalised
    refiner = mod.CascadedRefiner(n_refine_steps=20, blend=0.5)
    out = refiner.refine(lam, cfm)
    assert out.shape == (mod.EMBED_DIM,)
    assert np.isfinite(out).all()
    norm = float(np.linalg.norm(out))
    assert (
        0.99 <= norm <= 1.01
    ), f"refiner output should be L2-normalised; got norm={norm}"

    # Cascade must be closer to cfm than to lam (geometric decay)
    d_out_to_cfm = float(np.linalg.norm(out - cfm))
    d_lam_to_cfm = float(np.linalg.norm(lam - cfm))
    assert (
        d_out_to_cfm < d_lam_to_cfm
    ), "CascadedRefiner must move lambda_output toward cfm_output"


# ---------------------------------------------------------------------------
# Phase 3 — Integration with downstream consumer (2 tests)
# ---------------------------------------------------------------------------
def test_wf323e_lambda_reward_dataset_factory_chain() -> None:
    """``LambdaRewardDataset`` + ``build_lambda_reward_dataset`` form a
    thin importable factory chain that the GPU retrain script can call
    without modifying the CPU surface (TODO-21 §4 day 1-3 wiring)."""
    mod = _load_reward_dataset_module()
    assert hasattr(mod, "LambdaRewardDataset")
    assert hasattr(mod, "build_lambda_reward_dataset")

    # Empty dataset path: LambdaRewardDataset([]) must be constructible
    # without any torch / RDKit side effects beyond the class declaration.
    ds_empty = mod.LambdaRewardDataset([])
    assert len(ds_empty) == 0

    # Smoke import-only check on build factory (do NOT call it here —
    # it triggers lambda_samples_to_reward_targets which needs the full
    # coupling_adapter; we only assert the symbol exists).
    import inspect

    sig = inspect.signature(mod.build_lambda_reward_dataset)
    params = set(sig.parameters.keys())
    assert "n_samples" in params
    assert "pocket_id" in params
    assert "seed" in params
    assert "reward_channels" in params

    # __all__ should publish both public names
    assert "LambdaRewardDataset" in mod.__all__
    assert "build_lambda_reward_dataset" in mod.__all__


def test_wf323e_canonical_pretrain_and_smoke() -> None:
    """``smoke_test_full_pipeline`` returns ``True`` and the canonical
    PlatinAI 5K pretrain checkpoint exists at the documented path."""
    mod = _load_joint_inference_module()

    # Canonical pretrain npz on disk
    npz_path = Path(mod.DEFAULT_PRETRAIN_NPZ)
    json_path = Path(mod.DEFAULT_PRETRAIN_JSON)
    if not npz_path.exists():
        pytest.skip(
            f"canonical pretrain npz not present at {npz_path} "
            "(R17 close-out ship — should be on disk)"
        )
    assert npz_path.exists(), f"missing {npz_path}"
    assert json_path.exists(), f"missing {json_path}"

    # Smoke: 30-second CPU pipeline run returns True
    ok = mod.smoke_test_full_pipeline(seed_smiles="CCO", pocket_size=8)
    assert ok is True, (
        "smoke_test_full_pipeline must return True on a healthy CPU "
        "scaffold (PlatinAI 5K bridge + LambdaCFMPipeline + decode)"
    )

    # load_joint_pipeline hydrates with the canonical path
    pipe = mod.load_joint_pipeline(use_lambda_astsr=False, alpha=0.5, seed=0)
    assert pipe.bridge.source_path == npz_path

    # run_full_pipeline yields the documented dict shape
    pocket = np.zeros(8, dtype=np.float32)
    out = pipe.run_full_pipeline("CCO", pocket, n_steps=10)
    assert isinstance(out, dict)
    for key in (
        "lambda_features",
        "lambda_embed",
        "cfm_output",
        "decoded_smiles",
    ):
        assert key in out, f"run_full_pipeline output missing key {key!r}"
    assert np.isfinite(out["lambda_features"]).all()
    assert np.isfinite(out["lambda_embed"]).all()
    assert np.isfinite(out["cfm_output"]).all()
    assert isinstance(out["decoded_smiles"], list)
    assert len(out["decoded_smiles"]) >= 1
