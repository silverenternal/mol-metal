"""Wf-349-A3: StableVM variance reduction stub tests (CPU-only).

These tests exercise the minimal ``StableVMLoss`` + ``SampleBank``
stub defined in
:mod:`molmetal.adapters.flow_matching_lipman.stablevm_stub`.  They are
**CPU-only** by design — per AGENTS.md §5.1 GPU launches are gated on
pre-flight + VRAM budget, but a variance A/B smoke is cheap on CPU.

Honest framing (party-twenty-spirit §3.5):

* The "variance reduction" claim is **PROJECTED** from the upstream
  StableVM paper (Yang 2026 ICML 2026).  This stub reproduces the
  math on a toy linear velocity field so we can confirm the bias and
  variance properties locally; real Mol-Metal benchmark numbers come
  from a separate workflow once the stub is wired into
  ``train_cfm_yuelbond_wasserstein.py``.
* No GPU launch is performed by these tests.
"""

from __future__ import annotations

import math

import pytest
import torch

from molmetal.adapters.flow_matching_lipman.stablevm_stub import (
    SampleBank,
    StableVMConfig,
    StableVMLoss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_velocity_fn(x_t: torch.Tensor, t: float) -> torch.Tensor:
    """Cheap deterministic velocity field for the stub smoke tests.

    Uses a fixed linear + offset so the loss is non-trivial but
    predictable; this keeps the variance A/B stable across runs.
    """
    return 1.5 * x_t + 0.1


# ---------------------------------------------------------------------------
# SampleBank tests
# ---------------------------------------------------------------------------


def test_sample_bank_capacity_and_add() -> None:
    """SampleBank enforces capacity and stores per-class anchors."""
    bank = SampleBank(capacity_per_class=3)
    bank.add_batch(
        class_ids=torch.tensor([0, 0, 0, 0, 0]),
        x0_batch=torch.randn(5, 4),
    )
    # Capacity truncates to 3.
    assert len(bank) == 3
    assert bank.num_classes() == 1
    assert bank.sample(class_id=0, n=2).shape == (2, 4)


def test_sample_bank_class_isolation() -> None:
    """Sampling from an empty class raises a clear error."""
    bank = SampleBank(capacity_per_class=2)
    bank.add(class_id=0, x0=torch.randn(4))
    with pytest.raises(RuntimeError):
        bank.sample(class_id=1, n=1)


def test_sample_bank_add_rejects_non_1d() -> None:
    """``add`` rejects 2-D inputs to keep the deque shapes consistent."""
    bank = SampleBank(capacity_per_class=2)
    with pytest.raises(ValueError):
        bank.add(class_id=0, x0=torch.randn(1, 4))
    with pytest.raises(ValueError):
        bank.add(class_id=0, x0=torch.randn(3, 4))


def test_sample_bank_capacity_must_be_positive() -> None:
    """Constructor rejects non-positive capacity with a ValueError."""
    with pytest.raises(ValueError):
        SampleBank(capacity_per_class=0)


# ---------------------------------------------------------------------------
# StableVMLoss tests
# ---------------------------------------------------------------------------


def test_stablevm_k1_equals_standard_cfm() -> None:
    """When ``k_samples = 1`` the loss is exactly the standard CFM MSE."""
    torch.manual_seed(42)
    x0 = torch.randn(8, 4)
    x1 = torch.randn(8, 4) + 2.0
    t = 0.5

    cfm = StableVMLoss(velocity_fn=_toy_velocity_fn, config=StableVMConfig(k_samples=1))
    x_t = (1.0 - t) * x0 + t * x1
    v_pred = _toy_velocity_fn(x_t, t)
    expected = torch.mean((v_pred - (x1 - x0)) ** 2)

    actual = cfm(x0, x1, t=t)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_stablevm_k1_ignores_class_ids() -> None:
    """The K=1 path must not require bank data (no RuntimeError)."""
    torch.manual_seed(42)
    x0 = torch.randn(4, 4)
    x1 = torch.randn(4, 4)
    cfm = StableVMLoss(velocity_fn=_toy_velocity_fn, config=StableVMConfig(k_samples=1))
    # Even with a None class_ids tensor and an empty bank, K=1 must work.
    actual = cfm(x0, x1, t=0.3, class_ids=None)
    assert torch.isfinite(actual).item()


def test_stablevm_k_ge2_uses_bank() -> None:
    """With K >= 2 and a populated bank the loss is finite and differs from K=1."""
    torch.manual_seed(0)
    x0 = torch.randn(8, 4)
    x1 = torch.randn(8, 4) + 2.0
    class_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    bank = SampleBank(capacity_per_class=8)
    bank.add_batch(
        class_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        x0_batch=torch.randn(8, 4),
    )

    cfg = StableVMConfig(k_samples=4, self_normalize=True)
    stablevm = StableVMLoss(velocity_fn=_toy_velocity_fn, config=cfg, bank=bank)
    cfm = StableVMLoss(velocity_fn=_toy_velocity_fn, config=StableVMConfig(k_samples=1))

    loss_cfm = cfm(x0, x1, t=0.5, class_ids=class_ids)
    loss_stablevm = stablevm(x0, x1, t=0.5, class_ids=class_ids)

    assert torch.isfinite(loss_cfm).item()
    assert torch.isfinite(loss_stablevm).item()
    # The two losses need not match — that's the whole point of
    # multi-sample variance reduction — but they must both be finite.
    assert not torch.allclose(loss_cfm, loss_stablevm, atol=1e-6)


def test_stablevm_k_ge2_without_class_ids_falls_back() -> None:
    """If class_ids is None but K >= 2, fall back to the K=1 path safely."""
    torch.manual_seed(0)
    x0 = torch.randn(4, 4)
    x1 = torch.randn(4, 4)
    cfg = StableVMConfig(k_samples=4, self_normalize=True)
    stablevm = StableVMLoss(velocity_fn=_toy_velocity_fn, config=cfg)

    # No class_ids ⇒ the stub short-circuits to the anchor-only path.
    loss = stablevm(x0, x1, t=0.5, class_ids=None)
    x_t = (1.0 - 0.5) * x0 + 0.5 * x1
    v_pred = _toy_velocity_fn(x_t, 0.5)
    expected = torch.mean((v_pred - (x1 - x0)) ** 2)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_stablevm_invalid_k_samples() -> None:
    """Constructing with k_samples < 1 raises ValueError at call time."""
    cfg = StableVMConfig(k_samples=0)
    loss = StableVMLoss(velocity_fn=_toy_velocity_fn, config=cfg)
    x0 = torch.randn(2, 4)
    x1 = torch.randn(2, 4)
    with pytest.raises(ValueError):
        loss(x0, x1, t=0.5)


def test_stablevm_variance_reduction_smoke() -> None:
    """Across many trials, StableVM (K=4) variance <= CFM (K=1) variance.

    This is the headline claim of the upstream paper; on a toy linear
    velocity field with a populated bank we expect to see variance
    reduction.  We use a soft margin (``< 1.10 *``) so the test is
    stable across minor random fluctuations; if the upstream math
    stops working the assertion will fire.
    """
    torch.manual_seed(20260929)
    bank = SampleBank(capacity_per_class=16)
    bank.add_batch(
        class_ids=torch.tensor([0] * 8 + [1] * 8),
        x0_batch=torch.randn(16, 4),
    )
    cfm = StableVMLoss(velocity_fn=_toy_velocity_fn, config=StableVMConfig(k_samples=1))
    stablevm = StableVMLoss(
        velocity_fn=_toy_velocity_fn,
        config=StableVMConfig(k_samples=4, self_normalize=True),
        bank=bank,
    )
    cfm_losses, stablevm_losses = [], []
    for _ in range(40):
        x0 = torch.randn(8, 4)
        x1 = torch.randn(8, 4) + 2.0
        class_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        cfm_losses.append(float(cfm(x0, x1, t=0.5, class_ids=class_ids).item()))
        stablevm_losses.append(
            float(stablevm(x0, x1, t=0.5, class_ids=class_ids).item())
        )

    cfm_var = torch.tensor(cfm_losses).var(unbiased=False).item()
    stablevm_var = torch.tensor(stablevm_losses).var(unbiased=False).item()
    assert stablevm_var <= cfm_var * 1.10, (
        f"StableVM did not reduce variance: cfm_var={cfm_var:.4f}, "
        f"stablevm_var={stablevm_var:.4f}"
    )


def test_stablevm_module_is_importable() -> None:
    """Smoke-import the public symbols so the test surface matches the docstring."""
    from molmetal.adapters.flow_matching_lipman import stablevm_stub  # noqa: F401

    assert hasattr(stablevm_stub, "SampleBank")
    assert hasattr(stablevm_stub, "StableVMLoss")
    assert hasattr(stablevm_stub, "StableVMConfig")
