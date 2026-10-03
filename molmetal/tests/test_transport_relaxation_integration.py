"""Pytest: W2.2 — Transport-Relaxation reward-shaping adapter.

This test module pins the W2.2 contract end-to-end:

1. :class:`RelaxedEndpointTerm` produces a non-negative KL scalar.
2. The KL is exactly zero when ``λ_relax = 0`` (no-op contract).
3. The KL is exactly zero when all energies are equal (Gibbs-
   minimum contract).
4. :func:`augment_learned_shaping` returns a tensor with the same
   shape and dtype as the base shaping output.
5. :class:`LambdaRelaxSweep` returns one of the grid lambdas.
6. The grid-evaluation log is well-formed and reproducible.
7. Importing the adapter does NOT mutate
   ``molmetal.adapters.flow_matching_lipman`` (architectural
   contract — the R17 close-out is locked).

All tests run on CPU in <5 s — no GPU / external dependencies
required.  The tests do NOT touch the trained EGNN checkpoint or
any reward-fitting machinery; the base shaping callable is a
trivial closure for test isolation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _constant_shaping(value: float = 0.0):
    """Trivial base-shaping closure that ignores its input.

    Mirrors the F5 / ASTSR signature ``(row_dict) -> float`` without
    pulling in :mod:`molmetal.molmetal_lam.reward.learned_shaping`
    (which would re-run ``register_lambda_astsr`` side effects in
    the test process).  Used by the augmentation + sweep tests.
    """
    def _closure(row: dict) -> float:
        return float(value)
    return _closure


def _toy_trajectory(b: int = 4, t: int = 5, d: int = 3, seed: int = 0) -> torch.Tensor:
    """Deterministic (B, T, D) trajectory for tests."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(b, t, d, generator=g)


def _toy_energies(b: int = 4, seed: int = 0) -> torch.Tensor:
    """Deterministic (B,) energy vector for tests."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(b, generator=g)


# ---------------------------------------------------------------------------
# Test 1: RelaxedEndpointTerm — λ = 0 → identically zero
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_zero_lambda():
    """When ``λ_relax = 0`` the penalty is identically zero.

    This is the W2.2 "no-op" contract: the augmentation must
    preserve the base shaping exactly when the relaxation weight
    is zero.  Pinned as a regression test so any future change
    that adds a non-zero baseline fails CI.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=0.0, eps=0.1)
    trajectory = _toy_trajectory(b=4, t=5, d=3, seed=42)
    energies = _toy_energies(b=4, seed=42)
    out = term.forward(trajectory, energies)
    assert out.shape == (), f"penalty must be a scalar, got {tuple(out.shape)}"
    assert out.item() == 0.0, f"penalty must be zero when λ=0, got {out.item()}"


# ---------------------------------------------------------------------------
# Test 2: KL is non-negative (Gibbs' inequality)
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_kl_non_negative():
    """The KL term ``kl(p_T || q_T)`` is non-negative for any input.

    This pins the Gibbs' inequality contract — the penalty can
    only ever *add* cost to the shaping, never subtract.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=1.0, eps=0.1)
    for seed in range(5):
        trajectory = _toy_trajectory(b=8, t=4, d=3, seed=seed)
        energies = _toy_energies(b=8, seed=seed)
        kl = term.kl(trajectory, energies)
        assert kl.item() >= -1e-6, (
            f"KL must be >= 0; got {kl.item()} at seed={seed}"
        )


# ---------------------------------------------------------------------------
# Test 3: KL is exactly zero when energies are equal (Gibbs minimum)
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_kl_zero_for_equal_energies():
    """KL is exactly 0 when all energies are equal (``q_T`` is uniform).

    Gibbs' inequality gives ``KL = 0`` iff ``p_T = q_T``; uniform
    energies make ``q_T`` uniform, and ``p_T`` is uniform by
    construction, so the contract holds.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=1.0, eps=0.1)
    trajectory = _toy_trajectory(b=6, t=4, d=3)
    energies = torch.full((6,), 0.5)  # equal energies → uniform q_T
    kl = term.kl(trajectory, energies)
    assert abs(kl.item()) < 1e-6, (
        f"KL must be ~0 when energies are equal; got {kl.item()}"
    )


# ---------------------------------------------------------------------------
# Test 4: forward = lambda_relax * kl
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_forward_scales_with_lambda():
    """``forward = λ · kl`` (the W2.2 spec's exact contract).

    We verify the scaling by computing ``forward`` at two values
    of ``λ`` and checking that the ratio equals the ratio of
    the ``λ``s (within numerical tolerance).
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    trajectory = _toy_trajectory(b=5, t=3, d=3, seed=1)
    energies = _toy_energies(b=5, seed=1)
    kl_value = RelaxedEndpointTerm(lambda_relax=1.0, eps=0.1).kl(
        trajectory, energies,
    ).item()
    lam1 = 1e-3
    lam2 = 5e-3
    out1 = RelaxedEndpointTerm(lambda_relax=lam1, eps=0.1).forward(
        trajectory, energies,
    ).item()
    out2 = RelaxedEndpointTerm(lambda_relax=lam2, eps=0.1).forward(
        trajectory, energies,
    ).item()
    assert abs(out1 - lam1 * kl_value) < 1e-6
    assert abs(out2 - lam2 * kl_value) < 1e-6
    assert abs(out2 / out1 - lam2 / lam1) < 1e-3


# ---------------------------------------------------------------------------
# Test 5: KL is finite for large batch / small ε
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_finite_for_small_eps():
    """KL is finite even at ``ε = 1e-3`` and ``B = 32``.

    Pins the numerical-stability contract: the fp32-stable log
    form (rather than the textbook ``p log(p/q)`` form) must
    avoid ``log(0)`` overflow even when ``q`` becomes delta-
    like at small ``ε``.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=1e-3, eps=1e-3)
    g = torch.Generator().manual_seed(123)
    energies = torch.randn(32, generator=g) * 10.0  # large spread
    trajectory = torch.randn(32, 4, 3, generator=g)
    out = term.forward(trajectory, energies)
    assert torch.isfinite(out), f"penalty must be finite, got {out.item()}"
    assert out.item() >= -1e-6, f"penalty must be >= 0, got {out.item()}"


# ---------------------------------------------------------------------------
# Test 6: KL is bounded by log(B)
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_kl_finite_at_any_eps():
    """``KL(p ‖ q)`` is finite for any finite ``ε > 0``.

    The KL between a uniform ``p`` and any other distribution
    ``q`` is finite when ``q_i > 0`` everywhere; the softmax
    form we use keeps every ``q_i > 0`` whenever ``ε > 0`` and
    ``B`` is finite.  The test pins this finiteness contract on
    a wide energy spread.

    Notes
    -----
    The KL is **not** bounded above by ``log B`` — when ``q`` is
    near-delta, the KL grows without bound as ``ε → 0`` (this is
    the standard divergence of the reverse KL as ``q`` becomes
    a Dirac on a single element).  The right invariant is
    *finiteness for any finite ``ε > 0``*, which is what we
    check here.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    b = 16
    for eps_val in (1e-2, 1e-4, 1e-6):
        term = RelaxedEndpointTerm(lambda_relax=1.0, eps=eps_val)
        trajectory = _toy_trajectory(b=b, t=3, d=3)
        energies = torch.full((b,), 0.0)
        energies[0] = -50.0  # concentrated q on index 0
        kl = term.kl(trajectory, energies)
        assert torch.isfinite(kl), (
            f"KL must be finite for ε = {eps_val}; got {kl.item()}"
        )
        assert kl.item() >= 0.0, (
            f"KL must be >= 0 for ε = {eps_val}; got {kl.item()}"
        )


# ---------------------------------------------------------------------------
# Test 7: trajectory may be (B, D) (no time axis)
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_accepts_b_d_trajectory():
    """``trajectory`` may be ``(B, D)`` (no time axis) — the adapter
    uses the full tensor as the endpoint distribution."""
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)
    trajectory = torch.randn(4, 3)
    energies = _toy_energies(b=4)
    kl = term.kl(trajectory, energies)
    assert torch.isfinite(kl), "KL must be finite for (B, D) input"


# ---------------------------------------------------------------------------
# Test 8: energies may be (B, T) — time-mean is used
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_accepts_b_t_energies():
    """``energies`` may be ``(B, T)`` — the time-mean is the per-trajectory
    statistic that forms ``q_T``."""
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)
    trajectory = _toy_trajectory(b=4, t=5, d=3)
    energies = torch.randn(4, 5)
    kl = term.kl(trajectory, energies)
    assert torch.isfinite(kl), "KL must be finite for (B, T) energies"


# ---------------------------------------------------------------------------
# Test 9: NaN energies are repaired (replaced with batch mean)
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_nan_energies_repaired():
    """NaN / ±inf energies are replaced with the batch mean so the
    softmax remains well-defined.  This pins the NaN-safe contract.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)
    trajectory = _toy_trajectory(b=5, t=3, d=3)
    energies = torch.tensor([0.1, float("nan"), 0.3, float("inf"), 0.2])
    out = term.forward(trajectory, energies)
    assert torch.isfinite(out), (
        f"penalty must be finite even with NaN/inf energies; got {out.item()}"
    )


# ---------------------------------------------------------------------------
# Test 10: Constructor rejects negative λ
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_rejects_negative_lambda():
    """``__init__`` rejects ``λ_relax < 0`` with ``ValueError``."""
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    with pytest.raises(ValueError):
        RelaxedEndpointTerm(lambda_relax=-1e-3, eps=0.1)


# ---------------------------------------------------------------------------
# Test 11: Constructor rejects non-positive ε
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_rejects_nonpositive_eps():
    """``__init__`` rejects ``eps < 0`` with ``ValueError``.

    ``eps = 0`` is internally floored to ``_EPS_EPS_FLOOR`` so the
    softmax remains well-defined; only strictly negative ``eps``
    raises.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    with pytest.raises(ValueError):
        RelaxedEndpointTerm(lambda_relax=1e-3, eps=-0.1)


# ---------------------------------------------------------------------------
# Test 12: empirical_distribution sums to 1
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_empirical_distribution_sums_to_one():
    """``q_T = softmax(-E / ε)`` is a proper distribution (sums to 1)."""
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)
    energies = _toy_energies(b=8, seed=7)
    q = term.empirical_distribution(energies)
    assert q.shape == (8,)
    assert abs(q.sum().item() - 1.0) < 1e-6
    assert (q >= 0).all()


# ---------------------------------------------------------------------------
# Test 13: empirical_distribution flattens at large ε
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_empirical_distribution_uniform_at_large_eps():
    """At ``ε → ∞``, ``q_T`` becomes uniform (all entries ~1/B).

    Pins the high-temperature limit contract: the empirical
    distribution cannot over-concentrate when ``ε`` is much
    larger than the energy spread.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=1e-3, eps=1e6)
    energies = torch.tensor([0.0, 5.0, -3.0, 1.0])
    q = term.empirical_distribution(energies)
    expected = torch.full((4,), 0.25)
    assert torch.allclose(q, expected, atol=1e-3), (
        f"q must be uniform at large ε; got {q.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 14: augment_learned_shaping — output shape = base shape
# ---------------------------------------------------------------------------
def test_augment_learned_shaping_shape():
    """``augment_learned_shaping`` returns a tensor with the same shape
    and dtype as the base shaping output (additive contract)."""
    from molmetal.adapters.transport_relaxation_integration import (
        augment_learned_shaping,
    )

    base_scalar = torch.tensor(0.5)
    trajectory = _toy_trajectory(b=4, t=3, d=3)
    energies = _toy_energies(b=4)
    out = augment_learned_shaping(
        base_scalar, trajectory, energies,
        lambda_relax=1e-3, eps=0.1,
    )
    assert out.shape == base_scalar.shape
    assert out.dtype == base_scalar.dtype
    # Scalar output > base output (positive penalty).
    assert out.item() > base_scalar.item()


# ---------------------------------------------------------------------------
# Test 15: augment_learned_shaping — vector base output is preserved
# ---------------------------------------------------------------------------
def test_augment_learned_shaping_vector_base():
    """``augment_learned_shaping`` accepts a ``(B,)`` base output and
    preserves its shape (the KL is added as a broadcast scalar)."""
    from molmetal.adapters.transport_relaxation_integration import (
        augment_learned_shaping,
    )

    base_vec = torch.tensor([0.1, 0.2, 0.3, 0.4])
    trajectory = _toy_trajectory(b=4, t=3, d=3)
    energies = _toy_energies(b=4)
    out = augment_learned_shaping(
        base_vec, trajectory, energies,
        lambda_relax=1e-3, eps=0.1,
    )
    assert out.shape == base_vec.shape
    # Each component must be base_i + penalty (broadcast).
    assert (out >= base_vec).all(), (
        f"Penalty must be >= 0 broadcast-wise; got out={out.tolist()}, "
        f"base={base_vec.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 16: augment_learned_shaping — λ = 0 recovers the base exactly
# ---------------------------------------------------------------------------
def test_augment_learned_shaping_zero_lambda_recovers_base():
    """When ``λ_relax = 0`` the augmented shaping is bit-exact equal to
    the base shaping.  This is the W2.2 no-op contract.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        augment_learned_shaping,
    )

    base_scalar = torch.tensor(-1.234)
    trajectory = _toy_trajectory(b=6, t=4, d=3)
    energies = _toy_energies(b=6, seed=11)
    out = augment_learned_shaping(
        base_scalar, trajectory, energies,
        lambda_relax=0.0, eps=0.1,
    )
    assert torch.allclose(out, base_scalar, atol=1e-7), (
        f"λ=0 must recover base exactly; got {out.item()} vs "
        f"{base_scalar.item()}"
    )


# ---------------------------------------------------------------------------
# Test 17: LambdaRelaxSweep — best_lambda returns one of the grid values
# ---------------------------------------------------------------------------
def test_lambda_sweep_best_lambda_in_grid():
    """``LambdaRelaxSweep.best_lambda`` returns a value from ``self.lambdas``.

    The grid default is ``(1e-4, 1e-3, 1e-2)``; the function must
    not invent a new λ outside the grid.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        LambdaRelaxSweep,
    )

    sweep = LambdaRelaxSweep(
        base_shaping=_constant_shaping(0.0),
        lambdas=(1e-4, 1e-3, 1e-2),
    )
    trajectory = _toy_trajectory(b=8, t=4, d=3, seed=3)
    energies = _toy_energies(b=8, seed=3)
    val_features = [None] * 8
    best = sweep.best_lambda(trajectory, energies, val_features)
    assert best in (1e-4, 1e-3, 1e-2), (
        f"best_lambda must be from the grid, got {best}"
    )


# ---------------------------------------------------------------------------
# Test 18: LambdaRelaxSweep — equal energies → smallest λ wins
# ---------------------------------------------------------------------------
def test_lambda_sweep_ties_broken_by_smallest_lambda():
    """When the KL adjustment produces the same augmented reward at
    every λ (e.g. energies are all equal), the smallest ``λ`` wins.

    Pins the tie-break contract — the sweep must be deterministic
    and prefer the conservative (smallest λ) pick when the
    evidence is uniform.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        LambdaRelaxSweep,
    )

    sweep = LambdaRelaxSweep(
        base_shaping=_constant_shaping(0.0),
        lambdas=(1e-4, 1e-3, 1e-2),
    )
    trajectory = _toy_trajectory(b=6, t=3, d=3)
    energies = torch.full((6,), 0.5)  # equal → KL = 0 everywhere
    best = sweep.best_lambda(trajectory, energies, [None] * 6)
    assert best == 1e-4, (
        f"Equal energies must break ties by smallest λ, got {best}"
    )


# ---------------------------------------------------------------------------
# Test 19: LambdaRelaxSweep — evaluate_grid returns one row per λ
# ---------------------------------------------------------------------------
def test_lambda_sweep_evaluate_grid_shape():
    """``evaluate_grid`` returns one row per ``λ`` with the schema
    ``{lambda_relax, kl, penalty, base_mean, augmented_mean}``.

    Pins the logging shape so downstream dashboards (the W2.2
    TensorBoard integration) can ingest the rows without parsing.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        LambdaRelaxSweep,
    )

    sweep = LambdaRelaxSweep(
        base_shaping=_constant_shaping(0.5),
        lambdas=(1e-4, 1e-3, 1e-2),
    )
    trajectory = _toy_trajectory(b=5, t=3, d=3, seed=5)
    energies = _toy_energies(b=5, seed=5)
    val_features = [{"sa_mean_norm": 0.3}] * 5
    rows = sweep.evaluate_grid(trajectory, energies, val_features)
    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
    for row in rows:
        for key in ("lambda_relax", "kl", "penalty", "base_mean", "augmented_mean"):
            assert key in row, f"missing key {key!r} in row {row}"
            assert isinstance(row[key], float), (
                f"row[{key!r}] must be a float, got {type(row[key])}"
            )


# ---------------------------------------------------------------------------
# Test 20: LambdaRelaxSweep — constructor rejects empty grid
# ---------------------------------------------------------------------------
def test_lambda_sweep_rejects_empty_grid():
    """``LambdaRelaxSweep`` rejects an empty ``lambdas`` sequence."""
    from molmetal.adapters.transport_relaxation_integration import (
        LambdaRelaxSweep,
    )

    with pytest.raises(ValueError):
        LambdaRelaxSweep(base_shaping=_constant_shaping(0.0), lambdas=())


# ---------------------------------------------------------------------------
# Test 21: LambdaRelaxSweep — constructor rejects negative λ
# ---------------------------------------------------------------------------
def test_lambda_sweep_rejects_negative_lambda():
    """``LambdaRelaxSweep`` rejects any ``λ < 0`` in the grid."""
    from molmetal.adapters.transport_relaxation_integration import (
        LambdaRelaxSweep,
    )

    with pytest.raises(ValueError):
        LambdaRelaxSweep(
            base_shaping=_constant_shaping(0.0),
            lambdas=(1e-3, -1e-4, 1e-2),
        )


# ---------------------------------------------------------------------------
# Test 22: Adapter does NOT modify flow_matching_lipman (architectural)
# ---------------------------------------------------------------------------
def test_adapter_does_not_mutate_flow_matching_lipman():
    """Importing :mod:`molmetal.adapters.transport_relaxation_integration`
    must NOT mutate any module-level state in
    :mod:`molmetal.adapters.flow_matching_lipman`.

    This is the W2.2 architectural contract: the R17 close-out
    is locked, and the new adapter is the *only* file that ships
    under W2.2.
    """
    from molmetal.adapters import flow_matching_lipman
    from molmetal.adapters import transport_relaxation_integration

    pre = set(dir(flow_matching_lipman))
    import importlib
    importlib.reload(transport_relaxation_integration)
    post = set(dir(flow_matching_lipman))
    removed = pre - post
    added = post - pre
    assert removed == set(), (
        f"Importing transport_relaxation_integration must not remove "
        f"flow_matching_lipman attributes; removed={removed}"
    )
    assert added == set(), (
        f"Importing transport_relaxation_integration must not add "
        f"flow_matching_lipman attributes; added={added}"
    )
    # The adapter must expose the three W2.2 public symbols.
    for name in (
        "RelaxedEndpointTerm",
        "augment_learned_shaping",
        "LambdaRelaxSweep",
    ):
        assert hasattr(transport_relaxation_integration, name), (
            f"transport_relaxation_integration must expose {name!r}"
        )


# ---------------------------------------------------------------------------
# Test 23: Adapter does NOT modify learned_shaping.py (architectural)
# ---------------------------------------------------------------------------
def test_adapter_does_not_mutate_learned_shaping():
    """Importing the adapter must not modify the public symbols of
    :mod:`molmetal.molmetal_lam.reward.learned_shaping`.

    Pins the constraint "**不改** learned_shaping.py" from the
    W2.2 brief.
    """
    from molmetal.molmetal_lam.reward import learned_shaping
    from molmetal.adapters import transport_relaxation_integration

    pre = set(dir(learned_shaping))
    import importlib
    importlib.reload(transport_relaxation_integration)
    post = set(dir(learned_shaping))
    # The adapter adds no new public symbols to learned_shaping.
    added = post - pre
    for sym in added:
        # Allow module-level dunder / private additions, but no
        # public API mutations.
        assert sym.startswith("_"), (
            f"learned_shaping public surface must be unchanged; "
            f"unexpected symbol {sym!r}"
        )


# ---------------------------------------------------------------------------
# Test 24: Deterministic across repeated calls
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_is_deterministic():
    """Two consecutive calls with the same inputs must produce the same
    output — pins the test-determinism contract so CI is reproducible.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    torch.manual_seed(7)
    trajectory = _toy_trajectory(b=5, t=3, d=3, seed=7)
    energies = _toy_energies(b=5, seed=7)
    term = RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)
    out_a = term.forward(trajectory, energies)
    out_b = term.forward(trajectory, energies)
    assert torch.equal(out_a, out_b), (
        f"Two calls must be deterministic; max diff = "
        f"{(out_a - out_b).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 25: KL scales with ε (sharper at small ε)
# ---------------------------------------------------------------------------
def test_relaxed_endpoint_kl_grows_as_eps_shrinks():
    """The KL grows as ``ε → 0`` (sharper ``q_T``).

    Pins the qualitative contract: the temperature controls
    the *concentration* of ``q_T``.  At ``ε → ∞`` the KL → 0;
    at ``ε → 0`` the KL → ``log B``.  The test checks the
    monotone direction.
    """
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    trajectory = _toy_trajectory(b=8, t=3, d=3, seed=99)
    energies = torch.tensor([0.0, 5.0, -3.0, 1.0, 2.0, -1.0, 4.0, -2.0])
    kl_large = RelaxedEndpointTerm(lambda_relax=1.0, eps=10.0).kl(
        trajectory, energies,
    ).item()
    kl_small = RelaxedEndpointTerm(lambda_relax=1.0, eps=0.01).kl(
        trajectory, energies,
    ).item()
    assert kl_small > kl_large, (
        f"KL must grow as ε shrinks; got large={kl_large}, small={kl_small}"
    )
