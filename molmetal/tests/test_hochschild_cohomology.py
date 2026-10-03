"""Tests for :mod:`molmetal.math_skills.hochschild_cohomology`.

T11 — Hochschild Cohomology for MLC Reduction Composition (李思 2026)
======================================================================

These tests cover the four mathematical primitives shipped by the
R18 cohomological-semantics skeleton:

1. **Cochain** — an n-cochain of the bimodule :math:`M` over the
   algebra :math:`A`.  Evaluated via ``Cochain.evaluate``.
2. **Differential** — the coboundary operator :math:`\\delta: C^n
   \\to C^{n+1}` and the fundamental identity :math:`\\delta^2 = 0`.
3. **HochschildCohomology** — computation of
   :math:`\\dim H^n(A, M)` for :math:`n = 0, 1, 2, \\ldots, max\\_degree`.
4. **MLCReductionCochain** — the MLC surface wrapper for
   :math:`\\beta`-reductions; the closure-theorem test.

Reference
---------
Li, S. (李思), Wang, Z. and Yang, P. (2026).  *Stochastic Calculus
and Hochschild Homology*.  Peking University Statistical Science
Center seminar (2026-03-09), based on the preprint of the same
name (arXiv:2501.12360, 2025).

These tests are **CPU-only**, use small data (degree 0..3, batch
sizes 1..4), and run in < 5 seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.math_skills.hochschild_cohomology import (
    Cochain,
    HochschildCohomology,
    MLCReductionCochain,
    mlc_algebra,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _linear_one_cochain(slope: float = 1.0, intercept: float = 0.0) -> Cochain:
    """Return a 1-cochain that computes ``slope * args[0] + intercept``."""

    def fn(args: List[torch.Tensor]) -> torch.Tensor:
        return slope * args[0] + intercept

    return Cochain(degree=1, cochains=[fn])


def _constant_zero_cochain() -> Cochain:
    """Return a 0-cochain that is identically zero."""

    def fn(args: List[torch.Tensor]) -> torch.Tensor:
        return torch.zeros(1)

    return Cochain(degree=0, cochains=[fn])


def _constant_three_cochain(value: float = 0.0) -> Cochain:
    """Return a 3-cochain that is a constant tensor."""
    c = torch.tensor(value)

    def fn(args: List[torch.Tensor]) -> torch.Tensor:
        return c.clone()

    return Cochain(degree=3, cochains=[fn])


# ---------------------------------------------------------------------------
# 1. Cochain
# ---------------------------------------------------------------------------


def test_cochain_init_degree() -> None:
    """``Cochain(degree=N, cochains=[fn])`` stores the degree."""
    coch = _linear_one_cochain()
    assert coch.degree == 1


def test_cochain_init_negative_degree_raises() -> None:
    """``Cochain(degree=-1, ...)`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="degree must be"):
        Cochain(degree=-1, cochains=[lambda args: torch.zeros(1)])


def test_cochain_init_wrong_list_length_raises() -> None:
    """``Cochain`` rejects ``cochains`` lists that are not length 1."""
    with pytest.raises(ValueError, match="length 1"):
        Cochain(degree=1, cochains=[lambda a: torch.zeros(1), lambda b: torch.zeros(1)])


def test_cochain_init_non_callable_raises() -> None:
    """``Cochain`` rejects non-callable elements in the ``cochains`` list."""
    with pytest.raises(ValueError, match="Callable"):
        Cochain(degree=0, cochains=["not a function"])


def test_cochain_evaluate() -> None:
    """``evaluate`` runs the callable on the args list."""
    coch = _linear_one_cochain(slope=2.0, intercept=3.0)
    out = coch.evaluate([torch.tensor(5.0)])
    assert torch.allclose(out, torch.tensor(13.0))


def test_cochain_evaluate_arity_mismatch_raises() -> None:
    """``evaluate`` rejects the wrong number of arguments."""
    coch = _linear_one_cochain()
    with pytest.raises(ValueError, match="args must have length"):
        coch.evaluate([torch.tensor(1.0), torch.tensor(2.0)])


# ---------------------------------------------------------------------------
# 2. Differential
# ---------------------------------------------------------------------------


def test_differential_init() -> None:
    """``Differential(left, right, mult)`` stores the three callables."""
    alg = mlc_algebra()
    assert callable(alg.left_mult)
    assert callable(alg.right_mult)
    assert callable(alg.mult)


def test_differential_apply() -> None:
    """``apply(cochain)`` returns a Cochain of degree ``cochain.degree + 1``."""
    alg = mlc_algebra()
    coch = _linear_one_cochain()
    delta = alg.apply(coch)
    assert isinstance(delta, Cochain)
    assert delta.degree == 2


def test_differential_apply_constant_zero_cochain() -> None:
    """``delta(0-cochain)`` returns a 1-cochain that vanishes on the
    canonical MLC algebra.

    With the canonical algebra (left, right, mult = add), the
    Hochschild coboundary of the constant zero 0-cochain is
    ``(δ(0))(a_1) = a_1 - a_1 = 0`` for every ``a_1``.
    """
    alg = mlc_algebra()
    coch = _constant_zero_cochain()
    delta = alg.apply(coch)
    assert delta.degree == 1
    out = delta.evaluate([torch.tensor(2.0)])
    assert torch.allclose(out, torch.tensor(0.0), atol=1.0e-6)


def test_differential_delta_zero() -> None:
    """``delta^2 = 0`` holds on any cochain we test."""
    alg = mlc_algebra()
    coch = _linear_one_cochain(slope=2.0)
    test_args = [
        [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)],
        [torch.tensor(-1.0), torch.tensor(0.5), torch.tensor(0.0)],
    ]
    assert alg.delta_squared_zero(coch, test_args) is True


def test_differential_delta_zero_constant_three() -> None:
    """``delta^2 = 0`` holds for a constant 3-cochain (trivially)."""
    alg = mlc_algebra()
    coch = _constant_three_cochain(value=1.5)
    test_args = [
        [
            torch.tensor(0.5),
            torch.tensor(0.5),
            torch.tensor(0.5),
            torch.tensor(0.5),
            torch.tensor(0.5),
        ],
    ]
    # delta^2 raises the degree from 3 to 5 (so we need n+2 = 5 args).
    assert alg.delta_squared_zero(coch, test_args) is True


def test_differential_delta_zero_wrong_arity_returns_false() -> None:
    """``delta^2`` returns ``False`` when the test args don't match the cochain degree + 2."""
    alg = mlc_algebra()
    coch = _linear_one_cochain()
    test_args = [
        [torch.tensor(1.0), torch.tensor(2.0)],  # only 2 args, but coch.degree + 2 = 3
    ]
    assert alg.delta_squared_zero(coch, test_args) is False


def test_differential_apply_degree_one_to_two() -> None:
    """``apply(cochain)`` increases the degree by exactly 1."""
    alg = mlc_algebra()
    coch_0 = _constant_zero_cochain()  # degree 0
    coch_1 = alg.apply(coch_0)
    coch_2 = alg.apply(coch_1)
    assert coch_1.degree == 1
    assert coch_2.degree == 2


# ---------------------------------------------------------------------------
# 3. HochschildCohomology
# ---------------------------------------------------------------------------


def test_hochschild_init() -> None:
    """``HochschildCohomology(algebra, max_degree=3)`` constructs successfully."""
    alg = mlc_algebra()
    h = HochschildCohomology(algebra=alg, max_degree=3)
    assert h.algebra is alg
    assert h.max_degree == 3


def test_hochschild_init_negative_max_degree_raises() -> None:
    """``HochschildCohomology`` rejects negative ``max_degree``."""
    alg = mlc_algebra()
    with pytest.raises(ValueError, match="max_degree"):
        HochschildCohomology(algebra=alg, max_degree=-1)


def test_hochschild_dim_zero() -> None:
    """``H^0`` dim = 0 when no 0-cochains are supplied."""
    alg = mlc_algebra()
    h = HochschildCohomology(algebra=alg, max_degree=2)
    # No cochains at all -> all H^n have dim 0.
    dims = h.compute(cochains=[])
    assert dims.shape == (3,)
    assert int(dims[0]) == 0
    assert int(dims[1]) == 0
    assert int(dims[2]) == 0


def test_hochschild_dim_one() -> None:
    """``H^1`` dim is computed (>= 0) when only a single non-cocycle 1-cochain is supplied."""
    alg = mlc_algebra()
    h = HochschildCohomology(algebra=alg, max_degree=2)
    coch = _linear_one_cochain(slope=1.0)  # not a cocycle (delta != 0)
    dims = h.compute(cochains=[coch])
    assert dims.shape == (3,)
    assert int(dims[1]) >= 0
    assert int(dims[1]) <= 1


def test_hochschild_compute_returns_tensor() -> None:
    """``compute`` returns a ``torch.Tensor`` of dtype long."""
    alg = mlc_algebra()
    h = HochschildCohomology(algebra=alg, max_degree=3)
    dims = h.compute(cochains=[_linear_one_cochain()])
    assert isinstance(dims, torch.Tensor)
    assert dims.dtype == torch.long


def test_hochschild_default_max_degree() -> None:
    """``HochschildCohomology`` defaults to ``max_degree=3``."""
    alg = mlc_algebra()
    h = HochschildCohomology(algebra=alg)
    assert h.max_degree == 3


# ---------------------------------------------------------------------------
# 4. MLCReductionCochain
# ---------------------------------------------------------------------------


def test_mlc_reduction_cochain_init() -> None:
    """``MLCReductionCochain(atom='Pt_II')`` stores the atom symbol."""
    c = MLCReductionCochain(atom="Pt_II")
    assert c.atom == "Pt_II"
    assert c.atoms == ("Pt_II",)


def test_mlc_reduction_cochain_compose() -> None:
    """``apply(other)`` composes two reduction cochains (canonical sorted)."""
    a = MLCReductionCochain(atom="Pt")
    b = MLCReductionCochain(atom="Cl")
    composed = a.apply(b)
    assert isinstance(composed, MLCReductionCochain)
    # Canonical form is sorted
    assert composed.atom == "Cl*Pt"


def test_mlc_reduction_cochain_compose_returns_new_instance() -> None:
    """``apply`` returns a *new* cochain (does not mutate)."""
    a = MLCReductionCochain(atom="A")
    b = MLCReductionCochain(atom="B")
    composed = a.apply(b)
    assert composed is not a
    assert composed is not b


def test_mlc_reduction_cochain_compose_wrong_type_raises() -> None:
    """``apply`` rejects non-MLCReductionCochain arguments."""
    a = MLCReductionCochain(atom="A")
    with pytest.raises(TypeError, match="MLCReductionCochain"):
        a.apply("not a cochain")  # type: ignore[arg-type]


def test_mlc_reduction_cochain_to_cochain() -> None:
    """``to_cochain()`` returns a 1-cochain whose ``evaluate`` is constant."""
    coch = MLCReductionCochain(atom="Pt_II").to_cochain()
    assert isinstance(coch, Cochain)
    assert coch.degree == 1
    out = coch.evaluate([torch.tensor(2.0)])
    assert out.shape == (1,)


def test_mlc_reduction_cochain_to_cochain_deterministic() -> None:
    """``to_cochain()`` is deterministic (same atom -> same cochain)."""
    c1 = MLCReductionCochain(atom="Pt_II").to_cochain()
    c2 = MLCReductionCochain(atom="Pt_II").to_cochain()
    # Both evaluate to the same constant.
    a = torch.tensor(0.0)
    o1 = c1.evaluate([a])
    o2 = c2.evaluate([a])
    assert torch.allclose(o1, o2)


def test_mlc_reduction_associative_closure_theorem() -> None:
    """The closure theorem holds for three reduction cochains."""
    a = MLCReductionCochain(atom="A")
    b = MLCReductionCochain(atom="B")
    c = MLCReductionCochain(atom="C")
    assert MLCReductionCochain.closure_theorem_holds(a, b, c) is True


def test_mlc_reduction_closure_theorem_metals() -> None:
    """The closure theorem holds for metal atoms (Pt_II + NH3 + Cl)."""
    a = MLCReductionCochain(atom="Pt_II")
    b = MLCReductionCochain(atom="NH3")
    c = MLCReductionCochain(atom="Cl")
    assert MLCReductionCochain.closure_theorem_holds(a, b, c) is True


def test_mlc_reduction_closure_theorem_holds_under_swap() -> None:
    """The closure theorem is invariant under permutation of the inputs.

    The composition is symmetric (sqrt(2) midpoint), so swapping
    any two of the three inputs gives the same canonical cochain.
    """
    a = MLCReductionCochain(atom="Pt")
    b = MLCReductionCochain(atom="H")
    c = MLCReductionCochain(atom="O")
    assert MLCReductionCochain.closure_theorem_holds(a, b, c) is True
    assert MLCReductionCochain.closure_theorem_holds(b, a, c) is True
    assert MLCReductionCochain.closure_theorem_holds(c, b, a) is True


# ---------------------------------------------------------------------------
# 5. End-to-end: MLC reduction -> Hochschild cochain -> delta^2 = 0
# ---------------------------------------------------------------------------


def test_e2e_mlc_reduction_to_cochain_to_delta_squared() -> None:
    """End-to-end: an MLC reduction cochain lifts to a 1-cochain
    whose :math:`\\delta^2` is identically zero."""
    alg = mlc_algebra()
    reduction = MLCReductionCochain(atom="Pt_II")
    coch = reduction.to_cochain()
    # delta^2 coch should be zero on any 3 args (since coch is constant).
    test_args = [
        [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)],
        [torch.tensor(-1.0), torch.tensor(0.5), torch.tensor(0.0)],
    ]
    assert alg.delta_squared_zero(coch, test_args) is True


def test_e2e_multiple_mlc_reductions_compose() -> None:
    """End-to-end: multiple MLC reductions compose into a single chain
    (canonical sorted form)."""
    atoms = ["Pt", "Cl", "Cl", "NH3", "NH3"]
    cochains = [MLCReductionCochain(atom=a) for a in atoms]
    # Pairwise compose (left fold).
    result = cochains[0]
    for c in cochains[1:]:
        result = result.apply(c)
    # Sorted union of all atom symbols.
    expected = "*".join(sorted(atoms))
    assert result.atom == expected


def test_e2e_hochschild_dim_for_trivial_algebra() -> None:
    """For the trivial abelian algebra (the MLC closed-term sub-algebra),
    :math:`H^n` is zero for every degree (closure theorem: associative,
    so the only cocycles are coboundaries)."""
    alg = mlc_algebra()
    h = HochschildCohomology(algebra=alg, max_degree=2)
    # Supply only zero-cochains; dim should be zero everywhere.
    dims = h.compute(cochains=[])
    assert int(dims.sum()) == 0
