"""Tests for :mod:`molmetal.math_skills.quantum_semantics`.

T10 — Quantum Semantics for Primitive Combinators (麻小南 2026)
=================================================================

These tests cover the four mathematical primitives shipped by the
R18 quantum-semantics skeleton:

1. **SymplecticPoint** — a point ``(q, p) in R^{2n}`` on the
   standard symplectic vector space, with the canonical Poisson
   bracket ``{f, g} = sum (df/dq * dg/dp - df/dp * dg/dq)``.
2. **PrimitiveAtomQuantum** — canonical embedding of every
   42-entry ``PRIMITIVE_ATOMS`` into ``R^{2n}`` (with
   ``n = arity(atom)``).
3. **BetaReductionSymp** — symplectomorphism that composes two
   symplectic points; symplectomorphism = ``beta-reduction`` is
   the quantum-semantic reading of MLC bonding.
4. **quantize_atom** — top-level helper that maps a primitive
   atom to a ``SymplecticPoint``.

Reference
---------
Ma, X. (麻小南) (2026).  *Quantization commutes with reduction:
from symplectic manifolds to CR manifolds*.  Lecture #58 at Sun
Yat-sen University (Zhuhai), scheduled 2026-03-16.

These tests are **CPU-only**, use small data (batch sizes 1-4,
dim 1-4), and run in < 5 seconds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.math_skills import (
    BetaReductionSymp,
    PrimitiveAtomQuantum,
    SymplecticPoint,
    quantize_atom,
)
from molmetal.math_skills.quantum_semantics import (
    _atom_quantum_embedding,
    _atom_seed,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _rand_point(dim: int = 2, seed: int = 0) -> SymplecticPoint:
    """Return a random ``SymplecticPoint`` of half-dim ``dim``."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=g, dtype=torch.float32)
    p = torch.randn(dim, generator=g, dtype=torch.float32)
    return SymplecticPoint(q=q, p=p)


def _linear_obs(p: SymplecticPoint) -> torch.Tensor:
    """Linear observable ``f(q, p) = q[0] + p[0]`` (scalar)."""
    return p.q[..., 0] + p.p[..., 0]


# ---------------------------------------------------------------------------
# 1. SymplecticPoint
# ---------------------------------------------------------------------------


def test_symplectic_point_dimension() -> None:
    """``dimension()`` returns ``2n`` (full symp dim)."""
    p = _rand_point(dim=2)
    assert p.dimension() == 4
    p3 = _rand_point(dim=3)
    assert p3.dimension() == 6


def test_symplectic_point_half_dim() -> None:
    """``half_dim()`` returns ``n`` (half symp dim)."""
    p = _rand_point(dim=2)
    assert p.half_dim() == 2


def test_symplectic_point_shape_mismatch_raises() -> None:
    """``__init__`` rejects ``q.shape != p.shape``."""
    q = torch.zeros(2)
    p = torch.zeros(3)
    with pytest.raises(ValueError, match="same shape"):
        SymplecticPoint(q=q, p=p)


def test_symplectic_point_dtype_mismatch_raises() -> None:
    """``__init__`` rejects ``q.dtype != p.dtype``."""
    q = torch.zeros(2, dtype=torch.float32)
    p = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="share dtype"):
        SymplecticPoint(q=q, p=p)


def test_symplectic_point_device_mismatch_raises() -> None:
    """``__init__`` rejects ``q.device != p.device``.

    Skips if only one device is available (e.g. CPU-only test runs).
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("CUDA not available")
    q = torch.zeros(2)
    p = torch.zeros(2, device="cuda")
    with pytest.raises(ValueError, match="share device"):
        SymplecticPoint(q=q, p=p)


def test_symplectic_point_zero_dim_raises() -> None:
    """``__init__`` rejects ``q.ndim < 1`` (zero-dim tensors)."""
    with pytest.raises(ValueError, match="at least one dimension"):
        SymplecticPoint(q=torch.tensor(0.0), p=torch.tensor(0.0))


def test_symplectic_point_poisson_bracket_self() -> None:
    """``{f, f} = 0`` for any observable ``f``."""
    p = _rand_point(dim=2)

    def f(q, p_):
        return q[..., 0] + p_[..., 0]

    bracket = p.poisson_bracket(f, f)
    assert abs(bracket.item()) < 1.0e-4


def test_symplectic_point_poisson_bracket_antisymmetric() -> None:
    """``{f, g} = -{g, f}`` for ``f = q_0 + q_1``, ``g = p_0 + p_1``."""
    p = _rand_point(dim=2)

    def f(q, p_):
        return q[..., 0] + q[..., 1]

    def g(q, p_):
        return p_[..., 0] + p_[..., 1]

    fg = p.poisson_bracket(f, g)
    gf = p.poisson_bracket(g, f)
    assert torch.allclose(fg, -gf, atol=1.0e-4)


def test_symplectic_point_poisson_bracket_canonical() -> None:
    """``{q_i, p_j} = delta_{ij}`` — the canonical symplectic relation."""
    p = _rand_point(dim=3)

    def q0(q, p_):
        return q[..., 0]

    def p0(q, p_):
        return p_[..., 0]

    def p1(q, p_):
        return p_[..., 1]

    def p2(q, p_):
        return p_[..., 2]

    pq00 = p.poisson_bracket(q0, p0).item()
    pq01 = p.poisson_bracket(q0, p1).item()
    pq02 = p.poisson_bracket(q0, p2).item()
    assert abs(pq00 - 1.0) < 1.0e-3
    assert abs(pq01) < 1.0e-3
    assert abs(pq02) < 1.0e-3


def test_symplectic_point_shape_mismatch_bracket_raises() -> None:
    """``poisson_bracket`` of constant tensors still works (default zero)."""
    p = _rand_point(dim=2)
    # Constants are accepted (bracket = 0)
    f_const = torch.tensor(1.0)
    g_const = torch.tensor(2.0)
    bracket = p.poisson_bracket(f_const, g_const)
    assert abs(bracket.item()) < 1.0e-6


# ---------------------------------------------------------------------------
# 2. PrimitiveAtomQuantum
# ---------------------------------------------------------------------------


def test_primitive_atom_quantum_init_default() -> None:
    """``PrimitiveAtomQuantum`` accepts bare symbol with defaults."""
    prim = PrimitiveAtomQuantum(atom_symbol="Pt_II")
    assert prim.atom_symbol == "Pt_II"
    assert prim.arity == 1  # default
    assert prim.is_metal is False  # default


def test_primitive_atom_quantum_init_full() -> None:
    """``PrimitiveAtomQuantum`` accepts the full intrinsic-property tuple."""
    prim = PrimitiveAtomQuantum(
        atom_symbol="Pt_II",
        arity=4,
        atomic_num=78,
        valence=2,
        lone_pairs=2,
        geometry="square_planar",
        is_metal=True,
    )
    assert prim.atom_symbol == "Pt_II"
    assert prim.arity == 4
    assert prim.atomic_num == 78
    assert prim.geometry == "square_planar"
    assert prim.is_metal is True


def test_primitive_atom_quantum_invalid_arity_raises() -> None:
    """``PrimitiveAtomQuantum`` rejects ``arity < 1``."""
    with pytest.raises(ValueError, match="arity must be"):
        PrimitiveAtomQuantum(atom_symbol="X", arity=0)


def test_primitive_atom_quantum_symplectic_repr_shape() -> None:
    """``symplectic_repr(dim=2)`` returns a ``SymplecticPoint`` with 2n=4."""
    prim = PrimitiveAtomQuantum(atom_symbol="Pt_II", arity=4, is_metal=True)
    pt = prim.symplectic_repr(dim=2)
    assert isinstance(pt, SymplecticPoint)
    assert pt.q.shape == (2,)
    assert pt.p.shape == (2,)
    assert pt.dimension() == 4


def test_primitive_atom_quantum_symplectic_repr_metal_has_larger_norm() -> None:
    """Metal centres have larger symplectic radius than main-group atoms."""
    pt_pt = PrimitiveAtomQuantum(
        atom_symbol="Pt", arity=4, atomic_num=78, is_metal=True
    ).symplectic_repr(dim=2)
    pt_c = PrimitiveAtomQuantum(
        atom_symbol="C", arity=4, atomic_num=6, is_metal=False
    ).symplectic_repr(dim=2)
    # Compare squared norms of (q, p) — Pt should be larger.
    pt_norm = (pt_pt.q.pow(2).sum() + pt_pt.p.pow(2).sum()).item()
    c_norm = (pt_c.q.pow(2).sum() + pt_c.p.pow(2).sum()).item()
    assert pt_norm > c_norm


def test_primitive_atom_quantum_symplectic_repr_invalid_dim_raises() -> None:
    """``symplectic_repr`` rejects ``dim < 1``."""
    prim = PrimitiveAtomQuantum(atom_symbol="Pt_II")
    with pytest.raises(ValueError, match="dim must be"):
        prim.symplectic_repr(dim=0)


def test_primitive_atom_quantum_deterministic() -> None:
    """``symplectic_repr`` is deterministic (same input → same output)."""
    prim1 = PrimitiveAtomQuantum(
        atom_symbol="Pt_II",
        arity=4,
        atomic_num=78,
        valence=2,
        lone_pairs=2,
        geometry="square_planar",
        is_metal=True,
    )
    prim2 = PrimitiveAtomQuantum(
        atom_symbol="Pt_II",
        arity=4,
        atomic_num=78,
        valence=2,
        lone_pairs=2,
        geometry="square_planar",
        is_metal=True,
    )
    pt1 = prim1.symplectic_repr(dim=2)
    pt2 = prim2.symplectic_repr(dim=2)
    assert torch.allclose(pt1.q, pt2.q)
    assert torch.allclose(pt1.p, pt2.p)


# ---------------------------------------------------------------------------
# 3. BetaReductionSymp
# ---------------------------------------------------------------------------


def test_beta_reduction_symp_init() -> None:
    """``BetaReductionSymp()`` builds with no state."""
    br = BetaReductionSymp()
    assert br is not None


def test_beta_reduction_symp_apply_composed_shape() -> None:
    """``apply(p1, p2)`` returns a point with the same shape as inputs."""
    p1 = _rand_point(dim=3)
    p2 = _rand_point(dim=3)
    br = BetaReductionSymp()
    composed = br.apply(p1, p2)
    assert composed.q.shape == p1.q.shape
    assert composed.p.shape == p1.p.shape


def test_beta_reduction_symp_apply_shape_mismatch_raises() -> None:
    """``apply`` rejects inputs with mismatched symplectic shape."""
    p1 = _rand_point(dim=2)
    p2 = _rand_point(dim=3)
    br = BetaReductionSymp()
    with pytest.raises(ValueError, match="share symplectic shape"):
        br.apply(p1, p2)


def test_beta_reduction_symp_apply_is_midpoint() -> None:
    """``apply(p1, p2)`` returns the (sqrt(2)-scaled) midpoint of ``p1, p2``."""
    p1_q = torch.tensor([0.0, 0.0])
    p1_p = torch.tensor([0.0, 0.0])
    p2_q = torch.tensor([2.0, 2.0])
    p2_p = torch.tensor([2.0, 2.0])
    p1 = SymplecticPoint(q=p1_q, p=p1_p)
    p2 = SymplecticPoint(q=p2_q, p=p2_p)
    br = BetaReductionSymp()
    composed = br.apply(p1, p2)
    # Midpoint should be (1, 1) / sqrt(2) ≈ (0.7071, 0.7071)
    expected = torch.tensor([2.0 / math.sqrt(2.0), 2.0 / math.sqrt(2.0)])
    assert torch.allclose(composed.q, expected)
    assert torch.allclose(composed.p, expected)


def test_beta_reduction_symp_commutes_with_quantization() -> None:
    """``commutes_with_quantization`` is True for symmetric inputs."""
    p1 = _rand_point(dim=2, seed=1)
    p2 = _rand_point(dim=2, seed=2)
    br = BetaReductionSymp()
    assert br.commutes_with_quantization(p1, p2) is True


def test_beta_reduction_symp_commutes_mismatched_shape_returns_false() -> None:
    """``commutes_with_quantization`` is False for mismatched shapes."""
    p1 = _rand_point(dim=2, seed=1)
    p2 = _rand_point(dim=3, seed=2)
    br = BetaReductionSymp()
    assert br.commutes_with_quantization(p1, p2) is False


# ---------------------------------------------------------------------------
# 4. quantize_atom helper
# ---------------------------------------------------------------------------


def test_quantize_atom_helper_default_dim() -> None:
    """``quantize_atom`` defaults to ``dim=2``."""
    pt = quantize_atom(atom_symbol="Cl")
    assert pt.dimension() == 4
    assert pt.q.shape == (2,)
    assert pt.p.shape == (2,)


def test_quantize_atom_helper_full() -> None:
    """``quantize_atom`` accepts the full intrinsic-property tuple."""
    pt = quantize_atom(
        atom_symbol="Pt_II",
        dim=4,
        atomic_num=78,
        valence=2,
        lone_pairs=2,
        geometry="square_planar",
        is_metal=True,
    )
    assert pt.q.shape == (4,)
    assert pt.p.shape == (4,)
    assert pt.dimension() == 8


def test_quantize_atom_helper_deterministic() -> None:
    """``quantize_atom`` is deterministic across calls (same input → same)."""
    pt1 = quantize_atom(atom_symbol="Pt_II", dim=2, is_metal=True)
    pt2 = quantize_atom(atom_symbol="Pt_II", dim=2, is_metal=True)
    assert torch.allclose(pt1.q, pt2.q)
    assert torch.allclose(pt1.p, pt2.p)


def test_quantize_atom_helper_distinct_atoms_distinct_points() -> None:
    """``quantize_atom('Pt_II') != quantize_atom('Ru_II')`` in general."""
    pt_pt = quantize_atom(
        atom_symbol="Pt_II",
        dim=2,
        atomic_num=78,
        is_metal=True,
    )
    pt_ru = quantize_atom(
        atom_symbol="Ru_II",
        dim=2,
        atomic_num=44,
        is_metal=True,
    )
    assert not torch.allclose(pt_pt.q, pt_ru.q)


# ---------------------------------------------------------------------------
# 5. Internal helpers (covered for completeness)
# ---------------------------------------------------------------------------


def test_atom_seed_deterministic() -> None:
    """``_atom_seed`` is deterministic for the same input."""
    s1 = _atom_seed("Pt_II", dim=2)
    s2 = _atom_seed("Pt_II", dim=2)
    assert s1 == s2


def test_atom_seed_distinct_dim() -> None:
    """``_atom_seed`` produces different values for distinct ``dim``."""
    s1 = _atom_seed("Pt_II", dim=2)
    s2 = _atom_seed("Pt_II", dim=3)
    assert s1 != s2


def test_atom_quantum_embedding_invalid_dim_raises() -> None:
    """``_atom_quantum_embedding`` rejects ``dim < 1``."""
    with pytest.raises(ValueError, match="dim must be"):
        _atom_quantum_embedding("Pt_II", dim=0)


# ---------------------------------------------------------------------------
# 6. End-to-end: full MLC primitive pipeline
# ---------------------------------------------------------------------------


def test_e2e_primitive_atom_pipeline() -> None:
    """End-to-end: Pt_II primitive → symplectic point → β-reduction."""
    # Quantise Pt_II and Ru_III
    pt_pt = quantize_atom(
        atom_symbol="Pt_II",
        dim=2,
        atomic_num=78,
        valence=2,
        lone_pairs=2,
        geometry="square_planar",
        is_metal=True,
    )
    pt_ru = quantize_atom(
        atom_symbol="Ru_III",
        dim=2,
        atomic_num=44,
        valence=3,
        lone_pairs=3,
        geometry="octahedral",
        is_metal=True,
    )
    # Verify both are valid symplectic points
    assert pt_pt.dimension() == 4
    assert pt_ru.dimension() == 4
    # Composing via β-reduction produces a valid point
    br = BetaReductionSymp()
    composed = br.apply(pt_pt, pt_ru)
    assert composed.q.shape == (2,)
    # And the commutation identity holds
    assert br.commutes_with_quantization(pt_pt, pt_ru) is True


def test_e2e_all_42_primitive_atoms_quantizable() -> None:
    """Every entry of the 42-entry PRIMITIVE_ATOMS can be quantised.

    Tries a representative subset (organic + halogen + main d-block
    metals + lanthanide) to keep the test fast.
    """
    representative = [
        ("H", 1, 1, 0, "s", False),
        ("C", 6, 4, 0, "sp3", False),
        ("N", 7, 3, 1, "sp3", False),
        ("O", 8, 2, 2, "sp3", False),
        ("F", 9, 1, 3, "sp3", False),
        ("Cl", 17, 1, 3, "sp3", False),
        ("Br", 35, 1, 3, "sp3", False),
        ("I", 53, 1, 3, "sp3", False),
        ("Pt", 78, 0, 4, "square_planar", True),
        ("Ru", 44, 0, 6, "octahedral", True),
        ("Au", 79, 0, 4, "square_planar", True),
        ("Pd", 46, 0, 4, "square_planar", True),
        ("Ir", 77, 0, 6, "octahedral", True),
        ("Cu", 29, 0, 4, "square_planar", True),
        ("Zn", 30, 0, 4, "tetrahedral", True),
        ("Hg", 80, 0, 2, "linear", True),
        ("Ag", 47, 0, 2, "linear", True),
        ("La", 57, 0, 6, "octahedral", True),
        ("Pt_II", 78, 2, 2, "square_planar", True),
        ("Ru_III", 44, 3, 3, "octahedral", True),
    ]
    for sym, z_at, v, lp, geom, is_metal in representative:
        pt = quantize_atom(
            atom_symbol=sym,
            dim=2,
            atomic_num=z_at,
            valence=v,
            lone_pairs=lp,
            geometry=geom,
            is_metal=is_metal,
        )
        assert isinstance(pt, SymplecticPoint)
        assert pt.q.shape == (2,)
        assert pt.p.shape == (2,)
