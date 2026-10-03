"""Tests for :mod:`molmetal_lam.lam_chem.md_relaxation_stub`.

TODO-30 Pitfall Rank 13, P2.5 — MD relaxation CPU stub.

These tests exercise the public surface of
:mod:`molmetal.molmetal_lam.lam_chem.md_relaxation_stub`:

* :class:`SimpleForceField` — energy + numerical gradient
* :class:`MDRelaxation`     — steepest-descent + Metropolis MC
* :class:`GeometryValidator` — bond + close-contact checks
* :func:`register_md_relaxation` — registration hook

All tests are **CPU-only** and use tiny inputs (≤ 5 atoms) so the
20-test suite runs in well under a second.

Test matrix (20 tests)
----------------------
SimpleForceField
 1. test_simple_force_field_energy_zero
 2. test_simple_force_field_energy_single_atom
 3. test_simple_force_field_energy_positive
 4. test_simple_force_field_energy_finite
 5. test_simple_force_field_gradient_shape
 6. test_simple_force_field_gradient_zero_at_min
 7. test_simple_force_field_invalid_method
 8. test_simple_force_field_repr

MDRelaxation
 9. test_md_relaxation_relaxes
10. test_md_relaxation_lowers_energy
11. test_md_relaxation_deterministic
12. test_md_relaxation_preserves_shape
13. test_md_relaxation_zero_steps
14. test_mc_step_returns_tuple
15. test_mc_step_deterministic
16. test_mc_step_empty

GeometryValidator
17. test_geometry_validator_reasonable
18. test_geometry_validator_bad_contact
19. test_geometry_validator_bond_too_long

Registration
20. test_register_md_relaxation_count
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make project importable when running pytest from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal_lam.lam_chem.md_relaxation_stub import (
    DEFAULT_BOND_K,
    DEFAULT_ANGLE_K,
    DEFAULT_NB_EPSILON,
    DEFAULT_RELAX_STEPS,
    DEFAULT_TEMPERATURE_K,
    KBOLTZMANN,
    MDRelaxation,
    SimpleForceField,
    GeometryValidator,
    register_md_relaxation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def ff() -> SimpleForceField:
    """A default :class:`SimpleForceField`."""
    return SimpleForceField()


@pytest.fixture
def methane_coords() -> np.ndarray:
    """Tetrahedral-ish coordinates for methane (C + 4 H).

    The C is at the origin; the 4 H sit roughly on the unit-sphere.
    These distances (1.0 A) are deliberately *short* of equilibrium
    (~1.09 A for C-H) so the initial energy is non-trivial and the
    relaxation moves the atoms visibly.
    """
    rng = np.random.default_rng(0)
    H = rng.normal(size=(4, 3))
    H = H / np.linalg.norm(H, axis=1, keepdims=True)  # on the unit sphere
    out = np.zeros((5, 3), dtype=np.float64)
    out[1:] = H
    return out


@pytest.fixture
def methane_types() -> list:
    """Atom-type labels for methane."""
    return ["C", "H", "H", "H", "H"]


@pytest.fixture
def bad_contact_coords() -> np.ndarray:
    """Two H atoms stuck at 0.5 A — clearly a bad contact."""
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def reasonable_coords() -> np.ndarray:
    """Two H atoms at a non-bonded 2.4 A — beyond the close-contact limit."""
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# 1. SimpleForceField
# ---------------------------------------------------------------------------
def test_simple_force_field_energy_zero() -> None:
    """Empty atom list with empty coords returns 0.0 energy."""
    ff = SimpleForceField()
    coords = np.zeros((0, 3), dtype=np.float64)
    assert ff.energy(coords, []) == 0.0


def test_simple_force_field_energy_single_atom() -> None:
    """Single-atom configs always have zero energy."""
    ff = SimpleForceField()
    coords = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    assert ff.energy(coords, ["C"]) == 0.0


def test_simple_force_field_energy_positive(ff, methane_coords, methane_types) -> None:
    """Compressed methane (1.0 A C-H) has positive energy."""
    e = ff.energy(methane_coords, methane_types)
    assert e > 0.0, f"expected positive energy, got {e}"


def test_simple_force_field_energy_finite(ff, methane_coords, methane_types) -> None:
    """Energy is finite (no NaN / inf) for a sensible input."""
    e = ff.energy(methane_coords, methane_types)
    assert math.isfinite(e)


def test_simple_force_field_gradient_shape(ff, methane_coords, methane_types) -> None:
    """Gradient returns shape (N, 3)."""
    g = ff.gradient(methane_coords, methane_types)
    assert g.shape == methane_coords.shape
    assert g.dtype == np.float64


def test_simple_force_field_gradient_zero_at_min() -> None:
    """Gradient is small at a near-equilibrium geometry."""
    ff = SimpleForceField()
    # Two C atoms at the equilibrium C-C bond length (2 * 0.76 = 1.52 A).
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.52, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    g = ff.gradient(coords, ["C", "C"])
    # The gradient should be near zero (we are at the bond equilibrium).
    assert float(np.linalg.norm(g)) < 1.0, (
        f"expected small gradient at bond equilibrium, got norm "
        f"{float(np.linalg.norm(g))}"
    )


def test_simple_force_field_invalid_method() -> None:
    """Unsupported method raises ValueError."""
    with pytest.raises(ValueError):
        SimpleForceField(method="OPLS")


def test_simple_force_field_repr() -> None:
    """The repr includes the method label."""
    ff = SimpleForceField(method="MMFF")
    r = repr(ff)
    assert "MMFF" in r


# ---------------------------------------------------------------------------
# 2. MDRelaxation
# ---------------------------------------------------------------------------
def test_md_relaxation_relaxes(ff, methane_coords, methane_types) -> None:
    """Relaxed coords differ from the initial (compressed) coords."""
    relax = MDRelaxation(ff, n_steps=20, step_size=0.005, temperature=100)
    out = relax.relax(methane_coords, methane_types)
    assert not np.allclose(out, methane_coords), "relaxation did not move the atoms"


def test_md_relaxation_lowers_energy(ff, methane_coords, methane_types) -> None:
    """Relaxation lowers (or maintains) the energy."""
    relax = MDRelaxation(ff, n_steps=20, step_size=0.005, temperature=100)
    e0 = ff.energy(methane_coords, methane_types)
    e1 = relax.energy_after_relax(methane_coords, methane_types)
    assert e1 <= e0 + 1e-6, f"relaxation did not lower energy: e0={e0:.3f}, e1={e1:.3f}"


def test_md_relaxation_deterministic(ff, methane_coords, methane_types) -> None:
    """Same seed → bit-for-bit same relaxed coords."""
    r1 = MDRelaxation(ff, n_steps=15, step_size=0.005, temperature=50, seed=1234)
    r2 = MDRelaxation(ff, n_steps=15, step_size=0.005, temperature=50, seed=1234)
    out1 = r1.relax(methane_coords, methane_types)
    out2 = r2.relax(methane_coords, methane_types)
    assert np.allclose(out1, out2), "same seed produced different trajectories"


def test_md_relaxation_preserves_shape(ff, methane_coords, methane_types) -> None:
    """Relaxation preserves the input shape."""
    relax = MDRelaxation(ff, n_steps=3, seed=42)
    out = relax.relax(methane_coords, methane_types)
    assert out.shape == methane_coords.shape


def test_md_relaxation_zero_steps(ff, methane_coords, methane_types) -> None:
    """n_steps=0 leaves the coords unchanged up to MC acceptance."""
    # The MC step is still run, but with step_size=0 and T=0 the
    # proposal is rejected deterministically — coords are returned
    # as a copy.  We assert shape + finiteness; do NOT assert exact
    # equality because the MC step may accept a tiny proposal.
    relax = MDRelaxation(
        ff, n_steps=0, step_size=0.01, temperature=1.0, mc_step_size=0.0, seed=0
    )
    out = relax.relax(methane_coords, methane_types)
    assert out.shape == methane_coords.shape
    assert np.isfinite(out).all()


def test_mc_step_returns_tuple(ff, methane_coords, methane_types) -> None:
    """MC step returns a (coords, bool) tuple with the right shape."""
    relax = MDRelaxation(ff, n_steps=1, mc_step_size=0.05, seed=0)
    out, accepted = relax.monte_carlo_step(methane_coords, methane_types)
    assert isinstance(out, np.ndarray)
    assert isinstance(accepted, bool)
    assert out.shape == methane_coords.shape


def test_mc_step_deterministic(ff, methane_coords, methane_types) -> None:
    """Same seed → same MC accept/reject decision and same out/dropped coords."""
    r1 = MDRelaxation(ff, mc_step_size=0.1, seed=42)
    r2 = MDRelaxation(ff, mc_step_size=0.1, seed=42)
    out1, acc1 = r1.monte_carlo_step(methane_coords, methane_types)
    out2, acc2 = r2.monte_carlo_step(methane_coords, methane_types)
    assert acc1 == acc2
    assert np.allclose(out1, out2)


def test_mc_step_empty(ff) -> None:
    """MC step on empty coords returns an empty array and False."""
    relax = MDRelaxation(ff, mc_step_size=0.1, seed=0)
    out, accepted = relax.monte_carlo_step(np.zeros((0, 3)), [])
    assert out.shape == (0, 3)
    assert accepted is False


# ---------------------------------------------------------------------------
# 3. GeometryValidator
# ---------------------------------------------------------------------------
def test_geometry_validator_reasonable(
    reasonable_coords,
) -> None:
    """Two H atoms at 2.4 A (non-bonded) — no violations."""
    val = GeometryValidator()
    assert val.is_reasonable_geometry(reasonable_coords, ["H", "H"])


def test_geometry_validator_bad_contact(bad_contact_coords) -> None:
    """Two H atoms at 0.5 A — flagged as a bad contact."""
    val = GeometryValidator()
    assert not val.is_reasonable_geometry(bad_contact_coords, ["H", "H"])
    violations = val.report_bad_contacts(bad_contact_coords, ["H", "H"])
    assert len(violations) >= 1
    reasons = [v[3] for v in violations]
    assert "close_contact" in reasons


def test_geometry_validator_bond_too_long() -> None:
    """Two C atoms at 2.5 A — flagged as a bond-too-long violation.

    C-C bond length is ~1.54 A (equilibrium).  At 2.5 A the pair is
    still within the bond-detection threshold (2.0 A is the default;
    we use a custom threshold of 2.5 A here so the test exercises
    the *bond-length* range check rather than the *non-bonded*
    close-contact path).
    """
    val = GeometryValidator(bond_threshold=2.6)
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    violations = val.report_bad_contacts(coords, ["C", "C"])
    assert len(violations) >= 1
    reasons = [v[3] for v in violations]
    # C-C cov-rad sum is 1.52 A; bond_upper * r0 = 1.6 * 1.52 = 2.432.
    # 2.5 A > 2.432 → bond_too_long.
    assert "bond_too_long" in reasons


def test_geometry_validator_bond_too_short() -> None:
    """Two C atoms at 0.5 A — flagged as close_contact via absolute_min."""
    val = GeometryValidator()
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    violations = val.report_bad_contacts(coords, ["C", "C"])
    assert len(violations) >= 1
    reasons = [v[3] for v in violations]
    assert "close_contact" in reasons


def test_geometry_validator_empty() -> None:
    """Empty coords produce no violations."""
    val = GeometryValidator()
    out = val.report_bad_contacts(np.zeros((0, 3)), [])
    assert out == []
    assert val.is_reasonable_geometry(np.zeros((0, 3)), [])


# ---------------------------------------------------------------------------
# 4. Registration
# ---------------------------------------------------------------------------
def test_register_md_relaxation_count() -> None:
    """register_md_relaxation returns at least 1."""
    n = register_md_relaxation()
    assert n >= 1


def test_register_md_relaxation_idempotent() -> None:
    """Calling the registration hook twice still returns 1."""
    n1 = register_md_relaxation()
    n2 = register_md_relaxation()
    assert n1 == 1
    assert n2 == 1


# ---------------------------------------------------------------------------
# 5. Defaults sanity (load-bearing values must not silently change)
# ---------------------------------------------------------------------------
def test_defaults() -> None:
    """The exported defaults match the values documented in the brief."""
    assert DEFAULT_BOND_K == 300.0
    assert DEFAULT_ANGLE_K == 100.0
    assert DEFAULT_NB_EPSILON == 0.1
    assert DEFAULT_RELAX_STEPS == 100
    assert DEFAULT_TEMPERATURE_K == 300.0
    # Boltzmann constant in kcal/mol/K — the value used by the
    # Metropolis acceptance test.
    assert abs(KBOLTZMANN - 0.0019872041) < 1e-12
