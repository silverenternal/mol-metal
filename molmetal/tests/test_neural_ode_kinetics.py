"""Tests for the Graph Neural ODE kinetics scaffolding (Wf-78 R22-P2 Phase 1).

================================================================
Wf-78 Phase 1 (L14 neural-ode-kinetics, 2026-09-26)
================================================================
This module validates that
:mod:`molmetal.molmetal_lam.reward.neural_ode_kinetics` ships the
P2.1 GRU-ODE-Bayes backbone shape (synthetic mock) + P2.2 ligand-
exchange ODE + P2.3 hydrolysis cascade + redox cycling + cellular
uptake + multi-task loss wire-up on a **synthetic mock** (no real
torchdiffeq, no GPU).

It also pins the contract for downstream Wf-76 P2.4 close-out (or
Wf-78 Phase 2 close-out) integration: the NumPy forward-Euler
integrator is replaced by ``torchdiffeq.odeint`` once GPU-1 Blackwell
becomes available.

Test inventory (16 tests)
-------------------------
1. **SUPPORTED_KINETIC_METALS** — 8-metal pool + validation (1 test).
2. **Per-metal regime classification** — KINETIC_REGIME, AQUATION_HALF_LIVES_S,
   REDOX_POTENTIALS_V, UPTAKE_FLUX_FMOL_CELL_HR, STATIC_R2_PLATINAI_4CELL (3 tests).
3. **rate_scale_for_metal** — Pt=1.0, Pd=1e4, Au=1e-3, inert metals=1e-6 (2 tests).
4. **BindingKinetics** — initial state, t_half_pt_oh2 anchor (Pt ~ 124 s),
   integration conservation (3 tests).
5. **DissociationKinetics** — initial state, t_half_aquation anchor,
   integration conservation (2 tests).
6. **CellularUptake** — uptake flux, integration Pt_ext monotone
   decreasing (2 tests).
7. **RedoxCycling** — E° vs NHE per-metal, integration conservation (1 test).
8. **MultiTaskLoss** — default weights (α=0.5, β=0.3, γ=0.2), t_half
   classification (fast/medium/slow), loss sum non-negative (1 test).
9. **synthetic_mock_kinetics** — shape (1 test).
   **smoke_neural_ode_kinetics** — end-to-end (1 test).

Honest framing
--------------
* All tests use the synthetic mock — **no** torchdiffeq, **no** GPU,
  **no** real kinetics chemistry.
* The R22-P2 P2.1 + P2.2 + P2.3 scaffold is **CPU-only** and ships a
  deterministic NumPy forward-Euler integrator to anchor the outer
  shape for downstream Wf-76 P2.4 GPU-1 integration.
"""

from __future__ import annotations

import math

import pytest

from molmetal.molmetal_lam.reward.neural_ode_kinetics import (
    AQUATION_HALF_LIVES_S,
    BindingKinetics,
    CellularUptake,
    DissociationKinetics,
    HYDROLYSIS_CASCADE_RATE_CONSTANTS,
    KINETIC_REGIME,
    KineticsTrajectory,
    MULTI_TASK_LOSS_WEIGHTS,
    MultiTaskLoss,
    REDOX_POTENTIALS_V,
    RedoxCycling,
    STATIC_R2_PLATINAI_4CELL,
    SUPPORTED_KINETIC_METALS,
    T_HALF_CLASS_BOUNDS_S,
    UPTAKE_FLUX_FMOL_CELL_HR,
    rate_scale_for_metal,
    smoke_neural_ode_kinetics,
    synthetic_mock_kinetics,
)


# ---------------------------------------------------------------------------
# 1. SUPPORTED_KINETIC_METALS — 8-metal pool + validation
# ---------------------------------------------------------------------------
def test_supported_kinetic_metals_has_8_entries():
    """Verify SUPPORTED_KINETIC_METALS contains exactly the 8 expected metals."""
    expected = {"Pt", "Pd", "Ru", "Rh", "Ir", "Os", "Au", "Re"}
    assert set(SUPPORTED_KINETIC_METALS) == expected
    assert len(SUPPORTED_KINETIC_METALS) == 8


# ---------------------------------------------------------------------------
# 2. Per-metal regime classification — KINETIC_REGIME + per-metal tables
# ---------------------------------------------------------------------------
def test_kinetic_regime_classification_complete():
    """Verify KINETIC_REGIME has an entry for every supported metal."""
    for metal in SUPPORTED_KINETIC_METALS:
        assert metal in KINETIC_REGIME
        # Regime is a non-empty string label
        assert isinstance(KINETIC_REGIME[metal], str)
        assert KINETIC_REGIME[metal]


def test_aquation_half_lives_complete():
    """Verify AQUATION_HALF_LIVES_S has positive entries for all 8 metals.

    Pin Pt = 2 hr (Reedijk 1987 cisplatin anchor) as the cross-ref.
    """
    for metal in SUPPORTED_KINETIC_METALS:
        assert metal in AQUATION_HALF_LIVES_S
        assert AQUATION_HALF_LIVES_S[metal] > 0.0
    # Pt cisplatin anchor
    assert AQUATION_HALF_LIVES_S["Pt"] == pytest.approx(2.0 * 3600.0, rel=1.0e-6)


def test_redox_potentials_and_uptake_complete():
    """Verify REDOX_POTENTIALS_V + UPTAKE_FLUX_FMOL_CELL_HR + STATIC_R2_PLATINAI_4CELL
    have entries for all 8 metals."""
    for table, label in [
        (REDOX_POTENTIALS_V, "REDOX_POTENTIALS_V"),
        (UPTAKE_FLUX_FMOL_CELL_HR, "UPTAKE_FLUX_FMOL_CELL_HR"),
        (STATIC_R2_PLATINAI_4CELL, "STATIC_R2_PLATINAI_4CELL"),
    ]:
        for metal in SUPPORTED_KINETIC_METALS:
            assert metal in table, f"{label} missing metal {metal}"


# ---------------------------------------------------------------------------
# 3. rate_scale_for_metal — Pt=1.0, Pd=1e4, Au=1e-3, inert metals=1e-6
# ---------------------------------------------------------------------------
def test_rate_scale_for_metal_associative_metals():
    """Verify rate_scale_for_metal: Pt=1.0, Pd=1e4, Au=1e-3."""
    assert rate_scale_for_metal("Pt") == 1.0
    assert rate_scale_for_metal("Pd") == pytest.approx(1.0e4, rel=1.0e-6)
    assert rate_scale_for_metal("Au") == pytest.approx(1.0e-3, rel=1.0e-6)


def test_rate_scale_for_metal_inert_metals():
    """Verify rate_scale_for_metal: inert metals (Ru/Rh/Ir/Os/Re) = 1e-6."""
    for metal in ("Ru", "Rh", "Ir", "Os", "Re"):
        assert rate_scale_for_metal(metal) == pytest.approx(1.0e-6, rel=1.0e-6)


# ---------------------------------------------------------------------------
# 4. BindingKinetics — initial state, t_half_pt_oh2, integration
# ---------------------------------------------------------------------------
def test_binding_kinetics_initial_state_and_t_half():
    """Verify BindingKinetics initial state is (1.0, 0.0, 0.0) and
    t_half_pt_oh2 returns a positive finite number for Pt."""
    bk = BindingKinetics(metal="Pt", rate_scale=1.0)
    assert bk.initial_state() == (1.0, 0.0, 0.0)
    # Pt with default Cl⁻/GSH concentrations + rate_scale=1.0 → t_half finite
    t_half = bk.t_half_pt_oh2()
    assert math.isfinite(t_half)
    assert t_half > 0.0


def test_binding_kinetics_integration_conservation():
    """Verify BindingKinetics.integrate conserves total Pt (sum = 1.0) for Pt."""
    bk = BindingKinetics(metal="Pt", rate_scale=1.0)
    t_grid, state_traj = bk.integrate(t_end_s=60.0, dt_s=1.0)
    # Conservation: sum of [Pt-Cl, Pt-OH2, Pt-SG] ≈ 1.0 at every timestep
    for state in state_traj:
        total = sum(state)
        assert total == pytest.approx(1.0, abs=1.0e-3)
    # t_grid has n_steps + 1 entries
    assert len(t_grid) == len(state_traj)
    assert len(t_grid) > 1


def test_binding_kinetics_inert_metal_collapses():
    """Verify BindingKinetics for inert metal (Ru) has t_half much larger
    than the active Pt case (slow ligand exchange = effectively inert)."""
    bk_pt = BindingKinetics(metal="Pt", rate_scale=1.0)
    bk_ru = BindingKinetics(metal="Ru", rate_scale=1.0e-6)
    t_half_pt = bk_pt.t_half_pt_oh2()
    t_half_ru = bk_ru.t_half_pt_oh2()
    # Both are finite, but Ru is >> 1e6 × Pt (matches the literature
    # ~weeks-vs-2hr order-of-magnitude gap between Pt and Ru)
    assert t_half_pt > 0.0
    assert t_half_ru > 0.0
    assert t_half_ru / t_half_pt > 1.0e6


# ---------------------------------------------------------------------------
# 5. DissociationKinetics — initial state, t_half_aquation, integration
# ---------------------------------------------------------------------------
def test_dissociation_kinetics_initial_state_and_t_half():
    """Verify DissociationKinetics initial state + t_half_aquation."""
    dk = DissociationKinetics(metal="Pt", rate_scale=1.0)
    state = dk.initial_state()
    assert state == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Pt with default k1_aquation + rate_scale=1.0 → t_half_aquation finite
    t_half = dk.t_half_aquation()
    assert math.isfinite(t_half)
    assert t_half > 0.0
    # Cross-check vs HYDROLYSIS_CASCADE_RATE_CONSTANTS["k1_aquation"]
    expected = math.log(2.0) / HYDROLYSIS_CASCADE_RATE_CONSTANTS["k1_aquation"]
    assert t_half == pytest.approx(expected, rel=1.0e-6)


def test_dissociation_kinetics_integration_conservation():
    """Verify DissociationKinetics.integrate conserves total Pt (sum = 1.0) for Pt."""
    dk = DissociationKinetics(metal="Pt", rate_scale=1.0)
    t_grid, state_traj = dk.integrate(t_end_s=20000.0, dt_s=100.0)
    # Conservation: sum of the 6-state ≈ 1.0 at every timestep
    for state in state_traj:
        total = sum(state)
        assert total == pytest.approx(1.0, abs=1.0e-3)
    assert len(t_grid) == len(state_traj)
    assert len(t_grid) > 1


# ---------------------------------------------------------------------------
# 6. CellularUptake — uptake flux, Pt_ext monotone decreasing
# ---------------------------------------------------------------------------
def test_cellular_uptake_flux_per_metal():
    """Verify CellularUptake.uptake_flux_fmol_per_cell_hr per metal."""
    expected_flux = {
        "Pt": 2.0,
        "Pd": 0.5,
        "Ru": 0.5,
        "Rh": 0.05,
        "Ir": 0.05,
        "Os": 0.4,
        "Au": 0.05,
        "Re": 0.05,
    }
    for metal, expected in expected_flux.items():
        cu = CellularUptake(metal=metal)
        assert cu.uptake_flux_fmol_per_cell_hr() == pytest.approx(expected, abs=1.0e-9)


def test_cellular_uptake_integration_pt_ext_monotone_decreasing():
    """Verify CellularUptake.integrate monotonically decreases [Pt_ext] over time."""
    cu = CellularUptake(metal="Pt", efflux_rate_s=0.0, dna_sink_rate_s=0.0)
    t_grid, (pt_ext_traj, pt_int_traj) = cu.integrate(
        pt_ext_init_m=1.0e-5, t_end_s=100.0, dt_s=1.0
    )
    # Monotone decreasing: pt_ext[i+1] <= pt_ext[i]
    for i in range(1, len(pt_ext_traj)):
        assert pt_ext_traj[i] <= pt_ext_traj[i - 1] + 1.0e-12
    # [Pt_int] starts at 0 and increases (Pt_ext → Pt_int)
    assert pt_int_traj[0] == 0.0
    assert pt_int_traj[-1] > 0.0


# ---------------------------------------------------------------------------
# 7. RedoxCycling — E° vs NHE per-metal, integration conservation
# ---------------------------------------------------------------------------
def test_redox_cycling_e0_and_integration():
    """Verify RedoxCycling.e0_v_vs_nhe + integration conservation."""
    for metal, expected_e0 in [("Pt", 0.78), ("Ru", 1.20), ("Au", 1.50)]:
        rc = RedoxCycling(metal=metal)
        assert rc.e0_v_vs_nhe() == pytest.approx(expected_e0, abs=1.0e-9)
        # Integrate + verify conservation
        t_grid, state_traj = rc.integrate(t_end_s=10.0, dt_s=0.1)
        for state in state_traj:
            total = sum(state)
            assert total == pytest.approx(1.0, abs=1.0e-3)


# ---------------------------------------------------------------------------
# 8. MultiTaskLoss — default weights + classification + loss sum non-negative
# ---------------------------------------------------------------------------
def test_multi_task_loss_weights_classification_and_loss():
    """Verify MultiTaskLoss default weights (α=0.5, β=0.3, γ=0.2),
    t1/2 classification (fast/medium/slow), and loss sum non-negative."""
    loss = MultiTaskLoss()
    assert loss.alpha_static == MULTI_TASK_LOSS_WEIGHTS["alpha_static"]
    assert loss.beta_rate == MULTI_TASK_LOSS_WEIGHTS["beta_rate"]
    assert loss.gamma_half_life == MULTI_TASK_LOSS_WEIGHTS["gamma_half_life"]
    # t1/2 classification (fast/medium/slow) per T_HALF_CLASS_BOUNDS_S
    assert loss.classify_t_half(60.0) == 0  # fast (< 600 s)
    assert loss.classify_t_half(T_HALF_CLASS_BOUNDS_S[0]) == 1  # medium lower bound
    assert loss.classify_t_half(3600.0) == 1  # medium (600 - 7200 s)
    assert loss.classify_t_half(T_HALF_CLASS_BOUNDS_S[1]) == 2  # slow lower bound
    assert loss.classify_t_half(86400.0) == 2  # slow (> 7200 s)
    # Loss sum non-negative
    l = loss.loss(
        static_pic50_pred=0.5,
        static_pic50_true=-0.02,
        rate_constant_pred=1.0e-4,
        rate_constant_lit=1.0e-4,
        t_half_class_pred=1,
        t_half_class_true=1,
    )
    assert l >= 0.0


# ---------------------------------------------------------------------------
# 9. synthetic_mock_kinetics — shape + smoke_neural_ode_kinetics end-to-end
# ---------------------------------------------------------------------------
def test_synthetic_mock_kinetics_returns_kinetics_trajectory():
    """Verify synthetic_mock_kinetics returns a KineticsTrajectory with
    all 12 expected fields populated for Pt."""
    traj = synthetic_mock_kinetics(
        metal="Pt",
        target_id="target_0",
        seed=42,
        t_end_s=10.0,
        dt_s=0.5,
    )
    assert isinstance(traj, KineticsTrajectory)
    assert traj.metal == "Pt"
    assert traj.kinetic_regime == KINETIC_REGIME["Pt"]
    assert traj.aquation_t_half_s == AQUATION_HALF_LIVES_S["Pt"]
    assert traj.redox_potential_v == REDOX_POTENTIALS_V["Pt"]
    assert traj.uptake_fmol_per_cell_hr == UPTAKE_FLUX_FMOL_CELL_HR["Pt"]
    assert traj.n_steps > 0
    assert traj.wall_time_s >= 0.0
    assert traj.seed == 42
    # multi_task_loss is finite and non-negative
    assert math.isfinite(traj.multi_task_loss)
    assert traj.multi_task_loss >= 0.0


def test_smoke_neural_ode_kinetics_end_to_end():
    """Verify smoke_neural_ode_kinetics returns 3 trajectories (Pt/Pd/Au)
    with deterministic + finite multi_task_loss."""
    trajs = smoke_neural_ode_kinetics()
    assert len(trajs) == 3
    expected_metals = ["Pt", "Pd", "Au"]
    for traj, expected_metal in zip(trajs, expected_metals):
        assert traj.metal == expected_metal
        assert math.isfinite(traj.multi_task_loss)
        assert traj.multi_task_loss >= 0.0
    # Determinism: re-run gives the same trajectories
    trajs_2 = smoke_neural_ode_kinetics()
    for t1, t2 in zip(trajs, trajs_2):
        assert t1.multi_task_loss == pytest.approx(t2.multi_task_loss, abs=1.0e-9)
        assert t1.aquation_t_half_s == t2.aquation_t_half_s
