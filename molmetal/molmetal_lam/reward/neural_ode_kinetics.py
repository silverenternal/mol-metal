"""Graph Neural ODE for metallodrug kinetics scaffolding (Wf-78, R22-P2 Phase 1).

================================================================
Background — R22-P2 (L14 Graph Neural ODE kinetics, 2026-09-26)
================================================================
TODO-106 (Wf-78 R22-P2 Phase 1 CPU scaffolding) authors the **first
half of the L14 Lambda gap** (see ``TODO/pending/r22_l13_l16_gap_analysis.md``
§2). The L5/L7/L8 stack measures **static pIC50 R²** — a single
snapshot of binding affinity — but real metallodrug efficacy depends
on **dynamic lability** (ligand-exchange rate), **kinetic trans-effect**
(cisplatin aquation t1/2 ≈ 2 hr per Reedijk 1987), and **redox
cycling** (Pt(II) ↔ Pt(IV) under oxidative stress). None of those are
captured by the static ASTSR R².

The CPU smoke Wf-49 P2 returns PlatinAI 4-cell R² ≈ -0.02 (collapse
detector fires at epoch 4) per ``completed/93_r20_p2_hilbert_astsr_gpu.md``
§1.1 — but this is a **static-snapshot failure mode**, not a
kinetics-failure mode. No current L1-L12 row addresses continuous-time
dynamics.

L14 (Graph Neural ODE kinetics) is the **MEDIUM-priority candidate**
with 20 scan signals (6% of 330) per the gap analysis. R22-P2
implements L14 with a GRU-ODE-Bayes backbone (Chen et al. 2018
NeurIPS) + 4-step hydrolysis cascade + Ctr1/OCT2 cellular uptake
ODE + Pt(II)/Ru(II) redox cycling ODE.

This module authors the **CPU-only scaffold** (P2.1 backbone + P2.2
ligand-exchange + P2.3 hydrolysis + redox + uptake). It is a **pure
NumPy / no-torchdiffeq** scaffold so that the outer shape, per-metal
regimes, per-mechanism breakdown, and the multi-task loss wire-up
can be inspected without a GPU. The Wf-76 P2.4 close-out (or Wf-78
Phase 2 close-out) replaces the NumPy integrator with
``torchdiffeq.odeint`` once GPU-1 Blackwell is available.

The P2.1 + P2.2 + P2.3 outputs are:

  * :class:`BindingKinetics`      — ligand-exchange ODE (Cl⁻/H₂O/GSH).
  * :class:`DissociationKinetics` — 4-step cisplatin hydrolysis cascade.
  * :class:`CellularUptake`       — Ctr1/OCT2/passive ODE.
  * :class:`RedoxCycling`         — Pt(II)↔Pt(IV) + Ru(II)↔Ru(III) ODE.
  * :class:`MultiTaskLoss`        — α·R²_static + β·MSE_rate + γ·CE_t1/2.
  * :func:`synthetic_mock_kinetics` — deterministic mock kinetics driver
    returning a :class:`KineticsTrajectory` for the (Pt, target) pair.
  * :func:`smoke_neural_ode_kinetics` — end-to-end smoke call
    (1 metal × 1 target × 3 mechanisms × 200-step integration).

Honest framing
--------------
* This is a **CPU-only scaffold**. No torchdiffeq, no GPU runs, no
  literature kinetics MEASURED on disk. The integrator is plain
  NumPy forward-Euler with synthetic mock kinetics — chemistry is
  **modelled qualitatively** from Reedijk 1987 + Hartinger 2009 +
  Peacock 2006 + Taube 1952 + Fricker 1996 + Shaw 1999 + Hughes 2010.
* The scaffold is **shape-fixing only**. Per-metal rate constants are
  seeded from the literature tables in §3.4 of the S10 SKILL.md
  (``mol-metal-neural-ode-kinetics/SKILL.md``); the multi-task loss
  wires up the Wf-76 P2.3 anchor (α=0.5, β=0.3, γ=0.2) but does not
  retrain against a real dataset.
* Per :class:`KineticsTrajectory`, the observable metrics are
  ``metal``, ``kinetic_regime``, ``aquation_t_half_s``,
  ``redox_potential_v``, ``uptake_fmol_per_cell_hr``, ``static_loss``,
  ``rate_constant_loss``, ``t_half_class_loss``, ``multi_task_loss``.
  All computed deterministically from the synthetic mock.
* The R17 surface (closure.py + lam_chem/ + reward/lambda_astsr.py)
  is **not modified**. This module sits next to
  ``hierarchical_multimetal.py`` (Wf-78 R22-P1 P1.1+P1.2) as a
  parallel scaffold for R22-P2.

Public API
----------
* :class:`BindingKinetics`           — ligand-exchange ODE driver.
* :class:`DissociationKinetics`      — 4-step hydrolysis cascade.
* :class:`CellularUptake`            — Ctr1/OCT2/passive ODE driver.
* :class:`RedoxCycling`              — Pt/Ru redox cycling ODE driver.
* :class:`MultiTaskLoss`             — multi-task loss dataclass.
* :class:`KineticsTrajectory`        — result dataclass.
* :func:`synthetic_mock_kinetics`    — deterministic mock driver.
* :func:`smoke_neural_ode_kinetics`  — end-to-end smoke runner.
* :data:`SUPPORTED_KINETIC_METALS`   — 8-metal pool constant.
* :data:`KINETIC_REGIME`             — per-metal regime classification.
* :data:`AQUATION_HALF_LIVES_S`      — per-metal t1/2 literature values.
* :data:`REDOX_POTENTIALS_V`         — per-metal E° vs NHE.
* :data:`UPTAKE_FLUX_FMOL_CELL_HR`   — per-metal Ctr1/OCT2/passive flux.

Lit anchors
-----------
* Reedijk 1987 *Platinum Metals Rev.* (cisplatin aquation kinetics,
  Cl⁻/H₂O/GSH trans-effect series).
* Hartinger 2009 (GSH trapping kinetics).
* Peacock 2006 review (Pt(II)/Pt(IV) redox cycling).
* Taube 1952 (Ru(II)/Ru(III) + Os(II)/Os(III) inert regime).
* Chen, Rubanova, Bettencourt, Duvenaud 2018 *Neural Ordinary
  Differential Equations* (NeurIPS 2018, arXiv:1806.07366) — GRU-ODE-Bayes
  backbone. (R22-P2 P2.1 reference; the actual torchdiffeq backbone
  is deferred to Wf-76 P2.1 GPU-1 retrain.)
* Fricker 1996 (auranofin Au(I) thiol exchange).
* Shaw 1999 (Au(III) redox cycling).
* Hughes 2010 (Rh(III) kinetically inert scaffold).
* Lippard 1995 (Pt(II) square-planar associative substitution).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Metal pool — supported d-block metal centers for kinetics
# ---------------------------------------------------------------------------
# Eight d-block centers spanning 4d (Ru/Rh/Pd) and 5d (Os/Ir/Pt/Au)
# plus the L12 bottleneck metal Re.  This pool matches
# :data:`molmetal.molmetal_lam.reward.hierarchical_multimetal.SUPPORTED_METALS`
# so that L13 (hierarchical coord) and L14 (kinetics) cross-references
# are consistent.
SUPPORTED_KINETIC_METALS: Tuple[str, ...] = (
    "Pt",
    "Pd",
    "Ru",
    "Rh",
    "Ir",
    "Os",
    "Au",
    "Re",
)

#: Kinetic regime classification per metal (Reedijk 1987 + Taube 1952
#: + Hughes 2010 + Fricker 1996 + Shaw 1999).  Values are short labels
#: used for diagnostic breakdowns in :class:`KineticsTrajectory`.
KINETIC_REGIME: Dict[str, str] = {
    "Pt": "associative_slow",      # t1/2 ~ 2 hr for aquation (cisplatin anchor)
    "Pd": "associative_fast",      # ~10⁴× faster than Pt(II) (trans-effect)
    "Ru": "inert_octahedral",      # t1/2 ~ weeks; photoactivatable
    "Rh": "inert_scaffold",        # t1/2 ~ years; used as scaffold
    "Ir": "very_inert",            # t1/2 ~ months; used as inert binder
    "Os": "inert_octahedral",      # similar to Ru but slower (5d contraction)
    "Au": "thiol_fast_auranofin",  # t1/2 ~ minutes with serum albumin
    "Re": "inert_scaffold",        # similar to Rh; mid-lanthanide proxy
}

#: Per-metal aquation half-life in seconds (Reedijk 1987 + Hartinger
#: 2009 + Taube 1952 + Fricker 1996 + Hughes 2010).  These are the
#: **rate-limiting-step** half-lives for the first aquation.  Used
#: both as priors for the ligand-exchange ODE and as the literature
#: target for the t1/2 RMSE verification gate (P2.2).
AQUATION_HALF_LIVES_S: Dict[str, float] = {
    "Pt": 2.0 * 3600.0,            # ~ 2 hr (cisplatin anchor, Reedijk 1987)
    "Pd": 2.0 * 3600.0 / 1.0e4,    # ~ 0.72 s (10⁴× faster than Pt)
    "Ru": 2.0 * 7.0 * 24.0 * 3600.0,  # ~ 2 weeks (inert octahedral)
    "Rh": 2.0 * 365.0 * 24.0 * 3600.0,  # ~ 2 years (scaffold)
    "Ir": 2.0 * 30.0 * 24.0 * 3600.0,   # ~ 2 months (very inert)
    "Os": 2.0 * 14.0 * 24.0 * 3600.0,   # ~ 2 weeks (similar to Ru)
    "Au": 5.0 * 60.0,                   # ~ 5 min (auranofin thiol exchange)
    "Re": 2.0 * 90.0 * 24.0 * 3600.0,   # ~ 6 months (scaffold proxy)
}

#: Per-metal redox potential in volts vs NHE for the M²⁺/M³⁺ or
#: M²⁺/M⁴⁺ couple (Peacock 2006 + Taube 1952 + Shaw 1999).
REDOX_POTENTIALS_V: Dict[str, float] = {
    "Pt": 0.78,   # Pt²⁺/Pt⁴⁺ couple
    "Pd": 0.95,   # Pd²⁺/Pd⁴⁺ couple (similar to Pt, slightly higher)
    "Ru": 1.20,   # Ru²⁺/Ru³⁺ couple (Taube 1952)
    "Rh": 0.76,   # Rh²⁺/Rh³⁺ couple (proxy)
    "Ir": 0.86,   # Ir³⁺/Ir⁴⁺ couple
    "Os": 0.85,   # Os²⁺/Os³⁺ couple (similar to Ru but slower)
    "Au": 1.50,   # Au(I)/Au(III) couple (Shaw 1999)
    "Re": 0.30,   # Re(III)/Re(IV) couple (proxy)
}

#: Per-metal cellular uptake flux in fmol/cell/hr via the **dominant
#: transporter** (Ctr1 + OCT2 + passive per ORNL SOP-3 anchor in
#: ``molmetal/docs/wetlab_protocols/sop_cellular_uptake_assay.md``).
UPTAKE_FLUX_FMOL_CELL_HR: Dict[str, float] = {
    "Pt": 2.0,    # Ctr1 dominant (cisplatin anchor)
    "Pd": 0.5,    # Ctr1 moderate (Reedijk 2008)
    "Ru": 0.5,    # OCT2 dominant (Taube 1952)
    "Rh": 0.05,   # minimal uptake (Hughes 2010)
    "Ir": 0.05,   # minimal uptake (Hughes 2010)
    "Os": 0.4,    # similar to Ru
    "Au": 0.05,   # passive (Fricker 1996)
    "Re": 0.05,   # minimal uptake (scaffold proxy)
}

#: Per-metal static pIC50 loss contribution (mock R² baseline
#: borrowed from L5/L7 ship at PlatinAI 4-cell per
#: ``completed/100_wf71_p3_wf49_cross_round2.md``).  Negative values
#: are honest-static collapse (predict-the-mean or worse); only
#: A2780cis shows R²=+0.037.  These are the **static** loss terms
#: the multi-task loss anchors against.
STATIC_R2_PLATINAI_4CELL: Dict[str, float] = {
    "Pt": -0.02,     # 4-cell macro R² collapse detector fires
    "Pd": -0.05,     # no PlatinAI Pd data; honest-static predict-mean
    "Ru": -0.05,
    "Rh": -0.05,
    "Ir": -0.05,
    "Os": -0.05,
    "Au": -0.05,
    "Re": -0.05,
}


# ---------------------------------------------------------------------------
# Rate constants for ligand-exchange ODE (Reedijk 1987 + Hartinger 2009)
# ---------------------------------------------------------------------------
# Units: M⁻¹s⁻¹.  These are the learnable priors the S10 SKILL.md §3.4
# anchors; here we freeze them as module-level constants for the CPU
# scaffold (the Wf-76 P2.1 GPU-1 retrain will make them learnable).
LIGAND_EXCHANGE_RATE_CONSTANTS: Dict[str, float] = {
    "Cl_to_H2O": 1.0e-4,    # Pt(II) aquation rate (M⁻¹s⁻¹)
    "H2O_to_Cl": 1.0e2,     # reverse (Cl⁻ abundant in cell)
    "H2O_to_GSH": 1.0e-1,   # GSH trapping (Hartinger 2009)
    "GSH_to_H2O": 1.0e-3,   # reversible
    "Cl_to_GSH": 1.0e-3,    # direct Cl→GSH (slower than via aqua)
}

#: Hydrolysis cascade 4-step rate constants (Reedijk 1987 +
#: Hartinger 2009).  k1 = rate-limiting aquation.
HYDROLYSIS_CASCADE_RATE_CONSTANTS: Dict[str, float] = {
    "k1_aquation": 1.0e-4,   # Step 1: Cl⁻ loss (rate-limiting)
    "k2_second_loss": 1.0e-2,  # Step 2: 2nd Cl⁻ loss
    "pKa1_Pt_OH2": 6.5,      # Step 3: Pt-OH2 ⇌ Pt-OH (Reedijk 1987)
    "k_bind_DNA_N7": 1.0e2,  # Step 4: G-N7 capture
}

#: Multi-task loss weights (S10 SKILL.md §2.4 anchor).
MULTI_TASK_LOSS_WEIGHTS: Dict[str, float] = {
    "alpha_static": 0.5,    # static R² term (preserves L5 baseline)
    "beta_rate": 0.3,       # kinetic rate constant MSE
    "gamma_half_life": 0.2, # t1/2 classification (fast / medium / slow)
}

#: t1/2 classification thresholds (seconds): fast < 600 s (10 min),
#: medium 600-7200 s (10 min - 2 hr), slow > 7200 s.
T_HALF_CLASS_BOUNDS_S: Tuple[float, float] = (600.0, 7200.0)


# ---------------------------------------------------------------------------
# BindingKinetics — ligand-exchange ODE driver (CPU-only NumPy)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BindingKinetics:
    """Ligand-exchange ODE driver for Pt/Pd/Au square-planar complexes.

    Models the 3-state system [Pt-Cl, Pt-OH2, Pt-SG] under the
    5-rate-constant ODE in :data:`LIGAND_EXCHANGE_RATE_CONSTANTS`.
    The CPU scaffold uses NumPy forward-Euler integration with a
    fixed timestep ``dt_s``; Wf-76 P2.2 will swap this for
    ``torchdiffeq.odeint`` with dopri5.

    Attributes
    ----------
    metal : str
        One of :data:`SUPPORTED_KINETIC_METALS`.  Only metals with a
        non-inert regime (Pt, Pd, Au) drive meaningful ligand exchange;
        the inert metals (Ru, Rh, Ir, Os) collapse to ~zero flux.
    cl_conc_m : float
        Extracellular Cl⁻ concentration in M (default 4.0e-3 = 4 mM
        plasma anchor per Reedijk 1987).
    gsh_conc_m : float
        Intracellular GSH concentration in M (default 1.0e-3 = 1 mM
        cytosolic GSH per Hartinger 2009).
    h2o_conc_m : float
        Bulk water concentration in M (default 55.5 M).
    rate_scale : float
        Per-metal scaling factor (1.0 for Pt; 1.0e4 for Pd; 1.0e-3 for
        Au inert-mode; 0.0 for fully-inert metals).
    """

    metal: str
    cl_conc_m: float = 4.0e-3
    gsh_conc_m: float = 1.0e-3
    h2o_conc_m: float = 55.5
    rate_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.metal not in SUPPORTED_KINETIC_METALS:
            raise ValueError(
                f"BindingKinetics: unknown metal {self.metal!r}; "
                f"supported = {SUPPORTED_KINETIC_METALS}"
            )
        if self.cl_conc_m < 0.0 or self.gsh_conc_m < 0.0 or self.h2o_conc_m <= 0.0:
            raise ValueError(
                f"BindingKinetics: negative concentrations "
                f"(cl={self.cl_conc_m}, gsh={self.gsh_conc_m}, h2o={self.h2o_conc_m})"
            )
        if self.rate_scale < 0.0:
            raise ValueError(
                f"BindingKinetics: rate_scale must be non-negative; got {self.rate_scale}"
            )

    def initial_state(self) -> Tuple[float, float, float]:
        """Return the initial [Pt-Cl, Pt-OH2, Pt-SG] state (sum = 1.0)."""
        return (1.0, 0.0, 0.0)

    def derivatives(
        self, state: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Compute the 3-state ODE derivatives.

        Forward-Euler derivative:
            d[Pt-Cl]/dt = -k_Cl→H2O * [Pt-Cl] * [H2O]
                          + k_H2O→Cl * [Pt-OH2] * [Cl⁻]
                          - k_Cl→GSH  * [Pt-Cl] * [GSH]
            d[Pt-OH2]/dt = +k_Cl→H2O * [Pt-Cl] * [H2O]
                           - k_H2O→Cl * [Pt-OH2] * [Cl⁻]
                           - k_H2O→GSH * [Pt-OH2] * [GSH]
                           + k_GSH→H2O * [Pt-SG] * [H2O]
            d[Pt-SG]/dt = +k_Cl→GSH  * [Pt-Cl] * [GSH]
                          + k_H2O→GSH * [Pt-OH2] * [GSH]
                          - k_GSH→H2O * [Pt-SG] * [H2O]
        """
        pt_cl, pt_oh2, pt_sg = state
        k = LIGAND_EXCHANGE_RATE_CONSTANTS
        s = self.rate_scale

        # Note: Pt-Cl + Pt-OH2 + Pt-SG must sum to [Pt]_total (here 1.0).
        d_pt_cl = (
            -k["Cl_to_H2O"] * pt_cl * self.h2o_conc_m * s
            + k["H2O_to_Cl"] * pt_oh2 * self.cl_conc_m * s
            - k["Cl_to_GSH"] * pt_cl * self.gsh_conc_m * s
        )
        d_pt_oh2 = (
            +k["Cl_to_H2O"] * pt_cl * self.h2o_conc_m * s
            - k["H2O_to_Cl"] * pt_oh2 * self.cl_conc_m * s
            - k["H2O_to_GSH"] * pt_oh2 * self.gsh_conc_m * s
            + k["GSH_to_H2O"] * pt_sg * self.h2o_conc_m * s
        )
        d_pt_sg = (
            +k["Cl_to_GSH"] * pt_cl * self.gsh_conc_m * s
            + k["H2O_to_GSH"] * pt_oh2 * self.gsh_conc_m * s
            - k["GSH_to_H2O"] * pt_sg * self.h2o_conc_m * s
        )
        return (d_pt_cl, d_pt_oh2, d_pt_sg)

    def integrate(
        self, t_end_s: float, dt_s: float
    ) -> Tuple[List[float], List[Tuple[float, float, float]]]:
        """Forward-Euler integrate from t=0 to t=t_end_s with timestep dt_s.

        Returns ``(t_grid, state_traj)`` where ``state_traj[i]`` is the
        3-state tuple at ``t_grid[i]``.  Conservation is enforced via
        a soft renormalization at each step (sum → 1.0).
        """
        if t_end_s <= 0.0 or dt_s <= 0.0:
            raise ValueError(
                f"BindingKinetics.integrate: t_end_s={t_end_s}, dt_s={dt_s} "
                f"must both be > 0"
            )
        n_steps = max(1, int(math.ceil(t_end_s / dt_s)))
        actual_dt = t_end_s / n_steps
        state = self.initial_state()
        t_grid = [0.0]
        state_traj = [state]
        for _ in range(n_steps):
            deriv = self.derivatives(state)
            new_state = tuple(
                max(0.0, s + d * actual_dt) for s, d in zip(state, deriv)
            )
            total = sum(new_state)
            if total > 0.0:
                new_state = tuple(s / total for s in new_state)
            state = new_state
            t_grid.append(t_grid[-1] + actual_dt)
            state_traj.append(state)
        return t_grid, state_traj

    def t_half_pt_oh2(self) -> float:
        """Approximate time for [Pt-OH2] to reach 0.5 (half-aquated)."""
        # For Pt-Cl starting at 1.0, t_half ≈ ln(2) / (k_Cl→H2O * [H2O])
        k = LIGAND_EXCHANGE_RATE_CONSTANTS["Cl_to_H2O"]
        denom = k * self.h2o_conc_m * self.rate_scale
        if denom <= 0.0:
            return float("inf")
        return math.log(2.0) / denom


# ---------------------------------------------------------------------------
# DissociationKinetics — 4-step hydrolysis cascade (Reedijk 1987)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DissociationKinetics:
    """4-step cisplatin hydrolysis cascade.

    Models the 6-state system
    [PtCl2, PtClOH2, Pt(OH2)2, PtOH, PtN7, PtOHG]
    under the rate constants in :data:`HYDROLYSIS_CASCADE_RATE_CONSTANTS`.

    The cascade is the canonical Reedijk 1987 + Hartinger 2009 reference
    for cisplatin activation kinetics.  The CPU scaffold uses NumPy
    forward-Euler; Wf-76 P2.3 will swap for ``torchdiffeq.odeint``.

    Attributes
    ----------
    metal : str
        One of :data:`SUPPORTED_KINETIC_METALS`.  Only Pt is meaningful;
        Pd is scaled by 1e4 (10⁴× faster); inert metals (Ru/Rh/Ir/Os/Re)
        collapse to ~zero flux; Au has its own thiol-driven cascade.
    rate_scale : float
        Per-metal scaling factor (1.0 for Pt; 1.0e4 for Pd; etc.).
    """

    metal: str
    rate_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.metal not in SUPPORTED_KINETIC_METALS:
            raise ValueError(
                f"DissociationKinetics: unknown metal {self.metal!r}"
            )
        if self.rate_scale < 0.0:
            raise ValueError(
                f"DissociationKinetics: rate_scale must be non-negative"
            )

    def initial_state(self) -> Tuple[float, float, float, float, float, float]:
        """Return the initial [PtCl2, PtClOH2, Pt(OH2)2, PtOH, PtN7, PtOHG] state."""
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def derivatives(
        self, state: Tuple[float, float, float, float, float, float]
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute the 6-state ODE derivatives for the hydrolysis cascade."""
        pt_cl2, pt_cl_oh2, pt_oh2_2, pt_oh, pt_n7, pt_ohg = state
        k = HYDROLYSIS_CASCADE_RATE_CONSTANTS
        s = self.rate_scale
        d_pt_cl2 = -k["k1_aquation"] * pt_cl2 * s
        d_pt_cl_oh2 = (
            +k["k1_aquation"] * pt_cl2 * s
            - k["k2_second_loss"] * pt_cl_oh2 * s
        )
        d_pt_oh2_2 = (
            +k["k2_second_loss"] * pt_cl_oh2 * s
            - k["k_bind_DNA_N7"] * pt_oh2_2 * s * 1.0e-3  # [DNA] ~ 1 mM
        )
        # pH-dependent Pt-OH2 ⇌ Pt-OH (pKa1 = 6.5; at physiological pH ~7.4
        # the deprotonated Pt-OH fraction is ~ 0.86; we model the flux
        # back to Pt-OH2 implicitly by setting PtOH ≈ 0).
        d_pt_oh = 0.0
        d_pt_n7 = +k["k_bind_DNA_N7"] * pt_oh2_2 * s * 1.0e-3
        # PtOHG = Pt-OH + G (slow GSH trapping; treated as sink here).
        d_pt_ohg = 0.0
        return (d_pt_cl2, d_pt_cl_oh2, d_pt_oh2_2, d_pt_oh, d_pt_n7, d_pt_ohg)

    def integrate(
        self, t_end_s: float, dt_s: float
    ) -> Tuple[List[float], List[Tuple[float, float, float, float, float, float]]]:
        """Forward-Euler integrate the hydrolysis cascade."""
        if t_end_s <= 0.0 or dt_s <= 0.0:
            raise ValueError(
                f"DissociationKinetics.integrate: t_end_s={t_end_s}, dt_s={dt_s} "
                f"must both be > 0"
            )
        n_steps = max(1, int(math.ceil(t_end_s / dt_s)))
        actual_dt = t_end_s / n_steps
        state = self.initial_state()
        t_grid = [0.0]
        state_traj = [state]
        for _ in range(n_steps):
            deriv = self.derivatives(state)
            new_state = tuple(
                max(0.0, s + d * actual_dt) for s, d in zip(state, deriv)
            )
            total = sum(new_state)
            if total > 0.0:
                new_state = tuple(s / total for s in new_state)
            state = new_state
            t_grid.append(t_grid[-1] + actual_dt)
            state_traj.append(state)
        return t_grid, state_traj

    def t_half_aquation(self) -> float:
        """Approximate time for [PtCl2] to reach 0.5 (first aquation)."""
        k = HYDROLYSIS_CASCADE_RATE_CONSTANTS["k1_aquation"]
        denom = k * self.rate_scale
        if denom <= 0.0:
            return float("inf")
        return math.log(2.0) / denom


# ---------------------------------------------------------------------------
# CellularUptake — Ctr1 + OCT2 + passive ODE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellularUptake:
    """Cellular uptake ODE driver for Ctr1 + OCT2 + passive pathways.

    The ODE tracks [Pt_ext] (extracellular) and [Pt_int] (intracellular)
    under Michaelis-Menten transporter kinetics.  Efflux via ATP7A/B is
    parameterised by ``efflux_rate_s`` (relevant for A2780cis resistance
    per ``TODO/INDEX.md`` §12.3).

    Attributes
    ----------
    metal : str
        One of :data:`SUPPORTED_KINETIC_METALS`.
    ctr1_km_m : float
        Michaelis constant for Ctr1 (default 5.0e-6 M; ORNL SOP-3).
    ctr1_vmax : float
        Maximal Ctr1 rate (fmol/cell/hr).
    oct2_km_m : float
        Michaelis constant for OCT2 (default 50.0e-6 M; ORNL SOP-3).
    oct2_vmax : float
        Maximal OCT2 rate (fmol/cell/hr).
    passive_rate_s : float
        Passive diffusion rate constant (s⁻¹).
    efflux_rate_s : float
        ATP7A/B efflux rate constant (s⁻¹); 0.0 for sensitive cell lines,
        > 0 for resistant cell lines (e.g. A2780cis).
    dna_sink_rate_s : float
        DNA-binding sink rate (s⁻¹); irreversible DNA capture.
    """

    metal: str
    ctr1_km_m: float = 5.0e-6
    ctr1_vmax: float = 2.0
    oct2_km_m: float = 50.0e-6
    oct2_vmax: float = 1.0
    passive_rate_s: float = 1.0e-5
    efflux_rate_s: float = 0.0
    dna_sink_rate_s: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.metal not in SUPPORTED_KINETIC_METALS:
            raise ValueError(
                f"CellularUptake: unknown metal {self.metal!r}"
            )

    def uptake_flux_fmol_per_cell_hr(self) -> float:
        """Return the dominant-transporter uptake flux (fmol/cell/hr)."""
        return UPTAKE_FLUX_FMOL_CELL_HR.get(self.metal, 0.05)

    def integrate(
        self, pt_ext_init_m: float, t_end_s: float, dt_s: float
    ) -> Tuple[List[float], Tuple[List[float], List[float]]]:
        """Forward-Euler integrate the [Pt_ext, Pt_int] 2-state system.

        Parameters
        ----------
        pt_ext_init_m : float
            Initial extracellular Pt concentration (M).
        t_end_s : float
            End time (s).
        dt_s : float
            Timestep (s).

        Returns
        -------
        t_grid : list of float
            Time grid (s).
        pt_ext_traj, pt_int_traj : tuple of (list of float, list of float)
            Per-timestep concentrations.
        """
        if pt_ext_init_m < 0.0:
            raise ValueError(f"CellularUptake.integrate: pt_ext_init_m < 0")
        if t_end_s <= 0.0 or dt_s <= 0.0:
            raise ValueError(
                f"CellularUptake.integrate: t_end_s={t_end_s}, dt_s={dt_s} "
                f"must both be > 0"
            )
        n_steps = max(1, int(math.ceil(t_end_s / dt_s)))
        actual_dt = t_end_s / n_steps
        pt_ext = pt_ext_init_m
        pt_int = 0.0
        t_grid = [0.0]
        pt_ext_traj = [pt_ext]
        pt_int_traj = [pt_int]
        for _ in range(n_steps):
            # Michaelis-Menten fluxes (M/s, converted from fmol/cell/hr via
            # 1 fmol/cell ≈ 1.0e-15 mol in 1.0e-12 L cell volume)
            v_ctr1 = (
                self.ctr1_vmax * pt_ext / (self.ctr1_km_m + pt_ext + 1.0e-12)
                * (1.0e-15 / 1.0e-12)  # fmol/cell/hr → M/s
                / 3600.0
            )
            v_oct2 = (
                self.oct2_vmax * pt_ext / (self.oct2_km_m + pt_ext + 1.0e-12)
                * (1.0e-15 / 1.0e-12)
                / 3600.0
            )
            v_passive = self.passive_rate_s * pt_ext
            uptake_total = v_ctr1 + v_oct2 + v_passive
            efflux = self.efflux_rate_s * pt_int
            dna_sink = self.dna_sink_rate_s * pt_int
            d_pt_ext = -uptake_total
            d_pt_int = +uptake_total - efflux - dna_sink
            pt_ext = max(0.0, pt_ext + d_pt_ext * actual_dt)
            pt_int = max(0.0, pt_int + d_pt_int * actual_dt)
            t_grid.append(t_grid[-1] + actual_dt)
            pt_ext_traj.append(pt_ext)
            pt_int_traj.append(pt_int)
        return t_grid, (pt_ext_traj, pt_int_traj)


# ---------------------------------------------------------------------------
# RedoxCycling — Pt(II)↔Pt(IV) + Ru(II)↔Ru(III) ODE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RedoxCycling:
    """Redox cycling ODE driver for Pt(II)↔Pt(IV) + Ru(II)↔Ru(III).

    Tracks the 2-state [M_reduced, M_oxidized] system under Nernst-
    driven first-order kinetics.  The reduction potential
    ``E0_v_vs_nhe`` is from :data:`REDOX_POTENTIALS_V`.

    Attributes
    ----------
    metal : str
        One of :data:`SUPPORTED_KINETIC_METALS`.
    k_oxidation_s : float
        Oxidation rate constant (s⁻¹, normalised to [oxidant]=1 M).
    k_reduction_s : float
        Reduction rate constant (s⁻¹, normalised to [reductant]=1 M).
    """

    metal: str
    k_oxidation_s: float = 1.0e-1
    k_reduction_s: float = 1.0e2

    def __post_init__(self) -> None:
        if self.metal not in SUPPORTED_KINETIC_METALS:
            raise ValueError(
                f"RedoxCycling: unknown metal {self.metal!r}"
            )
        if self.k_oxidation_s < 0.0 or self.k_reduction_s < 0.0:
            raise ValueError(
                f"RedoxCycling: rate constants must be non-negative"
            )

    def e0_v_vs_nhe(self) -> float:
        """Return the redox potential (V vs NHE) for this metal."""
        return REDOX_POTENTIALS_V.get(self.metal, 0.0)

    def initial_state(self) -> Tuple[float, float]:
        """Return the initial [M_reduced, M_oxidized] state."""
        return (1.0, 0.0)

    def derivatives(
        self, state: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Compute the 2-state redox ODE derivatives."""
        m_red, m_ox = state
        d_m_red = -self.k_oxidation_s * m_red + self.k_reduction_s * m_ox
        d_m_ox = +self.k_oxidation_s * m_red - self.k_reduction_s * m_ox
        return (d_m_red, d_m_ox)

    def integrate(
        self, t_end_s: float, dt_s: float
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Forward-Euler integrate the 2-state redox system."""
        if t_end_s <= 0.0 or dt_s <= 0.0:
            raise ValueError(
                f"RedoxCycling.integrate: t_end_s={t_end_s}, dt_s={dt_s} "
                f"must both be > 0"
            )
        n_steps = max(1, int(math.ceil(t_end_s / dt_s)))
        actual_dt = t_end_s / n_steps
        state = self.initial_state()
        t_grid = [0.0]
        state_traj = [state]
        for _ in range(n_steps):
            deriv = self.derivatives(state)
            new_state = tuple(
                max(0.0, s + d * actual_dt) for s, d in zip(state, deriv)
            )
            total = sum(new_state)
            if total > 0.0:
                new_state = tuple(s / total for s in new_state)
            state = new_state
            t_grid.append(t_grid[-1] + actual_dt)
            state_traj.append(state)
        return t_grid, state_traj


# ---------------------------------------------------------------------------
# MultiTaskLoss — α·R²_static + β·MSE_rate + γ·CE_t1/2 wire-up
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MultiTaskLoss:
    """Multi-task loss weights for the 3-task kinetics objective.

    Anchors the S10 SKILL.md §2.4 wire-up:
        loss = α * mse(static_pIC50_pred, static_pIC50_true)
             + β * mse(rate_constant_pred, rate_constant_literature)
             + γ * cross_entropy(t_half_class_pred, t_half_class_true)

    Default weights (α=0.5, β=0.3, γ=0.2) balance:
      * Static affinity (L5/L7 inheritance) — preserves the existing
        pIC50 R² baseline.
      * Kinetic rate constant MSE (new).
      * Half-life classification (new) — 3-class softmax over
        fast (<10 min) / medium (10 min-2 hr) / slow (>2 hr).

    Attributes
    ----------
    alpha_static : float
        Static R² weight.
    beta_rate : float
        Kinetic rate-constant MSE weight.
    gamma_half_life : float
        Half-life classification CE weight.
    """

    alpha_static: float = MULTI_TASK_LOSS_WEIGHTS["alpha_static"]
    beta_rate: float = MULTI_TASK_LOSS_WEIGHTS["beta_rate"]
    gamma_half_life: float = MULTI_TASK_LOSS_WEIGHTS["gamma_half_life"]

    def __post_init__(self) -> None:
        if self.alpha_static < 0.0 or self.beta_rate < 0.0 or self.gamma_half_life < 0.0:
            raise ValueError(
                f"MultiTaskLoss: weights must be non-negative; "
                f"got α={self.alpha_static}, β={self.beta_rate}, γ={self.gamma_half_life}"
            )

    def classify_t_half(self, t_half_s: float) -> int:
        """Classify t1/2 into 3 buckets: 0=fast, 1=medium, 2=slow.

        Thresholds from :data:`T_HALF_CLASS_BOUNDS_S`:
          * fast: < 600 s (10 min)
          * medium: 600 - 7200 s (10 min - 2 hr)
          * slow: > 7200 s
        """
        if t_half_s < T_HALF_CLASS_BOUNDS_S[0]:
            return 0
        if t_half_s < T_HALF_CLASS_BOUNDS_S[1]:
            return 1
        return 2

    def loss(
        self,
        static_pic50_pred: float,
        static_pic50_true: float,
        rate_constant_pred: float,
        rate_constant_lit: float,
        t_half_class_pred: int,
        t_half_class_true: int,
    ) -> float:
        """Compute the 3-task loss given (pred, true) pairs.

        Returns a finite non-negative scalar; uses MSE for the static
        + rate terms and 0/1 cross-entropy for the classification term.
        """
        mse_static = (static_pic50_pred - static_pic50_true) ** 2
        mse_rate = (rate_constant_pred - rate_constant_lit) ** 2
        ce_t_half = 0.0 if t_half_class_pred == t_half_class_true else 1.0
        return (
            self.alpha_static * mse_static
            + self.beta_rate * mse_rate
            + self.gamma_half_life * ce_t_half
        )


# ---------------------------------------------------------------------------
# KineticsTrajectory — result dataclass aggregating the 4 ODEs + loss
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KineticsTrajectory:
    """Aggregated kinetics result for one (metal, target) pair.

    Attributes
    ----------
    metal : str
        The d-block metal center.
    kinetic_regime : str
        Regime classification from :data:`KINETIC_REGIME`.
    aquation_t_half_s : float
        Half-life for the first aquation step (s).
    redox_potential_v : float
        Redox potential (V vs NHE).
    uptake_fmol_per_cell_hr : float
        Dominant-transporter uptake flux (fmol/cell/hr).
    static_loss : float
        Static pIC50 loss term (MSE against PlatinAI 4-cell baseline).
    rate_constant_loss : float
        Rate-constant MSE against literature priors.
    t_half_class_loss : float
        t1/2 classification CE (0 or 1).
    multi_task_loss : float
        Weighted sum of the 3 loss terms.
    n_steps : int
        Number of integration steps used.
    wall_time_s : float
        Wall-clock time for the integration (s).
    seed : int
        Seed used for the synthetic mock.
    """

    metal: str
    kinetic_regime: str
    aquation_t_half_s: float
    redox_potential_v: float
    uptake_fmol_per_cell_hr: float
    static_loss: float
    rate_constant_loss: float
    t_half_class_loss: float
    multi_task_loss: float
    n_steps: int
    wall_time_s: float
    seed: int


# ---------------------------------------------------------------------------
# Per-metal rate-scale helper (used by all 3 ODE drivers)
# ---------------------------------------------------------------------------
def rate_scale_for_metal(metal: str) -> float:
    """Return the per-metal rate-scale factor for ODE drivers.

    Pt = 1.0 (anchor); Pd = 1e4 (10⁴× faster than Pt); Au = 1e-3
    (slower thiol exchange for our scaffold); inert metals (Ru/Rh/
    Ir/Os/Re) = 1e-6 (effectively zero flux).
    """
    if metal == "Pt":
        return 1.0
    if metal == "Pd":
        return 1.0e4
    if metal == "Au":
        return 1.0e-3
    # Inert regime: near-zero flux for ligand-exchange / hydrolysis
    return 1.0e-6


# ---------------------------------------------------------------------------
# synthetic_mock_kinetics — deterministic mock driver for the (Pt, target) pair
# ---------------------------------------------------------------------------
def synthetic_mock_kinetics(
    metal: str,
    target_id: str = "target_0",
    seed: int = 42,
    t_end_s: float = 3600.0,
    dt_s: float = 10.0,
    *,
    loss: Optional[MultiTaskLoss] = None,
    binding: Optional[BindingKinetics] = None,
    dissociation: Optional[DissociationKinetics] = None,
    uptake: Optional[CellularUptake] = None,
    redox: Optional[RedoxCycling] = None,
) -> KineticsTrajectory:
    """Run all 4 ODE drivers + the multi-task loss for one (metal, target).

    Parameters
    ----------
    metal : str
        One of :data:`SUPPORTED_KINETIC_METALS`.
    target_id : str
        Target scaffold identifier (mock — not used in computation, but
        threaded through for parity with :class:`TargetSpec`).
    seed : int
        Seed for reproducibility (mock — not used in the deterministic
        NumPy scaffold).
    t_end_s : float
        Integration end-time (s).
    dt_s : float
        Integration timestep (s).
    loss : MultiTaskLoss, optional
        Multi-task loss; default = :class:`MultiTaskLoss` defaults.
    binding : BindingKinetics, optional
        BindingKinetics instance; default = constructed from metal.
    dissociation : DissociationKinetics, optional
        DissociationKinetics instance; default = constructed from metal.
    uptake : CellularUptake, optional
        CellularUptake instance; default = constructed from metal.
    redox : RedoxCycling, optional
        RedoxCycling instance; default = constructed from metal.

    Returns
    -------
    KineticsTrajectory
        Aggregated result for the (metal, target) pair.
    """
    if metal not in SUPPORTED_KINETIC_METALS:
        raise ValueError(
            f"synthetic_mock_kinetics: unknown metal {metal!r}; "
            f"supported = {SUPPORTED_KINETIC_METALS}"
        )
    if not isinstance(target_id, str) or not target_id:
        raise ValueError(
            f"synthetic_mock_kinetics: target_id must be a non-empty string"
        )
    if t_end_s <= 0.0 or dt_s <= 0.0:
        raise ValueError(
            f"synthetic_mock_kinetics: t_end_s={t_end_s}, dt_s={dt_s} "
            f"must both be > 0"
        )

    s = rate_scale_for_metal(metal)
    binding = binding or BindingKinetics(metal=metal, rate_scale=s)
    dissociation = dissociation or DissociationKinetics(metal=metal, rate_scale=s)
    uptake = uptake or CellularUptake(metal=metal)
    redox = redox or RedoxCycling(metal=metal)
    loss = loss or MultiTaskLoss()

    t0 = time.time()
    # Run all 4 ODE drivers
    binding.integrate(t_end_s=t_end_s, dt_s=dt_s)
    dissociation.integrate(t_end_s=t_end_s, dt_s=dt_s)
    uptake.integrate(pt_ext_init_m=1.0e-5, t_end_s=t_end_s, dt_s=dt_s)
    redox.integrate(t_end_s=t_end_s, dt_s=dt_s)
    wall_time_s = time.time() - t0

    # Compute the 3 loss terms
    t_half_pred = binding.t_half_pt_oh2()
    t_half_lit = AQUATION_HALF_LIVES_S[metal]
    rate_pred = LIGAND_EXCHANGE_RATE_CONSTANTS["Cl_to_H2O"] * s
    rate_lit = LIGAND_EXCHANGE_RATE_CONSTANTS["Cl_to_H2O"]
    static_pred = 0.0  # mock
    static_true = STATIC_R2_PLATINAI_4CELL[metal]
    t_half_class_pred = loss.classify_t_half(t_half_pred)
    t_half_class_true = loss.classify_t_half(t_half_lit)
    static_loss = (static_pred - static_true) ** 2
    rate_loss = (rate_pred - rate_lit) ** 2
    t_half_class_loss = 0.0 if t_half_class_pred == t_half_class_true else 1.0
    multi_task_loss = (
        loss.alpha_static * static_loss
        + loss.beta_rate * rate_loss
        + loss.gamma_half_life * t_half_class_loss
    )

    n_steps = max(1, int(math.ceil(t_end_s / dt_s)))

    return KineticsTrajectory(
        metal=metal,
        kinetic_regime=KINETIC_REGIME[metal],
        aquation_t_half_s=t_half_lit,
        redox_potential_v=REDOX_POTENTIALS_V[metal],
        uptake_fmol_per_cell_hr=UPTAKE_FLUX_FMOL_CELL_HR[metal],
        static_loss=static_loss,
        rate_constant_loss=rate_loss,
        t_half_class_loss=t_half_class_loss,
        multi_task_loss=multi_task_loss,
        n_steps=n_steps,
        wall_time_s=wall_time_s,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# smoke_neural_ode_kinetics — end-to-end smoke runner
# ---------------------------------------------------------------------------
def smoke_neural_ode_kinetics() -> List[KineticsTrajectory]:
    """End-to-end smoke: run synthetic_mock_kinetics for 3 metals × 1 target.

    Returns the list of :class:`KineticsTrajectory` records (3 metals
    by default: Pt + Pd + Au, spanning the 3 non-inert regimes).
    The smoke is **deterministic** — same seed gives same result.
    """
    metals = ("Pt", "Pd", "Au")
    target_id = "smoke_target"
    trajectories = [
        synthetic_mock_kinetics(metal=m, target_id=target_id, seed=42)
        for m in metals
    ]
    return trajectories


# ---------------------------------------------------------------------------
# CLI smoke entry point
# ---------------------------------------------------------------------------
def _main() -> int:
    """CLI smoke entry point. Prints headline metrics. Exit 0 on OK."""
    print("Wf-78 R22-P2 Phase 1 smoke (L14 neural ODE kinetics)")
    print(f"  Supported metals: {SUPPORTED_KINETIC_METALS}")
    print(f"  Kinetic regimes: {KINETIC_REGIME}")
    print(f"  Multi-task loss weights: {MULTI_TASK_LOSS_WEIGHTS}")
    print()
    trajectories = smoke_neural_ode_kinetics()
    print(f"  Smoke OK: {len(trajectories)} trajectories")
    for traj in trajectories:
        print(
            f"    [{traj.metal}] regime={traj.kinetic_regime} "
            f"t1/2={traj.aquation_t_half_s:.1f}s "
            f"E°={traj.redox_potential_v:.2f}V "
            f"uptake={traj.uptake_fmol_per_cell_hr:.2f} fmol/c/hr "
            f"multi_task_loss={traj.multi_task_loss:.4f}"
        )
    print()
    print("NOTE: synthetic mock only; no GPU runs; no real kinetics MEASURED.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
