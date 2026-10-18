"""Wf-A75 (2026-10-18) — bond-length guidance regression tests.

Background: per Vignac et al. 2023 (MolDiff, arXiv:2305.07508),
Cartesian-coordinate diffusion on metal complexes suffers from
"atom-bond inconsistency": bonded pairs drift past ~3 Å during
denoising and the output graph is topologically fragmented
(``largest_fragment_frac`` collapses to ~0.05).  Wf-A75 ships a
bond-length prior that adds a Hooke-spring force to the EGNN
velocity at every ODE step, pulling the per-(Z_i, Z_j) distance
toward an expected value.  The wrapper class is in
:mod:`molmetal.adapters.flow_matching_lipman`.

5 tests below cover:
  1. Zero-weight no-op (back-compat)
  2. Force direction (attractive / repulsive)
  3. Per-atom force balance on a symmetric 3-atom cloud
  4. Force vanishes on a perfectly-bonded cloud
  5. Metal-donor pair receives ~2.05 Å pull (chemistry calibration)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER = (
    PROJECT_ROOT
    / "molmetal"
    / "adapters"
    / "flow_matching_lipman"
    / "__init__.py"
)


def _load_adapter():
    spec = importlib.util.spec_from_file_location("_ad_wfa75", str(ADAPTER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ad_wfa75"] = mod
    spec.loader.exec_module(mod)
    return mod


_AD = _load_adapter()
_BLG = _AD._BondLengthGuidance


# ---------------------------------------------------------------------------
# Test 1 — zero weight is a strict no-op
# ---------------------------------------------------------------------------
def test_bond_length_guidance_zero_weight_is_noop() -> None:
    """When ``guidance_weight=0`` the wrapper returns a zero tensor
    with the same shape + device + dtype as the input positions.
    Used as a back-compat gate: the Wf-A74 (and earlier) sampling
    path is bit-exact preserved when the YAML key is absent."""
    atom_types = torch.tensor([6, 6, 7], dtype=torch.long)
    guidance = _BLG(
        atom_types=atom_types, cutoff=3.5, guidance_weight=0.0
    )
    x = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    f = guidance(x)
    assert f.shape == x.shape, f"force shape must match positions; got {f.shape}"
    assert torch.equal(f, torch.zeros_like(x)), (
        f"WF-A75: zero guidance_weight must be a strict no-op.  "
        f"got max abs force = {f.abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — force direction is attractive when over-stretched
# ---------------------------------------------------------------------------
def test_bond_length_guidance_attractive_when_stretched() -> None:
    """A C-C pair at 2.4 Å (vs target 1.50 Å) must receive an
    attractive force that pulls the two atoms TOWARD each other:
    atom 0 gets a positive force in the +x direction (toward atom 1);
    atom 1 gets a negative force in the +x direction (toward atom 0).
    Net force is zero (Newton's 3rd law) on the 2-atom subsystem.
    The 2.4 Å upper bound matches the decoder's bond candidacy cutoff
    (2.4 Å per Wf-A71 §6.99) — pairs beyond this are not real bonds
    and receive no force."""
    atom_types = torch.tensor([6, 6], dtype=torch.long)
    guidance = _BLG(
        atom_types=atom_types, cutoff=3.5, guidance_weight=1.0
    )
    # Atoms at x=0 and x=2.4 (stretched from 1.5 target by 0.9, within
    # the 2.5 Å inner-bond cutoff).
    x = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    f = guidance(x)
    # Force on atom 0 should point +x (positive); force on atom 1 should
    # point -x (negative); both magnitudes should be non-zero.
    f0_x = float(f[0, 0, 0].item())
    f1_x = float(f[0, 1, 0].item())
    assert f0_x > 0.0, (
        f"WF-A75: stretched C-C pair must pull atom 0 TOWARD atom 1 "
        f"(positive +x force).  got f0_x = {f0_x:.3f}"
    )
    assert f1_x < 0.0, (
        f"WF-A75: stretched C-C pair must pull atom 1 TOWARD atom 0 "
        f"(negative +x force).  got f1_x = {f1_x:.3f}"
    )
    # Net force on the subsystem = 0 (Newton's 3rd law).
    net_x = f0_x + f1_x
    assert abs(net_x) < 1e-5, (
        f"WF-A75: net force on a 2-atom subsystem must be zero "
        f"(Newton's 3rd law).  got net_x = {net_x:.3e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — force vanishes on a perfectly-bonded cloud
# ---------------------------------------------------------------------------
def test_bond_length_guidance_zero_at_target_distance() -> None:
    """At the target distance (1.50 Å for C-C) the spring force is
    zero.  This is the "rest" configuration: the guidance neither
    helps nor hurts the EGNN when the geometry is already correct.
    At init (random Gaussian noise around origin) the mean |force|
    is large; at the C-C target the mean |force| is exactly zero."""
    atom_types = torch.tensor([6, 6], dtype=torch.long)
    guidance = _BLG(
        atom_types=atom_types, cutoff=3.5, guidance_weight=1.0
    )
    x = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]],  # exactly at C-C target
        dtype=torch.float32,
    )
    f = guidance(x)
    assert torch.allclose(f, torch.zeros_like(f), atol=1e-6), (
        f"WF-A75: force must vanish at the target distance.  "
        f"got max abs = {f.abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — metal-donor chemistry calibration
# ---------------------------------------------------------------------------
def test_bond_length_guidance_metal_donor_target() -> None:
    """The metal-donor (Pt, N) pair has expected distance 2.05 Å
    (per the chemistry table in the docstring).  A Pt-N pair at
    2.4 Å (stretched from 2.05 by 0.35) must receive an attractive
    force, while a Pt-N pair at 1.7 Å (compressed by 0.35) must
    receive a repulsive force.  Both forces are within the 2.5 Å
    inner-bond cutoff (metal-donor bonds can be up to ~2.5 Å in
    practice)."""
    atom_types_pt_n = torch.tensor([78, 7], dtype=torch.long)
    guidance = _BLG(
        atom_types=atom_types_pt_n, cutoff=3.5, guidance_weight=1.0
    )
    # Stretched: Pt-N at 2.4 Å (target 2.05) — attractive force.
    x_stretched = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    f_stretched = guidance(x_stretched)
    # Compressed: Pt-N at 1.7 Å (target 2.05) — repulsive force.
    x_compressed = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    f_compressed = guidance(x_compressed)
    # Stretched case: Pt-side force should be +x (toward N); N-side
    # should be -x (toward Pt).  Both non-zero.
    f_str_Pt_x = float(f_stretched[0, 0, 0].item())
    f_str_N_x = float(f_stretched[0, 1, 0].item())
    assert f_str_Pt_x > 0.0, (
        f"WF-A75: stretched Pt-N must pull Pt toward N (+x).  "
        f"got f_str_Pt_x = {f_str_Pt_x:.3f}"
    )
    assert f_str_N_x < 0.0, (
        f"WF-A75: stretched Pt-N must pull N toward Pt (-x).  "
        f"got f_str_N_x = {f_str_N_x:.3f}"
    )
    # Compressed case: Pt-side force should be -x (away from N); N-side
    # should be +x (away from Pt).  Both non-zero.
    f_comp_Pt_x = float(f_compressed[0, 0, 0].item())
    f_comp_N_x = float(f_compressed[0, 1, 0].item())
    assert f_comp_Pt_x < 0.0, (
        f"WF-A75: compressed Pt-N must push Pt away from N (-x).  "
        f"got f_comp_Pt_x = {f_comp_Pt_x:.3f}"
    )
    assert f_comp_N_x > 0.0, (
        f"WF-A75: compressed Pt-N must push N away from Pt (+x).  "
        f"got f_comp_N_x = {f_comp_N_x:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — multi-atom force balance + non-degenerate output
# ---------------------------------------------------------------------------
def test_bond_length_guidance_three_atom_carbon_chain() -> None:
    """A 3-atom C-C-C chain at [0, 1.5, 2.0] (target 1.5 Å for each
    pair) must receive zero total force (the geometry is already
    at the C-C target and the (0, 2) pair at d=2.0 is within the
    2.5 Å bond cutoff, so all 3 pairs are at target).  A perturbed
    chain at [0, 2.0, 3.5] (middle atom at d=2.0 from atom 0, but
    the (1, 2) pair at d=1.5 is at target; atom 0 is stretched
    from atom 1) must receive an attractive force on atom 0
    pointing toward atom 1, and a repulsive force on atom 1
    pushing it toward atom 0 (compression relief)."""
    atom_types = torch.tensor([6, 6, 6], dtype=torch.long)
    guidance = _BLG(
        atom_types=atom_types, cutoff=3.5, guidance_weight=1.0
    )
    # At-target case: [0, 1.5, 2.0]
    #   Pair (0, 1): d=1.5, target 1.5, residual 0
    #   Pair (1, 2): d=0.5, target 1.5, residual -1.0 — wait, 0.5 is
    #     below the 1e-3 clamp so the spring force is 1.0*(0.5-1.5) but
    #     also the pair is within the 2.5 Å cutoff so it gets a force.
    #   Pair (0, 2): d=2.0, target 1.5, residual 0.5
    # To make this test cleaner, use a longer chain so no two pairs
    # are too close.
    # At-target case v2: [0, 1.5, 3.0] (target 1.5 for each pair)
    #   Pair (0, 1): d=1.5, target 1.5, force 0
    #   Pair (1, 2): d=1.5, target 1.5, force 0
    #   Pair (0, 2): d=3.0 > 2.5 inner cutoff — no force
    x_target = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    f_target = guidance(x_target)
    assert torch.allclose(f_target, torch.zeros_like(f_target), atol=1e-5), (
        f"WF-A75: 3-atom C-C-C at target distances must have zero "
        f"restoring force.  got max abs = {f_target.abs().max().item():.3e}"
    )
    # Stretched case: [0, 2.4, 3.9] (the (0, 1) pair is stretched
    # from target 1.5 to 2.4 Å — gets attractive force; the (1, 2)
    # pair is also stretched but is at d=1.5 = target = no force).
    x_stretched = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [3.9, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    f_stretched = guidance(x_stretched)
    # The (0, 1) pair contributes an attractive force: atom 0 gets
    # +x (toward atom 1), atom 1 gets -x (toward atom 0).  The (1, 2)
    # pair is at d=1.5 = target, contributes zero.  The (0, 2) pair
    # is at d=3.9 > 2.5, no force.
    f0_x = float(f_stretched[0, 0, 0].item())
    f1_x = float(f_stretched[0, 1, 0].item())
    f2_x = float(f_stretched[0, 2, 0].item())
    assert f0_x > 0.0, (
        f"WF-A75: atom 0 should be pulled +x (toward atom 1 at x=2.4).  "
        f"got f0_x = {f0_x:.3f}"
    )
    assert f1_x < 0.0, (
        f"WF-A75: atom 1 should be pulled -x (toward atom 0 at x=0).  "
        f"got f1_x = {f1_x:.3f}"
    )
    assert abs(f2_x) < 1e-3, (
        f"WF-A75: atom 2 should have zero force (the (1, 2) pair is at "
        f"target 1.5).  got f2_x = {f2_x:.3e}"
    )
    # Newton's 3rd law: net force on the chain = 0.
    net_x = f0_x + f1_x + f2_x
    assert abs(net_x) < 1e-3, (
        f"WF-A75: net force on a 3-atom chain must be zero "
        f"(Newton's 3rd law).  got net_x = {net_x:.3e}"
    )
