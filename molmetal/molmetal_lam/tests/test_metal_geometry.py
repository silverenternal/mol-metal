"""Tests for the TODO-09 square-planar Pt(II) + generalised metal-prior.

Covers:
  1. Square-planar Pt(II) ideal 90 deg geometry (cisplatin Pt(NH3)2Cl2).
  2. Octahedral Ru(CO)6 — 6 donors, 90 / 180 deg angles.
  3. prior_loss is zero for a perfect geometry.
  4. prior_loss is positive for a deliberately distorted geometry.
  5. Smoke test that ``EGNNVelocityField.apply_metal_geometry_step`` runs
     end-to-end and respects the K-every gate (zero off-step).
"""

from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers — build small Pt(II) / Ru(II) complexes with explicit dative edges
# ---------------------------------------------------------------------------
def _square_planar_pt_complex(device: str = "cpu"):
    """Pt(II) with 4 donors at ideal 90 deg — cisplatin analogue.

    Coordinates:
        Pt at origin (0, 0, 0).
        Four donors on the xy-plane at distance 2.0 Å, mutually 90 deg:
            +x: (2, 0, 0)   NH3 donor (N, Z=7)
            -x: (-2, 0, 0)  NH3 donor (N, Z=7)
            +y: (0, 2, 0)   Cl donor  (Cl, Z=17)
            -y: (0, -2, 0)  Cl donor  (Cl, Z=17)
    """
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],   # Pt (Z=78)
            [2.0, 0.0, 0.0],   # N  +
            [-2.0, 0.0, 0.0],  # N  -
            [0.0, 2.0, 0.0],   # Cl +
            [0.0, -2.0, 0.0],  # Cl -
        ],
        device=device,
        dtype=torch.float32,
    )
    atom_types = torch.tensor(
        [78, 7, 7, 17, 17], device=device, dtype=torch.long
    )
    # Edge index for a fully-connected (no self-loop) graph over 5 nodes.
    n = coords.size(0)
    src = []
    dst = []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], device=device, dtype=torch.long)
    # EDGE_TYPE_DATIVE = 2 on donor -> Pt edges, single (0) elsewhere.
    edge_types = torch.zeros(edge_index.size(1), device=device, dtype=torch.long)
    # Pt is index 0; donors are 1..4.  Mark donor -> Pt edges as dative.
    pt_idx = 0
    for k in range(edge_index.size(1)):
        if int(edge_index[1, k].item()) == pt_idx and int(edge_index[0, k].item()) != pt_idx:
            edge_types[k] = 2
    return coords, atom_types, edge_index, edge_types


def _octahedral_RuCO6(device: str = "cpu"):
    """Ru(0) at origin, 6 CO donors on the +-axes.

    Donors at distance 2.0 Å along +/-x, +/-y, +/-z.  Ideal angles are
    90 deg (adjacent axes) and 180 deg (opposite axes).
    """
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],   # Ru (Z=44)
            [2.0, 0.0, 0.0],   # C +x
            [-2.0, 0.0, 0.0],  # C -x
            [0.0, 2.0, 0.0],   # C +y
            [0.0, -2.0, 0.0],  # C -y
            [0.0, 0.0, 2.0],   # C +z
            [0.0, 0.0, -2.0],  # C -z
        ],
        device=device,
        dtype=torch.float32,
    )
    atom_types = torch.tensor(
        [44, 6, 6, 6, 6, 6, 6], device=device, dtype=torch.long
    )
    n = coords.size(0)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], device=device, dtype=torch.long)
    edge_types = torch.zeros(edge_index.size(1), device=device, dtype=torch.long)
    ru_idx = 0
    for k in range(edge_index.size(1)):
        if int(edge_index[1, k].item()) == ru_idx and int(edge_index[0, k].item()) != ru_idx:
            edge_types[k] = 2
    return coords, atom_types, edge_index, edge_types


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_square_planar_cisplatin():
    """Pt(II) cis-platin analogue — Pt(NH3)2Cl2 donors should be 90 deg.

    Verifies the canonical Pt(II) d8 square-planar geometry: 4 donors at
    90 deg (the prior should report zero or near-zero penalty).
    """
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior,
        Geometry,
    )
    coords, atom_types, edge_index, edge_types = _square_planar_pt_complex()
    prior = MetalGeometryPrior(weight=1.0)
    loss = prior.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    # All four L-X-L angles are exactly 90 deg → deviation < 1 deg (radians).
    assert loss.item() < math.radians(1.0), (
        f"Expected ~0 rad deviation for ideal square-planar Pt(II); got "
        f"{loss.item():.4f} rad ({math.degrees(loss.item()):.2f} deg)"
    )


def test_octahedral_Ru():
    """Ru(CO)6 donors should be 90 deg (adjacent) and 180 deg (trans)."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior,
        Geometry,
    )
    coords, atom_types, edge_index, edge_types = _octahedral_RuCO6()
    prior = MetalGeometryPrior(weight=1.0)
    loss = prior.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.OCTAHEDRAL,
    )
    # Adjacent 90 deg, trans 180 deg → all 15 pairwise angles are in
    # the ideal set; min deviation should be effectively zero.
    assert loss.item() < math.radians(1.0), (
        f"Expected ~0 rad deviation for ideal octahedral Ru(CO)6; got "
        f"{loss.item():.4f} rad ({math.degrees(loss.item()):.2f} deg)"
    )


def test_prior_loss_zero_for_perfect_geometry():
    """A perfectly ideal square-planar geometry must yield loss ~= 0."""
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior,
        Geometry,
    )
    coords, atom_types, edge_index, edge_types = _square_planar_pt_complex()
    prior = MetalGeometryPrior(weight=1.0)
    loss = prior.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    # Tolerance 1e-3 rad ~ 0.057 deg.
    assert loss.item() < 1e-3, f"perfect-geometry loss not zero: {loss.item()}"


def test_prior_loss_positive_for_distorted():
    """A deliberately distorted geometry (45 deg between donors) must
    produce a positive penalty well above the perfect-geometry noise floor.
    """
    from molmetal.molmetal_lam.priors.metal_geometry import (
        MetalGeometryPrior,
        Geometry,
    )
    # Build a distorted Pt: 4 donors at 45 deg spacing on the xy-plane
    # (ideal for square-planar is 90 deg → off by 45 deg per pair).
    device = "cpu"
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [math.cos(math.radians(45)) * 2.0, math.sin(math.radians(45)) * 2.0, 0.0],
            [math.cos(math.radians(90)) * 2.0, math.sin(math.radians(90)) * 2.0, 0.0],
            [math.cos(math.radians(135)) * 2.0, math.sin(math.radians(135)) * 2.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    atom_types = torch.tensor(
        [78, 7, 7, 17, 17], device=device, dtype=torch.long
    )
    # All edges are covalent single; the prior uses EDGE_TYPE_DATIVE
    # (== 2) so we need an edge_types tensor with at least one dative
    # edge.  Wire donor → Pt edges explicitly.
    pt_idx = 0
    n = coords.size(0)
    src, dst, et = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            if j == pt_idx and i != pt_idx:
                et.append(2)  # EDGE_TYPE_DATIVE
            else:
                et.append(0)  # EDGE_TYPE_SINGLE
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_types = torch.tensor(et, device=device, dtype=torch.long)

    prior = MetalGeometryPrior(weight=1.0)
    loss = prior.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types, edge_index=edge_index,
        metal_idx=0, geometry=Geometry.SQUARE_PLANAR,
    )
    # Expect a substantial penalty (45 deg between some pairs → ~0.78 rad).
    assert loss.item() > math.radians(10.0), (
        f"Expected substantial penalty for distorted geometry; got "
        f"{loss.item():.4f} rad ({math.degrees(loss.item()):.2f} deg)"
    )


def test_egnn_velocity_applies_metal_prior():
    """Smoke test: EGNNVelocityField.apply_metal_geometry_step returns
    the prior loss on K-aligned steps and zero on off-steps, and is a
    differentiable tensor when on-step.
    """
    from molmetal.adapters.flow_matching_lipman import EGNNVelocityField

    vf = EGNNVelocityField(hidden_dim=32, n_layers=1, max_atomic_number=100)
    coords, atom_types, edge_index, edge_types = _square_planar_pt_complex()
    # Batched view expected by the velocity field (B, N, 3) / (B, E).
    coords_b = coords.unsqueeze(0)
    atom_types_b = atom_types.unsqueeze(0)
    edge_types_b = edge_types.unsqueeze(0)

    # Off-step (step % 10 == 3): returns zero.
    off = vf.apply_metal_geometry_step(
        positions=coords_b,
        atom_types=atom_types_b,
        edge_types=edge_types_b,
        edge_index=edge_index,
        step=3, k_every=10, weight=0.1,
    )
    assert off.dim() == 0
    assert off.item() == 0.0, f"off-step should be 0.0, got {off.item()}"

    # On-step (step % 10 == 0): returns the (weighted) prior.
    on = vf.apply_metal_geometry_step(
        positions=coords_b,
        atom_types=atom_types_b,
        edge_types=edge_types_b,
        edge_index=edge_index,
        step=10, k_every=10, weight=0.1,
    )
    assert on.dim() == 0
    # Perfect 90 deg geometry → penalty should be ~0; multiplied by
    # weight=0.1 → on <= 1e-4.
    assert on.item() < 1e-4, (
        f"On-step should be near-zero for perfect geometry; got {on.item()}"
    )

    # Distorted geometry on-step → penalty > 0.
    coords_dist = coords.clone()
    coords_dist[1, 0] = 1.0   # compress donor 1 towards Pt
    coords_dist[2, 0] = -1.0  # compress donor 2 towards Pt
    coords_dist_b = coords_dist.unsqueeze(0)
    on_dist = vf.apply_metal_geometry_step(
        positions=coords_dist_b,
        atom_types=atom_types_b,
        edge_types=edge_types_b,
        edge_index=edge_index,
        step=20, k_every=10, weight=0.1,
    )
    assert on_dist.item() > 0.0, (
        f"Distorted on-step should produce positive loss; got {on_dist.item()}"
    )


def test_egnn_edge_type_dative_enum_and_set():
    """Verify EDGE_TYPE_DATIVE=2 enum + EGNN.set_edge_types API round-trip."""
    from molmetal.adapters.egnn_rocm import (
        EDGE_TYPE_SINGLE,
        EDGE_TYPE_DOUBLE,
        EDGE_TYPE_DATIVE,
        EGNN,
    )
    # Enum values
    assert EDGE_TYPE_SINGLE == 0
    assert EDGE_TYPE_DOUBLE == 1
    assert EDGE_TYPE_DATIVE == 2

    # Set + clear round-trip
    g = EGNN(in_node_dim=8, hidden_dim=8, n_layers=1)
    et = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    g.set_edge_types(et)
    assert torch.equal(g._edge_types, et)
    g.set_edge_types(None)
    assert not hasattr(g, "_edge_types")

    # Reject unknown codes
    bad = torch.tensor([0, 5], dtype=torch.long)
    with pytest.raises(ValueError):
        g.set_edge_types(bad)

    # Reject non-tensor
    with pytest.raises(TypeError):
        g.set_edge_types([0, 1, 2])


def test_bondi_vdw_radii_present():
    """Bondi (1964) vdW radii should be defined for the metals the prior
    cares about and fall back sensibly for unknowns.
    """
    from molmetal.molmetal_lam.priors.metal_geometry import (
        bondi_van_der_waals_radius,
        BONDI_VAN_DER_WAALS_RADII,
    )
    # Canonical entries
    assert bondi_van_der_waals_radius(78) == 1.75   # Pt
    assert bondi_van_der_waals_radius(46) == 1.63   # Pd
    assert bondi_van_der_waals_radius(29) == 1.40   # Cu
    assert bondi_van_der_waals_radius(8)  == 1.52   # O
    # Fallback for unknown element (e.g. Tc=43 not in table) — should
    # return the 1.70 Å default.
    assert bondi_van_der_waals_radius(43) == 1.70
    # And the table should have at least 10 entries (sanity check).
    assert len(BONDI_VAN_DER_WAALS_RADII) >= 10