"""Known-Pt regression tests for the MetalGeometryPrior.

Background
----------
TODO-30 Rank-3 (P1.4 — Known-Pt regression tests).  This file is pure
INSURANCE — every test should pass on the existing prior as shipped in
``molmetal/molmetal_lam/priors/metal_geometry.py``.  The intent is to
lock down the geometry-penalty behaviour on canonical Pt drug
coordination spheres:

  1. Cisplatin     -> Pt_II square-planar (CN=4: 2 NH3 + 2 Cl)
  2. Carboplatin   -> Pt_II square-planar (CN=4: 2 NH3 + 2 O-donor)
  3. Satraplatin   -> Pt_IV octahedral   (CN=6: 2 NH3 + 2 Cl + 2 Ac)

For each canonical geometry we assert the prior penalty is
near-zero (the reference standard) so that a future edit to the
ideal-angle set or the dative-edge inference would fail loudly.

Honest framing
--------------
* The penalty uses ideal 90 deg for cis pairs and 180 deg for trans
  pairs in square-planar Pt(II); 90 deg for adjacent and 180 deg for
  trans in octahedral Pt(IV).  A perfect geometry yields penalty = 0.
* Tolerance is generous (~1 deg) because the penalty's arccos clamp
  introduces a small numerical floor (~1e-3 rad).
* All tests are CPU-only — we never import or touch the CFM velocity
  field; the prior is applied in isolation on a synthetic coord tensor.
"""

from __future__ import annotations

import math

import pytest
import torch

from molmetal.molmetal_lam.priors.metal_geometry import (
    Geometry,
    MetalGeometryPrior,
    PT_ATOMIC_NUMBER,
)


# ---------------------------------------------------------------------------
# Helpers — coord-tensor builders for canonical Pt coordination geometries
# ---------------------------------------------------------------------------
def _make_wired_complex(
    coords_list: list,
    atom_types_list: list,
    pt_idx: int = 0,
    donor_indices: list | None = None,
    device: str = "cpu",
):
    """Build (coords, atom_types, edge_index, edge_types) for a single
    metal complex with explicit donor->metal dative edges.

    Parameters
    ----------
    coords_list : list[list[float]]
        Per-atom coords in angstroms; coords_list[pt_idx] is the metal.
    atom_types_list : list[int]
        Per-atom atomic numbers; atom_types_list[pt_idx] is the metal.
    pt_idx : int
        Index of the metal atom in the coords tensor.
    donor_indices : list[int] | None
        Indices of donor atoms that form dative bonds to the metal.
        When ``None`` (default), every non-metal atom is treated as a
        donor — this matches the wire-up used in the existing
        ``test_metal_geometry.py`` suite.

    Returns
    -------
    coords : (N, 3) float tensor
    atom_types : (N,) long tensor
    edge_index : (2, E) long tensor
    edge_types : (E,) long tensor with EDGE_TYPE_DATIVE on donor->metal
    """
    n = len(coords_list)
    coords = torch.tensor(coords_list, device=device, dtype=torch.float32)
    atom_types = torch.tensor(atom_types_list, device=device, dtype=torch.long)
    # Fully-connected (no self-loop) edges.
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], device=device, dtype=torch.long)
    # EDGE_TYPE_DATIVE = 2 on donor -> metal edges; single (0) elsewhere.
    edge_types = torch.zeros(edge_index.size(1), device=device, dtype=torch.long)
    for k in range(edge_index.size(1)):
        if (
            int(edge_index[1, k].item()) == pt_idx
            and int(edge_index[0, k].item()) != pt_idx
        ):
            edge_types[k] = 2  # EDGE_TYPE_DATIVE
    return coords, atom_types, edge_index, edge_types


# ---------------------------------------------------------------------------
# Test 1 — cisplatin square-planar with 2 NH3 + 2 Cl
# ---------------------------------------------------------------------------
def test_cisplatin_sq_planar_with_2nh3_2cl():
    """Cisplatin Pt(II) square-planar: Pt at origin, 4 donors at +/-x/+/-y.

    Atom types: Pt (Z=78), N (Z=7), N (Z=7), Cl (Z=17), Cl (Z=17).
    All four L-X-L angles are exactly 90 deg (ideal cis) or 180 deg
    (trans L-Cl).  Prior penalty should be effectively zero.
    """
    # Pt at origin; donors at +/-x, +/-y at 2.0 Å (canonical Pt-L bond).
    coords_list = [
        [0.0, 0.0, 0.0],    # Pt (Z=78)
        [2.0, 0.0, 0.0],    # N  +x (NH3)
        [-2.0, 0.0, 0.0],   # N  -x (NH3)
        [0.0, 2.0, 0.0],    # Cl +y
        [0.0, -2.0, 0.0],   # Cl -y
    ]
    atom_types_list = [PT_ATOMIC_NUMBER, 7, 7, 17, 17]
    coords, atom_types, edge_index, edge_types = _make_wired_complex(
        coords_list, atom_types_list, pt_idx=0
    )
    # Sanity: Pt is at index 0 with Z=78.
    assert int(atom_types[0].item()) == PT_ATOMIC_NUMBER
    # Sanity: the 4 donor atoms include 2 N (NH3) and 2 Cl.
    donor_z = [int(atom_types[i].item()) for i in range(1, 5)]
    assert sorted(donor_z) == [7, 7, 17, 17], (
        f"expected 2 NH3 + 2 Cl donors; got {donor_z}"
    )

    # Apply the prior with SQUARE_PLANAR geometry.
    prior = MetalGeometryPrior(weight=1.0)
    loss = prior.prior_loss(
        coords=coords,
        atom_types=atom_types,
        edge_types=edge_types,
        edge_index=edge_index,
        metal_idx=0,
        geometry=Geometry.SQUARE_PLANAR,
    )
    # Perfect geometry -> penalty near zero.  Tolerance ~1 deg
    # (1e-3 rad ~ 0.057 deg, well within arccos clamp noise).
    assert loss.item() < math.radians(1.0), (
        f"Cisplatin square-planar penalty should be ~0; got "
        f"{loss.item():.4f} rad ({math.degrees(loss.item()):.2f} deg)"
    )


# ---------------------------------------------------------------------------
# Test 2 — carboplatin square-planar with 2 NH3 + 2 O-donor (CBDCA chelate)
# ---------------------------------------------------------------------------
def test_carboplatin_sq_planar_with_2nh3_2o_donor():
    """Carboplatin Pt(II) square-planar: 2 NH3 (N donors) + 2 O donors.

    Atom types: Pt (Z=78), N, N, O, O.  The two O donors come from the
    cyclobutane-1,1-dicarboxylate (CBDCA) chelate — one carboxylate
    arm + the other carboxylate arm.  The Pt-O bond length is
    slightly shorter than Pt-N (~1.95 Å vs ~2.05 Å in real X-ray
    structures), but the IDEAL ANGLE is the same (90 deg cis / 180
    deg trans).  Prior penalty is independent of bond length so the
    same ideal angle set applies.
    """
    # Pt at origin; 2 NH3 on x-axis, 2 O donors on y-axis at 2.0 Å.
    coords_list = [
        [0.0, 0.0, 0.0],    # Pt
        [2.0, 0.0, 0.0],    # N +x (NH3)
        [-2.0, 0.0, 0.0],   # N -x (NH3)
        [0.0, 1.95, 0.0],   # O +y (CBDCA carboxylate arm 1)
        [0.0, -1.95, 0.0],  # O -y (CBDCA carboxylate arm 2)
    ]
    atom_types_list = [PT_ATOMIC_NUMBER, 7, 7, 8, 8]
    coords, atom_types, edge_index, edge_types = _make_wired_complex(
        coords_list, atom_types_list, pt_idx=0
    )
    # Sanity: 2 N + 2 O donors (no Cl in carboplatin).
    donor_z = [int(atom_types[i].item()) for i in range(1, 5)]
    assert sorted(donor_z) == [7, 7, 8, 8], (
        f"expected 2 N + 2 O donors; got {donor_z}"
    )
    # Sanity: no Cl donors (carboxylate replaces both Cl in carboplatin).
    assert 17 not in donor_z

    # Apply the prior with SQUARE_PLANAR geometry.
    prior = MetalGeometryPrior(weight=1.0)
    loss = prior.prior_loss(
        coords=coords,
        atom_types=atom_types,
        edge_types=edge_types,
        edge_index=edge_index,
        metal_idx=0,
        geometry=Geometry.SQUARE_PLANAR,
    )
    # Perfect square-planar geometry -> penalty near zero.
    assert loss.item() < math.radians(1.0), (
        f"Carboplatin square-planar penalty should be ~0; got "
        f"{loss.item():.4f} rad ({math.degrees(loss.item()):.2f} deg)"
    )


# ---------------------------------------------------------------------------
# Test 3 — satraplain (sic; "satraplatin" used throughout the codebase)
# Pt(IV) octahedral with 2 Ac + 2 Cl + 2 NH3
# ---------------------------------------------------------------------------
def test_satraplain_pt_iv_octahedral_with_2ac_2cl_2nh3():
    """Satraplatin Pt(IV) octahedral: 2 acetate + 2 Cl + 2 NH3 = CN=6.

    Atom types: Pt (Z=78), N, N, Cl, Cl, O, O (the two acetate O
    donors).  Pt(IV) is d6 octahedral — 6 donors on +/-x, +/-y, +/-z
    at 2.0 Å.  Ideal angles: 90 deg (adjacent) + 180 deg (trans).

    The acetic acid / acetate SMILES abbreviates to ``[O-]C(=O)C``
    or ``OC(C)=O`` — we model the donor as a single O atom at the
    Pt-O bond endpoint (the carboxylate is monodentate on this site).
    """
    # Pt at origin; 6 donors on the +/-x, +/-y, +/-z axes at 2.0 Å.
    # Donor assignment (canonical Pt_IV d6 octahedral):
    #   +x = N (NH3), -x = N (NH3), +y = Cl, -y = Cl,
    #   +z = O (acetate), -z = O (acetate)
    coords_list = [
        [0.0, 0.0, 0.0],    # Pt (Z=78)
        [2.0, 0.0, 0.0],    # N +x (NH3)
        [-2.0, 0.0, 0.0],   # N -x (NH3)
        [0.0, 2.0, 0.0],    # Cl +y
        [0.0, -2.0, 0.0],   # Cl -y
        [0.0, 0.0, 2.0],    # O +z (acetate axial)
        [0.0, 0.0, -2.0],   # O -z (acetate axial)
    ]
    atom_types_list = [PT_ATOMIC_NUMBER, 7, 7, 17, 17, 8, 8]
    coords, atom_types, edge_index, edge_types = _make_wired_complex(
        coords_list, atom_types_list, pt_idx=0
    )
    # Sanity: 6 donors = 2 N + 2 Cl + 2 O.
    donor_z = [int(atom_types[i].item()) for i in range(1, 7)]
    assert sorted(donor_z) == [7, 7, 8, 8, 17, 17], (
        f"expected 2 N + 2 Cl + 2 O donors; got {donor_z}"
    )

    # Apply the prior with OCTAHEDRAL geometry.
    prior = MetalGeometryPrior(weight=1.0)
    loss = prior.prior_loss(
        coords=coords,
        atom_types=atom_types,
        edge_types=edge_types,
        edge_index=edge_index,
        metal_idx=0,
        geometry=Geometry.OCTAHEDRAL,
    )
    # All 15 pairwise angles in {90, 180} deg -> penalty near zero.
    assert loss.item() < math.radians(1.0), (
        f"Satraplatin octahedral penalty should be ~0; got "
        f"{loss.item():.4f} rad ({math.degrees(loss.item()):.2f} deg)"
    )


# ---------------------------------------------------------------------------
# Bonus regression — prior is differentiable end-to-end
# ---------------------------------------------------------------------------
def test_pt_geometry_prior_differentiable_square_planar():
    """Sanity: the prior returns a 0-dim tensor that participates in
    autograd — the standard CFM training loop adds it to the loss.
    We verify gradient flow on the cis-platin coords.
    """
    coords_list = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
    ]
    atom_types_list = [PT_ATOMIC_NUMBER, 7, 7, 17, 17]
    coords, atom_types, edge_index, edge_types = _make_wired_complex(
        coords_list, atom_types_list, pt_idx=0
    )
    coords.requires_grad_(True)
    prior = MetalGeometryPrior(weight=1.0)
    loss = prior.prior_loss(
        coords=coords,
        atom_types=atom_types,
        edge_types=edge_types,
        edge_index=edge_index,
        metal_idx=0,
        geometry=Geometry.SQUARE_PLANAR,
    )
    # Perfect geometry -> loss is ~0 but gradient must still flow
    # (the arccos path is differentiable).
    assert loss.dim() == 0
    loss.backward()
    assert coords.grad is not None, "gradient must flow through arccos"
    # Gradient norm should be small but finite for a perfect geometry.
    grad_norm = float(coords.grad.norm().item())
    assert 0.0 <= grad_norm < 1.0, (
        f"perfect-geometry gradient norm should be small; got {grad_norm}"
    )


# ---------------------------------------------------------------------------
# Bonus regression — distorted geometry must produce positive penalty
# ---------------------------------------------------------------------------
def test_distorted_cisplatin_produces_positive_penalty():
    """A 45-deg-distorted Pt(II) complex must yield a substantial
    penalty well above the perfect-geometry noise floor.
    """
    # Same atom types as cisplatin but with donors at 45-deg spacing
    # (off by 45 deg from ideal square-planar -> some pairs are 45 deg).
    coords_list = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [math.cos(math.radians(45)) * 2.0,
         math.sin(math.radians(45)) * 2.0,
         0.0],
        [math.cos(math.radians(90)) * 2.0,
         math.sin(math.radians(90)) * 2.0,
         0.0],
        [math.cos(math.radians(135)) * 2.0,
         math.sin(math.radians(135)) * 2.0,
         0.0],
    ]
    atom_types_list = [PT_ATOMIC_NUMBER, 7, 7, 17, 17]
    coords, atom_types, edge_index, edge_types = _make_wired_complex(
        coords_list, atom_types_list, pt_idx=0
    )
    prior = MetalGeometryPrior(weight=1.0)
    loss = prior.prior_loss(
        coords=coords,
        atom_types=atom_types,
        edge_types=edge_types,
        edge_index=edge_index,
        metal_idx=0,
        geometry=Geometry.SQUARE_PLANAR,
    )
    # Expect a substantial penalty (45 deg between some pairs -> 0.78 rad).
    assert loss.item() > math.radians(10.0), (
        f"Distorted geometry should produce positive penalty; got "
        f"{loss.item():.4f} rad ({math.degrees(loss.item()):.2f} deg)"
    )
