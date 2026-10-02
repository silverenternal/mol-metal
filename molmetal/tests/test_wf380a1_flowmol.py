"""Wf-380-A1 §2.8 — Test for flowmol_unified SCAFFOLD.

Verifies the licence gate + module importability (Qiang 2024).
"""

from __future__ import annotations

import os
import unittest

import torch

from molmetal.adapters.flow_matching_lipman.flowmol_unified import (
    LICENSE_GATED,
    LICENSE_NAME,
    FlowMolConfig,
    UnifiedAtomBondFlowHead,
)


def test_flowmol_license_gate_flag_set():
    """The module must self-declare LICENSE_GATED = True."""
    assert LICENSE_GATED is True
    assert "arXiv:2412.01966" in LICENSE_NAME or "FlowMol" in LICENSE_NAME


def test_flowmol_head_atom_velocity_shape():
    """atom_head output shape is (B, N, 1) per FlowMol spec."""
    cfg = FlowMolConfig(hidden_dim=64, use_bond_flow=True)
    head = UnifiedAtomBondFlowHead(cfg)
    b, n, e = 2, 5, 8
    h = torch.randn(b, n, cfg.hidden_dim)
    edge_attr_h = torch.randn(b, e, cfg.hidden_dim)
    atom_v, bond_v = head(h, edge_attr_h)
    assert atom_v.shape == (b, n, 1), f"atom_v shape mismatch: {atom_v.shape}"
    assert bond_v.shape == (b, e, 3), f"bond_v shape mismatch: {bond_v.shape}"


@unittest.skipUnless(
    os.environ.get("USER_OK", "0") not in ("1", "true", "True"),
    "USER_OK is set — license gate is intentionally permissive",
)
def test_flowmol_bond_head_zeroed_without_user_ok():
    """Without USER_OK=1, the bond head returns zeros (license gate)."""
    os.environ.pop("USER_OK", None)
    cfg = FlowMolConfig(hidden_dim=32, use_bond_flow=True, user_ok_required=True)
    head = UnifiedAtomBondFlowHead(cfg)
    h = torch.randn(1, 3, cfg.hidden_dim)
    edge_attr_h = torch.randn(1, 4, cfg.hidden_dim)
    _, bond_v = head(h, edge_attr_h)
    assert torch.allclose(bond_v, torch.zeros_like(bond_v), atol=1e-6)


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.flowmol_unified as mod

    assert hasattr(mod, "UnifiedAtomBondFlowHead")
    assert hasattr(mod, "FlowMolConfig")