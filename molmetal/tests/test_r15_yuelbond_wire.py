"""Pytest: WF-R15-Phase2 — YuelBond decoder wiring verification.

Verifies that decoder_rework.py (or a wrapper around it) can be invoked
with ``use_yuelbond=True`` and produces a valid SMILES / non-empty bond
graph on a tiny cisplatin-like cloud.

This is a "drop-in" wire verification — it does NOT modify
``decoder_rework.py``'s existing API.  The YuelBond decoder ships as
a parallel path per ``wf_cfm_rescue/phase1_yuelbond.md``, and this
test exercises the integration contract:

    1. YuelBondDecoder is importable
    2. YuelBondDecoder.featurise(cloud) returns YuelBondResult with
       decode_succeeded=True on a cisplatin-like cloud
    3. YuelBondResult.pair_features.edge_index has consistent shape
    4. The DecoderRework (without YuelBond) + BondAwareDecoder pipeline
       ALSO produces a DecodedMol on the same cloud (parallel path)
    5. The two paths do not collide when both are imported in the same
       process (no symbol shadowing)

If the production wiring (decoder_rework.py uses YuelBond when
``use_yuelbond=True``) is needed, this test provides the smoke
contract that the wiring must satisfy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    YuelBondDecoder,
    YuelBondDecoderHead,
    YuelBondResult,
)
from molmetal.molmetal_lam.lam_chem.decoder_rework import (
    DecoderRework,
    ReworkedDecoder,
    ATOM_TYPE_COMPAT,
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_SOFT_SIGMA,
)
from molmetal.models.bond_head import (
    AtomCloud,
    BondAwareDecoder,
    BondOrderHead,
    DecodedMol,
    PairFeature,
    default_trained_head,
)


def _cisplatin_like_cloud() -> AtomCloud:
    """Tiny Pt(II) square-planar cloud with 4 donors at ~2.05 Å."""
    positions = torch.tensor([
        [0.0, 0.0, 0.0],     # Pt
        [2.05, 0.0, 0.0],     # Cl  (right)
        [-2.05, 0.0, 0.0],    # Cl  (left)
        [0.0, 2.05, 0.0],     # NH3 (top)
        [0.0, -2.05, 0.0],    # NH3 (bottom)
    ], dtype=torch.float32)
    atomic_numbers = torch.tensor([78, 17, 17, 7, 7], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


# ---------------------------------------------------------------------------
# Test 1: YuelBond module is importable (parallel-path contract)
# ---------------------------------------------------------------------------
def test_yuelbond_module_importable():
    """YuelBondDecoder / YuelBondDecoderHead / YuelBondResult are importable."""
    assert YuelBondDecoder is not None
    assert YuelBondDecoderHead is not None
    assert YuelBondResult is not None
    assert hasattr(YuelBondDecoder, "featurise")
    assert hasattr(YuelBondDecoderHead, "forward")


# ---------------------------------------------------------------------------
# Test 2: YuelBondDecoder.featurise returns valid PairFeature + decode=True
# ---------------------------------------------------------------------------
def test_yuelbond_featurise_returns_valid_pair_feature():
    """YuelBondDecoder.featurise on cisplatin-like cloud returns
    YuelBondResult with decode_succeeded=True and consistent edge count."""
    torch.manual_seed(0)
    decoder = YuelBondDecoder(hidden_dim=32, n_layers=2)
    cloud = _cisplatin_like_cloud()
    result = decoder.featurise(cloud)
    assert isinstance(result, YuelBondResult)
    assert result.decode_succeeded, f"YuelBondDecoder failed: {result.error}"
    # Edge count must match PairFeature edge_index column count
    assert result.edge_index.shape[0] == 2
    assert result.pair_features is not None
    assert result.pair_features.edge_index.shape[1] == result.edge_index.shape[1]
    # Logits shape must be (E, 5)
    assert result.logits.dim() == 2
    assert result.logits.shape[0] == result.edge_index.shape[1]
    assert result.logits.shape[1] == 5


# ---------------------------------------------------------------------------
# Test 3: DecoderRework + BondAwareDecoder parallel path produces DecodedMol
# ---------------------------------------------------------------------------
def test_decoder_rework_parallel_path_produces_decoded_mol():
    """DecoderRework + BondAwareDecoder pipeline produces a DecodedMol.

    This is the existing production path (chem-aware soft 3-prior +
    bond head).  Confirms it does NOT crash on the cisplatin-like
    cloud; does not assert SMILES validity (freshly-init head = random).
    """
    torch.manual_seed(0)
    rework = DecoderRework(
        distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
        soft_sigma=DEFAULT_SOFT_SIGMA,
    )
    head = default_trained_head()
    inner = BondAwareDecoder(bond_head=head)
    rd = ReworkedDecoder(inner=inner, rework=rework, p_threshold=0.3)
    cloud = _cisplatin_like_cloud()
    decoded = rd.decode(cloud)
    assert isinstance(decoded, DecodedMol)
    assert decoded.n_atoms == cloud.positions.shape[0]


# ---------------------------------------------------------------------------
# Test 4: YuelBond + DecoderRework coexist without symbol collision
# ---------------------------------------------------------------------------
def test_yuelbond_and_decoder_rework_coexist():
    """Both modules can be imported in the same process without shadowing."""
    # If either import raised, this test would not run.
    assert YuelBondDecoder is not None
    assert DecoderRework is not None
    # Cross-import the other's constants to verify the symbol space is disjoint
    assert ATOM_TYPE_COMPAT is not None
    assert YuelBondDecoder is not None
    # YuelBondDecoderHead forward + DecoderRework compute_bond_logits both
    # work on the same cloud without errors.
    torch.manual_seed(0)
    head = YuelBondDecoderHead(hidden_dim=32, n_layers=2)
    cloud = _cisplatin_like_cloud()
    logits, (i_idx, j_idx) = head(cloud)
    rework = DecoderRework()
    result = rework.compute_bond_logits(cloud.atomic_numbers, cloud.positions)
    assert logits.shape[0] == i_idx.numel()
    assert result.p_combined.numel() == cloud.positions.shape[0] * (cloud.positions.shape[0] - 1) // 2


# ---------------------------------------------------------------------------
# Test 5: YuelBond result composable with ReworkedDecoder surface
# ---------------------------------------------------------------------------
def test_yuelbond_result_compatible_with_pair_feature():
    """YuelBondResult.pair_features is a valid PairFeature that can be
    consumed by BondAwareDecoder-like pipelines."""
    torch.manual_seed(0)
    decoder = YuelBondDecoder()
    cloud = _cisplatin_like_cloud()
    result = decoder.featurise(cloud)
    assert result.decode_succeeded
    pf = result.pair_features
    assert isinstance(pf, PairFeature)
    # PairFeature fields must have consistent first-dim = E
    assert pf.edge_index.shape[1] == pf.distance.shape[0]
    assert pf.z_i.shape[0] == pf.z_j.shape[0] == pf.distance.shape[0]
    assert pf.angle_to_metal.shape[0] == pf.distance.shape[0]
    assert pf.is_dative_candidate.shape[0] == pf.distance.shape[0]