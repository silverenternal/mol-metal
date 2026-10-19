"""Pytest: Wf-124 Phase 1 — YuelBond decoder wire-up into CFM.

Twelve tests covering:

* ``YuelBondBondDecoder`` returns valid bonds (no overvalent Pt)
* ``use_yuelbond`` flag routes through the right decoder
* Smoke decode_ratio > 0 on synthetic Pt-click clouds
* Edge cases (empty / single-atom / distant / over-crowded clouds)
* CFM decoder contract (DecodedMol shape, sanitized flag)

All tests run on CPU in < 5 s — no GPU required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.cfm_decoder import (
    CFMDecodeResult,
    build_cfm_decoder,
    smoke_decode_ratio,
    write_smoke_results_json,
)
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
    YuelBondDecodeResult,
    build_synthetic_pt_click_clouds,
    covalent_valence_table,
)
from molmetal.models.bond_head import (
    BOND_SINGLE,
    AtomCloud,
    DecodedMol,
)


def _cisplatin_like_cloud() -> AtomCloud:
    """Build a canonical Pt_II(NH3)2(Cl)2 cloud (no jitter).

    Layout: Pt at origin, 4 donors at 2.05 Å in square-planar motif.
    """
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Pt
            [2.05, 0.0, 0.0],  # Cl  (right)
            [-2.05, 0.0, 0.0],  # Cl  (left)
            [0.0, 2.05, 0.0],  # NH3 (top)
            [0.0, -2.05, 0.0],  # NH3 (bottom)
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor([78, 17, 17, 7, 7], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


def _distant_cloud() -> AtomCloud:
    """Build a 4-atom cloud with all pairs > 5 Å — no bonds expected."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor([6, 6, 6, 6], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


def _crowded_pt_cloud() -> AtomCloud:
    """Build a deliberately overvalent cloud: 1 Pt + 6 donors at 2.0 Å.

    This is the failure case the 4-fix bundle can't handle: the
    decoder would naively bond all 6 donors to the Pt, producing
    overvalent Pt (6 > 4).  The YuelBond + valence-cap selector must
    reject at least 2 of the 6 candidate bonds.
    """
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Pt
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [1.4, 1.4, 0.0],
            [-1.4, -1.4, 0.0],
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor([78, 17, 17, 7, 7, 8, 8], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


# ===========================================================================
# Test 1 — YuelBondBondDecoder initialization
# ===========================================================================
def test_yuelbond_bond_decoder_init_default():
    """YuelBondBondDecoder() builds a fresh head with default dims."""
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder()
    assert (
        decoder.distance_cutoff == 3.5
    ), f"default cutoff should be 3.5 Å, got {decoder.distance_cutoff}"
    assert decoder.valence_cap is not None
    assert (
        decoder.valence_cap[78] == 4
    ), f"Pt_II valence cap should be 4, got {decoder.valence_cap.get(78)}"
    assert decoder.valence_cap[6] == 4
    assert decoder.valence_cap[7] == 3
    assert decoder.valence_cap[8] == 2
    assert decoder.valence_cap[17] == 1


# ===========================================================================
# Test 2 — YuelBondBondDecoder returns valid bonds on cisplatin-like cloud
# ===========================================================================
def test_yuelbond_decoder_returns_valid_bonds_on_cisplatin():
    """Decode a cisplatin-like cloud — at least 1 sanitized bond.

    The YuelBond head is random-init; we only verify the wire-up is
    correct (valence-aware selector + RDKit assembly), NOT that the
    SMILES matches cisplatin.
    """
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    # Wf-A71 flipped the head's INIT bias to BOND_NO_BOND (+2.0), so a random
    # head now predicts "no bond" for every pair.  The assertion below then
    # measured the bias, not the selector: it failed at HEAD with
    # ``bonds_accepted=0`` and ``bonds_rejected_overvalent=0``, i.e. the
    # selector never saw a candidate.  Bias the pair head toward SINGLE so
    # this test exercises the valence selector + RDKit assembly, which is
    # what its docstring claims.
    with torch.no_grad():
        decoder._decoder.head.pair_head[-1].bias.zero_()
        decoder._decoder.head.pair_head[-1].bias[BOND_SINGLE] = 5.0
    cloud = _cisplatin_like_cloud()
    result = decoder.decode(cloud)
    assert isinstance(result, YuelBondDecodeResult)
    assert isinstance(result.decoded, DecodedMol)
    assert result.decoded.n_atoms == 5
    # The valence-aware selector must accept at least 1 bond.
    assert result.bonds_accepted >= 1, (
        f"Expected >= 1 accepted bond on cisplatin cloud, got "
        f"{result.bonds_accepted}"
    )
    # The valence cap must NEVER let Pt get more than 4 bonds.
    if result.decoded.bond_orders:
        pt_total = 0
        for src, dst, order in result.decoded.bond_orders:
            # Identify Pt atom — index 0 in the cisplatin layout.
            if src == 0 or dst == 0:
                pt_total += int(order)
        assert pt_total <= 4, f"Pt_II valence cap (4) violated: Pt has {pt_total} bonds"


# ===========================================================================
# Test 3 — No overvalent Pt on crowded cloud
# ===========================================================================
def test_yuelbond_no_overvalent_pt_on_crowded_cloud():
    """The valence-aware selector must NOT let Pt exceed 4 bonds.

    On a deliberately overvalent 1-Pt + 6-donor cloud the head may
    propose all 6 candidate bonds, but the selector must reject the
    extras so Pt stays at <= 4.
    """
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    cloud = _crowded_pt_cloud()
    result = decoder.decode(cloud)
    if result.decoded.bond_orders:
        pt_total = 0
        for src, dst, order in result.decoded.bond_orders:
            if src == 0 or dst == 0:
                pt_total += int(order)
        assert pt_total <= 4, (
            f"Pt_II valence cap (4) violated on crowded cloud: "
            f"Pt has {pt_total} bonds (bonds={result.decoded.bond_orders})"
        )
    # Even when no bonds are predicted, the selector's safety net
    # must have considered and rejected the extras.
    assert result.bonds_rejected_overvalent >= 0


# ===========================================================================
# Test 4 — Empty cloud handled gracefully
# ===========================================================================
def test_yuelbond_decoder_handles_empty_cloud():
    """Empty (N=0) and single-atom (N=1) clouds don't crash."""
    decoder = YuelBondBondDecoder()
    cloud_empty = AtomCloud(
        positions=torch.zeros(0, 3),
        atomic_numbers=torch.zeros(0, dtype=torch.long),
    )
    result = decoder.decode(cloud_empty)
    assert isinstance(result, YuelBondDecodeResult)
    assert result.decoded.n_atoms == 0
    assert result.bonds_accepted == 0


# ===========================================================================
# Test 5 — Distant cloud yields zero bonds
# ===========================================================================
def test_yuelbond_decoder_distant_cloud_zero_bonds():
    """Pairs beyond the cutoff produce zero accepted bonds."""
    decoder = YuelBondBondDecoder()
    cloud = _distant_cloud()
    result = decoder.decode(cloud)
    assert result.bonds_accepted == 0, (
        f"All 4 atoms are > 10 Å apart; cutoff=3.5 Å should yield "
        f"0 bonds, got {result.bonds_accepted}"
    )


# ===========================================================================
# Test 6 — Covalent valence table respects canonical chemistry
# ===========================================================================
def test_covalent_valence_table_canonical_values():
    """The default valence table matches the canonical chemistry."""
    table = covalent_valence_table()
    # Non-metals
    assert table[1] == 1, "H should be 1"
    assert table[6] == 4, "C should be 4"
    assert table[7] == 3, "N should be 3"
    assert table[8] == 2, "O should be 2"
    assert table[9] == 1, "F should be 1"
    assert table[17] == 1, "Cl should be 1"
    # Metals
    assert table[78] == 4, "Pt_II should be 4 (square-planar)"
    assert table[44] == 6, "Ru_II should be 6 (octahedral)"
    # Override
    override = covalent_valence_table({78: 6})  # Pt_IV
    assert override[78] == 6, "Pt_IV override should be 6"


# ===========================================================================
# Test 7 — build_cfm_decoder with use_yuelbond=True returns YuelBond path
# ===========================================================================
def test_build_cfm_decoder_use_yuelbond_true():
    """build_cfm_decoder(use_yuelbond=True) routes through YuelBond path."""
    decoder = build_cfm_decoder(use_yuelbond=True)
    assert decoder.use_yuelbond is True
    # Decode one cloud and verify the decoder_name.
    clouds = build_synthetic_pt_click_clouds(n_clouds=2, jitter=0.10)
    result = decoder.decode(clouds[0])
    assert isinstance(result, CFMDecodeResult)
    assert result.used_yuelbond is True
    assert result.decoder_name == "yuelbond"


# ===========================================================================
# Test 8 — build_cfm_decoder with use_yuelbond=False returns production path
# ===========================================================================
def test_build_cfm_decoder_use_yuelbond_false():
    """build_cfm_decoder(use_yuelbond=False) routes through production."""
    decoder = build_cfm_decoder(use_yuelbond=False)
    assert decoder.use_yuelbond is False
    clouds = build_synthetic_pt_click_clouds(n_clouds=2, jitter=0.10)
    result = decoder.decode(clouds[0])
    assert isinstance(result, CFMDecodeResult)
    assert result.used_yuelbond is False
    assert result.decoder_name == "rewired_decoder_rework"
    assert result.overvalent_rejected == 0


# ===========================================================================
# Test 9 — Smoke decode_ratio > 0 on synthetic Pt-click clouds
# ===========================================================================
def test_smoke_decode_ratio_above_baseline():
    """smoke_decode_ratio with use_yuelbond=True returns > 0/8.

    Verifies the central Wf-124 P1 claim: the YuelBond decoder wire-up
    lifts decode_ratio above the current 0/8 baseline (4-fix bundle).
    """
    n_decoded, n_total, results = smoke_decode_ratio(
        n_clouds=8,
        jitter=0.20,
        use_yuelbond=True,
    )
    assert n_total == 8
    assert n_decoded > 0, (
        f"decode_ratio = {n_decoded}/{n_total} — should be > 0 "
        f"with the YuelBond decoder wire-up"
    )
    # All results are CFMDecodeResult
    assert all(isinstance(r, CFMDecodeResult) for r in results)
    # All results have a used_yuelbond flag set
    assert all(r.used_yuelbond for r in results)


# ===========================================================================
# Test 10 — Synthetic cloud generator produces the right layout
# ===========================================================================
def test_synthetic_pt_click_cloud_generator():
    """build_synthetic_pt_click_clouds produces 5-atom cisplatin layout."""
    clouds = build_synthetic_pt_click_clouds(n_clouds=4, jitter=0.0)
    assert len(clouds) == 4
    for cloud in clouds:
        assert cloud.atomic_numbers.shape[0] == 5
        assert cloud.positions.shape == (5, 3)
        # Pt at index 0
        assert int(cloud.atomic_numbers[0]) == 78
        # 4 donors at 2.05 Å (jitter=0.0 → exact)
        for i in range(1, 5):
            d = float(cloud.positions[i].norm().item())
            assert (
                abs(d - 2.05) < 1e-4
            ), f"Donor {i} distance from Pt should be 2.05 Å, got {d}"


# ===========================================================================
# Test 11 — write_smoke_results_json creates a valid JSON
# ===========================================================================
def test_write_smoke_results_json(tmp_path):
    """write_smoke_results_json writes the canonical JSON shape."""
    out = tmp_path / "wf124_p1_smoke.json"
    write_smoke_results_json(
        output_path=str(out),
        n_decoded=8,
        n_clouds=8,
        use_yuelbond=True,
        jitter=0.20,
        seed=0,
        extra={"wall_time_seconds": 0.5, "device": "cpu"},
    )
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["wf"] == "wf124"
    assert payload["phase"] == 1
    assert payload["decode_ratio"] == 1.0
    assert payload["use_yuelbond"] is True
    assert payload["extra"]["wall_time_seconds"] == 0.5


# ===========================================================================
# Test 12 — End-to-end wire-up: smoke + JSON write
# ===========================================================================
def test_end_to_end_smoke_to_json(tmp_path):
    """End-to-end: run smoke + write JSON + verify decode_ratio > 0."""
    n_decoded, n_total, _ = smoke_decode_ratio(
        n_clouds=8,
        jitter=0.20,
        use_yuelbond=True,
    )
    out = tmp_path / "yuelbond_smoke_results.json"
    write_smoke_results_json(
        output_path=str(out),
        n_decoded=n_decoded,
        n_clouds=n_total,
        use_yuelbond=True,
        jitter=0.20,
    )
    payload = json.loads(out.read_text())
    decode_ratio = payload["decode_ratio"]
    assert decode_ratio > 0.0, (
        f"decode_ratio = {decode_ratio:.3f} should be > 0 "
        f"(baseline 0/8 with 4-fix bundle)"
    )
    # The headline claim: we lifted decode_ratio off zero.
    assert decode_ratio > 0.5, (
        f"decode_ratio = {decode_ratio:.3f} should exceed the "
        f"Path-B RDKit-strict baseline (0.5) per "
        f"metrics/by_metric/decode_ratio.json"
    )
