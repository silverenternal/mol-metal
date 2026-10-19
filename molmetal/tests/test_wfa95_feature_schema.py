"""Wf-A95 — a checkpoint decodes with the donor set it was trained under.

``is_dative`` is computed at DECODE time, so adding elements to the donor
set silently changes the pair-head input for every checkpoint that predates
the addition.  These tests pin the two halves of the fix:

1. the head's donor set is a constructor input (not a module global);
2. ``restore_feature_schema`` applies a checkpoint's stamped set + cutoff,
   and says so when the checkpoint carries no stamp.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.models.bond_head import AtomCloud
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    _DONOR_ZS,
    YuelBondDecoder,
    YuelBondDecoderHead,
)
from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
    checkpoint_feature_schema,
    restore_feature_schema,
)


def _two_atom_cloud(z_a: int, z_b: int, d: float) -> AtomCloud:
    return AtomCloud(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float32
        ),
        atomic_numbers=torch.tensor([z_a, z_b], dtype=torch.long),
    )


def test_head_donor_set_is_a_constructor_input() -> None:
    """A head built with the pre-A94 set does not flag Pt-Cl as dative."""
    narrow = YuelBondDecoderHead(hidden_dim=8, n_layers=1, donor_zs=(7, 8, 16))
    assert narrow.donor_zs == frozenset({7, 8, 16})
    pf = YuelBondDecoder(head=narrow).featurise(_two_atom_cloud(78, 17, 2.2))
    assert pf.pair_features is not None
    assert not bool(pf.pair_features.is_dative_candidate.any())

    wide = YuelBondDecoderHead(hidden_dim=8, n_layers=1)
    assert wide.donor_zs == frozenset(_DONOR_ZS)
    pf = YuelBondDecoder(head=wide).featurise(_two_atom_cloud(78, 17, 2.2))
    assert bool(pf.pair_features.is_dative_candidate.any())


def test_restore_feature_schema_applies_donor_set_and_cutoff(capsys) -> None:
    head = YuelBondDecoderHead(hidden_dim=8, n_layers=1)
    model = SimpleNamespace(head=head)
    decoder = YuelBondBondDecoder(hidden_dim=8, n_layers=1)
    decoder.dative_max_distance = 2.6
    ckpt = {
        "feature_schema": {
            "donor_zs": [7, 8, 16],
            "dative_max_distance": 2.2,
            "metal_placement": "legacy",
        }
    }
    schema = restore_feature_schema(model, decoder, ckpt)
    assert schema["metal_placement"] == "legacy"
    assert head.donor_zs == frozenset({7, 8, 16})
    assert decoder.dative_max_distance == 2.2
    out = capsys.readouterr().out
    assert "restored feature_schema" in out
    assert "dative_max_distance=2.20" in out


def test_restore_without_schema_keeps_current_defaults_and_says_so(capsys) -> None:
    head = YuelBondDecoderHead(hidden_dim=8, n_layers=1)
    model = SimpleNamespace(head=head)
    decoder = YuelBondBondDecoder(hidden_dim=8, n_layers=1)
    before = frozenset(head.donor_zs)
    assert restore_feature_schema(model, decoder, {"model_state_dict": {}}) == {}
    assert head.donor_zs == before
    out = capsys.readouterr().out
    assert "no feature_schema" in out
    assert "may differ" in out


def test_checkpoint_feature_schema_ignores_malformed_stamps() -> None:
    assert checkpoint_feature_schema(None) == {}
    assert checkpoint_feature_schema({"feature_schema": None}) == {}
    assert checkpoint_feature_schema({"feature_schema": [1, 2]}) == {}
    good = {"feature_schema": {"donor_zs": [7]}}
    assert checkpoint_feature_schema(good) == {"donor_zs": [7]}


def test_edge_head_dative_rule_is_order_free_and_uses_donor_zs() -> None:
    """Wf-A95 — the edge-MPNN head had the PRE-Wf-A92 rule.

    It tested a hardcoded N/O/P/S mask as ``donor(i) & metal(j)`` over triu
    indices: no halides, no carbon, and atom-order dependent.  The fix reads
    ``self.donor_zs`` and tests both directions.  This test reads the
    ``is_dative`` column of the edge_init input (2*H + 3) for both orders.
    """
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondEdgeMPNNHead,
    )

    head = YuelBondEdgeMPNNHead(hidden_dim=8, n_edge_layers=1)
    captured = {}

    def _hook(module, args, output):
        captured["x"] = args[0].detach()

    handle = head.edge_init.register_forward_hook(_hook)
    try:
        z = torch.tensor([78, 17], dtype=torch.long)
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]], dtype=torch.float32
        )
        head(AtomCloud(positions=pos, atomic_numbers=z))
        col = 2 * head.hidden_dim + 3
        assert float(captured["x"][0, col]) == 1.0
        perm = torch.tensor([1, 0], dtype=torch.long)
        head(AtomCloud(positions=pos[perm], atomic_numbers=z[perm]))
        assert float(captured["x"][0, col]) == 1.0
    finally:
        handle.remove()


def test_decoder_z_rule_reads_the_head_set() -> None:
    """A Pt-Cl pair is dative for a wide head and not for a narrow one."""
    cloud = _two_atom_cloud(78, 17, 2.2)
    wide = YuelBondBondDecoder(hidden_dim=8, n_layers=1)
    assert wide._is_dative_pair_at(cloud, 0, 1, 78, 17, None)
    wide._decoder.head.donor_zs = frozenset({7, 8, 16})
    assert not wide._is_dative_pair_at(cloud, 0, 1, 78, 17, None)