# SPDX-License-Identifier: Apache-2.0
"""Wf-A107f — the ring metric must say WHICH graph it measured.

WHY THIS EXISTS
---------------
Wf-A107c/d (HONEST_NEGATIVE §6.162) measured that the bond head's RAW 3-ring
share moved by 0.0045 between two arms while the assembled metric moved by
0.1211 -- a 27x amplification applied by ``assemble_connectivity`` -> RDKit on
a point cloud that was not a molecule.  Only the assembled number was in the
epoch record, so two waves of A/B read the amplification as chemistry and the
local probe that disagreed looked like a contradiction.

Two fixes are pinned here:

1. :func:`molmetal.eval.ring_profile.ring_sizes_from_edges` -- an SSSR profile
   for a graph that has not been assembled yet, so the pre-assembly number can
   be recorded beside the post-assembly one.
2. ``decode_clouds_and_classify`` attaches ``ring_sizes_raw`` to every chem row
   it emits, and ``summarise_decode_chemistry`` surfaces it as ``raw_*``.  The
   two blocks are computed on the SAME molecules, so their ratio is the
   assembler's amplification rather than a difference in denominator.

RED CHECK: drop the ``ring_sizes_raw`` assignment and
``test_decode_loop_attaches_the_pre_assembly_profile`` fails; return the
post-assembly sizes from ``ring_sizes_from_edges`` and
``test_raw_profile_is_the_pre_assembly_graph`` fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, List, Optional

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _trainer():
    if "_wfa107f_trainer" in sys.modules:
        return sys.modules["_wfa107f_trainer"]
    spec = importlib.util.spec_from_file_location("_wfa107f_trainer", str(TRAINER))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_wfa107f_trainer"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# 1. ring_sizes_from_edges
# ---------------------------------------------------------------------------
def test_triangle_is_one_three_ring_and_the_substituent_is_not_a_ring():
    from molmetal.eval.ring_profile import ring_sizes_from_edges

    # 0-1-2 triangle, 3 hangs off atom 2.
    assert ring_sizes_from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)]) == [3]


def test_hexagon_is_one_six_ring():
    from molmetal.eval.ring_profile import ring_sizes_from_edges

    edges = [(i, (i + 1) % 6) for i in range(6)]
    assert ring_sizes_from_edges(6, edges) == [6]


def test_two_fused_triangles_are_two_three_rings():
    """SSSR, not 'all simple cycles': the outer 4-cycle is not a ring."""
    from molmetal.eval.ring_profile import ring_sizes_from_edges

    # squares 0-1-2-3 and 2-3-4-5 share the edge 2-3
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5), (5, 3)]
    assert ring_sizes_from_edges(6, edges) == [4, 4]


def test_duplicate_self_loop_and_out_of_range_edges_are_dropped():
    from molmetal.eval.ring_profile import ring_sizes_from_edges

    # a duplicate would make RWMol.AddBond raise; a self-loop is not a ring.
    assert ring_sizes_from_edges(
        3, [(0, 1), (0, 1), (0, 0), (1, 2), (2, 0), (0, 99)]
    ) == [3]


def test_acyclic_graph_has_no_rings():
    from molmetal.eval.ring_profile import ring_sizes_from_edges

    assert ring_sizes_from_edges(4, [(0, 1), (1, 2), (2, 3)]) == []


def test_unsanitisable_valence_still_yields_its_rings():
    """A carbon with 5 bonds cannot be sanitised; the profile must not raise.

    This is the whole point of using an unsanitised RWMol: the bond head emits
    over-valent graphs, and a ring measurement that refuses to look at them
    would be blind exactly where the model is wrong.
    """
    from molmetal.eval.ring_profile import ring_sizes_from_edges

    hub = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2)]
    assert ring_sizes_from_edges(6, hub) == [3]


# ---------------------------------------------------------------------------
# 2. the decode loop records the pre-assembly graph
# ---------------------------------------------------------------------------
class _FakeDecoded:
    def __init__(self, smiles: str, bond_orders: Optional[List] = None,
                 n_atoms: int = 0):
        self.smiles = smiles
        self.sanitized = True
        self.error = ""
        self.bond_orders = bond_orders or []
        self.n_atoms = n_atoms


class _FakeResult:
    def __init__(self, decoded):
        self.decoded = decoded


class _ScriptedDecoder:
    def __init__(self, decodeds):
        self.decodeds = decodeds
        self.n = 0

    def decode(self, cloud):
        d = self.decodeds[self.n]
        self.n += 1
        return _FakeResult(d)


class _Cloud:
    def __init__(self):
        import torch

        self.atomic_numbers = torch.tensor([6, 6, 6, 6], dtype=torch.long)


def _decode_one(decoded) -> List[Any]:
    t = _trainer()
    rows: List[Any] = []
    t.decode_clouds_and_classify(
        _ScriptedDecoder([decoded]), [_Cloud()], [], None, None, rows, None, {}
    )
    return rows


def test_decode_loop_attaches_the_pre_assembly_profile():
    # benzene ring on atoms 0-5 of a 6-atom cloud, no substituent
    ring = [(i, (i + 1) % 6, 1) for i in range(6)]
    rows = _decode_one(_FakeDecoded("c1ccccc1", ring, n_atoms=6))
    assert rows and rows[0] is not None
    assert rows[0]["ring_sizes_raw"] == [6], (
        "the raw profile must be the head's own accepted graph; got "
        f"{rows[0].get('ring_sizes_raw')!r}"
    )


def test_raw_profile_is_the_pre_assembly_graph():
    """A graph RDKit would reject still reports ITS rings.

    The decoded SMILES here is acyclic, so the assembled profile is empty --
    but the head proposed a triangle.  A raw block that copied the assembled
    number would be empty too, and the assembler's contribution would stay
    invisible.
    """
    rows = _decode_one(_FakeDecoded("CC", [(0, 1, 1), (1, 2, 1), (2, 0, 1)], n_atoms=3))
    assert rows[0]["ring_sizes"] == []
    assert rows[0]["ring_sizes_raw"] == [3]


def test_raw_profile_absent_on_a_failed_decode():
    """A raise must not leave a half-filled row: the loop appends None."""
    t = _trainer()
    rows: List[Any] = []

    class _Boom:
        def decode(self, cloud):
            raise RuntimeError("simulated")

    t.decode_clouds_and_classify(_Boom(), [_Cloud()], [], None, None, rows, None, {})
    assert rows == [None]


# ---------------------------------------------------------------------------
# 3. the summary carries both blocks
# ---------------------------------------------------------------------------
def test_summary_emits_raw_block_alongside_the_assembled_one():
    t = _trainer()
    row = {
        "n_fragments": 1,
        "ring_sizes": [6],
        "ring_sizes_raw": [3],
        "n_rings": 1,
        "n_heavy": 6,
        "largest_fragment_frac": 1.0,
        "has_metal": False,
        "seed_metal_present": None,
        "seed_metal_z": [],
        "sa_score": None,
        "sa_target": None,
    }
    out = t.summarise_decode_chemistry([row])
    assert out["ring_3_share"] == 0.0 and out["ring_5_6_share"] == 1.0
    assert out["raw_ring_3_share"] == 1.0 and out["raw_ring_5_6_share"] == 0.0
    assert out["raw_ring_n_molecules"] == 1


def test_summary_raw_block_is_zeroed_not_missing_when_nothing_decoded():
    t = _trainer()
    out = t.summarise_decode_chemistry([])
    for key in (
        "raw_ring_size_distribution",
        "raw_rings_per_mol_mean",
        "raw_ring_5_6_share",
        "raw_ring_3_share",
        "raw_ring_n_molecules",
    ):
        assert key in out, f"{key} must exist so a reader never branches on n"


# ---------------------------------------------------------------------------
# 4. the epoch record can express both, plus the FM term
# ---------------------------------------------------------------------------
def test_epoch_metrics_has_the_new_fields():
    t = _trainer()
    m = t.EpochMetrics(
        epoch=0,
        wall_time_s=0.0,
        train_loss=0.0,
        bond_ce_loss=0.0,
        wasserstein_loss=0.0,
        catherine_audit_pass=False,
        decode_ratio_rdkit_strict=0.0,
        decode_ratio_val=0,
        decode_ratio_n=0,
        decoding_attempted=False,
        lr=0.0,
        grad_norm=0.0,
    )
    assert m.gen_raw_ring_3_share == 0.0
    assert m.gen_raw_ring_5_6_share == 0.0
    assert m.gen_raw_rings_per_mol == 0.0
    assert m.fm_term == 0.0


def test_epoch_term_means_channel_exists():
    """The side-channel that makes `fm_term` visible is a plain float dict.

    It has to be a separate dict from `_LAST_LOSS_TERMS`: that one holds
    tensors and is cleared every batch, so an epoch mean cannot live there.
    """
    t = _trainer()
    assert isinstance(t._LAST_EPOCH_TERM_MEANS, dict)
    t._LAST_EPOCH_TERM_MEANS["fm_term"] = 1.25
    assert t._LAST_EPOCH_TERM_MEANS["fm_term"] == pytest.approx(1.25)
    t._LAST_EPOCH_TERM_MEANS.clear()


# ---------------------------------------------------------------------------
# 5. the analysis script must not call SINGLE-class PR-AUC "bond AP"
# ---------------------------------------------------------------------------
def test_ab_analysis_labels_the_two_ap_fields_separately():
    import importlib.util as _u

    path = REPO / "molmetal" / "scripts" / "wfa107e_ab_analysis.py"
    spec = _u.spec_from_file_location("_wfa107f_ab", str(path))
    mod = _u.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert "bond_pr_auc" in mod._METRICS
    assert "bond_class_1_pr_auc" in mod._METRICS
    assert "existence" in mod._METRIC_LABELS["bond_pr_auc"].lower()
    assert "single" in mod._METRIC_LABELS["bond_class_1_pr_auc"].lower()
