#!/usr/bin/env python
"""Wf-A142 — regression tests for the bond-order DEMOTION demonstration.

What is guarded here
--------------------
1. The published rule (Yao et al., arXiv:2609.08333 §C.2 / Fig. 7) is
   quoted IN THE SOURCE at the site that implements it, and the
   implementation really is a demotion (a lowered ORDER) rather than a
   deletion (a removed EDGE).
2. The demotion path is REACHABLE and it is the thing that rescues a
   forced bridge: a saturated endpoint carrying a DOUBLE bond gets its
   order lowered and the bridge is added; with the demotion disabled
   (the counterfactual) the same bridge is refused.
3. The published rule has a real LIMIT: a saturated endpoint whose bonds
   are all SINGLE cannot be demoted, the bridge is refused, and the
   refused case is what the "delete an edge" alternative would have
   covered.
4. No edge is ever deleted on the demotion path.
5. The Wf-A142 demonstration script runs end-to-end on CPU and emits a
   report whose schema is stable.
6. The §5.5 filesystem-truth correction appended to
   `molmetal/reports/cfm_c7_fullgen_2026_10_20/synthesis.md` is present
   and says what it must say (the advertised checkpoint does not exist).

CPU-only: no CUDA call, no checkpoint load, no model forward.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
from rdkit import Chem

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
)
from molmetal.scripts.wfa142_demotion_demo import (
    _edges_to_delete_for_budget,
    _NoDemotion,
    _saturated_atoms,
    candidate_pairs,
    embed_3d,
    evaluate_scenario,
    heavy_arrays,
    place_b_next_to_atom,
    run_forced,
)
from molmetal.scripts.wfa142_demotion_demo import main as demo_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DECODER_SRC = (
    _REPO_ROOT
    / "molmetal"
    / "molmetal_lam"
    / "lam_chem"
    / "cfm_decoder_yuelbond.py"
)
_SYNTHESIS = (
    _REPO_ROOT
    / "molmetal"
    / "reports"
    / "cfm_c7_fullgen_2026_10_20"
    / "synthesis.md"
)
_REPORT = (
    _REPO_ROOT / "molmetal" / "reports" / "wfa142_demotion_demo" / "demo.json"
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _one_fragment_scenario(smiles_a: str, smiles_b: str, seed: int = 7):
    """Two fragments, placed so the bridge pair is (saturated atom of A, an atom of B).

    Returns ``(decoder, z, pos, bonds, pairs, bridge)``.
    """
    decoder = YuelBondBondDecoder()
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    assert mol_a is not None and mol_b is not None
    a3 = embed_3d(mol_a, seed)
    b3 = embed_3d(mol_b, seed + 1)
    assert a3 is not None and b3 is not None
    arr_a = heavy_arrays(a3)
    arr_b = heavy_arrays(b3)
    assert arr_a is not None and arr_b is not None
    z_a, pos_a, bonds_a = arr_a
    z_b, pos_b, bonds_b = arr_b
    offset = len(z_a)

    sat_demotable, sat_plain = _saturated_atoms(decoder, z_a, bonds_a)
    assert sat_demotable, "fixture must have a saturated, demotable heavy atom"
    atom_a = int(sat_demotable[0])

    shift, atom_b = place_b_next_to_atom(pos_a, pos_b, atom_a, 2.9)
    pos_b2 = pos_b + shift.astype("float32")
    z = list(z_a) + list(z_b)
    pos = list(pos_a) + list(pos_b2)
    bonds = list(bonds_a) + [(i + offset, j + offset, o) for i, j, o in bonds_b]
    pairs, _d = candidate_pairs(z, pos, decoder)
    bridge = (atom_a, atom_b + offset)
    return decoder, z, pos, bonds, pairs, bridge, sat_plain


# ----------------------------------------------------------------------
# 1. the published rule is quoted at the implementation site
# ----------------------------------------------------------------------
def test_published_rule_quoted_at_the_demotion_site():
    src = _DECODER_SRC.read_text(encoding="utf-8")
    assert "2609.08333" in src, "the arXiv id of the published repair must be cited"
    assert "lowered one level at a time" in src
    assert "No edge is deleted" in src
    # The helper really exists, and it is a method of the decoder.
    assert hasattr(YuelBondBondDecoder, "_demote_one_order")
    assert hasattr(YuelBondBondDecoder, "_force_connectivity")


# ----------------------------------------------------------------------
# 2. the demotion path rescues a forced bridge; the counterfactual does not
# ----------------------------------------------------------------------
def test_demotion_rescues_a_forced_bridge():
    """Acetone's carbonyl carbon is saturated AND carries a DOUBLE bond."""
    decoder, z, pos, bonds, pairs, bridge, _ = _one_fragment_scenario(
        "CC(C)=O", "C"
    )
    assert bridge in pairs, "the bridge pair must be inside the head's candidate set"

    res = evaluate_scenario(decoder, z, pos, bonds, pairs, bridge)
    assert res["outcome"] == "rescued_by_demotion", res
    assert res["bonds_added_with_demotion"] == 1
    assert res["bonds_added_without_demotion"] == 0, (
        "with the published repair disabled the bridge must be refused — "
        "otherwise the measurement is not measuring the demotion"
    )
    assert res["bonds_demoted"] >= 1
    assert res["orders_lowered"] >= 1
    assert res["edges_deleted"] == 0, "the published rule forbids deleting an edge"
    assert res["sanitize_after_forced_stage"] == "ok"


def test_demotion_only_lowers_an_order_it_never_removes_an_edge():
    """After the forced stage the edge set is a SUPERSET of the input's."""
    decoder, z, pos, bonds, pairs, bridge, _ = _one_fragment_scenario(
        "CC(C)=O", "C"
    )
    from molmetal.scripts.wfa142_demotion_demo import build_logits, make_yres
    import torch

    logits = build_logits(pairs, bonds, bridge)
    edge_index = torch.tensor(
        [[p[0] for p in pairs], [p[1] for p in pairs]], dtype=torch.long
    ).reshape(2, len(pairs))
    from molmetal.models.bond_head import AtomCloud

    cloud = AtomCloud(
        positions=torch.tensor(pos, dtype=torch.float32),
        atomic_numbers=torch.tensor(z, dtype=torch.long),
    )
    yres = make_yres(edge_index, logits)
    _stats, after = run_forced(decoder, cloud, yres, list(bonds))

    before_pairs = {(min(i, j), max(i, j)) for i, j, _o in bonds}
    after_pairs = {(min(i, j), max(i, j)) for i, j, _o in after}
    assert before_pairs.issubset(after_pairs), "an edge was deleted"
    before_orders = {(min(i, j), max(i, j)): o for i, j, o in bonds}
    after_orders = {(min(i, j), max(i, j)): o for i, j, o in after}
    lowered = [
        k for k, o0 in before_orders.items() if after_orders[k] < o0
    ]
    assert lowered, "the demotion must lower at least one order"
    assert all(after_orders[k] in (1, 2) for k in lowered)


# ----------------------------------------------------------------------
# 3. the limit: a saturated endpoint with no demotable bond cannot be saved
# ----------------------------------------------------------------------
def test_saturated_endpoint_without_a_demotable_bond_is_refused():
    """Neopentane's central carbon has four SINGLE bonds: nothing to lower."""
    decoder = YuelBondBondDecoder()
    mol = Chem.MolFromSmiles("CC(C)(C)C")
    assert mol is not None
    m3 = embed_3d(mol, 11)
    assert m3 is not None
    arr = heavy_arrays(m3)
    assert arr is not None
    z_a, pos_a, bonds_a = arr
    sat_demotable, sat_plain = _saturated_atoms(decoder, z_a, bonds_a)
    assert sat_demotable == []
    assert sat_plain, "neopentane's central carbon must be saturated"

    mol_b = Chem.MolFromSmiles("C")
    b3 = embed_3d(mol_b, 12)
    arr_b = heavy_arrays(b3)
    assert arr_b is not None
    z_b, pos_b, _bonds_b = arr_b
    offset = len(z_a)

    atom_a = int(sat_plain[0])
    shift, atom_b = place_b_next_to_atom(pos_a, pos_b, atom_a, 2.9)
    z = list(z_a) + list(z_b)
    pos = list(pos_a) + list(pos_b + shift.astype("float32"))
    bonds = list(bonds_a)
    pairs, _d = candidate_pairs(z, pos, decoder)
    bridge = (atom_a, atom_b + offset)
    assert bridge in pairs

    res = evaluate_scenario(decoder, z, pos, bonds, pairs, bridge)
    assert res["outcome"] == "bridge_refused", res
    assert res["bonds_added_with_demotion"] == 0
    assert res["bonds_demoted"] == 0
    # The alternative the published rule forbids: deleting an edge.
    n_del = _edges_to_delete_for_budget(bonds, atom_a)
    assert n_del == 1, "one deleted single bond would have freed the budget"


# ----------------------------------------------------------------------
# 4. the counterfactual helper really disables the demotion
# ----------------------------------------------------------------------
def test_no_demotion_context_manager_restores_the_method():
    decoder = YuelBondBondDecoder()
    original = decoder._demote_one_order
    with _NoDemotion(decoder):
        assert decoder._demote_one_order is not original
        assert decoder._demote_one_order(  # type: ignore[operator]
            [], 0, [6], {}
        ) is None
    assert decoder._demote_one_order == original


# ----------------------------------------------------------------------
# 5. the script runs end-to-end and its report schema is stable
# ----------------------------------------------------------------------
def test_demo_script_runs_and_emits_a_stable_report(tmp_path):
    out = tmp_path / "demo.json"
    rc = demo_main(
        [
            "--max-molecules",
            "6",
            "--max-reference",
            "8",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gpu_used"] is False
    for key in (
        "A_natural_placement",
        "B1_saturated_endpoint_with_demotable_bond",
        "B2_saturated_endpoint_without_demotable_bond",
        "C_contrast_cut_own_bond",
        "D_positive_control_real_chemistry",
    ):
        assert key in report["experiments"], key
        exp = report["experiments"][key]
        assert "rescued_by_demotion" in exp
        assert "bridge_without_demotion" in exp
        assert "bridge_refused" in exp
    census = report["EXP0_pool_chemistry_census"]
    assert "saturated_heavy_atoms" in census
    assert "saturated_with_demotable_bond" in census
    impl = report["implementation_under_test"]
    assert ":1959" in impl["demotion_helper"]
    assert ":1701" in impl["caller"]


@pytest.mark.skipif(not _REPORT.exists(), reason="demo.json not produced yet")
def test_published_report_is_tagged_measured():
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert report["tag"].startswith("MEASURED")
    assert report["gpu_used"] is False
    # The positive control must have fired somewhere, otherwise the
    # measurement harness itself is broken.
    control = report["experiments"]["D_positive_control_real_chemistry"]
    assert control["rescued_by_demotion"] > 0, (
        "the demotion path never fired on real chemistry — the harness "
        "cannot see the effect it is trying to measure"
    )


# ----------------------------------------------------------------------
# 6. the §5.5 filesystem-truth correction on synthesis.md
# ----------------------------------------------------------------------
def test_synthesis_md_carries_the_dated_checkpoint_correction():
    text = _SYNTHESIS.read_text(encoding="utf-8")
    assert "## Outputs" in text
    # The historical (false) line must still be there — §5.5 rule #6 makes
    # the correction ADDITIVE, never a deletion.
    assert (
        "molmetal/reports/cfm_c7_fullgen_2026_10_20/checkpoints/"
        "cfm_yuelbond_wasserstein.pt" in text
    )
    # And the dated correction must be present.
    assert "2026-10-31" in text
    assert "CORRECTION" in text.upper()
    assert "does NOT exist on disk" in text or "does not exist on disk" in text.lower()
    assert "ls -la" in text
    assert "git log" in text
    assert "6.248" in text
