"""Wf-A92 (2026-10-19) — dative provenance + order-invariant dative features.

Two defects this pins:

1. ``YuelBondDecoderHead.forward`` built ``is_dative`` as
   ``donor(i) & metal(j)`` on a ``triu`` (i<j) edge list, so the same
   physical metal-donor bond was dative or not depending on atom ORDER,
   and every metal-first cloud (tmQM) lost the feature entirely.
2. ``DecodedMol.dative_bonds`` recorded a Z-only guess as if it were
   observed chemistry, and ``teacher_forced`` was never surfaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.models.bond_head import (
    BOND_SINGLE,
    AtomCloud,
    DecodedMol,
    _classify_dative_source,
    _dative_truth_pairs,
)
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    _DONOR_ZS,
    YuelBondDecoder,
    YuelBondDecoderHead,
)


def _two_atom_cloud(z_a: int, z_b: int, d: float, truth=None) -> AtomCloud:
    kwargs = dict(
        positions=torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float32),
        atomic_numbers=torch.tensor([z_a, z_b], dtype=torch.long),
    )
    if truth is not None:
        kwargs["bonds"] = torch.tensor([[0], [1]], dtype=torch.long)
        kwargs["bond_types"] = torch.tensor([BOND_SINGLE], dtype=torch.long)
        kwargs["is_dative_pair"] = torch.tensor([bool(truth)], dtype=torch.bool)
    return AtomCloud(**kwargs)


def _biased_to_single(decoder: YuelBondBondDecoder) -> None:
    with torch.no_grad():
        decoder._decoder.head.pair_head[-1].bias.zero_()
        decoder._decoder.head.pair_head[-1].bias[BOND_SINGLE] = 5.0


# ---------------------------------------------------------------------------
# DecodedMol warnings
# ---------------------------------------------------------------------------
def test_decoded_mol_derives_warnings_from_its_fields() -> None:
    d = DecodedMol(teacher_forced=True, dative_manufactured=2)
    assert "teacher_forced" in d.warnings
    assert "dative_manufactured" in d.warnings
    assert DecodedMol().warnings == []
    assert DecodedMol(dative_source="z_rule").warnings == []


# ---------------------------------------------------------------------------
# Truth lookup
# ---------------------------------------------------------------------------
def test_dative_truth_pairs_is_direction_free() -> None:
    cloud = _two_atom_cloud(78, 7, 2.0, truth=True)
    truth = _dative_truth_pairs(cloud)
    assert truth is not None
    assert (0, 1) in truth and (1, 0) in truth


def test_dative_truth_pairs_is_none_without_truth() -> None:
    assert _dative_truth_pairs(_two_atom_cloud(78, 7, 2.0)) is None
    # A flags tensor that does not align with the bond list is not truth.
    cloud = _two_atom_cloud(78, 7, 2.0, truth=True)
    cloud.is_dative_pair = torch.tensor([True, False], dtype=torch.bool)
    assert _dative_truth_pairs(cloud) is None


# ---------------------------------------------------------------------------
# Provenance classification
# ---------------------------------------------------------------------------
def test_classify_dative_source_no_dative_bonds() -> None:
    assert _classify_dative_source(_two_atom_cloud(78, 7, 2.0), []) == ("", 0)


def test_classify_dative_source_without_truth_is_z_rule() -> None:
    cloud = _two_atom_cloud(78, 7, 2.0)
    assert _classify_dative_source(cloud, [(1, 0)]) == ("z_rule", 0)


def test_classify_dative_source_truth_backed_and_manufactured() -> None:
    dative = _two_atom_cloud(78, 7, 2.0, truth=True)
    assert _classify_dative_source(dative, [(1, 0)]) == ("truth", 0)
    covalent = _two_atom_cloud(78, 7, 2.0, truth=False)
    assert _classify_dative_source(covalent, [(1, 0)]) == ("z_rule", 1)


# ---------------------------------------------------------------------------
# Order invariance of the head's dative feature
# ---------------------------------------------------------------------------
def _dative_column_for_pair(head, cloud, a: int, b: int) -> float:
    """Value of the ``is_dative`` column fed to ``pair_head`` for one pair.

    The pair head concatenates ``[h_i, h_j, d, is_dative, is_short]``, so the
    dative column sits at ``2 * hidden_dim + 1``.  Reading the head's INPUT
    is the point: the head itself is direction-sensitive by design (it
    concatenates ``h_i`` then ``h_j``), so equal LOGITS across a permutation
    would be the wrong assertion.  Equal dative COLUMNS is the property that
    was broken.
    """
    captured: dict = {}

    def _hook(_module, args, _output):
        captured["x"] = args[0].detach()

    handle = head.pair_head[0].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _, (i_idx, j_idx) = head(cloud)
    finally:
        handle.remove()
    x = captured["x"]
    col = 2 * int(head.hidden_dim) + 1
    for k in range(int(i_idx.numel())):
        if {int(i_idx[k]), int(j_idx[k])} == {a, b}:
            return float(x[k, col].item())
    raise AssertionError(f"pair {(a, b)} is not in the edge list")


def test_dative_feature_does_not_depend_on_atom_order() -> None:
    torch.manual_seed(0)
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=2)
    head.eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.05, 0.0, 0.0], [-2.05, 0.0, 0.0]],
        dtype=torch.float32,
    )
    z = torch.tensor([78, 7, 7], dtype=torch.long)
    metal_first = AtomCloud(positions=positions, atomic_numbers=z)
    # Same geometry, metal LAST: the Pt-N pairs are now (1, 2) and (0, 2).
    perm = torch.tensor([1, 2, 0], dtype=torch.long)
    metal_last = AtomCloud(positions=positions[perm], atomic_numbers=z[perm])
    assert _dative_column_for_pair(head, metal_first, 0, 1) == 1.0
    assert _dative_column_for_pair(head, metal_last, 2, 0) == 1.0


def test_carbon_metal_pair_is_not_dative_in_either_order() -> None:
    torch.manual_seed(0)
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=2)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.05, 0.0, 0.0]], dtype=torch.float32
    )
    z = torch.tensor([78, 6], dtype=torch.long)
    assert _dative_column_for_pair(
        head, AtomCloud(positions=positions, atomic_numbers=z), 0, 1
    ) == 0.0
    perm = torch.tensor([1, 0], dtype=torch.long)
    assert _dative_column_for_pair(
        head,
        AtomCloud(positions=positions[perm], atomic_numbers=z[perm]),
        0,
        1,
    ) == 0.0


# ---------------------------------------------------------------------------
# featurise uses the head's own metal / donor sets
# ---------------------------------------------------------------------------
def test_featurise_marks_a_non_pt_metal_donor_pair_dative() -> None:
    assert 15 in _DONOR_ZS and 34 in _DONOR_ZS
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=1)
    decoder = YuelBondDecoder(head=head)
    cloud = _two_atom_cloud(44, 15, 2.2)  # Ru-P
    pf = decoder.featurise(cloud).pair_features
    assert pf is not None
    assert bool(pf.is_dative_candidate.any())


def test_featurise_does_not_mark_a_carbon_metal_pair_dative() -> None:
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=1)
    decoder = YuelBondDecoder(head=head)
    pf = decoder.featurise(_two_atom_cloud(44, 6, 2.2)).pair_features
    assert pf is not None
    assert not bool(pf.is_dative_candidate.any())

def test_halide_donors_are_dative_on_the_decode_path() -> None:
    """Wf-A94 — halides are donors, matching the cohort builder's labels.

    MEASURED on 400 real Pt tmQM clouds with Wiberg-BO truth: 332 of
    1,701 true Pt bonds (19.5%) are Pt-Cl/Br/I/F, and the pre-Wf-A94 donor
    set flagged 0 of them.  On the canonical 5-metal MetalCytoTox cohort
    the extension is bit-exact (0 of 1,894 builder-written metal bonds have
    a halide partner).
    """
    for z_halide in (9, 17, 35, 53):
        assert z_halide in _DONOR_ZS
        head = YuelBondDecoderHead(hidden_dim=16, n_layers=1)
        decoder = YuelBondDecoder(head=head)
        pf = decoder.featurise(_two_atom_cloud(78, z_halide, 2.2)).pair_features
        assert pf is not None
        assert bool(pf.is_dative_candidate.any()), z_halide


def test_decoder_dative_bound_matches_the_calibrated_default() -> None:
    """Wf-A94 — 2.6 A, the same cutoff the cohort builder labels with.

    Swept against tmQM BO truth: 2.6 A is the best-F1 point (0.9888) and
    the value every config already declares under
    ``flow.dative_distance_cutoff``; the decoder used to hardcode 2.5 and
    never read that key.
    """
    dec = YuelBondBondDecoder(hidden_dim=16, n_layers=1)
    assert dec.dative_max_distance == 2.6
    near = _two_atom_cloud(78, 7, 2.55)
    far = _two_atom_cloud(78, 7, 2.75)
    assert dec._is_dative_pair_at(near, 0, 1, 78, 7, None)
    assert not dec._is_dative_pair_at(far, 0, 1, 78, 7, None)


# ---------------------------------------------------------------------------
# The selector's rule: truth first, then Z + distance
# ---------------------------------------------------------------------------
def test_selector_prefers_truth_over_the_z_rule() -> None:
    decoder = YuelBondBondDecoder(hidden_dim=16, n_layers=1)
    covalent = _two_atom_cloud(78, 7, 2.0, truth=False)
    assert not decoder._is_dative_pair_at(
        covalent, 0, 1, 78, 7, _dative_truth_pairs(covalent)
    )
    dative = _two_atom_cloud(78, 7, 2.0, truth=True)
    assert decoder._is_dative_pair_at(
        dative, 0, 1, 78, 7, _dative_truth_pairs(dative)
    )


def test_selector_z_rule_has_a_distance_bound() -> None:
    decoder = YuelBondBondDecoder(hidden_dim=16, n_layers=1)
    near = _two_atom_cloud(78, 7, 2.0)
    far = _two_atom_cloud(78, 7, 3.0)
    assert decoder._is_dative_pair_at(near, 0, 1, 78, 7, None)
    assert not decoder._is_dative_pair_at(far, 0, 1, 78, 7, None)


# ---------------------------------------------------------------------------
# End to end: the decode says how it knew
# ---------------------------------------------------------------------------
def test_decode_labels_a_generated_cloud_as_z_rule() -> None:
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=16, n_layers=1)
    _biased_to_single(decoder)
    decoded = decoder.decode(_two_atom_cloud(78, 7, 2.0)).decoded
    assert decoded.dative_bonds == [(1, 0)]
    assert decoded.dative_source == "z_rule"
    assert decoded.dative_manufactured == 0


def test_decode_does_not_manufacture_a_dative_bond_truth_calls_covalent() -> None:
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=16, n_layers=1)
    _biased_to_single(decoder)
    decoded = decoder.decode(
        _two_atom_cloud(78, 7, 2.0, truth=False)
    ).decoded
    assert decoded.dative_bonds == []
    assert decoded.dative_source == ""
    # The covalent bond itself is still decoded.
    assert decoded.bond_orders == [(0, 1, BOND_SINGLE)]


def test_decode_reports_truth_backed_dative_bonds() -> None:
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=16, n_layers=1)
    _biased_to_single(decoder)
    decoded = decoder.decode(
        _two_atom_cloud(78, 7, 2.0, truth=True)
    ).decoded
    assert decoded.dative_bonds == [(1, 0)]
    assert decoded.dative_source == "truth"
    assert decoded.dative_manufactured == 0


# ---------------------------------------------------------------------------
# Trainer plumbing: decode_flags is NOT a failure bucket
# ---------------------------------------------------------------------------
def _trainer():
    import importlib.util

    name = "_wfa92_trainer"
    if name in sys.modules:
        return sys.modules[name]
    path = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FlagDecoded:
    smiles = "C"
    sanitized = True
    error = ""
    repair = ""
    teacher_forced = True
    dative_source = "z_rule"
    dative_manufactured = 2


class _FlagResult:
    decoded = _FlagDecoded()


class _FlagDecoder:
    def decode(self, cloud):
        return _FlagResult()


def test_decode_flags_are_recorded_and_do_not_enter_the_failure_taxonomy() -> None:
    t = _trainer()
    cloud = AtomCloud(
        positions=torch.zeros(1, 3), atomic_numbers=torch.tensor([6])
    )
    flags: dict = {}
    reasons: dict = {}
    t.decode_clouds_and_classify(
        _FlagDecoder(),
        [cloud],
        None,
        None,
        None,
        [],
        None,
        reasons,
        decode_flags_out=flags,
    )
    assert flags["teacher_forced"] == 1
    assert flags["dative_source:z_rule"] == 1
    assert flags["dative_manufactured_bonds"] == 2
    # The failure taxonomy is unchanged: one cloud, one bucket, no flags.
    assert sum(reasons.values()) == 1
    assert "teacher_forced" not in reasons
    assert "dative_manufactured" not in reasons
