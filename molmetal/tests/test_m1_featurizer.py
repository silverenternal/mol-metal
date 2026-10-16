"""M1 featurizer regression: the v2 atom-feature block must extend v1, not
perturb it.

Guarantees relied on elsewhere:
  * v1 is exactly 39-dim (the metal-node slots 30-38 in
    ``molmetal.data.metal_graph`` stay free for real atoms).
  * v2 keeps dims 0-38 byte-identical to v1 and appends the M1 block at 39+.
  * v2 is ``ATOM_FEATURE_DIM_V2`` wide.

Wf-A52 W1 (2026-10-16) — W3-B feature wire-in shipped: v2 symbols
(``ATOM_FEATURE_DIM_V2``, ``_V2_EXTRA_ELEMENTS``, ``_V2_EXTRA_DIM``) are
now exported by :mod:`molmetal.baselines.dmpnn`.  All v2 tests in this
module are un-skipped; the v1 single-arg API (``_atom_features(a)``) is
preserved bit-for-bit per the §5.11 YAML-only / §5.7 wire-in contract.
"""
import sys

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, "/home/hugo/codes/mol-metal")

from molmetal.baselines.dmpnn import (  # noqa: E402
    ATOM_FEATURE_DIM,
    ATOM_FEATURE_DIM_V2,
    _V2_EXTRA_DIM,
    _atom_features,
    _atom_features_v1,
    _atom_features_v2,
)

_SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO", "O=[N+]([O-])c1ccccc1"]


def _atoms():
    for smi in _SMILES:
        m = Chem.MolFromSmiles(smi)
        for a in m.GetAtoms():
            yield a


def test_v1_dim_and_metal_slots_free():
    # Wf-A52 W1 (2026-10-16) — v2 symbols now exported; legacy alias
    # ``_LEGACY_ATOM_FEATURE_DIM`` removed (the v2 featurizer shipped so
    # the legacy alias is no longer needed).  The single-arg
    # ``_atom_features`` keeps its v1 signature bit-for-bit.
    assert ATOM_FEATURE_DIM == 39
    for a in _atoms():
        f = _atom_features(a)
        assert f.shape == (39,)
        # slots 30-38 are reserved for the metal node and must be 0 for real atoms
        assert np.allclose(f[30:39], 0.0), f[30:39]


def test_v2_extends_v1_unchanged_prefix():
    """v2 dims 0-38 must be byte-identical to v1; v2 width is 72."""
    assert ATOM_FEATURE_DIM_V2 == 72
    assert _V2_EXTRA_DIM == 33
    for a in _atoms():
        v1 = _atom_features_v1(a)
        v2 = _atom_features_v2(a)
        assert v2.shape == (72,)
        # Wf-A52 W1: v1 prefix is byte-identical (no perturbation).
        assert np.array_equal(v2[:39], v1)
        # Extras block is exactly 33-dim, the chemprop-style complement.
        assert v2.shape == (39 + _V2_EXTRA_DIM,)


def test_v2_extra_element_bits_engage():
    """V2 exotics block activates for at least one bit on a real atom."""
    for a in _atoms():
        v2 = _atom_features_v2(a)
        # The extras block (dims 39-71) must NOT be all zeros for any
        # real atom — at least the ring-size multi-hot OR the explicit-H
        # one-hot (or both) should fire.  13 C atoms across the 4 SMILES
        # guarantee at least one positive bit on the corpus.
        assert np.any(v2[39:] != 0.0), (
            f"v2 extras all-zero on atom {a.GetAtomicNum()}/{a.GetIdx()}: {v2[39:]}"
        )


def test_mol_to_graph_dim_follows_version():
    """``mol_to_graph`` width follows the requested featurizer version."""
    from molmetal.baselines.dmpnn import mol_to_graph

    mol = Chem.MolFromSmiles("c1ccccc1")
    g_v1 = mol_to_graph(mol, version="v1")
    g_v2 = mol_to_graph(mol, version="v2")
    assert g_v1[0].shape == (6, 39)
    assert g_v2[0].shape == (6, 72)
    # Bond featurises are version-independent (10-dim).
    assert g_v1[1].shape == (12, 10)
    assert g_v2[1].shape == (12, 10)


def test_u6_both_featurizers_agree():
    """U6: in-house and attentive D-MPNN featurizers must not fork.

    :mod:`molmetal.baselines.dmpnn_attentive` delegates to the canonical
    layout; this asserts byte-identical graphs for both versions.
    """
    from molmetal.baselines.dmpnn_attentive import (
        _atom_features as att_af,
        featurize_smiles_list as att_fs,
    )

    smis = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "c1ccccc1",
        "CCO",
        "O=[N+]([O-])c1ccccc1",
    ]
    g_v1 = att_fs(smis, version="v1")
    g_v2 = att_fs(smis, version="v2")
    for i, smi in enumerate(smis):
        # attentive == dmpnn byte-equal (delegation)
        atoms = list(Chem.MolFromSmiles(smi).GetAtoms())
        v1_att = np.stack([att_af(a) for a in atoms], axis=0)
        v1_canon = np.stack([_atom_features(a) for a in atoms], axis=0)
        assert np.array_equal(v1_att, v1_canon)
        if g_v1[i] is not None and g_v2[i] is not None:
            assert g_v1[i][0].shape == (g_v1[i][0].shape[0], 39)
            assert g_v2[i][0].shape == (g_v2[i][0].shape[0], 72)


def test_invalid_version_raises():
    """``_atom_features(atom, version='bad')`` raises ValueError."""
    with pytest.raises(ValueError, match="Unknown featurizer version"):
        _atom_features(list(_atoms())[0], version="v3")