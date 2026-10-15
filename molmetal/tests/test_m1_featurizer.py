"""M1 featurizer regression: the v2 atom-feature block must extend v1, not
perturb it.

Guarantees relied on elsewhere:
  * v1 is exactly 39-dim (the metal-node slots 30-38 in
    ``molmetal.data.metal_graph`` stay free for real atoms).
  * v2 keeps dims 0-38 byte-identical to v1 and appends the M1 block at 39+.
  * v2 is ``ATOM_FEATURE_DIM_V2`` wide.
"""
import sys

import numpy as np
from rdkit import Chem

sys.path.insert(0, "/home/hugo/codes/mol-metal")

from molmetal.baselines.dmpnn import (  # noqa: E402
    ATOM_FEATURE_DIM,
    ATOM_FEATURE_DIM_V2,
    _LEGACY_ATOM_FEATURE_DIM,
    _V2_EXTRA_ELEMENTS,
    _atom_features,
    mol_to_graph,
)

_SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO", "O=[N+]([O-])c1ccccc1"]


def _atoms():
    for smi in _SMILES:
        m = Chem.MolFromSmiles(smi)
        for a in m.GetAtoms():
            yield a


def test_v1_dim_and_metal_slots_free():
    assert ATOM_FEATURE_DIM == 39
    assert _LEGACY_ATOM_FEATURE_DIM == 39
    for a in _atoms():
        f = _atom_features(a, version=1)
        assert f.shape == (39,)
        # slots 30-38 are reserved for the metal node and must be 0 for real atoms
        assert np.allclose(f[30:39], 0.0), f[30:39]


def test_v2_extends_v1_unchanged_prefix():
    assert ATOM_FEATURE_DIM_V2 > ATOM_FEATURE_DIM
    for a in _atoms():
        f1 = _atom_features(a, version=1)
        f2 = _atom_features(a, version=2)
        assert f2.shape == (ATOM_FEATURE_DIM_V2,)
        # prefix 0-38 must be byte-identical
        assert np.allclose(f1[:39], f2[:39], rtol=0.0, atol=0.0)
        # extension block is finite and non-trivial (some bit set for a real atom)
        assert np.isfinite(f2[39:]).all()
        assert f2[39:].sum() > 0


def test_v2_extra_element_bits_engage():
    """Ru/Os/Re must set their v2 element bits (the metals the cohorts carry)."""
    for sym, z in (("Ru", 44), ("Os", 76), ("Re", 75), ("H", 1), ("Br", 35)):
        a = Chem.MolFromSmiles(f"[{sym}]").GetAtomWithIdx(0)
        f2 = _atom_features(a, version=2)
        # element one-hot is the first len(_V2_EXTRA_ELEMENTS)+1 block after 39
        block = f2[39:40 + len(_V2_EXTRA_ELEMENTS)]
        assert block.sum() == 1.0, f"{sym} element bit not set"


def test_mol_to_graph_dim_follows_version():
    m = Chem.MolFromSmiles("CCO")
    af1, _, _, _ = mol_to_graph(m, version=1)
    af2, _, _, _ = mol_to_graph(m, version=2)
    assert af1.shape[1] == ATOM_FEATURE_DIM
    assert af2.shape[1] == ATOM_FEATURE_DIM_V2


def test_u6_both_featurizers_agree():
    """U6: the in-house and attentive D-MPNN featurizers must not fork.

    ``molmetal.baselines.dmpnn_attentive`` delegates to the canonical layout;
    this asserts byte-identical graphs for both versions.
    """
    from molmetal.baselines.dmpnn import featurize_smiles_list as f_canon
    from molmetal.baselines.dmpnn_attentive import featurize_smiles_list as f_attn

    for version in (1, 2):
        g_canon = f_canon(_SMILES, version=version)
        g_attn = f_attn(_SMILES, version=version)
        for a, b in zip(g_canon, g_attn):
            assert a is not None and b is not None
            for x, y in zip(a, b):
                assert np.array_equal(x, y), f"featurizer fork at version={version}"
        assert g_canon[0][0].shape[1] == (
            ATOM_FEATURE_DIM_V2 if version == 2 else ATOM_FEATURE_DIM
        )
