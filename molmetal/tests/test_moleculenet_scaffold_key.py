"""Pin the MoleculeNet scaffold key to the astartes/chemprop reference.

Wf-A53 (2026-10-19) — ``train_moleculenet._bemis_murcko_scaffold`` used
isomeric SMILES, so two stereoisomers of the same Bemis-Murcko scaffold got
different split keys.  The reference splitter chemprop moved to
(``astartes.samplers.extrapolation.scaffold``) calls
``MurckoScaffoldSmiles(includeChirality=False)``.  MEASURED consequence of
the mismatch: 39-87% of the test molecules for bbbp/bace/clintox were
assigned differently (metrics/by_metric/moleculenet_scaffold_key_alignment.json).

These tests fail if the key is ever reverted to the isomeric form.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_trainer():
    """Import ``train_moleculenet`` without executing its CLI."""
    spec = importlib.util.spec_from_file_location(
        "train_moleculenet_under_test",
        PROJECT_ROOT / "molmetal" / "scripts" / "train_moleculenet.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


@pytest.fixture(scope="module")
def murcko():
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return MurckoScaffold


# Two stereoisomers of decalin whose chiral centres sit IN the ring system.
# (1-phenylethanol would not do: its stereocentre is exocyclic and the
# Bemis-Murcko scaffold drops it, so both enantiomers key to benzene.)
ENANTIOMER_PAIR = ("C1CC[C@@H]2CCCC[C@H]2C1", "C1CC[C@@H]2CCCC[C@@H]2C1")
STEREO_BLIND_SCAFFOLD = "C1CCC2CCCCC2C1"


def test_key_matches_reference_call(trainer, murcko):
    """Bit-exact agreement with ``MurckoScaffoldSmiles(includeChirality=False)``."""
    smiles = [
        "CCO",
        "c1ccccc1",
        "C[C@@H](O)c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    ]
    for smi in smiles:
        expected = murcko.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
        assert trainer._bemis_murcko_scaffold(smi) == expected, smi


def test_enantiomers_share_a_scaffold_key(trainer):
    """The pre-fix key separated these; the reference groups them."""
    a, b = ENANTIOMER_PAIR
    ka, kb = trainer._bemis_murcko_scaffold(a), trainer._bemis_murcko_scaffold(b)
    assert ka == kb == STEREO_BLIND_SCAFFOLD


def test_isomeric_key_would_have_split_them(trainer):
    """Guard the direction of the fix: the old key really did differ."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    a, b = ENANTIOMER_PAIR
    old = [
        Chem.MolToSmiles(
            MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)), canonical=True
        )
        for s in (a, b)
    ]
    assert old[0] != old[1], (
        "If this fails RDKit changed behaviour and the premise of the fix "
        "needs re-checking."
    )


def test_achiral_scaffold_unchanged(trainer):
    """Achiral molecules are keyed identically under both forms."""
    assert trainer._bemis_murcko_scaffold("c1ccccc1") == "c1ccccc1"


def test_invalid_smiles_falls_back_to_itself(trainer):
    bad = "this-is-not-a-smiles"
    assert trainer._bemis_murcko_scaffold(bad) == bad


def test_split_groups_stereoisomers_together(trainer):
    """No scaffold group may be split across train/test because of stereo."""
    a, b = ENANTIOMER_PAIR
    # 20 copies of each enantiomer; a stereo-blind key yields ONE group.
    smiles = [a] * 20 + [b] * 20
    train_idx, val_idx, test_idx = trainer._scaffold_split(smiles, seed=0)
    in_train = set(train_idx.tolist())
    in_test = set(test_idx.tolist())
    assert not (in_train & in_test)
    # Both halves of the enantiomer pair must land on the same side.
    a_side = 0 in in_train
    b_side = 20 in in_train
    assert a_side == b_side
