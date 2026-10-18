"""Contract tests for ``molmetal.features.ligand`` (FIX-L1 / DEF-1).

The block exists to stop featurizing only the main fragment of a
multi-component complex.  These tests pin the decomposition and the failure
behaviour, not the model lift (that is measured in
``metrics/by_metric/msrs_complex_features.json``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molmetal.features.ligand import (
    ancillary_smiles,
    complex_ligand_blocks,
    component_smiles,
    main_component,
)

# 2,2'-bipyridine: 12 heavy atoms, so it is larger than PF6 (7) and chloride (1).
MAIN = "c1ccc(-c2ccccn2)nc1"
BENZENE = "c1ccccc1"
PF6 = "F[P-](F)(F)(F)(F)F"
# A realistic multi-component row: main ligand + PF6 counterion + chloride.
COMPLEX = f"{MAIN}.{PF6}.[Cl-]"


def test_components_are_sorted_by_heavy_atoms():
    comps = component_smiles(COMPLEX)
    assert comps[0] == MAIN
    assert set(comps[1:]) == {"[Cl-]", PF6}


def test_main_component_is_the_largest_fragment():
    assert main_component(COMPLEX) == MAIN
    assert main_component(BENZENE) == BENZENE
    # PF6 has 7 heavy atoms and benzene 6, so the counterion wins on atom count.
    # That is the documented definition, not a bug: real ligands are larger.
    assert main_component(f"{BENZENE}.{PF6}") == PF6


def test_ancillary_is_everything_but_the_main_fragment():
    anc = ancillary_smiles(COMPLEX)
    assert anc is not None
    assert MAIN not in anc
    assert "[Cl-]" in anc
    # A single-component molecule has no ancillary part, and that is not an error.
    assert ancillary_smiles(BENZENE) == ""


def test_unparseable_smiles_has_no_components():
    assert component_smiles("not_a_smiles") == []
    assert main_component("not_a_smiles") is None
    assert ancillary_smiles("not_a_smiles") is None


def test_blocks_are_in_the_declared_order_with_slices():
    frame = pd.DataFrame(
        {"ligand_canonical": [COMPLEX], "counterion": [PF6]}
    )
    X, names, slices = complex_ligand_blocks(frame, n_bits=64, counterion_vocabulary=[PF6, "[Cl-]"])
    assert X.shape == (1, 64 * 3 + 3 + 1)
    assert list(slices) == ["canonical", "main", "ancillary", "counterion", "n_fragments"]
    assert slices["canonical"] == (0, 64)
    assert slices["main"] == (64, 128)
    assert slices["ancillary"] == (128, 192)
    assert slices["counterion"] == (192, 195)
    assert slices["n_fragments"] == (195, 196)
    assert names[195] == "n_components"
    assert X[0, 195] == 3.0


def test_include_subset_drops_blocks_but_keeps_the_order():
    frame = pd.DataFrame({"ligand_canonical": [COMPLEX]})
    X, names, slices = complex_ligand_blocks(
        frame, n_bits=32, include=("main", "ancillary")
    )
    assert X.shape == (1, 64)
    assert list(slices) == ["main", "ancillary"]
    assert names[0] == "ecfp4_main_0"
    assert names[32] == "ecfp4_ancillary_0"


def test_counterion_vocabulary_is_fixed_and_other_is_last():
    frame = pd.DataFrame(
        {"ligand_canonical": [COMPLEX, BENZENE], "counterion": [PF6, "unseen_salt"]}
    )
    X, names, slices = complex_ligand_blocks(
        frame, n_bits=8, include=("counterion",), counterion_vocabulary=[PF6]
    )
    start, stop = slices["counterion"]
    assert stop - start == 2
    assert names[-1] == "counterion_other"
    assert X[0, start] == 1.0 and X[0, start + 1] == 0.0
    assert X[1, start] == 0.0 and X[1, start + 1] == 1.0


def test_canonical_block_differs_from_main_block_for_a_complex():
    frame = pd.DataFrame({"ligand_canonical": [COMPLEX]})
    X, _names, slices = complex_ligand_blocks(frame, n_bits=256, include=("canonical", "main"))
    can = X[:, slice(*slices["canonical"])].toarray()
    main = X[:, slice(*slices["main"])].toarray()
    # The complex fingerprint must contain bits the main fragment cannot.
    assert (can.sum() > main.sum())


def test_unparseable_rows_are_zero_not_an_exception():
    frame = pd.DataFrame({"ligand_canonical": ["not_a_smiles"]})
    X, _names, _slices = complex_ligand_blocks(
        frame, n_bits=16, include=("canonical", "main", "ancillary")
    )
    assert sparse.issparse(X)
    assert X.nnz == 0


def test_missing_canonical_column_raises():
    with pytest.raises(KeyError):
        complex_ligand_blocks(pd.DataFrame({"ligand_main_fragment": ["CCO"]}))


def test_empty_include_raises():
    frame = pd.DataFrame({"ligand_canonical": [BENZENE]})
    with pytest.raises(ValueError):
        complex_ligand_blocks(frame, include=())
