"""Tests for the single feature construction site (FIX-P1).

The point of these tests is *bit-exactness*: the new builder must reproduce each
historical matrix exactly, or the consolidation has quietly changed a result
that other artifacts are compared against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.features import (
    COMPLEX,
    LEGACY_MAIN,
    CellSpec,
    FeatureSpec,
    LigandSpec,
    MetalSpec,
    build,
    verify_bundle,
)
from molmetal.features.cell import cell_block, cell_vocabulary, native_codes
from molmetal.features.metal import load_descriptors, metal_block

SMILES = [
    "[Ru+2].[Cl-].[Cl-].c1ccc2c(c1)nc1ccccc12",  # 3 components
    "[Ir+3].c1ccncc1.c1ccncc1.c1ccncc1",  # main + 3 ancillary
    "c1ccc2c(c1)nc1ccccc12",  # single component
    "[Os+2].c1ccc(cc1)P(c1ccccc1)c1ccccc1",  # 2 components
    "[Re+1].c1ccncc1",  # 2 components, one tiny
]
MAIN = ["c1ccc2c(c1)nc1ccccc12", "c1ccncc1", "c1ccc2c(c1)nc1ccccc12", "c1ccc(cc1)P(c1ccccc1)c1ccccc1", "c1ccncc1"]
METALS = ["Ru", "Ir", "Os", "Os", "Re"]
CELLS = ["A549", "A549", "HepG2", "A549", "HeLa"]
COUNTERIONS = ["Cl-", "", "", "", ""]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand_canonical": SMILES,
            "ligand_main_fragment": MAIN,
            "metal": METALS,
            "cell_line_key": CELLS,
            "counterion": COUNTERIONS,
        }
    )


@pytest.fixture(scope="module")
def descriptors() -> pd.DataFrame:
    return load_descriptors()


def test_legacy_preset_reproduces_base_blocks(frame, descriptors):
    from molmetal.scripts.msrs_code_levers import base_blocks

    vocab = cell_vocabulary(frame["cell_line_key"].tolist(), top_n=40)
    lig_old, met_old, cell_old = base_blocks(frame, descriptors, vocab)

    bundle = build(
        frame,
        LEGACY_MAIN,
        metal_descriptors=descriptors,
        cell_vocabulary_override=vocab,
    )
    verify_bundle(bundle)

    assert (bundle.block("ligand") != lig_old).nnz == 0
    assert np.array_equal(bundle.block("metal").toarray(), met_old.toarray())
    assert (bundle.block("cell") != cell_old).nnz == 0
    assert bundle.X.shape[1] == lig_old.shape[1] + met_old.shape[1] + cell_old.shape[1]


def test_legacy_preset_reproduces_surface_model_design(frame, descriptors):
    from molmetal.scripts.msrs_surface_model import build_design

    vocab = cell_vocabulary(frame["cell_line_key"].tolist(), top_n=40)
    X_old, names_old, _groups = build_design(frame, descriptors, vocab)

    bundle = build(
        frame,
        LEGACY_MAIN,
        metal_descriptors=descriptors,
        cell_vocabulary_override=vocab,
    )
    assert np.array_equal(bundle.X.toarray(), X_old.toarray())
    assert bundle.names == names_old


def test_complex_preset_featurizes_the_whole_complex(frame, descriptors):
    vocab = cell_vocabulary(frame["cell_line_key"].tolist(), top_n=40)
    legacy = build(
        frame,
        LEGACY_MAIN,
        metal_descriptors=descriptors,
        cell_vocabulary_override=vocab,
    )
    complex_bundle = build(
        frame,
        COMPLEX,
        metal_descriptors=descriptors,
        cell_vocabulary_override=vocab,
    )
    # Same width, different content: the complex has bits the main fragment
    # cannot have (counterions / ancillary ligands).
    assert legacy.X.shape == complex_bundle.X.shape
    assert (legacy.block("ligand") != complex_bundle.block("ligand")).nnz > 0


def test_172d_table_is_unchanged(frame):
    from molmetal.features.descriptors import descriptor_table
    from molmetal.scripts.msrs_172d_rebuild import build_table

    frags = list(dict.fromkeys(MAIN))
    X_new, masks_new = descriptor_table(frags)
    X_old, masks_old = build_table(frags)
    assert np.array_equal(X_new, X_old, equal_nan=True)
    assert all(np.array_equal(masks_new[k], masks_old[k]) for k in masks_old)


def test_descriptor_table_attaches_to_the_declared_source(frame, descriptors):
    spec = FeatureSpec(
        ligand=LigandSpec(
            source="ligand_main_fragment",
            include=("main",),
            descriptor_table="rdkit_172d",
        )
    )
    vocab = cell_vocabulary(frame["cell_line_key"].tolist(), top_n=40)
    bundle = build(
        frame,
        spec,
        metal_descriptors=descriptors,
        cell_vocabulary_override=vocab,
    )
    verify_bundle(bundle)
    start, stop = bundle.slices["descriptors"]
    assert stop - start == 172


def test_spec_hash_is_stable_and_content_addressed():
    a = FeatureSpec()
    b = FeatureSpec()
    assert a.hash() == b.hash()
    assert a.hash().startswith("sha256:")

    changed = FeatureSpec(ligand=a.ligand.__class__(n_bits=1024))
    assert changed.hash() != a.hash()


def test_bundle_hash_matches_spec(frame, descriptors):
    vocab = cell_vocabulary(frame["cell_line_key"].tolist(), top_n=40)
    bundle = build(
        frame, LEGACY_MAIN, metal_descriptors=descriptors, cell_vocabulary_override=vocab
    )
    assert bundle.spec_hash == LEGACY_MAIN.hash()
    assert bundle.summary()["n_features"] == bundle.X.shape[1]


def test_unknown_metal_gets_nan_descriptors_not_zero(descriptors):
    block, names = metal_block(["Pt"], descriptors)
    assert block[0, :5].sum() == 0.0  # no one-hot column for Pt
    assert np.isnan(block[0, 5:]).all()


def test_cell_vocabulary_matches_historical_value_counts(frame):
    expected = frame["cell_line_key"].value_counts().head(40).index.tolist()
    assert cell_vocabulary(frame["cell_line_key"].tolist(), top_n=40) == expected


def test_native_codes_put_unseen_cells_in_other(frame):
    vocab = ["A549", "HepG2"]
    codes = native_codes(METALS, CELLS, vocab, mode="top40")
    other = len(vocab)
    assert codes[4, 1] == other  # HeLa is unseen
    assert codes[0, 1] == 0


def test_cell_block_shape_includes_other(frame):
    block, names = cell_block(CELLS, ["A549", "HepG2"])
    assert block.shape == (len(CELLS), 3)
    assert names[-1] == "cell_other"


def test_build_can_drop_metal_and_cell_blocks(frame):
    spec = FeatureSpec(
        ligand=LigandSpec(
            source="ligand_main_fragment", include=("main",), n_bits=16
        ),
        metal=MetalSpec(one_hot=False, columns=()),
        cell=CellSpec(mode="none"),
    )
    bundle = build(frame, spec, metal_descriptors=load_descriptors())
    assert bundle.X.shape[1] == 16
    assert set(bundle.slices) == {"ligand"}
