"""Tests for STR-4 feature blocks (no fitting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.scripts.msrs_feature_expansion import (
    RDKIT_DESCRIPTORS,
    assay_block,
    maccs_rdkit_block,
)


def test_assay_block_log1p_and_cancer_flag():
    frame = pd.DataFrame(
        {
            "time_h": [0.0, 24.0, np.nan, 168.0],
            "is_cancer_line": [True, False, True, "true"],
        }
    )
    block = assay_block(frame)
    assert block.shape == (4, 2)
    assert block[0, 0] == 0.0
    assert block[1, 0] == pytest.approx(np.log1p(24.0), rel=1e-6)
    assert block[2, 0] == -1.0  # missing time stays distinguishable
    assert list(block[:, 1]) == [1.0, 0.0, 1.0, 1.0]


def test_maccs_rdkit_block_is_dense_and_finite():
    frame = pd.DataFrame(
        {"ligand_canonical": ["c1ccccc1", "CCO", "not-a-smiles"]}
    )
    block = maccs_rdkit_block(frame)
    assert block.shape == (3, 167 + len(RDKIT_DESCRIPTORS))
    assert np.isfinite(block).all()
    # An unparseable SMILES becomes an all-zero row rather than NaN.
    assert np.all(block[2] == 0.0)


def test_maccs_rdkit_block_reuses_duplicate_smiles():
    frame = pd.DataFrame({"ligand_canonical": ["CCO", "CCO"]})
    block = maccs_rdkit_block(frame)
    assert np.array_equal(block[0], block[1])
