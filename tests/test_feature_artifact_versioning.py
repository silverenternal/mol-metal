"""Tests for FIX-P3 artifact versioning (sidecar + tamper detection)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from molmetal.features import LEGACY_MAIN, build
from molmetal.features.build import verify_artifact, write_artifact
from molmetal.features.cell import cell_vocabulary
from molmetal.features.metal import load_descriptors

FRAME = pd.DataFrame(
    {
        "ligand_canonical": ["[Ru+2].c1ccncc1.c1ccncc1", "c1ccncc1"],
        "ligand_main_fragment": ["c1ccncc1", "c1ccncc1"],
        "metal": ["Ru", "Ru"],
        "cell_line_key": ["A549", "A549"],
    }
)


@pytest.fixture(scope="module")
def bundle():
    vocab = cell_vocabulary(FRAME["cell_line_key"].tolist(), top_n=40)
    return build(
        FRAME,
        LEGACY_MAIN,
        metal_descriptors=load_descriptors(),
        cell_vocabulary_override=vocab,
    )


def _write(tmp_path, bundle, **kwargs):
    return write_artifact(
        tmp_path / "table.npz",
        {"X": np.arange(6, dtype=np.float32).reshape(2, 3)},
        artifact="test_table",
        bundle=bundle,
        **kwargs,
    )


def test_sidecar_records_spec_and_checksum(tmp_path, bundle):
    sidecar = _write(tmp_path, bundle)
    meta = json.loads(sidecar.read_text())
    assert meta["spec_hash"] == LEGACY_MAIN.hash()
    assert meta["n_rows"] == 2
    assert meta["missing_value_policy"] == "nan"
    assert meta["meta_checksum"]
    assert meta["arrays"]["X"]["shape"] == [2, 3]


def test_verify_returns_metadata(tmp_path, bundle):
    _write(tmp_path, bundle)
    meta = verify_artifact(tmp_path / "table.npz", expected_spec=LEGACY_MAIN)
    assert meta["artifact"] == "test_table"


def test_edited_sidecar_is_rejected(tmp_path, bundle):
    sidecar = _write(tmp_path, bundle)
    meta = json.loads(sidecar.read_text())
    meta["n_rows"] = 999
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_artifact(tmp_path / "table.npz")


def test_modified_npz_is_rejected(tmp_path, bundle):
    _write(tmp_path, bundle)
    np.savez_compressed(
        tmp_path / "table.npz", X=np.zeros((2, 3), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="npz hash mismatch"):
        verify_artifact(tmp_path / "table.npz")


def test_wrong_expected_spec_is_rejected(tmp_path, bundle):
    _write(tmp_path, bundle)
    from molmetal.features import COMPLEX

    with pytest.raises(ValueError, match="spec hash"):
        verify_artifact(tmp_path / "table.npz", expected_spec=COMPLEX)


def test_missing_sidecar_is_rejected(tmp_path, bundle):
    _write(tmp_path, bundle)
    (tmp_path / "table.meta.json").unlink()
    with pytest.raises(ValueError, match="sidecar missing"):
        verify_artifact(tmp_path / "table.npz")


def test_input_hashes_are_recorded(tmp_path, bundle):
    data = tmp_path / "input.csv"
    data.write_text("a,b\n1,2\n")
    _write(tmp_path, bundle, input_paths=[data])
    meta = verify_artifact(tmp_path / "table.npz")
    assert meta["inputs"][0]["sha256"]
    assert meta["inputs"][0]["path"].endswith("input.csv")
