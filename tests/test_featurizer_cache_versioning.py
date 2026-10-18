"""The on-disk D-MPNN featurization cache must be versioned by the featurizer.

Regression: the cache was keyed on (metal, seed) and validated only by comparing
the SMILES list, so a change to the atom/bond featurization silently reused
stale graphs.  Every downstream number would then be uninterpretable, and no
error would be raised.
"""

from __future__ import annotations

import pickle

import pytest

from molmetal.scripts import train_dmpnn_pic50_real as T


@pytest.fixture()
def counting_featurizer(monkeypatch):
    calls = {"n": 0}

    def fake(smiles):
        calls["n"] += 1
        return [f"graph::{s}" for s in smiles]

    monkeypatch.setattr(T, "featurize_smiles_list", fake)
    return calls


def test_same_schema_hits_the_cache(tmp_path, counting_featurizer):
    smiles = ["CCO", "c1ccccc1"]
    a = T._featurize_with_cache(smiles, "arm", tmp_path)
    b = T._featurize_with_cache(smiles, "arm", tmp_path)
    assert a == b
    assert counting_featurizer["n"] == 1


def test_changed_schema_misses_the_cache(tmp_path, counting_featurizer, monkeypatch):
    smiles = ["CCO"]
    T._featurize_with_cache(smiles, "arm", tmp_path)
    monkeypatch.setattr(T, "FEATURIZER_SCHEMA_VERSION", "dmpnn-graph-v2")
    T._featurize_with_cache(smiles, "arm", tmp_path)
    assert counting_featurizer["n"] == 2
    # both versions coexist: the old cache is not deleted, just not consulted
    assert len(list(tmp_path.glob("arm_*.pkl"))) == 2


def test_legacy_tuple_cache_is_not_trusted(tmp_path, counting_featurizer):
    smiles = ["CCO"]
    fingerprint = T._featurizer_fingerprint()
    legacy = tmp_path / f"arm_{fingerprint}.pkl"
    legacy.write_bytes(pickle.dumps((smiles, ["stale::CCO"])))
    out = T._featurize_with_cache(smiles, "arm", tmp_path)
    assert out == ["graph::CCO"]
    assert counting_featurizer["n"] == 1


def test_payload_fingerprint_mismatch_is_not_trusted(tmp_path, counting_featurizer):
    smiles = ["CCO"]
    fingerprint = T._featurizer_fingerprint()
    path = tmp_path / f"arm_{fingerprint}.pkl"
    path.write_bytes(
        pickle.dumps({"smiles": smiles, "graphs": ["stale::CCO"], "fingerprint": "deadbeef"})
    )
    out = T._featurize_with_cache(smiles, "arm", tmp_path)
    assert out == ["graph::CCO"]


def test_different_smiles_list_rebuilds(tmp_path, counting_featurizer):
    T._featurize_with_cache(["CCO"], "arm", tmp_path)
    T._featurize_with_cache(["CCC"], "arm", tmp_path)
    assert counting_featurizer["n"] == 2


def test_fingerprint_is_stable_across_calls():
    assert T._featurizer_fingerprint() == T._featurizer_fingerprint()
    assert len(T._featurizer_fingerprint()) == 12
