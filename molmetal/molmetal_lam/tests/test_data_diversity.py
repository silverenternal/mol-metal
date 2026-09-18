"""Tests for :mod:`molmetal.molmetal_lam.lam_chem.data_diversity`.

Covers
------
* Morgan fingerprint generation (parseable + unparseable)
* Greedy MaxMin diversity selection (count, no duplicates, n correctness)
* Source loaders (graceful empty when path missing)
* Combined-pool builder (size, source coverage)
* CSV cache writer
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest


pytest.importorskip("rdkit", reason="RDKit required for data_diversity tests")

from rdkit import Chem  # noqa: E402

from molmetal.molmetal_lam.lam_chem import data_diversity as dd  # noqa: E402


# A small synthetic pool used across most tests.  These are intentionally
# diverse (aliphatic + aromatic + heterocycle + halide) so MaxMin has real
# work to do.
SMILES_POOL = [
    "CCO",                 # ethanol
    "CCN",                 # ethylamine
    "c1ccccc1",            # benzene
    "c1ccncc1",            # pyridine
    "c1ccoc1",             # furan
    "c1ccsc1",             # thiophene
    "CC(=O)O",             # acetic acid
    "CC(=O)N",             # acetamide
    "C(F)(F)F",            # trifluoromethyl
    "C(Cl)(Cl)Cl",         # chloroform
    "CCS",                 # ethanethiol
    "CC(C)C",              # isobutane
    "CCCCCCCC",            # octane
    "c1ccc2ccccc2c1",      # naphthalene
    "C1CCCCC1",            # cyclohexane
    "OC(=O)c1ccccc1",      # benzoic acid
    "c1ccc(O)cc1",         # phenol
    "NCCN",                # ethylenediamine
    "OCCO",                # ethylene glycol
    "OCC(O)CO",            # glycerol
    "C(F)(F)(F)C(F)(F)F",  # hexafluoroethane
    "CCBr",                # bromoethane
    "CCI",                 # iodoethane
    "[Pt](Cl)(Cl)(N)N",    # cisplatin-like Pt complex
    "[Pd]Cl2",             # Pd dichloride
    "c1ccc(N)cc1",         # aniline
    "C1=CC=NC=C1",         # pyridine (alt)
    "CC(=O)c1ccccc1",      # acetophenone
    "COC",                 # dimethyl ether
    "CCOCC",               # diethyl ether
    "N#N",                 # dinitrogen
    "OO",                  # peroxide
    "CC#C",                # propyne
    "C=CC=C",              # butadiene
    "c1cnc2ccccc2c1",      # quinoline
    "c1ncc2ncn(C)c2n1",    # purine-like
    "[Au](Cl)Cl",          # Au dichloride
    "CCOP(=O)(OCC)OCC",    # phosphate triester
    "CS",                  # methanethiol
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cl",                  # HCl
    "BrCBr",               # dibromomethane
    "ICCI",                # 1,2-diiodoethane
    "C(F)(F)(F)c1ccccc1",  # trifluoromethylbenzene
    "c1ccc(S)cc1",         # thiophenol
    "c1ccc(P)cc1",         # phenylphosphine
    "CCCCC",               # pentane
    "CCCC",                # butane
    "CCC",                 # propane
    "C",                   # methane
]


def test_morgan_fp_parses_valid_smiles():
    fp = dd.morgan_fp("CCO", radius=2, n_bits=1024)
    assert fp is not None
    assert fp.dtype.name == "uint8"
    assert fp.shape == (1024,)
    # ethanol has at least one fingerprint bit set
    assert fp.sum() > 0


def test_morgan_fp_returns_none_for_unparseable():
    assert dd.morgan_fp("not-a-smiles@@@") is None
    assert dd.morgan_fp("") is None


def test_morgan_fps_drops_unparseable_and_returns_indices():
    smis = ["CCO", "garbage@@", "c1ccccc1", None, ""]
    fps, parsed = dd.morgan_fps(smis, n_bits=512)
    assert fps.shape == (2, 512)
    assert parsed == [0, 2]


def test_greedy_maxmin_returns_n_unique_smiles():
    selected_idx, selected_smi = dd.greedy_maxmin_diversity(
        SMILES_POOL, n=10, radius=2, n_bits=1024, seed=42
    )
    assert len(selected_idx) == 10
    assert len(selected_smi) == 10
    # No duplicates by canonical SMILES.
    assert len(set(selected_smi)) == 10


def test_greedy_maxmin_handles_smaller_pool_than_n():
    tiny = SMILES_POOL[:3]
    selected_idx, selected_smi = dd.greedy_maxmin_diversity(tiny, n=10)
    # Only 3 parseable — should return those 3.
    assert len(selected_idx) == 3
    assert len(selected_smi) == 3


def test_greedy_maxmin_is_deterministic():
    a, _ = dd.greedy_maxmin_diversity(SMILES_POOL, n=8, seed=123)
    b, _ = dd.greedy_maxmin_diversity(SMILES_POOL, n=8, seed=123)
    assert a == b


def test_greedy_maxmin_maximises_min_distance():
    # Compare the MaxMin-selected set vs a random-size-matched subset;
    # the MaxMin set should have a strictly larger minimum pairwise
    # distance (the criterion it optimises).
    import random
    selected_idx, _ = dd.greedy_maxmin_diversity(SMILES_POOL, n=10, seed=7)
    fps, parsed = dd.morgan_fps(SMILES_POOL, n_bits=1024)
    # Map selected_idx (indexes into SMILES_POOL) -> index into parsed/fps.
    pos = {orig: i for i, orig in enumerate(parsed)}
    selected_fps = fps[[pos[i] for i in selected_idx]]
    # Pairwise Tanimoto distance
    inter = selected_fps.astype("float32") @ selected_fps.T.astype("float32")
    pops = selected_fps.sum(axis=1)
    union = pops[:, None] + pops[None, :] - inter
    sim = np_fill(inter / np_max(union, 1e-9))
    np_d = 1.0 - sim
    np_min_dist = np_d[np_d > 0].min()

    # Random baseline (3 trials, take best).
    random.seed(7)
    rng_indices = list(range(len(parsed)))
    best = -1.0
    for _ in range(3):
        random.shuffle(rng_indices)
        samp = rng_indices[:10]
        fpf = fps[samp]
        inter = fpf.astype("float32") @ fpf.T.astype("float32")
        pops = fpf.sum(axis=1)
        union = pops[:, None] + pops[None, :] - inter
        sim = inter / np_max(union, 1e-9)
        d = 1.0 - sim
        best = max(best, d[d > 0].min())
    assert np_min_dist >= best - 1e-6


def test_load_platinai_smiles_missing_returns_empty(tmp_path):
    assert dd.load_platinai_smiles(tmp_path / "nope.xlsx") == []


def test_load_metal_cytotox_smiles_missing_returns_empty(tmp_path):
    assert dd.load_metal_cytotox_smiles(tmp_path / "nope.csv") == []


def test_load_tmqm_smiles_missing_returns_empty(tmp_path):
    assert dd.load_tmqm_smiles(tmp_path / "nope_dir") == []


def test_build_combined_pool_graceful_when_files_missing(tmp_path):
    smiles, sources = dd.build_combined_pool(
        platinai_path=tmp_path / "p.xlsx",
        cytotox_path=tmp_path / "c.csv",
        tmqm_dir=tmp_path / "t",
        n=10,
    )
    # All sources missing -> fallback to fingerprint pool built from synthetic
    # SMILES that the helper falls back to.  We accept either empty or
    # non-empty but the API contract must be satisfied (lists equal length).
    assert len(smiles) == len(sources)
    assert len(smiles) <= 10


def test_build_combined_pool_with_synthetic_pools(monkeypatch):
    """Smoke test using a synthetic per-loader monkey-patch."""
    fake_pt = ["[Pt](Cl)(Cl)(N)N", "[Pt]([NH3])([NH3])[Cl]", "[Pt](N)(N)(O)O"]
    fake_cy = ["[Ru](c1ccccc1)(c1ccccc1)(C)(C)", "[Ir](C)(C)(C)(C)(C)C"]
    fake_tm = ["[Fe](O)(O)(O)(O)(O)O", "[Zn](O)(O)(O)(O)"]
    monkeypatch.setattr(dd, "load_platinai_smiles", lambda *a, **k: fake_pt)
    monkeypatch.setattr(dd, "load_metal_cytotox_smiles", lambda *a, **k: fake_cy)
    monkeypatch.setattr(dd, "load_tmqm_smiles", lambda *a, **k: fake_tm)
    smiles, sources = dd.build_combined_pool(n=5)
    assert len(smiles) == 5
    assert len(sources) == 5
    assert all(src in {"platinai", "metal_cytotox", "tmqm"} for src in sources)


def test_cache_csv_writes_expected_columns(tmp_path):
    out = tmp_path / "pool.csv"
    smiles = ["CCO", "c1ccccc1", "[Pt](Cl)(Cl)(N)N"]
    sources = ["platinai", "tmqm", "metal_cytotox"]
    dd.cache_csv(smiles, sources, out)
    assert out.is_file()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["smiles"] == "CCO"
    assert rows[0]["source"] == "platinai"
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# Tiny shims to avoid the heavy numpy import in this test file's helpers
# ---------------------------------------------------------------------------
def np_fill(x):
    import numpy as np
    return np.array(x)


def np_max(a, b):
    import numpy as np
    return np.maximum(a, b)
