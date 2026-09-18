"""Tests for round-7 cherry-pick scoring wrappers.

Three required tests:

- ``test_each_wrapper_importable`` — every wrapper module imports and has
  an ``AVAILABLE`` flag.
- ``test_score_dispatcher_routes_correctly`` — ``score_with(method, ...)``
  routes to the right wrapper.
- ``test_wrappers_handle_missing_heavy_deps`` — wrappers do not crash when
  optional deps are missing (TDC, QVina, vina binary).

Plus a few sanity tests for the per-mol metric shape and the SoftMol
hit-rate gate.
"""
from __future__ import annotations

import importlib

import pytest


_SMILES_BASIC = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
]
_SMILES_BAD = ["", "NOT_A_SMILES", None]


# ---------------------------------------------------------------------------
# 1. Importability
# ---------------------------------------------------------------------------
def test_each_wrapper_importable():
    for mod_name in (
        "molmetal.eval.scoring",
        "molmetal.eval.scoring.targetdiff_score",
        "molmetal.eval.scoring.pocket2mol_score",
        "molmetal.eval.scoring.softmol_score",
        "molmetal.eval.scoring.sascorer",
    ):
        mod = importlib.import_module(mod_name)
        assert mod is not None

    # Top-level facade must expose the dispatcher and the helper.
    from molmetal.eval.scoring import score_with, available_methods, list_methods

    methods = list_methods()
    assert set(methods) == {"targetdiff", "pocket2mol", "softmol"}
    # on this env at least RDKit+SA must be installed → all three available
    avail = available_methods()
    assert isinstance(avail, list)
    # all three are RDKit+SA-backed, so they should be available
    assert set(avail) == {"targetdiff", "pocket2mol", "softmol"}


# ---------------------------------------------------------------------------
# 2. Dispatcher
# ---------------------------------------------------------------------------
def test_score_dispatcher_routes_correctly():
    from molmetal.eval.scoring import score_with

    # unknown method → None
    assert score_with("not_a_method", _SMILES_BASIC) is None

    # each known method returns its expected schema
    td = score_with("targetdiff", _SMILES_BASIC)
    assert td is not None and "per_mol" in td and "summary" in td and "vina_scores" in td
    assert td["vina_scores"] is None  # docking intentionally not lifted

    pm = score_with("pocket2mol", _SMILES_BASIC)
    assert pm is not None and "per_mol" in pm and "vina_scores" in pm
    assert pm["vina_scores"] is None

    sm = score_with(
        "softmol",
        _SMILES_BASIC,
        rv_scores=[11.0, 9.0, 9.5, 10.2],
        target="parp1",
    )
    assert sm is not None
    for k in ("n_input", "n_dedup", "n_qs", "n_hits", "target",
              "hit_threshold", "qed_threshold", "sa_threshold"):
        assert k in sm, f"softmol key missing: {k}"


# ---------------------------------------------------------------------------
# 3. Missing heavy deps do not crash
# ---------------------------------------------------------------------------
def test_wrappers_handle_missing_heavy_deps(monkeypatch):
    """Force the heavy-dep-import branches to fail and ensure wrappers
    still return a valid (possibly None-on-dep) output without raising."""
    # Block TDC: SoftMol wrapper must fall back to RDKit and still work.
    import molmetal.eval.scoring.softmol_score as sm_mod

    monkeypatch.setattr(sm_mod, "_TDC_OK", False)

    out = sm_mod.score(
        ["CCO", "c1ccccc1"],
        rv_scores=[11.0, 9.0],
        target="parp1",
    )
    assert out is not None
    assert out["backend"] == "rdkit"
    assert "n_input" in out

    # Simulate RDKit import failure for targetdiff: it should return None
    # rather than raising.
    import molmetal.eval.scoring.targetdiff_score as td_mod

    monkeypatch.setattr(td_mod, "_RDKIT_OK", False)
    assert td_mod.score(["CCO"]) is None

    # Same for pocket2mol.
    import molmetal.eval.scoring.pocket2mol_score as pm_mod

    monkeypatch.setattr(pm_mod, "_RDKIT_OK", False)
    assert pm_mod.score(["CCO"]) is None

    # unknown target on softmol → None (no crash)
    assert sm_mod.score(["CCO"], rv_scores=[11.0], target="not_a_target") is None


# ---------------------------------------------------------------------------
# 4. Sanity: per-mol metric shape
# ---------------------------------------------------------------------------
def test_targetdiff_per_mol_metric_shape():
    from molmetal.eval.scoring.targetdiff_score import score

    out = score(_SMILES_BASIC + _SMILES_BAD)
    assert out is not None
    assert out["summary"]["n_input"] == len(_SMILES_BASIC) + len(_SMILES_BAD)
    # Each bad row should be None
    n_none = sum(1 for r in out["per_mol"][-len(_SMILES_BAD):] if r is None)
    assert n_none == len(_SMILES_BAD)
    # Each good row should have the right keys
    for row in out["per_mol"][: len(_SMILES_BASIC)]:
        assert set(row.keys()) >= {
            "qed", "sa", "logp", "lipinski", "ring_size", "pains", "vina",
        }
        assert row["vina"] is None  # docking path not lifted
    # Summary numbers must be sensible
    assert out["summary"]["n_ok"] == len(_SMILES_BASIC)
    assert 0.0 <= out["summary"]["qed_mean"] <= 1.0
    assert 0.0 <= out["summary"]["sa_mean"] <= 10.0


def test_pocket2mol_per_mol_metric_shape():
    from molmetal.eval.scoring.pocket2mol_score import score

    out = score(_SMILES_BASIC)
    assert out is not None
    for row in out["per_mol"]:
        assert set(row.keys()) >= {
            "qed", "sa", "logp", "hacc", "hdon", "lipinski", "rdkit_rmsd", "vina",
        }
        assert isinstance(row["hacc"], int)
        assert isinstance(row["hdon"], int)
        assert isinstance(row["lipinski"], int)
        assert len(row["rdkit_rmsd"]) == 3  # [max, min, median]
        assert row["vina"] is None


def test_softmol_hit_rate_gate():
    from molmetal.eval.scoring.softmol_score import score, HIT_THR_BY_TARGET

    # Build a set where parp1 has 1 hit, the others are filtered out.
    smiles = ["CCO", "c1ccccc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"]
    rvs = [11.5, 7.0, 12.0]  # parp1 thr = 10.0 → hits = 2 (indexes 0 and 2)
    out = score(smiles, rv_scores=rvs, target="parp1")
    assert out is not None
    assert out["target"] == "parp1"
    assert out["hit_threshold"] == HIT_THR_BY_TARGET["parp1"]
    # hits should be >0
    assert out["n_hits"] >= 1
    # top5% stats should be floats
    assert out["top5pct_rv"] is not None
    assert out["mean_rv"] is not None


def test_softmol_no_rv_returns_zero_hits():
    from molmetal.eval.scoring.softmol_score import score

    out = score(["CCO", "c1ccccc1"], rv_scores=None, target="parp1")
    assert out is not None
    assert out["n_hits"] == 0
    assert out["top5pct_rv"] is None
