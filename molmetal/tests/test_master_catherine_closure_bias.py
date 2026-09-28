"""Tests for :mod:`molmetal.molmetal_lam.reward.master_catherine_closure_bias`.

Wf-247-D TODO-126 §3 M2 master facade — single-API orchestrator for
the three Catherine-closure bias flavours (oversample /
generation_bias / feature_flag).

This module exercises the master facade on every supported selector:

* ``"oversample"`` — synthetic cohort oversample (delegates to
  :func:`pocket_closure_bias.closure_oversample`, which in turn
  wraps :class:`closure_oversample.ClosureOversampler`).
* ``"generation_bias"`` — per-pocket Catherine-bounded candidate
  generation (delegates to
  :func:`pocket_closure_bias.closure_bounded_smiles_generation`).
* ``"feature_flag"`` — Boolean Catherine-closure gate for ASTSR
  (TODO-122; the **new** flavour shipped by Wf-247-D).

Plus the shared validation, payload schema, batched ASTSR path, and
the verification builder.

All tests are **CPU-only**; the RDKit-touching selectors
(``oversample``, ``generation_bias``) run a small ``n`` so the suite
completes within seconds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.reward.master_catherine_closure_bias import (  # noqa: E402
    ALL_ENGINE_TAGS,
    CatherineClosureFeatureFlag,
    ENGINE_FEATURE_FLAG,
    ENGINE_GENERATION_BIAS,
    ENGINE_OVERSAMPLE,
    MasterCatherineBiasResult,
    MasterCatherineClosureBias,
    SUPPORTED_BIAS_TYPES,
    build_master_verification,
)


# ---------------------------------------------------------------------------
# Backward-compat sanity — canonical sub-modules still import + work
# ---------------------------------------------------------------------------


def test_canonical_submodules_unchanged():
    """The master is additive; canonical APIs must remain importable."""
    from molmetal.molmetal_lam.reward.closure_oversample import (
        ClosureOversampleResult,
        ClosureOversampler,
        RH_SEED_SCAFFOLDS,
    )
    from molmetal.molmetal_lam.reward.pocket_closure_bias import (
        PocketBiasVerdict,
        closure_bounded_smiles_generation,
        closure_oversample,
        pocket_specific_closure_bias,
        per_pocket_pharmacophore_lookup,
    )
    assert callable(closure_oversample)
    assert callable(closure_bounded_smiles_generation)
    assert callable(pocket_specific_closure_bias)
    assert callable(per_pocket_pharmacophore_lookup)
    assert callable(ClosureOversampler)
    assert isinstance(ClosureOversampleResult, type)
    # Per-pocket bias is independently verified elsewhere; sanity only.
    v = pocket_specific_closure_bias("1h36")
    assert v.bias_sum == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_supported_bias_types(self):
        assert tuple(SUPPORTED_BIAS_TYPES) == (
            "oversample", "generation_bias", "feature_flag",
        )

    def test_engine_tags_distinct(self):
        assert len(set(ALL_ENGINE_TAGS)) == 3
        assert ENGINE_OVERSAMPLE == "closure_catherine_oversample"
        assert ENGINE_GENERATION_BIAS == "closure_catherine_bounded"
        assert ENGINE_FEATURE_FLAG == "closure_catherine_feature_flag"
        assert set(ALL_ENGINE_TAGS) == {
            ENGINE_OVERSAMPLE,
            ENGINE_GENERATION_BIAS,
            ENGINE_FEATURE_FLAG,
        }


# ---------------------------------------------------------------------------
# MasterCatherineClosureBias — validation
# ---------------------------------------------------------------------------


class TestInitValidation:
    def test_defaults(self):
        m = MasterCatherineClosureBias()
        assert m.bias_on_pass == 1.0
        assert m.bias_on_fail == 0.0

    @pytest.mark.parametrize("v", [-0.1, 1.5, -1e-9])
    def test_bias_on_pass_out_of_range(self, v):
        with pytest.raises(ValueError, match="bias_on_pass"):
            MasterCatherineClosureBias(bias_on_pass=v)

    @pytest.mark.parametrize("v", [-0.1, 1.5, 2.0])
    def test_bias_on_fail_out_of_range(self, v):
        with pytest.raises(ValueError, match="bias_on_fail"):
            MasterCatherineClosureBias(bias_on_fail=v)


# ---------------------------------------------------------------------------
# 1. feature_flag — the NEW ASTSR selector
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    def test_single_term_returns_flag(self):
        m = MasterCatherineClosureBias()
        payload = {
            "term_id": "test_term_1",
            "metal": "Ru",
            "witness_path": [],
            "smiles": "[Ru](N)(N)(N)(N)(N)N",
        }
        res = m.apply(payload, bias_type="feature_flag")
        assert isinstance(res, MasterCatherineBiasResult)
        assert res.bias_type == "feature_flag"
        assert res.engine == ENGINE_FEATURE_FLAG
        assert math.isfinite(res.wall_time_s)
        assert res.payload["term_id"] == "test_term_1"
        assert isinstance(res.payload["six_of_six_pass"], bool)
        assert set(res.payload["per_property"].keys()) == {
            "P1_reflexivity", "P2_transitivity", "P3_symmetry",
            "P4_dblock", "P5_reachability", "P6_composability",
        }
        assert res.payload["bias_used"] in (0.0, 1.0)

    def test_bias_used_matches_pass(self):
        m = MasterCatherineClosureBias()
        payload = {
            "term_id": "pass_term",
            "metal": "Pt",
            "witness_path": [],
            "smiles": "Nc1nc(N)nc(N)n1",
        }
        res = m.apply(payload, bias_type="feature_flag")
        # bias_used = bias_on_pass on PASS
        if res.payload["six_of_six_pass"]:
            assert res.payload["bias_used"] == m.bias_on_pass
        else:
            assert res.payload["bias_used"] == m.bias_on_fail

    def test_custom_bias_on_pass_used(self):
        m = MasterCatherineClosureBias(bias_on_pass=0.7, bias_on_fail=0.2)
        payload = {
            "term_id": "p",
            "metal": "Pt",
            "witness_path": [],
            "smiles": "Nc1nc(N)nc(N)n1",
        }
        res = m.apply(payload, bias_type="feature_flag")
        if res.payload["six_of_six_pass"]:
            assert res.payload["bias_used"] == 0.7
        else:
            assert res.payload["bias_used"] == 0.2

    def test_batch_returns_feature_flags(self):
        m = MasterCatherineClosureBias()
        batch = [
            {
                "term_id": f"b_{i}",
                "metal": "Pt",
                "witness_path": ["CuAAC"] if i % 2 == 0 else ["AmideCoupling"],
                "smiles": ("Nc1nc(N)nc(N)n1" if i % 2 == 0
                           else "NS(=O)(=O)c1ccc(C#C)cc1"),
            }
            for i in range(4)
        ]
        res = m.apply(batch, bias_type="feature_flag")
        assert res.engine == ENGINE_FEATURE_FLAG
        assert len(res.feature_flags) == 4
        assert res.payload["batch_size"] == 4
        assert res.payload["six_of_six_pass_rate"] == pytest.approx(
            sum(1 for f in res.feature_flags if f.six_of_six_pass) / 4.0,
            abs=1e-9,
        )

    def test_batch_empty_raises(self):
        m = MasterCatherineClosureBias()
        with pytest.raises(ValueError, match="empty"):
            m.apply([], bias_type="feature_flag")

    def test_missing_term_id_raises(self):
        m = MasterCatherineClosureBias()
        with pytest.raises(ValueError, match="term_id"):
            m.apply({"metal": "Pt"}, bias_type="feature_flag")

    def test_invalid_payload_type_raises(self):
        m = MasterCatherineClosureBias()
        with pytest.raises(TypeError, match="dict or list"):
            m.apply(42, bias_type="feature_flag")


# ---------------------------------------------------------------------------
# 2. generation_bias — delegates to closure_bounded_smiles_generation
# ---------------------------------------------------------------------------


class TestGenerationBias:
    def test_known_pocket_returns_candidates(self):
        m = MasterCatherineClosureBias()
        res = m.apply(
            {"pocket_id": "1h36", "n": 5},
            bias_type="generation_bias",
        )
        assert res.engine == ENGINE_GENERATION_BIAS
        assert res.bias_type == "generation_bias"
        assert res.payload["pocket_id"] == "1h36"
        assert isinstance(res.payload["n_seeds"], int)
        assert isinstance(res.payload["n_candidates_kept"], int)
        # All kept candidates must have catherine_pass=True (filter).
        for c in res.payload["candidates"]:
            assert c["catherine_pass"] is True
        # If any candidate was emitted, witness_rules must be a list.
        for c in res.payload["candidates"]:
            assert isinstance(c["witness_rules"], list)

    def test_unknown_pocket_returns_empty_with_notes(self):
        m = MasterCatherineClosureBias()
        res = m.apply(
            {"pocket_id": "XXXX", "n": 5},
            bias_type="generation_bias",
        )
        assert res.engine == ENGINE_GENERATION_BIAS
        assert res.payload["pocket_id"] == "XXXX"
        assert res.payload["n_seeds"] == 0
        assert res.payload["n_candidates_kept"] == 0
        # Notes must mention the unknown pocket (no silent fallback).
        assert "XXXX" in res.notes or "no closure seed" in res.notes

    def test_too_many_metals_in_payload_raises_typeerror(self):
        m = MasterCatherineClosureBias()
        with pytest.raises(TypeError, match="payload=dict"):
            m.apply(["pocket_id", "1h36"], bias_type="generation_bias")


# ---------------------------------------------------------------------------
# 3. oversample — delegates to closure_oversample
# ---------------------------------------------------------------------------


class TestOversample:
    def test_rh_cohort_minimal(self):
        m = MasterCatherineClosureBias()
        # Tiny cohort (n=12) to keep the test fast — the closure BFS
        # still fires + we exercise the full delegation chain.
        res = m.apply(
            {"target_metal": "Rh", "n": 12, "per_scaffold_cap": 3,
             "max_depth": 1},
            bias_type="oversample",
        )
        assert res.engine == ENGINE_OVERSAMPLE
        assert res.bias_type == "oversample"
        assert res.payload["metal"] == "Rh"
        assert res.payload["target_cohort_size"] == 12
        # Cathertine 6/6 pass-rate alias must exist.
        assert "catherine_6of6_pass_rate" in res.payload
        assert res.payload["engine"] == ENGINE_OVERSAMPLE

    def test_invalid_payload_raises_typeerror(self):
        m = MasterCatherineClosureBias()
        with pytest.raises(TypeError, match="payload=dict"):
            m.apply(("Rh", 12), bias_type="oversample")


# ---------------------------------------------------------------------------
# 4. Selector validation
# ---------------------------------------------------------------------------


class TestSelectorValidation:
    def test_unknown_bias_type_raises(self):
        m = MasterCatherineClosureBias()
        with pytest.raises(ValueError, match="bias_type"):
            m.apply({}, bias_type="bogus")

    def test_default_bias_type_is_feature_flag(self):
        m = MasterCatherineClosureBias()
        res_default = m.apply({
            "term_id": "x",
            "metal": "Pt",
            "witness_path": [],
            "smiles": "Nc1nc(N)nc(N)n1",
        })
        res_explicit = m.apply({
            "term_id": "x",
            "metal": "Pt",
            "witness_path": [],
            "smiles": "Nc1nc(N)nc(N)n1",
        }, bias_type="feature_flag")
        assert res_default.bias_type == res_explicit.bias_type == "feature_flag"


# ---------------------------------------------------------------------------
# 5. to_dict — diagnostic snapshot
# ---------------------------------------------------------------------------


class TestToDict:
    def test_no_compute_required(self):
        m = MasterCatherineClosureBias(bias_on_pass=0.8, bias_on_fail=0.1)
        d = m.to_dict()
        assert d["bias_on_pass"] == 0.8
        assert d["bias_on_fail"] == 0.1
        assert tuple(d["supported_bias_types"]) == SUPPORTED_BIAS_TYPES
        assert tuple(d["all_engine_tags"]) == ALL_ENGINE_TAGS


# ---------------------------------------------------------------------------
# 6. build_master_verification — round-trip
# ---------------------------------------------------------------------------


class TestBuildMasterVerification:
    def test_returns_expected_shape(self):
        v = build_master_verification()
        inner = v["master_catherine_closure_bias"]
        assert "feature_flag" in inner
        assert "feature_flag_batch" in inner
        # single feature_flag
        single = inner["feature_flag"]
        assert single["engine"] == ENGINE_FEATURE_FLAG
        assert isinstance(single["six_of_six_pass"], bool)
        assert single["bias_used"] in (0.0, 1.0)
        # batch
        batch = inner["feature_flag_batch"]
        assert batch["batch_size"] == 4
        assert 0.0 <= batch["pass_rate"] <= 1.0
        assert batch["engine"] == ENGINE_FEATURE_FLAG
        # outer metadata
        assert list(v["supported_bias_types"]) == list(SUPPORTED_BIAS_TYPES)
        assert list(v["engine_tags"]) == list(ALL_ENGINE_TAGS)
        assert v["cpu_only"] is True


# ---------------------------------------------------------------------------
# 7. CatherineClosureFeatureFlag — dataclass shape
# ---------------------------------------------------------------------------


class TestCatherineClosureFeatureFlag:
    def test_default_engine(self):
        f = CatherineClosureFeatureFlag(
            term_id="t1", six_of_six_pass=True,
            per_property={"P1_reflexivity": True, "P2_transitivity": True,
                          "P3_symmetry": True, "P4_dblock": True,
                          "P5_reachability": True, "P6_composability": True},
            bias_used=1.0,
        )
        assert f.engine == ENGINE_FEATURE_FLAG

    def test_engine_tag_is_the_feature_flag_constant(self):
        assert ENGINE_FEATURE_FLAG == "closure_catherine_feature_flag"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
