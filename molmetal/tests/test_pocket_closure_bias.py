"""Tests for :mod:`molmetal.molmetal_lam.reward.pocket_closure_bias`.

Wf-123 Phase 2 — M2 closure-bias module (TODO-126 §3 / M2)
============================================================

This module is the **shared closure-bias module** used by 4 strengthen
TODOs (119, 121, 124, 125).  These tests cover the 4 public functions
plus the supporting per-pocket tables:

1. :func:`pocket_specific_closure_bias` — returns a valid bias dict;
   bias_dict sums to 1.0; unknown pocket_id returns
   ``is_known_pocket=False`` (no silent fallback); metalloenzyme
   pockets carry non-zero metal-coord weights.
2. :func:`closure_bounded_smiles_generation` — emits candidates with
   Catherine 6/6 PASS; respects Catherine 6/6 (no false-PASS candidates
   slip through); unknown pocket_id returns an empty result.
3. :func:`closure_oversample` — thin wrapper over
   :class:`ClosureOversampler`; returns a dict with the expected
   schema; tagged with ``engine="closure_catherine_oversample"``.
4. :func:`per_pocket_pharmacophore_lookup` — returns a dict for known
   pockets; the dict has the schema documented in the docstring; an
   unknown pocket returns ``{}`` (no silent fallback).
5. :func:`build_per_function_verification` — round-trips; all 4
   functions report ``status``; unknown pocket handling is consistent.

All tests are **CPU-only** and run in < 30 seconds with a tight
``n=8`` per-pocket budget on the closure BFS.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imported lazily in the module itself, but for the test surface we
# want a direct import so missing dependencies fail fast.
from molmetal.molmetal_lam.reward.pocket_closure_bias import (
    ALL_BIAS_RULES,
    BoundedGenerationResult,
    DEFAULT_CLOSURE_MAX_DEPTH,
    DEFAULT_GENERATION_N,
    DEFAULT_OVERSAMPLE_SIZE,
    FIVE_CLICK_RULES,
    METAL_COORD_RULES,
    PocketBiasVerdict,
    SUPPORTED_POCKETS,
    build_per_function_verification,
    closure_bounded_smiles_generation,
    closure_oversample,
    per_pocket_pharmacophore_lookup,
    pocket_specific_closure_bias,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_click_weights_present(bias_dict):
    """True iff bias_dict has weight entries for all 5 click rules."""
    return all(r in bias_dict for r in FIVE_CLICK_RULES)


def _bias_sums_to_one(verdict, tol=1.0e-9):
    return abs(float(verdict.bias_sum) - 1.0) <= tol


def _require_closure_importable():
    """Skip the closure-BFS-dependent tests if RDKit can't import.

    The closure BFS loads RDKit + the closure-theorem machinery
    lazily; if either fails (e.g. exotic envs without the chemistry
    stack) we skip those tests rather than crash on import.
    """
    try:
        from molmetal_lam.lam_chem.closure import ProductiveSpace  # noqa: F401
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. pocket_specific_closure_bias
# ---------------------------------------------------------------------------


class TestPocketSpecificClosureBias:
    def test_returns_valid_verdict_for_known_pocket(self):
        verdict = pocket_specific_closure_bias("1h36")
        assert isinstance(verdict, PocketBiasVerdict)
        assert verdict.is_known_pocket is True
        assert verdict.pocket_id == "1h36"
        assert verdict.bias_sum > 0.0

    def test_bias_dict_has_all_five_click_rules(self):
        verdict = pocket_specific_closure_bias("3ny8")
        assert _all_click_weights_present(verdict.bias_dict)

    def test_bias_dict_sums_to_one_for_all_known_pockets(self):
        for pocket_id in SUPPORTED_POCKETS:
            verdict = pocket_specific_closure_bias(pocket_id)
            assert verdict.is_known_pocket, pocket_id
            assert _bias_sums_to_one(
                verdict
            ), f"pocket={pocket_id} bias_sum={verdict.bias_sum}"

    def test_click_rule_weights_are_non_negative(self):
        for pocket_id in SUPPORTED_POCKETS:
            verdict = pocket_specific_closure_bias(pocket_id)
            for rule in FIVE_CLICK_RULES:
                assert verdict.bias_dict[rule] >= 0.0, (
                    f"pocket={pocket_id} rule={rule} "
                    f"weight={verdict.bias_dict[rule]}"
                )

    def test_metal_coord_weights_non_zero_for_metalloenzyme_pockets(self):
        """Metalloenzyme pockets (1h36, 4aaw) must give metal-coord rules
        a non-zero bias (Zn-coordination is part of the chemistry)."""
        for pocket_id in ("1h36", "4aaw"):
            verdict = pocket_specific_closure_bias(pocket_id)
            assert verdict.metalloenzyme is True, pocket_id
            assert verdict.default_metal is not None, pocket_id
            for rule in METAL_COORD_RULES:
                assert verdict.bias_dict[rule] > 0.0, (
                    f"pocket={pocket_id} rule={rule} "
                    f"weight={verdict.bias_dict[rule]}"
                )

    def test_metal_coord_weights_zero_for_non_metalloenzyme_pockets(self):
        """Non-metalloenzyme pockets (3eml, 3ny8, 4rlu) must give
        metal-coord rules a zero weight (no Zn coordination in those
        receptors)."""
        for pocket_id in ("3eml", "3ny8", "4rlu"):
            verdict = pocket_specific_closure_bias(pocket_id)
            assert verdict.metalloenzyme is False, pocket_id
            assert verdict.default_metal is None, pocket_id
            for rule in METAL_COORD_RULES:
                assert verdict.bias_dict[rule] == 0.0, (
                    f"pocket={pocket_id} rule={rule} "
                    f"weight={verdict.bias_dict[rule]}"
                )

    def test_unknown_pocket_id_returns_no_silent_fallback(self):
        verdict = pocket_specific_closure_bias("xx99")
        assert verdict.is_known_pocket is False
        assert verdict.bias_dict == {}
        assert verdict.bias_sum == 0.0
        assert "unknown" in verdict.notes.lower()

    def test_include_metal_coord_false_zeros_metal_coord(self):
        verdict = pocket_specific_closure_bias("1h36", include_metal_coord=False)
        assert verdict.is_known_pocket is True
        for rule in METAL_COORD_RULES:
            assert verdict.bias_dict[rule] == 0.0
        # Click rules still sum to 1.0 (no rescaling needed because
        # metal-coord terms are absent from the bias dict).
        click_sum = sum(verdict.bias_dict[r] for r in FIVE_CLICK_RULES)
        assert abs(click_sum - 1.0) <= 1.0e-9

    def test_metal_coord_weight_param_clamped(self):
        """metal_coord_weight=0.0 makes metal-coord weights 0.0 (and
        preserves click-rule sum-of-1)."""
        verdict_zero = pocket_specific_closure_bias("1h36", metal_coord_weight=0.0)
        for rule in METAL_COORD_RULES:
            assert verdict_zero.bias_dict[rule] == 0.0
        assert _bias_sums_to_one(verdict_zero)


# ---------------------------------------------------------------------------
# 2. closure_bounded_smiles_generation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _require_closure_importable(),
    reason="closure-theorem machinery (RDKit + lam_chem.closure) not importable",
)
class TestClosureBoundedSmilesGeneration:
    @pytest.mark.parametrize("pocket_id", list(SUPPORTED_POCKETS))
    def test_emits_candidates_with_catherine_pass(self, pocket_id):
        result = closure_bounded_smiles_generation(pocket_id, n=4)
        assert isinstance(result, BoundedGenerationResult)
        assert result.pocket_id == pocket_id
        assert result.n_seeds >= 1, pocket_id
        # All kept candidates must have catherine_pass=True.
        for cand in result.candidates:
            assert cand["catherine_pass"] is True, (
                f"pocket={pocket_id} candidate failed Catherine: " f"{cand}"
            )

    @pytest.mark.parametrize("pocket_id", list(SUPPORTED_POCKETS))
    def test_engine_tag_consistent(self, pocket_id):
        result = closure_bounded_smiles_generation(pocket_id, n=2)
        assert result.engine == "closure_catherine_bounded"

    @pytest.mark.parametrize("pocket_id", list(SUPPORTED_POCKETS))
    def test_candidates_smiles_are_non_empty_strings(self, pocket_id):
        result = closure_bounded_smiles_generation(pocket_id, n=2)
        for cand in result.candidates:
            assert isinstance(cand["smiles"], str)
            assert len(cand["smiles"]) > 0
            assert cand["depth"] >= 1
            assert cand["source_seed"] in result.seed_smiles

    def test_unknown_pocket_id_returns_empty_result(self):
        result = closure_bounded_smiles_generation("xx99", n=4)
        assert result.n_seeds == 0
        assert result.n_candidates_raw == 0
        assert result.n_candidates_kept == 0
        assert result.candidates == []
        assert result.catherine_pass_rate == 1.0  # vacuous
        assert (
            "unknown" in result.notes.lower()
            or "no closure seed" in result.notes.lower()
        )

    def test_pass_rate_one_for_kept_candidates(self):
        """Catherine 6/6 PASS rate among kept candidates is exactly 1.0
        (since failing candidates are filtered out)."""
        result = closure_bounded_smiles_generation("3ny8", n=4)
        assert result.n_candidates_kept > 0
        assert result.catherine_pass_rate == 1.0


# ---------------------------------------------------------------------------
# 3. closure_oversample
# ---------------------------------------------------------------------------


class TestClosureOversample:
    def test_returns_dict_with_expected_schema(self):
        result = closure_oversample("Rh", n=10, per_scaffold_cap=3)
        assert isinstance(result, dict)
        # Required keys
        for key in (
            "metal",
            "n_seeds",
            "n_products_raw",
            "n_compounds_kept",
            "catherine_pass_rate",
            "catherine_6of6_pass_rate",
            "target_cohort_size",
            "wall_time_s",
            "engine",
            "per_compound",
        ):
            assert key in result, f"missing key {key!r} in result"

    def test_engine_tag_set_to_oversample(self):
        result = closure_oversample("Rh", n=10, per_scaffold_cap=3)
        assert result["engine"] == "closure_catherine_oversample"

    def test_default_n_is_500(self):
        # The default ``n`` argument mirrors the TODO-121 §2.2 target.
        # We don't run a 500-cohort BFS in the smoke test (too slow for
        # the unit suite); we verify the public symbol only.
        import inspect

        sig = inspect.signature(closure_oversample)
        assert sig.parameters["n"].default == 500
        assert DEFAULT_OVERSAMPLE_SIZE == 500

    def test_alias_pass_rate_matches_primary(self):
        result = closure_oversample("Rh", n=10, per_scaffold_cap=3)
        # catherine_6of6_pass_rate should mirror catherine_pass_rate.
        assert result.get("catherine_6of6_pass_rate") == result.get(
            "catherine_pass_rate"
        )


# ---------------------------------------------------------------------------
# 4. per_pocket_pharmacophore_lookup
# ---------------------------------------------------------------------------


class TestPerPocketPharmacophoreLookup:
    @pytest.mark.parametrize("pocket_id", list(SUPPORTED_POCKETS))
    def test_returns_non_empty_dict_for_known_pocket(self, pocket_id):
        descriptor = per_pocket_pharmacophore_lookup(pocket_id)
        assert isinstance(descriptor, dict)
        assert len(descriptor) > 0, f"empty descriptor dict for pocket={pocket_id}"

    @pytest.mark.parametrize("pocket_id", list(SUPPORTED_POCKETS))
    def test_descriptor_has_required_keys(self, pocket_id):
        descriptor = per_pocket_pharmacophore_lookup(pocket_id)
        for key in (
            "receptor_family",
            "metal_coord_required",
            "default_metal",
            "mw_range",
            "logp_range",
            "hbd_max",
            "hba_max",
            "rotb_max",
            "ring_count_min",
            "ring_count_max",
            "psa_max",
            "required_warheads",
            "preferred_scaffolds",
            "literature_refs",
        ):
            assert key in descriptor, f"pocket={pocket_id} missing key {key!r}"

    def test_metal_coord_required_matches_default_metal(self):
        for pocket_id in SUPPORTED_POCKETS:
            d = per_pocket_pharmacophore_lookup(pocket_id)
            if d["metal_coord_required"]:
                assert d["default_metal"] is not None
            else:
                assert d["default_metal"] is None

    def test_mw_range_is_tuple_of_floats(self):
        for pocket_id in SUPPORTED_POCKETS:
            d = per_pocket_pharmacophore_lookup(pocket_id)
            mw_range = d["mw_range"]
            assert isinstance(mw_range, (tuple, list))
            assert len(mw_range) == 2
            assert mw_range[0] < mw_range[1]
            assert all(isinstance(x, (int, float)) for x in mw_range)

    def test_unknown_pocket_id_returns_empty_dict(self):
        descriptor = per_pocket_pharmacophore_lookup("xx99")
        assert descriptor == {}

    def test_required_warheads_non_empty_list(self):
        for pocket_id in SUPPORTED_POCKETS:
            d = per_pocket_pharmacophore_lookup(pocket_id)
            assert isinstance(d["required_warheads"], list)
            assert len(d["required_warheads"]) >= 1


# ---------------------------------------------------------------------------
# 5. build_per_function_verification
# ---------------------------------------------------------------------------


class TestBuildPerFunctionVerification:
    def test_returns_serialisable_dict(self):
        bundle = build_per_function_verification()
        assert isinstance(bundle, dict)
        assert bundle["cpu_only"] is True
        # JSON round-trip (serialisable).
        json.dumps(bundle, default=lambda o: str(o))

    def test_top_level_engine_set_lists_all_four_functions(self):
        bundle = build_per_function_verification()
        assert set(bundle["engine_set"]) == set(
            (
                "pocket_specific_closure_bias",
                "closure_bounded_smiles_generation",
                "closure_oversample",
                "per_pocket_pharmacophore_lookup",
            )
        )

    def test_per_function_keys_present(self):
        bundle = build_per_function_verification()
        per_function = bundle["per_function"]
        for fn in (
            "pocket_specific_closure_bias",
            "closure_bounded_smiles_generation",
            "closure_oversample",
            "per_pocket_pharmacophore_lookup",
        ):
            assert fn in per_function, f"missing per-function key {fn!r}"
            assert "status" in per_function[fn]
            # status must be one of the declared status strings
            assert per_function[fn]["status"] in {"PASS", "FAIL", "DEFER"}

    def test_pocket_bias_per_pocket_subresult(self):
        bundle = build_per_function_verification()
        bias_section = bundle["per_function"]["pocket_specific_closure_bias"]
        assert bias_section["n_known_pockets"] == len(SUPPORTED_POCKETS)
        per_pocket = bias_section["per_pocket"]
        for pocket_id in SUPPORTED_POCKETS:
            assert pocket_id in per_pocket
            entry = per_pocket[pocket_id]
            assert entry["is_known_pocket"] is True
            assert abs(entry["bias_sum"] - 1.0) <= 1.0e-9

    def test_pharmacophore_per_pocket_subresult(self):
        bundle = build_per_function_verification()
        pharma_section = bundle["per_function"]["per_pocket_pharmacophore_lookup"]
        assert pharma_section["n_known_pockets"] == len(SUPPORTED_POCKETS)
        per_pocket = pharma_section["per_pocket"]
        for pocket_id in SUPPORTED_POCKETS:
            assert pocket_id in per_pocket
            entry = per_pocket[pocket_id]
            assert entry["has_mw_range"] is True
            assert entry["has_required_warheads"] is True


# ---------------------------------------------------------------------------
# 6. Constants / public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_supported_pockets_is_tuple_of_5(self):
        assert isinstance(SUPPORTED_POCKETS, tuple)
        assert len(SUPPORTED_POCKETS) == 5
        assert "1h36" in SUPPORTED_POCKETS
        assert "3eml" in SUPPORTED_POCKETS
        assert "3ny8" in SUPPORTED_POCKETS
        assert "4aaw" in SUPPORTED_POCKETS
        assert "4rlu" in SUPPORTED_POCKETS

    def test_five_click_rules_match_brief(self):
        assert set(FIVE_CLICK_RULES) == {
            "CuAAC",
            "SPAAC",
            "ThiolEne",
            "Suzuki",
            "AmideCoupling",
        }

    def test_metal_coord_rules_are_2(self):
        assert len(METAL_COORD_RULES) == 2
        assert set(METAL_COORD_RULES) == {
            "MetalLigandExchange",
            "AquaExchange",
        }

    def test_all_bias_rules_is_union_of_click_and_metal_coord(self):
        assert set(ALL_BIAS_RULES) == (set(FIVE_CLICK_RULES) | set(METAL_COORD_RULES))

    def test_default_n_is_100(self):
        assert DEFAULT_GENERATION_N == 100

    def test_default_closure_depth_is_2(self):
        assert DEFAULT_CLOSURE_MAX_DEPTH == 2
