"""Tests for SA closure-bias bench and pocket_closure_bias module.

Wf-126 Phase 3 (TODO-124 m5 SA closure-bias)
============================================

This module exercises both the M2 ``pocket_closure_bias`` module
and the ``bench_sa_closure_bias`` script.  The tests cover:

1. :func:`pocket_specific_closure_bias` — returns :class:`PocketBiasVerdict`.
2. :func:`closure_bounded_smiles_generation` — returns
   :class:`BoundedGenerationResult` with Catherine 6/6 PASS-tagged candidates.
3. :func:`closure_oversample` — thin wrapper over
   :class:`ClosureOversampler` returning a dict.
4. :func:`per_pocket_pharmacophore_lookup` — per-pocket descriptor dict.
5. SA scoring helper (``sa_score_ertl`` + batch aggregation).
6. AiZynth tagging helper.
7. CLI entry point for the bench script.

All tests are CPU-only and use small cohorts (n <= 30) so the suite
runs in < 60 seconds.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.reward.pocket_closure_bias import (  # noqa: E402
    ALL_BIAS_RULES,
    BoundedGenerationResult,
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
# Lazy import of the bench script (molmetal/scripts is not a package —
# we add the directory to sys.path on demand so the script is importable
# as ``scripts.bench_sa_closure_bias``).
# ---------------------------------------------------------------------------


def _import_bench_module():
    """Lazy-import the bench script module."""
    scripts_dir = PROJECT_ROOT / "molmetal" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("bench_sa_closure_bias")


# ---------------------------------------------------------------------------
# 1. pocket_specific_closure_bias — per-pocket bias verdict
# ---------------------------------------------------------------------------


class TestPocketSpecificClosureBias:
    def test_known_pocket_returns_pocketbiasverdict(self):
        v = pocket_specific_closure_bias("1h36")
        assert isinstance(v, PocketBiasVerdict)
        assert v.pocket_id == "1h36"
        assert v.is_known_pocket is True
        assert v.bias_sum == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("pid", list(SUPPORTED_POCKETS))
    def test_all_known_pockets_sum_to_one(self, pid):
        v = pocket_specific_closure_bias(pid)
        assert v.is_known_pocket is True
        assert v.bias_sum == pytest.approx(1.0, abs=1e-9)

    def test_bias_dict_has_all_rules(self):
        v = pocket_specific_closure_bias("1h36")
        # Must contain all 5 click rules.
        for name in FIVE_CLICK_RULES:
            assert name in v.bias_dict
        # And both metal-coord rules (metalloenzyme pocket).
        for name in METAL_COORD_RULES:
            assert name in v.bias_dict

    def test_unknown_pocket_returns_known_false(self):
        v = pocket_specific_closure_bias("XXXX")
        assert v.is_known_pocket is False
        assert v.bias_dict == {}
        assert v.bias_sum == 0.0
        assert "unknown pocket_id" in v.notes

    def test_metalloenzyme_pocket_for_1h36(self):
        v = pocket_specific_closure_bias("1h36")
        # 1h36 is metalloenzyme=True in the curated table.
        assert v.metalloenzyme is True
        assert v.default_metal == "Zn"

    def test_click_rule_weights_nonneg(self):
        v = pocket_specific_closure_bias("3eml")
        for w in v.bias_dict.values():
            assert w >= 0.0


# ---------------------------------------------------------------------------
# 2. closure_bounded_smiles_generation — per-pocket cohort
# ---------------------------------------------------------------------------


class TestClosureBoundedSmilesGeneration:
    @pytest.mark.parametrize("pid", list(SUPPORTED_POCKETS))
    def test_pocket_returns_bounded_result(self, pid):
        r = closure_bounded_smiles_generation(
            pocket_id=pid, n=10, max_depth=2,
            max_products=80,
        )
        assert isinstance(r, BoundedGenerationResult)
        assert r.pocket_id == pid
        assert r.n_seeds >= 1
        assert r.n_candidates_kept <= 10
        assert r.engine == "closure_catherine_bounded"
        # All kept candidates must pass Catherine 6/6.
        for c in r.candidates:
            assert c["catherine_pass"] is True
            assert "smiles" in c
            assert "source_seed" in c
            assert "depth" in c
            assert "witness_rules" in c

    def test_unknown_pocket_returns_empty(self):
        r = closure_bounded_smiles_generation(
            pocket_id="XXXX", n=10, max_depth=2,
        )
        assert r.n_seeds == 0
        assert r.n_candidates_kept == 0
        assert r.candidates == []

    def test_invalid_n_zero(self):
        # Should NOT raise — n is just a cap; with n=0, we cap at 0.
        r = closure_bounded_smiles_generation(
            pocket_id="1h36", n=0, max_depth=2,
        )
        # n=0 means we cap at 0 candidates; the BFS may still emit
        # raw candidates but the cap triggers immediately after one
        # candidate is collected.  Verify the cap is small (≤ 1).
        assert r.n_candidates_kept <= 1


# ---------------------------------------------------------------------------
# 3. closure_oversample — synthetic cohort oversample
# ---------------------------------------------------------------------------


class TestClosureOversample:
    def test_rh_oversample_returns_dict(self):
        d = closure_oversample("Rh", n=20, per_scaffold_cap=4)
        assert isinstance(d, dict)
        assert d.get("metal") == "Rh"
        # Engine tag set.
        assert "engine" in d
        assert d["engine"] == "closure_catherine_oversample"

    def test_unknown_metal_runs_without_curated_seeds(self):
        # For non-Rh metals, callers must supply seed_scaffolds; without
        # them the run will still attempt to use the ClosureOversampler
        # defaults (which may not have the metal scaffold pool).
        # We just verify the function returns a dict (not raises).
        d = closure_oversample(
            "unbiased", n=10, per_scaffold_cap=2,
            seed_scaffolds=("CCO",),
        )
        assert isinstance(d, dict)
        assert "metal" in d


# ---------------------------------------------------------------------------
# 4. per_pocket_pharmacophore_lookup
# ---------------------------------------------------------------------------


class TestPerPocketPharmacophoreLookup:
    @pytest.mark.parametrize("pid", list(SUPPORTED_POCKETS))
    def test_known_pocket_returns_nonempty(self, pid):
        d = per_pocket_pharmacophore_lookup(pid)
        assert isinstance(d, dict)
        assert "mw_range" in d
        assert "logp_range" in d
        assert "hbd_max" in d
        assert "hba_max" in d
        assert "rotb_max" in d
        assert "ring_count_min" in d
        assert "ring_count_max" in d
        assert "psa_max" in d
        assert "required_warheads" in d
        assert "preferred_scaffolds" in d
        assert "literature_refs" in d

    def test_unknown_pocket_returns_empty(self):
        d = per_pocket_pharmacophore_lookup("XXXX")
        assert d == {}

    def test_mw_range_is_tuple_of_two_floats(self):
        d = per_pocket_pharmacophore_lookup("1h36")
        assert isinstance(d["mw_range"], tuple)
        assert len(d["mw_range"]) == 2
        assert d["mw_range"][0] < d["mw_range"][1]


# ---------------------------------------------------------------------------
# 5. SA aggregation helper
# ---------------------------------------------------------------------------


class TestAggregateSA:
    def test_empty_returns_nan(self):
        mod = _import_bench_module()
        agg = mod._aggregate_sa([])
        assert agg["n_valid"] == 0
        assert agg["mean"] != agg["mean"]  # NaN

    def test_filters_nan(self):
        mod = _import_bench_module()
        agg = mod._aggregate_sa([2.0, 3.0, float("nan"), 1.0])
        assert agg["n_valid"] == 3
        assert abs(agg["mean"] - 2.0) < 1e-6
        assert agg["min"] == 1.0
        assert agg["max"] == 3.0

    def test_std_with_one_value(self):
        mod = _import_bench_module()
        agg = mod._aggregate_sa([2.5])
        assert agg["std"] == 0.0


# ---------------------------------------------------------------------------
# 6. Aggregate records helper
# ---------------------------------------------------------------------------


class TestAggregateRecords:
    def test_empty_returns_zero(self):
        mod = _import_bench_module()
        agg = mod._aggregate_records([])
        assert agg["n_seeds"] == 0
        assert agg["sa_mean_aggregated"] is None

    def test_nan_filtering(self):
        from bench_sa_closure_bias import CohortRecord  # type: ignore
        mod = _import_bench_module()
        records = [
            CohortRecord(
                cohort_label="x", pocket_id="x", seed=0,
                n_requested=10, n_emitted_raw=10, n_kept=10,
                catherine_pass_rate=1.0, sa_mean=2.5,
                sa_median=2.5, sa_std=0.1, sa_min=2.0,
                sa_max=3.0, n_valid_sa=10,
                aizynth_synthesizable_rate=0.5,
                aizynth_engine_breakdown={"smarts_fallback": 10},
                per_compound=[], wall_time_s=1.0,
            ),
            CohortRecord(
                cohort_label="x", pocket_id="x", seed=1,
                n_requested=10, n_emitted_raw=10, n_kept=10,
                catherine_pass_rate=1.0,
                sa_mean=float("nan"),  # NaN
                sa_median=float("nan"), sa_std=float("nan"),
                sa_min=float("nan"), sa_max=float("nan"),
                n_valid_sa=0,
                aizynth_synthesizable_rate=0.5,
                aizynth_engine_breakdown={"smarts_fallback": 10},
                per_compound=[], wall_time_s=1.0,
            ),
        ]
        agg = mod._aggregate_records(records)
        # Only seed 0 contributes a valid mean.
        assert agg["n_seeds"] == 2
        assert agg["n_seeds_aggregated"] == 1
        assert abs(agg["sa_mean_aggregated"] - 2.5) < 1e-6


# ---------------------------------------------------------------------------
# 7. Engine breakdown merge
# ---------------------------------------------------------------------------


class TestMergeEngineBreakdowns:
    def test_merge_two(self):
        mod = _import_bench_module()
        merged = mod._merge_engine_breakdowns([
            {"aizynthfinder": 2, "smarts_fallback": 3},
            {"aizynthfinder": 1, "smarts_fallback": 5},
        ])
        assert merged == {"aizynthfinder": 3, "smarts_fallback": 8}

    def test_merge_empty(self):
        mod = _import_bench_module()
        assert mod._merge_engine_breakdowns([]) == {}


# ---------------------------------------------------------------------------
# 8. CLI entry point — sample dry run
# ---------------------------------------------------------------------------


class TestBenchScriptCLI:
    """Smoke test the CLI entry point (1 seed, n=5)."""

    def test_cli_runs_closure_bounded(self, tmp_path):
        mod = _import_bench_module()
        bench_main = mod.main

        out_dir = tmp_path / "sa"
        out_dir.mkdir()

        argv = [
            "prog",
            "--cohort", "closure_bounded",
            "--pocket", "1h36",
            "--cohort-size", "5",
            "--seeds", "0",
            "--out-dir", str(out_dir),
            "--quiet",
        ]
        saved = sys.argv
        try:
            sys.argv = argv
            rc = bench_main()
        finally:
            sys.argv = saved
        assert rc == 0
        sa_metrics = out_dir / "sa_metrics.json"
        assert sa_metrics.is_file()
        d = json.loads(sa_metrics.read_text())
        assert d["metadata"]["cohort"] == "closure_bounded"
        assert d["metadata"]["pocket_id"] == "1h36"
        assert d["metadata"]["seeds"] == [0]
        assert len(d["per_seed"]) == 1

    def test_cli_requires_pocket_for_closure_bounded(self, tmp_path):
        mod = _import_bench_module()
        bench_main = mod.main
        argv = [
            "prog", "--cohort", "closure_bounded",
            "--cohort-size", "5", "--seeds", "0",
            "--out-dir", str(tmp_path), "--quiet",
        ]
        saved = sys.argv
        try:
            sys.argv = argv
            rc = bench_main()
        finally:
            sys.argv = saved
        assert rc == 2

    def test_cli_runs_unbiased(self, tmp_path):
        mod = _import_bench_module()
        bench_main = mod.main
        out_dir = tmp_path / "sa_unbiased"
        out_dir.mkdir()
        argv = [
            "prog", "--cohort", "unbiased",
            "--cohort-size", "5", "--seeds", "0",
            "--out-dir", str(out_dir), "--quiet",
        ]
        saved = sys.argv
        try:
            sys.argv = argv
            rc = bench_main()
        finally:
            sys.argv = saved
        assert rc == 0
        d = json.loads((out_dir / "sa_metrics.json").read_text())
        assert d["metadata"]["cohort"] == "unbiased"


# ---------------------------------------------------------------------------
# 9. build_per_function_verification (M2 self-verification)
# ---------------------------------------------------------------------------


class TestBuildPerFunctionVerification:
    def test_returns_dict_with_engines(self):
        v = build_per_function_verification()
        assert isinstance(v, dict)
        assert "engine_set" in v
        # The engine_set is the 4 main public functions of the module.
        assert set(v["engine_set"]) == {
            "pocket_specific_closure_bias",
            "closure_bounded_smiles_generation",
            "closure_oversample",
            "per_pocket_pharmacophore_lookup",
        }
        assert "per_function" in v
        assert v["cpu_only"] is True


# ---------------------------------------------------------------------------
# 10. Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_supported_pockets_complete(self):
        expected = {"1h36", "3eml", "3ny8", "4aaw", "4rlu"}
        assert set(SUPPORTED_POCKETS) == expected

    def test_five_click_rules_complete(self):
        expected = {"CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"}
        assert set(FIVE_CLICK_RULES) == expected

    def test_metal_coord_rules_complete(self):
        assert set(METAL_COORD_RULES) == {"MetalLigandExchange", "AquaExchange"}

    def test_all_bias_rules_union(self):
        assert set(ALL_BIAS_RULES) == set(FIVE_CLICK_RULES) | set(METAL_COORD_RULES)
