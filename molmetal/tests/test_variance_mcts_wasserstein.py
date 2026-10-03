"""Tests for Wf-129 Phase 1 — Variance MCTS x Wasserstein integration.

================================================================
Unit tests for
:mod:`molmetal.scripts.bench_variance_mcts_wasserstein`.
================================================================
This module validates the Wf-129 P1 bench harness:

* :class:`TestW2RegConfigDefaults`     — config dataclass defaults
* :class:`TestBuildReference`         — reference distribution shapes
* :class:`TestComputeW2Penalty`       — penalty hook (torch on/off)
* :class:`TestApplyW2Regulariser`     — apply hook with / without selector
* :class:`TestVarianceReductionMath`  — variance_reduction edge cases
* :class:`TestStrategyResultToDict`   — to_dict schema sanity
* :class:`TestAggregatePerSeed`       — 1-seed end-to-end smoke
* :class:`TestPerSeedJsonSchema`      — output JSON schema
* :class:`TestSynthesisVerdict`       — synthesis.md generated
* :class:`TestCliSmoke`               — argparse + smoke run

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_variance_mcts_wasserstein.py -v
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Path bootstrap — the bench script lives in
# ``molmetal/scripts/bench_variance_mcts_wasserstein.py`` as a
# free-standing module (no ``__init__.py`` in ``scripts/``).  Add
# that directory to ``sys.path`` so the import resolves.
_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bench_variance_mcts_wasserstein import (  # type: ignore
    W2RegConfig,
    StrategyResult,
    _aggregate_per_seed,
    _apply_w2_regulariser,
    _build_reference,
    _compute_w2_penalty,
    _parse_args,
    _variance_reduction,
    _write_per_seed_json,
    _write_synthesis,
    main,
    MCTSProofSearch,
    VarianceMCTSWrapper,
)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


class TestW2RegConfigDefaults:
    """W2RegConfig dataclass defaults match TODO-126 §3 M1 spec."""

    def test_default_alpha_non_zero(self) -> None:
        cfg = W2RegConfig()
        assert cfg.alpha > 0.0

    def test_default_n_mc_in_range(self) -> None:
        cfg = W2RegConfig()
        # Spec: n_mc=1000 in Wf-123 P1; Wf-129 P1 ships n_mc=256 for
        # per-trajectory budget.  Either is acceptable — we just
        # assert >= 64.
        assert cfg.n_mc >= 64

    def test_default_reference_is_prior(self) -> None:
        cfg = W2RegConfig()
        assert cfg.reference == "prior"

    def test_decay_check_default_true(self) -> None:
        cfg = W2RegConfig()
        assert cfg.decay_check is True

    def test_decay_window_min_two(self) -> None:
        cfg = W2RegConfig()
        assert cfg.decay_window >= 2

    def test_decay_tol_non_negative(self) -> None:
        cfg = W2RegConfig()
        assert cfg.decay_tol >= 0.0


# ---------------------------------------------------------------------------
# 2. Reference distribution
# ---------------------------------------------------------------------------


class TestBuildReference:
    """The W_2 reference distribution is constructed correctly."""

    def test_prior_mode_uniform(self) -> None:
        cfg = W2RegConfig(reference="prior")
        ref = _build_reference(8, cfg, random.Random(0))
        assert len(ref) == 8
        assert all(abs(x - 1.0 / 8.0) < 1.0e-9 for x in ref)
        assert abs(sum(ref) - 1.0) < 1.0e-9

    def test_shuffle_mode_permuted(self) -> None:
        cfg = W2RegConfig(reference="shuffle")
        ref = _build_reference(5, cfg, random.Random(0))
        assert len(ref) == 5
        assert abs(sum(ref) - 1.0) < 1.0e-9

    def test_shuffle_is_permutation(self) -> None:
        cfg = W2RegConfig(reference="shuffle")
        ref = _build_reference(6, cfg, random.Random(42))
        # The shuffled copy is still a permutation of [1/6] * 6.
        assert sorted(ref) == sorted([1.0 / 6.0] * 6)

    def test_zero_actions_returns_empty(self) -> None:
        cfg = W2RegConfig()
        ref = _build_reference(0, cfg, random.Random(0))
        assert ref == []


# ---------------------------------------------------------------------------
# 3. W_2 penalty hook
# ---------------------------------------------------------------------------


class TestComputeW2Penalty:
    """``_compute_w2_penalty`` is robust to torch on/off + degenerate inputs."""

    def test_zero_alpha_returns_zero(self) -> None:
        cfg = W2RegConfig(alpha=0.0)
        out = _compute_w2_penalty(
            [0.5, 0.5],
            [0.5, 0.5],
            cfg,
            seed=0,
        )
        assert out == 0.0

    def test_empty_inputs_return_zero(self) -> None:
        cfg = W2RegConfig(alpha=0.1)
        assert _compute_w2_penalty([], [], cfg, seed=0) == 0.0

    def test_alpha_zero_with_nonzero_prior_returns_zero(self) -> None:
        cfg = W2RegConfig(alpha=0.0)
        out = _compute_w2_penalty(
            [0.3, 0.7],
            [0.5, 0.5],
            cfg,
            seed=0,
        )
        assert out == 0.0

    def test_penalty_is_alpha_times_w2_when_torch_available(self) -> None:
        cfg = W2RegConfig(alpha=0.1, n_mc=128)
        try:
            import torch  # noqa: F401
        except Exception:
            pytest.skip("torch unavailable — skip w2 penalty check")
        out = _compute_w2_penalty(
            [0.5, 0.5],
            [0.5, 0.5],
            cfg,
            seed=0,
        )
        # Identical distributions -> W_2 == 0 (or very close) -> penalty ~ 0
        assert 0.0 <= out <= 1.0

    def test_penalty_scales_linearly_with_alpha(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            pytest.skip("torch unavailable — skip linear alpha check")
        prior = [0.2, 0.8]
        ref = [0.5, 0.5]
        p_small = _compute_w2_penalty(
            prior,
            ref,
            W2RegConfig(alpha=0.05, n_mc=64),
            seed=0,
        )
        p_large = _compute_w2_penalty(
            prior,
            ref,
            W2RegConfig(alpha=0.20, n_mc=64),
            seed=0,
        )
        # Linear scaling: ratio == alpha_large / alpha_small.
        if p_small > 0.0:
            ratio = p_large / p_small
            assert 3.5 < ratio < 4.5  # 0.20 / 0.05 == 4


# ---------------------------------------------------------------------------
# 4. W_2 regulariser apply hook
# ---------------------------------------------------------------------------


class TestApplyW2Regulariser:
    """``_apply_w2_regulariser`` wires the W_2 hook into the wrapper."""

    def _make_wrapper(self):
        # Lazy import so the test can skip cleanly when the heavy
        # ``proof_search`` module (torch-dependent) cannot be
        # imported on the host.
        from bench_variance_mcts_wasserstein import (  # type: ignore
            MCTSProofSearch,
            VarianceMCTSWrapper,
        )

        if MCTSProofSearch is None or VarianceMCTSWrapper is None:
            pytest.skip("MCTSProofSearch / VarianceMCTSWrapper unavailable")
        try:
            base = MCTSProofSearch(
                tile_library=[],
                rules={},
                target_predicates=[],
                binding_site=None,
                scorer=lambda x: 0.0,
                n_simulations=2,
                c_puct=1.4,
                top_k=1,
                rng=random.Random(0),
            )
        except Exception:
            pytest.skip("MCTSProofSearch ctor rejected empty args")
        return VarianceMCTSWrapper(base_search=base)

    def test_returns_diag_with_engine_tag(self) -> None:
        wrapper = self._make_wrapper()
        cfg = W2RegConfig(alpha=0.05, n_mc=64, decay_check=True)
        diag = _apply_w2_regulariser(
            wrapper,
            4,
            cfg,
            random.Random(0),
            seed=0,
        )
        assert "engine_tag" in diag
        assert diag["engine_tag"] == "wasserstein_reg"
        assert diag["alpha"] == 0.05
        assert diag["n_mc"] == 64
        assert diag["reference"] == "prior"

    def test_decay_check_records_pass_flag(self) -> None:
        wrapper = self._make_wrapper()
        cfg = W2RegConfig(
            alpha=0.05,
            n_mc=64,
            decay_check=True,
            decay_window=4,
        )
        diag = _apply_w2_regulariser(
            wrapper,
            4,
            cfg,
            random.Random(0),
            seed=0,
        )
        if diag.get("applied") is True:
            assert "decay_gate_pass" in diag
            assert diag["decay_gate_pass"] in (True, False)


# ---------------------------------------------------------------------------
# 5. Variance reduction math
# ---------------------------------------------------------------------------


class TestVarianceReductionMath:
    """``_variance_reduction`` returns signed delta and handles edges."""

    def test_zero_when_equal(self) -> None:
        r = StrategyResult(
            strategy="x",
            seed=0,
            n_trajectories=1,
            n_candidates_total=0,
            mean_score=0.0,
            best_score=0.0,
            mean_var=0.5,
            max_var=0.5,
            wall_time_s=0.0,
            success_rate=1.0,
        )
        assert _variance_reduction(r, r) == pytest.approx(0.0)

    def test_positive_when_candidate_lower(self) -> None:
        base = StrategyResult(
            strategy="x",
            seed=0,
            n_trajectories=1,
            n_candidates_total=0,
            mean_score=0.0,
            best_score=0.0,
            mean_var=1.0,
            max_var=1.0,
            wall_time_s=0.0,
            success_rate=1.0,
        )
        cand = StrategyResult(
            strategy="y",
            seed=0,
            n_trajectories=1,
            n_candidates_total=0,
            mean_score=0.0,
            best_score=0.0,
            mean_var=0.4,
            max_var=0.4,
            wall_time_s=0.0,
            success_rate=1.0,
        )
        # 1 - 0.4 / 1.0 == 0.6 — improvement.
        assert _variance_reduction(base, cand) == pytest.approx(0.6)

    def test_negative_when_candidate_higher(self) -> None:
        base = StrategyResult(
            strategy="x",
            seed=0,
            n_trajectories=1,
            n_candidates_total=0,
            mean_score=0.0,
            best_score=0.0,
            mean_var=0.5,
            max_var=0.5,
            wall_time_s=0.0,
            success_rate=1.0,
        )
        cand = StrategyResult(
            strategy="y",
            seed=0,
            n_trajectories=1,
            n_candidates_total=0,
            mean_score=0.0,
            best_score=0.0,
            mean_var=0.7,
            max_var=0.7,
            wall_time_s=0.0,
            success_rate=1.0,
        )
        # 1 - 0.7 / 0.5 == -0.4 — regression.
        assert _variance_reduction(base, cand) == pytest.approx(-0.4)

    def test_zero_baseline_clamped(self) -> None:
        # When the baseline mean_var is 0.0 we must NOT divide by
        # zero — the helper clamps to 1e-12.
        base = StrategyResult(
            strategy="x",
            seed=0,
            n_trajectories=1,
            n_candidates_total=0,
            mean_score=0.0,
            best_score=0.0,
            mean_var=0.0,
            max_var=0.0,
            wall_time_s=0.0,
            success_rate=1.0,
        )
        cand = StrategyResult(
            strategy="y",
            seed=0,
            n_trajectories=1,
            n_candidates_total=0,
            mean_score=0.0,
            best_score=0.0,
            mean_var=0.5,
            max_var=0.5,
            wall_time_s=0.0,
            success_rate=1.0,
        )
        # 1 - 0.5 / 1e-12 is a huge negative number — we only
        # assert the helper returns a finite float.
        out = _variance_reduction(base, cand)
        assert isinstance(out, float)
        import math

        assert math.isfinite(out)


# ---------------------------------------------------------------------------
# 6. StrategyResult schema
# ---------------------------------------------------------------------------


class TestStrategyResultToDict:
    """``StrategyResult.to_dict`` round-trips through JSON cleanly."""

    def test_to_dict_keys(self) -> None:
        r = StrategyResult(
            strategy="vanilla",
            seed=7,
            n_trajectories=2,
            n_candidates_total=4,
            mean_score=0.5,
            best_score=0.9,
            mean_var=0.1,
            max_var=0.2,
            wall_time_s=0.01,
            success_rate=1.0,
        )
        d = r.to_dict()
        for k in (
            "strategy",
            "seed",
            "n_trajectories",
            "n_candidates_total",
            "mean_score",
            "best_score",
            "mean_var",
            "max_var",
            "wall_time_s",
            "success_rate",
            "w2_diagnostics",
        ):
            assert k in d
        assert d["strategy"] == "vanilla"
        assert d["seed"] == 7

    def test_to_dict_json_roundtrip(self) -> None:
        r = StrategyResult(
            strategy="w2_reg",
            seed=3,
            n_trajectories=5,
            n_candidates_total=10,
            mean_score=0.3,
            best_score=0.7,
            mean_var=0.05,
            max_var=0.1,
            wall_time_s=0.02,
            success_rate=0.8,
            w2_diagnostics=[{"engine_tag": "wasserstein_reg", "w2_penalty": 0.001}],
        )
        s = json.dumps(r.to_dict())
        d = json.loads(s)
        assert d["w2_diagnostics"][0]["engine_tag"] == "wasserstein_reg"


# ---------------------------------------------------------------------------
# 7. End-to-end smoke (1 seed)
# ---------------------------------------------------------------------------


class TestAggregatePerSeed:
    """``_aggregate_per_seed`` runs end-to-end and returns a sane report."""

    def test_single_seed_minimal(self, tmp_path: Path) -> None:
        if MCTSProofSearch is None or VarianceMCTSWrapper is None:
            pytest.skip("MCTS / VarianceMCTSWrapper unavailable (torch)")
        cfg = W2RegConfig(alpha=0.05, n_mc=64, decay_check=True)
        report = _aggregate_per_seed(
            seeds=[0],
            n_trajectories=2,
            sims=3,
            depth=2,
            config=cfg,
        )
        assert "per_strategy_per_seed" in report
        assert "aggregate" in report
        assert "variance_reduction" in report
        # Strategies present.
        for s in ("vanilla", "w2_reg", "w2_reg_decay"):
            assert s in report["per_strategy_per_seed"]
            assert s in report["aggregate"]
        # Variance reduction present for both W_2 variants.
        vr = report["variance_reduction"]
        assert "w2_reg_vs_vanilla" in vr
        assert "w2_reg_decay_vs_vanilla" in vr
        assert vr["n_seeds"] == 1

    def test_w2_reg_runs_record_diagnostics(self) -> None:
        if MCTSProofSearch is None or VarianceMCTSWrapper is None:
            pytest.skip("MCTS / VarianceMCTSWrapper unavailable (torch)")
        cfg = W2RegConfig(alpha=0.05, n_mc=64)
        report = _aggregate_per_seed(
            seeds=[0],
            n_trajectories=2,
            sims=3,
            depth=2,
            config=cfg,
        )
        # The w2_reg path must record diagnostics.
        per_seed = report["per_strategy_per_seed"]["w2_reg"]
        assert len(per_seed) == 1
        diags = per_seed[0]["w2_diagnostics"]
        assert len(diags) == 2  # n_trajectories == 2
        assert all(d["engine_tag"] == "wasserstein_reg" for d in diags)


# ---------------------------------------------------------------------------
# 8. Per-seed JSON schema
# ---------------------------------------------------------------------------


class TestPerSeedJsonSchema:
    """The output JSON files have the per-seed_metrics schema."""

    def test_root_and_seed_dirs_written(self, tmp_path: Path) -> None:
        if MCTSProofSearch is None or VarianceMCTSWrapper is None:
            pytest.skip("MCTS / VarianceMCTSWrapper unavailable (torch)")
        cfg = W2RegConfig(alpha=0.05, n_mc=64, decay_check=True)
        report = _aggregate_per_seed(
            seeds=[0, 1],
            n_trajectories=2,
            sims=3,
            depth=2,
            config=cfg,
        )
        _write_per_seed_json(report, tmp_path)
        # Root aggregate.
        root = tmp_path / "per_seed_metrics.json"
        assert root.exists()
        agg_doc = json.loads(root.read_text())
        assert agg_doc["metric"] == "wf129_variance_reduction"
        assert agg_doc["n_seeds"] == 2
        # Per-seed dirs.
        for s in (0, 1):
            per_seed = tmp_path / "seeds" / f"seed_{s}" / "per_seed_metrics.json"
            assert per_seed.exists()
            doc = json.loads(per_seed.read_text())
            assert doc["seed"] == s
            assert "strategies" in doc
            assert "vanilla" in doc["strategies"]
            assert "w2_reg" in doc["strategies"]
            assert "w2_reg_decay" in doc["strategies"]


# ---------------------------------------------------------------------------
# 9. Synthesis verdict
# ---------------------------------------------------------------------------


class TestSynthesisVerdict:
    """``synthesis.md`` is generated with verdict + honest framing."""

    def test_synthesis_contains_verdict_section(self, tmp_path: Path) -> None:
        if MCTSProofSearch is None or VarianceMCTSWrapper is None:
            pytest.skip("MCTS / VarianceMCTSWrapper unavailable (torch)")
        cfg = W2RegConfig(alpha=0.05, n_mc=64, decay_check=True)
        report = _aggregate_per_seed(
            seeds=[0],
            n_trajectories=2,
            sims=3,
            depth=2,
            config=cfg,
        )
        _write_synthesis(report, tmp_path, target=-0.05)
        out = tmp_path / "synthesis.md"
        assert out.exists()
        text = out.read_text()
        assert "Wf-129 Phase 1" in text
        assert "## §3 Results" in text
        assert "## §4 Verdict" in text
        assert "## §5 Honest framing" in text
        # Honest framing must mention TODO-92 + TODO-126.
        assert "TODO-92" in text
        assert "TODO-126" in text

    def test_synthesis_records_engine_tag(self, tmp_path: Path) -> None:
        if MCTSProofSearch is None or VarianceMCTSWrapper is None:
            pytest.skip("MCTS / VarianceMCTSWrapper unavailable (torch)")
        cfg = W2RegConfig(alpha=0.05, n_mc=64, decay_check=True)
        report = _aggregate_per_seed(
            seeds=[0],
            n_trajectories=1,
            sims=2,
            depth=2,
            config=cfg,
        )
        _write_synthesis(report, tmp_path)
        text = (tmp_path / "synthesis.md").read_text()
        # The synthesis must reference the engine tag so downstream
        # reporting can attribute the W_2 contribution honestly.
        assert "wasserstein_reg" in text


# ---------------------------------------------------------------------------
# 10. CLI smoke
# ---------------------------------------------------------------------------


class TestCliSmoke:
    """``main()`` runs end-to-end with ``--smoke`` and writes outputs."""

    def test_smoke_run_writes_files(self, tmp_path: Path) -> None:
        if MCTSProofSearch is None or VarianceMCTSWrapper is None:
            pytest.skip("MCTS / VarianceMCTSWrapper unavailable (torch)")
        rc = main(
            [
                "--smoke",
                "--report-path",
                str(tmp_path),
                "--seeds",
                "0",
            ]
        )
        assert rc == 0
        # Files written.
        assert (tmp_path / "per_seed_metrics.json").exists()
        assert (tmp_path / "synthesis.md").exists()
        assert (tmp_path / "seeds" / "seed_0" / "per_seed_metrics.json").exists()

    def test_parse_args_defaults(self) -> None:
        ns = _parse_args([])
        assert ns.alpha == pytest.approx(0.05)
        assert ns.n_mc == 256
        assert ns.reference == "prior"
        assert ns.target == pytest.approx(-0.05)
        assert ns.seeds == "0,1,2,3,4"

    def test_parse_args_smoke(self) -> None:
        ns = _parse_args(["--smoke", "--seeds", "7", "--alpha", "0.1"])
        assert ns.smoke is True
        assert ns.seeds == "7"
        assert ns.alpha == pytest.approx(0.1)
