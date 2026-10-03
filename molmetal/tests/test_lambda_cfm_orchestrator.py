"""Tests for Wf-40 Phase 4 — Lambda-CFM pipeline orchestrator.

================================================================
Wf-40 Phase 4 test suite (15 tests — L9 final consolidation).
================================================================
Verifies:

* Helpers (``_utcnow_iso``, ``_parse_seeds``, ``_parse_phases``,
  ``_read_csv_column``, ``_memory_mb``) — 3 tests.
* ``LambdaCFMPipeline`` step methods (closure, astsr, hilbert,
  coupling, cfm, mcts) — 6 tests, each verifies the step emits the
  expected keys and degrades gracefully.
* ``LambdaCFMPipeline.run_full_pipeline`` orchestration — 3 tests
  (default phases, custom phases, full ``all`` expansion).
* ``LambdaCFMBenchmark.run`` multi-seed run + JSON persistence — 2 tests.
* ``run_orchestrator`` CLI round-trip — 1 test.

Each test is independent and CPU-only; the orchestrator's "use_*"
flags are toggled so the suite runs in < 5 s on a minimal host.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Repo-root path injection so the module is importable when invoked
# from any CWD (matches the convention in every test file).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.orchestrate_lambda_cfm_pipeline import (  # noqa: E402
    DEFAULT_CLOSURE_MAX_DEPTH,
    DEFAULT_POCKET_SIZE,
    LambdaCFMBenchmark,
    LambdaCFMPipeline,
    REPORT_DIR,
    _memory_mb,
    _parse_phases,
    _parse_seeds,
    _read_csv_column,
    _utcnow_iso,
    run_orchestrator,
)


# ---------------------------------------------------------------------------
# Helpers — 3 tests
# ---------------------------------------------------------------------------
class TestHelpers:
    """Helper-function smoke tests."""

    def test_utcnow_iso_isoformat(self) -> None:
        """``_utcnow_iso`` returns an ISO-8601 UTC string."""
        s = _utcnow_iso()
        assert isinstance(s, str)
        # Format: 2026-09-26T12:34:56Z
        assert s.endswith("Z")
        assert "T" in s
        assert len(s) == 20

    def test_parse_seeds_csv(self) -> None:
        """``_parse_seeds`` parses comma-separated seeds."""
        assert _parse_seeds("0,1,2") == [0, 1, 2]
        assert _parse_seeds("42") == [42]
        assert _parse_seeds(" 7 , 11 , 13 ") == [7, 11, 13]
        assert _parse_seeds("") == []

    def test_parse_phases_all_and_subset(self) -> None:
        """``_parse_phases`` handles ``all`` and custom subsets."""
        all_phases = _parse_phases("all")
        assert all_phases == [
            "closure",
            "astsr",
            "hilbert",
            "coupling",
            "cfm",
            "mcts",
        ]
        subset = _parse_phases("closure,hilbert,cfm")
        assert subset == ["closure", "hilbert", "cfm"]
        with pytest.raises(ValueError):
            _parse_phases("bogus_phase")


class TestCsvReader:
    """Verify ``_read_csv_column`` graceful degradation."""

    def test_missing_csv_returns_empty(self, tmp_path: Path) -> None:
        """Missing file → empty list."""
        assert _read_csv_column(tmp_path / "nope.csv") == []

    def test_csv_with_smiles(self, tmp_path: Path) -> None:
        """Reads the ``smiles`` column from a well-formed CSV."""
        csv_path = tmp_path / "x.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["smiles", "other"])
            w.writeheader()
            w.writerow({"smiles": "CCN", "other": "x"})
            w.writerow({"smiles": "  ", "other": "y"})  # blank
            w.writerow({"smiles": "C#C", "other": "z"})
        rows = _read_csv_column(csv_path, "smiles")
        assert rows == ["CCN", "C#C"]


class TestMemoryHelper:
    """Verify ``_memory_mb`` byte → MB conversion."""

    def test_memory_mb_bytes_to_mb(self) -> None:
        """Bytes → MB conversion."""
        assert _memory_mb(0) == 0.0
        assert abs(_memory_mb(1024 * 1024) - 1.0) < 1e-9
        assert abs(_memory_mb(2 * 1024 * 1024) - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# LambdaCFMPipeline step methods — 6 tests
# ---------------------------------------------------------------------------
class TestPipelineSteps:
    """Each step method runs and emits the expected contract."""

    def _make_pipe(self, **kw) -> LambdaCFMPipeline:
        return LambdaCFMPipeline(seed=0, **kw)

    def test_step1_closure_contract(self) -> None:
        """Step 1 returns ``status``, ``n_reachable``, ``witness_path_len``."""
        pipe = self._make_pipe()
        out = pipe.step1_closure("CCN=[N+]=[N-]")
        assert "status" in out
        assert "n_reachable" in out
        assert "witness_path_len" in out
        # When RDKit is missing the step is "skipped:<Exc>"; otherwise "ok".
        assert out["status"] in {"ok"} or out["status"].startswith("skipped:")

    def test_step2_astsr_contract(self) -> None:
        """Step 2 returns ``status`` + ``r2_in_sample``."""
        pipe = self._make_pipe(astr_pop=8)
        out = pipe.step2_lambda_astsr(n_features=5)
        assert "status" in out
        assert "r2_in_sample" in out
        assert out["status"] in {"ok"} or out["status"].startswith("skipped:")

    def test_step3_hilbert_contract(self) -> None:
        """Step 3 returns ``status`` + ``final_error`` + ``emp_convergence_rate``."""
        pipe = self._make_pipe(hilbert_iters=8)
        out = pipe.step3_hilbert(n_iters=8)
        assert "status" in out
        assert "final_error" in out
        assert "emp_convergence_rate" in out

    def test_step4_coupling_contract(self) -> None:
        """Step 4 returns ``status`` + ``embed_mean`` + ``embed_std``."""
        pipe = self._make_pipe()
        out = pipe.step4_coupling()
        assert "status" in out
        assert "embed_mean" in out
        assert "embed_std" in out

    def test_step5_cfm_decode_contract(self) -> None:
        """Step 5 returns ``status`` + ``terminal_rmse`` + ``path_length``."""
        pipe = self._make_pipe(cfm_n_steps=4)
        out = pipe.step5_cfm_decode(n_atoms=4, n_steps=4)
        assert "status" in out
        assert "terminal_rmse" in out
        assert "path_length" in out
        # Closed-form linear path: terminal RMSE must be exactly 0.
        if out["status"] == "ok":
            assert out["terminal_rmse"] == 0.0

    def test_step6_mcts_contract(self) -> None:
        """Step 6 returns ``status`` + ``selected_action`` + ``puct_score_max``."""
        pipe = self._make_pipe(mcts_n_actions=4)
        out = pipe.step6_mcts_select(n_actions=4)
        assert "status" in out
        assert "selected_action" in out
        assert "puct_score_max" in out
        if out["status"] == "ok":
            assert 0 <= out["selected_action"] < 4


# ---------------------------------------------------------------------------
# LambdaCFMPipeline.run_full_pipeline orchestration — 3 tests
# ---------------------------------------------------------------------------
class TestFullPipeline:
    """End-to-end pipeline orchestration contracts."""

    def test_full_pipeline_all_phases(self) -> None:
        """Default phases are all six."""
        pipe = LambdaCFMPipeline(seed=1)
        rep = pipe.run_full_pipeline()
        assert rep["phases"] == [
            "closure",
            "astsr",
            "hilbert",
            "coupling",
            "cfm",
            "mcts",
        ]
        assert len(rep["steps"]) == 6
        assert "wall_time_per_step_s" in rep
        assert "memory_per_step_mb" in rep
        assert rep["n_steps_total"] == 6
        assert 0 <= rep["n_steps_ok"] <= 6

    def test_full_pipeline_custom_phases(self) -> None:
        """Custom phase subset respects the user's order."""
        pipe = LambdaCFMPipeline(seed=2)
        rep = pipe.run_full_pipeline(
            phases=("cfm", "mcts", "closure"),
        )
        assert rep["phases"] == ["cfm", "mcts", "closure"]
        assert set(rep["steps"].keys()) == {"cfm", "mcts", "closure"}
        assert rep["n_steps_total"] == 3

    def test_full_pipeline_disabled_flags_skip_steps(self) -> None:
        """Disabled flags record ``skipped`` status with reason."""
        pipe = LambdaCFMPipeline(
            seed=3,
            use_closure=False,
            use_astsr=False,
            use_hilbert=False,
            use_coupling=False,
            use_cfm=False,
            use_mcts=False,
        )
        rep = pipe.run_full_pipeline()
        assert rep["n_steps_ok"] == 0
        for name, step in rep["steps"].items():
            assert step["status"] == "skipped"
            assert step.get("reason", "").endswith("=False")


# ---------------------------------------------------------------------------
# LambdaCFMBenchmark.run — 2 tests
# ---------------------------------------------------------------------------
class TestBenchmark:
    """Multi-seed benchmark JSON persistence."""

    def test_benchmark_writes_json(self, tmp_path: Path) -> None:
        """Benchmark writes the canonical JSON report to disk."""
        out_dir = tmp_path / "wf40_orch"
        bench = LambdaCFMBenchmark(
            n_samples=2,
            seeds=(0, 1),
            phases=("cfm", "mcts"),
            out_dir=out_dir,
        )
        rep = bench.run()
        out_path = out_dir / "orchestrator_bench.json"
        assert out_path.exists()
        with out_path.open() as fh:
            data = json.load(fh)
        assert data["n_samples"] == 2
        assert data["seeds"] == [0, 1]
        assert data["phases"] == ["cfm", "mcts"]
        assert len(data["per_seed"]) == 2
        for s in data["per_seed"]:
            assert s["n_samples"] == 2
            assert len(s["per_sample"]) == 2
            assert s["n_steps_total"] == 4  # 2 samples × 2 phases
        assert data["version"] == "wf40-phase4-v1"

    def test_benchmark_cfm_path_length_nonzero(self, tmp_path: Path) -> None:
        """CFM step records a positive path length when active."""
        out_dir = tmp_path / "wf40_orch_cfm"
        bench = LambdaCFMBenchmark(
            n_samples=1,
            seeds=(0,),
            phases=("cfm",),
            out_dir=out_dir,
        )
        rep = bench.run()
        cfm_step = rep["per_seed"][0]["per_sample"][0]["steps"]["cfm"]
        if cfm_step["status"] == "ok":
            assert cfm_step["path_length"] > 0.0


# ---------------------------------------------------------------------------
# run_orchestrator CLI — 1 test
# ---------------------------------------------------------------------------
class TestCli:
    """CLI round-trip via :func:`run_orchestrator`."""

    def test_run_orchestrator_round_trip(self, tmp_path: Path) -> None:
        """CLI persists the same report shape as :class:`LambdaCFMBenchmark`."""
        out_dir = tmp_path / "wf40_cli"
        rep = run_orchestrator(
            [
                "--n-samples",
                "1",
                "--seeds",
                "0,1",
                "--phases",
                "cfm,mcts",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert rep["n_samples"] == 1
        assert rep["seeds"] == [0, 1]
        assert rep["phases"] == ["cfm", "mcts"]
        # Persisted JSON exists.
        out_path = out_dir / "orchestrator_bench.json"
        assert out_path.exists()
        # Top-level keys are the canonical report keys.
        assert "total_wall_time_s" in rep
        assert "n_steps_ok_total" in rep
        assert "n_steps_total" in rep
        assert "timestamp_utc" in rep
