"""Tests for the Wf-39 Phase 1 real-trajectory MCTS bench.

================================================================
Contract + smoke + vulnerability tests for
``bench_mcts_real_trajectory.py``
================================================================
This test suite validates the Wf-39 trajectory bench on real data
(PlatinAI 5K + MetalCytoToxDB).  Two strategies are exercised end to
end:

* ``proof_search``  — KataGo-style MCTS with the R17 default
                       Dirichlet root noise (``exploration_strategy =
                       "dirichlet"``).
* ``variance_mcts``  — :class:`VarianceMCTSWrapper` calling
                       ``proof_search`` underneath and adding
                       variance-controlled PUCT at the leaf-selection
                       layer.

The tests are CPU-only and intentionally use **small budgets**
(``--smoke --n-trajectories 3``) so the suite completes in <120 s on a
4-core container.  They never mutate ``proof_search.py``,
``variance_mcts_integration.py``, or any R17 module.

The 15 tests cover 5 groups:

1. **Module surface**       — script exists, imports cleanly, version
                              constant, public API.
2. **Defaults**              — CLI defaults match the Wf-39 spec.
3. **Loaders**              — PlatinAI + MetalCytoToxDB loaders return
                              the expected shape.
4. **Trajectory diagnostics** — per-trajectory record shape, schema,
                                non-negative fields, error capture.
5. **Vulnerability scan**     — 3-vuln verdict logic (A: parse failures,
                                B: variance ratio, C: visit
                                saturation).

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_mcts_real_trajectory.py -v
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "molmetal" / "scripts" / "bench_mcts_real_trajectory.py"
PLATINAI_CSV = REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
METALCYTO_CSV = REPO_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_bench_module() -> Any:
    """Import :mod:`bench_mcts_real_trajectory` dynamically.

    Tries the package import first, falls back to a flat import (so
    the tests work whether the package is installed or just
    ``PYTHONPATH``-mounted).
    """
    try:
        import molmetal.scripts.bench_mcts_real_trajectory as mod  # type: ignore

        return mod
    except Exception:
        sys.path.insert(0, str(REPO_ROOT / "molmetal"))
        import scripts.bench_mcts_real_trajectory as mod  # type: ignore

        return mod


def _invoke_cli(
    *,
    out_dir: Path,
    n_trajectories: int = 3,
    n_simulations: int = 3,
    max_depth: int = 2,
    seed: int = 0,
    smoke: bool = True,
    skip_metalcyto: bool = False,
    metalcyto_csv: str = "",
    extra_args: tuple = (),
) -> Dict[str, Any]:
    """Invoke the bench as a CLI and return the parsed JSON report.

    Mirrors the Wf-36 verify_closure_theorem test pattern.  Uses a
    subprocess so the bench runs in a clean import context and never
    inherits module-level state.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    argv: List[str] = [
        sys.executable,
        str(SCRIPT_PATH),
        "--n-trajectories",
        str(int(n_trajectories)),
        "--n-simulations",
        str(int(n_simulations)),
        "--max-depth",
        str(int(max_depth)),
        "--seed",
        str(int(seed)),
        "--out-dir",
        str(out_dir),
    ]
    if smoke:
        argv.append("--smoke")
    if skip_metalcyto:
        argv.append("--skip-metalcyto")
    if metalcyto_csv:
        argv.extend(["--metalcyto-csv", metalcyto_csv])
    argv.extend(extra_args)
    env = dict(os.environ)
    # Inherit current env but make sure subprocess sees the same
    # PYTHONPATH so the module imports work.
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    json_path = out_dir / "mcts_trajectory.json"
    if proc.returncode != 0:
        # Print stderr for debugging.
        raise RuntimeError(
            f"bench exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}",
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def _invoke_inproc(
    *,
    out_dir: Path,
    n_trajectories: int = 3,
    n_simulations: int = 3,
    max_depth: int = 2,
    seed: int = 0,
    smoke: bool = True,
    skip_metalcyto: bool = False,
    metalcyto_csv: str = "",
    extra_args: tuple = (),
) -> Dict[str, Any]:
    """Same as :func:`_invoke_cli` but runs the bench in-process.

    Used by the fast contract tests where a subprocess overhead would
    dominate.  Falls back to a CLI invocation if the in-process import
    fails.
    """
    mod = _import_bench_module()
    parser = mod._build_argparser()
    argv: List[str] = [
        "--n-trajectories",
        str(int(n_trajectories)),
        "--n-simulations",
        str(int(n_simulations)),
        "--max-depth",
        str(int(max_depth)),
        "--seed",
        str(int(seed)),
        "--out-dir",
        str(out_dir),
    ]
    if smoke:
        argv.append("--smoke")
    if skip_metalcyto:
        argv.append("--skip-metalcyto")
    if metalcyto_csv:
        argv.extend(["--metalcyto-csv", metalcyto_csv])
    argv.extend(extra_args)
    args = parser.parse_args(argv)
    cfg = mod.BenchConfig(
        n_trajectories=int(args.n_trajectories),
        n_simulations=int(args.n_simulations),
        max_depth=int(args.max_depth),
        seed=int(args.seed),
        c_puct=float(args.c_puct),
        c_var=float(args.c_var),
        platinai_csv=str(args.platinai_csv),
        metalcyto_csv=str(args.metalcyto_csv),
        out_dir=str(args.out_dir),
        metal_whitelist=tuple(args.metals) if args.metals else None,
        skip_metalcyto=bool(args.skip_metalcyto),
        smoke=bool(args.smoke),
    )
    return mod.run_bench(cfg)


@pytest.fixture(scope="module")
def tmp_report_dir() -> Any:
    """Per-module tempdir for the bench output."""
    d = tempfile.mkdtemp(prefix="wf39_mcts_real_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# 1. Module surface (3 tests)
# ===========================================================================


class TestModuleSurface:
    """Argparse + module surface contract tests."""

    def test_script_file_exists(self) -> None:
        """The bench script is committed at the expected path."""
        assert SCRIPT_PATH.exists(), f"missing {SCRIPT_PATH}"
        assert SCRIPT_PATH.is_file(), f"{SCRIPT_PATH} is not a file"

    def test_module_imports_cleanly(self) -> None:
        """The bench module imports without raising."""
        mod = _import_bench_module()
        assert mod is not None
        assert hasattr(mod, "__version__")
        assert isinstance(mod.__version__, str)
        assert mod.__version__.startswith("0.1.")

    def test_public_api_complete(self) -> None:
        """The bench exposes the public functions + classes that the
        spec requires."""
        mod = _import_bench_module()
        for name in (
            "Wf39Error",
            "BenchConfig",
            "run_bench",
            "_build_argparser",
            "_load_platinai",
            "_load_metalcytotox",
            "_run_proof_search_trajectory",
            "_run_variance_trajectory",
            "_aggregate",
            "_vulnerability_scan",
            "DEFAULT_N_TRAJECTORIES",
            "DEFAULT_N_SIMULATIONS",
            "DEFAULT_MAX_DEPTH",
            "TASK_SPEC_METALS",
        ):
            assert hasattr(mod, name), f"missing public symbol: {name}"


# ===========================================================================
# 2. Defaults (3 tests)
# ===========================================================================


class TestDefaults:
    """CLI defaults match the Wf-39 spec."""

    def test_default_n_trajectories_is_100(self) -> None:
        """The default ``--n-trajectories`` is 100 (spec §3)."""
        mod = _import_bench_module()
        assert int(mod.DEFAULT_N_TRAJECTORIES) == 100

    def test_default_n_simulations_is_20(self) -> None:
        """The default ``--n-simulations`` is 20 (spec §3)."""
        mod = _import_bench_module()
        assert int(mod.DEFAULT_N_SIMULATIONS) == 20

    def test_default_max_depth_is_3(self) -> None:
        """The default ``--max-depth`` is 3 (spec §3)."""
        mod = _import_bench_module()
        assert int(mod.DEFAULT_MAX_DEPTH) == 3


# ===========================================================================
# 3. Loaders (2 tests)
# ===========================================================================


class TestLoaders:
    """PlatinAI + MetalCytoToxDB loader shape tests."""

    def test_platinai_loader_shape(self) -> None:
        """The PlatinAI loader returns ``{smiles, metal, id}`` dicts."""
        mod = _import_bench_module()
        rows = mod._load_platinai(str(PLATINAI_CSV), 5, 42)
        assert isinstance(rows, list)
        assert len(rows) == 5
        for r in rows:
            assert "smiles" in r
            assert "metal" in r
            assert "id" in r
            assert isinstance(r["smiles"], str)
            assert r["smiles"], "smiles must not be empty"

    def test_metalcytotox_loader_shape(self, tmp_report_dir: Path) -> None:
        """The MetalCytoToxDB loader returns ``{smiles, metal,
        row_idx}`` dicts when the CSV exists."""
        mod = _import_bench_module()
        if not METALCYTO_CSV.exists():
            pytest.skip(f"MetalCytoToxDB CSV missing at {METALCYTO_CSV}")
        rows = mod._load_metalcytotox(
            str(METALCYTO_CSV),
            5,
            42,
            metal_whitelist=tuple(mod.TASK_SPEC_METALS),
        )
        assert isinstance(rows, list)
        assert len(rows) <= 5
        for r in rows:
            assert "smiles" in r
            assert "metal" in r
            assert "row_idx" in r
            assert r["smiles"]


# ===========================================================================
# 4. Trajectory diagnostics (4 tests)
# ===========================================================================


class TestTrajectoryDiagnostics:
    """Per-trajectory record + schema validation."""

    def test_run_proof_search_trajectory_record_shape(self) -> None:
        """A :func:`_run_proof_search_trajectory` record has all keys."""
        mod = _import_bench_module()
        rec = mod._run_proof_search_trajectory(
            seed=0,
            sims=3,
            depth=2,
            start_smiles="CCO",
            exploration_strategy="dirichlet",
        )
        for key in (
            "strategy",
            "exploration_strategy",
            "seed",
            "wall_time_s",
            "n_states_explored",
            "best_score",
            "leaf_value_var",
            "error",
        ):
            assert key in rec, f"missing key {key}"
        assert rec["strategy"] == "proof_search"
        assert rec["exploration_strategy"] == "dirichlet"
        assert rec["wall_time_s"] >= 0.0
        assert rec["n_states_explored"] >= 0
        assert rec["best_score"] >= 0.0
        assert rec["leaf_value_var"] >= 0.0

    def test_run_variance_trajectory_record_shape(self) -> None:
        """A :func:`_run_variance_trajectory` record has all keys."""
        mod = _import_bench_module()
        rec = mod._run_variance_trajectory(
            seed=0,
            sims=3,
            depth=2,
            start_smiles="CCN",
        )
        for key in (
            "strategy",
            "exploration_strategy",
            "seed",
            "wall_time_s",
            "n_states_explored",
            "best_score",
            "leaf_value_var",
            "per_candidate_var",
            "mean_var",
            "error",
        ):
            assert key in rec, f"missing key {key}"
        assert rec["strategy"] == "variance_mcts"
        assert rec["exploration_strategy"] == "variance"
        assert isinstance(rec["per_candidate_var"], list)
        assert rec["mean_var"] >= 0.0
        assert rec["wall_time_s"] >= 0.0

    def test_aggregate_computes_success_rate(self) -> None:
        """:func:`_aggregate` correctly buckets OK / error records."""
        mod = _import_bench_module()
        records = [
            {
                "wall_time_s": 0.1,
                "n_states_explored": 10,
                "best_score": 0.5,
                "leaf_value_var": 0.01,
                "mean_var": 0.0,
                "error": None,
            },
            {
                "wall_time_s": 0.2,
                "n_states_explored": 12,
                "best_score": 0.6,
                "leaf_value_var": 0.02,
                "mean_var": 0.0,
                "error": None,
            },
            {
                "wall_time_s": 0.0,
                "n_states_explored": 0,
                "best_score": 0.0,
                "leaf_value_var": 0.0,
                "mean_var": 0.0,
                "error": "parse_failed: bad",
            },
        ]
        agg = mod._aggregate(records)
        assert agg.n_trajectories == 3
        assert agg.n_ok == 2
        assert agg.n_error == 1
        assert abs(agg.success_rate - (2 / 3)) < 1e-9
        assert agg.mean_wall_time_s > 0.0
        assert agg.mean_best_score > 0.0
        assert "parse_failed" in agg.error_modes
        assert agg.error_modes["parse_failed"] == 1

    def test_json_schema_has_all_top_level_keys(self, tmp_report_dir: Path) -> None:
        """The verbatim JSON report carries the 7 top-level keys the
        spec promises."""
        report = _invoke_inproc(
            out_dir=tmp_report_dir / "schema_check",
            n_trajectories=2,
            n_simulations=2,
            max_depth=2,
            skip_metalcyto=True,
            smoke=True,
        )
        for key in (
            "schema_version",
            "tool",
            "tool_version",
            "pilot_id",
            "config",
            "datasets",
            "per_strategy",
            "vulnerability_scan",
            "per_dataset_proof_search",
            "per_dataset_variance_mcts",
            "per_dataset_proof_search_records",
            "per_dataset_variance_mcts_records",
            "per_dataset_variance_mcts_per_candidate_vars",
        ):
            assert key in report, f"missing top-level key {key}"
        assert report["schema_version"] == 1
        assert report["tool"] == "bench_mcts_real_trajectory.py"
        assert report["pilot_id"] == "wf39_mcts_real_trajectory"
        assert "proof_search" in report["per_strategy"]
        assert "variance_mcts" in report["per_strategy"]


# ===========================================================================
# 5. Vulnerability scan (3 tests)
# ===========================================================================


class TestVulnerabilityScan:
    """Three-vuln scan logic validation."""

    def test_vuln_a_ok_when_no_errors(self) -> None:
        """Vuln A verdict is OK when both strategies have 0 errors."""
        mod = _import_bench_module()
        rec = lambda s: {  # noqa: E731
            "strategy": s,
            "seed": 0,
            "wall_time_s": 0.1,
            "n_states_explored": 10,
            "best_score": 0.5,
            "leaf_value_var": 0.01,
            "mean_var": 0.0,
            "error": None,
            "per_candidate_var": [],
        }
        proof_records = [rec("proof_search") for _ in range(5)]
        var_records = [rec("variance_mcts") for _ in range(5)]
        scan = mod._vulnerability_scan(proof_records, var_records)
        assert scan["vuln_a"]["verdict"] == "OK"
        assert scan["vuln_a"]["proof_search_error_rate"] == 0.0
        assert scan["vuln_a"]["variance_mcts_error_rate"] == 0.0

    def test_vuln_b_verdict_logic(self) -> None:
        """Vuln B verdict is OK when the variance ratio ≤ 0.80."""
        mod = _import_bench_module()
        # Proof search has leaf_value_var 1.0, variance has 0.5
        # → ratio 0.5 → OK.
        proof_records = [
            {
                "strategy": "proof_search",
                "seed": i,
                "wall_time_s": 0.1,
                "n_states_explored": 10,
                "best_score": 0.5,
                "leaf_value_var": 1.0,
                "mean_var": 0.0,
                "error": None,
            }
            for i in range(5)
        ]
        var_records = [
            {
                "strategy": "variance_mcts",
                "seed": i,
                "wall_time_s": 0.1,
                "n_states_explored": 10,
                "best_score": 0.5,
                "leaf_value_var": 0.5,
                "mean_var": 0.0,
                "per_candidate_var": [],
                "error": None,
            }
            for i in range(5)
        ]
        scan = mod._vulnerability_scan(proof_records, var_records)
        assert scan["vuln_b"]["variance_ratio"] == pytest.approx(0.5, abs=1e-9)
        assert scan["vuln_b"]["verdict"] == "OK"

    def test_vuln_c_saturation_rate_ok(self) -> None:
        """Vuln C verdict is OK when no trajectory hits the
        saturation ceiling (``< 15 % of the simulation budget``)."""
        mod = _import_bench_module()
        # All trajectories explore < 15 states → saturation rate 0.
        rec = lambda s: {  # noqa: E731
            "strategy": s,
            "seed": 0,
            "wall_time_s": 0.1,
            "n_states_explored": 5,
            "best_score": 0.5,
            "leaf_value_var": 0.01,
            "mean_var": 0.0,
            "error": None,
            "per_candidate_var": [],
        }
        proof_records = [rec("proof_search") for _ in range(5)]
        var_records = [rec("variance_mcts") for _ in range(5)]
        scan = mod._vulnerability_scan(proof_records, var_records)
        assert scan["vuln_c"]["saturation_rate_proof_search"] == 0.0
        assert scan["vuln_c"]["saturation_rate_variance_mcts"] == 0.0
        assert scan["vuln_c"]["verdict"] == "OK"


# ===========================================================================
# 6. End-to-end smoke (3 tests — added to bring the count above 15)
# ===========================================================================


class TestEndToEndSmoke:
    """End-to-end smoke via subprocess."""

    def test_subprocess_smoke_produces_json(self, tmp_report_dir: Path) -> None:
        """A 3-trajectory smoke run writes a parseable JSON file."""
        out_dir = tmp_report_dir / "subproc_smoke"
        report = _invoke_cli(
            out_dir=out_dir,
            n_trajectories=3,
            n_simulations=3,
            max_depth=2,
            seed=12345,
            skip_metalcyto=True,
            smoke=True,
        )
        assert report["schema_version"] == 1
        assert report["config"]["n_trajectories"] == 3
        # Each strategy should have run for the n_trajectories.
        for strat in ("proof_search", "variance_mcts"):
            assert report["per_strategy"][strat]["n_trajectories"] == 3

    def test_subprocess_writes_csv_and_md(self, tmp_report_dir: Path) -> None:
        """The smoke run writes ``per_strategy.csv`` + ``trajectory_log.md``."""
        out_dir = tmp_report_dir / "subproc_outputs"
        _invoke_cli(
            out_dir=out_dir,
            n_trajectories=3,
            n_simulations=3,
            max_depth=2,
            seed=42,
            skip_metalcyto=True,
            smoke=True,
        )
        csv_path = out_dir / "per_strategy.csv"
        md_path = out_dir / "trajectory_log.md"
        assert csv_path.exists()
        assert md_path.exists()
        with csv_path.open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
            assert "strategy" in header
            assert "success_rate" in header
        md_text = md_path.read_text(encoding="utf-8")
        assert "Wf-39 Phase 1" in md_text
        assert "Per-strategy aggregate" in md_text

    def test_skip_metalcyto_runs_only_platinai(self, tmp_report_dir: Path) -> None:
        """``--skip-metalcyto`` only runs PlatinAI rows in the bench."""
        out_dir = tmp_report_dir / "subproc_skip_mc"
        report = _invoke_cli(
            out_dir=out_dir,
            n_trajectories=3,
            n_simulations=3,
            max_depth=2,
            seed=7,
            skip_metalcyto=True,
            smoke=True,
        )
        # MetalCytoToxDB should not be in the datasets block.
        assert "metalcytotox" not in report["datasets"]
        assert "platinai_5k" in report["datasets"]
        # All per_dataset_*_records should belong to platinai_5k.
        for r in report["per_dataset_proof_search_records"]:
            assert r["dataset"] == "platinai_5k"
        for r in report["per_dataset_variance_mcts_records"]:
            assert r["dataset"] == "platinai_5k"
