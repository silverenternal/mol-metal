"""Tests for the Wf-40 Phase 3 per-node MCTS tree-stats bench.

================================================================
Contract + smoke + vulnerability tests for
``bench_mcts_tree_stats.py``
================================================================
This test suite validates the Wf-40 tree-stats bench on real data
(PlatinAI 5K + MetalCytoToxDB) in CPU-only mode.  Two strategies are
exercised end to end:

* ``proof_search``  — KataGo-style MCTS with the R17 default
                       Dirichlet root noise.
* ``variance_mcts``  — :class:`VarianceMCTSWrapper` calling
                       ``proof_search`` underneath and adding
                       variance-controlled PUCT.

The 15 tests cover 5 groups:

1. **Module surface**       — script exists, imports cleanly, version
                              constant, public API.
2. **Defaults**              — CLI defaults match the Wf-40 spec.
3. **Loaders**               — PlatinAI + MetalCytoToxDB loaders return
                              the expected shape.
4. **Per-tree extraction**   — tree walker yields the expected stats
                              (visit_count, value, prior, depth).
5. **Vulnerability scan**    — 3-vuln verdict logic (D: tree imbalance,
                              E: depth starvation, F: variance void).

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_mcts_tree_stats.py -v
"""

from __future__ import annotations

import importlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "molmetal" / "scripts" / "bench_mcts_tree_stats.py"
PLATINAI_CSV = REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
METALCYTO_CSV = REPO_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_bench_module() -> Any:
    """Import :mod:`bench_mcts_tree_stats` dynamically.

    Tries the package import first, falls back to a flat import (so
    the tests work whether the package is installed or just
    ``PYTHONPATH``-mounted).
    """
    try:
        return importlib.import_module("molmetal.scripts.bench_mcts_tree_stats")
    except Exception:
        sys.path.insert(
            0,
            str(REPO_ROOT / "molmetal" / "scripts"),
        )
        try:
            return importlib.import_module("bench_mcts_tree_stats")
        except Exception as exc:
            pytest.skip(f"bench_mcts_tree_stats not importable: {exc}")
            raise  # unreachable


def _run_cli_smoke(tmp_path: Path, *args: str) -> Dict[str, Any]:
    """Run the bench CLI in smoke mode and return the parsed JSON."""
    out_dir = tmp_path / "wf40_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        str(SCRIPT_PATH),
        "--smoke",
        "--n-trees",
        "2",
        "--n-simulations",
        "20",
        "--max-depth",
        "2",
        "--seeds",
        "0",
        "1",
        "--out-dir",
        str(out_dir),
        "--skip-metalcyto",
    ]
    cmd.extend(args)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        pytest.fail(
            f"CLI failed (rc={proc.returncode}): stderr={proc.stderr!r}",
        )
    json_path = out_dir / "tree_stats.json"
    assert json_path.exists(), f"expected JSON at {json_path}"
    return json.loads(json_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    """Module surface — script exists, imports cleanly, public API."""

    def test_script_path_exists(self) -> None:
        assert SCRIPT_PATH.exists(), f"missing: {SCRIPT_PATH}"
        assert SCRIPT_PATH.suffix == ".py"

    def test_module_imports_cleanly(self) -> None:
        mod = _import_bench_module()
        assert mod is not None
        assert hasattr(mod, "__version__")
        assert isinstance(mod.__version__, str)
        assert mod.__version__.startswith("0.1.0-wf40")

    def test_public_api_exposed(self) -> None:
        mod = _import_bench_module()
        for name in (
            "Wf40Error",
            "BenchConfig",
            "_build_argparser",
            "run_bench",
            "_cli_main",
            "_run_tree",
            "_aggregate",
            "_vulnerability_scan",
            "_walk_tree",
        ):
            assert hasattr(mod, name), f"missing public API: {name}"

    def test_v40_error_is_value_error(self) -> None:
        mod = _import_bench_module()
        assert issubclass(mod.Wf40Error, ValueError)

    def test_loaders_present(self) -> None:
        mod = _import_bench_module()
        assert callable(getattr(mod, "_load_platinai", None))
        assert callable(getattr(mod, "_load_metalcytotox", None))


# ---------------------------------------------------------------------------
# 2. Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    """CLI defaults match the Wf-40 spec."""

    def test_default_n_trees(self) -> None:
        mod = _import_bench_module()
        assert int(getattr(mod, "DEFAULT_N_TREES")) == 50

    def test_default_n_simulations(self) -> None:
        mod = _import_bench_module()
        assert int(getattr(mod, "DEFAULT_N_SIMULATIONS")) == 1000

    def test_default_max_depth(self) -> None:
        mod = _import_bench_module()
        assert int(getattr(mod, "DEFAULT_MAX_DEPTH")) == 3

    def test_argparser_runs(self) -> None:
        mod = _import_bench_module()
        parser = mod._build_argparser()
        # Help-exit (SystemExit 0) means argparse is happy.
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

    def test_argparser_accepts_all_spec_flags(self) -> None:
        mod = _import_bench_module()
        parser = mod._build_argparser()
        args = parser.parse_args(
            [
                "--n-trees",
                "10",
                "--n-simulations",
                "100",
                "--max-depth",
                "2",
                "--seeds",
                "0",
                "1",
                "--c-puct",
                "1.0",
                "--c-var",
                "0.3",
                "--smoke",
            ]
        )
        assert int(args.n_trees) == 10
        assert int(args.n_simulations) == 100
        assert int(args.max_depth) == 2
        assert [int(s) for s in args.seeds] == [0, 1]
        assert abs(float(args.c_puct) - 1.0) < 1.0e-9
        assert abs(float(args.c_var) - 0.3) < 1.0e-9
        assert bool(args.smoke)


# ---------------------------------------------------------------------------
# 3. Loaders
# ---------------------------------------------------------------------------


class TestLoaders:
    """PlatinAI + MetalCytoToxDB loaders return the expected shape."""

    def test_platinai_loader_shape(self) -> None:
        if not PLATINAI_CSV.exists():
            pytest.skip(f"platinai csv missing: {PLATINAI_CSV}")
        mod = _import_bench_module()
        rows = mod._load_platinai(
            str(PLATINAI_CSV),
            n_samples=3,
            seed=42,
        )
        assert isinstance(rows, list)
        assert len(rows) == 3
        for row in rows:
            assert "smiles" in row and isinstance(row["smiles"], str)
            assert row["smiles"], "smiles should be non-empty"
            assert "metal" in row
            assert "row_idx" in row

    def test_platinai_loader_missing_raises(self) -> None:
        mod = _import_bench_module()
        with pytest.raises(mod.Wf40Error):
            mod._load_platinai(
                "/nonexistent/platinai.csv",
                n_samples=2,
                seed=0,
            )

    def test_metalcytotox_loader_shape_or_skip(self) -> None:
        if not METALCYTO_CSV.exists():
            pytest.skip(f"metalcyto csv missing: {METALCYTO_CSV}")
        mod = _import_bench_module()
        try:
            rows = mod._load_metalcytotox(
                str(METALCYTO_CSV),
                n_samples=2,
                seed=42,
                metal_whitelist=("Pt",),
            )
        except mod.Wf40Error:
            pytest.skip("metalcyto csv unreadable in this env")
        assert isinstance(rows, list)
        assert len(rows) <= 2
        for row in rows:
            assert "smiles" in row and isinstance(row["smiles"], str)
            assert row.get("metal") == "Pt"


# ---------------------------------------------------------------------------
# 4. Per-tree extraction
# ---------------------------------------------------------------------------


class TestPerTreeExtraction:
    """Per-tree extraction yields the expected stats."""

    def test_per_tree_record_shape(self, tmp_path: Path) -> None:
        report = _run_cli_smoke(tmp_path)
        records = list(report["per_tree_records"]["proof_search"])
        assert len(records) > 0, "no per-tree records in smoke run"
        for rec in records:
            assert "strategy" in rec
            assert "seed" in rec
            assert "start_smiles" in rec
            assert "sims" in rec
            assert "max_depth" in rec
            assert "dataset" in rec
            if rec.get("error"):
                # Failed trees still carry a record but may omit
                # ``wall_time_s`` / ``aggregate`` (the parse fails
                # before the bench even instantiates the MCTS).  This
                # is the expected outcome when RDKit is missing or
                # the SMILES contains dative-bond notation that the
                # closed-term parser rejects (mirrors the Wf-39
                # behaviour on the same dataset).
                continue
            assert "wall_time_s" in rec
            agg = rec["aggregate"]
            assert "n_nodes" in agg and agg["n_nodes"] >= 1
            assert "max_depth" in agg and agg["max_depth"] >= 0
            assert "n_leaves" in agg and agg["n_leaves"] >= 1
            assert "mean_visit_count" in agg
            assert agg["mean_visit_count"] >= 0.0
            assert "max_visit_count" in agg and agg["max_visit_count"] >= 1
            assert "mean_value" in agg
            assert "mean_prior" in agg
            assert "leaf_value_var" in agg and agg["leaf_value_var"] >= 0.0
            assert "leaf_value_std" in agg and agg["leaf_value_std"] >= 0.0
            assert "n_simulations" in agg and agg["n_simulations"] >= 1

    def test_non_negative_tree_stats(self, tmp_path: Path) -> None:
        report = _run_cli_smoke(tmp_path)
        for strategy in ("proof_search", "variance_mcts"):
            for rec in report["per_tree_records"][strategy]:
                if rec.get("error"):
                    continue
                agg = rec["aggregate"]
                assert agg["n_nodes"] >= 1
                assert agg["max_visit_count"] >= 1
                assert agg["leaf_value_var"] >= 0.0
                assert agg["leaf_value_std"] >= 0.0
                assert agg["mean_prior"] >= 0.0

    def test_per_strategy_aggregate_present(self, tmp_path: Path) -> None:
        report = _run_cli_smoke(tmp_path)
        for strategy in ("proof_search", "variance_mcts"):
            agg = dict(report["per_strategy"][strategy])
            assert "n_trees" in agg
            assert "n_ok" in agg
            assert "success_rate" in agg
            assert "mean_n_nodes" in agg
            assert "mean_max_depth" in agg
            assert "mean_leaf_value_var" in agg
            assert "p50_visit_count" in agg
            assert "p90_visit_count" in agg
            assert "error_modes" in agg


# ---------------------------------------------------------------------------
# 5. Vulnerability scan
# ---------------------------------------------------------------------------


class TestVulnerabilityScan:
    """3-vuln verdict logic (D: tree imbalance, E: depth starvation, F: variance void)."""

    def test_vuln_keys_present(self, tmp_path: Path) -> None:
        report = _run_cli_smoke(tmp_path)
        vuln = report["vulnerability_scan"]
        for key in ("vuln_d", "vuln_e", "vuln_f", "vuln_g"):
            assert key in vuln, f"missing vuln key: {key}"
            assert "verdict" in vuln[key]
            assert vuln[key]["verdict"] in ("OK", "WARN", "CRITICAL")

    def test_vuln_d_imbalance_ratios_present(self, tmp_path: Path) -> None:
        report = _run_cli_smoke(tmp_path)
        block = report["vulnerability_scan"]["vuln_d"]
        assert "imbalance_ratio_proof_search" in block
        assert "imbalance_ratio_variance_mcts" in block
        assert block["imbalance_ratio_proof_search"] >= 0.0
        assert block["imbalance_ratio_variance_mcts"] >= 0.0

    def test_vuln_e_depth_ratios_present(self, tmp_path: Path) -> None:
        report = _run_cli_smoke(tmp_path)
        block = report["vulnerability_scan"]["vuln_e"]
        assert "depth_ratio_proof_search" in block
        assert "depth_ratio_variance_mcts" in block
        # Cap is 2 in smoke mode.
        assert 0.0 <= block["depth_ratio_proof_search"] <= 1.0
        assert 0.0 <= block["depth_ratio_variance_mcts"] <= 1.0

    def test_vuln_f_sigma_fractions_present(self, tmp_path: Path) -> None:
        report = _run_cli_smoke(tmp_path)
        block = report["vulnerability_scan"]["vuln_f"]
        assert "sigma_fraction_proof_search" in block
        assert "sigma_fraction_variance_mcts" in block
        assert block["sigma_fraction_proof_search"] >= 0.0
        assert block["sigma_fraction_variance_mcts"] >= 0.0

    def test_vuln_g_replay_starvation_present(self, tmp_path: Path) -> None:
        report = _run_cli_smoke(tmp_path)
        block = report["vulnerability_scan"]["vuln_g"]
        assert "replay_starvation_rate_variance_mcts" in block
        assert 0.0 <= block["replay_starvation_rate_variance_mcts"] <= 1.0


# ---------------------------------------------------------------------------
# 6. End-to-end smoke
# ---------------------------------------------------------------------------


class TestEndToEndSmoke:
    """Smoke harness — full CLI on real PlatinAI data."""

    def test_cli_writes_json(self, tmp_path: Path) -> None:
        if not PLATINAI_CSV.exists():
            pytest.skip(f"platinai csv missing: {PLATINAI_CSV}")
        report = _run_cli_smoke(tmp_path)
        # Validate the schema_version is set.
        assert int(report["schema_version"]) == 1
        # Validate the tool identity.
        assert report["tool"] == "bench_mcts_tree_stats"
        assert report["tool_version"].startswith("0.1.0-wf40")
        # Validate the config echoes what we passed on the CLI.
        cfg = report["config"]
        assert cfg["smoke"] is True
        assert cfg["seeds"] == [0, 1]
        # Validate at least one tree ran per strategy.  n_trees >= 1
        # is the structural contract; n_ok may be 0 when RDKit is
        # missing or the dataset SMILES contain dative-bond notation
        # that the closed-term parser rejects — the Wf-39 bench
        # exhibits the same behaviour.
        for strategy in ("proof_search", "variance_mcts"):
            assert (
                report["per_strategy"][strategy]["n_trees"] >= 1
            ), f"no trees ran for strategy {strategy}"
