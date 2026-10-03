"""Tests for the real-data MCTS benchmark script (Wf-MCTS-Real-Bench).

================================================================
Contract + smoke tests for ``benchmark_mcts_real.py``
================================================================
The benchmark compares three MCTS strategies on real PlatinAI 5K
SMILES (R17 ship, :file:`molmetal/data/platinai_5k.csv`):

1. **base**            — KataGo-style :class:`MCTSProofSearch.search`
                         (R17 ship, ``proof_search.py``).
2. **variance_puct**   — :class:`MCTSVarianceWrapper` (Wf-11 ship,
                         ``variance_puct_integration.py``).
3. **variance_mcts**   — :class:`VarianceMCTSWrapper` (Wf-14 ship,
                         ``variance_mcts_integration.py``).

The tests below are CPU-only and intentionally use **small budgets**
(``--n-samples 3 --n-simulations 2``) so the suite completes in
<60 s on a 4-core container.  They never mutate ``proof_search.py``,
``variance_puct_integration.py`` or ``variance_mcts_integration.py``.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_mcts_real_bench.py -v
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "molmetal" / "scripts" / "benchmark_mcts_real.py"
CSV_PATH = REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"


# ---------------------------------------------------------------------------
# Shared imports (lazy — tests fail-fast on ImportError)
# ---------------------------------------------------------------------------


def _import_benchmark_module() -> Any:
    """Import the benchmark module dynamically.

    Falls back between ``molmetal.scripts.benchmark_mcts_real`` and
    ``scripts.benchmark_mcts_real`` so the tests work whether the
    package is installed or just ``PYTHONPATH``-mounted.
    """
    try:
        import molmetal.scripts.benchmark_mcts_real as mod  # type: ignore

        return mod
    except Exception:
        sys.path.insert(0, str(REPO_ROOT / "molmetal"))
        import scripts.benchmark_mcts_real as mod  # type: ignore

        return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_benchmark(
    *,
    n_samples: int = 3,
    n_simulations: int = 2,
    max_depth: int = 2,
    top_k: int = 2,
    seed: int = 0,
    out_dir: Path,
    csv_arg: str = "",
    extra_args: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Invoke the benchmark script with controlled CLI args."""
    if csv_arg:
        csv_path = csv_arg
    else:
        csv_path = str(CSV_PATH)
    argv: List[str] = [
        "--n-samples",
        str(int(n_samples)),
        "--n-simulations",
        str(int(n_simulations)),
        "--max-depth",
        str(int(max_depth)),
        "--top-k",
        str(int(top_k)),
        "--seed",
        str(int(seed)),
        "--csv",
        csv_path,
        "--out-dir",
        str(out_dir),
        "--quiet",
    ]
    argv.extend(extra_args)
    mod = _import_benchmark_module()
    parser = mod._build_argparser()
    args = parser.parse_args(argv)
    return mod.run_benchmark(args)


@pytest.fixture(scope="module")
def tmp_report_dir() -> Any:
    """Per-module tempdir for the benchmark output."""
    d = tempfile.mkdtemp(prefix="wf_mcts_real_bench_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# 1. Script-level contract — argparse + version + paths
# ===========================================================================


class TestScriptContract:
    """Argparse + module surface contract tests."""

    def test_script_file_exists(self) -> None:
        """The benchmark script is committed at the expected path."""
        assert SCRIPT_PATH.exists(), f"missing {SCRIPT_PATH}"
        assert SCRIPT_PATH.is_file(), f"{SCRIPT_PATH} is not a file"

    def test_csv_file_exists(self) -> None:
        """The PlatinAI 5K CSV is shipped with the repo."""
        assert CSV_PATH.exists(), f"missing {CSV_PATH}"

    def test_csv_has_header(self) -> None:
        """CSV header contains 'smiles' (the column the benchmark reads)."""
        with CSV_PATH.open() as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "smiles" in header

    def test_csv_has_thousand_rows(self) -> None:
        """PlatinAI 5K has 5000 rows."""
        n = 0
        with CSV_PATH.open() as f:
            for _ in csv.DictReader(f):
                n += 1
        assert n >= 5000, f"expected >=5000 rows, got {n}"

    def test_module_imports(self) -> None:
        """The benchmark module imports without error."""
        mod = _import_benchmark_module()
        assert hasattr(mod, "run_benchmark")
        assert hasattr(mod, "__version__")
        assert mod.__version__.startswith("0.1.0")

    def test_default_constants(self) -> None:
        """Default constants are sane + match the CLI defaults."""
        mod = _import_benchmark_module()
        assert mod.DEFAULT_N_SIMULATIONS >= 1
        assert mod.DEFAULT_MAX_DEPTH >= 1
        assert mod.DEFAULT_TOP_K >= 1
        assert mod.DEFAULT_N_SAMPLES >= 1

    def test_default_csv_path_resolves(self) -> None:
        """``_resolve_csv_path`` finds the CSV when given the default arg."""
        mod = _import_benchmark_module()
        resolved = mod._resolve_csv_path(str(CSV_PATH))
        assert resolved.exists()

    def test_resolve_csv_raises_on_missing(self) -> None:
        """``_resolve_csv_path`` raises ``MCTSRealBenchError`` on a bad path."""
        mod = _import_benchmark_module()
        with pytest.raises(mod.MCTSRealBenchError):
            mod._resolve_csv_path(
                "/nonexistent/path/to/never_existed_platinai.csv",
            )


# ===========================================================================
# 2. Smoke runs — small budgets must complete + emit results.json
# ===========================================================================


class TestBenchmarkSmoke:
    """Smoke tests that exercise the full benchmark on tiny budgets."""

    def test_benchmark_runs(self, tmp_report_dir: Path) -> None:
        """``run_benchmark`` returns a dict with all required keys."""
        mod = _import_benchmark_module()
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        assert isinstance(summary, dict)
        for key in (
            "base_mcts_wall_time",
            "variance_puct_wall_time",
            "variance_mcts_wall_time",
            "success_rate_base",
            "success_rate_variance_puct",
            "success_rate_variance_mcts",
            "n_distinct_base",
            "n_distinct_variance_puct",
            "n_distinct_variance_mcts",
            "n_samples_searched",
        ):
            assert key in summary, f"missing key {key!r} in summary"

    def test_benchmark_results_json_written(self, tmp_report_dir: Path) -> None:
        """results.json is written and contains the summary."""
        mod = _import_benchmark_module()
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        json_path = mod._write_results_json(tmp_report_dir, summary)
        assert json_path.exists()
        with json_path.open() as f:
            data = json.load(f)
        assert data["n_samples_searched"] == summary["n_samples_searched"]

    def test_benchmark_log_written(self, tmp_report_dir: Path) -> None:
        """benchmark_log.md is written and is non-empty."""
        mod = _import_benchmark_module()
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        log_path = mod._write_benchmark_log(
            tmp_report_dir,
            summary,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            csv_path=str(CSV_PATH),
        )
        assert log_path.exists()
        text = log_path.read_text()
        assert "Wf-MCTS-Real-Bench" in text
        assert "base" in text
        assert "variance_puct" in text
        assert "variance_mcts" in text

    def test_benchmark_compare_3_strategies(self, tmp_report_dir: Path) -> None:
        """All three strategies are exercised and reported."""
        mod = _import_benchmark_module()
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        reports = summary["strategy_reports"]
        assert "base" in reports
        assert "variance_puct" in reports
        assert "variance_mcts" in reports
        # Each strategy must report a non-zero samples count.
        for k in ("base", "variance_puct", "variance_mcts"):
            assert reports[k]["n_samples"] == summary["n_samples_searched"]


# ===========================================================================
# 3. Per-strategy invariants — wall_time + success_rate bounded
# ===========================================================================


class TestStrategyInvariants:
    """Invariants on the per-strategy metrics."""

    def test_wall_time_non_negative(self, tmp_report_dir: Path) -> None:
        """Wall-time per strategy is >= 0."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        assert summary["base_mcts_wall_time"] >= 0.0
        assert summary["variance_puct_wall_time"] >= 0.0
        assert summary["variance_mcts_wall_time"] >= 0.0

    def test_wall_time_finite(self, tmp_report_dir: Path) -> None:
        """Wall-time per strategy is finite (no NaN / Inf)."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        for key in (
            "base_mcts_wall_time",
            "variance_puct_wall_time",
            "variance_mcts_wall_time",
        ):
            v = float(summary[key])
            assert math.isfinite(v), f"{key} is not finite: {v}"

    def test_success_rate_in_unit_interval(self, tmp_report_dir: Path) -> None:
        """Success rate is in [0, 1] for all three strategies."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        for key in (
            "success_rate_base",
            "success_rate_variance_puct",
            "success_rate_variance_mcts",
        ):
            v = float(summary[key])
            assert 0.0 <= v <= 1.0, f"{key} out of [0,1]: {v}"

    def test_n_distinct_non_negative(self, tmp_report_dir: Path) -> None:
        """Mean distinct SMILES is >= 0 for all three strategies."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        for key in (
            "n_distinct_base",
            "n_distinct_variance_puct",
            "n_distinct_variance_mcts",
        ):
            v = float(summary[key])
            assert v >= 0.0, f"{key} negative: {v}"

    def test_total_candidates_non_negative(self, tmp_report_dir: Path) -> None:
        """Total candidate counts are >= 0."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        for key in (
            "total_candidates_base",
            "total_candidates_variance_puct",
            "total_candidates_variance_mcts",
        ):
            assert int(summary[key]) >= 0

    def test_wall_time_ratio_finite(self, tmp_report_dir: Path) -> None:
        """Wall-time ratio (variance / base) is finite and positive."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        for key in (
            "wall_time_ratio_variance_puct_over_base",
            "wall_time_ratio_variance_mcts_over_base",
        ):
            v = float(summary[key])
            assert math.isfinite(v) and v > 0.0, f"{key} bad: {v}"


# ===========================================================================
# 4. CLI integration — exit code, results.json content
# ===========================================================================


class TestCLIIntegration:
    """End-to-end CLI invocation of the benchmark script."""

    def test_cli_smoke(self, tmp_report_dir: Path) -> None:
        """``python -m molmetal.scripts.benchmark_mcts_real --n-samples 2`` runs."""
        env = dict(os.environ)
        env.setdefault("MOLMETAL_DATA_DIR", str(REPO_ROOT / ".data" / "molmetal"))
        env.setdefault(
            "MOLMETAL_TMQM_DIR", str(REPO_ROOT / ".data" / "molmetal" / "tmQM")
        )
        cmd: List[str] = [
            sys.executable,
            "-m",
            "molmetal.scripts.benchmark_mcts_real",
            "--n-samples",
            "2",
            "--n-simulations",
            "2",
            "--max-depth",
            "2",
            "--top-k",
            "2",
            "--csv",
            str(CSV_PATH),
            "--out-dir",
            str(tmp_report_dir),
            "--quiet",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Some environments print RDKit warnings on stderr; check
        # stdout/exit-code only.
        assert proc.returncode == 0, (
            f"benchmark failed (rc={proc.returncode}); "
            f"stdout={proc.stdout[-400:]!r} stderr={proc.stderr[-400:]!r}"
        )
        # Verify results.json was emitted.
        json_path = tmp_report_dir / "results.json"
        assert json_path.exists(), "results.json not emitted"
        data = json.loads(json_path.read_text())
        assert data["n_samples_searched"] >= 1

    def test_results_json_has_expected_keys(
        self,
        tmp_report_dir: Path,
    ) -> None:
        """results.json contains every required top-level key."""
        mod = _import_benchmark_module()
        summary = _run_benchmark(
            n_samples=2,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        json_path = mod._write_results_json(tmp_report_dir, summary)
        data = json.loads(json_path.read_text())
        required = {
            "n_samples_loaded",
            "n_samples_searched",
            "n_parse_errors",
            "base_mcts_wall_time",
            "variance_puct_wall_time",
            "variance_mcts_wall_time",
            "success_rate_base",
            "success_rate_variance_puct",
            "success_rate_variance_mcts",
            "n_distinct_base",
            "n_distinct_variance_puct",
            "n_distinct_variance_mcts",
            "wall_time_ratio_variance_puct_over_base",
            "wall_time_ratio_variance_mcts_over_base",
            "strategy_reports",
            "version",
        }
        missing = required - set(data.keys())
        assert not missing, f"missing keys in results.json: {sorted(missing)}"


# ===========================================================================
# 5. Aggregate helper invariants
# ===========================================================================


class TestSummaryHelpers:
    """Sanity checks on the internal summarisation helpers."""

    def test_summarise_strategy_empty(self) -> None:
        """``_summarise_strategy`` handles an empty per_sample list."""
        mod = _import_benchmark_module()
        rep = mod._summarise_strategy("base", [])
        assert rep.n_samples == 0
        assert rep.n_errors == 0
        assert rep.success_rate == 0.0

    def test_summarise_strategy_counts_success(self) -> None:
        """``_summarise_strategy`` counts only error-free samples with ≥1 candidate."""
        mod = _import_benchmark_module()
        per = [
            mod.SampleResult(0, 0.1, 2, 2, 2, error=None),
            mod.SampleResult(1, 0.2, 0, 0, 0, error=None),
            mod.SampleResult(2, 0.3, 1, 1, 1, error="boom"),
        ]
        rep = mod._summarise_strategy("base", per)
        # 2 ok samples, 1 with error.
        assert rep.n_samples == 3
        assert rep.n_errors == 1
        # 1 of 2 ok samples has ≥1 candidate.
        assert rep.n_success == 1
        assert rep.success_rate == 0.5
        # Total wall-time only sums error-free samples.
        assert math.isclose(rep.total_wall_time_s, 0.3, abs_tol=1e-9)
        assert rep.total_candidates == 2

    def test_safe_canonical_smiles_handles_failure(self) -> None:
        """``_safe_canonical_smiles`` returns ``None`` on failure."""
        mod = _import_benchmark_module()

        class _Bad:
            def canonical_smiles(self) -> str:
                raise RuntimeError("nope")

        assert mod._safe_canonical_smiles(_Bad()) is None

    def test_load_platinai_rows_respects_n_samples(self) -> None:
        """``_load_platinai_rows`` truncates to n_samples rows."""
        mod = _import_benchmark_module()
        rows = mod._load_platinai_rows(CSV_PATH, n_samples=7)
        assert len(rows) == 7
        # Every loaded row must carry a non-empty SMILES.
        for r in rows:
            assert r.get("smiles"), f"row missing smiles: {r!r}"


# ===========================================================================
# 6. Behavioural contracts on per-sample aggregation
# ===========================================================================


class TestPerSampleBehaviour:
    """Properties of the per-sample + per-strategy bookkeeping."""

    def test_per_sample_records_match(self, tmp_report_dir: Path) -> None:
        """``n_samples_searched`` equals the number of base per-sample records."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        rep = summary["strategy_reports"]["base"]
        assert len(rep["per_sample"]) == summary["n_samples_searched"]

    def test_per_sample_wall_time_consistent(
        self,
        tmp_report_dir: Path,
    ) -> None:
        """Sum of per-sample wall-times equals total wall-time (base)."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        rep = summary["strategy_reports"]["base"]
        per_sum = sum(getattr(r, "wall_time_s") for r in rep["per_sample"])
        assert math.isclose(per_sum, rep["total_wall_time_s"], abs_tol=1e-6)

    def test_per_sample_n_distinct_le_n_candidates(
        self,
        tmp_report_dir: Path,
    ) -> None:
        """Distinct SMILES ≤ total candidates (dedup invariant)."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        rep = summary["strategy_reports"]["base"]
        for r in rep["per_sample"]:
            n_dist = getattr(r, "n_distinct")
            n_cand = getattr(r, "n_candidates")
            assert n_dist <= n_cand, f"distinct {n_dist} > cands {n_cand}"

    def test_summary_round_trip_json(
        self,
        tmp_report_dir: Path,
    ) -> None:
        """The summary dict survives JSON round-trip with no NaN / Inf.

        The summary embeds :class:`SampleResult` dataclass instances
        which are not natively JSON-serialisable; the production
        writer wires in a ``default=`` handler.  Here we mirror that
        contract by stripping the dataclass-laden ``strategy_reports``
        subtree before round-tripping — the focus of this test is the
        numerical scalars (``base_mcts_wall_time``,
        ``success_rate_*``, …) being finite, not the dataclass
        serialisation path.
        """
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        # Strip non-scalar fields before JSON round-trip.
        scrubbed = {k: v for k, v in summary.items() if k != "strategy_reports"}
        serialised = json.dumps(scrubbed)
        restored = json.loads(serialised)
        for k in (
            "base_mcts_wall_time",
            "variance_puct_wall_time",
            "variance_mcts_wall_time",
            "success_rate_base",
            "success_rate_variance_puct",
            "success_rate_variance_mcts",
        ):
            assert k in restored
            assert math.isfinite(restored[k])

    def test_n_errors_zero_on_clean_run(
        self,
        tmp_report_dir: Path,
    ) -> None:
        """A clean smoke run emits 0 search errors per strategy."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        assert int(summary["n_errors_base"]) == 0
        assert int(summary["n_errors_variance_puct"]) == 0
        assert int(summary["n_errors_variance_mcts"]) == 0

    def test_variance_puct_reuses_underlying_search(
        self,
        tmp_report_dir: Path,
    ) -> None:
        """VariancePUCT results track base within ±20% (same underlying MCTS)."""
        summary = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            out_dir=tmp_report_dir,
        )
        b = summary["base_mcts_wall_time"]
        vp = summary["variance_puct_wall_time"]
        if b > 1e-6:
            # Variance wrapper re-runs ``MCTSProofSearch.search`` and
            # re-scores — total wall-time must be within ±20% of base.
            ratio = vp / b
            assert (
                0.5 <= ratio <= 2.0
            ), f"variance_puct / base wall ratio out of range: {ratio:.3f}"

    def test_seed_changes_nothing_on_determinism(
        self,
        tmp_report_dir: Path,
    ) -> None:
        """Re-running with the same seed + same samples yields the same totals.

        This is a deterministic-run contract: with the same seed, n_samples,
        and PlatinAI rows, the wall-time totals must be byte-equal across
        two consecutive runs (modulo clock granularity, hence the 0.5 s
        tolerance for the totals).
        """
        a = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            seed=12345,
            out_dir=tmp_report_dir / "a",
        )
        b = _run_benchmark(
            n_samples=3,
            n_simulations=2,
            max_depth=2,
            top_k=2,
            seed=12345,
            out_dir=tmp_report_dir / "b",
        )
        # Same success / candidate counts under the same seed.
        assert a["total_candidates_base"] == b["total_candidates_base"]
        assert a["success_count_variance_puct"] == b["success_count_variance_puct"]
        assert (
            a["total_candidates_variance_mcts"] == b["total_candidates_variance_mcts"]
        )
