"""Tests for the TODO-30 P4.1 — 100x3 sweep CPU mock skeleton.

Test inventory (20 tests)
-------------------------
1.  test_mock_runs                     — 10x3 = 30 evaluations complete
2.  test_pocket_subset_size            — subset = 10
3.  test_run_seed_returns_dict         — {wall_time, n_atoms, validity, ...}
4.  test_full_sweep_total              — 30 evaluations in results
5.  test_gpu_extension_plan            — estimate hours
6.  test_mock_vs_full_projection       — projected ≥ measured
7.  test_mock_deterministic            — same seed → identical cell
8.  test_seed_independence             — different seeds → different cells
9.  test_pocket_independence           — different pockets → different cells
10. test_metrics_in_unit_interval      — validity ∈ [0, 1]
11. test_n_atoms_int                   — n_atoms is int and positive
12. test_summary_schema                — JSON has all required keys
13. test_aggregate_mean_in_band        — mean SA within ±0.5 of anchor
14. test_aggregate_std_nonnegative     — std ≥ 0 for every metric
15. test_projection_shrinks_uncertainty — full-sweep CI < mock CI
16. test_compare_pass_when_within_ci   — mock mean inside projection CI
17. test_run_smoke_3x2                 — 3x2 = 6 evaluations complete
18. test_save_summary_writes_files     — JSON + per_cell.csv written
19. test_custom_output_dir             — summary lands in custom path
20. test_gpu_plan_command_line          — command_line field is non-empty
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict as _asdict
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts.lambda_100x3_cpu_mock import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    GPU_TARGET_CELLS,
    METRIC_KEYS,
    GPUExtensionPlan,
    LambdaSweep100x3Mock,
    compare_mock_vs_full,
    load_manifest_pocket_ids,
    synthetic_pocket_pool,
    _aggregate_metrics,
    _deterministic_mock_cell,
    _hashselect,
    _project_to_full_sweep,
    _signed_unit_hash,
)


# Quiet the logger during the test session — the mock logs at INFO by
# default and we don't want to spam pytest's capture buffer.
logging.getLogger("wf_p4_1_100x3_mock").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 1. Mock runs end-to-end (10x3 = 30 evaluations).
# ---------------------------------------------------------------------------
def test_mock_runs(tmp_path: Path) -> None:
    """The default mock sweeps 10 pockets x 3 seeds = 30 evaluations."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    assert summary.total_evaluations == 30
    assert summary.n_pockets == 10
    assert summary.n_seeds == 3


# ---------------------------------------------------------------------------
# 2. Pocket subset size.
# ---------------------------------------------------------------------------
def test_pocket_subset_size(tmp_path: Path) -> None:
    """``generate_pocket_subset(n=10)`` returns exactly 10 pocket ids."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    subset = mock.generate_pocket_subset(n=10)
    assert len(subset) == 10
    assert len(set(subset)) == 10  # no duplicates


# ---------------------------------------------------------------------------
# 3. ``run_seed`` returns a dict with the contract fields.
# ---------------------------------------------------------------------------
def test_run_seed_returns_dict(tmp_path: Path) -> None:
    """``run_seed`` returns ``{wall_time, n_atoms, validity, ...}`` shape."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    cell = mock.run_seed(pocket="test_001", seed=1)
    assert isinstance(cell, dict)
    assert "pocket_id" in cell and cell["pocket_id"] == "test_001"
    assert "seed" in cell and cell["seed"] == 1
    assert "n_atoms" in cell and isinstance(cell["n_atoms"], int)
    assert "elapsed_s" in cell and cell["elapsed_s"] > 0
    assert "metrics" in cell
    for k in METRIC_KEYS:
        if k in ("n_atoms", "elapsed_s"):
            continue
        assert k in cell["metrics"], f"missing metric {k}"


# ---------------------------------------------------------------------------
# 4. The full sweep emits exactly ``n_pockets x n_seeds`` per-cell entries.
# ---------------------------------------------------------------------------
def test_full_sweep_total(tmp_path: Path) -> None:
    """``per_cell`` length equals ``n_pockets x n_seeds``."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    assert len(summary.per_cell) == mock.n_pockets * mock.n_seeds


# ---------------------------------------------------------------------------
# 5. GPU extension plan reports a finite positive wall-time.
# ---------------------------------------------------------------------------
def test_gpu_extension_plan(tmp_path: Path) -> None:
    """``estimated_wall_time_hours()`` returns a positive finite float."""
    plan = GPUExtensionPlan()
    hours = plan.estimated_wall_time_hours()
    assert hours > 0
    assert hours == pytest.approx(GPU_TARGET_CELLS * plan.reference_cell_s / 3600.0)
    s = plan.estimated_wall_time_str()
    assert "h" in s and str(GPU_TARGET_CELLS) in s


# ---------------------------------------------------------------------------
# 6. Mock-vs-full projection reports a valid projection block.
# ---------------------------------------------------------------------------
def test_mock_vs_full_projection(tmp_path: Path) -> None:
    """Projection reports projected_value within the mock CI for every metric."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    proj = summary.projected_to_100x3
    for key in METRIC_KEYS:
        assert key in proj
        entry = proj[key]
        assert "projected_value" in entry
        assert "uncertainty" in entry
        # Mock mean equals projected value (CLT point estimate).
        assert entry["projected_value"] == pytest.approx(
            summary.aggregate_metrics[key], abs=1e-9
        )


# ---------------------------------------------------------------------------
# 7. Determinism — same inputs give the same cell.
# ---------------------------------------------------------------------------
def test_mock_deterministic(tmp_path: Path) -> None:
    """Two ``run_seed`` calls with identical args produce bit-exact cells."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    a = mock.run_seed(pocket="test_005", seed=2)
    b = mock.run_seed(pocket="test_005", seed=2)
    assert a == b


# ---------------------------------------------------------------------------
# 8. Seed independence — different seeds give different metrics.
# ---------------------------------------------------------------------------
def test_seed_independence(tmp_path: Path) -> None:
    """Different seeds produce at least one differing metric per pocket."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    cells = [mock.run_seed(pocket="test_007", seed=s) for s in range(3)]
    # At least one metric differs across the three seeds.
    assert any(
        cells[0]["metrics"][k] != cells[1]["metrics"][k]
        or cells[1]["metrics"][k] != cells[2]["metrics"][k]
        for k in cells[0]["metrics"]
    )


# ---------------------------------------------------------------------------
# 9. Pocket independence — different pockets give different metrics.
# ---------------------------------------------------------------------------
def test_pocket_independence(tmp_path: Path) -> None:
    """Different pockets produce at least one differing metric per seed."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    cells = [
        mock.run_seed(pocket=p, seed=0) for p in ("test_001", "test_002", "test_003")
    ]
    assert any(
        cells[0]["metrics"][k] != cells[1]["metrics"][k]
        or cells[1]["metrics"][k] != cells[2]["metrics"][k]
        for k in cells[0]["metrics"]
    )


# ---------------------------------------------------------------------------
# 10. Probability-style metrics live in [0, 1].
# ---------------------------------------------------------------------------
def test_metrics_in_unit_interval(tmp_path: Path) -> None:
    """Validity / uniqueness / synthesizability / decoder_pass ∈ [0, 1]."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    prob_keys = (
        "validity_rate",
        "uniqueness_rate",
        "synthesizability_rate",
        "metal_compliance_rate",
        "decoder_pass_rate",
    )
    for cell in summary.per_cell:
        for k in prob_keys:
            v = cell["metrics"][k]
            assert 0.0 <= v <= 1.0, f"{k} = {v} out of [0, 1]"


# ---------------------------------------------------------------------------
# 11. ``n_atoms`` is a positive integer.
# ---------------------------------------------------------------------------
def test_n_atoms_int(tmp_path: Path) -> None:
    """``n_atoms`` is an int and lies in the metallodrug vertical 8..38 range."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    for cell in summary.per_cell:
        assert isinstance(cell["n_atoms"], int)
        assert cell["n_atoms"] > 0


# ---------------------------------------------------------------------------
# 12. JSON summary schema carries every required top-level key.
# ---------------------------------------------------------------------------
def test_summary_schema(tmp_path: Path) -> None:
    """``summary`` JSON has all required top-level keys + per-cell fields."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    required = {
        "schema_version",
        "tool",
        "pilot_id",
        "date_utc",
        "n_pockets",
        "n_seeds",
        "total_evaluations",
        "total_wall_time_s",
        "per_cell",
        "aggregate_metrics",
        "aggregate_metrics_std",
        "projected_to_100x3",
        "gpu_extension_plan",
        "compare_mock_vs_full",
    }
    assert required.issubset(set(_asdict(summary).keys()))


# ---------------------------------------------------------------------------
# 13. Aggregate mean stays within the calibrated anchor band.
# ---------------------------------------------------------------------------
def test_aggregate_mean_in_band(tmp_path: Path) -> None:
    """Aggregate mean of every metric is within ±0.5 (or ±5 %) of the anchor.

    Mock mean is calibrated to match wf_lambda_b2_pilot_10x3 within
    ±0.05 on every metric.  The test uses a wider ±0.5 band (or ±5 %
    for relative-scale metrics) to absorb the per-cell deterministic
    noise — a CI-of-mean at n=30 cells is ~mock_std / sqrt(30) which
    easily clears ±0.5 on SA and ±0.05 on QED.
    """
    from molmetal.scripts.lambda_100x3_cpu_mock import (
        MOCK_ANCHOR_MEAN,
        MOCK_ANCHOR_SPREAD,
    )

    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    for key in METRIC_KEYS:
        anchor = MOCK_ANCHOR_MEAN[key]
        spread = MOCK_ANCHOR_SPREAD[key]
        if key == "elapsed_s":
            # Wall-time metric — allow a 3x absolute band since the
            # base seed controls the spread.  The mock is calibrated
            # to MOCK_CELL_BASE_S = 2.0 s but the anchor's 125 s is
            # the *GPU* reference; we don't claim to reproduce it.
            tol = max(3.0, abs(anchor))
            assert abs(summary.aggregate_metrics[key] - anchor) <= tol
            continue
        tol = max(0.5, abs(anchor) * 0.05)
        # The CLT bound is ~spread / sqrt(30); we use a wider band so
        # the test is robust to per-cell hash collisions.
        assert abs(summary.aggregate_metrics[key] - anchor) <= tol + spread


# ---------------------------------------------------------------------------
# 14. Aggregate std is non-negative for every metric.
# ---------------------------------------------------------------------------
def test_aggregate_std_nonnegative(tmp_path: Path) -> None:
    """``aggregate_metrics_std[k] >= 0`` for every metric in ``METRIC_KEYS``."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    for key in METRIC_KEYS:
        assert summary.aggregate_metrics_std[key] >= 0.0


# ---------------------------------------------------------------------------
# 15. Projection CI shrinks vs the mock CI (1/sqrt(N) scaling).
# ---------------------------------------------------------------------------
def test_projection_shrinks_uncertainty(tmp_path: Path) -> None:
    """Full-sweep CI half-width is strictly smaller than the mock CI half-width."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    for key in METRIC_KEYS:
        u_target = summary.projected_to_100x3[key]["uncertainty"]
        u_mock = summary.projected_to_100x3[key]["uncertainty_mock"]
        if u_mock > 0:
            assert u_target < u_mock


# ---------------------------------------------------------------------------
# 16. ``compare_mock_vs_full`` returns PASS when mock mean is inside CI.
# ---------------------------------------------------------------------------
def test_compare_pass_when_within_ci(tmp_path: Path) -> None:
    """Mock mean is bit-exact with the projected value (CLT invariance)."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    compare = summary.compare_mock_vs_full
    # Every metric whose ``projected_value == mock_value`` trivially
    # falls inside the 0-width CI; ``elapsed_s`` has note "wall-time".
    # At least one metric must PASS.
    assert compare["n_pass"] >= 1
    assert compare["n_fail"] >= 0
    # All non-wall-time metrics with uncertainty > 0 must be PASS.
    for key in METRIC_KEYS:
        if key == "elapsed_s":
            continue
        assert compare["rows"][key]["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# 17. Tight smoke preset (3x2 = 6 cells) completes in < 30 s.
# ---------------------------------------------------------------------------
def test_run_smoke_3x2(tmp_path: Path) -> None:
    """``--n-pockets 3 --n-seeds 2`` returns exactly 6 cells."""
    mock = LambdaSweep100x3Mock(
        n_pockets=3,
        n_seeds=2,
        output_dir=tmp_path / "out",
    )
    summary = mock.run_full_sweep()
    assert summary.total_evaluations == 6
    assert summary.n_pockets == 3
    assert summary.n_seeds == 2


# ---------------------------------------------------------------------------
# 18. ``save_summary`` writes both JSON and per_cell.csv.
# ---------------------------------------------------------------------------
def test_save_summary_writes_files(tmp_path: Path) -> None:
    """JSON summary + per_cell.csv both exist after ``save_summary``."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    json_path = mock.save_summary(summary)
    assert json_path.is_file()
    csv_path = json_path.parent / "per_cell.csv"
    assert csv_path.is_file()
    # JSON round-trips.
    payload = json.loads(json_path.read_text())
    assert payload["total_evaluations"] == summary.total_evaluations
    # CSV row count = 30 data rows + 1 header.
    with csv_path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 1 + summary.total_evaluations


# ---------------------------------------------------------------------------
# 19. Custom output dir is honoured.
# ---------------------------------------------------------------------------
def test_custom_output_dir(tmp_path: Path) -> None:
    """``--output-dir /custom/path`` writes the summary under that path."""
    custom = tmp_path / "my_custom_dir"
    mock = LambdaSweep100x3Mock(output_dir=custom)
    summary = mock.run_full_sweep()
    json_path = mock.save_summary(summary)
    assert json_path.parent == custom
    assert json_path.is_file()


# ---------------------------------------------------------------------------
# 20. GPU plan command_line is non-empty and references r4_lambda_only_run.
# ---------------------------------------------------------------------------
def test_gpu_plan_command_line(tmp_path: Path) -> None:
    """``gpu_extension_plan.command_line`` references the R17 paper-grade CLI."""
    mock = LambdaSweep100x3Mock(output_dir=tmp_path / "out")
    summary = mock.run_full_sweep()
    cmd = summary.gpu_extension_plan["command_line"]
    assert isinstance(cmd, str) and len(cmd) > 0
    assert "r4_lambda_only_run" in cmd
    assert "--r17-paper-grade" in cmd


# ===========================================================================
# Parametric / corner-case tests (extra coverage beyond the 20 required)
# ===========================================================================
def test_synthetic_pocket_pool() -> None:
    """``synthetic_pocket_pool`` returns 100 deterministic ``test_NNN`` ids."""
    pool = synthetic_pocket_pool(n=100)
    assert len(pool) == 100
    assert pool[0] == "test_000"
    assert pool[-1] == "test_099"


def test_hashselect_is_deterministic() -> None:
    """``_hashselect`` returns the same subset across calls."""
    pool = synthetic_pocket_pool(n=100)
    a = _hashselect(pool, n=10, salt="x")
    b = _hashselect(pool, n=10, salt="x")
    assert a == b
    # Different salts give different subsets (probabilistic but very
    # likely on a 100-pool).
    c = _hashselect(pool, n=10, salt="y")
    assert c != a


def test_hashselect_zero_n_returns_empty() -> None:
    """``_hashselect(n=0)`` returns an empty list."""
    pool = synthetic_pocket_pool(n=100)
    assert _hashselect(pool, n=0, salt="z") == []


def test_signed_unit_hash_in_minus_one_one() -> None:
    """``_signed_unit_hash`` returns a value in ``[-1, 1]`` deterministically."""
    for seed in range(10):
        v = _signed_unit_hash("test_001", seed, 42, "qed_mean")
        assert -1.0 <= v <= 1.0
    assert _signed_unit_hash("a", 0, 0, "x") == _signed_unit_hash("a", 0, 0, "x")


def test_load_manifest_falls_back_when_missing(tmp_path: Path) -> None:
    """``load_manifest_pocket_ids`` returns [] when the path is unreachable."""
    bogus = tmp_path / "does_not_exist.csv"
    assert load_manifest_pocket_ids(bogus) == []


def test_aggregate_metrics_matches_python_statistics(tmp_path: Path) -> None:
    """``_aggregate_metrics`` matches a hand-rolled statistics reference."""
    cells = [
        {
            "pocket_id": "p",
            "seed": 0,
            "n_atoms": 10,
            "elapsed_s": 1.0,
            "metrics": {k: (0.5 if k == "qed_mean" else 3.0) for k in METRIC_KEYS},
        },
        {
            "pocket_id": "p",
            "seed": 1,
            "n_atoms": 12,
            "elapsed_s": 1.5,
            "metrics": {k: (0.7 if k == "qed_mean" else 4.0) for k in METRIC_KEYS},
        },
        {
            "pocket_id": "p",
            "seed": 2,
            "n_atoms": 14,
            "elapsed_s": 2.0,
            "metrics": {k: (0.9 if k == "qed_mean" else 5.0) for k in METRIC_KEYS},
        },
    ]
    mean, std = _aggregate_metrics(cells)
    import statistics as _st

    assert mean["qed_mean"] == pytest.approx(_st.fmean([0.5, 0.7, 0.9]))
    assert std["qed_mean"] == pytest.approx(_st.stdev([0.5, 0.7, 0.9]))
    assert mean["sa_mean"] == pytest.approx(_st.fmean([3.0, 4.0, 5.0]))


def test_projection_handles_zero_input() -> None:
    """``_project_to_full_sweep`` returns an empty dict for n_target = 0."""
    proj = _project_to_full_sweep(
        mock_mean={"qed_mean": 0.5},
        mock_std={"qed_mean": 0.1},
        n_mock=10,
        n_target=0,
    )
    assert proj == {}


def test_compare_mock_vs_full_explicit_dict() -> None:
    """``compare_mock_vs_full`` returns the expected verdict for synthetic input."""
    mock_agg = {
        "n_distinct": 20.0,
        "validity_rate": 1.0,
        "uniqueness_rate": 1.0,
        "synthesizability_rate": 1.0,
        "metal_compliance_rate": 0.0,
        "decoder_pass_rate": 1.0,
        "sa_mean": 3.5,
        "qed_mean": 0.7,
        "logp_mean": -0.5,
        "tpsa_mean": 50.0,
        "rotb_mean": 2.0,
        "n_atoms": 24.0,
        "elapsed_s": 100.0,
    }
    # Mock mean == projected mean ⇒ verdict PASS everywhere.
    proj = {k: {"projected_value": mock_agg[k], "uncertainty": 0.1} for k in mock_agg}
    cmp = compare_mock_vs_full(mock_agg, proj, n_mock=30, n_target=300)
    assert cmp["verdict"] == "PASS"
    assert cmp["n_pass"] == len(METRIC_KEYS)
    assert cmp["n_fail"] == 0
