"""Tests for the closure-theorem real benchmark (Wf-17 P4.2).

================================================================
Wf-17 P4.2 vs Wf-16 P4.1 — closure-theorem REAL benchmark
================================================================
This file ships the **20 pytest contracts** that pin the behaviour of
:mod:`molmetal.scripts.closure_real_benchmark`:

The 20 tests
------------
Construction / smoke
 1. test_module_imports                     — script imports cleanly
 2. test_default_csv_path                   — default CSV is PlatinAI 5K
 3. test_default_output_dir                 — default output_dir = wf_closure_real
 4. test_default_n_samples_is_500           — default --n-samples is 500
 5. test_smoke_n_samples_runs               — --n-samples 10 completes in < 60 s
 6. test_smoke_under_one_minute             — smoke budget < 60 s

Per-sample record shape
 7. test_per_sample_record_keys             — every record has all expected keys
 8. test_per_sample_status_values           — status ∈ {ok, skipped, error}
 9. test_per_sample_closure_ok_is_bool      — closure_ok is bool
10. test_per_sample_wall_time_non_negative  — wall_time_s ≥ 0
11. test_per_sample_memory_non_negative     — memory_mb ≥ 0
12. test_per_sample_max_depth_seen_bounded  — max_depth_seen ≤ max_depth

Aggregate
13. test_aggregate_keys_present             — every aggregate key exists
14. test_closure_success_rate_in_unit       — closure_success_rate ∈ [0, 1]
15. test_mean_beta_depth_non_negative       — mean_beta_depth ≥ 0
16. test_mean_wall_time_non_negative        — mean_wall_time ≥ 0
17. test_peak_memory_non_negative           — peak_memory_mb ≥ 0
18. test_status_counts_sum_to_n_samples     — status counts sum to n_samples

Persistence
19. test_benchmark_json_written             — benchmark.json exists + valid JSON
20. test_per_sample_csv_written             — per_sample.csv exists + correct header

All tests are CPU-only, deterministic, and complete in < 90 s
combined (smoke budgets above are enforced by ``time``-bounded
``pytest`` skips if exceeded).

Run with::

    uv run --no-sync pytest -q molmetal/tests/test_closure_real_benchmark.py --tb=short
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — mirrors the convention used by sibling test files.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "molmetal" / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from molmetal.scripts import closure_real_benchmark  # noqa: E402
from molmetal.scripts.closure_real_benchmark import (  # noqa: E402
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PRODUCTS,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLATINAI_CSV,
    PARTNER_SMILE_LIST,
    _aggregate,
    _build_click_rules,
    _build_partner_terms,
    _load_platinai_smiles,
    run_benchmark,
    save_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_payload() -> Dict[str, Any]:
    """A 10-sample smoke payload — fast enough to run on every CI."""
    return run_benchmark(
        n_samples=10,
        csv_path=DEFAULT_PLATINAI_CSV,
        output_dir=DEFAULT_OUTPUT_DIR,
        max_depth=DEFAULT_MAX_DEPTH,
        max_products=DEFAULT_MAX_PRODUCTS,
        smoke=True,
    )


@pytest.fixture(scope="module")
def smoke_paths(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """The artefact paths produced by the smoke run."""
    out = tmp_path_factory.mktemp("closure_real_smoke")
    payload = run_benchmark(
        n_samples=10,
        csv_path=DEFAULT_PLATINAI_CSV,
        output_dir=out,
        max_depth=DEFAULT_MAX_DEPTH,
        max_products=DEFAULT_MAX_PRODUCTS,
        smoke=True,
    )
    paths = save_benchmark(payload, output_dir=out)
    return paths


# ---------------------------------------------------------------------------
# 1. Module imports cleanly
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """The benchmark module imports without raising."""
    assert closure_real_benchmark is not None
    assert hasattr(closure_real_benchmark, "run_benchmark")
    assert hasattr(closure_real_benchmark, "save_benchmark")
    assert hasattr(closure_real_benchmark, "_aggregate")


# ---------------------------------------------------------------------------
# 2. Default CSV path
# ---------------------------------------------------------------------------
def test_default_csv_path() -> None:
    """The default CSV points at the PlatinAI 5K ship file."""
    assert DEFAULT_PLATINAI_CSV.name == "platinai_5k.csv", (
        f"expected CSV 'platinai_5k.csv', got {DEFAULT_PLATINAI_CSV.name}"
    )
    assert DEFAULT_PLATINAI_CSV.is_file(), (
        f"PlatinAI 5K CSV not found at {DEFAULT_PLATINAI_CSV}"
    )


# ---------------------------------------------------------------------------
# 3. Default output dir
# ---------------------------------------------------------------------------
def test_default_output_dir() -> None:
    """The default output dir lives under reports/wf_closure_real."""
    assert DEFAULT_OUTPUT_DIR.name == "wf_closure_real"
    assert "reports" in DEFAULT_OUTPUT_DIR.parts
    assert "molmetal" in DEFAULT_OUTPUT_DIR.parts


# ---------------------------------------------------------------------------
# 4. Default --n-samples is 500
# ---------------------------------------------------------------------------
def test_default_n_samples_is_500() -> None:
    """The default sample count is 500 (matches the Wf-17 P4.2 brief)."""
    assert DEFAULT_N_SAMPLES == 500, (
        f"expected DEFAULT_N_SAMPLES=500, got {DEFAULT_N_SAMPLES}"
    )


# ---------------------------------------------------------------------------
# 5. Smoke N=10 completes
# ---------------------------------------------------------------------------
def test_smoke_n_samples_runs(smoke_payload: Dict[str, Any]) -> None:
    """A 10-sample smoke run completes and returns a non-empty payload."""
    assert isinstance(smoke_payload, dict)
    assert smoke_payload["n_samples"] == 10
    assert "per_sample" in smoke_payload
    assert "aggregate" in smoke_payload
    assert len(smoke_payload["per_sample"]) == 10


# ---------------------------------------------------------------------------
# 6. Smoke under one minute
# ---------------------------------------------------------------------------
def test_smoke_under_one_minute() -> None:
    """The 10-sample smoke run completes in < 60 s wall."""
    t0 = time.perf_counter()
    payload = run_benchmark(
        n_samples=10,
        csv_path=DEFAULT_PLATINAI_CSV,
        output_dir=DEFAULT_OUTPUT_DIR,
        max_depth=DEFAULT_MAX_DEPTH,
        max_products=DEFAULT_MAX_PRODUCTS,
        smoke=True,
    )
    wall = time.perf_counter() - t0
    assert wall < 60.0, (
        f"smoke run took {wall:.2f}s; budget is 60s"
    )
    assert payload["n_samples"] == 10


# ---------------------------------------------------------------------------
# 7. Per-sample record keys
# ---------------------------------------------------------------------------
EXPECTED_RECORD_KEYS: List[str] = [
    "idx", "smiles", "status", "closure_ok", "n_visited", "n_products",
    "max_depth_seen", "witness_rate", "well_typed_rate",
    "n_pt_coordinated", "n_atoms_start", "wall_time_s", "memory_mb",
    "depth", "parse_error",
]


def test_per_sample_record_keys(smoke_payload: Dict[str, Any]) -> None:
    """Every per-sample record carries every expected key."""
    for rec in smoke_payload["per_sample"]:
        missing = [k for k in EXPECTED_RECORD_KEYS if k not in rec]
        assert not missing, (
            f"per-sample record missing keys {missing!r}: {rec!r}"
        )


# ---------------------------------------------------------------------------
# 8. Status ∈ {ok, skipped, error}
# ---------------------------------------------------------------------------
def test_per_sample_status_values(smoke_payload: Dict[str, Any]) -> None:
    """Every per-sample status is one of the three contract values."""
    for rec in smoke_payload["per_sample"]:
        assert rec["status"] in ("ok", "skipped", "error"), (
            f"unexpected status {rec['status']!r} in record {rec['idx']}"
        )


# ---------------------------------------------------------------------------
# 9. closure_ok is bool
# ---------------------------------------------------------------------------
def test_per_sample_closure_ok_is_bool(smoke_payload: Dict[str, Any]) -> None:
    """closure_ok is always a real bool (not int / not str)."""
    for rec in smoke_payload["per_sample"]:
        assert isinstance(rec["closure_ok"], bool), (
            f"closure_ok is not bool for idx={rec['idx']}: "
            f"{type(rec['closure_ok']).__name__}={rec['closure_ok']!r}"
        )


# ---------------------------------------------------------------------------
# 10. wall_time_s ≥ 0
# ---------------------------------------------------------------------------
def test_per_sample_wall_time_non_negative(
    smoke_payload: Dict[str, Any],
) -> None:
    """Every per-sample wall_time_s is non-negative and finite."""
    for rec in smoke_payload["per_sample"]:
        w = float(rec["wall_time_s"])
        assert w >= 0.0, (
            f"negative wall_time_s for idx={rec['idx']}: {w}"
        )
        assert w < 1e9, (
            f"wall_time_s too large for idx={rec['idx']}: {w}"
        )


# ---------------------------------------------------------------------------
# 11. memory_mb ≥ 0
# ---------------------------------------------------------------------------
def test_per_sample_memory_non_negative(
    smoke_payload: Dict[str, Any],
) -> None:
    """Every per-sample memory_mb is non-negative and finite."""
    for rec in smoke_payload["per_sample"]:
        m = float(rec["memory_mb"])
        assert m >= 0.0, (
            f"negative memory_mb for idx={rec['idx']}: {m}"
        )
        assert m < 1e9, (
            f"memory_mb too large for idx={rec['idx']}: {m}"
        )


# ---------------------------------------------------------------------------
# 12. max_depth_seen ≤ max_depth
# ---------------------------------------------------------------------------
def test_per_sample_max_depth_seen_bounded(
    smoke_payload: Dict[str, Any],
) -> None:
    """max_depth_seen ≤ max_depth (the BFS respects its budget)."""
    max_depth = int(smoke_payload["max_depth"])
    for rec in smoke_payload["per_sample"]:
        d = int(rec["max_depth_seen"])
        assert d <= max_depth, (
            f"max_depth_seen={d} > max_depth={max_depth} for idx={rec['idx']}"
        )


# ---------------------------------------------------------------------------
# 13. Aggregate keys present
# ---------------------------------------------------------------------------
EXPECTED_AGGREGATE_KEYS: List[str] = [
    "n_samples", "n_parsed", "n_closure_ok", "closure_success_rate",
    "mean_beta_depth", "mean_wall_time", "std_wall_time",
    "median_wall_time", "peak_memory_mb", "total_wall_time",
    "n_error", "n_status_ok",
]


def test_aggregate_keys_present(smoke_payload: Dict[str, Any]) -> None:
    """The aggregate dict carries every documented key."""
    agg = smoke_payload["aggregate"]
    missing = [k for k in EXPECTED_AGGREGATE_KEYS if k not in agg]
    assert not missing, (
        f"aggregate missing keys {missing!r}; got {sorted(agg.keys())!r}"
    )


# ---------------------------------------------------------------------------
# 14. closure_success_rate ∈ [0, 1]
# ---------------------------------------------------------------------------
def test_closure_success_rate_in_unit(smoke_payload: Dict[str, Any]) -> None:
    """closure_success_rate is a probability in the closed interval [0, 1]."""
    rate = float(smoke_payload["aggregate"]["closure_success_rate"])
    assert 0.0 <= rate <= 1.0, (
        f"closure_success_rate out of [0, 1]: {rate}"
    )


# ---------------------------------------------------------------------------
# 15. mean_beta_depth ≥ 0
# ---------------------------------------------------------------------------
def test_mean_beta_depth_non_negative(smoke_payload: Dict[str, Any]) -> None:
    """mean_beta_depth is a non-negative finite float."""
    d = float(smoke_payload["aggregate"]["mean_beta_depth"])
    assert d >= 0.0, f"mean_beta_depth is negative: {d}"
    assert d < 1e6, f"mean_beta_depth unreasonably large: {d}"


# ---------------------------------------------------------------------------
# 16. mean_wall_time ≥ 0
# ---------------------------------------------------------------------------
def test_mean_wall_time_non_negative(smoke_payload: Dict[str, Any]) -> None:
    """mean_wall_time is a non-negative finite float."""
    w = float(smoke_payload["aggregate"]["mean_wall_time"])
    assert w >= 0.0, f"mean_wall_time is negative: {w}"
    assert w < 1e9, f"mean_wall_time unreasonably large: {w}"


# ---------------------------------------------------------------------------
# 17. peak_memory_mb ≥ 0
# ---------------------------------------------------------------------------
def test_peak_memory_non_negative(smoke_payload: Dict[str, Any]) -> None:
    """peak_memory_mb is a non-negative finite float."""
    m = float(smoke_payload["aggregate"]["peak_memory_mb"])
    assert m >= 0.0, f"peak_memory_mb is negative: {m}"
    assert m < 1e6, f"peak_memory_mb unreasonably large: {m}"


# ---------------------------------------------------------------------------
# 18. Status counts sum to n_samples
# ---------------------------------------------------------------------------
def test_status_counts_sum_to_n_samples(
    smoke_payload: Dict[str, Any],
) -> None:
    """status_counts values sum to n_samples."""
    counts = smoke_payload["status_counts"]
    n = sum(int(v) for v in counts.values())
    assert n == int(smoke_payload["n_samples"]), (
        f"status counts sum to {n}, expected n_samples="
        f"{smoke_payload['n_samples']}"
    )


# ---------------------------------------------------------------------------
# 19. benchmark.json exists + valid JSON
# ---------------------------------------------------------------------------
def test_benchmark_json_written(smoke_paths: Dict[str, Path]) -> None:
    """save_benchmark writes a valid benchmark.json."""
    json_path = smoke_paths["json"]
    assert json_path.is_file(), f"benchmark.json missing at {json_path}"
    with json_path.open("r") as fh:
        payload = json.load(fh)
    assert payload["schema_version"] == 1
    assert payload["tool"] == "closure_real_benchmark.py"
    assert payload["pilot_id"] == "wf_closure_real"
    assert "aggregate" in payload
    assert "per_sample" in payload
    assert payload["n_samples"] == len(payload["per_sample"])


# ---------------------------------------------------------------------------
# 20. per_sample.csv exists + correct header
# ---------------------------------------------------------------------------
def test_per_sample_csv_written(smoke_paths: Dict[str, Path]) -> None:
    """save_benchmark writes a per_sample.csv with the documented header."""
    csv_path = smoke_paths["csv"]
    assert csv_path.is_file(), f"per_sample.csv missing at {csv_path}"
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        assert "idx" in header
        assert "smiles" in header
        assert "status" in header
        assert "closure_ok" in header
        assert "wall_time_s" in header
        assert "memory_mb" in header
        rows = list(reader)
    assert len(rows) == 10, (
        f"per_sample.csv should have 10 rows, got {len(rows)}"
    )


# ---------------------------------------------------------------------------
# Bonus invariants exercised through private helpers — pinned for
# the next regression that tries to alter the aggregate schema.
# ---------------------------------------------------------------------------
def test_load_platinai_smiles_first_n() -> None:
    """_load_platinai_smiles returns the first n SMILES from the CSV."""
    smiles_list = _load_platinai_smiles(DEFAULT_PLATINAI_CSV, 5)
    assert len(smiles_list) == 5
    assert all(len(s) > 0 for s in smiles_list)


def test_build_partner_terms_count() -> None:
    """_build_partner_terms returns >= 6 partner tiles."""
    partners = _build_partner_terms()
    assert len(partners) >= 6, (
        f"expected >= 6 partner terms, got {len(partners)}"
    )


def test_build_click_rules_count() -> None:
    """_build_click_rules returns the 5 canonical click rules."""
    rules = _build_click_rules()
    assert len(rules) == 5
    names = {getattr(r, "name", type(r).__name__) for r in rules}
    assert {"CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"}.issubset(
        names
    ), f"missing click rule names: {names!r}"


def test_partner_smiles_list_matches_default() -> None:
    """PARTNER_SMILE_LIST has the documented 7 tiles."""
    assert len(PARTNER_SMILE_LIST) == 7, (
        f"PARTNER_SMILE_LIST should have 7 entries, got {len(PARTNER_SMILE_LIST)}"
    )


def test_aggregate_empty_records() -> None:
    """_aggregate on an empty list returns the zero-vector aggregate."""
    agg = _aggregate([])
    assert agg["n_samples"] == 0
    assert agg["closure_success_rate"] == 0.0
    assert agg["mean_beta_depth"] == 0.0
    assert agg["mean_wall_time"] == 0.0
    assert agg["peak_memory_mb"] == 0.0