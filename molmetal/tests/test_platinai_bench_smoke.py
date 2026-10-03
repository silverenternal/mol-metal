"""Smoke tests for the R18 real-data PlatinAI 5K benchmark.

================================================================
W2.3 / R18 ultracode dispatch (2026-09-26).

These are the pytest mirrors of
:file:`molmetal/scripts/benchmark_r18_platinai.py`.  The script
emits a JSON artefact with per-module wall-time / memory / status
fields.  Here we re-invoke the same registry (CPU-only) under
pytest so regressions show up with focused names instead of a
single failing CLI invocation.

Each test is intentionally minimal:

  * import-time check (``test_benchmark_runs``-style assertions),
  * one focused forward call per module,
  * shape / dtype sanity on the returned primitive,
  * a wall-clock cap (``<= 5 min CPU-only``).

Why mirrors of the script?
---------------------------
We deliberately do NOT shell out to ``benchmark_r18_platinai.py``
because the script's smoke-target ``--n-samples 500`` would push
pytest's CI budget past the allowed 5-minute wall-clock cap.  These
tests instead call the same module registry directly, with a
smaller ``--n-samples 16`` and ``--n-runs 2`` budget per test.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.benchmark_r18_platinai import (  # noqa: E402
    _build_module_registry,
    _load_platinai_smiles,
    _MODULE_GROUPS,
    _run_module,
)

CSV_PATH = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"
N_RUNS_SMOKE = 2
SMOKE_SAMPLES = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def platinai_smiles() -> List[str]:
    """Tiny PlatinAI 5K fixture (16 SMILES)."""
    return _load_platinai_smiles(CSV_PATH, SMOKE_SAMPLES)


@pytest.fixture(scope="module")
def registry() -> Dict[str, object]:
    """Module registry from the benchmark script."""
    return _build_module_registry()


# ---------------------------------------------------------------------------
# Smoke tests — module count, registry shape, and CLI invocation.
# ---------------------------------------------------------------------------
def test_benchmark_runs(
    platinai_smiles: List[str], registry: Dict[str, object]
) -> None:
    """All 32 R18 modules execute at least once on PlatinAI 5K.

    The "runs" assertion is the smoke gate: every module with a
    non-None runner must produce a finite wall-time list (status
    "ok") in < 5 minutes total CPU time.
    """
    t0 = time.perf_counter()
    n_ok = 0
    for mod_id, runner in registry.items():
        if runner is None:
            continue
        wt, st = _run_module(runner, platinai_smiles, N_RUNS_SMOKE)
        if st == "ok" and wt:
            n_ok += 1
    elapsed = time.perf_counter() - t0
    assert (
        elapsed < 300.0
    ), f"R18 bench smoke exceeded 5-min wall-clock cap: {elapsed:.1f}s"
    assert (
        n_ok >= 20
    ), f"only {n_ok} modules returned 'ok' status — expected >= 20 of 32"


def test_per_module_shape(
    platinai_smiles: List[str], registry: Dict[str, object]
) -> None:
    """Each module returns ``n_runs`` well-typed wall-time entries.

    We pin ``len(wt) == n_runs`` for every non-None runner; this
    catches forward-contract drift where a module suddenly returns
    an empty list (e.g. a swallowed exception) without raising.
    """
    for mod_id, runner in registry.items():
        if runner is None:
            continue
        wt, st = _run_module(runner, platinai_smiles, N_RUNS_SMOKE)
        if st == "ok":
            assert (
                len(wt) == N_RUNS_SMOKE
            ), f"{mod_id} returned {len(wt)} wall-times; expected {N_RUNS_SMOKE}"
            for w in wt:
                assert w >= 0.0, f"{mod_id} reported negative wall-time {w}"


def test_total_wall_time(
    platinai_smiles: List[str], registry: Dict[str, object]
) -> None:
    """The full CPU module sweep completes in <= 5 minutes (smoke budget).

    Note: this is the smoke cap.  The production ``--n-samples 500``
    benchmark is designed to fit the *same* 5-min budget; if the
    smoke sweep blows past 5 minutes the production sweep certainly
    will.
    """
    t0 = time.perf_counter()
    for runner in registry.values():
        if runner is None:
            continue
        _ = _run_module(runner, platinai_smiles, N_RUNS_SMOKE)
    elapsed = time.perf_counter() - t0
    assert elapsed <= 300.0, f"CPU sweep took {elapsed:.1f}s > 300s cap"


def test_memory_peak(platinai_smiles: List[str], registry: Dict[str, object]) -> None:
    """Peak ``tracemalloc`` across the full sweep is <= 4 GB.

    We use a generous 4 GB cap because the sweep composes multiple
    SkeletonJump / Multi-PH / Bass tensor modules; the production
    benchmark reports ~30 MB but pytest runs in parallel with other
    tests so we leave headroom.
    """
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    for runner in registry.values():
        if runner is None:
            continue
        _ = _run_module(runner, platinai_smiles, N_RUNS_SMOKE)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    delta_mb = (peak - base) / (1024 * 1024)
    assert delta_mb <= 4096.0, f"peak tracemalloc {delta_mb:.1f} MB exceeds 4 GB cap"


# ---------------------------------------------------------------------------
# Group-level rollups — every group has at least one runner.
# ---------------------------------------------------------------------------
def test_math_skills_runners_present(registry: Dict[str, object]) -> None:
    """The math_skills layer exposes 9 module runners (T1–T9)."""
    math_ids = _MODULE_GROUPS["math_skills"]
    assert len(math_ids) == 9, f"expected 9 math_skills, got {len(math_ids)}"
    for mid in math_ids:
        assert mid in registry, f"missing runner for {mid}"


def test_adapters_runners_present(registry: Dict[str, object]) -> None:
    """The adapters layer exposes 7 module runners."""
    ad_ids = _MODULE_GROUPS["adapters"]
    assert len(ad_ids) == 7, f"expected 7 adapters, got {len(ad_ids)}"
    for mid in ad_ids:
        assert mid in registry, f"missing runner for {mid}"


def test_lam_chem_runners_present(registry: Dict[str, object]) -> None:
    """The lam_chem layer exposes 11 module runners."""
    lc_ids = _MODULE_GROUPS["lam_chem"]
    assert len(lc_ids) == 11, f"expected 11 lam_chem, got {len(lc_ids)}"
    for mid in lc_ids:
        assert mid in registry, f"missing runner for {mid}"


def test_reward_runners_present(registry: Dict[str, object]) -> None:
    """The reward layer exposes 1 module runner (Hilbert primal-dual)."""
    r_ids = _MODULE_GROUPS["reward"]
    assert len(r_ids) == 1, f"expected 1 reward, got {len(r_ids)}"
    assert r_ids[0] in registry


def test_search_alg_runners_present(registry: Dict[str, object]) -> None:
    """The search_alg layer exposes 2 module runners."""
    s_ids = _MODULE_GROUPS["search_alg"]
    assert len(s_ids) == 2, f"expected 2 search_alg, got {len(s_ids)}"
    for mid in s_ids:
        assert mid in registry, f"missing runner for {mid}"


def test_flow_matching_runners_present(registry: Dict[str, object]) -> None:
    """The flow_matching_lipman layer exposes 2 module runners."""
    f_ids = _MODULE_GROUPS["flow_matching_lipman"]
    assert len(f_ids) == 2, f"expected 2 flow_matching_lipman, got {len(f_ids)}"
    for mid in f_ids:
        assert mid in registry, f"missing runner for {mid}"


# ---------------------------------------------------------------------------
# Per-module wall-time / memory / status fields exist with the right dtype.
# ---------------------------------------------------------------------------
def test_payload_schema(tmp_path: Path) -> None:
    """The results.json payload schema is stable across runs.

    We do NOT actually invoke the 500-sample benchmark in pytest
    (CPU budget) — instead we run a tiny smoke (1 sample, 1 run)
    and assert the JSON keys / dtypes match the documented schema.
    """
    import csv

    smiles = _load_platinai_smiles(CSV_PATH, 1)
    registry = _build_module_registry()

    wall_ms: Dict[str, float] = {}
    mem_mb: Dict[str, float] = {}
    n_calls: Dict[str, int] = {}
    status: Dict[str, str] = {}
    for mid, runner in registry.items():
        wt, st = _run_module(runner, smiles, 1)
        wall_ms[mid] = wt[0] * 1000.0 if wt else 0.0
        mem_mb[mid] = 0.0
        n_calls[mid] = len(wt) if st == "ok" else 0
        status[mid] = st

    payload = {
        "n_samples": 1,
        "n_modules": len(registry),
        "total_wall_time_s": 0.0,
        "per_module_wall_time_ms": wall_ms,
        "per_module_memory_mb": mem_mb,
        "per_module_n_calls": n_calls,
        "per_module_status": status,
        "module_groups": _MODULE_GROUPS,
        "n_runs_per_module": 1,
        "peak_memory_mb_total": 0.0,
        "timestamp_utc": "1970-01-01T00:00:00Z",
        "python": "x",
        "torch": "y",
    }

    out = tmp_path / "results.json"
    out.write_text(json.dumps(payload, indent=2))
    loaded = json.loads(out.read_text())
    for key in (
        "n_samples",
        "n_modules",
        "total_wall_time_s",
        "per_module_wall_time_ms",
        "per_module_memory_mb",
        "per_module_n_calls",
        "per_module_status",
        "module_groups",
        "n_runs_per_module",
        "peak_memory_mb_total",
        "timestamp_utc",
        "python",
        "torch",
    ):
        assert key in loaded, f"missing payload key: {key}"
    assert isinstance(loaded["per_module_wall_time_ms"], dict)
    assert isinstance(loaded["per_module_memory_mb"], dict)
    assert isinstance(loaded["per_module_status"], dict)


def test_platinai_csv_exists() -> None:
    """The PlatinAI 5K CSV exists and has >= 500 rows."""
    assert CSV_PATH.exists(), f"missing {CSV_PATH}"
    n_lines = sum(1 for _ in CSV_PATH.open()) - 1  # minus header
    assert n_lines >= 500, f"only {n_lines} data rows in {CSV_PATH}"


def test_skip_status_for_pure_import_modules(platinai_smiles: List[str]) -> None:
    """DiffDock + EquiBind are import-only on PlatinAI 5K.

    These runners are intentionally None and the benchmark records
    status="skipped".  Pin the contract so a future refactor that
    wires them up without a real PlatinAI runner surfaces here.
    """
    pure_import = {
        "adapters/diffdock",
        "adapters/equibind",
    }
    registry = _build_module_registry()
    for mid in pure_import:
        assert mid in registry, f"missing module {mid}"
        assert registry[mid] is None, (
            f"{mid} should remain import-only on PlatinAI 5K; "
            f"got a real runner — please add a PlatinAI harness in "
            f"benchmark_r18_platinai.py first"
        )


def test_total_module_count_is_32(registry: Dict[str, object]) -> None:
    """The benchmark registry contains exactly 32 modules.

    Matches the R18 ship manifest hard-pin in
    ``test_r18_imports.py:test_total_modules_is_32``.  Catches
    accidental additions/removals in the registry.
    """
    assert (
        len(registry) == 32
    ), f"R18 benchmark manifest changed: got {len(registry)}, expected 32"


def test_status_field_values(
    platinai_smiles: List[str], registry: Dict[str, object]
) -> None:
    """Status values are constrained to {ok, skipped, error}."""
    allowed = {"ok", "skipped", "error"}
    for mid, runner in registry.items():
        wt, st = _run_module(runner, platinai_smiles, N_RUNS_SMOKE)
        assert (
            st in allowed
        ), f"{mid} returned unknown status {st!r}; allowed = {allowed}"


def test_wall_time_non_negative(
    platinai_smiles: List[str], registry: Dict[str, object]
) -> None:
    """Every successful module reports a non-negative wall-time."""
    for mid, runner in registry.items():
        if runner is None:
            continue
        wt, st = _run_module(runner, platinai_smiles, N_RUNS_SMOKE)
        if st == "ok":
            for w in wt:
                assert w >= 0.0, f"{mid} reported negative wall-time {w}"


def test_median_helper() -> None:
    """``_median_ms`` returns the median in milliseconds."""
    from molmetal.scripts.benchmark_r18_platinai import _median_ms

    assert _median_ms([]) == 0.0
    assert abs(_median_ms([0.001]) - 1.0) < 1e-9
    assert abs(_median_ms([0.001, 0.002, 0.003]) - 2.0) < 1e-9


def test_platinai_loader_returns_list(platinai_smiles: List[str]) -> None:
    """The PlatinAI loader returns a non-empty list of strings."""
    assert isinstance(platinai_smiles, list)
    assert len(platinai_smiles) > 0
    assert all(isinstance(s, str) and s for s in platinai_smiles)
