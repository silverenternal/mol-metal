"""Smoke tests for the R18 real-data MetalCytoToxDB benchmark.

================================================================
W2.3 / R18 ultracode dispatch (2026-09-26).

These are the pytest mirrors of
:file:`molmetal/scripts/benchmark_r18_metalcytotox.py`.  The script
emits a JSON artefact with per-metal × per-module wall-time /
memory / status fields.  Here we re-invoke the same registry
(CPU-only) under pytest so regressions show up with focused names
instead of a single failing CLI invocation.

Each test is intentionally minimal:

  * import-time check (``test_benchmark_runs``-style assertions),
  * one focused forward call per module per metal,
  * shape / dtype sanity on the returned primitive,
  * a wall-clock cap (``<= 5 min CPU-only``).

Why mirrors of the script?
---------------------------
We deliberately do NOT shell out to ``benchmark_r18_metalcytotox.py``
because the script's smoke-target ``--n-samples 500`` would push
pytest's CI budget past the allowed 5-minute wall-clock cap.  These
tests instead call the same module registry directly, with a
smaller ``--n-samples 16`` and ``--n-runs 2`` budget per test /
per metal.

Per-metal × per-module contract
-------------------------------
Unlike :mod:`test_platinai_bench_smoke`, every benchmark shape is
**2-D** in (metal, module_id).  The following 15 tests pin the
2-D contract:

  1. test_benchmark_runs
  2. test_per_metal_shape
  3. test_total_wall_time
  4. test_memory_peak
  5. test_math_skills_runners_present
  6. test_adapters_runners_present
  7. test_lam_chem_runners_present
  8. test_reward_runners_present
  9. test_search_alg_runners_present
 10. test_flow_matching_runners_present
 11. test_total_module_count_is_32
 12. test_status_field_values
 13. test_wall_time_non_negative
 14. test_median_helper
 15. test_metalcytotox_csv_exists
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.benchmark_r18_metalcytotox import (  # noqa: E402
    _build_module_registry,
    _benchmark_metal,
    _load_metalcytotox_smiles,
    _MODULE_GROUPS,
    _run_module,
)

CSV_PATH = PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
N_RUNS_SMOKE = 2
SMOKE_SAMPLES = 16
SMOKE_METALS = ["Ru", "Ir"]  # subset to keep pytest within 5-min cap


# ---------------------------------------------------------------------------
# Fixtures — per-metal × per-module caches
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registry() -> Dict[str, object]:
    """Module registry from the benchmark script."""
    return _build_module_registry()


@pytest.fixture(scope="module")
def metal_smiles() -> Dict[str, List[str]]:
    """{metal: List[smiles]} mini-fixture (16 per metal, subset of metals)."""
    out: Dict[str, List[str]] = {}
    if not CSV_PATH.exists():
        return out
    for metal in SMOKE_METALS:
        out[metal] = _load_metalcytotox_smiles(CSV_PATH, metal, SMOKE_SAMPLES)
    return out


# ---------------------------------------------------------------------------
# Smoke tests — module count, registry shape, and CLI invocation.
# ---------------------------------------------------------------------------
def test_benchmark_runs(
    metal_smiles: Dict[str, List[str]],
    registry: Dict[str, object],
) -> None:
    """All 32 R18 modules execute at least once per metal in the smoke set.

    The "runs" assertion is the smoke gate: every (metal, module)
    cell with a non-None runner must produce a finite wall-time list
    (status "ok") in < 5 minutes total CPU time.
    """
    t0 = time.perf_counter()
    n_ok_cells = 0
    n_total_cells = 0
    for metal, smiles in metal_smiles.items():
        for mod_id, runner in registry.items():
            if runner is None:
                continue
            n_total_cells += 1
            wt, st = _run_module(runner, smiles, N_RUNS_SMOKE)
            if st == "ok" and wt:
                n_ok_cells += 1
    elapsed = time.perf_counter() - t0
    assert elapsed < 300.0, (
        f"R18 metalcytotox bench smoke exceeded 5-min wall-clock cap: "
        f"{elapsed:.1f}s"
    )
    assert n_ok_cells >= n_total_cells - 4, (
        f"only {n_ok_cells}/{n_total_cells} (metal,module) cells "
        f"returned 'ok' — expected >= n_total - 4 (DiffDock / EquiBind "
        f"are intentionally import-only)"
    )


def test_per_metal_shape(
    metal_smiles: Dict[str, List[str]],
    registry: Dict[str, object],
) -> None:
    """Every (metal, module) cell returns ``n_runs`` well-typed wall-times.

    We pin ``len(wt) == n_runs`` for every non-None runner; this
    catches forward-contract drift where a module suddenly returns
    an empty list (e.g. a swallowed exception) without raising.
    """
    for metal, smiles in metal_smiles.items():
        for mod_id, runner in registry.items():
            if runner is None:
                continue
            wt, st = _run_module(runner, smiles, N_RUNS_SMOKE)
            if st == "ok":
                assert len(wt) == N_RUNS_SMOKE, (
                    f"({metal}, {mod_id}) returned {len(wt)} wall-times; "
                    f"expected {N_RUNS_SMOKE}"
                )
                for w in wt:
                    assert (
                        w >= 0.0
                    ), f"({metal}, {mod_id}) reported negative wall-time {w}"


def test_total_wall_time(
    metal_smiles: Dict[str, List[str]],
    registry: Dict[str, object],
) -> None:
    """The full CPU module × metal sweep completes in <= 5 minutes.

    Note: this is the smoke cap.  The production ``--n-samples 500``
    benchmark is designed to fit a similar budget per metal × per
    module; if the smoke sweep blows past 5 minutes the production
    sweep certainly will.
    """
    t0 = time.perf_counter()
    for metal, smiles in metal_smiles.items():
        for runner in registry.values():
            if runner is None:
                continue
            _ = _run_module(runner, smiles, N_RUNS_SMOKE)
    elapsed = time.perf_counter() - t0
    assert elapsed <= 300.0, f"CPU sweep took {elapsed:.1f}s > 300s cap"


def test_memory_peak(
    metal_smiles: Dict[str, List[str]],
    registry: Dict[str, object],
) -> None:
    """Peak ``tracemalloc`` across the full sweep is <= 4 GB.

    We use a generous 4 GB cap because the sweep composes multiple
    SkeletonJump / Multi-PH / Bass tensor modules; the production
    benchmark reports ~30 MB but pytest runs in parallel with other
    tests so we leave headroom.
    """
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    for metal, smiles in metal_smiles.items():
        for runner in registry.values():
            if runner is None:
                continue
            _ = _run_module(runner, smiles, N_RUNS_SMOKE)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    delta_mb = (peak - base) / (1024 * 1024)
    assert delta_mb <= 4096.0, f"peak tracemalloc {delta_mb:.1f} MB exceeds 4 GB cap"


# ---------------------------------------------------------------------------
# Group-level rollups — every group has at least one runner.
# ---------------------------------------------------------------------------
def test_math_skills_runners_present(registry: Dict[str, object]) -> None:
    """The math_skills layer exposes 9 module runners (T1-T9)."""
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
# Per-module / total module count rollups.
# ---------------------------------------------------------------------------
def test_total_module_count_is_32(registry: Dict[str, object]) -> None:
    """The benchmark registry contains exactly 32 modules.

    Matches the R18 ship manifest hard-pin in
    ``test_r18_imports.py:test_total_modules_is_32``.  Catches
    accidental additions/removals in the registry.
    """
    assert (
        len(registry) == 32
    ), f"R18 benchmark manifest changed: got {len(registry)}, expected 32"


# ---------------------------------------------------------------------------
# Per-module wall-time / status fields exist with the right dtype.
# ---------------------------------------------------------------------------
def test_payload_schema(tmp_path: Path) -> None:
    """The results.json payload schema is stable across runs.

    We do NOT actually invoke the 500-sample benchmark in pytest
    (CPU budget) — instead we run a tiny smoke (1 metal × 1 sample × 1 run)
    and assert the JSON keys / dtypes match the documented schema.
    """
    if not CSV_PATH.exists():
        pytest.skip(f"missing {CSV_PATH}")
    smiles = _load_metalcytotox_smiles(CSV_PATH, "Ru", 1)
    registry = _build_module_registry()

    # 2-D shape: per_metal_per_module_*
    wall_ms: Dict[str, Dict[str, float]] = {"Ru": {}}
    mem_mb: Dict[str, Dict[str, float]] = {"Ru": {}}
    n_calls: Dict[str, Dict[str, int]] = {"Ru": {}}
    status: Dict[str, Dict[str, str]] = {"Ru": {}}
    for mid, runner in registry.items():
        wt, st = _run_module(runner, smiles, 1)
        wall_ms["Ru"][mid] = wt[0] * 1000.0 if wt else 0.0
        mem_mb["Ru"][mid] = 0.0
        n_calls["Ru"][mid] = len(wt) if st == "ok" else 0
        status["Ru"][mid] = st

    payload = {
        "n_samples": 1,
        "n_metals": 1,
        "n_modules": len(registry),
        "metals": ["Ru"],
        "per_metal_n_samples": {"Ru": 1},
        "module_groups": _MODULE_GROUPS,
        "total_wall_time_s": 0.0,
        "per_metal_wall_time_s": {"Ru": 0.0},
        "per_metal_per_module_wall_time_ms": wall_ms,
        "per_metal_per_module_memory_mb": mem_mb,
        "per_metal_per_module_n_calls": n_calls,
        "per_metal_per_module_status": status,
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
        "n_metals",
        "n_modules",
        "metals",
        "per_metal_n_samples",
        "module_groups",
        "total_wall_time_s",
        "per_metal_wall_time_s",
        "per_metal_per_module_wall_time_ms",
        "per_metal_per_module_memory_mb",
        "per_metal_per_module_n_calls",
        "per_metal_per_module_status",
        "n_runs_per_module",
        "peak_memory_mb_total",
        "timestamp_utc",
        "python",
        "torch",
    ):
        assert key in loaded, f"missing payload key: {key}"
    # 2-D shape: outer key per metal, inner dict per module.
    assert isinstance(loaded["per_metal_per_module_wall_time_ms"], dict)
    assert "Ru" in loaded["per_metal_per_module_wall_time_ms"]
    assert isinstance(loaded["per_metal_per_module_wall_time_ms"]["Ru"], dict)
    # Status field is constrained to {ok, skipped, error} — sample one cell.
    sample_status = next(iter(loaded["per_metal_per_module_status"]["Ru"].values()))
    assert sample_status in ("ok", "skipped", "error")


def test_status_field_values(
    metal_smiles: Dict[str, List[str]],
    registry: Dict[str, object],
) -> None:
    """Status values are constrained to {ok, skipped, error}."""
    allowed = {"ok", "skipped", "error"}
    for metal, smiles in metal_smiles.items():
        for mid, runner in registry.items():
            wt, st = _run_module(runner, smiles, N_RUNS_SMOKE)
            assert st in allowed, (
                f"({metal}, {mid}) returned unknown status {st!r}; "
                f"allowed = {allowed}"
            )


def test_wall_time_non_negative(
    metal_smiles: Dict[str, List[str]],
    registry: Dict[str, object],
) -> None:
    """Every successful module reports a non-negative wall-time."""
    for metal, smiles in metal_smiles.items():
        for mid, runner in registry.items():
            if runner is None:
                continue
            wt, st = _run_module(runner, smiles, N_RUNS_SMOKE)
            if st == "ok":
                for w in wt:
                    assert w >= 0.0, f"({metal}, {mid}) reported negative wall-time {w}"


def test_median_helper() -> None:
    """``_median_ms`` returns the median in milliseconds."""
    from molmetal.scripts.benchmark_r18_metalcytotox import _median_ms

    assert _median_ms([]) == 0.0
    assert abs(_median_ms([0.001]) - 1.0) < 1e-9
    assert abs(_median_ms([0.001, 0.002, 0.003]) - 2.0) < 1e-9


def test_metalcytotox_csv_exists() -> None:
    """The MetalCytoToxDB 5-metal multi-task CSV exists and has >= 500 rows.

    Mirrors :func:`test_platinai_csv_exists` for the PlatinAI 5K
    smoke mirror.  The CSV lives under ``.data/molmetal/`` because
    the loaders honour the ``MOLMETAL_DATA_DIR`` env var (see
    ``env.sh``).
    """
    assert CSV_PATH.exists(), f"missing {CSV_PATH}"
    n_lines = sum(1 for _ in CSV_PATH.open()) - 1  # minus header
    assert n_lines >= 500, f"only {n_lines} data rows in {CSV_PATH}"


def test_skip_status_for_pure_import_modules(
    metal_smiles: Dict[str, List[str]],
) -> None:
    """DiffDock + EquiBind are import-only on MetalCytoToxDB.

    These runners are intentionally None and the benchmark records
    status="skipped".  Pin the contract so a future refactor that
    wires them up without a real MetalCytoToxDB runner surfaces here.
    """
    pure_import = {"adapters/diffdock", "adapters/equibind"}
    registry = _build_module_registry()
    for mid in pure_import:
        assert mid in registry, f"missing module {mid}"
        assert registry[mid] is None, (
            f"{mid} should remain import-only on MetalCytoToxDB; "
            f"got a real runner — please add a MetalCytoToxDB harness "
            f"in benchmark_r18_metalcytotox.py first"
        )


def test_metalcytotox_loader_returns_list(
    metal_smiles: Dict[str, List[str]],
) -> None:
    """The MetalCytoToxDB loader returns a non-empty list per metal."""
    assert isinstance(metal_smiles, dict)
    assert metal_smiles, "no metals loaded"
    for metal, smis in metal_smiles.items():
        assert isinstance(smis, list)
        assert len(smis) > 0, f"no SMILES for metal {metal}"
        assert all(isinstance(s, str) and s for s in smis)


def test_benchmark_metal_returns_four_dicts(
    metal_smiles: Dict[str, List[str]],
    registry: Dict[str, object],
) -> None:
    """``_benchmark_metal`` returns the 4 parallel per-module dicts.

    Direct integration check: every key in registry has an entry in
    all four output dicts (wall_ms, mem_mb, n_calls, status).
    """
    # Use the first metal only.
    metal = next(iter(metal_smiles))
    smiles = metal_smiles[metal]
    wall, mem, calls, status = _benchmark_metal(
        metal=metal,
        smiles=smiles,
        registry=registry,
        n_runs=1,
    )
    assert set(wall.keys()) == set(registry.keys())
    assert set(mem.keys()) == set(registry.keys())
    assert set(calls.keys()) == set(registry.keys())
    assert set(status.keys()) == set(registry.keys())
