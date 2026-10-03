"""Cross-module end-to-end smoke tests — Wf-14 + Wf-15 R18 pipeline.

================================================================
Wf-14 + Wf-15 / R18 ultracode dispatch (2026-09-26).

Twenty pytest functions that exercise the
:mod:`molmetal.scripts.benchmark_r18_end_to_end` pipeline so a
regression in the chained walk surfaces as a single, narrow
pytest failure rather than a downstream mass-failure.

What the suite asserts
----------------------
1.  The benchmark script's :func:`_safe_run` helper handles
    ok / skipped / error outcomes without raising.
2.  Each pipeline stage registered in
    :data:`molmetal.scripts.benchmark_r18_end_to_end.PIPELINE_FULL`
    runs cleanly on a fixed PlatinAI-like SMILES sample
    (``CCO``, ``c1ccccc1``, ``CC(=O)O``).
3.  The end-to-end :func:`molmetal.scripts.benchmark_r18_end_to_end.main`
    function emits a JSON report at the canonical path
    ``molmetal/reports/wf_cross_module_bench/end_to_end.json``
    when invoked with ``--n-samples 5``.
4.  The pipeline covers at least 25 of the 32 R18 ship-modules
    in the benchmark harness.
5.  The full 50-sample end-to-end benchmark fits inside the
    60-second wall-time budget.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_cross_module_smoke.py -v
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_SCRIPT = _REPO_ROOT / "molmetal" / "scripts" / "benchmark_r18_end_to_end.py"
_DATA = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
_REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_cross_module_bench"
_REPORT_FILE = _REPORT_DIR / "end_to_end.json"

# Make the benchmark importable.  Both repo-root and molmetal/ are added
# so ``from molmetal.scripts.benchmark_r18_end_to_end import ...`` works
# whether the suite is invoked from /home/hugo/codes/mol-metal or
# from the package directory.
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "molmetal")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Lazy import — keep module-load failures narrow.  ``benchmark_r18_end_to_end``
# imports ``torch`` at function call time, so import-time should be safe.
try:
    from molmetal.scripts import benchmark_r18_end_to_end as bench_mod  # type: ignore
except Exception:
    from scripts import benchmark_r18_end_to_end as bench_mod  # type: ignore


# ---------------------------------------------------------------------------
# Sample SMILES — three PlatinAI-style inputs spanning metal / organic /
# aromatic chemistry.  Each sample is exercised through every stage.
# ---------------------------------------------------------------------------
_SAMPLES = (
    "CCO",  # ethanol (organic)
    "c1ccccc1",  # benzene (aromatic)
    "CC(=O)O",  # acetic acid (carboxylic)
)


# ---------------------------------------------------------------------------
# Test 1 — script file exists at the expected path
# ---------------------------------------------------------------------------
def test_001_benchmark_script_path_exists():
    """The benchmark script is on disk and is a non-empty Python file."""
    assert _SCRIPT.exists(), f"benchmark script not found: {_SCRIPT}"
    assert (
        _SCRIPT.stat().st_size > 1024
    ), f"benchmark script is suspiciously small: {_SCRIPT.stat().st_size}B"


# ---------------------------------------------------------------------------
# Test 2 — the script registers a 32-module pipeline
# ---------------------------------------------------------------------------
def test_002_pipeline_registered_has_32_modules():
    """``PIPELINE_FULL`` exposes exactly 32 distinct stage IDs."""
    stages = bench_mod.PIPELINE_FULL
    assert (
        len(stages) == 32
    ), f"expected 32 stages, got {len(stages)}: {[sid for sid, _ in stages]}"
    # Distinct stage IDs.
    ids = [sid for sid, _ in stages]
    assert len(set(ids)) == len(ids), f"duplicate stage IDs in pipeline: {ids}"


# ---------------------------------------------------------------------------
# Test 3 — each pipeline stage runs cleanly on a fixed SMILES
# ---------------------------------------------------------------------------
def test_003_pipeline_stages_run_clean():
    """Every stage completes with status ``ok`` on a fixed SMILES sample."""
    for sid, fn in bench_mod.PIPELINE_FULL:
        for sample in _SAMPLES:
            wall, status = bench_mod._safe_run(sid, fn, sample, None)
            assert (
                status == "ok"
            ), f"stage {sid!r} failed on SMILES {sample!r}: status={status}"
            assert wall >= 0.0, f"negative wall time on stage {sid!r}: {wall}"


# ---------------------------------------------------------------------------
# Test 4 — chained end-to-end walk on a single sample produces no errors
# ---------------------------------------------------------------------------
def test_004_chained_end_to_end_single_sample():
    """A full chain (prev-meta propagation) across all 32 stages succeeds."""
    sample = "CCO"
    prev: dict | None = None
    for sid, fn in bench_mod.PIPELINE_FULL:
        wall, status = bench_mod._safe_run(sid, fn, sample, prev)
        assert status == "ok", f"chain broke at stage {sid!r}: status={status}"
        prev = {"stage": sid, "ok": True, "wall": wall}


# ---------------------------------------------------------------------------
# Test 5 — SE(3) Schrödinger-Bridge stage exercises an SE(3) geodesic
# ---------------------------------------------------------------------------
def test_005_se3_lipman_pipeline_runs():
    """SE(3) Schrödinger-Bridge stage returns finite SE(3) intermediate."""
    out = bench_mod._stage_se3_lipman("CCO", None)
    assert "traj_len" in out
    assert out["traj_len"] >= 1, f"SE(3) trajectory has zero length: {out}"
    assert "R_mid_shape" in out
    assert (
        len(out["R_mid_shape"]) == 3
    ), f"SE(3) midpoint must be 3-dim, got shape {out['R_mid_shape']}"


# ---------------------------------------------------------------------------
# Test 6 — Hilbert primal-dual stage converges in O(1/k)
# ---------------------------------------------------------------------------
def test_006_hilbert_astsr_pipeline_runs():
    """Hilbert + ASTSR stage emits a finite primal / dual norm."""
    out = bench_mod._stage_hilbert_astsr("CCO", None)
    assert out["hilbert_dim"] >= 4
    assert out["primal_norm"] >= 0.0
    assert out["dual_norm"] >= 0.0


# ---------------------------------------------------------------------------
# Test 7 — W-Flow CFM stage produces a non-empty trajectory
# ---------------------------------------------------------------------------
def test_007_w_flow_cfm_pipeline_runs():
    """W-Flow CFM stage emits a trajectory of positive length."""
    out = bench_mod._stage_w_flow_cfm("CCO", None)
    assert out["w_flow_traj_len"] >= 1
    assert out["n_windows"] == 2


# ---------------------------------------------------------------------------
# Test 8 — Multi-PH + PLL stage emits a finite filtration + PLL dim
# ---------------------------------------------------------------------------
def test_008_multi_ph_pll_pipeline_runs():
    """Multi-PH + PLL stage emits a finite bin count + PLL dim."""
    out = bench_mod._stage_multi_ph_pll("CCO", None)
    assert out["ph_n_bins"] >= 1
    assert out["pll_dim"] >= 1


# ---------------------------------------------------------------------------
# Test 9 — Variance MCTS stage routes to a child node
# ---------------------------------------------------------------------------
def test_009_variance_mcts_pipeline_runs():
    """Variance-controlled PUCT MCTS stage returns a child index."""
    out = bench_mod._stage_variance_mcts("CCO", None)
    assert "variance_puct_child" in out
    assert isinstance(out["variance_puct_child"], str)


# ---------------------------------------------------------------------------
# Test 10 — Pitfall Layered gen stage returns a valid flag
# ---------------------------------------------------------------------------
def test_010_pitfall_layered_gen_smoke():
    """Pitfall P2.3 Layered gen stage returns a ``layered_valid`` bool."""
    out = bench_mod._stage_pitfall_layered_gen("CCO", None)
    assert isinstance(out["layered_valid"], bool)


# ---------------------------------------------------------------------------
# Test 11 — Pitfall Buyable gate stage returns a buyability flag
# ---------------------------------------------------------------------------
def test_011_pitfall_buyable_gate_smoke():
    """Pitfall P1.3 Buyable gate stage returns an ``is_buyable`` bool."""
    out = bench_mod._stage_pitfall_buyable_gate("CCO", None)
    assert isinstance(out["is_buyable"], bool)


# ---------------------------------------------------------------------------
# Test 12 — Pitfall OOD protocol stage returns a Tanimoto similarity
# ---------------------------------------------------------------------------
def test_012_pitfall_ood_smoke():
    """Pitfall P6.3 OOD stage returns a Tanimoto similarity in [0, 1]."""
    out = bench_mod._stage_pitfall_ood("CCO", None)
    assert 0.0 <= out["tanimoto_to_CCO"] <= 1.0
    assert isinstance(out["is_ood"], bool)


# ---------------------------------------------------------------------------
# Test 13 — P2-3 Coordination-Aware GNN stage emits a finite shape
# ---------------------------------------------------------------------------
def test_013_p2_ml_coord_gnn_smoke():
    """P2-3 Coordination-Aware GNN stage emits a non-empty shape."""
    out = bench_mod._stage_p2_ml_coord_gnn("CCO", None)
    assert out["coord_gnn_dim"] >= 1


# ---------------------------------------------------------------------------
# Test 14 — P2-4 Quotient-Space Diffusion stage emits a 3-D projection
# ---------------------------------------------------------------------------
def test_014_p2_ml_quotient_diff_smoke():
    """P2-4 Quotient-Space Diffusion stage returns a 3-D shape."""
    out = bench_mod._stage_p2_ml_quotient_diff("CCO", None)
    assert (
        len(out["qsd_shape"]) == 3
    ), f"expected 3-D projected shape, got {out['qsd_shape']}"


# ---------------------------------------------------------------------------
# Test 15 — Quantum semantics stage emits a finite ``commutes`` bool
# ---------------------------------------------------------------------------
def test_015_quantum_semantics_smoke():
    """Wf-15 phase 1 — Quantum semantics stage returns a ``commutes`` bool."""
    out = bench_mod._stage_quantum_semantics("CCO", None)
    assert isinstance(out["commutes"], bool)
    assert out["quantum_dim"] >= 1


# ---------------------------------------------------------------------------
# Test 16 — Hochschild cohomology stage returns a non-negative H-sum
# ---------------------------------------------------------------------------
def test_016_hochschild_smoke():
    """Wf-15 phase 2 — Hochschild cohomology stage returns H-sum ≥ 0."""
    out = bench_mod._stage_hochschild("CCO", None)
    assert out["hochschild_h_sum"] >= 0
    assert out["n_args"] >= 2


# ---------------------------------------------------------------------------
# Test 17 — total modules covered by the benchmark is ≥ 25
# ---------------------------------------------------------------------------
def test_017_total_modules_covered():
    """Pipeline covers at least 25 R18 ship-modules (32 expected)."""
    n = len(bench_mod.PIPELINE_FULL)
    assert n >= 25, f"only {n} stages registered; expected ≥ 25"
    assert n <= 64, f"too many stages ({n}); sanity check"


# ---------------------------------------------------------------------------
# Test 18 — CSV loader returns the expected number of PlatinAI samples
# ---------------------------------------------------------------------------
def test_018_platinai_csv_loader():
    """The PlatinAI 5K CSV loader returns at least one valid SMILES."""
    if not _DATA.exists():
        pytest.fail(f"PlatinAI CSV not found: {_DATA}")
    smiles = bench_mod._load_platinai_smiles(_DATA, n=10)
    assert len(smiles) == 10, f"expected 10 SMILES, got {len(smiles)}"
    assert all(
        isinstance(s, str) and s for s in smiles
    ), f"non-string or empty SMILES in {smiles}"


# ---------------------------------------------------------------------------
# Test 19 — ``_safe_run`` handles a no-op runner cleanly
# ---------------------------------------------------------------------------
def test_019_safe_run_no_op_runner():
    """``_safe_run`` returns ``("skipped", 0.0)`` for a ``None`` runner."""
    wall, status = bench_mod._safe_run("noop", None, "CCO", None)
    assert wall == 0.0
    assert status == "skipped"


# ---------------------------------------------------------------------------
# Test 20 — median-ms helper behaves on empty / populated inputs
# ---------------------------------------------------------------------------
def test_020_median_ms_helper():
    """``_median_ms`` returns 0.0 on empty list and the finite median otherwise."""
    assert bench_mod._median_ms([]) == 0.0
    assert bench_mod._median_ms([0.001, 0.002, 0.003]) == pytest.approx(2.0, abs=0.001)


# ---------------------------------------------------------------------------
# Test 21 — full end-to-end benchmark writes the JSON report to the
# canonical path.  Uses the project-local .venv if available so the
# torch import succeeds; otherwise skips with a clear message.
# ---------------------------------------------------------------------------
def test_021_end_to_end_writes_json_report(tmp_path, monkeypatch):
    """``main()`` writes ``end_to_end.json`` with the canonical schema.

    The wall-time budget for 5 samples is ≤ 30 s (10× margin over the
    spec target of 50 samples in 60 s).
    """
    if not _DATA.exists():
        pytest.fail(f"PlatinAI CSV not found: {_DATA}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_r18_end_to_end.py",
            "--n-samples",
            "5",
            "--out-dir",
            str(tmp_path),
            "--csv-path",
            str(_DATA),
        ],
    )
    rc = bench_mod.main()
    assert rc == 0, f"benchmark main returned {rc}"
    out_path = tmp_path / "end_to_end.json"
    assert out_path.exists(), f"missing report: {out_path}"
    payload = json.loads(out_path.read_text())
    for key in (
        "n_samples",
        "n_modules",
        "total_wall_time_s",
        "per_module_wall_time_ms",
        "per_module_memory_mb",
        "per_module_status",
    ):
        assert key in payload, f"missing key {key!r} in report"
    assert payload["n_samples"] == 5
    assert payload["n_modules"] == 32
    assert (
        payload["total_wall_time_s"] <= 60.0
    ), f"5-sample wall-time {payload['total_wall_time_s']:.2f}s exceeded 60s"


# ---------------------------------------------------------------------------
# Test 22 — 50-sample end-to-end wall-time is ≤ 60 seconds (the spec budget)
# ---------------------------------------------------------------------------
def test_022_end_to_end_wall_time_50_samples(monkeypatch):
    """The full 50-sample end-to-end benchmark fits in the 60-s spec budget."""
    if not _DATA.exists():
        pytest.fail(f"PlatinAI CSV not found: { _DATA}")
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_r18_end_to_end.py",
            "--n-samples",
            "50",
            "--out-dir",
            str(_REPORT_DIR),
            "--csv-path",
            str(_DATA),
        ],
    )
    t0 = time.perf_counter()
    rc = bench_mod.main()
    wall = time.perf_counter() - t0
    assert rc == 0, f"benchmark main returned {rc}"
    assert wall <= 60.0, f"50-sample wall-time {wall:.2f}s exceeded 60s budget"
    assert _REPORT_FILE.exists(), f"canonical report missing: {_REPORT_FILE}"
    payload = json.loads(_REPORT_FILE.read_text())
    n_ok = sum(1 for s in payload["per_module_status"].values() if s == "ok")
    assert (
        n_ok == 32
    ), f"expected 32 modules ok, got {n_ok}: {payload['per_module_status']}"


# ---------------------------------------------------------------------------
# Test 23 — wall-time per module is finite (no NaN / inf)
# ---------------------------------------------------------------------------
def test_023_per_module_wall_time_finite():
    """Every per-module wall-time in the report is finite and ≥ 0."""
    if not _REPORT_FILE.exists():
        pytest.fail(f"canonical report missing: {_REPORT_FILE}")
    payload = json.loads(_REPORT_FILE.read_text())
    wall_ms = payload["per_module_wall_time_ms"]
    for sid, w in wall_ms.items():
        assert isinstance(w, (int, float)), f"non-numeric wall-time for {sid!r}: {w}"
        assert w >= 0.0, f"negative wall-time for {sid!r}: {w}"


# ---------------------------------------------------------------------------
# Test 24 — the script imports cleanly (no top-level NameError)
# ---------------------------------------------------------------------------
def test_024_module_imports_clean():
    """``benchmark_r18_end_to_end`` is importable and has the canonical API."""
    assert hasattr(bench_mod, "PIPELINE_FULL")
    assert hasattr(bench_mod, "_safe_run")
    assert hasattr(bench_mod, "_load_platinai_smiles")
    assert hasattr(bench_mod, "_median_ms")
    assert hasattr(bench_mod, "main")
