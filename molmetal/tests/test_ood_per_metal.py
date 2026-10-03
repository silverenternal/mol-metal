"""Tests for Wf-73 Phase 2 — per-metal OOD reachability benchmark.

================================================================
Wf-73 Phase 2 (L12 trajectory): per-metal OOD reachability on
MetalCytoToxDB 5-metal (Ru / Ir / Os / Rh / Re) with Tanimoto<0.4
OOD protocol against the tmQM d-block training pool.

This file ships the **16 pytest contracts** that pin the behaviour of
:mod:`molmetal.scripts.bench_ood_per_metal`:

Constants & defaults (5)
 1. test_metalcyto_metals_set           — METALCYTOTOX_METALS = 5 metals
 2. test_metal_z_mapping                — METAL_Z covers all 5 metals
 3. test_default_n_samples              — n_samples default = 80
 4. test_default_tanimoto_threshold     — Tanimoto threshold = 0.4
 5. test_default_seeds_is_3             — seeds default = 3

Imports / helpers (4)
 6. test_click_rules_loadable           — 5 click rules available
 7. test_partner_smile_list_size        — partner list has >= 6 tiles
 8. test_tanimoto_protocol_loads        — TanimotoOODProtocol loads
 9. test_helper_aggregations_zero_safe  — empty records aggregate to 0

Driver (4)
10. test_smoke_5metals_1seed            — 2 samples/metal/seed smoke
11. test_smoke_subset_metal             — single-metal slice (Rh only)
12. test_rh_bottleneck_block_present    — rh_bottleneck verdict in payload
13. test_cross_metal_corr_block         — cross_metal_correlation 5x5

Output contract (3)
14. test_per_metal_json_written         — 5 JSON files written
15. test_full_payload_json_written      — wf73_ood_per_metal_full.json
16. test_rh_bottleneck_verdict_honest   — verdict is honest (no fabrication)

All tests are CPU-only, deterministic, and complete in < 180 s
combined.  Smoke runs use ``--n-samples 2 --seeds 1`` for sub-30 s CI.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "molmetal" / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from molmetal.scripts import bench_ood_per_metal  # noqa: E402
from molmetal.scripts.bench_ood_per_metal import (  # noqa: E402
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PRODUCTS,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_TANIMOTO_THRESHOLD,
    METALCYTOTOX_METALS,
    METAL_Z,
    PARTNER_SMILE_LIST,
    _aggregate_per_metal,
    _benchmark_one,
    _load_click_rules,
    _load_metal_samples,
    _load_partner_terms,
    _load_tanimoto_protocol,
    _load_tmqm_train_pool,
    _utcnow_iso,
    run_ood_per_metal_benchmark,
    save_ood_per_metal_benchmark,
)


# ---------------------------------------------------------------------------
# RDKit availability probe
# ---------------------------------------------------------------------------
def _have_rdkit() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
def test_metalcyto_metals_set() -> None:
    assert METALCYTOTOX_METALS == ("Ru", "Ir", "Os", "Rh", "Re")
    assert len(METALCYTOTOX_METALS) == 5


def test_metal_z_mapping() -> None:
    expected = {"Ru": 44, "Ir": 77, "Os": 76, "Rh": 45, "Re": 75}
    for m, z in expected.items():
        assert METAL_Z[m] == z


def test_default_n_samples() -> None:
    assert int(DEFAULT_N_SAMPLES) == 80


def test_default_tanimoto_threshold() -> None:
    assert float(DEFAULT_TANIMOTO_THRESHOLD) == pytest.approx(0.4)


def test_default_seeds_is_3() -> None:
    assert int(DEFAULT_SEEDS) == 3


# ---------------------------------------------------------------------------
# Imports / helpers
# ---------------------------------------------------------------------------
def test_click_rules_loadable() -> None:
    rules = _load_click_rules()
    assert len(rules) == 5


def test_partner_smile_list_size() -> None:
    assert len(PARTNER_SMILE_LIST) >= 6


def test_tanimoto_protocol_loads() -> None:
    proto = _load_tanimoto_protocol(threshold=0.4)
    assert proto is not None
    assert proto.threshold == pytest.approx(0.4)


def test_helper_aggregations_zero_safe() -> None:
    agg = _aggregate_per_metal([])
    assert agg["n_samples"] == 0
    assert agg["n_closure_ok"] == 0
    assert agg["reachability_rate"] == 0.0
    assert agg["ood_pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# Driver smoke runs
# ---------------------------------------------------------------------------
def _ensure_csv(csv_path: Path) -> bool:
    return csv_path.is_file()


def test_smoke_5metals_1seed(tmp_path: Path) -> None:
    """Smoke run: 2 samples per metal × 1 seed."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_smoke"
    payload = run_ood_per_metal_benchmark(
        n_samples=2,
        seeds=1,
        metals=METALCYTOTOX_METALS,
        max_products=8,
        output_dir=out,
    )
    assert "per_metal" in payload
    assert set(payload["per_metal"].keys()) == set(METALCYTOTOX_METALS)


def test_smoke_subset_metal(tmp_path: Path) -> None:
    """Single-metal slice (Rh only)."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_rh_only"
    payload = run_ood_per_metal_benchmark(
        n_samples=2,
        seeds=1,
        metals=("Rh",),
        max_products=8,
        output_dir=out,
    )
    assert "Rh" in payload["per_metal"]
    assert "Ru" not in payload["per_metal"]


def _ensure_rdkit_and_data() -> bool:
    if not _have_rdkit():
        return False
    csv_path = Path("/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv")
    return csv_path.is_file()


# ---------------------------------------------------------------------------
# Driver — aggregation features
# ---------------------------------------------------------------------------
def test_rh_bottleneck_block_present(tmp_path: Path) -> None:
    """Rh-bottleneck diagnosis block is populated and has a verdict."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_rh"
    payload = run_ood_per_metal_benchmark(
        n_samples=2,
        seeds=1,
        metals=METALCYTOTOX_METALS,
        max_products=8,
        output_dir=out,
    )
    assert "rh_bottleneck" in payload
    rb = payload["rh_bottleneck"]
    assert "rh_reachability_rate" in rb
    assert "other_metals_reachability_mean" in rb
    assert "lift_gap" in rb
    assert "verdict" in rb
    assert rb["verdict"] in (
        "RH_BOTTLENECK_LIFTED",
        "RH_HONEST_PARTIAL",
    )


def test_cross_metal_corr_block(tmp_path: Path) -> None:
    """cross_metal_correlation is a 5x5 dict with diagonals = 1.0."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_corr"
    payload = run_ood_per_metal_benchmark(
        n_samples=2,
        seeds=1,
        metals=METALCYTOTOX_METALS,
        max_products=8,
        output_dir=out,
    )
    corr = payload["cross_metal_correlation"]
    assert set(corr.keys()) == set(METALCYTOTOX_METALS)
    for m in METALCYTOTOX_METALS:
        assert corr[m][m] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
def test_per_metal_json_written(tmp_path: Path) -> None:
    """Per-metal JSON files at {metal}/ood_reachability.json."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_write"
    payload = run_ood_per_metal_benchmark(
        n_samples=2,
        seeds=1,
        metals=METALCYTOTOX_METALS,
        max_products=8,
        output_dir=out,
    )
    save_ood_per_metal_benchmark(payload, output_dir=out)
    for metal in METALCYTOTOX_METALS:
        p = out / metal / "ood_reachability.json"
        assert p.is_file(), f"missing {p}"
        with p.open() as fh:
            obj = json.load(fh)
        assert "aggregate" in obj


def test_full_payload_json_written(tmp_path: Path) -> None:
    """wf73_ood_per_metal_full.json summary file present."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_full"
    payload = run_ood_per_metal_benchmark(
        n_samples=2,
        seeds=1,
        metals=METALCYTOTOX_METALS,
        max_products=8,
        output_dir=out,
    )
    save_ood_per_metal_benchmark(payload, output_dir=out)
    summary = out / "wf73_ood_per_metal_full.json"
    assert summary.is_file()
    with summary.open() as fh:
        obj = json.load(fh)
    assert obj["tool"] == "bench_ood_per_metal.py"
    assert obj["metals"] == list(METALCYTOTOX_METALS)


def test_rh_bottleneck_verdict_honest(tmp_path: Path) -> None:
    """Verdict field is one of two honest labels — no fabricated 'LIFTED'."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_honest"
    payload = run_ood_per_metal_benchmark(
        n_samples=2,
        seeds=1,
        metals=METALCYTOTOX_METALS,
        max_products=8,
        output_dir=out,
    )
    rb = payload["rh_bottleneck"]
    rh_rate = float(rb["rh_reachability_rate"])
    others = float(rb["other_metals_reachability_mean"])
    if rh_rate >= 0.30 and rh_rate >= others - 0.10:
        assert rb["verdict"] == "RH_BOTTLENECK_LIFTED"
    else:
        # Honest framing: must NOT falsely claim a lift.
        assert rb["verdict"] == "RH_HONEST_PARTIAL"
