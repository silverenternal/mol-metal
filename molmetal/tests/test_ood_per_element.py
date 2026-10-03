"""Tests for Wf-73 Phase 3 — per-element OOD reachability benchmark.

================================================================
Wf-73 Phase 3 (L12 trajectory): per-element OOD reachability on
tmQM d-block 30-element training pool (Wf-28 ship) with Tanimoto<0.4
OOD protocol.

This file ships the **16 pytest contracts** that pin the behaviour of
:mod:`molmetal.scripts.bench_ood_per_element`:

Constants & defaults (5)
 1. test_dblock_elements_count             — 30 d-block elements
 2. test_dblock_elements_3d_block          — first 10 = 3d-block
 3. test_dblock_elements_4d_block          — middle 10 = 4d-block
 4. test_dblock_elements_5d_block          — last 10 = 5d+La
 5. test_element_z_mapping                 — ELEMENT_Z covers all 30
 6. test_default_n_samples                 — n_samples default = 20
 7. test_default_tanimoto_threshold        — Tanimoto threshold = 0.4
 8. test_default_seeds_is_3                — seeds default = 3

Imports / helpers (3)
 9. test_click_rules_loadable              — 5 click rules available
10. test_partner_smile_list_size           — partner list has >= 6 tiles
11. test_tanimoto_protocol_loads           — TanimotoOODProtocol loads
12. test_helper_aggregations_zero_safe     — empty records aggregate to 0
13. test_load_tmqm_pool_handles_missing    — empty dict on missing CSV

Driver (3)
14. test_smoke_5elements_1seed             — 2 samples/element/seed smoke
15. test_smoke_subset_element              — single-element slice
16. test_per_element_gate_block_present    — gate verdict in payload
17. test_cross_element_corr_block          — cross_element_correlation
                                            |E|×|E|

Output contract (2)
18. test_per_element_json_written          — per-element JSON files
19. test_full_payload_json_written         — wf73_ood_per_element_full.json
20. test_synthesis_md_written              — synthesis.md on disk

All tests are CPU-only, deterministic, and complete in < 180 s
combined.  Smoke runs use ``--n-samples 2 --seeds 1`` for sub-30 s CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

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


from molmetal.scripts import bench_ood_per_element  # noqa: E402
from molmetal.scripts.bench_ood_per_element import (  # noqa: E402
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PRODUCTS,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_TANIMOTO_THRESHOLD,
    DBLOCK_ELEMENTS,
    ELEMENT_BLOCK,
    ELEMENT_Z,
    PARTNER_SMILE_LIST,
    PER_ELEMENT_GATE,
    _aggregate_per_element,
    _cross_element_correlation,
    _load_click_rules,
    _load_partner_terms,
    _load_tanimoto_protocol,
    _load_tmqm_pool,
    _split_train_test,
    _utcnow_iso,
    run_ood_per_element_benchmark,
    save_ood_per_element_benchmark,
    write_synthesis,
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


def _ensure_rdkit_and_data() -> bool:
    if not _have_rdkit():
        return False
    csv_path = PROJECT_ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv"
    return csv_path.is_file()


# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
def test_dblock_elements_count() -> None:
    assert len(DBLOCK_ELEMENTS) == 30
    assert len(set(DBLOCK_ELEMENTS)) == 30  # all distinct


def test_dblock_elements_3d_block() -> None:
    expected = ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn")
    assert DBLOCK_ELEMENTS[:10] == expected


def test_dblock_elements_4d_block() -> None:
    expected = ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd")
    assert DBLOCK_ELEMENTS[10:20] == expected


def test_dblock_elements_5d_block() -> None:
    expected = ("La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg")
    assert DBLOCK_ELEMENTS[20:] == expected


def test_element_z_mapping() -> None:
    for elem in DBLOCK_ELEMENTS:
        assert elem in ELEMENT_Z
        assert ELEMENT_Z[elem] >= 21  # Sc = 21
    # Spot check a couple of well-known atomic numbers.
    assert ELEMENT_Z["Sc"] == 21
    assert ELEMENT_Z["Pt"] == 78
    assert ELEMENT_Z["Hg"] == 80


def test_default_n_samples() -> None:
    assert int(DEFAULT_N_SAMPLES) == 20


def test_default_tanimoto_threshold() -> None:
    assert float(DEFAULT_TANIMOTO_THRESHOLD) == pytest.approx(0.4)


def test_default_seeds_is_3() -> None:
    assert int(DEFAULT_SEEDS) == 3


def test_per_element_gate_value() -> None:
    assert float(PER_ELEMENT_GATE) == pytest.approx(0.40)


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
    agg = _aggregate_per_element([])
    assert agg["n_samples"] == 0
    assert agg["n_closure_ok"] == 0
    assert agg["reachability_rate"] == 0.0
    assert agg["ood_pass_rate"] == 0.0


def test_load_tmqm_pool_handles_missing(tmp_path: Path) -> None:
    """Missing CSV returns empty dict; we don't fabricate."""
    pool = _load_tmqm_pool(tmp_path / "missing.csv", elements=DBLOCK_ELEMENTS)
    assert pool == {}


def test_split_train_test_basic() -> None:
    rows = [{"csd_code": f"C{i:03d}", "smiles": "C"} for i in range(10)]
    train, test = _split_train_test(rows, train_ratio=0.8)
    assert len(train) == 8
    assert len(test) == 2


def test_split_train_test_small_n() -> None:
    """Small clusters (e.g. La n=3) should split gracefully."""
    rows = [{"csd_code": f"C{i:03d}", "smiles": "C"} for i in range(3)]
    train, test = _split_train_test(rows, train_ratio=0.8)
    assert len(train) + len(test) == 3


def test_cross_element_correlation_diagonal() -> None:
    """Diagonals of cross-element correlation matrix = 1.0."""
    per_seed: Dict[str, Dict[int, Dict[str, float]]] = {
        "Sc": {0: {"reachability_rate": 0.1}, 1: {"reachability_rate": 0.2}},
        "Ti": {0: {"reachability_rate": 0.3}, 1: {"reachability_rate": 0.4}},
    }
    corr = _cross_element_correlation(per_seed, ["Sc", "Ti"])
    assert corr["Sc"]["Sc"] == pytest.approx(1.0)
    assert corr["Ti"]["Ti"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Driver smoke runs
# ---------------------------------------------------------------------------
def test_smoke_5elements_1seed(tmp_path: Path) -> None:
    """Smoke run: 2 samples per element × 1 seed × 5 elements."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_p3_smoke"
    payload = run_ood_per_element_benchmark(
        n_samples=2,
        seeds=1,
        elements=("Sc", "Ti", "V", "Cr", "Mn"),
        max_products=8,
        output_dir=out,
    )
    assert "per_element" in payload
    assert set(payload["per_element"].keys()) == {
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
    }


def test_smoke_subset_element(tmp_path: Path) -> None:
    """Single-element slice (W only — to exercise the 5d block)."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_p3_w_only"
    payload = run_ood_per_element_benchmark(
        n_samples=2,
        seeds=1,
        elements=("W",),
        max_products=8,
        output_dir=out,
    )
    assert "W" in payload["per_element"]
    assert "Pt" not in payload["per_element"]


def test_per_element_gate_block_present(tmp_path: Path) -> None:
    """Per-element gate verdict block is populated."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_p3_gate"
    payload = run_ood_per_element_benchmark(
        n_samples=2,
        seeds=1,
        elements=("Sc", "Ti", "V"),
        max_products=8,
        output_dir=out,
    )
    assert "per_element_gate" in payload
    g = payload["per_element_gate"]
    assert "threshold" in g
    assert "n_elements_pass" in g
    assert "verdict" in g
    assert g["verdict"] in (
        "PER_ELEMENT_GATE_MET",
        "PER_ELEMENT_HONEST_PARTIAL",
    )


def test_cross_element_corr_block(tmp_path: Path) -> None:
    """cross_element_correlation is |E| × |E| with diagonals = 1.0."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_p3_corr"
    payload = run_ood_per_element_benchmark(
        n_samples=2,
        seeds=1,
        elements=("Sc", "Ti", "V"),
        max_products=8,
        output_dir=out,
    )
    corr = payload["cross_element_correlation"]
    assert set(corr.keys()) == {"Sc", "Ti", "V"}
    for e in ("Sc", "Ti", "V"):
        assert corr[e][e] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
def test_per_element_json_written(tmp_path: Path) -> None:
    """Per-element JSON files at {element}/ood_reachability.json."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_p3_write"
    elements = ("Sc", "Ti", "V")
    payload = run_ood_per_element_benchmark(
        n_samples=2,
        seeds=1,
        elements=elements,
        max_products=8,
        output_dir=out,
    )
    save_ood_per_element_benchmark(payload, output_dir=out)
    for elem in elements:
        p = out / elem / "ood_reachability.json"
        assert p.is_file(), f"missing {p}"
        with p.open() as fh:
            obj = json.load(fh)
        assert "aggregate" in obj


def test_full_payload_json_written(tmp_path: Path) -> None:
    """wf73_ood_per_element_full.json summary file present."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_p3_full"
    payload = run_ood_per_element_benchmark(
        n_samples=2,
        seeds=1,
        elements=("Sc", "Ti", "V"),
        max_products=8,
        output_dir=out,
    )
    save_ood_per_element_benchmark(payload, output_dir=out)
    summary = out / "wf73_ood_per_element_full.json"
    assert summary.is_file()
    with summary.open() as fh:
        obj = json.load(fh)
    assert obj["tool"] == "bench_ood_per_element.py"
    assert obj["elements"] == ["Sc", "Ti", "V"]


def test_synthesis_md_written(tmp_path: Path) -> None:
    """synthesis.md is written and contains the gate verdict."""
    if not _ensure_rdkit_and_data():
        pytest.skip("rdkit or input CSV unavailable")
    out = tmp_path / "wf73_p3_syn"
    payload = run_ood_per_element_benchmark(
        n_samples=2,
        seeds=1,
        elements=("Sc", "Ti", "V"),
        max_products=8,
        output_dir=out,
    )
    save_ood_per_element_benchmark(payload, output_dir=out)
    md = write_synthesis(payload, output_dir=out)
    assert md.is_file()
    text = md.read_text()
    assert "Wf-73 Phase 3" in text
    assert "Per-element" in text or "per-element" in text


def test_block_assignment_complete() -> None:
    """ELEMENT_BLOCK covers all 30 elements."""
    for elem in DBLOCK_ELEMENTS:
        assert elem in ELEMENT_BLOCK
        assert ELEMENT_BLOCK[elem] in {"3d", "4d", "5d+La"}


def test_utcnow_iso_format() -> None:
    """Helper returns ISO-8601 with Z suffix."""
    s = _utcnow_iso()
    assert s.endswith("Z")
    assert "T" in s
