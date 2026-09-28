"""Tests for the closure theorem d-block R20 benchmark (Wf-67 Phase 2).

================================================================
Wf-67 Phase 2: 16 pytest contracts pinning
:mod:`molmetal.scripts.bench_closure_dblock_r20` to the spec:

 1. test_module_imports                  — script imports cleanly
 2. test_dblock_metals_30                — DBLOCK_METALS list has 30 entries
 3. test_default_n_samples_is_1000       — default n_samples = 1000
 4. test_default_n_metals_is_30          — default n_metals = 30
 5. test_default_max_depth_is_1          — default max_depth = 1
 6. test_default_output_dir              — default output dir = wf67_p4_2
 7. test_gate_target_is_0_95             — GATE_TARGET_REACHABILITY = 0.95
 8. test_dative_preprocessor_basic       — "<-" → "-"
 9. test_dative_preprocessor_idempotent  — normalize twice == normalize once
10. test_dative_preprocessor_preserves_arrows  — also handles "->"
11. test_dative_preprocessor_empty       — empty SMILES returns empty
12. test_aggregate_keys                  — _aggregate_per_metal keys
13. test_aggregate_gate_pass_false       — gate_pass=False at low rate
14. test_smoke_run_3_metals              — 5 samples × 3 metals smoke OK
15. test_per_metal_files_written         — per-metal reachability.json
16. test_synthesis_md_written            — synthesis.md exists with TL;DR

All tests CPU-only, deterministic, complete < 120 s combined.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "molmetal" / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from molmetal.scripts import bench_closure_dblock_r20  # noqa: E402
from molmetal.scripts.bench_closure_dblock_r20 import (  # noqa: E402
    DBLOCK_METALS,
    DEFAULT_DATASET,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PRODUCTS,
    DEFAULT_N_METALS,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_TMQM_CSV,
    GATE_TARGET_REACHABILITY,
    PARTNER_SMILE_LIST,
    _aggregate_per_metal,
    _benchmark_one,
    _load_click_rules,
    _load_partner_terms,
    _utcnow_iso,
    normalize_dative_to_covalent,
    run_benchmark,
    save_per_metal,
    write_synthesis,
)


# ---------------------------------------------------------------------------
# 1. test_module_imports
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """Script imports without error."""
    assert bench_closure_dblock_r20 is not None
    assert hasattr(bench_closure_dblock_r20, "run_benchmark")
    assert hasattr(bench_closure_dblock_r20, "normalize_dative_to_covalent")


# ---------------------------------------------------------------------------
# 2. test_dblock_metals_30
# ---------------------------------------------------------------------------
def test_dblock_metals_30() -> None:
    """DBLOCK_METALS has 30 entries (3d + 4d + 5d + La)."""
    assert len(DBLOCK_METALS) == 30
    # Spot-check key elements
    for m in ("Pt", "Ru", "Ir", "Rh", "Pd", "Ag", "Au", "Cu", "Zn", "Hg"):
        assert m in DBLOCK_METALS


# ---------------------------------------------------------------------------
# 3. test_default_n_samples_is_1000
# ---------------------------------------------------------------------------
def test_default_n_samples_is_1000() -> None:
    """Default n_samples = 1000 (vs Wf-37 P2 = 100)."""
    assert DEFAULT_N_SAMPLES == 1000


# ---------------------------------------------------------------------------
# 4. test_default_n_metals_is_30
# ---------------------------------------------------------------------------
def test_default_n_metals_is_30() -> None:
    """Default n_metals = 30 (all d-block)."""
    assert DEFAULT_N_METALS == 30


# ---------------------------------------------------------------------------
# 5. test_default_max_depth_is_1
# ---------------------------------------------------------------------------
def test_default_max_depth_is_1() -> None:
    """Default max_depth = 1 (single click-rule application)."""
    assert DEFAULT_MAX_DEPTH == 1


# ---------------------------------------------------------------------------
# 6. test_default_output_dir
# ---------------------------------------------------------------------------
def test_default_output_dir() -> None:
    """Default output dir is wf67_p4_2_closure_r20."""
    assert DEFAULT_OUTPUT_DIR.name == "wf67_p4_2_closure_r20"
    assert "reports" in str(DEFAULT_OUTPUT_DIR)


# ---------------------------------------------------------------------------
# 7. test_gate_target_is_0_95
# ---------------------------------------------------------------------------
def test_gate_target_is_0_95() -> None:
    """Gate target reachability is 0.95 (Wf-67 P2 spec)."""
    assert GATE_TARGET_REACHABILITY == 0.95


# ---------------------------------------------------------------------------
# 8. test_dative_preprocessor_basic
# ---------------------------------------------------------------------------
def test_dative_preprocessor_basic() -> None:
    """Dative-bond preprocessor: '<-' → '-'."""
    smi = "[Pt+2]<-(N)<-(N)<-(N)<-(N)"
    out = normalize_dative_to_covalent(smi)
    assert "<-" not in out
    assert out == "[Pt+2]-(N)-(N)-(N)-(N)"


# ---------------------------------------------------------------------------
# 9. test_dative_preprocessor_idempotent
# ---------------------------------------------------------------------------
def test_dative_preprocessor_idempotent() -> None:
    """normalize( normalize(s) ) == normalize( s )."""
    smi = "CC(C)c1cc(C(C)C)c(C2=CC=CC3=CN(c4c(C(C)C)cccc4C(C)C)[C](->[Ag+]<-[Cl-])N32)c(C(C)C)c1"
    once = normalize_dative_to_covalent(smi)
    twice = normalize_dative_to_covalent(once)
    assert once == twice


# ---------------------------------------------------------------------------
# 10. test_dative_preprocessor_preserves_arrows
# ---------------------------------------------------------------------------
def test_dative_preprocessor_preserves_arrows() -> None:
    """Preprocessor converts '<-' only; '->' is preserved.

    The preprocessor only converts ``<-`` (the tmQM convention for
    ligand-to-metal dative bonds).  ``->`` (reverse arrow) is left
    alone because converting it breaks chemistry for the donor atom.
    """
    smi = "N->[Pt+2]<-(N)"
    out = normalize_dative_to_covalent(smi)
    assert "<-" not in out
    assert "->" in out  # preserved
    assert out == "N->[Pt+2]-(N)"


# ---------------------------------------------------------------------------
# 11. test_dative_preprocessor_empty
# ---------------------------------------------------------------------------
def test_dative_preprocessor_empty() -> None:
    """Empty SMILES returns empty."""
    assert normalize_dative_to_covalent("") == ""


# ---------------------------------------------------------------------------
# 12. test_aggregate_keys
# ---------------------------------------------------------------------------
def test_aggregate_keys() -> None:
    """_aggregate_per_metal returns expected keys."""
    recs = [
        {"status": "ok", "closure_ok": True, "wall_time_s": 0.1,
         "memory_mb": 0.5, "dative_converted": True},
        {"status": "ok", "closure_ok": False, "wall_time_s": 0.2,
         "memory_mb": 0.6, "dative_converted": False},
    ]
    agg = _aggregate_per_metal(recs)
    for key in (
        "n_samples", "n_closure_ok", "closure_reachability_rate",
        "n_parse_error", "n_dative_converted",
        "mean_wall_time_s", "median_wall_time_s", "std_wall_time_s",
        "total_wall_time_s", "peak_memory_mb",
        "gate_target", "gate_pass",
    ):
        assert key in agg
    assert agg["n_samples"] == 2
    assert agg["n_closure_ok"] == 1
    assert agg["closure_reachability_rate"] == 0.5
    assert agg["n_dative_converted"] == 1
    assert agg["gate_pass"] is False  # 0.5 < 0.95


# ---------------------------------------------------------------------------
# 13. test_aggregate_gate_pass_false
# ---------------------------------------------------------------------------
def test_aggregate_gate_pass_false() -> None:
    """Aggregate below gate target → gate_pass=False."""
    recs = [
        {"status": "ok", "closure_ok": False, "wall_time_s": 0.1,
         "memory_mb": 0.5, "dative_converted": False}
        for _ in range(10)
    ]
    agg = _aggregate_per_metal(recs)
    assert agg["closure_reachability_rate"] == 0.0
    assert agg["gate_pass"] is False


# ---------------------------------------------------------------------------
# 14. test_smoke_run_3_metals
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_payload() -> Dict[str, Any]:
    """5 samples × 3 metals smoke payload."""
    return run_benchmark(
        n_metals=3,
        n_samples=5,
        metals=["Pt", "Ru", "Pd"],
        max_depth=1,
        max_products=64,
    )


def test_smoke_run_3_metals(smoke_payload: Dict[str, Any]) -> None:
    """3-metal × 5-sample smoke run returns expected schema."""
    assert "aggregate" in smoke_payload
    assert "per_metal" in smoke_payload
    assert set(smoke_payload["per_metal"].keys()) == {"Pt", "Ru", "Pd"}
    for metal in ("Pt", "Ru", "Pd"):
        agg = smoke_payload["per_metal"][metal]
        assert "closure_reachability_rate" in agg
        assert "gate_pass" in agg
        assert agg["gate_target"] == 0.95
    overall = smoke_payload["aggregate"]
    assert overall["n_samples"] >= 1
    assert "timestamp_utc" in smoke_payload


# ---------------------------------------------------------------------------
# 15. test_per_metal_files_written
# ---------------------------------------------------------------------------
def test_per_metal_files_written(tmp_path: Path) -> None:
    """Per-metal reachability.json files are written."""
    payload = run_benchmark(
        n_metals=2, n_samples=3, metals=["Pt", "Ru"], output_dir=tmp_path,
    )
    paths = save_per_metal(payload, output_dir=tmp_path)
    assert (tmp_path / "Pt" / "reachability.json").is_file()
    assert (tmp_path / "Ru" / "reachability.json").is_file()
    assert (tmp_path / "reachability_overall.json").is_file()
    # Validate JSON content
    pt_data = json.loads((tmp_path / "Pt" / "reachability.json").read_text())
    assert pt_data["metal"] == "Pt"
    assert "summary" in pt_data
    assert "n_samples" in pt_data["summary"]


# ---------------------------------------------------------------------------
# 16. test_synthesis_md_written
# ---------------------------------------------------------------------------
def test_synthesis_md_written(tmp_path: Path) -> None:
    """synthesis.md is written with TL;DR + per-metal breakdown."""
    payload = run_benchmark(
        n_metals=3, n_samples=3, metals=["Pt", "Ru", "Pd"], output_dir=tmp_path,
    )
    save_per_metal(payload, output_dir=tmp_path)
    md_path = write_synthesis(payload, output_dir=tmp_path)
    assert md_path.is_file()
    content = md_path.read_text()
    assert "Wf-67 Phase 2" in content
    assert "Overall reachability" in content
    assert "Per-metal breakdown" in content
    assert "Honest framing" in content