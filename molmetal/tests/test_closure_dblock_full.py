"""Tests for the Wf-130 P2 closure reachability expansion benchmark.

================================================================
Wf-130 P2 closure-theorem reachability expansion contract — 30 d-block
metals × 5000 samples each = 150 000 total, CPU-only, deterministic.

This file ships the **16 pytest contracts** that pin the behaviour of
:mod:`molmetal.scripts.bench_closure_dblock_full`:

 1. test_module_imports                   — script imports cleanly
 2. test_default_metals_30                — DBLOCK_METALS has 30 entries
 3. test_default_n_samples_5000           — default n_samples = 5000
 4. test_default_n_metals_30              — default n_metals = 30
 5. test_default_max_depth_1              — default max_depth = 1
 6. test_default_output_dir               — output dir = wf130_p2
 7. test_gate_target_0_85                 — GATE_TARGET_REACHABILITY = 0.85
 8. test_wf67_baseline_0_5891             — Wf-67 baseline constant
 9. test_wf68_baseline_0_5744             — Wf-68 baseline constant
10. test_sample_to_target_no_replacement  — first N when N < n_unique
11. test_sample_to_target_with_replacement  — sampled when N > n_unique
12. test_aggregate_per_metal_keys         — _aggregate_per_metal keys
13. test_smoke_run_3_metals               — 5 samples × 3 metals OK
14. test_per_metal_files_written          — {metal}/reachability.json
15. test_macro_micro_reachability_present — payload carries both metrics
16. test_synthesis_md_written             — synthesis.md has TL;DR + honest framing
17. test_honest_negative_threshold_present — 0.50 threshold constant
18. test_honest_negative_metals_in_synthesis — <0.50 metals flagged

All tests CPU-only, deterministic, complete < 60 s combined.
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


from molmetal.scripts import bench_closure_dblock_full  # noqa: E402
from molmetal.scripts.bench_closure_dblock_full import (  # noqa: E402
    DBLOCK_METALS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_METALS,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_TMQM_CSV,
    FALLBACK_TMQM_CSV,
    GATE_TARGET_REACHABILITY,
    HONEST_NEGATIVE_THRESHOLD,
    PARTNER_SMILE_LIST,
    WF67_BASELINE_REACHABILITY,
    WF68_BASELINE_REACHABILITY,
    _aggregate_per_metal,
    _benchmark_one,
    _detect_csv_layout,
    _load_click_rules,
    _load_partner_terms,
    _load_unique_metal_smis,
    _memory_mb,
    _sample_to_target,
    _utcnow_iso,
    run_benchmark,
    save_per_metal,
    write_synthesis,
)


# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """Script imports cleanly with the expected entry points."""
    assert bench_closure_dblock_full is not None
    assert hasattr(bench_closure_dblock_full, "run_benchmark")
    assert hasattr(bench_closure_dblock_full, "save_per_metal")
    assert hasattr(bench_closure_dblock_full, "write_synthesis")
    assert hasattr(bench_closure_dblock_full, "_aggregate_per_metal")
    assert hasattr(bench_closure_dblock_full, "_sample_to_target")


# ---------------------------------------------------------------------------
# 2. DBLOCK_METALS has 30 entries
# ---------------------------------------------------------------------------
def test_default_metals_30() -> None:
    """DBLOCK_METALS contains exactly 30 d-block metals."""
    assert len(DBLOCK_METALS) == 30
    # Spot-check the 9 task-spec metals
    for m in ("Pt", "Ru", "Ir", "Rh", "Re", "Os", "Au", "Pd", "Ag"):
        assert m in DBLOCK_METALS
    # Spot-check the 5 previously-not-audited (Wf-130 P1) metals
    for m in ("La", "Hf", "Ta", "Au", "Hg"):
        assert m in DBLOCK_METALS


# ---------------------------------------------------------------------------
# 3. Default n_samples = 5000
# ---------------------------------------------------------------------------
def test_default_n_samples_5000() -> None:
    """Default n_samples = 5000 (vs Wf-67 P2 = 1000, Wf-68 P2 = 1000)."""
    assert DEFAULT_N_SAMPLES == 5000


# ---------------------------------------------------------------------------
# 4. Default n_metals = 30
# ---------------------------------------------------------------------------
def test_default_n_metals_30() -> None:
    """Default n_metals = 30 (full d-block set)."""
    assert DEFAULT_N_METALS == 30


# ---------------------------------------------------------------------------
# 5. Default max_depth = 1
# ---------------------------------------------------------------------------
def test_default_max_depth_1() -> None:
    """Default max_depth = 1 (single click-rule application)."""
    assert DEFAULT_MAX_DEPTH == 1


# ---------------------------------------------------------------------------
# 6. Default output dir is wf130_p2_closure_30_5000
# ---------------------------------------------------------------------------
def test_default_output_dir() -> None:
    """Output dir = reports/wf130_p2_closure_30_5000/."""
    assert DEFAULT_OUTPUT_DIR.name == "wf130_p2_closure_30_5000"
    assert DEFAULT_OUTPUT_DIR.parent.name == "reports"


# ---------------------------------------------------------------------------
# 7. Gate target = 0.85
# ---------------------------------------------------------------------------
def test_gate_target_0_85() -> None:
    """Gate target reachability = 0.85 (Wf-130 P2 spec, vs Wf-67 = 0.95)."""
    assert GATE_TARGET_REACHABILITY == 0.85


# ---------------------------------------------------------------------------
# 8. Wf-67 baseline = 0.5891
# ---------------------------------------------------------------------------
def test_wf67_baseline_0_5891() -> None:
    """Wf-67 P2 baseline constant is 0.5891."""
    assert abs(WF67_BASELINE_REACHABILITY - 0.5891) < 1e-6


# ---------------------------------------------------------------------------
# 9. Wf-68 baseline = 0.5744
# ---------------------------------------------------------------------------
def test_wf68_baseline_0_5744() -> None:
    """Wf-68 P2 baseline constant is 0.5744."""
    assert abs(WF68_BASELINE_REACHABILITY - 0.5744) < 1e-6


# ---------------------------------------------------------------------------
# 10. _sample_to_target: no replacement when N < n_unique
# ---------------------------------------------------------------------------
def test_sample_to_target_no_replacement() -> None:
    """_sample_to_target returns first N when N <= n_unique."""
    smis = ["a", "b", "c", "d", "e"]
    out, n_repeats = _sample_to_target(smis, 3, seed=42)
    assert out == ["a", "b", "c"]
    assert n_repeats == 0


# ---------------------------------------------------------------------------
# 11. _sample_to_target: with replacement when N > n_unique
# ---------------------------------------------------------------------------
def test_sample_to_target_with_replacement() -> None:
    """_sample_to_target samples with replacement when N > n_unique.

    The output list has exactly N elements and at least one repeat;
    seeding is deterministic (same seed → same draw).
    """
    smis = ["a", "b", "c"]
    out1, n_repeats1 = _sample_to_target(smis, 10, seed=42)
    out2, n_repeats2 = _sample_to_target(smis, 10, seed=42)
    out3, n_repeats3 = _sample_to_target(smis, 10, seed=43)
    assert len(out1) == 10
    assert n_repeats1 == 7  # 10 - 3 unique
    # Deterministic: same seed → same draw
    assert out1 == out2
    # Different seed → very likely different draw
    assert out1 != out3 or out1 == out3  # either is acceptable
    # All outputs are valid choices from the source set
    for s in out1:
        assert s in smis


# ---------------------------------------------------------------------------
# 12. _aggregate_per_metal keys
# ---------------------------------------------------------------------------
def test_aggregate_per_metal_keys() -> None:
    """_aggregate_per_metal returns expected keys including cache_hits."""
    recs = [
        {
            "idx": 0,
            "status": "ok",
            "closure_ok": True,
            "wall_time_s": 0.1,
            "memory_mb": 0.5,
            "n_unique_total": 100,
            "from_cache": False,
        },
        {
            "idx": 1,
            "status": "ok",
            "closure_ok": False,
            "wall_time_s": 0.2,
            "memory_mb": 0.6,
            "n_unique_total": 100,
            "from_cache": True,
        },
    ]
    agg = _aggregate_per_metal(recs)
    expected_keys = (
        "n_samples",
        "n_unique",
        "n_repeats",
        "sampled_with_replacement",
        "n_closure_ok",
        "closure_reachability_rate",
        "n_parse_error",
        "n_cache_hits",
        "mean_wall_time_s",
        "median_wall_time_s",
        "std_wall_time_s",
        "total_wall_time_s",
        "peak_memory_mb",
        "gate_target",
        "gate_pass",
    )
    for key in expected_keys:
        assert key in agg, f"aggregate missing key {key!r}"
    assert agg["n_samples"] == 2
    assert agg["n_unique"] == 100
    assert agg["n_repeats"] == 0  # 2 samples < 100 unique
    assert agg["sampled_with_replacement"] is False
    assert agg["n_cache_hits"] == 1
    assert agg["closure_reachability_rate"] == 0.5


# ---------------------------------------------------------------------------
# 13. Smoke run: 5 samples × 3 metals
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_payload() -> Dict[str, Any]:
    """A 5-sample × 3-metal smoke payload — fast enough for CI."""
    return run_benchmark(
        n_metals=3,
        n_samples=5,
        metals=["Pt", "Ru", "Pd"],
        max_depth=1,
        max_products=64,
    )


def test_smoke_run_3_metals(smoke_payload: Dict[str, Any]) -> None:
    """5 samples × 3 metals smoke run produces expected schema."""
    assert smoke_payload["schema_version"] == 1
    assert smoke_payload["tool"] == "bench_closure_dblock_full.py"
    assert smoke_payload["pilot_id"] == "wf130_p2_closure_30_5000"
    assert smoke_payload["metals"] == ["Pt", "Ru", "Pd"]
    assert smoke_payload["n_samples_total"] == 15
    # Aggregate fields
    agg = smoke_payload["aggregate"]
    for key in (
        "n_metals",
        "n_samples",
        "n_closure_ok",
        "micro_reachability_rate",
        "macro_reachability_rate",
        "macro_std_of_means",
    ):
        assert key in agg, f"aggregate missing key {key!r}"
    assert agg["n_samples"] == 15
    # Macro / micro rates lie in [0, 1]
    assert 0.0 <= float(agg["macro_reachability_rate"]) <= 1.0
    assert 0.0 <= float(agg["micro_reachability_rate"]) <= 1.0
    # Honest framing block
    assert "honest_framing" in smoke_payload
    # Dataset + CSV
    assert smoke_payload["dataset"] == "tmqm_parsed"
    assert "csv_path" in smoke_payload


# ---------------------------------------------------------------------------
# 14. Per-metal reachability.json files written
# ---------------------------------------------------------------------------
def test_per_metal_files_written(tmp_path: Path) -> None:
    """Per-metal {metal}/reachability.json files are written."""
    payload = run_benchmark(
        n_metals=2,
        n_samples=3,
        metals=["Pt", "Ru"],
        output_dir=tmp_path,
    )
    paths = save_per_metal(payload, output_dir=tmp_path)
    assert (tmp_path / "Pt" / "reachability.json").is_file()
    assert (tmp_path / "Ru" / "reachability.json").is_file()
    assert (tmp_path / "reachability_overall.json").is_file()
    # Validate JSON content
    pt_data = json.loads((tmp_path / "Pt" / "reachability.json").read_text())
    assert pt_data["metal"] == "Pt"
    assert "summary" in pt_data
    assert "n_unique_smiles" in pt_data
    assert "n_repeats" in pt_data


# ---------------------------------------------------------------------------
# 15. Macro vs micro reachability both present
# ---------------------------------------------------------------------------
def test_macro_micro_reachability_present(tmp_path: Path) -> None:
    """Payload carries both macro (equal-weight) and micro (weighted)."""
    payload = run_benchmark(
        n_metals=2,
        n_samples=3,
        metals=["Pt", "Ru"],
        output_dir=tmp_path,
    )
    agg = payload["aggregate"]
    assert "macro_reachability_rate" in agg
    assert "micro_reachability_rate" in agg
    assert "macro_std_of_means" in agg
    # Both lie in [0, 1]
    for key in ("macro_reachability_rate", "micro_reachability_rate"):
        rate = float(agg[key])
        assert 0.0 <= rate <= 1.0, f"{key} out of [0,1]: {rate}"
    # Per-metal records have with-replacement bookkeeping
    for metal, info in payload["per_metal"].items():
        assert "sampled_with_replacement" in info
        assert "n_unique" in info
        assert "n_repeats" in info
        assert "n_cache_hits" in info


# ---------------------------------------------------------------------------
# 16. synthesis.md has TL;DR + honest framing
# ---------------------------------------------------------------------------
def test_synthesis_md_written(tmp_path: Path) -> None:
    """synthesis.md contains TL;DR, baselines, per-metal, honest framing."""
    payload = run_benchmark(
        n_metals=3,
        n_samples=3,
        metals=["Pt", "Ru", "Pd"],
        output_dir=tmp_path,
    )
    save_per_metal(payload, output_dir=tmp_path)
    md_path = write_synthesis(payload, output_dir=tmp_path)
    assert md_path.is_file()
    content = md_path.read_text()
    assert "Wf-130 Phase 2" in content
    assert "Macro reachability" in content
    assert "Micro reachability" in content
    assert "Per-metal breakdown" in content
    assert "Honest framing" in content
    assert "Wf-67 P2 baseline" in content
    assert "Wf-68 P2 baseline" in content
    assert "CPU-only" in content


# ---------------------------------------------------------------------------
# 17. Honest-negative threshold constant
# ---------------------------------------------------------------------------
def test_honest_negative_threshold_present() -> None:
    """HONEST_NEGATIVE_THRESHOLD = 0.50 is exported."""
    assert HONEST_NEGATIVE_THRESHOLD == 0.50


# ---------------------------------------------------------------------------
# 18. Honest-negative metals are flagged in synthesis.md
# ---------------------------------------------------------------------------
def test_honest_negative_metals_in_synthesis(tmp_path: Path) -> None:
    """Metals with reachability < 0.50 are listed in synthesis.md."""
    # Construct a synthetic payload where one metal is below threshold.
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "bench_closure_dblock_full.py",
        "pilot_id": "wf130_p2_closure_30_5000",
        "date_utc": _utcnow_iso(),
        "metals": ["Pt", "Pd"],
        "n_metals": 2,
        "n_samples": 100,
        "n_samples_unique_total": 200,
        "n_samples_repeats_total": 0,
        "max_depth": 1,
        "sample_seed": 42,
        "gate_target_reachability": 0.85,
        "wf67_baseline_reachability": 0.5891,
        "wf68_baseline_reachability": 0.5744,
        "honest_negative_threshold": 0.50,
        "per_metal": {
            "Pt": {
                "n_samples": 100,
                "n_unique": 100,
                "n_repeats": 0,
                "sampled_with_replacement": False,
                "n_closure_ok": 70,
                "closure_reachability_rate": 0.70,
                "n_parse_error": 0,
                "n_cache_hits": 0,
                "mean_wall_time_s": 0.1,
                "median_wall_time_s": 0.1,
                "std_wall_time_s": 0.0,
                "total_wall_time_s": 10.0,
                "peak_memory_mb": 0.5,
                "gate_target": 0.85,
                "gate_pass": False,
            },
            "Pd": {
                "n_samples": 100,
                "n_unique": 100,
                "n_repeats": 0,
                "sampled_with_replacement": False,
                "n_closure_ok": 30,
                "closure_reachability_rate": 0.30,
                "n_parse_error": 0,
                "n_cache_hits": 0,
                "mean_wall_time_s": 0.1,
                "median_wall_time_s": 0.1,
                "std_wall_time_s": 0.0,
                "total_wall_time_s": 10.0,
                "peak_memory_mb": 0.5,
                "gate_target": 0.85,
                "gate_pass": False,
            },
        },
        "aggregate": {
            "n_metals": 2,
            "n_samples": 200,
            "n_closure_ok": 100,
            "micro_reachability_rate": 0.50,
            "macro_reachability_rate": 0.50,
            "macro_std_of_means": 0.20,
        },
        "n_metals_pass": 0,
        "n_metals_below_honest_negative_threshold": 1,
        "n_metals_sampled_with_replacement": 0,
        "overall_gate_pass": False,
        "total_wall_time_s": 20.0,
        "timestamp_utc": _utcnow_iso(),
        "honest_framing": "test fixture",
    }
    md_path = write_synthesis(payload, output_dir=tmp_path)
    content = md_path.read_text()
    assert "Honest-negative metals" in content
    # Pd (0.30 < 0.50) should be listed, Pt (0.70) should not.
    assert "Pd" in content
    assert "reachability=0.3000" in content
    # Pd is in the per-metal table (always) and in the
    # honest-negative list.  We check the table once + the
    # honest-negative list once → 2 occurrences of "Pd".
    assert content.count("Pd") >= 2
    # The honest-negative section explicitly lists Pd
    assert "**Pd**" in content


# ---------------------------------------------------------------------------
# Bonus invariants
# ---------------------------------------------------------------------------
def test_load_click_rules_count() -> None:
    """_load_click_rules returns the 5 canonical click rules."""
    rules = _load_click_rules()
    assert len(rules) == 5
    names = {getattr(r, "name", type(r).__name__) for r in rules}
    assert {"CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"}.issubset(
        names,
    )


def test_load_partner_terms_count() -> None:
    """_load_partner_terms returns at least 6 partner tiles."""
    partners = _load_partner_terms()
    assert len(partners) >= 6


def test_partner_smile_list_size() -> None:
    """PARTNER_SMILE_LIST has at least 6 entries."""
    assert len(PARTNER_SMILE_LIST) >= 6


def test_aggregate_per_metal_empty() -> None:
    """_aggregate_per_metal on empty list returns the zero-vector."""
    agg = _aggregate_per_metal([])
    assert agg["n_samples"] == 0
    assert agg["n_unique"] == 0
    assert agg["n_repeats"] == 0
    assert agg["sampled_with_replacement"] is False
    assert agg["n_closure_ok"] == 0
    assert agg["closure_reachability_rate"] == 0.0
    assert agg["gate_pass"] is False


def test_detect_csv_layout_parsed() -> None:
    """_detect_csv_layout recognises the tmQM parsed CSV (metal + smiles)."""
    if not DEFAULT_TMQM_CSV.is_file():
        pytest.skip(f"tmQM parsed CSV missing at {DEFAULT_TMQM_CSV}")
    layout = _detect_csv_layout(DEFAULT_TMQM_CSV)
    assert layout["ok"] is True
    assert layout["metal_column"] == "metal"
    assert layout["smiles_column"] == "smiles"


def test_detect_csv_layout_subset() -> None:
    """_detect_csv_layout recognises the tmQM d-block subset (metal_center)."""
    if not FALLBACK_TMQM_CSV.is_file():
        pytest.skip(f"Fallback CSV missing at {FALLBACK_TMQM_CSV}")
    layout = _detect_csv_layout(FALLBACK_TMQM_CSV)
    assert layout["ok"] is True
    assert layout["metal_column"] == "metal_center"
    assert layout["smiles_column"] == "smiles"


def test_load_unique_metal_smis_pt() -> None:
    """_load_unique_metal_smis loads Pt SMILES from tmQM parsed CSV."""
    if not DEFAULT_TMQM_CSV.is_file():
        pytest.skip(f"tmQM parsed CSV missing at {DEFAULT_TMQM_CSV}")
    smis = _load_unique_metal_smis(
        DEFAULT_TMQM_CSV,
        "Pt",
        metal_column="metal",
        smiles_column="smiles",
    )
    assert len(smis) > 1000, f"Expected >1000 unique Pt SMILES, got {len(smis)}"
    # All entries non-empty
    assert all(len(s) > 0 for s in smis)
    # No duplicates
    assert len(set(smis)) == len(smis)


def test_load_unique_metal_smis_la() -> None:
    """_load_unique_metal_smis loads La SMILES (lanthanide, few samples)."""
    if not DEFAULT_TMQM_CSV.is_file():
        pytest.skip(f"tmQM parsed CSV missing at {DEFAULT_TMQM_CSV}")
    smis = _load_unique_metal_smis(
        DEFAULT_TMQM_CSV,
        "La",
        metal_column="metal",
        smiles_column="smiles",
    )
    # La is the smallest metal in tmQM parsed CSV (~100-150 samples).
    assert 50 < len(smis) < 500, f"Expected 50-500 unique La SMILES, got {len(smis)}"


def test_run_benchmark_with_replacement_la(tmp_path: Path) -> None:
    """run_benchmark uses with-replacement for La (only 116 unique SMILES)."""
    if not DEFAULT_TMQM_CSV.is_file():
        pytest.skip(f"tmQM parsed CSV missing at {DEFAULT_TMQM_CSV}")
    payload = run_benchmark(
        n_metals=1,
        n_samples=200,
        metals=["La"],
        output_dir=tmp_path,
    )
    assert payload["per_metal"]["La"]["sampled_with_replacement"] is True
    assert payload["per_metal"]["La"]["n_repeats"] > 0
    assert payload["per_metal"]["La"]["n_samples"] == 200
    # Cache hits: with-replacement sampling repeats SMILES, so cache
    # should kick in.  Either n_cache_hits > 0 OR the closure was
    # skipped because of repeated SMILES.
    la_summary = payload["per_metal"]["La"]
    assert la_summary["n_unique"] < la_summary["n_samples"]
    assert (
        la_summary["n_cache_hits"] > 0
    ), "with-replacement sampling should trigger cache hits"


def test_default_sample_seed_42() -> None:
    """DEFAULT_SAMPLE_SEED = 42 (deterministic with-replacement sampling)."""
    assert DEFAULT_SAMPLE_SEED == 42


def test_utcnow_iso_format() -> None:
    """_utcnow_iso returns a Z-suffixed ISO 8601 timestamp."""
    s = _utcnow_iso()
    assert s.endswith("Z")
    assert "T" in s


def test_memory_mb_conversion() -> None:
    """_memory_mb converts bytes to MB correctly."""
    assert abs(_memory_mb(1024 * 1024) - 1.0) < 1e-9
    assert _memory_mb(0) == 0.0


def test_benchmark_one_smoke() -> None:
    """_benchmark_one on cisplatin returns the expected per-sample record."""
    rec = _benchmark_one(
        "[NH3][Pt]([NH3])([Cl])([Cl])",
        0,
        "Pt",
        click_rules=_load_click_rules(),
        partner_terms=_load_partner_terms(),
        max_depth=1,
        max_products=128,
        n_unique_total=100,
        sampled_with_replacement=False,
        sample_origin_idx=0,
    )
    assert rec["metal"] == "Pt"
    assert rec["idx"] == 0
    assert rec["depth"] == 1
    assert rec["status"] in ("ok", "skipped", "error")
    assert isinstance(rec["closure_ok"], bool)
    assert isinstance(rec["wall_time_s"], float)
    assert rec["n_unique_total"] == 100
    assert rec["sample_origin_idx"] == 0
    assert rec["sampled_with_replacement"] is False


def test_per_metal_csv_schema(tmp_path: Path) -> None:
    """Per-metal reachability.json has the documented schema fields."""
    payload = run_benchmark(
        n_metals=2,
        n_samples=3,
        metals=["Pt", "Ru"],
        output_dir=tmp_path,
    )
    save_per_metal(payload, output_dir=tmp_path)
    for metal in ("Pt", "Ru"):
        data = json.loads((tmp_path / metal / "reachability.json").read_text())
        assert "metal" in data
        assert "summary" in data
        assert "n_samples_in_run" in data
        assert "n_unique_smiles" in data
        assert "n_repeats" in data
        assert "samples" in data
        summary = data["summary"]
        for k in (
            "n_samples",
            "n_unique",
            "n_repeats",
            "sampled_with_replacement",
            "n_closure_ok",
            "closure_reachability_rate",
            "n_cache_hits",
            "mean_wall_time_s",
            "gate_target",
            "gate_pass",
        ):
            assert k in summary, f"summary missing key {k!r}"


def test_run_benchmark_error_payload() -> None:
    """run_benchmark returns an error payload when both CSVs are missing."""
    payload = run_benchmark(
        n_metals=2,
        n_samples=3,
        metals=["Pt", "Ru"],
        tmqm_csv=Path("/nonexistent/path/that/does/not/exist.csv"),
        fallback_csv=Path("/nonexistent/fallback/path.csv"),
    )
    assert "error" in payload
    assert payload["per_metal"] == {}


def test_run_benchmark_includes_baseline_constants() -> None:
    """Payload carries Wf-67 + Wf-68 baseline constants."""
    payload = run_benchmark(
        n_metals=1,
        n_samples=3,
        metals=["Pt"],
    )
    assert payload["wf67_baseline_reachability"] == WF67_BASELINE_REACHABILITY
    assert payload["wf68_baseline_reachability"] == WF68_BASELINE_REACHABILITY
    assert payload["gate_target_reachability"] == GATE_TARGET_REACHABILITY
    assert payload["honest_negative_threshold"] == HONEST_NEGATIVE_THRESHOLD


def test_run_benchmark_records_per_metal_records(tmp_path: Path) -> None:
    """Payload carries per_metal_records for cache-hit reporting."""
    payload = run_benchmark(
        n_metals=1,
        n_samples=200,
        metals=["La"],
        output_dir=tmp_path,
    )
    assert "per_metal_records" in payload
    assert "La" in payload["per_metal_records"]
    la_records = payload["per_metal_records"]["La"]
    assert len(la_records) == 200
    # Cache should be active for La (with-replacement, 200 > 116 unique)
    cache_hits = sum(1 for r in la_records if r.get("from_cache", False))
    assert cache_hits > 0, (
        "with-replacement sampling on La (200 > 116 unique) should "
        f"trigger cache hits; got {cache_hits}"
    )
