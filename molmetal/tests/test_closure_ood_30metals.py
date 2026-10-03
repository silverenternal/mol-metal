"""Tests for the Wf-130 Phase 3 closure OOD 30-metals benchmark.

================================================================
Wf-130 P3 closure-OOD 30-metals expansion contract — 30 d-block
metals, Tanimoto<0.4 OOD protocol on PlatinAI + MetalCytoTox
cross-domain data, CPU-only, deterministic.

This file ships the **14 pytest contracts** that pin the behaviour of
:mod:`molmetal.scripts.bench_closure_ood_30metals`:

Configuration / module surface
 1. test_module_imports                — script imports cleanly
 2. test_default_30_metals             — DBLOCK_30_METALS has 30 entries
 3. test_gate_target_0_30              — per-metal gate = 0.30
 4. test_wf41_baselines_constant       — Wf-41 P2 baseline constants
 5. test_honest_negative_threshold_0_30 — HONEST_NEGATIVE_THRESHOLD = 0.30
 6. test_default_csv_paths             — CSV paths resolve

OOD filter mechanics
 7. test_filter_ood_subset_filters     — OOD filter drops known-similar
 8. test_filter_ood_subset_empty_train — empty train pool → all OOD

Benchmark mechanics
 9. test_aggregate_per_metal_no_data   — empty input → reachability None
10. test_aggregate_per_metal_with_data  — populated input → reachability float
11. test_source_breakdown_in_aggregate — per-source breakdown present
12. test_parse_args_defaults           — default CLI args = 30 metals × 100

Smoke run / output contract
13. test_smoke_run_3_metals_end_to_end — 5 samples × 3 metals smoke OK
14. test_per_metal_files_written       — 30 dirs (or as many as tested)

All tests CPU-only, deterministic, complete < 90 s combined.
Smoke runs use ``--n-samples 3 --metals Pt Ru Re --no-parallel`` for
sub-30 s CI.

Run with::

    uv run --no-sync pytest -q molmetal/tests/test_closure_ood_30metals.py --tb=short
"""

from __future__ import annotations

import csv
import json
import sys
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


from molmetal.scripts import bench_closure_ood_30metals  # noqa: E402
from molmetal.scripts.bench_closure_ood_30metals import (  # noqa: E402
    DBLOCK_30_METALS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PRODUCTS,
    DEFAULT_N_METALS,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLATINAI_CSV,
    DEFAULT_METALCYTO_CSV,
    DEFAULT_TMQM_CSV,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SEEDS,
    DEFAULT_TANIMOTO_THRESHOLD,
    FALLBACK_METALCYTO_CSV,
    GATE_TARGET_PER_METAL_REACHABILITY,
    HONEST_NEGATIVE_THRESHOLD,
    METAL_Z,
    PARTNER_SMILE_LIST,
    WF41_AU_REACHABILITY,
    WF41_IR_REACHABILITY,
    WF41_METALCYTO_MACRO_REACHABILITY,
    WF41_OS_REACHABILITY,
    WF41_PD_REACHABILITY,
    WF41_PLATINAI_MACRO_REACHABILITY,
    WF41_PT_REACHABILITY,
    WF41_RE_REACHABILITY,
    WF41_RH_REACHABILITY,
    WF41_RU_REACHABILITY,
    _aggregate_per_metal,
    _benchmark_one,
    _filter_ood_subset,
    _load_click_rules,
    _load_metal_samples,
    _load_partner_terms,
    _load_tanimoto_protocol,
    _load_tmqm_train_pool,
    _memory_mb,
    _utcnow_iso,
    run_benchmark,
    save_per_metal,
    write_synthesis,
)


# ---------------------------------------------------------------------------
# RDKit availability probe — needed by OOD filter
# ---------------------------------------------------------------------------
def _have_rdkit() -> bool:
    try:
        from rdkit import Chem  # noqa: F401

        return True
    except Exception:
        return False


pytestmark_rdkit = pytest.mark.skipif(
    not _have_rdkit(),
    reason="RDKit not available",
)


# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """Script imports cleanly with the expected entry points."""
    assert bench_closure_ood_30metals is not None
    assert hasattr(bench_closure_ood_30metals, "run_benchmark")
    assert hasattr(bench_closure_ood_30metals, "save_per_metal")
    assert hasattr(bench_closure_ood_30metals, "write_synthesis")
    assert hasattr(bench_closure_ood_30metals, "_aggregate_per_metal")
    assert hasattr(bench_closure_ood_30metals, "_filter_ood_subset")
    assert hasattr(bench_closure_ood_30metals, "_benchmark_metal_worker")


# ---------------------------------------------------------------------------
# 2. DBLOCK_30_METALS has 30 entries
# ---------------------------------------------------------------------------
def test_default_30_metals() -> None:
    """DBLOCK_30_METALS contains exactly 30 d-block metals."""
    assert len(DBLOCK_30_METALS) == 30
    # Spot-check the 9 task-spec metals
    for m in ("Pt", "Ru", "Ir", "Rh", "Re", "Os", "Au", "Pd", "Ag"):
        assert m in DBLOCK_30_METALS
    # Spot-check the 5 previously-not-audited (Wf-130 P1) metals
    for m in ("La", "Hf", "Ta", "Au", "Hg"):
        assert m in DBLOCK_30_METALS


# ---------------------------------------------------------------------------
# 3. Per-metal gate target = 0.30
# ---------------------------------------------------------------------------
def test_gate_target_0_30() -> None:
    """Per-metal gate target = 0.30 (Wf-130 P3 spec)."""
    assert GATE_TARGET_PER_METAL_REACHABILITY == 0.30


# ---------------------------------------------------------------------------
# 4. Wf-41 P2 baselines are constants
# ---------------------------------------------------------------------------
def test_wf41_baselines_constant() -> None:
    """Wf-41 P2 per-metal baselines (Pt=0.5833 / Re=0.0000) recorded."""
    assert abs(WF41_PT_REACHABILITY - 0.5833) < 1e-6
    assert abs(WF41_PD_REACHABILITY - 0.3333) < 1e-6
    assert abs(WF41_RU_REACHABILITY - 0.7000) < 1e-6
    assert abs(WF41_RH_REACHABILITY - 0.6471) < 1e-6
    assert abs(WF41_IR_REACHABILITY - 0.6000) < 1e-6
    assert abs(WF41_OS_REACHABILITY - 0.5294) < 1e-6
    assert abs(WF41_AU_REACHABILITY - 0.4545) < 1e-6
    assert abs(WF41_RE_REACHABILITY - 0.0000) < 1e-6
    # Macro baselines
    assert abs(WF41_PLATINAI_MACRO_REACHABILITY - 0.5581) < 1e-6
    assert abs(WF41_METALCYTO_MACRO_REACHABILITY - 0.0300) < 1e-6


# ---------------------------------------------------------------------------
# 5. Honest-negative threshold = 0.30
# ---------------------------------------------------------------------------
def test_honest_negative_threshold_0_30() -> None:
    """HONEST_NEGATIVE_THRESHOLD = 0.30 (mirrors the per-metal gate)."""
    assert HONEST_NEGATIVE_THRESHOLD == 0.30


# ---------------------------------------------------------------------------
# 6. Default CSV paths resolve
# ---------------------------------------------------------------------------
def test_default_csv_paths() -> None:
    """Default CSV paths point to known files."""
    assert DEFAULT_PLATINAI_CSV.name == "platinai_5k.csv"
    assert DEFAULT_TMQM_CSV.name == "tmqm_dblock_subset.csv"
    assert FALLBACK_METALCYTO_CSV.name == "metalcytotox_5metal.csv"


# ---------------------------------------------------------------------------
# 7. OOD filter mechanics (requires RDKit)
# ---------------------------------------------------------------------------
@pytestmark_rdkit
def test_filter_ood_subset_filters() -> None:
    """OOD filter drops SMILES too similar to the train pool."""
    ood_protocol = _load_tanimoto_protocol(threshold=0.4)
    train_pool = ["CCN=[N+]=[N-]", "C#CCN", "CS"]
    # Query in train pool → NOT OOD
    is_ood_in_train = ood_protocol.is_ood("CCN=[N+]=[N-]", train_pool)
    assert is_ood_in_train is False
    # Long-chain aliphatic → OOD vs the small azide/alkyne/thiol train
    is_ood_novel = ood_protocol.is_ood("CCCCCCCCCC", train_pool)
    assert is_ood_novel is True

    # Filter helper: returns (subset, audit)
    smis = ["CCN=[N+]=[N-]", "CCCCCCCCCC", "C#CCN"]
    subset, audit = _filter_ood_subset(smis, train_pool, ood_protocol, "Pt")
    assert isinstance(subset, list)
    assert "n_in" in audit
    assert audit["n_in"] == 3
    # The azide and alkyne are NOT OOD (in train pool); only decane
    # passes the OOD filter.
    assert audit["n_ood"] >= 1
    assert audit["ood_rate"] > 0.0
    ood_smiles = [s for (_, s, _) in subset]
    assert "CCCCCCCCCC" in ood_smiles
    assert "CCN=[N+]=[N-]" not in ood_smiles
    assert "mean_max_tanimoto" in audit


# ---------------------------------------------------------------------------
# 8. Empty train pool → all queries are OOD (fail-open)
# ---------------------------------------------------------------------------
@pytestmark_rdkit
def test_filter_ood_subset_empty_train() -> None:
    """Empty train pool → every query is OOD by definition."""
    ood_protocol = _load_tanimoto_protocol(threshold=0.4)
    smis = ["CCN=[N+]=[N-]", "c1ccc(O)cc1N", "C#CCN"]
    subset, audit = _filter_ood_subset(smis, [], ood_protocol, "Pt")
    assert audit["n_in"] == 3
    assert audit["n_ood"] == 3
    assert audit["ood_rate"] == 1.0
    assert audit["mean_max_tanimoto"] == 0.0


# ---------------------------------------------------------------------------
# 9. _aggregate_per_metal: empty input → reachability None
# ---------------------------------------------------------------------------
def test_aggregate_per_metal_no_data() -> None:
    """Empty records → reachability_rate = None (honest framing)."""
    out = _aggregate_per_metal(
        [],
        n_input_total=0,
        n_ood_total=0,
        n_cross_domain_sources=0,
    )
    assert out["n_samples"] == 0
    assert out["n_closure_ok"] == 0
    assert out["closure_reachability_rate"] is None
    assert out["gate_pass"] is False
    assert out["status"] == "no_cross_domain_source"
    assert out["n_cross_domain_sources"] == 0


# ---------------------------------------------------------------------------
# 10. _aggregate_per_metal: populated input → rate is float
# ---------------------------------------------------------------------------
def test_aggregate_per_metal_with_data() -> None:
    """Populated records → reachability_rate is a float in [0, 1]."""
    fake_records = [
        {
            "idx": 0,
            "metal": "Pt",
            "source": "platinai_5k",
            "closure_ok": True,
            "status": "ok",
            "wall_time_s": 0.05,
            "memory_mb": 1.0,
            "max_tanimoto": 0.1,
        },
        {
            "idx": 1,
            "metal": "Pt",
            "source": "platinai_5k",
            "closure_ok": False,
            "status": "ok",
            "wall_time_s": 0.04,
            "memory_mb": 0.8,
            "max_tanimoto": 0.2,
        },
        {
            "idx": 2,
            "metal": "Pt",
            "source": "platinai_5k",
            "closure_ok": True,
            "status": "ok",
            "wall_time_s": 0.06,
            "memory_mb": 1.1,
            "max_tanimoto": 0.05,
        },
    ]
    out = _aggregate_per_metal(
        fake_records,
        n_input_total=10,
        n_ood_total=3,
        n_cross_domain_sources=1,
    )
    assert out["n_samples"] == 3
    assert out["n_closure_ok"] == 2
    assert abs(out["closure_reachability_rate"] - 0.6666) < 0.01
    assert out["n_cross_domain_sources"] == 1
    assert out["status"] == "ok"
    assert out["gate_pass"] is True  # 0.6666 >= 0.30
    assert out["n_parse_error"] == 0


# ---------------------------------------------------------------------------
# 11. Source breakdown in aggregate
# ---------------------------------------------------------------------------
def test_source_breakdown_in_aggregate() -> None:
    """Aggregate includes per-source breakdown (platinai / metalcytotox)."""
    fake_records = [
        {
            "idx": 0,
            "metal": "Ru",
            "source": "platinai_5k",
            "closure_ok": True,
            "status": "ok",
            "wall_time_s": 0.05,
            "memory_mb": 1.0,
        },
        {
            "idx": 1,
            "metal": "Ru",
            "source": "metalcytotox_5metal",
            "closure_ok": False,
            "status": "ok",
            "wall_time_s": 0.04,
            "memory_mb": 0.8,
        },
    ]
    out = _aggregate_per_metal(
        fake_records,
        n_input_total=20,
        n_ood_total=2,
        n_cross_domain_sources=2,
    )
    assert "source_breakdown" in out
    sb = out["source_breakdown"]
    assert "platinai_5k" in sb
    assert "metalcytotox_5metal" in sb
    assert sb["platinai_5k"]["n_samples"] == 1
    assert sb["platinai_5k"]["n_closure_ok"] == 1
    assert sb["metalcytotox_5metal"]["n_samples"] == 1
    assert sb["metalcytotox_5metal"]["n_closure_ok"] == 0
    assert out["n_cross_domain_sources"] == 2


# ---------------------------------------------------------------------------
# 12. CLI argparse defaults
# ---------------------------------------------------------------------------
def test_parse_args_defaults() -> None:
    """Default args = 30 metals × 100 samples; max_depth=1; threshold=0.4."""
    from molmetal.scripts.bench_closure_ood_30metals import (
        _build_argparser,
    )

    parser = _build_argparser()
    args = parser.parse_args([])
    assert args.n_metals == DEFAULT_N_METALS == 30
    assert args.n_samples == DEFAULT_N_SAMPLES == 100
    assert args.max_depth == DEFAULT_MAX_DEPTH == 1
    assert args.max_products == DEFAULT_MAX_PRODUCTS == 64
    assert args.tanimoto_threshold == DEFAULT_TANIMOTO_THRESHOLD == 0.4
    assert args.sample_seed == DEFAULT_SAMPLE_SEED == 42
    assert args.log_level == "INFO"


# ---------------------------------------------------------------------------
# 13. Smoke run: 3 metals × 5 samples end-to-end
# ---------------------------------------------------------------------------
@pytestmark_rdkit
def test_smoke_run_3_metals_end_to_end(tmp_path: Path) -> None:
    """3-metal × 5-sample end-to-end smoke (no-parallel)."""
    out_dir = tmp_path / "wf130_p3_smoke"
    payload = run_benchmark(
        n_metals=3,
        n_samples=5,
        metals=("Pt", "Ru", "Re"),
        platinai_csv=DEFAULT_PLATINAI_CSV,
        metalcyto_csv=DEFAULT_METALCYTO_CSV,
        metalcyto_fallback_csv=FALLBACK_METALCYTO_CSV,
        tmqm_csv=DEFAULT_TMQM_CSV,
        output_dir=out_dir,
        max_depth=1,
        max_products=64,
        seeds=1,
        sample_seed=42,
        tanimoto_threshold=0.4,
        parallel=False,  # single-threaded for test determinism
    )
    # Validate payload shape
    assert "per_metal" in payload
    assert "aggregate" in payload
    for m in ("Pt", "Ru", "Re"):
        assert m in payload["per_metal"]
        s = payload["per_metal"][m]
        assert s["n_samples"] >= 0
        # Pt is in PlatinAI, Re is in MetalCytoTox → both should have
        # data.  Ru is in both → also has data.
        if m in ("Pt", "Re", "Ru"):
            assert s["n_cross_domain_sources"] >= 1
    agg = payload["aggregate"]
    assert agg["n_metals"] == 3
    assert agg["n_metals_with_data"] >= 1
    assert agg["n_samples"] >= 0

    # Save artefacts
    paths = save_per_metal(payload, output_dir=out_dir)
    for m in ("Pt", "Ru", "Re"):
        assert (out_dir / m / "ood_reachability.json").is_file()

    # Synthesis md
    md_path = write_synthesis(payload, output_dir=out_dir)
    assert md_path.is_file()
    md_text = md_path.read_text()
    assert "Wf-130 Phase 3" in md_text
    assert "Per-metal breakdown" in md_text
    assert "Honest framing" in md_text


# ---------------------------------------------------------------------------
# 14. Per-metal files written
# ---------------------------------------------------------------------------
@pytestmark_rdkit
def test_per_metal_files_written(tmp_path: Path) -> None:
    """Per-metal JSON written for every metal in the run."""
    out_dir = tmp_path / "wf130_p3_permetal"
    metals = (
        "Pt",
        "Ru",
        "Re",
        "Au",
    )  # 4 metals: Pt/Au/Ru (PlatinAI), Ru/Ir/Re (MetalCytoTox)
    payload = run_benchmark(
        n_metals=4,
        n_samples=3,
        metals=metals,
        platinai_csv=DEFAULT_PLATINAI_CSV,
        metalcyto_csv=DEFAULT_METALCYTO_CSV,
        metalcyto_fallback_csv=FALLBACK_METALCYTO_CSV,
        tmqm_csv=DEFAULT_TMQM_CSV,
        output_dir=out_dir,
        max_depth=1,
        max_products=64,
        seeds=1,
        sample_seed=42,
        tanimoto_threshold=0.4,
        parallel=False,
    )
    paths = save_per_metal(payload, output_dir=out_dir)
    for m in metals:
        assert m in paths
        assert paths[m].is_file()
        # Verify the JSON schema
        rec = json.loads(paths[m].read_text())
        assert "metal" in rec
        assert rec["metal"] == m
        assert "summary" in rec
        assert "ood_audit" in rec
        assert "samples" in rec
    # Overall JSON also exists
    overall = out_dir / "ood_reachability_overall.json"
    assert overall.is_file()
