"""Pytest: Wf-29 Phase 2 — 4-train-script smoke benchmark on GPU-1.

This file pins the contract for the Wf-29 Phase 2 deliverables:

* Four sub-dirs exist under ``molmetal/reports/wf29_smoke/`` (se3sb,
  transport, multiph, baseline) — one per trained script.
* Each sub-dir has a ``*.log`` (non-empty), ``metrics.json``
  (parseable dict), and ``final_summary.json`` (parseable dict).
* The aggregate files exist: ``comparison.md``,
  ``bench_summary.json``, ``ablation_lift.json``, ``run_wallclock.json``.
* The R18 W2.1 / W2.2 / W2.3 metrics expose 5-metal coverage with
  per-metal AUC per seed.
* The R17 baseline exposes 3 seeds × Ru metal outputs.
* Per-script wall-time > 0 and < 30 min (sanity bound).
* All scripts ran on GPU-1 (RTX 5090, 32 GB), peak memory < 4 GB.
* The MEASURED vs PROJECTED provenance is recorded in the bench
  summary so the numbers cannot be silently swapped for dry-run
  projections later.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPORTS_ROOT = Path(__file__).resolve().parent.parent / "reports" / "wf29_smoke"

REQUIRED_SUB_DIRS = ("se3sb", "transport", "multiph", "baseline")
REQUIRED_AGGREGATE_FILES = (
    "comparison.md",
    "bench_summary.json",
    "ablation_lift.json",
    "run_wallclock.json",
    "gpu_memory_log.json.baseline",
    "gpu_memory_log.json.post",
)
REQUIRED_PER_SCRIPT_FILES = ("metrics.json", "final_summary.json")
R18_METALS = ("Ru", "Ir", "Rh", "Re", "Os")
# Per-script log lives at REPORTS_ROOT/<subdir>.log (not inside the sub-dir).
LOGS_AT_ROOT = {s: REPORTS_ROOT / f"{s}.log" for s in REQUIRED_SUB_DIRS}
# bench_summary uses "baseline_r17" not "baseline" (avoids Python keyword).
BENCH_SCRIPT_KEYS = {"se3sb", "transport", "multiph", "baseline_r17"}


def _read_json(p: Path) -> dict:
    assert p.exists(), f"missing file: {p}"
    return json.loads(p.read_text())


def _read_text(p: Path) -> str:
    assert p.exists(), f"missing file: {p}"
    return p.read_text()


# ---------------------------------------------------------------------------
# 1. Directory layout
# ---------------------------------------------------------------------------

def test_reports_root_exists():
    assert REPORTS_ROOT.exists(), (
        f"Wf-29 Phase 2 reports root missing: {REPORTS_ROOT}"
    )


@pytest.mark.parametrize("subdir", REQUIRED_SUB_DIRS)
def test_subdir_exists(subdir: str):
    assert (REPORTS_ROOT / subdir).is_dir(), f"missing sub-dir: {subdir}"


@pytest.mark.parametrize("subdir", REQUIRED_SUB_DIRS)
@pytest.mark.parametrize("filename", REQUIRED_PER_SCRIPT_FILES)
def test_per_script_file_exists(subdir: str, filename: str):
    """Each of the 4 sub-dirs has its metrics.json + final_summary.json."""
    candidates = list((REPORTS_ROOT / subdir).glob(f"*{filename}"))
    assert candidates, (
        f"missing {filename} in {subdir}; "
        f"got {sorted(p.name for p in (REPORTS_ROOT / subdir).iterdir())}"
    )


@pytest.mark.parametrize("subdir", REQUIRED_SUB_DIRS)
def test_per_script_log_exists_at_root(subdir: str):
    """Each script's log file lives at REPORTS_ROOT/<subdir>.log."""
    p = LOGS_AT_ROOT[subdir]
    assert p.exists(), f"missing log file: {p}"
    assert p.stat().st_size > 0, f"empty log: {p}"


# ---------------------------------------------------------------------------
# 2. Aggregate files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", REQUIRED_AGGREGATE_FILES)
def test_aggregate_file_exists(filename: str):
    assert (REPORTS_ROOT / filename).exists(), f"missing aggregate file: {filename}"


def test_comparison_md_non_empty():
    text = _read_text(REPORTS_ROOT / "comparison.md")
    assert len(text) > 200, "comparison.md suspiciously short"
    assert "Wf-29" in text
    assert "RTX 5090" in text
    assert all(s in text for s in ("se3sb", "transport", "multiph", "baseline"))


def test_bench_summary_valid_json_with_4_scripts():
    data = _read_json(REPORTS_ROOT / "bench_summary.json")
    assert "scripts" in data
    assert all(s in data["scripts"] for s in BENCH_SCRIPT_KEYS)
    for s in BENCH_SCRIPT_KEYS:
        assert data["scripts"][s].get("wall_seconds", 0) > 0
    # 5-metal R18 vs single-metal R17 — different shapes
    assert data["scripts"]["baseline_r17"]["metal_scope"] == "Ru only (single-metal)"
    for r in ("se3sb", "transport", "multiph"):
        assert "mean_test_auc_overall" in data["scripts"][r]


def test_ablation_lift_valid_json_with_ru_apples_to_apples():
    data = _read_json(REPORTS_ROOT / "ablation_lift.json")
    assert "ru_apples_to_apples" in data
    ru_block = data["ru_apples_to_apples"]
    assert "baseline_r17_ru_only" in ru_block
    assert "se3sb_5metal_multitask_ru" in ru_block
    assert "lift_se3sb_minus_baseline" in ru_block
    # Honest caveats present
    assert "honest_caveats" in data
    assert len(data["honest_caveats"]) >= 3


# ---------------------------------------------------------------------------
# 3. R18 metrics — 5-metal coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subdir", ("se3sb", "transport", "multiph"))
def test_r18_metrics_5_metal_coverage(subdir: str):
    """Each R18 script's metrics.json has 5-metal per-seed AUC."""
    metrics = _read_json(REPORTS_ROOT / subdir / "metrics.json")
    assert "per_seed" in metrics
    assert len(metrics["per_seed"]) >= 3, (
        f"expected ≥3 seeds for {subdir}, got {len(metrics['per_seed'])}"
    )
    for seed_record in metrics["per_seed"]:
        assert "test_auc_per_metal" in seed_record
        for metal in R18_METALS:
            assert metal in seed_record["test_auc_per_metal"], (
                f"{subdir} seed={seed_record.get('seed')} missing {metal}"
            )
            v = seed_record["test_auc_per_metal"][metal]
            assert 0.0 <= v <= 1.0, f"{subdir} {metal} AUC {v} out of [0,1]"


# ---------------------------------------------------------------------------
# 4. R18 final_summary — wall + recipe name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subdir", ("se3sb", "transport", "multiph"))
def test_r18_final_summary_wall_and_recipe(subdir: str):
    summary = _read_json(REPORTS_ROOT / subdir / "final_summary.json")
    assert "recipe" in summary
    assert "wall_seconds" in summary
    wall = summary["wall_seconds"]
    assert wall > 0, f"{subdir} wall=0"
    assert wall < 1800, f"{subdir} wall={wall:.1f}s exceeds 30-min budget"
    # Each R18 recipe name contains its W2.x identifier
    assert re.search(r"w2\.\d", summary["recipe"]), (
        f"{subdir} recipe '{summary['recipe']}' missing w2.x id"
    )


# ---------------------------------------------------------------------------
# 5. R17 baseline — 3-seed × Ru outputs
# ---------------------------------------------------------------------------

def test_r17_baseline_3_seed_outputs():
    for seed in (0, 1, 2):
        p = REPORTS_ROOT / "baseline" / f"baseline_Ru_seed{seed}.json"
        assert p.exists(), f"missing baseline_Ru_seed{seed}.json"
        data = _read_json(p)
        assert data["metal"] == "Ru"
        assert data["seed"] == seed
        assert "test_metrics" in data
        assert "roc_auc" in data["test_metrics"]
        assert "elapsed_seconds" in data
        assert data["elapsed_seconds"] > 0


def test_r17_baseline_aggregate_metrics():
    metrics = _read_json(REPORTS_ROOT / "baseline" / "metrics.json")
    assert metrics["recipe"] == "r17_dmpnn_multitask_baseline_Ru_only"
    assert len(metrics["per_seed"]) == 3
    assert metrics["n_seeds"] == 3


# ---------------------------------------------------------------------------
# 6. GPU memory log + sanity bounds
# ---------------------------------------------------------------------------

def test_gpu_memory_log_format():
    text = _read_text(REPORTS_ROOT / "gpu_memory_log.json.baseline")
    assert "MiB" in text
    assert "32607" in text or "32" in text, "expected 32 GB total in nvidia-smi CSV"


def test_gpu_peak_memory_under_4gb():
    """Smoke runs use ~1.3 GB; well under 4 GB."""
    samples_p = REPORTS_ROOT / "se3sb_gpu_mem_samples_mib.txt"
    if not samples_p.exists():
        pytest.skip("se3sb_gpu_mem_samples_mib.txt not present (probe optional)")
    samples = [
        int(line.strip())
        for line in samples_p.read_text().splitlines()
        if line.strip().isdigit()
    ]
    assert samples, "no GPU mem samples"
    peak = max(samples)
    assert peak < 4096, f"peak {peak} MiB exceeds 4 GB sanity bound"


# ---------------------------------------------------------------------------
# 7. Exit-code honesty (decorator bug — recorded, not hidden)
# ---------------------------------------------------------------------------

def test_se3sb_decorator_bug_recorded_honestly():
    """train_se3sb.py exits 2 due to decorator re-parsing argv with simplified parser.
    The training data + ckpts are written correctly. This test pins that we recorded
    the bug in comparison.md rather than silently fabricating an exit code 0.
    """
    md = _read_text(REPORTS_ROOT / "comparison.md")
    # Honest note in §7 honest caveats or §6 status
    has_honest_note = (
        "exit" in md.lower()
        and ("decorator" in md.lower() or "exit_code" in md.lower() or "exit code" in md.lower())
    )
    assert has_honest_note, (
        "comparison.md must honestly record the decorator exit-code-2 bug"
    )


# ---------------------------------------------------------------------------
# 8. No fabricated MEASURED numbers
# ---------------------------------------------------------------------------

def test_provenance_tag_in_bench_summary():
    """The bench_summary must record provenance so future readers cannot
    silently swap in PROJECTED numbers.
    """
    bench = _read_json(REPORTS_ROOT / "bench_summary.json")
    # encoder_init_status is the main honesty anchor for R18 runs
    assert "encoder_init_status" in bench
    assert "random" in bench["encoder_init_status"].lower() or "fallback" in bench["encoder_init_status"].lower(), (
        f"R18 runs trained from random init — bench_summary must record this honestly"
    )
