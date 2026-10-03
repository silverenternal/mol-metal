"""Tests for Wf-381-A5 TODO-98 — Closure OOD cross-domain CPU SCAFFOLD.

====================================================================
Wf-381-A5 (2026-09-30) — 1 NEW regression test pinning the
:mod:`molmetal.scripts.wf381a5_r20_plan_cpu_close` TODO-98 item to
the spec:

  1. test_todo98_closure_ood_scaffold — both closure OOD cross-
     domain bench + closure OOD 30-metal bench exist on disk AND
     the per-domain reachability honesty-gate is preserved verbatim
     per Wf-41 P2 + Wf-130 P3.

The test is **CPU-only** — no GPU cross-domain retrain is launched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "molmetal" / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _import_scaffold():
    try:
        from molmetal.scripts import wf381a5_r20_plan_cpu_close as scaffold
        return scaffold
    except Exception as exc:  # pragma: no cover
        pytest.skip(
            f"wf381a5_r20_plan_cpu_close not importable: {exc}",
            allow_module_level=False,
        )


# ---------------------------------------------------------------------------
# Test 4 — TODO-98 Closure OOD cross-domain CPU SCAFFOLD pins.
# ---------------------------------------------------------------------------
def test_todo98_closure_ood_scaffold() -> None:
    """TODO-98 Wf-44 P2 / L12 Closure OOD cross-domain CPU SCAFFOLD pins.

    Per Wf-381-A5 verification 2026-09-30:
      * ``bench_closure_ood.py`` (Wf-41 P2 ship — PlatinAI 5K OOD
        reachability 0.5581 + MetalCytoToxDB OOD 0.0300).
      * ``bench_closure_ood_30metals.py`` (Wf-130 P3 ship — 30-metal
        tmQM d-block OOD reachability 0.5891).
      * Per-domain reachability honesty-gate 0.30 documented as
        partial lift (1/3 domains).
    """
    scaffold = _import_scaffold()

    # TODO-98 item is present in the 4-item aggregate.
    todo_ids = [item["todo_id"] for item in scaffold.R20_PLAN_ITEMS]
    assert "TODO-98" in todo_ids

    # Two script fields exposed.
    todo98 = next(
        item for item in scaffold.R20_PLAN_ITEMS if item["todo_id"] == "TODO-98"
    )
    for key in ("script_ood", "script_ood_30metals"):
        assert key in todo98, f"TODO-98 missing key {key}"
    assert todo98["script_ood"] == "molmetal/scripts/bench_closure_ood.py"
    assert todo98["script_ood_30metals"] == "molmetal/scripts/bench_closure_ood_30metals.py"

    # Both bench scripts are on disk.
    for script_path in (todo98["script_ood"], todo98["script_ood_30metals"]):
        on_disk = (PROJECT_ROOT / script_path).exists()
        assert on_disk, f"{script_path} missing on disk"

    # Skill alignment.
    assert todo98["skill"] == "mol-metal-closure-catherine"

    # Honest framing verbatim — must include the three per-domain
    # reachability values + AGENTS.md §5.1 GPU grant reference.
    framing = todo98["honest_framing"]
    assert "PlatinAI 5K" in framing
    assert "0.5581" in framing
    assert "MetalCytoToxDB" in framing
    assert "0.0300" in framing
    assert "tmQM" in framing
    assert "0.5891" in framing
    assert "AGENTS.md" in framing  # §5.1 GPU grant reference

    # me_measured_status preserves the "1/3 domains lift" honest-negative
    # framing on the 3-domain aggregation axis.
    status_str = todo98["me_measured_status"]
    assert "HONEST_NEGATIVE_PARTIAL" in status_str
    assert "1/3 domains lift" in status_str

    # CpuClosureScaffold picks gate_target = 0.30 (per-domain).
    scaffolds = scaffold.list_scaffolds()
    todo98_scaffold = next(s for s in scaffolds if s.todo_id == "TODO-98")
    assert todo98_scaffold.existing_partial is True
    assert todo98_scaffold.gate_target == pytest.approx(0.30, abs=1e-9)
    assert "HONEST_NEGATIVE_PARTIAL" in todo98_scaffold.me_measured_status
    # Both scripts in comma-joined script field.
    assert "bench_closure_ood.py" in todo98_scaffold.script
    assert "bench_closure_ood_30metals.py" in todo98_scaffold.script

    # Self-test reports both scripts on disk for TODO-98.
    verdict = scaffold.run_self_test()
    todo98_status = next(
        s for s in verdict["items_status"] if s["todo_id"] == "TODO-98"
    )
    assert todo98_status["all_scripts_on_disk"] is True
    assert todo98_status["n_scripts"] == 2

    # The closure OOD module exposes the per-domain stratification.
    from molmetal.scripts import bench_closure_ood as ood
    # Per-domain defaults: PlatinAI 5K + MetalCytoToxDB 5-metal.
    assert "Pt" in ood.PLATINAI_METALS
    assert "Ru" in ood.METALCYTO_METALS
    # Common metals intersection.
    assert "Ru" in ood.COMMON_METALS
    assert "Ir" in ood.COMMON_METALS
    # tmQM 30-metal d-block cohort.
    assert len(ood.TMQM_METALS) == 30

    # The 30-metals OOD module exposes the same 30-metal cohort.
    from molmetal.scripts import bench_closure_ood_30metals as ood_30
    # Public functions exist (bench loop + metal selection).
    assert hasattr(ood_30, "_benchmark_one")
    assert hasattr(ood_30, "_filter_ood_subset")
    assert hasattr(ood_30, "_aggregate_per_metal")
