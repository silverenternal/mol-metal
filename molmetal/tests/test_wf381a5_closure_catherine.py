"""Tests for Wf-381-A5 TODO-96 — R20-P5 Closure + Catherine 30-metal CPU SCAFFOLD.

====================================================================
Wf-381-A5 (2026-09-30) — 1 NEW regression test pinning the
:mod:`molmetal.scripts.wf381a5_r20_plan_cpu_close` TODO-96 item to
the spec:

  1. test_todo96_closure_catherine_scaffold — three bench scripts
     (Catherine 30-metal + closure reachability R20 + closure
     d-block R20) exist on disk AND the 0.95 honesty gate is
     PRESERVED HONEST_NEGATIVE_STRUCTURAL per Wf-68 §Gap.

The test is **CPU-only** — no GPU closure-theorem retrain is
launched.  The Catherine 6/6 PASS verdict at CPU bench layer is
preserved per Wf-57 Phase 2 + Wf-130 P3.
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
# Test 3 — TODO-96 R20-P5 Closure + Catherine CPU SCAFFOLD pins.
# ---------------------------------------------------------------------------
def test_todo96_closure_catherine_scaffold() -> None:
    """TODO-96 R20-P5 Closure + Catherine 30-metal CPU SCAFFOLD pins.

    Per Wf-381-A5 verification 2026-09-30:
      * ``bench_catherine_30metals.py`` (TODO-96 ship target script
        per the AGENTS.md §3.3 R20-P5 skill registration).
      * ``bench_closure_reachability_r20.py`` (Wf-68 Phase 2 ship).
      * ``bench_closure_dblock_r20.py`` (Wf-37 P2 + Wf-67 P2 ships).

    Honest framing per Wf-68 Phase 2 §Gap:
      * Catherine 6/6 PASS on 25/25 metals × 1000 samples at CPU
        bench layer (Wf-57).
      * Reachability gate ≥ 0.95 on 25 metals × 1000 samples
        documented as NOT achievable without changing closure-
        theorem semantics.
      * Wf-67 P2 baseline reachability 0.5891 — gate preserved
        HONEST_NEGATIVE_STRUCTURAL.
    """
    scaffold = _import_scaffold()

    # TODO-96 item is present in the 4-item aggregate.
    todo_ids = [item["todo_id"] for item in scaffold.R20_PLAN_ITEMS]
    assert "TODO-96" in todo_ids

    # Three script fields exposed.
    todo96 = next(
        item for item in scaffold.R20_PLAN_ITEMS if item["todo_id"] == "TODO-96"
    )
    for key in ("script_30metals", "script_reachability", "script_dblock"):
        assert key in todo96, f"TODO-96 missing key {key}"
    assert todo96["script_30metals"] == "molmetal/scripts/bench_catherine_30metals.py"
    assert todo96["script_reachability"] == "molmetal/scripts/bench_closure_reachability_r20.py"
    assert todo96["script_dblock"] == "molmetal/scripts/bench_closure_dblock_r20.py"

    # All three bench scripts are on disk.
    for script_path in (
        todo96["script_30metals"],
        todo96["script_reachability"],
        todo96["script_dblock"],
    ):
        on_disk = (PROJECT_ROOT / script_path).exists()
        assert on_disk, f"{script_path} missing on disk"

    # Skill alignment.
    assert todo96["skill"] == "mol-metal-closure-catherine"

    # Honest framing verbatim — must include the 0.95 gate target,
    # the Wf-67 P2 baseline 0.5891, and the Wf-68 §Gap reasoning.
    framing = todo96["honest_framing"]
    assert "0.95" in framing
    assert "0.5891" in framing
    assert "Wf-67 P2" in framing
    assert "Wf-68" in framing
    assert "Catherine 6/6 PASS" in framing
    assert "HONEST_NEGATIVE_STRUCTURAL" in framing

    # CpuClosureScaffold picks gate_target = 0.95.
    scaffolds = scaffold.list_scaffolds()
    todo96_scaffold = next(s for s in scaffolds if s.todo_id == "TODO-96")
    assert todo96_scaffold.existing_partial is True
    assert todo96_scaffold.gate_target == pytest.approx(0.95, abs=1e-9)
    assert todo96_scaffold.me_measured_status == "HONEST_NEGATIVE_STRUCTURAL"
    # All three scripts in comma-joined script field.
    assert "bench_catherine_30metals.py" in todo96_scaffold.script
    assert "bench_closure_reachability_r20.py" in todo96_scaffold.script
    assert "bench_closure_dblock_r20.py" in todo96_scaffold.script

    # Self-test reports all 3 scripts on disk for TODO-96.
    verdict = scaffold.run_self_test()
    todo96_status = next(
        s for s in verdict["items_status"] if s["todo_id"] == "TODO-96"
    )
    assert todo96_status["all_scripts_on_disk"] is True
    assert todo96_status["n_scripts"] == 3

    # The reachability R20 module exposes the canonical 0.95 gate target.
    from molmetal.scripts import bench_closure_reachability_r20 as reach
    assert reach.GATE_TARGET_REACHABILITY == pytest.approx(0.95, abs=1e-9)
    assert reach.DEFAULT_N_METALS == 25
    assert reach.DEFAULT_N_SAMPLES == 1000

    # The Catherine 30-metals module exposes the 30-metal cohort.
    from molmetal.scripts import bench_catherine_30metals as cath
    assert len(cath.DEFAULT_30_METALS) == 30
