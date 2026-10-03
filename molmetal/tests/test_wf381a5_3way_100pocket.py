"""Tests for Wf-381-A5 TODO-95 — R20-P4 3-way + 100-pocket × 10-seed CPU SCAFFOLD.

====================================================================
Wf-381-A5 (2026-09-30) — 1 NEW regression test pinning the
:mod:`molmetal.scripts.wf381a5_r20_plan_cpu_close` TODO-95 item to
the spec:

  1. test_todo95_3way_100pocket_scaffold — both 3-way real
     metalcytotox bench + 100-pocket × 10-seed sweep exist on disk
     AND honest-negative framing is preserved verbatim.

The test is **CPU-only** — no 3-way joint retrain is launched (GPU
BLOCKED on user OK per AGENTS.md §5.1) and no Vina docking is run.
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
# Test 2 — TODO-95 3-way + 100-pocket × 10-seed CPU SCAFFOLD pins.
# ---------------------------------------------------------------------------
def test_todo95_3way_100pocket_scaffold() -> None:
    """TODO-95 R20-P4 3-way + 100-pocket × 10-seed bench SHIPS at CPU layer.

    Per Wf-381-A5 verification 2026-09-30:
      * ``bench_3way_real_metalcytotox.py`` is the 3-way joint bench
        script (Wf-41 P1 ship).
      * ``bench_100pocket_10seed.py`` is the 100-pocket × 10-seed
        docking sweep script (Wf-67 P3 ship).
      * Honest framing: 3-way R²=-0.3595 ± 0.2806 (PlatinAI 5K
        synthetic) — CPU bench only; GPU joint retrain BLOCKED on
        user OK; honest stub fallback for Vina (Wf-63).
    """
    scaffold = _import_scaffold()

    # TODO-95 item is present in the 4-item aggregate.
    todo_ids = [item["todo_id"] for item in scaffold.R20_PLAN_ITEMS]
    assert "TODO-95" in todo_ids, todo_ids

    # The single TODO-95 dict exposes both script fields.
    todo95 = next(
        item for item in scaffold.R20_PLAN_ITEMS if item["todo_id"] == "TODO-95"
    )
    assert "script_3way" in todo95
    assert "script_100pocket" in todo95
    assert todo95["script_3way"] == "molmetal/scripts/bench_3way_real_metalcytotox.py"
    assert todo95["script_100pocket"] == "molmetal/scripts/bench_100pocket_10seed.py"

    # Both bench scripts are on disk (verified at Wf-381-A5 timestamp).
    for script_path in (todo95["script_3way"], todo95["script_100pocket"]):
        on_disk = (PROJECT_ROOT / script_path).exists()
        assert on_disk, f"{script_path} missing on disk"

    # Skill alignment.
    assert todo95["skill"] == "mol-metal-3way-multimetal"

    # Honest framing verbatim — must include both 3-way R² baseline
    # and 100-pocket × 10-seed Wf-63 honest-stub fallback.
    framing = todo95["honest_framing"]
    assert "3way R" in framing
    assert "0.3595" in framing
    assert "Wf-41 P1" in framing
    assert "100-pocket" in framing
    assert "Wf-63" in framing  # honest stub fallback reference

    # The CpuClosureScaffold dataclass coerces the dict correctly:
    #   - gate_target picks up `gate_target_3way` (0.30)
    #   - script is the comma-joined list of script_* fields
    #   - existing_partial is True (both scripts on disk)
    scaffolds = scaffold.list_scaffolds()
    todo95_scaffold = next(s for s in scaffolds if s.todo_id == "TODO-95")
    assert todo95_scaffold.existing_partial is True
    assert todo95_scaffold.gate_target == pytest.approx(0.30, abs=1e-9)
    assert "bench_100pocket_10seed.py" in todo95_scaffold.script
    assert "bench_3way_real_metalcytotox.py" in todo95_scaffold.script
    assert "HONEST_NEGATIVE" in todo95_scaffold.me_measured_status

    # Self-test reports both scripts on disk for TODO-95.
    verdict = scaffold.run_self_test()
    todo95_status = next(
        s for s in verdict["items_status"] if s["todo_id"] == "TODO-95"
    )
    assert todo95_status["all_scripts_on_disk"] is True
    assert todo95_status["n_scripts"] == 2
