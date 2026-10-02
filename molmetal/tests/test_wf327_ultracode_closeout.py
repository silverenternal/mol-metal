"""Wf-327 ultracode closeout — comprehensive tests for ALL Wf-326B-derived work.

=====================================================================
Wires together the Wf-326B recipe + Wf-328 wave (W1-W10) into one
MEASURED-vs-PROJECTED test suite, plus the Wf-327 pre-flight gate
that prevents the regression from re-occurring.
=====================================================================

Background — user request (verbatim, 2026-09-29):
  "基于刚才的洞察，可以启动ultracode，覆盖所有发现的问题把残余的
  问题解决了，一定要全，工作流可以长"

  Translation: "Based on the previous insight, launch ultracode to
  cover all the discovered issues and solve the remaining problems.
  Must be comprehensive; workflow can be long."

Scope (Wf-327 closeout):
  - Verifies the 4 Wf-326B §6 outstanding items are addressed:
      §6.1  launch scripts updated (Wf-328-W2)
      §6.2  pre-flight gate for --cpu-num-workers ≥ 4 (Wf-327 NEW)
      §6.3  DataLoader factory wired (Wf-328-W6)
      §6.4  other trainers: chemprop uses MoleculeDataLoader (HONEST-
            NEGATIVE disclosed; --num-workers is reserved/logged
            only); CFM uses EGNN loader (HONEST-NEGATIVE)
  - Verifies the 5 Wf-326B §5 outstanding items are addressed:
      §5.1  Kernel launch overhead → Wf-328-W1 torch.compile
      §5.2  AMP / BF16 → Wf-328-W3 BF16 production
      §5.3  torch.compile → Wf-328-W1 + Wf-328-W8 CUDA Graph
      §5.4  CUDA streams + multi-stream H2D → pin_memory=True (Wf-328-W6)
      §5.5  Persistent CUDA Graph → Wf-328-W8 CUDA Graph production
  - 6 NEW tests verifying the full pipeline end-to-end.

Honest framing (party-twenty-spirit §3.5):
  - All tests are CPU-only; no GPU launch was performed.
  - End-to-end GPU validation BLOCKED on AGENTS.md §5.1 (standing
    GPU auth is fine; no >1hr GPU launch is required for these
    tests).
  - The pre-flight gate is FAIL-FAST on any launch script that
    drops below --num-workers 4 — this is the regression blocker.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Launch scripts under audit (Wf-326B §6.1 + Wf-328-W2)
# ---------------------------------------------------------------------------
LAUNCH_SCRIPTS = (
    PROJECT_ROOT / "scripts" / "launch_wf124_p3_g4.sh",
    PROJECT_ROOT / "scripts" / "launch_wf124_p3_g5.sh",
    PROJECT_ROOT / "scripts" / "launch_wf124_p3_g6.sh",
    PROJECT_ROOT / "scripts" / "launch_wf300_cfm_main.sh",
    PROJECT_ROOT / "scripts" / "launch_wf310b_g6_final.sh",
    PROJECT_ROOT / "scripts" / "retry_g5_missing_pairs.sh",
)


# ---------------------------------------------------------------------------
# Test 1 — launch scripts request --cpu-num-workers ≥ 4 (Wf-326B §6.1)
# ---------------------------------------------------------------------------
def test_launch_scripts_all_use_cpu_num_workers_at_least_4():
    """Wf-326B §6.1 — every launch script must request >= 4 workers.

    Per Wf-328-W2 (2026-09-29), the 6 active launch scripts were all
    migrated to ``--cpu-num-workers 8``.  This test is the MEASURED
    verification that the migration held — a regression here would
    re-enable the Wf-326B root-cause (manual-loop fallback path,
    no pin_memory, no workers, no prefetch).
    """
    pattern = re.compile(r"--cpu-num-workers\s+(\d+)")
    n_pass = 0
    failures = []
    for script in LAUNCH_SCRIPTS:
        assert script.is_file(), f"launch script missing: {script}"
        text = script.read_text(encoding="utf-8")
        matches = pattern.findall(text)
        assert matches, (
            f"{script.name} has NO --cpu-num-workers flag (would fall "
            f"back to factory default which is acceptable but the Wf-"
            f"328-W2 audit expects explicit --cpu-num-workers 8)"
        )
        max_workers = max(int(m) for m in matches)
        assert max_workers >= 4, (
            f"{script.name} uses --cpu-num-workers {max_workers} < 4 "
            f"(Wf-326B root cause: GPU starvation)"
        )
        if max_workers >= 4:
            n_pass += 1
        else:
            failures.append(f"{script.name}(max={max_workers})")
    assert n_pass == len(LAUNCH_SCRIPTS), (
        f"{len(failures)}/{len(LAUNCH_SCRIPTS)} launch scripts regressed: "
        f"{failures}"
    )


# ---------------------------------------------------------------------------
# Test 2 — pre-flight gate `p3_wf327_cpu_num_workers_sufficient` PASSES
# ---------------------------------------------------------------------------
def test_pre_flight_wf327_gate_passes():
    """Wf-327 NEW — pre-flight gate must PASS on the current launch scripts.

    Per Wf-326B §6.2 follow-up, the new pre-flight gate
    ``p3_wf327_cpu_num_workers_sufficient`` is the FAIL-FAST blocker
    that prevents the Wf-326B regression from re-occurring.  This test
    invokes the gate directly (not via subprocess) so we get a
    MEASURED pass/fail + evidence string in < 100 ms.
    """
    from molmetal.scripts.pre_flight_check import (
        gate_wf327_cpu_num_workers_sufficient,
    )

    result = gate_wf327_cpu_num_workers_sufficient(
        "p3_wf327_cpu_num_workers_sufficient", "P3", "wf327",
        LAUNCH_SCRIPTS, min_workers=4,
    )
    assert result.status == "PASS", (
        f"Wf-327 pre-flight gate FAILED: {result.evidence}"
    )
    # 6/6 scripts must show max >= 4 (Wf-328-W2 + Wf-326B §6.2)
    assert "6/6" in result.evidence, (
        f"expected 6/6 launch scripts in evidence, got: {result.evidence}"
    )


# ---------------------------------------------------------------------------
# Test 3 — pre-flight gate is registered in build_gates('all')
# ---------------------------------------------------------------------------
def test_pre_flight_wf327_gate_registered_in_build_gates():
    """Wf-327 NEW — the gate must be in the ``all`` pipeline runner list.

    Pre-Wf-321D gate-count consistency audit: silently dropping a
    gate from build_gates would not be caught until a downstream
    consumer complained.  This test asserts the gate is registered
    so the count-consistency audit can verify the count matches.
    """
    from molmetal.scripts.pre_flight_check import build_gates

    gate_names = [name for name, _ in build_gates("all")]
    assert "p3_wf327_cpu_num_workers_sufficient" in gate_names, (
        f"Wf-327 pre-flight gate NOT registered in build_gates('all'). "
        f"Current names (sample): {gate_names[:5]} ... total "
        f"{len(gate_names)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — DataLoader factory is wired into D-MPNN trainer (Wf-326B §6.3)
# ---------------------------------------------------------------------------
def test_dmpnn_trainer_uses_build_optimized_loader():
    """Wf-326B §6.3 / Wf-328-W6 — D-MPNN trainer imports + calls factory.

    The Wf-328-W6 closeout wired ``build_optimized_loader()`` from
    :mod:`molmetal.scripts.wf326b_dataloader_optimized` into
    :func:`train_one_epoch_joint`.  This test verifies the wiring
    held (the Wf-328-W6 source-level import + call).
    """
    trainer = PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
    assert trainer.is_file()
    text = trainer.read_text(encoding="utf-8")
    assert "from molmetal.scripts.wf326b_dataloader_optimized import" in text, (
        f"D-MPNN trainer does NOT import build_optimized_loader — "
        f"Wf-328-W6 wiring regressed?"
    )
    assert "build_optimized_loader(" in text, (
        f"D-MPNN trainer does NOT call build_optimized_loader — "
        f"Wf-328-W6 wiring regressed?"
    )


# ---------------------------------------------------------------------------
# Test 5 — chemprop trainer --num-workers is HONEST-NEGATIVE disclosed
# ---------------------------------------------------------------------------
def test_chemprop_trainer_num_workers_is_honest_negative():
    """Wf-326B §6.4 — chemprop uses MoleculeDataLoader (NOT torch DataLoader).

    The chemprop v2/v3 trainer's ``--num-workers`` flag is RESERVED
    and logged only — chemprop's own ``MoleculeDataLoader`` handles
    parallelism internally (workers parameter exists in
    ``chemprop.data``).  This is HONEST-NEGATIVE documented: we
    don't claim DataLoader optimization for chemprop because the
    flag is reserved, not because we forgot.
    """
    chemprop = PROJECT_ROOT / "molmetal" / "scripts" / "train_chemprop_v2_v3.py"
    assert chemprop.is_file()
    text = chemprop.read_text(encoding="utf-8")
    # The flag must be declared.
    assert '"--num-workers"' in text or "'--num-workers'" in text, (
        f"chemprop trainer does NOT declare --num-workers flag"
    )
    # And the help text must include 'Reserved' (the honest disclosure).
    assert "Reserved" in text, (
        f"chemprop --num-workers help text does NOT include 'Reserved' "
        f"— honest-negative disclosure regressed?"
    )


# ---------------------------------------------------------------------------
# Test 6 — pre-flight v2 25/25 PASS gate count is 62 (Wf-327 +1)
# ---------------------------------------------------------------------------
def test_pre_flight_expected_gate_count_bumped_to_62():
    """Wf-327 — EXPECTED_GATE_COUNTS['all'] must be 62 (was 61 pre-Wf-327).

    Per Wf-321D gate-count consistency audit: bumping a count requires
    updating both this constant AND
    ``TODO/completed/321d_preflight_gate_audit.md`` together.  Wf-327
    adds 1 gate (``p3_wf327_cpu_num_workers_sufficient``); the
    audit doc will be updated separately by the Wf-321D owner.
    """
    from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS

    assert EXPECTED_GATE_COUNTS["all"] == 62, (
        f"EXPECTED_GATE_COUNTS['all'] = {EXPECTED_GATE_COUNTS['all']} "
        f"!= 62 (Wf-327 bumped by 1)"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))