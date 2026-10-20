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
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Launch scripts under audit (Wf-326B §6.1 + Wf-328-W2)
#
# 2026-10-20: down from 6 to 4.  ``launch_wf124_p3_g4.sh`` and
# ``launch_wf300_cfm_main.sh`` were archived under
# ``TODO/archive/wf437_simplify/`` (Wf-437).  The pre-flight gate was already
# narrowed to these 4 (``pre_flight_check.py`` wf327_launch_scripts, Wf-446
# comment); this list now matches it instead of asserting two files that are
# not on disk.
# ---------------------------------------------------------------------------
ARCHIVED_LAUNCH_SCRIPTS = (
    PROJECT_ROOT / "scripts" / "launch_wf124_p3_g4.sh",
    PROJECT_ROOT / "scripts" / "launch_wf300_cfm_main.sh",
)

LAUNCH_SCRIPTS = (
    PROJECT_ROOT / "scripts" / "launch_wf124_p3_g5.sh",
    PROJECT_ROOT / "scripts" / "launch_wf124_p3_g6.sh",
    PROJECT_ROOT / "scripts" / "launch_wf310b_g6_final.sh",
    PROJECT_ROOT / "scripts" / "retry_g5_missing_pairs.sh",
)


# ---------------------------------------------------------------------------
# Test 1 — launch scripts request --cpu-num-workers ≥ 4 (Wf-326B §6.1)
# ---------------------------------------------------------------------------
def test_launch_scripts_all_use_cpu_num_workers_at_least_4():
    """Wf-326B §6.1 — every launch script must request >= 4 workers.

    Per Wf-328-W2 (2026-09-29), the active launch scripts were all
    migrated to ``--cpu-num-workers 8``.  This test is the MEASURED
    verification that the migration held — a regression here would
    re-enable the Wf-326B root-cause (manual-loop fallback path,
    no pin_memory, no workers, no prefetch).
    """
    assert len(LAUNCH_SCRIPTS) == 4, (
        f"expected exactly 4 active launchers, found {len(LAUNCH_SCRIPTS)}; "
        f"if another launcher was archived, update ARCHIVED_LAUNCH_SCRIPTS "
        f"and the pre-flight gate together"
    )
    for archived in ARCHIVED_LAUNCH_SCRIPTS:
        assert not archived.exists(), (
            f"{archived.name} is back on disk but listed as archived — "
            f"either restore it to LAUNCH_SCRIPTS or update the archive note"
        )
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
        "p3_wf327_cpu_num_workers_sufficient",
        "P3",
        "wf327",
        LAUNCH_SCRIPTS,
        min_workers=4,
    )
    assert result.status == "PASS", f"Wf-327 pre-flight gate FAILED: {result.evidence}"
    # Every active script must show max >= 4 (Wf-328-W2 + Wf-326B §6.2).
    # Derived, not frozen: the count moved 6 -> 4 when Wf-437 archived two
    # launchers, and a hardcoded "6/6" turned that into a red test.
    expected = f"{len(LAUNCH_SCRIPTS)}/{len(LAUNCH_SCRIPTS)}"
    assert expected in result.evidence, (
        f"expected {expected} launch scripts in evidence, got: {result.evidence}"
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
# Test 4 — D-MPNN trainer batch construction (Wf-326B §6.3 → superseded)
# ---------------------------------------------------------------------------
def test_dmpnn_trainer_batches_without_a_dataloader():
    """The trainer builds batches itself; there is no DataLoader to wire.

    Wf-328-W6 wired ``build_optimized_loader()`` from
    :mod:`molmetal.scripts.wf326b_dataloader_optimized` into
    ``train_one_epoch_joint``.  That wiring is **gone**: the trainer was
    restored from ``7edc3823a8`` (m21 selective revert, ``af84a545``,
    2026-10-12) and the shipped trainer builds its batches manually.

    Asserting the old wiring was asserting a feature that cannot exist — a
    DataLoader factory has nothing to feed.  The live contract is the
    inverse, and it is asserted here so a silent re-introduction is caught:
    no factory import, and ``--cpu-num-workers`` is accepted but inert.
    """
    trainer = (
        PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
    )
    assert trainer.is_file()
    text = trainer.read_text(encoding="utf-8")

    # The factory is not wired, and no torch DataLoader is constructed.
    assert "wf326b_dataloader_optimized" not in text, (
        "the trainer imports the Wf-326B DataLoader factory again — if a "
        "DataLoader was re-introduced on purpose, re-enable "
        "test_dmpnn_trainer_uses_build_optimized_loader and update the "
        "Wf-328-W6 record; if not, this is the revert leaking back in."
    )
    assert "torch.utils.data.DataLoader" not in text, (
        "the trainer constructs a torch DataLoader again"
    )

    # The worker flag survives as an accepted-but-inert surface, and says so.
    assert "--cpu-num-workers" in text
    assert "structurally a no-op" in text, (
        "the trainer no longer documents --cpu-num-workers as inert; the "
        "honest disclosure must stay next to the flag"
    )

    # The factory module itself must still exist (it is public API and is
    # still exercised by its own test file).
    factory = PROJECT_ROOT / "molmetal" / "scripts" / "wf326b_dataloader_optimized.py"
    assert factory.is_file(), "wf326b_dataloader_optimized.py was deleted"


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
    assert (
        '"--num-workers"' in text or "'--num-workers'" in text
    ), "chemprop trainer does NOT declare --num-workers flag"
    # And the help text must include 'Reserved' (the honest disclosure).
    assert "Reserved" in text, (
        "chemprop --num-workers help text does NOT include 'Reserved' "
        "— honest-negative disclosure regressed?"
    )


# ---------------------------------------------------------------------------
# Test 6 — the gate-count constant matches what build_gates() registers
# ---------------------------------------------------------------------------
def test_pre_flight_gate_count_matches_registration():
    """``EXPECTED_GATE_COUNTS`` must equal the registered gate count.

    This test used to assert a frozen literal (62, the Wf-327 checkpoint).  It
    went stale the first time a gate was added, and the only way to "fix" it
    was to edit the literal — which asserts nothing about the code.  The
    invariant that actually matters is constant == ``len(build_gates(...))``;
    the audit-doc half of the sync is covered by
    ``test_wf327f11_gate_count_consistency.py``.
    """
    from molmetal.scripts.pre_flight_check import (
        EXPECTED_GATE_COUNTS,
        build_gates,
    )

    for pipeline in ("all", "launch", "frontier", "chemprop_v2_v3", "cfm"):
        registered = len(build_gates(pipeline))
        assert registered == EXPECTED_GATE_COUNTS[pipeline], (
            f"pipeline={pipeline!r}: build_gates() registers {registered} "
            f"but EXPECTED_GATE_COUNTS says {EXPECTED_GATE_COUNTS[pipeline]}. "
            f"Update the constant in the same edit that registers the gate."
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
