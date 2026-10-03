"""Tests for Wf-381-A5 TODO-93 — R20-P2 Hilbert ASTSR end-to-end CPU SCAFFOLD.

====================================================================
Wf-381-A5 (2026-09-30) — 1 NEW regression test pinning the
:mod:`molmetal.scripts.hilbert_astsr_e2e` CPU SCAFFOLD to the spec:

  1. test_hilbert_r2_gate_status — R20-P2 honesty-gate verdict DOC is
     PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL with verbatim
     Wf-365-A3 wording.

The test is **CPU-only** and uses the real ``hilbert_astsr_e2e.py``
module (no GPU retrain is performed).  All values trace to a single
DOC: ``TODO/completed/365a3_m6_kappa_verdict.md`` (Wf-365-A3).
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


# Lazy import so a missing dep only fails the relevant test.
def _import_scaffold():
    try:
        from molmetal.scripts import hilbert_astsr_e2e as scaffold
        return scaffold
    except Exception as exc:  # pragma: no cover
        pytest.skip(
            f"hilbert_astsr_e2e module not importable: {exc}",
            allow_module_level=False,
        )


# ---------------------------------------------------------------------------
# Test 1 — R20-P2 honesty gate verdict DOC is PRESERVED (CPU-only).
# ---------------------------------------------------------------------------
def test_hilbert_r2_gate_status() -> None:
    """The R20-P2 Hilbert ASTSR honesty gate is HONEST_NEGATIVE_PARTIAL_STRUCTURAL.

    Per Wf-365-A3 verdict DOC (the latest closed-out Wf in the Wf-381
    lineage):
      * best CV-R² +0.6814 CEILING_BREAK on 60-SMILES smoke cohort
        (structural ceiling LIFTED vs prior baseline +0.5410)
      * augmented_mean 0.4386 < 0.6 gate (threshold-reaching axis
        NOT lifted) → gate PRESERVED
    """
    scaffold = _import_scaffold()

    status = scaffold.hilbert_r2_gate_status()

    # Gate constants are stable across Wf-365-A3 + Wf-381-A5.
    assert status.gate_target == pytest.approx(0.6, abs=1e-9)
    assert status.r2_target == pytest.approx(0.85, abs=1e-9)

    # MEASURED-vs-PROJECTED discipline — only the gate-target constants
    # are PROJECTED; the latest-MEASURED values are hard-coded from the
    # Wf-365-A3 verdict DOC.
    assert status.best_cv_r2 == pytest.approx(0.6814, abs=1e-4)
    assert status.best_augmented_mean == pytest.approx(0.4386, abs=1e-4)

    # Structural ceiling LIFTED relative to Wf-335-M1 prior baseline
    # of +0.5410 (the Wf-365-A3 "CEILING_BREAK" marker).
    assert status.structural_ceiling_lifted is True

    # Threshold-reaching axis NOT lifted — augmented_mean 0.4386 < 0.6.
    assert status.threshold_axis_lifted is False

    # Verdict DOC verbatim — must include Wf-365-A3 + the 5-attempt list.
    assert status.gate_status == "HONEST_NEGATIVE_PARTIAL_STRUCTURAL"
    assert "Wf-365-A3 verdict DOC" in status.verdict
    assert "Wf-335-M1 chemprop+PLL" in status.verdict
    assert "Wf-359-A1 W" in status.verdict
    assert "Wf-361-B1 MAML" in status.verdict
    assert "Wf-361-B2 Neural ODE" in status.verdict
    assert "Wf-364-A2 combined" in status.verdict
    assert "HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL" in status.verdict

    # AGENTS.md §5.1 GPU grant caveat is preserved verbatim.
    assert "AGENTS.md §5.1" in status.verdict

    # Round-trip through JSON serialization (used by downstream
    # INDEX/AGENTS syncs).
    import dataclasses
    import json as json_mod
    payload = json_mod.dumps(dataclasses.asdict(status), default=str)
    rebuilt = json_mod.loads(payload)
    assert rebuilt["gate_status"] == status.gate_status
    assert rebuilt["best_cv_r2"] == pytest.approx(status.best_cv_r2)
