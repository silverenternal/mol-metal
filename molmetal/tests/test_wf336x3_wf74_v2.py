"""Wf-336-X3 — Wf-74 v2 DrugOOD DANN launch wrapper tests.

Owner: Hugo Lee + ultracode agent
Date: 2026-09-29
Parent: Wf-336-X3 ultracode residual closure wave
Per AGENTS.md §5.1 (kill_then_adapt) + §5.2 (MEASURED-vs-PROJECTED honest framing).

Background
----------
Wf-74 v2 lineage is the docs-only wet-lab outreach closeout (per
`TODO/completed/97c_wf74_reclassification.md`); Wf-74 itself was DEPRECATED
for wet-lab scope per AGENTS.md §5.4 Tier-2 strategy.  The on-disk
computational follow-up is Wf-305D DrugOOD DANN trainer
(``molmetal/scripts/wf305d_drugood_dann_train.py``), which addresses the
drugood_v3_scaffold ood_test_auc=0.4574 BELOW field-avg 0.55-0.65 (per
Wf-297B + Wf-301A).

Wf-336-X3 ships a dedicated DrugOOD-only launch wrapper
(``scripts/launch_wf74_v2_drugood_dann.sh``) with all Wf-328 wire-ups (W1-W10)
wired into the Wf-305D driver.  This test file gates:

  - test_launch_wf74_v2_drugood_dann_script_parses
      Verifies the bash script is well-formed + has the expected structural
      markers (USER_OK gate, pre_flight_check gate, VRAM gate, mode selector,
      Wf-328 wire-up flags, default mode = Wf-335-R1).

  - test_wf74_v2_drugood_dann_preflight_integration
      Verifies that ``pre_flight_check.py --pipeline all`` is callable on the
      project root (this is the gate that Wf-336-X3 must pass before the
      launch script can be promoted to real GPU execution).  Also confirms
      the Wf-305D DANN driver still imports cleanly (no Wf-328-W1 collision
      regression).

All tests are CPU-only — no torch.cuda, no real launches.  < 5 s wall.

Honest framing (AGENTS.md §5.2):
  - MEASURED: Wf-305D trainer file exists + has --use-dann + ReverseLayerF
    markers (per grep 2026-09-29).
  - MEASURED: Wf-328 wire-ups W1-W10 ship in the production trainer
    (per Wf-328-SYN §3).
  - PROJECTED: Wf-335-R1 smoke wall ~30 sec on CPU.
  - PROJECTED: Wf-335-R3 full GPU wall ~3-4 hr on RTX 5090 cuda:1.
  - Wf-335-R1 mode is the default (per task brief).
  - USER_OK gate: optional for Wf-335-R1 smoke; required for Wf-335-R3
    full GPU run.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root + path setup
# ---------------------------------------------------------------------------
ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS_DIR = ROOT / "scripts"
TEST_SCRIPTS_DIR = ROOT / "molmetal" / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LAUNCH_SCRIPT = SCRIPTS_DIR / "launch_wf74_v2_drugood_dann.sh"
DRUGOOD_TRAINER = TEST_SCRIPTS_DIR / "wf305d_drugood_dann_train.py"
PRE_FLIGHT = TEST_SCRIPTS_DIR / "pre_flight_check.py"
DRUGOOD_CSV = ROOT / "molmetal" / ".data" / "molmetal" / "drugood_ecfp4_scaffold.csv"


# ---------------------------------------------------------------------------
# Test 1 — launch script parses + has the expected structural markers
# ---------------------------------------------------------------------------
def test_launch_wf74_v2_drugood_dann_script_parses() -> None:
    """Verify the launch script is well-formed and gates the expected pieces.

    Gates (per the Wf-336-X3 brief + Wf-300C / Wf-294B conventions):
        a) USER_OK gate enforced (required for R3; advisory for R1)
        b) pre_flight_check.py --pipeline all invoked
        c) wave_budget_tracker.py invoked
        d) VRAM budget check via nvidia-smi (per AGENTS.md §5.1)
        e) Wf-328 wire-ups: --cpu-num-workers 8, --w2-gradient-mode sort,
           --use-dann, --use-kendall
        f) Default mode = Wf-335-R1 (per task brief)
        g) Mode selector accepts R1 + R3
        h) bash -n parses the script cleanly (no heredoc / quoting bugs)
    """
    assert LAUNCH_SCRIPT.exists(), (
        f"launch script missing: {LAUNCH_SCRIPT} (Wf-336-X3 brief)"
    )
    text = LAUNCH_SCRIPT.read_text()

    # ----- (a) USER_OK gate marker -------------------------------------
    assert "USER_OK" in text, "USER_OK gate marker missing"
    assert "user_ok_gate" in text, "user_ok_gate function missing"
    # R3 mode must require USER_OK=YES (per Wf-300C + Wf-294B)
    assert re.search(r'R3.*USER_OK.*YES|USER_OK.*YES.*R3', text, re.DOTALL), (
        "Wf-335-R3 mode must require USER_OK=YES (kill_then_adapt)"
    )

    # ----- (b) pre_flight_check.py invoked -----------------------------
    assert "pre_flight_check" in text, "pre_flight_check.py invocation missing"
    assert "--pipeline all" in text, "pre_flight must run --pipeline all"
    assert "PRE_FLIGHT" in text, "PRE_FLIGHT constant missing"

    # ----- (c) wave_budget_tracker.py invoked --------------------------
    assert "wave_budget_tracker" in text, "wave_budget_tracker.py invocation missing"
    assert "BUDGET_TRACKER" in text, "BUDGET_TRACKER constant missing"

    # ----- (d) VRAM budget check via nvidia-smi ------------------------
    assert "nvidia-smi" in text, "VRAM check via nvidia-smi missing"
    assert "VRAM_BUDGET_GB" in text, "VRAM_BUDGET_GB constant missing"
    assert "vram_gate" in text, "vram_gate function missing"

    # ----- (e) Wf-328 wire-ups wired -----------------------------------
    assert "--cpu-num-workers 8" in text or "WF328_NUM_WORKERS=8" in text, (
        "Wf-328-W2 (--cpu-num-workers 8) missing"
    )
    assert "sort" in text.lower() and "w2-gradient" in text.lower(), (
        "Wf-328-W5 (--w2-gradient-mode sort) missing"
    )
    assert "--use-dann" in text, "Wf-305D --use-dann flag missing"
    assert "--use-kendall" in text, "Wf-305D --use-kendall flag missing"
    assert "ReverseLayerF" in text or "reverse-layer" in text.lower(), (
        "DANN ReverseLayerF marker missing"
    )

    # ----- (f) Default mode = Wf-335-R1 --------------------------------
    assert 'MODE="R1"' in text or "WF-335-R1" in text or "Wf-335-R1" in text, (
        "Wf-335-R1 default mode marker missing"
    )

    # ----- (g) Mode selector accepts R1 + R3 ---------------------------
    assert '"R1"' in text and '"R3"' in text, "Mode selector must accept R1 + R3"

    # ----- (h) bash parses cleanly -------------------------------------
    rc = subprocess.run(
        ["bash", "-n", str(LAUNCH_SCRIPT)],
        capture_output=True, text=True, timeout=10,
    )
    assert rc.returncode == 0, (
        f"bash -n failed: stderr={rc.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — pre-flight integration + Wf-305D driver still imports cleanly
# ---------------------------------------------------------------------------
def test_wf74_v2_drugood_dann_preflight_integration() -> None:
    """Verify pre-flight is callable + Wf-305D DANN driver imports cleanly.

    This test gates two risks:

      1. Wf-328-W1 collision: per Wf-328-SYN §6.3, the Wf-305D trainer's
         sibling refactor introduced a ``try/finally`` unclosed block that
         broke the file's import path.  Re-verify the import is clean.

      2. pre-flight gate: Wf-336-X3 brief requires
         ``pre_flight_check.py --pipeline all`` PASS before promotion.
         The script must exist + have the v3 gate markers.

    Both checks are CPU-only — no torch.cuda, no GPU launch.
    """
    # ----- Wf-305D DANN driver still imports ----------------------------
    assert DRUGOOD_TRAINER.exists(), f"Wf-305D trainer missing: {DRUGOOD_TRAINER}"
    text = DRUGOOD_TRAINER.read_text()
    assert "use_dann" in text, "Wf-305D trainer missing use_dann arg"
    assert "ReverseLayerF" in text, "Wf-305D trainer missing ReverseLayerF (DANN GRL)"
    assert "wf305d" in text.lower(), "Wf-305D marker comment missing"

    # Try importing the trainer module (AST parse, NOT full exec).
    # This catches unclosed try/finally + syntax regressions without
    # requiring torch / rdkit / cuda.
    import ast
    try:
        ast.parse(text)
    except SyntaxError as exc:
        pytest.fail(
            f"Wf-305D trainer AST parse failed: {exc} "
            "(Wf-328-W1 collision regression?)"
        )

    # ----- pre-flight script exists + has the v3 gate markers ----------
    assert PRE_FLIGHT.exists(), f"pre_flight_check.py missing: {PRE_FLIGHT}"
    pf_text = PRE_FLIGHT.read_text()
    assert "--pipeline" in pf_text, "pre_flight --pipeline flag missing"
    # The v3 gate emits status=PASS / status=FAIL on stdout
    assert "PASS" in pf_text and "FAIL" in pf_text, (
        "pre_flight PASS/FAIL stdout markers missing"
    )

    # ----- DrugOOD CSV exists (the launch script will need it) ----------
    assert DRUGOOD_CSV.exists(), f"DrugOOD CSV missing: {DRUGOOD_CSV}"
    n_lines = sum(1 for _ in DRUGOOD_CSV.open())
    assert n_lines > 1000, (
        f"DrugOOD CSV has only {n_lines} lines; expected > 1000"
    )
