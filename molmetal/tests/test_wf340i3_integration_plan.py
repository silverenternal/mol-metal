"""Wf-340-I3 — Integration + risk plan verification (4 NEW tests).

Tests:
1. test_phase_A_B_C_ordering — verifies the 3-phase integration roadmap ordering
2. test_pre_flight_gates — verifies each phase has a pre-flight gate
3. test_rollback_path — verifies each phase has explicit rollback
4. test_backward_compat — verifies default OFF for safety

All tests are CPU-only, run in <1 sec, and verify a specific property of the
Wf-340-I3 integration plan defined in
`TODO/completed/340i3_integration_plan.md`.

Honest framing (per `party-twenty-spirit` §3.5 + AGENTS.md §5.2):
- These tests verify the PLAN document + CLI-flag surface; they do NOT
  verify actual CFM acceleration behavior (deferred to Wf-340-A8 GPU A/B).
- Defaults all OFF for safety — every CLI flag default `False` reproduces
  legacy output.
- Pre-flight v3 62/62 PASS preserved at every rollback boundary
  (Wf-335-SYN baseline).

Cross-ref: TODO/completed/340i3_integration_plan.md §2 + §3 + §6.
"""

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "TODO" / "completed" / "340i3_integration_plan.md"
CFM_TRAIN_SCRIPT = (
    REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
)
PRE_FLIGHT_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"


def _read(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def _phase_section(plan: str, phase: str) -> str:
    """Extract a Phase section from the plan markdown by header.

    Looks for '### §2.X Phase <LETTER>' (top-level single digit subsection)
    and returns from there to the next top-level '### §2.Y' header or
    '## §3' major section.

    Uses MULTILINE mode with `^` to anchor to start of line, avoiding
    false-positive matches inside nested `#### §2.X.Y` subsections
    (which also contain the substring `## §[0-9]`).
    """
    # Top-level Phase headers are like `### §2.1 Phase A — ...`
    # Nested subsections like `### §2.1.1 ...` should NOT match.
    # Use ^ anchor with MULTILINE mode to match only line starts.
    pattern = rf"^### §2\.[0-9] Phase {phase} — .*?(?=^### §2\.[0-9] |^## §[0-9])"
    m = re.search(pattern, plan, re.DOTALL | re.MULTILINE)
    assert m is not None, f"Could not find Phase {phase} section in plan"
    return m.group(0)


# ---------------------------------------------------------------------------
# Test 1 — Phase A B C ordering
# ---------------------------------------------------------------------------


def test_phase_A_B_C_ordering():
    """Verify the 3 phases (A: Mini-batch OT, B: EDM, C: Shortcut+EMA) appear
    in the correct sequential order in the plan.

    Phase A is Mini-batch OT (CPU + smoke); Phase B is EDM preconditioning;
    Phase C is Shortcut consistency + EMA. The integration roadmap should list
    them in this order with explicit ordering rationale.
    """
    plan = _read(PLAN_PATH)

    # Extract all Phase section headers in order
    phase_headers = re.findall(r"### §2\.\d+ Phase ([A-Z]) — ([^\n]+)", plan)
    assert (
        len(phase_headers) >= 3
    ), f"Plan should have at least 3 Phase sections, got {len(phase_headers)}"

    # The first 3 phase headers must be A, B, C in order
    first_three_letters = [h[0] for h in phase_headers[:3]]
    assert first_three_letters == [
        "A",
        "B",
        "C",
    ], f"First 3 phases must be A, B, C in order. Got {first_three_letters}"

    # Verify the titles match the expected optimizations
    phase_a_title = phase_headers[0][1]
    phase_b_title = phase_headers[1][1]
    phase_c_title = phase_headers[2][1]

    # Phase A should mention OT or optimal transport
    assert (
        "OT" in phase_a_title
        or "Optimal Transport" in phase_a_title
        or ("Mini-batch OT" in phase_a_title)
    ), f"Phase A title should reference Mini-batch OT, got: {phase_a_title}"

    # Phase B should mention EDM
    assert (
        "EDM" in phase_b_title or "preconditioning" in phase_b_title.lower()
    ), f"Phase B title should reference EDM preconditioning, got: {phase_b_title}"

    # Phase C should mention Shortcut or EMA
    assert (
        "Shortcut" in phase_c_title or "EMA" in phase_c_title
    ), f"Phase C title should reference Shortcut/EMA, got: {phase_c_title}"


# ---------------------------------------------------------------------------
# Test 2 — Pre-flight gates
# ---------------------------------------------------------------------------


def test_pre_flight_gates():
    """Verify each Phase has at least one pre-flight gate defined.

    Per the Wf-340-I3 plan §3.1, each phase must have CPU unit + CPU smoke +
    baseline pre-flight v3 62/62 PASS gate. Phase B additionally requires a
    license gate.
    """
    plan = _read(PLAN_PATH)

    # Each phase should appear with at least one 'pre_flight' or 'gate'
    # reference in its section.
    for phase in ("A", "B", "C"):
        section = _phase_section(plan, phase)
        # The §2.X.3 Pre-flight gate subsection MUST exist
        assert "§2." in section, f"Phase {phase} missing §2.X.3 Pre-flight gate"
        assert (
            "Pre-flight gate" in section or "pre_flight" in section.lower()
        ), f"Phase {phase} missing pre-flight gate reference"

        # Each phase must reference the baseline pre-flight v3 gate
        assert (
            "62/62" in section
            or "62 / 62" in section
            or ("pre_flight_v3_baseline" in section)
        ), f"Phase {phase} must reference baseline pre-flight v3 62/62 PASS " f"gate"

    # Phase B specifically must have a license gate (CC-BY-NC-SA 4.0)
    phase_b_section = _phase_section(plan, "B")
    assert (
        "license" in phase_b_section.lower() or "LICENSE" in phase_b_section
    ), "Phase B (EDM) must include a license gate (NVlabs/edm CC-BY-NC-SA 4.0)"

    # Verify pre_flight_check.py exists
    assert (
        PRE_FLIGHT_SCRIPT.exists()
    ), f"pre_flight_check.py must exist at {PRE_FLIGHT_SCRIPT}"


# ---------------------------------------------------------------------------
# Test 3 — Rollback path
# ---------------------------------------------------------------------------


def test_rollback_path():
    """Verify each Phase has explicit rollback plan + that all defaults are
    OFF for safety.

    Per the Wf-340-I3 plan §2.X.5, each phase must declare:
    - A rollback mechanism (CLI flag default `False` or `git revert`).
    - A risk level (LOW / MED / HIGH).
    - Pre-flight v3 62/62 PASS preserved at rollback boundary.
    """
    plan = _read(PLAN_PATH)

    for phase in ("A", "B", "C"):
        section = _phase_section(plan, phase)
        # Rollback subsection must exist
        assert (
            "Rollback" in section
        ), f"Phase {phase} missing §2.X.5 Rollback plan subsection"
        # Must mention `git revert` or default flag `False`
        has_revert = "git revert" in section
        has_default_false = (
            "False" in section
            or "default OFF" in section
            or ("defaulted to `False`" in section)
        )
        assert has_revert or has_default_false, (
            f"Phase {phase} rollback must mention `git revert` or default "
            f"`False` flag"
        )
        # Must reference 62/62 baseline preservation at rollback boundary
        assert (
            "62/62" in section or "62 / 62" in section
        ), f"Phase {phase} rollback must preserve pre-flight v3 62/62 PASS"

    # Aggregate rollback precedence section must list per-phase rollback
    assert "Phase A regression" in plan, (
        "Plan must have aggregate §3.3 rollback precedence with Phase A "
        "regression handling"
    )
    assert "Phase B regression" in plan, (
        "Plan must have aggregate §3.3 rollback precedence with Phase B "
        "regression handling"
    )
    assert "Phase C regression" in plan, (
        "Plan must have aggregate §3.3 rollback precedence with Phase C "
        "regression handling"
    )


# ---------------------------------------------------------------------------
# Test 4 — Backward compatibility
# ---------------------------------------------------------------------------


def test_backward_compat():
    """Verify default OFF for safety — every new CLI flag must default to
    `False` (or legacy behavior) so existing callers see bit-identical output.

    Per the Wf-340-I3 plan, defaults are:
    - `--use-exact-ot`: default `False` (legacy stochastic OT)
    - `--use-edm-precond`: default `False` (legacy Lipman CFM)
    - `--log-normal-t`: default `False` (legacy uniform t)
    - `--use-shortcut-loss`: default `False` (legacy single-step loss)
    - `--use-ema`: default `False` (legacy no EMA)
    """
    plan = _read(PLAN_PATH)

    # Verify each flag is mentioned with its default
    flags_with_defaults = [
        ("--use-exact-ot", "False"),
        ("--use-edm-precond", "False"),
        ("--log-normal-t", "False"),
        ("--use-shortcut-loss", "False"),
        ("--use-ema", "False"),
    ]

    for flag, default in flags_with_defaults:
        # The flag must appear in the plan
        assert flag in plan, f"Flag {flag} missing from plan"
        # Verify default is `False` somewhere in the plan with that flag
        # (the description may appear before the default statement).
        # Look for any of these patterns referencing the flag:
        # - "default `False`"
        # - "defaulted to `False`"
        # - "default=False"
        # - "{flag}** (default `False`)"
        # We use a broader search: find all occurrences of the flag, then
        # check each occurrence's 500-char context window for `False`.
        flag_indices = [m.start() for m in re.finditer(re.escape(flag), plan)]
        assert flag_indices, f"Flag {flag} not found in plan"
        # Find at least one occurrence where False appears within 500 chars
        has_false_near_flag = False
        for idx in flag_indices:
            context_window = plan[idx : idx + 500]
            if "False" in context_window or "false" in context_window.lower():
                has_false_near_flag = True
                break
        assert has_false_near_flag, (
            f"Flag {flag} default must be `False`; no occurrence has "
            f"'False' within 500 chars of the flag"
        )

    # Verify the legacy behavior is preserved at default
    assert "legacy" in plan.lower(), "Plan must mention legacy behavior preservation"

    # Verify each phase explicitly states the rollback is to legacy path
    for phase in ("A", "B", "C"):
        section = _phase_section(plan, phase)
        # Either "legacy" or "remains" or "preserved" must appear near the
        # flag context
        assert (
            "legacy" in section.lower()
            or "remains" in section.lower()
            or "preserved" in section.lower()
        ), f"Phase {phase} must state legacy behavior is preserved at default"


# ---------------------------------------------------------------------------
# Optional: smoke test that the test file itself is valid pytest
# ---------------------------------------------------------------------------


def test_plan_file_exists_and_readable():
    """Smoke test: verify the plan file exists and is readable UTF-8.

    This is a defensive test in case the plan is moved or corrupted.
    """
    assert PLAN_PATH.exists(), f"Plan file missing at {PLAN_PATH}"
    content = _read(PLAN_PATH)
    assert len(content) > 1000, "Plan file suspiciously short"
    assert "Wf-340-I3" in content, "Plan file does not contain Wf-340-I3 marker"
    assert "Mini-batch OT" in content, "Plan missing Mini-batch OT (Phase A)"
    assert "EDM" in content, "Plan missing EDM (Phase B)"
    assert "Shortcut" in content, "Plan missing Shortcut (Phase C)"
    assert "EMA" in content, "Plan missing EMA (Phase C)"


if __name__ == "__main__":
    # Allow running as `python -m pytest test_wf340i3_integration_plan.py`
    pytest.main([__file__, "-v"])
