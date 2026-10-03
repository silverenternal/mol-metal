"""Wf-371-A2 — Standalone Wf-368-C1 final_synthesis closeout test.

Per Wf-369-A1 §1.1: Wf-368-C1 file MISSING on disk (rolled-in under
368c1_index_sync.md). Wf-371-A2 reconstructed the standalone final_synthesis closeout
at TODO/completed/368c1_final_synthesis.md with 5 sections (§0 TL;DR + §1 pre-flight
+ §2 INDEX §22 + §3 AGENTS §6 + §4 honest framing + §5 cross-refs).

This test verifies the standalone closeout exists + has all 5 expected sections.

Reference: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) +
`AGENTS.md` §5.2 (honest framing) + Wf-369-A1 §1.1 (Wf-368-C1 gap-flag).
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLOSEOUT_PATH = REPO_ROOT / "TODO" / "completed" / "368c1_final_synthesis.md"

EXPECTED_SECTIONS = [
    r"^## §0 TL;DR",
    r"^## §1 Final pre-flight state",
    r"^## §2 INDEX\.md §22 entry",
    r"^## §3 AGENTS\.md §6 entry",
    r"^## §4 Honest framing",
    r"^## §5 Cross-references",
]


class TestWf371A2Wf368C1Standalone:
    """Standalone closeout reconstruction test (Wf-371-A2)."""

    def test_standalone_closeout_exists_and_has_six_sections(self) -> None:
        """Verify TODO/completed/368c1_final_synthesis.md exists + has all 5 sections."""
        assert CLOSEOUT_PATH.exists(), (
            f"Standalone closeout file MISSING at {CLOSEOUT_PATH} — Wf-371-A2 should "
            f"author this per Wf-369-A1 §1.1 HONEST-NEGATIVE_FLAG"
        )

        text = CLOSEOUT_PATH.read_text(encoding="utf-8")

        # Section count check
        missing_sections = []
        for pattern in EXPECTED_SECTIONS:
            if not re.search(pattern, text, re.MULTILINE):
                missing_sections.append(pattern)

        assert not missing_sections, (
            f"Wf-371-A2 standalone closeout is missing sections: {missing_sections}. "
            f"Expected 6 sections (§0 TL;DR + §1 + §2 + §3 + §4 + §5)."
        )

        # Sanity: file should be reasonably long (PROJECTED ~280 LOC)
        line_count = len(text.splitlines())
        assert line_count >= 100, (
            f"Wf-371-A2 standalone closeout is too short ({line_count} lines); "
            f"PROJECTED ~280 LOC per Wf-371-A2 brief"
        )

        # Sanity: pre-flight state numbers should be present
        assert (
            "92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL" in text
        ), "Wf-371-A2 standalone closeout should cite pre-flight 92/0/2/94 state"

        # Sanity: m6_kappa + m2 R² HONEST-NEGATIVE markers should be present
        assert (
            "HONEST_NEGATIVE_PARTIAL_STRUCTURAL" in text
        ), "Wf-371-A2 standalone closeout should preserve m6_kappa HONEST_NEGATIVE marker"
        assert (
            "HONEST_NEGATIVE_R2_DEEP" in text
        ), "Wf-371-A2 standalone closeout should preserve m2 R² HONEST_NEGATIVE marker"

        # Sanity: 7 USER_OK-gated launches should be referenced
        assert (
            "launch_wf124_p3_g5.sh" in text
        ), "Wf-371-A2 standalone closeout should reference G5 USER_OK-gated launch"
        assert (
            "recover_g6_failed_tuples.sh" in text
        ), "Wf-371-A2 standalone closeout should reference G6 recover USER_OK-gated launch"

        # Sanity: AGENTS.md §5 standing directives re-confirmed
        assert (
            "§5.1" in text and "§5.2" in text and "§5.4" in text
        ), "Wf-371-A2 standalone closeout should re-confirm AGENTS.md §5.1..§5.4 directives"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
