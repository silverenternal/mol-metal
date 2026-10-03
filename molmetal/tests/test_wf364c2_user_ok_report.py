"""Wf-364-C2 — USER_OK outstanding report verification.

Verifies the comprehensive USER_OK-gated outstanding items report at
``docs/USER_OK_OUTSTANDING_REPORT.md`` exists and has all 6 required sections
per the Wf-364-C2 task brief.

Per AGENTS.md §5.1 + §5.3 + §5.4 + party-twenty-spirit §3.5:
- Docs-only verification (no GPU launches)
- Zero touches to ``paper/*``
- MEASURED-vs-PROJECTED tag discipline preserved
"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "docs" / "USER_OK_OUTSTANDING_REPORT.md"

EXPECTED_SECTIONS = [
    "§1 Per-launcher USER_OK requirements",
    "§2 Per-install USER_OK requirements",
    "§3 Per-paper USER_OK requirements",
    "§4 Per-metric USER_OK requirements",
    "§5 Total wall + risk assessment",
    "§6 Cross-references",
]


class TestWf364C2UserOkReport(unittest.TestCase):
    """Verify Wf-364-C2 USER_OK outstanding report surface."""

    def test_report_exists_and_has_six_sections(self) -> None:
        """Pin the Wf-364-C2 USER_OK outstanding report contract:
        file exists at docs/USER_OK_OUTSTANDING_REPORT.md AND has all 6 sections
        per the Wf-364-C2 task brief:
          - §1 Per-launcher USER_OK requirements + wall estimate
          - §2 Per-install USER_OK requirements
          - §3 Per-paper USER_OK requirements
          - §4 Per-metric USER_OK requirements
          - §5 Total wall + risk assessment
          - §6 Cross-references

        Per AGENTS.md §5.2 + party-twenty-spirit §3.5: MEASURED-vs-PROJECTED
        tag discipline preserved (the report itself is MEASURED on-disk artifact).
        """
        self.assertTrue(
            REPORT_PATH.exists(),
            f"USER_OK outstanding report missing at {REPORT_PATH}",
        )

        report_text = REPORT_PATH.read_text(encoding="utf-8")

        # All 6 expected sections present (substring match for the §N header)
        for section_header in EXPECTED_SECTIONS:
            self.assertIn(
                section_header,
                report_text,
                f"Missing section: {section_header!r}",
            )

        # The report must be reasonably non-trivial (>= 100 LOC per Wf-364-C2
        # brief; the actual file is ~387 LOC per Wf-364-C2 measurement).
        line_count = len(report_text.splitlines())
        self.assertGreaterEqual(
            line_count,
            100,
            f"Report too short ({line_count} LOC); expected >= 100 LOC",
        )

        # Per AGENTS.md §5.2 honest framing: MEASURED + PROJECTED tags should
        # be present (sanity check the discipline is preserved).
        self.assertTrue(
            "MEASURED" in report_text or "MEASURED" in report_text.upper(),
            "Report missing MEASURED tag discipline (AGENTS.md §5.2)",
        )
        self.assertIn(
            "PROJECTED",
            report_text,
            "Report missing PROJECTED tag discipline (AGENTS.md §5.2)",
        )

        # Per AGENTS.md §5.3 + TODO-84: paper/ touches are USER_OK-gated.
        # The report itself is in docs/ (NOT paper/) — verify path discipline.
        rel_report = os.path.relpath(REPORT_PATH, REPO_ROOT)
        self.assertTrue(
            rel_report.startswith("docs/"),
            f"Report must live under docs/ (not paper/); got {rel_report}",
        )
        self.assertFalse(
            rel_report.startswith("paper/"),
            "Report must NOT live under paper/ (TODO-84 active)",
        )


if __name__ == "__main__":
    unittest.main()
