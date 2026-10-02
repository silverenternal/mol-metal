"""Tests for Wf-369-A2 — m2 R² FINAL VERDICT doc.

=====================================================================
Wf-369-A2 closeout — per Wf-335-M2 + Wf-367-B1, the m2 R² ceiling
recovery axis is structurally unliftable CPU-only on n=200 cohorts
(platinai_4cell + hilbert); only metalcytotox_combined (n=2000)
achieves CEILING_BREAK.

Test surface (1 test):
  * test_wf369a2_m2_r2_verdict
        Verdict doc exists at docs/M2_R2_FINAL_VERDICT.md; has 7
        sections (per Wf-369-A2 task brief); covers both Wf-335-M2
        and Wf-367-B1 attempts with per-cohort verdict labels.

Honest framing (party-twenty-spirit §3.5):
  * The verdict doc is the deliverable; the test pins its structural
    invariants (existence + 7 sections + both attempts referenced).
  * If the verdict doc is missing or has fewer than 7 sections: FAIL.
  * If the verdict doc fails to reference Wf-335-M2 or Wf-367-B1: FAIL.
"""
from __future__ import annotations

import os
import re
import unittest


REPO_ROOT = "/home/hugo/codes/mol-metal"
VERDICT_DOC_PATH = os.path.join(REPO_ROOT, "docs", "M2_R2_FINAL_VERDICT.md")
WF335M2_CLOSEOUT_PATH = os.path.join(
    REPO_ROOT, "TODO", "completed", "335m2_r2_w2_chemprop_full.md"
)
WF367B1_CLOSEOUT_PATH = os.path.join(
    REPO_ROOT, "TODO", "completed", "367b1_m2_r2_recovery.md"
)
WF367B1_RESULT_JSON = os.path.join(
    REPO_ROOT, "molmetal", "reports", "wf367b1_m2_r2", "result.json"
)


class TestWf369A2M2R2Verdict(unittest.TestCase):
    """m2 R² FINAL VERDICT doc structural invariants (Wf-369-A2)."""

    def test_wf369a2_m2_r2_verdict(self) -> None:
        """Verdict doc exists + has 7 sections + covers both attempts."""
        # 1. Verdict doc must exist on disk
        self.assertTrue(
            os.path.exists(VERDICT_DOC_PATH),
            f"verdict doc missing: {VERDICT_DOC_PATH}",
        )

        with open(VERDICT_DOC_PATH, "r", encoding="utf-8") as fh:
            content = fh.read()

        # 2. Verdict doc must have >= 7 top-level `## §N` sections
        section_headers = re.findall(r"^## §(\d+)\s", content, flags=re.MULTILINE)
        n_sections = len(section_headers)
        self.assertGreaterEqual(
            n_sections,
            7,
            f"verdict doc must have >= 7 sections; got {n_sections}: {section_headers}",
        )

        # 3. The 7 expected section topics (substring match on heading lines)
        expected_section_topics = [
            r"§1\s+Per-attempt summary",
            r"§2\s+Why R",
            r"§3\s+metalcytotox_combined",
            r"§4\s+platinai_4cell",
            r"§5\s+Forward-looking",
            r"§6\s+Paper",
            r"§7\s+Cross-references",
        ]
        for topic_pattern in expected_section_topics:
            self.assertRegex(
                content,
                topic_pattern,
                f"verdict doc missing required section matching: {topic_pattern}",
            )

        # 4. Both attempts (Wf-335-M2 + Wf-367-B1) must be referenced
        self.assertIn(
            "Wf-335-M2",
            content,
            "verdict doc must reference Wf-335-M2 attempt",
        )
        self.assertIn(
            "Wf-367-B1",
            content,
            "verdict doc must reference Wf-367-B1 attempt",
        )

        # 5. Both cohorts (platinai_4cell + hilbert) must appear with HONEST_NEGATIVE_R2_DEEP
        for cohort in ("platinai_4cell", "hilbert"):
            self.assertIn(
                cohort,
                content,
                f"verdict doc must reference cohort: {cohort}",
            )
        self.assertIn(
            "HONEST_NEGATIVE_R2_DEEP",
            content,
            "verdict doc must declare HONEST_NEGATIVE_R2_DEEP verdict",
        )

        # 6. CEILING_BREAK preserved for metalcytotox_combined
        self.assertIn(
            "metalcytotox_combined",
            content,
            "verdict doc must reference metalcytotox_combined cohort",
        )
        self.assertIn(
            "CEILING_BREAK",
            content,
            "verdict doc must declare CEILING_BREAK verdict",
        )

        # 7. Wf-367-B1 source artifacts cross-referenced
        self.assertIn(
            "wf367b1_m2_r2/result.json",
            content,
            "verdict doc must cross-reference Wf-367-B1 result JSON path",
        )
        self.assertIn(
            "335m2_r2_w2_chemprop_full.md",
            content,
            "verdict doc must cross-reference Wf-335-M2 closeout path",
        )

        # 8. Source artifacts exist on disk (back-reference integrity)
        self.assertTrue(
            os.path.exists(WF335M2_CLOSEOUT_PATH),
            f"Wf-335-M2 closeout missing on disk: {WF335M2_CLOSEOUT_PATH}",
        )
        self.assertTrue(
            os.path.exists(WF367B1_CLOSEOUT_PATH),
            f"Wf-367-B1 closeout missing on disk: {WF367B1_CLOSEOUT_PATH}",
        )
        self.assertTrue(
            os.path.exists(WF367B1_RESULT_JSON),
            f"Wf-367-B1 result JSON missing on disk: {WF367B1_RESULT_JSON}",
        )

        # 9. Honest framing (party-twenty-spirit §3.5) — at least one MEASURED tag
        self.assertIn(
            "MEASURED",
            content,
            "verdict doc must use MEASURED tag (party-twenty-spirit §3.5)",
        )

        # 10. Forward-looking path-A / path-B / path-C vocabulary present
        self.assertTrue(
            "USER_OK" in content or "USER-OK" in content,
            "verdict doc must reference USER_OK-gated forward-looking paths",
        )


if __name__ == "__main__":
    unittest.main()
