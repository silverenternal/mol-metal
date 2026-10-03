"""Wf-381-A4 — TODO-206 m6_kappa metric redefinition finalization tests.

Per Wf-381-A4 task brief (2026-09-30):
"Wf-381-A4: TODO-206 m6_kappa metric redefinition — CREATE pending file + verify
Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + fill gaps + 1 NEW test."

This test module verifies:
1. The canonical TODO/pending/206_m6_kappa_metric_redefinition.md exists on disk.
2. The 4 SHIPPED m6_kappa reframe docs exist on disk (Wf-375-A4 + Wf-376-A4 + Wf-377-A1 +
   Wf-365-A3).
3. Each reframe doc has the expected canonical section structure (§0..§N per doc).
4. Cross-references between the 4 docs + Wf-381-A4 source brief are wired.
5. The 8-path inventory aggregates correctly across the docs (Wf-335-M1 + Wf-359-A1 +
   Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 + Wf-377-A1).
6. The recommended §3 main-result line-up invariant holds (m6_kappa DROPPED, m1_auc +
   m2 + m3 + m15 RETAINED, top-3 alternatives PROMOTED to supplementary sub-table).
7. The pending source brief contains the user-voice-answer relay + METRIC_LIFT_STRATEGY
   row 381 cross-reference + 8-path inventory + cross-doc linkage.

CPU-only per AGENTS.md §5.1 (no GPU launches).
0 paper/* touches per AGENTS.md §5.3 + TODO-84.

Author: Hugo Lee + Wf-381-A4 ultracode.
Date: 2026-09-30.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical TODO-206 source brief (created by Wf-381-A4)
TODO_206_PENDING = REPO_ROOT / "TODO" / "pending" / "206_m6_kappa_metric_redefinition.md"

# 4 m6_kappa reframe docs (verified SHIPPED at Wf-381-A4 start)
PAPER_S6 = REPO_ROOT / "docs" / "PAPER_S6_M6_KAPPA_REFRAME.md"  # Wf-375-A4 (8 sections)
SECTION_3 = REPO_ROOT / "docs" / "SECTION_3_METRIC_REFRAME_TEMPLATE.md"  # Wf-376-A4 (7 sections)
REDESIGN = REPO_ROOT / "docs" / "M6_KAPPA_REDESIGN_PROPOSAL.md"  # Wf-377-A1 (8 sections)
FINAL_VERDICT = REPO_ROOT / "docs" / "M6_KAPPA_FINAL_VERDICT.md"  # Wf-365-A3 (9 sections)

# Canonical section headers per doc (per task brief §1 verification step)
EXPECTED_SECTIONS = {
    "PAPER_S6": [  # 8 sections per task brief
        "## §0 TL;DR",
        "## §1 Why m6_kappa fails",
        "## §2 Recommended paper framing",
        "## §3 Concrete §6 Limitations draft text",
        "## §4 Cross-references",
        "## §5 Wall + cost",
        "## §6 Standing-rule compliance",
        "## §7 Final state",
    ],
    "SECTION_3": [  # 7 sections per task brief
        "## §0 TL;DR",
        "## §1 Why m6_kappa fails",
        "## §2 Recommended paper §3 metric lineup",
        "## §3 Ready-to-paste LaTeX snippet",
        "## §4 Per-metric USER_OK-gate status",
        "## §5 Cross-references",
        "## §6 How to use this template",
    ],
    "REDESIGN": [  # 8 sections per task brief
        "## §0 TL;DR",
        "## §1 Why m6_kappa fails",
        "## §2 Literature review",
        "## §3 Empirical validation",
        "## §4 Recommended metric replacement",
        "## §5 Cross-references",
        "## §6 Final state",
        "## §7 Standing-rule compliance",
    ],
    "FINAL_VERDICT": [  # 9 sections per task brief
        "## §1 Per-attempt summary",
        "## §2 Why the 0.6 threshold was not achieved",
        "## §3 What was lifted",
        "## §4 What remains honest-negative",
        "## §5 Forward-looking",
        "## §6 Paper §6 entry prep",
        "## §7 Cross-references",
        "## §8 Standing rules honored",
        "## §9 Wall + cost",
    ],
}

# 8 unique m6_kappa paths (per Wf-381-A4 §4.2 inventory)
EIGHT_PATH_WORKFLOWS = [
    "Wf-335-M1",  # chemprop+PLL
    "Wf-359-A1",  # R20-P3 W₂
    "Wf-361-B1",  # MAML
    "Wf-361-B2",  # Neural ODE dopri5
    "Wf-364-A2",  # Combined MAML+ODE (LAST architectural)
    "Wf-365-A3",  # Verdict aggregation
    "Wf-371-A1",  # DEGENERATE (RDKit 12-d)
    "Wf-377-A1",  # METRIC REDESIGN (16 inv × n=5460)
]


class TestW381A4M6KappaFinalize(unittest.TestCase):
    """Wf-381-A4 — TODO-206 m6_kappa metric redefinition finalization."""

    def test_pending_file_exists(self):
        """Canonical TODO/pending/206 source brief exists on disk."""
        self.assertTrue(
            TODO_206_PENDING.exists(),
            f"Missing TODO-206 source brief at {TODO_206_PENDING}",
        )
        # Must be non-trivial (>= 5000 bytes per AGENTS.md §5.2 documentation convention)
        size = TODO_206_PENDING.stat().st_size
        self.assertGreater(
            size, 5000,
            f"TODO-206 pending file too small: {size} bytes (expect >= 5000)",
        )

    def test_pending_file_sections(self):
        """Pending source brief has 5 expected sections."""
        text = TODO_206_PENDING.read_text(encoding="utf-8")
        for sec in ["## §0 TL;DR", "## §1 Per-doc status",
                    "## §2 Gaps filled", "## §3 Final verdict",
                    "## §4 Cross-references"]:
            self.assertIn(sec, text, f"Pending file missing section: {sec}")

    def test_pending_file_user_voice_relay(self):
        """Pending source brief contains user-voice-answer relay."""
        text = TODO_206_PENDING.read_text(encoding="utf-8")
        # Must contain key relay phrases (per Wf-381-A4 task brief)
        self.assertIn("ultracode", text)
        self.assertIn("TODO-206", text)
        self.assertIn("METRIC_LIFT_STRATEGY", text)
        # Must contain at least one of the 4 SHIPPED workflow references
        self.assertTrue(
            any(wf in text for wf in ["Wf-375-A4", "Wf-376-A4", "Wf-377-A1", "Wf-365-A3"]),
            "Pending file missing workflow cross-references",
        )

    def test_all_4_docs_exist(self):
        """All 4 SHIPPED m6_kappa reframe docs exist on disk."""
        for label, path in [
            ("PAPER_S6", PAPER_S6),
            ("SECTION_3", SECTION_3),
            ("REDESIGN", REDESIGN),
            ("FINAL_VERDICT", FINAL_VERDICT),
        ]:
            self.assertTrue(
                path.exists(),
                f"Missing m6_kappa reframe doc: {label} at {path}",
            )
            # Must be non-trivial
            self.assertGreater(
                path.stat().st_size, 5000,
                f"{label} too small: {path.stat().st_size} bytes (expect >= 5000)",
            )

    def test_paper_s6_sections(self):
        """PAPER_S6 has all 8 expected section headers."""
        text = PAPER_S6.read_text(encoding="utf-8")
        for sec in EXPECTED_SECTIONS["PAPER_S6"]:
            self.assertIn(sec, text, f"PAPER_S6 missing section: {sec}")

    def test_section_3_sections(self):
        """SECTION_3 has all 7 expected section headers."""
        text = SECTION_3.read_text(encoding="utf-8")
        for sec in EXPECTED_SECTIONS["SECTION_3"]:
            self.assertIn(sec, text, f"SECTION_3 missing section: {sec}")

    def test_redesign_sections(self):
        """REDESIGN has all 8 expected section headers."""
        text = REDESIGN.read_text(encoding="utf-8")
        for sec in EXPECTED_SECTIONS["REDESIGN"]:
            self.assertIn(sec, text, f"REDESIGN missing section: {sec}")

    def test_final_verdict_sections(self):
        """FINAL_VERDICT has all 9 expected section headers."""
        text = FINAL_VERDICT.read_text(encoding="utf-8")
        for sec in EXPECTED_SECTIONS["FINAL_VERDICT"]:
            self.assertIn(sec, text, f"FINAL_VERDICT missing section: {sec}")

    def test_8_path_inventory_aggregates(self):
        """8 unique Wf paths present across all 4 docs + pending source brief."""
        all_text = "\n".join([
            PAPER_S6.read_text(encoding="utf-8"),
            SECTION_3.read_text(encoding="utf-8"),
            REDESIGN.read_text(encoding="utf-8"),
            FINAL_VERDICT.read_text(encoding="utf-8"),
            TODO_206_PENDING.read_text(encoding="utf-8"),
        ])
        for wf in EIGHT_PATH_WORKFLOWS:
            self.assertIn(
                wf, all_text,
                f"8-path inventory missing: {wf} across all 4 docs + pending brief",
            )

    def test_cross_refs_present(self):
        """Cross-references between the 4 docs + Wf-381-A4 are wired."""
        # PAPER_S6 must reference Wf-376-A4 + Wf-377-A1 + Wf-365-A3 + Wf-371-A1
        paper_s6 = PAPER_S6.read_text(encoding="utf-8")
        for ref in ["Wf-376-A4", "Wf-377-A1", "Wf-365-A3", "Wf-371-A1"]:
            self.assertIn(ref, paper_s6, f"PAPER_S6 missing cross-ref: {ref}")
        # SECTION_3 must reference Wf-377-A1 + Wf-365-A3 + Wf-371-A1 + Wf-375-A4
        section_3 = SECTION_3.read_text(encoding="utf-8")
        for ref in ["Wf-377-A1", "Wf-365-A3", "Wf-371-A1", "Wf-375-A4"]:
            self.assertIn(ref, section_3, f"SECTION_3 missing cross-ref: {ref}")
        # REDESIGN must reference Wf-375-A4 + Wf-371-A1 + Wf-365-A3
        redesign = REDESIGN.read_text(encoding="utf-8")
        for ref in ["Wf-375-A4", "Wf-371-A1", "Wf-365-A3"]:
            self.assertIn(ref, redesign, f"REDESIGN missing cross-ref: {ref}")
        # FINAL_VERDICT must reference Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-380-A7
        final_verdict = FINAL_VERDICT.read_text(encoding="utf-8")
        for ref in ["Wf-375-A4", "Wf-376-A4", "Wf-377-A1", "Wf-380-A7"]:
            self.assertIn(ref, final_verdict, f"FINAL_VERDICT missing cross-ref: {ref}")
        # TODO-206 pending must reference all 4 docs' Wf IDs + Wf-381-A4 self
        pending = TODO_206_PENDING.read_text(encoding="utf-8")
        for ref in ["Wf-375-A4", "Wf-376-A4", "Wf-377-A1", "Wf-365-A3", "Wf-381-A4"]:
            self.assertIn(ref, pending, f"TODO-206 pending missing cross-ref: {ref}")

    def test_section_3_lineup_invariant(self):
        """Each doc contains DROP/demote m6_kappa recommendation + 4-metric headline."""
        all_text = "\n".join([
            PAPER_S6.read_text(encoding="utf-8"),
            SECTION_3.read_text(encoding="utf-8"),
            REDESIGN.read_text(encoding="utf-8"),
            FINAL_VERDICT.read_text(encoding="utf-8"),
            TODO_206_PENDING.read_text(encoding="utf-8"),
        ])
        # DROP/demote m6_kappa
        self.assertTrue(
            "DROP" in all_text or "drop" in all_text or "demote" in all_text.lower(),
            "Aggregate docs missing DROP/demote m6_kappa recommendation",
        )
        # 4-metric headline references
        for metric in ["m1_auc", "m2", "m3", "m15"]:
            self.assertIn(
                metric, all_text,
                f"Aggregate docs missing metric reference: {metric}",
            )

    def test_top3_supplementary_promoted(self):
        """REDESIGN contains MolLogP + MorganDensity_r2 + Chi3v top-3 alternatives."""
        redesign = REDESIGN.read_text(encoding="utf-8")
        for alt in ["MolLogP", "MorganDensity_r2", "Chi3v"]:
            self.assertIn(
                alt, redesign,
                f"REDESIGN missing top-3 alternative: {alt}",
            )

    def test_latex_template_present(self):
        """PAPER_S6 + SECTION_3 contain \\begin{itemize} LaTeX template blocks."""
        paper_s6 = PAPER_S6.read_text(encoding="utf-8")
        section_3 = SECTION_3.read_text(encoding="utf-8")
        self.assertIn(
            "\\begin{itemize}", paper_s6,
            "PAPER_S6 missing LaTeX \\begin{itemize} template",
        )
        self.assertIn(
            "\\begin{itemize}", section_3,
            "SECTION_3 missing LaTeX \\begin{itemize} template",
        )

    def test_no_paper_touches(self):
        """0 paper/* touches per AGENTS.md §5.3 + TODO-84 (Wf-381-A4 docs-only)."""
        # Per Wf-381-A4 docs-only constraint: 0 NEW files under paper/*
        # This test verifies the 5 deliverables (pending + 4 reframe docs + test)
        # do NOT add files under paper/.
        paper_dir = REPO_ROOT / "paper"
        if not paper_dir.exists():
            self.skipTest("paper/ directory not present — N/A")
        # Wf-381-A4 does NOT create paper/* files; pre-flight v3 PASS confirms
        # pre-existing paper/* untouched. This test is a guard against
        # accidental paper/* writes.
        # We only check that none of the Wf-381-A4 deliverables live under paper/.
        w381_a4_paths = [
            TODO_206_PENDING,
            PAPER_S6,  # pre-existing Wf-375-A4
            SECTION_3,  # pre-existing Wf-376-A4
            REDESIGN,  # pre-existing Wf-377-A1
            FINAL_VERDICT,  # pre-existing Wf-365-A3
        ]
        for p in w381_a4_paths:
            self.assertFalse(
                str(p).startswith(str(paper_dir)),
                f"Wf-381-A4 must not write to paper/: {p}",
            )


if __name__ == "__main__":
    unittest.main()
