"""Wf-380-A7 — TODO-206 m6_kappa metric redefinition finalization tests.

Per Wf-380-A7 task brief (2026-09-30):
"Wf-380-A7: TODO-206 m6_kappa metric redefinition finalization — verify
Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + fill gaps + 1 NEW test."

This test module verifies:
1. All 4 SHIPPED m6_kappa docs exist on disk.
2. Each doc has the expected canonical section structure (§0..§N per doc).
3. Cross-references between the 4 docs are wired (Wf-375-A4 ↔ Wf-376-A4 ↔ Wf-377-A1
   ↔ Wf-365-A3 ↔ Wf-380-A7).
4. The 8-path inventory aggregates correctly across the docs.
5. The recommended §3 main-result line-up invariant holds (m6_kappa DROPPED,
   m1_auc + m2 + m3 + m15 RETAINED, top-3 alternatives PROMOTED to supplementary
   sub-table).

CPU-only per AGENTS.md §5.1 (no GPU launches).
0 paper/* touches per AGENTS.md §5.3 + TODO-84.

Author: Hugo Lee + Wf-380-A7 ultracode.
Date: 2026-09-30.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 4 m6_kappa reframe docs (verified SHIPPED at Wf-380-A7 start)
PAPER_S6 = REPO_ROOT / "docs" / "PAPER_S6_M6_KAPPA_REFRAME.md"  # Wf-375-A4
SECTION_3 = REPO_ROOT / "docs" / "SECTION_3_METRIC_REFRAME_TEMPLATE.md"  # Wf-376-A4
REDESIGN = REPO_ROOT / "docs" / "M6_KAPPA_REDESIGN_PROPOSAL.md"  # Wf-377-A1
FINAL_VERDICT = REPO_ROOT / "docs" / "M6_KAPPA_FINAL_VERDICT.md"  # Wf-365-A3

# Expected section headers per doc (per task brief §1 verification step)
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
        "## §3 Ready-to-paste LaTeX",
        "## §4 Per-metric USER_OK-gate status",
        "## §5 Cross-references",
        "## §6 How to use this template",
    ],
    "REDESIGN": [  # 8 sections (Wf-377-A1 SHIPPED)
        "## §0 TL;DR",
        "## §1 Why m6_kappa fails",
        "## §2 Literature review",
        "## §3 Empirical validation",
        "## §4 Recommended metric replacement",
        "## §5 Cross-references",
        "## §6 Final state",
        "## §7 Standing-rule compliance",
    ],
    "FINAL_VERDICT": [  # 9 sections (no §0 TL;DR)
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

# Cross-reference invariants: each (source_doc, target_doc) pair must have
# a mention of the target Wf label in the source doc.
CROSS_REF_INVARIANTS = [
    ("PAPER_S6", "Wf-376-A4"),
    ("PAPER_S6", "Wf-377-A1"),
    ("PAPER_S6", "Wf-365-A3"),
    ("PAPER_S6", "Wf-371-A1"),
    ("SECTION_3", "Wf-377-A1"),
    ("SECTION_3", "Wf-365-A3"),
    ("SECTION_3", "Wf-371-A1"),
    ("SECTION_3", "Wf-375-A4"),
    ("REDESIGN", "Wf-375-A4"),
    ("REDESIGN", "Wf-371-A1"),
    ("REDESIGN", "Wf-365-A3"),
    ("FINAL_VERDICT", "Wf-375-A4"),
    ("FINAL_VERDICT", "Wf-376-A4"),
    ("FINAL_VERDICT", "Wf-377-A1"),
]

DOC_PATHS = {
    "PAPER_S6": PAPER_S6,
    "SECTION_3": SECTION_3,
    "REDESIGN": REDESIGN,
    "FINAL_VERDICT": FINAL_VERDICT,
}


def _read_text(path: Path) -> str:
    assert path.exists(), f"DOC MISSING: {path}"
    return path.read_text(encoding="utf-8")


class TestW380A7M6KappaFinalize(unittest.TestCase):
    """Verify the 4 m6_kappa reframe docs are SHIPPED + cross-linked."""

    def setUp(self) -> None:
        self.text = {key: _read_text(path) for key, path in DOC_PATHS.items()}

    # --- Test 1: all 4 docs exist + size sanity ---
    def test_all_4_docs_exist(self) -> None:
        for key, path in DOC_PATHS.items():
            self.assertTrue(path.exists(), f"{key} doc missing: {path}")
            size = path.stat().st_size
            self.assertGreater(
                size, 5000,
                f"{key} doc too small ({size} bytes); expected > 5000 LOC worth of docs",
            )

    # --- Test 2: each doc has expected canonical section structure ---
    def test_paper_s6_sections(self) -> None:
        text = self.text["PAPER_S6"]
        for header in EXPECTED_SECTIONS["PAPER_S6"]:
            self.assertIn(
                header, text,
                f"PAPER_S6 missing section: {header}",
            )

    def test_section_3_sections(self) -> None:
        text = self.text["SECTION_3"]
        for header in EXPECTED_SECTIONS["SECTION_3"]:
            self.assertIn(
                header, text,
                f"SECTION_3 missing section: {header}",
            )

    def test_redesign_sections(self) -> None:
        text = self.text["REDESIGN"]
        for header in EXPECTED_SECTIONS["REDESIGN"]:
            self.assertIn(
                header, text,
                f"REDESIGN missing section: {header}",
            )

    def test_final_verdict_sections(self) -> None:
        text = self.text["FINAL_VERDICT"]
        for header in EXPECTED_SECTIONS["FINAL_VERDICT"]:
            self.assertIn(
                header, text,
                f"FINAL_VERDICT missing section: {header}",
            )

    # --- Test 3: cross-references are wired between the 4 docs ---
    def test_cross_refs_present(self) -> None:
        for src_key, target_wf in CROSS_REF_INVARIANTS:
            text = self.text[src_key]
            self.assertIn(
                target_wf, text,
                f"{src_key} missing cross-ref to {target_wf}",
            )

    # --- Test 4: 8-path inventory aggregates correctly ---
    def test_8_path_inventory_aggregates(self) -> None:
        """The 4 docs collectively cover all 8 m6_kappa attempt paths.

        7 architectural paths: Wf-335-M1, Wf-359-A1, Wf-361-B1, Wf-361-B2,
        Wf-364-A2, Wf-365-A3 (verdict aggregation), Wf-371-A1 (degenerate chain).
        1 metric-redesign path: Wf-377-A1 (16 invariants × n=5460).

        The aggregate is 8 paths; Wf-380-A7 is the finalization closeout (not a path).
        """
        # PAPER_S6 §1.4 explicitly lists 7 paths
        paper_s6 = self.text["PAPER_S6"]
        for path_label in [
            "Wf-335-M1",
            "Wf-359-A1",
            "Wf-361-B1",
            "Wf-361-B2",
            "Wf-364-A2",
            "Wf-365-A3",
            "Wf-371-A1",
        ]:
            self.assertIn(
                path_label, paper_s6,
                f"PAPER_S6 §1.4 7-path inventory missing {path_label}",
            )
        # REDESIGN extends to 8 paths (7 + Wf-377-A1 metric-redesign layer)
        redesign = self.text["REDESIGN"]
        self.assertIn(
            "Wf-377-A1", redesign,
            "REDESIGN must reference itself as the 8th path",
        )
        # 8-path aggregate check: total paths mentioned across docs == 8 unique
        all_paths = {
            "Wf-335-M1",
            "Wf-359-A1",
            "Wf-361-B1",
            "Wf-361-B2",
            "Wf-364-A2",
            "Wf-365-A3",
            "Wf-371-A1",
            "Wf-377-A1",
        }
        for path in all_paths:
            count = sum(
                1 for text in self.text.values() if path in text
            )
            self.assertGreater(
                count, 0,
                f"{path} not mentioned in any of the 4 docs",
            )

    # --- Test 5: §3 main-result line-up invariant ---
    def test_section_3_lineup_invariant(self) -> None:
        """Both Wf-375-A4 and Wf-376-A4 + Wf-377-A1 must recommend DROPPING
        m6_kappa from §3 main-result AND retaining m1_auc + m2 + m3 + m15
        as headline metrics. Wf-377-A1 extends with top-3 supplementary
        sub-table candidates.
        """
        # DROP/Demoted recommendation: accept "DROP m6_kappa" (Wf-375-A4 + Wf-377-A1)
        # OR "Demoted to §6" (Wf-376-A4 wording)
        drop_phrases = ["DROP m6_kappa", "Demoted to §6"]
        headline_metrics = ["m1_auc", "m2", "m3", "m15"]
        for key in ("PAPER_S6", "SECTION_3", "REDESIGN"):
            text = self.text[key]
            self.assertTrue(
                any(p in text for p in drop_phrases),
                f"{key} missing DROP/demote recommendation "
                f"(expected one of: {drop_phrases})",
            )
            # KEEP m1_auc + m2 + m3 + m15 as headline
            for headline_metric in headline_metrics:
                self.assertIn(
                    headline_metric, text,
                    f"{key} missing headline metric reference: {headline_metric}",
                )

    def test_top3_supplementary_promoted(self) -> None:
        """Wf-377-A1 promotes MolLogP + MorganDensity_r2 + Chi3v as top-3
        supplementary sub-table candidates. Verify they appear in REDESIGN.
        """
        redesign = self.text["REDESIGN"]
        for alt in ["MolLogP", "MorganDensity_r2", "Chi3v"]:
            self.assertIn(
                alt, redesign,
                f"REDESIGN missing top-3 alternative: {alt}",
            )

    # --- Test 6: Wf-380-A7 self-reference ---
    def test_w380a7_self_reference(self) -> None:
        """Wf-380-A7 closeout must reference itself and the 4 SHIPPED docs."""
        for key in ("PAPER_S6", "SECTION_3", "REDESIGN", "FINAL_VERDICT"):
            text = self.text[key]
            self.assertIn(
                "Wf-380-A7", text,
                f"{key} missing Wf-380-A7 self-reference (added by this closeout)",
            )

    # --- Test 7: LaTeX template present in §3 snippets ---
    def test_latex_template_present(self) -> None:
        """At least PAPER_S6 §3 + SECTION_3 §3 must contain ready-to-paste
        LaTeX with `\begin{itemize}` block (per task brief gap-fill spec).
        """
        for key in ("PAPER_S6", "SECTION_3"):
            text = self.text[key]
            self.assertIn(
                "\\begin{itemize}", text,
                f"{key} missing \\begin{{itemize}} LaTeX block (per task brief)",
            )
            self.assertIn(
                "\\MEASURED", text,
                f"{key} missing \\MEASURED tag discipline",
            )


if __name__ == "__main__":
    unittest.main()
