"""Wf-377-A1 — m6_kappa REDESIGN proposal + empirical validation tests.

Per Wf-377-A1 task brief + user directive 2026-09-30:
"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"
(If the same problem has not been fixed for a long time, do research + review
to attempt deeper fixes).

This test module verifies:
1. The m6_kappa redesign proposal doc exists with 7 sections + has the
   cohort-level ranking table.
2. The empirical validation was actually run (result.json exists with
   metrics for all 16 invariants).
3. The top-3 cohort-level alternatives (MolLogP + MorganDensity_r2 + Chi3v)
   have stronger Pearson r than Shape Kappa Index (m6_kappa_composite).

Sanity-check anchor: m6_kappa_composite Pearson r must reproduce Wf-325D
canonical value 0.08126287494327507 to >= 13 significant digits.

CPU-only per AGENTS.md §5.1 (no GPU launches).
0 paper/* touches per AGENTS.md §5.3 + TODO-84.

Author: Hugo Lee + Wf-377-A1 ultracode.
Date: 2026-09-30.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROPOSAL_DOC = REPO_ROOT / "docs" / "M6_KAPPA_REDESIGN_PROPOSAL.md"
RESULT_JSON = (
    REPO_ROOT / "molmetal" / "reports" / "wf377a1_m6_kappa_redesign" / "result.json"
)

# Wf-325D canonical anchor (metrics/by_metric/m6_kappa.json::metalcytotox_validation.global_pearson_r)
WF325D_CANONICAL_PEARSON_R = 0.08126287494327507
# Tolerance for bit-for-bit reproduction of Wf-325D canonical
BIT_FOR_BIT_TOL = 1e-13

# 7 canonical section headers in the proposal doc (per task brief §0)
EXPECTED_SECTIONS = [
    "## §0 TL;DR",
    "## §1 Why m6_kappa fails",
    "## §2 Literature review",
    "## §3 Empirical validation",
    "## §4 Recommended metric replacement",
    "## §5 Cross-references",
    "## §6 Final state",
]

# Per the task brief, 16 invariants are validated
EXPECTED_INVARIANTS = {
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "TPSA",
    "MolLogP",
    "MolMR",
    "BalabanJ",
    "HallKierAlpha",
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "Chi3v",
    "Phi",
    "MorganDensity_r2",
    "m6_kappa_composite",
}

# Top-3 alternatives per §3.2 cohort-level ranking
TOP3_NAMES = ["MolLogP", "MorganDensity_r2", "Chi3v"]


class TestWf377A1M6KappaRedesign(unittest.TestCase):
    """3 test classes for the 3 verification dimensions."""

    def test_proposal_file_exists_with_7_sections_and_table(self) -> None:
        """Verification 1: proposal file exists + has 7 sections + has ranking table."""
        self.assertTrue(
            PROPOSAL_DOC.exists(),
            f"Proposal doc missing: {PROPOSAL_DOC}",
        )
        content = PROPOSAL_DOC.read_text(encoding="utf-8")
        # 7 sections per task brief
        for section in EXPECTED_SECTIONS:
            self.assertIn(
                section,
                content,
                f"Missing expected section header: {section}",
            )
        # §7 standing-rule compliance is also present (8 sections total — §0..§7)
        self.assertIn("## §7 Standing-rule compliance", content)
        # Ranking table with top-3 names
        for name in TOP3_NAMES:
            self.assertIn(
                name,
                content,
                f"Top-3 invariant {name} missing from proposal doc",
            )
        # m6_kappa_composite (the anchor) must appear
        self.assertIn("m6_kappa_composite", content)
        # Verify document is non-trivial (>300 LOC per task brief ~400 LOC target)
        n_lines = content.count("\n")
        self.assertGreater(
            n_lines,
            300,
            f"Proposal doc too short: {n_lines} lines (expected >= 300)",
        )
        # Verify the bit-for-bit anchor is cited
        self.assertIn(
            "0.08126287494327507",
            content,
            "Wf-325D canonical anchor Pearson r not cited in proposal doc",
        )

    def test_empirical_validation_was_actually_run(self) -> None:
        """Verification 2: result.json exists + contains all 16 invariants + has metrics."""
        self.assertTrue(
            RESULT_JSON.exists(),
            f"Result JSON missing: {RESULT_JSON}",
        )
        result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
        # Verify all 16 invariants are present
        ranking = result.get("ranking", [])
        names_in_result = {r["name"] for r in ranking}
        missing = EXPECTED_INVARIANTS - names_in_result
        self.assertEqual(
            missing,
            set(),
            f"Missing invariants in result.json: {missing}",
        )
        # Verify each has Pearson r + Spearman rho + verdict
        for entry in ranking:
            self.assertIn("name", entry)
            self.assertIn("pearson_r", entry)
            self.assertIn("spearman_rho", entry)
            self.assertIn("verdict", entry)
            self.assertIn("n_used", entry)
            if entry["pearson_r"] is not None:
                self.assertIsInstance(entry["pearson_r"], (int, float))
                self.assertIsInstance(entry["spearman_rho"], (int, float))
        # Verify cohort size n=5460 per Wf-325D canonical
        self.assertEqual(
            result.get("n_loaded"),
            5460,
            f"Cohort size mismatch: {result.get('n_loaded')} (expected 5460)",
        )
        self.assertEqual(
            result.get("n_parsed_rdkit"),
            5460,
            f"RDKit parse count mismatch: {result.get('n_parsed_rdkit')}",
        )
        # Verify per-metal breakdown for 5 metals (Ru, Ir, Re, Os, Rh)
        self.assertIn("per_metal_breakdown", result)
        breakdown = result["per_metal_breakdown"]
        expected_metals = {"Ru", "Ir", "Re", "Os", "Rh"}
        actual_metals = set(breakdown.keys())
        self.assertEqual(
            actual_metals,
            expected_metals,
            f"Per-metal breakdown mismatch: {actual_metals} (expected {expected_metals})",
        )
        # Sanity-check anchor: m6_kappa_composite bit-for-bit reproducibility
        m6k_entry = next(
            (r for r in ranking if r["name"] == "m6_kappa_composite"), None
        )
        self.assertIsNotNone(m6k_entry, "m6_kappa_composite missing from ranking")
        measured_pearson_r = m6k_entry["pearson_r"]
        self.assertAlmostEqual(
            measured_pearson_r,
            WF325D_CANONICAL_PEARSON_R,
            delta=BIT_FOR_BIT_TOL,
            msg=(
                f"m6_kappa_composite Pearson r = {measured_pearson_r:.20f} "
                f"does NOT reproduce Wf-325D canonical {WF325D_CANONICAL_PEARSON_R:.20f} "
                f"to {BIT_FOR_BIT_TOL:.0e} tolerance — validation pipeline corrupted"
            ),
        )

    def test_top3_alternatives_have_stronger_correlation_than_shape_kappa_index(
        self,
    ) -> None:
        """Verification 3: top-3 alternatives (MolLogP + MorganDensity_r2 + Chi3v)
        have stronger |Pearson r| than m6_kappa_composite.
        """
        result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
        ranking = result["ranking"]
        by_name = {r["name"]: r for r in ranking}

        # m6_kappa_composite baseline
        m6k = by_name["m6_kappa_composite"]
        m6k_abs_r = abs(m6k["pearson_r"])

        # Verify each top-3 alternative has stronger absolute Pearson r
        for name in TOP3_NAMES:
            self.assertIn(name, by_name, f"Top-3 invariant {name} missing")
            entry = by_name[name]
            alt_abs_r = abs(entry["pearson_r"])
            self.assertGreater(
                alt_abs_r,
                m6k_abs_r,
                msg=(
                    f"{name} |Pearson r| = {alt_abs_r:.4f} is NOT stronger than "
                    f"m6_kappa_composite |r| = {m6k_abs_r:.4f}"
                ),
            )
            # All top-3 should be in WEAK band (0.1 <= |r| < 0.3)
            self.assertGreaterEqual(
                alt_abs_r,
                0.1,
                msg=f"{name} |Pearson r| = {alt_abs_r:.4f} should be >= 0.1 (WEAK band)",
            )

        # Verify top-3 alternatives are ranked 1-2-3 (or top-3) by |Pearson r|
        # (the ranking is sorted by abs_pearson_r descending in the validator script)
        # Top entry in ranking should be MolLogP (per §3.2)
        self.assertEqual(
            ranking[0]["name"],
            "MolLogP",
            f"Top-ranked invariant should be MolLogP, got {ranking[0]['name']}",
        )
        # The 2nd and 3rd ranked are MorganDensity_r2 and Chi3v (in some order)
        names_in_top3 = {ranking[i]["name"] for i in range(3)}
        expected_top3 = {"MolLogP", "MorganDensity_r2", "Chi3v"}
        self.assertEqual(
            names_in_top3,
            expected_top3,
            f"Top-3 ranking mismatch: {names_in_top3} (expected {expected_top3})",
        )

        # Verify m6_kappa_composite is NOT in top-3 (it's at rank 9)
        # (i.e. the ranking correctly identifies it as inferior)
        for i in range(3):
            self.assertNotEqual(
                ranking[i]["name"],
                "m6_kappa_composite",
                "m6_kappa_composite should NOT be in top-3 — it's NEGLIGIBLE band",
            )

        # Verify the structural ordering: m6_kappa_composite should rank BELOW all top-3
        m6k_rank = next(
            i for i, r in enumerate(ranking) if r["name"] == "m6_kappa_composite"
        )
        self.assertGreater(
            m6k_rank,
            2,
            f"m6_kappa_composite rank = {m6k_rank} should be > 2 (i.e. NOT in top-3)",
        )


if __name__ == "__main__":
    unittest.main()
