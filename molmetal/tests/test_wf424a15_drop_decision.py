"""Test: Wf-424-A15 m6_kappa drop decision convergence verification.

Tests per AGENTS.md §5.5 filesystem-truth rule:
1. docs/M6_KAPPA_DROP_DECISION.md exists on disk + has the 9 sections
2. metrics/by_metric/m6_kappa.json has wf424a15_drop_decision block
3. JSON verdict field is set to MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL
4. JSON paths_exhausted counter matches Wf-424-A15 §1.5 (10 paths)
5. The decision matrix is DROP_FROM_§3_MAIN_RESULT + PROMOTE_TOP3_TO_§3_SUPPLEMENTARY_SUBTABLE + RETAIN_AS_§6_COMPLEMENTARY_HEURISTIC
6. §6 LaTeX snippet for §6.1.6 is verbatim in the paper-conversion doc
7. §6 LaTeX snippet for §6.1.7 is verbatim in the paper-conversion doc
"""

import json
import os
import re
import sys

REPO = "/home/hugo/codes/mol-metal"
DOC_PATH = os.path.join(REPO, "docs/M6_KAPPA_DROP_DECISION.md")
JSON_PATH = os.path.join(REPO, "metrics/by_metric/m6_kappa.json")


def test_paper_conversion_doc_exists():
    """docs/M6_KAPPA_DROP_DECISION.md must exist + have the 9 sections."""
    assert os.path.exists(DOC_PATH), f"MISSING: {DOC_PATH}"
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    # Section count check: 9 sections (§0 TL;DR, §1 Convergence, §2 §3 removal,
    # §3 supplementary sub-table, §4 §6 honest-limitations, §5 conversion
    # walkthrough, §6 ready-to-paste LaTeX, §7 cross-references, §8 final
    # state + §9 standing-rule compliance)
    n_sections = len(re.findall(r"^## §\d+\s", content, flags=re.MULTILINE))
    assert n_sections >= 9, f"expected ≥9 sections, got {n_sections}"
    # §0 TL;DR marker
    assert "## §0 TL;DR" in content, "missing §0 TL;DR"
    # §6 ready-to-paste LaTeX
    assert "## §6 Ready-to-paste LaTeX snippets" in content, "missing §6"
    # §8 final state verdict
    assert "## §8 Final state" in content, "missing §8 final state"
    # §9 standing-rule compliance
    assert "## §9 Standing-rule compliance" in content, "missing §9"
    # ≥400 LOC content (paper-conversion brief requirement)
    n_loc = len(content.splitlines())
    assert n_loc >= 400, f"expected ≥400 LOC, got {n_loc}"


def test_m6_kappa_json_has_wf424a15_block():
    """metrics/by_metric/m6_kappa.json must have the wf424a15_drop_decision block."""
    assert os.path.exists(JSON_PATH), f"MISSING: {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "wf424a15_drop_decision" in data, "missing wf424a15_drop_decision block"
    return data


def test_json_verdict_field_set():
    """The verdict field must be MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL."""
    data = test_m6_kappa_json_has_wf424a15_block()
    assert (
        data["verdict"] == "MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL"
    ), f"verdict mismatch: {data['verdict']}"
    # Verdict field set per Wf-424-A15 §0 TL;DR row
    assert data["wf424a15_drop_decision"]["verdict_field_set"] is True
    assert data["wf424a15_drop_decision"]["verdict_field"] == data["verdict"]


def test_json_paths_exhausted_counter():
    """Wf-424-A15 §1.5 paths_exhausted = 10 (7 architectural + 1 introspection
    + 1 supplementary + 1 per-cohort)."""
    data = test_m6_kappa_json_has_wf424a15_block()
    paths = data["wf424a15_drop_decision"]["paths_exhausted"]
    assert paths == 10, f"paths_exhausted mismatch: {paths}"
    breakdown = data["wf424a15_drop_decision"]["paths_exhausted_breakdown"]
    assert (
        breakdown["architectural"] == 7
    ), f"architectural count mismatch: {breakdown['architectural']}"
    assert (
        breakdown["introspection"] == 1
    ), f"introspection count mismatch: {breakdown['introspection']}"
    assert (
        breakdown["supplementary"] == 1
    ), f"supplementary count mismatch: {breakdown['supplementary']}"
    assert (
        breakdown["per_cohort"] == 1
    ), f"per_cohort count mismatch: {breakdown['per_cohort']}"


def test_json_decision_pillars():
    """The decision must be DROP_FROM_§3_MAIN_RESULT +
    PROMOTE_TOP3_TO_§3_SUPPLEMENTARY_SUBTABLE + RETAIN_AS_§6_COMPLEMENTARY_HEURISTIC."""
    data = test_m6_kappa_json_has_wf424a15_block()
    decision = data["wf424a15_drop_decision"]["decision"]
    expected_pieces = [
        "DROP_FROM_§3_MAIN_RESULT",
        "PROMOTE_TOP3_TO_§3_SUPPLEMENTARY_SUBTABLE",
        "RETAIN_AS_§6_COMPLEMENTARY_HEURISTIC",
    ]
    for piece in expected_pieces:
        assert piece in decision, f"decision missing piece: {piece}; got {decision}"
    pillars = data["wf424a15_drop_decision"]["decision_pillars"]
    assert "drop_from_main_result" in pillars
    assert "promote_top3_to_supplementary_subtable" in pillars
    assert "retain_as_section6_complementary_heuristic" in pillars


def test_json_top3_promotion_cohort_level_evidence():
    """The promote pillar must cite MolLogP +0.1999 + MorganDensity_r2 +0.1644 +
    Chi3v +0.1541 cohort-level evidence (per Wf-377-A1 §3.2)."""
    data = test_m6_kappa_json_has_wf424a15_block()
    cohort_evidence = data["wf424a15_drop_decision"]["decision_pillars"][
        "promote_top3_to_supplementary_subtable"
    ]["cohort_level_evidence"]
    for invariant, expected_r in [
        ("MolLogP", "+0.1999"),
        ("MorganDensity_r2", "+0.1644"),
        ("Chi3v", "+0.1541"),
    ]:
        assert invariant in cohort_evidence, f"missing invariant: {invariant}"
        assert expected_r in cohort_evidence, f"missing expected_r: {expected_r}"


def test_json_per_cohort_evidence_13_15():
    """Per Wf-411-A4, 13/15 cohort-cells MODERATE per Cohen 1988 3-band."""
    data = test_m6_kappa_json_has_wf424a15_block()
    per_cohort_evidence = data["wf424a15_drop_decision"]["decision_pillars"][
        "promote_top3_to_supplementary_subtable"
    ]["per_cohort_evidence"]
    assert (
        "13/15" in per_cohort_evidence
    ), f"per_cohort 13/15 MODERATE evidence missing: {per_cohort_evidence}"
    assert "0 STRONG" in per_cohort_evidence, "STRONG 0 evidence missing"


def test_json_sanity_check_anchor_15_significant_digits():
    """m6_kappa_composite Pearson r bit-for-bit reproducibility: 15 significant digits."""
    data = test_m6_kappa_json_has_wf424a15_block()
    sanity_evidence = data["wf424a15_drop_decision"]["decision_pillars"][
        "retain_as_section6_complementary_heuristic"
    ]["sanity_check_anchor_evidence"]
    assert (
        "15 significant digits" in sanity_evidence
    ), f"15-sig-digit anchor missing: {sanity_evidence}"
    assert (
        "0.08126287494327505" in sanity_evidence
    ), f"MEASURED m6_kappa Pearson r missing: {sanity_evidence}"


def test_doc_section_6_1_6_latex_snippet():
    """§6.1.6 LaTeX snippet must be in §4.2 verbatim (per Wf-410-A1 §3.1.6)."""
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert (
        "label{sec:limitations:m6_kappa:v4}" in content
    ), "missing \\label{sec:limitations:m6_kappa:v4}"
    assert (
        "ten-path honest-negative inventory" in content
    ), "missing §6.1.6 phrase 'ten-path honest-negative inventory'"
    # Verbatim signature from Wf-422-A1 §4.2
    assert (
        "6.1.6 -- m6_kappa 10-path inventory" in content
    ), "missing §6.1.6 LaTeX section header"


def test_doc_section_6_1_7_latex_snippet():
    """§6.1.7 LaTeX snippet must be in §4.3 verbatim (per Wf-377-A1 §4.4)."""
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert (
        "label{sec:limitations:m6_kappa_redesign}" in content
    ), "missing §6.1.7 LaTeX label"
    assert (
        "literature + empirical review" in content
    ), "missing §6.1.7 phrase 'literature + empirical review'"


def test_json_no_paper_touch_and_no_gpu_launch():
    """Wf-424-A15 is docs-only + CPU-only — no paper/* touches + no GPU launches."""
    data = test_m6_kappa_json_has_wf424a15_block()
    assert data["wf424a15_drop_decision"]["no_paper_touch"] is True
    assert data["wf424a15_drop_decision"]["no_gpu_launch"] is True


def test_json_convergence_with_parent_wf():
    """Wf-424-A15 converges with Wf-377-A1 + Wf-389-A1 + Wf-411-A4 + Wf-410-A1 + Wf-365-A3 + Wf-371-A1."""
    data = test_m6_kappa_json_has_wf424a15_block()
    pillars = data["wf424a15_drop_decision"]["verdict_pillars"]
    for key in [
        "convergence_with_wf365a3_final_verdict",
        "convergence_with_wf371a1_degenerate_chain_final",
        "convergence_with_wf377a1_metric_redesign",
        "convergence_with_wf389a1_supplementary_reframe",
        "convergence_with_wf410a1_paper_section6_v4",
        "convergence_with_wf411a4_per_cohort_refinement",
    ]:
        assert pillars[key] is True, f"convergence key {key} not True"
    assert pillars["no_more_paths_to_try_at_wf_424_a15_time"] is True
    assert pillars["is_intrinsic_to_metric"] is True


if __name__ == "__main__":
    # Run tests
    test_paper_conversion_doc_exists()
    test_m6_kappa_json_has_wf424a15_block()
    test_json_verdict_field_set()
    test_json_paths_exhausted_counter()
    test_json_decision_pillars()
    test_json_top3_promotion_cohort_level_evidence()
    test_json_per_cohort_evidence_13_15()
    test_json_sanity_check_anchor_15_significant_digits()
    test_doc_section_6_1_6_latex_snippet()
    test_doc_section_6_1_7_latex_snippet()
    test_json_no_paper_touch_and_no_gpu_launch()
    test_json_convergence_with_parent_wf()
    print("ALL TESTS PASSED")
