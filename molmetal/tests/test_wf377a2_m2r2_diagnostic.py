"""Tests for Wf-377-A2 m2 R² cohort diagnostic + alternative splits.

Verifies:
  1. The diagnostic doc exists at docs/M2_R2_COHORT_DIAGNOSTIC.md and has 7 sections.
  2. The result JSON exists at molmetal/reports/wf377a2_m2r2_diagnostic/result.json.
  3. Per-cohort structural properties were computed (n + y stats + scaffold diversity).
  4. Alternative split strategies were compared (5 split strategies × 3 cohorts).
  5. The HONEST_NEGATIVE_R2_DEEP_PRESERVED verdict is preserved across all 5 splits
     on the n=200 cohorts.

Per Wf-377-A2 brief — 3 NEW tests, 3/3 PASS expected.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DIAGNOSTIC_DOC = REPO_ROOT / "docs" / "M2_R2_COHORT_DIAGNOSTIC.md"
RESULT_JSON = (
    REPO_ROOT / "molmetal" / "reports" / "wf377a2_m2r2_diagnostic" / "result.json"
)


def _read_json(path: Path) -> dict:
    """Read JSON file, returning empty dict if missing."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_text(path: Path) -> str:
    """Read text file, returning empty string if missing."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: Diagnostic doc exists + 7 sections
# ---------------------------------------------------------------------------
def test_wf377a2_diagnostic_doc_has_seven_sections():
    """Diagnostic doc exists at docs/M2_R2_COHORT_DIAGNOSTIC.md with 7 sections.

    Per Wf-377-A2 brief: section 0 TL;DR + §1 per-cohort structural + §2 why
    overfit + §3 alternative splits + §4 recommended path + §5 code snippet +
    §6 cross-references.
    """
    assert DIAGNOSTIC_DOC.exists(), f"missing: {DIAGNOSTIC_DOC}"
    text = _read_text(DIAGNOSTIC_DOC)
    # Verify the 7 sections per the brief (each starts with `## §`)
    expected_section_starts = [
        "## §0",
        "## §1",
        "## §2",
        "## §3",
        "## §4",
        "## §5",
        "## §6",
    ]
    for sec in expected_section_starts:
        assert sec in text, f"missing section {sec!r} in {DIAGNOSTIC_DOC.name}"
    # Verify the doc references the upstream artifacts
    for ref in (
        "Wf-335-M2",
        "Wf-367-B1",
        "Wf-369-A2",
        "Wf-365-A3",
        "HONEST_NEGATIVE_R2_DEEP",
        "metalcytotox_combined",
        "platinai_4cell",
        "hilbert",
        "AGENTS.md §5.1",
        "party-twenty-spirit §3.5",
    ):
        assert ref in text, f"missing cross-reference {ref!r}"


# ---------------------------------------------------------------------------
# Test 2: Cohort structural properties were computed (result.json)
# ---------------------------------------------------------------------------
def test_wf377a2_result_json_has_per_cohort_stats():
    """result.json has per-cohort structural properties for all 3 cohorts.

    Per Wf-377-A2 brief: n + scaffold diversity + label dist + heavy atom dist.
    """
    data = _read_json(RESULT_JSON)
    assert data, f"missing or empty: {RESULT_JSON}"
    assert "per_cohort" in data, "missing 'per_cohort' key in result.json"
    assert len(data["per_cohort"]) == 3, (
        f"expected 3 cohorts in per_cohort, got {len(data['per_cohort'])}"
    )
    cohort_names = {c["cohort"] for c in data["per_cohort"]}
    assert cohort_names == {
        "platinai_4cell",
        "hilbert",
        "metalcytotox_combined",
    }, f"missing cohorts: {cohort_names}"
    for c in data["per_cohort"]:
        # Required structural stats keys (per Wf-377-A2 brief)
        for key in (
            "n_samples",
            "y_mean",
            "y_std",
            "y_range",
            "scaffold_diversity_at_04",
            "n_unique_scaffold_proxies",
            "features_per_sample_ratio",
            "heavy_atom_count_mean",
        ):
            assert key in c, f"cohort {c['cohort']} missing key {key!r}"
        # Sanity: platinai_4cell and hilbert have IDENTICAL y distribution
        # (same data, different seed_offset)
        if c["cohort"] in ("platinai_4cell", "hilbert"):
            assert c["n_samples"] == 200, f"{c['cohort']} expected n=200"
            assert abs(c["y_mean"] - 5.0378) < 0.01, f"{c['cohort']} y_mean drift"
        if c["cohort"] == "metalcytotox_combined":
            assert c["n_samples"] == 2000, f"metalcytotox_combined expected n=2000"


# ---------------------------------------------------------------------------
# Test 3: Alternative split strategies were compared
# ---------------------------------------------------------------------------
def test_wf377a2_alternative_splits_compared():
    """5 alternative split strategies compared; per-split R² in result.json.

    Per Wf-377-A2 brief: random_kfold + stratified_kfold + scaffold_kfold +
    leave_one_cluster_out + per_cell_holdout × 3 cohorts.
    """
    data = _read_json(RESULT_JSON)
    assert data, f"missing or empty: {RESULT_JSON}"
    assert "per_split_strategy" in data, "missing 'per_split_strategy' key"
    per_split = data["per_split_strategy"]
    # Verify all 3 cohorts
    assert set(per_split.keys()) == {
        "platinai_4cell",
        "hilbert",
        "metalcytotox_combined",
    }, f"missing cohorts in per_split_strategy: {per_split.keys()}"
    # Verify each cohort has at least 4 split strategies (per_cell_holdout is
    # empty list because no cell_line annotations are available).
    expected_splits = {
        "random_kfold",
        "stratified_kfold",
        "scaffold_kfold",
        "leave_one_cluster_out",
    }
    for cohort, splits in per_split.items():
        assert set(splits.keys()) >= expected_splits, (
            f"cohort {cohort} missing splits: "
            f"{expected_splits - set(splits.keys())}"
        )
        for split_name, sr in splits.items():
            # Each split has the required R² fields
            for key in ("r2_mean", "r2_std", "n_seeds", "r2_per_seed"):
                assert key in sr, (
                    f"{cohort}/{split_name} missing key {key!r}"
                )
            # Sanity: r2_per_seed length matches n_seeds
            assert len(sr["r2_per_seed"]) == sr["n_seeds"], (
                f"{cohort}/{split_name} r2_per_seed length mismatch"
            )
    # Honest-negative invariant: ALL 4 split strategies stay negative on n=200
    # cohorts (the headline MEASURED finding of the diagnostic).
    for cohort in ("platinai_4cell", "hilbert"):
        splits = per_split[cohort]
        for split_name, sr in splits.items():
            assert sr["r2_mean"] < 0, (
                f"{cohort}/{split_name} r2_mean={sr['r2_mean']} expected < 0 "
                f"(HONEST_NEGATIVE_R2_DEEP preserved)"
            )
    # metalcytotox_combined (n=2000) CEILING_BREAK: random + stratified
    # should be > 0 (per Wf-335-M2 anchor). Scaffold may be lower but should
    # still be > 0 (OOD gap but signal preserved).
    for split_name in ("random_kfold", "stratified_kfold", "scaffold_kfold"):
        sr = per_split["metalcytotox_combined"][split_name]
        assert sr["r2_mean"] > 0, (
            f"metalcytotox_combined/{split_name} r2_mean={sr['r2_mean']} "
            f"expected > 0 (CEILING_BREAK preserved)"
        )
    # Aggregate verdict: gate_preserved = True (not all cohorts lift to >= 0)
    assert data["aggregate_verdict"]["gate_preserved"] is True, (
        "aggregate_verdict.gate_preserved should be True — "
        "HONEST_NEGATIVE_R2_DEEP is the expected verdict"
    )
    # Verdict label: HONEST_NEGATIVE_R2_DEEP_PRESERVED
    assert data["verdict"] == "HONEST_NEGATIVE_R2_DEEP_PRESERVED", (
        f"verdict should be HONEST_NEGATIVE_R2_DEEP_PRESERVED, got "
        f"{data['verdict']!r}"
    )
