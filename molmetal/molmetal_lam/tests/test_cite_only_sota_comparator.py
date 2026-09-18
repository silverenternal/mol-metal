"""Tests for ``molmetal_lam.sbdd_env.cite_only_sota_comparator``.

Background
----------
TODO-22 + TODO-25 (d) ships a cite-only SOTA comparator helper for
the paper §4 SOTA-comparison column.  The comparator must:

1. Load 9 SOTA rows from ``wf_3_citeonly_sota.tex`` (parse + hardcoded)
2. Compute per-row gaps (our_value - target_value)
3. Set protocol-mismatch flags (M1)-(M7)
4. Return the same dataclass shape across runs

These tests are CPU-only and hermetic — no real SOTA baselines are
re-run, no LaTeX compiler is invoked, no GPU/docking binaries are
needed.

Honest framing
--------------
* 9 SOTA rows are CITED-ONLY; tests verify the citation shape, not
  whether the values match a re-run (they cannot, by design).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Path bootstrap so the test can import molmetal_lam when invoked
# from project root via ``uv run pytest`` or from any cwd.
_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from molmetal_lam.sbdd_env.cite_only_sota_comparator import (  # noqa: E402
    CiteOnlySOTAComparator,
    SOTARow,
    ProtocolMismatchFlag,
    ComparisonReport,
    GapMetrics,
    SOTA_ROWS,
    PROTOCOL_MISMATCH_FLAGS,
    DEFAULT_PROTOCOL_FLAGS,
    _parse_tex_rows,
    build_per_pocket_comparison_rows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
TEX_PATH = ("/home/hugo/codes/try_triton_on_rocm/molmetal/reports/"
            "wf_3_citeonly_sota.tex")


@pytest.fixture
def tex_path() -> Path:
    """Path to the cite-only SOTA LaTeX table."""
    return Path(TEX_PATH)


@pytest.fixture
def default_protocol_flags() -> dict:
    """A protocol_flags dict that matches all 9 SOTA rows exactly.

    This is the "round-13 fair comparison" protocol: Luo 2021 split,
    Vina 1.2.7, RDKit-only validity, Tanimoto <0.4 novelty,
    Tanimoto (Morgan r=2) diversity, n_seeds=1, n_pockets=100.
    """
    return {
        "crossdocked_luo2021": True,
        "docking_engine": "qvina",
        "validity_def": "RDKit-only",
        "novelty_def": "Tanimoto <0.4",
        "diversity_def": "Tanimoto (Morgan r=2)",
        "n_seeds": 1,
        "n_pockets": 100,
    }


# ---------------------------------------------------------------------------
# Test 1: Parser handles the 9-row LaTeX table
# ---------------------------------------------------------------------------
def test_parser_handles_9_row_latex_table(tex_path: Path) -> None:
    """``_parse_tex_rows`` must extract 9 data rows from the cite-only
    SOTA .tex table.

    The cite-only table has 9 SOTA rows + 2 Mol-Metal rows + 1
    separator ``\\hline``; the regex matches exactly the 9 SOTA
    rows (the Mol-Metal rows have a different prefix ``\\textbf``
    that breaks the vina_mean regex).

    If this test fails the .tex was hand-edited and the parser
    needs to be updated.  The hardcoded inventory is the
    authoritative source regardless.
    """
    if not tex_path.is_file():
        pytest.skip(f"Cite-only SOTA .tex not present at {tex_path}")
    rows = _parse_tex_rows(str(tex_path))
    assert len(rows) == 9, f"expected 9 SOTA rows, got {len(rows)}"
    # First row is DiffSBDD, last is RoseTTAFold-AA (per .tex ordering).
    assert "DiffSBDD" in rows[0]["paper"]
    assert "RoseTTAFold" in rows[-1]["paper"]
    # Each row has all 11 expected keys.
    for row in rows:
        assert "paper" in row
        assert "dataset" in row
        assert "engine" in row
        assert "vina" in row
        assert "n_pockets" in row


# ---------------------------------------------------------------------------
# Test 2: Hardcoded inventory has 9 rows + 7 flags
# ---------------------------------------------------------------------------
def test_hardcoded_inventory_shape() -> None:
    """Hardcoded SOTA_ROWS / PROTOCOL_MISMATCH_FLAGS have the expected shapes."""
    assert len(SOTA_ROWS) == 9, f"expected 9 SOTA rows, got {len(SOTA_ROWS)}"
    assert len(PROTOCOL_MISMATCH_FLAGS) == 7, (
        f"expected 7 protocol-mismatch flags M1-M7, got {len(PROTOCOL_MISMATCH_FLAGS)}"
    )
    codes = {f.code for f in PROTOCOL_MISMATCH_FLAGS}
    expected_codes = {f"M{i}" for i in range(1, 8)}
    assert codes == expected_codes, f"flag codes mismatch: {codes} vs {expected_codes}"
    for row in SOTA_ROWS:
        assert row.n_pockets > 0
        assert row.n_seeds >= 1
        assert row.paper, "row.paper must be non-empty"
        assert not row.is_ours, "SOTA rows are CITED-ONLY, is_ours=False"


# ---------------------------------------------------------------------------
# Test 3: Gap calculation correct (our_value - target_value)
# ---------------------------------------------------------------------------
def test_gap_calculation_our_minus_target(default_protocol_flags: dict) -> None:
    """Gap is computed as our_value - target_value.

    For Vina (lower is better), a positive gap means our_value is
    LESS negative than the target (i.e. we are WORSE on binding
    affinity).  A negative gap means we are BETTER.
    """
    cmp = CiteOnlySOTAComparator()
    targetdiff = cmp.row_by_name("TargetDiff")
    assert targetdiff is not None
    assert targetdiff.vina_mean == -8.45

    # Our value is -7.0 (worse than TargetDiff's -8.45).
    # Expected gap = -7.0 - (-8.45) = +1.45 (positive = worse).
    reports = cmp.compare(our_value=-7.0, num_samples=100,
                          protocol_flags=default_protocol_flags,
                          row_name="TargetDiff")
    assert len(reports) == 1
    gap = reports[0].gap
    assert gap.vina == pytest.approx(1.45, abs=1e-6), (
        f"expected gap=+1.45, got {gap.vina}"
    )

    # Our value is -9.0 (better than TargetDiff's -8.45).
    # Expected gap = -9.0 - (-8.45) = -0.55 (negative = better).
    reports = cmp.compare(our_value=-9.0, num_samples=100,
                          protocol_flags=default_protocol_flags,
                          row_name="TargetDiff")
    assert reports[0].gap.vina == pytest.approx(-0.55, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 4: Protocol-mismatch flag set is correct
# ---------------------------------------------------------------------------
def test_protocol_mismatch_flags_set(default_protocol_flags: dict) -> None:
    """With the Luo 2021 + qvina + RDKit-only protocol, FLOWr should
    trigger M1 (different split: SPINDR) and M2 (different engine:
    DiffDock-L).  TargetDiff should match all flags.
    """
    cmp = CiteOnlySOTAComparator()

    # TargetDiff should match all 7 flags with default_protocol_flags.
    targetdiff_reports = cmp.compare(
        our_value=-7.0, num_samples=100,
        protocol_flags=default_protocol_flags,
        row_name="TargetDiff",
    )
    targetdiff_flags = targetdiff_reports[0].applicable_flags
    assert targetdiff_flags == [], (
        f"TargetDiff should match default protocol_flags; got {targetdiff_flags}"
    )
    assert targetdiff_reports[0].fairness_verdict == "comparable"

    # FLOWr uses SPINDR split + DiffDock-L → M1 + M2.
    flowr_reports = cmp.compare(
        our_value=-7.0, num_samples=100,
        protocol_flags=default_protocol_flags,
        row_name="FLOWr",
    )
    flowr_flags = set(flowr_reports[0].applicable_flags)
    assert "M1" in flowr_flags, f"FLOWr should trigger M1; got {flowr_flags}"
    assert "M2" in flowr_flags, f"FLOWr should trigger M2; got {flowr_flags}"


def test_seed_mismatch_triggers_m6() -> None:
    """M6 (n_seeds mismatch) fires when our n_seeds != row.n_seeds.

    Round-13 protocol uses n_seeds=3; rows with n_seeds=1 must
    trigger M6.
    """
    cmp = CiteOnlySOTAComparator()
    flags_3seeds = dict(DEFAULT_PROTOCOL_FLAGS)
    flags_3seeds["n_seeds"] = 3
    reports = cmp.compare(our_value=-7.0, num_samples=100,
                          protocol_flags=flags_3seeds, row_name="TargetDiff")
    assert "M6" in reports[0].applicable_flags


# ---------------------------------------------------------------------------
# Test 5: Dataclass shape consistency across runs
# ---------------------------------------------------------------------------
def test_dataclass_shape_consistent_across_runs(default_protocol_flags: dict) -> None:
    """Two consecutive ``compare`` calls return ComparisonReport objects
    with identical dataclass shape (same fields, same types).
    """
    cmp = CiteOnlySOTAComparator()
    reports_a = cmp.compare(our_value=-7.0, num_samples=100,
                            protocol_flags=default_protocol_flags)
    reports_b = cmp.compare(our_value=-8.5, num_samples=200,
                            protocol_flags=default_protocol_flags)

    assert len(reports_a) == len(reports_b) == 9
    for ra, rb in zip(reports_a, reports_b):
        assert type(ra) is type(rb) is ComparisonReport
        assert isinstance(ra.gap, GapMetrics)
        assert isinstance(ra.row, SOTARow)
        # All ComparisonReport fields must be present.
        for field_name in ("row", "our_value", "gap", "applicable_flags",
                           "n_samples_ours", "n_samples_target",
                           "fairness_verdict", "notes"):
            assert hasattr(ra, field_name), f"missing field {field_name}"
            assert hasattr(rb, field_name), f"missing field {field_name}"
        # To-dict round-trip.
        ra_dict = ra.to_dict()
        rb_dict = rb.to_dict()
        assert set(ra_dict.keys()) == set(rb_dict.keys())
        # Verdict is one of three valid values.
        assert ra.fairness_verdict in ("comparable", "flag_only", "incomparable")


# ---------------------------------------------------------------------------
# Test 6: Comparator emit per-pocket CSV-ready rows
# ---------------------------------------------------------------------------
def test_compare_per_pocket_emits_csv_ready_rows(default_protocol_flags: dict) -> None:
    """``compare_per_pocket`` returns list of dicts with the CSV-ready
    keys (pocket_id, paper, vina_target, gap_vina, applicable_flags,
    fairness_verdict).
    """
    cmp = CiteOnlySOTAComparator()
    rows = cmp.compare_per_pocket("test_001", our_value=-7.5, num_samples=100,
                                  protocol_flags=default_protocol_flags)
    assert len(rows) == 9
    expected_keys = {"pocket_id", "paper", "dataset", "vina_target",
                     "vina_std_target", "gap_vina", "n_samples_ours",
                     "n_samples_target", "applicable_flags", "fairness_verdict",
                     "notes"}
    for row in rows:
        assert expected_keys.issubset(row.keys()), (
            f"missing keys: {expected_keys - row.keys()}"
        )
        assert row["pocket_id"] == "test_001"


# ---------------------------------------------------------------------------
# Test 7: Convenience helper ``build_per_pocket_comparison_rows``
# ---------------------------------------------------------------------------
def test_build_per_pocket_comparison_rows_helper(default_protocol_flags: dict) -> None:
    """The one-shot helper ``build_per_pocket_comparison_rows`` returns
    the same shape as ``compare_per_pocket`` and is suitable for
    direct integration with ``r4_c_full_sweep.py``.
    """
    rows = build_per_pocket_comparison_rows("1h36", our_value=-5.923,
                                           num_samples=50,
                                           protocol_flags=default_protocol_flags)
    assert len(rows) == 9
    targetdiff_row = next(r for r in rows if "TargetDiff" in r["paper"])
    # -5.923 - (-8.45) = +2.527 (we are worse than TargetDiff; single pocket 1h36)
    assert targetdiff_row["gap_vina"] == pytest.approx(2.527, abs=1e-3)


# ---------------------------------------------------------------------------
# Test 8: Vina-undefined rows (RMSD-to-native) return None gap
# ---------------------------------------------------------------------------
def test_rmsd_only_rows_have_no_vina_gap(default_protocol_flags: dict) -> None:
    """DiffDock and BindNet use RMSD-to-native not Vina; gap.vina
    must be None for those rows (not a float).
    """
    cmp = CiteOnlySOTAComparator()
    reports = cmp.compare(our_value=-7.0, num_samples=100,
                          protocol_flags=default_protocol_flags,
                          row_name="DiffDock")
    assert len(reports) == 1
    assert reports[0].row.vina_mean is None
    assert reports[0].gap.vina is None
    # fairness_verdict should not be 'comparable' when no Vina target.
    assert reports[0].fairness_verdict != "comparable"
