"""Wf-264A-Updater unit tests — chemprop v3 DrugOOD post-fix doc updater.

This module covers the public surface of
``molmetal/scripts/wf264a_drugood_post_fix_updater.py``:

  1. test_module_imports_and_public_symbols
     The module imports cleanly + exposes the public symbols the rest
     of the test surface (and downstream callers) rely on.

  2. test_argparse_flag_count
     ``_build_argparser()`` exposes >= 4 flags (G2 acceptance gate).

  3. test_dry_run_default_true
     Default ``--dry-run`` is True (backward-compat gate G3).

  4. test_atomic_write_text_round_trip
     ``_atomic_write_text`` writes + overwrites an existing file with
     no leftover temp files (gate G4 atomic-update contract).

  5. test_parse_v3_measured_extracts_per_split_aucs
     ``parse_v3_measured`` pulls the per-split chemprop v3-fallback
     MEASURED numbers from a Wf-263A-Verify aggregate.json fixture.

  6. test_determine_promotion_above_v2_baseline
     ``determine_promotion`` returns True when assay split passes_v2_baseline.

  7. test_determine_promotion_below_v2_baseline
     ``determine_promotion`` returns False when assay split fails.

  8. test_update_207_replaces_projected_with_measured
     ``update_207_paragraph`` rewrites the §9.5 "projected ≥ v2 baseline"
     sentence with MEASURED numbers.

  9. test_update_207_no_match_when_sentence_missing
     ``update_207_paragraph`` returns applied=False when the doc structure
     has drifted (catastrophic-failure signal).

 10. test_update_188_appends_lead_paragraph_when_promoted
     ``update_188_paragraph`` appends a §4 LEAD paragraph when promotion
     criteria are met (assay split passes_v2_baseline).

 11. test_update_188_no_touch_when_not_promoted
     ``update_188_paragraph`` returns applied=False + skip_reason when
     v3 MEASURED is below v2 baseline (honest-negative, 188 untouched).

 12. test_end_to_end_dry_run_writes_only_report
     ``main()`` in dry-run mode writes ONLY the post_fix_update_report.json;
     neither the 207 nor the 188 doc is touched.

 13. test_end_to_end_apply_writes_207_only_when_not_promoted
     ``main()`` with ``--no-dry-run`` writes the updated 207 doc but
     leaves the 188 doc untouched when v3 is NOT promoted.

 14. test_end_to_end_apply_writes_both_docs_when_promoted
     ``main()`` with ``--no-dry-run`` writes BOTH docs when v3 IS
     promoted (assay split passes_v2_baseline).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from molmetal.scripts import wf264a_drugood_post_fix_updater as up


# ---------------------------------------------------------------------------
# Fixtures: Wf-263A-Verify aggregate JSON shapes (above-v2 + below-v2).
# ---------------------------------------------------------------------------
def _make_aggregate_v3_above_v2() -> dict:
    """Aggregate JSON where chemprop v3-fallback MEASURED > v2 baseline."""
    return {
        "date": "2026-09-28",
        "wf": "wf263a-verify",
        "chemprop_mode": "v3-fallback",
        "gate_1_all_ran": True,
        "gate_2_total_runs": 15,
        "gate_4_per_config_summary": {
            "lbap_refined_ic50_assay": {"mean": 0.7250, "std": 0.0062, "n": 5},
            "lbap_refined_ic50_scaffold": {"mean": 0.6789, "std": 0.0040, "n": 5},
            "lbap_refined_ic50_size": {"mean": 0.6812, "std": 0.0120, "n": 5},
        },
        "gate_5_v3_vs_v2_comparison": {
            "lbap_refined_ic50_assay": {
                "v2_baseline_mean": 0.7137,
                "v2_baseline_std": 0.0079,
                "v3_fallback_mean": 0.7250,
                "delta": 0.0113,
                "passes_v2_baseline": True,
            },
            "lbap_refined_ic50_scaffold": {
                "v2_baseline_mean": 0.6707,
                "v2_baseline_std": 0.0042,
                "v3_fallback_mean": 0.6789,
                "delta": 0.0082,
                "passes_v2_baseline": True,
            },
            "lbap_refined_ic50_size": {
                "v2_baseline_mean": 0.6757,
                "v2_baseline_std": 0.0142,
                "v3_fallback_mean": 0.6812,
                "delta": 0.0055,
                "passes_v2_baseline": True,
            },
        },
        "gate_6_overall_v3_fallback_matches_v2": True,
        "gate_7_total_wall_min": 80.5,
    }


def _make_aggregate_v3_below_v2() -> dict:
    """Aggregate JSON where chemprop v3-fallback MEASURED < v2 baseline."""
    agg = _make_aggregate_v3_above_v2()
    agg["gate_4_per_config_summary"]["lbap_refined_ic50_assay"] = {
        "mean": 0.6812,
        "std": 0.0095,
        "n": 5,
    }
    agg["gate_5_v3_vs_v2_comparison"]["lbap_refined_ic50_assay"] = {
        "v2_baseline_mean": 0.7137,
        "v2_baseline_std": 0.0079,
        "v3_fallback_mean": 0.6812,
        "delta": -0.0325,
        "passes_v2_baseline": False,
    }
    agg["gate_6_overall_v3_fallback_matches_v2"] = False
    return agg


DOC_207_TEMPLATE = """# TODO-207 — Unified Pre-Flight Checklist v2

## §9 Honest framing catalog (party-twenty-spirit §3.5)

### §9.5 Wf-263A v3 fallback to v2 BinaryClassificationFFN (fix for §9.4)

- **Wf-263A default disables EnhancedMetalHead + metal-attention**; v3 falls back to v2 BinaryClassificationFFN
- **v3 OOD AUC projected ≥ v2 baseline (0.7137 assay / 0.6707 scaffold / 0.6757 size)**; **MEASURED per Wf-263A verify phase**
- Root-cause attribution: the v3 innovation surface is the suspected ROOT CAUSE of BELOW_RANDOM 0.4544 OOD AUC

End of section.
"""

DOC_188_TEMPLATE = """# 188 — Paper §4 Final Draft

### §0.1 One-line verdict

> **On the real Ji 2023 DrugOOD IC50 lbap_refined_ic50_assay split**, **chemprop v2** (hidden=128, depth=2, 20 epochs, 5 seeds × 3 seeds CPU) achieves **ood_test AUC = 0.7113 ± 0.0216 mean across 3 seeds** per Wf-174 and **0.7137 ± 0.0079 mean across 5 seeds** per Wf-185.
>
> **Secondary metrics**: m14 real CFM decode_ratio 0.9958 (n=26,801) per Wf-188. **Wf-188 m14 FULL MetalCytoToxDB decode_ratio** 0.9958 MEASURED on n=26,801 (gate_4_decode_ratio=0.9958).
>
> Closing remark.

End of doc.
"""


# ---------------------------------------------------------------------------
# 1) Module + key public symbols exist
# ---------------------------------------------------------------------------
def test_module_imports_and_public_symbols():
    """The updater module imports cleanly and exposes the public surface."""
    assert up is not None
    for name in (
        "main",
        "parse_v3_measured",
        "determine_promotion",
        "update_207_paragraph",
        "update_188_paragraph",
        "render_updated_doc",
        "_atomic_write_text",
        "_atomic_write_json",
        "_build_argparser",
        "render_report",
        "DEFAULT_WF263A_VERIFY_JSON",
        "DEFAULT_DOC_207",
        "DEFAULT_DOC_188",
        "DEFAULT_OUT_REPORT",
        "SEC9_5_OLD_PATTERN",
        "SEC0_1_APPEND_MARKER",
        "V2_BASELINE",
        "SPLIT_LABELS",
    ):
        assert hasattr(up, name), f"missing public symbol: {name}"


# ---------------------------------------------------------------------------
# 2) argparser exposes >= 4 flags (G2 acceptance gate)
# ---------------------------------------------------------------------------
def test_argparse_flag_count():
    parser = up._build_argparser()
    # Each add_argument() registers one flag; subtract 1 for the implicit
    # --help action so we count user-exposed flags only.
    n_flags = sum(1 for a in parser._actions if a.dest != "help")
    assert n_flags >= 4, (
        f"expected >=4 CLI flags, got {n_flags}: "
        f"{[a.dest for a in parser._actions if a.dest != 'help']}"
    )


# ---------------------------------------------------------------------------
# 3) Default --dry-run is True (backward-compat gate G3)
# ---------------------------------------------------------------------------
def test_dry_run_default_true():
    parser = up._build_argparser()
    args = parser.parse_args(
        [
            "--wf263a-verify-json",
            "/tmp/x.json",
            "--207-doc",
            "/tmp/y.md",
            "--188-doc",
            "/tmp/z.md",
        ]
    )
    assert args.dry_run is True, f"--dry-run default should be True; got {args.dry_run}"


# ---------------------------------------------------------------------------
# 4) _atomic_write_text writes + overwrites with no leftover temp files (G4)
# ---------------------------------------------------------------------------
def test_atomic_write_text_round_trip(tmp_path: Path):
    out = tmp_path / "x.md"
    up._atomic_write_text(out, "hello world\n")
    assert out.read_text(encoding="utf-8") == "hello world\n"
    up._atomic_write_text(out, "replacement\n")
    assert out.read_text(encoding="utf-8") == "replacement\n"
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"tempfile leaked: {leftovers}"


# ---------------------------------------------------------------------------
# 5) parse_v3_measured pulls per-split AUCs from aggregate JSON
# ---------------------------------------------------------------------------
def test_parse_v3_measured_extracts_per_split_aucs():
    agg = _make_aggregate_v3_above_v2()
    splits, gate_meta = up.parse_v3_measured(agg)
    assert len(splits) == 3, f"expected 3 splits, got {len(splits)}"
    labels = [s.split_label for s in splits]
    assert labels == [
        "assay",
        "scaffold",
        "size",
    ], f"split order should be assay/scaffold/size; got {labels}"
    by_label = {s.split_label: s for s in splits}
    assert by_label["assay"].v3_mean == pytest.approx(0.7250)
    assert by_label["scaffold"].v3_mean == pytest.approx(0.6789)
    assert by_label["size"].v3_mean == pytest.approx(0.6812)
    assert by_label["assay"].passes_v2_baseline is True
    # gate_meta carries the Wf-263A provenance fields.
    assert gate_meta["n_splits_measured"] == 3
    assert gate_meta["gate_6_overall_v3_fallback_matches_v2"] is True


# ---------------------------------------------------------------------------
# 6) determine_promotion = True when assay passes_v2_baseline
# ---------------------------------------------------------------------------
def test_determine_promotion_above_v2_baseline():
    agg = _make_aggregate_v3_above_v2()
    splits, _ = up.parse_v3_measured(agg)
    assert up.determine_promotion(splits) is True


# ---------------------------------------------------------------------------
# 7) determine_promotion = False when assay FAILS passes_v2_baseline
# ---------------------------------------------------------------------------
def test_determine_promotion_below_v2_baseline():
    agg = _make_aggregate_v3_below_v2()
    splits, _ = up.parse_v3_measured(agg)
    assert up.determine_promotion(splits) is False


# ---------------------------------------------------------------------------
# 8) update_207_paragraph rewrites the §9.5 "projected ≥ v2 baseline" line
# ---------------------------------------------------------------------------
def test_update_207_replaces_projected_with_measured():
    splits, _ = up.parse_v3_measured(_make_aggregate_v3_above_v2())
    r = up.update_207_paragraph(DOC_207_TEMPLATE, splits, promotion=True)
    assert (
        r.applied is True
    ), f"update_207 should have applied; skip_reason={r.skip_reason!r}"
    # Old "projected" substring no longer survives in the new_substring.
    assert "projected ≥ v2 baseline" not in r.new_substring
    # New substring carries the MEASURED numbers + verdict.
    assert "0.7250" in r.new_substring
    assert "MEASURED" in r.new_substring
    assert "ELIGIBLE for §4 LEAD promotion" in r.new_substring


# ---------------------------------------------------------------------------
# 9) update_207_paragraph returns applied=False when the sentence has drifted
# ---------------------------------------------------------------------------
def test_update_207_no_match_when_sentence_missing():
    bad_doc = "# TODO-207 — different content\n\nNo §9.5 here."
    splits, _ = up.parse_v3_measured(_make_aggregate_v3_above_v2())
    r = up.update_207_paragraph(bad_doc, splits, promotion=True)
    assert r.applied is False
    assert r.old_substring == ""
    assert r.skip_reason  # non-empty error message


# ---------------------------------------------------------------------------
# 10) update_188_paragraph appends §4 LEAD paragraph when promoted
# ---------------------------------------------------------------------------
def test_update_188_appends_lead_paragraph_when_promoted():
    splits, _ = up.parse_v3_measured(_make_aggregate_v3_above_v2())
    r = up.update_188_paragraph(DOC_188_TEMPLATE, splits, promotion=True)
    assert (
        r.applied is True
    ), f"update_188 should have applied; skip_reason={r.skip_reason!r}"
    # The new substring carries the assay number + the Wf-263A provenance tag.
    assert "0.7250" in r.new_substring
    assert "Wf-263A-Verify" in r.new_substring
    assert "chemprop v3 OOD §4 LEAD claim" in r.new_substring


# ---------------------------------------------------------------------------
# 11) update_188_paragraph does NOT touch 188 when NOT promoted
# ---------------------------------------------------------------------------
def test_update_188_no_touch_when_not_promoted():
    splits, _ = up.parse_v3_measured(_make_aggregate_v3_below_v2())
    r = up.update_188_paragraph(DOC_188_TEMPLATE, splits, promotion=False)
    assert r.applied is False
    assert "NOT promoted" in r.skip_reason
    assert r.old_substring == ""
    assert r.new_substring == ""


# ---------------------------------------------------------------------------
# 12) End-to-end dry-run writes only the report (no doc mutations)
# ---------------------------------------------------------------------------
def test_end_to_end_dry_run_writes_only_report(tmp_path: Path):
    verify_path = tmp_path / "aggregate.json"
    doc207_path = tmp_path / "207.md"
    doc188_path = tmp_path / "188.md"
    out_report = tmp_path / "post_fix_update_report.json"
    updated_207 = tmp_path / "updated_207.md"
    updated_188 = tmp_path / "updated_188.md"

    verify_path.write_text(json.dumps(_make_aggregate_v3_above_v2()), encoding="utf-8")
    doc207_path.write_text(DOC_207_TEMPLATE, encoding="utf-8")
    doc188_path.write_text(DOC_188_TEMPLATE, encoding="utf-8")
    original_207 = doc207_path.read_text(encoding="utf-8")
    original_188 = doc188_path.read_text(encoding="utf-8")

    rc = up.main(
        [
            "--wf263a-verify-json",
            str(verify_path),
            "--207-doc",
            str(doc207_path),
            "--188-doc",
            str(doc188_path),
            "--out-report",
            str(out_report),
            "--updated-207",
            str(updated_207),
            "--updated-188",
            str(updated_188),
        ]
    )
    assert rc == 0
    assert out_report.exists(), "post_fix_update_report.json should be written"
    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["promote_v3_to_lead"] is True
    assert report["updates"]["doc_207"]["applied"] is True
    # Dry-run must NOT have written updated_* files.
    assert not updated_207.exists(), "dry-run should not write updated 207"
    assert not updated_188.exists(), "dry-run should not write updated 188"
    # And must NOT have touched the source docs.
    assert doc207_path.read_text(encoding="utf-8") == original_207
    assert doc188_path.read_text(encoding="utf-8") == original_188


# ---------------------------------------------------------------------------
# 13) End-to-end --no-dry-run with v3 BELOW v2 baseline: only 207 written
# ---------------------------------------------------------------------------
def test_end_to_end_apply_writes_207_only_when_not_promoted(tmp_path: Path):
    verify_path = tmp_path / "aggregate.json"
    doc207_path = tmp_path / "207.md"
    doc188_path = tmp_path / "188.md"
    out_report = tmp_path / "report.json"
    updated_207 = tmp_path / "updated_207.md"
    updated_188 = tmp_path / "updated_188.md"

    verify_path.write_text(json.dumps(_make_aggregate_v3_below_v2()), encoding="utf-8")
    doc207_path.write_text(DOC_207_TEMPLATE, encoding="utf-8")
    doc188_path.write_text(DOC_188_TEMPLATE, encoding="utf-8")
    original_188 = doc188_path.read_text(encoding="utf-8")

    rc = up.main(
        [
            "--wf263a-verify-json",
            str(verify_path),
            "--207-doc",
            str(doc207_path),
            "--188-doc",
            str(doc188_path),
            "--out-report",
            str(out_report),
            "--updated-207",
            str(updated_207),
            "--updated-188",
            str(updated_188),
            "--no-dry-run",
        ]
    )
    assert rc == 0
    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["promote_v3_to_lead"] is False
    # 207 IS rewritten (always), 188 is NOT (because not promoted).
    assert updated_207.exists(), "207 should be rewritten even when v3 not promoted"
    assert not updated_188.exists(), "188 must NOT be rewritten when v3 not promoted"
    # The new 207 line must carry the honest-negative verdict.
    body = updated_207.read_text(encoding="utf-8")
    assert "MEASURED below v2 baseline" in body
    assert "0.6812" in body
    # 188 doc on disk must be byte-identical to original.
    assert doc188_path.read_text(encoding="utf-8") == original_188


# ---------------------------------------------------------------------------
# 14) End-to-end --no-dry-run with v3 ABOVE v2 baseline: both docs written
# ---------------------------------------------------------------------------
def test_end_to_end_apply_writes_both_docs_when_promoted(tmp_path: Path):
    verify_path = tmp_path / "aggregate.json"
    doc207_path = tmp_path / "207.md"
    doc188_path = tmp_path / "188.md"
    out_report = tmp_path / "report.json"
    updated_207 = tmp_path / "updated_207.md"
    updated_188 = tmp_path / "updated_188.md"

    verify_path.write_text(json.dumps(_make_aggregate_v3_above_v2()), encoding="utf-8")
    doc207_path.write_text(DOC_207_TEMPLATE, encoding="utf-8")
    doc188_path.write_text(DOC_188_TEMPLATE, encoding="utf-8")

    rc = up.main(
        [
            "--wf263a-verify-json",
            str(verify_path),
            "--207-doc",
            str(doc207_path),
            "--188-doc",
            str(doc188_path),
            "--out-report",
            str(out_report),
            "--updated-207",
            str(updated_207),
            "--updated-188",
            str(updated_188),
            "--no-dry-run",
        ]
    )
    assert rc == 0
    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["promote_v3_to_lead"] is True
    assert report["updates"]["doc_207"]["applied"] is True
    assert report["updates"]["doc_188"]["applied"] is True
    assert updated_207.exists()
    assert updated_188.exists()

    # 207 body carries the MEASURED numbers + promotion verdict.
    body_207 = updated_207.read_text(encoding="utf-8")
    assert "MEASURED" in body_207
    assert "ELIGIBLE for §4 LEAD promotion" in body_207
    # 188 body carries the §4 LEAD appendix.
    body_188 = updated_188.read_text(encoding="utf-8")
    assert "chemprop v3 OOD §4 LEAD claim" in body_188
    assert "0.7250" in body_188

    # The original source docs must be untouched (atomic write goes to
    # updated_*, NOT in place).
    assert doc207_path.read_text(encoding="utf-8") == DOC_207_TEMPLATE
    assert doc188_path.read_text(encoding="utf-8") == DOC_188_TEMPLATE
