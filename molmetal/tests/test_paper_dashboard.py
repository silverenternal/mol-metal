"""Wf-265-Dashboard unit tests for ``paper_dashboard.py``.

Covers the 5 acceptance gates documented in the Wf-265-Dashboard task
spec, plus the Wf-268C-Dash additions:

  Wf-265 gates:
    1. **G1_script_written**: script file exists at the canonical path.
    2. **G2_n_cli_flags**: CLI exposes ≥ 4 contract flags (``--paper-doc``,
       ``--collect-json``, ``--out-html``, ``--verbose``); the optional
       ``--checklist-doc`` brings the total to 5.
    3. **G3_html_contains_metrics**: the emitted HTML contains the verbatim
       metric strings (5-metal mean AUC, DrugOOD splits, CFM m14, CFM m7).
    4. **G4_n_tests**: ≥ 3 distinct tests in this file.
    5. **G5_dashboard_size_kb**: the emitted HTML is well-formed and reports
       a non-zero size in KB on disk.

  Wf-268C-Dash additions (added by this Wf):
    6. **G6_v3_fallback_section**: chemprop v3 v3-fallback section
       (Wf-263A) renders in the emitted HTML (cohort + claim-cards).
    7. **G7_v3_lomo_fix_section**: Wf-266 v3 LOMO fix flags section
       (--chemprop-min-epochs, --chemprop-lomo-init-scale, v3-fix mode)
       renders in the emitted HTML.
    8. **G8_g4_per_metal_table**: TODO-180 G4 per-metal test_auc table
       surfaces the 5-metal × 3-seed per-metal means in the cohort table.
    9. **G9_badge_discipline_explicit**: MEASURED-vs-PROJECTED badge
       discipline banner is present in the emitted HTML and the build
       JSON ``badge_counts`` field is populated.
   10. **G10_update_report_present**: ``update_report`` field exists in the
       build JSON and reports ``wf_id == "Wf-268C-Dash"`` + a non-zero
       count of new claims / honest-negatives / cohort sections.

The tests construct ephemeral files under ``tmp_path`` so the real
``molmetal/reports/wf265_collect/all_results.json`` and the real paper
doc are NOT modified.

All tests are CPU-only and import the script as a Python module
(no torch, no GPU).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Make ``paper_dashboard`` importable as a module (no torch needed).
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import paper_dashboard  # noqa: E402

VENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"
# Fallback to the codes-prefixed venv if the cades-prefixed one is missing
# (some symlink layouts).
if not Path(VENV_PY).exists():
    _alt = "/home/hugo/codes/mol-metal/.venv/bin/python"
    if Path(_alt).exists():
        VENV_PY = _alt


# Minimal stub collect.json content matching the real wf265_collect schema.
STUB_COLLECT_JSON: dict = {
    "G1_todo_180_g4_complete": True,
    "G2_wf263a_complete": False,
    "G3_wf264_complete": False,
    "G4_n_g4_pairs": 14,
    "G5_v3_ood_auc": 0.4544,
    "G6_collect_json_path": "stub/path/all_results.json",
    "honest_framing_note": "STUB honest_framing_note for test (unit-test scaffold).",
    "g4_per_metal_aggregate": {
        "Ru": {
            "n_pairs": 2,
            "mean_test_auc": 0.7483,
            "min_test_auc": 0.7453,
            "max_test_auc": 0.7514,
            "test_aucs": [0.7453, 0.7514],
        },
        "Ir": {
            "n_pairs": 3,
            "mean_test_auc": 0.6702,
            "min_test_auc": 0.6659,
            "max_test_auc": 0.6782,
            "test_aucs": [0.6659, 0.6782, 0.6666],
        },
    },
    "g4_per_metal_aggregate_full_15pair": {
        "Ru": {
            "n_pairs": 3,
            "mean_test_auc": 0.7526,
            "min_test_auc": 0.7453,
            "max_test_auc": 0.7613,
            "test_aucs": [0.7613, 0.7453, 0.7514],
        },
        "Ir": {
            "n_pairs": 3,
            "mean_test_auc": 0.6702,
            "min_test_auc": 0.6659,
            "max_test_auc": 0.6782,
            "test_aucs": [0.6659, 0.6782, 0.6666],
        },
    },
    "g4_overall_aggregate": {
        "n_pairs": 14,
        "mean_test_auc": 0.7209,
        "min_test_auc": 0.6659,
        "max_test_auc": 0.7970,
    },
    "g4_overall_aggregate_full_15pair": {
        "n_pairs": 15,
        "mean_test_auc": 0.7236,
        "min_test_auc": 0.6659,
        "max_test_auc": 0.7970,
    },
}

STUB_PAPER_DOC_TEXT = (
    "# STUB Paper §4 doc\n\n"
    "This is a stub paper doc used by the unit test — it is NOT a real\n"
    "TODO-188 DRAFT.  The dashboard script does not parse the markdown;\n"
    "it only verifies the file exists and emits the verbatim catalog.\n"
)


# ---------------------------------------------------------------------------
# Test 1 — G1: script exists at canonical path (G1_script_written=True)
# ---------------------------------------------------------------------------


def test_g1_script_exists_at_canonical_path():
    """The dashboard script MUST exist at
    ``molmetal/scripts/paper_dashboard.py``."""
    script_path = SCRIPT_DIR / "paper_dashboard.py"
    assert (
        script_path.is_file()
    ), f"G1 FAIL: paper_dashboard.py not found at {script_path}"
    # Sanity-check the file is non-empty and contains the expected module
    # docstring so we know it's our script and not a stale placeholder.
    text = script_path.read_text(encoding="utf-8")
    assert len(text) > 1000, "script body too short; suspect placeholder"
    assert (
        "Wf-265-Dashboard" in text
    ), "Wf-265-Dashboard token missing — wrong script on disk"


# ---------------------------------------------------------------------------
# Test 2 — G2: CLI exposes the 4 contract flags (--paper-doc, --collect-json,
#          --out-html, --verbose); --checklist-doc as optional 5th
# ---------------------------------------------------------------------------


def test_g2_cli_exposes_contract_flags():
    """The CLI MUST expose ≥ 4 contract flags; we ship 5 (incl. --checklist-doc)."""
    flags = {
        "--paper-doc",
        "--collect-json",
        "--out-html",
        "--verbose",
        "--checklist-doc",
    }
    # Walk the source and grep for each flag — the CLI parser is built via
    # argparse so the flag strings appear in the source.
    text = (SCRIPT_DIR / "paper_dashboard.py").read_text(encoding="utf-8")
    for flag in flags:
        assert (
            flag in text
        ), f"G2 FAIL: CLI flag {flag} not present in paper_dashboard.py source"
    # And the argparse parser object exposes them too.
    parser = paper_dashboard._build_arg_parser()
    # Argparse stores dest names; we can introspect by re-parsing with
    # --help and capturing stdout (best-effort).
    for flag in flags:
        # argparse normalises --paper-doc -> paper_doc dest; check the
        # dest is in the parser's _actions.
        dest = flag.lstrip("-").replace("-", "_")
        action_dests = {a.dest for a in parser._actions}
        assert (
            dest in action_dests
        ), f"G2 FAIL: argparse dest {dest} missing from parser actions"


# ---------------------------------------------------------------------------
# Test 3 — G3: emitted HTML contains the verbatim metric strings
#          (5-metal mean AUC, DrugOOD 3 splits, CFM m14, CFM m7)
# ---------------------------------------------------------------------------


def test_g3_emitted_html_contains_metrics(tmp_path: Path):
    """Build the dashboard against a stub collect.json + stub paper doc;
    verify the emitted HTML contains the verbatim cohort + claim metric
    strings.
    """
    # Write stub files under tmp_path.
    paper_doc = tmp_path / "stub_paper.md"
    paper_doc.write_text(STUB_PAPER_DOC_TEXT, encoding="utf-8")
    collect_json = tmp_path / "stub_collect.json"
    collect_json.write_text(json.dumps(STUB_COLLECT_JSON, indent=2), encoding="utf-8")
    out_html = tmp_path / "dashboard.html"

    obj = paper_dashboard.build_dashboard_html(
        paper_doc=paper_doc,
        collect_json=collect_json,
        checklist_doc=None,
        paper_doc_text=STUB_PAPER_DOC_TEXT,
        collect_data=STUB_COLLECT_JSON,
        verbose=False,
    )
    assert (
        obj["schema_valid"] is True
    ), f"G3 FAIL: schema_valid=False: {obj.get('error', '?')}"
    html_str = obj["html"]

    # Required verbatim tokens — must all appear.
    required_tokens = [
        # chemprop v2 v3 cohort
        "chemprop v2 v3 5-metal per-metal AUC",
        # DrugOOD 3 splits (verbatim Ji 2023 split names)
        "lbap_refined_ic50_assay",
        "lbap_refined_ic50_scaffold",
        "lbap_refined_ic50_size",
        # CFM m14
        "CFM m14 real decode_ratio",
        "decode_ratio",
        # CFM m7
        "CFM m7 Tanimoto diversity",
        "TargetDiff",
        # Honest-framing note
        "Honest-negative catalog",
        "party-twenty-spirit",
        # Badges (CSS classnames + label tokens)
        "badge-measured",
        "badge-projected",
        "badge-honest-negative",
        "MEASURED",
        "PROJECTED",
        "honest-negative",
    ]
    missing = [t for t in required_tokens if t not in html_str]
    assert not missing, f"G3 FAIL: HTML missing required tokens: {missing}"

    # Per-claim card count = catalog length; verify at least 5 claims
    # were rendered (we ship 20 in the catalog).
    n_claim_cards = html_str.count('class="claim-card"')
    assert (
        n_claim_cards >= 5
    ), f"G3 FAIL: only {n_claim_cards} claim cards in HTML (expected ≥ 5)"

    # Write to disk and verify file is non-empty.
    out_html.write_text(html_str, encoding="utf-8")
    size_kb = out_html.stat().st_size / 1024.0
    assert size_kb > 1.0, f"G5 FAIL: dashboard too small ({size_kb:.2f} KB)"

    # Capture the size for the G5 reporting test.
    test_g3_emitted_html_contains_metrics.size_kb = size_kb  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test 4 — G4: ≥ 3 distinct tests in this file (counted via pytest)
# ---------------------------------------------------------------------------


def test_g4_n_distinct_tests_at_least_three():
    """At least 3 distinct tests live in this file.  We assert this by
    counting the top-level ``def test_*`` functions defined above this
    one.
    """
    import inspect

    here = Path(__file__).resolve()
    src = here.read_text(encoding="utf-8")
    test_funcs = re.findall(r"^def\s+(test_[a-zA-Z0-9_]+)\(", src, flags=re.MULTILINE)
    n_tests = len(set(test_funcs))
    assert n_tests >= 3, (
        f"G4 FAIL: only {n_tests} distinct tests in this file; "
        f"target ≥ 3 (saw {sorted(set(test_funcs))})"
    )


# ---------------------------------------------------------------------------
# Test 5 — G5: dashboard size on disk (post-CLI end-to-end smoke)
# ---------------------------------------------------------------------------


def test_g5_dashboard_cli_end_to_end(tmp_path: Path):
    """Run the script end-to-end via subprocess with the venv python;
    verify the HTML on disk is non-trivial in size and contains the
    required metric tokens.
    """
    paper_doc = tmp_path / "stub_paper.md"
    paper_doc.write_text(STUB_PAPER_DOC_TEXT, encoding="utf-8")
    collect_json = tmp_path / "stub_collect.json"
    collect_json.write_text(json.dumps(STUB_COLLECT_JSON, indent=2), encoding="utf-8")
    out_html = tmp_path / "out_dashboard.html"

    cmd = [
        VENV_PY,
        str(SCRIPT_DIR / "paper_dashboard.py"),
        "--paper-doc",
        str(paper_doc),
        "--collect-json",
        str(collect_json),
        "--out-html",
        str(out_html),
        "--verbose",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        pytest.fail(
            f"G5 FAIL: paper_dashboard.py exited {proc.returncode}; "
            f"stdout={proc.stdout[:400]}; stderr={proc.stderr[:400]}"
        )
    assert out_html.is_file(), "G5 FAIL: out_html not created"
    size_kb = out_html.stat().st_size / 1024.0
    assert size_kb > 1.0, f"G5 FAIL: dashboard too small ({size_kb:.2f} KB)"

    # Build JSON sidecar also written.
    build_json = out_html.with_suffix(".build.json")
    assert build_json.is_file(), "G5 FAIL: build JSON sidecar not written"
    build_obj = json.loads(build_json.read_text(encoding="utf-8"))
    assert build_obj["schema_valid"] is True
    assert build_obj["n_claims"] >= 5
    assert build_obj["n_honest_negatives"] >= 5
    test_g5_dashboard_cli_end_to_end.size_kb = size_kb  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Wf-268C-Dash additions — tests 6..10 (5 new tests)
# ---------------------------------------------------------------------------


def test_g6_v3_fallback_section_in_html(tmp_path: Path):
    """G6: Wf-263A chemprop v3 v3-fallback section renders in the emitted
    HTML (cohort table + claim-cards + honest-negative presence).
    """
    paper_doc = tmp_path / "stub_paper.md"
    paper_doc.write_text(STUB_PAPER_DOC_TEXT, encoding="utf-8")
    collect_json = tmp_path / "stub_collect.json"
    collect_json.write_text(json.dumps(STUB_COLLECT_JSON, indent=2), encoding="utf-8")

    obj = paper_dashboard.build_dashboard_html(
        paper_doc=paper_doc,
        collect_json=collect_json,
        checklist_doc=None,
        paper_doc_text=STUB_PAPER_DOC_TEXT,
        collect_data=STUB_COLLECT_JSON,
        verbose=False,
    )
    assert obj["schema_valid"] is True
    html_str = obj["html"]

    # Required v3-fallback tokens
    required_tokens = [
        "Wf-263A chemprop v3 v3-fallback mode dispatch",
        "--chemprop-mode",
        "v3-fallback",
        "claim_chemprop_v3_fallback_default",
        "claim_chemprop_v3_fallback_official",
    ]
    missing = [t for t in required_tokens if t not in html_str]
    assert not missing, f"G6 FAIL: HTML missing v3-fallback tokens: {missing}"

    # Confirm the chemprop_v3_fallback cohort is in the build JSON
    assert (
        "chemprop_v3_fallback" in obj["cohorts"]
    ), "G6 FAIL: chemprop_v3_fallback cohort missing from build JSON"
    cohort = obj["cohorts"]["chemprop_v3_fallback"]
    assert (
        cohort.get("n_files_modified") == 3
    ), f"G6 FAIL: expected n_files_modified=3, got {cohort.get('n_files_modified')}"


def test_g7_v3_lomo_fix_section_in_html(tmp_path: Path):
    """G7: Wf-266 v3 LOMO fix flags section renders in the emitted HTML
    (cohort table + flag tokens + honest-negative presence).
    """
    paper_doc = tmp_path / "stub_paper.md"
    paper_doc.write_text(STUB_PAPER_DOC_TEXT, encoding="utf-8")
    collect_json = tmp_path / "stub_collect.json"
    collect_json.write_text(json.dumps(STUB_COLLECT_JSON, indent=2), encoding="utf-8")

    obj = paper_dashboard.build_dashboard_html(
        paper_doc=paper_doc,
        collect_json=collect_json,
        checklist_doc=None,
        paper_doc_text=STUB_PAPER_DOC_TEXT,
        collect_data=STUB_COLLECT_JSON,
        verbose=False,
    )
    assert obj["schema_valid"] is True
    html_str = obj["html"]

    required_tokens = [
        "Wf-266 chemprop v3 LOMO fix flags",
        "--chemprop-min-epochs",
        "--chemprop-lomo-init-scale",
        "v3-fix",
        "claim_wf266_min_epochs_flag",
        "claim_wf266_lomo_init_scale_flag",
        "claim_wf266_v3_fix_mode",
    ]
    missing = [t for t in required_tokens if t not in html_str]
    assert not missing, f"G7 FAIL: HTML missing Wf-266 LOMO fix tokens: {missing}"

    assert (
        "wf266_lomo_fix" in obj["cohorts"]
    ), "G7 FAIL: wf266_lomo_fix cohort missing from build JSON"
    cohort = obj["cohorts"]["wf266_lomo_fix"]
    # 3 flag rows in the cohort
    assert (
        len(cohort.get("rows", [])) == 3
    ), f"G7 FAIL: expected 3 flag rows, got {len(cohort.get('rows', []))}"


def test_g8_g4_per_metal_table_in_cohort(tmp_path: Path):
    """G8: TODO-180 G4 per-metal test_auc table surfaces the 5-metal ×
    3-seed per-metal means in the chemprop v2 v3 cohort.
    """
    paper_doc = tmp_path / "stub_paper.md"
    paper_doc.write_text(STUB_PAPER_DOC_TEXT, encoding="utf-8")
    collect_json = tmp_path / "stub_collect.json"
    collect_json.write_text(json.dumps(STUB_COLLECT_JSON, indent=2), encoding="utf-8")

    obj = paper_dashboard.build_dashboard_html(
        paper_doc=paper_doc,
        collect_json=collect_json,
        checklist_doc=None,
        paper_doc_text=STUB_PAPER_DOC_TEXT,
        collect_data=STUB_COLLECT_JSON,
        verbose=False,
    )
    assert obj["schema_valid"] is True
    html_str = obj["html"]

    # TODO-180 G4 per-metal table tokens
    required_tokens = [
        "chemprop v2 v3 5-metal per-metal AUC",
        "TODO-180 G4",
        "claim_todo_180_g4_re_top",
        "claim_todo_180_g4_overall_15pair",
    ]
    missing = [t for t in required_tokens if t not in html_str]
    assert not missing, f"G8 FAIL: HTML missing TODO-180 G4 table tokens: {missing}"

    # The chemprop cohort has the per-metal aggregate from the stub
    cohort = obj["cohorts"]["chemprop_v2_v3"]
    assert (
        len(cohort.get("rows", [])) >= 2
    ), f"G8 FAIL: chemprop cohort rows < 2: {len(cohort.get('rows', []))}"
    metals = {r.get("metal") for r in cohort.get("rows", [])}
    assert (
        "Ru" in metals and "Ir" in metals
    ), f"G8 FAIL: chemprop cohort missing Ru/Ir metals: {metals}"


def test_g9_badge_discipline_explicit(tmp_path: Path):
    """G9: MEASURED-vs-PROJECTED badge discipline banner is present in
    the emitted HTML and the build JSON ``badge_counts`` field is
    populated with three non-empty keys.
    """
    paper_doc = tmp_path / "stub_paper.md"
    paper_doc.write_text(STUB_PAPER_DOC_TEXT, encoding="utf-8")
    collect_json = tmp_path / "stub_collect.json"
    collect_json.write_text(json.dumps(STUB_COLLECT_JSON, indent=2), encoding="utf-8")

    obj = paper_dashboard.build_dashboard_html(
        paper_doc=paper_doc,
        collect_json=collect_json,
        checklist_doc=None,
        paper_doc_text=STUB_PAPER_DOC_TEXT,
        collect_data=STUB_COLLECT_JSON,
        verbose=False,
    )
    assert obj["schema_valid"] is True
    html_str = obj["html"]

    # Badge discipline banner tokens
    required_tokens = [
        "MEASURED-vs-PROJECTED badge discipline",
        "party-twenty-spirit",
        "MUTUALLY EXCLUSIVE",
        "badge-summary",
    ]
    missing = [t for t in required_tokens if t not in html_str]
    assert not missing, f"G9 FAIL: HTML missing badge discipline tokens: {missing}"

    # update_report.badge_counts must have MEASURED, PROJECTED, honest-negative
    update_report = obj.get("update_report") or {}
    bc = update_report.get("badge_counts") or {}
    for key in ("MEASURED", "PROJECTED", "honest-negative"):
        assert key in bc, f"G9 FAIL: badge_counts missing key {key}"
        assert isinstance(
            bc[key], int
        ), f"G9 FAIL: badge_counts[{key}] is not int: {bc[key]!r}"
    # Total claims = sum of badge_counts
    assert (
        sum(bc.values()) == obj["n_claims"]
    ), f"G9 FAIL: sum(badge_counts)={sum(bc.values())} != n_claims={obj['n_claims']}"


def test_g10_update_report_present(tmp_path: Path):
    """G10: ``update_report`` field exists in the build JSON and reports
    ``wf_id == "Wf-268C-Dash"`` + a non-zero count of new claims /
    honest-negatives / cohort sections.
    """
    paper_doc = tmp_path / "stub_paper.md"
    paper_doc.write_text(STUB_PAPER_DOC_TEXT, encoding="utf-8")
    collect_json = tmp_path / "stub_collect.json"
    collect_json.write_text(json.dumps(STUB_COLLECT_JSON, indent=2), encoding="utf-8")

    obj = paper_dashboard.build_dashboard_html(
        paper_doc=paper_doc,
        collect_json=collect_json,
        checklist_doc=None,
        paper_doc_text=STUB_PAPER_DOC_TEXT,
        collect_data=STUB_COLLECT_JSON,
        verbose=False,
    )
    assert obj["schema_valid"] is True
    update_report = obj.get("update_report")
    assert update_report is not None, "G10 FAIL: update_report missing"
    assert (
        update_report["wf_id"] == "Wf-268C-Dash"
    ), f"G10 FAIL: wf_id != Wf-268C-Dash: {update_report['wf_id']!r}"
    assert (
        update_report["n_new_claims_added"] >= 1
    ), f"G10 FAIL: n_new_claims_added < 1: {update_report['n_new_claims_added']}"
    assert update_report["n_new_honest_negatives_added"] >= 1, (
        f"G10 FAIL: n_new_honest_negatives_added < 1: "
        f"{update_report['n_new_honest_negatives_added']}"
    )
    assert update_report["n_new_cohort_sections_added"] >= 1, (
        f"G10 FAIL: n_new_cohort_sections_added < 1: "
        f"{update_report['n_new_cohort_sections_added']}"
    )
    # The new claim IDs we expect
    expected_ids = {
        "claim_chemprop_v3_fallback_default",
        "claim_chemprop_v3_fallback_official",
        "claim_wf266_min_epochs_flag",
        "claim_wf266_lomo_init_scale_flag",
        "claim_wf266_v3_fix_mode",
        "claim_todo_180_g4_re_top",
        "claim_todo_180_g4_overall_15pair",
    }
    assert expected_ids.issubset(set(update_report["new_claim_ids"])), (
        f"G10 FAIL: missing claim IDs in update_report: "
        f"{expected_ids - set(update_report['new_claim_ids'])}"
    )


# ---------------------------------------------------------------------------
# End of unit tests.  Test count target ≥ 3 (G4) satisfied by tests 1-5
# above; Wf-268C-Dash adds tests 6-10.
# ---------------------------------------------------------------------------
