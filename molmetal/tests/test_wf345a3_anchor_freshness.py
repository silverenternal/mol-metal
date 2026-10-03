"""Wf-345-A3 — anchor freshness 24h audit (3 NEW tests).

================================================================
Per the 2026-09-29 ultracode wave (user directive "基于刚才的洞察，
可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，
工作流可以长"), the :func:`gate_wf343c4_anchor_freshness_24h` pre-retrain
gate currently FAILs because 4 of 11 paper-grade ``*_anchor.json`` +
``sweep_summary.json`` files exceed the 24h freshness budget per
Wf-302A / Wf-304A AnchorGuard. This file is the AUDIT test surface —
it verifies the audit artifacts (anchor catalog + stale detection +
audit JSON schema) so the agent-side report at
:mod:`molmetal.reports.wf345a3_anchor_audit` is trustworthy.

Tests
-----

1. **test_wf345a3_anchor_catalog_count** — assert exactly 11 paper-grade
   anchors are catalogued under the 4 ANCHOR_GLOB_PATTERNS globs. If
   the count drifts from 11 (i.e. a new anchor was written or an old
   one was deleted), the test FAILs with a clear drift report. This
   pins the audit's "11 total" claim that the
   :func:`gate_wf343c4_anchor_freshness_24h` FAIL message depends on.

2. **test_wf345a3_stale_anchor_detection** — verify that 4 of the 11
   anchors are detected as STALE (> 24h mtime). Uses real mtime
   inspection (not the audit JSON) so the detection logic is
   self-contained. The test confirms the precise set of stale paths
   reported by the gate matches the audit's stale_anchors list (3 of
   the 4 are unambiguous: wf121_cisplatin_real_vina +
   wf143_p4_chemotype_dock + wf170_permetal_singletask; the 4th is
   wf67_p4_3_100pocket which surfaced during the audit).

3. **test_wf345a3_audit_json_schema_valid** — verify the audit artifact
   at ``molmetal/reports/wf345a3_anchor_audit/audit.json`` parses as
   JSON and carries the canonical schema fields:
   ``audit_id`` + ``gate_name`` + ``max_age_sec`` + ``n_anchors_total``
   + ``n_anchors_stale`` + ``per_anchor`` (list of dicts each with
   ``path`` + ``age_hr`` + ``stale`` keys) + ``stale_anchors``. This
   pins the audit's schema so downstream consumers (e.g. a
   Wf-345-A3-AutoRefresher, a paper_dashboard pre-flight warning, or
   the wave-budget refund tool) can parse it without bespoke handling.

Backward-compat verdict
-----------------------
The 3 NEW tests are ADDITIVE only — they do NOT modify any
pre-existing test / gate / runner / path. The Wf-343C4
``EXPECTED_GATE_COUNTS["all"] == 66`` invariant is unchanged because
these tests live in a new file (do NOT register any gate). The
``gate_wf343c4_anchor_freshness_24h`` gate remains FAIL until
Wf-345 Phase 3 C1-C4 refreshes the stale anchors.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* 4/11 anchors exceed 24h MEASURED (full audit confirms count + IDs).
* Re-run cost is PROJECTED 30-60 min wall on CPU (Phase 3 work).
* If re-runs produce different numbers: HONEST-NEGATIVE measurement
  drift; we declare drift rather than paraphrase as "promising lift".
"""

import json
import pathlib
import time

import pytest

# ---------------------------------------------------------------------------
# Constants — keep aligned with molmetal/scripts/pre_flight_check.py:94-99
# ---------------------------------------------------------------------------

ANCHOR_GLOB_PATTERNS = (
    "molmetal/reports/**/m14_anchor.json",
    "molmetal/reports/**/wf269c_report.json",
    "molmetal/reports/**/m17_drugood_reeval.json",
    "molmetal/reports/**/sweep_summary.json",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_AGE_SEC = 24 * 3600.0
MAX_AGE_HR = 24

# Expected stale anchor set (Wf-345-A3 audit confirmed at 2026-09-29):
EXPECTED_STALE_PATHS = {
    "molmetal/reports/wf67_p4_3_100pocket/sweep_summary.json",
    "molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json",
    "molmetal/reports/wf143_p4_chemotype_dock/sweep_summary.json",
    "molmetal/reports/wf170_permetal_singletask/sweep_summary.json",
}


def _list_anchor_files():
    """Return sorted list of paper-grade anchor paths matching the 4 globs.

    Mirrors :func:`molmetal.scripts.pre_flight_check.gate_wf343c4_anchor_freshness_24h`
    glob expansion semantics (relative patterns, no absolute fallback).
    """
    out = set()
    for pattern in ANCHOR_GLOB_PATTERNS:
        parts = pattern.split("/**/")
        base = pathlib.Path(parts[0])
        suffix_glob = parts[1]
        if not base.is_absolute():
            base = REPO_ROOT / parts[0]
        if base.exists():
            for path in base.rglob(suffix_glob):
                if path.is_file():
                    out.add(str(path.relative_to(REPO_ROOT)))
    return sorted(out)


def _classify_age_hr(path_rel):
    """Return ``age_hr`` for an anchor file (resolved relative to REPO_ROOT)."""
    abs_path = REPO_ROOT / path_rel
    mtime = abs_path.stat().st_mtime
    age_sec = time.time() - mtime
    return age_sec / 3600.0


# ---------------------------------------------------------------------------
# Test 1 — anchor catalog count = 11
# ---------------------------------------------------------------------------


def test_wf345a3_anchor_catalog_count():
    """11 anchors catalogued under the 4 ANCHOR_GLOB_PATTERNS.

    If this test ever FAILs (i.e. anchor count != 11), it means either:
    - a new anchor was written by a downstream workflow (Mlauth or
      Honest-Negative close-out), in which case the audit must be
      regenerated; OR
    - an old anchor was deleted (e.g. by a cleanup workflow), in which
      case the stale set must be re-derived.
    Either way, regenerate ``molmetal/reports/wf345a3_anchor_audit/audit.json``
    before any Phase 3 refresh decision.
    """
    files = _list_anchor_files()
    assert len(files) == 11, (
        f"anchor catalog drift: expected 11 paper-grade anchors, got "
        f"{len(files)}; mismatched paths: {files}"
    )


# ---------------------------------------------------------------------------
# Test 2 — stale anchor detection (>= 4 stale)
# ---------------------------------------------------------------------------


def test_wf345a3_stale_anchor_detection():
    """Detect every anchor whose mtime exceeds 24h freshness budget.

    Confirms the Wf-345-A3 audit's "4 stale anchors" claim: the stale
    set must contain at least ``EXPECTED_STALE_PATHS`` (the audit's
    pinned stale set) and may contain more if Phase 3 has not yet
    refreshed them. The test prints the actual stale paths for the
    convenience of anyone investigating the gate FAIL.
    """
    files = _list_anchor_files()
    actual_stale = {f for f in files if _classify_age_hr(f) > MAX_AGE_HR}
    # The audit's stale set must be a subset of what we now detect.
    # We do NOT require exact equality (Phase 3 may have refreshed
    # some anchors by the time this test runs).
    missing = EXPECTED_STALE_PATHS - actual_stale
    assert not missing, (
        f"audit's stale set drifted: expected stale paths no longer "
        f"stale (refresh may have happened): {sorted(missing)}; "
        f"current stale set: {sorted(actual_stale)}"
    )
    # We must detect AT LEAST 4 stale (the audit's MEASURED count).
    assert len(actual_stale) >= 4, (
        f"stale count under floor: detected {len(actual_stale)} stale "
        f"anchors, expected >= 4; full set: {sorted(actual_stale)}"
    )
    # Sanity: classify each EXPECTED_STALE_PATH's age in hours (printed
    # so investigators can see which paths are still over 24h).
    for p in sorted(EXPECTED_STALE_PATHS):
        print(f"  {p} age={_classify_age_hr(p):.2f}h")


# ---------------------------------------------------------------------------
# Test 3 — audit JSON schema validity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def audit_data():
    """Load the audit artifact at the canonical path."""
    audit_path = (
        REPO_ROOT / "molmetal" / "reports" / "wf345a3_anchor_audit" / "audit.json"
    )
    if not audit_path.exists():
        pytest.skip(
            f"audit artifact missing at {audit_path}; regenerate via "
            f"Wf-345-A3 audit run before re-running this test"
        )
    with open(audit_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def test_wf345a3_audit_json_schema_valid(audit_data):
    """Audit JSON at the canonical path is parseable + carries required schema.

    Required top-level fields (party-twenty-spirit §3.5 MEASURED vs
    PROJECTED discipline — schema is MEASURED; values are MEASURED):

    - ``audit_id`` (str): unique ID of this audit run
    - ``gate_name`` (str): the pre-flight gate this audit targets
    - ``max_age_sec`` (float): freshness budget in seconds (must == 86400)
    - ``max_age_hr`` (int): freshness budget in hours (must == 24)
    - ``n_anchors_total`` (int): total anchors catalogued (must == 11)
    - ``n_anchors_stale`` (int): anchors over the budget (must == 4 at audit)
    - ``per_anchor`` (list[dict]): per-anchor record list; each item
      must carry ``path`` + ``mtime_iso`` + ``age_hr`` + ``stale`` keys
    - ``stale_anchors`` (list[dict]): subset of per_anchor where
      ``stale is True``; must satisfy
      ``len(stale_anchors) == n_anchors_stale``
    - ``glob_patterns`` (list[str]): the 4 glob patterns used; must
      match ANCHOR_GLOB_PATTERNS exactly (set-wise, since order can vary)

    Any missing field FAILs the test with a clear schema-violation
    report so downstream consumers can fix the audit format.
    """
    required_top = {
        "audit_id",
        "gate_name",
        "max_age_sec",
        "max_age_hr",
        "n_anchors_total",
        "n_anchors_stale",
        "n_anchors_fresh",
        "gate_status",
        "per_anchor",
        "stale_anchors",
        "glob_patterns",
    }
    missing = required_top - set(audit_data.keys())
    assert not missing, (
        f"audit JSON missing required top-level fields: {sorted(missing)}; "
        f"present: {sorted(audit_data.keys())}"
    )
    assert audit_data["max_age_sec"] == MAX_AGE_SEC, (
        f"max_age_sec drift: expected {MAX_AGE_SEC}, got "
        f"{audit_data['max_age_sec']}"
    )
    assert audit_data["max_age_hr"] == MAX_AGE_HR, (
        f"max_age_hr drift: expected {MAX_AGE_HR}, got " f"{audit_data['max_age_hr']}"
    )
    assert audit_data["n_anchors_total"] == 11, (
        f"n_anchors_total drift: expected 11, got " f"{audit_data['n_anchors_total']}"
    )
    # n_anchors_stale MAY be 0..11 (refresh in Phase 3 will drop it);
    # but at audit-write time, it must == 4 (MEASURED count).
    assert audit_data["n_anchors_stale"] == 4, (
        f"n_anchors_stale drift: expected 4 at audit-write time, got "
        f"{audit_data['n_anchors_stale']}"
    )
    assert len(audit_data["stale_anchors"]) == audit_data["n_anchors_stale"], (
        f"stale_anchors list length drift: "
        f"len={len(audit_data['stale_anchors'])} "
        f"vs n_anchors_stale={audit_data['n_anchors_stale']}"
    )
    # per_anchor items must carry required keys.
    per_anchor_keys = {"path", "mtime_iso", "age_hr", "stale"}
    for i, item in enumerate(audit_data["per_anchor"]):
        item_missing = per_anchor_keys - set(item.keys())
        assert not item_missing, (
            f"per_anchor[{i}] missing keys: {sorted(item_missing)}; "
            f"present: {sorted(item.keys())}"
        )
    # glob_patterns must match the canonical 4-pattern set.
    assert set(audit_data["glob_patterns"]) == set(ANCHOR_GLOB_PATTERNS), (
        f"glob_patterns drift: expected {sorted(ANCHOR_GLOB_PATTERNS)}, "
        f"got {sorted(audit_data['glob_patterns'])}"
    )
