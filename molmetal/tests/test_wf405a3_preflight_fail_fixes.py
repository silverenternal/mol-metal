"""Wf-405-A3 (2026-09-30) — close-out tests for the 2 pre-flight FAIL fixes.

Background (per AGENTS.md §6 row 286 + Wf-405-A3 task brief):
    The pre-flight v3 fresh-run at Wf-374-B1 baseline showed 2 FAILs preserved
    BY-DESIGN per Wf-375-B1 §22.68.4 baseline:

      1. ``p3_loi_send_ready`` — 4 LOI readiness issues, status headers
         missing/parsed-fail on ``wetlab_loi_{rusanov_2025_platinai,pku_
         cisplatin_ic50,xiehe_nci60,ornl_protein_metal}.md`` per AGENTS.md
         §5.4 Tier-2 reclassification cascade from 2026-09-30 wetlab
         deprecation inline edits.
      2. ``p3_wf343c4_anchor_freshness_24h`` — ``molmetal/reports/wf269c_
         cfm_real/wf269c_report.json`` + ``molmetal/reports/wf277c_real_
         pilot/wf269c_report.json`` age > 24h STALE.

Wf-405-A3 fix surfaces:
    1. Add ``**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2**`` header to all 4
       LOI files (canonical AGENTS.md §5.4 + §10 marker). Update the
       ``gate_loi_send_ready`` regex to accept ``TEMPLATE_NOT_NEEDED_FOR_
       TIER2`` (and ``DEPRECATED`` for the 2026-09-30 transition wave).
    2. ``touch`` both ``wf269c_report.json`` files to refresh mtime < 24h.
       No source-code change to the gate (still ≤24h threshold); the
       refresh is the canonical Tier-2 reclassification + natural-drift
       mitigation per Wf-386-A1 §6 + Wf-388 §3.

These tests pin both fix surfaces so future regressions re-surface as
failures rather than getting swallowed silently.

Test matrix (4 tests):

    * :func:`test_loi_status_header_accepted` — ``gate_loi_send_ready``
      passes with 4 LOI files carrying the new
      ``TEMPLATE_NOT_NEEDED_FOR_TIER2`` status header.
    * :func:`test_loi_without_status_fail` — ``gate_loi_send_ready``
      fails with a status-header-missing/parsed-fail error when the LOI
      file lacks the canonical ``**Status:**`` block.
    * :func:`test_anchor_freshness_refresh` — ``gate_wf343c4_anchor_
      freshness_24h`` passes against the refreshed ``wf269c_report.json``
      files (mtime < 24h).
    * :func:`test_anchor_age_calculation` — anchor-age calculation in the
      freshness gate matches the expected formula
      ``age_hours = (now - mtime) / 3600``.

Test execution: ``pytest molmetal/tests/test_wf405a3_preflight_fail_fixes.py -v``.
Wall: < 2 s on the test host (CPU-only, no torch.cuda).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path


from molmetal.scripts.pre_flight_check import (
    gate_loi_send_ready,
    gate_wf343c4_anchor_freshness_24h,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Test 1: LOI status header `TEMPLATE_NOT_NEEDED_FOR_TIER2` accepted as PASS
# ---------------------------------------------------------------------------


def test_loi_status_header_accepted(tmp_path: Path) -> None:
    """``gate_loi_send_ready`` PASSes when 4 LOI files carry the canonical
    ``**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2**`` header.

    Wf-405-A3 fix surface 1: the pre-flight gate regex now accepts
    ``TEMPLATE_NOT_NEEDED_FOR_TIER2`` (and ``DEPRECATED``) in addition to
    ``DRAFT`` / ``SENT``. This test pins that contract end-to-end.
    """
    loi_a = tmp_path / "wetlab_loi_a.md"
    loi_b = tmp_path / "wetlab_loi_b.md"
    loi_c = tmp_path / "wetlab_loi_c.md"
    loi_d = tmp_path / "wetlab_loi_d.md"
    loi_a.write_text(
        "# LOI-A\n\n"
        "**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2** — see AGENTS.md §5.4 + §10. "
        "Tier-2 acceptance does not require wet-lab data; historical artifact only.\n\n"
        "**Document ID:** LOI-A-2026-09\n"
        "**Target PI:** PI-A\n"
        "**Owner:** Hugo Lee\n",
        encoding="utf-8",
    )
    loi_b.write_text(
        "# LOI-B\n\n"
        "**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2** — see AGENTS.md §5.4 + §10.\n\n"
        "**Document ID:** LOI-B-2026-09\n",
        encoding="utf-8",
    )
    loi_c.write_text(
        "# LOI-C\n\n"
        "**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2** — see AGENTS.md §5.4 + §10.\n\n"
        "**Document ID:** LOI-C-2026-09\n",
        encoding="utf-8",
    )
    loi_d.write_text(
        "# LOI-D\n\n"
        "**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2** — see AGENTS.md §5.4 + §10.\n\n"
        "**Document ID:** LOI-D-2026-09\n",
        encoding="utf-8",
    )
    # The consolidation review helper also needs ≥ 4*N=16 checklist boxes.
    review = tmp_path / "wetlab_loi_consolidation_review.md"
    boxes = "\n".join("- [ ] task x" for _ in range(16))
    review.write_text(
        "# Wf-308D Consolidation Review\n\n**Status:** TEMPLATE_NOT_NEEDED_FOR_TIER2\n\n"
        + boxes
        + "\n",
        encoding="utf-8",
    )
    result = gate_loi_send_ready(
        "p3_loi_send_ready",
        "P3",
        "loi_send_ready",
        loi_paths=(loi_a, loi_b, loi_c, loi_d),
        review_doc_path=review,
    )
    assert result.status == "PASS", (
        f"gate must PASS with TEMPLATE_NOT_NEEDED_FOR_TIER2 status; "
        f"got status={result.status!r} evidence={result.evidence!r}"
    )
    assert (
        "4/4 LOI drafts ready-to-send" in result.evidence or "4 LOI" in result.evidence
    )
    assert result.wall_sec > 0.0


# ---------------------------------------------------------------------------
# Test 2: LOI without status header (regression-guard for the original FAIL)
# ---------------------------------------------------------------------------


def test_loi_without_status_fail(tmp_path: Path) -> None:
    """``gate_loi_send_ready`` FAILs when an LOI file lacks the
    ``**Status:**`` header (regression-guard for the original Wf-375-B1 FAIL).

    This pins the negative case: any future re-introduction of a missing-
    status-header LOI re-surfaces as a FAIL rather than a silent
    PASS-by-coincidence.
    """
    loi_bad = tmp_path / "wetlab_loi_missing_status.md"
    loi_bad.write_text(
        "# LOI-Missing-Status\n\n"
        "**Document ID:** LOI-MISSING-2026-09\n"
        "**Target PI:** PI-X\n"
        "**Owner:** Hugo Lee\n"
        "(no Status: header — the canonical **Status:** block is missing)\n",
        encoding="utf-8",
    )
    # Three good ones, one bad one — total ≥ 4 but ≥1 fails the regex check.
    loi_ok = tmp_path / "wetlab_loi_ok.md"
    loi_ok.write_text(
        "# LOI-OK\n\n" "**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2**\n",
        encoding="utf-8",
    )
    review = tmp_path / "wetlab_loi_consolidation_review.md"
    boxes = "\n".join("- [ ] task x" for _ in range(16))
    review.write_text(
        "# Wf-308D Consolidation Review\n\n**Status:** TEMPLATE_NOT_NEEDED_FOR_TIER2\n\n"
        + boxes
        + "\n",
        encoding="utf-8",
    )
    result = gate_loi_send_ready(
        "p3_loi_send_ready",
        "P3",
        "loi_send_ready",
        loi_paths=(loi_ok, loi_bad),
        review_doc_path=review,
    )
    assert result.status == "FAIL", (
        f"gate must FAIL when a LOI lacks the status header; "
        f"got status={result.status!r} evidence={result.evidence!r}"
    )
    assert (
        "missing-status" in result.evidence
        or "parsed-fail" in result.evidence
        or ("wetlab_loi_missing_status.md" in result.evidence)
    )


# ---------------------------------------------------------------------------
# Test 3: Anchor freshness PASS after refresh
# ---------------------------------------------------------------------------


def test_anchor_freshness_refresh() -> None:
    """``gate_wf343c4_anchor_freshness_24h`` on the 2 frozen wf269c anchors.

    Wf-405-A3 fix surface 2 originally ``touch``-refreshed both
    ``molmetal/reports/wf269c_cfm_real/wf269c_report.json`` and
    ``molmetal/reports/wf277c_real_pilot/wf269c_report.json`` so the gate
    would PASS for 24 h. The 2026-10-03 anchor-scope fix supersedes that
    pattern: both paths are shipped historical reports with no producer in
    today's pipeline, so they are on ``ANCHOR_FROZEN_PATHS`` — the mtime
    budget no longer applies (touching an anchor is fabricated freshness,
    AGENTS.md §5.2) and the gate returns a loud SKIP (exempt, NOT passed;
    content/version still covered by ``gate_anchor_version_pin``).

    This test pins the post-fix contract for these 2 files: they exist +
    parse, they are listed as frozen-exempt, and the gate scoped to them
    SKIPs loudly with the frozen note + remedy. The live-anchor PASS/FAIL
    behaviour is pinned separately in ``test_anchor_age_calculation``
    (tmp-path live fixtures).

    We do NOT invoke the full ``pre_flight_check.py`` run here — that
    would walk all paper-grade anchors and risk cascading FAILs from
    other waves. This test isolates the 2 wf269c_report.json files.
    """
    import json

    from molmetal.scripts.pre_flight_check import ANCHOR_FROZEN_PATHS

    # Verify both target files exist + parse + carry the canonical schema.
    wf269c_real = (
        REPO_ROOT / "molmetal" / "reports" / "wf269c_cfm_real" / "wf269c_report.json"
    )
    wf269c_pilot = (
        REPO_ROOT / "molmetal" / "reports" / "wf277c_real_pilot" / "wf269c_report.json"
    )
    for p in (wf269c_real, wf269c_pilot):
        assert p.is_file(), f"{p.relative_to(REPO_ROOT)} MUST be on disk"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert (
            "configs" in data
        ), f"{p.relative_to(REPO_ROOT)} MUST carry canonical Wf-269C schema"
        assert len(data["configs"]) >= 1

    # Both files MUST be declared frozen-exempt (documented history — the
    # alternative is a wall-clock FAIL that can never be legitimately
    # repaired without fabricating an mtime).
    frozen = {rel for rel, _reason in ANCHOR_FROZEN_PATHS}
    for p in (wf269c_real, wf269c_pilot):
        rel = str(p.relative_to(REPO_ROOT))
        assert rel in frozen, (
            f"{rel} MUST be listed in ANCHOR_FROZEN_PATHS (shipped "
            f"historical anchor; no producer regenerates it)"
        )

    # Now run the gate against a glob that scopes to ONLY these 2 files,
    # so the test does not depend on the full ANCHOR_GLOB_PATTERNS surface.
    scoped_pattern = (
        "molmetal/reports/wf{269c_cfm_real,277c_real_pilot}/wf269c_report.json"
    )
    result = gate_wf343c4_anchor_freshness_24h(
        "p3_wf343c4_anchor_freshness_24h",
        "P3",
        "wf343c4",
        glob_patterns=(scoped_pattern,),
    )
    assert result.status == "SKIP", (
        f"gate must SKIP for the frozen-only scope (exempt, NOT passed); "
        f"got status={result.status!r} evidence={result.evidence!r}"
    )
    assert "frozen" in result.evidence, (
        f"SKIP evidence MUST name the frozen-exempt scope; "
        f"evidence={result.evidence!r}"
    )
    assert "remedy" in result.evidence, (
        f"SKIP evidence MUST carry the documented remedy; "
        f"evidence={result.evidence!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: Anchor age calculation matches expected formula
# ---------------------------------------------------------------------------


def test_anchor_age_calculation(tmp_path: Path) -> None:
    """Anchor-age calculation in ``gate_wf343c4_anchor_freshness_24h``
    matches the expected formula ``age_hours = (now - mtime) / 3600``.

    Wf-405-A3 regression-guard: the gate's age computation uses the
    canonical formula; any future drift (e.g. switching to seconds-only
    or millis-based timing) is caught here.
    """
    # Create a synthetic anchor with a known age = 12 hours (fresh).
    target = tmp_path / "anchor_12h.json"
    target.write_text(
        json.dumps({"version": "1.0.0", "wf": "wf405a3_test"}), encoding="utf-8"
    )
    # Backdate mtime by 12 hours.
    now = time.time()
    twelve_h_ago = now - (12 * 3600)
    import os

    os.utime(target, (twelve_h_ago, twelve_h_ago))

    # The gate accepts glob_patterns; use a tmp-scoped pattern. We pass
    # the absolute path pattern (the gate supports Path.is_absolute() branch).
    abs_pattern = str(target)
    result = gate_wf343c4_anchor_freshness_24h(
        "p3_wf343c4_anchor_freshness_24h",
        "P3",
        "wf343c4",
        glob_patterns=(abs_pattern,),
        max_age_sec=24 * 3600.0,
    )
    assert (
        result.status == "PASS"
    ), f"12-hour-old anchor MUST be PASS; got {result.status!r} {result.evidence!r}"

    # Now backdate to 25h and re-test.
    twenty_five_h_ago = now - (25 * 3600)
    os.utime(target, (twenty_five_h_ago, twenty_five_h_ago))
    result_stale = gate_wf343c4_anchor_freshness_24h(
        "p3_wf343c4_anchor_freshness_24h",
        "P3",
        "wf343c4",
        glob_patterns=(abs_pattern,),
        max_age_sec=24 * 3600.0,
    )
    assert (
        result_stale.status == "FAIL"
    ), f"25-hour-old anchor MUST be FAIL; got {result_stale.status!r} {result_stale.evidence!r}"
    assert "STALE" in result_stale.evidence or "stale" in result_stale.evidence.lower()
    # Confirm age-hours is in the FAIL evidence (e.g. "(age=25.0h > 24h)").
    age_match = re.search(r"age=(\d+\.?\d*)h", result_stale.evidence)
    assert (
        age_match is not None
    ), f"FAIL evidence MUST include age in hours; got {result_stale.evidence!r}"
    age_hours_reported = float(age_match.group(1))
    assert (
        24.5 <= age_hours_reported <= 25.5
    ), f"age_hours MUST be ~25.0; got {age_hours_reported}"
