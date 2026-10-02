"""Wf-344-B3 — gate count consistency audit drift regression tests.

==============================================================================
Wf-344-B3 — regression test layer for the Wf-338-A3 finding 11
``p3_gate_count_consistency_audit`` drift bug.

Per Wf-338-A3 finding 11 (TODO/INDEX.md §22.16 root cause #6 + §22.16
gate-fail #15): pre-flight gate ``p3_gate_count_consistency_audit``
emitted ``gate count drift for pipeline='audit': actual=0 vs
expected=57 [-DRIFT (silent gate loss)]``.  Root cause: the audit
gate was invoked with ``pipeline='audit'`` (a pre-Wf-321D stale
value) which is NOT in the :data:`PIPELINES` constant — so
:func:`build_gates` returned ``[]`` silently, and the gate FAILed
with the misleading drift message.

The Wf-344-B3 code fix added a ``PIPELINES`` membership guard to
:func:`gate_gate_count_consistency_audit` (returns FAIL with
``INVALID_PIPELINE`` evidence).  This module is the additive
verification layer that pins the contract so future regressions
fail the test suite before they reach the pre-flight run.

This module adds **4 NEW tests** (CPU-only, < 200 ms total):

  1. ``TestAuditPipelineInvalid::test_audit_gate_fails_with_invalid_pipeline_marker``
     — pins the INVALID_PIPELINE evidence contract for an invalid
     pipeline arg.  Direct regression for Wf-338-A3 finding 11.

  2. ``TestAuditPipelineInvalid::test_audit_gate_does_not_silently_return_zero``
     — pins the silent-zero failure contract: when an invalid
     pipeline is passed, the FAIL must mention INVALID_PIPELINE,
     NOT the misleading ``actual=0 vs expected=N`` drift message.

  3. ``TestExpectedGateCountsSync::test_expected_gate_counts_keys_match_pipelines``
     — pins the invariant that :data:`EXPECTED_GATE_COUNTS` keys
     must be a subset of :data:`PIPELINES`.  An orphan entry
     (e.g. ``audit``) cannot re-introduce the drift.

  4. ``TestAuditDocSync::test_audit_doc_section_2_1_count_matches_constant``
     — pins the docs-drift guard that Wf-327F11 §2.1 count matches
     :data:`EXPECTED_GATE_COUNTS['all']`.  Wf-321D §5.3 requires
     that the doc is the single source of truth + the code constant
     must agree.

Honest framing (per AGENTS.md §1 + §5.2 + party-twenty-spirit §3.5):

* ``invalid pipeline = 'audit'`` — direct regression for the historical
  Wf-338-A3 finding 11 evidence string.
* This is a docs / config drift fix (NOT a real metric regression).
* CPU-only fix; runs in < 200 ms.
* The audit gate was PASSing at ship time post-Wf-343C4 (expected=66,
  actual=66); the INVALID_PIPELINE guard is a fail-fast defense for
  future regressions.
* Wf-321D §5.3 audit doc already updated to 66 (per Wf-343C4 ship
  surface — §2.1 + §5.2 + §6 all carry the 66 value).

Standing-rule compliance:
- AGENTS.md §1 MEASURED-vs-PROJECTED: every numeric is grep-derived.
- AGENTS.md §5.2 honest framing: this is a docs/config drift fix.
- party-twenty-spirit §3.5: every test is additive (no existing
  Wf-321D / Wf-327F11 test removed).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path constants — pin the audit doc on disk so the docs-drift guard has a
# fixed target.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DOC = (
    REPO_ROOT / "TODO" / "completed" / "321d_preflight_gate_audit.md"
)


# ---------------------------------------------------------------------------
# Test 1 — INVALID_PIPELINE marker surfaces for invalid pipeline args
# ---------------------------------------------------------------------------
class TestAuditPipelineInvalid:
    """Wf-344-B3 — invalid pipeline regression tests (Wf-338-A3 finding 11).

    Per Wf-338-A3 finding 11 the audit gate FAILed with
    ``actual=0 vs expected=57`` for ``pipeline='audit'`` — a misleading
    drift message that hid the real bug (caller passed an invalid
    pipeline name).  The Wf-344-B3 code fix adds a ``PIPELINES``
    membership guard; these tests pin the new INVALID_PIPELINE
    evidence contract so a future regression that reverts the guard
    surfaces immediately in the test suite (without waiting for a
    full pre-flight run that takes > 3 min).
    """

    def test_audit_gate_fails_with_invalid_pipeline_marker(self) -> None:
        """``pipeline='audit'`` → FAIL with INVALID_PIPELINE evidence.

        Direct regression for Wf-338-A3 finding 11.  Pre-Wf-344-B3 this
        FAILed silently with ``actual=0 vs expected=N [-DRIFT (silent
        gate loss)]`` which masked the caller-side config bug.  Post-
        Wf-344-B3 the evidence must contain ``INVALID_PIPELINE`` so
        debugging is direct (the fix is to change the lambda to use
        a valid PIPELINES member, not to bump EXPECTED_GATE_COUNTS).
        """
        from molmetal.scripts.pre_flight_check import (
            gate_gate_count_consistency_audit,
        )
        result = gate_gate_count_consistency_audit(
            "p3_gate_count_consistency_audit",
            "P3",
            "audit",  # The historical invalid pipeline from Wf-338-A3
            expected_total=57,  # The historical expected value
        )
        assert result.status == "FAIL", (
            f"audit gate for invalid pipeline='audit' must FAIL; got "
            f"status={result.status}, evidence={result.evidence!r}.  "
            f"Per Wf-338-A3 finding 11 the original bug was a silent "
            f"zero-drift; the INVALID_PIPELINE guard is the Wf-344-B3 fix."
        )
        assert "INVALID_PIPELINE" in result.evidence, (
            f"audit gate FAIL evidence for invalid pipeline must contain "
            f"'INVALID_PIPELINE' marker; got {result.evidence!r}.  "
            f"This marker is the Wf-344-B3 contract — without it the "
            f"caller cannot distinguish a config bug (use a valid "
            f"PIPELINES member) from a code constant drift (bump "
            f"EXPECTED_GATE_COUNTS)."
        )
        assert "PIPELINES" in result.evidence, (
            f"audit gate FAIL evidence must list the valid PIPELINES "
            f"tuple so the caller knows what to use; got "
            f"{result.evidence!r}."
        )

    def test_audit_gate_does_not_silently_return_zero(self) -> None:
        """Invalid pipeline FAIL must NOT use the misleading drift message.

        Per Wf-338-A3 finding 11 the pre-Wf-344-B3 FAIL message was
        ``gate count drift for pipeline='audit': actual=0 vs
        expected=57 [-DRIFT (silent gate loss)]`` — this pointed the
        debugger at the wrong layer (EXPECTED_GATE_COUNTS constant)
        when the real bug was the caller-side pipeline arg.  Post-
        Wf-344-B3 the FAIL must NOT contain ``actual=0`` or
        ``[-DRIFT`` because those markers imply a constant drift
        rather than an invalid-pipeline bug.
        """
        from molmetal.scripts.pre_flight_check import (
            gate_gate_count_consistency_audit,
        )
        # Try a few different invalid pipeline names + expected counts to
        # ensure the silent-zero contract holds regardless of the
        # expected_total the caller passed.
        for invalid_pipeline in ("audit", "AUDIT", "frontier-audit",
                                 "all-audit", "", "audit2"):
            for bad_expected in (57, 58, 62, 66, 100):
                result = gate_gate_count_consistency_audit(
                    "p3_gate_count_consistency_audit",
                    "P3",
                    invalid_pipeline,
                    expected_total=bad_expected,
                )
                assert "actual=0" not in result.evidence, (
                    f"invalid pipeline={invalid_pipeline!r} FAIL must "
                    f"NOT contain 'actual=0' drift marker (Wf-338-A3 "
                    f"finding 11 silent-zero regression); got "
                    f"{result.evidence!r}.  The INVALID_PIPELINE guard "
                    f"short-circuits BEFORE build_gates so actual=0 "
                    f"never surfaces."
                )
                assert "[-DRIFT" not in result.evidence, (
                    f"invalid pipeline={invalid_pipeline!r} FAIL must "
                    f"NOT contain '[-DRIFT' silent gate loss marker; "
                    f"got {result.evidence!r}."
                )


# ---------------------------------------------------------------------------
# Test 3 — EXPECTED_GATE_COUNTS keys match PIPELINES
# ---------------------------------------------------------------------------
class TestExpectedGateCountsSync:
    """EXPECTED_GATE_COUNTS keys must be a subset of PIPELINES.

    Pins the invariant that no orphan entry (e.g. ``audit`` from
    pre-Wf-321D legacy code) can re-introduce the Wf-338-A3 finding
    11 silent-zero drift.  Pre-Wf-344-B3 this invariant was not
    tested explicitly; the bug surfaced only at full pre-flight run
    time (> 3 min wall).
    """

    def test_expected_gate_counts_keys_match_pipelines(self) -> None:
        """``EXPECTED_GATE_COUNTS.keys() ⊆ PIPELINES``.

        MEASURED invariant: every documented gate-count entry must
        be a valid pipeline name that :func:`build_gates` knows
        about.  An orphan key (e.g. legacy ``'audit'``) cannot
        silently survive because (a) :func:`build_gates` would
        return ``[]`` for it, and (b) a future Wf could read
        ``EXPECTED_GATE_COUNTS['audit']`` and propagate the
        silent-zero bug.
        """
        from molmetal.scripts.pre_flight_check import (
            EXPECTED_GATE_COUNTS,
            PIPELINES,
            build_gates,
        )
        expected_keys = set(EXPECTED_GATE_COUNTS.keys())
        pipelines_set = set(PIPELINES)
        orphans = expected_keys - pipelines_set
        assert not orphans, (
            f"EXPECTED_GATE_COUNTS has orphan keys not in PIPELINES: "
            f"{sorted(orphans)}.  Per Wf-338-A3 finding 11, an orphan "
            f"key (e.g. 'audit') caused silent-zero drift because "
            f"build_gates() returned [].  Either remove the orphan "
            f"key or add it to PIPELINES."
        )
        # Reverse invariant: PIPELINES members must all be registered
        # in EXPECTED_GATE_COUNTS.  Otherwise a future gate added to
        # build_gates would silently drift.
        missing = pipelines_set - expected_keys
        assert not missing, (
            f"PIPELINES has members not in EXPECTED_GATE_COUNTS: "
            f"{sorted(missing)}.  Every pipeline that build_gates "
            f"supports must have a documented expected count; "
            f"otherwise adding a gate to build_gates silently drifts."
        )
        # All counts must be positive integers.
        for pipeline, count in EXPECTED_GATE_COUNTS.items():
            assert isinstance(count, int) and count > 0, (
                f"EXPECTED_GATE_COUNTS[{pipeline!r}] = {count!r}; "
                f"must be a positive integer."
            )
            # And must equal build_gates() output.
            actual = len(build_gates(pipeline))
            assert actual == count, (
                f"EXPECTED_GATE_COUNTS[{pipeline!r}] = {count} drift "
                f"from build_gates() = {actual}.  Per Wf-321D §5.3, "
                f"update both this value AND "
                f"TODO/completed/321d_preflight_gate_audit.md §2 "
                f"together."
            )


# ---------------------------------------------------------------------------
# Test 4 — audit doc §2.1 count matches EXPECTED_GATE_COUNTS['all']
# ---------------------------------------------------------------------------
class TestAuditDocSync:
    """Audit doc references match the code constant.

    Wf-321D §5.3 mandates that the audit doc is the single source
    of truth for the per-pipeline gate counts; the code constant
    must agree.  Wf-327F11 ships a similar test for §2.1 + §6; this
    test extends the coverage to specifically pin the ``'all'``
    pipeline section header that the Wf-338-A3 finding 11 drift
    surfaced.  A future bump to ``EXPECTED_GATE_COUNTS['all']``
    without bumping the §2.1 doc string would fail this test.
    """

    def test_audit_doc_section_2_1_count_matches_constant(self) -> None:
        """Per-pipeline gate counts in §2.1 match EXPECTED_GATE_COUNTS.

        Per Wf-321D §5.3, the audit doc is the SINGLE SOURCE OF TRUTH
        for the per-pipeline gate counts; the code constant must agree.
        This test enforces that agreement with a direct text search on
        the audit doc's §2.1 section header (the specific surface
        that the Wf-338-A3 finding 11 drift bug surfaced).

        The test guards against the regression class where
        :data:`EXPECTED_GATE_COUNTS['all']` is bumped from 66 → N
        but the audit doc §2.1 still claims 66 (silent docs drift;
        the code constant would be the new N and the doc would be
        the old 66 — a class of bug Wf-327F11 explicitly guards).
        """
        from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS

        assert AUDIT_DOC.is_file(), (
            f"audit doc missing: {AUDIT_DOC}.  Wf-321D ship surface "
            f"must remain on disk for the docs-drift guard to function."
        )
        text = AUDIT_DOC.read_text(encoding="utf-8")

        # Section header for the `all` pipeline: "### §2.1 `all` pipeline — N gates"
        all_section_re = re.compile(
            r"### §2\.1\s+`all`\s+pipeline\s+—\s+(\d+)\s+gates"
        )
        all_section_match = all_section_re.search(text)
        assert all_section_match is not None, (
            f"audit doc §2.1 `all` pipeline section header not found in "
            f"{AUDIT_DOC}.  Wf-321D ship surface contract: §2.1 must "
            f"declare `### §2.1 \\`all\\` pipeline — N gates` with N = "
            f"EXPECTED_GATE_COUNTS['all'] = {EXPECTED_GATE_COUNTS['all']}."
        )
        all_section_count = int(all_section_match.group(1))
        assert all_section_count == EXPECTED_GATE_COUNTS["all"], (
            f"audit doc §2.1 `all` pipeline count drift: "
            f"doc={all_section_count} vs code={EXPECTED_GATE_COUNTS['all']}.  "
            f"Per Wf-321D §5.3, the doc is the single source of truth; "
            f"the code constant must agree.  Bump protocol: update "
            f"BOTH §2.1 (this test) AND the constant (test 3) together."
        )

        # §6 audit summary table: "N (`all`) / N (`frontier`) / N ..."
        # The number preceding `(`all`)` must match
        # EXPECTED_GATE_COUNTS["all"].
        summary_re = re.compile(
            r"(\d+)\s+\(`all`\)\s+/\s+(\d+)\s+\(`frontier`\)"
            r"\s+/\s+(\d+)\s+\(`chemprop_v2_v3`\)\s+/\s+(\d+)\s+\(`cfm`\)"
        )
        summary_match = summary_re.search(text)
        assert summary_match is not None, (
            f"audit doc §6 summary table format not found in {AUDIT_DOC}.  "
            f"Expected: N (`all`) / N (`frontier`) / N (`chemprop_v2_v3`) "
            f"/ N (`cfm`)."
        )
        all_summary = int(summary_match.group(1))
        assert all_summary == EXPECTED_GATE_COUNTS["all"], (
            f"audit doc §6 summary table drift for pipeline='all': "
            f"doc={all_summary} vs code={EXPECTED_GATE_COUNTS['all']}.  "
            f"Per Wf-321D §5.3, the doc is the single source of truth; "
            f"the code constant must agree."
        )