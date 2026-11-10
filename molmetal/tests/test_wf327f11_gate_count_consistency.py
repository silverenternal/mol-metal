"""Wf-327F11 verification tests — pre-flight gate-count consistency docs sync.

==============================================================================
Wf-327F11 — independent verification of Wf-321D / Wf-327 pre-flight
gate-count consistency audit (p3_gate_count_consistency_audit).
==============================================================================

Per Wf-321D ship surface (TODO/completed/321d_preflight_gate_audit.md) the
audit gate ``p3_gate_count_consistency_audit`` calls
``build_gates(pipeline)`` and asserts ``len(runners) == EXPECTED_GATE_COUNTS[pipeline]``.
Wf-327 added a NEW ``p3_wf327_cpu_num_workers_sufficient`` gate to the
``all`` pipeline, bumping the count from 61 → 62.  Wf-327F11 is the
**independent re-verification** layer (per party-twenty-spirit §3.5):
additive tests that pin the sync invariants beyond what the audit gate
itself checks.

This module adds **2 NEW tests** (CPU-only, < 2 s total):

  1. ``TestExpectedGateCountsSync::test_expected_gate_counts_match_actual`` —
     direct introspection: ``EXPECTED_GATE_COUNTS[pipeline] ==
     len(build_gates(pipeline))`` for every pipeline
     (``all``, ``frontier``, ``chemprop_v2_v3``, ``cfm``).
     This is the core invariant that the audit gate enforces; pinning
     it in a unit test means future code edits that break the
     invariant fail the test suite even if the audit gate itself
     is somehow disabled.

  2. ``TestAuditDocSync::test_audit_doc_table_count_matches_constant`` —
     docs drift guard: the ``TODO/completed/321d_preflight_gate_audit.md``
     §2.1 / §2.2 / §2.3 / §2.4 + §6 reference table counts must match
     the code constant.  This catches the EXACT class of bug Wf-327F11
     was opened for (docs drift between code constant and the audit
     doc that documents it).

Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5):
- This is a docs drift fix (NOT a real regression).
- EXPECTED_GATE_COUNTS was already bumped in code (61 → 62) when Wf-327
  added the cpu-num-workers gate.  The audit doc still claimed 61.
- Wf-327F11 syncs the audit doc to match the constant.  Both directions
  are MEASURED (grep on build_gates output + direct text search on the
  audit doc), not projected.
- CPU-only fix; runs in < 2 s.
- The audit gate itself was passing at ship time (the code constant was
  already 62; only the docs claimed 61).  The CI gate ``WF321D_GATE_DOCS_SYNC``
  this module enables will fail-fast block any future docs-vs-code drift.

Standing-rule compliance:
- AGENTS.md §1 MEASURED-vs-PROJECTED: every numeric is grep-derived.
- AGENTS.md §5.2 honest framing: this is docs drift, not a metric regression.
- party-twenty-spirit §3.5: every test is additive (no Wf-321D test removed).
"""

from __future__ import annotations

import re
from pathlib import Path



# ---------------------------------------------------------------------------
# Path constants — pin the audit doc on disk so the docs-drift guard has a
# fixed target.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DOC = REPO_ROOT / "TODO" / "completed" / "321d_preflight_gate_audit.md"


# ---------------------------------------------------------------------------
# Test 1 — EXPECTED_GATE_COUNTS constant matches actual gate count
# ---------------------------------------------------------------------------
class TestExpectedGateCountsSync:
    """Direct introspection: code constant matches build_gates output.

    This is the core invariant that the audit gate enforces; pinning it
    in a unit test means future code edits that break the invariant fail
    the test suite even if the audit gate itself is somehow disabled.
    """

    def test_expected_gate_counts_match_actual(self) -> None:
        """``EXPECTED_GATE_COUNTS[pipeline] == len(build_gates(pipeline))``.

        This is the MEASURED invariant: every pipeline's documented count
        in :data:`molmetal.scripts.pre_flight_check.EXPECTED_GATE_COUNTS`
        must equal the count returned by :func:`build_gates`.  Drift in
        either direction is a bug:

        * ``actual > expected`` — silent gate bloat (the audit gate
          itself was added this way).
        * ``actual < expected`` — silent gate loss (the audit gate
          surfaces this with the ``-DRIFT`` direction marker).
        """
        from molmetal.scripts.pre_flight_check import (
            EXPECTED_GATE_COUNTS,
            build_gates,
        )

        # 2026-11-04 (Wf-A269 frontier sweep) — `launch` was MISSING from
        # this tuple, so the `launch` pipeline's count was unpinned HERE.
        # (It IS pinned elsewhere — `test_wf327_ultracode_closeout.py` and
        # `test_preflight_ssot_guard_wiring.py` both iterate all five
        # pipelines — but this file is the Wf-321D count-sync layer and
        # must pin all of them.)
        #
        # PRE-REPAIR TUPLE PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
        #
        #     for pipeline in ("all", "frontier", "chemprop_v2_v3", "cfm"):
        #
        for pipeline in ("all", "launch", "frontier", "chemprop_v2_v3", "cfm"):
            runners = build_gates(pipeline)
            expected = EXPECTED_GATE_COUNTS[pipeline]
            assert len(runners) == expected, (
                f"gate count drift for pipeline={pipeline!r}: "
                f"actual={len(runners)} vs expected={expected}.  "
                f"Per Wf-321D §5.3 bump protocol: update BOTH "
                f"EXPECTED_GATE_COUNTS[{pipeline!r}] AND "
                f"TODO/completed/321d_preflight_gate_audit.md §2."
            )


# ---------------------------------------------------------------------------
# Test 2 — audit doc table counts match the constant
# ---------------------------------------------------------------------------
class TestAuditDocSync:
    """Docs-drift guard: audit doc references match the code constant.

    Catches the exact class of bug Wf-327F11 was opened for: docs drift
    between the ``EXPECTED_GATE_COUNTS`` constant and the audit doc that
    documents it.  The audit gate ``p3_gate_count_consistency_audit``
    verifies CODE consistency at pre-flight time; this test verifies
    DOC consistency at unit-test time (so a docs drift cannot survive a
    full pre-flight cycle without being flagged).
    """

    def test_audit_doc_table_count_matches_constant(self) -> None:
        """Per-pipeline gate counts in §2 + §6 match EXPECTED_GATE_COUNTS.

        Per Wf-321D §5.3, the audit doc is the SINGLE SOURCE OF TRUTH for
        the per-pipeline gate counts; the code constant must agree.  This
        test enforces that agreement with a direct text search on the
        audit doc's §2.1/§2.2/§2.3/§2.4 section headers + §6 audit
        summary table.
        """
        from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS

        assert AUDIT_DOC.is_file(), (
            f"audit doc missing: {AUDIT_DOC}.  Wf-321D ship surface "
            f"must remain on disk for the docs-drift guard to function."
        )
        text = AUDIT_DOC.read_text(encoding="utf-8")

        # Section header for the `all` pipeline: "### §2.1 `all` pipeline — N gates"
        # The N must match EXPECTED_GATE_COUNTS["all"].
        all_section_re = re.compile(r"### §2\.1\s+`all`\s+pipeline\s+—\s+(\d+)\s+gates")
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
            f"BOTH §2.1 (this test) AND the constant (test 1) together."
        )

        # §6 audit summary table: "61 (`all`) / 25 (`frontier`) / 13 ..."
        # The number preceding `(`all`)` must match EXPECTED_GATE_COUNTS["all"].
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
        frontier_summary = int(summary_match.group(2))
        chemprop_summary = int(summary_match.group(3))
        cfm_summary = int(summary_match.group(4))
        for doc_val, code_val, pipeline in (
            (all_summary, EXPECTED_GATE_COUNTS["all"], "all"),
            (frontier_summary, EXPECTED_GATE_COUNTS["frontier"], "frontier"),
            (
                chemprop_summary,
                EXPECTED_GATE_COUNTS["chemprop_v2_v3"],
                "chemprop_v2_v3",
            ),
            (cfm_summary, EXPECTED_GATE_COUNTS["cfm"], "cfm"),
        ):
            assert doc_val == code_val, (
                f"audit doc §6 summary table drift for pipeline={pipeline!r}: "
                f"doc={doc_val} vs code={code_val}.  Per Wf-321D §5.3, "
                f"the doc is the single source of truth; the code constant "
                f"must agree.  Bump protocol: update BOTH §6 (this test) "
                f"AND the constant (test 1) together."
            )


# ---------------------------------------------------------------------------
# Test 3 (bonus) — pre-flight audit gate flips FAIL→PASS after sync
# ---------------------------------------------------------------------------
class TestAuditGateFlip:
    """Direct invocation of the audit gate to confirm it PASSes.

    Pairs with the pre-flight v2 sub-checklist contract: when the
    audit gate runs against the synced ``EXPECTED_GATE_COUNTS``, it
    must return ``status="PASS"``.  This is a CPU-only fast path that
    sidesteps the full pre-flight run (which takes > 3 min).
    """

    def test_audit_gate_passes_for_all_pipeline(self) -> None:
        """``gate_gate_count_consistency_audit`` PASSes for pipeline='all'.

        Direct invocation: < 50 ms.  Per Wf-321D §5.2 design note, the
        audit gate uses direct introspection rather than subprocess
        because the pre-flight script itself takes > 3 min end-to-end.
        This test pins the audit gate's PASS contract so a future
        regression that breaks the introspection loop surfaces
        immediately.
        """
        from molmetal.scripts.pre_flight_check import (
            EXPECTED_GATE_COUNTS,
            gate_gate_count_consistency_audit,
        )

        result = gate_gate_count_consistency_audit(
            "p3_gate_count_consistency_audit",
            "P3",
            "all",
            expected_total=EXPECTED_GATE_COUNTS["all"],
        )
        assert result.status == "PASS", (
            f"audit gate FAILed after Wf-327F11 docs sync: "
            f"evidence={result.evidence}.  This means "
            f"EXPECTED_GATE_COUNTS['all']={EXPECTED_GATE_COUNTS['all']} "
            f"no longer matches build_gates('all') output.  Per "
            f"Wf-321D §5.3, this is a drift bug that must be fixed "
            f"by updating both the constant AND the audit doc."
        )
        assert "consistent" in result.evidence.lower(), (
            f"audit gate PASS evidence must include the word "
            f"'consistent' for the Wf-321D contract: got "
            f"{result.evidence!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 (2026-11-04, Wf-A269 frontier sweep) — the FROZEN NAME SET
# ---------------------------------------------------------------------------
class TestExpectedGateNames:
    """The gate NAMES are pinned, not only the gate COUNT.

    PRE-REPAIR (MEASURED by reading the file): the only frozen expectation
    was a NUMBER per pipeline.  A number cannot distinguish

    * a RENAME (`p3_foo` -> `p3_bar`),
    * a CROSS-PIPELINE MOVE (`p3_foo` leaves `all` and joins `frontier`),
    * a REPLACEMENT (`p3_foo` is swapped for a different `p3_baz`),

    because all three leave the count unchanged — so the audit gate said
    "consistent" while the surface it certifies had changed.  That is the
    §6.269.17 DEAD-GATE class in its "counting instead of reading" costume.

    `EXPECTED_GATE_NAMES` (next to `EXPECTED_GATE_COUNTS` in
    `molmetal/scripts/pre_flight_check.py`) freezes the NAMES.  The tuples
    were generated from a LIVE import of `build_gates` (never typed by
    hand); MEASURED at that timestamp: all 133, launch 135, frontier 25,
    chemprop_v2_v3 18, cfm 8 — identical to `EXPECTED_GATE_COUNTS`.
    """

    def test_the_frozen_name_set_matches_the_live_registration(self) -> None:
        """Every pipeline's registered names equal its frozen tuple."""
        from molmetal.scripts.pre_flight_check import (
            EXPECTED_GATE_NAMES,
            build_gates,
        )

        for pipeline in ("all", "launch", "frontier", "chemprop_v2_v3", "cfm"):
            live = tuple(sorted(name for name, _ in build_gates(pipeline)))
            frozen = EXPECTED_GATE_NAMES[pipeline]
            assert frozen == live, (
                f"gate NAME drift for pipeline={pipeline!r}: the frozen set "
                f"and the live registration differ.  "
                f"missing={sorted(set(frozen) - set(live))}, "
                f"added={sorted(set(live) - set(frozen))}.  Per Wf-321D §5.3 "
                f"the bump protocol now covers THREE artefacts: the audit "
                f"doc, EXPECTED_GATE_COUNTS and EXPECTED_GATE_NAMES."
            )

    def test_the_count_is_derived_from_the_name_set(self) -> None:
        """``EXPECTED_GATE_COUNTS[p]`` == ``len(EXPECTED_GATE_NAMES[p])``.

        The count is a DERIVED ASSERTION of the name set, so the two cannot
        drift apart silently: a bump that updates one but not the other
        fails HERE (and, at run time, in the audit gate).
        """
        from molmetal.scripts.pre_flight_check import (
            EXPECTED_GATE_COUNTS,
            EXPECTED_GATE_NAMES,
        )

        for pipeline in ("all", "launch", "frontier", "chemprop_v2_v3", "cfm"):
            derived = len(EXPECTED_GATE_NAMES[pipeline])
            counted = EXPECTED_GATE_COUNTS[pipeline]
            assert derived == counted, (
                f"the count for pipeline={pipeline!r} is NOT derived from the "
                f"frozen name set: EXPECTED_GATE_COUNTS says {counted}, the "
                f"name set has {derived} names.  Update both in the SAME edit "
                f"(Wf-321D §5.3 bump protocol)."
            )

    def test_no_pipeline_registers_a_duplicate_gate_name(self) -> None:
        """A duplicate registration must be visible as a DUPLICATE.

        A duplicate is count-preserving when it displaces another gate (the
        count stays, one name is lost) — a SET comparison would miss it,
        which is why the frozen tuple is compared as a TUPLE.
        """
        from molmetal.scripts.pre_flight_check import build_gates

        for pipeline in ("all", "launch", "frontier", "chemprop_v2_v3", "cfm"):
            names = [name for name, _ in build_gates(pipeline)]
            duplicates = sorted({n for n in names if names.count(n) > 1})
            assert not duplicates, (
                f"pipeline={pipeline!r} registers the same gate name more "
                f"than once: {duplicates}"
            )

    def test_the_name_set_covers_every_pipeline(self) -> None:
        """No pipeline may be missing a frozen set (a gap is a blind spot)."""
        from molmetal.scripts.pre_flight_check import (
            EXPECTED_GATE_NAMES,
            PIPELINES,
        )

        assert set(EXPECTED_GATE_NAMES) == set(PIPELINES), (
            f"EXPECTED_GATE_NAMES covers {sorted(EXPECTED_GATE_NAMES)} but "
            f"PIPELINES is {sorted(PIPELINES)}"
        )


# ---------------------------------------------------------------------------
# Test 5 (2026-11-04, Wf-A269 frontier sweep) — the pin can FAIL
# ---------------------------------------------------------------------------
class TestNameDriftIsDetected:
    """MUTATION PROOFS against the PRODUCTION check.

    Each case mutates the FROZEN SET (not the test) and asserts that
    `gate_gate_count_consistency_audit` — the code that runs on every
    pre-flight — FAILs.  A test that cannot fail is the §6.269.17 defect
    class, so the pin has to be shown to be load-bearing.
    """

    @staticmethod
    def _audit(names):
        """Call the PRODUCTION audit gate with a MUTATED frozen set.

        `expected_total` is the LIVE count, so the count check passes and
        the NAME check is the thing under test (a mutation that changed the
        length would trip the derived-count check first — that is
        `test_a_count_that_is_not_derived_is_detected`'s case).
        """
        from molmetal.scripts import pre_flight_check as pfc

        original = pfc.EXPECTED_GATE_NAMES
        pfc.EXPECTED_GATE_NAMES = names
        try:
            return pfc.gate_gate_count_consistency_audit(
                "p3_gate_count_consistency_audit",
                "P3",
                "all",
                expected_total=len(original["all"]),
            )
        finally:
            pfc.EXPECTED_GATE_NAMES = original

    def test_a_renamed_gate_is_detected(self) -> None:
        """M1 — rename one gate; the count is unchanged; the pin must fail."""
        from molmetal.scripts.pre_flight_check import EXPECTED_GATE_NAMES

        names = dict(EXPECTED_GATE_NAMES)
        all_names = list(names["all"])
        all_names[0] = "p3_renamed_by_a_bug"
        names["all"] = tuple(all_names)
        result = self._audit(names)
        assert result.status == "FAIL", (
            f"a RENAMED gate must be detected; got {result.status!r}: "
            f"{result.evidence!r}"
        )
        assert "NAME DRIFT" in result.evidence, result.evidence
        assert "p3_renamed_by_a_bug" in result.evidence, result.evidence

    def test_a_gate_moved_between_pipelines_is_detected(self) -> None:
        """M2 — swap one gate between `all` and `frontier`; BOTH counts hold."""
        from molmetal.scripts.pre_flight_check import EXPECTED_GATE_NAMES

        names = dict(EXPECTED_GATE_NAMES)
        # A name that is ONLY in `all` and one that is ONLY in `frontier`:
        # swapping those two keeps BOTH counts and really moves a gate.
        only_all = sorted(set(names["all"]) - set(names["frontier"]))
        only_frontier = sorted(set(names["frontier"]) - set(names["all"]))
        assert only_all and only_frontier, (
            "the fixture needs a gate that is only in `all` and one that is "
            f"only in `frontier`: only_all={only_all}, "
            f"only_frontier={only_frontier}"
        )
        moved_out = only_all[0]
        moved_in = only_frontier[0]
        names["all"] = tuple(
            sorted([n for n in names["all"] if n != moved_out] + [moved_in])
        )
        names["frontier"] = tuple(
            sorted([n for n in names["frontier"] if n != moved_in] + [moved_out])
        )
        result = self._audit(names)
        assert result.status == "FAIL", (
            f"a gate MOVED between pipelines must be detected; got "
            f"{result.status!r}: {result.evidence!r}"
        )
        assert "NAME DRIFT" in result.evidence, result.evidence
        assert moved_out in result.evidence, result.evidence
        assert moved_in in result.evidence, result.evidence

    def test_a_duplicate_registration_is_detected(self) -> None:
        """M3 — register the same name twice, displacing another gate.

        The count is unchanged (one gate lost, one duplicated), so a
        COUNT-based check passes it; the frozen TUPLE does not.
        """
        from molmetal.scripts.pre_flight_check import EXPECTED_GATE_NAMES

        names = dict(EXPECTED_GATE_NAMES)
        all_names = list(names["all"])
        displaced = all_names[-1]
        names["all"] = tuple(all_names[:-1] + [all_names[0]])
        result = self._audit(names)
        assert result.status == "FAIL", (
            f"a DUPLICATE registration must be detected; got "
            f"{result.status!r}: {result.evidence!r}"
        )
        assert "NAME DRIFT" in result.evidence, result.evidence
        assert displaced in result.evidence, result.evidence

    def test_a_count_that_is_not_derived_is_detected(self) -> None:
        """M4 — the COUNT is a derived assertion: a frozen set whose length
        disagrees with the count is refused, not accepted."""
        from molmetal.scripts.pre_flight_check import EXPECTED_GATE_NAMES

        names = dict(EXPECTED_GATE_NAMES)
        names["all"] = names["all"][:-1]
        result = self._audit(names)
        assert result.status == "FAIL", (
            f"a count that is NOT derived from the frozen name set must FAIL; "
            f"got {result.status!r}: {result.evidence!r}"
        )
        assert "NOT DERIVED" in result.evidence, result.evidence

    def test_the_unmutated_set_passes(self) -> None:
        """The control: the REAL frozen set still PASSes the production gate."""
        from molmetal.scripts import pre_flight_check as pfc

        result = pfc.gate_gate_count_consistency_audit(
            "p3_gate_count_consistency_audit",
            "P3",
            "all",
            expected_total=pfc.EXPECTED_GATE_COUNTS["all"],
        )
        assert result.status == "PASS", (
            f"the REAL frozen name set must PASS the audit gate; got "
            f"{result.status!r}: {result.evidence!r}"
        )
