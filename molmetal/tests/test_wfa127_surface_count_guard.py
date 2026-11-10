#!/usr/bin/env python
"""(§6.241 class guard) — the quoted affected-surface count must match reality.

**The class.**  A repair that changes an artifact invalidates every document
that describes it, and nothing in this repo sweeps for it (catalog §6.241).
Wf-A127 hit the class inside the wave that wrote §6.241: adding the 20th
guard test changed the affected test surface *after* the wave's verification
numbers had already been written into five documents.

**What ships here is a MECHANISM, not a method.**  Catalog §6.241's own STATE
line says the class "remains unguarded" — no gate re-reads prose against the
artifact it describes.  This test does exactly that for the one volatile
value the wave quoted: the pass count of the affected surface.

It works by **collecting** the surface (cheap, side-effect free, no GPU) and
requiring every document that quotes the surface's size to quote the
collected number — unless the quote sits under a **dated** correction marker,
in which case it is a preserved historical record (AGENTS.md §5.5 rule #6) and
is allowed to disagree.

**Which quantity.** The guarded quantity is the **collected** count — the
number a `pytest --collect-only` run reports — because that is the only one
the guard can measure without running the surface.  MEASURED 2026-11-02: the
affected surface collects 170 tests but runs 168 passed / 2 failed (the
disclosed USER-gated `p3_run_ledger_provenance` defect), so the collected and
passed counts DIFFER.  A guard that compared a *passed* quote against a
*collected* measurement would be unsatisfiable by a true statement, so the
documents quote the collected count and report the run outcome alongside.

A test added to or removed from the surface therefore breaks THIS test, which
forces the documents to be re-read before the wave can be called done.

**Scope.**  Only the affected-surface count is guarded.  Other counts in the
same documents (single-file counts, gate counts, registry counts) have their
own pins and are not read here — a guard that tries to check every number in
a document is a guard that will be silenced.

Deliberately kept OUT of the affected surface itself: a guard inside the
surface would change the surface's count, and the surface's count is what the
guard measures.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The 15 files whose joint pass count the A127 documents quote.
AFFECTED_SURFACE: tuple[str, ...] = (
    "molmetal/tests/test_wf415a3_preflight_fail_audit.py",
    "molmetal/tests/test_wf327f11_gate_count_consistency.py",
    "molmetal/tests/test_pre_flight_check.py",
    "molmetal/tests/test_wfa127_residue_closure.py",
    "molmetal/tests/test_wfa125_gpu_contract.py",
    "molmetal/tests/test_wfa125_root_conftest_gpu_lock.py",
    "molmetal/tests/test_wfa125_preflight_warn_contract.py",
    "molmetal/tests/test_wfa125_registry_sha_refresh.py",
    "molmetal/tests/test_wfa125_published_tree_guard.py",
    "molmetal/tests/test_wfa123_gpu_contract.py",
    "molmetal/tests/test_g6_receipt_integrity.py",
    "molmetal/tests/test_wf_a126_lever_fingerprint.py",
    "molmetal/tests/test_run_ledger.py",
    "molmetal/tests/test_bench_pb_valid.py",
    "molmetal/tests/test_wf32_phase3_pb_valid.py",
)

#: (document, regex, group index).  Every match must quote the collected count,
#: unless it is a preserved historical record under a dated marker.
#:
#: The close-out file's pattern is anchored on the **command line** that
#: precedes the 15-file block, because the same file also quotes a SINGLE
#: file's count ("20 passed, 1 warning") a few lines below — that number is a
#: different measurement and must not be checked against the surface count.
#: Wf-A269 wave-7 (2026-11-02) — the patterns capture the COLLECTED count.
#:
#: WHY (MEASURED 2026-11-02, CPU-only): the patterns below used to capture a
#: **passed** count while this guard MEASURED a **collected** count
#: (`pytest --collect-only`).  Those are two different quantities, and they
#: coincide only while the surface is 100 % green.  MEASURED 2026-11-02: the
#: surface collects 170 tests but runs 168 passed / 2 failed (the disclosed
#: Wf-A269 (2026-11-02, on publication of catalog §6.269.16) - CORRECTION:
#: the defect's home is **§6.269.9**, not §6.269.16.  The number §6.269.16
#: was RESERVED for it when this docstring was written; the numbering then
#: moved and §6.269.16 took the wave-5-closure entry instead.  The
#: superseded spelling is preserved verbatim below, and the seven-site census
#: of this collision is recorded at catalog §6.269.16.12.
#:
#: SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#:     #: USER-gated `p3_run_ledger_provenance` defect, catalog §6.269.16), so the
#: USER-gated `p3_run_ledger_provenance` defect, catalog §6.269.9), so the
#: guard was UNSATISFIABLE BY A TRUE STATEMENT — quoting the truthful "168
#: passed" was flagged stale, and quoting "170 passed" would have been a
#: fabrication.  The patterns now capture the quantity
#: `_collect_surface_count()` actually measures.
#:
#: The class this guard exists to catch is UNCHANGED: a test added to or
#: removed from the surface still breaks it, because the documents' live
#: quotes carry the collected count.
#:
#: SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#:     QUOTES: tuple[tuple[str, str, int], ...] = (
#:         (
#:             "docs/HONEST_NEGATIVE_CATALOG.md",
#:             r"\*\*15-file\*\* affected surface ->\s*\*\*(\d+) passed\*\*",
#:             1,
#:         ),
#:         (
#:             "docs/HONEST_NEGATIVE_CATALOG.md",
#:             r"affected surface \(\*\*(\d+) passed\*\* in",
#:             1,
#:         ),
#:         ("TODO/INDEX.md", r"\*\*15-file\*\* affected surface ->\s*\*\*(\d+) passed\*\*", 1),
#:         ("TODO/README.md", r"\*\*15-file\*\* affected surface ->\s*\*\*(\d+) passed\*\*", 1),
#:         (
#:             "TODO/completed/a127_residue_closure.md",
#:             r"<15 affected files>\n(\d+) passed",
#:             1,
#:         ),
#:     )
#: A `>` blockquote prefix may sit between a line break and the token that
#: follows it.  MEASURED 2026-11-04 (catalog §6.269.37): the patterns bridged
#: a plain line break with `\s*`, but `\s*` CANNOT cross the `>` that an
#: AGENTS.md §5.5 rule-#6 blockquote puts at the START of the next line — so a
#: blockquoted, LINE-BROKEN quote matched NOTHING and its site silently
#: became unwatched (a guard that cannot see a shape cannot catch that shape
#: going stale — catalog §6.269.22).  `BQ` bridges that prefix.  It is
#: OPTIONAL, so a plain-prose quote matches exactly as it did before.
BQ = r"(?:>[ \t]*)?"

QUOTES: tuple[tuple[str, str, int], ...] = (
    (
        "docs/HONEST_NEGATIVE_CATALOG.md",
        r"\*\*15-file\*\* affected surface ->\s*" + BQ + r"\*\*(\d+) collected\*\*",
        1,
    ),
    (
        "docs/HONEST_NEGATIVE_CATALOG.md",
        r"affected surface \(\*\*(\d+) collected\*\*",
        1,
    ),
    (
        "TODO/INDEX.md",
        r"\*\*15-file\*\* affected surface ->\s*" + BQ + r"\*\*(\d+) collected\*\*",
        1,
    ),
    (
        "TODO/README.md",
        r"\*\*15-file\*\* affected surface ->\s*" + BQ + r"\*\*(\d+) collected\*\*",
        1,
    ),
    (
        "TODO/completed/a127_residue_closure.md",
        r"<15 affected files>\n" + BQ + r"(\d+) tests? collected",
        1,
    ),
    # Wf-A269 wave-27 (2026-11-04) — the FOURTH rendering shape.  MEASURED by a
    # VALUE-SWEEP of the whole SSOT corpus (NOT by reading this pattern list — a
    # guard's own census is not a coverage measurement, catalog §6.244): the surface
    # count is quoted in FOUR shapes, and this one was in none of the first three —
    #
    #     **MEASURED 2026-11-02:** the surface collects **170** tests; ...
    #
    # (catalog §6.269.16.9's own measurement line, 1 site).  The site is DATED, so it
    # is not a false live claim — but a guard that cannot SEE a shape cannot catch
    # that shape going stale, and §6.269.22 rules that "DISCLOSED BUT NOT
    # ENFORCED" is itself the defect class.  Hence the pattern, and hence the
    # one-site blockquote conversion that satisfies it (catalog §6.269.37).
    (
        "docs/HONEST_NEGATIVE_CATALOG.md",
        r"the surface collects \*\*(\d+)\*\* tests",
        1,
    ),
    # Wf-A269 wave-27 (2026-11-04) — the FIFTH rendering shape, found by the
    # adversarial re-read (catalog §6.269.37).  MEASURED by a semantic sweep of
    # the guarded corpus (a line that NAMES the affected surface + a
    # collected-count token), NOT by reading this pattern list:
    #
    #     > * `pytest --collect-only -q` over the 15 `AFFECTED_SURFACE` files
    #       → **170 tests collected** (5.34 s). Unchanged.
    #
    # (catalog §6.269.16.9's own measurement bullet, 1 site).  The site is
    # DATED and block-quoted, so it is a preserved historical record — but
    # until this pattern existed it was matched by NOTHING, so the guard
    # could not see it at all.  §6.269.22 rules that "DISCLOSED BUT NOT
    # ENFORCED" is itself the defect class ⇒ the pattern, and the sweep test
    # below, so that the next unseen shape cannot arrive unnoticed.
    (
        "docs/HONEST_NEGATIVE_CATALOG.md",
        r"\*\*(\d+) tests collected\*\*",
        1,
    ),
)

#: The documents this guard reads.
GUARDED_DOCS: tuple[str, ...] = tuple(dict.fromkeys(rel for rel, _p, _g in QUOTES))

#: A line that quotes THE affected surface's count is recognised by an anchor
#: that NAMES the surface.  The bare number is not enough — the corpus quotes
#: many other collected counts (the top-level `tests/` tree alone contributes
#: one with 447).
SURFACE_ANCHOR = re.compile(
    r"affected surface|AFFECTED_SURFACE|surface collects|15 affected files|15-file",
    re.IGNORECASE,
)

#: A collected-count token, in any of its spellings.
COLLECTED_TOKEN = re.compile(r"(\d{2,3})\s*(?:\*\*)?\s*(?:tests?\s+)?(?:\*\*)?\s*collected")

#: The other rendering shape — "the surface collects **N** tests".
COLLECTS_TOKEN = re.compile(r"surface collects \*\*(\d+)\*\* tests")

#: A dated correction marker (AGENTS.md §5.5 rule #6) on a block-quoted line.
DATED_MARKER = re.compile(
    r"^>.*\b(CORRECTED|HISTORICAL|SUPERSEDED)\b.*\d{4}-\d{2}-\d{2}",
)

#: How many lines up a block-quoted correction may run before the date is lost.
BLOCKQUOTE_WINDOW = 12


def _collect_surface_count() -> int:
    """Collect (do not run) the affected surface and return its test count.

    Collect-only is side-effect free, so this can run inside a pytest run
    without re-entering the suite.  ``CUDA_VISIBLE_DEVICES`` is cleared
    because the whole wave is CPU-only (the 2026-10-31 user directive).
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *AFFECTED_SURFACE],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "collect-only of the affected surface failed "
            f"(rc={proc.returncode}); the guard cannot measure the count:\n"
            f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
        )
    m = re.search(r"(\d+) tests? collected", proc.stdout)
    if not m:
        raise AssertionError(
            "could not parse the collected count from pytest --collect-only:\n"
            f"{proc.stdout[-4000:]}"
        )
    return int(m.group(1))


def _is_preserved_historical_quote(lines: list[str], match_line_idx: int) -> bool:
    """True if the quoted number sits INSIDE a dated correction blockquote.

    AGENTS.md §5.5 rule #6 preserves a superseded measurement by quoting it
    inside a dated blockquote.  Such a quote is allowed to disagree with the
    live count.

    The test is deliberately positional, not proximity-based: a dated marker
    *nearby* does not legitimise a quote in plain prose.  An earlier version
    of this guard accepted any quote within N lines below a dated marker, and
    that let a live quote hide behind the historical quote's marker — the
    guard passed on a document whose live count was wrong.  Only a quote
    that is itself inside the dated blockquote is exempt.
    """
    if not lines[match_line_idx].lstrip().startswith(">"):
        return False
    for j in range(match_line_idx, max(-1, match_line_idx - BLOCKQUOTE_WINDOW), -1):
        if not lines[j].lstrip().startswith(">"):
            break
        if re.search(r"\d{4}-\d{2}-\d{2}", lines[j]):
            return True
    return False


def test_every_quoted_affected_surface_count_matches_the_collected_count() -> None:
    """Every LIVE document quote of the affected surface's count must be right.

    A repair that changes the surface changes this number; this test is what
    makes that change loud instead of silent (catalog §6.241).
    """
    collected = _collect_surface_count()

    stale: list[str] = []
    live_seen = 0
    for rel, pattern, group in QUOTES:
        path = REPO / rel
        assert path.exists(), f"guarded document is missing: {rel}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for m in re.finditer(pattern, text, flags=re.MULTILINE):
            line_no = text[: m.start(group)].count("\n")
            quoted = int(m.group(group))
            if quoted == collected:
                live_seen += 1
                continue
            if _is_preserved_historical_quote(lines, line_no):
                # A preserved historical record (AGENTS.md §5.5 rule #6).
                continue
            stale.append(f"{rel}:{line_no + 1} quotes {quoted}, collected {collected}")

    assert live_seen, (
        f"no document quotes the affected surface's collected count ({collected}) "
        "— either the documents stopped quoting it (the guard is now vacuous) "
        "or the phrasing changed and the guard must be updated to match."
    )
    assert not stale, (
        "the affected surface collects "
        f"{collected} tests, but these document quote(s) disagree — a repair "
        "changed the surface and the documents that describe it were not "
        f"re-read (catalog §6.241): {stale}"
    )


def test_the_guard_is_not_inside_the_affected_surface() -> None:
    """A guard inside the surface would change the surface's own count."""
    self_name = Path(__file__).resolve().relative_to(REPO).as_posix()
    assert self_name not in AFFECTED_SURFACE, (
        f"{self_name} guards the affected surface's count and must not be part "
        "of it, or the guard measures its own effect."
    )
    assert len(AFFECTED_SURFACE) == len(set(AFFECTED_SURFACE)), (
        "the affected surface list contains duplicates"
    )
    for rel in AFFECTED_SURFACE:
        assert (REPO / rel).exists(), f"affected-surface file is missing: {rel}"


def _matched_sites() -> set[tuple[str, int]]:
    """Every (document, line) the QUOTES patterns can see."""
    seen: set[tuple[str, int]] = set()
    for rel, pattern, group in QUOTES:
        text = (REPO / rel).read_text(encoding="utf-8")
        for m in re.finditer(pattern, text, flags=re.MULTILINE):
            seen.add((rel, text[: m.start(group)].count("\n") + 1))
    return seen


def _swept_sites() -> set[tuple[str, int]]:
    """Every (document, line) that quotes the surface's collected count.

    Found by a route INDEPENDENT of the pattern list: an anchor that names the
    affected surface, plus a collected-count token.  The two routes must agree
    — that is the point of the test below.
    """
    seen: set[tuple[str, int]] = set()
    for rel in GUARDED_DOCS:
        lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            prev = lines[i - 2] if i >= 2 else ""
            if not (SURFACE_ANCHOR.search(line) or SURFACE_ANCHOR.search(prev)):
                continue
            for rx in (COLLECTED_TOKEN, COLLECTS_TOKEN):
                for m in rx.finditer(line):
                    seen.add((rel, i))
    return seen


def test_every_surface_count_quote_site_is_watched_by_a_pattern() -> None:
    """A quote in a shape the pattern list does not contain is UNWATCHED.

    The pattern list is a census of the shapes the guard can SEE.  A site in
    any other shape is invisible to it: the guard passes while that document's
    claim rots.  MEASURED 2026-11-04: two such sites existed — the catalogue's
    own §6.269.16.9 measurement bullet (`**170 tests collected**`) and the
    close-out file's dated transcript (whose `> ` prefix broke the newline
    bridge) — and a guard that cannot see a shape cannot catch that shape
    going stale (catalog §6.269.22).

    The two routes are independent by construction: the sweep finds sites by
    MEANING (the line names the affected surface) and the patterns find them
    by SPELLING.  A site found by the sweep but missed by every pattern is a
    hole, and this test fails on it.
    """
    unmatched = sorted(_swept_sites() - _matched_sites())
    assert not unmatched, (
        "these quotes of the affected surface's collected count are matched by "
        "NO guard pattern, so the guard cannot see them and they can go stale "
        f"silently (catalog §6.269.37): {unmatched}"
    )
    undeclared = sorted(_matched_sites() - _swept_sites())
    assert not undeclared, (
        "these guard matches are not found by the semantic sweep, so the two "
        "routes disagree — either the sweep's anchor is too narrow or a "
        f"pattern matches prose that is not a surface quote: {undeclared}"
    )


def test_the_blockquote_line_break_bridge_works() -> None:
    r"""A block-quoted, line-broken quote must still MATCH (and stay exempt).

    Regression for the bridge above: with the old `\s*` the pattern matched
    NOTHING across a `>` prefix, so the site was not merely un-exempt — it was
    not seen at all.  MEASURED 2026-11-04: the catalogue's wave-7 block and
    the close-out file's dated transcript were both in that state.
    """
    sample = (
        "> **MEASURED 2026-11-02 (Wf-A269 wave-7 re-measurement, CPU-only):**\n"
        "> VERBATIM per AGENTS.md §5.5 rule #6:** The **15-file** affected surface ->\n"
        "> **170 collected** (168 passed, 2 failed) in 397.17s\n"
    )
    pattern = QUOTES[0][1]
    m = re.search(pattern, sample, flags=re.MULTILINE)
    assert m is not None, (
        "the block-quoted, line-broken quote does not match the pattern — the "
        "`>` prefix breaks the line-break bridge and the site is unwatched"
    )
    assert int(m.group(1)) == 170
    lines = sample.splitlines()
    assert _is_preserved_historical_quote(lines, 2) is True, (
        "a dated block-quoted quote must stay exempt from the live-count check"
    )
