#!/usr/bin/env python
# Copyright 2026 The Mol-Metal Authors. Licensed under the Apache-2.0 License.
"""The PATH-TOKEN census's own invariants (Wf-A269 wave-32b, 2026-11-07).

**WHAT THIS FILE GUARDS.**  ``molmetal/scripts/wfa269_path_token_census.py``
measures both sides of the path-token repair (catalogue §6.269.42): the
PRE-FIX regex could not tokenise a citation at the END OF A SENTENCE, so a
broken pointer in that position was INVISIBLE to the path-reference guard.
The repair closed that shape.  The census is what makes the claim
re-derivable — so **the census itself needs a guard**, or it decays into a
report nobody re-runs (§6.269.29: a check that no runner executes).

**WHY THESE INVARIANTS AND NOT FROZEN COUNTS.**  Per §6.269.32 (*"a pin that
was born false"*) and §6.269.40.1 (*"a published batch number must name its
population"*), a frozen count is a pin that dies the moment the tree moves —
and the corpus counts here move on every edit.  So every assertion below is
either (a) a **RELATION** between two measured sides of the same repair, or
(b) a **CLOSURE** property that must hold for the corpus at hand, or (c) a
**MECHANISM** property proved by a GHOST probe that can never be vacuous.

**THE THREE INVARIANTS.**

1. **THE MECHANISM IS PROVED, NOT ASSUMED.**  The GHOST probe writes a broken
   path in 8 citation shapes.  The pre-fix regex must MISS the
   sentence-final shape and the live regex must SEE it.  This is independent
   of the corpus: it can never go vacuous, so the census itself can never
   become a census that cannot fail (§6.269.6).
2. **THE REPAIR STAYS VERDICT-PRESERVING.**  The ADMITTED token set (the
   tokens the resolver actually accepts, i.e. the ones that can move the
   guard's verdict) must be IDENTICAL under the pre-fix regex and the live
   one.  If it ever differs, the repair has stopped being neutral and the
   guard's BASELINE must be re-adjudicated (§6.242 / §6.269.35.7).
3. **THE CORPUS EXPOSURE IS ZERO-AND-STAYING.**  No path in the four SSOT
   documents may be simultaneously (a) in the blind shape, (b) INVISIBLE to
   the guard, and (c) DANGLING (does not exist on disk).  This is the growth
   detector: the moment somebody writes a broken pointer in the blind shape,
   this test FAILS.  Today the shape exists in the corpus but is already
   visible (a backtick terminates the token) ⇒ the defect was **LATENT**,
   not firing.

CPU-only. 0 GPU. Imports nothing outside the standard library (the guard
module is stdlib-only), so this file is CI-enforceable in the pytest-only
venv (no torch / numpy / rdkit needed).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from molmetal.scripts import wfa269_path_token_census as C  # noqa: E402
from molmetal.tests import wfa131_path_reference_guard as G  # noqa: E402


@pytest.fixture(scope="module")
def census() -> dict:
    """The census, computed ONCE (it reads four large documents)."""
    return C.census(G)


# ---------------------------------------------------------------------------
# INVARIANT 1 — the MECHANISM is proved by a GHOST probe that cannot be vacuous
# ---------------------------------------------------------------------------

def test_the_ghost_probe_covers_every_declared_shape(census: dict) -> None:
    """The probe must cover every shape the producer declares.

    If a shape were dropped from the probe, the mechanism proof would
    silently weaken — a probe that covers fewer shapes than it claims is a
    census that cannot fail.
    """
    probe = census["ghost_probe"]
    assert set(probe) == set(C.GHOST_SHAPES), (
        f"the probe covers {sorted(probe)} but the producer declares "
        f"{sorted(C.GHOST_SHAPES)}"
    )
    assert len(probe) == 8, f"expected 8 citation shapes, got {len(probe)}"


def test_the_blind_shape_is_open_in_the_prefix_regex_and_closed_in_the_live_one(
    census: dict,
) -> None:
    """THE MECHANISM PROOF — the reason the repair was made.

    The pre-fix regex's lookahead excluded ``.``, so a citation at the end of
    a sentence produced NO TOKEN AT ALL: the greedy class swallowed the
    sentence period, the backtrack that would have yielded the path was
    rejected by the lookahead, and the whole match FAILED.  A broken pointer
    there was therefore invisible — not "extracted then rejected", but never
    tokenised.

    The live regex must see the shape (the repair works) and the pre-fix one
    must miss it (the defect was real).  Both halves are asserted: a
    regression that breaks the LIVE regex would also fail here.
    """
    blind = census["ghost_probe"]["sentence-final (period)"]
    assert blind["seen_by_prefix_regex"] is False, (
        "the PRE-FIX regex now SEES the sentence-final shape — the defect "
        "this catalogue entry documents is no longer reproducible, so the "
        "entry's claim must be re-measured before it is quoted"
    )
    assert blind["seen_by_live_guard"] is True, (
        "the LIVE regex does NOT see the sentence-final shape — the repair "
        "has been reverted or broken"
    )


def test_every_other_shape_was_already_visible_to_the_prefix_regex(
    census: dict,
) -> None:
    """The defect was ONE shape, not a class of shapes.

    If the pre-fix regex now misses a shape it used to see, then the
    mechanism claim ("only the sentence-final shape was blind") is false and
    the entry must be corrected.  This assertion is what makes the claim
    *falsifiable* rather than decorative.
    """
    blind_only = [name for name, g in census["ghost_probe"].items()
                  if not g["seen_by_prefix_regex"]]
    assert blind_only == ["sentence-final (period)"], (
        f"the set of blind shapes changed: {blind_only} — the pre-fix regex "
        f"now misses {len(blind_only)} shape(s), so the published mechanism "
        f"claim must be re-measured"
    )


def test_the_live_guard_reports_a_broken_sentence_final_pointer() -> None:
    """The falsifiable half, end to end, against the LIVE guard.

    A broken pointer at the end of a sentence must be CLASSIFIED, not merely
    tokenised — and it must land in a FAILING bucket, or the guard would not
    block on it (§6.269.17: a check that cannot fail).
    """
    bad = "docs/wfa269_definitely_not_a_real_file_zzz.md"
    res = G.scan_texts(REPO, {"probe": f"See {bad}."})
    assert bad in res, f"a broken sentence-final pointer is INVISIBLE: {res}"
    bucket = res[bad][0]
    assert bucket in G.FAILING_BUCKETS, (
        f"the broken pointer was classified {bucket!r}, which is not a "
        f"failing bucket — the guard would not block on it"
    )


# ---------------------------------------------------------------------------
# INVARIANT 2 — the repair stays VERDICT-PRESERVING
# ---------------------------------------------------------------------------

def test_the_admitted_token_set_is_identical_on_both_sides(census: dict) -> None:
    """The ADMITTED set is what can move the guard's verdict.

    If the repair changed the admitted set, the guard's BASELINE would have
    to be re-adjudicated — and until it is, every failing count in the guard
    would be ambiguous.  Today the sets are byte-identical, so the repair
    needed NO BASELINE adjudication.

    This is asserted as a RELATION (the two sides agree), never as a frozen
    COUNT (the corpus moves on every edit).
    """
    assert census["admitted_set_changed"] is False, (
        "the ADMITTED token set differs between the pre-fix regex and the "
        "live one — the repair is no longer verdict-preserving and the "
        "guard's BASELINE must be re-adjudicated: "
        f"{ {k: v for k, v in census['per_document'].items() if v['added'] or v['removed']} }"
    )


def test_no_document_lost_or_gained_an_admitted_token(census: dict) -> None:
    """Per-document half of the same invariant, naming the offender."""
    for rel, d in census["per_document"].items():
        assert not d["added"], f"{rel}: {len(d['added'])} admitted token(s) ADDED"
        assert not d["removed"], (
            f"{rel}: {len(d['removed'])} admitted token(s) REMOVED — "
            f"{d['removed'][:3]}"
        )


def test_the_raw_token_stream_moves_but_the_admitted_stream_does_not(
    census: dict,
) -> None:
    """THE TWO-SIDED PROOF.

    The raw stream is what the pre-fix regex SUPPRESSED: a sentence-final
    word-like token was not merely mis-classified, it was never emitted.  So
    the raw count MUST move under the repair — that is the mechanism, measured
    rather than asserted.

    The admitted stream is what can move the guard's verdict — and it MUST NOT
    move.  A repair that changed both would be a silent verdict change.

    The invariant is the RELATION, not the magnitude: the raw delta is
    corpus-dependent and is published with its population (§6.269.40.1).
    """
    raw = census["raw_token_counts"]
    assert raw["delta"] != 0, (
        "the raw token stream did NOT move under the repair — the pre-fix "
        "regex's suppression of sentence-final tokens is not measurable on "
        "this corpus, so the mechanism claim needs re-measuring"
    )
    assert census["admitted_set_changed"] is False, (
        "the admitted stream moved — the repair is not verdict-preserving"
    )


# ---------------------------------------------------------------------------
# INVARIANT 3 — the CORPUS EXPOSURE is zero-AND-STAYING (the growth detector)
# ---------------------------------------------------------------------------

def test_the_census_is_not_vacuous(census: dict) -> None:
    """The FIRST version of this census was vacuous BY CONSTRUCTION.

    It counted emitted tokens ending in ``.`` — but the token's final
    character class is ``[A-Za-z0-9_]``, so that count is **always 0** and
    the census could not fail (§6.269.6).  The census was therefore rebuilt
    around the LOOSE path+period shape, which CAN be non-zero.

    ⇒ the loose census must be non-vacuous, i.e. the blind shape must
    actually OCCUR in the corpus.  If it stops occurring, the census has
    become vacuous again and must be rebuilt rather than trusted.
    """
    loose = census["loose_path_period_census"]
    assert loose["total"] >= 1, (
        "the LOOSE path+period census found 0 shapes — the census has become "
        "VACUOUS again (the blind shape no longer occurs in the corpus, so "
        "the census cannot fail).  A census that cannot fail is not a census."
    )


def test_the_end_dot_census_is_zero_by_construction(census: dict) -> None:
    """The VACUOUS census is kept, with its reason attached.

    It is published as 0 BY CONSTRUCTION, not as a measurement of safety.
    Deleting it would hide the lesson; keeping it without the reason would
    make it look like evidence.
    """
    ed = census["end_dot_census"]
    assert ed["prefix"] == 0 and ed["live"] == 0, (
        "the end-dot census moved — that would mean the token's final "
        "character class changed, and the '0 by construction' note would "
        "have to be re-derived"
    )
    assert "0 BY CONSTRUCTION" in ed["note"], (
        "the end-dot census lost its '0 BY CONSTRUCTION' disclosure — it "
        "would then read as a measurement of safety rather than a "
        "structural impossibility"
    )


def test_no_failing_pointer_hides_in_the_blind_shape(census: dict) -> None:
    """THE GROWTH DETECTOR — the assertion that can actually fail later.

    Today: the blind shape occurs in the corpus exactly **once**, the pre-fix
    regex could not see it, and it happens to point at a file that EXISTS
    ⇒ the defect was **LATENT** (exposure without harm).  If somebody later
    writes a BROKEN pointer in the blind shape, it would be invisible AND
    failing ⇒ this test FAILS.

    The measure is POSITION-LEVEL (a span must cover the citation's own
    position on its own line) — see
    :func:`test_the_exposure_measure_is_position_level_not_document_level`.
    """
    loose = census["loose_path_period_census"]
    assert loose["invisible_to_guard"] == 0, (
        f"{loose['invisible_to_guard']} citation(s) are in the blind shape "
        f"and INVISIBLE to the guard: {loose['examples'][:5]} — the repair "
        f"has regressed"
    )
    assert loose["of_which_failing"] == 0, (
        f"{loose['of_which_failing']} FAILING citation(s) hide in the blind "
        f"shape: {loose['examples'][:5]} — a broken pointer is invisible to "
        f"the path-reference guard"
    )


def test_the_exposure_measure_is_position_level_not_document_level(
    census: dict,
) -> None:
    """THE MEASURE IS THE RIGHT ONE — and the wrong one is pinned as wrong.

    The FIRST version of this census asked a DOCUMENT-LEVEL question: "is
    this path visible somewhere in the document?".  That is BLIND to the
    thing it claims to measure, because the corpus's single blind citation
    names a path that is ALSO cited — in a visible form — elsewhere in the
    same document.  ⇒ the document-level set difference would report
    "not invisible" while the citation at that position IS invisible.

    ⇒ **A DOCUMENT-LEVEL SET DIFFERENCE CANNOT MEASURE A POSITION-LEVEL
    BLIND SPOT.**  (Found by the M1 mutation of this wave's own proof: the
    document-level growth detector did NOT turn red when the regex was
    reverted, while the position-level one did.)

    This test asserts the disagreement EXISTS, so a future refactor that
    silently goes back to the set difference is caught.
    """
    loose = census["loose_path_period_census"]
    assert loose["examples"], (
        "the census found no blind-shape citation — the exposure measure "
        "has nothing to measure, and the census would be VACUOUS"
    )
    resolver = G._resolver(REPO)
    checked = 0
    for ex in loose["examples"]:
        if ex["prefix_seen"]:
            continue          # the pre-fix regex saw it — nothing to compare
        text = (REPO / ex["document"]).read_text(encoding="utf-8")
        # DOCUMENT-LEVEL visibility: the path IS somewhere in the document
        assert ex["path"] in set(resolver.extract(text)), (
            f"{ex['document']}: the blind citation's path is NOT visible "
            f"anywhere in the document, so this test cannot demonstrate the "
            f"position/document disagreement"
        )
        checked += 1
    assert checked >= 1, (
        "no blind-shape citation was exposed to the pre-fix regex — the "
        "exposure measure is not exercising the defect it documents"
    )


def test_the_blind_shape_verdict_is_latent_not_firing(census: dict) -> None:
    """The verdict must be reported at the resolution the measurement supports.

    FIRING = the pre-fix regex could not see a citation the guard would call
    FAILING.  LATENT = it could not see one, but that one resolves.
    NOT-EXPOSED = the corpus never uses the shape.

    **THIS IS A COHERENCE CHECK, NOT A TRUTH CHECK** — and the distinction is
    measured, not asserted: the M4 mutation of this wave's own proof broke the
    position-level measure (so the counters collapsed to "exposed = 0" ⇒ the
    verdict legitimately became NOT-EXPOSED) and this test stayed **GREEN**.
    A broken measure and a corpus that does not use the shape are
    INDISTINGUISHABLE to a coherence check ⇒ the TRUTH is pinned by
    :func:`test_the_exposure_measure_is_position_level_not_document_level`,
    which M4 does redden.  The pair is load-bearing; either alone is not.

    (The M5 mutation pins this test's own logic: it forces the verdict to
    "FIRING" and this test must go RED.)
    """
    loose = census["loose_path_period_census"]
    if loose["of_which_failing_exposed"]:
        expected = "FIRING"
    elif loose["exposed_to_prefix_regex"]:
        expected = "LATENT"
    else:
        expected = "NOT-EXPOSED"
    assert loose["verdict"] == expected, (
        f"the verdict {loose['verdict']!r} does not follow from "
        f"exposed_to_prefix_regex={loose['exposed_to_prefix_regex']} / "
        f"of_which_failing_exposed={loose['of_which_failing_exposed']}"
    )


# ---------------------------------------------------------------------------
# The census must be RE-RUNNABLE — a producer nobody re-runs is a report
# ---------------------------------------------------------------------------

def test_the_producer_runs_end_to_end_and_writes_a_json_artefact(
    tmp_path: Path,
) -> None:
    """The producer must be runnable as a SCRIPT and emit a machine artefact.

    §6.269.29: a check that no runner executes is indistinguishable from a
    check that does not exist.  This asserts the entry point works, that the
    artefact is JSON, and that it carries the census's own disclosure about
    which documents it covers.

    **HONEST LIMIT — this test is NOT mutation-reachable.**  It runs the
    ON-DISK producer as a subprocess, so an in-process mutation of the
    imported module cannot reach it (the same class as §6.269.23's
    "the flag was assembled at run time").  It therefore pins the CLI
    surface (argument parsing + JSON emission), while the module-scoped
    ``census`` fixture pins ``census()`` itself — and the fixture IS
    mutation-covered.  The two surfaces together cover the producer; neither
    covers the other.
    """
    out = tmp_path / "census.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "molmetal/scripts/wfa269_path_token_census.py"),
         "--out", str(out)],
        cwd=str(REPO), capture_output=True, text=True,
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert proc.returncode == 0, f"the census producer failed:\n{proc.stderr}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["documents"] == list(C.DOCUMENTS)
    assert "AGENTS.md is NOT" in payload["documents_note"], (
        "the artefact must disclose that AGENTS.md is outside the census — "
        "the guard does not scan it, and a silent omission would read as "
        "coverage"
    )
    assert payload["admitted_set_changed"] is False
    assert payload["loose_path_period_census"]["verdict"] == "LATENT"
