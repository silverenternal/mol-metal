# SPDX-License-Identifier: Apache-2.0
"""Wf-A152 — census guard coverage: the blind spots the Wf-A134 guard cannot see.

WHY THIS EXISTS
---------------
Wf-A134's guard (``test_wfa134_census_requote_guard.py``) polices the census's
HEADLINE rates through a CLOSED VOCABULARY of hand-picked retired numbers
(``STALE_RENDERINGS``) plus PARAGRAPH-SCOPED historical exemptions.  It
reported **0** sites on today's tree.  That zero was misleading.

MEASURED 2026-10-31 by Wf-A152: with the guard's own exemptions neutered, the
same scan reports **40** sites (15 rate + 1 min-claim + 24 flip).  Hand-
classifying all 40 found **4 LIVE-STALE blocks in 3 documents** that the
guard structurally cannot see, and it found them for a reason unrelated to the
vocabulary: the guard's ``MIN_CLAIM_RE`` exempts the WHOLE unit as soon as the
word "min" appears anywhere in it, and every one of the four blocks quotes a
``min`` next to its median.

The seven mechanisms, each MEASURED:

  1. ``HIST_LINE_RE`` exempts a WHOLE PARAGRAPH on the strength of ONE
     historical word anywhere in it.
  2. ``STALE_RENDERINGS`` is a CLOSED vocabulary of hand-picked numbers, so a
     retired value that is not on the list is invisible.
  3. ``DOC_CONTEXT_RE`` only looks at lines carrying a keyword from its list;
     a census claim that avoids the keywords is never examined.
  4. The null-vs-``0.0`` convention is not expressible as a numeric pattern.
  5. Vocabulary gaps — a retired value nobody listed.
  6. ``MIN_CLAIM_RE`` exempts the WHOLE UNIT when "min"/"minimum" appears
     anywhere in it.
  7. ``PER_RUN_CONTEXT_RE`` is LINE-scoped, so a paragraph that mixes a
     per-run reading with a census reading is exempted wholesale.

WHAT THIS GUARD ASSERTS
-----------------------
A. **The census's own changelog prose agrees with its own ``runs[]`` array.**
   The Wf-A150 changelog entry attributed ``cfm_fm_control_2026_10_09``'s
   best-epoch rate to ``cfm_bond_head_30ep_onecycle_2026_10_09`` and vice
   versa; the published artifact carried the swap and NO gate in the repo
   could detect it — ``check_pinned_values``, ``check_generation_metadata``
   and ``check_registry_sha`` all PASSed on the defective artifact.

B. **Every census statistic a document quotes is the CURRENT value.**
   The oracle is the ARTIFACT, not a hand-written list of retired numbers.
   Each claim is resolved to (population x epoch x cohort x key) and compared
   with the artifact's own current value for that tuple.  This closes
   mechanisms 2, 3, 5 and 6 with NO curated vocabulary.

C. **A run whose rate is ``null`` in the artifact is not rendered as
   ``0.0``/``0.0000``** in a live census-context sentence.

D. **The guard really covers the class that bit us** — the four blocks
   Wf-A152 struck must be caught by B, or the guard is decorative.

E. **Mutation tests** — the guard must FAIL on the defects it exists for.

HOW A CLAIM IS RESOLVED
-----------------------
A census claim is a NUMBER introduced by a statistic word (``median``,
``mean``, ``max``/``maximum``, ``min``/``minimum``), by ``key = N``
(``n = 27``, ``n_below_0.50 = 23``), or by a counter phrase
(``23 of 27 below 0.50``, ``16 below 0.10``).

A statistic word only introduces a number when NOTHING between the word and
the number says otherwise: a UNIT word means the number is a duration
("~60 min CPU + ~31.9 hr GPU") and a THRESHOLD word means it is a cutoff
("… / mean 0.6376953125 / 5 at or above 0.50").

The (population, epoch, cohort) is then resolved, in this order:

  1. a **jsonpath anchor** in the text (``fragmentation_summary.
     end_to_end_single_molecule_rate`` and friends) names population AND
     epoch outright;
  2. a **census qualifier parenthetical** overrides the running state — the
     INNERMOST one CONTAINING the claim first ("(ALL, n = 9)" makes the
     claim an ALL-runs count), then the nearest one FOLLOWING it
     ("0.125 / 0.0364…  (last epoch, CLEAN)").  A qualifier that names
     nothing, or names two things, names nothing;
  3. otherwise the **running state** — the population / epoch / cohort labels
     seen so far, left to right;
  4. a **cohort pair** — ``A / B`` where the sentence names exactly two
     cohorts assigns them positionally, in order of first appearance.

A claim whose population is undetermined is SKIPPED (never guessed).  When
only the epoch is undetermined the claim is checked against the UNION over
epochs — a value that was never current for that population is still caught.

A LIST ITEM is a sentence boundary: a new bullet is a new topic, so the
running population / epoch / cohort must not cross it.  MEASURED
consequence of not having it: a Ridge sample size ("n=200 SMILES") was
checked against the census run count, and a project-wide median was checked
against a cohort median, both because an unrelated bullet had opened with
the word "CLEAN".

KNOWN LIMITS (honest, not hidden)
---------------------------------
* A table whose header names no statistic (a free-form
  ``stale claim | corrected`` table) is skipped: its left column is by
  construction a historical claim.
* A claim on the RIGHT-hand side of a transition arrow (``old -> new``) is
  treated as history: an arrow states a historical move, so the guard cannot
  catch a transition whose destination has since moved again.  The two known
  sites of that shape (``TODO/INDEX.md`` L1105 and L1337) are annotated by
  hand instead — catalog **§6.264**.
* A sentence naming TWO OR MORE cohorts, whose claims are not in a
  ``A / B`` pair, has its cohort-scoped claims skipped — the assignment is
  lexical and the documents do disagree with themselves about which cohort a
  trailing ``n=`` belongs to.
* The pair convention (``A / B`` = cohort1 / cohort2) is a DOCUMENT
  convention, not a law; it is applied only when the sentence names exactly
  two cohorts, in order of first appearance.
* A claim whose population is stated only AFTER it ("the project-wide median
  0.0625 over the 27 CLEAN runs") is skipped: the running state has not
  reached the population yet.  The guard never reads a claim backwards.

0 GPU.  0 ``paper/*`` touches (TODO-84).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "metrics" / "by_metric" / "cfm_run_census.json"

from molmetal.tests.test_wfa134_census_requote_guard import (  # noqa: E402
    _paragraphs,
    _table_rows,
    doc_files,
    load_artifact,
)

# --------------------------------------------------------------------------
# A. changelog prose  <->  runs[]
# --------------------------------------------------------------------------
#: The two runs the Wf-A150 changelog entry names, with the rates the
#: artifact's own ``runs[]`` array carries for them.
WF_A150_RUNS = {
    "cfm_fm_control_2026_10_09": 0.6875,
    "cfm_bond_head_30ep_onecycle_2026_10_09": 0.875,
}

#: Runs the artifact publishes with a ``null`` last-epoch rate.
NULL_RUNS = (
    "cfm_fm_control_2026_10_09",
    "cfm_bond_head_30ep_onecycle_2026_10_09",
)


def _changelog_rate_claims(artifact: dict) -> list[tuple[str, str, float]]:
    """Yield (changelog_key, run_name, quoted_rate) for every explicit claim."""
    names = {(r.get("run") or r.get("run_name")) for r in artifact.get("runs", [])}
    names.discard(None)
    claims = []
    for key, entries in artifact.get("changelog", {}).items():
        for entry in entries:
            text = entry if isinstance(entry, str) else json.dumps(entry)
            for match in re.finditer(
                r"([A-Za-z0-9_.\-]*cfm[A-Za-z0-9_.\-]*)"
                r"\s*\((?:best[- ]epoch rate\s*)?(\d+\.\d+)\s*\)",
                text,
            ):
                run, rate = match.group(1), match.group(2)
                if run in names:
                    claims.append((key, run, float(rate)))
    return claims


def test_census_changelog_prose_agrees_with_the_runs_array():
    """A changelog sentence may not attribute a run's rate to a different run."""
    artifact = load_artifact(ARTIFACT)
    runs = {(r.get("run") or r.get("run_name")): r for r in artifact.get("runs", [])}
    bad = []
    for key, run, rate in _changelog_rate_claims(artifact):
        truth = runs[run].get("gen_single_molecule_rate_best")
        if truth is None:
            continue
        if abs(rate - float(truth)) > 1e-12:
            bad.append((key, run, rate, truth))
    assert not bad, (
        "census changelog prose disagrees with runs[].gen_single_molecule_rate_best:\n"
        + "\n".join(
            f"  changelog[{k}]: {r} quoted {q}, artifact says {t}"
            for k, r, q, t in bad
        )
    )


def test_the_two_wf_a150_runs_carry_the_rates_the_changelog_names():
    """The Wf-A150 swap, pinned directly against the artifact's own array."""
    artifact = load_artifact(ARTIFACT)
    runs = {r.get("run") or r.get("run_name"): r for r in artifact["runs"]}
    for name, rate in WF_A150_RUNS.items():
        assert runs[name]["gen_single_molecule_rate_best"] == pytest.approx(
            rate, abs=1e-12
        )
    entry = "\n".join(artifact["changelog"]["wf_a150"])
    for name, rate in WF_A150_RUNS.items():
        assert f"{name} (best-epoch rate {rate})" in entry


# --------------------------------------------------------------------------
# B — every census statistic a document quotes must be the CURRENT value
# --------------------------------------------------------------------------
#: (population, epoch) -> the artifact path holding that block's statistics.
#: ``epoch`` may be ``None`` = "unknown" -> the union over epochs.
BLOCK_PATHS = {
    ("clean", "last"): "fragmentation_summary.end_to_end_single_molecule_rate",
    ("all", "last"): "fragmentation_summary.end_to_end_single_molecule_rate_all_runs",
    ("clean", "best"): "fragmentation_summary.best_epoch_single_molecule_rate",
    ("all", "best"): "fragmentation_summary.best_epoch_single_molecule_rate_all_runs",
}

#: The statistic keys a document quotes and the WORD that introduces them.
KEY_WORDS = {
    "median": re.compile(r"\bmedians?\b", re.I),
    "mean": re.compile(r"\bmeans?\b", re.I),
    "max": re.compile(r"\bmax(?:imum)?\b", re.I),
    "min": re.compile(r"\bmin(?:imum)?\b", re.I),
}

#: Keys whose value is written as ``key = N``.
KEY_EQUALS = {
    "n": re.compile(r"\bn\s*=\s*(\d{1,3})(?!\w)"),
    "n_below_0.50": re.compile(r"n_below_0\.50\s*=\s*(\d{1,3})(?!\w)"),
    "n_below_0.10": re.compile(r"n_below_0\.10\s*=\s*(\d{1,3})(?!\w)"),
    "n_at_or_above_0.50": re.compile(r"n_at_or_above_0\.50\s*=\s*(\d{1,3})(?!\w)"),
    "n_at_or_above_0.10": re.compile(r"n_at_or_above_0\.10\s*=\s*(\d{1,3})(?!\w)"),
}

#: The prose forms of the counters.
PROSE_COUNT_PATTERNS = {
    "n_below_0.50": re.compile(
        r"(\d{1,3})\s+of\s+\d{1,3}\s+(?:runs\s+)?below\s+0\.50", re.I
    ),
    "n_below_0.10": re.compile(
        r"(\d{1,3})\s+of\s+\d{1,3}\s+(?:runs\s+)?below\s+0\.10", re.I
    ),
    "n_at_or_above_0.50": re.compile(
        r"(\d{1,3})\s+of\s+\d{1,3}\s+(?:runs\s+)?(?:at or above|≥|>=)\s*0\.50", re.I
    ),
    "n_at_or_above_0.10": re.compile(
        r"(\d{1,3})\s+of\s+\d{1,3}\s+(?:runs\s+)?(?:at or above|≥|>=)\s*0\.10", re.I
    ),
    # The BARE forms read a count that is written without an "N of M"
    # denominator ("16 are below 0.10").  The look-behind stops them from
    # re-reading the DENOMINATOR of an "N of M" construction ("23 of 27 below
    # 0.50" already yields 23 through the "of" form above; reading 27 there
    # compared the population SIZE against the BELOW-0.50 count).
    "n_below_0.10_bare": re.compile(
        r"(?<!of )(?<![\w.])(\d{1,3})\s+(?:are\s+|is\s+)?below\s+0\.10", re.I
    ),
    "n_below_0.50_bare": re.compile(
        r"(?<!of )(?<![\w.])(\d{1,3})\s+(?:are\s+|is\s+)?below\s+0\.50", re.I
    ),
    "n_at_or_above_0.50_bare": re.compile(
        r"(?<!of )(?<![\w.])(\d{1,3})\s+(?:at or above|≥|>=)\s*0\.50", re.I
    ),
    "n_at_or_above_0.10_bare": re.compile(
        r"(?<!of )(?<![\w.])(\d{1,3})\s+(?:at or above|≥|>=)\s*0\.10", re.I
    ),
}

#: A jsonpath anchor names population AND epoch at once.  Longest-first, so
#: ``_all_runs`` wins over the bare path.
ANCHOR_RE = re.compile(
    r"best_epoch_single_molecule_rate_all_runs"
    r"|end_to_end_single_molecule_rate_all_runs"
    r"|best_epoch_single_molecule_rate"
    r"|end_to_end_single_molecule_rate"
)
ANCHOR_STATE = {
    "end_to_end_single_molecule_rate_all_runs": ("all", "last"),
    "best_epoch_single_molecule_rate_all_runs": ("all", "best"),
    "end_to_end_single_molecule_rate": ("clean", "last"),
    "best_epoch_single_molecule_rate": ("clean", "best"),
}
#: The ``…_all_runs`` shorthand that follows an anchor.
ALL_RUNS_RE = re.compile(r"_all_runs\b")

#: A population label.
#:
#: "CLEAN" is matched CASE-SENSITIVELY on purpose.  The lowercase form is far
#: too common in ordinary English — "chemistry-clean", "a clean run of the
#: decoder" — to resolve a population, and matching it case-insensitively
#: produced false positives on unrelated documents ("n=200 SMILES",
#: "n=1").  The lowercase form is honoured only when the NEXT word makes it a
#: population ("clean runs", "clean population", "clean cohort").
#:
#: "ALL" is matched only when the next word makes it a population ("ALL
#: runs", "ALL 32 runs"), or when the whole population + epoch + statistic
#: triple is present ("ALL best-epoch median") -- see the POP_RE comment.
POP_RE = re.compile(
    r"\bCLEAN\b"
    r"|(?i:\bclean\s+(?:runs?|populations?|cohorts?)\b)"
    r"|(?i:\ball[- ]runs?\b)"
    r"|(?i:\ball\s+\**\d+\**\s+runs?\b)"
    r"|(?i:\bclean_only\b)"
    # The ALL + epoch + statistic TRIPLE (Wf-A154, 2026-11-01).  The bare
    # "ALL" is ignored on purpose, but "ALL best-epoch median" is a
    # population + an epoch + a statistic, and no ordinary English sentence
    # contains that triple -- so it is safe to resolve, and NOT resolving it
    # is a hole: the label resolved against the CLEAN population, so a
    # STALE all-runs value under it produced 0 hits (false negative) and a
    # CORRECT one produced a false positive.  MEASURED 2026-11-01 on the
    # production path: stale "ALL best-epoch median 0.35546875" -> 0 hits.
    r"|(?i:\ball\s+(?:best|last)[- _]epoch\s+(?:median|mean|max|min)\b)"
)
#: A population word as it appears INSIDE a census qualifier.  "ALL" alone
#: is far too common in running prose to be a population label, but inside a
#: qualifier parenthetical ("(ALL, n=9)") it is exactly that.
QUALIFIER_POP_RE = re.compile(r"\b(all|clean)\b", re.I)
#: A parenthetical that carries census scope: it names an epoch, or it binds
#: a ``key = N`` counter.  Only such a parenthetical may override the running
#: state — a free-form parenthetical ("n=200 SMILES Ridge overfits") is not a
#: census qualifier.
QUALIFIER_SHAPE_RE = re.compile(
    r"\b(?:best|last)[- _]epoch\b|\b[\w.]+\s*=\s*\d", re.I
)
#: An epoch label.
EPOCH_RE = re.compile(r"\b(best|last)[- _]epoch\b", re.I)
#: A cohort label.  ``c4/c5`` is a STRICT ``(n_train, n_val)`` cohort — the
#: ``by_cohort`` axis ("26715/192"); ``c6`` is the loose ``n_train`` axis named
#: in a contrast ("26715 ahead:").
COHORT_RE = re.compile(
    r"by_n_train\[\s*[\"']?(?P<c1>\d{2,6})[\"']?\s*\]"
    r"|n_train\s*[=:]?\s*(?P<c2>\d{2,6})"
    r"|(?<![\d.])(?P<c3>\d{2,6})[- ]cohort\b"
    r"|(?<![\d.])(?P<c4>\d{2,6})\s*/\s*(?P<c5>\d{2,6})(?![\d.])"
    r"|(?<![\d.])(?P<c6>\d{2,6})\s+(?:ahead|behind)\b"
)
#: A contrast marker.  The value AFTER a "vs" belongs to the CONTRASTED
#: cohort, which is not the cohort named before it.
VS_RE = re.compile(r"\bvs\.?\b", re.I)

#: A historical marker.  Looked for NEAR the value, inside the claim's own
#: sentence — this is the value-scoped replacement for the Wf-A134 guard's
#: paragraph-scoped exemption.
SENT_HIST_RE = re.compile(
    r"(?:\bSTALE\b|\bstale\b|\bSUPERSEDED\b|\bsuperseded\b|\bHISTORICAL\b"
    r"|\bhistorical\b|\bPREVIOUS\b|\bprevious\b|\bpreviously\b|\bformerly\b"
    r"|no longer|\boutdated\b|\bstruck\b|preserved verbatim|originally read"
    r"|originally said)",
    re.I,
)
#: Past tense immediately before the quote: ``was 0.057291666666666664``.
PRE_VALUE_HIST_RE = re.compile(
    r"(?:\bwas\b|\bwere\b|\bformerly\b|\bpreviously\b|\bsuperseded\b|\bstale\b)"
    r"\s*$"
)
#: A PAST-MOVE verb.  "The cells changed to their current values (X)" reports
#: what a table said AT THAT TIME, so X is history — the sentence is a
#: correction narrative, not a live claim.  The same holds for "was corrected
#: to Y" and "the column did not move".
CHANGE_RE = re.compile(
    r"\bchanged\b|\bcorrected\b|\bmoved\b|\bre-?published\b|\bre-?written\b", re.I
)
#: An ALL-CAPS CURRENT defeats the past-move exemption.  A banner that says
#: "the CURRENT values are: X" publishes live values even when the same
#: sentence also says the census "was re-published".
LIVE_CURRENT_RE = re.compile(r"\bCURRENT\b")
#: A transition arrow — a move statement is history on both sides.
ARROW_RE = re.compile(r"(?:→|->)")
#: A "from X to Y" transition — the X side is history.
FROM_RE = re.compile(r"\bfrom\s*$")

#: A BROADER census-context detector than the Wf-A134 guard's — mechanism 3
#: is that a census claim avoiding the old keyword list was never examined.
CENSUS_CONTEXT_RE = re.compile(
    r"cfm_run_census|fragmentation_summary|end_to_end_single_molecule_rate"
    r"|single_molecule_rate|budget_matched|flipped_budgets|\bcensus\b"
    r"|last[- ]epoch|best[- ]epoch|all-runs|\bALL runs\b|\bCLEAN\b"
    r"|n_below_0\.\d+|n_at_or_above_0\.\d+|by_n_train|by_cohort|clean_only"
    r"|fragmentation|one connected|single molecule|connected rate"
    r"|single-fragment|single_fragment",
    re.I,
)

#: A sentence boundary: a period/!/? followed by whitespace.  Decimals are
#: NOT boundaries ("0.03125" has no whitespace after the dot).
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
#: A LIST-ITEM boundary.  A new bullet or numbered item is a new topic, so
#: the running population / epoch / cohort must not cross it.
LIST_ITEM_RE = re.compile(r"\n[ \t]*(?:[-*+][ \t]+|\d+[.)][ \t]+)")
#: A LIST MARKER — the ``*`` that OPENS a bullet.  It is structural, NOT
#: emphasis: blanking it removes the bullet marker, and ``LIST_ITEM_RE``
#: can then no longer split the paragraph at the bullet boundary, so the
#: running population / epoch / cohort leaks out of one bullet into the
#: next.  MEASURED 2026-11-01 (Wf-A153): blanking every ``*`` merged the
#: audit doc's "primary bar" / "secondary bar" / "context" bullets into ONE
#: sentence, so the cohort ``26715/192`` named in the second bullet was
#: resolved against the PROJECT-WIDE median quoted in the third — a false
#: positive on a claim that is correct at its stated scope.
#:
#: ``**bold**`` at the start of a line is NOT a marker: the first ``*`` is
#: followed by another ``*``, not by whitespace.
LIST_MARKER_RE = re.compile(r"(?m)^[ \t>]*(?P<mark>[-*+])[ \t]")
#: How far before a value a historical marker may sit and still exempt it.
MARKER_WINDOW_BEFORE = 120
MARKER_WINDOW_AFTER = 60
#: How close an arrow / "from" must be to a value for it to read as a move.
ARROW_WINDOW = 12
#: How far the parenthetical qualifier may sit after a claim.  Wide enough
#: for a slash PAIR to inherit one trailing qualifier ("0.125 / 0.0364…  (last
#: epoch, CLEAN)") — the pair's first number is ~28 chars from the paren.
PAREN_WINDOW = 45
#: How long a keyword window may run before it is cut.
KEYWORD_WINDOW = 45
#: How close a rendered ``0.0`` must sit to a run's NAME before it counts as
#: that run's reading (rather than as a quoted placeholder elsewhere in the
#: same sentence).
NULL_WINDOW = 60
#: A unit word between a statistic keyword and its number means the keyword
#: is NOT introducing a statistic: "~60 min CPU + ~31.9 hr GPU" is a
#: DURATION, not a minimum.
UNIT_WORD_RE = re.compile(
    r"\b(?:cpu|gpu|hrs?|hours?|mins?|minutes?|secs?|seconds?|ms|gb|mb|kb|tb"
    r"|wall|clock|install|loc)\b",
    re.I,
)
#: A THRESHOLD marker between a statistic keyword and its number means the
#: number is the CUTOFF, not the statistic: "… / mean 0.6376953125 / 5 at or
#: above 0.50" reads 0.50 as a mean, which is a different quantity.
THRESHOLD_RE = re.compile(
    r"\bat\s+or\s+above\b|\bbelow\b|\babove\b|\bunder\b|≥|>=|<=|≤", re.I
)

NUMBER_RE = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")
INT_RE = re.compile(r"(?<![\w.])(\d{1,3})(?![\w.])")
STRUCK_RE = re.compile(r"~~[^~]*~~")
#: A slash pair of two numbers: ``0.125 / 0.03645833333333333``.
PAIR_RE = re.compile(
    r"(?<![\w.])(\d+\.\d+)\s*/\s*(?<![\w.])(\d+\.\d+)(?![\w.])"
)
#: A parenthetical immediately after a claim.
PAREN_RE = re.compile(r"\(([^)]{0,60})\)")
#: A parenthetical of ANY length, for the CONTAINING / FOLLOWING qualifier
#: search.  ``PAREN_RE``'s 60-char ceiling is too narrow for a census
#: qualifier that carries several counters ("(CLEAN, n = 29, min 0.0, median
#: …)"), and the innermost-parenthetical rule needs to see it.
PAREN_SPAN_RE = re.compile(r"\(([^()]{0,200})\)")
#: A CLAUSE boundary — a bracketed group, a semicolon, an em dash or a full
#: stop.  Two things in different clauses are not in a reading relation.
CLAUSE_SPLIT_RE = re.compile(r"[;()\[\]]|—|—|\.\s")


def _clause_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every clause in ``text``."""
    spans = []
    start = 0
    for match in CLAUSE_SPLIT_RE.finditer(text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _clause_of(spans, position: int) -> int:
    """The index of the clause that contains ``position`` (-1 if none)."""
    for index, (start, end) in enumerate(spans):
        if start <= position < end:
            return index
    return -1


def _list_marker_indices(text: str) -> set[int]:
    """Offsets of the ``*`` characters that OPEN a list item.

    Those are structural, not emphasis.  Blanking them removes the bullet
    marker, and ``LIST_ITEM_RE`` then cannot split the paragraph at the
    bullet boundary — so the running cohort leaks from one bullet into the
    next.  Wf-A153, MEASURED 2026-11-01: see ``LIST_MARKER_RE``.
    """
    return {
        match.start("mark")
        for match in LIST_MARKER_RE.finditer(text)
        if match.group("mark") == "*"
    }


def _normalize(text: str) -> str:
    """Blank ``~~struck~~`` spans, ``*`` emphasis and backticks.

    Offsets are preserved, so a match found in the normalised text maps back
    to the original character-for-character.

    A list marker (``*`` at the start of a bullet) is NOT blanked — see
    ``_list_marker_indices``.
    """
    out = list(text)
    for match in STRUCK_RE.finditer(text):
        for index in range(match.start(), match.end()):
            out[index] = " "
    markers = _list_marker_indices(text)
    for index, char in enumerate(text):
        if char == "*" and index in markers:
            continue
        if char in "*`":
            out[index] = " "
    return "".join(out)


def _flatten_paragraph(block: list[tuple[int, str]]) -> tuple[str, list[int]]:
    """Flatten a paragraph, blanking ``~~struck~~`` spans ACROSS line breaks.

    The Wf-A134 guard blanks struck spans PER LINE, so a span that opens on
    one line and closes on the next ("~~25 are below\\n> 0.50~~") survives as
    live text and the guard reads the STRUCK number as a live claim.  Here the
    paragraph is joined FIRST and the spans are blanked on the joined text.

    Lines are joined with a NEWLINE and the LIST MARKER is KEPT, so
    ``_sentences`` can end a sentence at a list-item boundary.  A list item
    is a new topic: without that boundary the running population / epoch /
    cohort leaks out of one bullet into the next.  MEASURED consequence of
    not having it: a Ridge sample size ("n=200 SMILES") was checked against
    the census run count, and a project-wide median was checked against a
    cohort median, both because an unrelated bullet had opened with the word
    "CLEAN".
    """
    chars: list[str] = []
    owners: list[int] = []
    for position, (lineno, raw) in enumerate(block):
        if position:
            chars.append("\n")
            owners.append(lineno)
        text = re.sub(r"^\s*(?:>\s*)?", "", raw)
        for char in text:
            chars.append(char)
            owners.append(lineno)
    joined = "".join(chars)
    struck = list(joined)
    for match in STRUCK_RE.finditer(joined):
        for index in range(match.start(), match.end()):
            struck[index] = " "
    # Wf-A153 — blank ``*`` and backticks exactly as ``_normalize`` does.
    # Without this the PRODUCTION path (``_scan_repo``) and the MUTATION
    # path (``_scan_text``) disagree: ``_normalize`` blanks the emphasis
    # delimiters, so a bolded claim like ``n=**29**`` matches the
    # ``n=<int>`` pattern, while this function used to leave the ``*``
    # in place — the ``KEY_EQUALS`` pattern needs a digit right after the
    # ``=``, and ``**`` sits in the way, so the SAME text was invisible
    # to the scan that actually runs over the repo.  MEASURED 2026-11-01:
    # ``n=**29**`` -> 1 hit on the mutation path, 0 on the production path.
    #
    # A LIST MARKER is the exception: blanking the bullet's own ``*``
    # removes the ``LIST_ITEM_RE`` boundary, which merges the bullets into
    # one sentence and leaks the previous bullet's cohort into the next
    # bullet's claims.  Wf-A153, MEASURED 2026-11-01: the audit doc's
    # project-wide median was resolved against the cohort named in the
    # PREVIOUS bullet — a false positive on a correct claim.
    markers = _list_marker_indices(joined)
    for index, char in enumerate(joined):
        if char == "*" and index in markers:
            continue
        if char in "*`":
            struck[index] = " "
    return "".join(struck), owners


def _sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of every sentence in ``text``.

    A sentence ends at a ``.``/``!``/``?`` followed by whitespace, and at a
    LIST-ITEM boundary — a new bullet or numbered item is a new topic.
    """
    cuts = [match.start() for match in SENT_SPLIT_RE.finditer(text)]
    cuts.extend(match.start() + 1 for match in LIST_ITEM_RE.finditer(text))
    spans = []
    start = 0
    for cut in sorted(set(cuts)):
        if cut <= start or cut >= len(text):
            continue
        spans.append((start, cut))
        while cut < len(text) and text[cut].isspace():
            cut += 1
        start = cut
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _dig(node, path: str):
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _headline_legal(artifact: dict, population, epoch, key) -> set[float]:
    """Every CURRENT headline value the census publishes for one statistic."""
    values: set[float] = set()
    for (pop, ep), path in BLOCK_PATHS.items():
        if pop != population:
            continue
        if epoch is not None and ep != epoch:
            continue
        holder = _dig(artifact, path)
        if isinstance(holder, dict):
            value = holder.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.add(float(value))
    return values


def _cohort_legal(artifact: dict, cohort, population, epoch, key) -> set[float]:
    """Every CURRENT value the census publishes for one cohort statistic.

    ``cohort`` names either the loose ``n_train`` axis (``by_n_train``) or the
    strict ``(n_train, n_val)`` axis (``by_cohort``) — the census publishes
    both, and they are NOT interchangeable: ``by_n_train["26715"]`` holds 14
    runs across two validation protocols, ``by_cohort["26715/192"]`` holds 9.
    """
    values: set[float] = set()
    axis = "by_cohort" if "/" in cohort else "by_n_train"
    block = _dig(artifact, f"{axis}.{cohort}")
    if not isinstance(block, dict):
        return values

    def _collect(holder):
        if not isinstance(holder, dict):
            return
        value = holder.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.add(float(value))

    epochs = [epoch] if epoch is not None else ["last", "best"]
    for ep in epochs:
        holder = block.get(f"{ep}_epoch")
        if population in (None, "all"):
            _collect(holder)
        if population in (None, "clean"):
            clean = block.get("clean_only")
            if isinstance(clean, dict):
                _collect(clean.get(f"{ep}_epoch"))
    return values


def _events(text: str) -> list[tuple[int, str, object]]:
    """(position, kind, payload) for every state-changing token.

    ``kind`` is one of ``anchor`` / ``pop`` / ``epoch`` / ``cohort``.  The
    anchor spans are excluded from the ``_all_runs`` shorthand scan so an
    anchor is not immediately overwritten by its own suffix.
    """
    events: list[tuple[int, str, object]] = []
    anchor_spans = []
    for match in ANCHOR_RE.finditer(text):
        anchor_spans.append((match.start(), match.end()))
        events.append((match.start(), "anchor", ANCHOR_STATE[match.group(0)]))
    for match in ALL_RUNS_RE.finditer(text):
        if any(start <= match.start() < end for start, end in anchor_spans):
            continue
        events.append((match.start(), "pop", "all"))
    for match in POP_RE.finditer(text):
        token = match.group(0).lower()
        events.append((match.start(), "pop", "all" if "all" in token else "clean"))
    for match in EPOCH_RE.finditer(text):
        epoch = "best" if match.group(1).lower() == "best" else "last"
        events.append((match.start(), "epoch", epoch))
    for match in COHORT_RE.finditer(text):
        cohort = (
            match.group("c1")
            or match.group("c2")
            or match.group("c3")
            or match.group("c6")
        )
        if cohort is None and match.group("c4") and match.group("c5"):
            cohort = f"{match.group('c4')}/{match.group('c5')}"
        if cohort:
            events.append((match.start(), "cohort", cohort))
    events.sort(key=lambda item: item[0])
    return events


def _keyword_spans(text: str) -> list[tuple[str, int, int]]:
    """(key, value_start, value_end) for every keyword-introduced number."""
    boundaries = []
    for _key, pattern in KEY_WORDS.items():
        for match in pattern.finditer(text):
            boundaries.append((match.start(), match.end(), _key))
    for _key, pattern in KEY_EQUALS.items():
        for match in pattern.finditer(text):
            boundaries.append((match.start(), match.start(), None))
    boundaries.sort()
    spans = []
    for index, (start, end, key) in enumerate(boundaries):
        if key is None:
            continue
        stop = min(start + KEYWORD_WINDOW, len(text))
        for other_start, _other_end, _other_key in boundaries[index + 1 :]:
            stop = min(stop, other_start)
        for closer in (",", ";", ")", "]"):
            position = text.find(closer, end)
            if position != -1:
                stop = min(stop, position)
        window = text[end:stop]
        for number in NUMBER_RE.finditer(window):
            # A float the window cut in half is a FALSE reading ("0.06" for
            # "0.0625"), which made a live claim look stale.  Take the number
            # from the ORIGINAL text so it is never truncated.
            full = NUMBER_RE.match(text, end + number.start())
            if full is None:
                continue
            # A UNIT word between the keyword and the number means the keyword
            # is not introducing a statistic: "~60 min CPU + ~31.9 hr GPU" is a
            # DURATION, and the "min" in "60 min" is not a minimum.
            if UNIT_WORD_RE.search(text[end : full.start()]):
                continue
            # A THRESHOLD marker likewise: the number is the cutoff the
            # statistic is compared against, not the statistic.
            if THRESHOLD_RE.search(text[end : full.start()]):
                continue
            spans.append((key, full.start(), full.end()))
    return spans


def _claims_in(text: str) -> list[tuple[str, float, int, int]]:
    """(key, value, start, end) for every statistic claim in ``text``."""
    claims: list[tuple[str, float, int, int]] = []
    for key, start, end in _keyword_spans(text):
        claims.append((key, float(text[start:end]), start, end))
    for key, pattern in KEY_EQUALS.items():
        for match in pattern.finditer(text):
            claims.append((key, float(match.group(1)), match.start(1), match.end(1)))
    for key, pattern in PROSE_COUNT_PATTERNS.items():
        base_key = key.replace("_bare", "")
        for match in pattern.finditer(text):
            claims.append((base_key, float(match.group(1)), match.start(1), match.end(1)))
    return claims


def _is_historical(text: str, start: int, end: int, sent_start: int, sent_end: int) -> bool:
    """Is THIS value presented as history rather than as the current claim?"""
    before = text[max(sent_start, start - MARKER_WINDOW_BEFORE) : start]
    after = text[end : min(sent_end, end + MARKER_WINDOW_AFTER)]
    if SENT_HIST_RE.search(before) or SENT_HIST_RE.search(after):
        return True
    # A PAST-MOVE verb puts the value in history ("the cells changed to their
    # current values (X)") — unless the same window says CURRENT in caps,
    # which is how a banner publishes LIVE values.
    if CHANGE_RE.search(before) and not LIVE_CURRENT_RE.search(before):
        return True
    if PRE_VALUE_HIST_RE.search(text[max(sent_start, start - 40) : start]):
        return True
    if ARROW_RE.search(text[max(sent_start, start - ARROW_WINDOW) : start]):
        return True
    if ARROW_RE.search(text[end : min(sent_end, end + ARROW_WINDOW)]):
        return True
    if FROM_RE.search(text[max(sent_start, start - 12) : start]):
        return True
    # A value inside a double-quoted span is a QUOTATION ("8 of 29 at or above
    # 0.50"), not a live claim.
    if text[sent_start:start].count('"') % 2 == 1 and '"' in after:
        return True
    return False


def _state_at(events, position: int) -> tuple:
    """The (population, epoch, cohort) in force at ``position``."""
    population = epoch = cohort = None
    for pos, kind, payload in events:
        if pos > position:
            break
        if kind == "anchor":
            population, epoch = payload
            cohort = None
        elif kind == "pop":
            population = payload
        elif kind == "epoch":
            epoch = payload
        elif kind == "cohort":
            cohort = payload
    return population, epoch, cohort


def _cohort_events(events, sent_start: int, sent_end: int) -> list[tuple[int, str]]:
    """(position, cohort) for every DISTINCT cohort a sentence names.

    Order of first appearance.  The POSITION matters: a value after a "vs"
    belongs to the contrasted cohort, which is the one named AFTER the "vs".
    """
    seen: list[tuple[int, str]] = []
    for pos, kind, payload in events:
        if kind != "cohort" or not (sent_start <= pos < sent_end):
            continue
        if payload not in [cohort for _pos, cohort in seen]:
            seen.append((pos, payload))
    return seen


# --------------------------------------------------------------------------
# census qualifier parentheticals
# --------------------------------------------------------------------------
def _paren_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, group) for every parenthetical in ``text``."""
    return [
        (match.start(), match.end(), match.group(1))
        for match in PAREN_SPAN_RE.finditer(text)
    ]


def _containing_paren(parens, position: int):
    """The INNERMOST parenthetical that CONTAINS ``position``."""
    best = None
    for start, end, group in parens:
        if start <= position < end and (best is None or start > best[0]):
            best = (start, end, group)
    return best


def _visible_distance(text: str, start: int, end: int) -> int:
    """How much READABLE text sits between two offsets.

    A struck span is blanked to spaces, so a raw offset distance measures the
    blanked stretch rather than the prose.  "0.125 / ~~0.0260…~~ 0.0364…  (last
    epoch, CLEAN)" is 50 raw characters from the qualifier but 23 VISIBLE
    ones — the raw distance pushed a legitimate qualifier out of the window.
    """
    return sum(1 for char in text[start:end] if not char.isspace())


def _following_paren(text: str, end: int):
    """The nearest parenthetical that STARTS after ``end``."""
    for match in PAREN_SPAN_RE.finditer(text, end):
        if _visible_distance(text, end, match.start()) > PAREN_WINDOW:
            break
        return (match.start(), match.end(), match.group(1))
    return None


def _qualifier_scope(group: str) -> tuple:
    """(population, epoch) named by a census qualifier parenthetical.

    ``None`` for a field the parenthetical does not name — and for a field it
    names AMBIGUOUSLY.  "(**CLEAN**, n = 29, all-runs …)" names two
    populations, so it names none and the running state stands.

    A parenthetical that carries no census scope at all ("n=200 SMILES Ridge
    overfits") is not a qualifier and contributes nothing.
    """
    if not QUALIFIER_SHAPE_RE.search(group):
        return None, None
    pops = set()
    for match in POP_RE.finditer(group):
        pops.add("all" if "all" in match.group(0).lower() else "clean")
    for match in QUALIFIER_POP_RE.finditer(group):
        pops.add(match.group(1).lower())
    epochs = {
        "best" if match.group(1).lower() == "best" else "last"
        for match in EPOCH_RE.finditer(group)
    }
    population = pops.pop() if len(pops) == 1 else None
    epoch = epochs.pop() if len(epochs) == 1 else None
    return population, epoch


def _resolve_claim(
    artifact, text, sent_start, sent_end, events, claim, cohort_events, pairs, parens
):
    """Resolve one claim to (population, epoch, cohort) and compare."""
    key, value, start, end = claim
    cohorts = [cohort for _pos, cohort in cohort_events]

    population = epoch = cohort = None
    run_pop, run_epoch, run_cohort = _state_at(events, start)

    # 1. a census qualifier parenthetical.  The innermost one CONTAINING the
    #    claim wins over the nearest one FOLLOWING it, and either wins over
    #    the running state:
    #      * the containing parenthetical is the claim's own scope — "(ALL,
    #        n = 9)" makes the claim an ALL-runs count;
    #      * the following parenthetical is the qualifier of the clause the
    #        claim closes — "0.125 / 0.0364…  (last epoch, CLEAN)".
    #    A qualifier that names nothing, or names two things, contributes
    #    nothing.
    scopes = []
    containing = _containing_paren(parens, start)
    if containing is not None:
        scopes.append(containing[2])
    following = _following_paren(text, end)
    if following is not None:
        scopes.append(following[2])
    for group in scopes:
        scope_pop, scope_epoch = _qualifier_scope(group)
        if population is None and scope_pop is not None:
            population = scope_pop
        if epoch is None and scope_epoch is not None:
            epoch = scope_epoch

    if population is None:
        population = run_pop
    if epoch is None:
        epoch = run_epoch
    if population is None:
        return None  # the population is undetermined -> skip, never guess

    # 2. cohort — three rules, in this order:
    #    (a) a slash pair names its cohorts POSITIONALLY: "A / B" is cohort
    #        A's value followed by cohort B's value;
    #    (b) after a "vs" the value belongs to the CONTRASTED cohort, which
    #        must be named in the sentence or the claim is skipped — never
    #        silently attributed to the cohort named before the "vs";
    #    (c) otherwise the cohort in force (running state), then the single
    #        cohort the sentence names.  Several named cohorts and no
    #        attachment -> skip; a named cohort NEVER falls back to the
    #        project-wide headline.
    for first_start, first_end, second_start, second_end in pairs:
        if first_start <= start < first_end:
            if len(cohorts) == 2:
                cohort = cohorts[0]
            break
        if second_start <= start < second_end:
            if len(cohorts) == 2:
                cohort = cohorts[1]
            break

    if cohort is None and cohorts:
        vs_positions = [
            match.start() for match in VS_RE.finditer(text) if match.start() < start
        ]
        if vs_positions:
            last_vs = vs_positions[-1]
            after = [
                name for pos, name in cohort_events if pos > last_vs
            ]
            if len(after) == 1:
                cohort = after[0]
            else:
                return None  # the contrasted cohort is not named -> skip
        else:
            cohort = run_cohort
            if cohort is None:
                if len(cohorts) == 1:
                    cohort = cohorts[0]
                else:
                    return None  # several cohorts, this claim is attached to none
    else:
        cohort = run_cohort if cohort is None else cohort

    if cohort:
        legal = _cohort_legal(artifact, cohort, population, epoch, key)
    else:
        legal = _headline_legal(artifact, population, epoch, key)
    if not legal:
        return None
    if any(abs(value - candidate) <= 1e-12 for candidate in legal):
        return None
    if _is_historical(text, start, end, sent_start, sent_end):
        return None
    return {
        "stat": f"{population}/{epoch}/{cohort or 'headline'}/{key}",
        "quoted": value,
        "current": sorted(legal),
        "text": text[max(sent_start, start - 80) : end + 30].strip(),
        "offset": start,
    }


def _scan_sentence(artifact, text, sent_start, sent_end) -> list[dict]:
    """Every stale census statistic in one sentence."""
    body = text[sent_start:sent_end]
    events = _events(body)
    cohort_events = _cohort_events(events, 0, len(body))
    parens = _paren_spans(body)
    pairs = []
    for match in PAIR_RE.finditer(body):
        pairs.append(
            (match.start(1), match.end(1), match.start(2), match.end(2))
        )
    findings = []
    for claim in _claims_in(body):
        hit = _resolve_claim(
            artifact, body, 0, len(body), events, claim, cohort_events, pairs, parens
        )
        if hit is not None:
            findings.append(hit)
    return findings


# --------------------------------------------------------------------------
# table path
# --------------------------------------------------------------------------
#: A table column header that names a statistic.
HEADER_KEYS = {
    "n": "n",
    "runs": "n",
    "count": "n",
    "median": "median",
    "mean": "mean",
    "max": "max",
    "min": "min",
}
#: A table column header that names the population.
POP_HEADER_RE = re.compile(r"population|\bpop\b", re.I)
#: A table column header that names the reading / epoch.
EPOCH_HEADER_RE = re.compile(r"reading|\bview\b|epoch", re.I)
#: A table column header that names the tail counter.
TAIL_HEADER_RE = re.compile(r"tail|counter", re.I)


def _normalise_header(cell: str) -> str:
    return re.sub(r"[*`_\s]+", "", cell).lower()


def _scan_table(rows, artifact: dict) -> list[dict]:
    """Every stale census statistic in one table."""
    header = [_normalise_header(cell) for cell in rows[0][1]]
    stat_cols: dict[int, str] = {}
    pop_col = epoch_col = tail_col = None
    for index, name in enumerate(header):
        if name in HEADER_KEYS and index not in stat_cols:
            stat_cols[index] = HEADER_KEYS[name]
        elif POP_HEADER_RE.search(name) and pop_col is None:
            pop_col = index
        elif EPOCH_HEADER_RE.search(name) and epoch_col is None:
            epoch_col = index
        elif TAIL_HEADER_RE.search(name) and tail_col is None:
            tail_col = index
    if not stat_cols and tail_col is None:
        return []  # a narrative table — historical by construction

    findings = []
    for lineno, cells in rows[1:]:
        if cells and all(set(cell.strip()) <= set("-: ") for cell in cells):
            continue
        population = epoch = None
        if pop_col is not None and pop_col < len(cells):
            population = _state_at(_events(_normalize(cells[pop_col])), 0)[0]
        if epoch_col is not None and epoch_col < len(cells):
            epoch = _state_at(_events(_normalize(cells[epoch_col])), 0)[1]
        if population is None or epoch is None:
            continue
        for index, cell in enumerate(cells):
            key = stat_cols.get(index)
            if key is None and tail_col is not None and index == tail_col:
                key = "tail"
            if key is None:
                continue
            text = _normalize(cell)
            if key == "n":
                claims = [
                    ("n", float(number.group(1)), number.start(), number.end())
                    for number in INT_RE.finditer(text)
                ]
            elif key == "tail":
                claims = _claims_in(text)
            else:
                claims = [
                    (key, float(number.group(1)), number.start(), number.end())
                    for number in NUMBER_RE.finditer(text)
                ]
            for claim_key, value, start, end in claims:
                legal = _headline_legal(artifact, population, epoch, claim_key)
                if not legal:
                    continue
                if any(abs(value - candidate) <= 1e-12 for candidate in legal):
                    continue
                if _is_historical(text, start, end, 0, len(text)):
                    continue
                findings.append(
                    {
                        "stat": f"{population}/{epoch}/headline/{claim_key}",
                        "quoted": value,
                        "current": sorted(legal),
                        "text": text[:120].strip(),
                        "line": lineno,
                    }
                )
    return findings


# --------------------------------------------------------------------------
# prose path
# --------------------------------------------------------------------------
def _scan_prose(text: str, owners, artifact: dict) -> list[dict]:
    """Every stale census statistic in one paragraph."""
    findings = []
    for sent_start, sent_end in _sentences(text):
        for hit in _scan_sentence(artifact, text, sent_start, sent_end):
            # Wf-A153 — ``_scan_sentence`` works on the SENTENCE BODY, so the
            # offset it returns is BODY-relative, while ``owners`` is
            # PARAGRAPH-relative.  Reporting the body offset directly
            # pointed at the START of the sentence instead of the line the
            # claim is on.  MEASURED 2026-11-01: the audit doc's claim on
            # line 941 was reported as line 927.
            offset = sent_start + hit.get("offset", 0)
            hit["line"] = owners[offset] if owners and offset < len(owners) else 0
            findings.append(hit)
    return findings


def _scan_repo(artifact: dict) -> list[dict]:
    findings = []
    for rel in doc_files(REPO):
        path = REPO / rel
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for _first, _last, block in _paragraphs(lines):
            rows = _table_rows(block)
            if rows is not None:
                for hit in _scan_table(rows, artifact):
                    hit["file"] = rel
                    findings.append(hit)
                continue
            text, owners = _flatten_paragraph(block)
            if not CENSUS_CONTEXT_RE.search(text):
                continue
            for hit in _scan_prose(text, owners, artifact):
                hit["file"] = rel
                findings.append(hit)
    return findings


def test_every_census_statistic_a_document_quotes_is_the_current_value():
    artifact = load_artifact(ARTIFACT)
    findings = _scan_repo(artifact)
    assert not findings, (
        f"{len(findings)} stale census statistic claim(s) — the Wf-A134 guard "
        f"cannot see these because its exemptions are unit-scoped, not "
        f"value-scoped:\n"
        + "\n".join(
            f"  {f['file']}:{f.get('line', 0)}  {f['stat']}  quoted {f['quoted']}, "
            f"current {f['current']}  |  {f['text'][:140]}"
            for f in findings
        )
    )


# --------------------------------------------------------------------------
# C — null is not 0.0
# --------------------------------------------------------------------------
def test_null_rates_are_not_rendered_as_measured_zeros():
    """A run whose rate the artifact publishes as null must not be printed
    as 0.0 / 0.0000 NEAR its name.

    "Near" is measured, not "in the same sentence": the TODO/INDEX.md
    trailer quotes the trainer's `0.0` PLACEHOLDER and then names the two
    runs 100 characters later, and the placeholder is not their reading.
    """
    artifact = load_artifact(ARTIFACT)
    runs = {r.get("run") or r.get("run_name"): r for r in artifact["runs"]}
    findings = []
    for rel in doc_files(REPO):
        path = REPO / rel
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, raw in enumerate(lines, start=1):
            if not CENSUS_CONTEXT_RE.search(raw):
                continue
            for sentence in SENT_SPLIT_RE.split(raw):
                for run in NULL_RUNS:
                    if run not in sentence:
                        continue
                    if runs[run].get("gen_single_molecule_rate_end") is not None:
                        continue
                    text = _normalize(sentence)
                    if _is_historical(text, 0, len(text), 0, len(text)):
                        continue
                    name_at = text.find(run)
                    clauses = _clause_spans(text)
                    name_clause = _clause_of(clauses, name_at)
                    for match in re.finditer(
                        r"(?<![\w.])0\.0{1,3}(?![\w.])", text
                    ):
                        if abs(match.start() - name_at) > NULL_WINDOW:
                            continue
                        # A reading is stated AFTER the subject it belongs to,
                        # or in the same clause as it.  The TODO/INDEX.md
                        # trailer quotes the trainer's `0.0` PLACEHOLDER in
                        # one clause and names the two runs in the next — the
                        # placeholder is not their reading.
                        if match.start() < name_at:
                            if _clause_of(clauses, match.start()) != name_clause:
                                continue
                        findings.append(
                            (rel, lineno, run, sentence.strip()[:120])
                        )
                        break
    assert not findings, (
        "a run the artifact publishes as null is rendered as a measured zero:\n"
        + "\n".join(f"  {f}:{n}  {r}  |  {s}" for f, n, r, s in findings)
    )


def test_the_null_guard_still_bites_next_to_the_run_name():
    """The window must not disarm the guard: a zero right beside the run's
    name IS that run's reading and must be reported."""
    artifact = load_artifact(ARTIFACT)
    runs = {r.get("run") or r.get("run_name"): r for r in artifact["runs"]}
    run = NULL_RUNS[0]
    assert runs[run].get("gen_single_molecule_rate_end") is None, (
        "the fixture run is no longer a null-rate run"
    )
    sentence = (
        f"`cfm_run_census.json` lists `{run}` at 0.0 for the last-epoch "
        f"single-molecule rate."
    )
    text = _normalize(sentence)
    name_at = text.find(run)
    clauses = _clause_spans(text)
    name_clause = _clause_of(clauses, name_at)
    hits = [
        match
        for match in re.finditer(r"(?<![\w.])0\.0{1,3}(?![\w.])", text)
        if abs(match.start() - name_at) <= NULL_WINDOW
        and (
            match.start() >= name_at
            or _clause_of(clauses, match.start()) == name_clause
        )
    ]
    assert hits, "the null guard no longer reports a zero beside the run's name"


# --------------------------------------------------------------------------
# D — the guard really covers the class that bit us
# --------------------------------------------------------------------------
def _scan_text(text: str, artifact: dict) -> list[dict]:
    """Scan one prose snippet (used by the mutation tests)."""
    normalised = _normalize(text)
    findings = []
    for sent_start, sent_end in _sentences(normalised):
        for hit in _scan_sentence(artifact, normalised, sent_start, sent_end):
            findings.append(hit)
    return findings


#: The four blocks Wf-A152 struck, in their PRE-repair wording.
STRUCK_BLOCKS = (
    (
        "audit §2.5 — the CLEAN last-epoch block",
        "the `fragmentation_summary.end_to_end_single_molecule_rate` block",
        "> `fragmentation_summary.end_to_end_single_molecule_rate` (**CLEAN**, n = 29,\n"
        "> min 0.0, **median 0.057291666666666664**, mean 0.16531059620275626, **max\n"
        "> 0.5572916666666666**, `n_below_0.50` = 25, `n_below_0.10` = 18),\n"
        "> `…_all_runs` (n = 32, **median 0.05989583333333333**, max **0.572265625**\n"
        "> — still the WITHDRAWN `cfm_r11_eval256` value — `n_below_0.50` = 27),\n",
    ),
    (
        "audit §6.5 — the 256-cohort last-epoch medians",
        "the `by_n_train` cohort medians",
        "> the 26715 cohort now holds **14** runs with an end-to-end rate (11 CLEAN)\n"
        "> and the 256 cohort **8** (all CLEAN), and the medians are now\n"
        "> **0.125 / 0.026041666666666664** (last epoch, CLEAN),\n"
        "> **0.12357954545454546 / 0.026041666666666664** (last epoch, all),\n",
    ),
    (
        "pending/wfa114 — the CLEAN medians block",
        "the project-wide median",
        "> (contiguous-from-zero epoch history AND no `nan` train_loss): **median\n"
        "> 0.057291666666666664 over 29 clean runs of 32**, of which **25 are below\n"
        "> 0.50** and **18 are below 0.10**. The all-runs median is\n"
        "> `0.05989583333333333` over 32 runs, and **its maximum, `0.572265625`, is the\n",
    ),
)


def test_the_guard_catches_each_block_wfa152_struck():
    """D — inject each pre-repair block and assert the parser flags it."""
    artifact = load_artifact(ARTIFACT)
    missed = []
    for label, _where, phrase in STRUCK_BLOCKS:
        hits = _scan_text(phrase, artifact)
        if not hits:
            missed.append((label, phrase.strip()[:120]))
    assert not missed, (
        "the guard does NOT catch these superseded renderings:\n"
        + "\n".join(f"  {label}  |  {phrase}" for label, phrase in missed)
    )


def test_the_guard_catches_the_256_cohort_last_epoch_median():
    """The 256-cohort last-epoch median moved 0.026041666666666664 -> 0.03645833333333333."""
    artifact = load_artifact(ARTIFACT)
    phrase = (
        "> n_train 256, last epoch, CLEAN: median **0.026041666666666664** over 6 runs\n"
    )
    hits = _scan_text(phrase, artifact)
    assert hits, "the guard did NOT catch the superseded 256-cohort median"


# --------------------------------------------------------------------------
# D2 — the EMPHASIS blind spot: the production path and the mutation path
#      disagreed, so the guard proved it catches stale claims while being
#      blind to every BOLDED one.
# --------------------------------------------------------------------------
def _production_scan(text: str, artifact: dict) -> list[dict]:
    """Run the PRODUCTION path (``_flatten_paragraph``) over one paragraph."""
    block = [(index + 1, line) for index, line in enumerate(text.splitlines())]
    flattened, owners = _flatten_paragraph(block)
    return _scan_prose(flattened, owners, artifact)


def test_the_production_path_catches_a_stale_bolded_count():
    """D2 — ``_flatten_paragraph`` must blank ``*`` exactly like ``_normalize``.

    ``_scan_text`` (the MUTATION path) applies ``_normalize``, which blanks
    ``*`` and backticks, so ``n=**29**`` matches the ``n=<int>`` pattern.
    ``_scan_repo`` (the PRODUCTION path) applies ``_flatten_paragraph``,
    which until Wf-A153 blanked only the struck spans and left ``*`` in
    place — the pattern needs a digit right after the ``=``, and the ``**``
    sat in the way, so the SAME text was invisible to the scan that actually
    runs over the repo.  MEASURED 2026-11-01: ``n=**29**`` -> 1 hit on the
    mutation path, 0 on the production path.
    """
    artifact = load_artifact(ARTIFACT)
    stale = (
        "> (contiguous-from-zero epoch history AND no `nan` train_loss): the\n"
        "> CLEAN last-epoch median is **0.0625** (n=**29**, max 0.5572916666666666).\n"
    )
    hits = _production_scan(stale, artifact)
    assert hits, (
        "the PRODUCTION path is blind to a BOLDED count — `_flatten_paragraph` "
        "must blank `*` and backticks exactly as `_normalize` does"
    )
    assert hits[0].get("stat") == "clean/last/headline/n", hits[0]


def test_the_production_path_accepts_a_correct_bolded_count():
    """D2 — the fix must not make the guard fire on a CORRECT bolded claim."""
    artifact = load_artifact(ARTIFACT)
    current = (
        "> (contiguous-from-zero epoch history AND no `nan` train_loss): the\n"
        "> CLEAN last-epoch median is **0.0625** (n=**27**, max 0.5572916666666666).\n"
    )
    hits = _production_scan(current, artifact)
    assert not hits, f"the guard fired on a CORRECT bolded count: {hits}"


def test_the_two_paths_agree_on_a_bolded_claim():
    """D2 — the production and mutation paths must not disagree."""
    artifact = load_artifact(ARTIFACT)
    stale = (
        "> (contiguous-from-zero epoch history AND no `nan` train_loss): the\n"
        "> CLEAN last-epoch median is **0.0625** (n=**29**, max 0.5572916666666666).\n"
    )
    production = _production_scan(stale, artifact)
    mutation = _scan_text(stale, artifact)
    assert len(production) == len(mutation) and len(production) > 0, (
        f"the paths disagree: production={len(production)} mutation={len(mutation)}"
    )


# --------------------------------------------------------------------------
# D3 — the LIST MARKER: blanking every ``*`` destroyed the bullet boundary
# --------------------------------------------------------------------------
#: The audit doc's §7 bullet list, in the shape that matters: the SECOND
#: bullet names a cohort, the THIRD quotes a PROJECT-WIDE value.  The value
#: is correct at its stated scope; only a leak carries the cohort across.
BULLET_LEAK = (
    "   * **secondary bar:** against `cfm_fullcorpus_seed1` (best epoch 13;\n"
    "     strict cohort **26715/192**) — on a best-epoch reading;\n"
    "   * **context:** the project-wide last-epoch median **0.0625** over the\n"
    "     **27 CLEAN** runs of 30 (all-runs median **0.06380208333333333** over 30);\n"
)


def test_a_bullet_markers_cohort_does_not_leak_into_the_next_bullet():
    """D3 — the bullet boundary must survive the emphasis blanking.

    MEASURED 2026-11-01: blanking every ``*`` (the Wf-A153 emphasis fix)
    removed the bullet's own marker, so ``LIST_ITEM_RE`` could no longer
    split the paragraph at the bullet boundary.  The three bullets merged
    into ONE sentence, the cohort ``26715/192`` named in the second bullet
    stayed in force, and the PROJECT-WIDE median quoted in the third was
    resolved against that cohort — a FALSE POSITIVE on a correct claim.
    """
    artifact = load_artifact(ARTIFACT)
    hits = _production_scan(BULLET_LEAK, artifact)
    assert not hits, (
        "the guard fired on a project-wide value because the previous bullet's "
        f"cohort leaked across the bullet boundary: {hits}"
    )


def test_the_guard_still_bites_inside_a_bullet():
    """D3 — the bullet-boundary fix must not blind the guard.

    The SAME bullet list, with the project-wide all-runs median replaced by
    its superseded value: the guard must still fire, and it must fire at
    the PROJECT-WIDE scope — not at the cohort named in the previous bullet.
    """
    artifact = load_artifact(ARTIFACT)
    stale = BULLET_LEAK.replace("0.06380208333333333", "0.05989583333333333")
    hits = _production_scan(stale, artifact)
    assert len(hits) == 1, hits
    assert hits[0].get("stat") == "all/last/headline/median", hits[0]
    assert abs(hits[0]["quoted"] - 0.05989583333333333) < 1e-12, hits[0]


def test_the_mutation_path_keeps_the_bullet_boundary_too():
    """D3 — the fix must land on BOTH paths.

    ``_normalize`` (mutation) and ``_flatten_paragraph`` (production) blank
    emphasis independently, so both have to spare the list marker.
    """
    artifact = load_artifact(ARTIFACT)
    assert not _scan_text(BULLET_LEAK, artifact), (
        "the MUTATION path still leaks a cohort across a bullet boundary"
    )


# --------------------------------------------------------------------------
# D4 — the reported LINE was the sentence's first line, not the claim's
# --------------------------------------------------------------------------
def test_the_reported_line_is_the_claim_s_own_line():
    """D4 — ``_scan_sentence`` returns a BODY-relative offset.

    ``_scan_prose`` indexed that offset against the PARAGRAPH-relative
    ``owners``, so every finding was reported at the START of its sentence.
    MEASURED 2026-11-01: the audit doc's claim on line 941 was reported as
    line 927 — 14 lines away, in a different bullet.
    """
    artifact = load_artifact(ARTIFACT)
    text = (
        "The bar is a NAMED RUN, not a cohort median.\n"
        "Both baselines are full-corpus runs with an intact epoch history.\n"
        "* **context:** the project-wide all-runs last-epoch median\n"
        "  **0.05989583333333333** over 30 runs.\n"
    )
    hits = _production_scan(text, artifact)
    stale = [h for h in hits if abs(h["quoted"] - 0.05989583333333333) < 1e-12]
    assert len(stale) == 1, hits
    assert stale[0]["line"] == 4, stale[0]


# --------------------------------------------------------------------------
# E — mutation tests: the guard must FAIL on the defects it exists for
# --------------------------------------------------------------------------
def test_the_changelog_guard_detects_a_swapped_rate():
    """Re-introduce the Wf-A150 swap IN MEMORY and assert the guard bites."""
    artifact = load_artifact(ARTIFACT)
    swapped = json.loads(json.dumps(artifact))
    hits = 0
    for index, entry in enumerate(swapped["changelog"]["wf_a150"]):
        if isinstance(entry, str) and "cfm_fm_control_2026_10_09 (best-epoch rate" in entry:
            swapped["changelog"]["wf_a150"][index] = entry.replace(
                "cfm_fm_control_2026_10_09 (best-epoch rate 0.6875)",
                "cfm_fm_control_2026_10_09 (best-epoch rate 0.875)",
            ).replace(
                "cfm_bond_head_30ep_onecycle_2026_10_09 (best-epoch rate 0.875)",
                "cfm_bond_head_30ep_onecycle_2026_10_09 (best-epoch rate 0.6875)",
            )
            hits += 1
            break
    assert hits, "wf_a150 changelog entry not found"

    runs = {(r.get("run") or r.get("run_name")): r for r in swapped["runs"]}
    bad = []
    for key, run, rate in _changelog_rate_claims(swapped):
        truth = runs[run].get("gen_single_molecule_rate_best")
        if truth is not None and abs(rate - float(truth)) > 1e-12:
            bad.append((key, run, rate, truth))
    assert bad, "the changelog guard did NOT detect the re-introduced swap"


def test_the_statistic_guard_detects_an_injected_live_quote():
    """A retired rate quoted in a live census paragraph must be flagged even
    though the SAME paragraph carries a historical word (mechanism 1)."""
    artifact = load_artifact(ARTIFACT)
    phrase = (
        "> `fragmentation_summary.end_to_end_single_molecule_rate` (CLEAN, "
        "n = 29, min 0.0, **median 0.057291666666666664**, mean "
        "0.16531059620275626) — this paragraph is historical in tone.\n"
    )
    hits = _scan_text(phrase, artifact)
    assert hits, "the statistic guard did NOT detect the injected quote"


def test_the_statistic_guard_ignores_a_correctly_marked_historical_quote():
    """The same quote, marked as history next to the value, is NOT a finding."""
    artifact = load_artifact(ARTIFACT)
    phrase = (
        "> `fragmentation_summary.end_to_end_single_molecule_rate` (CLEAN, "
        "n = ~~29~~ **27**, min 0.0, **median ~~0.057291666666666664~~ 0.0625**, "
        "mean ~~0.16531059620275626~~ 0.17755582555110858) — historical.\n"
    )
    hits = _scan_text(phrase, artifact)
    assert not hits, f"the statistic guard flagged a correctly marked quote: {hits}"


# --------------------------------------------------------------------------
# Wf-A154 (2026-11-01) — the POP_RE vocabulary gap: the population label
#      "ALL best-epoch median" / "ALL last-epoch median" was INVISIBLE, so
#      the value under it resolved against the CLEAN population.  A STALE
#      all-runs value produced 0 hits (false negative) and a CORRECT one
#      produced a false positive against catalog §6.264.
# --------------------------------------------------------------------------
def test_the_guard_catches_a_stale_all_runs_value_under_the_all_epoch_label():
    """The blind spot is CLOSED: a stale all-runs median is now flagged.

    MEASURED 2026-11-01, before the POP_RE extension: this phrase produced
    **0 hits** on the production path — the value resolved against the
    CLEAN population, where 0.35546875 IS current, so the stale all-runs
    claim sailed through.
    """
    artifact = load_artifact(ARTIFACT)
    phrase = "> ALL best-epoch median **0.35546875** (n=32)\n"
    hits = _production_scan(phrase, artifact)
    assert hits, (
        "the guard is BLIND to a stale all-runs value under the "
        "'ALL best-epoch median' label — POP_RE cannot resolve the label"
    )
    assert hits[0].get("stat") == "all/best/headline/median", hits[0]


def test_the_guard_catches_a_stale_all_runs_last_epoch_value():
    """The same closure for the LAST-epoch label."""
    artifact = load_artifact(ARTIFACT)
    phrase = "> ALL last-epoch median **0.0625** (n=30)\n"
    hits = _production_scan(phrase, artifact)
    assert hits, (
        "the guard is BLIND to a stale all-runs last-epoch value under the "
        "'ALL last-epoch median' label"
    )
    assert hits[0].get("stat") == "all/last/headline/median", hits[0]


def test_the_guard_accepts_the_current_all_runs_values_under_that_label():
    """The closure must not turn a CORRECT all-runs claim into a finding."""
    artifact = load_artifact(ARTIFACT)
    for phrase in (
        "> ALL best-epoch median **0.34440104166666663** (n=32)\n",
        "> ALL last-epoch median **0.06380208333333333** (n=30)\n",
    ):
        hits = _production_scan(phrase, artifact)
        assert not hits, f"the guard fired on a CORRECT all-runs claim: {hits}"


def test_the_all_epoch_triple_does_not_fire_on_ordinary_english():
    """The new POP_RE alternative is narrow on purpose.

    It requires a population + an epoch + a statistic TRIPLE, so ordinary
    prose that merely says "all" and "epoch" must stay invisible — the
    objection that keeps a BARE "ALL" out of POP_RE in the first place.
    """
    artifact = load_artifact(ARTIFACT)
    for phrase in (
        "> we ran all last epochs of the sweep and kept the best\n",
        "> the all last epoch numbers agree with the changelog\n",
        "> in all epochs the median moved\n",
    ):
        hits = _production_scan(phrase, artifact)
        assert not hits, f"the new POP_RE alternative fired on prose: {hits}"
