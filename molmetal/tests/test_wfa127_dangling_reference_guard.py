#!/usr/bin/env python
"""(§6.242) — a cross-reference to a file that does not exist is a dead end.

**The finding.**  Wf-A127's value-directed sweep over the SSOT documents
found **50** paths of the form ``TODO/<dir>/<name>.md`` that the documents
point at and the repository does not contain — **40** in ``TODO/completed/``
and **10** in ``TODO/pending/``.  Under AGENTS.md §5.5 (filesystem-truth) a
"SHIPPED" claim whose evidence file is absent is a soft false claim: a
reader who follows the reference gets nothing.

**Why a baseline and not a hard failure.**  The **50** are PRE-EXISTING debt
inherited from waves older than this chain — this wave did not create them,
and deleting the historical references would violate AGENTS.md §5.5 rule #6
(the historical trail is preserved verbatim; corrections are additive and
dated).  So the guard asserts that the dangling set **has not GROWN**: any
path that is neither present on disk nor in the recorded baseline fails the
test.  Fixing a baseline entry (restoring the file, or annotating the
reference) shrinks the set and is always allowed.

**Scope.**  Four SSOT documents: the honest-negative catalog, ``TODO/INDEX.md``,
``TODO/README.md`` and ``docs/WF_HISTORY.md``.

**What this guard covers — and what it does not.**  The path family is
``TODO/(completed|pending/archive)/<name>.md``.  Every other repo-relative
path family (``metrics/``, ``molmetal/``, ``scripts/``, ``configs/``,
``docs/``, ``tests/`` …) is owned by the **Wf-A131** guard
(``test_wfa131_repo_path_reference_guard.py``), which resolves them against
the filesystem and against ``registry/metrics.yaml``.  The two guards
coexist; neither duplicates the other.

**The escape hatches are closed (Wf-A131 C2 adversarial review).**  Four
conventions could previously hide a dangling reference, and each is now a
**named, audited** list rather than a silent hole:

0. the **documented non-existence** exemption (:data:`DOCUMENTED_NON_EXISTENCES`)
   — Wf-A269 wave-3 (2026-11-02) added it as the FOURTH hatch, when
   wave-2's new ``NEVER_EXISTED`` catalog row made this guard report a
   TRUE statement (a path named in order to say it does not exist).  The
   superseded rendering above is preserved VERBATIM per AGENTS.md §5.5
   rule #6: it read "Three conventions could previously hide a dangling
   reference".  The three it named are:

1. the repo's ``...`` prose-placeholder convention — only the two audited
   abbreviations (:data:`PROSE_ABBREVIATIONS`) and the one audited exemplar
   (:data:`ILLUSTRATIVE_EXEMPLARS`) are exempt; every other dotted name is
   checked like any other path;
2. the ``HISTORICAL_MOVES`` exemption — it is **site-keyed**: a FROM-path is
   exempt only inside the *record* (table row, fenced block or paragraph)
   that also names the live counterpart, so a *new* citation of the same
   string is reported;
3. the "counterpart is still cited" assertion — it resolves the counterpart
   through a strict, path-boundary **code-span** reference extractor
   instead of a raw substring test, so ``...redefinition.md.ORIG`` or a
   prose "since deleted" mention no longer satisfy it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The SSOT documents whose cross-references are checked.
DOCUMENTS: tuple[str, ...] = (
    "docs/HONEST_NEGATIVE_CATALOG.md",
    "TODO/INDEX.md",
    "TODO/README.md",
    "docs/WF_HISTORY.md",
)

#: ``TODO/<dir>/<name>.md`` — a cross-reference to a project document.
REF = re.compile(r"TODO/(?:completed|pending|archive)/[A-Za-z0-9_.-]+\.md")

#: A reference that must END at a path boundary.  ``REF`` is greedy, so
#: ``...redefinition.md.ORIG`` still *matches* ``...redefinition.md`` — this
#: stricter form refuses it, because the matched path must not be continued
#: by a word character, a slash, a dash, or a dot followed by a word
#: character.
STRICT_REF = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"TODO/(?:completed|pending|archive)/[A-Za-z0-9_.-]+\.md"
    r"(?![A-Za-z0-9_/-])"
    r"(?!\.[A-Za-z0-9])"
)

#: A markdown code span.  A path the repo means as a POINTER is written
#: inside one; a path mentioned in prose is not.
CODE_SPAN = re.compile(r"`[^`\n]*`")

#: The number of PRE-EXISTING dangling references, as recorded in catalog
#: §6.242 (MEASURED 2026-10-31) and CLOSED in Wf-A128 (2026-10-31).
#: The **50** were repaired: **47** were pointers to a document that
#: exists under a different name, and were redirected at it; the other
#: **3** are historical MOVE records whose counterpart is cited in the
#: same row and are exempted below (with an assertion that each still
#: has that live counterpart).  The baseline is therefore EMPTY and the
#: guard is now ZERO-TOLERANCE: any dangling reference at all fails,
#: not merely any NEW one.
BASELINE_SIZE = 0

#: The PRE-EXISTING dangling references (catalog §6.242).  These are
#: inherited debt; the guard exists to stop the set from GROWING, not to
#: delete the historical trail (AGENTS.md §5.5 rule #6).
BASELINE: frozenset[str] = frozenset()

#: HISTORICAL MOVE RECORDS — the FROM column of a "MOVED pending/ ->
#: completed/" record (catalog §6.242 / TODO/INDEX.md §22).  These paths
#: are **not** broken pointers: the file was moved (or its stale
#: ``pending/`` duplicate was dropped) and the record's own row carries
#: the live path.  Rewriting them would destroy the record of the move,
#: which AGENTS.md §5.5 rule #6 forbids.
#:
#: Each entry maps the historical path to the live path that replaces
#: it; ``test_every_historical_move_still_has_a_live_counterpart``
#: asserts that the live file exists AND is still cited, so the
#: exemption cannot hide a file that was actually lost.
#:
#: This is deliberately NOT implemented with the repo's ``...``
#: placeholder convention: that convention is an un-guarded escape hatch
#: (catalog §6.242 honest negative #5).  A named mapping with its own
#: assertion is auditable; a hole is not.
HISTORICAL_MOVES: dict[str, str] = {
    "TODO/pending/206_m6_kappa_metric_redefinition.md": "TODO/completed/206_m6_kappa_metric_redefinition.md",
    "TODO/pending/234_wf234_frontier_math_audit_integration.md": "TODO/completed/234_wf234_frontier_math_audit_integration.md",
    "TODO/pending/383_se3_sb_fast_converging_integration.md": "TODO/completed/383_se3_sb_fast_converging_integration.md",
}

#: The audited prose abbreviations — the repo's ``...`` convention, made
#: explicit and checkable.  Each entry maps the abbreviated citation to the
#: LIVE file it abbreviates; the live file must exist, must start with the
#: abbreviated stem, and must itself be cited as a live pointer somewhere in
#: the guarded documents — so the abbreviation always has a resolvable
#: destination.
#:
#: ``TODO/completed/383_...md`` abbreviates ``383_se3_sb_fast_converging_integration.md``
#: and NOT ``383_final_synthesis.md``; the stem prefix alone is ambiguous
#: (two files match it), which is exactly why the mapping is named.
PROSE_ABBREVIATIONS: dict[str, str] = {
    "TODO/completed/234_...md": "TODO/completed/234_wf234_frontier_math_audit_integration.md",
    "TODO/completed/383_...md": "TODO/completed/383_se3_sb_fast_converging_integration.md",
}

#: The audited illustrative exemplars — paths that are DELIBERATELY
#: fictional.  ``TODO/completed/zzz_...md`` is the guard's own negative-test
#: exemplar in catalog §6.242: the catalog quotes it while showing what the
#: guard prints when it fires.  It must not exist, and it may only be cited
#: inside a negative-test context (a fenced block, or a line that says the
#: guard FAILED / the path does not exist) — never as evidence.
ILLUSTRATIVE_EXEMPLARS: frozenset[str] = frozenset({"TODO/completed/zzz_...md"})

#: A citation of an exemplar is only legitimate in a negative-test context.
EXEMPLAR_CONTEXT_MARKERS: tuple[str, ...] = (
    "FAIL",
    "does not exist",
    "not a real file",
    "baseline",
    "negative",
)

#: The audited **documented non-existences** — paths that a guarded document
#: names in order to ASSERT that they do not exist.
#:
#: The honest-negative catalog carries a ``NEVER_EXISTED`` category row
#: whose subject is precisely a path that exists nowhere in the repo.  Such
#: a citation is a **statement about** the path, not a pointer **to** it, so
#: counting it as a dangling reference makes the zero-tolerance guard
#: report the truth it is supposed to record.
#:
#: Each entry maps the path to the LABEL that classifies it as a
#: non-existence.  The exemption is **site-keyed**, exactly like
#: :data:`HISTORICAL_MOVES`: the path is exempt ONLY inside the record
#: (table row / fenced block / paragraph) that carries that label — a NEW
#: citation of the same string, in any other record, is still reported.
#:
#: Two assertions keep it auditable (see
#: ``test_every_documented_non_existence_is_absent_and_cited_as_such``):
#: the path must still be ABSENT from disk, and it must still be cited in a
#: record carrying the label.  A third test proves the exemption is
#: CONDITIONAL on the label.
#:
#: Wf-A269 wave-3 (2026-11-02) — added when wave-2's NEW ``NEVER_EXISTED``
#: catalog row made this zero-tolerance guard RED on a TRUE statement.
DOCUMENTED_NON_EXISTENCES: dict[str, str] = {
    "TODO/completed/220_wf220_temporal_split_acceptance.md": "NEVER_EXISTED",
}


#: Which guarded documents the last scan actually read.  A guard that stops
#: reading a document cannot report what that document cites, so the scan
#: surface is asserted rather than assumed.
SCANNED_DOCUMENTS: list[str] = []


# ------------------------------------------------------------------ records
def _records(text: str) -> list[tuple[int, int, int, int]]:
    """The document's citation records as ``(line_lo, line_hi, off_lo, off_hi)``.

    A *record* is one unit of citation context:

    * a fenced code block (its whole span),
    * a markdown table ROW (the ``|``-prefixed line plus its continuation
      lines),
    * a paragraph (consecutive non-blank lines).

    The character offsets are given so that a record's text can be sliced
    without re-splitting the document.
    """
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    offsets.append(pos)

    out: list[tuple[int, int, int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            start = i
            i += 1
            while i < n:
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            out.append((start + 1, i, offsets[start], offsets[i]))
            continue
        if not stripped:
            i += 1
            continue
        start = i
        if stripped.startswith("|"):
            i += 1
            while (
                i < n
                and lines[i].strip()
                and not lines[i].strip().startswith("|")
                and not lines[i].strip().startswith("```")
                and not lines[i].strip().startswith("~~~")
            ):
                i += 1
        else:
            while (
                i < n
                and lines[i].strip()
                and not lines[i].strip().startswith("```")
                and not lines[i].strip().startswith("~~~")
            ):
                i += 1
        out.append((start + 1, i, offsets[start], offsets[i]))
    return out


def _line_of(offsets: list[int], position: int) -> int:
    """The 1-indexed line number of a character offset."""
    lo, hi = 0, len(offsets) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= position:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _line_offsets(text: str) -> list[int]:
    offsets: list[int] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        offsets.append(pos)
        pos += len(line)
    offsets.append(pos)
    return offsets


# --------------------------------------------------------------------- scan
def _move_targets(old: str) -> tuple[str, ...]:
    """The acceptable destinations of a move record for a given FROM-path."""
    counterpart = HISTORICAL_MOVES[old]
    targets = [counterpart]
    targets += [
        abbrev for abbrev, live in PROSE_ABBREVIATIONS.items() if live == counterpart
    ]
    return tuple(targets)


def _is_move_site(
    text: str,
    records: list[tuple[int, int, int, int]],
    offsets: list[int],
    position: int,
    old: str,
) -> bool:
    """True when the FROM-path at ``position`` sits in a move record.

    The exemption is **site-keyed**, not path-keyed: a FROM-path is exempt
    only inside the record (table row / fenced block / paragraph) that also
    names the live counterpart.  A new citation of the same string, in any
    other record, is a dangling reference and is reported.
    """
    line = _line_of(offsets, position)
    for line_lo, line_hi, off_lo, off_hi in records:
        if not (line_lo <= line <= line_hi):
            continue
        chunk = text[off_lo:off_hi]
        if any(target in chunk for target in _move_targets(old)):
            return True
    return False


def _is_non_existence_site(
    text: str,
    records: list[tuple[int, int, int, int]],
    offsets: list[int],
    position: int,
    marker: str,
) -> bool:
    """True when the path at ``position`` sits in a record that carries the
    label that classifies it as a documented non-existence.

    Like the move exemption, this is **site-keyed**: the path is exempt
    only inside the record (table row / fenced block / paragraph) that
    carries ``marker``.  A new citation of the same string, in any other
    record, is a dangling reference and is reported.
    """
    line = _line_of(offsets, position)
    for line_lo, line_hi, off_lo, off_hi in records:
        if not (line_lo <= line <= line_hi):
            continue
        if marker in text[off_lo:off_hi]:
            return True
    return False


def _referenced_paths(
    repo: Path | None = None,
    texts: dict[str, str] | None = None,
) -> set[str]:
    """Every cross-reference path named by the SSOT documents.

    ``texts`` overrides the on-disk content of individual documents.  It
    exists so that a PERTURBED COPY of a document can be scanned against the
    REAL filesystem — the proof that the guard fires is written to ``/tmp``,
    never to the repository (AGENTS.md §5.5).
    """
    root = repo or REPO
    refs: set[str] = set()
    for rel in DOCUMENTS:
        text = texts[rel] if texts and rel in texts else _document_text(rel, root)
        SCANNED_DOCUMENTS.append(rel)
        records = _records(text)
        offsets = _line_offsets(text)
        for match in REF.finditer(text):
            token = match.group(0)
            if "..." in token:
                # The repo's ``...`` placeholder convention.  Only the two
                # audited abbreviations and the one audited exemplar are
                # exempt; every OTHER dotted name is a fully specified path
                # that does not exist, and is reported like any other.
                if token in PROSE_ABBREVIATIONS or token in ILLUSTRATIVE_EXEMPLARS:
                    continue
            if token in HISTORICAL_MOVES:
                # A recorded move — exempt only inside a move record.
                if _is_move_site(text, records, offsets, match.start(), token):
                    continue
            if token in DOCUMENTED_NON_EXISTENCES:
                # A documented non-existence — exempt only inside the record
                # that says the path does not exist.
                marker = DOCUMENTED_NON_EXISTENCES[token]
                if _is_non_existence_site(
                    text, records, offsets, match.start(), marker
                ):
                    continue
            refs.add(token)
    return refs


def _dangling_paths(
    repo: Path | None = None,
    texts: dict[str, str] | None = None,
) -> set[str]:
    root = repo or REPO
    return {ref for ref in _referenced_paths(root, texts) if not (root / ref).is_file()}


def _document_text(rel: str, repo: Path) -> str:
    path = repo / rel
    assert path.exists(), f"guarded document is missing: {rel}"
    return path.read_text(encoding="utf-8")


def _read_documents(repo: Path | None = None) -> dict[str, str]:
    """``{rel: text}`` for the four guarded documents.

    Every read of the guarded documents goes through this ONE function, so
    that a perturbed COPY of a document can be substituted for the on-disk
    text — the perturbation then lives in ``/tmp`` and the repository is
    never written to (AGENTS.md §5.5).
    """
    root = repo or REPO
    return {rel: _document_text(rel, root) for rel in DOCUMENTS}


def _fenced_line_numbers(text: str) -> set[int]:
    """1-indexed line numbers that sit inside a fenced code block."""
    inside: set[int] = set()
    fence = False
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = not fence
            continue
        if fence:
            inside.add(i)
    return inside


def _live_pointer_references(text: str) -> set[str]:
    """The references in ``text`` that point at a file.

    Two filters, both needed:

    * **path boundary** — ``...redefinition.md.ORIG`` is not a reference to
      ``...redefinition.md``, and a reference may not be a fragment of a
      longer word;
    * **code span** — the repo writes a path it means as a POINTER inside
      backticks.  A bare path in prose ("the file formerly at X (since
      deleted)") is a mention, not a pointer, and must not satisfy the
      "counterpart is still cited" assertion.
    """
    out: set[str] = set()
    for span in CODE_SPAN.findall(text):
        for match in STRICT_REF.finditer(span):
            out.add(match.group(0))
    return out


# -------------------------------------------------------------------- tests
def test_no_new_dangling_cross_reference_was_introduced() -> None:
    """Every cross-reference the SSOT docs make must resolve, or be baseline."""
    dangling = _dangling_paths()
    new_dangling = sorted(dangling - BASELINE)
    assert not new_dangling, (
        "these cross-references point at files that do not exist and are NOT "
        "part of the recorded §6.242 baseline — a document now points at "
        "nothing (AGENTS.md §5.5 filesystem-truth): "
        f"{new_dangling}"
    )


def test_the_baseline_matches_the_recorded_measurement() -> None:
    """The baseline must match the historical measurement, not drift.

    The recorded measurement is "zero dangling references" (``BASELINE_SIZE``
    is 0), so the live corpus is measured against it — the test fails on
    document content, not only when the constant is edited.
    """
    assert len(BASELINE) == BASELINE_SIZE, (
        f"the §6.242 baseline holds {len(BASELINE)} paths, but the catalog "
        f"records {BASELINE_SIZE} — the baseline was edited without updating "
        "the entry."
    )
    dangling = _dangling_paths()
    assert len(dangling) == BASELINE_SIZE, (
        f"the corpus now holds {len(dangling)} dangling references, but the "
        f"recorded §6.242 measurement is {BASELINE_SIZE} — the guard's zero "
        f"tolerance is not met.  Offending paths: {sorted(dangling)}"
    )
    stale = sorted(BASELINE - dangling)
    # A repaired baseline entry is good news, not a failure; report it so the
    # baseline can be tightened, but do not fail the test.
    if stale:
        print("§6.242 baseline can be tightened — these are now resolved: " f"{stale}")


def test_every_historical_move_still_has_a_live_counterpart() -> None:
    """A move exemption must not be able to hide a file that was actually lost.

    The ``HISTORICAL_MOVES`` exemption exists because a move record's FROM
    column names a path that legitimately no longer exists.  That is only
    safe while the file really did move: the replacement path must still
    name a file that is on disk, and that path must still be cited — so
    the record still points somewhere.  If a move record ever outruns its
    file, this fails — the exemption cannot silently become a place to hide
    a dangling reference.
    """
    for old, new in sorted(HISTORICAL_MOVES.items()):
        assert (REPO / new).exists(), (
            f"historical move record {old} points at {new}, which does not "
            "exist — the exemption would be hiding a lost file"
        )
        docs = _read_documents()
        cited = any(new in docs[rel] for rel in DOCUMENTS)
        assert cited, (
            f"historical move record {old} has a counterpart {new} that is no "
            "longer cited anywhere — the record now points at nothing"
        )
        # The raw substring test above is necessary but not sufficient: a
        # mention that is not a pointer must not satisfy it.
        cited_as_pointer = any(
            new in _live_pointer_references(docs[rel]) for rel in DOCUMENTS
        )
        assert cited_as_pointer, (
            f"historical move record {old} has a counterpart {new} that is "
            "no longer cited as a LIVE POINTER — the raw text mentions it, "
            "but no reference points at the file any more"
        )


def test_every_documented_non_existence_is_absent_and_cited_as_such() -> None:
    """Each documented non-existence must still be a non-existence, and the
    sentence that says so must still be there.

    Without the first half, a path that later APPEARS would stay exempt
    (a stale exemption is a hole).  Without the second half, the
    exemption could outlive the statement that justified it.
    """
    for path, marker in sorted(DOCUMENTED_NON_EXISTENCES.items()):
        assert not (REPO / path).exists(), (
            f"{path} is adjudicated as a documented non-existence, but it now "
            "exists on disk — the exemption is stale and must be removed."
        )
        cited = False
        for rel in DOCUMENTS:
            text = _document_text(rel, REPO)
            records = _records(text)
            offsets = _line_offsets(text)
            for match in REF.finditer(text):
                if match.group(0) != path:
                    continue
                if _is_non_existence_site(
                    text, records, offsets, match.start(), marker
                ):
                    cited = True
        assert cited, (
            f"{path} is adjudicated as a documented non-existence, but no "
            f"guarded document cites it in a record carrying the label "
            f"{marker!r} — the exemption has outlived its justification."
        )


def test_the_non_existence_exemption_needs_its_label(tmp_path: Path) -> None:
    """The exemption is CONDITIONAL on the label.

    A PERTURBED COPY of the catalog (in memory — the repository is never
    written to, AGENTS.md §5.5) whose label is replaced by an unrelated one
    must make the path a REPORTED dangling reference again.  Without this,
    the exemption would be a silent hole for any path that happened to
    appear in the same document.
    """
    rel = "docs/HONEST_NEGATIVE_CATALOG.md"
    text = _document_text(rel, REPO)
    path, marker = next(iter(sorted(DOCUMENTED_NON_EXISTENCES.items())))
    assert marker in text, f"the label {marker!r} is not in {rel}"
    perturbed = text.replace(marker, "AN_UNRELATED_LABEL")
    dangling = _dangling_paths(texts={rel: perturbed})
    assert path in dangling, (
        "dropping the label must make the documented non-existence a "
        "reported dangling reference again"
    )


def test_the_guard_is_not_vacuous() -> None:
    """The guard must still be able to FAIL.

    A guard that passes because it inspects nothing is worse than no
    guard.  This asserts the scan actually reads the SSOT documents and
    finds cross-references in them.
    """
    refs = _referenced_paths()
    assert len(refs) >= 20, (
        f"the dangling-reference guard only found {len(refs)} cross-references "
        "across the four SSOT documents — it is not scanning them, so it "
        "cannot fail"
    )


# --------------------------------------------------------------------------
# Wf-A131: the escape hatches, closed and audited.
# --------------------------------------------------------------------------


def test_every_guarded_document_is_scanned() -> None:
    """The guard must read all four SSOT documents, not a nominal subset.

    ``docs/WF_HISTORY.md`` happens to contribute ZERO references of this
    family today, which used to make dropping it from ``DOCUMENTS``
    invisible.  The scan log makes the surface explicit.
    """
    SCANNED_DOCUMENTS.clear()
    _referenced_paths()
    assert tuple(SCANNED_DOCUMENTS) == DOCUMENTS, (
        f"the guard read {tuple(SCANNED_DOCUMENTS)} but claims to guard "
        f"{DOCUMENTS} — a guarded document was not scanned"
    )


def test_every_prose_abbreviation_resolves_to_a_live_cited_file() -> None:
    """An audited ``...`` abbreviation must have a real, findable target.

    This is the assertion that keeps the repo's ``...`` convention from
    becoming an escape hatch: the abbreviation is not exempt because it
    looks like a placeholder, it is exempt because it is NAMED here and
    its destination is proved to exist and to be cited.
    """
    docs = _read_documents()
    for abbrev, live in sorted(PROSE_ABBREVIATIONS.items()):
        assert (REPO / live).is_file(), (
            f"prose abbreviation {abbrev} points at {live}, which does not "
            "exist — the exemption would be hiding a lost file"
        )
        stem = abbrev.rsplit("/", 1)[-1].split("...", 1)[0]
        assert live.rsplit("/", 1)[-1].startswith(stem), (
            f"prose abbreviation {abbrev} claims to abbreviate {live}, whose "
            f"name does not start with {stem!r}"
        )
        cited = any(live in _live_pointer_references(docs[rel]) for rel in DOCUMENTS)
        assert cited, (
            f"prose abbreviation {abbrev} abbreviates {live}, but that path "
            "is not cited anywhere — the abbreviation has no destination a "
            "reader could follow"
        )


def test_every_illustrative_exemplar_is_fictional_and_only_used_as_one() -> None:
    """An audited exemplar must stay fictional and stay an exemplar.

    The guard's own negative-test exemplar is quoted inside the catalog.
    It must NOT exist on disk (it is the point of the example), it must be
    cited at least once (so the exemption is load-bearing, not dead code),
    and it may only appear in a negative-test context — never as evidence.
    """
    docs = _read_documents()
    for exemplar in sorted(ILLUSTRATIVE_EXEMPLARS):
        assert not (REPO / exemplar).exists(), (
            f"illustrative exemplar {exemplar} now EXISTS on disk — it is no "
            "longer an exemplar, and the exemption must be re-adjudicated"
        )
        cited_somewhere = False
        for rel in DOCUMENTS:
            text = docs[rel]
            lines = text.splitlines()
            fenced = _fenced_line_numbers(text)
            for i, line in enumerate(lines, start=1):
                if exemplar not in line:
                    continue
                cited_somewhere = True
                assert i in fenced or any(
                    marker in line for marker in EXEMPLAR_CONTEXT_MARKERS
                ), (
                    f"illustrative exemplar {exemplar} is cited at "
                    f"{rel}:{i} outside a negative-test context — an "
                    "exemplar must never be cited as evidence"
                )
        assert cited_somewhere, (
            f"illustrative exemplar {exemplar} is not cited anywhere — the "
            "exemption is dead code and the negative test it documents has "
            "been removed from the corpus"
        )


def test_no_unaudited_ellipsis_path_escapes_the_guard() -> None:
    """Only the audited ``...`` names may be exempt.

    Every other reference containing three dots is a fully specified name
    that does not exist, and must be reported like any other dangling
    reference.
    """
    docs = _read_documents()
    for rel in DOCUMENTS:
        text = docs[rel]
        for match in REF.finditer(text):
            token = match.group(0)
            if "..." not in token:
                continue
            assert token in PROSE_ABBREVIATIONS or token in ILLUSTRATIVE_EXEMPLARS, (
                f"{rel} cites {token}, which contains the repo's ``...`` "
                "placeholder convention but is NOT an audited abbreviation "
                "or exemplar — the convention is an escape hatch unless it "
                "is named"
            )


def test_a_move_exemption_is_used_only_at_a_move_site() -> None:
    """A ``HISTORICAL_MOVES`` FROM-path is exempt only inside a move record.

    The exemption is keyed on the SITE, not on the path string: a new
    citation of the same string, presented as fresh evidence, must be
    reported.
    """
    docs = _read_documents()
    for rel in DOCUMENTS:
        text = docs[rel]
        records = _records(text)
        offsets = _line_offsets(text)
        for match in REF.finditer(text):
            token = match.group(0)
            if token not in HISTORICAL_MOVES:
                continue
            assert _is_move_site(text, records, offsets, match.start(), token), (
                f"{rel} cites the historical-move FROM-path {token} outside a "
                "move record (no live counterpart in the same record) — the "
                "exemption must not cover a fresh citation"
            )


def test_the_counterpart_assertion_rejects_a_non_pointer_mention() -> None:
    """The "still cited" check must not be satisfiable by prose.

    A raw substring search is satisfied by ``...redefinition.md.ORIG`` and by
    a sentence announcing that the file was deleted.  The strict extractor
    refuses both.
    """
    text = "the file formerly at TODO/completed/206_m6_kappa_metric_redefinition.md (since deleted)"
    assert (
        "TODO/completed/206_m6_kappa_metric_redefinition.md"
        not in _live_pointer_references(text)
    )
    text = "see TODO/completed/206_m6_kappa_metric_redefinition.md.ORIG"
    assert (
        "TODO/completed/206_m6_kappa_metric_redefinition.md"
        not in _live_pointer_references(text)
    )
    text = "see `TODO/completed/206_m6_kappa_metric_redefinition.md` for details"
    assert (
        "TODO/completed/206_m6_kappa_metric_redefinition.md"
        in _live_pointer_references(text)
    )


# --------------------------------------------------------------------------
# Wf-A131: the guard must still be able to FAIL (negative tests).
# --------------------------------------------------------------------------


def _fixture_repo(tmp_path: Path) -> Path:
    """A miniature repository with the four guarded documents present."""
    repo = tmp_path / "repo"
    (repo / "TODO/completed").mkdir(parents=True)
    (repo / "TODO/pending").mkdir(parents=True)
    (repo / "docs").mkdir(parents=True)
    (repo / "TODO/completed/real_doc.md").write_text("x\n", encoding="utf-8")
    for rel in DOCUMENTS:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("empty\n", encoding="utf-8")
    return repo


def test_the_guard_detects_a_dangling_reference(tmp_path: Path) -> None:
    """NEGATIVE TEST: the detection path is reachable, not just the scan.

    A guard that can scan but cannot detect is worse than no guard.  This
    builds a miniature repository, puts a dangling reference in it, and
    asserts the guard reports it.
    """
    repo = _fixture_repo(tmp_path)
    (repo / "TODO/README.md").write_text(
        "See `TODO/completed/999_new_dangling.md` for details.\n",
        encoding="utf-8",
    )
    dangling = _dangling_paths(repo)
    assert "TODO/completed/999_new_dangling.md" in dangling, (
        "the guard did not detect a dangling reference in a synthetic tree — "
        "the detection path is unreachable"
    )
    assert "TODO/completed/real_doc.md" not in dangling


def test_the_guard_detects_an_unaudited_ellipsis_path(tmp_path: Path) -> None:
    """NEGATIVE TEST: the ``...`` convention is no longer a hole.

    ``TODO/completed/999_wf999_brand_new_missing_doc_...md`` is fully
    specified (three dots in the middle of the stem, not at the end) and
    the file does not exist — it must be reported.
    """
    repo = _fixture_repo(tmp_path)
    (repo / "TODO/README.md").write_text(
        "See `TODO/completed/999_wf999_brand_new_missing_doc_...md`.\n",
        encoding="utf-8",
    )
    dangling = _dangling_paths(repo)
    assert "TODO/completed/999_wf999_brand_new_missing_doc_...md" in dangling, (
        "an unaudited '...' path escaped the guard — the placeholder "
        "convention is still a hole"
    )


def test_the_guard_detects_a_move_path_cited_outside_a_move_record(
    tmp_path: Path,
) -> None:
    """NEGATIVE TEST: the move exemption cannot hide a fresh citation.

    The FROM-path of a historical move is a live-looking citation.  Citing
    it as fresh evidence — with no live counterpart in the same record —
    must be reported.
    """
    repo = _fixture_repo(tmp_path)
    (repo / "TODO/README.md").write_text(
        "NEW-ENTRY evidence: the metric redefinition was shipped in "
        "`TODO/pending/206_m6_kappa_metric_redefinition.md` (this file is "
        "the authoritative artifact for the kappa redefinition claim).\n",
        encoding="utf-8",
    )
    dangling = _dangling_paths(repo)
    assert "TODO/pending/206_m6_kappa_metric_redefinition.md" in dangling, (
        "a fresh citation of a historical-move FROM-path escaped the guard "
        "— the exemption is path-keyed, not site-keyed"
    )


def test_the_guard_detects_a_directory_named_like_a_document(tmp_path: Path) -> None:
    """NEGATIVE TEST: a directory named ``*.md`` does not resolve a document.

    ``exists()`` accepts a directory; a reader following the reference gets
    a directory listing, not the document.
    """
    repo = _fixture_repo(tmp_path)
    (repo / "TODO/pending/999_directory_not_a_file.md").mkdir(parents=True)
    (repo / "TODO/README.md").write_text(
        "See `TODO/pending/999_directory_not_a_file.md` for details.\n",
        encoding="utf-8",
    )
    dangling = _dangling_paths(repo)
    assert (
        "TODO/pending/999_directory_not_a_file.md" in dangling
    ), "a directory named like a document resolved the reference"
