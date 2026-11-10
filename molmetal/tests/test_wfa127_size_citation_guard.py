#!/usr/bin/env python
"""(§6.243) — a byte size quoted in prose must be a size that was true, on a date.

**The finding.**  The §6.242 sweep (value-directed: grep the repo for each
superseded VALUE, not for each file) also swept the **byte sizes** the SSOT
documents quote next to a ``TODO/<dir>/<name>.md`` path.  It found **28**
citation sites, of which **6** quote a size the file on disk no longer has —
the file was edited after the size was written down and the document was never
re-read.  That is the §6.241 class again (a repair that changes an artifact
invalidates every document that describes it), on a different number.

**Why a date is part of the claim.**  A bare byte size is unfalsifiable: it
has no timestamp, so nothing distinguishes "this size was true" from "this
size was true once".  Every byte-size citation therefore carries the date it
was measured on, and this guard enforces three things at once:

  1. the citation carries a measurement date;
  2. the quoted size equals the file's size on disk;
  3. the file has not been modified **after** the measurement date — a file
     edited later invalidates the size even if the size happens to match.

**Why this guard is not noisy.**  A guard on a volatile number is a guard that
gets disabled (the §6.241 lesson).  This one is anchored on the file's
**mtime**, not on the wall clock: it fires only when a cited file is actually
edited, which is exactly the event that invalidates the citation.  Idle time
never breaks it.

**What is deliberately NOT gated.**  Two of the cited documents also quote the
file's **mtime**.  An mtime is not reproducible — a fresh `git clone` gives
every file the checkout time — so gating it would fail for every reader who
clones the repo.  Mtimes are corrected when they are found stale, but they
are not gated.

**Scope.**  The four SSOT documents: the honest-negative catalog,
``TODO/INDEX.md``, ``TODO/README.md`` and ``docs/WF_HISTORY.md``.  A citation
site is a fully-qualified path whose **size token immediately follows the
path** (``path (12,898 B)``, ``path 8,155 B atomic-written``).  Short-form
references (``A5_final_synthesis.md``, bare ``(7,706 B) — A2`` list items) do
not name a directory and are outside the scope of this guard.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The SSOT documents whose byte-size citations are checked.
DOCUMENTS: tuple[str, ...] = (
    "docs/HONEST_NEGATIVE_CATALOG.md",
    "TODO/INDEX.md",
    "TODO/README.md",
    "docs/WF_HISTORY.md",
)

#: ``TODO/<dir>/<name>.md`` — a fully-qualified cross-reference.
PATH_RE = re.compile(r"TODO/(?:completed|pending|archive)/[A-Za-z0-9_.-]+\.md")

#: The size token that immediately follows a path.  The optional second
#: number is the ``old → new`` chain used by dated corrections.
#:
#: No ``^`` anchor: ``Pattern.match(text, pos)`` already anchors at ``pos``,
#: and ``^`` would only match at offset 0.
SITE_RE = re.compile(
    r"[\s`]*\(?[\s`]*"
    r"(\d{1,3}(?:,\d{3})+|\d{3,7})[\s`]*B"
    r"(?:[\s`]*→[\s`]*(\d{1,3}(?:,\d{3})+|\d{3,7})[\s`]*B)?"
)

#: The measurement date a byte-size citation must carry.
DATE_RE = re.compile(r"measured\s+(\d{4}-\d{2}-\d{2})")

#: How far after the size chain the measurement date may sit.
DATE_WINDOW = 60


def _resolve(path_text: str) -> Path | None:
    """Resolve a citation to a file on disk, or None if it cannot be resolved.

    Abbreviated references (``TODO/completed/234_...md``) are resolved by
    prefix glob; the prefix must match exactly one file.
    """
    if "..." in path_text:
        prefix = path_text.split("...")[0]
        matches = sorted(REPO.glob(prefix + "*"))
        if len(matches) != 1:
            return None
        return matches[0]
    candidate = REPO / path_text
    return candidate if candidate.exists() else None


def _sites() -> list[tuple[str, int, str, int, str]]:
    """Every byte-size citation site.

    Returns (document, line, path, quoted size, following context).  The
    context is the text immediately after the size chain — that is where the
    measurement date lives (``12,898 B, measured 2026-10-31``).
    """
    out: list[tuple[str, int, str, int, str]] = []
    for rel in DOCUMENTS:
        path = REPO / rel
        assert path.exists(), f"guarded document is missing: {rel}"
        text = path.read_text(encoding="utf-8")
        for m in PATH_RE.finditer(text):
            site = SITE_RE.match(text, m.end())
            if site is None:
                continue
            sizes = [int(g.replace(",", "")) for g in site.groups() if g]
            quoted = sizes[-1]
            line_no = text[: m.start()].count("\n")
            context = text[site.end() : site.end() + DATE_WINDOW]
            out.append((rel, line_no + 1, m.group(0), quoted, context))
    return out


def test_every_byte_size_citation_is_dated_and_true() -> None:
    """Every quoted byte size must be dated, current, and not superseded."""
    sites = _sites()
    assert len(sites) >= 20, (
        f"only {len(sites)} byte-size citation sites were found — the guard's "
        "scope has drifted (a document was restructured, or the guard is now "
        "vacuous)."
    )

    problems: list[str] = []
    resolved = 0
    checked = 0
    for rel, line_no, path_text, quoted, context in sites:
        target = _resolve(path_text)
        if target is None:
            # Unresolvable abbreviation: out of scope, not a failure.
            continue
        resolved += 1
        date_m = DATE_RE.search(context)
        if date_m is None:
            problems.append(
                f"{rel}:{line_no} quotes {quoted} B for {path_text} with NO "
                "measurement date — a bare byte size cannot be falsified"
            )
            continue
        measured = _dt.date.fromisoformat(date_m.group(1))
        checked += 1
        actual = target.stat().st_size
        if actual != quoted:
            problems.append(
                f"{rel}:{line_no} quotes {quoted} B for {path_text}, but the "
                f"file on disk is {actual} B (re-measure and re-date it)"
            )
            continue
        touched = _dt.date.fromtimestamp(target.stat().st_mtime)
        if touched > measured:
            problems.append(
                f"{rel}:{line_no} quotes {quoted} B for {path_text} as "
                f"measured on {measured}, but the file was modified on "
                f"{touched} — the size was superseded and the document was "
                "never re-read"
            )

    assert resolved, "no byte-size citation could be resolved — the guard is vacuous"
    assert not problems, (
        "byte-size citations in the SSOT documents that are undated, wrong, or "
        f"superseded (catalog §6.243): {problems}"
    )


def test_the_guard_does_not_measure_its_own_outputs() -> None:
    """The guard must not treat its own test file as a cited document."""
    assert "molmetal/tests/test_wfa127_size_citation_guard.py" not in DOCUMENTS
    for rel in DOCUMENTS:
        assert (REPO / rel).exists(), f"guarded document is missing: {rel}"
