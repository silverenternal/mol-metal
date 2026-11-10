# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Wf-A269 wave-28 addendum — a published batch claim must NAME its population.

Run ONLY this file, and always with the GPU hidden::

    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        -m pytest molmetal/tests/test_wfa269a_published_batch_population_guard.py -v

WHY THIS FILE EXISTS (catalog §6.269.40.1).

A regression number that names its population's SIZE but not the population
itself is a **census**, not a measurement: two readers running "the 12-file
batch" get different counts, and neither can tell which one the published
number describes.  The repo already has the ruling — "PIN THE AXIS, NOT THE
COUNT" (§6.269.13) — and §6.269.35 published it **in the same entry** that
carries a populationless "58 files" census.  So the rule was published while
its violation was published next to it.

**THE RULE THIS FILE ENFORCES.**  A published claim of the shape
``"<N> files … batch/regression"`` must be resolvable to a population of
EXACTLY N existing files, named either

  * in the claim's own paragraph, or in its own section, as EXPLICIT
    ``molmetal/tests/...`` paths; or
  * in a section the claim points at (e.g. ``catalog §6.269.40.1``).

**GLOBS DO NOT COUNT.**  A glob is a MOVING population: the moment a new
test file lands, the glob resolves to a different set and every historical
claim that quoted it silently changes meaning.  A claim whose population is
named only by a glob is reported with its OWN message, because it is a
different (and worse) defect than naming nothing at all — it LOOKS like it
names a population.

If it is not, it must be a **DECLARED SKIP** — listed below with a reason,
and reported as a skip every run (the §5.11 config-binding gate's precedent:
*a declared skip is visible; an undeclared one is a violation, always*).

**WHAT THIS FILE DELIBERATELY DOES NOT DO.**  It does not pin the batch's
PASS/FAIL counts.  Those are dated readings (wall time drifts, the tree moves);
a freeze would be the §6.269.32 born-false pin in a new costume.  It pins the
POPULATION, which is the thing that makes the number mean anything.

**TWO ROUTES, AND THEY MUST AGREE.**  The claims are found by SPELLING (the
patterns below) and, independently, by MEANING (a size token in the same
sentence as a batch/regression concept word).  A site the meaning-sweep finds
and no pattern matches is a hole in the guard — a guard that cannot SEE a
shape cannot catch that shape going stale (catalog §6.269.37).

CPU-ONLY BY CONSTRUCTION: this file imports NOTHING but the standard library
and pytest — no torch, no numpy, no rdkit — so it is CI-enforceable.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The documents that publish batch claims.
CORPUS: tuple[str, ...] = (
    "docs/HONEST_NEGATIVE_CATALOG.md",
    "AGENTS.md",
    "TODO/INDEX.md",
    "TODO/README.md",
)

#: The document that DEFINES the sections a claim may point at.
CATALOGUE = "docs/HONEST_NEGATIVE_CATALOG.md"

#: A heading that names a section: ``### §6.269.40 — ...``.
SECTION_HEADING = re.compile(r"^#{1,6}\s*§(?P<id>[\d.]+)", re.MULTILINE)

#: A reference to a section from inside prose: ``§6.269.40.1``.
SECTION_REF = re.compile(r"§(?P<id>\d+(?:\.\d+)+)")

#: A token that names a test file.
#:
#: **A GLOB IS A MOVING POPULATION; AN EXPLICIT LIST IS A FIXED ONE.**  The
#: tokens are resolved as explicit paths only — a ``molmetal/tests/test_*.py``
#: glob is *not* counted, because the moment a new test file lands the glob
#: resolves to a different set and every historical claim that quoted it
#: would silently change meaning.  That is the §6.269.13 ruling ("PIN THE
#: AXIS, NOT THE COUNT") applied one level down: pin the MEMBERS.
FILE_TOKEN = re.compile(r"molmetal/tests/[A-Za-z0-9_.*/-]+\.py")

#: A glob token — recognised so the guard can SAY that a section names only a
#: moving population instead of silently counting zero.
GLOB_TOKEN = re.compile(r"molmetal/tests/[A-Za-z0-9_.*/-]*\*[A-Za-z0-9_.*/-]*\.py")

#: A size token: ``58 files`` / ``12-file``.
#:
#: The number must be STANDALONE — ``(?<![\w-])`` keeps ``A34-2 file`` (a
#: file NAME quoting a version, not a population size) out of the corpus.
#: Without that guard the sweep fires on the catalogue's own §6.54e line,
#: where "the post-A34-2 file" sits 60 characters from "the per-batch
#: tensor" — a FALSE POSITIVE that would have taught the sweep to be
#: ignored (§6.269.6: a census that only finds noise is not a census).
SIZE_TOKEN = re.compile(
    r"(?<![\w-])(?P<n>\d+)\s+files?\b|(?<![\w-])(?P<n2>\d+)-file\b"
)

#: The concept word that makes a size token a BATCH claim.
CONCEPT = re.compile(r"\bbatch(?:es)?\b|\bregression\b", re.IGNORECASE)

#: How close (in characters, same line) a size token must sit to the concept
#: word for the pair to be ONE claim.
WINDOW = 60

#: Claims found by SPELLING.  Each pattern must capture the size in group 1.
CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "the **12-file CPU-only regression batch: 164 passed ...**"
    re.compile(r"\b(\d+)-file\b[^\n]{0,80}?\bbatch\b", re.IGNORECASE),
    # "**THE REGRESSION BATCH (MEASURED, 58 files, CPU-only).**"
    re.compile(r"\bBATCH\b[^\n]{0,40}?\(MEASURED,\s*(\d+)\s+files\b", re.IGNORECASE),
    # "**THE 8 REGRESSION REDS, CLASSIFIED (MEASURED, 58 files, 431.73 s, ...)**"
    re.compile(r"\bREGRESSION REDS\b[^\n]{0,60}?\(MEASURED,\s*(\d+)\s+files\b", re.IGNORECASE),
    # "the batch over those 37 files measures ..." — the DELEGATED form, where
    # the population is written out below the claim rather than beside it.
    # Without this pattern the semantic sweep would find the site and no
    # pattern would match it (a §6.269.37 hole).
    re.compile(r"\bbatch(?:es)?\b[^\n]{0,60}?\b(\d+)\s+files\b", re.IGNORECASE),
)

#: Claims that are DECLARED SKIPS: (document, claim snippet, size, reason).
#:
#: A declared skip is REPORTED on every run (it is not silent) and must still
#: match at least one live site — otherwise the table has rotted and the
#: declaration is dead (catalog §6.269.32: a pin that cannot pass is as dead
#: as a guard that cannot fail).
#:
#: **THE SIZE IS PART OF THE DECLARATION.**  A skip exempts exactly ONE
#: claim SIZE, not every claim that happens to sit on the same line.  Without
#: that, a preserved "12-file" rendering in the same AGENTS.md table cell as
#: a new "37-file" claim would silently exempt the 37-file claim too — a skip
#: that silences the whole line is not a skip (the same defect the
#: config-binding gate's declared-skip table would have).
DECLARED_SKIPS: tuple[tuple[str, str, int, str], ...] = (
    (
        CATALOGUE,
        r"12-file CPU-only regression batch",
        12,
        "the wave-28 reading of the mechanism-shipped batch; its 12-file "
        "population was NEVER PUBLISHED and the session transcript that would "
        "carry it is unreadable, so the population is unrecoverable — the "
        "rendering is preserved verbatim per AGENTS.md §5.5 rule #6 and the "
        "live replacement publishes its population (catalog §6.269.40.1)",
    ),
    (
        CATALOGUE,
        r"58 files, CPU-only",
        58,
        "the §6.269.35 regression batch (2026-11-04); its 58-file population "
        "was never published and is unrecoverable — a census, not a "
        "measurement; preserved verbatim per AGENTS.md §5.5 rule #6 and "
        "disclosed at catalog §6.269.40.1",
    ),
    (
        CATALOGUE,
        r"58 files, 431\.73 s",
        58,
        "the §6.269.35 regression reds (2026-11-04); same unrecoverable "
        "58-file population as the batch line above — preserved verbatim per "
        "AGENTS.md §5.5 rule #6 and disclosed at catalog §6.269.40.1",
    ),
    (
        CATALOGUE,
        r"58-file CPU-only batch",
        58,
        "the §6.269.40.1 site's own TABLE ROW, which quotes the AGENTS.md "
        "rendering verbatim so the reader can see the census it describes — "
        "the quoted rendering names a size but not a population, which is the "
        "very defect the entry documents; same unrecoverable 58-file "
        "population as the two rows above",
    ),
    (
        "AGENTS.md",
        r"58-file CPU-only batch",
        58,
        "the §6.269.35 regression batch (2026-11-04), quoted in three AGENTS.md "
        "sites; its 58-file population was never published and is "
        "unrecoverable — preserved verbatim per AGENTS.md §5.5 rule #6 and "
        "disclosed at catalog §6.269.40.1",
    ),
    (
        "AGENTS.md",
        r"12-file CPU-only regression batch",
        12,
        "the wave-28 reading, quoted in three AGENTS.md sites; its 12-file "
        "population was never published and is unrecoverable — the live "
        "replacement publishes its population at catalog §6.269.40.1",
    ),
)


# ---------------------------------------------------------------------------
# Section / population resolution.
# ---------------------------------------------------------------------------


def _sections(text: str) -> dict[str, str]:
    """Map section id -> body text (the lines under its heading)."""
    out: dict[str, str] = {}
    heads = list(SECTION_HEADING.finditer(text))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[h.end() : end]
        out.setdefault(h.group("id"), body)
    return out


def _resolve_tokens(tokens: list[str]) -> set[str]:
    """Files on disk that the given EXPLICIT path tokens resolve to.

    **GLOBS ARE NOT EXPANDED.**  A glob is a MOVING population: the moment a
    new test file lands, the glob resolves to a different set and every
    historical claim that quoted it silently changes meaning.  So a claim
    whose population is named only by a glob is NOT resolved — it is
    reported as a moving population (see ``GLOB_TOKEN``).
    """
    found: set[str] = set()
    for tok in tokens:
        if "*" in tok:
            continue
        if (REPO / tok).exists():
            found.add(tok)
    return found


def _paragraph_sizes(text: str) -> set[int]:
    """Every population size named by a PARAGRAPH's own explicit paths.

    **PARAGRAPH-SCOPED, NOT SECTION-SCOPED.**  A section that names TWO
    populations (say 37 files in one list and 5 in another) must let both
    claims resolve; a single union over the section would yield 42 and
    resolve NEITHER — a false negative that would teach a reader to ignore
    the guard (§6.269.6: a census that only finds noise is not a census).
    """
    sizes: set[int] = set()
    for para in re.split(r"\n\s*\n", text):
        tokens = FILE_TOKEN.findall(para)
        if not tokens:
            continue
        resolved = _resolve_tokens(tokens)
        if resolved:
            sizes.add(len(resolved))
    return sizes


def _population_sizes(
    claim_para: str, claim_section_body: str, catalogue_sections: dict[str, str]
) -> set[int]:
    """Every population size this claim can be read as naming.

    Three sources, unioned:

      * the claim's OWN PARAGRAPH (a claim that lists its population
        inline);
      * the claim's OWN SECTION (a claim whose population is listed
        elsewhere in the same entry — e.g. the section's bullet list);
      * every section the claim's PARAGRAPH points at (the delegation
        form: "population: catalog §6.269.40.1").
    """
    sizes: set[int] = set()
    sizes |= _paragraph_sizes(claim_para)
    sizes |= _paragraph_sizes(claim_section_body)
    for ref in SECTION_REF.findall(claim_para):
        body = catalogue_sections.get(ref)
        if body is None:
            continue
        sizes |= _paragraph_sizes(body)
    return sizes


def _paragraph_of(text: str, line_no: int) -> str:
    """The PARAGRAPH (contiguous non-blank lines) containing ``line_no``.

    A claim is resolved against its own paragraph first, because a
    paragraph is the unit a reader uses: "the 37-file batch" and the list
    of 37 files belong to the same block of text, not merely to the same
    section.
    """
    lines = text.splitlines()
    idx = line_no - 1
    if not (0 <= idx < len(lines)):
        return ""
    i = idx
    while i > 0 and lines[i - 1].strip():
        i -= 1
    j = idx
    while j + 1 < len(lines) and lines[j + 1].strip():
        j += 1
    return "\n".join(lines[i : j + 1])


def _claim_sites(text: str) -> list[tuple[int, int, str]]:
    """Every (line number, claimed size, snippet) claim, found by SPELLING."""
    sites: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for pattern in CLAIM_PATTERNS:
        for m in pattern.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            n = int(m.group(1))
            if (line_no, n) in seen:
                continue
            seen.add((line_no, n))
            sites.append((line_no, n, m.group(0)[:120]))
    return sorted(sites)


def _swept_sites(text: str) -> list[tuple[int, int, str]]:
    """Every claim found by MEANING, independently of the pattern list.

    A size token counts as a claim when a batch/regression concept word sits
    within WINDOW characters on the SAME line.
    """
    sites: list[tuple[int, int, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        concepts = [m.start() for m in CONCEPT.finditer(line)]
        if not concepts:
            continue
        for m in SIZE_TOKEN.finditer(line):
            if any(abs(m.start() - c) <= WINDOW for c in concepts):
                n = int(m.group("n") or m.group("n2"))
                sites.append((i, n, line[m.start() : m.start() + 60]))
    return sites


def find_unresolved_claims(rel: str, text: str, catalogue_sections: dict[str, str]) -> list[str]:
    """Claims whose population cannot be resolved to exactly their size."""
    sections = _sections(text)
    # The section that contains a line.
    bounds = sorted(SECTION_HEADING.finditer(text), key=lambda m: m.start())

    def section_of(line_no: int) -> str:
        offset = sum(len(l) + 1 for l in text.splitlines()[: line_no - 1])
        current = ""
        for h in bounds:
            if h.start() <= offset:
                current = h.group("id")
            else:
                break
        return current

    problems: list[str] = []
    lines = text.splitlines()
    for line_no, n, snippet in _claim_sites(text):
        # A declared skip is allowed — but only if it really matches THIS
        # site, AND only for the size it declares.  Matching the whole
        # document instead would let one declared skip exempt every claim in
        # that document; matching the whole LINE instead would let it exempt
        # every claim that shares the line (a skip that silences the whole
        # line is not a skip either).
        line = lines[line_no - 1] if line_no - 1 < len(lines) else snippet
        declared = any(
            d_rel == rel and n == d_size and re.search(d_pat, line)
            for d_rel, d_pat, d_size, _d_reason in DECLARED_SKIPS
        )
        if declared:
            continue
        body = sections.get(section_of(line_no), "")
        para = _paragraph_of(text, line_no)
        sizes = _population_sizes(para, body, catalogue_sections)
        if n in sizes:
            continue
        # The two ways a population can fail to resolve.  They are REPORTED
        # differently because they are different defects: a claim that names
        # nothing at all is a census, and a claim that names only a GLOB is
        # worse — it LOOKS like it names a population while the population
        # moves under the number.
        globs = sorted(set(GLOB_TOKEN.findall(para + "\n" + body)))
        explicit = _resolve_tokens(FILE_TOKEN.findall(para + "\n" + body))
        if globs and not explicit:
            problems.append(
                f"{rel}:{line_no} claims a {n}-file batch but names only a "
                f"MOVING population (glob {globs[0]}) — a glob's membership "
                "changes the moment a file lands, so the count it denotes is "
                "not the count that was measured (catalog §6.269.40.1)"
            )
        else:
            problems.append(
                f"{rel}:{line_no} claims a {n}-file batch but the section "
                f"resolves to {sorted(sizes) or 'no'} file(s) — a claim that "
                "does not name its population is a census, not a measurement "
                "(catalog §6.269.40.1)"
            )
    return problems


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_every_published_batch_claim_names_a_resolvable_population() -> None:
    """A live claim must resolve to a population of exactly its stated size."""
    catalogue_sections = _sections(_read(CATALOGUE))
    problems: list[str] = []
    for rel in CORPUS:
        problems.extend(find_unresolved_claims(rel, _read(rel), catalogue_sections))
    assert not problems, (
        "these published batch claims do not resolve to a population of the "
        "size they state — a number whose population is not published is a "
        "census, not a measurement (catalog §6.269.40.1): " + "; ".join(problems)
    )


def test_every_batch_claim_site_is_watched_by_a_pattern() -> None:
    """The two routes must agree: a shape no pattern sees is a hole."""
    catalogue_sections = _sections(_read(CATALOGUE))
    holes: list[str] = []
    for rel in CORPUS:
        text = _read(rel)
        spelled = {(line, n) for line, n, _s in _claim_sites(text)}
        swept = {(line, n) for line, n, _s in _swept_sites(text)}
        for line_no, n in sorted(swept - spelled):
            snippet = text.splitlines()[line_no - 1][:160]
            # A site the sweep finds but no pattern matches is invisible to
            # the guard above — the §6.269.37 "a guard that cannot see a
            # shape cannot catch that shape going stale" hole.
            if any(
                d_rel == rel and n == d_size and re.search(d_pat, snippet)
                for d_rel, d_pat, d_size, _r in DECLARED_SKIPS
            ):
                continue
            holes.append(f"{rel}:{line_no} (size {n}) :: {snippet}")
    assert not holes, (
        "these batch-claim sites are found by the semantic sweep but by NO "
        "guard pattern, so the guard cannot see them (catalog §6.269.37): "
        + "; ".join(holes)
    )


def test_every_declared_skip_still_matches_a_live_site() -> None:
    """A declared skip that no longer matches is a dead declaration."""
    dead: list[str] = []
    for rel, pattern, size, _reason in DECLARED_SKIPS:
        text = _read(rel)
        if not re.search(pattern, text):
            dead.append(f"{rel} :: {pattern!r}")
            continue
        # The declared size must still be the size the site claims — a skip
        # that no longer matches its own size is a dead declaration too.
        claimed = {
            n
            for _line, n, _snip in _claim_sites(text)
            if re.search(pattern, text.splitlines()[_line - 1])
        }
        if size not in claimed:
            dead.append(f"{rel} :: {pattern!r} :: declared size {size} not found")
    assert not dead, (
        "these DECLARED_SKIPS no longer match any live site — the declaration "
        "has rotted and must be removed (a dead declaration is worse than a "
        f"live violation): {dead}"
    )


def test_the_guard_is_not_inside_the_population_it_guards() -> None:
    """A guard inside the surface would change the surface's own count."""
    self_name = Path(__file__).resolve().relative_to(REPO).as_posix()
    assert not self_name.startswith("molmetal/tests/test_wfa269_"), (
        f"{self_name} guards the Wf-A269 guard population and must not be part "
        "of it, or the guard measures its own effect (the A127 lesson)."
    )


def test_the_detector_flags_a_synthetic_populationless_claim() -> None:
    """The detector must catch the class it exists for (a self-mutation).

    MEASURED in a synthetic string, not against the live documents: a claim
    whose section names no file and delegates to no section must be reported.
    """
    synthetic = (
        "### §9.9.9 — SYNTHETIC\n\n"
        "The **9-file CPU-only regression batch** measures **1 passed**.\n"
    )
    catalogue_sections = _sections(synthetic)
    problems = find_unresolved_claims(
        "SYNTHETIC.md", synthetic, catalogue_sections
    )
    assert problems, (
        "the detector did NOT flag a synthetic populationless claim — the "
        "guard cannot fail, so it is not a guard (catalog §6.269.17)"
    )
    assert "9-file" in problems[0], problems


def test_the_detector_reports_a_glob_only_population() -> None:
    """A glob is a MOVING population — it must be reported, not counted.

    A claim that names only a glob LOOKS like it names a population, so it
    needs its own diagnosis: the count it denotes is not the count that was
    measured, because the membership changes the moment a file lands.
    """
    synthetic = (
        "### §9.9.8 — SYNTHETIC\n\n"
        "The **3-file CPU-only regression batch** over "
        "`molmetal/tests/test_wfa269_*.py` measures **3 passed**.\n"
    )
    problems = find_unresolved_claims(
        "SYNTHETIC.md", synthetic, _sections(synthetic)
    )
    assert problems, (
        "the detector did NOT flag a claim whose population is named only by "
        "a glob — the guard cannot fail, so it is not a guard "
        "(catalog §6.269.17)"
    )
    assert "MOVING population" in problems[0], problems


def test_a_declared_skip_does_not_exempt_a_different_size() -> None:
    """A skip exempts ONE size — not every claim that shares its line.

    The AGENTS.md gloss carries the preserved "12-file" rendering and the
    new "37-file" claim on the SAME line (both live in one table cell).
    Without the size in the declaration, the 12-file skip would silently
    exempt the 37-file claim as well.
    """
    synthetic = (
        "### §9.9.7 — SYNTHETIC\n\n"
        "The **12-file CPU-only regression batch** and the "
        "**37-file CPU-only regression batch** measure **1 passed**.\n"
    )
    problems = find_unresolved_claims(
        "AGENTS.md", synthetic, _sections(synthetic)
    )
    assert len(problems) == 1, (
        "expected exactly one unresolved claim (the 37-file one; the 12-file "
        f"one is a declared skip), got: {problems}"
    )
    assert "37-file" in problems[0], problems


def test_the_detector_accepts_a_claim_that_delegates_to_a_section() -> None:
    """The escape that the live documents use must work — and only when true."""
    catalogue = (
        "### §6.269.40.1 — POPULATION\n\n"
        "The population is `molmetal/tests/test_wfa269_argparse_prefix_differential.py` "
        "and `molmetal/tests/test_wfa269_bond_candidate_trio_launchable.py`.\n"
    )
    synthetic = (
        "### §9.9.9 — SYNTHETIC\n\n"
        "The **2-file CPU-only regression batch** (population: catalog "
        "§6.269.40.1) measures **2 passed**.\n"
    )
    problems = find_unresolved_claims(
        "SYNTHETIC.md", synthetic, _sections(catalogue)
    )
    assert not problems, (
        "the detector rejected a claim that DOES name its population by "
        f"delegating to a section: {problems}"
    )
