"""Wf-A196 (2026-11-02) -- the X5 reading-COUNT scope repair.

THE DEFECT CLASS.  The repo asserted, AS A SETTLED COUNT, how many competing
readings of the 92.7-94 % frontier band exist: "TWO competing readings", "the
repo holds TWO competing readings", "the two readings below are BOTH on disk",
"each asserts ONE of the TWO competing readings as settled", and so on.  A
settled count of the contest is a claim the repo does not settle on disk:
the repo's own primary-source verification
(``molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json``
-> ``invariants``.``competing_readings``) records a FURTHER reading that also
does not reproduce the quoted numerals, so two on-disk records disagree about
the SIZE of the contest.

THE HONEST REPAIR asserts NO count at all -- not "TWO" and not "THREE".  It
says the NUMBER of competing readings is itself NOT settled on disk, and it
keeps all four band invariants exactly as they were:

  i.   the band is EXTERNAL to Mol-Metal (a FRONTIER composite, not ours);
  ii.  it is NOT one homogeneous quantity;
  iii. it is NOT on our scale (OUR 8.51 % IS a connected-component count on
       the RAW emitted SMILES, and nothing else);
  iv.  the source cells of the band are NOT settled on disk.

WHAT THIS FILE PINS.

  (a) no LIVE count claim about the size of the contest survives in the
      artefacts this wave owns;
  (b) the superseded rendering survives VERBATIM in a DATED position
      (AGENTS.md §5.5 rule #6 -- append-only, dated, never a silent edit);
  (c) the four invariants are STILL PRESENT.

WHAT THIS FILE DOES NOT PIN.  "Under BOTH readings ..." is an INVARIANT-USE,
not a count claim about the size of the contest, and is deliberately left
alone wherever it stands.  A MUST-NOT-ASSERT list, a test constant, a census
dump and a snapshot are not count claims either.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
from typing import Dict, List, Tuple

REPO = pathlib.Path(__file__).resolve().parents[2]

#: The artefacts this wave owns.  A /tmp driver may repoint these at COPIES
#: to prove the tests BITE (a test that passes against a broken artefact is a
#: failure), so every test reads the paths from here and never from a literal.
ARTEFACTS: Dict[str, pathlib.Path] = {
    "producer": REPO / "molmetal" / "scripts" / "wfa156_bond_threshold_frontier.py",
    "decoder": REPO / "molmetal" / "molmetal_lam" / "lam_chem" / "yuelbond_decoder.py",
    "report_wfa156": REPO / "molmetal" / "reports" / "wfa156_bond_threshold_frontier" / "report.md",
    "report_wfa164": REPO / "molmetal" / "reports" / "wfa164_bond_threshold_narrative" / "report.md",
    "report_wfa164_c7": REPO / "molmetal" / "reports" / "wfa164_bond_threshold_narrative" / "c7_pool_check" / "report.md",
}

REPORT_KEYS = ("report_wfa156", "report_wfa164", "report_wfa164_c7")
PY_KEYS = ("producer", "decoder")

#: A COUNT CLAIM about the size of the competing-readings contest.  Matched
#: case-insensitively; ``\s+`` spans the line breaks the clause is wrapped at,
#: and the optional markdown prefix tolerates a block-quoted rendering.
COUNT_CLAIM = re.compile(
    r"(?i)\b(?:two|2)\s+(?:[>#*\-]\s*)?"
    r"(?:competing\s+(?:[>#*\-]\s*)?)?"
    r"(?:provenance\s+(?:[>#*\-]\s*)?)?readings\b"
)

#: The repair's own marker: the clause no longer puts a number on the contest.
REPAIRED_MARKER = "NUMBER of competing readings is itself NOT settled on disk"
REPAIRED_HOLDS = "holds COMPETING readings"

#: The four band invariants, as the artefacts state them.
INVARIANTS = (
    "EXTERNAL",
    "is NOT one homogeneous quantity",
    "is NOT on our scale",
    "source cells are NOT settled on disk",
)

#: The four invariants, as the A186 record states them (the canonical source).
A186_INVARIANTS = (
    "The published 92.7-94 % band is EXTERNAL to Mol-Metal",
    "It is NOT one homogeneous quantity",
    "It is NOT on our scale",
    "The source cells of the band are NOT settled on disk",
)

#: The dated regions in which a count claim is ALLOWED to survive: the
#: superseded rendering (preserved VERBATIM) and the dated records that
#: quote it in order to explain why it was superseded.
DATED_REGIONS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "producer": (
        (
            "#: Wf-A196 (2026-11-02) -- the X5 reading-COUNT scope repair.",
            "#: Wf-A196 (2026-11-02): the REPAIRED canonical reading-safe band",
        ),
        (
            "CANONICAL_READING_SAFE_BAND_CLAUSE_SUPERSEDED_2026_11_01 = ",
            'one connected component.)"""',
        ),
        (
            '"defect": (\n            "X5 -- the emitted text asserted',
            '),\n        "locators"',
        ),
    ),
    "decoder": (
        (
            "#: Wf-A196 (2026-11-02) — the X5 reading-COUNT scope repair.",
            "BAND_PROVENANCE_CLAUSE_SUPERSEDED_2026_11_01 = ",
        ),
        (
            "BAND_PROVENANCE_CLAUSE_SUPERSEDED_2026_11_01 = ",
            'the TWO competing readings as settled."""',
        ),
    ),
}


def _norm_ws(text: str) -> str:
    """Collapse every whitespace run to a single space."""
    return re.sub(r"\s+", " ", text)


def _squash(text: str) -> str:
    """Strip markdown / comment prefixes and ALL whitespace.

    The clause is line-wrapped and, inside a report, block-quoted, so a
    literal substring check would miss it.  In the decoder the block is a
    ``#:`` comment, whose colon would otherwise land in the middle of a
    phrase.  Squashing makes the check about the WORDS and not the layout.
    """
    text = re.sub(r"(?m)^\s*(?:>\s*|#+[:]?\s*|[-*]\s+|\d+\.\s+)", "", text)
    return re.sub(r"\s+", "", text)


def _strip_quote(text: str) -> str:
    """Strip the markdown block-quote prefix the renderer adds, line by line."""
    out = []
    for line in text.splitlines():
        out.append(re.sub(r"^\s*>\s?", "", line))
    return "\n".join(out)


def _read(key: str) -> str:
    return ARTEFACTS[key].read_text(encoding="utf-8")


def _line_no(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _section_0c(text: str) -> str:
    """The dated Wf-A196 section of a report."""
    start = text.index("### 0c.")
    end = text.index("## 1. The frontier rule", start)
    return text[start:end]


def _section_0b(text: str) -> str:
    """The dated Wf-A180 section of a report."""
    start = text.index("### 0b.")
    end = text.index("### 0c.", start)
    return text[start:end]


def _dated_ranges(key: str, text: str) -> List[Tuple[int, int]]:
    """Character ranges of the DATED regions of one artefact."""
    ranges: List[Tuple[int, int]] = []
    for begin, finish in DATED_REGIONS.get(key, ()):
        i = text.find(begin)
        assert i >= 0, f"{key}: dated region start not found: {begin!r}"
        j = text.find(finish, i + len(begin))
        assert j >= 0, f"{key}: dated region end not found: {finish!r}"
        ranges.append((i, j + len(finish)))
    return ranges


# ---------------------------------------------------------------------------
# (a) no LIVE count claim survives
# ---------------------------------------------------------------------------


def test_no_live_count_claim_in_the_owned_python_sources() -> None:
    """Every surviving count claim in the owned .py files is inside a DATED
    region -- the superseded rendering, or the dated record that quotes it in
    order to explain why it was superseded."""
    for key in PY_KEYS:
        text = _read(key)
        ranges = _dated_ranges(key, text)
        for match in COUNT_CLAIM.finditer(text):
            inside = any(lo <= match.start() < hi for lo, hi in ranges)
            assert inside, (
                f"{key}:{_line_no(text, match.start())}: a LIVE count claim "
                f"survives outside every dated region: {match.group(0)!r}"
            )


def test_the_live_clause_sites_assert_no_count() -> None:
    """The live rendering of the band clause carries no count of the contest."""
    producer = _read("producer")
    live_start = producer.index(
        "#: Wf-A196 (2026-11-02): the REPAIRED canonical reading-safe band"
    )
    live_end = producer.index("COMPARABILITY_CAVEAT_TEMPLATE_SUPERSEDED", live_start)
    live_clause = producer[live_start:live_end]
    assert not COUNT_CLAIM.search(live_clause), (
        "the producer's LIVE canonical clause still puts a number on the "
        "size of the contest"
    )
    assert _squash(REPAIRED_HOLDS) in _squash(live_clause)
    assert _squash(REPAIRED_MARKER) in _squash(live_clause)

    decoder = _read("decoder")
    # the three LIVE sites of the decoder's band clause, bounded by markers so
    # the DATED sibling constant is not swept into them
    decoder_sites = (
        (
            "Band-provenance clause corrected 2026-11-02 (Wf-A196)",
            "Superseded rendering 1 --",
        ),
        (
            "#: **The band's provenance is NOT settled on disk** (reading-safe clause,",
            "connected-component count on the RAW emitted SMILES, and nothing else.",
        ),
        (
            "provenance is **NOT settled on disk** (reading-safe clause,",
            "(this wave is CPU-only).",
        ),
    )
    for begin, finish in decoder_sites:
        assert begin in decoder, f"the decoder lost its repaired site: {begin!r}"
        site = decoder[decoder.index(begin): decoder.index(finish, decoder.index(begin))]
        # the clause is line-wrapped and, in the `#:` block, carries a `#:`
        # prefix on every line -- compare on the WORDS, not on the layout
        assert _squash(REPAIRED_HOLDS) in _squash(site), (
            f"the decoder site {begin!r} is not repaired"
        )
        assert _squash(REPAIRED_MARKER) in _squash(site), (
            f"the decoder site {begin!r} does not carry the repaired marker"
        )
        assert not COUNT_CLAIM.search(site), (
            "a LIVE site of the decoder's band clause still puts a number on "
            "the size of the contest"
        )


def test_no_live_count_claim_in_the_owned_reports() -> None:
    """A report may carry the superseded count claim ONLY inside its dated
    Wf-A180 section, and a dated Wf-A196 section must supersede it."""
    for key in REPORT_KEYS:
        text = _read(key)
        matches = list(COUNT_CLAIM.finditer(text))
        assert matches, f"{key}: the superseded rendering disappeared entirely"
        for match in matches:
            line = _line_no(text, match.start())
            assert line < text.index("### 0c."), (
                f"{key}:{line}: a count claim survives OUTSIDE the dated "
                f"supersession: {match.group(0)!r}"
            )
            assert match.start() >= text.index("### 0b."), (
                f"{key}:{line}: a count claim survives outside the dated "
                f"Wf-A180 section: {match.group(0)!r}"
            )
        section_0c = _section_0c(text)
        assert not COUNT_CLAIM.search(section_0c), (
            f"{key}: the dated Wf-A196 section itself asserts a count"
        )
        assert REPAIRED_HOLDS in _norm_ws(section_0c)
        assert REPAIRED_MARKER in _norm_ws(section_0c)


# ---------------------------------------------------------------------------
# (b) the superseded rendering survives VERBATIM, in a dated position
# ---------------------------------------------------------------------------


def test_the_superseded_clause_survives_verbatim_in_the_python_sources() -> None:
    """The Wf-A180 clause survives VERBATIM in a dated sibling constant."""
    producer = _read("producer")
    start = producer.index(
        "CANONICAL_READING_SAFE_BAND_CLAUSE_SUPERSEDED_2026_11_01 = "
    )
    body = producer[start:]
    body = body[body.index('"""\\') + 4:]
    body = body[: body.index('"""')]
    flat = _norm_ws(body)
    assert "holds TWO competing readings" in flat
    assert "Under BOTH readings" in flat

    decoder = _read("decoder")
    start = decoder.index("BAND_PROVENANCE_CLAUSE_SUPERSEDED_2026_11_01 = ")
    body = decoder[start:]
    body = body[body.index('"""\\') + 4:]
    body = body[: body.index('"""')]
    flat = _norm_ws(body)
    assert "holds TWO competing readings" in flat
    assert "Under BOTH readings" in flat


def test_the_superseded_clause_survives_verbatim_in_the_reports() -> None:
    """Each report's dated Wf-A180 section still carries the old clause, and
    the old clause is byte-identical across the three reports."""
    bodies = []
    for key in REPORT_KEYS:
        text = _read(key)
        section = _strip_quote(_section_0b(text))
        flat = _norm_ws(section)
        marker = "the published band's provenance is NOT settled on disk"
        start = flat.index(marker)
        body = flat[start:]
        body = body[: body.index("one connected component.)") + len(
            "one connected component.)"
        )]
        bodies.append(body)
        assert "holds TWO competing readings" in body, (
            f"{key}: the dated Wf-A180 section lost the superseded clause"
        )
    assert bodies[0] == bodies[1], "the two c8-pool reports disagree"
    assert bodies[0] != bodies[2], "the c7-pool report carries the c8 clause"
    assert "529 of 2112" in bodies[0]
    assert "1078 of 12668" in bodies[2]


# ---------------------------------------------------------------------------
# (c) the four invariants are STILL PRESENT
# ---------------------------------------------------------------------------


def test_the_four_invariants_are_still_present_in_every_owned_artefact() -> None:
    """The repair touches the COUNT and nothing else."""
    for key in ARTEFACTS:
        text = _squash(_read(key))
        for invariant in INVARIANTS:
            assert _squash(invariant) in text, (
                f"{key}: the invariant {invariant!r} is GONE after the repair"
            )


def test_the_four_invariants_are_still_present_in_the_reports_sections() -> None:
    """Both dated sections of a report carry the four invariants."""
    for key in REPORT_KEYS:
        text = _read(key)
        for name, section in (
            ("0b", _section_0b(text)),
            ("0c", _section_0c(text)),
        ):
            flat = _squash(section)
            for invariant in INVARIANTS:
                assert _squash(invariant) in flat, (
                    f"{key} section {name}: the invariant {invariant!r} is GONE"
                )


def test_the_a186_record_still_states_the_four_invariants() -> None:
    """The canonical source of the four invariants is untouched and still
    states them, and it is the record the repaired clause cites."""
    a186 = (
        REPO
        / "molmetal"
        / "reports"
        / "wfa186_frontier_primary_source"
        / "primary_source_verification.json"
    )
    if not a186.is_file():  # pragma: no cover - the record is another wave's
        return
    import json

    invariants = json.loads(a186.read_text())["invariants"]
    for key, expected in (
        ("i_the_band_is_external", A186_INVARIANTS[0]),
        ("ii_it_is_not_one_homogeneous_quantity", A186_INVARIANTS[1]),
        ("iii_it_is_not_on_our_scale", A186_INVARIANTS[2]),
        ("iv_the_source_cells_are_not_settled", A186_INVARIANTS[3]),
    ):
        assert expected in invariants[key], f"A186 invariant {key} changed"


# ---------------------------------------------------------------------------
# the producer emits the repaired text (reproducibility)
# ---------------------------------------------------------------------------


def test_the_producer_module_emits_the_repaired_clause() -> None:
    """The live clause constant carries no count; the superseded sibling
    carries the old one; the live caveat carries the repaired clause once
    and the superseded clause ZERO times."""
    path = ARTEFACTS["producer"]
    spec = importlib.util.spec_from_file_location("wfa156_x5_probe", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    live = mod.CANONICAL_READING_SAFE_BAND_CLAUSE
    old = mod.CANONICAL_READING_SAFE_BAND_CLAUSE_SUPERSEDED_2026_11_01
    assert not COUNT_CLAIM.search(live), (
        "the producer's LIVE canonical clause still puts a number on the size "
        "of the contest"
    )
    assert COUNT_CLAIM.search(old), (
        "the producer's DATED sibling lost the superseded count claim"
    )
    assert REPAIRED_HOLDS in live
    assert REPAIRED_MARKER in _norm_ws(live)
    assert mod.COMPARABILITY_CAVEAT_TEMPLATE.count(live) == 1, (
        "the LIVE caveat template must carry the repaired clause exactly once"
    )
    assert mod.COMPARABILITY_CAVEAT_TEMPLATE.count(old) == 0, (
        "the LIVE caveat template still carries the superseded clause"
    )
    assert mod.COMPARABILITY_CAVEAT_TEMPLATE_SUPERSEDED_2026_11_01.count(old) == 0, (
        "the DATED caveat template still carries the superseded clause"
    )

    rendered = mod.render_reading_count_correction_section(
        mod.build_reading_count_correction(
            n_rows_parsed=2112, rows_with_one_fragment=529
        )
    )
    flat = _norm_ws("\n".join(rendered))
    assert REPAIRED_MARKER in flat
    assert not COUNT_CLAIM.search("\n".join(rendered))
