#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A197 -- one BITING guard per residual defect of the frontier
band-provenance record and its producer.

CPU-only by construction: this file imports no torch and touches no GPU.
Run with::

    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \\
        -m pytest molmetal/tests/test_wfa197_record_residuals.py -v

The six defects an adversarial verifier found (all re-measured by this wave
on 2026-11-02):

===== ============================== ==================================
ID    defect                          guard
===== ============================== ==================================
D1    a wrong numeral and a false     the dated sources census says SIX
      'UNCHANGED' claim
D2    a dated 2026-11-01 block        the live divergence sentence claims
      edited IN PLACE                  NO divergence, and BOTH prior
                                       renderings are preserved VERBATIM
D3    a false 'nothing above this     the FALSE assertion is preserved
      line was re-rendered'            VERBATIM and the corrected assertion
                                       + the re-rendered passages are listed
R1    live cross-artefact line        the live string cites JSON PATH +
      numbers                          sha256, and the superseded string
                                       survives VERBATIM
R2    a date that is not provable     the dated note exists and the key was
      from disk                        NOT renamed
X5    a settled count claim about     the live string asserts NO count --
      the competing readings           'TWO' AND 'THREE' both fail
===== ============================== ==================================

Every guard BITES: each test puts the defect back into a /tmp COPY of the
artefact (never into the published one) and shows the guard FAILS.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytest  # noqa: E402

from molmetal.scripts import wfa158_frontier_postproc_matrix as m  # noqa: E402

ART_DIR = REPO / "molmetal/reports/wfa158_frontier_postproc_matrix"
RECORD_PATH = ART_DIR / "frontier_matrix.json"
MD_PATH = ART_DIR / "frontier_matrix.md"

_RECORD = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
_MD = MD_PATH.read_text(encoding="utf-8")

#: The FALSE 2026-11-02 assertion, as it stands in the published markdown.
#: It must survive VERBATIM: a dated section is not rewritten in place
#: (AGENTS.md §5.5 rule #6).
D3_FALSE_ASSERTION_NEEDLE = (
    "rule #6 nothing above this line was deleted, edited, removed, "
    "re-rendered or re-ordered"
)

#: The pre-Wf-A194 (PRIOR) rendering of the D2 divergence sentence, recovered
#: VERBATIM from the pre-Wf-A194 published bytes of this record.
D2_PRIOR_RENDERING_HEAD = "The producer emits a Wf-A180-SHAPED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _geoldm_observation(record, needle):
    """The live `sources`.`GeoLDM`.`extra_observations` entry carrying `needle`."""
    for obs in record["sources"]["GeoLDM"]["extra_observations"]:
        if needle in obs:
            return obs
    raise AssertionError(f"no GeoLDM observation carries {needle!r}")


def _copy_with_defect(tmp_path, name, mutate):
    """Write a /tmp COPY of the published artefacts, put the defect back.

    The copy is RE-READ FROM DISK, so the guard under test sees the bytes a
    reader would see -- not the in-memory object that was mutated.
    """
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    record = copy.deepcopy(_RECORD)
    md = _MD
    new_md = mutate(record, md)
    if new_md is not None:
        md = new_md
    (d / "frontier_matrix.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (d / "frontier_matrix.md").write_text(md, encoding="utf-8")
    return (
        json.loads((d / "frontier_matrix.json").read_text(encoding="utf-8")),
        (d / "frontier_matrix.md").read_text(encoding="utf-8"),
    )


# ---------------------------------------------------------------------------
# The six guards
# ---------------------------------------------------------------------------


def guard_d1(record, md):
    """D1 -- the sources census says SIX and names the one that changed."""
    lc = record["band_provenance"]["locator_class_and_geo_ldm_source_2026_11_02"]
    corr = lc.get("the_sources_census_correction_2026_11_02")
    assert corr is not None, "D1: the dated sources-census correction is MISSING"
    corrected = corr.get("corrected_sentence", "")
    assert "SIX pre-existing source entries" in corrected, (
        "D1: the corrected census does not say SIX: " + corrected
    )
    assert "`sources`.`HLTF`" in corrected, (
        "D1: the corrected census does not name the one source that changed: "
        + corrected
    )
    assert "other five are byte-identical" in corrected, (
        "D1: the corrected census does not record the five byte-identical "
        "sources: " + corrected
    )
    superseded = corr.get("superseded_sentence_preserved_verbatim", "")
    assert superseded == (
        "The five pre-existing source entries are UNCHANGED; "
        "`sources`.`GeoLDM` was ADDED beside them."
    ), "D1: the superseded sentence is not preserved VERBATIM"
    assert superseded in lc["what_this_repair_does_not_change"], (
        "D1: the paragraph the superseded sentence came from was rewritten in "
        "place -- a dated section is not rewritten in place"
    )
    assert "The SIX pre-existing source entries are UNCHANGED except" in md, (
        "D1: the corrected census is not carried by the human rendering"
    )


def guard_d2(record, md):
    """D2 -- no divergence is claimed, and BOTH prior renderings survive."""
    bp = record["band_provenance"]
    rec = bp["producer_and_record_reconciled_2026_11_01"]
    live = rec.get("divergence_from_the_producer", "")
    assert "Wf-A180-SHAPED" not in live, (
        "D2: the live divergence sentence still claims the producer emits a "
        "Wf-A180-SHAPED object"
    )
    assert "Wf-A177-SHAPED" not in live, (
        "D2: the live divergence sentence still claims the record keeps a "
        "Wf-A177-SHAPED object"
    )
    assert "BYTE-IDENTICAL" in live, (
        "D2: the live divergence sentence does not state that the producer's "
        "object and the record's object are byte-identical"
    )
    superseded = rec.get("divergence_from_the_producer_superseded_2026_11_02")
    assert superseded is not None, (
        "D2: the 2026-11-01 rendering was DELETED instead of being preserved"
    )
    assert superseded.startswith(D2_PRIOR_RENDERING_HEAD), (
        "D2: the preserved 2026-11-01 rendering is not the rendering it "
        "replaced"
    )
    prior = bp.get("producer_and_record_reconciled_2026_11_01_prior_rendering_2026_11_02")
    assert prior is not None, (
        "D2: the PRIOR (pre-Wf-A194) rendering is preserved NOWHERE -- the "
        "exact defect this residual is about"
    )
    prior_text = prior.get("divergence_from_the_producer", "")
    assert prior_text.startswith(D2_PRIOR_RENDERING_HEAD), (
        "D2: the preserved prior rendering is not the pre-Wf-A194 text"
    )
    assert "`current_line_numbers_on_disk`" in prior_text, (
        "D2: the preserved prior rendering is not the pre-Wf-A194 text "
        "(it must still name `current_line_numbers_on_disk`)"
    )
    assert prior_text != superseded, (
        "D2: the two preserved renderings are the SAME string -- one of the "
        "two prior renderings was not preserved"
    )


def guard_d3(record, md):
    """D3 -- the false assertion survives; the corrected one is emitted."""
    rr = record["band_provenance"].get("record_residuals_closed_2026_11_02")
    assert rr is not None, "D3: the dated residual key is MISSING"
    append = rr.get("the_append_assertion", "")
    assert "the DATED SECTIONS above are PRESERVED VERBATIM" in append, (
        "D3: the corrected assertion does not say that the DATED SECTIONS "
        "were preserved verbatim"
    )
    assert "BODY above WAS re-rendered by Wf-A194" in append, (
        "D3: the corrected assertion does not say that the BODY was "
        "re-rendered"
    )
    assert D3_FALSE_ASSERTION_NEEDLE in md, (
        "D3: the FALSE 2026-11-02 assertion was DELETED instead of being "
        "preserved VERBATIM"
    )
    assert "## Record residuals closed (2026-11-02)" in md, (
        "D3: the corrected assertion is not carried by the human rendering"
    )
    for item in rr.get("the_re_rendered_passages", []):
        assert item in md, "D3: a re-rendered passage is not listed in the md: " + item


def guard_r1(record, md):
    """R1 -- a durable locator, and the superseded string survives."""
    src = record["sources"]["GeoLDM"]
    live = _geoldm_observation(record, "THE TWO SOURCE TABLES DISAGREE")
    for stale in (
        "report.md:40",
        "report.md:133",
        "report.md:138",
        "sources.json:170",
        "sources.json:417",
    ):
        assert stale not in live, f"R1: the live string still cites {stale}"
    assert "/sources/4/metrics/0/value" in live, (
        "R1: the live string does not cite the JSON PATH of the GeoLDM cell"
    )
    assert "/sources/11/metrics/2/evidence_quote" in live, (
        "R1: the live string does not cite the JSON PATH of the HLTF cell"
    )
    assert "as measured on 2026-11-02" in live, (
        "R1: the live string does not date its durable locator"
    )
    superseded = src.get("extra_observations_superseded_2026_11_02")
    assert superseded, "R1: the superseded rendering was DELETED"
    assert len(superseded) == 1, "R1: the superseded rendering is not a single string"
    assert "report.md:40" in superseded[0], (
        "R1: the superseded rendering is not preserved VERBATIM"
    )
    assert superseded[0] != live, "R1: the superseded rendering IS the live one"


def guard_r2(record, md):
    """R2 -- the dated note exists, and the key was NOT renamed."""
    bp = record["band_provenance"]
    assert "frontier_band_residuals_closed_2026_11_01" in bp, (
        "R2: the original key was RENAMED -- other artefacts cite it by name"
    )
    note = bp.get(
        "frontier_band_residuals_closed_2026_11_01_identifier_not_provable_2026_11_02"
    )
    assert note is not None, (
        "R2: the dated note recording that the identifier is not provable "
        "from disk is MISSING"
    )
    measurement = note.get("measurement", {})
    assert measurement.get("the_key_is_absent_from_the_pre_wf_a194_published_bytes") is True, (
        "R2: the note does not record that the key is absent from the "
        "pre-Wf-A194 published bytes"
    )
    assert measurement.get("the_key_was_first_published_in_the_write_of") == (
        "2026-11-02 01:23:34"
    ), "R2: the note does not record the first-publication write"
    assert "SAME TWO closures" in note.get("overlaps", ""), (
        "R2: the note does not record the overlap with the dated 2026-11-02 block"
    )


# Wf-A204 (2026-11-02) -- this guard used a FOUR-STRING BLACKLIST, and the
# adversarial verifier of Wf-A197 put the defect back into a /tmp copy in two
# forms that the blacklist did NOT catch, and the guard PASSED both times:
#
#   * "The repo holds two competing readings"   (lower-case numeral word)
#   * "The repo holds 3 competing readings"      (arabic numeral)
#
# A test that passes against a broken artefact is a FAILURE by this file's own
# criterion, so the detector is now a REGEX over any number token -- a digit or
# a spelled-out count -- that settles the size of the contest.  The canonical
# live rendering carries NO number token ("The NUMBER of competing readings is
# itself NOT settled on disk"), so it still passes, and so does the live
# sentence "this record enumerates two of them (readings A and B, below)",
# which enumerates what THIS record holds and does not settle what the repo
# holds.
_SETTLED_COUNT_RE = re.compile(
    r"\b(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
    r"(?:\s+[A-Za-z]+){0,3}\s+competing\s+(?:provenance\s+)?readings?\b",
    re.IGNORECASE,
)


def guard_x5(record, md):
    """X5 -- NO settled count.  'TWO' fails; so does 'THREE'."""
    cr = record["band_provenance"]["competing_reading_and_what_this_does_not_settle"]
    live = cr.get("why_this_key_exists", "")
    settled = _SETTLED_COUNT_RE.search(live)
    assert settled is None, (
        "X5: the live string still asserts a settled count: "
        f"{settled.group(0)!r}"
    )
    assert "The NUMBER of competing readings is itself NOT settled on disk" in live, (
        "X5: the live string does not say that the number of competing "
        "readings is NOT settled on disk"
    )
    superseded = cr.get("why_this_key_exists_superseded_2026_11_02")
    assert superseded is not None, (
        "X5: the superseded count claim was DELETED instead of being preserved"
    )
    assert "The repo holds TWO competing readings" in superseded, (
        "X5: the superseded rendering is not preserved VERBATIM"
    )
    assert superseded != live, "X5: the superseded rendering IS the live one"


# ---------------------------------------------------------------------------
# The tests: each guard passes on the published artefacts and BITES on a
# /tmp copy with the defect put back.
# ---------------------------------------------------------------------------


def test_d1_sources_census_says_six_and_bites(tmp_path):
    guard_d1(_RECORD, _MD)

    def mutate(record, md):
        del record["band_provenance"]["locator_class_and_geo_ldm_source_2026_11_02"][
            "the_sources_census_correction_2026_11_02"
        ]

    broken, broken_md = _copy_with_defect(tmp_path, "d1", mutate)
    with pytest.raises(AssertionError):
        guard_d1(broken, broken_md)


def test_d2_no_divergence_claimed_and_both_prior_renderings_kept(tmp_path):
    guard_d2(_RECORD, _MD)

    def mutate(record, md):
        record["band_provenance"]["producer_and_record_reconciled_2026_11_01"][
            "divergence_from_the_producer"
        ] = record["band_provenance"]["producer_and_record_reconciled_2026_11_01"][
            "divergence_from_the_producer_superseded_2026_11_02"
        ]

    broken, broken_md = _copy_with_defect(tmp_path, "d2", mutate)
    with pytest.raises(AssertionError):
        guard_d2(broken, broken_md)


def test_d3_false_assertion_preserved_and_corrected_one_emitted(tmp_path):
    guard_d3(_RECORD, _MD)

    def mutate(record, md):
        cut = md.index("## Record residuals closed (2026-11-02)")
        return md[:cut]

    broken, broken_md = _copy_with_defect(tmp_path, "d3", mutate)
    with pytest.raises(AssertionError):
        guard_d3(broken, broken_md)


def test_r1_durable_locator_and_superseded_string_kept(tmp_path):
    guard_r1(_RECORD, _MD)

    def mutate(record, md):
        record["sources"]["GeoLDM"]["extra_observations"][1] = (
            record["sources"]["GeoLDM"]["extra_observations_superseded_2026_11_02"][0]
        )

    broken, broken_md = _copy_with_defect(tmp_path, "r1", mutate)
    with pytest.raises(AssertionError):
        guard_r1(broken, broken_md)


def test_r2_dated_note_exists_and_key_not_renamed(tmp_path):
    guard_r2(_RECORD, _MD)

    def mutate(record, md):
        del record["band_provenance"][
            "frontier_band_residuals_closed_2026_11_01_identifier_not_provable_2026_11_02"
        ]

    broken, broken_md = _copy_with_defect(tmp_path, "r2a", mutate)
    with pytest.raises(AssertionError):
        guard_r2(broken, broken_md)

    def rename(record, md):
        bp = record["band_provenance"]
        bp["frontier_band_residuals_closed_2026_11_02"] = bp.pop(
            "frontier_band_residuals_closed_2026_11_01"
        )

    broken, broken_md = _copy_with_defect(tmp_path, "r2b", rename)
    with pytest.raises(AssertionError):
        guard_r2(broken, broken_md)


def test_x5_no_settled_count_and_bites_on_both_two_and_three(tmp_path):
    guard_x5(_RECORD, _MD)

    def put_back_two(record, md):
        record["band_provenance"]["competing_reading_and_what_this_does_not_settle"][
            "why_this_key_exists"
        ] = record["band_provenance"]["competing_reading_and_what_this_does_not_settle"][
            "why_this_key_exists_superseded_2026_11_02"
        ]

    broken, broken_md = _copy_with_defect(tmp_path, "x5_two", put_back_two)
    with pytest.raises(AssertionError):
        guard_x5(broken, broken_md)

    def put_back_three(record, md):
        record["band_provenance"]["competing_reading_and_what_this_does_not_settle"][
            "why_this_key_exists"
        ] = record["band_provenance"]["competing_reading_and_what_this_does_not_settle"][
            "why_this_key_exists_superseded_2026_11_02"
        ].replace("TWO", "THREE")

    broken, broken_md = _copy_with_defect(tmp_path, "x5_three", put_back_three)
    with pytest.raises(AssertionError):
        guard_x5(broken, broken_md)


def test_x5_bites_on_the_case_and_numeral_forms_the_old_blacklist_missed(tmp_path):
    """Wf-A204 (2026-11-02).

    The old guard used a four-string blacklist and PASSED against two broken
    artefacts: a lower-case numeral word and an arabic numeral.  Both must now
    bite, and the LIVE string must still pass.
    """
    # The live string itself carries no settled count -> the guard passes.
    guard_x5(_RECORD, _MD)

    base = _RECORD["band_provenance"]["competing_reading_and_what_this_does_not_settle"]
    superseded = base["why_this_key_exists_superseded_2026_11_02"]

    def put_back(lowercase):
        def mutate(record, md):
            key = (
                "band_provenance"
                "->competing_reading_and_what_this_does_not_settle"
                "->why_this_key_exists"
            )
            record["band_provenance"][
                "competing_reading_and_what_this_does_not_settle"
            ]["why_this_key_exists"] = superseded.replace("TWO", lowercase)

        return mutate

    for label, variant in (
        ("x5_lower_two", "two"),
        ("x5_numeral_three", "3"),
        ("x5_numeral_two", "2"),
    ):
        broken, broken_md = _copy_with_defect(tmp_path, label, put_back(variant))
        with pytest.raises(AssertionError):
            guard_x5(broken, broken_md)

    # And the detector must NOT fire on the live rendering's own wording --
    # "The NUMBER of competing readings ..." carries no number token, and
    # "this record enumerates two of them" enumerates THIS record, it does not
    # settle what the repo holds.
    live = base["why_this_key_exists"]
    assert _SETTLED_COUNT_RE.search(live) is None, (
        "the settled-count detector false-positives on the published live string"
    )
    assert _SETTLED_COUNT_RE.search(
        "this record enumerates two of them (readings A and B, below)"
    ) is None, (
        "the settled-count detector false-positives on an enumeration of what "
        "this record holds"
    )
    assert _SETTLED_COUNT_RE.search("The repo holds competing provenance readings") is None


# ---------------------------------------------------------------------------
# The producer must EMIT the preserved renderings -- a NAMED DATED CONSTANT
# whose bytes differ from the published one would silently drop the history.
# ---------------------------------------------------------------------------


def test_producer_constants_reproduce_the_preserved_renderings():
    bp = _RECORD["band_provenance"]
    assert m.WFA197_D1_SUPERSEDED_SENTENCE_2026_11_02 == (
        bp["locator_class_and_geo_ldm_source_2026_11_02"]
        ["the_sources_census_correction_2026_11_02"]
        ["superseded_sentence_preserved_verbatim"]
    )
    assert m.WFA197_D2_SUPERSEDED_DIVERGENCE_2026_11_02 == (
        bp["producer_and_record_reconciled_2026_11_01"]
        ["divergence_from_the_producer_superseded_2026_11_02"]
    )
    assert m.WFA197_D2_PRIOR_RENDERING_2026_11_02 == (
        bp["producer_and_record_reconciled_2026_11_01_prior_rendering_2026_11_02"]
        ["divergence_from_the_producer"]
    )
    assert m.WFA197_X5_SUPERSEDED_WHY_2026_11_02 == (
        bp["competing_reading_and_what_this_does_not_settle"]
        ["why_this_key_exists_superseded_2026_11_02"]
    )
    assert m.WFA197_R1_SUPERSEDED_OBSERVATION_2026_11_02 == (
        _RECORD["sources"]["GeoLDM"]["extra_observations_superseded_2026_11_02"][0]
    )


def test_producer_reproduces_the_published_artefacts():
    """The producer still emits the published bytes (producer <-> artefact parity)."""
    matrix = m.build_matrix()
    assert m._json_bytes(matrix) == RECORD_PATH.read_text(encoding="utf-8")
    assert m.render_markdown(matrix) == _MD
