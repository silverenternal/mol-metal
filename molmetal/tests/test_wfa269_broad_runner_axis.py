#!/usr/bin/env python
# Copyright 2026 The Mol-Metal Authors. Licensed under the Apache-2.0 License.
"""Wf-A269 wave-32 — guards for the BROAD runner-coverage axis adjudication.

WHAT IS UNDER TEST
------------------
`molmetal/scripts/wfa269_broad_runner_axis.py` adjudicates the published
catalogue row (AGENTS.md 6.269.13 "PIN THE AXIS, NOT THE COUNT" + the new class
"A PUBLISHED ROW'S POPULATION SIZE AND ITS COVERED SET MUST COME FROM THE SAME
SURFACE").  Its verdict fields are:

    MIXED_SURFACE                        -- no single (definition x list-source x
                                          content-source) cell reproduces BOTH
                                          the published surface COUNT and the
                                          published covered PAIR -- decided
                                          over the DERIVED candidate family,
                                          not over the declared set (see below)
    MIXED_SURFACE_over_declared_definitions
                                       -- the SAME question over the TWO
                                          declared definitions only.  It is
                                          PRESERVED under its own name
                                          (AGENTS.md 5.5 rule 6) because the
                                          two answers DIFFER.
    SURFACE_UNDER_DETERMINED             -- the family's members do NOT all
                                          give the same covered pair, so the
                                          published SIZE does not determine
                                          the published COVERED SET
    SPLIT_TREE_IRREDUCIBLE               -- no single surface reproduces the
                                          published pair at EVERY tree
    broad1_invariant_across_definitions -- BROAD-1's covered count is the SAME
                                          under every DECLARED surface
                                          definition (NOT over the family)
    damage_is_bounded_to                -- which published row(s) the mixing
                                          actually moves (scoped to the
                                          DECLARED definitions)

WHY THE CANDIDATE SPACE IS WIDER THAN THE DECLARED SET
-------------------------------------------------------
The first version of this file pinned a verdict computed over the TWO
declared definitions in `SURFACE_DEFS`.  That is the class
"A BOOLEAN VERDICT COMPUTED OVER A DECLARED CANDIDATE SET IS NOT A VERDICT
ABOUT THE WORLD": a witness definition OUTSIDE the declared set reproduced
BOTH the published count AND the published pair, so the declared verdict was
an artefact of the candidate set.  The producer now searches a MECHANICALLY
DERIVED family (every tracked file outside the base surface -- the family is
COMPUTED, so a witness cannot hide by being unlisted) and REPORTS the
declared reading under its own name.  M5/M6/M7 pin that.

WHY THE MUTATION PROOFS DOMINATE THIS FILE
------------------------------------------
A check that cannot fail is the defect class of AGENTS.md 6.269.17.  The
adjudication's three verdict fields are therefore driven with SYNTHETIC grids
(M1..M4 below) that are independent of the moving working tree, so the tests
stay meaningful after the population changes.  The live-tree assertions are
RELATIONS (is BROAD-1's count the same across definitions?), never frozen
counts -- freezing a moving number is the "pin that was born false" class of
AGENTS.md 6.269.32.

M8..M10 (wave-32b, 2026-11-07)
------------------------------
M8  -- the published DENOMINATOR is a FREE PARAMETER: a candidate that names
       ZERO population basenames moves the surface SIZE and nothing else, so
       the published SIZE does not determine the published COVERED SET.  The
       negative control (the same null contributors landing on a DIFFERENT
       surface size) is what makes it a check and not a tautology.
M9  -- the split-tree verdict is computed over a COMPLETE grid: all THREE
       trees (HEAD / INDEX / DISK) on both axes.  A missing cell would make
       "reproducible in exactly one of nine" an artefact of an incomplete
       measurement.
M10 -- the MATERIALITY reading is present and DERIVED.  The earlier note
       carried a vacuous conditional (no alternative falls within the
       population's own drift, so the conditional never fires); a vacuous
       conditional in a published note reads as a claim.  The counts that
       decide it are reported, and the boolean is a function of the count.

CPU-ONLY.  The script under test imports only the standard library, so this
file is CI-enforceable without torch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))

import wfa269_broad_runner_axis as AX  # noqa: E402


@pytest.fixture(scope="module")
def live_adjudication() -> dict:
    """The live grids + adjudication, computed ONCE.

    `grid_for` reads every surface file from disk, so recomputing it per test
    turned this file into a 51 s test.  The grids are a SNAPSHOT of the
    working tree; the assertions below are RELATIONS, so a module-scoped
    snapshot is the right granularity.
    """
    pop = AX.population()
    grids = {defn: AX.grid_for(defn, pop) for defn in AX.SURFACE_DEFS}
    return {
        "pop": pop,
        "grids": grids,
        "adjudication": AX.adjudicate_published_row(grids, pop),
    }


def _cell(defn: str, list_mode: str, content_mode: str, n_files: int,
          b1: int, b2: int, pop: int) -> dict:
    """A synthetic grid cell with the same shape `measure()` produces."""
    return {
        "surface_definition": defn,
        "file_list_source": list_mode,
        "content_source": content_mode,
        "surface_n_files": n_files,
        "broad1_covered": b1,
        "broad1_uncovered": pop - b1,
        "broad1_uncovered_pct": round(100.0 * (pop - b1) / pop, 2) if pop else None,
        "broad2_covered": b2,
        "broad2_uncovered": pop - b2,
        "broad2_uncovered_pct": round(100.0 * (pop - b2) / pop, 2) if pop else None,
    }


def _grid(defn: str, n_files: int, cells: dict[tuple[str, str], tuple[int, int]],
          pop: int) -> list[dict]:
    out = []
    for (lm, cm), (b1, b2) in cells.items():
        out.append(_cell(defn, lm, cm, n_files, b1, b2, pop))
    return out


# ---------------------------------------------------------------- M1..M4


def test_m1_broad1_diverging_across_definitions_breaks_the_bound():
    """M1: if BROAD-1's count DIFFERS between the two definitions, the bound
    must go False and the damage must widen to BOTH rows."""
    pop = 100
    grids = {
        "d_a": _grid("d_a", 10, {("head", "disk"): (50, 60)}, pop),
        "d_b": _grid("d_b", 10, {("head", "disk"): (55, 60)}, pop),
    }
    out = AX.adjudicate_published_row(grids, list(range(pop)),
                                      search_witnesses=False,
                                      published_surface_count=10,
                                      published_b1=50, published_b2=60,
                                      published_population=pop)
    assert out["broad1_invariant_across_definitions"] is False
    assert out["damage_is_bounded_to"] == "BROAD-1 AND BROAD-2"
    assert out["broad1_cells_that_differ"], "the diverging cell must be NAMED"


def test_m2_one_definition_reproducing_both_count_and_pair_is_not_mixed():
    """M2: a single definition+cell reproducing BOTH the published count and
    the published pair means the row is COHERENT -- MIXED_SURFACE must be
    False."""
    pop = 100
    grids = {
        "d_a": _grid("d_a", 7, {("head", "disk"): (50, 60)}, pop),
        "d_b": _grid("d_b", 9, {("head", "disk"): (51, 61)}, pop),
    }
    out = AX.adjudicate_published_row(grids, list(range(pop)),
                                      search_witnesses=False,
                                      published_surface_count=7,
                                      published_b1=50, published_b2=60,
                                      published_population=pop)
    assert out["MIXED_SURFACE"] is False
    assert out["surface_count_reproduced_by"] == ["d_a:head-list/disk-content"]
    assert out["covered_set_reproduced_by"] == ["d_a:head-list/disk-content"]


def test_m3_no_cell_reproduces_the_published_count_is_not_mixed():
    """M3: when the published surface COUNT is not reproducible at all, the
    verdict must NOT be vacuously 'mixed' -- there is nothing to mix."""
    pop = 100
    grids = {
        "d_a": _grid("d_a", 3, {("head", "disk"): (50, 60)}, pop),
        "d_b": _grid("d_b", 4, {("head", "disk"): (50, 60)}, pop),
    }
    out = AX.adjudicate_published_row(grids, list(range(pop)),
                                      search_witnesses=False,
                                      published_surface_count=7,
                                      published_b1=50, published_b2=60,
                                      published_population=pop)
    assert out["MIXED_SURFACE"] is False
    assert out["surface_count_reproduced_by"] == []
    assert out["covered_set_reproduced_by"]


def test_m4_broad2_diverging_keeps_broad1_bound():
    """M4: BROAD-2 differing across definitions (the live situation) must keep
    the bound on BROAD-1 -- i.e. the bound is NOT an artefact of 'nothing
    differs'."""
    pop = 100
    grids = {
        "d_a": _grid("d_a", 10, {("head", "disk"): (50, 60)}, pop),
        "d_b": _grid("d_b", 11, {("head", "disk"): (50, 61)}, pop),
    }
    out = AX.adjudicate_published_row(grids, list(range(pop)),
                                      search_witnesses=False,
                                      published_surface_count=10,
                                      published_b1=50, published_b2=60,
                                      published_population=pop)
    assert out["broad1_invariant_across_definitions"] is True
    assert out["damage_is_bounded_to"] == "BROAD-2 ONLY"
    assert out["broad2_cells_that_differ"], "the BROAD-2 divergence must be NAMED"


# ------------------------------------------------------- M5..M7 (widened)


def _witness_space(found: bool, valid: bool = True,
                   members_disagree: bool = True) -> dict:
    """A synthetic widened-search report, with the same shape
    `witness_search()` produces."""
    return {
        "search_valid": valid,
        "family": "one-file-extension of four_path",
        "derivation": "synthetic",
        "scope": "synthetic",
        "n_candidates": {"head": 10, "index": 10},
        "cells": [],
        "members_disagree": members_disagree,
        "witness_found": found,
        "witness_definitions_n": 1 if found else 0,
        "witness_definitions": [],
        "witness_definitions_is_a_sample": False,
        "witness_definitions_sample_size": 0,
        "witness_definitions_note": "synthetic",
        "published_pair_reproducible_cells": ["c1"] if found else [],
        "published_pair_reproducible_in_every_cell_by_one_member": False,
        "published_pair_members_in_all_cells": [],
        "published_pair_members_in_all_cells_n": 0,
        "split_tree_irreducible": found,
    }


def _mixed_declared_grid(pop: int) -> dict:
    """A grid whose DECLARED verdict is 'mixed': one definition has the
    published COUNT, another has the published PAIR, and NO cell has both."""
    return {
        "d_a": _grid("d_a", 10, {("head", "disk"): (50, 61)}, pop),
        "d_b": _grid("d_b", 11, {("head", "disk"): (50, 60)}, pop),
    }


def test_m5_a_witness_outside_the_declared_set_refutes_the_declared_verdict():
    """M5 -- THE REPAIR THIS FILE EXISTS FOR.

    A BOOLEAN VERDICT COMPUTED OVER A DECLARED CANDIDATE SET IS NOT A
    VERDICT ABOUT THE WORLD.  The declared grid says 'mixed'; a witness
    definition OUTSIDE the declared candidate set reproduces BOTH the
    published count AND the published pair.  The verdict must therefore flip
    to False -- and the declared-grid reading must be PRESERVED under its own
    name (AGENTS.md 5.5 rule 6: a correction is additive + dated, never a
    silent deletion)."""
    pop = 100
    out = AX.adjudicate_published_row(
        _mixed_declared_grid(pop), list(range(pop)),
        search_witnesses=False,
        published_surface_count=10, published_b1=50, published_b2=60,
        published_population=pop,
        witness_space=_witness_space(found=True))

    assert out["MIXED_SURFACE_over_declared_definitions"] is True, (
        "the declared-grid reading must still be reported")
    assert out["MIXED_SURFACE"] is False, (
        "a witness OUTSIDE the declared candidate set must refute the verdict")
    assert out["SURFACE_UNDER_DETERMINED"] is True
    assert "NONE MEASURED" in out["published_row_damage"]
    assert out["declared_definition_divergence"]["broad1_invariant"] is True
    assert out["declared_definition_divergence"]["scope"], (
        "the declared-pair divergence must carry an explicit SCOPE")


def test_m6_a_failed_widened_search_is_fail_closed():
    """M6 -- a FAILED widened search must NOT be read as 'no witness exists'.

    That would be a FALSE refutation: the verdict would flip to 'not mixed'
    because the control failed, not because the world says so.  The verdict
    must fall back to the declared-grid reading AND SAY THAT IT DID."""
    pop = 100
    out = AX.adjudicate_published_row(
        _mixed_declared_grid(pop), list(range(pop)),
        search_witnesses=False,
        published_surface_count=10, published_b1=50, published_b2=60,
        published_population=pop,
        witness_space=_witness_space(found=False, valid=False))

    assert out["MIXED_SURFACE"] is True, (
        "a failed search must not silently flip the verdict")
    assert "FAILED" in out["verdict_basis"]
    assert "NOT DETERMINED" in out["published_row_damage"]


def test_m7_the_split_tree_must_be_irreducible_over_the_derived_family(live_adjudication):
    """LIVE -- THE SPLIT TREE IS THE DEFECT THAT SURVIVES THE REFUTATION.

    The published pair is reproducible in at least one (file-list x content)
    cell; if a SINGLE surface member reproduced it in EVERY cell, then the
    split tree would be REDUCIBLE (a different choice of surface would make
    the published number tree-independent) and the defect would be gone.

    This is a RELATION, not a frozen count: if the working tree is ever
    reconciled (so that the committed and working-tree readings agree), this
    test FAILS -- which is correct, because the defect would be fixed."""
    out = live_adjudication["adjudication"]
    wf = out["witness_family"]
    assert wf is not None, "the widened search must have run"
    assert wf["search_valid"] is True, "the widened search must have succeeded"
    assert wf["published_pair_reproducible_cells"], (
        "the published pair must be reproducible SOMEWHERE, else the row is "
        "unreachable -- a different defect")
    assert wf["published_pair_reproducible_in_every_cell_by_one_member"] is False, (
        "the split tree is REDUCIBLE -- a single surface reproduces the "
        "published pair at every tree, so the split-tree defect is gone")
    assert out["SPLIT_TREE_IRREDUCIBLE"] is True


# ---------------------------------------------------------------- live tree


def test_live_broad1_is_invariant_across_the_two_surface_definitions(live_adjudication):
    """LIVE GUARD (a RELATION, not a frozen count): BROAD-1's covered count
    must be the SAME under four_path and five_path.

    If a future edit gives `pyproject.toml` a `tests/<name>.py` PATH-form
    mention, BROAD-1 would move too -- and this test must FAIL, because the
    published '90.24 %' row would then be contaminated as well."""
    grids = live_adjudication["grids"]

    def counts(defn: str, key: str) -> dict[tuple[str, str], int]:
        return {(g["file_list_source"], g["content_source"]): g[key]
                for g in grids[defn]}

    for cell in counts("four_path", "broad1_covered"):
        assert counts("four_path", "broad1_covered")[cell] == \
            counts("five_path", "broad1_covered")[cell], (
            f"BROAD-1 diverged across surface definitions at cell {cell} -- "
            "the published BROAD-1 row would then be contaminated too")


def test_live_adjudication_verdict_fields_are_self_consistent(live_adjudication):
    """The three verdict fields must agree with the evidence they summarise --
    i.e. they are derived, not asserted."""
    out = live_adjudication["adjudication"]

    assert out["broad1_invariant_across_definitions"] == (not out["broad1_cells_that_differ"])
    expected = "BROAD-2 ONLY" if out["broad1_invariant_across_definitions"] \
        else "BROAD-1 AND BROAD-2"
    assert out["damage_is_bounded_to"] == expected

    count_repro = set(out["surface_count_reproduced_by"])
    pair_repro = set(out["covered_set_reproduced_by"])
    if out["MIXED_SURFACE"]:
        assert count_repro and pair_repro
        assert not (count_repro & pair_repro), (
            "MIXED_SURFACE=True but one cell reproduces BOTH the count and "
            "the pair -- the verdict contradicts its own evidence")

    # THE VERDICT MUST BE A FUNCTION OF ITS EVIDENCE, over the WIDENED
    # candidate space (the declared reading survives under its own name).
    wf = out["witness_family"]
    assert wf is not None
    if wf["search_valid"]:
        expected = (out["MIXED_SURFACE_over_declared_definitions"]
                    and not wf["witness_found"])
        assert out["MIXED_SURFACE"] == expected, (
            "the widened verdict contradicts its own evidence")
        assert out["SURFACE_UNDER_DETERMINED"] == wf["members_disagree"]
    else:
        assert out["MIXED_SURFACE"] == out["MIXED_SURFACE_over_declared_definitions"], (
            "a failed widened search must fall back to the declared reading")
        assert "FAILED" in out["verdict_basis"]


def test_live_the_candidate_space_is_wider_than_the_declared_set(live_adjudication):
    """LIVE -- THE REPAIR ITSELF MUST NOT BE UNDONE.

    The whole defect was a verdict computed over a 2-member DECLARED
    candidate set.  If the space ever collapses back to the declared set,
    the verdict is again a statement about the candidate set rather than
    about the world -- so this test pins the SPACE, not the verdict."""
    out = live_adjudication["adjudication"]
    wf = out["witness_family"]
    assert wf is not None
    assert wf["search_valid"] is True
    n_declared = len(AX.SURFACE_DEFS)
    for lm, n in wf["n_candidates"].items():
        assert n > n_declared, (
            f"the derived candidate space for list={lm} is {n}, which is not "
            f"wider than the {n_declared} declared definitions -- the "
            "candidate-space repair has been undone")
    assert wf["derivation"], "the candidate space must state HOW it was derived"
    assert wf["scope"], "the candidate space must state WHAT IT DOES NOT COVER"


def test_live_separating_mentions_are_bare_basename_only(live_adjudication):
    """THE MECHANISM behind the bound: the ONLY file separating four_path from
    five_path is `pyproject.toml`, and its mention must be BARE-BASENAME-ONLY.

    A basename-only mention can never enter BROAD-1 (which requires the
    '...tests/<name>.py' PATH form) -- which is WHY BROAD-1 is invariant.
    If this test fails, the published BROAD-1 row is contaminated too and the
    catalogue entry must be corrected ADDITIVELY (AGENTS.md 5.5 rule 6).

    BOTH CONTENT SOURCES ARE CHECKED (REPAIRED 2026-11-07).  This guard used
    to read the adjudication's `mentions` list, which the producer built from
    the HEAD blob only -- so a WORKING-TREE edit was invisible to it, which is
    the same mixed-tree defect the entry is about, living inside the guard.
    The producer now reports `mentions_by_content_source` with a `head` and a
    `disk` list; the disk list is the one that explains the covered COUNTS
    (which are measured over disk content)."""
    out = live_adjudication["adjudication"]

    sep = out["files_separating_the_definitions"]
    assert sep, "no separating file found -- the two definitions are identical?"
    for entry in sep:
        assert entry["path"] == "pyproject.toml", (
            f"a NEW file separates the definitions: {entry['path']}")
        for source, mlist in entry["mentions_by_content_source"].items():
            for m in mlist:
                assert "tests/" not in m["line_text"], (
                    f"separating mention is PATH-form in the {source!r} content, "
                    f"so BROAD-1 WOULD move: {m['line_text']!r}")


def test_live_separating_mentions_report_both_content_sources(live_adjudication):
    """SHAPE GUARD: the adjudication must report the mechanism for BOTH
    content sources, and must LABEL the file-list source.

    Without this, a future refactor could collapse the two lists back to one
    (the HEAD blob) and this file's other guard would go quiet again -- a
    guard that cannot fail is AGENTS.md 6.269.17's defect class.  The label
    matters for the same reason: the covered counts were measured over a
    DISK-content cell, so the reader has to be able to see which axis each
    half of the record was measured on."""
    out = live_adjudication["adjudication"]
    sep = out["files_separating_the_definitions"]
    assert sep, "no separating file found -- the two definitions are identical?"
    for entry in sep:
        assert entry.get("file_list_source") == "head", (
            "the separating file LIST must be labelled with its source")
        by_src = entry["mentions_by_content_source"]
        assert set(by_src) == {"head", "disk"}, (
            f"the mechanism must be reported per content source, got "
            f"{sorted(by_src)}")
        # The published cell is head-list/DISK-content, so the disk list must
        # not be empty whenever the head list is not.
        if by_src["head"]:
            assert by_src["disk"], (
                "the disk-content mention list is empty while the head list is "
                "not -- the mechanism that explains the covered COUNTS is "
                "missing")


def test_published_row_is_preserved_verbatim_on_disk():
    """AGENTS.md 5.5 rule 6 -- the SUPERSEDED rendering must survive the
    corrective append.  This asserts the published row is still on disk with
    its original numbers (966 / 133 / 176 / 90.24 % / 87.09 %)."""
    cat = REPO / "docs" / "HONEST_NEGATIVE_CATALOG.md"
    text = cat.read_text(encoding="utf-8", errors="replace")
    b1 = [ln for ln in text.splitlines() if "BROAD-1" in ln and "966" in ln]
    b2 = [ln for ln in text.splitlines() if "BROAD-2" in ln and "176" in ln]
    assert b1, "the published BROAD-1 row (966) is gone from the catalogue"
    assert b2, "the published BROAD-2 row (176) is gone from the catalogue"
    assert re.search(r"133", b1[0]), "the published BROAD-1 covered count 133 is gone"
    assert re.search(r"90\.24", b1[0]) or re.search(r"90\.24", text), \
        "the published '90.24 %' rendering is gone"
    assert re.search(r"87\.09", b2[0]) or re.search(r"87\.09", text), \
        "the published '87.09 %' rendering is gone"


# ------------------------------------------------------- M8..M10 (wave-32b)


def _witness_space_with_nulls(null_count: int, base_n_files: int,
                               cell: str = "c1") -> dict:
    """A synthetic widened-search report that carries NULL CONTRIBUTORS.

    A NULL CONTRIBUTOR is a candidate file that names ZERO population
    basenames: adding it moves the SURFACE SIZE and nothing else.  If the
    base surface already reproduces the published covered pair, then
    base u {null contributor} is a surface of size base_n_files + 1 with the
    SAME covered pair -- so the published SIZE does not determine the
    published COVERED SET.
    """
    ws = _witness_space(found=True)
    ws["null_contributor_count"] = {cell: null_count}
    ws["cells"] = [{
        "cell": cell,
        "n_candidates": 10,
        "null_contributors": null_count,
        "base_surface_n_files": base_n_files,
        "base_reproduces_published_pair": True,
        "published_pair_present": True,
        "published_pair_n_members": 1,
    }]
    return ws


def test_m8_a_null_contributor_dials_the_published_denominator():
    """M8 -- THE PUBLISHED DENOMINATOR IS A FREE PARAMETER.

    A candidate file that names ZERO population basenames moves the SURFACE
    SIZE and NOTHING ELSE.  If the base surface already reproduces the
    published covered pair, then base u {null contributor} is a surface whose
    size is the published one and whose covered pair is the published one --
    so the published row's surface is a DENOMINATOR, not a SET.

    THE NEGATIVE CONTROL is what makes this a check and not a tautology: if
    adding the null contributor would NOT land on the published surface size,
    the verdict must be False."""
    pop = 100
    grid = _mixed_declared_grid(pop)
    common = dict(
        search_witnesses=False,
        published_surface_count=10, published_b1=50, published_b2=60,
        published_population=pop,
    )
    out = AX.adjudicate_published_row(
        grid, list(range(pop)),
        witness_space=_witness_space_with_nulls(5, 9), **common)
    assert out["SURFACE_SIZE_IS_FREE_PARAMETER"] is True, (
        "a null contributor that lands the surface on the published size must "
        "leave the published covered pair unchanged -- the denominator is "
        "dialable")
    assert out["null_contributor_count"], (
        "the null contributors must be REPORTED, not merely counted")

    # THE NEGATIVE CONTROL: the same null contributors, but the base surface
    # is one file too big, so adding one lands on 11 != the published 10.
    out2 = AX.adjudicate_published_row(
        grid, list(range(pop)),
        witness_space=_witness_space_with_nulls(5, 10), **common)
    assert out2["SURFACE_SIZE_IS_FREE_PARAMETER"] is False, (
        "the free-parameter verdict must be a FUNCTION of the evidence: a "
        "null contributor that does not land on the published size does not "
        "make the denominator dialable")


def test_m9_the_grid_must_contain_every_tree_cell(live_adjudication):
    """M9 -- THE SPLIT-TREE VERDICT IS COMPUTED OVER A COMPLETE GRID.

    The published pair is reproducible in exactly one of the nine
    (file-list x content) cells, and that cell is a HYBRID -- a file list
    from one tree and file contents from another.  If the grid ever loses a
    cell, the split-tree verdict is computed over an incomplete grid and
    would read as 'reproducible everywhere' or 'nowhere' for the wrong
    reason.  ALL THREE TREES must be present: HEAD (committed), INDEX
    (staged) and DISK (the true working tree)."""
    assert set(AX.LIST_MODES) == {"head", "index", "disk"}, (
        "the file-list axis must carry all three trees")
    assert set(AX.CONTENT_MODES) == {"head", "index", "disk"}, (
        "the content axis must carry all three trees")
    expected = {(lm, cm) for lm in AX.LIST_MODES for cm in AX.CONTENT_MODES}
    for defn, grid in live_adjudication["grids"].items():
        cells = {(g["file_list_source"], g["content_source"]) for g in grid}
        assert cells == expected, (
            f"the grid for {defn} is missing cells {sorted(expected - cells)} -- "
            "the split-tree verdict would be computed over an incomplete grid")


def test_m10_the_materiality_reading_is_present_and_derived(live_adjudication):
    """M10 -- THE MATERIALITY READING MUST BE PRESENT, AND IT MUST BE DERIVED.

    An earlier version of this adjudication carried a CONDITIONAL note --
    "an alternative whose |delta| is at or below the population drift cannot
    be distinguished from the published reading" -- which is VACUOUS when no
    alternative falls within the drift.  A vacuous conditional in a published
    note reads as a claim.  The counts that decide it must be reported so the
    reading can be checked.

    The published percentages must be recomputed over the PUBLISHED
    population (the one the row used), not over the live one -- otherwise the
    comparison is not like-for-like."""
    out = live_adjudication["adjudication"]
    mat = out["materiality"]
    pub_pop = mat["population_used_by_the_published_row"]
    live_pop = mat["population_live"]

    assert isinstance(pub_pop, int) and pub_pop > 0, (
        "the population the published row used must be reported")
    assert isinstance(live_pop, int) and live_pop > 0
    assert pub_pop <= live_pop, (
        "the live population cannot be smaller than the published one")
    assert mat["published_broad1_uncovered_pct"] == pytest.approx(
        100.0 * (pub_pop - out["published_broad1_covered"]) / pub_pop, abs=1e-3), (
        "the published BROAD-1 percentage must be recomputed over the "
        "PUBLISHED population")
    assert mat["published_broad2_uncovered_pct"] == pytest.approx(
        100.0 * (pub_pop - out["published_broad2_covered"]) / pub_pop, abs=1e-3), (
        "the published BROAD-2 percentage must be recomputed over the "
        "PUBLISHED population")

    # THE BOOLEAN MUST BE DERIVED FROM THE COUNT, NEVER ASSERTED.
    n_within = mat["n_alternatives_within_population_drift"]
    assert isinstance(n_within, int), (
        "the count of alternatives within the population drift must be "
        "reported as a number, so the boolean can be checked against it")
    assert mat["materiality_supports_indistinguishability"] == (n_within > 0), (
        "the indistinguishability verdict must be a function of the measured "
        "count, not an independent assertion")

    # AND THE NEAREST ALTERNATIVE MUST AGREE WITH ITS OWN DELTA.
    drift = abs(mat["population_drift_pp_broad2"])
    nearest = mat["nearest_alternative_delta_pp"]
    assert nearest is not None, "the nearest alternative must be reported"
    assert mat["nearest_alternative_is_within_drift"] == (abs(nearest) <= drift), (
        "the nearest-alternative flag must agree with its own delta")


def test_live_the_surface_size_does_not_determine_the_covered_set(live_adjudication):
    """LIVE -- THE SIZE DOES NOT DETERMINE THE SET.

    For the cell that reproduces the published pair, the family's members
    span MANY distinct covered pairs, so a reader who knows only the
    published SIZE cannot recover the published COVERED SET: the published
    row is ONE reading of many.

    The published covered counts sit at the family's MINIMUM, i.e. the
    published row reports the MOST CONSERVATIVE reading of its own
    denominator (the largest uncovered fraction).  That is a RELATION: if the
    family ever moves, the published claim must be corrected additively
    (AGENTS.md 5.5 rule 6)."""
    out = live_adjudication["adjudication"]
    detail = out["surface_under_determination_detail"]
    assert detail, "the under-determination detail must be reported"
    repro = [c for c in detail if c["base_reproduces_published_pair"]]
    assert len(repro) == 1, (
        f"expected exactly one reproducing cell, got {len(repro)}")
    cell = repro[0]
    assert cell["n_distinct_covered_pairs"] > 1, (
        "the family must span more than one covered pair -- else the size "
        "DOES determine the set and this defect is gone")
    assert cell["broad2_covered_min"] == out["published_broad2_covered"], (
        "the published BROAD-2 covered count is no longer the family's "
        "minimum -- the published row is NOT the conservative reading any "
        "more, so the published claim must be corrected")
    assert cell["broad1_covered_min"] == out["published_broad1_covered"], (
        "the published BROAD-1 covered count is no longer the family's "
        "minimum")
    assert cell["broad2_covered_max"] > cell["broad2_covered_min"], (
        "the family's BROAD-2 span must be non-degenerate")


def test_live_the_free_parameter_verdict_has_an_evidencing_cell(live_adjudication):
    """LIVE -- A VERDICT MUST NOT CONTRADICT ITS OWN EVIDENCE.

    `SURFACE_SIZE_IS_FREE_PARAMETER` is a boolean; if it is True there must
    be a cell that actually carries a null contributor, whose base surface
    reproduces the published pair, and whose base size + 1 IS the published
    surface size.  Otherwise the verdict is an assertion (AGENTS.md 6.269.17:
    a check that cannot fail)."""
    out = live_adjudication["adjudication"]
    nulls = out["null_contributor_count"]
    assert nulls, "the null contributors must be reported"
    wf = out["witness_family"]
    evidencing = [
        c for c in wf["cells"]
        if c.get("base_reproduces_published_pair")
        and (c.get("base_surface_n_files", -1) + 1) == out["published_surface_count"]
        and nulls.get(c.get("cell"), 0) > 0
    ]
    assert out["SURFACE_SIZE_IS_FREE_PARAMETER"] == bool(evidencing), (
        "the free-parameter verdict contradicts its own evidence "
        f"(evidencing cells: {len(evidencing)})")

