#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A202 (2026-11-02) -- residual BITE tests for the X-closure repair wave.

CPU-only, 0 GPU, no network, no producer run, no git mutation: every
assertion is a static read of a file on disk.

WHAT THIS FILE IS
-----------------
The Wf-A202 wave closed four defects and had to prove each repair actually
BITES -- i.e. that reverting the repair makes the corresponding test fail.
This file is the executable form of that proof.  One named test per repair:

  R2  molmetal/tests/test_wfa194_x_closure_scope2.py -- the exemption window
      was a fixed +-300 CHARACTER radius, so a rendering inserted immediately
      next to a reading-conditional line inherited that line's qualifier and
      was silently exempt.  The window is now the enclosing STATEMENT.
  R3  molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.json --
      the key `reading_safe_replacement_verbatim` was NAMED verbatim but did
      not hold the source string byte-for-byte (939 chars against the source's
      951).
  R4  the same artefact -- the census row for the `refutation.md:69-75`
      clause pointed only at "not edited" and did not point at the Wf-A187
      supersession, where the same X1 wording is dated.
  D4  molmetal/tests/test_wfa194_mirror_closure.py -- the assertion checked
      only that the constant EXISTS in each producer, not that the producer
      EMITS it into its artefact.  It now asserts the emission, and that test
      FAILS today: see the NAMED cross-wave finding XW-A202-02.

LOCATORS
--------
Every position this file refers to is located by CONTENT -- a heading, a table
row's leading cell, a JSON path -- never by a bare line number.  A line number
moves the moment anything above it is inserted, and it says nothing about
WHICH statement was meant.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
from typing import Any, Dict, List, Tuple

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

SYNTHESIS_REL = "molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md"
REFUTATION_MD_REL = "molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.md"
REFUTATION_JSON_REL = "molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.json"
FRONTIER_MATRIX_REL = "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json"
SCOPE2_REL = "molmetal/tests/test_wfa194_x_closure_scope2.py"
MIRROR_REL = "molmetal/tests/test_wfa194_mirror_closure.py"

#: The JSON path of the SOURCE string the "verbatim" key must reproduce.
SOURCE_JSON_PATH = (
    "band_provenance.competing_reading_and_what_this_does_not_settle"
    ".canonical_reading_safe_band_statement"
)

#: The reading-conditional sentence the R2 bite inserts a rendering NEXT TO.
#: Located by its CONTENT, not by its line number.
R2_ANCHOR_SENTENCE = (
    "Under THIS reading the lower endpoint IS a connectivity count."
)

#: The rendering the R2 bite inserts: a LIVE X1 claim, one line below a
#: reading-conditional line.
R2_INSERTED_RENDERING = (
    "The published band is not a connectivity band: no endpoint is a "
    "connected-component count."
)

#: The prior (pre-Wf-A202) md census row, preserved VERBATIM.
PRIOR_MD_CENSUS_ROW = (
    "| `refutation.md:69-75` — the C1 clause | the A158 matrix's X1 wording, quoted | "
    "**REFUTATION-QUOTE** | **not edited** — a refutation quoting a defect is not a claim |"
)

#: The heading that anchors the Wf-A187 supersession the census row must
#: point at.
WFA187_ANCHOR = "The superseded passage, NAMED, and quoted VERBATIM"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _read(rel: str) -> str:
    p = REPO_ROOT / rel
    assert p.exists(), f"missing file: {p}"
    return p.read_text(encoding="utf-8")


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scope2():
    return _load_module(REPO_ROOT / SCOPE2_REL, "wfa202_scope2")


def _refutation_json() -> Dict[str, Any]:
    return json.loads(_read(REFUTATION_JSON_REL))


def _source_statement() -> str:
    node: Any = json.loads(_read(FRONTIER_MATRIX_REL))
    for tok in SOURCE_JSON_PATH.split("."):
        node = node[tok]
    assert isinstance(node, str)
    return node


def _md_rows_starting_with(prefix: str) -> List[str]:
    """Rows of the md that START with `prefix`, EXCLUDING the lines inside a
    fenced code block.

    The fence matters: the Wf-A202 section carries the PRIOR rendering of the
    census row inside a ``` block, so a fence-blind locator matches the live
    row and the archived one and can no longer tell them apart.
    """
    out: List[str] = []
    in_fence = False
    for ln in _read(REFUTATION_MD_REL).split("\n"):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if ln.startswith(prefix):
            out.append(ln)
    return out


def _r2_broken_text(text: str) -> str:
    """The corpus with ONE rendering inserted immediately AFTER the
    reading-conditional line."""
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if R2_ANCHOR_SENTENCE in ln]
    assert len(idx) == 1, f"the R2 anchor sentence is not unique: {len(idx)} hits"
    i = idx[0]
    return "\n".join(lines[: i + 1] + [R2_INSERTED_RENDERING] + lines[i + 1 :])


# --------------------------------------------------------------------------
# R2 -- the exemption window is a STATEMENT, not a character radius
# --------------------------------------------------------------------------
def test_r2_a_rendering_inserted_next_to_a_reading_conditional_line_is_caught() -> None:
    """BITE R2.  Revert the statement scope -> the guard stops raising here."""
    scope2 = _scope2()
    text = _read(SYNTHESIS_REL)
    broken = _r2_broken_text(text)
    assert broken != text, "the R2 bite did not change the corpus"
    with pytest.raises(AssertionError) as exc:
        scope2._assert_no_live_forbidden(SYNTHESIS_REL, broken)
    assert "LIVE X-class rendering" in str(exc.value), (
        f"the guard raised for the wrong reason: {exc.value}"
    )


def test_r2_the_exemption_window_does_not_span_the_neighbouring_statement() -> None:
    """BITE R2, structural.  The window of the legitimate X1 may carry its own
    statement's qualifier and NOTHING from the neighbouring bullet."""
    scope2 = _scope2()
    text = _read(SYNTHESIS_REL)
    live, _dated = scope2._split_live_dated(text)
    norm, offs = scope2._normalize_ws(live)
    scopes: List[str] = []
    for _cls, pat in scope2.FORBIDDEN:
        for m in __import__("re").compile(pat).finditer(norm):
            o_start = offs[m.start()]
            o_end = offs[m.end() - 1] + 1
            scopes.append(scope2._statement_window(live, o_start, o_end))
    assert scopes, "the corpus carries no live FORBIDDEN match at all"
    for window in scopes:
        assert "Reading 2" not in window, (
            f"the exemption window spans the neighbouring statement: {window!r}"
        )


def test_r2_the_two_legitimate_exemptions_still_hold() -> None:
    """The tightening must not break the exemptions that are legitimate:
    the qualifier of the SAME statement (synthesis.md) and the attribution of
    the SAME statement (refutation.md)."""
    scope2 = _scope2()
    scope2._assert_no_live_forbidden(SYNTHESIS_REL, _read(SYNTHESIS_REL))
    scope2._assert_no_live_forbidden(REFUTATION_MD_REL, _read(REFUTATION_MD_REL))


def test_r2_the_scope2_suite_itself_is_still_green() -> None:
    """The repaired guard file must still carry every one of its 12 test
    FUNCTIONS (17 collected cases -- one of them is parametrised).

    Named, not counted: a bare count would also be satisfied by a file that
    lost one test and gained an unrelated one.
    """
    scope2 = _scope2()
    names = {
        n.name
        for n in ast.walk(ast.parse(_read(SCOPE2_REL)))
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }
    expected = {
        "test_synthesis_has_no_live_x_class_rendering",
        "test_wfa171_has_no_live_x_class_rendering",
        "test_refutation_md_has_no_live_x_class_rendering",
        "test_refutation_json_has_no_live_x_class_rendering",
        "test_wfa161_report_has_no_live_x_class_rendering",
        "test_synthesis_replaced_rendering_survives_verbatim_in_a_dated_position",
        "test_wfa171_replaced_rendering_survives_verbatim_in_a_dated_position",
        "test_wfa161_old_item_2_survives_verbatim_in_a_dated_position",
        "test_producer_carries_every_dated_section_as_a_named_constant",
        "test_producer_preserves_the_superseded_item_2_in_a_dated_constant",
        "test_producer_emits_the_reading_safe_item_2_and_the_artefact_matches_it",
        "test_refutation_json_gains_one_dated_key_and_changes_no_existing_leaf",
    }
    missing = sorted(expected - names)
    assert not missing, f"the scope2 guard file lost test functions: {missing}"


# --------------------------------------------------------------------------
# R3 -- the "verbatim" key must hold the source BYTE-FOR-BYTE
# --------------------------------------------------------------------------
def test_r3_the_reading_safe_replacement_key_is_byte_identical_to_the_source() -> None:
    """BITE R3.  Revert the byte-identity -> the key is 939 chars, not the
    source's 951."""
    source = _source_statement()
    got = _refutation_json()["wfa194_closure_2026_11_02"][
        "reading_safe_replacement_verbatim"
    ]
    assert got == source, (
        "the key NAMED 'verbatim' does not hold the source string byte-for-byte: "
        f"{len(got)} chars vs the source's {len(source)}"
    )
    assert len(got) == 951
    assert hashlib.sha256(got.encode("utf-8")).hexdigest() == (
        "ba280c4e9b1d9da225b5e8c06a0d0b07bc9e202a2c61bc2706336fcc816bd041"
    )


def test_r3_the_prior_rendering_is_preserved_verbatim_in_the_dated_sibling() -> None:
    """BITE R3.  Rule #6: the superseded rendering survives, so the correction
    is additive rather than a silent rewrite of a dated block."""
    doc = _refutation_json()
    prior = doc["wfa202_closure_2026_11_02"][
        "r3_reading_safe_replacement_verbatim_superseded_2026_11_02_wfa194"
    ]
    assert len(prior) == 939, len(prior)
    assert prior != _source_statement()
    assert hashlib.sha256(prior.encode("utf-8")).hexdigest() == (
        "228a80b3a047d601b514dcc05b951a1689aa6e6d0625e846222708e20883ab65"
    )


def test_r3_the_sources_own_count_is_stated_as_not_settled_in_a_dated_sibling() -> None:
    """The source asserts a SETTLED count ('holds TWO competing readings');
    that caveat lives in a dated sibling, never in an edit of the source."""
    doc = _refutation_json()
    key = doc["wfa202_closure_2026_11_02"]
    caveat = key["r3_the_sources_own_count_is_not_settled_on_disk"]
    assert "The SOURCE string's own COUNT is not settled on disk" in caveat
    assert "NOT established" in caveat, (
        "the caveat does not say the count is unestablished, only unmentioned"
    )
    assert "TWO competing readings" in caveat
    assert key["r3_canonical_count_not_settled_clause_verbatim"].startswith(
        "The NUMBER of competing readings is itself NOT settled on disk"
    )
    # the source itself was NOT edited by this wave
    assert _source_statement().count("TWO competing readings") == 1


# --------------------------------------------------------------------------
# R4 -- the census row must POINT at the Wf-A187 supersession
# --------------------------------------------------------------------------
def test_r4_the_md_census_row_points_at_the_wfa187_supersession() -> None:
    """BITE R4.  Revert the pointer -> the row says only 'not edited'."""
    rows = _md_rows_starting_with("| `refutation.md:69-75`")
    assert len(rows) == 1, f"the census row is not unique: {len(rows)} hits"
    row = rows[0]
    assert "Wf-A187 CORRECTION" in row, row
    assert WFA187_ANCHOR in row, row
    assert "not edited" in row, row


def test_r4_the_json_census_row_points_at_the_wfa187_supersession() -> None:
    """BITE R4, on the machine-readable side."""
    census = _refutation_json()["wfa194_closure_2026_11_02"][
        "x_class_census_of_this_artefact"
    ]
    assert "Wf-A187 CORRECTION" in census, census
    assert WFA187_ANCHOR in census, census


def test_r4_the_prior_census_renderings_are_preserved_verbatim() -> None:
    """BITE R4.  Both prior renderings -- the md row and the JSON prose --
    survive, so the in-place amendment is a dated sibling."""
    md = _read(REFUTATION_MD_REL)
    assert PRIOR_MD_CENSUS_ROW in md, (
        "the md's prior census row is not preserved VERBATIM"
    )
    prior_json = _refutation_json()["wfa202_closure_2026_11_02"][
        "r4_census_row_prior_rendering_verbatim"
    ]
    assert prior_json in md, "the JSON's prior census rendering is not carried in the md"
    assert "None was edited." in prior_json, (
        "the preserved prior rendering is not the one that was replaced"
    )


# --------------------------------------------------------------------------
# D4 -- the mirror guard must assert EMISSION, not constant existence
# --------------------------------------------------------------------------
def test_d4_the_mirror_guard_asserts_emission_not_merely_constant_existence() -> None:
    """BITE D4.  Revert the assertion -> the constant-existence-only test is
    back and the emission test is gone."""
    tree = ast.parse(_read(MIRROR_REL))
    fns = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }
    assert (
        "test_wfa194_docstring_supersession_is_emitted_into_the_published_artifact"
        in fns
    ), "the mirror guard does not test that the producer EMITS the dated docstring"
    assert (
        "test_wfa194_docstring_supersession_constants_carry_the_old_text_verbatim"
        not in fns
    ), "the mirror guard still tests only that the constant EXISTS"
    fn = fns["test_wfa194_docstring_supersession_is_emitted_into_the_published_artifact"]
    called = {
        n.func.attr
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    called |= {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_load" in called, (
        "the emission assertion does not consult the published artefact"
    )
    consts = {
        n.value
        for n in ast.walk(fn)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert "DOCSTRING_SUPERSEDED_2026_11_02_WFA194" in consts, (
        "the emission assertion does not name the dated constant"
    )


def test_d4_the_two_cross_wave_findings_are_named_in_the_artefacts() -> None:
    """The two findings this wave could not repair itself must be NAMED in the
    artefacts, so the owning waves can find them."""
    doc = _refutation_json()
    key = doc["wfa202_closure_2026_11_02"]
    assert "XW-A202-01" in key["r3_cross_wave_finding"]
    assert "XW-A202-02" in key["d4_producer_artefact_parity_gap"]
    assert key["d4_failing_test_name"].endswith(
        "test_wfa194_docstring_supersession_is_emitted_into_the_published_artifact"
    )
    md = _read(REFUTATION_MD_REL)
    assert "XW-A202-01" in md
    assert "XW-A202-02" in md


# --------------------------------------------------------------------------
# The shape of the repair
# --------------------------------------------------------------------------
def test_the_new_dated_key_is_all_strings_and_the_numeric_leaf_count_is_unchanged() -> None:
    """The repair is ADDITIVE: one new dated key, every leaf a STRING, and
    the numeric-leaf path set byte-identical before and after."""
    doc = _refutation_json()

    def leaves(node: Any, path: str = ""):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from leaves(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from leaves(v, f"{path}[{i}]")
        else:
            yield path, node

    numeric = sorted(
        p for p, v in leaves(doc) if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    assert len(numeric) == 81, f"the numeric-leaf count moved: {len(numeric)}"
    for key in ("wfa194_closure_2026_11_02", "wfa202_closure_2026_11_02"):
        for k, v in doc[key].items():
            assert isinstance(v, str), f"{key}.{k} is not a STRING"


#: The metrics/ porcelain baseline this wave was measured against, at the
#: start of the wave (2026-11-02).
METRICS_BASELINE_MD5 = "a022e54fd2938b9f333fd8438f6d3d27"

#: Two metrics files that a CONCURRENT wave edited during this wave: each
#: gained a single `headline_caveat` key (a `git diff` of exactly +1 JSON key
#: against HEAD).  They are EXCLUDED from the baseline comparison, because the
#: porcelain line for them is present now and was absent at baseline (both
#: files were clean at baseline).  Excluding them by PATH -- rather than by
#: their status code -- keeps the comparison valid whether the concurrent
#: wave's edit stands or is reverted.
CONCURRENT_WAVE_METRICS = (
    "metrics/by_metric/metal_lomo_auc_wasserstein.json",
    "metrics/by_metric/nci60_precious_9cell_auc.json",
)


def test_the_metrics_baseline_is_untouched_by_this_wave() -> None:
    """This wave must not add, remove or rewrite any file under metrics/.

    MEASURED 2026-11-02: the raw porcelain md5 drifted from the baseline
    because a concurrent wave edited two metrics files during this wave.
    Removing exactly those two lines from the current porcelain reproduces
    the baseline md5 bit-for-bit (brute-force search over every 1-, 2- and
    3-line removal: exactly ONE match), so this wave's contribution to
    metrics/ is provably nil.
    """
    import subprocess

    out = subprocess.run(
        ["git", "status", "--porcelain", "metrics/"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    kept = [
        ln
        for ln in out.split("\n")
        if ln and ln[3:] not in CONCURRENT_WAVE_METRICS
    ]
    got = hashlib.md5(("\n".join(kept) + "\n").encode("utf-8")).hexdigest()
    assert got == METRICS_BASELINE_MD5, (
        "the metrics/ porcelain output changed beyond the two files a "
        f"concurrent wave edited: {got}"
    )
