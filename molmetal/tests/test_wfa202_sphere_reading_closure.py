#!/usr/bin/env python3
"""Wf-A202 -- the coordination-sphere artefact's READING CLOSURE.

DEFECT CLASS (one artefact, three leaves):
  the metal-coordination-sphere metric asserted, in LIVE prose leaves, a
  conclusion that the repo's OWN record says is NOT settled on disk.

    D1  `frontier_reference.primary_record` names the published band's source
        cells as if settled, while the SAME artefact's live
        `frontier_reference.what_the_band_is` says those cells are NOT settled.
    D2  `frontier_reference.not_attributed_to_ef_talfm` asserts a SETTLED
        NON-ATTRIBUTION, while the live frontier-matrix record holds EF-TALFM
        as one of the COMPETING provenance readings.
    X5  `frontier_reference.what_the_band_is` asserts a SETTLED COUNT of
        competing readings, while Wf-A196 shipped a canonical clause
        elsewhere in the repo asserting that NO count is settled.

REPAIR SHAPE (AGENTS.md §5.5 rule #6 -- append-only + dated):
  * nothing is deleted and no dated block is rewritten in place;
  * D1 gets a NEW dated sibling key (the measured quotes survive byte-identical);
  * D2 and X5 keep their prior rendering VERBATIM inside a NEW dated sibling
    key, and the LIVE leaf becomes READING-CONDITIONAL -- it states only what
    holds under EVERY competing reading and names those readings by ROBUST
    JSON-PATH locators (never line numbers);
  * the canonical reading-count clause Wf-A196 shipped is copied
    CHARACTER-FOR-CHARACTER (never "THREE", never "TWO");
  * NO numeric leaf moves.

Every guard below BITES: each test fails when the defect is reintroduced.
The bite demonstration (revert -> observe the failure -> restore -> verify the
sha256 returns) is driven by /tmp/wfa202_bite.py, which points this module's
``ARTEFACT`` / ``PRODUCER`` globals at defective COPIES.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = pathlib.Path(__file__).resolve().parents[2]
ARTEFACT = REPO / "metrics/by_metric/wfa155_metal_coordination_sphere.json"
PRODUCER = REPO / "molmetal/scripts/wf_a155_metal_coordination_sphere.py"
WFA196_RECORD = REPO / "molmetal/reports/wfa175_synthesis/_write_md.py"
FRONTIER_MATRIX = (
    REPO / "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json"
)
PRIMARY_SOURCE = (
    REPO
    / "molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json"
)

# the artefact's sha256 before and after the Wf-A202 repair
PRE_REPAIR_SHA256 = (
    "55034ba3e6376bb887c9543f5ea2550208f5294f332c3c76e1e4727ca5eeb94b"
)
POST_REPAIR_SHA256 = (
    "ce38a21e7e42ca770966ac690af89747a2233e86d2e80a550f85fcca085e3bea"
)

# the fingerprint of the PRE-REPAIR numeric leaf map (path -> value, over all
# 42821 numeric leaves).  It is an INDEPENDENT anchor -- computed from the
# pre-repair bytes -- so moving a numeric leaf is detected even though the
# before/after comparison alone is self-consistent.
NUMERIC_LEAF_MAP_SHA256 = (
    "e2306470ef73c684f88804830e3801e19be63bdebb3044c2b7bf94230f47299b"
)

# --------------------------------------------------------------------------
# the canonical Wf-A196 reading-count clause -- copied CHARACTER-FOR-CHARACTER
# --------------------------------------------------------------------------
CANONICAL_COUNT_CLAUSE = "The NUMBER of competing readings is itself NOT settled on disk: this record enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals."

# --------------------------------------------------------------------------
# the PRIOR renderings, preserved VERBATIM inside the dated siblings
# --------------------------------------------------------------------------
PRIOR_WHAT_THE_BAND_IS = "TWO COMPETING PROVENANCE READINGS, NOT ONE HOMOGENEOUS QUANTITY: both are recorded in the frontier post-processing matrix record (`frontier_matrix.json`) -- the first under `band_provenance`, the second under `sources`.`EF-TALFM` -- and they do not even agree on the numerals, so the band's source cells are NOT settled on disk. What holds under BOTH readings: the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale (OUR number is a connected-component count on the RAW emitted SMILES, and nothing else), so it is not comparable to a coordination-sphere profile."
PRIOR_NOT_ATTRIBUTED_TO_EF_TALFM = "EF-TALFM (arXiv:2609.08333v4) must NOT be cited for this band: its own abstract says 'unconditional generation yields 87.9% unique and novel molecules passing sanitization and PoseBusters checks' on PCQM4Mv2 — 87.9 %, not 92.7-94 %."
PRIOR_BREAK_2_FRONTIER = "the frontier publishes the post-processing axis in ONE table -- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW and POST-PROCESSED side by side -- so 'the frontier number' is one of two numbers depending on the arm; the other frontier papers print a single arm. The published 92.7-94 % band's two endpoints are RAW under the `band_provenance` reading (post_processed: false in that reading), while the competing `sources`.`EF-TALFM` reading pairs a raw connectivity cell with a post-processed companion cell -- so whether the band as a whole is RAW is NOT settled on disk; which frontier cells the band names is NOT settled on disk (two competing provenance readings, both in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -- the first under `band_provenance`, the second under `sources`.`EF-TALFM` -- that do not even agree on the numerals), but under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale."
PRIOR_BREAK_3_POST_PROCESSING = "POST-PROCESSING — the frontier publishes the post-processing axis in ONE table -- HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW and POST-PROCESSED side by side -- so 'the frontier number' is one of two numbers depending on the arm; the other frontier papers print a single arm. The published 92.7-94 % band's two endpoints are RAW under the `band_provenance` reading (post_processed: false in that reading), while the competing `sources`.`EF-TALFM` reading pairs a raw connectivity cell with a post-processed companion cell -- so whether the band as a whole is RAW is NOT settled on disk; which frontier cells the band names is NOT settled on disk (two competing provenance readings, both in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -- the first under `band_provenance`, the second under `sources`.`EF-TALFM` -- that do not even agree on the numerals), but under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale. the pool measured here is RAW (no post-processing, no repair), so it is comparable only to a RAW frontier arm, never to a post-processed one."

# the D1 leaves that must survive BYTE-IDENTICAL
PRIMARY_RECORD_QUOTE_LOW = "Val&Uniq (%) | 92.7 ± 0.5 | 93.3 ± 0.0 * | 92.9 ± 0.2"
PRIMARY_RECORD_QUOTE_HIGH = "PB-Valid (%) | 93.1 ± 0.4 | 91.9 ± 0.5 | 94 ± 0.5 *"
PRIMARY_RECORD_VERIFICATION = "RE-VERIFIED LIVE in Wf-A164: both quotes above were read out of Table 1(b) of the fetched primary PDF (sha256 above). The 92.7 % endpoint is NOT in the abstract — the abstract's own numbers are 98.8 / 92.9 / 94.0 — it is the GeoLDM column of Table 1(b)."

# a live leaf that asserts a SETTLED count of competing readings
SETTLED_COUNT_RE = re.compile(
    r"\b(?:two|three|four|2|3|4)\s+competing\b"
    r"|\bBOTH\s+readings\b"
    r"|\bunder\s+BOTH\b"
    r"|\bBOTH\s+competing\b",
    re.IGNORECASE,
)

# a dated record is exempt from the live-leaf scan: its whole purpose is to
# PRESERVE the superseded rendering verbatim (AGENTS.md §5.5 rule #6)
DATED_KEY_RE = re.compile(r"superseded|demotion|stale_by_construction", re.IGNORECASE)

# a line-number locator into a growing artefact is the demoted locator class
# (Wf-A190): the live leaves must use JSON PATHS.  The shape is the repo's own
# LOCATOR_RE shape -- a file (with an extension) followed by `:` and digits.
# An arXiv id such as `arXiv:2603.27113` is an IDENTIFIER, not a locator.
LINE_LOCATOR_RE = re.compile(
    r"[\w./-]+\.(?:json|py|md|yaml|yml|csv|txt|tex|jsonl):\d+"
)

_PRODUCER_CACHE = {}


def _load_producer(path=None):
    """Load the producer module from ``path`` WITHOUT executing ``main()``.

    The module is loaded from its file so a bite harness can point at a
    defective COPY; ``main()`` is guarded by ``__name__ == "__main__"``.
    """
    path = pathlib.Path(path or PRODUCER).resolve()
    if path not in _PRODUCER_CACHE:
        spec = importlib.util.spec_from_file_location(
            "wfa202_sphere_producer", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _PRODUCER_CACHE[path] = module
    return _PRODUCER_CACHE[path]


def _artefact(path=None):
    return json.loads(
        pathlib.Path(path or ARTEFACT).read_text(encoding="utf-8")
    )


def _leaf_map(doc):
    """path -> leaf, mirroring the producer's own leaf walk."""
    out = {}

    def walk(node, prefix):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                walk(value, f"{prefix}[{idx}]")
        else:
            out[prefix] = node

    walk(doc, "")
    return out


def _is_dated(path):
    return any(DATED_KEY_RE.search(seg) for seg in path.split("."))


def _reconstruct_before(doc):
    """Rebuild the PRE-repair artefact from the dated siblings' renderings.

    A byte-exact rebuild proves the dated siblings preserve the superseded
    renderings VERBATIM (AGENTS.md §5.5 rule #6).
    """
    before = json.loads(json.dumps(doc))
    fr = before["frontier_reference"]
    fr["what_the_band_is"] = fr.pop(
        "what_the_band_is_superseded_2026_11_02_wfa202"
    )["rendering"]
    fr["not_attributed_to_ef_talfm"] = fr.pop(
        "not_attributed_to_ef_talfm_superseded_2026_11_02_wfa202"
    )["rendering"]
    fr.pop("primary_record_reading_conditional_2026_11_02_wfa202")
    cs = before["comparability_statement"]
    cs["break_3_post_processing"] = cs.pop(
        "break_3_post_processing_superseded_2026_11_02_wfa202"
    )["rendering"]
    cs["breaks"][2]["frontier"] = cs.pop(
        "breaks_2_frontier_superseded_2026_11_02_wfa202"
    )["rendering"]
    fr_order = (
        "band_as_quoted_in_the_repo",
        "denominator",
        "what_the_band_is",
        "what_the_band_is_superseded_2026_11_01_wfa189",
        "primary_record",
        "not_attributed_to_ef_talfm",
        "repo_provenance",
    )
    cs_order = (
        "summary",
        "n_breaks",
        "breaks",
        "break_1_denominator",
        "break_2_metric",
        "break_3_post_processing",
        "break_3_post_processing_superseded_2026_11_01_wfa183",
        "break_3_post_processing_superseded_2026_11_01_wfa176",
        "break_3_post_processing_superseded_2026_11_01_wfa189",
        "breaks_2_frontier_superseded_2026_11_01_wfa189",
        "break_4_hydrogens",
    )
    before["frontier_reference"] = {k: fr[k] for k in fr_order}
    before["comparability_statement"] = {k: cs[k] for k in cs_order}
    return before


# ==========================================================================
# 1. the canonical clause
# ==========================================================================
def test_the_producer_clause_is_character_for_character_the_wfa196_clause():
    """The producer's clause MUST be Wf-A196's clause, character for character.

    Extracted by AST, NOT by import: importing ``_write_md.py`` re-renders
    ``synthesis.md`` as a module-level side effect.
    """
    tree = ast.parse(WFA196_RECORD.read_text(encoding="utf-8"))
    upstream = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_X5_COUNT_NOT_SETTLED":
                    upstream = ast.literal_eval(node.value)
    assert upstream is not None, "Wf-A196 no longer ships _X5_COUNT_NOT_SETTLED"
    producer_clause = _load_producer().WFA196_COUNT_NOT_SETTLED
    assert producer_clause == upstream
    assert producer_clause == CANONICAL_COUNT_CLAUSE
    assert len(producer_clause) == 334
    # and it must NOT assert a count
    assert not SETTLED_COUNT_RE.search(producer_clause)


# ==========================================================================
# 2. X5 -- the live `what_the_band_is`
# ==========================================================================
def test_the_live_what_the_band_is_asserts_no_settled_reading_count():
    live = _artefact()["frontier_reference"]["what_the_band_is"]
    assert "NOT settled on disk" in live
    assert not SETTLED_COUNT_RE.search(live), (
        "the live leaf still asserts a SETTLED count of competing readings"
    )
    assert CANONICAL_COUNT_CLAUSE in live
    assert "EVERY reading" in live
    assert "NEITHER" not in live


def test_the_live_what_the_band_is_names_the_competing_readings_by_json_path():
    live = _artefact()["frontier_reference"]["what_the_band_is"]
    assert "`band_provenance`" in live
    assert "`sources`.`EF-TALFM`" in live
    assert not LINE_LOCATOR_RE.search(live), (
        "the live leaf must use JSON PATHS, not line numbers"
    )


def test_the_superseded_what_the_band_is_survives_verbatim():
    frontier = _artefact()["frontier_reference"]
    assert "what_the_band_is_superseded_2026_11_02_wfa202" in frontier
    block = frontier["what_the_band_is_superseded_2026_11_02_wfa202"]
    assert block["rendering"] == PRIOR_WHAT_THE_BAND_IS
    assert block["date"] == "2026-11-02"
    assert block["wave"] == "Wf-A202"
    assert block["field"] == "what_the_band_is"
    assert block["supersedes"] == "frontier_reference.what_the_band_is"
    # the prior rendering DID assert a settled count -- that is the defect
    assert SETTLED_COUNT_RE.search(block["rendering"])


# ==========================================================================
# 3. D2 -- the live `not_attributed_to_ef_talfm`
# ==========================================================================
def test_the_live_not_attributed_to_ef_talfm_is_reading_conditional():
    live = _artefact()["frontier_reference"]["not_attributed_to_ef_talfm"]
    assert "ONE OF THE COMPETING PROVENANCE READINGS" in live
    assert "`sources`.`EF-TALFM`" in live
    assert "`retrieval_log`[1].`extracted`" in live
    assert "NOT settled on disk" in live
    assert "EVERY reading" in live
    # it must NOT publish the SETTLED non-attribution any more
    assert "must NOT be cited for this band" not in live
    assert "NOT settled that EF-TALFM must not be cited" in live
    assert not LINE_LOCATOR_RE.search(live)


def test_the_superseded_not_attributed_to_ef_talfm_survives_verbatim():
    frontier = _artefact()["frontier_reference"]
    assert "not_attributed_to_ef_talfm_superseded_2026_11_02_wfa202" in frontier
    block = frontier["not_attributed_to_ef_talfm_superseded_2026_11_02_wfa202"]
    assert block["rendering"] == PRIOR_NOT_ATTRIBUTED_TO_EF_TALFM
    assert block["date"] == "2026-11-02"
    assert block["wave"] == "Wf-A202"
    assert block["field"] == "not_attributed_to_ef_talfm"
    assert block["supersedes"] == "frontier_reference.not_attributed_to_ef_talfm"
    assert "must NOT be cited for this band" in block["rendering"]


# ==========================================================================
# 4. D1 -- the primary_record measurement survives; the annotation is added
# ==========================================================================
def test_the_primary_record_quotes_survive_byte_identical():
    primary = _artefact()["frontier_reference"]["primary_record"]
    assert primary["quote_low_endpoint"] == PRIMARY_RECORD_QUOTE_LOW
    assert primary["quote_high_endpoint"] == PRIMARY_RECORD_QUOTE_HIGH
    assert primary["verification"] == PRIMARY_RECORD_VERIFICATION
    assert primary["arxiv_id"] == "2603.27113"


def test_the_d1_annotation_is_a_new_dated_key_that_qualifies_primary_record():
    frontier = _artefact()["frontier_reference"]
    assert "primary_record_reading_conditional_2026_11_02_wfa202" in frontier
    block = frontier["primary_record_reading_conditional_2026_11_02_wfa202"]
    assert block["date"] == "2026-11-02"
    assert block["wave"] == "Wf-A202"
    assert block["field"] == "primary_record"
    assert block["qualifies"] == "frontier_reference.primary_record"
    annotation = block["annotation"]
    assert "NOT SETTLED ON DISK" in annotation
    assert "READING-CONDITIONAL" in annotation
    assert "`band_provenance`" in annotation
    assert "`sources`.`EF-TALFM`" in annotation
    assert "invariants`.`competing_readings" in annotation
    assert not LINE_LOCATOR_RE.search(annotation)


# ==========================================================================
# 5. the whole-artefact live scan
# ==========================================================================
def test_no_live_leaf_of_the_artefact_asserts_a_settled_reading_count():
    """No LIVE leaf may assert a settled count of competing readings.

    Dated records are exempt: their purpose is to preserve the superseded
    rendering verbatim.
    """
    offenders = []
    for path, value in _leaf_map(_artefact()).items():
        if not isinstance(value, str) or _is_dated(path):
            continue
        if SETTLED_COUNT_RE.search(value):
            offenders.append(path)
    assert not offenders, f"live leaves still assert a settled reading count: {offenders}"


def test_no_live_leaf_of_the_artefact_asserts_a_settled_non_attribution():
    offenders = []
    for path, value in _leaf_map(_artefact()).items():
        if not isinstance(value, str) or _is_dated(path):
            continue
        if "must NOT be cited for this band" in value and "NOT settled" not in value:
            offenders.append(path)
    assert not offenders, f"live leaves still assert a settled non-attribution: {offenders}"


# ==========================================================================
# 6. producer <-> artefact parity
# ==========================================================================
def test_the_producer_regenerates_the_published_frontier_reference():
    block = _load_producer()._frontier_reference_block()
    published = _artefact()["frontier_reference"]
    assert block == published


def test_the_producer_regenerates_the_published_comparability_statement():
    tool = _load_producer()
    doc = _artefact()
    population = doc.get("population", {})
    n_rows = population.get("n_rows_analyzed") or population.get("n_rows")
    if n_rows is None:
        n_rows = 1
    rebuilt = tool.build_comparability_statement(
        doc.get("population", {}).get("pool_name", "pool"),
        int(n_rows),
        n_rows_with_explicit_h=int(
            population.get("n_rows_with_explicit_h") or 0
        ),
    )
    published = doc["comparability_statement"]
    assert rebuilt["breaks"][2]["frontier"] == published["breaks"][2]["frontier"]
    assert (
        rebuilt["break_3_post_processing"]
        == published["break_3_post_processing"]
    )
    assert (
        rebuilt["break_3_post_processing_superseded_2026_11_02_wfa202"]
        == published["break_3_post_processing_superseded_2026_11_02_wfa202"]
    )
    assert (
        rebuilt["breaks_2_frontier_superseded_2026_11_02_wfa202"]
        == published["breaks_2_frontier_superseded_2026_11_02_wfa202"]
    )


def test_the_producer_carries_the_dated_blocks_of_this_wave():
    tool = _load_producer()
    assert (
        tool.WHAT_THE_BAND_IS_SUPERSEDED_2026_11_02_WFA202["rendering"]
        == PRIOR_WHAT_THE_BAND_IS
    )
    assert (
        tool.NOT_ATTRIBUTED_TO_EF_TALFM_SUPERSEDED_2026_11_02_WFA202["rendering"]
        == PRIOR_NOT_ATTRIBUTED_TO_EF_TALFM
    )
    assert (
        tool.BREAKS_2_FRONTIER_SUPERSEDED_2026_11_02_WFA202["rendering"]
        == PRIOR_BREAK_2_FRONTIER
    )
    assert (
        tool.BREAK_3_POST_PROCESSING_SUPERSEDED_2026_11_02_WFA202["rendering"]
        == PRIOR_BREAK_3_POST_PROCESSING
    )


# ==========================================================================
# 7. the numeric-leaf invariant
# ==========================================================================
def test_the_dated_siblings_rebuild_the_pre_repair_artefact_byte_for_byte():
    """The repair is ADDITIVE: the dated siblings rebuild the BEFORE file."""
    after = _artefact()
    frontier = after["frontier_reference"]
    for key in (
        "what_the_band_is_superseded_2026_11_02_wfa202",
        "not_attributed_to_ef_talfm_superseded_2026_11_02_wfa202",
        "primary_record_reading_conditional_2026_11_02_wfa202",
    ):
        assert key in frontier, f"the dated sibling {key} is MISSING"
    before = _reconstruct_before(after)
    text = json.dumps(before, indent=2, ensure_ascii=False, sort_keys=False)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == PRE_REPAIR_SHA256


def test_no_numeric_leaf_moved_by_the_wfa202_repair():
    after = _artefact()
    before = _reconstruct_before(after)
    lb, la = _leaf_map(before), _leaf_map(after)

    def is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    moved = [
        (p, lb[p], la.get(p))
        for p, v in lb.items()
        if is_num(v) and (p not in la or la[p] != v)
    ]
    assert not moved, f"numeric leaves MOVED: {moved[:5]}"
    introduced = [
        p for p, v in la.items() if p not in lb and is_num(v)
    ]
    assert not introduced, f"numeric leaves INTRODUCED: {introduced[:5]}"

    counts_before = collections.Counter(
        "numeric" if is_num(v) else
        "bool" if isinstance(v, bool) else
        "string" if isinstance(v, str) else
        "null" if v is None else "other"
        for v in lb.values()
    )
    counts_after = collections.Counter(
        "numeric" if is_num(v) else
        "bool" if isinstance(v, bool) else
        "string" if isinstance(v, str) else
        "null" if v is None else "other"
        for v in la.values()
    )
    assert counts_before["numeric"] == counts_after["numeric"] == 42821
    assert counts_before["bool"] == counts_after["bool"] == 6346
    assert counts_after["string"] - counts_before["string"] == 49
    assert counts_after["null"] == counts_before["null"] == 333

    # the INDEPENDENT anchor: the numeric leaf map of the published artefact
    # must still carry the PRE-REPAIR fingerprint, so a moved numeric leaf is
    # caught even when the before/after pair is self-consistent.
    def numeric_fingerprint(doc):
        payload = json.dumps(
            {
                p: v
                for p, v in _leaf_map(doc).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    assert numeric_fingerprint(after) == NUMERIC_LEAF_MAP_SHA256
    assert numeric_fingerprint(before) == NUMERIC_LEAF_MAP_SHA256


def test_the_recorded_guard_counters_are_unchanged():
    """The recorded counters are deliberately NOT refreshed.

    The artefact's own dated blocks say so (`guard_counters_note`): the
    recorded `narrative_self_consistency` block is the emitter's pre-guard
    output and is left alone so that NO numeric leaf moves.
    """
    recorded = _artefact()["narrative_self_consistency"]
    assert recorded["n_prose_leaves"] == 52
    assert recorded["n_prose_numbers"] == 58
    assert recorded["n_machine_leaves"] == 64139
    assert recorded["n_machine_numbers"] == 42816
    assert recorded["n_machine_strings"] == 14645


def test_the_artefacts_own_guard_still_passes_on_the_repaired_artefact():
    guard = _load_producer()._narrative_self_consistency(_artefact())
    assert guard["pass"] is True
    assert guard["unbacked_numbers"] == {}
    assert guard["unresolved_narrative_paths"] == []


def test_the_published_artefact_sha256_is_the_repaired_one():
    raw = pathlib.Path(ARTEFACT).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == POST_REPAIR_SHA256
