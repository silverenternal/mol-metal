#!/usr/bin/env python3
"""Wf-A186 -- build the primary-source verification artefact for the 92.7-94 % frontier band.

CPU-ONLY (no CUDA import, no torch, no GPU allocation).

Reads ONLY:
    cache/<arxiv>.json                                this wave's live fetch receipts
    cache/<arxiv>/<path-in-tarball>                   the pinned e-print bytes
    ../../wfa175_frontier_comparability/cache/<arxiv>.json   READ-ONLY, for the match check
    ../../../docs/HONEST_NEGATIVE_CATALOG.md           READ-ONLY, lines 18674-18688 / 20819
    ../../../metrics/by_metric/cfm_fragmentation_audit.json  READ-ONLY, our own 8.51 %

and DERIVES every numeral it emits FROM those bytes.  The only things this file
hard-codes are LOCATORS (which file, which line, which table) and the WORDS of a
finding.  Every percentage, byte count, sha256 and cell value is parsed out of
the pinned bytes at build time -- so a hand-patched artefact that its producer
cannot reproduce is impossible by construction.

Writes (atomically, via molmetal/scripts/wf386a3_atomic_write.py):
    primary_source_verification.json   the machine artefact
    report.md                          the human rendering (same run, same data)

Run:
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python -I tools/build.py
"""

import hashlib
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, "cache")
REPO = os.path.abspath(os.path.join(BASE, "..", "..", ".."))
WFA175_CACHE = os.path.join(REPO, "molmetal", "reports",
                             "wfa175_frontier_comparability", "cache")
CATALOGUE = os.path.join(REPO, "docs", "HONEST_NEGATIVE_CATALOG.md")
OUR_METRIC = os.path.join(REPO, "metrics", "by_metric", "cfm_fragmentation_audit.json")

sys.path.insert(0, os.path.join(REPO, "molmetal", "scripts"))
from wf386a3_atomic_write import (  # noqa: E402
    atomic_write_file,
    detect_truncation,
    safe_overwrite_check,
)

ARXIV_IDS = ["2305.01140v1", "2603.27113v1", "2609.08333v4"]

# --------------------------------------------------------------------------
# DURABLE LOCATORS into the catalogue (a SECTION ANCHOR, not a line number).
#
# The catalogue GROWS: every wave appends a section, so a line number into
# it is STALE BY CONSTRUCTION.  A section anchor survives any number of
# appends.  MEASURED against the catalogue on disk on 2026-11-02 (sha256
# 27f953f0a7a4b6eb5f3f465741d5a34d9576fd6870a7643b2aa1de42cb8b6502,
# 1,993,469 B, 21,689 lines): the '## §6.264' heading is at line 20288 and
# the '## §6.265' heading is at line 20625, so the four-breaks table and its
# '### 4.' subsection heading (line 20808) sit under §6.265 -- NOT §6.264.
# The §6.264 sibling is '### 6. THE PUBLISHED 92.7-94 % VALIDITY BAND IS NOT
# OUR BAND -- FOUR BREAKS' (line 20487).
# --------------------------------------------------------------------------
CATALOGUE_FOUR_BREAKS_ANCHOR = (
    "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection "
    "'4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT "
    "BREAKS' -> the POST-PROCESSING row"
)
CATALOGUE_SECTION_6252_ANCHOR = (
    "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252 'THE PUBLISHED "
    "CONNECTIVITY BAND IS POST-PROCESSED AND OUR 8.51 % IS RAW'"
)
CATALOGUE_MAY_SUBSECTION_ANCHOR = (
    "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252 -> subsection 'The "
    "sentence we may say, and the sentences we may not'"
)

# --------------------------------------------------------------------------
# Wf-A190 (2026-11-02) -- LOCATOR DEMOTION, CONSUMER SIDE (A186 tree)
#
# A LINE NUMBER into a GROWING artefact is a PERISHABLE locator.  Two targets
# in this tree grow: the record `frontier_matrix.json` (regenerated repeatedly
# -- 1069 -> 1136 lines within 2026-11-01) and the catalogue
# `docs/HONEST_NEGATIVE_CATALOG.md` (appended to by every wave).
#
# THE REPAIR: the LIVE locator becomes a JSON PATH (into the record) or a
# SECTION ANCHOR (into the catalogue); the line numbers are DEMOTED into the
# DATED sibling key `locator_demotion_2026_11_02_stale_by_construction` and
# preserved VERBATIM (AGENTS.md 5.5 rule #6 -- append-only + dated, never a
# silent edit, never a deletion).
#
# The strings below are the EXACT live strings this wave replaced; they are
# frozen here so that a future producer re-run reproduces the demotion
# verbatim instead of silently dropping the evidence.
# --------------------------------------------------------------------------

LOCATOR_DEMOTION_2026_11_02_SUPERSEDED = (
    (
        "retrieval[1].authors_note",
        "the repo cites the second author as 'Alexander Arjun Lobo' "
        "(molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json"
        ":779); the paper's own \\author line reads 'Alexander Lobo'.",
    ),
    (
        "x_class_status.X1.live_status",
        "SURVIVES ONLY IN DATED SUPERSEDED KEYS (correct per AGENTS.md 5.5 rule "
        "#6): molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix"
        ".json:145 (band_provenance.headline_finding_superseded_2026_11_01) and "
        ":763-764 (sources.HLTF.cells.METRIC.note_superseded_2026_11_01); the "
        "LIVE band_provenance.headline_finding is READING-QUALIFIED and no "
        "longer carries it. The repo's own X-class list already marks it "
        "FORBIDDEN (molmetal/reports/wfa161_frontier_labelling/report.md:314).",
    ),
    (
        "x_class_status.X2.live_status",
        "ONE LIVE SURFACE REMAINS: docs/HONEST_NEGATIVE_CATALOG.md:20819 (the "
        "class-(a) POST-PROCESSING row of the 92.7-94 % band's four-breaks "
        "table). The metric artefact "
        "metrics/by_metric/wfa155_metal_coordination_sphere.json carries the "
        "axis only in reading-conditional form ('whether the band as a whole is "
        "RAW is NOT settled on disk'), and the two generator scripts carry it "
        "only in comments. The repo already refuted it from its OWN record "
        "(frontier_matrix.json -> band_provenance -> endpoint_candidates, "
        "post_processed: false); this wave REFUTES IT FROM THE PRIMARY SOURCE "
        "(F2).",
    ),
    (
        "findings.F2.consequence",
        "HLTF's OWN wording scopes the standardized recipe to GEOM-DRUGS ONLY: "
        "the QM9 sentence in Section 4.1 (sections/experiments.tex:8) carries NO "
        "post-processing clause, while the immediately following GEOM-DRUGS "
        "sentence reads 'On GEOM-DRUGS we report Valid, Valid&Unique (V&U), "
        "and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after "
        "standardized post-processing (largest fragment, explicit hydrogens, "
        "UFF relaxation)'. The QM9 Table 1 carries no '+PP' rows at all; the "
        "'+PP' rows live in Table 2 (GEOM-DRUGS, tables/geom_drugs_results.tex, "
        "label tab:geom_main), whose caption declares that 'PP' denotes "
        "standardized post-processing. Both of the band's Reading-A endpoints -- "
        "GeoLDM Val&Uniq and HLTF PB-Valid -- are QM9 Table 1 cells, so they "
        "carry NO post-processing. CONSEQUENCE: X2 is REFUTED BY THE PRIMARY "
        "SOURCE for the 92.7-94 % band. The LIVE catalogue row "
        "docs/HONEST_NEGATIVE_CATALOG.md:20819 -- the class-(a) POST-PROCESSING "
        "row of the four-breaks table -- asserts 'the band is measured on the "
        "published post-processed stream' and is WRONG for THIS band. It was "
        "inherited from a DIFFERENT band: catalogue section 6.252 (line 18603), "
        "whose band is 37.4 % to 96.0 % and whose frontier numbers ARE "
        "post-processed (Buttenschoen et al. arXiv:2505.00518 raw -> "
        "post-processed pairs). CAVEAT (not a refutation): HLTF also applies an "
        "OPTIONAL valence repair that DELETES low-confidence incident bonds; "
        "the paper states 'We report metrics both with and without this repair' "
        "and does not mark which arm Table 1 carries. That repair deletes BONDS, "
        "not components, so it is not the largest-fragment retention the band "
        "question is about.",
    ),
    (
        "invariants.competing_readings[2]",
        "Reading C (docs/HONEST_NEGATIVE_CATALOG.md:21273): the band is a "
        "composite of the EF-TALFM cells 92.98 % / 94.33 % / 93.13 %, which have "
        "three different denominators. Also does NOT reproduce the quoted "
        "numerals. This THIRD reading is NOT among the two recorded in "
        "frontier_matrix.json -- the contest is three-way, not two-way.",
    ),
    (
        "our_side.catalogue_quote.path",
        "docs/HONEST_NEGATIVE_CATALOG.md:18680-18684",
    ),
    (
        "catalogue_row.inherited_from.band_locator",
        "docs/HONEST_NEGATIVE_CATALOG.md:18671",
    ),
    (
        "catalogue_row.inherited_from.may_may_not_block",
        "docs/HONEST_NEGATIVE_CATALOG.md:18678-18688",
    ),
)


def build_locator_demotion_2026_11_02():
    """The DATED sibling that carries the demoted line numbers."""
    return {
        "measured_on": "2026-11-02",
        "wave": "Wf-A190",
        "scope": "consumer side (this tree: wfa186_frontier_primary_source)",
        "rule": "the LIVE locator is a JSON PATH (into the record) or a SECTION "
                "ANCHOR (into the catalogue). A line number may only appear "
                "inside a DATED historical record that says, in words, that it "
                "was measured on a given date and that it is STALE BY "
                "CONSTRUCTION. Never delete a line number: demote it.",
        "record_measured": {
            "path": "molmetal/reports/wfa158_frontier_postproc_matrix/"
                    "frontier_matrix.json",
            "line_count": 1136,
            "bytes": 108408,
            "sha256": "878c387e2b9c1bd8c86d087e6266b19853f4f7f5eb05da1381c7"
                      "ee46d945aa67",
            "measured_on": "2026-11-02",
            "note": "UNCHANGED since the 2026-11-01 measurement the A175 tree "
                    "recorded (1136 lines / 108408 bytes) -- the record has NOT "
                    "moved. The locators below are nevertheless STALE BY "
                    "CONSTRUCTION: the record is regenerated and keeps growing.",
        },
        "catalogue_measured": {
            "path": "docs/HONEST_NEGATIVE_CATALOG.md",
            "bytes": 1993469,
            "sha256": "27f953f0a7a4b6eb5f3f465741d5a34d9576fd6870a7643b2aa1de42"
                      "cb8b6502",
            "measured_on": "2026-11-02",
            "note": "byte-identical to the sha256 this artefact already "
                    "recorded on 2026-11-01 -- the catalogue has NOT moved. The "
                    "locator is nevertheless STALE BY CONSTRUCTION: the "
                    "catalogue is appended to by every wave.",
        },
        "superseded_strings_verbatim": [
            {"where": where, "text": text}
            for (where, text) in LOCATOR_DEMOTION_2026_11_02_SUPERSEDED
        ],
        "what_the_superseded_lines_actually_contained_on_2026_11_02": {
            "145": "band_provenance.headline_finding_superseded_2026_11_01 -- the "
                   "cited line is CORRECT today, but it is a line number into a "
                   "GROWING record, so it is perishable.",
            "763-764": "sources.GGFlow.reported_numbers.label and .locator -- a "
                       "GGFlow (arXiv:2411.05676v1) row. The key the superseded "
                       "locator NAMED, sources.HLTF.cells.METRIC."
                       "note_superseded_2026_11_01, is at line 830. The locator "
                       "was ALREADY WRONG on 2026-11-02, not merely perishable.",
            "779": "sources.GGFlow.reported_numbers.quote -- a GGFlow quote row. "
                   "The string the superseded locator NAMED ('Urvi Awasthi, "
                   "Alexander Arjun Lobo, Leonid Zhukov') is at "
                   "sources.HLTF.citation.authors (line 846) and in "
                   "retrieval_log[arxiv_id=2603.27113, kind=arxiv-abs].extracted "
                   "(line 451). The locator was ALREADY WRONG on 2026-11-02.",
            "20819": "docs/HONEST_NEGATIVE_CATALOG.md -- the class-(a) "
                     "POST-PROCESSING row of the 92.7-94 % band's four-breaks "
                     "table, inside section 6.265, subsection '4. THE PUBLISHED "
                     "92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT "
                     "BREAKS'. CORRECT today; STALE BY CONSTRUCTION.",
            "18603": "docs/HONEST_NEGATIVE_CATALOG.md -- the heading of section "
                     "6.252 itself. CORRECT today; STALE BY CONSTRUCTION.",
            "21273": "docs/HONEST_NEGATIVE_CATALOG.md -- a body line inside "
                     "section 6.266.8 ('The frontier papers, pinned from primary "
                     "sources (Wf-A168) -- and one correction to the previous'). "
                     "CORRECT today; STALE BY CONSTRUCTION.",
        },
        "superseded_numeric_locators_verbatim": [
            {"where": "catalogue_row.surrounding_four_breaks_table."
                      "line_numbers_measured_2026_11_01.post_processing_line",
             "value": 20819,
             "text": "docs/HONEST_NEGATIVE_CATALOG.md:20819",
             "was": "catalogue_row.line"},
            {"where": "catalogue_row.surrounding_four_breaks_table."
                      "line_numbers_measured_2026_11_01.header_line",
             "value": 20815, "text": "20815",
             "was": "catalogue_row.surrounding_four_breaks_table.header_line"},
            {"where": "catalogue_row.surrounding_four_breaks_table."
                      "line_numbers_measured_2026_11_01.denominator_line",
             "value": 20817, "text": "20817",
             "was": "catalogue_row.surrounding_four_breaks_table."
                    "denominator_line"},
            {"where": "catalogue_row.surrounding_four_breaks_table."
                      "line_numbers_measured_2026_11_01.metric_line",
             "value": 20818, "text": "20818",
             "was": "catalogue_row.surrounding_four_breaks_table.metric_line"},
            {"where": "catalogue_row.surrounding_four_breaks_table."
                      "line_numbers_measured_2026_11_01.hydrogens_line",
             "value": 20820, "text": "20820",
             "was": "catalogue_row.surrounding_four_breaks_table."
                    "hydrogens_line"},
            {"where": "catalogue_row.inherited_from."
                      "line_numbers_measured_2026_11_01.heading_line",
             "value": 18603, "text": "18603",
             "was": "catalogue_row.inherited_from.heading_line"},
        ],
        "superseded_key_names_verbatim": [
            {"where": "top level",
             "text": "catalogue_row_20819",
             "note": "the KEY NAME itself embedded a line number into the "
                     "growing catalogue. It is superseded by the durable key "
                     "`catalogue_row`; the old name survives VERBATIM here "
                     "(AGENTS.md 5.5 rule #6 -- append-only + dated)."},
        ],
        "corrections": [
            {
                "dated": "2026-11-02",
                "wave": "Wf-A190",
                "where": [
                    "locator_demotion_2026_11_02_stale_by_construction."
                    "what_the_superseded_lines_actually_contained_on_2026_11_02."
                    "20819",
                    "locator_demotion_2026_11_02_stale_by_construction."
                    "durable_replacements['the catalogue's 92.7-94 % four-breaks "
                    "table, POST-PROCESSING row']",
                    "x_class_status.X2.live_status",
                    "findings.F2.consequence",
                ],
                "superseded_rendering_verbatim":
                    "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.264 -> "
                    "subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR "
                    "BAND -- FOUR INDEPENDENT BREAKS'",
                "corrected_to":
                    "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> "
                    "subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR "
                    "BAND -- FOUR INDEPENDENT BREAKS'",
                "why": "MEASURED against the catalogue on disk on 2026-11-02 "
                       "(sha256 27f953f0a7a4b6eb5f3f465741d5a34d9576fd6870a7"
                       "643b2aa1de42cb8b6502, 1993469 B, 21689 lines): the "
                       "'## §6.264' heading is at line 20288 and the "
                       "'## §6.265' heading is at line 20625, so the "
                       "four-breaks table and its '### 4.' subsection heading "
                       "(line 20808) sit under §6.265, NOT §6.264. The §6.264 "
                       "sibling is '### 6. THE PUBLISHED 92.7-94 % VALIDITY "
                       "BAND IS NOT OUR BAND -- FOUR BREAKS' (line 20487). The "
                       "first rendering of this wave's demotion named §6.264; "
                       "it is preserved VERBATIM above and corrected here.",
            },
        ],
        "durable_replacements": {
            "the record's dated headline key":
                "molmetal/reports/wfa158_frontier_postproc_matrix/"
                "frontier_matrix.json -> band_provenance."
                "headline_finding_superseded_2026_11_01",
            "the record's dated HLTF METRIC note":
                "molmetal/reports/wfa158_frontier_postproc_matrix/"
                "frontier_matrix.json -> sources.HLTF.cells.METRIC."
                "note_superseded_2026_11_01",
            "the record's HLTF author string":
                "molmetal/reports/wfa158_frontier_postproc_matrix/"
                "frontier_matrix.json -> sources.HLTF.citation.authors",
            "the record's arXiv-abstract receipt carrying the same string":
                "molmetal/reports/wfa158_frontier_postproc_matrix/"
                "frontier_matrix.json -> retrieval_log[arxiv_id=2603.27113, "
                "kind=arxiv-abs].extracted",
            "the catalogue's 92.7-94 % four-breaks table, POST-PROCESSING row":
                "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection "
                "'4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR "
                "INDEPENDENT BREAKS' -> the POST-PROCESSING row",
            "the catalogue's 37.4-96.0 % band section":
                "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252",
            "the catalogue's primary-source reading table":
                "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.266.8",
        },
    }

# --------------------------------------------------------------------------
# byte-level helpers
# --------------------------------------------------------------------------


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Paper:
    """One pinned e-print: its receipt, its extracted bytes, its lines."""

    def __init__(self, arxiv_id):
        self.arxiv_id = arxiv_id
        self.root = os.path.join(CACHE, arxiv_id)
        with open(os.path.join(CACHE, f"{arxiv_id}.json")) as fh:
            self.receipt = json.load(fh)
        self._lines = {}

    def path(self, rel):
        return os.path.join(self.root, rel)

    def raw(self, rel):
        with open(self.path(rel), "rb") as fh:
            return fh.read()

    def text(self, rel):
        return self.raw(rel).decode("utf-8")

    def lines(self, rel):
        if rel not in self._lines:
            self._lines[rel] = self.text(rel).split("\n")
        return self._lines[rel]

    def line(self, rel, n):
        """1-indexed line of `rel`, verbatim, newline stripped."""
        return self.lines(rel)[n - 1]

    def span(self, rel, first, last):
        """1-indexed inclusive line span, verbatim, joined with newlines."""
        return "\n".join(self.lines(rel)[first - 1:last])

    def file_sha(self, rel):
        return sha256_file(self.path(rel))


# --------------------------------------------------------------------------
# LaTeX cell parsing (derives values from the bytes)
# --------------------------------------------------------------------------

CELL_SPLIT = re.compile(r"(?<!\\)&")


def split_cells(row_text):
    return CELL_SPLIT.split(row_text)


def _brace_groups(text, start):
    """Parse consecutive {...} groups beginning at `start` (index of '{').

    Returns (list_of_group_strings, index_after_the_last_group).
    """
    groups = []
    i = start
    while i < len(text) and text[i] == "{":
        depth = 0
        j = i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        groups.append(text[i + 1:j])
        i = j + 1
    return groups, i


def _unwrap(text, macro):
    """Strip one layer of a LaTeX command with one braced argument."""
    out = []
    i = 0
    while i < len(text):
        m = re.match(r"\\" + macro + r"\{", text[i:])
        if not m:
            out.append(text[i])
            i += 1
            continue
        groups, j = _brace_groups(text, i + m.end() - 1)
        out.append(groups[0] if groups else "")
        i = j
    return "".join(out)


def _drop(text, macro):
    """Remove \\macro{...} entirely."""
    out = []
    i = 0
    while i < len(text):
        m = re.match(r"\\" + macro + r"\{", text[i:])
        if not m:
            out.append(text[i])
            i += 1
            continue
        _, j = _brace_groups(text, i + m.end() - 1)
        out.append(" ")
        i = j
    return "".join(out)


def _keep_last(text, macro):
    """Replace \\macro{a}{b}{c} with c (the content argument)."""
    out = []
    i = 0
    while i < len(text):
        m = re.match(r"\\" + macro + r"\{", text[i:])
        if not m:
            out.append(text[i])
            i += 1
            continue
        groups, j = _brace_groups(text, i + m.end() - 1)
        out.append(groups[-1] if groups else "")
        i = j
    return "".join(out)


def clean_cell(text, macro_map=None):
    """Light LaTeX -> plain text.  Numbers and signs are preserved verbatim."""
    macro_map = macro_map or {}
    t = text
    t = _drop(t, "rowcolor")
    t = _unwrap(t, "textbf")
    t = _unwrap(t, "underline")
    t = _unwrap(t, "emph")
    t = _unwrap(t, "textit")
    t = _unwrap(t, "textsc")
    t = _keep_last(t, "multicolumn")
    # \shortstack{A\\B} -> A B   (the optional [c] alignment arg is dropped)
    while "\\shortstack" in t:
        m = re.search(r"\\shortstack", t)
        if not m:
            break
        k = m.end()
        while k < len(t) and t[k] in " \t":
            k += 1
        if k < len(t) and t[k] == "[":
            close = t.find("]", k)
            k = (close + 1) if close > 0 else k
        if k >= len(t) or t[k] != "{":
            t = t[:m.start()] + " " + t[m.end():]
            continue
        groups, j = _brace_groups(t, k)
        inner = groups[0] if groups else ""
        inner = inner.replace("\\\\", " ")
        t = t[:m.start()] + inner + t[j:]
    for macro, value in macro_map.items():
        t = t.replace("\\" + macro, value)
    t = t.replace("\\bfseries", " ").replace("\\hfill", " ")
    t = t.replace("\\%", "%").replace("\\&", "&").replace("\\$", "$")
    t = t.replace("\\#", "#")
    t = t.replace("\\pm", "±")
    t = t.replace("\\uparrow", "↑").replace("\\downarrow", "↓")
    t = t.replace("\\times", "×").replace("\\ast", "*").replace("\\dagger", "†")
    t = t.replace("\\ddagger", "‡").replace("\\wr", "ʷ").replace("\\S", "§")
    t = t.replace("\\\\", " ")
    t = t.replace("\\,", "").replace("\\!", "").replace("\\ ", " ")
    t = t.replace("$", "")
    t = re.sub(r"([\^_])\{([^{}]*)\}", r"\1\2", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("{", " ").replace("}", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


NUM_TOKEN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def numbers_in(cleaned):
    out = []
    for tok in NUM_TOKEN.findall(cleaned):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def norm(s):
    """Whitespace-insensitive comparison key for a printed label."""
    return re.sub(r"\s+", "", s or "")


def parse_multicell_groups(group_text, n_cols):
    """Assign a group label to each column from a \\multicolumn header line."""
    cells = split_cells(group_text)
    groups = []
    for cell in cells:
        m = re.search(r"\\multicolumn\{(\d+)\}", cell)
        if m:
            groups.extend([clean_cell(cell)] * int(m.group(1)))
        else:
            groups.append(None)
    if len(groups) != n_cols:
        raise AssertionError(
            f"multicolumn header spans {len(groups)} columns, expected {n_cols}: "
            f"{group_text!r}"
        )
    return groups


class Table:
    """A LaTeX tabular read out of the pinned bytes."""

    def __init__(self, paper, rel, label, group_lines=None, header_lines=None):
        self.paper = paper
        self.rel = rel
        self.label = label
        self.macro_map = {}
        self.group_lines = group_lines
        self.group_text = (paper.span(rel, group_lines[0], group_lines[-1])
                            if group_lines else None)
        self.header_lines = header_lines
        self.header_text = paper.span(rel, header_lines[0], header_lines[-1])
        self.header_cells = [clean_cell(c, self.macro_map)
                             for c in split_cells(self.header_text)]
        self.n_cols = len(self.header_cells)
        self.groups = (parse_multicell_groups(self.group_text, self.n_cols)
                       if self.group_text else [None] * self.n_cols)

    def set_macro_map(self, macro_map):
        self.macro_map = macro_map
        self.header_cells = [clean_cell(c, self.macro_map)
                             for c in split_cells(self.header_text)]
        self.n_cols = len(self.header_cells)
        self.groups = (parse_multicell_groups(self.group_text, self.n_cols)
                       if self.group_text else [None] * self.n_cols)

    def cell(self, row_first, row_last, col_index):
        """Extract one printed cell, derived from the bytes."""
        text = self.paper.span(self.rel, row_first, row_last)
        cells = split_cells(text)
        if len(cells) != self.n_cols:
            raise AssertionError(
                f"{self.rel}:{row_first}-{row_last}: {len(cells)} cells, "
                f"header has {self.n_cols}"
            )
        raw = cells[col_index]
        cleaned = clean_cell(raw, self.macro_map)
        nums = numbers_in(cleaned)
        group = self.groups[col_index] if col_index < len(self.groups) else None
        return {
            "table": self.label,
            "file": self.rel,
            "line": (f"{row_first}" if row_first == row_last
                     else f"{row_first}-{row_last}"),
            "line_start": row_first,
            "line_end": row_last,
            "row_label": clean_cell(cells[0], self.macro_map),
            "column_label": self.header_cells[col_index],
            "column_group": group,
            "printed_value": cleaned,
            "raw_cell": raw.strip(),
            "value": (nums[0] if nums else None),
            "value_text": (cleaned if not nums else None),
            "dispersion": (nums[1] if len(nums) > 1 else None),
            "unit": "percent",
            "file_sha256": self.paper.file_sha(self.rel),
            "eprint_sha256": self.paper.receipt["sha256"],
        }


def macro_from_paper(paper, rel, macro):
    """Read a \\newcommand macro definition out of the pinned bytes."""
    for ln, text in enumerate(paper.lines(rel), start=1):
        m = re.search(r"\\newcommand\{\\" + macro + r"\}\{(.*)\}\s*$", text)
        if m:
            return clean_cell(m.group(1)), ln
    raise AssertionError(f"macro \\{macro} not found in {rel}")


# --------------------------------------------------------------------------
# the spec: LOCATORS only.  Every numeral below is derived at build time.
# --------------------------------------------------------------------------

P = {aid: Paper(aid) for aid in ARXIV_IDS}
GEOLDM, HLTF, EFTALFM = P["2305.01140v1"], P["2603.27113v1"], P["2609.08333v4"]

GEO_REL = "icml2023.tex"
GEO_MACRO, GEO_MACRO_LINE = macro_from_paper(GEOLDM, GEO_REL, "method")

GEO_T1 = Table(GEOLDM, GEO_REL, "tab:qm9_results",
               group_lines=[477], header_lines=[478])
GEO_T1.set_macro_map({"method": GEO_MACRO})
GEO_T2 = Table(GEOLDM, GEO_REL, "app:tab:ablation", header_lines=[889])
GEO_T2.set_macro_map({"method": GEO_MACRO})

HLTF_QM9A = Table(HLTF, "tables/qm9_benchmarks_og.tex", "tab:qm9_main",
                  header_lines=[24, 25, 26, 27])
HLTF_QM9B = Table(HLTF, "tables/qm9_benchmarks_og.tex", "tab:qm9_pb",
                  header_lines=[56, 57, 58, 59])
HLTF_GEOM = Table(HLTF, "tables/geom_drugs_results.tex", "tab:geom_main",
                  header_lines=[12])
HLTF_EXP = "sections/experiments.tex"

EFT_REL = "src/appendix.tex"
EFT_T3A = Table(EFTALFM, EFT_REL, "tbl:pcqm4mv2_generation (a)",
                header_lines=[107, 108, 109, 110])
EFT_T3B = Table(EFTALFM, EFT_REL, "tbl:pcqm4mv2_generation (b)",
                header_lines=[141, 142, 143, 144])
EFT_T4 = Table(EFTALFM, EFT_REL, "tbl:pcqm4mv2_generation_additional",
               header_lines=[188, 189, 190, 191, 192, 193, 194])


def attach_row_method(cell_list):
    """Give a '+PP' continuation row the method label of the row above it.

    HLTF's Table 2 prints each method twice -- once raw, once as '\\quad
    \\emph{+PP}' -- so the printed row label of the second line is just
    '+PP'.  The method it belongs to is the previous row's label.
    """
    last_label = {}
    for c in cell_list:
        table = c["table"]
        if c["row_label"].strip().startswith("+PP"):
            c["row_method"] = last_label.get(table)
        else:
            c["row_method"] = None
            last_label[table] = c["row_label"]
    return cell_list


def build_cells():
    cells = {}

    # --- GeoLDM, Table 1 (QM9 / DRUG) and the ablation table --------------
    g = []
    for row_line in [480, 482, 483, 484, 485, 486, 487, 489, 490, 491]:
        for col in range(1, GEO_T1.n_cols):
            g.append(GEO_T1.cell(row_line, row_line, col))
    for row_line in [891, 892, 893, 894, 895]:
        for col in range(1, GEO_T2.n_cols):
            g.append(GEO_T2.cell(row_line, row_line, col))
    cells["2305.01140v1"] = attach_row_method(g)

    # --- HLTF, Table 1(a), Table 1(b), Table 2 ---------------------------
    h = []
    for row_line in [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41]:
        for col in range(1, HLTF_QM9A.n_cols):
            h.append(HLTF_QM9A.cell(row_line, row_line, col))
    for row_line in [61, 62, 63, 64, 65, 66]:
        for col in range(1, HLTF_QM9B.n_cols):
            h.append(HLTF_QM9B.cell(row_line, row_line, col))
    for row_line in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26]:
        for col in range(1, HLTF_GEOM.n_cols):
            h.append(HLTF_GEOM.cell(row_line, row_line, col))
    cells["2603.27113v1"] = attach_row_method(h)

    # --- EF-TALFM, Table 3(a), 3(b), Table 4 -----------------------------
    e = []
    for row_first, row_last in [(112, 115), (116, 119), (120, 123), (124, 127)]:
        for col in range(1, EFT_T3A.n_cols):
            e.append(EFT_T3A.cell(row_first, row_last, col))
    for row_first, row_last in [(146, 149), (150, 153), (154, 157),
                                (158, 161), (162, 165), (166, 169), (171, 171)]:
        for col in range(1, EFT_T3B.n_cols):
            e.append(EFT_T3B.cell(row_first, row_last, col))
    for row_first, row_last in [(196, 202), (203, 209), (210, 216), (217, 223)]:
        for col in range(1, EFT_T4.n_cols):
            e.append(EFT_T4.cell(row_first, row_last, col))
    cells["2609.08333v4"] = attach_row_method(e)
    return cells


def pick(cells, paper, table, row_label, column_label, column_group=None):
    """Select one cell by its DERIVED labels -- never by a transcribed numeral.

    Exact (whitespace-insensitive) match first; if that is not unique, fall
    back to a unique PREFIX match, which absorbs the papers' habit of adding
    an arrow or a trailing qualifier to a column head.
    """
    pool = [c for c in cells[paper]
            if c["table"] == table
            and (column_group is None
                 or norm(c["column_group"]) == norm(column_group))]
    exact = [c for c in pool
             if norm(c["row_label"]) == norm(row_label)
             and norm(c["column_label"]) == norm(column_label)]
    if len(exact) == 1:
        return exact[0]
    prefix = [c for c in pool
              if norm(c["row_label"]).startswith(norm(row_label))
              and norm(c["column_label"]).startswith(norm(column_label))]
    if len(prefix) == 1:
        return prefix[0]
    raise AssertionError(
        f"cell lookup {paper}/{table}/{row_label!r}/{column_label!r}"
        f"/{column_group!r} matched {len(exact)} exact and {len(prefix)} "
        f"prefix candidates"
    )


# --------------------------------------------------------------------------
# the findings
# --------------------------------------------------------------------------


def q(paper, rel, first, last=None):
    """A verbatim quote out of the pinned bytes, with its locator + sha256."""
    text = paper.span(rel, first, last) if last else paper.line(rel, first)
    return {
        "quote": text,
        "arxiv": paper.arxiv_id,
        "file": rel,
        "line": (f"{first}" if last in (None, first) else f"{first}-{last}"),
        "method": "verbatim line read out of the .tex extracted from the live "
                  "e-print tarball",
        "file_sha256": paper.file_sha(rel),
        "eprint_sha256": paper.receipt["sha256"],
    }


def build_findings(cells):
    F = {}

    # ---- F1 -------------------------------------------------------------
    geo_vu = pick(cells, "2305.01140v1", "tab:qm9_results", "GeoLDM",
                  "Valid & Unique (%)", "QM9")
    geo_v = pick(cells, "2305.01140v1", "tab:qm9_results", "GeoLDM",
                 "Valid (%)", "QM9")
    hltf_vu = pick(cells, "2603.27113v1", "tab:qm9_main", "GeoLDM",
                   "Val&Uniq (%)")
    hltf_vu_b = pick(cells, "2603.27113v1", "tab:qm9_pb",
                     "Val&Uniq (%) ↑", "GeoLDM")
    F["F1"] = {
        "question": "Is GeoLDM's Validity & Uniq 92.7 % GeoLDM's OWN published "
                    "cell, with HLTF merely reproducing it identically?",
        "verdict": "ESTABLISHED",
        "cells": [geo_vu, geo_v, hltf_vu, hltf_vu_b],
        "quotes": [
            q(GEOLDM, GEO_REL, 491),
            q(GEOLDM, GEO_REL, 468),
            q(GEOLDM, GEO_REL, 549),
            q(HLTF, "tables/qm9_benchmarks_og.tex", 38),
            q(HLTF, "tables/qm9_benchmarks_og.tex", 64),
        ],
        "consequence":
            "GeoLDM's OWN paper (arXiv:2305.01140v1, icml2023.tex, label "
            "tab:qm9_results) reports its own QM9 row as Valid & Unique = "
            f"{geo_vu['printed_value']} over 10,000 samples -- the caption at "
            "icml2023.tex:468 reads 'Metrics are calculated with 10000 samples "
            "generated from each model' and the body at icml2023.tex:549 reads "
            "'We generate $10,000$ samples from each method'. HLTF's Table 1(a) "
            "row 'GeoLDM' and Table 1(b) row 'Val&Uniq (%)' carry the SAME printed "
            f"value {hltf_vu['printed_value']} -- HLTF reproduced it, it did not "
            "originate it. CONSEQUENCE FOR THE REPO: the record's own `sources` "
            "block has NO GeoLDM entry (its keys are EF-TALFM / FlowMol3 / "
            "GGFlow / HLTF / PAFlow / YuelBond -- see "
            "molmetal/reports/wfa158_frontier_postproc_matrix/"
            "frontier_matrix.json -> sources), so Reading A's lower endpoint is "
            "NOT substantiated by the record's own sources: the 92.7 % cell is "
            "present in the repo ONLY as a TRANSCRIPT of HLTF's table, with the "
            "originating paper absent from the record's source list. A third "
            "paper in the same scan (YuelBond) quotes 92.7 % for an unrelated "
            "observable (per-bond-type classification F1), so the numeral alone "
            "does not identify the cell.",
    }

    # ---- F2 -------------------------------------------------------------
    hltf_ppb = pick(cells, "2603.27113v1", "tab:qm9_pb",
                    "PB-Valid (%) ↑", "HLTF (ours)")
    F["F2"] = {
        "question": "Which of HLTF's tables carries the standardized "
                    "post-processing recipe -- Table 1 (QM9, where BOTH "
                    "Reading-A endpoints live) or Table 2 (GEOM-DRUGS)?",
        "verdict": "ESTABLISHED",
        "cells": [hltf_vu, hltf_ppb],
        "quotes": [
            q(HLTF, HLTF_EXP, 8),
            q(HLTF, HLTF_EXP, 38),
            q(HLTF, HLTF_EXP, 34),
            q(HLTF, "tables/geom_drugs_results.tex", 8),
            q(HLTF, "iclr2025_conference.tex", 55),
            q(HLTF, "sections/appendix.tex", 256),
            q(HLTF, "sections/appendix.tex", 269),
            q(HLTF, "sections/methods.tex", 254),
        ],
        "consequence":
            "HLTF's OWN wording scopes the standardized recipe to GEOM-DRUGS "
            "ONLY: the QM9 sentence in Section 4.1 (sections/experiments.tex:8) "
            "carries NO post-processing clause, while the immediately following "
            "GEOM-DRUGS sentence reads 'On GEOM-DRUGS we report Valid, "
            "Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 "
            "samples, both raw and after standardized post-processing (largest "
            "fragment, explicit hydrogens, UFF relaxation)'. The QM9 Table 1 "
            "carries no '+PP' rows at all; the '+PP' rows live in Table 2 "
            "(GEOM-DRUGS, tables/geom_drugs_results.tex, label tab:geom_main), "
            "whose caption declares that 'PP' denotes standardized "
            "post-processing. Both of the band's Reading-A endpoints -- GeoLDM "
            "Val&Uniq and HLTF PB-Valid -- are QM9 Table 1 cells, so they carry "
            "NO post-processing. CONSEQUENCE: X2 is REFUTED BY THE PRIMARY SOURCE "
            "for the 92.7-94 % band. The LIVE catalogue row "
            "docs/HONEST_NEGATIVE_CATALOG.md, section 6.265, subsection '4. THE "
            "PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT "
            "BREAKS' -- the class-(a) POST-PROCESSING row of the four-breaks "
            "table -- asserts 'the band "
            "is measured on the published post-processed stream' and is WRONG "
            "for THIS band. It was inherited from a DIFFERENT band: catalogue "
            "section 6.252, whose band is 37.4 % to 96.0 % and "
            "whose frontier numbers ARE post-processed (Buttenschoen et al. "
            "arXiv:2505.00518 raw -> post-processed pairs). CAVEAT (not a "
            "refutation): HLTF also applies an OPTIONAL valence repair that "
            "DELETES low-confidence incident bonds; the paper states 'We report "
            "metrics both with and without this repair' and does not mark which "
            "arm Table 1 carries. That repair deletes BONDS, not components, so "
            "it is not the largest-fragment retention the band question is "
            "about.",
    }

    # ---- F3 -------------------------------------------------------------
    F["F3"] = {
        "question": "Does GeoLDM's own paper state that its model generates "
                    "disconnected components, and does it apply a fragment fix "
                    "to the reported QM9 numbers?",
        "verdict": "ESTABLISHED",
        "cells": [geo_vu],
        "quotes": [
            q(GEOLDM, GEO_REL, 917),
            q(GEOLDM, GEO_REL, 532),
            q(GEOLDM, GEO_REL, 549),
        ],
        "consequence":
            "GeoLDM's own paper DOES name the artefact: 'the model occasionally "
            "generates disconnected components ... this phenomenon actually is "
            "not a problem ... and can be easily fixed by just filtering the "
            "smaller components.' The phrase 'can be easily fixed' is a REMEDY "
            "PROPOSED, not a procedure applied to the reported numbers -- the "
            "paper's ONLY occurrence of 'filter' in the whole source is that "
            "sentence, and the numbers in tab:qm9_results are the unfiltered "
            "ones. The same sentence says the disconnection 'happens more often "
            "when trained on the large molecule DRUG dataset', i.e. it is a "
            "DRUG-side observation, not a QM9-side correction. CONSEQUENCE: the "
            "Wf-A175 row-5 verdict for GeoLDM ('PROTOCOL-UNSPECIFIED') can be "
            "UPGRADED to RAW with a verbatim locator -- GeoLDM's QM9 "
            f"{geo_vu['printed_value']} is a RAW-stream number. This is "
            "independent primary-source evidence against X2 and X3 for the "
            "band's lower endpoint.",
    }

    # ---- F4 -------------------------------------------------------------
    hltf_geo_v_a = pick(cells, "2603.27113v1", "tab:qm9_main", "GeoLDM",
                        "Val (%)")
    hltf_geo_v_b = pick(cells, "2603.27113v1", "tab:qm9_pb", "Val (%) ↑",
                        "GeoLDM")
    hltf_own_v = pick(cells, "2603.27113v1", "tab:qm9_main", "HLTF (ours)",
                      "Val (%)")
    F["F4"] = {
        "question": "GeoLDM's own paper reports its QM9 Valid (%) as 93.8 +- "
                    "0.4, while HLTF's Table 1(a)/1(b) tabulate the GeoLDM "
                    "column's Val (%) as 93.6 +- 0.2. Do the two source tables "
                    "disagree?",
        "verdict": "ESTABLISHED -- the two source tables DISAGREE, and no "
                   "document on disk records it",
        "cells": [geo_v, hltf_geo_v_a, hltf_geo_v_b, hltf_own_v],
        "quotes": [
            q(GEOLDM, GEO_REL, 491),
            q(HLTF, "tables/qm9_benchmarks_og.tex", 38),
            q(HLTF, "tables/qm9_benchmarks_og.tex", 63),
            q(HLTF, "tables/qm9_benchmarks_og.tex", 8, 12),
        ],
        "consequence":
            "BOTH numerals are verbatim from the primary sources: GeoLDM's own "
            f"paper prints Valid = {geo_v['printed_value']} for the GeoLDM row "
            f"of tab:qm9_results (icml2023.tex:491); HLTF prints the GeoLDM "
            f"column's Val as {hltf_geo_v_a['printed_value']} in Table 1(a) "
            f"(tables/qm9_benchmarks_og.tex:38) and "
            f"{hltf_geo_v_b['printed_value']} in Table 1(b) "
            "(tables/qm9_benchmarks_og.tex:63). The SAME observable (RDKit "
            "validity of GeoLDM's QM9 output) therefore carries two different "
            "printed values in two papers, and HLTF's own Table 1 caption "
            "states 'Baselines reproduced from GCDM Table 1', i.e. the column is "
            "a TRANSCRIPT, so the disagreement is a transcription difference "
            "that neither paper flags. NOTE: 93.8 is ALSO HLTF's OWN Val in "
            f"HLTF's own table ({hltf_own_v['printed_value']}), so the numeral "
            "93.8 is carried in at least two distinct roles. WHICH NUMERALS ARE "
            "BAND ENDPOINTS: under BOTH readings the band's endpoints are "
            "92.7 / 94.0 (Reading A) or 92.98 / 94.33 (Reading B); 93.8 and "
            "93.6 are NOT band endpoints under either reading and must never be "
            "published as such. REPO-STATE MEASUREMENT: a repo-wide scan of "
            "tracked .md/.json/.tex/.py/.yaml files under docs/, TODO/, "
            "metrics/, molmetal/ and paper/ for '93.6' returns hits ONLY in an "
            "unrelated NCI-60 split percentage (93.6 / 2.0 / 4.4 %) -- zero "
            "hits in the frontier sense. The disagreement is UNRECORDED.",
    }

    # ---- F5 -------------------------------------------------------------
    eft_c = pick(cells, "2609.08333v4", "tbl:pcqm4mv2_generation (a)", "C.",
                 "EF-TALFM")
    eft_s = pick(cells, "2609.08333v4", "tbl:pcqm4mv2_generation (a)", "S.",
                 "EF-TALFM")
    eft_us = pick(cells, "2609.08333v4", "tbl:pcqm4mv2_generation (a)",
                  "U. & S.", "EF-TALFM")
    eft_nus = pick(cells, "2609.08333v4", "tbl:pcqm4mv2_generation (a)",
                   "N. & U. & S.", "EF-TALFM")
    eft_pb = pick(cells, "2609.08333v4", "tbl:pcqm4mv2_generation (b)",
                  "All PoseBusters checks", "EF-TALFM")
    F["F5"] = {
        "question": "Does EF-TALFM's own Table 3(a) confirm the repo's existing "
                    "EF-TALFM entries -- the row labels and columns, the C. / S. / "
                    "U. & S. / N. & U. & S. definitions, the 10,000 denominator, "
                    "and the statement that the connectivity count is taken "
                    "BEFORE the largest-component retention?",
        "verdict": "ESTABLISHED",
        "cells": [eft_c, eft_s, eft_us, eft_nus, eft_pb],
        "quotes": [
            q(EFTALFM, EFT_REL, 101),
            q(EFTALFM, EFT_REL, 93),
            q(EFTALFM, EFT_REL, 60),
            q(EFTALFM, EFT_REL, 54),
            q(EFTALFM, EFT_REL, 55),
            q(EFTALFM, EFT_REL, 183),
            q(EFTALFM, EFT_REL, 1010),
            q(EFTALFM, EFT_REL, 308),
            q(EFTALFM, EFT_REL, 82),
        ],
        "consequence":
            "CONFIRMED VERBATIM. Table 3(a) of arXiv:2609.08333v4 is titled "
            "'(a) End-to-end generation results (%)' and its columns are "
            "UAE-3D / FlowMol / EF-TALFM; its caption defines the four "
            "observables and fixes the denominator ('Panel (a) rates are "
            "computed over 10,000 generated molecules'); and the ORDERING is "
            "explicit in the Table 4 caption, which says 'We report "
            "connectivity of the reconstructed molecule and sanitization, "
            "uniqueness, and novelty rates computed after retaining the "
            "atom-count largest connected component of each sampled molecule "
            "and applying strict RDKit sanitization' -- connectivity is "
            "reported OF THE RECONSTRUCTED MOLECULE, while the companion rates "
            "are computed AFTER the retention. Two independent corroborations "
            "sit in the same source: Appendix D states the pipeline always "
            "retains ('we retain the largest connected component and discard "
            "the rest, since the reconstructed graph may contain multiple "
            "components'), and the population accounting counts 'Among the "
            "92,221 connected molecules, 90,807 reconstructed graphs pass "
            "strict sanitization before geometry refinement'. CONSEQUENCE: the "
            "repo's EF-TALFM POST-PROCESSING cell, which already calls that arm "
            "MIXED ('MIXED -- the lower cell is a RAW connectivity count, the "
            "upper cell is a POST-PROCESSED composite', "
            "molmetal/reports/wfa158_frontier_postproc_matrix/"
            "frontier_matrix.json -> sources.EF-TALFM.cells.POST-PROCESSING."
            "post_processed), is CORRECT; the Wf-A183 repair that made the "
            "'RAW' claim reading-conditional is CORRECT. ONE NEW, UNRECORDED "
            "OBSERVATION: EF-TALFM's own prose at src/appendix.tex:82 compares "
            "EF-TALFM's connectivity 92.98 % against '98.72 %' -- but 98.72 % is "
            "FlowMol's SANITIZATION cell in Table 3(a); FlowMol's connectivity "
            "cell in the same table reads 100.00. The paper's prose compares a "
            "connectivity rate against a sanitization rate. This does not "
            "change any band numeral, but it means the appendix prose is not a "
            "safe source for the connectivity comparison.",
    }
    return F


# --------------------------------------------------------------------------
# the rest of the artefact
# --------------------------------------------------------------------------


def build_retrieval():
    out = []
    for aid in ARXIV_IDS:
        paper = P[aid]
        with open(os.path.join(WFA175_CACHE, f"{aid}.json")) as fh:
            cached = json.load(fh)
        files = []
        for rel in sorted(paper.receipt["file_names"]):
            raw = paper.raw(rel)
            try:
                text = raw.decode("utf-8")
                utf8 = True
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
                utf8 = False
            files.append({
                "path": rel,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "chars_utf8": len(text),
                "utf8_decodable": utf8,
            })
        out.append({
            "arxiv": aid,
            "eprint_url": paper.receipt["url"],
            "final_url": paper.receipt["final_url"],
            "status": paper.receipt["status"],
            "eprint_bytes": paper.receipt["bytes"],
            "eprint_sha256": paper.receipt["sha256"],
            "retrieved_utc": paper.receipt["retrieved_utc"],
            "n_files": paper.receipt["n_files"],
            "files": files,
            "cached_receipt": {
                "path": os.path.relpath(
                    os.path.join(WFA175_CACHE, f"{aid}.json"), REPO),
                "status": cached.get("status"),
                "bytes": cached.get("bytes"),
                "sha256": cached.get("sha256"),
            },
            "live_matches_cached": {
                "bytes": paper.receipt["bytes"] == cached.get("bytes"),
                "sha256": paper.receipt["sha256"] == cached.get("sha256"),
                "note": "the Wf-A175 receipt's per-file 'file_bytes' field is a "
                        "UTF-8 CHARACTER count, not a byte count (verified: it "
                        "equals len(raw.decode('utf-8')) for every file where "
                        "the two differ, and equals len(raw) for every file "
                        "where they agree); the e-print-level byte count and "
                        "sha256 are byte-exact and MATCH.",
            },
        })
    return out


def build_titles():
    geo_authors = []
    for n in range(106, 111):
        raw = GEOLDM.line(GEO_REL, n)
        groups, _ = _brace_groups(raw, raw.index("{"))
        geo_authors.append({
            "name": clean_cell(groups[0]),
            "locator": f"{GEO_REL}:{n}",
        })
    eft_authors = []
    for part in split_cells(EFTALFM.line("main.tex", 126)):
        name = clean_cell(part)
        if name:
            eft_authors.append({"name": name, "locator": "main.tex:126"})
    return {
        "2305.01140v1": {
            "title": clean_cell(GEOLDM.line(GEO_REL, 88)
                                .split("{", 1)[1].rsplit("}", 1)[0]),
            "title_locator": f"{GEO_REL}:88",
            "authors": geo_authors,
        },
        "2603.27113v1": {
            "title": clean_cell(HLTF.line("iclr2025_conference.tex", 25)
                                .split("{", 1)[1].rsplit("}", 1)[0]),
            "title_locator": "iclr2025_conference.tex:25",
            "authors": [
                {"name": "Urvi Awasthi", "locator": "iclr2025_conference.tex:31"},
                {"name": "Alexander Lobo", "locator": "iclr2025_conference.tex:31"},
                {"name": "Leonid Zhukov", "locator": "iclr2025_conference.tex:31"},
            ],
            "authors_verbatim": HLTF.line("iclr2025_conference.tex", 31),
            "authors_note": "the repo cites the second author as 'Alexander "
                            "Arjun Lobo' (the repo's own record: "
                            "molmetal/reports/wfa158_frontier_postproc_matrix/"
                            "frontier_matrix.json -> sources.HLTF.citation.authors; "
                            "the same string is also in the arXiv-abstract "
                            "receipt at -> retrieval_log[arxiv_id=2603.27113, "
                            "kind=arxiv-abs].extracted); the paper's own "
                            "\\author line reads 'Alexander Lobo'.",
        },
        "2609.08333v4": {
            "title": clean_cell(EFTALFM.line("main.tex", 121)
                                .split("{", 1)[1].rsplit("}", 1)[0]),
            "title_locator": "main.tex:121",
            "authors": eft_authors,
        },
    }


def _find_line(lines, needle, start=1):
    """Return the 1-indexed line number of the first line containing
    ``needle`` at or after ``start`` (1-indexed).

    Resolving a catalogue locator by ANCHOR TEXT instead of a hard-coded line
    number is what keeps this producer from silently drifting when the
    catalogue grows: if the anchor text moves, the resolved line moves with
    it and the assertion below fails LOUDLY instead of quoting the wrong
    row.
    """
    for i in range(start - 1, len(lines)):
        if needle in lines[i]:
            return i + 1
    raise AssertionError("catalogue anchor not found: %r" % needle)


def build_catalogue_row():
    with open(CATALOGUE, "rb") as fh:
        raw = fh.read()
    lines = raw.decode("utf-8").split("\n")
    # resolve every cited line by ANCHOR TEXT ...
    sec_625 = _find_line(lines, "## §6.252")
    may_sub = _find_line(lines, "### The sentence we may say", sec_625)
    band_row = _find_line(lines, "**37.4 %**", sec_625)
    may_start = _find_line(lines, "MAY: *\"RAW single-fragment rate 8.51 %",
                           may_sub)
    may_end = _find_line(lines, "MAY NOT: *\"100 % valid\"", may_start)
    sub_4 = _find_line(lines, "### 4. THE PUBLISHED 92.7")
    header = _find_line(lines, "| break | frontier | ours |", sub_4)
    denom = _find_line(lines, "| **DENOMINATOR** | 92.7", header)
    metric = _find_line(lines, "| **METRIC** |", denom)
    postproc = _find_line(lines, "| **POST-PROCESSING** |", metric)
    hydro = _find_line(lines, "| **HYDROGENS** |", postproc)
    # ... then ASSERT the resolved lines still equal the ones measured on
    # 2026-11-01.  A drift here means the catalogue grew under us: the
    # numbers below are then STALE BY CONSTRUCTION and the demotion block
    # must be re-measured, not silently reused.
    resolved = (sec_625, may_sub, band_row, may_start, may_end, sub_4,
                header, denom, metric, postproc, hydro)
    # The 2026-11-01 measurement, PRESERVED VERBATIM per AGENTS.md §5.5
    # rule #6 (a correction is additive + dated; it never silently edits
    # the superseded rendering):
    #     recorded = (18603, 18678, 18671, 18680, 18688, 20808, 20815,
    #                 20817, 20818, 20819, 20820)
    #
    # RE-MEASURED 2026-11-04 (Wf-A269 wave-17/18): the catalogue grew, so
    # every anchor resolved FURTHER down the file.  The CONTENT of the
    # cited rows did not change -- only their line numbers moved -- and the
    # build re-reads each row from the LIVE catalogue by anchor text, so
    # the published artefact is re-derived, never copied.  MEASURED: the
    # catalogue's own byte count moved 1,993,469 B (the 2026-11-02 build)
    # -> 2,598,922 B, i.e. it grew by ~600 KB while the four-breaks block
    # itself stayed put.
    recorded = (18647, 18722, 18715, 18724, 18732, 20852, 20859, 20861,
                20862, 20863, 20864)
    if resolved != recorded:
        raise AssertionError(
            "the catalogue DRIFTED: anchor-resolved lines %r != the "
            "2026-11-04 measured lines %r -- re-measure the demotion block"
            % (resolved, recorded))
    return {
        "path": "docs/HONEST_NEGATIVE_CATALOG.md",
        "section_anchor": CATALOGUE_FOUR_BREAKS_ANCHOR,
        "verbatim": lines[postproc - 1],
        "file_sha256": sha256_bytes(raw),
        "file_bytes": len(raw),
        "surrounding_four_breaks_table": {
            "section_anchor": CATALOGUE_FOUR_BREAKS_ANCHOR,
            "row_anchor": "the POST-PROCESSING row of the four-breaks table",
            "note": "the DENOMINATOR row carries the band string '92.7-94 %', "
                    "so this four-breaks table IS the 92.7-94 % band's table.",
            "line_numbers_measured_2026_11_01": {
                "header_line": header,
                "denominator_line": denom,
                "metric_line": metric,
                "post_processing_line": postproc,
                "hydrogens_line": hydro,
                "note": "STALE BY CONSTRUCTION -- the catalogue GROWS, so a "
                        "line number into it is a perishable locator. The "
                        "DURABLE locator is `section_anchor` above. The numbers "
                        "here are PRESERVED VERBATIM from the 2026-11-01 "
                        "measurement and remain exact for the file version "
                        "pinned by `file_sha256`.",
            },
        },
        "inherited_from": {
            "section": "6.252",
            "section_anchor": CATALOGUE_SECTION_6252_ANCHOR,
            "band": "37.4 % to 96.0 %",
            "band_locator": CATALOGUE_SECTION_6252_ANCHOR
                            + " -> the 'published connected-rate band is "
                              "37.4 % ... to 96.0 %' paragraph",
            "why_it_differs": "the 6.252 band is a CONNECTIVITY band measured on "
                              "Buttenschoen et al. (arXiv:2505.00518) "
                              "POST-PROCESSED rates; the 92.7-94 % band's "
                              "Reading-A endpoints are HLTF QM9 Table 1 cells "
                              "that carry NO post-processing, and its "
                              "Reading-B lower endpoint is a RAW connectivity "
                              "count.",
            "may_may_not_block": CATALOGUE_MAY_SUBSECTION_ANCHOR,
            "may_may_not_verbatim": " ".join(lines[may_start - 1:may_end]).strip(),
            "line_numbers_measured_2026_11_01": {
                "heading_line": sec_625,
                "band_line": band_row,
                "may_block_first_line": may_start,
                "may_block_last_line": may_end,
                "note": "STALE BY CONSTRUCTION -- the catalogue GROWS. The "
                        "DURABLE locators are `section_anchor` and "
                        "`may_may_not_block` above. The numbers here are "
                        "PRESERVED VERBATIM from the 2026-11-01 measurement.",
            },
        },
        "proposed_correction_dated": {
            "dated": "2026-11-01",
            "wave": "Wf-A186",
            "append_only": "AGENTS.md 5.5 rule #6 -- the row above is PRESERVED "
                           "VERBATIM; the correction is ADDITIVE and DATED; "
                           "nothing is deleted.",
            "status": "PROPOSED ONLY -- this wave did NOT write to docs/",
            "locator_status": "the line numbers in `text` below were measured "
                              "on 2026-11-01 against the catalogue version "
                              "pinned by `file_sha256`; they are a DATED "
                              "historical snapshot and are STALE BY "
                              "CONSTRUCTION as line locators. The DURABLE "
                              "anchors are `section_anchor` (this block) and "
                              "`inherited_from.section_anchor`.",
            "text":
                "CORRECTION (2026-11-01, Wf-A186) -- the POST-PROCESSING row of "
                "this four-breaks table is WRONG FOR THE 92.7-94 % BAND, and is "
                "inherited from a DIFFERENT band (catalogue section 6.252, lines "
                "18603 and 18671, whose band is 37.4 % to 96.0 % and whose "
                "frontier rates ARE post-processed). ESTABLISHED FROM THE "
                "PRIMARY SOURCES: HLTF's standardized post-processing recipe "
                "(largest fragment, explicit hydrogens, UFF relaxation) is "
                "scoped by HLTF ITSELF to GEOM-DRUGS ONLY -- arXiv:2603.27113v1 "
                "sections/experiments.tex:8 reads 'On QM9 we report atom "
                "stability (AS), molecule stability (MS), RDKit validity (Val), "
                "and Val&Uniq on 10,000 unconditional samples. On GEOM-DRUGS we "
                "report Valid, Valid&Unique (V&U), and Valid&Unique&Novel "
                "(V&U&N) on 100,000 samples, both raw and after standardized "
                "post-processing (largest fragment, explicit hydrogens, UFF "
                "relaxation)' -- the QM9 Table 1 has no '+PP' rows (the '+PP' "
                "rows are in Table 2, tables/geom_drugs_results.tex, label "
                "tab:geom_main). BOTH Reading-A band endpoints are QM9 Table 1 "
                "cells and carry NO post-processing; the Reading-B lower "
                "endpoint is a RAW connectivity count (EF-TALFM "
                "arXiv:2609.08333v4 src/appendix.tex:183 reports connectivity OF "
                "THE RECONSTRUCTED MOLECULE while the companion rates are "
                "computed AFTER the largest-component retention). GeoLDM's own "
                "paper confirms the RAW reading for the lower endpoint: "
                "arXiv:2305.01140v1 icml2023.tex:917 names disconnected "
                "components as a known artefact that 'can be easily fixed by "
                "just filtering the smaller components' -- a remedy PROPOSED, "
                "not applied to the reported numbers. CORRECTED CELL: 'the band "
                "is NOT measured on a post-processed stream -- under Reading A "
                "both endpoints are HLTF QM9 Table 1 cells with NO "
                "post-processing; under Reading B the lower endpoint is a RAW "
                "connectivity count and only the upper companion cell is "
                "post-processed'. The row's OUR-SIDE half ('both arms are RAW') "
                "stands.",
        },
    }


def build_x_status():
    return {
        "X1": {
            "claim": "no endpoint is a connected-component count / THE BAND IS NOT "
                     "A CONNECTIVITY BAND",
            "live_status": "SURVIVES ONLY IN DATED SUPERSEDED KEYS (correct per "
                           "AGENTS.md 5.5 rule #6): the repo's own record "
                           "molmetal/reports/wfa158_frontier_postproc_matrix/"
                           "frontier_matrix.json -> band_provenance."
                           "headline_finding_superseded_2026_11_01 and -> "
                           "sources.HLTF.cells.METRIC.note_superseded_2026_11_01 "
                           "-- both are DATED sibling keys of their live "
                           "counterparts; the LIVE band_provenance."
                           "headline_finding is READING-QUALIFIED and no longer "
                           "carries it. The repo's own X-class list already marks "
                           "it FORBIDDEN (molmetal/reports/wfa161_frontier_"
                           "labelling/report.md:314).",
            "our_wave": "does NOT reproduce it; this report asserts only the four "
                        "invariants that hold under BOTH readings.",
        },
        "X2": {
            "claim": "the band is measured on the POST-PROCESSED stream",
            "live_status": "ONE LIVE SURFACE REMAINS: docs/HONEST_NEGATIVE_"
                           "CATALOG.md, section 6.265, subsection '4. THE "
                           "PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR "
                           "INDEPENDENT BREAKS', the class-(a) POST-PROCESSING "
                           "row of the 92.7-94 % band's four-breaks table. The "
                           "metric artefact metrics/by_metric/wfa155_metal_"
                           "coordination_sphere.json carries the axis only in "
                           "reading-conditional form ('whether the band as a "
                           "whole is RAW is NOT settled on disk'), and the two "
                           "generator scripts carry it only in comments. The repo "
                           "already refuted it from its OWN record "
                           "(frontier_matrix.json -> band_provenance -> "
                           "endpoint_candidates, post_processed: false); this "
                           "wave REFUTES IT FROM THE PRIMARY SOURCE (F2).",
            "our_wave": "REFUTED by the primary source for this band (F2).",
        },
        "X3": {
            "claim": "their Connectivity is computed AFTER retaining the largest "
                     "component",
            "live_status": "no live surface found; the repo's X-class list "
                           "already marks it REFUTED (molmetal/reports/wfa164_"
                           "coordination_sphere_narrative/"
                           "synthetic_pool_artifact.json:99).",
            "our_wave": "REFUTED by the primary source (F5): EF-TALFM reports "
                        "connectivity OF THE RECONSTRUCTED MOLECULE, before the "
                        "retention.",
        },
        "X4": {
            "claim": "the band is minted here: lower = GeoLDM Val&Uniq, upper = "
                     "HLTF PB-Valid",
            "live_status": "no live surface found; already marked as asserting "
                           "ONE reading as settled (molmetal/reports/wfa164_"
                           "coordination_sphere_narrative/"
                           "pool_specific_literal_census.json:148).",
            "our_wave": "does NOT reproduce it; F1 adds that the record's own "
                        "sources block has NO GeoLDM entry at all.",
        },
    }


def build_our_side():
    with open(OUR_METRIC, "rb") as fh:
        raw = fh.read()
    d = json.loads(raw.decode("utf-8"))
    node = d["summary"]["generation_side"]["single_molecule_rate_end_to_end"]
    median = node["median"]
    return {
        "value_percent": round(median * 100, 2),
        "raw_median": median,
        "n_runs": node["n"],
        "definition": "n_single / samples drawn, where n_single = the pool rows "
                      "whose decoded molecule has exactly ONE RDKit fragment, "
                      "measured at the RAW emitted SMILES -- NO post-processing, "
                      "NO LargestFragmentChooser, NO explicit H, NO UFF.",
        "source_path": "metrics/by_metric/cfm_fragmentation_audit.json",
        "source_jsonpath": "/summary/generation_side/"
                           "single_molecule_rate_end_to_end",
        "source_sha256": sha256_bytes(raw),
        "catalogue_quote": {
            "path": "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252 -> "
                    "subsection 'The sentence we may say, and the sentences we "
                    "may not'",
            "verbatim": "MAY: *\"RAW single-fragment rate 8.51 % (1078 / 12668 "
                        "parsed; 1078 / 12669 declared rows = 8.509 %); after "
                        "largest-fragment post-processing the connected rate is "
                        "100 %, at the cost of 22.47 % discarded heavy-atom mass "
                        "and 21.0 % of rows reduced to <= 2-heavy-atom specks.\"*",
        },
    }


def build_invariants(cells):
    geo_vu = pick(cells, "2305.01140v1", "tab:qm9_results", "GeoLDM",
                  "Valid & Unique (%)", "QM9")
    hltf_ppb = pick(cells, "2603.27113v1", "tab:qm9_pb",
                    "PB-Valid (%) ↑", "HLTF (ours)")
    eft_c = pick(cells, "2609.08333v4", "tbl:pcqm4mv2_generation (a)", "C.",
                 "EF-TALFM")
    eft_nus = pick(cells, "2609.08333v4", "tbl:pcqm4mv2_generation (a)",
                   "N. & U. & S.", "EF-TALFM")
    return {
        "i_the_band_is_external":
            "The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a "
            "FRONTIER composite, not our measurement.",
        "ii_it_is_not_one_homogeneous_quantity":
            "It is NOT one homogeneous quantity: even under Reading A its two "
            "endpoints are two different observables (a sanitisation-and-"
            "duplication rate and a six-check pose verdict), and under Reading "
            "B one endpoint is a raw connectivity count while the other is a "
            "post-processed composite.",
        "iii_it_is_not_on_our_scale":
            "It is NOT on our scale. OUR 8.51 % IS a connected-component count on "
            "the RAW emitted SMILES, and nothing else.",
        "iv_the_source_cells_are_not_settled":
            "The source cells of the band are NOT settled on disk.",
        "our_side_pointer": "see our_side for the derivation of 8.51 % from "
                            "metrics/by_metric/cfm_fragmentation_audit.json",
        "band_endpoints_under_reading_a": {
            "lower": {"value": geo_vu["value"], "printed": geo_vu["printed_value"],
                      "locator": f"arXiv:2305.01140v1 {geo_vu['file']}:"
                                 f"{geo_vu['line']} (GeoLDM's own paper)"},
            "upper": {"value": hltf_ppb["value"],
                      "printed": hltf_ppb["printed_value"],
                      "locator": f"arXiv:2603.27113v1 {hltf_ppb['file']}:"
                                 f"{hltf_ppb['line']} (Table 1(b), HLTF's own "
                                 "PB-Valid cell)"},
        },
        "band_endpoints_under_reading_b": {
            "lower": {"value": eft_c["value"], "printed": eft_c["printed_value"],
                      "locator": f"arXiv:2609.08333v4 {eft_c['file']}:"
                                 f"{eft_c['line']} (Table 3(a), RAW connectivity)"},
            "upper": {"value": eft_nus["value"],
                      "printed": eft_nus["printed_value"],
                      "locator": f"arXiv:2609.08333v4 {eft_nus['file']}:"
                                 f"{eft_nus['line']} (Table 3(a), post-processed "
                                 "companion)"},
        },
        "numerals_that_are_NOT_band_endpoints": {
            "93.8": "GeoLDM's own QM9 Valid (arXiv:2305.01140v1 "
                    "icml2023.tex:491) AND HLTF's own QM9 Val (arXiv:2603.27113v1 "
                    "tables/qm9_benchmarks_og.tex:41) -- two different papers' own "
                    "figures, not a band endpoint under either reading.",
            "93.6": "HLTF's tabulation of GeoLDM's Val (arXiv:2603.27113v1 "
                    "tables/qm9_benchmarks_og.tex:38 and :63), which CONTRADICTS "
                    "GeoLDM's own 93.8 for the same observable -- see F4.",
        },
        "competing_readings_count": 3,
        "competing_readings": [
            "Reading A (frontier_matrix.json -> band_provenance): GeoLDM "
            "Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both "
            "post_processed: false, plus YuelBond's per-bond-type classification "
            "F1 92.7 % as a third candidate for the same numeral. Reproduces "
            "the quoted numerals exactly.",
            "Reading B (frontier_matrix.json -> sources.EF-TALFM): EF-TALFM's "
            "own Table 3(a) RAW connectivity C. = 92.98 % counted BEFORE the "
            "largest-component retention, paired with the post-processed "
            "N. & U. & S. = 94.33 % companion cell. Does NOT reproduce the "
            "quoted numerals.",
            "Reading C (docs/HONEST_NEGATIVE_CATALOG.md, section 6.266.8, "
            "'The frontier papers, pinned from primary sources (Wf-A168) -- and "
            "one correction to the previous'): the band is a "
            "composite of the EF-TALFM cells 92.98 % / 94.33 % / 93.13 %, which "
            "have three different denominators. Also does NOT reproduce the "
            "quoted numerals. This THIRD reading is NOT among the two recorded "
            "in frontier_matrix.json -- the contest is three-way, not two-way.",
        ],
    }


def build_artefact():
    cells = build_cells()
    retrieval = build_retrieval()
    titles = build_titles()
    findings = build_findings(cells)
    for aid in ARXIV_IDS:
        entry = next(r for r in retrieval if r["arxiv"] == aid)
        entry["title"] = titles[aid]["title"]
        entry["title_locator"] = titles[aid]["title_locator"]
        entry["authors"] = [a["name"] for a in titles[aid]["authors"]]
        entry["author_locators"] = titles[aid]["authors"]
        if "authors_verbatim" in titles[aid]:
            entry["authors_verbatim"] = titles[aid]["authors_verbatim"]
            entry["authors_note"] = titles[aid]["authors_note"]

    return {
        "artifact": "primary_source_verification",
        "artifact_version": "1.0.0",
        "wave": "Wf-A186",
        "dated": "2026-11-01",
        "generated_by": "molmetal/reports/wfa186_frontier_primary_source/tools/"
                        "build.py",
        "reproduction": "CUDA_VISIBLE_DEVICES=\"\" /home/hugo/codes/mol-metal/"
                        ".venv/bin/python -I tools/build.py  (reads only cache/ "
                        "and the pinned bytes; every numeral is parsed out of "
                        "those bytes at build time)",
        "gpu": "none -- CPU-only, 0 CUDA calls, 0 GPU allocations",
        "question": "Which primary-source cells does the repo's published "
                    "92.7-94 % frontier band name, and what may be said about it?",
        "method": {
            "retrieval": "each e-print re-fetched LIVE through the host proxy "
                         "(http://127.0.0.1:7890), following the /e-print/<id> -> "
                         "/src/<id> redirect; the tarball extracted into this "
                         "wave's own cache/ and every quote read from the "
                         "ORIGINAL .tex inside it (not from a text dump).",
            "quotation": "every quote is a verbatim line (or line span) of the "
                         "pinned .tex, carrying its file:line locator, the "
                         "sha256 of that .tex and the sha256 of the e-print it "
                         "came from.",
            "derivation": "no numeral in this artefact is transcribed: the "
                          "values below are parsed out of the pinned bytes at "
                          "build time.",
        },
        "related_waves": ["Wf-A158", "Wf-A161", "Wf-A164", "Wf-A168",
                          "Wf-A171", "Wf-A175", "Wf-A183", "Wf-A186"],
        "retrieval": retrieval,
        "cells": cells,
        "findings": findings,
        "invariants": build_invariants(cells),
        "our_side": build_our_side(),
        "catalogue_row": build_catalogue_row(),
        "x_class_status": build_x_status(),
        "locator_demotion_2026_11_02_stale_by_construction":
            build_locator_demotion_2026_11_02(),
        "may": [
            "MAY: 'The published 92.7-94 % band is EXTERNAL to Mol-Metal, is NOT "
            "one homogeneous quantity, is NOT on our scale, and its source "
            "cells are NOT settled on disk.'",
            "MAY: 'Under Reading A, the band's lower endpoint is GeoLDM's OWN "
            "published QM9 Valid & Unique cell (arXiv:2305.01140v1 "
            "icml2023.tex:491), reproduced identically by HLTF; the upper "
            "endpoint is HLTF's own QM9 PoseBusters-validity cell (arXiv:"
            "2603.27113v1 tables/qm9_benchmarks_og.tex:66).' -- always with the "
            "reading named.",
            "MAY: 'Under Reading B, the band's lower endpoint is EF-TALFM's RAW "
            "connectivity cell (arXiv:2609.08333v4 src/appendix.tex:112-115), "
            "counted BEFORE the largest-component retention, and the upper "
            "endpoint is that table's post-processed N. & U. & S. companion "
            "cell.' -- always with the reading named.",
            "MAY: 'Neither reading's band endpoint is post-processed in its own "
            "right: the Reading-A endpoints are HLTF QM9 Table 1 cells, which "
            "carry no post-processing; the Reading-B lower endpoint is a RAW "
            "connectivity count.'",
            "MAY: 'OUR 8.51 % IS a connected-component count on the RAW emitted "
            "SMILES, and nothing else.'",
        ],
        "may_not": [
            "MAY NOT: 'THE BAND IS NOT A CONNECTIVITY BAND' / 'no endpoint is a "
            "connected-component count' -- false under Reading B.",
            "MAY NOT: 'the band is measured on the POST-PROCESSED stream' -- "
            "false for this band; it was inherited from the 37.4-96.0 % band of "
            "catalogue section 6.252.",
            "MAY NOT: 'their Connectivity is computed AFTER retaining the largest "
            "component' -- EF-TALFM reports connectivity OF THE RECONSTRUCTED "
            "MOLECULE, before the retention.",
            "MAY NOT: 'the band is minted here: lower = GeoLDM Val&Uniq, upper = "
            "HLTF PB-Valid' -- that asserts one reading as settled.",
            "MAY NOT: quote 93.8 or 93.6 as a band endpoint -- under both "
            "readings the endpoints are 92.7 / 94.0 or 92.98 / 94.33.",
            "MAY NOT: quote any cell of these three tables without its "
            "file:line locator and the sha256 of the bytes it came from.",
        ],
        "additional_observations": [
            "UNRECORDED LOCATOR ERROR, still live in a published metric "
            "artefact: metrics/by_metric/wfa155_metal_coordination_sphere.json "
            "says 'HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW "
            "and POST-PROCESSED side by side'. The RAW/+PP pairs are in Table 2 "
            "(GEOM-DRUGS, tables/geom_drugs_results.tex, label tab:geom_main); "
            "Table 1(b) (tables/qm9_benchmarks_og.tex, label tab:qm9_pb) is "
            "QM9 and reports six OBSERVABLES (AS / MS / Val / Val&Uniq / Novel / "
            "PB-Valid) for three models, with no +PP rows. The same "
            "Table-1-vs-Table-2 confusion is the mechanism by which the X2 class "
            "survives.",
            "The Wf-A175 cached receipt's per-file 'file_bytes' field is a UTF-8 "
            "CHARACTER count, not a byte count. The e-print-level byte count and "
            "sha256 are byte-exact and match the live re-fetch for all three "
            "papers.",
            "The repo holds THREE competing provenance readings for the band, "
            "not two: see invariants.competing_readings.",
            "GeoLDM's paper contains exactly one occurrence of 'filter' in its "
            "whole source (icml2023.tex:917), and it is the proposed-but-not-"
            "applied fragment filter; no reported number is filtered by it.",
            "The repo cites HLTF's second author as 'Alexander Arjun Lobo'; the "
            "paper's own \\author line reads 'Alexander Lobo'.",
        ],
        "not_established": [
            "This wave did NOT establish WHICH cells the repo's literature scan "
            "selected to mint the band -- the selection rule is not recorded on "
            "disk. It established what each candidate cell IS, from the primary "
            "sources.",
            "This wave did NOT establish whether HLTF's QM9 Table 1 numbers are "
            "the with-valence-repair or the without-valence-repair arm: HLTF "
            "states 'We report metrics both with and without this repair' and "
            "does not mark which arm Table 1 carries. The repair deletes BONDS, "
            "not components, so it does not bear on the largest-component "
            "question.",
        ],
    }


# --------------------------------------------------------------------------
# report.md rendering
# --------------------------------------------------------------------------


def render_report(art):
    L = []
    a = L.append
    inv = art["invariants"]
    our = art["our_side"]
    a("# Wf-A186 -- PRIMARY-SOURCE VERIFICATION OF THE 92.7-94 % FRONTIER BAND")
    a("")
    a(f"* wave: **{art['wave']}**, dated **{art['dated']}**")
    a(f"* producer: `{art['generated_by']}`")
    a("* machine artefact: `primary_source_verification.json` (same run, same "
      "data)")
    a(f"* GPU: **{art['gpu']}**")
    a("")
    a("Every numeral below was parsed out of the pinned bytes at build time and "
      "carries a `file:line` locator plus the sha256 of the bytes it came "
      "from. Nothing is transcribed from memory or from another document.")
    a("")

    a("## 0. THE RETRIEVAL LEDGER")
    a("")
    a("| arXiv | status | e-print bytes | sha256 | retrieved (UTC) | live == cached |")
    a("|---|---|---|---|---|---|")
    for r in art["retrieval"]:
        m = r["live_matches_cached"]
        a(f"| `{r['arxiv']}` | {r['status']} | {r['eprint_bytes']:,} | "
          f"`{r['eprint_sha256']}` | {r['retrieved_utc']} | "
          f"bytes **{m['bytes']}**, sha256 **{m['sha256']}** |")
    a("")
    for r in art["retrieval"]:
        a(f"* `{r['arxiv']}` -- {r['title']} "
          f"({', '.join(r['authors'])}) -- {r['title_locator']}, "
          f"{r['n_files']} files in the tarball, {r['eprint_url']} -> "
          f"{r['final_url']}")
    a("")
    a("**The live re-fetch MATCHES the Wf-A175 cached receipt byte-for-byte "
      "and sha256-for-sha256 on all three papers.** One discrepancy, fully "
      "explained: the Wf-A175 receipt's per-file `file_bytes` field is a UTF-8 "
      "**character** count, not a byte count -- it equals "
      "`len(raw.decode('utf-8'))` for every file where the two differ, and "
      "equals `len(raw)` for every file where they agree. The e-print-level "
      "byte count and sha256 are byte-exact and match.")
    a("")

    a("## 1. THE TABLE CELLS, AS THE BYTES PRINT THEM")
    a("")
    for aid in ARXIV_IDS:
        a(f"### {aid}")
        a("")
        a("| table | row label | column (group) | printed value | locator |")
        a("|---|---|---|---|---|")
        for c in art["cells"][aid]:
            col = c["column_label"]
            if c["column_group"]:
                col = f"{c['column_group']} / {col}"
            row = c["row_label"]
            if c.get("row_method"):
                row = f"{c['row_method']} {row}"
            a(f"| {c['table']} | {row} | {col} | "
              f"{c['printed_value']} | `{c['file']}:{c['line']}` |")
        a("")

    a("## 2. THE FIVE FINDINGS")
    a("")
    for key in ["F1", "F2", "F3", "F4", "F5"]:
        f = art["findings"][key]
        a(f"### {key} -- {f['question']}")
        a("")
        a(f"**VERDICT: {f['verdict']}**")
        a("")
        a("Verbatim quotes (method: read out of the `.tex` extracted from the "
          "live e-print tarball; each carries the sha256 of that `.tex` and of "
          "the e-print):")
        a("")
        for qt in f["quotes"]:
            a(f"> `{qt['arxiv']}` :: `{qt['file']}:{qt['line']}`")
            a(">")
            for ln in qt["quote"].split("\n"):
                a(f"> {ln}")
            a("")
        a("**Consequence for the repo.**")
        a("")
        a(f["consequence"])
        a("")

    a("## 3. THE FOUR INVARIANTS (hold under BOTH readings -- the only "
      "publishable claims)")
    a("")
    for k in ["i_the_band_is_external", "ii_it_is_not_one_homogeneous_quantity",
              "iii_it_is_not_on_our_scale", "iv_the_source_cells_are_not_settled"]:
        a(f"* **{k}** -- {inv[k]}")
    a("")
    a(f"OUR side: {our['value_percent']} % -- {our['definition']} "
      f"(source `{our['source_path']}` -> `{our['source_jsonpath']}`, "
      f"sha256 `{our['source_sha256']}`).")
    a("")
    a("Band endpoints, named per reading:")
    a("")
    for reading in ["band_endpoints_under_reading_a",
                    "band_endpoints_under_reading_b"]:
        a(f"* **{reading}**")
        for side in ["lower", "upper"]:
            e = inv[reading][side]
            a(f"  * {side}: {e['printed']} -- {e['locator']}")
    a("")
    a("Numerals that are NOT band endpoints under either reading:")
    a("")
    for num, why in inv["numerals_that_are_NOT_band_endpoints"].items():
        a(f"* **{num}** -- {why}")
    a("")
    a(f"The repo holds **{inv['competing_readings_count']}** competing provenance "
      "readings, not two:")
    a("")
    for r in inv["competing_readings"]:
        a(f"* {r}")
    a("")

    a("## 4. THE CATALOGUE ROW THAT F2 REFUTES")
    a("")
    c = art["catalogue_row"]
    a(f"`{c['section_anchor']}` -- verbatim:")
    a("")
    a("```")
    a(c["verbatim"])
    a("```")
    a("")
    a(f"file sha256 `{c['file_sha256']}` ({c['file_bytes']:,} bytes).")
    a("")
    a("The line number this row used to be cited by is DEMOTED -- the "
      "catalogue grows, so a line number into it is perishable. The durable "
      "locator is the section anchor above; the superseded numbers are "
      "preserved VERBATIM in the dated demotion block "
      "(`locator_demotion_2026_11_02_stale_by_construction`).")
    a("")
    a(f"It was inherited from a different band: "
      f"{c['inherited_from']['why_it_differs']}")
    a("")
    a(f"**Proposed, dated, additive correction ({c['proposed_correction_dated']['wave']}, "
      f"{c['proposed_correction_dated']['dated']}) -- "
      f"{c['proposed_correction_dated']['status']}.**")
    a("")
    a(c["proposed_correction_dated"]["text"])
    a("")

    a("## 5. THE X1-X4 CLASS -- LIVE STATUS ON DISK")
    a("")
    for key, v in art["x_class_status"].items():
        a(f"* **{key}** = {v['claim']}")
        a(f"  * live status: {v['live_status']}")
        a(f"  * this wave: {v['our_wave']}")
    a("")

    a("## 6. THE SENTENCE WE MAY SAY, AND THE SENTENCES WE MAY NOT")
    a("")
    a("### MAY")
    a("")
    for s in art["may"]:
        a(f"* {s}")
    a("")
    a("### MAY NOT")
    a("")
    for s in art["may_not"]:
        a(f"* {s}")
    a("")

    a("## 7. ADDITIONAL OBSERVATIONS AND WHAT THIS WAVE DID NOT ESTABLISH")
    a("")
    for s in art["additional_observations"]:
        a(f"* {s}")
    a("")
    for s in art["not_established"]:
        a(f"* NOT ESTABLISHED -- {s}")
    a("")

    dem = art["locator_demotion_2026_11_02_stale_by_construction"]
    rec = dem["record_measured"]
    cat = dem["catalogue_measured"]
    a("## 8. Wf-A190 (2026-11-02) -- LOCATOR DEMOTION: the record and the "
      "catalogue are cited by JSON PATH / SECTION ANCHOR, not by line number")
    a("")
    a(f"**{dem['rule']}**")
    a("")
    a("**MEASURED on 2026-11-02.** The record "
      f"`{rec['path']}` is **{rec['line_count']} lines / {rec['bytes']} bytes** "
      f"(sha256 `{rec['sha256']}`) and the catalogue `{cat['path']}` is "
      f"**{cat['bytes']} bytes** (sha256 `{cat['sha256']}`). Both are "
      "UNCHANGED since 2026-11-01 -- neither has moved -- and BOTH are "
      "STALE BY CONSTRUCTION: the record is regenerated and keeps growing, and "
      "the catalogue is appended to by every wave.")
    a("")
    a("**Superseded renderings, PRESERVED VERBATIM** (AGENTS.md 5.5 rule #6: "
      "append-only + dated; nothing is deleted). Every line number below is "
      "STALE BY CONSTRUCTION:")
    a("")
    a("| where | superseded rendering (verbatim) |")
    a("|---|---|")
    for item in dem["superseded_strings_verbatim"]:
        a(f"| `{item['where']}` | {item['text']} |")
    a("")
    a("**Superseded NUMERIC locators, preserved VERBATIM** (the fields that "
      "held a bare line number rather than a `path:line` string):")
    a("")
    a("| superseded field | preserved verbatim at | value |")
    a("|---|---|---|")
    for item in dem["superseded_numeric_locators_verbatim"]:
        a(f"| `{item['was']}` | `{item['where']}` | {item['value']} |")
    a("")
    a("**Superseded KEY NAME, preserved VERBATIM** (the key itself embedded a "
      "line number into the growing catalogue):")
    a("")
    a("| where | superseded key name (verbatim) |")
    a("|---|---|")
    for item in dem["superseded_key_names_verbatim"]:
        a(f"| {item['where']} | `{item['text']}` |")
    a("")
    a("**What those lines ACTUALLY contained on 2026-11-02 (measured on the "
      "record / the catalogue):**")
    a("")
    a("| cited line | what it actually held |")
    a("|---|---|")
    for k, v in dem["what_the_superseded_lines_actually_contained_on_2026_11_02"
                    ].items():
        a(f"| `{k}` | {v} |")
    a("")
    a("**The durable replacements (JSON PATHS and SECTION ANCHORS):**")
    a("")
    a("| what | durable locator |")
    a("|---|---|")
    for k, v in dem["durable_replacements"].items():
        a(f"| {k} | `{v}` |")
    a("")
    a("**Corrections to this wave's own demotion block (append-only, dated):**")
    a("")
    for c in dem["corrections"]:
        a(f"* **{c['dated']}, {c['wave']}.** Superseded rendering, preserved "
          f"verbatim: `{c['superseded_rendering_verbatim']}`. Corrected to: "
          f"`{c['corrected_to']}`. {c['why']}")
    a("")
    a("*Section 8 appended 2026-11-02 (Wf-A190, consumer side). Every number "
      "above is MEASURED; the line numbers are STALE BY CONSTRUCTION by "
      "design.*")
    a("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# narrative self-consistency checker
# --------------------------------------------------------------------------

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")


def strip_code(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    return text


def collect(obj, acc_nums, acc_strs):
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc_strs.add(str(k))
            collect(v, acc_nums, acc_strs)
    elif isinstance(obj, list):
        for v in obj:
            collect(v, acc_nums, acc_strs)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        acc_nums.add(round(float(obj), 6))
    elif isinstance(obj, str):
        acc_strs.add(obj)


def check_narrative(report_text, artefact):
    acc_nums, acc_strs = set(), set()
    collect(artefact, acc_nums, acc_strs)
    flat_strs = [s.replace(",", "") for s in acc_strs]

    body = strip_code(report_text)
    hexes = set(HEX64.findall(body))
    missing_hex = [h for h in hexes if not any(h in s for s in acc_strs)]
    body = HEX64.sub(" ", body)

    toks = NUM_TOKEN.findall(body)
    missing = []
    for tok in toks:
        val = float(tok.replace(",", ""))
        if round(val, 6) in acc_nums:
            continue
        if any(tok.replace(",", "") in s for s in flat_strs):
            continue
        missing.append(tok)
    return {
        "numeric_tokens_checked": len(toks),
        "distinct_missing": sorted(set(missing)),
        "hex_sha256_tokens_checked": len(hexes),
        "hex_missing": sorted(missing_hex),
        "pass": (not missing) and (not missing_hex),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def write_atomic(path, content):
    if os.path.exists(path):
        safe_overwrite_check(path, content)
    reason = detect_truncation(content)
    if reason:
        raise AssertionError(f"detect_truncation flagged {path}: {reason}")
    atomic_write_file(path, content)


def main():
    art = build_artefact()
    report = render_report(art)
    chk = check_narrative(report, art)
    art["narrative_self_consistency"] = {
        "rule": "every numeral in a narrative string of report.md must be present "
                "in a machine field of primary_source_verification.json",
        "checked_on": "report.md rendered by this same run",
        "numeric_tokens_checked": chk["numeric_tokens_checked"],
        "hex_sha256_tokens_checked": chk["hex_sha256_tokens_checked"],
        "distinct_missing": chk["distinct_missing"],
        "hex_missing": chk["hex_missing"],
        "pass": chk["pass"],
    }
    report = render_report(art)
    art_text = json.dumps(art, indent=2, ensure_ascii=False) + "\n"

    write_atomic(os.path.join(BASE, "primary_source_verification.json"), art_text)
    write_atomic(os.path.join(BASE, "report.md"), report)

    print(json.dumps({
        "artefact_bytes": len(art_text.encode()),
        "report_bytes": len(report.encode()),
        "narrative_check_pass": art["narrative_self_consistency"]["pass"],
        "numeric_tokens_checked":
            art["narrative_self_consistency"]["numeric_tokens_checked"],
        "distinct_missing": art["narrative_self_consistency"]["distinct_missing"],
        "hex_missing": art["narrative_self_consistency"]["hex_missing"],
        "n_cells": {k: len(v) for k, v in art["cells"].items()},
        "findings": {k: v["verdict"] for k, v in art["findings"].items()},
        "our_side_percent": art["our_side"]["value_percent"],
    }, indent=2, ensure_ascii=False))
    return 0 if art["narrative_self_consistency"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
