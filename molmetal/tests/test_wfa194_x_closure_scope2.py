#!/usr/bin/env python
"""Wf-A194 (2026-11-02) -- X-class closure guard for the scope-2 axis.

CPU-only, 0 GPU, no network, no producer run: every assertion is a static
read of a file on disk.  ``WFA194_SCOPE_ROOT`` overrides the repository root so
that the guard can be pointed at a /tmp COPY -- that is how the bite proof is
run (break the copy, watch this file FAIL).

What is guarded
---------------
1. NO LIVE position may carry a refuted band-provenance rendering.  A position
   is LIVE when it sits in the part of a document that is NOT a dated
   CORRECTION section, a must-not-assert list, a refutation quote of a
   superseded wording, a test constant or a retrieval record.
2. EVERY rendering this wave replaced must survive VERBATIM in a DATED
   position, and must be ABSENT from the live part.
3. The wfa161 PRODUCER must emit the reading-safe item 2, must carry the
   superseded item 2 in a NAMED DATED CONSTANT, and the published artefact's
   live item 2 must be byte-identical to the string the producer emits.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re

import pytest

SCOPE_ROOT = pathlib.Path(
    os.environ.get("WFA194_SCOPE_ROOT", "/home/hugo/codes/mol-metal")
)

SYNTHESIS = "molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md"
REFUTATION_MD = "molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.md"
REFUTATION_JSON = "molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.json"
WFA171 = "molmetal/reports/wfa171_frontier_postproc/report.md"
WFA161_REPORT = "molmetal/reports/wfa161_frontier_labelling/report.md"
WFA161_PRODUCER = "molmetal/scripts/wfa161_frontier_labelling.py"

# --------------------------------------------------------------------------
# The refuted renderings (defect classes X1, X2, X3, X4, MIRROR).
#
# X1  "no endpoint is a connected-component count" / "THE BAND IS NOT A
#     CONNECTIVITY BAND" -- TRUE under Reading 1, FALSE under Reading 2.
# X2  "the band is measured on the published POST-PROCESSED stream".
# X3  "their Connectivity is computed AFTER retaining the largest component" /
#     the blanket "computed on the atom-count largest connected component
#     after RDKit sanitization (EF-TALFM)".  The `(?<!NOT )` look-behind keeps
#     the REPAIRED rendering ("are NOT computed on ...") out of the pattern.
# X4  "the band is minted here: lower = GeoLDM, upper = HLTF" and the
#     membership form "a band in which the *lowest* member ... is ...".
# MIRROR  an unconditional "both arms are RAW" / "band as a whole is RAW".
# --------------------------------------------------------------------------
FORBIDDEN = [
    ("X1", r"no endpoint is a connected-component count"),
    ("X1", r"NEITHER is a connected-component count"),
    ("X1", r"is not a connectivity band"),
    ("X1", r"None is a connected-component count"),
    ("X2", r"band is measured on the published POST-PROCESSED stream"),
    ("X3", r"their Connectivity is computed AFTER retaining the largest component"),
    ("X3", r"(?<!NOT )computed on the atom-count largest connected component after RDKit sanitization \(EF-TALFM\)"),
    ("X4", r"band is minted here"),
    ("X4", r"lower = GeoLDM"),
    ("X4", r"upper = HLTF"),
    ("X4", r"in which the \*lowest\* member"),
    ("MIRROR", r"both arms are RAW"),
    ("MIRROR", r"both endpoints are RAW"),
    ("MIRROR", r"band as a whole is RAW"),
]

# A rendering that carries one of these is reading-CONDITIONAL, i.e. it names
# a reading instead of asserting the band.
QUALIFIERS = (
    "Reading 1", "Reading 2", "reading 1", "reading 2", "THIS reading",
    "NOT settled on disk", "not settled on disk",
    "reading-conditional", "per reading", "per-cell AND reading-conditional",
    "UNDER BOTH readings", "under BOTH readings", "under both readings",
    "WITHDRAWN as unsettled", "TRUE under", "FALSE under", "true under",
    "false under", "two competing readings", "TWO competing readings",
)

# An explicitly ATTRIBUTED quotation of the matrix's (superseded) wording.
QUOTE_MARKERS = ("on-disk matrix calls", "the on-disk matrix calls")

# A DEFECT-RECORD position quotes the defective rendering in order to NAME it:
# a defect-table row, a must-not-assert entry, a "contradicted / WRONG /
# CLOSED" annotation.  Such a quotation identifies the defect; it is not a
# claim, and it is not what this guard polices.
DEFECT_MARKERS = (
    "**contradicted**", "**WRONG**", "**CLOSED", "**FORBIDDEN",
    "Never assert", "Never say", "never publish", "REFUTED on disk",
)

DATED_HEADING = re.compile(
    r"^##\s+(?:\d+\.\s+)?Wf-A\d+[^\n]*CORRECTION[^\n]*$", re.MULTILINE
)


def _read(rel: str) -> str:
    p = SCOPE_ROOT / rel
    assert p.exists(), f"missing scope file: {p}"
    return p.read_text(encoding="utf-8")


def _split_live_dated(text: str) -> tuple[str, str]:
    """Split a markdown document into (live part, dated part).

    Everything from the FIRST dated CORRECTION heading onwards is dated: the
    dated sections are where superseded renderings are preserved on purpose.
    """
    m = DATED_HEADING.search(text)
    if not m:
        return text, ""
    return text[: m.start()], text[m.start():]


def _unquote(block: str) -> str:
    """Strip the markdown block-quote marker from every line."""
    out = []
    for line in block.split("\n"):
        out.append(re.sub(r"^\s*(?:[-*+]\s*)?>\s?", "", line))
    return "\n".join(out)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset)


# --------------------------------------------------------------------------
# The STATEMENT scope of a match.
#
# The exemption window used to be a fixed +-300 CHARACTER radius.  A radius
# cannot tell "the qualifier belongs to THIS statement" from "the qualifier
# belongs to the statement NEXT DOOR": a rendering inserted immediately after
# a reading-conditional line inherited that line's qualifier and was
# silently exempt.  (Wf-A202, 2026-11-02.)  The window is therefore
# STRUCTURAL -- it is the enclosing STATEMENT: the span between the nearest
# SENTENCE TERMINATORS on either side, clipped at the hard markdown
# boundaries (blank line / heading / list item / table row).
#
# A sentence terminator is a full stop that is not part of an ellipsis, or an
# exclamation / question mark, followed by whitespace (or the end of text).
# "..." is NOT a terminator: the refutation record quotes the matrix's wording
# across an ellipsis ("... IS NOT A CONNECTIVITY NUMBER AT ALL ... NEITHER is
# a connected-component count"), and the attribution marker that exempts that
# quotation ("which the on-disk matrix calls") lives in the SAME sentence.
# --------------------------------------------------------------------------
_TERMINATOR = re.compile(r"(?<!\.)\.(?!\.)(?=\s)|(?<!\.)\.(?=$)|[!?](?=\s)|[!?](?=$)")


def _is_hard_boundary(line: str) -> bool:
    """True when a markdown line opens a NEW statement unit."""
    if not line.strip():
        return True                      # blank line: paragraph break
    if re.match(r"^#{1,6}\s", line):     # heading
        return True
    if re.match(r"^\s*[-*+]\s", line):  # bullet
        return True
    if re.match(r"^\s*\d+[.)]\s", line):  # numbered item
        return True
    if re.match(r"^\s*\|", line):       # table row
        return True
    return False


def _last_terminator(text: str, lo: int, hi: int) -> int:
    """The index of the LAST sentence terminator in text[lo:hi], or -1."""
    last = -1
    for m in _TERMINATOR.finditer(text, lo, hi):
        last = m.start()
    return last


def _next_terminator(text: str, lo: int, hi: int) -> int:
    """The index of the FIRST sentence terminator in text[lo:hi], or -1."""
    m = _TERMINATOR.search(text, lo, hi)
    return m.start() if m else -1


def _statement_window(text: str, start: int, end: int) -> str:
    """The STATEMENT that contains [start, end).

    The statement is the enclosing SENTENCE -- the span between the nearest
    sentence terminators on either side -- clipped at hard markdown
    boundaries (blank line / heading / list item / table row).  It is never
    larger than one sentence, so a qualifier, a quote marker or a defect
    marker can only exempt a rendering that lives in the SAME statement.
    """
    lines = text.split("\n")
    starts: list[int] = []
    off = 0
    for ln in lines:
        starts.append(off)
        off += len(ln) + 1

    def line_index(pos: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    first = line_index(start)
    last = line_index(max(start, end - 1))

    # ---- backward: the last terminator before the match, else whole lines --
    term = _last_terminator(text, starts[first], start)
    if term >= 0:
        lo = term + 1
    else:
        lo = starts[first]
        i = first - 1
        while i >= 0:
            if _is_hard_boundary(lines[i]):
                break
            term = _last_terminator(text, starts[i], starts[i] + len(lines[i]))
            if term >= 0:
                lo = term + 1
                break
            lo = starts[i]
            i -= 1

    # ---- forward: the first terminator after the match, else whole lines ---
    term = _next_terminator(text, end, starts[last] + len(lines[last]))
    if term >= 0:
        hi = term
    else:
        hi = starts[last] + len(lines[last])
        i = last + 1
        while i < len(lines):
            if _is_hard_boundary(lines[i]):
                break
            term = _next_terminator(text, starts[i], starts[i] + len(lines[i]))
            if term >= 0:
                hi = term
                break
            hi = starts[i] + len(lines[i])
            i += 1
    return text[lo:hi]


def _is_source_quote_line(line: str) -> bool:
    """A markdown block-quote line is a VERBATIM quotation of a source."""
    return bool(re.match(r"^\s*(?:[-*+]\s*)?>", line))


def _normalize_ws(text: str) -> tuple[str, list[int]]:
    """Collapse every whitespace run to ONE space, keeping an offset map.

    A refuted rendering may straddle a line break (e.g. "is not a\n"
    "connectivity band"), so the FORBIDDEN patterns are matched against a
    whitespace-normalised copy.  The map lets every match be reported at its
    ORIGINAL offset, hence its original line number.
    """
    out: list[str] = []
    offs: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if out:
                out.append(" ")
                offs.append(i)
            i = j
            continue
        out.append(c)
        offs.append(i)
        i += 1
    return "".join(out), offs


def _assert_no_live_forbidden(rel: str, text: str, exempt_statements: tuple[str, ...] = ()) -> None:
    """Assert that no LIVE position of `text` carries a refuted rendering.

    `exempt_statements` is a tuple of STRUCTURAL anchors: a match whose own
    STATEMENT carries one of them is exempt.  (It used to be a tuple of bare
    line numbers, which is a perishable locator -- a line number moves the
    moment anything above it is inserted, and it says nothing about WHICH
    statement was meant.)
    """
    live, _dated = _split_live_dated(text)
    lines = live.split("\n")
    norm, offs = _normalize_ws(live)
    for cls, pat in FORBIDDEN:
        rx = re.compile(pat)
        for m in rx.finditer(norm):
            o_start = offs[m.start()]
            o_end = offs[m.end() - 1] + 1
            ln = _line_of(live, o_start)
            window = _statement_window(live, o_start, o_end)
            if any(a in window for a in exempt_statements):
                continue
            if any(q in window for q in QUALIFIERS):
                continue
            if any(q in window for q in QUOTE_MARKERS):
                continue
            if any(q in window for q in DEFECT_MARKERS):
                continue
            if _is_source_quote_line(lines[ln]):
                continue
            raise AssertionError(
                f"{rel}: LIVE X-class rendering ({cls}) at line {ln + 1}: "
                f"{lines[ln].strip()[:160]!r}\n"
                f"  statement scope: {window.strip()[:240]!r}"
            )


# --------------------------------------------------------------------------
# 1. NO LIVE X-class rendering
# --------------------------------------------------------------------------
def test_synthesis_has_no_live_x_class_rendering():
    _assert_no_live_forbidden(SYNTHESIS, _read(SYNTHESIS))


def test_wfa171_has_no_live_x_class_rendering():
    _assert_no_live_forbidden(WFA171, _read(WFA171))


def test_refutation_md_has_no_live_x_class_rendering():
    # refutation.md is a REFUTATION record: its X-class wording is a
    # quotation of the matrix's superseded rendering, introduced by the
    # attribution marker "which the on-disk matrix calls".  The marker is
    # what exempts it -- the wording itself is deliberately preserved.
    _assert_no_live_forbidden(REFUTATION_MD, _read(REFUTATION_MD))


def test_refutation_json_has_no_live_x_class_rendering():
    doc = json.loads(_read(REFUTATION_JSON))
    dated_keys = {
        "wfa187_supersession_2026_11_01",
        "wfa194_closure_2026_11_02",
        "wfa202_closure_2026_11_02",
    }

    def walk(node, path, in_dated):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k,
                     in_dated or k in dated_keys)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]", in_dated)
        elif isinstance(node, str) and not in_dated:
            for cls, pat in FORBIDDEN:
                if re.search(pat, _normalize_ws(node)[0]):
                    window = node
                    if not any(q in window for q in QUALIFIERS) and not any(
                        q in window for q in QUOTE_MARKERS
                    ):
                        raise AssertionError(
                            f"{REFUTATION_JSON}: LIVE X-class rendering ({cls}) "
                            f"at {path}: {node[:160]!r}"
                        )

    walk(doc, "", False)


def test_wfa161_report_has_no_live_x_class_rendering():
    _assert_no_live_forbidden(WFA161_REPORT, _read(WFA161_REPORT))


# --------------------------------------------------------------------------
# 2. Every replaced rendering survives VERBATIM in a DATED position
# --------------------------------------------------------------------------
SYNTHESIS_REPLACED = {
    "s0-verdict": (
        "direction, class, measurement and population). The repo's published "
        "**92.7–94 % band is not a\n"
        "connectivity band at all** (its endpoints are a Validity&Uniqueness rate and a "
        "six-check PoseBusters\n"
        "pose verdict) and, per Wf-A168, is **cell-heterogeneous in post-processing**: "
        "some cells are RAW,\n"
        "some are POST-PROCESSED, so the standing \"the band is post-processed\" break "
        "cannot be asserted\n"
        "blanket-ward in either direction."
    ),
    "s1_3-lead-in": (
        "`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` →\n"
        "`band_provenance.headline_finding`: **\"THE BAND IS NOT A CONNECTIVITY BAND.\"**"
    ),
    "s2_3-row": (
        "| CONNECTIVITY (the band) | **NO** | **NO** | "
        "the band is not a connectivity band (A158) |"
    ),
    "s5-may-item-1": (
        "1. **\"The 92.7–94 % band is not a connectivity band.\"** Cite A158: "
        "the 92.7 % endpoint is HLTF Table\n"
        "   1(b) \"Val&Uniq (%)\" GeoLDM column (a per-molecule "
        "RDKit-sanitisable-AND-non-duplicate rate over\n"
        "   10,000 QM9 samples); the 94 % endpoint is HLTF Table 1(b) \"PB-Valid (%)\" "
        "HLTF column (a six-check\n"
        "   PoseBusters pose verdict over 10,000 QM9 samples); a third 92.7 % is "
        "YuelBond's per-bond-type\n"
        "   classification F1. None is a connected-component count."
    ),
    "s5-may-not-item-8": (
        "8. **Never say the frontier's connectivity is computed after "
        "largest-component retention.** Wf-A161's\n"
        "   `report.md:139` says exactly that for EF-TALFM and is **WRONG** "
        "(A158 + the EF-TALFM branch)."
    ),
    "s6-d1-row": (
        "| D1 | `molmetal/reports/wfa161_frontier_labelling/report.md:139` | "
        "\"The frontier's connectivity rates are computed on the atom-count largest "
        "connected component after RDKit sanitization (EF-TALFM)\" — "
        "**contradicted**: EF-TALFM's 92.98 % is RAW (A158 matrix + EF-TALFM branch) "
        "| MEDIUM |"
    ),
}

WFA171_REPLACED = {
    "s3-closing": (
        "It is also a band in which the *lowest* member (92.7) is a **baseline's** "
        "number on one dataset and the\n"
        "*highest* (94.33) is a *different metric* on another. This is the fourth "
        "independent comparability break,\n"
        "and it is the repo's own band, not the frontier's."
    ),
}

WFA161_OLD_ITEM_2 = (
    "2. **Post-processing.** The frontier's connectivity rates are computed on the "
    "atom-count largest connected component after RDKit sanitization (EF-TALFM) or with "
    "no stated fragment handling at all (GEOM-DD, GGFlow). Our 92.7-94 % band comes "
    "from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, "
    "so it is not on the frontier's scale."
)


@pytest.mark.parametrize("key", sorted(SYNTHESIS_REPLACED))
def test_synthesis_replaced_rendering_survives_verbatim_in_a_dated_position(key):
    text = _read(SYNTHESIS)
    live, dated = _split_live_dated(text)
    old = SYNTHESIS_REPLACED[key]
    assert dated, "synthesis.md has no dated CORRECTION section"
    assert old in _unquote(dated), (
        f"{SYNTHESIS}: the replaced rendering {key!r} does NOT survive VERBATIM in a "
        "dated position"
    )
    assert old not in live, (
        f"{SYNTHESIS}: the replaced rendering {key!r} is still present in the LIVE part"
    )


@pytest.mark.parametrize("key", sorted(WFA171_REPLACED))
def test_wfa171_replaced_rendering_survives_verbatim_in_a_dated_position(key):
    text = _read(WFA171)
    live, dated = _split_live_dated(text)
    old = WFA171_REPLACED[key]
    assert dated, "wfa171 report.md has no dated CORRECTION section"
    assert old in _unquote(dated), (
        f"{WFA171}: the replaced rendering {key!r} does NOT survive VERBATIM in a "
        "dated position"
    )
    assert old not in live, (
        f"{WFA171}: the replaced rendering {key!r} is still present in the LIVE part"
    )


def test_wfa161_old_item_2_survives_verbatim_in_a_dated_position():
    report = _read(WFA161_REPORT)
    live, dated = _split_live_dated(report)
    assert dated, "the wfa161 report has no dated CORRECTION section"
    assert WFA161_OLD_ITEM_2 in _unquote(dated), (
        f"{WFA161_REPORT}: the superseded item 2 does NOT survive VERBATIM in a dated "
        "position"
    )
    assert WFA161_OLD_ITEM_2 not in live, (
        f"{WFA161_REPORT}: the superseded item 2 is still present in the LIVE part"
    )


# --------------------------------------------------------------------------
# 3. Producer <-> artefact parity for the wfa161 axis
# --------------------------------------------------------------------------
def _producer_constants() -> dict:
    tree = ast.parse(_read(WFA161_PRODUCER))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
            out[node.targets[0].id] = value
    return out


def test_producer_carries_every_dated_section_as_a_named_constant():
    consts = _producer_constants()
    for name in (
        "WFA176_CORRECTION_SECTION",
        "WFA176_CORRECTION2_SECTION",
        "WFA183_CORRECTION_SECTION",
        "WFA187_CORRECTION_SECTION",
        "WFA194_CORRECTION_SECTION",
    ):
        assert isinstance(consts.get(name), str) and consts[name].strip(), (
            f"{WFA161_PRODUCER}: the dated section constant {name} is missing"
        )


def test_producer_preserves_the_superseded_item_2_in_a_dated_constant():
    consts = _producer_constants()
    assert WFA161_OLD_ITEM_2 in consts["WFA176_CORRECTION_SECTION"], (
        f"{WFA161_PRODUCER}: the superseded item 2 is not preserved VERBATIM in "
        "WFA176_CORRECTION_SECTION"
    )


def test_producer_emits_the_reading_safe_item_2_and_the_artefact_matches_it():
    src = _read(WFA161_PRODUCER)
    tree = ast.parse(src)
    emitted = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.startswith("2. **Post-processing.**")
    ]
    assert len(emitted) == 1, (
        f"{WFA161_PRODUCER}: expected exactly one emitted '## Comparability breaks' "
        f"item 2, found {len(emitted)}"
    )
    item2 = emitted[0]
    assert "are NOT computed on the atom-count largest connected component" in item2, (
        f"{WFA161_PRODUCER}: the emitted item 2 does not carry the X3 repair"
    )
    assert "Reading 1" in item2 and "Reading 2" in item2, (
        f"{WFA161_PRODUCER}: the emitted item 2 is not reading-conditional"
    )
    assert WFA161_OLD_ITEM_2 not in item2

    report = _read(WFA161_REPORT)
    live, _dated = _split_live_dated(report)
    live_item2 = [
        ln for ln in live.split("\n") if ln.startswith("2. **Post-processing.**")
    ]
    assert len(live_item2) == 1, (
        f"{WFA161_REPORT}: expected exactly one live '## Comparability breaks' item 2"
    )
    assert live_item2[0] == item2, (
        f"{WFA161_REPORT}: the published live item 2 is NOT byte-identical to the "
        "string the producer emits -- producer/artefact parity is broken"
    )


def test_refutation_json_gains_one_dated_key_and_changes_no_existing_leaf():
    doc = json.loads(_read(REFUTATION_JSON))
    assert "wfa194_closure_2026_11_02" in doc, (
        f"{REFUTATION_JSON}: the dated closure key is missing"
    )
    # Wf-A202 (2026-11-02) added a SECOND dated key, for the byte-identity
    # repair (R3), the census pointer (R4) and the two cross-wave findings.
    assert "wfa202_closure_2026_11_02" in doc, (
        f"{REFUTATION_JSON}: the Wf-A202 dated key is missing"
    )

    def leaves(node, path=""):
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
    # the numeric-leaf path set is unchanged by this wave: every leaf in the
    # dated keys is a STRING.
    assert len(numeric) == 81, (
        f"{REFUTATION_JSON}: the numeric-leaf count changed ({len(numeric)} != 81)"
    )
    for key in ("wfa194_closure_2026_11_02", "wfa202_closure_2026_11_02"):
        new_key = doc[key]
        for k, v in new_key.items():
            assert isinstance(v, str), (
                f"{REFUTATION_JSON}: the dated key leaf {key}.{k!r} is not a STRING"
            )
