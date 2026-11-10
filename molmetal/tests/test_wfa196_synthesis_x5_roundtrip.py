"""Wf-A196 -- the X5 reading-COUNT scope repair + the Wf-A193 D1/D2/D3 residuals.

WHAT THIS FILE GUARDS
---------------------
The X5 defect class: a sentence that puts a NUMBER on the size of the contest
(the number of competing provenance readings of the "92.7-94 %" band's source
cells) asserts a settled count that the repo does NOT settle.  Writing a LARGER
number is the SAME defect in the other direction -- how many DISTINCT readings
exist is itself a judgement call.  The honest repair asserts NO settled count.

The four invariants that hold under EVERY reading must stay present, the
superseded renderings must survive VERBATIM in a DATED position, and the
producer must reproduce the published artefact byte-for-byte (a re-run must not
rewrite or drop a dated section).

CPU-only.  No GPU.  No git mutation.  The producer is rendered into a TEMP dir;
the published artefact is only ever READ.

`WFA196_SYNTHESIS_MD` overrides the .md under test (used by the bite proof:
point it at a mutated COPY and the guards below must FAIL).
"""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path("/home/hugo/codes/mol-metal")
MD = pathlib.Path(
    os.environ.get("WFA196_SYNTHESIS_MD")
    or (REPO / "molmetal/reports/wfa175_synthesis/synthesis.md")
)
JSON_PATH = REPO / "molmetal/reports/wfa175_synthesis/synthesis.json"
PRODUCER = REPO / "molmetal/reports/wfa175_synthesis/_write_md.py"

# --------------------------------------------------------------------------
# The X5 count-claim patterns.  A NUMERAL immediately in front of "reading(s)"
# -- optionally qualified by "competing" / "provenance" -- is a COUNT CLAIM
# about the size of the contest.  "the NUMBER of competing readings is itself
# NOT settled on disk" does NOT match (no numeral), and neither does
# "Under BOTH/EVERY reading".
# --------------------------------------------------------------------------
COUNT_CLAIM_RES = (
    re.compile(
        r"(?i)\b(?:two|three|four|five|six|2|3|4|5|6)\s+"
        r"(?:competing\s+|provenance\s+|distinct\s+|independent\s+|separate\s+)*"
        r"readings\b"
    ),
    re.compile(r"(?i)\b(?:two|three|four)\s+competing\b"),
)

#: The marker a dated section must carry (directly, or a LATER dated section
#: must carry) for its count claims to count as superseded.
COUNT_NOT_SETTLED_MARKER = "NUMBER of competing readings is itself NOT settled on disk"

#: The FOUR invariants, quoted VERBATIM from
#: `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` ->
#: `band_provenance.competing_reading_and_what_this_does_not_settle`.
#: `the_only_claims_that_may_be_published`.
FOUR_INVARIANTS = {
    "i_the_band_is_external": "The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a FRONTIER composite, not our measurement.",
    "ii_it_is_not_one_homogeneous_quantity": "It is NOT one homogeneous quantity: even under Reading A its two endpoints are two different observables (a sanitisation-and-duplication rate and a PoseBusters validity verdict -- HLTF states NO check count for the latter, NOT-STATED-IN-SOURCE), and under Reading B one endpoint is a raw connectivity count while the other is a post-processed composite.",
    "iii_it_is_not_on_our_scale": "It is NOT on our scale.  OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.",
    "iv_the_source_cells_are_not_settled": "The source cells of the band are NOT settled on disk.",
}

#: The superseded renderings that must survive VERBATIM in a DATED position.
SUPERSEDED_VERBATIM = {
    # D2 -- the LIVE summary-table row that carried the refuted X1 claim.
    "x1_summary_row": '| A2 | A-published-number | The quoted "92.7-94 %" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict) |',
    # X5 -- the 2026-11-01 renderings that put a NUMBER on the contest.
    "x5_clause": "holds TWO competing readings of which frontier cells the band names",
    "x5_row_label": "TWO COMPETING readings (`band_provenance` vs `sources`.`EF-TALFM`).",
    "x5_status": "TWO COMPETING readings of the frontier cells it names",
    "x5_evidence_tail": "The two readings do not even agree on the numerals",
}

#: The numeric-leaf map of `synthesis.json`, FROZEN against the pre-Wf-A196
#: published artefact (MEASURED 2026-11-02: 24 numeric leaves, none moved).
FROZEN_NUMERIC_LEAF_COUNT = 24
FROZEN_NUMERIC_LEAF_DIGEST = (
    "51c634a4a6fcdc7bbd2400dec0322a360c04863d61cc2cab8224b08bf660f72e"
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
DATED_HEADING_RE = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _lines():
    return MD.read_text(encoding="utf-8").split("\n")


def _sections(lines):
    """Every heading with the line range it owns (to the next heading of the
    same or a higher level)."""
    heads = []
    for i, ln in enumerate(lines):
        m = HEADING_RE.match(ln)
        if m:
            heads.append((i, len(m.group(1)), m.group(2)))
    out = []
    for j, (i, lvl, txt) in enumerate(heads):
        end = len(lines)
        for k in range(j + 1, len(heads)):
            if heads[k][1] <= lvl:
                end = heads[k][0]
                break
        out.append({"i": i, "level": lvl, "text": txt, "end": end,
                    "date": (DATED_HEADING_RE.search(txt).group(1)
                             if DATED_HEADING_RE.search(txt) else "")})
    return out


def _ancestors(lines, idx, secs):
    """The heading sections that contain line `idx` (nearest first)."""
    out = []
    for s in secs:
        if s["i"] <= idx < s["end"]:
            out.append(s)
    return out


def _is_dated(lines, idx, secs):
    return any(DATED_HEADING_RE.search(s["text"]) for s in _ancestors(lines, idx, secs))


def _count_claims(text):
    return [m.group(0) for r in COUNT_CLAIM_RES for m in r.finditer(text)]


def _numeric_leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            for p, val in _numeric_leaves(v, prefix + "." + str(k)):
                yield p, val
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            for p, val in _numeric_leaves(v, prefix + "[" + str(i) + "]"):
                yield p, val
    else:
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            yield prefix, obj


# --------------------------------------------------------------------------
# (a) no LIVE COUNT CLAIM about the size of the contest survives
# --------------------------------------------------------------------------
def test_no_live_count_claim_about_the_size_of_the_contest():
    lines = _lines()
    secs = _sections(lines)
    live = []
    for i, ln in enumerate(lines):
        if _count_claims(ln) and not _is_dated(lines, i, secs):
            live.append("line %d: %s" % (i + 1, ln.strip()[:200]))
    assert not live, (
        "LIVE count claim(s) about the size of the contest survive in %s:\n%s"
        % (MD, "\n".join(live))
    )


def test_every_dated_count_claim_is_superseded():
    """A dated section may carry a count claim only if it declares the count
    NOT settled itself, or a LATER dated section does."""
    lines = _lines()
    secs = [s for s in _sections(lines) if DATED_HEADING_RE.search(s["text"])]
    offenders = []
    for s in secs:
        region = "\n".join(lines[s["i"]:s["end"]])
        if not _count_claims(region):
            continue
        self_ok = COUNT_NOT_SETTLED_MARKER.lower() in region.lower()
        later_ok = any(
            t["date"] > s["date"]
            and COUNT_NOT_SETTLED_MARKER.lower()
            in "\n".join(lines[t["i"]:t["end"]]).lower()
            for t in secs if t is not s
        )
        if not (self_ok or later_ok):
            offenders.append("section %r (date %s)" % (s["text"], s["date"]))
    assert not offenders, (
        "dated section(s) carry a count claim that nothing supersedes:\n%s"
        % "\n".join(offenders)
    )


# --------------------------------------------------------------------------
# (b) the OLD text survives VERBATIM in a DATED position
# --------------------------------------------------------------------------
def test_superseded_renderings_survive_verbatim_in_a_dated_position():
    lines = _lines()
    secs = _sections(lines)
    text = MD.read_text(encoding="utf-8")
    for name, snippet in SUPERSEDED_VERBATIM.items():
        assert snippet in text, "superseded rendering %r was DELETED" % name
        idx = text.index(snippet)
        line_no = text[:idx].count("\n")
        assert _is_dated(lines, line_no, secs), (
            "superseded rendering %r survives OUTSIDE a dated position" % name
        )


def test_the_wf177_dated_sections_are_present_byte_for_byte():
    """D1 -- the published '## Wf-A177 CORRECTION' body and its
    '### Wf-A177 locator addendum' subsection must be present verbatim."""
    text = MD.read_text(encoding="utf-8")
    for marker in (
        "## Wf-A177 CORRECTION (2026-11-01) — the A2 claim is contested on disk",
        "### Wf-A177 locator addendum (2026-11-01) — the cited line numbers are pre-Wf-A177",
        "**The ROBUST locators are the JSON PATHS, not the line numbers**",
        "Nothing above this subsection was edited or removed (AGENTS.md §5.5 rule #6 — append-only).",
    ):
        assert marker in text, "the published Wf-A177 dated section lost %r" % marker


# --------------------------------------------------------------------------
# (c) the four invariants are STILL PRESENT
# --------------------------------------------------------------------------
def test_the_four_invariants_are_still_present():
    md_text = MD.read_text(encoding="utf-8")
    py_text = PRODUCER.read_text(encoding="utf-8")
    for key, invariant in FOUR_INVARIANTS.items():
        assert invariant in md_text, "invariant %s is missing from synthesis.md" % key
        assert invariant in py_text, "invariant %s is missing from the producer" % key


# --------------------------------------------------------------------------
# D2 -- the LIVE summary-table row must carry the reading-safe form
# --------------------------------------------------------------------------
def test_live_a2_summary_row_is_reading_safe():
    lines = _lines()
    secs = _sections(lines)
    live_rows = [
        (i, ln) for i, ln in enumerate(lines)
        if ln.startswith("| A2 |") and not _is_dated(lines, i, secs)
    ]
    assert live_rows, "the summary table carries no LIVE A2 row"
    for i, ln in live_rows:
        assert "is NOT a connectivity band" not in ln, (
            "the LIVE A2 summary row still carries the refuted X1 claim (line %d)" % (i + 1)
        )
        assert "NOT settled on disk" in ln, (
            "the LIVE A2 summary row does not carry the reading-safe form (line %d)" % (i + 1)
        )


# --------------------------------------------------------------------------
# D1 -- the producer reproduces the published artefact byte-for-byte
# --------------------------------------------------------------------------
def test_producer_reproduces_the_published_md_byte_for_byte(tmp_path):
    (tmp_path / "synthesis.json").write_bytes(JSON_PATH.read_bytes())
    env = dict(os.environ)
    env["WFA175_OUT_DIR"] = str(tmp_path)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, str(PRODUCER)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, "the producer failed:\n%s\n%s" % (proc.stdout, proc.stderr)
    produced = (tmp_path / "synthesis.md").read_bytes()
    assert produced == MD.read_bytes(), (
        "the producer does NOT reproduce %s byte-for-byte (%d vs %d bytes) -- a "
        "re-run would rewrite or drop a dated section"
        % (MD, len(produced), len(MD.read_bytes()))
    )


# --------------------------------------------------------------------------
# no numeric leaf may move
# --------------------------------------------------------------------------
def test_no_numeric_leaf_moved_in_synthesis_json():
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    numeric = dict(_numeric_leaves(payload))
    assert len(numeric) == FROZEN_NUMERIC_LEAF_COUNT, (
        "synthesis.json now carries %d numeric leaves (was %d)"
        % (len(numeric), FROZEN_NUMERIC_LEAF_COUNT)
    )
    digest = hashlib.sha256(
        "\n".join("%s=%r" % (p, v) for p, v in sorted(numeric.items())).encode()
    ).hexdigest()
    assert digest == FROZEN_NUMERIC_LEAF_DIGEST, "a numeric leaf moved in synthesis.json"


# --------------------------------------------------------------------------
# D3 -- the corrected published-only line count is recorded
# --------------------------------------------------------------------------
def test_d3_corrected_published_only_line_count_is_recorded():
    text = MD.read_text(encoding="utf-8")
    assert "17 published-only lines" in text, (
        "the corrected published-only line count (17) is not recorded in synthesis.md"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
