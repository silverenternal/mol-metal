# SPDX-License-Identifier: Apache-2.0
"""Wf-A192 / Wf-A194 -- the refuted band-provenance renderings are CLOSED, and
the closure is MACHINE-CHECKED.

THE DEFECT CLASS (this wave's scope)
------------------------------------
Five renderings of the published 92.7-94 % frontier band were ASSERTED as
settled.  Three of them are refuted by the repo's own record
``molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json``,
which holds TWO competing provenance readings whose numerals disagree
(92.7 / 94.0 vs 92.98 / 94.33), so the band's provenance is NOT settled on
disk.  The five renderings are named in ``REFUTED_PATTERNS`` below.

WHAT THIS TEST ASSERTS
----------------------
1. No LIVE position of any owned artefact carries a refuted rendering.
   "LIVE" is POSITION-AWARE: a dated supersession section (a ``## §X``
   heading or any dated heading), a dated JSON key, a producer comment, a
   docstring and a test constant are NOT live claims -- AGENTS.md §5.5
   rule #6 REQUIRES the superseded rendering to survive verbatim there, so
   a scanner that flagged them would demand the deletion of the very
   history the rule protects.
2. Every rendering this wave replaced survives VERBATIM in a dated
   position (the NEGATIVE CONTROL: if the scanner could not SEE the defect
   in its dated home, "no live position" would be vacuously true).
3. The hand-authored ``report.md`` live body is byte-equal to the
   PRODUCER's machine field (``sources.json``), so the two cannot drift.
4. The two JSON artefacts' live renderings are byte-equal to what their
   own producers emit, and both artefacts pass their producers' guards.
5. ``sources.json`` is byte-reproducible from ``tools/build.py`` in a TEMP
   dir.

CPU-only.  No GPU is touched, queried or trained on.  The repo is never
walked: only the files this wave owns are read.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

# --------------------------------------------------------------------------
# The artefacts this wave owns
# --------------------------------------------------------------------------
WFA175_DIR = REPO_ROOT / "molmetal" / "reports" / "wfa175_frontier_comparability"
WFA175_REPORT = WFA175_DIR / "report.md"
WFA175_SOURCES = WFA175_DIR / "sources.json"
WFA175_BUILD = WFA175_DIR / "tools" / "build.py"

C7_JSON = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wfa162_c7_pool_ligand_connected_rate"
    / "ligand_connected_rate_c7.json"
)
C7_REPORT = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wfa162_c7_pool_ligand_connected_rate"
    / "report.md"
)
PB_JSON = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wfa164_posebusters_narrative"
    / "default_pool_bounded_repaired.json"
)
SCAN_PY = [
    REPO_ROOT / "molmetal" / "scripts" / "wfa163_c7_rederive.py",
    REPO_ROOT / "molmetal" / "scripts" / "wfa163_narrative_truth_scan.py",
]
THIS_TEST = Path(__file__).resolve()

#: the first dated heading in the hand-authored report
REPORT_DATED_HEADING = "## §C — CORRECTIONS"
#: the dated section that preserves the three superseded renderings
REPORT_D1_HEADING = "### §D.1 SUPERSEDED"

# --------------------------------------------------------------------------
# The refuted renderings.  Each entry is (defect class, compiled pattern).
# The MIRROR pattern carries a NEGATIVE LOOK-AHEAD: the CORRECTED rendering
# says "whether the band as a whole is RAW is NOT settled on disk", which is
# the denial of the over-claim, not the over-claim.
# --------------------------------------------------------------------------
REFUTED_PATTERNS_2026_11_02_WFA194: list[tuple[str, re.Pattern[str]]] = [
    ("X1", re.compile(r"no endpoint is a connected-component count", re.I)),
    ("X1", re.compile(r"the band is not a connectivity band", re.I)),
    ("X1", re.compile(r"neither is a connected-component count", re.I)),
    ("X1", re.compile(r"neither endpoint is a connected-component count", re.I)),
    ("X2", re.compile(r"measured on the published POST-PROCESSED stream", re.I)),
    ("X2", re.compile(r"not post-processed and theirs is", re.I)),
    ("X3", re.compile(r"computed after retaining the largest component", re.I)),
    ("X4", re.compile(r"the band is minted here", re.I)),
    ("X4", re.compile(r"two cells of one table", re.I)),
    (
        "X4",
        re.compile(
            r"two endpoints are different metrics of different models in one table",
            re.I,
        ),
    ),
    ("MIRROR", re.compile(r"both endpoints are RAW", re.I)),
    ("MIRROR", re.compile(r"the band as a whole is RAW(?!\s+is NOT settled)", re.I)),
    # "both arms are RAW" is only a DEFECT when its referent is the BAND.
    # In the c7 artefact the phrase describes OUR OWN two arms
    # (`metal_inclusive` and `ligand_only`), which IS the truth.
    ("MIRROR", re.compile(r"both arms are\s*\**\s*RAW", re.I)),
]

#: A JSON path segment (or a python constant name) that makes a leaf DATED.
DATED_SEGMENT_RE = re.compile(
    r"(superseded|_wfa\d+|\d{4}_\d{2}_\d{2})", re.IGNORECASE
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _matches(text: str) -> list[str]:
    """The defect classes whose pattern fires in ``text``."""
    norm = _normalise(text)
    hits = []
    for cls, rx in REFUTED_PATTERNS_2026_11_02_WFA194:
        if rx.search(norm):
            if cls == "MIRROR" and "band" not in norm.lower():
                # the referent is not the band -> not this defect class
                continue
            hits.append(cls)
    return hits


# --------------------------------------------------------------------------
# The renderings this wave REPLACED -- they must survive VERBATIM in a dated
# position.  These constants are themselves dated test constants, so this
# file does not flag its own copy of the defect.
# --------------------------------------------------------------------------
SUPERSEDED_REPORT_HEADING_2026_11_02_WFA194 = (
    "### 4.1 Source layer -- the band's endpoints are two cells of one table in HLTF"
)
SUPERSEDED_REPORT_BODY_2026_11_02_WFA194 = """So the band is minted here:

* **lower endpoint 92.7 %** = **GeoLDM's Val&Uniq** (a *chemical* QM9 graph metric);
* **upper endpoint 94 %** = **HLTF's own PB-Valid** (a *3D PoseBusters* metric, a different test set).

They are **different models, different metrics, different axes, in one table**. That is the whole defect in a single screenshot. The other members the repo swept -- 92.98 (EF-TALFM connectivity), 94.33 (EF-TALFM yield), 93.13 (EF-TALFM all-PoseBusters-checks), 93.1 (GeoLDM PB-Valid), 93.3 (GCDM Val&Uniq), 92.9 (HLTF Val&Uniq) -- are additional cells of the same heterogeneous family, and the EF-TALFM ones are confirmed in its primary record at `src/appendix.tex:82`, `:89`, `:92`, `:115`, `:127`."""
SUPERSEDED_REPORT_CONCLUSION_C_2026_11_02_WFA194 = (
    "**Conclusion (c):** the band is a frontier composite whose two endpoints are "
    "different metrics of different models in one table (HLTF Table 1(b)); it "
    "entered this repository only in the 2026-11-01 working tree (0 committed "
    "renderings), was labelled as the frontier's from its first metric artifact, "
    "was briefly mis-labelled \"our band\" in the Wf-A161 report, and has never "
    "reached `paper/` in any rendering."
)
SUPERSEDED_C7_FRONTIER_2026_11_02_WFA194 = (
    "the published band is measured on the published POST-PROCESSED stream."
)
SUPERSEDED_PB_FRONTIER_2026_11_02_WFA194 = (
    "measured on the published POST-PROCESSED stream"
)
SUPERSEDED_PB_CONSEQUENCE_2026_11_02_WFA194 = (
    "our number is not post-processed and theirs is"
)


# --------------------------------------------------------------------------
# Position classification
# --------------------------------------------------------------------------
def _md_live_lines(text: str) -> list[tuple[int, str]]:
    """(line_no, line) for the LIVE lines of a markdown file.

    A section whose heading starts with ``## §`` or carries a date is a DATED
    supersession / correction section: AGENTS.md §5.5 rule #6 requires the
    superseded rendering to live there, so it is NOT a live claim.
    """
    out: list[tuple[int, str]] = []
    dated = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            dated = line.startswith("## §") or bool(
                re.search(r"\d{4}-\d{2}-\d{2}", line)
            )
        if not dated:
            out.append((i, line))
    return out


def _json_leaves(node, path: str = ""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _json_leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _json_leaves(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def _json_is_dated(path: str) -> bool:
    for seg in re.split(r"[.\[]", path):
        if seg and DATED_SEGMENT_RE.search(seg):
            return True
    return False


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Line numbers covered by any docstring (module / function / class)."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                for ln in range(body[0].lineno, body[0].end_lineno + 1):
                    out.add(ln)
    return out


def _py_line_is_live(lines: list[str], line_no: int) -> bool:
    """Is this python line a LIVE claim?

    A COMMENT that quotes the defect is a producer comment, and a string
    inside a DATED-named constant is a dated record: neither is a live claim.
    """
    stripped = lines[line_no - 1].lstrip()
    if stripped.startswith("#"):
        return False
    for back in range(line_no - 1, max(-1, line_no - 60), -1):
        m = re.match(r"^\s*([A-Za-z_0-9\"]+)\s*[:=]", lines[back])
        if m:
            name = m.group(1).strip('"')
            return not DATED_SEGMENT_RE.search(name)
    return True


# --------------------------------------------------------------------------
# 1. NO LIVE POSITION CARRIES A REFUTED RENDERING
# --------------------------------------------------------------------------
def _scan_markdown(path: Path) -> list[str]:
    problems = []
    for line_no, line in _md_live_lines(path.read_text(encoding="utf-8")):
        for cls in _matches(line):
            problems.append(f"{path.name}:{line_no}: [{cls}] {line.strip()[:120]}")
    return problems


def _scan_json(path: Path) -> list[str]:
    problems = []
    data = json.loads(path.read_text(encoding="utf-8"))
    for jpath, text in _json_leaves(data):
        if _json_is_dated(jpath):
            continue
        for cls in _matches(text):
            problems.append(f"{path.name} :: {jpath}: [{cls}] {text[:120]}")
    return problems


def _scan_python(path: Path) -> list[str]:
    problems = []
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    docstrings = _docstring_lines(ast.parse(src))
    norm_lines = [_normalise(l) for l in lines]
    for i, line in enumerate(lines, 1):
        if i in docstrings:
            continue
        if not _py_line_is_live(lines, i):
            continue
        for cls, rx in REFUTED_PATTERNS_2026_11_02_WFA194:
            if not rx.search(norm_lines[i - 1]):
                continue
            if cls == "MIRROR" and "band" not in norm_lines[i - 1].lower():
                continue
            problems.append(f"{path.name}:{i}: [{cls}] {line.strip()[:120]}")
    return problems


def test_no_live_position_carries_a_refuted_rendering():
    """NO LIVE position of ANY owned artefact may assert a refuted rendering."""
    problems: list[str] = []
    for md in (WFA175_REPORT, C7_REPORT):
        problems += _scan_markdown(md)
    for js in (WFA175_SOURCES, C7_JSON, PB_JSON):
        problems += _scan_json(js)
    for py in (WFA175_BUILD, *SCAN_PY, THIS_TEST):
        problems += _scan_python(py)
    assert not problems, (
        "these LIVE positions carry a refuted band-provenance rendering:\n  "
        + "\n  ".join(problems)
    )


def test_the_dated_positions_still_carry_the_superseded_renderings():
    """The NEGATIVE CONTROL for the test above.

    The scanner must be able to SEE the defect: if it could not find the
    renderings in their dated positions, then "no live position" would be
    vacuously true.
    """
    report = WFA175_REPORT.read_text(encoding="utf-8")
    assert SUPERSEDED_REPORT_HEADING_2026_11_02_WFA194 in report, (
        "the superseded 4.1 heading no longer survives anywhere in report.md"
    )
    assert SUPERSEDED_REPORT_BODY_2026_11_02_WFA194 in report, (
        "the superseded 4.1 body no longer survives anywhere in report.md"
    )
    assert SUPERSEDED_REPORT_CONCLUSION_C_2026_11_02_WFA194 in report, (
        "the superseded Conclusion (c) no longer survives anywhere in report.md"
    )

    c7 = json.loads(C7_JSON.read_text(encoding="utf-8"))
    assert (
        c7["comparability"]["superseded_frontier_2026_11_02_wfa194"]["rendering"]
        == SUPERSEDED_C7_FRONTIER_2026_11_02_WFA194
    )

    pb = json.loads(PB_JSON.read_text(encoding="utf-8"))
    texts = [
        item["text"]
        for item in pb["comparability_vs_frontier_band"][
            "superseded_reading_2026_11_01"
        ]["superseded"]
    ]
    assert SUPERSEDED_PB_FRONTIER_2026_11_02_WFA194 in texts
    assert SUPERSEDED_PB_CONSEQUENCE_2026_11_02_WFA194 in texts


# --------------------------------------------------------------------------
# 2. EVERY REPLACED RENDERING SURVIVES VERBATIM IN A DATED POSITION
# --------------------------------------------------------------------------
def test_the_report_superseded_renderings_survive_in_the_dated_section_d1():
    """AGENTS.md §5.5 rule #6: the superseded rendering is PRESERVED.

    All three live renderings replaced by Wf-A194 are still present, byte
    for byte, inside section D.1's dated fenced blocks.
    """
    report = WFA175_REPORT.read_text(encoding="utf-8")
    parts = report.split(REPORT_D1_HEADING, 1)
    assert len(parts) == 2, "section D.1 is missing"
    dated = parts[1]
    assert SUPERSEDED_REPORT_HEADING_2026_11_02_WFA194 in dated, (
        "section D.1 no longer preserves the superseded 4.1 heading"
    )
    assert SUPERSEDED_REPORT_BODY_2026_11_02_WFA194 in dated, (
        "section D.1 no longer preserves the superseded 4.1 body"
    )
    assert SUPERSEDED_REPORT_CONCLUSION_C_2026_11_02_WFA194 in dated, (
        "section D.1 no longer preserves the superseded Conclusion (c)"
    )


def test_the_live_renderings_are_not_the_superseded_renderings():
    """A repair that keeps the defect is not a repair."""
    report = WFA175_REPORT.read_text(encoding="utf-8")
    live = report.split(REPORT_DATED_HEADING, 1)[0]
    assert SUPERSEDED_REPORT_HEADING_2026_11_02_WFA194 not in live
    assert SUPERSEDED_REPORT_BODY_2026_11_02_WFA194 not in live
    assert SUPERSEDED_REPORT_CONCLUSION_C_2026_11_02_WFA194 not in live

    c7 = json.loads(C7_JSON.read_text(encoding="utf-8"))
    assert (
        c7["comparability"]["breaks"][2]["frontier"]
        != SUPERSEDED_C7_FRONTIER_2026_11_02_WFA194
    ), "the c7 live field still carries the superseded X2 rendering"
    pb = json.loads(PB_JSON.read_text(encoding="utf-8"))
    assert (
        pb["comparability_vs_frontier_band"]["breaks"][2]["frontier"]
        != SUPERSEDED_PB_FRONTIER_2026_11_02_WFA194
    ), "the posebusters live frontier field still carries the superseded X2 rendering"
    assert (
        pb["comparability_vs_frontier_band"]["breaks"][2]["consequence"]
        != SUPERSEDED_PB_CONSEQUENCE_2026_11_02_WFA194
    ), "the posebusters live consequence still carries the superseded X2 rendering"


# --------------------------------------------------------------------------
# 3. THE HAND-AUTHORED REPORT IS TIED TO THE PRODUCER'S MACHINE FIELD
# --------------------------------------------------------------------------
def test_the_report_live_renderings_equal_the_producer_machine_fields():
    """`report.md` has no producer of its own, so its machine anchor IS the
    producer's field: the two must be byte-equal or they will drift.

    Each of the three live renderings must appear EXACTLY ONCE in the
    report, and that one occurrence must be in the LIVE region (before the
    first dated heading).
    """
    report = WFA175_REPORT.read_text(encoding="utf-8")
    sources = json.loads(WFA175_SOURCES.read_text(encoding="utf-8"))
    record = sources["live_section_4_1_reading_safe_2026_11_02_wfa194"]
    live_end = report.index(REPORT_DATED_HEADING)

    for key in (
        "live_section_4_1_heading",
        "live_section_4_1_rendering",
        "live_conclusion_c_rendering",
    ):
        rendering = record[key]
        assert rendering, key
        assert report.count(rendering) == 1, (
            f"{key} must appear exactly once in report.md, "
            f"found {report.count(rendering)}"
        )
        assert report.index(rendering) < live_end, (
            f"{key} is not in the LIVE region of report.md"
        )


def test_the_live_rendering_states_both_competing_readings():
    """The reading-safe rendering must name BOTH readings, not one."""
    sources = json.loads(WFA175_SOURCES.read_text(encoding="utf-8"))
    record = sources["live_section_4_1_reading_safe_2026_11_02_wfa194"]
    rendering = record["live_section_4_1_rendering"]
    assert "reading A" in rendering and "reading B" in rendering
    assert "NOT settled on disk" in rendering
    assert "band_provenance" in rendering
    assert "EF-TALFM" in rendering


# --------------------------------------------------------------------------
# 4. PRODUCER PARITY -- the two JSON artefacts vs their own producers
# --------------------------------------------------------------------------
def test_the_c7_live_rendering_equals_the_producer_output():
    from molmetal.scripts import wf_a155_ligand_connected_rate as tool

    art = json.loads(C7_JSON.read_text(encoding="utf-8"))
    producer_breaks = [
        dict(b)
        for b in tool.build_comparability_breaks(
            "cfm_c7_fullgen_2026_10_20", art["denominators"]["n_analyzed"]
        )
    ]
    assert art["comparability"]["breaks"] == producer_breaks


def test_the_c7_artefact_passes_the_producer_guard():
    from molmetal.scripts import wf_a155_ligand_connected_rate as tool

    art = json.loads(C7_JSON.read_text(encoding="utf-8"))
    guard = tool._narrative_self_consistency(art)
    assert guard["pass"] is True, guard
    assert guard["unbacked_numbers"] == {}


def test_the_posebusters_live_rendering_equals_the_producer_output():
    from molmetal.scripts import wf_a155_posebusters_validity as tool

    art = json.loads(PB_JSON.read_text(encoding="utf-8"))
    producer_breaks = [
        dict(b)
        for b in tool.build_comparability_breaks(
            art["source_pool"]["measured_profile"]
        )
    ]
    assert art["comparability_vs_frontier_band"]["breaks"] == producer_breaks


def test_the_posebusters_artefact_passes_the_producer_guard():
    from molmetal.scripts import wf_a155_posebusters_validity as tool

    art = json.loads(PB_JSON.read_text(encoding="utf-8"))
    guard = tool._narrative_self_consistency(art)
    assert guard["pass"] is True, guard
    assert guard["unbacked_numbers"] == {}
    assert guard["unclassified_leaves"] == {}
    assert guard["unresolved_narrative_paths"] == []


def test_the_posebusters_artefact_carries_the_producers_dated_records():
    """The dated records that preserve the superseded X2 rendering are the
    PRODUCER's own, so a producer re-run reproduces them instead of dropping
    the history."""
    from molmetal.scripts import wf_a155_posebusters_validity as tool

    art = json.loads(PB_JSON.read_text(encoding="utf-8"))
    band = art["comparability_vs_frontier_band"]
    assert band["superseded_reading_2026_11_01"] == dict(
        tool.COMPARABILITY_SUPERSEDED_READING_2026_11_01
    )
    assert band["superseded_reading_2026_11_01_wfa180"] == dict(
        tool.COMPARABILITY_SUPERSEDED_READING_2026_11_01_WFA180
    )
    assert band["superseded_reading_2026_11_01_wfa182"] == dict(
        tool.COMPARABILITY_SUPERSEDED_READING_2026_11_01_WFA182
    )


# --------------------------------------------------------------------------
# 5. THE PRODUCER REPRODUCES sources.json
# --------------------------------------------------------------------------
def test_the_wfa175_sources_json_is_reproduced_by_its_producer(tmp_path):
    """PRODUCER<->ARTEFACT PARITY, proved by a round trip.

    ``tools/build.py`` writes ``sources.json`` and ``tools/ratios.json`` into
    its own ROOT, so a copy of the script in a TEMP dir emits into that dir.
    The published artefact must be reproduced byte for byte.
    """
    out = tmp_path / "gen"
    (out / "tools").mkdir(parents=True)
    shutil.copy2(WFA175_BUILD, out / "tools" / "build.py")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [PYTHON, "tools/build.py"],
        cwd=str(out),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    regenerated = out / "sources.json"
    assert regenerated.is_file()
    published = WFA175_SOURCES.read_text(encoding="utf-8")
    assert regenerated.read_text(encoding="utf-8") == published, (
        "the producer no longer reproduces the published sources.json"
    )


# --------------------------------------------------------------------------
# 6. NO CUDA
# --------------------------------------------------------------------------
def test_the_scan_touches_no_gpu():
    """STATIC: none of the scanned modules imports torch."""
    for path in (WFA175_BUILD, *SCAN_PY):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        assert "torch" not in roots, path
