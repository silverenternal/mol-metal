# SPDX-License-Identifier: Apache-2.0
"""Wf-A200 -- the CROSS-WAVE LOCATOR class is closed, and the closure is
MACHINE-CHECKED.

THE DEFECT CLASS
----------------
A document in one wave's tree cites a LINE NUMBER inside another wave's
artefact.  The cited tree is re-rendered; the line number silently starts
pointing at a DIFFERENT sentence while the citation still looks plausible.
Nothing fails, nothing looks wrong, and the number is still a number.

The repo's own rule already said so: the Wf-A161 generator's ``X3`` clause --
*"line-number locators (``:51-89``, ``:360, :378-379``).  Line numbers rot as
the record gains keys; cite the JSON PATHS."*  Wf-A200 turns that rule into a
machine check.

WHAT THIS TEST ASSERTS
----------------------
1. The AUDITOR DETECTS a stale locator -- proved in a TEMP copy, never in the
   repo.
2. The auditor RESOLVES a good locator and does NOT flag everything: a
   correctly cited line is ``RESOLVES-AND-MATCHES`` and the fixture's drift
   count is exactly the number of defects the fixture contains.
3. NO LIVE position of the owned tree carries a perishable line-number locator
   into the Wf-A161 record.  "LIVE" is position-aware: a dated section, a
   dated JSON key and a dated python constant are NOT live claims -- AGENTS.md
   section 5.5 rule #6 REQUIRES the superseded rendering to survive verbatim
   there, so a scanner that flagged them would demand the deletion of the very
   history the rule protects.
4. The NEGATIVE CONTROL for (3): every superseded rendering this wave replaced
   survives VERBATIM in a dated position, and the auditor's dated-position
   detection can SEE it there.  Without (4), (3) would be vacuously true.
5. The auditor is read-only, never walks the repo with ``os.walk``, and never
   touches a GPU.

CPU-only.  No GPU is touched, queried or trained on.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from molmetal.scripts import wfa200_crosswave_locator_audit as auditor  # noqa: E402

# --------------------------------------------------------------------------
# The tree this wave owns.  ``WFA200_TREE_ROOT`` / ``WFA200_AUDIT_ROOT`` /
# ``WFA200_AUDIT_PREFIX`` exist so that the BITE demonstration can point the
# guards at a COPY of the tree with the defect put back.
# --------------------------------------------------------------------------
OWNED = REPO_ROOT / "molmetal" / "reports" / "wfa175_frontier_comparability"
TREE = Path(os.environ.get("WFA200_TREE_ROOT", str(OWNED)))
AUDIT_ROOT = Path(os.environ.get("WFA200_AUDIT_ROOT", str(REPO_ROOT)))
AUDIT_PREFIX = os.environ.get(
    "WFA200_AUDIT_PREFIX",
    "molmetal/reports/wfa175_frontier_comparability",
)

REPORT_MD = TREE / "report.md"
SOURCES_JSON = TREE / "sources.json"
BUILD_PY = TREE / "tools" / "build.py"
REFUTATION_MD = TREE / "refutation" / "refutation.md"
REFUTATION_JSON = TREE / "refutation" / "refutation.json"

#: the Wf-A161 record this wave's tree cites into
WFA161_DIR = "molmetal/reports/wfa161_frontier_labelling"
WFA161_RECORD = WFA161_DIR + "/report.md"
WFA161_PRODUCER = "molmetal/scripts/wfa161_frontier_labelling.py"

#: a LIVE line-number locator into either Wf-A161 artefact
WFA161_LINE_LOCATOR_RE = re.compile(
    r"wfa161_frontier_labelling(?:\.py|/report\.md):\d+"
)

#: the dated key that carries the superseded Wf-A161 renderings
DATED_KEY = "crosswave_locator_repair_2026_11_02_wfa200"

#: the dated constant in the producer that carries the same record
PRODUCER_CONSTANT = "CROSSWAVE_LOCATOR_REPAIR_2026_11_02_WFA200"


# --------------------------------------------------------------------------
# helpers: the LIVE positions of a file
# --------------------------------------------------------------------------
def _live_lines_md(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    dated = auditor._md_dated_lines(text)
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if i not in dated:
            out.append((i, line))
    return out


def _json_leaf_paths(data) -> dict[str, str]:
    """Every string leaf of a JSON document, keyed by its JSON PATH."""
    out: dict[str, str] = {}

    def collect(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                collect(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                collect(v, f"{path}[{i}]")
        elif isinstance(node, str):
            out[path] = node

    collect(data, "")
    return out


def _live_leaves_json(path: Path) -> list[tuple[str, str]]:
    """The LIVE leaves of a JSON document (a dated key is not a live claim)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    dated = auditor._json_dated_paths(data)
    out = []
    for jpath, text in _json_leaf_paths(data).items():
        if jpath in dated:
            continue
        if any(auditor.DATED_SEGMENT_RE.search(seg)
               for seg in re.split(r"[.\[]", jpath)):
            continue
        out.append((jpath, text))
    return out


def _live_lines_py(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    dated = auditor._py_dated_lines(lines)
    return [(i, line) for i, line in enumerate(lines, 1) if i not in dated]


# --------------------------------------------------------------------------
# 1. + 2. THE AUDITOR DETECTS A STALE LOCATOR AND RESOLVES A GOOD ONE
# --------------------------------------------------------------------------
def _fixture(root: Path) -> None:
    """A two-file mini-repo: one good citation, one stale one.

    ``target.md`` line 1 carries the claim; line 2 carries a DIFFERENT
    sentence -- the re-rendered tree.  ``citing_stale.md`` still cites line 2;
    ``citing_good.md`` cites line 1.
    """
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "target.md").write_text(
        "The connectivity rate is 42.0 % on the c7 pool.\n"
        "A different sentence about the frontier band, 92.7-94 %.\n",
        encoding="utf-8",
    )
    (root / "citing_stale.md").write_text(
        "* The rate is recorded at `target.md:2` -- "
        "\"The connectivity rate is 42.0 % on the c7 pool\".\n",
        encoding="utf-8",
    )
    (root / "citing_good.md").write_text(
        "* The rate is recorded at `target.md:1` -- "
        "\"The connectivity rate is 42.0 % on the c7 pool\".\n",
        encoding="utf-8",
    )


def _hits(rep: dict, citing: str) -> list[dict]:
    return [h for h in rep["hits"] if h["citing_file"] == citing]


def test_the_auditor_detects_a_stale_locator_in_a_temporary_copy(tmp_path):
    """The detector BITES: a re-rendered target makes the citation stale."""
    root = tmp_path / "mini"
    root.mkdir()
    _fixture(root)
    rep = auditor.audit(root)

    stale = _hits(rep, "citing_stale.md")
    assert stale, "the auditor did not report the stale citation at all"
    hit = stale[0]
    assert hit["cited_path"] == "target.md", hit
    assert hit["cited_line"] == 2, hit
    assert hit["classification"] == "RESOLVES-BUT-DIFFERS", hit
    assert "different sentence" in (hit.get("cited_line_text") or ""), hit


def test_the_auditor_resolves_a_good_locator_and_does_not_flag_everything(
    tmp_path,
):
    """A correct citation must RESOLVE, or the auditor is a noise generator."""
    root = tmp_path / "mini"
    root.mkdir()
    _fixture(root)
    rep = auditor.audit(root)

    good = _hits(rep, "citing_good.md")
    assert good, "the auditor did not report the good citation at all"
    hit = good[0]
    assert hit["cited_line"] == 1, hit
    assert hit["classification"] == "RESOLVES-AND-MATCHES", hit

    drift = [h for h in rep["hits"]
             if h["classification"] == "RESOLVES-BUT-DIFFERS"]
    assert len(drift) == 1, (
        "the fixture holds exactly ONE defect; the auditor reported %d: %s"
        % (len(drift), [(h["citing_file"], h["cited_line"]) for h in drift])
    )


# --------------------------------------------------------------------------
# 3. NO LIVE POSITION CARRIES A PERISHABLE LOCATOR INTO THE Wf-A161 RECORD
# --------------------------------------------------------------------------
def test_no_live_position_carries_a_line_number_locator_into_wfa161():
    """The LIVE locator must be a JSON PATH or a SECTION ANCHOR."""
    problems: list[str] = []
    for path, live in (
        (REPORT_MD, _live_lines_md(REPORT_MD)),
        (REFUTATION_MD, _live_lines_md(REFUTATION_MD)),
    ):
        for line_no, line in live:
            if WFA161_LINE_LOCATOR_RE.search(line):
                problems.append(f"{path.name}:{line_no}: {line.strip()[:120]}")
    for path, live in (
        (SOURCES_JSON, _live_leaves_json(SOURCES_JSON)),
        (REFUTATION_JSON, _live_leaves_json(REFUTATION_JSON)),
    ):
        for jpath, text in live:
            if WFA161_LINE_LOCATOR_RE.search(text):
                problems.append(f"{path.name} :: {jpath}: {text[:120]}")
    for path, live in ((BUILD_PY, _live_lines_py(BUILD_PY)),):
        for line_no, line in live:
            if WFA161_LINE_LOCATOR_RE.search(line):
                problems.append(f"{path.name}:{line_no}: {line.strip()[:120]}")

    assert not problems, (
        "these LIVE positions carry a perishable line-number locator into the "
        "Wf-A161 record:\n  " + "\n  ".join(problems)
    )


def test_the_auditor_reports_no_live_drift_in_the_owned_tree():
    """The auditor's own verdict on the owned tree must be CLEAN."""
    rep = auditor.audit(AUDIT_ROOT, restrict=[AUDIT_PREFIX])
    assert rep["n_live_drift"] == 0, (
        "the auditor still reports %d LIVE drift(s) in %s:\n  %s"
        % (rep["n_live_drift"], AUDIT_PREFIX,
           "\n  ".join(
               "%s line %d -> %s line %d"
               % (h["citing_file"], h["citing_line"], h["cited_path"],
                  h["cited_line"])
               for h in rep["live_drift"]
           ))
    )
    assert rep["n_files_scanned"] > 0, "the auditor scanned nothing"


# --------------------------------------------------------------------------
# 4. THE NEGATIVE CONTROL: the superseded renderings survive VERBATIM
# --------------------------------------------------------------------------
def test_the_superseded_renderings_survive_verbatim_in_their_dated_homes():
    """AGENTS.md section 5.5 rule #6: the superseded rendering is PRESERVED.

    Without this, "no live position carries a stale locator" would be
    vacuously true -- the scanner might simply be unable to SEE the defect.
    """
    report = REPORT_MD.read_text(encoding="utf-8")

    # the superseded bullet that carried the two `TODO/` line-number locators
    assert "`TODO/README.md:3` (2026-11-01T00:27:56)" in report, (
        "the superseded TODO/README.md:3 locator no longer survives in report.md"
    )
    assert "`TODO/INDEX.md:3` (01:09:29)" in report, (
        "the superseded TODO/INDEX.md:3 locator no longer survives in report.md"
    )
    # the superseded mis-attribution pointer
    assert (
        "`molmetal/reports/wfa161_frontier_labelling/report.md:139` "
        "(mtime 2026-11-01T01:59:40)" in report
    ), "the superseded report.md:139 pointer no longer survives in report.md"
    # the superseded quote, with the emphasis markers the source does NOT have
    assert (
        '  > "**Our 92.7-94 % band** comes from a DIFFERENT perception branch '
        "(Open Babel) and a DIFFERENT post-processing, so it is not on the "
        'frontier\'s scale."' in report
    ), "the superseded mis-attribution quote no longer survives in report.md"
    # the two SELF-locators of section C.1
    assert "**Superseded rendering \u2014 `report.md:79` (" in report, (
        "the superseded report.md:79 self-locator no longer survives"
    )
    assert "**Superseded rendering \u2014 `report.md:186` (" in report, (
        "the superseded report.md:186 self-locator no longer survives"
    )

    ref = json.loads(REFUTATION_JSON.read_text(encoding="utf-8"))
    record = ref[DATED_KEY]
    assert ":139" in record["superseded_what_i_checked_3_verbatim"]
    assert "wfa161_frontier_labelling.py:934" in (
        record["superseded_counter_evidence_3_verbatim"]
    )
    assert "wfa161_frontier_labelling.py:934" in (
        record["superseded_refutation_md_line_15_verbatim"]
    )

    producer = BUILD_PY.read_text(encoding="utf-8")
    assert "CROSSWAVE_LOCATOR_REPAIR_2026_11_02_WFA200" in producer
    assert '"TODO/README.md:3"' in producer, (
        "the producer's dated record no longer carries the superseded locator"
    )
    assert '"molmetal/scripts/wfa161_frontier_labelling.py:934"' in producer


def test_the_auditors_dated_detection_CAN_SEE_the_superseded_renderings():
    """The NEGATIVE CONTROL for the guard above, on the AUDITOR's own side.

    The auditor must classify the superseded locators as DATED (not as live
    drift) -- i.e. it must be able to SEE them in their dated homes.
    """
    rep = auditor.audit(AUDIT_ROOT, restrict=[AUDIT_PREFIX])
    dated_wfa161 = [
        h for h in rep["hits"]
        if h["classification"] == "DATED-SUPERSESSION"
        and "wfa161" in h["cited_path"]
    ]
    assert dated_wfa161, (
        "the auditor cannot see the superseded Wf-A161 locators in their dated "
        "positions -- 'no live drift' would be vacuously true"
    )


# --------------------------------------------------------------------------
# 5. THE AUDITOR IS READ-ONLY, NEVER WALKS, NEVER TOUCHES A GPU
# --------------------------------------------------------------------------
def test_the_auditor_never_walks_the_repo_and_never_imports_torch():
    src = Path(auditor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    roots = set()
    walks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
            if node.module == "os":
                for alias in node.names:
                    if alias.name == "walk":
                        walks.append(("from os import walk", node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr == "walk":
            base = node.value
            if isinstance(base, ast.Name) and base.id == "os":
                walks.append(("os.walk", node.lineno))
    assert not walks, (
        "the auditor must never call os.walk (MEMORY: a repo-wide read blew "
        "the session); found %r" % (walks,)
    )
    assert "torch" not in roots, "the auditor must not import torch"


def test_the_auditor_writes_nothing(tmp_path):
    """A read-only auditor must not touch the tree it audits."""
    root = tmp_path / "mini"
    root.mkdir()
    _fixture(root)
    before = {
        p.name: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.iterdir())
    }
    auditor.audit(root)
    after = {
        p.name: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.iterdir())
    }
    assert before == after, "the auditor modified the tree it audited"
    assert sorted(p.name for p in root.iterdir()) == [
        "citing_good.md", "citing_stale.md", "sub", "target.md",
    ]


def test_the_auditor_refuses_to_open_a_file_for_writing(tmp_path):
    """The auditor's file access is READ-ONLY by construction."""
    target = tmp_path / "x.md"
    target.write_text("hello\n", encoding="utf-8")
    try:
        auditor._open_read(target, "w")
    except auditor.NotReadOnly:
        pass
    else:
        raise AssertionError("the auditor opened a file in write mode")
    with auditor._open_read(target) as fh:
        assert fh.read() == "hello\n"


# --------------------------------------------------------------------------
# 6. THE CENSUS MUST ACTUALLY READ THE AUDITED TREE'S GITIGNORED ARTEFACTS
# --------------------------------------------------------------------------
#: the gitignored artefacts of the audited tree (``.gitignore:158``
#: ignores ``molmetal/reports/**/*.json``)
GITIGNORED_ARTEFACTS = (
    "molmetal/reports/wfa175_frontier_comparability/sources.json",
    "molmetal/reports/wfa175_frontier_comparability/band_census.json",
    "molmetal/reports/wfa175_frontier_comparability/refutation/refutation.json",
)


def test_the_census_really_reads_the_gitignored_artefacts_of_the_audited_tree():
    """A census that skips its own subjects is not a census.

    NEGATIVE CONTROL included: without the prefix, the gitignored
    artefacts are ABSENT -- which is exactly why they have to be named.
    """
    with_ignored, _ = auditor.census_files(
        REPO_ROOT, ignored_under=[AUDIT_PREFIX]
    )
    for rel in GITIGNORED_ARTEFACTS:
        assert rel in with_ignored, (
            "the census does not contain %s -- it is gitignored "
            "(.gitignore:158) and is silently skipped" % rel
        )

    without_ignored, _ = auditor.census_files(REPO_ROOT)
    skipped = [r for r in GITIGNORED_ARTEFACTS if r in without_ignored]
    assert not skipped, (
        "the NEGATIVE CONTROL failed: %r were already in the plain census, "
        "so this guard would pass vacuously" % (skipped,)
    )


def test_the_auditor_scans_the_gitignored_artefacts_of_the_audited_tree():
    """The census must not only LIST them -- the scan must OPEN them."""
    rep = auditor.audit(REPO_ROOT, restrict=[AUDIT_PREFIX])
    # the report does not name the scanned files, so assert on what the
    # scan actually FOUND in them: refutation.json carries the Wf-A161
    # locators, so a scan that skipped it would find strictly fewer hits.
    hits = rep["hits"]
    citing_json = [
        h for h in hits
        if h["citing_file"].endswith("refutation/refutation.json")
    ]
    assert citing_json, (
        "no hit is CITED BY refutation/refutation.json -- the scan did not "
        "read that gitignored file"
    )
    assert rep["n_files_scanned"] > 0


# --------------------------------------------------------------------------
# 7. A LOCATOR INTO A PINNED PRIMARY SOURCE IS SNAPSHOT, NOT DRIFT
# --------------------------------------------------------------------------
def test_a_locator_into_a_pinned_eprint_is_snapshot_not_drift(tmp_path, monkeypatch):
    """A sha256-pinned e-print's line numbers cannot drift.

    The auditor must classify such a locator SNAPSHOT.  The fixture
    cites ``main.tex:929`` -- a file that exists ONLY inside a pinned
    e-print tarball, not in the repo -- so it must NOT be reported as
    drift.
    """
    root = tmp_path / "pinned"
    root.mkdir()
    (root / "citing.md").write_text(
        'The frontier table cites `main.tex:929` for the GeoLDM column.\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        auditor, "build_pinned_source_files", lambda r: {"main.tex"}
    )
    rep = auditor.audit(root)
    hits = [h for h in rep["hits"] if h["cited_path"] == "main.tex"]
    assert hits, "the auditor did not even report the locator"
    for h in hits:
        assert h["classification"] == "SNAPSHOT", (
            "a locator into a sha256-pinned e-print was classified %r; its "
            "line numbers are stable by construction" % h["classification"]
        )
        assert h["classification"] != "RESOLVES-BUT-DIFFERS"
    assert rep["n_live_drift"] == 0, (
        "the pinned-e-print locator was counted as live drift"
    )


def test_the_auditor_knows_the_repos_pinned_eprint_file_names():
    """The pin index must actually find the repo's e-print file names."""
    names = auditor.build_pinned_source_files(REPO_ROOT)
    assert names, "no pinned e-print file names were found in the repo"
    assert "main.tex" in names, (
        "the frontier table's primary sources cite main.tex, but the pin "
        "index does not know it"
    )


# --------------------------------------------------------------------------
# 8. THE Wf-A200e DATED CORRECTION SURVIVES IN ITS DATED HOME
# --------------------------------------------------------------------------
def test_the_wfa200e_dated_correction_survives_verbatim():
    """The correction of the live index label and of the drift count."""
    correction = "crosswave_locator_record_correction_2026_11_02_wfa200e"
    ref = json.loads(REFUTATION_JSON.read_text(encoding="utf-8"))
    assert correction in ref, (
        "the Wf-A200e dated correction no longer survives in refutation.json"
    )
    rec = ref[correction]
    idx = rec["correction_1_the_live_list_index_label"][
        "measured_on_disk_2026_11_02"
    ]
    assert idx["verdicts.CLAIM_3.counter_evidence_index_of_the_repaired_entry"] == 4
    assert idx["verdicts.CLAIM_3.what_i_checked_index_of_the_repaired_entry"] == 4
    assert rec["correction_2_the_live_drift_count"][
        "measured_now_2026_11_02_over_this_branch"
    ]["counts_by_class"]["RESOLVES-BUT-DIFFERS"] == 0

    md = REFUTATION_MD.read_text(encoding="utf-8")
    assert "### Wf-A200e -- two DATED corrections" in md, (
        "the Wf-A200e dated correction section no longer survives in "
        "refutation.md"
    )
    assert (
        "The corrected auditor reports **exactly one** live locator left in this"
        in md
    ), "the superseded live-drift-count rendering no longer survives VERBATIM"


# --------------------------------------------------------------------------
# 9. THE Wf-A200f DATED CORRECTION SURVIVES, AND SO DOES WHAT IT CORRECTS
# --------------------------------------------------------------------------
def test_the_wfa200f_dated_correction_survives_verbatim():
    """The '94.0 appears LITERALLY only' scope sentence was over-claimed.

    MEASURED counter-example: NEAT-POCKET arXiv 2609.05097v1 prints 94.0 in
    ``sections/results.tex:23`` and ``:126`` as well.  The correction is
    ADDITIVE and DATED; the superseded rendering it corrects must still be
    there, VERBATIM.
    """
    key = "crosswave_locator_record_correction_2026_11_02_wfa200f"
    ref = json.loads(REFUTATION_JSON.read_text(encoding="utf-8"))
    assert key in ref, (
        "the Wf-A200f dated correction no longer survives in refutation.json"
    )
    rec = ref[key]
    corr = rec["correction_3_the_94_0_scope"]

    # the SUPERSEDED rendering is preserved verbatim ...
    superseded = corr["superseded_rendering_verbatim"]
    assert "LITERALLY only" in superseded, (
        "the superseded 'LITERALLY only' rendering is not preserved verbatim"
    )
    assert superseded.startswith(
        "the numeral 94.0 appears LITERALLY only in HLTF's abstract"
    )

    # ... and the correction is SCOPED, not a different number
    assert "SCOPED" in corr["corrected_rendering"]
    assert "NOT unique to HLTF" in corr["corrected_rendering"]

    # the counter-example is pinned by digest, not by assertion
    ce = corr["measured_counter_example"]
    assert ce["eprint"] == "NEAT-POCKET arXiv 2609.05097v1"
    assert len(ce["eprint_sha256"]) == 64
    assert ce["file_bytes_in_the_pinned_tarball"] == 40202
    assert ce["verbatim_line_126"].startswith("Overall, 94.0")

    # the branch's live drift is still zero
    assert rec["the_live_drift_count_after_this_correction"]["counts_by_class"][
        "RESOLVES-BUT-DIFFERS"
    ] == 0

    # the sources.json copy of the correction is present, and the string it
    # corrects was NOT silently rewritten
    src = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    assert "primary_source_reverification_correction_2026_11_02_wfa200f" in src
    live_94_0 = src["crosswave_locator_repair_2026_11_02_wfa200"][
        "primary_source_reverification_2026_11_02"
    ]["cells_measured"]["94.0"]
    assert "LITERALLY only" in live_94_0, (
        "the superseded 'LITERALLY only' rendering was silently rewritten in "
        "sources.json -- AGENTS.md 5.5 rule #6 requires it to survive"
    )

    # the human-readable homes survive too
    md = REFUTATION_MD.read_text(encoding="utf-8")
    assert "### Wf-A200f -- the numeral 94.0 is NOT unique to HLTF" in md, (
        "the Wf-A200f dated correction section no longer survives in "
        "refutation.md"
    )
    assert "## §G -- Wf-A200f" in REPORT_MD.read_text(encoding="utf-8"), (
        "the Wf-A200f dated section no longer survives in report.md"
    )
