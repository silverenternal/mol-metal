#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`molmetal.scripts.wfa158_frontier_postproc_matrix` (Wf-A158).

CPU-only by construction: this file imports no torch and asserts that the
tool imports none either.  Run with::

    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \\
        -m pytest molmetal/tests/test_wfa158_frontier_postproc_matrix.py -v

The contract this file enforces is the branch brief's:

* every cell carries a verbatim quote OR an explicit NOT-STATED /
  NOT-RETRIEVED marker -- never a bare assertion;
* every numeric claim carries a locator;
* the artifact is deterministic (re-runnable, offline);
* the tool touches no GPU.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import hashlib

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytest  # noqa: E402

from molmetal.scripts import wfa158_frontier_postproc_matrix as m  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: LIVE vs DATED renderings
# ---------------------------------------------------------------------------
#
# A leaf is DATED when its path carries a date stamp `_YYYY_MM_DD`.  Those
# are the PRESERVED superseded renderings: AGENTS.md §5.5 rule #6 requires
# them to keep the superseded wording verbatim -- including wording that a
# LIVE rendering may not carry.  Everything else is LIVE.
#
# 2026-11-02 (Wf-A194): the stamp is no longer hard-coded to `2026_11_01`.
# The record now carries dated keys under TWO dates (2026_11_01 and
# 2026_11_02), and a hard-coded date literal would have let the whole
# 2026-11-02 subtree read as LIVE.


DATED_STAMP_RE = re.compile(r"_20\d{2}_\d{2}_\d{2}")


def _is_dated(path):
    """True when ``path`` carries a `_YYYY_MM_DD` date stamp."""
    return bool(DATED_STAMP_RE.search(path))


def _leaf_paths(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _leaf_paths(value, f"{prefix}/{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from _leaf_paths(value, f"{prefix}/{idx}")
    else:
        yield prefix, obj


def _live_keys(obj, prefix=""):
    """Yield ``(path, key)`` for every dict key in ``obj``."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}/{key}"
            yield path, key
            yield from _live_keys(value, path)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from _live_keys(value, f"{prefix}/{idx}")


def _live_strings(obj):
    """Return every LIVE (non-dated) string leaf of ``obj``."""
    return [
        value
        for path, value in _leaf_paths(obj)
        if not _is_dated(path) and isinstance(value, str)
    ]


def _live_blob(obj):
    return "\n".join(_live_strings(obj))


def _dated_strings(obj):
    return [
        value
        for path, value in _leaf_paths(obj)
        if _is_dated(path) and isinstance(value, str)
    ]


# ---------------------------------------------------------------------------
# CPU-only contract
# ---------------------------------------------------------------------------


def test_tool_does_not_import_torch():
    """MEASURED contract: the matrix tool never imports torch.

    Checked in a FRESH SUBPROCESS, because this repo's
    ``molmetal/tests/conftest.py`` imports torch itself for its GPU-contract
    gate -- an in-process ``sys.modules`` check would measure the conftest,
    not the tool.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import molmetal.scripts.wfa158_frontier_postproc_matrix;"
        "print('torch' in sys.modules)" % str(REPO)
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("False"), (
        "importing the tool must not pull torch into sys.modules: " + proc.stdout
    )


def test_no_gpu_calls_in_source():
    """MEASURED contract: no GPU probe or cuda call in the tool's source."""
    src = Path(m.__file__).read_text(encoding="utf-8")
    for forbidden in ("nvidia-smi", "torch.cuda", "device('cuda", 'device("cuda'):
        assert forbidden not in src, f"forbidden GPU token in tool source: {forbidden}"
    assert "import torch" not in src, "the tool must not import torch"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matrix():
    return m.build_matrix()


def test_matrix_validates(matrix):
    """The shipped matrix passes its own invariants."""
    problems = m.validate_matrix(matrix)
    assert problems == [], "invariant violations: " + "; ".join(problems)


def test_top_level_keys(matrix):
    for key in (
        "artifact",
        "generated_by",
        "retrieval_date",
        "scope",
        "verdict_vocabulary",
        "breaks",
        "verdicts",
        "anchor_our_side",
        "sources",
        "retrieval_log",
        "honest_framing",
        "what_this_branch_did_not_establish",
    ):
        assert key in matrix, f"missing top-level key {key!r}"


def test_breaks_are_the_four_named_ones(matrix):
    assert matrix["breaks"] == ["DENOMINATOR", "METRIC", "POST-PROCESSING", "HYDROGENS"]


def test_verdict_vocabulary_is_complete(matrix):
    assert set(matrix["verdict_vocabulary"]) == {
        "BREAKS",
        "DOES-NOT-BREAK",
        "NOT-STATED",
        "NOT-RETRIEVED",
    }


def test_seven_sources_present(matrix):
    """MEASURED: the matrix covers the four briefed sources plus the two
    that carry the band's endpoints (HLTF, YuelBond) plus GeoLDM.

    2026-11-02 (Wf-A194): GeoLDM was ADDED as a first-class source -- it is
    the ORIGINATING paper of Reading A's lower endpoint (92.7 %), which
    until then was present only as a transcript of HLTF's table.
    """
    assert set(matrix["sources"]) == {
        "EF-TALFM",
        "FlowMol3",
        "GGFlow",
        "GeoLDM",
        "HLTF",
        "PAFlow",
        "YuelBond",
    }


def test_every_source_has_a_citation_with_arxiv_id_and_date(matrix):
    """Every source must be traceable: an arXiv id OR a DOI, plus a date.

    YuelBond is a bioRxiv preprint (no arXiv id), so the identifier
    requirement is "arXiv id OR DOI" -- never a bare assertion.
    """
    for name, src in matrix["sources"].items():
        cit = src["citation"]
        assert cit.get("arxiv_id") or cit.get("doi"), name
        assert cit["retrieved_at"] == matrix["retrieval_date"], name
        assert cit["retrieval_url"].startswith("http"), name
        assert cit["version_used"], name


def test_yuelbond_has_no_arxiv_id_but_has_a_doi(matrix):
    """MEASURED: the arXiv-less source is pinned, not silently accepted."""
    cit = matrix["sources"]["YuelBond"]["citation"]
    assert not cit.get("arxiv_id")
    assert "10.1101/2025.05.06.652517" in cit["doi"]


# ---------------------------------------------------------------------------
# THE invariant: no cell without a quote or an explicit marker
# ---------------------------------------------------------------------------


def test_no_cell_without_a_quote_or_an_explicit_marker(matrix):
    """MEASURED: every one of the 24 cells is quoteless only if it is marked.

    A quoteless cell is allowed ONLY when its verdict is one of the explicit
    markers AND it carries a non-empty ``marker`` string saying what was not
    found.
    """
    n_quote = 0
    n_marker = 0
    for name, src in matrix["sources"].items():
        for brk, cell in src["cells"].items():
            verdict = cell["verdict"]
            quote = (cell.get("quote") or "").strip()
            marker = (cell.get("marker") or "").strip()
            if verdict in m.MARKER_VERDICTS:
                assert marker, (
                    f"{name}/{brk}: verdict {verdict!r} has no explicit marker"
                )
                assert marker.startswith(verdict), (
                    f"{name}/{brk}: marker must start with the verdict, got {marker!r}"
                )
                n_marker += 1
            else:
                assert quote, (
                    f"{name}/{brk}: verdict {verdict!r} has neither a quote nor a marker"
                )
                n_quote += 1
    # 2026-11-02 (Wf-A194): the cell count is COMPUTED, not hard-coded --
    # `sources` grew from six to seven when GeoLDM was added, and a
    # hard-coded 24 would have been a claim the record no longer backs.
    assert n_quote + n_marker == len(matrix["sources"]) * len(matrix["breaks"])
    assert len(matrix["sources"]) == 7
    assert n_marker > 0, "the matrix must contain at least one explicit marker cell"


def test_paflow_cells_are_all_explicitly_marked(matrix):
    """MEASURED: the one source we could not read is entirely NOT-STATED.

    PAFlow's arXiv HTML was retrieved, but the paper states neither a
    connectivity denominator, nor a validity metric definition, nor a
    post-processing recipe.  Every one of its four cells must therefore be an
    explicit marker -- never a BREAKS verdict derived from our own reading.
    """
    cells = matrix["sources"]["PAFlow"]["cells"]
    for brk, cell in cells.items():
        assert cell["verdict"] == "NOT-STATED", f"PAFlow/{brk}: {cell['verdict']}"
        assert cell["marker"].startswith("NOT-STATED"), f"PAFlow/{brk}"


def test_every_cell_has_a_locator(matrix):
    for name, src in matrix["sources"].items():
        for brk, cell in src["cells"].items():
            assert (cell.get("locator") or "").strip(), f"{name}/{brk}: empty locator"


def test_post_processing_cell_states_the_recipe_verdict(matrix):
    """The brief's core question: is the RECIPE stated, or only its effect?"""
    for name, src in matrix["sources"].items():
        cell = src["cells"]["POST-PROCESSING"]
        assert "recipe_verdict" in cell, f"{name}: POST-PROCESSING has no recipe_verdict"
        assert cell["recipe_verdict"] in m.RECIPE_VERDICTS, name
        assert (cell.get("recipe") or "").strip(), f"{name}: empty recipe description"


def test_ggflow_recipe_is_not_stated(matrix):
    """MEASURED: GGFlow names its correction phase but never describes it.

    This is a first-class result of the branch, so it is pinned: if a future
    revision of the paper describes the recipe, this test must be updated in
    the same commit, not silently left to rot.
    """
    cell = matrix["sources"]["GGFlow"]["cells"]["POST-PROCESSING"]
    assert cell["recipe_verdict"] == "NOT-STATED"
    assert "never described" in cell["recipe"].lower()


# ---------------------------------------------------------------------------
# Numbers must carry a locator
# ---------------------------------------------------------------------------


def test_every_reported_number_carries_a_locator_and_a_quote(matrix):
    for name, src in matrix["sources"].items():
        numbers = src.get("reported_numbers") or []
        assert numbers, f"{name}: no reported numbers"
        for num in numbers:
            assert isinstance(num["value"], (int, float)), (
                f"{name}: {num.get('label')!r} value is not numeric"
            )
            assert (num.get("locator") or "").strip(), (
                f"{name}: {num.get('label')!r} has no locator"
            )
            assert (num.get("quote") or "").strip(), (
                f"{name}: {num.get('label')!r} has no quote"
            )
            assert num.get("unit"), f"{name}: {num.get('label')!r} has no unit"


def test_reported_numbers_are_within_plausible_range(matrix):
    """MEASURED guard: a percentage is a percentage, not 0.997 or 997."""
    for name, src in matrix["sources"].items():
        for num in src["reported_numbers"]:
            unit = num.get("unit", "")
            if unit.startswith("percent"):
                assert 0.0 <= num["value"] <= 100.0, (
                    f"{name}: {num['label']!r} value {num['value']} is not a percentage"
                )


def test_anchor_matches_the_published_metric_file(matrix):
    """MEASURED: the anchor value is the one ON DISK, not a remembered one."""
    path = REPO / matrix["anchor_our_side"]["path"]
    assert path.is_file(), f"anchor metric file missing: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    node = payload
    for part in matrix["anchor_our_side"]["jsonpath"].strip("/").split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    assert node == pytest.approx(matrix["anchor_our_side"]["value"], abs=1e-12)
    assert node == pytest.approx(0.08509630565203663, abs=1e-15)


def test_postprocessed_counterpart_matches_the_published_metric_file(matrix):
    """MEASURED: the frontier-style post-processing arm is the one on disk."""
    pc = matrix["anchor_our_side"]["postprocessed_counterpart"]
    path = REPO / pc["path"]
    assert path.is_file(), f"post-processing arm file missing: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    node = payload
    for part in pc["jsonpath"].strip("/").split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    assert node == pytest.approx(pc["value"], abs=1e-12)
    assert node == pytest.approx(1.0, abs=1e-12)


def _jsonpath(payload, jp):
    node = payload
    for part in jp.strip("/").split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def test_like_for_like_retention_matches_disk(matrix):
    """MEASURED: the heavy-atom retention number is the one on disk.

    This is the ONE quantity that is directly comparable to FlowMol3's
    ``avg_frag_frac``, so it must not drift away from the published metric.
    """
    lfl = matrix["anchor_our_side"]["postprocessed_counterpart"]["like_for_like_retention"]
    path = REPO / matrix["anchor_our_side"]["postprocessed_counterpart"]["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert _jsonpath(payload, lfl["our_jsonpath"]) == pytest.approx(
        lfl["our_value"], abs=1e-12
    )
    assert _jsonpath(payload, lfl["our_median_jsonpath"]) == pytest.approx(
        lfl["our_median"], abs=1e-12
    )
    assert 0.0 < lfl["our_value"] < 1.0


def test_uff_applicability_matches_disk(matrix):
    """MEASURED: the frontier recipe's third step does not apply to our corpus."""
    uff = matrix["anchor_our_side"]["postprocessed_counterpart"]["uff_step_applicability"]
    path = REPO / matrix["anchor_our_side"]["postprocessed_counterpart"]["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert _jsonpath(payload, uff["jsonpath"]) == pytest.approx(uff["value"], abs=1e-12)
    assert uff["value"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Retrieval log
# ---------------------------------------------------------------------------


def test_retrieval_log_entries_are_complete(matrix):
    log = matrix["retrieval_log"]
    assert len(log) >= 4
    for entry in log:
        assert entry["url"].startswith("http"), entry
        assert entry["retrieved_at"] == matrix["retrieval_date"], entry
        assert (entry.get("extracted") or "").strip(), entry


def test_every_source_url_has_a_retrieval_log_entry(matrix):
    urls = {e["url"] for e in matrix["retrieval_log"]}
    for name, src in matrix["sources"].items():
        assert src["citation"]["retrieval_url"] in urls, name


def test_the_failed_openreview_retrieval_is_recorded(matrix):
    """MEASURED: the 403 is in the log, not glossed over."""
    urls = {e["url"] for e in matrix["retrieval_log"]}
    assert any("openreview" in u for u in urls)
    entry = next(e for e in matrix["retrieval_log"] if "openreview" in e["url"])
    assert entry["http_status"] == 403
    assert "NOT" in entry["extracted"].upper()


def test_not_retrieved_blocks_use_the_right_verdict(matrix):
    for name, src in matrix["sources"].items():
        for nr in src.get("not_retrieved", []):
            assert nr["verdict"] == "NOT-RETRIEVED", name
            assert nr["reason"].strip(), name
            assert nr["consequence"].strip(), name


# ---------------------------------------------------------------------------
# Honesty: no scoreboard language
# ---------------------------------------------------------------------------


def test_no_scoreboard_language_in_the_artifact(matrix):
    """The deliverable is a comparability matrix, not a ranking.

    The artifact may say a protocol BREAKS (i.e. the two measurements are
    not the same), but it must not claim a model is worse/better, and must
    not claim the frontier is cheating.

    The check runs over the SUBSTANTIVE parts of the artifact.  The
    ``scope`` / ``honest_framing`` blocks are disclaimers and necessarily
    contain the very phrases this test bans (they say the artifact does NOT
    conclude those things) -- those are asserted separately below.
    """
    substantive = {
        k: v
        for k, v in matrix.items()
        if k not in ("scope", "honest_framing", "verdict_vocabulary")
    }
    blob = json.dumps(substantive, ensure_ascii=False).lower()
    for banned in ("our model is bad", "frontier is cheating", "the frontier cheats",
                   "we lose to", "they beat us", "inferior to the frontier",
                   "better than the frontier", "outperforms", "underperforms"):
        assert banned not in blob, f"scoreboard language present: {banned!r}"
    # the disclaimer itself must be present -- that is where the ban is honoured
    framing = " ".join(matrix["honest_framing"]).lower()
    assert "not a scoreboard" in framing or "does not rank" in framing
    assert "not conclude that our model is bad" in framing or (
        "does not conclude" in framing and "our model is bad" in framing
    )


def test_honest_framing_names_the_disclaimers(matrix):
    """MEASURED: the framing block must name BOTH failure modes it avoids."""
    framing = " ".join(matrix["honest_framing"]).lower()
    assert "not a scoreboard" in framing
    assert "our model is bad" in framing
    assert "frontier is cheating" in framing


def test_honest_framing_is_non_empty(matrix):
    assert len(matrix["honest_framing"]) >= 3
    assert matrix["what_this_branch_did_not_establish"]


def test_every_source_states_the_sample_nature(matrix):
    """MEASURED: we must say whether the band is held-out or training-distribution."""
    for name, src in matrix["sources"].items():
        nature = src["reported_band"]["sample_nature"]
        assert nature, name
        assert ("training-distribution" in nature) or ("held-out" in nature), (
            f"{name}: sample nature does not say training-distribution vs held-out: {nature!r}"
        )


# ---------------------------------------------------------------------------
# Quote pinning + end-to-end run
# ---------------------------------------------------------------------------


def test_quote_verification_index_is_populated():
    assert m.quote_count() >= 40, "the branch pinned too few quotes to be useful"
    ids = [q["id"] for q in m.QUOTE_VERIFICATION_INDEX]
    assert len(ids) == len(set(ids)), "duplicate quote ids"
    for q in m.QUOTE_VERIFICATION_INDEX:
        assert q["file"], q
        assert q["quote"].strip(), q


def test_tool_runs_end_to_end_and_writes_the_artifact(tmp_path):
    """MEASURED end-to-end: the script runs and writes both artifacts."""
    out_dir = tmp_path / "wfa158_frontier_postproc_matrix"
    rc = m.main(["--out-dir", str(out_dir)])
    assert rc == 0

    json_path = out_dir / "frontier_matrix.json"
    md_path = out_dir / "frontier_matrix.md"
    assert json_path.is_file()
    assert md_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "wfa158_frontier_postproc_matrix"
    assert set(payload["sources"]) == {
        "EF-TALFM",
        "FlowMol3",
        "GGFlow",
        "GeoLDM",
        "HLTF",
        "PAFlow",
        "YuelBond",
    }

    md = md_path.read_text(encoding="utf-8")
    assert "frontier post-processing comparability matrix" in md.lower()
    assert "8.51" in md
    assert "92.98" in md


def test_the_dated_parity_evidence_file_is_written_and_honest(tmp_path):
    """P3: the dated sibling file carries the byte evidence for BOTH
    artefacts, and every md5 in it is the md5 of the bytes that are really
    written.

    A file cannot contain its own hash, which is why the markdown's own
    size and md5 live in this sibling rather than inside the markdown.
    """
    out_dir = tmp_path / "wfa158_parity_evidence"
    rc = m.main(["--out-dir", str(out_dir)])
    assert rc == 0
    ev_path = out_dir / m.PARITY_EVIDENCE_FILENAME
    assert ev_path.is_file()
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    assert ev["reconciliation_key"] == m.RECONCILIATION_KEY
    assert ev["numeric_leaves_moved_by_this_repair"] == 0
    for name in ("frontier_matrix.json", "frontier_matrix.md"):
        raw = (out_dir / name).read_bytes()
        entry = ev["artifacts"][name]
        assert entry["bytes"] == len(raw), name
        assert entry["md5"] == hashlib.md5(raw).hexdigest(), name
    # the markdown quotes the RECORD's md5 -- it can: the record does not
    # contain the markdown's hash, so there is no circularity
    md = (out_dir / "frontier_matrix.md").read_text(encoding="utf-8")
    assert ev["artifacts"]["frontier_matrix.json"]["md5"] in md
    # and the leaf counts agree with the record that was written
    record = json.loads(
        (out_dir / "frontier_matrix.json").read_text(encoding="utf-8")
    )
    leaves = list(_leaf_paths(record))
    assert ev["leaf_counts"]["total"] == len(leaves)
    assert ev["leaf_counts"]["numeric"] == sum(
        1 for _, v in leaves if isinstance(v, (int, float)) and not isinstance(v, bool)
    )


# ---------------------------------------------------------------------------
# Band provenance: the repo's "92.7-94 %" band must be traced
# ---------------------------------------------------------------------------


def test_band_provenance_block_is_present(matrix):
    assert "band_provenance" in matrix
    bp = matrix["band_provenance"]
    assert bp["band_as_quoted_in_the_repo"]
    assert bp["headline_finding"]
    assert bp["where_the_repo_states_it"]
    assert bp["what_this_does_not_settle"]


def test_band_provenance_endpoint_candidates_are_attributable(matrix):
    """Every candidate endpoint must name a source we actually read.

    A candidate that names no source is a guess and must not ship.
    """
    bp = matrix["band_provenance"]
    endpoints = {c["endpoint"] for c in bp["endpoint_candidates"]}
    assert endpoints == {"92.7 %", "94 %"}
    for entry in bp["endpoint_candidates"]:
        for key, cand in entry.items():
            if not key.startswith("candidate_"):
                continue
            # 2026-11-02 (Wf-A194) -- a DATED SIBLING of a candidate holds a
            # STRING (the superseded rendering), not a candidate object.  It
            # is not a candidate and is not attributable; it is checked as a
            # dated string instead (see
            # test_dated_keys_preserve_the_superseded_text_verbatim).
            if not isinstance(cand, dict):
                continue
            assert cand["paper"], entry
            assert cand["what_it_is"], cand
            assert cand["denominator"], cand
            assert cand["locator"], cand
            assert cand["quote"].strip(), cand
            assert cand["status"], cand


def test_band_provenance_headline_finding_is_reading_safe(matrix):
    """Wf-A180 (2026-11-01): the finding must assert the four INVARIANTS and
    must NOT assert either contested provenance reading as settled.

    The superseded rendering ("THE BAND IS NOT A CONNECTIVITY BAND ... None
    of the three is a connected-component count") is TRUE under Reading A and
    FALSE under Reading B, so it may not be published as fact.  What may be
    published are the four claims that hold under BOTH readings.
    """
    finding = matrix["band_provenance"]["headline_finding"]
    up = finding.upper()
    # the contested claims (X1) may NOT be asserted as settled
    assert "NOT A CONNECTIVITY BAND" not in up
    assert "NONE OF THE THREE IS A CONNECTED-COMPONENT COUNT" not in up
    # the four invariants that hold under BOTH readings
    assert "EXTERNAL" in up
    assert "NOT ONE HOMOGENEOUS QUANTITY" in up
    assert "NOT ON OUR SCALE" in up
    assert "SOURCE CELLS ARE NOT SETTLED" in up
    # OUR number is a connected-component count on the RAW emitted SMILES
    assert "CONNECTED-COMPONENT COUNT ON THE RAW EMITTED SMILES" in up
    # both readings are named, and they are addressed by JSON PATH
    assert "`band_provenance`" in finding
    assert "`sources`.`EF-TALFM`" in finding
    assert "92.98" in finding and "94.33" in finding


def test_band_provenance_emits_the_competing_reading_key(matrix):
    """Wf-A180 (2026-11-01): the contest must be EMITTED, not only annotated.

    Before Wf-A180 the correction lived only in the artefact on disk, so a
    re-run of the producer dropped it.  The producer now emits the key, and
    it must carry the canonical clause, the two readings and the four
    invariants -- addressed by JSON PATH, never by line number.
    """
    bp = matrix["band_provenance"]
    assert "competing_reading_and_what_this_does_not_settle" in bp
    cr = bp["competing_reading_and_what_this_does_not_settle"]
    assert cr["status"]
    stmt = cr["canonical_reading_safe_band_statement"]
    up = stmt.upper()
    assert "EXTERNAL" in up
    assert "NOT ONE HOMOGENEOUS QUANTITY" in up
    assert "NOT ON OUR SCALE" in up
    assert "SOURCE CELLS ARE NOT SETTLED" in up
    assert len(cr["readings"]) == 2
    assert {r["reading"] for r in cr["readings"]} == {"A", "B"}
    for r in cr["readings"]:
        assert r["locator"]
        assert "frontier_matrix.json" in r["locator"]
    # The non-existent key must not be pointed at BY A LIVE RENDERING.
    # (It survives only inside the DATED superseded keys, which must keep
    # it verbatim -- see test_dated_keys_preserve_the_superseded_text.)
    assert "frontier_axis_mapping" not in _live_blob(cr)
    # And no LIVE LOCATOR carries a self-referential line number (they go
    # stale on every edit).  The dated `line_number_note` and the
    # `current_line_numbers_on_disk` fields are a RECORD OF THAT DRIFT,
    # not locators; they are checked separately below.
    for r in cr["readings"]:
        assert not re.search(r"\(:\d", r["locator"]), r["locator"]
        assert not re.search(r"\s:\d", r["locator"]), r["locator"]
    # the four publishable invariants are enumerated
    pub = cr["the_only_claims_that_may_be_published"]
    assert set(pub) == {
        "i_the_band_is_external",
        "ii_it_is_not_one_homogeneous_quantity",
        "iii_it_is_not_on_our_scale",
        "iv_the_source_cells_are_not_settled",
    }
    for key, text in pub.items():
        assert text.strip(), key


def test_hltf_metric_note_is_reading_safe(matrix):
    """Wf-A180 (2026-11-01): the sibling 'NEITHER is a connected-component
    count' rendering may not stay live either."""
    note = matrix["sources"]["HLTF"]["cells"]["METRIC"]["note"]
    up = note.upper()
    assert "NEITHER IS A CONNECTED-COMPONENT COUNT" not in up
    assert "TOTAL METRIC BREAK" not in up
    assert "NOT SETTLED ON DISK" in up
    assert "EXTERNAL" in up
    assert "NOT ONE HOMOGENEOUS QUANTITY" in up
    assert "NOT ON OUR SCALE" in up
    # the two readings are named, with their numerals
    assert "92.98" in note and "94.33" in note and "92.7" in note


def test_band_provenance_what_this_does_not_settle_is_reading_safe(matrix):
    """Wf-A180 (2026-11-01): 'every candidate endpoint ... rather than a
    connectivity rate' is not settled either."""
    wt = matrix["band_provenance"]["what_this_does_not_settle"]
    assert "every candidate endpoint of the band" not in wt
    assert "NOT settled on disk" in wt
    assert "C. = 92.98 %" in wt


def test_band_provenance_quotes_come_from_the_quote_index(matrix):
    """Every quote inside the band-provenance block must be pinned for
    re-verification against the saved primary-source extraction."""
    pinned = {q["quote"] for q in m.QUOTE_VERIFICATION_INDEX}
    bp = matrix["band_provenance"]
    quotes = []
    for entry in bp["endpoint_candidates"]:
        for key, cand in entry.items():
            # 2026-11-02 (Wf-A194) -- a DATED SIBLING of a candidate holds a
            # STRING, not a candidate object; it carries no quote to pin.
            if key.startswith("candidate_") and isinstance(cand, dict):
                quotes.append(cand["quote"])
    for q in quotes:
        assert q in pinned, q[:80]


def test_band_provenance_raw_to_post_processed_pairs_are_arithmetically_sound(matrix):
    """MEASURED: the six raw->PP deltas must be the arithmetic they claim."""
    ev = matrix["band_provenance"]["post_processing_evidence_from_the_frontier_itself"]
    pairs = ev["raw_to_post_processed_pairs"]
    assert len(pairs) == 6
    for row in pairs:
        assert row["pp"] - row["raw"] == pytest.approx(row["delta_pp"], abs=1e-9), row
        assert 0.0 <= row["raw"] <= 100.0, row
        assert 0.0 <= row["pp"] <= 100.0, row
    deltas = [r["delta_pp"] for r in pairs]
    assert min(deltas) == pytest.approx(5.6, abs=1e-9)
    assert max(deltas) == pytest.approx(95.0, abs=1e-9)


def test_band_provenance_recovery_fraction_on_our_side_is_one_by_construction(matrix):
    """MEASURED: our recovery fraction is 1.0 BY CONSTRUCTION, and the block
    says so -- the caveat must be present, not implied."""
    rec = matrix["band_provenance"]["post_processing_evidence_from_the_frontier_itself"][
        "recovery_fraction_our_side"
    ]
    assert rec["value"] == pytest.approx(1.0, abs=1e-12)
    assert "BY CONSTRUCTION" in rec["caveat"].upper()
    assert rec["raw"] == pytest.approx(0.08509630565203663, abs=1e-15)
    # the units must be stated: raw/post_processed are FRACTIONS, and the
    # formula must be the fraction form to be re-derivable from them
    assert "fraction" in rec["units"].lower()
    assert "(1 - raw)" in rec["formula"]


def test_tool_is_deterministic(tmp_path):
    """MEASURED: two runs produce byte-identical output."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    assert m.main(["--out-dir", str(a_dir)]) == 0
    assert m.main(["--out-dir", str(b_dir)]) == 0
    assert (a_dir / "frontier_matrix.json").read_bytes() == (
        b_dir / "frontier_matrix.json"
    ).read_bytes()
    assert (a_dir / "frontier_matrix.md").read_bytes() == (
        b_dir / "frontier_matrix.md"
    ).read_bytes()


def test_check_only_writes_nothing(tmp_path):
    out_dir = tmp_path / "nothing"
    rc = m.main(["--check-only", "--out-dir", str(out_dir)])
    assert rc == 0
    assert not out_dir.exists()


def test_verify_quotes_from_missing_dir_exits_2(tmp_path):
    """The SKIP path must exit 2, not crash and not silently pass."""
    rc = m.main(["--verify-quotes-from", str(tmp_path / "nope"), "--check-only"])
    assert rc == 2


# ---------------------------------------------------------------------------
# P1 PARITY: the producer must REPRODUCE its artefact
# ---------------------------------------------------------------------------
#
# MEASURED before the repair: the producer emitted 710 leaves where the record
# on disk carries 739, so an ordinary re-run would have DELETED 29 leaves --
# including the whole Wf-A177 additive history -- and 6 live strings differed.
# The markdown renderer emitted NEITHER of the two dated correction sections,
# so a re-run would have deleted the record of the contest as well.


def _record_on_disk():
    path = m.DEFAULT_OUT_DIR / "frontier_matrix.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def test_producer_drops_no_leaf_the_record_carries(matrix):
    """P1/P3: a re-run must not DELETE anything the artefact carries.

    This is the property that makes the regeneration safe.  Before the repair
    the producer did not emit 29 of the record's leaves.
    """
    on_disk = _record_on_disk()
    if on_disk is None:
        pytest.skip("no record on disk to compare against")
    report = m.parity_report(matrix, m.DEFAULT_OUT_DIR)
    assert report["leaves_only_on_disk"] == 0, (
        "a re-run would DELETE leaves the record carries: " + str(report)
    )


def test_producer_moves_no_numeric_leaf(matrix):
    """P2: 0 existing numeric leaf may move.  This is the publish gate."""
    on_disk = _record_on_disk()
    if on_disk is None:
        pytest.skip("no record on disk to compare against")
    report = m.parity_report(matrix, m.DEFAULT_OUT_DIR)
    assert report["numeric_leaves_moved"] == 0, (
        "a numeric leaf of the record moved: " + str(report)
    )
    # and leaf by leaf, so a moved number cannot hide behind an aggregate
    disk_leaves = dict(_leaf_paths(on_disk))
    prod_leaves = dict(_leaf_paths(matrix))
    for path, value in disk_leaves.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert path in prod_leaves, path
            assert prod_leaves[path] == value, path


def test_producer_output_is_idempotent(matrix):
    """P1: two builds of the record are byte-identical, and so are two
    renderings of the markdown."""
    a = json.dumps(m.build_matrix(), indent=2, sort_keys=True, ensure_ascii=False)
    b = json.dumps(m.build_matrix(), indent=2, sort_keys=True, ensure_ascii=False)
    assert a == b
    assert m.render_markdown(m.build_matrix()) == m.render_markdown(m.build_matrix())


def test_parity_check_detects_a_parity_break(matrix, tmp_path):
    """P1: the parity check must be able to DETECT a break, not only report OK.

    A tampered artefact (a leaf removed, a number changed) must make the
    check report it.
    """
    art = tmp_path / "tampered"
    art.mkdir()
    payload = json.loads(
        (m.DEFAULT_OUT_DIR / "frontier_matrix.json").read_text(encoding="utf-8")
    )
    # (a) a leaf the producer does not emit -> "leaves only on disk" > 0
    payload["band_provenance"]["a_leaf_the_producer_does_not_emit"] = "x"
    # (b) a numeric leaf moved -> "numeric leaves moved" > 0
    numeric_path = next(
        path
        for path, value in _leaf_paths(payload)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    parent = payload
    parts = [p for p in numeric_path.split("/") if p]
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    last = parts[-1]
    if isinstance(parent, list):
        parent[int(last)] = parent[int(last)] + 1
    else:
        parent[last] = parent[last] + 1
    (art / "frontier_matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = m.parity_report(matrix, art)
    assert report["json_byte_identical"] is False
    assert report["leaves_only_on_disk"] == 1, report
    assert report["numeric_leaves_moved"] == 1, report
    assert report["ok"] is False
    # and the CLI can turn that break into a FAIL
    rc = m.main(
        [
            "--check-only",
            "--parity-dir",
            str(art),
            "--require-parity",
            "--out-dir",
            str(tmp_path / "nowhere"),
        ]
    )
    assert rc == 1


def test_check_only_still_passes(matrix, tmp_path, capsys):
    """P1: `--check-only` must still PASS and write nothing."""
    out_dir = tmp_path / "not_written"
    rc = m.main(["--check-only", "--out-dir", str(out_dir)])
    assert rc == 0
    assert not out_dir.exists()
    assert "PASS (check-only: nothing written)" in capsys.readouterr().out


def test_write_is_refused_when_a_leaf_would_be_dropped(matrix, tmp_path):
    """P3: the guard that would have caught the divergence.

    A record carrying a leaf the producer does not emit must NOT be
    overwritten.
    """
    art = tmp_path / "guarded"
    art.mkdir()
    payload = json.loads(
        (m.DEFAULT_OUT_DIR / "frontier_matrix.json").read_text(encoding="utf-8")
    )
    payload["band_provenance"]["a_leaf_the_producer_does_not_emit"] = "x"
    (art / "frontier_matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rc = m.main(["--out-dir", str(art)])
    assert rc == 1
    # the artefact on disk is untouched
    after = json.loads((art / "frontier_matrix.json").read_text(encoding="utf-8"))
    assert "a_leaf_the_producer_does_not_emit" in after["band_provenance"]


# ---------------------------------------------------------------------------
# P4: the four invariants survive, and no LIVE rendering carries a contested
#      claim
# ---------------------------------------------------------------------------

X1_A = "no endpoint is a connected-component count"
X1_B = "THE BAND IS NOT A CONNECTIVITY BAND"
# the same contested claim, in the wording the record's own cross-references
# used to quote in a LIVE rendering ("NEITHER is a connected-component count")
X1_C = "NEITHER is a connected-component count"
X2 = "the band is measured on the POST-PROCESSED stream"
X3 = "their Connectivity is computed AFTER retaining the largest component"
X4 = "the band is minted here: lower = GeoLDM Val&Uniq, upper = HLTF PB-Valid"


def test_the_four_invariants_survive_in_the_band_provenance_block(matrix):
    """P4: the band-provenance block must still carry all four invariants."""
    bp = matrix["band_provenance"]
    cr = bp["competing_reading_and_what_this_does_not_settle"]
    pub = cr["the_only_claims_that_may_be_published"]
    assert set(pub) == {
        "i_the_band_is_external",
        "ii_it_is_not_one_homogeneous_quantity",
        "iii_it_is_not_on_our_scale",
        "iv_the_source_cells_are_not_settled",
    }
    joined = " ".join(
        [
            bp["headline_finding"],
            cr["status"],
            cr["canonical_reading_safe_band_statement"],
        ]
        + list(pub.values())
    )
    up = joined.upper()
    assert "EXTERNAL" in up
    assert "NOT ONE HOMOGENEOUS QUANTITY" in up
    assert "NOT ON OUR SCALE" in up
    assert "SOURCE CELLS ARE NOT SETTLED" in up
    # OUR number stays what it is: a connected-component count on the RAW
    # emitted SMILES
    assert "CONNECTED-COMPONENT COUNT ON THE RAW EMITTED SMILES" in up


def test_no_live_rendering_carries_a_contested_claim(matrix):
    """P4: X1..X4 may not appear in any LIVE (non-dated) rendering.

    A DATED rendering may keep them -- AGENTS.md §5.5 rule #6 requires the
    superseded wording to survive verbatim.
    """
    live = _live_blob(matrix)
    for token in (X1_A, X1_B, X1_C, X2, X3, X4):
        assert token not in live, token
        assert token.lower() not in live.lower(), token
    # the token that is NOT a key of this record may not be pointed at
    assert "frontier_axis_mapping" not in live


def test_the_cross_references_are_not_stale_and_carry_no_contested_wording(
    matrix,
):
    """P4: the two `cross_references` entries that name the contested
    renderings must not carry a stale assertion, nor the contested wording.

    The divergence census found BOTH defects in the record as it stood:
    each entry still asserted 'the emitted string is UNCHANGED' -- which
    stopped being true when Wf-A180 replaced those two strings -- and the
    second quoted 'NEITHER is a connected-component count' in a LIVE
    rendering.  Both are repaired; the previous strings are preserved
    VERBATIM in the dated reconciliation key.
    """
    cr = matrix["band_provenance"]["competing_reading_and_what_this_does_not_settle"]
    refs = cr["cross_references"]
    assert len(refs) == 5
    for ref in refs[:2]:
        assert "the emitted string is UNCHANGED" not in ref, ref
        assert "UNCHANGED" not in ref, ref
    assert "connected-component count" not in refs[1], refs[1]
    # the previous values survive verbatim in the dated reconciliation key
    recon = matrix["band_provenance"][m.RECONCILIATION_KEY]
    superseded = {s["path"]: s["rendering"] for s in recon["superseded_renderings"]}
    base = "band_provenance/competing_reading_and_what_this_does_not_settle"
    for idx in (0, 1):
        path = f"{base}/cross_references[{idx}]"
        assert path in superseded, path
        assert "the emitted string is UNCHANGED" in superseded[path], path
    assert (
        "NEITHER is a connected-component count"
        in superseded[f"{base}/cross_references[1]"]
    )
    # and the change is named in the dated key
    joined = "\n".join(recon["changed_in_this_record"])
    assert "cross_references[0]" in joined
    assert "cross_references[1]" in joined


def test_the_markdown_body_carries_no_contested_claim(matrix):
    """P4: the md BODY -- everything before the dated sections -- must not
    carry a contested claim either.

    The record's live block was already clean; the markdown body on disk
    was NOT, and a re-run would have carried that defect forward.
    """
    body = m.render_markdown_body(matrix)
    for token in (X1_A, X1_B, X1_C, X2, X3, X4):
        assert token not in body, token
        assert token.lower() not in body.lower(), token


def test_dated_keys_preserve_the_superseded_text_verbatim(matrix):
    """P3/P5: the dated keys must keep the superseded wording verbatim.

    They are the ONLY place the contested wording may live now.
    """
    dated = "\n".join(_dated_strings(matrix))
    assert "THE BAND IS NOT A CONNECTIVITY BAND" in dated
    assert "THE BAND'S UPPER ENDPOINT IS NOT A CONNECTIVITY NUMBER AT ALL" in dated
    assert "every candidate endpoint of the band" in dated
    # the Wf-A177-era canonical clause and the stale locator survive too
    assert "frontier_axis_mapping" in dated
    cr = matrix["band_provenance"]["competing_reading_and_what_this_does_not_settle"]
    assert cr["canonical_reading_safe_band_statement_superseded_2026_11_01"]
    assert cr["readings"][1]["locator_superseded_2026_11_01"]
    # the drift record that names the robust locators.  2026-11-02
    # (Wf-A194): the two keys that asserted CURRENT line numbers were
    # RENAMED to dated `historical_line_numbers_measured_2026_11_01` keys
    # -- a key named `current_...` is itself a LIVE key presenting line
    # numbers as current -- and their VALUES survive VERBATIM.
    assert cr["line_number_note"]
    for idx in (0, 1):
        hist = cr["readings"][idx]["historical_line_numbers_measured_2026_11_01"]
        assert hist.strip(), idx
        assert "current_line_numbers_on_disk" not in hist, idx
        assert cr["readings"][idx]["historical_line_numbers_measured_note_2026_11_01"]
    # the OLD key name survives only inside DATED strings
    assert "current_line_numbers_on_disk" not in _live_blob(matrix)


def test_live_locators_are_json_paths(matrix):
    """P5: the robust locators are JSON PATHS -- never line numbers, and
    never a token that is not a key of this record."""
    cr = matrix["band_provenance"]["competing_reading_and_what_this_does_not_settle"]
    assert "`band_provenance`" in cr["readings"][0]["locator"]
    assert "`sources`.`EF-TALFM`" in cr["readings"][1]["locator"]
    for r in cr["readings"]:
        assert not re.search(r"\(:\d", r["locator"]), r["locator"]
        assert not re.search(r"\s:\d", r["locator"]), r["locator"]
        assert "frontier_axis_mapping" not in r["locator"], r["locator"]


def test_both_readings_stay_visible_and_one_canonical_clause(matrix):
    """P5: both competing readings remain visible, and there is exactly ONE
    live canonical clause."""
    cr = matrix["band_provenance"]["competing_reading_and_what_this_does_not_settle"]
    assert len(cr["readings"]) == 2
    assert {r["reading"] for r in cr["readings"]} == {"A", "B"}
    assert "92.7" in json.dumps(cr["readings"][0], ensure_ascii=False)
    assert "92.98" in json.dumps(cr["readings"][1], ensure_ascii=False)
    assert "94.33" in json.dumps(cr["readings"][1], ensure_ascii=False)
    clause = cr["canonical_reading_safe_band_statement"]
    live = _live_strings(matrix["band_provenance"])
    assert sum(1 for s in live if s == clause) == 1
    up = clause.upper()
    assert "EXTERNAL" in up
    assert "NOT ONE HOMOGENEOUS QUANTITY" in up
    assert "NOT ON OUR SCALE" in up
    assert "SOURCE CELLS ARE NOT SETTLED" in up


def test_the_canonical_live_rendering_names_the_band(matrix):
    """The tie-break: the canonical live rendering names '92.7-94 %' AND
    states all four invariants.  The competing rendering that does not name
    the band is preserved verbatim in the dated key."""
    bp = matrix["band_provenance"]
    assert "92.7-94 %" in bp["headline_finding"]
    up = bp["headline_finding"].upper()
    for token in (
        "EXTERNAL",
        "NOT ONE HOMOGENEOUS QUANTITY",
        "NOT ON OUR SCALE",
        "SOURCE CELLS ARE NOT SETTLED",
    ):
        assert token in up
    # the rendering it replaced survives verbatim, in the dated key
    dated = "\n".join(_dated_strings(matrix))
    assert bp["headline_finding_superseded_2026_11_01"] in dated


# ---------------------------------------------------------------------------
# The markdown artefact
# ---------------------------------------------------------------------------


def test_markdown_emits_the_two_dated_sections_verbatim(matrix):
    """P3: the two dated correction sections must be in the emitted markdown
    VERBATIM -- a re-run used to delete them."""
    md = m.render_markdown(matrix)
    assert m.WF_A177_MD_SECTION in md
    assert m.WF_A180_MD_SECTION in md
    assert m.WF_A177_MD_SECTION.startswith("## Wf-A177 CORRECTION (2026-11-01)")
    assert m.WF_A180_MD_SECTION.startswith("## Wf-A180 CORRECTION (2026-11-01)")
    # each appears exactly once
    assert md.count(m.WF_A177_MD_SECTION) == 1
    assert md.count(m.WF_A180_MD_SECTION) == 1
    # and they match the artefact on disk byte-for-byte
    disk_md = m.DEFAULT_OUT_DIR / "frontier_matrix.md"
    if disk_md.is_file():
        text = disk_md.read_text(encoding="utf-8")
        i177 = text.index("## Wf-A177 CORRECTION (2026-11-01)")
        i180 = text.index("## Wf-A180 CORRECTION (2026-11-01)")
        assert text.startswith(m.WF_A177_MD_SECTION, i177)
        assert text.startswith(m.WF_A180_MD_SECTION, i180)


def test_markdown_carries_the_reconciliation_section(matrix):
    """The third dated section records what changed, why, and the byte
    evidence -- and it is rendered from the record's own dated key."""
    md = m.render_markdown(matrix)
    recon = matrix["band_provenance"][m.RECONCILIATION_KEY]
    assert recon["heading"] in md
    assert recon["what_changed"] in md
    assert recon["why"] in md
    assert "**The byte evidence.**" in md
    for item in recon["changed_in_this_record"]:
        assert item in md
    for item in recon["superseded_renderings"]:
        assert item["rendering"] in md


def test_markdown_source_order_matches_the_json_source_order(matrix):
    """The md body must list the sources in the SAME ORDER as the JSON ships.

    The JSON is written with `sort_keys=True`; the body used to render them
    in insertion order, so the md never matched the json it ships.
    """
    body = m.render_markdown_body(matrix)
    names = sorted(matrix["sources"])
    positions = [body.index(f"## {name}\n") for name in names]
    assert positions == sorted(positions), positions
    parsed = json.loads(
        json.dumps(matrix, indent=2, sort_keys=True, ensure_ascii=False)
    )
    assert list(parsed["sources"]) == names


def test_the_reconciliation_key_records_what_changed_and_the_evidence(matrix):
    """P3: ONE dated key records what changed, why, and the byte evidence."""
    bp = matrix["band_provenance"]
    assert m.RECONCILIATION_KEY in bp
    recon = bp[m.RECONCILIATION_KEY]
    assert recon["dated"] == "2026-11-01"
    assert recon["what_this_is"]
    assert recon["why"]
    assert recon["what_it_does_not_change"]
    assert recon["producer_now_reproduces_the_artefact"]
    assert recon["append_only"]
    # every changed live string is named, with the superseded rendering kept
    assert len(recon["changed_in_this_record"]) == 8
    assert len(recon["superseded_renderings"]) == 7
    for item in recon["superseded_renderings"]:
        assert item["path"]
        assert item["rendering"]
        assert item["note"]
    # the byte evidence is present and numeric
    ev = recon["byte_evidence"]
    assert set(ev) == {
        "md_body_chars",
        "wf_a177_md_section_chars",
        "wf_a180_md_section_chars",
        "record_leaf_count",
        "record_numeric_leaf_count",
        "numeric_leaves_moved_by_this_repair",
    }
    assert ev["numeric_leaves_moved_by_this_repair"] == 0
    assert ev["md_body_chars"] > 0
    assert ev["wf_a177_md_section_chars"] == len(m.WF_A177_MD_SECTION)
    assert ev["wf_a180_md_section_chars"] == len(m.WF_A180_MD_SECTION)
    # and it agrees with the matrix it describes
    assert ev["record_leaf_count"] == len(dict(_leaf_paths(matrix)))
    assert ev["record_numeric_leaf_count"] == sum(
        1
        for _, value in _leaf_paths(matrix)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


# ---------------------------------------------------------------------------
# Wf-A194 (2026-11-02): the LOCATOR CLASS + the GeoLDM source entry
# ---------------------------------------------------------------------------
#
# A LINE NUMBER is a PERISHABLE locator: this record keeps growing, so every
# line number cited INTO it rots, and a note that asserts a set of CURRENT
# line numbers is STALE BY CONSTRUCTION.  The LIVE locator for any part of
# this record is its JSON PATH.  A line number may appear ONLY inside a
# DATED historical record that says, in the same place, that it is a dated
# historical snapshot.
#
# The two guards below are FUNCTIONS, not bare assertions, so that the
# "the guard bites" test can call them against a deliberately broken COPY.


#: a backticked JSON path followed by a line number -- e.g. "`sources` :380"
LINE_NUMBER_LOCATOR_RE = re.compile(r"`[^`]*`\s*:\s*\d")

#: a LIVE key NAME that asserts CURRENCY for line numbers
LIVE_CURRENCY_KEY_RE = re.compile(
    r"line.*(current|on_disk)|(current|on_disk).*line"
)


def _assert_no_live_line_number_locator(record):
    """Assert that no LIVE key presents a line number as a current locator.

    Two things are checked: the VALUE of every LIVE string leaf (a
    backticked JSON path must not be followed by a line number) and the
    NAME of every LIVE key (a key called ``current_line_numbers_on_disk``
    asserts currency whatever it holds).
    """
    offenders = []
    for path, value in _leaf_paths(record):
        if not isinstance(value, str) or _is_dated(path):
            continue
        hit = LINE_NUMBER_LOCATOR_RE.search(value)
        if hit:
            offenders.append(f"{path}: {hit.group(0)!r}")
    assert not offenders, (
        "a LIVE rendering presents a line number as a current locator: "
        + "; ".join(offenders)
    )
    bad_keys = []
    for path, key in _live_keys(record):
        if LIVE_CURRENCY_KEY_RE.search(key):
            bad_keys.append(path)
    assert not bad_keys, (
        "a LIVE key name asserts currency for a line number: "
        + "; ".join(bad_keys)
    )


def _assert_geoldm_entry(record):
    """Assert `sources`.`GeoLDM` exists and carries the F4 disagreement."""
    assert "GeoLDM" in record["sources"], "sources.GeoLDM is missing"
    g = record["sources"]["GeoLDM"]
    # the same field shape as the other sources
    for key in (
        "label",
        "citation",
        "representation",
        "reported_band",
        "reported_numbers",
        "cells",
        "caveats_stated_by_the_source",
        "extra_observations",
    ):
        assert key in g, key
    cit = g["citation"]
    assert cit["arxiv_id"] == "2305.01140"
    assert cit["version_used"]
    assert cit["retrieved_at"] == record["retrieval_date"]
    assert cit["retrieval_url"].startswith("http")
    # the same four breaks as every other source
    assert set(g["cells"]) == set(record["breaks"])
    # GeoLDM's OWN reported numbers carry 92.7 and 93.8
    values = {r["value"] for r in g["reported_numbers"]}
    assert 92.7 in values, values
    assert 93.8 in values, values
    # HLTF's transcript numeral 93.6 must NOT be presented as a measured
    # GeoLDM cell -- the two tables disagree and the record must not pick.
    assert 93.6 not in values, values
    # the disagreement is RECORDED, in both numerals, AS a disagreement
    blob = json.dumps(g, ensure_ascii=False)
    assert "93.8" in blob and "93.6" in blob
    assert "DISAGREE" in blob.upper()
    # the F3 disconnected-components fact is carried, with the paper's own
    # wording for the PROPOSED remedy
    assert "disconnected components" in blob
    assert "filtering the smaller components" in blob
    # the dated summary records the F4 disagreement too, and names both
    lc = record["band_provenance"][m.LOCATOR_CLASS_KEY]
    f4 = lc["the_geo_ldm_entry"]["the_f4_disagreement"]
    assert "93.8" in f4 and "93.6" in f4
    assert "DISAGREE" in f4.upper()
    # and the F3 fact is there, in the dated key
    f3 = lc["the_geo_ldm_entry"]["the_f3_fact_and_why_it_matters_here"]
    assert "disconnected components" in f3
    # the entry's own locator class is a JSON PATH, never a line number
    assert "`sources`.`GeoLDM`" in lc["the_geo_ldm_entry"]["locator_class"]


def test_no_live_key_presents_a_line_number_as_a_current_locator(matrix):
    """Wf-A194 (2026-11-02): the LIVE locator of this record is a JSON PATH.

    Measured before the repair: 6 of the 8 positions the record's own
    line-number note named were already WRONG, because the record keeps
    growing.  The line numbers were therefore DEMOTED into dated sibling
    keys instead of being corrected.

    The guard is applied to the PRODUCER'S BUILD *and* to the PUBLISHED
    ARTEFACT on disk -- a producer that is clean but an artefact that was
    never republished is exactly the defect class this wave closes.
    """
    _assert_no_live_line_number_locator(matrix)
    on_disk = _record_on_disk()
    if on_disk is not None:
        _assert_no_live_line_number_locator(on_disk)
    # the demotion is real: the line numbers survive ONLY inside dated keys
    dated = "\n".join(_dated_strings(matrix))
    assert ":380" in dated
    assert ":426-427" in dated
    # and the record names the dated siblings that hold them
    lc = matrix["band_provenance"][m.LOCATOR_CLASS_KEY]
    preserved = lc["the_demoted_line_numbers_are_preserved_verbatim_in"]
    assert len(preserved) == 3
    for item in preserved:
        assert item.startswith("`band_provenance`"), item


def test_sources_geoldm_exists_and_records_the_f4_disagreement(matrix):
    """Wf-A194 (2026-11-02): `sources`.`GeoLDM` exists, and the F4
    disagreement (93.8 +/- 0.4 vs 93.6 +/- 0.2) is recorded explicitly.

    GeoLDM's OWN paper prints Valid = 93.8 +/- 0.4 for its own QM9 row;
    HLTF's Table 1(a)/1(b) print the GeoLDM column's Val as 93.6 +/- 0.2.
    Neither numeral is a band endpoint, and neither may be silently picked.

    The guard is applied to the PRODUCER'S BUILD *and* to the PUBLISHED
    ARTEFACT on disk.
    """
    _assert_geoldm_entry(matrix)
    on_disk = _record_on_disk()
    if on_disk is not None:
        _assert_geoldm_entry(on_disk)


def test_the_locator_class_guard_bites_on_a_broken_copy(matrix):
    """Wf-A194 (2026-11-02): the two new guards must DETECT a break.

    A guard that can only PASS is not a guard.  The broken copies below
    live in pytest's ``tmp_path`` (a /tmp directory) and are never written
    over the published artefact.
    """
    # (a) a LIVE rendering that pairs a JSON path with a line number
    broken = json.loads(json.dumps(matrix))
    broken["band_provenance"]["competing_reading_and_what_this_does_not_settle"][
        "line_number_note"
    ] = "the current positions are `sources` :380 -> `EF-TALFM` :381"
    with pytest.raises(AssertionError) as excinfo:
        _assert_no_live_line_number_locator(broken)
    assert "presents a line number as a current locator" in str(excinfo.value)

    # (b) a LIVE key NAME that asserts currency for line numbers
    broken_b = json.loads(json.dumps(matrix))
    broken_b["band_provenance"]["competing_reading_and_what_this_does_not_settle"][
        "current_line_numbers_on_disk"
    ] = "a record of drift, not a locator"
    with pytest.raises(AssertionError) as excinfo_b:
        _assert_no_live_line_number_locator(broken_b)
    assert "LIVE key name asserts currency" in str(excinfo_b.value)

    # (c) `sources`.`GeoLDM` removed
    broken_c = json.loads(json.dumps(matrix))
    del broken_c["sources"]["GeoLDM"]
    with pytest.raises(AssertionError) as excinfo_c:
        _assert_geoldm_entry(broken_c)
    assert "sources.GeoLDM is missing" in str(excinfo_c.value)

    # (d) the F4 disagreement silently resolved to ONE numeral
    broken_d = json.loads(json.dumps(matrix))
    for row in broken_d["sources"]["GeoLDM"]["reported_numbers"]:
        if row["value"] == 93.8:
            row["value"] = 93.6
    with pytest.raises(AssertionError) as excinfo_d:
        _assert_geoldm_entry(broken_d)
    assert "93.8" in str(excinfo_d.value)
