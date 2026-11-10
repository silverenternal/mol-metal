"""Wf-A199 (2026-11-02) -- the producer<->artefact PARITY CENSUS and the guard
for INSTANCE 4's repair.

WHAT THIS FILE GUARDS
---------------------
1. The published census report is well formed: every candidate carries a
   classification from the tool's own enum and a tier from the tool's own
   enum, and the parity-gap list is EXACTLY the DATED ``PRODUCER-MISSING-KEY``
   subset of the candidates.
2. The census's candidate list is REPRODUCIBLE: re-reading each candidate's
   artefact from disk and re-running the tool's own record-collection and
   key-pattern functions reproduces the same (artefact, key, key_path,
   match_reasons, tier) rows.
3. INSTANCE 4 IS CLOSED: the dated rule-#6 record
   ``comparability.superseded_frontier_2026_11_02_wfa194`` is emitted by the
   producer ``molmetal/scripts/wf_a155_ligand_connected_rate.py``, the census
   records it as ``PRODUCER-EMITTED`` (not as a parity gap), and the census's
   own evidence names the constant that carries it.
4. The producer's emitted value is BYTE-IDENTICAL to the published artefact's
   value (AGENTS.md §5.5 rule #6: the published rendering is preserved
   VERBATIM).
5. The published artefact's ``comparability`` key set EQUALS the producer's
   emitted key set, in the same order.

THE GUARD MUST BITE
-------------------
The last two tests delete the dated key from a /tmp COPY of the producer
source and from a /tmp COPY of the artefact, and assert that the parity guard
RAISES on both.  A guard that cannot fail is not a guard.

CPU-only.  No GPU, no training, no sampling, no metric writes.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

CENSUS_TOOL_REL = "molmetal/scripts/wfa199_parity_census.py"
CENSUS_TOOL = REPO / CENSUS_TOOL_REL
CENSUS_REPORT_REL = "molmetal/reports/wfa199_parity_census/report.json"
CENSUS_REPORT = REPO / CENSUS_REPORT_REL

EMITTER_REL = "molmetal/scripts/wf_a155_ligand_connected_rate.py"
EMITTER = REPO / EMITTER_REL

C7_REL = (
    "molmetal/reports/wfa162_c7_pool_ligand_connected_rate/"
    "ligand_connected_rate_c7.json"
)
C7_ARTEFACT = REPO / C7_REL

DATED_KEY = "superseded_frontier_2026_11_02_wfa194"
DATED_CONSTANT = "SUPERSEDED_FRONTIER_2026_11_02_WFA194"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_census_tool():
    """Import the census tool as a module (it has no import-time side effects)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("wfa199_census_under_test", CENSUS_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparability_dict_node(source: str) -> ast.Dict:
    """The producer's `comparability` dict literal, found by its shape."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            k.value
            for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
        if "statement" in keys and "breaks" in keys and "frontier_band" in keys:
            found.append(node)
    assert len(found) == 1, (
        f"expected exactly ONE `comparability` dict literal in {EMITTER_REL}, "
        f"found {len(found)}"
    )
    return found[0]


def _comparability_keys_from_source(source: str) -> list[str]:
    """The key set the producer's `comparability` dict emits, in order."""
    node = _comparability_dict_node(source)
    keys = []
    for k in node.keys:
        assert isinstance(k, ast.Constant) and isinstance(k.value, str), (
            "the `comparability` dict literal has a non-literal key; the "
            "emitted key set is then not statically knowable"
        )
        keys.append(k.value)
    return keys


def _published_comparability_keys(doc: dict | None = None) -> list[str]:
    if doc is None:
        doc = json.loads(C7_ARTEFACT.read_text(encoding="utf-8"))
    return list(doc["comparability"].keys())


def _module_dict_constant(source: str, name: str) -> dict:
    """A module-level dict constant, read out of the source with `ast`."""
    tree = ast.parse(source)
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{EMITTER_REL} carries no module-level constant {name}")


def _assert_comparability_parity(
    emitted: list[str], published: list[str], *, source_desc: str
) -> None:
    """THE GUARD.  Raises with a diff-shaped message when they disagree."""
    if emitted == published:
        return
    missing = [k for k in published if k not in emitted]
    extra = [k for k in emitted if k not in published]
    raise AssertionError(
        f"producer<->artefact parity broken ({source_desc}): the producer's "
        f"emitted `comparability` key set differs from the published "
        f"artefact's. missing_from_producer={missing} "
        f"extra_in_producer={extra} emitted={emitted} published={published}"
    )


def _delete_comparability_key_from_source(source: str, key: str) -> str:
    """Return `source` with the given key removed from the comparability dict."""
    node = _comparability_dict_node(source)
    lines = source.splitlines(keepends=True)
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            del lines[k.lineno - 1 : v.end_lineno]
            return "".join(lines)
    raise AssertionError(f"the comparability dict literal carries no key {key}")


def _report() -> dict:
    return json.loads(CENSUS_REPORT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. the published census report is well formed
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CENSUS_REPORT.is_file(), reason="census report not published")
def test_every_candidate_carries_a_classification_and_a_tier():
    census = _load_census_tool()
    rep = _report()
    enum = list(census.CLASSIFICATION_ENUM)
    tiers = list(census.TIER_ENUM)
    assert rep["candidates"], "the census published no candidates at all"
    for c in rep["candidates"]:
        assert c["classification"] in enum, c
        assert c["tier"] in tiers, c
        assert c["match_reasons"], c
        assert c["artefact"] and c["key"] and c["key_path"], c


@pytest.mark.skipif(not CENSUS_REPORT.is_file(), reason="census report not published")
def test_the_parity_gaps_are_exactly_the_dated_producer_missing_candidates():
    rep = _report()
    expected = {
        (c["artefact"], c["key"])
        for c in rep["candidates"]
        if c["classification"] == "PRODUCER-MISSING-KEY" and c["tier"] == "DATED"
    }
    got = {(g["artefact"], g["key"]) for g in rep["parity_gaps"]}
    assert got == expected, (
        "the parity-gap list is not the DATED PRODUCER-MISSING-KEY subset of "
        f"the candidates: extra={sorted(got - expected)} "
        f"missing={sorted(expected - got)}"
    )


# ---------------------------------------------------------------------------
# 2. the candidate list is reproducible from disk
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CENSUS_REPORT.is_file(), reason="census report not published")
def test_every_candidate_row_is_reproducible_from_its_key_name_alone():
    """DISK-INDEPENDENT half of the reproducibility guard.

    The tool decides the tier and the match reasons from the KEY NAME only,
    so every published row must be re-derivable from that name alone.
    """
    census = _load_census_tool()
    rep = _report()
    for c in rep["candidates"]:
        assert census.match_reasons(c["key"]) == c["match_reasons"], c
        tier = "DATED" if census.DATE_TOKEN_RE.search(c["key"]) else "UNDATED"
        assert tier == c["tier"], f"{c['artefact']}:{c['key_path']}"


@pytest.mark.skipif(not CENSUS_REPORT.is_file(), reason="census report not published")
def test_the_candidate_list_is_reproducible_from_disk():
    """DISK-DEPENDENT half of the reproducibility guard.

    Every candidate must still be collected from its artefact by the tool's
    own collector, with the same key path and the same match reasons.  An
    artefact that was REWRITTEN after this census ran belongs to a wave that
    is working in the same tree; the published row is then a snapshot of a
    file that has since moved, so it is reported rather than counted as a
    reproducibility failure.
    """
    census = _load_census_tool()
    rep = _report()
    report_mtime = CENSUS_REPORT.stat().st_mtime
    stale = []
    for c in rep["candidates"]:
        artefact = REPO / c["artefact"]
        assert artefact.is_file(), c["artefact"]
        if artefact.stat().st_mtime > report_mtime:
            stale.append(c["artefact"])
            continue
        doc = json.loads(artefact.read_text(encoding="utf-8"))
        records = census.collect_dated_records(doc)
        hits = [
            r
            for r in records
            if r["key"] == c["key"] and r["key_path"] == c["key_path"]
        ]
        assert hits, (
            f"{c['artefact']}:{c['key_path']} is no longer collected from disk "
            "by the census's own collector -- the published candidate list is "
            "not reproducible"
        )
        assert hits[0]["match_reasons"] == c["match_reasons"], (
            f"{c['artefact']}:{c['key_path']} -- the match reasons changed"
        )
    if stale:
        print(
            "[wfa199] %d candidate artefact(s) were rewritten after the census "
            "ran (a sibling wave in the same tree): %s"
            % (len(stale), sorted(set(stale)))
        )


# ---------------------------------------------------------------------------
# 3. INSTANCE 4 is closed
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CENSUS_REPORT.is_file(), reason="census report not published")
def test_instance_four_is_closed_in_the_published_census():
    rep = _report()
    gaps = {(g["artefact"], g["key"]) for g in rep["parity_gaps"]}
    assert (C7_REL, DATED_KEY) not in gaps, (
        "the c7 dated rule-#6 record is STILL a parity gap -- a producer "
        "re-run would still drop it"
    )
    records = [
        e
        for e in rep["all_dated_records"]
        if e["artefact"] == C7_REL and e["key"] == DATED_KEY
    ]
    assert len(records) == 1, records
    rec = records[0]
    assert rec["tier"] == "DATED", rec
    assert rec["classification"] == "PRODUCER-EMITTED", rec
    own = [h["producer"] for h in rec["evidence"]["own_producer_literal_hits"]]
    assert EMITTER_REL in own, rec["evidence"]
    parity = rec["evidence"].get("value_parity_constant", [])
    assert any(p["constant"] == DATED_CONSTANT for p in parity), rec["evidence"]


# ---------------------------------------------------------------------------
# 4. the producer's value is byte-identical to the published artefact's
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not C7_ARTEFACT.is_file(), reason="c7 artefact not on disk")
def test_the_emitter_emits_the_dated_record_byte_identically():
    source = EMITTER.read_text(encoding="utf-8")
    constant = _module_dict_constant(source, DATED_CONSTANT)
    doc = json.loads(C7_ARTEFACT.read_text(encoding="utf-8"))
    published = doc["comparability"][DATED_KEY]
    assert list(constant.keys()) == list(published.keys()), (
        "the producer's constant and the published artefact's record carry "
        "different KEYS -- rule #6 requires the published rendering verbatim"
    )
    assert json.dumps(constant, ensure_ascii=False, sort_keys=False) == json.dumps(
        published, ensure_ascii=False, sort_keys=False
    ), (
        "the producer's constant and the published artefact's record are not "
        "byte-identical -- rule #6 requires the published rendering verbatim"
    )


@pytest.mark.skipif(not C7_ARTEFACT.is_file(), reason="c7 artefact not on disk")
def test_the_emitter_emits_the_dated_record_through_the_dict_treatment():
    """The key is emitted as `dict(<NAMED CONSTANT>)`, like its two siblings."""
    source = EMITTER.read_text(encoding="utf-8")
    node = _comparability_dict_node(source)
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == DATED_KEY:
            assert isinstance(v, ast.Call), ast.dump(v)
            assert isinstance(v.func, ast.Name) and v.func.id == "dict", ast.dump(v)
            assert len(v.args) == 1 and isinstance(v.args[0], ast.Name), ast.dump(v)
            assert v.args[0].id == DATED_CONSTANT, ast.dump(v)
            return
    raise AssertionError(f"the comparability dict literal emits no key {DATED_KEY}")


# ---------------------------------------------------------------------------
# 5. the published key set equals the producer's emitted key set
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not C7_ARTEFACT.is_file(), reason="c7 artefact not on disk")
def test_the_published_comparability_key_set_equals_the_producers():
    emitted = _comparability_keys_from_source(EMITTER.read_text(encoding="utf-8"))
    _assert_comparability_parity(
        emitted, _published_comparability_keys(), source_desc="repo emitter vs published c7"
    )


# ---------------------------------------------------------------------------
# 6. THE GUARD MUST BITE
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not C7_ARTEFACT.is_file(), reason="c7 artefact not on disk")
def test_the_guard_bites_when_the_dated_key_is_deleted_from_the_emitter(tmp_path):
    source = EMITTER.read_text(encoding="utf-8")
    broken = _delete_comparability_key_from_source(source, DATED_KEY)
    (tmp_path / "wf_a155_broken.py").write_text(broken, encoding="utf-8")
    emitted = _comparability_keys_from_source(broken)
    with pytest.raises(AssertionError) as exc:
        _assert_comparability_parity(
            emitted, _published_comparability_keys(), source_desc="emitter copy"
        )
    message = str(exc.value)
    assert DATED_KEY in message, message
    assert "missing_from_producer" in message, message


@pytest.mark.skipif(not C7_ARTEFACT.is_file(), reason="c7 artefact not on disk")
def test_the_guard_bites_when_the_dated_key_is_deleted_from_the_artefact(tmp_path):
    doc = json.loads(C7_ARTEFACT.read_text(encoding="utf-8"))
    doc["comparability"].pop(DATED_KEY)
    broken_path = tmp_path / "ligand_connected_rate_c7.json"
    broken_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    broken_keys = _published_comparability_keys(
        json.loads(broken_path.read_text(encoding="utf-8"))
    )
    emitted = _comparability_keys_from_source(EMITTER.read_text(encoding="utf-8"))
    with pytest.raises(AssertionError) as exc:
        _assert_comparability_parity(
            emitted, broken_keys, source_desc="artefact copy"
        )
    message = str(exc.value)
    assert DATED_KEY in message, message
    assert "extra_in_producer" in message, message
