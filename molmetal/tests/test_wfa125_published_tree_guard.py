# SPDX-License-Identifier: Apache-2.0
"""Wf-A125 — the published-tree write guard's own regression coverage.

THE GUARD IS IMPORTED, NOT RE-IMPLEMENTED
------------------------------------------
Every helper below comes from :mod:`molmetal.tests.conftest`.  A guard that
is re-implemented inside its own test proves nothing about the guard that
actually runs.

WHAT THIS FILE COVERS (and why the wave created it)
----------------------------------------------------
``molmetal/tests/test_published_tree_guard.py`` already covered the guard's
three original cases, but it is **UNTRACKED** (``git status`` says ``??``),
so a clean checkout of HEAD has NO guard coverage at all.  Wf-A125 G10
therefore ships a tracked-in-the-next-commit file that carries the same
three cases PLUS the three the wave added:

* **G2** — the guard covered three tree ROOTS and nothing else.  The registry
  SSOT (``registry/{metrics,datasets,protocols}.yaml``) and the ledger SSOT
  (``molmetal/reports/run_ledger.jsonl``) were unguarded, so a
  write-then-restore to identical bytes on either was invisible.
  ``molmetal/reports/`` CANNOT be a tree root (486 files under
  ``molmetal/scripts/`` mention it), so the guard uses a CURATED FILE list.
* **G3** — the snapshot recorded sha256 only, so a write-then-restore to the
  SAME bytes was invisible even on a guarded path.  The snapshot now records
  ``(sha256, mtime_ns, size, ino)`` and the diff gains a ``touched`` kind.
* **G4** — the declaration file's three entries are keyed by SCRIPTS, so all
  three were permanently inert as authorisations.  The loader now classifies
  a key as an AUTHORISATION only if it names a test module; every other key
  is a CAPABILITY RECORD.

HONEST CLOSURE NOTE: this file is created by the wave and the wave does not
commit.  It is ``??`` (untracked) until the next commit — see the wave
close-out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from molmetal.tests.conftest import (
    DECLARATIONS_PATH,
    PUBLISHED_FILE_PATHS,
    PUBLISHED_TREE_ROOTS,
    diff_published_trees,
    find_undeclared_writes,
    format_violations,
    is_test_module_key,
    load_capability_records,
    load_declarations,
    snapshot_published_trees,
    split_declarations,
)

THIS_MODULE = "molmetal/tests/test_wfa125_published_tree_guard.py"

REGISTRY_WRITER = "molmetal/tests/test_registry_generate_import_path.py"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write(root: Path, relpath: str, text: str) -> Path:
    p = root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _replace_with_same_bytes(path: Path, text: str) -> None:
    """Rewrite ``path`` with IDENTICAL bytes through a NEW inode.

    This is the real-world shape (AGENTS.md §5.6 mandates
    ``atomic_write_file`` → ``os.replace``): the bytes are unchanged, the
    inode is not.  A sha-only diff cannot see it; the ``touched`` kind can.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _curated(repo_root: Path) -> dict:
    """A synthetic repo carrying every curated published file."""
    _write(repo_root, "registry/metrics.yaml", "metrics: {}\n")
    _write(repo_root, "registry/datasets.yaml", "datasets: {}\n")
    _write(repo_root, "registry/protocols.yaml", "protocols: {}\n")
    _write(repo_root, "molmetal/reports/run_ledger.jsonl", '{"metric_id": "m"}\n')
    return {
        "registry/metrics.yaml": "metrics: {}\n",
        "registry/datasets.yaml": "datasets: {}\n",
        "registry/protocols.yaml": "protocols: {}\n",
        "molmetal/reports/run_ledger.jsonl": '{"metric_id": "m"}\n',
    }


# ---------------------------------------------------------------------------
# G10 — the three original cases (unchanged tree / declared / undeclared)
# ---------------------------------------------------------------------------
def test_guard_passes_on_unchanged_tree(tmp_path) -> None:
    """(G10-i) An unchanged published tree produces no violations."""
    _write(tmp_path, "metrics/by_metric/unchanged.json", '{"a": 1}')
    before = snapshot_published_trees(tmp_path)
    after = snapshot_published_trees(tmp_path)
    assert before, "synthetic tree was not picked up by the snapshot"
    violations = find_undeclared_writes(before, after, {}, [THIS_MODULE])
    assert violations == [], violations


def test_guard_passes_on_declared_write(tmp_path) -> None:
    """(G10-ii) A write DECLARED by an active test module is allowed."""
    relpath = "metrics/by_metric/declared.json"
    _write(tmp_path, relpath, '{"a": 1}')
    before = snapshot_published_trees(tmp_path)
    _write(tmp_path, relpath, '{"a": 2}')
    after = snapshot_published_trees(tmp_path)
    declarations = {THIS_MODULE: [{"path": relpath, "reason": "synthetic"}]}
    violations = find_undeclared_writes(before, after, declarations, [THIS_MODULE])
    assert violations == [], violations


def test_guard_fails_on_undeclared_write(tmp_path) -> None:
    """(G10-iii) An UNDECLARED write fails, and the message names the file."""
    relpath = "metrics/by_metric/undeclared.json"
    _write(tmp_path, relpath, '{"a": 1}')
    before = snapshot_published_trees(tmp_path)
    _write(tmp_path, relpath, '{"a": 2}')
    after = snapshot_published_trees(tmp_path)
    violations = find_undeclared_writes(before, after, {}, [THIS_MODULE])
    assert len(violations) == 1, violations
    violation = violations[0]
    assert violation["path"] == relpath
    assert violation["kind"] == "modified"
    assert violation["before_sha"] != violation["after_sha"]
    message = format_violations(violations, [THIS_MODULE])
    assert "undeclared.json" in message, message
    assert violation["after_sha"] in message, message


# ---------------------------------------------------------------------------
# G2 — the curated FILE list
# ---------------------------------------------------------------------------
def test_the_curated_list_covers_the_registry_and_the_ledger() -> None:
    """(G2-a) The registry SSOT and the ledger SSOT are in the guarded set."""
    paths = {str(spec["path"]) for spec in PUBLISHED_FILE_PATHS}
    assert "registry/metrics.yaml" in paths, paths
    assert "registry/datasets.yaml" in paths, paths
    assert "registry/protocols.yaml" in paths, paths
    assert "molmetal/reports/run_ledger.jsonl" in paths, paths
    # The ledger has NO authorised writer: no test may append to the real
    # ledger (every ledger test uses a tmp ledger).
    ledger_spec = next(
        spec
        for spec in PUBLISHED_FILE_PATHS
        if spec["path"] == "molmetal/reports/run_ledger.jsonl"
    )
    assert tuple(ledger_spec["authorised_writers"]) == (), ledger_spec


def test_the_curated_list_is_closed(tmp_path) -> None:
    """(G2-d) A path that is NOT named is NOT guarded."""
    assert "registry/narrative_claims.yaml" not in {
        str(spec["path"]) for spec in PUBLISHED_FILE_PATHS
    }
    assert "registry/file_provenance.json" not in {
        str(spec["path"]) for spec in PUBLISHED_FILE_PATHS
    }
    _write(tmp_path, "registry/metrics.yaml", "metrics: {}\n")
    _write(tmp_path, "registry/narrative_claims.yaml", "claims: {}\n")
    before = snapshot_published_trees(tmp_path)
    assert "registry/metrics.yaml" in before
    assert "registry/narrative_claims.yaml" not in before, sorted(before)


def test_a_curated_file_write_is_reported_when_the_writer_is_unauthorised(
    tmp_path,
) -> None:
    """(G2-a) A write to registry/metrics.yaml by an unknown module FAILS."""
    _curated(tmp_path)
    before = snapshot_published_trees(tmp_path)
    _write(tmp_path, "registry/metrics.yaml", "metrics: {changed: true}\n")
    after = snapshot_published_trees(tmp_path)
    violations = find_undeclared_writes(before, after, {}, [THIS_MODULE])
    paths = {v["path"] for v in violations}
    assert "registry/metrics.yaml" in paths, violations


def test_the_authorised_writer_is_not_reported(tmp_path) -> None:
    """(G2-b) The SAME write by the authorised writer is ALLOWED."""
    _curated(tmp_path)
    before = snapshot_published_trees(tmp_path)
    _write(tmp_path, "registry/metrics.yaml", "metrics: {changed: true}\n")
    after = snapshot_published_trees(tmp_path)
    violations = find_undeclared_writes(
        before, after, {}, [REGISTRY_WRITER, THIS_MODULE]
    )
    paths = {v["path"] for v in violations}
    assert "registry/metrics.yaml" not in paths, violations


def test_the_ledger_is_guarded_against_every_writer(tmp_path) -> None:
    """(G2-c) The ledger has no authorised writer, so ANY write is reported."""
    _curated(tmp_path)
    before = snapshot_published_trees(tmp_path)
    _write(tmp_path, "molmetal/reports/run_ledger.jsonl", '{"metric_id": "n"}\n')
    after = snapshot_published_trees(tmp_path)
    violations = find_undeclared_writes(
        before, after, {}, [REGISTRY_WRITER, THIS_MODULE]
    )
    paths = {v["path"] for v in violations}
    assert "molmetal/reports/run_ledger.jsonl" in paths, violations


# ---------------------------------------------------------------------------
# G3 — the ``touched`` kind
# ---------------------------------------------------------------------------
def test_an_identical_byte_rewrite_is_reported_as_touched(tmp_path) -> None:
    """(G3) A write-then-restore to the SAME bytes is a violation."""
    _curated(tmp_path)
    before = snapshot_published_trees(tmp_path)
    path = tmp_path / "registry/metrics.yaml"
    original = path.read_text(encoding="utf-8")
    _replace_with_same_bytes(path, original)
    after = snapshot_published_trees(tmp_path)
    diff = diff_published_trees(before, after)
    assert "registry/metrics.yaml" in diff, sorted(diff)
    assert diff["registry/metrics.yaml"]["kind"] == "touched", diff
    entry = diff["registry/metrics.yaml"]
    assert entry["before"]["sha256"] == entry["after"]["sha256"]
    assert entry["before"]["ino"] != entry["after"]["ino"], (
        "the synthetic rewrite must change the inode, otherwise the case "
        "under test is not the one that matters"
    )
    violations = find_undeclared_writes(before, after, {}, [THIS_MODULE])
    kinds = {v["path"]: v["kind"] for v in violations}
    assert kinds["registry/metrics.yaml"] == "touched", kinds


def test_an_in_place_identical_rewrite_is_reported_as_touched(tmp_path) -> None:
    """(G3) The in-place shape (same inode, moved mtime) is caught too."""
    _curated(tmp_path)
    before = snapshot_published_trees(tmp_path)
    path = tmp_path / "registry/metrics.yaml"
    original = path.read_text(encoding="utf-8")
    path.write_text(original, encoding="utf-8")
    # Make the mtime move deterministic: a filesystem with coarse mtime
    # granularity would otherwise hide the case under test.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    after = snapshot_published_trees(tmp_path)
    diff = diff_published_trees(before, after)
    assert diff["registry/metrics.yaml"]["kind"] == "touched", diff


def test_a_touched_violation_is_named_in_the_message(tmp_path) -> None:
    """(G3) The failure message says WHICH kind of mutation it was."""
    _curated(tmp_path)
    before = snapshot_published_trees(tmp_path)
    path = tmp_path / "registry/metrics.yaml"
    _replace_with_same_bytes(path, path.read_text(encoding="utf-8"))
    after = snapshot_published_trees(tmp_path)
    violations = find_undeclared_writes(before, after, {}, [THIS_MODULE])
    message = format_violations(violations, [THIS_MODULE])
    assert "touched" in message, message
    assert "registry/metrics.yaml" in message, message


# ---------------------------------------------------------------------------
# G4 — authorisation vs capability record
# ---------------------------------------------------------------------------
def test_only_a_test_module_key_is_an_authorisation() -> None:
    """(G4-a) A script key is a CAPABILITY RECORD, never an authorisation."""
    assert is_test_module_key(THIS_MODULE) is True
    assert is_test_module_key("molmetal/scripts/cfm_fragmentation_audit.py") is False
    assert is_test_module_key("tests/test_fused_ops.py") is True
    assert is_test_module_key("molmetal/scripts/registry.py") is False
    assert is_test_module_key("") is False


def test_the_declaration_file_splits_into_zero_authorisations() -> None:
    """(G4-a) The shipped file yields 0 authorisations and 3 capability records."""
    declarations = load_declarations(DECLARATIONS_PATH)
    authorisations, capability_records = split_declarations(declarations)
    assert authorisations == {}, authorisations
    assert len(capability_records) == 3, sorted(capability_records)
    assert set(capability_records) == {
        "molmetal/scripts/cfm_fragmentation_audit.py",
        "molmetal/scripts/metal_activity_transfer_tie_sensitivity.py",
        "molmetal/scripts/dmpnn_sha256_audit.py",
    }, sorted(capability_records)
    assert len(load_capability_records(DECLARATIONS_PATH)) == 3


def test_a_capability_record_cannot_authorise_a_write(tmp_path) -> None:
    """(G4-b) The inverse fragility is CLOSED.

    Before G4 the three script-keyed entries were inert only BY ACCIDENT (a
    script path can never be an active test module).  If the matcher were
    loosened to "any key in the file", all three would start authorising
    writes into ``metrics/by_metric/``.  Now the contract rejects them even
    when one of them is passed AS the active module.
    """
    relpath = "metrics/by_metric/cfm_fragmentation_audit.json"
    _write(tmp_path, relpath, '{"a": 1}')
    before = snapshot_published_trees(tmp_path)
    _write(tmp_path, relpath, '{"a": 2}')
    after = snapshot_published_trees(tmp_path)
    declarations = {
        "molmetal/scripts/cfm_fragmentation_audit.py": [
            {"path": relpath, "reason": "capability record, not a test"}
        ]
    }
    violations = find_undeclared_writes(
        before, after, declarations, ["molmetal/scripts/cfm_fragmentation_audit.py"]
    )
    assert len(violations) == 1, violations
    assert violations[0]["path"] == relpath


def test_the_declaration_file_is_unchanged_and_still_parses() -> None:
    """(G4) The JSON is byte-identical: the loader now enforces what it says."""
    raw = json.loads(DECLARATIONS_PATH.read_text(encoding="utf-8"))
    assert set(raw) == {
        "molmetal/scripts/cfm_fragmentation_audit.py",
        "molmetal/scripts/metal_activity_transfer_tie_sensitivity.py",
        "molmetal/scripts/dmpnn_sha256_audit.py",
    }, sorted(raw)
    for module, entries in raw.items():
        assert entries, module
        for entry in entries:
            assert entry["path"].startswith("metrics/by_metric/"), entry
            assert entry["reason"].strip(), entry


def test_load_declarations_rejects_a_malformed_file(tmp_path) -> None:
    """A malformed declaration file must raise, not silently allow everything."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"mod": "not-a-list"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_declarations(bad)


def test_the_guard_still_covers_the_three_published_trees() -> None:
    """The original tree roots are unchanged by this wave."""
    assert {p.as_posix() for p in PUBLISHED_TREE_ROOTS} == {
        "metrics/by_metric",
        "metrics/paper_tables",
        "molmetal/metrics/by_metric",
    }
