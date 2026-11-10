"""Wf-A123 — the published-tree write guard is itself tested.

A guard that cannot fail is not a guard.  This module exercises the
guard's own logic (imported from :mod:`molmetal.tests.conftest`) against
synthetic trees built in ``tmp_path``:

1. **unchanged tree** -> PASS (no violations),
2. **changed and DECLARED** by an active test module -> PASS,
3. **changed and UNDECLARED** -> FAIL, and the failure message names the
   offending file.

It also pins the shipped declaration file: it must parse, and it must
contain exactly the generator scripts the Wf-A123 scout listed in PART 3
(capability records, not test authorisations).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molmetal.tests.conftest import (
    DECLARATIONS_PATH,
    PUBLISHED_TREE_ROOTS,
    diff_published_trees,
    find_undeclared_writes,
    format_violations,
    load_declarations,
    snapshot_published_trees,
)

THIS_MODULE = "molmetal/tests/test_published_tree_guard.py"


def _write_published(root: Path, relpath: str, text: str) -> Path:
    """Write a synthetic published artifact under ``root``."""
    p = root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_guard_passes_on_unchanged_tree(tmp_path) -> None:
    """(i) An unchanged published tree produces no violations."""
    _write_published(tmp_path, "metrics/by_metric/unchanged.json", '{"a": 1}')
    before = snapshot_published_trees(tmp_path)
    after = snapshot_published_trees(tmp_path)
    assert before, "synthetic tree was not picked up by the snapshot"
    violations = find_undeclared_writes(before, after, {}, [THIS_MODULE])
    assert violations == [], violations


def test_guard_passes_on_declared_write(tmp_path) -> None:
    """(ii) A write DECLARED by an active test module is allowed."""
    relpath = "metrics/by_metric/declared.json"
    _write_published(tmp_path, relpath, '{"a": 1}')
    before = snapshot_published_trees(tmp_path)
    _write_published(tmp_path, relpath, '{"a": 2}')
    after = snapshot_published_trees(tmp_path)

    declarations = {
        THIS_MODULE: [{"path": relpath, "reason": "synthetic guard test"}]
    }
    violations = find_undeclared_writes(before, after, declarations, [THIS_MODULE])
    assert violations == [], violations


def test_guard_fails_on_undeclared_write(tmp_path) -> None:
    """(iii) An UNDECLARED write fails, and the message names the file."""
    relpath = "metrics/by_metric/undeclared.json"
    _write_published(tmp_path, relpath, '{"a": 1}')
    before = snapshot_published_trees(tmp_path)
    _write_published(tmp_path, relpath, '{"a": 2}')
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


def test_guard_detects_created_and_deleted(tmp_path) -> None:
    """A created file and a deleted file are both mutations."""
    keep = "metrics/by_metric/keep.json"
    gone = "metrics/by_metric/gone.json"
    _write_published(tmp_path, keep, '{"a": 1}')
    _write_published(tmp_path, gone, '{"a": 1}')
    before = snapshot_published_trees(tmp_path)

    (tmp_path / gone).unlink()
    _write_published(tmp_path, "metrics/by_metric/created.json", '{"a": 1}')
    after = snapshot_published_trees(tmp_path)

    diff = diff_published_trees(before, after)
    assert set(diff) == {
        "metrics/by_metric/gone.json",
        "metrics/by_metric/created.json",
    }, diff

    violations = find_undeclared_writes(before, after, {}, [THIS_MODULE])
    kinds = {v["path"]: v["kind"] for v in violations}
    assert kinds["metrics/by_metric/gone.json"] == "deleted", kinds
    assert kinds["metrics/by_metric/created.json"] == "created", kinds


def test_guard_ignores_declaration_from_inactive_module(tmp_path) -> None:
    """A declaration by a module that did NOT run does not authorise a write."""
    relpath = "metrics/by_metric/other_module.json"
    _write_published(tmp_path, relpath, '{"a": 1}')
    before = snapshot_published_trees(tmp_path)
    _write_published(tmp_path, relpath, '{"a": 2}')
    after = snapshot_published_trees(tmp_path)

    declarations = {
        "molmetal/tests/test_something_else.py": [
            {"path": relpath, "reason": "did not run"}
        ]
    }
    violations = find_undeclared_writes(before, after, declarations, [THIS_MODULE])
    assert len(violations) == 1, violations


def test_declaration_file_parses_and_lists_part3_scripts() -> None:
    """The shipped declaration file is valid and lists the scout's PART 3 modules."""
    assert DECLARATIONS_PATH.is_file(), DECLARATIONS_PATH
    raw = json.loads(DECLARATIONS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    expected = {
        "molmetal/scripts/cfm_fragmentation_audit.py",
        "molmetal/scripts/metal_activity_transfer_tie_sensitivity.py",
        "molmetal/scripts/dmpnn_sha256_audit.py",
    }
    assert set(raw) == expected, sorted(raw)
    for module, entries in raw.items():
        assert entries, module
        for entry in entries:
            assert entry["path"].startswith("metrics/by_metric/"), entry
            assert entry["reason"].strip(), entry


def test_load_declarations_rejects_malformed_file(tmp_path) -> None:
    """A malformed declaration file must raise, not silently allow everything."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"mod": "not-a-list"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_declarations(bad)

    bad2 = tmp_path / "bad2.json"
    bad2.write_text('{"mod": [{"no_path": 1}]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_declarations(bad2)


def test_published_tree_roots_cover_both_metric_trees() -> None:
    """The guard covers metrics/by_metric, metrics/paper_tables and the package tree."""
    rels = {p.as_posix() for p in PUBLISHED_TREE_ROOTS}
    assert rels == {
        "metrics/by_metric",
        "metrics/paper_tables",
        "molmetal/metrics/by_metric",
    }, rels
