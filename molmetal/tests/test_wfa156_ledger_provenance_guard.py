# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Wf-A156 ledger provenance guard.

CPU-only.  Every test builds a THROW-AWAY git repository in ``tmp_path`` and
runs the guard against it, so nothing here reads or writes the repo's own
ledger, registry or metrics trees.

The test that matters most is
:func:`test_bulk_resolver_sees_merge_introduced_version`: ``git log --raw``
prints no diff for a MERGE commit, so a content version introduced by a merge
is invisible without ``-m``.  A resolver that omits ``-m`` calls a provable
row "unprovable" -- a FALSE POSITIVE.  That blind spot was measured on this
repo's own ledger (398 paths: 700 blobs without ``-m``, 754 with it), so it is
pinned here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import wfa156_ledger_provenance_guard as guard  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}"
        )
    return proc.stdout


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repo, "config", "user.email", "guard@example.invalid")
    _git(repo, "config", "user.name", "guard")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    # an empty commit needs something in the index
    if not _git(repo, "diff", "--cached", "--name-only").strip():
        _write(repo, ".keep", "")
        _git(repo, "add", ".keep")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


def _write(repo: Path, rel: str, content: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    metric_id: str,
    source_json: str,
    sha256: Optional[str],
    *,
    value: Any = None,
) -> Dict[str, Any]:
    return {
        "metric_id": metric_id,
        "value": value,
        "source_json": source_json,
        "source_json_sha256": sha256,
        "git_sha": "0" * 40,
        "data_sha": None,
        "config_sha": None,
        "protocol_id": None,
        "host": "pytest",
        "timestamp": "2026-11-01T00:00:00+00:00",
        "seed": True,
    }


def _write_ledger(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def _run(
    tmp_path: Path,
    repo: Path,
    rows: List[Dict[str, Any]],
    *,
    ledger: Optional[Path] = None,
    extra: Optional[List[str]] = None,
) -> tuple:
    """Run the guard CLI against a synthetic ledger; return (rc, report)."""
    ledger_path = ledger or (tmp_path / "ledger.jsonl")
    _write_ledger(ledger_path, rows)
    argv = [
        "--ledger",
        str(ledger_path),
        "--repo-root",
        str(repo),
        "--no-registry",
    ]
    if extra:
        argv.extend(extra)
    rc = guard.main(argv)
    return rc, ledger_path


# --------------------------------------------------------------------------
# the contract the brief asks for
# --------------------------------------------------------------------------
def test_one_provable_and_one_unprovable_row_exits_one(tmp_path):
    """A synthetic ledger with one provable + one unprovable row => exit 1."""
    repo = _init_repo(tmp_path)
    _write(repo, "m/a.json", '{"v": 1}\n')
    _commit(repo, "add a")
    good = _sha256(repo / "m" / "a.json")

    rc, _ = _run(
        tmp_path,
        repo,
        [
            _row("m_good", "m/a.json", good, value=1),
            _row("m_bad", "m/a.json", "f" * 64, value=1),
        ],
    )
    assert rc == guard.EXIT_DEFECT


def test_all_provable_ledger_exits_zero(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo, "m/a.json", '{"v": 1}\n')
    _commit(repo, "add a")
    good = _sha256(repo / "m" / "a.json")

    rc, _ = _run(tmp_path, repo, [_row("m_good", "m/a.json", good, value=1)])
    assert rc == guard.EXIT_OK


def test_empty_ledger_does_not_crash(tmp_path):
    repo = _init_repo(tmp_path)
    _commit(repo, "empty")
    rc, _ = _run(tmp_path, repo, [])
    assert rc == guard.EXIT_OK


def test_missing_ledger_is_an_error_not_a_crash(tmp_path):
    repo = _init_repo(tmp_path)
    _commit(repo, "empty")
    rc = guard.main(
        [
            "--ledger",
            str(tmp_path / "does_not_exist.jsonl"),
            "--repo-root",
            str(repo),
            "--no-registry",
        ]
    )
    assert rc == guard.EXIT_ERROR


def test_missing_path_does_not_crash(tmp_path):
    """A row naming a path that exists nowhere is a finding, not an exception."""
    repo = _init_repo(tmp_path)
    _commit(repo, "empty")
    rc, _ = _run(
        tmp_path,
        repo,
        [_row("m_ghost", "m/never_existed.json", "a" * 64, value=1)],
    )
    assert rc == guard.EXIT_DEFECT


def test_corrupt_ledger_line_is_an_error(tmp_path):
    repo = _init_repo(tmp_path)
    _commit(repo, "empty")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("{not json\n", encoding="utf-8")
    rc = guard.main(
        [
            "--ledger",
            str(ledger),
            "--repo-root",
            str(repo),
            "--no-registry",
        ]
    )
    assert rc == guard.EXIT_ERROR


# --------------------------------------------------------------------------
# verdicts + repair classification
# --------------------------------------------------------------------------
def test_verdict_committed_is_changed_since_not_a_defect(tmp_path):
    """A committed-but-superseded content version is honest history."""
    repo = _init_repo(tmp_path)
    p = _write(repo, "m/a.json", "v1\n")
    _commit(repo, "v1")
    v1_sha = _sha256(p)
    p.write_text("v2\n", encoding="utf-8")  # changed AFTER the record, uncommitted

    rc, ledger = _run(tmp_path, repo, [_row("m_a", "m/a.json", v1_sha, value=1)])
    assert rc == guard.EXIT_OK
    report = guard.analyse_ledger(ledger, repo)
    row = report["rows"][0]
    assert row["verdict"] == guard.VERDICT_COMMITTED
    assert row["repair_class"] == guard.REPAIR_CHANGED_SINCE


def test_superseded_unprovable_row_is_a_documentation_note(tmp_path):
    repo = _init_repo(tmp_path)
    p = _write(repo, "m/a.json", "v1\n")
    _commit(repo, "v1")
    good = _sha256(p)

    rc, ledger = _run(
        tmp_path,
        repo,
        [
            _row("m_a", "m/a.json", "b" * 64, value=1),  # never existed
            _row("m_a", "m/a.json", good, value=1),  # supersedes it
        ],
    )
    assert rc == guard.EXIT_DEFECT
    report = guard.analyse_ledger(ledger, repo)
    bad = report["rows"][0]
    assert bad["verdict"] == guard.VERDICT_UNPROVABLE
    assert bad["row_state"] == "superseded"
    assert bad["repair_class"] == guard.REPAIR_DOC_NOTE
    assert report["rows"][1]["row_state"] == "latest"
    assert report["rows"][1]["verdict"] == guard.VERDICT_ON_DISK
    # the finding carries the evidence that the live provenance is intact
    finding = report["findings"][0]
    assert finding["latest_row_verdict"] == guard.VERDICT_ON_DISK
    assert finding["suggested_repair"]["record_run_kwargs"] is None


def test_open_unprovable_row_suggests_an_append(tmp_path):
    repo = _init_repo(tmp_path)
    p = _write(repo, "m/a.json", "v1\n")
    _commit(repo, "v1")
    good = _sha256(p)

    rc, ledger = _run(tmp_path, repo, [_row("m_a", "m/a.json", "c" * 64, value=7)])
    assert rc == guard.EXIT_DEFECT
    report = guard.analyse_ledger(ledger, repo)
    row = report["rows"][0]
    assert row["row_state"] == "latest"
    assert row["repair_class"] == guard.REPAIR_APPEND
    kwargs = report["findings"][0]["suggested_repair"]["record_run_kwargs"]
    assert kwargs["metric_id"] == "m_a"
    assert kwargs["value"] == 7
    assert kwargs["source_json"] == "m/a.json"
    # the sha to carry is the CURRENT content, and it is committed here
    assert kwargs["source_json_sha256"] == good


# --------------------------------------------------------------------------
# the merge-commit blind spot (the false positive this guard must not have)
# --------------------------------------------------------------------------
def _make_merge_repo(repo: Path) -> str:
    """Create a merge commit whose conflict resolution introduces a NEW content.

    Returns the sha256 of the merge result's content.  That content exists at
    NO other commit: only the merge commit holds it.
    """
    _write(repo, "m/a.json", "base\n")
    _commit(repo, "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "m/a.json", "feature\n")
    _commit(repo, "feature")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "m/a.json", "main\n")
    _commit(repo, "main")
    merge = subprocess.run(
        ["git", "-C", str(repo), "merge", "--no-commit", "feature"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert merge.returncode != 0, "expected a merge conflict"
    _write(repo, "m/a.json", "resolved\n")
    _git(repo, "add", "m/a.json")
    _commit(repo, "merge with resolution")
    return _sha256(repo / "m" / "a.json")


def test_bulk_resolver_sees_merge_introduced_version(tmp_path):
    """`git log --raw` prints NO diff for a merge commit -- `-m` is load-bearing."""
    repo = _init_repo(tmp_path)
    merge_sha = _make_merge_repo(repo)

    bulk = guard.committed_content_shas_bulk(repo, ["m/a.json"])
    assert merge_sha in bulk["m/a.json"], (
        "the bulk resolver missed a content version introduced by a merge "
        "commit -- `git log --raw` needs `-m`"
    )


def test_merge_introduced_version_is_provable(tmp_path):
    repo = _init_repo(tmp_path)
    merge_sha = _make_merge_repo(repo)
    # change the working tree so the on-disk sha differs from the merge result
    (repo / "m" / "a.json").write_text("later\n", encoding="utf-8")

    rc, ledger = _run(tmp_path, repo, [_row("m_a", "m/a.json", merge_sha, value=1)])
    assert rc == guard.EXIT_OK
    report = guard.analyse_ledger(ledger, repo)
    assert report["rows"][0]["verdict"] == guard.VERDICT_COMMITTED


def test_the_two_resolvers_agree(tmp_path):
    repo = _init_repo(tmp_path)
    _make_merge_repo(repo)
    paths = ["m/a.json"]
    bulk = guard.committed_content_shas_bulk(repo, paths)
    per_path = guard.committed_content_shas_per_path(repo, paths)
    assert bulk == per_path


def test_resolver_disagreement_is_an_error_not_a_verdict(tmp_path, monkeypatch):
    """If the two git-history resolvers disagree the guard must NOT render a verdict.

    A disagreement means one of them has a blind spot, and a verdict built on
    a blind spot is a false positive.  Exit 2 (SKIP/ERROR), not exit 0 or 1.
    """
    repo = _init_repo(tmp_path)
    p = _write(repo, "m/a.json", "v1\n")
    _commit(repo, "v1")
    good = _sha256(p)
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [_row("m_a", "m/a.json", good, value=1)])

    def _wrong(repo_root, paths):  # noqa: ANN001
        return {p: set() for p in paths}

    monkeypatch.setattr(guard, "committed_content_shas_per_path", _wrong)
    rc = guard.main(
        [
            "--ledger",
            str(ledger),
            "--repo-root",
            str(repo),
            "--no-registry",
        ]
    )
    assert rc == guard.EXIT_ERROR


def test_cross_check_disabled_still_renders_a_verdict(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo, "m/a.json", "v1\n")
    _commit(repo, "v1")
    good = _sha256(repo / "m" / "a.json")
    rc, ledger = _run(
        tmp_path,
        repo,
        [_row("m_a", "m/a.json", good, value=1)],
        extra=["--no-cross-check"],
    )
    assert rc == guard.EXIT_OK
    report = guard.analyse_ledger(ledger, repo, cross_check=False)
    assert report["resolvers"]["cross_check"] is False
    assert report["rows"][0]["verdict"] == guard.VERDICT_ON_DISK


# --------------------------------------------------------------------------
# registry (read-only)
# --------------------------------------------------------------------------
def test_registry_check_reports_counts_without_writing(tmp_path):
    repo = _init_repo(tmp_path)
    p = _write(repo, "m/a.json", '{"v": 1}\n')
    _commit(repo, "v1")
    good = _sha256(p)
    registry = tmp_path / "metrics.yaml"
    registry.write_text(
        "metrics:\n"
        "  m_good:\n"
        f'    source_json: "m/a.json"\n'
        f'    source_json_sha256: "{good}"\n'
        "  m_bad:\n"
        '    source_json: "m/a.json"\n'
        f'    source_json_sha256: "{"d" * 64}"\n',
        encoding="utf-8",
    )
    before = registry.read_bytes()
    rep = guard.analyse_registry(registry, repo)
    assert rep["counts"]["on_disk"] == 1
    assert rep["counts"]["unprovable"] == 1
    assert rep["status"] == "FAIL"
    assert registry.read_bytes() == before, "the registry check must not write"


# --------------------------------------------------------------------------
# the real ledger -- the state the brief expects (a guard that cannot fire
# is not a guard)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not (REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl").is_file(),
    reason="real ledger not present",
)
def test_real_ledger_has_at_least_one_unprovable_row():
    """The real ledger currently carries unprovable rows -> the guard must fire."""
    report = guard.analyse_ledger(
        REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl",
        REPO_ROOT,
        cross_check=False,
    )
    assert report["counts"]["ledger_rows"] > 0
    assert report["counts"]["unprovable"] > 0
    assert report["status"] == "FAIL"
    # every unprovable row must name a path and a hash
    for f in report["findings"]:
        assert f["source_json"]
        assert f["source_json_sha256"]
        assert f["repair_class"] in (guard.REPAIR_DOC_NOTE, guard.REPAIR_APPEND)
