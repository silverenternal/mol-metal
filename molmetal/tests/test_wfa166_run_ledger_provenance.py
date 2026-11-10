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

"""Wf-A166 — tests for the run-ledger git-provenance validator.

Covers the four contracts the validator must honour:

1. NEGATIVE CONTROLS — a synthetic ledger row whose ``git_sha`` (i) does not
   resolve to a commit, (ii) is unreachable from every branch and tag, or
   (iii) never touched that row's ``source_json`` MUST be flagged.
2. POSITIVE CONTROL — a synthetic ledger row whose ``git_sha`` DID touch the
   ``source_json`` (and whose sha matches the file at that commit) MUST NOT
   be flagged.
3. VACUITY GUARD — every count the audit reports must equal the number of
   findings it lists, and the denominator must be present.
4. CPU-ONLY GUARD — the validator's import graph has no torch.

Every test runs on CPU with ``CUDA_VISIBLE_DEVICES=""``.  The synthetic
controls use a THROWAWAY git repository under ``tmp_path`` so the controls
never depend on this repository's history.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from molmetal.scripts import wfa166_run_ledger_provenance as tool

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wfa166_run_ledger_provenance.py"
REAL_LEDGER = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"

_CPU_ENV = dict(os.environ)
_CPU_ENV["CUDA_VISIBLE_DEVICES"] = ""
_CPU_ENV["GIT_AUTHOR_NAME"] = "wfa166"
_CPU_ENV["GIT_AUTHOR_EMAIL"] = "wfa166@example.invalid"
_CPU_ENV["GIT_COMMITTER_NAME"] = "wfa166"
_CPU_ENV["GIT_COMMITTER_EMAIL"] = "wfa166@example.invalid"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=_CPU_ENV,
        check=False,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_repo(tmp_path_factory) -> Path:
    """A throwaway git repo with a known commit topology.

    * commit A adds ``metrics/by_metric/synthetic_ok.json``
    * commit B adds ``metrics/by_metric/other.json``
    * commit C adds ``metrics/by_metric/second.json``  (touches NEITHER of
      the two files above — this is the "commit that never touched the
      source_json" control)
    * commit D is an EMPTY commit on a branch that is then deleted — it
      still exists as a git object but is reachable from no branch and no
      tag (the "unreachable commit" control)
    """
    repo = tmp_path_factory.mktemp("wfa166_repo")
    (repo / "metrics" / "by_metric").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "master")
    _git(repo, "config", "user.name", "wfa166")
    _git(repo, "config", "user.email", "wfa166@example.invalid")
    _git(repo, "config", "gc.auto", "0")

    def _commit(rel: str, content: str) -> str:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(content, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"add {rel}")
        return _git(repo, "rev-parse", "HEAD").stdout.strip()

    sha_a = _commit("metrics/by_metric/synthetic_ok.json", '{"v":1}\n')
    sha_b = _commit("metrics/by_metric/other.json", '{"v":2}\n')
    sha_c = _commit("metrics/by_metric/second.json", '{"v":3}\n')

    # unreachable commit: empty commit on a throwaway branch, branch deleted
    _git(repo, "checkout", "-q", "-b", "wfa166_temp")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "dangling")
    sha_d = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "master")
    _git(repo, "branch", "-q", "-D", "wfa166_temp")

    (repo / "_shas.json").write_text(
        json.dumps({"a": sha_a, "b": sha_b, "c": sha_c, "d": sha_d}),
        encoding="utf-8",
    )
    return repo


@pytest.fixture(scope="module")
def synthetic_ledger(synthetic_repo: Path, tmp_path_factory) -> Path:
    """A synthetic ledger with one positive and three negative rows."""
    shas = json.loads((synthetic_repo / "_shas.json").read_text(encoding="utf-8"))
    ok_sha = _sha256_text('{"v":1}\n')
    other_sha = _sha256_text('{"v":2}\n')

    def _row(metric_id, source_json, sha, git_sha):
        return {
            "metric_id": metric_id,
            "value": 1.0,
            "source_json": source_json,
            "source_json_sha256": sha,
            "git_sha": git_sha,
            "data_sha": None,
            "config_sha": None,
            "protocol_id": None,
            "host": "test",
            "timestamp": "2026-11-01T00:00:00+00:00",
            "seed": False,
            "extra": {},
        }

    rows = [
        # POSITIVE: the commit DID add this file, the sha is the file's
        # content at that commit, and the commit is reachable from master.
        _row("synthetic_ok", "metrics/by_metric/synthetic_ok.json", ok_sha, shas["a"]),
        # NEGATIVE (i): a sha that is not a git object at all.
        _row(
            "bad_missing_commit",
            "metrics/by_metric/synthetic_ok.json",
            ok_sha,
            "0123456789abcdef0123456789abcdef01234567",
        ),
        # NEGATIVE (ii): a real commit, reachable from no branch and no tag.
        _row("bad_unreachable", "metrics/by_metric/synthetic_ok.json", ok_sha, shas["d"]),
        # NEGATIVE (iii): a reachable commit whose diff does NOT touch the
        # row's source_json.
        _row("bad_untouched", "metrics/by_metric/other.json", other_sha, shas["c"]),
    ]
    path = tmp_path_factory.mktemp("wfa166_ledger") / "ledger.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# 1. negative controls
# ---------------------------------------------------------------------------
def _axes_for(rep: dict, metric_id: str) -> set:
    return {
        f["axis"]
        for f in rep["findings"]
        if f["metric_id"] == metric_id and f["severity"] == "blocking"
    }


def test_negative_control_missing_commit_is_flagged(
    synthetic_ledger: Path, synthetic_repo: Path
) -> None:
    """(i) a ``git_sha`` that does not resolve to a commit MUST be flagged."""
    rep = tool.audit_ledger(ledger_path=synthetic_ledger, repo_root=synthetic_repo)
    axes = _axes_for(rep, "bad_missing_commit")
    assert "commit_unresolvable" in axes, (
        f"a non-existent git_sha was not flagged; axes={axes}"
    )


def test_negative_control_unreachable_commit_is_flagged(
    synthetic_ledger: Path, synthetic_repo: Path
) -> None:
    """(ii) a commit unreachable from every branch and tag MUST be flagged."""
    rep = tool.audit_ledger(ledger_path=synthetic_ledger, repo_root=synthetic_repo)
    axes = _axes_for(rep, "bad_unreachable")
    assert "commit_unreachable" in axes, (
        f"an unreachable git_sha was not flagged; axes={axes}"
    )


def test_negative_control_untouched_source_is_flagged(
    synthetic_ledger: Path, synthetic_repo: Path
) -> None:
    """(iii) a commit that never touched the row's ``source_json`` MUST be
    flagged."""
    rep = tool.audit_ledger(ledger_path=synthetic_ledger, repo_root=synthetic_repo)
    axes = _axes_for(rep, "bad_untouched")
    assert "commit_never_touched_source" in axes, (
        f"a commit that never touched the source_json was not flagged; axes={axes}"
    )


# ---------------------------------------------------------------------------
# 2. positive control
# ---------------------------------------------------------------------------
def test_positive_control_is_not_flagged(
    synthetic_ledger: Path, synthetic_repo: Path
) -> None:
    """A commit that DID touch the row's ``source_json`` MUST NOT be flagged."""
    rep = tool.audit_ledger(ledger_path=synthetic_ledger, repo_root=synthetic_repo)
    row = next(r for r in rep["rows"] if r["metric_id"] == "synthetic_ok")
    assert row["verdicts"]["commit_exists"] == "commit"
    assert row["verdicts"]["commit_reachable"] == "reachable"
    assert row["verdicts"]["commit_touches_source"] == "touched"
    assert row["verdicts"]["commit_content_matches"] == "match"
    assert row["verdicts"]["sha_vs_disk"] == "match"
    assert row["verdicts"]["sha_vs_committed"] == "match"
    assert not [
        f for f in rep["findings"] if f["metric_id"] == "synthetic_ok"
    ], "the positive control row produced findings"


# ---------------------------------------------------------------------------
# 3. vacuity guard
# ---------------------------------------------------------------------------
def _assert_vacuity(rep: dict) -> None:
    """Every reported count MUST equal the number of findings listed, and the
    denominator MUST be present."""
    c = rep["counts"]
    findings = rep["findings"]
    assert c["findings_total"] == len(findings)
    assert c["findings_latest_blocking"] == sum(
        1 for f in findings if f["scope"] == "latest" and f["severity"] == "blocking"
    )
    assert c["findings_latest_advisory"] == sum(
        1 for f in findings if f["scope"] == "latest" and f["severity"] == "advisory"
    )
    assert c["findings_superseded_blocking"] == sum(
        1 for f in findings if f["scope"] == "superseded" and f["severity"] == "blocking"
    )
    assert c["findings_superseded_advisory"] == sum(
        1 for f in findings if f["scope"] == "superseded" and f["severity"] == "advisory"
    )
    # denominators are present and self-consistent
    assert c["ledger_rows"] == len(rep["rows"])
    assert c["latest_rows"] + c["superseded_rows"] == c["ledger_rows"]
    assert c["distinct_metrics"] == c["latest_rows"]
    assert c["blocking_latest_metrics"] == len(
        {f["metric_id"] for f in findings if f["scope"] == "latest" and f["severity"] == "blocking"}
    )
    for axis, n in c["blocking_latest_metrics_by_axis"].items():
        assert n == len(
            {
                f["metric_id"]
                for f in findings
                if f["scope"] == "latest"
                and f["severity"] == "blocking"
                and f["axis"] == axis
            }
        ), f"axis {axis} count drift"
    # every finding names a metric and a scope
    for f in findings:
        assert f["scope"] in ("latest", "superseded")
        assert f["axis"] in tool.ALL_AXES
        assert f["severity"] in ("blocking", "advisory")


def test_vacuity_guard_on_the_real_ledger() -> None:
    """The vacuity guard MUST hold on the REAL ledger too."""
    _assert_vacuity(tool.audit_ledger())


def test_vacuity_guard_on_a_synthetic_ledger(
    synthetic_ledger: Path, synthetic_repo: Path
) -> None:
    """The vacuity guard MUST hold on a synthetic ledger too."""
    _assert_vacuity(
        tool.audit_ledger(ledger_path=synthetic_ledger, repo_root=synthetic_repo)
    )


def test_vacuity_guard_on_synthetic_ledger(
    synthetic_ledger: Path, synthetic_repo: Path
) -> None:
    """The synthetic ledger's four rows MUST produce exactly three blocking
    metrics (the three negative controls) and none for the positive one."""
    rep = tool.audit_ledger(ledger_path=synthetic_ledger, repo_root=synthetic_repo)
    blocked = {
        f["metric_id"]
        for f in rep["findings"]
        if f["scope"] == "latest" and f["severity"] == "blocking"
    }
    assert blocked == {
        "bad_missing_commit",
        "bad_unreachable",
        "bad_untouched",
    }, f"unexpected blocked set: {sorted(blocked)}"
    assert rep["counts"]["distinct_metrics"] == 4
    assert rep["counts"]["ledger_rows"] == 4


# ---------------------------------------------------------------------------
# 4. CPU-only guard
# ---------------------------------------------------------------------------
def test_validator_import_graph_has_no_torch() -> None:
    """STATIC: the validator's import graph has no torch, so it cannot touch
    a GPU even by accident."""
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    assert "torch" not in roots, f"torch import found: {sorted(roots)}"
    # and no module-level attribute access on a torch symbol
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert not (node.value.id == "torch"), "torch attribute access"


def test_validator_runs_without_torch_imported() -> None:
    """RUNTIME: importing and running the validator in a fresh interpreter
    must not pull torch into ``sys.modules``."""
    code = (
        "import sys;"
        "from molmetal.scripts.wfa166_run_ledger_provenance import audit_ledger;"
        "audit_ledger();"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m);"
        "print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=_CPU_ENV,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# 5. read-only + live cross-check against run_ledger.py
# ---------------------------------------------------------------------------
def test_audit_does_not_modify_the_ledger() -> None:
    """The audit is READ-ONLY: the ledger bytes must not change."""
    before = hashlib.sha256(REAL_LEDGER.read_bytes()).hexdigest()
    mtime_before = REAL_LEDGER.stat().st_mtime_ns
    tool.audit_ledger()
    after = hashlib.sha256(REAL_LEDGER.read_bytes()).hexdigest()
    assert before == after, "the audit modified the ledger"
    assert mtime_before == REAL_LEDGER.stat().st_mtime_ns


def test_agrees_with_run_ledger_on_the_sha_axes() -> None:
    """The validator's sha axes MUST agree with
    ``molmetal.scripts.run_ledger.uncommitted_provenance`` on the real
    ledger — the two implementations are independent."""
    from molmetal.scripts.run_ledger import uncommitted_provenance

    mine = tool.audit_ledger()
    theirs = uncommitted_provenance(include_superseded=True)
    mc = mine["counts"]
    tc = theirs["counts"]
    assert mc["ledger_rows"] == tc["ledger_rows"]
    assert mc["distinct_metrics"] == tc["distinct_metrics"]
    assert mc["distinct_source_paths"] == tc["distinct_source_paths"]
    assert mc["distinct_source_json_sha256"] == tc["distinct_pairs"]
    never_latest = [
        r
        for r in mine["rows"]
        if r["scope"] == "latest" and r["verdicts"]["sha_vs_committed"] == "differ"
    ]
    never_sup = [
        r
        for r in mine["rows"]
        if r["scope"] == "superseded" and r["verdicts"]["sha_vs_committed"] == "differ"
    ]
    # 2026-11-04 (Wf-A269 wave-22): the two implementations disagreed — 14 vs
    # 2 — because they implemented DIFFERENT PREDICATES.  The validator's
    # ``sha_vs_committed == "differ"`` means "no commit reproduces this sha";
    # ``run_ledger.uncommitted_provenance``'s ``uncommitted`` verdict
    # short-circuits on ``on_disk``, so a row whose content is on disk but in
    # no commit was classified "reproducible" and never reported — which
    # contradicts that function's OWN docstring ("no commit can reproduce
    # it", "the file version it points at exists in NO commit").  The
    # producer now carries that ORTHOGONAL axis (``never_committed``) and the
    # two are compared ON IT.
    #
    # SUPERSEDED ASSERTION (2026-11-04, PRESERVED VERBATIM per AGENTS.md §5.5
    # rule #6): ``assert len(never_latest) == tc["open_uncommitted_metrics"]``
    # — it compared the validator's "no commit reproduces this sha" against
    # run_ledger's "neither on disk nor committed", which are different
    # predicates; it was red for 12 of the 14 rows (MEASURED).
    assert len(never_latest) == tc["open_never_committed_metrics"], (
        "the validator's sha_vs_committed axis (%d) disagrees with "
        "run_ledger's never_committed axis (%d)"
        % (len(never_latest), tc["open_never_committed_metrics"])
    )
    assert len(never_sup) == tc["superseded_never_committed_rows"]
    # The SUBSET both implementations call "uncommitted" — the rows whose
    # content is neither on disk nor in any commit — must still agree.
    neither = [
        r for r in never_latest if r["verdicts"]["sha_vs_disk"] != "match"
    ]
    assert len(neither) == tc["open_uncommitted_metrics"], (
        "the rows that are neither on disk nor committed (%d) disagree with "
        "run_ledger's open_uncommitted_metrics (%d)"
        % (len(neither), tc["open_uncommitted_metrics"])
    )
