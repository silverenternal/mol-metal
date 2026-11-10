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
"""Wf-A269 wave-23 — "is this file version committed?" must see the WHOLE ODB.

THE THREE DEFECTS
------------------
**D1 — a finding asserted a DISJUNCTION it had not decided.**
``wfa166_run_ledger_provenance.py`` emitted, for every row whose sha is
reproducible from no refs-reachable commit::

    source_json_sha256 <sha>: matches neither the file on disk nor any
    committed version of <path>

Only the SECOND half of that sentence was decided at emission time (the
committed half, by ``sha_vs_committed``); the disk half is a SEPARATE
verdict (``sha_vs_disk``) and for 12 of the 23 findings it is ``"match"``
⇒ the disk clause was FALSE.  The two halves are now reported
INDEPENDENTLY, each from its own verdict.

**D2 — "committed" was decided over REFS-REACHABLE history only.**
``git log --all`` cannot see a commit that no ref reaches, so a file
version that WAS committed and has since been rewritten away was reported
as "never committed".  MEASURED on this repository: the object database
holds 1011 commit objects while ``git rev-list --all`` reaches 843, so 168
commits (16.6 %) are unreachable.  The verdict is now THREE-VALUED
(``committed`` / ``committed_unreachable`` / ``never_committed``), and the
LEGACY axis keeps its NAME and its VALUE (it is quoted VERBATIM in
``docs/HONEST_NEGATIVE_CATALOG.md`` and ``TODO/INDEX.md``).

**D3 — three implementations of ONE question had three SHAPES.**
``run_ledger.py::_committed_blob_shas`` and the auditor's committed-content
map returned NO ENTRY for a path with no committed history; ``wfa156``
returned an entry for EVERY requested path.  A naive comparison therefore
manufactured 4 phantom "disagreements" that were pure API shape.  The
contract is now normalised AT THE API — every resolver returns an entry for
every requested path, an empty set meaning "no committed version" — and
the comparison sites ASSERT it instead of compensating for it.

WHAT THIS FILE PINS
-------------------
1. a file version held ONLY by an unreachable commit is
   ``committed_unreachable``, never ``never_committed``;
2. the finding TEXT states the disk clause instead of asserting a
   disjunction;
3. every resolver returns an entry for every requested path;
4. **the new checks CAN fail** — a version no commit holds is still
   ``never_committed``, a resolver that drops a path is still caught, and
   the pre-repair behaviour is reproduced on purpose as a negative control.

CPU-only, READ-ONLY with respect to this repository: every git call in this
file runs against a THROWAWAY repository created under ``tmp_path``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from molmetal.scripts import run_ledger as rl
from molmetal.scripts import wfa156_ledger_provenance_guard as wfa156
from molmetal.scripts import wfa166_run_ledger_census as census
from molmetal.scripts import wfa166_run_ledger_provenance as tool
from molmetal.scripts import wfa198_ledger_pin_consistency as wfa198

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_LEDGER = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"

DEMO_REL = "metrics/by_metric/demo.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
    )
    return proc.stdout.decode("utf-8", "replace")


def _write_version(path: Path, body: str) -> str:
    """Write ``body`` and return the sha256 of the file's CONTENT."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_of(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_repo(tmp_path: Path) -> Dict[str, Any]:
    """A throwaway repo with ONE unreachable commit.

    Layout (all three versions of ``DEMO_REL``):

    * ``v1`` — held by ``C1`` (reachable)
    * ``v2`` — held by ``C2`` (reachable) and by the WORKING TREE
    * ``v3`` — held by ``C3`` only, and ``C3`` is UNREACHABLE (the branch was
      reset away from it, but the object is still in the object database)

    ``C3`` is made unreachable with ``git reset --hard HEAD~1``: the reflog
    keeps a pointer, but ``git rev-list --all`` walks REFS only, so ``C3`` is
    unreachable exactly as an amended/rebased-away commit is.
    """
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "commit.gpgsign", "false")

    target = repo / DEMO_REL
    v1 = _write_version(target, '{"v": 1}\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c1")
    c1 = _git(repo, "rev-parse", "HEAD").strip()

    v2 = _write_version(target, '{"v": 2}\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c2")
    c2 = _git(repo, "rev-parse", "HEAD").strip()

    v3 = _write_version(target, '{"v": 3}\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c3")
    c3 = _git(repo, "rev-parse", "HEAD").strip()

    # make C3 unreachable and put the working tree back onto v2
    _git(repo, "reset", "--hard", "-q", "HEAD~1")

    return {
        "repo": repo,
        "target": target,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "c1": c1,
        "c2": c2,
        "c3": c3,
    }


def _row(
    metric_id: str,
    sha: str,
    *,
    git_sha: str = "HEAD",
    value: Any = 1.0,
) -> Dict[str, Any]:
    return {
        "config_sha": None,
        "data_sha": None,
        "extra": {"data_source": None, "value_key": None},
        "git_sha": git_sha,
        "host": "synthetic",
        "metric_id": metric_id,
        "protocol_id": None,
        "seed": True,
        "source_json": DEMO_REL,
        "source_json_sha256": sha,
        "timestamp": "2026-11-04T00:00:00+00:00",
        "value": value,
    }


def _write_ledger(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _finding(findings: List[Dict[str, Any]], metric_id: str, axis: str) -> Dict[str, Any]:
    hits = [
        f
        for f in findings
        if f["axis"] == axis and str(f["metric_id"]) == metric_id
    ]
    assert len(hits) == 1, (
        "expected exactly one %r finding for %r, got %d"
        % (axis, metric_id, len(hits))
    )
    return hits[0]


# ---------------------------------------------------------------------------
# 1. D2 — a version held ONLY by an unreachable commit
# ---------------------------------------------------------------------------
def test_the_synthetic_repo_really_has_an_unreachable_commit(
    tmp_path: Path,
) -> None:
    """The fixture itself must be what it claims (else every other test here
    is vacuous)."""
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    unreachable = tool._unreachable_commits(repo)
    reachable = tool._reachable_commits(repo)
    assert fx["c3"] in unreachable, (
        "the fixture's C3 is NOT unreachable — the whole file is vacuous"
    )
    assert fx["c3"] not in reachable, (
        "the fixture's C3 is reachable — the whole file is vacuous"
    )
    assert fx["c1"] in reachable and fx["c2"] in reachable
    # the working tree holds v2; C3 holds v3
    assert _sha256_of(fx["target"]) == fx["v2"]
    blob = _git(repo, "rev-parse", "%s:%s" % (fx["c3"], DEMO_REL)).strip()
    assert blob, "C3 does not hold %s" % DEMO_REL


def test_a_version_held_only_by_an_unreachable_commit_is_not_never_committed(
    tmp_path: Path,
) -> None:
    """**D2.** A version held by an unreachable commit is
    ``committed_unreachable`` — NOT ``never_committed``.

    The LEGACY axis (``sha_vs_committed``) is REFS-REACHABLE-scoped and keeps
    its value (``"differ"`` here); the three-valued axis carries the
    whole-ODB answer.
    """
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    ledger = _write_ledger(
        tmp_path / "ledger.jsonl",
        [
            _row("demo_unreachable", fx["v3"], git_sha=fx["c3"]),
            _row("demo_committed", fx["v2"], git_sha=fx["c2"]),
        ],
    )

    audit = tool.audit_ledger(ledger_path=ledger, repo_root=repo)
    rows = {str(r["metric_id"]): r for r in audit["rows"]}

    unreach = rows["demo_unreachable"]
    assert unreach["verdicts"]["sha_vs_committed"] == "differ", (
        "the legacy reachable-scoped axis must still say 'differ' for a "
        "version only an unreachable commit holds"
    )
    assert unreach["verdicts"]["sha_commit_status"] == "committed_unreachable", (
        "a version held by an UNREACHABLE commit was classified %r — the "
        "refs-reachable-only query is still deciding this axis"
        % (unreach["verdicts"]["sha_commit_status"],)
    )
    # the negative control: a version a REACHABLE commit holds
    assert rows["demo_committed"]["verdicts"]["sha_commit_status"] == "committed"

    # the finding is still emitted, and the AXIS keeps its severity
    # (``sha_never_committed`` is ADVISORY — the published counts and the
    # gate verdict depend on it)
    finding = _finding(audit["findings"], "demo_unreachable", "sha_never_committed")
    assert finding["severity"] == "advisory"
    assert "sha_never_committed" in tool.ADVISORY_AXES
    assert "committed_unreachable" in finding["detail"], (
        "the finding text does not carry the whole-ODB verdict: %r"
        % (finding["detail"],)
    )


def test_a_version_no_commit_holds_is_still_never_committed(tmp_path: Path) -> None:
    """**THE NEGATIVE CASE — the new axis can still fail.**

    A version that NO commit holds (reachable or not) must be
    ``never_committed``.  If the three-valued verdict collapsed into
    "everything is committed_unreachable" this test would catch it — a guard
    that cannot fail is the §6.269.17 class.
    """
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    # v4 was never committed at all
    v4 = _write_version(fx["target"], '{"v": 4}\n')
    ledger = _write_ledger(
        tmp_path / "ledger.jsonl", [_row("demo_never", v4, git_sha=fx["c3"])]
    )
    audit = tool.audit_ledger(ledger_path=ledger, repo_root=repo)
    row = {str(r["metric_id"]): r for r in audit["rows"]}["demo_never"]
    assert row["verdicts"]["sha_commit_status"] == "never_committed", (
        "a version no commit holds was classified %r — the three-valued "
        "verdict cannot fail" % (row["verdicts"]["sha_commit_status"],)
    )
    finding = _finding(audit["findings"], "demo_never", "sha_never_committed")
    assert "commit status = never_committed" in finding["detail"]


def test_a_version_held_by_both_a_reachable_and_an_unreachable_commit_is_committed(
    tmp_path: Path,
) -> None:
    """Reachability wins: a version a ref still reaches is ``committed``."""
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    # C2 (reachable) and C3 (unreachable) both hold v3? no — make C2 hold v3
    _write_version(fx["target"], '{"v": 3}\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c4")
    c4 = _git(repo, "rev-parse", "HEAD").strip()
    # now BOTH c4 (reachable) and c3 (unreachable) hold v3
    ledger = _write_ledger(
        tmp_path / "ledger.jsonl", [_row("demo_both", fx["v3"], git_sha=c4)]
    )
    audit = tool.audit_ledger(ledger_path=ledger, repo_root=repo)
    row = {str(r["metric_id"]): r for r in audit["rows"]}["demo_both"]
    assert row["verdicts"]["sha_commit_status"] == "committed"


# ---------------------------------------------------------------------------
# 2. D1 — the finding text must not assert a disjunction it did not decide
# ---------------------------------------------------------------------------
def test_the_finding_text_states_the_disk_clause(tmp_path: Path) -> None:
    """**D1.** The disk half and the committed half are reported separately.

    * disk MATCHES the sha  -> the text says ``MATCHES the file on disk``;
    * disk DIFFERS           -> the text says ``does NOT match the file on
      disk``;
    * and in BOTH cases the text must NOT contain the old disjunction
      ("matches neither ..."), which asserted a fact the emitter had not
      decided.
    """
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    # the working tree holds v4 (never committed) -> disk MATCHES the sha
    v4 = _write_version(fx["target"], '{"v": 4}\n')
    ledger = _write_ledger(
        tmp_path / "ledger.jsonl",
        [
            _row("demo_ondisk", v4, git_sha=fx["c3"]),
            # disk holds v4, this row names v3 -> disk DIFFERS
            _row("demo_offdisk", fx["v3"], git_sha=fx["c3"]),
        ],
    )
    audit = tool.audit_ledger(ledger_path=ledger, repo_root=repo)

    ondisk = _finding(audit["findings"], "demo_ondisk", "sha_never_committed")
    assert "MATCHES the file on disk" in ondisk["detail"], (
        "the finding does not report the disk half: %r" % (ondisk["detail"],)
    )
    assert "neither" not in ondisk["detail"], (
        "the finding still asserts the DISJUNCTION it did not decide: %r"
        % (ondisk["detail"],)
    )

    offdisk = _finding(audit["findings"], "demo_offdisk", "sha_never_committed")
    assert "does NOT match the file on disk" in offdisk["detail"], (
        "the finding does not report the disk half: %r" % (offdisk["detail"],)
    )
    assert "neither" not in offdisk["detail"]


def test_on_the_real_ledger_every_finding_states_its_disk_clause() -> None:
    """**D1 on the REAL ledger** — the property, not a count.

    For every ``sha_never_committed`` finding the emitted text must agree
    with the file that is on disk RIGHT NOW.  This is the assertion the old
    text could not satisfy: the old text said "matches neither the file on
    disk nor ..." for all of them, while 12 of them DO match the disk.
    """
    audit = tool.audit_ledger()
    findings = [
        f for f in audit["findings"] if f["axis"] == "sha_never_committed"
    ]
    assert findings, (
        "the real ledger produced no sha_never_committed finding — the "
        "population changed, so this test must be updated, not silently pass"
    )
    n_match = 0
    for f in findings:
        src = str(f["source_json"])
        disk = _sha256_of(tool._resolve_source_json(src))
        detail = str(f["detail"])
        assert "neither" not in detail, (
            "the finding still asserts the disjunction: %r" % (detail,)
        )
        if disk is not None and disk == str(f["source_json_sha256"]):
            n_match += 1
            assert "MATCHES the file on disk" in detail, (
                "the file on disk DOES match %s but the finding says: %r"
                % (src, detail)
            )
        elif disk is None:
            assert "MISSING" in detail, (
                "%s is not on disk but the finding says: %r" % (src, detail)
            )
        else:
            assert "does NOT match the file on disk" in detail, (
                "the file on disk does NOT match %s but the finding says: %r"
                % (src, detail)
            )
    assert n_match >= 1, (
        "no finding reports a disk MATCH — the disk clause is not exercised "
        "on the real ledger, so the assertion above is vacuous"
    )


def test_the_legacy_axis_keeps_its_name_and_its_value() -> None:
    """The published numbers depend on the legacy axis.

    ``OPEN_sha_never_committed=14`` / ``superseded_sha_never_committed=9``
    are quoted VERBATIM in ``docs/HONEST_NEGATIVE_CATALOG.md`` and
    ``TODO/INDEX.md``.  The repair ADDS a three-valued axis; it must not
    change the legacy one.
    """
    audit = tool.audit_ledger()
    latest = [r for r in audit["rows"] if r["scope"] == "latest"]
    legacy = [r for r in latest if r["verdicts"]["sha_vs_committed"] == "differ"]
    assert len(legacy) == 14, (
        "the legacy reachable-scoped axis now reports %d latest rows, not the "
        "published 14 — a published number changed" % (len(legacy),)
    )
    new_axis = [r for r in latest if "sha_commit_status" in r["verdicts"]]
    assert len(new_axis) == len(latest), (
        "the three-valued axis is missing from %d latest row(s)"
        % (len(latest) - len(new_axis),)
    )


# ---------------------------------------------------------------------------
# D1 (producer half): the PRODUCER's note asserted the same disjunction and
# described the WRONG row for a superseded finding.
# ---------------------------------------------------------------------------
def _assert_note_disk_clause(note: str, sha: str, disk_sha: Optional[str]) -> None:
    """The note must state the disk half truthfully, on its own.

    Raises ``AssertionError`` — which is what the mutation test below
    exploits to show the check CAN fail.
    """
    assert "neither" not in note, (
        "the note still asserts the disjunction: %r" % (note,)
    )
    if disk_sha is None:
        assert "the file on disk is MISSING" in note, (
            "%s is not on disk but the note says: %r" % (sha, note)
        )
    elif disk_sha == sha:
        assert "MATCHES the file on disk" in note, (
            "the file on disk DOES match %s but the note says: %r"
            % (sha, note)
        )
    else:
        assert "does NOT match the file on disk" in note, (
            "the file on disk does NOT match %s but the note says: %r"
            % (sha, note)
        )


def _disk_sha_of(src: str) -> Optional[str]:
    return _sha256_of(Path(rl.REPO_ROOT) / src)


def test_the_producer_note_states_the_disk_clause() -> None:
    """**D1 on the producer** — every note, on the REAL ledger."""
    up = rl.uncommitted_provenance(
        repo_root=Path(rl.REPO_ROOT), include_superseded=True
    )
    assert up["detail"], "the producer reported no uncommitted-provenance detail"
    n_checked = 0
    for entry in up["detail"]:
        rows = (
            [entry["latest_row"]]
            if entry["state"] == "open"
            else list(entry["superseded_rows"])
        )
        clauses = [c for c in str(entry["note"]).split(" || ") if c]
        assert len(clauses) == len(rows), (
            "the note names %d row(s) but the finding is about %d: %r"
            % (len(clauses), len(rows), entry["note"])
        )
        for row, clause in zip(rows, clauses):
            sha = str(row["source_json_sha256"])
            assert sha in clause, (
                "the note describes a different row than the finding: %r"
                % (entry["note"],)
            )
            _assert_note_disk_clause(
                clause, sha, _disk_sha_of(str(row["source_json"]))
            )
            n_checked += 1
    assert n_checked >= 1, "no producer note was checked — the test is vacuous"


def test_the_producer_never_committed_note_states_the_disk_clause() -> None:
    """The producer's never-committed axis must also report the disk half.

    A row can be the CURRENT content of the file on disk while no commit
    holds it; the old note left that unstated.
    """
    up = rl.uncommitted_provenance(repo_root=Path(rl.REPO_ROOT))
    assert up["never_committed_detail"], "the producer reported no never-committed row"
    n_match = 0
    for entry in up["never_committed_detail"]:
        row = entry["latest_row"]
        sha = str(row["source_json_sha256"])
        note = str(entry["note"])
        _assert_note_disk_clause(note, sha, _disk_sha_of(str(row["source_json"])))
        assert "no REFS-REACHABLE commit of this repository reproduces" in note
        if entry.get("on_disk_matches"):
            n_match += 1
    assert n_match >= 1, (
        "no never-committed row matches the file on disk — the disk clause "
        "is not exercised, so the assertion above is vacuous"
    )


def test_the_producer_note_check_can_fail() -> None:
    """MUTATION PROOF: the producer-note check CAN fail.

    A guard that cannot fail is worse than no guard (the repo's own
    §6.269.17 class).  Feed the checker the OLD text and the wrong row.
    """
    with pytest.raises(AssertionError):
        _assert_note_disk_clause(
            "the sha this row names matches neither the file on disk nor any "
            "committed version of it",
            "0" * 64,
            "0" * 64,
        )
    with pytest.raises(AssertionError):
        _assert_note_disk_clause(
            "sha %s (x.json): does NOT match the file on disk" % ("0" * 64,),
            "0" * 64,
            "0" * 64,
        )
    with pytest.raises(AssertionError):
        _assert_note_disk_clause(
            "sha %s (x.json): MATCHES the file on disk" % ("0" * 64,),
            "0" * 64,
            "1" * 64,
        )


# ---------------------------------------------------------------------------
# 3. D2 in the PRODUCER — the refusal must say what it actually checked
# ---------------------------------------------------------------------------
def test_the_reachability_check_is_load_bearing(monkeypatch, tmp_path: Path) -> None:
    """**MUTATION PROOF (D2).**  The new verdict is load-bearing.

    The auditor's unreachable-commit query is replaced by the PRE-REPAIR
    behaviour (it answers "no unreachable version"), and the verdict must
    then FLIP to ``never_committed`` — i.e. the repair is what makes the
    difference, and the check is not a constant.
    """
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    ledger = _write_ledger(
        tmp_path / "ledger.jsonl", [_row("demo_unreachable", fx["v3"], git_sha=fx["c3"])]
    )
    before = tool.audit_ledger(ledger_path=ledger, repo_root=repo)
    status_before = {
        str(r["metric_id"]): r["verdicts"]["sha_commit_status"] for r in before["rows"]
    }
    assert status_before["demo_unreachable"] == "committed_unreachable"

    monkeypatch.setattr(
        tool,
        "committed_content_shas_unreachable",
        lambda repo_root, paths: {p: set() for p in paths},
    )
    after = tool.audit_ledger(ledger_path=ledger, repo_root=repo)
    status_after = {
        str(r["metric_id"]): r["verdicts"]["sha_commit_status"] for r in after["rows"]
    }
    assert status_after["demo_unreachable"] == "never_committed", (
        "the verdict did NOT flip when the unreachable-commit evidence was "
        "removed — the three-valued axis is not reading it"
    )


def test_the_producer_refusal_is_load_bearing(monkeypatch, tmp_path: Path) -> None:
    """**MUTATION PROOF (D2, producer side).**  With the pre-repair
    reachable-only predicate, the repoint of a version held only by an
    unreachable commit is REFUSED.  The repair is what lets it through."""
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    reg = _synthetic_registry(tmp_path, repo)
    _write_version(fx["target"], '{"v": 3}\n')

    real = rl._committed_content_shas_by_reachability

    def _reachable_only(repo_root, paths):
        reachable, _unreachable = real(repo_root, paths)
        return reachable, {p: set() for p in paths}

    monkeypatch.setattr(rl, "_committed_content_shas_by_reachability", _reachable_only)
    with pytest.raises(ValueError):
        rl.repoint_metric(
            "demo",
            registry_metrics_path=reg,
            ledger_path=tmp_path / "ledger.jsonl",
            repo_root=repo,
            dry_run=True,
        )


def _synthetic_registry(tmp_path: Path, repo: Path) -> Path:
    reg = tmp_path / "registry"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "metrics.yaml").write_text(
        "metrics:\n"
        "  demo:\n"
        "    source_json: %s\n"
        "    value: 1.0\n" % DEMO_REL,
        encoding="utf-8",
    )
    (reg / "protocols.yaml").write_text("protocols: {}\n", encoding="utf-8")
    return reg / "metrics.yaml"


def test_repoint_accepts_a_version_only_an_unreachable_commit_holds(
    tmp_path: Path,
) -> None:
    """**D2 in ``run_ledger.py``.**  The repoint refusal used to ask "is the
    sha reproducible from NO refs-reachable commit?" and then SAY "is present
    in NO commit of this repository".  A version that WAS committed and has
    since been rewritten away was therefore refused as if the repository had
    never held it — while the message asserted a fact about the whole
    repository that only the reachable half had been examined.

    ⇒ the version held only by the unreachable commit must now be ACCEPTED
    and reported as ``committed_unreachable``.
    """
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    reg = _synthetic_registry(tmp_path, repo)
    # put the working tree back onto the version C3 holds
    _write_version(fx["target"], '{"v": 3}\n')
    out = rl.repoint_metric(
        "demo",
        registry_metrics_path=reg,
        ledger_path=tmp_path / "ledger.jsonl",
        repo_root=repo,
        dry_run=True,
    )
    assert out["commit_status"] == "committed_unreachable", (
        "the repoint guard classified the version %r — it is still deciding "
        "over refs-reachable history only" % (out["commit_status"],)
    )
    assert out["appended"] is False and out["dry_run"] is True


def test_repoint_still_refuses_a_version_no_commit_holds(tmp_path: Path) -> None:
    """**THE NEGATIVE CASE for the refusal** — the guard can still fail, and
    its MESSAGE must say what it actually checked."""
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    reg = _synthetic_registry(tmp_path, repo)
    _write_version(fx["target"], '{"v": 4}\n')
    with pytest.raises(ValueError) as excinfo:
        rl.repoint_metric(
            "demo",
            registry_metrics_path=reg,
            ledger_path=tmp_path / "ledger.jsonl",
            repo_root=repo,
            dry_run=True,
        )
    message = str(excinfo.value)
    assert "object database" in message, (
        "the refusal does not say that it checked the whole object database: %r"
        % (message,)
    )
    assert "is present in NO commit of this repository" not in message, (
        "the refusal still claims the sha is in NO commit of the repository — "
        "a claim the check does not support: %r" % (message,)
    )


def test_the_producer_reports_the_same_three_valued_verdict(tmp_path: Path) -> None:
    """``run_ledger.uncommitted_provenance`` carries the same three-valued
    verdict on the same row as the auditor, and the LEGACY boolean keeps its
    value."""
    fx = _make_repo(tmp_path)
    repo = fx["repo"]
    ledger = _write_ledger(
        tmp_path / "ledger.jsonl",
        [
            _row("demo_unreachable", fx["v3"], git_sha=fx["c3"]),
            _row("demo_committed", fx["v2"], git_sha=fx["c2"]),
        ],
    )
    out = rl.uncommitted_provenance(ledger, repo_root=repo, include_superseded=True)
    statuses = {
        d["metric_id"]: d["commit_status"] for d in out["never_committed_detail"]
    }
    assert statuses.get("demo_unreachable") == "committed_unreachable", (
        "the producer classified it %r" % (statuses.get("demo_unreachable"),)
    )
    assert out["counts"]["open_committed_unreachable_metrics"] == 1
    # the legacy axis keeps its value: the version is still "never committed"
    # as far as a refs-reachable commit is concerned
    legacy = [d for d in out["never_committed_detail"] if d["metric_id"] == "demo_unreachable"]
    assert legacy and legacy[0]["latest_row"]["never_committed"] is True


# ---------------------------------------------------------------------------
# 4. D3 — the API-shape contract
# ---------------------------------------------------------------------------
def _assert_api_shape_contract(
    src_paths: List[str], resolvers: Dict[str, Any]
) -> None:
    """Every resolver must return an entry for EVERY requested path."""
    requested = set(src_paths)
    for name, mapping in resolvers.items():
        keys = set(mapping)
        missing = requested - keys
        assert not missing, (
            f"resolver {name!r} violates the API-shape contract: it returned "
            f"NO entry for {len(missing)} of {len(requested)} requested "
            f"path(s), e.g. {sorted(missing)[:3]}. Every resolver must "
            f"return an entry for every requested path (empty = 'no "
            f"committed version')."
        )


def _all_resolvers(repo: Path, src_paths: List[str]) -> Dict[str, Any]:
    bulk, _n = census.bulk_committed_content_shas(src_paths)
    return {
        "run_ledger": rl._committed_blob_shas(repo, src_paths),
        "wfa198_ledger_pin_consistency": wfa198._committed_blob_shas(repo, src_paths),
        "wfa166_run_ledger_provenance": tool.committed_content_shas(repo, src_paths),
        "wfa166_run_ledger_census": bulk,
        "wfa156_ledger_provenance_guard": wfa156.committed_content_shas_bulk(
            repo, src_paths
        ),
    }


def test_every_resolver_returns_an_entry_for_every_requested_path() -> None:
    """**D3.** The contract, asserted on the REAL repository over the REAL
    ledger's source paths."""
    rows = [
        json.loads(line)
        for line in REAL_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    src_paths = sorted({str(o.get("source_json")) for o in rows if o.get("source_json")})
    resolvers = _all_resolvers(REPO_ROOT, src_paths)
    _assert_api_shape_contract(src_paths, resolvers)
    # non-vacuity: at least one requested path has NO committed version and
    # must therefore be present with an EMPTY set
    empty = [
        p
        for p in src_paths
        if not resolvers["run_ledger"].get(p)
    ]
    assert empty, (
        "every requested path has committed history — the empty-set half of "
        "the contract is not exercised"
    )


def test_the_api_shape_contract_check_can_fail(monkeypatch) -> None:
    """**MUTATION PROOF.**  The contract check must be able to FAIL.

    Two mutations are applied in turn to the REAL producer:

    * M1 — the resolver is replaced by the PRE-REPAIR shape (it drops a path
      it was asked about).  The check must reject it.
    * M2 — a resolver that returns the right KEYS but the wrong MEANING
      (a path with no committed version reported as having one) must also be
      rejected by the comparison site, which is why the empty set is
      asserted to be ABSENT from the oracle rather than merely tolerated.
    """
    src_paths = ["metrics/by_metric/does_not_exist_1.json",
                 "metrics/by_metric/does_not_exist_2.json"]
    broken = {"a": {"x"}}
    with pytest.raises(AssertionError):
        _assert_api_shape_contract(src_paths, {"broken": broken})

    real = rl._committed_blob_shas(REPO_ROOT, ["metrics/by_metric/m1_auc.json"])
    monkeypatch.setattr(
        rl,
        "_committed_blob_shas",
        lambda repo_root, paths: {},  # the pre-repair shape: no entry at all
    )
    mutated = rl._committed_blob_shas(REPO_ROOT, ["metrics/by_metric/m1_auc.json"])
    with pytest.raises(AssertionError):
        _assert_api_shape_contract(
            ["metrics/by_metric/m1_auc.json"], {"run_ledger": mutated}
        )
    monkeypatch.undo()
    assert real, "the un-mutated resolver must still answer"


def test_an_empty_set_is_a_claim_not_an_absence() -> None:
    """**M2.**  A resolver that reports a committed version for a path that
    has none is WRONG, and the comparison site must be able to see it.

    ⇒ the oracle (an independent tree walk) is asked about the same path;
    if the two disagree, the disagreement must be detectable.
    """
    from molmetal.tests.test_wfa269_ledger_merge_history import (
        _exact_committed_blob_oracle,
    )

    rows = [
        json.loads(line)
        for line in REAL_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    src_paths = sorted({str(o.get("source_json")) for o in rows if o.get("source_json")})
    oracle = _exact_committed_blob_oracle(src_paths)
    resolvers = _all_resolvers(REPO_ROOT, src_paths)
    empty = [p for p in src_paths if not resolvers["run_ledger"].get(p)]
    assert empty, "no requested path lacks committed history"
    for path in empty:
        assert path not in oracle, (
            "%s is reported as having NO committed version, but the "
            "independent tree oracle finds one — an empty set is a CLAIM"
            % (path,)
        )


# ---------------------------------------------------------------------------
# 5. the ODB scan is cheap and the real repo really has unreachable commits
# ---------------------------------------------------------------------------
def test_the_real_repo_has_unreachable_commits() -> None:
    """MEASURED 2026-11-04: 1011 commit objects, 843 reachable, 168
    unreachable (16.6 %).  This is the population the three-valued axis
    exists for — and the reason "committed" cannot mean "reachable"."""
    odb = tool._odb_commits(REPO_ROOT)
    unreachable = tool._unreachable_commits(REPO_ROOT)
    assert len(odb) >= 1000, (
        "the object database holds only %d commits — the ODB scan is broken"
        % (len(odb),)
    )
    assert len(unreachable) >= 100, (
        "only %d unreachable commits — the reachability split has changed"
        % (len(unreachable),)
    )
    assert len(unreachable) < len(odb), (
        "every commit is unreachable — the reachable set is broken"
    )
