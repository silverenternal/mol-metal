"""Wf-A123 FIX 7 -- the ledger's NEVER-COMMITTED shas.

The ledger is the artifact that carries ``source_json_sha256``: one
append-only line per recorded value, naming the source file and the sha256
of that file.  ``registry.py check`` validates the REGISTRY rows and
``run_ledger.py::staleness`` compares the LEDGER's sha with the REGISTRY's
sha -- NEITHER compares a sha with GIT HISTORY.  So a ledger row could name
a file version that exists in NO commit, and every existing check still
reported "current".

MEASURED 2026-10-31 (before this fix): 9 of 408 distinct
``(source_json, sha256)`` pairs matched neither the file on disk nor any
committed version of it.  The check lives in
``molmetal/scripts/run_ledger.py::uncommitted_provenance`` (the ledger is the
artifact that carries the sha; ``registry.py`` has no per-row git notion).

These tests build a THROWAWAY git repo in ``tmp_path`` so they never touch
the real ledger, and then check the real ledger once for the shipped state.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import run_ledger as rl  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return proc.stdout


@pytest.fixture()
def sandbox_repo(tmp_path: Path) -> Path:
    """A tiny git repo with one committed artifact and one uncommitted edit."""
    repo = tmp_path / "repo"
    (repo / "metrics").mkdir(parents=True)
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    art = repo / "metrics" / "m.json"
    art.write_text(json.dumps({"value": 1.0}), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _write_ledger(repo: Path, rows: list) -> Path:
    led = repo / "ledger.jsonl"
    led.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    return led


def _sha(path: Path) -> str:
    return rl._sha256_file(path) or ""


def test_a_committed_sha_is_not_flagged(sandbox_repo: Path) -> None:
    """The committed version of a file IS reproducible -> PASS."""
    art = sandbox_repo / "metrics" / "m.json"
    led = _write_ledger(
        sandbox_repo,
        [
            {
                "metric_id": "m",
                "value": 1.0,
                "source_json": "metrics/m.json",
                "source_json_sha256": _sha(art),
                "git_sha": "deadbeef",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "seed": True,
            }
        ],
    )
    rep = rl.uncommitted_provenance(led, repo_root=sandbox_repo)
    assert rep["status"] == "PASS", rep
    assert rep["counts"]["open_uncommitted_metrics"] == 0, rep


def test_the_disk_sha_is_not_flagged(sandbox_repo: Path) -> None:
    """The file on disk (uncommitted edit) IS reproducible -> PASS."""
    art = sandbox_repo / "metrics" / "m.json"
    art.write_text(json.dumps({"value": 2.0}), encoding="utf-8")  # uncommitted
    led = _write_ledger(
        sandbox_repo,
        [
            {
                "metric_id": "m",
                "value": 2.0,
                "source_json": "metrics/m.json",
                "source_json_sha256": _sha(art),
                "git_sha": "deadbeef",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "seed": True,
            }
        ],
    )
    rep = rl.uncommitted_provenance(led, repo_root=sandbox_repo)
    assert rep["status"] == "PASS", rep


def test_a_never_committed_sha_IS_flagged(sandbox_repo: Path) -> None:
    """A sha that matches NEITHER disk NOR any commit -> WARN + a detail."""
    led = _write_ledger(
        sandbox_repo,
        [
            {
                "metric_id": "m",
                "value": 99.0,
                "source_json": "metrics/m.json",
                "source_json_sha256": "de" * 32,
                "git_sha": "deadbeef",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "seed": True,
            }
        ],
    )
    rep = rl.uncommitted_provenance(led, repo_root=sandbox_repo)
    assert rep["status"] == "WARN", rep
    assert rep["counts"]["open_uncommitted_metrics"] == 1, rep
    detail = rep["detail"][0]
    assert detail["metric_id"] == "m"
    assert detail["state"] == "open"
    assert detail["latest_row"]["source_json_sha256"] == "de" * 32


def test_a_later_row_supersedes_an_uncommitted_one(sandbox_repo: Path) -> None:
    """The APPEND-ONLY remedy: a later, reproducible row closes the finding."""
    art = sandbox_repo / "metrics" / "m.json"
    led = _write_ledger(
        sandbox_repo,
        [
            {
                "metric_id": "m",
                "value": 99.0,
                "source_json": "metrics/m.json",
                "source_json_sha256": "de" * 32,
                "git_sha": "deadbeef",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "seed": True,
            },
            {
                "metric_id": "m",
                "value": 1.0,
                "source_json": "metrics/m.json",
                "source_json_sha256": _sha(art),
                "git_sha": "deadbeef",
                "timestamp": "2026-01-02T00:00:00+00:00",
                "seed": False,
            },
        ],
    )
    rep = rl.uncommitted_provenance(led, repo_root=sandbox_repo)
    assert rep["status"] == "PASS", rep
    assert rep["counts"]["open_uncommitted_metrics"] == 0, rep
    # ... but the history is not hidden.
    rep_all = rl.uncommitted_provenance(
        led, repo_root=sandbox_repo, include_superseded=True
    )
    assert rep_all["counts"]["superseded_uncommitted_rows"] == 1, rep_all


def test_the_real_ledger_has_no_open_uncommitted_provenance() -> None:
    """THE SHIPPED STATE: no metric's CURRENT row has an uncommitted sha.

    MEASURED 2026-10-31: before the Wf-A123 FIX 7 corrections there were 8
    OPEN findings (plus 1 superseded); the corrections were APPENDED, never
    rewritten, so the history still holds the 9 bad rows.
    """
    rep = rl.uncommitted_provenance()
    assert rep["counts"]["open_uncommitted_metrics"] == 0, rep["detail"]
    assert rep["status"] == "PASS", rep
    # The history is preserved: the bad rows are still visible with --all.
    rep_all = rl.uncommitted_provenance(include_superseded=True)
    assert rep_all["counts"]["superseded_uncommitted_rows"] >= 8, rep_all["counts"]


def test_the_ledger_is_append_only_and_the_corrections_are_present() -> None:
    """The 9 correction entries are in the ledger, and the old rows survive."""
    entries = rl._iter_entries()
    corrections = [
        e
        for e in entries
        if isinstance(e.get("extra"), dict) and e["extra"].get("wf") == "Wf-A123 FIX 7"
    ]
    assert len(corrections) == 9, [e.get("metric_id") for e in corrections]
    ids = {e["metric_id"] for e in corrections}
    assert ids == {
        "dmpnn_ckpt_status",
        "l5_hilbert_r2",
        "l6_closure",
        "metalcytotox_combined_r2",
        "ood_per_element_reachability",
        "ood_per_metal_reachability",
        "platinai_combined_r2",
        "tmqm_combined_r2",
        "vina_status",
    }, ids
    for e in corrections:
        assert e["source_json_sha256"] in e["extra"]["committed_sha256"] or True
        assert e["extra"]["committed_sha256"] == e["source_json_sha256"], e
        assert "matches NEITHER the file on disk NOR any committed version" in (
            e["extra"]["reason"]
        ), e["metric_id"]
    # The l5_hilbert_r2 correction carries the COMMITTED value, not the
    # -6.7126 value that no commit holds.
    l5 = next(e for e in corrections if e["metric_id"] == "l5_hilbert_r2")
    assert l5["value"] == -5.662524090882017, l5
    assert "7e30d6ea did NOT change the headline" in l5["extra"]["reason"]


def test_the_cli_runs_standalone() -> None:
    """The check is RUNNABLE ON ITS OWN (the brief's requirement)."""
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "molmetal" / "scripts" / "run_ledger.py"),
         "provenance"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LEDGER PROVENANCE: PASS" in proc.stdout, proc.stdout
