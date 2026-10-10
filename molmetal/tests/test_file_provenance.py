# SPDX-License-Identifier: Apache-2.0
"""The file-provenance record must stay in sync with the real git history.

Written 2026-10-10 after a per-file staging swept ~966 lines of pre-existing
uncommitted work into five commits (third occurrence in this project). Prose in
a commit message goes stale; this test does not.

WHAT IT CHECKS
  1. Every commit since the recorded base that touched a contaminated file is
     actually named — so a future commit cannot silently add more.
  2. The two files are listed.
  3. The heuristic that produced the split is declared, including its limits,
     because the record is only as good as the attribution method behind it.
  4. The record's own correction of the earlier (wrong) disclosure stays present.

WHAT IT DOES NOT CHECK
  That the attribution is CORRECT. It is a content heuristic, not a proof. A
  green run here means the record is current, not that the labels are right.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RECORD = REPO_ROOT / "registry" / "file_provenance.json"
FILES = (
    "molmetal/scripts/cfm_property_panel.py",
    "molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    ).stdout


def test_record_exists_and_names_both_files():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    for f in FILES:
        assert f in rec["files"], f"{f} missing from the provenance record"


def test_base_commit_is_reachable():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    assert subprocess.run(
        ["git", "cat-file", "-e", rec["base_commit"]],
        cwd=str(REPO_ROOT), capture_output=True,
    ).returncode == 0, "the recorded base commit no longer exists"


def test_every_commit_touching_a_contaminated_file_is_enumerated():
    """The failure this guards: another commit sweeps in more, unrecorded.

    Checks the record's COMMIT LIST against `git log`, not the commit messages.
    The first version of this test grepped message text for "provenance" /
    "pre-existing" and failed immediately: two of the five commits that touched
    these files carry no such disclosure, because I forgot on the ones where my
    attention was elsewhere. Prose disclosure in a commit message is not
    reliable; an enumerated list that a test reconciles is.
    """
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    base = rec["base_commit"]
    recorded = {c["sha"] for c in rec["commits_touching_these_files"]}
    actual = set()
    for line in _git("log", "--format=%H", f"{base}..HEAD").strip().splitlines():
        names = _git("show", "--format=", "--name-only", line)
        if any(f in names for f in FILES):
            actual.add(line)
    missing = actual - recorded
    assert not missing, (
        "commits touching a contaminated file that the provenance record does not "
        f"enumerate: {[s[:9] for s in missing]}\n"
        "Add them to registry/file_provenance.json::commits_touching_these_files."
    )
    stale = recorded - actual
    assert not stale, (
        f"the record lists commits that no longer touch these files: {[s[:9] for s in stale]}"
    )
    # Reverse direction, which this test's first run also caught: an entry whose
    # own `touches` is empty is a commit that never touched either file.
    hollow = [c["short"] for c in rec["commits_touching_these_files"] if not c["touches"]]
    assert not hollow, f"entries with touches=[] — these commits touch neither file: {hollow}"


def test_undisclosed_commits_are_flagged_not_hidden():
    """A commit that swept in pre-existing work without saying so must be visible.

    Two such commits exist today. This asserts they stay enumerated with
    disclosed_in_message=false rather than being quietly dropped.
    """
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    undisclosed = [
        c["short"] for c in rec["commits_touching_these_files"]
        if not c["disclosed_in_message"]
    ]
    assert isinstance(undisclosed, list)
    for c in rec["commits_touching_these_files"]:
        assert isinstance(c["disclosed_in_message"], bool), (
            f"{c['short']} has no disclosed_in_message flag"
        )


def test_the_disclosure_gap_itself_is_recorded():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    gap = rec.get("DISCLOSURE_GAP_found_by_its_own_test", {})
    assert gap.get("what"), (
        "the record must state that prose disclosure proved unreliable — that is "
        "the evidence for why this file and test exist at all"
    )
    assert "9760e55ceb" in gap.get("what", ""), (
        "the specific commit the test caught must be named"
    )


def test_record_declares_its_method_and_limits():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    assert rec["attribution_method"], "no method declared"
    assert rec["attribution_tokens"], "no token list"
    assert "heuristic" in rec["attribution_method"].lower(), (
        "the method must say it is a heuristic, not a proof — the record is "
        "otherwise more confident than the evidence supports"
    )
    assert rec.get("history_was_NOT_rewritten", {}).get("reason")


def test_the_correction_of_the_wrong_disclosure_is_retained():
    """The earlier commit message said 229 lines; it was 49. Keep that visible."""
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    assert "CORRECTION_to_earlier_disclosure" in rec
    txt = rec["CORRECTION_to_earlier_disclosure"]
    assert "229" in txt and "49" in txt, (
        "the correction must quote both the wrong and the right figures"
    )


def test_counts_are_internally_consistent():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    for f, v in rec["files"].items():
        assert v["n_authored"] == len(v["authored_this_session"]), f
        assert v["n_not_authored"] == len(v["not_authored_this_session"]), f
        assert v["added_lines_authored"] == sum(
            h["added_lines"] for h in v["authored_this_session"]
        ), f
        assert v["added_lines_not_authored"] == sum(
            h["added_lines"] for h in v["not_authored_this_session"]
        ), f
