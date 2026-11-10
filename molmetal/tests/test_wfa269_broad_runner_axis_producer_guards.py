# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Wf-A269 wave-32 (2026-11-07) — mechanism locks for the BROAD runner-coverage
producer's THREE defect classes.

WHAT THIS FILE LOCKS
--------------------
``molmetal/scripts/wfa269_broad_runner_axis.py`` measures the two published
BROAD runner-coverage rows.  A read-only audit found three defects in it.
Each is a MECHANISM, not a symptom, so each gets a lock here:

  G1  GIT C-QUOTING.  ``git ls-tree`` / ``git ls-files`` / ``git grep -l``
      C-quote every path that is not pure ASCII (``core.quotePath`` defaults
      to true): a Chinese path comes back as ``"\\347\\247\\221..."``.  A
      quoted path CANNOT BE READ BACK — ``git cat-file --batch`` reports it
      ``missing`` (MEASURED, this file, ``test_g1b_a_c_quoted_path_cannot_be_
      read_back``), the quoted string is not a file on disk, and
      ``git show HEAD:"\\347..."`` fails.  So a listing that keeps the
      quoting turns every non-ASCII tracked path into a file that reads as
      EMPTY — and an EMPTY file still counts as a WITNESS.  This repo has
      exactly 3 such tracked paths.

        NEW RULE (this wave): A PATH LISTED BY GIT IS QUOTED WHEN IT IS NOT
        ASCII — SO A PATH THAT CANNOT BE READ BACK IS A PATH THAT CANNOT BE
        EVIDENCE.

  G2  BATCH CURSOR.  ``git cat-file --batch`` emits EXACTLY ONE header per
      input spec, in order — including for a spec that does not exist, whose
      header is ``"<spec> missing"`` (two fields).  A consumer that indexes
      the spec list by the number of objects it has collected so far advances
      the cursor only on a FOUND object, so ONE missing spec shifts every
      later object onto the WRONG spec.

  G3  TREE SEPARATION.  ``git grep`` with NO revision searches the WORKING
      TREE; the index is only reachable with ``git grep --cached``.  A
      producer that asks for the "index" content source without ``--cached``
      silently measures the working tree, and a cell whose BASE came from the
      index and whose PER-CANDIDATE delta came from the disk is a mixed-tree
      cell — the very class the wave's catalogue entry is about.

WHY TWO LAYERS
--------------
Layer 1 (G1a..G3a) states the git-plumbing CONTRACT this repo relies on and
does NOT need the producer, so it still runs on a clean checkout.  Layer 2
(G1d..G3c) states the producer's OBLIGATIONS and needs the producer file.

HONEST LIMIT (AGENTS.md 6.269.23.12 — working tree vs repository claim)
-----------------------------------------------------------------------
``molmetal/scripts/wfa269_broad_runner_axis.py`` is UNTRACKED (``??``) as of
2026-11-07 and has never been committed, so the Layer-2 locks are a
WORKING-TREE claim until the U1 commit lands: a clean checkout of HEAD runs
Layer 1 only.  The Layer-2 skip is DECLARED, with its reason, and the reason
is printed by pytest — an UNDECLARED skip would be a violation
(AGENTS.md 6.269.21).

CPU-only. 0 GPU. Standard library + pytest only (no torch, no numpy) so the
group is enforceable in the pytest-only CI venv.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO_ROOT / "molmetal" / "scripts" / "wfa269_broad_runner_axis.py"

#: The pattern the producer's witness search uses ("names a population file").
MENTION_PATTERN = r"test_[A-Za-z0-9_]+\.py"

PRODUCER_ABSENT_REASON = (
    "DECLARED-SKIP: molmetal/scripts/wfa269_broad_runner_axis.py is absent "
    "from this checkout. It is UNTRACKED in the working tree and was never "
    "committed (AGENTS.md 6.269.23.12), so the Layer-2 producer locks cannot "
    "run here. The Layer-1 git-plumbing contracts still run."
)


# --------------------------------------------------------------------------- #
# git helpers                                                                   #
# --------------------------------------------------------------------------- #
def _git_bytes(*args: str, input_bytes: bytes | None = None,
               check: bool = True) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        input=input_bytes, capture_output=True,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed rc={proc.returncode}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:400]}")
    return proc.stdout


def _git_text(*args: str, check: bool = True) -> str:
    return _git_bytes(*args, check=check).decode("utf-8", "replace")


def _raw_paths(*args: str) -> list[str]:
    """The RAW (never C-quoted) paths of a git listing."""
    out = _git_bytes("-c", "core.quotePath=false", *args)
    return [p for p in out.decode("utf-8", "surrogateescape").split("\x00") if p]


def _batch_headers(specs: list[str]) -> tuple[list[list[str]], dict[str, str]]:
    """Run `git cat-file --batch` and parse it.

    Returns ``(headers, contents)`` where ``headers`` is one entry per input
    spec (each a list of header fields) and ``contents`` maps a spec to its
    content — a MISSING spec maps to ``""``.

    The parser advances on EVERY header, which is the whole point of G2:
    ``git cat-file --batch`` emits one header per spec, including the
    two-field ``"<spec> missing"`` header.
    """
    if not specs:
        return [], {}
    out = _git_bytes("cat-file", "--batch",
                     input_bytes=("\n".join(specs) + "\n").encode())
    headers: list[list[str]] = []
    contents: dict[str, str] = {}
    i = 0
    n = len(out)
    cursor = 0
    while i < n and cursor < len(specs):
        nl = out.find(b"\n", i)
        if nl < 0:
            break
        header = out[i:nl].decode("utf-8", "replace").split()
        i = nl + 1
        spec = specs[cursor]
        cursor += 1
        headers.append(header)
        if len(header) < 3:
            contents[spec] = ""
            continue
        try:
            size = int(header[2])
        except ValueError:
            contents[spec] = ""
            continue
        contents[spec] = out[i:i + size].decode("utf-8", "replace")
        i += size + 1
    return headers, contents


def _non_ascii_tracked_paths() -> list[str]:
    return [p for p in _raw_paths("ls-files", "-z")
            if any(ord(c) > 127 for c in p)]


def _load_producer():
    spec = importlib.util.spec_from_file_location(
        "wfa269_broad_runner_axis_under_test", PRODUCER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =========================================================================== #
# LAYER 1 — the git-plumbing CONTRACT this repo relies on (no producer needed) #
# =========================================================================== #
def test_g1a_the_default_git_listing_c_quotes_the_non_ascii_paths() -> None:
    """The PREMISE of G1: git's default listing hides non-ASCII paths."""
    raw = _raw_paths("ls-files", "-z")
    non_ascii = [p for p in raw if any(ord(c) > 127 for c in p)]
    if not non_ascii:
        pytest.skip(
            "DECLARED-SKIP: this checkout tracks no non-ASCII path, so the "
            "C-quoting defect cannot arise here.")

    default_lines = _git_text("ls-files").splitlines()
    assert len(default_lines) == len(raw), (
        "the default listing and the -z listing must have the same length")

    for p in non_ascii:
        assert p not in default_lines, (
            f"expected {p!r} to be C-quoted in `git ls-files` output; a raw "
            f"path here means git no longer quotes non-ASCII paths and this "
            f"premise test should be re-derived, not silently kept.")


def test_g1b_a_c_quoted_path_cannot_be_read_back() -> None:
    """The MECHANISM of G1: a quoted path is missing, not empty."""
    raw = _raw_paths("ls-files", "-z")
    default_lines = _git_text("ls-files").splitlines()
    non_ascii = [p for p in raw if any(ord(c) > 127 for c in p)]
    if not non_ascii:
        pytest.skip(
            "DECLARED-SKIP: this checkout tracks no non-ASCII path, so the "
            "quoted-path-unreadable mechanism cannot be demonstrated here.")

    quoted = []
    for path, line in zip(raw, default_lines):
        if path in non_ascii:
            quoted.append(line)
    assert quoted, "expected at least one C-quoted line"
    for q in quoted:
        assert q.startswith('"') and q.endswith('"'), (
            f"expected a C-quoted path, got {q!r}")

    headers, contents = _batch_headers([":" + q for q in quoted])
    assert len(headers) == len(quoted)
    for q, header in zip(quoted, headers):
        assert len(header) < 3, (
            f"the quoted path {q!r} was FOUND by git cat-file --batch; the "
            f"defect mechanism has changed and this test must be re-derived.")
        assert contents[":" + q] == "", (
            f"a missing object must read as an empty file, not as content")


def test_g1c_every_raw_tracked_path_is_readable_back() -> None:
    """The INVARIANT the producer must be able to rely on."""
    raw = _raw_paths("ls-files", "-z")
    # A path containing a newline would break the `--batch` line protocol.
    raw = [p for p in raw if "\n" not in p]
    specs = [":" + p for p in raw]
    headers, contents = _batch_headers(specs)
    assert len(headers) == len(specs), (
        f"git cat-file --batch emitted {len(headers)} headers for "
        f"{len(specs)} specs")
    missing = [s for s, h in zip(specs, headers) if len(h) < 3]
    assert not missing, (
        f"{len(missing)} tracked path(s) cannot be read back, e.g. "
        f"{missing[:3]} — a path that cannot be read back cannot be evidence")


def test_g2a_cat_file_batch_emits_exactly_one_header_per_spec() -> None:
    """The PLUMBING CONTRACT of G2: one header per spec, missing included."""
    specs = [
        ":README.md",
        ":molmetal/tests/test__wfa269_definitely_absent_zzz.py",
        ":AGENTS.md",
        ":molmetal/tests/test__wfa269_definitely_absent_yy.py",
    ]
    headers, contents = _batch_headers(specs)
    assert len(headers) == len(specs), (
        "git cat-file --batch must emit ONE header per input spec; a "
        "consumer that indexes its spec list by the number of objects it has "
        "collected so far mis-attributes every object after a missing spec.")
    two_field = [i for i, h in enumerate(headers) if len(h) < 3]
    assert two_field == [1, 3], (
        f"expected the two absent specs to produce two-field 'missing' "
        f"headers at positions 1 and 3, got {two_field}")
    assert contents[specs[1]] == "" and contents[specs[3]] == ""


def test_g3a_git_grep_without_a_revision_reads_the_working_tree() -> None:
    """The SEMANTICS of G3: no revision == the working tree, NOT the index."""
    diff_paths = [p for p in _git_text("diff", "--name-only").splitlines() if p]
    if not diff_paths:
        pytest.skip(
            "DECLARED-SKIP: the index and the working tree are identical on "
            "this checkout, so `git grep` (working tree) and `git grep "
            "--cached` (index) cannot be told apart here.")

    specs = [":" + p for p in diff_paths]
    _, index_contents = _batch_headers(specs)
    index_matches = {
        p for p in diff_paths
        if re.search(MENTION_PATTERN, index_contents.get(":" + p, ""))
    }
    disk_matches = {
        p for p in diff_paths
        if re.search(MENTION_PATTERN,
                     (REPO_ROOT / p).read_text(encoding="utf-8",
                                               errors="replace"))
    }
    discriminating = index_matches.symmetric_difference(disk_matches)
    if not discriminating:
        pytest.skip(
            "DECLARED-SKIP: the index and the working tree differ on "
            f"{len(diff_paths)} path(s), but the mention pattern matches the "
            "same set in both trees, so the two searches cannot be told "
            "apart on this checkout.")

    idx_set = set(_raw_paths("grep", "--cached", "-l", "-z", "-a", "-E",
                              MENTION_PATTERN))
    wt_set = set(_raw_paths("grep", "-l", "-z", "-a", "-E", MENTION_PATTERN))
    assert idx_set != wt_set, (
        "the index and the working tree give different answers for a path "
        "that differs between them, so `git grep` and `git grep --cached` "
        "MUST be told apart — a producer that reads the 'index' content "
        "source without --cached is measuring the working tree.")
    assert discriminating & (idx_set ^ wt_set), (
        "a path whose match status differs between the index and the working "
        "tree must appear in exactly one of the two grep sets")


# =========================================================================== #
# LAYER 2 — the producer's OBLIGATIONS                                          #
# =========================================================================== #
@pytest.mark.skipif(not PRODUCER_PATH.exists(), reason=PRODUCER_ABSENT_REASON)
def test_g1d_the_producer_lists_paths_without_c_quoting() -> None:
    """G1: the producer's own git listing must be quote-free."""
    producer = _load_producer()
    non_ascii = _non_ascii_tracked_paths()
    if not non_ascii:
        pytest.skip(
            "DECLARED-SKIP: this checkout tracks no non-ASCII path, so the "
            "producer's quoting behaviour cannot be observed here.")

    plain = producer._git("ls-files").splitlines()
    for p in non_ascii:
        assert p in plain, (
            f"the producer's own `git ls-files` listing does not contain the "
            f"raw path {p!r} — it is C-quoted, and a C-quoted path cannot be "
            f"read back, so it can never be evidence.")

    helpers = [n for n in ("_git_raw_paths", "_raw_paths", "_git_raw")
               if hasattr(producer, n)]
    assert helpers, (
        "the producer must expose a NUL-separated, quote-free path-listing "
        "helper (e.g. `_git_raw_paths`): a newline-separated listing cannot "
        "carry a path that contains a newline.")
    listed = set(getattr(producer, helpers[0])("ls-files", "-z"))
    for p in non_ascii:
        assert p in listed, (
            f"the producer's quote-free listing is missing {p!r}")


@pytest.mark.skipif(not PRODUCER_PATH.exists(), reason=PRODUCER_ABSENT_REASON)
def test_g1e_the_producer_can_read_a_non_ascii_tracked_path() -> None:
    """G1 end-to-end: a non-ASCII tracked path must be READABLE, not empty."""
    producer = _load_producer()
    non_ascii = _non_ascii_tracked_paths()
    if not non_ascii:
        pytest.skip(
            "DECLARED-SKIP: this checkout tracks no non-ASCII path.")
    specs = [":" + p for p in non_ascii]
    got = producer._batch_blobs(specs)
    for spec in specs:
        assert spec in got, f"{spec!r} was not returned by _batch_blobs"
        assert got[spec], (
            f"{spec!r} read back as EMPTY — a path that reads as empty is a "
            f"phantom witness")
        truth = _git_bytes("show", spec)
        assert got[spec] == truth.decode("utf-8", "replace")


@pytest.mark.skipif(not PRODUCER_PATH.exists(), reason=PRODUCER_ABSENT_REASON)
def test_g2b_the_producer_attributes_every_blob_to_its_own_spec() -> None:
    """G2: one missing spec must not shift every later blob."""
    producer = _load_producer()
    absent = ":molmetal/tests/test__wfa269_definitely_absent_zzz.py"
    specs = [":README.md", absent, ":AGENTS.md"]
    got = producer._batch_blobs(specs)
    assert set(got) == set(specs), (
        f"the producer must return one entry per spec; got {sorted(got)}")
    assert got[absent] == "", "an absent object must read as an empty file"
    for spec in (":README.md", ":AGENTS.md"):
        truth = _git_bytes("show", spec).decode("utf-8", "replace")
        assert got[spec] == truth, (
            f"{spec!r} was attributed the WRONG blob — one missing spec "
            f"shifted the whole batch.")


@pytest.mark.skipif(not PRODUCER_PATH.exists(), reason=PRODUCER_ABSENT_REASON)
def test_g3b_the_producer_reads_the_index_with_cached(monkeypatch) -> None:
    """G3: the 'index' content source must be the INDEX, not the working tree."""
    producer = _load_producer()
    calls: list[str | None] = []

    def recorder(rev):
        calls.append(rev)
        return set(), True

    monkeypatch.setattr(producer, "_mentioning_tracked_files", recorder)
    monkeypatch.setattr(producer, "surface", lambda *a, **k: [])
    monkeypatch.setattr(producer, "cover", lambda *a, **k: (set(), set()))

    producer.witness_search(pop=[])

    assert calls, (
        "the producer must measure its mention sets through "
        "`_mentioning_tracked_files`; if that call moved, this lock must be "
        "re-derived against the new call site.")
    assert "--cached" in calls, (
        f"the index content source must be measured with `git grep "
        f"--cached`; the producer asked for {calls!r}")
    assert calls.count(None) == 1, (
        f"exactly ONE content source may read the working tree (the 'disk' "
        f"source); the producer asked for {calls!r} — a second None means "
        f"the 'index' source is silently reading the working tree.")


@pytest.mark.skipif(not PRODUCER_PATH.exists(), reason=PRODUCER_ABSENT_REASON)
def test_g3c_the_producer_source_does_not_map_index_to_the_working_tree() -> None:
    """G3, source-level companion: the mapping must be pinned in the code."""
    src = PRODUCER_PATH.read_text(encoding="utf-8")
    assert not re.search(r'\(\s*"index"\s*,\s*None\s*\)', src), (
        "the producer maps the 'index' content source to the WORKING TREE "
        "(`None`); it must map to `--cached`.")
    assert "--cached" in src, (
        "the producer never mentions `--cached`; the index content source "
        "cannot be the index without it.")
