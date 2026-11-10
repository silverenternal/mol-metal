# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`molmetal.scripts.wfa156_text_scan`.

Wf-A156 (2026-11-01): the repository's PATH ``grep`` is a zsh function wrapping
``ugrep`` with ``-I`` and ``--ignore-files``.  ``-I`` skips files whose bytes
are not valid UTF-8 **even when they are plain text with no NUL byte**, so a
``grep -c`` over such a file returns exit status 1 with *no output at all* --
which an audit read as "0 hits".  These tests pin the replacement semantics:

* a non-UTF-8 text file is **scanned**, not skipped;
* "skipped" and "0 matches" are reported through **different** fields;
* the module never shells out to ``grep``.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from molmetal.scripts import wfa156_text_scan as ts


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


LATIN1_LOG = (
    "This is a pdflatex transcript\n"
    "line two mentions YuelBond once\n"
    "line three mentions yuelbond again\n"
).encode("latin-1") + b"caf\xe9 latin-1 accent\n"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        check=True,
        env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
             "PATH": "/usr/bin:/bin"},
    )


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """A throwaway git repo with one tracked, one ignored, one untracked file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "kept.txt").write_text("kept body\n")
    (repo / "ignored.txt").write_text("ignored body\n")
    (repo / "untracked.txt").write_text("untracked body\n")
    _git(repo, "add", ".gitignore", "kept.txt")
    return repo


# --------------------------------------------------------------------------
# byte-level classification
# --------------------------------------------------------------------------
def test_classify_bytes_utf8():
    assert ts.classify_bytes(b"plain ascii\n") == ts.KIND_UTF8
    assert ts.classify_bytes("café\n".encode("utf-8")) == ts.KIND_UTF8


def test_classify_bytes_latin1_text_is_not_binary():
    """A latin-1 file has no NUL byte -- it is text, not binary."""
    assert b"\x00" not in LATIN1_LOG
    assert ts.classify_bytes(LATIN1_LOG) == ts.KIND_NON_UTF8_TEXT


def test_classify_bytes_nul_is_binary():
    assert ts.classify_bytes(b"abc\x00def") == ts.KIND_BINARY


def test_classify_bytes_matches_file_command_semantics():
    """GNU grep reads the latin-1 file; ugrep -I skips it."""
    data = LATIN1_LOG
    assert b"\x00" not in data  # GNU grep would NOT treat this as binary
    assert ts.classify_bytes(data) != ts.KIND_UTF8  # ugrep -I WOULD skip it


def test_decode_bytes_never_raises_and_counts_replacements():
    text, n_rep = ts.decode_bytes(b"ok\xe9\n")
    assert n_rep == 1
    assert "�" in text
    assert text.startswith("ok")


def test_read_text_safe_scans_latin1(tmp_path: Path):
    p = _write_bytes(tmp_path / "build_log.txt", LATIN1_LOG)
    text, kind, n_rep, err = ts.read_text_safe(p)
    assert err is None
    assert kind == ts.KIND_NON_UTF8_TEXT
    assert n_rep >= 1
    assert "YuelBond" in text  # the content survived the lossy decode


def test_read_text_safe_missing_file(tmp_path: Path):
    text, kind, n_rep, err = ts.read_text_safe(tmp_path / "nope.txt")
    assert kind == ts.KIND_MISSING
    assert err is not None
    assert text == ""


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
def test_count_lines_is_case_insensitive_and_one_based():
    text = "alpha\nBETA\ngamma\nbeta\n"
    assert ts.count_lines(text, "beta") == [2, 4]
    assert ts.count_lines(text, "beta", case_insensitive=False) == [4]


def test_count_lines_invalid_regex_raises_not_zero():
    with pytest.raises(re.error):
        ts.count_lines("alpha\n", "beta[")


def test_scan_file_finds_needle_in_latin1(tmp_path: Path):
    """The exact Wf-A151 shape: the needle lives in a non-UTF-8 text file."""
    p = _write_bytes(tmp_path / "build_log.txt", LATIN1_LOG)
    res = ts.scan_file(p, "yuelbond")
    assert res.matches == 2
    assert res.line_numbers == (2, 3)
    assert res.kind == ts.KIND_NON_UTF8_TEXT
    assert res.skipped is False
    assert res.ugrep_I_skipped is True  # PATH grep -I would have dropped it


def test_scan_file_utf8_is_not_ugrep_skipped(tmp_path: Path):
    p = _write_bytes(tmp_path / "ok.md", b"yuelbond here\n")
    res = ts.scan_file(p, "yuelbond")
    assert res.matches == 1
    assert res.ugrep_I_skipped is False


def test_scan_file_no_match_is_zero_not_skipped(tmp_path: Path):
    p = _write_bytes(tmp_path / "ok.md", b"nothing here\n")
    res = ts.scan_file(p, "yuelbond")
    assert res.matches == 0
    assert res.skipped is False


def test_scan_file_binary_is_flagged_not_silent(tmp_path: Path):
    """A binary file is still searched; ``ugrep_I_skipped`` flags the risk."""
    p = _write_bytes(tmp_path / "blob.bin", b"\x00\x01yuelbond\x00")
    res = ts.scan_file(p, "yuelbond")
    assert res.kind == ts.KIND_BINARY
    assert res.ugrep_I_skipped is True
    assert res.skipped is False
    assert res.matches == 1  # searched, not hidden


# --------------------------------------------------------------------------
# summary: "skipped" must never be reported as "0 hits"
# --------------------------------------------------------------------------
def test_summarise_separates_skipped_from_zero():
    results = [
        ts.ScanResult("a.md", ts.KIND_UTF8, matches=0),
        ts.ScanResult("b.log", ts.KIND_NON_UTF8_TEXT, matches=0),
        ts.ScanResult("c.md", ts.KIND_UTF8, matches=3),
        ts.ScanResult("d.md", ts.KIND_MISSING, 0, (), 0, "OSError: gone"),
    ]
    s = ts.summarise(results)
    assert s["files_total"] == 4
    assert s["files_scanned"] == 3  # b.log WAS searched
    assert s["files_skipped"] == 1  # only the unreadable one
    assert s["files_non_utf8_text"] == 1
    assert s["files_ugrep_I_skipped"] == 1
    assert s["total_matches"] == 3
    assert s["skipped_paths"] == ["d.md"]
    assert s["ugrep_I_would_skip"] == ["b.log"]
    assert "d.md" in s["errors"]


# --------------------------------------------------------------------------
# path enumeration
# --------------------------------------------------------------------------
def test_iter_repo_files_include_exclude(git_repo: Path):
    files = ts.iter_repo_files(git_repo, include=("*.txt",), exclude=("ignored*",))
    assert "kept.txt" in files
    assert "untracked.txt" in files
    assert ".gitignore" not in files
    assert "ignored.txt" not in files


def test_iter_repo_files_exclude_directory_recursive(git_repo: Path):
    (git_repo / "sub").mkdir()
    (git_repo / "sub" / "deep.txt").write_text("x\n")
    files = ts.iter_repo_files(git_repo, include=("*.txt",), exclude=("sub/**",))
    assert not any(f.startswith("sub/") for f in files)


def test_gitignore_channel_is_visible_only_on_request(git_repo: Path):
    """PATH ``grep --ignore-files`` hides gitignored paths; this API must not."""
    default = ts.iter_repo_files(git_repo, untracked=True, include_ignored=False)
    assert "ignored.txt" not in default
    assert "kept.txt" in default

    with_ignored = ts.iter_repo_files(git_repo, untracked=True, include_ignored=True)
    assert "ignored.txt" in with_ignored


def test_tracked_files_are_found(git_repo: Path):
    tracked = ts.tracked_files(git_repo)
    assert "kept.txt" in tracked
    assert ".gitignore" in tracked


def test_iter_repo_files_explicit_files_bypasses_git(tmp_path: Path):
    files = ts.iter_repo_files(tmp_path, files=["x.md", "y.md"], include=("*.md",))
    assert files == ["x.md", "y.md"]


# --------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------
def test_census_classifies_and_lists(tmp_path: Path):
    _write_bytes(tmp_path / "a.md", b"utf8 ok\n")
    _write_bytes(tmp_path / "b.log", LATIN1_LOG)
    _write_bytes(tmp_path / "c.bin", b"\x00\x01")
    out = ts.census(tmp_path, untracked=False, files=["a.md", "b.log", "c.bin"])
    assert out["files_total"] == 3
    assert out["kinds"][ts.KIND_UTF8] == 1
    assert out["kinds"][ts.KIND_NON_UTF8_TEXT] == 1
    assert out["kinds"][ts.KIND_BINARY] == 1
    assert out["non_utf8_text"] == ["b.log"]
    assert out["ugrep_I_skipped_total"] == 2


# --------------------------------------------------------------------------
# the whole point: no grep, ever
# --------------------------------------------------------------------------
def test_module_never_shells_out_to_grep():
    """Parse the module source: no grep-family binary may be invoked."""
    src = Path(ts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    grep_families = {"grep", "rg", "ugrep", "ag", "ack", "ripgrep"}

    # 1. no list-literal command vector may name a grep-family binary
    bad_lists: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    if elt.value.lower() in grep_families:
                        bad_lists.append(ast.dump(node))
    assert not bad_lists, f"grep-family command vectors: {bad_lists}"

    # 2. no os.system / os.popen / os.exec* anywhere
    bad_shell: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                if node.func.attr in {"system", "popen", "execv", "execvp", "execvpe"}:
                    bad_shell.append(ast.dump(node))
    assert not bad_shell, f"shell invocations: {bad_shell}"

    # 3. every subprocess argv literal must start with "git"
    bad_argv: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"):
            continue
        if node.func.attr not in {"run", "Popen", "call", "check_call", "check_output"}:
            continue
        if not node.args:
            bad_argv.append(ast.dump(node))
            continue
        first = node.args[0]
        if isinstance(first, ast.List) and first.elts:
            head = first.elts[0]
            if isinstance(head, ast.Constant) and head.value == "git":
                continue
        bad_argv.append(ast.dump(first))
    assert not bad_argv, f"non-git subprocess argv: {bad_argv}"


def test_module_defines_no_grep_shim():
    """The module must not expose a grep-like alias."""
    src = Path(ts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in targets:
                if name.lower() in {"grep", "grep_bin", "grep_cmd", "ugrep"}:
                    bad.append(name)
    assert not bad, f"grep-like aliases defined: {bad}"
