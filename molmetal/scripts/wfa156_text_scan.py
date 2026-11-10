#!/usr/bin/env python3
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
"""UTF-8-safe text scanner (Wf-A156).

Why this module exists
----------------------
On 2026-11-01 Wf-A156 measured that the ``grep`` on PATH in this environment is a
zsh *shell function* that execs ``ugrep 7.8.4`` with ``-I`` (skip files detected as
binary) and ``--ignore-files`` (respect ``.gitignore``). Two independent
under-count channels follow from that:

1. ``-I`` skips a file whose bytes are not valid UTF-8, **even when the file is
   plain text with no NUL byte**.  ``paper/build_log.txt`` is exactly that file:
   a 256 716-line pdflatex transcript in ISO-8859.  ``grep -c -i yuelbond
   paper/build_log.txt`` returns exit status 1 **with zero bytes of stdout**,
   which an audit read as "0 hits".  The true count is 68.
2. ``--ignore-files`` hides every ``.gitignore``-matched path, so a recursive
   scan silently covers less than the tree an author believes it covers.

This module is the shared replacement: pure Python, byte-level reads, decode
with ``errors="replace"``, explicit include/exclude lists, and a hard
requirement that **"skipped" is never reported as "0"**.

Design rules
------------
* Never invoke ``grep`` (or any shell) for matching.  Only ``git`` is ever
  shelled out to, and only to enumerate paths.
* Never ``os.walk`` this repository (201 GB, 43 GB of ``.pt`` files).  Path
  enumeration goes through ``git ls-files``.
* Never silently drop a file.  A file that cannot be decoded or read is
  returned with ``kind != "utf8"`` and reported in ``files_skipped``.
* ``binary`` means "contains a NUL byte"; ``non-utf8-text`` means "no NUL byte
  but not valid UTF-8".  Both are *scannable text* for our purposes and are
  scanned with ``errors="replace"``.  ``ugrep -I`` skips both.

Public API
----------
``classify_bytes`` / ``decode_bytes`` / ``read_text_safe``
    byte-level helpers.
``iter_repo_files`` / ``tracked_files`` / ``ignored_files``
    path enumeration via ``git ls-files`` (no filesystem walk).
``scan_file`` / ``scan`` / ``summarise``
    matching, with skipped-vs-zero kept distinguishable.
``census``
    encoding census over a path set.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

__all__ = [
    "ScanResult",
    "census",
    "classify_bytes",
    "decode_bytes",
    "ignored_files",
    "iter_repo_files",
    "read_text_safe",
    "scan",
    "scan_file",
    "summarise",
    "tracked_files",
    "untracked_files",
]

#: Encoding classifications produced by :func:`classify_bytes`.
KIND_UTF8 = "utf8"
KIND_NON_UTF8_TEXT = "non-utf8-text"
KIND_BINARY = "binary"
KIND_MISSING = "missing"

#: ``ugrep -I`` skips everything that is not valid UTF-8 *or* contains a NUL.
#: These two kinds are therefore the ones PATH ``grep`` silently drops.
UGREP_I_SKIPPED_KINDS = frozenset({KIND_NON_UTF8_TEXT, KIND_BINARY})

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# byte-level helpers
# --------------------------------------------------------------------------
def classify_bytes(data: bytes) -> str:
    """Classify raw bytes as ``utf8`` / ``non-utf8-text`` / ``binary``.

    ``binary`` is decided by the presence of a NUL byte -- the same rule GNU
    grep uses.  A file that is *not* valid UTF-8 but carries no NUL byte is
    text (an ISO-8859 transcript, a latin-1 CSV, ...), and GNU grep reads it
    while ``ugrep -I`` skips it.
    """
    if b"\x00" in data:
        return KIND_BINARY
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return KIND_NON_UTF8_TEXT
    return KIND_UTF8


def decode_bytes(data: bytes) -> tuple[str, int]:
    """Decode ``data`` as UTF-8 with ``errors="replace"``.

    Returns ``(text, n_replacement_chars)``.  Never raises on malformed input.
    """
    text = data.decode("utf-8", errors="replace")
    return text, text.count("�")


def read_text_safe(path: os.PathLike | str) -> tuple[str, str, int, str | None]:
    """Read a file as text without ever skipping it.

    Returns ``(text, kind, n_replacement_chars, error)``.  On an OS error the
    text is empty, ``kind`` is ``"missing"`` and ``error`` carries the message.
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:  # pragma: no cover - exercised by the unit tests
        return "", KIND_MISSING, 0, f"{type(exc).__name__}: {exc}"
    kind = classify_bytes(data)
    text, n_rep = decode_bytes(data)
    return text, kind, n_rep, None


# --------------------------------------------------------------------------
# path enumeration (git ls-files only -- never os.walk)
# --------------------------------------------------------------------------
def _git_ls_files(root: Path, args: Sequence[str]) -> list[str]:
    """Run ``git ls-files`` with ``args`` and return repo-relative paths."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", *args],
            cwd=str(root),
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    out = proc.stdout.decode("utf-8", errors="replace")
    return [p for p in out.split("\0") if p]


def tracked_files(root: os.PathLike | str = _DEFAULT_ROOT) -> list[str]:
    """Repo-relative paths of every tracked file."""
    return _git_ls_files(Path(root), ["--cached"])


def untracked_files(
    root: os.PathLike | str = _DEFAULT_ROOT, *, include_ignored: bool = False
) -> list[str]:
    """Untracked working-tree paths.

    With ``include_ignored=False`` (the default) ``.gitignore``-matched paths
    are excluded -- the same blind spot as PATH ``grep --ignore-files``.  With
    ``include_ignored=True`` they are returned too.
    """
    root = Path(root)
    untracked = _git_ls_files(root, ["--others", "--exclude-standard"])
    if not include_ignored:
        return untracked
    ignored = _git_ls_files(root, ["--others", "--ignored", "--exclude-standard"])
    return untracked + ignored


def ignored_files(
    root: os.PathLike | str = _DEFAULT_ROOT, patterns: Sequence[str] = ("*",)
) -> list[str]:
    """Working-tree paths that ``.gitignore`` hides from git's default view."""
    root = Path(root)
    args = ["--others", "--ignored", "--exclude-standard"]
    args.extend(patterns)
    return _git_ls_files(root, args)


def _match_rel(rel: str, pattern: str) -> bool:
    """Match a repo-relative POSIX path against a glob-ish pattern.

    Supported forms:

    ``*.md``
        basename match (case-sensitive on POSIX).
    ``docs/*.md``
        ``fnmatch`` over the whole relative path.
    ``paper/**``
        the directory itself and everything beneath it.
    """
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return rel == prefix or rel.startswith(prefix + "/")
    if "/" not in pattern:
        return fnmatch.fnmatch(os.path.basename(rel), pattern)
    return fnmatch.fnmatch(rel, pattern)


def iter_repo_files(
    root: os.PathLike | str = _DEFAULT_ROOT,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    untracked: bool = True,
    include_ignored: bool = False,
    files: Sequence[str] | None = None,
) -> list[str]:
    """Enumerate candidate files, honouring explicit include/exclude lists.

    ``include`` empty means "everything".  A path is kept when it matches at
    least one include pattern (or there are none) and matches no exclude
    pattern.
    """
    root = Path(root)
    if files is not None:
        candidates = list(files)
    else:
        candidates = tracked_files(root)
        if untracked:
            candidates.extend(untracked_files(root, include_ignored=include_ignored))
    seen: set[str] = set()
    out: list[str] = []
    for rel in candidates:
        if rel in seen:
            continue
        seen.add(rel)
        if include and not any(_match_rel(rel, pat) for pat in include):
            continue
        if exclude and any(_match_rel(rel, pat) for pat in exclude):
            continue
        out.append(rel)
    return sorted(out)


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScanResult:
    """Outcome of scanning one file.

    ``matches`` is the number of matching lines.  ``kind`` says how the bytes
    were classified.  Two different properties must not be conflated:

    ``skipped``
        True only when the file could **not** be searched at all (unreadable /
        missing).  A non-UTF-8 *text* file is **not** skipped -- it is
        searched with a lossy decode, so a hit in it is a hit.
    ``ugrep_I_skipped``
        True iff PATH ``grep -I`` would have dropped this file.  This is the
        under-count surface: the number of files a PATH-``grep`` census
        silently omits.
    """

    path: str
    kind: str
    matches: int = 0
    line_numbers: tuple[int, ...] = ()
    replacement_chars: int = 0
    error: str | None = None

    @property
    def skipped(self) -> bool:
        """True iff the file was **not searched** (unreadable / missing)."""
        return self.kind == KIND_MISSING or self.error is not None

    @property
    def ugrep_I_skipped(self) -> bool:
        """True iff PATH ``grep -I`` would have dropped this file."""
        return self.kind in UGREP_I_SKIPPED_KINDS


def count_lines(text: str, pattern: str, *, case_insensitive: bool = True) -> list[int]:
    """1-based line numbers of lines matching ``pattern``.

    ``pattern`` is a Python regular expression.  Invalid patterns raise
    ``re.error`` -- never a silent zero.
    """
    flags = re.IGNORECASE if case_insensitive else 0
    rx = re.compile(pattern, flags)
    hits: list[int] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if rx.search(line):
            hits.append(idx)
    return hits


def scan_file(
    path: os.PathLike | str,
    pattern: str,
    *,
    case_insensitive: bool = True,
) -> ScanResult:
    """Scan a single file.  Non-UTF-8 text is still scanned."""
    text, kind, n_rep, err = read_text_safe(path)
    if err is not None or kind == KIND_MISSING:
        return ScanResult(str(path), KIND_MISSING, 0, (), n_rep, err)
    try:
        lines = count_lines(text, pattern, case_insensitive=case_insensitive)
    except re.error as exc:
        return ScanResult(str(path), kind, 0, (), n_rep, f"re.error: {exc}")
    return ScanResult(str(path), kind, len(lines), tuple(lines), n_rep, None)


def scan(
    root: os.PathLike | str = _DEFAULT_ROOT,
    pattern: str = "",
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    untracked: bool = True,
    include_ignored: bool = False,
    files: Sequence[str] | None = None,
    case_insensitive: bool = True,
    limit: int | None = None,
) -> list[ScanResult]:
    """Scan a path set and return one :class:`ScanResult` per file."""
    root = Path(root)
    rels = iter_repo_files(
        root,
        include=include,
        exclude=exclude,
        untracked=untracked,
        include_ignored=include_ignored,
        files=files,
    )
    out: list[ScanResult] = []
    for rel in rels:
        out.append(
            scan_file(
                root / rel,
                pattern,
                case_insensitive=case_insensitive,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def summarise(results: Iterable[ScanResult]) -> dict:
    """Aggregate scan results.

    The returned dict keeps three populations apart so that "no match" can
    never be confused with "not looked at" -- the exact confusion that
    produced the Wf-A151 "0 hits" artefact:

    * ``files_scanned``  -- files actually searched (every readable file,
      including non-UTF-8 text).
    * ``files_skipped``  -- files that could not be searched at all.
    * ``files_ugrep_I_skipped`` -- files a PATH ``grep -I`` census would have
      silently omitted.  These ARE in ``files_scanned``.
    """
    results = list(results)
    scanned = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]
    return {
        "files_total": len(results),
        "files_scanned": len(scanned),
        "files_skipped": len(skipped),
        "files_non_utf8_text": sum(1 for r in results if r.kind == KIND_NON_UTF8_TEXT),
        "files_binary": sum(1 for r in results if r.kind == KIND_BINARY),
        "files_ugrep_I_skipped": sum(1 for r in results if r.ugrep_I_skipped),
        "total_matches": sum(r.matches for r in results),
        "files_with_matches": sum(1 for r in results if r.matches > 0),
        "skipped_paths": [r.path for r in skipped],
        "ugrep_I_would_skip": [r.path for r in results if r.ugrep_I_skipped],
        "errors": {r.path: r.error for r in results if r.error},
    }


def census(
    root: os.PathLike | str = _DEFAULT_ROOT,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    untracked: bool = True,
    include_ignored: bool = False,
    files: Sequence[str] | None = None,
) -> dict:
    """Encoding census over a path set.

    Returns counts per kind plus the explicit lists of the files that PATH
    ``grep -I`` would skip.
    """
    root = Path(root)
    rels = iter_repo_files(
        root,
        include=include,
        exclude=exclude,
        untracked=untracked,
        include_ignored=include_ignored,
        files=files,
    )
    kinds: dict[str, int] = {
        KIND_UTF8: 0,
        KIND_NON_UTF8_TEXT: 0,
        KIND_BINARY: 0,
        KIND_MISSING: 0,
    }
    non_utf8_text: list[str] = []
    binary: list[str] = []
    total_bytes = 0
    for rel in rels:
        p = root / rel
        try:
            data = p.read_bytes()
        except OSError:
            kinds[KIND_MISSING] += 1
            continue
        total_bytes += len(data)
        kind = classify_bytes(data)
        kinds[kind] += 1
        if kind == KIND_NON_UTF8_TEXT:
            non_utf8_text.append(rel)
        elif kind == KIND_BINARY:
            binary.append(rel)
    return {
        "files_total": len(rels),
        "total_bytes": total_bytes,
        "kinds": kinds,
        "non_utf8_text": non_utf8_text,
        "binary": binary,
        "ugrep_I_skipped_total": kinds[KIND_NON_UTF8_TEXT] + kinds[KIND_BINARY],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wfa156_text_scan",
        description="UTF-8-safe text scanner (never invokes grep).",
    )
    p.add_argument("--root", default=str(_DEFAULT_ROOT), help="repository root")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="count pattern matches per file")
    s.add_argument("--root", dest="sub_root", default=None, help="repository root (overrides the global --root)")
    s.add_argument("--pattern", required=True, help="Python regular expression")
    s.add_argument("--include", action="append", default=[], help="include glob (repeatable)")
    s.add_argument("--exclude", action="append", default=[], help="exclude glob (repeatable)")
    s.add_argument("--file", action="append", default=[], help="explicit path relative to --root (repeatable)")
    s.add_argument("--case-sensitive", action="store_true")
    s.add_argument("--include-ignored", action="store_true", help="also scan .gitignore'd paths")
    s.add_argument("--no-untracked", action="store_true")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--json", action="store_true")

    c = sub.add_parser("census", help="encoding census over a path set")
    c.add_argument("--root", dest="sub_root", default=None, help="repository root (overrides the global --root)")
    c.add_argument("--include", action="append", default=[])
    c.add_argument("--exclude", action="append", default=[])
    c.add_argument("--include-ignored", action="store_true")
    c.add_argument("--no-untracked", action="store_true")
    c.add_argument("--json", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(getattr(args, "sub_root", None) or args.root or _DEFAULT_ROOT)
    common = {
        "include": tuple(args.include),
        "exclude": tuple(args.exclude),
        "untracked": not args.no_untracked,
        "include_ignored": args.include_ignored,
    }
    if args.cmd == "scan":
        results = scan(
            root,
            args.pattern,
            case_insensitive=not args.case_sensitive,
            limit=args.limit,
            files=tuple(args.file) if args.file else None,
            **common,
        )
        summary = summarise(results)
        if args.json:
            print(
                json.dumps(
                    {
                        "summary": summary,
                        "results": [
                            {
                                "path": r.path,
                                "kind": r.kind,
                                "matches": r.matches,
                                "skipped": r.skipped,
                            }
                            for r in results
                            if r.matches or r.skipped
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            for r in results:
                if r.matches or r.skipped:
                    flag = "SKIP" if r.skipped else "HIT "
                    print(f"{flag} {r.matches:>8}  {r.path}  [{r.kind}]")
            print(
                f"scanned={summary['files_scanned']} skipped={summary['files_skipped']} "
                f"matches={summary['total_matches']} files_with_matches={summary['files_with_matches']} "
                f"non_utf8_text={summary['files_non_utf8_text']} binary={summary['files_binary']} "
                f"path_grep_I_would_skip={summary['files_ugrep_I_skipped']}"
            )
        # exit 2 = "we looked at every file and found nothing"
        # exit 3 = "some files could not be searched at all"
        if summary["files_skipped"]:
            return 3
        return 0 if summary["total_matches"] else 2

    census_out = census(root, **common)
    if args.json:
        print(json.dumps(census_out, indent=2, ensure_ascii=False))
    else:
        print(f"files_total={census_out['files_total']} bytes={census_out['total_bytes']}")
        for kind, n in sorted(census_out["kinds"].items()):
            print(f"  {kind:<16} {n}")
        print(f"ugrep_I_would_skip={census_out['ugrep_I_skipped_total']}")
        for rel in census_out["non_utf8_text"]:
            print(f"  non-utf8-text: {rel}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
