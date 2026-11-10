# SPDX-License-Identifier: Apache-2.0
"""Wf-A200 -- the CROSS-WAVE LOCATOR auditor (read-only).

THE DEFECT CLASS
----------------
A *locator* is how one document points at a fact.  Two kinds:

* a **durable locator** -- a JSON PATH (e.g.
  ``band_provenance.competing_reading_and_what_this_does_not_settle``) or a
  SECTION HEADING.  It survives a re-render of the cited tree.
* a **perishable locator** -- a LINE NUMBER (e.g. ``report.md:139``).  It
  dies the moment the cited file is re-rendered, and it dies SILENTLY: the
  citation still parses, still looks plausible, and now points at a
  DIFFERENT sentence.

This module finds the perishable kind repo-wide and RESOLVES each one
against the file it claims, so a stale citation is distinguished from a
good one.  It is NOT a "flag every line number" checker: it compares the
text the citation claims with the text actually present at the cited line.

CLASSES
-------
``RESOLVES-AND-MATCHES``  the cited line exists and contains the claimed text
``RESOLVES-BUT-DIFFERS``  the cited line exists but does NOT contain the
                          claimed text -- the silent drift
``RESOLVES-NO-CLAIM``     the cited line exists; the citation carries no text
                          long enough to discriminate (honest, not a defect)
``TARGET-ABSENT``         the cited path does not exist on disk
``TARGET-IS-SELF``        the citation points at the citing file itself
``DATED-SUPERSESSION``    the citation sits in a DATED position (a dated
                          heading / dated JSON key / dated constant).  Per
                          AGENTS.md 5.5 rule #6 the superseded rendering is
                          REQUIRED to survive there, so this is history, not
                          a live claim
``SNAPSHOT``              the citation records a recorded scan result, or the
                          cited target is a sha256-PINNED primary-source
                          artefact whose recorded sha256 still matches (a
                          pinned target cannot drift)

READ-ONLY BY CONSTRUCTION
-------------------------
This module opens NOTHING for writing.  Every read goes through
``_open_read`` which refuses any mode other than ``"r"``/``"rb"``.  It
deletes nothing, moves nothing, and does not register itself as a
pre-flight gate.  It is a standalone script.

CPU-only.  No GPU is touched, queried or trained on.  The repo is never
walked with ``os.walk``: the file census comes from ``git ls-files`` (plus
``git ls-files --others --exclude-standard`` for repo-present untracked
artefacts), with a bounded depth-limited ``scandir`` fallback for roots
that are not git repositories.

Usage
-----
    python molmetal/scripts/wfa200_crosswave_locator_audit.py \
        --json-out molmetal/reports/wfa200_crosswave_locator_audit/report.json \
        --md-out   molmetal/reports/wfa200_crosswave_locator_audit/report.md
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

try:  # the atomic-write helper (AGENTS.md 5.6) -- same directory
    from wf386a3_atomic_write import atomic_write_file
except ImportError:  # pragma: no cover - when imported from elsewhere
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wf386a3_atomic_write import atomic_write_file

__version__ = "wfa200-crosswave-locator-audit/1.0"

# --------------------------------------------------------------------------
# READ-ONLY GUARD
# --------------------------------------------------------------------------
_ALLOWED_READ_MODES = ("r", "rb", "rt", "rU")


class NotReadOnly(RuntimeError):
    """Raised when something tries to open a file in a writing mode."""


def _open_read(path: "os.PathLike[str] | str", mode: str = "r"):
    """Open ``path`` for READING ONLY.  Any writing mode is refused."""
    if mode not in _ALLOWED_READ_MODES:
        raise NotReadOnly(
            "wfa200 auditor is read-only by construction; refused mode %r for %s"
            % (mode, path)
        )
    if mode == "rb":
        return open(path, mode)
    return open(path, mode, encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# FILE CENSUS -- git ls-files, NEVER os.walk
# --------------------------------------------------------------------------
TEXT_SUFFIXES = (
    ".md", ".py", ".json", ".tex", ".sh", ".yaml", ".yml", ".txt", ".bib",
    ".cfg", ".toml", ".ini", ".js", ".ts", ".csv", ".jsonl", ".rst", ".html",
    ".sql", ".env", ".make", ".mk", ".log", ".diff", ".patch", ".tex",
)

#: directories never descended into (huge / binary / vendored)
SKIP_DIR_NAMES = (
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "site-packages", ".idea",
)

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_SCAN_DEPTH = 12


def _git_lines(root: Path, args: list[str]) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root)] + args,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    # ``-z`` output is NUL-separated; plain output is newline-separated
    out = []
    for chunk in proc.stdout.split("\0"):
        for ln in chunk.split("\n"):
            ln = ln.strip()
            if ln:
                out.append(ln)
    return out


def census_files(
    root: Path,
    extra_dirs: list[str] | None = None,
    ignored_under: list[str] | None = None,
) -> tuple[list[str], str]:
    """Return (repo-relative file paths, census method).

    Primary method is ``git ls-files`` (tracked) plus ``git ls-files
    --others --exclude-standard`` (repo-present, untracked).  A bounded,
    depth-limited ``scandir`` fallback is used when the root is not a git
    repository -- ``os.walk`` is never used.

    ``ignored_under`` adds the GITIGNORED files under the given prefixes.
    Without it the census silently skips every ``*.json`` under
    ``molmetal/reports/**`` -- the repo ignores that pattern
    (``.gitignore:158``) -- so three of the five artefacts of the audited
    tree (``sources.json``, ``band_census.json`` and
    ``refutation/refutation.json``) are never read at all.  A census that
    skips its own subjects is not a census.
    """
    tracked = _git_lines(root, ["ls-files", "-z"])
    untracked = _git_lines(root, ["ls-files", "-z", "--others", "--exclude-standard"])
    if tracked or untracked:
        out = set(tracked) | set(untracked)
        n_ignored = 0
        for prefix in (ignored_under or []):
            extra = _git_lines(
                root,
                ["ls-files", "-z", "--others", "--ignored", "--exclude-standard",
                 "--", prefix.rstrip("/")],
            )
            n_ignored += len(extra)
            out.update(extra)
        method = "git ls-files (+ --others --exclude-standard"
        if n_ignored:
            method += "; + %d gitignored file(s) under %d prefix(es): %s" % (
                n_ignored,
                len(ignored_under or []),
                ", ".join(sorted(p.rstrip("/") for p in (ignored_under or []))),
            )
        method += ")"
        return sorted(out), method

    # fallback: bounded scandir recursion (never os.walk)
    found: list[str] = []
    roots = [root] + [root / d for d in (extra_dirs or [])]
    for base in roots:
        if not base.is_dir():
            continue
        stack: list[tuple[Path, int]] = [(base, 0)]
        while stack:
            cur, depth = stack.pop()
            if depth > MAX_SCAN_DEPTH:
                continue
            try:
                entries = list(os.scandir(cur))
            except OSError:
                continue
            for ent in entries:
                if ent.is_dir(follow_symlinks=False):
                    if ent.name in SKIP_DIR_NAMES:
                        continue
                    stack.append((Path(ent.path), depth + 1))
                elif ent.is_file(follow_symlinks=False):
                    rel = os.path.relpath(ent.path, str(root))
                    found.append(rel.replace(os.sep, "/"))
    return sorted(set(found)), "bounded scandir fallback (root is not a git repo)"


def _is_text_candidate(rel: str) -> bool:
    low = rel.lower()
    if low.endswith(TEXT_SUFFIXES):
        return True
    return False


# --------------------------------------------------------------------------
# LOCATOR EXTRACTION
# --------------------------------------------------------------------------
#: ``<path>:<line>`` or ``<path>:<line>-<line2>``
LOCATOR_RE = re.compile(
    r"(?<![A-Za-z0-9_./\\-])"
    r"(?P<path>(?:[A-Za-z0-9_.\\-]+/)*[A-Za-z0-9_.\\-]+\.[A-Za-z0-9]{1,8})"
    r":(?P<line>\d{1,7})"
    r"(?:\s*[-,]\s*(?P<line2>\d{1,7}))?"
)

#: a quoted claim on the same line, either single or double quoted
QUOTE_RE = re.compile(r"['\"]([^'\"]{4,400})['\"]")
#: a markdown block quote on one of the following lines
BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
#: how far after a locator a quoted claim may start and still be the
#: locator's claim
CLAIM_WINDOW = 120
#: text that is metadata, not a claim about the cited content
METADATA_PROBE_RE = re.compile(
    r"^\(?\s*(mtime|updated|modified|created|sha256|bytes?|line)\b|^\(?\s*20\d\d-\d\d-\d\d",
    re.IGNORECASE,
)
#: a line that RECORDS a scan result (a census, an inventory, a coverage
#: sweep).  Deliberately narrow: the word "scanned" alone is not enough --
#: a prose sentence may mention a scan while making an unrelated claim.
SCAN_LINE_RE = re.compile(
    r"(recorded scan|scan result|files[_ ]scanned|census result|recorded census"
    r"|scan of \d|scanned \d{2,}|HEAD blobs? (were|scanned))",
    re.IGNORECASE,
)

#: words that carry no discriminating power
STOPWORDS = frozenset(
    """a an the and or of to in is are was were be been being it its this that
    these those with for from on at by as not no we our us they them their
    he she his her i you your but if then than so such can could may might
    will would shall should do does did done have has had having there here
    what which who whom whose when where why how all any both each few more
    most other some only own same too very s t just don now also""".split()
)


def _norm(text: str) -> str:
    """Normalise for comparison: lowercase, drop emphasis/backticks, collapse ws.

    The typographic space before a percent sign is dropped, because the same
    number is printed both as ``F1=92.7 %`` (prose) and as ``F1=92.7\\%``
    (LaTeX); without this the auditor would call a CORRECT citation drifted.
    """
    t = text.lower()
    t = t.replace("*", "").replace("`", "").replace("\\%", "%")
    t = t.replace("\\,", ",").replace("\\&", "&").replace("\\_", "_")
    t = re.sub(r"[^\w\s%.,:;()\[\]/&+-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+%", "%", t)
    return t.strip()


def _content_words(text: str) -> list[str]:
    return [w for w in _norm(text).split() if w not in STOPWORDS and len(w) > 2]


def _alpha_words(text: str) -> list[str]:
    return [w for w in _norm(text).split() if w.isalpha() and len(w) >= 3]


def _quote_spans(s: str) -> list[tuple[int, int, str]]:
    """Spans of single- or double-quoted text, as (start, end, content).

    An apostrophe between two letters (``wfa161's``) is NOT a delimiter.
    """
    spans: list[tuple[int, int, str]] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in ("'", '"'):
            if (
                ch == "'"
                and 0 < i < n - 1
                and (s[i - 1].isalnum()) and (s[i + 1].isalnum())
            ):
                i += 1
                continue
            j = s.find(ch, i + 1)
            if j == -1:
                i += 1
                continue
            content = s[i + 1 : j]
            if len(content) >= 4:
                spans.append((i, j + 1, content))
            i = j + 1
        else:
            i += 1
    return spans


def _all_quote_spans(s: str) -> list[tuple[int, int, str]]:
    """Quote spans INCLUDING the ones nested inside an outer quote.

    A JSON value like ``"a:139 -- 'claim here'"`` has an inner single-quoted
    span that the flat scanner would skip.  The inner span is exactly the
    machine-verifiable claim, so it must be visible.
    """
    out: list[tuple[int, int, str]] = []
    for a, b, content in _quote_spans(s):
        out.append((a, b, content))
        for a2, b2, c2 in _quote_spans(content):
            out.append((a + 1 + a2, a + 1 + b2, c2))
    out.sort()
    return out


def _probe(text: str) -> str | None:
    """The distinctive fragment of a claim: the longest '...' -separated piece.

    Returns None when the claim carries too little signal to discriminate,
    or when it is metadata (a timestamp, a byte count, a sha) rather than a
    statement about the cited content.
    """
    best = None
    best_len = -1
    for piece in text.split("..."):
        piece = piece.strip().strip("'\"` ").strip()
        if not piece:
            continue
        words = _content_words(piece)
        if len(words) < 1:
            continue
        if not (_alpha_words(piece) or any(any(c.isdigit() for c in w) for w in words)):
            # a claim with neither words nor digits discriminates nothing
            continue
        norm = _norm(piece)
        if METADATA_PROBE_RE.search(norm):
            continue
        if len(norm) > best_len:
            best_len = len(norm)
            best = norm
    if best is None or best_len < 8:
        return None
    # cap the probe so a long claim still matches a single cited line
    return best[:120]


def _anchor(text: str) -> str | None:
    """The leading words of a claim that must be present ON the cited line.

    A citation asserts that the claimed text STARTS at the cited line, so the
    leading words of the claim must be present ON that line, contiguously;
    the REST of the claim may WRAP onto the following lines.  A single-word
    anchor is too weak -- a content word like ``band`` occurs on almost any
    line of the corpus -- so the anchor is the first three words.
    """
    for piece in text.split("..."):
        piece = piece.strip().strip("'\"` ").strip()
        if not piece:
            continue
        toks = _norm(piece).split()
        if not toks:
            continue
        return " ".join(toks[:3])
    return None


# --------------------------------------------------------------------------
# POSITION CLASSIFICATION (dated / snapshot)
# --------------------------------------------------------------------------
DATED_SEGMENT_RE = re.compile(r"(superseded|_wfa\d+|\d{4}_\d{2}_\d{2})", re.IGNORECASE)
DATED_HEADING_RE = re.compile(r"(^##\s)|(\d{4}-\d{2}-\d{2})")
SCAN_KEY_RE = re.compile(r"(scan|census|inventory|coverage|snapshot|ledger)", re.IGNORECASE)


def _md_dated_lines(text: str) -> set[int]:
    """Line numbers inside a dated markdown section (a supersession record).

    Only a LEVEL-2 heading (``## ...``) opens or closes a dated region; a
    ``###`` sub-heading inside a dated section does NOT close it.  This is
    the same rule the repo's own ``test_wfa192_live_x_closure.py`` uses.
    """
    out: set[int] = set()
    dated = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith("## ") and not line.startswith("###"):
            dated = line.startswith("## §") or bool(
                re.search(r"\d{4}-\d{2}-\d{2}", line)
            )
        if dated:
            out.add(i)
    return out


def _json_dated_paths(data) -> set[str]:
    """JSON paths whose key segment is dated."""
    out: set[str] = set()

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}.{k}" if path else str(k)
                if DATED_SEGMENT_RE.search(str(k)):
                    out.add(p)
                walk(v, p)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(data, "")
    return out


def _json_leaves(data) -> dict[str, str]:
    """Every string leaf of a JSON document, keyed by its JSON PATH.

    ``json.loads`` preserves DOCUMENT ORDER in its dicts, so the leaves come
    out in the order they appear in the file.
    """
    out: dict[str, str] = {}

    def collect(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                collect(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                collect(v, f"{path}[{i}]")
        elif isinstance(node, str):
            out[path] = node

    collect(data, "")
    return out


def _json_leaf_spans(text: str, data) -> dict[str, tuple[int, int]]:
    """The character span each string leaf occupies in the RAW file text.

    The span is located by searching for the leaf's own (escaped) text with
    a MONOTONIC cursor, because the leaves come out in document order.  The
    span is what lets an occurrence be attributed to the leaf it actually
    sits in.  Attributing it instead to "the first leaf that mentions the
    cited path" mis-attributes a DATED copy of a locator to the LIVE leaf
    that mentions the same path -- which would make the auditor classify
    correctly-dated history as a live locator and demand its deletion, which
    AGENTS.md 5.5 rule #6 forbids.
    """
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for jpath, value in _json_leaves(data).items():
        needle = json.dumps(value)[1:-1]
        idx = text.find(needle, cursor)
        if idx == -1:
            idx = text.find(value, cursor)
        if idx == -1:
            continue
        spans[jpath] = (idx, idx + len(needle))
        cursor = idx + len(needle)
    return spans


#: a real python assignment (``NAME = ...``); a quoted dict key is NOT one
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=")


def _py_dated_lines(lines: list[str]) -> set[int]:
    """Python line numbers inside a dated constant / comment / docstring.

    A DATED CONSTANT is a dated position in its WHOLE BODY: AGENTS.md 5.5
    rule #6 requires the superseded rendering to survive verbatim inside a
    dated constant, so every line of that constant is dated.  The enclosing
    assignment is tracked forward (the nearest enclosing ``NAME = ...``),
    because a dict key is not an assignment.
    """
    out: set[int] = set()
    try:
        with warnings.catch_warnings():
            # some repo .py files carry invalid escape sequences; the parse
            # still succeeds and the warning is not the auditor's business.
            warnings.simplefilter("ignore")
            tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                for ln in range(body[0].lineno, body[0].end_lineno + 1):
                    out.add(ln)
    current: str | None = None
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out.add(i)
            continue
        m = _ASSIGN_RE.match(line)
        if m:
            current = m.group(1)
        elif re.match(r"^\s*(def|class)\s", line):
            current = None
        if current and DATED_SEGMENT_RE.search(current):
            out.add(i)
    return out


# --------------------------------------------------------------------------
# TARGET RESOLUTION
# --------------------------------------------------------------------------
class Resolver:
    """Resolve a cited path to concrete files on disk."""

    def __init__(self, root: Path, files: list[str]):
        self.root = root
        self.files = files
        self.by_basename: dict[str, list[str]] = {}
        for rel in files:
            self.by_basename.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)
        self._line_cache: dict[str, list[str]] = {}
        self._sha_cache: dict[str, str | None] = {}

    def _abs(self, rel: str) -> Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        try:
            return self._abs(rel).is_file()
        except OSError:
            return False

    def lines(self, rel: str) -> list[str]:
        if rel not in self._line_cache:
            try:
                with _open_read(self._abs(rel)) as fh:
                    self._line_cache[rel] = fh.read().splitlines()
            except OSError:
                self._line_cache[rel] = []
        return self._line_cache[rel]

    def sha256(self, rel: str) -> str | None:
        if rel not in self._sha_cache:
            h = hashlib.sha256()
            try:
                with _open_read(self._abs(rel), "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                self._sha_cache[rel] = h.hexdigest()
            except OSError:
                self._sha_cache[rel] = None
        return self._sha_cache[rel]

    def candidates(self, cited: str, citing_rel: str) -> list[str]:
        """Files that the cited path could denote, best first."""
        cited = cited.replace("\\", "/").lstrip("./")
        out: list[str] = []
        if self.exists(cited):
            out.append(cited)
        # relative to the citing file's directory
        parent = os.path.dirname(citing_rel)
        cand = f"{parent}/{cited}" if parent else cited
        if self.exists(cand) and cand not in out:
            out.append(cand)
        # suffix match: ``src/appendix.tex`` -> ``<cache>/<id>/src/appendix.tex``
        base = cited.rsplit("/", 1)[-1]
        for rel in self.by_basename.get(base, ()):
            if rel.endswith("/" + cited) and rel not in out:
                out.append(rel)
        return out


# --------------------------------------------------------------------------
# PIN INDEX (sha256-pinned primary-source artefacts cannot drift)
# --------------------------------------------------------------------------
PIN_RECORD_GLOBS = (
    "molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json",
    "molmetal/reports/wfa175_frontier_comparability/sources.json",
    "molmetal/reports/wfa175_frontier_comparability/band_census.json",
)


def build_pin_index(root: Path) -> dict[str, set[str]]:
    """Map basename -> {sha256} recorded by a pin record on disk."""
    index: dict[str, set[str]] = {}

    def add(name: str, sha: str):
        if not sha or len(sha) != 64:
            return
        index.setdefault(name.rsplit("/", 1)[-1], set()).add(sha.lower())

    for rel in PIN_RECORD_GLOBS:
        p = root / rel
        if not p.is_file():
            continue
        try:
            with _open_read(p) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        for s in data.get("retrieval", []) or []:
            for f in s.get("files", []) or []:
                add(f.get("path", ""), f.get("sha256", ""))
        for s in data.get("sources", []) or []:
            if isinstance(s, dict):
                for f in s.get("files", []) or []:
                    if isinstance(f, dict):
                        add(f.get("path", ""), f.get("sha256", ""))
                add("", s.get("eprint_sha256", ""))
    return index


def build_pinned_source_files(root: Path) -> set[str]:
    """The file names INSIDE the sha256-pinned primary-source e-prints.

    A locator into one of these files points into an IMMUTABLE, pinned
    blob: the tarball's digest is recorded in the pin record, so the file
    inside it -- and therefore its LINE NUMBERS -- cannot drift.  Such a
    locator is SNAPSHOT, not drift.  Without this the auditor resolved the
    frontier table's ``main.tex:929`` to the repo's own ``paper/main.tex``
    and called it a defect, which it is not.
    """
    names: set[str] = set()

    for rel in PIN_RECORD_GLOBS:
        p = root / rel
        if not p.is_file():
            continue
        try:
            with _open_read(p) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for s in data.get("retrieval", []) or []:
            if not isinstance(s, dict):
                continue
            for f in s.get("files", []) or []:
                if isinstance(f, dict) and f.get("path"):
                    names.add(str(f["path"]).rsplit("/", 1)[-1])

    # the cached e-print records name the files inside each tarball
    try:
        cache_records = sorted(root.glob("molmetal/reports/*/cache/*.json"))
    except OSError:
        cache_records = []
    for p in cache_records:
        try:
            with _open_read(p) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for n in data.get("file_names", []) or []:
            names.add(str(n).rsplit("/", 1)[-1])
    return names


# --------------------------------------------------------------------------
# THE AUDIT
# --------------------------------------------------------------------------
CLASSES = (
    "RESOLVES-AND-MATCHES",
    "RESOLVES-BUT-DIFFERS",
    "RESOLVES-NO-CLAIM",
    "TARGET-ABSENT",
    "TARGET-AMBIGUOUS",
    "TARGET-IS-SELF",
    "DATED-SUPERSESSION",
    "SNAPSHOT",
)

#: more than this many candidate files for one cited path -> the citation is
#: a BARE FILENAME and does not identify one artefact.  One, not three: a
#: citation is machine-verifiable only when it names exactly ONE file.
MAX_UNAMBIGUOUS_CANDIDATES = 1


def _claim_candidates(citing_rel: str, lines: list[str], idx: int, start: int,
                      end: int) -> list[tuple[str, str]]:
    """Candidate claim texts attached to a locator occurrence.

    Returns a list of ``(kind, text)`` where ``kind`` is ``"quote"`` (the
    citation asserts the cited line CONTAINS this text -- machine
    verifiable) or ``"pointer"`` (the locator sits inside a larger sentence
    that is about something else -- the content is not machine-verifiable).
    """
    cands: list[tuple[str, str]] = []
    same = lines[idx - 1] if 0 < idx <= len(lines) else ""
    spans = _all_quote_spans(same)

    enclosing = [s for s in spans if s[0] < start and s[1] > end]
    after_spans = [s for s in spans if s[0] >= end]

    # (1) the locator sits INSIDE a quoted phrase -> the quote is about
    #     something else and the locator is a sub-reference inside it.  The
    #     quoted text is NOT a machine-verifiable claim about the cited
    #     line, so the occurrence is a POINTER.  (This is the shape of a
    #     QUOTED OLD TITLE -- "'THE ONLY MIS-ATTRIBUTION IS report.md:139'
    #     IS INCOMPLETE" -- and of the honest-negative catalog's long
    #     concatenated status lines, where a locator deep inside the line
    #     is followed by an unrelated quote thousands of characters later.)
    if enclosing:
        cands.append(("pointer", same[end : enclosing[0][1]]))
        return cands

    # (2) a quoted claim that BEGINS at/after the locator -- the citation
    #     asserts the cited line contains it.  It must be ADJACENT: a
    #     quote that starts more than CLAIM_WINDOW characters after the
    #     locator is not the locator's claim.
    for s in after_spans:
        if s[0] - end > CLAIM_WINDOW:
            break
        cands.append(("quote", s[2]))

    # (3) a markdown block quote on one of the following lines
    for nxt in range(idx + 1, min(idx + 4, len(lines) + 1)):
        m = BLOCKQUOTE_RE.match(lines[nxt - 1])
        if m:
            cands.append(("quote", m.group(1)))
            break
        if lines[nxt - 1].strip():
            break

    # (4) the remainder of the line is a POINTER, not a verifiable claim.
    rest = same[end:]
    if len(rest.strip()) > 12:
        cands.append(("pointer", rest))
    return cands


def audit(
    root: Path,
    restrict: list[str] | None = None,
    max_bytes: int = MAX_FILE_BYTES,
    gitignored_under: list[str] | None = None,
) -> dict:
    """Run the audit.  Returns the machine-readable summary."""
    t0 = time.time()
    # the census may read the repo's GITIGNORED report artefacts under
    # explicitly named prefixes; ``--paths`` restricts the SCANNING, this
    # only widens the CENSUS.
    ignored_prefixes = list(gitignored_under or []) + list(restrict or [])
    all_files, census_method = census_files(root, ignored_under=ignored_prefixes)
    if restrict:
        keep = set()
        for r in restrict:
            rp = r.rstrip("/")
            keep.update(
                f for f in all_files if f == rp or f.startswith(rp + "/")
            )
        files = sorted(keep)
    else:
        files = all_files
    # The RESOLVER sees the WHOLE census, not only the restricted part: a
    # citation from an audited tree into a file OUTSIDE that tree must
    # still resolve, or the audit reports "target absent" for a file that is
    # right there on disk.  Only the SCANNING is restricted.
    resolver = Resolver(root, all_files)
    pin_index = build_pin_index(root)
    pinned_source_files = build_pinned_source_files(root)

    hits: list[dict] = []
    scanned = 0
    skipped_large = 0
    skipped_binary = 0

    for rel in files:
        if not _is_text_candidate(rel):
            continue
        try:
            size = os.path.getsize(root / rel)
        except OSError:
            continue
        if size > max_bytes:
            skipped_large += 1
            continue
        try:
            with _open_read(root / rel) as fh:
                text = fh.read()
        except OSError:
            skipped_binary += 1
            continue
        if "\x00" in text[:4096]:
            skipped_binary += 1
            continue
        scanned += 1
        lines = text.splitlines()

        # position classification for this citing file
        dated_lines: set[int] = set()
        json_dated: set[str] = set()
        json_leaves: dict[str, str] = {}
        if rel.endswith(".md"):
            dated_lines = _md_dated_lines(text)
        elif rel.endswith(".py"):
            dated_lines = _py_dated_lines(lines)
        elif rel.endswith(".json"):
            try:
                data = json.loads(text)
            except ValueError:
                data = None
            if data is not None:
                json_dated = _json_dated_paths(data)
                json_leaves = _json_leaves(data)
                json_spans = _json_leaf_spans(text, data)

        for m in LOCATOR_RE.finditer(text):
            cited = m.group("path")
            lineno = int(m.group("line"))
            lineno2 = int(m.group("line2") or lineno)
            if lineno < 1 or lineno2 < lineno:
                continue
            line_no = text.count("\n", 0, m.start()) + 1

            hit = {
                "citing_file": rel,
                "citing_line": line_no,
                "cited_path": cited,
                "cited_line": lineno,
                "cited_line_end": lineno2,
                "cited_path_is_self": False,
                "resolved_to": [],
                "matched_probe": None,
                "claim_used": None,
                "cited_line_text": None,
            }

            # --- resolution -------------------------------------------------
            cands = resolver.candidates(cited, rel)
            hit["resolved_to"] = cands[:8]

            if cited.replace("\\", "/").lstrip("./") == rel.replace("\\", "/"):
                hit["cited_path_is_self"] = True
            elif rel in cands:
                # the cited path RESOLVED to the citing file itself (e.g. a
                # bare ``report.md`` inside its own tree)
                hit["cited_path_is_self"] = True

            # --- claim ------------------------------------------------------
            # the locator offsets are ABSOLUTE in ``text``; the claim lives on
            # the LINE, so convert to line-relative offsets.
            line_start = text.rfind("\n", 0, m.start()) + 1
            rel_start = m.start() - line_start
            rel_end = m.end() - line_start
            claims = _claim_candidates(rel, lines, line_no, rel_start, rel_end)
            probe = None
            claim_kind = None
            anchor = None
            for kind, c in claims:
                p = _probe(c)
                if p:
                    probe = p
                    anchor = _anchor(c)
                    claim_kind = kind
                    hit["claim_used"] = c[:200]
                    break
            hit["claim_kind"] = claim_kind

            # --- classify ---------------------------------------------------
            in_dated = line_no in dated_lines
            is_scan_record = False
            if rel.endswith(".json"):
                # the leaf path of this occurrence: the leaf whose own
                # CHARACTER SPAN contains the occurrence, so a DATED copy of
                # a locator is not attributed to the LIVE leaf that mentions
                # the same path.
                leaf = None
                for jp, span in json_spans.items():
                    if span[0] <= m.start() < span[1]:
                        leaf = jp
                        break
                if leaf is None:
                    for jp, lv in json_leaves.items():
                        if lv and cited in lv and rel not in lv:
                            leaf = jp
                            break
                if leaf:
                    if leaf in json_dated or any(
                        DATED_SEGMENT_RE.search(seg) for seg in re.split(r"[.\[]", leaf)
                    ):
                        in_dated = True
                    # a RECORDED SCAN RESULT is demoted by the LEAF'S OWN key
                    # only -- an ancestor key (e.g. ``band_census``) does not
                    # make every child a scan record.
                    leaf_key = re.split(r"[.\[]", leaf)[-1]
                    if SCAN_KEY_RE.search(leaf_key):
                        is_scan_record = True
            if rel.endswith(".py") and SCAN_LINE_RE.search(
                lines[line_no - 1] if 0 < line_no <= len(lines) else ""
            ):
                is_scan_record = True

            pinned_source = (
                not resolver.exists(cited)
                and cited.rsplit("/", 1)[-1] in pinned_source_files
            )
            if hit["cited_path_is_self"]:
                hit["classification"] = "TARGET-IS-SELF"
            elif in_dated:
                hit["classification"] = "DATED-SUPERSESSION"
            elif pinned_source:
                # the locator names a file INSIDE a sha256-pinned e-print
                # tarball: its line numbers are stable by construction.
                hit["classification"] = "SNAPSHOT"
                hit["pinned_source_file"] = cited.rsplit("/", 1)[-1]
            elif not cands:
                hit["classification"] = "TARGET-ABSENT"
            elif len(cands) > MAX_UNAMBIGUOUS_CANDIDATES:
                # a BARE FILENAME: it does not identify ONE artefact, so the
                # citation is not machine-verifiable against a single file.
                hit["classification"] = "TARGET-AMBIGUOUS"
                hit["n_candidates"] = len(cands)
            else:
                # pinned target?  -> cannot drift
                pinned = False
                for cand in cands:
                    sha = resolver.sha256(cand)
                    if sha and sha.lower() in pin_index.get(
                        cand.rsplit("/", 1)[-1], set()
                    ):
                        pinned = True
                        break
                snapshot_line = is_scan_record
                if pinned or snapshot_line:
                    hit["classification"] = "SNAPSHOT"
                elif probe is None:
                    hit["classification"] = "RESOLVES-NO-CLAIM"
                else:
                    matched = False
                    used_text = None
                    for cand in cands:
                        tgt = resolver.lines(cand)
                        for ln in range(lineno, min(lineno2, len(tgt)) + 1):
                            # The claim must START at the cited line (its
                            # anchor word is ON that line); the rest may WRAP
                            # onto the following lines.
                            anchor_here = _norm(tgt[ln - 1])
                            window = _norm(
                                " ".join(tgt[ln - 1 : min(len(tgt), ln + 2)])
                            )
                            ok = (
                                probe in window
                                if anchor is None
                                else (anchor in anchor_here and probe in window)
                            )
                            if ok:
                                matched = True
                                used_text = tgt[ln - 1][:300]
                                break
                        if matched:
                            break
                    if matched:
                        hit["classification"] = "RESOLVES-AND-MATCHES"
                        hit["matched_probe"] = probe
                        hit["cited_line_text"] = used_text
                    elif claim_kind == "pointer":
                        # the locator is a POINTER inside a larger sentence
                        # about something else: the content is not machine
                        # verifiable, so we do NOT assert drift.
                        hit["classification"] = "RESOLVES-NO-CLAIM"
                        hit["matched_probe"] = probe
                    else:
                        hit["classification"] = "RESOLVES-BUT-DIFFERS"
                        hit["matched_probe"] = probe
                        if cands:
                            tgt = resolver.lines(cands[0])
                            if 0 < lineno <= len(tgt):
                                hit["cited_line_text"] = tgt[lineno - 1][:300]
            hits.append(hit)

    elapsed = time.time() - t0
    counts = {c: 0 for c in CLASSES}
    for h in hits:
        counts[h["classification"]] = counts.get(h["classification"], 0) + 1

    live_drift = [
        h for h in hits
        if h["classification"] == "RESOLVES-BUT-DIFFERS"
    ]

    return {
        "artifact": "wfa200_crosswave_locator_audit",
        "auditor_version": __version__,
        "measured_on": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "census_method": census_method,
        "gitignored_under": sorted(ignored_prefixes),
        "scanning_restricted_to": sorted(restrict or []),
        "n_files_in_census": len(files),
        "n_files_scanned": scanned,
        "n_skipped_large": skipped_large,
        "n_skipped_binary": skipped_binary,
        "n_locator_hits": len(hits),
        "counts_by_class": counts,
        "n_live_drift": len(live_drift),
        "live_drift": live_drift,
        "hits": hits,
        "wall_seconds": round(elapsed, 3),
        "read_only": True,
        "gpu": "0 (CPU-only; no torch import, no CUDA call)",
    }


# --------------------------------------------------------------------------
# REPORT WRITING (the ONLY writing this module does, and only when asked)
# --------------------------------------------------------------------------
def _human_summary(rep: dict) -> str:
    """The human summary.

    It is written as markdown whose FIRST level-2 heading carries the run
    date, so that every line of the published record is a DATED position
    (AGENTS.md 5.5 rule #6).  That matters: this file quotes the very
    ``<path>:<line>`` locators it reports, so an UNDATED copy of it would be
    a live claim about stale locators -- i.e. the auditor would flag its own
    record as live drift.
    """
    lines = []
    lines.append("# Wf-A200 -- CROSS-WAVE LOCATOR AUDIT (read-only)")
    lines.append("")
    lines.append("## §A -- RECORD (%s) -- a DATED record of the locators it reports"
                 % rep["measured_on"][:10])
    lines.append("")
    lines.append("root            : %s" % rep["root"])
    lines.append("census method   : %s" % rep["census_method"])
    lines.append("files in census : %d" % rep["n_files_in_census"])
    lines.append("files scanned   : %d  (skipped large=%d binary=%d)" % (
        rep["n_files_scanned"], rep["n_skipped_large"], rep["n_skipped_binary"]))
    lines.append("locator hits    : %d" % rep["n_locator_hits"])
    lines.append("wall seconds    : %.3f" % rep["wall_seconds"])
    lines.append("")
    lines.append("### counts by class")
    lines.append("")
    for cls in CLASSES:
        lines.append("* `%s` : %d" % (cls, rep["counts_by_class"].get(cls, 0)))
    lines.append("")
    lines.append("### LIVE DRIFT (RESOLVES-BUT-DIFFERS) -- the actionable set: %d"
                 % rep["n_live_drift"])
    lines.append("")
    for h in rep["live_drift"][:200]:
        lines.append("* `%s` line %d -> `%s` line %d" % (
            h["citing_file"], h["citing_line"],
            h["cited_path"], h["cited_line"]))
        if h.get("cited_line_text"):
            lines.append("    * the cited line now reads: %s"
                         % h["cited_line_text"].strip()[:150])
        if h.get("matched_probe"):
            lines.append("    * the claim was: %s" % h["matched_probe"][:150])
    lines.append("")
    lines.append("## §B -- HOW TO READ THIS RECORD (its classes, its limits, its errors)")
    lines.append("")
    lines.append("### §B.1 the classes")
    lines.append("")
    lines.append("| class | meaning | actionable? |")
    lines.append("|---|---|---|")
    lines.append("| `RESOLVES-BUT-DIFFERS` | the cited path resolves to ONE file, the "
                 "cited line exists, and the quoted claim next to the locator is NOT on "
                 "that line | **YES -- this is the defect class** |")
    lines.append("| `RESOLVES-AND-MATCHES` | the cited line does carry the claim | no |")
    lines.append("| `RESOLVES-NO-CLAIM` | the locator resolves, but the text next to it "
                 "is a POINTER (a sub-reference inside a longer sentence), so the "
                 "content is not machine-verifiable | no -- reported, not asserted |")
    lines.append("| `DATED-SUPERSESSION` | the occurrence sits inside a DATED section, "
                 "key or constant -- a supersession record that AGENTS.md 5.5 rule #6 "
                 "requires to survive | no |")
    lines.append("| `SNAPSHOT` | the cited file is pinned by sha256, or is a file "
                 "INSIDE a sha256-pinned e-print tarball, so its line numbers cannot "
                 "drift | no |")
    lines.append("| `TARGET-ABSENT` | the cited path names no file in the census | "
                 "usually a stale path, but often just a foreign path |")
    lines.append("| `TARGET-AMBIGUOUS` | the cited path matches SEVERAL files, so it "
                 "does not identify one artefact and cannot be machine-verified | no |")
    lines.append("| `TARGET-IS-SELF` | the cited path resolves to the citing file "
                 "itself | no |")
    lines.append("")
    lines.append("### §B.2 what the CLAIM next to a locator is, and how it is chosen")
    lines.append("")
    lines.append("A locator asserts drift only when the text next to it is a "
                 "MACHINE-VERIFIABLE claim about the cited line.  The claim is:")
    lines.append("")
    lines.append("1. a quoted string that starts within %d characters AFTER the "
                 "locator; or" % CLAIM_WINDOW)
    lines.append("2. a markdown block quote on one of the next three lines; or")
    lines.append("3. otherwise a POINTER -- reported, never asserted as drift.")
    lines.append("")
    lines.append("A locator that sits INSIDE a quoted phrase is always a POINTER: the "
                 "quote is about something else and the locator is a sub-reference "
                 "inside it -- the shape of a QUOTED OLD TITLE such as "
                "\"'THE ONLY MIS-ATTRIBUTION IS report.md:139' IS INCOMPLETE\".")
    lines.append("")
    lines.append("### §B.3 the KNOWN error modes -- read the findings with these in hand")
    lines.append("")
    lines.append("* **FALSE POSITIVE (the likely one).**  The claim is taken from the "
                 "text NEXT TO the locator, not from the sentence that carries the "
                 "citation's subject.  A line that packs several citations and several "
                 "quotes together can attach the wrong quote to a locator.  Every "
                 "finding in this record prints both \"the cited line now reads\" and "
                 "\"the claim was\" -- READ BOTH before acting on a finding.")
    lines.append("* **FALSE NEGATIVE.**  A locator whose claim is a paraphrase, not a "
                 "quotation, is a POINTER and is never asserted as drift.  The drift "
                 "count is therefore a LOWER bound on the defect class.")
    lines.append("* **UNSCANNED.**  This run's census method: `%s`.  It read %d of %d "
                 "files in the census; %d were skipped as binary and %d as too large.  "
                 "The census is `git ls-files` + untracked-not-ignored, so the repo's "
                 "gitignored files (e.g. `molmetal/reports/**/*.json`) are NOT read "
                 "unless the run is restricted to a prefix -- a restricted run adds the "
                 "gitignored files UNDER that prefix."
                 % (rep["census_method"], rep["n_files_scanned"],
                    rep["n_files_in_census"], rep["n_skipped_binary"],
                    rep["n_skipped_large"]))
    lines.append("* **NOT A GATE.**  This auditor is READ-ONLY.  It is not wired into "
                 "`pre_flight_check.py`, it adds no gate, and it does not change "
                 "`EXPECTED_GATE_COUNTS`.")
    lines.append("")
    lines.append("read-only: %s | gpu: %s" % (rep["read_only"], rep["gpu"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None,
                    help="repo root (default: the repo this script lives in)")
    ap.add_argument("--paths", nargs="*", default=None,
                    help="restrict the SCANNING to these repo-relative prefixes")
    ap.add_argument("--gitignored-under", nargs="*", default=None,
                    help="also read the GITIGNORED files under these prefixes "
                         "(e.g. molmetal/reports/**/*.json, which .gitignore:158 "
                         "hides); widens the CENSUS, does not restrict the scan")
    ap.add_argument("--json-out", default=None, help="write the JSON summary here")
    ap.add_argument("--md-out", default=None, help="write the human summary here")
    ap.add_argument("--print-json", action="store_true",
                    help="print the JSON summary to stdout")
    ap.add_argument("--max-bytes", type=int, default=MAX_FILE_BYTES)
    args = ap.parse_args(argv)

    if args.root:
        root = Path(args.root).resolve()
    else:
        root = Path(__file__).resolve().parents[2]

    rep = audit(
        root,
        restrict=args.paths,
        max_bytes=args.max_bytes,
        gitignored_under=args.gitignored_under,
    )
    human = _human_summary(rep)

    if args.print_json:
        print(json.dumps(rep, indent=1, sort_keys=True))
    else:
        print(human, end="")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # The PUBLISHED record is wrapped in a DATED envelope key, so every
        # leaf of it is a dated position: the record quotes the very
        # ``<path>:<line>`` locators it reports, and an undated copy of it
        # would be a live claim about stale locators.
        envelope = {"wfa200_crosswave_locator_audit_%s" % time.strftime("%Y_%m_%d"): rep}
        atomic_write_file(
            out, json.dumps(envelope, indent=1, sort_keys=True) + "\n"
        )
    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_file(out, human)
    return 0


if __name__ == "__main__":
    sys.exit(main())
