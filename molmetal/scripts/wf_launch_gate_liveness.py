#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Wf-A269 wave-11 — the LIVENESS GUARD for launch-wrapper pre-launch gates.

WHAT THIS IS
------------
A **dead gate** is a check that LOOKS like a check but CANNOT FAIL.  The
repo's own pre-flight advertises "129 gates" (131 on the launch pipeline),
but a gate that cannot fail contributes exactly as much to that count as a
real one: it prints PASS and the reader believes a measurement happened.

Four measured sub-modes are detected here:

  (i)   the wrapper passes a flag the tool does not define (argparse exits 2
        instantly, so the gate is decorative);
  (ii)  the wrapper greps for a marker that has no producer;
  (iii) the wrapper keys a gate on a tool exit code that is 0 BY
        CONSTRUCTION;
  (iv)  the wrapper redirects stdout+stderr into a ``.json`` path (so the
        artefact is not JSON at all), or it passes a BARE ``--json`` to a
        tool whose ``--json`` is ``type=Path, nargs="?"`` (a bare ``--json``
        writes NO file and the payload goes to stdout).

WHY IT IS STATIC
----------------
The guard **executes nothing**.  It does not run a wrapper, does not run
``--help`` on any tool (that would import torch), and never allocates a GPU.
Flag validity is decided by the tool's own **argparse AST** — the same
technique the repo already uses in
``molmetal/scripts/pre_flight_check.py::_trainer_argparse_dests``
(the trainer is NOT imported or executed).

WHY IT ABSTAINS
---------------
The oracle is never guessed.  When the guard cannot decide a case it
reports it as UNRESOLVED, and the report prints its own coverage bound
(what it CANNOT see).  A census that does not state its denominator is a
defect (AGENTS.md §6.269.6).

EXIT CODES (AGENTS.md §4 invariant)
-----------------------------------
  0 = PASS   — no unallowlisted finding and every mode was evaluated
  1 = FAIL   — at least one unallowlisted finding
  2 = SKIP   — the guard could not produce a complete verdict (unreadable
                corpus, or the mode-(iii) exit-code registry is missing,
                which leaves that mode UNRESOLVED)

EXIT 1 DOMINATES EXIT 2.  A finding is always checked before the
"could not complete" conditions, so a guard that finds a dead gate exits 1
even when some mode was abstained on.  This guard must not become a dead
gate itself (the class it hunts): if it could only ever exit 2, its FAIL
would be unreachable and the repo would learn to ignore it.  The
precedence is MEASURED, not asserted — see the T-A / T-B transcript in the
acceptance run.

The shared verdict consumer for launch-gate REPORTS is
``molmetal/scripts/wf_launch_gate_verdict.py``; this guard deliberately
does NOT re-implement report parsing — it names that consumer in its
findings instead of guessing a verdict from a report's bytes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

TOOL_NAME = "wf_launch_gate_liveness.py"
SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ALLOWLIST = SCRIPT_DIR / "wf_launch_gate_liveness_allow.json"
DEFAULT_REGISTRY = SCRIPT_DIR / "wf_launch_gate_exitcode_registry.json"

MODES = ("i", "ii", "iii", "iv")
STALE_ALLOWLIST_DAYS = 30

# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------

# A PRODUCER is a script that runs.  Only code files can produce a marker.
PRODUCER_SUFFIXES = {".py", ".sh"}

# These trees are OUTPUTS or PROSE, not producers: a marker that only shows
# up in a test assertion, in a report artefact or in a document was never
# emitted by anything.
PRODUCER_EXCLUDE_PREFIXES = (
    "molmetal/tests/",
    "molmetal/reports/",
    "metrics/",
    "results/",
    "docs/",
    "TODO/",
    "paper/",
    ".venv/",
    ".git/",
)

MAX_PRODUCER_BYTES = 8 * 1024 * 1024

# Tokens that make an occurrence a SEARCH SITE: the file talks ABOUT the
# marker (a grep, an assertion, a docstring quoting a grep) instead of
# PRODUCING it.
SEARCH_TOKENS = (
    "grep",
    "assert",
    "not in",
    ".find(",
    "count(",
    "in text",
    "in code",
    "search(",
    "re.compile",
)

# A marker that contains these is a REGEX, not a literal.
REGEX_METACHARS = set(".*+?[](){}|^$\\")

# ---------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    mode: str
    kind: str
    file: str
    line: int
    detail: str
    population: str = "primary"

    def key(self) -> str:
        return f"{self.file}:{self.line}:{self.mode}"

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
            "detail": self.detail,
            "population": self.population,
        }


@dataclass
class Unresolved:
    mode: str
    kind: str
    file: str
    line: int
    detail: str
    population: str = "primary"

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
            "detail": self.detail,
            "population": self.population,
        }


@dataclass
class Counts:
    i: int = 0
    ii: int = 0
    iii: int = 0
    iv: int = 0

    def bump(self, mode: str) -> None:
        if mode == "i":
            self.i += 1
        elif mode == "ii":
            self.ii += 1
        elif mode == "iii":
            self.iii += 1
        elif mode == "iv":
            self.iv += 1

    def total(self) -> int:
        return self.i + self.ii + self.iii + self.iv

    def as_dict(self) -> dict:
        return {"i": self.i, "ii": self.ii, "iii": self.iii, "iv": self.iv}


# ---------------------------------------------------------------------------
# shell lexical pass
# ---------------------------------------------------------------------------

_HEREDOC_RE = re.compile(
    r"<<(?P<dash>-?)(?P<space>\s*)(?P<q>['\"]?)(?P<delim>[A-Za-z_][A-Za-z0-9_]*)(?P=q)"
)


def _heredoc_delims(line: str) -> List[Tuple[str, bool]]:
    """Heredoc delimiters opened by this line: [(delimiter, allow_tabs)]."""
    out: List[Tuple[str, bool]] = []
    for m in _HEREDOC_RE.finditer(line):
        delim = m.group("delim")
        quoted = bool(m.group("q"))
        if not quoted and not re.fullmatch(r"[A-Z0-9_]+", delim):
            # `x << y` (arithmetic shift) is not a heredoc.
            continue
        out.append((delim, m.group("dash") == "-"))
    return out


def logical_lines(text: str) -> List[Tuple[int, str]]:
    """Strip full-line comments and heredoc bodies; join backslash
    continuations.  Returns [(first_physical_line_no, logical_line)].

    Full-line comments are stripped because the wrappers preserve the
    superseded rendering VERBATIM (AGENTS.md §5.5 rule #6) — a dead flag
    string survives inside a comment, and a substring test on raw text is
    satisfied by that comment.
    """
    raw_lines = text.splitlines()
    kept: List[Tuple[int, str]] = []
    pending_heredocs: List[Tuple[str, bool]] = []
    idx = 0
    while idx < len(raw_lines):
        raw = raw_lines[idx]
        if pending_heredocs:
            delim, allow_tabs = pending_heredocs[-1]
            probe = raw.lstrip("\t") if allow_tabs else raw
            if probe.rstrip() == delim:
                pending_heredocs.pop()
            idx += 1
            continue
        if raw.strip().startswith("#"):
            idx += 1
            continue
        kept.append((idx + 1, raw))
        pending_heredocs.extend(_heredoc_delims(raw))
        idx += 1

    joined: List[Tuple[int, str]] = []
    buf: Optional[str] = None
    start = 0
    for lineno, raw in kept:
        if buf is None:
            start = lineno
            buf = raw.rstrip()
        else:
            buf = buf + " " + raw.strip()
        if buf.endswith("\\"):
            buf = buf[:-1].rstrip()
            continue
        joined.append((start, buf))
        buf = None
    if buf is not None:
        joined.append((start, buf))
    return joined


@dataclass
class Segment:
    text: str
    sep_before: str
    sep_after: str


def split_commands(text: str) -> List[Segment]:
    """Split one logical line into individual commands.

    Separators: ``;`` ``&&`` ``||`` ``|`` ``&`` ``|&``.  NOT separators:
    ``2>&1`` / ``&>`` / ``>&`` — splitting on the ``&`` inside ``2>&1``
    produced ZERO mode-(iv) hits in the research prototype.
    """
    segs: List[Segment] = []
    buf: List[str] = []
    sep_before = ""
    i = 0
    n = len(text)
    quote = ""
    while i < n:
        ch = text[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(text[i + 1])
            i += 2
            continue
        if ch == ";":
            segs.append(Segment("".join(buf), sep_before, ";"))
            buf = []
            sep_before = ";"
            i += 1
            continue
        if ch == "|":
            if i + 1 < n and text[i + 1] == "|":
                segs.append(Segment("".join(buf), sep_before, "||"))
                buf = []
                sep_before = "||"
                i += 2
                continue
            if i + 1 < n and text[i + 1] == "&":
                segs.append(Segment("".join(buf), sep_before, "|&"))
                buf = []
                sep_before = "|&"
                i += 2
                continue
            segs.append(Segment("".join(buf), sep_before, "|"))
            buf = []
            sep_before = "|"
            i += 1
            continue
        if ch == "&":
            if i + 1 < n and text[i + 1] == ">":
                buf.append("&>")
                i += 2
                continue
            if buf and buf[-1] == ">":
                buf.append("&")
                i += 1
                continue
            if i + 1 < n and text[i + 1] == "&":
                segs.append(Segment("".join(buf), sep_before, "&&"))
                buf = []
                sep_before = "&&"
                i += 2
                continue
            segs.append(Segment("".join(buf), sep_before, "&"))
            buf = []
            sep_before = "&"
            i += 1
            continue
        buf.append(ch)
        i += 1
    segs.append(Segment("".join(buf), sep_before, ""))
    return [s for s in segs if s.text.strip()]


@dataclass
class Tok:
    text: str
    quoted: bool


def tokenize(text: str) -> List[Tok]:
    """Whitespace tokenisation that respects shell quoting.

    A token that is entirely inside one quoted string is ONE token: that
    is what keeps ``printf ' ... --device/--cuda ...'`` from being read as
    a flag list.

    The quote CHARACTERS are removed (that is what the shell does), so
    ``PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"`` yields the token
    ``PRE_FLIGHT=${SCRIPTS_DIR}/pre_flight_check.py`` and not a value with
    a stray leading double quote.
    """
    toks: List[Tok] = []
    buf: List[str] = []
    quoted = False
    started_quoted = False
    quote = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            if not buf:
                started_quoted = True
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(text[i + 1])
            i += 2
            continue
        if ch.isspace():
            if buf:
                toks.append(Tok("".join(buf), started_quoted))
                buf = []
                started_quoted = False
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        toks.append(Tok("".join(buf), started_quoted))
    return toks


# ---------------------------------------------------------------------------
# shell variable resolution
# ---------------------------------------------------------------------------

_VAR_BRACE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}")
_VAR_PLAIN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_ASSIGN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def expand(value: str, variables: Dict[str, str]) -> str:
    """Best-effort expansion of ``$VAR`` / ``${VAR}``.

    Unknown variables are left verbatim, so an unresolved reference keeps
    the token UNRESOLVED instead of silently becoming a wrong path.
    """
    out = value
    for _ in range(8):
        pieces: List[str] = []
        pos = 0
        changed = False
        for m in _VAR_BRACE.finditer(out):
            name = m.group(1)
            rest = m.group(2)
            replacement: Optional[str] = None
            if name in variables:
                candidate = variables[name]
                if rest.startswith(":-") and ("${" + name) in candidate:
                    # self-reference: ``PY="${PY:-/abs/path}"`` — the
                    # default is the real value, the assignment is a no-op
                    replacement = rest[2:]
                else:
                    replacement = candidate
            elif rest.startswith(":-"):
                replacement = rest[2:]
            elif rest.startswith("##"):
                base = variables.get(name, "")
                if base:
                    replacement = base.rsplit("/", 1)[-1]
            if replacement is None:
                continue
            pieces.append(out[pos : m.start()])
            pieces.append(replacement)
            pos = m.end()
            changed = True
        pieces.append(out[pos:])
        new = "".join(pieces)
        new = _VAR_PLAIN.sub(lambda m: variables.get(m.group(1), m.group(0)), new)
        if not changed and new == out:
            break
        out = new
    return out


def collect_variables(logical: Sequence[Tuple[int, str]]) -> Dict[str, str]:
    """Collect shell variable assignments, last one wins."""
    variables: Dict[str, str] = {}
    for _lineno, line in logical:
        for seg in split_commands(line):
            toks = tokenize(seg.text)
            for tok in toks:
                if tok.text.startswith("-"):
                    break
                if "=" not in tok.text:
                    break
                name, value = tok.text.split("=", 1)
                if not _ASSIGN_NAME.match(name):
                    break
                variables[name] = value
    return variables


_SHELL_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "while", "until", "do", "done",
    "case", "esac", "!", "{", "}", "(", ")", "time", "command", "exec",
    "nohup", "setsid", "builtin", "eval", "source", ".", "select",
}


def command_word(toks: Sequence[Tok]) -> Tuple[int, str]:
    """The command word of a segment: the first token that is not a shell
    keyword and not a leading ``VAR=value`` environment assignment.
    """
    i = 0
    while i < len(toks):
        text = toks[i].text
        if text in _SHELL_KEYWORDS:
            i += 1
            continue
        if "=" in text and _ASSIGN_NAME.match(text.split("=", 1)[0]):
            i += 1
            continue
        break
    return i, (toks[i].text if i < len(toks) else "")


def shell_functions(text: str) -> Set[str]:
    """Names of the shell functions this wrapper defines."""
    out: Set[str] = set()
    for raw in text.splitlines():
        m = re.match(r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)\s*\{", raw)
        if m:
            out.add(m.group(1))
    return out


def resolve_repo_file(value: str) -> Optional[Path]:
    """Map a resolved token to an existing repo file, or None."""
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# tool AST scan (the mode-(i) oracle)
# ---------------------------------------------------------------------------


@dataclass
class FlagSpec:
    flag: str
    dest: str
    required: bool
    takes_value: bool
    optional_value: bool
    path_typed: bool
    choices: Optional[Set[str]]
    line: int


@dataclass
class ToolSpec:
    path: Path
    flags: Dict[str, FlagSpec] = field(default_factory=dict)
    scan_note: str = ""
    has_argparse: bool = True


def _module_string_constants(tree: ast.AST) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    targets: List[Tuple[str, ast.AST]] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                targets.append((node.targets[0].id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                targets.append((node.target.id, node.value))
    for name, value in targets:
        if not isinstance(value, (ast.Tuple, ast.List)):
            continue
        items: List[str] = []
        ok = True
        for el in value.elts:
            if isinstance(el, ast.Constant) and isinstance(el.value, str):
                items.append(el.value)
            else:
                ok = False
                break
        if ok:
            out[name] = items
    return out


def scan_tool(path: Path) -> ToolSpec:
    """Build the flag table of a tool from its SOURCE AST.

    The tool is never imported and never executed.  This mirrors
    ``pre_flight_check.py::_trainer_argparse_dests``.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolSpec(path, {}, f"unreadable: {exc}", has_argparse=False)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ToolSpec(path, {}, f"AST parse failed: {exc}", has_argparse=False)

    module_consts = _module_string_constants(tree)
    parse_vars: Set[str] = set()
    add_calls: List[Tuple[str, ast.Call]] = []
    has_argparse = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "argparse":
                    has_argparse = True
        if isinstance(node, ast.ImportFrom) and node.module == "argparse":
            has_argparse = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            recv = node.func.value.id if isinstance(node.func.value, ast.Name) else "?"
            if node.func.attr == "parse_args":
                parse_vars.add(recv)
            elif node.func.attr == "add_argument":
                add_calls.append((recv, node))
    if not has_argparse:
        return ToolSpec(path, {}, "no argparse import: the oracle is UNKNOWN", has_argparse=False)

    no_help: Dict[str, bool] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "ArgumentParser"
        ):
            recv = node.func.value.id if isinstance(node.func.value, ast.Name) else "?"
            flag = False
            for k in node.keywords:
                if k.arg == "add_help" and isinstance(k.value, ast.Constant):
                    flag = k.value.value is False
            no_help[recv] = flag

    note = ""
    if len(parse_vars) == 1:
        only = next(iter(parse_vars))
        add_calls = [(r, c) for (r, c) in add_calls if r == only]
    elif len(parse_vars) > 1:
        note = (
            f"{len(parse_vars)} parser variables in the tool "
            f"({sorted(parse_vars)}); the flag table is the union of all of them"
        )

    flags: Dict[str, FlagSpec] = {}
    for recv, call in add_calls:
        names: List[str] = []
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("-"):
                    names.append(arg.value)
        if not names:
            continue
        kw = {k.arg: k.value for k in call.keywords}

        def _const(key: str) -> Optional[object]:
            node = kw.get(key)
            if isinstance(node, ast.Constant):
                return node.value
            return None

        dest = _const("dest")
        required = bool(_const("required") is True)
        action = _const("action")
        nargs = _const("nargs")
        type_node = kw.get("type")
        type_name = ""
        if isinstance(type_node, ast.Name):
            type_name = type_node.id
        elif isinstance(type_node, ast.Attribute):
            type_name = type_node.attr

        takes_value = True
        if action in ("store_true", "store_false", "count", "store_const", "append_const"):
            takes_value = False
        if nargs == "0":
            takes_value = False
        optional_value = nargs == "?"

        choices: Optional[Set[str]] = None
        cnode = kw.get("choices")
        if isinstance(cnode, (ast.List, ast.Tuple)):
            vals = [
                el.value
                for el in cnode.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            ]
            if len(vals) == len(cnode.elts):
                choices = set(vals)
        elif isinstance(cnode, ast.Name) and cnode.id in module_consts:
            choices = set(module_consts[cnode.id])

        for name in names:
            flag_dest = dest if isinstance(dest, str) else name.lstrip("-").replace("-", "_")
            flags[name] = FlagSpec(
                flag=name,
                dest=str(flag_dest),
                required=required,
                takes_value=takes_value,
                optional_value=optional_value,
                path_typed=(type_name == "Path"),
                choices=choices,
                line=getattr(call, "lineno", 0),
            )
        if not no_help.get(recv, False):
            for implicit in ("--help", "-h"):
                flags.setdefault(
                    implicit,
                    FlagSpec(
                        flag=implicit,
                        dest="help",
                        required=False,
                        takes_value=False,
                        optional_value=False,
                        path_typed=False,
                        choices=None,
                        line=0,
                    ),
                )
    if not flags:
        note = (note + "; " if note else "") + "no literal add_argument found: oracle is UNKNOWN"
    return ToolSpec(path, flags, note, has_argparse=True)


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def load_corpus(repo_root: Path) -> List[str]:
    """Repo file list: tracked + untracked (no os.walk — 201 GB repo)."""
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {proc.stderr.strip()[:200]}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


class ProducerIndex:
    """Repo-wide search for a marker's producer."""

    def __init__(self, repo_root: Path, corpus: Sequence[str]) -> None:
        self.repo_root = repo_root
        self.files: List[str] = []
        self._cache: Dict[str, str] = {}
        for rel in corpus:
            if rel.startswith(PRODUCER_EXCLUDE_PREFIXES):
                continue
            if Path(rel).suffix not in PRODUCER_SUFFIXES:
                continue
            self.files.append(rel)

    def text(self, rel: str) -> Optional[str]:
        if rel in self._cache:
            return self._cache[rel]
        path = self.repo_root / rel
        try:
            if path.stat().st_size > MAX_PRODUCER_BYTES:
                self._cache[rel] = ""
                return ""
            data = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            data = ""
        self._cache[rel] = data
        return data

    def contains(self, rel: str, needle: str) -> bool:
        data = self.text(rel)
        return bool(data) and needle in data

    def is_search_site(self, rel: str, needle: str) -> bool:
        """True when EVERY occurrence of the needle sits in a SEARCH context.

        A file that greps for a marker (or asserts on it) is not its
        producer.  A file that both greps for a marker AND prints it is a
        producer — so the test is "all hits are search sites", not "any hit
        is a search site".
        """
        data = self.text(rel)
        if not data:
            return False
        lines = data.splitlines()
        hits = 0
        search_hits = 0
        for i, line in enumerate(lines):
            if needle not in line:
                continue
            hits += 1
            window = "\n".join(lines[max(0, i - 3) : i + 4])
            if any(tok in window for tok in SEARCH_TOKENS):
                search_hits += 1
        return hits > 0 and search_hits == hits

    def producers(self, needle: str, exclude: Sequence[str] = ()) -> List[str]:
        out: List[str] = []
        excl = set(exclude)
        for rel in self.files:
            if rel in excl:
                continue
            if not self.contains(rel, needle):
                continue
            if self.is_search_site(rel, needle):
                continue
            out.append(rel)
        return out


# ---------------------------------------------------------------------------
# redirect / grep extraction
# ---------------------------------------------------------------------------

_REDIRECT_RE = re.compile(
    r"(?<![0-9A-Za-z_])>{1,2}\s*(?P<q>[\"']?)(?P<target>[^\"'\s;&|<>()]+)(?P=q)"
)
_DUP_REDIRECT_RE = re.compile(r"2>&1|&>|>&")


def file_redirects(segment_text: str) -> List[str]:
    """Redirect targets in a command (dup-redirects like 2>&1 excluded)."""
    masked = _DUP_REDIRECT_RE.sub(" ", segment_text)
    return [m.group("target") for m in _REDIRECT_RE.finditer(masked)]


def has_dup_redirect(segment_text: str) -> bool:
    return bool(_DUP_REDIRECT_RE.search(segment_text))


def basename_of(token: str) -> str:
    return token.rsplit("/", 1)[-1]


_DYNAMIC_TOKEN_RE = re.compile(
    r"\$\{[^}]*\[@\]|\$\{?[@*]|\$\(|`|\beval\b"
)
_NEGATED_IF_RE = re.compile(r"^\s*if\s+!")
_STATUS_CAPTURE_RE = re.compile(r"(?<![\w.$\{])([A-Za-z_][A-Za-z0-9_]*)=\$\?")
_ELSE_RE = re.compile(r"(^|[\s;{}])else(\s|$)")

_REDIRECT_TOKEN_RE = re.compile(r"\d*>&?\d*")


def redirect_token_indices(toks: Sequence[Tok]) -> Set[int]:
    """Indices of the redirect operators and their targets.

    The shell removes a redirection operator from the argv, so the token
    after ``--json`` in ``--json > file.json`` is NOT the flag's value:
    argparse sees a BARE ``--json``.  Counting the redirect target as a
    value would hide exactly the mode-(iv) defect this guard exists for.
    """
    out: Set[int] = set()
    for i, tok in enumerate(toks):
        text = tok.text
        if text.startswith(">") or text == "&>" or _REDIRECT_TOKEN_RE.fullmatch(text):
            out.add(i)
            if i + 1 < len(toks):
                out.add(i + 1)
    return out


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


@dataclass
class Wrapper:
    rel: str
    text: str
    logical: List[Tuple[int, str]]
    variables: Dict[str, str]
    population: str = "primary"
    writers: Dict[str, object] = field(default_factory=dict)
    functions: Set[str] = field(default_factory=set)
    sha256: str = ""


class LivenessGuard:
    def __init__(
        self,
        repo_root: Path,
        corpus: Sequence[str],
        producers: ProducerIndex,
        registry: Optional[dict],
        registry_path: Path,
        registry_present: bool,
        include_retired: bool,
    ) -> None:
        self.repo_root = repo_root
        self.corpus = corpus
        self.producers = producers
        self.registry = registry
        self.registry_path = registry_path
        self.registry_present = registry_present
        self.include_retired = include_retired
        self.tool_specs: Dict[str, ToolSpec] = {}
        self.wrappers: List[Wrapper] = []
        self.findings: List[Finding] = []
        self.unresolved: List[Unresolved] = []
        self.counts = Counts()
        self.stats: Dict[str, int] = {
            "wrappers_primary": 0,
            "wrappers_advisory": 0,
            "tool_invocations": 0,
            "tool_ast_unavailable": 0,
            "reachability_degraded": 0,
            "dynamic_flag_sites": 0,
            "grep_sites": 0,
            "exit_code_gates": 0,
            "exit_code_discarded": 0,
        }

    # -- tools ------------------------------------------------------------

    def tool_spec(self, path: Path) -> ToolSpec:
        key = str(path)
        if key not in self.tool_specs:
            self.tool_specs[key] = scan_tool(path)
        return self.tool_specs[key]

    # -- helpers ----------------------------------------------------------

    def add_finding(self, mode: str, kind: str, rel: str, line: int, detail: str, pop: str) -> None:
        self.findings.append(Finding(mode, kind, rel, line, detail, pop))
        self.counts.bump(mode)

    def add_unresolved(self, mode: str, kind: str, rel: str, line: int, detail: str, pop: str) -> None:
        self.unresolved.append(Unresolved(mode, kind, rel, line, detail, pop))

    # -- the corpus of wrappers ------------------------------------------

    def wrapper_files(self) -> List[Tuple[str, str]]:
        """[(rel_path, population)] for every .sh under scripts/."""
        out: List[Tuple[str, str]] = []
        for rel in self.corpus:
            if not rel.startswith("scripts/") or not rel.endswith(".sh"):
                continue
            retired = rel.startswith("scripts/_retired/")
            if retired:
                out.append((rel, "advisory"))
            else:
                out.append((rel, "primary"))
        out.sort()
        return out

    # -- main scan --------------------------------------------------------

    def scan(self) -> None:
        for rel, population in self.wrapper_files():
            if population == "advisory" and not self.include_retired:
                continue
            path = self.repo_root / rel
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            logical = logical_lines(text)
            wrapper = Wrapper(
                rel=rel,
                text=text,
                logical=logical,
                variables=collect_variables(logical),
                population=population,
                functions=shell_functions(text),
                sha256=hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
            )
            wrapper.writers = self.collect_writers(wrapper)
            self.wrappers.append(wrapper)
            if population == "primary":
                self.stats["wrappers_primary"] += 1
            else:
                self.stats["wrappers_advisory"] += 1
            self.scan_wrapper(wrapper)

    def scan_wrapper(self, wrapper: Wrapper) -> None:
        for li, (lineno, line) in enumerate(wrapper.logical):
            segments = split_commands(line)
            self.scan_line(wrapper, lineno, line, segments, li)

    def collect_writers(self, wrapper: Wrapper) -> Dict[str, object]:
        """basename of a redirect target -> the command that wrote it.

        Wrapper-wide, not line-wide: a wrapper writes its pre-flight
        artefact on one line and greps it twenty lines later.
        """
        writers: Dict[str, object] = {}
        for _lineno, line in wrapper.logical:
            for seg in split_commands(line):
                for target in file_redirects(seg.text):
                    writers.setdefault(basename_of(target), seg)
        return writers

    # -- per logical line -------------------------------------------------

    def scan_line(
        self,
        wrapper: Wrapper,
        lineno: int,
        line: str,
        segments: List[Segment],
        li: int,
    ) -> None:
        writers = wrapper.writers

        for i, seg in enumerate(segments):
            toks = tokenize(seg.text)
            tool_path, tool_idx = self.find_tool(wrapper, toks)
            if tool_path is not None:
                self.stats["tool_invocations"] += 1
                spec = self.tool_spec(tool_path)
                if not spec.has_argparse or not spec.flags:
                    self.stats["tool_ast_unavailable"] += 1
                self.check_mode_i(wrapper, lineno, seg, toks, tool_idx, spec)
                self.check_mode_iv(wrapper, lineno, seg, toks, tool_idx, spec)
                self.check_mode_iii(wrapper, lineno, segments, i, seg, spec, tool_path, li)
            self.check_negated_status(wrapper, lineno, segments, i, seg, li)
            self.check_mode_ii(wrapper, lineno, segments, i, seg, toks, writers)
            self.check_exit_code_discarded(wrapper, lineno, seg, toks, tool_idx)

    # -- tool resolution --------------------------------------------------

    def find_tool(self, wrapper: Wrapper, toks: Sequence[Tok]) -> Tuple[Optional[Path], int]:
        """The repo .py a command invokes, and its token index.

        The tool is the ARGUMENT of the command, not any token in it:

        * the command word IS a ``.py`` file  -> that file is the tool;
        * the command word is a python interpreter -> the first
          non-assignment argument is the tool;
        * the command word is a shell FUNCTION defined by this wrapper ->
          no tool here (the function's arguments are the FUNCTION's
          arguments; the real invocation lives in the function body);
        * anything else -> no tool.

        Command-scoped: the flags that matter are the ones AFTER the .py
        token INSIDE THAT SAME COMMAND.  A quoted literal that merely
        mentions a .py file (``printf '... pre_flight_check.py ...'``) is
        NOT a tool: a QUOTED token only counts when it contains a variable
        reference, an UNQUOTED token always counts.
        """
        idx, word = command_word(toks)
        if not word:
            return None, -1
        if word in wrapper.functions:
            return None, -1
        resolved_word = expand(word, wrapper.variables)
        if resolved_word.endswith(".py"):
            target = resolve_repo_file(resolved_word)
            if target is not None and target.suffix == ".py":
                return target, idx
            return None, -1
        base = basename_of(resolved_word)
        if not re.fullmatch(r"python[0-9.]*", base):
            return None, -1
        j = idx + 1
        while j < len(toks):
            text = toks[j].text
            if text.startswith("-") and "=" not in text:
                j += 1
                continue
            if "=" in text and _ASSIGN_NAME.match(text.split("=", 1)[0]):
                j += 1
                continue
            break
        if j >= len(toks):
            return None, -1
        resolved = expand(toks[j].text, wrapper.variables)
        if not resolved.endswith(".py"):
            return None, -1
        target = resolve_repo_file(resolved)
        if target is not None and target.suffix == ".py":
            return target, j
        return None, -1

    # -- mode (i) ---------------------------------------------------------

    def check_mode_i(
        self,
        wrapper: Wrapper,
        lineno: int,
        seg: Segment,
        toks: Sequence[Tok],
        tool_idx: int,
        spec: ToolSpec,
    ) -> None:
        if tool_idx < 0:
            return
        if not spec.has_argparse:
            return
        if not spec.flags:
            self.add_unresolved(
                "i",
                "oracle-unavailable",
                wrapper.rel,
                lineno,
                f"{spec.path.name}: no literal add_argument found; the flag "
                f"oracle is UNKNOWN ({spec.scan_note})",
                wrapper.population,
            )
            return
        seen: Set[str] = set()
        redirect_idx = redirect_token_indices(toks)
        idx = tool_idx + 1
        n = len(toks)
        while idx < n:
            if idx in redirect_idx:
                idx += 1
                continue
            tok = toks[idx]
            text = tok.text
            if text == "--":
                break
            if not text.startswith("--"):
                idx += 1
                continue
            name = text
            value: Optional[str] = None
            if "=" in text:
                name, value = text.split("=", 1)
            flag = spec.flags.get(name)
            if flag is None:
                self.add_finding(
                    "i",
                    "undefined-flag",
                    wrapper.rel,
                    lineno,
                    f"{spec.path.name} does not define {name} "
                    f"(argparse exits 2 immediately: the gate measures nothing)",
                    wrapper.population,
                )
                idx += 1
                continue
            seen.add(name)
            if value is None and flag.takes_value and idx + 1 < n:
                nxt = toks[idx + 1]
                if not nxt.text.startswith("--") and (idx + 1) not in redirect_idx:
                    value = nxt.text
                    idx += 1
            if (
                value is not None
                and flag.choices
                and "$" not in value
                and value not in flag.choices
            ):
                self.add_finding(
                    "i",
                    "invalid-choice",
                    wrapper.rel,
                    lineno,
                    f"{spec.path.name} {name} does not accept {value!r} "
                    f"(argparse choices: {sorted(flag.choices)})",
                    wrapper.population,
                )
            idx += 1
        # A required option that may be supplied through an ARRAY expansion
        # (`"${cl_args[@]}"`) or a command substitution is NOT counted as
        # missing: the static pass does not resolve the array's contents, so
        # the option may well be there (COVERAGE BOUND: dynamic flags are
        # not counted).
        dynamic = any(_DYNAMIC_TOKEN_RE.search(tok.text) for tok in toks)
        for name, flag in sorted(spec.flags.items()):
            if flag.required and name not in seen:
                if dynamic:
                    self.stats["dynamic_flag_sites"] += 1
                    continue
                self.add_finding(
                    "i",
                    "required-option-missing",
                    wrapper.rel,
                    lineno,
                    f"{spec.path.name} requires {name} (required=True) but the "
                    f"command does not pass it (argparse exits 2)",
                    wrapper.population,
                )

    # -- mode (iii) -------------------------------------------------------

    def block_tail(
        self, wrapper: Wrapper, segments: List[Segment], idx: int, li: int, need: str
    ) -> str:
        """The text after a command, extended across LOGICAL LINES when the
        block it opens is not closed on this line.

        ``tool || {`` / ``exit 1`` / ``}`` and ``if ! tool; then`` /
        ``die ...`` / ``fi`` are the two shapes a wrapper uses, and both
        put the deciding statement on the NEXT line.
        """
        tail = " ".join(seg.text for seg in segments[idx + 1 :])
        for k in range(li + 1, len(wrapper.logical)):
            if tail.count("{") <= tail.count("}") and (
                need != "fi" or re.search(r"(^|[\s;{])fi(\s|$)", tail)
            ):
                break
            tail += " ; " + wrapper.logical[k][1]
        return tail

    def check_mode_iii(
        self,
        wrapper: Wrapper,
        lineno: int,
        segments: List[Segment],
        idx: int,
        seg: Segment,
        spec: ToolSpec,
        tool_path: Path,
        li: int,
    ) -> None:
        """A gate keyed on a tool exit code that is 0 BY CONSTRUCTION."""
        kind: Optional[str] = None
        if seg.sep_after == "||":
            # `tool || { ...; exit N; }` / `tool || exit N`
            tail = self.block_tail(wrapper, segments, idx, li, "}")
            if re.search(r"(^|[\s;{])exit\s", tail):
                kind = "exit-code-gate-hard"
            elif re.search(r"(^|[\s;{])(die|fatal)\s", tail):
                kind = "exit-code-gate-hard"
        stripped = seg.text.lstrip()
        if stripped.startswith("if !") or stripped.startswith("if  !"):
            tail = self.block_tail(wrapper, segments, idx, li, "fi")
            if re.search(r"(^|[\s;{])exit\s", tail):
                kind = "exit-code-gate-hard"
            elif re.search(r"(^|[\s;{])(die|fatal)\s", tail):
                kind = "exit-code-gate-hard"
            else:
                kind = "exit-code-gate-advisory"
        if kind is None:
            return
        self.stats["exit_code_gates"] += 1
        if not self.registry_present or self.registry is None:
            self.add_unresolved(
                "iii",
                "registry-absent",
                wrapper.rel,
                lineno,
                f"{tool_path.name}: the exit-code registry "
                f"({self.registry_path.name}) is missing or unreadable, so the "
                f"tool's exit code under a failing input is UNKNOWN — ABSTAIN",
                wrapper.population,
            )
            return
        entry = (self.registry.get("tools") or {}).get(tool_path.name)
        if entry is None:
            self.add_unresolved(
                "iii",
                "tool-not-in-registry",
                wrapper.rel,
                lineno,
                f"{tool_path.name}: the exit-code registry has no measured entry "
                f"for this tool — ABSTAIN",
                wrapper.population,
            )
            return
        code = entry.get("exit_code_under_failing_input")
        if not isinstance(code, int):
            self.add_unresolved(
                "iii",
                "registry-entry-unreadable",
                wrapper.rel,
                lineno,
                f"{tool_path.name}: registry entry has no integer "
                f"exit_code_under_failing_input — ABSTAIN",
                wrapper.population,
            )
            return
        if code == 0:
            self.add_finding(
                "iii",
                kind,
                wrapper.rel,
                lineno,
                f"{tool_path.name} exits {code} under a failing input "
                f"(MEASURED, registry {self.registry_path.name}); the gate keys "
                f"on that exit code, so it cannot fail. Verdict location: "
                f"{entry.get('verdict_location') or 'unknown'} — read it with "
                f"molmetal/scripts/wf_launch_gate_verdict.py",
                wrapper.population,
            )


    # -- mode (iii), shape B: `VAR=$?` inside the then-branch of `if !` ----

    def check_negated_status(
        self,
        wrapper: Wrapper,
        lineno: int,
        segments: List[Segment],
        idx: int,
        seg: Segment,
        li: int,
    ) -> None:
        """``if ! cmd; then VAR=$?`` — bash's ``$?`` inside that then-branch
        is the status of the ``!`` compound, which is 0 on EVERY path.

        MEASURED (bash, this session):
            $ f(){ return 7; }; if ! f; then echo $?; fi   ->  0
            $ f(){ return 7; }; f; s=$?; echo $s          ->  7
            $ f(){ return 7; }; if f; then :; else s=$?; echo $s  ->  7

        So ``VAR`` is 0 whatever the tool did, and a later gate on ``VAR``
        can never fire.  The capture is only reported when the variable is
        LATER COMPARED against 0 in a conditional, and when every other
        assignment to it in the wrapper is either inside a negated
        then-branch or a plain initialiser.
        """
        if not _NEGATED_IF_RE.match(seg.text):
            return
        tail = self.block_tail(wrapper, segments, idx, li, "fi")
        m = _ELSE_RE.search(tail)
        then_branch = tail[: m.start()] if m else tail
        for cap in _STATUS_CAPTURE_RE.finditer(then_branch):
            var = cap.group(1)
            if not self._status_is_gated(wrapper, var):
                continue
            if not self._all_captures_negated(wrapper, var):
                continue
            self.stats["exit_code_gates"] += 1
            self.add_finding(
                "iii",
                "status-capture-after-negation",
                wrapper.rel,
                lineno,
                f"`{var}=$?` sits inside the then-branch of `if ! ...`, where "
                f"bash's `$?` is the status of the `!` compound -- 0 on EVERY "
                f"path (MEASURED: `if ! f; then echo $?` prints 0 for "
                f"`f(){{ return 7; }}`). `{var}` is therefore always 0 and the "
                f"later gate on it can never fire; capture the status in the "
                f"else-branch (`if cmd; then ...; else {var}=$?; fi`) or use "
                f"`cmd || {var}=$?`",
                wrapper.population,
            )
            return

    def _status_is_gated(self, wrapper: Wrapper, var: str) -> bool:
        """Is ``var`` LATER compared against 0 inside a conditional?"""
        pat = re.compile(
            r"\[\s*\"?\$\{?" + re.escape(var) + r"\}?\"?\s*-(?:ne|eq|gt|ge|lt|le)\s*0"
        )
        for _lineno, line in wrapper.logical:
            if not pat.search(line):
                continue
            if re.search(r"(^|[\s;&|(\[])if\s|\bif\s|\belif\s|\bwhile\s|&&|\|\|", line):
                return True
        return False

    def _all_captures_negated(self, wrapper: Wrapper, var: str) -> bool:
        """Every ``VAR=$?`` in the wrapper is inside a negated then-branch.

        A capture that sits on a plain line (the normal `cmd; VAR=$?` shape)
        is legitimate, so the wrapper is not reported.
        """
        cap_pat = re.compile(r"(?<![\w.$\{])" + re.escape(var) + r"=\$\?")
        for k, (_lineno, line) in enumerate(wrapper.logical):
            if not cap_pat.search(line):
                continue
            prev = wrapper.logical[k - 1][1] if k > 0 else ""
            if _NEGATED_IF_RE.match(prev) and "fi" not in prev:
                continue  # inside a negated then-branch
            return False
        return True

    # -- mode (iv) --------------------------------------------------------

    def check_mode_iv(
        self,
        wrapper: Wrapper,
        lineno: int,
        seg: Segment,
        toks: Sequence[Tok],
        tool_idx: int,
        spec: ToolSpec,
    ) -> None:
        kinds: List[str] = []
        details: List[str] = []
        for target in file_redirects(seg.text):
            if not target.endswith(".json"):
                continue
            if has_dup_redirect(seg.text):
                kinds.append("json-redirect-captures-stderr")
                details.append(
                    f"{target}: stdout+stderr are redirected into the .json "
                    f"path, so the artefact is NOT JSON (json.load -> "
                    f"'Expecting value: line 1 column 2')"
                )
        if tool_idx >= 0 and spec.flags:
            redirect_idx = redirect_token_indices(toks)
            for idx in range(tool_idx + 1, len(toks)):
                if idx in redirect_idx:
                    continue
                tok = toks[idx]
                if not tok.text.startswith("--"):
                    continue
                name = tok.text
                inline_value = None
                if "=" in name:
                    name, inline_value = name.split("=", 1)
                flag = spec.flags.get(name)
                if flag is None:
                    continue
                if not (flag.path_typed and flag.optional_value):
                    continue
                if inline_value is not None:
                    continue
                nxt = toks[idx + 1] if idx + 1 < len(toks) else None
                if nxt is not None and not nxt.text.startswith("--"):
                    if (idx + 1) in redirect_idx:
                        pass  # the "value" is a redirect target: the flag is BARE
                    else:
                        continue  # the flag DOES take a path
                kinds.append("bare-path-flag-writes-nothing")
                details.append(
                    f"{spec.path.name} {name} is type=Path nargs='?' — a BARE "
                    f"{name} writes NO file, the payload goes to stdout"
                )
        if kinds:
            self.add_finding(
                "iv",
                "+".join(sorted(set(kinds))),
                wrapper.rel,
                lineno,
                "; ".join(details),
                wrapper.population,
            )

    # -- exit code discarded ---------------------------------------------

    def check_exit_code_discarded(
        self,
        wrapper: Wrapper,
        lineno: int,
        seg: Segment,
        toks: Sequence[Tok],
        tool_idx: int,
    ) -> None:
        if tool_idx < 0:
            return
        if seg.sep_after != "||":
            return
        # `|| true` DISCARDS the exit code; it does not key a gate on it.
        # Whether that is a defect depends on what the wrapper does next
        # (a wrapper that then reads the REPORT is live), so this is a
        # census number, never a finding.
        self.stats["exit_code_discarded"] += 1

    # -- mode (ii) --------------------------------------------------------

    def check_mode_ii(
        self,
        wrapper: Wrapper,
        lineno: int,
        segments: List[Segment],
        idx: int,
        seg: Segment,
        toks: Sequence[Tok],
        writers: Dict[str, Segment],
    ) -> None:
        grep_idx = -1
        for i, tok in enumerate(toks):
            if tok.text in ("grep", "egrep", "fgrep") or tok.text.endswith("/grep"):
                grep_idx = i
                break
        if grep_idx < 0:
            return
        self.stats["grep_sites"] += 1
        pattern: Optional[str] = None
        files: List[str] = []
        i = grep_idx + 1
        flags_used: Set[str] = set()
        while i < len(toks):
            text = toks[i].text
            if text == "--":
                if i + 1 < len(toks):
                    pattern = toks[i + 1].text
                    i += 2
                break
            if text.startswith("-") and text != "-":
                flags_used.add(text)
                if text in ("-e", "--regexp"):
                    if i + 1 < len(toks):
                        pattern = toks[i + 1].text
                        i += 2
                        continue
                i += 1
                continue
            if pattern is None:
                pattern = text
                i += 1
                continue
            files.append(text)
            i += 1
        if pattern is None:
            return
        if "$" in pattern:
            self.add_unresolved(
                "ii",
                "variable-operand",
                wrapper.rel,
                lineno,
                f"grep operand {pattern!r} contains a shell variable: the "
                f"marker cannot be resolved statically — ABSTAIN",
                wrapper.population,
            )
            return

        ere = any(f in ("-E", "-P", "--extended-regexp", "--perl-regexp") for f in flags_used)
        alternatives = _split_alternatives(pattern, ere)
        target_files = list(files)
        pipeline = not files

        for alt in alternatives:
            if not alt:
                continue
            if pipeline:
                upstream = self._pipeline_upstream(wrapper, segments, idx)
                self._check_marker_pipeline(
                    wrapper, lineno, alt, upstream, segments, idx
                )
                continue
            self._check_marker_file(
                wrapper, lineno, alt, target_files, writers, segments
            )

    def _pipeline_upstream(
        self, wrapper: Wrapper, segments: List[Segment], idx: int
    ) -> Optional[Path]:
        """The tool whose stdout a piped grep reads."""
        for j in range(idx - 1, -1, -1):
            seg = segments[j]
            if seg.sep_after != "|":
                continue
            toks = tokenize(seg.text)
            path, _ = self.find_tool(wrapper, toks)
            if path is not None:
                return path
        return None

    def _check_marker_pipeline(
        self,
        wrapper: Wrapper,
        lineno: int,
        marker: str,
        upstream: Optional[Path],
        segments: List[Segment],
        idx: int,
    ) -> None:
        if upstream is None:
            self.add_unresolved(
                "ii",
                "pipeline-producer-unknown",
                wrapper.rel,
                lineno,
                f"the grep reads a PIPELINE, not a file; the upstream producer "
                f"could not be resolved statically — ABSTAIN (marker "
                f"{marker!r})",
                wrapper.population,
            )
            return
        rel = self._rel_of(upstream)
        if rel and self.producers.contains(rel, marker):
            return
        if marker.startswith("--"):
            # A marker that is the NAME OF AN OPTION is a CAPABILITY PROBE:
            # the grep asks "does this tool expose the flag?".  Its absence
            # is the expected state and the branch that fires on absence
            # BLOCKS, so the gate can fail — it is not a gate that cannot
            # fail.  (Example: launch_wf310f_frontier_math_gpu.sh:329.)
            self.add_unresolved(
                "ii",
                "capability-probe",
                wrapper.rel,
                lineno,
                f"marker {marker!r} is an OPTION NAME probed against "
                f"{upstream.name}: a capability probe, not a log marker — "
                f"NOT counted as a dead gate",
                wrapper.population,
            )
            return
        if _is_regexish(marker):
            self.add_unresolved(
                "ii",
                "regex-operand",
                wrapper.rel,
                lineno,
                f"marker {marker!r} is a regex, not a literal — ABSTAIN",
                wrapper.population,
            )
            return
        self.add_finding(
            "ii",
            "dead-marker",
            wrapper.rel,
            lineno,
            f"marker {marker!r} has NO producer: {upstream.name} never prints "
            f"it, so the grep can never match",
            wrapper.population,
        )

    def _check_marker_file(
        self,
        wrapper: Wrapper,
        lineno: int,
        marker: str,
        target_files: List[str],
        writers: Dict[str, Segment],
        segments: List[Segment],
    ) -> None:
        # 1. the grepped file itself may be the producer
        resolvable = 0
        for target in target_files:
            base = basename_of(expand(target, wrapper.variables))
            if "$" in base:
                continue
            resolvable += 1
            if self.producers.contains(target, marker):
                return
            if base.endswith(".json"):
                writer = writers.get(base)
                if writer is not None:
                    wpath, _ = self.find_tool(wrapper, tokenize(writer.text))
                    if wpath is not None:
                        return  # the writer tool produced that structured output
                self.add_unresolved(
                    "ii",
                    "json-report-producer-unknown",
                    wrapper.rel,
                    lineno,
                    f"the grep reads {base} but no writer command for it was "
                    f"found in this wrapper — ABSTAIN (marker {marker!r})",
                    wrapper.population,
                )
                return
        if not resolvable:
            self.add_unresolved(
                "ii",
                "unknown-target-file",
                wrapper.rel,
                lineno,
                f"the grep's file operand could not be resolved statically — "
                f"ABSTAIN (marker {marker!r})",
                wrapper.population,
            )
            return

        producers = self.producers.producers(marker, exclude=[wrapper.rel])
        if not producers:
            if _is_regexish(marker):
                self.add_unresolved(
                    "ii",
                    "regex-operand",
                    wrapper.rel,
                    lineno,
                    f"marker {marker!r} is a regex, not a literal — ABSTAIN",
                    wrapper.population,
                )
                return
            self.add_finding(
                "ii",
                "dead-marker",
                wrapper.rel,
                lineno,
                f"marker {marker!r} has NO producer anywhere in the repo code "
                f"corpus: the branch that requires it can never be taken",
                wrapper.population,
            )
            return
        # producers exist — is one of them reachable from the writer?
        writer_tool = self._writer_tool(wrapper, target_files, writers)
        if writer_tool is None:
            self.stats["reachability_degraded"] += 1
            return
        names = {Path(p).name for p in producers}
        stem = Path(writer_tool).stem
        if writer_tool.name in names or any(stem in n or n in stem for n in names):
            return
        if any(Path(p).name in wrapper.text for p in producers):
            return
        self.add_finding(
            "ii",
            "unreachable-producer",
            wrapper.rel,
            lineno,
            f"marker {marker!r} is produced by {sorted(names)[:3]} but NOT by "
            f"the command that writes the grepped log ({Path(writer_tool).name})",
            wrapper.population,
        )

    def _writer_tool(
        self,
        wrapper: Wrapper,
        target_files: Sequence[str],
        writers: Dict[str, Segment],
    ) -> Optional[str]:
        for target in target_files:
            base = basename_of(expand(target, wrapper.variables))
            if "$" in base:
                continue
            writer = writers.get(base)
            if writer is None:
                continue
            wpath, _ = self.find_tool(wrapper, tokenize(writer.text))
            if wpath is not None:
                return wpath.name
            return None  # writer known but its tool is not resolvable
        return None

    def _rel_of(self, path: Path) -> Optional[str]:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return None


def _is_regexish(marker: str) -> bool:
    if any(ch.isspace() for ch in marker):
        return True
    return any(ch in REGEX_METACHARS for ch in marker)


def _split_alternatives(pattern: str, ere: bool) -> List[str]:
    parts = pattern.split("\\|")
    if ere:
        expanded: List[str] = []
        for part in parts:
            expanded.extend(part.split("|"))
        parts = expanded
    return [p for p in parts]


# ---------------------------------------------------------------------------
# allow-list
# ---------------------------------------------------------------------------


@dataclass
class AllowlistEntry:
    id: str
    wrapper: str
    line: int
    mode: str
    reason: str
    producer: str
    owner: str
    expires: str

    def key(self) -> str:
        return f"{self.wrapper}:{self.line}:{self.mode}"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "wrapper": self.wrapper,
            "line": self.line,
            "mode": self.mode,
            "reason": self.reason,
            "producer": self.producer,
            "owner": self.owner,
            "expires": self.expires,
        }


@dataclass
class Allowlist:
    path: Path
    exists: bool
    entries: List[AllowlistEntry] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Allowlist":
        if not path.is_file():
            return cls(path, False, [], ["allow-list file not found"])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return cls(path, True, [], [f"allow-list unreadable: {exc}"])
        raw = data.get("entries") if isinstance(data, dict) else data
        entries: List[AllowlistEntry] = []
        errors: List[str] = []
        if not isinstance(raw, list):
            return cls(path, True, [], ["allow-list has no 'entries' list"])
        for item in raw:
            if not isinstance(item, dict):
                errors.append(f"allow-list entry is not an object: {item!r}")
                continue
            missing = [
                k
                for k in ("wrapper", "line", "mode", "reason", "owner", "expires")
                if k not in item
            ]
            if missing:
                errors.append(
                    f"allow-list entry {item.get('id', '?')} is missing {missing}"
                )
                continue
            entries.append(
                AllowlistEntry(
                    id=str(item.get("id", "")),
                    wrapper=str(item["wrapper"]),
                    line=int(item["line"]),
                    mode=str(item["mode"]),
                    reason=str(item["reason"]),
                    producer=str(item.get("producer", "")),
                    owner=str(item["owner"]),
                    expires=str(item["expires"]),
                )
            )
        return cls(path, True, entries, errors)

    def stale(self, today: date) -> List[AllowlistEntry]:
        out = []
        for entry in self.entries:
            try:
                exp = datetime.strptime(entry.expires, "%Y-%m-%d").date()
            except ValueError:
                out.append(entry)
                continue
            if (today - exp).days > STALE_ALLOWLIST_DAYS:
                out.append(entry)
        return out

    def expired(self, today: date) -> List[AllowlistEntry]:
        out = []
        for entry in self.entries:
            try:
                exp = datetime.strptime(entry.expires, "%Y-%m-%d").date()
            except ValueError:
                out.append(entry)
                continue
            if exp < today:
                out.append(entry)
        return out


# ---------------------------------------------------------------------------
# coverage bound
# ---------------------------------------------------------------------------

COVERAGE_BOUND = [
    "shell constructs the lexical pass does not model: process substitution "
    "(<(...)), here-doc substitution, $'...' quoting, arithmetic contexts, "
    "functions defined in another file, `case` bodies, and shell builtins "
    "that generate commands at runtime",
    "flags resolved dynamically (eval, arrays, \"$@\", runtime-generated "
    "commands, flags coming from a config file) are NOT counted; a required "
    "option that may arrive through an array expansion is NOT reported as "
    "missing, and the count of such sites is reported",
    "mode (i) oracle = the tool's own argparse AST; a tool whose parser is "
    "built dynamically has NO literal add_argument and is reported "
    "UNRESOLVED, never as a finding",
    "mode (iii) oracle = the MEASURED exit-code registry written by "
    "molmetal/scripts/wf_launch_gate_exitcode_probe.py; with no registry the "
    "mode is reported UNRESOLVED (the guard abstains, it never guesses)",
    "mode (ii) producer search covers .py and .sh files only and excludes "
    "molmetal/tests/, molmetal/reports/, metrics/, results/, docs/, TODO/ and "
    "paper/ — a marker that only occurs in a test, a report or a document "
    "counts as having NO producer",
    "mode (ii) reachability degrades to a repo-wide producer search when "
    "the command that wrote the grepped log cannot be resolved (a "
    "positional parameter trainer, a variable indirection); the count of "
    "such sites is reported",
    "mode (ii) cannot see a producer that is only reached conditionally at "
    "runtime (the source contains the marker, the branch never prints it)",
    "the guard checks the INTERFACE of a gate, not its SEMANTICS: a gate "
    "whose success branch is reachable but computes the wrong thing is "
    "invisible to it",
    "gates written in any language other than bash are not scanned; Python "
    "launchers, Makefiles and CI YAML are out of scope",
    "whether a wrapper is ever LAUNCHED is not observable: this is a census "
    "of GATES THAT CANNOT FAIL, not of launches that failed",
    "the guard EXECUTES nothing: it never runs a wrapper, never runs "
    "--help on a tool and never allocates a GPU",
    "this report is a COVERAGE census, not a defect count (AGENTS.md "
    "§6.269.6): classified / unresolved / dead are three separate numbers "
    "and a reader must never read 'dead' as the number of dead gates in the "
    "repo",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Static liveness guard for launch-wrapper pre-launch gates: "
            "reports the DEAD-GATE class (modes i-iv) over scripts/*.sh."
        )
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the machine-readable report to this path",
    )
    ap.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help=f"allow-list JSON (default: {DEFAULT_ALLOWLIST.name} next to this script)",
    )
    ap.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=(
            "MEASURED exit-code registry written by "
            f"{DEFAULT_REGISTRY.name} (default: next to this script)"
        ),
    )
    ap.add_argument(
        "--include-retired",
        action="store_true",
        help="also scan scripts/_retired/** (advisory population)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat UNRESOLVED as FAIL",
    )
    ap.add_argument(
        "--only",
        choices=list(MODES),
        default=None,
        help="report only one mode",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: derived from this script's location)",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    today = datetime.now(timezone.utc).date()

    try:
        corpus = load_corpus(repo_root)
    except Exception as exc:  # noqa: BLE001 - the guard must abstain, not crash
        print(f"SKIP the guard could not run: {exc}")
        return 2
    if not corpus:
        print("SKIP the guard could not run: the repo corpus is empty")
        return 2

    registry_present = args.registry.is_file()
    registry: Optional[dict] = None
    registry_error = ""
    if registry_present:
        try:
            registry = json.loads(args.registry.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            registry_present = False
            registry_error = str(exc)

    allowlist = Allowlist.load(args.allowlist)
    producers = ProducerIndex(repo_root, corpus)

    guard = LivenessGuard(
        repo_root=repo_root,
        corpus=corpus,
        producers=producers,
        registry=registry,
        registry_path=args.registry,
        registry_present=registry_present,
        include_retired=args.include_retired,
    )
    guard.scan()

    findings = guard.findings
    unresolved = guard.unresolved
    if args.only:
        findings = [f for f in findings if f.mode == args.only]
        unresolved = [u for u in unresolved if u.mode == args.only]

    suppressed: List[Finding] = []
    live: List[Finding] = []
    allow_keys = {e.key(): e for e in allowlist.entries}
    for finding in findings:
        entry = allow_keys.get(finding.key())
        if entry is None:
            live.append(finding)
            continue
        try:
            exp = datetime.strptime(entry.expires, "%Y-%m-%d").date()
        except ValueError:
            live.append(finding)
            continue
        if exp < today:
            live.append(finding)  # an expired entry re-flags the finding
            continue
        suppressed.append(finding)

    counts = Counts()
    for finding in live:
        counts.bump(finding.mode)
    unresolved_counts = Counts()
    for item in unresolved:
        unresolved_counts.bump(item.mode)

    stale_entries = allowlist.stale(today)
    expired_entries = allowlist.expired(today)
    # ROT CENSUS: an allow-list entry that no longer matches any finding is
    # dead weight -- it is reported (unused) so the list cannot grow silently.
    live_keys = {f.key() for f in findings}
    unused_entries = [e for e in allowlist.entries if e.key() not in live_keys]

    corpus_digest = hashlib.sha256(
        "\n".join(
            f"{w.rel}:{w.sha256}" for w in sorted(guard.wrappers, key=lambda x: x.rel)
        ).encode("utf-8")
    ).hexdigest()

    if live:
        verdict = "FAIL"
        reason = f"{len(live)} unallowlisted dead-gate finding(s)"
        exit_code = 1
    elif unresolved and args.strict:
        verdict = "FAIL"
        reason = f"{len(unresolved)} unresolved gate(s) under --strict"
        exit_code = 1
    elif not registry_present:
        verdict = "SKIP"
        reason = (
            f"the mode-(iii) exit-code registry is missing or unreadable "
            f"({args.registry}); that mode is UNRESOLVED and the guard abstains"
        )
        exit_code = 2
    elif unresolved:
        verdict = "PASS"
        reason = (
            f"no unallowlisted finding; {len(unresolved)} gate(s) UNRESOLVED "
            f"(reported, not guessed)"
        )
        exit_code = 0
    else:
        verdict = "PASS"
        reason = "no dead-gate finding"
        exit_code = 0

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "guard": f"molmetal/scripts/{TOOL_NAME}",
        "verdict": verdict,
        "exit_code": exit_code,
        "reason": reason,
        "counts": counts.as_dict(),
        "unresolved_counts": unresolved_counts.as_dict(),
        "findings": [f.as_dict() for f in live],
        "suppressed_by_allowlist": [f.as_dict() for f in suppressed],
        "unresolved": [u.as_dict() for u in unresolved],
        "allowlist": {
            "path": str(args.allowlist),
            "exists": allowlist.exists,
            "entries": len(allowlist.entries),
            "stale": len(stale_entries),
            "expired": len(expired_entries),
            "unused": len(unused_entries),
            "unused_detail": [e.as_dict() for e in unused_entries],
            "errors": allowlist.errors,
            "entries_detail": [e.as_dict() for e in allowlist.entries],
        },
        "registry": {
            "path": str(args.registry),
            "present": registry_present,
            "error": registry_error,
            "measured_by": (registry or {}).get("measured_by"),
            "measured_at": (registry or {}).get("measured_at"),
        },
        "population": {
            "primary_wrappers": guard.stats["wrappers_primary"],
            "advisory_wrappers": guard.stats["wrappers_advisory"],
            "tool_invocations": guard.stats["tool_invocations"],
            "tool_ast_unavailable": guard.stats["tool_ast_unavailable"],
            "grep_sites": guard.stats["grep_sites"],
            "exit_code_gates": guard.stats["exit_code_gates"],
            "reachability_degraded": guard.stats["reachability_degraded"],
            "dynamic_flag_sites": guard.stats["dynamic_flag_sites"],
        },
        "corpus_digest": corpus_digest,
        "wrapper_sha256": {
            w.rel: w.sha256 for w in guard.wrappers
        },
        "coverage_bound": COVERAGE_BOUND,
        "tool_ast_notes": {
            spec.path.name: spec.scan_note
            for spec in guard.tool_specs.values()
            if spec.scan_note
        },
    }

    print(f"{verdict} {reason}")
    for finding in live:
        print(
            f"{finding.file}:{finding.line} mode={finding.mode} "
            f"kind={finding.kind} [{finding.population}] {finding.detail}"
        )

    for entry in allowlist.entries:
        print(
            f"ALLOWLIST {entry.id} {entry.wrapper}:{entry.line} mode={entry.mode} "
            f"expires={entry.expires} owner={entry.owner} reason={entry.reason[:120]}"
        )
    if args.json:
        out = args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
        print(f"REPORT {out}")
    for line in COVERAGE_BOUND:
        print(f"COVERAGE-BOUND {line}")
    summary = (
        f"SUMMARY modes: i={counts.i} ii={counts.ii} iii={counts.iii} "
        f"iv={counts.iv} | dead={counts.total()} "
        f"unresolved={len(unresolved)} classified={counts.total() + len(unresolved)} "
        f"| allowlist={len(allowlist.entries)} entries "
        f"({len(stale_entries)} stale>{STALE_ALLOWLIST_DAYS}d, "
        f"{len(expired_entries)} expired, {len(suppressed)} used, "
        f"{len(unused_entries)} unused) "
        f"| population: primary={guard.stats['wrappers_primary']} "
        f"advisory={guard.stats['wrappers_advisory']} "
        f"| registry={'present' if registry_present else 'ABSENT'} "
        f"| corpus={corpus_digest[:12]}"
    )
    print(summary)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
