#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wf-A171 — an EVIDENCE-BASED live/archived classifier for the GPU-lock guard.

Why this module exists
----------------------
``molmetal/scripts/wfa167_command_prefix_guard.py`` decides whether a
``CUDA_VISIBLE_DEVICES`` defeat site is *archived* (ignored) or *live* (a real
finding) with a **path-prefix heuristic**::

    ARCHIVE_PREFIXES = ("TODO/archive/", "results/")          # :120-124
    archived = path.startswith(ARCHIVE_PREFIXES)              # :727

``results/`` is not an archive — it holds live run trees, and
``results/cfm_full_retrain/launch_loop.sh`` is a tracked, executable
5-seed x 5-pocket x 50-epoch CFM retrain loop.  Filing it under "archived"
made the guard report ``real_defeaters_live: 0`` while a live hard GPU
override sat on disk.  A guard that hides a live defeater behind a path
substring manufactures false assurance, which is worse than no guard.

The fix is to replace the heuristic with **positive, checkable evidence**:

* A file is ARCHIVED only when some **tracked text declares it archived** —
  either an *in-tree marker* (a ``README*`` inside the tree that says the tree
  is retired/archived) or an *external declaration* ("... archived under
  ``<tree>/``").  This is evidence of intent, not a guess from a path shape.
* A file is LIVE when it is referenced by a tracked, non-archived file
  (reachability), or when nothing declares it archived at all.

**Polarity.**  Absence of an archival declaration means LIVE, not ARCHIVED.
An unproven "archived" claim is exactly the failure this module exists to
remove, so the default is the safe one: report it.

Denominators
------------
Every count in the emitted census carries its denominator:

* ``tracked_files_total``        — ``git ls-files -z`` (the whole tracked tree)
* ``text_candidate_files``       — tracked files whose extension is in
  ``TEXT_EXTENSIONS`` (or that have no extension)
* ``files_read`` / ``files_skipped_non_utf8`` / ``files_skipped_too_large``
* ``files_missing_on_disk``      — tracked but absent (deleted / gitignored)

Hard constraints
----------------
* READ-ONLY: this module never writes to a scanned file.
* CPU-ONLY / stdlib-only: no ``torch``, no ``nvidia-smi``, no GPU calls.
* No ``os.walk``: every file list comes from ``git ls-files -z``.
* UTF-8 safe: files are decoded explicitly and non-UTF-8 files are *counted*
  and *reported*, never silently skipped (AGENTS.md / the ugrep ``-I`` trap).

Usage
-----
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wfa171_gpu_lock_guard_path.py --json
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wfa171_gpu_lock_guard_path.py \
        --out molmetal/reports/wfa171_gpu_lock_guard/census.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "CVD",
    "TEXT_EXTENSIONS",
    "BUCKETS",
    "classify_value",
    "classify_liveness",
    "build_liveness_index",
    "scan_text",
    "census",
    "main",
]

CVD = "CUDA_VISIBLE_DEVICES"

#: Tracked-file extensions considered "possibly text".  A file outside this
#: set is not scanned; the census reports the denominator so the coverage
#: claim stays auditable.
TEXT_EXTENSIONS: Tuple[str, ...] = (
    ".sh",
    ".bash",
    ".zsh",
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
    ".toml",
    ".env",
    ".rst",
    ".jsonl",
    ".jsonc",
)

#: Files larger than this are not read (reported as ``skipped_too_large``).
MAX_FILE_BYTES = 4 * 1024 * 1024

#: The three semantic buckets the census reports, plus two non-assignment
#: buckets.  ``DEFEATER`` and ``POTENTIAL_DEFEATER`` both mean "this site can
#: put a GPU back on"; they are separated because one is *proven* statically
#: and the other depends on an unresolved variable.
BUCKETS: Tuple[str, ...] = (
    "DEFEATER",
    "POTENTIAL_DEFEATER",
    "LOCK_HONOURING",
    "NEUTRAL_PRESET",
    "NOT_AN_ASSIGNMENT",
)

# --------------------------------------------------------------------------- #
# Value classification
# --------------------------------------------------------------------------- #
#: A self reference to the ambient variable, with or without the POSIX colon.
#: NOTE: the separator alternation is ``:-`` (colon) vs ``-`` (no colon); a
#: character class ``[:-?]`` would silently fail to match the no-colon form,
#: which is precisely the lock-preserving spelling.
_SELF_REF_RE = re.compile(
    r"\$\{?%s\b(?P<sep>:-|-)?(?P<def>[^}]*)\}?" % re.escape(CVD)
)
#: Any parameter expansion at all.
_PARAM_EXP_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?P<sep>:-|-)(?P<def>[^}]*)\}")
#: A bare ``$VAR`` reference.
_BARE_VAR_RE = re.compile(r"^\$[A-Za-z_][A-Za-z0-9_]*$")
#: A literal device list: 0, 1, 2, all, cuda:0, GPU-<uuid>, ...
_LITERAL_DEVICE_RE = re.compile(r"^(?:all|\d+(?:,\d+)*|cuda:\d+|GPU-[0-9a-fA-F\-]+)$")
#: A bare Python identifier / attribute path (an unresolved expression).
_PY_EXPR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _strip_trailing_dots(path: str) -> str:
    return path.rstrip(".")


def classify_value(
    raw_value: Optional[str], single_quoted: bool = False
) -> Dict[str, Any]:
    """Classify the right-hand side of a ``CUDA_VISIBLE_DEVICES=`` assignment.

    The bucket semantics are:

    ``DEFEATER``
        The site puts a **non-empty** device list in place of the ambient mask
        whatever the ambient value is.  A literal ``0``/``1``/``cuda:0``, or a
        ``${CVD:-default}`` (the colon form substitutes the default when the
        variable is EMPTY, so the empty lock is lost).

    ``POTENTIAL_DEFEATER``
        The site overwrites the ambient mask with a value that **cannot be
        resolved statically** and does not self-reference the ambient variable
        (``${OTHER}``, ``${OTHER-default}``, ``$OTHER``, a Python expression).
        It defeats the lock whenever the referenced variable is unset or
        non-empty; it cannot be *proven* to do so from the text alone.

    ``LOCK_HONOURING``
        The site **preserves an empty ambient value**: an empty literal
        (``""``), ``${CVD}``, ``${CVD-default}`` (the *no-colon* POSIX form
        substitutes the default only when the variable is UNSET), or
        ``${CVD:-}`` (a colon with an EMPTY default).

    ``NEUTRAL_PRESET``
        The value is a literal that is not a device list (for example a
        single-quoted ``'$_AMBIENT_CVD'``, which bash assigns verbatim), so it
        cannot put a GPU back on.

    The no-colon form ``${VAR-default}`` is the lock-preserving POSIX spelling
    and the colon form ``${VAR:-default}`` is not — that distinction is the
    whole point of the repair, so it is decided here, once.
    """
    if raw_value is None:
        return {
            "bucket": "NOT_AN_ASSIGNMENT",
            "reason": "no_value_token",
            "self_reference": False,
            "defeats_ambient_empty_mask": False,
        }
    value = _strip_quotes(raw_value)
    if value == "":
        return {
            "bucket": "LOCK_HONOURING",
            "reason": "forces_blank_mask",
            "self_reference": False,
            "defeats_ambient_empty_mask": False,
        }
    if single_quoted:
        # bash assigns a single-quoted value VERBATIM: no expansion happens.
        if _LITERAL_DEVICE_RE.match(value):
            return {
                "bucket": "DEFEATER",
                "reason": "single_quoted_literal_device_list",
                "self_reference": False,
                "defeats_ambient_empty_mask": True,
            }
        return {
            "bucket": "NEUTRAL_PRESET",
            "reason": "single_quoted_literal_is_not_a_device_list",
            "self_reference": False,
            "defeats_ambient_empty_mask": False,
        }

    # A self reference decides the outcome.
    self_ref = _SELF_REF_RE.search(value)
    if self_ref:
        if self_ref.group("sep") == ":-":
            default = _strip_quotes(self_ref.group("def") or "")
            if default == "":
                return {
                    "bucket": "LOCK_HONOURING",
                    "reason": "self_reference_colon_empty_default_preserves_lock",
                    "self_reference": True,
                    "defeats_ambient_empty_mask": False,
                }
            return {
                "bucket": "DEFEATER",
                "reason": "self_reference_colon_default_replaces_empty_lock",
                "self_reference": True,
                "defeats_ambient_empty_mask": True,
            }
        if self_ref.group("sep") == "-":
            return {
                "bucket": "LOCK_HONOURING",
                "reason": "self_reference_no_colon_default_preserves_empty_lock",
                "self_reference": True,
                "defeats_ambient_empty_mask": False,
            }
        # A bare "${CUDA_VISIBLE_DEVICES}" — the value is the ambient one.
        return {
            "bucket": "LOCK_HONOURING",
            "reason": "self_reference_bare_preserves_empty_lock",
            "self_reference": True,
            "defeats_ambient_empty_mask": False,
        }

    if _LITERAL_DEVICE_RE.match(value):
        return {
            "bucket": "DEFEATER",
            "reason": "literal_device_list_replaces_ambient_mask",
            "self_reference": False,
            "defeats_ambient_empty_mask": True,
        }

    # A parameter expansion that is NOT a self reference.
    param = _PARAM_EXP_RE.search(value)
    if param:
        default = _strip_quotes(param.group("def"))
        if param.group("sep") == ":-" and default != "":
            return {
                "bucket": "DEFEATER",
                "reason": "colon_default_of_foreign_var_replaces_empty_lock",
                "self_reference": False,
                "defeats_ambient_empty_mask": True,
            }
        return {
            "bucket": "POTENTIAL_DEFEATER",
            "reason": "foreign_var_expansion_overwrites_ambient_mask",
            "self_reference": False,
            "defeats_ambient_empty_mask": False,
        }
    if _BARE_VAR_RE.match(value):
        return {
            "bucket": "POTENTIAL_DEFEATER",
            "reason": "foreign_bare_var_overwrites_ambient_mask",
            "self_reference": False,
            "defeats_ambient_empty_mask": False,
        }
    if _PY_EXPR_RE.match(value):
        return {
            "bucket": "POTENTIAL_DEFEATER",
            "reason": "unresolved_expression_overwrites_ambient_mask",
            "self_reference": False,
            "defeats_ambient_empty_mask": False,
        }
    return {
        "bucket": "POTENTIAL_DEFEATER",
        "reason": "unresolved_value_overwrites_ambient_mask",
        "self_reference": False,
        "defeats_ambient_empty_mask": False,
    }


# --------------------------------------------------------------------------- #
# Site scanning
# --------------------------------------------------------------------------- #
#: A ``NAME=VALUE`` word at the start of a shell command.
_SHELL_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
#: Shell keywords that introduce an assignment rather than a command.
_SHELL_ASSIGN_KEYWORDS = ("export", "local", "declare", "readonly", "typeset")
#: Tokens that introduce a simple command.  An assignment that follows one
#: of them is still in COMMAND-WORD position, i.e. a real environment
#: assignment.
_COMMAND_INTRODUCERS = frozenset(
    (
        "if",
        "then",
        "else",
        "elif",
        "do",
        "while",
        "until",
        "!",
        "{",
        "time",
        "coproc",
        "select",
    )
)
#: Shell operators that end a simple command.
_SHELL_OPERATORS = frozenset((";", "&", "&&", "|", "||", "(", ")", "{", "}"))
#: Tokeniser: shell operators stay attached, everything else splits on space.
_TOKEN_RE = re.compile(r"[;&|()]+|[^\s;&|()]+")
#: ``os.environ["CUDA_VISIBLE_DEVICES"] = <value>``
_PY_ENV_ASSIGN_RE = re.compile(
    r"os\.environ\s*\[\s*[\"']%s[\"']\s*\]\s*=(?!=)\s*(?P<value>[^\n#]+)"
    % re.escape(CVD)
)
#: ``os.environ.setdefault("CUDA_VISIBLE_DEVICES", <value>)``
_PY_ENV_SETDEFAULT_RE = re.compile(
    r"os\.environ\.setdefault\s*\(\s*[\"']%s[\"']\s*,\s*(?P<value>[^)\n]+)" % re.escape(CVD)
)
#: ``os.environ.get("CUDA_VISIBLE_DEVICES", <value>)`` — a *default*, not an
#: assignment; it never writes the environment.
_PY_ENV_GET_RE = re.compile(
    r"os\.environ\.get\s*\(\s*[\"']%s[\"']\s*,\s*(?P<value>[^)\n]+)" % re.escape(CVD)
)
#: ``os.environ.pop("CUDA_VISIBLE_DEVICES")`` — drops the mask entirely.
_PY_ENV_POP_RE = re.compile(
    r"os\.environ\.pop\s*\(\s*[\"']%s[\"']" % re.escape(CVD)
)
#: YAML key form.
_YAML_KEY_RE = re.compile(
    r"^(?P<indent>\s*)%s\s*:\s*(?P<value>.*?)\s*(?:#.*)?$" % re.escape(CVD), re.MULTILINE
)


def _mask_shell_comments(line: str) -> str:
    """Blank out the comment part of a shell line (quote aware)."""
    out = list(line)
    i = 0
    n = len(line)
    quote: Optional[str] = None
    while i < n:
        ch = line[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            for j in range(i, n):
                out[j] = " "
            break
        i += 1
    return "".join(out)


def _shell_line_regions(line: str) -> List[str]:
    """Tag every character of a shell line with its syntactic region.

    Regions: ``code``, ``dquote``, ``squote``, ``comment``.  An occurrence of
    ``CUDA_VISIBLE_DEVICES`` inside ``dquote``/``squote``/``comment`` is a
    **reference inside a string**, not an assignment — that distinction is
    what keeps ``echo "... CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"`` out
    of the assignment census.
    """
    n = len(line)
    regions = ["code"] * n
    i = 0
    while i < n:
        ch = line[i]
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            for j in range(i, n):
                regions[j] = "comment"
            break
        if ch == "'":
            j = line.find("'", i + 1)
            if j == -1:
                j = n - 1
            for k in range(i, min(j + 1, n)):
                regions[k] = "squote"
            i = j + 1
            continue
        if ch == '"':
            j = i + 1
            while j < n:
                if line[j] == "\\":
                    j += 2
                    continue
                if line[j] == '"':
                    break
                j += 1
            j = min(j, n - 1)
            for k in range(i, min(j + 1, n)):
                regions[k] = "dquote"
            i = j + 1
            continue
        i += 1
    return regions


def _read_shell_value(text: str, start: int) -> Tuple[Optional[str], int]:
    """Read a shell assignment value starting at ``start``.

    Returns ``(raw_value, end_index)``.  ``raw_value`` is ``None`` when the
    value token cannot be read.  Quoted strings and ``${...}`` expansions are
    kept whole.
    """
    i = start
    n = len(text)
    while i < n and text[i] in " \t":
        i += 1
    if i >= n:
        return None, i
    quote: Optional[str] = None
    depth = 0
    buf: List[str] = []
    while i < n:
        ch = text[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "$" and i + 1 < n and text[i + 1] == "{":
            depth = 1
            buf.append(ch)
            i += 1
            continue
        if depth:
            buf.append(ch)
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
            continue
        if ch in " \t\n;&|<>()":
            break
        buf.append(ch)
        i += 1
    return "".join(buf), i


def _in_command_word_position(before: str) -> bool:
    """Would an assignment at this position set the ENVIRONMENT of a command?

    POSIX: ``VAR=value`` sets the environment of the command only when it
    appears in the command-word position (before the command name).  In
    argument position it is an ordinary argv element and the environment is
    left alone.  MEASURED on this host with plain bash::

        $ CUDA_VISIBLE_DEVICES="" bash -c 'setsid nohup env | grep "^CUDA"'
        CUDA_VISIBLE_DEVICES=                 # still EMPTY -> the lock survived
        $ CUDA_VISIBLE_DEVICES="" bash -c 'CUDA_VISIBLE_DEVICES=0 env | grep "^CUDA"'
        CUDA_VISIBLE_DEVICES=0                # the prefix DID override it

    ``before`` is the text that precedes the assignment token on its line.
    """
    stripped = before.strip()
    if not stripped:
        return True
    tokens = _TOKEN_RE.findall(stripped)
    if not tokens:
        return True
    last = tokens[-1]
    if last in _SHELL_OPERATORS or last in _COMMAND_INTRODUCERS:
        return True
    # Another assignment keeps us in the command-word prefix
    # (``FOO=1 CUDA_VISIBLE_DEVICES=0 cmd``).
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", last):
        return True
    return False


def _shell_sites(text: str, path: str) -> List[Dict[str, Any]]:
    """Line-oriented census of CVD assignments in a shell script.

    Precision bound (stated so the coverage claim is auditable): this scanner
    resolves the *value* of an assignment exactly, and the *form* of the
    assignment (``export`` / command prefix / argument position / plain) at
    line granularity.  It does not join backslash-continued logical lines, and
    it does not model here-documents; a continued assignment whose ``=`` sits
    on the next physical line is reported at that line.

    Argument position is treated as INERT (it does not touch the
    environment) — see :func:`_in_command_word_position` for the measured
    bash semantics.  Stated limitation: a wrapper that re-interprets argv
    assignments as environment assignments (``sudo VAR=value cmd``) would be
    a false negative here.  No tracked shell script uses ``sudo``
    (measured: ``git ls-files '*.sh' | xargs grep -ln sudo`` returns nothing).
    """
    sites: List[Dict[str, Any]] = []
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        if CVD not in raw_line:
            continue
        regions = _shell_line_regions(raw_line)
        for match in re.finditer(re.escape(CVD), raw_line):
            start = match.start()
            end = match.end()
            region = regions[start]
            # The token must not be part of a longer identifier and must not
            # be a ``${CVD...}`` expansion (that is a *reference*).
            if start > 0 and (
                raw_line[start - 1].isalnum() or raw_line[start - 1] in "_$"
            ):
                continue
            if region != "code":
                sites.append(
                    {
                        "file": path,
                        "line": lineno,
                        "column": start + 1,
                        "lang": "shell",
                        "form": "REFERENCE",
                        "value_raw": None,
                        "bucket": "NOT_AN_ASSIGNMENT",
                        "reason": "occurrence_inside_%s" % region,
                        "text": raw_line.strip()[:200],
                    }
                )
                continue
            after = raw_line[end:]
            eq = re.match(r"[ \t]*=(?!=)", after)
            if not eq:
                sites.append(
                    {
                        "file": path,
                        "line": lineno,
                        "column": start + 1,
                        "lang": "shell",
                        "form": "REFERENCE",
                        "value_raw": None,
                        "bucket": "NOT_AN_ASSIGNMENT",
                        "reason": "no_assignment_operator",
                        "text": raw_line.strip()[:200],
                    }
                )
                continue
            value_start = end + eq.end()
            raw_value, value_end = _read_shell_value(raw_line, value_start)
            before = raw_line[:start]
            keyword = None
            for kw in _SHELL_ASSIGN_KEYWORDS:
                if re.search(r"(?:^|\s|\|)%s\s+$" % re.escape(kw), before):
                    keyword = kw
                    break
            rest = raw_line[value_end:].strip()
            if keyword is not None:
                form = "EXPORT_ASSIGNMENT" if keyword == "export" else "LOCAL_ASSIGNMENT"
            elif not _in_command_word_position(before):
                # ``setsid nohup CUDA_VISIBLE_DEVICES=0 cmd`` — the assignment
                # is an argv element, so the environment (and the ambient
                # lock) is untouched.
                form = "ARG_POSITION_ASSIGNMENT"
            elif rest and not rest.startswith("#"):
                form = "COMMAND_PREFIX"
            else:
                form = "PLAIN_ASSIGNMENT"
            single_quoted = bool(raw_value) and raw_value.lstrip().startswith("'")
            info = classify_value(raw_value, single_quoted=single_quoted)
            if form == "ARG_POSITION_ASSIGNMENT":
                # The value never reaches the environment, so a defeater
                # value in argv position cannot defeat the lock.
                info = dict(info)
                info["bucket"] = "NOT_AN_ASSIGNMENT"
                info["reason"] = (
                    "assignment_in_argument_position_is_not_an_env_assignment"
                )
                info["defeats_ambient_empty_mask"] = False
            sites.append(
                {
                    "file": path,
                    "line": lineno,
                    "column": start + 1,
                    "lang": "shell",
                    "form": form,
                    "value_raw": raw_value,
                    "bucket": info["bucket"],
                    "reason": info["reason"],
                    "self_reference": info["self_reference"],
                    "defeats_ambient_empty_mask": info["defeats_ambient_empty_mask"],
                    "text": raw_line.strip()[:200],
                }
            )
    return sites


def _python_sites(text: str, path: str) -> List[Dict[str, Any]]:
    sites: List[Dict[str, Any]] = []
    for match in _PY_ENV_ASSIGN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        info = classify_value(match.group("value").strip())
        sites.append(
            {
                "file": path,
                "line": line,
                "column": match.start() + 1,
                "lang": "python",
                "form": "ENV_ASSIGN",
                "value_raw": match.group("value").strip(),
                "bucket": info["bucket"],
                "reason": info["reason"],
                "self_reference": info["self_reference"],
                "defeats_ambient_empty_mask": info["defeats_ambient_empty_mask"],
                "text": match.group(0).strip()[:200],
            }
        )
    for match in _PY_ENV_SETDEFAULT_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        info = classify_value(match.group("value").strip())
        sites.append(
            {
                "file": path,
                "line": line,
                "column": match.start() + 1,
                "lang": "python",
                "form": "ENV_SETDEFAULT",
                "value_raw": match.group("value").strip(),
                "bucket": info["bucket"],
                "reason": info["reason"],
                "self_reference": info["self_reference"],
                "defeats_ambient_empty_mask": info["defeats_ambient_empty_mask"],
                "text": match.group(0).strip()[:200],
            }
        )
    for match in _PY_ENV_GET_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        info = classify_value(match.group("value").strip())
        sites.append(
            {
                "file": path,
                "line": line,
                "column": match.start() + 1,
                "lang": "python",
                "form": "ENV_GET_DEFAULT",
                "value_raw": match.group("value").strip(),
                "bucket": "NEUTRAL_PRESET",
                "reason": "read_only_default_never_writes_the_environment",
                "self_reference": info["self_reference"],
                "defeats_ambient_empty_mask": False,
                "text": match.group(0).strip()[:200],
            }
        )
    for match in _PY_ENV_POP_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        sites.append(
            {
                "file": path,
                "line": line,
                "column": match.start() + 1,
                "lang": "python",
                "form": "ENV_POP",
                "value_raw": None,
                "bucket": "DEFEATER",
                "reason": "pops_the_ambient_mask_entirely",
                "self_reference": False,
                "defeats_ambient_empty_mask": True,
                "text": match.group(0).strip()[:200],
            }
        )
    return sites


def _yaml_sites(text: str, path: str) -> List[Dict[str, Any]]:
    sites: List[Dict[str, Any]] = []
    for match in _YAML_KEY_RE.finditer(text):
        raw_value = match.group("value").strip()
        if raw_value.startswith("&") or raw_value.startswith("*"):
            continue
        info = classify_value(raw_value)
        line = text.count("\n", 0, match.start()) + 1
        sites.append(
            {
                "file": path,
                "line": line,
                "column": match.start() + 1,
                "lang": "yaml",
                "form": "YAML_KEY",
                "value_raw": raw_value,
                "bucket": info["bucket"],
                "reason": info["reason"],
                "self_reference": info["self_reference"],
                "defeats_ambient_empty_mask": info["defeats_ambient_empty_mask"],
                "text": match.group(0).strip()[:200],
            }
        )
    return sites


def scan_text(text: str, path: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the CVD sites of one file, dispatching on its language."""
    ext = os.path.splitext(path)[1].lower()
    if lang is None:
        lang = {
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
            ".py": "python",
            ".yaml": "yaml",
            ".yml": "yaml",
        }.get(ext, "text")
    if lang == "shell":
        return _shell_sites(text, path)
    if lang == "python":
        return _python_sites(text, path)
    if lang == "yaml":
        return _yaml_sites(text, path)
    return []


# --------------------------------------------------------------------------- #
# Archival-declaration evidence
# --------------------------------------------------------------------------- #
#: "archived under TODO/archive/wf437_simplify/"  — an EXTERNAL declaration
#: that a tree is archival.  This is positive evidence of intent.  The tree
#: must ALSO be a directory that actually exists in the tracked tree, so a
#: sentence that happens to contain the word "archived" cannot invent an
#: archive.
_ARCHIVE_TREE_DECL_RE = re.compile(
    r"\barchiv\w*\s+(?:under|into|in|to)\s+[`'\"\*]*"
    r"(?P<tree>[A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)*/?)",
    re.IGNORECASE,
)
#: Terms that mark an in-tree marker file as an archival declaration.
_IN_TREE_ARCHIVE_TERM_RE = re.compile(
    r"\b(retired|retirement|archive|archived|archival|deprecated|obsolete"
    r"|do[- ]not\s+run|read[- ]only\s+historical)\b",
    re.IGNORECASE,
)
#: Terms that mark a sibling directory as archival in a directory index.
_SIBLING_ARCHIVE_TERM_RE = re.compile(
    r"\b(retired|retirement|archive|archived|archival|deprecated|obsolete"
    r"|stale|superseded|no[- ]longer\s+relevant|do[- ]not\s+run)\b",
    re.IGNORECASE,
)
#: A directory index is a tracked ``README*`` / ``INDEX*`` file.
_INDEX_NAMES = ("readme", "index")
#: A directory name that stands alone on a line (not a path prefix of a
#: longer name): ``archive/`` yes, ``TODO/03_baselines/MOCK.md`` no.
_STANDALONE_DIR_RE = re.compile(
    r"(?<![\w/.\-])(?P<name>[A-Za-z0-9_\-]+)/(?![\w.\-])"
)
#: An annotation that separates a directory listing from its description.
_ANNOTATION_RE = re.compile(r"(←|->|—|–|:)")

#: How far after the directory name an archival term may sit (characters).
_SIBLING_TERM_WINDOW = 200


def _read_text(path: str) -> Tuple[Optional[str], str]:
    """Read a file as UTF-8.  Returns ``(text, status)``."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None, "missing"
    if size > MAX_FILE_BYTES:
        return None, "too_large"
    try:
        with open(path, "rb") as raw:
            blob = raw.read()
    except OSError:
        return None, "unreadable"
    try:
        return blob.decode("utf-8"), "ok"
    except UnicodeDecodeError:
        return None, "non_utf8"


def _git_ls_files(repo_root: str) -> Tuple[List[str], List[str]]:
    """``git ls-files -z`` -> ``(paths, errors)``."""
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - git missing
        return [], ["git_ls_files_failed: %s" % exc]
    if proc.returncode != 0:
        return [], ["git_ls_files_exit_%d" % proc.returncode]
    paths = [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]
    return paths, []


def _git_ls_files_modes(repo_root: str) -> Dict[str, str]:
    """``git ls-files -s -z`` -> ``{path: mode}`` (the git index mode)."""
    modes: Dict[str, str] = {}
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "ls-files", "-s", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git missing
        return modes
    if proc.returncode != 0:
        return modes
    for record in proc.stdout.decode("utf-8", "replace").split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if len(parts) >= 2 and path:
            modes[path] = parts[0]
    return modes


def _git_head(repo_root: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git missing
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace").strip() or None


def _is_text_candidate(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return True
    # Extension-less tracked files (Makefile, LICENSE, CODEOWNERS, ...).
    return ext == ""


def _collect_archive_declarations(
    repo_root: str,
    tracked: Sequence[str],
    texts: Dict[str, str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Find the trees that tracked text DECLARES archival.

    Three independent kinds of positive evidence, all content-based:

    * **in-tree marker** — a tracked ``README*`` inside the tree whose TITLE
      BLOCK (first five non-empty lines) carries an archival term
      (``scripts/_retired/README.md``: "# Retired launch wrappers").
    * **external declaration** — tracked text that says a tree was "archived
      under <tree>/" (``molmetal/tests/test_wf328w2_launcher_num_workers.py``).
      The named tree must exist as a directory in the tracked tree.
    * **sibling-index declaration** — a tracked ``README*`` / ``INDEX*`` in
      directory D that names one of D's own immediate subdirectories together
      with an archival term on the same line (``TODO/README.md``:
      "``archive/`` ... 4 stale plans superseded by newer ones").

    Returns ``(archive_trees, archive_paths)``.
    """
    archive_trees: Dict[str, List[Dict[str, Any]]] = {}
    archive_paths: Dict[str, List[Dict[str, Any]]] = {}

    # The set of directories that actually exist in the tracked tree.
    directories: Set[str] = set()
    for path in tracked:
        parts = path.split("/")
        for i in range(1, len(parts)):
            directories.add("/".join(parts[:i]) + "/")

    for path in tracked:
        text = texts.get(path)
        if text is None:
            continue
        base = os.path.basename(path)
        directory = os.path.dirname(path)
        stem = base.split(".")[0].lower()

        # (1) in-tree marker — the archival term must appear in the TITLE
        # LINE of the marker file.  A term buried in prose ("Archive
        # SHA256: ...") is a data label, not a declaration, and a heading
        # like "## Archived Work" inside a PARENT's index refers to that
        # parent's ``archive/`` subdirectory, not to the parent itself.
        if stem.startswith("readme") and directory:
            title_lines = [ln for ln in text.splitlines() if ln.strip()]
            candidate = title_lines[0] if title_lines else ""
            if candidate and len(candidate) <= 120:
                match = _IN_TREE_ARCHIVE_TERM_RE.search(candidate)
                if match:
                    tree = directory + "/"
                    archive_trees.setdefault(tree, []).append(
                        {
                            "file": path,
                            "kind": "in_tree_marker",
                            "term": match.group(0),
                            "quote": candidate.strip()[:240],
                        }
                    )

        # (3) sibling-index declaration — an index that names its own
        # immediate subdirectory together with an archival term.  The name
        # must be a STANDALONE token (so a *file* inside a directory does
        # not count), an annotation marker must separate it from the term,
        # and the resulting tree must exist in the tracked tree.
        if stem in _INDEX_NAMES and directory:
            for line in text.splitlines():
                if len(line) > 300:
                    continue
                for candidate in _STANDALONE_DIR_RE.finditer(line):
                    name = candidate.group(1)
                    if not name or name.lower() in _INDEX_NAMES:
                        continue
                    tail = line[candidate.end():]
                    term = _SIBLING_ARCHIVE_TERM_RE.search(tail)
                    if not term:
                        continue
                    if not _ANNOTATION_RE.search(tail[: term.start()]):
                        continue
                    tree = directory + "/" + name + "/"
                    if tree not in directories:
                        continue
                    archive_trees.setdefault(tree, []).append(
                        {
                            "file": path,
                            "kind": "sibling_index_declaration",
                            "term": term.group(0),
                            "quote": line.strip()[:240],
                        }
                    )

        # (2) external declaration.
        for match in _ARCHIVE_TREE_DECL_RE.finditer(text):
            tree = _strip_trailing_dots(match.group("tree"))
            if not tree or "/" not in tree:
                continue
            if not tree.endswith("/"):
                tree += "/"
            if tree not in directories:
                continue
            archive_trees.setdefault(tree, []).append(
                {
                    "file": path,
                    "kind": "external_declaration",
                    "term": match.group(0),
                    "quote": _quote_line(text, match.start()),
                }
            )
    return archive_trees, archive_paths


def _quote_line(text: str, index: int) -> str:
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    if end == -1:
        end = len(text)
    return text[start:end].strip()[:240]


#: Suffixes that mark a tracked file as something a machine can run.
_EXECUTABLE_SUFFIXES = (".sh", ".bash", ".zsh", ".py")


def _is_executable_like(path: str, modes: Dict[str, str]) -> bool:
    """Could this file be RUN?  (git mode 100755, or a script suffix.)"""
    if modes.get(path) == "100755":
        return True
    return path.endswith(_EXECUTABLE_SUFFIXES)


def build_liveness_index(repo_root: str) -> Dict[str, Any]:
    """Build the evidence index used by :func:`classify_liveness`.

    The index is built in three passes so it is not circular:

    1. archival declarations (trees / paths) — content evidence only;
    2. "declared archived" per file, using pass 1 alone;
    3. "live references" — full-path references from files that pass 2 says
       are NOT archived.  Only an EXECUTABLE target can be revived this
       way: a live document that mentions an archived document does not
       make it runnable again, but a live script that still calls an
       archived launcher does.

    The final verdict is ``declared_archived AND NOT live_referenced``.
    """
    tracked, errors = _git_ls_files(repo_root)
    modes = _git_ls_files_modes(repo_root)
    texts: Dict[str, str] = {}
    read_status: Dict[str, str] = {}
    for path in tracked:
        if not _is_text_candidate(path):
            continue
        text, status = _read_text(os.path.join(repo_root, path))
        read_status[path] = status
        if text is not None:
            texts[path] = text

    archive_trees, archive_paths = _collect_archive_declarations(
        repo_root, tracked, texts
    )

    def _declared_archived(path: str) -> Optional[Dict[str, Any]]:
        hits = [
            {"tree": tree, "evidence": ev}
            for tree, evs in archive_trees.items()
            if path.startswith(tree)
            for ev in evs
        ]
        if path in archive_paths:
            hits.extend(
                dict(path=path, evidence=ev) for ev in archive_paths[path]
            )
        return hits or None

    declared: Dict[str, List[Dict[str, Any]]] = {}
    for path in tracked:
        hits = _declared_archived(path)
        if hits:
            declared[path] = hits

    # Pass 3: live references (full relative path as a substring of a tracked,
    # non-archived file).  Only an EXECUTABLE target can be revived by a
    # reference: a document that mentions an archived document does not make
    # it live, while a launcher that a live script still calls is live.
    references: Dict[str, List[Dict[str, Any]]] = {}
    candidates = [p for p in tracked if p in texts and p not in declared]
    for target in tracked:
        if target not in declared:
            continue
        if not _is_executable_like(target, modes):
            continue
        for referrer in candidates:
            if referrer == target:
                continue
            if target in texts[referrer]:
                references.setdefault(target, []).append(
                    {"file": referrer, "kind": "full_path_reference"}
                )

    return {
        "repo_root": repo_root,
        "tracked": tracked,
        "modes": modes,
        "texts": texts,
        "read_status": read_status,
        "archive_trees": archive_trees,
        "archive_paths": archive_paths,
        "declared_archived": declared,
        "references": references,
        "errors": errors,
    }


def classify_liveness(path: str, index: Dict[str, Any]) -> Dict[str, Any]:
    """Classify one tracked file as LIVE or ARCHIVED, with its evidence.

    The verdict is:

    * **ARCHIVED** only when the file is covered by a *declaration* (an
      in-tree marker or an external "archived under ..." statement) **and**
      nothing live references it by full path.
    * **LIVE** otherwise — including when no declaration exists at all.

    The default is LIVE on purpose: an unproven "archived" claim is the exact
    failure mode this classifier exists to remove.
    """
    tracked = index["tracked"]
    declared = index["declared_archived"]
    references = index["references"]

    if path not in tracked:
        return {
            "file": path,
            "tracked": False,
            "archived": False,
            "basis": "UNTRACKED",
            "declarations": [],
            "live_references": [],
        }
    declarations = declared.get(path, [])
    live_refs = references.get(path, [])
    archived = bool(declarations) and not live_refs
    if archived:
        basis = "DECLARED_ARCHIVED"
    elif declarations and live_refs:
        basis = "DECLARED_BUT_REFERENCED_LIVE"
    elif live_refs:
        basis = "REFERENCED_BY_TRACKED_FILE"
    else:
        basis = "NO_ARCHIVAL_DECLARATION"
    return {
        "file": path,
        "tracked": True,
        "executable": index["modes"].get(path) == "100755",
        "archived": archived,
        "basis": basis,
        "declarations": declarations,
        "live_references": live_refs,
    }


# --------------------------------------------------------------------------- #
# Census
# --------------------------------------------------------------------------- #
def census(repo_root: str) -> Dict[str, Any]:
    """Full census: every CVD site in the tracked tree + its liveness."""
    index = build_liveness_index(repo_root)
    tracked: List[str] = index["tracked"]
    texts: Dict[str, str] = index["texts"]
    read_status: Dict[str, str] = index["read_status"]

    sites: List[Dict[str, Any]] = []
    files_scanned = 0
    files_skipped_non_utf8: List[str] = []
    files_skipped_too_large: List[str] = []
    files_missing: List[str] = []
    files_not_text_candidate = 0

    for path in tracked:
        if path not in read_status:
            files_not_text_candidate += 1
            continue
        status = read_status[path]
        if status == "non_utf8":
            files_skipped_non_utf8.append(path)
            continue
        if status == "too_large":
            files_skipped_too_large.append(path)
            continue
        if status != "ok":
            files_missing.append(path)
            continue
        files_scanned += 1
        found = scan_text(texts[path], path)
        sites.extend(found)

    by_bucket: Dict[str, int] = {}
    by_form: Dict[str, int] = {}
    by_lang: Dict[str, int] = {}
    for site in sites:
        by_bucket[site["bucket"]] = by_bucket.get(site["bucket"], 0) + 1
        by_form[site["form"]] = by_form.get(site["form"], 0) + 1
        by_lang[site["lang"]] = by_lang.get(site["lang"], 0) + 1

    defeaters = [
        s
        for s in sites
        if s["bucket"] in ("DEFEATER", "POTENTIAL_DEFEATER")
    ]

    # Liveness is reported for EVERY file that carries a CVD site, not only
    # the defeat ones, so the artifact can answer "is this file live?" for
    # any site the guard reports.
    liveness: Dict[str, Any] = {}
    for path in sorted({s["file"] for s in sites}):
        liveness[path] = classify_liveness(path, index)

    live_defeaters = [
        s for s in defeaters if not liveness.get(s["file"], {}).get("archived", False)
    ]
    archived_defeaters = [
        s for s in defeaters if liveness.get(s["file"], {}).get("archived", False)
    ]

    per_site = []
    for site in sites:
        site_live = liveness.get(site["file"])
        per_site.append(
            {
                "file": site["file"],
                "line": site["line"],
                "column": site["column"],
                "lang": site["lang"],
                "form": site["form"],
                "bucket": site["bucket"],
                "reason": site["reason"],
                "value_raw": site.get("value_raw"),
                "archived": bool(
                    site_live and site_live.get("archived", False)
                ),
                "liveness_basis": site_live["basis"] if site_live else "NO_DEFEAT_SITE",
                "text": site["text"],
            }
        )

    archive_trees = {
        tree: evs for tree, evs in sorted(index["archive_trees"].items())
    }
    return {
        "tool": "molmetal/scripts/wfa171_gpu_lock_guard_path.py",
        "wf": "A171",
        "title": "evidence-based live/archived classification for the GPU-lock guard",
        "read_only": True,
        "cpu_only": True,
        "torch_free": True,
        "snapshot": {
            "git_head": _git_head(repo_root),
            "repo_root": repo_root,
        },
        "method": {
            "archive_rule": (
                "ARCHIVED iff a tracked text DECLARES the file/tree archival "
                "(in-tree README marker or an 'archived under <tree>/' "
                "statement) AND no tracked non-archived file references it by "
                "full path; otherwise LIVE."
            ),
            "default_polarity": "LIVE (an unproven 'archived' claim is a bug)",
            "value_rule": (
                "DEFEATER = the site installs a non-empty device list whatever "
                "the ambient value is; LOCK_HONOURING = an empty ambient value "
                "survives (empty literal, ${CVD}, ${CVD-default}, ${CVD:-}); "
                "POTENTIAL_DEFEATER = the value is an unresolved foreign "
                "variable that overwrites the ambient mask."
            ),
            "scan_scope": "tracked files with a text extension; per-file UTF-8 "
            "decode with explicit skip accounting",
        },
        "denominator": {
            "tracked_files_total": len(tracked),
            "text_candidate_files": len(read_status),
            "files_scanned": files_scanned,
            "files_skipped_non_utf8": len(files_skipped_non_utf8),
            "files_skipped_too_large": len(files_skipped_too_large),
            "files_missing_on_disk": len(files_missing),
            "files_not_text_candidate": files_not_text_candidate,
            "non_utf8_paths": files_skipped_non_utf8[:50],
            "too_large_paths": files_skipped_too_large[:50],
            "missing_paths": files_missing[:50],
            "enumeration_errors": index["errors"],
        },
        "counts": {
            "sites_total": len(sites),
            "by_bucket": by_bucket,
            "by_form": by_form,
            "by_lang": by_lang,
            "defeaters_total": len(defeaters),
            "defeaters_proven": sum(
                1 for s in defeaters if s["bucket"] == "DEFEATER"
            ),
            "defeaters_potential": sum(
                1 for s in defeaters if s["bucket"] == "POTENTIAL_DEFEATER"
            ),
            "defeaters_live": len(live_defeaters),
            "defeaters_archived": len(archived_defeaters),
            "files_with_defeaters": len(liveness),
        },
        "declared_archive_trees": archive_trees,
        "liveness": liveness,
        "live_defeater_sites": [
            {
                "file": s["file"],
                "line": s["line"],
                "column": s["column"],
                "lang": s["lang"],
                "form": s["form"],
                "bucket": s["bucket"],
                "value_raw": s.get("value_raw"),
                "reason": s["reason"],
                "liveness_basis": liveness.get(s["file"], {}).get("basis"),
            }
            for s in live_defeaters
        ],
        "archived_defeater_sites": [
            {
                "file": s["file"],
                "line": s["line"],
                "column": s["column"],
                "lang": s["lang"],
                "form": s["form"],
                "bucket": s["bucket"],
                "value_raw": s.get("value_raw"),
                "reason": s["reason"],
                "liveness_basis": liveness.get(s["file"], {}).get("basis"),
            }
            for s in archived_defeaters
        ],
        "sites": per_site,
        "self_check": {
            "sites_equal_sum_of_buckets": sum(by_bucket.values()) == len(sites),
            "defeater_split_sums": (
                sum(1 for s in defeaters if s["bucket"] == "DEFEATER")
                + sum(1 for s in defeaters if s["bucket"] == "POTENTIAL_DEFEATER")
                == len(defeaters)
            ),
            "live_plus_archived_equals_total": (
                len(live_defeaters) + len(archived_defeaters) == len(defeaters)
            ),
            "no_path_prefix_archive_rule": (
                "results/" not in index["archive_trees"]
                and "TODO/archive/" in index["archive_trees"]
            ),
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _render_human(report: Dict[str, Any], max_print: int = 60) -> str:
    lines: List[str] = []
    d = report["denominator"]
    c = report["counts"]
    lines.append("Wf-A171 GPU-lock guard — evidence-based liveness census")
    lines.append(
        "DENOMINATOR: %d tracked files, %d text candidates, %d scanned, "
        "%d non-utf8, %d too large, %d missing"
        % (
            d["tracked_files_total"],
            d["text_candidate_files"],
            d["files_scanned"],
            d["files_skipped_non_utf8"],
            d["files_skipped_too_large"],
            d["files_missing_on_disk"],
        )
    )
    lines.append(
        "SITES: %d total | by bucket: %s"
        % (c["sites_total"], json.dumps(c["by_bucket"], sort_keys=True))
    )
    lines.append(
        "DEFEATERS: %d (%d proven / %d potential) | LIVE %d | ARCHIVED %d"
        % (
            c["defeaters_total"],
            c["defeaters_proven"],
            c["defeaters_potential"],
            c["defeaters_live"],
            c["defeaters_archived"],
        )
    )
    lines.append("DECLARED ARCHIVE TREES:")
    for tree, evs in report["declared_archive_trees"].items():
        lines.append(
            "  %s  <- %s" % (tree, "; ".join("%s (%s)" % (e["file"], e["kind"]) for e in evs[:3]))
        )
    lines.append("LIVE DEFEAT SITES:")
    for site in report["live_defeater_sites"][:max_print]:
        lines.append(
            "  %s:%s:%s  %s  %s  value=%r"
            % (
                site["file"],
                site["line"],
                site["column"],
                site["bucket"],
                site["form"],
                site["value_raw"],
            )
        )
    if len(report["live_defeater_sites"]) > max_print:
        lines.append("  ... %d more" % (len(report["live_defeater_sites"]) - max_print))
    lines.append("ARCHIVED DEFEAT SITES:")
    for site in report["archived_defeater_sites"][:max_print]:
        lines.append(
            "  %s:%s  %s  value=%r"
            % (site["file"], site["line"], site["bucket"], site["value_raw"])
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json", action="store_true", help="print JSON to stdout")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--max-print", type=int, default=60)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(
        args.repo_root
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    report = census(repo_root)

    if args.out:
        out_path = os.path.abspath(args.out)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, out_path)

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        sys.stdout.write("\n")
    elif not args.quiet:
        sys.stdout.write(_render_human(report, max_print=args.max_print) + "\n")

    # A live defeat site is a finding: exit 1 so the guard is usable in CI.
    return 1 if report["counts"]["defeaters_live"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
