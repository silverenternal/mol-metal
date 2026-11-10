#!/usr/bin/env python3
"""Wf-A167 — independent census of the COMMAND-PREFIX ``CUDA_VISIBLE_DEVICES``
defect class, plus the DENOMINATOR for it.

This tool is deliberately INDEPENDENT of ``wfa165_gpu_lock_guard.py``: it does
not import it, does not reuse its regexes and does not read its output. The
whole point of the A167 branch is an independent re-derivation, so that the
cross-check in STEP 2 is a genuine second opinion rather than a tautology.

Why the class exists
--------------------
The repo's launchers are supposed to honour an *ambient* GPU lock, i.e. an
``CUDA_VISIBLE_DEVICES=""`` exported by the operator (the "no GPU for this
session" lock).  Wf-A165 repaired three ways of defeating it:

  M1  ``export CUDA_VISIBLE_DEVICES=0``               (literal override)
  M2  ``export CUDA_VISIBLE_DEVICES="${VAR:-default}"`` (``:`` replaces an EMPTY
      value with the default, so the empty lock is silently replaced)
  M3  the YAML-derived form in ``scripts/start.sh``

A FOURTH way survives and is *not* an ``export`` line, so the M1/M2 repair
cannot see it:

      CUDA_VISIBLE_DEVICES=0 cmd            # a command-prefix assignment
      if ! CUDA_VISIBLE_DEVICES=0 cmd      # ... after a shell keyword
      if CUDA_VISIBLE_DEVICES=0 cmd

In bash, ``VAR=val cmd`` puts ``VAR=val`` into ``cmd``'s environment, so the
assignment is inherited by the whole child process tree and it *replaces* the
ambient mask.  A command prefix is therefore exactly as strong as an
``export``, but it lives in a different syntactic position.

Bash semantics that the classifier is built on (all verified empirically on
this machine with ``bash``; see the module docstring of the test file):

  * ``VAR=val cmd``            -> assignment applies to ``cmd`` (prefix).
  * ``if VAR=val cmd``         -> assignment applies to ``cmd``.
  * ``if ! VAR=val cmd``       -> assignment applies to ``cmd``.
  * ``cmd VAR=val``            -> ``VAR=val`` is an ARGUMENT to ``cmd``, NOT an
    environment assignment.  ``setsid nohup VAR=val cmd`` therefore fails
    outright ("nohup: failed to run command 'VAR=val'").  This is the sharp
    edge the A165 guard's line-start anchor cannot see.
  * ``${VAR-default}``         -> substitutes only when VAR is UNSET, so an
    explicitly empty (locked) value survives.  LOCK-PRESERVING.
  * ``${VAR:-default}``        -> substitutes when VAR is unset OR EMPTY, so the
    empty lock is replaced by the default.  LOCK-DEFEATING.

What this tool measures
-----------------------
Every occurrence of the token ``CUDA_VISIBLE_DEVICES`` in every tracked
``*.sh`` file is located and classified by syntactic shape:

  COMMAND_PREFIX          ``VAR=val`` in command-prefix position (defeats)
  ARG_POSITION_ASSIGNMENT ``VAR=val`` after the command word (an ARGUMENT;
                          not an env assignment at all)
  EXPORT_ASSIGNMENT      ``export VAR=val`` / ``local`` / ``declare`` / ...
  PLAIN_ASSIGNMENT       bare ``VAR=val`` statement
  QUOTED_STRING          occurrence inside a double/single quoted string
                          (e.g. ``log "  CUDA_VISIBLE_DEVICES=0 ..."``)
  TEST                   occurrence inside a ``[ ... ]`` / ``[[ ... ]]`` test
  COMMENT                occurrence inside a ``#`` comment
  EXPANSION              ``${CUDA_VISIBLE_DEVICES...}`` / ``$CUDA_VISIBLE_DEVICES``
  MENTION                bare token, no assignment, no expansion
  HEREDOC                occurrence inside a here-document body

and by lock semantics:

  REAL_DEFEATER            the site forces a non-empty device mask regardless
                           of the ambient value  -> defeats the empty lock
  ALREADY_LOCK_PRESERVING  the site self-references the ambient value with the
                           no-colon form (``${CVD-V}`` / ``${CVD}``) -> an
                           empty lock survives
  BENIGN                   everything else
  ARCHIVED                 the file is a read-only historical artefact
                           (``TODO/archive/``, ``results/``)

The DENOMINATOR (tracked ``*.sh`` files enumerated / read / unreadable) is
reported alongside the counts so that every number has a denominator.

Hard constraints honoured by this tool
--------------------------------------
* READ-ONLY: it never writes to any file except the report it is asked to
  write, and it never modifies a scanned file.
* CPU-ONLY / stdlib-only: no ``torch``, no ``nvidia-smi``, no GPU calls.
  Enforced by ``molmetal/tests/test_wfa167_command_prefix_guard.py``.
* No ``os.walk``: the file list comes from ``git ls-files -z -- '*.sh'``.

Usage
-----
    python molmetal/scripts/wfa167_command_prefix_guard.py --json
    python molmetal/scripts/wfa167_command_prefix_guard.py \
        --out molmetal/reports/wfa167_command_prefix_guard/census.json

The VERDICT AXIS (Wf-A269 wave-28, 2026-11-06)
------------------------------------------------
The exit code is computed on the **LIVE** axis
(``counts["real_defeaters_live"]``).  A site under ``TODO/archive/`` is
out of scope by this tool's OWN classification (the ``ARCHIVED`` class:
"the file is a read-only historical artefact") and is never launched, so
it must not be able to keep the CLI red forever.  The legacy predicate
counted the archived sites too, which made the CLI **unable to pass**
without ``--advisory`` — a gate that cannot PASS (the §6.269.17 class).
The strict axis (archived sites included) is still reachable with
``--include-archived``; ``--advisory`` still exits 0 unconditionally.
Both counts are always reported in the report and in the human rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CVD",
    "ARCHIVE_PREFIXES",
    "scan_shell_text",
    "scan_repository",
    "classify_value",
    "lock_preserving_form",
]

CVD = "CUDA_VISIBLE_DEVICES"

#: Read-only historical artefacts.  A site inside one of these trees is
#: reported with class ``ARCHIVED`` (its semantic class is still recorded in
#: ``semantic_class``).
ARCHIVE_PREFIXES: Tuple[str, ...] = (
    "TODO/archive/",
    "results/",
)

#: Words that may precede a simple command and therefore do not terminate the
#: command-prefix region.
RESERVED_PREFIX: frozenset = frozenset(
    {
        "if",
        "elif",
        "while",
        "until",
        "do",
        "then",
        "else",
        "time",
        "coproc",
        "select",
        "function",
        "in",
        "case",
        "esac",
        "fi",
        "done",
    }
)

#: Builtins that consume an assignment word as part of their own syntax.
ASSIGNMENT_BUILTINS: frozenset = frozenset(
    {"export", "local", "readonly", "typeset", "declare"}
)

#: ``env`` consumes trailing ``VAR=val`` words and exports them.
ENV_BUILTIN = "env"

_WORD_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: A self reference to the ambient mask.  The negative lookbehind makes sure
#: that ``${WF310C_CUDA_VISIBLE_DEVICES:-1}`` is NOT treated as a self
#: reference (the preceding ``_`` is a word character).
_SELF_REF_RE = re.compile(r"(?<![A-Za-z0-9_])\$\{?CUDA_VISIBLE_DEVICES(?P<sep>:-|-|\}|\b)")

#: Operators, longest first.
_OPS: Tuple[str, ...] = (
    ";;&",
    "<<-",
    "&&",
    "||",
    ";;",
    ";&",
    ">>",
    "&>>",
    "&>",
    ">&",
    "<&",
    ">|",
    "<<",
    "[[",
    "]]",
    "((",
    "))",
    "<>",
    "|",
    ";",
    "&",
    "<",
    ">",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    "!",
)

#: Operators that terminate the current command (the command-prefix region
#: restarts after them).
_SEPARATORS = frozenset({";;&", "&&", "||", ";;", ";&", "|", ";", "&"})

_OP_CHARS = frozenset(";|&<>(){}[]!")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _is_ident(text: str) -> bool:
    return bool(text) and bool(_IDENT_RE.match(text))


def _strip_quotes(text: str) -> str:
    """Remove one layer of surrounding quotes, if present."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _trailing_backslash(line: str) -> bool:
    """True if ``line`` ends with an odd number of backslashes (line join)."""
    stripped = line.rstrip()
    if not stripped.endswith("\\"):
        return False
    n = 0
    for ch in reversed(stripped):
        if ch == "\\":
            n += 1
        else:
            break
    return n % 2 == 1


# --------------------------------------------------------------------------- #
# Lexer
# --------------------------------------------------------------------------- #
class _Word:
    """One shell word (or operator), with just enough structure to reason."""

    __slots__ = ("text", "name", "value", "quoted", "start", "end", "is_op")

    def __init__(self, text: str, name: Optional[str], value: Optional[str],
                 quoted: bool, start: int, end: int, is_op: bool = False) -> None:
        self.text = text
        self.name = name          # identifier before the first top-level '='
        self.value = value        # raw text after that '=' (None if no '=')
        self.quoted = quoted      # the whole word was quoted
        self.start = start
        self.end = end
        self.is_op = is_op


def _scan_dquote(text: str, i: int, kinds: List[str]) -> int:
    """Return the index just past the closing double quote of ``text[i]``."""
    n = len(text)
    j = i + 1
    depth = 0
    while j < n:
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "$" and j + 1 < n and text[j + 1] == "(":
            depth += 1
            j += 2
            continue
        if ch == ")" and depth > 0:
            depth -= 1
            j += 1
            continue
        if ch == '"' and depth == 0:
            break
        j += 1
    end = min(j + 1, n)
    for k in range(i, end):
        if kinds[k] == "code":
            kinds[k] = "dquote"
    return end


def _scan_squote(text: str, i: int, kinds: List[str]) -> int:
    n = len(text)
    j = text.find("'", i + 1)
    if j < 0:
        j = n - 1
    end = min(j + 1, n)
    for k in range(i, end):
        if kinds[k] == "code":
            kinds[k] = "squote"
    return end


def _scan_cmdsub(text: str, i: int, kinds: List[str]) -> int:
    """Scan ``$( ... )`` (or backticks) starting at ``text[i]``; return the end."""
    n = len(text)
    if text[i] == "`":
        j = text.find("`", i + 1)
        if j < 0:
            j = n - 1
        end = min(j + 1, n)
        for k in range(i, end):
            if kinds[k] == "code":
                kinds[k] = "cmdsub"
        return end
    # $( ... )
    depth = 0
    j = i + 1
    while j < n:
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "'":
            j = _scan_squote(text, j, kinds)
            continue
        if ch == '"':
            j = _scan_dquote(text, j, kinds)
            continue
        if ch == "$" and j + 1 < n and text[j + 1] == "(":
            depth += 1
            j += 2
            continue
        if ch == ")":
            depth -= 1
            j += 1
            if depth == 0:
                break
            continue
        j += 1
    end = min(j, n)
    for k in range(i, end):
        if kinds[k] == "code":
            kinds[k] = "cmdsub"
    return end


def _read_word(text: str, i: int, kinds: List[str]) -> Tuple[_Word, int]:
    """Read one shell word starting at ``text[i]``."""
    n = len(text)
    parts: List[str] = []
    pre_eq_plain = ""
    pre_eq_quoted = False
    name: Optional[str] = None
    value: Optional[str] = None
    seen_eq = False
    whole_quoted = False
    first = True
    j = i
    while j < n:
        ch = text[j]
        if ch in " \t" or ch in _OP_CHARS:
            break
        if ch == "\\":
            if j + 1 < n:
                if not seen_eq:
                    pre_eq_plain += text[j + 1]
                parts.append(text[j : j + 2])
                j += 2
                continue
            j += 1
            break
        if ch == "'":
            end = _scan_squote(text, j, kinds)
            if not seen_eq:
                pre_eq_quoted = True
            parts.append(text[j:end])
            if first:
                whole_quoted = True
            first = False
            j = end
            continue
        if ch == '"':
            end = _scan_dquote(text, j, kinds)
            if not seen_eq:
                pre_eq_quoted = True
            parts.append(text[j:end])
            if first:
                whole_quoted = True
            first = False
            j = end
            continue
        if ch == "$" and j + 1 < n and text[j + 1] == "(":
            end = _scan_cmdsub(text, j, kinds)
            if not seen_eq:
                pre_eq_quoted = True
            parts.append(text[j:end])
            first = False
            j = end
            continue
        if ch == "`":
            end = _scan_cmdsub(text, j, kinds)
            if not seen_eq:
                pre_eq_quoted = True
            parts.append(text[j:end])
            first = False
            j = end
            continue
        if ch == "=" and not seen_eq:
            seen_eq = True
            candidate = pre_eq_plain
            if _is_ident(candidate) and not pre_eq_quoted:
                name = candidate
            parts.append("=")
            j += 1
            continue
        if not seen_eq:
            pre_eq_plain += ch
        parts.append(ch)
        first = False
        j += 1
    if seen_eq:
        value = text[i:j]
        eq = value.find("=")
        value = value[eq + 1 :]
    return _Word(text[i:j], name, value, whole_quoted, i, j), j


def _lex(text: str) -> Tuple[List[_Word], List[str], List[bool]]:
    """Lex one logical line.

    Returns ``(words, kinds, in_test)`` where ``kinds[i]`` describes the
    syntactic region of ``text[i]`` and ``in_test[i]`` marks characters that
    live inside a ``[ ... ]`` / ``[[ ... ]]`` conditional test.
    """
    n = len(text)
    kinds: List[str] = ["code"] * n
    in_test: List[bool] = [False] * n
    words: List[_Word] = []
    i = 0
    while i < n:
        ch = text[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "#" and (i == 0 or text[i - 1] in " \t" or _is_op_boundary(text, i)):
            for k in range(i, n):
                kinds[k] = "comment"
            break
        if ch in _OP_CHARS:
            op = _match_op(text, i)
            for k in range(i, min(i + len(op), n)):
                kinds[k] = "op"
            words.append(_Word(op, None, None, False, i, i + len(op), True))
            i += len(op)
            continue
        word, nxt = _read_word(text, i, kinds)
        words.append(word)
        i = nxt
    # mark test regions
    _mark_test_regions(text, words, kinds, in_test)
    return words, kinds, in_test


def _is_op_boundary(text: str, i: int) -> bool:
    """True when ``text[i-1]`` closes a word (so a '#' starts a comment)."""
    return i > 0 and text[i - 1] not in " \t"


def _match_op(text: str, i: int) -> str:
    for op in _OPS:
        if text.startswith(op, i):
            return op
    return text[i]


def _mark_test_regions(text: str, words: List[_Word], kinds: List[str],
                       in_test: List[bool]) -> None:
    """Mark the interiors of ``[[ ... ]]`` / ``[ ... ]`` as test regions."""
    # Re-walk the raw text to find the bracketed regions (word-level walking
    # would miss the fact that '[' is an operator token).
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("[[", i) and kinds[i] == "op":
            close = text.find("]]", i + 2)
            end = n if close < 0 else close + 2
            for k in range(i, end):
                in_test[k] = True
            i = end
            continue
        if text[i] == "[" and kinds[i] == "op" and not text.startswith("[[", i):
            close = text.find("]", i + 1)
            end = n if close < 0 else close + 1
            for k in range(i, end):
                in_test[k] = True
            i = end
            continue
        i += 1


# --------------------------------------------------------------------------- #
# Logical lines
# --------------------------------------------------------------------------- #
def _logical_lines(text: str) -> List[Tuple[int, str, List[int], List[int]]]:
    """Join backslash-continued physical lines.

    Returns one tuple per logical line:
    ``(start_line_no, joined_text, [physical line numbers],
    [chunk start offsets in joined_text])``.

    The joined text keeps the trailing backslash removed and the continuation
    joined with a single space; the chunk offsets let an offset inside the
    joined text be mapped back to the physical line it came from.
    """
    raw = text.split("\n")
    out: List[Tuple[int, str, List[int], List[int]]] = []
    i = 0
    while i < len(raw):
        start = i
        buf = raw[i]
        linenos = [i + 1]
        offsets = [0]
        while _trailing_backslash(buf) and i + 1 < len(raw):
            buf = buf.rstrip()[:-1] + " " + raw[i + 1]
            i += 1
            linenos.append(i + 1)
            offsets.append(len(buf))
        out.append((start + 1, buf, linenos, offsets))
        i += 1
    return out


def _heredoc_body_lines(text: str) -> Dict[int, str]:
    """Map physical line number -> delimiter for here-document bodies."""
    bodies: Dict[int, str] = {}
    raw = text.split("\n")
    idx = 0
    while idx < len(raw):
        line = raw[idx]
        stripped = line.strip()
        if not stripped.startswith("#"):
            delims = re.findall(
                r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", line
            )
            if delims:
                for _quote, delim in delims:
                    indented = ("<<-" in line)
                    probe = idx + 1
                    while probe < len(raw):
                        body = raw[probe]
                        candidate = body.strip() if indented else body
                        if candidate == delim:
                            break
                        bodies[probe + 1] = delim
                        probe += 1
                    idx = max(idx, probe)
        idx += 1
    return bodies


# --------------------------------------------------------------------------- #
# Value semantics
# --------------------------------------------------------------------------- #
def classify_value(raw_value: Optional[str]) -> Dict[str, Any]:
    """Classify the right-hand side of a ``CUDA_VISIBLE_DEVICES=`` assignment.

    Returns a dict with ``effect`` in {``unmask``, ``forces_blank``,
    ``preserves_ambient``, ``none``} and ``defeats_ambient_empty_mask``.
    """
    if raw_value is None:
        return {
            "effect": "none",
            "defeats_ambient_empty_mask": False,
            "reason": "no_value_token",
            "self_reference": False,
        }
    value = _strip_quotes(raw_value)
    if value == "":
        return {
            "effect": "forces_blank",
            "defeats_ambient_empty_mask": False,
            "reason": "forces_blank_mask",
            "self_reference": False,
        }
    match = _SELF_REF_RE.search(value)
    if match:
        sep = match.group("sep")
        if sep == ":-":
            return {
                "effect": "unmask",
                "defeats_ambient_empty_mask": True,
                "reason": "self_reference_colon_default_replaces_empty_lock",
                "self_reference": True,
            }
        # `${CVD-V}` (no colon) or `${CVD}` -> an explicitly empty value survives
        return {
            "effect": "preserves_ambient",
            "defeats_ambient_empty_mask": False,
            "reason": "self_reference_no_colon_default_preserves_empty_lock",
            "self_reference": True,
        }
    return {
        "effect": "unmask",
        "defeats_ambient_empty_mask": True,
        "reason": "hard_override_replaces_ambient_mask",
        "self_reference": False,
    }


def lock_preserving_form(value_expr: str) -> str:
    """Return the lock-preserving bash form for a command-prefix assignment.

    ``value_expr`` is the *original* value expression of the script (e.g. ``0``
    or ``${gpu##cuda:}``).  The repair keeps the script's own default when the
    ambient variable is UNSET and keeps an explicitly empty (locked) ambient
    value.  The no-colon ``${VAR-default}`` form is essential: the colon form
    ``${VAR:-default}`` would replace an empty lock with the default.
    """
    return 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-%s}"' % value_expr


# --------------------------------------------------------------------------- #
# Per-file scan
# --------------------------------------------------------------------------- #
def _split_segments(
    words: Sequence[_Word],
) -> List[Tuple[List[_Word], List[int]]]:
    """Split a word list into simple commands at control operators.

    Returns ``[(segment_words, [global word indices])]`` so that a
    segment-local position can be mapped back to the global word list.
    """
    segments: List[Tuple[List[_Word], List[int]]] = []
    current: List[_Word] = []
    indices: List[int] = []
    for index, word in enumerate(words):
        if word.is_op and word.text in _SEPARATORS:
            segments.append((current, indices))
            current = []
            indices = []
            continue
        current.append(word)
        indices.append(index)
    segments.append((current, indices))
    return segments


def _segment_positions(segment: Sequence[_Word]) -> Dict[int, str]:
    """Classify every assignment word of one simple command.

    Returns ``{word_index: shape}`` where shape is one of
    ``COMMAND_PREFIX`` / ``EXPORT_ASSIGNMENT`` / ``PLAIN_ASSIGNMENT`` /
    ``ARG_POSITION_ASSIGNMENT``.

    Bash rules applied here:

    * ``VAR=val cmd``  -> the assignment is in command-prefix position and
      becomes part of ``cmd``'s environment.
    * ``if`` / ``elif`` / ``while`` / ``until`` / ``then`` / ``else`` / ``do`` /
      ``!`` / ``time`` may precede the command; they do not stop the prefix
      region.
    * ``export`` / ``local`` / ``declare`` / ``readonly`` / ``typeset`` take
      their assignments as arguments, not as a command prefix.
    * ``env`` DOES treat trailing ``VAR=val`` words as a command prefix.
    * ``cmd VAR=val`` -> ``VAR=val`` is an ARGUMENT to ``cmd``; it is not an
      environment assignment at all.
    """
    shapes: Dict[int, str] = {}
    idx = 0
    n = len(segment)
    # Skip leading reserved words / grouping openers.
    while idx < n:
        word = segment[idx]
        if word.is_op:
            if word.text in ("(", "{", "!"):
                idx += 1
                continue
            break
        if not word.quoted and word.text in RESERVED_PREFIX:
            idx += 1
            continue
        break
    assign_idx: List[int] = []
    while idx < n:
        word = segment[idx]
        if word.is_op or word.name is None:
            break
        assign_idx.append(idx)
        idx += 1
    cmd_idx = idx if idx < n else None
    cmd_text = None
    if cmd_idx is not None:
        word = segment[cmd_idx]
        if not word.is_op and not word.quoted:
            cmd_text = word.text
    if cmd_idx is None:
        # A pure assignment statement.
        for k in assign_idx:
            shapes[k] = "PLAIN_ASSIGNMENT"
        return shapes
    if cmd_text in ASSIGNMENT_BUILTINS:
        for k in range(cmd_idx + 1, n):
            word = segment[k]
            if word.is_op or word.name is None:
                break
            shapes[k] = "EXPORT_ASSIGNMENT"
        for k in assign_idx:
            shapes.setdefault(k, "COMMAND_PREFIX")
        return shapes
    if cmd_text == ENV_BUILTIN:
        for k in range(cmd_idx + 1, n):
            word = segment[k]
            if word.is_op or word.name is None:
                break
            shapes[k] = "COMMAND_PREFIX"
        for k in assign_idx:
            shapes.setdefault(k, "COMMAND_PREFIX")
        return shapes
    for k in assign_idx:
        shapes[k] = "COMMAND_PREFIX"
    # Any assignment AFTER the command word is an argument, not an
    # environment assignment.
    for k in range(cmd_idx + 1, n):
        word = segment[k]
        if word.is_op or word.name is None:
            continue
        shapes[k] = "ARG_POSITION_ASSIGNMENT"
    return shapes


def scan_shell_text(text: str, path: str = "<memory>") -> List[Dict[str, Any]]:
    """Scan one shell script and return the list of CVD sites."""
    sites: List[Dict[str, Any]] = []
    heredoc = _heredoc_body_lines(text)
    archived = path.startswith(ARCHIVE_PREFIXES)

    for start_line, logical, linenos, offsets in _logical_lines(text):
        words, kinds, in_test = _lex(logical)

        # Map every word index to its syntactic position inside the simple
        # command it belongs to.
        positions: Dict[int, str] = {}
        command_word: Optional[str] = None
        for segment, global_indices in _split_segments(words):
            for local_index, shape in _segment_positions(segment).items():
                positions[global_indices[local_index]] = shape
            for word in segment:
                if not word.is_op and not word.quoted and word.text in (
                    "setsid",
                    "nohup",
                    "env",
                    "sudo",
                    "export",
                    "local",
                    "declare",
                ):
                    command_word = word.text
                    break
            if command_word:
                break

        for idx, word in enumerate(words):
            if word.name != CVD:
                continue
            shape = positions.get(idx, "ARG_POSITION_ASSIGNMENT")

            phys = _physical_line_for(linenos, offsets, word.start)
            if phys in heredoc:
                shape = "HEREDOC"

            value_info = classify_value(word.value)
            semantic = _semantic_class(shape, value_info)
            klass = "ARCHIVED" if archived else semantic

            sites.append(
                {
                    "file": path,
                    "line": phys,
                    "column": word.start + 1,
                    "shape": shape,
                    "class": klass,
                    "semantic_class": semantic,
                    "reason": value_info["reason"]
                    if shape in ("COMMAND_PREFIX", "EXPORT_ASSIGNMENT", "PLAIN_ASSIGNMENT")
                    else _benign_reason(shape),
                    "effect": value_info["effect"],
                    "defeats_ambient_empty_mask": bool(
                        value_info["defeats_ambient_empty_mask"]
                    )
                    and shape in ("COMMAND_PREFIX", "EXPORT_ASSIGNMENT", "PLAIN_ASSIGNMENT"),
                    "self_reference": value_info["self_reference"],
                    "value_raw": word.value,
                    "command_word": command_word,
                    "text": word.text.strip()[:200],
                    "archived": archived,
                }
            )

        # Non-assignment occurrences (references / mentions / strings / ...).
        for match in re.finditer(re.escape(CVD), logical):
            off = match.start()
            if off in _word_offsets(words):
                continue  # already reported as an assignment word
            region = kinds[off] if off < len(kinds) else "code"
            preceded_by_dollar = off > 0 and logical[off - 1] == "$"
            in_test_region = off < len(in_test) and in_test[off]
            if region == "comment":
                shape = "COMMENT"
            elif in_test_region:
                shape = "TEST"
            elif region == "heredoc":
                shape = "HEREDOC"
            elif preceded_by_dollar:
                shape = "EXPANSION"
            elif region in ("dquote", "squote", "cmdsub"):
                shape = "QUOTED_STRING"
            else:
                shape = "MENTION"
            phys = _physical_line_for(linenos, offsets, off)
            if phys in heredoc:
                shape = "HEREDOC"
            sites.append(
                {
                    "file": path,
                    "line": phys,
                    "column": off + 1,
                    "shape": shape,
                    "class": "ARCHIVED" if archived else "BENIGN",
                    "semantic_class": "BENIGN",
                    "reason": _benign_reason(shape),
                    "effect": "none",
                    "defeats_ambient_empty_mask": False,
                    "self_reference": False,
                    "value_raw": None,
                    "command_word": None,
                    "text": logical[off : off + 60].strip()[:200],
                    "archived": archived,
                }
            )

    sites.sort(key=lambda s: (s["line"], s["column"]))
    return sites


def _word_offsets(words: Iterable[_Word]) -> set:
    out = set()
    for w in words:
        out.add(w.start)
    return out


def _physical_line_for(linenos: List[int], offsets: List[int], offset: int) -> int:
    """Map an offset inside the joined logical line back to a physical line."""
    idx = 0
    for k, start in enumerate(offsets):
        if offset >= start:
            idx = k
        else:
            break
    return linenos[idx]


def _benign_reason(shape: str) -> str:
    return {
        "QUOTED_STRING": "occurrence_inside_quoted_string_not_an_assignment",
        "TEST": "occurrence_inside_conditional_test_not_an_assignment",
        "COMMENT": "occurrence_inside_comment_not_an_assignment",
        "HEREDOC": "occurrence_inside_heredoc_body_not_an_assignment",
        "EXPANSION": "variable_reference_not_an_assignment",
        "MENTION": "bare_token_mention_not_an_assignment",
        "ARG_POSITION_ASSIGNMENT": "assignment_after_command_word_is_an_argument_not_an_env_assignment",
    }.get(shape, "not_an_assignment")


def _semantic_class(shape: str, value_info: Dict[str, Any]) -> str:
    if shape not in ("COMMAND_PREFIX", "EXPORT_ASSIGNMENT", "PLAIN_ASSIGNMENT"):
        return "BENIGN"
    if value_info["self_reference"] and not value_info["defeats_ambient_empty_mask"]:
        return "ALREADY_LOCK_PRESERVING"
    if value_info["defeats_ambient_empty_mask"]:
        return "REAL_DEFEATER"
    return "BENIGN"


# --------------------------------------------------------------------------- #
# Repository scan
# --------------------------------------------------------------------------- #
def _git_head(repo_root: str) -> Optional[str]:
    """Return the current git HEAD sha, or None (best effort, never fatal)."""
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


def _git_tracked_sh_files(repo_root: str) -> Tuple[List[str], List[str]]:
    """Return ``(tracked_sh_files, errors)`` from ``git ls-files -z -- '*.sh'``."""
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "ls-files", "-z", "--", "*.sh"],
            capture_output=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - git missing
        return [], ["git_ls_files_failed: %s" % exc]
    if proc.returncode != 0:
        return [], ["git_ls_files_exit_%d" % proc.returncode]
    files = [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]
    return files, []


def scan_repository(repo_root: str) -> Dict[str, Any]:
    """Scan every tracked ``*.sh`` file under ``repo_root``."""
    tracked, errors = _git_tracked_sh_files(repo_root)
    sites: List[Dict[str, Any]] = []
    per_file: Dict[str, Dict[str, int]] = {}
    sha_by_file: Dict[str, str] = {}
    files_read = 0
    files_unreadable: List[str] = []
    files_lossy: List[str] = []

    for rel in tracked:
        abs_path = os.path.join(repo_root, rel)
        try:
            with open(abs_path, "rb") as raw:
                blob = raw.read()
            text = blob.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            files_unreadable.append("%s: %s" % (rel, exc))
            continue
        files_read += 1
        sha_by_file[rel] = hashlib.sha256(blob).hexdigest()
        if "�" in text:
            files_lossy.append(rel)
        file_sites = scan_shell_text(text, rel)
        sites.extend(file_sites)
        counts: Dict[str, int] = {}
        for site in file_sites:
            counts[site["shape"]] = counts.get(site["shape"], 0) + 1
        counts["sites_total"] = len(file_sites)
        counts["real_defeater"] = sum(
            1 for s in file_sites if s["semantic_class"] == "REAL_DEFEATER"
        )
        counts["already_lock_preserving"] = sum(
            1 for s in file_sites if s["semantic_class"] == "ALREADY_LOCK_PRESERVING"
        )
        per_file[rel] = counts

    by_shape: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    by_semantic: Dict[str, int] = {}
    by_top_level: Dict[str, int] = {}
    for site in sites:
        by_shape[site["shape"]] = by_shape.get(site["shape"], 0) + 1
        by_class[site["class"]] = by_class.get(site["class"], 0) + 1
        by_semantic[site["semantic_class"]] = (
            by_semantic.get(site["semantic_class"], 0) + 1
        )
        top = site["file"].split("/")[0]
        by_top_level[top] = by_top_level.get(top, 0) + 1

    real_defeaters = [s for s in sites if s["semantic_class"] == "REAL_DEFEATER"]
    command_prefix = [s for s in sites if s["shape"] == "COMMAND_PREFIX"]

    denominator = {
        "tracked_sh_files_total": len(tracked),
        "files_read": files_read,
        "files_unreadable": len(files_unreadable),
        "files_decoded_lossy": len(files_lossy),
        "unreadable_paths": files_unreadable,
        "lossy_paths": files_lossy,
        "enumeration_errors": errors,
    }

    report = {
        "tool": "molmetal/scripts/wfa167_command_prefix_guard.py",
        "wf": "A167",
        "purpose": (
            "independent census of the command-prefix CUDA_VISIBLE_DEVICES "
            "assignment class + the denominator for it"
        ),
        "scope": "tracked *.sh files (git ls-files -z -- '*.sh')",
        "read_only": True,
        "cpu_only": True,
        "snapshot": {
            "git_head": _git_head(repo_root),
            "per_file_sha256": sha_by_file,
        },
        "denominator": denominator,
        "counts": {
            "sites_total": len(sites),
            "by_shape": by_shape,
            "by_class": by_class,
            "by_semantic_class": by_semantic,
            "by_top_level": by_top_level,
            "command_prefix_sites": len(command_prefix),
            "real_defeaters": len(real_defeaters),
            "real_defeaters_live": sum(
                1 for s in real_defeaters if not s["archived"]
            ),
            "real_defeaters_archived": sum(
                1 for s in real_defeaters if s["archived"]
            ),
        },
        "per_file": per_file,
        "sites": sites,
        "real_defeater_sites": [
            {
                "file": s["file"],
                "line": s["line"],
                "column": s["column"],
                "shape": s["shape"],
                "class": s["class"],
                "reason": s["reason"],
                "value_raw": s["value_raw"],
                "lock_preserving_form": (
                    lock_preserving_form(_strip_quotes(s["value_raw"] or "0"))
                ),
            }
            for s in real_defeaters
        ],
        "self_check": {
            "counts_match_lists": (
                by_class.get("REAL_DEFEATER", 0)
                + by_class.get("BENIGN", 0)
                + by_class.get("ARCHIVED", 0)
                + by_class.get("ALREADY_LOCK_PRESERVING", 0)
                == len(sites)
            ),
            "real_defeater_list_matches_count": (
                len(real_defeaters) == by_semantic.get("REAL_DEFEATER", 0)
            ),
        },
    }
    return report


# --------------------------------------------------------------------------- #
# Rendering / CLI
# --------------------------------------------------------------------------- #
def render_human(report: Dict[str, Any], max_print: int = 200) -> str:
    den = report["denominator"]
    cnt = report["counts"]
    lines = [
        "Wf-A167 command-prefix CUDA_VISIBLE_DEVICES census",
        "DENOMINATOR: %d tracked *.sh files; %d read; %d unreadable; %d lossy"
        % (
            den["tracked_sh_files_total"],
            den["files_read"],
            den["files_unreadable"],
            den["files_decoded_lossy"],
        ),
        "sites: %d total" % cnt["sites_total"],
        "  by shape: %s" % ", ".join(
            "%s=%d" % (k, v) for k, v in sorted(cnt["by_shape"].items())
        ),
        "  by class: %s" % ", ".join(
            "%s=%d" % (k, v) for k, v in sorted(cnt["by_class"].items())
        ),
        "  by semantic class: %s"
        % ", ".join(
            "%s=%d" % (k, v) for k, v in sorted(cnt["by_semantic_class"].items())
        ),
        "REAL_DEFEATER: %d (live %d / archived %d)"
        % (
            cnt["real_defeaters"],
            cnt["real_defeaters_live"],
            cnt["real_defeaters_archived"],
        ),
    ]
    for site in report["real_defeater_sites"][:max_print]:
        lines.append(
            "  %s:%d:%d [%s] %s"
            % (
                site["file"],
                site["line"],
                site["column"],
                site["shape"],
                site["value_raw"],
            )
        )
    if len(report["real_defeater_sites"]) > max_print:
        lines.append(
            "  ... %d more (see the JSON report)"
            % (len(report["real_defeater_sites"]) - max_print)
        )
    return "\n".join(lines)


def _atomic_write(path: str, content: str) -> None:
    """Write ``content`` to ``path`` through the repo's atomic-write helper."""
    helper = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "wf386a3_atomic_write.py"
    )
    if os.path.exists(helper):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_wfa167_aw", helper)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.atomic_write_file(path, content)
        return
    # Fallback: plain write (still atomic via os.replace).
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json", action="store_true", help="print JSON to stdout")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--max-print", type=int, default=200)
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="exit 0 even when real defeaters are found (on either axis)",
    )
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help=(
            "gate on ARCHIVED sites too (the legacy verdict axis).  The "
            "default axis is the LIVE one, because an archived site is "
            "never launched"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root or os.getcwd()
    report = scan_repository(repo_root)

    if args.out:
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        _atomic_write(os.path.abspath(args.out), payload)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif not args.quiet:
        print(render_human(report, args.max_print))

    if args.advisory:
        return 0
    # Wf-A269 wave-28 (2026-11-06) — THE VERDICT AXIS IS THE LIVE ONE.
    # The legacy predicate was `counts["real_defeaters"]`, which counts the
    # ARCHIVED sites as well — and the guard itself classifies those as
    # out of scope (they live under TODO/archive/ and are never launched).
    # MEASURED 2026-11-06: real_defeaters=3, real_defeaters_live=0,
    # real_defeaters_archived=3 ⇒ the CLI exited 1 FOREVER and could never
    # pass without --advisory, i.e. a gate that cannot PASS (§6.269.17's
    # second costume — the same class as a guard that cannot FAIL).
    # REPAIR: the verdict is on the LIVE axis; the archived count is still
    # reported in the report and in the human rendering; the strict axis
    # stays reachable via --include-archived.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #   return 1 if report["counts"]["real_defeaters"] else 0
    axis = "real_defeaters" if args.include_archived else "real_defeaters_live"
    return 1 if report["counts"][axis] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
