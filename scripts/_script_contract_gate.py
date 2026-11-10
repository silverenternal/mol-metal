#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""AGENTS.md §5.11 gate — every flag reaches a parser that DECLARES it.

THE DEFECT THIS GATE CLOSES (MEASURED 2026-11-04, read-only census)

``scripts/_launch_contract_gate.py`` asks "does the flag reach a parser that
declares it?" — but it ANCHORS that question on the trainer-NAME pattern
``^train_[a-z0-9_]+[.]py$``.  ``scripts/_yaml_only_gate.py`` rule 2 is
anchored the same way.  Therefore "the flag reaches a parser that DECLARES
it?" was UNCHECKED for every NON-``train_*`` repo script.

Two live instances were found by that census, and BOTH were real, total
launch-killers:

1. ``molmetal/scripts/smoke/smoke_lambda.sh`` passed ``--out-dir`` to
   ``molmetal/scripts/r4_lambda_only_run.py``, which declares
   ``--output-dir`` and NOT ``--out-dir``.  argparse exits 2 with
   "unrecognized arguments: --out-dir"; the smoke had NEVER passed.
2. ``scripts/launch_wf306b_wf219_ru_gpu.sh`` passed ``--out`` to
   ``molmetal/scripts/wf219_krasnov_aligned.py``, which declares
   ``--json-dir`` and NOT ``--out``.  argparse would have exited 2 — the
   wrapper could never have run.

Both were repaired by the main thread before this gate existed, so the live
tree now carries ZERO instances of THOSE two defects.  That is exactly why
this gate has to be exercised against FIXTURES: a gate that only ever sees a
repaired tree cannot prove it can fail.

THE LOOKAHEAD IS PART OF THE CONTRACT, NOT A DETAIL

``scripts/_yaml_only_gate.py::_script_index`` returns the token at the COMMAND
position.  When that token is an interpreter held in an UNRESOLVED shell
variable (``$PY``, assigned in several branches), the value cannot be
resolved, so the gate sees the interpreter token itself and NEVER sees the
script that follows it (``$PY -u molmetal/scripts/foo.py --flag``).  This gate
therefore looks PAST an unresolved interpreter to the next non-option token
and uses THAT as the script.  If that token is itself an unresolved variable,
the invocation is UNDECIDABLE — reported as a fail-closed coverage gap, never
as clean.  A gate that answers "probably fine" when it cannot tell is the
defect it was written to close (``docs/HONEST_NEGATIVE_CATALOG.md`` §6.269).

WHY THE FLAGS ARE READ BY AST AND NOT BY IMPORT

Importing a repo script can WRITE FILES.  MEASURED:
``molmetal/scripts/wf394a3_wf299_pre_launch_verify.py`` runs its whole check
at import time and rewrites ``molmetal/reports/wf394a3_wf299_pre_launch/
result.json``.  So this gate NEVER imports a repo script.  It extracts the
declared options with :func:`ast.parse` — walking the module for
``add_argument`` calls and collecting the STRING-LITERAL option strings (the
first positional argument and any other string-literal positional argument,
because argparse accepts aliases such as ``add_argument("--output-dir",
"--out-dir")``).  ``parser.add_argument`` and ``group.add_argument`` are
treated alike.

A script whose declared flags cannot be read is a COVERAGE GAP and is a
VIOLATION, fail-closed: a check that skipped what it could not read and
still reported PASS is a check that can PASS WITHOUT CHECKING.

Stdout invariant (AGENTS.md §4): PASS / FAIL / SKIP.  Exit 0=PASS, 1=FAIL,
2=SKIP.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_gate_module():
    """Import ``scripts/_yaml_only_gate.py`` — the shell-parsing home.

    It is imported by path (not as a package) because ``scripts/`` is not a
    package and the gate is a script, not a library.  Importing it has NO
    side effects: the module defines constants and functions and writes
    nothing at import time (its only ``subprocess.run`` is inside
    :func:`_git_tracked_sh_paths`).
    """
    path = REPO_ROOT / "scripts" / "_yaml_only_gate.py"
    spec = importlib.util.spec_from_file_location("_yaml_only_gate_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GATE = _load_gate_module()


#: The population of INVOCATIONS this gate audits: a repo-local PYTHON
#: script under ``molmetal/scripts/``.  The extension matters: a ``.sh``
#: script has no argparse vocabulary, so comparing a shell flag against it
#: would be a category error.
TARGET_SCRIPT_ROOT = REPO_ROOT / "molmetal" / "scripts"

#: Tokens that END a command segment.  ``_shell_tokens`` emits ``;``, ``&``,
#: ``|``, ``(``, ``)``, ``<`` and ``>`` as single-character tokens, so a
#: pipeline or a redirection splits one physical line into several commands.
_SEPARATORS = frozenset({";", "&", "|", "(", ")", "<", ">"})

#: A segment that STARTS with one of these is a shell KEYWORD, not a command.
#: ``if [[ ... ]]; then CMD; fi`` produces the segments ``if [[ ... ]]``,
#: ``then CMD`` and ``fi`` — the first and the last carry no command, and
#: the second carries the real one after ``then``.
_SEGMENT_SKIP_KEYWORDS = frozenset(
    {
        "if", "elif", "while", "until", "case", "for", "select", "function",
        "{", "}", "fi", "done", "esac", "in", "[[", "]]",
    }
)

#: A segment that STARTS with one of these is a command WITH a keyword in
#: front of it: the keyword is dropped and the rest is re-resolved.
_SEGMENT_STRIP_KEYWORDS = frozenset({"then", "else", "do", "!", "time"})

#: Wrapper commands that take a VALUE argument before the real command:
#: ``timeout 300 python x.py``.  Without this the duration would be read as
#: the command and the invocation would be missed.
_WRAPPER_VALUE_ARGS = {"timeout": 1}

#: ``uv run python x.py`` / ``pipenv run python x.py`` — the launcher form
#: that carries its own interpreter choice.  The sub-command is skipped
#: together with the launcher, and the launcher's own options with it.
_LAUNCH_PREFIXES = {
    "uv": {"run"},
    "pipenv": {"run"},
    "poetry": {"run"},
    "conda": {"run"},
    "hatch": {"run"},
}

#: A long option, as a standalone token: ``--flag`` or ``--flag=value``.
_LONG_FLAG = re.compile(r"^--[A-Za-z0-9][A-Za-z0-9_-]*$")

#: A SHORT option, as a standalone token: ``-x``.  Only the single-letter
#: form is collected.  An attached-value form (``-Zvalue``) is AMBIGUOUS —
#: argparse reads ``-abc`` as ``-a`` with the value ``bc`` when ``-a`` is
#: declared, and as ``-ab`` with the value ``c`` when ``-ab`` is declared —
#: so the gate cannot decide which option the token names, and reporting it
#: would be a guess (the §6.269.28.5 false-attribution class).
_SHORT_FLAG = re.compile(r"^-[A-Za-z]$")

#: A negative number is a VALUE, not an option: ``-1`` is a positional, and
#: argparse's own ``_negative_number_matcher`` treats it as one.
_NEGATIVE_NUMBER = re.compile(r"^-\d+(?:\.\d+)?$")

#: The shell interpreters whose ``-c`` argument is a COMMAND STRING that the
#: shell evaluates at RUN TIME.  MEASURED 2026-11-04: ``scripts/start.sh:1053``
#: runs ``nohup bash -c "$TRAINER_CMD >> $LOG_FILE 2>&1" &``, and
#: ``scripts/_wf_r28_gpu_master.sh:183`` runs ``nohup bash -c "${cmd}"``.  A
#: gate that stops at the shell interpreter never sees the trainer command
#: that the wrapper actually launches.
_SHELL_INTERPRETERS = frozenset({"bash", "sh", "zsh", "ksh", "dash"})

#: ``eval "..."`` — the same runtime transport as ``bash -c``, spelled as a
#: shell builtin.  ``eval`` is in the owner module's ``_DATA_COMMANDS``
#: (it takes DATA, not a script), so this gate has to look for it BEFORE
#: the data-command test.
_RUNTIME_CARRIERS = frozenset({"eval"})

#: How many times a command position may be expanded as a COMMAND-LINE
#: VARIABLE before the gate gives up and reports the line as undecidable.
#: Wf-A269 wave-19 (2026-11-04).  Two is enough for the repo's own shape
#: (``exec $TUI_CMD`` -> ``$PYTHON script.py ...``); the bound exists so a
#: self-referential assignment cannot make the gate loop.
_MAX_COMMAND_EXPANSION_DEPTH = 3

#: A heredoc marker: ``<<EOF``, ``<<'EOF'``, ``<<-EOF``.  ``<<<`` (herestring)
#: does NOT match — the third ``<`` is followed by a quote or a space, not by
#: a word character.
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


# ---------------------------------------------------------------------------
# Shell segmentation
# ---------------------------------------------------------------------------
def _scan_line(line: str, stack: list[str]) -> tuple[str, list[str]]:
    """Advance the shell-lexing state across one physical line.

    Returns ``(comment_free, new_stack)``.  ``comment_free`` is the line
    with its COMMENT removed but its quoting intact — it is what a heredoc
    marker must be searched in, so a commented-out example cannot open a
    heredoc.  The quoting must stay intact because the marker itself is
    usually quoted (``<<'PY'``) and stripping the quotes would erase the
    very token the search needs.

    The stack holds the open lexical contexts: ``"`` (double-quoted string),
    ``'`` (single-quoted string), ``$`` (a ``$( ... )`` command
    substitution) and ``(`` (a subshell).  The distinction between a
    ``"`` and a ``$`` matters: a multi-line DOUBLE-QUOTED STRING holds
    DATA, while a multi-line ``$( ... )`` holds COMMANDS — MEASURED
    2026-11-04: ``scripts/launch_wf409a3_metal_coord_retrain.sh:136-140``
    runs ``molmetal/scripts/wf_launch_gate_verdict.py`` inside a
    ``pf_verdict="$( ... )"`` substitution, and treating that as a string
    would have hidden the invocation.
    """
    i = 0
    n = len(line)
    out: list[str] = []
    while i < n:
        ch = line[i]
        top = stack[-1] if stack else None
        if top == "'":
            out.append(ch)
            if ch == "'":
                stack.pop()
            i += 1
            continue
        if top == '"':
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(line[i + 1])
                i += 2
                continue
            if ch == '"':
                stack.pop()
            i += 1
            continue
        # top is None, "$" or "("
        if ch == "#" and (i == 0 or line[i - 1].isspace()):
            break
        out.append(ch)
        if ch == "\\" and i + 1 < n:
            out.append(line[i + 1])
            i += 2
            continue
        if line.startswith("$(", i):
            stack.append("$")
            out.append("$(")
            i += 2
            continue
        if ch == "'":
            stack.append("'")
            i += 1
            continue
        if ch == '"':
            stack.append('"')
            i += 1
            continue
        if ch == "(":
            stack.append("(")
            i += 1
            continue
        if ch == ")":
            if stack:
                stack.pop()
            i += 1
            continue
        i += 1
    return "".join(out), stack


def _data_lines(text: str) -> set[int]:
    """The physical line numbers that hold DATA rather than shell commands.

    Two kinds:

    * a heredoc BODY — the repo's ``scripts/start.sh`` embeds a python
      program (and a YAML ``command:`` template) in one, and reading those
      lines as shell commands is how a gate ends up reporting a YAML string
      as an undecidable script invocation;
    * a line that CONTINUES an unterminated quoted string — the body of an
      inline ``python -c "..."`` program.

    The heredoc marker is searched in the COMMENT-FREE part of the line, so a
    commented-out example cannot open a heredoc that swallows the rest of
    the file.  (MEASURED 2026-11-04: ``scripts/launch_wf124_p3_g5.sh:250``
    holds a commented-out ``<<'PYEOF'`` example; matching it hid 518 lines
    of that wrapper, including five live invocations of repo-local scripts.)
    """
    data: set[int] = set()
    stack: list[str] = []
    pending_heredoc: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        if pending_heredoc is not None:
            if line.strip() == pending_heredoc:
                pending_heredoc = None
            else:
                data.add(number)
            continue
        incoming = stack[-1] if stack else None
        comment_free, stack = _scan_line(line, stack)
        if incoming in ("'", '"'):
            data.add(number)
        match = _HEREDOC.search(comment_free)
        if match:
            pending_heredoc = match.group(2)
    return data


def _is_array_assignment(token: str) -> bool:
    """Is this token the head of a bash ARRAY literal (``ARR=(``)?"""
    match = GATE._ASSIGN.match(token)
    if match is None:
        return False
    return match.group(2) == "+=" or match.group(3).strip() == ""


# ---------------------------------------------------------------------------
# Command substitutions — the owner's tokenizer keeps ``pf_verdict="$( ... )"``
# as ONE token, so the body of the substitution is invisible to the segment
# splitter unless it is extracted explicitly.
# ---------------------------------------------------------------------------
def _match_paren(text: str, start: int) -> tuple[str | None, int]:
    """The body of the ``$(`` opened at ``start - 2``, and the index just
    past its closing ``)``.

    ``None`` is returned for the body when the substitution is unterminated.
    """
    depth = 0
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'":
            j = text.find("'", i + 1)
            if j < 0:
                return None, n
            i = j + 1
            continue
        if ch == '"':
            j, _inner = _skip_dquote(text, i)
            if j is None:
                return None, n
            i = j
            continue
        if text.startswith("$(", i):
            _body, end = _match_paren(text, i + 2)
            if end >= n:
                return None, n
            i = end
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            if depth == 0:
                return text[start:i], i + 1
            depth -= 1
            i += 1
            continue
        i += 1
    return None, n


def _skip_dquote(text: str, start: int) -> tuple[int | None, list[tuple[int, int, str]]]:
    """The index just past the ``"`` that closes the double quote opened at
    ``start``, and the command substitutions found inside it.

    A ``$( ... )`` inside a double quote is still CODE (bash evaluates it),
    so it must be reported, not swallowed.
    """
    found: list[tuple[int, int, str]] = []
    i = start + 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            return i + 1, found
        if text.startswith("$(", i):
            body, end = _match_paren(text, i + 2)
            if body is None:
                return None, found
            found.append((i, end, body))
            i = end
            continue
        i += 1
    return None, found


def _substitutions(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, body)`` for every ``$( ... )`` command substitution.

    Substitutions inside single quotes are literal text and are NOT
    reported; substitutions inside double quotes ARE (bash evaluates them).
    """
    found: list[tuple[int, int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'":
            j = text.find("'", i + 1)
            if j < 0:
                break
            i = j + 1
            continue
        if ch == '"':
            j, inner = _skip_dquote(text, i)
            found.extend(inner)
            if j is None:
                break
            i = j
            continue
        if text.startswith("$(", i):
            body, end = _match_paren(text, i + 2)
            if body is None:
                break
            found.append((i, end, body))
            i = end
            continue
        i += 1
    return found


def _strip_substitutions(text: str) -> str:
    """``text`` with every ``$( ... )`` span removed."""
    spans = _substitutions(text)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for start, end, _body in spans:
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _with_substitution_segments(
    segments: list[tuple[list[str], bool, bool]]
) -> list[tuple[list[str], bool, bool]]:
    """The given segments plus the segments of every command substitution.

    The owner's tokenizer keeps ``pf_verdict="$( ... )"`` as ONE token, so
    without this the body of the substitution would never reach the segment
    splitter.  MEASURED 2026-11-04: six launchers invoke
    ``molmetal/scripts/wf_launch_gate_verdict.py`` inside such a
    substitution (``scripts/launch_wf409a3_metal_coord_retrain.sh:137``),
    and a gate that cannot see it would have certified those launches
    without checking a single flag.
    """
    extra: list[tuple[list[str], bool, bool]] = []
    for segment, is_redirect, is_array_value in segments:
        if is_redirect or is_array_value:
            continue
        for token in segment:
            for _start, _end, body in _substitutions(token):
                if not body.strip():
                    continue
                extra.extend(_split_segments(GATE._shell_tokens(body)))
    return list(segments) + extra


def _split_segments(tokens: list[str]) -> list[tuple[list[str], bool, bool]]:
    """Split a token list into command segments.

    Returns ``(segment, is_redirect_target, is_array_value)`` triples.

    * ``is_redirect_target`` — the segment after a ``>``/``<`` holds a FILE
      NAME, not a command (``cat > "${LOG}/plan.json"``).
    * ``is_array_value`` — the segment holds a value of a bash array literal
      (``ARR=(a b)``), which is DATA and not a command.
    """
    segments: list[tuple[list[str], bool, bool]] = []
    current: list[str] = []
    redirect_next = False
    array_paren = False
    after_assignment = False
    for token in tokens:
        if token == "(":
            array_paren = after_assignment
            after_assignment = False
            if current:
                segments.append((current, redirect_next, False))
                redirect_next = False
                current = []
            continue
        if token == ")":
            if current:
                segments.append((current, redirect_next, array_paren))
                redirect_next = False
                current = []
            array_paren = False
            after_assignment = False
            continue
        if token in ("<", ">"):
            if current:
                segments.append((current, redirect_next, False))
                redirect_next = False
                current = []
            redirect_next = True
            after_assignment = False
            continue
        if token in _SEPARATORS:
            if current:
                segments.append((current, redirect_next, False))
                redirect_next = False
                current = []
            after_assignment = False
            continue
        current.append(token)
        after_assignment = _is_array_assignment(token)
    if current:
        segments.append((current, redirect_next, False))
    return segments


def _command_index(segment: list[str], env: dict) -> int | None:
    """Index of the token that holds the command, or None.

    Leading ``VAR=value`` environment prefixes and wrapper commands
    (``env``, ``setsid``, ``nohup``, ``timeout``, ``uv run``, ...) are
    skipped.  A BASH ARRAY literal (``ARR=(a b)``) or an array append
    (``ARR+=(x)``) is DATA, not a command: the segment is not an invocation.
    """
    i = 0
    n = len(segment)
    while i < n:
        token = segment[i]
        assign = GATE._ASSIGN.match(token)
        if assign:
            if _is_array_assignment(token):
                # ``ARR+=(x)`` / ``ARR=(x)`` — the tokenizer splits the
                # parenthesis off, so the value reads empty.  Either way the
                # segment holds an array literal, not a command.
                return None
            i += 1
            continue
        name = GATE._last_segment(GATE._expand(token, env))
        if name in GATE._WRAPPERS:
            i += 1
            nargs = _WRAPPER_VALUE_ARGS.get(name)
            if nargs:
                i += nargs
            continue
        if name in _LAUNCH_PREFIXES:
            following = GATE._unquote(segment[i + 1]) if i + 1 < n else ""
            if following in _LAUNCH_PREFIXES[name]:
                i += 2
                while i < n and GATE._unquote(segment[i]).startswith("-"):
                    i += 1
                continue
            i += 1
            continue
        break
    return i if i < n else None


def _next_non_option(segment: list[str], start: int) -> int | None:
    """Index of the next token that is not an option, or None."""
    for j in range(start, len(segment)):
        bare = GATE._unquote(segment[j])
        if not bare:
            continue
        if bare.startswith("-"):
            continue
        return j
    return None


class _Resolution:
    """What a command segment turned out to be."""

    __slots__ = ("status", "token", "index", "why", "tokens", "payload")

    def __init__(self, status: str, token: str | None = None, index: int | None = None,
                 why: str = "", tokens: list[str] | None = None,
                 payload: str | None = None) -> None:
        self.status = status
        self.token = token
        self.index = index
        self.why = why
        # The token list ``index`` refers to.  It is the ORIGINAL segment
        # for a direct resolution, and the EXPANDED command line when the
        # resolution came from a command-line variable (see
        # :func:`_resolve_segment`).  The caller slices THIS list, not its
        # own segment, so an index into an expanded command cannot be
        # applied to the raw line.
        self.tokens = tokens
        # The COMMAND STRING a shell interpreter evaluates at RUN TIME
        # (``bash -c "..."`` / ``eval "..."``).  It is None for every other
        # resolution.
        self.payload = payload


def _runtime_payload(segment: list[str], index: int, env: dict, name: str) -> str | None:
    """The command string a runtime carrier evaluates, or None.

    Two carriers exist in the repo's own shape:

    * ``bash -c "<command>"`` — MEASURED 2026-11-04:
      ``scripts/start.sh:1053`` runs ``nohup bash -c "$TRAINER_CMD >> $LOG_FILE
      2>&1" &`` and ``scripts/_wf_r28_gpu_master.sh:183`` runs
      ``nohup bash -c "${cmd}" > "${log_file}" 2>&1 &``;
    * ``eval "<command>"`` — the same transport spelled as a builtin.  The
      owner module lists ``eval`` in ``_DATA_COMMANDS`` (it takes DATA, not
      a script), so this gate has to look for it BEFORE the data-command
      test.

    The payload is EXPANDED against the wrapper's own assignments, so a
    payload that holds a literal command line is resolved to the script and
    the flags it runs — which is exactly the invocation a gate that stops at
    the shell interpreter never sees.
    """
    start = index + 1
    if name in _RUNTIME_CARRIERS:
        if start >= len(segment):
            return None
        return GATE._expand(GATE._unquote(segment[start]), env)
    # A shell interpreter: the ``-c`` option (possibly bundled, e.g. ``-lc``)
    # is followed by the command string.
    i = start
    while i < len(segment):
        bare = GATE._unquote(segment[i])
        if not bare.startswith("-") or bare == "--":
            break
        letters = bare[1:]
        if letters and all(ch.isalpha() for ch in letters) and "c" in letters:
            if i + 1 < len(segment):
                return GATE._expand(GATE._unquote(segment[i + 1]), env)
            return None
        i += 1
    return None


def _resolve_segment(segment: list[str], env: dict, depth: int = 0) -> _Resolution:
    """Decide what a command segment invokes.

    Statuses:

    * ``none`` — no script is invoked (a data command, an inline
      ``python -c`` program, a shell keyword, a redirect target);
    * ``shell`` — a SHELL interpreter runs a script, which is out of this
      gate's contract (a shell script has no argparse vocabulary).  When the
      shell evaluates a command STRING (``bash -c`` / ``eval``), the
      resolution carries that string in ``payload`` and the caller audits
      it;
    * ``literal`` — the script token resolved to a literal path;
    * ``undecidable`` — the script token is a variable the gate cannot
      resolve, so it cannot tell what is being run.
    """
    resolutions = _resolve_core(segment, env, depth)
    return resolutions[0] if resolutions else _Resolution("none", why="no command in segment")


def _resolve_core(
    segment: list[str], env: dict, depth: int = 0
) -> list[_Resolution]:
    """Every reading of a command segment (a segment can name several)."""
    index = _command_index(segment, env)
    if index is None:
        return [_Resolution("none", why="no command in segment")]
    command = segment[index]
    bare = GATE._unquote(command)
    if bare in _SEGMENT_SKIP_KEYWORDS:
        return [_Resolution("none", why=f"shell keyword {bare}")]
    if bare in _SEGMENT_STRIP_KEYWORDS:
        rest = segment[index + 1:]
        if not rest:
            return [_Resolution("none", why=f"shell keyword {bare}")]
        return _resolve_core(rest, env, depth)

    expanded = GATE._expand(GATE._unquote(command), env)
    unresolved = bool(GATE._VARREF.search(expanded))
    # The command token IS a variable reference.  This is NOT the same test
    # as ``unresolved``: a variable whose VALUE is a literal command line
    # (``CMD="python foo.py --flag"; $CMD``) expands to something with no
    # variable left in it, and the gate used to read that as a plain
    # (non-interpreter) command and drop the invocation entirely.
    raw_is_var = bool(GATE._VARREF.search(bare))
    name = GATE._last_segment(expanded)

    # A RUNTIME CARRIER is checked BEFORE the data-command test: the owner
    # module lists ``eval`` among the data commands, and a gate that stops
    # there never sees the command the wrapper actually runs.
    if not unresolved and name in _RUNTIME_CARRIERS:
        payload = _runtime_payload(segment, index, env, name)
        return [
            _Resolution(
                "shell",
                why=f"runtime carrier {name}",
                payload=payload,
            )
        ]

    if not unresolved and name in GATE._DATA_COMMANDS:
        return [_Resolution("none", why=f"data command {name}")]
    if not unresolved and GATE._is_interpreter(name):
        following = GATE._unquote(segment[index + 1]) if index + 1 < len(segment) else ""
        if name.startswith("python") and following == "-c":
            return [_Resolution("none", why="inline python program (python -c)")]
        script_index = _next_non_option(segment, index + 1)
        if script_index is None:
            return [_Resolution("none", why=f"interpreter {name} without a script")]
        if name in _SHELL_INTERPRETERS:
            payload = _runtime_payload(segment, index, env, name)
            return [
                _Resolution(
                    "shell",
                    why=f"shell interpreter {name}",
                    payload=payload,
                )
            ]
        return _resolve_candidate(segment, script_index, env, f"after interpreter {name}")

    if raw_is_var:
        # Wf-A269 wave-19 (2026-11-04) -- the COMMAND-LINE VARIABLE class.
        #
        # A command position can hold a variable whose VALUE IS A WHOLE
        # COMMAND LINE.  MEASURED 2026-11-04 on scripts/start.sh:1086:
        #
        #   TUI_CMD="$PYTHON scripts/_tui_dashboard.py --log-file $LOG_FILE \
        #            --output-dir $OUT_DIR --gpu $GPU ..."
        #   exec $TUI_CMD
        #
        # The expansion names scripts/_tui_dashboard.py, so the invocation
        # IS decidable -- the gate was reporting "undecidable" where it could
        # DECIDE.  That is the same false-attribution class as
        # §6.269.28.5 (a check that GUESSES where it could decide produces a
        # false attribution), and the fix is to resolve the EXPANDED command
        # rather than the variable.
        #
        # The guard is deliberately narrow: the command token must BE a
        # variable reference, and its expansion must split into MORE THAN
        # ONE token.  A variable that expands to a single word (a python
        # path, a program name) is NOT a command line and must not be
        # re-split.
        #
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6
        # (the gate went straight to the lookahead, which found no script
        # token and returned "undecidable"):
        #
        #   script_index = _next_non_option(segment, index + 1)
        #   if script_index is None:
        #       return _Resolution("undecidable", token=command, why=...)
        #   return _resolve_candidate(segment, script_index, env, ...)
        if depth < _MAX_COMMAND_EXPANSION_DEPTH:
            expanded_tokens = GATE._shell_tokens(expanded)
            if len(expanded_tokens) > 1:
                return _resolve_core(expanded_tokens, env, depth + 1)

        # THE BRANCH UNION.  A variable assigned in SEVERAL branches names
        # SEVERAL scripts, and only the LAST assignment used to be checked:
        #
        #   if [[ -n "${WHICH:-}" ]]; then
        #       SCRIPT="molmetal/scripts/r4_lambda_only_run.py"   # no --json-dir
        #   else
        #       SCRIPT="molmetal/scripts/wf219_krasnov_aligned.py" # has --json-dir
        #   fi
        #   "$PY" -u "$SCRIPT" --json-dir /tmp/out
        #
        # The defective branch was invisible.  The union checks every branch.
        candidates = _variable_script_candidates(bare, env)
        if candidates:
            return [
                _Resolution(
                    "literal", token=candidate, index=index,
                    why="branch union", tokens=segment,
                )
                for candidate in candidates
            ]

        # ONE token.  It may be the SCRIPT itself, or an INTERPRETER whose
        # script follows.
        single = GATE._shell_tokens(expanded)
        if len(single) == 1:
            value = single[0]
            if not GATE._VARREF.search(value):
                if _names_a_python_script(value):
                    return _resolve_candidate(segment, index, env, "direct")
                if not GATE._is_interpreter(GATE._last_segment(value)):
                    # A resolved, NON-interpreter program (cat, cp, mkdir):
                    # the segment takes DATA, not a script.
                    return [_Resolution("none", why=f"program {value}")]
            elif _names_a_python_script(bare):
                # The token IS a script path, but the path itself is
                # unresolvable: the script is UNDECIDABLE, fail-closed.
                return [
                    _Resolution(
                        "undecidable", token=command,
                        why="the script token is an unresolved variable, so the "
                            "gate cannot tell which script this line runs and "
                            "therefore cannot check the flags it passes",
                    )
                ]

        # THE LOOKAHEAD: the command position holds an interpreter this gate
        # cannot resolve (a python path assigned in several branches).  The
        # script is the next non-option token; if that is a variable too,
        # the invocation is UNDECIDABLE — fail-closed, never "clean".
        script_index = _next_non_option(segment, index + 1)
        if script_index is None:
            return [
                _Resolution(
                    "undecidable", token=command,
                    why="the command position holds an unresolved variable and no "
                        "script token follows it, so the gate cannot tell what "
                        "this line runs",
                )
            ]
        return _resolve_candidate(segment, script_index, env, "after an unresolved interpreter")

    return _resolve_candidate(segment, index, env, "direct")


#: The repo-relative SHAPE of a repo-local python script.  This is a shape
#: test, not a filesystem test: the value may carry an unresolved prefix
#: (``${REPO_ROOT}/...``) and the gate must still be able to say whether the
#: value names a repo script.  A host-bound path check would make the answer
#: depend on this machine — the §6.269.13 host-bound-path class.
_SCRIPT_PATH_SHAPE = re.compile(r"(?:^|/)molmetal/scripts/[A-Za-z0-9_./-]+\.py$")


def _looks_like_script_path(text: str) -> bool:
    """Does this string NAME a python file under ``molmetal/scripts/``?

    The repo's launchers spell the repo root in several ways — an absolute
    path, ``${REPO_ROOT}``, or a ``$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )``
    substitution — so the test is made on the SHAPE of the string, after the
    substitutions are stripped, and never on the host's filesystem.
    """
    if not text:
        return False
    stripped = _strip_substitutions(text).strip()
    for variant in (text.strip(), stripped, stripped.lstrip("/")):
        if variant and _SCRIPT_PATH_SHAPE.search(variant):
            return True
    return False


def _variable_script_candidates(raw: str, env: dict) -> list[str]:
    """The repo-local script paths a variable can name, over every branch.

    The owner module records every assignment of a name in
    ``env.branch_values``; a scalar assigned in two branches therefore has
    TWO values, and the gate must check both.  Only a value that names a
    file under ``molmetal/scripts/`` is returned — anything else is not a
    candidate script and must not turn the invocation into a false one.
    """
    match = GATE._VARREF.search(raw)
    if match is None:
        return []
    name = match.group(1) or match.group(4)
    if name is None:
        return []
    values = getattr(env, "branch_values", {}).get(name)
    if not values:
        return []
    candidates: list[str] = []
    for value in values:
        if isinstance(value, list):
            elements = [str(item) for item in value]
        else:
            elements = [str(value)]
        for element in elements:
            expanded = GATE._expand(element, env)
            if GATE._VARREF.search(expanded):
                continue
            if _names_a_python_script(expanded) and expanded not in candidates:
                candidates.append(expanded)
    return candidates


def _resolve_candidate(
    segment: list[str], index: int, env: dict, how: str
) -> list[_Resolution]:
    token = segment[index]
    raw = GATE._unquote(token)
    # THE BRANCH UNION comes FIRST: a variable assigned in two branches
    # names TWO scripts, and expanding it to its LAST value alone hides the
    # branch that does not run (the owner module's own doctrine: "the union
    # of the branches is the fail-closed reading" — see
    # ``scripts/_yaml_only_gate.py::_ShellEnv.branch_values``).
    candidates = _variable_script_candidates(raw, env)
    if candidates:
        return [
            _Resolution(
                "literal", token=candidate, index=index,
                why=f"{how} (branch union)", tokens=segment,
            )
            for candidate in candidates
        ]
    expanded = GATE._expand(raw, env)
    if GATE._VARREF.search(expanded):
        return [
            _Resolution(
                "undecidable", token=token,
                why=f"{how}: the script token is an unresolved variable, so the "
                    "gate cannot tell which script this line runs and therefore "
                    "cannot check the flags it passes",
            )
        ]
    return [_Resolution("literal", token=expanded, index=index, why=how, tokens=segment)]


def _names_a_python_script(text: str) -> bool:
    """Does this token NAME a python script (a file, or a module)?

    Two spellings count:

    * a path that ends in ``.py`` (``molmetal/scripts/foo.py``);
    * the MODULE form ``python -m molmetal.scripts.foo`` — MEASURED
      2026-11-04: the module form is a REAL invocation shape (the repo's
      ``pyproject.toml`` exposes console scripts, and ``python -m`` is how
      a module is run as ``__main__``), and a gate that stops at the
      ``.py`` suffix reads it as out of contract.
    """
    if not text:
        return False
    if text.endswith(".py"):
        return True
    return "/" not in text and text.startswith("molmetal.scripts.")


def _module_to_path(token: str) -> Path | None:
    """``molmetal.scripts.foo.bar`` -> ``molmetal/scripts/foo/bar.py``."""
    if "/" in token or not token.startswith("molmetal.scripts."):
        return None
    parts = token.split(".")
    if len(parts) < 3:
        return None
    relative = Path(*parts[2:]).with_suffix(".py")
    return REPO_ROOT / "molmetal" / "scripts" / relative


def _repo_local_script(token: str, cwd: Path | None = None) -> Path | None:
    """The repo-local python script a resolved token names, or None.

    A token whose PREFIX is a command substitution still names a repo-local
    script.  The repo's launchers spell the repo root as
    ``$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )`` and then append the
    repo-relative path, so the resolved token reads
    ``$( cd ... && pwd )/molmetal/scripts/pre_flight_check.py``.  The
    substitution only decides WHERE the file lives; the file itself is
    addressed by its repo-relative path, so the gate follows it.

    HONEST LIMIT: the substitution is assumed to denote THIS repository's
    root.  A substitution that denoted a different tree would make the gate
    read that tree's parser instead of this one's — disclosed here rather
    than assumed silently.

    ``cwd`` is the wrapper's WORKING DIRECTORY at the point of the
    invocation.  MEASURED 2026-11-04: a wrapper that does ``cd
    molmetal/scripts`` and then runs ``python r4_lambda_only_run.py`` names
    the script RELATIVE to that directory, and a gate that resolves every
    relative path against the repo root reads it as out of contract.
    """
    root = TARGET_SCRIPT_ROOT.resolve()
    variants = [token]
    stripped = _strip_substitutions(token)
    if stripped != token:
        # the dynamic prefix carried the leading separator with it.
        variants.append(stripped)
        variants.append(stripped.lstrip("/"))
    for text in variants:
        if not text:
            continue
        module = _module_to_path(text)
        try:
            if module is not None:
                candidate = module.resolve()
            else:
                candidate = Path(text)
                if not candidate.is_absolute():
                    candidate = (cwd or REPO_ROOT) / candidate
                candidate = candidate.resolve()
        except OSError:
            continue
        if root in candidate.parents and candidate.suffix == ".py":
            return candidate
    return None


# ---------------------------------------------------------------------------
# Declared flags (AST only — never an import)
# ---------------------------------------------------------------------------
def _option_string(node: ast.AST) -> str | None:
    """The option string a literal AST node holds, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if node.value.startswith("-"):
            return node.value
    return None


def _string_sequence(node: ast.AST) -> list[str] | None:
    """The string constants a constant tuple/list holds, or None.

    Only a tuple/list whose elements are ALL string constants qualifies: a
    container that holds anything else is not a constant the gate can read.
    """
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    values: list[str] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return None
        values.append(element.value)
    return values


def _constant_bindings(tree: ast.Module) -> dict[str, list[str]]:
    """``name -> the string constants that name can hold``.

    A module may declare its options through a VARIABLE instead of a
    literal.  MEASURED 2026-11-04 on
    ``molmetal/scripts/train_dmpnn_multitask_wasserstein.py:3232``::

        for _flag, _dest in (
            ("--use-triton-relu-mlp", "use_triton_relu_mlp"),
            ("--use-triton-wasserstein-sliced", "use_triton_wasserstein_sliced"),
            ("--use-triton-bond-head", "use_triton_bond_head"),
            ("--use-triton-smiles-featurize", "use_triton_smiles_featurize"),
        ):
            p.add_argument(_flag, dest=_dest, action="store_true", ...)

    Reading only the STRING LITERAL option strings made the gate report
    ``--use-triton-bond-head`` as UNDECLARED — a false positive on a real
    trainer's real option (the option IS declared; argparse accepts it).

    This map resolves the two shapes that carry a declaration:

    * ``NAME = ("a", "b")`` and ``a, b = ("x", "y")``;
    * ``for name, other in (("a", "x"), ("b", "y")):`` — element-wise.

    A name this map cannot resolve is NOT resolved: the caller then reports
    the script's vocabulary as UNREADABLE, never as empty.
    """
    bindings: dict[str, list[str]] = {}

    def add(name: str, value: str) -> None:
        bindings.setdefault(name, []).append(value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            values = _string_sequence(node.value)
            if values is None:
                continue
            if isinstance(node.targets[0], ast.Name):
                for value in values:
                    add(node.targets[0].id, value)
            elif isinstance(node.targets[0], (ast.Tuple, ast.List)):
                targets = node.targets[0].elts
                if len(targets) != len(values):
                    continue
                for target, value in zip(targets, values):
                    if isinstance(target, ast.Name):
                        add(target.id, value)
        elif isinstance(node, ast.For):
            if not isinstance(node.target, (ast.Tuple, ast.List)):
                continue
            if not isinstance(node.iter, (ast.Tuple, ast.List)):
                continue
            targets = node.target.elts
            for element in node.iter.elts:
                values = _string_sequence(element)
                if values is None or len(values) != len(targets):
                    continue
                for target, value in zip(targets, values):
                    if isinstance(target, ast.Name):
                        add(target.id, value)
    return bindings


def _arity(node: ast.Call) -> str:
    """How many values the option consumes: none | one | many | optional."""
    nargs = None
    action = None
    for keyword in node.keywords:
        if keyword.arg == "nargs" and isinstance(keyword.value, ast.Constant):
            nargs = keyword.value.value
        if keyword.arg == "action":
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                action = keyword.value.value
            elif isinstance(keyword.value, ast.Attribute):
                action = keyword.value.attr
    if action in {
        "store_true", "store_false", "count", "store_const", "append_const",
        "help", "version",
        # MEASURED 2026-11-04: ``argparse.BooleanOptionalAction`` declares
        # ``nargs=0`` (CPython's ``_BooleanOptionalAction.__init__``), so
        # ``--smoke`` consumes NO value.  Reading it as "one" made the gate
        # report a legal ``--smoke --out-dir DIR`` as "--smoke without a
        # value" — a FALSE POSITIVE on three live wrappers
        # (scripts/launch_wf388_a1_metal_coord_retrain.sh:534,
        # scripts/launch_wf409a3_metal_coord_retrain.sh:465,
        # scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh:282).
        "BooleanOptionalAction",
    }:
        return "none"
    if nargs == 0:
        return "none"
    if nargs in ("*", "+"):
        return "many"
    if nargs == "?":
        return "optional"
    return "one"


def _forwards_to_add_argument(tree: ast.Module, path: Path) -> bool:
    """Does this module route its declarations through a helper that
    forwards to ``add_argument``?

    The repo has exactly one such helper: ``_add`` in
    ``molmetal/scripts/_common_train_args.py``, which ends in
    ``group.add_argument(*args, **kwargs)``.  A module that calls it with a
    literal option string IS declaring that option; reading only the direct
    ``add_argument`` calls would miss it and turn a readable script into a
    false coverage gap.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_add":
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "add_argument"
                ):
                    return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not node.module.endswith("_common_train_args"):
                continue
            if any(alias.name == "_add" for alias in node.names):
                helper = TARGET_SCRIPT_ROOT / "_common_train_args.py"
                if helper.is_file():
                    try:
                        helper_tree = ast.parse(
                            helper.read_text(encoding="utf-8", errors="replace")
                        )
                    except (OSError, SyntaxError):
                        return False
                    return _forwards_to_add_argument(helper_tree, helper)
    return False


def declared_flags(
    script_path: Path,
) -> tuple[dict[str, str] | None, str | None, str]:
    """Every long option the script's parser declares, with its arity.

    Returns ``(flags, error, state)``.  ``flags`` maps an option string to
    its arity (``none``/``one``/``many``/``optional``).  ``state`` is the
    VOCABULARY STATE — which is NOT the same thing as "the flags came back
    empty", and the difference is the difference between a DECIDABLE
    invocation and a coverage gap:

    * ``declared`` -- the vocabulary was read.  ``flags`` holds every long
      option the script declares.
    * ``empty`` -- the script declares NO options at all: no parser, no
      ``add_argument``, and no read of ``sys.argv``.  MEASURED 2026-11-04
      over ``molmetal/scripts/*.py``: **134** scripts are in this state,
      **11** more declare no options but DO read ``sys.argv`` on their own,
      and **1** has a parser whose option strings are not literals.  The
      vocabulary is empty AND KNOWN to be empty, so a flagless invocation
      is legal and a flag-carrying invocation passes a flag the script will
      silently ignore.
    * ``unreadable`` -- the script HAS options but the gate cannot
      enumerate them: a non-literal option string, or a module that reads
      ``sys.argv`` on its own.  This is a genuine COVERAGE GAP.
    * ``broken`` -- the file could not be read or parsed.  Also a genuine
      coverage gap, and a defect in its own right.

    ``error`` is set for every state except ``declared``.
    """
    try:
        source = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"cannot read {script_path.name}: {exc}"[:140], "broken"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return None, f"cannot parse {script_path.name}: {exc}"[:140], "broken"

    flags: dict[str, str] = {}
    forwarded = _forwards_to_add_argument(tree, script_path)
    bindings = _constant_bindings(tree)
    parsers_with_help = 0
    parsers_without_help = 0
    has_parser = False
    n_option_calls = 0
    n_unresolved_option_calls = 0

    def _record(name: str, node: ast.Call) -> None:
        flags.setdefault(name, _arity(node))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "add_argument":
                n_option_calls += 1
                boolean_optional = any(
                    keyword.arg == "action"
                    and isinstance(keyword.value, ast.Attribute)
                    and keyword.value.attr == "BooleanOptionalAction"
                    for keyword in node.keywords
                )
                for argument in node.args:
                    name = _option_string(argument)
                    if name is None:
                        # The option string may be held in a VARIABLE.  The
                        # module's own constant tables resolve the repo's
                        # one such site; an unresolvable name leaves the
                        # vocabulary INCOMPLETE, which is recorded below.
                        if isinstance(argument, ast.Name):
                            candidates = bindings.get(argument.id) or []
                            resolved = [
                                value
                                for value in candidates
                                if value.startswith("-")
                            ]
                            if resolved:
                                for value in resolved:
                                    _record(value, node)
                                    if boolean_optional and value.startswith("--"):
                                        flags.setdefault("--no-" + value[2:], "none")
                                continue
                        n_unresolved_option_calls += 1
                        continue
                    _record(name, node)
                    if boolean_optional and name.startswith("--"):
                        # argparse's BooleanOptionalAction accepts BOTH
                        # ``--x`` and ``--no-x``.
                        flags.setdefault("--no-" + name[2:], "none")
            elif func.attr == "ArgumentParser":
                has_parser = True
                disabled = any(
                    keyword.arg == "add_help"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is False
                    for keyword in node.keywords
                )
                if disabled:
                    parsers_without_help += 1
                else:
                    parsers_with_help += 1
        elif isinstance(func, ast.Name):
            if func.id == "ArgumentParser":
                # ``from argparse import ArgumentParser`` — the Name form.
                # MEASURED 2026-11-04: the Attribute form was the only one
                # recognised, so a parser built this way was reported as
                # "the AST holds no ArgumentParser".
                has_parser = True
            elif forwarded and func.id == "_add":
                n_option_calls += 1
                for argument in node.args:
                    name = _option_string(argument)
                    if name is not None:
                        flags.setdefault(name, "one")

    if not flags:
        # The vocabulary is empty.  The question that decides EVERYTHING is
        # whether the script HAS options the gate could not enumerate.
        if n_option_calls:
            # An ``add_argument`` call exists but its option string is not a
            # literal, so the gate cannot know what the parser accepts.
            return None, (
                f"no readable option declaration in {script_path.name}: the AST "
                "holds an add_argument call whose option string is not a "
                "literal, so the gate cannot enumerate the vocabulary"
            ), "unreadable"
        if has_parser:
            # A parser with NO option declarations at all declares nothing.
            return None, (
                f"no readable option declaration in {script_path.name}: the AST "
                "holds an ArgumentParser with no add_argument call, so the "
                "script declares no options"
            ), "empty"
        if _reads_argv(tree):
            # No parser, no option call, but the module reads sys.argv: it
            # may consume options on its own, so the gate cannot decide.
            return None, (
                f"no readable option declaration in {script_path.name}: the AST "
                "holds no ArgumentParser and no add_argument call, but the "
                "module reads sys.argv, so it may consume options on its own "
                "and the gate cannot prove that the flags on this invocation "
                "are accepted"
            ), "unreadable"
        return None, (
            f"no readable option declaration in {script_path.name}: the AST "
            "holds no ArgumentParser, no add_argument call and no read of "
            "sys.argv, so the script declares NO options"
        ), "empty"

    if parsers_with_help:
        # argparse adds --help AND -h to every parser that was not built
        # with add_help=False; they are declarations of the parser, not of
        # the script's own options.  MEASURED 2026-11-04: leaving ``-h``
        # out made the gate report ``-h`` as an undeclared flag on every
        # wrapper that asks for help — a false positive, because argparse
        # accepts it.
        flags.setdefault("--help", "none")
        flags.setdefault("-h", "none")

    if n_unresolved_option_calls:
        # The vocabulary is INCOMPLETE: some ``add_argument`` calls carry an
        # option string the gate could not resolve.  The flags that WERE
        # resolved are still checked, and the incompleteness is reported as
        # a coverage gap — fail-closed, because a check that silently
        # skipped what it could not read is a check that can PASS WITHOUT
        # CHECKING (§6.269.17).
        return flags, (
            f"the declared options of {script_path.name} are only PARTIALLY "
            f"readable: {n_unresolved_option_calls} add_argument call(s) carry "
            "an option string the gate cannot resolve, so the gate cannot "
            "enumerate the whole vocabulary"
        ), "partial"

    return flags, None, "declared"


def _reads_argv(tree: ast.AST) -> bool:
    """Does this module read ``sys.argv`` (or an imported ``argv``)?

    A module that reads argv may consume options on its own, so a gate that
    cannot see an option vocabulary must NOT conclude that the vocabulary is
    empty.  MEASURED 2026-11-04 over ``molmetal/scripts/*.py``: 11 scripts
    declare no options but DO read argv, and 134 declare no options and do
    not — the difference is what separates an "empty" vocabulary from an
    "unreadable" one.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "argv":
            return True
        if isinstance(node, ast.Name) and node.id == "argv":
            return True
    return False


def _argparse_accepts(declared: dict[str, str], flag: str) -> bool:
    """Would ``argparse`` accept ``flag`` given the script's declared options?

    An EXACT-match-only rule is WRONG: ``argparse`` enables prefix
    abbreviation by default (``allow_abbrev`` defaults to True), so
    ``--seed`` IS accepted by a parser that declares only ``--seeds`` and
    the launch does NOT die.  CPython's own rule (``_get_option_tuples``):
    an option string matches when it is a PREFIX of a declared option;
    exactly one match is accepted, more than one raises "ambiguous option",
    none raises "unrecognized arguments".  This reproduces that rule, and
    the sibling gate ``scripts/_runtime_transport_gate.py::_argparse_accepts``
    reproduces it too, so the two §5.11 gates cannot disagree.
    """
    if flag in declared:
        return True
    matches = [name for name in declared if name.startswith(flag)]
    return len(matches) == 1


def _declared_arity(declared: dict[str, str], flag: str) -> str:
    """The arity of the option ``flag`` names (exact or unambiguous prefix)."""
    if flag in declared:
        return declared[flag]
    matches = [name for name in declared if name.startswith(flag)]
    if len(matches) == 1:
        return declared[matches[0]]
    return "none"


def _long_option(bare: str) -> str | None:
    """The long option a token names, or None.

    ``--flag`` and ``--flag=value`` both name ``--flag``.
    """
    if not bare.startswith("--"):
        return None
    candidate = bare.split("=", 1)[0]
    if _LONG_FLAG.match(candidate):
        return candidate
    return None


def _short_option(bare: str) -> str | None:
    """The single-letter short option a token names, or None.

    Only ``-x`` is collected.  The attached-value form (``-Zvalue``) is
    AMBIGUOUS — argparse reads ``-abc`` as ``-a`` with the value ``bc`` when
    ``-a`` is declared, and as ``-ab`` with the value ``c`` when ``-ab`` is
    declared — so the gate cannot decide which option such a token names,
    and reporting it would be a guess (the §6.269.28.5 false-attribution
    class).  A negative number is a VALUE, never an option.
    """
    if not bare.startswith("-") or bare.startswith("--"):
        return None
    if _NEGATIVE_NUMBER.match(bare):
        return None
    if not _SHORT_FLAG.match(bare):
        return None
    return bare


def _looks_like_option(bare: str) -> bool:
    """Would argparse read this token as an OPTION rather than a value?

    CPython's own rules: a token that starts with a prefix character is an
    option unless it is a negative number (``_negative_number_matcher``)
    or the bare ``-``/``--``.

    THE SPACE RULE (MEASURED 2026-11-04, CPython 3 argparse).  argparse's
    ``_parse_optional`` returns None — "this is a POSITIONAL" — for a token
    that CONTAINS A SPACE, so such a token is accepted as the value of a
    value-taking option:

        >>> p.add_argument("--flags", action="append")
        >>> p.parse_args(["--flags", "--config --out-dir"])
        Namespace(flags=['--config --out-dir'])

    Without this rule the gate reported the repo's own legal form
    (``molmetal/scripts/wf_argv_capability_check.py --flags
    "--config --out-dir"``, whose help text says the whitespace-separated
    form is the intended one) as "the option was passed without a value" —
    a false attribution on a launch that works.  MEASURED on the live
    invocation: the capability check parses and runs.
    """
    if not bare or bare in ("-", "--"):
        return False
    if _NEGATIVE_NUMBER.match(bare):
        return False
    if " " in bare:
        # argparse treats a token with a space as a POSITIONAL.
        return False
    return bare.startswith("-")


class _Unresolved:
    """A shell reference the gate could NOT resolve, standing in a token.

    The owner module's ``_expand`` leaves an unresolvable reference IN
    PLACE, so the gate can tell "resolved to something" from "not resolved
    at all".  The second case is the fail-closed one (AGENTS.md §5.11
    requirement 5, extended to the ARGV: a flag that hides in an unresolved
    reference is a flag the gate cannot check).
    """

    __slots__ = ("name", "text")

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Unresolved({self.name!r}, {self.text!r})"


def flags_in_argv(
    tokens: list, declared: dict[str, str]
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """The options in an argv, the options that got NO value, and the
    UNRESOLVED references that stand where an option could stand.

    Returns ``(flags, missing_values, unresolved_refs)``.  ``flags`` holds
    the option strings the argv passes, with the option VALUES excluded;
    ``missing_values`` holds ``(option, offending_token)`` for every
    value-taking option that is followed by an option-looking token;
    ``unresolved_refs`` names the shell references the gate could not
    resolve and that stand where an OPTION could stand.

    A token is a VALUE, not a flag, when it is the argument of a preceding
    option: ``--flags "--config --out-dir"`` passes ONE token that happens
    to contain two flag-shaped strings.  Counting those would report a legal
    launch as broken.

    THE ARITY IS NOT INERT.  The old rule decided "is this a flag?" from the
    token's SHAPE alone, so the value bookkeeping never ran and two real
    defects were misdiagnosed:

    * a VALUE that itself starts with ``--`` was read as an option
      (``--output-dir --weird``); argparse raises "expected one argument"
      there, so ``--weird`` is a MISSING VALUE, not an unknown option;
    * the ``--`` end-of-options marker was ignored, so a POSITIONAL that
      follows it was reported as an undeclared flag.

    Both are argparse's own rules (CPython's ``_get_nargs_pattern`` and
    ``_parse_known_args``), reproduced here so the gate's verdict and its
    diagnosis agree.

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6 (the
    "one" branch reported EVERY next token as a missing value, so a legal
    ``--json-dir x`` was reported as "--json-dir without a value (x follows
    it)"):

        if pending == "one":
            missing.append((pending_option, bare))
            pending = "none"
            continue
    """
    found: list[str] = []
    missing: list[tuple[str, str]] = []
    unresolved: list[str] = []
    pending = "none"
    pending_option = ""
    after_ddash = False
    for token in tokens:
        if isinstance(token, _Unresolved):
            # An UNRESOLVED shell reference.  In a VALUE position it is
            # harmless — the option takes whatever the value turns out to
            # be.  In an OPTION position it could be a flag the gate cannot
            # check, so it is reported.
            if pending in ("one", "optional"):
                pending = "none"
                continue
            if pending == "many":
                continue
            unresolved.append(token.name)
            continue
        bare = GATE._unquote(token)
        if after_ddash:
            # argparse stops parsing options at the bare ``--``: every token
            # after it is a POSITIONAL.
            continue
        if bare == "--":
            after_ddash = True
            pending = "none"
            continue
        if pending == "one":
            # argparse consumes the next token as the VALUE of a
            # value-taking option, and an option-looking token is NOT
            # accepted there: it raises "expected one argument".  The token
            # is therefore not an unknown option — the defect is the option's
            # MISSING VALUE.
            if _looks_like_option(bare):
                missing.append((pending_option, bare))
            pending = "none"
            continue
        if pending == "optional":
            # argparse's OPTIONAL arity ("?") consumes the next token ONLY
            # when that token does not look like an option.
            if not _looks_like_option(bare):
                pending = "none"
                continue
            pending = "none"
        elif pending == "many":
            if not _looks_like_option(bare):
                continue
            pending = "none"
        name = _long_option(bare)
        if name is not None:
            found.append(name)
            if "=" in bare:
                # ``--flag=value`` carries its own value.
                pending = "none"
            else:
                pending = _declared_arity(declared, name)
                pending_option = name
            continue
        short = _short_option(bare)
        if short is not None:
            found.append(short)
            pending = _declared_arity(declared, short)
            pending_option = short
            continue
    return found, missing, unresolved


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def _argv_after(
    segment: list[str], index: int, env: dict
) -> tuple[list[str], list[str], list[str]]:
    """The argv AFTER the script token, with bash arrays spliced in.

    Returns ``(tokens, unresolved_arrays, hidden_flag_vars)``.

    * ``unresolved_arrays`` names every bash array the line references that
      this gate could NOT read.  The flags inside such an array are
      invisible to the scan, so the caller must not treat the line as clean.
    * ``hidden_flag_vars`` names every QUOTED variable whose value carries a
      flag-shaped word.  A quoted expansion stays ONE argv word, so the gate
      can see that a flag is hiding in there but cannot verify it against
      the script's parser.

    AN UNQUOTED variable expansion is WORD-SPLIT by bash, so the flags it
    carries ARE argv words and the gate can check them.  MEASURED
    2026-11-04: ``"$PY" -u molmetal/scripts/r4_lambda_only_run.py
    $EXTRA_FLAGS`` with ``EXTRA_FLAGS="--out-dir /tmp/out"`` was reported
    as PASS with ZERO flags checked — the flag was invisible because the
    expansion was kept as one token.  The same shape hides a bash array
    referenced WITHOUT the ``[@]`` subscript: ``$ARGS`` expands to the
    array's elements joined by spaces, which are separate argv words too.
    """
    tokens: list[str] = []
    unresolved: list[str] = []
    hidden: list[str] = []
    for token in segment[index + 1:]:
        match = GATE._ARRAY_REF.search(token)
        if match:
            name = match.group(1)
            value = env.get(name)
            if isinstance(value, list) or name in getattr(env, "conditional_appends", {}):
                live, _dead = GATE._array_elements(name, env)
                tokens.extend(live)
                continue
            unresolved.append(name)
            tokens.append(token)
            continue
        raw = GATE._unquote(token)
        expanded = GATE._expand(raw, env)
        quoted = len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'"
        if expanded != raw and not quoted and any(c.isspace() for c in expanded):
            # An UNQUOTED expansion is word-split by the shell, so its parts
            # are separate argv words — and therefore separate FLAGS.
            for part in GATE._shell_tokens(expanded):
                if GATE._VARREF.search(part):
                    # A PARTIALLY resolved expansion: the value still holds
                    # a reference the gate cannot read.
                    match = GATE._VARREF.search(part)
                    tokens.append(
                        _Unresolved(match.group(1) or match.group(4), part)
                    )
                    continue
                tokens.append(part)
            continue
        if expanded != raw and quoted:
            # A QUOTED expansion stays ONE argv word.  If that word carries
            # a flag-shaped token, the gate can see that a flag is hiding
            # there but cannot check it against the script's parser.
            parts = [part for part in GATE._shell_tokens(expanded) if part]
            if len(parts) > 1 and any(part.startswith("--") for part in parts):
                hidden.append(raw)
            if GATE._VARREF.search(expanded):
                # The value is not resolvable EITHER, so the gate cannot
                # even tell whether a flag is hiding in it.
                match = GATE._VARREF.search(expanded)
                tokens.append(_Unresolved(match.group(1) or match.group(4), raw))
                continue
            tokens.append(expanded)
            continue
        if GATE._VARREF.search(expanded):
            # An UNRESOLVED reference.  MEASURED 2026-11-04 (reviewer
            # finding M-2): ``"$PY" -u molmetal/scripts/foo.py $EXTRA_FLAGS``
            # was reported as PASS with ZERO flags checked — the reference
            # was neither a flag nor a gap, so a flag that hides in it was
            # invisible AND uncounted.  The gate now carries the reference
            # as an :class:`_Unresolved` token; whether it matters depends
            # on the POSITION it stands in (a value position is harmless).
            #
            # MEASURED 2026-11-04 (the same finding, still open after the
            # first repair): ``GATE._expand`` leaves an unresolvable
            # reference IN PLACE, so ``expanded == raw`` and the branch
            # above (``expanded != raw``) never fired for the very case the
            # reviewer named — an UNASSIGNED ``$EXTRA_FLAGS`` was still
            # invisible.  The test below is therefore on the EXPANDED text
            # alone, not on whether the expansion changed anything.
            match = GATE._VARREF.search(expanded)
            tokens.append(_Unresolved(match.group(1) or match.group(4), raw))
            continue
        tokens.append(expanded)
    return tokens, unresolved, hidden


#: PROCESS-WIDE cache of the AST-derived option vocabulary, keyed by
#: ``(path, mtime_ns)``.  MEASURED 2026-11-04: the cache used to be
#: PER-WRAPPER, so a script invoked by twenty wrappers was parsed twenty
#: times and a full-tree audit took 11.9 s — too slow for a §4 gate that
#: has to run inside a pre-flight.  The key carries the mtime, so a file
#: that changes is re-read; nothing is certified from a stale parse.
_declared_cache: dict[
    tuple[str, int], tuple[dict[str, str] | None, str | None, str]
] = {}


def _declared_flags_cached(
    script: Path,
) -> tuple[dict[str, str] | None, str | None, str]:
    """The declared options of ``script``, parsed once per (path, mtime)."""
    try:
        stamp = script.stat().st_mtime_ns
    except OSError:
        stamp = -1
    key = (str(script), stamp)
    if key not in _declared_cache:
        _declared_cache[key] = declared_flags(script)
    return _declared_cache[key]


def _audit_wrapper(path: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Return ``(violations, undecidable, disclosed, counters)`` for one wrapper."""
    violations: list[dict] = []
    undecidable: list[dict] = []
    disclosed: list[dict] = []
    counters = {
        "n_segments": 0,
        "n_invocations": 0,
        "n_flags_checked": 0,
        "n_shell_invocations": 0,
        "n_out_of_contract": 0,
        "n_data_lines": 0,
        # Wf-A269 wave-19 (2026-11-04): invocations of a script whose option
        # vocabulary is NOT enumerable, broken down by the MEASURED state
        # (see :func:`declared_flags`).  An invocation carrying NO flags has
        # nothing to certify, so it is disclosed rather than fatal — but it
        # is COUNTED AND DISCLOSED, never silently dropped (no silent caps).
        "n_vocab_empty": 0,
        "n_vocab_unreadable": 0,
        "n_vocab_broken": 0,
        "n_vocab_partial": 0,
        "n_vocab_no_flags": 0,
        # Wf-A269 wave-20 (2026-11-04): the runtime-transport residual, the
        # scripts that do not exist, the options that got no value, and the
        # flags hidden inside a quoted expansion.  Each is COUNTED and
        # DISCLOSED — never silently dropped.
        "n_runtime_payloads": 0,
        "n_runtime_residual": 0,
        "n_missing_scripts": 0,
        "n_missing_values": 0,
        "n_hidden_flags": 0,
        # The repo-local scripts this wrapper actually AUDITED.  MEASURED
        # 2026-11-04: the report's ``n_scripts`` used to be built from the
        # VIOLATIONS alone, so a clean tree reported "0 scripts" — a false
        # "nothing was audited" reading (the silent-caps shape).
        "scripts": set(),
    }
    rel = GATE._rel(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    env = GATE._collect_assignments(text)
    data_lines = _data_lines(text)
    declared_cache = _declared_cache
    # The wrapper's WORKING DIRECTORY.  ``cd`` moves it for the rest of the
    # wrapper, and a script named relative to that directory is still a
    # repo-local script (MEASURED 2026-11-04: a wrapper does ``cd
    # molmetal/scripts`` and then runs ``python r4_lambda_only_run.py``).
    cwd: Path | None = None

    def _declared_for(script: Path):
        return _declared_flags_cached(script)

    def _check_invocation(
        resolution: _Resolution,
        segment: list[str],
        lineno: int,
    ) -> None:
        """Check ONE resolved invocation against the script's own parser."""
        if resolution.status == "none":
            counters["n_out_of_contract"] += 1
            return
        if resolution.status == "undecidable":
            undecidable.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "undecidable_script",
                    "token": str(resolution.token),
                    "detail": (
                        f"{resolution.why}; the gate cannot prove that this "
                        "line does not run a repo-local script, so it cannot "
                        "certify the flags it passes"
                    ),
                }
            )
            return
        if resolution.status == "shell":
            # A SHELL that evaluates a command STRING.  The payload is the
            # command the wrapper actually launches at RUN TIME, so it is
            # audited here; a payload the gate cannot resolve is a counted
            # runtime residual, delegated to the sibling gate
            # ``scripts/_runtime_transport_gate.py`` and DISCLOSED here.
            counters["n_shell_invocations"] += 1
            payload = resolution.payload
            if not payload or not payload.strip():
                counters["n_runtime_residual"] += 1
                return
            counters["n_runtime_payloads"] += 1
            payload_segments = _split_segments(GATE._shell_tokens(payload))
            if not payload_segments:
                counters["n_runtime_residual"] += 1
                return
            for p_segment, p_redirect, p_array in payload_segments:
                if p_redirect or p_array:
                    continue
                for p_resolution in _resolve_core(p_segment, env):
                    _check_invocation(p_resolution, p_segment, lineno)
            return

        token = resolution.token or ""
        script = _repo_local_script(token, cwd)
        if script is None:
            counters["n_out_of_contract"] += 1
            return

        if not script.exists():
            # A repo-local script path that does NOT EXIST.  This is a
            # different — and more severe — defect than a coverage gap: the
            # launch would die with "No such file or directory", and a
            # message that says "the gate could not read the declared
            # options" implies the file IS there (MEASURED 2026-11-04,
            # reviewer finding FP-5).
            counters["n_invocations"] += 1
            counters["n_missing_scripts"] += 1
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "missing_script",
                    "script": GATE._rel(script),
                    "detail": (
                        f"the wrapper invokes {token}, which does not exist in "
                        "the tree; the launch would die with 'No such file or "
                        "directory' before the script ever runs"
                    ),
                }
            )
            return

        counters["n_invocations"] += 1
        counters["scripts"].add(GATE._rel(script))
        declared, error, state = _declared_for(script)
        argv, unresolved_arrays, hidden_flag_vars = _argv_after(
            resolution.tokens or segment, resolution.index or 0, env
        )
        flags, missing_values, unresolved_refs = flags_in_argv(argv, declared or {})

        if unresolved_arrays:
            names = ", ".join(sorted(set(unresolved_arrays)))
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "coverage_gap",
                    "script": GATE._rel(script),
                    "detail": (
                        f"the invocation of {GATE._rel(script)} expands the "
                        f"bash array(s) {names}, which this gate could not "
                        "read, so it cannot prove that the flags inside "
                        "them are declared; a check that cannot check must "
                        "not certify the launch"
                    ),
                }
            )

        if hidden_flag_vars:
            # A QUOTED expansion stays ONE argv word, so a flag-shaped word
            # inside it cannot be checked against the parser.
            #
            # DISCLOSED, NOT FATAL (Wf-A269 wave-19, 2026-11-04).  argparse's
            # own rule (MEASURED on CPython 3, see :func:`_looks_like_option`)
            # is that a token CONTAINING A SPACE is a POSITIONAL, so the blob
            # is not passed as an option at all: the flags inside it never
            # reach the script as options.  That is a stray POSITIONAL, not an
            # undeclared FLAG — so the §5.11 vocabulary question ("is this
            # flag declared?") is VACUOUS here, and failing the launch on it
            # would be a false attribution (the §6.269.28.5 class).
            #
            # It is ALSO a legal form in this repo: the sibling capability
            # check `molmetal/scripts/wf_argv_capability_check.py` declares
            # `--flags` as a whitespace-separated STRING payload
            # (``action="append"``) and its own help text says so:
            #     "--flags '--no-cuda-graphs --num-workers'"
            # MEASURED on the live invocation: that form parses and runs.
            #
            # So the gate counts it and prints it on every run (no silent
            # caps) and does NOT fail the launch: whether the blob is a
            # deliberate string payload or a wrapper that means to pass
            # several flags is not decidable from the argv alone.
            counters["n_hidden_flags"] += 1
            names = ", ".join(sorted(set(hidden_flag_vars)))
            disclosed.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "quoted_flag_blob",
                    "script": GATE._rel(script),
                    "detail": (
                        f"the invocation of {GATE._rel(script)} carries the "
                        f"quoted expansion(s) {names}, whose value holds a "
                        "flag-shaped word; a quoted expansion stays ONE argv "
                        "word, so argparse reads that word as a POSITIONAL and "
                        "the flags inside it never reach the script as options"
                    ),
                }
            )

        if unresolved_refs:
            # An UNRESOLVED reference standing where an OPTION could stand.
            # The gate cannot read it, so it cannot prove that the flags
            # inside it are declared — a gap the gate cannot close
            # (AGENTS.md §5.11 requirement 5, applied to the ARGV).
            names = ", ".join(sorted(set(unresolved_refs)))
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "coverage_gap",
                    "script": GATE._rel(script),
                    "detail": (
                        f"the invocation of {GATE._rel(script)} carries the "
                        f"unresolved shell reference(s) {names} where an option "
                        "could stand; the gate cannot read the value, so it "
                        "cannot prove that the flags inside it are declared"
                    ),
                }
            )

        if missing_values:
            # argparse raises "expected one argument" for these, so the
            # launch dies.  Counted and disclosed — the defect class is the
            # OPTION's missing value, not an unknown option.
            counters["n_missing_values"] += len(missing_values)
            for option, offending in missing_values:
                violations.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "kind": "missing_value",
                        "script": GATE._rel(script),
                        "flag": option,
                        "detail": (
                            f"{option} is passed to {GATE._rel(script)} without "
                            f"a value ({offending} follows it); argparse raises "
                            "'expected one argument', so the launch dies before "
                            "the run starts"
                        ),
                    }
                )

        if declared is None:
            # The vocabulary is not enumerable.  COUNT the state first:
            # which of the states it is decides what happens next, and the
            # census must be able to disclose it (no silent caps).
            state_key = f"n_vocab_{state}"
            if state_key in counters:
                counters[state_key] = counters.get(state_key, 0) + 1
            if state == "empty":
                # The vocabulary is empty AND KNOWN to be empty (the script
                # declares no options at all and does not read argv), so the
                # gate CAN decide: every flag on this line is undeclared.
                # This is NOT a coverage gap — it is a decidable violation,
                # and it is a SILENT one: a script with no parser ignores the
                # flag instead of aborting, so the wrapper configures nothing
                # while appearing to.
                #
                # A FLAGLESS invocation of such a script is legal: there is
                # no option to violate, and the gate can PROVE that (no
                # parser, no add_argument, no read of sys.argv) — so this
                # is a decidable PASS, not a gap.
                for flag in sorted(set(flags)):
                    counters["n_flags_checked"] += 1
                    violations.append(
                        {
                            "file": rel,
                            "line": lineno,
                            "kind": "undeclared_flag",
                            "script": GATE._rel(script),
                            "flag": flag,
                            "detail": (
                                f"{flag} is passed to {GATE._rel(script)}, which "
                                "declares NO options at all (no parser, no "
                                "add_argument, no read of sys.argv), so the "
                                "flag is SILENTLY IGNORED: the launch does "
                                "not abort, it just does not do what the "
                                "wrapper asked"
                            ),
                        }
                    )
                return
            # Fail-closed.  A script whose options the gate CANNOT read is a
            # gap the gate cannot close, so it FAILS (AGENTS.md §5.11
            # requirement 4; §6.269.17's DEAD-GATE class: a check that
            # skips what it could not read and still reports PASS is a check
            # that can PASS WITHOUT CHECKING).
            #
            # The gap is UNCONDITIONAL.  MEASURED 2026-11-04 (reviewer
            # finding B1): the gate used to make it non-fatal when the
            # invocation carried no flags, which contradicts requirement #4
            # ("a script whose declared flags cannot be read => VIOLATION,
            # fail-closed") and re-opens the exact shape the sibling gate
            # removed — a check that skipped what it could not read and
            # still reported PASS.
            #
            # The distinction that DOES hold is the one the gate already
            # draws: an EMPTY vocabulary (provably no options at all) is
            # DECIDABLE, and a flagless invocation of it is legal — see the
            # ``state == "empty"`` branch above.  An UNREADABLE vocabulary
            # is not decidable, and the gate says so.
            #
            # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5
            # rule #6 — the flagless disclosure that used to stand here:
            #
            #     if not flags:
            #         # DISCLOSED, not fatal: the invocation carries no
            #         # flags, so there is nothing to certify.  Counted,
            #         # never hidden.
            #         counters["n_vocab_no_flags"] = (
            #             counters.get("n_vocab_no_flags", 0) + 1
            #         )
            #         return
            #
            # The COUNTER stays (no silent caps): it now counts the
            # flagless coverage gaps that are REPORTED, so a reader can see
            # how many of them carried nothing to check.
            if not flags:
                counters["n_vocab_no_flags"] = (
                    counters.get("n_vocab_no_flags", 0) + 1
                )
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "coverage_gap",
                    "script": GATE._rel(script),
                    "detail": (
                        f"the gate could not read the declared options of "
                        f"{GATE._rel(script)} ({error or 'unknown'}), so it "
                        "cannot prove that the flags on this invocation — "
                        "nor any other flag on it — are declared; a check "
                        "that cannot check must not certify the launch"
                    ),
                }
            )
            return

        for flag in sorted(set(flags)):
            counters["n_flags_checked"] += 1
            if _argparse_accepts(declared, flag):
                continue
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "undeclared_flag",
                    "script": GATE._rel(script),
                    "flag": flag,
                    "detail": (
                        f"{flag} is passed to {GATE._rel(script)}, whose "
                        "parser does not declare it; the launch would abort "
                        'with "unrecognized arguments" before the run '
                        "starts (§5.11: the legal flag set is defined by "
                        "the script's own parser, not by a fixed allowlist)"
                    ),
                }
            )

    for lineno, line in GATE._logical_lines(text):
        tokens = GATE._shell_tokens(GATE._strip_detach_head(line))
        direct_segments = _split_segments(tokens)
        n_direct = len(direct_segments)
        # The command-substitution bodies are added BEFORE the data-line
        # filter, so the safety valve below sees them: a `$( ... )` inside a
        # multi-line quoted string is CODE, and the filter must not hide it.
        segments = _with_substitution_segments(direct_segments)
        if lineno in data_lines and not _data_line_invokes_a_script(segments, env):
            # A heredoc body or a multi-line quoted string is DATA.  The
            # safety valve above keeps any line that DOES name a repo-local
            # script in the scan, so the filter can never hide an invocation.
            counters["n_data_lines"] += 1
            continue

        for index_of_segment, (segment, is_redirect, is_array_value) in enumerate(segments):
            # MEASURED 2026-11-04 (reviewer finding, accounting nit): the
            # segment counter used to be incremented AFTER the redirect/array
            # early exit, so the census printed "17325 out of contract" out
            # of "16298 segment(s)" — an IMPOSSIBLE line that a reader could
            # not interpret.  Every segment is now counted, and
            # "out of contract" is a SUBSET of it.
            counters["n_segments"] += 1
            if is_redirect or is_array_value:
                counters["n_out_of_contract"] += 1
                continue
            if index_of_segment < n_direct and _is_cd(segment):
                # A ``cd`` in the WRAPPER's own line moves the working
                # directory.  A ``cd`` inside a command substitution does
                # NOT: it runs in a subshell, so it cannot move the
                # wrapper's directory (MEASURED 2026-11-04:
                # ``REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
                # && pwd)"`` — treating that ``cd`` as the wrapper's moved
                # the working directory to /home/hugo and cost the gate 27
                # live invocations).
                cwd = _cd_target(segment, env, cwd)
                continue
            for resolution in _resolve_core(segment, env):
                _check_invocation(resolution, segment, lineno)
    return violations, undecidable, disclosed, counters


def _is_cd(segment: list[str]) -> bool:
    """Is this segment a ``cd`` command?"""
    if not segment:
        return False
    index = _command_index(segment, {})
    if index is None:
        return False
    return GATE._unquote(segment[index]) == "cd"


def _cd_target(segment: list[str], env: dict, cwd: Path | None) -> Path | None:
    """The directory a ``cd`` segment moves to, or None when it cannot tell.

    The rule is deliberately STRICT: the target must be ONE literal word
    that names an existing directory.  A ``cd`` whose target the gate cannot
    resolve leaves the working directory UNCHANGED, and the gate then keeps
    resolving relative script paths against the repo root — the reading
    that holds for every wrapper in this repo, which spells its scripts
    relative to the repo root.
    """
    index = _command_index(segment, env)
    if index is None or index + 1 >= len(segment):
        return None
    target = GATE._expand(GATE._unquote(segment[index + 1]), env)
    if GATE._VARREF.search(target) or not target or any(c.isspace() for c in target):
        return None
    base = cwd or REPO_ROOT
    try:
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = base / candidate
        candidate = candidate.resolve()
    except OSError:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def _data_line_invokes_a_script(segments: list[tuple[list[str], bool, bool]], env: dict) -> bool:
    """Does a DATA line (heredoc body / multi-line string) name a script?

    This is the safety valve of the data-line filter: if the line that the
    scanner believes is data actually names a repo-local script, it is
    scanned anyway.  The filter can therefore never hide an invocation.
    """
    for segment, is_redirect, is_array_value in segments:
        if is_redirect or is_array_value:
            continue
        resolution = _resolve_segment(segment, env)
        if resolution.status == "literal" and _repo_local_script(resolution.token or ""):
            return True
    return False


def audit(wrappers: list[Path] | None = None) -> dict:
    """Audit every launch wrapper against the parser of the script it runs."""
    if wrappers is None:
        wrappers, coverage = GATE._population()
    else:
        _default, coverage = GATE._population()
        coverage = dict(coverage)
        coverage["population_source"] = "explicit --wrapper list"
        coverage["population_degraded"] = False
        coverage["n_scanned"] = len(wrappers)
        coverage["n_excluded"] = 0
        # FP-7 (MEASURED 2026-11-04): the ``--wrapper`` path used to set
        # ``coverage_pct = None`` while the denominator stayed 113, so the
        # owner's coverage line printed "None% scanned" — a reader could not
        # tell what fraction of the repo the run covered.  The percentage is
        # computed against the SAME denominator the owner reports.
        total = coverage.get("n_tracked_sh_total")
        coverage["coverage_pct"] = (
            round(100.0 * len(wrappers) / total, 1) if total else None
        )
        coverage["excluded"] = []

    violations: list[dict] = []
    undecidable: list[dict] = []
    disclosed: list[dict] = []
    scanned: list[str] = []
    scripts: set[str] = set()
    # Wf-A269 wave-20 (2026-11-04): the wrappers that contributed ZERO
    # audited invocations.  MEASURED 2026-11-04: 31 of the 101 in-contract
    # wrappers contributed nothing, including ``scripts/start.sh`` — a glob
    # that matches but audits nothing is the "silent caps" shape, and a
    # reader of the PASS line could not tell which wrappers the PASS
    # actually covers.
    zero_invocation_wrappers: list[str] = []
    totals = {
        "n_segments": 0,
        "n_invocations": 0,
        "n_flags_checked": 0,
        "n_shell_invocations": 0,
        "n_out_of_contract": 0,
        # Wf-A269 wave-19 (2026-11-04): invocations of a script whose option
        # vocabulary is not enumerable, by the MEASURED state, plus the count
        # of those that carried NO flags (nothing to certify — disclosed,
        # never fatal, never silently dropped).
        "n_vocab_empty": 0,
        "n_vocab_unreadable": 0,
        "n_vocab_broken": 0,
        "n_vocab_partial": 0,
        "n_vocab_no_flags": 0,
        # Wf-A269 wave-20 (2026-11-04): the runtime-transport residual (a
        # command the wrapper assembles at RUN TIME and the gate cannot
        # resolve), the scripts that do not exist, the options that got no
        # value, and the flags hidden inside a quoted expansion.
        "n_runtime_payloads": 0,
        "n_runtime_residual": 0,
        "n_missing_scripts": 0,
        "n_hidden_flags": 0,
        "n_missing_values": 0,
    }

    for wrapper in wrappers:
        if not wrapper.exists():
            violations.append(
                {
                    "file": GATE._rel(wrapper),
                    "line": 0,
                    "kind": "missing_wrapper",
                    "detail": (
                        f"{GATE._rel(wrapper)} does not exist; the gate cannot "
                        "audit a wrapper it cannot read"
                    ),
                }
            )
            continue
        rel = GATE._rel(wrapper)
        if rel not in scanned:
            scanned.append(rel)
        wrapper_violations, wrapper_undecidable, wrapper_disclosed, counters = (
            _audit_wrapper(wrapper)
        )
        violations.extend(wrapper_violations)
        undecidable.extend(wrapper_undecidable)
        disclosed.extend(wrapper_disclosed)
        for key in totals:
            totals[key] += counters.get(key, 0)
        if counters.get("n_invocations", 0) == 0:
            zero_invocation_wrappers.append(rel)
        scripts.update(counters.get("scripts", set()))
        for violation in wrapper_violations:
            script = violation.get("script")
            if script:
                scripts.add(script)

    # An undecidable invocation is a fail-closed COVERAGE GAP: the gate
    # cannot prove that the line does not run a repo-local script, so it
    # cannot certify the flags it passes.  It is reported as a violation of
    # its own kind, never as "clean".
    for item in undecidable:
        violations.append(
            {
                "file": item["file"],
                "line": item["line"],
                "kind": item["kind"],
                "detail": item["detail"],
            }
        )

    degraded = bool(coverage.get("population_degraded"))
    if violations or degraded:
        # A violation ALWAYS fails, even when nothing was scanned: a wrapper
        # the gate could not read is a fail-closed outcome, not a skip.
        verdict = "FAIL"
    elif not scanned:
        verdict = "SKIP"
    else:
        verdict = "PASS"

    return {
        "gate": "script-contract",
        "rule": (
            "AGENTS.md §5.11 — every flag a launch wrapper passes to a "
            "repo-local script must be declared by that script's own parser"
        ),
        "n_wrappers_scanned": len(scanned),
        "coverage": coverage,
        "excluded_residual": GATE._residual_census(coverage.get("excluded", [])),
        "wrappers": scanned,
        "n_invocations": totals["n_invocations"],
        "n_scripts": len(scripts),
        "scripts": sorted(scripts),
        "n_flags_checked": totals["n_flags_checked"],
        "n_shell_invocations": totals["n_shell_invocations"],
        "n_out_of_contract_segments": totals["n_out_of_contract"],
        "n_vocab_empty": totals["n_vocab_empty"],
        "n_vocab_unreadable": totals["n_vocab_unreadable"],
        "n_vocab_broken": totals["n_vocab_broken"],
        "n_vocab_partial": totals["n_vocab_partial"],
        "n_vocab_no_flags": totals["n_vocab_no_flags"],
        "n_runtime_payloads": totals["n_runtime_payloads"],
        "n_runtime_residual": totals["n_runtime_residual"],
        "n_missing_scripts": totals["n_missing_scripts"],
        "n_hidden_flags": totals["n_hidden_flags"],
        "n_missing_values": totals["n_missing_values"],
        "n_zero_invocation_wrappers": len(zero_invocation_wrappers),
        "zero_invocation_wrappers": zero_invocation_wrappers,
        "n_segments": totals["n_segments"],
        "n_undecidable": len(undecidable),
        "undecidable": undecidable,
        # Wf-A269 wave-19 (2026-11-04): findings that are NOT violations but
        # are printed on every run (no silent caps) — a quoted flag blob that
        # argparse reads as a positional.
        "n_disclosed": len(disclosed),
        "disclosed": disclosed,
        "n_violations": len(violations),
        "violations": violations,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument(
        "--wrapper",
        action="append",
        default=None,
        metavar="PATH",
        help="audit only this wrapper (repeatable); used to exercise the gate "
        "against a fixture outside the repo",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="accepted for interface parity with the other §4 gates; a "
        "violation already fails without it",
    )
    args = ap.parse_args(argv)

    wrappers = None
    if args.wrapper:
        wrappers = [Path(w) for w in args.wrapper]
    report = audit(wrappers)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        GATE._print_coverage(report, tag="script-contract")
        print(
            f"[script-contract] scanned {report['n_wrappers_scanned']} wrapper(s), "
            f"{report['n_invocations']} repo-local script invocation(s), "
            f"{report['n_flags_checked']} flag(s) checked"
        )
        print(
            f"[script-contract] CENSUS: {report['n_segments']} command segment(s); "
            f"{report['n_out_of_contract_segments']} out of contract, "
            f"{report['n_shell_invocations']} shell interpreter(s), "
            f"{report['n_undecidable']} undecidable"
        )
        if report["n_vocab_no_flags"]:
            print(
                f"[script-contract] COVERAGE GAP (fatal, requirement #4): "
                f"{report['n_vocab_no_flags']} invocation(s) of a script whose "
                "option vocabulary is not enumerable carried NO flags — the "
                "gate still FAILS them, because it cannot prove that such an "
                "invocation is option-free at RUN TIME.  By state: "
                f"empty={report['n_vocab_empty']}, "
                f"unreadable={report['n_vocab_unreadable']}, "
                f"broken={report['n_vocab_broken']}, "
                f"partial={report['n_vocab_partial']}"
            )
        if report["n_runtime_payloads"] or report["n_runtime_residual"]:
            print(
                f"[script-contract] RUNTIME TRANSPORT: "
                f"{report['n_runtime_payloads']} command string(s) assembled at "
                "RUN TIME were parsed and audited; "
                f"{report['n_runtime_residual']} could not be resolved "
                "statically — that residual belongs to the sibling gate "
                "scripts/_runtime_transport_gate.py (p3_runtime_transport_check)"
            )
        if report["n_missing_values"]:
            print(
                f"[script-contract] MISSING VALUES: {report['n_missing_values']} "
                "option(s) were passed without a value; argparse raises "
                "'expected one argument'"
            )
        if report["n_hidden_flags"]:
            print(
                f"[script-contract] HIDDEN FLAGS: {report['n_hidden_flags']} "
                "invocation(s) carry a flag-shaped word inside a QUOTED "
                "expansion; a quoted expansion stays ONE argv word, so "
                "argparse reads that word as a POSITIONAL and the flags inside "
                "it never reach the script as options.  DISCLOSED, not fatal: "
                "the form is legal when the receiving option is a "
                "whitespace-separated STRING payload (as in "
                "molmetal/scripts/wf_argv_capability_check.py --flags), and "
                "the argv alone cannot tell the two cases apart.  Sites: "
                + ", ".join(
                    f"{item['file']}:{item['line']}"
                    for item in report.get("disclosed", [])
                )
            )
        if report["n_zero_invocation_wrappers"]:
            names = ", ".join(report["zero_invocation_wrappers"])
            print(
                f"[script-contract] ZERO-INVOCATION WRAPPERS: "
                f"{report['n_zero_invocation_wrappers']} of "
                f"{report['n_wrappers_scanned']} scanned wrapper(s) contributed "
                "NO audited repo-local invocation — their PASS covers nothing "
                f"this gate could check: {names}"
            )
        for violation in report["violations"]:
            loc = f"{violation['file']}:{violation['line']}"
            print(f"  VIOLATION {loc} [{violation['kind']}] {violation['detail']}")
        if report["n_violations"] == 0:
            # SCOPED CLAIM.  MEASURED 2026-11-04 (reviewer finding FN-2): the
            # old line — "every flag a wrapper passes to a repo-local script
            # is declared by that script's own parser" — is a UNIVERSAL
            # claim the data does not support: 31 of the scanned wrappers
            # contributed no invocation this gate could see, and a runtime
            # payload the gate could not resolve is delegated to the sibling
            # gate.  The claim is now scoped to what the gate actually read.
            print(
                f"  every flag a wrapper passes to a repo-local script THAT "
                f"THIS GATE COULD RESOLVE ({report['n_invocations']} "
                f"invocation(s)) is declared by that script's own parser"
            )
        print(f"[script-contract] {report['verdict']}")

    if report["verdict"] == "SKIP":
        return 2
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
