#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""AGENTS.md §4 launcher-hardening gate — BLK-I1 (silent death) + BLK-I2 (pre-flight).

Two failure mechanisms have cost real runs in this repo, and they are NOT the
same bug, so the gate keeps them as separate findings:

  (a) process-group death — the parent shell exits and the trainer is SIGTERM'd
      because it was never detached. Fix: nohup / setsid / disown.
  (b) SIGPIPE silent death — `trainer | tee log` under `set -e` + `pipefail`.
      DataLoader / torch workers still hold the pipe open after the trainer
      exits, so `tee` sees EOF first; the trainer's next write takes SIGPIPE and
      the wrapper dies on the propagated 141. The failure lands AFTER the
      metrics you wanted, or before a checkpoint flush.

(2) is NOT fixed by adding nohup, and (1) is NOT fixed by touching pipefail.
A gate that only looked for one of them would let the other through, so both
are checked separately.

A THIRD, more severe finding came out of building this gate (see
`mangled_detach` below): six launchers had `setsid nohup` spliced onto a
backslash-continued line by an earlier hardening pass, which turns the two
words into literal argv tokens. `train_dmpnn_multitask_wasserstein.py` then
exits 2 with "unrecognized arguments: setsid nohup" and never trains. That is
a launcher that cannot run at all, so it outranks both silent-death mechanisms.

Design note — false positives are the failure mode here. A gate that fires on
healthy wrappers gets disabled inside a week and protects nothing, so:

  * `set -e` WITHOUT `pipefail` + `PIPESTATUS[0]` is the CORRECT tee pattern and
    is never a finding. Only `set -e` AND `pipefail` together are, because only
    that combination makes a `| tee` pipeline able to abort the wrapper.
  * A wrapper that another in-repo script invokes under `nohup`/`setsid` is
    already detached by its caller, so `no_detach` is a WARN, not a FAIL.
  * `no_preflight` is a WARN. Roughly a third of the launchers are panels,
    smokes, recovery helpers and CPU utilities for which pre-flight is not the
    binding constraint; making it FAIL would pin the gate to a number that
    cannot be reached without a rewrite wave.

Stdout invariant (AGENTS.md §4): PASS / FAIL / SKIP. Exit 0=PASS, 1=FAIL, 2=SKIP.

§5.9 single-venv reuse: `scripts/_wrapper_audit.py` already owns the shipped
canonical-surface check. This gate imports its markers rather than writing a
second venv regex, and adds the check `_wrapper_audit` does not make — the
NEGATIVE one (a bare `python`/`python3` actually invoking something), which its
presence-based marker cannot see.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Token patterns
# ---------------------------------------------------------------------------

#: The canonical single-venv interpreter, however it is spelled in-shell.
CANONICAL_PY = re.compile(r"(?:\$\{?PY(?:THON)?\}?|\.venv/bin/python)")

#: An interpreter invocation that is NOT the canonical venv one. Anchored on a
#: shell word boundary that excludes `python3` inside a path or a filename.
BARE_PY = re.compile(r"(?<![.\w/-])python[23](?![\w-])")

#: Something that runs a project trainer / analysis script.
TRAINER_CALL = re.compile(r"\.py\b")

#: A line that starts a python process. Broader than TRAINER_CALL on purpose:
#: most launchers hold the trainer in a variable (`${SCRIPT}`, `${TRAINER}`), so
#: an argv-position `setsid` on a line like
#:     CUDA_VISIBLE_DEVICES=0 ${PY} "${TRAINER}" --ckpt-dir X \
#:     setsid nohup "${COMMON_ARGS[@]}" | tee LOG
#: has no literal `.py` at all. Matching only on `.py` made this check pass
#: every real instance it exists to catch.
_STARTS_PYTHON = re.compile(r"\$\{?PY(?:THON)?\}?|(?<![.\w/-])python[23](?![\w-])|\.venv/bin/python")

#: A `| tee` pipeline. The tee must be the last stage of the pipeline.
TEE_STAGE = re.compile(r"\|\s*(?:\S+\s*\|\s*)*tee(?:\s|$)")

#: `set` lines. errexit and pipefail are tracked independently because only
#: their COMBINATION can abort a wrapper on a `| tee` pipeline.
_ERREXIT = re.compile(r"(?m)^\s*set\s+-[a-zA-Z]*e")
_PIPEFAIL = re.compile(r"(?m)^\s*set\s+(?:-[a-zA-Z]+\s+)*-o\s+pipefail\b|\bpipefail\b")

#: Wrappers that ARE detached, either by themselves or by an in-repo caller.
DETACH = re.compile(r"\b(?:nohup|setsid)\b|\bdisown\b")

#: A `&`-backgrounded trainer invocation that the wrapper then waits on.
_BACKGROUND_WAIT = re.compile(r"&\s*(?:;|$|\n)|^\s*wait\b", re.M)

#: pre-flight, by either entry point (AGENTS.md §4 names pre_flight_check.py;
#: scripts/start.sh reaches it through scripts/_pre_flight_check.sh).
PREFLIGHT = re.compile(r"pre_flight_check\.py|_pre_flight_check\.sh")

#: A shell command that cannot be a real trainer invocation — echoed text,
#: log-formatting and `die`/`log` helpers. These mention the trainer path or a
#: python word but start no process, so they must not anchor a finding.
_NON_COMMAND = re.compile(
    r"^\s*(?:echo|printf|log|die|warn|ok|comment|cat|test|:|\[\[?|#)\b",
    re.I,
)

#: A single-line `/* ... */` comment. `[^/]` before the closer rejects `/*/`,
#: which is a `case` glob (`/*) ;;`) rather than a comment terminator.
_BLOCK_COMMENT = re.compile(r"(?:(?<=\s)|(?<=^))/\*.*?\*/(?![*/])")


# ---------------------------------------------------------------------------
# `-c` snippet analysis (AGENTS.md §5.9)
# ---------------------------------------------------------------------------
#
# `python3 -c "<snippet>"` is a shell-side helper in this repo's wrappers
# (JSON parsing in the dashboard, an inline import probe). The exemption that
# covered it used to be UNCONDITIONAL: the comment claimed "no project trainer
# involved", but no code ever checked that claim, so a snippet importing a
# NON-STDLIB module (`torch`, `pandas`, `rdkit`, a repo module) under the
# SYSTEM interpreter -- the exact failure §5.9 exists to prevent ("the system
# interpreter has no torch/pandas and the trainer dies at import") -- was
# invisible to the gate. The exemption is now CONDITIONAL on the claim: the
# snippet must be STDLIB-ONLY, and every failure to prove that fails CLOSED
# (the line falls through and is flagged).

#: The stdlib module names (Python 3.10+). `None` on an older interpreter, and
#: a `-c` snippet is then treated as NOT exempt (fail closed toward flagging).
_STDLIB_MODULE_NAMES = getattr(sys, "stdlib_module_names", None)

#: A `-c` option token: `-c` at the end of the word, or `-c` followed by a
#: quote or whitespace. `-config` is not the `-c` option.
_DASH_C_OPTION = re.compile(r"^-c(?=$|[\s'\"])")

#: Sentinel for a RELATIVE import (`from . import x`): it names no top-level
#: module and is repo-local by construction, so it is never stdlib.
_RELATIVE_IMPORT = "<relative>"

#: A dynamic import inside a snippet names no module at parse time, so the
#: import census below cannot see it. Treated as NOT stdlib-only.
_DYNAMIC_IMPORT = re.compile(r"\b(?:__import__|import_module)\s*\(")


def _extract_dash_c_snippet(rest: str) -> str | None:
    """The Python source of a `python3 -c` snippet, or None if not extractable.

    `rest` is the text that follows the `python3` token on the line. The
    snippet may be glued to the option (`-c"..."`, `-c'...'`) or a separate
    word (`-c "..."`). Only a QUOTED word is recognised: an unquoted snippet
    is shell-interpreted before Python sees it, so its source cannot be
    recovered faithfully => None => the caller fails closed (flags the line).
    """
    if not _DASH_C_OPTION.match(rest):
        return None
    tail = rest[2:].lstrip()
    if not tail or tail[0] not in ("'", '"'):
        return None
    quote = tail[0]
    out: list[str] = []
    i = 1
    while i < len(tail):
        ch = tail[i]
        if quote == '"' and ch == "\\" and i + 1 < len(tail):
            nxt = tail[i + 1]
            # POSIX: inside double quotes a backslash is special only before
            # `$`, a backtick, `"` and `\`; anywhere else it stays literal.
            if nxt in ("$", "`", '"', "\\"):
                out.append(nxt)
                i += 2
                continue
            out.append(ch)
            i += 1
            continue
        if ch == quote:
            return "".join(out)
        out.append(ch)
        i += 1
    return None  # unterminated quote


def _snippet_imports(source: str) -> tuple[list[str], bool]:
    """(top-level imported module names, parsed_ok) for a `-c` snippet.

    Every import form is collected: `import X`, `import X.Y`, `from X import
    ...` and `from X.Y import ...`. A RELATIVE import (`from . import x`) names
    no top-level module and is repo-local by construction => reported as
    `_RELATIVE_IMPORT`, which is never in the stdlib.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return ([], False)
    mods: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            mods.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                mods.append(_RELATIVE_IMPORT)
            elif node.module:
                mods.append(node.module.split(".")[0])
            else:
                mods.append(_RELATIVE_IMPORT)
    return (mods, True)


def _dash_c_snippet_is_stdlib_only(rest: str) -> bool:
    """True when a `python3 -c` snippet is a stdlib-only shell-side helper.

    Fail-closed everywhere: an unextractable snippet, an unparseable
    snippet, a missing `sys.stdlib_module_names`, a dynamic import, or ANY
    non-stdlib import (a project module, `torch`, `pandas`, ...) => NOT
    exempt => the line falls through to the trainer-literal test.
    """
    stdlib = _STDLIB_MODULE_NAMES
    if not stdlib:
        return False
    source = _extract_dash_c_snippet(rest)
    if source is None:
        return False
    mods, parsed = _snippet_imports(source)
    if not parsed:
        return False
    if _DYNAMIC_IMPORT.search(source):
        return False
    return all(mod in stdlib for mod in mods)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def logical_lines(text: str) -> list[tuple[int, str]]:
    """(start_lineno, code) for comment-free, continuation-joined lines."""
    return [(s, c) for s, _e, c in _logical_spans(text)]


def _logical_spans(text: str) -> list[tuple[int, int, str]]:
    """(start_lineno, end_lineno, code) for comment-free, joined lines.

    The end line is carried so a caller can prove the scan reached the end of
    the file. Joining backslash continuations is not a nicety here: every
    launcher in this repo puts the trainer flags on continuation lines, and the
    `mangled_detach` defect is only visible once the join happens. Scanning
    physical lines would see `setsid nohup` as the head of its own command and
    pass it.
    """
    out: list[tuple[int, int, str]] = []
    buf: list[str] = []
    start = 0
    end = 0

    def flush() -> None:
        nonlocal start, end
        if buf:
            joined = " ".join(buf)
            hash_at = joined.find(" #")
            if hash_at != -1:
                joined = joined[:hash_at]
            if joined.strip():
                out.append((start, end, joined))
            buf.clear()

    in_block = False
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw
        if in_block:
            close = line.find("*/")
            if close == -1:
                continue
            line, in_block = line[close + 2 :], False
        # Strip a `/* ... */` block ONLY when both delimiters are on this same
        # line. A multi-line opener is left alone on purpose: shell wrappers
        # use `/*` as a glob and as a `case` pattern (`/*)` in
        # launch_wf125_p1_controlled_pair.sh:118), and a stripper that treats
        # either as a comment opener silently swallows the rest of the file —
        # which is how this gate first "lost" two real findings. Leaving a few
        # comment lines in the scan is safe; dropping code is not.
        line = _BLOCK_COMMENT.sub(" ", line)
        if not buf and (line.lstrip().startswith("#") or not line.strip()):
            continue
        stripped = line.rstrip()
        end = i
        if stripped.endswith("\\"):
            if not buf:
                start = i
            buf.append(stripped[:-1].strip())
            continue
        if not buf:
            start = i
        buf.append(stripped.strip())
        flush()
    flush()
    return out


def _command_lines(ll: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Logical lines that actually start a process (not echo/log/die/…)."""
    return [(n, l) for n, l in ll if not _NON_COMMAND.match(l)]


def _all_invocations(ll: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Logical lines that invoke a project .py through some interpreter."""
    return [
        (n, l)
        for n, l in _command_lines(ll)
        if TRAINER_CALL.search(l) and (CANONICAL_PY.search(l) or BARE_PY.search(l))
    ]


def _long_running_invocations(
    ll: list[tuple[int, str]], tvars: set[str]
) -> list[tuple[int, str]]:
    """Command lines that invoke a multi-hour trainer.

    Deliberately NOT filtered through `_all_invocations`, which requires a
    literal `.py` on the line. The wrappers that hold their trainer in a
    variable invoke it as `exec "${PY}" "${TRAINER}"` — no literal extension
    anywhere on the line — so routing through that filter found zero of them
    and silently disabled the detach check on exactly the wrappers that needed
    it.
    """
    out = []
    for n, l in _command_lines(ll):
        if not (CANONICAL_PY.search(l) or BARE_PY.search(l)):
            continue
        if TRAINER_LITERAL.search(l):
            out.append((n, l))
            continue
        if any(
            re.search(r"\$\{?" + re.escape(v) + r"(?:\[\])?\}?", l) for v in tvars
        ):
            out.append((n, l))
    return out


def tvars_on_line(line: str, tvars: set[str]) -> bool:
    """True when the line references one of the wrapper's trainer variables."""
    return any(
        re.search(r"\$\{?" + re.escape(v) + r"(?:\[\])?\}?", line) for v in tvars
    )


def _long_running(
    line: str, tvars: set[str]
) -> bool:
    """True when this invocation is a multi-hour trainer, not a CPU gate.

    Detachment and SIGPIPE both only matter for a process that outlives its
    parent shell and holds worker threads. `pre_flight_check.py`,
    `per_wave_checklist.py`, `wave_budget_tracker.py` and friends are seconds-
    long CPU gates: nohup-ing them is cargo-culting, and their pipelines run
    under `if` conditions that already consume the status.
    """
    if TRAINER_LITERAL.search(line):
        return True
    return tvars_on_line(line, tvars)


def _has_flag_guard(line: str) -> bool:
    """True when a pipeline already neutralises a non-zero right-hand stage.

    `cmd | tee LOG || true` is the documented fix for mechanism (b) and is used
    by launch_cfm_*.sh and launch_dmpnn_clean5*.sh on disk, so the gate must
    accept it rather than re-flag the fix as the bug.
    """
    return bool(re.search(r"\|\|\s*(?:true|:\s*$)", line))


def _pipefail_disabled_for(ll: list[tuple[int, str]], lineno: int) -> bool:
    """True when ``set +o pipefail`` is in force at ``lineno``.

    This is the ONLY cross-line pattern that neutralises mechanism (b) while
    keeping ``PIPESTATUS[0]`` usable. Capturing ``rc=${PIPESTATUS[0]}`` on the
    line AFTER a ``cmd | tee LOG`` pipeline is NOT: under ``set -e`` errexit
    fires at the pipeline and never reaches the capture. Measured 2026-10-09 —
    the capture never ran and the wrapper still died at the pipeline.

    Accepted guards are, in order of directness:
      * ``|| true`` / ``|| :`` on the pipeline line itself (``_has_flag_guard``),
      * ``set +o pipefail`` in force before the pipeline (here).
    """
    at = next((i for i, (n, _c) in enumerate(ll) if n == lineno), None)
    if at is None:
        return False
    disabled = False
    for _n, code in ll[:at]:
        if re.search(r"set\s+\+o\s+pipefail", code):
            disabled = True
        elif re.search(r"set\s+(?:-\w+\s+)*o\s+pipefail|set\s+-\w*o?\s*pipefail", code):
            disabled = False
    return disabled


def _status_consumed(line: str) -> bool:
    """True when errexit can never act on this pipeline's exit status.

    A pipeline in an `if`/`while`/`until` condition, or one whose status is
    captured by `|| { ... }` / `|| exit N`, does not abort the wrapper — bash
    tests the condition and carries on. Four of the five SIGPIPE candidates this
    gate found on disk are exactly that shape (`if pre_flight_check | tee | grep
    -q PASS; then`), and flagging them would be the cry-wolf outcome.
    """
    stripped = line.strip()
    if re.match(r"(?:if|elif|while|until|!|\[\[?)\b", stripped):
        return True
    # `... | tee LOG || { ... }` / `|| exit N` — the status is handled.
    return bool(re.search(r"\|\|\s*(?:\{|\[|exit\b|die\b|return\b)", line))


def _trainer_vars(code: str) -> set[str]:
    """Shell variable names assigned a `train_*.py` path anywhere in the wrapper.

    Long-running trainers are held in variables (`TRAINER`, `SCRIPT`,
    `WF146_TRAINER`, ...), so detecting them by literal `.py` in the invocation
    line misses exactly the lines that matter. Resolving the assignments once
    per wrapper is what lets the SIGPIPE and detach checks tell a 3-hour trainer
    apart from a 3-second CPU gate.
    """
    out: set[str] = set()
    for m in re.finditer(
        r"(?m)^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[\"']?[^\"'\n]*"
        r"\btrain_[a-z0-9_]+\.py",
        code,
    ):
        out.add(m.group(1))
    return out


#: A literal long-running trainer entry point (as opposed to a CPU gate such as
#: pre_flight_check.py / per_wave_checklist.py, which has no worker pool and
#: cannot produce the late flush that makes mechanism (b) fire).
TRAINER_LITERAL = re.compile(r"\btrain_[a-z0-9_]+\.py\b")


def _detached_caller_map(scripts_dir: Path) -> dict[str, str]:
    """wrapper filename -> the script that invokes it under nohup/setsid.

    A wrapper run as `nohup bash scripts/launch_x.sh` is already in its own
    session; requiring it to also nohup its own trainer would be a false
    positive on the most common way these are launched.
    """
    out: dict[str, str] = {}
    for src in sorted(scripts_dir.glob("*.sh")):
        try:
            code = "\n".join(l for _, l in logical_lines(src.read_text(errors="replace")))
        except OSError:
            continue
        for m in re.finditer(
            r"\b(?:nohup|setsid)\b[^|;&\n]*?\bbash\s+(?:[\w./-]*/)?([\w.-]+\.sh)", code
        ):
            out.setdefault(m.group(1), src.name)
    return out


# ---------------------------------------------------------------------------
# Per-wrapper audit
# ---------------------------------------------------------------------------


def audit_wrapper(path: Path, detached_by: dict[str, str] | None = None) -> dict:
    """Classify one wrapper. Returns a record dict; never raises."""
    detached_by = detached_by or {}
    rel = _rel(path)
    record: dict = {
        "file": rel,
        "errexit": False,
        "pipefail": False,
        "has_tee_pipe": False,
        "has_detach": False,
        "detached_by_caller": None,
        "has_preflight": False,
        "canonical_venv": False,
        "n_trainer_calls": 0,
        "findings": [],
    }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        record["findings"].append(
            {
                "severity": "WARN",
                "kind": "unreadable",
                "line": 0,
                "detail": f"cannot read wrapper: {exc}",
            }
        )
        return record

    ll = logical_lines(text)
    code = "\n".join(l for _, l in ll)
    tvars = _trainer_vars(code)
    record["errexit"] = bool(_ERREXIT.search(code))
    record["pipefail"] = "pipefail" in code
    record["has_detach"] = bool(DETACH.search(code))
    record["detached_by_caller"] = detached_by.get(path.name)
    record["has_preflight"] = bool(PREFLIGHT.search(code))
    record["canonical_venv"] = "/.venv/bin/python" in code
    record["trainer_vars"] = sorted(tvars)

    trainers = _long_running_invocations(ll, tvars)
    record["n_trainer_calls"] = len(trainers)

    # --- mechanism (b): SIGPIPE silent death ------------------------------
    # Only errexit AND pipefail together let a `| tee` stage abort the wrapper.
    # errexit alone is the correct pattern (the tee stage's own status becomes
    # the pipeline status, which is 0), and PIPESTATUS[0] reads the trainer's
    # real exit code. A pipeline whose status an `if`/`||` already consumes
    # cannot abort anything, so it is not a finding.
    sigpipe_sites = [
        (n, l)
        for idx, (n, l) in enumerate(trainers)
        if TEE_STAGE.search(l)
        and record["errexit"]
        and record["pipefail"]
        and not _has_flag_guard(l)
        and not _pipefail_disabled_for(ll, n)
        and not _status_consumed(l)
    ]
    record["has_tee_pipe"] = any(TEE_STAGE.search(l) for _, l in trainers)
    if sigpipe_sites:
        for n, _ in sigpipe_sites:
            record["findings"].append(
                {
                    "severity": "FAIL",
                    "kind": "sigpipe_unprotected",
                    "line": n,
                    "detail": (
                        "trainer is piped into tee under `set -e` + `pipefail` with "
                        "no `|| true` on the pipeline. DataLoader workers still hold "
                        "the pipe open after the trainer exits, so tee takes EOF "
                        "first, the trainer's next write takes SIGPIPE, and the 141 "
                        "aborts the wrapper — after the metrics, or before a "
                        "checkpoint flush (BLK-I1 mechanism b). Drop pipefail and "
                        "read PIPESTATUS[0], or append `|| true` to the tee."
                    ),
                }
            )

    # --- mangled detach (outranks both mechanisms) ------------------------
    # A `setsid`/`nohup` token is in COMMAND position — a correct detach — when
    # everything before it is a command prefix: nothing at all, env assignments
    # (`CUDA_VISIBLE_DEVICES=0 setsid nohup cmd`), or an earlier detach word
    # (`setsid nohup cmd`, where `nohup` is `setsid`'s argument). `setsid`'s own
    # options count as command position too: `setsid --wait nohup cmd` is the
    # form that makes `rc=$?` after the command meaningful, because bare `setsid`
    # may fork and return 0 before the trainer exits. Anything else
    # means a backslash continuation already started the trainer and these words
    # are being handed to it as argv tokens.
    _CMD_PREFIX = re.compile(
        r"(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S*)|setsid(?:\s+--?[A-Za-z-]+)*|nohup)"
        r"(?:\s+(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S*)|setsid(?:\s+--?[A-Za-z-]+)*|nohup))*"
    )
    # A command inside `if` / `while` / `until` still starts at the keyword:
    # `if CUDA_VISIBLE_DEVICES=0 setsid --wait "${PY}" trainer; then` is a
    # correctly detached command, and the gate reported it as mangled until the
    # keyword was stripped before the prefix test (2026-10-26, two launchers).
    # Stripping cannot hide a real mangled detach: those heads start with a
    # command word (`${PY}` / `python`) after an env assignment, which the
    # prefix pattern does not accept.
    _SHELL_KEYWORD_PREFIX = re.compile(
        r"^(?:(?:if|elif|while|until|then|else|do|!|time|command)\s+)+"
    )
    for n, line in _command_lines(ll):
        for m in re.finditer(r"(?<!\S)(setsid|nohup)(?!\S)", line):
            head = _SHELL_KEYWORD_PREFIX.sub("", line[: m.start()].strip())
            if not head or _CMD_PREFIX.fullmatch(head):
                continue  # command position — a correct detach
            if not _STARTS_PYTHON.search(line):
                continue  # not a python launch; out of this gate's scope
            record["findings"].append(
                {
                    "severity": "FAIL",
                    "kind": "mangled_detach",
                    "line": n,
                    "detail": (
                        f"'{m.group(1)}' appears at argv position, not in command "
                        "position: the backslash continuation already started the "
                        "trainer, so these two words are passed TO the trainer as "
                        "arguments. The trainer exits 2 with 'unrecognized "
                        "arguments: setsid nohup' and never runs. MEASURED against "
                        "train_dmpnn_multitask_wasserstein.py::parse_args. Move the "
                        "detach onto the command's first line, or drop it where "
                        "the invocation is already `${PY} trainer`."
                    ),
                }
            )

    # --- mechanism (a): process-group death -------------------------------
    # `exec` is detachment by another name: it REPLACES the wrapper shell with
    # the trainer, so there is no parent left whose exit could SIGTERM it. Four
    # wrappers use this shape (launch_wf388 / launch_wf409a3) and flagging them
    # would be the cry-wolf outcome.
    foreground = [
        (n, l)
        for n, l in trainers
        if not DETACH.search(l)
        and not re.search(r"(?:^|[;&|(]|\s)exec\s", l)
        and not _BACKGROUND_WAIT.search(l)
    ]
    if foreground and not record["has_detach"]:
        severity = "WARN" if record["detached_by_caller"] else "FAIL"
        n, _ = foreground[0]
        via = (
            f"; it is invoked under nohup/setsid by {record['detached_by_caller']}, "
            "so the caller already detaches the whole process group"
            if record["detached_by_caller"]
            else ""
        )
        record["findings"].append(
            {
                "severity": severity,
                "kind": "no_detach",
                "line": n,
                "detail": (
                    "launches a trainer in the foreground with no nohup/setsid/"
                    "disown and no `&`. If the invoking shell exits, the trainer is "
                    "SIGTERM'd with no traceback (BLK-I1 mechanism a)" + via
                ),
            }
        )

    # --- BLK-I2: pre-flight not enforced in this file --------------------
    if trainers and not record["has_preflight"]:
        n, _ = trainers[0]
        record["findings"].append(
            {
                "severity": "WARN",
                "kind": "no_preflight",
                "line": n,
                "detail": (
                    "launches a trainer but does not gate on pre_flight_check.py in "
                    "this file, so a STALE or FAILING pre-flight report does not stop "
                    "it (BLK-I2, AGENTS.md §4). WARN rather than FAIL: panels, "
                    "smokes and recovery helpers legitimately do not need the "
                    "retrain gate."
                ),
            }
        )

    # --- §5.9 single-venv -------------------------------------------------
    for n, line in _command_lines(ll):
        for m in BARE_PY.finditer(line):
            rest = line[m.end() :].strip()
            # `-c` / `-` / heredoc is a shell-side helper (JSON parsing in the
            # dashboard, an inline import probe) — no project trainer involved.
            # The `-c` exemption is CONDITIONAL on that claim: it holds only
            # when the snippet is STDLIB-ONLY. A snippet that imports a
            # third-party module (torch / pandas / rdkit / a repo module) runs
            # under the SYSTEM interpreter — the exact failure §5.9 exists to
            # prevent ("the system interpreter has no torch/pandas and the
            # trainer dies at import") — so it must NOT be exempt.
            if _DASH_C_OPTION.match(rest):
                if _dash_c_snippet_is_stdlib_only(rest):
                    continue
            elif rest.startswith("-") or rest.startswith("<<"):
                continue
            # Same trap as the detach check: a wrapper holding its trainer in a
            # variable writes `python3 "${TRAINER}"`, with no `.py` literal.
            if not (TRAINER_LITERAL.search(line) or tvars_on_line(line, tvars)):
                continue
            record["findings"].append(
                {
                    "severity": "FAIL",
                    "kind": "bare_interpreter",
                    "line": n,
                    "detail": (
                        f"`{m.group(0)}` invokes a project .py outside "
                        "/home/hugo/codes/mol-metal/.venv/bin/python. The system "
                        "interpreter has no torch/pandas and the trainer dies at "
                        "import (AGENTS.md §5.9)."
                    ),
                }
            )
            break

    return record


# ---------------------------------------------------------------------------
# Repo-wide audit
# ---------------------------------------------------------------------------


def audit(scripts_dir: Path | None = None) -> dict:
    """Audit every wrapper in scripts/. Returns a report dict; never raises."""
    scripts_dir = scripts_dir or SCRIPTS_DIR
    detached_by = _detached_caller_map(scripts_dir)

    records = []
    for wrapper in sorted(scripts_dir.glob("*.sh")):
        records.append(audit_wrapper(wrapper, detached_by))

    findings = [
        {**f, "file": r["file"]} for r in records for f in r["findings"]
    ]
    # One finding per (file, line, kind): a mangled splice contains BOTH
    # `setsid` and `nohup`, and the pair is one defect, not two.
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for f in findings:
        key = (f["file"], f["line"], f["kind"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    findings = deduped
    hard = [f for f in findings if f["severity"] == "FAIL"]
    warn = [f for f in findings if f["severity"] == "WARN"]
    by_kind: dict[str, int] = {}
    for f in findings:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1

    clean = [r["file"] for r in records if not r["findings"]]

    # §5.9 canonical-surface markers, imported rather than re-implemented.
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_wrapper_audit", scripts_dir / "_wrapper_audit.py"
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            missing_surface = sorted(
                r["file"]
                for r in records
                if r["file"].startswith("scripts/launch_wf")
                and mod.classify(mod.surface_check(REPO_ROOT / r["file"]), REPO_ROOT / r["file"])[0]
                != "REAL"
            )
        else:
            missing_surface = []
    except Exception:
        missing_surface = []

    return {
        "gate": "launcher_hardening",
        "rule": (
            "AGENTS.md §4 launch wrappers must not die silently: no unprotected "
            "`trainer | tee` under errexit+pipefail (BLK-I1b), no trainer launched "
            "in an undetached foreground (BLK-I1a), no `setsid`/`nohup` stranded in "
            "argv position, pre-flight gated in-file (BLK-I2), and the canonical "
            "single-venv interpreter only (AGENTS.md §5.9)"
        ),
        "n_wrappers_scanned": len(records),
        "n_fail": len(hard),
        "n_warn": len(warn),
        "n_findings_by_kind": by_kind,
        "n_clean": len(clean),
        "clean": clean,
        "wrappers": records,
        "failures": hard,
        "warnings": warn,
        "wrapper_audit_non_real": missing_surface,
        "verdict": "PASS" if not hard else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat WARN findings as FAIL (exit 1) instead of advisory",
    )
    args = ap.parse_args(argv)

    report = audit()

    if args.json:
        import json

        print(json.dumps(report, indent=2))
    else:
        print(
            f"[launcher-hardening] scanned {report['n_wrappers_scanned']} wrapper(s): "
            f"{report['n_fail']} FAIL, {report['n_warn']} WARN, "
            f"{report['n_clean']} clean"
        )
        for f in report["failures"] + report["warnings"]:
            loc = f"{f['file']}:{f['line']}" if f["line"] else f["file"]
            print(f"  {f['severity']} {loc} [{f['kind']}] {f['detail']}")
        if not report["failures"] and not report["warnings"]:
            print("  every wrapper is hardened: detached, tee-protected, pre-flight gated, single-venv")
        if report["verdict"] == "FAIL":
            print(
                f"[launcher-hardening] FAIL — {report['n_fail']} hard finding(s); "
                "the wrappers listed as FAIL cannot complete a run as written"
            )
        elif args.strict:
            print(
                f"[launcher-hardening] FAIL (--strict) — {report['n_warn']} advisory "
                "finding(s) promoted to FAIL"
            )
        else:
            print("[launcher-hardening] PASS — no hard findings; advisory findings above")

    if args.strict and (report["warnings"] or report["failures"]):
        return 1
    if report["failures"]:
        return 1
    if not report["wrappers"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())