# Licensed under the Apache License, Version 2.0.
r"""Static detectors for the launch-gate CONTRACT (Wf-A269 wave-18, 2026-11-03).

WHY THIS IS A SEPARATE MODULE
------------------------------
A guard that cannot fail is the defect AGENTS.md §6.269.17 names, and a guard
whose detector is buried inside a test file cannot be MUTATION-tested: the test
cannot plant a defect and watch the detector flip, because the detector and
the assertion are the same code.  So the detectors live here, they are pure
functions of TEXT, and the test drives them with both the real wrappers and
deliberately broken copies.

THE CONTRACT (what every launcher that gates on the pre-flight must satisfy)
----------------------------------------------------------------------------
  C1  the wrapper actually invokes ``pre_flight_check.py``;
  C2  the invocation passes a report PATH via ``--json`` (a bare ``--json``
      writes the payload to stdout, which is not a file a consumer can read);
  C3  the report path is NOT the target of a merged redirect (``> path 2>&1``)
      — the tool writes its gate log to stderr and the payload to stdout, so a
      merged capture produces a file that is not JSON at all (MEASURED on the
      published report: 190 245 B / 1 826 lines, first ``{`` at line 270,
      ``json.loads`` => "Expecting value: line 1 column 2");
  C4  the PREVIOUS report is removed (``rm -f``) BEFORE the invocation — a
      crashed or killed pre-flight must not leave the last run's PASS report
      on disk for the consumer to read ("stale is treated as missing");
  C5  no live occurrence of the TIMEOUT-BLIND grep idiom
      (``grep`` over ``"fail": 0``) — it cannot see a TIMEOUT at all, so a
      report with 1 TIMEOUT and 0 FAILs matched it (Wf-A135);
  C6  the VERDICT is read from the repo's own FAIL-CLOSED consumer
      (``molmetal/scripts/wf_launch_gate_verdict.py``), either directly or
      through a local helper that invokes it.

C1-C4 and C6 are the repairs of Wf-A269 wave-17; C5 was already satisfied by
the 16 wrappers that use the consumer.  The point of stating C5 anyway is that
a future wrapper copied from the pre-fix template must not reintroduce it.

WHAT THESE DETECTORS DO NOT COVER (stated, not implied)
-------------------------------------------------------
* An INLINE trailing comment (``cmd --flag # note``) is stripped only when the
  ``#`` is preceded by a space; a ``#`` inside a quoted string is not a
  comment and is not stripped.  A wrapper whose trainer invocation ends with a
  quoted ``#`` would therefore be mis-scanned.
* The report path is compared as a SHELL EXPRESSION (``${VAR}/name.json``), so
  two wrappers that write to the same file through different spellings are not
  detected as colliding.  :func:`resolved_report_path` narrows that by
  expanding the variables the wrapper itself assigns.
* These are TEXT detectors.  They prove the SHAPE of the gate, not that the
  gate passes on this machine.  The behavioural half of the guard lives in the
  test file and drives ``wf_launch_gate_verdict.check_preflight`` directly.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

#: The pre-flight producer.
PREFLIGHT_TOOL = "pre_flight_check.py"

#: The FAIL-CLOSED verdict consumer.
CONSUMER_TOOL = "wf_launch_gate_verdict.py"

#: The consumer's report-kind argument for a pre-flight report.
CONSUMER_KIND = "preflight"

#: A shell assignment: NAME=value, optionally prefixed by export/local/etc.
_ASSIGN = re.compile(
    r"^\s*(?:export\s+|declare\s+-\w+\s+|readonly\s+|local\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)=(.*)$"
)

#: A shell variable reference.
_VARREF = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-=+?%!])([^}]*))?\}"
    r"|\$([A-Za-z_][A-Za-z0-9_]*)"
)

#: ``--json`` with its value (the value may be quoted).
_JSON_ARG = re.compile(r"--json[=\s]+(\"[^\"]*\"|'[^']*'|\S+)")

#: A ``grep`` that looks for a literal ``"fail": 0`` — the TIMEOUT-blind idiom.
_TIMEOUT_BLIND_GREP = re.compile(r"grep\b[^\n]*?\"fail\"\s*:\s*0")

#: A heredoc opener: ``<<'EOF'`` / ``<<EOF`` / ``<<-EOF`` / ``<<"EOF"``.
#:
#: The ``(?<!<)`` guard matters: a HERE-STRING (``<<< "$x"``) contains ``<<``
#: and would otherwise be read as a heredoc whose terminator is ``"$x"`` —
#: which silently swallows the rest of the file, including the very lines the
#: detector is looking for.  That is a detector that reports "no finding"
#: because it stopped reading, which is the §6.269.17 dead-gate class wearing
#: a different hat.
_HEREDOC_OPEN = re.compile(
    r"(?<!<)<<-?\s*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))"
)


def strip_comments(text: str) -> str:
    """Drop comment lines and inline trailing comments.

    Two levels, because the repo's wrappers keep the superseded rendering of a
    repaired block in comment blocks (AGENTS.md §5.5 rule #6) — scanning those
    would pin every repaired flag as broken forever, which is how a gate gets
    disabled instead of fixed.
    """
    out: List[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        hash_at = stripped.find(" #")
        if hash_at != -1:
            stripped = stripped[:hash_at].rstrip()
        if stripped:
            out.append(stripped)
    return "\n".join(out)


def logical_lines(text: str) -> List[Tuple[int, str]]:
    """Return ``(lineno, code)`` for comment-free, continuation-joined lines.

    Every launcher in this repo puts the trainer/pre-flight flags on
    continuation lines, so a per-physical-line scan sees ``--json`` on one line
    and the tool on another and cannot tell they belong together.

    HEREDOC BODIES ARE SKIPPED.  The repo preserves the superseded rendering
    of a repaired block two ways: as ``#`` comments, and as a quoted heredoc
    (e.g. ``scripts/_wrapper_template.sh`` keeps the retired template inside
    ``cat <<'WFA269_SUPERSEDED_RENDERING_EOF_DO_NOT_REUSE'``).  A heredoc body
    is DATA, not code — the shell writes it to a file or a stream — so a
    detector that scanned it would read the OLD code as if it were LIVE, and
    would then "find" a defect that was repaired two waves ago.  That is how a
    gate gets disabled instead of fixed.
    """
    out: List[Tuple[int, str]] = []
    buf: List[str] = []
    start = 0
    heredoc_term: Optional[str] = None
    for i, raw in enumerate(text.splitlines(), start=1):
        if heredoc_term is not None:
            if raw.strip() == heredoc_term:
                heredoc_term = None
            continue
        # A FULL-LINE COMMENT IS NOT CODE.  The repo keeps the superseded
        # rendering of a repaired block in comment blocks (AGENTS.md §5.5
        # rule #6), and those blocks quote the OLD code verbatim — including
        # the old `grep -q '"fail": 0'` idiom.  Scanning them would report
        # the repaired wrapper as broken.
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        line = raw.rstrip()
        # A heredoc opener ends the CODE line, but its BODY is not code.
        hd = _heredoc_opener(line)
        if hd:
            heredoc_term = hd.group(1) or hd.group(2) or hd.group(3)
        if line.endswith("\\"):
            if not buf:
                start = i
            buf.append(line[:-1].strip())
            continue
        if not buf:
            start = i
        buf.append(line.strip())
        joined = " ".join(x for x in buf if x)
        hash_at = joined.find(" #")
        if hash_at != -1:
            joined = joined[:hash_at]
        if joined.strip():
            out.append((start, joined))
        buf = []
    if buf:
        joined = " ".join(x for x in buf if x)
        if joined.strip():
            out.append((start, joined))
    return out


def _unquote(token: str) -> str:
    tok = token.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    return tok


def collect_assignments(text: str) -> Dict[str, str]:
    """Map every shell variable the wrapper assigns to its unexpanded value."""
    env: Dict[str, str] = {}
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _ASSIGN.match(stripped)
        if not m:
            continue
        value = m.group(2).strip()
        hash_at = value.find(" #")
        if hash_at != -1:
            value = value[:hash_at].strip()
        env[m.group(1)] = _unquote(value)
    return env


def _expand(value: str, env: Dict[str, str], depth: int = 0, seen: Tuple[str, ...] = ()) -> str:
    """Expand ``$VAR`` / ``${VAR}`` against ``env``; unknown refs stay in place."""
    if depth > 10:
        return value

    def repl(m: re.Match) -> str:
        name = m.group(1) or m.group(4)
        if name is None:
            return m.group(0)
        default = m.group(3) if m.group(2) else None
        if name in seen:
            return default if default is not None else m.group(0)
        if name in env:
            return _expand(env[name], env, depth + 1, seen + (name,))
        if default is not None:
            return default
        return m.group(0)

    return _VARREF.sub(repl, value)


def resolved_report_path(text: str, expression: str) -> str:
    """Expand a report-path EXPRESSION against the wrapper's own assignments."""
    env = collect_assignments(text)
    return _expand(_unquote(expression), env)


def _tool_vars(text: str, tool: str) -> List[str]:
    """Variables whose assigned value NAMES ``tool``.

    The repo holds tool paths in variables (``PRE_FLIGHT=...``,
    ``GATE_VERDICT_CHECKER=...``), so a literal scan misses the invocation and
    finds only the assignment.  Resolving the variable is what lets the
    detector tell an ASSIGNMENT from a CALL.
    """
    names: List[str] = []
    for name, value in collect_assignments(text).items():
        if tool in value:
            names.append(name)
    return names


def _mentions(line: str, tool: str, vars_: List[str]) -> bool:
    """True when a line names ``tool`` literally or through one of ``vars_``."""
    if tool in line:
        return True
    for name in vars_:
        if f"${{{name}}}" in line or f"${name}" in line:
            return True
    return False


def _quoted_spans(line: str) -> List[Tuple[int, int]]:
    """The ``(start, end)`` character spans of every quoted region in a line."""
    spans: List[Tuple[int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch in "'\"":
            j = line.find(ch, i + 1)
            if j == -1:
                spans.append((i, n))
                break
            spans.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return spans


def _heredoc_opener(line: str) -> Optional[re.Match]:
    """The heredoc opener on this line, if there is one.

    The search runs on the RAW line, not on the unquoted one, because the
    delimiter is usually quoted (``cat <<'EOF'``) and stripping the quotes
    would delete the very token the regex needs.  What is excluded instead is
    an opener that lies INSIDE a quoted span (``log "a << EOF b"``) — that is
    a character in a message, not a redirection.
    """
    spans = _quoted_spans(line)
    for m in _HEREDOC_OPEN.finditer(line):
        if any(start <= m.start() < end for start, end in spans):
            continue
        return m
    return None


def _unquoted(line: str) -> str:
    """The line with every single- and double-quoted span removed."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", " ", line)


def _is_invocation(line: str) -> bool:
    """True when a line that names a tool is a CALL rather than an assignment.

    Two things make a line a NON-invocation even though it names the tool:

    * it is an ASSIGNMENT (``PRE_FLIGHT=".../pre_flight_check.py"``) — it
      passes neither ``--pipeline`` nor ``--json``;
    * the tool name appears only inside a QUOTED STRING, i.e. it is a MESSAGE
      (``log "pre_flight_check.py --pipeline launch (TODO-207 v3 gate)"``).
      The repo's wrappers log the gate they are about to run, and a detector
      that read the log line as the call would pick the wrong line for every
      downstream check.

    So the flags are looked for OUTSIDE the quoted spans.
    """
    bare = _unquoted(line)
    return "--pipeline" in bare or "--json" in bare


def preflight_calls(text: str) -> List[Tuple[int, str]]:
    """Every logical line that INVOKES the pre-flight tool."""
    vars_ = _tool_vars(text, PREFLIGHT_TOOL)
    out: List[Tuple[int, str]] = []
    for ln, line in logical_lines(text):
        if _mentions(line, PREFLIGHT_TOOL, vars_) and _is_invocation(line):
            out.append((ln, line))
    return out


def report_path_expression(text: str) -> Optional[str]:
    """The ``--json`` argument of the FIRST pre-flight invocation, as written."""
    for _ln, line in preflight_calls(text):
        m = _JSON_ARG.search(line)
        if m:
            return _unquote(m.group(1))
    return None


def report_paths(text: str) -> List[Tuple[int, Optional[str]]]:
    """``(lineno, --json path)`` for EVERY pre-flight invocation.

    A wrapper may run the pre-flight more than once (``launch_wf306a`` does:
    once for the ``all`` pipeline and once for the ``launch`` pipeline), and
    each invocation needs its own report path, its own ``rm -f``, and its own
    verdict.  A detector that looked only at the FIRST invocation would let
    the second one go unchecked — which is exactly the "the guard checks one
    of two call sites" shape that lets a dead gate hide in the tail.
    """
    out: List[Tuple[int, Optional[str]]] = []
    for ln, line in preflight_calls(text):
        m = _JSON_ARG.search(line)
        out.append((ln, _unquote(m.group(1)) if m else None))
    return out


def consumer_calls(text: str) -> List[Tuple[int, str]]:
    """Every logical line that names the FAIL-CLOSED consumer."""
    vars_ = _tool_vars(text, CONSUMER_TOOL)
    return [
        (ln, line)
        for ln, line in logical_lines(text)
        if _mentions(line, CONSUMER_TOOL, vars_)
    ]


#: A shell function definition: ``name() {`` or ``function name {``.
_FUN_DEF = re.compile(r"^(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\))?\s*\{\s*$")


def helper_functions_calling_consumer(text: str) -> List[str]:
    """Names of the wrapper's own functions that reach the consumer.

    A wrapper may reach the consumer DIRECTLY (the call site names
    ``wf_launch_gate_verdict.py preflight ...``) or through a local helper
    (``run_gate_check() { ... wf_launch_gate_verdict.py "$@" ... }`` called as
    ``run_gate_check preflight <report>``).  Both are legitimate; what is NOT
    legitimate is a wrapper that reaches the consumer for some OTHER report
    kind only, or not at all.

    The resolution is TRANSITIVE.  Two of the repo's wrappers add a second
    layer (``gate_verdict() { run_gate_check "$1" "$2" ...; }``, called as
    ``gate_verdict preflight <report>``), so a detector that stopped at the
    first layer would read those wrappers as "the consumer is never reached
    for a preflight report" — a FALSE POSITIVE that would send the next
    repair wave to edit a wrapper that is already correct.
    """
    lines = logical_lines(text)
    cvars = _tool_vars(text, CONSUMER_TOOL)

    bodies: Dict[str, List[str]] = {}
    order: List[str] = []
    idx = 0
    while idx < len(lines):
        m = _FUN_DEF.match(lines[idx][1].strip())
        if not m:
            idx += 1
            continue
        name = m.group(1)
        body: List[str] = []
        cursor = idx + 1
        while cursor < len(lines) and lines[cursor][1].strip() != "}":
            body.append(lines[cursor][1])
            cursor += 1
        bodies[name] = body
        order.append(name)
        idx = cursor + 1

    # Fixed point: a function reaches the consumer if its body names the
    # consumer, OR if its body calls another function that reaches it.
    reaching = {
        n for n in order
        if any(_mentions(b, CONSUMER_TOOL, cvars) for b in bodies[n])
    }
    changed = True
    while changed:
        changed = False
        for n in order:
            if n in reaching:
                continue
            for b in bodies[n]:
                if any(
                    re.search(r"\b" + re.escape(r) + r"\b", b) for r in reaching
                ):
                    reaching.add(n)
                    changed = True
                    break
    return sorted(reaching)


def _calls_helper_with_kind(text: str, helper: str, kind: str) -> bool:
    """True when some live line calls ``helper`` with the ``kind`` argument."""
    pattern = re.compile(
        r"\b" + re.escape(helper) + r"\b\s+(?:--\S+\s+)*" + re.escape(kind) + r"\b"
    )
    return any(pattern.search(line) for _ln, line in logical_lines(text))


def contract_findings(text: str) -> List[str]:
    """Every contract violation in a wrapper.  Empty list == the contract holds.

    This is the function the guard's MUTATION test drives: break a wrapper,
    and the matching finding must appear.
    """
    findings: List[str] = []
    calls = preflight_calls(text)
    if not calls:
        findings.append("C1: the wrapper never invokes pre_flight_check.py")
        return findings

    # Every invocation is checked on its own.  A wrapper that runs the
    # pre-flight twice must satisfy the contract at BOTH call sites.
    for call_line, expr in report_paths(text):
        if expr is None:
            findings.append(
                f"C2: the pre-flight invocation at line {call_line} passes no "
                f"--json PATH, so no report file is written for a consumer to "
                f"read"
            )
            continue

        # C3 — the report path must not be a merged-capture target.
        redirect = re.compile(
            r">\s*[\"']?\s*" + re.escape(expr) + r"(?![A-Za-z0-9_])"
        )
        for ln, line in logical_lines(text):
            if expr in line and redirect.search(line) and re.search(
                r"2>&1|&>>", line
            ):
                findings.append(
                    f"C3: line {ln} redirects stderr into the report path "
                    f"({expr}); the tool logs to stderr and prints the "
                    f"payload to stdout, so a merged capture is not JSON"
                )
                break
        # C4 — the previous report must be removed BEFORE the invocation.
        removed = False
        for ln, line in logical_lines(text):
            if ln >= call_line:
                break
            if re.search(r"\brm\b\s+-[a-z]*f", line) and expr in line:
                removed = True
                break
        if not removed:
            findings.append(
                f"C4: the previous report ({expr}) is not removed before the "
                f"pre-flight invocation (line {call_line}); a crashed "
                f"pre-flight would leave the last run's PASS report on disk"
            )

    # C5 — the TIMEOUT-blind grep idiom, in live code only.
    for ln, line in logical_lines(text):
        if _TIMEOUT_BLIND_GREP.search(line):
            findings.append(
                f"C5: line {ln} greps the report for a literal '\"fail\": 0'; "
                f"that predicate is TIMEOUT-blind (a report with 1 TIMEOUT and "
                f"0 FAILs matches it)"
            )
            break

    # C6 — the verdict must come from the consumer, for the PRE-FLIGHT kind.
    if not consumer_calls(text):
        findings.append(
            "C6: the wrapper never invokes wf_launch_gate_verdict.py, so its "
            "pre-flight verdict is not read through the FAIL-CLOSED consumer"
        )
    else:
        direct = [
            line for _ln, line in consumer_calls(text) if CONSUMER_KIND in line
        ]
        helpers = helper_functions_calling_consumer(text)
        via_helper = [
            h for h in helpers if _calls_helper_with_kind(text, h, CONSUMER_KIND)
        ]
        if not direct and not via_helper:
            findings.append(
                f"C6: the consumer is reached, but never for the "
                f"'{CONSUMER_KIND}' report kind -- a verdict about the "
                f"pre-flight must be read from the pre-flight report"
            )

    return findings


def _logical_span(lines: List[str], start_lineno: int) -> Tuple[int, int]:
    """The (start, stop) PHYSICAL span of the logical line at ``start_lineno``."""
    i = max(0, start_lineno - 1)
    stop = i
    while stop < len(lines) and lines[stop].rstrip().endswith("\\"):
        stop += 1
    return i, stop + 1


def broken_variants(text: str) -> List[Tuple[str, str]]:
    """Deliberately broken copies of a wrapper, one per contract rule.

    Each variant breaks exactly ONE rule, so a guard that asserts "the
    findings are empty for the real file" can also assert "each finding is
    REACHABLE" — the mutation proof that the detector is not a tautology.

    The mutations are applied to the LIVE invocation block, never to a
    whole-file string replacement.  A whole-file replacement finds the FIRST
    occurrence of a token, and in this repo the first occurrence is very
    often inside the preserved-rendering COMMENT block (AGENTS.md §5.5
    rule #6) — so the mutation would land in a comment, change nothing, and
    the rule would look untested for a reason that has nothing to do with the
    rule.

    A variant that cannot be built is NOT reported: the caller asserts that
    every rule it claims to cover has a reachable mutation, so a silent
    "could not build the broken copy" would turn the mutation proof into
    another dead gate.
    """
    variants: List[Tuple[str, str]] = []
    expr = report_path_expression(text)
    raw_lines = text.splitlines()
    calls = preflight_calls(text)

    # C2 — delete the --json PATH from the live invocation, so no report
    # file is written for a consumer to read.
    if calls:
        ln = calls[0][0]
        start, stop = _logical_span(raw_lines, ln)
        block = "\n".join(raw_lines[start:stop])
        m = _JSON_ARG.search(block)
        if m:
            new_block = block[: m.start()] + block[m.end():]
            v = "\n".join(raw_lines[:start] + [new_block] + raw_lines[stop:])
            variants.append(("C2", v))

    # C3 — make the report path the target of a MERGED capture.
    if expr:
        v = text + f'\ncat "${{REPORT}}" > "{expr}" 2>&1\n'
        variants.append(("C3", v))

    # C4 — remove the `rm -f` of the previous report.
    if expr:
        kept = [
            ln for ln in raw_lines
            if not (re.search(r"\brm\b\s+-[a-z]*f", ln) and expr in ln)
        ]
        v = "\n".join(kept)
        if v != text:
            variants.append(("C4", v))

    # C5 — reintroduce the TIMEOUT-blind grep.
    v = text + '\ngrep -q \'"fail": 0\' "${REPORT}"\n'
    variants.append(("C5", v))

    # C6 — remove every mention of the consumer (literal or variable-held).
    cvars = _tool_vars(text, CONSUMER_TOOL)
    kept = [
        ln for ln in raw_lines
        if not _mentions(ln, CONSUMER_TOOL, cvars)
    ]
    v = "\n".join(kept)
    if v != text:
        variants.append(("C6", v))

    return variants
