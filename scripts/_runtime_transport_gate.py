#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""AGENTS.md §5.11 runtime-transport gate — the ONE launch surface no static gate sees.

THE BLIND SPOT (MEASURED 2026-11-04 on this tree)
--------------------------------------------------
``scripts/_yaml_only_gate.py`` and ``scripts/_launch_contract_gate.py`` are both
SOURCE scans: they read a wrapper's text and look for a trainer invocation
LINE.  ``scripts/start.sh`` has no such line.  It assembles the trainer command
as a RUNTIME STRING::

    scripts/start.sh:963   TRAINER_CMD="$ENTRY_COMMAND$APPENDED_FLAGS $EVENTS_FLAG ${DMPNN_NEW_FLAGS:-} $EXTRA_FLAGS"
    scripts/start.sh:1001  nohup bash -c "$TRAINER_CMD >> $LOG_FILE 2>&1" &

so both static gates see no trainer invocation in ``start.sh`` and PASS it —
correctly as far as they can see, and BLIND to the actual launch.  This gate
covers exactly that blind spot.

THE RULE
--------
AGENTS.md §5.11 (a STANDING USER DIRECTIVE) says the legal flag set is
PER-TRAINER, defined by the TRAINER'S OWN PARSER, and that a hyperparameter
must live in the YAML (a knob that lives only in a shell script is a knob
nobody audits).  Transported at runtime, that becomes two assertions:

  R1  every flag the runtime command can carry must be DECLARED by the parser
      of the trainer that runtime command would invoke;
  R2  every such flag's VALUE must be YAML-derived — the flag is emitted only
      because the YAML declared the corresponding key.

FAIL-CLOSED.  A transport site whose population mechanism this gate does not
understand, or a trainer whose parser cannot be captured, is a COVERAGE GAP
and makes the verdict FAIL.  A check that cannot check must not certify the
launch (AGENTS.md §6.269.17: a gate that cannot fail is a defect).

WHAT IS NOT CHANGED HERE
------------------------
The transports themselves are NOT closed or rewritten.  §5.11's allowlist
wording is USER-gated, and the eight ``_add_flag`` transports in
``scripts/start.sh`` are LEGAL whenever the invoked trainer declares the flag
(and their values already come from the YAML).  This gate proves that; it does
not remove them.

SCOPE NOTE — where R2 is and is not asserted
--------------------------------------------
R2 is asserted on the LAUNCHER-side transports: the ``_add_flag`` call sites,
the ``EVENTS_FLAG`` assignments and the ``_dmpnn_new_flags`` emitter block.
Those live in the shell script, so a literal value there is exactly the
"knob nobody audits" §5.11 forbids.

R2 is NOT asserted on a CONFIG's own ``entry.command_template``: a literal
inside a command template is the YAML declaring the command, so it is
config-resident by construction.  The SHELL-side legacy template in
``scripts/start.sh`` (the one start.sh synthesises for an entry-less config)
IS asserted, because that string lives in the launcher, not in the YAML.

Stdout invariant (AGENTS.md §4): PASS / FAIL / SKIP.  Exit 0=PASS, 1=FAIL,
2=SKIP (no runtime-string launch surface on this tree).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(path: Path, name: str):
    """Import a repo script by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: The shell-parsing home.  ``_yaml_only_gate`` owns the shell tokeniser, the
#: YAML->env name census and the yaml-local census; this gate reuses them
#: rather than re-implementing them.
GATE = _load_module(
    REPO_ROOT / "scripts" / "_yaml_only_gate.py", "_wfa269_yaml_only_gate"
)

#: The launch-contract gate.  ``_declared_flags`` is the CORRECT parser
#: capture (it handles the zero-argument ``main()`` case — see
#: ``_launch_contract_gate.py``'s SUPERSEDED RENDERING PRESERVED VERBATIM
#: block for the 5 trainers it used to miss).  It is imported, not copied.
LC = _load_module(
    REPO_ROOT / "scripts" / "_launch_contract_gate.py", "_wfa269_launch_contract_gate"
)


# ---------------------------------------------------------------------------
# Shell lexing helpers
# ---------------------------------------------------------------------------
#: `VAR="..."` / `local VAR="..."` / `export VAR="..."` — one shell assignment.
_SHELL_ASSIGN = re.compile(
    r'^[ \t]*(?:local[ \t]+|export[ \t]+|declare[ \t]+)?'
    r'([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',
    re.M,
)

#: `f'NAME={...}'` — the YAML->env SERIALISATION inside start.sh's python
#: block.  Every name in that block is YAML-derived by construction.
_ENV_ASSIGN = re.compile(r"f?'([A-Z][A-Z0-9_]*)=\{([^{}]*)\}'")

#: `$VAR` / `${VAR}` / `${VAR:-}` — a shell variable reference.
_VARREF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")

#: `name() {` — a shell function definition.
_FUNC_DEF = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{", re.M)

#: A runtime command execution: `bash -c "$VAR"` / `sh -c "$VAR"`.
_RUNTIME_EXEC = re.compile(r'(?:\bba|\bz|\bk)?sh\s+-c\s+"\$\{?([A-Za-z_][A-Za-z0-9_]*)')

#: A runtime command execution via `eval "$VAR"`.
_EVAL_EXEC = re.compile(r'\beval\s+"\$\{?([A-Za-z_][A-Za-z0-9_]*)')

#: A flag and the value token that follows it (`--flag value`).
_FLAG_OCCURRENCE = re.compile(r"(--[a-z0-9][a-z0-9-]*)(?:\s+(?!--)([^\s]+))?")

#: A `{placeholder}` — filled from the YAML by start.sh's own format() call.
_PLACEHOLDER = re.compile(r"^\{[A-Za-z_][A-Za-z0-9_]*\}$")

#: `Path(w).name == 'train_x.py'` — the trainer-basename guard of a python
#: emitter block.
_BASENAME_GUARD = re.compile(r"Path\(w\)\.name\s*==\s*'([^']+)'")

#: An emptiness test on an indirect expansion: `-z "${!_var}"`.
_EMPTY_TEST = re.compile(r'-z\s+"\$\{!([A-Za-z_][A-Za-z0-9_]*)\}"')

# ---------------------------------------------------------------------------
# DEFECT C (2026-11-04, Wf-A269 dead-gates sweep) — the assembly's OWN
# literal text, and the `for ... in` loop that can feed a runtime command.
# ---------------------------------------------------------------------------

#: A shell variable expansion — `$VAR` / `${VAR}` / `${VAR:-default}`.
_VAR_EXPANSION = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")

#: `for VAR in <list>` — the loop that feeds a runtime command in
#: ``scripts/_wf_r28_gpu_master.sh`` (``for spec in "1|${T1_CMD}|..."``).
_FOR_LOOP = re.compile(
    r"^[ \t]*for[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]+in[ \t]+(.*)$"
)

#: ANY shell assignment on a line.  The gate's main ``_SHELL_ASSIGN`` only
#: matches the FIRST assignment of a physical line, which is enough for the
#: assembly lookups but NOT for the delegation chain:
#: ``scripts/_wf_r28_gpu_master.sh`` writes ``n="${spec%%|*}"; rest="${spec#*|}"``
#: — two assignments on one line.  The ``(?:^|[;\s])`` guard keeps a
#: ``name="value"`` INSIDE a quoted string from being mistaken for an
#: assignment.
_ASSIGN_ANYWHERE = re.compile(
    r"(?:^|[;\s])([A-Za-z_][A-Za-z0-9_]*)=(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')"
)

#: A command substitution — `$(...)` or a backtick.  The VALUE of such an
#: expression cannot be enumerated statically.
_COMMAND_SUBSTITUTION = re.compile(r"\$\(|`")


def _shell_assignments(text: str) -> list[tuple[int, str, str]]:
    """(lineno, name, inner value) for every shell string assignment."""
    out: list[tuple[int, str, str]] = []
    for m in _SHELL_ASSIGN.finditer(text):
        out.append((text.count("\n", 0, m.start()) + 1, m.group(1), m.group(2)[1:-1]))
    return out


def _function_span(text: str, name: str) -> tuple[int, int, str] | None:
    """(start offset, end offset, body) of the shell function `name`."""
    for m in _FUNC_DEF.finditer(text):
        if m.group(1) != name:
            continue
        depth = 0
        started = False
        for i in range(m.end() - 1, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return (m.start(), i, text[m.end() : i])
    return None


def _enclosing_function(text: str, offset: int) -> tuple[str, int, str] | None:
    """The shell function whose body contains the byte `offset`."""
    best = None
    for m in _FUNC_DEF.finditer(text):
        if m.start() > offset:
            break
        name = m.group(1)
        span = _function_span(text, name)
        if span is None:
            continue
        if span[0] <= offset <= span[1]:
            best = (name, text.count("\n", 0, m.start()) + 1, span[2])
    return best


def _flag_occurrences(text: str) -> list[tuple[str, str | None]]:
    """(flag, value token) for every ``--flag`` in `text`."""
    return [(m.group(1), m.group(2)) for m in _FLAG_OCCURRENCE.finditer(text)]


def _literal_text(value: str) -> str:
    """The LITERAL part of a shell value — every expansion removed.

    DEFECT C (2026-11-04, Wf-A269 dead-gates sweep): the transport scan
    enumerated only the VARIABLES a runtime command references (the
    "contributors").  The literal text BETWEEN them was never scanned, so a
    flag written as a literal INSIDE the runtime command string was
    invisible to this gate — and to the other three §5.11 gates, all of
    which read STATIC invocation lines.  MEASURED on a synthetic fixture
    (a runtime string carrying ``--epochs 99``): the gate reported
    ``n_sites 0 / n_violations 0 / n_coverage_gaps 0`` and a PASS verdict.

    This helper returns the literal remainder so it can be scanned too.
    """
    return _VAR_EXPANSION.sub(" ", value)


def _for_loop_refs(text: str) -> dict[str, list[str]]:
    """The variable references each ``for VAR in ...`` loop can feed.

    ``scripts/_wf_r28_gpu_master.sh`` assembles its runtime command from a
    loop over ``"1|${T1_CMD}|${T1_WALL}|${T1_WHY}" ...``, so the chain that
    produces the command runs THROUGH a for-list — which is not a shell
    assignment and was therefore invisible to the delegation escape hatch.
    """
    out: dict[str, list[str]] = {}
    for _lineno, code in GATE._logical_lines(text):
        m = _FOR_LOOP.match(code)
        if not m:
            continue
        loop_var = m.group(1)
        refs: list[str] = []
        for ref in _VARREF.findall(m.group(2)):
            if ref != loop_var and ref not in refs:
                refs.append(ref)
        out.setdefault(loop_var, []).extend(refs)
    return out


def _delegation_reachable(text: str, runtime_var: str) -> list[str]:
    """The ``.sh`` wrappers the runtime command REACHES.

    DEFECT C (2026-11-04) — the delegation escape hatch used to scan EVERY
    shell assignment in the wrapper for a ``.sh`` token and call the wrapper
    "covered" on that basis.  A ``.sh`` token somewhere in the file is NOT a
    proof that the RUNTIME COMMAND delegates to it: the token may belong to
    a command the runtime command never runs.  This walks the chain the
    runtime command actually depends on — shell assignments AND ``for ... in``
    loop lists — and returns only the ``.sh`` targets that chain reaches.
    """
    assigns: dict[str, list[str]] = {}
    for m in _ASSIGN_ANYWHERE.finditer(text):
        name = m.group(1)
        value = m.group(2)[1:-1]
        assigns.setdefault(name, []).append(value)
    loops = _for_loop_refs(text)
    seen: set[str] = set()
    found: list[str] = []
    stack = [runtime_var]
    while stack:
        var = stack.pop()
        if var in seen:
            continue
        seen.add(var)
        for value in assigns.get(var, []):
            for tok in value.split():
                if tok.endswith(".sh") and tok not in found:
                    found.append(tok)
            for ref in _VARREF.findall(value):
                if ref != var:
                    stack.append(ref)
        # A `for VAR in ...` list feeds the loop variable with the NAMES it
        # lists (the names are bare, not `$`-prefixed, so they are pushed
        # directly rather than re-scanned).
        for ref in loops.get(var, []):
            if ref != var:
                stack.append(ref)
    return found


def _argparse_accepts(declared: set[str], flag: str) -> bool:
    """Would `argparse` accept `flag` given the trainer's declared options?

    MEASURED 2026-11-04 on this tree: an EXACT-match-only rule is WRONG.
    ``argparse`` enables prefix abbreviation by default (``allow_abbrev``
    defaults to True; MEASURED: 0 of the repo's ``train_*.py`` /
    ``bench_*.py`` parsers pass ``allow_abbrev=False``), so ``--hidden`` IS
    accepted by ``train_cfm_yuelbond_wasserstein.py`` — whose parser
    declares only ``--hidden-dim`` — and the launch does NOT die.

    argparse's own rule (CPython ``_get_option_tuples``): an option string
    matches when it is a PREFIX of a declared option; if exactly one
    declared option matches, the abbreviation is accepted; if more than one
    matches, argparse raises "ambiguous option"; if none matches, it
    raises "unrecognized arguments".  This function reproduces that rule.

    The gate's R1 check uses THIS, not exact membership: a flag that
    argparse would accept is NOT a launch-aborting defect, and reporting it
    as one is a FALSE POSITIVE — which is worse than a miss, because it
    trains the reader to ignore the gate (AGENTS.md §6.269.17).
    """
    if flag in declared:
        return True
    matches = [d for d in declared if d.startswith(flag)]
    return len(matches) == 1


def _argparse_ambiguous(declared: set[str], flag: str) -> list[str]:
    """The declared options a `--flag` prefix would be AMBIGUOUS with."""
    if flag in declared:
        return []
    return sorted(d for d in declared if d.startswith(flag))


def _env_local_name(expr: str) -> str:
    """The python local a YAML->env expression reads its value from.

    ``shlex.quote(cmd)`` -> ``cmd``;  ``shlex.quote(e.get("script", ""))`` ->
    ``e``.  String literals are stripped first, so a key name inside a string
    is never mistaken for the local.
    """
    stripped = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', ' ', expr)
    names: list[str] = []
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", stripped):
        after = stripped[m.end() : m.end() + 1]
        before = stripped[m.start() - 1 : m.start()] if m.start() else ""
        if after == "(" or before == ".":
            continue
        names.append(m.group(0))
    return names[-1] if names else ""


def _py_string_chunks(text: str) -> list[str]:
    """The literal contents of every python string literal in `text`.

    ``_yaml_only_gate._PY_STRING`` has two groups (single-quoted, then
    double-quoted), so ``findall`` yields tuples; this normalises them back to
    the literal contents.
    """
    out: list[str] = []
    for m in GATE._PY_STRING.finditer(text):
        single, double = m.group(1), m.group(2)
        out.append(single if single is not None else (double or ""))
    return out


def _func_params(body: str) -> list[str]:
    """The `local a="$1" b="$2"` parameter names of a shell function."""
    m = re.search(r"^[ \t]*local\s+(.*)$", body, re.M)
    if not m:
        return []
    return re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"\$\d"', m.group(1))


def _call_sites(text: str, func_name: str) -> list[tuple[int, list[str]]]:
    """(lineno, args) for every call of the shell function `func_name`."""
    out: list[tuple[int, list[str]]] = []
    pattern = re.compile(r"^[ \t]*" + re.escape(func_name) + r"[ \t]+(.*)$", re.M)
    for m in pattern.finditer(text):
        out.append((text.count("\n", 0, m.start()) + 1, m.group(1).split()))
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Population 1 — the runtime-string launch surfaces
# ---------------------------------------------------------------------------
def runtime_wrappers() -> list[dict]:
    """Every wrapper that executes a command assembled at RUNTIME.

    Discovered, not hardcoded: every non-recursive ``*.sh`` under the repo's
    two wrapper roots is read and searched for a runtime command execution.

    MEASURED 2026-11-04: 2 wrappers — ``scripts/start.sh`` and
    ``scripts/_wf_r28_gpu_master.sh``.  MEASURED over all tracked ``*.sh``:
    no other wrapper runs ``bash -c "$VAR"`` / ``eval "$VAR"``, and no tracked
    ``*.py`` under ``scripts/`` or ``molmetal/scripts/`` uses ``shell=True``
    (1 hit repo-wide, and it is a test file).
    """
    found: list[dict] = []
    for root in GATE.WRAPPER_ROOTS:
        for path in sorted((REPO_ROOT / root).glob("*.sh")):
            if GATE._is_excluded_wrapper(path):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            sites: list[dict] = []
            for m in _RUNTIME_EXEC.finditer(text):
                sites.append(
                    {
                        "line": text.count("\n", 0, m.start()) + 1,
                        "var": m.group(1),
                        "form": "sh -c",
                    }
                )
            for m in _EVAL_EXEC.finditer(text):
                sites.append(
                    {
                        "line": text.count("\n", 0, m.start()) + 1,
                        "var": m.group(1),
                        "form": "eval",
                    }
                )
            if sites:
                found.append({"path": path, "text": text, "exec_sites": sites})
    return found


def _assembly(text: str, runtime_var: str) -> tuple[int, str] | None:
    """(lineno, value) of the assignment that builds the runtime command."""
    for lineno, name, value in _shell_assignments(text):
        if name == runtime_var:
            return lineno, value
    return None


# ---------------------------------------------------------------------------
# Population 2 — the transport SITES
# ---------------------------------------------------------------------------
def _enumerate_contributor(
    text: str, contributor: str, seen: set[str]
) -> tuple[list[dict], list[str], bool, bool]:
    """Every place `contributor` can receive a flag.

    Returns (emissions, gaps, understood, per_config).

    * ``understood``  False when the contributor's population mechanism could
      not be traced — the caller turns that into a COVERAGE GAP (fail-closed).
    * ``per_config``  True when the contributor's flags are the YAML's own
      command template, which the per-config pass enumerates.
    """
    emissions: list[dict] = []
    gaps: list[str] = []
    if contributor in seen:
        return emissions, gaps, True, False
    seen = seen | {contributor}

    # (a) the python YAML->env serialisation: `f'NAME={...}'`.
    for m in _ENV_ASSIGN.finditer(text):
        if m.group(1) != contributor:
            continue
        expr = m.group(2)
        local = _env_local_name(expr)
        # EVERY assignment of the local is a candidate: a python emitter is
        # often INITIALISED to '' and then REBUILT inside a conditional
        # (`_dmpnn_new_flags = ''` then `_dmpnn_new_flags = ( ... )`), and the
        # first assignment alone would hide the flags.
        cands = list(
            re.finditer(r"^[ \t]*" + re.escape(local) + r"[ \t]*=[ \t]*", text, re.M)
        )
        if not cands:
            gaps.append(
                f"{contributor}: the env serialisation names the python local "
                f"{local!r}, which has no assignment in the wrapper"
            )
            return emissions, gaps, False, False
        saw_template = False
        for pm in cands:
            rhs = text[pm.end() : pm.end() + 400]
            head = "\n".join(rhs.split("\n")[:2])
            if ".format(" in head or re.search(r"command_template", head):
                saw_template = True
                break
            # A python emitter tuple: `_dmpnn_new_flags = ( ... )`.
            if "(" in text[pm.end() : pm.end() + 80]:
                span = GATE._balanced_span(
                    text, text.index("(", pm.end() - 1), "(", ")"
                )
            else:
                span = rhs.split("\n")[0]
            guard = _basename_guard(text, pm.start())
            for chunk in _py_string_chunks(span):
                for flag, value in _flag_occurrences(chunk):
                    emissions.append(
                        {
                            "kind": "python_literal",
                            "line": text.count("\n", 0, pm.start()) + 1,
                            "flag": flag,
                            "value_expr": value or "",
                            "value_var": "",
                            "guarded": guard,
                            "conditional": True,
                            "chunk": chunk,
                        }
                    )
        if saw_template:
            # The value is the YAML's own command template -> per-config flags.
            return emissions, gaps, True, True
        return emissions, gaps, True, False

    # (b) shell assignments to the contributor.
    assigns = [(ln, val) for ln, name, val in _shell_assignments(text) if name == contributor]
    if not assigns:
        return emissions, gaps, False, False
    understood = True
    for lineno, value in assigns:
        refs = _VARREF.findall(value)
        flags = _flag_occurrences(value)
        if flags:
            for flag, raw in flags:
                emissions.append(
                    {
                        "kind": "shell_literal",
                        "line": lineno,
                        "flag": flag,
                        "value_expr": raw or "",
                        "value_var": "",
                        "guarded": None,
                        "conditional": False,
                        "raw": value,
                    }
                )
            continue
        if contributor in refs:
            # Self-append inside a function: the flags come from the
            # function's PARAMETER, so the emission sites are its CALL sites.
            offset = text.index(value, 0)
            body_info = _enclosing_function(text, offset)
            if body_info is None:
                gaps.append(
                    f"{contributor} (line {lineno}): the assignment appends to "
                    "itself but no enclosing shell function could be located"
                )
                understood = False
                continue
            func_name, func_line, func_body = body_info
            params = _func_params(func_body)
            value_param = params[1] if len(params) > 1 else ""
            conditional = bool(_EMPTY_TEST.search(func_body)) and bool(value_param)
            for call_line, call_args in _call_sites(text, func_name):
                if len(call_args) < 2 or not call_args[0].startswith("--"):
                    gaps.append(
                        f"{func_name} (line {call_line}): a call site does not "
                        "supply a literal flag and a value variable"
                    )
                    understood = False
                    continue
                emissions.append(
                    {
                        "kind": "add_flag",
                        "line": call_line,
                        "flag": call_args[0],
                        "value_expr": "$" + call_args[1],
                        "value_var": call_args[1],
                        "guarded": None,
                        "conditional": conditional,
                        "func": func_name,
                        "func_line": func_line,
                        "value_param": value_param,
                    }
                )
            continue
        if refs:
            sub_e, sub_g, sub_u, sub_p = _enumerate_contributor(text, refs[0], seen)
            emissions.extend(sub_e)
            gaps.extend(sub_g)
            understood = understood and sub_u
            continue
        # An assignment with no flag and no reference: understood, empty.
    return emissions, gaps, understood, False


def _basename_guard(text: str, before: int) -> set[str] | None:
    """The trainer-basename guard in force at `before`, if any.

    ``None`` means "no guard found" -> the site can fire for ANY trainer.
    """
    hits = _BASENAME_GUARD.findall(text[:before])
    if not hits:
        return None
    return {hits[-1]}


# ---------------------------------------------------------------------------
# Population 3 — the configs and the trainers they invoke
# ---------------------------------------------------------------------------
def _config_paths(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.yaml") if not any(part.startswith(".") for part in p.parts)
    )


def _legacy_entries(start_text: str) -> list[tuple[str, str]]:
    """The (script, command) pairs of start.sh's synthesised legacy entries.

    Extracted from ``start.sh`` itself, so the gate does not carry a second
    copy of the legacy branch that could drift away from the launcher.
    """
    return [(script, command) for _lineno, script, command in GATE._legacy_entry_commands(start_text)]


def resolve_entry(cfg: dict, legacy: list[tuple[str, str]]) -> tuple[str, str, str]:
    """(trainer token, command template, source) for a config.

    Mirrors ``scripts/start.sh``'s ``parse_yaml`` block: a config with no
    ``entry`` gets a synthesised one, chosen by ``data.dataset``.  ``source``
    is ``"yaml"`` (the config's own template) or ``"shell"`` (start.sh's
    synthesised legacy template, which lives in the launcher).
    """
    entry = cfg.get("entry") or {}
    if not isinstance(entry, dict):
        entry = {}
    if entry:
        cmd = str(entry.get("command_template") or entry.get("command") or "")
        return str(entry.get("script") or ""), cmd, "yaml"
    data = cfg.get("data")
    data = data if isinstance(data, dict) else {}
    idx = 0 if data.get("dataset", "qm9") == "qm9" else 1
    if idx < len(legacy):
        return legacy[idx][0], legacy[idx][1], "shell"
    return "", "", "shell"


def resolve_trainer(
    script_token: str, command: str, extra_bases: tuple[Path, ...] = ()
) -> Path | None:
    """The trainer file start.sh would invoke for this config.

    Mirrors start.sh's probe: the first candidate that IS a file, relative to
    the CWD first and then to the repo root.  ``extra_bases`` are tried last
    (a config's own directory), which is what lets a synthetic fixture
    resolve a synthetic trainer.
    """
    candidates = [script_token]
    candidates += [w for w in command.split() if w.endswith(".py")]
    for word in candidates:
        if not word:
            continue
        candidate = Path(word)
        if candidate.is_file():
            return candidate
        rooted = REPO_ROOT / word
        if rooted.is_file():
            return rooted
        for base in extra_bases:
            other = base / word
            if other.is_file():
                return other
    return None


def _delegated_targets(text: str) -> list[str]:
    """The wrapper/trainer names a runtime command delegates to.

    SUPERSEDED 2026-11-04 (Wf-A269 dead-gates sweep) by
    :func:`_delegation_reachable`, which follows the chain the runtime
    command actually depends on.  This function scans EVERY shell assignment
    in the file, so a ``.sh`` token that belongs to an unrelated command
    makes a wrapper look "covered".  Kept for reference; the escape hatch no
    longer uses it.
    """
    names: list[str] = []
    for _lineno, _name, value in _shell_assignments(text):
        for tok in value.split():
            if tok.endswith(".sh") and tok not in names:
                names.append(tok)
    return names


def _trainer_named_by(
    literal: str, trainer_by_resolved: dict[str, str]
) -> str | None:
    """The ``trainers`` key of the trainer a runtime assembly LITERALLY names.

    ``None`` when the assembly names no trainer, or names more than one (in
    which case the gate cannot attribute a flag to a single parser).
    """
    hits: list[str] = []
    for tok in literal.split():
        if not tok.endswith(".py"):
            continue
        cand = Path(tok)
        if not cand.is_absolute():
            cand = REPO_ROOT / tok
        try:
            resolved = str(cand.resolve())
        except OSError:  # pragma: no cover — an unresolvable path
            continue
        key = trainer_by_resolved.get(resolved)
        if key is not None and key not in hits:
            hits.append(key)
    return hits[0] if len(hits) == 1 else None


def _has_static_trainer_call(path: Path) -> bool:
    """Does the wrapper carry a trainer invocation LINE (a static one)?"""
    text = path.read_text(encoding="utf-8", errors="replace")
    env = GATE._collect_assignments(text)
    for _lineno, line in GATE._logical_lines(text):
        if GATE._trainer_call(line, env) is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------
def audit(
    config_root: Path | None = None, wrappers: list[Path] | None = None
) -> dict:
    """Audit every runtime-string launch surface against the trainers' parsers.

    ``wrappers`` overrides the DISCOVERED population (used by the regression
    tests to audit a synthetic launcher without touching the repo's own).
    """
    config_root = config_root or (REPO_ROOT / "configs")
    violations: list[dict] = []
    gaps: list[dict] = []

    if wrappers is None:
        wrappers = runtime_wrappers()
    else:
        wrappers = [
            {
                "path": Path(p),
                "text": Path(p).read_text(encoding="utf-8", errors="replace"),
                "exec_sites": [],
            }
            for p in wrappers
        ]
        for w in wrappers:
            text = w["text"]
            for m in _RUNTIME_EXEC.finditer(text):
                w["exec_sites"].append(
                    {
                        "line": text.count("\n", 0, m.start()) + 1,
                        "var": m.group(1),
                        "form": "sh -c",
                    }
                )
            for m in _EVAL_EXEC.finditer(text):
                w["exec_sites"].append(
                    {
                        "line": text.count("\n", 0, m.start()) + 1,
                        "var": m.group(1),
                        "form": "eval",
                    }
                )
    if not any(w["exec_sites"] for w in wrappers):
        return {
            "gate": "runtime_transport",
            "verdict": "SKIP",
            "reason": (
                "no runtime-string launch surface found (no wrapper executes a "
                "command assembled at runtime)"
            ),
            "n_wrappers": len(wrappers),
            "wrappers": [_rel(Path(w["path"])) for w in wrappers],
            "n_sites": 0,
            "n_violations": 0,
            "n_coverage_gaps": 0,
            "violations": [],
            "coverage_gaps": [],
        }

    yaml_env_names: dict[str, set[str]] = {}
    yaml_local_names: dict[str, set[str]] = {}
    #: wrapper-relative-path -> wrapper text.  The R1/R2 passes below look up
    #: the guard of a transport SITE, and a site belongs to ONE wrapper; the
    #: text must be that wrapper's.  (Reading a leaked loop variable instead
    #: would silently attribute another wrapper's `if` to the site.)
    texts_by_rel: dict[str, str] = {}
    for w in wrappers:
        rel = _rel(w["path"])
        texts_by_rel[rel] = w["text"]
        yaml_env_names[rel] = GATE._yaml_env_names(w["text"])
        yaml_local_names[rel] = GATE._yaml_local_names(w["text"])

    start_path = REPO_ROOT / "scripts" / "start.sh"
    start_text = (
        start_path.read_text(encoding="utf-8", errors="replace")
        if start_path.is_file()
        else ""
    )
    legacy = _legacy_entries(start_text)

    # ---- the config population -------------------------------------------
    cfg_paths = _config_paths(config_root)
    configs: list[dict] = []
    trainers: dict[str, Path] = {}
    unresolved_configs: list[str] = []
    for path in cfg_paths:
        try:
            import yaml as _yaml

            cfg = _yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001 — an unreadable config is a gap
            unresolved_configs.append(f"{_rel(path)}: unreadable ({exc})")
            continue
        if not isinstance(cfg, dict):
            unresolved_configs.append(f"{_rel(path)}: not a mapping")
            continue
        script, command, source = resolve_entry(cfg, legacy)
        trainer = resolve_trainer(script, command, extra_bases=(path.parent,))
        if trainer is None:
            unresolved_configs.append(
                f"{_rel(path)}: no trainer resolvable from entry.script={script!r}"
            )
            continue
        trainers[str(trainer)] = trainer
        configs.append(
            {
                "path": _rel(path),
                "trainer": trainer,
                "command": command,
                "command_source": source,
            }
        )

    # ---- the trainer parsers (fail-closed on an uncapturable one) ---------
    declared_cache = {key: LC._declared_flags(t) for key, t in trainers.items()}

    #: resolved-path -> the ``trainers`` key.  The R1 pass needs the trainer
    #: a runtime assembly LITERALLY names; ``trainers`` is keyed by whatever
    #: ``resolve_trainer`` returned (sometimes a relative path), so the
    #: lookup is done on resolved paths.
    trainer_by_resolved: dict[str, str] = {}
    for key, trainer in trainers.items():
        try:
            trainer_by_resolved[str(Path(trainer).resolve())] = key
        except OSError:  # pragma: no cover — an unresolvable path
            continue

    #: Flags that argparse accepts only as an UNAMBIGUOUS PREFIX of a declared
    #: option.  These are NOT violations (the launch does not abort), but they
    #: are recorded rather than dropped: an abbreviation is a fragile contract
    #: — the day the trainer declares a second `--hidden*` option, the same
    #: command line becomes argparse's "ambiguous option" error.  Reporting
    #: them keeps the census honest without crying wolf.
    abbreviations: list[dict] = []

    # ---- the transport sites ---------------------------------------------
    sites: list[dict] = []
    per_config_contributors: set[str] = set()
    delegated: list[dict] = []
    for w in wrappers:
        rel = _rel(w["path"])
        text = w["text"]
        wrapper_emissions = 0
        local_gaps: list[dict] = []
        contributor_flags: set[str] = set()
        for exec_site in w["exec_sites"]:
            runtime_var = exec_site["var"]
            asm = _assembly(text, runtime_var)
            if asm is None:
                local_gaps.append(
                    {
                        "kind": "assembly_not_found",
                        "file": rel,
                        "line": exec_site["line"],
                        "detail": (
                            f"{rel}:{exec_site['line']} executes ${runtime_var} at "
                            "runtime, but no assignment of that variable could be "
                            "located; the gate cannot enumerate what the runtime "
                            "command carries, so it cannot certify it"
                        ),
                    }
                )
                continue
            asm_line, asm_value = asm
            contributors: list[str] = []
            for ref in _VARREF.findall(asm_value):
                if ref != runtime_var and ref not in contributors:
                    contributors.append(ref)
            for contributor in contributors:
                emissions, sub_gaps, understood, per_config = _enumerate_contributor(
                    text, contributor, set()
                )
                for note in sub_gaps:
                    local_gaps.append(
                        {
                            "kind": "contributor_not_understood",
                            "file": rel,
                            "line": asm_line,
                            "detail": note,
                        }
                    )
                if per_config:
                    per_config_contributors.add(contributor)
                elif not understood:
                    local_gaps.append(
                        {
                            "kind": "contributor_not_understood",
                            "file": rel,
                            "line": asm_line,
                            "detail": (
                                f"{rel}:{asm_line} builds ${runtime_var} from "
                                f"${contributor}, and the gate could not trace how "
                                f"${contributor} is populated; it cannot enumerate "
                                "the flags the runtime command carries"
                            ),
                        }
                    )
                for em in emissions:
                    em = dict(em)
                    em["wrapper"] = rel
                    em["contributor"] = contributor
                    em["assembly_line"] = asm_line
                    em["runtime_var"] = runtime_var
                    sites.append(em)
                    wrapper_emissions += 1
                    contributor_flags.add(em["flag"])

            # ---- DEFECT C repair (2026-11-04) — the assembly's OWN literal
            # text.  The contributor scan above enumerates the VARIABLES the
            # runtime command references; the literal text between them was
            # never scanned, so a flag written as a literal inside the
            # runtime command string was invisible to this gate AND to the
            # other three §5.11 gates (all of which read static invocation
            # lines only).
            literal = _literal_text(asm_value)
            if _COMMAND_SUBSTITUTION.search(asm_value):
                local_gaps.append(
                    {
                        "kind": "assembly_not_enumerable",
                        "file": rel,
                        "line": asm_line,
                        "detail": (
                            f"{rel}:{asm_line} builds ${runtime_var} with a "
                            "command substitution; the gate cannot enumerate "
                            "what the substitution produces, so it cannot "
                            "certify the flags the runtime command carries"
                        ),
                    }
                )
            for flag, raw in _flag_occurrences(literal):
                if flag in contributor_flags:
                    continue  # already seen through a contributor
                em = {
                    "kind": "assembly_literal",
                    "line": asm_line,
                    "flag": flag,
                    "value_expr": raw or "",
                    "value_var": "",
                    "guarded": None,
                    "conditional": False,
                    "wrapper": rel,
                    "contributor": f"<literal text of ${runtime_var}>",
                    "assembly_line": asm_line,
                    "runtime_var": runtime_var,
                    "trainer_key": _trainer_named_by(literal, trainer_by_resolved),
                }
                sites.append(em)
                wrapper_emissions += 1
                contributor_flags.add(flag)

        # A wrapper whose runtime command names OTHER wrappers (a dispatcher)
        # is not a trainer-command builder: its flags live on the STATIC
        # invocation lines of the wrappers it delegates to, which
        # scripts/_launch_contract_gate.py already audits.  The gate asserts
        # that every delegation target IS in that audited population AND has
        # a static trainer invocation; only then are the buffered gaps moot.
        #
        # DEFECT C repair (2026-11-04): the delegation target must be one the
        # RUNTIME COMMAND REACHES — the chain is followed through shell
        # assignments AND `for ... in` loop lists.  The previous
        # implementation scanned EVERY shell assignment in the file, so a
        # `.sh` token belonging to an unrelated command made an UNTRACEABLE
        # wrapper look "covered" (scripts/_wf_r28_gpu_master.sh: the gate
        # could not trace how `cmd` is populated — `_enumerate_contributor`
        # returned understood=False — yet reported 0 coverage gaps).
        if wrapper_emissions == 0:
            reachable: list[str] = []
            for exec_site in w["exec_sites"]:
                for name in _delegation_reachable(text, exec_site["var"]):
                    if name not in reachable:
                        reachable.append(name)
            for name in reachable:
                target = REPO_ROOT / name
                if not target.is_file():
                    target = Path(name)
                delegated.append(
                    {
                        "file": rel,
                        "delegates_to": name,
                        "reachable_from": sorted(
                            {
                                s["var"]
                                for s in w["exec_sites"]
                                if name in _delegation_reachable(text, s["var"])
                            }
                        ),
                        "exists": target.is_file(),
                        "in_launch_contract_population": target
                        in set(GATE._wrapper_paths()),
                        "has_static_trainer_call": target.is_file()
                        and _has_static_trainer_call(target),
                    }
                )
            mine = [d for d in delegated if d["file"] == rel]
            covered = bool(mine) and all(
                d["exists"]
                and d["in_launch_contract_population"]
                and d["has_static_trainer_call"]
                for d in mine
            )
            if not covered:
                gaps.extend(local_gaps)
        else:
            gaps.extend(local_gaps)

    for d in delegated:
        if not d["exists"]:
            gaps.append(
                {
                    "kind": "delegated_wrapper_missing",
                    "file": d["file"],
                    "line": 0,
                    "detail": (
                        f"{d['file']} delegates to {d['delegates_to']}, which does "
                        "not exist on disk; the runtime command cannot be traced"
                    ),
                }
            )
        elif not d["in_launch_contract_population"]:
            gaps.append(
                {
                    "kind": "delegated_wrapper_not_audited",
                    "file": d["file"],
                    "line": 0,
                    "detail": (
                        f"{d['file']} delegates to {d['delegates_to']}, which is "
                        "not in the launch-contract gate's audited population, so "
                        "the flags it passes are not checked anywhere"
                    ),
                }
            )

    # ---- R2: is the value YAML-derived? (launcher-side transports only) ---
    for site in sites:
        rel = site["wrapper"]
        env_names = yaml_env_names.get(rel, set())
        local_names = yaml_local_names.get(rel, set())
        flag = site["flag"]
        if site["kind"] == "add_flag":
            var = site.get("value_var", "")
            if var not in env_names:
                violations.append(
                    {
                        "kind": "value_not_yaml_derived",
                        "file": rel,
                        "line": site["line"],
                        "flag": flag,
                        "detail": (
                            f"{rel}:{site['line']} transports {flag} with the value "
                            f"${var}, and ${var} is not one of the environment "
                            "variables the YAML->env serialisation writes; the "
                            "value would live only in the shell script (§5.11)"
                        ),
                    }
                )
            if not site.get("conditional"):
                violations.append(
                    {
                        "kind": "value_not_conditional",
                        "file": rel,
                        "line": site["line"],
                        "flag": flag,
                        "detail": (
                            f"{rel}:{site['line']} transports {flag} "
                            "unconditionally; §5.11 requires the flag to be "
                            "emitted only when the YAML declares the key (an "
                            "emptiness guard on the value)"
                        ),
                    }
                )
        elif site["kind"] == "shell_literal":
            raw = site.get("raw", "")
            refs = _VARREF.findall(raw)
            # The guard of a switch-like transport is the `if` above it.
            guard_refs = _guard_refs(texts_by_rel.get(rel, ""), site["line"])
            for ref in list(refs) + list(guard_refs):
                if ref not in env_names:
                    violations.append(
                        {
                            "kind": "value_not_yaml_derived",
                            "file": rel,
                            "line": site["line"],
                            "flag": flag,
                            "detail": (
                                f"{rel}:{site['line']} transports {flag}, and "
                                f"${ref} is not one of the environment variables "
                                "the YAML->env serialisation writes (§5.11)"
                            ),
                        }
                    )
                    break
        elif site["kind"] == "assembly_literal":
            # DEFECT C repair (2026-11-04): the flag is written as a LITERAL
            # into the runtime command string.  A value that lives only in
            # the shell script is a knob nobody audits (§5.11's own words) —
            # this is the defect class that let `use_fm_objective` go missing
            # from 5 of 7 CFM configs.  The wrapper-legal flag set (the flags
            # that select WHICH run) is exempt, exactly as on a static
            # invocation line.
            if flag not in GATE.ALLOWED_TRAINER_FLAGS:
                value = site.get("value_expr") or ""
                violations.append(
                    {
                        "kind": "value_not_yaml_derived",
                        "file": rel,
                        "line": site["line"],
                        "flag": flag,
                        "detail": (
                            f"{rel}:{site['line']} writes {flag}"
                            + (f" {value}" if value else "")
                            + " as a LITERAL into the runtime command it "
                            "assembles; a hyperparameter that lives only in "
                            "the shell script is a knob nobody audits (§5.11). "
                            "The wrapper-legal flag set is "
                            + " ".join(sorted(GATE.ALLOWED_TRAINER_FLAGS))
                        ),
                    }
                )
        elif site["kind"] == "python_literal":
            chunk = site.get("chunk", "")
            sourced = any(
                re.search(r"\b" + re.escape(name) + r"\b", chunk)
                for name in local_names
            )
            if not sourced:
                violations.append(
                    {
                        "kind": "value_not_yaml_derived",
                        "file": rel,
                        "line": site["line"],
                        "flag": flag,
                        "detail": (
                            f"{rel}:{site['line']} transports {flag} from the "
                            "python emitter block, and that block's value does "
                            "not come from a YAML read (§5.11)"
                        ),
                    }
                )

    # ---- R1: does the trainer declare the flag? --------------------------
    n_trainer_invocations = 0
    n_config_invocations = 0
    #: How many configs in the population route to each trainer.  A site-level
    #: violation is a statement about the LAUNCH CONTRACT ("this wrapper can
    #: transport this flag to this trainer"), not a prediction that some
    #: shipped config trips it today — but the reader must be able to tell
    #: how wide that contract is, so the count travels with the finding.
    configs_per_trainer: dict[str, int] = {}
    for cfg_entry in configs:
        k = str(cfg_entry["trainer"])
        configs_per_trainer[k] = configs_per_trainer.get(k, 0) + 1
    for site in sites:
        rel = site["wrapper"]
        flag = site["flag"]
        guard = site.get("guarded")
        site_flags = {
            s["flag"] for s in sites
            if s["wrapper"] == rel and s["kind"] == "shell_literal"
        }
        if site["kind"] == "assembly_literal":
            # The assembly LITERALLY names its trainer (or none): the R1
            # question — "does that trainer's parser declare the flag?" —
            # has exactly one subject, so the trainer set is narrowed to it
            # instead of every trainer in the population.
            tkey = site.get("trainer_key")
            candidate_trainers = (
                [(tkey, trainers[tkey])] if tkey in trainers else []
            )
        else:
            candidate_trainers = list(trainers.items())
        for key, trainer in candidate_trainers:
            if guard is not None and trainer.name not in guard:
                continue
            src = trainer.read_text(encoding="utf-8", errors="ignore")
            if site["kind"] == "add_flag":
                # start.sh appends the flag only when its SOURCE PROBE found
                # the flag literal in the trainer's source (or the probe
                # failed, in which case the whole legacy set is appended).
                probe_hit = ('"%s"' % flag) in src or ("'%s'" % flag) in src
                if not probe_hit:
                    continue
            elif site["kind"] == "shell_literal":
                # the events probe: the source carries one of the site's flags
                if not any(
                    ('"%s"' % f) in src or ("'%s'" % f) in src for f in site_flags
                ):
                    continue
            declared = declared_cache[key][0]
            if declared is None:
                gaps.append(
                    {
                        "kind": "parser_not_captured",
                        "file": _rel(trainer),
                        "line": site["line"],
                        "detail": (
                            f"the parser of {_rel(trainer)} could not be captured, "
                            f"so the gate cannot prove that {flag} — nor any other "
                            "flag on this runtime launch — is declared; a check "
                            "that cannot check must not certify the launch"
                        ),
                    }
                )
                continue
            n_trainer_invocations += 1
            if flag not in declared:
                ambiguous = _argparse_ambiguous(declared, flag)
                if len(ambiguous) == 1:
                    # argparse accepts this as an unambiguous prefix
                    abbreviations.append(
                        {
                            "kind": "abbreviated_flag",
                            "file": rel,
                            "line": site["line"],
                            "flag": flag,
                            "trainer": _rel(trainer),
                            "detail": (
                                f"{rel}:{site['line']} transports {flag} to "
                                f"{_rel(trainer)}, whose parser declares "
                                f"{ambiguous[0]} but not {flag}; argparse accepts "
                                "the abbreviation today (allow_abbrev defaults to "
                                "True), so the launch does NOT abort — but the "
                                "spelling is a fragile contract: a second "
                                f"{ambiguous[0].split('--')[1][:1]}* option would "
                                'make it argparse\'s "ambiguous option" error'
                            ),
                        }
                    )
                    continue
                why = (
                    "ambiguous with "
                    + ", ".join(ambiguous)
                    + ' -> argparse raises "ambiguous option"'
                    if ambiguous
                    else 'the launch would abort with "unrecognized arguments"'
                )
                n_routing = configs_per_trainer.get(key, 0)
                trigger = ""
                cond = _guard_condition(texts_by_rel.get(rel, ""), site["line"])
                if cond:
                    trigger = f" Trigger: {cond}."
                violations.append(
                    {
                        "kind": "undeclared_flag",
                        "file": rel,
                        "line": site["line"],
                        "flag": flag,
                        "trainer": _rel(trainer),
                        "n_configs_routing_to_trainer": n_routing,
                        "reachability": (
                            "live" if n_routing else "no-config-routes-here"
                        ),
                        "detail": (
                            f"{rel}:{site['line']} can transport {flag} to "
                            f"{_rel(trainer)}, whose parser does not declare it; "
                            f"{why} before the run starts (§5.11: the legal flag "
                            "set is defined by the trainer's own parser). "
                            f"{n_routing} config(s) in the population route to "
                            f"{_rel(trainer)}.{trigger}"
                        ),
                    }
                )

    # ---- the per-config part: the YAML's own command template -------------
    for cfg_entry in configs:
        trainer = cfg_entry["trainer"]
        key = str(trainer)
        command = cfg_entry["command"]
        if not command:
            continue
        declared = declared_cache[key][0]
        if declared is None:
            gaps.append(
                {
                    "kind": "parser_not_captured",
                    "file": _rel(trainer),
                    "line": 0,
                    "detail": (
                        f"the parser of {_rel(trainer)} could not be captured, so "
                        "the gate cannot prove that the flags in "
                        f"{cfg_entry['path']}'s entry command are declared; a "
                        "check that cannot check must not certify the launch"
                    ),
                }
            )
            continue
        for flag, value in _flag_occurrences(command):
            n_config_invocations += 1
            if flag not in declared:
                ambiguous = _argparse_ambiguous(declared, flag)
                if len(ambiguous) == 1:
                    # argparse accepts the abbreviation: NOT a violation, but
                    # recorded (a fragile spelling).  The R2 check below still
                    # runs — the value's provenance is a separate question.
                    abbreviations.append(
                        {
                            "kind": "abbreviated_flag",
                            "file": cfg_entry["path"],
                            "line": 0,
                            "flag": flag,
                            "trainer": _rel(trainer),
                            "detail": (
                                f"{cfg_entry['path']} passes {flag} to "
                                f"{_rel(trainer)}, whose parser declares "
                                f"{ambiguous[0]} but not {flag}; argparse accepts "
                                "the abbreviation today (allow_abbrev defaults to "
                                "True), so the launch does NOT abort — but the "
                                "spelling is a fragile contract (§5.11)"
                            ),
                        }
                    )
                else:
                    why = (
                        "ambiguous with "
                        + ", ".join(ambiguous)
                        + ' -> argparse raises "ambiguous option"'
                        if ambiguous
                        else 'the launch would abort with "unrecognized arguments"'
                    )
                    violations.append(
                        {
                            "kind": "undeclared_flag",
                            "file": cfg_entry["path"],
                            "line": 0,
                            "flag": flag,
                            "trainer": _rel(trainer),
                            "detail": (
                                f"{cfg_entry['path']} passes {flag} to "
                                f"{_rel(trainer)} (through the runtime command "
                                "start.sh assembles), whose parser does not declare "
                                f"it; {why} (§5.11)"
                            ),
                        }
                    )
            if cfg_entry["command_source"] == "shell" and value and not _PLACEHOLDER.match(value):
                violations.append(
                    {
                        "kind": "value_not_yaml_derived",
                        "file": cfg_entry["path"],
                        "line": 0,
                        "flag": flag,
                        "detail": (
                            f"{cfg_entry['path']} reaches the launcher's "
                            f"synthesised legacy command, which gives {flag} the "
                            f"literal value {value!r}; the legacy command lives in "
                            "scripts/start.sh, so only a YAML placeholder may "
                            "carry it (§5.11)"
                        ),
                    }
                )

    # ---- the counts -------------------------------------------------------
    mention_counts = {}
    for w in wrappers:
        rel = _rel(w["path"])
        mention_counts[rel] = sum(
            w["text"].count(s["flag"]) for s in sites if s["wrapper"] == rel
        )

    n_sites = len({(s["wrapper"], s["line"], s["flag"]) for s in sites})
    n_distinct_flags = len({s["flag"] for s in sites})

    return {
        "gate": "runtime_transport",
        "rule": (
            "AGENTS.md §5.11 — a flag transported at runtime (inside a command "
            "string, where no static invocation line exists) must be declared by "
            "the invoked trainer's own parser, and its value must be "
            "YAML-derived"
        ),
        "verdict": "PASS" if not violations and not gaps else "FAIL",
        "n_wrappers": len(wrappers),
        "wrappers": [_rel(w["path"]) for w in wrappers],
        "n_sites": n_sites,
        "n_site_flags": len(sites),
        "n_distinct_flags": n_distinct_flags,
        "sites": [
            {
                "wrapper": s["wrapper"],
                "line": s["line"],
                "kind": s["kind"],
                "flag": s["flag"],
                "value_expr": s.get("value_expr", ""),
                "value_var": s.get("value_var", ""),
                "guarded": sorted(s["guarded"]) if s.get("guarded") else None,
                "conditional": s.get("conditional"),
            }
            for s in sites
        ],
        "n_per_config_contributors": len(per_config_contributors),
        "n_mentions": mention_counts,
        "count_basis": (
            "sites and (site, flag) PAIRS — an INVOCATION census, not a census "
            "of textual mentions; the mention count is reported separately so "
            "the two are never confused"
        ),
        "n_configs": len(configs),
        "n_configs_unresolved": len(unresolved_configs),
        "configs_unresolved": unresolved_configs,
        "n_trainers": len(trainers),
        "n_trainer_invocations_checked": n_trainer_invocations,
        "n_config_invocations_checked": n_config_invocations,
        "n_invocations_checked": n_trainer_invocations + n_config_invocations,
        "delegated": delegated,
        "n_violations": len(violations),
        "violations": violations,
        "n_abbreviations": len(abbreviations),
        "abbreviations": abbreviations,
        "n_coverage_gaps": len(gaps),
        "coverage_gaps": gaps,
    }


def _guard_refs(text: str, lineno: int) -> list[str]:
    """The variable references of the `if` condition guarding line `lineno`."""
    lines = text.split("\n")
    for i in range(lineno - 2, max(-1, lineno - 6), -1):
        if i < 0:
            break
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.startswith("if ") or stripped.startswith("if["):
            return _VARREF.findall(stripped)
        return []
    return []


def _guard_condition(text: str, lineno: int) -> str:
    """The `if` CONDITION that guards line `lineno`, or "" if there is none.

    Used to say WHICH condition makes a transport fire, so the finding names
    the trigger instead of asserting an unconditional abort.  A switch-like
    transport (``if COND; then A=...; else A=...; fi``) fires only when COND
    holds — which is why "can transport" and "will transport" are different
    claims and the gate must not conflate them.
    """
    lines = text.split("\n")
    for i in range(lineno - 2, max(-1, lineno - 6), -1):
        if i < 0:
            break
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.startswith("if ") or stripped.startswith("if["):
            cond = stripped[3:].strip()
            for suffix in ("; then", ";then"):
                if cond.endswith(suffix):
                    cond = cond[: -len(suffix)].strip()
            return cond
        return ""
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument(
        "--config-root",
        default=None,
        metavar="PATH",
        help="the config tree to enumerate (default: configs/)",
    )
    ap.add_argument(
        "--wrapper",
        action="append",
        default=None,
        metavar="PATH",
        help="audit only this runtime-string wrapper (repeatable)",
    )
    args = ap.parse_args(argv)

    root = Path(args.config_root) if args.config_root else None
    report = audit(root, wrappers=[Path(w) for w in args.wrapper] if args.wrapper else None)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["verdict"] == "PASS" else (
            2 if report["verdict"] == "SKIP" else 1
        )

    if report["verdict"] == "SKIP":
        print(f"[runtime-transport] SKIP: {report.get('reason', 'not applicable')}")
        return 2

    print(
        f"[runtime-transport] {report['n_wrappers']} runtime-string wrapper(s), "
        f"{report['n_sites']} transport site(s), "
        f"{report['n_site_flags']} (site, flag) pair(s), "
        f"{report['n_distinct_flags']} distinct flag(s)"
    )
    print(
        f"[runtime-transport] count basis: {report['count_basis']}; textual "
        f"mentions {report['n_mentions']}"
    )
    print(
        f"[runtime-transport] population: {report['n_configs']} config(s) -> "
        f"{report['n_trainers']} trainer(s); "
        f"{report['n_trainer_invocations_checked']} (trainer, flag) + "
        f"{report['n_config_invocations_checked']} (config, flag) invocation(s) "
        "checked against the trainers' parsers"
    )
    for v in report["violations"]:
        loc = f"{v['file']}:{v['line']}" if v["line"] else v["file"]
        print(f"  VIOLATION {loc} [{v['kind']}] {v['detail']}")
    if report["n_abbreviations"]:
        print(
            f"[runtime-transport] NOTE: {report['n_abbreviations']} flag(s) reach "
            "the trainer only as an argparse PREFIX of a declared option; the "
            "launch does not abort, but the spelling is fragile (a second "
            "option with that prefix makes it 'ambiguous option')"
        )
        for a in report["abbreviations"]:
            loc = f"{a['file']}:{a['line']}" if a["line"] else a["file"]
            print(f"    {loc} [{a['kind']}] {a['detail']}")
    if report["n_coverage_gaps"]:
        print(
            f"[runtime-transport] COVERAGE GAP: {report['n_coverage_gaps']} "
            "item(s) the gate could not check; the verdict is FAIL"
        )
        for g in report["coverage_gaps"]:
            loc = f"{g['file']}:{g['line']}" if g["line"] else g["file"]
            print(f"    {loc} [{g['kind']}] {g['detail']}")
    if report["n_configs_unresolved"]:
        print(
            f"[runtime-transport] NOTE: {report['n_configs_unresolved']} config(s) "
            "name no resolvable trainer (their launch dies before any flag "
            "matters); listed in the JSON report"
        )
    print(f"[runtime-transport] {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
