#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""AGENTS.md §5.11 gate — training and inference launch from YAML only.

The rule (user directive 2026-10-09, verbatim "以后要求能且只能用yaml配置启动
训练和推理，no cli参数"): every knob that reaches a trainer must be declared in
the YAML config. A launch wrapper may set up the environment, detach the
process, and redirect output — it may not hand the trainer a hyperparameter.

What this gate checks, and why each check is the right shape:

1. `--extra-flags` anywhere in `scripts/launch_*.sh` is a violation. That is the
   escape hatch which let 12 hyperparameters live outside the YAML on
   2026-10-09 (BLK-C5's five levers + device/batch/early-stop/save-smiles).
2. A `train_*.py` invocation carrying any `--flag` outside the allowlist is a
   violation. The allowlist is deliberately tiny and covers only plumbing that
   identifies WHICH config to run, never WHAT to train. The trainer may be
   named LITERALLY (`... train_cfm_yuelbond_wasserstein.py`) or HELD IN A
   SHELL VARIABLE (`"${TRAINER}"`) — both are resolved, because the second
   form is how a whole class of wrappers hid every flag they passed.
3. `scripts/start.sh` must refuse `--extra-flags`, so the escape hatch is closed
   at the surface every wrapper funnels through — otherwise a future wrapper
   would simply re-introduce it.

Why source-scanning rather than a dry-run diff: a dry-run only proves the
command a wrapper builds *today*, for the one config it names. These wrappers
each hardcode a single YAML path, so a dry-run cannot see the other 12 configs,
and any new wrapper would start at zero coverage. The rule being enforced is
about the shape of every launch surface, so the check has to be about shape too.

The literal-regex blind spot (2026-11-02): `_TRAINER_CALL` sees the literal
text `train_x.py` and nothing else, so a wrapper that holds its trainer in a
shell variable —

    TRAINER="${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py"
    ...
    "${PY}" "${TRAINER}" --epochs 30 --hidden 128 --w2-weight 0.05

— hid every flag it passed to the trainer. That made check 2 a check that
COULD NOT FAIL on an entire class of wrappers (the same defect class as the
launch gates in §6.269.17). The gate now resolves variable-held trainer paths
too, and the resolution is CONSERVATIVE and FAIL-CLOSED: when the gate cannot
resolve a variable, it treats the line as containing a trainer call and flags
it. A gate that answers "probably clean" when it cannot tell is the same
defect it was written to close.

Stdout invariant (AGENTS.md §4): PASS / FAIL / SKIP. Exit 0=PASS, 1=FAIL, 2=SKIP.

The fail-closed GUESS (2026-11-04, Wf-A269 wave-19): "the gate cannot resolve
the variable, so treat the line as a trainer call" is the right DEFAULT, but
it is a guess whenever the gate could have DECIDED. The live case:

    PY="${SHARED_VENV}"            # branch 1
    PY="${PYTHON}"                 # branch 2  <- the last assignment
    ...
    "${PY}" -u molmetal/scripts/r4_lambda_only_run.py --config ... --output-dir ...

The interpreter is assigned in SEVERAL branches, so the gate cannot resolve
`${PY}` — but the SCRIPT is a literal token two positions later. The old
resolution stopped at the interpreter, classified the line UNRESOLVED, and
therefore reported `--output-dir` as a TRAINER flag on a NON-trainer script:
a FALSE ATTRIBUTION (the file is `r4_lambda_only_run.py`, not `train_*.py`),
on a script that DECLARES `--output-dir`. The gate was guessing where it
could have decided.

The fix is a LOOKAHEAD, and it is deliberately narrow so that fail-closed
survives everywhere the gate still cannot decide:

  * the command position holds a variable the gate could not resolve, AND
  * the next non-option token is a LITERAL `.py` path (not a variable), AND
  * every token between them is a SHORT option (`-u`, `-m`, `-X`) — an
    interpreter's own options, not the script's flags.

The third condition is what keeps the fix honest. A script invocation is
followed by its own `--long-options`; an interpreter invocation is followed by
its short options and then the script. So a `--` long option BEFORE the `.py`
means the `.py` is that option's VALUE, not the script — and the lookahead
declines, leaving the line fail-closed. MEASURED 2026-11-04 over the whole
101-wrapper in-contract population: 16 lines hold an unresolved command; the
lookahead decides 4 of them (all four run a NON-trainer script, and every
flag they pass is DECLARED by that script — checked by hand against
`r4_lambda_only_run.py`, `eval_all_evaluators.py`,
`wf_pitfall_guard/check_all.py` and `scripts/reproduce_paper_a.py`); the other
12 stay fail-closed, including the two heredoc drivers
(`"${PY}" - <<'PYEOF'`), where there is no script token to find.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Every directory whose shell files are launch surfaces (2026-11-02, Wf-A269
#: wave-10 finding C4-2).  The gate used to scan ``scripts/launch_*.sh`` and
#: nothing else, so 58 of the repo's 113 tracked ``.sh`` files were never
#: audited — including two that carry exactly the §5.11 violation the gate
#: exists to catch (``scripts/gpu_smoke_w2_sliced.sh`` and
#: ``molmetal/scripts/wf193_run_pairs.sh``).
WRAPPER_ROOTS = ("scripts", "molmetal/scripts")

#: Dated EXCLUSION (2026-11-02, finding C4-2): wrappers under
#: ``scripts/_retired/`` are archived copies kept for history, not launch
#: surfaces.  The glob below is non-recursive, so they are already out of
#: scope — this constant makes that a DELIBERATE decision instead of an
#: accident of the glob, so a future recursive glob cannot quietly re-admit
#: them.
#:
#: 2026-11-04 (the COVERAGE half of the §5.11 gate): the constants became
#: ``dict``s — path -> reason — because a gate that cannot name WHY a file is
#: out of scope is a gate whose population cannot be reviewed.  Every reason
#: below cites the artifact that establishes it; none of them is a path
#: heuristic.  A substring match on a path (e.g. "archived" appearing anywhere
#: in it) is EXACTLY the Wf-A167 mis-classification recorded in the
#: honest-negative catalogue (§6.269): a guard filed
#: ``results/cfm_full_retrain/launch_loop.sh`` under "archived" on a
#: path-based heuristic and reported a green tree while a tracked, executable
#: 5-seed x 5-pocket x 50-epoch CFM retrain loop still hard-forced GPU 0.
#: ⇒ **Every exclusion below is an EXPLICIT LITERAL PATH.**
EXCLUDED_WRAPPER_DIRS: dict[str, str] = {
    "scripts/_retired": (
        "archived copies kept for history, not launch surfaces "
        "(2026-11-02, Wf-A269 wave-10 finding C4-2); the in-tree marker "
        "scripts/_retired/README.md reads '# Retired launch wrappers'"
    ),
    "TODO/archive/wf437_simplify": (
        "archived by Wf-437 (TODO-180 CLOSED); AGENTS.md §7 lists "
        "TODO/archive/wf437_simplify/launch_wf124_p3_g4.sh as the archived G4 "
        "launcher and molmetal/tests/test_wf337g2_launcher_v2.py asserts it "
        "stays archived"
    ),
}

#: Per-FILE exclusions.  A file entry is matched by its EXACT repo-relative
#: path — never by a substring of that path.  Each reason states what the
#: exclusion is and what evidence establishes it, so a reader can re-admit the
#: file the moment that evidence stops holding.
#:
#: The two entries below are **MIGRATION DEFERRALS, not archival claims**:
#: neither file is declared archived anywhere in the tracked tree.  They are
#: excluded because their §5.11 migration is a follow-up wave, and the gate
#: REPORTS their unmigrated flag counts on every run (see
#: ``_residual_census``), so the deferral is visible rather than hidden.  The
#: census deliberately does NOT use the word "VIOLATION": the pre-flight
#: consumer counts lines containing that word to size a failing gate, and a
#: disclosed residual must not be miscounted as one.
#:
#: 2026-11-04 (Wf-A269) — a THIRD entry was REMOVED from this dict, and the
#: superseded rendering is preserved VERBATIM per AGENTS.md §5.5 rule #6:
#:
#:     "results/cfm_full_retrain/launch_loop.sh": (
#:         "MIGRATION DEFERRAL (not an archival claim): LIVE per the repo's own "
#:         "evidence-based classifier (molmetal/scripts/wfa171_gpu_lock_guard_"
#:         "path.py, basis NO_ARCHIVAL_DECLARATION) and per "
#:         "molmetal/reports/wfa165_branchA_refutation/refutation.md; it carries "
#:         "5 unmigrated §5.11 trainer flags, disclosed by the excluded-file "
#:         "residual census on every run"
#:     ),
#:
#: WHY IT WAS REMOVED: the deferral's stated reason ("it carries 5 unmigrated
#: §5.11 trainer flags") became FALSE on 2026-11-04 — the Wf-A269 repair moved
#: all five (--epochs / --hidden-dim / --n-layers / --pocket-id / --seed) out
#: of that wrapper's invocation line and into a static per-(pocket, seed) YAML
#: family (configs/cfm_full_retrain_<pocket>_seed<seed>.yaml).  MEASURED after
#: the repair: the file is CLEAN (0 unmigrated trainer flags) and the
#: yaml-only gate PASSes it when it is scanned explicitly
#: (``--wrapper results/cfm_full_retrain/launch_loop.sh``).
#:
#: WHY THE REMOVAL MATTERS: keeping the entry would have left the repaired
#: file OUTSIDE the gate's CONTRACT with a residual of 0 — a file that is
#: "DISCLOSED BUT NOT ENFORCED", i.e. the §6.269.22 dead-gate class.  The
#: residual census lists only files that still carry flags, so a 0-flag
#: exclusion disappears from the census entirely and the hole becomes SILENT.
#: The file is now IN-CONTRACT: the global population is 113 tracked, 13
#: excluded, 100 scanned.
#:
#: 2026-11-04 (Wf-A269 §5.11 migration of run_wf187_remaining.sh) — a FOURTH
#: entry was REMOVED from this dict, and the superseded rendering is preserved
#: VERBATIM per AGENTS.md §5.5 rule #6:
#:
#:     "run_wf187_remaining.sh": (
#:         "MIGRATION DEFERRAL (not an archival claim): a one-off Wf-187 scratch "
#:         "launcher at the repo root that no tracked file references; it carries "
#:         "77 unmigrated §5.11 trainer flags, disclosed by the excluded-file "
#:         "residual census on every run"
#:     ),
#:
#: WHY IT WAS REMOVED: the deferral's stated reason ("it carries 77 unmigrated
#: §5.11 trainer flags") became FALSE on 2026-11-04 — the migration moved all
#: 77 (11 flags x 7 invocation lines) out of that wrapper's invocation lines
#: and into a static YAML family (configs/wf187_rh_oversample.yaml +
#: configs/wf187_re_oversample.yaml cohort declarations, and the seven
#: per-(metal, seed) launch configs configs/wf187_oversample_<Metal>_seed<n>.yaml).
#: MEASURED after the repair: the file is CLEAN (0 unmigrated trainer flags)
#: and the yaml-only gate PASSes it when it is scanned explicitly
#: (``--wrapper run_wf187_remaining.sh``).  The wrapper also gained a
#: fail-closed pre-launch guard (scripts/_wf_yaml_static_config_c1.py + a
#: cohort/per-pair agreement check), so the migrated YAML cannot drift away
#: from the run it claims to describe.
#:
#: The superseded CLI rendering of every changed invocation line is preserved
#: VERBATIM in dated comments inside run_wf187_remaining.sh itself (AGENTS.md
#: §5.5 rule #6).
EXCLUDED_WRAPPER_FILES: dict[str, str] = {
    "molmetal/reports/wf268a_smoke/run_smoke.sh": (
        "RETIRED 2026-11-04 (not a migration deferral): a Wf-268A wave runner "
        "committed under molmetal/reports/, i.e. a RUN RECORD. The interpreter "
        "it names (/home/hugo/codes/CadButEaas/.venv-vlm/bin/python) no longer "
        "exists on this host and the artifacts it produced sit beside it and "
        "must not be overwritten, so it is retired rather than repaired (the "
        "retirement banner + early exit are inside the file). Its 10 §5.11 "
        "trainer flags are kept VERBATIM as the provenance of the preserved "
        "artifacts — the run.log files do not record the command line — and "
        "they stay disclosed by the excluded-file residual census on every run"
    ),
}

#: The PRE-2026-11-02 population, kept so the report can state how much the
#: scan GREW (finding C4-2).  A coverage change with no number attached is
#: invisible to the next reader, which is how a coverage regression hides.
LEGACY_WRAPPER_GLOBS = ("launch_*.sh",)

#: Flags a launch wrapper may pass to a trainer. Each one selects the RUN, not
#: the MODEL: which config, where output goes, which checkpoint dir. Every
#: hyperparameter (lr, batch, epochs, scheduler, all mechanism toggles) is
#: deliberately absent — those belong in the YAML per §5.11.
ALLOWED_TRAINER_FLAGS = frozenset(
    {
        "--config",
        "--out-dir",
        "--ckpt-dir",
        "--help",
    }
)

#: A trainer entry point, e.g. `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`.
_TRAINER_CALL = re.compile(r"\btrain_[a-z0-9_]+\.py\b")
_FLAG = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")

# Leading detach prefix: env assignments, then `setsid`/`nohup` and their own
# options. `setsid --wait nohup "$PY" trainer.py` is a legal command head whose
# `--wait` belongs to setsid, not to the trainer — without stripping it the gate
# reports `--wait` as a smuggled trainer hyperparameter and pushes authors back
# toward the bare `setsid`, whose non-forking behaviour is what makes
# `rc=$?` after the command readable in the first place.
_DETACH_HEAD = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+)*"
    r"(?:(?:setsid|nohup)(?:\s+--?[A-Za-z][A-Za-z-]*)*\s+)+"
)

#: A leading conditional keyword. `if CUDA_VISIBLE_DEVICES=... setsid --wait
#: "${PY}" "${TRAINER}" \` is a trainer invocation, and without dropping the
#: `if` the detach head below cannot match — so `setsid --wait`'s own `--wait`
#: would be read as a smuggled trainer hyperparameter.
_COND_HEAD = re.compile(r"^\s*(?:if|while|until|elif)\b\s*")


def _strip_detach_head(line: str) -> str:
    """Drop a leading setsid/nohup (plus options) so only the payload is scanned."""
    return _DETACH_HEAD.sub("", _COND_HEAD.sub("", line, count=1), count=1)


# ---------------------------------------------------------------------------
# Variable-held trainer paths (2026-11-02)
# ---------------------------------------------------------------------------
#: A shell assignment: NAME=value (or NAME+=value), optionally prefixed by
#: export/declare/readonly/local. The value keeps any `${...}` references it
#: holds — they are resolved later, against the whole file, because a wrapper
#: may assign the variable before or after the line that uses it.
#:
#: The 2026-11-03 repair added the `+=` operator: a bash ARRAY is assigned
#: with `NAME=( ... )` and appended to with `NAME+=( ... )`, and the array
#: form is where a whole class of launch surfaces hid their flags (see
#: `_read_array`).
_ASSIGN = re.compile(
    r"^\s*(?:export\s+|declare\s+-\w+\s+|readonly\s+|local\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)(\+?=)(.*)$"
)

#: A shell variable reference: ${NAME}, ${NAME:-default}, ${NAME-default}, $NAME.
_VARREF = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-=+?%!])([^}]*))?\}"
    r"|\$([A-Za-z_][A-Za-z0-9_]*)"
)

#: Commands that WRAP another command: they are skipped when looking for the
#: script a line runs.
_WRAPPERS = frozenset(
    {
        "env", "exec", "setsid", "nohup", "time", "nice", "ionice", "stdbuf",
        "timeout", "command", "xargs", "sudo", "doas", "builtin",
    }
)

#: Commands that take DATA arguments, never a script. `echo "... --prereg"`
#: and `grep -qE -- "--metals"` mention flags but execute nothing, so they
#: must never be mistaken for a trainer invocation.
_DATA_COMMANDS = frozenset(
    {
        "echo", "printf", "cat", "tee", "true", "false", "cd", "mkdir", "rm",
        "ls", "grep", "egrep", "sed", "awk", "tail", "head", "test", "[",
        "source", ".", "wait", "sleep", "kill", "wc", "sort", "uniq", "cut",
        "tr", "basename", "dirname", "realpath", "readlink", "date", "id",
        "whoami", "which", "type", "export", "local", "declare", "readonly",
        "return", "exit", "eval", "set", "unset", "shift", "trap", "umask",
        "jobs", "fg", "bg", "mkdir", "rmdir", "touch", "ln", "cp", "mv",
    }
)

#: Interpreters: the token after one of these is the script that runs.
_INTERPRETERS = frozenset({"bash", "sh", "zsh", "ksh", "dash"})


def _shell_tokens(text: str) -> list[str]:
    """Split a shell command head into tokens.

    Quoted spans stay inside one token and a ``$( ... )`` command substitution
    stays attached to the token that holds it: the substituted text is the
    OUTPUT of a command, not a literal the shell would execute, so it must not
    be mistaken for the script being run.
    """
    tokens: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    i = 0
    n = len(text)

    def flush() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    while i < n:
        ch = text[i]
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(text[i : i + 2])
            i += 2
            continue
        if text.startswith("$(", i):
            depth += 1
            buf.append("$(")
            i += 2
            continue
        if ch == ")" and depth:
            depth -= 1
            buf.append(ch)
            i += 1
            continue
        if depth:
            buf.append(ch)
            i += 1
            continue
        if ch in "();&|<>":
            flush()
            tokens.append(ch)
            i += 1
            continue
        if ch.isspace():
            flush()
            i += 1
            continue
        buf.append(ch)
        i += 1
    flush()
    return tokens


def _unquote(token: str) -> str:
    """Strip one layer of surrounding quotes."""
    tok = token.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    return tok


def _match_paren(text: str) -> int | None:
    """Index of the ``)`` that closes the ``(`` opening ``text``.

    Quote-aware: a parenthesis inside a string literal does not open a
    nesting level.
    """
    depth = 0
    quote: str | None = None
    for pos, ch in enumerate(text):
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return pos
    return None


def _split_array(body: str) -> list[str]:
    """The shell words of a bash array literal body."""
    out: list[str] = []
    for tok in _shell_tokens(body):
        bare = _unquote(tok)
        if bare in {"", ";", "|", "&", "<", ">", "(", ")"}:
            continue
        out.append(bare)
    return out


def _read_array(lines: list[str], i: int, first: str) -> tuple[list[str] | None, int]:
    """Read the bash array literal whose ``(`` follows the ``=`` on ``lines[i]``.

    Returns ``(elements, last_line_index)``.  ``elements`` is ``None`` when the
    literal is not an array or is unterminated — the caller then treats the
    value as an opaque scalar rather than pretending it read an array.
    """
    if not first.startswith("("):
        return None, i
    text = first
    last = i
    end = _match_paren(text)
    while end is None and last + 1 < len(lines):
        last += 1
        nxt = lines[last].strip()
        if nxt.startswith("#"):
            continue
        text = f"{text} {nxt}"
        end = _match_paren(text)
    if end is None:
        return None, i
    return _split_array(text[1:end]), last


class _ShellEnv(dict):
    """``dict[str, str | list[str]]`` plus two side tables.

    ``conditional_appends``
        A bash array can be appended to inside an ``if`` block::

            if [[ "${TRAINER}" == *"chemprop_v2_v3"* ]]; then
                COMMON_ARGS+=( --chemprop-v3-cache )
            fi

        MEASURED 2026-11-03 on ``scripts/launch_wf390a1_m2_r2_path_a.sh``:
        that flag NEVER reaches the trainer, because the wrapper runs the
        D-MPNN trainer.  A gate that reports it as a live flag states
        something FALSE about the launch — the same class of error as a
        census that counts a superseded rendering as a live defect
        (§6.269.18.11).  So the append is recorded WITH its condition, and
        the condition is evaluated at the call site: a false condition
        excludes the flag from the argv (and records it as a census entry),
        an unevaluable one keeps it (conservative).

    ``branch_values``
        EVERY value a name is ever assigned, not just the last one.
        MEASURED 2026-11-03 on ``scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh``:
        ``ABLATION_ARGS`` is assigned once per ``case`` branch of a
        four-way ablation, and a gate that keeps only the LAST assignment
        sees 2 of the 4 branches' flags — the other branches' flags are
        invisible, so the gate UNDER-REPORTS.  For a gate, the union of the
        branches is the fail-closed reading: a flag that ANY branch could
        pass is a flag the launch surface carries.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional_appends: dict[str, list[tuple[str, list[str]]]] = {}
        self.branch_values: dict[str, list[str | list[str]]] = {}


#: `if`/`elif` ... `then` — the condition text between them.
_IF_HEAD = re.compile(r"^\s*(?:if|elif)\b(.*?)\bthen\b")
#: `fi` — the end of an `if` block.
_FI = re.compile(r"^\s*fi\b")


def _eval_condition(cond: str, env: dict) -> bool | None:
    """Evaluate a simple shell condition. True / False / None (cannot tell).

    Only the forms this repo actually uses are supported:

    * ``[[ "${VAR}" == *"pat"* ]]`` / ``== "lit"`` / ``!= …``
    * ``[[ -n "${VAR}" ]]`` / ``[[ -z "${VAR}" ]]``

    Anything else returns ``None`` — "cannot tell" — and the caller keeps the
    conservative reading.
    """
    text = cond.strip()
    if text.startswith("[["):
        text = text[2:]
    if text.endswith("]]"):
        text = text[:-2]
    text = text.strip()

    m = re.match(r'^"?\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?"?\s*(==|!=)\s*(.+)$', text)
    if m:
        name, op, rhs = m.group(1), m.group(2), m.group(3).strip()
        if name not in env:
            return None
        value = env[name]
        if isinstance(value, list):
            value = " ".join(value)
        value = _expand(value, env)
        pattern = rhs.strip()
        if pattern.startswith('"') and pattern.endswith('"'):
            pattern = pattern[1:-1]
            matched = value == pattern
        elif pattern.startswith("*") and pattern.endswith("*"):
            needle = pattern[1:-1]
            matched = needle in value
        else:
            matched = value == pattern
        return matched if op == "==" else not matched

    m = re.match(r'^-([nz])\s+"?\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?"?$', text)
    if m:
        name = m.group(2)
        if name not in env:
            return None
        value = env[name]
        if isinstance(value, list):
            value = " ".join(value)
        value = _expand(value, env)
        return bool(value) if m.group(1) == "n" else not value
    return None


def _collect_assignments(text: str) -> _ShellEnv:
    """Map every shell variable the wrapper assigns to its (unexpanded) value.

    Read from PHYSICAL lines, not the continuation-joined logical lines: a
    joined line would fold the trainer's own ``--flags`` into the assigned
    value and the expansion would then read a flag as part of a path.

    A bash ARRAY (``NAME=( ... )``) is stored as the LIST of its elements,
    because the array is the construct a launch surface uses to hide its flags
    from a line-oriented scan: ``"${COMMON_ARGS[@]}"`` puts sixteen
    hyperparameters on the trainer's command line while the literal text of
    the invocation line carries none of them (MEASURED 2026-11-03 on
    ``scripts/launch_wf390a1_m2_r2_path_a.sh``, which the gate PASSed while
    the wrapper passed sixteen flags).  The array is resolved at the call
    site by :func:`_trainer_argv`.
    """
    env = _ShellEnv()
    cond_stack: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if _FI.match(stripped):
            if cond_stack:
                cond_stack.pop()
            i += 1
            continue
        m_if = _IF_HEAD.match(stripped)
        if m_if:
            cond_stack.append(m_if.group(1))
            i += 1
            continue
        m = _ASSIGN.match(stripped)
        if not m:
            i += 1
            continue
        name, op, raw = m.group(1), m.group(2), m.group(3).strip()
        cond = cond_stack[-1] if cond_stack else None
        if raw.startswith("("):
            elements, last = _read_array(lines, i, raw)
            if elements is not None:
                if op == "+=" and isinstance(env.get(name), list):
                    if cond:
                        env.conditional_appends.setdefault(name, []).append(
                            (cond, elements)
                        )
                    else:
                        env[name] = list(env[name]) + elements
                else:
                    env[name] = elements
                env.branch_values.setdefault(name, []).append(list(elements))
                i = last + 1
                continue
            # An unterminated '(' is not an array this gate can read; fall
            # through and keep the value as an opaque scalar.
        value = raw
        while value.endswith("\\") and i + 1 < len(lines):
            value = value[:-1].strip()
            i += 1
            nxt = lines[i].strip()
            if nxt.startswith("#"):
                break
            value = f"{value} {nxt}".strip()
        hash_at = value.find(" #")
        if hash_at != -1:
            value = value[:hash_at]
        env[name] = _unquote(value)
        env.branch_values.setdefault(name, []).append(env[name])
        i += 1
    return env


def _expand(value: str, env: dict[str, str], depth: int = 0, seen: tuple = ()) -> str:
    """Expand shell variable references in ``value`` against ``env``.

    A reference the environment cannot answer is LEFT IN PLACE, so the caller
    can tell "resolved to something that is not a trainer" from "not resolved
    at all". The second case is the fail-closed one.
    """
    if depth > 10:
        return value

    def repl(m: re.Match) -> str:
        name = m.group(1) or m.group(4)
        if name is None:
            return m.group(0)
        default = m.group(3) if m.group(2) else None
        if name in seen:
            # A self-reference (`PY="${PY:-/path/to/python}"`) means "use the
            # default if unset" — the default IS the answer.
            return default if default is not None else m.group(0)
        if name in env:
            value = env[name]
            if isinstance(value, list):
                # A bash array referenced WITHOUT the [@] subscript expands to
                # its elements joined by spaces (bash's own rule).
                return " ".join(value)
            return _expand(value, env, depth + 1, seen + (name,))
        if default is not None:
            return default
        return m.group(0)

    return _VARREF.sub(repl, value)


def _last_segment(path: str) -> str:
    """The final path segment of an expanded value."""
    return path.rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Bash arrays (2026-11-03, Wf-A269 wave-19)
# ---------------------------------------------------------------------------
#: A bash array reference: ``${NAME[@]}`` or ``${NAME[*]}``.  The 2026-11-02
#: `_VARREF` cannot match these — its operator group accepts only
#: `-`/`=`/`+`/`?`/`%`/`!`, and `[` is not among them, so the closing brace
#: can never be reached.  MEASURED: ``_VARREF.search('${COMMON_ARGS[@]}')`` →
#: ``None``.  That single fact is what let 13 live wrappers hide 130 flags
#: from this gate.
_ARRAY_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\[([@*])\]")


def _array_elements(
    name: str, env, _depth: int = 0
) -> tuple[list[str], list[str]]:
    """The elements of a bash array that REACH the command line, and those that do not.

    A conditional append — ``if [[ ... ]]; then ARR+=( x ); fi`` — reaches the
    command line only when its condition holds.  MEASURED 2026-11-03 on
    ``scripts/launch_wf390a1_m2_r2_path_a.sh``: ``--chemprop-v3-cache`` is
    appended under ``if [[ "${TRAINER}" == *"chemprop_v2_v3"* ]]`` while the
    wrapper runs the D-MPNN trainer, so the flag NEVER reaches the command
    line.  Reporting it as a live flag would state something FALSE about the
    launch — the same class of error as a census that counts a superseded
    rendering as a live defect (§6.269.18.11).

    An UNEVALUABLE condition keeps the elements (conservative: a gate must
    not drop a flag it cannot prove is dead) and is reported as such.

    Two further resolutions happen here, both of which were blind spots
    before 2026-11-03:

    * **BRANCH UNION.**  A name assigned in several branches (``case`` arms,
      ``if``/``else`` arms) contributes the UNION of its values.  Keeping only
      the last assignment UNDER-REPORTS — MEASURED on
      ``scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh``: ``ABLATION_ARGS`` has
      four branch values and the last one alone exposed 2 of the flags the
      other three branches pass.
    * **NESTED ARRAYS.**  ``ARR=( "${OTHER[@]}" )`` holds the *reference*, not
      the elements.  Resolving one level only turns the reference into a
      single opaque token, and the flags inside ``OTHER`` vanish from the
      scan — the same blind spot one level down.  So the elements are
      resolved recursively.
    """
    live: list[str] = []
    dead: list[str] = []
    if _depth > 5:
        # A cyclic reference — refuse to expand further, and treat what we
        # have as live so the caller fails closed.
        return live, dead

    values = getattr(env, "branch_values", {}).get(name)
    if values is None:
        value = env.get(name)
        values = [value] if value is not None else []
    for value in values:
        if isinstance(value, list):
            live.extend(value)
        elif value is not None:
            live.append(str(value))

    for cond, elements in getattr(env, "conditional_appends", {}).get(name, []):
        if _eval_condition(cond, env) is False:
            dead.extend(elements)
        else:
            live.extend(elements)

    # Nested array references inside the elements.
    resolved: list[str] = []
    for tok in live:
        m = _ARRAY_REF.search(tok)
        if m:
            sub_live, sub_dead = _array_elements(m.group(1), env, _depth + 1)
            resolved.extend(sub_live)
            dead.extend(sub_dead)
            continue
        resolved.append(tok)
    return resolved, dead


def _trainer_argv(
    line: str, env: dict[str, str | list[str]]
) -> tuple[list[str], list[str], list[str]]:
    """Resolve the argv a trainer invocation passes, and name what it could not.

    Returns ``(tokens, unresolved_array_names, conditionally_excluded_flags)``.

    The tokens are the arguments AFTER the script token, with bash array
    expansions (``"${COMMON_ARGS[@]}"``) spliced in element by element and
    scalar references expanded.  ``unresolved_array_names`` names every array
    the line references that this gate could NOT read — the FAIL-CLOSED case:
    the flags inside such an array are invisible to the scan, so the caller
    must not treat the line as clean.

    ``conditionally_excluded_flags`` names the flags that are in a bash array
    but whose append condition is FALSE for this wrapper, so they do not
    reach the trainer.  They are a CENSUS entry, not a violation: the
    launch does not pass them today.  See :func:`_array_elements`.
    """
    toks = _shell_tokens(_strip_detach_head(line))
    idx = _script_index(toks, env)
    if idx is None:
        return [], [], []
    resolved: list[str] = []
    unresolved: list[str] = []
    excluded: list[str] = []
    for tok in toks[idx + 1 :]:
        m = _ARRAY_REF.search(tok)
        if m:
            name = m.group(1)
            value = env.get(name)
            if isinstance(value, list) or name in getattr(
                env, "conditional_appends", {}
            ):
                # A bare "${NAME[@]}" contributes every element; the
                # "${NAME[@]+"${NAME[@]}"}" guard contributes them too when
                # the array is set.  Either way the elements ARE the argv.
                live, dead = _array_elements(name, env)
                resolved.extend(live)
                excluded.extend(dead)
                continue
            unresolved.append(name)
            resolved.append(tok)
            continue
        resolved.append(_expand(tok, env))
    return resolved, unresolved, excluded


def _is_interpreter(segment: str) -> bool:
    """Is this path segment a program that RUNS a script given to it?"""
    return segment in _INTERPRETERS or segment.startswith("python")


def _looks_like_script_path(token: str) -> bool:
    """Is this token a LITERAL path to a python script?

    Literal means: the path is spelled out. A variable (``${TRAINER}``) is
    not, because the gate cannot prove what it names — and a token that ends
    in ``.py`` only because a VARIABLE happens to expand to one is exactly
    the case the fail-closed rule exists for.
    """
    bare = _unquote(token)
    if not bare.endswith(".py"):
        return False
    return not _VARREF.search(bare)


def _unresolved_command(command: str, env: dict[str, str]) -> bool:
    """Is the command position a variable this gate could NOT resolve?

    A variable that DOES resolve to a concrete path is not unresolved: the
    gate can act on it.  This is the fail-closed case — the command may be
    an interpreter the gate cannot see through.
    """
    if not _VARREF.search(command):
        return False
    return "$" in _expand(command, env)


def _head_tokens(line: str) -> list[str]:
    """Tokenize the command head of a logical line.

    The head stops at the first flag: everything after ``--something`` belongs
    to the command being run, not to the command head itself.
    """
    s = _COND_HEAD.sub("", line, count=1)
    s = _DETACH_HEAD.sub("", s, count=1)
    tokens: list[str] = []
    for tok in _shell_tokens(s):
        bare = _unquote(tok)
        if not bare:
            continue
        if bare.startswith("--"):
            break
        if bare in {";", "|", "||", "&&", "&", "(", ")", ">", ">>", "<"}:
            break
        tokens.append(bare)
    return tokens


def _script_token(tokens: list[str], env: dict[str, str]) -> str | None:
    """Return the token that holds the script this command runs, or None.

    * a wrapper (``env``, ``setsid``, ...) is skipped;
    * a DATA command (``echo``, ``grep``, ...) runs no script, so there is
      nothing to audit on that line;
    * an interpreter's next token is the script;
    * anything else is the script itself.
    """
    idx = _script_index(tokens, env)
    if idx is None:
        return None
    return tokens[idx]


def _script_index(tokens: list[str], env: dict[str, str]) -> int | None:
    """Index of the token that holds the script this command runs, or None.

    The index (rather than the token) is what the argv resolution needs: the
    trainer's ARGUMENTS are the tokens after it.
    """
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if _ASSIGN.match(token):
            i += 1  # `VAR=...` / `VAR=$(...)` — an assignment, not the command
            continue
        segment = _last_segment(_expand(token, env))
        if segment in _WRAPPERS:
            i += 1
            continue
        break
    if i >= n:
        return None
    command = tokens[i]
    segment = _last_segment(_expand(command, env))
    if segment in _DATA_COMMANDS:
        return None
    if _is_interpreter(segment):
        i += 1
        while i < n and tokens[i].startswith("-"):
            i += 1  # `python -u train_x.py` — the option is not the script
        if i >= n:
            return None
        script_index = i
    elif _unresolved_command(command, env):
        # The command position holds a variable the gate could not resolve: it
        # may be the INTERPRETER (the script follows it) or it may BE the
        # script (its own `--long-options` follow it).  The two are
        # distinguishable — an interpreter is followed by its SHORT options and
        # then the script, a script by its LONG options — so the gate can
        # DECIDE instead of guessing.  When it cannot, the line stays
        # fail-closed.  MEASURED 2026-11-04 (see the module docstring).
        script_index = i
        j = i + 1
        while j < n:
            tok = tokens[j]
            if tok.startswith("--"):
                break  # a long option: a following `.py` would be its VALUE
            if tok.startswith("-"):
                j += 1  # an interpreter's own short option (-u, -m, -X)
                continue
            if _looks_like_script_path(tok):
                script_index = j
            break
    else:
        script_index = i
    if tokens[script_index].lstrip()[:1] in {"|", ";", "&", "<", ">"}:
        return None  # a pipeline/redirect fragment, not a script path
    return script_index


def _classify_script(script: str, env: dict[str, str]) -> str:
    """TRAINER | NOT | UNRESOLVED for the script a command line runs.

    ``UNRESOLVED`` is the fail-closed case: the token holds a variable the
    wrapper never assigns (or assigns to something this gate cannot expand),
    so the gate cannot prove the command is not a trainer call. Per the
    CONSERVATIVE rule at the top of this file, the caller treats that as a
    trainer call rather than as clean — a gate that guesses "probably fine"
    when it cannot tell is the defect it was written to close.
    """
    expanded = _expand(script, env)
    if _TRAINER_CALL.search(expanded):
        return "TRAINER"
    if "$" not in script:
        return "NOT"
    # The trainer test is a FILE-NAME test, so the basename decides it. If the
    # basename is visible it is decisive — an unresolvable directory prefix
    # cannot turn `helper_check.py` into `train_x.py`.
    basename = expanded.rstrip("/").split("/")[-1]
    if not _VARREF.search(basename):
        return "NOT"
    return "UNRESOLVED"


def _trainer_call(line: str, env: dict[str, str]) -> str | None:
    """How this line invokes a trainer, or None if it does not.

    Returns one of ``"literal"`` (the `train_*.py` regex matched),
    ``"variable:<NAME>"`` (a variable resolved to a trainer path) or
    ``"variable-unresolved"`` (fail-closed: the gate could not tell, so it
    assumes the worst).
    """
    if _TRAINER_CALL.search(line):
        return "literal"
    script = _script_token(_head_tokens(line), env)
    if script is None:
        return None
    verdict = _classify_script(script, env)
    if verdict == "TRAINER":
        m = _VARREF.search(script)
        name = (m.group(1) or m.group(4)) if m else "resolved"
        return f"variable:{name}"
    if verdict == "UNRESOLVED":
        return "variable-unresolved"
    return None


def _rel(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    A wrapper outside the repo root is a test fixture, not a crash: the gate
    must report on whatever it is handed.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Wrapper population (2026-11-02, Wf-A269 wave-10 finding C4-2;
#                        2026-11-04, the COVERAGE half)
# ---------------------------------------------------------------------------
def _exclusion_for(path: Path, rel: str | None = None) -> tuple[str, str] | None:
    """``(kind, reason)`` if this wrapper is excluded, else ``None``.

    Matching is by EXPLICIT LITERAL PATH, never by a substring of the path:

    * a file is excluded when its repo-relative path is a key of
      :data:`EXCLUDED_WRAPPER_FILES` (exact equality);
    * a file is excluded when it lives under a key of
      :data:`EXCLUDED_WRAPPER_DIRS` (exact directory, or an exact path
      prefix of that directory plus ``/``).

    The directory check is also applied to the RESOLVED path, so a symlink
    into an excluded tree is excluded too — an exclusion that can be walked
    around is not an exclusion.  There is deliberately NO substring rule
    ("archived" anywhere in the path, a path segment called "old", ...): that
    is the Wf-A167 mis-classification the honest-negative catalogue records
    (§6.269) — a path-based "archived" bucket hid a live, tracked, executable
    CFM retrain loop behind a green report.
    """
    if rel is None:
        rel = _rel(path)
    if rel in EXCLUDED_WRAPPER_FILES:
        return ("file", EXCLUDED_WRAPPER_FILES[rel])

    candidates = {rel}
    try:
        resolved = path.resolve()
        candidates.add(str(resolved.relative_to(REPO_ROOT)))
    except (OSError, ValueError):
        pass
    for excluded, reason in EXCLUDED_WRAPPER_DIRS.items():
        for candidate in candidates:
            if candidate == excluded or candidate.startswith(excluded + "/"):
                return ("dir", reason)
    return None


def _is_excluded_wrapper(path: Path) -> bool:
    """Is this wrapper excluded by :data:`EXCLUDED_WRAPPER_DIRS` / :files?

    Kept as a BOOL for the other gates that consume this module
    (``scripts/_runtime_transport_gate.py``); the reason is available from
    :func:`_exclusion_for`.
    """
    return _exclusion_for(path) is not None


def _git_tracked_sh_paths() -> tuple[list[str] | None, str]:
    """Every TRACKED ``*.sh`` in the repo, or ``(None, reason)``.

    The population is driven by ``git ls-files`` rather than by a directory
    glob, because a glob cannot see the whole repository: the 2026-11-02
    population was two NON-RECURSIVE globs under two roots, so 27 of the
    repo's 113 tracked ``.sh`` files were in no gate population at all — and
    the gate printed "scanned 86 wrapper(s)" with NO denominator, so a reader
    could not tell 86/113 from 86/86 (§6.269.13's class).

    A failure here is NOT swallowed: the caller falls back to the old glob
    and says so, and the coverage record carries ``population_degraded``.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", "*.sh"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip() or f"exit {proc.returncode}"
        return None, f"git ls-files failed ({detail})"
    paths = sorted(p for p in proc.stdout.split("\0") if p)
    if not paths:
        return None, "git ls-files returned no *.sh paths"
    return paths, ""


def _glob_fallback_paths() -> list[Path]:
    """The pre-2026-11-04 population: two non-recursive globs."""
    paths: list[Path] = []
    for root in WRAPPER_ROOTS:
        paths.extend(sorted((REPO_ROOT / root).glob("*.sh")))
    return paths


def _population() -> tuple[list[Path], dict]:
    """``(paths to scan, coverage record)``.

    The coverage record is the point of this function: it states the
    DENOMINATOR (how many tracked ``*.sh`` the repo has), how many were
    scanned, how many were excluded and why, and the resulting percentage.
    A gate that cannot report its own denominator cannot be reviewed
    (§6.269.13).
    """
    tracked, error = _git_tracked_sh_paths()
    if tracked is None:
        # Fail-closed: fall back to the OLD population and say so loudly.
        paths = [p for p in _glob_fallback_paths() if not _is_excluded_wrapper(p)]
        return paths, {
            "population_source": "glob fallback (git ls-files unavailable)",
            "population_degraded": True,
            "population_error": error,
            "n_tracked_sh_total": None,
            "n_scanned": len(paths),
            "n_excluded": None,
            "coverage_pct": None,
            "excluded": [],
        }

    scanned: list[Path] = []
    excluded: list[dict] = []
    for rel in tracked:
        path = REPO_ROOT / rel
        found = _exclusion_for(path, rel)
        if found is None:
            scanned.append(path)
        else:
            kind, reason = found
            excluded.append({"path": rel, "kind": kind, "reason": reason})

    total = len(tracked)
    n_excluded = len(excluded)
    return scanned, {
        "population_source": "git ls-files '*.sh'",
        "population_degraded": False,
        "population_error": "",
        "n_tracked_sh_total": total,
        "n_scanned": len(scanned),
        "n_excluded": n_excluded,
        "coverage_pct": round(100.0 * len(scanned) / total, 1) if total else None,
        "excluded": excluded,
    }


def _wrapper_paths() -> list[Path]:
    """Every launch wrapper the gate audits.

    MEASURED 2026-11-02 (finding C4-2): the old population was
    ``scripts/launch_*.sh`` (54 files) plus ``scripts/start.sh`` — 55 of the
    repo's 113 tracked ``.sh`` files.  The other 58 were never looked at.

    MEASURED 2026-11-04 (the COVERAGE half): the population is now EVERY
    TRACKED ``*.sh`` in the repo minus the explicit exclusion list — 113
    tracked, 13 excluded, 100 scanned (was 86, with no denominator printed).

    2026-11-04 (Wf-A269): the count moved 99 -> 100 scanned / 14 -> 13
    excluded when ``results/cfm_full_retrain/launch_loop.sh`` left the
    exclusion list, because its §5.11 migration landed (see
    :data:`EXCLUDED_WRAPPER_FILES`).
    """
    paths, _coverage = _population()
    return paths


def _legacy_wrapper_paths() -> list[Path]:
    """The population the gate scanned before 2026-11-02 (finding C4-2)."""
    paths: list[Path] = []
    for pattern in LEGACY_WRAPPER_GLOBS:
        paths.extend(sorted(SCRIPTS_DIR.glob(pattern)))
    return paths


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Return (start_lineno, code) for comment-free, continuation-joined lines.

    Two things make this more than a comment stripper, and both were caught by
    the gate's own regression suite rather than by inspection:

    * Comments are dropped, because the migrated wrappers document the flags
      they used to pass. Scanning those would pin every one of them as broken
      forever, which is how a gate gets disabled instead of fixed.
    * Backslash-continued lines are JOINED. Every launcher in this repo puts
      the trainer flags on continuation lines, so a per-physical-line scan sees
      `--config` on line 1 and a bare `--lr` on line 2 with no trainer call on
      it to anchor to — and reports PASS on a wrapper that smuggles in a
      hyperparameter. That is precisely the bug class §5.11 exists to close, so
      a scanner that misses it is worse than no scanner.
    """
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0

    def flush() -> None:
        if not buf:
            return
        joined = " ".join(buf)
        hash_at = joined.find(" #")
        if hash_at != -1:
            joined = joined[:hash_at]
        if joined.strip():
            out.append((start, joined))
        buf.clear()

    in_block = False
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw
        if in_block:
            end = line.find("*/")
            if end == -1:
                continue
            line, in_block = line[end + 2 :], False
        if not buf and (line.lstrip().startswith("#") or not line.strip()):
            continue
        stripped = line.rstrip()
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


def _audit_wrapper(path: Path) -> tuple[list[dict], bool]:
    """Return (violations, trainer_seen) for a single launch wrapper."""
    violations: list[dict] = []
    rel = _rel(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    env = _collect_assignments(text)

    for lineno, line in _logical_lines(text):
        if "--extra-flags" in line:
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "extra_flags",
                    "detail": (
                        "--extra-flags passes trainer hyperparameters outside the "
                        "YAML; declare them in the config instead (§5.11)"
                    ),
                }
            )

        via = _trainer_call(line, env)
        if via is None:
            continue

        # The flags of this invocation, resolved THROUGH bash arrays.  A
        # line-scoped regex alone cannot see them: `"${COMMON_ARGS[@]}"`
        # carries the flags in the array body, not in the invocation line
        # (MEASURED 2026-11-03: 13 wrappers / 130 flags).
        argv, unresolved, excluded = _trainer_argv(line, env)
        flags = set(_FLAG.findall(_strip_detach_head(line)))
        for tok in argv:
            flags.update(_FLAG.findall(_unquote(tok)))

        for name in sorted(set(unresolved)):
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "unresolved_expansion",
                    "via": via,
                    "detail": (
                        f"the trainer invocation expands the bash array "
                        f'"${{{name}[@]}}", and this gate could not read that '
                        "array, so it cannot prove which flags reach the "
                        "trainer; declare the values in the YAML so the "
                        "invocation is inspectable (§5.11)"
                    ),
                }
            )

        for flag in sorted(flags):
            if flag not in ALLOWED_TRAINER_FLAGS:
                violations.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "kind": "trainer_flag",
                        "via": via,
                        "detail": (
                            f"{flag} is passed on a trainer invocation line; only "
                            + " ".join(sorted(ALLOWED_TRAINER_FLAGS))
                            + " may appear there (§5.11)"
                        ),
                    }
                )
    return violations, bool(violations)


def _conditional_exclusion_census(path: Path) -> list[dict]:
    """Flags that sit in a bash array but whose append condition is FALSE.

    These are NOT violations — the launch does not pass them today — but they
    are the reason a wrapper can look compliant while still carrying a flag
    that the trainer would reject the moment the condition flips.  Reporting
    them keeps the census honest without inflating the defect count
    (MEASURED 2026-11-03 on ``scripts/launch_wf390a1_m2_r2_path_a.sh``:
    ``--chemprop-v3-cache`` is appended under a condition that is false for
    the D-MPNN trainer the wrapper runs).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    env = _collect_assignments(text)
    rows: list[dict] = []
    for lineno, line in _logical_lines(text):
        if _trainer_call(line, env) is None:
            continue
        _argv, _unresolved, excluded = _trainer_argv(line, env)
        for tok in excluded:
            rows.append(
                {
                    "file": _rel(path),
                    "line": lineno,
                    "flag": _FLAG.findall(tok),
                    "detail": (
                        "the flag is in a bash array but its append condition is "
                        "false for this wrapper, so it does NOT reach the "
                        "trainer today; it becomes live the moment the "
                        "condition flips"
                    ),
                }
            )
    return rows


def _detection_census(path: Path) -> dict[str, int]:
    """Count how each trainer call was detected.

    A guard's own census is not a defect count, but it IS a coverage
    measurement: it says how much of the trainer-call space the gate can see
    at all, which is the question the literal-regex blind spot left open.
    """
    counts = {"literal": 0, "variable": 0, "variable_unresolved_fail_closed": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    env = _collect_assignments(text)
    for _lineno, line in _logical_lines(text):
        via = _trainer_call(line, env)
        if via is None:
            continue
        if via == "literal":
            counts["literal"] += 1
        elif via == "variable-unresolved":
            counts["variable_unresolved_fail_closed"] += 1
        else:
            counts["variable"] += 1
    return counts


def _audit_start_sh(path: Path) -> list[dict]:
    violations: list[dict] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    code = _logical_lines(text)
    accepts = any(
        "--extra-flags" in line and ("EXTRA_FLAGS=" in line or '"${2}"' in line)
        for _, line in code
    )
    refuses = any(
        re.search(r"extra-flags", line) and re.search(r"refus|reject|forbid|FATAL", line, re.I)
        for _, line in code
    )
    if accepts and not refuses:
        violations.append(
            {
                "file": _rel(path),
                "line": 0,
                "kind": "start_sh_permits_extra_flags",
                "detail": (
                    "scripts/start.sh still parses --extra-flags. It is the surface "
                    "every wrapper funnels through, so the §5.11 escape hatch stays "
                    "open until start.sh refuses the flag."
                ),
            }
        )
    return violations


# ---------------------------------------------------------------------------
# start.sh's OWN trainer-command construction (2026-11-02, finding C4-3)
# ---------------------------------------------------------------------------
#: A Python string literal, single- or double-quoted.
_PY_STRING = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")

#: `legacy_script = '<path>'` — the trainer a legacy (entry-less) config runs.
_LEGACY_SCRIPT = re.compile(r"legacy_script\s*=\s*(?:'([^']*)'|\"([^\"]*)\")")

#: `'command': (` — the legacy ``entry.command`` literal block.
_COMMAND_KEY = re.compile(r"['\"]command['\"]\s*:\s*\(")

#: A flag occurrence and the value token that follows it.  A value that
#: itself starts with `--` is the NEXT flag, not this flag's value.
_FLAG_OCCURRENCE = re.compile(r"(--[a-z0-9][a-z0-9-]*)(?:\s+(?!--)([^\s]+))?")

#: A `{placeholder}` value — filled from the YAML by start.sh's format() call.
_PLACEHOLDER_VALUE = re.compile(r"^\{[A-Za-z_][A-Za-z0-9_]*\}$")

#: A python local assigned from a YAML read inside start.sh's parse_yaml block.
_YAML_READ = re.compile(
    r"_sec\(|_proto_or_train\(|\bd\.get\(|\bd\[|\bout_block\b|\b_proto\b|\b_train\b"
)


def _balanced_span(text: str, open_idx: int, opener: str = "(", closer: str = ")") -> str:
    """The text from ``open_idx`` through its matching ``closer``.

    Quote-aware: a parenthesis inside a string literal does not open a
    nesting level.  ``start.sh`` writes env lines like
    ``f'EPOCHS={... else "(trainer default)"}'``, so a naive counter sees
    brackets the source does not have.
    """
    depth = 0
    quote: str | None = None
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[open_idx : i + 1]
        i += 1
    return text[open_idx:]


def _flag_occurrences(text: str) -> list[tuple[str, str | None]]:
    """(flag, value) for every ``--flag`` in ``text``."""
    return [(m.group(1), m.group(2)) for m in _FLAG_OCCURRENCE.finditer(text)]


def _legacy_entry_commands(text: str) -> list[tuple[int, str, str]]:
    """(lineno, trainer_path, rendered_command) for each legacy entry.command.

    The legacy branch of start.sh synthesises an ``entry`` block for configs
    that carry none, and that block bakes trainer flags into a shell string.
    """
    out: list[tuple[int, str, str]] = []
    for m in _COMMAND_KEY.finditer(text):
        block = _balanced_span(text, m.end() - 1)
        parts: list[str] = []
        for sm in _PY_STRING.finditer(block):
            parts.append(sm.group(1) if sm.group(1) is not None else sm.group(2))
        command = "".join(parts)
        if not command.strip():
            continue
        scripts = _LEGACY_SCRIPT.findall(text[: m.start()])
        script = ""
        if scripts:
            script = scripts[-1][0] or scripts[-1][1]
        out.append((text.count("\n", 0, m.start()) + 1, script, command))
    return out


def _yaml_env_names(text: str) -> set[str]:
    """The env-var names start.sh's YAML->env block writes.

    That block IS the YAML serialisation (``lines = [...]`` -> the env file
    the shell sources), so every name in it is YAML-derived by construction.
    A flag whose value is one of those names is therefore yaml-sourced.
    """
    m = re.search(r"^[ \t]*lines[ \t]*=[ \t]*\[", text, re.M)
    if not m:
        return set()
    block = _balanced_span(text, text.index("[", m.start()), "[", "]")
    return set(re.findall(r"f?'([A-Z][A-Z0-9_]*)=", block))


def _yaml_local_names(text: str) -> set[str]:
    """Python locals start.sh assigns from a YAML read.

    Names shorter than four characters are dropped: they match too much by
    accident when the caller looks for a name inside a string literal.
    """
    names: set[str] = set()
    for m in re.finditer(r"^[ \t]*([a-z_][a-z0-9_]*)[ \t]*=[ \t]*(.+)$", text, re.M):
        name = m.group(1)
        if len(name) < 4:
            continue
        if _YAML_READ.search(m.group(2)):
            names.add(name)
    return names


def _audit_start_sh_command(path: Path) -> list[dict]:
    """Audit scripts/start.sh's OWN trainer-command construction (§5.11).

    MEASURED 2026-11-02 (finding C4-3): start.sh was audited ONLY for
    refusing ``--extra-flags``.  Its own command construction was never
    flag-scanned, and it bakes literal trainer flags into the shell:

    * the legacy ``entry.command`` literals carry ``--device cuda:0`` and
      ``--metals Pt`` — values no YAML chose;
    * the ``TRAINER_CMD=`` assembly and the ``_add_flag`` /
      ``_dmpnn_new_flags`` emitters are where a future literal flag would
      land, so they are scanned for flag literals that the YAML cannot
      influence.

    The rule for "is this flag yaml-sourced" is deliberately narrow: a
    ``{placeholder}`` value is filled from the same YAML by start.sh's own
    ``format()`` call, a shell variable reference is filled from the env file
    the YAML wrote, and a python local assigned from a YAML read is the same
    thing one level down.  Anything else is a literal in the shell.

    Known limit: the ``EVENTS_FLAG=`` assignments are NOT scanned here (their
    values come from ``output.events_jsonl``, which is yaml-driven).  They are
    listed so the gap is on the record rather than implied.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = _rel(path)
    violations: list[dict] = []
    env_names = _yaml_env_names(text)
    local_names = _yaml_local_names(text)

    # (i) the legacy `entry.command` string literals.
    for lineno, script, command in _legacy_entry_commands(text):
        for flag, value in _flag_occurrences(command):
            if flag in ALLOWED_TRAINER_FLAGS:
                continue
            if value and _PLACEHOLDER_VALUE.match(value):
                continue  # filled from the same yaml by start.sh's format()
            occurrence = f"{flag} {value}" if value else flag
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "start_sh_literal_trainer_flag",
                    "detail": (
                        f"start.sh's legacy entry.command bakes the literal trainer "
                        f"flag {occurrence!r} into the command line"
                        + (f" (trainer {script})" if script else "")
                        + "; only "
                        + " ".join(sorted(ALLOWED_TRAINER_FLAGS))
                        + " may appear there (§5.11)"
                    ),
                }
            )

    # (ii) the TRAINER_CMD= assembly: it may only interpolate variables.
    for m in re.finditer(r"^[ \t]*TRAINER_CMD=(.*)$", text, re.M):
        lineno = text.count("\n", 0, m.start()) + 1
        for flag, value in _flag_occurrences(m.group(1)):
            if flag in ALLOWED_TRAINER_FLAGS:
                continue
            if value and ("$" in value or "{" in value):
                continue  # a shell expansion, not a literal
            occurrence = f"{flag} {value}" if value else flag
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "start_sh_literal_trainer_flag",
                    "detail": (
                        f"the TRAINER_CMD= assembly bakes the literal trainer flag "
                        f"{occurrence!r} into the command line; the assembly may "
                        "only interpolate variables (§5.11)"
                    ),
                }
            )

    # (ii) the `_add_flag` emitters: the value must be a yaml-written env var.
    for m in re.finditer(r"^[ \t]*_add_flag[ \t]+(--[a-z0-9-]+)[ \t]+(\S+)", text, re.M):
        flag, var = m.group(1), m.group(2)
        if flag in ALLOWED_TRAINER_FLAGS:
            continue
        if var in env_names:
            continue  # the env var is written from the yaml
        lineno = text.count("\n", 0, m.start()) + 1
        violations.append(
            {
                "file": rel,
                "line": lineno,
                "kind": "start_sh_literal_trainer_flag",
                "detail": (
                    f"_add_flag emits {flag} with the value {var!r}, which is not "
                    "a variable the yaml writes; the flag's value would live only "
                    "in the shell script (§5.11)"
                ),
            }
        )

    # (ii) the `_dmpnn_new_flags` emitter: every value must come from a yaml read.
    m = re.search(r"^[ \t]*_dmpnn_new_flags[ \t]*=[ \t]*\(", text, re.M)
    if m:
        block = _balanced_span(text, text.index("(", m.start()))
        lineno = text.count("\n", 0, m.start()) + 1
        for sm in _PY_STRING.finditer(block):
            chunk = sm.group(1) if sm.group(1) is not None else sm.group(2)
            if not chunk.strip():
                continue
            sourced = any(
                re.search(r"\b" + re.escape(name) + r"\b", chunk) for name in local_names
            )
            if sourced:
                continue
            for flag, value in _flag_occurrences(chunk):
                if flag in ALLOWED_TRAINER_FLAGS:
                    continue
                occurrence = f"{flag} {value}" if value else flag
                violations.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "kind": "start_sh_literal_trainer_flag",
                        "detail": (
                            f"the _dmpnn_new_flags emitter bakes the literal trainer "
                            f"flag {occurrence!r} into the command line; the value "
                            "must come from a yaml read (§5.11)"
                        ),
                    }
                )

    return violations


def _residual_census(excluded: list[dict]) -> dict:
    """Measure the §5.11 flags carried by the EXCLUDED files.

    Excluding a file from the gate's CONTRACT must not HIDE that it carries
    unmigrated §5.11 trainer flags — the gate measures them anyway and prints
    the count, so a deferral is visible on every run instead of being a quiet
    hole.  The report deliberately never contains the word "VIOLATION": the
    pre-flight consumer sizes a failing gate by counting stdout lines that
    contain it, and a disclosed residual must not be miscounted as one.

    The two exclusion kinds are kept apart, because they mean different
    things: ``dir`` = archival (not actionable), ``file`` = migration
    deferral (actionable, and re-admitting the file flips the verdict).
    """
    rows: list[dict] = []
    total = 0
    deferred = 0
    for entry in excluded:
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            continue
        try:
            found, _ = _audit_wrapper(path)
        except OSError:  # pragma: no cover — unreadable file
            continue
        n = len(found)
        total += n
        if entry["kind"] == "file":
            deferred += n
        rows.append(
            {
                "path": entry["path"],
                "kind": entry["kind"],
                "n_unmigrated_flags": n,
            }
        )
    return {
        "n_files": len(rows),
        "n_unmigrated_flags": total,
        "n_deferred_flags": deferred,
        "files": rows,
    }


def audit(launchers: list[Path] | None = None) -> dict:
    """Audit every launch wrapper. Returns a report dict; never raises."""
    coverage: dict
    if launchers is None:
        launchers, coverage = _population()
    else:
        _default_paths, coverage = _population()
        coverage = dict(coverage)
        coverage["population_source"] = "explicit --wrapper list"
        coverage["population_degraded"] = False
        coverage["n_scanned"] = len(launchers)
        coverage["n_excluded"] = 0
        coverage["coverage_pct"] = None
        coverage["excluded"] = []

    violations: list[dict] = []
    scanned: list[str] = []
    detection = {
        "literal": 0,
        "variable": 0,
        "variable_unresolved_fail_closed": 0,
    }
    start_sh = SCRIPTS_DIR / "start.sh"
    conditionally_excluded: list[dict] = []
    for wrapper in launchers:
        if not wrapper.exists():
            continue
        rel = _rel(wrapper)
        if rel not in scanned:
            scanned.append(rel)
        if wrapper == start_sh:
            # start.sh gets its own auditors below.  Its `--extra-flags`
            # REFUSAL is a legitimate mention of the flag, and its trainer
            # command is assembled inside an embedded python heredoc, so the
            # generic wrapper scan would report both as violations.
            continue
        found, _ = _audit_wrapper(wrapper)
        violations.extend(found)
        conditionally_excluded.extend(_conditional_exclusion_census(wrapper))
        for key, value in _detection_census(wrapper).items():
            detection[key] += value

    if start_sh.exists():
        rel = _rel(start_sh)
        if rel not in scanned:
            scanned.append(rel)
        violations.extend(_audit_start_sh(start_sh))
        violations.extend(_audit_start_sh_command(start_sh))

    # How much did the 2026-11-02 widening actually buy?  A coverage change
    # with no number attached is invisible to the next reader.
    legacy_scanned = {_rel(p) for p in _legacy_wrapper_paths()} | {_rel(start_sh)}
    newly_scanned = [s for s in scanned if s not in legacy_scanned]

    # The residual: what the EXCLUDED files still carry.  Measured on every
    # run so an exclusion cannot quietly become a hole.
    residual = _residual_census(coverage.get("excluded", []))

    return {
        "gate": "yaml_only_launch",
        "rule": "AGENTS.md §5.11 — training and inference launch from YAML only",
        "allowed_trainer_flags": sorted(ALLOWED_TRAINER_FLAGS),
        "n_wrappers_scanned": len(scanned),
        "n_newly_scanned": len(newly_scanned),
        "newly_scanned": newly_scanned,
        "wrappers": scanned,
        "coverage": coverage,
        "excluded_residual": residual,
        "trainer_call_detection": detection,
        "n_conditionally_excluded": len(conditionally_excluded),
        "conditionally_excluded": conditionally_excluded,
        "n_violations": len(violations),
        "violations": violations,
        # A DEGRADED population cannot certify coverage, so it cannot PASS:
        # the gate that cannot state its denominator is the §6.269.13 class.
        "verdict": (
            "FAIL"
            if (violations or coverage.get("population_degraded"))
            else "PASS"
        ),
    }


def _print_coverage(report: dict, tag: str = "yaml-only-launch") -> None:
    """Print the gate's OWN COVERAGE — the denominator, the numerator, the
    exclusions and their reasons, and the residual those exclusions carry.

    A gate that prints "scanned N wrapper(s)" with no denominator cannot be
    reviewed: a reader cannot tell N/N from N/M (§6.269.13's class).  The
    companion launch-contract gate prints the SAME block, so the two gates
    can never disagree about what they audited.
    """
    cov = report.get("coverage") or {}
    total = cov.get("n_tracked_sh_total")
    scanned = cov.get("n_scanned")
    excluded = cov.get("n_excluded")
    pct = cov.get("coverage_pct")
    if total is None:
        print(
            f"[{tag}] COVERAGE: denominator UNKNOWN | {scanned} scanned | "
            f"{excluded} excluded | source: {cov.get('population_source')}"
        )
    else:
        print(
            f"[{tag}] COVERAGE: {total} tracked *.sh in the repo | {scanned} scanned "
            f"| {excluded} excluded | {pct}% scanned | source: "
            f"{cov.get('population_source')}"
        )
    if cov.get("population_degraded"):
        print(
            f"[{tag}] POPULATION DEGRADED: {cov.get('population_error')} — the gate "
            "cannot state its denominator, so it cannot certify coverage"
        )
    for entry in cov.get("excluded", []):
        print(f"  EXCLUDED {entry['path']} [{entry['kind']}] — {entry['reason']}")
    residual = report.get("excluded_residual") or {}
    if residual.get("n_unmigrated_flags"):
        detail = ", ".join(
            f"{row['path']}: {row['n_unmigrated_flags']}"
            for row in residual.get("files", [])
            if row["n_unmigrated_flags"]
        )
        print(
            f"[{tag}] EXCLUDED-FILE RESIDUAL (out of this gate's contract, disclosed "
            f"every run): {residual['n_unmigrated_flags']} unmigrated §5.11 trainer "
            f"flag(s) across {residual['n_files']} excluded file(s) — {detail}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument(
        "--wrapper",
        action="append",
        default=None,
        metavar="PATH",
        help="audit only this wrapper (repeatable) instead of the default "
        "scripts/launch_*.sh population; used to exercise the gate against a "
        "fixture outside the repo",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="accepted for interface parity with the other §4 gates; a violation "
        "already fails without it",
    )
    args = ap.parse_args(argv)

    launchers = None
    if args.wrapper:
        launchers = [Path(w) for w in args.wrapper]
    report = audit(launchers)

    if args.json:
        import json

        print(json.dumps(report, indent=2))
    else:
        _print_coverage(report)
        for v in report["violations"]:
            loc = f"{v['file']}:{v['line']}" if v["line"] else v["file"]
            print(f"  VIOLATION {loc} [{v['kind']}] {v['detail']}")
        if not report["violations"]:
            print(
                "  every launch wrapper passes --config only; all hyperparameters "
                "come from the YAML"
            )
        print(f"[yaml-only-launch] {report['verdict']}")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
