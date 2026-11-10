#!/usr/bin/env python3
"""wfa165_gpu_lock_guard.py — repo-wide GPU-lock-defeater guard (Wf-A165).

WHAT THIS IS
------------
A READ-ONLY, CPU-ONLY static guard that enumerates the repository's
tracked ``*.sh`` and ``*.py`` files (via ``git ls-files`` — NEVER
``os.walk``; this repo is ~201 GB) and reports every site that can
DEFEAT an ambient ``CUDA_VISIBLE_DEVICES=""`` session lock.

The session lock is "the GPU is invisible".  A *defeater* is a site that
replaces, drops, or bypasses the ambient mask so that a child process
can still see a GPU.

FOUR MECHANISMS (the detection rules, stated exactly)
------------------------------------------------------
M1 ``hard_override``
    A shell line that assigns ``CUDA_VISIBLE_DEVICES=<value>`` where
    ``<value>`` does NOT reference ``${CUDA_VISIBLE_DEVICES...}``.  Both
    the ``export``/plain-assignment form and the per-command prefix form
    (``CUDA_VISIBLE_DEVICES=0 cmd``) are covered, because both set the
    variable for the child.

M2 ``empty_default_substitution``
    A shell line using ``${CUDA_VISIBLE_DEVICES:-<default>}``.  The
    ``:-`` form substitutes ``<default>`` when the variable is EMPTY, so
    it defeats an empty ambient mask.
    The ``${CUDA_VISIBLE_DEVICES-<default>}`` form (no colon) PRESERVES
    an explicitly-empty value and is NOT a defeater.

M3 ``env_dropping_spawn``
    (a) Python that spawns a child with an ``env=`` that neither
        inherits ``os.environ`` nor mentions ``CUDA_VISIBLE_DEVICES``
        (the census's ``child_spawn`` / ``drops_lock`` classes); or
    (b) Python that sets ``os.environ["CUDA_VISIBLE_DEVICES"]`` to a
        non-empty literal; or
    (c) Python that DELETES the variable (``del os.environ[...]`` /
        ``os.environ.pop(...)``), which removes the mask entirely.

M4 ``mask_dropping_default``
    (a) Python that passes ``CUDA_VISIBLE_DEVICES=""`` into a child
        ``env=``; or
    (b) Python that sets ``os.environ["CUDA_VISIBLE_DEVICES"]`` to a
        literal device index (``"0"``, ``"0,1"``, ``0``, ...); or
    (c) Python that pins the variable via ``os.environ.setdefault(...)``
        to any literal.

OVERLAP RESOLUTION (documented, deterministic)
----------------------------------------------
M3(b) and M4(b) both match ``os.environ["CUDA_VISIBLE_DEVICES"] = "0"``.
The guard assigns each site to exactly ONE mechanism using the
*more specific* clause:

    * empty literal          -> M4  (hard-coded mask that drops the parent's value)
    * device-index literal   -> M4  (the brief's "literal index" clause)
    * other non-empty literal-> M3  (the brief's "non-empty literal" clause)
    * symbolic (name/call)   -> M3  unless the value is an empty-literal-ish
                                 expression, in which case M4
    * deletion / pop         -> M3  (drops the mask entirely)

Every finding also carries ``matched_rules``: the full list of clauses
the site matched, so nothing is hidden by the single-mechanism
assignment.  ``defeats_ambient_empty_mask`` is a tri-state
("yes" / "no" / "unknown") and answers the question that actually
matters: can this site turn an ambient ``CUDA_VISIBLE_DEVICES=""`` into
a non-empty mask?

WHAT IT MUST NOT DO
-------------------
Never import torch.  Never call a CUDA API.  Never write to the repo.
The only file this tool can write is the explicit ``--out`` path.

INTERFACE
---------
::

    python molmetal/scripts/wfa165_gpu_lock_guard.py               # human summary
    python molmetal/scripts/wfa165_gpu_lock_guard.py --json        # machine JSON
    python molmetal/scripts/wfa165_gpu_lock_guard.py --advisory    # exit 0 always
    python molmetal/scripts/wfa165_gpu_lock_guard.py --out X.json  # atomic write

EXIT CODES
----------
    0  no findings (or ``--advisory``: findings reported, exit forced to 0)
    1  findings exist  -> the guard can therefore GATE a launch
    2  usage / internal error

``--advisory`` still prints the full report and the denominator; it only
changes the exit code, so it is safe to run in a "report-only" pipeline
slot.  ``--json`` and ``--advisory`` compose: ``--json --advisory``
prints JSON and exits 0.

MEASURED-vs-PROJECTED
----------------------
Every number this tool prints is MEASURED: the denominator is the number
of tracked files enumerated and the number actually decoded and scanned;
the finding counts are the number of findings this tool produced.  There
are no estimates and no aspirational numbers anywhere in the output.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "MECHANISMS",
    "Finding",
    "scan_repository",
    "scan_shell_text",
    "scan_python_text",
    "main",
]

CVD = "CUDA_VISIBLE_DEVICES"

# --------------------------------------------------------------------------
# Mechanism registry — the single source of truth for ids and prose.
# --------------------------------------------------------------------------
MECHANISMS: Dict[str, Dict[str, str]] = {
    "M1_hard_override": {
        "title": "hard override",
        "rule": (
            "A shell line that assigns CUDA_VISIBLE_DEVICES=<value> where "
            "<value> does not reference ${CUDA_VISIBLE_DEVICES...} "
            "(export / plain assignment / per-command prefix form)."
        ),
    },
    "M2_empty_default_substitution": {
        "title": "empty-default substitution",
        "rule": (
            "A shell line using ${CUDA_VISIBLE_DEVICES:-<default>}.  The "
            "':-' form substitutes the default when the variable is EMPTY "
            "and so defeats an empty ambient mask.  The no-colon form "
            "${CUDA_VISIBLE_DEVICES-<default>} PRESERVES an explicitly-empty "
            "value and is NOT a defeater."
        ),
    },
    "M3_env_dropping_spawn": {
        "title": "env-dropping spawn",
        "rule": (
            "Python that spawns a child with an env= that neither inherits "
            "os.environ nor mentions CUDA_VISIBLE_DEVICES; or Python that "
            "sets os.environ['CUDA_VISIBLE_DEVICES'] to a non-empty "
            "literal; or Python that deletes/pops the variable."
        ),
    },
    "M4_mask_dropping_default": {
        "title": "mask-dropping default",
        "rule": (
            "Python that passes CUDA_VISIBLE_DEVICES=\"\" into a child env; "
            "or sets os.environ['CUDA_VISIBLE_DEVICES'] to a literal device "
            "index; or pins the variable via os.environ.setdefault(...)."
        ),
    },
}

MECHANISM_ORDER = ["M1_hard_override", "M2_empty_default_substitution",
                   "M3_env_dropping_spawn", "M4_mask_dropping_default"]

# --------------------------------------------------------------------------
# Shell detection
# --------------------------------------------------------------------------

# `${CUDA_VISIBLE_DEVICES:-...}` — the empty-default (defeating) form.
_EMPTY_DEFAULT_RE = re.compile(r"\$\{CUDA_VISIBLE_DEVICES:-([^}]*)\}")
# Any self reference to the variable: `${CUDA_VISIBLE_DEVICES...}` or
# `$CUDA_VISIBLE_DEVICES`.  A self reference PRESERVES the ambient value
# (the `:-` case is handled first, above).
_SELF_REF_RE = re.compile(r"\$\{?CUDA_VISIBLE_DEVICES\b")
# A device-index list: "0", "1", "0,1", "all".
_INDEX_LITERAL_RE = re.compile(r"^(?:all|\d+(?:,\d+)*)$")

_SHELL_MODS_RE = re.compile(
    r"^(?P<mods>(?:(?:export|local|readonly|typeset)\s+|declare\s+-\w+\s+)*)"
    r"CUDA_VISIBLE_DEVICES\s*=\s*(?P<rest>.*)$"
)
_SHELL_ENV_CMD_RE = re.compile(r"^env\s+CUDA_VISIBLE_DEVICES\s*=\s*(?P<rest>.*)$")
_SHELL_KEYWORD_RE = re.compile(r"^(?:if|elif|while|until|then|do|else|!|time)\s+")


def _strip_shell_keywords(text: str) -> str:
    """Remove leading shell keywords so ``if VAR=x cmd`` parses like ``VAR=x cmd``."""
    out = text.lstrip()
    while True:
        m = _SHELL_KEYWORD_RE.match(out)
        if not m:
            return out
        out = out[m.end():].lstrip()


def _shell_words(text: str) -> List[str]:
    """Split a shell fragment into words, honouring ' and " quoting."""
    words: List[str] = []
    cur: List[str] = []
    i = 0
    n = len(text)
    in_quote = ""
    while i < n:
        ch = text[i]
        if in_quote:
            cur.append(ch)
            i += 1
            if ch == in_quote:
                in_quote = ""
            continue
        if ch in "'\"":
            in_quote = ch
            cur.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            cur.append(ch)
            cur.append(text[i + 1])
            i += 2
            continue
        if ch.isspace():
            if cur:
                words.append("".join(cur))
                cur = []
            i += 1
            continue
        if ch == "#" and not cur:
            break  # comment to end of line
        cur.append(ch)
        i += 1
    if cur:
        words.append("".join(cur))
    return words


def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return v


def _value_kind(value: str) -> str:
    """literal | dynamic | symbolic — how the assigned value is produced."""
    if value is None:
        return "none"
    v = value.strip()
    if v == "":
        return "literal"
    if "$" in v or "`" in v:
        return "dynamic"
    return "literal"


def _classify_shell_value(value: Optional[str]) -> Dict[str, Any]:
    """Classify an assigned shell value against the M1 / M2 rules.

    Returns a dict with ``mechanism`` (None when the value PRESERVES the
    ambient mask), ``matched_rules``, ``effect`` and
    ``defeats_ambient_empty_mask``.
    """
    raw = "" if value is None else value.strip()
    unq = _unquote(raw)

    m_empty = _EMPTY_DEFAULT_RE.search(raw)
    if m_empty:
        default = m_empty.group(1).strip()
        default_empty = default == "" or _unquote(default) == ""
        return {
            "mechanism": "M2_empty_default_substitution",
            "matched_rules": ["M2_empty_default_substitution"],
            "default_value": default,
            "default_is_empty": default_empty,
            "value_kind": "dynamic",
            "effect": "neutral" if default_empty else "unmask",
            "defeats_ambient_empty_mask": "no" if default_empty else "yes",
        }

    if _SELF_REF_RE.search(raw):
        # Self reference — e.g. "${CUDA_VISIBLE_DEVICES}" or
        # "${CUDA_VISIBLE_DEVICES-0}" (no colon).  PRESERVES the ambient
        # value, including an explicitly-empty one.
        return {
            "mechanism": None,
            "matched_rules": [],
            "default_value": None,
            "default_is_empty": None,
            "value_kind": "dynamic",
            "effect": "preserves_ambient",
            "defeats_ambient_empty_mask": "no",
        }

    kind = _value_kind(unq)
    if kind == "literal" and unq == "":
        return {
            "mechanism": "M1_hard_override",
            "matched_rules": ["M1_hard_override"],
            "default_value": None,
            "default_is_empty": None,
            "value_kind": "literal",
            "effect": "forces_blank",
            "defeats_ambient_empty_mask": "no",
        }
    return {
        "mechanism": "M1_hard_override",
        "matched_rules": ["M1_hard_override"],
        "default_value": None,
        "default_is_empty": None,
        "value_kind": kind,
        "effect": "unmask",
        "defeats_ambient_empty_mask": "yes",
    }


def scan_shell_text(text: str) -> List[Dict[str, Any]]:
    """Scan one shell file's text for M1 / M2 sites.  Pure function."""
    findings: List[Dict[str, Any]] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\n")
        stripped = _strip_shell_keywords(line)
        m = _SHELL_MODS_RE.match(stripped)
        env_cmd = False
        if not m:
            m2 = _SHELL_ENV_CMD_RE.match(stripped)
            if m2:
                m = m2
                env_cmd = True
        if not m:
            continue
        mods = (m.group("mods") or "").strip()
        rest = m.group("rest")
        words = _shell_words(rest)
        value = words[0] if words else ""
        if not mods and not env_cmd:
            # plain assignment vs. per-command prefix assignment
            if len(words) > 1:
                klass = "shell_prefix_assignment"
            else:
                klass = "shell_assignment"
        elif env_cmd:
            klass = "shell_prefix_assignment"
        else:
            klass = "shell_export"
        verdict = _classify_shell_value(value)
        if verdict["mechanism"] is None:
            continue
        findings.append(
            {
                "mechanism": verdict["mechanism"],
                "matched_rules": verdict["matched_rules"],
                "class": klass,
                "line": lineno,
                "source": line.strip(),
                "value": value,
                "value_kind": verdict["value_kind"],
                "effect": verdict["effect"],
                "defeats_ambient_empty_mask": verdict["defeats_ambient_empty_mask"],
                "default_value": verdict["default_value"],
                "default_is_empty": verdict["default_is_empty"],
                "lang": "shell",
            }
        )
    return findings


# --------------------------------------------------------------------------
# Python detection (AST)
# --------------------------------------------------------------------------

_SUBPROCESS_SPAWN_ATTRS = {
    "run", "call", "check_call", "check_output", "Popen",
}
_OS_SPAWN_ATTRS = {
    "system", "popen", "execv", "execve", "execvp", "execvpe", "execl", "execlp",
    "execle", "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle",
    "spawnlp", "spawnlpe", "posix_spawn", "posix_spawnp",
}
_SUBPROCESS_MODULE_NAMES = {"subprocess"}
_OS_MODULE_NAMES = {"os"}

_SHELL_STRING_ASSIGN_RE = re.compile(
    r"(?<![A-Za-z0-9_])CUDA_VISIBLE_DEVICES\s*=\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s;|&]+)"
)


def _imported_names(tree: ast.AST) -> Tuple[set, set, set]:
    """Return (subprocess module aliases, os module aliases, direct spawn names)."""
    sub_mods = set(_SUBPROCESS_MODULE_NAMES)
    os_mods = set(_OS_MODULE_NAMES)
    direct: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    sub_mods.add(alias.asname or "subprocess")
                elif alias.name == "os":
                    os_mods.add(alias.asname or "os")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "subprocess":
                for alias in node.names:
                    if alias.name in _SUBPROCESS_SPAWN_ATTRS:
                        direct.add(alias.asname or alias.name)
            elif node.module == "os":
                for alias in node.names:
                    if alias.name in _OS_SPAWN_ATTRS:
                        direct.add(alias.asname or alias.name)
    return sub_mods, os_mods, direct


def _root_name(node: ast.AST) -> Optional[str]:
    """Return the root Name of an attribute chain (e.g. os.environ -> 'os')."""
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def _attr_name(node: ast.AST) -> Optional[str]:
    return node.attr if isinstance(node, ast.Attribute) else None


def _mentions_environ(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr == "environ":
            return True
        if isinstance(sub, ast.Name) and sub.id == "environ":
            return True
    return False


def _mentions_cvd(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if sub.value == CVD:
                return True
        elif isinstance(sub, ast.Name) and sub.id == CVD:
            return True
    return False


def _dict_cvd_value(node: ast.AST) -> Tuple[Optional[ast.AST], bool]:
    """Return (value node for the CVD key, key_found) inside a Dict literal.

    ``**``-unpacked inner dicts are searched too (``{**a, **{"CUDA_...": x}}``).
    """
    if not isinstance(node, ast.Dict):
        return None, False
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value in (CVD, CVD.encode()):
            return value, True
        if key is None:  # **unpacked
            inner, found = _dict_cvd_value(value)
            if found:
                return inner, True
    return None, False


# --------------------------------------------------------------------------
# Scope-aware resolution of the `env=` expression
# --------------------------------------------------------------------------

# Parameter names that conventionally carry a child environment.
ENV_PARAM_NAMES = {"env", "environ", "child_env", "env_extra", "env_vars",
                   "extra_env", "sub_env", "run_env", "proc_env"}


def _scope_statements(scope: ast.AST) -> List[ast.AST]:
    """Statements belonging to this scope, excluding nested function bodies."""
    out: List[ast.AST] = []

    def rec(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            out.append(child)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # the def itself belongs here; its body does not
            rec(child)

    rec(scope)
    return out


def _is_cvd_key(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (CVD, CVD.encode())


def _param_facts(func: ast.AST) -> Dict[str, Dict[str, Any]]:
    """Facts for a function's own parameters (an `env` parameter defaults to None)."""
    out: Dict[str, Dict[str, Any]] = {}

    def record(arg_name: str, default: Optional[ast.AST]) -> None:
        if arg_name not in ENV_PARAM_NAMES:
            return
        is_none = default is None or (
            isinstance(default, ast.Constant) and default.value is None
        )
        out[arg_name] = {
            "assign": None if is_none else default,
            "cvd": None,
            "cvd_origin": None,
            "kind": "param_none" if is_none else "param_dict",
        }

    pos = list(getattr(func.args, "posonlyargs", []) or []) + list(func.args.args)
    pos_defaults = list(func.args.defaults)
    offset = len(pos) - len(pos_defaults)
    for idx, a in enumerate(pos):
        record(a.arg, pos_defaults[idx - offset] if idx >= offset else None)
    for a, d in zip(func.args.kwonlyargs, func.args.kw_defaults):
        record(a.arg, d)
    return out


def _build_scope_facts(stmts: Sequence[ast.AST]) -> Dict[str, Dict[str, Any]]:
    """Per-scope facts about names that may carry a child environment.

    facts[name] = {
        "assign":     the AST node the name was assigned (any expression), or None
        "cvd":        the AST node assigned to name["CUDA_VISIBLE_DEVICES"], or None
        "cvd_origin": "dict_literal" | "subscript_assign" | "deleted" | None
        "kind":       "assign" | "param_none" | "param_dict"
    }
    """
    facts: Dict[str, Dict[str, Any]] = {}
    for st in stmts:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for name, fact in _param_facts(st).items():
                facts.setdefault(name, fact)
        elif isinstance(st, ast.Assign):
            for target in st.targets:
                if isinstance(target, ast.Name):
                    f = facts.setdefault(target.id, {
                        "assign": None, "cvd": None, "cvd_origin": None,
                        "kind": "assign",
                    })
                    f["assign"] = st.value
                    f["kind"] = "assign"
                elif (isinstance(target, ast.Subscript)
                      and isinstance(target.value, ast.Name)
                      and _is_cvd_key(target.slice)):
                    f = facts.setdefault(target.value.id, {
                        "assign": None, "cvd": None, "cvd_origin": None,
                        "kind": "assign",
                    })
                    f["cvd"] = st.value
                    f["cvd_origin"] = "subscript_assign"
        elif isinstance(st, ast.AnnAssign):
            if isinstance(st.target, ast.Name) and st.value is not None:
                f = facts.setdefault(st.target.id, {
                    "assign": None, "cvd": None, "cvd_origin": None,
                    "kind": "assign",
                })
                f["assign"] = st.value
                f["kind"] = "assign"
        elif isinstance(st, ast.Delete):
            for target in st.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and _is_cvd_key(target.slice)):
                    f = facts.setdefault(target.value.id, {
                        "assign": None, "cvd": None, "cvd_origin": None,
                        "kind": "assign",
                    })
                    f["cvd"] = None
                    f["cvd_origin"] = "deleted"
    return facts


def _scope_facts(tree: ast.AST) -> List[Tuple[Tuple[int, int], Dict[str, Dict[str, Any]]]]:
    """[(lineno-range, facts)] for the module scope and every function scope."""
    scopes: List[Tuple[Tuple[int, int], Dict[str, Dict[str, Any]]]] = [
        ((0, 10 ** 9), _build_scope_facts(_scope_statements(tree)))
    ]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stmts = _scope_statements(node)
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", None) or (10 ** 9)
            facts = _build_scope_facts(stmts)
            for name, fact in _param_facts(node).items():
                facts.setdefault(name, fact)
            scopes.append(((start, end), facts))
    return scopes


def _lookup_fact(name: str, lineno: int,
                 scopes: List[Tuple[Tuple[int, int], Dict[str, Dict[str, Any]]]]
                 ) -> Optional[Dict[str, Any]]:
    """Innermost scope containing `lineno` that binds `name` wins."""
    best: Optional[Dict[str, Any]] = None
    best_span = 10 ** 9
    for (start, end), facts in scopes:
        if start <= lineno <= end and name in facts:
            span = end - start
            if span < best_span:
                best_span = span
                best = facts[name]
    return best


def _analyse_env_expr(
    expr: ast.AST,
    scopes: List[Tuple[Tuple[int, int], Dict[str, Dict[str, Any]]]],
) -> Dict[str, Any]:
    """Describe an `env=` expression: inheritance, CVD presence, resolution.

    Returns {"inherits": bool|None, "cvd_node": node|None, "resolution": str,
             "mentions": bool}.  ``inherits`` is None when the expression
    cannot be resolved statically (the caller must not report it).
    """
    inherits: Optional[bool] = None
    cvd_node: Optional[ast.AST] = None
    resolution = "literal"
    mentions = False

    if isinstance(expr, ast.Name):
        fact = _lookup_fact(expr.id, getattr(expr, "lineno", 0), scopes)
        if fact is None:
            return {"inherits": None, "cvd_node": None, "mentions": False,
                    "resolution": "unresolved_name"}
        assign = fact.get("assign")
        if fact.get("kind") == "param_none":
            inherits = True
            resolution = "conditional_param"
        elif assign is not None:
            resolution = "resolved_assign"
            if isinstance(assign, ast.Dict):
                # A dict literal is definitive: does it inherit os.environ?
                inherits = _mentions_environ(assign)
                node, found = _dict_cvd_value(assign)
                if found:
                    cvd_node = node
            elif _mentions_environ(assign):
                # e.g. `env = {k: v for k, v in os.environ.items() ...}` or
                # `env = dict(os.environ)` / `env = os.environ.copy()`
                inherits = True
            else:
                # Opaque: the guard must NOT claim the child loses os.environ.
                inherits = None
                resolution = "opaque_assign"
        else:
            inherits = None
            resolution = "unresolved_name"
        if fact.get("cvd_origin") == "subscript_assign":
            cvd_node = fact.get("cvd")
            resolution = "resolved_subscript"
        elif fact.get("cvd_origin") == "deleted":
            cvd_node = None
            resolution = "resolved_deleted"
    elif isinstance(expr, ast.Dict):
        inherits = _mentions_environ(expr)
        node, found = _dict_cvd_value(expr)
        if found:
            cvd_node = node
    elif isinstance(expr, ast.Call):
        if expr.args and isinstance(expr.args[0], ast.Dict):
            inherits = _mentions_environ(expr.args[0])
            node, found = _dict_cvd_value(expr.args[0])
            if found:
                cvd_node = node
        elif _mentions_environ(expr):
            inherits = True
        else:
            inherits = None
            resolution = "opaque_call"
    elif isinstance(expr, ast.DictComp):
        inherits = _mentions_environ(expr)
    elif isinstance(expr, (ast.Attribute, ast.Subscript, ast.IfExp)):
        inherits = None
        resolution = "opaque_expression"
    else:
        inherits = None
        resolution = "opaque_expression"

    if cvd_node is not None:
        mentions = True
    elif _mentions_cvd(expr):
        mentions = True
        resolution = "mentions_but_unresolved"
    return {"inherits": inherits, "cvd_node": cvd_node, "mentions": mentions,
            "resolution": resolution}



def _literal_value_kind(node: ast.AST) -> Tuple[str, Optional[str]]:
    """Classify an assigned value node: (kind, static_text).

    kind in {literal, symbolic}; static_text is the literal string when the
    value is a constant (else None).
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return "literal", node.value
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return "literal", str(node.value)
        return "literal", ""
    return "symbolic", None


def _python_value_verdict(node: ast.AST) -> Dict[str, Any]:
    """Classify a Python assignment value for the M3 / M4 rules."""
    kind, static = _literal_value_kind(node)
    if kind == "literal":
        text = static or ""
        if text == "":
            # hard-coded EMPTY mask -> pins the mask, drops the parent's value
            return {
                "mechanism": "M4_mask_dropping_default",
                "matched_rules": ["M4_mask_dropping_default"],
                "value_kind": "literal",
                "static_value": text,
                "effect": "forces_blank",
                "defeats_ambient_empty_mask": "no",
            }
        if _INDEX_LITERAL_RE.match(text.strip()):
            return {
                "mechanism": "M4_mask_dropping_default",
                "matched_rules": ["M4_mask_dropping_default", "M3_env_dropping_spawn"],
                "value_kind": "literal",
                "static_value": text,
                "effect": "unmask",
                "defeats_ambient_empty_mask": "yes",
            }
        return {
            "mechanism": "M3_env_dropping_spawn",
            "matched_rules": ["M3_env_dropping_spawn"],
            "value_kind": "literal",
            "static_value": text,
            "effect": "unmask",
            "defeats_ambient_empty_mask": "yes",
        }
    return {
        "mechanism": "M3_env_dropping_spawn",
        "matched_rules": ["M3_env_dropping_spawn"],
        "value_kind": "symbolic",
        "static_value": None,
        "effect": "unknown",
        "defeats_ambient_empty_mask": "unknown",
    }


def _is_environ_subscript(target: ast.AST) -> bool:
    if not isinstance(target, ast.Subscript):
        return False
    if _attr_name(target.value) != "environ":
        return False
    sl = target.slice
    if isinstance(sl, ast.Constant) and sl.value in (CVD, CVD.encode()):
        return True
    return False


def scan_python_text(text: str,
                     unresolved: Optional[List[Dict[str, Any]]] = None,
                     parse_failures: Optional[List[str]] = None,
                     ) -> List[Dict[str, Any]]:
    """Scan one python file's text for M3 / M4 sites.  Pure function.

    ``unresolved`` is an optional out-list: sites whose `env=` expression
    cannot be resolved statically (an unbound name, an opaque expression)
    are NOT reported as findings but are recorded there, because a guard
    that guesses is worse than a guard that says "I could not tell".
    """
    findings: List[Dict[str, Any]] = []
    if unresolved is None:
        unresolved = []
    try:
        with _quiet_parse():
            tree = ast.parse(text)
    except SyntaxError:
        if parse_failures is not None:
            parse_failures.append("<file>")
        return _scan_python_fallback(text)

    sub_mods, os_mods, direct = _imported_names(tree)
    scopes = _scope_facts(tree)

    def add(mechanism: str, matched: Sequence[str], klass: str, node: ast.AST,
            value_text: Optional[str], value_kind: str, effect: str,
            defeats: str, extra: Optional[Dict[str, Any]] = None) -> None:
        rec = {
            "mechanism": mechanism,
            "matched_rules": list(matched),
            "class": klass,
            "line": getattr(node, "lineno", 0),
            "source": "",  # filled in below from the real source line
            "value": value_text,
            "value_kind": value_kind,
            "effect": effect,
            "defeats_ambient_empty_mask": defeats,
            "default_value": None,
            "default_is_empty": None,
            "lang": "python",
        }
        if extra:
            rec.update(extra)
        findings.append(rec)

    # -- os.environ mutations ------------------------------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_environ_subscript(target):
                    verdict = _python_value_verdict(node.value)
                    add(
                        verdict["mechanism"], verdict["matched_rules"],
                        "python_os_environ_assign", node, verdict["static_value"],
                        verdict["value_kind"], verdict["effect"],
                        verdict["defeats_ambient_empty_mask"],
                        {"_src_node": node},
                    )
                elif (
                    isinstance(target, ast.Attribute)
                    and target.attr == "update"
                    and _attr_name(target.value) == "environ"
                ):
                    cvd_node, found = _dict_cvd_value(node.value)
                    if found and cvd_node is not None:
                        verdict = _python_value_verdict(cvd_node)
                        add(
                            verdict["mechanism"], verdict["matched_rules"],
                            "python_os_environ_update", node,
                            verdict["static_value"], verdict["value_kind"],
                            verdict["effect"], verdict["defeats_ambient_empty_mask"],
                            {"_src_node": node},
                        )
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if _is_environ_subscript(target):
                    add(
                        "M3_env_dropping_spawn", ["M3_env_dropping_spawn"],
                        "python_os_environ_del", node, None, "none", "unmask",
                        "yes", {"_src_node": node},
                    )
        elif isinstance(node, ast.Call):
            func = node.func
            fname = _attr_name(func)
            if (
                fname == "setdefault"
                and isinstance(func, ast.Attribute)
                and _attr_name(func.value) == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == CVD
            ):
                value_node = node.args[1] if len(node.args) > 1 else None
                if value_node is None:
                    add("M4_mask_dropping_default",
                        ["M4_mask_dropping_default"],
                        "python_os_environ_setdefault", node, None, "symbolic",
                        "unknown", "no", {"_src_node": node})
                else:
                    verdict = _python_value_verdict(value_node)
                    matched = ["M4_mask_dropping_default"]
                    if verdict["static_value"]:
                        matched.append("M3_env_dropping_spawn")
                    add("M4_mask_dropping_default", matched,
                        "python_os_environ_setdefault", node,
                        verdict["static_value"], verdict["value_kind"],
                        "forces_blank" if verdict["effect"] == "forces_blank"
                        else "unmask_when_unset",
                        "no", {"_src_node": node})
            if (
                fname == "pop"
                and isinstance(func, ast.Attribute)
                and _attr_name(func.value) == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == CVD
            ):
                add("M3_env_dropping_spawn", ["M3_env_dropping_spawn"],
                    "python_os_environ_pop", node, None, "none", "unmask", "yes",
                    {"_src_node": node})

    # -- subprocess / os spawns ----------------------------------------------
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        fname = _attr_name(func)
        root = _root_name(func)
        is_spawn = False
        if fname in _SUBPROCESS_SPAWN_ATTRS and (
            (root in sub_mods) or (isinstance(func, ast.Name) and func.id in direct)
        ):
            is_spawn = True
        elif fname in _OS_SPAWN_ATTRS and root in os_mods:
            is_spawn = True
        if not is_spawn:
            continue

        # (a) env= that neither inherits os.environ nor mentions CVD
        for kw in node.keywords:
            if kw.arg != "env":
                continue
            env_expr = kw.value
            info = _analyse_env_expr(env_expr, scopes)
            inherits = info["inherits"]
            mentions = info["mentions"]
            cvd_node = info["cvd_node"]
            if inherits is None:
                unresolved.append({
                    "line": getattr(node, "lineno", 0),
                    "resolution": info["resolution"],
                    "source": "",
                })
                continue
            if not mentions and not inherits:
                add("M3_env_dropping_spawn", ["M3_env_dropping_spawn"],
                    "python_env_no_inherit", node, None, "none", "unmask", "yes",
                    {"_src_node": node,
                     "resolution": info["resolution"],
                     "note": "env= neither inherits os.environ nor mentions "
                             "CUDA_VISIBLE_DEVICES"})
            elif mentions:
                if cvd_node is None:
                    add("M4_mask_dropping_default",
                        ["M4_mask_dropping_default"],
                        "python_env_cvd_unresolved", node, None, "symbolic",
                        "unknown", "unknown",
                        {"_src_node": node, "resolution": info["resolution"]})
                else:
                    verdict = _python_value_verdict(cvd_node)
                    add(verdict["mechanism"], verdict["matched_rules"],
                        "python_env_cvd_literal" if verdict["value_kind"] == "literal"
                        else "python_env_cvd_symbolic",
                        node, verdict["static_value"], verdict["value_kind"],
                        verdict["effect"], verdict["defeats_ambient_empty_mask"],
                        {"_src_node": node, "resolution": info["resolution"]})

        # (b) shell command strings that carry their own CVD assignment
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                continue
            for m in _SHELL_STRING_ASSIGN_RE.finditer(arg.value):
                verdict = _classify_shell_value(m.group("value"))
                if verdict["mechanism"] is None:
                    continue
                add(verdict["mechanism"], verdict["matched_rules"],
                    "python_shell_command_string", node, m.group("value"),
                    verdict["value_kind"], verdict["effect"],
                    verdict["defeats_ambient_empty_mask"],
                    {"_src_node": node,
                     "note": "shell command string carries its own "
                             "CUDA_VISIBLE_DEVICES assignment"})

    # resolve the real source line for every finding
    lines = text.splitlines()
    for rec in unresolved:
        ln = rec.get("line") or 0
        rec["source"] = lines[ln - 1].strip() if 0 < ln <= len(lines) else ""
    for rec in findings:
        node = rec.pop("_src_node", None)
        lineno = rec.get("line") or 0
        if node is not None:
            lineno = getattr(node, "lineno", lineno)
        rec["line"] = lineno
        if 0 < lineno <= len(lines):
            rec["source"] = lines[lineno - 1].strip()
        else:
            rec["source"] = ""
    return findings


_PY_FALLBACK_ASSIGN_RE = re.compile(
    r"""os\.environ\[\s*["']CUDA_VISIBLE_DEVICES["']\s*\]\s*=\s*(?P<val>[^\n]*)"""
)


def _scan_python_fallback(text: str) -> List[Dict[str, Any]]:
    """Textual fallback for python files that do not parse."""
    findings: List[Dict[str, Any]] = []
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        m = _PY_FALLBACK_ASSIGN_RE.search(line)
        if not m:
            continue
        raw = m.group("val").strip()
        if raw.startswith(("'", '"')):
            quote = raw[0]
            end = raw.find(quote, 1)
            static = raw[1:end] if end > 0 else ""
        else:
            static = None
        if static is None:
            verdict = {"mechanism": "M3_env_dropping_spawn",
                       "matched_rules": ["M3_env_dropping_spawn"],
                       "value_kind": "symbolic", "static_value": None,
                       "effect": "unknown", "defeats_ambient_empty_mask": "unknown"}
        elif static == "":
            verdict = {"mechanism": "M4_mask_dropping_default",
                       "matched_rules": ["M4_mask_dropping_default"],
                       "value_kind": "literal", "static_value": static,
                       "effect": "forces_blank", "defeats_ambient_empty_mask": "no"}
        elif _INDEX_LITERAL_RE.match(static.strip()):
            verdict = {"mechanism": "M4_mask_dropping_default",
                       "matched_rules": ["M4_mask_dropping_default",
                                         "M3_env_dropping_spawn"],
                       "value_kind": "literal", "static_value": static,
                       "effect": "unmask", "defeats_ambient_empty_mask": "yes"}
        else:
            verdict = {"mechanism": "M3_env_dropping_spawn",
                       "matched_rules": ["M3_env_dropping_spawn"],
                       "value_kind": "literal", "static_value": static,
                       "effect": "unmask", "defeats_ambient_empty_mask": "yes"}
        findings.append({
            "mechanism": verdict["mechanism"],
            "matched_rules": verdict["matched_rules"],
            "class": "python_os_environ_assign",
            "line": lineno,
            "source": line.strip(),
            "value": verdict["static_value"],
            "value_kind": verdict["value_kind"],
            "effect": verdict["effect"],
            "defeats_ambient_empty_mask": verdict["defeats_ambient_empty_mask"],
            "default_value": None,
            "default_is_empty": None,
            "lang": "python",
            "note": "textual fallback (file did not parse as python)",
        })
    return findings


# --------------------------------------------------------------------------
# Repository scan
# --------------------------------------------------------------------------

Finding = Dict[str, Any]


@contextmanager
def _quiet_parse():
    """Parse without the SyntaxWarning noise that docstring-heavy files emit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _git_head(repo_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True,
        )
        return proc.stdout.decode("utf-8", errors="replace").strip()
    except Exception:  # pragma: no cover — provenance is best-effort
        return None


def _git_ls_files(repo_root: Path) -> List[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", "*.py", "*.sh"],
        check=True,
        capture_output=True,
    )
    raw = proc.stdout.decode("utf-8", errors="replace")
    return [p for p in raw.split("\0") if p]


def _decode(path: Path) -> Tuple[str, bool]:
    """Decode a file; returns (text, lossy)."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8"), False
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), True


def scan_repository(repo_root: Path) -> Dict[str, Any]:
    """Scan the repo.  Returns the full report dict (MEASURED numbers only)."""
    repo_root = repo_root.resolve()
    tracked_all = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True, capture_output=True,
    ).stdout.decode("utf-8", errors="replace").split("\0")
    tracked_all = [p for p in tracked_all if p]

    paths = _git_ls_files(repo_root)
    findings: List[Finding] = []
    files_read = 0
    files_unreadable: List[str] = []
    files_lossy: List[str] = []
    parse_failures: List[str] = []
    unresolved_sites: List[Dict[str, Any]] = []
    by_ext = {"py": 0, "sh": 0}
    unresolved_sites_by_file: Dict[str, List[Dict[str, Any]]] = {}
    by_top: Dict[str, int] = {}
    files_with_findings: Dict[str, int] = {}

    for rel in paths:
        by_ext["py" if rel.endswith(".py") else "sh"] += 1
        top = rel.split("/", 1)[0]
        by_top[top] = by_top.get(top, 0) + 1
        full = repo_root / rel
        try:
            text, lossy = _decode(full)
        except OSError as exc:
            files_unreadable.append(f"{rel}: {exc}")
            continue
        files_read += 1
        if lossy:
            files_lossy.append(rel)
        if rel.endswith(".sh"):
            found = scan_shell_text(text)
            new_unresolved: List[Dict[str, Any]] = []
        else:
            before = len(unresolved_sites)
            file_parse_failures: List[str] = []
            found = scan_python_text(text, unresolved=unresolved_sites,
                                     parse_failures=file_parse_failures)
            new_unresolved = unresolved_sites[before:]
            if file_parse_failures:
                parse_failures.append(rel)
        for rec in found:
            rec = dict(rec)
            rec["file"] = rel
            findings.append(rec)
            files_with_findings[rel] = files_with_findings.get(rel, 0) + 1
        for rec in new_unresolved:
            rec["file"] = rel
            unresolved_sites_by_file.setdefault(rel, []).append(rec)

    findings.sort(key=lambda r: (r["file"], r["line"], r["mechanism"]))

    by_mechanism: Dict[str, List[Finding]] = {m: [] for m in MECHANISM_ORDER}
    for rec in findings:
        by_mechanism.setdefault(rec["mechanism"], []).append(rec)

    per_file: Dict[str, Dict[str, Any]] = {}
    for rec in findings:
        entry = per_file.setdefault(
            rec["file"], {"count": 0, "mechanisms": {}, "sites": []}
        )
        entry["count"] += 1
        entry["mechanisms"][rec["mechanism"]] = entry["mechanisms"].get(rec["mechanism"], 0) + 1
        entry["sites"].append(
            {"line": rec["line"], "mechanism": rec["mechanism"], "class": rec["class"]}
        )

    counts_match_lists = True  # recomputed below against the rendered report

    report = {
        "tool": "wfa165_gpu_lock_guard",
        "wf": "Wf-A165",
        "title": "repo-wide GPU lock-defeater guard (static, read-only, CPU-only)",
        "generated": _utc_now(),
        "git_head": _git_head(repo_root),
        "repo_root": str(repo_root),
        "denominator": {
            "tracked_files_total": len(tracked_all),
            "files_enumerated": len(paths),
            "files_read": files_read,
            "files_unreadable": len(files_unreadable),
            "files_decoded_lossy": len(files_lossy),
            "py_files": by_ext["py"],
            "sh_files": by_ext["sh"],
            "files_with_findings": len(files_with_findings),
            "by_top_level": dict(sorted(by_top.items())),
            "by_scope_prefix": _scope_breakdown(paths, findings),
            "parse_failures": sorted(parse_failures),
            "unreadable_paths": files_unreadable,
            "note": (
                "files_enumerated = tracked *.py + *.sh returned by "
                "`git ls-files -z -- *.py *.sh`; files_read = files actually "
                "decoded and scanned.  No os.walk was used anywhere."
            ),
        },
        "mechanisms": {
            m: {
                "title": MECHANISMS[m]["title"],
                "rule": MECHANISMS[m]["rule"],
                "count": len(by_mechanism.get(m, [])),
                "defeats_ambient_empty_mask": sum(
                    1 for r in by_mechanism.get(m, [])
                    if r.get("defeats_ambient_empty_mask") == "yes"
                ),
                "effect_breakdown": _effect_breakdown(by_mechanism.get(m, [])),
            }
            for m in MECHANISM_ORDER
        },
        "unresolved_env_sites": len(unresolved_sites),
        "unresolved_env_sites_by_file": {
            k: len(v) for k, v in sorted(unresolved_sites_by_file.items())
        },
        "unresolved_note": (
            "env= expressions whose child environment cannot be resolved "
            "statically (unbound name / opaque expression).  These are NOT "
            "counted as findings; they are reported separately because a "
            "static guard must not guess."
        ),
        "findings_total": len(findings),
        "findings_defeat_ambient_empty_mask": sum(
            1 for r in findings if r.get("defeats_ambient_empty_mask") == "yes"
        ),
        "counts_match_lists": counts_match_lists,
        "findings_by_mechanism": by_mechanism,
        "findings": findings,
        "per_file": dict(sorted(per_file.items())),
    }

    # VACUITY GUARD (self-check): every mechanism's reported count must equal
    # the number of findings it lists, and the per-mechanism counts must sum
    # to the total.  A guard that reports a count but lists fewer is broken.
    listed_total = sum(
        len(report["findings_by_mechanism"].get(m, [])) for m in MECHANISM_ORDER
    )
    report["counts_match_lists"] = bool(
        listed_total == report["findings_total"]
        and all(
            report["mechanisms"][m]["count"]
            == len(report["findings_by_mechanism"].get(m, []))
            for m in MECHANISM_ORDER
        )
        and len(report["findings"]) == report["findings_total"]
    )
    report["self_check"] = {
        "listed_total": listed_total,
        "findings_total": report["findings_total"],
        "counts_match_lists": report["counts_match_lists"],
    }
    return report


# The scope prefixes the Wf-A163 GPU-tooling census used.  Reporting the
# same buckets makes the two censuses directly comparable.
SCOPE_PREFIXES = ("molmetal/", "scripts/", "tests/")


def _scope_breakdown(paths: Sequence[str], findings: Sequence[Finding]) -> Dict[str, Any]:
    """Files enumerated/read and findings per Wf-A163 scope prefix."""
    by_prefix: Dict[str, Dict[str, Any]] = {}
    for prefix in SCOPE_PREFIXES:
        by_prefix[prefix] = {"files_enumerated": 0, "files_with_findings": 0,
                              "findings": 0}
    by_prefix["<other>"] = {"files_enumerated": 0, "files_with_findings": 0,
                            "findings": 0}
    files_with = {rec["file"] for rec in findings}
    for rel in paths:
        key = next((p for p in SCOPE_PREFIXES if rel.startswith(p)), "<other>")
        by_prefix[key]["files_enumerated"] += 1
        if rel in files_with:
            by_prefix[key]["files_with_findings"] += 1
    for rec in findings:
        key = next((p for p in SCOPE_PREFIXES if rec["file"].startswith(p)), "<other>")
        by_prefix[key]["findings"] += 1
    return by_prefix


def _effect_breakdown(findings: Sequence[Finding]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rec in findings:
        key = str(rec.get("effect"))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_human(report: Dict[str, Any], max_print: int = 200) -> str:
    den = report["denominator"]
    out: List[str] = []
    out.append("Wf-A165 GPU lock guard — static scan for ambient-mask defeaters")
    out.append(f"repo root    : {report['repo_root']}")
    out.append(
        "denominator  : {en} tracked *.py/*.sh enumerated, {rd} read, "
        "{un} unreadable, {ly} decoded-lossy "
        "(tracked files in repo: {tt})".format(
            en=den["files_enumerated"], rd=den["files_read"],
            un=den["files_unreadable"], ly=den["files_decoded_lossy"],
            tt=den["tracked_files_total"],
        )
    )
    out.append(
        "              {py} .py + {sh} .sh ; {fw} files carry >=1 finding".format(
            py=den["py_files"], sh=den["sh_files"], fw=den["files_with_findings"]
        )
    )
    summary_bits = []
    for m in MECHANISM_ORDER:
        summary_bits.append(f"{m}={report['mechanisms'][m]['count']}")
    out.append(
        "findings     : {n} total  ({b})".format(
            n=report["findings_total"], b=", ".join(summary_bits)
        )
    )
    out.append(
        "               defeat an ambient CUDA_VISIBLE_DEVICES=\"\" mask : "
        "{n}".format(n=report["findings_defeat_ambient_empty_mask"])
    )
    out.append("")
    printed = 0
    for m in MECHANISM_ORDER:
        rows = report["findings_by_mechanism"].get(m, [])
        eff = report["mechanisms"][m]["effect_breakdown"]
        out.append(
            "{m} {title} — {n} finding(s); effects {e}".format(
                m=m, title=report["mechanisms"][m]["title"], n=len(rows), e=eff or "{}"
            )
        )
        if not rows:
            out.append("  (none — scanned every tracked *.sh / *.py listed in the denominator)")
            continue
        for rec in rows:
            if printed >= max_print:
                out.append("  ... truncated at --max-print; full list in --json")
                printed += 1
                break
            out.append(
                "  {f}:{l}  [{c}]  {s}".format(
                    f=rec["file"], l=rec["line"], c=rec["class"],
                    s=(rec["source"][:160]),
                )
            )
            printed += 1
    out.append("")
    out.append(
        "exit policy  : 1 when findings exist (gates a launch); "
        "--advisory forces 0 while still reporting."
    )
    return "\n".join(out)


def _atomic_write(path: Path, content: str) -> None:
    """Atomic write (POSIX os.replace).  Uses the repo helper when available."""
    helper = Path(__file__).resolve().parent / "wf386a3_atomic_write.py"
    if helper.exists():
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_wf386a3_atomic_write", helper)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.atomic_write_file(path, content)
                return
        except Exception:  # pragma: no cover — fall through to the local writer
            pass
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wfa165_gpu_lock_guard.py",
        description=(
            "Static, read-only, CPU-only guard that reports every repo site that "
            "can defeat an ambient CUDA_VISIBLE_DEVICES='' session lock. "
            "Exit 1 when findings exist; --advisory exits 0 while still reporting."
        ),
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="repository root (default: derived from this script's location)",
    )
    parser.add_argument("--json", action="store_true",
                        help="emit the machine-readable JSON report on stdout")
    parser.add_argument("--advisory", action="store_true",
                        help="report findings but exit 0 (report-only mode)")
    parser.add_argument("--out", default=None,
                        help="also write the JSON report to this path (atomic)")
    parser.add_argument("--max-print", type=int, default=200,
                        help="max findings printed in the human summary (default 200)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the human summary (still honours --json/--out)")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        print(f"ERROR: not a git repository: {repo_root}", file=sys.stderr)
        return 2

    report = scan_repository(repo_root)

    if args.out:
        _atomic_write(Path(args.out), json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif not args.quiet:
        print(render_human(report, max_print=args.max_print))

    if report["findings_total"] and not args.advisory:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
