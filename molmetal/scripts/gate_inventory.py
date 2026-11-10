"""Gate inventory + zombie-gate audit for the Mol-Metal pre-flight surface.

================================================================
The pre-flight gate surface lives in one 14,000+ line module
(``molmetal/scripts/pre_flight_check.py``) with ~90 ``gate_*`` functions and
~145 registrations inside :func:`build_gates`.  That monolith is hard to
audit: nothing tells you WHICH gates are actually enforced in a normal run
versus which are registered but silently SKIP unless an env var is set (the
"zombie gate" class).

This module makes the surface auditable:

  * :func:`build_inventory` — calls :func:`build_gates` (WITHOUT executing
    any gate) and emits, per registered gate: ``name``, ``callable_name``
    (the underlying ``gate_*`` function the lambda dispatches to),
    ``priority`` (P0/P1/P2/P3 if discoverable from the lambda's closure /
    args, else ``"?"``), ``opt_in_env`` (the env var the gate SKIPs on, or
    ``None``), ``opt_in_env_source`` (WHERE that env var was found — the
    wrapper itself or the one-hop DELEGATE it dispatches to), and
    ``source_line`` (the gate function's ``def`` line).

  * :func:`audit_zombie_gates` — flags gates that are opt-in (have an
    ``opt_in_env``) AND whose env var is UNSET in a normal run.  These are
    the "registered but never enforced" zombies.

  * :func:`undisclosed_zombies` — the zombies that are NOT declared in
    :data:`DECLARED_OPT_IN_GATES`.  This is the set that BLOCKS.

  * :func:`main` — writes the inventory JSON to
    ``molmetal/reports/gate_inventory.json``, prints a summary, and exits
    NON-ZERO when an undisclosed zombie exists.

Design note (no gate execution): the inventory introspects each registered
callable's CPython code object (``__code__.co_names`` / ``co_consts`` /
``co_freevars``) rather than CALLING it.  Many gates shell out to
subprocesses, load multi-GB artifacts, or run GPU probes; calling ~119 gates
to inventory them would be both slow and side-effecting.  Introspection is
exact for the lambdas this surface uses (a lambda's ``co_names`` names the
``gate_*`` function it dispatches to and its ``co_consts`` carry the
priority / pipeline string args).

Honest framing (AGENTS.md §1 + §5.2):
  * MEASURED counts on disk 2026-10-08: ``all`` = 119, ``launch`` = 121.
  * The zombie count is report-only — it does NOT change any pre-flight
    PASS/FAIL verdict.

  * **2026-11-04 (Wf-A269 dead-gates sweep) — the second bullet above is
    SUPERSEDED; it is PRESERVED VERBATIM here per AGENTS.md §5.5 rule #6.**
    The zombie count is NO LONGER report-only: ``p0_gate_inventory_audit``
    now FAILS on any opt-in ("zombie") gate that is NOT declared in
    :data:`DECLARED_OPT_IN_GATES`, mirroring the
    ``LEDGER_CONFIG_UNPOINTERED_BASELINE`` precedent already used by
    ``pre_flight_check.py`` (a documented BASELINE + "FAIL on anything
    outside it").  "A gate nobody can fail is not a gate" — the repo's own
    wording (``molmetal/reports/wfa166_synthesis/synthesis.md``, §6
    "Rejected options").  The MEASUREMENT stays STATIC — no gate is
    executed.

  * **2026-11-04 (same sweep) — the zombie DETECTION itself was repaired.**
    The audit reported ``zombie_gates=0`` on a tree that registers 5 opt-in
    gates.  Root cause: :func:`_discover_opt_in_env` scanned the THIN
    WRAPPER registered in ``pre_flight_check.py`` (e.g.
    ``gate_wf_r43_m16_pt_ckpt_gap``), whose source contains neither
    ``os.environ.get`` nor the ``SKIP`` literal — the real env read lives
    one module deeper, in the DELEGATE the wrapper imports.  MEASURED on
    this working tree BEFORE the repair: ``all`` = 133 gates, **0** with an
    ``opt_in_env``, **0** zombies.  The repair follows the one-hop
    delegation.  See :func:`_discover_opt_in_env_detail`.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
REPORTS_DIR = MOLMETAL_ROOT / "reports"
INVENTORY_JSON = REPORTS_DIR / "gate_inventory.json"

# Env vars that gate the PRE-FLIGHT RUN ITSELF (not an individual gate) — a
# gate referencing one of these is NOT an opt-in zombie.  Kept small + exact.
_NON_GATE_ENV_ALLOWLIST = frozenset(
    {
        "WF343C4_GATE",
        "USER_OK",
        "PREFLIGHT_DISABLE",
    }
)

#: The DISCLOSED opt-in ("zombie") gates — the gates that are registered but
#: SKIP unless an env var is set, each with the reason it is declared.
#:
#: This is the SAME shape as ``LEDGER_CONFIG_UNPOINTERED_BASELINE`` in
#: ``pre_flight_check.py``: a documented BASELINE, against which the audit
#: FAILS on anything OUTSIDE it, so the population cannot grow silently
#: while the disclosed entries stay visible in every pre-flight report.
#:
#: MEASURED 2026-11-04 on this working tree by live import of
#: :func:`build_inventory` + :func:`audit_zombie_gates` (the 5 entries
#: below are the COMPLETE set of opt-in gates the static scan finds; the
#: count is re-derived live on every run, never typed by hand).  The
#: detection was repaired in the same wave — before the repair the scan
#: found 0, i.e. it was blind to every entry below.
DECLARED_OPT_IN_GATES: Dict[str, str] = {
    "p3_wf_r43_m16_pt_ckpt_gap": (
        "Wf-R43 m16 — the D-MPNN pIC50 Pt checkpoint gap (Pt is absent from "
        "the 25-checkpoint inventory).  The gap is NOT trainable away from "
        "this repo (the MetalCytoToxDB-derived CSVs contain zero Pt rows), "
        "so the gate audits and fails loud while the gap is open.  Opt-in "
        "via WF43_M16_PT_CKPT=1 (Wf-R35 P-A precedent): registering the "
        "gate makes the surface auditable without changing an existing "
        "pre-flight run's verdict.  Implementation: "
        "molmetal/scripts/wf_r43a1_m16_pt_ckpt_gap.py."
    ),
    "p3_wf_r43_m1_paper_cohort": (
        "Wf-R43 m1 — the Ru-AUC paper-cohort comparability guard (the "
        "0.7717 +/- 0.0380 10-fold CV ran on 591 cell lines vs the paper's "
        "23-line Ru ML set, so the -0.038 delta is CROSS-COHORT).  Opt-in "
        "via WF43_M1_PAPER_COHORT=1 (Wf-R35 P-A precedent).  "
        "Implementation: molmetal/scripts/wf_r43a1_m1_paper_cohort_guard.py."
    ),
    "p3_wf_r43_m2_cohort_independence": (
        "Wf-R43 m2 — the per-cohort R² independence guard (2 of 3 'cohorts' "
        "were one measurement).  Opt-in via "
        "WF43_M2_COHORT_INDEPENDENCE=1 (Wf-R35 P-A precedent).  "
        "Implementation: "
        "molmetal/scripts/wf_r43a2_m2_cohort_independence_guard.py."
    ),
    "p3_wf_r43_m7_subpocket_uniform": (
        "Wf-R43 m7 — the diversity_subpocket uniform-placeholder guard.  "
        "Opt-in via WF43_M7_SUBPOCKET_UNIFORM=1 (Wf-R35 P-A precedent).  "
        "Implementation: "
        "molmetal/scripts/wf_r43a3_m7_subpocket_uniform_guard.py."
    ),
    "p3_wf_r43_m19_lomo_headline": (
        "Wf-R43 m19 — the m19 LOMO (Rh+Re) headline real-data-usability "
        "guard (500 SYNTHETIC Rh rows were appended to the shared 5-metal "
        "pool before the LOMO split, so the headline is held-out but NOT "
        "real-data-only).  Opt-in via WF43_M19_LOMO_HEADLINE=1 (Wf-R35 P-A "
        "precedent).  Implementation: "
        "molmetal/scripts/wf_r43a1_m19_lomo_headline_guard.py."
    ),
}

# Regex for an env-var opt-in read.  Catches both a literal
# (os.environ.get("NAME")) and a module-constant reference
# (os.environ.get(CONST_NAME)); the latter is resolved to its string value in
# :func:`_discover_opt_in_env`.
_ENV_READ_RE = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*(?:"([A-Z_][A-Z0-9_]*)"|([A-Za-z_][A-Za-z0-9_]*))""",
)
# A gate SKIPs on an unset env var when it compares the read to a truthy set;
# we detect the SKIP-return on the opt-in path via the SKIP string presence.
_SKIP_RE = re.compile(r"SKIP")

# Opt-in SHAPE: the gate returns SKIP on the branch where the env var is NOT
# truthy — i.e. ``if <env-read> ... not in (...): ... SKIP``.  A downgrade hatch
# has the opposite shape (``if <env-read> in (...): ... SKIP``) and must not be
# counted as a zombie.  We look for `not in` near an env-read within the same
# source block.
_OPT_IN_SHAPE_RE = re.compile(
    r"os\.(?:environ\.get|getenv)\([^\n]*(\n[^\n]*){0,3}?\bnot\s+in\b",
)


def _callable_name(fn: Callable[..., Any]) -> str:
    """Return the underlying gate-function name a registered callable calls.

    Registrations are lambdas like
    ``lambda: gate_foo("p0_foo", "P0", "all", ...)``.  ``co_names`` names the
    globals the lambda references — ``gate_foo`` is the first one.  For a
    plain function registration (rare) fall back to ``fn.__name__``.
    """
    code = getattr(fn, "__code__", None)
    if code is None:
        return getattr(fn, "__name__", repr(fn))
    names = list(code.co_names)
    # Prefer an explicit ``gate_*`` global — that is the dispatcher.
    for n in names:
        if n.startswith("gate_") or n.startswith("_gate_"):
            return n
    # Fallback: a private helper dispatch (e.g. ``_run_wf343c1_gate``).
    for n in names:
        if not n.startswith("__"):
            return n
    return getattr(fn, "__name__", repr(fn))


def _grep_priority_from_consts(fn: Callable[..., Any]) -> Optional[str]:
    """Find a P0/P1/P2/P3 priority string among the lambda's const args."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    for c in code.co_consts:
        if isinstance(c, str) and re.fullmatch(r"P[0-3]", c):
            return c
    # Priority may be captured as a freevar (e.g. loop-built registrations).
    freevars = getattr(code, "co_freevars", ())
    if freevars:
        closure = dict(zip(freevars, (fn.__closure__ or ())))
        for cell in closure.values():
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if isinstance(v, str) and re.fullmatch(r"P[0-3]", v):
                return v
    return None


def _gate_source_line(fn: Callable[..., Any]) -> Optional[int]:
    """Return the ``def`` line of the gate function ``fn`` dispatches to."""
    try:
        src_file = inspect.getsourcefile(fn)
        if src_file is None:
            return None
        base = Path(src_file).name
        # Locate the gate function by name in the module source.
        module = inspect.getmodule(fn)
        if module is None:
            return None
        target_name = _callable_name(fn)
        try:
            target = getattr(module, target_name, None)
        except Exception:  # noqa: BLE001
            target = None
        if target is not None and callable(target):
            try:
                return inspect.getsourcelines(target)[1]
            except (OSError, TypeError):
                pass
        # Fallback: scan the module's own source path.
        mod_file = inspect.getsourcefile(module)
        if mod_file is None:
            return None
        for i, line in enumerate(
            Path(mod_file).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.startswith(f"def {target_name}("):
                return i
        _ = base
        return None
    except (OSError, TypeError):
        return None


# ---------------------------------------------------------------------------
# DEFECT A repair (2026-11-04, Wf-A269 dead-gates sweep) — the thin-wrapper
# -> delegate hop.
#
# A registered gate is often a THIN WRAPPER in ``pre_flight_check.py`` that
# does nothing but import and call a DELEGATE living in its own module:
#
#     def gate_wf_r43_m16_pt_ckpt_gap(name, priority, pipeline, inventory_path, ...):
#         from molmetal.scripts.wf_r43a1_m16_pt_ckpt_gap import (
#             gate_m16_pt_ckpt_gap,
#         )
#         return gate_m16_pt_ckpt_gap(name, priority, pipeline, inventory_path, ...)
#
# The opt-in env read (``os.environ.get(OPT_IN_ENV, "")`` + a ``SKIP``
# return) lives in the DELEGATE.  The wrapper's own source has neither, so a
# source scan of the wrapper alone reports "not opt-in" and the audit
# published ``zombie_gates=0`` on a tree that registers 5 opt-in gates.
#
# The helpers below resolve the delegation WITHOUT importing anything: the
# delegate's source is read from disk and parsed, so the audit stays a pure
# READ-ONLY inspection (no module import side effects, no gate executed).
# ---------------------------------------------------------------------------


def _module_path(module_name: str) -> Optional[Path]:
    """Resolve a dotted module name to a file path WITHOUT importing it."""
    rel = module_name.replace(".", "/")
    for cand in (REPO_ROOT / f"{rel}.py", REPO_ROOT / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _file_constants(path: Path) -> Dict[str, str]:
    """Map a module's module-level string constants to their values.

    Source-level (AST) rather than import-level: importing a module to read
    its constants would execute that module's import-time code, which this
    audit must never do.
    """
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    out: Dict[str, str] = {}
    for node in tree.body:
        targets: List[Any] = []
        value: Any = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                out[t.id] = value.value
    return out


def _function_source(path: Path, func_name: str) -> Optional[str]:
    """Return the source text of ``func_name`` as defined in ``path``."""
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            try:
                seg = ast.get_source_segment(text, node)
            except (OSError, TypeError, ValueError):
                return None
            return seg or None
    return None


def _delegate_functions(
    wrapper_src: str,
) -> List[Tuple[str, str, Dict[str, str]]]:
    """Resolve the functions a thin gate wrapper delegates to (one hop).

    Returns ``(qualified_name, source_text, module_constants)`` triples.  Only
    ``molmetal.*`` modules are considered (a repo-local module); a module
    whose file cannot be found on disk is skipped, never raised.
    """
    out: List[Tuple[str, str, Dict[str, str]]] = []
    try:
        tree = ast.parse(wrapper_src)
    except SyntaxError:
        return out
    seen: set[Tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        mod = node.module or ""
        if not (mod == "molmetal" or mod.startswith("molmetal.")):
            continue
        path = _module_path(mod)
        if path is None:
            continue
        consts = _file_constants(path)
        for alias in node.names:
            key = (mod, alias.name)
            if key in seen:
                continue
            seen.add(key)
            src = _function_source(path, alias.name)
            if src:
                out.append((f"{mod}.{alias.name}", src, consts))
    return out


def _opt_in_env_from_source(
    src: str, consts: Mapping[str, str]
) -> Optional[str]:
    """Return the opt-in env var a source block reads, or ``None``.

    A gate is opt-in (a potential zombie) when its source reads
    ``os.environ.get("<VAR>")`` and emits a ``SKIP`` on the branch taken when
    the var is NOT truthy.  The shape matters: ``if ENV not in ("1", ...):
    return SKIP`` is opt-in (default SKIP), whereas ``if ENV in ("0", ...):
    return SKIP`` is a DOWNGRADE hatch (default enforces) and must NOT be
    reported as a zombie.  This is detected statically (no gate is run).
    """
    if not _SKIP_RE.search(src):
        return None
    if not _OPT_IN_SHAPE_RE.search(src):
        return None  # env read present, but not an opt-in (default-SKIP) shape
    for literal, const_name in _ENV_READ_RE.findall(src):
        var = literal or None
        if var is None and const_name:
            # Resolve a module-level constant (e.g. OPT_IN_ENV) to its
            # string value.
            resolved = consts.get(const_name)
            if isinstance(resolved, str):
                var = resolved
        if not var or var in _NON_GATE_ENV_ALLOWLIST:
            continue
        return var
    return None


def _discover_opt_in_env_detail(
    fn: Callable[..., Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(env_var, where)`` for the opt-in env var of a gate.

    ``where`` names the source the env var was read from — ``wrapper:<fn>``
    when the registered gate function itself reads it, ``delegate:<module>.<fn>``
    when the read is one hop down in a delegated implementation (the
    DEFECT A repair).

    The delegation hop is resolved WITHOUT importing the delegate: the
    delegate's source is read from disk and parsed.  Nothing is executed.
    """
    try:
        module = inspect.getmodule(fn)
        if module is None:
            return None, None
        target_name = _callable_name(fn)
        target = getattr(module, target_name, None)
        if target is None or not callable(target):
            return None, None
        try:
            src = inspect.getsource(target)
        except (OSError, TypeError):
            return None, None
        consts = {
            k: v for k, v in vars(module).items() if isinstance(v, str)
        }
        var = _opt_in_env_from_source(src, consts)
        if var is not None:
            return var, f"wrapper:{target_name}"
        # --- DEFECT A repair (2026-11-04) — follow the delegate hop. ---
        for qualname, dsrc, dconsts in _delegate_functions(src):
            dvar = _opt_in_env_from_source(dsrc, dconsts)
            if dvar is not None:
                return dvar, f"delegate:{qualname}"
        return None, None
    except (OSError, TypeError, AttributeError, ValueError):
        return None, None


def _discover_opt_in_env(fn: Callable[..., Any]) -> Optional[str]:
    """Return the env var a gate SKIPs on by default, or ``None``.

    Thin wrapper: see :func:`_discover_opt_in_env_detail` for the full
    contract (including the 2026-11-04 delegate-hop repair).
    """
    var, _where = _discover_opt_in_env_detail(fn)
    return var


def build_inventory(pipeline: str = "all") -> List[Dict[str, Any]]:
    """Build the machine-readable gate inventory for ``pipeline``.

    Calls :func:`build_gates` (introspection only — NO gate is executed) and
    returns one dict per registered gate with keys:

      * ``name`` — the registered tuple name.
      * ``callable_name`` — the ``gate_*`` function the lambda dispatches to.
      * ``priority`` — P0/P1/P2/P3 or ``"?"`` when not discoverable.
      * ``opt_in_env`` — the env var the gate SKIPs on, or ``None``.
      * ``opt_in_env_source`` — where that env var was found (``wrapper:`` /
        ``delegate:``), or ``None``.
      * ``source_line`` — the gate function's ``def`` line, or ``None``.
    """
    from molmetal.scripts.pre_flight_check import build_gates

    gates: List[Tuple[str, Callable[[], Any]]] = build_gates(pipeline)
    inventory: List[Dict[str, Any]] = []
    for name, fn in gates:
        callable_name = _callable_name(fn)
        priority = _grep_priority_from_consts(fn) or "?"
        opt_in_env, opt_in_env_source = _discover_opt_in_env_detail(fn)
        source_line = _gate_source_line(fn)
        inventory.append(
            {
                "name": name,
                "callable_name": callable_name,
                "priority": priority,
                "opt_in_env": opt_in_env,
                "opt_in_env_source": opt_in_env_source,
                "source_line": source_line,
            }
        )
    return inventory


def audit_zombie_gates(
    inventory: Optional[List[Dict[str, Any]]] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Flag opt-in gates that are NEVER enforced in a normal pre-flight run.

    A "zombie gate" is registered in :func:`build_gates` but SKIPs by default,
    so it never blocks anything.  Detection is STATIC (no gate is executed —
    running 121 gates here would cost ~150s and is what this audit must not
    do): a gate is a zombie iff its source reads an env var in the opt-in
    SHAPE (``if ENV not in (...): return SKIP``) and that var is unset.

    The shape test matters: a gate can read an env var as a *downgrade* hatch
    (default-blocking, env turns it OFF — e.g. ``gate_csv_hash``'s
    ``PREFLIGHT_CSV_HASH_STRICT``).  Those have the opposite shape
    (``if ENV in (...): return SKIP``) and are NOT zombies; a name-only scan
    wrongly reports them.  See :func:`_discover_opt_in_env`.

    Returns a list of ``{"name", "opt_in_env", "evidence"}`` dicts (one per
    zombie).  ``env`` defaults to ``os.environ`` for the current run.
    """
    if inventory is None:
        inventory = build_inventory("all")
    if env is None:
        env = dict(os.environ)
    zombies: List[Dict[str, Any]] = []
    for entry in inventory:
        opt_in_env = entry.get("opt_in_env")
        if not opt_in_env:
            continue
        raw = str(env.get(opt_in_env, "")).strip().lower()
        if raw in ("1", "true", "yes"):
            continue  # env var set → gate is live, not a zombie
        zombies.append(
            {
                "name": entry["name"],
                "opt_in_env": opt_in_env,
                "evidence": (
                    f"{entry['name']} dispatches to "
                    f"{entry['callable_name']}; opt-in env "
                    f"{opt_in_env} unset (default run) → gate emits SKIP "
                    f"and is never enforced"
                ),
            }
        )
    return zombies


def undisclosed_zombies(
    inventory: Optional[List[Dict[str, Any]]] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Return the zombies that are NOT declared in :data:`DECLARED_OPT_IN_GATES`.

    This is the BLOCKING set.  A declared entry is a disclosed,
    documented-intent opt-in gate; an UNDECLARED one is a gate that was
    registered and never enforced without anyone saying so — the
    "registered but never enforced" defect class (AGENTS.md
    ``docs/HONEST_NEGATIVE_CATALOG.md`` §6.269.17).
    """
    zombies = audit_zombie_gates(inventory=inventory, env=env)
    return [z for z in zombies if z["name"] not in DECLARED_OPT_IN_GATES]


def _summarize(inventory: List[Dict[str, Any]], zombies: List[Dict[str, Any]]) -> str:
    n = len(inventory)
    n_optin = sum(1 for e in inventory if e.get("opt_in_env"))
    undisclosed = [z for z in zombies if z["name"] not in DECLARED_OPT_IN_GATES]
    lines = [
        f"gate inventory: {n} registered gates",
        f"  opt-in gates: {n_optin}",
        f"  zombie gates (opt-in + env unset): {len(zombies)}",
        f"    declared (disclosed): {len(zombies) - len(undisclosed)}",
        f"    UNDECLARED (blocking): {len(undisclosed)}",
    ]
    for z in sorted(zombies, key=lambda e: e["name"])[:10]:
        tag = "declared" if z["name"] in DECLARED_OPT_IN_GATES else "UNDECLARED"
        lines.append(f"    - [{tag}] {z['name']} (env {z['opt_in_env']})")
    return "\n".join(lines)


def _utc_now_iso() -> str:
    """ISO-8601 UTC instant of THIS production, second resolution.

    Wf-A269 wave-28 (2026-11-06).  The artefact this module writes used to
    carry NO timestamp at all, so a stale artefact read as a current one
    (the §6.269.38 class: an artefact that lags its repaired producer).
    """
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _surface_sha256() -> str:
    """sha256 of the SURFACE module this inventory introspects.

    Wf-A269 wave-28 (2026-11-06).  The inventory's content has TWO inputs —
    the producer and ``pre_flight_check.py`` (which owns ``build_gates``) —
    so pinning only the producer would leave the gate LIST unpinned: the
    surface could change under an artefact that still claimed to describe it.

    HONEST LIMIT: the per-gate ``opt_in_env`` discovery also reaches ONE-HOP
    DELEGATE modules (the wrappers dispatch to), whose versions are NOT
    pinned here.  A delegate edit can move the artefact's ZOMBIE bookkeeping
    without moving either pin.  The remedy is to re-run the producer; this
    guard cannot see that class of drift.
    """
    surface = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
    if not surface.is_file():
        return "MISSING_SURFACE"
    return hashlib.sha256(surface.read_bytes()).hexdigest()


def _producer_sha256() -> str:
    """sha256 of THIS producer's bytes — a VERSION PIN, not a locator.

    Wf-A269 wave-28 (2026-11-06).  A consumer can now detect "the artefact on
    disk was produced by an OLDER producer" without comparing mtimes, because
    mtime is not reproducible across a clone (the repo's own size-citation
    guard ``molmetal/tests/test_wfa127_size_citation_guard.py`` states that
    rule explicitly).  The repo's standing lesson is that a sha256 is a
    POOR DURABLE LOCATOR for a file's CONTENT but an EXACT provenance pin for
    "which version of the producer wrote this" — and that is the only
    question this field is asked to answer.
    """
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _emit_json(
    inventory: List[Dict[str, Any]],
    zombies: List[Dict[str, Any]],
    out_path: Optional[Path] = None,
    pipeline: str = "all",
) -> Path:
    target = Path(out_path) if out_path is not None else INVENTORY_JSON
    target.parent.mkdir(parents=True, exist_ok=True)
    undisclosed = [z for z in zombies if z["name"] not in DECLARED_OPT_IN_GATES]
    payload = {
        # Wf-A269 wave-28 (2026-11-06) — the pipeline label was HARD-CODED to
        # "all" while `main()` accepts `--pipeline`: `--pipeline launch` wrote
        # the LAUNCH inventory under an "all" label.  A flag that reaches the
        # producer but not its artefact is the §6.269.29 class.
        "pipeline": pipeline,
        # Wf-A269 wave-28 (2026-11-06) — provenance.  Without these two fields
        # the artefact could not say WHEN it was produced nor BY WHICH
        # producer, so a stale artefact read as current.
        "generated_at": _utc_now_iso(),
        "producer_sha256": _producer_sha256(),
        "surface_sha256": _surface_sha256(),
        "n_gates": len(inventory),
        "n_zombie_gates": len(zombies),
        "n_declared_zombie_gates": len(zombies) - len(undisclosed),
        "n_undisclosed_zombie_gates": len(undisclosed),
        "declared_opt_in_gates": dict(DECLARED_OPT_IN_GATES),
        "inventory": inventory,
        "zombie_gates": zombies,
    }
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        atomic_write_file(target, json.dumps(payload, indent=2))
    except Exception:  # noqa: BLE001 — fall back to a plain write
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Gate inventory + zombie audit")
    parser.add_argument("--pipeline", default="all")
    parser.add_argument("--json-out", default=str(INVENTORY_JSON))
    args = parser.parse_args(argv)

    inventory = build_inventory(args.pipeline)
    zombies = audit_zombie_gates(inventory)
    out = _emit_json(
        inventory, zombies, Path(args.json_out), pipeline=args.pipeline
    )
    print(_summarize(inventory, zombies))
    print(f"[gate-inventory] wrote {out}")
    # 2026-11-04 (Wf-A269 dead-gates sweep) — the CLI exit code now REFLECTS
    # the audit.  The previous implementation returned 0 UNCONDITIONALLY, so
    # the tool could not report a finding even when it found one (the
    # dead-gate class: a check that cannot fail).  The superseded behaviour
    # is preserved in the module docstring per AGENTS.md §5.5 rule #6.
    undisclosed = [z for z in zombies if z["name"] not in DECLARED_OPT_IN_GATES]
    if undisclosed:
        for z in sorted(undisclosed, key=lambda e: e["name"]):
            print(
                f"[gate-inventory] FAIL undisclosed opt-in gate (never enforced): "
                f"{z['name']} (env {z['opt_in_env']})"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
