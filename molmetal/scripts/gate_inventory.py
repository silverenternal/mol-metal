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
    ``None``), and ``source_line`` (the gate function's ``def`` line).

  * :func:`audit_zombie_gates` — flags gates that are opt-in (have an
    ``opt_in_env``) AND whose env var is UNSET in a normal run.  These are
    the "registered but never enforced" zombies.

  * :func:`main` — writes the inventory JSON to
    ``molmetal/reports/gate_inventory.json`` and prints a summary.

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
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _discover_opt_in_env(fn: Callable[..., Any]) -> Optional[str]:
    """Return the env var a gate SKIPs on, or ``None``.

    A gate is opt-in when its source reads ``os.environ.get("<VAR>")`` and
    emits a ``SKIP`` on the falsy branch.  We read the dispatched gate
    function's source and look for the env-var read + a nearby SKIP.
    """
    try:
        module = inspect.getmodule(fn)
        if module is None:
            return None
        target_name = _callable_name(fn)
        target = getattr(module, target_name, None)
        if target is None or not callable(target):
            return None
        try:
            src = inspect.getsource(target)
        except (OSError, TypeError):
            return None
        if not _SKIP_RE.search(src):
            return None
        for literal, const_name in _ENV_READ_RE.findall(src):
            var = literal or None
            if var is None and const_name:
                # Resolve a module-level constant (e.g. PROTOCOL_LOCK_OPT_IN_ENV)
                # to its string value.
                resolved = getattr(module, const_name, None)
                if isinstance(resolved, str):
                    var = resolved
            if not var or var in _NON_GATE_ENV_ALLOWLIST:
                continue
            return var
        return None
    except (OSError, TypeError):
        return None


def build_inventory(pipeline: str = "all") -> List[Dict[str, Any]]:
    """Build the machine-readable gate inventory for ``pipeline``.

    Calls :func:`build_gates` (introspection only — NO gate is executed) and
    returns one dict per registered gate with keys:

      * ``name`` — the registered tuple name.
      * ``callable_name`` — the ``gate_*`` function the lambda dispatches to.
      * ``priority`` — P0/P1/P2/P3 or ``"?"`` when not discoverable.
      * ``opt_in_env`` — the env var the gate SKIPs on, or ``None``.
      * ``source_line`` — the gate function's ``def`` line, or ``None``.
    """
    from molmetal.scripts.pre_flight_check import build_gates

    gates: List[Tuple[str, Callable[[], Any]]] = build_gates(pipeline)
    inventory: List[Dict[str, Any]] = []
    for name, fn in gates:
        callable_name = _callable_name(fn)
        priority = _grep_priority_from_consts(fn) or "?"
        opt_in_env = _discover_opt_in_env(fn)
        source_line = _gate_source_line(fn)
        inventory.append(
            {
                "name": name,
                "callable_name": callable_name,
                "priority": priority,
                "opt_in_env": opt_in_env,
                "source_line": source_line,
            }
        )
    return inventory


def audit_zombie_gates(
    inventory: Optional[List[Dict[str, Any]]] = None,
    env: Optional[Dict[str, str]] = None,
    *,
    confirm_by_running: bool = True,
) -> List[Dict[str, Any]]:
    """Flag opt-in gates that are NEVER enforced in a normal pre-flight run.

    A "zombie gate" is registered in :func:`build_gates` but SKIPs by default,
    so it never blocks anything.  Two things are checked:

      1. the gate reads an ``opt_in_env`` (the cheap static signal), AND
      2. (``confirm_by_running``) the gate ACTUALLY emits SKIP when run with
         that env var unset.

    Step 2 matters because reading an env var is not the same as being opt-in:
    a gate can read an env var as a *downgrade* switch (default-blocking, env
    turns it OFF).  ``gate_csv_hash`` is exactly that shape — its
    ``PREFLIGHT_CSV_HASH_STRICT`` var defaults blocking ON — so a
    static-only audit wrongly reports it as a zombie.  Running the gate is
    the only way to tell "SKIPs by default" from "BLOCKS by default, env can
    downgrade".

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
        if confirm_by_running and not _skips_by_default(entry["name"]):
            continue  # env-reading gate that actually enforces by default
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


def _skips_by_default(gate_name: str) -> bool:
    """Run one registered gate with its opt-in env var cleared; True iff it
    emits SKIP.  Any exception is treated as "not a clean SKIP" (False) so a
    crashing gate is not silently filed as a harmless zombie."""
    from molmetal.scripts.pre_flight_check import build_gates

    fn = dict((g[0], g[1]) for g in build_gates("all")).get(gate_name)
    if fn is None:
        return False
    # Which env var to clear is discovered statically for this one gate.
    var = _discover_opt_in_env(fn)
    saved = os.environ.pop(var, None) if var else None
    try:
        result = fn()
        status = getattr(result, "status", None)
        return status == "SKIP"
    except Exception:
        return False
    finally:
        if var and saved is not None:
            os.environ[var] = saved


def _summarize(inventory: List[Dict[str, Any]], zombies: List[Dict[str, Any]]) -> str:
    n = len(inventory)
    n_optin = sum(1 for e in inventory if e.get("opt_in_env"))
    lines = [
        f"gate inventory: {n} registered gates",
        f"  opt-in gates: {n_optin}",
        f"  zombie gates (opt-in + env unset): {len(zombies)}",
    ]
    for z in sorted(zombies, key=lambda e: e["name"])[:10]:
        lines.append(f"    - {z['name']} (env {z['opt_in_env']})")
    return "\n".join(lines)


def _emit_json(inventory: List[Dict[str, Any]], zombies: List[Dict[str, Any]]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": "all",
        "n_gates": len(inventory),
        "n_zombie_gates": len(zombies),
        "inventory": inventory,
        "zombie_gates": zombies,
    }
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        atomic_write_file(INVENTORY_JSON, json.dumps(payload, indent=2))
    except Exception:  # noqa: BLE001 — fall back to a plain write
        INVENTORY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return INVENTORY_JSON


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Gate inventory + zombie audit")
    parser.add_argument("--pipeline", default="all")
    parser.add_argument("--json-out", default=str(INVENTORY_JSON))
    args = parser.parse_args(argv)

    inventory = build_inventory(args.pipeline)
    zombies = audit_zombie_gates(inventory)
    out = _emit_json(inventory, zombies)
    print(_summarize(inventory, zombies))
    print(f"[gate-inventory] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
