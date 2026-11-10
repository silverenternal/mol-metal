#!/usr/bin/env python
# Wf-A163 ADVERSARIAL refutation helper (refuter id: gpucheck).
#
# INDEPENDENT measurement of three of the branch's load-bearing numbers.
#   1. primary-context-creating call sites in the census's own scope
#   2. import-time probe sites
#   3. the true size of the "28 child spawns inherit NOTHING" population
#
# Ground truth for (1) is DERIVED FROM THE INSTALLED TORCH SOURCE by AST
# (which torch.cuda.* functions transitively reach `_lazy_init()`), NOT from
# a hand-written list -- so the scanner's API table is not assumed correct
# or incorrect a priori.
#
# NEVER imports torch. NEVER touches CUDA. NEVER os.walk (git ls-files only).
#
#   CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
#     molmetal/scripts/wfa163_refute_gpucheck.py
"""Independent audit for the Wf-A163 branch-G GPU-tooling census."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCOPES = ("molmetal", "scripts", "tests")

SPAWN_FUNCS = {"run", "call", "check_call", "check_output", "Popen"}
CVD = "CUDA_VISIBLE_DEVICES"


# ---------------------------------------------------------------- enumerate
def enumerate_files() -> list[str]:
    """git ls-files (tracked) + untracked-not-ignored. NEVER os.walk."""
    out: list[str] = []
    for extra in ([], ["--others", "--exclude-standard"]):
        r = subprocess.run(
            ["git", "ls-files", "-z", *extra, *SCOPES],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        out.extend(p for p in r.stdout.split("\0") if p)
    return sorted(set(out))


# --------------------------------------------- (1) torch ground truth
def torch_primary_context_apis() -> tuple[set[str], dict[str, str]]:
    """Which torch.cuda.* functions transitively reach `_lazy_init()`?

    Derived by AST from the installed torch/cuda/__init__.py: a function is a
    primary-context creator if it calls `_lazy_init()` directly, or calls
    another function that does (transitive closure inside the module).
    """
    init_py = (
        ROOT / ".venv/lib/python3.12/site-packages/torch/cuda/__init__.py"
    )
    src = init_py.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()

    # map: function name -> set of called names (module-level funcs only)
    calls: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Name):
                        inner.add(fn.id)
                    elif isinstance(fn, ast.Attribute):
                        inner.add(fn.attr)
            calls[node.name] = inner

    # transitive closure over module-level functions
    creators: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, inner in calls.items():
            if name in creators:
                continue
            if "_lazy_init" in inner or (inner & creators):
                creators.add(name)
                changed = True

    evidence: dict[str, str] = {}
    for name in sorted(creators):
        # find the line where this function calls _lazy_init (directly or via
        # a creator) -- for citation only
        evidence[name] = f"torch/cuda/__init__.py::{name}"
    return creators, evidence


# --------------------------------------------- repo-side scan helpers
def _module_level_line_set(tree: ast.Module) -> set[int]:
    """Lines whose expression evaluates at IMPORT time.

    Python semantics: module-level statements execute at import; a CLASS
    BODY executes at class-creation time (also import time), but a method
    BODY does not. So the class body's DIRECT statements count, and nested
    function bodies do not.
    """
    lines: set[int] = set()

    def collect(stmts):
        for st in stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(st, ast.ClassDef):
                collect(st.body)
                continue
            for sub in ast.walk(st):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                ln = getattr(sub, "lineno", None)
                if ln is not None:
                    lines.add(ln)

    collect(tree.body)
    return lines


def _call_api(node: ast.AST) -> str | None:
    """Return 'torch.cuda.<fn>' for a Call node, else None."""
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    if (
        isinstance(fn, ast.Attribute)
        and isinstance(fn.value, ast.Attribute)
        and fn.value.attr == "cuda"
        and isinstance(fn.value.value, ast.Name)
        and fn.value.value.id == "torch"
    ):
        return fn.attr
    return None


def scan_repo() -> dict:
    files = enumerate_files()
    py_files = [f for f in files if f.endswith(".py")]

    creators, torch_evidence = torch_primary_context_apis()
    # set_device does not reach _lazy_init() in the installed torch -- it calls
    # torch._C._cuda_setDevice() directly (torch/cuda/__init__.py:739, :752),
    # which retains the device's primary context just the same. It is added
    # explicitly because the transitive closure over module-level functions
    # cannot see the C-level call.
    creators = set(creators) | {"set_device"}

    primary_sites: list[dict] = []
    import_time_sites: list[dict] = []
    probe_sites: list[dict] = []
    parse_failures: list[str] = []

    for rel in py_files:
        p = ROOT / rel
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            parse_failures.append(rel)
            continue
        mod_lines = _module_level_line_set(tree)
        for node in ast.walk(tree):
            api = _call_api(node)
            if api is None:
                continue
            ln = node.lineno
            rec = {"file": rel, "line": ln, "api": api}
            if api in creators:
                primary_sites.append({**rec, "at_import": ln in mod_lines})
            if api in ("is_available", "device_count", "current_device", "set_device", "init"):
                probe_sites.append({**rec, "at_import": ln in mod_lines})

    # also: TRACKED files OUTSIDE the census's three scope prefixes
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    scope_set = set(files)
    out_of_scope: list[dict] = []
    for rel in sorted(p for p in tracked if p.endswith(".py")):
        if rel in scope_set:
            continue
        p = ROOT / rel
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            api = _call_api(node)
            if api in creators:
                out_of_scope.append(
                    {"file": rel, "line": node.lineno, "api": api, "tracked": True}
                )

    # which of the in-scope primary sites does the CENSUS actually record?
    census_path = ROOT / "molmetal/reports/wfa163_gpu_tooling_census/gpu_tooling_census.json"
    census_hitkeys: set[tuple[str, int]] = set()
    if census_path.exists():
        census = json.loads(census_path.read_text(encoding="utf-8"))
        for h in census.get("hits", []):
            census_hitkeys.add((h.get("file", ""), h.get("line", -1)))
    absent = [
        s for s in primary_sites if (s["file"], s["line"]) not in census_hitkeys
    ]

    return {
        "files_enumerated": len(files),
        "py_files": len(py_files),
        "parse_failures": parse_failures,
        "torch_primary_context_apis": sorted(creators),
        "torch_primary_context_api_count": len(creators),
        "torch_evidence": torch_evidence,
        "primary_context_sites_in_scope": len(primary_sites),
        "primary_context_sites_at_import_in_scope": sum(
            1 for s in primary_sites if s["at_import"]
        ),
        "primary_context_sites_by_api": _count(primary_sites, "api"),
        "primary_context_sites_by_scope": _count(
            [_scope(s["file"]) for s in primary_sites], None
        ),
        "primary_sites_absent_from_census_hits": len(absent),
        "primary_sites_detail": primary_sites,
        "probe_sites_in_scope": len(probe_sites),
        "probe_sites_at_import_in_scope": sum(1 for s in probe_sites if s["at_import"]),
        "probe_sites_at_import_detail": [s for s in probe_sites if s["at_import"]],
        "primary_sites_out_of_scope": len(out_of_scope),
        "primary_sites_out_of_scope_detail": out_of_scope,
    }


def _count(items, key):
    out: dict[str, int] = {}
    for i in items:
        k = i[key] if isinstance(i, dict) else i
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _scope(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) >= 2 and parts[0] == "molmetal":
        return "molmetal/" + parts[1]
    return parts[0]


# --------------------------------------------- (3) env-provenance classifier
def _env_provenance(node: ast.AST, assigns: dict[str, ast.AST], src: str) -> str:
    """Resolve an `env=` value to a provenance class, through local assigns.

    `hand_made_no_cvd` is the class that MATTERS: an env dict that carries no
    CUDA_VISIBLE_DEVICES key and does not copy os.environ -- such a child
    inherits NOTHING from the parent, including the CPU lock.
    """
    seen = 0
    while isinstance(node, ast.Name) and node.id in assigns and seen < 6:
        node = assigns[node.id]
        seen += 1
    seg = ast.get_source_segment(src, node) or ""

    if isinstance(node, ast.Dict):
        keys = {
            k.value
            for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        has_unpack = any(k is None for k in node.keys)
        if CVD in keys:
            return "carries_cvd"
        if has_unpack:
            return "dict_unpack"
        if not keys:
            return "empty_dict"
        return "hand_made_no_cvd"

    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "dict":
            # dict(os.environ) / dict(**os.environ) / dict(__import__("os").environ)
            # NOTE: the spelling is ("os").environ -- the literal substring
            # "os.environ" is ABSENT, which is exactly why a substring-based
            # classifier gets this wrong.
            if "environ" in seg:
                return "inherits_os_environ"
            return "dict_other"
        if isinstance(fn, ast.Attribute) and fn.attr == "copy":
            if "environ" in seg:
                return "inherits_os_environ"
            return "copy_other"
    if "environ" in seg:
        return "inherits_os_environ"
    return "other"


def classify_droppers() -> dict:
    """Independently classify EVERY child spawn's env provenance."""
    files = enumerate_files()
    py_files = [f for f in files if f.endswith(".py")]
    out: list[dict] = []
    for rel in py_files:
        p = ROOT / rel
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        for fn_node in ast.walk(tree):
            if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assigns: dict[str, ast.AST] = {}
            for sub in ast.walk(fn_node):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            assigns[t.id] = sub.value
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    assigns[sub.target.id] = sub.value
            for node in ast.walk(fn_node):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = None
                base = None
                if isinstance(fn, ast.Attribute) and fn.attr in SPAWN_FUNCS:
                    name = fn.attr
                    base = getattr(fn.value, "id", None)
                elif isinstance(fn, ast.Name) and fn.id in SPAWN_FUNCS:
                    name = fn.id
                if name is None or base not in (None, "subprocess"):
                    continue
                env_node = None
                for kw in node.keywords:
                    if kw.arg == "env":
                        env_node = kw.value
                if env_node is None:
                    out.append(
                        {
                            "file": rel,
                            "line": node.lineno,
                            "fn": name,
                            "provenance": "no_env_kwarg_inherits",
                        }
                    )
                    continue
                out.append(
                    {
                        "file": rel,
                        "line": node.lineno,
                        "fn": name,
                        "provenance": _env_provenance(env_node, assigns, src),
                        "env_src": ast.get_source_segment(src, env_node) or "",
                    }
                )

    # ---- cross-check against the CENSUS's own drops_lock list ----
    census_path = ROOT / "molmetal/reports/wfa163_gpu_tooling_census/gpu_tooling_census.json"
    census_drops: set[tuple[str, int]] = set()
    if census_path.exists():
        census = json.loads(census_path.read_text(encoding="utf-8"))
        for h in census.get("hits", []):
            if h.get("kind") == "child_spawn" and h.get("env_mode") == "drops_lock":
                census_drops.add((h.get("file", ""), h.get("line", -1)))
    my_hand = {
        (o["file"], o["line"]) for o in out if o["provenance"] == "hand_made_no_cvd"
    }
    census_only = sorted(census_drops - my_hand)

    prov = _count(out, "provenance")
    hand = [o for o in out if o["provenance"] == "hand_made_no_cvd"]
    return {
        "total_child_spawns": len(out),
        "by_provenance": prov,
        "hand_made_no_cvd_count": len(hand),
        "hand_made_no_cvd_files": len({o["file"] for o in hand}),
        "census_claimed_drops_lock": len(census_drops),
        "census_drops_not_hand_made_by_my_classifier": len(census_only),
        "census_only_detail": [
            {
                "file": f,
                "line": ln,
                "my_classification": next(
                    (
                        o["provenance"]
                        for o in out
                        if (o["file"], o["line"]) == (f, ln)
                    ),
                    "not_a_direct_spawn (wrapper call site)",
                ),
                "my_env_src": next(
                    (
                        o.get("env_src", "")
                        for o in out
                        if (o["file"], o["line"]) == (f, ln)
                    ),
                    "",
                ),
            }
            for f, ln in census_only
        ],
        "hand_made_detail": hand,
        "all_detail": out,
    }


if __name__ == "__main__":
    res = scan_repo()
    res["droppers"] = classify_droppers()
    out = ROOT / "molmetal/reports/wfa163_refute_gpucheck/independent_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    slim = {k: v for k, v in res.items() if not k.endswith("_detail")}
    print(json.dumps(slim, indent=2))
    print("CVD in env of this process:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
    sys.exit(0)
