#!/usr/bin/env python3
# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-A163 — the GPU-TOOLING CENSUS: every path in this repo that can touch
GPU memory when the session forbids it.

WHAT THIS IS
------------
A *census*, not a guard.  ``wfa156_gpu_safety_guard.py`` already guards the
TEST + SMOKE surfaces for the ``"cuda" if torch.cuda.is_available() else
"cpu"`` idiom.  This script covers the WHOLE repo and every GPU-relevant
construct, not one idiom:

  1. ``cuda_probe``        — a call to a torch CUDA API
                            (``is_available`` / ``device_count`` /
                            ``current_device``).
  2. ``device_autoselect`` — a device-shaped binding or argparse default that
                            can yield a CUDA device WITHOUT an opt-in.
  3. ``child_spawn``       — a ``subprocess`` / ``os.system`` launch of a
                            child that may import torch.
  4. ``session_lock``      — a write to ``CUDA_VISIBLE_DEVICES`` (the lock,
                            or an OVERRIDE of it).
  5. ``device_override``   — a ``MOLMETAL_*DEVICE``-style env override (the
                            *good* pattern: the device becomes a decision,
                            not an accident).

Every hit is classified on FOUR axes, computed mechanically (the rules are
printed in the artefact so a verifier can refute them):

  ``allocates_context``   Would a CUDA context be created if the session did
                          NOT blank CUDA?  Read from the installed torch
                          source, not assumed.
  ``runs_workload``       Could this run TRAINING / SAMPLING / DECODE on the
                          GPU?
  ``honours_session_lock``Does it (a) force ``CUDA_VISIBLE_DEVICES=""``,
                          (b) check ``CUDA_VISIBLE_DEVICES == ""`` before
                          probing, or (c) offer a ``MOLMETAL_SMOKE_DEVICE``-
                          style override?
  ``gate``                gate / smoke / test / tool vs a production
                          training script.

CPU-ONLY
--------
This module imports **stdlib only** (``ast``, ``re``, ``json``, ``subprocess``
for ``git`` only).  It never imports torch, never touches CUDA, and never
runs a child that could.  It is safe to run inside a CPU-locked session.

NO os.walk
----------
Path enumeration goes through ``git ls-files`` (tracked + untracked-not-
ignored).  The repository is ~201 GB with ~43 GB of ``.pt`` files; a walk
once OOM-killed the session.

USAGE
-----
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \\
        molmetal/scripts/wfa163_gpu_tooling_census.py \\
        --out molmetal/reports/wfa163_gpu_tooling_census/gpu_tooling_census.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The scan scope, as required by Wf-A163: ``molmetal/**``, ``scripts/**``,
#: ``tests/**``.  Everything else in the repo is out of scope BY CONSTRUCTION
#: and that fact is recorded in the artefact (a scan that silently covers
#: less than the reader believes is not a measurement).
SCOPE_PREFIXES: Tuple[str, ...] = ("molmetal/", "scripts/", "tests/")

CVD = "CUDA_VISIBLE_DEVICES"
NVML_ENV = "PYTORCH_NVML_BASED_CUDA_CHECK"

#: torch CUDA APIs that the census treats as probes, with the ALLOCATION
#: verdict read from the installed torch source
#: (``.venv/lib/python3.12/site-packages/torch/cuda/__init__.py``).
PROBE_APIS: Dict[str, Dict[str, str]] = {
    "is_available": {
        "allocates_context": "yes",
        "primary_context": "no",
        "basis": (
            "torch/cuda/__init__.py:209-229 — is_available() takes the else "
            "branch unless PYTORCH_NVML_BASED_CUDA_CHECK=1; that branch calls "
            "torch._C._cuda_getDeviceCount(), whose own comment says it "
            "'uses the CUDA Runtime API cudaGetDeviceCount which in turn "
            "initializes the CUDA Driver API via cuInit'. cuInit is a DRIVER "
            "initialisation, not a primary-context creation: the ~500-600 MiB "
            "class of allocation belongs to a PRIMARY CONTEXT (cudaSetDevice "
            "/ cuDevicePrimaryCtxRetain / the first tensor allocation). Under "
            "CUDA_VISIBLE_DEVICES='' the query returns 0 devices, so torch "
            "never creates a context at all."
        ),
    },
    "device_count": {
        "allocates_context": "conditional",
        "primary_context": "no",
        "basis": (
            "torch/cuda/__init__.py:1212-1238 — device_count() prefers "
            "_device_count_nvml() (NVML, no cuInit) but falls back to "
            "torch._C._cuda_getDeviceCount() (-> cuInit) when NVML discovery "
            "fails (nvml_count < 0) or when CUDA is already initialised. "
            "Driver-level only: no primary context."
        ),
    },
    "current_device": {
        "allocates_context": "yes",
        "primary_context": "yes",
        "basis": (
            "torch/cuda/__init__.py:1265-1268 — current_device() calls "
            "_lazy_init() -> torch._C._cuda_init(), which initialises CUDA "
            "and creates the primary context."
        ),
    },
    "set_device": {
        "allocates_context": "yes",
        "primary_context": "yes",
        "basis": (
            "torch/cuda/__init__.py:739-752 — set_device() calls "
            "torch._C._cuda_setDevice(device) for a non-negative index; "
            "cudaSetDevice retains the device's PRIMARY CONTEXT, which is "
            "the ~500-600 MiB class of allocation. No-op for a negative index."
        ),
    },
    "init": {
        "allocates_context": "yes",
        "primary_context": "yes",
        "basis": (
            "torch/cuda/__init__.py:549,563-591 — init() calls _lazy_init() "
            "-> torch._C._cuda_init(), which initialises CUDA and creates the "
            "primary context."
        ),
    },
}

#: Names a binding must have to count as a *device* selection.
DEVICE_NAME_RE = re.compile(
    r"^(device|devices|dev|DEVICE|device_str|device_name|device_str_|_device"
    r"|torch_device|selected_device|target_device|compute_device|gpu_device"
    r"|resolved_device|active_device|default_device|DEVICE_STR)$"
)

#: Markers that make a child launch a *workload* (train / sample / decode)
#: rather than a *tool* (gate / check / guard / audit).
WORKLOAD_MARKER_RE = re.compile(
    r"train|retrain|decode|sample|sweep|bench|panel|generate|cfm|dmpnn|"
    r"chemprop|moleculenet|property",
    re.IGNORECASE,
)
TOOL_MARKER_RE = re.compile(
    r"gate|check|guard|audit|verify|census|probe|pre_flight|doctor|lint",
    re.IGNORECASE,
)

GATE_NAME_RE = re.compile(
    r"(gate|pre_flight|check|guard|audit|census|verify|doctor|probe|"
    r"smoke|contract|validator|test_|_test)",
    re.IGNORECASE,
)

#: Env overrides that make the device an explicit DECISION (the good pattern).
DEVICE_OVERRIDE_RE = re.compile(
    r"MOLMETAL_(SMOKE_)?DEVICE|MOLMETAL_FORCE_DEVICE|MOLMETAL_GPU",
)


# ---------------------------------------------------------------------------
# Path enumeration (git ls-files; NEVER os.walk)
# ---------------------------------------------------------------------------
def _git_ls_files(extra: Sequence[str] = ()) -> List[str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", *extra],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed rc={proc.returncode}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:400]}"
        )
    return [
        p.decode("utf-8", "replace")
        for p in proc.stdout.split(b"\0")
        if p
    ]


def enumerate_scope_files() -> List[Dict[str, str]]:
    """Every file in scope, tagged tracked / untracked."""
    out: List[Dict[str, str]] = []
    seen = set()
    for kind, extra in (
        ("tracked", ()),
        ("untracked", ("--others", "--exclude-standard")),
    ):
        for rel in _git_ls_files(extra):
            if rel in seen:
                continue
            seen.add(rel)
            if not rel.startswith(SCOPE_PREFIXES):
                continue
            if rel.endswith("/"):
                continue
            out.append({"path": rel, "tracked": kind})
    return sorted(out, key=lambda d: d["path"])


def classify_scope(rel: str) -> str:
    """Sub-scope label for the denominator."""
    if rel.startswith("molmetal/reports/"):
        return "molmetal/reports (archived snapshots)"
    if rel.startswith("molmetal/references/"):
        return "molmetal/references (vendored)"
    if rel.startswith("molmetal/tests/"):
        return "molmetal/tests"
    if rel.startswith("molmetal/scripts/"):
        return "molmetal/scripts"
    if rel.startswith("molmetal/"):
        return "molmetal/<package>"
    if rel.startswith("scripts/"):
        return "scripts"
    if rel.startswith("tests/"):
        return "tests"
    return "other"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
def _seg(src: str, node: ast.AST) -> str:
    try:
        seg = ast.get_source_segment(src, node)
    except Exception:  # pragma: no cover - defensive
        seg = None
    if seg:
        return seg
    try:
        lineno = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if lineno and end:
            return "\n".join(src.splitlines()[lineno - 1 : end])
    except Exception:  # pragma: no cover - defensive
        pass
    return ""


def _iter_module_level(tree: ast.Module) -> Iterable[ast.stmt]:
    """Module-scope statements, descending into if/try/with but NOT into
    function or class bodies."""

    def walk(body: Sequence[ast.stmt]) -> Iterable[ast.stmt]:
        for node in body:
            yield node
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith)):
                yield from walk(node.body)
                yield from walk(getattr(node, "orelse", []) or [])
                yield from walk(getattr(node, "finalbody", []) or [])
                for handler in getattr(node, "handlers", []) or []:
                    yield from walk(handler.body)

    return walk(tree.body)


def _cvd_target(node: ast.AST) -> bool:
    """True when ``node`` writes to ``os.environ["CUDA_VISIBLE_DEVICES"]``."""
    if not isinstance(node, ast.Subscript):
        return False
    sl = node.slice
    if isinstance(sl, ast.Constant) and sl.value == CVD:
        return True
    return False


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _contains_cuda_const(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if sub.value == "cuda" or sub.value.startswith("cuda:"):
                return True
    return False


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


# ---------------------------------------------------------------------------
# The Python scanner
# ---------------------------------------------------------------------------
class PyFileScan:
    def __init__(self, rel: str, src_lines: List[str], tree: ast.Module) -> None:
        self.rel = rel
        self.src_lines = src_lines
        self.src = "\n".join(src_lines)
        self.tree = tree
        self.hits: List[Dict[str, Any]] = []

        # --- file-level facts -------------------------------------------
        self.module_level_cvd_write: Optional[Dict[str, Any]] = None
        self.any_cvd_write: List[Dict[str, Any]] = []
        self.lock_check_lines: List[int] = []
        self.has_cvd_nonblank_write = False
        self.has_cvd_blank_write = False

        self._scan_module_level()
        self._index_env_symbols()
        self._index_module_level_lines()
        self._scan_probes()
        self._scan_device_autoselect()
        self._scan_spawns()
        self._scan_lock_writes()
        self._scan_overrides()
        self._scan_lock_checks()

    # -- file-level -----------------------------------------------------
    def _scan_module_level(self) -> None:
        for node in _iter_module_level(self.tree):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if _cvd_target(tgt) and _is_os_environ(tgt.value):
                            info = {
                                "lineno": sub.lineno,
                                "value_src": _seg(self.src, sub.value),
                            }
                            self.any_cvd_write.append(info)
                            if self.module_level_cvd_write is None:
                                self.module_level_cvd_write = info
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if (
                        isinstance(fn, ast.Attribute)
                        and fn.attr in ("setdefault", "update")
                        and _is_os_environ(fn.value)
                    ):
                        seg = _seg(self.src, sub)
                        if CVD in seg:
                            self.any_cvd_write.append(
                                {"lineno": sub.lineno, "value_src": seg}
                            )

    def _index_module_level_lines(self) -> None:
        """Line numbers executed at IMPORT time (module scope, not inside a
        function or class body)."""
        lines: set = set()
        for node in _iter_module_level(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start and end:
                lines.update(range(start, end + 1))
        self._module_level_lines = lines

    def _scan_lock_checks(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Compare):
                seg = _seg(self.src, node)
                if CVD in seg and re.search(r'==\s*""|!=\s*""', seg):
                    self.lock_check_lines.append(node.lineno)

    def _index_env_symbols(self) -> None:
        """Index local assignments, function defs and parameter names so the
        ``env=`` expression can be resolved instead of guessed."""
        self._assigns: List[Tuple[int, str, ast.AST]] = []
        self._funcs: Dict[str, ast.AST] = {}
        self._param_names: set = set()
        self._attr_assigns: Dict[str, ast.AST] = {}
        self._cvd_mutations: List[Tuple[int, str, ast.AST]] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        self._assigns.append((node.lineno, tgt.id, node.value))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.value is not None:
                    self._assigns.append(
                        (node.lineno, node.target.id, node.value)
                    )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._funcs[node.name] = node
                a = node.args
                for arg in (
                    list(a.posonlyargs)
                    + list(a.args)
                    + list(a.kwonlyargs)
                ):
                    self._param_names.add(arg.arg)
                if a.vararg:
                    self._param_names.add(a.vararg.arg)
                if a.kwarg:
                    self._param_names.add(a.kwarg.arg)
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute):
                        dotted = self._dotted(tgt)
                        if dotted:
                            self._attr_assigns[dotted] = node.value
                    elif (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and _const_str(tgt.slice) == CVD
                    ):
                        self._cvd_mutations.append(
                            (node.lineno, tgt.value.id, node.value)
                        )
        self._assigns.sort(key=lambda t: t[0])
        # function name -> {param: default node}
        self._func_params: Dict[str, Dict[str, Optional[ast.AST]]] = {}
        for fname, fnode in self._funcs.items():
            a = fnode.args
            pos = list(a.posonlyargs) + list(a.args)
            defaults: Dict[str, Optional[ast.AST]] = {}
            if a.defaults:
                for p, d in zip(pos[len(pos) - len(a.defaults) :], a.defaults):
                    defaults[p.arg] = d
            for p, d in zip(a.kwonlyargs, a.kw_defaults):
                defaults[p.arg] = d
            for p in pos:
                defaults.setdefault(p.arg, None)
            for p in a.kwonlyargs:
                defaults.setdefault(p.arg, None)
            self._func_params[fname] = defaults

    def _enclosing_func(self, node: ast.AST) -> Optional[str]:
        """Name of the innermost function containing ``node``."""
        best: Optional[Tuple[int, str]] = None
        for fname, fnode in self._funcs.items():
            for sub in ast.walk(fnode):
                if sub is node:
                    depth = getattr(fnode, "lineno", 0)
                    if best is None or depth > best[0]:
                        best = (depth, fname)
                    break
        return best[1] if best else None

    @staticmethod
    def _dotted(node: ast.AST) -> str:
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return ""

    # -- hits -----------------------------------------------------------
    def _add(self, **kw: Any) -> None:
        self.hits.append(kw)

    def _scan_probes(self) -> None:
        direct = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module == "torch.cuda":
                for alias in node.names:
                    if alias.name in PROBE_APIS:
                        direct.add(alias.name)
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            api: Optional[str] = None
            if isinstance(f, ast.Attribute) and f.attr in PROBE_APIS:
                v = f.value
                if isinstance(v, ast.Attribute) and v.attr == "cuda":
                    base = v.value
                    if isinstance(base, ast.Name) and base.id == "torch":
                        api = f.attr
            elif isinstance(f, ast.Name) and f.id in PROBE_APIS and f.id in direct:
                api = f.id
            if api is None:
                continue
            meta = PROBE_APIS[api]
            module_level = node.lineno in self._module_level_lines
            self._add(
                kind="cuda_probe",
                file=self.rel,
                line=node.lineno,
                source=_seg(self.src, node).splitlines()[0][:400]
                if _seg(self.src, node)
                else "",
                api=api,
                module_level=module_level,
                allocates_context=meta["allocates_context"],
                allocates_context_basis=meta["basis"],
                creates_primary_context=meta.get("primary_context", "unknown"),
                runs_workload=False,
                runs_workload_basis="a probe allocates a context; it does not "
                "run a workload by itself",
                gate=self._gate_label(),
            )

    def _scan_device_autoselect(self) -> None:
        for node in ast.walk(self.tree):
            # argparse --device with a default
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_argument":
                    flags = [
                        a.value
                        for a in node.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)
                    ]
                    if any(f in ("--device", "--gpu", "--devices") for f in flags):
                        default = next(
                            (kw.value for kw in node.keywords if kw.arg == "default"),
                            None,
                        )
                        if default is not None:
                            self._add(
                                kind="device_autoselect",
                                file=self.rel,
                                line=node.lineno,
                                source=_seg(self.src, node).splitlines()[0][:400]
                                if _seg(self.src, node)
                                else "",
                                detail="argparse default",
                                default_src=_seg(self.src, default)[:200],
                                allocates_context="no",
                                allocates_context_basis="a default string "
                                "allocates nothing; it only selects a device "
                                "for a LATER call",
                                runs_workload=False,
                                runs_workload_basis="the default is consumed by "
                                "the caller",
                                gate=self._gate_label(),
                            )
            # torch.device("cuda...")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "device"
            ):
                if _contains_cuda_const(node):
                    self._add(
                        kind="device_autoselect",
                        file=self.rel,
                        line=node.lineno,
                        source=_seg(self.src, node).splitlines()[0][:400]
                        if _seg(self.src, node)
                        else "",
                        detail="torch.device(<cuda>)",
                        allocates_context="no",
                        allocates_context_basis="constructing a torch.device "
                        "object allocates no GPU memory; using it does",
                        runs_workload=False,
                        runs_workload_basis="the object is consumed later",
                        gate=self._gate_label(),
                    )
            # device-shaped binding
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and DEVICE_NAME_RE.match(tgt.id):
                        if _contains_cuda_const(node.value):
                            self._add(
                                kind="device_autoselect",
                                file=self.rel,
                                line=node.lineno,
                                source=_seg(self.src, node).splitlines()[0][:400]
                                if _seg(self.src, node)
                                else "",
                                detail=f"binding to `{tgt.id}`",
                                allocates_context="no",
                                allocates_context_basis="the binding allocates "
                                "nothing; a LATER probe or tensor move does",
                                runs_workload=False,
                                runs_workload_basis="the binding selects a "
                                "device for a later call",
                                gate=self._gate_label(),
                            )

    # -- child spawns ---------------------------------------------------
    def _resolve_env_node(self, node: ast.AST, depth: int = 0) -> Dict[str, str]:
        """Resolve an ``env=`` expression into a lock verdict.

        The naive reading of the ``env=`` kwarg text is WRONG in this repo:
        ``env=env`` is used with ``env = dict(os.environ)`` (inherits) in one
        file and with ``env = {"PATH": ...}`` (drops the lock) in another, and
        ``env={**_env_cpu_only()}`` is a helper that FORCES the lock.  So the
        expression is resolved through local assignments and helper bodies.
        """
        if depth > 5:
            return {"kind": "unresolved"}
        if isinstance(node, ast.Name):
            best = None
            for lineno, name, val in self._assigns:
                if name == node.id and lineno <= node.lineno:
                    best = (lineno, val)
            if best is None:
                return {"kind": "unresolved"}
            verdict = self._resolve_env_node(best[1], depth + 1)
            # A later mutation of the SAME name can override the copy:
            #   env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
            for mlineno, mname, mval in self._cvd_mutations:
                if (
                    mname == node.id
                    and best[0] < mlineno <= node.lineno
                ):
                    mseg = _seg(self.src, mval).strip()
                    if mseg in ('""', "''"):
                        return {"kind": "forces_blank"}
                    if re.fullmatch(r'["\'][^"\']*["\']', mseg):
                        return {"kind": "sets_cvd_nonblank"}
            return verdict
        if isinstance(node, ast.Dict):
            seg = _seg(self.src, node)
            verdict = self._classify_env_text(seg)
            if verdict["kind"] != "drops_lock":
                return verdict
            # `{**helper()}` — resolve the unpacked expression.
            for key, value in zip(node.keys, node.values):
                if key is None:  # a ** unpacking
                    r = self._resolve_env_node(value, depth + 1)
                    if r["kind"] in ("forces_blank", "sets_cvd_nonblank"):
                        return r
                    if r["kind"] == "inherits_parent_env":
                        verdict = {"kind": "inherits_parent_env"}
            return verdict
        if isinstance(node, ast.IfExp):
            for branch in (node.orelse, node.body):
                r = self._resolve_env_node(branch, depth + 1)
                if r["kind"] in ("forces_blank", "sets_cvd_nonblank"):
                    return r
                if r["kind"] == "inherits_parent_env":
                    return r
            return {"kind": "unresolved"}
        if isinstance(node, ast.Attribute):
            dotted = self._dotted(node)
            if dotted and dotted in self._attr_assigns:
                return self._resolve_env_node(self._attr_assigns[dotted], depth + 1)
            return {"kind": "unresolved"}
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                r = self._resolve_env_node(v, depth + 1)
                if r["kind"] != "unresolved":
                    return r
            return {"kind": "unresolved"}
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "copy":
                if "os.environ" in _seg(self.src, fn.value):
                    return {"kind": "inherits_parent_env"}
            if isinstance(fn, ast.Name) and fn.id == "dict":
                if "os.environ" in _seg(self.src, node):
                    return {"kind": "inherits_parent_env"}
            if isinstance(fn, ast.Name) and fn.id in self._funcs:
                body = _seg(self.src, self._funcs[fn.id])
                if CVD in body:
                    return self._classify_env_text(body)
                if "os.environ" in body:
                    return {"kind": "inherits_parent_env"}
                return {"kind": "unresolved"}
        seg = _seg(self.src, node)
        return self._classify_env_text(seg)

    @staticmethod
    def _classify_env_text(seg: str) -> Dict[str, str]:
        if CVD in seg:
            if re.search(
                r'CUDA_VISIBLE_DEVICES["\']?\]?\s*["\']?\s*[:=]\s*["\']{2}', seg
            ):
                return {"kind": "forces_blank"}
            return {"kind": "sets_cvd_nonblank"}
        if "os.environ" in seg:
            return {"kind": "inherits_parent_env"}
        if not seg.strip():
            return {"kind": "unresolved"}
        return {"kind": "drops_lock"}

    def _env_mode(self, env_kw: Optional[ast.keyword]) -> str:
        """How the child's ``CUDA_VISIBLE_DEVICES`` is decided.

        ``forces_blank``        the child is given an empty mask.
        ``sets_cvd_nonblank``   the child is given a NON-empty mask -> the
                                ambient blank is DEFEATED.
        ``inherits_parent_env`` the child gets the parent's environment (no
                                ``env=``, or ``env=`` built from ``os.environ``)
                                -> the ambient blank PROPAGATES, but only if
                                the parent had it.
        ``drops_lock``          the child gets an ``env=`` that neither
                                mentions CUDA_VISIBLE_DEVICES nor inherits
                                ``os.environ`` -> the child sees EVERY device,
                                so an ambient blank does NOT reach it.
        ``parameterised``       the env comes from a wrapper PARAMETER; the
                                verdict is decided at the wrapper's call
                                sites.
        """
        if env_kw is None:
            return "inherits_parent_env"
        return self._resolve_env_node(env_kw.value)["kind"]

    def _spawn_fn_name(self, node: ast.Call) -> Optional[str]:
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            if f.value.id == "subprocess" and f.attr in (
                "run",
                "Popen",
                "call",
                "check_call",
                "check_output",
            ):
                return f"subprocess.{f.attr}"
            if f.value.id == "os" and f.attr in ("system", "popen"):
                return f"os.{f.attr}"
        return None

    def _mentions_torch(self, seg: str) -> bool:
        return bool(re.search(r"torch|cuda|\.pt\b", seg, re.IGNORECASE))

    def _scan_spawns(self) -> None:
        # Pass 1 — direct spawns, and the WRAPPERS that contain them.
        wrappers: Dict[str, ast.AST] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(
                    self._spawn_fn_name(sub) is not None
                    for sub in ast.walk(node)
                    if isinstance(sub, ast.Call)
                ):
                    wrappers[node.name] = node
        self.spawn_wrappers = sorted(wrappers)

        # Which PARAMETER supplies the wrapper's env, and what it defaults to.
        self._wrapper_env_param: Dict[str, Tuple[str, Optional[ast.AST]]] = {}
        for wname, wnode in wrappers.items():
            defaults = self._func_params.get(wname, {})
            for sub in ast.walk(wnode):
                if not isinstance(sub, ast.Call):
                    continue
                if self._spawn_fn_name(sub) is None:
                    continue
                env_kw = next(
                    (kw for kw in sub.keywords if kw.arg == "env"), None
                )
                if env_kw is None:
                    continue
                if isinstance(env_kw.value, ast.Name):
                    self._wrapper_env_param[wname] = (
                        env_kw.value.id,
                        defaults.get(env_kw.value.id),
                    )
                    break

        wrapper_node_ids = {
            id(x) for f in wrappers.values() for x in ast.walk(f)
        }
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            fname = self._spawn_fn_name(node)
            if fname is None:
                continue
            seg = _seg(self.src, node)
            env_kw = next((kw for kw in node.keywords if kw.arg == "env"), None)
            enclosing = self._enclosing_func(node)
            env_mode = self._env_mode(env_kw)
            if (
                env_kw is not None
                and isinstance(env_kw.value, ast.Name)
                and enclosing is not None
                and env_kw.value.id in self._func_params.get(enclosing, {})
            ):
                env_mode = "parameterised"
            workload = bool(WORKLOAD_MARKER_RE.search(seg))
            tool = bool(TOOL_MARKER_RE.search(seg))
            mentions_torch = self._mentions_torch(seg)
            if not mentions_torch:
                alloc, alloc_basis = (
                    "no",
                    "the command does not mention torch or cuda; the child "
                    "cannot create a CUDA context by itself",
                )
            elif workload:
                alloc, alloc_basis = (
                    "yes",
                    "the child imports torch and runs a train/decode/sweep "
                    "target; a visible device means a CUDA context",
                )
            else:
                alloc, alloc_basis = (
                    "conditional",
                    "the child imports torch and may probe CUDA",
                )
            self._add(
                kind="child_spawn",
                file=self.rel,
                line=node.lineno,
                source=seg.splitlines()[0][:400] if seg else "",
                spawn=fname,
                role="wrapper_definition"
                if id(node) in wrapper_node_ids
                else "direct",
                env_mode=env_mode,
                launches_workload=workload,
                launches_tool=tool,
                allocates_context=alloc,
                allocates_context_basis=alloc_basis,
                runs_workload=workload,
                runs_workload_basis=(
                    "the command names a train/decode/sweep/bench target"
                    if workload
                    else "no train/decode/sweep marker in the command"
                ),
                gate=self._gate_label(),
            )

        # Pass 2 — CALL SITES of the wrapper functions.  A wrapper hides the
        # real launch site from a naive `subprocess.run` grep; the pre-flight
        # ships its trainer smokes through `_run(...)`.
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Name) and f.id in wrappers):
                continue
            if any(
                isinstance(x, ast.Call) and x is node
                for x in ast.walk(wrappers[f.id])
            ):
                continue
            seg = _seg(self.src, node)
            env_kw = next((kw for kw in node.keywords if kw.arg == "env"), None)
            env_mode = self._env_mode(env_kw)
            if env_mode == "parameterised" and f.id in self._wrapper_env_param:
                _pname, default = self._wrapper_env_param[f.id]
                if default is None:
                    # subprocess.run(env=None) -> the child inherits the
                    # parent's environment EXACTLY.
                    env_mode = "inherits_parent_env"
                else:
                    env_mode = self._resolve_env_node(default)["kind"]
            arg_seg = " ".join(_seg(self.src, a) for a in node.args)
            workload = bool(WORKLOAD_MARKER_RE.search(arg_seg))
            mentions_torch = self._mentions_torch(arg_seg)
            if not mentions_torch:
                alloc, alloc_basis = (
                    "no",
                    "the command arguments do not mention torch or cuda",
                )
            elif workload:
                alloc, alloc_basis = (
                    "yes",
                    "the wrapper launches a train/decode/sweep target",
                )
            else:
                alloc, alloc_basis = (
                    "conditional",
                    "the wrapper launches a child that may import torch",
                )
            self._add(
                kind="child_spawn",
                file=self.rel,
                line=node.lineno,
                source=seg.splitlines()[0][:400] if seg else "",
                spawn=f"call to `{f.id}()`",
                role="wrapper_call_site",
                wrapper=f.id,
                env_mode=env_mode,
                launches_workload=workload,
                allocates_context=alloc,
                allocates_context_basis=alloc_basis,
                runs_workload=workload,
                runs_workload_basis=(
                    "the arguments name a train/decode/sweep/bench target"
                    if workload
                    else "no train/decode/sweep marker in the arguments"
                ),
                gate=self._gate_label(),
            )

    def _cvd_set_guard(self, node: ast.AST) -> bool:
        """True when the enclosing function checks that CUDA_VISIBLE_DEVICES
        is ALREADY SET before writing it.

        ``molmetal/molmetal_lam/sbdd_env/training_utils.py`` writes the mask
        twice (lines 136 and 217), and both writes are preceded by an
        ``os.environ.get(CVD) is not None`` branch — so an explicitly-set
        empty mask is honoured verbatim.  A writer that is guarded this way
        does NOT defeat an ambient blank.
        """
        fname = self._enclosing_func(node)
        if fname is None:
            return False
        fnode = self._funcs.get(fname)
        if fnode is None:
            return False
        for sub in ast.walk(fnode):
            if not isinstance(sub, ast.Compare):
                continue
            if getattr(sub, "lineno", 0) >= getattr(node, "lineno", 0):
                continue
            seg = _seg(self.src, sub)
            if not re.search(r"is\s+not\s+None|not\s+in\s+os\.environ", seg):
                continue
            # The comparison may be against a local alias of the env read
            # (`cur = os.environ.get(CVD)` then `if cur is not None:`), so
            # also require the function to READ the variable.
            fseg = _seg(self.src, fnode)
            if not re.search(
                r"os\.environ\.get\(\s*[\"']" + CVD + r"[\"']", fseg
            ) and not re.search(
                r"[\"']" + CVD + r"[\"']\s+in\s+os\.environ", fseg
            ):
                continue
            return True
        return False

    def _scan_lock_writes(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if _cvd_target(tgt) and _is_os_environ(tgt.value):
                        val = _seg(self.src, node.value).strip()
                        effect = self._cvd_effect(val)
                        if effect == "dynamic" and self._cvd_set_guard(node):
                            effect = "guarded_by_existing_value"
                        self._add(
                            kind="session_lock",
                            file=self.rel,
                            line=node.lineno,
                            source=_seg(self.src, node).splitlines()[0][:400]
                            if _seg(self.src, node)
                            else "",
                            value_src=val[:200],
                            effect=effect,
                            allocates_context="no",
                            allocates_context_basis="an env write allocates "
                            "nothing; it decides whether a LATER probe does",
                            runs_workload=False,
                            runs_workload_basis="an env write runs nothing",
                            gate=self._gate_label(),
                        )

    def _scan_overrides(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                seg = _seg(self.src, node)
                m = DEVICE_OVERRIDE_RE.search(seg)
                if m and ("environ" in seg or "get(" in seg):
                    self._add(
                        kind="device_override",
                        file=self.rel,
                        line=node.lineno,
                        source=seg.splitlines()[0][:400] if seg else "",
                        env_name=m.group(0),
                        allocates_context="no",
                        allocates_context_basis="reading an env var allocates "
                        "nothing",
                        runs_workload=False,
                        runs_workload_basis="reading an env var runs nothing",
                        gate=self._gate_label(),
                    )

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _cvd_effect(value_src: str) -> str:
        s = value_src.strip()
        if s in ('""', "''", "u''", 'u""'):
            return "forces_blank"
        if re.fullmatch(r'["\'][^"\']*["\']', s):
            return "sets_nonblank"
        if "environ" in s:
            return "dynamic"
        return "dynamic"

    def _gate_label(self) -> str:
        base = os.path.basename(self.rel)
        if "/tests/" in self.rel or base.startswith("test_") or base.endswith("_test.py"):
            return "test"
        if self.rel.startswith("scripts/") and (
            base.startswith("launch_") or base == "start.sh"
        ):
            return "launcher"
        if GATE_NAME_RE.search(base):
            return "gate"
        return "production"


# ---------------------------------------------------------------------------
# The shell scanner
# ---------------------------------------------------------------------------
CVD_LINE_RE = re.compile(r"CUDA_VISIBLE_DEVICES")

#: Shell operators that introduce a parameter expansion, longest first so
#: that ``:-`` wins over ``-``.
_SHELL_PARAM_OPS = (":-", ":=", ":+", "##", "%%", "-", "+", "=", "#", "%")


def _strip_shell_comment(line: str) -> str:
    """Drop a trailing ``# comment`` (quote-aware).

    A comment may not contain the mask -- a comment that merely MENTIONS
    ``CUDA_VISIBLE_DEVICES`` (e.g. ``GPU="cuda:0"  # ... (CUDA_VISIBLE_DEVICES=0)``)
    does not write anything.  Counting it as a write is a false positive.
    """
    in_s = False
    in_d = False
    for i, ch in enumerate(line):
        if in_s:
            if ch == "'":
                in_s = False
            continue
        if in_d:
            if ch == '"':
                in_d = False
            continue
        if ch == "'":
            in_s = True
        elif ch == '"':
            in_d = True
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _quote_mask(s: str) -> List[bool]:
    """Per-character mask: True when the character is inside a quoted string."""
    mask = [False] * len(s)
    in_s = False
    in_d = False
    for i, ch in enumerate(s):
        if in_s:
            mask[i] = True
            if ch == "'":
                in_s = False
            continue
        if in_d:
            mask[i] = True
            if ch == '"':
                in_d = False
            continue
        if ch == "'":
            in_s = True
            mask[i] = True
        elif ch == '"':
            in_d = True
            mask[i] = True
    return mask


def _split_shell_param(v: str) -> Optional[Tuple[str, str, str]]:
    """Split ``${NAME<op>word}`` into ``(NAME, op, word)``.

    Handles NESTED braces in the word (``${VAR-${GPU}}``), so the
    ``${VAR-default}`` family is recognised correctly -- a flat regex
    cannot close a nested ``${...}``.
    """
    if not v.startswith("${"):
        return None
    depth = 0
    end = -1
    for i in range(2, len(v)):
        if v[i] == "{":
            depth += 1
        elif v[i] == "}":
            if depth == 0:
                end = i
                break
            depth -= 1
    if end < 0:
        return None
    inner = v[2:end]
    for op in _SHELL_PARAM_OPS:
        pos = inner.find(op)
        if pos > 0:
            name = inner[:pos]
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                continue
            return name, op, inner[pos + len(op):]
    return None


def _classify_shell_value(value: str) -> str:
    """Classify a shell ``CUDA_VISIBLE_DEVICES=<value>`` assignment.

    The distinction that matters is between the two parameter-expansion
    forms:

    ``${VAR-default}``  substitutes only when VAR is UNSET, so an ambient
                        EMPTY mask survives -> the lock is preserved.
    ``${VAR:-default}`` substitutes when VAR is unset OR EMPTY, so an
                        ambient EMPTY mask is REPLACED -> the lock is
                        defeated.
    """
    v = value.strip()
    if " " in v and not (v.startswith('"') or v.startswith("${")):
        v = v.split(" ")[0]
    v = v.strip('"').strip("'")
    if v == "":
        return "forces_blank"
    param = _split_shell_param(v)
    if param is not None:
        name, op, word = param
        if op in (":-", "=") and name == CVD:
            # ${VAR:-default} / ${VAR=default} — the default replaces an
            # EMPTY ambient -> the lock is defeated.
            return "defeats_lock"
        if op in ("-", ":=", ":+", "+"):
            # ${VAR-default} / ${VAR:=default} / ${VAR:+x} / ${VAR+x} —
            # an EMPTY ambient is still SET, so the empty mask survives.
            return "preserves_blank"
        if op in ("##", "#", "%%", "%"):
            # a string operation on ${CVD} itself -> derived from the
            # ambient value, so an empty ambient stays empty.
            return "preserves_blank" if name == CVD else "sets_nonblank_or_dynamic"
    if re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", v):
        return "passthrough"
    if v.startswith("$"):
        return "sets_nonblank_or_dynamic"
    return "sets_nonblank_or_dynamic"


def _shell_cvd_assignments(line: str) -> List[Tuple[str, str]]:
    """Return ``(value, scope)`` for each real mask WRITE in ``line``.

    ``scope`` is ``"export"`` (the whole script) or ``"command_prefix"``
    (that one command).  A mention of the variable inside a quoted string
    (an ``echo``/``log`` message) or inside a comment is NOT a write and
    is not reported.
    """
    out: List[Tuple[str, str]] = []
    body = _strip_shell_comment(line)
    if CVD not in body:
        return out
    mask = _quote_mask(body)
    start = 0
    while True:
        idx = body.find(CVD, start)
        if idx < 0:
            break
        start = idx + len(CVD)
        if mask[idx]:
            # inside a quoted string -> an argument, not an assignment
            continue
        j = idx + len(CVD)
        while j < len(body) and body[j] in " \t":
            j += 1
        if j >= len(body) or body[j] != "=":
            continue  # a bare reference (e.g. `export CUDA_VISIBLE_DEVICES`)
        if j + 1 < len(body) and body[j + 1] == "=":
            continue  # `==` is a comparison, not an assignment
        # the scope: look at the word before the token
        before = body[:idx].rstrip()
        scope = "command_prefix"
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", before)
        if m and m.group(1) in ("export", "declare", "typeset", "local", "readonly"):
            scope = "export"
        # the value: from after '=' up to the end of the word
        k = j + 1
        value = ""
        if k < len(body) and body[k] in "'\"":
            q = body[k]
            end_q = body.find(q, k + 1)
            if end_q < 0:
                end_q = len(body)
            value = body[k:end_q + 1]
        else:
            m2 = re.match(r"[^\s;]+", body[k:])
            value = m2.group(0) if m2 else ""
        out.append((value, scope))
    return out


def scan_shell(rel: str, text: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if not CVD_LINE_RE.search(line):
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assignments = _shell_cvd_assignments(line)
        if not assignments:
            continue
        value, scope = assignments[-1]
        effect = _classify_shell_value(value)
        if effect == "forces_blank":
            allocates = "no"
            basis = "an export of an empty mask forces the child to CPU"
        elif effect == "defeats_lock":
            allocates = "yes"
            basis = (
                "this line OVERWRITES the ambient CUDA_VISIBLE_DEVICES with a "
                "non-empty value, so a child launched after it sees a GPU even "
                "when the session blanked the mask"
            )
        elif effect == "sets_nonblank_or_dynamic":
            allocates = "yes"
            basis = (
                "this line sets CUDA_VISIBLE_DEVICES to a non-empty or "
                "dynamic value, overriding an ambient blank"
            )
        elif effect == "preserves_blank":
            allocates = "no"
            basis = (
                "the ${VAR-default} form substitutes only when VAR is UNSET, "
                "so an ambient blank survives"
            )
        else:
            allocates = "no"
            basis = "a reference to the variable does not change it"
        hits.append(
            {
                "kind": "session_lock",
                "file": rel,
                "line": idx,
                "source": line.strip()[:400],
                "value_src": value[:120],
                "effect": effect,
                "scope": scope,
                "n_assignments_on_line": len(assignments),
                "allocates_context": allocates,
                "allocates_context_basis": basis,
                "runs_workload": bool(
                    re.search(r"python|torchrun|setuid", line)
                ),
                "runs_workload_basis": "the line launches a python child"
                if re.search(r"python|torchrun", line)
                else "the line only sets an env var",
                "gate": "launcher"
                if os.path.basename(rel).startswith(("launch_", "start.sh"))
                else "production",
            }
        )
    return hits


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------
def build_census() -> Dict[str, Any]:
    files = enumerate_scope_files()
    py_files = [f for f in files if f["path"].endswith(".py")]
    sh_files = [f for f in files if f["path"].endswith(".sh")]

    hits: List[Dict[str, Any]] = []
    per_file: Dict[str, Dict[str, Any]] = {}
    parse_failures: List[Dict[str, Any]] = []
    files_read = 0
    files_unreadable = 0

    for entry in py_files:
        rel = entry["path"]
        path = REPO_ROOT / rel
        try:
            raw = path.read_bytes()
        except OSError as exc:
            files_unreadable += 1
            parse_failures.append({"file": rel, "error": f"unreadable: {exc}"})
            continue
        files_read += 1
        text = raw.decode("utf-8", "replace")
        src_lines = text.splitlines()
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            parse_failures.append(
                {"file": rel, "error": f"SyntaxError line {exc.lineno}: {exc.msg}"}
            )
            continue
        scan = PyFileScan(rel, src_lines, tree)
        if scan.hits:
            per_file[rel] = {
                "file_blanks_cvd_at_import": bool(
                    scan.module_level_cvd_write
                    and scan.module_level_cvd_write["value_src"].strip() in ('""', "''")
                ),
                "lock_check_lines": scan.lock_check_lines,
                "hit_count": len(scan.hits),
            }
            for h in scan.hits:
                h["file_blanks_cvd_at_import"] = per_file[rel][
                    "file_blanks_cvd_at_import"
                ]
                h["lock_check_before_hit"] = any(
                    ln < h["line"] for ln in scan.lock_check_lines
                )
                h["honours_session_lock"] = bool(
                    h.get("effect") == "forces_blank"
                    or h.get("env_mode") == "forces_blank"
                    or h.get("kind") == "device_override"
                    or h.get("file_blanks_cvd_at_import")
                    or h.get("lock_check_before_hit")
                )
                h["honours_session_lock_mechanisms"] = [
                    m
                    for m in (
                        "forces_blank"
                        if h.get("effect") == "forces_blank"
                        else None,
                        "forces_env_blank"
                        if h.get("env_mode") == "forces_blank"
                        else None,
                        "device_override" if h.get("kind") == "device_override" else None,
                        "file_blanks_cvd_at_import"
                        if h.get("file_blanks_cvd_at_import")
                        else None,
                        "checks_lock_before_probe"
                        if h.get("lock_check_before_hit")
                        else None,
                    )
                    if m
                ]
                hits.append(h)

    for entry in sh_files:
        rel = entry["path"]
        path = REPO_ROOT / rel
        try:
            raw = path.read_bytes()
        except OSError as exc:
            files_unreadable += 1
            parse_failures.append({"file": rel, "error": f"unreadable: {exc}"})
            continue
        files_read += 1
        text = raw.decode("utf-8", "replace")
        for h in scan_shell(rel, text):
            h["file_blanks_cvd_at_import"] = False
            h["lock_check_before_hit"] = False
            h["honours_session_lock"] = h.get("effect") == "forces_blank"
            h["honours_session_lock_mechanisms"] = (
                ["forces_blank"] if h.get("effect") == "forces_blank" else []
            )
            hits.append(h)

    # ---- summary / question B -----------------------------------------
    by_kind: Dict[str, int] = {}
    for h in hits:
        by_kind[h["kind"]] = by_kind.get(h["kind"], 0) + 1

    by_scope: Dict[str, Dict[str, int]] = {}
    for f in files:
        scope = classify_scope(f["path"])
        by_scope.setdefault(scope, {"files": 0, "hits": 0})
        by_scope[scope]["files"] += 1
    for h in hits:
        by_scope[classify_scope(h["file"])]["hits"] += 1

    lock_defeaters = [
        {
            "file": h["file"],
            "line": h["line"],
            "source": h["source"],
            "effect": h.get("effect") or h.get("env_mode"),
            "kind": h["kind"],
        }
        for h in hits
        if h.get("effect") == "defeats_lock"
        or h.get("effect") == "sets_nonblank_or_dynamic"
        or h.get("env_mode") == "sets_cvd_nonblank"
    ]
    lock_forcers = [
        {
            "file": h["file"],
            "line": h["line"],
            "source": h["source"],
            "kind": h["kind"],
        }
        for h in hits
        if h.get("effect") == "forces_blank" or h.get("env_mode") == "forces_blank"
    ]
    lock_guarded_writers = [
        {
            "file": h["file"],
            "line": h["line"],
            "source": h["source"],
            "effect": h.get("effect"),
        }
        for h in hits
        if h.get("effect") == "guarded_by_existing_value"
    ]
    inheriting_spawns = [
        {
            "file": h["file"],
            "line": h["line"],
            "source": h["source"],
            "launches_workload": h.get("launches_workload"),
        }
        for h in hits
        if h["kind"] == "child_spawn" and h.get("env_mode") == "inherits_parent_env"
    ]
    lock_droppers = [
        {
            "file": h["file"],
            "line": h["line"],
            "source": h["source"],
            "role": h.get("role"),
            "launches_workload": h.get("launches_workload"),
        }
        for h in hits
        if h["kind"] == "child_spawn" and h.get("env_mode") == "drops_lock"
    ]
    mask_respecting_probes = [
        {
            "file": h["file"],
            "line": h["line"],
            "source": h["source"],
            "api": h["api"],
            "honours_session_lock": h.get("honours_session_lock"),
        }
        for h in hits
        if h["kind"] == "cuda_probe"
        and h["api"] in ("is_available", "device_count")
    ]

    probes = [h for h in hits if h["kind"] == "cuda_probe"]
    probes_ctx = [h for h in probes if h["allocates_context"] == "yes"]
    probes_conditional = [h for h in probes if h["allocates_context"] == "conditional"]
    probes_mask_respecting = mask_respecting_probes
    probes_import_time = [
        h for h in probes if h.get("module_level")
    ]
    probes_import_time_ctx = [
        h
        for h in probes_import_time
        if h["allocates_context"] in ("yes", "conditional")
    ]

    workload_spawns = [
        h for h in hits if h["kind"] == "child_spawn" and h.get("launches_workload")
    ]

    return {
        "wf": "Wf-A163",
        "generated": "2026-11-01",
        "title": "GPU-tooling census: every repo path that can touch GPU memory "
        "when the session forbids it",
        "scanner": {
            "script": "molmetal/scripts/wfa163_gpu_tooling_census.py",
            "method": "git ls-files enumeration (NEVER os.walk) + Python AST "
            "for .py + regex for .sh; stdlib only, torch never imported",
            "python": sys.version.split()[0],
            "cwd": str(REPO_ROOT),
        },
        "denominator": {
            "scope_prefixes": list(SCOPE_PREFIXES),
            "files_enumerated_in_scope": len(files),
            "files_read": files_read,
            "files_unreadable": files_unreadable,
            "py_files": len(py_files),
            "sh_files": len(sh_files),
            "other_extension_files": len(files) - len(py_files) - len(sh_files),
            "files_scanned_per_category": {
                "cuda_probe": len(py_files),
                "device_autoselect": len(py_files),
                "child_spawn": len(py_files),
                "session_lock": len(py_files) + len(sh_files),
                "device_override": len(py_files),
            },
            "files_with_hits": len({h["file"] for h in hits}),
            "per_scope": by_scope,
            "parse_failures": parse_failures,
            "note": "files_enumerated_in_scope counts EVERY file under the "
            "scope prefixes (including .md/.json report archives that carry "
            "no executable GPU path); files_read is the number actually "
            "decoded and scanned (the .py + .sh set).  files_with_hits counts "
            "DISTINCT files carrying >=1 hit; per_scope['hits'] counts HITS.",
        },
        "summary": {
            "hits_total": len(hits),
            "hits_by_kind": by_kind,
            "cuda_probes_total": len(probes),
            "cuda_probes_that_allocate_a_context": len(probes_ctx),
            "cuda_probes_with_conditional_allocation": len(probes_conditional),
            "cuda_probes_that_respect_the_ambient_mask": len(
                probes_mask_respecting
            ),
            "cuda_probes_at_import_time": len(probes_import_time),
            "cuda_probes_at_import_time_that_allocate": len(
                probes_import_time_ctx
            ),
            "cuda_probes_that_create_a_primary_context": sum(
                1 for h in probes if h.get("creates_primary_context") == "yes"
            ),
            "cuda_probes_that_create_a_primary_context_at_import_time": sum(
                1
                for h in probes
                if h.get("creates_primary_context") == "yes" and h.get("module_level")
            ),
            "child_spawns_total": sum(
                1 for h in hits if h["kind"] == "child_spawn"
            ),
            "child_spawns_that_run_a_workload": len(workload_spawns),
            "child_spawns_that_inherit_the_parent_env": len(inheriting_spawns),
            "child_spawns_that_drop_the_lock": len(lock_droppers),
            "hits_honouring_the_session_lock": sum(
                1 for h in hits if h.get("honours_session_lock")
            ),
            "lock_defeaters": len(lock_defeaters),
            "lock_forcers": len(lock_forcers),
            "lock_guarded_writers": len(lock_guarded_writers),
            "by_gate": {
                g: sum(1 for h in hits if h.get("gate") == g)
                for g in sorted({h.get("gate", "?") for h in hits})
            },
            "by_env_mode": {
                m: sum(
                    1
                    for h in hits
                    if h["kind"] == "child_spawn" and h.get("env_mode") == m
                )
                for m in sorted(
                    {
                        h.get("env_mode")
                        for h in hits
                        if h["kind"] == "child_spawn"
                    }
                    - {None}
                )
            },
        },
        "question_b": {
            "question": "With CUDA_VISIBLE_DEVICES='' in the environment, can "
            "any of the repo's own tooling still allocate GPU memory?",
            "answer": "YES — but only via the paths that WRITE or DROP the "
            "mask. The probe-and-gate surfaces do NOT allocate.",
            "reasoning": [
                "A torch probe that respects the ambient mask "
                "(torch.cuda.is_available / device_count) returns 0 visible "
                "devices when CUDA_VISIBLE_DEVICES='' , so it selects cpu and "
                "creates no context. That is the mechanism the landed Wf-A162 "
                "pre-flight repair relies on.",
                "A child process inherits the parent's environment, so a "
                "spawn WITHOUT env= (or with env={**os.environ,...}) "
                "propagates the ambient blank into the child. The lock "
                "therefore holds for every gate that spawns a child that way "
                "-- but ONLY if the PARENT process had the blank.",
                "A spawn with an env= that neither mentions "
                "CUDA_VISIBLE_DEVICES nor inherits os.environ DROPS the lock: "
                "the child sees every device. See lock_droppers.",
                "A path that WRITES a non-empty CUDA_VISIBLE_DEVICES defeats "
                "the ambient blank outright. Every shell launcher that "
                "exports CUDA_VISIBLE_DEVICES=0 and scripts/start.sh (which "
                "derives the mask from the yaml compute.gpu field, default "
                "'cuda:0') falls in this class.",
            ],
            "lock_defeaters": lock_defeaters,
            "lock_forcers": lock_forcers,
            "lock_guarded_writers": lock_guarded_writers,
            "lock_droppers": lock_droppers,
            "inheriting_spawns": inheriting_spawns,
            "mask_respecting_probes": mask_respecting_probes,
        },
        "hits": hits,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "molmetal/reports/wfa163_gpu_tooling_census/gpu_tooling_census.json"
        ),
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    census = build_census()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO_ROOT / "molmetal/scripts"))
    try:
        from wf386a3_atomic_write import atomic_write_file  # type: ignore

        atomic_write_file(out_path, json.dumps(census, indent=2, ensure_ascii=False))
    except Exception:
        # Atomic write is the rule; a fallback that truncates in place is NOT
        # acceptable, so write to a sibling temp file and os.replace.
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(census, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out_path)

    if not args.quiet:
        s = census["summary"]
        d = census["denominator"]
        print(f"Wf-A163 GPU-tooling census -> {out_path}")
        print(
            f"  denominator: {d['files_enumerated_in_scope']} files in scope "
            f"({d['py_files']} .py + {d['sh_files']} .sh), "
            f"{d['files_read']} read, {d['files_unreadable']} unreadable"
        )
        print(f"  hits: {s['hits_total']}  by kind: {s['hits_by_kind']}")
        print(
            f"  cuda probes: {s['cuda_probes_total']} "
            f"({s['cuda_probes_that_allocate_a_context']} allocate a context, "
            f"{s['cuda_probes_with_conditional_allocation']} conditional)"
        )
        print(
            f"  child spawns: {s['child_spawns_total']} "
            f"({s['child_spawns_that_run_a_workload']} workload, "
            f"{s['child_spawns_that_inherit_the_parent_env']} inherit env)"
        )
        print(
            f"  lock: {s['lock_forcers']} forcers, {s['lock_defeaters']} "
            f"defeaters, {s['hits_honouring_the_session_lock']} hits honour it"
        )
        print(f"  by gate: {s['by_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
