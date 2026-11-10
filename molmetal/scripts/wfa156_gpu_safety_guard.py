#!/usr/bin/env python3
"""Wf-A156 — GPU safety guard: no pytest session may silently take VRAM.

THE DEFECT CLASS
----------------
Repo code selects its device with the idiom

    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

which fires the moment CUDA is VISIBLE, with no opt-in.  In a test or a
smoke path that is a violation of the standing 禁止使用gpu directive and of
AGENTS.md §5.1 ("VRAM budget check first"): the device is taken because it
happened to be there, not because anybody asked for it.

MEASURED (2026-10-31, quoted verbatim in ``molmetal/tests/conftest.py``): a
full ``pytest molmetal/tests`` run held 1444 MiB (the pytest process) plus
2218 MiB (a smoke subprocess) of GPU 0 VRAM.

WHAT THIS GUARD IS
------------------
A **static** guard.  It parses every Python file in the repo's TEST and
SMOKE surfaces, finds every device-selection site, and classifies it:

``auto_cuda``
    The site can pick CUDA with no opt-in.  THIS IS THE VIOLATION.
``guarded``
    The CUDA reference sits behind an explicit skip guard (a
    ``pytest.mark.skipif`` on the availability probe, an
    ``if not <available>: pytest.skip(...)`` in the enclosing function, an
    ``if <available>: <work> else: pytest.skip(...)``, or a monkeypatched
    probe), or it is a pure availability PROBE with no device consequence.
``routed``
    The site resolves its device through the canonical helper
    ``molmetal._device.resolve_device`` — which returns CPU in a test/smoke
    context unless the session explicitly opted in.
``waived``
    An explicitly declared exception, listed in :data:`WAIVERS` with a
    reason.  A waiver is VISIBLE in the report, never silent.
``production``
    Outside the TEST+SMOKE surface.  Reported for the census, never a
    violation.

Exit codes
----------
``0``
    Clean — no ``auto_cuda`` site in any TEST/SMOKE surface.
``1``
    At least one ``auto_cuda`` site was found in a TEST/SMOKE surface
    (they are listed).  A PRODUCTION auto-CUDA site is a legitimate
    opt-in surface (AGENTS.md §5.1): it is reported as a census but does
    NOT fail the guard.
``2``
    Skip / error (the root is not a directory, or a file could not be
    parsed).  Never 0 on an error.

CPU-ONLY
--------
This module imports **stdlib only** — it never imports torch, never touches
CUDA, and never runs a subprocess that could.  It is safe to run inside a
CPU-locked pytest session.

USAGE
-----
::

    CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa156_gpu_safety_guard.py
    CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa156_gpu_safety_guard.py \
        --root /tmp/synthetic-tree --json /tmp/report.json

RULES (and their limits — read before trusting a "clean" verdict)
-----------------------------------------------------------------
The classification is AST-based, with a small number of documented
heuristics.  They are stated here so a verifier can refute them:

1. **Device sinks.**  A ternary that yields a bare device *string* counts as
   a device selection only when the string is bound to a device-shaped name
   (``device`` / ``devices`` / ``dev`` / ``DEVICE`` / …) or is consumed by a
   device sink (a ``device=`` / ``devices=`` keyword, ``.to(...)``,
   ``torch.device(...)``).  A ternary that yields a string which is only
   COMPARED (e.g. ``expected = "cuda" if ... else "cpu"``) is a probe, not a
   selection, and is classified ``guarded``.  A ternary that yields a
   ``torch.device(...)`` OBJECT is always a selection.
2. **Local helpers.**  A module-level helper (or pytest fixture) that
   selects a device from an availability probe is classified by its CALL
   SITES: if every call site is inside a guarded function, the helper is
   ``guarded``; otherwise it is ``auto_cuda``.  A pytest FIXTURE is never
   excused this way — a fixture that hands a device to a test is a
   selection.
3. **Guarded functions.**  A function is guarded when it carries a
   ``pytest.mark.skipif`` decorator whose condition mentions the
   availability probe (directly, through a module-level alias such as
   ``requires_cuda``, or through a string condition), when its body
   contains ``if not <available>: pytest.skip(...)`` / ``raise``, when its
   body contains ``if <available>: <work> else: pytest.skip(...)``, or
   when it monkeypatches the availability probe away.
4. **Scope.**  ``molmetal/references/`` (vendored third-party clones) and
   ``molmetal/reports/`` (archived report snapshots) are NOT scanned.  They
   are counted in the census and reported as ``vendored`` / ``archive``.
5. **Launch entry points.**  A file under ``molmetal/scripts/`` or
   ``scripts/`` is a LAUNCH ENTRY POINT: a deliberate opt-in surface, even
   when its name contains "smoke".  The four scripts in
   :data:`SMOKE_SELFCHECK_SCRIPTS` are imported BY a test module and are
   therefore classified ``smoke``.
6. **Routed means routed.**  A call is classified ``routed`` only when the
   module imports the callee FROM ``molmetal._device``.  Two production
   launch entry points (``scripts/generate.py`` and ``scripts/train.py``)
   define their own ``resolve_device(requested)`` with a different
   signature; calling one of those is not routing and is not reported as
   such.

Cross-references
----------------
* ``molmetal/_device.py`` — the canonical CPU-safe resolver.
* ``molmetal/tests/conftest.py`` + ``conftest.py`` (repo root) — the
  runtime CPU lock (env-var level).  This guard is the STATIC counterpart:
  the lock depends on the session honouring the environment, this one
  depends only on the source text.
* ``molmetal/tests/test_wfa156_gpu_safety_guard.py`` — the contract test.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

GUARD_NAME = "wfa156_gpu_safety_guard"
GUARD_VERSION = "1.1.0"

# ---------------------------------------------------------------------------
# Scan scope
# ---------------------------------------------------------------------------

#: Directory names that are never descended into.  ``checkpoints`` and
#: ``.data`` hold hundreds of GB of binary blobs; ``references`` is
#: vendored third-party source; ``reports`` is an archive of report
#: snapshots.  None of them is a live TEST/SMOKE surface.
PRUNE_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".ipynb_checkpoints",
        "node_modules",
        ".venv",
        "venv",
        "checkpoints",
        ".data",
        "references",
        "reports",
    }
)

#: Extra top-level scan roots, relative to the repo root.
SCAN_ROOTS = ("molmetal", "scripts", "tests")

#: Directories whose files are LAUNCH ENTRY POINTS (a deliberate opt-in
#: surface) rather than a test surface.
LAUNCH_ENTRY_POINT_DIRS = ("molmetal/scripts", "scripts")

#: Smoke scripts that are IMPORTED by a test module — a test surface, not a
#: launch entry point.  (Verified by grepping the test trees for imports.)
SMOKE_SELFCHECK_SCRIPTS = frozenset(
    {
        "molmetal/scripts/wf329w5_w2_chemprop_smoke.py",
        "molmetal/scripts/wf399a1_smoke_protocol.py",
        "molmetal/scripts/wf404a5_wf29_attr_fix_smoke.py",
        "molmetal/scripts/wf409a4_wf29_extended_smoke.py",
    }
)

#: The grep-equivalent patterns this guard implements.  Kept as data so the
#: report can name the pattern that matched (the brief asks for
#: "per file:line, the pattern, the classification").
PATTERNS: Dict[str, str] = {
    "availability_probe": r"cuda\.is_available\(\)",
    "startswith_check": r"device\.startswith\(\s*[\"']cuda",
    "cuda_literal": r"torch\.device\(\s*[\"']cuda",
    "auto_select": r"[\"']cuda(:\d+)?[\"']\s*if\s*.*cuda\.is_available\(\)",
    "gpu_first_helper": r"(get_device|resolve_default_device|detect_training_gpu)",
    "routed": r"resolve_device\(",
}

#: Helpers that resolve a device **GPU-first** (they return ``cuda:0``
#: whenever a CUDA device is visible).  They are NOT safe to call from a
#: test/smoke surface: ``molmetal/utils/device.py::get_device`` and
#: ``molmetal/molmetal_lam/sbdd_env/training_utils.py::detect_training_gpu``
#: both allocate a real tensor on ``cuda:0``.
GPU_FIRST_HELPERS = frozenset(
    {
        "get_device",
        "resolve_default_device",
        "setup_training_device_env",
        "detect_training_gpu",
        "detect_active_gpu",
    }
)

#: Module-level attributes that are GPU-first device descriptors.
GPU_FIRST_ATTRIBUTES = frozenset({"DEFAULT_DEVICE"})

#: The canonical CPU-safe resolver, and the module it lives in.
CANONICAL_RESOLVER = "resolve_device"
CANONICAL_RESOLVER_MODULE = "molmetal._device"
#: Every public name of the canonical resolver module that resolves a
#: device.  A call to one of these is "routed" ONLY when the module
#: actually imports it from :data:`CANONICAL_RESOLVER_MODULE` — a local
#: function that happens to share the name (``scripts/generate.py`` and
#: ``scripts/train.py`` both define their own ``resolve_device(requested)``)
#: is NOT routed.
CANONICAL_RESOLVER_NAMES = frozenset({CANONICAL_RESOLVER, "cuda_device_indices"})

#: Names that are treated as device-shaped when a ternary yields a bare
#: device string (see rule 1 in the module docstring).
DEVICE_NAMES = frozenset(
    {"device", "devices", "dev", "devs", "_device", "_devices", "target_device"}
)

#: Keyword arguments that are device sinks.
DEVICE_SINK_KEYWORDS = frozenset({"device", "devices", "device_type"})

# ---------------------------------------------------------------------------
# Waivers — explicit, visible, reasoned
# ---------------------------------------------------------------------------

#: ``rel_path`` -> list of ``(matcher, reason)``.
#:
#: ``matcher`` is EITHER
#:
#: * a substring that must appear on the SITE's own source line, OR
#: * ``"@L1,L2,L3"`` — the exact line numbers the waiver covers.
#:
#: Every waiver is echoed in the report with its reason, so a waiver is
#: VISIBLE, never silent.  A waiver says "this site does not take VRAM";
#: it never says "this site is fine to ignore".
WAIVERS: Dict[str, List[Tuple[str, str]]] = {
    "molmetal/tests/test_wf328w3_bf16_production.py": [
        (
            "never used (no compute)",
            "constructs a cuda:0 device STRING to assert the BF16 factory "
            "accepts a CUDA device; the object is never used for compute and "
            "allocates no VRAM (the source comment says so).",
        ),
    ],
    "molmetal/tests/test_gpu_wrappers.py": [
        (
            "diffdock_pose_batch(",
            "the diffdock wrapper is monkeypatched to report NO CUDA backend, "
            "so it raises AdapterUnavailable before any tensor is created; the "
            "device string is the value under test, not a placement.",
        ),
    ],
    "molmetal/tests/test_cfm_cuda_mem_limit.py": [
        (
            '_apply_cuda_mem_limit(torch.device("cuda"))',
            "the helper returns before touching CUDA when the cap env var is "
            "unset / 0 / non-numeric — which is the case in these three tests "
            "(the CUDA-capping test itself is skipif-guarded).",
        ),
    ],
    "molmetal/tests/test_clone_integration_adapters.py": [
        (
            'stub.setup(device="cuda")',
            "the argument is handed to a STUB whose setup() records the "
            "value; no tensor is created on it.",
        ),
    ],
    "molmetal/tests/test_amp_context.py": [
        (
            'expected = "cuda" if torch.cuda.is_available() else "cpu"',
            "computes the EXPECTED device_type STRING to assert the AMP helper "
            "agrees with the host; the value is only compared, never used to "
            "place a tensor — a probe, not a selection.",
        ),
        (
            'device="cuda:1"',
            "constructs an AMP context OBJECT with enabled=False to prove the "
            "device string is parsed, not used to place a tensor.",
        ),
    ],
    "molmetal/tests/test_cfm_run_fingerprint_guard.py": [
        (
            'device="cuda:0")',
            "a CONFIG DICT handed to a mocked trainer; the value is "
            "fingerprinted, never used to allocate.",
        ),
    ],
    "molmetal/tests/test_wf390a2_alt_encoder.py": [
        (
            'device="cuda")',
            "the smoke scaffold returns a status dict and echoes the device "
            "string; it allocates nothing on it (verified in "
            "molmetal/scripts/wf390a2_alt_encoder_audit.py::smoke_alt_encoder).",
        ),
    ],
    "molmetal/tests/test_wf326d_amp_batch_optimization.py": [
        (
            "@137,155,195",
            "the torch_device FIXTURE resolves the device through "
            "molmetal._device.resolve_device(); in a CPU-locked session it "
            "yields CPU, the guard `if torch_device.type != \"cuda\": "
            "pytest.skip(...)` skips, and no CUDA context is created.  The "
            "static rule cannot see through the fixture, so the waiver is "
            "stated here rather than left implicit.",
        ),
    ],
    "molmetal/molmetal_lam/tests/test_device_selection.py": [
        (
            "detect_active_gpu",
            "this module IS the unit test of the device-selection helper: "
            "every test monkeypatches detect_active_gpu (or "
            "torch.cuda.is_available), so the allocation probe is mocked and "
            "no CUDA context is created.",
        ),
        (
            "get_device()",
            "same module: the probe behind get_device() is monkeypatched in "
            "every test, so no CUDA context is created.",
        ),
    ],
}

# ---------------------------------------------------------------------------
# Site model
# ---------------------------------------------------------------------------


@dataclass
class Site:
    """One device-selection site."""

    file: str
    line: int
    rule: str
    surface: str
    classification: str
    pattern: str
    text: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "rule": self.rule,
            "surface": self.surface,
            "classification": self.classification,
            "pattern": self.pattern,
            "text": self.text,
            "reason": self.reason,
        }


@dataclass
class ScanResult:
    sites: List[Site] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------


def _dotted(node: ast.AST) -> str:
    """``a.b.c`` for a Name/Attribute chain, else ``""``."""
    parts: List[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _attr_names(node: ast.AST) -> List[str]:
    """Every attribute name in the chain, whatever the base is.

    ``__import__("torch").cuda.is_available`` -> ``["cuda", "is_available"]``.
    """
    names: List[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        names.append(cur.attr)
        cur = cur.value
    return names


def _is_avail_ref(node: ast.AST) -> bool:
    """Is ``node`` the attribute ``<...>.cuda.is_available``?"""
    if not (isinstance(node, ast.Attribute) and node.attr == "is_available"):
        return False
    return "cuda" in _attr_names(node.value)


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_cuda_string(node: ast.AST) -> bool:
    """Is ``node`` a string literal that names a CUDA device?"""
    value = _const_str(node)
    return bool(value) and value.startswith("cuda")


def _is_cuda_device_object(node: ast.AST) -> bool:
    """Is ``node`` a ``torch.device(...)`` call naming a CUDA device?"""
    if not isinstance(node, ast.Call):
        return False
    if not _dotted(node.func).endswith("torch.device"):
        return False
    if not node.args:
        return False
    return _is_cuda_value(node.args[0])


def _is_cuda_value(node: ast.AST) -> bool:
    """Does ``node`` evaluate to a CUDA device (object, string or index)?"""
    if _is_cuda_string(node):
        return True
    if _is_cuda_device_object(node):
        return True
    if isinstance(node, ast.Call):
        dotted = _dotted(node.func)
        if "cuda.current_device" in dotted:
            return True
        if dotted.endswith("torch.device"):
            return bool(node.args) and _is_cuda_value(node.args[0])
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_is_cuda_value(elt) for elt in node.elts)
    if isinstance(node, ast.IfExp):
        return _is_cuda_value(node.body) or _is_cuda_value(node.orelse)
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(v, ast.Constant)
            and isinstance(v.value, str)
            and v.value.startswith("cuda")
            for v in node.values
        )
    return False


def _parent_map(tree: ast.AST) -> Dict[int, ast.AST]:
    parents: Dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


# ---------------------------------------------------------------------------
# Module analysis
# ---------------------------------------------------------------------------


class _ModuleAnalyzer:
    """Classify the device-selection sites of one Python module."""

    def __init__(self, rel_path: str, source: str) -> None:
        self.rel_path = rel_path
        self.source = source
        self.lines = source.splitlines()
        self.tree = ast.parse(source, filename=rel_path)
        self.parents = _parent_map(self.tree)
        self.local_funcs: Dict[str, ast.AST] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.local_funcs.setdefault(node.name, node)
        self.skipif_aliases = self._collect_skipif_aliases()
        self.canonical_resolver_names = self._collect_canonical_resolver_names()
        self.guarded_funcs: Set[int] = {
            id(fn)
            for fn in self.local_funcs.values()
            if self._is_gpu_guarded(fn)
        }

    # -- helpers ---------------------------------------------------------

    def _collect_canonical_resolver_names(self) -> Set[str]:
        """Which canonical-resolver names this module imports, and from where.

        Only a name imported FROM ``molmetal._device`` counts.  Two
        production launch entry points (``scripts/generate.py`` and
        ``scripts/train.py``) define their OWN ``resolve_device(requested)``
        with a different signature; calling one of those is NOT "routed
        through the canonical resolver" and must not be reported as such.
        """
        imported: Set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module != CANONICAL_RESOLVER_MODULE:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        imported |= set(CANONICAL_RESOLVER_NAMES)
                    else:
                        imported.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == CANONICAL_RESOLVER_MODULE:
                        imported |= set(CANONICAL_RESOLVER_NAMES)
        return imported & CANONICAL_RESOLVER_NAMES

    def _collect_skipif_aliases(self) -> Set[str]:
        """Module-level names bound to a ``pytest.mark.skipif(...)``."""
        aliases: Set[str] = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None or not isinstance(value, ast.Call):
                continue
            if not _dotted(value.func).endswith("pytest.mark.skipif"):
                continue
            if not any(self._mentions_availability(arg) for arg in value.args):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    aliases.add(target.id)
        return aliases

    def _mentions_availability(self, node: ast.AST) -> bool:
        """Does ``node`` (an expression or a string condition) mention the probe?"""
        for sub in ast.walk(node):
            if _is_avail_ref(sub):
                return True
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if "cuda.is_available" in sub.value.replace(" ", ""):
                    return True
        return False

    def _is_gpu_guarded(self, fn: ast.AST) -> bool:
        """Is ``fn`` guarded against taking the GPU?"""
        for dec in getattr(fn, "decorator_list", []) or []:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "skipif":
                if any(
                    self._is_avail_expr(a) or self._mentions_availability(a)
                    for a in dec.args
                ):
                    return True
                if any(
                    self._is_avail_expr(kw.value) or self._mentions_availability(kw.value)
                    for kw in dec.keywords
                ):
                    return True
                continue
            if isinstance(target, ast.Name) and target.id in self.skipif_aliases:
                return True
            if isinstance(target, ast.Attribute) and target.attr in self.skipif_aliases:
                return True
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and self._is_guard_condition(node.test):
                # ``if not <available>: pytest.skip(...)``  /  ``if <available>:
                # pytest.skip(...)`` — either polarity, as long as the body
                # refuses to run.
                if self._body_blocks(node.body):
                    return True
                # ``if <available>: <work> else: pytest.skip(...)``
                if node.orelse and self._body_blocks(node.orelse):
                    return True
            if self._mocks_probe(node):
                return True
        return False

    def _mocks_probe(self, node: ast.AST) -> bool:
        """Does this node monkeypatch the availability probe away?"""
        if not isinstance(node, ast.Call):
            return False
        if not _dotted(node.func).endswith("setattr"):
            return False
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            value = _const_str(arg)
            if value and "is_available" in value:
                return True
            if isinstance(arg, ast.Attribute) and arg.attr == "is_available":
                return True
        return False

    @staticmethod
    def _body_blocks(body: Sequence[ast.AST]) -> bool:
        """Does this body SKIP or RAISE (i.e. refuse to run without a GPU)?"""
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Raise):
                    return True
                if isinstance(node, ast.Call):
                    dotted = _dotted(node.func)
                    if dotted.endswith("pytest.skip") or dotted.endswith("skip"):
                        return True
        return False

    def _is_fixture(self, fn: ast.AST) -> bool:
        for dec in getattr(fn, "decorator_list", []) or []:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "fixture":
                return True
            if isinstance(target, ast.Name) and target.id == "fixture":
                return True
        return False

    def _helper_is_availability_driven(self, name: str) -> bool:
        """Does the local helper / fixture ``name`` probe CUDA availability?"""
        fn = self.local_funcs.get(name)
        if fn is None:
            return False
        for node in ast.walk(fn):
            if _is_avail_ref(node):
                return True
        return False

    def _is_avail_expr(self, node: ast.AST) -> bool:
        """Is ``node`` an availability probe (directly or via a local helper)?"""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._is_avail_expr(node.operand)
        if _is_avail_ref(node):
            return True
        if isinstance(node, ast.Call):
            if _is_avail_ref(node.func):
                return True
            if isinstance(node.func, ast.Name):
                return self._helper_is_availability_driven(node.func.id)
        if isinstance(node, ast.Name):
            return self._helper_is_availability_driven(node.id)
        return False

    def _is_guard_condition(self, node: ast.AST) -> bool:
        """Is ``node`` a COMPLETE guard on GPU availability?

        A guard is complete only when its condition is a SINGLE availability
        test (``torch.cuda.is_available()``, a local helper that probes it,
        or a ``torch.cuda.device_count()`` comparison) — possibly negated.

        A CONJUNCTION is NOT a complete guard, because the guarded code can
        still reach a GPU when the other operand holds.  The canonical
        example, measured in this repo::

            if _TRITON_AGGREGATE is not None and not torch.cuda.is_available():
                pytest.skip(...)

        skips only when Triton is LOADED and no GPU is visible; with Triton
        absent the body still runs and its
        ``"cuda" if torch.cuda.is_available() else "cpu"`` picks CUDA.
        """
        if isinstance(node, ast.BoolOp):
            # A conjunction (or a disjunction) is a PARTIAL guard.
            return False
        if self._is_avail_expr(node):
            return True
        if isinstance(node, ast.Compare):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    dotted = _dotted(sub.func)
                    if dotted.endswith("cuda.device_count"):
                        return True
        return False

    def _mentions_gpu(self, node: ast.AST) -> bool:
        """Does ``node`` mention the GPU at all?"""
        for sub in ast.walk(node):
            if _is_avail_ref(sub):
                return True
            if isinstance(sub, ast.Call):
                dotted = _dotted(sub.func)
                if dotted.endswith("cuda.device_count"):
                    return True
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if sub.value.startswith("cuda"):
                    return True
        return False

    def _call_sites(self, name: str) -> List[ast.Call]:
        out: List[ast.Call] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == name:
                    out.append(node)
        return out

    def _enclosing_func(self, node: ast.AST) -> Optional[ast.AST]:
        cur = self.parents.get(id(node))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = self.parents.get(id(cur))
        return None

    def _inside_skipping_else(self, node: ast.AST) -> bool:
        """Is the site inside ``if <available>: <work> else: pytest.skip(...)``?"""
        cur = self.parents.get(id(node))
        while cur is not None:
            if isinstance(cur, ast.If):
                if (
                    self._is_guard_condition(cur.test)
                    and cur.orelse
                    and self._body_blocks(cur.orelse)
                ):
                    return True
            cur = self.parents.get(id(cur))
        return False

    def _scope_guarded(self, node: ast.AST, _seen: Optional[Set[int]] = None) -> bool:
        """Is the site inside a GPU-guarded scope?"""
        seen = _seen if _seen is not None else set()
        if self._inside_skipping_else(node):
            return True
        fn = self._enclosing_func(node)
        if fn is None:
            return False
        if id(fn) in self.guarded_funcs:
            return True
        if self._is_fixture(fn):
            return False
        name = getattr(fn, "name", None)
        if not name or id(fn) in seen:
            return False
        seen.add(id(fn))
        sites = self._call_sites(name)
        if not sites:
            return False
        for call in sites:
            if not self._scope_guarded(call, seen):
                return False
        return True

    # -- device-sink analysis -------------------------------------------

    def _is_device_sink_usage(self, name_node: ast.Name) -> bool:
        parent = self.parents.get(id(name_node))
        if parent is None:
            return False
        if isinstance(parent, ast.keyword) and parent.arg in DEVICE_SINK_KEYWORDS:
            return True
        if isinstance(parent, ast.Call):
            if isinstance(parent.func, ast.Attribute) and parent.func.attr == "to":
                if any(child is name_node for child in parent.args):
                    return True
            if _dotted(parent.func).endswith("torch.device"):
                if any(child is name_node for child in parent.args):
                    return True
        return False

    def _name_is_device_shaped(self, name_node: ast.Name) -> bool:
        ident = name_node.id
        if ident.lower() in DEVICE_NAMES:
            return True
        return bool(ident) and ident.upper() == ident and "DEV" in ident.upper()

    def _string_flows_to_device_sink(self, target: ast.Name) -> bool:
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Name) or node.id != target.id:
                continue
            if node is target:
                continue
            if self._is_device_sink_usage(node):
                return True
        return False

    # -- rules -----------------------------------------------------------

    def _text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()
        return ""

    def _surface(self) -> str:
        parts = self.rel_path.split("/")
        if "tests" in parts:
            return "test"
        if self.rel_path in SMOKE_SELFCHECK_SCRIPTS:
            return "smoke"
        if self.rel_path.startswith(LAUNCH_ENTRY_POINT_DIRS):
            return "production"
        name = parts[-1]
        if "smoke" in name.lower():
            return "smoke"
        return "production"

    def _waiver_for(self, text: str, lineno: int) -> Optional[str]:
        """The waiver that covers this site, if any.

        A waiver matches when its matcher is a substring of the SITE's own
        source line, or when the matcher is an explicit ``@L1,L2`` line list
        that contains the site's line.
        """
        for entries in WAIVERS.get(self.rel_path, []):
            matcher, reason = entries
            if matcher.startswith("@"):
                try:
                    lines = {int(x) for x in matcher[1:].split(",") if x}
                except ValueError:
                    continue
                if lineno in lines:
                    return reason
                continue
            if matcher in text:
                return reason
        return None

    def _emit(
        self,
        node: ast.AST,
        rule: str,
        classification: str,
        reason: str = "",
    ) -> Optional[Site]:
        lineno = getattr(node, "lineno", 0)
        text = self._text(lineno)
        waiver = self._waiver_for(text, lineno)
        if waiver is not None:
            classification = "waived"
            reason = waiver
        return Site(
            file=self.rel_path,
            line=lineno,
            rule=rule,
            surface=self._surface(),
            classification=classification,
            pattern=PATTERNS.get(rule, rule),
            text=text[:200],
            reason=reason,
        )

    def analyze(self) -> List[Site]:
        sites: List[Site] = []
        consumed: Set[int] = set()
        used_avail: Set[int] = set()

        # --- RULE 1: availability-driven selection (ternary) --------------
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.IfExp):
                continue
            if not self._is_avail_expr(node.test):
                continue
            if not (_is_cuda_value(node.body) or _is_cuda_value(node.orelse)):
                continue
            consumed.add(id(node))
            for sub in ast.walk(node.test):
                if _is_avail_ref(sub):
                    used_avail.add(id(sub))
            selection = self._ternary_is_selection(node)
            if selection:
                classification = (
                    "guarded" if self._scope_guarded(node) else "auto_cuda"
                )
                reason = (
                    "availability-driven device selection with no opt-in"
                    if classification == "auto_cuda"
                    else "availability-driven device selection inside a "
                    "GPU-guarded scope"
                )
            else:
                classification = "guarded"
                reason = (
                    "availability probe; the value is never used to place a tensor"
                )
            site = self._emit(node, "auto_select", classification, reason)
            if site is not None:
                sites.append(site)

        # --- RULE 2: hard-coded CUDA device literal ------------------------
        for node in ast.walk(self.tree):
            if id(node) in consumed:
                continue
            if not self._is_cuda_literal_site(node):
                continue
            consumed.add(id(node))
            classification = (
                "guarded" if self._scope_guarded(node) else "auto_cuda"
            )
            reason = (
                "hard-coded CUDA device outside a GPU-guarded scope"
                if classification == "auto_cuda"
                else "hard-coded CUDA device inside a GPU-guarded scope"
            )
            site = self._emit(node, "cuda_literal", classification, reason)
            if site is not None:
                sites.append(site)

        # --- RULE 3: GPU-first helper call --------------------------------
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted(node.func)
            leaf = dotted.rsplit(".", 1)[-1] if dotted else ""
            if leaf not in GPU_FIRST_HELPERS and leaf not in GPU_FIRST_ATTRIBUTES:
                continue
            if leaf == CANONICAL_RESOLVER:
                continue
            consumed.add(id(node))
            classification = (
                "guarded" if self._scope_guarded(node) else "auto_cuda"
            )
            reason = (
                f"GPU-first device helper '{leaf}()' returns cuda:0 whenever a "
                "CUDA device is visible"
            )
            site = self._emit(node, "gpu_first_helper", classification, reason)
            if site is not None:
                sites.append(site)

        # --- RULE 4: routed through the canonical resolver -----------------
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted(node.func)
            leaf = dotted.rsplit(".", 1)[-1] if dotted else ""
            if leaf not in CANONICAL_RESOLVER_NAMES:
                continue
            if leaf not in self.canonical_resolver_names:
                # A same-named LOCAL function (scripts/generate.py,
                # scripts/train.py) — not the canonical resolver.
                continue
            site = self._emit(
                node,
                "routed",
                "routed",
                f"device resolved through {CANONICAL_RESOLVER_MODULE}.{leaf} "
                "(CPU by default in a test/smoke context)",
            )
            if site is not None:
                sites.append(site)

        # --- RULE 5: bare availability probe -------------------------------
        for node in ast.walk(self.tree):
            if not _is_avail_ref(node):
                continue
            if id(node) in used_avail:
                continue
            parent = self.parents.get(id(node))
            if isinstance(parent, ast.Call) and parent.func is node:
                if id(parent) in consumed:
                    continue
            used_avail.add(id(node))
            site = self._emit(
                node,
                "availability_probe",
                "guarded",
                "availability probe (no device consequence on this line)",
            )
            if site is not None:
                sites.append(site)

        # --- RULE 6: ``device.startswith("cuda")`` -------------------------
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted(node.func)
            if not dotted.endswith("startswith"):
                continue
            if not node.args:
                continue
            if not _is_cuda_string(node.args[0]):
                continue
            consumed.add(id(node))
            site = self._emit(
                node,
                "startswith_check",
                "guarded",
                "device-string check, not a selection",
            )
            if site is not None:
                sites.append(site)

        sites.sort(key=lambda s: (s.file, s.line, s.rule))
        return sites

    # -- helpers used by the rules ---------------------------------------

    def _ternary_is_selection(self, node: ast.IfExp) -> bool:
        """Does this ternary SELECT a device, or merely COMPUTE a string?"""
        branch = node.body if _is_cuda_value(node.body) else node.orelse
        if _is_cuda_device_object(branch):
            return True
        # A device index list (``[torch.cuda.current_device()]``) is a selection.
        if isinstance(branch, (ast.List, ast.Tuple, ast.Set)):
            return True
        # A bare string: a selection only if it is bound to a device-shaped
        # name or consumed by a device sink.
        target = self._assignment_target(node)
        if target is None:
            fn = self._enclosing_func(node)
            name = getattr(fn, "name", "") if fn is not None else ""
            return bool(name) and (
                name.lower() in DEVICE_NAMES or "device" in name.lower()
            )
        if self._name_is_device_shaped(target):
            return True
        return self._string_flows_to_device_sink(target)

    def _assignment_target(self, node: ast.IfExp) -> Optional[ast.Name]:
        parent = self.parents.get(id(node))
        while parent is not None:
            if isinstance(parent, ast.Assign):
                if len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
                    return parent.targets[0]
                return None
            if isinstance(parent, (ast.AnnAssign, ast.NamedExpr)):
                if isinstance(parent.target, ast.Name):
                    return parent.target
                return None
            if isinstance(
                parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                return None
            parent = self.parents.get(id(parent))
        return None

    def _is_cuda_literal_site(self, node: ast.AST) -> bool:
        """Is ``node`` a CUDA device literal in a device position?"""
        if not isinstance(node, ast.Call):
            return False
        dotted = _dotted(node.func)
        if dotted.endswith("torch.device"):
            if not node.args:
                return False
            # A ternary inside ``torch.device(...)`` is RULE 1's business.
            if isinstance(node.args[0], ast.IfExp):
                return False
            return _is_cuda_value(node.args[0])
        if dotted.endswith(".to") and node.args:
            if isinstance(node.args[0], ast.IfExp):
                return False
            return _is_cuda_value(node.args[0])
        for kw in node.keywords:
            if kw.arg not in DEVICE_SINK_KEYWORDS:
                continue
            if isinstance(kw.value, ast.IfExp):
                # RULE 1 owns the ternary.
                continue
            if _is_cuda_value(kw.value):
                return True
        return False


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield every ``*.py`` file under ``root``, pruning the heavy trees.

    Symlinked DIRECTORIES are not followed (the vendored clones are reached
    through symlinks and following them can loop).
    """
    stack: List[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    if entry.is_file() and entry.suffix == ".py":
                        yield entry
                    continue
                if entry.is_dir():
                    if entry.name in PRUNE_DIR_NAMES:
                        continue
                    stack.append(entry)
                    continue
                if entry.suffix == ".py" and entry.is_file():
                    yield entry
            except OSError:
                continue


def discover_files(root: Path) -> List[Path]:
    """All candidate Python files for the scan."""
    files: List[Path] = []
    seen: Set[Path] = set()
    for scan_root in SCAN_ROOTS:
        base = root / scan_root
        if not base.is_dir():
            continue
        for path in iter_python_files(base):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    root_conftest = root / "conftest.py"
    if root_conftest.is_file():
        files.append(root_conftest)
    return files


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def scan_tree(root: Path) -> ScanResult:
    """Scan ``root`` and return every device-selection site it finds."""
    result = ScanResult()
    if not root.is_dir():
        result.errors.append({"path": str(root), "error": "root is not a directory"})
        return result
    for path in discover_files(root):
        rel = relative_to_root(path, root)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.errors.append({"path": rel, "error": f"unreadable: {exc}"})
            result.files_skipped += 1
            continue
        result.files_scanned += 1
        try:
            analyzer = _ModuleAnalyzer(rel, source)
        except SyntaxError as exc:
            result.errors.append({"path": rel, "error": f"syntax error: {exc}"})
            result.files_skipped += 1
            continue
        except (ValueError, RecursionError) as exc:  # pragma: no cover
            result.errors.append({"path": rel, "error": f"unparsable: {exc}"})
            result.files_skipped += 1
            continue
        result.sites.extend(analyzer.analyze())
    return result


def build_report(root: Path, result: ScanResult) -> Dict[str, Any]:
    by_classification: Dict[str, int] = {}
    by_surface: Dict[str, int] = {}
    for site in result.sites:
        by_classification[site.classification] = (
            by_classification.get(site.classification, 0) + 1
        )
        by_surface[site.surface] = by_surface.get(site.surface, 0) + 1
    auto = [s for s in result.sites if s.classification == "auto_cuda"]
    routed = [s for s in result.sites if s.classification == "routed"]
    waived = [s for s in result.sites if s.classification == "waived"]
    # The TEST+SMOKE surfaces are what the guard is FOR.  A production
    # auto-CUDA site is a legitimate opt-in surface (AGENTS.md §5.1) and is
    # reported as a CENSUS, but it does not make the guard fail.
    guarded_surfaces = ("test", "smoke")
    auto_test = [
        s for s in auto if s.surface in guarded_surfaces
    ]
    return {
        "guard": GUARD_NAME,
        "version": GUARD_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "files_scanned": result.files_scanned,
        "files_skipped": result.files_skipped,
        "scanned_nothing": result.files_scanned == 0,
        "sites_total": len(result.sites),
        "counts": {
            "by_classification": dict(sorted(by_classification.items())),
            "by_surface": dict(sorted(by_surface.items())),
        },
        "auto_cuda_count": len(auto),
        "auto_cuda_files": sorted({s.file for s in auto}),
        "auto_cuda_sites": [s.to_dict() for s in auto],
        "test_surface_auto_cuda_count": len(auto_test),
        "test_surface_auto_cuda_files": sorted({s.file for s in auto_test}),
        "test_surface_auto_cuda_sites": [s.to_dict() for s in auto_test],
        "routed_sites": [s.to_dict() for s in routed],
        "waived_sites": [s.to_dict() for s in waived],
        "sites": [s.to_dict() for s in result.sites],
        "waivers": {
            path: [{"matcher": matcher, "reason": reason} for matcher, reason in entries]
            for path, entries in sorted(WAIVERS.items())
        },
        "errors": result.errors,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-A156 GPU safety guard — find TEST/SMOKE surfaces that can "
            "auto-select CUDA.  Exit 0 = clean, 1 = auto-CUDA sites found, "
            "2 = skip/error."
        )
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent.parent),
        help="repo root to scan (default: the repo this script lives in)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="also write the JSON report to this path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the JSON report to stdout",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    result = scan_tree(root)
    report = build_report(root, result)

    if args.json_path:
        out = Path(args.json_path)
        if out.parent and str(out.parent):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if not args.quiet:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if not root.is_dir():
        return 2
    if result.errors:
        return 2
    if report["test_surface_auto_cuda_count"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
