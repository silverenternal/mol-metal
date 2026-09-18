"""WF-BUG-2 fix tests — pocket_macro_inference CWD-relative path.

Bug
---
``pocket_macro_inference.py`` exposed the v2 checkpoint path as the
CWD-relative literal ``"models/pocket_macro_skeleton_v2.pt"``.  Any
caller that invoked ``PocketMacroInference()`` from a directory other
than the project root silently fell back to the metadata-only path
(no weights → ``is_available() == False`` → graceful no-op that hid
the model failures in downstream smoke tests).

Fix
---
The defaults are now resolved at construction time via
:meth:`_resolve_default_paths`, which checks the module-relative
``molmetal/models/`` directory first and falls back to the legacy
CWD-relative ``models/`` path so older callers continue to work.

These tests exercise the **pure-path** resolution logic (no torch
import required) from three different working directories
(``molmetal/tests/``, ``molmetal/``, ``/tmp/``) to confirm the fix
actually works in the wild.  The test deliberately avoids importing
the full module on a host where torch is broken — the path resolver
only depends on ``pathlib`` and ``os``, which are in the Python
standard library.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers — load the pure-path helpers without triggering torch import
# ---------------------------------------------------------------------------
PMI_FILE = Path(
    "/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/"
    "pocket_macro_inference.py"
)
REPO_ROOT = Path("/home/hugo/codes/try_triton_on_rocm")
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
TESTS_DIR = MOLMETAL_ROOT / "tests"
CHECKPOINT = MOLMETAL_ROOT / "models" / "pocket_macro_skeleton_v2.pt"
METADATA = MOLMETAL_ROOT / "models" / "pocket_macro_skeleton_v2.pt.json"


def _load_path_helpers():
    """Extract just the path-resolution surface from
    ``pocket_macro_inference.py`` via AST inspection.  We parse the
    module and re-build a tiny wrapper module that exposes:
        * ``_CHECKPOINT_FILENAME``
        * ``_METADATA_FILENAME``
        * ``_MODULE_RELATIVE_MODELS_DIR``
        * ``DEFAULT_CHECKPOINT_PATH``
        * ``DEFAULT_METADATA_PATH``
        * ``_resolve_default_paths``

    The wrapper also includes the ``pathlib.Path`` import so the
    function bodies execute.  Every other top-level statement (torch,
    numpy, dataclass, the v2 mirror class, etc.) is dropped.
    """
    import ast

    if not PMI_FILE.exists():
        pytest.skip(f"pocket_macro_inference.py not present at {PMI_FILE}")
    src = PMI_FILE.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(PMI_FILE))
    except SyntaxError as exc:  # pragma: no cover
        pytest.skip(f"pocket_macro_inference.py has syntax error: {exc}")

    keep_names = {
        "PER_RESIDUE_FEATURES",
        "V1_PER_RESIDUE_FEATURES",
        "_CHECKPOINT_FILENAME",
        "_METADATA_FILENAME",
        "_MODULE_RELATIVE_MODELS_DIR",
        "_PACKAGE_ROOT",
        "_REPO_PKG_ROOT",
        "_MODULE_DIR",
        "DEFAULT_CHECKPOINT_PATH",
        "DEFAULT_METADATA_PATH",
        "DEFAULT_CONFIDENCE_FLOOR",
        "_resolve_default_paths",
    }
    # Collect the source lines for the kept names only.
    kept_lines: list[str] = [
        "from pathlib import Path",
        "from typing import Dict, List, Optional, Tuple",
    ]

    def _names_in_node(node: ast.AST) -> set[str]:
        names: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                # ``pathlib.Path`` -> "Path"
                if isinstance(sub.value, ast.Name):
                    names.add(sub.value.id)
        return names

    def _resolve(node: ast.AST) -> set[str]:
        """Best-effort free-name resolution for the kept subtrees."""
        return _names_in_node(node)

    # Walk top-level statements and keep only the ones whose name is
    # in ``keep_names`` (for assignments / AnnAssign / FunctionDef).
    for node in tree.body:
        target_name: str | None = None
        if isinstance(node, ast.Assign):
            # ``DEFAULT_CHECKPOINT_PATH: str = ...`` lands here too.
            for t in node.targets:
                if isinstance(t, ast.Name):
                    target_name = t.id
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target_name = node.target.id
        elif isinstance(node, ast.FunctionDef):
            target_name = node.name

        if target_name in keep_names:
            # Re-emit the source slice verbatim.
            kept_lines.append(ast.unparse(node))
        elif target_name is not None and target_name.startswith("_"):
            # Private helpers we did not whitelist — skip silently.
            continue
        # else: silently drop heavy / unrelated statements.

    snippet = "\n\n".join(kept_lines)
    snippet += (
        "\n\n_PATH_HELPERS = {\n"
        "    '_CHECKPOINT_FILENAME': _CHECKPOINT_FILENAME,\n"
        "    '_METADATA_FILENAME': _METADATA_FILENAME,\n"
        "    '_MODULE_RELATIVE_MODELS_DIR': _MODULE_RELATIVE_MODELS_DIR,\n"
        "    'DEFAULT_CHECKPOINT_PATH': DEFAULT_CHECKPOINT_PATH,\n"
        "    'DEFAULT_METADATA_PATH': DEFAULT_METADATA_PATH,\n"
        "    '_resolve_default_paths': _resolve_default_paths,\n"
        "}\n"
    )
    ns: dict = {"__name__": "_pmi_path_only", "__file__": str(PMI_FILE)}
    try:
        exec(compile(snippet, str(PMI_FILE), "exec"), ns)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not load path helpers: {exc}")
    return ns["_PATH_HELPERS"]


# ---------------------------------------------------------------------------
# 1. Module-relative default paths point at the canonical checkpoint
# ---------------------------------------------------------------------------
def test_default_paths_are_module_relative_absolute():
    h = _load_path_helpers()
    # Both defaults must be absolute paths (NOT CWD-relative literals).
    assert os.path.isabs(h["DEFAULT_CHECKPOINT_PATH"]), (
        f"DEFAULT_CHECKPOINT_PATH must be absolute, got "
        f"{h['DEFAULT_CHECKPOINT_PATH']!r}"
    )
    assert os.path.isabs(h["DEFAULT_METADATA_PATH"]), (
        f"DEFAULT_METADATA_PATH must be absolute, got "
        f"{h['DEFAULT_METADATA_PATH']!r}"
    )
    # The defaults must point at the canonical molmetal/models/ directory
    # derived from __file__.
    expected_ckpt = MOLMETAL_ROOT / "models" / "pocket_macro_skeleton_v2.pt"
    expected_meta = MOLMETAL_ROOT / "models" / "pocket_macro_skeleton_v2.pt.json"
    assert Path(h["DEFAULT_CHECKPOINT_PATH"]).resolve() == expected_ckpt.resolve()
    assert Path(h["DEFAULT_METADATA_PATH"]).resolve() == expected_meta.resolve()


# ---------------------------------------------------------------------------
# 2. CWD-relative fallback works when module-relative path is absent
# ---------------------------------------------------------------------------
def test_cwd_relative_fallback_when_module_path_missing(tmp_path):
    """When the module-relative path is missing (e.g. a slim install
    that only puts the .pt in CWD), the CWD-relative fallback must
    win.  We simulate this by passing a non-existent module dir to
    the resolver; a real ``models/pocket_macro_skeleton_v2.pt`` lives
    in ``tmp_path/models/``."""
    h = _load_path_helpers()
    fake_models = tmp_path / "models"
    fake_models.mkdir(parents=True, exist_ok=True)
    fake_ckpt = fake_models / "pocket_macro_skeleton_v2.pt"
    fake_meta = fake_models / "pocket_macro_skeleton_v2.pt.json"
    fake_ckpt.write_bytes(CHECKPOINT.read_bytes())
    fake_meta.write_text(METADATA.read_text())

    # Pass a non-existent module dir; pass tmp_path as the cwd.  The
    # resolver must skip the module path and find the CWD-relative
    # fixture instead.
    ckpt, meta = h["_resolve_default_paths"](
        module_models_dir=tmp_path / "_does_not_exist",
        cwd=tmp_path,
    )
    assert Path(ckpt).resolve() == fake_ckpt.resolve(), (
        f"CWD-relative fallback failed: got {ckpt!r}, expected "
        f"{str(fake_ckpt)!r}"
    )
    assert Path(meta).resolve() == fake_meta.resolve()


# ---------------------------------------------------------------------------
# 3. Module-relative path wins when BOTH candidates exist
# ---------------------------------------------------------------------------
def test_module_relative_wins_when_both_exist(tmp_path):
    """When both the module-relative and CWD-relative paths exist the
    module-relative path MUST win (so CWD never affects the
    canonical lookup).  We stage both files with distinct content
    and assert the module-relative one is returned."""
    h = _load_path_helpers()
    # Build a fake module-relative dir with both files present.
    fake_module = tmp_path / "module_models"
    fake_module.mkdir(parents=True, exist_ok=True)
    fake_module_ckpt = fake_module / "pocket_macro_skeleton_v2.pt"
    fake_module_meta = fake_module / "pocket_macro_skeleton_v2.pt.json"
    fake_module_ckpt.write_bytes(b"MODULE-RELATIVE-PT")
    fake_module_meta.write_text('{"source": "module-relative"}')

    # Build a different CWD-relative fixture that should NOT be picked.
    fake_cwd = tmp_path / "cwd_models_dir"
    fake_cwd.mkdir(parents=True, exist_ok=True)
    cwd_models = fake_cwd / "models"
    cwd_models.mkdir(parents=True, exist_ok=True)
    (cwd_models / "pocket_macro_skeleton_v2.pt").write_bytes(b"CWD-PT")
    (cwd_models / "pocket_macro_skeleton_v2.pt.json").write_text(
        '{"source": "cwd"}'
    )

    ckpt, meta = h["_resolve_default_paths"](
        module_models_dir=fake_module, cwd=fake_cwd,
    )
    assert Path(ckpt).resolve() == fake_module_ckpt.resolve()
    assert Path(meta).resolve() == fake_module_meta.resolve()


# ---------------------------------------------------------------------------
# 4. Missing checkpoint → resolver still returns module-relative path
# ---------------------------------------------------------------------------
def test_resolve_returns_module_relative_path_when_missing(tmp_path):
    """When BOTH candidates are absent the resolver must return the
    module-relative path (so downstream ``is_available()`` and
    ``load()`` produce informative errors that point at the
    canonical location)."""
    h = _load_path_helpers()
    nonexistent_module = tmp_path / "_module_missing"
    nonexistent_cwd = tmp_path / "_cwd_missing"
    ckpt, meta = h["_resolve_default_paths"](
        module_models_dir=nonexistent_module, cwd=nonexistent_cwd,
    )
    assert ckpt == str(nonexistent_module / "pocket_macro_skeleton_v2.pt")
    assert meta == str(
        nonexistent_module / "pocket_macro_skeleton_v2.pt.json"
    )


# ---------------------------------------------------------------------------
# 5. Real module-relative default IS the canonical file
# ---------------------------------------------------------------------------
def test_real_module_relative_default_exists():
    """Sanity check: the module-relative default computed at import
    time points at the real ``molmetal/models/pocket_macro_skeleton_v2.pt``
    file that ships with the repo."""
    h = _load_path_helpers()
    assert CHECKPOINT.exists(), (
        f"sanity: real checkpoint must exist at {CHECKPOINT}"
    )
    assert METADATA.exists(), (
        f"sanity: real metadata must exist at {METADATA}"
    )
    default_ckpt = Path(h["DEFAULT_CHECKPOINT_PATH"])
    default_meta = Path(h["DEFAULT_METADATA_PATH"])
    assert default_ckpt.resolve() == CHECKPOINT.resolve()
    assert default_meta.resolve() == METADATA.resolve()


# ---------------------------------------------------------------------------
# 6. Subprocess construction succeeds from multiple CWDs
# ---------------------------------------------------------------------------
def test_subprocess_construction_succeeds_from_each_cwd(tmp_path):
    """Run a fresh Python process in each CWD and assert the resolver
    returns the canonical checkpoint.  This is the integration-level
    guard that catches any side-effect that would let ``Path(__file__)``
    resolve to a different location across processes (it should not,
    but we exercise it anyway)."""
    h = _load_path_helpers()
    # We re-exec the same in-memory helper via a subprocess so the
    # resolver's _MODULE_RELATIVE_MODELS_DIR is recomputed under the
    # subprocess's environment (which has a different CWD but the
    # same __file__ since we point at the same source file).
    code = (
        "import sys, ast, pathlib\n"
        f"src_path = {str(PMI_FILE)!r}\n"
        "src = pathlib.Path(src_path).read_text(encoding='utf-8')\n"
        "tree = ast.parse(src, filename=src_path)\n"
        "keep = {\n"
        "    'PER_RESIDUE_FEATURES', 'V1_PER_RESIDUE_FEATURES',\n"
        "    '_CHECKPOINT_FILENAME', '_METADATA_FILENAME',\n"
        "    '_MODULE_RELATIVE_MODELS_DIR', '_PACKAGE_ROOT', '_REPO_PKG_ROOT',\n"
        "    '_MODULE_DIR', 'DEFAULT_CHECKPOINT_PATH', 'DEFAULT_METADATA_PATH',\n"
        "    'DEFAULT_CONFIDENCE_FLOOR', '_resolve_default_paths',\n"
        "}\n"
        "lines = ['from pathlib import Path',\n"
        "         'from typing import Dict, List, Optional, Tuple']\n"
        "for node in tree.body:\n"
        "    target = None\n"
        "    if isinstance(node, ast.Assign):\n"
        "        for t in node.targets:\n"
        "            if isinstance(t, ast.Name):\n"
        "                target = t.id; break\n"
        "    elif isinstance(node, ast.AnnAssign):\n"
        "        if isinstance(node.target, ast.Name):\n"
        "            target = node.target.id\n"
        "    elif isinstance(node, ast.FunctionDef):\n"
        "        target = node.name\n"
        "    if target in keep:\n"
        "        lines.append(ast.unparse(node))\n"
        "snippet = '\\n\\n'.join(lines)\n"
        "ns = {'__name__': '_pmi_path_only', '__file__': src_path}\n"
        "exec(compile(snippet, src_path, 'exec'), ns)\n"
        "ckpt, meta = ns['_resolve_default_paths']()\n"
        "import os\n"
        "assert ckpt.endswith('pocket_macro_skeleton_v2.pt'), ckpt\n"
        "assert meta.endswith('pocket_macro_skeleton_v2.pt.json'), meta\n"
        f"expected = {str(CHECKPOINT)!r}\n"
        "assert pathlib.Path(ckpt).resolve() == pathlib.Path(expected).resolve(), (\n"
        "    f'BUG-2 regression from CWD={os.getcwd()!r}: '\n"
        "    f'got {ckpt!r} expected {expected!r}')\n"
        "print('OK', os.getcwd(), ckpt)\n"
    )
    for cwd in (REPO_ROOT, MOLMETAL_ROOT, TESTS_DIR, Path("/tmp")):
        if not cwd.exists():
            continue
        result = __import__("subprocess").run(
            [sys.executable, "-c", code],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"BUG-2 regression from CWD={cwd!r}:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "OK" in result.stdout, (
            f"subprocess from {cwd} did not print OK: {result.stdout!r}"
        )