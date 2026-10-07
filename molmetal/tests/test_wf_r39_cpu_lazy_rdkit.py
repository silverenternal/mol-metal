# Copyright 2026 The Mol-Metal Authors. All rights reserved.
#
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
"""wf-R39-CPU — regression guard for the TODO-92 §3.2 lazy-RDKit invariant.

TODO-92 (``TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md``) §3 step 2
requires that RDKit is NOT imported at module import time on the two
descriptor/panel script paths, so that CPU-only smoke tests that never touch
RDKit descriptors do not hard-fail when RDKit is absent.

wf-R39 Phase-1/2 audit found the CODE half of TODO-92 §3 step 2 is already
implemented (MEASURED via AST scan, not assumption):

* ``molmetal/scripts/cfm_property_panel.py`` — 3 RDKit imports, ALL nested
  inside function bodies (``_load_sascorer`` L105, ``compute_one`` L178-179);
  module level has zero RDKit imports.
* ``molmetal/scripts/extract_rdkit_2d_descriptors.py`` — module-level RDKit
  import is wrapped in ``try/except Exception`` (Wf-39 P1 vuln_a fix) and sets
  ``_HAS_RDKIT``; the module therefore still imports cleanly with RDKit absent.

What was genuinely MISSING is the regression guard: nothing in
``molmetal/tests/`` pinned either fact.  Per AGENTS.md §5.7 (a wire-in is not
SHIPPED until an automated check exercises it) this module pins it.

These tests are pure CPU + AST-only.  They never import RDKit, never touch
CUDA, and never invoke a trainer.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"

#: Scripts whose TODO-92 §3 step 2 contract is "RDKit is not a hard
#: module-import-time dependency".
LAZY_RDKIT_SCRIPTS: Tuple[str, ...] = (
    "cfm_property_panel.py",
    "extract_rdkit_2d_descriptors.py",
)


def _module_level_rdkit_imports(path: Path) -> List[Tuple[int, str]]:
    """Return ``(lineno, source)`` for every RDKit import in module scope.

    Uses :mod:`ast` rather than ``grep`` so an indented import inside an
    ``if TYPE_CHECKING:`` or a function body cannot be miscounted as a
    module-level import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: List[Tuple[int, str]] = []
    for node in tree.body:  # module scope ONLY
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rendered = ast.unparse(node)
            if "rdkit" in rendered.lower():
                hits.append((node.lineno, rendered))
    return hits


def _function_level_rdkit_imports(path: Path) -> List[Tuple[str, int]]:
    """Return ``(enclosing_function, lineno)`` for RDKit imports inside defs."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: List[Tuple[str, int]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for stmt in func.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                rendered = ast.unparse(stmt)
                if "rdkit" in rendered.lower():
                    hits.append((func.name, stmt.lineno))
    return hits


@pytest.mark.parametrize("script_name", LAZY_RDKIT_SCRIPTS)
def test_rdkit_not_imported_at_module_scope(script_name: str) -> None:
    """TODO-92 §3.2: no bare module-level ``import rdkit``."""
    path = SCRIPTS_DIR / script_name
    assert path.is_file(), f"missing script: {path}"

    offenders = [
        (lineno, src)
        for lineno, src in _module_level_rdkit_imports(path)
        # A module-level import is only acceptable when it is guarded by a
        # try/except; a bare one would make RDKit a hard dependency.
        if not _is_guarded(path, lineno)
    ]
    assert offenders == [], (
        f"{script_name} imports RDKit at module scope without a "
        f"try/except guard (TODO-92 §3.2 lazy-import contract): {offenders}"
    )


def _is_guarded(path: Path, lineno: int) -> bool:
    """True when the module-level statement at ``lineno`` sits in a try/except."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.body:
            if getattr(stmt, "lineno", None) == lineno:
                return True
    return False


def test_cfm_property_panel_uses_function_level_rdkit() -> None:
    """``cfm_property_panel.py`` must keep RDKit inside function bodies."""
    hits = _function_level_rdkit_imports(SCRIPTS_DIR / "cfm_property_panel.py")
    assert hits, (
        "cfm_property_panel.py no longer defers RDKit to a function body — "
        "the TODO-92 §3.2 lazy-import contract appears to have been reverted"
    )
    funcs = {name for name, _ in hits}
    assert "compute_one" in funcs, (
        f"expected RDKit import inside compute_one, found {sorted(funcs)}"
    )


def test_extract_rdkit_2d_sets_has_rdkit_flag() -> None:
    """The Wf-39 P1 vuln_a fallback flag must remain present.

    ``extract_rdkit_2d_descriptors.py`` is allowed a module-level RDKit
    import because it is wrapped in ``try/except``; the observable
    contract is the ``_HAS_RDKIT`` sentinel.
    """
    src = (SCRIPTS_DIR / "extract_rdkit_2d_descriptors.py").read_text(
        encoding="utf-8"
    )
    assert "_HAS_RDKIT" in src, (
        "extract_rdkit_2d_descriptors.py lost its _HAS_RDKIT guard flag — "
        "a module-level RDKit import is now unguarded (Wf-39 P1 vuln_a)"
    )


@pytest.mark.parametrize("script_name", LAZY_RDKIT_SCRIPTS)
def test_script_imports_without_rdkit_in_sys_modules(script_name: str) -> None:
    """Subprocess proof: importing the script must not pull RDKit into sys.modules.

    Run in a clean interpreter with ``rdkit`` poisoned by an import hook, so
    the assertion holds even on hosts where RDKit IS installed.  CPU-only.
    """
    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'rdkit' or name.startswith('rdkit.') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('rdkit blocked by wf-R39-CPU guard')\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'rdkit' or name.startswith('rdkit.'):\n"
        "            raise ImportError('rdkit blocked by wf-R39-CPU guard')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"sys.path.insert(0, {str(SCRIPTS_DIR)!r})\n"
        f"import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location("
        f"'wf_r39_cpu_{script_name[:-3]}', {str(SCRIPTS_DIR / script_name)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'rdkit' not in sys.modules, 'rdkit entered sys.modules'\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"{script_name} does not import cleanly with RDKit blocked:\n"
        f"stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}"
    )
    assert "OK" in proc.stdout


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
