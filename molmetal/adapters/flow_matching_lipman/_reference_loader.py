"""Load the optional upstream library in its own stable Python namespace.

Upstream uses absolute ``flow_matching`` imports, which collide with MolFlow's
public package. Only upstream modules receive a private builtins dictionary
whose import function redirects that one package name. No public module is
replaced, no search path changes, and calls made after initialization use the
same redirect. Ordinary import locks/caches preserve class and pickle identity.
"""
from __future__ import annotations

import builtins
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from threading import RLock


NAMESPACE = "molmetal.adapters.flow_matching_lipman._reference"
DEFAULT_REFERENCE = Path(__file__).resolve().parents[2] / "references" / "flow_matching"
_reference_root: Path | None = None
_lock = RLock()


def _reference_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "flow_matching" or name.startswith("flow_matching.")):
        relocated = NAMESPACE + name[len("flow_matching"):]
        result = builtins.__import__(relocated, globals, locals, fromlist, 0)
        # `import flow_matching.foo` binds the upstream package, whereas a
        # from-import must return the requested submodule. The builtin would
        # otherwise return the top-level `molmetal` package for an empty list.
        return result if fromlist else sys.modules[NAMESPACE]
    return builtins.__import__(name, globals, locals, fromlist, level)


class _ReferenceLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": _reference_import}
        super().exec_module(module)


class _ReferenceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(NAMESPACE + ".") or _reference_root is None:
            return None
        relative = fullname[len(NAMESPACE) + 1:].split(".")
        source = _reference_root.joinpath(*relative)
        package = source / "__init__.py"
        source = package if package.is_file() else source.with_suffix(".py")
        if not source.is_file():
            return None
        return importlib.util.spec_from_file_location(
            fullname, source, loader=_ReferenceLoader(fullname, str(source)),
            submodule_search_locations=[str(source.parent)] if source == package else None,
        )


_finder = _ReferenceFinder()


def configure_reference(reference_path: str | Path | None = None) -> Path:
    """Choose one clone per process; repeated loads preserve module identity.

    Default-clone classes unpickle in a fresh process through the real private
    package entry point. When using an alternate clone, configure the same path
    before unpickling. Changing the clone after loading is rejected rather than
    returning classes from a different source under an existing module name.
    """
    global _reference_root
    with _lock:
        ref = Path(reference_path).resolve() if reference_path is not None else (
            _reference_root.parent if _reference_root is not None else DEFAULT_REFERENCE)
        root = ref / "flow_matching"
        if not (root / "__init__.py").is_file():
            raise FileNotFoundError(f"Could not find cloned flow_matching at {ref}; clone https://github.com/facebookresearch/flow_matching.git there")
        if _reference_root is not None and _reference_root != root:
            raise RuntimeError(f"Upstream flow_matching already loaded from {_reference_root.parent}; cannot switch to {ref} in the same process")
        _reference_root = root
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)
        return root
