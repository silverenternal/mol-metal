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


# Wf-A62 W1 (RL1) — surface a missing reference clone at import time
# rather than at first adapter-init time.  Previously
# ``DEFAULT_REFERENCE`` was silently consumed by ``configure_reference``
# when the caller passed ``reference_path=None``, and the failure
# surfaced as a confusing "Could not find cloned flow_matching"
# FileNotFoundError long after the import.  Importing this module is
# cheap; the warn is no-op when the clone is present.
try:
    _DEFAULT_REFERENCE_OK = DEFAULT_REFERENCE.is_dir()
except OSError:
    _DEFAULT_REFERENCE_OK = False
if not _DEFAULT_REFERENCE_OK:
    import warnings as _rl_warnings  # local alias avoids polluting top-of-file import order

    _rl_warnings.warn(
        f"_reference_loader.DEFAULT_REFERENCE ({DEFAULT_REFERENCE}) is not a "
        f"directory on disk; the upstream flow_matching clone must be "
        f"present at molmetal/references/flow_matching (or "
        f"``configure_reference(Path)`` must be called before any "
        f"``_import_fm_lib`` consumer).",
        stacklevel=2,
    )


def _reference_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "flow_matching" or name.startswith("flow_matching.")):
        relocated = NAMESPACE + name[len("flow_matching") :]
        result = builtins.__import__(relocated, globals, locals, fromlist, 0)
        # `import flow_matching.foo` binds the upstream package, whereas a
        # from-import must return the requested submodule. The builtin would
        # otherwise return the top-level `molmetal` package for an empty list.
        return result if fromlist else sys.modules[NAMESPACE]
    return builtins.__import__(name, globals, locals, fromlist, level)


class _ReferenceLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        module.__dict__["__builtins__"] = {
            **vars(builtins),
            "__import__": _reference_import,
        }
        super().exec_module(module)


class _ReferenceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(NAMESPACE + ".") or _reference_root is None:
            return None
        relative = fullname[len(NAMESPACE) + 1 :].split(".")
        source = _reference_root.joinpath(*relative)
        package = source / "__init__.py"
        source = package if package.is_file() else source.with_suffix(".py")
        if not source.is_file():
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            source,
            loader=_ReferenceLoader(fullname, str(source)),
            submodule_search_locations=[str(source.parent)]
            if source == package
            else None,
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
        ref = (
            Path(reference_path).resolve()
            if reference_path is not None
            else (
                _reference_root.parent
                if _reference_root is not None
                else DEFAULT_REFERENCE
            )
        )
        # Wf-A56 W2 (2026-10-16) — P6 auto-detect.  Accept the
        # clone-root convention where ``ref`` IS the ``flow_matching``
        # package (has ``__init__.py`` directly) in addition to the
        # parent-dir convention (``ref/flow_matching/__init__.py``).
        # R12-R17 regression: callers using the legacy convention
        # silently failed the auto-detect, leaving the upstream
        # adapter un-imported.
        if (ref / "__init__.py").is_file() and ref.name == "flow_matching":
            root = ref
        else:
            root = ref / "flow_matching"
        if not (root / "__init__.py").is_file():
            raise FileNotFoundError(
                f"Could not find cloned flow_matching at {ref}; "
                f"clone https://github.com/facebookresearch/flow_matching.git there"
            )
        if _reference_root is not None and _reference_root != root:
            raise RuntimeError(
                f"Upstream flow_matching already loaded from {_reference_root.parent}; "
                f"cannot switch to {ref} in the same process"
            )
        _reference_root = root
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)
        return root
