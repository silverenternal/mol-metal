"""molmetal_lam — top-level shim.

The canonical package lives at molmetal/molmetal_lam/ (as a sub-package
of molmetal/). This file re-exports it so both ``import molmetal_lam``
and ``from molmetal import molmetal_lam`` resolve to the same code.

We also redirect ``__path__`` so that ``molmetal_lam.atoms``,
``molmetal_lam.bonds``, etc. are looked up inside the canonical
``molmetal/molmetal_lam/`` directory (the top-level directory at the
repo root is intentionally empty).

See molmetal/molmetal_lam/__init__.py for the full MLC framework
documentation.
"""
from __future__ import annotations

import os as _os

from molmetal import molmetal_lam as _impl  # canonical implementation

# Point submodule search at the canonical nested package so that
# `molmetal_lam.atoms.combinators` resolves correctly regardless of
# whether the import entered via `molmetal.molmetal_lam.atoms` or the
# top-level `molmetal_lam.atoms` shim.
_canonical_dir = _os.path.dirname(_impl.__file__)
__path__ = [_canonical_dir]  # type: ignore[attr-defined]

# Re-export everything from the canonical package at the top level.
__version__ = getattr(_impl, "__version__", "0.1.0")
__all__ = getattr(_impl, "__all__", [])
try:
    from molmetal.molmetal_lam import *  # noqa: F401,F403
except ImportError:  # pragma: no cover
    pass