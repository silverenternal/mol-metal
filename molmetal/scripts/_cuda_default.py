#!/usr/bin/env python3
"""_cuda_default.py — package-import-free transport for the CUDA default.

Why this shim exists
--------------------
The canonical resolver lives in
``molmetal/molmetal_lam/sbdd_env/training_utils.py``.  Importing it the
normal way (``from molmetal.molmetal_lam.sbdd_env.training_utils import
...``) imports the ``molmetal`` package first — and
``molmetal/__init__.py`` initialises CUDA at import time
(``_apply_rocm_gfx1101_env`` probes ``torch.cuda.is_available()``).
CUDA reads ``CUDA_VISIBLE_DEVICES`` exactly once, at init — so an env
var applied *after* the package import is too late.

Training scripts therefore call :func:`ensure_cuda_visible_devices_default`
from their module top, BEFORE any molmetal / torch import.  This shim
loads the resolver module directly from its file path (no package
import, no torch) and re-exports the public helpers, giving the
module-top call sites a single-line transport to the canonical
resolver (Wf-130; CUDA_VISIBLE_DEVICES unification 2026-10-03).

Contract: identical semantics to the resolver — a set
``CUDA_VISIBLE_DEVICES`` (incl. ``""`` = CPU-only) wins; otherwise the
default (``"0"`` = GPU 0, RTX PRO 6000 Blackwell) is applied.  No
CUDA / torch state is touched by this module.

Verify
------
::

    python -c "import _cuda_default as m; \\
        print(m.DEFAULT_CUDA_VISIBLE_DEVICES)"
"""

from __future__ import annotations

import importlib.util as _importlib_util
from pathlib import Path as _Path

_RESOLVER_PATH = (
    _Path(__file__).resolve().parent.parent
    / "molmetal_lam"
    / "sbdd_env"
    / "training_utils.py"
)

_spec = _importlib_util.spec_from_file_location(
    "_molmetal_cuda_resolver", _RESOLVER_PATH
)
if _spec is None or _spec.loader is None:  # pragma: no cover — defensive
    raise ImportError(f"cannot load CUDA resolver from {_RESOLVER_PATH}")
_resolver = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_resolver)

DEFAULT_CUDA_VISIBLE_DEVICES = _resolver.DEFAULT_CUDA_VISIBLE_DEVICES
ensure_cuda_visible_devices_default = _resolver.ensure_cuda_visible_devices_default
apply_default_cuda_visible_devices = _resolver.apply_default_cuda_visible_devices

__all__ = [
    "DEFAULT_CUDA_VISIBLE_DEVICES",
    "apply_default_cuda_visible_devices",
    "ensure_cuda_visible_devices_default",
]
