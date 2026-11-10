"""molmetal/_device.py — the CPU-safe device resolver (Wf-A156).

WHY THIS MODULE EXISTS
======================
The repo has TWO device-selection idioms and only one of them is safe.

UNSAFE (the defect class Wf-A156 exists to kill)::

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

This fires the moment a CUDA device is VISIBLE, with no opt-in.  In a test
or a smoke path that violates the standing 禁止使用gpu directive and
AGENTS.md §5.1 ("VRAM budget check first"): the device is taken because it
happened to be there, not because anybody asked for it.  MEASURED
2026-10-31: a full ``pytest molmetal/tests`` run held 1444 MiB (the pytest
process) + 2218 MiB (a smoke subprocess) of GPU 0 VRAM.

SAFE (this module)::

    from molmetal._device import resolve_device
    device = resolve_device()

``resolve_device()`` returns ``torch.device("cpu")`` in a TEST/SMOKE context
and ``torch.device("cuda" if torch.cuda.is_available() else "cpu")`` in a
PRODUCTION context — so the production default is UNCHANGED and only the
test/smoke default moves.

THE CONTRACT
============
``resolve_device(allow_gpu=None)``
    * ``allow_gpu=True``  → explicit opt-in: CUDA if it is available.
    * ``allow_gpu=False`` → CPU, always, without probing.
    * ``allow_gpu=None``  → context-dependent (the default):
        - TEST/SMOKE context → CPU, WITHOUT calling
          ``torch.cuda.is_available()`` at all (a probe is not an
          allocation, but there is no reason to make it);
        - PRODUCTION context → ``torch.cuda.is_available()``.

    The session-level opt-in ``MOLMETAL_TEST_ALLOW_GPU=1`` (the SAME
    variable the repo's two conftests honour — see
    ``conftest.py`` and ``molmetal/tests/conftest.py``) turns the GPU back
    on everywhere, so an explicit GPU test run is still possible.

``cuda_device_indices(allow_gpu=None)``
    The replacement for ``[torch.cuda.current_device()] if
    torch.cuda.is_available() else []`` — the ``torch.random.fork_rng``
    idiom.  Returns ``[]`` on CPU, so ``fork_rng`` never touches a CUDA
    RNG state.

``gpu_permitted()``
    Whether this session may place a tensor on a GPU at all.

WHAT COUNTS AS A "TEST/SMOKE CONTEXT"
=====================================
The CALLER's file decides, not a global flag — a resolver that looked only
at the environment could not tell a test from a trainer:

1. the caller's file lives under a directory named ``tests`` / ``test``;
2. the caller's file stem starts with ``test_``;
3. the caller's file is one of the four SMOKE SELF-CHECK SCRIPTS
   (``molmetal/scripts/wf399a1_smoke_protocol.py`` & co) — these are
   imported BY a test module, so they are a test surface even though they
   live under ``molmetal/scripts/``;
4. the caller's file stem contains ``smoke`` and it is NOT under a launch
   entry-point directory (``molmetal/scripts/``, ``scripts/``) — a launch
   entry point is a deliberate opt-in surface (AGENTS.md §5.1) and its
   defaults are NOT this module's to change.

``MOLMETAL_TEST_ALLOW_GPU=1`` overrides all of the above: it is the
repo-wide, documented opt-in.

CPU-ONLY
========
Importing this module does NOT import torch: the torch import is lazy, so
the module can be imported by a static tool (e.g.
``molmetal/scripts/wfa156_gpu_safety_guard.py``) without pulling in the
torch stack.  Nothing in this module allocates VRAM, ever.

Cross-references
----------------
* ``molmetal/scripts/wfa156_gpu_safety_guard.py`` — the static guard that
  finds auto-CUDA selection in TEST/SMOKE surfaces.
* ``molmetal/tests/test_wfa156_gpu_safety_guard.py`` — its contract test.
* ``conftest.py`` (repo root) + ``molmetal/tests/conftest.py`` — the
  RUNTIME CPU lock (environment-variable level).  This module is the
  SOURCE level counterpart: the lock depends on the session honouring the
  environment, this one depends only on the call site's context.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

__all__ = [
    "MOLMETAL_TEST_ALLOW_GPU_ENV",
    "cuda_device_indices",
    "gpu_permitted",
    "in_test_context",
    "resolve_device",
]


#: The repo-wide, documented GPU opt-in.  The SAME variable the two
#: conftests honour (``conftest.py`` at the repo root and
#: ``molmetal/tests/conftest.py``); setting it to ``"1"`` leaves
#: ``CUDA_VISIBLE_DEVICES`` untouched and lets the GPU be used.
MOLMETAL_TEST_ALLOW_GPU_ENV = "MOLMETAL_TEST_ALLOW_GPU"


#: The repo root (this file is ``<repo>/molmetal/_device.py``).
_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directories whose files are LAUNCH ENTRY POINTS — a deliberate opt-in
#: surface whose defaults this module does not change.
_LAUNCH_ENTRY_POINT_DIRS = ("molmetal/scripts", "scripts")

#: Smoke scripts that are IMPORTED by a test module — a test/smoke surface
#: even though they live under ``molmetal/scripts/``.  Kept in sync with
#: ``SMOKE_SELFCHECK_SCRIPTS`` in
#: ``molmetal/scripts/wfa156_gpu_safety_guard.py``.
_SMOKE_SELFCHECK_SCRIPTS = frozenset(
    {
        "molmetal/scripts/wf329w5_w2_chemprop_smoke.py",
        "molmetal/scripts/wf399a1_smoke_protocol.py",
        "molmetal/scripts/wf404a5_wf29_attr_fix_smoke.py",
        "molmetal/scripts/wf409a4_wf29_extended_smoke.py",
    }
)


def _torch():
    """Import torch lazily — this module must stay import-light."""
    import torch  # noqa: WPS433 — lazy on purpose (see the module docstring)

    return torch


def _caller_file() -> Optional[str]:
    """The source file of the first frame outside this module."""
    try:
        frame = sys._getframe(1)
    except ValueError:  # pragma: no cover — defensive
        return None
    here = os.path.abspath(__file__)
    while frame is not None:
        fname = frame.f_globals.get("__file__")
        if fname and os.path.abspath(fname) != here:
            return fname
        frame = frame.f_back
    return None


def _repo_relative(path: Optional[str]) -> Optional[str]:
    """``path`` relative to the repo root, POSIX-style, or ``None``."""
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(_REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return None


def _is_test_surface(path: Optional[str]) -> bool:
    """Is ``path`` a TEST surface?"""
    if not path:
        return False
    parts = Path(path).parts
    if "tests" in parts or "test" in parts:
        return True
    return Path(path).stem.startswith("test_")


def _is_smoke_surface(path: Optional[str]) -> bool:
    """Is ``path`` a SMOKE surface (and not a launch entry point)?"""
    if not path:
        return False
    rel = _repo_relative(path)
    if rel is not None:
        if rel in _SMOKE_SELFCHECK_SCRIPTS:
            return True
        if rel.startswith(_LAUNCH_ENTRY_POINT_DIRS):
            return False
    return "smoke" in Path(path).stem.lower()


def in_test_context() -> bool:
    """True when the CALLER is in a TEST / SMOKE surface.

    ``MOLMETAL_TEST_ALLOW_GPU=1`` wins: it is the repo-wide opt-in, and it
    is honoured by the two conftests, so the resolver must agree with them.
    """
    if os.environ.get(MOLMETAL_TEST_ALLOW_GPU_ENV) == "1":
        return False
    caller = _caller_file()
    if _is_test_surface(caller):
        return True
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    return _is_smoke_surface(caller)


def gpu_permitted() -> bool:
    """True when this session may place a tensor on a GPU."""
    if os.environ.get(MOLMETAL_TEST_ALLOW_GPU_ENV) == "1":
        return True
    return not in_test_context()


def resolve_device(allow_gpu: Optional[bool] = None) -> "object":
    """Resolve the torch device for this call site.

    Parameters
    ----------
    allow_gpu
        ``None`` (the default) resolves by CONTEXT: CPU in a test/smoke
        surface, ``cuda``-if-available in production.  ``True`` is an
        explicit per-call opt-in.  ``False`` forces CPU and never probes
        the driver.

    Returns
    -------
    torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    """
    torch = _torch()
    if allow_gpu is None:
        allow_gpu = gpu_permitted()
    if not allow_gpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_device_indices(allow_gpu: Optional[bool] = None) -> List[int]:
    """The CUDA device indices to hand to ``torch.random.fork_rng``.

    The CPU-safe replacement for::

        [torch.cuda.current_device()] if torch.cuda.is_available() else []

    Returns ``[]`` on CPU, so ``fork_rng`` never saves/restores a CUDA RNG
    state (which would create a CUDA context).
    """
    torch = _torch()
    if allow_gpu is None:
        allow_gpu = gpu_permitted()
    if not allow_gpu:
        return []
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.current_device()]
