"""Stable import/pickle entry point for the isolated optional Meta reference.

The original source files remain unchanged; their absolute imports are handled
by the private loader. MolFlow's public ``flow_matching`` package stays intact.
"""

from pathlib import Path

from .._reference_loader import configure_reference

_upstream = configure_reference()

# 2026-10-04 fix.  `__path__` used to be `[str(configure_reference())]`, i.e.
# ONLY the upstream clone -- which is the post-2024 FLAT layout and contains no
# `path/`, `solver/` or `utils/` subpackage.  But this directory holds exactly
# those three as legacy-API shims (see `path/__init__.py`: upstream flattened
# `path/`, `solver/` and `utils/` into `loss.py`, `sampler.py`,
# `interpolation.py` and `sampler_config.py`).  Overwriting `__path__` made
# every shim invisible, so `import _reference.path` raised
# `ModuleNotFoundError` even though the file is right there on disk.
#
# This broke 6 tests in test_cfm_p0_fixes.py (and anything else that reaches
# the legacy names) from the moment the shims landed until now.
#
# Keep BOTH entries: this package's own directory so the shims resolve, and the
# upstream clone so the original lookup still works.  The shims come first so a
# legacy name is served by our shim rather than accidentally matching an
# upstream module of the same name.
__path__ = [str(Path(__file__).resolve().parent), str(_upstream)]
