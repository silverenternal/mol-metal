"""Shim — re-exports upstream flat flow_matching as the legacy ``path`` API.

The upstream facebookresearch/flow_matching library restructured in 2024:
the legacy ``flow_matching/path/`` (AffineProbPath), ``path/scheduler/``
(CondOTScheduler), ``solver/`` (ODESolver), and ``utils/`` (ModelWrapper)
submodules were flattened into ``flow_matching/loss.py``,
``flow_matching/sampler.py``, ``flow_matching/interpolation.py``, and
``flow_matching/sampler_config.py``.

The molmetal adapter still calls into the legacy names via
``_import_fm_lib``. To keep the adapter code unchanged, this shim package
re-exposes the modern symbols under the legacy paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Locate the upstream flow_matching clone via the reference loader.
from molmetal.adapters.flow_matching_lipman._reference_loader import (
    _reference_root,
    configure_reference,
)

if _reference_root is None:
    # Auto-bootstrap: try the canonical locations in order.
    for cand in (
        Path("/home/hugo/codes/mol-metal/molmetal/references/flow_matching"),
        Path("/home/hugo/codes/mol-metal/flow_matching"),
    ):
        if cand.is_dir() and (cand / "__init__.py").is_file():
            configure_reference(cand)
            break

_ref = _reference_root
if _ref is None:  # pragma: no cover
    raise ImportError(
        "flow_matching upstream not configured; call "
        "configure_reference(<path>) before importing _reference.path"
    )

# Wf-415-A3 fix: when `_reference_root` points to the shim directory (where
# `path/` + `solver/` + `utils/` shim subpackages live), the upstream flow_matching
# clone lives at `_reference_root / flow_matching/` (a symlink). Detect this and
# adjust `_UPSTREAM` accordingly so the auto-bootstrap loop finds the upstream
# `interpolation.py` + `loss.py` + `sampler.py` modules.
_UPSTREAM_CANDIDATE = _ref / "flow_matching"
if _UPSTREAM_CANDIDATE.is_dir() and (_UPSTREAM_CANDIDATE / "__init__.py").is_file():
    _ref = _UPSTREAM_CANDIDATE

# Inject the upstream clone into sys.modules under the ``flow_matching``
# name so the import shims below can ``from flow_matching import ...``.
import importlib.util
import importlib.abc
import importlib.machinery
import types
import builtins

_UPSTREAM = _ref
_UPSTREAM_NAME = "flow_matching"

# Build / reuse a private alias under our private namespace.
_NSNAME = "molmetal.adapters.flow_matching_lipman._reference.upstream_flow_matching"
if _NSNAME not in sys.modules:
    _pkg = types.ModuleType(_NSNAME)
    _pkg.__path__ = [str(_UPSTREAM)]
    _pkg.__file__ = str(_UPSTREAM / "__init__.py")
    sys.modules[_NSNAME] = _pkg
    # Import every submodule present in the upstream clone.
    for _py in sorted(_UPSTREAM.glob("*.py")):
        if _py.name == "__init__.py":
            continue
        _modname = _NSNAME + "." + _py.stem
        _spec = importlib.util.spec_from_file_location(_modname, _py)
        _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
        sys.modules[_modname] = _mod
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    for _sub in sorted(_UPSTREAM.iterdir()):
        if _sub.is_dir() and (_sub / "__init__.py").is_file():
            _subname = _NSNAME + "." + _sub.name
            _subpkg = types.ModuleType(_subname)
            _subpkg.__path__ = [str(_sub)]
            _subpkg.__file__ = str(_sub / "__init__.py")
            sys.modules[_subname] = _subpkg
            _spec = importlib.util.spec_from_file_location(
                _subname, _sub / "__init__.py"
            )
            _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
            sys.modules[_subname] = _mod
            _spec.loader.exec_module(_mod)  # type: ignore[union-attr]

# Build the legacy ``flow_matching`` namespace alias so user code can
# ``from flow_matching.path import AffineProbPath``.
_flow_matching_alias = sys.modules.get("flow_matching")
if _flow_matching_alias is None:
    _flow_matching_alias = sys.modules[_NSNAME]
    sys.modules["flow_matching"] = _flow_matching_alias

# ---------------------------------------------------------------------------
# Legacy ``path`` module shim
# ---------------------------------------------------------------------------

# The adapter expects:
#   from .path import AffineProbPath
# In the flat upstream, there is no ``AffineProbPath`` class — the path is
# exposed via ``flow_matching.interpolate`` and ``flow_matching.target_velocity``.
# We rebuild a thin ``AffineProbPath``-compatible facade.

from molmetal.adapters.flow_matching_lipman._reference.upstream_flow_matching.interpolation import (  # noqa: E501
    interpolate as _upstream_interpolate,
    target_velocity as _upstream_target_velocity,
)


class _PathSample:
    """Dataclass returned by ``AffineProbPath.sample``.

    Holds the interpolated state ``x_t = (1 - t) x_0 + t x_1`` and the
    target conditional velocity ``dx_t = x_1 - x_0``. Used by the
    adapter's ``train_step`` to extract ``path_sample.x_t`` /
    ``path_sample.dx_t`` (matching the legacy Lipman 2023 API).
    """

    __slots__ = ("x_t", "dx_t", "t", "x_0", "x_1")

    def __init__(self, x_t, dx_t, t, x_0, x_1):
        self.x_t = x_t
        self.dx_t = dx_t
        self.t = t
        self.x_0 = x_0
        self.x_1 = x_1


class AffineProbPath:
    """Facade exposing the legacy ``AffineProbPath`` API.

    Wraps upstream ``interpolate`` + ``target_velocity`` so callers can
    treat the path the way the legacy 2023 Lipman API did. Returns a
    ``_PathSample`` carrying both ``x_t`` (interpolated state) and
    ``dx_t`` (target velocity) for the adapter's ``train_step``.
    """

    def __init__(self, scheduler=None):
        self.scheduler = scheduler

    def sample(self, x_0: "torch.Tensor", x_1: "torch.Tensor", t: "torch.Tensor"):
        """Affine path: ``x_t = (1 - t) x_0 + t x_1`` (OT-scheduler semantics).

        Returns a ``_PathSample`` with ``x_t`` and ``dx_t`` attributes.
        """
        x_t = _upstream_interpolate(x_0, x_1, t)
        dx_t = _upstream_target_velocity(x_0, x_1)
        return _PathSample(x_t=x_t, dx_t=dx_t, t=t, x_0=x_0, x_1=x_1)

    def target_velocity_from_data(self, x_0: "torch.Tensor", x_1: "torch.Tensor"):
        """OT-scheduler target velocity: ``v_t = x_1 - x_0``."""
        return _upstream_target_velocity(x_0, x_1)


# Make ``AffineProbPath`` importable from this shim.
__all__ = ["AffineProbPath", "_PathSample"]
