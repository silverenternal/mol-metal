"""Wf-A61 (2026-10-17) L19 — canonical ``sascorer`` aggregator.

Three distinct import patterns for the RDKit SA-score exist across the
codebase (the Wf-A53 W2 9-sites audit confirmed these):

1. ``chemistry_losses.py`` — ``importlib.util.spec_from_file_location``
   against ``RDConfig.__file__`` (legacy opt-in for rdkit.Chem.Contrib).
2. ``end_of_training.py`` — the canonical in-tree wrapper at
   ``molmetal.eval.scoring.sascorer``.
3. ``adapters/rdkit_predictor.py`` — ``RDContribDir`` direct path.

All three can drift when RDKit ships a new layout (e.g. dropping
``Contrib.SA_Score``).  This module provides ONE callable
(:func:`try_load`) that returns the first available loader as a
``Optional[Callable[[Mol], float]]``, with ``None`` on full failure.

Backwards-compatible: existing sites keep their import paths; new
callers prefer :func:`try_load` so future RDKit layout changes are
absorbed in one place.

Public API
----------
- :func:`try_load`        — return a callable SA scorer or ``None``.
- :func:`compute_sa`      — score a single RDKit ``Mol`` or ``None`` input.
"""

from __future__ import annotations

import os as _os
from typing import Callable, Optional

# Lazy types — RDKit may be unavailable on CPU-only hosts.
try:
    from rdkit.Chem import Mol as _Mol  # type: ignore
except ImportError:  # pragma: no cover - RDKit always available in CI
    _Mol = None  # type: ignore


_SASCORER_CALLABLE = Optional[Callable[[object], float]]


def _try_importlib_rdkconfig() -> _SASCORER_CALLABLE:
    """Path 1 — chemistry_losses.py:250-269 importlib+RDConfig."""
    try:
        from rdkit.Chem import RDConfig  # noqa: F401
    except ImportError:
        return None
    try:
        import importlib.util as _ilu
        _path = _os.path.join(
            _os.path.dirname(RDConfig.RDContribDir or RDConfig.__file__),  # type: ignore[attr-defined]
            "Contrib",
            "SA_Score",
            "sascorer.py",
        )
        if not _os.path.isfile(_path):
            return None
        spec = _ilu.spec_from_file_location("_sascorer_wfa61", _path)
        if spec is None or spec.loader is None:
            return None
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.calculateScore
    except (ImportError, OSError, AttributeError, RuntimeError, ValueError):
        return None


def _try_canonical_wrapper() -> _SASCORER_CALLABLE:
    """Path 2 — end_of_training.py:58 + sa_penalty.py:119 in-tree wrapper."""
    try:
        from molmetal.eval.scoring.sascorer import compute_sa_score  # type: ignore
    except ImportError:
        return None
    return compute_sa_score


def _try_rdkit_direct() -> _SASCORER_CALLABLE:
    """Path 3 — adapters/rdkit_predictor.py:34-55 RDContribDir."""
    try:
        from rdkit.Chem import RDConfig
    except ImportError:
        return None
    try:
        contrib = getattr(RDConfig, "RDContribDir", None)
        if not contrib:
            return None
        # Lazy import via the importlib path used by rdkit_predictor.
        import importlib.util as _ilu
        _path = _os.path.join(contrib, "SA_Score", "sascorer.py")
        if not _os.path.isfile(_path):
            return None
        spec = _ilu.spec_from_file_location("_sascorer_rdkit_path", _path)
        if spec is None or spec.loader is None:
            return None
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.calculateScore
    except (ImportError, OSError, AttributeError, RuntimeError, ValueError):
        return None


def try_load() -> _SASCORER_CALLABLE:
    """Return the first available SA-score callable, or ``None``.

    Order: canonical in-tree wrapper (Path 2, fastest) → RDKit Contrib
    importlib+RDConfig (Path 1) → RDKit Contrib RDContribDir direct
    (Path 3).  Returns ``None`` when RDKit is unavailable or all three
    paths fail (a missing Contrib addon is the common case).
    """
    for loader in (
        _try_canonical_wrapper,
        _try_importlib_rdkconfig,
        _try_rdkit_direct,
    ):
        fn = loader()
        if fn is not None:
            return fn
    return None


def compute_sa(mol: Optional[object]) -> Optional[float]:
    """Score a single RDKit ``Mol``; returns ``None`` on any failure.

    Wraps :func:`try_load` so a caller doesn't need to check the loader
    availability separately.  Returns ``None`` for ``None`` input,
    invalid ``Mol``, or when no SA scorer is importable.
    """
    if mol is None:
        return None
    fn = try_load()
    if fn is None:
        return None
    try:
        return float(fn(mol))
    except (RuntimeError, ValueError, AttributeError):
        return None


__all__ = ["try_load", "compute_sa"]