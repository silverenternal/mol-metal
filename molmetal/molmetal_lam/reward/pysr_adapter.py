"""pysr_adapter.py — thin PySR wrapper for symbolic-reward regression.

Reference: PySR (Cranmer 2023, arXiv:2305.01582). Source:
``/home/hugo/codes/try_triton_on_rocm/molmetal/references/PySR/pysr/sr.py``.

This module is the R3 deliverable of the WF-SOTA-Reuse workflow. It
REPLACES the algebraic-enumeration fitters (1247 LOC in
``symbolic_regression.py``) for the case where PySR is installed and the
Julia backend has initialised.

Honest framing (audit ``wf_sota_reuse/audit_PySR.md``):
- PySR is OPT-IN. The default code path stays on the hand-rolled
  family fitters (F1..F8) because PySR requires Julia 1.8+ and a
  ~1.5 GB first-import depot download.
- The adapter has a graceful fallback: if ``pysr`` cannot be imported,
  :func:`fit` returns ``None`` (with a clear reason) and downstream
  callers can switch to the closed-form fitters.
- The adapter never raises on a missing PySR install; this keeps the
  test suite green even when Julia is unavailable.

Public API
----------
:class:`PySRFitResult`
    Frozen dataclass mirroring the result row from
    ``PySRRegressor.equations_`` for the chosen model.
:func:`is_available`
    Returns True iff ``pysr`` is importable (and Julia can be reached).
:func:`fit`
    Fit a symbolic-regression model on (X, y). Returns
    ``PySRFitResult`` on success or ``None`` on fallback.
:func:`evaluate`
    Vectorised evaluation of a fitted expression string on new X.

Usage
-----
>>> from molmetal_lam.reward.pysr_adapter import fit
>>> result = fit(X_cell, y_metric)        # numpy arrays, shape (n, k), (n,)
>>> if result is None:
...     # fall back to hand-rolled fitters
...     ...
>>> else:
...     print(result.formula_str, result.complexity, result.loss)
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Module-level probe — done once, cached.
# -----------------------------------------------------------------------------
_PYSR_AVAILABLE: Optional[bool] = None
_PYSR_IMPORT_ERROR: Optional[str] = None


def _probe_pysr() -> bool:
    """Return True iff ``pysr.PySRRegressor`` can be imported.

    We do NOT trigger the Julia depot download here. PySR's first
    ``import pysr`` will pull ~1.5 GB on initial use; we expose that
    explicit cost at ``fit()`` time so the caller can decide.
    """
    global _PYSR_AVAILABLE, _PYSR_IMPORT_ERROR
    if _PYSR_AVAILABLE is not None:
        return _PYSR_AVAILABLE
    try:
        from pysr import PySRRegressor  # noqa: F401
        _PYSR_AVAILABLE = True
    except Exception as exc:  # noqa: BLE001
        _PYSR_AVAILABLE = False
        _PYSR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    return _PYSR_AVAILABLE


def is_available() -> bool:
    """Return True iff PySR can be imported on this Python.

    Note: a True return does NOT guarantee Julia has finished its
    first-run depot install. ``fit()`` may still raise on first call
    if the depot is empty.
    """
    return _probe_pysr()


def get_import_error() -> Optional[str]:
    """Return the captured import error (if any) from the PySR probe."""
    _probe_pysr()
    return _PYSR_IMPORT_ERROR


# -----------------------------------------------------------------------------
# Result dataclass — mirrors the schema of ``equations_.iloc[idx]``
# in PySRRegressor (see sr.py:1764 ``get_best`` and sr.py:1541 __repr__).
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PySRFitResult:
    """One fitted expression, plus enough metadata to compare across runs.

    Attributes
    ----------
    formula_str : str
        The chosen expression in PySR's string format (e.g. ``"(x0 + x1)"``).
    complexity : int
        Total operator count (sum of operator complexities in the
        chosen row of ``equations_``). Higher = more complex.
    loss : float
        PySR ``loss`` column for the chosen row. Smaller = better fit.
    score : float
        PySR ``score`` column (negative MSE-like, larger = better).
    equation_idx : int
        Index into ``PySRRegressor.equations_`` for reproducibility.
    n_samples : int
        Number of training samples (rows in X).
    n_features : int
        Number of features (columns in X).
    feature_names : tuple[str, ...]
        Names PySR was given via ``variable_names``. ``("x0", "x1", ...)``
        is the default if not supplied.
    """
    formula_str: str
    complexity: int
    loss: float
    score: float
    equation_idx: int
    n_samples: int
    n_features: int
    feature_names: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, Any]:
        """Serialisable dict view (handy for JSON reports)."""
        return {
            "formula_str": self.formula_str,
            "complexity": self.complexity,
            "loss": self.loss,
            "score": self.score,
            "equation_idx": self.equation_idx,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
        }


# -----------------------------------------------------------------------------
# Fit function — the public entry point.
# -----------------------------------------------------------------------------
def fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    binary_ops: Optional[List[str]] = None,
    unary_ops: Optional[List[str]] = None,
    niterations: int = 100,
    populations: int = 20,
    variable_names: Optional[List[str]] = None,
    niterations_warmup: int = 10,
    random_state: int = 42,
    model_selection: str = "best",
    temp_equation_file: bool = True,
    progress: bool = False,
    **extra_pysr_kwargs: Any,
) -> Optional[PySRFitResult]:
    """Fit PySR on (X, y) and return the best expression per ``model_selection``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    binary_ops : list[str], optional
        Defaults to ``["+", "-", "*", "/"]`` per the spec.
    unary_ops : list[str], optional
        Defaults to ``[]`` (no unary operators) for the spec's lite
        baseline; pass ``["exp", "log", "sqrt"]`` for a richer search.
    niterations : int
        Number of evolutionary iterations (default 100 per the spec).
    populations : int
        Number of populations running in parallel (default 20 per spec).
    variable_names : list[str], optional
        Names for X columns. Defaults to ``x0, x1, ...``.
    niterations_warmup : int
        Iterations for the initial random search before tournament
        selection kicks in. Default 10 (PySR default).
    random_state : int
        Seed for both NumPy and Julia RNG. Default 42 for reproducibility.
    model_selection : str
        PySR model-selection strategy. ``"best"`` (default) picks the
        expression that minimises loss; ``"accuracy"`` picks the
        expression with the best held-out score.
    temp_equation_file : bool
        If True (default), equations are written to a temp dir that is
        cleaned up on process exit. Set False to persist for debugging.
    progress : bool
        If True, print search progress. Default False (CI-friendly).
    **extra_pysr_kwargs
        Forwarded to ``PySRRegressor`` for advanced tuning
        (e.g. ``nested_constraints``, ``complexity_of_operators``).

    Returns
    -------
    Optional[PySRFitResult]
        - The chosen expression and metadata on success.
        - ``None`` if PySR is not importable, or if the input shape is
          invalid (e.g. n_samples < 5), or if the fit raised before any
          equation was produced.

    Notes
    -----
    - Defaults match the R3 spec: ``niterations=100``, ``populations=20``,
      ``binary_ops=['+', '-', '*', '/']``.
    - The adapter never raises on a missing PySR install; downstream
      callers can switch to the closed-form fitters when ``fit()``
      returns None.
    - On a cold install, the first call to ``fit()`` will trigger the
      Julia depot download (~1.5 GB, 3-5 min). Pass
      ``progress=True`` to see the download progress.
    """
    # --- 1. Validate inputs ---
    if binary_ops is None:
        binary_ops = ["+", "-", "*", "/"]
    if unary_ops is None:
        unary_ops = []
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.ndim != 2:
        logger.warning(
            "pysr_adapter.fit: X must be 2D, got shape %s; returning None",
            X_arr.shape,
        )
        return None
    if y_arr.ndim != 1:
        logger.warning(
            "pysr_adapter.fit: y must be 1D, got shape %s; returning None",
            y_arr.shape,
        )
        return None
    n_samples, n_features = X_arr.shape
    if n_samples < 5:
        logger.warning(
            "pysr_adapter.fit: n_samples=%d < 5; PySR search needs >=5 "
            "rows; returning None", n_samples,
        )
        return None
    if y_arr.shape[0] != n_samples:
        logger.warning(
            "pysr_adapter.fit: X and y shape mismatch (%d vs %d); "
            "returning None", n_samples, y_arr.shape[0],
        )
        return None
    if np.isnan(X_arr).any() or np.isnan(y_arr).any():
        # PySR does not tolerate NaN inputs by default.
        # Clean it here so callers don't get cryptic Julia errors.
        valid = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_arr))
        X_arr = X_arr[valid]
        y_arr = y_arr[valid]
        n_samples = X_arr.shape[0]
        if n_samples < 5:
            logger.warning(
                "pysr_adapter.fit: after NaN filter, n_samples=%d < 5; "
                "returning None", n_samples,
            )
            return None

    # --- 2. Resolve variable names ---
    if variable_names is None:
        variable_names = [f"x{i}" for i in range(n_features)]
    if len(variable_names) != n_features:
        variable_names = [f"x{i}" for i in range(n_features)]

    # --- 3. Try to import PySR ---
    if not _probe_pysr():
        logger.info(
            "pysr_adapter.fit: PySR not importable (%s); returning None",
            _PYSR_IMPORT_ERROR,
        )
        return None

    # --- 4. Build the regressor ---
    # Lazy import so the test suite stays green on hosts without Julia.
    from pysr import PySRRegressor  # type: ignore

    # Initialise with the spec defaults. ``temp_equation_file=True`` keeps
    # the file system clean for CI; pass False to debug.
    try:
        regressor = PySRRegressor(
            binary_operators=list(binary_ops),
            unary_operators=list(unary_ops),
            niterations=int(niterations),
            populations=int(populations),
            niterations_warmup=int(niterations_warmup),
            random_state=int(random_state),
            model_selection=model_selection,
            temp_equation_file=bool(temp_equation_file),
            progress=bool(progress),
            verbosity=0,
            **extra_pysr_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pysr_adapter.fit: PySRRegressor __init__ failed: %s; "
            "returning None", exc,
        )
        return None

    # --- 5. Run the search ---
    try:
        regressor.fit(X_arr, y_arr, variable_names=variable_names)
    except Exception as exc:  # noqa: BLE001
        # PySR's first call downloads the Julia depot; that failure mode
        # is the most common. Surface the underlying reason in the log.
        logger.warning(
            "pysr_adapter.fit: PySRRegressor.fit failed: %s; "
            "returning None", exc,
        )
        return None

    # --- 6. Pull the chosen equation from the hall of fame ---
    equations = getattr(regressor, "equations_", None)
    if equations is None:
        logger.warning(
            "pysr_adapter.fit: PySRRegressor.equations_ is None after fit; "
            "returning None",
        )
        return None
    # ``equations_`` is either a DataFrame (single-output) or a list
    # of DataFrames (multi-output). We only support single-output here.
    import pandas as pd  # local import; pandas is already a dep
    if isinstance(equations, list):
        if not equations:
            return None
        equations = equations[0]
    if not isinstance(equations, pd.DataFrame) or equations.empty:
        return None

    # Honour ``model_selection`` by calling ``get_best()``.
    try:
        best = regressor.get_best()
    except Exception:  # noqa: BLE001
        best = None
    if best is None or not isinstance(best, pd.Series):
        # Fall back to the first row.
        best = equations.iloc[0]

    # PySR exposes ``equation`` (str), ``complexity`` (int),
    # ``loss`` (float), ``score`` (float) as columns. The actual
    # column names depend on the PySR version; we probe safely.
    formula = str(best.get("equation", best.get("sympy_format", "")))
    try:
        complexity = int(best.get("complexity", 0))
    except (TypeError, ValueError):
        complexity = 0
    try:
        loss = float(best.get("loss", np.inf))
    except (TypeError, ValueError):
        loss = float("inf")
    try:
        score = float(best.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    # Find the row index for reproducibility.
    try:
        idx = int(best.name) if best.name is not None else 0
    except (TypeError, ValueError):
        idx = 0

    return PySRFitResult(
        formula_str=formula,
        complexity=complexity,
        loss=loss,
        score=score,
        equation_idx=idx,
        n_samples=int(n_samples),
        n_features=int(n_features),
        feature_names=tuple(variable_names),
    )


# -----------------------------------------------------------------------------
# Vectorised evaluator for a fitted formula.
# -----------------------------------------------------------------------------
def evaluate(formula_str: str, X: np.ndarray) -> np.ndarray:
    """Evaluate a fitted formula on new X using SymPy + NumPy.

    Mirrors the ``evaluate_formula`` API in
    :mod:`molmetal_lam.reward.symbolic_regression` but is self-contained
    so the adapter has no circular deps. Falls back to zeros on
    parse errors.
    """
    import sympy as sp  # local import; keeps import-time light
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n_features = X_arr.shape[1]
    names = [f"x{i}" for i in range(n_features)]
    sym_map = {n: sp.Symbol(n) for n in names}
    try:
        expr = sp.sympify(formula_str, locals=sym_map)
        fn = sp.lambdify(list(sym_map.values()), expr, modules="numpy")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"pysr_adapter.evaluate: cannot parse {formula_str!r}: {exc}")
        return np.zeros(X_arr.shape[0], dtype=float)
    return np.asarray(fn(*X_arr.T), dtype=float)


# -----------------------------------------------------------------------------
# Self-test (kept thin — full smoke lives in the test file).
# -----------------------------------------------------------------------------
def _self_test() -> None:
    """Tiny self-test that prints the availability verdict."""
    if is_available():
        print(f"[self_test] PySR available: {_PYSR_IMPORT_ERROR or 'importable'}")
    else:
        print(f"[self_test] PySR NOT available: {_PYSR_IMPORT_ERROR}")
        print("[self_test] fit() will return None on every call.")


if __name__ == "__main__":
    _self_test()