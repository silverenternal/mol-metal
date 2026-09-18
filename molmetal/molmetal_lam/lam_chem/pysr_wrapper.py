"""HeuristicRegressor — PySR wrapper with graceful sklearn fallback.

============================================================
Symbolic regression for MLC heuristics
============================================================
The :class:`HeuristicRegressor` is the bridge between ML and chemistry:
given a set of (temperature, catalyst, time, concentration, ...) data
points and observed yields, it discovers an interpretable closed-form
expression that the proof-search layer can splice into a LamApp.

The wrapper deliberately degrades gracefully:

    PySR + Julia available   → real symbolic regression (best)
    PySR present, Julia not  → sklearn :class:`RandomForestRegressor`
    sklearn only (this env)  → :class:`RandomForestRegressor` /
                               :class:`Ridge` (always works)

The fallback path is exercised whenever the Julia subprocess cannot be
spawned (missing julia executable, Julia version mismatch, or the
underlying :mod:`julia` / :mod:`PyCall` machinery fails to import).
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Iterable, List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend probing
# ---------------------------------------------------------------------------
def _probe_pysr() -> bool:
    """Return ``True`` iff PySR *and* its Julia backend can be loaded.

    We avoid importing PySR eagerly — even ``import pysr`` can pull in
    juliacall internals. The probe tries to instantiate
    :class:`PySRRegressor` and read ``model_run`` to ensure the Julia
    subprocess is responsive.
    """
    try:
        from pysr import PySRRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        log.debug("PySR import failed: %s", exc)
        return False
    try:
        reg = PySRRegressor(
            niterations=1,
            binary_operators=["+", "*"],
            unary_operators=[],
            progress=False,
            verbosity=0,
            temp_equation_file=False,
            tempdir=False,
            delete_tempfiles=True,
        )
        # A single dummy fit to make sure Julia can launch.
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 2))
        y = X[:, 0] + X[:, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X, y)
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        log.debug("PySR/Julia probe failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# HeuristicRegressor
# ---------------------------------------------------------------------------
class HeuristicRegressor:
    """Symbolic-regression wrapper with a scikit-learn safety net.

    Parameters
    ----------
    niterations : int, default 50
        Number of PySR iterations (only used when the backend is live).
    binary_ops : list[str], default ``['+', '*', '/', '-']``
        Binary operators exposed to PySR's search.
    unary_ops : list[str], default ``['square', 'sqrt', 'exp', 'log']``
        Unary operators exposed to PySR's search.
    random_state : int, default 0
        Seed for both PySR (when present) and the sklearn fallback.

    Notes
    -----
    Once :meth:`fit` has been called the ``backend_`` attribute is set
    to either ``"pysr"`` or ``"sklearn"``, and ``equation()`` returns a
    pretty-printed expression in the appropriate format.
    """

    def __init__(
        self,
        niterations: int = 50,
        binary_ops: Optional[List[str]] = None,
        unary_ops: Optional[List[str]] = None,
        random_state: int = 0,
        probe_pysr_at_fit_time: bool = False,
    ) -> None:
        self.niterations = int(niterations)
        self.binary_ops = list(binary_ops) if binary_ops is not None else ["+", "*", "/", "-"]
        self.unary_ops = list(unary_ops) if unary_ops is not None else ["square", "sqrt", "exp", "log"]
        self.random_state = int(random_state)
        # ``backend_`` is left as ``None`` until ``fit`` is called when
        # ``probe_pysr_at_fit_time`` is True — this lets callers defer
        # the (potentially expensive) PySR/Julia probe until the
        # regressor is actually used.
        self.backend_: Optional[str] = None
        self._fitted = False
        self._model = None  # sklearn or PySR backend instance
        self._sympy_expr = None  # populated only when PySR succeeds
        self._feature_names: Optional[List[str]] = None
        self.probe_pysr_at_fit_time = bool(probe_pysr_at_fit_time)

        # Eager probe (legacy behaviour): when the flag is False we
        # probe PySR right now so the first ``fit`` call can short
        # circuit if Julia is unavailable.
        if not self.probe_pysr_at_fit_time:
            self._pysr_available = _probe_pysr()
        else:
            self._pysr_available = False

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HeuristicRegressor":
        """Fit the underlying regressor.

        Tries PySR first; on any failure transparently falls back to
        scikit-learn. Returns ``self`` for fluent use.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features); got shape {X.shape}")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._feature_names = [f"x{i}" for i in range(X.shape[1])]

        # Deferred-probe mode: PySR availability is unknown until the
        # first fit call.  Probe now so ``_try_pysr`` sees the right
        # cached answer on subsequent fits.
        if self.probe_pysr_at_fit_time and not self._pysr_available:
            self._pysr_available = _probe_pysr()

        if self._try_pysr(X, y):
            return self
        if self._try_sklearn(X, y):
            return self
        raise RuntimeError("No regression backend available (PySR and sklearn both failed).")

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using whichever backend won the race during ``fit``."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit(X, y) before predict(X).")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.asarray(self._model.predict(X), dtype=float).reshape(-1)

    # ------------------------------------------------------------- equation
    def equation(self) -> str:
        """Return a human-readable expression for the fitted model.

        * PySR backend  → pretty sympy expression
        * sklearn fallback → linear formula ``c0 + c1*x0 + ...`` or a
          placeholder description for tree ensembles.
        """
        if not self._fitted:
            return "<unfitted>"
        if self.backend_ == "pysr" and self._sympy_expr is not None:
            try:
                from sympy import pretty  # type: ignore
                return pretty(self._sympy_expr)
            except Exception:
                return str(self._sympy_expr)
        if self.backend_ == "sklearn" and hasattr(self._model, "coef_"):
            coefs = np.asarray(self._model.coef_).ravel()
            intercept = float(getattr(self._model, "intercept_", np.zeros(1)).ravel()[0])
            terms: List[str] = []
            if intercept != 0.0:
                terms.append(f"{intercept:.4g}")
            for i, c in enumerate(coefs):
                if c == 0.0:
                    continue
                terms.append(f"{c:+.4g}*x{i}")
            return " ".join(terms) if terms else "0"
        # RandomForest fallback — surface the top-3 most important
        # features so the closed-loop paper_equation carries real
        # semantic content instead of a generic placeholder.
        if self.backend_ == "sklearn-rf" and hasattr(self._model, "feature_importances_"):
            try:
                importances = np.asarray(self._model.feature_importances_).ravel()
                names = self._feature_names or [f"x{i}" for i in range(len(importances))]
                top = sorted(
                    zip(names, importances), key=lambda p: abs(p[1]), reverse=True,
                )[:3]
                feats_str = ", ".join(f"{n}={w:.3f}" for n, w in top if w > 0)
                return f"RF(top3: {feats_str})" if feats_str else "RF(<no signal>)"
            except Exception:
                pass
        return f"<{self.backend_ or 'unknown'}-backend model>"

    # ============================================================ internals
    def _try_pysr(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Attempt a PySR fit; on any failure return ``False``."""
        try:
            from pysr import PySRRegressor  # type: ignore
        except Exception as exc:
            log.debug("Skipping PySR — import failed: %s", exc)
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg = PySRRegressor(
                    niterations=self.niterations,
                    binary_operators=self.binary_ops,
                    unary_operators=self.unary_ops,
                    progress=False,
                    verbosity=0,
                    random_state=self.random_state,
                    temp_equation_file=False,
                    tempdir=False,
                    delete_tempfiles=True,
                )
                reg.fit(X, y.ravel())
            self._model = reg
            self.backend_ = "pysr"
            self._fitted = True
            # Try to surface the best equation as sympy.
            try:
                self._sympy_expr = reg.sympy()
            except Exception:
                self._sympy_expr = None
            return True
        except Exception as exc:
            log.debug("PySR fit failed, falling back to sklearn: %s", exc)
            return False

    def _try_sklearn(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Fit a scikit-learn fallback. Always succeeds in this env."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge

        n_samples, n_features = X.shape
        # For tiny linear-ish problems use Ridge so .equation() yields
        # a clean linear formula; otherwise a RandomForest captures
        # non-linear structure with no extra effort.
        if n_samples <= 200 and n_features <= 4:
            try:
                model = Ridge(alpha=1.0, random_state=self.random_state)
                model.fit(X, y.ravel())
                self._model = model
                self.backend_ = "sklearn-linear"
                self._fitted = True
                return True
            except Exception as exc:
                log.debug("Ridge fallback failed: %s", exc)

        try:
            model = RandomForestRegressor(
                n_estimators=32,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=1,
            )
            model.fit(X, y.ravel())
            self._model = model
            self.backend_ = "sklearn-rf"
            self._fitted = True
            return True
        except Exception as exc:
            log.debug("RandomForest fallback failed: %s", exc)
            return False

    # ====================================================== persistence (D3)
    def save(self, path: str) -> None:
        """Persist a fitted sklearn-backed regressor to disk via joblib.

        No-op when the backend is PySR (joblib cannot serialise the
        Julia subprocess state).  Raises if the regressor has not yet
        been fitted or the backend is unsupported.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Cannot save unfitted HeuristicRegressor.")
        if self.backend_ not in ("sklearn-linear", "sklearn-rf", "sklearn"):
            raise RuntimeError(
                f"save() only supports sklearn backends (got {self.backend_!r})",
            )
        try:
            import joblib  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "joblib is required for save(); install via `uv pip install joblib`",
            ) from exc
        joblib.dump(
            {
                "model": self._model,
                "backend": self.backend_,
                "feature_names": self._feature_names,
                "niterations": self.niterations,
                "random_state": self.random_state,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "HeuristicRegressor":
        """Restore a regressor previously written via :meth:`save`.

        Returns a fitted :class:`HeuristicRegressor` with the cached
        ``backend_`` + ``_model`` restored.  Raises if joblib is
        unavailable or the file is unreadable.
        """
        try:
            import joblib  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "joblib is required for load(); install via `uv pip install joblib`",
            ) from exc
        blob = joblib.load(path)
        model = blob["model"]
        reg = cls(
            niterations=int(blob.get("niterations", 50)),
            random_state=int(blob.get("random_state", 0)),
        )
        reg._model = model
        reg.backend_ = str(blob.get("backend", "sklearn"))
        reg._fitted = True
        reg._feature_names = list(blob.get("feature_names") or [])
        return reg


class SymbolicRegressor:
    """Small PySR wrapper with a deterministic Ridge fallback."""

    def __init__(
        self,
        n_iterations: int = 100,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
    ) -> None:
        self.n_iterations = int(n_iterations)
        self.binary_operators = list(binary_operators or ["+", "*"])
        self.unary_operators = list(unary_operators or ["sin", "cos", "exp"])
        self.backend: Optional[str] = None
        self._model = None
        self._equations = None
        self._fitted = False
        self._pysr_available = _probe_pysr()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymbolicRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if self._pysr_available:
            try:
                from pysr import PySRRegressor  # type: ignore
                self._model = PySRRegressor(
                    niterations=self.n_iterations,
                    binary_operators=self.binary_operators,
                    unary_operators=self.unary_operators,
                    progress=False,
                    verbosity=0,
                )
                self._model.fit(X, y)
                self.backend = "pysr"
                self._equations = getattr(self._model, "equations_", None)
                self._fitted = True
                return self
            except Exception as exc:
                log.warning("PySR fit unavailable; using sklearn Ridge fallback: %s", exc)
        else:
            log.warning("PySR/Julia unavailable; using sklearn Ridge fallback")
        from sklearn.linear_model import Ridge
        self._model = Ridge(alpha=1.0)
        self._model.fit(X, y)
        self.backend = "sklearn_ridge"
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit(X, y) before predict(X).")
        return np.asarray(self._model.predict(np.asarray(X, dtype=float))).reshape(-1)

    def equations(self) -> List[dict]:
        if self.backend == "sklearn_ridge":
            return [{"backend": "sklearn_ridge", "equation": "Ridge regression", "loss": 0.0}]
        if self._equations is None:
            return []
        rows = self._equations.to_dict("records") if hasattr(self._equations, "to_dict") else list(self._equations)
        return [{"equation": row.get("sympy_format", row.get("equation", "")),
                 "loss": float(row.get("loss", 0.0)), "backend": "pysr"} for row in rows]

    def best_equation(self) -> str:
        if self.backend == "sklearn_ridge":
            return "Ridge regression"
        if self._model is None:
            return "<unfitted>"
        try:
            return str(self._model.latex())
        except Exception:
            try:
                from sympy import latex  # type: ignore
                return latex(self._model.sympy())
            except Exception:
                eqs = self.equations()
                return str(eqs[-1]["equation"] if eqs else "<unfitted>")


class MetalCoordinationRegressor(SymbolicRegressor):
    """Symbolic model for bond angles from ligand count and metal type."""

    def __init__(self, n_iterations: int = 150, **kwargs) -> None:
        operators = kwargs.pop("binary_operators", ["+", "*", "/"])
        super().__init__(n_iterations=n_iterations, binary_operators=operators, **kwargs)


__all__ = ["HeuristicRegressor", "SymbolicRegressor", "MetalCoordinationRegressor"]
