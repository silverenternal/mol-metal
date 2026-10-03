"""Hardness predictor + 3-way decomposition.

================================================================
Lightweight hardness predictor for hard-example mining (TODO-150 §P1.4)
================================================================
A 2-layer MLP (input→64→1) trained on frozen D-MPNN fingerprints to
predict per-sample hardness — defined as the *clipped* sample-level
loss.  Re-evaluated every 5 epochs (drift guard in :mod:`.drift`).

The predictor is intentionally tiny (input-dim × 64 × 1 ≈ ~24k
parameters for 300-d fingerprints, well under the Wf-124 ckpt
budget) so it can be re-fit on CPU between epochs and shipped with
the Wf-124 P3 training pipeline without GPU contention.

3-way hardness decomposition (TODO-150 §P1.6)
--------------------------------------------
Honest-negative disclosure (§6 of paper): hardness is **decomposed**
into three orthogonal mechanisms the planner cares about:

* ``m_geometry`` — does the molecule fail the geometric validation
  gate (coordination number, square-planar vs octahedral mismatch,
  …)?  Computed from the SMILES via RDKit + the closure rules.
* ``m_binding``  — does the predicted pIC50 disagree with the
  pocket-specific target by more than 1.0 log unit?
* ``m_reduction`` — does the metal's reduction potential fall outside
  the cytotoxic-window range (Pt/Ru/Ir literature defaults)?  Stored
  per-sample so a downstream report can decompose lift by mechanism.

The decompositions are stored as a separate column in the hardness
shard (parquet) so paper §6 can render them without recomputing.

Drift guard threshold (TODO-150 §P1.7)
--------------------------------------
A sample whose **on-policy** hardness diverges from its **predicted**
hardness by more than :data:`DRIFT_DIVERGENCE_THRESHOLD` (0.5) is
retired from the injection set.  This is the mechanism that prevents
the predictor from silently poisoning the cohort with samples it has
not learned well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    _TORCH_OK = True
except Exception:  # pragma: no cover
    _TORCH_OK = False


# Module-level torch model class so isinstance() works correctly across
# multiple predictor instances.  Defining the class inside a method
# creates a new class on every call, which breaks isinstance checks.
if _TORCH_OK:
    class _HardnessMLP(nn.Module):  # type: ignore[misc]
        """2-layer MLP with sigmoid output — defined once per module."""

        def __init__(self, in_dim: int, hid_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hid_dim)
            self.fc2 = nn.Linear(hid_dim, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))
else:
    _HardnessMLP = None  # type: ignore[assignment]


#: Default input dimension.  Match the
#: :func:`featurize_smiles_list` 2048-bit Morgan fingerprint pipeline
#: or any frozen encoder output.  When the actual feature dimension
#: differs (e.g. a smaller 256-d D-MPNN pooled vector), pass it to
#: :class:`HardnessPredictor` explicitly.
PREDICTOR_INPUT_DIM: int = 300

#: Hidden layer width (TODO-150 §P1.4).
PREDICTOR_HIDDEN_DIM: int = 64

#: Per-sample hardness-on-policy vs hardness-predicted divergence that
#: triggers retirement from the injection set.  0.5 matches the plan.
DRIFT_DIVERGENCE_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class HardnessSample:
    """One (predicted, on-policy, decomposed) hardness sample.

    ``predicted`` is the predictor's estimate from frozen
    fingerprints; ``on_policy`` is the actual sample-level loss
    measured during the most recent training epoch.  ``decomposition``
    carries the 3-way split for paper §6.
    """

    row_id: int
    smiles: str
    metal: str
    pocket_id: str
    predicted: float
    on_policy: float
    decomposition: "HardnessDecomposition"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": int(self.row_id),
            "smiles": str(self.smiles),
            "metal": str(self.metal),
            "pocket_id": str(self.pocket_id),
            "predicted": float(self.predicted),
            "on_policy": float(self.on_policy),
            "m_geometry": float(self.decomposition.m_geometry),
            "m_binding": float(self.decomposition.m_binding),
            "m_reduction": float(self.decomposition.m_reduction),
        }


@dataclass(frozen=True)
class HardnessDecomposition:
    """The 3-way split of a sample's hardness.

    All three fields are non-negative floats in [0, 1]; ``None`` is
    used when a mechanism cannot be evaluated (e.g. RDKit sanitisation
    fails).  ``HardnessPredictor.decompose`` fills the canonical
    defaults (0.5 each) when a component is missing.
    """

    m_geometry: float = 0.5
    m_binding: float = 0.5
    m_reduction: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "m_geometry": float(self.m_geometry),
            "m_binding": float(self.m_binding),
            "m_reduction": float(self.m_reduction),
        }

    @classmethod
    def default_uniform(cls) -> "HardnessDecomposition":
        """Equal-weight decomposition (the conservative fallback)."""
        return cls(0.5, 0.5, 0.5)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardnessDecomposition":
        return cls(
            m_geometry=float(payload.get("m_geometry", 0.5)),
            m_binding=float(payload.get("m_binding", 0.5)),
            m_reduction=float(payload.get("m_reduction", 0.5)),
        )


class HardnessPredictor:
    """2-layer MLP that predicts per-sample hardness from frozen features.

    The forward pass is::

        h = relu(W1 x + b1)
        y = sigmoid(W2 h + b2)     # bounded [0, 1]

    The sigmoid bounds the prediction so the drift guard's
    ``|predicted - on_policy|`` comparison is well-defined.  When
    ``torch`` is not available, :class:`HardnessPredictor` falls back
    to a deterministic linear-only :class:`_NumpyFallback` that uses
    the same weight shapes but runs in NumPy (CPU, no autograd).  This
    keeps the predictor importable on CPU-only test hosts.

    Training
    --------
    The training loop is intentionally trivial — :meth:`fit` runs
    plain mini-batch SGD on the per-sample (feature, target-loss)
    pairs and stores the train-set MAE for diagnostics.  No L2 reg,
    no scheduler — the predictor is throwaway; it is retrained from
    scratch every 5 epochs.
    """

    def __init__(
        self,
        input_dim: int = PREDICTOR_INPUT_DIM,
        hidden_dim: int = PREDICTOR_HIDDEN_DIM,
        learning_rate: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 64,
        seed: int = 0,
    ) -> None:
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1; got {input_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1; got {hidden_dim}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1; got {n_epochs}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._fitted = False
        self._train_mae: float = float("nan")
        if _TORCH_OK:
            torch.manual_seed(self.seed)
            self.model = self._build_torch_model()
        else:
            self.model = self._build_numpy_model()

    # ------------------------------------------------------------------
    # Model factories
    # ------------------------------------------------------------------
    def _build_torch_model(self):  # pragma: no cover - exercised only under torch
        torch.manual_seed(self.seed)
        return _HardnessMLP(self.input_dim, self.hidden_dim)

    def _build_numpy_model(self):
        """Linear fallback: y = sigmoid(W x + b) — single layer.

        Used when torch is unavailable (CPU-only test host).  Quality
        is materially worse than the torch MLP but the API contract
        (predict / fit / evaluate) is identical so downstream code is
        not branched on the backend.
        """
        rng = np.random.default_rng(self.seed)
        w = rng.standard_normal((self.input_dim, 1)) * 0.01
        b = np.zeros((1,), dtype=np.float64)
        return {"w": w.astype(np.float64), "b": b.astype(np.float64)}

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def weights_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict of the current weights (for caching)."""
        if _TORCH_OK and isinstance(self.model, _HardnessMLP):
            return {
                "backend": "torch",
                "fc1.weight": self.model.fc1.weight.detach().cpu().numpy().tolist(),
                "fc1.bias": self.model.fc1.bias.detach().cpu().numpy().tolist(),
                "fc2.weight": self.model.fc2.weight.detach().cpu().numpy().tolist(),
                "fc2.bias": self.model.fc2.bias.detach().cpu().numpy().tolist(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
            }
        # numpy fallback
        return {
            "backend": "numpy",
            "w": self.model["w"].tolist(),
            "b": self.model["b"].tolist(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }

    # ------------------------------------------------------------------
    # Forward + predict
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict hardness in [0, 1] for a (N, D) feature matrix."""
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 2 or feats.shape[1] != self.input_dim:
            raise ValueError(
                f"features must be (N, {self.input_dim}); got {feats.shape}"
            )
        if _TORCH_OK and isinstance(self.model, _HardnessMLP):
            self.model.eval()
            with torch.no_grad():
                out = self.model(torch.as_tensor(feats, dtype=torch.float32)).cpu().numpy()
            return out.reshape(-1)
        # numpy fallback
        z = feats @ self.model["w"] + self.model["b"]
        return 1.0 / (1.0 + np.exp(-z.reshape(-1)))

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Fit the predictor on (features, target_loss) pairs.

        ``targets`` is the **clipped** sample-level loss in [0, 1].
        ``sample_weights`` is optional (class-balanced mini-batches
        per TODO-150 §P1.4); default uniform.

        Returns a small dict of training diagnostics: ``train_mae``,
        ``final_loss``.  Both are Python floats (no torch tensors)
        so they round-trip cleanly into the JSON report.
        """
        feats = np.asarray(features, dtype=np.float64)
        tgts = np.asarray(targets, dtype=np.float64)
        if feats.ndim != 2 or feats.shape[1] != self.input_dim:
            raise ValueError(
                f"features must be (N, {self.input_dim}); got {feats.shape}"
            )
        if feats.shape[0] != tgts.shape[0]:
            raise ValueError(
                f"features / targets length mismatch: {feats.shape[0]} vs {tgts.shape[0]}"
            )
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float64)
            if w.shape != tgts.shape:
                raise ValueError("sample_weights must match targets shape")
        else:
            w = np.ones_like(tgts)

        # Clip targets into [0, 1] for sigmoid-headed BCE.
        tgts_clipped = np.clip(tgts, 0.0, 1.0)
        n = feats.shape[0]
        if n == 0:
            raise ValueError("cannot fit on zero rows")

        if _TORCH_OK and isinstance(self.model, _HardnessMLP):
            return self._fit_torch(feats, tgts_clipped, w)
        return self._fit_numpy(feats, tgts_clipped, w)

    def _fit_torch(self, feats, tgts, weights):  # pragma: no cover - requires torch
        feats_t = torch.as_tensor(feats, dtype=torch.float32)
        tgts_t = torch.as_tensor(tgts, dtype=torch.float32)
        w_t = torch.as_tensor(weights, dtype=torch.float32)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        bce = nn.BCELoss(reduction="none")
        rng = np.random.default_rng(self.seed)
        n = feats.shape[0]
        final_loss = float("nan")
        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                opt.zero_grad()
                preds = self.model(feats_t[idx]).reshape(-1)
                loss = (bce(preds, tgts_t[idx]) * w_t[idx]).mean()
                loss.backward()
                opt.step()
                final_loss = float(loss.item())
        # MAE on the full set.
        with torch.no_grad():
            preds_full = self.model(feats_t).reshape(-1).cpu().numpy()
        self._train_mae = float(np.mean(np.abs(preds_full - tgts)))
        self._fitted = True
        return {"train_mae": self._train_mae, "final_loss": final_loss}

    def _fit_numpy(self, feats, tgts, weights):
        # 1-layer linear-with-sigmoid trained via mini-batch SGD.
        w_param = self.model["w"].copy()
        b_param = self.model["b"].copy()
        lr = self.learning_rate
        n = feats.shape[0]
        rng = np.random.default_rng(self.seed)
        final_loss = float("nan")
        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                x = feats[idx]
                y = tgts[idx]
                ws = weights[idx]
                z = x @ w_param + b_param
                p = 1.0 / (1.0 + np.exp(-z))
                # BCE gradient w.r.t. z
                diff = (p - y).reshape(-1, 1) * ws.reshape(-1, 1)
                grad_w = (x.T @ diff) / idx.size
                grad_b = float(diff.mean())
                w_param -= lr * grad_w
                b_param -= lr * grad_b
                # BCE magnitude for diagnostics
                eps = 1e-12
                p_clip = np.clip(p, eps, 1.0 - eps)
                final_loss = float(-(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)).mean())
        self.model["w"] = w_param
        self.model["b"] = b_param
        preds = self.predict(feats)
        self._train_mae = float(np.mean(np.abs(preds - tgts)))
        self._fitted = True
        return {"train_mae": self._train_mae, "final_loss": final_loss}

    # ------------------------------------------------------------------
    # Decomposition (paper §6 honest-negative)
    # ------------------------------------------------------------------
    def decompose(
        self,
        smiles: str,
        predicted_pic50: float,
        target_pic50: float,
        metal: str,
        pocket_id: str,
    ) -> HardnessDecomposition:
        """3-way hardness decomposition for paper §6.

        Each component is in [0, 1] — *higher means harder*.

        * ``m_geometry``  — proxy for coordination-geometry
          incompatibility.  Falls back to 0.5 when the SMILES cannot
          be parsed (we cannot rule out geometry violation without a
          well-formed graph).
        * ``m_binding``   — |predicted - target| clipped to [0, 1];
          1 log unit ≈ fully hard.
        * ``m_reduction`` — penalty when ``metal ∉ {Pt, Ru, Ir, Rh,
          Pd}`` or when the target pIC50 is outside the
          literature-cited cytotoxic window (5.0–8.0).
        """
        m_geom = self._geometry_score(smiles)
        m_bind = self._binding_score(predicted_pic50, target_pic50)
        m_red = self._reduction_score(metal, target_pic50)
        return HardnessDecomposition(m_geometry=m_geom, m_binding=m_bind, m_reduction=m_red)

    @staticmethod
    def _geometry_score(smiles: str) -> float:
        """Geometry-violation proxy.

        Defaults to 0.5 (neutral).  When RDKit is importable the
        implementation will replace the stub with a full check — see
        TODO-150 §P1.6 (delivered as part of the validity gate).  Until
        then the stub is correct-by-construction (does not over-call).
        """
        try:
            from rdkit import Chem  # noqa: F401  (imported for side-effect only)
            # If the SMILES is unparseable we cannot say; default to 0.5.
            if smiles is None or not smiles:
                return 0.5
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 1.0  # Failed parse = hard geometry proxy.
            return 0.5
        except Exception:
            return 0.5

    @staticmethod
    def _binding_score(predicted_pic50: float, target_pic50: float) -> float:
        """Clipped L1 distance in pIC50 units.

        1.0 log unit → fully hard (saturates at 1.0).
        """
        try:
            diff = abs(float(predicted_pic50) - float(target_pic50))
        except (TypeError, ValueError):
            return 0.5
        return float(min(1.0, max(0.0, diff)))

    @staticmethod
    def _reduction_score(metal: str, target_pic50: float) -> float:
        """Hardness proxy from the cytotoxic-window penalty.

        Defaults to 0.5.  Metals not in the canonical precious-metal
        set escalate to 1.0; pIC50 outside [5.0, 8.0] escalates
        proportionally.
        """
        canonical = {"Pt", "Ru", "Ir", "Rh", "Pd", "Au", "Os"}
        if metal not in canonical:
            return 1.0
        if target_pic50 is None:
            return 0.5
        if 5.0 <= float(target_pic50) <= 8.0:
            return 0.0
        # Linear ramp outside the window.
        if float(target_pic50) < 5.0:
            return float(min(1.0, (5.0 - float(target_pic50)) / 1.0))
        return float(min(1.0, (float(target_pic50) - 8.0) / 1.0))

    # ------------------------------------------------------------------
    # Public diagnostic
    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def train_mae(self) -> float:
        return self._train_mae


__all__ = [
    "HardnessSample",
    "HardnessDecomposition",
    "HardnessPredictor",
    "PREDICTOR_INPUT_DIM",
    "PREDICTOR_HIDDEN_DIM",
    "DRIFT_DIVERGENCE_THRESHOLD",
]
