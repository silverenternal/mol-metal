"""Morgan FP + LightGBM baseline (parallel to ``morgan_xgb``).

LightGBM is the standard GBDT alternative to XGBoost (Ke et al., 2017,
https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
and is commonly reported alongside it as a classical ML control on
cheminformatics benchmarks (MoleculeNet, TDC).  We expose a drop-in
:class:`MorganLightGBMBaseline` that mirrors :class:`MorganXGBBaseline` so
that a ``--model {xgb,lightgbm}`` flag can dispatch to either one with no
behavioural change to the JSON report.

The import is *guarded*: ``import lightgbm`` is wrapped in ``try/except
ImportError`` so the module remains importable on hosts where LightGBM
is not installed (the same fallback policy as the XGBoost baseline —
:class:`MorganXGBBaseline` raises inside ``run()`` if xgboost is missing).

Honest framing
--------------
This is a *classical ML baseline*.  We do **not** claim any SOTA from
these numbers — they exist so that the paper's §4 can state that the
proposed Lambda / CFM de novo pipeline is "competitive with the strong
Morgan-FP + GBDT ceiling on the MetalCytoToxDB activity-prediction
task".  Anything beyond that claim would be an over-reach.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from molmetal.baselines.eval_utils import (
    compute_metrics,
    morgan_features,
    per_cell_line_auc,
)
from molmetal.baselines.morgan_xgb import (
    MORGAN_NBITS,
    MORGAN_RADIUS,
    SPLITTER_FACTORIES,
    MorganXGBResult,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset


# Try-import lightgbm — kept top-level so callers can check `lightgbm is
# None` for graceful fallback.  On ImportError the class still loads.
try:  # pragma: no cover - depends on optional dep
    import lightgbm as _lgb
    _LIGHTGBM_AVAILABLE = True
    _LIGHTGBM_VERSION = getattr(_lgb, "__version__", "unknown")
except ImportError:  # pragma: no cover
    _lgb = None
    _LIGHTGBM_AVAILABLE = False
    _LIGHTGBM_VERSION = None


def lightgbm_available() -> bool:
    """Return True iff lightgbm could be imported."""
    return _LIGHTGBM_AVAILABLE


def lightgbm_version() -> Optional[str]:
    """Return lightgbm.__version__ or None."""
    return _LIGHTGBM_VERSION


# ---------------------------------------------------------------------------
# Hyperparameters (parallel to XGB_PARAMS — same depth / lr / subsample)
# ---------------------------------------------------------------------------
LIGHTGBM_PARAMS: Dict[str, object] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,           # LightGBM requires explicit freq for bagging
    "colsample_bytree": 0.8,
    "objective": "binary",
    "metric": "auc",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
    "min_child_samples": 5,        # keep small so the imbalanced positives still fire
}

RANDOM_SEED = 42


# Re-export MorganXGBResult so callers can treat the two baselines uniformly
__all__ = [
    "MorganLightGBMBaseline",
    "MorganXGBResult",
    "LIGHTGBM_PARAMS",
    "MORGAN_RADIUS",
    "MORGAN_NBITS",
    "SPLITTER_FACTORIES",
    "lightgbm_available",
    "lightgbm_version",
]


# ---------------------------------------------------------------------------
# Baseline class
# ---------------------------------------------------------------------------
class MorganLightGBMBaseline:
    """Morgan-fingerprint + LightGBM classifier for cytotoxicity.

    Interface mirrors :class:`MorganXGBBaseline` so the existing
    ``molmetal/scripts/baselines.py`` can dispatch to either model via
    a single ``--model`` flag.
    """

    def __init__(
        self,
        metal: str = "Ru",
        morgan_radius: int = MORGAN_RADIUS,
        morgan_nbits: int = MORGAN_NBITS,
        lightgbm_params: Optional[Dict[str, object]] = None,
        seed: int = RANDOM_SEED,
        splitter: Optional[callable] = None,
    ) -> None:
        self.metal = metal
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.seed = int(seed)
        self.lightgbm_params: Dict[str, object] = {
            **LIGHTGBM_PARAMS,
            **(lightgbm_params or {}),
        }
        self.lightgbm_params["random_state"] = self.seed
        self.splitter = splitter
        # Hold split artifacts for downstream analysis
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None
        self._lgb_model = None

    # ------------------------------------------------------------------
    # Data loading & splitting (identical to XGB so splits are comparable)
    # ------------------------------------------------------------------
    def _load_dataset(self) -> MetalCytotoxDataset:
        flt = CytotoxFilter(
            time_threshold=24.0,
            ic50_min=0.01,
            metal_whitelist=[self.metal],
            compute_pic50=True,
            compute_active=True,
        )
        ds = MetalCytotoxDataset.from_csv(filters=flt)
        if len(ds) == 0:
            raise RuntimeError(
                f"No rows after filtering for metal={self.metal!r} "
                f"(time<=24h, IC50>=0.01uM)."
            )
        return ds

    def _split(
        self,
        dataset: MetalCytotoxDataset,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the configured splitter (default: 80/10/10 stratified)."""
        if self.splitter is not None:
            result = self.splitter(dataset)
            return result.train_idx, result.val_idx, result.test_idx
        y = np.asarray(dataset.active).astype(int)
        idx = np.arange(len(y))
        idx_train, idx_tmp, _, _ = train_test_split(
            idx, y, test_size=0.2, stratify=y, random_state=self.seed
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_tmp,
            np.asarray(y[idx_tmp]).astype(int),
            test_size=0.5,
            stratify=np.asarray(y[idx_tmp]).astype(int),
            random_state=self.seed,
        )
        return idx_train, idx_val, idx_test

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> MorganXGBResult:
        """Train and evaluate the baseline end-to-end. Returns metrics.

        Raises
        ------
        RuntimeError
            If ``lightgbm`` is not installed in the active environment.
        """
        if not _LIGHTGBM_AVAILABLE:
            raise RuntimeError(
                "lightgbm is not installed in this environment; "
                "MorganLightGBMBaseline.run() requires it.  "
                "Install via `uv pip install lightgbm` or `pip install lightgbm`."
            )

        # 1. Load + filter
        ds = self._load_dataset()
        smiles = ds.smiles
        y_all = ds.active.astype(int)
        cell_lines = ds.df["Cell_line"].astype(str).to_numpy()
        keep = np.array([bool(s) for s in smiles], dtype=bool)
        smiles = smiles[keep]
        y_all = y_all[keep]
        cell_lines = cell_lines[keep]
        if len(y_all) == 0:
            raise RuntimeError(
                f"After dropping empty SMILES nothing remains for {self.metal}."
            )

        # 2. 80/10/10 split (strategy depends on self.splitter)
        idx_train, idx_val, idx_test = self._split(ds)
        self._train_idx, self._val_idx, self._test_idx = idx_train, idx_val, idx_test

        # 3. Featurise
        x_all = morgan_features(smiles, self.morgan_radius, self.morgan_nbits)
        x_train, y_train = x_all[idx_train], y_all[idx_train]
        x_val, y_val = x_all[idx_val], y_all[idx_val]
        x_test, y_test = x_all[idx_test], y_all[idx_test]
        cl_test = cell_lines[idx_test]

        # 4. Class-weight balancing (is_unbalance=True is the LightGBM
        # equivalent of xgboost's scale_pos_weight).
        pos = max(1, int(y_train.sum()))
        neg = max(1, int(len(y_train) - pos))
        params = dict(self.lightgbm_params)
        params["scale_pos_weight"] = float(neg) / float(pos)

        # 5. Train
        from lightgbm import LGBMClassifier  # local import keeps dep optional

        model = LGBMClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="auc",
        )
        self._lgb_model = model

        # 6. Predict + evaluate
        train_score = model.predict_proba(x_train)[:, 1]
        val_score = model.predict_proba(x_val)[:, 1]
        test_score = model.predict_proba(x_test)[:, 1]
        train_metrics = compute_metrics(y_train, train_score)
        val_metrics = compute_metrics(y_val, val_score)
        test_metrics = compute_metrics(y_test, test_score)
        per_cl = per_cell_line_auc(cl_test, y_test, test_score)

        return MorganXGBResult(
            metal=self.metal,
            model="lightgbm",
            test_metrics=test_metrics,
            val_metrics=val_metrics,
            per_cell_line=per_cl,
            n_train=int(len(idx_train)),
            n_val=int(len(idx_val)),
            n_test=int(len(idx_test)),
            pos_rate_train=float(y_train.mean()),
            pos_rate_test=float(y_test.mean()),
        )


# ---------------------------------------------------------------------------
# Convenience: smoke-test on tiny subset
# ---------------------------------------------------------------------------
def quick_smoke(
    metal: str = "Ru",
    max_rows: int = 200,
) -> MorganXGBResult:
    """Quick training run on ``max_rows`` rows — used for unit tests."""
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[metal],
    )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    df = ds.df.head(max_rows).copy()
    ds_small = MetalCytotoxDataset(df=df)
    baseline = MorganLightGBMBaseline(metal=metal)
    baseline._load_dataset = lambda: ds_small  # type: ignore[assignment]
    return baseline.run()