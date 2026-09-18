"""Morgan FP + RandomForest baseline (classical control)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from molmetal.baselines.eval_utils import (
    compute_metrics,
    morgan_features,
    per_cell_line_auc,
)
from molmetal.baselines.morgan_xgb import MorganXGBBaseline, MORGAN_RADIUS, MORGAN_NBITS
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset


RF_PARAMS: Dict[str, object] = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": 42,
    "class_weight": "balanced",
}


@dataclass
class RFBaselineResult:
    metal: str
    model: str
    test_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    per_cell_line: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "metal": self.metal,
                "model": self.model,
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "test_metrics": self.test_metrics,
                "val_metrics": self.val_metrics,
                "per_cell_line_auc": self.per_cell_line,
            },
            indent=2,
        )


class RFBaseline:
    """RandomForest classifier on Morgan fingerprints."""

    def __init__(
        self,
        metal: str = "Ru",
        morgan_radius: int = MORGAN_RADIUS,
        morgan_nbits: int = MORGAN_NBITS,
        rf_params: Optional[Dict[str, object]] = None,
        seed: int = 42,
        splitter: Optional[callable] = None,
    ) -> None:
        """
        Parameters
        ----------
        splitter : callable, optional
            Same semantics as :class:`MorganXGBBaseline.splitter`.
        """
        self.metal = metal
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.seed = seed
        self.rf_params = {**RF_PARAMS, **(rf_params or {})}
        self.rf_params["random_state"] = seed
        self.splitter = splitter
        self._model = None

    def _split(
        self,
        dataset: MetalCytotoxDataset,
        smiles: np.ndarray,
        y_all: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the configured splitter (default: 80/10/10 stratified random)."""
        if self.splitter is not None:
            result = self.splitter(dataset)
            return result.train_idx, result.val_idx, result.test_idx
        idx = np.arange(len(y_all))
        idx_train, idx_tmp, _, _ = train_test_split(
            idx, y_all, test_size=0.2, stratify=y_all, random_state=self.seed
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_tmp,
            np.asarray(y_all[idx_tmp]).astype(int),
            test_size=0.5,
            stratify=np.asarray(y_all[idx_tmp]).astype(int),
            random_state=self.seed,
        )
        return idx_train, idx_val, idx_test

    def run(self) -> RFBaselineResult:
        # Re-use the XGB pipeline's load + split (identical splits → fair
        # head-to-head comparison).
        flt = CytotoxFilter(
            time_threshold=24.0,
            ic50_min=0.01,
            metal_whitelist=[self.metal],
        )
        ds = MetalCytotoxDataset.from_csv(filters=flt)
        smiles = ds.smiles
        y_all = ds.active.astype(int)
        cell_lines = ds.df["Cell_line"].astype(str).to_numpy()
        keep = np.array([bool(s) for s in smiles], dtype=bool)
        smiles, y_all, cell_lines = smiles[keep], y_all[keep], cell_lines[keep]
        idx_all = np.arange(len(y_all))

        idx_train, idx_val, idx_test = self._split(ds, smiles, y_all)

        x_all = morgan_features(smiles, self.morgan_radius, self.morgan_nbits)
        x_train, y_train = x_all[idx_train], y_all[idx_train]
        x_val, y_val = x_all[idx_val], y_all[idx_val]
        x_test, y_test = x_all[idx_test], y_all[idx_test]
        cl_test = cell_lines[idx_test]

        model = RandomForestClassifier(**self.rf_params)
        model.fit(x_train, y_train)
        self._model = model

        train_score = model.predict_proba(x_train)[:, 1]
        val_score = model.predict_proba(x_val)[:, 1]
        test_score = model.predict_proba(x_test)[:, 1]

        return RFBaselineResult(
            metal=self.metal,
            model="rf",
            test_metrics=compute_metrics(y_test, test_score),
            val_metrics=compute_metrics(y_val, val_score),
            per_cell_line=per_cell_line_auc(cl_test, y_test, test_score),
            n_train=int(len(idx_train)),
            n_val=int(len(idx_val)),
            n_test=int(len(idx_test)),
        )


__all__ = ["RFBaseline", "RFBaselineResult", "RF_PARAMS"]
