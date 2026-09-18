"""Morgan FP + XGBoost baseline (reproduction of Krasnov 2026 main result).

Pipeline (per TODO/02_data/datasets.md and Krasnov 10.1021/acs.jmedchem.5c02755):

1. Load :class:`MetalCytotoxDataset` filtered by ``time_threshold=24``,
   ``ic50_min=0.01`` (drops sub-nanomolar noise).
2. Restrict to a single metal centre (``Ru``, ``Ir``, …).
3. Build the binary label ``active = (IC50 < 10 µM)``.
4. 80/10/10 random split, seed=42.
5. Featurise the ligand SMILES with Morgan fingerprints
   (radius=2, nBits=2048).
6. Train :class:`xgboost.XGBClassifier` with the paper's hyperparameters.
7. Report ROC-AUC, PR-AUC, hit-rate@top-5% on the held-out test split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from molmetal.baselines.eval_utils import (
    compute_metrics,
    metric_summary,
    morgan_features,
    per_cell_line_auc,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import (
    ChemicalSplitter,
    LigandDeduplicatedSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    TemporalSplitter,
)

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Hyperparameters (Krasnov 2026 paper)
# ---------------------------------------------------------------------------
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

XGB_PARAMS: Dict[str, object] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "random_state": 42,
    "tree_method": "hist",
}

RANDOM_SEED = 42


# Map of `--split` CLI flag values to splitter *factories*.
# Each factory returns a fresh splitter (so `seed` is honoured per call).
SPLITTER_FACTORIES: Dict[str, callable] = {
    "random": lambda seed: RandomSplitter(seed=seed),
    "ligand_dedup": lambda seed: LigandDeduplicatedSplitter(
        strategy="largest_first", seed=seed
    ),
    "scaffold": lambda seed: ScaffoldSplitter(
        strategy="largest_first", seed=seed
    ),
    "temporal": lambda seed: TemporalSplitter(cutoff_year=2024),
    "chemical": lambda seed: ChemicalSplitter(threshold=0.7, seed=seed),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class MorganXGBResult:
    metal: str
    model: str
    test_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    per_cell_line: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    pos_rate_train: float = 0.0
    pos_rate_test: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "metal": self.metal,
                "model": self.model,
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "pos_rate_train": self.pos_rate_train,
                "pos_rate_test": self.pos_rate_test,
                "test_metrics": self.test_metrics,
                "val_metrics": self.val_metrics,
                "per_cell_line_auc": self.per_cell_line,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Baseline class
# ---------------------------------------------------------------------------
class MorganXGBBaseline:
    """Morgan-fingerprint + XGBoost classifier for cytotoxicity."""

    def __init__(
        self,
        metal: str = "Ru",
        morgan_radius: int = MORGAN_RADIUS,
        morgan_nbits: int = MORGAN_NBITS,
        xgb_params: Optional[Dict[str, object]] = None,
        seed: int = RANDOM_SEED,
        splitter: Optional[callable] = None,
    ) -> None:
        """
        Parameters
        ----------
        splitter : callable, optional
            A splitter instance (any class with ``__call__(dataset)`` that
            returns a :class:`SplitResult`).  When ``None``, the default
            ``RandomSplitter(seed=self.seed)`` is used.
        """
        self.metal = metal
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.seed = seed
        self.xgb_params = {**XGB_PARAMS, **(xgb_params or {})}
        self.xgb_params["random_state"] = seed
        self.splitter = splitter
        # Hold split artifacts for downstream analysis
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None
        self._xgb_model = None

    # ------------------------------------------------------------------
    # Data loading & splitting
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
        # Default: 80/10/10 stratified random split (preserves the original
        # Krasnov-paper behaviour).
        y = np.asarray(dataset.active).astype(int)
        idx = np.arange(len(y))
        idx_train, idx_tmp, _, _ = train_test_split(
            idx, y, test_size=0.2, stratify=y, random_state=self.seed
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_tmp, np.asarray(y[idx_tmp]).astype(int),
            test_size=0.5, stratify=np.asarray(y[idx_tmp]).astype(int),
            random_state=self.seed,
        )
        return idx_train, idx_val, idx_test

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> MorganXGBResult:
        """Train and evaluate the baseline end-to-end. Returns metrics."""
        # 1. Load + filter
        ds = self._load_dataset()
        smiles = ds.smiles
        y_all = ds.active.astype(int)
        cell_lines = ds.df["Cell_line"].astype(str).to_numpy()
        # Drop rows with empty SMILES (RDKit would map them to zeros and the
        # fingerprint would be uninformative — better to skip them up front)
        keep = np.array([bool(s) for s in smiles], dtype=bool)
        smiles = smiles[keep]
        y_all = y_all[keep]
        cell_lines = cell_lines[keep]
        idx_all = np.arange(len(y_all))
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

        # 4. Class-weight balancing
        pos = max(1, int(y_train.sum()))
        neg = max(1, int(len(y_train) - pos))
        params = dict(self.xgb_params)
        params["scale_pos_weight"] = float(neg) / float(pos)

        # 5. Train
        from xgboost import XGBClassifier  # local import keeps xgboost optional

        model = XGBClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        self._xgb_model = model

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
            model="xgb",
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
    baseline = MorganXGBBaseline(metal=metal)
    baseline._load_dataset = lambda: ds_small  # type: ignore[assignment]
    return baseline.run()


__all__ = [
    "MorganXGBBaseline",
    "MorganXGBResult",
    "MORGAN_RADIUS",
    "MORGAN_NBITS",
    "XGB_PARAMS",
    "SPLITTER_FACTORIES",
    "quick_smoke",
]
