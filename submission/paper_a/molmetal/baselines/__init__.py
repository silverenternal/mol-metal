"""Classical ML baselines for MetalCytoToxDB cytotoxicity classification.

Reproduces the Krasnov 2026 (10.1021/acs.jmedchem.5c02755) baseline: Morgan
fingerprints + gradient-boosted trees / random forest.

Modules
-------
- ``eval_utils``       — shared evaluation (ROC-AUC, PR-AUC, per-cell-line AUC)
- ``morgan_xgb``       — Morgan FP + XGBoost baseline
- ``morgan_lightgbm``  — Morgan FP + LightGBM baseline (drop-in alternative)
- ``rf_baseline``      — Morgan FP + RandomForest baseline
"""

from __future__ import annotations

__all__ = ["eval_utils", "morgan_xgb", "morgan_lightgbm", "rf_baseline"]
