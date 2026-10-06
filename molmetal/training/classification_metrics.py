"""Pure classification-metric helpers — extracted from
`molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (lines 2450-2468
in the pre-split trainer; the trainer re-exports `_safe_classification_metrics`
for backwards compatibility).

Wf-??? monolith-split audit (2026-10-05): this function is a pure leaf:
no module-global reads, no `self.*`, only numpy + sklearn. Safe to
extract; behaviour-preserving move (function body unchanged).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    """ROC-AUC + count; NaN-safe if only one class is present."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {
            "roc_auc": float("nan"),
            "n": int(y_true.size),
            "n_pos": int(y_true.sum()) if y_true.size else 0,
            "n_neg": int((1 - y_true).sum()) if y_true.size else 0,
        }
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "n": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }
