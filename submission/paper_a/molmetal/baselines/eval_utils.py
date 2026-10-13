"""Shared evaluation helpers for classical ML baselines.

Provides:

- :func:`compute_metrics` — ROC-AUC, PR-AUC (Average Precision), and
  hit-rate-at-top-5% from predicted probabilities.
- :func:`per_cell_line_auc` — AUC broken out per cell line.
- :func:`featurize_morgan` — RDKit Morgan fingerprint featurisation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------------
def morgan_features(
    smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Vectorise ``smiles_list`` as Morgan / ECFP4-like fingerprints.

    Returns an ``(N, n_bits)`` ``uint8`` array.  Invalid SMILES are mapped to
    the zero fingerprint (silently — RDKit's parse warnings are suppressed).
    """
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        # Explicit loop is faster than np.frombuffer for n_bits=2048
        for bit in fp.GetOnBits():
            arr[bit] = 1
        out[i] = arr
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Compute ROC-AUC, PR-AUC (Average Precision) and hit-rate@top-5%.

    ``y_true`` must be 0/1, ``y_score`` must be continuous probability-like.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "hit_rate_top5pct": float("nan"),
            "n": int(y_true.size),
        }
    roc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    n = y_true.size
    k = max(1, int(round(0.05 * n)))
    top_k_idx = np.argpartition(-y_score, k - 1)[:k]
    hit = float(np.mean(y_true[top_k_idx]))
    return {
        "roc_auc": roc,
        "pr_auc": ap,
        "hit_rate_top5pct": hit,
        "n": int(n),
    }


def per_cell_line_auc(
    cell_lines: Sequence[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_count: int = 30,
) -> Dict[str, float]:
    """ROC-AUC computed independently for each cell line.

    Cell lines with fewer than ``min_count`` samples are skipped (their AUC
    is too noisy).  Returns ``{cell_line: auc}`` sorted by sample count
    descending.
    """
    y_true = np.asarray(y_true).astype(int).tolist()
    y_score = np.asarray(y_score).astype(float).tolist()
    by_line: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for cl, yi, si in zip(cell_lines, y_true, y_score):
        by_line[str(cl)].append((int(yi), float(si)))
    out: Dict[str, float] = {}
    for cl, items in by_line.items():
        if len(items) < min_count:
            continue
        ys = np.array([t[0] for t in items])
        ss = np.array([t[1] for t in items])
        if len(np.unique(ys)) < 2:
            continue
        out[cl] = float(roc_auc_score(ys, ss))
    # Sort by count desc (most-supported cell lines first)
    counts = {cl: len(items) for cl, items in by_line.items()}
    return dict(sorted(out.items(), key=lambda kv: -counts[kv[0]]))


def metric_summary(metrics_dict: Dict[str, float]) -> str:
    """One-line printable summary of a metrics dict."""
    return (
        f"AUC={metrics_dict['roc_auc']:.4f} | "
        f"AP={metrics_dict['pr_auc']:.4f} | "
        f"Hit@5%={metrics_dict['hit_rate_top5pct']:.3f} | "
        f"n={metrics_dict['n']}"
    )
