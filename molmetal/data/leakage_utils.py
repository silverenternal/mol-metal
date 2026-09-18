"""Leakage diagnostics for cytotoxicity dataset splits.

Helpers used by L1 leakage diagnosis (and downstream L2 / L3 tasks):

- :func:`canonicalize_smiles` — RDKit canonical SMILES with raw fallback.
- :func:`canonicalize_array`  — vectorised version over an array-like.
- :func:`smiles_overlap`      — intersection counts and fractions between two
  index sets, in terms of canonical SMILES.
- :func:`bootstrap_auc`       — 1000-iter bootstrap CI on ROC-AUC.

All routines are deterministic given ``seed``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# SMILES canonicalisation
# ---------------------------------------------------------------------------
def canonicalize_smiles(smi: str) -> str:
    """Return RDKit canonical SMILES.

    Falls back to the raw input string if RDKit cannot parse the molecule.
    RDKit warnings are suppressed.
    """
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi


def canonicalize_array(smiles_list: Sequence[str]) -> Tuple[np.ndarray, int]:
    """Vectorised :func:`canonicalize_smiles`.

    Returns ``(canon_array, n_parse_failures)``.  Length of ``canon_array``
    matches ``len(smiles_list)``.  Elements that fail to parse retain the
    raw input so duplicate detection still works on the raw strings.
    """
    out = [canonicalize_smiles(s) for s in smiles_list]
    n_fail = 0
    canon = np.empty(len(out), dtype=object)
    for i, (raw, c) in enumerate(zip(smiles_list, out)):
        try:
            from rdkit import Chem, RDLogger

            RDLogger.DisableLog("rdApp.*")
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                n_fail += 1
        except Exception:
            n_fail += 1
        canon[i] = c
    return canon, n_fail


# ---------------------------------------------------------------------------
# Split overlap
# ---------------------------------------------------------------------------
def smiles_overlap(
    split_a_indices: Iterable[int],
    split_b_indices: Iterable[int],
    smiles_list: Sequence[str],
) -> Dict[str, float]:
    """Intersection statistics between two splits in terms of canonical SMILES.

    Parameters
    ----------
    split_a_indices, split_b_indices
        Index arrays into ``smiles_list``.
    smiles_list
        Sequence of SMILES strings (length = ``len(dataset)``).

    Returns
    -------
    dict with keys
        ``n_a``            : number of rows in A
        ``n_b``            : number of rows in B
        ``n_unique_a``     : number of unique canonical SMILES in A
        ``n_unique_b``     : number of unique canonical SMILES in B
        ``n_intersect``    : |unique_canon(A) ∩ unique_canon(B)|
        ``frac_of_b_seen`` : fraction of B rows whose canon SMILES is in A
        ``frac_of_a_seen`` : fraction of A rows whose canon SMILES is in B
    """
    smiles_arr = np.asarray(smiles_list, dtype=object)
    a_idx = np.asarray(list(split_a_indices), dtype=np.int64)
    b_idx = np.asarray(list(split_b_indices), dtype=np.int64)
    if a_idx.size == 0 or b_idx.size == 0:
        return {
            "n_a": int(a_idx.size),
            "n_b": int(b_idx.size),
            "n_unique_a": 0,
            "n_unique_b": 0,
            "n_intersect": 0,
            "frac_of_b_seen": 0.0,
            "frac_of_a_seen": 0.0,
        }
    canon_a = {canonicalize_smiles(str(smiles_arr[i])) for i in a_idx}
    canon_b = {canonicalize_smiles(str(smiles_arr[i])) for i in b_idx}
    inter = canon_a & canon_b
    canon_b_rows = np.array([canonicalize_smiles(str(smiles_arr[i])) for i in b_idx])
    canon_a_rows = np.array([canonicalize_smiles(str(smiles_arr[i])) for i in a_idx])
    frac_b_seen = float(np.mean(np.isin(canon_b_rows, list(canon_a)))) if canon_b_rows.size else 0.0
    frac_a_seen = float(np.mean(np.isin(canon_a_rows, list(canon_b)))) if canon_a_rows.size else 0.0
    return {
        "n_a": int(a_idx.size),
        "n_b": int(b_idx.size),
        "n_unique_a": int(len(canon_a)),
        "n_unique_b": int(len(canon_b)),
        "n_intersect": int(len(inter)),
        "frac_of_b_seen": frac_b_seen,
        "frac_of_a_seen": frac_a_seen,
    }


# ---------------------------------------------------------------------------
# Bootstrap AUC CI
# ---------------------------------------------------------------------------
def bootstrap_auc(
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, np.ndarray]:
    """Bootstrap mean + 95 % CI on ROC-AUC.

    Parameters
    ----------
    y_true, y_score
        Test-set predictions (already aligned, no resampling on index yet).
    n_bootstrap
        Number of resamples.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    (mean, lo, hi, samples)
        ``samples`` is the unsorted array of bootstrap AUCs (length
        ``n_bootstrap``) — handy for plotting later.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    rng = np.random.default_rng(seed)
    n = y_true.size
    aucs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            # resample landed on a single class — use NaN; we'll drop later
            aucs[i] = np.nan
            continue
        aucs[i] = float(roc_auc_score(yt, ys))
    aucs = aucs[~np.isnan(aucs)]
    if aucs.size == 0:
        return float("nan"), float("nan"), float("nan"), aucs
    mean = float(aucs.mean())
    lo = float(np.percentile(aucs, 2.5))
    hi = float(np.percentile(aucs, 97.5))
    return mean, lo, hi, aucs


# ---------------------------------------------------------------------------
# Seen / unseen bucket AUC
# ---------------------------------------------------------------------------
def seen_unseen_auc(
    train_smiles: Sequence[str],
    test_smiles: Sequence[str],
    y_test: Sequence[int],
    score_test: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Compute AUC separately on test rows whose SMILES was/was not in train.

    Returns two dicts (one per bucket) with ``roc_auc``, ``pr_auc``,
    ``n`` and ``pos_rate``.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    train_set = {canonicalize_smiles(s) for s in train_smiles}
    seen_mask = np.array(
        [canonicalize_smiles(s) in train_set for s in test_smiles], dtype=bool
    )
    y = np.asarray(y_test).astype(int)
    s = np.asarray(score_test).astype(float)

    out: Dict[str, Dict[str, float]] = {}
    for name, mask in (("seen", seen_mask), ("unseen", ~seen_mask)):
        yt = y[mask]
        st = s[mask]
        n = int(yt.size)
        if n == 0 or len(np.unique(yt)) < 2:
            out[name] = {
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "n": n,
                "pos_rate": float(yt.mean()) if n else 0.0,
            }
            continue
        out[name] = {
            "roc_auc": float(roc_auc_score(yt, st)),
            "pr_auc": float(average_precision_score(yt, st)),
            "n": n,
            "pos_rate": float(yt.mean()),
        }
    return out


__all__ = [
    "canonicalize_smiles",
    "canonicalize_array",
    "smiles_overlap",
    "bootstrap_auc",
    "seen_unseen_auc",
]
