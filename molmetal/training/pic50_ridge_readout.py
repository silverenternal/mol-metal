#!/usr/bin/env python
"""pIC50 ridge readout on frozen D-MPNN encoder embeddings.

Motivation (WF diagnostic `wf_pearson_r_pic50_head_vs_feature.py`, 2026-10-08):
on the Ru HeLa48h/dark cohort the multi-task D-MPNN's *neural* pIC50 regression
head reaches test pearson ≈ 0.08-0.31, while a plain Ridge regressor fit on the
SAME encoder's 2H-dim pooled embedding reaches ≈ 0.38. The encoder therefore
carries more pIC50 signal than the neural head extracts — the bottleneck is the
readout, not the features, and NOT the multi-task alpha (0.5 -> 0.2 moved
pearson < 0.001).

This module exposes a linear (ridge) readout on the frozen encoder embedding so
the signal the encoder already encodes can be recovered without re-tuning the
loss. It is deliberately a *readout swap*, not a loss change, so it composes
with any encoder-training recipe.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from molmetal.baselines.dmpnn_multitask import regression_metrics
from molmetal.training.collate_multimetal import collate_multimetal


def embed_encoder(
    model,
    graphs: List[Optional[Tuple]],
    metal_ids: List[int],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Frozen-encoder embeddings ``(N, 2H)`` for the rows in ``indices``."""
    model.eval()
    chunks: List[np.ndarray] = []
    idx = np.asarray(indices)
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            b = idx[start : start + batch_size]
            bg = [graphs[i] for i in b]
            bm = [metal_ids[i] for i in b]
            data = collate_multimetal(
                bg, bm, np.zeros(len(b), dtype=np.float32), np.zeros(len(b), dtype=np.int64)
            )
            if data is None:
                continue
            atom_f, bond_f, e_src, e_dst, _, _, _ = data
            atom_f = atom_f.to(device)
            bond_f = bond_f.to(device)
            e_src = e_src.to(device)
            e_dst = e_dst.to(device)
            h = model.encode_pool(atom_f, bond_f, e_src, e_dst)
            chunks.append(h.detach().float().cpu().numpy())
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


#: Deterministic seed for the internal train->val carve-out when the caller
#: supplies no validation fold.  Fixed so two runs on the same split pick the
#: same alpha (the readout must be reproducible without a seed argument).
VAL_SPLIT_SEED = 20261008
VAL_FRACTION = 0.2


def fit_ridge_readout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    alphas: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Standardise -> Ridge(alpha grid) -> eval metrics at ONE chosen alpha.

    The alpha is selected on a *validation* fold, never on ``X_eval``.  Passing
    ``X_val``/``y_val`` uses the trainer's real validation split; omitting them
    carves a deterministic ``VAL_FRACTION`` slice off the training rows.  Either
    way ``X_eval`` is touched exactly once, after the alpha is frozen, so the
    reported pearson carries no selection bias.  Returns the eval metrics plus
    the chosen ``ridge_alpha``, ``r2``, and the val-side selection diagnostics.
    """
    if X_train.shape[0] < 2 or X_eval.shape[0] < 2:
        return {"pearson": float("nan"), "n": int(X_eval.shape[0])}
    Xtr_full, ytr_full = np.asarray(X_train), np.asarray(y_train)
    if X_val is None or y_val is None:
        n_val = max(1, int(round(VAL_FRACTION * len(ytr_full))))
        n_val = min(n_val, len(ytr_full) - 1)
        perm = np.random.default_rng(VAL_SPLIT_SEED).permutation(len(ytr_full))
        val_pos, tr_pos = perm[:n_val], perm[n_val:]
        X_val, y_val = Xtr_full[val_pos], ytr_full[val_pos]
        Xtr_full, ytr_full = Xtr_full[tr_pos], ytr_full[tr_pos]
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)
    if X_val.shape[0] < 2 or Xtr_full.shape[0] < 2:
        return {"pearson": float("nan"), "n": int(X_eval.shape[0])}

    sc = StandardScaler().fit(Xtr_full)
    Xtr = sc.transform(Xtr_full)
    Xva = sc.transform(X_val)
    Xte = sc.transform(X_eval)
    # alpha is chosen on the val fold only.
    best_a, best_val_pearson = None, -np.inf
    for a in alphas:
        reg = Ridge(alpha=float(a)).fit(Xtr, ytr_full)
        vp = float(regression_metrics(y_val, reg.predict(Xva))["pearson"])
        if not np.isfinite(vp):
            vp = -np.inf
        if vp > best_val_pearson:
            best_a, best_val_pearson = float(a), vp
    if best_a is None:
        return {"pearson": float("nan"), "n": int(X_eval.shape[0])}
    # Refit on the FULL training rows at the frozen alpha, then score eval once.
    reg = Ridge(alpha=best_a).fit(sc.transform(X_train), y_train)
    pred = reg.predict(Xte)
    m = dict(regression_metrics(y_eval, pred))
    m["ridge_alpha"] = best_a
    m["r2"] = float(r2_score(y_eval, pred))
    m["val_pearson_at_selected_alpha"] = float(best_val_pearson)
    m["n"] = int(X_eval.shape[0])
    m["n_train"] = int(np.asarray(X_train).shape[0])
    m["n_val"] = int(X_val.shape[0])
    return m


def ridge_readout_from_model(
    model,
    graphs: List[Optional[Tuple]],
    metal_ids: List[int],
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    alphas: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    val_indices: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """End-to-end: freeze the model -> embed -> ridge readout -> metrics.

    Drop-in companion to the neural head's ``regression_metrics``: run both on
    the same split to compare readouts against the SAME encoder.  Passing
    ``val_indices``/``y_val`` selects the alpha on that validation fold; omitting
    them makes ``fit_ridge_readout`` carve its own val slice off the train rows.
    """
    X_train = embed_encoder(model, graphs, metal_ids, train_indices, batch_size, device)
    X_eval = embed_encoder(model, graphs, metal_ids, eval_indices, batch_size, device)
    X_val = y_val_arr = None
    if val_indices is not None and y_val is not None:
        X_val = embed_encoder(model, graphs, metal_ids, val_indices, batch_size, device)
        y_val_arr = np.asarray(y_val)
    return fit_ridge_readout(
        X_train, y_train, X_eval, y_eval, alphas=alphas,
        X_val=X_val, y_val=y_val_arr,
    )
