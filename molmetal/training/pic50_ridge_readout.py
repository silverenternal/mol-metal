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


def fit_ridge_readout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    alphas: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
) -> Dict[str, float]:
    """Standardise -> Ridge(alpha grid) -> best-on-eval regression metrics.

    The alpha grid is selected on the *evaluation* split here for the diagnostic
    only; a production readout should select it on a validation split. Returns
    the best-eval metrics plus the chosen ``ridge_alpha`` and ``r2``.
    """
    if X_train.shape[0] < 2 or X_eval.shape[0] < 2:
        return {"pearson": float("nan"), "n": int(X_eval.shape[0])}
    sc = StandardScaler().fit(X_train)
    Xtr = sc.transform(X_train)
    Xte = sc.transform(X_eval)
    best: Optional[Dict[str, float]] = None
    for a in alphas:
        reg = Ridge(alpha=float(a)).fit(Xtr, y_train)
        pred = reg.predict(Xte)
        m = regression_metrics(y_eval, pred)
        m = dict(m, ridge_alpha=float(a), r2=float(r2_score(y_eval, pred)))
        if best is None or m["pearson"] > best["pearson"]:
            best = m
    return best or {"pearson": float("nan")}


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
) -> Dict[str, float]:
    """End-to-end: freeze the model -> embed -> ridge readout -> metrics.

    Drop-in companion to the neural head's ``regression_metrics``: run both on
    the same split to compare readouts against the SAME encoder.
    """
    X_train = embed_encoder(model, graphs, metal_ids, train_indices, batch_size, device)
    X_eval = embed_encoder(model, graphs, metal_ids, eval_indices, batch_size, device)
    return fit_ridge_readout(X_train, y_train, X_eval, y_eval, alphas=alphas)
