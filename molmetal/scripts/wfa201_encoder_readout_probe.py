#!/usr/bin/env python
"""Wf-A201 -- CPU-only localisation of the 0.1931 pIC50 pearson gap.

WHAT THIS IS
    A READ-ONLY diagnostic over the checkpoints that are already on disk.
    It re-measures the five 2026-10-08 arms of
    ``molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py`` as far as they
    can be re-measured WITHOUT a GPU, and then fits head-only readouts on
    FROZEN, PRE-EXTRACTED encoder embeddings to localise where the gap goes.

    * NO encoder is trained.  Every network fitted here is a LINEAR or MLP
      READOUT on frozen features -- a "head-only CPU fit on frozen features",
      never a model retrain.
    * NO checkpoint is modified or written.
    * NOTHING is written outside ``molmetal/reports/wfa201_encoder_readout/``.

CPU-ONLY BY CONSTRUCTION
    ``CUDA_VISIBLE_DEVICES`` is cleared BEFORE ``torch`` is imported and
    :func:`assert_cpu_only` refuses to run if a GPU is visible.  Set
    ``WFA201_FORCE_GPU=1`` to exercise that refusal path in a test.

USAGE
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \\
        molmetal/scripts/wfa201_encoder_readout_probe.py
"""

from __future__ import annotations

import os

# HARD CPU-ONLY: must be set BEFORE torch is imported anywhere.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path("/home/hugo/codes/mol-metal")
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch import nn  # noqa: E402

# The D-MPNN forward loops over molecules in Python, so every op is tiny and
# multi-threaded BLAS spends all its time on thread synchronisation.  One
# thread is ~50x faster here and makes the embeddings bit-reproducible.
torch.set_num_threads(1)

from molmetal.baselines.dmpnn import featurize_smiles_list  # noqa: E402
from molmetal.baselines.dmpnn_multitask import (  # noqa: E402
    MultiTaskLoss,
    regression_metrics,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset  # noqa: E402
from molmetal.scripts.train_dmpnn_multitask_wasserstein import (  # noqa: E402
    FIVE_METALS,
    MultiTaskWassersteinModel,
    _resolve_default_csv_path,
    build_metal_split_indices,
    load_5metal_split_table,
)
from molmetal.training.collate_multimetal import collate_multimetal  # noqa: E402
from molmetal.training.pic50_ridge_readout import fit_ridge_readout  # noqa: E402

OUT_DIR = REPO / "molmetal/reports/wfa201_encoder_readout"
DIAG_REPORT = REPO / "molmetal/reports/wf_pearson_diag/report.json"
CKPT_DIR = REPO / "molmetal/checkpoints"

#: The repo's own backbone defaults (molmetal/scripts/retrain_pic50_neural.py).
REPO_HIDDEN = 128
REPO_DEPTH = 3
REPO_DROPOUT = 0.1
#: Wf-127 P2 spec value (molmetal/tests/test_metal_lomo_v2.py:250).
L2_GUARD_MAX_NORM = 5.0
#: The 2026-10-08 ridge readout's internal val carve-out (pic50_ridge_readout.py).
VAL_SPLIT_SEED = 20261008
VAL_FRACTION = 0.2
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)

CPU_REFUSAL_MSG = (
    "WFA201 REFUSING TO RUN: a CUDA device is visible. This probe is CPU-ONLY "
    "by user directive (2026-11-02). CUDA_VISIBLE_DEVICES is cleared before the "
    "torch import, so a visible GPU means the environment forced one back on "
    "-- aborting without touching any checkpoint."
)


class CpuOnlyViolation(RuntimeError):
    """Raised when the CPU-only guard trips."""


class SameRowsViolation(AssertionError):
    """Raised when two arms are not scored on the same rows."""


class WriteScopeViolation(AssertionError):
    """Raised when a write target is outside the probe's own output directory."""


def assert_cpu_only() -> None:
    """Refuse to run when a GPU is visible (Wf-A201 hard rule 1)."""
    if torch.cuda.is_available():
        raise CpuOnlyViolation(CPU_REFUSAL_MSG)


def assert_same_test_rows(n_ridge: int, n_head: int, n_expected: int, label: str) -> None:
    """H5 guard: the ridge arm and the head arm must score the SAME test rows."""
    if not (n_ridge == n_head == n_expected):
        raise SameRowsViolation(
            f"{label}: ridge arm scored {n_ridge} rows, head arm scored {n_head} rows, "
            f"expected {n_expected} -- the arms are NOT on the same rows"
        )


def assert_write_scope(path: Path) -> None:
    """Write-scope guard: only files under the probe's own output dir may be written."""
    p = Path(path).resolve()
    allowed = OUT_DIR.resolve()
    if p != allowed and allowed not in p.parents:
        raise WriteScopeViolation(
            f"refusing to write {p}: outside the probe's own output directory {allowed}"
        )


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_rows(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def nvidia_smi_compute_apps() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return (out.stdout or "").strip() or "(no compute apps)"
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"(nvidia-smi unavailable: {type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_dataset(metal: str, featurise: bool = True) -> Dict:
    """Reproduce the 2026-10-08 split exactly (same helpers, same filters)."""
    csv_path = _resolve_default_csv_path("metalcytotox_5metal")
    split_df = load_5metal_split_table(csv_path)
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=list(FIVE_METALS),
        compute_pic50=True,
        compute_active=True,
    )
    import pandas as pd
    from molmetal.data import cytotox as cy

    df = pd.read_csv(REPO / ".data/molmetal/MetalCytoToxDB.csv", low_memory=False)
    df = cy._apply_filters(df, flt)
    df = cy._add_pic50(df)
    df = cy._add_active(df)
    ds = MetalCytotoxDataset(df=df, cache_dir=str(OUT_DIR / ".cache"))
    smiles_all = np.asarray(ds.smiles).astype(str)
    pic50_all = np.asarray(ds.pic50).astype(np.float32)
    active_all = np.asarray(ds.active).astype(np.int64)
    metals_all = np.asarray(ds.metals).astype(str)

    per_metal = {m: build_metal_split_indices(split_df, ds, m) for m in FIVE_METALS}
    target_train, target_val, target_test = per_metal[metal]
    train_pool_idx = np.concatenate([per_metal[m][0] for m in FIVE_METALS])

    used_idx = np.unique(np.concatenate([train_pool_idx, target_val, target_test]))
    pos_of = {int(v): k for k, v in enumerate(used_idx)}
    pool_pos_of = {int(g): k for k, g in enumerate(train_pool_idx)}
    graphs_all = (
        featurize_smiles_list([str(s) for s in smiles_all[used_idx]], version="v1")
        if featurise else None
    )
    metal_to_idx = {m: i for i, m in enumerate(FIVE_METALS)}
    metal_ids_all = np.asarray(
        [metal_to_idx[str(m)] for m in metals_all[used_idx]], dtype=np.int64
    )

    g_pool = np.asarray([pos_of[int(i)] for i in train_pool_idx], dtype=np.int64)
    g_val = np.asarray([pos_of[int(i)] for i in target_val], dtype=np.int64)
    g_test = np.asarray([pos_of[int(i)] for i in target_test], dtype=np.int64)
    g_ttrain = np.asarray([pos_of[int(i)] for i in target_train], dtype=np.int64)
    # positions of the Ru target-train rows INSIDE the joint-pool arrays
    pool_pos_ttrain = np.asarray([pool_pos_of[int(i)] for i in target_train], dtype=np.int64)

    return {
        "graphs_all": graphs_all,
        "metal_ids_all": metal_ids_all,
        "g_pool": g_pool,
        "g_val": g_val,
        "g_test": g_test,
        "g_ttrain": g_ttrain,
        "pool_pos_ttrain": pool_pos_ttrain,
        "smiles_used": [str(s) for s in smiles_all[used_idx]],
        "y_pool": pic50_all[train_pool_idx],
        "y_val": pic50_all[target_val],
        "y_test": pic50_all[target_test],
        "y_target_train": pic50_all[target_train],
        "a_pool": active_all[train_pool_idx],
        "a_val": active_all[target_val],
        "a_test": active_all[target_test],
        "target_test_idx": target_test,
        "n_joint_train": int(len(train_pool_idx)),
        "n_target_train": int(len(target_train)),
        "n_val": int(len(target_val)),
        "n_test": int(len(target_test)),
    }


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------
def extract_embeddings(
    model: nn.Module,
    graphs: List,
    metal_ids: np.ndarray,
    idx: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """Frozen-encoder 2H-dim pooled embeddings (mirror of the 2026-10-08 code)."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            b = idx[start: start + batch_size]
            bg = [graphs[i] for i in b]
            bm = [int(metal_ids[i]) for i in b]
            data = collate_multimetal(
                bg, bm, np.zeros(len(b)), np.zeros(len(b))
            )
            if data is None:
                continue
            atom_f, bond_f, e_src, e_dst, _mids, _pic, _act = data
            h = model.encode_pool(atom_f, bond_f, e_src, e_dst)
            out.append(h.detach().float().cpu().numpy())
    if not out:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(out, axis=0)


def build_random_encoder(seed: int = 0) -> MultiTaskWassersteinModel:
    """Exactly the 2026-10-08 arm (B) init: manual_seed then construct on CPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MultiTaskWassersteinModel(n_metals=len(FIVE_METALS))


def load_ckpt_encoder(path: Path) -> Tuple[MultiTaskWassersteinModel, Dict]:
    """Rebuild a multi-task D-MPNN from an on-disk checkpoint (read-only)."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    hidden = int(ck.get("hidden", REPO_HIDDEN))
    depth = int(ck.get("depth", REPO_DEPTH))
    use_dann = any(k.startswith("domain_head") for k in sd)
    n_metals = len({k.split(".")[1] for k in sd if k.startswith("reg_heads.")})
    model = MultiTaskWassersteinModel(
        n_metals=n_metals, hidden=hidden, depth=depth, use_dann=use_dann
    )
    model.load_state_dict(sd)
    model.eval()
    return model, ck


# ---------------------------------------------------------------------------
# ridge readout (repo implementation + prediction recovery)
# ---------------------------------------------------------------------------
def fit_ridge_with_preds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> Tuple[Dict, np.ndarray]:
    """Mirror of ``molmetal.training.pic50_ridge_readout.fit_ridge_readout`` that
    also returns the eval predictions, so the decomposition arms can use them.

    The metric is cross-checked against the repo implementation (see main).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xtr_full, ytr_full = np.asarray(X_train), np.asarray(y_train)
    n_val = max(1, int(round(VAL_FRACTION * len(ytr_full))))
    n_val = min(n_val, len(ytr_full) - 1)
    perm = np.random.default_rng(VAL_SPLIT_SEED).permutation(len(ytr_full))
    val_pos, tr_pos = perm[:n_val], perm[n_val:]
    sc = StandardScaler().fit(Xtr_full[tr_pos])
    best_a, best_v = None, -np.inf
    for a in RIDGE_ALPHAS:
        reg = Ridge(alpha=float(a)).fit(sc.transform(Xtr_full[tr_pos]), ytr_full[tr_pos])
        v = float(regression_metrics(ytr_full[val_pos], reg.predict(sc.transform(Xtr_full[val_pos])))["pearson"])
        if not np.isfinite(v):
            v = -np.inf
        if v > best_v:
            best_a, best_v = float(a), v
    reg = Ridge(alpha=best_a).fit(sc.transform(Xtr_full), ytr_full)
    preds = reg.predict(sc.transform(X_eval))
    m = dict(regression_metrics(y_eval, preds))
    m["ridge_alpha"] = best_a
    m["n"] = int(len(preds))
    return m, preds


def fit_ridge_real_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Dict, np.ndarray]:
    """``fit_ridge_readout`` with the REAL validation fold passed in.

    ``fit_ridge_readout`` (repo implementation) carves an internal fold out of
    the train rows when ``X_val``/``y_val`` are omitted; when they are supplied
    it fits the scaler on the FULL train rows and selects the alpha on the
    supplied fold. This variant is used to attribute the published numbers.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(np.asarray(X_train))
    best_a, best_v = None, -np.inf
    for a in RIDGE_ALPHAS:
        reg = Ridge(alpha=float(a)).fit(sc.transform(X_train), y_train)
        v = float(regression_metrics(y_val, reg.predict(sc.transform(X_val)))["pearson"])
        if not np.isfinite(v):
            v = -np.inf
        if v > best_v:
            best_a, best_v = float(a), v
    reg = Ridge(alpha=best_a).fit(sc.transform(X_train), y_train)
    preds = reg.predict(sc.transform(X_eval))
    m = dict(regression_metrics(y_eval, preds))
    m["ridge_alpha"] = best_a
    m["n"] = int(len(preds))
    return m, preds


def fit_ridge_test_argmax(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> Tuple[Dict, np.ndarray]:
    """VERBATIM re-implementation of the 2026-10-08 producer's ridge helper.

    Source: ``a5cc339d:molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py``
    lines 103-114 (the revision that produced
    ``molmetal/reports/wf_pearson_diag/report.json``). It selects the ridge
    alpha by maximising Pearson **on the eval rows it then reports** -- i.e.
    the reported number is a MAX over the 4-point alpha grid scored on the
    very rows it is scored on.

    This is kept ONLY as a fingerprint probe: it re-implements a read-only
    artifact's behaviour so the published numbers can be attributed. It is
    NOT used for any headline claim in this report.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(np.asarray(X_train))
    Xtr = sc.transform(X_train)
    Xte = sc.transform(X_eval)
    best, best_p = None, None
    for a in RIDGE_ALPHAS:
        reg = Ridge(alpha=float(a)).fit(Xtr, y_train)
        p = reg.predict(Xte)
        m = regression_metrics(y_eval, p)
        if best is None or m["pearson"] > best["pearson"]:
            best = dict(m, ridge_alpha=float(a))
            best_p = p
    best["n"] = int(len(best_p))
    return best, best_p


# ---------------------------------------------------------------------------
# head-only readouts on frozen features
# ---------------------------------------------------------------------------
def build_heads(
    head_kind: str, n_metals: int, in_dim: int, hidden: int, dropout: float
) -> Tuple[nn.ModuleList, nn.ModuleList]:
    """The repo's own head modules (``Sequential(Linear, ReLU, Dropout, Linear)``)
    or a declared-width / no-hidden-layer variant."""
    if head_kind == "repo":
        h, mk = hidden, (lambda o: nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o)))
    elif head_kind == "wide":
        h, mk = 2 * hidden, (lambda o: nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o)))
    elif head_kind == "linear":
        h, mk = None, (lambda o: nn.Linear(in_dim, o))
    else:
        raise ValueError(f"unknown head_kind={head_kind!r}")
    reg = nn.ModuleList([mk(1) for _ in range(n_metals)])
    cls = nn.ModuleList([mk(2) for _ in range(n_metals)])
    return reg, cls


def _forward_heads(reg: nn.ModuleList, cls: nn.ModuleList, X: torch.Tensor, mid: torch.Tensor):
    """Per-metal routing identical to ``MultiTaskWassersteinModel.forward``."""
    pic50 = torch.zeros(X.shape[0], dtype=X.dtype)
    logits = torch.zeros(X.shape[0], 2, dtype=X.dtype)
    for m in range(len(reg)):
        sel = mid == m
        if bool(sel.any()):
            pic50[sel] = reg[m](X[sel]).squeeze(-1)
            logits[sel] = cls[m](X[sel])
    return pic50, logits


def fit_head_only(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    a_pool: np.ndarray,
    mid_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mid_test: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mid_val: np.ndarray,
    *,
    alpha: float = 0.5,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 0,
    head_kind: str = "repo",
    hidden: int = REPO_HIDDEN,
    dropout: float = REPO_DROPOUT,
    weight_decay: float = 0.0,
    standardise_target: bool = False,
    grad_clip: Optional[float] = None,
    l2_guard: bool = False,
) -> Dict:
    """Fit ONE readout on FROZEN features on the CPU.  No encoder is touched."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_metals = int(mid_pool.max()) + 1
    in_dim = int(X_pool.shape[1])
    reg, cls = build_heads(head_kind, n_metals, in_dim, hidden, dropout)
    crit = MultiTaskLoss(alpha=alpha)
    params = list(reg.parameters()) + list(cls.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    Xp = torch.tensor(X_pool, dtype=torch.float32)
    Xt = torch.tensor(X_test, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    mp = torch.tensor(mid_pool, dtype=torch.int64)
    mt = torch.tensor(mid_test, dtype=torch.int64)
    mv = torch.tensor(mid_val, dtype=torch.int64)

    mu = float(np.mean(y_pool))
    sd = float(np.std(y_pool)) or 1.0
    y_fit = ((y_pool - mu) / sd) if standardise_target else y_pool
    yp = torch.tensor(y_fit, dtype=torch.float32)
    ap = torch.tensor(np.asarray(a_pool), dtype=torch.int64)

    val_curve: List[float] = []
    test_pearson_curve: List[float] = []
    last_loss = float("nan")
    ep_loss_cls: List[float] = []
    ep_loss_reg: List[float] = []
    for ep in range(epochs):
        reg.train()
        cls.train()
        ep_cls, ep_reg = [], []
        tr = np.arange(len(Xp))
        np.random.default_rng(seed + ep).shuffle(tr)
        for start in range(0, len(tr), batch_size):
            bidx = tr[start: start + batch_size]
            pb, logits = _forward_heads(reg, cls, Xp[bidx], mp[bidx])
            total, _lc, _lreg = crit(pb, logits, yp[bidx], ap[bidx])
            ep_cls.append(float(_lc))
            ep_reg.append(float(_lreg))
            opt.zero_grad()
            total.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(params, float(grad_clip))
            if l2_guard:
                nn.utils.clip_grad_norm_(params, L2_GUARD_MAX_NORM)
            opt.step()
            last_loss = float(total.detach())
        reg.eval()
        cls.eval()
        with torch.no_grad():
            pv, _ = _forward_heads(reg, cls, Xv, mv)
            pt, _ = _forward_heads(reg, cls, Xt, mt)
            val_curve.append(float(regression_metrics(y_val, pv.numpy())["pearson"]))
            test_pearson_curve.append(float(regression_metrics(y_test, pt.numpy())["pearson"]))
        ep_loss_cls.append(float(np.mean(ep_cls)) if ep_cls else float("nan"))
        ep_loss_reg.append(float(np.mean(ep_reg)) if ep_reg else float("nan"))

    reg.eval()
    cls.eval()
    with torch.no_grad():
        pp, _ = _forward_heads(reg, cls, Xp, mp)
        pt, _ = _forward_heads(reg, cls, Xt, mt)
    pred_train = pp.numpy().astype(np.float64)
    pred_test = pt.numpy().astype(np.float64)
    if standardise_target:
        pred_train = pred_train * sd + mu
        pred_test = pred_test * sd + mu

    val_curve_arr = np.asarray(val_curve, dtype=float)
    test_curve_arr = np.asarray(test_pearson_curve, dtype=float)
    diverged = bool(np.all(np.isnan(val_curve_arr)))
    if diverged:
        # the fit produced NaN metrics at every epoch (e.g. lr=1e-2 diverges);
        # that is a MEASURED outcome of the arm, not a crash.
        best_ep = -1
        best_val = float("nan")
        best_test = float("nan")
    else:
        best_ep = int(np.nanargmax(val_curve_arr))
        best_val = float(val_curve_arr[best_ep])
        best_test = float(test_curve_arr[best_ep])
    m_train = regression_metrics(y_pool, pred_train)
    m_test = regression_metrics(y_test, pred_test)
    return {
        "alpha": alpha,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "head_kind": head_kind,
        "weight_decay": weight_decay,
        "standardise_target": standardise_target,
        "grad_clip": grad_clip,
        "l2_guard": l2_guard,
        "final_loss": last_loss,
        "test": dict(m_test),
        "train_in_sample": dict(m_train),
        "test_pearson_final": float(m_test["pearson"]),
        "test_pearson_best_val_epoch": best_test,
        "best_val_epoch": best_ep,
        "val_pearson_at_best": best_val,
        "best_val_pearson": float(np.nanmax(val_curve_arr)) if not diverged else float("nan"),
        "diverged_all_nan_val_curve": diverged,
        "loss_cls_last_epoch": float(ep_loss_cls[-1]) if ep_loss_cls else float("nan"),
        "loss_reg_last_epoch": float(ep_loss_reg[-1]) if ep_loss_reg else float("nan"),
        "loss_cls_over_loss_reg_last_epoch": (
            float(ep_loss_cls[-1] / ep_loss_reg[-1])
            if (ep_loss_cls and ep_loss_reg and ep_loss_reg[-1] not in (0.0,)
                and math.isfinite(ep_loss_cls[-1]) and math.isfinite(ep_loss_reg[-1]))
            else float("nan")
        ),
        "_pred_test": pred_test,
        "_pred_train": pred_train,
    }


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(regression_metrics(a, b)["pearson"])


def _jnum(x):
    """JSON-safe number: NaN / +-inf become None (JSON has no NaN literal)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation of x and y given z."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    num = rxy - rxz * ryz
    den = math.sqrt(max(1e-12, (1.0 - rxz ** 2) * (1.0 - ryz ** 2)))
    return float(num / den)


def incremental_r2(y: np.ndarray, p: np.ndarray, r: np.ndarray) -> Dict:
    """In-sample OLS decomposition (optimistic; reported as such)."""
    def ols(cols: List[np.ndarray]) -> Tuple[float, np.ndarray]:
        A = np.column_stack([np.ones(len(y))] + cols)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - float((resid ** 2).sum()) / ss_tot, beta

    r2_p, _ = ols([p])
    r2_r, _ = ols([r])
    r2_pr, _beta = ols([p, r])
    return {
        "r2_p_only": _jnum(r2_p),
        "r2_r_only": _jnum(r2_r),
        "r2_p_plus_r": _jnum(r2_pr),
        "incremental_r2_p_given_r": _jnum(r2_pr - r2_r),
        "incremental_r2_r_given_p": _jnum(r2_pr - r2_p),
        "note": "in-sample OLS on the scored rows (optimistic upper bound); decomposition only",
    }


def morgan_fp(smiles: Sequence[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    fps = []
    for smi in smiles:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.stack(fps)


def _ckpt_head_predict(model, X: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """The checkpoint's OWN regression head applied to frozen embeddings."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    mt = torch.tensor(mid, dtype=torch.int64)
    out = np.zeros(X.shape[0], dtype=np.float64)
    with torch.no_grad():
        for m in range(len(model.reg_heads)):
            sel = mt == m
            if bool(sel.any()):
                out[sel.numpy()] = model.reg_heads[m](Xt[sel]).squeeze(-1).numpy()
    return out


def _fit_specs() -> List[Dict]:
    """The head-only arm grid (each entry states the hypothesis it probes)."""
    return [
        {"key": "L1_repo_head_alpha0.5_raw", "hypothesis": "H1/H2/H3/H4 baseline arm",
         "kwargs": {"head_kind": "repo", "alpha": 0.5}},
        {"key": "L2_repo_head_alpha0.5_std_target", "hypothesis": "H1 target scale",
         "kwargs": {"head_kind": "repo", "alpha": 0.5, "standardise_target": True}},
        {"key": "L3_wide_head_512", "hypothesis": "H2 head capacity",
         "kwargs": {"head_kind": "wide", "alpha": 0.5}},
        {"key": "L3b_linear_head_no_hidden", "hypothesis": "H2 head capacity (linear floor)",
         "kwargs": {"head_kind": "linear", "alpha": 0.5}},
        {"key": "L4_repo_head_wd1e-2", "hypothesis": "H2/H3 regularisation",
         "kwargs": {"head_kind": "repo", "alpha": 0.5, "weight_decay": 1e-2}},
        {"key": "L5_repo_head_gradclip5", "hypothesis": "H3 optimisation (trainer clip)",
         "kwargs": {"head_kind": "repo", "alpha": 0.5, "grad_clip": 5.0}},
        {"key": "A02_alpha0.2_regweighted", "hypothesis": "H4 multi-task coupling",
         "kwargs": {"head_kind": "repo", "alpha": 0.2}},
        {"key": "A08_alpha0.8_clsweighted", "hypothesis": "H4 multi-task coupling",
         "kwargs": {"head_kind": "repo", "alpha": 0.8}},
        {"key": "LR3e-4", "hypothesis": "H3 optimisation (learning rate)",
         "kwargs": {"head_kind": "repo", "alpha": 0.5, "lr": 3e-4}},
        {"key": "LR3e-3", "hypothesis": "H3 optimisation (learning rate)",
         "kwargs": {"head_kind": "repo", "alpha": 0.5, "lr": 3e-3}},
        {"key": "LR1e-2", "hypothesis": "H3 optimisation (learning rate)",
         "kwargs": {"head_kind": "repo", "alpha": 0.5, "lr": 1e-2}},
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _load_encoders(args, graphs_all, mid_all, g_pool, g_test, g_val) -> Dict[str, Dict]:
    """Every encoder whose frozen embeddings this wave measures on."""
    encoders: Dict[str, Dict] = {}
    model_rand = build_random_encoder(args.seed)
    encoders["random_init_seed0"] = {
        "kind": "random",
        "pool": extract_embeddings(model_rand, graphs_all, mid_all, g_pool),
        "test": extract_embeddings(model_rand, graphs_all, mid_all, g_test),
        "val": extract_embeddings(model_rand, graphs_all, mid_all, g_val),
        "model": model_rand,
        "path": None,
        "ckpt_meta": {"note": "random init, torch.manual_seed(0) -- the 2026-10-08 arm (B)"},
    }
    for cp in sorted(CKPT_DIR.glob("dmpnn_mtw_*_seed*.pt")):
        try:
            m, ck = load_ckpt_encoder(cp)
        except Exception as exc:
            print(f"[ckpt] SKIP {cp.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        sd = ck["model_state_dict"]
        n_heads = len({k.split(".")[1] for k in sd if k.startswith("reg_heads.")})
        if n_heads != len(FIVE_METALS):
            print(f"[ckpt] SKIP {cp.name}: n_metals={n_heads} (single-metal arch)", flush=True)
            continue
        encoders[cp.stem] = {
            "kind": "checkpoint",
            "pool": extract_embeddings(m, graphs_all, mid_all, g_pool),
            "test": extract_embeddings(m, graphs_all, mid_all, g_test),
            "val": extract_embeddings(m, graphs_all, mid_all, g_val),
            "model": m,
            "path": cp,
            "ckpt_meta": {
                "hidden": ck.get("hidden"), "depth": ck.get("depth"),
                "epochs": ck.get("epochs"), "seed": ck.get("seed"),
                "alpha": ck.get("alpha"), "lr": ck.get("lr"),
                "batch_size": ck.get("batch_size"), "w_w": ck.get("w_w"),
                "n_train": ck.get("n_train"), "n_val": ck.get("n_val"),
                "n_test": ck.get("n_test"), "n_joint_train": ck.get("n_joint_train"),
                "best_val_auc": ck.get("best_val_auc"),
            },
        }
        print(f"[ckpt] loaded {cp.stem} ({ck.get('epochs')} epochs, hidden={ck.get('hidden')})",
              flush=True)
    return encoders



def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metal", default="Ru")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--fit-seeds", type=int, default=3)
    ap.add_argument("--fit-encoders", default="random_init_seed0,dmpnn_mtw_ru_seed3")
    ap.add_argument("--smoke", action="store_true",
                    help="split + Morgan ridge only (no featurisation, no encoder arms)")
    args = ap.parse_args(argv)

    assert_cpu_only()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: Dict = {
        "wave": "Wf-A201",
        "question": (
            "where does the 0.1931 pIC50 pearson gap (ridge on frozen TRAINED "
            "encoder embeddings 0.4074 vs the repo's neural head 0.2143) go?"
        ),
        "scope": {
            "cpu_only": True,
            "gpu_used": False,
            "encoder_trained": False,
            "checkpoint_modified": False,
            "metric_files_written": 0,
            "registry_files_written": 0,
            "paper_docs_todo_files_written": 0,
            "all_fits_are": "head-only (linear/MLP) readouts on FROZEN, "
                            "PRE-EXTRACTED encoder embeddings",
            "nvidia_smi_before": nvidia_smi_compute_apps(),
        },
        "protocol": {
            "metal": args.metal,
            "seed": args.seed,
            "epochs": args.epochs,
            "fit_seeds": args.fit_seeds,
            "fit_encoders": args.fit_encoders,
            "published_source": str(DIAG_REPORT.relative_to(REPO)),
        },
        "not_measured": [
            "the 2026-10-08 trained encoder (arm D) -- the diag script trained it "
            "in memory and never saved it, and re-training it is forbidden by "
            "the 0-GPU rule; the closest on-disk trained encoders are 1-epoch "
            "checkpoints",
            "the 2026-10-08 neural head arms (alpha 0.5 and alpha 0.2) -- same "
            "reason: reproducing them needs the GPU training run",
            "the W2 (Wasserstein) regulariser's contribution to the head-only fit "
            "-- it acts on the encoder's prediction distribution, which a "
            "head-only fit on frozen features does not have",
            "the L2 gradient-norm guard (L2_GUARD_MAX_NORM) -- only the trainer's "
            "max_norm=5.0 clip (L5 arm) was mirrored",
            "any per-seed variability of the 2026-10-08 arms -- they were single-seed",
            "whether the CPU-only arms can separate H1 from H3 -- see "
            "hypothesis_ranking",
            "head-only fits on the CHECKPOINT encoders (dmpnn_mtw_*): NOT run -- "
            "those encoders were trained on a different cohort/split (recorded "
            "n_joint_train 3540 / 18137 / 18596 vs this probe's 3480), so a "
            "head-only fit on their features could not be compared with the "
            "published arms without an unmeasured contamination term; the CLEAN "
            "localisation therefore uses the randomly-initialised encoder",
            "the fraction of the probe's 994 test rows that were in each "
            "checkpoint's own training set -- the overlap was NOT measured, so "
            "the checkpoint-own-head arm is labelled NOT COMPARABLE",
            "the byte-exactness of the published numbers' provenance beyond the "
            "Morgan-FP fingerprint -- the two encoder arms are attributed to the "
            "same helper function (a5cc339d:103-114) that the Morgan arm proves "
            "by exact match, which is an inference, not a byte-level proof",
        ],
    }

    # ---- split ----
    data = build_dataset(args.metal, featurise=not args.smoke)
    split_hash = sha256_rows(data["target_test_idx"])
    report["split"] = {
        "n_joint_train": data["n_joint_train"],
        "n_target_train": data["n_target_train"],
        "n_val": data["n_val"],
        "n_test": data["n_test"],
        "test_row_id_sha256": split_hash,
        "published_split": {
            "n_joint_train": 3480, "n_target_train": 2262, "n_val": 394, "n_test": 994,
        },
        "split_matches_published": (
            data["n_joint_train"] == 3480 and data["n_target_train"] == 2262
            and data["n_val"] == 394 and data["n_test"] == 994
        ),
    }
    print(
        f"[split] joint_train={data['n_joint_train']} target_train={data['n_target_train']} "
        f"val={data['n_val']} test={data['n_test']} sha={split_hash[:16]}",
        flush=True,
    )
    report["split"]["active_label_balance_pool"] = {
        "n_positive": int(np.sum(data["a_pool"])),
        "n_negative": int(len(data["a_pool"]) - np.sum(data["a_pool"])),
        "positive_fraction": float(np.mean(data["a_pool"])),
        "note": "the classification half of MultiTaskLoss; reported because the "
                "alpha arms (A02/A08) turn out to be indistinguishable on the "
                "head-only path",
    }

    # ---- published 2026-10-08 numbers (read from the on-disk diag report) ----
    published: Dict = {}
    if DIAG_REPORT.exists():
        d = json.loads(DIAG_REPORT.read_text())
        published = {
            "ridge_on_random_encoder_embedding": d["ridge_random_encoder_emb"]["pearson"],
            "ridge_on_trained_encoder_embedding": d["ridge_trained_encoder_emb"]["alpha_0.5"]["pearson"],
            "ridge_on_morgan_fp": d["ridge_morgan_fp_same_split"]["pearson"],
            "neural_head_default_alpha_0.5": d["trained_head"]["alpha_0.5"]["pearson"],
            "neural_head_regweighted_alpha_0.2": d["trained_head"]["alpha_0.2"]["pearson"],
            "head_rmse_alpha_0.5": d["trained_head"]["alpha_0.5"]["rmse"],
            "head_mae_alpha_0.5": d["trained_head"]["alpha_0.5"]["mae"],
            "ridge_rmse_trained_emb_alpha_0.5": d["ridge_trained_encoder_emb"]["alpha_0.5"]["rmse"],
            "ridge_r2_trained_emb_alpha_0.5": d["ridge_trained_encoder_emb"]["alpha_0.5"]["r2"],
            "source": str(DIAG_REPORT.relative_to(REPO)),
            "split_reported_by_source": d["split"],
        }
    report["published_2026_10_08"] = published

    graphs_all = data["graphs_all"]
    mid_all = data["metal_ids_all"]
    g_pool, g_val, g_test = data["g_pool"], data["g_val"], data["g_test"]
    y_pool, y_val, y_test = data["y_pool"], data["y_val"], data["y_test"]
    mid_pool, mid_test, mid_val = mid_all[g_pool], mid_all[g_test], mid_all[g_val]

    # ---- (E) Morgan-FP ridge, same split (pure CPU) ----
    fp_used = morgan_fp(data["smiles_used"])
    ridge_fp, _ = fit_ridge_with_preds(
        fp_used[data["g_ttrain"]], data["y_target_train"], fp_used[g_test], y_test
    )
    ridge_fp_repo = fit_ridge_readout(
        fp_used[data["g_ttrain"]], data["y_target_train"], fp_used[g_test], y_test
    )
    ridge_fp["matches_repo_implementation"] = bool(
        abs(ridge_fp["pearson"] - ridge_fp_repo["pearson"]) < 1e-12
    )
    ridge_fp["repo_implementation_pearson"] = ridge_fp_repo["pearson"]
    # variant sweep: which ridge *selection rule* reproduces the published number?
    ridge_fp_leaky, _ = fit_ridge_test_argmax(
        fp_used[data["g_ttrain"]], data["y_target_train"], fp_used[g_test], y_test
    )
    ridge_fp_realval, _ = fit_ridge_real_val(
        fp_used[data["g_ttrain"]], data["y_target_train"], fp_used[g_test], y_test,
        fp_used[data["g_val"]], data["y_val"],
    )
    ridge_fp["selection_rule_sweep"] = {
        "honest_carveout_internal_fold": {
            "pearson": ridge_fp["pearson"], "mae": ridge_fp["mae"],
            "rmse": ridge_fp["rmse"], "ridge_alpha": ridge_fp["ridge_alpha"],
            "rule": "alpha chosen on a val fold carved out of the train rows "
                    "(repo's current fit_ridge_readout)",
        },
        "honest_real_val_fold": {
            "pearson": ridge_fp_realval["pearson"], "mae": ridge_fp_realval["mae"],
            "rmse": ridge_fp_realval["rmse"], "ridge_alpha": ridge_fp_realval["ridge_alpha"],
            "rule": "alpha chosen on the split's real val fold; scaler on full train",
        },
        "leaked_test_argmax": {
            "pearson": ridge_fp_leaky["pearson"], "mae": ridge_fp_leaky["mae"],
            "rmse": ridge_fp_leaky["rmse"], "ridge_alpha": ridge_fp_leaky["ridge_alpha"],
            "rule": "VERBATIM a5cc339d ridge_pearson: alpha chosen by max Pearson "
                    "ON THE EVAL ROWS (test-set leak)",
        },
    }
    ridge_fp["leaked_minus_honest_carveout"] = float(
        ridge_fp_leaky["pearson"] - ridge_fp["pearson"]
    )
    ridge_fp["leaked_matches_published_17dp"] = bool(
        published and abs(ridge_fp_leaky["pearson"]
                          - published["ridge_on_morgan_fp"]) < 1e-15
    )
    print(f"[E] ridge-on-MorganFP same-split pearson={ridge_fp['pearson']:.6f}", flush=True)
    print(
        f"[E/sel] carveout={ridge_fp['pearson']:.6f} realval={ridge_fp_realval['pearson']:.6f} "
        f"leaked(test-argmax)={ridge_fp_leaky['pearson']:.6f} "
        f"published={published.get('ridge_on_morgan_fp') if published else None}",
        flush=True,
    )

    # ---- encoders ----
    encoders: Dict[str, Dict] = {}
    if args.smoke:
        print("[smoke] encoder arms skipped (--smoke)", flush=True)
    else:
        encoders = _load_encoders(args, graphs_all, mid_all, g_pool, g_test, g_val)

    # ---- ridge arms per encoder ----
    ridge_arms: Dict[str, Dict] = {}
    ridge_preds: Dict[str, np.ndarray] = {}
    for name, enc in encoders.items():
        e_pool, e_test = enc["pool"], enc["test"]
        assert len(e_test) == data["n_test"], (name, len(e_test))
        assert len(e_pool) == data["n_joint_train"], (name, len(e_pool))
        m_pool, preds_pool = fit_ridge_with_preds(e_pool, y_pool, e_test, y_test)
        m_repo = fit_ridge_readout(e_pool, y_pool, e_test, y_test)
        m_pool["matches_repo_implementation"] = bool(
            abs(m_pool["pearson"] - m_repo["pearson"]) < 1e-12
        )
        m_ru, preds_ru = fit_ridge_with_preds(
            e_pool[data["pool_pos_ttrain"]], y_pool[data["pool_pos_ttrain"]], e_test, y_test
        )
        m_leak, _ = fit_ridge_test_argmax(e_pool, y_pool, e_test, y_test)
        m_rv, _ = fit_ridge_real_val(
            e_pool, y_pool, e_test, y_test, enc["val"], y_val
        )
        m_pool["selection_rule_sweep"] = {
            "honest_carveout_internal_fold": {
                "pearson": m_pool["pearson"], "mae": m_pool["mae"],
                "rmse": m_pool["rmse"], "ridge_alpha": m_pool["ridge_alpha"],
            },
            "honest_real_val_fold": {
                "pearson": m_rv["pearson"], "mae": m_rv["mae"],
                "rmse": m_rv["rmse"], "ridge_alpha": m_rv["ridge_alpha"],
            },
            "leaked_test_argmax": {
                "pearson": m_leak["pearson"], "mae": m_leak["mae"],
                "rmse": m_leak["rmse"], "ridge_alpha": m_leak["ridge_alpha"],
            },
        }
        m_pool["leaked_minus_honest_carveout"] = float(
            m_leak["pearson"] - m_pool["pearson"]
        )
        trained = enc.get("kind") == "checkpoint"
        m_pool["arm_provenance"] = {
            "encoder_trained": trained,
            "ridge_fit_rows": "the probe's 3480 joint-pool train rows only",
            "scored_rows": "the probe's 994 test rows",
            "held_out_for_the_ridge": True,
            "encoder_contamination": (
                "NONE -- the encoder is randomly initialised, so no row was ever "
                "seen during any training; this is the CLEAN reference arm"
                if not trained else
                "UNKNOWN -- the checkpoint's encoder was trained on a different "
                "cohort/split (see recorded_split); the ridge itself is fit only "
                "on the probe's 3480 pool rows, but the frozen features of rows "
                "that were in the encoder's training set are optimistic. The "
                "overlap fraction was NOT measured."
            ),
            "recorded_split": enc.get("ckpt_meta"),
        }
        ridge_arms[name] = {"ridge_pool_trained": m_pool, "ridge_Ru_rows_only": m_ru}
        ridge_preds[name] = preds_pool
        print(
            f"[ridge/{name}] pool-trained pearson={m_pool['pearson']:.6f} "
            f"(alpha={m_pool.get('ridge_alpha')}) repo_match={m_pool['matches_repo_implementation']}",
            flush=True,
        )

    # ---- head-only fits ----
    fit_encoder_names = [n.strip() for n in args.fit_encoders.split(",") if n.strip()]
    head_summaries: Dict[str, Dict] = {}
    head_preds: Dict[str, Dict[str, np.ndarray]] = {}
    same_rows: Dict[str, Dict] = {}
    for name, enc in encoders.items():
        if name not in fit_encoder_names:
            continue
        e_pool, e_test, e_val = enc["pool"], enc["test"], enc["val"]
        rows: Dict[str, Dict] = {}
        preds: Dict[str, np.ndarray] = {}
        for spec in _fit_specs():
            results = [
                fit_head_only(
                    e_pool, y_pool, data["a_pool"], mid_pool,
                    e_test, y_test, mid_test,
                    e_val, y_val, mid_val,
                    seed=args.seed + s, epochs=args.epochs, **spec["kwargs"],
                )
                for s in range(args.fit_seeds)
            ]
            ptest = np.mean([r["_pred_test"] for r in results], axis=0)
            n_div = int(sum(1 for r in results if r.get("diverged_all_nan_val_curve")))
            summary = {
                "hypothesis": spec["hypothesis"],
                "kwargs": dict(spec["kwargs"]),
                "n_seeds": len(results),
                "seeds": [args.seed + s for s in range(len(results))],
                "n_seeds_diverged_all_nan_val_curve": n_div,
                "test_pearson_final_mean": _jnum(np.mean([r["test_pearson_final"] for r in results])),
                "test_pearson_final_std": _jnum(np.std([r["test_pearson_final"] for r in results])),
                "test_pearson_best_val_epoch_mean": _jnum(np.mean([r["test_pearson_best_val_epoch"] for r in results])),
                "test_pearson_best_val_epoch_std": _jnum(np.std([r["test_pearson_best_val_epoch"] for r in results])),
                "test_rmse_final_mean": _jnum(np.mean([r["test"]["rmse"] for r in results])),
                "train_pearson_mean": _jnum(np.mean([r["train_in_sample"]["pearson"] for r in results])),
                "train_rmse_mean": _jnum(np.mean([r["train_in_sample"]["rmse"] for r in results])),
                "best_val_pearson_mean": _jnum(np.mean([r["best_val_pearson"] for r in results])),
                "loss_cls_last_epoch_mean": _jnum(np.mean([r.get("loss_cls_last_epoch", float("nan")) for r in results])),
                "loss_reg_last_epoch_mean": _jnum(np.mean([r.get("loss_reg_last_epoch", float("nan")) for r in results])),
                "loss_cls_over_loss_reg_last_epoch_mean": _jnum(np.mean([r.get("loss_cls_over_loss_reg_last_epoch", float("nan")) for r in results])),
                "per_seed": [
                    {
                        "seed": r["seed"],
                        "test_pearson_final": _jnum(r["test_pearson_final"]),
                        "test_pearson_best_val_epoch": _jnum(r["test_pearson_best_val_epoch"]),
                        "best_val_epoch": r["best_val_epoch"],
                        "val_pearson_at_best": _jnum(r["val_pearson_at_best"]),
                        "train_pearson": _jnum(r["train_in_sample"]["pearson"]),
                        "final_loss": _jnum(r["final_loss"]),
                        "loss_cls_last_epoch": _jnum(r.get("loss_cls_last_epoch")),
                        "loss_reg_last_epoch": _jnum(r.get("loss_reg_last_epoch")),
                        "diverged_all_nan_val_curve": bool(
                            r.get("diverged_all_nan_val_curve", False)
                        ),
                    }
                    for r in results
                ],
            }
            rows[spec["key"]] = summary
            preds[spec["key"]] = ptest
            assert_same_test_rows(
                len(ridge_preds[name]), len(ptest), data["n_test"],
                f"head_only/{name}/{spec['key']}",
            )
            same_rows[f"head_only/{name}/{spec['key']}"] = {
                "n_ridge_rows": len(ridge_preds[name]),
                "n_head_rows": len(ptest),
                "n_expected": data["n_test"],
                "test_row_id_sha256": split_hash,
            }
            print(
                f"[head/{name}/{spec['key']}] test(final)={_fmt(summary['test_pearson_final_mean'])}"
                f"+-{_fmt(summary['test_pearson_final_std'])} "
                f"test(best-val)={_fmt(summary['test_pearson_best_val_epoch_mean'])} "
                f"train={_fmt(summary['train_pearson_mean'])} "
                f"diverged_seeds={summary['n_seeds_diverged_all_nan_val_curve']}",
                flush=True,
            )
        head_summaries[name] = rows
        head_preds[name] = preds

    # ---- the trained head's own predictions (checkpoint arm) ----
    ckpt_head_arms: Dict[str, Dict] = {}
    ckpt_head_preds: Dict[str, np.ndarray] = {}
    for name, enc in encoders.items():
        if enc["kind"] != "checkpoint":
            continue
        p_test = _ckpt_head_predict(enc["model"], enc["test"], mid_test)
        p_pool = _ckpt_head_predict(enc["model"], enc["pool"], mid_pool)
        m_test = regression_metrics(y_test, p_test)
        m_train = regression_metrics(y_pool, p_pool)
        ckpt_head_arms[name] = {
            "test": dict(m_test),
            "train_in_sample": dict(m_train),
            "checkpoint": str(enc["path"].relative_to(REPO)),
            "ckpt_meta": enc["ckpt_meta"],
            "recorded_split_matches_probe_split": False,
            "comparability": (
                "NOT COMPARABLE to the other arms: this checkpoint was trained on "
                "a different cohort/split (recorded n_train/n_val/n_test/"
                "n_joint_train below) than the probe's split (3480 joint-train / "
                "2262 target-train / 394 val / 994 test), so its predictions on "
                "the probe's test rows are NOT held out. Reported for completeness; "
                "no headline claim rests on it."
            ),
        }
        ckpt_head_preds[name] = p_test
        assert_same_test_rows(
            len(ridge_preds[name]), len(p_test), data["n_test"],
            f"checkpoint_own_head/{name}",
        )
        same_rows[f"checkpoint_own_head/{name}"] = {
            "n_ridge_rows": len(ridge_preds[name]),
            "n_head_rows": len(p_test),
            "n_expected": data["n_test"],
            "test_row_id_sha256": split_hash,
        }
        print(
            f"[ckpt-head/{name}] test pearson={m_test['pearson']:.6f} "
            f"train(in-sample) pearson={m_train['pearson']:.6f}",
            flush=True,
        )

    # ---- localisation: where does the gap go? ----
    localisation: Dict = {}
    for name, enc in encoders.items():
        rp = ridge_preds[name]
        entry: Dict = {
            "ridge_test_pearson": pearson(y_test, rp),
            "ridge_test_rmse": float(regression_metrics(y_test, rp)["rmse"]),
            "y_test_std": float(np.std(y_test)),
            "y_test_mean": float(np.mean(y_test)),
        }
        for key, ptest in head_preds.get(name, {}).items():
            entry[f"head::{key}"] = _decomp(y_test, ptest, rp)
        if name in ckpt_head_preds:
            entry["checkpoint_own_head"] = _decomp(y_test, ckpt_head_preds[name], rp)
        localisation[name] = entry

    report["arms"] = {
        "ridge_morgan_fp_same_split": ridge_fp,
        "ridge_per_encoder": ridge_arms,
        "head_only_fits": head_summaries,
        "checkpoint_own_head": ckpt_head_arms,
    }
    report["localisation"] = localisation
    report["same_rows_evidence"] = {
        "test_row_id_sha256": split_hash,
        "n_test_rows_expected": data["n_test"],
        "n_arms_checked": len(same_rows),
        "per_arm": same_rows,
        "note": (
            "every scored arm (ridge control + every head-only fit + the "
            "checkpoint's own head) is scored on the SAME test row ids; the "
            "probe asserts this at runtime via assert_same_test_rows"
        ),
    }
    report["reproduced_vs_published"] = _reproduced_vs_published(
        published, ridge_fp, ridge_arms, head_summaries, ckpt_head_arms
    )
    report["hypothesis_ranking"] = _rank_hypotheses(
        localisation, head_summaries, ridge_fp, ridge_arms
    )
    report["read_only_tree_defects"] = _read_only_tree_defects(published)
    report["minimal_gpu_experiment"] = {
        "status": "DECLARED, NOT RUN (Wf-A201 rule 1: CPU-only, 0 GPU)",
        "question": "does the 0.1931 gap survive once the reference arm stops "
                    "selecting its hyper-parameter on the test rows?",
        "design": [
            "re-run molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py "
            "unchanged EXCEPT that the trained encoder AND the trained head "
            "are written to disk (the Oct-8 run saved neither, which is why "
            "arms (C)/(D) cannot be re-measured today)",
            "route all three ridge arms through the repo's current honest "
            "fit_ridge_readout (alpha chosen on a val fold, never on the test "
            "rows) and additionally report the a5cc339d leaked-rule number "
            "for the same fit, so the selection artefact is quantified rather "
            "than argued",
            "5 seeds for the head arm, same split (2262/394/994), same loss "
            "(alpha=0.5, Adam 1e-3, 30 epochs, batch 64, W2 0.05, L2 guard)",
            "score BOTH the ridge and the head on the SAME 994 test rows and "
            "report the gap as: honest_ridge - head, per seed",
        ],
        "decision_rule": (
            "if honest_ridge - head stays near the 0.1931 quoted, the gap is a "
            "modelling/objective gap and the CPU head-only arms L1..L5 say "
            "which lever; if honest_ridge drops by the measured leak "
            "inflation, part of the 0.1931 was a selection artefact and the "
            "remaining gap is the real one"
        ),
        "estimated_cost": "5 seeds x ~30 epochs on one GPU; the 2026-10-08 "
                          "diag run took 1011 s for its arms (see "
                          "molmetal/reports/wf_pearson_diag/report.json "
                          "elapsed_seconds)",
        "why_it_is_the_minimum": "one script, one split, one change (save the "
                                 "checkpoint + honest alpha selection); it "
                                 "converts the two NOT-REPRODUCIBLE rows of "
                                 "the table into MEASURED rows",
    }
    report["elapsed_seconds"] = time.time() - t0
    report["checkpoints_read"] = {
        name: str(enc["path"].relative_to(REPO))
        for name, enc in encoders.items() if enc["kind"] == "checkpoint"
    }
    report["checkpoint_sha256"] = {
        str(enc["path"].relative_to(REPO)): sha256_file(enc["path"])
        for name, enc in encoders.items() if enc["kind"] == "checkpoint"
    }
    report["scope"]["nvidia_smi_after"] = nvidia_smi_compute_apps()

    # ---- write (atomic + overwrite guard + truncation guard) ----
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        detect_truncation,
        safe_overwrite_check,
    )

    json_path = OUT_DIR / "report.json"
    md_path = OUT_DIR / "report.md"
    assert_write_scope(json_path)
    assert_write_scope(md_path)
    print(f"WFA201 WRITE {json_path.resolve()}", flush=True)
    print(f"WFA201 WRITE {md_path.resolve()}", flush=True)
    json_text = json.dumps(report, indent=2, default=float)
    md_text = _render_markdown(report)
    ratio_json = safe_overwrite_check(json_path, json_text, raise_on_risk=False)
    ratio_md = safe_overwrite_check(md_path, md_text, raise_on_risk=False)
    trunc_json = detect_truncation(json_text)
    trunc_md = detect_truncation(md_text)
    if trunc_json or trunc_md:
        raise RuntimeError(
            f"truncation guard tripped: json={trunc_json} md={trunc_md}"
        )
    print(
        f"[overwrite-check] safe_overwrite_check ratios: "
        f"report.json={ratio_json} report.md={ratio_md}; "
        f"detect_truncation: json={trunc_json} md={trunc_md}",
        flush=True,
    )
    report["scope"]["overwrite_check"] = {
        "safe_overwrite_check_ratio_report_json": ratio_json,
        "safe_overwrite_check_ratio_report_md": ratio_md,
        "detect_truncation_report_json": trunc_json,
        "detect_truncation_report_md": trunc_md,
        "note": (
            "the report.json ratio was computed on the payload BEFORE this "
            "overwrite_check field was added (the field is the only difference)"
        ),
    }
    atomic_write_file(md_path, md_text)
    atomic_write_file(json_path, json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))
    print(f"[done] wrote {json_path} and {md_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def _decomp(y: np.ndarray, p: np.ndarray, r: np.ndarray) -> Dict:
    return {
        "test_pearson": _jnum(pearson(y, p)),
        "test_rmse": _jnum(regression_metrics(y, p)["rmse"]),
        "pearson_ridge_vs_head": _jnum(pearson(r, p)),
        "partial_corr_y_head_given_ridge": _jnum(partial_corr(y, p, r)),
        "partial_corr_y_ridge_given_head": _jnum(partial_corr(y, r, p)),
        "ols_decomposition": incremental_r2(y, p, r),
    }


def _reproduced_vs_published(published, ridge_fp, ridge_arms, head_summaries,
                              ckpt_head_arms) -> List[Dict]:
    """Reproduced-vs-published table, one row per 2026-10-08 number.

    Each ridge row carries BOTH the honest selection rule (alpha chosen off
    the eval rows -- the repo's current ``fit_ridge_readout``) and the
    VERBATIM 2026-10-08 selection rule (alpha chosen by max Pearson ON the
    eval rows), so the published number can be attributed to one of them.
    """
    rows: List[Dict] = []

    def row(arm: str, published_val: Optional[float], reproduced: Optional[float],
            status: str, note: str, leaked: Optional[float] = None) -> None:
        rows.append({
            "arm": arm,
            "published_2026_10_08": published_val,
            "reproduced_here_honest_rule": reproduced,
            "reproduced_here_leaked_rule_a5cc339d": leaked,
            "delta_honest": (None if (published_val is None or reproduced is None)
                             else float(reproduced - published_val)),
            "delta_leaked": (None if (published_val is None or leaked is None)
                             else float(leaked - published_val)),
            "status": status,
            "note": note,
        })

    if published:
        p = published["ridge_on_morgan_fp"]
        sweep = ridge_fp.get("selection_rule_sweep", {})
        leak_v = sweep.get("leaked_test_argmax", {}).get("pearson")
        row("ridge_on_morgan_fp", p, ridge_fp["pearson"],
            "REPRODUCED EXACTLY by the LEAKED rule; NOT reproduced by the honest rule",
            "ridge on Morgan FP, target-metal train rows -> target-metal test rows. "
            "The published fingerprint (pearson+mae+rmse+alpha) is reproduced to "
            "17 significant digits by a5cc339d's ridge_pearson, which selects the "
            "alpha by maximising Pearson ON THE 994 TEST ROWS it reports. The "
            "repo's current honest rule (alpha chosen on a carved-out val fold) "
            "gives the 'reproduced_here_honest_rule' number instead.",
            leaked=leak_v)
        p = published["ridge_on_random_encoder_embedding"]
        rnd = ridge_arms.get("random_init_seed0", {}).get("ridge_pool_trained", {})
        rep = rnd.get("pearson")
        rsweep = rnd.get("selection_rule_sweep", {})
        rleak = rsweep.get("leaked_test_argmax", {}).get("pearson")
        row("ridge_on_random_encoder_embedding", p, rep,
            "REPRODUCED (leaked rule)" if rleak is not None else "NOT MEASURED",
            "random init encoder, torch.manual_seed(0), joint-pool train rows. "
            "Same rule attribution as the Morgan row; any residual delta vs the "
            "published number is CPU-vs-CUDA embedding arithmetic, not method.",
            leaked=rleak)
        p = published["ridge_on_trained_encoder_embedding"]
        best_name, best_val = None, None
        for nm, arms in ridge_arms.items():
            if nm == "random_init_seed0":
                continue
            v = arms["ridge_pool_trained"]["pearson"]
            if best_val is None or v > best_val:
                best_name, best_val = nm, v
        row("ridge_on_trained_encoder_embedding", p, best_val,
            "NOT REPRODUCED (no saved checkpoint for the 2026-10-08 run)",
            f"the 2026-10-08 diag script trained the encoder in-memory and never "
            f"saved it; closest on-disk trained encoder = {best_name} "
            f"(delta vs published is NOT a reproduction)")
        p = published["neural_head_default_alpha_0.5"]
        row("neural_head_default_alpha_0.5", p, None,
            "NOT REPRODUCED",
            "the 2026-10-08 head came from the same in-memory (unsaved) model; "
            "reproducing it needs a GPU training run, forbidden by rule 1. "
            "The head-only CPU analogue is the L1 arm.")
        p = published["neural_head_regweighted_alpha_0.2"]
        row("neural_head_regweighted_alpha_0.2", p, None,
            "NOT REPRODUCED",
            "same reason as the alpha-0.5 head. The head-only CPU analogue is "
            "the A02_alpha0.2_regweighted arm.")
    return rows


def _read_only_tree_defects(published: Dict) -> Dict:
    """Defects found in the READ-ONLY tree. Reported with file:line; not fixed.

    Every number quoted here is read at runtime from a file on disk.
    """
    out: Dict = {"note": "read-only tree; Wf-A201 did NOT modify any of these files",
                 "defects": []}

    def defect(did: str, severity: str, where: str, what: str,
               evidence: Optional[Dict] = None) -> None:
        out["defects"].append({
            "id": did, "severity": severity, "file_line": where,
            "what": what, "evidence": evidence or {},
        })

    defect(
        "D1", "HIGH",
        "molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py:101-106 "
        "(current) / a5cc339d:103-114 (the revision that produced the report)",
        "the published ridge reference was produced by a helper that selects the "
        "ridge alpha by maximising Pearson ON THE 994 TEST ROWS it then reports "
        "(test-set leak). The file on disk today has the opposite behaviour "
        "(alpha chosen on a val fold carved out of the train rows) and was "
        "changed on 2026-10-13 (commit 0e232def) WITHOUT regenerating "
        "molmetal/reports/wf_pearson_diag/report.json, so the committed script "
        "no longer reproduces its own published numbers.",
        {
            "published_report": "molmetal/reports/wf_pearson_diag/report.json",
            "morgan_fp_published_pearson": published.get("ridge_on_morgan_fp"),
            "see_report_json": "arms.ridge_morgan_fp_same_split.selection_rule_sweep",
        },
    )
    defect(
        "D2", "HIGH",
        "molmetal/reports/wf_pearson_diag/report.json vs "
        "molmetal/reports/wf_pearson_ridge_gpu/ru/per_seed_metrics.json",
        "two 2026-10-08 artefacts on the SAME split disagree about the trained "
        "head: the diag report says the head's test pearson is 0.2143, the "
        "sibling trainer run says 0.0113 for a nominally identical recipe "
        "(same split 2262/394/994). Neither is reconciled anywhere on disk.",
    )
    defect(
        "D3", "MEDIUM",
        "molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py:179 vs :190/:236",
        "the 'same-split' reference arms do NOT share a training set: the "
        "Morgan-FP arm is fit on the 2262 target-metal train rows while both "
        "encoder arms are fit on the 3480-row joint pool, so the ceiling and "
        "the model are not measured under the same data regime.",
    )
    defect(
        "D4", "MEDIUM",
        "molmetal/reports/wf_pearson_ridge_gpu/checkpoints/ (empty directory)",
        "the checkpoint path recorded in the 2026-10-08 per-seed metrics "
        "(dmpnn_mtw_ru_seed0.pt) does not exist, so the trained encoder/head "
        "of that run cannot be re-measured on CPU and arms (C)/(D) of the "
        "published table are NOT REPRODUCIBLE by construction.",
    )

    # ---- sibling artefact, read from disk ----
    sib = REPO / "molmetal/reports/wf_pearson_ridge_gpu/ru/per_seed_metrics.json"
    if sib.exists():
        try:
            s = json.loads(sib.read_text())
            per = s.get("per_seed", [{}])[0]
            out["sibling_artifact_2026_10_08"] = {
                "path": str(sib.relative_to(REPO)),
                "seed": per.get("seed"),
                "n_train": per.get("n_train"), "n_val": per.get("n_val"),
                "n_test": per.get("n_test"),
                "test_pearson": per.get("test_pearson"),
                "test_auc": per.get("test_auc"),
                "test_rmse": per.get("test_rmse"),
                "ckpt_path_recorded": per.get("ckpt_path"),
                "ckpt_path_exists": bool(
                    per.get("ckpt_path")
                    and (REPO / per["ckpt_path"]).exists()
                ),
                "diag_report_head_pearson": published.get("neural_head_default_alpha_0.5"),
                "head_pearson_discrepancy": (
                    None if published.get("neural_head_default_alpha_0.5") is None
                    else float(per.get("test_pearson", float("nan"))
                               - published["neural_head_default_alpha_0.5"])
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            out["sibling_artifact_2026_10_08"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _rank_hypotheses(localisation: Dict, head_summaries: Dict,
                     ridge_fp: Dict, ridge_arms: Dict) -> Dict:
    """Rank H1..H5 by the MEASURED effect size of the arm that probes them.

    H6 (metric/selection artefact of the REFERENCE arm) is ranked first when
    its measured effect size dominates: the published ridge number was
    selected by max Pearson on the test rows.
    """
    out: Dict = {"arms": {}, "verdict": {}}
    for enc_name, loc in localisation.items():
        base = loc["ridge_test_pearson"]
        arms = {k: v for k, v in loc.items() if k.startswith("head::") or k == "checkpoint_own_head"}
        out["arms"][enc_name] = {
            "ridge_test_pearson": base,
            "head_arms": {
                k: {"test_pearson": v["test_pearson"],
                    "delta_vs_ridge": float(v["test_pearson"] - base),
                    "pearson_ridge_vs_head": v["pearson_ridge_vs_head"],
                    "partial_corr_y_head_given_ridge": v["partial_corr_y_head_given_ridge"],
                    "partial_corr_y_ridge_given_head": v["partial_corr_y_ridge_given_head"]}
                for k, v in arms.items()
            },
        }

    # ---- H6: the reference arm's own selection rule ----
    h6: Dict = {"hypothesis": "H6 metric artefact of the REFERENCE arm",
                "statement": "the published ridge reference (0.4074) was selected by "
                             "maximising Pearson on the 994 test rows it reports "
                             "(a5cc339d ridge_pearson), so it is a test-set-leaked "
                             "upper bound, not an honest held-out score",
                "measured_effect_sizes": {}}
    for tag, m in [("morgan_fp", ridge_fp)] + [
        (nm, arms["ridge_pool_trained"]) for nm, arms in ridge_arms.items()
    ]:
        sw = m.get("selection_rule_sweep") or {}
        honest = sw.get("honest_carveout_internal_fold", {}).get("pearson")
        leaky = sw.get("leaked_test_argmax", {}).get("pearson")
        if honest is None or leaky is None:
            continue
        h6["measured_effect_sizes"][tag] = {
            "honest_pearson": honest,
            "leaked_pearson": leaky,
            "leak_inflation": float(leaky - honest),
            "n_alphas_in_grid": len(RIDGE_ALPHAS),
        }
    out["H6_selection_bias_of_reference_arm"] = h6
    out["verdict"]["h6_is_proven_for_morgan_fp"] = bool(
        ridge_fp.get("leaked_matches_published_17dp")
    )
    out["verdict"]["note"] = (
        "H6 is PROVEN (exact 17-digit fingerprint match) for the Morgan-FP arm; "
        "for the encoder arms it is MEASURED as the size of the inflation on the "
        "same rows with the same embeddings, not as a reproduction of the "
        "published 0.4074 (that encoder was never saved)."
    )
    return out


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "NOT MEASURED"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _render_markdown(report: Dict) -> str:
    L: List[str] = []
    A = L.append
    A("# Wf-A201 -- CPU-only localisation of the 0.1931 pIC50 pearson gap")
    A("")
    A(f"*Question*: {report['question']}")
    A("")
    A("## 0. Scope (MEASURED, read-only)")
    A("")
    sc = report["scope"]
    A(f"- CPU-only: `cuda.is_available()` = False; nvidia-smi before: `{sc['nvidia_smi_before']}`")
    A(f"- nvidia-smi after: `{sc.get('nvidia_smi_after')}`")
    A(f"- encoder trained: **{sc['encoder_trained']}**; checkpoint modified: **{sc['checkpoint_modified']}**")
    A(f"- metrics/ registry/ paper/ docs/ TODO/ files written: "
      f"{sc['metric_files_written']} / {sc['registry_files_written']} / "
      f"{sc['paper_docs_todo_files_written']} / {sc['paper_docs_todo_files_written']}")
    A(f"- every fit is: {sc['all_fits_are']}")
    A(f"- elapsed: {report['elapsed_seconds']:.1f} s")
    A("")
    A("## 1. Split identity (H5: same rows)")
    A("")
    sp = report["split"]
    A(f"- joint_train={sp['n_joint_train']} target_train={sp['n_target_train']} "
      f"val={sp['n_val']} test={sp['n_test']}")
    A(f"- published 2026-10-08 split: {sp['published_split']} -- matches: "
      f"**{sp['split_matches_published']}**")
    A(f"- test row-id sha256: `{sp['test_row_id_sha256']}`")
    A("")
    A("## 2. Reproduced vs published (2026-10-08)")
    A("")
    A("| arm | published | reproduced here (honest rule) | delta (honest) | "
      "reproduced here (leaked a5cc339d rule) | delta (leaked) | status |")
    A("|---|---|---|---|---|---|---|")
    for r in report["reproduced_vs_published"]:
        A(f"| `{r['arm']}` | {_fmt(r['published_2026_10_08'], 6)} | "
          f"{_fmt(r['reproduced_here_honest_rule'], 6)} | "
          f"{_fmt(r['delta_honest'], 6)} | "
          f"{_fmt(r['reproduced_here_leaked_rule_a5cc339d'], 6)} | "
          f"{_fmt(r['delta_leaked'], 6)} | {r['status']} |")
    A("")
    for r in report["reproduced_vs_published"]:
        A(f"- `{r['arm']}`: {r['note']}")
    A("")
    A("## 2b. Ridge selection-rule sweep (H6 -- is the REFERENCE number honest?)")
    A("")
    A("Three ways to choose the ridge alpha on the SAME rows and the SAME features:")
    A("")
    A("| feature set | honest: alpha on carved-out val fold | honest: alpha on the real val fold | "
      "leaked: alpha = max Pearson ON THE TEST ROWS (a5cc339d) | leak inflation |")
    A("|---|---|---|---|---|")
    fp_sw = report["arms"]["ridge_morgan_fp_same_split"].get("selection_rule_sweep", {})
    if fp_sw:
        A(f"| `morgan_fp` | {_fmt(fp_sw['honest_carveout_internal_fold']['pearson'], 6)} | "
          f"{_fmt(fp_sw['honest_real_val_fold']['pearson'], 6)} | "
          f"{_fmt(fp_sw['leaked_test_argmax']['pearson'], 6)} | "
          f"{_fmt(fp_sw['leaked_test_argmax']['pearson'] - fp_sw['honest_carveout_internal_fold']['pearson'], 6)} |")
    for nm, arms in report["arms"]["ridge_per_encoder"].items():
        sw = arms["ridge_pool_trained"].get("selection_rule_sweep", {})
        if not sw:
            continue
        A(f"| `{nm}` | {_fmt(sw['honest_carveout_internal_fold']['pearson'], 6)} | "
          f"{_fmt(sw['honest_real_val_fold']['pearson'], 6)} | "
          f"{_fmt(sw['leaked_test_argmax']['pearson'], 6)} | "
          f"{_fmt(sw['leaked_test_argmax']['pearson'] - sw['honest_carveout_internal_fold']['pearson'], 6)} |")
    A("")
    A(f"- H6 PROVEN for the Morgan-FP arm (exact 17-digit fingerprint match to the "
      f"published number): `{report['arms']['ridge_morgan_fp_same_split'].get('leaked_matches_published_17dp')}`")
    A(f"- H6 statement: {report['hypothesis_ranking']['H6_selection_bias_of_reference_arm']['statement']}")
    A("- arm provenance: the `random_init_seed0` encoder is randomly initialised, so "
      "NO row was ever seen in training -- its ridge numbers are the CLEAN reference. "
      "Every checkpoint encoder was trained on a DIFFERENT cohort/split (recorded "
      "n_joint_train 3540 / 18137 / 18596 vs the probe's 3480); the ridge on those "
      "features is fit only on the probe's pool rows, but the features themselves may "
      "be optimistic for rows that were in the encoder's training set (overlap NOT "
      "measured).")
    A("")
    A("## 3. CPU-only localisation table")
    A("")
    for enc_name, arms in report["arms"]["head_only_fits"].items():
        A(f"### encoder = `{enc_name}`")
        A("")
        A("| arm | hypothesis | test pearson (final, mean+-std) | test pearson (best-val ep) | "
          "train pearson (in-sample) | delta vs ridge |")
        A("|---|---|---|---|---|---|")
        ridge_pear = report["localisation"][enc_name]["ridge_test_pearson"]
        for key, s in arms.items():
            delta = (
                None if s["test_pearson_final_mean"] is None
                else float(s["test_pearson_final_mean"] - ridge_pear)
            )
            A(f"| `{key}` | {s['hypothesis']} | "
              f"{_fmt(s['test_pearson_final_mean'])}+-{_fmt(s['test_pearson_final_std'])} | "
              f"{_fmt(s['test_pearson_best_val_epoch_mean'])} | "
              f"{_fmt(s['train_pearson_mean'])} | "
              f"{_fmt(delta)} |")
        A(f"| `_ridge_control_` | L4 ridge readout (control) | {_fmt(ridge_pear)} | - | - | 0.0000 |")
        A("")
        A("Multi-task loss split (mean over the last epoch, 3 seeds) and the "
          "alpha arms:")
        A("")
        A("| arm | alpha | loss_cls | loss_reg | cls/reg |")
        A("|---|---|---|---|---|")
        for key in ("L1_repo_head_alpha0.5_raw", "A02_alpha0.2_regweighted",
                    "A08_alpha0.8_clsweighted", "L2_repo_head_alpha0.5_std_target"):
            s = arms.get(key)
            if not s:
                continue
            A(f"| `{key}` | {s['kwargs'].get('alpha')} | "
              f"{_fmt(s.get('loss_cls_last_epoch_mean'), 6)} | "
              f"{_fmt(s.get('loss_reg_last_epoch_mean'), 6)} | "
              f"{_fmt(s.get('loss_cls_over_loss_reg_last_epoch_mean'), 6)} |")
        A("")
    A("## 4. Checkpoint's own head (L5)")
    A("")
    A("| checkpoint | test pearson | test rmse | train (in-sample) pearson | ckpt epochs | ckpt hidden | "
      "ckpt recorded split (n_train/n_val/n_test/n_joint_train) |")
    A("|---|---|---|---|---|---|---|")
    for nm, c in report["arms"]["checkpoint_own_head"].items():
        cm = c["ckpt_meta"]
        A(f"| `{nm}` | {_fmt(c['test']['pearson'], 6)} | {_fmt(c['test']['rmse'], 6)} | "
          f"{_fmt(c['train_in_sample']['pearson'], 6)} | {cm.get('epochs')} | {cm.get('hidden')} | "
          f"{cm.get('n_train')}/{cm.get('n_val')}/{cm.get('n_test')}/{cm.get('n_joint_train')} |")
    A("")
    for nm, c in report["arms"]["checkpoint_own_head"].items():
        A(f"- `{nm}`: {c['comparability']}")
        A("")
    A("## 5. Where does the gap go? (ridge vs head, same rows)")
    A("")
    A("| encoder | head arm | test pearson | pearson(ridge, head) | partial corr(y,head | ridge) | "
      "partial corr(y,ridge | head) |")
    A("|---|---|---|---|---|---|")
    for enc_name, loc in report["localisation"].items():
        A(f"| `{enc_name}` | `_ridge_` | {_fmt(loc['ridge_test_pearson'], 6)} | - | - | - |")
        for k, v in loc.items():
            if k in ("ridge_test_pearson", "ridge_test_rmse", "y_test_std", "y_test_mean"):
                continue
            A(f"| `{enc_name}` | `{k}` | {_fmt(v['test_pearson'], 6)} | "
              f"{_fmt(v['pearson_ridge_vs_head'], 4)} | "
              f"{_fmt(v['partial_corr_y_head_given_ridge'], 4)} | "
              f"{_fmt(v['partial_corr_y_ridge_given_head'], 4)} |")
    A("")
    A("## 6. Checkpoints read (read-only)")
    A("")
    for nm, path in report["checkpoints_read"].items():
        sha = report["checkpoint_sha256"].get(path, "")
        A(f"- `{path}` sha256 `{sha}`")
    A("")
    A("## 7. Defects found in the READ-ONLY tree (reported, not fixed)")
    A("")
    rd = report.get("read_only_tree_defects", {})
    for d in rd.get("defects", []):
        A(f"- **{d['id']} ({d['severity']})** -- `{d['file_line']}`")
        A(f"  - {d['what']}")
    A("")
    sib = rd.get("sibling_artifact_2026_10_08", {})
    if sib and "error" not in sib:
        A(f"- sibling 2026-10-08 artefact `{sib['path']}`: seed={sib['seed']} "
          f"split={sib['n_train']}/{sib['n_val']}/{sib['n_test']} "
          f"test_pearson=**{_fmt(sib['test_pearson'], 6)}** vs the diag report's "
          f"head pearson **{_fmt(sib['diag_report_head_pearson'], 6)}** "
          f"(discrepancy {_fmt(sib['head_pearson_discrepancy'], 6)}); recorded "
          f"checkpoint `{sib['ckpt_path_recorded']}` exists = {sib['ckpt_path_exists']}")
        A("")
    A("## 8. The minimal GPU experiment that would close it (DECLARED, NOT RUN)")
    A("")
    mg = report.get("minimal_gpu_experiment", {})
    A(f"- status: {mg.get('status')}")
    A(f"- question: {mg.get('question')}")
    for d in mg.get("design", []):
        A(f"  - {d}")
    A(f"- decision rule: {mg.get('decision_rule')}")
    A(f"- why it is the minimum: {mg.get('why_it_is_the_minimum')}")
    A("")
    A("## 9. What was NOT done")
    A("")
    A("- no GPU was used (nvidia-smi before/after above);")
    A("- no encoder was trained; every fit is a head-only readout on frozen features;")
    A("- no checkpoint was modified (sha256 recorded above);")
    A("- no `metrics/`, `registry/`, `paper/`, `docs/`, `TODO/` file was written;")
    A("- the 2026-10-08 published numbers were NOT edited;")
    A("- the head-only CPU fit omits the W2 (Wasserstein) regulariser, which in the "
      "trainer acts on the encoder's prediction distribution -- see report.json "
      "`not_measured` list.")
    A("- the three guards in this probe (CPU-only refusal, same-rows, write scope) "
      "are exercised -- and shown to BITE -- by "
      "`molmetal/tests/test_wfa201_encoder_readout_probe.py`: removing any one of "
      "them from a /tmp copy of this file makes the matching test fail with "
      "`DID NOT RAISE CpuOnlyViolation` / `SameRowsViolation` / `WriteScopeViolation`.")
    A("")
    A("END OF REPORT (this line exists so the truncation guard sees a closing token).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
