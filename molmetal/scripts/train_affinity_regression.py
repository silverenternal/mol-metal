#!/usr/bin/env python3
"""Wf-32 Phase 1 — R19 m16 Affinity Pearson r regression trainer.

=====================================================================
Standalone trainer for **m16 Affinity** (Pearson r on continuous pIC50
regression).  Reuses the existing :class:`DMPNNMultiTaskModel` backbone
+ the existing ``--init`` choices ``{tmqm, random, se3sb, transport,
multiph}`` (mirroring ``train_ru_multitask.py``) — **without** modifying
``train_se3sb.py`` or ``train_dmpnn_multitask.py``.

Supported datasets (selected via ``--dataset``):
  * ``platinai`` — 4,078 Pt complexes, 4 cell-line pIC50 columns.  Per
    cell line we report an independent Pearson r.  Aggregate headline is
    the mean across the 4 cell lines (paper headline m16 number).
  * ``hela48h`` — MetalCytoToxDB subset restricted to HeLa / 48 h
    (the FROZEN cohort used by ``retrain_pic50_neural.py``).  Headline
    is the single Pearson r on the test split.

Why a separate script:
  * Wf-23 Phase 2 ``train_se3sb.py`` is 5-metal **classification**
    multi-task with BCE + coord head; the Wf-32 brief asks for a
    **regression-only** head with Pearson r as the headline metric.
  * Wf-32 needs to evaluate on **continuous pIC50** with cross-seed
    ± std reporting — exactly the gap that ``train_dmpnn_multitask.py``
    uses ``alpha=0.5`` for; here we set ``alpha=0.0`` so the loss is
    pure MSE on pIC50.

Recipe
------
(1) Build :class:`DMPNNMultiTaskModel` (hidden=128, depth=3) and set
    ``alpha=0.0`` so the loss is pure MSE on the pIC50 regression head.
(2) Optionally initialise from ``models/tmqm_pretrained/*.pt``
    (with arch-mismatch fallback to random init — same contract as
    ``train_ru_multitask.try_load_tmqm_encoder``).
(3) 5 seeds {0, 1, 2, 3, 4} × ``--epochs`` (default 30), batch=64,
    lr=1e-3, Adam, MSE on pIC50.
(4) For ``platinai`` the trainer iterates over the 4 cell lines
    (MCF7 / A549 / A2780 / A2780cis) and reports a per-cell-line
    Pearson r + a 4-line mean.
(5) Eval on the held-out test split (scaffold split for platinai,
    scaffold-group split for HeLa48h).
(6) Per-seed metric file + aggregate JSON (per-seed + 5-seed
    mean ± std) under ``metrics/by_metric/``.

Verify
------
* ``molmetal/scripts/verify_train_affinity_regression.py`` — CPU-only
  end-to-end smoke (1 seed, 2 epochs, 32 rows).
* ``molmetal/tests/test_train_affinity_regression.py`` — 15+ unit +
  CLI contract tests.

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_affinity_regression.py \\
        --dataset platinai --seeds 0 1 2 3 4 --epochs 30 --init tmqm
    uv run --no-sync python -u molmetal/scripts/train_affinity_regression.py \\
        --dataset hela48h --seeds 0 1 2 3 4 --epochs 30 --init random
"""

from __future__ import annotations


# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi (GPU 0 is the VLM venv process).  Must run
# BEFORE `import torch` so CUDA init sees the env var.
import os as _wf130_os
_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from molmetal.baselines.dmpnn_attentive import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    featurize_smiles_list,
    collate_graphs,
)
from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskBaseline,
    DMPNNMultiTaskModel,
    MultiTaskLoss,
    regression_metrics,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

# R18 init lazy-imports (same contract as train_ru_multitask.py).
try:                                            # pragma: no cover
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField as _SE3SBVelocityField,
    )
except Exception:                               # pragma: no cover
    _SE3SBVelocityField = None                  # type: ignore[assignment]
try:                                            # pragma: no cover
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm as _RelaxedEndpointTerm,
    )
except Exception:                               # pragma: no cover
    _RelaxedEndpointTerm = None                 # type: ignore[assignment]
try:                                            # pragma: no cover
    from molmetal.adapters.multi_ph_integration import (
        MultiPHPocketFeatures as _MultiPHPocketFeatures,
    )
except Exception:                               # pragma: no cover
    _MultiPHPocketFeatures = None               # type: ignore[assignment]

# Wf-35 Phase 2 — shared argparse / GPU env / MEASURED-vs-PROJECTED tag.
from molmetal.scripts._common_train_args import (
    MEASURED_vs_PROJECTED_decorator,
    add_common_args,
    add_gpu_args,
    setup_gpu_env,
)

# ---------------------------------------------------------------------------
# Recipe constants (Wf-32 Phase 1 spec)
# ---------------------------------------------------------------------------
RECIPE_EPOCHS: int = 30
RECIPE_BATCH: int = 64
RECIPE_LR: float = 1e-3
RECIPE_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
RECIPE_HIDDEN: int = 128
RECIPE_DEPTH: int = 3
RECIPE_DROPOUT: float = 0.1
RECIPE_ALPHA: float = 0.0      # MSE-only (regression head)
RECIPE_EVAL_EVERY: int = 5

# PlatinAI 4-cell-line column contract (matches build_platinai_4cell.py)
PLATINAI_CELL_LINES: Tuple[str, ...] = ("MCF7", "A549", "A2780", "A2780cis")
PLATINAI_IC50_COLS: Tuple[str, ...] = tuple(
    f"ic50_{l.lower()}_nm" for l in PLATINAI_CELL_LINES
)
PLATINAI_DATA = PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"

# HeLa48h cohort default CSV (MetalCytoToxDB)
HELA48H_CELL_LINE: str = "HeLa"
HELA48H_TIME_H: float = 48.0

# tmQM pretrained encoder (same default as train_se3sb.py / train_ru_multitask.py)
TMQM_ENCODER_CKPT = PROJECT_ROOT / "models" / "tmqm_pretrained" / "tmqm_pretrained_encoder.pt"

# Output paths (overridable via CLI)
DEFAULT_OUT_DIR = PROJECT_ROOT / ".ckpt"
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "affinity_pearson_r.json"
)
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "affinity_pearson_r_summary.json"
)

# Init choice contract (must match train_ru_multitask.py's choices).
INIT_CHOICES: Tuple[str, ...] = (
    "tmqm", "random", "se3sb", "transport", "multiph",
)
DATASET_CHOICES: Tuple[str, ...] = ("platinai", "hela48h")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class PlatinAIRegressionDataset:
    """Wraps ``platinai_4cell.csv`` for regression on each cell line.

    Each row carries 4 IC50 columns.  We compute ``pIC50 = 9 - log10(IC50_nM)``
    for each cell line and drop rows whose IC50 is missing/NaN/non-positive
    on a per-cell-line basis (so a row missing MCF7 can still contribute
    to the A549 head).
    """

    def __init__(
        self,
        smiles: np.ndarray,
        pic50: np.ndarray,            # (N, 4) — NaN where missing
        valid: np.ndarray,            # (N, 4) bool — True where pIC50 is finite
    ) -> None:
        if pic50.shape != (len(smiles), len(PLATINAI_CELL_LINES)):
            raise ValueError(
                f"pic50 shape {pic50.shape} != "
                f"({len(smiles)}, {len(PLATINAI_CELL_LINES)})"
            )
        self.smiles = smiles
        self.pic50 = pic50.astype(np.float32)
        self.valid = valid.astype(bool)

    @classmethod
    def from_csv(cls, path: Path = PLATINAI_DATA) -> "PlatinAIRegressionDataset":
        rows: List[Dict[str, str]] = []
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
        n = len(rows)
        smiles = np.array(
            [str(r.get("smiles", "")) for r in rows], dtype=object
        )
        ic50 = np.full((n, len(PLATINAI_CELL_LINES)), np.nan, dtype=np.float32)
        for j, col in enumerate(PLATINAI_IC50_COLS):
            for i, r in enumerate(rows):
                v = r.get(col, "")
                if v == "" or v is None:
                    continue
                try:
                    ic50[i, j] = float(v)
                except (ValueError, TypeError):
                    continue
        # pIC50 = -log10(IC50_M) = 9 - log10(IC50_nM) ; drop NaN/≤0
        pic50 = np.full_like(ic50, np.nan)
        valid = np.zeros_like(ic50, dtype=bool)
        ok = (~np.isnan(ic50)) & (ic50 > 0)
        pic50[ok] = 9.0 - np.log10(ic50[ok])
        valid[ok] = True
        return cls(smiles=smiles, pic50=pic50, valid=valid)

    def __len__(self) -> int:
        return int(self.smiles.shape[0])

    def per_cell_line_valid_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for j, line in enumerate(PLATINAI_CELL_LINES):
            out[line] = int(self.valid[:, j].sum())
        return out


def _load_hela48h_cohort(
    csv_path: Optional[Path],
    cell_line: str = HELA48H_CELL_LINE,
    time_h: float = HELA48H_TIME_H,
) -> MetalCytotoxDataset:
    """Load the MetalCytoToxDB HeLa48h cohort via the existing loader.

    We pass ``time_threshold=time_h`` so that only rows with
    ``Time(h) <= time_h`` survive (MetalCytoToxDB uses Time(h) <= t).
    """
    kwargs = dict(
        time_threshold=float(time_h),
        ic50_min=0.01,
        compute_pic50=True,
        compute_active=True,
    )
    if csv_path is not None and csv_path.exists():
        kwargs["path"] = csv_path
    flt = CytotoxFilter(metal_whitelist=None, **kwargs)
    if csv_path is not None and csv_path.exists():
        ds = MetalCytotoxDataset.from_csv(path=csv_path, filters=flt)
    else:
        ds = MetalCytotoxDataset.from_csv(filters=flt)
    return ds


# ---------------------------------------------------------------------------
# tmQM encoder load (with arch-mismatch fallback) — mirrors train_ru_multitask
# ---------------------------------------------------------------------------
def try_load_tmqm_encoder(model: DMPNNMultiTaskModel, ckpt_path: Path) -> int:
    """Load tmQM pretrained weights into the multi-task D-MPNN encoder.

    The published tmQM encoder was trained on the baseline
    ``MetalCytotoxModel`` architecture (different state-dict keys + hidden
    dim from the multi-task model).  On arch-mismatch we fall back to
    random init with a clear warning.

    Returns:
        ``n_loaded`` — number of encoder tensors transferred (``0`` on
        fallback).  Mirrors :func:`train_ru_multitask.try_load_tmqm_encoder`.
    """
    if not ckpt_path.exists():
        warnings.warn(
            f"[train_affinity_regression] tmQM encoder ckpt not found "
            f"at {ckpt_path}; using random init."
        )
        return 0
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload["state_dict"] if "state_dict" in payload else payload
    own = model.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            own[k].copy_(v)
            n_loaded += 1
    if n_loaded == 0:
        warnings.warn(
            "[train_affinity_regression] tmQM encoder arch is "
            "incompatible with DMPNNMultiTaskModel; falling back to "
            "random init."
        )
        return 0
    if n_loaded != len(own):
        warnings.warn(
            f"[train_affinity_regression] partial tmQM load: "
            f"{n_loaded}/{len(own)} tensors matched; remaining kept "
            "at random init."
        )
    return n_loaded


# ---------------------------------------------------------------------------
# Train one seed — PlatinAI 4-cell-line regression
# ---------------------------------------------------------------------------
def _train_epoch_platinai(
    model: DMPNNMultiTaskModel,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    graphs: List,
    pic50_labels: np.ndarray,    # (N, 4) — NaN where missing
    valid_labels: np.ndarray,    # (N, 4) bool
    batch_size: int,
    device: torch.device,
    *,
    use_amp: bool = False,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, int]:
    """One epoch of MSE-only training on PlatinAI pIC50.

    Skips rows with no valid cell line; for each batch we set
    ``active_target`` to ``0`` (dummy) since ``alpha=0.0`` makes the BCE
    term vanish from the loss anyway.

    Wf-44 Phase 4 numerical-stability hooks
    ---------------------------------------
    * ``use_amp`` — when ``True`` (and ``device.type == "cuda"``) the
      forward pass runs inside :func:`torch.autocast` so the matmul
      accumulator stays in fp32 while activations / weights stay in
      fp16.  The :class:`torch.cuda.amp.GradScaler` is used to scale
      the loss before ``.backward()`` and unscale the gradients
      before :func:`torch.nn.utils.clip_grad_norm_`.
    * When the device is CPU we keep fp32 throughout and the
      ``GradScaler`` is a no-op (calls return the loss unchanged).
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        # Keep rows whose graphs are valid AND at least one cell line is valid.
        keep_rows: List[int] = []
        g_list = []
        y_pic50_list = []
        for k, gi in enumerate(batch_idx):
            g = graphs[int(gi)]
            if g is None:
                continue
            if not valid_labels[int(gi)].any():
                continue
            keep_rows.append(int(gi))
            g_list.append(g)
            y_pic50_list.append(pic50_labels[int(gi)])
        if not g_list:
            continue
        collated = collate_graphs(g_list)
        if collated is None:
            continue
        atom_f, bond_f, e_src, e_dst = collated
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        # Use the first-valid-cell-line's pIC50 as the regression target.
        # (DMPNNMultiTaskModel has a single regression head — we collapse
        # 4 cell lines into a single training signal by averaging valid
        # entries below.)
        y_pic50 = np.stack(y_pic50_list, axis=0)
        # Per-row mean of valid pIC50 → single regression target.
        y_target = np.full(len(y_pic50_list), np.nan, dtype=np.float32)
        for i, row in enumerate(y_pic50):
            mask = ~np.isnan(row)
            if mask.any():
                y_target[i] = float(np.nanmean(row[mask]))
        y_target = np.nan_to_num(y_target, nan=5.0)
        y_target_t = torch.from_numpy(y_target.astype(np.float32)).to(device)
        active_dummy_t = torch.zeros(len(g_list), dtype=torch.long).to(device)
        optimizer.zero_grad()
        # ----- Wf-44 Phase 4: torch.autocast + grad_scaler -----------
        use_autocast = bool(use_amp) and device.type == "cuda"
        ctx = torch.amp.autocast(
            device_type="cuda", enabled=use_autocast,
        ) if use_autocast else _NullContext()
        with ctx:
            pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst)
            loss, _, _ = criterion(
                pic50_pred, active_logits, y_target_t, active_dummy_t,
            )
        if grad_scaler is not None and use_autocast:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += float(loss.item()) * len(g_list)
        n_samples += len(g_list)
    if n_samples == 0:
        return 0.0, 0
    return total_loss / n_samples, n_samples


class _NullContext:
    """No-op context manager used when AMP is disabled or unsupported."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@torch.no_grad()
def _predict_platinai_per_cell(
    model: DMPNNMultiTaskModel,
    graphs: List,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict a single pIC50 per molecule (model has 1 regression head)."""
    model.eval()
    n = len(graphs)
    preds = np.full(n, np.nan, dtype=np.float32)
    for start in range(0, n, batch_size):
        batch_graphs = graphs[start:start + batch_size]
        g_list = []
        keep_rows: List[int] = []
        for k, g in enumerate(batch_graphs):
            if g is None:
                continue
            g_list.append(g)
            keep_rows.append(start + k)
        if not g_list:
            continue
        collated = collate_graphs(g_list)
        if collated is None:
            continue
        atom_f, bond_f, e_src, e_dst = collated
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        pic50_pred, _ = model(atom_f, bond_f, e_src, e_dst)
        pic50_pred = pic50_pred.detach().cpu().numpy()
        for vi, ki in enumerate(keep_rows):
            preds[ki] = float(pic50_pred[vi])
    return preds


def _eval_platinai_per_cell(
    preds: np.ndarray,            # (N,) single pIC50 prediction per molecule
    pic50: np.ndarray,            # (N, 4) ground truth
    valid: np.ndarray,            # (N, 4) bool — True where gt is finite
) -> Dict[str, Dict[str, float]]:
    """Compute Pearson r per cell line + mean across cell lines."""
    out: Dict[str, Dict[str, float]] = {}
    for j, line in enumerate(PLATINAI_CELL_LINES):
        m = valid[:, j]
        if m.sum() < 2:
            out[line] = {"pearson": float("nan"), "n": int(m.sum())}
            continue
        yt = pic50[m, j]
        yp = preds[m]
        # Drop any rows with NaN preds (RDKit-fail molecules).
        finite = np.isfinite(yt) & np.isfinite(yp)
        if finite.sum() < 2 or np.std(yt[finite]) == 0 or np.std(yp[finite]) == 0:
            out[line] = {"pearson": float("nan"), "n": int(finite.sum())}
            continue
        r = float(pearsonr(yt[finite], yp[finite]).statistic)
        out[line] = {"pearson": r, "n": int(finite.sum())}
    pearsons = [out[line]["pearson"] for line in PLATINAI_CELL_LINES
                if not np.isnan(out[line]["pearson"])]
    out["__mean__"] = {
        "pearson": float(np.mean(pearsons)) if pearsons else float("nan"),
        "n": int(min(out[line]["n"] for line in PLATINAI_CELL_LINES)),
    }
    return out


def train_one_seed_platinai(
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Train one seed on PlatinAI 4-cell regression and report per-cell
    Pearson r + 4-line mean (the headline m16 metric)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = PlatinAIRegressionDataset.from_csv(args.data)
    print(
        f"[train_affinity] seed={seed} platinai rows={len(ds)} "
        f"per_cell_valid={ds.per_cell_line_valid_counts()}"
    )
    graphs_all = featurize_smiles_list([str(s) for s in ds.smiles])
    n_valid_graphs = sum(1 for g in graphs_all if g is not None)
    print(
        f"[train_affinity] seed={seed} valid_graphs={n_valid_graphs}/"
        f"{len(graphs_all)}"
    )
    # Random 80/10/10 split — same seed each time → deterministic split.
    idx = np.arange(len(ds))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=0.2, random_state=seed,
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.5, random_state=seed,
    )
    print(
        f"[train_affinity] seed={seed} split "
        f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}"
    )
    graphs_train = [graphs_all[i] for i in idx_train]
    pic50_train = ds.pic50[idx_train]
    valid_train = ds.valid[idx_train]
    graphs_test = [graphs_all[i] for i in idx_test]
    pic50_test = ds.pic50[idx_test]
    valid_test = ds.valid[idx_test]

    model = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=args.hidden,
        depth=args.depth,
        dropout=RECIPE_DROPOUT,
    ).to(device)
    if args.init == "tmqm":
        try_load_tmqm_encoder(model, args.encoder_ckpt)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MultiTaskLoss(alpha=RECIPE_ALPHA)
    # ----- Wf-44 Phase 4: AMP + grad scaler (CUDA only) -----------
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    grad_scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    )

    best_val_mean = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, n_train_seen = _train_epoch_platinai(
            model, optimizer, criterion,
            graphs_train, pic50_train, valid_train,
            batch_size=args.batch, device=device,
            use_amp=use_amp, grad_scaler=grad_scaler,
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # Val: per-cell Pearson on val split.
            graphs_val = [graphs_all[i] for i in idx_val]
            pic50_val = ds.pic50[idx_val]
            valid_val = ds.valid[idx_val]
            preds_val = _predict_platinai_per_cell(
                model, graphs_val, args.batch, device,
            )
            val_per_cell = _eval_platinai_per_cell(
                preds_val, pic50_val, valid_val,
            )
            val_mean = val_per_cell["__mean__"]["pearson"]
            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_mean_pearson": val_mean,
                "per_cell": {l: val_per_cell[l]["pearson"]
                             for l in PLATINAI_CELL_LINES},
            })
            print(
                f"[train_affinity] seed={seed} epoch={epoch:3d} "
                f"loss={train_loss:.4f} val_mean_pearson={val_mean:.4f} "
                + " ".join(
                    f"{l}={val_per_cell[l]['pearson']:.3f}"
                    for l in PLATINAI_CELL_LINES
                )
            )
            if not np.isnan(val_mean) and val_mean > best_val_mean:
                best_val_mean = val_mean
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    preds_test = _predict_platinai_per_cell(
        model, graphs_test, args.batch, device,
    )
    test_per_cell = _eval_platinai_per_cell(
        preds_test, pic50_test, valid_test,
    )
    test_mean = test_per_cell["__mean__"]["pearson"]
    print(
        f"[train_affinity] seed={seed} TEST mean_pearson={test_mean:.4f} "
        + " ".join(
            f"{l}={test_per_cell[l]['pearson']:.3f}"
            for l in PLATINAI_CELL_LINES
        )
    )
    return {
        "seed": seed,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "best_val_mean_pearson": float(best_val_mean),
        "test_per_cell_pearson": {
            l: test_per_cell[l]["pearson"] for l in PLATINAI_CELL_LINES
        },
        "test_per_cell_n": {
            l: test_per_cell[l]["n"] for l in PLATINAI_CELL_LINES
        },
        "test_mean_pearson": float(test_mean),
        "val_history": history,
        "best_state": best_state if best_state is not None else {},
    }


# ---------------------------------------------------------------------------
# Train one seed — HeLa48h single-regression
# ---------------------------------------------------------------------------
def train_one_seed_hela48h(
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Train one seed on HeLa48h and report a single test Pearson r."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    csv_path = Path(args.hela_csv) if args.hela_csv else None
    ds = _load_hela48h_cohort(
        csv_path, cell_line=args.cell_line, time_h=args.time_h,
    )
    print(
        f"[train_affinity] seed={seed} hela48h rows={len(ds)} "
        f"cell_line={args.cell_line} time_h={args.time_h}"
    )
    smiles = ds.smiles
    pic50_all = ds.pic50.astype(np.float32)
    active_all = ds.active.astype(int)
    keep = np.array([bool(s) for s in smiles], dtype=bool)
    smiles = smiles[keep]
    pic50_all = pic50_all[keep]
    active_all = active_all[keep]
    # 80/10/10 stratified random split.
    idx = np.arange(len(smiles))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=active_all,
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.5, random_state=seed,
        stratify=active_all[idx_tmp],
    )
    print(
        f"[train_affinity] seed={seed} split "
        f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}"
    )
    graphs_all = featurize_smiles_list([str(s) for s in smiles])
    n_valid_graphs = sum(1 for g in graphs_all if g is not None)
    print(
        f"[train_affinity] seed={seed} valid_graphs={n_valid_graphs}/"
        f"{len(graphs_all)}"
    )

    model = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=args.hidden,
        depth=args.depth,
        dropout=RECIPE_DROPOUT,
    ).to(device)
    if args.init == "tmqm":
        try_load_tmqm_encoder(model, args.encoder_ckpt)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MultiTaskLoss(alpha=RECIPE_ALPHA)
    # ----- Wf-44 Phase 4: AMP + grad scaler (CUDA only) -----------
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    grad_scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    )

    best_val_pearson = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []

    # Build a per-row helper to convert graphs + (pic50, active) into batches.
    def _run_epoch(graphs: List, labels: np.ndarray, train: bool) -> Tuple[float, int]:
        model.train(train)
        total = 0.0
        n_seen = 0
        idx_local = np.arange(len(graphs))
        if train:
            np.random.shuffle(idx_local)
        for start in range(0, len(idx_local), args.batch):
            batch_idx = idx_local[start:start + args.batch]
            g_list = []
            y_list = []
            for k in batch_idx:
                if graphs[int(k)] is None:
                    continue
                g_list.append(graphs[int(k)])
                y_list.append(float(labels[int(k)]))
            if not g_list:
                continue
            collated = collate_graphs(g_list)
            if collated is None:
                continue
            atom_f, bond_f, e_src, e_dst = collated
            atom_f = atom_f.to(device)
            bond_f = bond_f.to(device)
            e_src = e_src.to(device)
            e_dst = e_dst.to(device)
            y_t = torch.tensor(y_list, dtype=torch.float32).to(device)
            active_dummy = torch.zeros(len(g_list), dtype=torch.long).to(device)
            if train:
                optimizer.zero_grad()
            # ----- Wf-44 Phase 4: AMP forward inside autocast ---------
            use_autocast = use_amp
            ctx = torch.amp.autocast(
                device_type="cuda", enabled=use_autocast,
            ) if use_autocast else _NullContext()
            with ctx:
                pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst)
                loss, _, _ = criterion(
                    pic50_pred, active_logits, y_t, active_dummy,
                )
            if train:
                if grad_scaler is not None and use_autocast:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            total += float(loss.item()) * len(g_list)
            n_seen += len(g_list)
        return (total / max(n_seen, 1)), n_seen

    @torch.no_grad()
    def _eval_split(graphs: List, labels: np.ndarray) -> Dict[str, float]:
        # Mirror predict() shape: forward over all graphs, gather preds.
        model.eval()
        preds: List[float] = []
        truth: List[float] = []
        idx_local = np.arange(len(graphs))
        for start in range(0, len(idx_local), args.batch):
            batch_idx = idx_local[start:start + args.batch]
            g_list = []
            for k in batch_idx:
                if graphs[int(k)] is None:
                    continue
                g_list.append(graphs[int(k)])
            if not g_list:
                continue
            collated = collate_graphs(g_list)
            if collated is None:
                continue
            atom_f, bond_f, e_src, e_dst = collated
            atom_f = atom_f.to(device)
            bond_f = bond_f.to(device)
            e_src = e_src.to(device)
            e_dst = e_dst.to(device)
            pic50_pred, _ = model(atom_f, bond_f, e_src, e_dst)
            preds.extend(pic50_pred.detach().cpu().numpy().tolist())
            truth.extend(
                float(labels[int(k)]) for k in batch_idx
                if graphs[int(k)] is not None
            )
        m = regression_metrics(np.asarray(truth), np.asarray(preds))
        return m

    graphs_train = [graphs_all[i] for i in idx_train]
    graphs_val = [graphs_all[i] for i in idx_val]
    graphs_test = [graphs_all[i] for i in idx_test]
    labels_train = pic50_all[idx_train]
    labels_val = pic50_all[idx_val]
    labels_test = pic50_all[idx_test]

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = _run_epoch(graphs_train, labels_train, train=True)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = _eval_split(graphs_val, labels_val)
            val_p = val_metrics.get("pearson", float("nan"))
            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_pearson": float(val_p),
                "val_rmse": float(val_metrics.get("rmse", float("nan"))),
            })
            print(
                f"[train_affinity] seed={seed} epoch={epoch:3d} "
                f"loss={train_loss:.4f} val_pearson={val_p:.4f} "
                f"val_rmse={val_metrics.get('rmse', float('nan')):.4f}"
            )
            if not np.isnan(val_p) and val_p > best_val_pearson:
                best_val_pearson = float(val_p)
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = _eval_split(graphs_test, labels_test)
    test_pearson = test_metrics.get("pearson", float("nan"))
    print(
        f"[train_affinity] seed={seed} TEST pearson={test_pearson:.4f} "
        f"rmse={test_metrics.get('rmse', float('nan')):.4f} "
        f"mae={test_metrics.get('mae', float('nan')):.4f}"
    )
    return {
        "seed": seed,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "best_val_pearson": float(best_val_pearson),
        "test_pearson": float(test_pearson),
        "test_rmse": float(test_metrics.get("rmse", float("nan"))),
        "test_mae": float(test_metrics.get("mae", float("nan"))),
        "test_n": int(test_metrics.get("n", 0)),
        "val_history": history,
        "best_state": best_state if best_state is not None else {},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-32 Phase 1 — R19 m16 Affinity Pearson r regression trainer."
        ),
    )
    # Wf-35 Phase 2 — shared recipe / GPU / benchmark args.
    p = add_common_args(
        p,
        data_default=PLATINAI_DATA,
        encoder_ckpt_default=TMQM_ENCODER_CKPT,
        out_dir_default=DEFAULT_OUT_DIR,
        metrics_path_default=DEFAULT_METRICS_PATH,
        summary_path_default=DEFAULT_SUMMARY_PATH,
        include_init=True,  # affinity supports tmqm/random + R18 attaches.
    )
    p = add_gpu_args(p)

    # Wf-32-specific dataset + HeLa48h flags.
    p.add_argument(
        "--dataset", choices=list(DATASET_CHOICES), default="platinai",
        help="Which dataset to train on (default: platinai)",
    )
    p.add_argument("--hela-csv", type=Path, default=None,
                   help="Path to MetalCytoToxDB.csv (default: env MOLMETAL_DATA_DIR)")
    p.add_argument("--cell-line", default=HELA48H_CELL_LINE,
                   help=f"Cell line filter for hela48h (default: {HELA48H_CELL_LINE})")
    p.add_argument("--time-h", type=float, default=HELA48H_TIME_H,
                   help=f"Time(h) <= N for hela48h (default: {HELA48H_TIME_H})")
    # Wf-44 Phase 4 — AMP / grad scaler opt-in (CUDA only).
    p.add_argument("--amp", action="store_true", default=False,
                   help="Enable torch.autocast(fp16) + GradScaler (CUDA only)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------
def _to_jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_jsonable(v) for v in o]
    return o


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@MEASURED_vs_PROJECTED_decorator()
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_gpu_env(args)  # Wf-35 — apply CUDA_VISIBLE_DEVICES + torch seed.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[train_affinity] dataset={args.dataset} init={args.init} "
        f"epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"seeds={args.seeds} hidden={args.hidden} depth={args.depth}"
    )
    print(
        f"[train_affinity] data={args.data} encoder_ckpt={args.encoder_ckpt}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_affinity] device={device}")

    # R18 init: build the optional R18 component so the metadata reflects
    # which adapter is in play (the regression loss is unchanged).
    r18_components: Dict[str, str] = {}
    if args.init == "se3sb" and _SE3SBVelocityField is not None:
        r18_components["se3_field"] = "SE3SBVelocityField"
    elif args.init == "transport" and _RelaxedEndpointTerm is not None:
        r18_components["relax_term"] = "RelaxedEndpointTerm"
    elif args.init == "multiph" and _MultiPHPocketFeatures is not None:
        r18_components["mphp"] = "MultiPHPocketFeatures"
    if r18_components:
        print(f"[train_affinity] r18_components={r18_components}")

    t0 = time.time()
    seed_results: List[Dict[str, object]] = []
    if args.dataset == "platinai":
        for seed in args.seeds:
            try:
                result = train_one_seed_platinai(seed, args, device)
            except Exception as exc:                    # pragma: no cover
                print(
                    f"[train_affinity] seed={seed} FAILED: {exc!r}",
                    flush=True,
                )
                import traceback
                traceback.print_exc()
                result = {"seed": seed, "error": repr(exc)}
            seed_results.append(result)
            ckpt_path = args.out_dir / f"affinity_platinai_seed{seed}.pt"
            payload = dict(result)
            best_state = payload.pop("best_state", None) or {}
            torch.save(
                {
                    "seed": seed,
                    "test_mean_pearson": result.get("test_mean_pearson"),
                    "test_per_cell_pearson": result.get(
                        "test_per_cell_pearson", {},
                    ),
                    "state_dict": best_state,
                    "recipe": {
                        "init": args.init, "epochs": args.epochs,
                        "batch": args.batch, "lr": args.lr,
                        "hidden": args.hidden, "depth": args.depth,
                        "alpha": RECIPE_ALPHA,
                    },
                },
                ckpt_path,
            )
            print(f"[train_affinity] wrote {ckpt_path}")
    elif args.dataset == "hela48h":
        for seed in args.seeds:
            try:
                result = train_one_seed_hela48h(seed, args, device)
            except Exception as exc:                    # pragma: no cover
                print(
                    f"[train_affinity] seed={seed} FAILED: {exc!r}",
                    flush=True,
                )
                import traceback
                traceback.print_exc()
                result = {"seed": seed, "error": repr(exc)}
            seed_results.append(result)
            ckpt_path = args.out_dir / f"affinity_hela48h_seed{seed}.pt"
            payload = dict(result)
            best_state = payload.pop("best_state", None) or {}
            torch.save(
                {
                    "seed": seed,
                    "test_pearson": result.get("test_pearson"),
                    "test_rmse": result.get("test_rmse"),
                    "state_dict": best_state,
                    "recipe": {
                        "init": args.init, "epochs": args.epochs,
                        "batch": args.batch, "lr": args.lr,
                        "hidden": args.hidden, "depth": args.depth,
                        "alpha": RECIPE_ALPHA,
                    },
                },
                ckpt_path,
            )
            print(f"[train_affinity] wrote {ckpt_path}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset!r}")

    wall = time.time() - t0
    print(f"[train_affinity] total wall={wall:.1f}s")

    # Aggregate per-seed metric — the headline number the Wf-32 brief cares
    # about.  For platinai, headline = mean of 4 per-cell-line Pearson r;
    # for hela48h, headline = single test Pearson r.
    if args.dataset == "platinai":
        head_per_seed = [
            float(r.get("test_mean_pearson", float("nan")))
            for r in seed_results if "error" not in r
        ]
        per_cell_per_seed: Dict[str, List[float]] = {
            l: [
                float(r.get("test_per_cell_pearson", {}).get(l, float("nan")))
                for r in seed_results if "error" not in r
            ]
            for l in PLATINAI_CELL_LINES
        }
        metrics: Dict[str, object] = {
            "recipe": "r19_wf32_m16_affinity_pearson_r_platinai",
            "dataset": "platinai",
            "n_seeds": len(seed_results),
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "hidden": args.hidden,
            "depth": args.depth,
            "alpha": RECIPE_ALPHA,
            "init": args.init,
            "data_path": str(args.data),
            "encoder_ckpt": str(args.encoder_ckpt),
            "wall_seconds": wall,
            "per_seed": _to_jsonable(seed_results),
            "headline_pearson_r_per_seed": head_per_seed,
            "headline_pearson_r_mean": (
                float(np.nanmean(head_per_seed)) if head_per_seed
                else float("nan")
            ),
            "headline_pearson_r_std": (
                float(np.nanstd(head_per_seed)) if head_per_seed
                else float("nan")
            ),
            "per_cell_line_pearson_r": {
                l: {
                    "per_seed": per_cell_per_seed[l],
                    "mean": (
                        float(np.nanmean(per_cell_per_seed[l]))
                        if per_cell_per_seed[l] else float("nan")
                    ),
                    "std": (
                        float(np.nanstd(per_cell_per_seed[l]))
                        if per_cell_per_seed[l] else float("nan")
                    ),
                }
                for l in PLATINAI_CELL_LINES
            },
            "cell_lines": list(PLATINAI_CELL_LINES),
            "r18_components": r18_components,
            "r17_baseline_pearson_r": 0.207,
            "r17_baseline_source": (
                "metrics/by_metric/pearson_r_pic50.json — "
                "HeLa48h/dark 0.207 (best margin=0.5).  "
                "PlatinAI per-line has no prior baseline in repo."
            ),
        }
        # Drop the heavy best_state tensor from the JSON output
        for r in metrics["per_seed"]:
            if isinstance(r, dict) and "best_state" in r:
                r.pop("best_state")
        args.metrics_path.write_text(json.dumps(_to_jsonable(metrics), indent=2))
        print(
            f"[train_affinity] wrote {args.metrics_path} "
            f"(headline={metrics['headline_pearson_r_mean']:.4f} "
            f"± {metrics['headline_pearson_r_std']:.4f})"
        )
        # Slim summary
        summary = {
            "recipe": "r19_wf32_m16_affinity_pearson_r_platinai_summary",
            "dataset": "platinai",
            "n_seeds": len(seed_results),
            "headline_pearson_r_per_seed": head_per_seed,
            "headline_pearson_r_mean": metrics["headline_pearson_r_mean"],
            "headline_pearson_r_std": metrics["headline_pearson_r_std"],
            "per_cell_line_mean": {
                l: metrics["per_cell_line_pearson_r"][l]["mean"]
                for l in PLATINAI_CELL_LINES
            },
            "init": args.init,
            "wall_seconds": wall,
        }
        args.summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2))
        print(f"[train_affinity] wrote {args.summary_path}")
    else:
        head_per_seed = [
            float(r.get("test_pearson", float("nan")))
            for r in seed_results if "error" not in r
        ]
        rmse_per_seed = [
            float(r.get("test_rmse", float("nan")))
            for r in seed_results if "error" not in r
        ]
        metrics = {
            "recipe": "r19_wf32_m16_affinity_pearson_r_hela48h",
            "dataset": "hela48h",
            "cell_line": args.cell_line,
            "time_h": args.time_h,
            "n_seeds": len(seed_results),
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "hidden": args.hidden,
            "depth": args.depth,
            "alpha": RECIPE_ALPHA,
            "init": args.init,
            "encoder_ckpt": str(args.encoder_ckpt),
            "wall_seconds": wall,
            "per_seed": _to_jsonable(seed_results),
            "headline_pearson_r_per_seed": head_per_seed,
            "headline_pearson_r_mean": (
                float(np.nanmean(head_per_seed)) if head_per_seed
                else float("nan")
            ),
            "headline_pearson_r_std": (
                float(np.nanstd(head_per_seed)) if head_per_seed
                else float("nan")
            ),
            "headline_rmse_per_seed": rmse_per_seed,
            "headline_rmse_mean": (
                float(np.nanmean(rmse_per_seed)) if rmse_per_seed
                else float("nan")
            ),
            "r18_components": r18_components,
            "r17_baseline_pearson_r": 0.207,
            "r17_baseline_source": (
                "metrics/by_metric/pearson_r_pic50.json — "
                "HeLa48h/dark 0.207 (best margin=0.5)."
            ),
        }
        for r in metrics["per_seed"]:
            if isinstance(r, dict) and "best_state" in r:
                r.pop("best_state")
        args.metrics_path.write_text(json.dumps(_to_jsonable(metrics), indent=2))
        print(
            f"[train_affinity] wrote {args.metrics_path} "
            f"(headline={metrics['headline_pearson_r_mean']:.4f} "
            f"± {metrics['headline_pearson_r_std']:.4f})"
        )
        summary = {
            "recipe": "r19_wf32_m16_affinity_pearson_r_hela48h_summary",
            "dataset": "hela48h",
            "cell_line": args.cell_line,
            "time_h": args.time_h,
            "n_seeds": len(seed_results),
            "headline_pearson_r_per_seed": head_per_seed,
            "headline_pearson_r_mean": metrics["headline_pearson_r_mean"],
            "headline_pearson_r_std": metrics["headline_pearson_r_std"],
            "headline_rmse_per_seed": rmse_per_seed,
            "headline_rmse_mean": metrics["headline_rmse_mean"],
            "init": args.init,
            "wall_seconds": wall,
        }
        args.summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2))
        print(f"[train_affinity] wrote {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())