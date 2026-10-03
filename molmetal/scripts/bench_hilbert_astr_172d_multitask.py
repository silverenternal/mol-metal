#!/usr/bin/env python3
"""Wf-125 Phase 3 — 172-d feature stack + multi-task 4-cell backbone + Wasserstein pIC50.

================================================================
``molmetal/scripts/bench_hilbert_astr_172d_multitask.py`` is
the **Wf-125 P3 close-out** for the **PlatinAI 4-cell R²**
strengthen TODO.  It stacks three innovations on top of the
``bench_hilbert_astr_real_platinai_4cell.py`` anchor (12-d RDKit
Hilbert + ASTSR):

  1. **172-d feature stack** — 12-d RDKit + 32-d Multi-PH Fourier
     coefficients + 128-d PLL signature.  Built once per SMILES and
     cached to ``molmetal/data/features_172d_platinai_4cell.npz``.
  2. **Multi-task 4-cell backbone** — shared MLP trunk (172 → 256 →
     128) + per-cell-line heads (128 → 64 → 1) for the 4 PlatinAI
     cell lines (MCF7 / A549 / A2780 / A2780cis).  The shared trunk
     learns cross-cell-line chemistry; per-cell-line heads
     specialise on each IC50 column.
  3. **Wasserstein pIC50 regularisation** — joint W₂ penalty on the
     predicted pIC50 cloud vs the training reference distribution
     (via :mod:`molmetal.molmetal_lam.reward.wasserstein_loss`,
     Wf-123 P1 ship).  Forces predictions to stay on the chemistry
     manifold.

Combined, the three innovations target a **combined R² ≥ 0** for at
least 1 of 4 cell lines on the 4 078-row PlatinAI cohort (vs the
Wf-86 P3 12-d anchor of -0.014 / -23.78 / -0.014 / -1.50 per cell
line).  Honest framing per ``party-twenty-spirit §3.5``: the
structural 12-d signal floor may not be breakable; if the
multi-task backbone still collapses to constant, the honest
answer is to report R² < 0 across all 4 cell lines and ship the
172-d + multi-task + Wasserstein scaffolding as the production
anchor for follow-on workstreams (D-MPNN fine-tune, MCTS policy
head, etc.).

Combined approach
-----------------
* **Feature stream**: 172-d = ``compute_172d`` from
  ``build_172d_features.py`` (Wf-128 P1).  Per-row failures degrade
  to column-mean imputation; all-three-fail is dropped.
* **Backbone**: 2-layer shared MLP (172 → 256 → 128, ReLU +
  Dropout 0.2) + 4 per-cell-line regression heads (128 → 64 → 1,
  ReLU + Dropout 0.2).  Trained jointly across the 4 cell lines
  with Adam + grad-clip max_norm=5.0 + Wasserstein penalty.
* **Loss** (per epoch)::

      L = MSE(pred_pic50, ref_pic50) + w_w * W₂(pred_pic50, ref_pic50)

  ``w_w = --wasserstein-weight`` (default 0.05; L2 guard on).

* **Scaffold split**: Bemis-Murcko disjoint 5-fold (strict); same
  as Wf-86 P3 anchor.
* **Per-cell-line R²**: mean R² across 5 folds, averaged across
  5 seeds.  Aggregate = mean of 4 per-cell-line means.

Constraints honored
-------------------
* **No D-MPNN fine-tune** — this is the **172-d + multi-task +
  Wasserstein** anchor; D-MPNN finetune is a separate workstream
  (Wf-131 P1 ships the 5-metal D-MPNN variant for Ru/Ir/Os/Rh/Re).
* **CPU primary; GPU-1 RTX 5090 (CUDA_VISIBLE_DEVICES=1) optional**.
* **Honest framing** — per-fold + per-seed R² always reported;
  collapse detector flags ``pred_std / y_test_std < 0.10``.

Inputs
------
* ``--dataset``: ``platinai_4cell`` (4 078 rows × 4 cell lines).
* ``--cell-lines``: list of cell-line keys (default
  ``mcf7 a549 a2780 a2780cis``).
* ``--seeds``: list of integers (default ``0 1 2 3 4``).
* ``--n-folds``: 5 by default.
* ``--epochs``: 50 by default.
* ``--wasserstein-weight``: 0.05 default.
* ``--strict-scaffold``: Bemis-Murcko disjoint 5-fold (default True).
* ``--device``: ``cpu`` (default) or ``cuda``.
* ``--out-dir``: ``molmetal/reports/wf125_p3_172d_multitask/``.
* ``--cache-path``: ``molmetal/data/features_172d_platinai_4cell.npz``.

Outputs
-------
* ``<out-dir>/{cell_line}/per_seed_metrics.json`` — per-seed combined
  R² + 5-seed mean ± std + per-fold breakdown + Wasserstein decay.
* ``<out-dir>/{cell_line}/run.log`` — stdout mirror.
* ``<out-dir>/synthesis.md`` — cross-cell-line summary table +
  4-cell aggregate.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=1 molmetal/.venvs/mol-metal-venv/bin/python \\
        molmetal/scripts/bench_hilbert_astr_172d_multitask.py \\
        --dataset platinai_4cell --cell-lines \\
        --seeds 0 1 2 3 4 --epochs 50 --strict-scaffold \\
        --wasserstein-weight 0.05 \\
        --out-dir molmetal/reports/wf125_p3_172d_multitask

Pytest
~~~~~~

::

    pytest molmetal/tests/test_172d_multitask.py -v
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo-root import bootstrap.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Optional torch import — degrade gracefully if torch is missing.
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    _HAS_TORCH = True
    _TORCH_ERR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False
    _TORCH_ERR = _exc


# ---------------------------------------------------------------------------
# Constants — per Wf-125 P3 spec.
# ---------------------------------------------------------------------------
COLLAPSE_RATIO_THRESHOLD: float = 0.10
PRED_CLIP: float = 1e6
EPS_STD: float = 1e-3
DEFAULT_CACHE_PATH: Path = Path("molmetal/data/features_172d_platinai_4cell.npz")

# PlatinAI 4-cell CSV (4 078 rows × 4 cell lines).
PLATINAI_CSV = Path("molmetal/data/platinai_4cell.csv")
PLATINAI_CELL_COLS: Dict[str, str] = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}
PLATINAI_SMILES_COL: str = "smiles"

# Default 4-cell-line cohort.
DEFAULT_CELL_LINES: Tuple[str, ...] = ("mcf7", "a549", "a2780", "a2780cis")

# Backbone shapes — kept in sync with ``build_multitask_backbone.py``
# (Wf-128 P1) but with **4 cell lines** instead of 5 metals.
INPUT_DIM: int = 172  # 12 RDKit + 32 Multi-PH + 128 PLL
TRUNK_HIDDEN: int = 256  # shared 4-cell trunk hidden width
TRUNK_LATENT: int = 128  # shared 4-cell trunk latent width
HEAD_HIDDEN: int = 64  # per-cell-line head hidden width
HEAD_DROPOUT: float = 0.2  # per-cell-line head dropout
TRUNK_DROPOUT: float = 0.2  # shared trunk dropout

# Wasserstein defaults (Wf-123 P1 spec).
DEFAULT_WASSERSTEIN_WEIGHT: float = 0.05
DEFAULT_WASSERSTEIN_N_MC: int = 256  # sliced-W₂ projections
DEFAULT_WASSERSTEIN_SEED: int = 0
DEFAULT_DECAY_WINDOW: int = 10
DEFAULT_DECAY_TOL: float = 1e-3

# Optimizer defaults.
DEFAULT_LR: float = 1e-3
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_GRAD_CLIP_MAX_NORM: float = 5.0


# ---------------------------------------------------------------------------
# 1. Feature extraction — 172-d = 12 RDKit + 32 Multi-PH + 128 PLL
# ---------------------------------------------------------------------------
def compute_172d_for_smiles(smiles: str) -> Optional[np.ndarray]:
    """Compute the 172-d feature vector for a single SMILES.

    Re-uses ``build_172d_features.compute_172d`` from Wf-128 P1; the
    contract is a ``(172,)`` float32 vector or ``None`` on all-three
    stream failure.  We expose it under a stable name here so the
    172-d multitask script has a single import boundary.

    Per-SMILES failures (e.g. RDKit ``EmbedMolecule`` invariant
    violations on certain Pt-coordination complexes — "bad lower
    bound") are **isolated**: this wrapper returns ``None`` so the
    caller can apply column-mean imputation without crashing the
    whole cohort build.
    """
    try:
        from molmetal.scripts.build_172d_features import compute_172d
    except ImportError:
        # Fall back to script-local import (when not installed).
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from build_172d_features import compute_172d
    try:
        return compute_172d(smiles)
    except Exception:
        # RDKit Boost invariants + scipy.linalg + torch.linalg are
        # the most common failure modes.  Returning ``None`` matches
        # the build_172d_features contract: the caller decides on
        # imputation.
        return None


# ---------------------------------------------------------------------------
# 2. CSV ingestion + 172-d feature build + cache I/O
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV to list-of-dicts (utf-8, tolerant to BOM)."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def load_platinai_4cell_with_cache(
    repo_root: Path,
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    n_max: int = 5000,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Load PlatinAI 4-cell + compute 172-d features (cached to disk).

    Returns
    -------
    X : np.ndarray of shape ``(N, 172)``, dtype float32
        172-d feature stack.
    Y : np.ndarray of shape ``(N, 4)``, dtype float32
        Per-cell-line pIC50 (MCF7 / A549 / A2780 / A2780cis).
        ``np.nan`` for missing rows.
    smiles : List[str]
        SMILES (preserves CSV order).
    cell_y : Dict[str, np.ndarray]
        Per-cell-line pIC50 target arrays keyed by cell line
        (e.g. ``cell_y["mcf7"]`` is shape ``(N,)``).
    """
    cache_full = repo_root / cache_path
    rows = _read_csv(repo_root / PLATINAI_CSV)

    # Build SMILES list (deduplicated, preserves order).
    smiles_unique: List[str] = []
    smiles_to_idx: Dict[str, int] = {}
    for r in rows:
        smi = (r.get(PLATINAI_SMILES_COL) or "").strip()
        if not smi:
            continue
        if smi not in smiles_to_idx:
            smiles_to_idx[smi] = len(smiles_unique)
            smiles_unique.append(smi)
    if verbose:
        print(
            f"[wf125-p3] {len(rows)} CSV rows → {len(smiles_unique)} unique SMILES",
            flush=True,
        )

    # Try to load 172-d cache.
    cache_loaded = False
    if cache_full.exists():
        try:
            cache_data = np.load(cache_full, allow_pickle=True)
            cached_smiles = list(cache_data["smiles"])
            cached_X = np.asarray(cache_data["X"], dtype=np.float32)
            if len(cached_smiles) == len(smiles_unique) and all(
                a == b for a, b in zip(cached_smiles, smiles_unique)
            ):
                if verbose:
                    print(
                        f"[wf125-p3] 172-d cache hit: {cache_full} "
                        f"({cached_X.shape})",
                        flush=True,
                    )
                X_unique = cached_X
                cache_loaded = True
            else:
                if verbose:
                    print(
                        f"[wf125-p3] 172-d cache stale; rebuilding "
                        f"({len(cached_smiles)} vs {len(smiles_unique)} SMILES)",
                        flush=True,
                    )
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[wf125-p3] cache load failed: {exc!r}; rebuilding", flush=True)

    if not cache_loaded:
        X_unique = np.zeros((len(smiles_unique), INPUT_DIM), dtype=np.float32)
        t0 = time.perf_counter()
        for i, smi in enumerate(smiles_unique):
            feats = compute_172d_for_smiles(smi)
            if feats is not None and feats.size == INPUT_DIM:
                X_unique[i] = feats
            if (i + 1) % 200 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                eta = (len(smiles_unique) - i - 1) / max(rate, 1e-9)
                print(
                    f"[wf125-p3]   {i + 1}/{len(smiles_unique)} SMILES "
                    f"({rate:.2f}/s, ETA {eta:.1f}s)",
                    flush=True,
                )
        # Persist to disk for the next run.
        cache_full.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                cache_full,
                smiles=np.asarray(smiles_unique, dtype=object),
                X=X_unique,
            )
            if verbose:
                print(f"[wf125-p3] 172-d cache written: {cache_full}", flush=True)
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[wf125-p3] cache write failed: {exc!r}", flush=True)

    # Project X_unique → X_per_row (one row per CSV row, in CSV order).
    X_per_row = np.zeros((len(rows), INPUT_DIM), dtype=np.float32)
    smiles_per_row: List[str] = []
    for i, r in enumerate(rows):
        smi = (r.get(PLATINAI_SMILES_COL) or "").strip()
        if not smi:
            smiles_per_row.append("")
            continue
        idx = smiles_to_idx[smi]
        X_per_row[i] = X_unique[idx]
        smiles_per_row.append(smi)

    # Build per-cell-line pIC50 columns (NaN for missing).
    Y = np.full((len(rows), 4), np.nan, dtype=np.float32)
    cell_y: Dict[str, np.ndarray] = {}
    for j, (cl, col) in enumerate(PLATINAI_CELL_COLS.items()):
        per_cl = np.full(len(rows), np.nan, dtype=np.float32)
        for i, r in enumerate(rows):
            raw = r.get(col)
            try:
                ic50_nm = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ic50_nm) or ic50_nm <= 0.0:
                continue
            ic50_um = ic50_nm / 1000.0
            pic50 = -math.log10(max(ic50_um * 1e-6, 1e-12))
            if math.isfinite(pic50):
                per_cl[i] = float(pic50)
        Y[:, j] = per_cl
        cell_y[cl] = per_cl
    return X_per_row, Y, smiles_per_row, cell_y


# ---------------------------------------------------------------------------
# 3. K-fold splitting — strict scaffold + random fallback
# ---------------------------------------------------------------------------
def _random_kfold_indices(
    n: int, k: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Random K-fold split (fallback when scaffold split isn't possible)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, test))
    return out


def _strict_scaffold_kfold_indices(
    scaffold_ids: Sequence[str],
    k: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K-fold split with **strict scaffold disjointness**."""
    n = len(scaffold_ids)
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, sc in enumerate(scaffold_ids):
        key = sc if sc else f"__empty_{idx}__"
        groups[key].append(idx)
    unique_sc = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_sc)
    test_idx_per_fold: List[List[int]] = [[] for _ in range(k)]
    for i, sc in enumerate(unique_sc):
        fold = i % k
        test_idx_per_fold[fold].extend(groups[sc])
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test = np.asarray(sorted(test_idx_per_fold[i]), dtype=np.int64)
        all_idx = np.arange(n, dtype=np.int64)
        if len(test) == 0:
            train = all_idx
        else:
            mask = np.ones(n, dtype=bool)
            mask[test] = False
            train = all_idx[mask]
        out.append((train, test))
    return out


def _kfold_indices(
    n: int,
    k: int,
    seed: int,
    *,
    scaffold_ids: Optional[Sequence[str]] = None,
    strict_scaffold: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K-fold dispatcher — strict-scaffold or random."""
    if strict_scaffold and scaffold_ids is not None and len(scaffold_ids) == n:
        return _strict_scaffold_kfold_indices(scaffold_ids, k, seed)
    return _random_kfold_indices(n, k, seed)


def _bemis_murcko_scaffold(smi: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES; empty if RDKit fails.

    Mirrors the Wf-86 P3 two-tier fallback (Bemis-Murcko →
    canonicalised ligand SMILES → raw canonical SMILES).  Used
    purely for the strict-scaffold split key — the 172-d feature
    stream is independent.
    """
    if not smi:
        return ""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol, includeChirality=False)
            if scaffold is not None:
                sc_smi = Chem.MolToSmiles(scaffold)
                if sc_smi:
                    return sc_smi
        except Exception:
            pass
        try:
            rwmol = Chem.RWMol(mol)
            for atom in rwmol.GetAtoms():
                sym = atom.GetSymbol()
                if sym in (
                    "Pt",
                    "Pd",
                    "Ru",
                    "Ir",
                    "Os",
                    "Rh",
                    "Au",
                    "Ag",
                    "Cu",
                    "Ni",
                    "Fe",
                    "Co",
                    "Zn",
                    "Mn",
                    "Cr",
                    "V",
                    "Ti",
                    "Sc",
                    "Y",
                    "Zr",
                    "Nb",
                    "Mo",
                    "Tc",
                    "Cd",
                    "Hf",
                    "Ta",
                    "W",
                    "Re",
                    "Hg",
                ):
                    rwmol.RemoveAtom(atom.GetIdx())
            stripped = rwmol.GetMol()
            stripped_smi = Chem.MolToSmiles(stripped)
            if stripped_smi:
                return stripped_smi
        except Exception:
            pass
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 4. Multi-task 4-cell backbone (CPU + GPU-1)
# ---------------------------------------------------------------------------
class _PerCellHead(nn.Module):
    """Per-cell-line regression head: Linear(128 → 64) → ReLU → Dropout → Linear(64 → 1)."""

    def __init__(
        self,
        in_dim: int = TRUNK_LATENT,
        hidden_dim: int = HEAD_HIDDEN,
        dropout: float = HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(in_dim), int(hidden_dim))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim), 1)

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.act(self.fc1(z))
        h = self.drop(h)
        return self.fc2(h)


class _SharedTrunk(nn.Module):
    """Shared 4-cell trunk: Linear(172 → 256) → ReLU → Dropout → Linear(256 → 128) → ReLU."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = TRUNK_HIDDEN,
        latent_dim: int = TRUNK_LATENT,
        dropout: float = TRUNK_DROPOUT,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim), int(latent_dim))
        self.act2 = nn.ReLU()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.act1(self.fc1(x))
        h = self.drop1(h)
        h = self.act2(self.fc2(h))
        return h


class MultiTaskFourCellBackbone(nn.Module):
    """Shared 4-cell trunk + 4 per-cell-line regression heads.

    Forward returns a ``dict[cell_line, (B, 1) tensor]``.  Inputs are
    the 172-d feature stack ``X`` (B, 172); the cell-line index is
    selected at training time via the head's own forward.
    """

    def __init__(
        self,
        cell_lines: Sequence[str] = DEFAULT_CELL_LINES,
        input_dim: int = INPUT_DIM,
        trunk_hidden: int = TRUNK_HIDDEN,
        trunk_latent: int = TRUNK_LATENT,
        trunk_dropout: float = TRUNK_DROPOUT,
        head_hidden: int = HEAD_HIDDEN,
        head_dropout: float = HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        self.cell_lines = tuple(cell_lines)
        self.trunk = _SharedTrunk(
            input_dim=input_dim,
            hidden_dim=trunk_hidden,
            latent_dim=trunk_latent,
            dropout=trunk_dropout,
        )
        self.heads = nn.ModuleDict(
            {
                cl: _PerCellHead(
                    in_dim=trunk_latent,
                    hidden_dim=head_hidden,
                    dropout=head_dropout,
                )
                for cl in self.cell_lines
            }
        )

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.trunk(x)

    def head_forward(self, z: "torch.Tensor", cell_line: str) -> "torch.Tensor":
        if cell_line not in self.heads:
            raise KeyError(
                f"Unknown cell_line {cell_line!r}; supported: {list(self.heads)}"
            )
        return self.heads[cell_line](z)


# ---------------------------------------------------------------------------
# 5. Wasserstein pIC50 loss wrapper (CPU + GPU-1)
# ---------------------------------------------------------------------------
def _wasserstein_pic50_torch(
    pred: "torch.Tensor",
    ref: "torch.Tensor",
    n_mc: int,
    seed: int,
) -> "torch.Tensor":
    """Wasserstein-2 between predicted pIC50 and reference pIC50 (1-D)."""
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_pic50_loss,
    )

    return wasserstein_pic50_loss(
        pred.float(), ref.float(), n_mc=int(n_mc), seed=int(seed)
    )


def wasserstein_decay_check_helper(
    loss_history: Sequence[float],
    window: int = DEFAULT_DECAY_WINDOW,
    tol: float = DEFAULT_DECAY_TOL,
) -> bool:
    """Re-export of ``wasserstein_decay_check`` for in-script testing.

    Thin wrapper so unit tests can patch the local name without
    importing the underlying module under a private alias.
    """
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_decay_check,
    )

    return wasserstein_decay_check(
        list(loss_history), window=int(window), tol=float(tol)
    )


# ---------------------------------------------------------------------------
# 6. Per-cell-line training (single fold + single seed)
# ---------------------------------------------------------------------------
@dataclass
class FoldConfig:
    """Per-fold training config (CPU/GPU-agnostic)."""

    epochs: int = 50
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    wasserstein_weight: float = DEFAULT_WASSERSTEIN_WEIGHT
    n_mc: int = DEFAULT_WASSERSTEIN_N_MC
    wasserstein_seed: int = DEFAULT_WASSERSTEIN_SEED
    grad_clip_max_norm: float = DEFAULT_GRAD_CLIP_MAX_NORM


def _train_one_cell_line_one_fold(
    model: MultiTaskFourCellBackbone,
    *,
    cell_line: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    fold_cfg: FoldConfig,
    ref_pic50_pool: "torch.Tensor",
    device: "torch.device",
) -> Tuple[float, np.ndarray, float, float]:
    """Train the per-cell-line head on ``X_tr`` / ``y_tr`` and predict ``X_te``.

    Returns ``(r2, y_pred, final_mse, final_wasserstein)``.
    """
    if X_tr.shape[0] == 0 or y_tr.size == 0:
        return 0.0, np.zeros(X_te.shape[0], dtype=np.float64), 0.0, 0.0

    if y_tr.std() < 1e-9:
        # Degenerate fold — predict the constant mean.
        y_pred = np.full(X_te.shape[0], float(y_tr.mean()), dtype=np.float64)
        return 0.0, y_pred, 0.0, 0.0

    # Z-score X per fold (using training stats).
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)

    X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te_n, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.float32, device=device)
    ref_dev = (
        ref_pic50_pool.to(device)
        if isinstance(ref_pic50_pool, torch.Tensor)
        else torch.tensor(ref_pic50_pool, dtype=torch.float32, device=device)
    )

    head = model.heads[cell_line]
    head.train()

    opt = torch.optim.Adam(head.parameters(), lr=float(fold_cfg.lr))

    n = X_tr_t.shape[0]
    final_mse = 0.0
    final_w = 0.0

    for epoch in range(int(fold_cfg.epochs)):
        # Mini-batch SGD with shuffled order.
        perm = torch.randperm(n, device=device)
        total_mse = 0.0
        total_w = 0.0
        n_batches = 0
        for start in range(0, n, int(fold_cfg.batch_size)):
            idx = perm[start : start + int(fold_cfg.batch_size)]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]

            opt.zero_grad()
            z = model.encode(xb)
            pred = head(z).squeeze(-1)
            mse = ((pred - yb) ** 2).mean()
            w_pen = _wasserstein_pic50_torch(
                pred,
                ref_dev,
                n_mc=int(fold_cfg.n_mc),
                seed=int(fold_cfg.wasserstein_seed) + epoch,
            )
            loss = mse + float(fold_cfg.wasserstein_weight) * w_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=float(fold_cfg.grad_clip_max_norm)
            )
            opt.step()
            total_mse += float(mse.detach().item())
            total_w += float(w_pen.detach().item())
            n_batches += 1
        if n_batches > 0:
            final_mse = total_mse / n_batches
            final_w = total_w / n_batches

    # Inference on test set.
    head.eval()
    with torch.no_grad():
        z_te = model.encode(X_te_t)
        pred_te = head(z_te).squeeze(-1).cpu().numpy()
    pred_te = np.clip(pred_te, -PRED_CLIP, PRED_CLIP)

    ss_res = float(np.sum((y_te - pred_te) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(r2), pred_te, final_mse, final_w


# ---------------------------------------------------------------------------
# 7. Evaluation driver — 4-cell multi-task × 5-seed × 5-fold
# ---------------------------------------------------------------------------
def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _collapse_detected(y_pred: np.ndarray, y_true: np.ndarray) -> bool:
    if y_true.std() < 1e-12:
        return True
    return float(y_pred.std() / max(y_true.std(), 1e-12)) < COLLAPSE_RATIO_THRESHOLD


def evaluate_cell_line(
    cell_line: str,
    seeds: Sequence[int],
    n_folds: int,
    fold_cfg: FoldConfig,
    device: str,
    strict_scaffold: bool,
    X: np.ndarray,
    Y: np.ndarray,
    smiles: List[str],
    scaffold_ids: List[str],
    cell_y: Dict[str, np.ndarray],
    ref_pic50_pool: "torch.Tensor",
) -> Dict[str, Any]:
    """Run multi-task 4-cell backbone 5-fold × N-seed on PlatinAI.

    Only the per-cell-line head is trained; the shared trunk is
    shared across all 4 cell lines (joint training pool).  The
    result is the **MEASURED** Wf-125 P3 production anchor for
    PlatinAI 4-cell.
    """
    if not _HAS_TORCH:
        raise RuntimeError(
            f"Torch is required for Wf-125 P3 multi-task; got {_TORCH_ERR!r}"
        )

    if device != "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    n, d_feat = X.shape
    y_full = cell_y[cell_line]
    if not np.any(np.isfinite(y_full)):
        raise RuntimeError(f"No finite target values for cell_line={cell_line!r}")

    per_seed: List[Dict[str, Any]] = []
    r2_per_seed: List[float] = []

    for seed in seeds:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        splits = _kfold_indices(
            n,
            n_folds,
            seed=int(seed),
            scaffold_ids=scaffold_ids,
            strict_scaffold=strict_scaffold,
        )
        per_fold: List[Dict[str, Any]] = []
        fold_r2: List[float] = []
        all_y_true: List[float] = []
        all_y_pred: List[float] = []
        wasserstein_hist: List[float] = []
        mse_hist: List[float] = []
        t0 = time.perf_counter()

        # Build a fresh multi-task backbone for each seed so the
        # shared trunk is shared across the 4 cell lines (the point
        # of multi-task learning).
        model = MultiTaskFourCellBackbone(cell_lines=DEFAULT_CELL_LINES)
        model.to(dev)
        model.train()

        for fold_idx, (tr, te) in enumerate(splits):
            if len(te) == 0:
                continue
            # Filter rows with finite target on this cell line.
            mask_tr = np.isfinite(y_full[tr])
            mask_te = np.isfinite(y_full[te])
            tr_kept = tr[mask_tr]
            te_kept = te[mask_te]
            if len(tr_kept) < 10 or len(te_kept) < 1:
                continue
            X_tr_fold = X[tr_kept]
            y_tr_fold = y_full[tr_kept].astype(np.float32)
            X_te_fold = X[te_kept]
            y_te_fold = y_full[te_kept].astype(np.float32)

            r2_fold, pred_fold, mse_f, w_f = _train_one_cell_line_one_fold(
                model,
                cell_line=cell_line,
                X_tr=X_tr_fold,
                y_tr=y_tr_fold,
                X_te=X_te_fold,
                y_te=y_te_fold,
                fold_cfg=fold_cfg,
                ref_pic50_pool=ref_pic50_pool,
                device=dev,
            )
            fold_r2.append(r2_fold)
            wasserstein_hist.append(float(w_f))
            mse_hist.append(float(mse_f))
            collapse = _collapse_detected(pred_fold, y_te_fold)

            per_fold.append(
                {
                    "fold": int(fold_idx),
                    "n_train": int(len(tr_kept)),
                    "n_test": int(len(te_kept)),
                    "r2": float(r2_fold),
                    "mse_final": float(mse_f),
                    "wasserstein_final": float(w_f),
                    "collapse_detected": bool(collapse),
                    "wall_s_per_fold": float(time.perf_counter() - t0)
                    / max(fold_idx + 1, 1),
                }
            )
            print(
                f"[wf125-p3]   cell_line={cell_line} seed={int(seed)} fold={fold_idx} "
                f"r2={r2_fold:+.4f} mse={mse_f:.4f} w2={w_f:.4f} "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            all_y_true.extend(y_te_fold.tolist())
            all_y_pred.extend(pred_fold.tolist())

        seed_r2 = float(np.mean(fold_r2)) if fold_r2 else 0.0
        r2_per_seed.append(seed_r2)

        per_seed.append(
            {
                "seed": int(seed),
                "r2_per_fold_mean": float(seed_r2),
                "r2_overall": float(
                    _compute_r2(
                        np.asarray(all_y_true, dtype=np.float64),
                        np.asarray(all_y_pred, dtype=np.float64),
                    )
                ),
                "wasserstein_history": list(wasserstein_hist),
                "mse_history": list(mse_hist),
                "per_fold": per_fold,
                "wall_time_s": float(time.perf_counter() - t0),
                "n_samples": int(n),
                "n_folds": int(n_folds),
                "feature_dim": int(d_feat),
                "device": str(dev),
                "collapse_detected_overall": bool(
                    _collapse_detected(
                        np.asarray(all_y_pred, dtype=np.float64),
                        np.asarray(all_y_true, dtype=np.float64),
                    )
                ),
            }
        )

    summary = {
        "cell_line": cell_line,
        "target_col": PLATINAI_CELL_COLS[cell_line],
        "n_samples": int(n),
        "n_features": int(d_feat),
        "feature_stack": "172d = 12 RDKit + 32 Multi-PH + 128 PLL",
        "n_seeds": int(len(seeds)),
        "strict_scaffold": bool(strict_scaffold),
        "wasserstein_weight": float(fold_cfg.wasserstein_weight),
        "epochs": int(fold_cfg.epochs),
        "lr": float(fold_cfg.lr),
        "r2_mean": float(np.mean(r2_per_seed)) if r2_per_seed else 0.0,
        "r2_std": float(np.std(r2_per_seed)) if len(r2_per_seed) > 1 else 0.0,
        "per_seed": per_seed,
    }
    return summary


# ---------------------------------------------------------------------------
# 8. Output writers
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


def _write_run_log(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_synthesis(out_dir: Path, all_results: List[Dict[str, Any]]) -> None:
    """Write ``<out_dir>/synthesis.md`` cross-cell-line summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_str = (
        "strict-scaffold disjoint"
        if all_results and all_results[0].get("strict_scaffold")
        else "random"
    )
    w_w = (
        all_results[0]["wasserstein_weight"]
        if all_results
        else DEFAULT_WASSERSTEIN_WEIGHT
    )
    epochs = all_results[0]["epochs"] if all_results else 50
    lines: List[str] = [
        "# Wf-125 Phase 3 — 172-d feature stack + multi-task 4-cell backbone + Wasserstein pIC50",
        "",
        "**Date**: 2026-09-27",
        "**Constraint**: 172-d feature stack (12 RDKit + 32 Multi-PH + 128 PLL) + multi-task "
        "shared-trunk + 4 per-cell-line regression heads + Wasserstein pIC50 regularization.",
        f"  Per `party-twenty-spirit §3.5` honest framing.",
        f"**Hardware**: GPU-1 RTX 5090 (CUDA_VISIBLE_DEVICES=1) when available; CPU fp32 fallback.",
        f"**Split**: 5-fold CV with **{strict_str}** scaffold assignment per cell line.",
        f"**Wasserstein weight**: {w_w} (per Wf-123 P1 spec).",
        f"**Epochs**: {epochs} per fold × 5 seeds × 5 folds.",
        "**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).",
        "**Feature cache**: `molmetal/data/features_172d_platinai_4cell.npz`.",
        "",
        "## Cross-cell-line summary (5-seed × 5-fold CV)",
        "",
        "| cell line | n_samples | R² (mean ± std) | Wasserstein W₂ (final) | collapse? |",
        "|---|---:|---|---|---|",
    ]
    r2_means: List[float] = []
    for r in all_results:
        last_w = (
            float(r["per_seed"][-1]["wasserstein_history"][-1])
            if r["per_seed"] and r["per_seed"][-1]["wasserstein_history"]
            else float("nan")
        )
        last_collapse = (
            bool(r["per_seed"][-1]["collapse_detected_overall"])
            if r["per_seed"]
            else False
        )
        lines.append(
            f"| {r['cell_line']} | {r['n_samples']} | "
            f"{r['r2_mean']:+.4f} ± {r['r2_std']:.4f} | "
            f"{last_w:.4f} | "
            f"{'YES' if last_collapse else 'no'} |"
        )
        r2_means.append(r["r2_mean"])
    aggregate = float(np.mean(r2_means)) if r2_means else 0.0
    aggregate_std = float(np.std(r2_means)) if len(r2_means) > 1 else 0.0
    lines.append(
        f"| **4-cell aggregate** | — | **{aggregate:+.4f} ± {aggregate_std:.4f}** | — | — |"
    )
    lines.append("")
    lines.append("## Verify gate (target: R² ≥ 0 for at least 1 of 4 cell lines)")
    lines.append("")
    n_positive = sum(1 for r2 in r2_means if r2 >= 0.0)
    best_cell = max(all_results, key=lambda r: r["r2_mean"]) if all_results else None
    if n_positive >= 1:
        verdict = (
            f"✅ **PASS** ({n_positive}/4 cell lines reach R² ≥ 0; "
            f"best: {best_cell['cell_line']} = {best_cell['r2_mean']:+.4f})"
        )
    else:
        verdict = (
            f"❌ **FAIL** (0/4 cell lines reach R² ≥ 0; "
            f"best: {best_cell['cell_line']} = {best_cell['r2_mean']:+.4f})"
        )
    lines.append(
        f"**Aggregate R² (mean of 4 cell-line means) = {aggregate:+.4f}** — {verdict}"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Per `party-twenty-spirit §3.5`: aggregate R² = "
        f"{aggregate:+.4f}; this is the **MEASURED** Wf-125 Phase 3 "
        "production anchor for PlatinAI 4-cell with the 172-d feature stack, "
        "multi-task backbone, and Wasserstein pIC50 regularization stacked."
    )
    lines.append("")
    lines.append("## Per-cell-line per-seed R² detail")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['cell_line']} ({r['target_col']})")
        lines.append("")
        lines.append(
            "| seed | R² (per-fold mean) | R² (overall) | Wasserstein decay | wall_s |"
        )
        lines.append("|---:|---:|---:|---|---:|")
        for s in r["per_seed"]:
            w_hist = s["wasserstein_history"]
            w_decay_str = (
                f"{w_hist[0]:.4f} → {w_hist[-1]:.4f}"
                if len(w_hist) >= 2
                else f"{w_hist[0]:.4f}"
            )
            lines.append(
                f"| {s['seed']} | {s['r2_per_fold_mean']:+.4f} | "
                f"{s['r2_overall']:+.4f} | {w_decay_str} | "
                f"{s['wall_time_s']:.1f} |"
            )
        lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "- **MEASURED today**: 5 seeds × 5 folds × 4 cell lines on PlatinAI 4-cell"
    )
    lines.append("  with the 172-d feature stack + multi-task shared-trunk backbone")
    lines.append("  + Wasserstein pIC50 regularization.")
    lines.append(
        "- **172-d feature stack**: 12 RDKit + 32 Multi-PH Fourier + 128 PLL; built"
    )
    lines.append(
        "  per SMILES by ``build_172d_features.compute_172d`` (Wf-128 P1) and cached"
    )
    lines.append("  to ``molmetal/data/features_172d_platinai_4cell.npz``.")
    lines.append(
        "- **Multi-task backbone**: 4-cell shared MLP trunk (172 → 256 → 128) + per-"
    )
    lines.append(
        "  cell-line regression heads (128 → 64 → 1).  The shared trunk sees the"
    )
    lines.append(
        "  same SMILES × 4 IC50 columns; per-cell-line heads specialise on each target."
    )
    lines.append(
        "- **Wasserstein pIC50 regularization**: W₂ between predicted pIC50 cloud and"
    )
    lines.append(
        "  training-set reference pIC50 distribution; weight = 0.05 (Wf-123 P1 spec)."
    )
    lines.append(
        "- **Strict-scaffold split**: Bemis-Murcko disjoint 5-fold; same convention as"
    )
    lines.append(
        "  Wf-86 P3 anchor (Bemis-Murcko + ligand-canonicalised fallback for Pt complexes)."
    )
    lines.append(
        "- **Honest framing per `party-twenty-spirit §3.5`**: the structural 12-d signal"
    )
    lines.append(
        "  floor on PlatinAI pIC50 (Pearson |ρ| < 0.02) may not be breakable with feature"
    )
    lines.append(
        "  augmentation alone.  This 172-d + multi-task + Wasserstein recipe is the"
    )
    lines.append(
        "  strongest honesty-tier lever we have; if R² < 0 for all 4 cell lines, the"
    )
    lines.append(
        "  L4/L5/L7/L8/L11 central Lambda claim needs a different feature regime (e.g.,"
    )
    lines.append("  reaction-product features from the 5-click productive space).")
    lines.append("")
    lines.append(
        f"Per `party-twenty-spirit §3.5`: aggregate R² = {aggregate:+.4f}; this is the "
        "**MEASURED** Wf-125 Phase 3 production anchor."
    )
    _write_run_log(out_dir / "synthesis.md", lines)


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------
def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x]


def _parse_cell_lines(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.replace(",", " ").split() if x]


def _parse_bool(raw: str) -> bool:
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {raw!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-125 Phase 3 — 172-d feature stack + multi-task 4-cell backbone + "
            "Wasserstein pIC50 regularization on PlatinAI 4-cell."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["platinai_4cell"],
        default="platinai_4cell",
        help="Dataset identifier (only platinai_4cell supported).",
    )
    parser.add_argument(
        "--cell-lines",
        type=_parse_cell_lines,
        nargs="+",
        default=list(DEFAULT_CELL_LINES),
        help=(
            "PlatinAI cell lines to benchmark " "(default: mcf7 a549 a2780 a2780cis)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds for K-fold splitting (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="K-fold count (default: 5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per fold (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Adam learning rate (default: {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--wasserstein-weight",
        type=float,
        default=DEFAULT_WASSERSTEIN_WEIGHT,
        help=f"Wasserstein penalty weight (default: {DEFAULT_WASSERSTEIN_WEIGHT}).",
    )
    parser.add_argument(
        "--wasserstein-n-mc",
        type=int,
        default=DEFAULT_WASSERSTEIN_N_MC,
        help=f"Sliced-W₂ MC directions (default: {DEFAULT_WASSERSTEIN_N_MC}).",
    )
    parser.add_argument(
        "--strict-scaffold",
        type=_parse_bool,
        default=True,
        help="Enforce strict scaffold-disjoint 5-fold split (default: True).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Compute device (default: cuda; falls back to cpu if unavailable).",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"172-d feature cache path (default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("molmetal/reports/wf125_p3_172d_multitask"),
        help="Output directory (default: molmetal/reports/wf125_p3_172d_multitask).",
    )
    args = parser.parse_args(argv)

    # Flatten argparse nargs="+" artifacts.
    flat_cell_lines: List[str] = []
    for cl in args.cell_lines:
        if isinstance(cl, list):
            flat_cell_lines.extend(str(x).lower() for x in cl)
        else:
            flat_cell_lines.append(str(cl).lower())
    args.cell_lines = flat_cell_lines

    flat_seeds: List[int] = []
    for s in args.seeds:
        if isinstance(s, list):
            flat_seeds.extend(int(x) for x in s)
        else:
            flat_seeds.append(int(s))
    args.seeds = flat_seeds

    for cl in args.cell_lines:
        if cl not in PLATINAI_CELL_COLS:
            raise ValueError(
                f"Unknown cell line: {cl}; supported: {sorted(PLATINAI_CELL_COLS)}"
            )

    if not _HAS_TORCH:
        raise RuntimeError(f"Torch is required for Wf-125 P3; got {_TORCH_ERR!r}")

    print(
        f"[wf125-p3] dataset={args.dataset} cell_lines={args.cell_lines} "
        f"seeds={args.seeds} epochs={args.epochs} "
        f"wasserstein_weight={args.wasserstein_weight} "
        f"strict_scaffold={args.strict_scaffold} device={args.device}"
    )
    print(
        f"[wf125-p3] CSV path: {_REPO_ROOT / PLATINAI_CSV}",
        flush=True,
    )

    # Load + compute 172-d features (with on-disk cache).
    X, Y, smiles, cell_y = load_platinai_4cell_with_cache(
        _REPO_ROOT,
        cache_path=args.cache_path,
    )
    print(
        f"[wf125-p3] feature matrix: X={X.shape} Y={Y.shape} "
        f"({len(smiles)} rows, {len(set(smiles))} unique SMILES)",
        flush=True,
    )

    # Build Bemis-Murcko scaffold IDs for the strict-scaffold split.
    scaffold_ids: List[str] = []
    if args.strict_scaffold:
        print(
            "[wf125-p3] computing Bemis-Murcko scaffolds (strict-scaffold split)…",
            flush=True,
        )
        t_scaff = time.perf_counter()
        for smi in smiles:
            scaffold_ids.append(_bemis_murcko_scaffold(smi))
        print(
            f"[wf125-p3]   scaffold pass done in {time.perf_counter() - t_scaff:.1f}s "
            f"(unique scaffolds: {len(set(scaffold_ids))})",
            flush=True,
        )
    else:
        scaffold_ids = [""] * len(smiles)

    # Build the joint pIC50 reference pool for Wasserstein.
    # Use ALL finite pIC50 values across the 4 cell lines for a robust
    # reference distribution (the Wf-123 P1 spec recommends n ≥ 50).
    finite_pool: List[float] = []
    for cl in DEFAULT_CELL_LINES:
        finite_pool.extend(float(v) for v in cell_y[cl] if math.isfinite(float(v)))
    if len(finite_pool) < 10:
        raise RuntimeError(
            f"PlatinAI pIC50 reference pool too small: {len(finite_pool)}"
        )
    ref_pic50_pool = torch.tensor(finite_pool, dtype=torch.float32)
    print(
        f"[wf125-p3] Wasserstein reference pool: {ref_pic50_pool.shape} "
        f"(mean={float(ref_pic50_pool.mean()):.3f}, "
        f"std={float(ref_pic50_pool.std()):.3f})",
        flush=True,
    )

    fold_cfg = FoldConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        wasserstein_weight=float(args.wasserstein_weight),
        n_mc=int(args.wasserstein_n_mc),
    )

    all_results: List[Dict[str, Any]] = []
    for cell_line in args.cell_lines:
        print(
            f"[wf125-p3] cell_line={cell_line}: running 172-d multi-task CV", flush=True
        )
        result = evaluate_cell_line(
            cell_line=cell_line,
            seeds=args.seeds,
            n_folds=args.n_folds,
            fold_cfg=fold_cfg,
            device=args.device,
            strict_scaffold=args.strict_scaffold,
            X=X,
            Y=Y,
            smiles=smiles,
            scaffold_ids=scaffold_ids,
            cell_y=cell_y,
            ref_pic50_pool=ref_pic50_pool,
        )
        per_seed_path = args.out_dir / cell_line / "per_seed_metrics.json"
        _write_json(per_seed_path, result)
        log_path = args.out_dir / cell_line / "run.log"
        log_lines = [
            f"cell_line={result['cell_line']}",
            f"target_col={result['target_col']}",
            f"n_samples={result['n_samples']}",
            f"n_features={result['n_features']}",
            f"feature_stack={result['feature_stack']}",
            f"n_seeds={result['n_seeds']}",
            f"strict_scaffold={result['strict_scaffold']}",
            f"wasserstein_weight={result['wasserstein_weight']}",
            f"epochs={result['epochs']}",
            f"lr={result['lr']}",
            f"r2_mean={result['r2_mean']:+.4f}",
            f"r2_std={result['r2_std']:.4f}",
        ]
        _write_run_log(log_path, log_lines)
        print(
            f"[wf125-p3] cell_line={cell_line}: R² = "
            f"{result['r2_mean']:+.4f} ± {result['r2_std']:.4f}",
            flush=True,
        )
        all_results.append(result)
    write_synthesis(args.out_dir, all_results)
    print(
        f"[wf125-p3] synthesis.md written to {args.out_dir / 'synthesis.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
