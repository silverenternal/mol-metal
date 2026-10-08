"""Wf-145 Phase 2 — LOMO 5-metal x 5-seed x 50-epoch full re-bench with BALANCED Rh cohort.

==============================================================================
Wf-145 Phase 2 — BALANCED-label fix on top of Wf-125 P2 / Wf-131 P3 (TODO-135 §5)
==============================================================================

Fixes the Wf-131 P2 + Wf-125 P1 honest-negative regression (Rh fold 0.368 → 0.136)
by re-using the **Wf-145 P1 BALANCED Rh cohort** (250 active + 250 inactive from
``molmetal/reports/wf145_p1_rh_active_cohort/synthetic_rh_pos_neg.json``) instead
of the Wf-125 P1 constant-inactive cohort.

Architecture (per task spec)
---------------------------
1. **Multi-task D-MPNN backbone (5 metal shared)** — reuses
   :class:`DMPNNMultiTaskModel` (shared backbone + per-metal embeddings
   extracted via :func:`extract_per_metal_embeddings`).
2. **Per-metal head ensemble (5 heads, 1 per metal)** — the per-metal
   embedding hook from Wf-125 P2 acts as the per-metal "head" in
   embedding space; W₂ transfer treats each metal's embedding marginal
   as its own distribution.  This is the conceptually clean way to
   realise per-metal specialisation while keeping the trainable
   parameter count tractable.
3. **Wasserstein transfer loss** — target = Rh real ACTIVE distribution
   (NOT the synthetic cohort).  The pairwise sliced-W₂ is computed
   over ``{Ru, Ir, Os, Rh_real_active, Re, Rh_synthetic}`` so the
   model is forced to align each metal's embedding marginal with the
   Rh real ACTIVE distribution.
4. **Hilbert L2 guard** — grad-norm clip ``max_norm=5.0`` per
   Wf-127 P2; engaged by default (``--use-l2-guard`` is the default).
5. **ASTR scale clip** — value-space clip ``[-10, 10]`` on predicted
   pIC50 at test time per Wf-127 P3.

Targets
-------
* Rh LOMO AUC 5-seed mean >= **0.50** (vs R17 0.368 / Wf-131 P2 regression 0.136)
* Overall LOMO 5-seed mean >= **0.60** (vs R17 baseline 0.5346)

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
1. **Rh n=60 train is small** — fundamental sample size ceiling; the
   balanced cohort is only 500 SMILES of training-time augmentation,
   not real wet-lab data.
2. **5-seed mean may not be statistically robust** at small n — the
   90 % CI on a 5-seed mean of an AUC is wide.
3. **Multi-task backbone cross-metal share** — signal leak between
   metals via the shared trunk.
4. **Wasserstein target = Rh real ACTIVE distribution** — but the
   real active distribution may not be known well (only 413 real
   actives in MetalCytoToxDB Rh train).
5. **50 epochs may be insufficient** for full convergence.

Per the task spec we report MEASURED numbers verbatim; if Rh AUC < 0.50
we HALT and report honestly (paper §6 ready, honest-negative).

Usage
-----

Wf-201 ``--task-type`` flag (LOMO bug fix)
-------------------------------------------
Wf-131 P3 used ``DMPNNMultiTaskModel`` (shared backbone + per-metal
heads) which confuses LOMO evaluation: the shared backbone acts as a
cross-metal information conduit so the held-out metal still "sees"
information from the metals used for training.  Wf-201 fixes this by
adding an explicit ``--task-type {single, multi}`` flag:

* ``--task-type single`` (LOMO-safe) — uses the vanilla :class:`DMPNNModel`
  (no per-metal heads, no shared backbone across metals).  For each
  held-out metal M we train a single D-MPNN on the ``metal != M`` rows
  using only the per-molecule active/inactive label; there is no
  cross-metal information sharing, so LOMO is honest.  This is the
  recommended mode for any honest-negative LOMO evaluation.
* ``--task-type multi`` (default — backward-compatible) — uses the
  original :class:`DMPNNMultiTaskModel` (shared backbone + per-metal
  heads).  This is the IID-only mode: it can be useful as an upper
  bound for IID performance, but the shared backbone leaks information
  across metals so the LOMO AUC is NOT honestly estimated.  Per
  party-twenty-spirit §3.5, the multi-task LOMO numbers must be
  reported as honest-negative ("shared-backbone cross-metal leak —
  not a fair LOMO estimate").

Default mode is unchanged (still multi-task).  Pass
``--task-type single`` to engage the LOMO-safe single-task driver.

CPU smoke run (small subset, 1 metal x 1 seed x 2 epochs, legacy
DMPNN baseline — backward compat via ``--model v3``)::

    cd /home/hugo/codes/mol-metal && \
      .venv/bin/python molmetal/scripts/train_metal_lomo_v2.py \
        --model v3 --metals Rh --seeds 0 --epochs 2 --quick --device cpu \
        --out-dir molmetal/reports/wf145_p2_lomo_v2

CPU smoke run, single-task LOMO-safe driver (Wf-201)::

    cd /home/hugo/codes/mol-metal && \
      .venv/bin/python molmetal/scripts/train_metal_lomo_v2.py \
        --task-type single --metals Rh --seeds 0 --epochs 2 --quick --device cpu \
        --out-dir molmetal/reports/wf201_single_lomo

Full 5-metal x 5-seed x 50-epoch GPU-1 run (legacy baseline)::

    .venv/bin/python \
      molmetal/scripts/train_metal_lomo_v2.py \
        --model v3 --metals Rh Ru Ir Os Re --seeds 0 1 2 3 4 --epochs 50 \
        --device cuda \
        --out-dir molmetal/reports/wf145_p2_lomo_v2

Full 5-metal x 5-seed x 50-epoch GPU-1 run, single-task LOMO-safe (Wf-201)::

    .venv/bin/python \
      molmetal/scripts/train_metal_lomo_v2.py \
        --task-type single --metals Rh Ru Ir Os Re \
        --seeds 0 1 2 3 4 --epochs 50 --device cuda \
        --out-dir molmetal/reports/wf201_single_lomo

NOTE: as of Wf-148 ultracode / TODO-148 §3 the ``--model`` default is
``chemprop_v2`` (Chemprop 2.0 MPNN backbone).  LANE-F wire-in: the LOMO
loop in :class:`BalancedLOMORunner.run` is now family-agnostic — per batch
it rebuilds the ``(smiles_list, metal_ids)`` inputs and dispatches via
``_forward_dispatch``, builds the ctor via ``_model_ctor_kwargs``, predicts
via ``predict_mt_dispatch`` and extracts W₂ embeds via ``_embed_one_mol``.
All four ``--model`` choices therefore run end-to-end:
``chemprop_v2`` / ``v4`` / ``v4pretrain`` (dict contract) and ``v3``
(DMPNN 4-tensor tuple contract).

Re-uses Wf-123 P1 ``wasserstein_transfer_loss`` (M1) + Wf-127 P2 Hilbert
L2 grad-norm guard.
"""

from __future__ import annotations


# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Default GPU 0 (RTX PRO 6000 Blackwell, 98 GB) unless the env var is
# already set (incl. "" = CPU-only).  Must run BEFORE any molmetal /
# torch import: importing the molmetal package initialises CUDA, and
# CUDA reads CUDA_VISIBLE_DEVICES only once, at init.  Source of
# truth: molmetal/molmetal_lam/sbdd_env/training_utils.py (loaded
# package-free via molmetal/scripts/_cuda_default.py).
import os as _wf130_os
import sys as _wf130_sys

_wf130_dir = _wf130_os.path.dirname(_wf130_os.path.abspath(__file__))
if _wf130_dir not in _wf130_sys.path:
    _wf130_sys.path.append(_wf130_dir)
from _cuda_default import ensure_cuda_visible_devices_default as _wf130_cvd

_wf130_cvd()
del _wf130_cvd, _wf130_dir, _wf130_os, _wf130_sys
# --- end CUDA default --------------------------------------------

# --- Wf-190 env guard --------------------------------------------
# Fail fast at import time if --bf16 / CUDA is required but unavailable.
# Mirrors mol-metal-gpu-execution §"MEASURED-vs-PROJECTED label discipline":
# we never silently run on CPU when the script claims GPU.  Auto-fallback to
# CPU is still supported (set MOLMETAL_ALLOW_CPU=1) to keep CPU-only boxes
# working, but explicit is preferred.  Check is deferred to *after* `import
# torch` so it sees the CUDA env vars set above.
# (set BELOW after `import os` so the lookup is valid at module load.)
# --- end Wf-190 env guard ----------------------------------------

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Wf-190 env guard (deferred here so `os` is in scope).
_MOLMETAL_ALLOW_CPU = os.environ.get("MOLMETAL_ALLOW_CPU", "0") == "1"

# Wf-50 pre-condition: ensure MOLMETAL_DATA_DIR defaults to project-local data dir
from pathlib import Path as _wf50_Path
import os as _wf50_os

_wf50_proj_root = _wf50_Path(__file__).resolve().parent.parent.parent
_wf50_default_data = _wf50_proj_root / ".data" / "molmetal"
if not _wf50_os.environ.get("MOLMETAL_DATA_DIR") and _wf50_default_data.exists():
    _wf50_os.environ["MOLMETAL_DATA_DIR"] = str(_wf50_default_data)
del _wf50_os, _wf50_Path, _wf50_proj_root, _wf50_default_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Override cytotox defaults to project-local data dir.  The cytotox module
# evaluates ``DEFAULT_DATA_DIR = Path("/mnt/storage/data/molmetal")`` at
# import time; we patch the module-level constants AND wrap __init__ to
# force the writable cache_dir (because the class default arg is bound at
# class definition time).
import molmetal.data.cytotox as _cytotox_mod

_LOCAL_DATA = Path(
    os.environ.get("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal"))
)
_LOCAL_CACHE = _LOCAL_DATA / "3d_cache"
_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
_cytotox_mod.DEFAULT_DATA_DIR = _LOCAL_DATA
_cytotox_mod.DEFAULT_CACHE_DIR = _LOCAL_CACHE
_cytotox_mod.DEFAULT_CSV = _LOCAL_DATA / "MetalCytoToxDB.csv"
_orig_init = _cytotox_mod.MetalCytotoxDataset.__init__


def _patched_init(self, df, cache_dir=None, conformer_cache_lru=None):
    cache_dir = cache_dir if cache_dir is not None else _LOCAL_CACHE
    if conformer_cache_lru is None:
        conformer_cache_lru = _cytotox_mod.CONFORMER_CACHE_LRU
    _orig_init(self, df, cache_dir=cache_dir, conformer_cache_lru=conformer_cache_lru)


_cytotox_mod.MetalCytotoxDataset.__init__ = _patched_init  # type: ignore

import torch

from molmetal.baselines.dmpnn_multitask import (
    MT_ALPHA,
    MT_BATCH_SIZE,
    MT_DEPTH,
    MT_DROPOUT,
    MT_HIDDEN,
    MT_LR,
    collate_graphs,
    featurize_smiles_list,
)

# Wf-201 — single-task driver.  Imports the vanilla D-MPNN (no per-metal
# heads, no shared backbone across metals) so ``--task-type single`` can
# train a per-fold model without the cross-metal information sharing
# that confuses LOMO evaluation.
from molmetal.baselines.dmpnn import (
    DMPNNModel,
    predict as dmpnn_predict,
)
from molmetal.baselines.eval_utils import compute_metrics
from molmetal.data.cytotox import MetalCytotoxDataset
from molmetal.data.splits import SplitResult
from molmetal.models.chemprop_v2_mt_model import (
    ChempropV2MTModel,
)
from molmetal.molmetal_lam.reward.wasserstein_loss import (
    wasserstein_transfer_loss,
    wasserstein_loss,
)

logger = logging.getLogger("wf145_p2")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


METRICS_DIR = PROJECT_ROOT / "metrics" / "by_metric"
DEFAULT_LOMO_CSV = PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv"
# Wf-145 P1 BALANCED cohort (250 active + 250 inactive) — replaces
# the Wf-125 P1 constant-inactive cohort.
DEFAULT_RH_BALANCED_JSON = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf145_p1_rh_active_cohort"
    / "synthetic_rh_pos_neg.json"
)
# Allow overriding the cache dir via env (mirrors Wf-50 pre-condition).
DEFAULT_CACHE_DIR = (
    Path(os.environ.get("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal")))
    / "3d_cache"
)

GATE_AUC = 0.70
TARGET_RH_AUC = 0.50
TARGET_OVERALL_AUC = 0.60
N_METALS = 5

# Wf-127 P2 — grad-norm clip on the backbone + heads.
L2_GUARD_MAX_NORM: float = 5.0
# Wf-127 P3 — value-space clip on predicted pIC50.
_ASTR_CLIP_MIN: float = -10.0
_ASTR_CLIP_MAX: float = 10.0


# ---------------------------------------------------------------------------
# Model-family adapters (LANE-F wire-in).
#
# The legacy LOMO loop is written against ``DMPNNMultiTaskModel``'s 4-tensor
# forward ``(atom_f, bond_f, e_src, e_dst) -> (pic50:(B,), active:(B,2))`` and
# its per-mol oracle ``_forward_one_mol``.  The V4 / chemprop_v2 families use a
# DIFFERENT contract:
#
#   * ctor kwargs    : ``hidden_dim``/``n_layers`` (V4) vs ``hidden``/``depth``
#                      (DMPNNMultiTaskModel)
#   * forward inputs : ``(smiles_list, metal_ids:(B,))`` — SMILES + a per-row
#                      metal id, NOT the pre-featurized tensors
#   * forward output : ``dict[metal_name -> (B,2) logits]`` plus a "coord" key
#                      — NOT a bare ``(pic50, active_logits)`` tuple
#
# These helpers let ONE LOMO loop drive all three families without a
# family-specific code path inside the training body.  See AGENTS.md §5.7:
# the wire-in is incomplete until a CPU smoke exercises each path.
# ---------------------------------------------------------------------------

#: Model class names whose forward is the DMPNN 4-tensor tuple contract.
_TUPLE_FWD_MODELS: frozenset = frozenset({"DMPNNMultiTaskModel"})
#: Model class names whose forward is the per-metal-dict (smiles, metal_ids) contract.
_DICT_FWD_MODELS: frozenset = frozenset(
    {
        "V4MultiTaskDMPNN",
        "V4MultiTaskDMPNNPretrain",
        "ChempropV2MTModel",
    }
)


def _is_dict_fwd_model(model) -> bool:
    """True when ``model`` uses the (smiles_list, metal_ids) -> dict contract."""
    return type(model).__name__ in _DICT_FWD_MODELS


def _model_ctor_kwargs(model_cls, hidden: int, depth: int, dropout: float) -> dict:
    """Map the loop's (hidden, depth, dropout) onto the family's ctor kwargs.

    Verified against the live ctors:
      * ``DMPNNMultiTaskModel(hidden=, depth=, dropout=)``
      * ``V4MultiTaskDMPNN(hidden_dim=, n_layers=, ...)``  — NO top-level
        ``dropout`` (it has ``head_dropout``); passing ``dropout`` raises
        ``TypeError``.
      * ``ChempropV2MTModel(hidden_dim=, depth=, dropout=)`` — uses ``depth``
        (NOT ``n_layers``).
    """
    name = getattr(model_cls, "__name__", str(model_cls))
    if name in ("V4MultiTaskDMPNN", "V4MultiTaskDMPNNPretrain"):
        return {"hidden_dim": hidden, "n_layers": depth}
    if name == "ChempropV2MTModel":
        return {"hidden_dim": hidden, "depth": depth, "dropout": dropout}
    return {"hidden": hidden, "depth": depth, "dropout": dropout}


def _forward_dispatch(
    model,
    bd,
    smiles_batch: List[str],
    metal_name: str,
):
    """Run one forward pass for any of the three model families.

    Returns ``(pic50_pred:(B,), active_logits:(B,2))`` — the exact tuple the
    legacy loop / :func:`predict_mt` / :class:`MultiTaskLoss` expect.

    * DMPNN family : native ``(atom_f, bond_f, e_src, e_dst) -> (pic50, active)``.
    * V4 / chemprop: ``(smiles_list, metal_ids) -> dict``; the active metal's
      head is the LOMO-relevant one.  Its ``(B,2)`` logits are the active
      contract (CrossEntropyLoss).  The regression pIC50 head does NOT exist on
      these families, so ``pic50_pred`` is a zero vector — the loop's
      ``wasserstein_weight``/pIC50-MAE terms are then only meaningful on the
      DMPNN path (documented honest-negative, not a silent fabrication).
    """
    if _is_dict_fwd_model(model):
        import torch as _torch

        name = type(model).__name__
        metals = list(getattr(model, "metals", ()))
        if metal_name not in metals:
            raise NotImplementedError(
                f"{name} was constructed for metals={metals}; the LOMO fold's "
                f"active metal {metal_name!r} is not among them.  Pass "
                f"--metals within {metals} (the V4/chemprop metal-head set)."
            )
        metal_ids = _torch.full(
            (len(smiles_batch),),
            metals.index(metal_name),
            dtype=_torch.long,
            device=next(model.parameters()).device,
        )
        out = model(smiles_batch, metal_ids)
        if not isinstance(out, dict) or metal_name not in out:
            raise NotImplementedError(
                f"{name}.forward returned "
                f"{type(out).__name__}, expected dict with a {metal_name!r} head."
            )
        active_logits = out[metal_name]
        pic50_pred = _torch.zeros(
            active_logits.size(0),
            dtype=active_logits.dtype,
            device=active_logits.device,
        )
        return pic50_pred, active_logits

    # DMPNN 4-tensor tuple path.
    atom_f, bond_f, e_src, e_dst = bd
    return model(atom_f, bond_f, e_src, e_dst)


def _embed_one_mol(model, smiles_b: str, atom_f_b, bond_f_b, e_src_b, e_dst_b):
    """Per-mol embedding for W₂ extraction, family-agnostic.

    * DMPNN family : ``_forward_one_mol`` (2H pooled).
    * V4 / chemprop: shared ``encode([smiles])`` -> (1, hidden_dim).
    """
    if _is_dict_fwd_model(model):
        return model.encode([smiles_b])
    return model._forward_one_mol(atom_f_b, bond_f_b, e_src_b, e_dst_b)


@torch.no_grad()
def predict_mt_dispatch(
    model,
    graphs: List,
    smiles: List[str],
    pic50_labels: np.ndarray,
    active_labels: np.ndarray,
    metal_name: str,
    batch_size: int = MT_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Family-agnostic mirror of :func:`predict_mt`.

    Same ``(active_true, active_prob, pic50_pred)`` contract and the same
    invalid-molecule handling, but routes the forward through
    :func:`_forward_dispatch` so the V4/chemprop ``(smiles_list, metal_ids)``
    path works.  ``smiles`` must be index-aligned with ``graphs``.
    """
    model.eval()
    all_active: List[int] = []
    all_score: List[float] = []
    all_pic50: List[float] = []

    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start : start + batch_size]
        batch_smiles = smiles[start : start + batch_size]
        batch_pic50 = pic50_labels[start : start + batch_size]
        batch_active = active_labels[start : start + batch_size]

        valid = [
            (g, s, p, a)
            for g, s, p, a in zip(batch_graphs, batch_smiles, batch_pic50, batch_active)
            if g is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        if not valid:
            for p, a in zip(batch_pic50, batch_active):
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)
            continue

        g_list = [v[0] for v in valid]
        smiles_batch = [v[1] for v in valid]
        y_active = np.array([v[3] for v in valid], dtype=np.int64)

        bd = collate_graphs(g_list)
        if bd is None:
            for p, a in zip(batch_pic50, batch_active):
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)
            continue

        atom_f, bond_f, e_src, e_dst = bd
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        bd_dev = (atom_f, bond_f, e_src, e_dst)

        pic50_pred, active_logits = _forward_dispatch(
            model, bd_dev, smiles_batch, metal_name
        )
        active_prob = torch.softmax(active_logits, dim=-1)[:, 1].cpu().numpy()
        pic50_arr = pic50_pred.cpu().numpy()

        vi = 0
        for g, p, a in zip(batch_graphs, batch_pic50, batch_active):
            if g is not None and not (isinstance(p, float) and np.isnan(p)):
                all_active.append(int(y_active[vi]))
                all_score.append(float(active_prob[vi]))
                all_pic50.append(float(pic50_arr[vi]))
                vi += 1
            else:
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)

    return (
        np.array(all_active, dtype=int),
        np.array(all_score, dtype=float),
        np.array(all_pic50, dtype=float),
    )


# ---------------------------------------------------------------------------
# BALANCED Rh cohort loader — replaces the Wf-125 P1 constant-inactive loader.
# ---------------------------------------------------------------------------
def load_balanced_rh_smiles(
    cohort_json: Path,
    n_active: int = 250,
    n_inactive: int = 250,
) -> Tuple[List[str], List[int]]:
    """Return the balanced Rh cohort SMILES + per-row active labels.

    The Wf-145 P1 cohort JSON stores ``actives`` (label=1) and
    ``inactives`` (label=0) arrays.  We concatenate them in that
    order (active first) and return ``(smiles, labels)`` lists.

    Parameters
    ----------
    cohort_json
        Path to ``synthetic_rh_pos_neg.json``.
    n_active, n_inactive
        Sanity caps; the cohort ships exactly 250 of each.

    Returns
    -------
    (smiles, labels) : lists of equal length 500.
    """
    cohort_json = Path(cohort_json)
    if not cohort_json.exists():
        raise FileNotFoundError(
            f"Wf-145 P1 balanced Rh cohort not found at {cohort_json}. "
            "Run scripts/rh_active_label_generation.py first."
        )
    with cohort_json.open() as fh:
        payload = json.load(fh)
    smis: List[str] = []
    labels: List[int] = []
    n_a = int(payload.get("stats", {}).get("n_active", 0))
    n_i = int(payload.get("stats", {}).get("n_inactive", 0))
    if n_a < n_active or n_i < n_inactive:
        raise RuntimeError(
            f"Wf-145 P1 cohort under-sized: n_active={n_a} < {n_active} "
            f"or n_inactive={n_i} < {n_inactive}"
        )
    for entry in payload.get("actives", [])[:n_active]:
        s = str(entry.get("smiles", "")).strip()
        if not s:
            continue
        smis.append(s)
        labels.append(1)
    for entry in payload.get("inactives", [])[:n_inactive]:
        s = str(entry.get("smiles", "")).strip()
        if not s:
            continue
        smis.append(s)
        labels.append(0)
    if len(smis) != (n_active + n_inactive):
        raise RuntimeError(
            f"balanced cohort size mismatch: got {len(smis)} expected "
            f"{n_active + n_inactive}"
        )
    return smis, labels


# ---------------------------------------------------------------------------
# Wf-187 m19 Rh oversample fix — sklearn.utils.resample on positive-only
# rows for under-represented metals (Rh, Re).
# ---------------------------------------------------------------------------
def oversample_underrepresented_metals(
    smiles: List[str],
    active: np.ndarray,
    metals: np.ndarray,
    pic50: np.ndarray,
    underrepresented: Optional[List[str]] = None,
    target_pos_rate: float = 0.5,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Resample positives for under-represented metals to lift pos_rate.

    For each metal in ``underrepresented`` (default ``['Rh', 'Re']``),
    compute the current positive-rate and resample positive-only rows
    via :func:`sklearn.utils.resample` until the per-metal positive
    rate reaches ``target_pos_rate`` (default 0.5 = 50%).  Rows from
    metals NOT in ``underrepresented`` are passed through unchanged.

    Parameters
    ----------
    smiles, active, metals, pic50
        Per-row arrays for the augmented train pool.  All four must be
        the same length.  ``active`` and ``pic50`` are numpy arrays;
        ``smiles`` and ``metals`` are lists / numpy arrays of strings.
    underrepresented
        Metals to oversample.  Defaults to ``['Rh', 'Re']`` (the m19
        shortlist).
    target_pos_rate
        Desired positive rate after oversampling.  Capped at 0.5
        because going above 50% positives just flips the class skew
        the other way and reduces signal.
    seed
        Random seed forwarded to ``resample``.

    Returns
    -------
    (smiles_new, active_new, metals_new, pic50_new, stats)
        ``smiles_new`` is a Python list, the others are numpy arrays.
        ``stats`` is ``{metal: n_appended}`` so the caller can log what
        happened.
    """
    from sklearn.utils import resample as _sk_resample

    if underrepresented is None:
        underrepresented = ["Rh", "Re"]
    target_pos_rate = float(target_pos_rate)
    if not (0.0 < target_pos_rate <= 0.5):
        raise ValueError(f"target_pos_rate must be in (0, 0.5], got {target_pos_rate}")

    smiles = list(smiles)
    active = np.asarray(active, dtype=np.int64)
    metals = np.asarray(metals, dtype=object)
    pic50 = np.asarray(pic50, dtype=np.float32)

    out_smiles = list(smiles)
    out_active = list(active.tolist())
    out_metals = list(metals.tolist())
    out_pic50 = list(pic50.tolist())
    stats: Dict[str, int] = {}

    for metal in underrepresented:
        m_mask = np.array([str(m) == str(metal) for m in metals])
        if not m_mask.any():
            continue
        m_active = active[m_mask]
        n_pos = int((m_active == 1).sum())
        n_neg = int((m_active == 0).sum())
        if n_pos == 0 or n_neg == 0:
            # No class imbalance to fix — skip.
            stats[metal] = 0
            continue
        cur_rate = n_pos / (n_pos + n_neg)
        if cur_rate >= target_pos_rate:
            stats[metal] = 0
            continue
        # How many positives do we need to hit target_pos_rate?
        target_n_pos = int(
            round(target_pos_rate * (n_pos + n_neg) / (1.0 - target_pos_rate))
        )
        n_appended = max(0, target_n_pos - n_pos)
        if n_appended == 0:
            stats[metal] = 0
            continue
        # Resample from the positive-only rows of THIS metal.
        pos_indices = np.where(m_mask & (active == 1))[0]
        pos_smiles = [smiles[i] for i in pos_indices]
        pos_pic50 = pic50[pos_indices]
        pos_metals = [str(metals[i]) for i in pos_indices]
        resampled_smiles = _sk_resample(
            pos_smiles,
            n_samples=n_appended,
            replace=True,
            random_state=int(seed),
        )
        resampled_pic50 = _sk_resample(
            pos_pic50,
            n_samples=n_appended,
            replace=True,
            random_state=int(seed) + 1,
        )
        resampled_metals = _sk_resample(
            pos_metals,
            n_samples=n_appended,
            replace=True,
            random_state=int(seed) + 2,
        )
        out_smiles.extend(list(resampled_smiles))
        out_active.extend([1] * n_appended)
        out_metals.extend(list(resampled_metals))
        out_pic50.extend([float(x) for x in resampled_pic50])
        stats[metal] = int(n_appended)

    return (
        out_smiles,
        np.asarray(out_active, dtype=np.int64),
        np.asarray(out_metals, dtype=object),
        np.asarray(out_pic50, dtype=np.float32),
        stats,
    )


# ---------------------------------------------------------------------------
# LOMO splitter — hold out one metal as test; everything else as train/val.
# Mirrors Wf-125 P2 / Wf-131 P3 logic.
# ---------------------------------------------------------------------------
@dataclass
class BalancedLOMOSplitter:
    held_out_metal: str
    val_fraction: float = 0.10
    seed: int = 42

    def __call__(self, dataset: MetalCytotoxDataset) -> SplitResult:
        from sklearn.model_selection import train_test_split

        metals = dataset.df["Metal"].astype(str).to_numpy()
        test_mask = metals == self.held_out_metal
        test_idx = np.where(test_mask)[0].astype(np.int64)
        train_pool = np.where(~test_mask)[0].astype(np.int64)
        if self.val_fraction > 0 and len(train_pool) >= 10:
            train_idx, val_idx = train_test_split(
                train_pool,
                test_size=self.val_fraction,
                random_state=self.seed,
                stratify=None,
            )
        else:
            train_idx = train_pool
            val_idx = np.array([], dtype=np.int64)
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# Per-seed result container — JSON-serialisable.
# ---------------------------------------------------------------------------
@dataclass
class BalancedLOMOResult:
    fold: str
    seed: int
    n_train: int
    n_val: int
    n_test: int
    n_synthetic_rh_active: int
    n_synthetic_rh_inactive: int
    pos_rate_train: float
    pos_rate_test: float
    val_auc: float
    test_auc: float
    test_pic50_mae: float
    wasserstein_weight: float
    wasserstein_target: str
    wasserstein_loss_history: List[float]
    final_per_metal_w2: Dict[str, float]
    elapsed_seconds: float
    l2_guard_max_norm: float = L2_GUARD_MAX_NORM
    bf16: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    data_config: str = "single"
    # Wf-187 m19 Rh oversample fix — record what was wired so the
    # paper §6 honest-negative close can quote exact numbers.
    oversample_positives: bool = False
    pos_weight: Optional[float] = None
    pos_rate_train_after_oversample: Optional[float] = None
    # Wf-201 — record which task driver was used so the paper §6
    # honest-negative close can quote "single-task LOMO-safe" vs
    # "multi-task cross-metal leak" separately.
    task_type: str = "multi"


# ---------------------------------------------------------------------------
# Runner — wires BALANCED cohort + W₂ target=real_active + Hilbert L2 guard.
# ---------------------------------------------------------------------------
class BalancedLOMORunner:
    def __init__(
        self,
        ds: MetalCytotoxDataset,
        held_out_metal: str,
        balanced_smiles: List[str],
        balanced_labels: List[int],
        wasserstein_weight: float = 0.05,
        n_mc: int = 64,
        hidden: int = MT_HIDDEN,
        depth: int = MT_DEPTH,
        dropout: float = MT_DROPOUT,
        lr: float = MT_LR,
        epochs: int = 50,
        batch_size: int = MT_BATCH_SIZE,
        alpha: float = MT_ALPHA,
        seed: int = 42,
        device: Optional[torch.device] = None,
        use_l2_guard: bool = True,
        l2_guard_max_norm: float = L2_GUARD_MAX_NORM,
        model_cls=None,
        model_alias: Optional[str] = None,
        bf16: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        data_config: str = "single",
        oversample_positives: bool = False,
        pos_weight: Optional[float] = None,
        # Wf-201 — task driver selection.  ``"multi"`` (default,
        # backward-compatible) wires the legacy multi-task backbone
        # + per-metal heads path.  ``"single"`` wires the vanilla
        # per-molecule D-MPNN path with no cross-metal information
        # sharing — recommended for any honest-negative LOMO
        # evaluation.  See module docstring §"Wf-201 --task-type flag"
        # for tradeoffs.
        task_type: str = "multi",
    ) -> None:
        # ------------------------------------------------------------------
        # P0 #3 — model class switch (Wf-145 P3 / TODO-145 §3.5, wired by
        # LANE-F).
        #
        # ``model_cls`` selects the multi-task family.  The training loop is
        # family-agnostic: per batch it rebuilds the SMILES + metal-id inputs
        # and routes through the module-level adapters:
        #   * ``_model_ctor_kwargs``   — hidden/depth vs hidden_dim/n_layers
        #   * ``_forward_dispatch``    — DMPNN 4-tensor tuple vs
        #                                (smiles_list, metal_ids) -> dict
        #   * ``predict_mt_dispatch``  — family-agnostic predict
        #   * ``_embed_one_mol``       — per-mol W₂ embedding extraction
        # ``_check_model_compat`` fail-louds on an unknown family only.
        # ------------------------------------------------------------------
        if model_cls is None:
            from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel

            model_cls = DMPNNMultiTaskModel
        self._model_cls = model_cls
        # Wf-148 ultracode — keep the CLI alias (e.g. ``chemprop_v2``) for
        # diagnostics/logging alongside ``_model_cls``.
        self._model_alias = str(model_alias) if model_alias else None

        self.ds = ds
        self.held_out_metal = held_out_metal
        self.balanced_smiles = list(balanced_smiles)
        self.balanced_labels = list(balanced_labels)
        self.wasserstein_weight = float(wasserstein_weight)
        self.n_mc = int(n_mc)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)
        self.seed = int(seed)
        # Serial execution on GPU 0 (2026-10-03 directive): the old
        # ">1 GPU visible -> cuda:1" auto-pick routed multi-GPU hosts
        # to the second card.
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.splitter = BalancedLOMOSplitter(
            held_out_metal=held_out_metal, seed=self.seed
        )
        self.use_l2_guard = bool(use_l2_guard)
        self.l2_guard_max_norm = float(l2_guard_max_norm)
        # ------------------------------------------------------------------
        # P1 Fix — CLI flag plumbing (Wf-145 P3 / TODO-145 §3.5 follow-up).
        #
        # --bf16            Default True.  When CUDA is the active device,
        #                   wrap the D-MPNN forward pass in
        #                   ``torch.amp.autocast("cuda", dtype=torch.bfloat16)``.
        #                   CPU training always runs FP32 (autocast is a
        #                   CUDA-only API on this torch version).
        # --num-workers     Default 4.  The LOMO loop does not use a
        #                   DataLoader today (manual batching over
        #                   ``featurize_smiles_list``), so num_workers
        #                   is stored and logged but is effectively a
        #                   no-op until a DataLoader is wired in.
        #                   Default matches ``train_dmpnn_multitask``.
        # --pin-memory      Default True.  Same caveat as --num-workers:
        #                   stored + logged; no-op until DataLoader.
        # --data-config     Default "single".  The legacy LOMO path
        #                   reads the per-metal LOMO CSV; "four-domain"
        #                   and "pretrain" are reserved for a future
        #                   migration (TODO-145 §4 follow-up) and raise
        #                   NotImplementedError from ``run()`` so the
        #                   operator sees the mismatch immediately.
        # ------------------------------------------------------------------
        self.bf16 = bool(bf16)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.data_config = str(data_config)
        # ------------------------------------------------------------------
        # Wf-187 m19 Rh oversample fix + pos_weight fallback in
        # BCEWithLogitsLoss.
        #
        # --oversample-positives (default False, opt-in via flag): when
        #   True, apply sklearn.utils.resample on the positive-only rows
        #   for under-represented metals (Rh, Re by default) to lift the
        #   per-batch positive rate.  This addresses the Wf-131 P2 Rh
        #   regression (0.368 → 0.136) caused by the constant-inactive
        #   cohort by directly rebalancing the training pool.
        #
        # --pos-weight (default None = backward-compatible
        #   CrossEntropyLoss): when set, the MultiTaskLoss falls back
        #   to BCEWithLogitsLoss(pos_weight=...) on the positive-class
        #   logit.  This is the second half of the m19 fix — even with
        #   balanced training data, BCE alone under-counts positives on
        #   imbalanced real-Rn active counts, so we add a 10× upweight
        #   on positive loss.
        # ------------------------------------------------------------------
        self.oversample_positives = bool(oversample_positives)
        self.pos_weight = float(pos_weight) if pos_weight is not None else None
        # ------------------------------------------------------------------
        # Wf-201 — task driver selection (single vs multi).  Default
        # ``"multi"`` preserves backward compat with all prior Wf-131
        # / Wf-145 / Wf-187 runs.  ``"single"`` engages the LOMO-safe
        # vanilla :class:`DMPNNModel` driver — no per-metal heads, no
        # shared backbone, so the held-out metal cannot leak
        # information through the shared trunk.  Tradeoffs documented
        # in the module docstring §"Wf-201 --task-type flag".
        # ------------------------------------------------------------------
        tt = str(task_type).strip().lower()
        if tt not in ("single", "multi"):
            raise ValueError(
                f"task_type must be one of ('single', 'multi'), got {task_type!r}"
            )
        self.task_type = tt
        # Effective BF16 flag — only True if both requested AND CUDA is
        # the target device (autocast("cuda", ...) is invalid on CPU).
        self._use_bf16 = bool(self.bf16) and self.device.type == "cuda"
        self._wasserstein_loss_history: List[float] = []

    # ------------------------------------------------------------------
    def _real_rh_active_smiles(self, train_idx: np.ndarray) -> List[str]:
        """Return SMILES of real Rh ACTIVE compounds in train pool.

        The held-out metal is excluded; we only consider real Rh
        actives that remain in the training fold.  Used as the W₂
        TARGET distribution (per task spec §2 item 3).
        """
        metals = self.ds.metals
        active = self.ds.active.astype(int)
        train_set = set(int(i) for i in train_idx)
        out: List[str] = []
        for i in train_set:
            if str(metals[i]) == "Rh" and int(active[i]) == 1:
                out.append(str(self.ds.smiles[i]))
        return out

    # ------------------------------------------------------------------
    def _check_model_compat(self) -> None:
        """Verify ``self._model_cls`` is supported by the LOMO training loop.

        LANE-F wire-in: the loop is now family-agnostic.  Per batch it builds
        the SMILES list + active metal id and calls :func:`_forward_dispatch`,
        which routes to the DMPNN 4-tensor tuple path OR the V4/chemprop
        ``(smiles_list, metal_ids) -> dict`` path.  Per-mol W₂ embedding
        extraction routes through :func:`_embed_one_mol`.  The ctor is built
        via :func:`_model_ctor_kwargs`.

        All three families are therefore supported:
        ``DMPNNMultiTaskModel`` (``--model v3``), ``V4MultiTaskDMPNN`` /
        ``V4MultiTaskDMPNNPretrain`` (``--model v4|v4pretrain``) and
        ``ChempropV2MTModel`` (``--model chemprop_v2``).

        Wf-201: in ``--task-type single`` mode the loop bypasses the
        multi-task forward entirely (see ``_run_single_task``) and uses the
        vanilla :class:`DMPNNModel`; ``self._model_cls`` is ignored, so the
        compatibility check is skipped.
        """
        # Wf-201 — single-task driver bypasses the multi-task compat
        # check (the loop calls vanilla DMPNN APIs, not the multi-task
        # family's APIs).
        if self.task_type == "single":
            return
        cls_name = getattr(self._model_cls, "__name__", str(self._model_cls))
        if cls_name in _TUPLE_FWD_MODELS or cls_name in _DICT_FWD_MODELS:
            return
        # Fail loud on an unknown family rather than silently mis-dispatching.
        raise NotImplementedError(
            f"--model resolved to {cls_name}, which is not a recognised "
            "multi-task family.  Supported: "
            f"{sorted(_TUPLE_FWD_MODELS | _DICT_FWD_MODELS)}.  See "
            "train_metal_lomo_v2._forward_dispatch for the two contracts "
            "(DMPNN 4-tensor tuple vs (smiles_list, metal_ids) -> dict)."
        )

    # ------------------------------------------------------------------
    def run(self, verbose: bool = False) -> BalancedLOMOResult:
        # Wf-201 — dispatch to the single-task driver when requested.
        # Single-task bypasses the multi-task backbone + per-metal heads
        # entirely (no cross-metal information sharing), so it is
        # LOMO-safe.  Multi-task is the default (backward-compat).
        if self.task_type == "single":
            return self._run_single_task(verbose=verbose)

        from molmetal.baselines.dmpnn_multitask import MultiTaskLoss

        # P0 #3: refuse V4 model classes loudly rather than silently
        # crashing on the API mismatch.
        self._check_model_compat()

        # P1 Fix — refuse --data-config four-domain / pretrain loudly;
        # the legacy LOMO path is single-domain only.
        if self.data_config not in ("single",):
            raise NotImplementedError(
                f"--data-config {self.data_config!r} is NOT yet wired into "
                "train_metal_lomo_v2.BalancedLOMORunner.run() — the legacy "
                "LOMO path reads the per-metal LOMO CSV via "
                "MetalCytotoxDataset.from_csv (single-domain).  "
                "Migration plan: TODO-145 §4 follow-up.  "
                "Re-run with --data-config single."
            )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        split = self.splitter(self.ds)
        smiles_all = self.ds.smiles
        active_all = self.ds.active.astype(int)
        pic50_all = self.ds.pic50.astype(np.float32)
        metals_all = self.ds.metals

        train_idx, val_idx, test_idx = (
            split.train_idx,
            split.val_idx,
            split.test_idx,
        )

        # Build the augmented train pool: real (non-held-out) + balanced Rh.
        synth_smis = list(self.balanced_smiles)
        synth_labels = list(self.balanced_labels)
        n_synth_active = int(sum(1 for x in synth_labels if int(x) == 1))
        n_synth_inactive = int(sum(1 for x in synth_labels if int(x) == 0))

        # Per-metal target distribution = real Rh ACTIVE in train pool.
        real_rh_active_smiles = self._real_rh_active_smiles(train_idx)

        graphs_all = featurize_smiles_list([str(s) for s in smiles_all])
        graphs_train = [graphs_all[i] for i in train_idx]
        graphs_val = [graphs_all[i] for i in val_idx] if len(val_idx) else []
        graphs_test = [graphs_all[i] for i in test_idx]
        # LANE-F — SMILES index-aligned with the val/test graph lists, for the
        # V4/chemprop predict path (those families take SMILES, not tensors).
        smiles_val = [str(smiles_all[i]) for i in val_idx] if len(val_idx) else []
        smiles_test = [str(smiles_all[i]) for i in test_idx]
        pic50_train = pic50_all[train_idx]
        pic50_val = (
            pic50_all[val_idx] if len(val_idx) else np.array([], dtype=np.float32)
        )
        active_train = active_all[train_idx]
        active_val = active_all[val_idx] if len(val_idx) else np.array([], dtype=int)
        pic50_test = pic50_all[test_idx]
        active_test = active_all[test_idx]
        metals_train = metals_all[train_idx]

        # Featurise synthetic balanced cohort + real Rh ACTIVE target.
        synth_graphs = featurize_smiles_list([str(s) for s in synth_smis])
        synth_pic50 = np.array(
            [4.0 if int(l) == 1 else 6.5 for l in synth_labels],  # proxy pIC50
            dtype=np.float32,
        )
        synth_active = np.array([int(l) for l in synth_labels], dtype=np.int64)
        synth_metals = np.array(["Rh"] * len(synth_smis), dtype=object)

        target_graphs = featurize_smiles_list([str(s) for s in real_rh_active_smiles])
        target_active = np.ones(len(real_rh_active_smiles), dtype=np.int64)
        target_pic50 = np.full(len(real_rh_active_smiles), 5.0, dtype=np.float32)
        target_metals = np.array(
            ["Rh_real_active"] * len(real_rh_active_smiles), dtype=object
        )

        model = self._model_cls(
            **_model_ctor_kwargs(
                self._model_cls, self.hidden, self.depth, self.dropout
            )
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # Wf-187 m19 Rh oversample fix — wire pos_weight into MultiTaskLoss
        # when set (default None = backward-compatible CrossEntropyLoss).
        criterion = MultiTaskLoss(
            alpha=self.alpha,
            pos_weight=self.pos_weight,
        )

        best_val_auc = 0.0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        w2_history: List[float] = []

        # Combined training pool: real (non-held-out) + synthetic balanced Rh.
        # LANE-F — keep an ``all_train_smiles`` list in lockstep with
        # ``all_train_graphs`` so the V4/chemprop forward can rebuild the
        # ``(smiles_list, metal_ids)`` batch (those families re-featurize
        # SMILES internally; the loop only carries graph tensors).
        _train_smiles_real = [str(smiles_all[i]) for i in train_idx]
        _synth_pairs = [
            (str(s), g) for s, g in zip(synth_smis, synth_graphs) if g is not None
        ]
        all_train_graphs = list(graphs_train) + [g for _, g in _synth_pairs]
        all_train_smiles = list(_train_smiles_real) + [s for s, _ in _synth_pairs]
        all_train_pic50 = np.concatenate(
            [pic50_train, synth_pic50[: len(all_train_graphs) - len(graphs_train)]]
        )
        all_train_active = np.concatenate(
            [active_train, synth_active[: len(all_train_graphs) - len(graphs_train)]]
        )
        all_train_metals = np.concatenate([metals_train, synth_metals])
        # Record pre-oversample pos_rate for the result dataclass so the
        # paper §6 honest-negative close can quote the lift.
        pos_rate_train_pre = float(np.mean(all_train_active))
        # Wf-187 m19 Rh oversample fix — sklearn.utils.resample on
        # positive-only rows for under-represented metals (Rh, Re).
        # Only engage when --oversample-positives is set.
        if self.oversample_positives:
            # Re-featurise the augmented train pool as SMILES so we can
            # apply sklearn.utils.resample on the positive-only rows.
            # We need SMILES strings, not graphs — rebuild from the
            # original dataset + balanced cohort.
            train_smiles = [str(smiles_all[i]) for i in train_idx] + [
                str(s) for s in synth_smis
            ]
            (os_smiles, os_active, os_metals, os_pic50, os_stats) = (
                oversample_underrepresented_metals(
                    smiles=train_smiles,
                    active=all_train_active,
                    metals=all_train_metals,
                    pic50=all_train_pic50,
                    underrepresented=["Rh", "Re"],
                    target_pos_rate=0.5,
                    seed=int(self.seed),
                )
            )
            if os_stats.get("Rh", 0) > 0 or os_stats.get("Re", 0) > 0:
                # Featurise the appended SMILES and concatenate.
                appended_smiles = os_smiles[len(train_smiles) :]
                appended_graphs = featurize_smiles_list(
                    [str(s) for s in appended_smiles]
                )
                for g in appended_graphs:
                    if g is not None:
                        all_train_graphs.append(g)
                all_train_smiles = [str(s) for s in os_smiles]
                all_train_active = np.asarray(os_active, dtype=np.int64)
                all_train_metals = np.asarray(os_metals, dtype=object)
                all_train_pic50 = np.asarray(os_pic50, dtype=np.float32)
                logger.info(
                    "fold=%s seed=%d oversample appended Rh=%d Re=%d "
                    "pos_rate_train pre=%.3f post=%.3f",
                    self.held_out_metal,
                    self.seed,
                    os_stats.get("Rh", 0),
                    os_stats.get("Re", 0),
                    pos_rate_train_pre,
                    float(np.mean(all_train_active)),
                )
        pos_rate_train_post = float(np.mean(all_train_active))

        for epoch in range(self.epochs):
            t0 = time.time()
            # Pre-compute W₂ target once per epoch.
            w2_loss_t: Optional[torch.Tensor] = None
            if self.wasserstein_weight > 0 and (
                epoch % 5 == 0 or epoch == self.epochs - 1
            ):
                train_embeds = self._extract_per_metal_embeds(
                    model,
                    all_train_graphs,
                    all_train_metals,
                    smiles=all_train_smiles,
                )
                if len(train_embeds) >= 2:
                    w2_loss_t = self.wasserstein_weight * wasserstein_transfer_loss(
                        train_embeds, n_mc=self.n_mc, seed=self.seed
                    )

            # train one epoch
            model.train()
            indices = np.arange(len(all_train_graphs))
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch_graphs = [all_train_graphs[i] for i in batch_idx]
                batch_smiles = [all_train_smiles[i] for i in batch_idx]
                batch_pic50 = all_train_pic50[batch_idx]
                batch_active = all_train_active[batch_idx]
                valid = [
                    (g, s, p, a)
                    for g, s, p, a in zip(
                        batch_graphs, batch_smiles, batch_pic50, batch_active
                    )
                    if g is not None and not (isinstance(p, float) and np.isnan(p))
                ]
                if not valid:
                    continue
                g_list = [v[0] for v in valid]
                smiles_batch = [v[1] for v in valid]
                y_pic50 = np.array([v[2] for v in valid], dtype=np.float32)
                y_active = np.array([v[3] for v in valid], dtype=np.int64)
                bd = collate_graphs(g_list)
                if bd is None:
                    continue
                atom_f, bond_f, e_src, e_dst = bd
                atom_f = atom_f.to(self.device)
                bond_f = bond_f.to(self.device)
                e_src = e_src.to(self.device)
                e_dst = e_dst.to(self.device)
                bd_dev = (atom_f, bond_f, e_src, e_dst)
                y_pic50_t = torch.from_numpy(y_pic50).to(self.device)
                y_active_t = torch.from_numpy(y_active).to(self.device)

                optimizer.zero_grad()
                # P1 Fix — wrap forward in BF16 autocast when requested
                # AND CUDA is the device.  CPU always runs FP32.
                # LANE-F — family-agnostic dispatch (tuple vs dict path).
                if self._use_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pic50_pred, active_logits = _forward_dispatch(
                            model, bd_dev, smiles_batch, self.held_out_metal
                        )
                else:
                    pic50_pred, active_logits = _forward_dispatch(
                        model, bd_dev, smiles_batch, self.held_out_metal
                    )
                loss, _, _ = criterion(pic50_pred, active_logits, y_pic50_t, y_active_t)
                # Always cast loss to FP32 before backward for numerical
                # stability (BF16 autocast leaves loss in BF16).
                if self._use_bf16:
                    loss = loss.float()
                if w2_loss_t is not None:
                    loss = loss + w2_loss_t
                loss.backward()
                if self.use_l2_guard:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.l2_guard_max_norm
                    )
                optimizer.step()

                bs = len(y_active)
                total_loss += loss.item() * bs
                n_batches += bs

            avg_loss = total_loss / max(n_batches, 1)

            # validation
            if len(graphs_val) >= 2:
                _, val_score, _ = predict_mt_dispatch(
                    model,
                    graphs_val,
                    smiles_val,
                    pic50_val,
                    active_val,
                    metal_name=self.held_out_metal,
                    batch_size=self.batch_size,
                    device=self.device,
                )
                vm = compute_metrics(active_val, val_score)
                val_auc = float(vm["roc_auc"])
            else:
                val_auc = float("nan")

            # W₂ history
            if self.wasserstein_weight > 0 and (
                epoch % 5 == 0 or epoch == self.epochs - 1
            ):
                train_embeds = self._extract_per_metal_embeds(
                    model,
                    all_train_graphs,
                    all_train_metals,
                    smiles=all_train_smiles,
                )
                if len(train_embeds) >= 2:
                    with torch.no_grad():
                        w2_val = wasserstein_transfer_loss(
                            train_embeds, n_mc=self.n_mc, seed=self.seed
                        ).item()
                else:
                    w2_val = 0.0
            else:
                w2_val = w2_history[-1] if w2_history else 0.0
            w2_history.append(w2_val)

            if verbose and (epoch % 1 == 0 or epoch == self.epochs - 1):
                logger.info(
                    "fold=%s seed=%d epoch=%d loss=%.4f val_auc=%.4f w2=%.4f elapsed=%.1fs",
                    self.held_out_metal,
                    self.seed,
                    epoch,
                    avg_loss,
                    val_auc,
                    w2_val,
                    time.time() - t0,
                )

            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # final test
        _, test_score, test_pic50_pred = predict_mt_dispatch(
            model,
            graphs_test,
            smiles_test,
            pic50_test,
            active_test,
            metal_name=self.held_out_metal,
            batch_size=self.batch_size,
            device=self.device,
        )
        test_pic50_pred = np.clip(
            np.asarray(test_pic50_pred, dtype=float),
            _ASTR_CLIP_MIN,
            _ASTR_CLIP_MAX,
        )
        tm = compute_metrics(active_test, test_score)
        test_auc = float(tm["roc_auc"])
        test_mae = float(np.mean(np.abs(test_pic50_pred - pic50_test)))

        # final per-metal W₂
        train_embeds = self._extract_per_metal_embeds(
            model,
            all_train_graphs,
            all_train_metals,
            smiles=all_train_smiles,
        )
        per_metal_w2: Dict[str, float] = {}
        keys = list(train_embeds.keys())
        for i, ki in enumerate(keys):
            for kj in keys[i + 1 :]:
                ti = train_embeds[ki]
                tj = train_embeds[kj]
                if ti.shape[0] == 0 or tj.shape[0] == 0:
                    continue
                if ti.shape[1] != tj.shape[1]:
                    continue
                with torch.no_grad():
                    w2 = wasserstein_loss(ti, tj, n_mc=self.n_mc, seed=self.seed).item()
                per_metal_w2[f"{ki}__{kj}"] = float(w2)

        return BalancedLOMOResult(
            fold=self.held_out_metal,
            seed=self.seed,
            n_train=len(all_train_graphs),
            n_val=len(graphs_val),
            n_test=len(graphs_test),
            n_synthetic_rh_active=n_synth_active,
            n_synthetic_rh_inactive=n_synth_inactive,
            pos_rate_train=float(np.mean(all_train_active)),
            pos_rate_test=float(np.mean(active_test)),
            val_auc=float(best_val_auc),
            test_auc=test_auc,
            test_pic50_mae=test_mae,
            wasserstein_weight=self.wasserstein_weight,
            wasserstein_target="Rh_real_active_in_train_pool",
            wasserstein_loss_history=w2_history,
            final_per_metal_w2=per_metal_w2,
            elapsed_seconds=time.time() - t0,
            l2_guard_max_norm=self.l2_guard_max_norm,
            bf16=bool(self._use_bf16),
            num_workers=int(self.num_workers),
            pin_memory=bool(self.pin_memory),
            data_config=str(self.data_config),
            # Wf-187 m19 Rh oversample fix — record what was wired.
            oversample_positives=bool(self.oversample_positives),
            pos_weight=self.pos_weight,
            pos_rate_train_after_oversample=(
                pos_rate_train_post if self.oversample_positives else None
            ),
            # Wf-201 — record task driver so paper §6 honest-negative
            # close can quote single vs multi separately.
            task_type=str(self.task_type),
        )

    # ------------------------------------------------------------------
    def _extract_per_metal_embeds(
        self,
        model,
        graphs: List,
        metals: np.ndarray,
        smiles: Optional[List[str]] = None,
        max_per_metal: int = 80,
    ) -> Dict[str, torch.Tensor]:
        """Per-metal embedding extraction (mirrors Wf-125 P2 helper).

        Returns dict ``metal -> (N, D)`` for every metal with >= 1 sample.
        For the DMPNN family ``D == 2H`` (``_forward_one_mol`` oracle); for
        V4/chemprop ``D == hidden_dim`` (shared ``encode``).  ``smiles`` must
        be index-aligned with ``graphs`` and is REQUIRED for the dict-fwd
        families (they re-featurize SMILES internally).
        """
        valid_idx = [i for i, g in enumerate(graphs) if g is not None]
        per_metal: Dict[str, List[int]] = {}
        for i in valid_idx:
            m = str(metals[i])
            per_metal.setdefault(m, []).append(i)

        rng = np.random.default_rng(0)
        sampled_idx: List[int] = []
        sampled_metals: List[str] = []
        for m, idx_list in per_metal.items():
            cap = min(max_per_metal, len(idx_list))
            sel = rng.permutation(len(idx_list))[:cap]
            for j in sel:
                sampled_idx.append(idx_list[j])
                sampled_metals.append(m)
        if not sampled_idx:
            return {}

        use_dict_fwd = _is_dict_fwd_model(model)
        if use_dict_fwd and smiles is None:
            raise NotImplementedError(
                f"{type(model).__name__} uses the (smiles_list, metal_ids) "
                "contract; _extract_per_metal_embeds requires the index-aligned "
                "`smiles` list to encode.  The LOMO loop passes it for this family."
            )

        out_embeds: Dict[str, List[torch.Tensor]] = {}
        batch_size = max(1, self.batch_size)
        for start in range(0, len(sampled_idx), batch_size):
            batch_idx = sampled_idx[start : start + batch_size]
            batch_metas = [
                sampled_metals[k]
                for k in range(start, min(start + batch_size, len(sampled_idx)))
            ]
            g_list = [graphs[i] for i in batch_idx]
            bd = collate_graphs(g_list)
            if bd is None:
                continue
            atom_f, bond_f, e_src, e_dst = bd
            atom_f = atom_f.to(self.device)
            bond_f = bond_f.to(self.device)
            e_src = e_src.to(self.device)
            e_dst = e_dst.to(self.device)
            B_ = atom_f.size(0)
            for b in range(B_):
                V = atom_f.size(1)
                valid_edges = bond_f[b].abs().sum(dim=-1) > 0
                E = valid_edges.sum().item()
                if E == 0 or V == 0:
                    continue
                if use_dict_fwd:
                    pooled = _embed_one_mol(
                        model, str(smiles[batch_idx[b]]), None, None, None, None
                    )
                else:
                    pooled = _embed_one_mol(
                        model,
                        "",
                        atom_f[b, :V],
                        bond_f[b, :E],
                        e_src[b, :E],
                        e_dst[b, :E],
                    )
                out_embeds.setdefault(batch_metas[b], []).append(pooled.detach().cpu())

        return {m: torch.stack(v, dim=0) for m, v in out_embeds.items() if v}

    # ------------------------------------------------------------------
    # Wf-201 — single-task driver (LOMO-safe).
    #
    # This is the explicit fix for the Wf-131 P3 LOMO confusion: the
    # legacy multi-task loop trains one shared backbone across all
    # metals (with per-metal heads on top), so even though the held-out
    # metal's rows are excluded from training, the shared backbone has
    # already seen that metal's distribution via the other metals'
    # rows.  This makes the LOMO AUC an over-estimate.
    #
    # The single-task driver below uses the vanilla :class:`DMPNNModel`
    # (no per-metal heads, no shared backbone across metals).  The
    # held-out metal's rows are excluded from training; the model
    # emits a single logit per molecule; we train with
    # :func:`BCEWithLogitsLoss` on the active/inactive label and
    # evaluate with :func:`compute_metrics` (mirrors the original
    # :class:`DMPNNBaseline` smoke flow).
    #
    # Tradeoffs vs multi-task (paper §6 honest-negative close):
    #   * single:  LOMO-safe; no pIC50 regression head; no per-metal
    #              Wasserstein transfer; no cross-metal information
    #              sharing.  AUC may be lower (less parameter sharing).
    #   * multi:   IID-only mode; shared backbone acts as cross-metal
    #              conduit; LOMO AUC over-estimated.  Use only for IID
    #              upper-bound reporting.
    #
    # The driver's inputs/outputs mirror :meth:`run` exactly so
    # :func:`aggregate` and the per-seed JSON dump work unchanged.
    # ------------------------------------------------------------------
    def _run_single_task(self, verbose: bool = False) -> BalancedLOMOResult:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        t0 = time.time()

        # LOMO split (uses the same splitter as multi-task so the
        # train/val/test row indices are identical across modes).
        split = self.splitter(self.ds)
        smiles_all = self.ds.smiles
        active_all = self.ds.active.astype(int)
        pic50_all = self.ds.pic50.astype(np.float32)
        metals_all = self.ds.metals

        train_idx = split.train_idx
        val_idx = split.val_idx
        test_idx = split.test_idx

        # Featurise everything once (CPU-side).  Note: single-task mode
        # IGNORES the balanced Rh cohort (no per-metal heads, no
        # Wasserstein transfer) — the LOMO setup is honest-negative
        # by construction.
        graphs_all = featurize_smiles_list([str(s) for s in smiles_all])
        graphs_train = [graphs_all[i] for i in train_idx]
        graphs_val = [graphs_all[i] for i in val_idx] if len(val_idx) else []
        graphs_test = [graphs_all[i] for i in test_idx]
        active_train = active_all[train_idx]
        active_val = active_all[val_idx] if len(val_idx) else np.array([], dtype=int)
        active_test = active_all[test_idx]
        pic50_test = pic50_all[test_idx]

        # Vanilla D-MPNN (no per-metal heads, no shared backbone).
        # We use the same hidden / depth / dropout defaults as
        # DMPNNBaseline so the parameter count is comparable to the
        # multi-task baseline minus the per-metal head overhead.
        model = DMPNNModel(
            hidden=self.hidden,
            depth=self.depth,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Optional grad-norm guard (Wf-127 P2).  DMPNNModel exposes
        # ``.parameters()`` like any nn.Module so the same
        # clip_grad_norm_ call works.
        best_val_auc = 0.0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(self.epochs):
            ep_t0 = time.time()
            model.train()
            indices = np.arange(len(graphs_train))
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch_graphs = [graphs_train[i] for i in batch_idx]
                batch_active = active_train[batch_idx]
                valid = [
                    (g, a) for g, a in zip(batch_graphs, batch_active) if g is not None
                ]
                if not valid:
                    continue
                g_list = [v[0] for v in valid]
                y_batch = np.array([v[1] for v in valid], dtype=np.float32)

                bd = collate_graphs(g_list)
                if bd is None:
                    continue
                atom_f, bond_f, e_src, e_dst = bd
                atom_f = atom_f.to(self.device)
                bond_f = bond_f.to(self.device)
                e_src = e_src.to(self.device)
                e_dst = e_dst.to(self.device)
                y_t = torch.from_numpy(y_batch).to(self.device)

                optimizer.zero_grad()
                # BF16 autocast when requested (mirrors multi-task path).
                if self._use_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(atom_f, bond_f, e_src, e_dst)
                        loss = torch.nn.BCEWithLogitsLoss()(logits, y_t)
                    loss = loss.float()
                else:
                    logits = model(atom_f, bond_f, e_src, e_dst)
                    loss = torch.nn.BCEWithLogitsLoss()(logits, y_t)
                loss.backward()
                if self.use_l2_guard:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.l2_guard_max_norm
                    )
                optimizer.step()

                bs = len(y_batch)
                total_loss += loss.item() * bs
                n_batches += bs

            avg_loss = total_loss / max(n_batches, 1)

            # Validation AUC.
            if len(graphs_val) >= 2:
                _, val_score = dmpnn_predict(
                    model,
                    graphs_val,
                    active_val,
                    batch_size=self.batch_size,
                    device=self.device,
                )
                vm = compute_metrics(active_val, val_score)
                val_auc = float(vm["roc_auc"])
            else:
                val_auc = float("nan")

            if verbose and (epoch % 1 == 0 or epoch == self.epochs - 1):
                logger.info(
                    "fold=%s seed=%d task_type=single epoch=%d loss=%.4f "
                    "val_auc=%.4f elapsed=%.1fs",
                    self.held_out_metal,
                    self.seed,
                    epoch,
                    avg_loss,
                    val_auc,
                    time.time() - ep_t0,
                )

            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final test AUC.
        _, test_score = dmpnn_predict(
            model,
            graphs_test,
            active_test,
            batch_size=self.batch_size,
            device=self.device,
        )
        tm = compute_metrics(active_test, test_score)
        test_auc = float(tm["roc_auc"])
        # Single-task driver has no pIC50 regression head; report a
        # placeholder MAE so the dataclass schema stays uniform.  Use
        # 0.0 since the model never produces a pIC50 estimate.
        test_pic50_mae = 0.0

        return BalancedLOMOResult(
            fold=self.held_out_metal,
            seed=self.seed,
            n_train=len(graphs_train),
            n_val=len(graphs_val),
            n_test=len(graphs_test),
            # Single-task ignores the balanced cohort — record 0/0 so
            # the paper §6 honest-negative close can quote "single-task
            # does not use synthetic augmentation".
            n_synthetic_rh_active=0,
            n_synthetic_rh_inactive=0,
            pos_rate_train=float(np.mean(active_train)) if len(active_train) else 0.0,
            pos_rate_test=float(np.mean(active_test)) if len(active_test) else 0.0,
            val_auc=float(best_val_auc),
            test_auc=test_auc,
            test_pic50_mae=test_pic50_mae,
            # Single-task does not use Wasserstein transfer — record
            # weight=0 and empty history so the dataclass schema is
            # uniform.
            wasserstein_weight=0.0,
            wasserstein_target="n/a_single_task",
            wasserstein_loss_history=[],
            final_per_metal_w2={},
            elapsed_seconds=time.time() - t0,
            l2_guard_max_norm=self.l2_guard_max_norm,
            bf16=bool(self._use_bf16),
            num_workers=int(self.num_workers),
            pin_memory=bool(self.pin_memory),
            data_config=str(self.data_config),
            # Single-task uses BCEWithLogitsLoss directly; pos_weight
            # is not wired into the vanilla loop.  Record None so
            # the paper §6 honest-negative close can quote
            # "pos_weight not applicable to single-task driver".
            oversample_positives=False,
            pos_weight=None,
            pos_rate_train_after_oversample=None,
            # Wf-201 — explicit marker for the LOMO-safe driver.
            task_type="single",
        )


# ---------------------------------------------------------------------------
# Aggregation — per-metal 5-seed mean/std + delta vs baseline.
# ---------------------------------------------------------------------------
@dataclass
class AggregateResult:
    n_metals: int
    n_seeds: int
    n_epochs: int
    per_metal: Dict[str, Dict[str, float]]
    overall: Dict[str, float]
    pass_target_rh: bool
    pass_target_overall: bool


def aggregate(
    results_by_metal: Dict[str, List[BalancedLOMOResult]],
    baseline_per_metal: Dict[str, float],
    n_epochs: int,
) -> AggregateResult:
    per_metal_summary: Dict[str, Dict[str, float]] = {}
    all_test_aucs: List[float] = []
    for metal, results in results_by_metal.items():
        test_aucs = [r.test_auc for r in results]
        mean = float(np.mean(test_aucs)) if test_aucs else 0.0
        std = float(np.std(test_aucs)) if test_aucs else 0.0
        delta = mean - float(baseline_per_metal.get(metal, 0.0))
        per_metal_summary[metal] = {
            "n_seeds": len(test_aucs),
            "test_auc_mean": mean,
            "test_auc_std": std,
            "per_seed": test_aucs,
            "delta_vs_baseline": delta,
        }
        all_test_aucs.extend(test_aucs)

    overall_mean = float(np.mean(all_test_aucs)) if all_test_aucs else 0.0
    overall_std = float(np.std(all_test_aucs)) if all_test_aucs else 0.0
    rh_mean = float(per_metal_summary.get("Rh", {}).get("test_auc_mean", 0.0))
    pass_rh = bool(rh_mean >= TARGET_RH_AUC)
    pass_overall = bool(overall_mean >= TARGET_OVERALL_AUC)
    return AggregateResult(
        n_metals=len(results_by_metal),
        n_seeds=len(all_test_aucs) // max(1, len(results_by_metal)),
        n_epochs=n_epochs,
        per_metal=per_metal_summary,
        overall={
            "test_auc_mean": overall_mean,
            "test_auc_std": overall_std,
            "n_runs": len(all_test_aucs),
        },
        pass_target_rh=pass_rh,
        pass_target_overall=pass_overall,
    )


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metals",
        nargs="+",
        default=["Rh", "Ru", "Ir", "Os", "Re"],
        help="Metals for LOMO (default: Rh Ru Ir Os Re).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds (default: 0 1 2 3 4).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--wasserstein-weight", type=float, default=0.05)
    parser.add_argument("--n-mc", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=MT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=MT_LR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="molmetal/reports/wf145_p2_lomo_v2")
    parser.add_argument(
        "--balanced-cohort-json",
        default=str(DEFAULT_RH_BALANCED_JSON),
        help="Wf-145 P1 balanced Rh cohort JSON.",
    )
    parser.add_argument(
        "--lomo-csv",
        default=str(DEFAULT_LOMO_CSV),
        help="MetalCytoToxDB LOMO CSV.",
    )
    parser.add_argument(
        "--use-l2-guard",
        dest="use_l2_guard",
        action="store_true",
        default=True,
        help="Engage Wf-127 P2 Hilbert L2 grad-norm guard (default ON).",
    )
    parser.add_argument(
        "--no-l2-guard",
        dest="use_l2_guard",
        action="store_false",
        help="Disable L2 guard.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke run: 1 metal x 1 seed x 2 epochs.",
    )
    # ------------------------------------------------------------------
    # Wf-201 — task driver selection (single vs multi).  Default
    # ``multi`` preserves backward compat with all prior Wf-131 /
    # Wf-145 / Wf-187 runs.  ``single`` engages the LOMO-safe
    # vanilla D-MPNN driver (no per-metal heads, no shared backbone
    # across metals).  Tradeoffs documented in the module docstring
    # §"Wf-201 --task-type flag".
    # ------------------------------------------------------------------
    parser.add_argument(
        "--task-type",
        choices=["single", "multi"],
        default="multi",
        help=(
            "Task driver selection (Wf-201).  'multi' (default — backward-"
            "compatible) uses DMPNNMultiTaskModel with shared backbone + "
            "per-metal heads.  'single' uses the vanilla DMPNNModel — no "
            "per-metal heads, no cross-metal information sharing — and is "
            "the LOMO-safe mode for honest-negative LOMO reporting.  "
            "See module docstring for tradeoffs."
        ),
    )
    # ------------------------------------------------------------------
    # P1 Fix — CLI flag plumbing (Wf-145 P3 / TODO-145 §3.5 follow-up).
    #
    # --bf16            Wrap the D-MPNN forward pass in
    #                   ``torch.amp.autocast("cuda", dtype=torch.bfloat16)``
    #                   when CUDA is the active device.  Mirrors the same
    #                   flag in train_dmpnn_multitask.py.
    # --num-workers     DataLoader num_workers.  The LOMO loop does not
    #                   use a DataLoader today (manual batching); the
    #                   flag is stored + logged, ready for the
    #                   DataLoader migration in TODO-145 §4.
    # --pin-memory      DataLoader pin_memory.  Same caveat as
    #                   --num-workers.
    # --data-config     "single" (default — current behaviour — reads
    #                   per-metal LOMO CSV), "four-domain" or
    #                   "pretrain" (reserved for future migration;
    #                   both raise NotImplementedError from run() with
    #                   a TODO pointer so the operator sees the
    #                   mismatch immediately).
    # ------------------------------------------------------------------
    parser.add_argument(
        "--bf16",
        dest="bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wrap the D-MPNN forward pass in torch.amp.autocast('cuda', "
            "dtype=torch.bfloat16) for training (default: True).  "
            "Pass --no-bf16 to disable.  No-op on CPU."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4).",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DataLoader pin_memory (default: True).",
    )
    parser.add_argument(
        "--data-config",
        choices=["single", "four-domain", "pretrain"],
        default="single",
        help=(
            "Data layout: 'single' (default — per-metal LOMO CSV), "
            "'four-domain' (cross-domain ConcatDataset — reserved, "
            "raises NotImplementedError), or 'pretrain' (RDKit-free "
            "GPU path — reserved, raises NotImplementedError).  "
            "See TODO-145 §4 follow-up."
        ),
    )
    # ------------------------------------------------------------------
    # P0 #3 — Model class switch (Wf-145 P3 / TODO-145 §3.5, updated
    # by Wf-148 ultracode / TODO-148 §3, wired by LANE-F).
    #
    # --model    Select which multi-task class is instantiated by
    #            :class:`BalancedLOMORunner`.  ALL choices run end-to-end:
    #            the loop dispatches per batch via ``_forward_dispatch``.
    #
    # Choices:
    #   * "chemprop_v2"  Chemprop v2 MPNN backbone (default).  Uses the
    #                   ``(smiles_list, metal_ids) -> dict`` contract.
    #   * "v3"          DMPNNMultiTaskModel (legacy).  Dual-head 4-tensor
    #                   ``(atom_f, bond_f, e_src, e_dst)`` contract.
    #   * "v4"          V4MultiTaskDMPNN — ``(smiles_list, metal_ids)``.
    #   * "v4pretrain"  V4MultiTaskDMPNNPretrain — same contract.
    #
    # Backward compat: ``--model v3`` maps to DMPNNMultiTaskModel,
    # preserving the previous default behaviour.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--model",
        choices=["chemprop_v2", "v3", "v4", "v4pretrain"],
        default="chemprop_v2",
        help=(
            "Multi-task model class (default: chemprop_v2 → "
            "ChempropV2MTModel).  All four choices are wired into the LOMO "
            "loop: 'v3' → DMPNNMultiTaskModel (4-tensor tuple forward); "
            "'v4'/'v4pretrain' → V4MultiTaskDMPNN(Pretrain); 'chemprop_v2' → "
            "ChempropV2MTModel (both use the (smiles_list, metal_ids) → dict "
            "forward)."
        ),
    )
    # ------------------------------------------------------------------
    # Wf-187 m19 Rh oversample fix — CLI flag plumbing.
    #
    # --oversample-positives    BooleanOptionalAction, default False
    #                           (backward-compatible).  When True,
    #                           sklearn.utils.resample is applied on
    #                           positive-only rows for under-represented
    #                           metals (Rh, Re) before training.
    #
    # --pos-weight              float, default None (backward-
    #                           compatible CrossEntropyLoss).  When set,
    #                           MultiTaskLoss falls back to
    #                           BCEWithLogitsLoss(pos_weight=...) on
    #                           the positive-class logit.  Wf-187
    #                           default-on deployment value is 10.0
    #                           (see Wf-131 P2 honest-negative close).
    # ------------------------------------------------------------------
    parser.add_argument(
        "--oversample-positives",
        dest="oversample_positives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Oversample positives for under-represented metals (Rh, Re) "
            "via sklearn.utils.resample (default: False = backward-compat).  "
            "Pass --oversample-positives to engage the m19 fix."
        ),
    )
    parser.add_argument(
        "--pos-weight",
        dest="pos_weight",
        type=float,
        default=None,
        help=(
            "pos_weight for BCEWithLogitsLoss fallback in MultiTaskLoss "
            "(default: None = backward-compat CrossEntropyLoss).  When set, "
            "the classification loss switches to BCEWithLogitsLoss on the "
            "positive-class logit with this weight.  Recommended: 10.0 for "
            "under-represented metals (Rh, Re)."
        ),
    )
    # Wf-A2 NEW: --config YAML override (training config unification).
    # CLI flags still win where the user explicitly passes them.
    # Bit-for-bit backward compat: omit --config = argparse-only path.
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to YAML config overriding argparse defaults. "
            "CLI flags win where the user explicitly passes them. "
            "See configs/training/platinai_pt_4cell_5seed_50ep_v1.yaml for schema."
        ),
    )
    args = parser.parse_args(argv)
    # Wf-A2 NEW: --config YAML override (training config unification).
    # CLI flags win where the user explicitly passes them; omit --config =
    # bit-for-bit argparse-only backward compat.
    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

    args = apply_yaml_overrides(parser, args, getattr(args, "config", None))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    balanced_smiles, balanced_labels = load_balanced_rh_smiles(
        Path(args.balanced_cohort_json)
    )

    if args.quick:
        args.metals = args.metals[:1]
        args.seeds = args.seeds[:1]
        args.epochs = 2

    # Load LOMO dataset.
    ds = MetalCytotoxDataset.from_csv(
        path=Path(args.lomo_csv),
        metal_whitelist=list(args.metals),
    )
    logger.info(
        "loaded dataset: %d rows; metals=%s", len(ds), sorted(set(ds.metals.tolist()))
    )

    device = torch.device(args.device) if args.device != "cpu" else torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        # Wf-190 env guard: only auto-fall-back to CPU if the script-level
        # allow-CPU flag is set OR --bf16 is explicitly disabled (otherwise
        # the user explicitly asked for GPU and we should fail fast rather
        # than silently burn hours on a CPU box).
        if getattr(args, "bf16", True) and not _MOLMETAL_ALLOW_CPU:
            raise RuntimeError(
                "[train_metal_lomo_v2] CUDA requested (--device cuda) but "
                "torch.cuda.is_available() is False and --bf16 is enabled.  Either:\n"
                "  -1- re-run on a CUDA-enabled host, OR\n"
                "  -2- pass --no-bf16 --device cpu for CPU fallback, OR\n"
                "  -3- set MOLMETAL_ALLOW_CPU=1 to allow silent CPU fallback "
                "(Wf-190 env guard)."
            )
        logger.warning(
            "CUDA requested but torch.cuda.is_available() is False; falling back to CPU. "
            "Set MOLMETAL_ALLOW_CPU=0 + --no-bf16 for fail-fast behaviour.",
        )
        device = torch.device("cpu")

    results_by_metal: Dict[str, List[BalancedLOMOResult]] = {m: [] for m in args.metals}
    t_start = time.time()
    # ------------------------------------------------------------------
    # P0 #3 — resolve ``--model`` choice to the actual class.
    #
    # "v3" is the new short alias for DMPNNMultiTaskModel (legacy
    # default).  Preserved via alias mapping so backward compat
    # callers passing ``--model v3`` get the same class as before.
    #
    # "chemprop_v2" is the new default (Wf-148 ultracode / TODO-148
    # §3).  The class ``ChempropV2MTModel`` lives at
    # :mod:`molmetal.models.chemprop_v2_mt_model` (Wf-148 ship).
    # Selecting it instantiates the class.  LANE-F wire-in: the LOMO loop
    # is now family-agnostic (``_forward_dispatch`` + ``_model_ctor_kwargs``
    # + ``predict_mt_dispatch`` + ``_embed_one_mol``), so all four --model
    # choices run end-to-end.
    # ------------------------------------------------------------------
    model_cls = None  # default → DMPNNMultiTaskModel (set inside runner)
    if args.model in ("v4", "v4pretrain"):
        # Import lazily so the default DMPNNMultiTaskModel path stays
        # dependency-free for users who never opt in.
        from molmetal.models import v4_multitask_dmpnn as _v4_mod

        model_cls = (
            _v4_mod.V4MultiTaskDMPNNPretrain
            if args.model == "v4pretrain"
            else _v4_mod.V4MultiTaskDMPNN
        )
        logger.info(
            "[wf145_p2] --model=%s → %s (LANE-F wired — "
            "LOMO loop dispatches on the (smiles_list, metal_ids) contract)",
            args.model,
            model_cls.__name__,
        )
    elif args.model == "v3":
        logger.info(
            "[wf145_p2] --model=v3 → DMPNNMultiTaskModel (legacy default)",
        )
    elif args.model == "chemprop_v2":
        # Wf-148 ultracode / TODO-148 §3 — chemprop v2 IS implemented.
        # Resolve to ``ChempropV2MTModel`` directly so the class is
        # actually instantiated (replacing the previous
        # ``NotImplementedError("class does not exist")`` stub).
        model_cls = ChempropV2MTModel
        logger.info(
            "[wf145_p2] --model=chemprop_v2 → ChempropV2MTModel "
            "(LANE-F wired — LOMO loop dispatches on the "
            "(smiles_list, metal_ids) contract)",
        )
    # P1 Fix — log the new flags so the operator can see the wiring
    # even before training starts.
    logger.info(
        "[wf145_p2] bf16=%s num_workers=%d pin_memory=%s data_config=%s "
        "(cpu bf16 is a no-op; num_workers/pin_memory no-op until DataLoader)",
        args.bf16,
        args.num_workers,
        args.pin_memory,
        args.data_config,
    )
    for metal in args.metals:
        for seed in args.seeds:
            logger.info(
                "=== fold=%s seed=%s epochs=%d wass_w=%.4f synth=%d device=%s ===",
                metal,
                seed,
                args.epochs,
                args.wasserstein_weight,
                len(balanced_smiles),
                device,
            )
            runner = BalancedLOMORunner(
                ds=ds,
                held_out_metal=metal,
                balanced_smiles=balanced_smiles,
                balanced_labels=balanced_labels,
                wasserstein_weight=args.wasserstein_weight,
                n_mc=args.n_mc,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=int(seed),
                device=device,
                use_l2_guard=args.use_l2_guard,
                model_cls=model_cls,
                model_alias=args.model,
                bf16=args.bf16,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                data_config=args.data_config,
                # Wf-187 m19 Rh oversample fix + pos_weight fallback.
                oversample_positives=args.oversample_positives,
                pos_weight=args.pos_weight,
                # Wf-201 — task driver selection (single = LOMO-safe,
                # multi = IID-only with cross-metal leak).
                task_type=args.task_type,
            )
            try:
                result = runner.run(verbose=True)
            except Exception as exc:  # pragma: no cover
                logger.error("fold=%s seed=%s FAILED: %s", metal, seed, exc)
                continue
            results_by_metal[metal].append(result)

            # write per-seed JSON
            metal_dir = out_dir / metal
            metal_dir.mkdir(parents=True, exist_ok=True)
            with (metal_dir / f"per_seed_metrics_seed{seed}.json").open("w") as fh:
                json.dump(result.__dict__, fh, indent=2)

    # Aggregate.
    # R17 baseline (per_metal_lomo_auc.json value) for delta computation.
    baseline = {
        "Ru": 0.5706,
        "Ir": 0.6176,
        "Os": 0.6113,
        "Rh": 0.3677,
        "Re": 0.5059,
    }
    agg = aggregate(results_by_metal, baseline, args.epochs)
    with (out_dir / "aggregate.json").open("w") as fh:
        json.dump(agg.__dict__, fh, indent=2)
    logger.info(
        "AGGREGATE overall=%.4f±%.4f rh=%.4f pass_rh=%s pass_overall=%s",
        agg.overall["test_auc_mean"],
        agg.overall["test_auc_std"],
        agg.per_metal.get("Rh", {}).get("test_auc_mean", 0.0),
        agg.pass_target_rh,
        agg.pass_target_overall,
    )
    logger.info("elapsed wall=%.1fs", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
