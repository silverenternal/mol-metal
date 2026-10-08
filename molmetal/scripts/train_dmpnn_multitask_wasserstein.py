"""Wf-131 Phase 1 — 5-metal multi-task D-MPNN with Wasserstein pIC50 regularisation.

=====================================================================
Multi-task D-MPNN (5-metal joint) + Wasserstein pIC50 regularisation
=====================================================================

Wf-131 P1 closes the m1 Ru AUC gap (current 0.7123 single-task → ≥ 0.86
Krasnov baseline) by stacking two innovations:

1. **Multi-task D-MPNN** — shared encoder + per-metal heads (Ru/Ir/Os/Rh/Re).
   Joint training on the 5-metal cohort (~26.8k rows after Wf-86 expansion)
   gives the Ru head access to the much larger cross-metal signal.
2. **Wasserstein pIC50 regularisation** (TODO-126 §3 M1) — penalises
   predicted pIC50 distributions that drift off the reference training
   pIC50 distribution. Implemented in
   :mod:`molmetal.molmetal_lam.reward.wasserstein_loss` (Wf-123 P1).

The loss for one training step is::

    L = α · Σ_m BCE(active_logits[m], active_target[m])
      + (1 - α) · Σ_m MSE(pic50_pred[m], pic50_target[m])
      + w_w · W₂(predicted_pIC50_all, ref_pIC50_all)

A L2 (gradient-norm) guard is ARMED each step (max_norm=5.0, per the
"L2 guard" flag, same convention as Wf-44 / Wf-49). Note the wording: it is
a safety net, not an active constraint — MEASURED on the CFM sibling trainer
(results/cfm_per_innovation_ablation_p1.4/events.jsonl, 14 epoch records) the
gradient norm sits at min 2.37 / median 2.83 / max 3.58, i.e. it never
approaches 5.0 and the clip never fires. The guard therefore does not
currently shape training; it exists to catch an exploding-gradient step.
Do not lower max_norm to "make it do something" — at 1.0 it would bind on
every step and silently rescale the effective learning rate. See
TODO/completed/config_dead_key_remediation.md.

**Inputs**
  * ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv`` — split definitions
  * ``molmetal/checkpoints/dmpnn_pic50_{ru,ir,os,rh,re}_seed{0..4}.pt`` (25
    ckpts from Wf-120 P2 — used as backbones for optional warm-start;
    default = random init)
  * ``molmetal/baselines/dmpnn_multitask.py`` — DMPNNMultiTaskModel +
    MultiTaskLoss + train_epoch_mt + predict_mt (re-used)

**Outputs**
  * Per-(metal, seed) checkpoint ``dmpnn_mtw_{metal}_seed{seed}.pt``
  * ``molmetal/reports/wf131_p1_ru_auc_multitask/{metal}/per_seed_metrics.json``
  * ``molmetal/reports/wf131_p1_ru_auc_multitask/training_summary.json``
  * ``molmetal/reports/wf131_p1_ru_auc_multitask/synthesis.md``

**Hardware**: CPU primary; CUDA if available (RTX 5090 GPU-1 when
invoked with ``CUDA_VISIBLE_DEVICES=0``).

**Honest framing** (per ``party-twenty-spirit §3.5``):
  * Smaller metals (Rh/Os/Re < 1.2k rows each) give noisier per-seed
    pIC50 R² and AUC — we *report* the per-seed spread but do *not* paper
    over it.
  * Wasserstein regularisation may slow convergence in early epochs (it
    forces the predicted pIC50 cloud to track the reference distribution
    before the encoder has learned anything).  We report the decay curve.
  * The Ru AUC ≥ 0.86 target is **projected**, not measured; the recipe
    stacks multi-task (already published in the prior art to lift cross-
    metal signal) + Wasserstein (Wf-123 P1 introduced the shared module).

**Usage**
-----
::

    source env.sh
    \\
    python molmetal/scripts/train_dmpnn_multitask_wasserstein.py \\
        --metals Ru Ir Os Rh Re --seeds 0 1 2 3 4 --epochs 50 \\
        --wasserstein-weight 0.05 \\
        --use-l2-guard \\
        --out-dir molmetal/reports/wf131_p1_ru_auc_multitask
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

import argparse
import hashlib
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    featurize_smiles_list,
)
from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskModel,
    MT_BATCH_SIZE,
    MT_DEPTH,
    MT_DROPOUT,
    MT_HIDDEN,
    MT_PROD_DEPTH,
    MT_PROD_HIDDEN,
    MultiTaskLoss,
    regression_metrics,
)
from molmetal.molmetal_lam.reward.wasserstein_loss import (
    WassersteinConfig,
    wasserstein_pic50_gradient,
    wasserstein_pic50_loss,
)
from molmetal.molmetal_lam.sbdd_env.training_utils import (
    apply_default_cuda_visible_devices,
    resolve_default_device,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_5METAL_CSV = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
DEFAULT_PLATINAI_4CELL_CSV = PROJECT_ROOT / ".data" / "molmetal" / "platinai_4cell.csv"
PLATINAI_4CELL_METALS: Tuple[str, ...] = ("Pt",)
PLATINAI_4CELL_CELL_LINES: Tuple[str, ...] = ("mcf7", "a549", "a2780", "a2780cis")
# Wf-A10 (2026-10-03) — NCI-60 precious-metal 9-cell-line cohort.  Unlike the
# platinai cohort (synthetic labels, see platinai_4cell.meta.json) this file
# carries REAL measured pIC50 from MetalCytoToxDB, and its 9 cell lines
# include all four Rusanov 2025 reports (MCF7 / A549 / A2780 / A2780cis).
# Long form: smiles,cell_line,pIC50,active — one row per (SMILES, cell) pair.
DEFAULT_NCI60_9CELL_CSV = (
    PROJECT_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
)
NCI60_9CELL_CELL_LINES: Tuple[str, ...] = (
    "MCF7",
    "A549",
    "A2780",
    "A2780cis",
    "HCT116",
    "PC3",
    "HL60",
    "SKOV3",
    "DU145",
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf131_p1_ru_auc_multitask"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints"

FIVE_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 50
DEFAULT_ALPHA: float = 0.5  # BCE weight; 0.5 balances reg + cls
DEFAULT_WASSERSTEIN_WEIGHT: float = 0.05  # w_w in the loss
DEFAULT_N_MC: int = 256  # sliced-W2 projections
DEFAULT_LR: float = 1e-3
DEFAULT_BATCH_SIZE: int = MT_BATCH_SIZE
DEFAULT_HIDDEN: int = MT_HIDDEN
DEFAULT_DEPTH: int = MT_DEPTH
DEFAULT_DROPOUT: float = MT_DROPOUT
L2_GUARD_MAX_NORM: float = 5.0  # grad-norm clip

# ---------------------------------------------------------------------------
# Wf-A16 production-toggle defaults + helpers (Wf-328 family).
#
# These four named surfaces are the module-level contract the Wf-328-family
# "production" test modules import.  The underlying levers were already LIVE
# in the trainer (CLI flags + inline torch.backends toggles); this block only
# names them so the tests can pin their defaults and the helpers are testable
# in isolation.  See TODO/completed/wfa16_unimplemented_feature_register.md
# entries 10-13 + 16 and AGENTS.md §5.7.
# ---------------------------------------------------------------------------

#: Default for ``--tf32`` (Wf-328-W4): TF32 matmul ON for Ampere+ GPUs.
#: Mirrors the argparse default below; exposed as a constant so the trainer,
#: the tests and the launcher scripts share ONE source of truth.
DEFAULT_TF32: bool = True

#: Default for ``--bf16`` (Wf-328-W3): BF16 autocast ON for tensor-core GPUs.
#: On CUDA + Ampere-or-newer this wraps the forward + W₂ block in
#: ``torch.autocast(dtype=torch.bfloat16)``; everywhere else it degrades to a
#: ``nullcontext`` no-op via :func:`bf16_autocast`.
DEFAULT_BF16: bool = True

#: Default W₂ penalty mode (Wf-328-W5): the closed-form 1-D sort path.
#: NOTE (honest framing): this flips the *constant/default* from the historical
#: literal ``"sliced"`` to ``"sort"``.  The live ``train_one_epoch_joint`` /
#: ``train_one_metal_one_seed`` signatures now default to this constant.  The
#: slicer remains reachable via ``--w2-gradient-mode sliced`` for Wf-131 P1
#: reproducibility.
DEFAULT_W2_GRADIENT_MODE: str = "sort"

#: Default D-MPNN encoder backend (Wf-334u1).  ``"pyg"`` = the vectorised
#: flat-graph encoder (``_forward_flat``); ``"loop"`` = the legacy Python
#: per-molecule loop (``_forward_one_mol``).  The trainer calls whichever the
#: CLI selects through ``MultiTaskWassersteinModel.encode_pool``.  Both paths
#: route through the same encoder weights — ``loop`` is bit-equivalent up to
#: FP associativity, kept for reproducibility/ablation.
DEFAULT_DMPNN_BACKEND: str = "pyg"

#: The four Inductor ``torch.compile`` modes accepted by ``--torch-compile-mode``
#: (and the ``-v2`` alias).  Wf-335-R1: default is ``max-autotune-no-cudagraphs``
#: (CUDA Graphs disabled) to avoid the Wf-330-B4 RuntimeError.
TORCH_COMPILE_MODES: Tuple[str, ...] = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
DEFAULT_TORCH_COMPILE_MODE: str = "max-autotune-no-cudagraphs"


@dataclass
class TF32State:
    """Post-config TF32 state, persisted into ``training_summary.json``.

    Mirrors :func:`configure_bf16`'s dict contract so both precision toggles
    report the same shape for paper §6 reproducibility audits.
    """

    enabled: bool
    current_allow_tf32_matmul: bool
    current_allow_tf32_cudnn: bool
    matmul_precision: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "current_allow_tf32_matmul": self.current_allow_tf32_matmul,
            "current_allow_tf32_cudnn": self.current_allow_tf32_cudnn,
            "matmul_precision": self.matmul_precision,
        }

    # Dict-style access so the Wf-328-W4 tests can index the result directly
    # (``state["enabled"]``) without unwrapping ``.to_dict()``.
    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


def configure_tf32(enabled: bool = DEFAULT_TF32) -> TF32State:
    """Toggle the three TF32 knobs in one call (Wf-328-W4).

    Sets ``torch.backends.cuda.matmul.allow_tf32`` +
    ``torch.backends.cudnn.allow_tf32`` and the
    ``torch.set_float32_matmul_precision(...)`` API together so the levers
    cannot drift out of sync.  Reads-back the *live* torch state into the
    returned :class:`TF32State` (safe on CPU-only hosts — the flags exist
    regardless of CUDA availability).
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    return TF32State(
        enabled=bool(enabled),
        current_allow_tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        current_allow_tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        matmul_precision=str(torch.get_float32_matmul_precision()),
    )


def configure_bf16(enabled: bool = DEFAULT_BF16, device: object = None) -> Dict[str, object]:
    """Resolve whether BF16 autocast is applicable (Wf-328-W3).

    Returns a dict with the six documented keys:
    ``requested`` / ``apply`` / ``device`` / ``is_cuda`` /
    ``fallback_occurred`` / ``fallback_reason``.

    Rules (CPU-safe; never touches a CUDA device):

    * ``apply`` is True ONLY when ``enabled`` AND the target device is CUDA
      (BF16 has no tensor-core benefit on CPU, so we never open an autocast
      window there).
    * ``fallback_occurred`` is True when the user *requested* BF16 but it was
      not applicable (CPU / pre-Ampere) — so paper §6 audits can tell a real
      BF16 run from a silent fallback.  It is False when the user explicitly
      opted out (no fallback, just OFF).
    """
    device_str = str(device) if device is not None else "cpu"
    is_cuda = device_str.startswith("cuda")
    apply = bool(enabled) and is_cuda
    if apply:
        fallback_occurred = False
        fallback_reason = ""
    elif enabled:
        fallback_occurred = True
        fallback_reason = (
            "BF16 autocast not applied: device is CPU (no tensor cores); "
            "requested BF16 falls back to FP32."
            if not is_cuda
            else "BF16 autocast not applicable on this device."
        )
    else:
        fallback_occurred = False
        fallback_reason = ""
    return {
        "requested": bool(enabled),
        "apply": apply,
        "device": device_str,
        "is_cuda": is_cuda,
        "fallback_occurred": fallback_occurred,
        "fallback_reason": fallback_reason,
    }


def bf16_autocast(device: object = None, enabled: bool = DEFAULT_BF16):
    """Return a BF16 autocast context manager, or a ``nullcontext`` no-op.

    CUDA + ``enabled`` → ``torch.autocast(device_type="cuda",
    dtype=torch.bfloat16)``.  Everything else (CPU, ``enabled=False``) → a
    ``contextlib.nullcontext`` so the wrapped block runs bit-for-bit unchanged
    and emits no autocast UserWarning noise on CPU.
    """
    import contextlib

    state = configure_bf16(enabled=enabled, device=device)
    if state["apply"]:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _make_compiled_encoder_forward(encoder: object, mode: str = DEFAULT_TORCH_COMPILE_MODE):
    """Factory: return a compiled replacement for ``encoder._forward_one_mol``.

    Wf-328-W1 / Wf-335-R1.  ``encode_pool`` looks up ``_forward_one_mol`` at
    call-time, so a returned callable can be monkey-patched onto the encoder
    instance (the production trainer does exactly this in
    :func:`train_one_metal_one_seed`).  The compile is applied to the *bound
    method* so Inductor sees the per-molecule ``(V, E)`` forward; on CPU this
    is a valid (if not speedup-yielding) ``torch.compile`` wrapper, and on
    failure we fall back to the eager method so a corrupt cache cannot kill a
    run.
    """
    eager_forward = encoder._forward_one_mol

    def _compiled(*args, **kwargs):
        return eager_forward(*args, **kwargs)

    try:
        return torch.compile(_compiled, mode=mode)
    except Exception:  # noqa: BLE001 - corrupt inductor cache ⇒ eager fallback
        return _compiled


def _make_compiled_w2_penalty(
    w_weight: float = 1.0, mode: str = DEFAULT_TORCH_COMPILE_MODE
):
    """Factory: return a compiled callable computing the sort-mode W₂ penalty.

    Wf-328-W1 / Wf-335-R1.  The returned callable has the signature
    ``(pred, target) -> scalar tensor`` and matches
    :func:`wasserstein_pic50_gradient` within FP32 tolerance.  Per Wf-335-R1
    the default ``mode`` is ``max-autotune-no-cudagraphs`` (CUDA Graphs
    disabled); ``reduce-overhead`` remains accepted for Wf-326C / Wf-328-W1
    bench reproducibility.
    """
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_pic50_gradient,
    )

    def _penalty(pred, target):
        return wasserstein_pic50_gradient(pred, target, w_weight=w_weight)

    try:
        return torch.compile(_penalty, mode=mode)
    except Exception:  # noqa: BLE001 - eager fallback on compile failure
        return _penalty

# Graceful degradation: every model has a known random-init cost in epochs
# before pIC50 predictions start being useful; we log Wasserstein penalty
# every epoch so the decay curve is observable.
WASSERSTEIN_LOG_EVERY: int = 1


# ---------------------------------------------------------------------------
# Runtime event emission (Wf-TUI deep integration)
# ---------------------------------------------------------------------------
# Module-level state for JSONL event emission. Set once in main() via
# _install_event_emitter(); consumed by emit_event() at each print site.
# Default (None) means stdout-only — preserves the Wf-131 P1 + Wf-251-F4
# baseline bit-for-bit when --emit-json-events is False at the CLI.
_EVENT_LOG_PATH: Optional[Path] = None
_EVENT_FH = None  # open file handle (lazy, flushed per-event)
_EVENT_ENABLED: bool = False  # toggle via --emit-json-events CLI flag
_EVENT_TYPES = frozenset(
    {"train_epoch", "eval_epoch", "lr_step", "resume", "batch", "done"}
)


def emit_event(event_type: str, **fields) -> None:
    """Emit a structured runtime event to stdout (text) AND events.jsonl.

    event_type: 'train_epoch' | 'eval_epoch' | 'lr_step' | 'resume' | 'batch' | 'done'
    fields: metric names → values (numbers, strings, lists, bools)
        ─ bool/int/float serialised as-is; str as JSON string; list/tuple as JSON list
        ─ path-like fields are coerced to str()
    Output schema (one JSON object per line):
        {"event": "<event_type>", ...fields, "ts": <unix-time-float>}
    Stdout: also writes a backslash-escaped prefix so log scrapers see
        "[EVENT <type>] ..." for grep-tail compat (preserves Wf-131 P1 text).
    Per AGENTS.md §5.5 filesystem-truth + §5.6 atomic write: this helper
    is the canonical wire-in for the TUI dashboard JSON parsing path
    (vs the legacy regex-based text path per _tui_dashboard.py line 147).
    """
    if event_type not in _EVENT_TYPES:
        # Defensive: refuse unknown event types so JSON-parsing downstream
        # consumers can rely on a stable schema.
        raise ValueError(
            f"emit_event: unknown event_type={event_type!r}; "
            f"valid={sorted(_EVENT_TYPES)}"
        )
    # Coerce non-JSON-native types to JSON-friendly primitives.
    payload: Dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, Path):
            payload[k] = str(v)
        elif isinstance(v, (bool, int, float, str)) or v is None:
            payload[k] = v
        else:
            payload[k] = str(v)
    payload["event"] = event_type
    payload["ts"] = time.time()
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if _EVENT_ENABLED:
        # Stdout JSON line (parseable by `cat | jq`).
        print(line, flush=True)
    else:
        # Stdout backslash-escaped prefix preserves Wf-131 P1 text log
        # format for grep/tail downstream consumers (TUI text regex path).
        # Format mirrors the existing print(...) lines so log history
        # for backward compat is preserved.
        msg = " ".join(f"{k}={payload[k]}" for k in sorted(payload) if k != "ts")
        print(f"  [EVENT {event_type}] {msg}", flush=True)
    # Append to events.jsonl (always when emit_event is called, even
    # when --emit-json-events is False — JSONL consumers downstream
    # want the full event log regardless of text vs JSON stdout format).
    if _EVENT_LOG_PATH is not None:
        global _EVENT_FH
        if _EVENT_FH is None:
            try:
                _EVENT_FH = _EVENT_LOG_PATH.open("a", buffering=1)
            except Exception:
                _EVENT_FH = None
        if _EVENT_FH is not None:
            try:
                _EVENT_FH.write(line + "\n")
            except Exception:
                pass


def _install_event_emitter(
    emit_enabled: bool,
    events_jsonl_path: Path,
) -> None:
    """Module-level install hook called from main() once per job.

    ``events_jsonl_path=None`` means "no sink": emit_event() gates its jsonl
    write on ``_EVENT_LOG_PATH is not None``, so None disables the file while
    leaving the stdout event stream intact.  Passed by --no-events-jsonl, which
    the yamls' output.events_jsonl:false maps to.
    """
    global _EVENT_ENABLED, _EVENT_LOG_PATH, _EVENT_FH
    _EVENT_ENABLED = bool(emit_enabled)
    _EVENT_LOG_PATH = None if events_jsonl_path is None else Path(events_jsonl_path)
    if _EVENT_FH is not None:
        try:
            _EVENT_FH.close()
        except Exception:
            pass
    _EVENT_FH = None


def _close_event_emitter() -> None:
    """Module-level teardown hook called from main() at end of job."""
    global _EVENT_FH, _EVENT_LOG_PATH, _EVENT_ENABLED
    if _EVENT_FH is not None:
        try:
            _EVENT_FH.close()
        except Exception:
            pass
    _EVENT_FH = None
    _EVENT_LOG_PATH = None
    _EVENT_ENABLED = False


# ---------------------------------------------------------------------------
# Wf-A6 A2 BLOCKER C fix — LR scheduler helper (Loshchilov-Hutter 2017 SGDR /
# Smith-Topin 2018 OneCycle). Opt-in via --lr-scheduler {cosine, onecycle}.
# Default OFF (return None) preserves Wf-131 P1 + Wf-251-F4 baseline bit-for-bit.
# ---------------------------------------------------------------------------
from torch.optim.lr_scheduler import (  # Wf-A6 A2 BLOCKER C fix
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    scheduler_type: str,
    warmup_steps: int,
    min_lr: float,
) -> object:
    """Wf-A6 A2 BLOCKER C fix — opt-in LR scheduler factory.

    Args:
        optimizer: torch.optim.Optimizer (Adam for L7 SEMA + D-MPNN).
        total_steps: total training steps (epochs * steps_per_epoch).
        scheduler_type: 'none' | 'cosine' | 'onecycle'.
        warmup_steps: linear warmup duration in steps (0 = no warmup).
        min_lr: cosine floor / onecycle final_div_factor reference.

    Returns:
        None if scheduler_type == 'none' (default; preserves baseline).
        torch.optim.lr_scheduler.LRScheduler otherwise.

    Notes:
        - 'cosine': LinearLR warmup (start_factor=0.01 -> 1.0) +
          SequentialLR([LinearLR, CosineAnnealingLR], milestones=[warmup_steps]).
          Total cosine steps = total_steps - warmup_steps.
        - 'onecycle': OneCycleLR(max_lr=base_lr*10, total_steps=total_steps,
          pct_start=0.3, anneal_strategy='cos', div_factor=25,
          final_div_factor=1e4). No separate warmup phase needed.
    """
    if scheduler_type == "none":
        return None
    if scheduler_type == "cosine":
        # Guard: warmup_steps must be < total_steps (else cosine gets 0 steps).
        warmup_steps = max(0, min(int(warmup_steps), int(total_steps) - 1))
        cosine_steps = max(1, int(total_steps) - warmup_steps)
        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=float(min_lr))
        if warmup_steps == 0:
            return cosine
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    if scheduler_type == "onecycle":
        base_lr = float(optimizer.defaults[0]["lr"])
        return OneCycleLR(
            optimizer,
            max_lr=base_lr * 10,
            total_steps=int(total_steps),
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        )
    raise ValueError(f"build_lr_scheduler: unknown scheduler_type={scheduler_type!r}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_5metal_split_table(csv_path: Path) -> pd.DataFrame:
    """Load the 5-metal multitask CSV with splits encoded per row.

    Returns a DataFrame with columns: ``smiles, metal, scaffold_id, split,
    activity`` and a stable ``row_id`` (0-indexed in original CSV order).
    """
    df = pd.read_csv(csv_path)
    expected = {"smiles", "metal", "scaffold_id", "split", "activity"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"5-metal CSV missing columns: {missing}")
    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})
    return df


def _resolve_default_csv_path(dataset_name: str) -> Path:
    """Resolve the default CSV path for a given ``--dataset`` choice.

    Wf-A8 PlatinAI 4-cell loader (audit gap #1 fix):
        ``metalcytotox_5metal`` -> ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv``
        ``platinai_4cell`` -> ``.data/molmetal/platinai_4cell.csv``
    Raises ``FileNotFoundError`` with a clear "Available datasets" message
    if the resolved CSV does not exist on disk.
    """
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "metalcytotox_5metal":
        return DEFAULT_5METAL_CSV
    if dataset_name == "platinai_4cell":
        return DEFAULT_PLATINAI_4CELL_CSV
    if dataset_name == "nci60_9cell":
        return DEFAULT_NCI60_9CELL_CSV
    raise ValueError(
        f"Unknown --dataset={dataset_name!r}. Available datasets: "
        f"{{'metalcytotox_5metal', 'platinai_4cell', 'nci60_9cell'}}."
    )


def _stable_hash_int(text: str, modulus: int) -> int:
    """Process-independent hash of *text* into ``[0, modulus)``.

    Wf-A10 FIX (2026-10-03) — Python salts the hash of ``str`` per process
    (PYTHONHASHSEED), so the previous

        abs(hash(smiles)) % (10**8)

    produced a DIFFERENT scaffold id — and therefore a different
    train/val/test split — on every run.  Measured for ``hash("CCO")``
    across three processes: 60767441 / 88287468 / 44141501.  That silently
    made every platinai result irreproducible.

    This uses a fixed SHA-256 digest instead, so the same SMILES always
    yields the same id on any process and any machine.
    """
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(modulus)


def load_platinai_4cell_split_table(
    csv_path: Path,
    split_strategy: str = "scaffold_70_10_10",
) -> pd.DataFrame:
    """Load the PlatinAI 4-cell CSV (Rusanov 2025).

    The CSV stores one row per SMILES with 4 cell-line pIC50 columns::

        SMILES_Ligands,pIC50_mcf7,pIC50_a549,pIC50_a2780,pIC50_a2780cis,metal

    We pivot to **long format** (one row per SMILES-cell combination) so the
    downstream multi-task pool machinery (which expects columns
    ``smiles, metal, scaffold_id, split, activity``) works unmodified.

    Scaffold split: deterministic hash of canonicalised SMILES → 70/10/10
    train/val/test per Rusanov 2025 §3.2 + per-cell breakdown.

    Returns DataFrame with columns: ``smiles, metal, scaffold_id, split,
    activity, cell_line, row_id`` (long format).
    """
    df = pd.read_csv(csv_path)
    if "SMILES_Ligands" not in df.columns:
        raise ValueError(
            f"platinai_4cell CSV missing 'SMILES_Ligands' column. "
            f"Found columns: {list(df.columns)}."
        )
    metal_value = "Pt" if "metal" not in df.columns else df["metal"].iloc[0]
    records = []
    for col in PLATINAI_4CELL_CELL_LINES:
        ic_col = f"pIC50_{col}"
        if ic_col not in df.columns:
            continue
        sub = df[["SMILES_Ligands", ic_col]].copy()
        sub = sub.dropna(subset=[ic_col])
        sub = sub.rename(columns={"SMILES_Ligands": "smiles", ic_col: "activity"})
        sub["metal"] = metal_value
        sub["cell_line"] = col
        records.append(sub)
    if not records:
        raise ValueError(
            f"platinai_4cell CSV at {csv_path} produced no long-format rows. "
            f"Expected columns prefixed with 'pIC50_' matching "
            f"{PLATINAI_4CELL_CELL_LINES}."
        )
    long_df = pd.concat(records, ignore_index=True)
    # Wf-A10 FIX (2026-10-03) — stable scaffold id.  Was
    # ``abs(hash(smiles)) % (10**8)``, which is salted per process and made
    # the split change on every run (see _stable_hash_int).  Because the id
    # depends only on the SMILES, the same molecule lands in the same split
    # for all 4 cell lines — that is what prevents cross-cell leakage.
    long_df["scaffold_id"] = (
        long_df["smiles"].astype(str).map(lambda s: _stable_hash_int(s, 10**8))
    )
    if split_strategy == "scaffold_70_10_10":
        # 70/10/10 bucket, derived from the stable id only.  (Was
        # ``abs(hash(sid)) % 100`` on an int — stable in practice, but it
        # still leaned on a builtin whose stability is not part of its
        # contract, so it now goes through the same digest helper.)
        def _split_bucket(sid: int) -> str:
            h = _stable_hash_int(str(sid), 100)
            if h < 70:
                return "train"
            if h < 80:
                return "val"
            return "test"

        long_df["split"] = long_df["scaffold_id"].map(_split_bucket)
    else:
        raise ValueError(
            f"Unknown split_strategy={split_strategy!r}. Supported: "
            f"'scaffold_70_10_10'."
        )
    long_df = long_df.reset_index(drop=False).rename(columns={"index": "row_id"})
    return long_df


def _derive_active_labels(
    pic50: pd.Series,
    task_labels: pd.Series,
    labeling: str,
    global_threshold: float,
) -> pd.Series:
    """Boolean ``active`` labels for the platinai cell-line axis (Wf-A10).

    ``labeling='per_cell_median'`` (default): each task is split at ITS OWN
    median pIC50, so every per-cell head sees a balanced 50/50 problem.
    ``labeling='global'``: a single pIC50 cut for all tasks.

    Why per-task is the default here: measured on platinai_4cell.csv, a single
    global cut cannot balance the four heads, because the cell lines have
    different potency distributions (median pIC50 = a2780 5.69, mcf7 5.99,
    a549 6.15, a2780cis 6.59). At a global 1 uM cut the per-cell active
    fraction is a2780 0.242 / mcf7 0.491 / a549 0.632 / a2780cis 0.902 — the
    a2780cis head, the one this cohort exists to characterise, is 90% positive
    and its AUC is near-degenerate. A per-task median keeps every head at 0.50
    and makes the per-cell AUC a within-cell potency ranking, comparable
    across cell lines.
    """
    if labeling == "per_cell_median":
        return (
            pic50.groupby(task_labels)
            .transform(lambda s: s > float(s.median()))
            .astype(bool)
        )
    if labeling == "global":
        return (pic50 > float(global_threshold)).astype(bool)
    raise ValueError(
        f"Unknown active labeling {labeling!r}. "
        f"Supported: 'per_cell_median', 'global'."
    )


def build_platinai_cell_line_axis(
    csv_path: Path,
    active_pic50_threshold: float = 6.0,
    active_labeling: str = "per_cell_median",
) -> Tuple[pd.DataFrame, "MetalCytotoxDataset"]:
    """Build ``(split_df, ds)`` for the platinAI 4-cell cohort on a CELL-LINE task axis.

    Wf-A10 (2026-10-03).  :class:`MultiTaskWassersteinModel` is a shared encoder
    plus one (regression + classification) head per task, selected by an integer
    task id — nothing about it is metal-specific.  The platinai cohort is one
    metal (Pt) across four cell lines, so the task axis becomes the *cell line*::

        MCF7 / A549 / A2780 / A2780cis  ->  4 heads over one shared encoder

    That is how the field does multi-cell-line cytotoxicity prediction: PECAN
    predicts 59 NCI-60 lines from a single shared network (arguing the shared
    objective forces generalisable features), while per-cell-line separate
    models are the superseded approach.  It also matches Rusanov 2025, whose
    PlatinAI benchmark is reported as AUC **per cell line**.

    The long-format loader already puts the cell line into the ``metal`` column,
    so the downstream pool machinery (``build_metal_split_indices``) works
    unmodified; ``ds`` is then built with exactly the four columns the trainer
    reads — ``SMILES_Ligands`` / ``Metal`` / ``pIC50`` / ``active``.

    ``active`` labels use ``active_pic50_threshold`` (default 6.0, i.e.
    IC50 < 1 uM).  Note this deliberately differs from the repo's
    MetalCytoToxDB standard of 5.0: measured on the real platinai cohort,
    5.0 marks 98.1% of rows active, which degenerates the classification
    task.  See the ``--active-pic50-threshold`` help text for the full
    measured sweep.
    """
    split_df = load_platinai_4cell_split_table(csv_path).copy()
    split_df["metal"] = split_df["cell_line"].astype(str)

    ds_df = pd.DataFrame(
        {
            "SMILES_Ligands": split_df["smiles"].astype(str),
            "Metal": split_df["cell_line"].astype(str),
            "pIC50": split_df["activity"].astype(float),
        }
    )
    ds_df["active"] = _derive_active_labels(
        ds_df["pIC50"],
        ds_df["Metal"],
        active_labeling,
        active_pic50_threshold,
    )
    ds_df = ds_df.reset_index(drop=True)
    if ds_df.empty:
        raise ValueError(
            f"platinai cell-line axis produced an empty dataset from {csv_path}."
        )
    # Local cache dir avoids /mnt/storage permission errors (same convention
    # as the --data-csv branch below).
    cache_dir = Path("/tmp") / "wf131_mtw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return split_df, MetalCytotoxDataset(df=ds_df, cache_dir=cache_dir)


class _SmilesOnlyAdapter:
    """Minimal ``.smiles`` / ``__len__`` shim so ``ScaffoldSplitter`` can group.

    ``ScaffoldSplitter.__call__`` only reads ``dataset.smiles`` (and ``len``),
    so this is enough to run it over a bare array of unique SMILES.
    """

    def __init__(self, smiles: "np.ndarray") -> None:
        self.smiles = smiles

    def __len__(self) -> int:
        return int(len(self.smiles))


def load_nci60_9cell_split_table(
    csv_path: Path,
    split_strategy: str = "scaffold_70_10_10",
) -> pd.DataFrame:
    """Load the NCI-60 precious-metal 9-cell long-form cohort + scaffold split.

    Wf-A10 (2026-10-03).  Schema: ``smiles,cell_line,pIC50,active`` — one row
    per (SMILES, cell) pair (7,823 rows / 5,207 unique SMILES / 9 cell lines).

    Why this cohort
    ---------------
    Unlike ``platinai_4cell`` (whose labels are synthetic noise — see
    ``molmetal/data/platinai_4cell.meta.json``), this file carries **real
    measured** pIC50 from MetalCytoToxDB, and its 9 cell lines include all
    four that Rusanov 2025 reports (MCF7 / A549 / A2780 / A2780cis).  It is
    therefore the cohort that can actually answer the multi-cell-line
    question; the platinai per-cell AUC near 0.50 is a null test (it proves
    no label leakage), not a measurement.

    Split protocol — deliberately m21's, for comparability
    -----------------------------------------------------
    The split reproduces ``molmetal/scripts/train_nci60_10cell.py`` exactly,
    so an AUC measured here is directly comparable to that script's shipped
    m21 baseline (0.8007 +/- 0.0160):

      * ``ScaffoldSplitter`` (Bemis-Murcko, ``largest_first``, 70/10/20,
        seed=0) is run on the **unique canonical SMILES**, not on rows.
        That is what m21 does, because it pivots to wide form so its
        ``ds.smiles`` holds one entry per molecule.
      * MEASURED: doing so reproduces m21's split byte-for-byte — seed=0
        gives train/val/test = 3645/521/1041, md5
        ``86d08f8ace`` / ``7a9edd4f9d`` / ``548a90d992`` — and the
        unique-SMILES population is identical (0 elements differ in either
        direction; 5,207 both sides).
      * Splitting unique SMILES also guarantees no cross-cell leakage: a
        molecule lands in exactly one split, so all of its cell rows follow.

    Labels — use the CSV's own ``active`` column, do NOT re-derive it
    -----------------------------------------------------------------
    ``prepare_nci60_data.py`` computes that column as
    ``ic50_um < ACTIVE_THRESHOLD_UM`` with the same 10 uM constant m21 uses,
    so it *is* m21's label.  Re-deriving it from pIC50 reproduces the column
    only to 99.32% (53/7823 boundary rows differ), and comparability with
    m21 is this cohort's entire purpose.
    """
    if split_strategy != "scaffold_70_10_10":
        raise ValueError(
            f"Unknown split_strategy={split_strategy!r} for nci60_9cell. "
            f"Available: {{'scaffold_70_10_10'}}."
        )
    df = pd.read_csv(csv_path, low_memory=False)
    required = {"smiles", "cell_line", "pIC50", "active"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"nci60_9cell CSV missing columns: {sorted(missing)} in {csv_path}. "
            f"Expected at least {sorted(required)}."
        )
    df = df[df["pIC50"].notna()].copy()
    if df.empty:
        raise ValueError(f"nci60_9cell CSV has no labelled rows: {csv_path}")

    df["smiles"] = df["smiles"].astype(str)
    df["cell_line"] = df["cell_line"].astype(str)
    df["activity"] = df["pIC50"].astype(float)
    df["active"] = df["active"].astype(int)

    from molmetal.data.splits import ScaffoldSplitter

    uniq = np.asarray(sorted(df["smiles"].unique()), dtype=object)
    result = ScaffoldSplitter(
        strategy="largest_first",
        fractions=(0.7, 0.1, 0.2),
        seed=0,
    )(_SmilesOnlyAdapter(uniq))

    smi_to_split: Dict[str, str] = {}
    for _name, _idx in (
        ("train", result.train_idx),
        ("val", result.val_idx),
        ("test", result.test_idx),
    ):
        for _i in _idx:
            smi_to_split[str(uniq[int(_i)])] = _name

    df["split"] = df["smiles"].map(smi_to_split)
    _unassigned = int(df["split"].isna().sum())
    if _unassigned:
        raise ValueError(
            f"{_unassigned} nci60 rows received no split assignment — the "
            f"splitter saw a different SMILES set than the rows carry."
        )
    # The task axis is the cell line; the downstream pool machinery keys off
    # a ``metal`` column, so alias it (same convention as the platinai axis).
    df["metal"] = df["cell_line"]
    return df.reset_index(drop=True)


def build_nci60_cell_line_axis(
    csv_path: Path,
) -> Tuple[pd.DataFrame, "MetalCytotoxDataset"]:
    """Build ``(split_df, ds)`` for the NCI-60 9-cell cohort on a cell-line axis.

    Wf-A10 (2026-10-03).  Mirrors :func:`build_platinai_cell_line_axis`:
    :class:`MultiTaskWassersteinModel` is a shared encoder plus one
    (regression + classification) head per integer task id, so the 9 cell
    lines become 9 heads over one shared encoder — the PECAN-style shared
    network that the field uses for multi-cell-line cytotoxicity, rather
    than 9 independent models.

    Note this builder deliberately takes **no** threshold/labeling
    arguments: the ``active`` label comes from the CSV verbatim (it is
    m21's label — see :func:`load_nci60_9cell_split_table`).  Accepting a
    threshold here and ignoring it would be a silent footgun, so it is not
    offered.
    """
    split_df = load_nci60_9cell_split_table(csv_path).copy()

    ds_df = pd.DataFrame(
        {
            "SMILES_Ligands": split_df["smiles"].astype(str),
            "Metal": split_df["cell_line"].astype(str),
            "pIC50": split_df["activity"].astype(float),
            # m21's own label (IC50 < 10 uM), taken from the CSV.
            "active": split_df["active"].astype(int),
        }
    )
    ds_df = ds_df.reset_index(drop=True)
    if ds_df.empty:
        raise ValueError(
            f"nci60 cell-line axis produced an empty dataset from {csv_path}."
        )
    cache_dir = Path("/tmp") / "wf131_mtw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return split_df, MetalCytotoxDataset(df=ds_df, cache_dir=cache_dir)


def _featurize_with_cache(
    smiles: List[str],
    cache_key: str,
    cache_dir: Path,
) -> List:
    """Featurise SMILES with on-disk pickle cache (reuse from Wf-84 P3).

    The first call writes ``{cache_dir}/{cache_key}.pkl``; subsequent
    calls reuse it.  Critical for multi-seed runs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.pkl"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_smiles, cached_graphs = pickle.load(f)
            if cached_smiles == smiles:
                return cached_graphs
        except Exception as _cache_err:
            # Wf-ultracode FIX — warn loudly so a corrupt pickle / RDKit
            # version drift / partial-write does NOT silently trigger a
            # full re-featurisation (the unfixed `except Exception: pass`
            # made this invisible to downstream metrics).
            print(
                f"WARNING: cache load failed for {cache_path} "
                f"({type(_cache_err).__name__}: {_cache_err}); "
                f"re-featurising from scratch",
                file=sys.stderr,
                flush=True,
            )

    t0 = time.time()
    graphs = featurize_smiles_list([str(s) for s in smiles])
    elapsed = time.time() - t0
    try:
        with cache_path.open("wb") as f:
            pickle.dump((smiles, graphs), f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    print(
        f"    featurised {len(smiles)} SMILES in {elapsed:.1f}s "
        f"(cache: {cache_path})",
        flush=True,
    )
    return graphs


def build_metal_split_indices(
    split_df: pd.DataFrame,
    ds: MetalCytotoxDataset,
    metal: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect 5-metal CSV rows for *metal* with the loaded dataset."""
    metal_df = split_df[split_df["metal"] == metal].copy()
    if metal_df.empty:
        raise RuntimeError(f"No 5-metal rows for metal={metal!r}")

    smiles_set = set(metal_df["smiles"].astype(str).tolist())

    smi_to_split: Dict[str, str] = {}
    for _, row in metal_df.iterrows():
        smi = str(row["smiles"])
        smi_to_split[smi] = str(row["split"])

    ds_smiles = np.asarray(ds.smiles).astype(str)
    ds_metal = np.asarray(ds.metals).astype(str)
    in_metal = ds_metal == metal
    in_smiles = np.array([s in smiles_set for s in ds_smiles], dtype=bool)
    keep = in_metal & in_smiles
    keep_idx_all = np.where(keep)[0]

    splits = np.array(
        [smi_to_split.get(ds_smiles[i], "train") for i in keep_idx_all],
        dtype=object,
    )

    idx_train = keep_idx_all[splits == "train"]
    idx_val = keep_idx_all[splits == "val"]
    idx_test = keep_idx_all[splits == "test"]
    return idx_train, idx_val, idx_test


# ---------------------------------------------------------------------------
# Multi-task trainer (joint across all 5 metals)
# ---------------------------------------------------------------------------
@dataclass
class MetalSeedResult:
    """Per-(metal, seed) outcome."""

    metal: str
    seed: int
    ckpt_path: str
    n_train: int
    n_val: int
    n_test: int
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    val_pic50_metrics: Dict[str, float]
    test_pic50_metrics: Dict[str, float]
    wasserstein_history: List[float]
    elapsed_seconds: float
    status: str = "OK"
    # 2026-10-08 — optional ridge readout on the frozen encoder embedding
    # (None unless --pic50-ridge-readout).  Reported alongside the neural
    # head's test_pic50_metrics, never replacing it.
    pic50_ridge_metrics: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Wf-335-P3 — Triton dispatch wrapper for the per-metal ReLU MLP heads.
# ---------------------------------------------------------------------------
class _FusedReLUMLPWrapper(nn.Module):
    """Eager-fallback wrapper for the Triton-fused ReLU-MLP head.

    The Wf-335-P3 reference kernel ``fused_relu_mlp`` (Blackwell sm_120)
    replaces ``nn.Sequential(Linear, ReLU, Dropout, Linear)`` with a
    single fused Triton call.  When CUDA is unavailable OR the
    ``FUSED_RELU_MLP`` env gate is closed (default) the eager 2-layer
    MLP is invoked and the wrapper is bit-for-bit identical to the
    original sequential.

    Per AGENTS.md §5.7 the wrapper MUST be wired into
    ``MultiTaskWassersteinModel`` (this file's class above) so that
    ``--use-triton-relu-mlp`` reaches a live computation site.  The
    wrapper's ``dispatch_stats`` attribute is the runtime evidence the
    Triton path is genuinely invoked when the gate opens.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        dropout: float = 0.0,
        dropout_p: Optional[float] = None,
    ) -> None:
        if dropout_p is not None:
            dropout = float(dropout_p)
        super().__init__()
        # Eager 2-layer MLP — bit-for-bit identical to the original
        # ``nn.Sequential(Linear, ReLU, Dropout, Linear)`` shape.
        self.eager = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        # Triton dispatch gate state — checked at forward time so the
        # wrapper can be toggled by setting FUSED_RELU_MLP=1 in env.
        self.dispatch_stats: Dict[str, int] = {
            "triton_dispatch_attempted": 0,
            "triton_dispatch_succeeded": 0,
            "triton_dispatch_fell_back": 0,
            "triton_dispatch_fallback": 0,
        }
        self._fused_module = None  # lazily resolved on first forward

    def _maybe_resolve_fused(self) -> bool:
        """Lazy-import the Triton kernel + gate decision. Returns
        True iff the fused path is selected for this forward."""
        if self._fused_module is not None:
            return True
        try:
            import os as _os

            if _os.environ.get("FUSED_RELU_MLP", "0") != "1":
                return False
            if not torch.cuda.is_available():
                return False
            from molmetal.adapters.flow_matching_lipman._reference import (  # noqa: E501
                fused_relu_mlp as _frm,
            )

            self._fused_module = _frm
            return True
        except Exception:
            return False

    def forward(self, x: Tensor) -> Tensor:
        if self._maybe_resolve_fused():
            self.dispatch_stats["triton_dispatch_attempted"] += 1
            try:
                return self._fused_module.fused_relu_mlp(  # noqa: E501
                    x, self.eager[0], self.eager[3]
                )
            except Exception:
                self.dispatch_stats["triton_dispatch_fell_back"] += 1
                return self.eager(x)
        return self.eager(x)


class MultiTaskWassersteinModel(nn.Module):
    """5-metal multi-task D-MPNN: shared encoder + per-metal heads.

    The encoder is :class:`DMPNNMultiTaskModel` minus its existing
    2-head readout; we replace it with five independent
    (regression + classification) heads, one per metal.  Each metal
    head consumes the same 2H-dim pooled backbone representation.
    """

    def __init__(
        self,
        n_metals: int,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = DEFAULT_HIDDEN,
        depth: int = DEFAULT_DEPTH,
        dropout: float = DEFAULT_DROPOUT,
        # Wf-285B — Kendall UW + DANN GRL wiring (regression-test
        # contract pinned by ``molmetal/tests/test_kendall_dann_wiring.py``
        # test_dann_on_off).  ``use_dann`` plumbs through to the
        # :class:`DMPNNMultiTaskModel` backbone which already owns the
        # ``DomainAdversarialHead`` (TODO-143 §2.2).  All kwargs default
        # OFF to preserve Wf-131 P1 / Wf-380 backward compat bit-for-bit.
        use_dann: bool = False,
        num_domains: int = 4,
        domain_head_dropout: float = 0.0,
        # Wf-335-P3 — Triton dispatch gate (default OFF; env-gated).
        use_triton_relu_mlp: bool = False,
        # Wf-334u1 — pluggable D-MPNN backend.  ``"flat"`` (default) uses the
        # vectorised flat-graph encoder; ``"per_mol"`` forces the legacy
        # Python per-molecule loop (bit-equivalent, slower).  Consumed by
        # ``encode_pool`` below.
        dmpnn_backend: str = DEFAULT_DMPNN_BACKEND,
    ):
        super().__init__()
        self.n_metals = int(n_metals)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.use_triton_relu_mlp = bool(use_triton_relu_mlp)
        self.dmpnn_backend = str(dmpnn_backend)

        # ----- Shared encoder (re-uses DMPNNMultiTaskModel backbone) -----
        # The ``use_dann`` / ``num_domains`` / ``domain_head_dropout`` kwargs
        # are forwarded so the model's ``domain_head`` attribute is set
        # up front; the per-step training loop (``train_one_epoch_joint``)
        # is responsible for invoking ``ReverseLayerF.apply`` before the
        # forward pass when ``use_dann=True`` (canonical DANN pattern).
        self.encoder = DMPNNMultiTaskModel(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden=hidden,
            depth=depth,
            dropout=dropout,
            use_dann=use_dann,
            num_domains=num_domains,
            domain_head_dropout=domain_head_dropout,
        )
        # DANN surface mirrors :class:`DMPNNMultiTaskModel` so
        # ``test_dann_on_off`` can introspect ``model.use_dann`` /
        # ``model.domain_head`` / ``model.num_domains`` directly without
        # digging into the encoder sub-module.
        self.use_dann: bool = bool(use_dann)
        self.num_domains: int = int(num_domains)
        self.domain_head = self.encoder.domain_head
        # ----- Per-metal heads: each is a (regression + classification) pair -----
        # Wf-335-P3 — when ``use_triton_relu_mlp`` is set we swap the
        # eager Sequential for the _FusedReLUMLPWrapper (which delegates
        # back to eager on CPU + gate-closed per the §5.7 contract).
        if self.use_triton_relu_mlp:
            _head_reg_factory = lambda: _FusedReLUMLPWrapper(  # noqa: E731
                in_dim=2 * hidden, hidden=hidden, out_dim=1, dropout=dropout
            )
            _head_cls_factory = lambda: _FusedReLUMLPWrapper(  # noqa: E731
                in_dim=2 * hidden, hidden=hidden, out_dim=2, dropout=dropout
            )
        else:
            _head_reg_factory = lambda: nn.Sequential(  # noqa: E731
                nn.Linear(2 * hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            _head_cls_factory = lambda: nn.Sequential(  # noqa: E731
                nn.Linear(2 * hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
        self.reg_heads = nn.ModuleList([_head_reg_factory() for _ in range(n_metals)])
        self.cls_heads = nn.ModuleList([_head_cls_factory() for _ in range(n_metals)])

    # ------------------------------------------------------------------
    # EMA — exponential moving average of the parameters (Karras 2022 EDM §5)
    # ------------------------------------------------------------------
    # 2026-10-04: `train.ema_decay: 0.999` in configs/default.yaml had NO
    # consumer -- this trainer had no EMA code at all (the 20 grep hits for
    # "ema" in this file are all the substring inside "schema").  Same
    # situation as the CFM trainer's --use-ema, and wired the same way so the
    # two entries behave alike.
    def init_ema(self, decay: float = 0.999) -> None:
        """Snapshot the current parameters as the EMA starting point."""
        self._ema_decay = float(decay)
        self._ema_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        self._ema_backup = None

    def update_ema(self, decay: Optional[float] = None) -> None:
        """``ema = decay * ema + (1 - decay) * model`` over parameters+buffers."""
        state = getattr(self, "_ema_state", None)
        if state is None:
            return
        d = float(self._ema_decay if decay is None else decay)
        with torch.no_grad():
            msd = self.state_dict()
            for k, v_ema in state.items():
                v = msd[k]
                if not torch.is_floating_point(v):
                    # Integer buffers (e.g. num_batches_tracked) -- copy verbatim.
                    v_ema.copy_(v)
                else:
                    v_ema.mul_(d).add_(v.detach(), alpha=1.0 - d)

    def ema_swap_in(self) -> bool:
        """Back up the live weights and load the EMA snapshot over them."""
        state = getattr(self, "_ema_state", None)
        if state is None:
            return False
        self._ema_backup = {k: v.detach().clone() for k, v in self.state_dict().items()}
        msd = self.state_dict()
        with torch.no_grad():
            for k, v_ema in state.items():
                msd[k].copy_(v_ema)
        return True

    def ema_swap_out(self) -> None:
        """Restore the weights backed up by :meth:`ema_swap_in`."""
        backup = getattr(self, "_ema_backup", None)
        if backup is None:
            return
        msd = self.state_dict()
        with torch.no_grad():
            for k, v in backup.items():
                msd[k].copy_(v)
        self._ema_backup = None

    def encode_pool(
        self,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the shared encoder, returning the 2H-dim pooled vector per mol.

        Wf-334u1 — ``self.dmpnn_backend`` selects the path:

        * ``"pyg"`` (default) — the vectorised flat-graph encoder
          ``_forward_flat`` (bit-equivalent to the per-mol loop up to FP
          summation order; measured 8-10x faster — see
          ``tests/test_dmpnn_flat_forward.py``).
        * ``"loop"`` — the legacy Python per-molecule loop over
          ``_forward_one_mol`` (the historical path; kept for
          reproducibility/ablation).

        The flat path needs padded ``edge_src``/``edge_dst`` with ``-1``
        sentinels (the collate already supplies those); the per-mol path
        slices out the valid edges itself.
        """
        B = atom_feats.size(0)
        device = atom_feats.device
        if str(self.dmpnn_backend) != "loop":
            return self.encoder._forward_flat(atom_feats, bond_feats, edge_src, edge_dst)
        pooled_list: List[torch.Tensor] = []
        for b in range(B):
            V = atom_feats.size(1)
            valid_edges = bond_feats[b].abs().sum(dim=-1) > 0
            E = valid_edges.sum().item()
            if E == 0 or V == 0:
                pooled_list.append(torch.zeros(2 * self.hidden, device=device))
                continue
            pooled_list.append(
                self.encoder._forward_one_mol(
                    atom_feats[b, :V],
                    bond_feats[b, :E],
                    edge_src[b, :E],
                    edge_dst[b, :E],
                )
            )
        return torch.stack(pooled_list, dim=0)  # (B, 2H)

    def forward(
        self,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        metal_ids: torch.Tensor,  # (B,) long
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-batch forward. Returns (pic50_preds, active_logits) shape (B,)."""
        h_pool = self.encode_pool(atom_feats, bond_feats, edge_src, edge_dst)
        pic50 = torch.zeros(h_pool.size(0), device=h_pool.device)
        cls = torch.zeros(h_pool.size(0), 2, device=h_pool.device)
        for i in range(h_pool.size(0)):
            m = int(metal_ids[i].item())
            if m < 0 or m >= self.n_metals:
                continue
            pic50[i] = self.reg_heads[m](h_pool[i]).squeeze(-1)
            cls[i] = self.cls_heads[m](h_pool[i])
        return pic50, cls


# `collate_multimetal` was extracted to molmetal/training/collate_multimetal.py
# (Wf-??? monolith-split, 2026-10-05). Re-imported here as a backwards-compat
# shim so existing callers — tests, scripts — keep working.
from molmetal.training.collate_multimetal import collate_multimetal  # noqa: E402,F401


def train_one_epoch_joint(
    model: MultiTaskWassersteinModel,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    w_w: float,
    use_l2_guard: bool,
    graphs: List[Optional[Tuple]],
    metal_ids: List[int],
    pic50: np.ndarray,
    active: np.ndarray,
    ref_pic50_dist: torch.Tensor,
    batch_size: int,
    device: torch.device,
    w_config: WassersteinConfig,
    w2_gradient_mode: str = DEFAULT_W2_GRADIENT_MODE,
    # Wf-405-A2 — Wf-380 NEW CLI flag wire-in kwargs.  All default
    # to OFF / zero-contribution so backward compat with Wf-131 P1 +
    # Wf-251-F4 baseline is preserved bit-for-bit when no flag is
    # set on the CLI.
    use_ot_cfm: bool = False,  # False = use legacy W₂ (sliced/sort)
    # 2026-10-08 §5.7 — --use-wasserstein-w1 / --use-wasserstein-2 wire-in.
    # Select the closed-form 1-D penalty order via wasserstein_pic50_gradient
    # (wasserstein_order=1 → W₁ mean-abs; =2 → W₂ RMS).  Default OFF keeps the
    # legacy sliced-W₂ path bit-for-bit.
    use_wasserstein_w1: bool = False,  # True = W₁ closed-form sort penalty
    use_wasserstein_2: bool = False,  # True = W₂ closed-form RMS penalty
    per_cohort_features: bool = False,  # False = no per-cohort logging
    # Wf-432-A2 — Wf-246-C metal-coordination term wire-in (Fix-B).
    # Default values preserve bit-for-bit backward compat with Wf-131 P1
    # + Wf-251-F4 baseline when --metal-coord-term is False at the
    # CLI. When metal_coord_term=True, a MetalCoordinationTerm is
    # lazily instantiated and its kcal_adjustment() is added to the
    # composite loss with weight ``metal_coord_weight`` (default 0.5).
    # The kcal sign is sign-fixed per Wf-427-A1 line 641:
    # ``(1 - 2*total) * kcal_per_unit`` (Vina convention: lower
    # kcal/mol = better binding). The ``cal_sign_vina_convention``
    # flag gates this sign fix for backward-compat testing.
    metal_coord_term: bool = False,
    metal_coord_weight: float = 0.5,
    cal_sign_vina_convention: bool = True,
    # Wf-A9 FIX (2026-10-03) — gradient-clip threshold threaded as a
    # kwarg.  The Wf-A8 patch referenced ``args`` inside this function,
    # but ``args`` is ``main()``-scoped → NameError at the first
    # ``total.backward()`` (AGENTS.md §5.7 smoke floor #2).  Default 5.0
    # matches the yaml ``protocol.max_norm`` + the ``--max-norm`` argparse
    # default; 0.0 disables this clip (the separate ``use_l2_guard``
    # clip below is unaffected).
    max_norm: float = 5.0,
    # 2026-10-04 — kwargs for the two remaining dead yaml keys, threaded the
    # same way (main() → train_one_metal_one_seed() → here; never read off a
    # main()-scoped ``args`` object from inside this function).
    #   * ``gradient_accumulation_steps`` — real accumulation: divide each
    #     micro-batch loss by N, clip+step every N micro-batches.  Default 1
    #     keeps the pre-flag per-batch loop bit-for-bit (no division by 1
    #     is even performed on that path).
    #   * ``pin_memory`` — gates the ``non_blocking`` hint on the manual
    #     H2D copies (this trainer has no torch DataLoader).  Default True
    #     matches ``data.pin_memory``; pageable tensors make
    #     ``non_blocking=True`` a documented no-op, so numerics unchanged.
    gradient_accumulation_steps: int = 1,
    pin_memory: bool = True,
    # 2026-10-04 NEW — EMA kwargs.  `train.ema_decay: 0.999` in
    # configs/default.yaml had no consumer because this trainer had no EMA
    # code at all; see the EMA methods on MultiTaskWassersteinModel.
    use_ema: bool = False,
    ema_decay: float = 0.999,
    # Wf-328-W3 — BF16 autocast toggle threaded from main().  When True on a
    # CUDA device, the forward + W₂ block runs inside bf16_autocast(); on CPU
    # it degrades to a nullcontext (bit-for-bit unchanged).  Default False in
    # the epoch-loop signature (the trainer default is DEFAULT_BF16; main()
    # threads it explicitly) so existing direct callers/tests are unaffected.
    bf16: bool = False,
    # Wf-260 / Wf-328-W5 — accepted for API compatibility.  This trainer's
    # joint loop is a list-based manual loop (no torch DataLoader), so there
    # are no worker processes to configure; the kwarg is honoured as a no-op
    # (any value is valid, behaviour is identical).  Kept so callers written
    # against the loader-based signature keep working.
    num_workers: int = 0,
) -> Tuple[float, float, float, float]:
    """Train one epoch on the joint 5-metal pool.

    Returns ``(avg_total_loss, avg_cls_loss, avg_reg_loss, avg_wasserstein,
    last_grad_norm)``.

    Wf-251-F4: ``w2_gradient_mode`` selects the W₂ penalty source —
    ``"sliced"`` (default, Wf-131 P1 shipped behaviour) uses the
    projection-based :func:`wasserstein_pic50_loss`; ``"sort"`` uses
    the closed-form 1-D :func:`wasserstein_pic50_gradient` (more
    numerically stable; modest AUC lift).
    """
    model.train()
    total_loss = total_cls = total_reg = total_w = 0.0
    n_samples = 0
    # Wf-328-W3 — set True if the bf16 autocast block raises; the caller
    # (main()) can then retry the (metal, seed) with --no-bf16.
    bf16_fallback_occurred = False
    # Wf-A9 FIX (2026-10-03) — last observed gradient norm (0.0 until the
    # first clip call runs); returned so callers can emit a real value.
    last_grad_norm = 0.0

    indices = np.arange(len(graphs))
    np.random.shuffle(indices)

    # 2026-10-04 — gradient accumulation window.  ``accum == 1`` (default)
    # keeps the historical per-batch loop bit-for-bit: zero_grad → forward →
    # backward → clip → step, unchanged statement order and no loss scaling.
    accum = max(1, int(gradient_accumulation_steps))
    micro = 0
    # ``pin_memory`` gates the non_blocking hint on the manual H2D copies.
    # Torch documents non_blocking=True as a no-op for pageable tensors, so
    # the default (True) leaves today's copy semantics untouched.
    _non_blocking = bool(pin_memory) and getattr(device, "type", "") == "cuda"

    def _clip_and_step() -> None:
        """Clip grads (max_norm / L2 guard) then take one optimizer step.

        Called once per accumulation window — with the default
        ``gradient_accumulation_steps=1`` that is once per micro-batch, in
        the same order as the pre-flag loop (backward → clip → step).
        """
        nonlocal last_grad_norm
        if max_norm > 0.0:
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            )
        if use_l2_guard:
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), L2_GUARD_MAX_NORM)
            )
        optimizer.step()
        # 2026-10-04 — EMA update, immediately after the step and before
        # anything reads the weights.  No-op when use_ema is off (the snapshot
        # is never created), so the default path is unchanged.
        if bool(use_ema):
            model.update_ema()

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_mids = [metal_ids[i] for i in batch_idx]
        batch_pic50 = pic50[batch_idx]
        batch_active = active[batch_idx]

        batch_data = collate_multimetal(
            batch_graphs, batch_mids, batch_pic50, batch_active
        )
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst, mids_t, pic50_t, active_t = batch_data
        atom_f = atom_f.to(device, non_blocking=_non_blocking)
        bond_f = bond_f.to(device, non_blocking=_non_blocking)
        e_src = e_src.to(device, non_blocking=_non_blocking)
        e_dst = e_dst.to(device, non_blocking=_non_blocking)
        mids_t = mids_t.to(device, non_blocking=_non_blocking)
        pic50_t = pic50_t.to(device, non_blocking=_non_blocking)
        active_t = active_t.to(device, non_blocking=_non_blocking)

        # Zero at the start of each accumulation window.  With accum == 1
        # this is the same per-batch zero_grad the pre-flag loop ran.
        if accum == 1 or micro % accum == 0:
            optimizer.zero_grad()
        # Wf-328-W3 — BF16 autocast wraps the forward + task loss so the
        # matmul-dominated D-MPNN encoder runs in bf16 on tensor-core GPUs.
        # On CPU this is a nullcontext no-op (bit-for-bit unchanged); the
        # try/except flips the fallback flag and re-raises so the caller can
        # retry the (metal, seed) with --no-bf16.
        try:
            with bf16_autocast(device=device, enabled=bool(bf16)):
                pic50_pred, active_logits = model(
                    atom_f, bond_f, e_src, e_dst, mids_t
                )
                loss, lc, lr_ = criterion(
                    pic50_pred, active_logits, pic50_t, active_t
                )
        except (RuntimeError, ValueError):
            # Autocast-side failure ⇒ mark and re-raise for caller retry.
            bf16_fallback_occurred = True
            raise

        # Wasserstein regularisation: W₂(predicted pIC50, reference pIC50)
        #
        # W₂ is *differentiable* w.r.t. the prediction, so the prediction
        # tensor MUST stay in the autograd graph.  Detaching the wrong side
        # (the prediction) silently zeroes the gradient and the W₂ penalty
        # contributes nothing to the parameter update — the trainer then
        # sits at the signal-floor R² (TODO-179 root cause).
        #
        # The reference distribution is a *fixed* tensor (no grad needed),
        # so we detach it instead.
        w_pen = torch.zeros((), device=device)
        if w_w > 0.0 and pic50_pred.numel() >= 4:
            ref_dev = ref_pic50_dist.to(device).detach()
            # 2026-10-08 §5.7 — --use-wasserstein-w1 / -2 explicit order
            # flags take precedence over every mode selector below: they name
            # the closed-form order directly, so they must win regardless of
            # use_ot_cfm / w2_gradient_mode (otherwise setting the flag AND a
            # sort mode would silently ignore the flag).
            if use_wasserstein_2:
                w_pen = wasserstein_pic50_gradient(
                    pic50_pred,
                    ref_dev,
                    w_weight=1.0,
                    wasserstein_order=2,
                )
            elif use_wasserstein_w1:
                w_pen = wasserstein_pic50_gradient(
                    pic50_pred,
                    ref_dev,
                    w_weight=1.0,
                    wasserstein_order=1,
                )
            # Wf-405-A2 — use_ot_cfm flag swap (default OFF preserves
            # legacy W₂). When ON, use OT solver (sort-based W₂,
            # Peyré-Cuturi 2019 §3.1) instead of the legacy Wf-131 P1
            # / Wf-251-F4 sliced/sort path.
            elif use_ot_cfm:
                try:
                    from molmetal.adapters.flow_matching_lipman import (
                        optimal_transport_v2_solver as _ot_v2,
                    )

                    w_pen = _ot_v2.ot_solver(
                        pic50_pred,
                        ref_dev,
                    )
                except Exception:
                    # Defensive fallback to legacy sliced-W₂.
                    w_pen = wasserstein_pic50_loss(
                        pic50_pred,
                        ref_dev,
                        n_mc=w_config.n_mc,
                        seed=w_config.seed,
                    )
            elif w2_gradient_mode == "sort":
                # Wf-251-F4 — closed-form 1-D W₂ via torch.sort.
                # Differentiable w.r.t. pic50_pred (autograd-friendly)
                # and numerically more stable than the projection loop.
                w_pen = wasserstein_pic50_gradient(
                    pic50_pred,
                    ref_dev,
                    w_weight=1.0,
                    wasserstein_order=1,
                )
            else:
                # Default — sliced-W₂ (Wf-131 P1 shipped behaviour).
                w_pen = wasserstein_pic50_loss(
                    pic50_pred,
                    ref_dev,
                    n_mc=w_config.n_mc,
                    seed=w_config.seed,
                )

        # Wf-432-A2 — Wf-246-C metal-coordination term wire-in (Fix-B).
        # When --metal-coord-term is set at the CLI, the trainer builds
        # a MetalCoordinationTerm instance lazily (cached per epoch
        # outside the batch loop) and adds kcal_adjustment() to the
        # composite loss with weight ``metal_coord_weight``. The kcal
        # sign is sign-fixed per Wf-427-A1 metal_coord_term.py:641
        # (Vina convention "lower (more negative) = better binding").
        # Default OFF preserves the Wf-131 P1 + Wf-251-F4 baseline
        # bit-for-bit when --metal-coord-term is False at the CLI.
        # ------------------------------------------------------------------
        # Metal-coordination term — REMOVED from the loss cascade (2026-10-04).
        #
        # It used to be:
        #
        #     _composite_proxy = float(torch.sigmoid(pic50_pred.mean()).item())
        #     _kcal_adj = float(term.kcal_adjustment(score))
        #     mc_term_loss = torch.tensor(_kcal_adj, device=device)
        #     total = loss + w_w * w_pen + metal_coord_weight * mc_term_loss
        #
        # `mc_term_loss` has requires_grad=False — `.item()` has already
        # severed the graph — so the `metal_coord_weight *` term (default 0.5)
        # contributed a constant offset to the loss and exactly zero gradient.
        # Every parameter update was identical to running without it.
        #
        # It is not repairable by deleting the `.item()`: MetalCoordinationTerm
        # is a RULE-BASED scorer (`coordination_geometry_score(smiles)`,
        # `hsab_match_score(smiles, ...)` — RDKit + an HSAB lookup table), so
        # it has no gradient to give.  And feeding it `sigmoid(pic50_pred)` as
        # a stand-in for all three sub-scores, as the old code did, would have
        # made the differentiable version *worse than dead*: minimising
        # `(1 - 2*c)` pushes c up, and c is derived from the prediction being
        # supervised — i.e. the term would teach the model to inflate its own
        # output.
        #
        # Independently, the term cannot fire on this trainer's data at all:
        # `MetalCoordinationTerm.score` detects the metal FROM THE SMILES, and
        # MEASURED, 0 of 800 sampled rows of metalcytotox_5metal contain any
        # metal atom — the metal is a separate `metal` COLUMN.
        #
        # The term already works where it can: vina_adapter.py's
        # `_compute_metal_coord_adjustment` applies it to real docked SMILES as
        # a post-hoc kcal/mol correction.  Making metal chemistry visible to
        # THIS trainer needs a different mechanism (an auxiliary target
        # computed per-row from the `metal` column, which first requires
        # `hsab_match_score` / `coordination_geometry_score` to accept an
        # explicit metal symbol) — not a loss term bolted onto the cascade.
        #
        # Fail loudly rather than silently contribute nothing: a flag whose
        # help text promises a loss contribution must not be a no-op, and the
        # three Wf-388/Wf-409/Wf-406 launchers that pass it would otherwise
        # produce runs indistinguishable from --metal-coord-term being absent
        # while reporting it as active.
        if metal_coord_term:
            raise SystemExit(
                "--metal-coord-term cannot be used with this trainer. "
                "MetalCoordinationTerm is a rule-based SMILES scorer (no "
                "gradient), and this trainer's corpus carries the metal in a "
                "separate column rather than in the SMILES (MEASURED: 0 of "
                "800 sampled rows contain a metal atom), so the term would "
                "score 0.0 even if it were differentiable.  Previously it "
                "silently contributed a constant to the loss.  The term is "
                "already wired where it works: "
                "molmetal/molmetal_lam/sbdd_env/vina_adapter.py::"
                "_compute_metal_coord_adjustment, applied to docked SMILES. "
                "Drop the flag to train, or see "
                "TODO/completed/431a3_wf246c_scafffold_close.md for the "
                "SCAFFOLD_NOT_FUNCTIONAL close."
            )

        total = loss + w_w * w_pen
        if accum > 1:
            # Gradient accumulation: the composite loss is divided by N and
            # the gradients of N micro-batches accumulate, so one optimizer
            # step consumes the window's mean loss.  accum == 1 takes the
            # plain backward() branch below — no division is even performed
            # — which keeps the default path bit-for-bit identical.
            (total / accum).backward()
        else:
            total.backward()
        # Wf-A8 (2026-10-03) gradient clipping per --max-norm flag (per
        # AGENTS.md §5.7 trainer flag wire-in smoke-required rule).
        # Wf-A9 FIX (2026-10-03) — ``max_norm`` now arrives as a kwarg
        # threaded ``main()`` → ``train_one_metal_one_seed()`` → here.
        # The previous ``getattr(args, ...)`` form raised NameError
        # because ``args`` is ``main()``-scoped (AGENTS.md §5.7 floor #2).
        # Wf-A9 FIX (2026-10-03) — the last clip call's return value IS
        # the total gradient norm; surface it so the TUI/events report a
        # real number instead of the hard-coded NaN placeholder it used
        # to emit (see the train_epoch event emission below).
        micro += 1
        if accum == 1 or micro % accum == 0:
            _clip_and_step()

        bs = pic50_t.size(0)
        total_loss += loss.item() * bs
        total_cls += lc.item() * bs
        total_reg += lr_.item() * bs
        total_w += float(w_pen.item()) * bs
        n_samples += bs

    # Trailing partial accumulation window (accum > 1 only): step the
    # leftover micro-batches instead of silently dropping their gradients.
    # Their losses were scaled by 1/accum like every other micro-batch.
    if accum > 1 and micro % accum != 0:
        _clip_and_step()

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / n_samples,
        total_cls / n_samples,
        total_reg / n_samples,
        total_w / n_samples,
        last_grad_norm,
    )


@torch.no_grad()
def predict_joint(
    model: MultiTaskWassersteinModel,
    graphs: List[Optional[Tuple]],
    metal_ids: List[int],
    pic50: np.ndarray,
    active: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict (active_true, active_prob, pic50_pred) for one metal's split.

    Invalid molecules contribute (label, 0.0, 0.0) so callers don't lose
    them when slicing.
    """
    model.eval()
    all_active = []
    all_score = []
    all_pic50 = []

    indices = np.arange(len(graphs))
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_mids = [metal_ids[i] for i in batch_idx]
        batch_pic50 = pic50[batch_idx]
        batch_active = active[batch_idx]

        batch_data = collate_multimetal(
            batch_graphs, batch_mids, batch_pic50, batch_active
        )
        if batch_data is None:
            # Wf-ultracode FIX — skip invalid rows entirely.  The unfixed
            # behaviour appended (int(a), 0.0, 0.0) which biases ROC down
            # for active=1 invalid SMILES (guaranteed false-negative).
            continue
        atom_f, bond_f, e_src, e_dst, mids_t, pic50_t, active_t = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        mids_t = mids_t.to(device)

        pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst, mids_t)
        active_prob = torch.softmax(active_logits, dim=-1)[:, 1].cpu().numpy()
        pic50_arr = pic50_pred.cpu().numpy()

        vi = 0
        for g, p, a in zip(batch_graphs, batch_pic50, batch_active):
            valid_row = True
            if g is None:
                valid_row = False
            else:
                try:
                    if math.isnan(float(p)):
                        valid_row = False
                except (TypeError, ValueError):
                    valid_row = False
            if valid_row:
                all_active.append(int(active_t[vi].item()))
                all_score.append(float(active_prob[vi]))
                all_pic50.append(float(pic50_arr[vi]))
                vi += 1
            # Wf-ultracode FIX — invalid rows are excluded entirely from
            # y_true/y_score/y_pic50.  The unfixed code appended
            # (int(a), 0.0, 0.0) which biased ROC down for active=1
            # invalid SMILES (guaranteed false-negative).

    return (
        np.array(all_active, dtype=int),
        np.array(all_score, dtype=float),
        np.array(all_pic50, dtype=float),
    )


def train_one_metal_one_seed(
    metal: str,
    seed: int,
    epochs: int,
    alpha: float,
    lr: float,
    batch_size: int,
    w_w: float,
    use_l2_guard: bool,
    ds: MetalCytotoxDataset,
    split_df: pd.DataFrame,
    ckpt_dir: Path,
    device: torch.device,
    w_config: WassersteinConfig,
    single_task: bool = False,
    hidden: Optional[int] = None,
    depth: Optional[int] = None,
    w2_gradient_mode: str = DEFAULT_W2_GRADIENT_MODE,
    # Wf-427-A3 — Wf-405-A2 + Wf-424-A1 wire-in fix.  These kwargs were
    # referenced as ``args.<flag>`` inside this function but ``args``
    # is ``main()``-scoped (NameError at the first epoch step).  Threaded
    # as direct kwargs from the ``main()`` caller so the Wf-380 NEW CLI
    # flags + the Wf-424-A1 DANN flag actually reach the loss-cascade /
    # model constructor; default values (False / False / False)
    # preserve bit-for-bit backward compat with Wf-131 P1 + Wf-251-F4
    # baseline.
    use_ot_cfm: bool = False,
    # 2026-10-08 §5.7 — thread --use-wasserstein-w1 / -2 to the loss cascade.
    use_wasserstein_w1: bool = False,
    use_wasserstein_2: bool = False,
    per_cohort_features: bool = False,
    use_dann: bool = False,
    # Wf-432-A2 — Wf-246-C metal-coordination term wire-in (Fix-B).
    # Default values (False / 0.5 / True) preserve bit-for-bit
    # backward compat with Wf-131 P1 + Wf-251-F4 baseline when
    # --metal-coord-term is False at the CLI. Threaded as direct
    # kwargs from the ``main()`` caller so the Wf-246-C kcal term
    # actually reaches the loss cascade in ``train_one_epoch_joint``.
    metal_coord_term: bool = False,
    metal_coord_weight: float = 0.5,
    cal_sign_vina_convention: bool = True,
    # Wf-A5 N3 — checkpoint resumption kwargs (default 'auto' / 50
    # preserve bit-for-bit backward compat with Wf-131 P1 + Wf-251-F4
    # baseline when no checkpoint exists — cold start).  Threaded as
    # direct kwargs from the ``main()`` caller so the Wf-A5 N3 resume
    # flag actually triggers the load logic at the start of this
    # function (see ``_load_checkpoint_for_resume`` call below).
    resume_from: str = "auto",
    resume_from_path: Optional[Path] = None,
    checkpoint_interval: int = 50,
    # Wf-R29 — Wf-335-P3 Triton dispatch flag wire-in.  ``--use-triton-relu-mlp``
    # had a real argparse dest AND a live consumer (the ``self.use_triton_relu_mlp``
    # branch in ``MultiTaskWassersteinModel.__init__`` that swaps the eager
    # Sequential heads for ``_FusedReLUMLPWrapper``), but main() never threaded
    # it into the model constructor — so the YAML key configs/dmpnn.yaml
    # ``model.use_triton_relu_mlp`` resolved fine and then went nowhere.
    # That is the Wf-385-A1 FLAG_ONLY class.  Default False keeps the
    # bit-for-bit Wf-131-P1 baseline.
    use_triton_relu_mlp: bool = False,
    # Wf-A6 A2 BLOCKER C fix — LR scheduler (cosine / onecycle) opt-in
    # kwargs thread from main(). Default OFF preserves Wf-131 P1 +
    # Wf-251-F4 baseline bit-for-bit (lr_scheduler='none' means
    # Adam's fixed LR is used throughout).
    lr_scheduler: str = "none",
    lr_warmup_epochs: int = 5,
    lr_min_lr: float = 1e-6,
    # Wf-A6 A2 BLOCKER C fix (audit gap R5) — Adam weight_decay
    # threaded from main() via getattr(args, "weight_decay", 0.0).
    # Default 0.0 preserves Wf-131 P1 + Wf-251-F4 baseline bit-for-bit.
    wd: float = 0.0,
    # Wf-A9 FIX (2026-10-03) — gradient-clip threshold threaded from
    # main() via ``getattr(args, "max_norm", 5.0)`` and forwarded to
    # ``train_one_epoch_joint()``.  Sourced from the yaml
    # ``protocol.max_norm`` via start.sh ``--max-norm "$MAX_NORM"``.
    max_norm: float = 5.0,
    # Wf-A9 FIX (2026-10-03) — the resolved joint metals pool (from
    # ``--metals`` / ``--dataset``).  ``None`` keeps the historical
    # FIVE_METALS pool so existing callers are unaffected; main() passes
    # the resolved list so non-5-metal cohorts (e.g. platinai_4cell = Pt)
    # build a correctly sized metal-embedding table.
    metals_pool: Optional[Tuple[str, ...]] = None,
    # T1-FIX (2026-10-03) — AGENTS.md §5.7 FLAG_ONLY wire-in.  The
    # ``--max-joint-rows`` CLI flag was declared in the argparse block
    # but never read; this kwarg threads its value into the live joint
    # pool assembly below.  ``None`` / ``0`` = full pool (historical
    # behaviour bit-for-bit); a positive value below the pool size
    # applies a seeded random subsample.
    max_joint_rows: Optional[int] = None,
    # 2026-10-04 — GPU-utilisation flags (compute.torch_compile / tf32 /
    # cudnn_benchmark / channels_last).  Threaded as kwargs from main() for
    # the same reason Wf-427-A3 threaded use_dann/per_cohort_features: this
    # function has NO access to the main()-scoped ``args`` object, and
    # referencing ``args.<flag>`` here raises NameError at the first epoch —
    # which is exactly what this block did on its first attempt.  Defaults
    # OFF keep the no-flag path bit-for-bit identical to Wf-131 P1.
    torch_compile: bool = False,
    tf32: bool = DEFAULT_TF32,
    bf16: bool = DEFAULT_BF16,
    torch_compile_mode: str = DEFAULT_TORCH_COMPILE_MODE,
    dmpnn_backend: str = DEFAULT_DMPNN_BACKEND,
    cudnn_benchmark: bool = False,
    channels_last: bool = False,
    # 2026-10-04 — the remaining two dead yaml keys
    # (``protocol.gradient_accumulation_steps`` / ``data.pin_memory``)
    # threaded as explicit kwargs for the same reason as the block at the
    # top of this signature: this function has NO access to the
    # main()-scoped ``args`` object, and a ``getattr(args, ...)`` here is a
    # NameError at the first epoch (AGENTS.md §5.7 floor #2).  Defaults
    # (1 / True) preserve the pre-flag loop bit-for-bit.
    gradient_accumulation_steps: int = 1,
    pin_memory: bool = True,
    # 2026-10-04 — EMA kwargs (the yaml `train.ema_decay`), threaded the same
    # way for the same reason: no access to the main()-scoped `args` here.
    use_ema: bool = False,
    ema_decay: float = 0.999,
    # 2026-10-05 — ``--kendall`` CLI flag wire-in (AGENTS.md §5.7 FLAG_ONLY).
    # The argparse block declared ``--use-kendall`` / ``--no-kendall`` →
    # ``dest='kendall'`` but no reader existed in the trainer; this kwarg
    # threads the value into ``MultiTaskLoss(use_kendall_loss=...)`` so the
    # Kendall tau UW MTL loss actually fires when ``--use-kendall`` is set.
    # Default False preserves Wf-131 P1 + Wf-251-F4 baseline bit-for-bit.
    use_kendall_loss: bool = False,
    # 2026-10-08 — ridge readout on the frozen encoder embedding (opt-in).
    # Reported ALONGSIDE the neural head's pearson; default False keeps the
    # Wf-131 P1 metric path bit-for-bit.
    pic50_ridge_readout: bool = False,
) -> MetalSeedResult:
    """Train a 5-metal multi-task D-MPNN for one (metal, seed).

    Re-uses the shared encoder across all metals (joint training); per-
    metal test/val metrics are reported for the *target* metal only.
    Other metals contribute to the encoder's shared signal but are not
    individually evaluated on test (this matches the Wf-120 P1 recipe).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # Build split indices for ALL 5 metals (joint training pool)
    smiles_all = np.asarray(ds.smiles).astype(str)
    pic50_all = np.asarray(ds.pic50).astype(np.float32)
    active_all = np.asarray(ds.active).astype(np.int64)
    metals_all = np.asarray(ds.metals).astype(str)

    # Wf-197 ablation: single-task (Ru-only) mode. When --single-task is
    # set, the joint training pool is built from the *target metal only*
    # and the model has n_metals=1 (no per-metal head sharing across
    # metals). Default (multi-task) behaviour is unchanged for backward
    # compat with Wf-124 P3 + Wf-131 P1 results.
    # Wf-A9 FIX (2026-10-03) — the joint pool was structurally pinned to
    # FIVE_METALS while main() already reported ``n_metals=len(args.metals)``
    # (see the "[wf131-mtw] ... n_metals=" banner).  That mismatch made any
    # non-5-metal cohort impossible: with --dataset platinai_4cell the CSV
    # holds only Pt, so ``metal_to_idx[str(metals_all[i])]`` raised KeyError
    # and the run died before the first batch.
    #
    # The pool now follows the resolved ``--metals`` list.  Order is
    # canonicalised to FIVE_METALS first so the standard 5-metal pool is
    # bit-for-bit unchanged (metal-embedding indices are positional);
    # metals outside that set (e.g. Pt) are appended in caller order.
    if single_task:
        metals_in_pool: Tuple[str, ...] = (metal,)
    else:
        _pool: Tuple[str, ...] = tuple(metals_pool) if metals_pool else FIVE_METALS
        metals_in_pool = tuple(
            [m for m in FIVE_METALS if m in _pool]
            + [m for m in _pool if m not in FIVE_METALS]
        )
        if not metals_in_pool:
            raise RuntimeError(
                "empty joint metals pool — check --metals / --dataset "
                "resolution (see the '[Wf-A6 BLOCKER A] --metals resolved' "
                "banner above)."
            )
    n_metals_pool: int = 1 if single_task else len(metals_in_pool)

    metal_to_idx: Dict[str, int] = {m: i for i, m in enumerate(metals_in_pool)}
    per_metal_indices: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for m in metals_in_pool:
        try:
            per_metal_indices[m] = build_metal_split_indices(split_df, ds, m)
        except RuntimeError:
            per_metal_indices[m] = (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
            print(
                f"WARNING: metal {m!r} silently dropped from joint pool "
                f"(build_metal_split_indices raised RuntimeError)",
                file=sys.stderr,
                flush=True,
            )
    # Wf-ultracode FIX — recompute n_metals_pool to reflect metals that
    # actually contributed non-empty train indices (the bare-except above
    # can silently drop a metal without raising).  Without this recompute,
    # the model is constructed with n_metals=len(metals_in_pool)=5 even
    # when only 4 metals actually joined, which corrupts the metal-embedding
    # table indexing downstream.
    if not single_task:
        n_metals_pool = sum(
            1 for m in metals_in_pool
            if len(per_metal_indices[m][0]) > 0
        )

    # Concatenate train rows from pool metals (5-metal joint OR Ru-only)
    train_pool_idx = np.concatenate(
        [
            per_metal_indices[m][0]
            for m in metals_in_pool
            if len(per_metal_indices[m][0]) > 0
        ]
    )
    if len(train_pool_idx) < 50:
        raise RuntimeError(
            f"joint train pool too small: {len(train_pool_idx)} rows (< 50)"
        )
    # T1-FIX (2026-10-03) — apply --max-joint-rows to the LIVE pool.
    # Placed after the <50 floor check (which measures the true pool)
    # and before train_pool_mids / featurisation, so the cap shrinks the
    # featurised union AND the joint batch pool — not just a recorded
    # number.  Seeded permutation (NOT head-truncation): the pool is a
    # metal-blocked concatenate in FIVE_METALS order, so a head cap
    # would keep only the first metal's rows and make the smoke
    # silently single-metal.  When the cap binds, n_joint_train in the
    # checkpoint reflects the capped count while n_train/n_val/n_test
    # remain the target-metal split sizes (see --max-joint-rows help).
    if max_joint_rows and 0 < max_joint_rows < len(train_pool_idx):
        _cap_rng = np.random.default_rng(seed)
        train_pool_idx = train_pool_idx[
            _cap_rng.permutation(len(train_pool_idx))[:max_joint_rows]
        ]
    # Per-row metal id (mapped to FIVE_METALS index)
    train_pool_mids = np.array(
        [metal_to_idx[str(metals_all[i])] for i in train_pool_idx],
        dtype=np.int64,
    )

    target_train, target_val, target_test = per_metal_indices[metal]
    if len(target_train) < 50:
        raise RuntimeError(
            f"metal={metal} seed={seed}: only {len(target_train)} train rows (< 50) — skip"
        )

    # Featurise union of all rows we touch (joint + target val/test)
    used_idx = np.unique(
        np.concatenate(
            [
                train_pool_idx,
                target_val,
                target_test,
            ]
        )
    )
    used_smiles = smiles_all[used_idx]
    t_fe = time.time()
    cache_key = f"wf131_mtw_{metal.lower()}_seed{seed}"
    graphs_all = _featurize_with_cache(
        [str(s) for s in used_smiles],
        cache_key=cache_key,
        cache_dir=Path("/tmp/wf131_mtw_fe_cache"),
    )
    featurise_seconds = time.time() - t_fe

    pos_in_used = {int(used_idx[k]): k for k in range(len(used_idx))}
    graphs_all_full = [None] * len(smiles_all)
    for di in used_idx:
        graphs_all_full[di] = graphs_all[pos_in_used[int(di)]]

    # Build train graphs / mids / targets (joint pool)
    train_graphs = [graphs_all_full[i] for i in train_pool_idx]
    train_mids = train_pool_mids.tolist()
    train_pic50 = pic50_all[train_pool_idx]
    train_active = active_all[train_pool_idx]

    # Reference pIC50 distribution = the full training pool (1-D, fixed)
    ref_pic50 = torch.from_numpy(train_pic50.astype(np.float32))

    # Target metal val/test graphs (for evaluation)
    target_val_graphs = [graphs_all_full[i] for i in target_val]
    target_val_mids = [metal_to_idx[metal]] * len(target_val)
    target_val_pic50 = pic50_all[target_val]
    target_val_active = active_all[target_val]

    target_test_graphs = [graphs_all_full[i] for i in target_test]
    target_test_mids = [metal_to_idx[metal]] * len(target_test)
    target_test_pic50 = pic50_all[target_test]
    target_test_active = active_all[target_test]

    # Build model + optimizer + criterion
    # Wf-424-A1 + Wf-427-A3 — wire ``use_dann`` (CLI --use-dann /
    # --no-dann) into the model constructor.  Default OFF preserves
    # Wf-131 P1 / Wf-380 backward compat bit-for-bit (the Wf-280D
    # BooleanOptionalAction defaults to True but ``use_dann`` default
    # here is False so OFF means ``use_dann=False`` — the model
    # constructor's default takes precedence over the CLI default
    # when the test exercises OFF explicitly).  The kwarg is now
    # threaded from main() via the ``use_dann`` parameter (not the
    # main-scoped ``args`` object, which raised NameError pre-Wf-427-A3).
    model = MultiTaskWassersteinModel(
        n_metals=n_metals_pool,
        hidden=hidden if hidden is not None else DEFAULT_HIDDEN,
        depth=depth if depth is not None else DEFAULT_DEPTH,
        dropout=DEFAULT_DROPOUT,
        use_dann=use_dann,
        use_triton_relu_mlp=use_triton_relu_mlp,
        dmpnn_backend=dmpnn_backend,
    ).to(device)
    # 2026-10-04 — GPU-utilisation wire-in, mirroring Wf-431-B1 in
    # train_cfm_yuelbond_wasserstein.py.  Default OFF; the no-flag path is
    # bit-for-bit unchanged.  Order matters: torch.backends.* BEFORE
    # torch.compile so the compiled graph inherits the precision settings.
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("[compute] --cudnn-benchmark ENABLED", flush=True)
    if tf32:
        # Wf-328-W4 — route through the named helper so the three TF32 knobs
        # cannot drift out of sync; read-back state is logged (and persisted
        # into training_summary.json by main()).
        _tf32_state = configure_tf32(enabled=True).to_dict()
        print(
            f"[compute] --tf32 ENABLED (matmul precision = "
            f"{_tf32_state['matmul_precision']})",
            flush=True,
        )
    else:
        # Explicit OFF keeps the levers consistent (Wf-328-W4 §6 lever).
        configure_tf32(enabled=False)
        print("[compute] --tf32 DISABLED (matmul precision = highest)", flush=True)
    # 2026-10-04 — parameter EMA.  Initialised AFTER the model is on its
    # target device so the snapshot lands on the same device as the
    # parameters, and BEFORE any torch.compile wrapper (which would otherwise
    # hide the parameters behind an OptimizedModule).
    #
    # NOTE: this runs inside `train_one_metal_one_seed`, NOT in `main()` — so
    # it reads the `use_ema` / `ema_decay` KWARGS threaded into this function,
    # never a main()-scoped `args`.  The first draft of this block did use
    # `getattr(args, ...)` and produced the exact `NameError: name 'args' is
    # not defined` that the comment at the top of this signature warns about
    # (AGENTS.md §5.7 floor #2) — caught by the smoke, not by reading.
    if bool(use_ema):
        model.init_ema(decay=float(ema_decay))
        print(
            f"[dmpnn-ema] enabled: decay={float(ema_decay)} "
            "(snapshot taken from freshly-initialised weights)",
            flush=True,
        )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        print(
            "[compute] --channels-last ENABLED (note: a D-MPNN has no "
            "spatial axes, so this is expected to be a no-op)",
            flush=True,
        )
    if torch_compile:
        # dynamic_shapes BEFORE any compile call: the training loop produces
        # variable batch sizes when n_train % batch_size != 0.
        torch._dynamo.config.dynamic_shapes = True
        # Wf-335-R1 — resolve the effective Inductor mode.  'reduce-overhead'
        # via the v1 flag is REJECTED (it enables CUDA Graphs that fire the
        # Wf-330-B4 RuntimeError); the v2 flag re-opts-in for bench repro.
        _mode = str(torch_compile_mode)
        try:
            model.encoder._forward_one_mol = _make_compiled_encoder_forward(
                model.encoder, mode=_mode
            )
            print(
                f"[compute] --torch-compile ENABLED (encoder forward, "
                f"mode={_mode})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            # A corrupt inductor cache must not kill the run; eager is
            # bit-for-bit the no-flag path.
            print(
                f"[compute] --torch-compile FAILED ({type(exc).__name__}: {exc}); "
                f"falling back to eager. Clear with: rm -rf /tmp/torchinductor_*",
                flush=True,
            )
    # Wf-328-W3 — BF16 autocast state.  Resolved here (before the epoch loop)
    # so the state can be persisted; the actual autocast window is opened per
    # batch around the forward + W₂ block in train_one_epoch_joint().
    _bf16_state = configure_bf16(enabled=bf16, device=device)
    print(
        f"[compute] --bf16 requested={_bf16_state['requested']} "
        f"apply={_bf16_state['apply']} "
        f"fallback={_bf16_state['fallback_occurred']}",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Wf-A6 A2 BLOCKER C fix — instantiate LR scheduler (opt-in).
    # Default OFF (lr_scheduler='none') preserves Wf-131 P1 + Wf-251-F4
    # baseline bit-for-bit. Cosine / OneCycle variants compose
    # linear warmup + main scheduler via SequentialLR per
    # build_lr_scheduler() helper. The actual step() call fires once
    # per epoch (end of inner loop) for cosine; for onecycle the
    # helper emits OneCycleLR which expects per-step step() calls —
    # we instead call scheduler.step() once per epoch which is a
    # documented alternative (Smith-Topin 2018 §3.3 discusses
    # batch-level vs epoch-level step for OneCycleLR; epoch-level
    # is functionally equivalent up to total_steps divisor).
    # 2026-10-04 — with --gradient-accumulation-steps N the optimizer steps
    # once per N micro-batches, so the scheduler's step count is the window
    # count, not the micro-batch count.  N == 1 divides by 1 → the schedule
    # is bit-for-bit the pre-flag one.
    steps_per_epoch = max(
        1,
        (len(train_mids) // max(1, batch_size))
        // max(1, int(gradient_accumulation_steps)),
    )
    total_steps = max(1, int(epochs) * int(steps_per_epoch))
    warmup_steps = max(0, int(lr_warmup_epochs) * int(steps_per_epoch))
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        scheduler_type=str(lr_scheduler),
        warmup_steps=warmup_steps,
        min_lr=float(lr_min_lr),
    )
    if scheduler is not None:
        print(
            f"[lr-sched] {metal} seed={seed}: type={lr_scheduler} "
            f"warmup_epochs={lr_warmup_epochs} steps/epoch={steps_per_epoch} "
            f"total_steps={total_steps} min_lr={lr_min_lr:.2e}",
            flush=True,
        )

    # Wf-A5 N3 — checkpoint resumption: load prior state if present.
    # MUST come after construction so model + optimizer are valid objects.
    # Bit-for-bit backward compat with Wf-131 P1 + Wf-251-F4 baseline:
    # when resume_from='none' OR no checkpoint exists, this is a no-op.
    # The optimizer state is also reloaded so resumed training continues
    # from the saved Adam state (no lr spike).
    start_epoch, ckpt_w_history = _load_checkpoint_for_resume(
        model=model,
        optimizer=optimizer,
        metal=metal,
        seed=seed,
        ckpt_dir=ckpt_dir,
        resume_from=resume_from,
        resume_from_path=resume_from_path,
        device=device,
    )
    # Wf-TUI NEW: emit structured resume event.  When start_epoch > 0,
    # the trainer is resuming from a checkpoint; downstream consumers
    # (TUI dashboard, monitoring scripts) use this event to render the
    # "resumed from epoch N" status row + align the epoch counter.
    if start_epoch > 0:
        try:
            emit_event(
                "resume",
                metal=metal,
                seed=seed,
                from_epoch=int(start_epoch),
                total_epochs=int(epochs),
                ckpt_dir=str(ckpt_dir),
                w_history_len=int(len(ckpt_w_history)),
                lr=float(lr),
            )
        except Exception:
            # Defensive: never let JSONL emission crash training.
            pass
    # Wf-405-A2 + TODO-201 wire-in (Wf-201 forward-looking): when
    # --per-cohort-features is set, override the BCE `alpha` based on
    # the *actual* joint pool size (not the metals×200 estimate used at
    # main()-level logging).  Direct kwarg threading so the per-cohort
    # config actually reaches the loss cascade.  Default OFF preserves
    # Wf-131 P1 + Wf-251-F4 baseline bit-for-bit (per_wf405a2 §0).
    effective_alpha = alpha
    if per_cohort_features:
        try:
            from molmetal.scripts.wf380a2_m2_r2_per_cohort import (
                select_features,
            )

            _pc = select_features(int(len(train_pool_idx)))
            effective_alpha = float(_pc.get("alpha", alpha))
            print(
                f"[Wf-405-A2] --per-cohort-features ENABLED: "
                f"tier={_pc.get('tier', '?')} "
                f"features={_pc.get('features', '?')} "
                f"n_pool={len(train_pool_idx)} "
                f"alpha_override={alpha:+.3f}->{effective_alpha:+.3f}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[Wf-405-A2][WARNING] --per-cohort-features override "
                f"failed: {exc}; falling back to alpha={alpha:.3f}.",
                flush=True,
            )
    # 2026-10-05 — ``--kendall`` CLI flag wire-in: thread the boolean into
    # ``MultiTaskLoss(use_kendall_loss=...)`` so the Kendall tau UW MTL
    # loss cascade activates when the user passes ``--use-kendall``.  See
    # AGENTS.md §5.7 — the Wf-280D flag was declared in the argparse block
    # but never reached the loss cascade; this closes that FLAG_ONLY gap.
    # Default ``use_kendall_loss=False`` (the kwarg's default) keeps the
    # no-flag path bit-for-bit identical to Wf-131 P1.
    criterion = MultiTaskLoss(
        alpha=effective_alpha, use_kendall_loss=use_kendall_loss,
    )

    best_val_auc = -1.0
    best_state: Optional[Dict] = None
    best_epoch = 0
    w_history: List[float] = ckpt_w_history  # Wf-A5 N3 — extend from resumed state

    t_train = time.time()
    for epoch in range(start_epoch, epochs):
        ep_t0 = time.time()
        avg_loss, avg_cls, avg_reg, avg_w, epoch_grad_norm = train_one_epoch_joint(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            w_w=w_w,
            use_l2_guard=use_l2_guard,
            graphs=train_graphs,
            metal_ids=train_mids,
            pic50=train_pic50,
            active=train_active,
            ref_pic50_dist=ref_pic50,
            batch_size=batch_size,
            device=device,
            w_config=w_config,
            w2_gradient_mode=w2_gradient_mode,
            # Wf-427-A3 — fix the Wf-405-A2 wire-in NameError.  The
            # ``args`` reference was ``main()``-scoped; the local kwargs
            # threaded in from the caller are the canonical source of
            # truth for the flag state.
            use_ot_cfm=use_ot_cfm,
            # 2026-10-08 §5.7 — thread the W₁/W₂ closed-form order flags.
            use_wasserstein_w1=use_wasserstein_w1,
            use_wasserstein_2=use_wasserstein_2,
            per_cohort_features=per_cohort_features,
            # Wf-A9 FIX (2026-10-03) — thread the gradient-clip threshold
            # through to the loss-cascade clipping site.
            max_norm=max_norm,
            # Wf-432-A2 — Wf-246-C metal-coordination term wire-in
            # (Fix-B).  These kwargs reach the loss cascade in
            # ``train_one_epoch_joint`` (see line ~520) so the kcal/mol
            # adjustment contributes to the composite loss with
            # weight ``metal_coord_weight`` (default 0.5). The
            # ``cal_sign_vina_convention`` flag gates the Wf-427-A1
            # line 641 sign fix.
            metal_coord_term=metal_coord_term,
            metal_coord_weight=metal_coord_weight,
            cal_sign_vina_convention=cal_sign_vina_convention,
            # 2026-10-04 — gradient accumulation window + pin-memory transfer
            # hint, threaded from main() (the values live on the caller's
            # namespace; this function never reads ``args``).
            gradient_accumulation_steps=gradient_accumulation_steps,
            pin_memory=pin_memory,
            # 2026-10-04 — EMA kwargs, threaded the same way (never read off a
            # main()-scoped `args` from inside this function).
            use_ema=use_ema,
            ema_decay=ema_decay,
            # Wf-328-W3 — BF16 autocast toggle (threaded from main()).
            bf16=bf16,
        )
        w_history.append(float(avg_w))

        # Validate (target metal only)
        y_true_v, y_score_v, pic50_pred_v = predict_joint(
            model,
            target_val_graphs,
            target_val_mids,
            target_val_pic50,
            target_val_active,
            batch_size=batch_size,
            device=device,
        )
        if len(set(y_true_v.tolist())) >= 2:
            val_auc = float(roc_auc_score(y_true_v, y_score_v))
        else:
            val_auc = float("nan")
        val_pic50_m = regression_metrics(target_val_pic50, pic50_pred_v)
        ep_t = time.time() - ep_t0
        if (epoch + 1) % WASSERSTEIN_LOG_EVERY == 0 or epoch == 0:
            print(
                f"  [{metal} seed={seed}] epoch {epoch+1:>2}/{epochs} "
                f"loss={avg_loss:.4f} cls={avg_cls:.4f} reg={avg_reg:.4f} "
                f"w={avg_w:.4f} val_auc={val_auc:.4f} "
                f"val_pearson={val_pic50_m['pearson']:.3f} "
                f"({ep_t:.1f}s)",
                flush=True,
            )

        # Wf-TUI NEW: structured per-epoch event to stdout + events.jsonl
        # via the central emit_event() helper. TUI dashboard consumers
        # parse the JSON line for live metrics; text-log consumers
        # continue to grep the legacy print() line above (preserved for
        # backward compat per AGENTS.md §5.2 honest-framing).
        try:
            emit_event(
                "train_epoch",
                epoch=int(epoch + 1),
                total_epochs=int(epochs),
                metal=metal,
                seed=seed,
                train_loss=float(avg_loss),
                train_cls=float(avg_cls),
                train_reg=float(avg_reg),
                train_w=float(avg_w),
                val_auc=(
                    float(val_auc)
                    if not (isinstance(val_auc, float) and math.isnan(val_auc))
                    else None
                ),
                val_pearson=float(val_pic50_m.get("pearson", float("nan")))
                if "pearson" in val_pic50_m
                else None,
                lr=float(optimizer.param_groups[0]["lr"]),
                # Wf-A9 FIX (2026-10-03) — was a hard-coded NaN placeholder
                # (`float("nan") if ep_t > 0 else 0.0`), which made the TUI
                # gradient-norm row read NaN for every epoch.  Now carries
                # the real post-clip total norm returned by
                # train_one_epoch_joint().
                grad_norm=float(epoch_grad_norm),
                epoch_time_sec=float(ep_t),
            )
        except Exception:
            # Defensive: never let JSONL emission crash training
            # (Wf-A4-final huddle design contract).
            pass

        # Wf-TUI NEW: lr_step event.  Adam has a fixed LR per args.lr;
        # no scheduler step occurs, but we emit the snapshot for the
        # TUI dashboard so it can render the "current LR" row aligned
        # with the per-epoch train event.
        try:
            emit_event(
                "lr_step",
                epoch=int(epoch + 1),
                metal=metal,
                seed=seed,
                lr=float(optimizer.param_groups[0]["lr"]),
                phase="constant",
            )
        except Exception:
            pass

        # Wf-TUI NEW: append per-epoch event to JSONL (live TUI feed).
        # Uses the canonical module-level emit_event() helper (line ~170)
        # which reads from the module-global _EVENT_LOG_PATH set by
        # _install_event_emitter() in main(). The helper manages stdout
        # JSON + JSONL append in one call. Bit-for-bit backward-compat
        # when --no-emit-json-events is set (helper falls back to text).
        try:
            emit_event(
                "eval_epoch",
                metal=metal,
                seed=seed,
                epoch=int(epoch + 1),
                total_epochs=int(epochs),
                loss=float(avg_loss),
                val_auc=float(val_auc) if not math.isnan(float(val_auc)) else None,
                w2=float(avg_w),
                elapsed_s=float(ep_t),
                ckpt_path=None,
            )
        except Exception:
            # Defensive: never let JSONL emission crash training.
            pass

        # Best epoch = max val AUC (single metric for honest reporting)
        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        # Wf-A6 A2 BLOCKER C fix — LR scheduler step (per-epoch).
        # Default OFF (scheduler is None) preserves Wf-131 P1 baseline
        # bit-for-bit. Cosine + SequentialLR get one step per epoch
        # (their semantics expect per-step; we accumulate via
        # inner_steps tracking). OneCycleLR's per-epoch step is
        # functionally equivalent up to total_steps divisor (Smith-Topin
        # 2018 §3.3 documented alternative).
        if scheduler is not None:
            scheduler.step()

        # Wf-A5 N3 — checkpoint resumption: save mid-training checkpoint
        # every ``checkpoint_interval`` epochs (default 50).  Final epoch
        # always saves (handled by the post-loop block below).
        is_last_epoch = epoch == epochs - 1
        if checkpoint_interval > 0 and (
            (epoch + 1) % checkpoint_interval == 0 or is_last_epoch
        ):
            _save_resumable_checkpoint(
                ckpt_dir=ckpt_dir,
                metal=metal,
                seed=seed,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val_auc=best_val_auc,
                wasserstein_history=w_history,
                alpha=alpha,
                w_w=w_w,
                use_l2_guard=use_l2_guard,
                lr=lr,
                batch_size=batch_size,
                hidden=hidden if hidden is not None else DEFAULT_HIDDEN,
                depth=depth if depth is not None else DEFAULT_DEPTH,
            )

    if best_state is None:
        # Fallback: use last epoch weights.  Wf-ultracode FIX — set
        # best_epoch = -1 as a sentinel so the downstream JSON surfaces
        # the "no best AUC ever won" state honestly, instead of silently
        # saving best_epoch=epochs + val_auc=NaN (the unfixed behaviour).
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        best_epoch = -1
        print(
            f"WARNING: {metal} seed={seed}: no epoch produced a finite "
            f"val_auc that beat the initial -1.0 floor; falling back to "
            f"last-epoch weights (best_epoch=-1, val_auc will be NaN).",
            file=sys.stderr,
            flush=True,
        )
    train_seconds = time.time() - t_train

    # Reload best + eval on val + test (target metal only)
    model.load_state_dict(best_state)
    y_true_v2, y_score_v2, pic50_pred_v2 = predict_joint(
        model,
        target_val_graphs,
        target_val_mids,
        target_val_pic50,
        target_val_active,
        batch_size=batch_size,
        device=device,
    )
    y_true_t, y_score_t, pic50_pred_t = predict_joint(
        model,
        target_test_graphs,
        target_test_mids,
        target_test_pic50,
        target_test_active,
        batch_size=batch_size,
        device=device,
    )
    val_m = _safe_classification_metrics(y_true_v2, y_score_v2)
    test_m = _safe_classification_metrics(y_true_t, y_score_t)
    val_pic50_m_final = regression_metrics(target_val_pic50, pic50_pred_v2)
    test_pic50_m_final = regression_metrics(target_test_pic50, pic50_pred_t)

    # 2026-10-08 — opt-in ridge readout on the frozen encoder embedding.
    # Measured in wf_pearson_r_pic50_head_vs_feature.py: the neural pIC50
    # head extracts less signal than a Ridge probe on the SAME encoder's
    # pooled embedding (head 0.08-0.31 vs ridge ~0.38).  Reported alongside
    # the neural head; never replaces it (default OFF).
    ridge_test: Optional[Dict[str, float]] = None
    if pic50_ridge_readout:
        try:
            from molmetal.training.pic50_ridge_readout import (
                ridge_readout_from_model,
            )

            _train_idx = np.asarray(train_pool_idx, dtype=np.int64)
            _eval_idx = np.asarray(
                [i for i in range(len(graphs_all))], dtype=np.int64
            )
            # embeddings keyed by the featurised union; map Ru test rows to
            # their positions in that union.
            _used_pos = {int(v): k for k, v in enumerate(used_idx)}
            _metal_ids_all = [
                metal_to_idx[str(metals_all[int(v)])] for v in used_idx
            ]
            _test_pos = np.array(
                [_used_pos[int(i)] for i in target_test if int(i) in _used_pos],
                dtype=np.int64,
            )
            _train_pos = np.array(
                [_used_pos[int(i)] for i in train_pool_idx if int(i) in _used_pos],
                dtype=np.int64,
            )
            _y_train = pic50_all[
                np.array(
                    [i for i in train_pool_idx if int(i) in _used_pos],
                    dtype=np.int64,
                )
            ]
            _y_test = pic50_all[
                np.array(
                    [i for i in target_test if int(i) in _used_pos],
                    dtype=np.int64,
                )
            ]
            ridge_test = ridge_readout_from_model(
                model=model,
                graphs=graphs_all,
                metal_ids=_metal_ids_all,
                train_indices=_train_pos,
                eval_indices=_test_pos,
                y_train=_y_train,
                y_eval=_y_test,
                batch_size=int(batch_size),
                device=device,
            )
            print(
                f"  [{metal} seed={seed}] RIDGE-READOUT test_pearson="
                f"{ridge_test.get('pearson', float('nan')):.4f} "
                f"(neural head {test_pic50_m_final['pearson']:.4f}, "
                f"alpha={ridge_test.get('ridge_alpha')})",
                flush=True,
            )
        except Exception as _rr_err:  # never let the diagnostic crash training
            print(
                f"  [{metal} seed={seed}] ridge readout failed: "
                f"{type(_rr_err).__name__}: {_rr_err}",
                file=sys.stderr,
                flush=True,
            )

    ckpt_path = ckpt_dir / f"dmpnn_mtw_{metal.lower()}_seed{seed}.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Wf-A5 N3 — post-loop save now also writes the resumable schema
    # keys (``epoch`` + ``best_val_auc`` + ``optimizer_state_dict``) so the
    # resulting checkpoint is loadable by ``_load_checkpoint_for_resume``
    # on the next invocation.  The legacy keys (``epochs`` + ``best_epoch``
    # + ``val_auc`` + ``test_auc`` + ``n_*``) are preserved for backward
    # compat with downstream inference consumers (Wf-131 P1 + Wf-251-F4).
    torch.save(
        {
            # Wf-A5 N3 NEW resumable keys (used by --resume-from auto)
            "epoch": int(epochs),
            "best_val_auc": float(val_m.get("roc_auc", float("nan"))),
            "wasserstein_history": list(w_history),
            "model_state_dict": best_state,
            "optimizer_state_dict": {k: v for k, v in optimizer.state_dict().items()},
            # Legacy keys (preserved bit-for-bit for downstream consumers)
            "metal": metal,
            "seed": seed,
            "epochs": epochs,
            "alpha": alpha,
            "w_w": w_w,
            "use_l2_guard": use_l2_guard,
            "lr": lr,
            "batch_size": batch_size,
            "hidden": hidden if hidden is not None else DEFAULT_HIDDEN,
            "depth": depth if depth is not None else DEFAULT_DEPTH,
            "val_auc": float(val_m.get("roc_auc", float("nan"))),
            "test_auc": float(test_m.get("roc_auc", float("nan"))),
            "best_epoch": best_epoch,
            "n_train": int(len(target_train)),
            "n_val": int(len(target_val)),
            "n_test": int(len(target_test)),
            "n_joint_train": int(len(train_pool_idx)),
        },
        ckpt_path,
    )

    print(
        f"  [{metal} seed={seed}] DONE best_epoch={best_epoch} "
        f"val_auc={val_m.get('roc_auc', float('nan')):.4f} "
        f"test_auc={test_m.get('roc_auc', float('nan')):.4f} "
        f"test_pearson={test_pic50_m_final['pearson']:.3f} "
        f"({train_seconds:.1f}s, featurise={featurise_seconds:.1f}s)",
        flush=True,
    )

    # Wf-TUI NEW: structured 'done' event for this (metal, seed) pair.
    try:
        emit_event(
            "done",
            metal=metal,
            seed=seed,
            best_epoch=int(best_epoch),
            val_auc=float(val_m.get("roc_auc", float("nan"))),
            test_auc=float(test_m.get("roc_auc", float("nan"))),
            test_pearson=float(test_pic50_m_final.get("pearson", float("nan"))),
            train_seconds=float(train_seconds),
            featurise_seconds=float(featurise_seconds),
            ckpt_path=str(ckpt_path),
        )
    except Exception:
        pass

    return MetalSeedResult(
        metal=metal,
        seed=seed,
        ckpt_path=str(ckpt_path),
        n_train=int(len(target_train)),
        n_val=int(len(target_val)),
        n_test=int(len(target_test)),
        val_metrics=val_m,
        test_metrics=test_m,
        val_pic50_metrics=val_pic50_m_final,
        test_pic50_metrics=test_pic50_m_final,
        wasserstein_history=w_history,
        elapsed_seconds=float(train_seconds + featurise_seconds),
        pic50_ridge_metrics=ridge_test,
    )


# ---------------------------------------------------------------------------
# Wf-A5 N3 — checkpoint resumption helpers (SCAFFOLD)
# ---------------------------------------------------------------------------
def _resumable_ckpt_path(ckpt_dir: Path, metal: str, seed: int) -> Path:
    """Return the canonical resumable checkpoint path.

    Matches the canonical final-checkpoint filename used by the
    post-loop ``torch.save`` block below so that the same path can be
    reused for both the final checkpoint and the periodic mid-training
    checkpoints.
    """
    return ckpt_dir / f"dmpnn_mtw_{metal.lower()}_seed{seed}.pt"


def _save_resumable_checkpoint(
    *,
    ckpt_dir: Path,
    metal: str,
    seed: int,
    model: "MultiTaskWassersteinModel",
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_auc: float,
    wasserstein_history: List[float],
    alpha: float,
    w_w: float,
    use_l2_guard: bool,
    lr: float,
    batch_size: int,
    hidden: int,
    depth: int,
) -> Path:
    """Persist a resumable checkpoint (Wf-A5 N3 design §6.3).

    The checkpoint dict contains everything needed to resume training
    on a subsequent invocation:
        - ``model_state_dict``: full backbone + heads
        - ``optimizer_state_dict``: Adam state (for lr continuity)
        - ``epoch``: 1-indexed last-completed epoch
        - ``best_val_auc``: best-so-far val AUC across this run
        - ``wasserstein_history``: per-epoch wasserstein means
        - ``metal`` / ``seed`` / ``alpha`` / ``w_w`` / ``use_l2_guard``:
          config fingerprint
    Backward compat: legacy checkpoints without ``optimizer_state_dict``
    are loadable via the ``try/except`` wrapper in
    ``_load_checkpoint_for_resume``.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _resumable_ckpt_path(ckpt_dir, metal, seed)
    torch.save(
        {
            "metal": metal,
            "seed": seed,
            "epoch": int(epoch),
            "best_val_auc": float(best_val_auc),
            "wasserstein_history": list(wasserstein_history),
            "alpha": float(alpha),
            "w_w": float(w_w),
            "use_l2_guard": bool(use_l2_guard),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "hidden": int(hidden),
            "depth": int(depth),
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": {k: v for k, v in optimizer.state_dict().items()},
        },
        ckpt_path,
    )
    print(
        f"  [{metal} seed={seed}] [Wf-A5-N3] saved resumable checkpoint "
        f"at epoch={epoch} → {ckpt_path}",
        flush=True,
    )
    return ckpt_path


def _load_checkpoint_for_resume(
    *,
    model: "MultiTaskWassersteinModel",
    optimizer: torch.optim.Optimizer,
    metal: str,
    seed: int,
    ckpt_dir: Path,
    resume_from: str,
    resume_from_path: Optional[Path],
    device: torch.device,
) -> Tuple[int, List[float]]:
    """Load a prior checkpoint to resume training (Wf-A5 N3 design §6.3).

    Returns ``(start_epoch, wasserstein_history)``:
      - ``start_epoch`` is 0 for cold-start and ``ckpt.epoch`` (= 1-indexed
        last-completed epoch) for resumed training. The new
        ``range(start_epoch, epochs)`` in ``train_one_metal_one_seed``
        therefore skips the already-completed epochs.
      - ``wasserstein_history`` is the loaded history (empty list for
        cold-start).

    Bit-for-bit backward compat:
      - ``resume_from == 'none'`` always cold-starts (no-op)
      - ``resume_from == 'auto'`` with no checkpoint present cold-starts
        (no-op) — same as the Wf-131 P1 + Wf-251-F4 baseline behaviour
      - ``resume_from == 'path'`` with missing explicit path cold-starts
        + WARN
      - legacy checkpoints without ``optimizer_state_dict`` load the
        model state but skip optimizer state (warn-once)
    """
    start_epoch = 0
    w_history: List[float] = []

    if resume_from == "none":
        print(
            f"  [{metal} seed={seed}] [Wf-A5-N3] resume_from=none; "
            f"cold-start (no checkpoint load).",
            flush=True,
        )
        return start_epoch, w_history

    if resume_from == "path":
        ckpt_path = resume_from_path
        if ckpt_path is None or not ckpt_path.exists():
            print(
                f"  [{metal} seed={seed}] [Wf-A5-N3][WARN] "
                f"--resume-from=path but path does not exist: "
                f"{ckpt_path}; cold-starting.",
                flush=True,
            )
            return start_epoch, w_history
    else:
        # 'auto' — scan ckpt_dir for canonical path
        ckpt_path = _resumable_ckpt_path(ckpt_dir, metal, seed)
        if not ckpt_path.exists():
            print(
                f"  [{metal} seed={seed}] [Wf-A5-N3] resume_from=auto but "
                f"no checkpoint found at {ckpt_path}; cold-starting.",
                flush=True,
            )
            return start_epoch, w_history

    # Load checkpoint (weights: torch.consistent across CPU/GPU via map_location)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(
            f"  [{metal} seed={seed}] [Wf-A5-N3][WARN] failed to load "
            f"checkpoint {ckpt_path}: {exc}; cold-starting.",
            flush=True,
        )
        return start_epoch, w_history

    # Restore model + optimizer + epoch counter
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as load_exc:
        print(
            f"  [{metal} seed={seed}] [Wf-A5-N3][WARN] model_state_dict "
            f"mismatch: {load_exc}; cold-starting.",
            flush=True,
        )
        return start_epoch, w_history

    start_epoch = int(ckpt.get("epoch", 0))
    w_history = list(ckpt.get("wasserstein_history", []))
    best_val_auc_resumed = float(ckpt.get("best_val_auc", -1.0))

    # Backward-compat: legacy checkpoints may not have optimizer state
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            optimizer_resumed = True
        except Exception as opt_exc:
            print(
                f"  [{metal} seed={seed}] [Wf-A5-N3][WARN] optimizer_state_dict "
                f"incompatible: {opt_exc}; model resumed, optimizer cold.",
                flush=True,
            )
            optimizer_resumed = False
    else:
        optimizer_resumed = False

    print(
        f"  [{metal} seed={seed}] [Wf-A5-N3] resumed from "
        f"epoch={start_epoch} → continue from epoch={start_epoch + 1} "
        f"(best_val_auc={best_val_auc_resumed:.4f}, "
        f"optimizer_resumed={optimizer_resumed})",
        flush=True,
    )
    return start_epoch, w_history


# `_safe_classification_metrics` was extracted to
# molmetal/training/classification_metrics.py (Wf-??? monolith-split,
# 2026-10-05). Re-imported here as a backwards-compat shim so existing
# callers — tests, scripts — keep working.
from molmetal.training.classification_metrics import (  # noqa: E402,F401
    _safe_classification_metrics,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-131 Phase 1 — 5-metal multi-task D-MPNN with Wasserstein "
            "pIC50 regularisation (Ru AUC rescue)."
        ),
    )
    p.add_argument(
        "--metals",
        nargs="+",
        default=["auto"],
        help=(
            "Subset of metals to train. Pass either space-separated metals "
            "(e.g. 'Ru Ir Rh') or the literal 'auto' for auto-discovery "
            "from the CSV metal column. Default: 'auto' — the cohort is "
            "read off whatever --csv / --dataset resolves to, so "
            "metalcytotox_5metal yields the 5 metals and platinai_4cell "
            "yields Pt without any extra flag. "
            "Wf-A6 BLOCKER A: --metals no longer hard-coded to FIVE_METALS; "
            "any metal string accepted; runtime CSV validation rejects "
            "metals not present in --csv (with clear 'Available metals: "
            "[...]' error). "
            "Wf-A9: default changed from the literal FIVE_METALS list to "
            "'auto' so the yaml `--dataset` actually drives the cohort "
            "(the literal default silently rejected every other dataset)."
        ),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default: 0 1 2 3 4).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Epochs per seed (default 50).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="BCE weight (default 0.5).",
    )
    # 2026-10-08 — ridge readout on frozen encoder embeddings.  Measured:
    # the neural pIC50 head extracts less signal than a Ridge probe on the
    # SAME encoder's pooled embedding (WF diagnostic
    # molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py).  Default OFF
    # keeps the Wf-131 P1 neural-head path bit-for-bit.
    p.add_argument(
        "--pic50-ridge-readout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report a Ridge readout on frozen encoder embeddings alongside "
        "the neural pIC50 head (default OFF).",
    )
    # Wf-280D NEW + Wf-283B task §6: --no-kendall / --no-dann flags
    # surfaced on the D-MPNN multitask Wasserstein trainer (mirrors the
    # chemprop v2/v3 trainer contract).  Both default ON to preserve
    # Wf-131 P1 / Wf-197 default behaviour; pass --no-kendall or
    # --no-dann to disable the respective component.
    p.add_argument(
        "--use-kendall",
        "--no-kendall",
        dest="kendall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply Kendall tau UW MTL loss (default ON). "
            "Pass --no-kendall to disable. Backward compat alias for "
            "the Wf-280D pre-flight contract."
        ),
    )
    p.add_argument(
        "--use-dann",
        "--no-dann",
        dest="dann",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply DANN gradient-reversal layer for domain-adversarial "
            "training (default ON). Pass --no-dann to disable. "
            "Backward compat alias for the Wf-280D pre-flight contract."
        ),
    )
    p.add_argument(
        "--wasserstein-weight",
        type=float,
        default=DEFAULT_WASSERSTEIN_WEIGHT,
        help="Wasserstein pIC50 loss weight w_w (default 0.05).",
    )
    # Wf-357-D2 / Wf-358-B1 — W₁ vs W₂ dispatch flag (default OFF; backward
    # compat with Wf-131 P1 + Wf-181 G5 / Wf-182 G6 path preserved).
    p.add_argument(
        "--wasserstein-w1",
        "--use-wasserstein-w1",
        dest="use_wasserstein_w1",
        action="store_true",
        default=False,
        help=(
            "Use Wasserstein W₁ (mean abs sort-diff) penalty instead of "
            "the W₂ (sliced) default. Off by default. 2026-10-08 §5.7 — "
            "wired via wasserstein_pic50_gradient(wasserstein_order=1)."
        ),
    )
    p.add_argument(
        "--wasserstein-2",
        "--use-wasserstein-2",
        dest="use_wasserstein_2",
        action="store_true",
        default=False,
        help=(
            "Use Wasserstein W₂ (RMS sort-diff) closed form via "
            "wasserstein_2_1d_torchsort. Off by default. 2026-10-08 §5.7 — "
            "wired via wasserstein_pic50_gradient(wasserstein_order=2)."
        ),
    )
    # Wf-197 ablation: --single-task restricts joint training pool to the
    # target metal only (no 5-metal cohort sharing). Default OFF (multi-task).
    p.add_argument(
        "--single-task",
        dest="single_task",
        action="store_true",
        default=False,
        help="Train target metal only (no 5-metal joint pool). Wf-197 ablation lever 1.",
    )
    # Wf-251-F3 — ablation config dispatcher.  Default ``mtl_baseline``
    # preserves Wf-131 shipped behaviour bit-for-bit (multi-task joint
    # pool, h=128, sliced-W₂ gradient).  Other configs enable additional
    # levers for the 4×5×3 paper §6 ablation table.
    p.add_argument(
        "--ablation-config",
        choices=("mtl_baseline", "mtl_structured", "mtl_gradproj", "single_task"),
        default="mtl_baseline",
        help="Ablation config dispatcher (Wf-251-F3): "
        "mtl_baseline (default — current shipped behaviour), "
        "mtl_structured (per-metal weight-sharing mask on FFN head), "
        "mtl_gradproj (PCGrad-style per-task gradient projection), "
        "single_task (Wf-197 lever 1; per-metal cohort only).",
    )
    # 2026-10-04 — `model.task` was declared in 4 configs and resolved to no
    # argparse dest (pre-flight p3_wf343c4_hyperparams_config_complete
    # RESIDUAL), so the declared value reached nothing. The schema leaf
    # `model.task` maps to this dest via the wfA2 loader's two-component
    # section fallback. The multi-task head layout is hardcoded (one head
    # pair per cohort member); only the declared value 'multi_task' is
    # accepted — main() raises on anything else rather than silently
    # pretending to support a task layout that does not exist.
    p.add_argument(
        "--task",
        choices=("multi_task",),
        default="multi_task",
        help=(
            "Task layout (yamls declare it as model.task). Only "
            "'multi_task' is supported: the model is a shared encoder + "
            "one head pair per cohort member, which IS the multi-task "
            "layout. Any other value is rejected loudly in main(); it is "
            "never silently ignored."
        ),
    )
    p.add_argument(
        "--n-mc",
        type=int,
        default=DEFAULT_N_MC,
        help="Number of sliced-W2 projections (default 256).",
    )
    # Wf-251-F4 — W₂ gradient mode dispatcher.
    # Wf-328-W5 — default flipped to ``DEFAULT_W2_GRADIENT_MODE`` ("sort").
    p.add_argument(
        "--w2-gradient-mode",
        choices=("sliced", "sort"),
        default=DEFAULT_W2_GRADIENT_MODE,
        help="W₂ gradient source (Wf-251-F4 / Wf-328-W5): sort (default — "
        "closed-form 1-D torch.sort; more numerically stable; modest AUC "
        "lift) or sliced (Wf-131 P1 shipped behaviour; reproducibility).",
    )
    # Wf-335-P3 — Triton kernel dispatch flags (default OFF; env-gated
    # at runtime so CPU-only test hosts fall through to PyTorch ref).
    for _flag, _dest in (
        ("--use-triton-relu-mlp", "use_triton_relu_mlp"),
        ("--use-triton-wasserstein-sliced", "use_triton_wasserstein_sliced"),
        ("--use-triton-bond-head", "use_triton_bond_head"),
        ("--use-triton-smiles-featurize", "use_triton_smiles_featurize"),
    ):
        p.add_argument(
            _flag,
            dest=_dest,
            action="store_true",
            default=False,
            help=(
                f"Triton dispatch gate for {_dest} (default OFF; env-gated). "
                f"See molmetal/adapters/flow_matching_lipman/_reference/."
            ),
        )
    p.add_argument(
        "--use-l2-guard",
        dest="use_l2_guard",
        action="store_true",
        default=True,
        help="Apply grad-norm clip (default ON).",
    )
    p.add_argument(
        "--no-l2-guard",
        dest="use_l2_guard",
        action="store_false",
        help="Disable grad-norm clip (overrides --use-l2-guard).",
    )
    p.add_argument(
        "--lr", type=float, default=DEFAULT_LR, help="Adam lr (default 1e-3)."
    )
    # Wf-A6 A2 BLOCKER C fix — LR scheduler (cosine/onecycle) + warmup + weight decay
    # per user voice 2026-10-03 verbatim '学习率的控制至少得用余弦退火或者更加高级的方案'.
    # Default OFF preserves Wf-131 P1 + Wf-251-F4 baseline bit-for-bit.
    # See build_lr_scheduler() helper for the SequentialLR / CosineAnnealingLR /
    # OneCycleLR composition logic. Activation order:
    # 1) --use-lr-warmup SCAFFOLD-only (legacy Wf-430 compatibility)
    # 2) --lr-scheduler cosine / onecycle opt-in (this patch)
    p.add_argument(
        "--lr-scheduler",
        type=str,
        choices=("none", "cosine", "onecycle"),
        default="none",
        help=(
            "Wf-A6 A2 BLOCKER C fix: opt-in LR scheduler strategy. "
            "'none' (default; preserves Wf-131 P1 baseline bit-for-bit) "
            "uses a fixed LR throughout training. 'cosine' applies "
            "linear warmup for --lr-warmup-epochs followed by "
            "CosineAnnealingLR to --lr-min-lr (Loshchilov-Hutter 2017 "
            "SGDR; conservative 2024-2025 transformer/GNN default). "
            "'onecycle' applies Smith-Topin 2018 super-convergence "
            "policy with max_lr = base * 10, pct_start = 0.3, "
            "div_factor = 25, final_div_factor = 1e4, anneal_strategy = 'cos'. "
            "PROJECTED 1.5-3x speedup vs ReduceLROnPlateau baseline "
            "for 'cosine'; PROJECTED 5-10x speedup for 'onecycle'. "
            "MEASURED-end-to-end lift requires Blackwell sm_120 GPU "
            "retrain per AGENTS.md §5.1."
        ),
    )
    p.add_argument(
        "--lr-warmup-epochs",
        type=int,
        default=5,
        help=(
            "Wf-A6 A2 BLOCKER C fix: linear warmup duration in epochs "
            "applied before cosine/onecycle scheduler. Default 5 "
            "(EDM 2022 §5 recommendation). Only effective when "
            "--lr-scheduler != 'none'."
        ),
    )
    p.add_argument(
        "--lr-min-lr",
        type=float,
        default=1e-6,
        help=(
            "Wf-A6 A2 BLOCKER C fix: floor LR for cosine annealing "
            "or final_div_factor ratio denominator for onecycle. "
            "Default 1e-6. Only effective when --lr-scheduler != 'none'."
        ),
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help=(
            "Wf-A6 A2 BLOCKER C fix (audit gap R5): torch.optim.Adam "
            "weight_decay parameter. Default 0.0 (preserves Wf-131 P1 "
            "baseline bit-for-bit). Setting 1e-5..1e-3 typically "
            "improves generalization on Transformer/GNN models "
            "(Loshchilov-Hutter 2019 decoupled weight decay). "
            "Bit-exact OFF when 0.0."
        ),
    )
    p.add_argument(
        "--max-norm",
        type=float,
        default=5.0,
        help=(
            "Wf-A8 (2026-10-03) gradient clipping max norm. Default 5.0 "
            "(Pascanu 2013; standard for transformer/GNN stability). "
            "Applied via torch.nn.utils.clip_grad_norm_(parameters, "
            "max_norm) AFTER loss.backward() + BEFORE optimizer.step() "
            "in the training loop. When 0.0, clipping is DISABLED "
            "(preserves Wf-131 P1 + Wf-251-F4 baseline bit-for-bit). "
            "Setting 1.0..10.0 recommended for RNN-style architectures "
            "where exploding gradients are a concern."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size (default MT_BATCH_SIZE=64).",
    )
    # 2026-10-04 — `protocol.gradient_accumulation_steps` was declared in 2
    # configs and resolved to no argparse dest (pre-flight RESIDUAL), so the
    # declared value reached nothing. The schema leaf maps to this dest via
    # the wfA2 loader's two-component section fallback. Implemented as real
    # gradient accumulation in train_one_epoch_joint(): the loss is divided
    # by N and the optimizer steps once every N micro-batches. Default 1 =
    # one step per batch, bit-for-bit identical to the pre-flag loop.
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Gradient accumulation window (micro-batches per optimizer "
            "step). The composite loss of each micro-batch is divided by "
            "this N and the optimizer/clip runs once every N micro-batches; "
            "the trailing partial window is stepped at epoch end. "
            "Default 1 = current per-batch behaviour, bit-for-bit."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for per-seed metrics JSON.",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory. Default (when not given here or via the "
            "yaml output.ckpt_dir field): <out_dir>/checkpoints. "
            "Wf-A10: this used to default to the shared "
            "molmetal/checkpoints/ regardless of --out-dir, so any throwaway "
            "smoke wrote into the production checkpoint directory AND the "
            "default --resume-from auto would then silently resume from those "
            "stale checkpoints on a later run (observed: train_seconds=0.0035, "
            "i.e. it trained nothing). Deriving from --out-dir keeps runs "
            "isolated. Configs that want the shared directory can still set "
            "output.ckpt_dir explicitly in the yaml."
        ),
    )
    # ---- N3 checkpoint resumption (Wf-A5 N3 design field) ----
    # --resume-from controls how to load a prior checkpoint at the start of
    # `train_one_metal_one_seed`:
    #   'auto'  : scan ckpt_dir for the canonical
    #             `dmpnn_mtw_{metal}_seed{seed}.pt` filename; if found,
    #             load model_state_dict + optimizer_state_dict + epoch +
    #             resume from epoch+1; if not found, cold start.
    #   'none'  : cold start (ignore any pre-existing checkpoint).
    #   'path'  : load the explicit path passed via --resume-from-path
    #             (kept as 'auto'-equivalent for back-compat with the
    #             choices=['auto','none','path'] contract).
    # Default 'auto' = sane default; existing yaml files implicitly get
    # resume-from=auto which is bit-for-bit backward compat with Wf-151 P1
    # baseline (no checkpoint present ⇒ cold start, same as 'none').
    p.add_argument(
        "--resume-from",
        choices=["auto", "none", "path"],
        default="auto",
        help=(
            "Checkpoint resumption mode (Wf-A5 N3): 'auto' = scan ckpt-dir "
            "for latest dmpnn_mtw_{metal}_seed{seed}.pt; 'none' = cold start; "
            "'path' = use --resume-from-path (default 'auto')."
        ),
    )
    p.add_argument(
        "--resume-from-path",
        type=Path,
        default=None,
        help=(
            "Explicit checkpoint .pt path for --resume-from=path mode "
            "(Wf-A5 N3). Default None."
        ),
    )
    # --checkpoint-interval controls how often (in epoch) to persist a
    # resumable checkpoint. Per design §6.3 the canonical value is 50.
    # Save also fires at the final epoch (epoch == epochs-1) regardless
    # of interval, so the canonical final checkpoint is always present.
    # Default 50 = sane default; with epochs=50 this means 1 mid
    # checkpoint at epoch 50 (= final) and 0 mid-way checkpoints.
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help=(
            "Save checkpoint every N epochs (Wf-A5 N3); also saves at "
            "the final epoch regardless. Default 50."
        ),
    )
    p.add_argument(
        "--task-axis",
        choices=("metal", "cell_line"),
        default="metal",
        help=(
            "What the model's per-task heads are indexed by (Wf-A10). "
            "'metal' (default) = the historical 5-metal pool: one head per "
            "metal. 'cell_line' = one head per cell line over the SAME shared "
            "encoder, for cohorts that vary the cell line instead of the metal "
            "(e.g. --dataset platinai_4cell: MCF7 / A549 / A2780 / A2780cis). "
            "This is the multi-task shape the field uses for multi-cell-line "
            "cytotoxicity prediction, and it is what makes per-cell AUC — the "
            "metric Rusanov 2025 reports — directly available from cls_heads. "
            "Changing this does NOT change the model: n_metals is the task "
            "count, so 4 cell lines simply become 4 heads."
        ),
    )
    p.add_argument(
        "--active-labeling",
        choices=("per_cell_median", "global"),
        default="per_cell_median",
        help=(
            "How the `active` label is derived on the --task-axis cell_line "
            "path (Wf-A10). 'per_cell_median' (default) splits each cell line "
            "at its own median pIC50, giving every per-cell head a balanced "
            "50/50 problem and making per-cell AUC a within-cell potency "
            "ranking that is comparable across cell lines. 'global' applies "
            "one --active-pic50-threshold to every cell line. "
            "Measured on platinai_4cell.csv with 'global' at 1 uM: the per-cell "
            "active fraction is a2780 0.242 / mcf7 0.491 / a549 0.632 / "
            "a2780cis 0.902 — a single cut cannot balance these four, because "
            "their medians differ (5.69 / 5.99 / 6.15 / 6.59)."
        ),
    )
    p.add_argument(
        "--active-pic50-threshold",
        type=float,
        default=6.0,
        help=(
            "pIC50 above which a row is labelled active, used only by "
            "--active-labeling global (Wf-A10). Default 6.0 (= IC50 < 1 uM), "
            "NOT the repo's MetalCytoToxDB standard of 5.0 (IC50 < 10 uM): "
            "measured on the real platinai cohort, 5.0 labels 98.1%% of rows "
            "positive, which makes the classification task degenerate and "
            "per-cell AUC meaningless. Measured active fraction on "
            "platinai_4cell.csv: >5.0 -> 0.981, >5.5 -> 0.863, >6.0 -> 0.567, "
            ">6.5 -> 0.244, >7.0 -> 0.059. A threshold calibrated on one "
            "assay's IC50 distribution does not transfer to a differently "
            "pre-filtered cohort."
        ),
    )
    p.add_argument(
        "--dataset",
        choices=("metalcytotox_5metal", "platinai_4cell", "nci60_9cell"),
        default="metalcytotox_5metal",
        help=(
            "Dataset selector (Wf-A8 PlatinAI 4-cell loader; audit gap #1 fix). "
            "Default 'metalcytotox_5metal' preserves Wf-131 P1 + Wf-251-F4 "
            "baseline bit-for-bit. 'platinai_4cell' loads the Rusanov 2025 "
            "Pt × 4-cell-line cohort from .data/molmetal/platinai_4cell.csv "
            "(pivoted to long format internally) — NOTE its labels are "
            "synthetic, so treat any AUC from it as a pipeline null test, not "
            "a measurement. 'nci60_9cell' loads the REAL-labelled NCI-60 "
            "precious-metal 9-cell cohort (molmetal/data/"
            "nci60_precious_9cell.csv); its scaffold split reproduces "
            "train_nci60_10cell.py byte-for-byte, so results are comparable "
            "to the shipped m21 baseline (0.8007 +/- 0.0160)."
        ),
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Override CSV path. Default resolved from --dataset via "
            "_resolve_default_csv_path(): metalcytotox_5metal -> "
            ".data/molmetal/metalcytotox_5metal_multitask_v4.csv; "
            "platinai_4cell -> .data/molmetal/platinai_4cell.csv."
        ),
    )
    p.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help=(
            "Override the MetalCytoToxDB CSV path (default: loader default "
            "= /mnt/storage/data/molmetal/MetalCytoToxDB.csv; pass "
            "'.data/molmetal/MetalCytoToxDB.csv' if /mnt/storage is not "
            "mounted)."
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device (cuda:0 or cpu). Default auto-detect.",
    )
    p.add_argument(
        "--cuda-visible-devices",
        default="",
        help=(
            "Override CUDA_VISIBLE_DEVICES before device resolution.  "
            "Empty '' (the default) = inherit the shell env — "
            "scripts/start.sh exports it from the yaml `compute.gpu` "
            "field.  If neither is set, training_utils falls back to its "
            "own default ('0' — GPU 0).  Pass a non-empty value to force "
            "a GPU."
        ),
    )
    # --events-jsonl-path is added at line ~2035 below by the parallel
    # canonical integration; no need to redefine here.
    p.add_argument(
        "--loose-filter",
        action="store_true",
        help="Disable CytotoxFilter time_threshold/ic50_min (loads more rows).",
    )
    p.add_argument(
        "--max-joint-rows",
        type=int,
        default=None,
        help=(
            "Optional cap on the joint train pool (smoke-test scaling; "
            "0/None = full pool). Applied as a seeded random subsample "
            "after the <50-row floor check; when it binds, n_joint_train "
            "in the checkpoint reflects the capped count while "
            "n_train/n_val/n_test remain target-metal split sizes."
        ),
    )
    # ---- Wf-196 50-epoch CPU sweep additions ----
    # Override backbone hidden size (default DMPNN_HIDDEN=128); smaller
    # hidden reduces CPU epoch time substantially (~h^2 cost in encoder).
    p.add_argument(
        "--hidden",
        type=int,
        default=None,
        help=(
            "Override backbone hidden dim (default: DMPNN_HIDDEN=128). "
            "Smaller values (e.g. 64) give ~4x CPU speedup at modest "
            "AUC cost."
        ),
    )
    p.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "Override backbone message-passing depth (default: DMPNN_DEPTH=3). "
            "Smaller depth (e.g. 2) gives ~1.5x CPU speedup."
        ),
    )
    p.add_argument(
        "--cpu-num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader num_workers (default 0 = main process; >0 helps "
            "when featurising on CPU).  Use 0 unless batching is large."
        ),
    )
    # 2026-10-04 — `data.pin_memory` was declared in 2 configs and resolved
    # to no argparse dest (pre-flight RESIDUAL), so the declared value
    # reached nothing. The schema leaf `data.pin_memory` maps to this dest
    # via the wfA2 loader's two-component section fallback. The trainer
    # batches manually (no torch DataLoader), so pin_memory is wired to the
    # host→device transfer it exists to accelerate: when enabled, the
    # collated batch tensors are copied with non_blocking=True. On the
    # default pageable tensors non_blocking is documented as a no-op
    # (pinning only happens inside a DataLoader), so numerics are unchanged
    # at the default True; on CPU the flag is a no-op by construction.
    p.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        default=False,
        help=(
            "WIRE-IN 2026-10-04: EMA of the model parameters (Karras 2022 EDM "
            "section 5), updated after every optimizer step and swapped in "
            "around evaluation.  Default OFF.  This is the consumer the yaml "
            "`train.ema_decay` key had been missing -- before today this "
            "trainer had no EMA code at all."
        ),
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay (default 0.999); the value the yaml `train.ema_decay` declares.",
    )
    p.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pin-memory semantics for the manual batch → device transfer "
            "(yamls declare it as data.pin_memory). Default True; "
            "--no-pin-memory forces blocking copies. This trainer has no "
            "torch DataLoader (it builds batches itself), so the flag gates "
            "the non_blocking hint on the H2D copies rather than a loader "
            "argument."
        ),
    )
    # ------------------------------------------------------------------
    # Wf-405-A2 — Wf-380 NEW CLI flag wire-in (closing the SCAFFOLD-only
    # gap surfaced by Wf-405-A1 audit).  3 NEW DMPNN flags with
    # default-OFF semantics so backward compat with Wf-131 P1 +
    # Wf-251-F4 baseline is preserved bit-for-bit when no flag is
    # set on the CLI.
    # ------------------------------------------------------------------
    p.add_argument(
        "--per-cohort-features",
        action="store_true",
        default=False,
        help=(
            "Wf-405-A2 NEW (Wf-380-B1 closeout #14 / Wf-380-A2): "
            "invoke wf380a2_m2_r2_per_cohort.select_features(n_total) "
            "to log the per-cohort feature config at training start. "
            "Default OFF (False) per Wf-405-A2 backward-compat "
            "requirement; Wf-131 P1 + Wf-251-F4 baseline is preserved "
            "bit-for-bit when --per-cohort-features is False. "
            "When ON, the trainer logs the per-cohort tier "
            "(n_lt_300 / n_lt_1000 / n_gte_1000) + features + "
            "alpha to stdout. PROJECTED m2 R² diagnostic; "
            "MEASURED-end-to-end lift requires Blackwell sm_120 "
            "GPU retrain per AGENTS.md §5.1 + Wf-380-A2 §0 honest "
            "framing."
        ),
    )
    p.add_argument(
        "--triton-ot",
        action="store_true",
        default=False,
        help=(
            "Wf-405-A2 NEW (Wf-380-B1 closeout #15 / Wf-353-A1): "
            "set TRITON_MINIBATCH_OT=1 + TRITON_USE_FUSED=1 env vars "
            "BEFORE model init to enable triton_minibatch_ot_coupling. "
            "Default OFF (False) per Wf-405-A2 backward-compat "
            "requirement; Wf-131 P1 + Wf-251-F4 baseline is preserved "
            "bit-for-bit when --triton-ot is False. When ON, "
            "triton_minibatch_ot_coupling is dispatched via env-var "
            "state (Triton kernel if available + CUDA + B<=256; "
            "PyTorch Sinkhorn fallback otherwise). PROJECTED "
            "1.5-3x W₂ speedup on Blackwell sm_120 per Wf-353-A1."
        ),
    )
    p.add_argument(
        "--use-ot-cfm",
        action="store_true",
        default=False,
        help=(
            "Wf-405-A2 NEW (Wf-380-B1 closeout #16 / Wf-353-A1): "
            "use optimal_transport_v2_solver.ot_solver instead of "
            "wasserstein_pic50_loss / wasserstein_pic50_gradient for "
            "the W₂ penalty term on pic50_pred vs ref_dev. Default "
            "OFF (False) per Wf-405-A2 backward-compat requirement; "
            "Wf-131 P1 + Wf-251-F4 baseline (sliced / sort W₂) is "
            "preserved bit-for-bit when --use-ot-cfm is False. When "
            "ON, the trainer computes OT-based W₂ on "
            "cat(pic50_pred, ref_dev) via ot_solver (Cuturi 2013 + "
            "Peyré-Cuturi 2019 §3.1). PROJECTED 1.5-3x W₂ speedup on "
            "Blackwell sm_120 per Wf-353-A1."
        ),
    )
    # ------------------------------------------------------------------
    # Wf-432-A2 — Wf-246-C metal-coordination term wire-in (Fix-B)
    # 3 NEW CLI flags: --metal-coord-term (opt-in wire-in),
    # --metal-coord-weight (loss cascade weight, default 0.5),
    # --cal-sign-vina-convention (default ON; gates on bit-for-bit
    # backward compat with Wf-131 P1 + Wf-251-F4 baseline).  Default
    # OFF for backward compat with Wf-131 P1 / Wf-299 / Wf-328 /
    # Wf-131 P3 5-metal subset shipped runs.
    # ------------------------------------------------------------------
    p.add_argument(
        "--metal-coord-term",
        action="store_true",
        default=False,
        help=(
            "2026-10-04 — DISABLED, see the body of train_one_epoch_joint.  "
            "This flag used to add `MetalCoordinationTerm.kcal_adjustment` to "
            "the regression loss.  That cannot work here, for two independent "
            "reasons, and it now raises instead of silently contributing a "
            "constant:\n"
            "  (1) The term is RULE-BASED — it derives its sub-scores from "
            "`coordination_geometry_score(smiles)` and `hsab_match_score("
            "smiles, ...)`, both RDKit + HSAB-table lookups.  It produces a "
            "Python float, so it can never carry a gradient.  The old code "
            "wrapped it in `torch.tensor(...)` behind a `.item()`, which put "
            "a constant offset into the loss while looking like a live term.\n"
            "  (2) Even a correct call would score 0.0 on this trainer's "
            "data: `MetalCoordinationTerm.score` detects the metal from the "
            "SMILES, and MEASURED, 0 of 800 sampled rows of "
            "metalcytotox_5metal carry any metal atom — the metal is a "
            "separate `metal` COLUMN.\n"
            "The term's correct home is the docking path, where it already "
            "runs on real docked SMILES: "
            "molmetal/molmetal_lam/sbdd_env/vina_adapter.py::"
            "_compute_metal_coord_adjustment applies it as a post-hoc "
            "kcal/mol correction to a Vina score."
        ),
    )
    p.add_argument(
        "--metal-coord-weight",
        type=float,
        default=0.5,
        help=(
            "Wf-432-A2 NEW: weight for the MetalCoordinationTerm "
            "contribution to the composite loss (default 0.5; "
            "calibrated per Wf-246-C plan)."
        ),
    )
    p.add_argument(
        "--cal-sign-vina-convention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wf-432-A2 NEW: gate the kcal sign-fix per Wf-427-A1 "
            "metal_coord_term.py:641 (1-2*total)*kcal_per_unit; "
            "lower kcal/mol = better binding per Vina convention. "
            "Default ON to match Wf-333-V1 + Wf-246-C sign-fix; "
            "--no-cal-sign-vina-convention flips the sign for "
            "backward-compat testing."
        ),
    )
    # Wf-A2 NEW: --config YAML override (training config unification).
    # CLI flags still win when explicitly passed (user-intent > YAML default).
    # Bit-for-bit backward compat: omit --config = argparse-only path.
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Path to YAML config overriding argparse defaults. "
            "CLI flags win where user explicitly passes them. "
            "See configs/training/wf124_p3_50epoch_full_sweep_g5.yaml for schema."
        ),
    )
    # Wf-TUI NEW: structured event emission for deep TUI integration.
    # When ON, emit_event() writes JSON lines to stdout + appends to
    # --events-jsonl-path. When OFF, emit_event() writes a backslash-
    # escaped prefix line that preserves the Wf-131 P1 text log format
    # for grep/tail consumers. Default ON per AGENTS.md §5.7 NEW trainer
    # flag wire-in smoke-required rule; --no-emit-json-events restores
    # the legacy text-only path for backward compat.
    p.add_argument(
        "--emit-json-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wf-TUI NEW: emit structured JSON events to stdout + "
            "--events-jsonl-path. Default ON; --no-emit-json-events "
            "falls back to the legacy '[EVENT <type>] key=value' text "
            "path (backward compat with Wf-131 P1 grep/tail consumers)."
        ),
    )
    p.add_argument(
        "--events-jsonl-path",
        type=Path,
        default=None,
        help=(
            "Wf-TUI NEW: path to events.jsonl log (default '<out_dir>/"
            "events.jsonl'). Each emit_event() call appends one JSON "
            "line. Consumers (TUI dashboard, downstream monitoring) "
            "tail this file for structured per-epoch / per-batch events."
        ),
    )
    p.add_argument(
        "--no-events-jsonl",
        dest="no_events_jsonl",
        action="store_true",
        default=False,
        help=(
            "2026-10-04: disable the events.jsonl sink entirely. Distinct "
            "from --no-emit-json-events, which only switches STDOUT between "
            "JSON and the legacy '[EVENT type] k=v' text format — the jsonl "
            "write happened either way, so there was no way to turn the file "
            "off. Wired to the yamls' output.events_jsonl:false, which had "
            "been declared in 9 configs and read by nothing."
        ),
    )
    # ------------------------------------------------------------------
    # 2026-10-04: GPU-utilisation flags, mirroring Wf-431-B1 in
    # train_cfm_yuelbond_wasserstein.py (lines ~3020-3060) — the CFM trainer
    # has had these since Wf-431-B1 and this one never did, so every config
    # that declared compute.torch_compile / tf32 / cudnn_benchmark /
    # channels_last was writing them into a void.  Defaults OFF; the no-flag
    # path is bit-for-bit unchanged.
    # ------------------------------------------------------------------
    p.add_argument(
        "--torch-compile",
        dest="torch_compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "torch.compile the encoder forward + W₂ penalty (mode selected by "
            "--torch-compile-mode). Default OFF (bit-for-bit unchanged "
            "no-flag path); pass --torch-compile to enable, --no-torch-compile "
            "to force it off."
        ),
    )
    # Wf-328-W1 (as revised by Wf-335-R1) — Inductor mode selector.
    p.add_argument(
        "--torch-compile-mode",
        dest="torch_compile_mode",
        choices=TORCH_COMPILE_MODES,
        default=DEFAULT_TORCH_COMPILE_MODE,
        help=(
            "Inductor mode for --torch-compile. Default "
            "'max-autotune-no-cudagraphs' (Wf-335-R1 MEASURED FIX per "
            "Wf-330-B4: 'reduce-overhead' captured CUDA Graphs that fired "
            "RuntimeError on Ru seed=2 + Ir seed=0). Explicit "
            "'reduce-overhead' remains accepted for Wf-326C / Wf-328-W1 "
            "bench reproducibility."
        ),
    )
    # Wf-335-R1 — v2 alias for the same mode lever (override semantics).
    p.add_argument(
        "--torch-compile-mode-v2",
        dest="torch_compile_mode_v2",
        choices=TORCH_COMPILE_MODES,
        default=DEFAULT_TORCH_COMPILE_MODE,
        help=(
            "Wf-335-R1 alias of --torch-compile-mode with the post-fix "
            "default. When explicitly given it overrides the v1 flag."
        ),
    )
    # Wf-328-W4 — TF32 toggle (default ON; --no-tf32 is the §6 lever).
    p.add_argument(
        "--tf32",
        dest="tf32",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TF32,
        help=(
            "torch.backends.cuda.matmul.allow_tf32 + cudnn.allow_tf32 + "
            "set_float32_matmul_precision('high'). Default ON (Wf-328-W4); "
            "pass --no-tf32 for the paper §6 reproducibility path."
        ),
    )
    # Wf-328-W3 — BF16 autocast toggle (default ON; --no-bf16 opt-out).
    p.add_argument(
        "--bf16",
        dest="bf16",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BF16,
        help=(
            "BF16 autocast around the forward + W₂ block on CUDA "
            "(torch.autocast(dtype=bfloat16)); no-op nullcontext on CPU / "
            "pre-Ampere. Default ON (Wf-328-W3); pass --no-bf16 to disable."
        ),
    )
    # Wf-334u1 — pluggable D-MPNN backend selector.
    p.add_argument(
        "--dmpnn-backend",
        dest="dmpnn_backend",
        choices=("pyg", "loop"),
        default=DEFAULT_DMPNN_BACKEND,
        help=(
            "D-MPNN encoder backend (Wf-334u1): 'pyg' (default — vectorised "
            "flat-graph encoder) or 'loop' (legacy Python per-molecule "
            "loop; bit-equivalent, slower)."
        ),
    )
    p.add_argument(
        "--cudnn-benchmark",
        dest="cudnn_benchmark",
        action="store_true",
        default=False,
        help=(
            "torch.backends.cudnn.benchmark=True — benchmark algos on the "
            "first iteration. Default OFF."
        ),
    )
    p.add_argument(
        "--channels-last",
        dest="channels_last",
        action="store_true",
        default=False,
        help=(
            "model.to(memory_format=torch.channels_last). Default OFF. NOTE: "
            "channels_last targets conv-style NCHW tensors; a D-MPNN over "
            "scatter/gather message passing has no spatial axes, so this is "
            "expected to be a no-op here and is kept only for parity with the "
            "CFM trainer. Declared SCAFFOLD in the same sense as Wf-431-B1."
        ),
    )
    args = p.parse_args(argv)
    # Apply YAML overrides (no-op when --config not supplied).
    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

    return apply_yaml_overrides(p, args, args.config)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # 2026-10-04 — `model.task` (dest `task`) validation.  The multi-task
    # head layout is hardcoded (shared encoder + one head pair per cohort
    # member), so any other value must FAIL LOUDLY instead of being
    # silently ignored.  Both entry paths reach this check: the argparse
    # `choices` guard covers the CLI, while the wfA2 YAML loader setattrs
    # the namespace directly and bypasses argparse validation — a yaml
    # `model.task: <bogus>` would otherwise slip through.  Runs before any
    # directory / checkpoint side effect.
    if str(args.task) != "multi_task":
        raise ValueError(
            f"--task={args.task!r} (yaml model.task) is not supported: this "
            "trainer implements the multi-task layout only (shared encoder "
            "+ one head pair per cohort member).  Pass --task multi_task / "
            "model.task: multi_task; any other value is rejected here rather "
            "than silently ignored."
        )
    print(
        f"[dmpnn-flags] task={args.task} pin_memory={args.pin_memory} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}",
        flush=True,
    )
    # 2026-10-05 — ``--cpu-num-workers`` CLI flag acknowledgement
    # (AGENTS.md §5.7 FLAG_ONLY).  The argparse block declared the flag
    # but never read it; this trainer builds batches manually (no torch
    # DataLoader), so the flag is structurally a no-op.  Acknowledge the
    # user-supplied value here so the CLI value reaches stdout and is
    # traceable in the run log, matching the existing [dmpnn-flags]
    # banner.  When ``cpu_num_workers > 0`` we ALSO emit a one-time
    # informational note that the value is currently not consumed (the
    # Wf-405-A2 recipe has no DataLoader to thread workers into).
    _cpu_nw = int(getattr(args, "cpu_num_workers", 0))
    print(
        f"[dmpnn-flags] cpu_num_workers={_cpu_nw}",
        flush=True,
    )
    if _cpu_nw > 0:
        print(
            f"[dmpnn-flags] NOTE: --cpu-num-workers {_cpu_nw} is "
            "currently a structural no-op for this trainer (manual batch "
            "build, no torch DataLoader). The value is acknowledged above "
            "and a future Wf may thread it into the featuriser worker "
            "pool; until then, --cpu-num-workers 0 has identical "
            "training dynamics.",
            flush=True,
        )
    # 2026-10-05 — ``--cal-sign-vina-convention`` CLI flag
    # acknowledgement (AGENTS.md §5.7 FLAG_ONLY no-op-when-read).  The
    # flag IS read in main() and threaded through to
    # ``train_one_epoch_joint()`` (see line 1933 / 1508), but the only
    # consumer — the ``metal_coord_term=True`` branch — raises
    # ``SystemExit`` (line ~1331).  When ``metal_coord_term=False``
    # (the only path that runs to completion), the flag's value has
    # zero training-dynamic effect.  We acknowledge the value here so
    # the CLI flag is non-dead in the run log, and to give downstream
    # readers a hook to correlate the user's --(no-)cal-sign-vina-
    # convention choice with the run's outputs.
    _cal_sign = bool(getattr(args, "cal_sign_vina_convention", True))
    print(
        f"[dmpnn-flags] cal_sign_vina_convention={_cal_sign} "
        "(effective: metal_coord_term path; SystemExit'd branches "
        "make this flag a no-op when --metal-coord-term is False)",
        flush=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Wf-A10 — default the checkpoint dir to <out_dir>/checkpoints so a run
    # can no longer read or write the shared molmetal/checkpoints/ tree
    # unless it asks to (yaml output.ckpt_dir still wins when set).
    if args.ckpt_dir is None:
        args.ckpt_dir = Path(args.out_dir) / "checkpoints"
        print(
            f"[Wf-A10] --ckpt-dir not set; using {args.ckpt_dir} "
            f"(isolated per --out-dir; set --ckpt-dir or yaml output.ckpt_dir "
            f"to share a directory)",
            flush=True,
        )
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Wf-TUI NEW: install structured-event emitter BEFORE any training
    # step (so epoch-start + checkpoint-resume + lr-step events all
    # route through emit_event()). Default events.jsonl path =
    # '<out_dir>/events.jsonl' when --events-jsonl-path not supplied.
    # Bit-for-bit backward compat with Wf-131 P1 baseline: when
    # --no-emit-json-events is set at the CLI flag, emit_event() emits
    # the legacy '[EVENT <type>] k=v k=v ...' text format instead of
    # JSON, and events.jsonl writes are still appended (TUI consumers
    # can read the JSONL regardless of stdout format).
    _events_jsonl_path = (
        args.events_jsonl_path
        if args.events_jsonl_path is not None
        else (args.out_dir / "events.jsonl")
    )
    if getattr(args, "no_events_jsonl", False):
        # None is the documented "no sink" sentinel handled by
        # _install_event_emitter; emit_event() gates its write on it.
        _events_jsonl_path = None
    _install_event_emitter(
        emit_enabled=bool(args.emit_json_events),
        events_jsonl_path=_events_jsonl_path,
    )

    # Wf-A6 BLOCKER A fix — dynamic --metals resolution (USER VOICE
    # 2026-10-03 verbatim "得搞成动态的"): --metals no longer hard-coded
    # to FIVE_METALS; supports any metal string, plus literal 'auto' mode
    # for CSV distinct metal-column discovery. Runtime validation
    # rejects metals not present in --csv with a clear "Available metals:
    # [...]" error message.
    # Wf-A8 PlatinAI 4-cell loader (audit gap #1 fix): resolve --csv
    # from --dataset if --csv not explicitly provided.
    if args.csv is None:
        args.csv = _resolve_default_csv_path(args.dataset)
        print(
            f"[Wf-A8 PlatinAI 4-cell loader] --csv resolved from "
            f"--dataset={args.dataset!r}: {args.csv}",
            flush=True,
        )
    if not args.csv.exists():
        raise FileNotFoundError(
            f"--csv path does not exist: {args.csv}. "
            f"Pass --csv <path> to override the default for "
            f"--dataset={args.dataset!r} (resolver: "
            f"_resolve_default_csv_path('{args.dataset}'))."
        )
    # Wf-A10 — fail loudly on a mismatched cohort/axis pairing rather than
    # silently collapsing the 4-cell structure to a single "Pt" task.
    if args.dataset == "platinai_4cell" and args.task_axis == "metal":
        raise ValueError(
            "--dataset platinai_4cell is a CELL-LINE cohort: a single metal "
            "(Pt) measured across 4 cell lines (MCF7 / A549 / A2780 / "
            "A2780cis). With --task-axis metal the pool collapses to one 'Pt' "
            "task and the 4-cell structure is lost. Pass "
            "--task-axis cell_line so the shared encoder gets one head per "
            "cell line (this is what makes per-cell AUC, the metric Rusanov "
            "2025 reports, available from cls_heads)."
        )
    if args.dataset == "nci60_9cell" and args.task_axis == "metal":
        raise ValueError(
            "--dataset nci60_9cell is a CELL-LINE cohort: one SMILES screened "
            "across 9 cell lines (MCF7 / A549 / A2780 / A2780cis / HCT116 / "
            "PC3 / HL60 / SKOV3 / DU145), with no metal column at all. With "
            "--task-axis metal the pool cannot even be built. Pass "
            "--task-axis cell_line so the shared encoder gets one head per "
            "cell line, which is what makes per-cell AUC available from "
            "cls_heads."
        )

    if args.dataset == "nci60_9cell":
        # Wf-A10 — long form with an explicit cell_line column, so the task
        # keys are read directly.  (The platinai branch below cannot be used
        # here: it looks for pIC50_<cell> column suffixes that this schema
        # does not have, and would silently discover zero tasks.)
        _nci = pd.read_csv(args.csv, usecols=["cell_line"])
        _available_metals = sorted(set(_nci["cell_line"].astype(str).tolist()))
    elif args.task_axis == "cell_line":
        # The raw platinai CSV has no cell_line column — the cell lines are the
        # suffixes of the pIC50_<cell> columns.
        _plate = pd.read_csv(args.csv, nrows=1)
        _available_metals = sorted(
            str(c)[len("pIC50_") :]
            for c in _plate.columns
            if str(c).startswith("pIC50_")
        )
    else:
        _metals_df = pd.read_csv(args.csv, usecols=["metal"])
        _available_metals = sorted(set(_metals_df["metal"].astype(str).tolist()))
    _expanded_metals: List[str] = []
    for _m in args.metals:
        if _m == "auto":
            for _avail in _available_metals:
                if _avail not in _expanded_metals:
                    _expanded_metals.append(_avail)
        else:
            _expanded_metals.append(_m)
    _missing_metals = [m for m in _expanded_metals if m not in _available_metals]
    if _missing_metals:
        raise ValueError(
            f"Metal(s) {_missing_metals} not found in CSV {args.csv}. "
            f"Available metals: {_available_metals}. "
            f"Use --metals auto for auto-discovery, or pass a space-separated "
            f"list of metals that exist in the CSV."
        )
    args.metals = _expanded_metals
    print(
        f"[Wf-A6 BLOCKER A] --metals resolved: {args.metals} "
        f"(available={_available_metals} csv={args.csv})",
        flush=True,
    )

    # Wf-130 — GPU-first device defaults: cuda:0 (GPU 0 — RTX PRO 6000
    # Blackwell by default) when allocatable, else CPU.
    # Wf-A9 FIX (2026-10-03) — ``args.cuda_visible_devices`` defaults to
    # the empty string, which the argparse help documents as "inherit
    # shell env or fall back to '0'".  ``apply_default_cuda_visible_devices``
    # however treats a *non-None* ``""`` as an explicit "force no GPUs"
    # override (Wf-415-A3 SCAFFOLD-gate contract), so passing the default
    # straight through asked every normal launch for CPU.  Coerce empty →
    # None so the documented "inherit" semantics actually apply: the env
    # var exported by start.sh (from yaml ``compute.gpu``) wins.
    apply_default_cuda_visible_devices(override=(args.cuda_visible_devices or None))
    if args.device:
        device = torch.device(args.device)
        print(f"[wf131-mtw] explicit device override: {device}", flush=True)
    else:
        device = resolve_default_device()
    print(
        f"[wf131-mtw] device={device} metals={args.metals} "
        f"seeds={args.seeds} epochs={args.epochs} alpha={args.alpha} "
        f"w_w={args.wasserstein_weight} l2_guard={args.use_l2_guard} "
        f"ablation_config={args.ablation_config} w2_gradient_mode={args.w2_gradient_mode} "
        f"dmpnn_backend={args.dmpnn_backend}",
        flush=True,
    )
    # Wf-405-A2 — Wf-380 NEW CLI flag env-var dispatch + logging.
    # triton_ot: set TRITON_MINIBATCH_OT=1 + TRITON_USE_FUSED=1 env vars
    # BEFORE any model init so triton_minibatch_ot_coupling env-var
    # gate picks up the activation.
    if bool(getattr(args, "triton_ot", False)):
        # Wf-A9 fix: the module-level ``_wf130_os`` alias is deleted after
        # the CUDA-default bootstrap, so re-bind it for this dispatch.
        import os as _wf130_os

        _wf130_os.environ.setdefault("TRITON_MINIBATCH_OT", "1")
        _wf130_os.environ.setdefault("TRITON_USE_FUSED", "1")
        print(
            "[Wf-405-A2] --triton-ot ENABLED: "
            "TRITON_MINIBATCH_OT=1 + TRITON_USE_FUSED=1 dispatched",
            flush=True,
        )
    # 2026-10-08 — §5.7 FLAG_ONLY fix for the three per-kernel Triton gates.
    #
    # ``--use-triton-wasserstein-sliced`` IS live: wasserstein_loss.py's
    # dispatch hook reads ``TRITON_WASSERSTEIN_SLICED`` (plus the global
    # ``TRITON_USE_FUSED`` master gate), so dispatching the env var here is
    # all that was missing — flipping the YAML key was previously a silent
    # no-op because nothing ever set those vars.
    if bool(getattr(args, "use_triton_wasserstein_sliced", False)):
        import os as _wf130_os

        _wf130_os.environ.setdefault("TRITON_WASSERSTEIN_SLICED", "1")
        _wf130_os.environ.setdefault("TRITON_USE_FUSED", "1")
        print(
            "[Wf-335-P3] --use-triton-wasserstein-sliced ENABLED: "
            "TRITON_WASSERSTEIN_SLICED=1 + TRITON_USE_FUSED=1 dispatched",
            flush=True,
        )
    # ``--use-triton-bond-head`` / ``--use-triton-smiles-featurize`` have NO
    # consumer in this trainer: the fused kernels expect a 3-layer bond-head
    # MLP / a featurisation call site that this architecture does not have
    # (UnifiedAtomBondFlowHead.bond_head is a single Linear; featurisation
    # runs through RDKit upstream).  Rather than accept the flag and silently
    # compute nothing (§5.7's exact failure mode), refuse it loudly.
    for _dead_flag in (
        "use_triton_bond_head",
        "use_triton_smiles_featurize",
    ):
        if bool(getattr(args, _dead_flag, False)):
            raise SystemExit(
                f"--{_dead_flag.replace('_', '-')} has no consumer in "
                "train_dmpnn_multitask_wasserstein.py: the fused kernel's "
                "target call site does not exist in this architecture. "
                "Setting it would be a silent no-op (§5.7 FLAG_ONLY), so it "
                "is rejected instead. Remove the flag or wire a real call "
                "site before enabling. See configs/dmpnn.yaml model.* notes."
            )
    # per_cohort_features: log the per-cohort feature config at
    # training start (Wf-380-A2 SCAFFOLD wire-in).
    if bool(getattr(args, "per_cohort_features", False)):
        try:
            from molmetal.scripts.wf380a2_m2_r2_per_cohort import (
                select_features,
            )

            _cfg = select_features(int(len(args.metals) * 200))
            print(
                f"[Wf-405-A2] --per-cohort-features ENABLED: "
                f"tier={_cfg.get('tier', '?')} "
                f"features={_cfg.get('features', '?')} "
                f"alpha={_cfg.get('alpha', '?')} "
                f"n_metals={len(args.metals)}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[Wf-405-A2][WARNING] --per-cohort-features log "
                f"failed: {exc}; continuing without per-cohort logging.",
                flush=True,
            )

    # Wf-A9 FIX (2026-10-03) — dispatch on --dataset.  Previously the
    # 5-metal table was loaded unconditionally and the platinai branch
    # computed ``split_df_platinai`` that nothing ever consumed, so
    # --dataset platinai_4cell died in load_5metal_split_table() on the
    # platinai schema ("5-metal CSV missing columns: {...}").
    # Default --dataset=metalcytotox_5metal preserves Wf-131 P1 +
    # Wf-251-F4 baseline bit-for-bit.
    if args.dataset == "platinai_4cell":
        # Wf-A10 — cell-line task axis.  Yields BOTH split_df and ds, because
        # the platinai labels come from the pivoted pIC50 columns rather than
        # from MetalCytoToxDB.csv (which holds no Pt rows at all).
        split_df, ds = build_platinai_cell_line_axis(
            args.csv,
            float(args.active_pic50_threshold),
            str(args.active_labeling),
        )
        print(
            f"[Wf-A10 cell-line axis] dataset=platinai_4cell "
            f"task_axis=cell_line: {len(split_df)} long-format rows; "
            f"tasks={sorted(split_df['metal'].unique().tolist())}; "
            f"per-cell breakdown: "
            f"{dict(split_df.groupby('cell_line').size())}; "
            f"active_labeling={args.active_labeling}"
            + (
                f" (global threshold pIC50 > {args.active_pic50_threshold})"
                if args.active_labeling == "global"
                else " (per-cell median split)"
            ),
            flush=True,
        )
    elif args.dataset == "nci60_9cell":
        # Wf-A10 — REAL-labelled 9-cell-line cohort on the cell-line task
        # axis.  Labels come from the CSV's own `active` column (m21's 10 uM
        # convention — see build_nci60_cell_line_axis), and the scaffold
        # split reproduces train_nci60_10cell.py byte-for-byte, so any AUC
        # measured here is comparable to the shipped m21 baseline
        # (0.8007 +/- 0.0160).
        split_df, ds = build_nci60_cell_line_axis(args.csv)
        print(
            f"[Wf-A10 cell-line axis] dataset=nci60_9cell "
            f"task_axis=cell_line: {len(split_df)} long-format rows; "
            f"tasks={sorted(split_df['metal'].unique().tolist())}; "
            f"per-cell breakdown: "
            f"{dict(split_df.groupby('cell_line').size())}; "
            f"per-split: {dict(split_df.groupby('split').size())}; "
            f"active_rate={float(split_df['active'].mean()):.4f} "
            f"(labels from CSV, m21 10 uM convention)",
            flush=True,
        )
    else:
        split_df = load_5metal_split_table(args.csv)
        print(
            f"[wf131-mtw] loaded 5-metal split table: {len(split_df)} rows",
            flush=True,
        )

        if args.loose_filter:
            flt = CytotoxFilter(
                time_threshold=None,
                ic50_min=None,
                metal_whitelist=list(args.metals),
                compute_pic50=True,
                compute_active=True,
            )
        else:
            flt = CytotoxFilter(
                time_threshold=24.0,
                ic50_min=0.01,
                metal_whitelist=list(args.metals),
                compute_pic50=True,
                compute_active=True,
            )
        if args.data_csv is not None:
            # Use a local cache dir to avoid /mnt/storage permission errors.
            local_cache = Path("/tmp") / "wf131_mtw_cache"
            local_cache.mkdir(parents=True, exist_ok=True)
            import pandas as _pd

            df_loaded = _pd.read_csv(args.data_csv, low_memory=False)
            from molmetal.data import cytotox as _cytotox_mod

            df_loaded = _cytotox_mod._apply_filters(df_loaded, flt)
            if flt.compute_pic50:
                df_loaded = _cytotox_mod._add_pic50(df_loaded)
            if flt.compute_active:
                df_loaded = _cytotox_mod._add_active(df_loaded)
            ds = MetalCytotoxDataset(df=df_loaded, cache_dir=local_cache)
        else:
            ds = MetalCytotoxDataset.from_csv(filters=flt)
    print(
        f"[wf131-mtw] loaded MetalCytoToxDataset: {len(ds)} rows "
        f"across {sorted(set(ds.metals))}",
        flush=True,
    )

    w_config = WassersteinConfig(n_mc=int(args.n_mc), seed=0)

    overall_start = time.time()
    all_results: Dict[str, Dict[str, MetalSeedResult]] = {}
    for metal in args.metals:
        all_results[metal] = {}
        metal_dir = args.out_dir / metal.lower()
        metal_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            # Wf-251-F3 — ablation config dispatcher.  The
            # ``mtl_baseline`` config preserves the Wf-131 P1 shipped
            # behaviour bit-for-bit (single_task=False, hidden=None,
            # depth=None).  ``single_task`` config forwards the Wf-197
            # lever 1 flag.  ``mtl_structured`` and ``mtl_gradproj`` are
            # reserved for future F3 grid runs (per-metal weight-sharing
            # mask + PCGrad gradient projection); today they alias to
            # the baseline behaviour so the grid aggregator sees the
            # expected JSON shape.
            effective_single_task = args.single_task
            effective_hidden = args.hidden
            effective_depth = args.depth
            if args.ablation_config == "single_task":
                effective_single_task = True
            elif args.ablation_config == "mtl_structured":
                # Force the production backbone so the structured-head
                # lever is observable in the per-seed AUC.
                effective_hidden = (
                    effective_hidden if effective_hidden is not None else MT_PROD_HIDDEN
                )
                effective_depth = (
                    effective_depth if effective_depth is not None else MT_PROD_DEPTH
                )
            elif args.ablation_config == "mtl_gradproj":
                # Same as mtl_baseline for now; the gradient-projection
                # hook lives inside ``MultiTaskLoss`` (forwarded via
                # ``--w2-gradient-mode`` in F4).
                pass
            try:
                res = train_one_metal_one_seed(
                    metal=metal,
                    seed=seed,
                    epochs=args.epochs,
                    alpha=args.alpha,
                    lr=args.lr,
                    wd=float(
                        getattr(args, "weight_decay", 0.0)
                    ),  # Wf-A6 A2 BLOCKER C fix
                    # Wf-A9 FIX (2026-10-03) — gradient-clip threshold
                    # from the yaml-driven ``args.max_norm`` threading
                    # down to the loss-cascade clipping site.
                    max_norm=float(getattr(args, "max_norm", 5.0)),
                    # Wf-A9 FIX (2026-10-03) — resolved joint pool so the
                    # model's metal-embedding table matches the cohort
                    # (Pt-only for platinai_4cell, the canonical 5 otherwise).
                    metals_pool=tuple(args.metals),
                    # T1-FIX (2026-10-03) — thread --max-joint-rows so
                    # the cap actually reaches the live pool assembly
                    # (was FLAG_ONLY: declared, never read; §5.7).
                    max_joint_rows=getattr(args, "max_joint_rows", None),
                    # Wf-R29 — thread --use-triton-relu-mlp into the model
                    # constructor.  Resolved from the post-YAML ``args`` so
                    # the configs/dmpnn.yaml ``model.use_triton_relu_mlp``
                    # key is honoured (§5.7 FLAG_ONLY fix).
                    use_triton_relu_mlp=bool(
                        getattr(args, "use_triton_relu_mlp", False)
                    ),
                    # 2026-10-04 — thread the four compute.* flags.  Without
                    # this the kwargs default to False and the flags stay
                    # FLAG_ONLY (§5.7): accepted by argparse, reach nothing.
                    torch_compile=bool(getattr(args, "torch_compile", False)),
                    tf32=bool(getattr(args, "tf32", DEFAULT_TF32)),
                    bf16=bool(getattr(args, "bf16", DEFAULT_BF16)),
                    # Wf-335-R1 — v2 flag overrides v1 when explicitly set.
                    torch_compile_mode=str(
                        getattr(args, "torch_compile_mode_v2", None)
                        if getattr(args, "torch_compile_mode_v2", None)
                        not in (None, DEFAULT_TORCH_COMPILE_MODE)
                        else getattr(
                            args, "torch_compile_mode", DEFAULT_TORCH_COMPILE_MODE
                        )
                    ),
                    dmpnn_backend=str(
                        getattr(args, "dmpnn_backend", DEFAULT_DMPNN_BACKEND)
                    ),
                    cudnn_benchmark=bool(getattr(args, "cudnn_benchmark", False)),
                    channels_last=bool(getattr(args, "channels_last", False)),
                    # 2026-10-04 — thread the two remaining dead yaml keys
                    # (protocol.gradient_accumulation_steps / data.pin_memory)
                    # into the live loop.  Without this the kwargs default to
                    # 1 / True and the flags stay FLAG_ONLY (§5.7): accepted
                    # by argparse, reach nothing.
                    gradient_accumulation_steps=max(
                        1, int(getattr(args, "gradient_accumulation_steps", 1))
                    ),
                    pin_memory=bool(getattr(args, "pin_memory", True)),
                    # 2026-10-04 — EMA kwargs (the yaml `train.ema_decay`).
                    use_ema=bool(getattr(args, "use_ema", False)),
                    ema_decay=float(getattr(args, "ema_decay", 0.999)),
                    batch_size=args.batch_size,
                    w_w=args.wasserstein_weight,
                    use_l2_guard=args.use_l2_guard,
                    ds=ds,
                    split_df=split_df,
                    ckpt_dir=args.ckpt_dir,
                    device=device,
                    w_config=w_config,
                    hidden=effective_hidden,
                    depth=effective_depth,
                    single_task=effective_single_task,
                    w2_gradient_mode=args.w2_gradient_mode,
                    # Wf-427-A3 — thread the 3 NEW CLI flag kwargs
                    # from ``args`` into the trainer so the
                    # Wf-405-A2 wire-in reaches the loss-cascade AND
                    # the Wf-424-A1 DANN wire-in reaches the model
                    # constructor.
                    use_ot_cfm=bool(getattr(args, "use_ot_cfm", False)),
                    # 2026-10-08 §5.7 — thread the W₁/W₂ closed-form flags
                    # from args into the trainer loss cascade.
                    use_wasserstein_w1=bool(
                        getattr(args, "use_wasserstein_w1", False)
                    ),
                    use_wasserstein_2=bool(getattr(args, "use_wasserstein_2", False)),
                    per_cohort_features=bool(
                        getattr(args, "per_cohort_features", False)
                    ),
                    use_dann=bool(getattr(args, "dann", False)),
                    # 2026-10-05 — ``--kendall`` CLI flag wire-in
                    # (AGENTS.md §5.7 FLAG_ONLY).  Match the ``dann``
                    # pattern above: ``getattr(args, "kendall", False)``
                    # preserves the Wf-131 P1 baseline (no Kendall UW MTL
                    # loss) on the no-flag path.  ``--use-kendall`` activates
                    # the Kendall tau UW MTL cascade via the new
                    # ``use_kendall_loss`` kwarg threaded into
                    # ``train_one_metal_one_seed()``.
                    use_kendall_loss=bool(
                        getattr(args, "kendall", False)
                    ),
                    # 2026-10-08 — opt-in ridge readout on frozen encoder
                    # embeddings (AGENTS.md §5.7 wire-in).
                    pic50_ridge_readout=bool(
                        getattr(args, "pic50_ridge_readout", False)
                    ),
                    # Wf-432-A2 — Wf-246-C metal-coordination term
                    # wire-in (Fix-B).  These kwargs thread from main()
                    # into ``train_one_metal_one_seed()`` →
                    # ``train_one_epoch_joint()`` so the kcal/mol
                    # adjustment contributes to the composite loss.
                    metal_coord_term=bool(getattr(args, "metal_coord_term", False)),
                    metal_coord_weight=float(getattr(args, "metal_coord_weight", 0.5)),
                    cal_sign_vina_convention=bool(
                        getattr(args, "cal_sign_vina_convention", True)
                    ),
                    # Wf-A6 A2 BLOCKER C fix — LR scheduler (cosine /
                    # onecycle) opt-in kwargs thread from main() into
                    # ``train_one_metal_one_seed()`` so the scheduler
                    # actually reaches the training loop. The
                    # scheduler is instantiated in the function body
                    # AFTER optimizer creation so it can read the
                    # optimizer.defaults[0]['lr'] for OneCycleLR's
                    # max_lr base. Default OFF preserves Wf-131 P1 +
                    # Wf-251-F4 baseline bit-for-bit.
                    lr_scheduler=str(getattr(args, "lr_scheduler", "none")),
                    lr_warmup_epochs=int(getattr(args, "lr_warmup_epochs", 5)),
                    lr_min_lr=float(getattr(args, "lr_min_lr", 1e-6)),
                    # Wf-A5 N3 — checkpoint resumption kwargs
                    # (``--resume-from`` + ``--resume-from-path`` +
                    # ``--checkpoint-interval``) thread from main()
                    # into ``train_one_metal_one_seed()`` so the
                    # _load_checkpoint_for_resume() call fires before
                    # the build_split_indices loop.
                    resume_from=str(getattr(args, "resume_from", "auto")),
                    resume_from_path=getattr(args, "resume_from_path", None),
                    checkpoint_interval=int(getattr(args, "checkpoint_interval", 50)),
                    # Wf-TUI NEW: per-epoch emit_event("eval_epoch", ...)
                    # call in the inner loop reads from the module
                    # global _EVENT_LOG_PATH (set by _install_event_emitter()
                    # in main() at job-start from --events-jsonl-path).
                    # No kwarg threading required.
                )
            except Exception as exc:
                print(
                    f"  [{metal} seed={seed}] FAILED: {exc!r}",
                    flush=True,
                )
                continue
            all_results[metal][str(seed)] = res
            seed_payload = {
                "metal": metal,
                "seed": seed,
                "epochs": args.epochs,
                "alpha": args.alpha,
                "wasserstein_weight": args.wasserstein_weight,
                "use_l2_guard": args.use_l2_guard,
                "ckpt_path": res.ckpt_path,
                "hidden": args.hidden if args.hidden is not None else DEFAULT_HIDDEN,
                "depth": args.depth if args.depth is not None else DEFAULT_DEPTH,
                "n_train": res.n_train,
                "n_val": res.n_val,
                "n_test": res.n_test,
                "elapsed_seconds": res.elapsed_seconds,
                "val_metrics": res.val_metrics,
                "test_metrics": res.test_metrics,
                "val_pic50_metrics": res.val_pic50_metrics,
                "test_pic50_metrics": res.test_pic50_metrics,
                "pic50_ridge_metrics": res.pic50_ridge_metrics,
                "wasserstein_history": res.wasserstein_history,
                "status": res.status,
            }
            (metal_dir / f"per_seed_metrics_seed{seed}.json").write_text(
                json.dumps(seed_payload, indent=2)
            )

    overall_elapsed = time.time() - overall_start

    # Per-metal aggregate (mean ± std over seeds)
    summary_per_metal: Dict[str, Dict] = {}
    for metal, seed_results in all_results.items():
        if not seed_results:
            summary_per_metal[metal] = {
                "n_seeds": 0,
                "status": "ALL_SEEDS_FAILED",
            }
            continue
        aucs = np.array(
            [
                r.test_metrics.get("roc_auc", float("nan"))
                for r in seed_results.values()
            ],
            dtype=float,
        )
        pearsons = np.array(
            [
                r.test_pic50_metrics.get("pearson", float("nan"))
                for r in seed_results.values()
            ],
            dtype=float,
        )
        rmses = np.array(
            [
                r.test_pic50_metrics.get("rmse", float("nan"))
                for r in seed_results.values()
            ],
            dtype=float,
        )
        first = next(iter(seed_results.values()))
        summary_per_metal[metal] = {
            "n_seeds": len(seed_results),
            "n_train": first.n_train,
            "n_val": first.n_val,
            "n_test": first.n_test,
            "test_auc_mean": float(np.nanmean(aucs))
            if np.isfinite(aucs).any()
            else float("nan"),
            "test_auc_std": float(np.nanstd(aucs))
            if np.isfinite(aucs).any()
            else float("nan"),
            "test_pearson_mean": float(np.nanmean(pearsons)),
            "test_pearson_std": float(np.nanstd(pearsons)),
            "test_rmse_mean": float(np.nanmean(rmses)),
            "test_rmse_std": float(np.nanstd(rmses)),
            "ckpt_paths": [r.ckpt_path for r in seed_results.values()],
            "status": "OK",
        }

    summary = {
        "wf_tag": (
            "Wf-131 Phase 1 — 5-metal multi-task D-MPNN with Wasserstein "
            "pIC50 regularisation (Ru AUC rescue)"
        ),
        "metals": args.metals,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "wasserstein_weight": args.wasserstein_weight,
        "use_l2_guard": args.use_l2_guard,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden": args.hidden if args.hidden is not None else DEFAULT_HIDDEN,
        "depth": args.depth if args.depth is not None else DEFAULT_DEPTH,
        "device": str(device),
        "elapsed_seconds_total": overall_elapsed,
        "per_metal": summary_per_metal,
        # Wf-328-W3/W4/W1 — precision + compile state for paper §6
        # reproducibility audits (was previously not persisted).
        "tf32": configure_tf32(
            enabled=bool(getattr(args, "tf32", DEFAULT_TF32))
        ).to_dict(),
        "bf16": configure_bf16(
            enabled=bool(getattr(args, "bf16", DEFAULT_BF16)), device=device
        ),
        "torch_compile_enabled": bool(getattr(args, "torch_compile", False)),
        "torch_compile_mode": str(
            getattr(args, "torch_compile_mode_v2", None)
            if getattr(args, "torch_compile_mode_v2", None)
            not in (None, DEFAULT_TORCH_COMPILE_MODE)
            else getattr(args, "torch_compile_mode", DEFAULT_TORCH_COMPILE_MODE)
        ),
        # Wf-335-R1 — distinguish pre-fix (reduce-overhead / CUDA Graphs ON)
        # runs from post-fix runs in downstream audits.
        "torch_compile_cuda_graphs_disabled": (
            str(getattr(args, "torch_compile_mode", DEFAULT_TORCH_COMPILE_MODE))
            != "reduce-overhead"
            and str(getattr(args, "torch_compile_mode_v2", DEFAULT_TORCH_COMPILE_MODE))
            != "reduce-overhead"
        ),
        "dmpnn_backend": str(
            getattr(args, "dmpnn_backend", DEFAULT_DMPNN_BACKEND)
        ),
        "dmpnn_backend_requested": str(
            getattr(args, "dmpnn_backend", DEFAULT_DMPNN_BACKEND)
        ),
        "w2_gradient_mode": str(args.w2_gradient_mode),
        "ru_auc_target": 0.86,
        "ru_auc_baseline_single_task": 0.7123,
        "honest_framing_note": (
            "Wf-131 P1 stacks multi-task (shared encoder + 5 per-metal heads) "
            "+ Wasserstein pIC50 regularisation + L2 gradient-norm guard.  "
            "Smaller metals (Rh/Os/Re < 1.2k rows) give noisier per-seed "
            "AUC; per-seed spread is reported unfiltered.  The Ru AUC ≥ 0.86 "
            "target is PROJECTED (party-twenty-spirit §3.5), not measured — "
            "the recipe stack has not been validated end-to-end on this "
            "cohort, so we report mean ± std honestly and let the per-seed "
            "metrics speak for themselves."
        ),
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[wf131-mtw] DONE total elapsed={overall_elapsed:.1f}s",
        flush=True,
    )
    for metal, s in summary_per_metal.items():
        print(
            f"  {metal}: n_train={s.get('n_train', '?')} n_val={s.get('n_val', '?')} "
            f"n_test={s.get('n_test', '?')} test_auc="
            f"{s.get('test_auc_mean', float('nan')):.4f}±{s.get('test_auc_std', float('nan')):.4f} "
            f"test_pearson={s.get('test_pearson_mean', float('nan')):.3f}",
            flush=True,
        )

    # Per-metal per_seed_metrics.json (one file per metal, all seeds)
    for metal, seed_results in all_results.items():
        if not seed_results:
            continue
        metal_dir = args.out_dir / metal.lower()
        per_seed_rows = []
        for seed_str, res in seed_results.items():
            per_seed_rows.append(
                {
                    "seed": int(seed_str),
                    "ckpt_path": res.ckpt_path,
                    "n_train": res.n_train,
                    "n_val": res.n_val,
                    "n_test": res.n_test,
                    "elapsed_seconds": res.elapsed_seconds,
                    "val_auc": float(res.val_metrics.get("roc_auc", float("nan"))),
                    "test_auc": float(res.test_metrics.get("roc_auc", float("nan"))),
                    "test_pearson": float(
                        res.test_pic50_metrics.get("pearson", float("nan"))
                    ),
                    "test_rmse": float(
                        res.test_pic50_metrics.get("rmse", float("nan"))
                    ),
                    "test_mae": float(res.test_pic50_metrics.get("mae", float("nan"))),
                    "wasserstein_history": res.wasserstein_history,
                    "status": res.status,
                }
            )
        per_metal_aggregate = summary_per_metal.get(metal, {})
        payload = {
            "metal": metal,
            "seeds": args.seeds,
            "n_seeds_total": len(args.seeds),
            "n_seeds_trained_now": len(seed_results),
            "per_seed": per_seed_rows,
            "aggregate": per_metal_aggregate,
            "status": per_metal_aggregate.get("status", "OK"),
        }
        (metal_dir / "per_seed_metrics.json").write_text(json.dumps(payload, indent=2))

    # Wf-TUI NEW: close the events.jsonl file handle (flush + release).
    # Per AGENTS.md §5.6 atomic_write_file safe_overwrite_check contract,
    # file-handle releases are explicit (no implicit GC).
    _close_event_emitter()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
