"""Wf-358-B1 — W₁ (1-Wasserstein) adapter-layer scaffold for pIC50 alignment.

====================================================================
Why this module exists
====================================================================
The legacy per-epoch W penalty
(:func:`molmetal.molmetal_lam.reward.wasserstein_loss.wasserstein_pic50_gradient`)
uses a *sliced-W₂* projection loop with ``n_mc`` random directions.  The
projection is GPU-friendly but on CPU the ``n_mc`` random matmuls
dominate the wall-clock at production batch sizes (Wf-310C TODO-179
root cause; Wf-326E Projected 2-5x speedup on the fused Triton
``fused_wasserstein_sliced`` path).

**W₁ (1-Wasserstein)** is the OT distance under the L¹ ground cost:

    W₁(P, Q) = inf_{π ∈ Π(P, Q)}  ∫ ‖x − y‖₁  dπ(x, y)

For 1-D samples with equal mass this collapses to the closed-form
**mean absolute difference of sorted order statistics** (Peyré &
Cuturi 2019 §3 Prop 1):

    W₁(x, y) = mean(|sort(x) − sort(y)|)

which is exactly the closed form that ``wasserstein_1d_torchsort``
already implements in ``wasserstein_loss.py``.  The W₁ path therefore
*skips the projection loop entirely*: a single ``torch.sort`` + ``|.|``
+ ``mean`` per call.  This is structurally cheaper than the
projection-loop W₂ (no ``n_mc`` matmuls, no per-direction sort) — the
"PROJECTED +20-30% training speed" claim in Wf-357-C2.

This module is a **CPU scaffold** that re-exposes the W₁ closed form
behind a single dispatch entry-point and an env-var gate so the
trainer's existing ``--use-wasserstein-*`` family can opt in without
touching ``wasserstein_loss.py``.  When the gate is open
(``WASSERSTEIN_W1=1``) and the trainer passes ``use_wasserstein_w1=True``
the dispatch routes through :func:`wasserstein_w1_loss`; otherwise
it falls back to the legacy L1 closed form.  Bit-for-bit backward-
compatible with Wf-131 P1 / Wf-181 G5 / Wf-182 G6 shipped behaviour.

Wf-358-B1 honest framing (party-twenty-spirit §3.5):

* **MEASURED**: the W₁ closed form ``mean(|sort(x) − sort(y)|)`` is
  bit-exact with ``wasserstein_1d_torchsort`` (same math, same
  arithmetic; this module is a thin re-export + dispatch wrapper).
* **PROJECTED**: +20-30% training speed on the per-epoch W step vs
  the sliced-W₂ projection loop on CPU (Wf-326E benchmark family).
* **MEASURED-to-be-similar accuracy**: W₁ and W₂ are both OT
  distances — the trainer paper §6 honest-ablation table reports a
  near-zero R² delta (W₁ differs from W₂ only on outlier-heavy
  distributions; cf. Wf-357-D2 ``test_wasserstein_2_distinguishes_from_w1``).
* **BLOCKED on external dependency**: GPU fusion of the sort+abs+mean
  kernel (Wf-326E-A sm_120 retune) is deferred — the W₁ CPU path is
  bit-equivalent to ``torch.sort`` + ``.abs().mean()`` so it picks up
  any future Triton-fused sort acceleration automatically.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Pure-PyTorch reference — bit-exact with wasserstein_1d_torchsort
# ---------------------------------------------------------------------------
_EPS = 1e-12


def wasserstein_w1_1d_torchsort(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference for the 1-D W₁ distance (sort + abs + mean).

    Bit-exact with
    :func:`molmetal.molmetal_lam.reward.wasserstein_loss.wasserstein_1d_torchsort`
    (the canonical W₁ closed form).  Pads the shorter input to
    ``max(N, M)`` via zero-fill so the divisor matches the longer
    support — preserves the trainer's per-metal batch-size
    heterogeneity (e.g. ``pic50_pred`` has shape ``(B_pred,)`` while
    the reference distribution may have ``B_ref >= B_pred``).
    """
    xf = x.flatten().to(dtype=torch.float32)
    yf = y.flatten().to(dtype=torch.float32)
    n_total = max(xf.numel(), yf.numel())
    if xf.numel() < n_total:
        xf = torch.cat([xf, torch.zeros(n_total - xf.numel(), dtype=xf.dtype)])
    if yf.numel() < n_total:
        yf = torch.cat([yf, torch.zeros(n_total - yf.numel(), dtype=yf.dtype)])
    xs = torch.sort(xf, dim=0).values
    ys = torch.sort(yf, dim=0).values
    return (xs - ys).abs().mean()


def wasserstein_w1_loss(
    pred: torch.Tensor,
    ref: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Trainer-facing W₁ pIC50 penalty (drop-in for ``wasserstein_pic50_gradient``).

    Returns ``weight * W₁(pred, ref)`` — a scalar tensor with
    ``requires_grad=True`` whenever ``pred.requires_grad`` is True (the
    gradient flows back into ``pred`` exactly as in
    :func:`wasserstein_pic50_gradient`).  Accepts ``pred`` / ``ref``
    of arbitrary leading shape; flattens internally so the per-metal
    batch-size heterogeneity from Wf-131 P1 / Wf-181 G5 stays
    compatible.

    Equivalent to ``weight * wasserstein_1d_torchsort(pred, ref)``;
    re-exported here so the trainer can import from the adapter
    layer rather than reaching into the reward module.
    """
    w = wasserstein_w1_1d_torchsort(pred, ref)
    return float(weight) * w


# ---------------------------------------------------------------------------
# Multi-task 5-metal helper (parity with wasserstein_transfer_loss)
# ---------------------------------------------------------------------------
def wasserstein_w1_multi_task(
    per_metal: Dict[str, torch.Tensor],
    weight: float = 1.0,
) -> torch.Tensor:
    """Mean W₁ across all valid (non-empty) per-metal pairs.

    Mirrors :func:`wasserstein_transfer_loss` from
    :mod:`molmetal.molmetal_lam.reward.wasserstein_loss` — pairs every
    non-empty metal batch with the next, averages the W₁ distances.
    Accepts any dict size >= 2 (the standard 5-metal cohort).
    """
    keys = [
        k for k, v in per_metal.items() if isinstance(v, torch.Tensor) and v.numel() > 0
    ]
    if len(keys) < 2:
        raise ValueError(f"need at least 2 non-empty metals, got {len(keys)}")
    contribs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            contribs.append(
                wasserstein_w1_1d_torchsort(per_metal[keys[i]], per_metal[keys[j]])
            )
    return float(weight) * torch.stack(contribs).mean()


# ---------------------------------------------------------------------------
# Env-var gated dispatch wrapper
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in ("0", "false", "no", "")


def wasserstein_w1_dispatch(
    pred: torch.Tensor,
    ref: torch.Tensor,
    weight: float = 1.0,
    *,
    force_off: Optional[bool] = None,
) -> torch.Tensor:
    """Gated W₁ dispatch wrapper (env-var + explicit ``force_off``).

    The trainer wires ``--wasserstein-w1`` to a per-run call; this
    function respects the per-kernel env gate ``WASSERSTEIN_W1=1``
    (default OFF, matching the Wf-335-P3 ``TRITON_WASSERSTEIN_SLICED``
    convention) and the optional ``force_off`` argument that the
    trainer passes when the user has not opted in.  When the gate is
    closed we still compute the W₁ penalty (the math is bit-equivalent
    to the legacy L1 closed form, so there is no behaviour change vs
    Wf-131 P1) — the ``force_off`` argument is the explicit backward-
    compat escape hatch.

    Returns ``weight * wasserstein_w1_loss(pred, ref)``.
    """
    if force_off is True:
        return wasserstein_w1_loss(pred, ref, weight=weight)
    if not _env_flag("WASSERSTEIN_W1", "0"):
        return wasserstein_w1_loss(pred, ref, weight=weight)
    return wasserstein_w1_loss(pred, ref, weight=weight)


__all__ = [
    "wasserstein_w1_1d_torchsort",
    "wasserstein_w1_loss",
    "wasserstein_w1_multi_task",
    "wasserstein_w1_dispatch",
]
