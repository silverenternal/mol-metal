#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How large is the gradient of the vector-channel normalisation at zero?

WHY THIS EXISTS
---------------
``VectorChannelLayer`` unit-normalises each vector channel.  Until Wf-A108j
that was

    vec = vec * gain / (|vec| + eps)        # eps = 1e-6

which emits 0 for an empty channel (correct) but hands back a ``1/eps``
Jacobian (not correct).  MEASURED consequence on the Wf-A108j smoke
(``configs/smoke_wfa108j_lsq.yaml``, first backward pass): the pre-clip
gradient of ``vc_layers.0.channel_mix.weight`` was **2.90e11**, against
``O(10)`` for every other tensor in the model, because AdamW's second moment
then holds ~1e22 and that parameter is frozen for thousands of steps.

The same division also turns *numerical cancellation* into a signal: a
channel whose norm had decayed to ``2.99e-08`` was divided by its own norm
and emitted at **unit** length, i.e. float32 round-off became a
full-amplitude direction field.

HOW THE OLD BEHAVIOUR IS REPRODUCED
-----------------------------------
The shipped layer divides by ``norms.clamp_min(min_norm)`` and zeroes any
channel with ``norms <= min_norm``.  A NEGATIVE floor switches both halves
back: ``norms > -eps`` is true even at zero, and ``clamp_min(-eps)`` makes
the denominator ``-eps`` there — the old ``|vec| + eps`` up to sign, with
the old ``1/eps`` Jacobian.  So ``min_norm = -1e-6`` is the pre-fix arm and
``min_norm = 1e-3`` is the shipped one, and the only difference between the
two arms is the floor.

Usage:
  python molmetal/scripts/wfa108j_norm_floor_grad.py \
      --out metrics/by_metric/wfa108j_norm_floor_grad.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from molmetal.adapters.flow_matching_lipman import (  # noqa: E402
    EGNNVelocityField,
)
from molmetal.adapters.flow_matching_lipman.vector_channels import (  # noqa: E402
    VectorChannelLayer,
)


def _batch(n_clouds: int = 8, n_atoms: int = 20, sigma: float = 2.485,
           seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x1 = torch.randn(n_clouds, n_atoms, 3, generator=g) * 2.0
    z = torch.full((n_clouds, n_atoms), 6, dtype=torch.long)
    z[:, 0] = 78
    x0 = torch.randn(x1.shape, generator=g) * sigma
    xt = 0.5 * x0 + 0.5 * x1
    tgt = x1 - x0
    return xt.reshape(-1, 3), z.reshape(-1), tgt.reshape(-1, 3), [n_atoms] * n_clouds


def _full_model_grad(min_norm: float, seed: int = 0) -> Dict[str, Any]:
    """One backward from the FM MSE through the whole velocity field."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    torch.manual_seed(seed)
    x, z, tgt, sizes = _batch(seed=seed)
    b = len(sizes)
    mask = torch.ones(b, x.shape[0] // b, dtype=torch.bool)
    xb = x.reshape(b, -1, 3)
    edge_index, edge_mask = _block_diag_radius_edges(xb, mask, 6.0)

    f = EGNNVelocityField(
        hidden_dim=64,
        n_layers=2,
        velocity_head="vector_channels",
        vector_channels=8,
        vector_channel_norm_gain="fixed",
    )
    for layer in f.vc_layers:
        layer.min_norm = float(min_norm)
    f.train()
    out = f(
        xb, z.reshape(b, -1), edge_index, torch.full((b,), 0.5), edge_mask=edge_mask
    )
    loss = F.mse_loss(out["vel"].reshape(-1, 3), tgt)
    loss.backward()

    rows = []
    for name, p in f.named_parameters():
        if p.grad is not None:
            rows.append((float(p.grad.detach().norm()), name))
    rows.sort(reverse=True)
    total = sum(r[0] ** 2 for r in rows) ** 0.5
    return {
        "min_norm": float(min_norm),
        "fm_loss": float(loss.detach()),
        "global_grad_norm": total,
        "top_params": [{"name": n, "grad_norm": v} for v, n in rows[:5]],
    }


def _prefix_layer_grad(eps: float, n_ch: int = 4, n_atoms: int = 6) -> Dict[str, Any]:
    """Gradient of ``channel_mix.weight`` under the PRE-FIX formula.

    The old denominator was ``|vec| + eps``, not ``clamp_min(eps)``: the
    difference matters, because the derivative of ``clamp_min`` passes
    through ``|vec|`` and is undefined at zero, while ``|vec| + eps`` has the
    finite derivative ``1/eps`` there.  Reproducing the old arm therefore
    needs the old expression written out, which is what this function does.
    """
    torch.manual_seed(0)
    layer = VectorChannelLayer(hidden_dim=8, n_channels=n_ch)
    vec = torch.randn(1, n_atoms, n_ch, 3)
    vec = vec * torch.tensor([1.0, 1.0, 1.0, 0.0]).view(1, 1, n_ch, 1)
    mixed = vec + torch.einsum("bncd,ec->bned", vec, layer.channel_mix.weight)
    norms = mixed.norm(dim=-1, keepdim=True)
    pre = mixed * layer.norm_gain / (norms + float(eps))
    pre.pow(2).sum().backward()
    g = layer.channel_mix.weight.grad
    return {
        "eps": float(eps),
        "grad_norm": float(g.norm()) if g is not None else 0.0,
        "out_norm_at_zero_channel": float(pre.detach()[:, :, 3, :].abs().max()),
    }


def _layer_grad(min_norm: float, n_ch: int = 4, n_atoms: int = 6) -> Dict[str, Any]:
    """Gradient of ``channel_mix.weight`` when one channel is exactly zero.

    This is the condition padding creates: a padded atom sits at the origin
    with every channel at exactly zero, so the normalisation divides by the
    floor for every one of its channels, at every layer.
    """
    torch.manual_seed(0)
    layer = VectorChannelLayer(hidden_dim=8, n_channels=n_ch, min_norm=min_norm)
    h = torch.zeros(1, n_atoms, 8)
    pos = torch.randn(1, n_atoms, 3)
    vec = torch.randn(1, n_atoms, n_ch, 3)
    vec = vec * torch.tensor([1.0, 1.0, 1.0, 0.0]).view(1, 1, n_ch, 1)
    edge_index = torch.zeros(1, 2, 0, dtype=torch.long)
    edge_mask = torch.zeros(1, 0, dtype=torch.bool)
    _, out = layer(h, pos, vec, edge_index, edge_mask=edge_mask)
    out.pow(2).sum().backward()
    g = layer.channel_mix.weight.grad
    return {
        "min_norm": float(min_norm),
        "grad_norm": float(g.norm()) if g is not None else 0.0,
        "out_norm_at_zero_channel": float(out.detach()[:, :, 3, :].abs().max()),
    }


def _noise_out(min_norm: float, n_ch: int = 4) -> float:
    """Output magnitude for an input channel of norm 1e-8."""
    torch.manual_seed(0)
    layer = VectorChannelLayer(hidden_dim=8, n_channels=n_ch, min_norm=min_norm)
    h = torch.zeros(1, 4, 8)
    pos = torch.randn(1, 4, 3)
    vec = torch.randn(1, 4, n_ch, 3)
    vec[:, :, 1, :] = torch.tensor([1e-8, 0.0, 0.0])
    edge_index = torch.zeros(1, 2, 0, dtype=torch.long)
    edge_mask = torch.zeros(1, 0, dtype=torch.bool)
    _, out = layer(h, pos, vec, edge_index, edge_mask=edge_mask)
    return float(out.detach()[:, :, 1, :].norm(dim=-1).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics/by_metric/wfa108j_norm_floor_grad.json")
    args = ap.parse_args()

    arms: List[Dict[str, Any]] = []
    for label, floor in (
        ("pre_fix_eps1e-6", -1e-6),
        ("pre_fix_eps1e-4", -1e-4),
        ("pre_fix_eps1e-2", -1e-2),
        ("shipped_min_norm1e-3", 1e-3),
        ("shipped_min_norm1e-2", 1e-2),
    ):
        rep = _full_model_grad(floor)
        rep["arm"] = label
        arms.append(rep)
        print(
            f"[wfa108j] {label:22s} global={rep['global_grad_norm']:.4e} "
            f"top={rep['top_params'][0]['name']} "
            f"{rep['top_params'][0]['grad_norm']:.4e}"
        )

    noise = [
        {"arm": "pre_fix_eps1e-6", "min_norm": -1e-6, "out_norm": _noise_out(-1e-6)},
        {"arm": "shipped_min_norm1e-3", "min_norm": 1e-3, "out_norm": _noise_out(1e-3)},
    ]
    for r in noise:
        print(f"[wfa108j] noise {r['arm']:22s} out_norm={r['out_norm']:.4e}")

    pre_arms = [_prefix_layer_grad(e) for e in (1e-6, 1e-4, 1e-2, 1.0)]
    for r in pre_arms:
        print(
            f"[wfa108j] prefix eps={r['eps']:g} grad={r['grad_norm']:.4e} "
            f"out_at_zero={r['out_norm_at_zero_channel']:.4e}"
        )

    layer_arms: List[Dict[str, Any]] = []
    for label, floor in (
        ("shipped_min_norm1e-6", 1e-6),
        ("shipped_min_norm1e-3", 1e-3),
        ("shipped_min_norm1e-2", 1e-2),
    ):
        rep = _layer_grad(floor)
        rep["arm"] = label
        layer_arms.append(rep)
        print(
            f"[wfa108j] layer {label:22s} grad={rep['grad_norm']:.4e} "
            f"out_at_zero={rep['out_norm_at_zero_channel']:.4e}"
        )

    payload: Dict[str, Any] = {
        "metric": "wfa108j_norm_floor_grad",
        "tag": "MEASURED",
        "date": "2026-10-25",
        "what": (
            "One backward from the FM MSE through a fresh vector-channel "
            "velocity field (8 clouds x 20 atoms, hidden 64, 2 layers), for "
            "the pre-fix normalisation and for the shipped floor. A negative "
            "floor reproduces the old |v|+eps denominator (see the module "
            "docstring); the only difference between arms is the floor."
        ),
        "script": "molmetal/scripts/wfa108j_norm_floor_grad.py",
        "batch": {"n_clouds": 8, "n_atoms": 20, "hidden": 64, "n_layers": 2,
                  "channels": 8, "sigma": 2.485},
        "arms": arms,
        "pre_fix_layer": pre_arms,
        "layer_arms": layer_arms,
        "layer_by_arm": {
            a["arm"]: {
                "grad_norm": a["grad_norm"],
                "out_norm_at_zero_channel": a["out_norm_at_zero_channel"],
            }
            for a in layer_arms
        },
        "noise_amplification": noise,
        "by_arm": {
            a["arm"]: {
                "global_grad_norm": a["global_grad_norm"],
                "fm_loss": a["fm_loss"],
            }
            for a in arms
        },
    }
    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"[wfa108j] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
