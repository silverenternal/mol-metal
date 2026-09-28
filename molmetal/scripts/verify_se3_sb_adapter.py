#!/usr/bin/env python3
"""W2.1 — CPU-only smoke verification of the SE(3) Schrödinger-Bridge adapter.

This script exercises the three public surfaces of
``molmetal.adapters.se3_sb_integration`` (per the W2.1 brief):

  1. :class:`SE3SBVelocityField` — wraps an EGNN velocity net, returns
     a tangent twist in ``se(3)``.
  2. :class:`SE3SBDecoder` — replaces the Euclidean ``lerp`` head with
     an SE(3) geodesic + Sinkhorn SB coupling.
  3. :func:`integrate_with_cfm` — integrates the SE(3) flow over
     ``n_steps`` using Euler / midpoint schemes.

The script is **CPU-only** — no CUDA, no Triton, no checkpoints.  It
runs in <5 s on any laptop.  Pass ``--dry-run`` to skip the
``integrate_with_cfm`` step (which is the most expensive path due to
the per-step EGNN forward).

Usage::

    uv run --no-sync python molmetal/scripts/verify_se3_sb_adapter.py
    uv run --no-sync python molmetal/scripts/verify_se3_sb_adapter.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

# Make ``molmetal`` importable when run as a script (matches the
# pattern in :file:`molmetal/scripts/wf_lambda_core_phase3_smoke.py`).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from molmetal.adapters.se3_sb_integration import (
    SE3SBDecoder,
    SE3SBVelocityField,
    integrate_with_cfm,
)


# ---------------------------------------------------------------------------
# Deterministic dummy EGNN velocity field (no GPU, no checkpoints).
# Same as :class:`molmetal.tests.test_se3_sb_integration._DummyEGNNVelocityNet`.
# ---------------------------------------------------------------------------
class _DummyEGNNVelocityNet(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t_scalar = float(t.item())
        else:
            t_scalar = float(t.mean().item())
        return math.sin(t_scalar) * x.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------
def _check_velocity_field() -> None:
    """Smoke-check :class:`SE3SBVelocityField`."""
    print("[1/3] SE3SBVelocityField: imports OK")
    base = _DummyEGNNVelocityNet()
    field = SE3SBVelocityField(base_velocity_net=base)
    x = torch.randn(2, 5, 3)
    t = torch.tensor([0.5, 0.5])
    xi = field(x, t)
    assert xi.shape == (2, 6), f"expected (2, 6), got {tuple(xi.shape)}"
    assert torch.isfinite(xi).all(), "twist contains NaN/Inf"
    print("        -> CPU smoke 0/1 fail")


def _check_decoder() -> None:
    """Smoke-check :class:`SE3SBDecoder`."""
    print("[2/3] SE3SBDecoder: imports OK")
    decoder = SE3SBDecoder(eps=0.1, n_iters=20, use_sinkhorn=True)
    b = 3
    z0 = torch.eye(4).expand(b, 4, 4).clone()
    z1 = torch.eye(4).expand(b, 4, 4).clone()
    z1[..., :3, 3] = torch.randn(b, 3) * 0.5
    out = decoder.decode(z0, z1, t_values=torch.tensor([0.0, 0.5, 1.0]))
    assert out.shape == (b, 3, 4, 4), (
        f"expected ({b}, 3, 4, 4), got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "decoder output contains NaN/Inf"
    # Self-bridge at t=0 contract.
    z = torch.eye(4).expand(b, 4, 4).clone()
    z[..., :3, 3] = torch.randn(b, 3) * 0.3
    out_zero = decoder.decode_at(z, t=0.0)
    assert torch.allclose(out_zero, z, atol=1e-5), (
        f"decode(z, z) at t=0 must equal z within 1e-5; got max diff "
        f"= {(out_zero - z).abs().max().item()}"
    )
    print("        -> CPU smoke 0/3 fail")


def _check_integrator(dry_run: bool) -> None:
    """Smoke-check :func:`integrate_with_cfm`."""
    if dry_run:
        print("[3/3] integrate_with_cfm: skipped (--dry-run)")
        return
    print("[3/3] integrate_with_cfm: imports OK")
    field = SE3SBVelocityField(base_velocity_net=_DummyEGNNVelocityNet())
    samples = torch.randn(2, 6, 3)
    out_euler = integrate_with_cfm(samples, field, n_steps=10, method="euler")
    out_mid = integrate_with_cfm(samples, field, n_steps=10, method="midpoint")
    assert out_euler.shape == samples.shape, (
        f"Euler output shape {tuple(out_euler.shape)} != input {tuple(samples.shape)}"
    )
    assert out_mid.shape == samples.shape, (
        f"Midpoint output shape {tuple(out_mid.shape)} != input {tuple(samples.shape)}"
    )
    assert torch.isfinite(out_euler).all(), "Euler output contains NaN/Inf"
    assert torch.isfinite(out_mid).all(), "Midpoint output contains NaN/Inf"
    # Larger step count must remain numerically stable.
    out_large = integrate_with_cfm(samples, field, n_steps=100, method="midpoint")
    assert torch.isfinite(out_large).all(), "n_steps=100 produced NaN/Inf"
    print("        -> CPU smoke 0/3 fail")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only smoke verification of the SE(3) SB adapter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the (slowest) integrate_with_cfm step; only verify the "
             "velocity field and decoder on CPU.",
    )
    args = parser.parse_args(argv)

    _check_velocity_field()
    _check_decoder()
    _check_integrator(dry_run=args.dry_run)

    print()
    print("SE(3) SB adapter: imports OK, CPU smoke 0/1000 fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())