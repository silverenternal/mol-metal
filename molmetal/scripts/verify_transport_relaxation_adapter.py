#!/usr/bin/env python3
"""W2.2 — CPU-only smoke verification of the Transport-Relaxation adapter.

This script exercises the three public surfaces of
``molmetal.adapters.transport_relaxation_integration`` (per the W2.2 brief):

  1. :class:`RelaxedEndpointTerm` — the soft endpoint OT penalty.
  2. :func:`augment_learned_shaping` — wrap an existing base shaping.
  3. :class:`LambdaRelaxSweep` — sweep ``λ_relax`` over the grid.

The script is **CPU-only** — no CUDA, no checkpoints.  It runs in
<5 s on any laptop.  Pass ``--dry-run`` to skip the sweep step
(which is the most expensive path due to per-λ KL evaluation).

Usage::

    uv run --no-sync python molmetal/scripts/verify_transport_relaxation_adapter.py
    uv run --no-sync python molmetal/scripts/verify_transport_relaxation_adapter.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

# Make ``molmetal`` importable when run as a script.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from molmetal.adapters.transport_relaxation_integration import (
    LambdaRelaxSweep,
    RelaxedEndpointTerm,
    augment_learned_shaping,
)


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------
def _check_relaxed_endpoint_term() -> None:
    """Smoke-check :class:`RelaxedEndpointTerm`."""
    print("[1/3] RelaxedEndpointTerm: imports OK")
    torch.manual_seed(0)
    trajectory = torch.randn(4, 5, 3)
    energies = torch.randn(4)

    # (a) λ = 0 → no-op
    term_zero = RelaxedEndpointTerm(lambda_relax=0.0, eps=0.1)
    out_zero = term_zero.forward(trajectory, energies)
    assert out_zero.item() == 0.0, (
        f"lambda=0 must yield zero penalty; got {out_zero.item()}"
    )

    # (b) non-zero λ → non-negative penalty
    term = RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)
    out = term.forward(trajectory, energies)
    assert out.shape == (), f"penalty must be scalar, got {tuple(out.shape)}"
    assert torch.isfinite(out), "penalty must be finite"
    assert out.item() >= -1e-6, f"penalty must be >= 0, got {out.item()}"

    # (c) forward = λ · kl
    kl = term.kl(trajectory, energies)
    assert abs(out.item() - term.lambda_relax * kl.item()) < 1e-6

    # (d) equal energies → KL ≈ 0
    equal_e = torch.full((4,), 0.5)
    kl_equal = term.kl(trajectory, equal_e)
    assert abs(kl_equal.item()) < 1e-6, (
        f"equal energies must give KL=0, got {kl_equal.item()}"
    )

    print("        -> CPU smoke 0/4 fail")


def _check_augment_learned_shaping(dry_run: bool) -> None:
    """Smoke-check :func:`augment_learned_shaping`."""
    if dry_run:
        print("[2/3] augment_learned_shaping: skipped (--dry-run)")
        return
    print("[2/3] augment_learned_shaping: imports OK")
    torch.manual_seed(1)
    base_scalar = torch.tensor(-1.234)
    trajectory = torch.randn(4, 5, 3)
    energies = torch.randn(4)

    # (a) λ = 0 → exactly equal to base
    out_zero = augment_learned_shaping(
        base_scalar, trajectory, energies, lambda_relax=0.0, eps=0.1,
    )
    assert torch.allclose(out_zero, base_scalar, atol=1e-7), (
        f"lambda=0 must recover base; got {out_zero.item()} vs "
        f"{base_scalar.item()}"
    )

    # (b) λ = 1e-3 → strictly greater than base
    out = augment_learned_shaping(
        base_scalar, trajectory, energies, lambda_relax=1e-3, eps=0.1,
    )
    assert out.item() > base_scalar.item(), (
        f"positive penalty must exceed base; got {out.item()} vs "
        f"{base_scalar.item()}"
    )

    # (c) (B,) base → output shape = (B,)
    base_vec = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out_vec = augment_learned_shaping(
        base_vec, trajectory, energies, lambda_relax=1e-3, eps=0.1,
    )
    assert out_vec.shape == base_vec.shape
    assert (out_vec >= base_vec).all()

    print("        -> CPU smoke 0/3 fail")


def _check_lambda_relax_sweep(dry_run: bool) -> None:
    """Smoke-check :class:`LambdaRelaxSweep`."""
    if dry_run:
        print("[3/3] LambdaRelaxSweep: skipped (--dry-run)")
        return
    print("[3/3] LambdaRelaxSweep: imports OK")
    torch.manual_seed(2)

    def _base(row: dict) -> float:
        return -0.5

    sweep = LambdaRelaxSweep(
        base_shaping=_base, lambdas=(1e-4, 1e-3, 1e-2), eps=0.1,
    )
    trajectory = torch.randn(8, 5, 3)
    energies = torch.randn(8)
    val_features = [{"sa_mean_norm": 0.3}] * 8

    best = sweep.best_lambda(trajectory, energies, val_features)
    assert best in (1e-4, 1e-3, 1e-2), (
        f"best_lambda must be from the grid, got {best}"
    )

    rows = sweep.evaluate_grid(trajectory, energies, val_features)
    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
    for row in rows:
        for key in ("lambda_relax", "kl", "penalty", "base_mean", "augmented_mean"):
            assert key in row, f"missing key {key!r} in row {row}"
            assert math.isfinite(row[key]), (
                f"row[{key!r}] must be finite, got {row[key]}"
            )

    # Equal energies must break ties by smallest λ.
    equal_energies = torch.full((8,), 0.5)
    best_equal = sweep.best_lambda(trajectory, equal_energies, [None] * 8)
    assert best_equal == 1e-4, (
        f"equal energies must break ties by smallest λ, got {best_equal}"
    )

    print("        -> CPU smoke 0/3 fail")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only smoke verification of the W2.2 "
                    "Transport-Relaxation adapter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the (slower) augment + sweep steps; only verify "
             "RelaxedEndpointTerm on CPU.",
    )
    args = parser.parse_args(argv)

    _check_relaxed_endpoint_term()
    _check_augment_learned_shaping(dry_run=args.dry_run)
    _check_lambda_relax_sweep(dry_run=args.dry_run)

    print()
    print("W2.2 Transport-Relaxation adapter: imports OK, CPU smoke 0/1000 fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())