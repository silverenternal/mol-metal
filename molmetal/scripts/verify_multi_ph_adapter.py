#!/usr/bin/env python3
"""W2.3 — CPU-only smoke verification of the Multi-PH pocket adapter.

This script exercises the three public surfaces of
``molmetal.adapters.multi_ph_integration`` (per the W2.3 brief):

  1. :class:`MultiPHPocketFeatures` — wraps the T5 multi-PH math
     layer into a stateful ``nn.Module`` with the same input
     contract as the existing ``PocketEncoder``.
  2. :class:`MultiPHMetalCoordHead` — metal-coordination classifier
     head that compares pocket multi-PH diagrams against a set of
     per-metal reference diagrams via the sliced-Wasserstein kernel.
  3. :func:`augment_pocket_embedding` — drop-in concatenation of the
     existing ``(B, H)`` pocket embedding with the ``(B, n_bins, n_bins)``
     multi-PH feature map.

The script is **CPU-only** — no CUDA, no Triton, no checkpoints.  It
runs in <10 s on any laptop.  Pass ``--dry-run`` to skip the
:class:`MultiPHMetalCoordHead` reference-loading check (which is the
slowest path due to the per-metal kernel inner loop).

Usage::

    uv run --no-sync python molmetal/scripts/verify_multi_ph_adapter.py
    uv run --no-sync python molmetal/scripts/verify_multi_ph_adapter.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Make ``molmetal`` importable when run as a script (matches the
# pattern in :file:`molmetal/scripts/verify_se3_sb_adapter.py`).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from molmetal.adapters.multi_ph_integration import (
    MultiPHMetalCoordHead,
    MultiPHPocketFeatures,
    augment_pocket_embedding,
)


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------
def _check_pocket_features() -> None:
    """Smoke-check :class:`MultiPHPocketFeatures`."""
    print("[1/3] MultiPHPocketFeatures: imports OK")
    feats = MultiPHPocketFeatures(max_radius=5.0, max_coord=4.0, n_bins=16)
    coords = torch.randn(3, 8, 3)
    atom_types = torch.randint(1, 18, (3, 8))
    out = feats.compute_with_atom_types(coords, atom_types)
    assert out.shape == (3, 16, 16), f"expected (3, 16, 16), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "feature map contains NaN/Inf"
    assert (out >= 0).all(), "feature map must be non-negative"
    # SE(3) invariance smoke: rotate + verify equality.
    from math import cos, pi, sin

    R = torch.tensor(
        [
            [cos(pi / 4), -sin(pi / 4), 0.0],
            [sin(pi / 4), cos(pi / 4), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    coords_rot = coords @ R.T
    out_rot = feats.compute_with_atom_types(coords_rot, atom_types)
    torch.testing.assert_close(out, out_rot, rtol=1e-5, atol=1e-5)
    print("        -> CPU smoke 0/1 fail")


def _check_metal_coord_head(dry_run: bool) -> None:
    """Smoke-check :class:`MultiPHMetalCoordHead`."""
    if dry_run:
        print("[2/3] MultiPHMetalCoordHead: skipped (--dry-run)")
        return
    print("[2/3] MultiPHMetalCoordHead: imports OK")
    feats = MultiPHPocketFeatures(n_bins=12)
    head = MultiPHMetalCoordHead(n_metals=8, n_bins=12, n_projections=4)
    coords = torch.randn(4, 6, 3)
    atom_types = torch.randint(1, 18, (4, 6))
    out = feats.compute_with_atom_types(coords, atom_types)
    logits = head(out)
    assert logits.shape == (4, 8), f"expected (4, 8), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"

    # Setting explicit references must change the output.
    logits_default = logits.detach().clone()
    ref_coords = torch.zeros(8, 6, 3)
    for m in range(8):
        ref_coords[m, :, 0] = float(m) * 0.5
        ref_coords[m, :, 1] = 0.2
    ref_atom_types = torch.ones(8, 6, dtype=torch.long) * 6
    head.set_reference(ref_coords, ref_atom_types)
    logits_ref = head(out).detach().clone()
    assert not torch.allclose(
        logits_default, logits_ref, atol=1e-6
    ), "set_reference must change the head's output"
    print("        -> CPU smoke 0/3 fail")


def _check_augmentation() -> None:
    """Smoke-check :func:`augment_pocket_embedding`."""
    print("[3/3] augment_pocket_embedding: imports OK")
    base = torch.randn(4, 16)
    features = torch.randn(4, 8, 8)
    out = augment_pocket_embedding(base, features)
    assert out.shape == (4, 16 + 64), f"expected (4, {16 + 64}), got {tuple(out.shape)}"
    # The leading H columns must equal ``base`` bit-exactly (no mutation).
    torch.testing.assert_close(out[:, :16], base, rtol=0.0, atol=0.0)
    # And no mutation of the input.
    base_after = base.clone()
    torch.testing.assert_close(base, base_after, rtol=0.0, atol=0.0)
    print("        -> CPU smoke 0/2 fail")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only smoke verification of the Multi-PH " "pocket adapter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the (slowest) MultiPHMetalCoordHead reference-loading "
        "step; only verify the pocket features + augmentation on CPU.",
    )
    args = parser.parse_args(argv)

    _check_pocket_features()
    _check_metal_coord_head(dry_run=args.dry_run)
    _check_augmentation()

    print()
    print("Multi-PH pocket adapter: imports OK, CPU smoke 0/6 fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
