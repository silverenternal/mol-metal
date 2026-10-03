"""BatchNorm → GroupNorm replacement utility (TODO-143 §2.5 Step 1.2).

Why this exists
---------------
chemprop v2's ``MPNN`` backbone uses ``nn.BatchNorm1d`` on the message-passing
output (``chemprop/models/model.py`` line 94) and on per-atom/predictor hidden
states (``chemprop/models/mol_atom_bond.py`` line 164). BatchNorm running
statistics are fragile when batch sizes are small or vary across multi-domain
training (MetalCytoToxDB ≈ 5K rows, DrugOOD ≈ 1.6M, NCI-60 ≈ 4.9M,
BindingDB ≈ 1.1M). GroupNorm normalises per-sample and is insensitive to
batch size — a better fit for our cross-domain D-MPNN training.

This module ships the *generic* replacement helper so it can be applied:
  - to a chemprop v2 ``MPNN`` instance if/when we drop it in, or
  - to any third-party ``nn.Module`` we adapt that ships with BatchNorm.

Honest scope note (2026-09-27)
------------------------------
The Mol-Metal in-tree D-MPNN (``molmetal/baselines/dmpnn*.py`` and
``molmetal/models/{dmpnn,v4_multitask_dmpnn}.py``) currently has **zero**
``nn.BatchNorm*`` layers — it is a no-norm architecture by design. So the
hook we wire into ``train_dmpnn_multitask.py`` is currently a no-op for our
backbone; it will become load-bearing the moment a chemprop-style MPNN is
imported as a backbone. The hook is wired in now so the workflow is ready.

API
---
    replace_batchnorm_with_groupnorm(module, num_groups=32, verbose=False)
        In-place replacement. Returns the (mutated) ``module`` for chaining.
    count_batchnorm(module) -> int
        Non-mutating diagnostic count.
    num_groups_for(num_channels, requested=32) -> int
        Resolves the largest divisor of ``num_channels`` that is <= ``requested``
        (or ``num_channels`` itself when no such divisor exists).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "replace_batchnorm_with_groupnorm",
    "count_batchnorm",
    "num_groups_for",
]


def num_groups_for(num_channels: int, requested: int = 32) -> int:
    """Resolve a GroupNorm-friendly ``num_groups`` for ``num_channels``.

    GroupNorm requires ``num_channels % num_groups == 0``. We choose the
    largest divisor of ``num_channels`` that does not exceed ``requested``;
    if no such divisor exists (e.g. very small ``num_channels``), we fall
    back to ``num_channels`` itself (equivalent to LayerNorm across channels).
    """
    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    if requested <= 0:
        raise ValueError(f"requested num_groups must be > 0, got {requested}")

    # Try requested first, then halve until it divides num_channels cleanly.
    g = min(requested, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return g


def count_batchnorm(module: nn.Module) -> int:
    """Count BatchNorm1d / BatchNorm2d / BatchNorm3d submodules (non-mutating)."""
    return sum(
        1
        for _ in module.modules()
        if isinstance(_, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    )


def replace_batchnorm_with_groupnorm(
    module: nn.Module,
    num_groups: int = 32,
    verbose: bool = False,
) -> nn.Module:
    """Recursively replace every BatchNorm{1d,2d,3d} with a GroupNorm.

    Parameters
    ----------
    module : nn.Module
        Module to mutate in place. Also returned for chaining.
    num_groups : int, default 32
        Requested number of groups. Falls back via ``num_groups_for`` so the
        result is always a valid ``nn.GroupNorm`` (channels % groups == 0).
    verbose : bool, default False
        If True, print a one-line summary per replacement.

    Returns
    -------
    nn.Module
        The same module, mutated in place. Returned for fluent chaining:

            model = replace_batchnorm_with_groupnorm(model, num_groups=32)

    Notes
    -----
    - In-place; side effects: any BatchNorm running buffers (``running_mean``,
      ``running_var``, ``num_batches_tracked``) are discarded because
      GroupNorm has no running stats. This is intended — we are abandoning
      BatchNorm behaviour by design.
    - BatchNorm ``affine`` is preserved: GroupNorm default has
      ``affine=True``, so learnable ``weight``/``bias`` are kept.
    - The replacement does NOT touch ``nn.SyncBatchNorm`` — that class lives
      outside the standard ``BatchNorm*`` hierarchy and is rarely used in
      single-GPU D-MPNN training.
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be > 0, got {num_groups}")

    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = child.num_features
            resolved_groups = num_groups_for(num_channels, requested=num_groups)
            new_layer = nn.GroupNorm(
                num_groups=resolved_groups,
                num_channels=num_channels,
                eps=child.eps,
                affine=child.affine,
            )
            # Copy affine params if present (GroupNorm has weight/bias too).
            if child.affine:
                with torch.no_grad():
                    new_layer.weight.copy_(child.weight)
                    new_layer.bias.copy_(child.bias)
            setattr(module, name, new_layer)
            if verbose:
                cls_name = type(child).__name__
                print(
                    f"[groupnorm] {cls_name}({num_channels}) -> "
                    f"GroupNorm({resolved_groups}, {num_channels})"
                )
        else:
            replace_batchnorm_with_groupnorm(
                child, num_groups=num_groups, verbose=verbose
            )
    return module
