"""Per-EGNN-layer gradient-checkpointing wrapper.

Wraps a single message-passing block (e.g.
:class:`molmetal.adapters.egnn_rocm.EquivariantGraphConv`) in
:class:`torch.utils.checkpoint.Checkpoint` so the activations produced
inside the block are freed after the forward pass and recomputed during
backprop.  This trades a roughly 30% increase in wall-clock per step for
``O(n_layers)`` activation memory, which is the headline VRAM relief for
Mol-Metal CFM training on a 16 GB RX 7800 XT (gfx1101, ROCm 7.2).

Why this layer (and not a blanket ``checkpoint(EGNN)``)?

The EGNN is a stack of message-passing layers + two linear projections
(`input_proj`, `output_proj`).  Checkpointing the whole module would
recompute the input/output projections for free, but those projections
are cheap and small — the activation-memory hot spot is the *inner loop*
of message passing, where each layer materialises a ``[n_edges,
hidden_dim]`` scalar message tensor and a ``[n_edges, 3]`` coord-message
tensor.  Wrapping each layer individually gives us per-layer activation
relief at the cost of one extra forward per layer in backward, which is
negligible relative to the message-passing compute.

Why ``use_reentrant=False``?

On PyTorch 2.14 + ROCm 7.2 + gfx1101, ``torch.utils.checkpoint.checkpoint``
with ``use_reentrant=True`` (the legacy default) raises::

    Error: Expected tensor metadata
    ...
    File "torch/_functorch/_aot_autograd/utils.py", line ...
        in tree_flatten_spec

The non-reentrant path uses the new functional autograd infrastructure,
which is fully supported on ROCm 7.2.  This wrapper **always** passes
``use_reentrant=False``.

Why ``preserve_rng_state=True``?

EGNN message passing does not currently use dropout, but the wrapper is
applied generically and the host module is free to add stochastic
regularisation.  ``preserve_rng_state=True`` ensures that the forward
RNG state is captured before the checkpointed forward and restored
inside the recomputed backward, so dropout / feature-noise / any other
stochastic op that runs inside ``self.block`` is bit-equivalent whether
the block runs once (eval / no-checkpoint) or twice (training /
checkpointed).

Lit anchors
-----------
- Chen et al. 2016 "Training Deep Nets with Sublinear Memory Cost"
  (arXiv:1604.06174) — the gradient-checkpointing technique.
- PyTorch docs "torch.utils.checkpoint.checkpoint" — ``use_reentrant``
  semantics (added in PyTorch 1.11+, defaults to ``False`` since
  PyTorch 2.0).
- PyTorch issue #72681 — known ``use_reentrant=True`` regressions on
  ROCm / HIP builds (reproduced on gfx1101 / ROCm 7.2 in
  ``molmetal/reports/wf_vram_fix/02_checkpoint.md``).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

# Process-wide default for the env-driven enable flag.  Read once at
# import; tests can monkeypatch this constant or override the env var
# before importing the wrapper.
#
# Default is "1" (enabled) because the whole point of this wrapper is to
# ship the VRAM TOP #1 fix; callers that want the legacy behaviour set
# ``MOLMETAL_EGNN_CHECKPOINT=0``.
_DEFAULT_ENABLED: bool = os.environ.get("MOLMETAL_EGNN_CHECKPOINT", "1") not in {
    "0",
    "false",
    "False",
    "no",
    "No",
    "NO",
    "",
}


def egnn_checkpoint_default_enabled() -> bool:
    """Return the process-wide default for the EGNN checkpoint flag.

    Mirrors the dispatch pattern of :func:`molmetal.adapters.egnn_rocm.
    use_fused_coord_update` so tests can introspect the gate without
    re-reading the env var themselves.
    """
    return _DEFAULT_ENABLED


def set_egnn_checkpoint_default_enabled(value: bool) -> None:
    """Override the process-wide default.

    Mirrors :func:`molmetal.adapters.egnn_rocm.set_use_fused_coord_update`
    so callers (and tests) can toggle the new path without touching the
    env var directly.  Pass ``False`` to revert to legacy behaviour.
    """
    global _DEFAULT_ENABLED
    _DEFAULT_ENABLED = bool(value)


class CheckpointedEGNNLayer(nn.Module):
    """Wraps a single EGNN message-passing block in
    :func:`torch.utils.checkpoint.checkpoint`.

    The wrapper is intentionally minimal — it does NOT inspect, project,
    or rewrite the wrapped block's I/O.  It is the caller's
    responsibility to ensure ``self.block.forward`` accepts the same
    arguments it would have accepted in the un-wrapped stack (e.g. for
    :class:`molmetal.adapters.egnn_rocm.EquivariantGraphConv` that is
    ``forward(h, x, edge_index, dative_bond_edge_attr=None,
    edge_types=None)``).

    Parameters
    ----------
    block : nn.Module
        The message-passing block to wrap.  Must be a drop-in for the
        unwrapped layer: ``forward(h, x, edge_index, **kwargs)``.
    enabled : bool, default ``True``
        When ``True``, :meth:`forward` routes through
        :func:`torch.utils.checkpoint.checkpoint` with
        ``use_reentrant=False`` and ``preserve_rng_state=True``.
        When ``False``, :meth:`forward` is an identity passthrough
        (``self.block(*args, **kwargs)``) so eval / inference and the
        legacy CPU-only debug path pay zero checkpointing overhead.

    Notes
    -----
    - ``use_reentrant=False`` is mandatory on PyTorch 2.14 + ROCm 7.2 +
      gfx1101 — the reentrant path raises ``Expected tensor metadata``
      during backward on HIP builds.  This wrapper **always** uses
      ``use_reentrant=False`` regardless of caller preferences.
    - ``preserve_rng_state=True`` keeps dropout (and any other RNG-
      consuming op inside ``self.block``) bit-equivalent between the
      first forward and the recomputed backward.
    - The wrapper exposes ``self.block`` so a caller can introspect,
      swap, or load state_dict into the underlying layer; loading
      state_dicts into the wrapper itself will go through to ``self.block``
      because ``nn.Module.__setattr__`` forwards attribute assignment
      of :class:`nn.Module` types into the sub-module registry.
    """

    def __init__(self, block: nn.Module, enabled: bool = True):
        super().__init__()
        if not isinstance(block, nn.Module):
            raise TypeError(
                f"CheckpointedEGNNLayer expected an nn.Module block, got "
                f"{type(block).__name__}"
            )
        self.block = block
        self.enabled = bool(enabled)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run ``self.block`` on ``*args, **kwargs``.

        When ``self.enabled is True``, route through
        :func:`torch.utils.checkpoint.checkpoint` with
        ``use_reentrant=False`` and ``preserve_rng_state=True``.

        When ``self.enabled is False``, return ``self.block(*args,
        **kwargs)`` directly so there is zero overhead vs. the
        unwrapped layer (no ``torch.utils.checkpoint`` context-manager
        allocation, no RNG-state capture/restore).
        """
        if not self.enabled:
            # Zero-overhead passthrough.  Identical behaviour to
            # ``self.block(*args, **kwargs)`` — kept explicit so a
            # reader can confirm the disabled path is just a forward.
            return self.block(*args, **kwargs)

        # NOTE: use_reentrant=False is mandatory on PyTorch 2.14 +
        # ROCm 7.2 + gfx1101 (raises "Expected tensor metadata" on the
        # reentrant path).  preserve_rng_state=True keeps dropout (and
        # any other RNG-consuming op inside self.block) bit-equivalent
        # between the first forward and the recomputed backward.
        return _torch_checkpoint(
            self.block,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
            **kwargs,
        )

    def extra_repr(self) -> str:
        return f"enabled={self.enabled}, block={type(self.block).__name__}"


__all__ = [
    "CheckpointedEGNNLayer",
    "egnn_checkpoint_default_enabled",
    "set_egnn_checkpoint_default_enabled",
]