"""Top-level PyTorch modules composing MolFlow-Triton.

This package groups the high-level neural network components.  The
model layer never imports a GPU backend directly: scatter-style
aggregations go through :func:`models._scatter.scatter_sum` which
hides the underlying Triton / PyTorch implementation choice behind a
single primitive.

- :class:`models.encoder.MolEncoder`    - encodes a batch of molecular
  graphs into per-node and per-edge hidden representations.
- :class:`models.velocity_net.VelocityNet` - SE(3)-equivariant network
  predicting a per-atom velocity field, used as the flow-matching
  score / velocity model.
- :class:`models.conditioner.Conditioner` (with ``GaussianSmearing``)
  - encodes scalar physical / molecular properties into a dense
  embedding that is injected into the velocity network.

The model layer never owns training state (no optimizer, no loop);
training lives in :mod:`scripts` and consumes the modules from here.
"""

from __future__ import annotations

from .encoder import MolEncoder
from .velocity_net import VelocityNet, EGNNLayer
from .conditioner import Conditioner, GaussianSmearing

__all__ = [
    "MolEncoder",
    "VelocityNet",
    "EGNNLayer",
    "Conditioner",
    "GaussianSmearing",
]