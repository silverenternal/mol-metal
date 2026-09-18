"""MolFlow-Triton low-level Triton kernel library.

This package provides custom GPU kernels for the MolFlow-Triton project,
targeting AMD ROCm GPUs (gfx1101) via ``triton-rocm==3.8.0``.

Modules
-------
autotune
    Shared :func:`triton.autotune` configuration restricted to
    ``num_warps in {2, 4, 8}`` and ``num_stages in {2, 3, 4}`` — the
    only knobs that matter on RDNA3 (``waves_per_eu`` is CDNA-only).
ode_solver
    Custom ODE integrator step kernels (Euler and RK4) operating on
    batched state vectors of shape ``(batch_size, dim)``.
equivariant_ops
    SE(3) equivariant primitives: neighbor vector aggregation
    (scatter-sum with a per-edge mask) and rotation matrices built
    from axis-angle representations.
fused_norm
    Fused LayerNorm / RMSNorm kernels that compute mean, variance,
    normalization and the affine transform in a single launch.
matmul
    Reference Triton matmul kernel (port of the canonical
    ``03-matrix-multiplication.py`` tutorial), with the autotune grid
    trimmed for gfx1101.

All kernels are written so that no hard-coded device string is
required: callers pass a :class:`torch.device` (typically
``torch.device('cuda')``, which on ROCm resolves to the HIP device)
or rely on ``torch.zeros``/``torch.empty`` to pick the same device
as the input tensor.

Note on naming
--------------
The submodule ``matmul`` is imported as ``from . import matmul`` so it
is reachable as ``triton_kernels.matmul`` (the module).  The public
function inside it is exposed via the package attribute
``triton_kernels.matmul_fn`` so it does not shadow the submodule name.
Typical usage::

    from triton_kernels.matmul import matmul as triton_matmul
    # or
    from triton_kernels import matmul_fn
"""

from __future__ import annotations

# Submodule imports — these register each submodule as an attribute of
# this package (``triton_kernels.autotune``, ``triton_kernels.matmul``,
# ...).  Do NOT use ``from .matmul import matmul`` here, because that
# would overwrite the submodule attribute with the function.
from . import autotune
from . import batched_mlp
from . import config
from . import equivariant_ops
from . import fused_cross_entropy
from . import fused_dropout
from . import fused_mlp
from . import fused_gelu_mlp as fused_gelu_mlp_module
from . import fused_norm
from . import fused_residual_add
from . import fused_rmsnorm_residual
from . import fused_softmax
from . import matmul
from . import ode_solver

# Convenience re-exports at the package root.
AUTOTUNE_CONFIGS = autotune.AUTOTUNE_CONFIGS
autotune_fn = autotune.autotune  # avoid shadowing the submodule
euler_step = ode_solver.euler_step
rk4_step = ode_solver.rk4_step
aggregate_vectors = equivariant_ops.aggregate_vectors
rotation_from_axis_angle = equivariant_ops.rotation_from_axis_angle
fused_layer_norm = fused_norm.fused_layer_norm
fused_rms_norm = fused_norm.fused_rms_norm
fused_silu_mlp = fused_mlp.fused_silu_mlp
fused_gelu_mlp = fused_mlp.fused_gelu_mlp
FusedGeluMLP = fused_gelu_mlp_module.FusedGeluMLP
softmax_last_dim = fused_softmax.softmax_last_dim
fused_cross_entropy = fused_cross_entropy.fused_cross_entropy
fused_dropout_residual = fused_dropout.fused_dropout_residual
fused_residual_add = fused_residual_add.fused_residual_add
fused_rmsnorm_residual = fused_rmsnorm_residual.fused_rmsnorm_residual
batched_fused_silu_mlp = batched_mlp.batched_fused_silu_mlp
batched_fused_gelu_mlp = batched_mlp.batched_fused_gelu_mlp
batched_fused_silu_mlp_per_k = batched_mlp.batched_fused_silu_mlp_per_k
batched_fused_gelu_mlp_per_k = batched_mlp.batched_fused_gelu_mlp_per_k
matmul_fn = matmul.matmul  # expose the function under a non-conflicting name
triton_config = config.triton_config  # process-wide dispatch gate

__all__ = [
    # Autotune
    "AUTOTUNE_CONFIGS",
    "autotune_fn",
    # ODE integrator
    "euler_step",
    "rk4_step",
    # Equivariant ops
    "aggregate_vectors",
    "rotation_from_axis_angle",
    # Fused normalization
    "fused_layer_norm",
    "fused_rms_norm",
    # Fused MLP
    "fused_silu_mlp",
    "fused_gelu_mlp",
    "FusedGeluMLP",
    # Fused softmax + cross-entropy
    "softmax_last_dim",
    "fused_cross_entropy",
    # Fused dropout + residual
    "fused_dropout_residual",
    # Fused element-wise residual addition
    "fused_residual_add",
    "fused_rmsnorm_residual",
    # Batched fused MLP (K MLPs in one launch)
    "batched_fused_silu_mlp",
    "batched_fused_gelu_mlp",
    "batched_fused_silu_mlp_per_k",
    "batched_fused_gelu_mlp_per_k",
    # Matmul
    "matmul_fn",
    # Dispatch configuration
    "triton_config",
]

__version__ = "0.2.0"
