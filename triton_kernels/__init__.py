"""MolFlow-Triton low-level Triton kernel library.

This package provides custom GPU kernels for the MolFlow-Triton project.
It targets two backends: CUDA Blackwell (``sm_100`` / ``sm_120``) and AMD
ROCm (``gfx1101``, RDNA3) via ``triton-rocm``.  Which tuning grid and
which kernels are enabled is decided at import time by
:mod:`triton_kernels.backend` — the two targets do not share a grid, and
one kernel that is correct on Blackwell corrupts tiles on gfx1101.

Modules
-------
backend
    Backend / SM-version detection plus the ``tl.dot`` precision policy.
    Everything else branches on this.
autotune
    Shared :func:`triton.autotune` configuration.  Two grids —
    ``AUTOTUNE_CONFIGS_CUDA`` (Blackwell) and ``AUTOTUNE_CONFIGS_ROCM``
    (gfx1101) — with ``AUTOTUNE_CONFIGS`` aliasing the active one.
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
    ``03-matrix-multiplication.py`` tutorial), with a per-backend
    autotune grid.

All kernels are written so that no hard-coded device string is
required: callers pass a :class:`torch.device` (typically
``torch.device('cuda')``, which resolves to the HIP device on ROCm)
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
from . import backend
from . import batched_mlp
from . import config
from . import equivariant_ops
from . import fused_cross_entropy
from . import fused_dropout
from . import fused_mlp
from . import fused_gelu_mlp as fused_gelu_mlp_module
from . import fused_norm
from . import fused_relu_mlp as fused_relu_mlp_module
from . import fused_residual_add
from . import fused_rmsnorm_residual
from . import fused_softmax
from . import matmul
from . import ode_solver

# Convenience re-exports at the package root.
AUTOTUNE_CONFIGS = autotune.AUTOTUNE_CONFIGS
AUTOTUNE_CONFIGS_CUDA = autotune.AUTOTUNE_CONFIGS_CUDA
AUTOTUNE_CONFIGS_ROCM = autotune.AUTOTUNE_CONFIGS_ROCM
AUTOTUNE_CONFIGS_FAST = autotune.AUTOTUNE_CONFIGS_FAST
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
# D-MPNN edge-update kernels.  ``fused_relu_mlp`` here is the *function*; the
# submodule stays importable as ``triton_kernels.fused_relu_mlp`` only until
# this attribute assignment shadows it, so the module is kept under the
# ``_module`` alias (same convention as ``fused_gelu_mlp_module``).
fused_relu_linear = fused_relu_mlp_module.fused_relu_linear
fused_relu_mlp = fused_relu_mlp_module.fused_relu_mlp
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
    "AUTOTUNE_CONFIGS_CUDA",
    "AUTOTUNE_CONFIGS_FAST",
    "AUTOTUNE_CONFIGS_ROCM",
    "autotune_fn",
    # Backend detection
    "backend",
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
    # Fused ReLU linear / MLP (D-MPNN edge update)
    "fused_relu_linear",
    "fused_relu_mlp",
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
