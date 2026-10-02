# -*- coding: utf-8 -*-
"""
Centralized precision policy — mirrors OmegaFold pipeline.py:38-50 `_set_precision`.

Single source of truth for:
  - torch.backends.cuda.matmul.allow_tf32
  - torch.backends.cudnn.allow_tf32
  - torch.backends.cudnn.benchmark
  - torch.set_float32_matmul_precision("high" | "medium" | "highest")
  - torch.backends.cuda.matmul.fp32_precision
  - torch.backends.cuda.conv.fp32_precision (BF16 / TF32 / FP32)

Used by train_cfm_yuelbond_wasserstein.py + infer_cfm.py for cross-call
consistency. Default OFF for bit-for-bit backward compat with Wf-431-B1.

NOTE: PyTorch quirk — `torch.set_float32_matmul_precision('highest')` resets
`torch.backends.cuda.matmul.allow_tf32` to False; `set_float32_matmul_precision('high')`
sets it to True. Order matters: matmul.allow_tf32 must be set AFTER
set_float32_matmul_precision, or set_float32_matmul_precision wins.
"""
import torch


# Default precision policy — bit-for-bit backward compat with Wf-431-B1
DEFAULT_POLICY = {
    "allow_tf32": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,  # for reproducibility
    "float32_matmul_precision": "highest",
    "channels_last": False,
}


def set_precision(
    allow_tf32: bool = DEFAULT_POLICY["allow_tf32"],
    cudnn_benchmark: bool = DEFAULT_POLICY["cudnn_benchmark"],
    cudnn_deterministic: bool = DEFAULT_POLICY["cudnn_deterministic"],
    float32_matmul_precision: str = DEFAULT_POLICY["float32_matmul_precision"],
    channels_last: bool = DEFAULT_POLICY["channels_last"],
) -> dict:
    """Apply precision policy globally.

    Mirrors omegafold/pipeline.py:38-50 `_set_precision(allow_tf32)`.

    Order matters: set_float32_matmul_precision is called BEFORE
    matmul.allow_tf32 because PyTorch's set_float32_matmul_precision('highest')
    RESETS matmul.allow_tf32 to False.

    Args:
        allow_tf32: enable TF32 fast paths on matmul/conv (default False)
        cudnn_benchmark: enable cudnn benchmark mode (default False)
        cudnn_deterministic: deterministic algorithms (default True)
        float32_matmul_precision: 'highest'/'high'/'medium' (default 'highest')
        channels_last: memory format hint (default False)

    Returns:
        policy: dict of applied settings (for logging / verification)
    """
    policy = {
        "allow_tf32": bool(allow_tf32),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "cudnn_deterministic": bool(cudnn_deterministic),
        "float32_matmul_precision": str(float32_matmul_precision),
        "channels_last": bool(channels_last),
    }
    # cudnn settings first (independent of matmul precision)
    torch.backends.cudnn.benchmark = policy["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = policy["cudnn_deterministic"]
    # Float32 matmul precision (PyTorch 1.12+) — BEFORE matmul.allow_tf32
    # because set_float32_matmul_precision('highest') RESETS matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision(policy["float32_matmul_precision"])
    except AttributeError:
        pass  # Older PyTorch versions
    # Matmul precision — AFTER set_float32_matmul_precision so it wins
    torch.backends.cuda.matmul.allow_tf32 = policy["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = policy["allow_tf32"]
    # Channels last is per-tensor, applied lazily by caller
    return policy


def get_current_precision() -> dict:
    """Read back current precision settings (for /metrics/before training)."""
    return {
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def reset_precision() -> None:
    """Reset to DEFAULT_POLICY (for unit-test fixtures)."""
    set_precision(**DEFAULT_POLICY)
