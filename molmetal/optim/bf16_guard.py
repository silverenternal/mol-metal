"""BF16 / AMP hardware-compatibility guard (Wf-A11 / optimizer wave).

Single source of truth for "is BF16 autocast safe on the current GPU?".
The CFM trainer defaults to BF16 autocast on sm_80+ GPUs; on older
hardware the autocast path silently downcasts but tensors lose precision
on losses that are summed across batches.

Why this lives in the optimizer package
---------------------------------------
The autocast decision is an *optimization* concern, not a data-loading one.
Putting the guard here means the CFM trainer's ``--bf16`` flag and the
``molmetal.adapters.flow_matching_lipman.amp`` module both share one
honest probe, and a config-side ``optimizer.bf16_auto_disable`` flag can
opt-out without changing the trainer.

Function contract
----------------
:func:`check_bf16_compatibility` returns a 2-tuple ``(compatible, reason)``:

  * ``compatible=True``  -> safe to enable BF16 autocast.
  * ``compatible=False`` -> caller MUST disable autocast (the trainer
    prints the reason and falls back to FP32). ``reason`` is a short
    string suitable for a ``warnings.warn(..., RuntimeWarning)`` payload.

The check is per-device, computed at call time (not module import time),
so multi-GPU hosts get one verdict per ``torch.cuda.device(idx)`` they
actually use.  A CPU host returns ``(False, "cpu")`` — there is no
compute capability to probe and BF16 on CPU has been very slow historically.

Lit anchors
-----------
* NVIDIA Ampere Architecture Whitepaper 2020 — sm_80 (A100) is the
  first architecture where BF16 tensor cores are GA (sm_80 = full BF16;
  sm_86 is a refresh, sm_89 = Ada Lovelace, sm_90 = Hopper, sm_120 = Blackwell).
* PyTorch docs, ``torch.cuda.get_device_capability`` — returns a tuple of
  ints.  We treat the first int (the major version) as the SM arch.
"""

from __future__ import annotations

from typing import Tuple

import torch


def check_bf16_compatibility(min_compute_capability: float = 8.0) -> Tuple[bool, str]:
    """Probe the current GPU for BF16 autocast safety.

    Parameters
    ----------
    min_compute_capability : float, default 8.0
        Lowest acceptable major SM version.  The Ampere whitepaper marks
        sm_80 (A100) as the first architecture with full BF16 tensor core
        support, so 8.0 is the project-wide conservative floor.

    Returns
    -------
    (compatible, reason) : (bool, str)
        ``compatible`` is True iff a CUDA device is available AND its
        reported compute capability is ``>= min_compute_capability``.
        ``reason`` is "" when compatible, otherwise a short message like
        ``"compute_capability=7.5 < min_compute_capability=8.0"`` or
        ``"cpu"``.

    Notes
    -----
    This function performs no allocation — it only reads
    :func:`torch.cuda.get_device_capability`.  Callers may invoke it at
    every step (e.g. inside a hot reload path) without VRAM pressure.
    """
    if not torch.cuda.is_available():
        return False, "cpu"

    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception as exc:  # noqa: BLE001 — guarded probe, never raise
        return False, f"probe_failed:{type(exc).__name__}"

    if float(major) < float(min_compute_capability):
        return (
            False,
            f"compute_capability={float(major):.1f} < "
            f"min_compute_capability={float(min_compute_capability):.1f}",
        )
    return True, ""


__all__ = ["check_bf16_compatibility"]