"""Wf-130 — SBDD training device defaults (GPU-first).

This module provides the canonical helpers every ``molmetal`` SBDD
training script should call before allocating a single tensor:

* :func:`resolve_default_device` — pick ``cuda`` if available, else CPU
  with an informative (not silent) warning.  Honours
  ``--cuda-visible-devices`` CLI flag and :envvar:`CUDA_VISIBLE_DEVICES`
  env var.  When neither is set, defaults to ``"1"`` — the RTX 5090 in
  the standard kanzi dual-GPU box (GPU 0 is the ``vlm`` venv process).
* :func:`setup_training_device_env` — apply CUDA visibility + log the
  resolved device at INFO level.  Idempotent: re-running it is a no-op
  apart from the message.

Why this module exists (Wf-130 honest-framing)
----------------------------------------------
Prior to Wf-130 the training scripts had three failure modes:

1. **Default-to-CPU.**  ``device = torch.device("cpu")`` in a function
   signature meant a CUDA-visible cluster was never picked up unless the
   caller explicitly passed a CUDA device.
2. **Silent fallback.**  ``device = torch.device("cuda" if
   torch.cuda.is_available() else "cpu")`` is fine in isolation but was
   not gated by a log line — silent CPU fallbacks cost us several
   hours of debug time during R19 retrain.
3. **GPU 0 contention.**  Without ``CUDA_VISIBLE_DEVICES=1`` set, the
   training scripts collided with the VLM venv on GPU 0 of the kanzi
   host, dropping effective throughput by ~6× (the VLM process holds
   the SM scheduler busy even when it is idle).

Design rules
------------
1. **GPU-first by default.**  If a CUDA device is allocatable, use it.
2. **No silent fallbacks.**  Every call to :func:`resolve_default_device`
   emits one log line — either "using cuda:0 / cuda:1 …" or "no GPU
   detected, falling back to CPU — this will be slow."
3. **GPU 1 by default on kanzi.**  ``DEFAULT_CUDA_VISIBLE_DEVICES = "1"``
   picks the RTX 5090.  Override with ``--cuda-visible-devices=0`` or
   ``CUDA_VISIBLE_DEVICES=0,1`` if you really mean GPU 0.

Verify
------
::

    uv run --no-sync python -c \
        "from molmetal.molmetal_lam.sbdd_env.training_utils import (
            resolve_default_device, setup_training_device_env,
        ); print(resolve_default_device()); print(
            setup_training_device_env())"
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

# IMPORTANT: do NOT ``import torch`` at module load time.  CUDA only
# reads ``CUDA_VISIBLE_DEVICES`` once at CUDA init — so the env var
# must be applied *before* torch initialises CUDA.  Callers that want
# to control GPU visibility must therefore import this module *before*
# importing torch (or any submodule that imports torch).  All helpers
# in this module do ``import torch`` lazily, inside the function body,
# so a script can ``from molmetal.molmetal_lam.sbdd_env.training_utils
# import apply_default_cuda_visible_devices; apply_default_cuda_visible
# _devices(override="0")`` *before* any other torch-using import and
# get the expected GPU selection.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default value for ``CUDA_VISIBLE_DEVICES`` when neither the env var
#: nor the CLI flag overrides it.  ``"1"`` = second GPU = RTX 5090 on
#: the standard kanzi dual-GPU box (historically GPU 0 was reserved for
#: the ``vlm`` venv process).
#:
#: Wf-A9 (2026-10-03): the launch path no longer relies on this fallback
#: for retrains — ``scripts/start.sh`` exports ``CUDA_VISIBLE_DEVICES``
#: from the yaml ``compute.gpu`` field, so the yaml decides the device.
#: The fallback is kept unchanged for direct/standalone invocations that
#: have their own tests pinning ``"1"``.
DEFAULT_CUDA_VISIBLE_DEVICES: str = "1"


def ensure_cuda_visible_devices_default() -> str:
    """Set ``CUDA_VISIBLE_DEVICES`` to :data:`DEFAULT_CUDA_VISIBLE_DEVICES`
    IF the env var is not already set — and only via ``os.environ``.

    This is the **module-top safe** form of
    :func:`apply_default_cuda_visible_devices`: it does not import
    torch, does not log, and is a no-op when the env var is already
    set.  Use it at the very top of training scripts (BEFORE
    ``import torch``) so the CUDA init picks up the right GPU.

    Example
    -------
    ::

        # At the TOP of the script, BEFORE any torch import:
        from molmetal.molmetal_lam.sbdd_env.training_utils import (
            ensure_cuda_visible_devices_default,
        )
        ensure_cuda_visible_devices_default()  # → CUDA_VISIBLE_DEVICES='1'

        import torch  # ← CUDA init picks up '1' now

    Returns
    -------
    str
        The value applied (or already present) in
        :envvar:`CUDA_VISIBLE_DEVICES`.
    """
    cur = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cur:
        return cur
    os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_CUDA_VISIBLE_DEVICES
    return DEFAULT_CUDA_VISIBLE_DEVICES

_LOGGER = logging.getLogger("molmetal.sbdd_env.training_utils")
if not _LOGGER.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    )
    _LOGGER.addHandler(h)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def apply_default_cuda_visible_devices(
    *,
    override: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Set ``CUDA_VISIBLE_DEVICES`` to a sensible default if unset.

    Precedence
    ----------
    1. ``override`` (CLI ``--cuda-visible-devices`` value) — wins. An
       empty-string ``""`` is treated as an explicit "force empty"
       override (used by wf385a3 SCAFFOLD wire-in gate to force CPU).
    2. ``os.environ['CUDA_VISIBLE_DEVICES']`` if already set (including
       empty-string "").
    3. :data:`DEFAULT_CUDA_VISIBLE_DEVICES` (``"1"``).

    Parameters
    ----------
    override :
        Explicit value from CLI.  ``None`` means "no override, fall
        through to env var / default"; ``""`` means "force empty
        CUDA_VISIBLE_DEVICES" (CPU-only mode for wf385a3 SCAFFOLD gate).
    logger :
        Logger for the "defaulting to GPU 1" message.  Defaults to
        the module-level :data:`_LOGGER`.

    Returns
    -------
    str
        The value actually applied to ``CUDA_VISIBLE_DEVICES``.  Useful
        for echoing back to the user in their log.
    """
    log = logger or _LOGGER
    chosen: Optional[str] = None
    # Wf-A9 FIX (2026-10-03) — "already set" must be tested with
    # ``is not None``, not truthiness.  The previous ``.strip()`` guard
    # read an explicitly-empty ``CUDA_VISIBLE_DEVICES=""`` as *unset* and
    # silently replaced it with the ``"1"`` default, contradicting this
    # function's own docstring (branch 2: "if already set (including
    # empty-string \"\")") and CUDA's own semantics, where ``""`` means
    # "no visible devices".  The wf385a3 SCAFFOLD wire-in gate relies on
    # exactly that — it spawns its 50-step CPU smoke with
    # ``CUDA_VISIBLE_DEVICES=""`` — so the regression surfaced there as
    # ``RuntimeError: No CUDA GPUs are available``.
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if override is not None:
        # Wf-415-A3 fix: explicit "" override forces CPU even when env
        # var is unset (default fallback would otherwise kick in).
        chosen = str(override).strip()
    elif env_value is not None:
        # Set — honour it verbatim ("" → force CPU, "0"/"1"/… → that GPU).
        chosen = env_value.strip()
    else:
        chosen = DEFAULT_CUDA_VISIBLE_DEVICES
        log.warning(
            "CUDA_VISIBLE_DEVICES not set — defaulting to %s. "
            "Override with --cuda-visible-devices=N or "
            "CUDA_VISIBLE_DEVICES=N before launching.",
            chosen,
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    return chosen


def detect_training_gpu() -> Dict[str, Any]:
    """Honest probe of the GPU the training process can actually use.

    Mirrors :func:`molmetal.utils.device.detect_active_gpu` but is
    scoped to this module so callers don't have to import from
    ``molmetal.utils`` (which is the ROCm-specific path).  Always safe
    to call; allocation failures are caught and reported as
    ``alloc_ok=False``.

    Keys
    ----
    cuda_visible_devices : str
        The value of :envvar:`CUDA_VISIBLE_DEVICES` after
        :func:`apply_default_cuda_visible_devices` ran (or before, if
        this probe runs first).
    torch_reports_cuda   : bool
        ``torch.cuda.is_available()`` after the env var is applied.
    device_count         : int
        ``torch.cuda.device_count()``.
    device_name          : str
        ``torch.cuda.get_device_name(0)`` or ``"N/A"``.
    alloc_ok             : bool
        A 1-element tensor allocates on ``cuda:0`` without error.
    selected_device      : str
        ``"cuda:0"`` if alloc_ok else ``"cpu"``.
    reason               : str
        Human-readable verdict for log lines.
    """
    import torch  # noqa: WPS433 — lazy import; see module docstring
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        cuda_visible = bool(torch.cuda.is_available())
    except Exception:
        cuda_visible = False
    try:
        device_count = int(torch.cuda.device_count()) if cuda_visible else 0
    except Exception:
        device_count = 0
        cuda_visible = False
    try:
        device_name = (
            str(torch.cuda.get_device_name(0)) if cuda_visible else "N/A"
        )
    except Exception:
        device_name = "N/A"
        cuda_visible = False

    alloc_ok = False
    if cuda_visible:
        try:
            t = torch.zeros(1, device="cuda:0")
            del t
            alloc_ok = True
        except Exception:
            alloc_ok = False

    if alloc_ok:
        reason = (
            f"cuda:0 allocatable on '{device_name}' "
            f"(CUDA_VISIBLE_DEVICES={cvd or '<unset>'})"
        )
        selected = "cuda:0"
    elif cuda_visible:
        reason = (
            f"torch sees {device_count} GPU(s) but cuda:0 allocation "
            f"failed (driver / CUDA init issue?). CUDA_VISIBLE_DEVICES="
            f"{cvd or '<unset>'}"
        )
        selected = "cpu"
    else:
        reason = (
            "no CUDA device visible to torch — CPU fallback "
            f"(CUDA_VISIBLE_DEVICES={cvd or '<unset>'})"
        )
        selected = "cpu"

    return {
        "cuda_visible_devices": cvd,
        "torch_reports_cuda": cuda_visible,
        "device_count": device_count,
        "device_name": device_name,
        "alloc_ok": alloc_ok,
        "selected_device": selected,
        "reason": reason,
    }


def resolve_default_device(
    *,
    cuda_visible_devices: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> torch.device:
    """Return the default training device — ``cuda:0`` if allocatable.

    Calls :func:`apply_default_cuda_visible_devices` first so the env
    var is set even if the caller forgot.  Then runs
    :func:`detect_training_gpu` and emits a single INFO log line.

    Parameters
    ----------
    cuda_visible_devices :
        Value to apply to ``CUDA_VISIBLE_DEVICES`` before probing.
        ``None`` or ``""`` keeps whatever is already in the environment
        (or applies :data:`DEFAULT_CUDA_VISIBLE_DEVICES` if unset).
    logger :
        Logger for the message.  Defaults to the module logger.

    Returns
    -------
    torch.device
        ``cuda:0`` if a CUDA device is allocatable, else ``cpu``.
        Never silently returns ``cuda`` when alloc fails — the message
        explicitly says "falling back to CPU".
    """
    log = logger or _LOGGER
    apply_default_cuda_visible_devices(override=cuda_visible_devices, logger=log)
    info = detect_training_gpu()
    log.info(
        "[training_utils] device=%s | %s",
        info["selected_device"],
        info["reason"],
    )
    import torch  # noqa: WPS433 — lazy import; see module docstring
    return torch.device(info["selected_device"])


def setup_training_device_env(
    *,
    cuda_visible_devices: Optional[str] = None,
    deterministic: bool = True,
    seed: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """One-call setup for the GPU environment of any training script.

    Combines :func:`apply_default_cuda_visible_devices` +
    :func:`detect_training_gpu` + torch seed + deterministic flags +
    the info dict returned for downstream tagging.

    Parameters
    ----------
    cuda_visible_devices :
        Override value applied via
        :func:`apply_default_cuda_visible_devices`.
    deterministic :
        When True (default), call
        :func:`torch.use_deterministic_algorithms` and set
        :envvar:`CUBLAS_WORKSPACE_CONFIG` so cuBLAS doesn't reject
        the request.
    seed :
        Seed for :func:`torch.manual_seed`.  Always set so the
        first-batch results are reproducible across reruns.
    logger :
        Logger to use for the setup messages.

    Returns
    -------
    dict
        Snapshot of what was actually applied: ``device``,
        ``cuda_visible_devices``, ``seed``, ``deterministic``, plus the
        ``detect_training_gpu()`` info dict.  Suitable for tagging
        a metrics JSON via :func:`molmetal.scripts._common_train_args
        .tag_metrics_payload`.
    """
    log = logger or _LOGGER
    apply_default_cuda_visible_devices(override=cuda_visible_devices, logger=log)
    info = detect_training_gpu()

    import torch  # noqa: WPS433 — lazy import; see module docstring
    torch.manual_seed(int(seed))

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover — older torch
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

    device = torch.device(info["selected_device"])
    log.info(
        "[training_utils] setup | device=%s seed=%d deterministic=%s "
        "| cuda_visible_devices=%s | %s",
        device,
        int(seed),
        bool(deterministic),
        info["cuda_visible_devices"],
        info["reason"],
    )

    return {
        "device": device,
        "cuda_visible_devices": info["cuda_visible_devices"],
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "gpu_info": info,
    }


__all__ = [
    "DEFAULT_CUDA_VISIBLE_DEVICES",
    "apply_default_cuda_visible_devices",
    "detect_training_gpu",
    "ensure_cuda_visible_devices_default",
    "resolve_default_device",
    "setup_training_device_env",
]
