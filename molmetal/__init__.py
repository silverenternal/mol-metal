"""molmetal — Mol-Metal: precious-metal anticancer drug design framework.

Hexagonal architecture on top of MolFlow-Triton (Triton kernels on CUDA
Blackwell).  Three layers:

  domain/     — frozen dataclasses (Pocket, Molecule, Complex)
  ports/      — Protocol interfaces (MoleculeGenerator, DockingEngine,
                PropertyPredictor, ScoringFunction, DesignLoop)
  adapters/   — concrete implementations wrapping cloned SOTA code
                (facebookresearch/flow_matching, DiffDock, …)
  orchestration/ — closed-loop driver (generate → dock → score → refine)
  references/ — git-cloned SOTA repos (read-only, treated as data)

Environment bootstrap
---------------------
On import, :func:`_apply_process_env` sets two hygiene variables so the
import is quiet:

  * ``TOKENIZERS_PARALLELISM=false``
  * ``TRANSFORMERS_VERBOSITY=error``

Nothing else is set.  The ROCm/gfx1101 branch this function used to
carry (``PYTORCH_HIP_ALLOC_CONF`` / ``TORCH_BLAS_PREFER_HIPBLASLT``)
was removed on 2026-10-23 with the rest of the ROCm targeting: the
project has no AMD host, so those vars could only ever be skipped, and
the summary line said so on every import.
"""

from __future__ import annotations

import os
from typing import Dict

__version__ = "0.1.0"


#: Sentinel used by :func:`_apply_process_env` to record that it has
#: already run at least once during the current interpreter session.
_ENV_APPLIED: bool = False


def _apply_process_env(
    *,
    force: bool = False,
    quiet: bool = False,
) -> Dict[str, str]:
    """Apply the process-wide hygiene env vars.

    Parameters
    ----------
    force
        When ``True``, apply even if the sentinel is already set.
    quiet
        When ``True``, suppress the summary print.

    Returns
    -------
    dict
        Mapping ``{var_name: value}`` of vars this call set.  Vars
        already present with a different value are left untouched so a
        user override wins.

    Notes
    -----
    Idempotent: the sentinel short-circuits a second call unless
    ``force=True``.
    """
    global _ENV_APPLIED

    if _ENV_APPLIED and not force:
        return {}

    applied: Dict[str, str] = {}
    for name, value in (
        ("TOKENIZERS_PARALLELISM", "false"),
        ("TRANSFORMERS_VERBOSITY", "error"),
    ):
        if os.environ.get(name) != value:
            os.environ[name] = value
            applied[name] = value

    _ENV_APPLIED = True

    if applied and not quiet:
        kv = ", ".join(f"{k}={v}" for k, v in applied.items())
        print(f"[molmetal] applied env vars: {kv}")

    return applied


#: Run the bootstrap once at import time so downstream ``import torch``
#: users pick up the env.
_apply_process_env()


__all__ = [
    "__version__",
    "_apply_rocm_gfx1101_env",
    "_is_gfx1101_gpu",
    "ROCM_GFX1101_ENV_APPLIED_VARS",
]
