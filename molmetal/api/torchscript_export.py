# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""TorchScript export with eager-fallback honesty.

``export_to_torchscript(model, sample_input, save_path)`` tries
``torch.jit.script`` first; on failure it logs + returns ``None`` and
the caller is expected to keep the eager path live. This is consistent
with the AGENTS.md §5.2 mandate: a disabled feature is reported, not
swallowed.

Per design brief step E (HONEST-NEGATIVE): EGNN + YuelBond + BondOrderHead
have control-flow patterns which ``jit.script`` is expected to FAIL on
(≥60% failure rate). The failure modes are logged for paper §6 limitations.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Any

log = logging.getLogger(__name__)


def export_to_torchscript(
    model: Any,
    sample_input: Any,
    save_path: str,
    *,
    method: str = "trace",
) -> Optional[str]:
    """Try ``torch.jit.{trace,script}(model)``; return save path or None on failure."""
    try:
        import torch  # noqa: PLC0415
        if method == "script":
            scripted = torch.jit.script(model)
        else:
            scripted = torch.jit.trace(model, sample_input, strict=False)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.jit.save(scripted, save_path)
        log.info("TorchScript export OK: %s (method=%s)", save_path, method)
        return save_path
    except Exception as e:
        log.warning(
            "TorchScript export FAILED (method=%s): %s — keeping eager path. "
            "EGNN + YuelBond + BondOrderHead have complex control flow; "
            "see paper §6 limitations.",
            method,
            e,
        )
        return None


__all__ = ["export_to_torchscript"]