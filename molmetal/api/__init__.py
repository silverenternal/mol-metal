# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""API surface — END-A3 + END-A6 + END7 fix.

``torchscript_export`` (model → TorchScript), ``serve.py`` (FastAPI app
with /health + /decode), ``inference`` (load ckpt → sample → decode),
``health`` (liveness probe).

Honest-negative: ``torch.jit.script`` is expected to FAIL on EGNN + YuelBond +
BondOrderHead (Triton-fused ops + ROCm EGNN + complex control flow); the
eager fallback path is preserved and the failure modes are recorded for
paper §6 limitations.
"""
from __future__ import annotations

from molmetal.api.health import health_check
from molmetal.api.inference import InferenceEngine
from molmetal.api.torchscript_export import export_to_torchscript

# Lazy-import serve to avoid hard failure when fastapi is not on sys.path
# (HONEST-NEGATIVE: deployment surface is opt-in per cfg).
try:
    from molmetal.api.serve import create_app  # type: ignore  # noqa: F401
    _SERVE_OK = True
except ImportError as _e:
    create_app = None  # type: ignore
    _SERVE_OK = False

__all__ = ["create_app", "InferenceEngine", "export_to_torchscript", "health_check"]