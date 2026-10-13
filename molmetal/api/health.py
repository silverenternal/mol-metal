# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Liveness probe used by the FastAPI /health endpoint."""
from __future__ import annotations


def health_check() -> dict:
    return {"status": "ok", "service": "molmetal", "version": "end_of_training.ultracode9"}


__all__ = ["health_check"]