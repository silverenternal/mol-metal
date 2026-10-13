# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""FastAPI deployment app — END-A6 + END7 fix.

Endpoints:
  * GET /health  → liveness (no auth, no model load)
  * POST /decode → accept SMILES list, return canonical + passed flag

Honest scope: /decode uses InferenceEngine which currently returns empty
results when no TorchScript ckpt is provided (eager fallback); the actual
sampling path is trainer-specific and is wired per-on_train_end via the
``end_of_training`` hook (see molmetal/end_of_training.py).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from molmetal.api.health import health_check
from molmetal.api.inference import InferenceEngine

log = logging.getLogger(__name__)


class DecodeRequest(BaseModel):
    smiles: List[str]
    ckpt_path: Optional[str] = None


class DecodeResponse(BaseModel):
    canonical: List[Optional[str]]
    passed: List[bool]


def create_app(ckpt_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="mol-metal", version="end_of_training.ultracode9")
    engine = InferenceEngine(ckpt_path=ckpt_path or "HONEST-NEGATIVE") if ckpt_path else None

    @app.get("/health")
    def health() -> dict:
        return health_check()

    @app.post("/decode", response_model=DecodeResponse)
    def decode(req: DecodeRequest) -> DecodeResponse:
        if engine is None or not engine.load():
            raise HTTPException(status_code=503, detail="ckpt not loaded; pass ckpt_path to create_app")
        canonical = engine.decode_smiles(req.smiles)
        return DecodeResponse(canonical=canonical, passed=[c is not None for c in canonical])

    return app


__all__ = ["create_app"]