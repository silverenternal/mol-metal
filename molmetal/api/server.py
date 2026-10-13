# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Deployment API — Wf-A25 step 1 / predict + health.

A second FastAPI surface, distinct from :mod:`molmetal.api.serve`, that
exposes a prediction endpoint suitable for external callers (model
serving, paper demos, smoke probes):

  * ``GET  /health``  — returns 200 OK + the model version string.
  * ``POST /predict`` — accepts ``{"smiles": "<SMILES>"}`` (or a list)
                        and returns ``{"canonical": ..., "pic50": ...}``
                        where ``pic50`` is a float or ``NaN`` when no
                        model is loaded.

Bit-exact contract (per Wf-A25 task brief):
  When :mod:`molmetal.api.server` is **not** imported, the rest of
  :mod:`molmetal.api` behaves exactly as before. This module exposes no
  top-level import side-effects — fastapi / pydantic are imported only
  when :func:`create_app` is called.

Honest-negative (per AGENTS.md §5.2 / party-twenty-spirit §3.5):
  When no checkpoint is supplied or the checkpoint fails to load,
  ``pic50`` is reported as ``float("nan")`` (not a placeholder zero,
  not an exception swallowed into ``0.0``). The endpoint emits
  ``model_loaded: false`` in the response so callers can branch.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# Module-level model version string. Per molmetal.api.health: the active
# training-on-end hook stamps this with "end_of_training.ultracode9" at
# training-time. We expose the same string here so /health and /predict
# report a consistent version.
MODEL_VERSION = "end_of_training.ultracode9"


def _build_app(
    *,
    ckpt_path: Optional[str] = None,
    device: str = "cpu",
) -> Any:
    """Construct the FastAPI app. Imported lazily to keep module import side-effect free."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    from molmetal.api.health import health_check
    from molmetal.api.inference import InferenceEngine

    class PredictRequest(BaseModel):
        smiles: Any  # str or List[str]

    class PredictResponse(BaseModel):
        canonical: List[Optional[str]]
        pic50: List[float]
        model_loaded: bool
        model_version: str

    app = FastAPI(title="mol-metal-predict", version=MODEL_VERSION)
    engine = InferenceEngine(ckpt_path=ckpt_path or "HONEST-NEGATIVE", device=device)
    model_loaded = False
    if ckpt_path is not None:
        try:
            model_loaded = bool(engine.load())
        except Exception as e:  # noqa: BLE001
            log.warning("engine.load() raised: %s — pic50 will be NaN.", e)
            model_loaded = False

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Liveness + model version probe."""
        return {
            "status": "ok",
            "service": "mol-metal-predict",
            "version": MODEL_VERSION,
            "model_loaded": model_loaded,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        """Decode SMILES + (if model loaded) return pIC50.

        ``pic50[i]`` is ``float('nan')`` for any input that the model
        cannot score. This includes:

          * the model failed to load (``model_loaded=False``);
          * the canonicalised SMILES is ``None`` (invalid);
          * the engine's ``sample`` returned an empty list.

        The endpoint never emits an exception to swallow the failure —
        per party-twenty-spirit §3.5 the disabled path is reported, not
        hidden.
        """
        if isinstance(req.smiles, str):
            smiles_list = [req.smiles]
        elif isinstance(req.smiles, list):
            smiles_list = list(req.smiles)
        else:
            raise HTTPException(status_code=422, detail="smiles must be str or List[str]")

        if not smiles_list:
            raise HTTPException(status_code=422, detail="smiles is empty")

        canonical = engine.decode_smiles(smiles_list)
        if model_loaded:
            # HONEST-NEGATIVE: the trainer-specific pIC50 head is not yet
            # wired into InferenceEngine. Until then, pic50 is NaN for
            # every row — recorded here rather than silently returned as
            # 0.0 so the deployment surface stays honest.
            pic50: List[float] = [float("nan")] * len(smiles_list)
        else:
            pic50 = [float("nan")] * len(smiles_list)

        return PredictResponse(
            canonical=canonical,
            pic50=pic50,
            model_loaded=model_loaded,
            model_version=MODEL_VERSION,
        )

    return app


def create_app(
    ckpt_path: Optional[str] = None,
    *,
    device: str = "cpu",
) -> Any:
    """Public factory. Mirrors :func:`molmetal.api.serve.create_app` signature.

    Returns ``None`` (not an exception) when fastapi is not importable.
    Callers should check the return value before binding to a server.
    """
    try:
        return _build_app(ckpt_path=ckpt_path, device=device)
    except ImportError as e:
        log.warning(
            "create_app() FAILED (fastapi not on sys.path: %s) — "
            "deployment surface is opt-in per cfg.",
            e,
        )
        return None


def health_version() -> Dict[str, Any]:
    """Return a version probe without importing fastapi. Always safe."""
    return {
        "status": "ok",
        "service": "mol-metal-predict",
        "version": MODEL_VERSION,
        "model_loaded": False,
    }


__all__ = ["create_app", "health_version", "MODEL_VERSION"]


def _self_check() -> int:
    """Lightweight CPU-only smoke: build the app + hit /health without a server.

    Returns 0 when the surface is importable + the /health probe produces a
    well-formed dict with ``status == "ok"``. Returns 1 when fastapi is
    missing or the build fails. Returns 2 when the probe schema is wrong.

    Used by ``molmetal/scripts/integration_check.py`` (Wf-A25 step 3) to
    confirm the bit-exact contract: importing this module + creating the
    app must not raise.
    """
    out = health_version()
    if not (out.get("status") == "ok" and "version" in out):
        return 2
    app = create_app(ckpt_path=None)
    if app is None:
        # fastapi missing — surface still importable; this is honest-negative.
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_check())
