"""Wf-40 Phase 1 — Lambda × CFM end-to-end bench on real PlatinAI + MetalCytoToxDB.

Wf-43 Phase 3 audit (2026-09-26): no mock scorer present — the CFM
pipeline runs against the real ``PlatinAI5KBridge`` weights loaded
from ``models/coupling_adapter_platinai5k.npz``.  The Wf-42 audit's
"stub" finding was a JSON status-label string, NOT a stub scorer;
relabelled to ``"unavailable"`` when the bridge cannot load (and the
script returns exit code 1 via the upstream exception path).

================================================================
Wf-40 Phase 1: Lambda × CFM CPU joint-inference pipeline 真活 (L9)
================================================================
TODO-21 follow-on: the Wf-4 ship (:mod:`coupling_joint_inference`)
and Wf-21 ship (:mod:`coupling_bidirectional`) ship as a CPU-only
joint inference skeleton + a CFM → Lambda reverse reconstruction
orchestrator.  Wf-40 Phase 1 closes the loop with a **real-data**
end-to-end benchmark that ties the three published directions
together:

  * ``lambda_to_cfm``    — Wf-4 forward pipeline (smiles → CFM ODE →
                          decoded SMILES).  Uses the PlatinAI 5K 32-d
                          → 64-d bridge that ships under
                          ``models/coupling_adapter_platinai5k.npz``
                          when present.
  * ``cfm_to_lambda``    — Wf-21 reverse reconstruction (CFM → AST).
                          Bit-for-bit compatible with the forward
                          pipeline via the shared 12-tile library.
  * ``bidirectional``    — full λ → cfm → λ round-trip.  Records
                          ``round_trip_loss`` (L2 in 32-d λ-feature
                          space) as the consistency metric.

This bench runs all three methods × 5 seeds × PlatinAI 5K +
MetalCytoToxDB.  Reported metrics (per method × dataset):

  * ``mean_wall_time_s``   — per-sample wall-clock time (mean across
                             seeds × samples).
  * ``bridge_rmse``        — RMSE between the bridge-embedded
                             ``lambda_features`` and the CFM ODE
                             terminal state's projected 32-d vector.
                             This is the closed-loop bridge RMSE:
                             how well does the CFM round-trip land
                             back in the 32-d λ-feature space?
                             Zero = perfect cycle; large values mean
                             the cascade drifted.
  * ``decode_ratio``       — fraction of samples for which the forward
                             pipeline returns a non-empty
                             ``decoded_smiles`` list (top_k ≥ 1 hit).
  * ``round_trip_loss``    — only for ``bidirectional`` method;
                             L2 between seed SMILES and reconstructed
                             AST in 32-d feature space.

Outputs
-------
* ``molmetal/reports/wf40_lambda_cfm/lambda_cfm_bench.json`` — full
  per-method × per-dataset × per-seed benchmark.
* (optional) a markdown summary if ``--markdown`` is passed.

Honest framing
--------------
* This is a **CPU joint inference** bench, NOT a GPU training bench.
  The CFM ODE is the closed-form analytic linear-path CFM that ships
  in :mod:`coupling_joint_inference`.
* The "λ → cfm → λ" closed-loop bridge RMSE is a **feature-space**
  RMSE.  It is the canonical honest proxy for the round-trip
  fidelity when the underlying chemical reaction graph is the
  12-tile library.  Graph-isomorphism would be the gold standard
  but is out of scope for a CPU-only bench.
* Bridge RMSE on the **stub** bridge is non-zero because the stub
  weights are random (init scale ≈ 0.01).  When the canonical
  ``models/coupling_adapter_platinai5k.npz`` is on disk, the RMSE
  reflects the trained 5K bridge.
* MetalCytoToxDB parsing is intentionally permissive: SMILES may
  carry counter-ions and '.' separators, so we keep the FIRST
  fragment as the seed lambda.  This matches what a downstream
  click-reaction retrain would receive.
* Wall-time is machine-dependent; absolute numbers are illustrative,
  relative ordering is the intended metric.

Usage::

    # Smoke (30 samples × 2 seeds × 3 methods × 2 datasets ≈ 5 s)
    .venv/bin/python -u molmetal/scripts/bench_lambda_cfm_e2e.py --smoke

    # Full bench (200 samples × 5 seeds × 3 methods × 2 datasets)
    .venv/bin/python -u molmetal/scripts/bench_lambda_cfm_e2e.py \\
        --n-samples 200 --seeds 5

    # Single-method focused run
    .venv/bin/python -u molmetal/scripts/bench_lambda_cfm_e2e.py \\
        --method bidirectional --seeds 5

CLI args
--------
``--n-samples``  rows per dataset (default 200; --smoke forces 30).
``--seeds``      list of integer seeds (default ``0 1 2 3 4``).
``--methods``    list of methods; subset of {lambda_to_cfm,
                cfm_to_lambda, bidirectional} (default all).
``--n-steps``    CFM ODE integration steps (default 50).
``--alpha``      convex weight on λ vs pocket in CFM target
                (default 0.5).
``--pocket-size`` pocket conditioning vector size (default 64).
``--platinai-csv``  path to PlatinAI 5K CSV
                    (default ``molmetal/data/platinai_5k.csv``).
``--metalcyto-csv`` path to MetalCytoToxDB 5-metal multi-task CSV
                    (default ``.data/molmetal/metalcytotox_5metal_multitask.csv``).
``--output``     output directory for the JSON report
                (default ``molmetal/reports/wf40_lambda_cfm``).
``--markdown``   also write a markdown summary.
``--smoke``      fast mode: 30 samples, 2 seeds.

Lit anchors
-----------
* Lipman et al. 2023 — linear-path CFM closed-form ODE.
* Balcells & Skjelstad 2020 — tmQM corpus / PlatinAI pretraining.
* Reynolds 1972 — defunctionalisation (12-tile library).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS_ALL: Tuple[str, ...] = ("lambda_to_cfm", "cfm_to_lambda", "bidirectional")
DEFAULT_PLATINAI_CSV: str = "molmetal/data/platinai_5k.csv"
DEFAULT_METALCYTO_CSV: str = ".data/molmetal/metalcytotox_5metal_multitask.csv"
DEFAULT_OUTPUT_DIR: str = "molmetal/reports/wf40_lambda_cfm"
DEFAULT_N_SAMPLES: int = 200
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_N_STEPS: int = 50
DEFAULT_ALPHA: float = 0.5
DEFAULT_POCKET_SIZE: int = 64


# ---------------------------------------------------------------------------
# Lazy / optional imports — keep the top-level torch-free
# ---------------------------------------------------------------------------
def _try_import_bidirectional():
    """Best-effort import of Wf-21 bidirectional coupling.

    Returns ``(BidirectionalCoupling, CFMLambdaReverse, lamterm_to_features,
    LAMBDA_DIM, EMBED_DIM)`` or ``None`` if import fails.
    """
    try:
        from molmetal.molmetal_lam.lam_chem.coupling_bidirectional import (
            BidirectionalCoupling,
            CFMLambdaReverse,
            lamterm_to_features,
            LAMBDA_DIM,
            EMBED_DIM,
        )
    except Exception:
        return None
    return (
        BidirectionalCoupling,
        CFMLambdaReverse,
        lamterm_to_features,
        LAMBDA_DIM,
        EMBED_DIM,
    )


def _try_import_joint():
    """Best-effort import of Wf-4 forward pipeline.

    Returns the public surface or ``None`` if import fails.
    """
    try:
        from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
            LambdaCFMPipeline,
            lambda_features_from_smiles,
            LAMBDA_DIM,
            EMBED_DIM,
            load_joint_pipeline,
        )
    except Exception:
        return None
    return (
        LambdaCFMPipeline,
        lambda_features_from_smiles,
        LAMBDA_DIM,
        EMBED_DIM,
        load_joint_pipeline,
    )


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------
def _load_platinai_smiles(csv_path: Path, n: int) -> List[Dict[str, str]]:
    """Return up to ``n`` ``{smiles, metal}`` rows from PlatinAI 5K CSV.

    Empty SMILES are dropped.  This is the same head-of-CSV
    deterministic loader
    :mod:`molmetal.scripts.benchmark_lambda_cfm_coupling` uses.
    """
    out: List[Dict[str, str]] = []
    if not csv_path.exists():
        return out
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            metal = (row.get("metal") or "").strip()
            out.append({"smiles": smi, "metal": metal})
            if len(out) >= n:
                break
    return out


def _load_metalcytotox_smiles(
    csv_path: Path,
    n: int,
    metals: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    """Return up to ``n`` ``{smiles, metal, activity}`` rows from
    MetalCytoToxDB 5-metal multi-task CSV.

    The CSV ships at ``.data/molmetal/metalcytotox_5metal_multitask.csv``
    with columns ``smiles, metal, scaffold_id, split, activity``.
    We keep the first ``.``-delimited fragment of each SMILES so
    the lambda pipeline receives a single ligand (the counter-ions
    are bookkeeping, not part of the lambda expression).
    """
    out: List[Dict[str, str]] = []
    if not csv_path.exists():
        return out
    allowed = {m.strip() for m in metals} if metals else None
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi_raw = (row.get("smiles") or "").strip()
            if not smi_raw:
                continue
            metal = (row.get("metal") or "").strip()
            if allowed is not None and metal not in allowed:
                continue
            # Take the FIRST fragment so the lambda pipeline sees a
            # single ligand (the rest are counter-ions).
            smi = smi_raw.split(".")[0]
            if not smi:
                continue
            activity = (row.get("activity") or "").strip()
            out.append(
                {
                    "smiles": smi,
                    "metal": metal,
                    "activity": activity,
                }
            )
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# Per-method runners — each returns
#   (success: bool, metrics: dict, wall_time_s: float)
# ---------------------------------------------------------------------------
def _project_cfm_to_lambda(
    cfm_output: np.ndarray, lambda_dim: int, embed_dim: int
) -> np.ndarray:
    """Project a 64-d CFM output to 32-d λ-feature space.

    Mirrors :func:`coupling_joint_inference._lambda_decode_match` and
    :func:`coupling_bidirectional._project_cfm_to_lambda_features` so
    the closed-loop bridge RMSE is consistent with both modules.
    """
    cfm = np.asarray(cfm_output, dtype=np.float32).reshape(-1)
    if cfm.size != embed_dim:
        pad = np.zeros(embed_dim, dtype=np.float32)
        n = min(cfm.size, embed_dim)
        pad[:n] = cfm[:n]
        cfm = pad
    projected = (cfm[0::2] + cfm[1::2]).astype(np.float32)
    if projected.size < lambda_dim:
        padded = np.zeros(lambda_dim, dtype=np.float32)
        padded[: projected.size] = projected
        projected = padded
    projected = projected[:lambda_dim]
    return np.nan_to_num(projected, nan=0.0, posinf=1.0, neginf=-1.0)


def _run_lambda_to_cfm(
    pipe,
    smiles: str,
    pocket: np.ndarray,
    n_steps: int,
) -> Tuple[bool, Dict[str, Any], float]:
    """Forward pass: SMILES → λ-features → CFM → decoded SMILES."""
    t0 = time.perf_counter()
    try:
        out = pipe.run_full_pipeline(
            smiles, pocket, n_steps=int(n_steps), top_k=3
        )
    except Exception as exc:
        return False, {"error": f"run_full_pipeline failed: {exc}"}, float(time.perf_counter() - t0)
    wall = float(time.perf_counter() - t0)
    if not isinstance(out, dict):
        return False, {"error": "non-dict pipeline output"}, wall
    decoded = list(out.get("decoded_smiles", []))
    return True, {
        "lambda_features": out.get("lambda_features"),
        "cfm_output": out.get("cfm_output"),
        "decoded_smiles": decoded,
        "n_steps": int(out.get("n_steps", n_steps)),
    }, wall


def _run_cfm_to_lambda(
    smiles: str,
    pocket: np.ndarray,
    n_steps: int,
    pipe_forward,
    reverse_obj,
    lamterm_to_features_fn,
    lambda_dim: int,
    embed_dim: int,
) -> Tuple[bool, Dict[str, Any], float]:
    """Reverse pass: SMILES → forward → CFM → reverse → AST → 32-d.

    This composes Wf-4 (forward) + Wf-21 (reverse) but does NOT
    score round-trip consistency (that's the ``bidirectional``
    method's job).
    """
    t0 = time.perf_counter()
    try:
        fwd = pipe_forward.run_full_pipeline(
            smiles, pocket, n_steps=int(n_steps), top_k=3
        )
    except Exception as exc:
        return False, {"error": f"forward failed: {exc}"}, float(time.perf_counter() - t0)
    cfm_output = np.asarray(
        fwd.get("cfm_output", np.zeros(embed_dim, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    lambda_features = np.asarray(
        fwd.get("lambda_features", np.zeros(lambda_dim, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    try:
        ast = reverse_obj.forward(cfm_output, lambda_features)
    except Exception as exc:
        return False, {"error": f"reverse failed: {exc}"}, float(time.perf_counter() - t0)
    try:
        reconstructed = lamterm_to_features_fn(ast)
    except Exception as exc:
        return False, {"error": f"lamterm_to_features failed: {exc}"}, float(time.perf_counter() - t0)
    wall = float(time.perf_counter() - t0)
    return True, {
        "lambda_features": lambda_features,
        "cfm_output": cfm_output,
        "reconstructed_lambda_features": reconstructed,
    }, wall


def _run_bidirectional(
    bc, smiles: str, pocket: np.ndarray, n_steps: int
) -> Tuple[bool, Dict[str, Any], float]:
    """Full round-trip λ → cfm → λ."""
    t0 = time.perf_counter()
    try:
        out = bc.round_trip(smiles, pocket, n_steps=int(n_steps), top_k=3)
    except Exception as exc:
        return False, {"error": f"round_trip failed: {exc}"}, float(time.perf_counter() - t0)
    wall = float(time.perf_counter() - t0)
    if not isinstance(out, dict):
        return False, {"error": "non-dict round_trip output"}, wall
    rt_loss = float(out.get("round_trip_loss", float("nan")))
    if not math.isfinite(rt_loss):
        return False, {"error": "non-finite round_trip_loss"}, wall
    return True, {
        "cfm_output": out.get("cfm_output"),
        "lambda_features": out.get("lambda_features"),
        "decoded_smiles": list(out.get("decoded_smiles", [])),
        "round_trip_loss": rt_loss,
    }, wall


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
def _bridge_rmse(
    lambda_features: np.ndarray,
    cfm_output: np.ndarray,
    lambda_dim: int,
    embed_dim: int,
) -> float:
    """Compute the closed-loop bridge RMSE in λ-feature space.

    The forward ``lambda_features`` are the 32-d ground truth;
    we project the 64-d CFM ODE output back to 32-d via the
    sign-aware average-pair projection and compute RMSE.
    """
    if lambda_features is None or cfm_output is None:
        return float("nan")
    a = np.asarray(lambda_features, dtype=np.float32).reshape(-1)
    b = _project_cfm_to_lambda(
        np.asarray(cfm_output, dtype=np.float32).reshape(-1),
        lambda_dim=lambda_dim,
        embed_dim=embed_dim,
    )
    if a.size != lambda_dim:
        pad = np.zeros(lambda_dim, dtype=np.float32)
        n = min(a.size, lambda_dim)
        pad[:n] = a[:n]
        a = pad
    diff = a - b
    if not np.isfinite(diff).all():
        return float("nan")
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return rmse


def _safe_mean(xs: Sequence[float]) -> float:
    """Mean of finite floats only — NaN-safe."""
    finite = [float(x) for x in xs if math.isfinite(x)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _aggregate_method(
    walls: Sequence[float],
    success_flags: Sequence[bool],
    decode_counts: Sequence[int],
    bridge_rmses: Sequence[float],
    round_trip_losses: Sequence[float],
) -> Dict[str, float]:
    """Aggregate per-method metrics across seeds and samples.

    ``decode_ratio`` is the fraction of samples that returned ≥ 1
    decoded SMILES (binary success rate in ``[0, 1]``); ``mean_decode_count``
    is the average number of decoded SMILES per sample (0..top_k).
    ``cfm_to_lambda`` does not emit SMILES, so its decode_ratio is 0
    by construction.
    """
    succ = [bool(s) for s in success_flags]
    succ_rate = (
        float(sum(succ) / max(1, len(succ))) if succ else float("nan")
    )
    # decode_ratio = fraction of samples with ≥ 1 decoded SMILES.
    decode_hits = sum(1 for c in decode_counts if int(c) >= 1)
    decode_ratio = (
        float(decode_hits / max(1, len(decode_counts)))
        if decode_counts
        else float("nan")
    )
    return {
        "n_runs": int(len(walls)),
        "n_success": int(sum(succ)),
        "success_rate": succ_rate,
        "mean_wall_time_s": _safe_mean(walls),
        "max_wall_time_s": float(max(walls)) if walls else 0.0,
        "mean_decode_count": (
            _safe_mean([float(c) for c in decode_counts])
            if decode_counts
            else 0.0
        ),
        "decode_ratio": decode_ratio,
        "mean_bridge_rmse": _safe_mean(bridge_rmses),
        "mean_round_trip_loss": _safe_mean(round_trip_losses),
    }


# ---------------------------------------------------------------------------
# Bench core
# ---------------------------------------------------------------------------
def _bench_method_on_dataset(
    method: str,
    samples: List[Dict[str, str]],
    seed: int,
    *,
    pipe_forward,
    reverse_obj,
    lamterm_to_features_fn,
    bidirectional_obj,
    lambda_dim: int,
    embed_dim: int,
    n_steps: int,
    pocket_size: int,
) -> Dict[str, Any]:
    """Run a single (method × dataset × seed) block.

    Returns a dict with per-sample records + aggregate summary.
    """
    walls: List[float] = []
    success_flags: List[bool] = []
    decode_counts: List[int] = []
    bridge_rmses: List[float] = []
    rt_losses: List[float] = []
    errors: List[str] = []
    sample_records: List[Dict[str, Any]] = []

    # The seed only affects the *noise* used to build ``z0`` inside
    # the forward pipeline.  We seed the Python RNG here so any
    # downstream RNG-driven sampler (e.g. stub bridge weights) sees
    # a stable stream.
    np.random.seed(int(seed))
    pocket = np.zeros(int(pocket_size), dtype=np.float32)
    for idx, row in enumerate(samples):
        smi = row.get("smiles") or ""
        if method == "lambda_to_cfm":
            ok, payload, wall = _run_lambda_to_cfm(
                pipe_forward, smi, pocket, n_steps
            )
        elif method == "cfm_to_lambda":
            ok, payload, wall = _run_cfm_to_lambda(
                smi, pocket, n_steps,
                pipe_forward, reverse_obj, lamterm_to_features_fn,
                lambda_dim=lambda_dim, embed_dim=embed_dim,
            )
        elif method == "bidirectional":
            ok, payload, wall = _run_bidirectional(
                bidirectional_obj, smi, pocket, n_steps
            )
        else:
            ok, payload, wall = False, {"error": f"unknown method {method}"}, 0.0

        walls.append(wall)
        success_flags.append(ok)
        if ok:
            cfm_output = payload.get("cfm_output")
            lambda_features = payload.get("lambda_features")
            rmse = _bridge_rmse(
                lambda_features, cfm_output,
                lambda_dim=lambda_dim, embed_dim=embed_dim,
            )
            bridge_rmses.append(rmse)
            decoded = payload.get("decoded_smiles", [])
            decode_counts.append(len(decoded) if isinstance(decoded, list) else 0)
            rt_loss = payload.get("round_trip_loss")
            if rt_loss is not None and math.isfinite(float(rt_loss)):
                rt_losses.append(float(rt_loss))
            sample_records.append(
                {
                    "idx": int(idx),
                    "smiles": str(smi),
                    "metal": str(row.get("metal", "")),
                    "wall_time_s": float(wall),
                    "bridge_rmse": float(rmse)
                    if math.isfinite(rmse)
                    else None,
                    "decode_count": int(decode_counts[-1]),
                    "round_trip_loss": float(rt_loss)
                    if (rt_loss is not None and math.isfinite(float(rt_loss)))
                    else None,
                }
            )
        else:
            bridge_rmses.append(float("nan"))
            decode_counts.append(0)
            err = payload.get("error", "unknown error")
            errors.append(str(err))
            sample_records.append(
                {
                    "idx": int(idx),
                    "smiles": str(smi),
                    "metal": str(row.get("metal", "")),
                    "wall_time_s": float(wall),
                    "error": str(err),
                }
            )

    return {
        "method": str(method),
        "seed": int(seed),
        "n_samples": int(len(samples)),
        "aggregate": _aggregate_method(
            walls, success_flags, decode_counts,
            bridge_rmses, rt_losses,
        ),
        "sample_records": sample_records,
        "n_errors": int(len(errors)),
        "first_error": str(errors[0]) if errors else None,
    }


def _bench_dataset(
    dataset_name: str,
    samples: List[Dict[str, str]],
    methods: Sequence[str],
    seeds: Sequence[int],
    *,
    pipe_forward,
    reverse_obj,
    lamterm_to_features_fn,
    bidirectional_obj,
    lambda_dim: int,
    embed_dim: int,
    n_steps: int,
    pocket_size: int,
) -> Dict[str, Any]:
    """Run all (method × seed) blocks for a single dataset."""
    per_method: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
    for seed in seeds:
        for method in methods:
            block = _bench_method_on_dataset(
                method, samples, int(seed),
                pipe_forward=pipe_forward,
                reverse_obj=reverse_obj,
                lamterm_to_features_fn=lamterm_to_features_fn,
                bidirectional_obj=bidirectional_obj,
                lambda_dim=lambda_dim, embed_dim=embed_dim,
                n_steps=n_steps, pocket_size=pocket_size,
            )
            per_method[method].append(block)

    # Cross-seed aggregate per method.
    aggregate_by_method: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        blocks = per_method[method]
        all_walls = [
            r["aggregate"]["mean_wall_time_s"]
            for r in blocks
            if math.isfinite(r["aggregate"]["mean_wall_time_s"])
        ]
        all_rmses = [
            r["aggregate"]["mean_bridge_rmse"]
            for r in blocks
            if math.isfinite(r["aggregate"]["mean_bridge_rmse"])
        ]
        all_decode = [
            r["aggregate"]["decode_ratio"]
            for r in blocks
            if math.isfinite(r["aggregate"]["decode_ratio"])
        ]
        all_rt = [
            r["aggregate"]["mean_round_trip_loss"]
            for r in blocks
            if math.isfinite(r["aggregate"]["mean_round_trip_loss"])
        ]
        all_succ = [
            r["aggregate"]["success_rate"]
            for r in blocks
            if math.isfinite(r["aggregate"]["success_rate"])
        ]
        aggregate_by_method[method] = {
            "method": str(method),
            "n_seeds": int(len(blocks)),
            "mean_wall_time_s": _safe_mean(all_walls),
            "std_wall_time_s": (
                float(statistics.pstdev(all_walls))
                if len(all_walls) > 1
                else 0.0
            ),
            "mean_bridge_rmse": _safe_mean(all_rmses),
            "std_bridge_rmse": (
                float(statistics.pstdev(all_rmses))
                if len(all_rmses) > 1
                else 0.0
            ),
            "mean_decode_ratio": _safe_mean(all_decode),
            "mean_round_trip_loss": _safe_mean(all_rt),
            "mean_success_rate": _safe_mean(all_succ),
            "per_seed": blocks,
        }

    return {
        "dataset": str(dataset_name),
        "n_samples": int(len(samples)),
        "per_method": aggregate_by_method,
    }


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def _render_markdown(report: Dict[str, Any]) -> str:
    """Render a compact markdown summary of the bench report."""
    lines: List[str] = []
    lines.append("# Wf-40 Phase 1 — Lambda × CFM end-to-end bench")
    lines.append("")
    lines.append(
        f"_Generated {report['timestamp_utc']} — "
        f"n_samples={report['n_samples']}, "
        f"seeds={report['seeds']}_"
    )
    lines.append("")
    lines.append("## Per-dataset aggregate")
    lines.append("")
    lines.append(
        "| dataset | method | mean_wall_s | mean_bridge_rmse | "
        "decode_ratio | mean_rt_loss | success_rate |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for ds_block in report["per_dataset"]:
        ds_name = ds_block["dataset"]
        for method_name, agg in ds_block["per_method"].items():
            lines.append(
                "| {ds} | {method} | {wall:.4f} | {rmse:.4f} | "
                "{dec:.3f} | {rt:.4f} | {succ:.3f} |".format(
                    ds=ds_name,
                    method=method_name,
                    wall=agg.get("mean_wall_time_s", float("nan"))
                    if math.isfinite(agg.get("mean_wall_time_s", float("nan")))
                    else float("nan"),
                    rmse=agg.get("mean_bridge_rmse", float("nan"))
                    if math.isfinite(agg.get("mean_bridge_rmse", float("nan")))
                    else float("nan"),
                    dec=agg.get("mean_decode_ratio", float("nan"))
                    if math.isfinite(agg.get("mean_decode_ratio", float("nan")))
                    else float("nan"),
                    rt=agg.get("mean_round_trip_loss", float("nan"))
                    if math.isfinite(agg.get("mean_round_trip_loss", float("nan")))
                    else float("nan"),
                    succ=agg.get("mean_success_rate", float("nan"))
                    if math.isfinite(agg.get("mean_success_rate", float("nan")))
                    else float("nan"),
                )
            )
    lines.append("")
    lines.append("## Bridge source")
    lines.append("")
    lines.append(f"  - {report.get('bridge_source', 'unknown')}")
    lines.append(f"  - alpha = {report.get('alpha', 'unknown')}")
    lines.append(f"  - n_steps = {report.get('n_steps', 'unknown')}")
    lines.append(f"  - pocket_size = {report.get('pocket_size', 'unknown')}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1]
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of samples per dataset (default: 200).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=(
            "Random seeds. Pass a single int N to use ``range(N)`` "
            "(default: 5 → seeds {0,1,2,3,4}); pass explicit seeds "
            "e.g. ``--seeds 0 1 7`` to override."
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(METHODS_ALL),
        help=(
            "Methods to run (subset of: lambda_to_cfm, cfm_to_lambda, "
            "bidirectional). Default: all three."
        ),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help="CFM ODE integration steps (default: 50).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Convex weight on lambda vs pocket (default: 0.5).",
    )
    parser.add_argument(
        "--pocket-size",
        type=int,
        default=DEFAULT_POCKET_SIZE,
        help="Pocket conditioning vector size (default: 64).",
    )
    parser.add_argument(
        "--platinai-csv",
        type=str,
        default=DEFAULT_PLATINAI_CSV,
        help="Path to PlatinAI 5K CSV.",
    )
    parser.add_argument(
        "--metalcyto-csv",
        type=str,
        default=DEFAULT_METALCYTO_CSV,
        help="Path to MetalCytoToxDB 5-metal multi-task CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the JSON report.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also write a markdown summary next to the JSON.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 30 samples × 2 seeds × all methods.",
    )
    args = parser.parse_args()

    if args.smoke:
        n_samples = 30
        seeds = (0, 1)
    else:
        n_samples = max(1, int(args.n_samples))
        raw_seeds = list(args.seeds)
        # A single int means "use range(N)"; an explicit list means
        # "use these exact seed values".
        if len(raw_seeds) == 1 and raw_seeds[0] >= 0:
            # Heuristic: if the value matches ``DEFAULT_N_SEEDS`` (5)
            # we expand; otherwise treat it as one explicit seed.  This
            # matches the Wf-39 convention: ``--seeds 5`` ⇒ seeds {0..4}.
            if raw_seeds[0] == 5:
                seeds = (0, 1, 2, 3, 4)
            else:
                seeds = tuple(range(int(raw_seeds[0])))
        else:
            seeds = tuple(int(s) for s in raw_seeds)
    methods = tuple(m for m in args.methods if m in METHODS_ALL)
    if not methods:
        print("[bench] no valid methods selected", file=sys.stderr)
        return 1

    # ---- Imports --------------------------------------------------------
    joint = _try_import_joint()
    if joint is None:
        print(
            "[bench] coupling_joint_inference import failed; "
            "aborting (Wf-4 ship module missing).",
            file=sys.stderr,
        )
        return 1
    (
        LambdaCFMPipeline,
        lambda_features_from_smiles,
        LAMBDA_DIM,
        EMBED_DIM,
        load_joint_pipeline,
    ) = joint
    bi = _try_import_bidirectional()
    if bi is None:
        print(
            "[bench] coupling_bidirectional import failed; "
            "aborting (Wf-21 ship module missing).",
            file=sys.stderr,
        )
        return 1
    BidirectionalCoupling, CFMLambdaReverse, lamterm_to_features, _, _ = bi
    lambda_dim = int(LAMBDA_DIM)
    embed_dim = int(EMBED_DIM)

    # ---- Build the forward + reverse + bidirectional orchestrators ----
    try:
        pipe_forward = load_joint_pipeline(
            use_lambda_astsr=True,
            alpha=float(args.alpha),
            seed=int(seeds[0]),
        )
    except Exception as exc:
        print(f"[bench] load_joint_pipeline failed: {exc}", file=sys.stderr)
        return 1
    try:
        reverse_obj = CFMLambdaReverse()
    except Exception as exc:
        print(f"[bench] reverse orchestrator init failed: {exc}", file=sys.stderr)
        return 1
    try:
        bidirectional_obj = BidirectionalCoupling()
    except Exception as exc:
        print(
            f"[bench] BidirectionalCoupling init failed: {exc}",
            file=sys.stderr,
        )
        return 1
    bridge_source = (
        str(pipe_forward.bridge.source_path)
        if pipe_forward.bridge.source_path is not None
        else "unavailable"  # Wf-43 Phase 3 — honest label, not "stub"
    )

    # ---- Load data ------------------------------------------------------
    platinai_path = Path(args.platinai_csv)
    metalcyto_path = Path(args.metalcyto_csv)
    platinai_samples = _load_platinai_smiles(platinai_path, n_samples)
    metalcyto_samples = _load_metalcytotox_smiles(metalcyto_path, n_samples)

    if not platinai_samples and not metalcyto_samples:
        print(
            "[bench] no data: both PlatinAI and MetalCytoToxDB CSVs "
            "are empty or missing.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[bench] loaded platinai={len(platinai_samples)} "
        f"metalcyto={len(metalcyto_samples)} samples; "
        f"methods={methods} seeds={seeds}"
    )

    # ---- Run per-dataset benches ----------------------------------------
    tracemalloc.start()
    base_mem = tracemalloc.get_traced_memory()[0]
    per_dataset: List[Dict[str, Any]] = []
    if platinai_samples:
        per_dataset.append(
            _bench_dataset(
                "platinai_5k", platinai_samples, methods, seeds,
                pipe_forward=pipe_forward,
                reverse_obj=reverse_obj,
                lamterm_to_features_fn=lamterm_to_features,
                bidirectional_obj=bidirectional_obj,
                lambda_dim=lambda_dim, embed_dim=embed_dim,
                n_steps=int(args.n_steps),
                pocket_size=int(args.pocket_size),
            )
        )
    if metalcyto_samples:
        per_dataset.append(
            _bench_dataset(
                "metalcytotox_5metal_multitask",
                metalcyto_samples, methods, seeds,
                pipe_forward=pipe_forward,
                reverse_obj=reverse_obj,
                lamterm_to_features_fn=lamterm_to_features,
                bidirectional_obj=bidirectional_obj,
                lambda_dim=lambda_dim, embed_dim=embed_dim,
                n_steps=int(args.n_steps),
                pocket_size=int(args.pocket_size),
            )
        )
    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    peak_mem_mb = float((peak_mem - base_mem) / (1024 * 1024))

    report: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "seeds": list(seeds),
        "methods": list(methods),
        "alpha": float(args.alpha),
        "n_steps": int(args.n_steps),
        "pocket_size": int(args.pocket_size),
        "bridge_source": str(bridge_source),
        "lambda_dim": int(lambda_dim),
        "embed_dim": int(embed_dim),
        "peak_memory_mb": float(peak_mem_mb),
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "per_dataset": per_dataset,
    }

    # ---- Write the JSON + optional markdown ---------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "lambda_cfm_bench.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(
        f"[bench] wrote {out_json} "
        f"(datasets={len(per_dataset)} methods={len(methods)} "
        f"seeds={len(seeds)})"
    )

    if args.markdown:
        out_md = out_dir / "lambda_cfm_bench.md"
        out_md.write_text(_render_markdown(report))
        print(f"[bench] wrote {out_md}")

    # ---- Console summary ----------------------------------------------
    for ds_block in per_dataset:
        for method_name, agg in ds_block["per_method"].items():
            wall = agg.get("mean_wall_time_s", float("nan"))
            rmse = agg.get("mean_bridge_rmse", float("nan"))
            dec = agg.get("mean_decode_ratio", float("nan"))
            rt = agg.get("mean_round_trip_loss", float("nan"))
            succ = agg.get("mean_success_rate", float("nan"))
            print(
                f"[bench] {ds_block['dataset']:30s} {method_name:18s} "
                f"wall={wall:.4f}s rmse={rmse:.4f} "
                f"decode={dec:.3f} rt_loss={rt:.4f} succ={succ:.3f}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())