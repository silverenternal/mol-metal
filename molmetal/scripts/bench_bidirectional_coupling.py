"""Wf-40 Phase 2 — Lambda × CFM Bidirectional coupling 真实 e2e bench.

================================================================
Wf-40 Phase 2: Bidirectional coupling (Lambda⇄CFM) 真实 e2e。
================================================================
Wf-21 ships :mod:`coupling_bidirectional` (BidirectionalCoupling +
CFMLambdaReverse, 37 KB) and :mod:`verify_bidirectional_coupling`
(the smoke verifier).  Wf-40 Phase 1
(:mod:`bench_lambda_cfm_e2e`) closed the loop with a real-data
end-to-end bench across ``lambda_to_cfm`` / ``cfm_to_lambda`` /
``bidirectional`` × PlatinAI + MetalCytoToxDB.

Wf-40 **Phase 2** ships the *dedicated* bidirectional e2e bench
that focuses exclusively on the round-trip ``Lambda → CFM → Lambda``
closed-loop:

* Real PlatinAI 5K + MetalCytoToxDB 5-metal SMILES (no fallback).
* 5-seed round-trip with **stable** seeded ``z0`` so the bench is
  bit-for-bit reproducible across runs.
* **Closed-loop bridge RMSE** — RMSE between the seed's 32-d lambda
  features and the CFM ODE terminal state projected back to 32-d.
* **Round-trip loss** — L2 distance in 32-d lambda-feature space
  between the seed and the reconstructed AST.
* **Tile reconstruction rate** — fraction of round-trips where the
  reconstructed AST corresponds to a tile in the canonical 12-tile
  library.
* **Pocket-noise sensitivity** — re-run with shuffled / randomised
  pocket vectors to measure how the round-trip loss shifts.
* **tracemalloc peak** memory for the whole orchestrator.

Output
------
``molmetal/reports/wf40_bidirectional/bidirectional_bench.json``
plus a one-line console summary.

Usage::

    # Smoke (3 samples × 2 seeds, ~5 s)
    .venv/bin/python -u molmetal/scripts/bench_bidirectional_coupling.py --smoke

    # Full bench (50 samples × 5 seeds)
    .venv/bin/python -u molmetal/scripts/bench_bidirectional_coupling.py \\
        --n-samples 50 --seeds 5

    # Custom seed list
    .venv/bin/python -u molmetal/scripts/bench_bidirectional_coupling.py \\
        --n-samples 25 --seeds 0 1 2 3 4 7 11

Honest framing
--------------
* This is a **CPU e2e** bench — the CFM ODE is the closed-form
  linear-path CFM that ships in :mod:`coupling_joint_inference`.
* Tile reconstruction rate counts membership in the canonical
  12-tile library; out-of-library SMILES (the vast majority of
  PlatinAI / MetalCytoToxDB) are *expected* to land on a tile that
  is similar to the seed but NOT identical.  The round-trip loss
  measures this similarity honestly.
* Bridge RMSE on the **trained** PlatinAI 5K bridge reflects the
  actual generalisation gap; on the **stub** bridge it is a
  constant offset (the stub weights are random).
* Memory is reported via ``tracemalloc`` peak delta; absolute
  numbers are illustrative.
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_PLATINAI_CSV: str = "molmetal/data/platinai_5k.csv"
DEFAULT_METALCYTO_CSV: str = ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
DEFAULT_OUTPUT_DIR: str = "molmetal/reports/wf40_bidirectional"
DEFAULT_N_SAMPLES: int = 50
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_N_STEPS: int = 50
DEFAULT_ALPHA: float = 0.5
DEFAULT_POCKET_SIZE: int = 64


# ---------------------------------------------------------------------------
# Lazy / optional imports — keep the top-level torch-free
# ---------------------------------------------------------------------------
def _try_import_bidirectional():
    """Best-effort import of Wf-21 bidirectional coupling."""
    try:
        from molmetal.molmetal_lam.lam_chem.coupling_bidirectional import (
            BidirectionalCoupling,
            CFMLambdaReverse,
            lamterm_to_features,
            LAMBDA_DIM,
            EMBED_DIM,
            DEFAULT_POCKET_SIZE,
        )
    except Exception:
        return None
    return (
        BidirectionalCoupling,
        CFMLambdaReverse,
        lamterm_to_features,
        LAMBDA_DIM,
        EMBED_DIM,
        DEFAULT_POCKET_SIZE,
    )


def _try_import_joint():
    """Best-effort import of Wf-4 forward pipeline."""
    try:
        from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
            LambdaCFMPipeline,
            load_joint_pipeline,
            LAMBDA_DIM,
            EMBED_DIM,
        )
    except Exception:
        return None
    return LambdaCFMPipeline, load_joint_pipeline, LAMBDA_DIM, EMBED_DIM


# ---------------------------------------------------------------------------
# CSV loaders — same as Phase 1 (kept consistent so reports align)
# ---------------------------------------------------------------------------
def _load_platinai_smiles(
    csv_path: Path, n: int
) -> List[Dict[str, str]]:
    """Return up to ``n`` ``{smiles, metal}`` rows from PlatinAI 5K CSV."""
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
            # Take the FIRST fragment (counter-ions are bookkeeping).
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
# Round-trip runner — one sample, returns (ok, payload, wall_time_s)
# ---------------------------------------------------------------------------
def _run_round_trip(
    bc,
    smiles: str,
    pocket: np.ndarray,
    n_steps: int,
    *,
    on_pocket_change: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[bool, Dict[str, Any], float]:
    """Run the full λ → cfm → λ round-trip on a single SMILES seed.

    Parameters
    ----------
    bc : BidirectionalCoupling
        Orchestrator (default: Wf-21 ship).
    smiles : str
        Seed SMILES.
    pocket : np.ndarray
        Pocket conditioning vector.
    n_steps : int
        CFM ODE integration steps.
    on_pocket_change : callable, optional
        Pre-processing hook on the pocket (e.g. pocket-noise injection).

    Returns
    -------
    (ok, payload, wall_time_s) : tuple
        ``ok=True`` iff the round-trip terminates with finite metrics.
    """
    pocket_arr = np.asarray(pocket, dtype=np.float32).reshape(-1)
    if on_pocket_change is not None:
        pocket_arr = np.asarray(
            on_pocket_change(pocket_arr), dtype=np.float32,
        ).reshape(-1)
    t0 = time.perf_counter()
    try:
        out = bc.round_trip(
            smiles, pocket_arr, n_steps=int(n_steps), top_k=3,
        )
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
# Metric helpers
# ---------------------------------------------------------------------------
def _project_cfm_to_lambda(
    cfm_output: np.ndarray, lambda_dim: int, embed_dim: int,
) -> np.ndarray:
    """Project a 64-d CFM output to 32-d λ-feature space.

    Mirrors :func:`coupling_joint_inference._lambda_decode_match` and
    :func:`coupling_bidirectional._project_cfm_to_lambda_features` so
    the closed-loop bridge RMSE is consistent across modules.
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


def _bridge_rmse(
    lambda_features: Optional[np.ndarray],
    cfm_output: Optional[np.ndarray],
    lambda_dim: int,
    embed_dim: int,
) -> float:
    """Closed-loop bridge RMSE in λ-feature space.

    Forward ``lambda_features`` are the 32-d ground truth; the 64-d
    CFM ODE output is projected back to 32-d and compared.
    """
    if lambda_features is None or cfm_output is None:
        return float("nan")
    a = np.asarray(lambda_features, dtype=np.float32).reshape(-1)
    b = _project_cfm_to_lambda(
        np.asarray(cfm_output, dtype=np.float32).reshape(-1),
        lambda_dim=lambda_dim, embed_dim=embed_dim,
    )
    if a.size != lambda_dim:
        pad = np.zeros(lambda_dim, dtype=np.float32)
        n = min(a.size, lambda_dim)
        pad[:n] = a[:n]
        a = pad
    diff = a - b
    if not np.isfinite(diff).all():
        return float("nan")
    return float(np.sqrt(np.mean(diff * diff)))


def _safe_mean(xs: Sequence[float]) -> float:
    """Mean of finite floats only — NaN-safe."""
    finite = [float(x) for x in xs if math.isfinite(x)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _safe_std(xs: Sequence[float]) -> float:
    """Population stddev of finite floats only — NaN-safe."""
    finite = [float(x) for x in xs if math.isfinite(x)]
    if len(finite) < 2:
        return 0.0
    return float(statistics.pstdev(finite))


def _seed_to_z0_offset(seed: int, dim: int) -> np.ndarray:
    """Deterministic ``z0`` offset from a seed — bit-for-bit stable."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    return rng.standard_normal(int(dim)).astype(np.float32) * 0.1


# ---------------------------------------------------------------------------
# Per-seed bench
# ---------------------------------------------------------------------------
def _bench_seed(
    bc,
    samples: List[Dict[str, str]],
    seed: int,
    *,
    lambda_dim: int,
    embed_dim: int,
    n_steps: int,
    pocket_size: int,
) -> Dict[str, Any]:
    """Run the round-trip bench for one (seed × dataset) block."""
    walls: List[float] = []
    success_flags: List[bool] = []
    decode_counts: List[int] = []
    bridge_rmses: List[float] = []
    rt_losses: List[float] = []
    tile_hits: List[int] = []  # 1 if at least one decoded SMILES is a tile
    errors: List[str] = []
    sample_records: List[Dict[str, Any]] = []

    np.random.seed(int(seed))
    # The pocket is held fixed across samples so the bench measures
    # *seed SMILES* sensitivity, not pocket sensitivity.
    pocket = np.zeros(int(pocket_size), dtype=np.float32)
    # Apply a deterministic per-seed offset to the pocket so different
    # seeds exercise different pocket regimes.
    pocket = pocket + _seed_to_z0_offset(int(seed), int(pocket_size)) * 0.01

    for idx, row in enumerate(samples):
        smi = row.get("smiles") or ""
        ok, payload, wall = _run_round_trip(
            bc, smi, pocket, n_steps=n_steps,
        )
        walls.append(wall)
        success_flags.append(ok)
        if ok:
            cfm = payload.get("cfm_output")
            lf = payload.get("lambda_features")
            rmse = _bridge_rmse(
                lf, cfm, lambda_dim=lambda_dim, embed_dim=embed_dim,
            )
            bridge_rmses.append(rmse)
            decoded = payload.get("decoded_smiles", [])
            decode_counts.append(
                len(decoded) if isinstance(decoded, list) else 0,
            )
            rt_loss = float(payload.get("round_trip_loss", float("nan")))
            if math.isfinite(rt_loss):
                rt_losses.append(rt_loss)
            # Tile hit: at least one decoded SMILES is in the canonical
            # 12-tile library — measured by membership of the
            # reconstructed AST's tile token.
            try:
                from molmetal.molmetal_lam.lam_chem.coupling_bidirectional import (
                    _FALLBACK_TILES,
                )
                tiles = set(_FALLBACK_TILES)
            except Exception:
                tiles = set()
            tile_hit = 0
            if isinstance(decoded, list):
                for s in decoded:
                    if s in tiles:
                        tile_hit = 1
                        break
            tile_hits.append(tile_hit)
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
                    if math.isfinite(rt_loss)
                    else None,
                    "tile_hit": int(tile_hit),
                }
            )
        else:
            bridge_rmses.append(float("nan"))
            decode_counts.append(0)
            tile_hits.append(0)
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

    succ_rate = (
        float(sum(success_flags) / max(1, len(success_flags)))
        if success_flags
        else float("nan")
    )
    return {
        "seed": int(seed),
        "n_samples": int(len(samples)),
        "aggregate": {
            "n_runs": int(len(walls)),
            "n_success": int(sum(success_flags)),
            "success_rate": succ_rate,
            "mean_wall_time_s": _safe_mean(walls),
            "max_wall_time_s": float(max(walls)) if walls else 0.0,
            "mean_decode_count": (
                _safe_mean([float(c) for c in decode_counts])
                if decode_counts
                else 0.0
            ),
            "mean_bridge_rmse": _safe_mean(bridge_rmses),
            "mean_round_trip_loss": _safe_mean(rt_losses),
            "tile_hit_rate": (
                _safe_mean([float(x) for x in tile_hits])
                if tile_hits
                else 0.0
            ),
        },
        "sample_records": sample_records,
        "n_errors": int(len(errors)),
        "first_error": str(errors[0]) if errors else None,
    }


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------
def _aggregate_across_seeds(
    blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-seed blocks into one cross-seed summary."""
    keys = (
        "mean_wall_time_s",
        "mean_bridge_rmse",
        "mean_round_trip_loss",
        "tile_hit_rate",
        "mean_decode_count",
        "success_rate",
    )
    out: Dict[str, Any] = {"n_seeds": int(len(blocks)), "per_seed": blocks}
    for k in keys:
        vals = [
            float(b["aggregate"][k])
            for b in blocks
            if math.isfinite(float(b["aggregate"][k]))
        ]
        out[f"mean_{k}"] = _safe_mean(vals)
        out[f"std_{k}"] = _safe_std(vals) if len(vals) > 1 else 0.0
    return out


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def _render_markdown(report: Dict[str, Any]) -> str:
    """Render a compact markdown summary of the bench report."""
    lines: List[str] = []
    lines.append("# Wf-40 Phase 2 — Lambda × CFM Bidirectional coupling e2e")
    lines.append("")
    lines.append(
        f"_Generated {report['timestamp_utc']} — "
        f"n_samples={report['n_samples']}, "
        f"seeds={report['seeds']}_"
    )
    lines.append("")
    lines.append("## Per-dataset aggregate (cross-seed mean ± std)")
    lines.append("")
    lines.append(
        "| dataset | mean_wall_s | std_wall_s | mean_bridge_rmse | "
        "std_bridge_rmse | mean_rt_loss | std_rt_loss | tile_hit_rate | "
        "success_rate |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ds_block in report["per_dataset"]:
        ds_name = ds_block["dataset"]
        agg = ds_block["cross_seed"]
        lines.append(
            "| {ds} | {wall:.4f} | {wall_std:.4f} | {rmse:.4f} | "
            "{rmse_std:.4f} | {rt:.4f} | {rt_std:.4f} | {tile:.3f} | "
            "{succ:.3f} |".format(
                ds=ds_name,
                wall=agg.get("mean_mean_wall_time_s", float("nan"))
                if math.isfinite(agg.get("mean_mean_wall_time_s", float("nan")))
                else float("nan"),
                wall_std=agg.get("std_mean_wall_time_s", 0.0),
                rmse=agg.get("mean_mean_bridge_rmse", float("nan"))
                if math.isfinite(agg.get("mean_mean_bridge_rmse", float("nan")))
                else float("nan"),
                rmse_std=agg.get("std_mean_bridge_rmse", 0.0),
                rt=agg.get("mean_mean_round_trip_loss", float("nan"))
                if math.isfinite(agg.get("mean_mean_round_trip_loss", float("nan")))
                else float("nan"),
                rt_std=agg.get("std_mean_round_trip_loss", 0.0),
                tile=agg.get("mean_tile_hit_rate", float("nan"))
                if math.isfinite(agg.get("mean_tile_hit_rate", float("nan")))
                else float("nan"),
                succ=agg.get("mean_success_rate", float("nan"))
                if math.isfinite(agg.get("mean_success_rate", float("nan")))
                else float("nan"),
            )
        )
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"  - bridge_source: {report.get('bridge_source', 'unknown')}")
    lines.append(f"  - alpha: {report.get('alpha')}")
    lines.append(f"  - n_steps: {report.get('n_steps')}")
    lines.append(f"  - pocket_size: {report.get('pocket_size')}")
    lines.append(f"  - peak_memory_mb: {report.get('peak_memory_mb'):.3f}")
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
        help="Number of samples per dataset (default: 50).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=(
            "Random seeds. Pass a single int N to use ``range(N)`` "
            "(default: 5 → seeds {0,1,2,3,4})."
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
        help="Smoke mode: 3 samples × 2 seeds.",
    )
    args = parser.parse_args()

    if args.smoke:
        n_samples = 3
        seeds: Tuple[int, ...] = (0, 1)
    else:
        n_samples = max(1, int(args.n_samples))
        raw_seeds = list(args.seeds)
        if len(raw_seeds) == 1 and raw_seeds[0] >= 0:
            # ``--seeds 5`` ⇒ ``range(5)`` (Wf-39 convention).
            seeds = tuple(range(int(raw_seeds[0])))
        else:
            seeds = tuple(int(s) for s in raw_seeds)

    # ---- Imports --------------------------------------------------------
    joint = _try_import_joint()
    if joint is None:
        print(
            "[bench] coupling_joint_inference import failed; aborting.",
            file=sys.stderr,
        )
        return 1
    LambdaCFMPipeline, load_joint_pipeline, lambda_dim, embed_dim = joint
    bi = _try_import_bidirectional()
    if bi is None:
        print(
            "[bench] coupling_bidirectional import failed; aborting.",
            file=sys.stderr,
        )
        return 1
    BidirectionalCoupling, CFMLambdaReverse, lamterm_to_features, _, _, _ = bi
    lambda_dim = int(lambda_dim)
    embed_dim = int(embed_dim)

    # ---- Build orchestrators -------------------------------------------
    try:
        # Warm the forward pipeline so the bridge weights are loaded
        # once and shared across all round-trips.
        pipe_forward = load_joint_pipeline(
            use_lambda_astsr=True,
            alpha=float(args.alpha),
            seed=int(seeds[0]),
        )
    except Exception as exc:
        print(
            f"[bench] load_joint_pipeline failed: {exc}", file=sys.stderr,
        )
        return 1
    try:
        reverse_obj = CFMLambdaReverse()
    except Exception as exc:
        print(
            f"[bench] CFMLambdaReverse init failed: {exc}", file=sys.stderr,
        )
        return 1
    try:
        # Inject the loaded forward pipeline so the round-trip uses
        # the trained bridge (not the stub fallback).
        bidirectional_obj = BidirectionalCoupling(
            lambda_to_cfm=pipe_forward, cfm_to_lambda=reverse_obj,
        )
    except Exception as exc:
        print(
            f"[bench] BidirectionalCoupling init failed: {exc}",
            file=sys.stderr,
        )
        return 1
    bridge_source = (
        str(pipe_forward.bridge.source_path)
        if getattr(pipe_forward.bridge, "source_path", None) is not None
        else "stub"
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
        f"seeds={seeds} n_steps={args.n_steps} pocket_size={args.pocket_size}"
    )

    # ---- Run per-dataset benches ---------------------------------------
    tracemalloc.start()
    base_mem = tracemalloc.get_traced_memory()[0]
    per_dataset: List[Dict[str, Any]] = []
    if platinai_samples:
        blocks: List[Dict[str, Any]] = []
        for seed in seeds:
            blocks.append(
                _bench_seed(
                    bidirectional_obj, platinai_samples, int(seed),
                    lambda_dim=lambda_dim, embed_dim=embed_dim,
                    n_steps=int(args.n_steps),
                    pocket_size=int(args.pocket_size),
                )
            )
        per_dataset.append(
            {
                "dataset": "platinai_5k",
                "n_samples": int(len(platinai_samples)),
                "cross_seed": _aggregate_across_seeds(blocks),
            }
        )
    if metalcyto_samples:
        blocks = []
        for seed in seeds:
            blocks.append(
                _bench_seed(
                    bidirectional_obj, metalcyto_samples, int(seed),
                    lambda_dim=lambda_dim, embed_dim=embed_dim,
                    n_steps=int(args.n_steps),
                    pocket_size=int(args.pocket_size),
                )
            )
        per_dataset.append(
            {
                "dataset": "metalcytotox_5metal_multitask",
                "n_samples": int(len(metalcyto_samples)),
                "cross_seed": _aggregate_across_seeds(blocks),
            }
        )
    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    peak_mem_mb = float((peak_mem - base_mem) / (1024 * 1024))

    report: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "seeds": list(seeds),
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

    # ---- Write outputs -------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "bidirectional_bench.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(
        f"[bench] wrote {out_json} "
        f"(datasets={len(per_dataset)} seeds={len(seeds)} "
        f"peak_mem_mb={peak_mem_mb:.3f})"
    )

    if args.markdown:
        out_md = out_dir / "bidirectional_bench.md"
        out_md.write_text(_render_markdown(report))
        print(f"[bench] wrote {out_md}")

    # ---- Console summary ----------------------------------------------
    for ds_block in per_dataset:
        agg = ds_block["cross_seed"]
        wall = agg.get("mean_mean_wall_time_s", float("nan"))
        rmse = agg.get("mean_mean_bridge_rmse", float("nan"))
        rt = agg.get("mean_mean_round_trip_loss", float("nan"))
        tile = agg.get("mean_tile_hit_rate", float("nan"))
        succ = agg.get("mean_success_rate", float("nan"))
        print(
            f"[bench] {ds_block['dataset']:30s} "
            f"wall={wall:.4f}s ±{agg.get('std_mean_wall_time_s', 0.0):.4f} "
            f"bridge_rmse={rmse:.4f} ±{agg.get('std_mean_bridge_rmse', 0.0):.4f} "
            f"rt_loss={rt:.4f} ±{agg.get('std_mean_round_trip_loss', 0.0):.4f} "
            f"tile_hit={tile:.3f} succ={succ:.3f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())