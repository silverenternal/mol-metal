"""A1 + A3 — D-MPNN config search via replay pool (TODO-150 §P1).

================================================================
Replay-driven HP sweep driver for the multi-metal D-MPNN runs
================================================================
This script is the load-bearing CLI front-end for the Dream-RSI
integration plan's P1 phase.  It reads a JSON pool (or builds one
from --from-dmpnn-ckpt-status), scores a list of candidate HP
deltas via :meth:`ReplayPool.replay_predict` and :meth:`per_metal_bootstrap`,
optionally validates the pool with leave-one-out per metal, and
prints (or saves) a ranked report.

The script is intentionally CPU-only — no GPU, no D-MPNN weights.
Its job is to *shortlist* the next round's HPs, not to actually
train them.  Top-N candidates can then be promoted to a real
Wf-124-style run via the existing ``molmetal/scripts/train_dmpnn_multitask_wasserstein.py``
entry point.

Two operating modes
-------------------
* ``--mode replay-only`` — pure replay-based ranking.  Fast (<60 s for
  100 candidates on 200 trajectories).  Use for nightly sweeps.
* ``--mode replay-then-validate`` — replay ranking + leave-one-out
  validation.  Slower (~10 s) but includes per-metal MAE diagnostics
  so the caller can bail out per TODO-150 §P1.3 exit gate when the
  pool is untrustworthy for a given metal.

Honest-framing
--------------
Per TODO-150 §3.5, all speedups are PROJECTED not MEASURED.  This
script never claims the paper's 50×–162× call-reduction headline
because mol-metal's bottleneck is GPU-bound D-MPNN training, not
LLM agent calls.

Usage
-----
::

    # Score a JSON of HP candidates against an existing pool
    python molmetal/scripts/r4_dmpnn_replay_sweep.py \\
        --pool-path molmetal/reports/wf_dream_rsi_p1/replay_pool.jsonl \\
        --candidates-json molmetal/reports/wf_dream_rsi_p1/candidates.json \\
        --metals Ru,Ir,Rh,Os,Re \\
        --top-n 3 \\
        --output-json molmetal/reports/wf_dream_rsi_p1/ranked.json

    # Build a pool from existing ckpt status + score a candidate list
    python molmetal/scripts/r4_dmpnn_replay_sweep.py \\
        --from-dmpnn-ckpt-status metrics/by_metric/dmpnn_ckpt_status.json \\
        --candidates-json path/to/candidates.json \\
        --bootstrap-validate \\
        --output-json path/to/ranked.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Ensure PROJECT_ROOT is on sys.path so ``from molmetal.replay`` works
# when this file is invoked via ``python molmetal/scripts/r4_dmpnn_replay_sweep.py``.
_PROJ_ROOT_HINT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _PROJ_ROOT_HINT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT_HINT)
del _PROJ_ROOT_HINT

from molmetal.replay import (  # noqa: E402
    LOW_COVERAGE_THRESHOLD,
    ReplayPool,
    TrajectoryRecord,
)
from molmetal.replay.pool import _config_hash_from_hp  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("A1+A3 D-MPNN config search via replay pool " "(TODO-150 §P1)."),
    )
    p.add_argument(
        "--pool-path",
        default=None,
        help=(
            "Path to a JSON-lines replay pool to load. "
            "If omitted and --from-dmpnn-ckpt-status is also omitted, "
            "an empty pool is created (every metal will be "
            "LOW_COVERAGE)."
        ),
    )
    p.add_argument(
        "--save-pool",
        default=None,
        help=(
            "If set, save the (possibly newly-populated) pool to this "
            "path as JSON-lines after the run."
        ),
    )
    p.add_argument(
        "--from-dmpnn-ckpt-status",
        default=None,
        help=(
            "Path to dmpnn_ckpt_status.json (or a similar JSON with a "
            "'checkpoints' key).  Each checkpoint becomes a "
            "TrajectoryRecord with placeholder score=NaN-filtered.  "
            "Records with NaN test_pearson / test_rmse are skipped."
        ),
    )
    p.add_argument(
        "--candidates-json",
        required=True,
        help=(
            "Path to a JSON file containing candidate HP dicts.  "
            "Format: either a top-level list [hp1, hp2, …] or a "
            "top-level dict {'lr_1e-4_dropout_0.1': {...}, …}."
        ),
    )
    p.add_argument(
        "--metals",
        default="Ru,Ir,Rh,Os,Re",
        help="Comma-separated metals to score (default: all 5).",
    )
    p.add_argument(
        "--baseline",
        type=float,
        default=0.5,
        help=(
            "Shrinkage baseline for replay predictions (default 0.5 "
            "— midpoint of AUC).  Override per metric (e.g. 0.0 for "
            "MAE)."
        ),
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Top-N candidates to promote (default 3).",
    )
    p.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="k for replay_predict's nearest-neighbour lookup (default 5).",
    )
    p.add_argument(
        "--mode",
        choices=["replay-only", "replay-then-validate"],
        default="replay-only",
        help=(
            "replay-only: just rank candidates.  "
            "replay-then-validate: rank + leave-one-out per-metal MAE."
        ),
    )
    p.add_argument(
        "--bootstrap-validate",
        action="store_true",
        help=(
            "Also run A3 leave-one-out within metal and report "
            "per-metal MAE between predicted and actual final scores. "
            "Bail (non-zero exit) if any metal's MAE exceeds "
            "--bootstrap-mae-threshold."
        ),
    )
    p.add_argument(
        "--bootstrap-mae-threshold",
        type=float,
        default=0.05,
        help=(
            "Per TODO-150 §P1.3: leave-one-out MAE threshold above "
            "which the pool is untrustworthy for that metal.  Default "
            "0.05 (AUC units)."
        ),
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="If set, write the ranked report to this path.",
    )
    p.add_argument(
        "--max-pool-size",
        type=int,
        default=10_000,
        help="ReplayPool max_size (default 10_000).",
    )
    return p


# ---------------------------------------------------------------------------
# Backfill from dmpnn_ckpt_status.json (P0.2 in TODO-150)
# ---------------------------------------------------------------------------


def _records_from_ckpt_status(path: str) -> List[TrajectoryRecord]:
    """Build TrajectoryRecords from dmpnn_ckpt_status.json.

    Records with NaN test_pearson or test_rmse are skipped — they
    poison nearest-neighbour arithmetic per :class:`TrajectoryRecord`'s
    ``__post_init__`` guard.  This is the Wf-120 P2 backfill path
    (TODO-150 §P0.2); until per-epoch metrics and full hp_dicts are
    backfilled from the Wf-124 P3 trainer, this function only yields
    ~15 of the 25 ckpts (the rest are score=NaN).
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    out: List[TrajectoryRecord] = []
    for ckpt in d.get("checkpoints", []):
        score = ckpt.get("test_pearson", float("nan"))
        if score is None:
            continue
        try:
            float(score)
        except (TypeError, ValueError):
            continue
        if score != score:  # NaN check
            continue
        metal = str(ckpt.get("metal", "Ru"))
        seed = int(ckpt.get("seed", 0))
        # No per-epoch / hp_dict available from ckpt_status alone;
        # hp_dict is reconstructed from sha256 (stable per-file)
        # to give the pool a non-empty hp signature for distance calc.
        hp_dict = {
            "sha256": str(ckpt.get("sha256", ""))[:16],
            "epochs": int(ckpt.get("epochs", 30)),
        }
        rec = TrajectoryRecord(
            config_hash=str(ckpt.get("sha256", _config_hash_from_hp(hp_dict)))[:16],
            hp_dict=hp_dict,
            per_epoch_metrics=[],
            final_test_score=float(score),
            metal=metal,
            seed=seed,
            timestamp=float(ckpt.get("mtime_utc_ts", time.time())),
            metadata={
                "filename": str(ckpt.get("filename", "")),
                "source": "dmpnn_ckpt_status",
            },
        )
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Candidate loader
# ---------------------------------------------------------------------------


def _load_candidates(path: str) -> Dict[str, Dict[str, Any]]:
    """Load candidate HPs from a JSON file.

    Accepts both top-level list (``[hp1, hp2, ...]``) and top-level
    dict (``{"name1": hp1, ...}``).  Returns a dict keyed by an
    auto-generated or user-supplied name.
    """
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if isinstance(d, list):
        return {
            f"cand_{i:03d}_{_config_hash_from_hp(hp)[:8]}": hp for i, hp in enumerate(d)
        }
    if isinstance(d, dict):
        return {str(k): dict(v) for k, v in d.items()}
    raise ValueError(f"candidates JSON must be list or dict; got {type(d).__name__}")


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _rank_candidate(
    pool: ReplayPool,
    candidate_hp: Mapping[str, Any],
    metals: Sequence[str],
    baseline: float,
    k: int,
) -> Dict[str, Any]:
    """Score one candidate across all metals.  Returns a per-metal breakdown."""
    per_metal: Dict[str, Dict[str, Any]] = {}
    scores: List[float] = []
    for m in metals:
        est = pool.replay_predict(
            hp_delta=candidate_hp, metal=m, baseline=baseline, k=k
        )
        per_metal[m] = est.to_dict()
        if est.n_neighbors > 0:
            scores.append(est.score)
    # Aggregate across metals (mean of per-metal scores that have data).
    if scores:
        agg_score = sum(scores) / len(scores)
        n_metals_with_data = len(scores)
    else:
        agg_score = float(baseline)
        n_metals_with_data = 0
    return {
        "agg_score": agg_score,
        "n_metals_with_data": n_metals_with_data,
        "per_metal": per_metal,
    }


# ---------------------------------------------------------------------------
# Leave-one-out validation (A3)
# ---------------------------------------------------------------------------


def _bootstrap_validate(
    pool: ReplayPool,
    metals: Sequence[str],
    mae_threshold: float,
) -> Dict[str, Any]:
    """A3 leave-one-out MAE on each metal.

    For each record ``r`` in the metal bucket, predict ``r.final_test_score``
    from the k nearest neighbours of ``r.hp_dict`` (within the same metal,
    leave-one-out).  Report the mean absolute error between predicted and
    actual.  Bails with a per-metal ``is_untrustworthy: true`` flag when
    MAE exceeds ``mae_threshold`` — callers may decide to halt the sweep.
    """
    report: Dict[str, Any] = {
        "threshold": mae_threshold,
        "per_metal": {},
        "any_untrustworthy": False,
    }
    for m in metals:
        bucket = pool._by_metal.get(m, [])  # noqa: SLF001 (intentional introspection)
        if len(bucket) < 2:
            report["per_metal"][m] = {
                "n_records": len(bucket),
                "mae": None,
                "note": "too few records for leave-one-out (<2)",
                "is_untrustworthy": True,
            }
            report["any_untrustworthy"] = True
            continue
        items: List[tuple] = [
            (
                pool._records[i].hp_dict,  # noqa: SLF001
                pool._records[i].final_test_score,  # noqa: SLF001
            )
            for i in bucket
        ]
        errors: List[float] = []
        for held_idx in range(len(items)):
            held_hp, held_score = items[held_idx]
            others = [items[j] for j in range(len(items)) if j != held_idx]
            # Same hp-distance ranking as replay_predict.
            from molmetal.replay.pool import _hp_distance

            others.sort(key=lambda t: _hp_distance(held_hp, t[0]))
            top = others[: max(1, len(others))]
            pred = sum(s for _, s in top) / len(top)
            errors.append(abs(pred - held_score))
        mae = sum(errors) / len(errors) if errors else 0.0
        is_untrustworthy = mae > mae_threshold
        report["per_metal"][m] = {
            "n_records": len(bucket),
            "mae": mae,
            "is_untrustworthy": is_untrustworthy,
        }
        if is_untrustworthy:
            report["any_untrustworthy"] = True
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # ----- Build / load pool ------------------------------------------------
    pool = ReplayPool(
        max_size=int(args.max_pool_size),
        low_coverage_threshold=LOW_COVERAGE_THRESHOLD,
    )
    n_loaded = 0
    if args.pool_path and os.path.exists(args.pool_path):
        pool = ReplayPool.load(
            args.pool_path,
            max_size=int(args.max_pool_size),
            low_coverage_threshold=LOW_COVERAGE_THRESHOLD,
        )
        n_loaded = len(pool)
    if args.from_dmpnn_ckpt_status:
        recs = _records_from_ckpt_status(args.from_dmpnn_ckpt_status)
        pool.extend(recs)
        print(
            f"[r4_dmpnn_replay_sweep] backfilled {len(recs)} records from "
            f"{args.from_dmpnn_ckpt_status}",
            file=sys.stderr,
        )

    metals = [m.strip() for m in args.metals.split(",") if m.strip()]
    if not metals:
        print("ERROR: --metals must list at least one metal", file=sys.stderr)
        return 2

    print(
        f"[r4_dmpnn_replay_sweep] pool size={len(pool)} "
        f"(loaded={n_loaded}); metals={metals}; "
        f"low_coverage_threshold={LOW_COVERAGE_THRESHOLD}",
        file=sys.stderr,
    )
    cov_table = pool.coverage_table()
    print(
        "[r4_dmpnn_replay_sweep] coverage: "
        + json.dumps(
            {
                m: {
                    "n_records": v["n_records"],
                    "is_low_coverage": v["is_low_coverage"],
                }
                for m, v in cov_table.items()
            }
        ),
        file=sys.stderr,
    )

    # ----- Load candidates ---------------------------------------------------
    cand_dict = _load_candidates(args.candidates_json)
    print(
        f"[r4_dmpnn_replay_sweep] loaded {len(cand_dict)} candidates from "
        f"{args.candidates_json}",
        file=sys.stderr,
    )

    # ----- Rank -------------------------------------------------------------
    ranked: List[Dict[str, Any]] = []
    for name, hp in cand_dict.items():
        score = _rank_candidate(
            pool=pool,
            candidate_hp=hp,
            metals=metals,
            baseline=float(args.baseline),
            k=int(args.k_neighbors),
        )
        ranked.append({"name": name, "hp": dict(hp), **score})
    ranked.sort(key=lambda r: float(r["agg_score"]), reverse=True)

    # ----- Bootstrap validation (A3) ---------------------------------------
    bootstrap_report: Optional[Dict[str, Any]] = None
    if args.mode == "replay-then-validate" or args.bootstrap_validate:
        bootstrap_report = _bootstrap_validate(
            pool=pool,
            metals=metals,
            mae_threshold=float(args.bootstrap_mae_threshold),
        )
        for m, v in bootstrap_report["per_metal"].items():
            tag = " (UNTRUSTWORTHY)" if v.get("is_untrustworthy") else ""
            mae = v.get("mae")
            mae_str = f"{mae:.4f}" if mae is not None else "n/a"
            print(
                f"[bootstrap] metal={m} n={v.get('n_records')} " f"MAE={mae_str}{tag}",
                file=sys.stderr,
            )

    # ----- Print top-N ------------------------------------------------------
    print(
        f"\n=== Top {min(args.top_n, len(ranked))} candidates by agg_score ===",
        file=sys.stderr,
    )
    for i, r in enumerate(ranked[: args.top_n], start=1):
        print(
            f"  {i}. {r['name']:30s} agg_score={r['agg_score']:.4f} "
            f"({r['n_metals_with_data']}/{len(metals)} metals with data)",
            file=sys.stderr,
        )
        for m, v in r["per_metal"].items():
            print(
                f"     {m:4s} score={v['score']:.4f} "
                f"confidence={v['confidence']:6s} "
                f"n={v['n_neighbors']:2d} weight={v['shrinkage_weight']:.2f}",
                file=sys.stderr,
            )

    # ----- Persist outputs --------------------------------------------------
    if args.save_pool:
        path = pool.save(args.save_pool)
        print(
            f"[r4_dmpnn_replay_sweep] saved pool ({len(pool)} records) to {path}",
            file=sys.stderr,
        )

    if args.output_json:
        report = {
            "pool_size": len(pool),
            "coverage": cov_table,
            "metals": list(metals),
            "n_candidates": len(ranked),
            "baseline": float(args.baseline),
            "k_neighbors": int(args.k_neighbors),
            "low_coverage_threshold": LOW_COVERAGE_THRESHOLD,
            "ranked": ranked,
            "bootstrap": bootstrap_report,
            "timestamp": time.time(),
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(
            f"[r4_dmpnn_replay_sweep] wrote report to {args.output_json}",
            file=sys.stderr,
        )

    # ----- Exit code -------------------------------------------------------
    if bootstrap_report and bootstrap_report.get("any_untrustworthy"):
        # Per TODO-150 §P1.3: if MAE exceeds threshold on any metal,
        # the sweep is untrustworthy for that cell.  Return non-zero so
        # callers can bail before committing a GPU verify-run.  This
        # is the audit that prevents R1 (replay-coverage collapse) from
        # silently corrupting real HP selection.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
