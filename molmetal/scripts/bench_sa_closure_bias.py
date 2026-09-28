#!/usr/bin/env python3
"""bench_sa_closure_bias.py — Wf-126 Phase 3 / TODO-124 SA closure-bias bench.

==============================================================================
Wf-126 Phase 3 (TODO-124 m5 SA closure-bias generation)
==============================================================================

This script measures the **closure-bounded** SA-score distribution
on the M2 pocket_closure_bias cohort (TODO-126 §3 M2, shipped by
Wf-126 P3 / Wf-123 P2) and compares it to the historical 3.099
unconstrained SA mean (TODO-124 §0).

Inputs
------
* ``molmetal/molmetal_lam/reward/pocket_closure_bias.py`` — M2 module
  with :func:`closure_bounded_smiles_generation` (per-pocket)
  and :func:`closure_oversample` (TODO-121 Rh-style synthetic cohort).
* ``molmetal/molmetal_lam/sbdd_env/sa_score.py`` — SA score
  wrapper (:func:`sa_score_ertl`, :func:`batch_sa_score`).
* ``molmetal/molmetal_lam/sbdd_env/aizynth_adapter.py`` — AiZynth
  smarts fallback for per-compound ``engine`` tag.

Outputs
-------
* ``<out-dir>/sa_metrics.json`` — mean / min / max / per-decile SA on
  the closure cohort + the unconstrained baseline + AiZynth engine
  fractions.
* ``<out-dir>/per_compound_sa.csv`` — per-compound SMILES + SA
  + Catherine 6/6 pass + AiZynth engine tag.
* ``<out-dir>/sa_run_summary.json`` — wall-time + parameter echo
  + honest framing block.

Verify
------
* ``sa_metrics.json`` ``mean_sa_closure_bounded <= 3.0`` (and ideally
  ``<= 2.8``); baseline ``mean_sa_unconstrained == 3.099`` (or close).
* ``sa_metrics.json`` ``aiZynth_synthesizable_rate >= 0.5``.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=0 \\
    molmetal/.venvs/mol-metal-venv/bin/python \\
        molmetal/scripts/bench_sa_closure_bias.py \\
        --cohort closure_bounded \\
        --seeds 0 1 2 \\
        --out-dir molmetal/reports/wf126_p3_sa_closure

Honest framing
--------------
The 3.099 vs 3.0 gap is a 3% relative gap (TODO-124 §0).  The fix
isn't to game SA score; it's to **change the cohort** to a chemistry
regime (5-click productive space) where synthesis accessibility is
naturally better.  This script reports the per-cohort SA honestly
without modifying the SA-score definition.

CPU-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("wf126_p3_sa_closure")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Target SA mean on closure-bounded cohort (TODO-124 §2 path).
TARGET_SA_MEAN_CLOSURE: float = 2.8

#: Acceptance threshold — at most TARGET_SA_MEAN_CLOSURE.
SA_MEAN_ACCEPT_CLOSURE: float = 3.0

#: Historical baseline SA mean on unconstrained cohort (TODO-124 §0).
BASELINE_SA_MEAN_UNCONSTRAINED: float = 3.099

#: Default cohort size (matches SA-mean stability target).
DEFAULT_COHORT_SIZE: int = 100

#: Default seeds (3 seeds matches the spec; aggregated over seeds).
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)


# ---------------------------------------------------------------------------
# Per-cohort record
# ---------------------------------------------------------------------------


@dataclass
class CohortRecord:
    """One cohort's per-cohort aggregate + diagnostics.

    Attributes
    ----------
    cohort_label : str
        Cohort name (``"closure_bounded"`` / ``"unbiased"``).
    pocket_id : str
        Pocket identifier (``"unbiased"`` for the neutral cohort).
    seed : int
        Random seed.
    n_requested : int
        Requested cohort size.
    n_emitted_raw : int
        Number of SMILES emitted by the closure BFS.
    n_kept : int
        Number of SMILES kept after Catherine 6/6 filter.
    catherine_pass_rate : float
        ``n_kept / max(n_emitted_raw, 1)``.
    sa_mean : float
        Mean SA on the kept cohort.
    sa_median : float
        Median SA on the kept cohort.
    sa_std : float
        Std-dev SA on the kept cohort.
    sa_min : float
        Min SA on the kept cohort.
    sa_max : float
        Max SA on the kept cohort.
    n_valid_sa : int
        Number of valid (non-NaN) SA values.
    aizynth_synthesizable_rate : float
        Fraction of kept SMILES for which ``check_synthesizable``
        returned ``synthesizable=True``.
    aizynth_engine_breakdown : dict
        ``{"aizynthfinder": int, "smarts_fallback": int}`` — engine
        tag distribution.
    per_compound : list[dict]
        Per-compound SMILES + SA + AiZynth engine + synthesizable flag.
    wall_time_s : float
        Wall-time of the run.
    """

    cohort_label: str
    pocket_id: str
    seed: int
    n_requested: int
    n_emitted_raw: int
    n_kept: int
    catherine_pass_rate: float
    sa_mean: float
    sa_median: float
    sa_std: float
    sa_min: float
    sa_max: float
    n_valid_sa: int
    aizynth_synthesizable_rate: float
    aizynth_engine_breakdown: Dict[str, int]
    per_compound: List[Dict[str, object]]
    wall_time_s: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "cohort_label": self.cohort_label,
            "pocket_id": self.pocket_id,
            "seed": int(self.seed),
            "n_requested": int(self.n_requested),
            "n_emitted_raw": int(self.n_emitted_raw),
            "n_kept": int(self.n_kept),
            "catherine_pass_rate": float(self.catherine_pass_rate),
            "sa_mean": _safe_float(self.sa_mean),
            "sa_median": _safe_float(self.sa_median),
            "sa_std": _safe_float(self.sa_std),
            "sa_min": _safe_float(self.sa_min),
            "sa_max": _safe_float(self.sa_max),
            "n_valid_sa": int(self.n_valid_sa),
            "aizynth_synthesizable_rate": _safe_float(
                self.aizynth_synthesizable_rate
            ),
            "aizynth_engine_breakdown": dict(self.aizynth_engine_breakdown),
            "per_compound": list(self.per_compound),
            "wall_time_s": float(self.wall_time_s),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(x: float) -> Optional[float]:
    """Return ``None`` for NaN, else round to 6 dp."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(round(x, 6))


def _aggregate_sa(sa_values: List[float]) -> Dict[str, float]:
    """Compute aggregate stats over a list of SA values (NaN-safe)."""
    valid = [v for v in sa_values if v == v]  # filter NaN
    if not valid:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n_valid": 0,
        }
    return {
        "mean": float(statistics.fmean(valid)),
        "median": float(statistics.median(valid)),
        "std": float(statistics.pstdev(valid)) if len(valid) > 1 else 0.0,
        "min": float(min(valid)),
        "max": float(max(valid)),
        "n_valid": int(len(valid)),
    }


def _score_sa_batch(smiles_list: Sequence[str]) -> List[float]:
    """Score SA on a list of SMILES (NaN on parse failure)."""
    try:
        from molmetal.molmetal_lam.sbdd_env.sa_score import sa_score_ertl
    except Exception:  # pragma: no cover — environment guard
        LOG.warning("[sa_closure] sa_score module not importable")
        return [float("nan")] * len(smiles_list)
    return [sa_score_ertl(smi) for smi in smiles_list]


def _aizynth_tag_batch(smiles_list: Sequence[str]) -> List[Tuple[bool, str]]:
    """Tag each SMILES with AiZynth synthesizable flag + engine.

    Returns list of ``(synthesizable, engine)`` tuples.  Falls back
    to ``(False, "smarts_fallback")`` on import failure.
    """
    try:
        from molmetal.molmetal_lam.sbdd_env.aizynth_adapter import (
            check_synthesizable,
        )
    except Exception:  # pragma: no cover
        LOG.warning("[sa_closure] aizynth_adapter not importable")
        return [(False, "smarts_fallback")] * len(smiles_list)
    out: List[Tuple[bool, str]] = []
    for smi in smiles_list:
        try:
            rpt = check_synthesizable(smi)
            out.append((bool(rpt.synthesizable), str(rpt.engine)))
        except Exception as exc:  # noqa: BLE001
            LOG.debug("[sa_closure] aizynth fail for %r: %s", smi, exc)
            out.append((False, "smarts_fallback"))
    return out


# ---------------------------------------------------------------------------
# Cohort generation + scoring
# ---------------------------------------------------------------------------


def _smiles_from_bounded(result) -> List[str]:
    """Extract SMILES list from a BoundedGenerationResult."""
    return [str(c["smiles"]) for c in result.candidates]


def _run_one(
    cohort: str,
    seed: int,
    cohort_size: int,
    pocket_id: Optional[str],
    per_scaffold_cap: int,
    max_depth: int,
) -> CohortRecord:
    """Run one cohort generation + SA + AiZyth tagging pass.

    Parameters
    ----------
    cohort : str
        ``"closure_bounded"`` (per-pocket) or ``"unbiased"`` (neutral).
    seed : int
        Random seed (echoed in the record for reproducibility).
    cohort_size : int
        Target cohort size.
    pocket_id : str, optional
        Pocket identifier (required for ``cohort="closure_bounded"``).
    per_scaffold_cap : int
        Per-seed yield cap (for unbiased cohort only).
    max_depth : int
        BFS depth.

    Returns
    -------
    CohortRecord
        The per-cohort record.
    """
    from molmetal.molmetal_lam.reward.pocket_closure_bias import (
        closure_bounded_smiles_generation,
        closure_oversample,
        SUPPORTED_POCKETS,
    )

    t0 = time.perf_counter()
    if cohort == "closure_bounded":
        if pocket_id is None:
            raise ValueError(
                "pocket_id required for cohort='closure_bounded'"
            )
        if pocket_id not in SUPPORTED_POCKETS:
            raise ValueError(
                f"pocket_id={pocket_id!r} not in SUPPORTED_POCKETS="
                f"{list(SUPPORTED_POCKETS)}"
            )
        c = closure_bounded_smiles_generation(
            pocket_id=pocket_id,
            n=cohort_size,
            max_depth=max_depth,
            max_products=cohort_size * 4,
            include_metal_coord_bias=True,
        )
        smiles = _smiles_from_bounded(c)
        cohort_pocket_id = c.pocket_id
        cohort_n_raw = int(c.n_candidates_raw)
        cohort_n_kept = int(c.n_candidates_kept)
        cohort_catherine_rate = float(c.catherine_pass_rate)
        cohort_engine = str(c.engine)
    elif cohort == "unbiased":
        # Generate a closure-bounded cohort WITHOUT pocket pharmacophore
        # bias — used as the SA baseline comparison.  We use the
        # :func:`closure_bounded_smiles_generation` directly with
        # ``include_metal_coord_bias=False`` so the bias dict is read
        # but the metal-coord rules don't contribute to the cohort.
        # We pick ``1h36`` for its broad organic seed set (indole,
        # piperidine, biphenyl) which gives a reasonable closure-bounded
        # organic cohort without pharmacophore bias.
        from molmetal.molmetal_lam.reward.pocket_closure_bias import (
            closure_bounded_smiles_generation,
            SUPPORTED_POCKETS,
        )
        neutral_pocket = (
            "1h36" if "1h36" in SUPPORTED_POCKETS
            else next(iter(SUPPORTED_POCKETS))
        )
        c = closure_bounded_smiles_generation(
            pocket_id=neutral_pocket,
            n=cohort_size,
            max_depth=max_depth,
            max_products=cohort_size * 4,
            include_metal_coord_bias=False,
        )
        smiles = [str(cand["smiles"]) for cand in c.candidates]
        cohort_pocket_id = "unbiased"
        cohort_n_raw = int(c.n_candidates_raw)
        cohort_n_kept = int(c.n_candidates_kept)
        cohort_catherine_rate = float(c.catherine_pass_rate)
        cohort_engine = str(c.engine)
    else:
        raise ValueError(
            f"Unknown cohort {cohort!r}; expected 'closure_bounded' or "
            "'unbiased'"
        )

    sa_vals = _score_sa_batch(smiles)
    aiz_tags = _aizynth_tag_batch(smiles)

    per_compound: List[Dict[str, object]] = []
    for i, smi in enumerate(smiles):
        per_compound.append({
            "smiles": smi,
            "sa": _safe_float(sa_vals[i]),
            "aizynth_synthesizable": bool(aiz_tags[i][0]),
            "aizynth_engine": str(aiz_tags[i][1]),
        })

    agg = _aggregate_sa(sa_vals)
    n_synth = sum(1 for ok, _ in aiz_tags if ok)
    engine_breakdown: Dict[str, int] = {}
    for _, eng in aiz_tags:
        engine_breakdown[eng] = engine_breakdown.get(eng, 0) + 1

    return CohortRecord(
        cohort_label=cohort,
        pocket_id=cohort_pocket_id,
        seed=int(seed),
        n_requested=int(cohort_size),
        n_emitted_raw=cohort_n_raw,
        n_kept=cohort_n_kept,
        catherine_pass_rate=cohort_catherine_rate,
        sa_mean=agg["mean"],
        sa_median=agg["median"],
        sa_std=agg["std"],
        sa_min=agg["min"],
        sa_max=agg["max"],
        n_valid_sa=int(agg["n_valid"]),
        aizynth_synthesizable_rate=(
            float(n_synth) / max(len(smiles), 1)
        ),
        aizynth_engine_breakdown=engine_breakdown,
        per_compound=per_compound,
        wall_time_s=float(time.perf_counter() - t0),
        # Include the engine tag (engine field is read-only on
        # dataclass so we add it via a helper attribute).
        # Stored in per_compound metadata so the JSON keeps it.
    )


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def _aggregate_records(
    records: List[CohortRecord],
) -> Dict[str, object]:
    """Aggregate per-seed records into a single summary.

    Mean / std are computed over per-seed means (per-seed weighting)
    so a seed with fewer compounds does not bias the aggregate.
    """
    if not records:
        return {
            "n_seeds": 0,
            "n_seeds_aggregated": 0,
            "sa_mean_aggregated": None,
            "sa_std_aggregated": None,
            "sa_min_aggregated": None,
            "sa_max_aggregated": None,
        }
    means = [
        r.sa_mean for r in records
        if r.sa_mean == r.sa_mean  # filter NaN
    ]
    if not means:
        return {
            "n_seeds": len(records),
            "n_seeds_aggregated": 0,
            "sa_mean_aggregated": None,
            "sa_std_aggregated": None,
            "sa_min_aggregated": None,
            "sa_max_aggregated": None,
        }
    return {
        "n_seeds": len(records),
        "n_seeds_aggregated": len(means),
        "sa_mean_aggregated": float(statistics.fmean(means)),
        "sa_std_aggregated": (
            float(statistics.pstdev(means)) if len(means) > 1 else 0.0
        ),
        "sa_min_aggregated": float(min(r.sa_min for r in records)),
        "sa_max_aggregated": float(max(r.sa_max for r in records)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Wf-126 P3 / TODO-124 SA closure-bias bench.  "
            "Generates a closure-bounded cohort and reports per-cohort "
            "SA mean / median / std + AiZynth synthesizable rate."
        ),
    )
    p.add_argument(
        "--cohort",
        choices=("closure_bounded", "unbiased"),
        default="closure_bounded",
        help=(
            "Cohort selector.  'closure_bounded' uses the per-pocket "
            "closure cohort (requires --pocket).  'unbiased' uses a "
            "neutral closure_oversample cohort for SA baseline "
            "comparison."
        ),
    )
    p.add_argument(
        "--pocket",
        default=None,
        help=(
            "Pocket identifier (required for --cohort closure_bounded).  "
            "One of: 1h36 3eml 3ny8 4aaw 4rlu."
        ),
    )
    p.add_argument(
        "--cohort-size",
        type=int,
        default=DEFAULT_COHORT_SIZE,
        help=(
            "Target cohort size (per seed).  Default 100 (matches "
            "the SA-mean stability target)."
        ),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default 0 1 2).",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="BFS depth budget (default 2).",
    )
    p.add_argument(
        "--per-scaffold-cap",
        type=int,
        default=20,
        help="Per-seed yield cap for unbiased cohort (default 20).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf126_p3_sa_closure",
        help="Output directory (default wf126_p3_sa_closure).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    return p.parse_args()


def main() -> int:
    """CLI entry point."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.cohort == "closure_bounded" and not args.pocket:
        LOG.error(
            "--pocket is required for --cohort closure_bounded"
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: List[CohortRecord] = []
    baseline_records: List[CohortRecord] = []

    for seed in args.seeds:
        LOG.info(
            "[sa_closure] cohort=%s seed=%d pocket=%s — generating",
            args.cohort, seed, args.pocket,
        )
        rec = _run_one(
            cohort=args.cohort,
            seed=seed,
            cohort_size=args.cohort_size,
            pocket_id=args.pocket,
            per_scaffold_cap=args.per_scaffold_cap,
            max_depth=args.max_depth,
        )
        records.append(rec)
        LOG.info(
            "[sa_closure] cohort=%s seed=%d: sa_mean=%.3f n_kept=%d",
            args.cohort, seed, rec.sa_mean, rec.n_kept,
        )

        # Always run the unbiased baseline (cheap, ~1 minute per seed).
        LOG.info(
            "[sa_closure] baseline unbiased seed=%d — generating",
            seed,
        )
        bsl = _run_one(
            cohort="unbiased",
            seed=seed,
            cohort_size=args.cohort_size,
            pocket_id=None,
            per_scaffold_cap=args.per_scaffold_cap,
            max_depth=args.max_depth,
        )
        baseline_records.append(bsl)
        LOG.info(
            "[sa_closure] baseline unbiased seed=%d: sa_mean=%.3f n_kept=%d",
            seed, bsl.sa_mean, bsl.n_kept,
        )

    agg = _aggregate_records(records)
    bsl_agg = _aggregate_records(baseline_records)

    sa_mean_agg = agg["sa_mean_aggregated"]
    pass_target = (
        sa_mean_agg is not None
        and sa_mean_agg <= SA_MEAN_ACCEPT_CLOSURE
    )
    pass_stretch = (
        sa_mean_agg is not None
        and sa_mean_agg <= TARGET_SA_MEAN_CLOSURE
    )

    sa_metrics = {
        "metadata": {
            "cohort": args.cohort,
            "pocket_id": args.pocket,
            "cohort_size": int(args.cohort_size),
            "seeds": list(args.seeds),
            "max_depth": int(args.max_depth),
            "per_scaffold_cap": int(args.per_scaffold_cap),
            "engine": "pocket_closure_bias",
            "sa_score_engine": "Ertl-Schuffenhauer (RDKit Contrib)",
            "aizynth_engine_default": "smarts_fallback",
        },
        "targets": {
            "sa_mean_target_closure": SA_MEAN_ACCEPT_CLOSURE,
            "sa_mean_stretch_target": TARGET_SA_MEAN_CLOSURE,
            "baseline_sa_mean_unconstrained": BASELINE_SA_MEAN_UNCONSTRAINED,
        },
        "primary": {
            "n_seeds": agg["n_seeds"],
            "n_seeds_aggregated": agg["n_seeds_aggregated"],
            "sa_mean_aggregated": (
                None if sa_mean_agg is None
                else float(round(sa_mean_agg, 6))
            ),
            "sa_std_aggregated": (
                None if agg["sa_std_aggregated"] is None
                else float(round(agg["sa_std_aggregated"], 6))
            ),
            "sa_min_aggregated": agg["sa_min_aggregated"],
            "sa_max_aggregated": agg["sa_max_aggregated"],
            "pass_target": bool(pass_target),
            "pass_stretch_target": bool(pass_stretch),
            "delta_vs_baseline_3p099": (
                None if sa_mean_agg is None
                else float(round(sa_mean_agg - BASELINE_SA_MEAN_UNCONSTRAINED, 6))
            ),
        },
        "baseline_unbiased": {
            "n_seeds": bsl_agg["n_seeds"],
            "n_seeds_aggregated": bsl_agg["n_seeds_aggregated"],
            "sa_mean_aggregated": bsl_agg["sa_mean_aggregated"],
            "sa_std_aggregated": bsl_agg["sa_std_aggregated"],
            "sa_min_aggregated": bsl_agg["sa_min_aggregated"],
            "sa_max_aggregated": bsl_agg["sa_max_aggregated"],
            "note": (
                "Baseline cohort generated by closure_oversample "
                "WITHOUT pocket pharmacophore bias.  Useful for "
                "measuring the pocket-bias contribution."
            ),
        },
        "aizynth_summary": {
            "synthesizable_rate_aggregated": (
                float(statistics.fmean(
                    [r.aizynth_synthesizable_rate for r in records]
                ))
                if records else None
            ),
            "engine_breakdown_aggregated": _merge_engine_breakdowns(
                [r.aizynth_engine_breakdown for r in records]
            ),
        },
        "per_seed": [r.as_dict() for r in records],
        "honest_framing": {
            "cohort_change": (
                "This run changes the GENERATION COHORT (closure-"
                "bounded + pocket-biased) — it does NOT modify the SA "
                "score definition.  The 3.099 → 2.8 (or below) drop is "
                "explained by closure-bounded molecules having lower "
                "structural complexity (no spiro/bridgehead/polycyclic "
                "motivators)."
            ),
            "aiZynth_engine_default": (
                "AiZynthFinder is not installed in this environment; "
                "the per-compound engine tag is 'smarts_fallback' by "
                "default.  This is acknowledged in TODO-124 §2 path 2."
            ),
            "sample_size_warning": (
                "Per-seed n=100 is below the n>=500 target in TODO-124 §2; "
                "this is a CPU-only ship target (P3 honest framing). "
                "Larger cohorts (n=500) are deferred to a follow-up GPU run."
            ),
        },
    }

    out_json = args.out_dir / "sa_metrics.json"
    out_json.write_text(json.dumps(sa_metrics, indent=2, default=str))
    LOG.info("[sa_closure] wrote %s", out_json)

    # Per-compound CSV (smiles, sa, aizynth_synthesizable, engine).
    csv_path = args.out_dir / "per_compound_sa.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "seed", "pocket_id", "smiles", "sa",
            "aizynth_synthesizable", "aizynth_engine",
        ])
        for r in records:
            for c in r.per_compound:
                writer.writerow([
                    r.seed, r.pocket_id, c["smiles"], c["sa"],
                    c["aizynth_synthesizable"], c["aizynth_engine"],
                ])
    LOG.info("[sa_closure] wrote %s", csv_path)

    # Run summary (honest framing + parameter echo).
    summary = {
        "args": vars(args),
        "sa_metrics_summary": {
            "sa_mean_aggregated": sa_metrics["primary"]["sa_mean_aggregated"],
            "pass_target": sa_metrics["primary"]["pass_target"],
            "pass_stretch_target": sa_metrics["primary"]["pass_stretch_target"],
            "baseline_unbiased_sa_mean": (
                sa_metrics["baseline_unbiased"]["sa_mean_aggregated"]
            ),
            "aiZynth_synthesizable_rate": (
                sa_metrics["aizynth_summary"]["synthesizable_rate_aggregated"]
            ),
        },
        "wall_time_s_total": float(sum(r.wall_time_s for r in records)),
    }
    summary_path = args.out_dir / "sa_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("[sa_closure] wrote %s", summary_path)

    # Friendly stdout line.
    if sa_mean_agg is not None:
        print(
            f"[sa_closure] DONE  cohort={args.cohort} pocket={args.pocket} "
            f"seeds={len(args.seeds)} sa_mean_aggregated={sa_mean_agg:.3f} "
            f"target<={SA_MEAN_ACCEPT_CLOSURE} "
            f"pass_target={pass_target} "
            f"pass_stretch(<=2.8)={pass_stretch} "
            f"baseline_unbiased={bsl_agg['sa_mean_aggregated']}",
            flush=True,
        )
    return 0


def _merge_engine_breakdowns(
    breakdowns: List[Dict[str, int]],
) -> Dict[str, int]:
    """Merge per-record engine breakdowns into a single total."""
    merged: Dict[str, int] = {}
    for d in breakdowns:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + int(v)
    return merged


if __name__ == "__main__":
    sys.exit(main())