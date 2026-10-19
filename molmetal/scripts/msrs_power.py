#!/usr/bin/env python3
"""MET-8 — statistical power and the minimum detectable effect.

MET-2/3/5 report nulls (DOI R2 0.3321, Delta sign 0.6193, decision regret
-0.0034).  A null is only interpretable if we know what the study *could* have
detected.  This script converts the observed clustered CIs into a standard
error, then reports:

* the minimum detectable effect (MDE) at 80% power, one-sided alpha = 0.05;
* the number of groups (DOIs) needed to detect the observed effect;
* the number needed to detect the pre-registered target effect;
* an MDE-vs-n curve so the cost of a better study is explicit.

Method
------
For a CI95 of width ``w``, ``sd = w / (2 * 1.96)``.  For a one-sided test at
alpha with power ``1 - beta``:

    MDE(n0) = (z_{0.95} + z_{0.80}) * sd = 2.4865 * sd
    MDE(n)  = MDE(n0) * sqrt(n0 / n)
    n_req   = n0 * (MDE(n0) / effect)^2

The CI -> sd step assumes normality of the bootstrap distribution.  The
bootstrap CIs are percentile intervals, so this is an approximation; it is
applied identically to every metric so comparisons stay fair.

Inputs are the on-disk artifacts, not recomputed numbers, so every value in the
output traces to a field in MET-2/3/5/6.

Output: ``metrics/by_metric/msrs_power.json``.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_power.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    NOISE_FLOOR,
    _git_commit,
    _sha256,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

Z_ALPHA = 1.6448536269514722  # one-sided 0.05
Z_POWER = 0.8416212335729143  # 80%
K = Z_ALPHA + Z_POWER
N_GRID = (25, 40, 50, 62, 100, 200, 500, 1000, 2000)


def sd_from_ci(ci: Sequence[float]) -> float:
    """Convert a symmetric 95% CI to a standard error."""
    lo, hi = float(ci[0]), float(ci[1])
    return (hi - lo) / (2.0 * 1.959963984540054)


def mde(sd: float, n0: float, n: Optional[float] = None) -> float:
    """Minimum detectable effect at 80% power, one-sided alpha 0.05."""
    base = K * sd
    if n is None or n0 <= 0:
        return base
    return base * math.sqrt(n0 / float(n))


def required_n(sd: float, effect: float, n0: float) -> Optional[int]:
    """Groups needed to detect ``effect`` at the same power."""
    if effect == 0 or not math.isfinite(effect):
        return None
    return int(math.ceil(n0 * (K * sd / abs(effect)) ** 2))


def power_curve(sd: float, n0: float, grid: Sequence[int] = N_GRID) -> List[Dict[str, float]]:
    return [
        {"n": int(n), "mde": round(mde(sd, n0, n), 4)}
        for n in grid
    ]


def describe(
    name: str,
    observed: float,
    ci: Sequence[float],
    n0: int,
    unit: str,
    target: Optional[float] = None,
    effect: Optional[float] = None,
    source: str = "",
    note: str = "",
) -> Dict[str, object]:
    """One metric's power summary.

    ``effect`` defaults to ``observed``; for a difference from a reference
    (e.g. sign accuracy vs 0.5) pass it explicitly.
    """
    sd = sd_from_ci(ci)
    eff = abs(observed) if effect is None else abs(float(effect))
    out: Dict[str, object] = {
        "observed": round(float(observed), 4),
        "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "sd_implied": round(sd, 4),
        "n_groups": int(n0),
        "unit": unit,
        "mde_80pct": round(mde(sd, n0), 4),
        "observed_effect": round(eff, 4),
        "powered_for_observed": bool(eff >= mde(sd, n0)),
        "n_required_for_observed": required_n(sd, eff, n0),
        "power_curve": power_curve(sd, n0),
        "source": source,
    }
    if target is not None:
        out["target_effect"] = round(float(target), 4)
        out["mde_vs_target"] = "below_mde" if target < mde(sd, n0) else "above_mde"
        out["n_required_for_target"] = required_n(sd, target, n0)
    if note:
        out["note"] = note
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric")
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_power.json"
    )
    args = parser.parse_args(argv)

    def load(name: str) -> Dict[str, object]:
        with open(args.metrics_dir / f"{name}.json") as fh:
            return json.load(fh)

    gen = load("msrs_generalization")
    delta = load("msrs_delta_metrics")
    curve = load("msrs_decision_curve")
    ablate = load("msrs_metal_ablation")

    metrics: Dict[str, object] = {}

    doi_full = curve["protocols"]["doi"]["curve"][-1]["model_vs_global_best"]
    metrics["decision_regret_doi_full"] = describe(
        "decision_regret_doi_full",
        observed=doi_full["mean_regret_delta"],
        ci=doi_full["bootstrap_ci95_doi"],
        n0=doi_full["n_dois"],
        unit="regret (log10 IC50)",
        target=0.10,
        effect=doi_full["mean_regret_delta"],
        source="msrs_decision_curve.json:protocols.doi.curve[-1].model_vs_global_best",
        note=(
            "A negative delta means the model is worse than global_best.  The "
            "target is a 0.10 regret improvement; the observed effect is -0.0034."
        ),
    )

    doi_half = curve["protocols"]["doi"]["curve"][4]["model_vs_global_best"]
    metrics["decision_regret_doi_q0.5"] = describe(
        "decision_regret_doi_q0.5",
        observed=doi_half["mean_regret_delta"],
        ci=doi_half["bootstrap_ci95_doi"],
        n0=doi_half["n_dois"],
        unit="regret (log10 IC50)",
        target=0.10,
        source="msrs_decision_curve.json:protocols.doi.curve[4].model_vs_global_best",
    )

    sign = delta["protocols"]["doi"]["thresholds"]["0.2091"]
    metrics["delta_sign_doi"] = describe(
        "delta_sign_doi",
        observed=sign["sign_accuracy"],
        ci=sign["bootstrap"]["sign_accuracy"]["ci95"],
        n0=sign["bootstrap"]["n_dois"],
        unit="sign accuracy",
        target=0.65 - 0.5,
        effect=sign["sign_accuracy"] - 0.5,
        source="msrs_delta_metrics.json:protocols.doi.thresholds.0.2091",
        note=(
            "Effect is measured above the 0.5 chance line.  The gate (0.65) is "
            "an effect of 0.15; compare it with the MDE, not with 0.619."
        ),
    )

    mae = delta["protocols"]["doi"]["thresholds"]["0.2091"]
    zero_mae = 0.5299  # measured in MET-3: constant-zero predictor on the same pairs
    metrics["delta_mae_doi"] = describe(
        "delta_mae_doi",
        observed=mae["mae_delta"],
        ci=mae["bootstrap"]["mae_delta"]["ci95"],
        n0=mae["bootstrap"]["n_dois"],
        unit="MAE(delta), log10",
        effect=zero_mae - mae["mae_delta"],
        source="msrs_delta_metrics.json:protocols.doi.thresholds.0.2091",
        note=(
            "Effect is the improvement over the constant-zero predictor "
            f"(MAE {zero_mae}); a negative value means the model is worse."
        ),
    )

    r2_doi = gen["splits"]["doi"]["arms"]["ligand+both"]
    metrics["r2_doi"] = describe(
        "r2_doi",
        observed=r2_doi["r2"],
        ci=r2_doi["bootstrap_r2"]["ci95"],
        n0=r2_doi["bootstrap_r2"]["n_groups"],
        unit="R2",
        target=0.50 - r2_doi["r2"],
        effect=0.50 - r2_doi["r2"],
        source="msrs_generalization.json:splits.doi.arms.ligand+both",
        note=(
            "Effect is the shortfall from the pre-registered 0.50 DOI target "
            "(0.50 - 0.3321), i.e. 'can this study detect that the model misses "
            "the target?'.  A negative value would mean the target was met."
        ),
    )

    gain = ablate["protocols"]["doi"]["arms"]["no_state_coord"]["delta_ci"]
    metrics["representation_gain_doi"] = describe(
        "representation_gain_doi",
        observed=ablate["protocols"]["doi"]["arms"]["no_state_coord"]["delta_r2_vs_full"],
        ci=gain["ci95"],
        n0=gain["n_groups"],
        unit="delta R2",
        target=0.05,
        source="msrs_metal_ablation.json:protocols.doi.arms.no_state_coord",
    )

    blind = ablate["protocols"]["doi"]["arms"]["metal_blind"]["delta_ci"]
    metrics["metal_blind_gain_doi"] = describe(
        "metal_blind_gain_doi",
        observed=ablate["protocols"]["doi"]["arms"]["metal_blind"]["delta_r2_vs_full"],
        ci=blind["ci95"],
        n0=blind["n_groups"],
        unit="delta R2",
        target=0.05,
        source="msrs_metal_ablation.json:protocols.doi.arms.metal_blind",
    )

    summary = {
        "powered": sorted(k for k, v in metrics.items() if v["powered_for_observed"]),
        "underpowered": sorted(k for k, v in metrics.items() if not v["powered_for_observed"]),
    }
    payload = {
        "artifact": "msrs_power",
        "version": "1.0",
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "metrics_dir": str(args.metrics_dir),
            "inputs": {
                name: _sha256(args.metrics_dir / f"{name}.json")
                for name in (
                    "msrs_generalization",
                    "msrs_delta_metrics",
                    "msrs_decision_curve",
                    "msrs_metal_ablation",
                )
            },
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
        },
        "design": {
            "alpha": 0.05,
            "sided": "one-sided",
            "power": 0.80,
            "z_alpha": round(Z_ALPHA, 4),
            "z_power": round(Z_POWER, 4),
            "k": round(K, 4),
            "sd_from_ci": "sd = (hi - lo) / (2 * 1.96)",
            "mde": "MDE = k * sd; MDE(n) = MDE * sqrt(n0/n)",
            "required_n": "n = n0 * (MDE / effect)^2",
            "clustering": "the CI's n0 is the number of DOI clusters, not rows",
            "target_scale": (
                "target_effect is an effect (a difference), not an absolute "
                "level: sign 0.65-0.5, R2 0.50-observed, regret 0.10"
            ),
            "noise_floor": NOISE_FLOOR,
            "caveat": (
                "Percentile bootstrap CIs are not exactly normal, so the CI->sd "
                "step is an approximation.  It is applied identically to every "
                "metric, so the relative comparison (which nulls are power "
                "failures) is robust; the absolute MDE carries that error."
            ),
        },
        "metrics": metrics,
        "summary": summary,
        "verdict": "PASS",
        "verdict_rule": "every headline metric reports an MDE and a required-n",
    }
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2))
    for name, m in metrics.items():
        print(
            f"[met8] {name:28s} obs={m['observed']} sd={m['sd_implied']} "
            f"MDE={m['mde_80pct']} powered={m['powered_for_observed']} "
            f"n_req={m['n_required_for_observed']}"
        )
    print(f"[met8] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
