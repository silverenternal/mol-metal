#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What magnitude can the velocity field emit at each ``t``, before any model?

WHY THIS EXISTS
---------------
Gate M4 used to demand ``|v_pred| / |v_target|`` in ``[0.5, 2.0]`` at every
``t`` in ``{0.05, 0.25, 0.5, 0.75, 0.95}``.  That criterion is not about the
model: the field sees only ``x_t = (1-t) x0 + t x1``, so the best it can do in
MSE is the conditional mean ``E[x1 - x0 | x_t]``, and at the midpoint of this
interpolant the conditional mean is nearly zero because ``x0`` and ``x1``
contribute equally and independently.  A gate written against
``|x1 - x0|`` therefore asks for magnitude that the Bayes-optimal field does
not have.

This script measures the reference the gate needs, with no network involved:

* ``bayes_ratio(t)`` — through-origin regression of ``x1 - x0`` on ``x_t``
  over Monte-Carlo noise draws, and the resulting
  ``|pred| / |target|``;
* ``signal_share(t)`` — ``1 - MSE(pred) / MSE(0)``, i.e. how much of the
  target is predictable at that ``t``;
* ``sampler_coverage`` — for each t-sampler the trainer can be configured
  with, the mean signal share it visits and the fraction of its mass that
  lands where less than 5% of the target is predictable.

MEASURED (8-cloud val cohort, 405 atoms, sigma 2.485, data coord std 2.36):
``bayes_ratio`` is 0.69 at t=0.05, **0.054 at t=0.5**, 0.65 at t=0.95, and
the best possible field reaches **99.7% of the v=0 MSE at t=0.5**.  Uniform
t-sampling puts **20.1%** of its mass in the sub-5%-signal region; the
trainer's ``log_normal`` default (mu -0.5, sigma 1.0) makes that **26.6%**
worse, because it moves mass toward t~0.40; ``sigmoid(N(0, 3))`` cuts it to
**10.6%**.

Usage:
  python molmetal/scripts/wfa108j_bayes_ratio.py \
      --out metrics/by_metric/wfa108j_bayes_ratio_curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_CSV = ".data/molmetal/metalcytotox_5metal_multitask_v3.csv"


def _cohort(csv: str, n_clouds: int, seed: int) -> torch.Tensor:
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_csv_cohort

    clouds = build_csv_cohort(csv, split="val", n_clouds=int(n_clouds), seed=int(seed))
    return torch.cat([c.positions.float() for c in clouds], 0)


def _curve(x1: torch.Tensor, sigma: float, ts: List[float],
           n_draws: int, seed: int) -> List[Dict[str, float]]:
    g = torch.Generator().manual_seed(int(seed))
    rows: List[Dict[str, float]] = []
    for t in ts:
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for _ in range(int(n_draws)):
            x0 = torch.randn(x1.shape, generator=g) * float(sigma)
            xs.append(((1.0 - t) * x0 + t * x1).reshape(-1))
            ys.append((x1 - x0).reshape(-1))
        x = torch.cat(xs)
        y = torch.cat(ys)
        slope = float((x * y).sum() / (x * x).sum().clamp_min(1e-12))
        pred = slope * x
        mse = float(F.mse_loss(pred, y))
        mse0 = float(F.mse_loss(torch.zeros_like(y), y))
        rows.append(
            {
                "t": float(t),
                "slope": slope,
                "bayes_ratio": float(pred.norm() / y.norm().clamp_min(1e-12)),
                "mse": mse,
                "mse_zero": mse0,
                "signal_share": 1.0 - mse / max(mse0, 1e-12),
            }
        )
    return rows


def _coverage(curve: List[Dict[str, float]], ts: torch.Tensor) -> Dict[str, float]:
    grid = torch.tensor([r["t"] for r in curve])
    share = torch.tensor([r["signal_share"] for r in curve])
    idx = (ts.clamp(0.0, 1.0) * (len(curve) - 1)).round().long().clamp(0, len(curve) - 1)
    sampled = share[idx]
    return {
        "mean_signal_share": float(sampled.mean()),
        "frac_t_with_signal_below_5pct": float((sampled < 0.05).float().mean()),
        "mean_t": float(ts.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--n-clouds", type=int, default=8)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=2.485)
    ap.add_argument("--n-t", type=int, default=21)
    ap.add_argument("--n-draws", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default="metrics/by_metric/wfa108j_bayes_ratio_curve.json"
    )
    args = ap.parse_args()

    x1 = _cohort(args.csv, args.n_clouds, args.val_seed)
    ts = [i / float(args.n_t - 1) for i in range(int(args.n_t))]
    curve = _curve(x1, args.sigma, ts, args.n_draws, args.seed)

    gg = torch.Generator().manual_seed(int(args.seed) + 1)
    n = 200_000
    # Keys carry no "." because the doc's machine check resolves dot paths.
    samplers = {
        "uniform": torch.rand(n, generator=gg),
        "log_normal_mu_neg0p5_sigma1p0": torch.sigmoid(
            -0.5 + 1.0 * torch.randn(n, generator=gg)
        ),
        "log_normal_mu0_sigma1p0": torch.sigmoid(
            0.0 + 1.0 * torch.randn(n, generator=gg)
        ),
        "log_normal_mu0_sigma2p0": torch.sigmoid(
            0.0 + 2.0 * torch.randn(n, generator=gg)
        ),
        "log_normal_mu0_sigma3p0": torch.sigmoid(
            0.0 + 3.0 * torch.randn(n, generator=gg)
        ),
    }
    coverage = {k: _coverage(curve, v) for k, v in samplers.items()}

    payload: Dict[str, Any] = {
        "metric": "wfa108j_bayes_ratio_curve",
        "tag": "MEASURED",
        "date": "2026-10-25",
        "what": (
            "The magnitude any model can emit at each t, measured from the "
            "interpolant alone (no network): through-origin regression of "
            "x1-x0 on x_t over Monte-Carlo noise draws, plus the signal share "
            "1 - MSE/MSE(v=0), plus what fraction of each available "
            "t-sampler's mass lands in the sub-5%-signal region."
        ),
        "script": "molmetal/scripts/wfa108j_bayes_ratio.py",
        "csv": str(args.csv),
        "n_clouds": int(args.n_clouds),
        "n_atoms": int(x1.shape[0]),
        "sigma": float(args.sigma),
        "coord_std": float(x1.std()),
        "n_t": int(args.n_t),
        "n_draws": int(args.n_draws),
        "seed": int(args.seed),
        "curve": curve,
        "sampler_coverage": coverage,
        "by_t": {
            f"t{int(round(r['t'] * 100)):03d}": {
                "bayes_ratio": r["bayes_ratio"],
                "signal_share": r["signal_share"],
            }
            for r in curve
        },
    }
    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    for r in curve:
        print(
            f"[wfa108j] t={r['t']:.2f} bayes_ratio={r['bayes_ratio']:.4f} "
            f"signal={100*r['signal_share']:5.1f}% mse={r['mse']:.4f} "
            f"(v=0 {r['mse_zero']:.4f})"
        )
    for k, v in coverage.items():
        print(
            f"[wfa108j] {k:32s} mean_signal={v['mean_signal_share']:.3f} "
            f"below_5pct={100*v['frac_t_with_signal_below_5pct']:.1f}%"
        )
    print(f"[wfa108j] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
