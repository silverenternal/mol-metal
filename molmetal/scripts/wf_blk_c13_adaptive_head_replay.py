# SPDX-License-Identifier: Apache-2.0
"""Replay candidate adaptive bond-head controllers over RECORDED runs.

WHY
---
MEASURED on molmetal/reports/cfm_fm_control_2026_10_09 (29 epochs, n_train
256) — the two objectives move in opposite directions:

    ep 6    train_loss 1.473   gen_validity_val 48 (peak)   bond_f1 0.438
    ep 16   train_loss 1.168   gen_validity_val 41          bond_f1 0.570
    ep 21   train_loss 1.090   gen_validity_val  0          bond_f1 0.647
    ep 29   train_loss 0.968   gen_validity_val  0          bond_f1 0.784

`train_loss` falls monotonically for all 29 epochs at 0.7-2.0 %/epoch and
NEVER plateaus. So the literature-standard adaptive thaw ("unfreeze on a
validation-loss plateau") would never fire here and would hold the head
frozen forever — strictly worse than the fixed 5-epoch warmup it replaces.
That is the reason this replay exists rather than a guess.

Worse, from ep21 on the loss keeps improving while generation has been zero
for a third of the run. `train_loss` is not a proxy for generation quality,
so any controller keyed on it steers the wrong way.

`gen_validity_val` is the signal that actually moves (48 -> 0) and it moves
in the direction that matters.

WHAT THIS DOES
--------------
Replays candidate controllers over the per-epoch CSVs ALREADY ON DISK. No
GPU, no training. The question is narrow and mechanical: does a candidate
produce a NON-DEGENERATE weight schedule that keeps the head at full weight
through the healthy window and backs off only once generation degrades?

This is a MECHANISM check, not a performance result. The controllers are
tuned against the same two runs they are replayed on, so nothing here says
any of them improves a metric. Two runs is two data points.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNS = {
    "control": "molmetal/reports/cfm_fm_control_2026_10_09/per_epoch_metrics.csv",
    "fm_ab": "molmetal/reports/cfm_fm_ab_2026_10_09/per_epoch_metrics.csv",
}


def load_run(rel: str) -> Dict[str, np.ndarray]:
    rows = list(csv.DictReader((REPO / rel).open(encoding="utf-8")))
    def col(name, default=np.nan):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (KeyError, TypeError, ValueError):
                out.append(default)
        return np.array(out)
    gv = col("gen_validity_val")
    gv_n = col("gen_validity_n")
    # gen_validity_val is a COUNT. Normalise before it is used as a ratio, so
    # the controller cannot silently depend on the eval sample count.
    denom = np.where(np.isfinite(gv_n) & (gv_n > 0), gv_n, np.nan)
    return {
        "epoch": col("epoch"),
        "train_loss": col("train_loss"),
        "gv_frac": np.where(np.isfinite(denom), gv / denom, np.nan),
        "bond_f1": col("bond_f1"),
        "bond_ce": col("bond_ce"),
    }


# --------------------------------------------------------------------------
# Candidate controllers. Each maps the observed history -> a bond-head weight
# for the NEXT epoch. Signature: (gv_frac_history, bond_f1_history) -> float.
# --------------------------------------------------------------------------

def ctrl_fixed(w_max: float = 1.0) -> Callable:
    """The status quo: constant weight, fixed 5-epoch warmup handled elsewhere.

    Included as the null hypothesis. A candidate that cannot beat this on
    non-degeneracy is not worth running.
    """
    def f(gv, f1):
        return w_max
    return f


def ctrl_linear_ratio(floor: float = 0.2) -> Callable:
    """Head weight proportional to current generation validity.

    w = floor + (1-floor) * gv / gv_peak, where gv_peak is the running max.

    Keeps full weight while generation is healthy and backs off linearly as
    it degrades. The floor is what stops the controller from deleting the
    bond objective outright — BLK-C12 already recorded what a term driven to
    zero does to the loss.
    """
    def f(gv, f1):
        if not len(gv) or not np.isfinite(gv[-1]):
            return 1.0
        peak = np.nanmax(gv)
        if not np.isfinite(peak) or peak <= 0:
            return 1.0
        return floor + (1.0 - floor) * float(gv[-1]) / float(peak)
    return f


def ctrl_ema_ratio(floor: float = 0.2, beta: float = 0.3) -> Callable:
    """Same shape, but on a smoothed validity signal.

    gen_validity_val is noisy epoch to epoch (46,46,46,45,44,47,48,...), so
    an unsmoothed ratio makes the head weight jitter and the effective
    schedule stops being interpretable.
    """
    def f(gv, f1):
        if not len(gv) or not np.isfinite(gv[-1]):
            return 1.0
        ema = gv[0]
        for v in gv:
            if np.isfinite(v):
                ema = beta * v + (1 - beta) * ema
        peak = np.nanmax(gv)
        if not np.isfinite(peak) or peak <= 0:
            return 1.0
        return floor + (1.0 - floor) * float(ema) / float(peak)
    return f


def ctrl_hysteresis(
    floor: float = 0.2, on_thresh: float = 0.8, off_thresh: float = 0.4
) -> Callable:
    """Two-threshold version: hold full weight until validity falls BELOW
    `on_thresh` of its peak, then drop to the floor once it falls below
    `off_thresh`. Deadband between the two.

    Rationale: during the healthy window the ratio controller still wobbles
    the weight around as validity wobbles, and the head's own convergence is
    fast enough that weight noise costs more than the fine-tuning is worth.
    """
    state = {"on": True}

    def f(gv, f1):
        if not len(gv) or not np.isfinite(gv[-1]):
            return 1.0
        peak = np.nanmax(gv)
        if not np.isfinite(peak) or peak <= 0:
            return 1.0
        r = float(gv[-1]) / float(peak)
        if state["on"] and r < on_thresh:
            state["on"] = False
        elif (not state["on"]) and r > off_thresh:
            state["on"] = True
        return 1.0 if state["on"] else floor
    return f


CANDIDATES = {
    "fixed_null": ctrl_fixed(),
    "linear_ratio": ctrl_linear_ratio(),
    "ema_ratio": ctrl_ema_ratio(),
    "hysteresis": ctrl_hysteresis(),
}


def replay(ctrl: Callable, run: Dict[str, np.ndarray]) -> Dict:
    gvs, f1s, weights = [], [], []
    for i in range(len(run["epoch"])):
        w = float(ctrl(np.array(gvs), np.array(f1s)))
        weights.append(w)
        gvs.append(run["gv_frac"][i])
        f1s.append(run["bond_f1"][i])
    w = np.array(weights)
    ep = run["epoch"].astype(int)
    gv = run["gv_frac"]
    # "healthy window" = epochs where generation validity is at least 90% of
    # its own peak. Derived from the run, not assumed.
    peak = np.nanmax(gv)
    healthy = np.isfinite(gv) & (gv >= 0.9 * peak) if np.isfinite(peak) else np.zeros_like(gv, bool)
    # Collapse window = validity has hit exactly zero at some point after the
    # peak. Derived, not assumed.
    zeroed = np.isfinite(gv) & (gv <= 0.0)
    after_zero = zeroed.copy()
    if zeroed.any():
        idx = int(np.argmax(zeroed))
        after_zero[:idx] = False
    return {
        "weights": [round(float(x), 4) for x in w],
        "n_distinct": int(len(set(np.round(w, 3)))),
        "weight_range": [round(float(w.min()), 4), round(float(w.max()), 4)],
        "mean_w_healthy": round(float(w[healthy].mean()), 4) if healthy.any() else None,
        "mean_w_after_collapse": round(float(w[after_zero].mean()), 4) if after_zero.any() else None,
        "first_epoch_weight_below_0.9": (
            int(ep[np.argmax(w < 0.9)]) if (w < 0.9).any() else None
        ),
        "mean_abs_epoch_delta": round(float(np.abs(np.diff(w)).mean()), 5) if len(w) > 1 else 0.0,
    }


def main() -> int:
    runs = {name: load_run(rel) for name, rel in RUNS.items()}
    out = {
        "purpose": (
            "Mechanism check only. Controllers are replayed over runs already "
            "on disk; no GPU, no training, no performance claim."
        ),
        "loss_plateau_criterion_verdict": {
            "question": "Would the literature-standard 'unfreeze on train_loss plateau' fire here?",
            "answer": "NO",
            "evidence": {
                name: {
                    "relative_improvement_pct_per_epoch": [
                        round(float(abs(run['train_loss'][i] - run['train_loss'][i - 1])
                                    / abs(run['train_loss'][i - 1]) * 100), 3)
                        for i in range(1, len(run['epoch']))
                    ],
                    "epochs": int(len(run['epoch'])),
                }
                for name, run in runs.items()
            },
            "note": (
                "train_loss decreases 0.7-2.0 %/epoch for the whole run and never "
                "plateaus, so a loss-plateau trigger would hold the head frozen "
                "forever. This is why the key is gen_validity, not the loss."
            ),
        },
        "generation_validity_signal": {
            name: {
                "gv_frac": [round(float(x), 4) for x in run["gv_frac"]],
                "epochs": run["epoch"].astype(int).tolist(),
                "peak": round(float(np.nanmax(run["gv_frac"])), 4),
                "n_zero_epochs": int(np.nansum(run["gv_frac"] <= 0.0)),
            }
            for name, run in runs.items()
        },
        "candidates": {},
    }
    for cname, ctrl in CANDIDATES.items():
        out["candidates"][cname] = {rname: replay(ctrl, run) for rname, run in runs.items()}

    print(f"{'candidate':<14}{'run':<9}{'distinct':<10}{'range':<18}"
          f"{'w_healthy':<11}{'w_collapse':<12}{'first<0.9':<10}{'|dw|'}")
    for cname, per_run in out["candidates"].items():
        for rname, d in per_run.items():
            print(f"{cname:<14}{rname:<9}{d['n_distinct']:<10}"
                  f"{str(d['weight_range']):<18}{str(d['mean_w_healthy']):<11}"
                  f"{str(d['mean_w_after_collapse']):<12}"
                  f"{str(d['first_epoch_weight_below_0.9']):<10}"
                  f"{d['mean_abs_epoch_delta']}")

    dest = REPO / "metrics" / "by_metric" / "blk_c13_adaptive_head_replay.json"
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(str(dest), json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())