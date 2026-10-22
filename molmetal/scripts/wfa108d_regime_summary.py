# SPDX-License-Identifier: Apache-2.0
"""Wf-A108d — summarise the regime-gate arms into one citable artifact.

WHY THIS EXISTS
---------------
``wfa107h_velocity_capacity.py`` writes one JSON per arm, and Wf-A108d ran ~20
of them.  Quoting a number then means naming a file, which is how the same
metric ends up with two different values in two documents.  This script reads
the arms back and emits ONE table with the baseline, the reached value and the
fraction of the zero-prediction gap that was closed, so every claim in the
catalog and the survey points at a single file.

Definition used throughout:

    gap_closed = (mse_zero - mse_reached) / mse_zero

It is deliberately relative to the ``v = 0`` predictor rather than to the
radial ceiling: the radial ceiling in ``wfa107h`` is computed from the TARGET
(``coeff = (tgt . unit).sum()``), i.e. it is an oracle bound, not a result any
model can reach by reading its inputs.

Usage:
  python molmetal/scripts/wfa108d_regime_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_METRICS = _REPO / "metrics" / "by_metric"

# (arm name, file, what it varies)
ARMS = [
    ("single_batch_vc_125", "wfa108d_cycle0_vc_125.json", "single cohort, t=0.5"),
    ("cycle1_tuniform_vc_125", "wfa108d_cycle1_tuniform_vc_125.json", "1 cohort, t~U"),
    ("cycle25_tfix_vc_125", "wfa108d_cycle25_tfix_vc_125.json", "25 cohorts, t=0.5"),
    ("cycle25_tuniform_vc_125", "wfa108d_cycle25_tuniform_vc_125.json", "25 cohorts, t~U"),
    ("cycle25_tuniform_vc_250", "wfa108d_cycle25_tuniform_vc_250.json", "25 cohorts, t~U"),
    ("cycle25_tuniform_vc_500", "wfa108d_cycle25_tuniform_vc_500.json", "25 cohorts, t~U"),
    ("cycle25_tuniform_vc_1000", "wfa108d_cycle25_tuniform_vc_1000.json", "25 cohorts, t~U"),
    ("cycle25_tuniform_vc_2000", "wfa108d_cycle25_tuniform_vc_2000.json", "25 cohorts, t~U"),
    ("cycle25_tuniform_radial_1000", "wfa108d_cycle25_tuniform_radial_1000.json", "radial head, 25 cohorts, t~U"),
    ("ho_cycle25_vc_500", "wfa108d_ho_cycle25_vc_500.json", "held-out eval, 25 cohorts"),
    ("ho_cycle100_vc_500", "wfa108d_ho_cycle100_vc_500.json", "held-out eval, 100 cohorts"),
    ("ho_cycle25_vc_2000", "wfa108d_ho_cycle25_vc_2000.json", "held-out eval, 25 cohorts"),
    ("ho_cycle25_vc_10000", "wfa108d_ho_cycle25_vc_10000.json", "held-out eval, 25 cohorts"),
    ("ho_cycle25_vc32_2000", "wfa108d_ho_cycle25_vc32_2000.json", "held-out eval, C=32"),
    ("ho_cycle25_vc_temb64_500", "wfa108d_ho_cycle25_vc_temb64_500.json", "held-out eval, sinusoidal t"),
    ("ho_cycle25_vc_temb64_2000", "wfa108d_ho_cycle25_vc_temb64_2000.json", "held-out eval, sinusoidal t"),
]


def _row(name: str, fname: str, what: str) -> Dict[str, Any]:
    path = _METRICS / fname
    if not path.exists():
        return {"arm": name, "file": fname, "missing": True}
    d = json.loads(path.read_text())
    last = d["trajectory"][-1]
    zero = float(d["zero_baseline_mse"])
    reached = float(last["mse"])
    return {
        "arm": name,
        "file": fname,
        "varies": what,
        "velocity_head": d.get("velocity_head"),
        "vector_channels": d.get("vector_channels"),
        "t_embedding_dim": d.get("t_embedding_dim", 0),
        "cycle_batches": d.get("cycle_batches"),
        "t_uniform": d.get("t_uniform"),
        "eval_seed": d.get("eval_seed"),
        "steps": d.get("steps"),
        "lr": d.get("lr"),
        "mse_zero": zero,
        "mse_reached": reached,
        "gap_closed": (zero - reached) / zero,
        "cosine": float(last["cosine"]),
        "mean_v_norm": float(last["mean_v_norm"]),
    }


def main() -> int:
    rows = [_row(*a) for a in ARMS]
    tsen: List[Dict[str, Any]] = []
    for name, fname in (
        ("vc8_10k", "wfa108d_t_sensitivity.json"),
        ("vc8_temb64_2k", "wfa108d_t_sensitivity_temb64.json"),
        ("trainer_ckpt_epoch4", "wfa108d_t_sensitivity_trainer_ckpt.json"),
    ):
        p = _METRICS / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        tsen.append(
            {
                "field": name,
                "file": fname,
                "endpoint_cosine": d["endpoint_cosine"],
                "endpoint_relative_l2_change": d["endpoint_relative_l2_change"],
                "sign_flip_achieved": d["sign_flip_achieved"],
                "per_t": d["per_t"],
            }
        )

    doc = {
        "metric": "wfa108d_regime_gate",
        "tag": "MEASURED",
        "date": "2026-10-22",
        "what": (
            "Wf-A108d: the Wf-A108 capacity gate fitted ONE 8-cloud batch at ONE "
            "t, which measures memorisation.  These arms train the same field on "
            "the trainer's own sampling distribution (N cohorts cycling, t ~ "
            "U(0,1) per cloud per step) and read the result on a fixed cohort, "
            "so 'the head can fit this batch' and 'the trainer's per-step "
            "gradient carries the information' are separated."
        ),
        "gap_definition": "gap_closed = (mse_zero - mse_reached) / mse_zero",
        "arms": rows,
        "t_sensitivity": tsen,
        "honest_limits": [
            "Run-to-run GPU nondeterminism is NOT zero: the identical "
            "single-cohort 125-step command returned 7.120 in one process and "
            "7.957 in another (gap_closed 0.332 vs 0.253), so differences below "
            "~0.8 MSE at 125 steps are not effects.  The long arms differ by "
            "much more than that.",
            "The held-out arms use eval_seed 999, where the cohort builder "
            "skips 1 of 8 rows as un-embeddable, so the held-out cohort has 7 "
            "clouds, not 8.  The zero baseline differs (9.008 vs 10.652) and "
            "held-out numbers are NOT comparable to in-rotation numbers.",
            "The radial ceiling reported by wfa107h is an ORACLE bound (it uses "
            "the target to pick the coefficient), so it is not a reachable "
            "number and is not used as a denominator here.",
            "Each arm is one seed.  The 'null' readings (C=32, 100 cohorts, "
            "sinusoidal t) are nulls against a single-seed spread, not "
            "equivalence tests.",
        ],
        "source_files": [f"metrics/by_metric/{a[1]}" for a in ARMS],
    }
    out = _METRICS / "wfa108d_regime_gate.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"{'arm':32s} {'steps':>6} {'zero':>7} {'reached':>8} {'gap':>7}")
    for r in rows:
        if r.get("missing"):
            print(f"{r['arm']:32s}  MISSING {r['file']}")
            continue
        print(
            f"{r['arm']:32s} {r['steps']:6d} {r['mse_zero']:7.3f} "
            f"{r['mse_reached']:8.3f} {r['gap_closed']:7.3f}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
