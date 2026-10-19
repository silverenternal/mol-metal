#!/usr/bin/env python3
"""Wf-A96 (2026-10-19) -- OPT-5 ring-size gate.

WHY THIS EXISTS
---------------
cfm_todo.json::OPT-5_ring_template_prior shipped the prior (Wf-A78) behind
flow.ring_template_weight but left the measurement open: "GPU 5-epoch smoke
+ ring-size MEASURED gate (rings/mol, 3-ring share, 5/6-ring share on
generated output)".  The trainer already emits the three numbers every
epoch (EpochMetrics.gen_rings_per_mol + gen_ring_size_distribution via
molmetal/eval/ring_profile.py); this script turns them into a verdict
against the two references that matter:

  real tmQM          rings/mol 4.193, 5/6 share 0.969, 3-ring share 0.0049
                     (OPT-2, molmetal/reports/cfm_opt2_ring_content_2026_10_13)
  pre-prior gen      rings/mol 1.636, 5/6 share 0.019, 3-ring share 0.898
                     (same OPT-2 artifact, n=66 generated)

HONEST LIMIT (stated here, repeated in the JSON)
------------------------------------------------
The smoke generates ~32 molecules per epoch.  A ring-size share over ~100
rings cannot separate a real effect from sampling noise; this gate reports
DIRECTION and MAGNITUDE, and refuses to call a 5-epoch smoke a quality
result.  A PASS means "the term is wired and moves the profile the right
way", not "the defect is fixed".

Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP (no ring fields on disk).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]

REF_REAL = {"rings_per_mol": 4.193, "share_5_6": 0.9692344383496303, "share_3": 0.004889100882423086}
REF_PRE_PRIOR = {"rings_per_mol": 1.6363636363636365, "share_5_6": 0.018518518518518517, "share_3": 0.8981481481481481}


def _as_dist(v: Any) -> Dict[str, int]:
    if isinstance(v, dict):
        return {str(k): int(x) for k, x in v.items()}
    if isinstance(v, str) and v.strip():
        try:
            d = json.loads(v)
            return {str(k): int(x) for k, x in d.items()} if isinstance(d, dict) else {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _shares(dist: Dict[str, int]) -> Dict[str, float]:
    n = sum(dist.values())
    if not n:
        return {"n_rings": 0, "share_5_6": 0.0, "share_3": 0.0}
    return {
        "n_rings": n,
        "share_5_6": (dist.get("5", 0) + dist.get("6", 0)) / n,
        "share_3": dist.get("3", 0) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="OPT-5 ring-size gate.")
    ap.add_argument("--run-dir", type=Path, default=REPO / "results/cfm_5metal_wfa78_ring_prior")
    ap.add_argument("--out", type=Path, default=REPO / "metrics/by_metric/wfa96_ring_prior_gate.json")
    ns = ap.parse_args()

    metrics_path = ns.run_dir / "per_epoch_metrics.json"
    rows: List[Dict[str, Any]] = []
    source = metrics_path
    if metrics_path.exists():
        doc = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = doc.get("per_epoch_metrics") or []
    if not rows:
        # The trainer writes per_epoch_metrics.json at the END of a run; the
        # events stream is flushed every epoch.  Reading events.jsonl lets the
        # gate run against a live or aborted run instead of reporting SKIP on
        # a run that already measured the thing.
        events = ns.run_dir / "events.jsonl"
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                try:
                    ev = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if ev.get("event") == "train_epoch":
                    rows.append(ev)
            if rows:
                source = events
    if not rows:
        print(f"SKIP: neither {metrics_path} nor events.jsonl carries per-epoch rows")
        return 2

    epochs: List[Dict[str, Any]] = []
    for r in rows:
        dist = _as_dist(r.get("gen_ring_size_distribution"))
        # A genuine zero ("no rings in any generated molecule") is a
        # RESULT, not a missing field.  Key presence decides, not truthiness:
        # the first version of this gate skipped exactly the case it exists
        # to report.
        if "gen_ring_size_distribution" not in r and "gen_rings_per_mol" not in r:
            continue
        sh = _shares(dist)
        epochs.append({
            "epoch": int(r.get("epoch", -1)),
            "rings_per_mol": float(r.get("gen_rings_per_mol") or 0.0),
            "ring_size_distribution": dist,
            **sh,
            "ring_prior_term": float(r.get("ring_prior_term") or 0.0),
            # per_epoch_metrics.json spells it bond_ce_loss; the events stream
            # spells it bond_ce.  Read both rather than reporting 0.0 for a
            # field that is present under the other name.
            "bond_ce_loss": float(r.get("bond_ce_loss") or r.get("bond_ce") or 0.0),
            "grad_norm": float(r.get("grad_norm") or 0.0),
            "train_loss": float(r.get("train_loss") or 0.0),
        })

    if not epochs:
        print("SKIP: no epoch carries gen_rings_per_mol / gen_ring_size_distribution")
        return 2

    last = epochs[-1]
    # Direction, not significance.  Thresholds are the pre-prior generated
    # values with a margin: the 5/6 share must at least double, and the
    # 3-ring share must fall by at least a fifth.  Both are deliberately
    # weak because a 5-epoch, ~32-molecule/epoch smoke cannot support more.
    gate = {
        "share_5_6_doubled": last["share_5_6"] >= 2.0 * REF_PRE_PRIOR["share_5_6"],
        "share_3_reduced_20pct": last["share_3"] <= 0.8 * REF_PRE_PRIOR["share_3"],
        "rings_per_mol_moved_toward_real": last["rings_per_mol"] > REF_PRE_PRIOR["rings_per_mol"],
    }
    # Direction is not the whole story.  Report the OVERSHOOT and the loss
    # share so a reader cannot mistake "moved the right way" for "matches
    # real": at weight 0.01 the last epoch produces MORE rings than real.
    diagnostics = {
        "rings_per_mol_vs_real": last["rings_per_mol"] - REF_REAL["rings_per_mol"],
        "share_5_6_vs_real": last["share_5_6"] - REF_REAL["share_5_6"],
        "share_3_vs_real": last["share_3"] - REF_REAL["share_3"],
        "ring_prior_share_of_train_loss": (
            last["ring_prior_term"] / last["train_loss"]
            if last.get("train_loss")
            else None
        ),
        "train_loss_first_to_last": (
            [epochs[0].get("train_loss"), epochs[-1].get("train_loss")]
        ),
    }
    verdict = "PASS" if all(gate.values()) else "FAIL"

    report = {
        "metric": "wfa96_ring_prior_gate",
        "tag": "MEASURED",
        "date": "2026-10-19",
        "source": str(source),
        "run_dir": str(ns.run_dir),
        "n_epochs_with_ring_fields": len(epochs),
        "epochs": epochs,
        "references": {"real_tmqm_opt2": REF_REAL, "pre_prior_generated_opt2": REF_PRE_PRIOR},
        "gate": gate,
        "diagnostics": diagnostics,
        "verdict": verdict,
        "honest_limits": [
            "5 epochs, ~32 generated molecules per epoch: this measures DIRECTION, not quality.",
            "The 5/6-share reference is the OPT-2 real tmQM sample (n=2000 complexes), not this cohort.",
            "A PASS means the prior is wired and moves the profile the right way; it does not fix the ring-size defect.",
            "Weight 0.01 was not tuned; Wf-A78 measured grad_norm 137.8 vs 1.5 bond-CE-only, so early epochs are ring-dominated.",
        ],
    }

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = ns.out.with_suffix(ns.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(tmp, ns.out)
    print(json.dumps({k: report[k] for k in ("verdict", "gate", "n_epochs_with_ring_fields")}, indent=1))
    print("last epoch:", json.dumps(last, indent=1))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
