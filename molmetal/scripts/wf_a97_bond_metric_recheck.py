#!/usr/bin/env python3
"""Wf-A97 (2026-10-19) -- re-measure the bond ranking metrics after the fix.

WHAT WAS WRONG
--------------
_bond_ranking_report_per_class set bond_pr_auc = bond_class_0_pr_auc (the
NO_BOND one-vs-rest AP) on the claim that reversing the score preserves the
ranking.  It does not.  On a corpus where 70-94% of candidate pairs are NOT
bonded, the class-0 AP is trivially high, and the headline read 0.92-0.98
while the same pairs scored F1 0.00-0.11 at any global threshold.

This script re-runs evaluate_bond_accuracy on the same cohort and reports:
  * bond_pr_auc          -- EXISTENCE AP (1 - P(NO_BOND) vs bonded), corrected
  * bond_pr_auc_class0   -- the old number, labelled for what it is
  * bond_existence_ap    -- the same corrected number, explicit name
  * bond_best_f1_threshold / bond_best_f1_at_threshold -- 1% grid
  * bond_f1 at the shipped 0.5 threshold

GPU is used whenever a device is available (one featuriser pass per cloud).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-clouds", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    ns = ap.parse_args()

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "molmetal" / "scripts"))
    from wf_blkc_l3_guidance_probe import load_clouds, load_model

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        evaluate_bond_accuracy,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, a, state = load_model(ns.ckpt, device)
    clouds = load_clouds(a, ns.n_clouds)

    out = evaluate_bond_accuracy(
        model,
        clouds,
        decision=str(a.get("bond_decision", "threshold")),
        threshold=float(ns.threshold),
    )
    h = hashlib.sha256()
    with ns.ckpt.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    report = {
        "metric": "wfa97_bond_metric_recheck",
        "tag": "MEASURED",
        "date": "2026-10-19",
        "checkpoint": {
            "path": str(ns.ckpt),
            "sha256": h.hexdigest(),
        },
        "device": str(device),
        "n_clouds": len(clouds),
        "threshold_used": float(ns.threshold),
        "metrics": out,
        "reading": {
            "existence_ap": out.get("bond_existence_ap"),
            "class0_ap_old_headline": out.get("bond_pr_auc_class0"),
            "best_f1_ceiling": out.get("bond_best_f1_at_threshold"),
            "best_threshold": out.get("bond_best_f1_threshold"),
            "f1_at_shipped_threshold": out.get("bond_f1"),
        },
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = ns.out.with_suffix(ns.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(tmp, ns.out)
    print(json.dumps(report["reading"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
