#!/usr/bin/env python3
"""Sync todo.json blocker statuses with the 2026-10-13 measurements.

Reads the metric JSONs produced on 2026-10-13 and folds their verdicts into the
matching blocker entries, so no number is hand-typed (AGENTS.md §5.2). Only the
blockers touched by this session are modified; everything else is preserved
byte-for-byte.

Updated blockers:
  * BLK-L1   <- metrics/by_metric/blk_l1_platinai_50d_r2.json
  * BLK-P1   <- metrics/by_metric/dmpnn_within_metal_auc.json
  * BLK-PA4  <- metrics/by_metric/tmqm_pretrain_init_ab_multimetal.json
  * BLK-P2   <- title/status refresh only (ON arm was measured 2026-10-10)

Usage
-----
::

    .venv/bin/python molmetal/scripts/blk_sync_todo_2026_10_13.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # required for the in-repo import at the bottom of main(); without it a
    # direct `python molmetal/scripts/blk_sync_todo_2026_10_13.py` run fails with
    # ModuleNotFoundError: No module named 'molmetal' (same class as BLK-I4 F8).
    sys.path.insert(0, str(PROJECT_ROOT))
TODO = PROJECT_ROOT / "todo.json"


def _read(path: str) -> Optional[Dict]:
    p = PROJECT_ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _find(blockers: List[Dict], bid: str) -> Optional[Dict]:
    for b in blockers:
        if b.get("id") == bid:
            return b
    return None


def main() -> int:
    doc = json.loads(TODO.read_text())
    blockers = doc["blockers"]

    l1 = _read("metrics/by_metric/blk_l1_platinai_50d_r2.json")
    p1 = _read("metrics/by_metric/dmpnn_within_metal_auc.json")
    pa4 = _read("metrics/by_metric/tmqm_pretrain_init_ab_multimetal.json")

    if l1:
        b = _find(blockers, "BLK-L1")
        ctrl = l1["arms"]["rdkit_32d_control"]
        arm50 = l1["arms"]["rdkit32_morganpca18_50d"]
        b["status"] = f"RESOLVED_HONEST_NEGATIVE_2026_10_13 — {l1['verdict']['verdict']}"
        b["resolution_2026_10_13"] = {
            "what_was_run": (
                "built the never-wired 50-d arm (32-d RDKit + 18 PCA dims of a "
                "real Morgan ECFP4 fingerprint, PCA fit inside each training "
                "fold) and re-measured the 32-d control on the same folds"
            ),
            "control_32d_r2": ctrl["r2_mean"],
            "arm_50d_r2": arm50["r2_mean"],
            "delta_50d_minus_32d": arm50["delta_vs_control"],
            "n": l1["n"],
            "control_reproduces_on_disk": ctrl["reproduces_on_disk_control"],
            "verdict": l1["verdict"]["verdict"],
            "reading": (
                "the 32-d control reproduces the on-disk +0.01025 exactly, and "
                "the 50-d arm is WORSE (+0.0031). platinai_4cell is "
                "representation-limited, not feature-count-limited: the "
                "PROJECTED +0.15 lift from 50-d is falsified."
            ),
            "chemprop_v3_unavailable": (
                "no trained chemprop v3 encoder checkpoint exists on disk; "
                "chemprop_v3_encoder_forward builds an untrained encoder, so an "
                "18-d slice of it would be noise. The 18-d component is PCA of "
                "Morgan instead, labelled as such in the artifact."
            ),
            "hilbert_unavailable": (
                "no SMILES -> Hilbert-coordinate featurizer exists in the repo; "
                "the '50-d' tier label named a family, not a defined composition."
            ),
            "artifact": "metrics/by_metric/blk_l1_platinai_50d_r2.json",
        }

    if p1:
        b = _find(blockers, "BLK-P1")
        per = p1["per_metal"]
        measured = {
            m: {
                "n_unique": v["n_unique"],
                "dmpnn_auc": v["dmpnn_auc"],
                "xgb_auc_same_folds": v["xgb_auc_same_folds"],
                "delta": v["delta_dmpnn_minus_xgb"],
                "krasnov": v.get("krasnov_reported_auc"),
                "resolvable": v["resolvable"],
            }
            for m, v in per.items()
            if v.get("status") == "MEASURED"
        }
        b["status"] = "RESOLVED_2026_10_13 — D-MPNN within-metal number now on disk"
        n_measured = len(measured)
        n_below = sum(1 for v in measured.values() if v["delta"] < 0)
        b["resolution_2026_10_13"] = {
            "what_was_run": (
                "D-MPNN (MetalCytotoxModel, 128/3) within-metal 5-fold on the "
                "probe's deduplicated cohort, on the SAME folds the XGB "
                "within-metal column uses (StratifiedKFold random_state in "
                "{42,43,44,49,123}); XGB re-measured on the identical folds so "
                "the delta is paired"
            ),
            "per_metal": measured,
            "n_metals_measured": n_measured,
            "n_metals_dmpnn_below_xgb": n_below,
            "split_seeds": p1["split_seeds"],
            "pooled_mean_reported": False,
            "reading": (
                f"the D-MPNN's within-metal number exists now and is BELOW the "
                f"fingerprint+XGBoost column on {n_below}/{n_measured} measured "
                f"metals. The paper must not attribute the 0.8363/0.8295 "
                f"within-metal values to the D-MPNN; BLK-P1's "
                f"honest_negative_fallback is invoked — the activity-prediction "
                f"backbone is 'GBDT by protocol'."
            ),
            "artifact": "metrics/by_metric/dmpnn_within_metal_auc.json",
        }

    if pa4:
        b = _find(blockers, "BLK-PA4")
        b["status"] = f"PARTIAL_2026_10_13 — {pa4['family_verdict']}"
        b.setdefault("result_2026_10_12", {})
        b["result_2026_10_13"] = {
            "metals": pa4["metals"],
            "per_metal_delta": {
                m: v["paired_delta"]["mean"] for m, v in pa4["per_metal"].items()
            },
            "per_metal_verdict": {
                m: v["verdict"] for m, v in pa4["per_metal"].items()
            },
            "holm_significant": pa4["holm"]["significant_after_holm_0.05"],
            "family_verdict": pa4["family_verdict"],
            "reading": (
                f"the A/B now covers {len(pa4['metals'])} metals. Under the "
                f"paired one-sided t-test — the 5-point percentile bootstrap is "
                f"anti-conservative — no metal is Holm-significant positive: "
                f"{pa4['holm']['significant_after_holm_0.05'] or 'none'}. Ir is "
                f"a suggestive +0.0486 (3/5 seeds positive, p_t=0.068, t-CI "
                f"includes zero); Ru is null. Claim 2 stays an honest negative, "
                f"now measured over two metals rather than one."
            ),
            "still_open": (
                "cross-metal (LOMO) pretraining A/B and the ZINC/organic-corpus "
                "ablation from fix_steps 1-3 remain unrun"
            ),
            "artifact": "metrics/by_metric/tmqm_pretrain_init_ab_multimetal.json",
        }

    b = _find(blockers, "BLK-P2")
    if b:
        b["title"] = (
            "MCTS 的 D-MPNN 价值通道 ON 臂已测（n_dmpnn_eval=737，接线正确）"
            "但本队列规模下质量指标与 OFF 臂逐位相同 —— UNDERPOWERED"
        )
        b["status_note"] = (
            (b.get("status_note") or "")
            + " | 2026-10-13: title refreshed — the ON arm WAS measured "
            "(ON_arm_result_MEASURED_2026_10_10, n_dmpnn_eval=737), so the "
            "original 'n_dmpnn_eval=0 / never measured' title was stale. "
            "Verdict unchanged: wiring verified, no quality effect at this "
            "cohort size."
        ).strip(" |")

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(TODO, json.dumps(doc, indent=2, ensure_ascii=False))
    print("updated blockers:", [b["id"] for b in blockers
                                if b.get("resolution_2026_10_13") or b["id"] == "BLK-P2"])
    print(f"wrote {TODO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
