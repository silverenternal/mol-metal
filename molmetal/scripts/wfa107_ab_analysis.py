#!/usr/bin/env python
"""Wf-A107 — aggregate the four-arm ring-prior / bond-feature A/B.

Reads `results/wfa107_<arm>/per_epoch_metrics.json` and writes ONE
metrics JSON.  It computes no new metric: every number is copied from
the trainer's own per-epoch record (MEASURED-vs-PROJECTED §5.2).

Arms (one factor each, same cohort/protocol as Wf-A96):
    control   ring weight 0.0, no new bond features
    count     ring weight 0.01, formulation=count      (Wf-A96 protocol)
    per_atom  ring weight 0.01, formulation=per_atom   (Wf-A107)
    features  ring weight 0.0, RBF + pair chemistry    (Wf-A72 Op2/Op3)

Usage:
    python molmetal/scripts/wfa107_ab_analysis.py \
        --root results --out metrics/by_metric/wfa107_ring_prior_ab.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ARMS = ("control", "count", "per_atom", "features", "count_w1e3", "count_w1e4")

#: Weight sweep, in increasing order.  ``control`` is weight 0 by construction.
_SWEEP = ("control", "count_w1e4", "count_w1e3", "count")

#: Fields copied verbatim from each per-epoch record.
_ROW_FIELDS = (
    "epoch",
    "train_loss",
    "bond_ce_loss",
    "grad_norm",
    "ring_prior_term",
    "bond_pr_auc",
    "bond_class_0_pr_auc",
    "bond_class_1_pr_auc",
    "bond_class_2_pr_auc",
    "bond_class_3_pr_auc",
    "bond_class_4_pr_auc",
    "bond_best_f1_at_threshold",
    "bond_best_f1_threshold",
    "bond_f1",
    "bond_precision",
    "bond_recall",
    "gen_validity",
    "decode_ratio",
    "gen_single_fragment_ratio",
    "gen_largest_fragment_frac",
    "gen_rings_per_mol",
    "gen_ring_3_share",
    "gen_ring_5_6_share",
    "gen_ring_size_distribution",
    "gen_metal_ratio",
    "epoch_time_sec",
)


def _read_knobs(arm: str) -> Optional[Dict[str, Any]]:
    """Read the arm's YAML so the metric can name its own prior weight.

    ``per_epoch_metrics.json`` did not record the ring-prior weight until
    Wf-A107 added it, so the five arms that ran before that change would
    otherwise be indistinguishable from each other in the artifact.
    """
    path = Path("configs") / f"wfa107_{arm}.yaml"
    if not path.exists():
        return None
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return None
    doc = yaml.safe_load(path.read_text()) or {}
    flow = doc.get("flow", {}) or {}
    return {
        "source": str(path),
        "ring_template_weight": flow.get("ring_template_weight"),
        "ring_template_formulation": flow.get("ring_template_formulation"),
        "bond_rbf_features": flow.get("bond_rbf_features"),
        "bond_pair_chemistry": flow.get("bond_pair_chemistry"),
    }


def _load_arm(root: Path, arm: str) -> Optional[Dict[str, Any]]:
    path = root / f"wfa107_{arm}" / "per_epoch_metrics.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    rows: List[Dict[str, Any]] = []
    for rec in doc.get("per_epoch_metrics", []):
        rows.append({k: rec.get(k) for k in _ROW_FIELDS})
    cfg = doc.get("config", {})
    return {
        "path": str(path),
        "knobs": _read_knobs(arm),
        "config": {
            k: cfg.get(k)
            for k in ("epochs", "batch_size", "n_train", "n_val", "hidden_dim",
                      "seed", "device")
        },
        "wall_time_s": doc.get("wall_time_s"),
        "rows": rows,
        "last": rows[-1] if rows else None,
    }


def _mean_of_last_two(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [r[key] for r in rows[-2:] if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="metrics/by_metric/wfa107_ring_prior_ab.json")
    args = ap.parse_args()

    root = Path(args.root)
    arms = {arm: _load_arm(root, arm) for arm in ARMS}
    missing = [a for a, v in arms.items() if v is None]

    comparison: Dict[str, Any] = {}
    c, p = arms.get("count"), arms.get("per_atom")
    if c and p and c.get("rows") and p.get("rows"):
        comparison["per_atom_vs_count"] = {
            "last_epoch": {
                k: {
                    "count": c["last"].get(k),
                    "per_atom": p["last"].get(k),
                    "delta": _delta(p["last"].get(k), c["last"].get(k)),
                }
                for k in ("gen_ring_3_share", "gen_ring_5_6_share",
                          "gen_rings_per_mol", "bond_pr_auc",
                          "bond_best_f1_at_threshold", "gen_validity")
            },
            "last_two_epoch_mean": {
                k: {
                    "count": _mean_of_last_two(c["rows"], k),
                    "per_atom": _mean_of_last_two(p["rows"], k),
                    "delta": _delta(
                        _mean_of_last_two(p["rows"], k),
                        _mean_of_last_two(c["rows"], k),
                    ),
                }
                for k in ("gen_ring_3_share", "gen_ring_5_6_share",
                          "gen_rings_per_mol", "bond_pr_auc",
                          "bond_best_f1_at_threshold")
            },
        }
    ctrl, feat = arms.get("control"), arms.get("features")
    if ctrl and feat and ctrl.get("rows") and feat.get("rows"):
        comparison["features_vs_control"] = {
            "last_epoch": {
                k: {
                    "control": ctrl["last"].get(k),
                    "features": feat["last"].get(k),
                    "delta": _delta(feat["last"].get(k), ctrl["last"].get(k)),
                }
                for k in ("bond_pr_auc", "bond_class_0_pr_auc",
                          "bond_class_1_pr_auc", "bond_class_2_pr_auc",
                          "bond_class_3_pr_auc", "bond_class_4_pr_auc",
                          "bond_best_f1_at_threshold", "gen_validity")
            },
            "last_two_epoch_mean": {
                k: {
                    "control": _mean_of_last_two(ctrl["rows"], k),
                    "features": _mean_of_last_two(feat["rows"], k),
                    "delta": _delta(
                        _mean_of_last_two(feat["rows"], k),
                        _mean_of_last_two(ctrl["rows"], k),
                    ),
                }
                for k in ("bond_pr_auc", "bond_best_f1_at_threshold")
            },
        }

    # Wf-A107 follow-up — is the prior's failure a magnitude problem or a
    # formulation problem?  Same protocol, count form, four weights.
    sweep_rows = []
    for arm in _SWEEP:
        rec = arms.get(arm)
        if not rec or not rec.get("last"):
            continue
        knobs = rec.get("knobs") or {}
        last = rec["last"]
        sweep_rows.append({
            "arm": arm,
            "ring_template_weight": knobs.get("ring_template_weight"),
            "last_epoch": {
                k: last.get(k)
                for k in ("grad_norm", "bond_ce_loss", "bond_pr_auc",
                          "bond_best_f1_at_threshold", "gen_rings_per_mol",
                          "gen_ring_3_share", "gen_ring_5_6_share")
            },
            "max_grad_norm": max(
                (r["grad_norm"] for r in rec["rows"]
                 if isinstance(r.get("grad_norm"), (int, float))),
                default=None,
            ),
        })
    if len(sweep_rows) >= 2:
        comparison["ring_weight_sweep"] = {
            "formulation": "count",
            "note": (
                "control is weight 0; the other three are the same run with "
                "only ring_template_weight changed."
            ),
            "rows": sweep_rows,
        }

    doc = {
        "metric": "wfa107_ring_prior_ab",
        "tag": "MEASURED",
        "date": "2026-10-21",
        "what": (
            "One-factor A/B on the Wf-A96 protocol (n_train=200, n_val=64, "
            "5 metals, seed 0, 5 epochs, batch 8, edge-MPNN-residual head): "
            "ring-prior formulation count vs per_atom, and bond-head "
            "Wf-A72 Operators 2+3 (RBF distance + pair chemistry) vs control."
        ),
        "arms": arms,
        "missing_arms": missing,
        "comparison": comparison,
        "honest_limits": [
            "5 epochs, 1 seed, n_train=200.  This is a direction test, not a "
            "quality result; the C7 verdict (SA<=4.5 0.28%, joint gate "
            "0/10,190) was measured on an 18-epoch full-corpus pool and is "
            "not overturned by a 5-epoch smoke.",
            "The ring profile is computed on the DECODED val molecules "
            "(gen_eval_n x gen_eval_seeds = 96 per epoch), not on the "
            "bond head's graph over truth geometry.",
            "Each arm is a single training run, so a delta smaller than the "
            "run-to-run spread of this protocol is not interpretable.  No "
            "spread is measured here (n_seeds=1).",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"[wfa107] wrote {out}")
    if missing:
        print(f"[wfa107] missing arms: {missing}")
    for arm, rec in arms.items():
        if rec and rec.get("last"):
            r = rec["last"]
            print(
                f"[wfa107] {arm:9s} ring3={r.get('gen_ring_3_share')} "
                f"ring56={r.get('gen_ring_5_6_share')} "
                f"rings/mol={r.get('gen_rings_per_mol')} "
                f"existAP={r.get('bond_pr_auc')} "
                f"bestF1={r.get('bond_best_f1_at_threshold')}"
            )
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
