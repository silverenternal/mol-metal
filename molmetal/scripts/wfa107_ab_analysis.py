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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.cfm_cohort_recovery import (  # noqa: E402
    COHORT_SOURCE_UNATTESTED,
)
from molmetal.scripts.cfm_pool_epoch_rule import resolve_population  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ARMS = (
    "control", "count", "per_atom", "features", "count_w1e3", "count_w1e4",
    "edge_ring_w1", "edge_ring_w03",
    "control_seed1", "control_seed2", "features_seed1", "features_seed2",
    "edge_ring_w1_seed1", "edge_ring_w1_seed2",
)

#: Weight sweep, in increasing order.  ``control`` is weight 0 by construction.
_SWEEP = ("control", "count_w1e4", "count_w1e3", "count")

#: Arms that differ from their base arm only in ``seed``.  Used to put a
#: spread on the one number this protocol is quoted for.
_SEED_ARMS = {
    "control": ("control", "control_seed1", "control_seed2"),
    "features": ("features", "features_seed1", "features_seed2"),
    "edge_ring_w1": ("edge_ring_w1", "edge_ring_w1_seed1", "edge_ring_w1_seed2"),
}

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
    "bond_class_0_pos_rate",
    "bond_class_1_pos_rate",
    "bond_class_2_pos_rate",
    "bond_class_3_pos_rate",
    "bond_class_4_pos_rate",
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
        # seed variants reuse their base arm's config
        base = arm.split("_seed")[0]
        path = Path("configs") / f"wfa107_{base}.yaml"
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


def _cohort_from_owner(run_dir: Path) -> Dict[str, Any]:
    """The run's cohort, read through the OWNER -- never from the file itself.

    ``per_epoch_metrics.json`` is written ONCE, at the END of training, to a
    FIXED path, so two runs launched into the same ``--out-dir`` clobber each
    other's record and the survivor is whichever finished LAST.  Reading
    ``n_train`` / ``n_val`` straight out of that file therefore publishes a
    SIBLING POPULATION's cohort.

    The owner is ``molmetal/scripts/cfm_pool_epoch_rule.py`` (which in turn
    asks ``molmetal/scripts/cfm_cohort_recovery.py``): it resolves the run of
    record and attaches the cohort attested for THAT population.  An
    unattested cohort is published as ``None`` plus the reason -- never as the
    clobbered file's numbers.
    """
    try:
        population = resolve_population(run_dir, None)
    except Exception as exc:  # pragma: no cover - the owner must not be fatal
        return {
            "n_train": None,
            "n_val": None,
            "source": COHORT_SOURCE_UNATTESTED,
            "reason": (
                f"the cohort owner raised {type(exc).__name__}: {exc}; no "
                f"cohort is published for {run_dir}"
            ),
        }
    cohort = population.get("cohort") or {}
    return {
        "n_train": cohort.get("n_train"),
        "n_val": cohort.get("n_val"),
        "source": cohort.get("source"),
        "reason": cohort.get("reason"),
        "unattested": bool(cohort.get("unattested"))
        or cohort.get("source") == COHORT_SOURCE_UNATTESTED,
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
    # The COHORT comes from the owner, not from the (clobberable) file: the
    # numbers are published only when an artifact on disk ties them to the run
    # of record.  An unattested cohort is published as null + the reason.
    cohort = _cohort_from_owner(root / f"wfa107_{arm}")
    config = {
        k: cfg.get(k)
        for k in ("epochs", "batch_size", "n_train", "n_val", "hidden_dim",
                  "seed", "device")
    }
    config["n_train"] = cohort.get("n_train")
    config["n_val"] = cohort.get("n_val")
    if cohort.get("unattested"):
        config["cohort_source"] = cohort.get("source")
        config["cohort_reason"] = cohort.get("reason")
    return {
        "path": str(path),
        "knobs": _read_knobs(arm),
        "seed": (doc.get("config", {}) or {}).get("seed"),
        "config": config,
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

    # Wf-A107b — the per-edge (intensive) formulation against the control.
    er_rows = []
    for arm in ("control", "edge_ring_w03", "edge_ring_w1"):
        rec = arms.get(arm)
        if not rec or not rec.get("last"):
            continue
        last = rec["last"]
        knobs = rec.get("knobs") or {}
        er_rows.append({
            "arm": arm,
            "ring_template_formulation": knobs.get("ring_template_formulation"),
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
    if len(er_rows) >= 2:
        comparison["edge_ring_vs_control"] = {
            "note": (
                "Same protocol as the weight sweep.  edge_ring replaces the "
                "cycle count with a per-edge ring-membership BCE, whose "
                "gradient is intensive in graph density (probe: 0.225 at "
                "p=0.50 vs 0.084 at p=0.90, a 2.7x change, against the count "
                "form's 7,500x collapse)."
            ),
            "rows": er_rows,
        }

    # Seed spread — the protocol has n_seeds=1 everywhere else.
    spread = {}
    for base, members in _SEED_ARMS.items():
        vals = {}
        for arm in members:
            rec = arms.get(arm)
            if rec and rec.get("last"):
                vals[arm] = {
                    "seed": rec.get("seed"),
                    "bond_pr_auc": rec["last"].get("bond_pr_auc"),
                    "bond_best_f1_at_threshold": rec["last"].get(
                        "bond_best_f1_at_threshold"),
                    "gen_ring_3_share": rec["last"].get("gen_ring_3_share"),
                    "gen_ring_5_6_share": rec["last"].get("gen_ring_5_6_share"),
                }
        if len(vals) < 2:
            continue
        for key in ("bond_pr_auc", "bond_best_f1_at_threshold",
                    "gen_ring_3_share", "gen_ring_5_6_share"):
            xs = [v[key] for v in vals.values() if isinstance(v.get(key), (int, float))]
            if xs:
                spread.setdefault(base, {})[key] = {
                    "values": xs,
                    "mean": sum(xs) / len(xs),
                    "min": min(xs),
                    "max": max(xs),
                    "range": max(xs) - min(xs),
                }
        spread[base]["seeds"] = vals
    if spread:
        # Paired delta: each arm - control at the SAME seed.
        for base in ("features", "edge_ring_w1"):
            pairs = []
            for i in (0, 1, 2):
                suffix = "" if i == 0 else f"_seed{i}"
                c = arms.get("control" + suffix)
                f = arms.get(base + suffix)
                if c and f and c.get("last") and f.get("last"):
                    pairs.append({
                        "seed": i,
                        "control_ap": c["last"].get("bond_pr_auc"),
                        "arm_ap": f["last"].get("bond_pr_auc"),
                        "delta_ap": _delta(f["last"].get("bond_pr_auc"),
                                           c["last"].get("bond_pr_auc")),
                        "control_ring_3": c["last"].get("gen_ring_3_share"),
                        "arm_ring_3": f["last"].get("gen_ring_3_share"),
                        "delta_ring_3": _delta(f["last"].get("gen_ring_3_share"),
                                               c["last"].get("gen_ring_3_share")),
                        "delta_ring_5_6": _delta(
                            f["last"].get("gen_ring_5_6_share"),
                            c["last"].get("gen_ring_5_6_share")),
                    })
            if pairs:
                spread[f"paired_{base}_minus_control"] = pairs
                for key in ("delta_ap", "delta_ring_3", "delta_ring_5_6"):
                    xs = [x[key] for x in pairs if isinstance(x.get(key), (int, float))]
                    if len(xs) < 2:
                        continue
                    mean = sum(xs) / len(xs)
                    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
                    sd = var ** 0.5
                    entry = {
                        "values": xs,
                        "mean": mean,
                        "min": min(xs),
                        "max": max(xs),
                        "n_positive": sum(1 for x in xs if x > 0),
                        "n": len(xs),
                        "sd": sd,
                    }
                    try:
                        from scipy import stats as _st
                        # Two-sided paired t on (arm - control) vs 0.  The
                        # direction is carried by n_positive and the sign of
                        # the mean, not by the p-value.
                        res = _st.ttest_1samp(xs, 0.0)
                        entry["t"] = float(res.statistic)
                        entry["p_two_sided"] = float(res.pvalue)
                    except Exception:  # pragma: no cover
                        pass
                    spread.setdefault(f"paired_{base}_stats", {})[key] = entry
        comparison["seed_spread"] = spread

    # Per-class lift — the SINGLE gate (0.7) has to be read against the base
    # rate: 2.36% of candidate pairs are SINGLE on this cohort.
    lifts = {}
    for arm, rec in arms.items():
        if not rec or not rec.get("last"):
            continue
        row = {}
        for c in range(5):
            ap = rec["last"].get(f"bond_class_{c}_pr_auc")
            pr = rec["last"].get(f"bond_class_{c}_pos_rate")
            if isinstance(ap, (int, float)) and isinstance(pr, (int, float)) and pr > 0:
                row[f"class_{c}"] = {
                    "pos_rate": pr, "pr_auc": ap, "lift_over_base": ap / pr,
                }
        if row:
            lifts[arm] = row
    if lifts:
        comparison["per_class_lift"] = {
            "note": (
                "PR-AUC divided by the class base rate.  On this cohort "
                "SINGLE is 2.36% of candidate pairs, so the Wf-A72 gate "
                "'SINGLE >= 0.7' would require near-perfect ranking; it was "
                "already recorded as aspirational in §6.100."
            ),
            "rows": lifts,
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
    atomic_write_file(out, json.dumps(doc, ensure_ascii=False, indent=1))
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
