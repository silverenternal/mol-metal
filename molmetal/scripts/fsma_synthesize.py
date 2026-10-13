#!/usr/bin/env python3
"""FSMA synthesis — the paper's statistics for the predictive line.

Reads the raw per-(metal, seed) K-shot records emitted by
``train_chemprop_lomo_paper_cohort.py`` (and the XGBoost cross-backbone file)
and produces the tables the paper quotes, with:

* paired one-sided **t-test** as the primary test (the 5-seed percentile
  bootstrap is anti-conservative at n=5 — see
  memory/project_five_seed_bootstrap_anticonservative.md), plus a bootstrap
  interval kept as a reference only;
* **Holm-Bonferroni** across each family (K values / sampler arms);
* per-metal rows ALWAYS, plus a pooled row that is explicitly labelled as
  Ru-dominated (Ru is ~64% of the deduplicated cohort).

Usage::

    .venv/bin/python -m molmetal.scripts.fsma_synthesize \\
        --run k5=molmetal/reports/l3_kshot_pf_k5 \\
        --run k10=molmetal/reports/l3_kshot_pf_k10 \\
        --out-metric metrics/by_metric/fsma_kshot_curve_paperfunnel.json

MEASURED only — every number is read off a run report on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
CONTROL_ARMS = ("from_scratch", "concat_retrain")


def _load_run(d: Path) -> Tuple[Dict[str, Dict[str, dict]], dict]:
    """Return {(metal, seed): kshot_record} plus the run's protocol block."""
    rep = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    cells: Dict[str, Dict[str, dict]] = {}
    for fold in rep.get("per_fold", []):
        m = str(fold.get("metal"))
        for s, rec in fold.get("per_seed", {}).items():
            ks = rec.get("kshot")
            if ks:
                cells[f"{m}|{s}"] = ks
    return cells, rep.get("protocol", {})


def _paired_stats(delta: np.ndarray) -> Dict[str, object]:
    """Paired one-sided t (primary) + percentile bootstrap (reference only)."""
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    n = int(d.size)
    out: Dict[str, object] = {
        "n_cells": n,
        "mean": float(d.mean()) if n else float("nan"),
        "std": float(d.std(ddof=1)) if n > 1 else 0.0,
        "n_positive": int(np.sum(d > 0)),
        "per_cell": [float(x) for x in d],
    }
    if n < 2:
        out["p_one_sided_ttest"] = float("nan")
        out["ci95_ttest"] = [float("nan"), float("nan")]
        return out
    from scipy import stats as _st

    t, _ = _st.ttest_1samp(d, 0.0)
    t = float(t)
    p_one = float(_st.t.sf(t, df=n - 1)) if t > 0 else 1.0 - float(_st.t.sf(t, df=n - 1))
    half = float(_st.t.ppf(0.975, df=n - 1)) * float(d.std(ddof=1)) / np.sqrt(n)
    out["t"] = t
    out["p_one_sided_ttest"] = p_one
    out["ci95_ttest"] = [float(d.mean() - half), float(d.mean() + half)]
    # bootstrap kept as a reference only; at n=5 it cannot generate enough
    # distinct resamples and is anti-conservative.
    rng = np.random.default_rng(42)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
    out["ci95_bootstrap_reference_only"] = [
        float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    ]
    out["p_one_sided_bootstrap_reference_only"] = float(np.mean(boot <= 0.0))
    return out


def _holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(((k, v) for k, v in pvals.items() if v == v), key=lambda kv: kv[1])
    n = len(items)
    adj: Dict[str, float] = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, float(p) * (n - rank)))
        adj[k] = running
    return adj


def _arm_curve(cells: Dict[str, dict], arm: str) -> Dict[str, Dict[str, object]]:
    """Per-metal + pooled paired deltas of ``arm`` vs ``zero_shot_query_auc``."""
    per_metal: Dict[str, List[float]] = {m: [] for m in METALS}
    pooled: List[float] = []
    for key, rec in cells.items():
        metal = key.split("|")[0]
        base = rec.get("zero_shot_query_auc")
        if arm == "kshot":
            val = rec.get("kshot_query_auc")
        else:
            val = rec.get(f"{arm}_query_auc")
        if val is None or base is None or np.isnan(val) or np.isnan(base):
            continue
        d = float(val - base)
        per_metal.setdefault(metal, []).append(d)
        pooled.append(d)
    out: Dict[str, object] = {
        m: _paired_stats(np.asarray(v)) for m, v in per_metal.items() if v
    }
    out["_pooled_ru_dominated"] = _paired_stats(np.asarray(pooled))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", default=[],
                   help="label=path to a run dir (repeatable)")
    p.add_argument("--family", action="append", default=[],
                   help="name=label1,label2,... Holm-Bonferroni family over the "
                        "pooled k-shot delta (repeatable). Without this, one "
                        "family spans every --run, which mixes K values / "
                        "samplers / adapt scopes and is not a valid family.")
    p.add_argument("--xgb-json", default=None,
                   help="optional fsma_xgb_warmstart_kshot.json for the C5 block")
    p.add_argument("--ceiling-run", default=None,
                   help="path to a --mode within run dir on the SAME cohort; its "
                        "within_metal[*].mean_auc becomes the ceiling denominator "
                        "of a recovery_to_ceiling block (P2.2).")
    p.add_argument("--out-metric",
                   default="metrics/by_metric/fsma_kshot_curve_paperfunnel.json")
    args = p.parse_args(argv)

    runs: Dict[str, Path] = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs[label] = (PROJECT_ROOT / path) if not Path(path).is_absolute() else Path(path)

    report: Dict[str, object] = {
        "metric": "fsma_kshot_curve_paperfunnel",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC delta (higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/fsma_synthesize.py",
        "description": (
            "FSMA K-shot synthesis on the paper-funnel cohort: paired "
            "adaptation deltas, the C2 controls (from_scratch / concat_retrain) "
            "and the C3 support-sampler ablation.  Primary test is a paired "
            "one-sided t; the bootstrap is reference-only because n is small."
        ),
        "significance_policy": (
            "paired one-sided t-test is primary; Holm-Bonferroni across the "
            "family; percentile bootstrap retained as *_reference_only"
        ),
        "runs": {},
        "per_run": {},
    }

    for label, d in runs.items():
        if not (d / "metrics.json").exists():
            report["runs"][label] = {"status": "MISSING", "path": str(d)}
            continue
        cells, proto = _load_run(d)
        block: Dict[str, object] = {
            "status": "MEASURED",
            "path": str(d.relative_to(PROJECT_ROOT)) if str(d).startswith(str(PROJECT_ROOT)) else str(d),
            "protocol": proto,
            "n_cells": len(cells),
            "arms": {},
        }
        # the primary adaptation arm
        block["arms"]["kshot"] = _arm_curve(cells, "kshot")
        # C2 controls, when the run carried them
        present = sorted({
            a for rec in cells.values() for a in CONTROL_ARMS
            if f"{a}_query_auc" in rec
        })
        for arm in present:
            block["arms"][arm] = _arm_curve(cells, arm)
        block["controls_present"] = present
        report["per_run"][label] = block

    if args.ceiling_run:
        cpath = PROJECT_ROOT / args.ceiling_run
        if (cpath / "metrics.json").exists():
            crep = json.loads((cpath / "metrics.json").read_text(encoding="utf-8"))
            ceiling = {
                m: float(v["mean_auc"])
                for m, v in crep.get("within_metal", {}).items()
                if v.get("status") == "MEASURED" and v.get("mean_auc") == v.get("mean_auc")
            }
            rec: Dict[str, object] = {
                "ceiling_source": str(cpath.relative_to(PROJECT_ROOT)),
                "ceiling_within_metal_mean_auc": ceiling,
                "note": ("recovery_to_ceiling[m] = mean k-shot query AUC / within-metal "
                         "ceiling AUC, both on the paper-funnel cohort. >100% is possible "
                         "when the LOMO source pool adds information the within-metal CV "
                         "does not see."),
                "per_run": {},
            }
            for label, block in report["per_run"].items():
                if block.get("status") != "MEASURED":
                    continue
                d = runs[label]
                cells, _ = _load_run(d)
                rows: Dict[str, List[float]] = {}
                for key, cell in cells.items():
                    metal = key.split("|")[0]
                    v = cell.get("kshot_query_auc")
                    if v is None or v != v:
                        continue
                    rows.setdefault(metal, []).append(float(v))
                per_m = {
                    m: {
                        "kshot_query_mean": float(np.mean(vs)),
                        "ceiling": ceiling.get(m),
                        "recovery_pct": (100.0 * float(np.mean(vs)) / ceiling[m])
                        if ceiling.get(m) else None,
                    }
                    for m, vs in rows.items()
                }
                rec["per_run"][label] = per_m
            report["recovery_to_ceiling"] = rec

    # family correction across K values, on the pooled k-shot delta
    pvals: Dict[str, float] = {}
    for label, block in report["per_run"].items():
        if block.get("status") != "MEASURED":
            continue
        p = block["arms"]["kshot"]["_pooled_ru_dominated"]["p_one_sided_ttest"]
        pvals[label] = float(p)
    report["family_holm_across_runs"] = {
        "family": list(pvals.keys()),
        "p_raw": pvals,
        "p_holm": _holm(pvals),
        "significant_after_holm_0.05": sorted(
            k for k, v in _holm(pvals).items() if v < 0.05
        ),
    }

    # per-family Holm: each family is a set of run labels that share a single
    # inferential context (one K curve, one sampler sweep, one scope sweep).
    if args.family:
        families: Dict[str, Dict[str, object]] = {}
        for spec in args.family:
            name, _, labels_s = spec.partition("=")
            labels = [x for x in labels_s.split(",") if x]
            fam_p = {lb: pvals[lb] for lb in labels if lb in pvals}
            fam_holm = _holm(fam_p)
            families[name] = {
                "runs": labels,
                "p_raw": fam_p,
                "p_holm": fam_holm,
                "significant_after_holm_0.05": sorted(
                    k for k, v in fam_holm.items() if v < 0.05
                ),
            }
        report["families"] = families

    if args.xgb_json:
        xp = PROJECT_ROOT / args.xgb_json
        if xp.exists():
            report["cross_backbone_xgb"] = json.loads(xp.read_text(encoding="utf-8"))

    out = PROJECT_ROOT / args.out_metric
    out.parent.mkdir(parents=True, exist_ok=True)
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(out, json.dumps(report, ensure_ascii=False, indent=1) + "\n")

    for label, block in report["per_run"].items():
        if block.get("status") != "MEASURED":
            print(f"{label:28} MISSING")
            continue
        k = block["arms"]["kshot"]
        pooled = k["_pooled_ru_dominated"]
        print(f"{label:28} n_cells={pooled['n_cells']:3d} "
              f"delta={pooled['mean']:+.4f} "
              f"CI[{pooled['ci95_ttest'][0]:+.4f},{pooled['ci95_ttest'][1]:+.4f}] "
              f"p_t={pooled['p_one_sided_ttest']:.4f} "
              f"{pooled['n_positive']}/{pooled['n_cells']}+")
        for m in METALS:
            if m in k:
                mm = k[m]
                print(f"    {m:3} n={mm['n_cells']:3d} delta={mm['mean']:+.4f} "
                      f"p_t={mm['p_one_sided_ttest']:.4f} "
                      f"{mm['n_positive']}/{mm['n_cells']}+")
    print("significant after Holm:", report["family_holm_across_runs"]["significant_after_holm_0.05"])
    for name, fam in report.get("families", {}).items():
        print(f"  family[{name}] holm-sig: {fam['significant_after_holm_0.05']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
