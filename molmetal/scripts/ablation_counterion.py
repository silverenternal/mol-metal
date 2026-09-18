"""A2 — Counter-ion ablation study.

Krasnov (10.1021/acs.jmedchem.5c02755) reports Morgan-FP + XGBoost
baselines on the MetalCytoToxDB but does **not** ablate how the
counter-ion (``Counterion`` column) is handled.  This script runs three
configurations and reports test ROC-AUC under both random and temporal
splits, isolating the effect of (a) ignoring the counter-ion entirely,
(b) adding 5 numeric counter-ion features to the Morgan fingerprint, and
(c) concatenating the counter-ion SMILES as a multi-component SMILES
(``ligand.cation``).

Usage
-----
    source .venv/bin/activate
    python -m molmetal.scripts.ablation_counterion --metal Ru
    python -m molmetal.scripts.ablation_counterion --metal Ru --config counterion_features

The script writes a Markdown table to
``molmetal/reports/a2_counterion_ablation.md`` and prints the same table
to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Make ``molmetal.*`` importable when run as ``python -m molmetal.scripts.ablation_counterion``
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.baselines.eval_utils import compute_metrics
from molmetal.data.counterion_utils import (
    CONFIGS,
    FEATURE_NAMES,
    apply_config,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import RandomSplitter, TemporalSplitter

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
RANDOM_SEED = 42
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
XGB_PARAMS: Dict[str, object] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_dataset(metal: str) -> MetalCytotoxDataset:
    """Load MetalCytoToxDB filtered to ``metal``, requiring Counterion & IC50."""
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[metal],
        require_active_field=True,
        compute_pic50=True,
        compute_active=True,
    )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    if len(ds) == 0:
        raise RuntimeError(f"No rows after filtering for metal={metal!r}.")
    # Drop rows with missing Counterion OR missing IC50
    counterion_col = ds.df["Counterion"]
    valid = counterion_col.notna() & ds.df["IC50_Dark_value"].notna()
    df = ds.df.loc[valid].reset_index(drop=True).copy()
    # Drop empty Counterion strings too (NaN-as-string artefacts)
    mask_nonempty = df["Counterion"].astype(str).str.len() > 0
    df = df.loc[mask_nonempty].reset_index(drop=True).copy()
    if len(df) == 0:
        raise RuntimeError(
            f"No rows with valid Counterion + IC50 for metal={metal!r}."
        )
    return MetalCytotoxDataset(df=df)


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------
def _evaluate_config(
    ds: MetalCytotoxDataset,
    config: str,
) -> Dict[str, object]:
    """Run random + temporal splits for one config; return metrics dict."""
    smiles = ds.smiles
    counterions = ds.df["Counterion"].astype(str).to_numpy()
    y = ds.active.astype(int)

    # Drop rows with empty SMILES (Morgan FP would be uninformative)
    keep = np.array([bool(s) for s in smiles], dtype=bool)
    smiles = smiles[keep]
    counterions = counterions[keep]
    y = y[keep]
    if len(y) == 0:
        raise RuntimeError(f"After dropping empty SMILES nothing remains ({config}).")

    X = apply_config(
        list(smiles),
        list(counterions),
        config,
        morgan_radius=MORGAN_RADIUS,
        morgan_nbits=MORGAN_NBITS,
    )
    n_features = int(X.shape[1])

    # ---- Random split ----
    rnd_split = RandomSplitter(seed=RANDOM_SEED)(ds.__class__(df=ds.df.iloc[np.where(keep)[0]].reset_index(drop=True)))
    # Re-fetch ds view restricted to `keep` mask
    ds_kept = ds.__class__(df=ds.df.iloc[np.where(keep)[0]].reset_index(drop=True))
    rnd_split = RandomSplitter(seed=RANDOM_SEED)(ds_kept)
    random_auc = _train_eval(X, y, rnd_split)

    # ---- Temporal split ----
    tmp_split = TemporalSplitter(cutoff_year=2024)(ds_kept)
    temporal_auc = _train_eval(X, y, tmp_split)

    # n_features differs across configs:
    return {
        "config": config,
        "n_features": n_features,
        "n_samples": int(len(y)),
        "random_auc": random_auc,
        "temporal_auc": temporal_auc,
        "n_train_random": int(rnd_split.n_train()),
        "n_test_random": int(rnd_split.n_test()),
        "n_train_temporal": int(tmp_split.n_train()),
        "n_test_temporal": int(tmp_split.n_test()),
    }


def _train_eval(
    X: np.ndarray,
    y: np.ndarray,
    split,
) -> float:
    """Train XGBoost on split.train/val, evaluate ROC-AUC on split.test."""
    from xgboost import XGBClassifier

    if len(split.train_idx) == 0 or len(split.test_idx) == 0:
        return float("nan")

    X_train = X[split.train_idx]
    y_train = y[split.train_idx]
    X_val = X[split.val_idx] if len(split.val_idx) > 0 else X_train
    y_val = y[split.val_idx] if len(split.val_idx) > 0 else y_train
    X_test = X[split.test_idx]
    y_test = y[split.test_idx]

    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - pos))
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = float(neg) / float(pos)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    score = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, score)
    return float(metrics["roc_auc"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _format_table(rows: List[Dict[str, object]]) -> str:
    """Markdown table summarising one (metal, all-configs) result."""
    lines: List[str] = []
    lines.append(f"# A2 — Counter-ion Ablation ({rows[0].get('metal', '?')})\n")
    lines.append(
        "| config | n_features | random AUC | temporal AUC | "
        "n_train (R / T) | n_test (R / T) |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in rows:
        lines.append(
            "| {config} | {nf} | {rauc:.4f} | {tauc:.4f} | "
            "{ntr}/{ntt} | {ntr2}/{ntt2} |".format(
                config=r["config"],
                nf=r["n_features"],
                rauc=r["random_auc"],
                tauc=r["temporal_auc"],
                ntr=r["n_train_random"],
                ntt=r["n_train_temporal"],
                ntr2=r["n_test_random"],
                ntt2=r["n_test_temporal"],
            )
        )
    return "\n".join(lines) + "\n"


def _interpret(rows: List[Dict[str, object]]) -> str:
    """One-paragraph interpretation of the ablation."""
    by_cfg = {r["config"]: r for r in rows}
    a = by_cfg.get("no_counterion", {}).get("random_auc", float("nan"))
    b = by_cfg.get("counterion_features", {}).get("random_auc", float("nan"))
    c = by_cfg.get("counterion_concat_smiles", {}).get("random_auc", float("nan"))
    a_t = by_cfg.get("no_counterion", {}).get("temporal_auc", float("nan"))
    b_t = by_cfg.get("counterion_features", {}).get("temporal_auc", float("nan"))
    c_t = by_cfg.get("counterion_concat_smiles", {}).get("temporal_auc", float("nan"))

    def _fmt_delta(x: float, base: float) -> str:
        if np.isnan(x) or np.isnan(base):
            return "n/a"
        return f"{x - base:+.4f}"

    lines: List[str] = []
    lines.append("## Interpretation\n")
    lines.append(
        "- `no_counterion` (Krasnov-baseline): random AUC = "
        f"{a:.4f}, temporal AUC = {a_t:.4f}."
    )
    lines.append(
        f"- `counterion_features` (5 added dims): random AUC = {b:.4f} "
        f"(delta {_fmt_delta(b, a)} vs no_counterion), temporal AUC = {b_t:.4f} "
        f"(delta {_fmt_delta(b_t, a_t)})."
    )
    lines.append(
        f"- `counterion_concat_smiles` (multi-component SMILES): random AUC = {c:.4f} "
        f"(delta {_fmt_delta(c, a)} vs no_counterion), temporal AUC = {c_t:.4f} "
        f"(delta {_fmt_delta(c_t, a_t)})."
    )

    # Heuristic verdict
    if not np.isnan(a) and not np.isnan(b) and (b - a) > 0.01:
        lines.append(
            "\nAdding explicit counter-ion features **helps** random-split AUC by "
            ">1pt — the model is exploiting counter-ion chemistry (likely as a "
            "proxy for the complex identity)."
        )
    elif not np.isnan(a) and not np.isnan(b) and abs(b - a) <= 0.005:
        lines.append(
            "\nExplicit counter-ion features give ~0 change on random-split AUC — "
            "the counter-ion is mostly redundant with the metal + ligand signal."
        )
    else:
        lines.append(
            "\nExplicit counter-ion features **hurt** random-split AUC — likely "
            "they enable over-fitting shortcuts (e.g. specific counter-ions as "
            "labels)."
        )

    if not np.isnan(a_t) and not np.isnan(b_t) and (b_t - a_t) < -0.005:
        lines.append(
            "On the **temporal** split the picture inverts: explicit features "
            "**hurt** temporal AUC, suggesting the counter-ion is a *spurious* "
            "leakage channel (some counter-ions are publication-period specific, "
            "e.g. trends in anion choice track the rise of OTf / PF6 across eras)."
        )
    elif not np.isnan(a_t) and not np.isnan(b_t) and (b_t - a_t) > 0.005:
        lines.append(
            "On the **temporal** split explicit features **help** — the "
            "counter-ion carries incremental signal beyond the ligand + metal "
            "(e.g. specific anions track class of complex)."
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A2 counter-ion ablation.")
    p.add_argument("--metal", default="Ru", help="Metal centre (Ru/Ir/Rh/Os/Re).")
    p.add_argument(
        "--config",
        default="all",
        choices=("all",) + tuple(CONFIGS),
        help="Which config to run; default runs all three.",
    )
    p.add_argument(
        "--report",
        default=str(REPORTS_DIR / "a2_counterion_ablation.md"),
        help="Markdown report path.",
    )
    p.add_argument(
        "--json",
        default=None,
        help="Optional JSON dump of the raw results.",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    print(f"[A2] loading MetalCytoToxDB filtered to metal={args.metal!r}", flush=True)
    ds = _load_dataset(args.metal)
    print(
        f"[A2] dataset: n={len(ds)} | metals={ds.metal_counts()} | "
        f"year={int(np.min(ds._years))}-{int(np.max(ds._years))}",
        flush=True,
    )

    cfgs = list(CONFIGS) if args.config == "all" else [args.config]
    rows: List[Dict[str, object]] = []
    for cfg in cfgs:
        print(f"[A2] === config={cfg} ===", flush=True)
        t_cfg = time.time()
        res = _evaluate_config(ds, cfg)
        res["metal"] = args.metal
        rows.append(res)
        print(
            f"[A2]   n_features={res['n_features']} "
            f"random AUC={res['random_auc']:.4f} "
            f"temporal AUC={res['temporal_auc']:.4f} "
            f"({time.time()-t_cfg:.1f}s)",
            flush=True,
        )

    md = _format_table(rows)
    md += "\n" + _interpret(rows)
    md += f"\n_Run walltime: {time.time()-t0:.1f}s_\n"

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"[A2] wrote {out_path}", flush=True)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2))
        print(f"[A2] wrote {json_path}", flush=True)

    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
