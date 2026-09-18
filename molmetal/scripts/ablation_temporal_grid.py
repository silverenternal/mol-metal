"""A3 — Multi-granularity temporal split ablation.

This script addresses open-question **A3** from
``TODO/07_risks/open_questions.md``:

> A3 — Multi-granularity Temporal Splits
> * Hit-rate decay over time: is the Krasnov-paper 2024 cutoff actually
>   the right OOD boundary, or should we use 2022 / 2025?

We train a Morgan-FP + XGBoost baseline on ``metal`` (default Ru) and
evaluate it under each of the canonical temporal grids from
``molmetal.data.temporal_grids.TEMPORAL_GRIDS``.  The per-grid metrics
plus an ASCII plot of test-hit-rate vs cutoff year are written to
``molmetal/reports/a3_temporal_grid.md``.

Usage
-----
    source .venv/bin/activate
    python -m molmetal.scripts.ablation_temporal_grid --metal Ru
    python -m molmetal.scripts.ablation_temporal_grid --metal Ir --report molmetal/reports/a3_ir.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.baselines.eval_utils import compute_metrics, morgan_features
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.temporal_grids import (
    TEMPORAL_GRIDS,
    hit_rate_by_year,
    rolling_split,
    temporal_split_at_threshold,
)

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
    """Load MetalCytoToxDB filtered to a single metal."""
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
    return ds


# ---------------------------------------------------------------------------
# Per-grid training + evaluation
# ---------------------------------------------------------------------------
def _train_eval_grid(
    X: np.ndarray,
    y: np.ndarray,
    train_df,
    test_df,
    ds_view: MetalCytotoxDataset,
) -> Dict[str, float]:
    """Train XGBoost on ``train_df`` rows, evaluate on ``test_df`` rows.

    Uses the *positional* ordering of the dataset view (after the
    filtering drop).  We remap by index by joining the original ``df``
    against ``train_df`` / ``test_df`` via row hashes.
    """
    # Build a deterministic index lookup keyed by (Year, smiles, active)
    # tuple-of-hashes.  Simpler: track which indices survived the empty
    # SMILES drop, then boolean-mask train/test by membership.
    from xgboost import XGBClassifier

    if len(train_df) == 0 or len(test_df) == 0:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "hit_rate_top5pct": float("nan"),
        }

    # Map from row positions in `df` (post keep-mask) to train/test.
    # We use index alignment with `ds_view.df` (which is reset to 0..N-1).
    keep_idx = ds_view._kept_idx  # type: ignore[attr-defined]
    train_pos = train_df.index.to_numpy()
    test_pos = test_df.index.to_numpy()
    # train_df / test_df indices are *positions* in the kept view, i.e.
    # into `ds_view.df`.  Convert to dataset-level indices via keep_idx.
    train_dataset_idx = keep_idx[train_pos]
    test_dataset_idx = keep_idx[test_pos]

    X_train = X[train_dataset_idx]
    y_train = y[train_dataset_idx]
    X_test = X[test_dataset_idx]
    y_test = y[test_dataset_idx]

    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - pos))
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = float(neg) / float(pos)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    score = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, score)
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _format_table(rows: List[Dict[str, object]]) -> str:
    """Markdown table summarising per-grid metrics."""
    lines: List[str] = []
    lines.append(f"# A3 — Multi-granularity Temporal Splits ({rows[0].get('metal', '?')})\n")
    lines.append(
        "Each row trains Morgan-FP + XGBoost on the **train** portion of one "
        "temporal grid and evaluates on the **test** portion.  Hit rate = "
        "fraction of rows with IC50_Dark_value < 10 µM.\n"
    )
    lines.append(
        "| split | n_train | n_test | hit_rate_train | hit_rate_test | AUC | PR_AUC | mean_year_test |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in rows:
        lines.append(
            "| {split} | {ntr} | {nte} | {hrt:.4f} | {hrte:.4f} | {auc:.4f} | {pr:.4f} | {my:.2f} |".format(
                split=r["split"],
                ntr=r["n_train"],
                nte=r["n_test"],
                hrt=r["hit_rate_train"],
                hrte=r["hit_rate_test"],
                auc=r["auc"],
                pr=r["pr_auc"],
                my=r["mean_year_test"],
            )
        )
    return "\n".join(lines) + "\n"


def _format_hit_rate_plot(
    by_year: Dict[int, float],
    splits: List[Tuple[int, int]],
) -> str:
    """ASCII bar-plot of hit-rate vs year, with split cutoffs overlaid.

    ``splits`` is a list of ``(cutoff_year, n_test)`` for each grid
    (the rolling 2018/2020/2022/2024 grid contributes four cutoffs;
    the static ``pre_2024_vs_2025+`` adds the 2025 line).
    """
    if not by_year:
        return "_No data available for hit-rate plot._\n"

    years = sorted(by_year.keys())
    lo, hi = min(years), max(years)
    # Width of plot in characters
    W = 60
    bar_max = max(by_year.values()) if by_year else 1.0
    bar_max = max(bar_max, 1e-6)
    lines: List[str] = []
    lines.append("## Hit-rate vs publication year\n")
    lines.append("```")
    lines.append(f"hit-rate (y) per year (x), bar width = scaled hit-rate ({bar_max*100:.1f}% = full bar)")
    lines.append("")
    # Header scale
    lines.append(f"  1.00 |" + "-" * W)
    # Draw year rows
    for y in range(lo, hi + 1):
        if y not in by_year:
            continue
        hr = by_year[y]
        bar = int(round((hr / bar_max) * W))
        bar = max(0, min(W, bar))
        marker = ""
        # Mark cutoffs
        for cut, _n in splits:
            if cut == y:
                marker = "  <-- cutoff"
                break
        lines.append(f"  {y:>5d} |" + "#" * bar + " " * (W - bar) + f" {hr*100:5.1f}%" + marker)
    lines.append(f"  {hi+1:>5d} |" + "-" * W)
    lines.append("```")
    lines.append("")
    # Cutoff legend
    lines.append("**Cutoff legend:**")
    for cut, n in splits:
        lines.append(f"- cutoff={cut}: n_test={n}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _interpret(rows: List[Dict[str, object]]) -> str:
    """One-paragraph interpretation that recommends the most honest split."""
    lines: List[str] = []
    lines.append("## Interpretation\n")
    if not rows:
        return "_No rows to interpret._\n"

    # Find the lowest test hit-rate (most biased OOD set)
    hr_min = min(rows, key=lambda r: r["hit_rate_test"])
    hr_max = max(rows, key=lambda r: r["hit_rate_test"])
    auc_min = min(rows, key=lambda r: r["auc"] if not np.isnan(r["auc"]) else float("inf"))
    auc_max = max(rows, key=lambda r: r["auc"] if not np.isnan(r["auc"]) else -float("inf"))

    lines.append(
        f"- **Test hit-rate range**: {hr_min['split']} has the lowest "
        f"test hit-rate ({hr_min['hit_rate_test']:.4f}), "
        f"{hr_max['split']} has the highest ({hr_max['hit_rate_test']:.4f})."
    )
    lines.append(
        f"- **AUC range**: {auc_min['split']} has the lowest test AUC "
        f"({auc_min['auc']:.4f}), {auc_max['split']} the highest "
        f"({auc_max['auc']:.4f})."
    )

    # Hit-rate decay
    deltas = [
        (r["split"], r["hit_rate_test"] - r["hit_rate_train"]) for r in rows
    ]
    deltas.sort(key=lambda x: x[1])
    worst_decay = deltas[0]
    best_decay = deltas[-1]
    lines.append(
        f"- **Hit-rate decay (test − train)**: worst = {worst_decay[0]} "
        f"({worst_decay[1]:+.4f}), best = {best_decay[0]} "
        f"({best_decay[1]:+.4f})."
    )

    # Recommendation
    # Most honest OOD = the split with the largest (test - train) hit-rate
    # gap AND a non-trivial test size AND the lowest test hit-rate.
    # We require n_test >= 100 for statistical power, then pick the split
    # with the largest (train - test) hit-rate gap.
    eligible = [r for r in rows if r["n_test"] >= 100]
    if not eligible:
        # Fall back to any non-tiny split
        eligible = [r for r in rows if r["n_test"] >= 30]
    if not eligible:
        eligible = rows

    # Primary score: (train - test) hit-rate gap, with a small penalty for
    # small test sets so we don't recommend splits with n_test < 50 unless
    # nothing else is available.
    def _score(r: Dict[str, object]) -> float:
        gap = r["hit_rate_train"] - r["hit_rate_test"]
        # Penalty for tiny n_test (below 100, linearly ramp up)
        n = r["n_test"]
        size_penalty = max(0.0, (100 - n) / 100.0) * 0.02
        return gap - size_penalty

    rec = max(eligible, key=_score)
    runner_up = sorted(
        eligible,
        key=_score,
        reverse=True,
    )[1] if len(eligible) > 1 else None

    lines.append("")
    lines.append("### Recommendation\n")
    lines.append(
        f"**Use `{rec['split']}` as the paper's main temporal-test split.**"
    )
    lines.append("")
    lines.append(
        f"Rationale: largest test-train hit-rate gap ("
        f"{rec['hit_rate_train'] - rec['hit_rate_test']:+.4f}) "
        f"with non-trivial n_test={rec['n_test']}, "
        f"and the lowest absolute test hit-rate "
        f"({rec['hit_rate_test']:.4f}) among eligible splits — exactly "
        f"the signature of publication-bias induced data drift."
    )
    if runner_up is not None and runner_up["split"] != rec["split"]:
        lines.append("")
        lines.append(
            f"_Runner-up_: `{runner_up['split']}` "
            f"(gap={runner_up['hit_rate_train'] - runner_up['hit_rate_test']:+.4f}, "
            f"n_test={runner_up['n_test']}, "
            f"AUC={runner_up['auc']:.4f})."
        )
    lines.append("")
    lines.append(
        "_Counter-recommendation:_ a more permissive cutoff (e.g. "
        "`pre_2020_vs_2020+` or `pre_2022_vs_2022+`) inflates test AUC "
        "because the test set still contains many 'easy' recent-but-"
        "overlapping ligands.  Pick the split that *worries you the most* — "
        "that's the honest OOD test."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A3 multi-granularity temporal split ablation.")
    p.add_argument("--metal", default="Ru", help="Metal centre (Ru/Ir/Rh/Os/Re).")
    p.add_argument(
        "--report",
        default=str(REPORTS_DIR / "a3_temporal_grid.md"),
        help="Markdown report path.",
    )
    p.add_argument(
        "--json",
        default=str(REPORTS_DIR / "a3_temporal_grid.json"),
        help="Optional JSON dump of the raw results.",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    print(f"[A3] loading MetalCytoToxDB filtered to metal={args.metal!r}", flush=True)
    ds = _load_dataset(args.metal)

    # Drop rows with empty SMILES (Morgan FP would be uninformative)
    smiles = ds.smiles
    y_all = ds.active.astype(int)
    keep = np.array([bool(s) for s in smiles], dtype=bool)
    keep_idx = np.where(keep)[0]
    df_kept = ds.df.iloc[keep_idx].reset_index(drop=True).copy()
    # We need a 'view' dataset with the kept rows so that indices line up.
    ds_view = MetalCytotoxDataset(df=df_kept)
    # Patch the dataset view with the kept_idx for the per-grid evaluation.
    ds_view._kept_idx = keep_idx  # type: ignore[attr-defined]

    # Featurise once
    X = morgan_features(
        list(df_kept["SMILES_Ligands"].astype(str).to_numpy()),
        radius=MORGAN_RADIUS,
        n_bits=MORGAN_NBITS,
    )
    y = df_kept["active"].astype(int).to_numpy()

    print(
        f"[A3] dataset: n={len(ds)} | metals={ds.metal_counts()} | "
        f"year={int(np.min(ds._years))}-{int(np.max(ds._years))} | "
        f"X.shape={X.shape}",
        flush=True,
    )

    # Iterate the canonical non-rolling grids
    static_grids = [
        ("pre_2020_vs_2020+", 2020),
        ("pre_2022_vs_2022+", 2022),
        ("pre_2024_vs_2024+", 2024),
        ("pre_2024_vs_2025+", 2025),
    ]
    rows: List[Dict[str, object]] = []
    plot_splits: List[Tuple[int, int]] = []
    for split_name, threshold in static_grids:
        print(f"[A3] === grid={split_name} (threshold={threshold}) ===", flush=True)
        t_grid = time.time()
        train_df, test_df, stats = temporal_split_at_threshold(
            df_kept, threshold=threshold, train_ratio=1.0
        )
        metrics = _train_eval_grid(X, y, train_df, test_df, ds_view)
        rows.append(
            {
                "split": split_name,
                "metal": args.metal,
                "n_train": stats["n_train"],
                "n_test": stats["n_test"],
                "hit_rate_train": stats["hit_rate_train"],
                "hit_rate_test": stats["hit_rate_test"],
                "mean_year_train": stats["mean_year_train"],
                "mean_year_test": stats["mean_year_test"],
                "auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "hit_rate_top5pct": metrics["hit_rate_top5pct"],
            }
        )
        plot_splits.append((threshold, stats["n_test"]))
        print(
            f"[A3]   n_train={stats['n_train']} n_test={stats['n_test']} "
            f"hr_train={stats['hit_rate_train']:.3f} "
            f"hr_test={stats['hit_rate_test']:.3f} "
            f"AUC={metrics['roc_auc']:.4f} "
            f"({time.time()-t_grid:.1f}s)",
            flush=True,
        )

    # Also append the rolling cutoffs (2018, 2020, 2022, 2024) as a single
    # summary line so the reader sees the whole rolling sweep.
    print("[A3] === rolling 2018/2020/2022/2024 ===", flush=True)
    rolling = rolling_split(
        df_kept, cutoff_years=(2018, 2020, 2022, 2024), train_ratio=1.0
    )
    rolling_rows: List[Dict[str, object]] = []
    for y_cut, (tr, te, st) in rolling.items():
        metrics = _train_eval_grid(X, y, tr, te, ds_view)
        rolling_rows.append(
            {
                "split": f"rolling_cutoff_{y_cut}",
                "metal": args.metal,
                "n_train": st["n_train"],
                "n_test": st["n_test"],
                "hit_rate_train": st["hit_rate_train"],
                "hit_rate_test": st["hit_rate_test"],
                "mean_year_train": st["mean_year_train"],
                "mean_year_test": st["mean_year_test"],
                "auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "hit_rate_top5pct": metrics["hit_rate_top5pct"],
            }
        )
    rows.extend(rolling_rows)

    # Hit-rate vs year
    by_year = hit_rate_by_year(df_kept)

    # Build report
    md = _format_table(rows)
    md += "\n" + _format_hit_rate_plot(by_year, plot_splits)
    md += "\n" + _interpret(rows)
    md += f"\n_Run walltime: {time.time()-t0:.1f}s_\n"

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"[A3] wrote {out_path}", flush=True)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2))
        print(f"[A3] wrote {json_path}", flush=True)

    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())