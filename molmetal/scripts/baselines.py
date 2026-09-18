"""CLI for running classical ML baselines on MetalCytoToxDB.

Usage
-----
    source .venv/bin/activate
    python -m molmetal.scripts.baselines --metal Ru --model xgb
    python -m molmetal.scripts.baselines --metal Ir --model rf
    python -m molmetal.scripts.baselines --metal Ru --model xgb --split ligand_dedup

The ``--split`` flag picks a train/val/test strategy.  Leak-free strategies
(``ligand_dedup``, ``scaffold``, ``temporal``) eliminate the seen-SMILES
overlap diagnosed in ``molmetal/reports/leakage_diagnosis.md`` and emit
JSON files of the form ``baseline_<metal>_<model>_<split>.json`` so the
honest comparison lives side-by-side with the original (suspect) numbers.

Outputs the test-set ROC-AUC, PR-AUC, hit-rate@top-5% to stdout **and** to
``molmetal/reports/baseline_<metal>_<model>[_<split>].json`` for the
dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make ``molmetal.*`` importable when running as ``python -m molmetal.scripts.baselines``
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.baselines.morgan_xgb import SPLITTER_FACTORIES as XGB_SPLITTER_FACTORIES, MorganXGBBaseline
from molmetal.baselines.rf_baseline import RFBaseline
from molmetal.baselines.dmpnn import DMPNNBaseline, SPLITTER_FACTORIES as DMPNN_SPLITTER_FACTORIES
from molmetal.baselines.dmpnn_attentive import AttentiveDMPNNBaseline, SPLITTER_FACTORIES as ATTN_DMPNN_SPLITTER_FACTORIES
# LightGBM is an optional classical-ML control; we always import the module
# so --model lightgbm is wireable, and the class raises RuntimeError at
# ``run()`` time when lightgbm isn't installed.
from molmetal.baselines.morgan_lightgbm import (
    MorganLightGBMBaseline,
    lightgbm_available,
)

# Merge splitter factories
SPLITTER_FACTORIES = {**XGB_SPLITTER_FACTORIES, **DMPNN_SPLITTER_FACTORIES, **ATTN_DMPNN_SPLITTER_FACTORIES}


REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a classical ML baseline.")
    p.add_argument(
        "--metal",
        default="Ru",
        choices=["Ru", "Ir", "Rh", "Os", "Re", "Pt"],
        help="Metal centre to filter on (default: Ru)",
    )
    p.add_argument(
        "--model",
        default="xgb",
        choices=["xgb", "lightgbm", "rf", "dmpnn", "dmpnn_attn"],
        help=(
            "Which baseline to run (default: xgb).  "
            "'xgb' and 'lightgbm' share hyperparameters (depth=6, lr=0.05) "
            "and split strategies so they are directly comparable.  "
            "'lightgbm' requires the optional `lightgbm` package."
        ),
    )
    p.add_argument(
        "--split",
        default="random",
        choices=sorted(SPLITTER_FACTORIES.keys()),
        help=(
            "Splitting strategy. 'random' (default) is the original "
            "Krasnov-paper stratified 80/10/10 split. 'ligand_dedup' and "
            "'scaffold' are leak-free (each canonical SMILES / scaffold "
            "appears in exactly one split). 'temporal' uses pre-2024 rows "
            "for train/val and post-2024 for test. 'chemical' applies a "
            "Tanimoto-0.7 dissimilarity filter."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (forwarded to the splitter and the model)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path. Default: molmetal/reports/baseline_<metal>_<model>[_<split>].json",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for D-MPNN (default: 30)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[baselines] metal={args.metal} model={args.model} "
        f"split={args.split} seed={args.seed}"
    )
    t0 = time.time()
    splitter = SPLITTER_FACTORIES[args.split](args.seed)
    if args.model == "xgb":
        runner = MorganXGBBaseline(metal=args.metal, seed=args.seed, splitter=splitter)
    elif args.model == "lightgbm":
        # Graceful fallback: if lightgbm isn't installed we emit a clear
        # error message and exit 2 (preserving the non-zero exit code so
        # CI / pipelines can detect the missing optional dep).
        if not lightgbm_available():
            print(
                "[baselines] ERROR: --model lightgbm requires the optional "
                "`lightgbm` package.  Install with `uv pip install lightgbm`.",
                file=sys.stderr,
            )
            return 2
        runner = MorganLightGBMBaseline(metal=args.metal, seed=args.seed, splitter=splitter)
    elif args.model == "rf":
        runner = RFBaseline(metal=args.metal, seed=args.seed, splitter=splitter)
    elif args.model == "dmpnn_attn":
        runner = AttentiveDMPNNBaseline(metal=args.metal, seed=args.seed, splitter=splitter, epochs=args.epochs)
    else:
        runner = DMPNNBaseline(metal=args.metal, seed=args.seed, splitter=splitter, epochs=args.epochs)

    result = runner.run()
    elapsed = time.time() - t0

    tm = result.test_metrics
    vm = result.val_metrics
    print(f"[baselines] train/val/test = "
          f"{result.n_train}/{result.n_val}/{result.n_test}")
    print(f"[baselines] val  : "
          f"AUC={vm['roc_auc']:.4f} AP={vm['pr_auc']:.4f} "
          f"Hit@5%={vm['hit_rate_top5pct']:.3f}")
    print(f"[baselines] test : "
          f"AUC={tm['roc_auc']:.4f} AP={tm['pr_auc']:.4f} "
          f"Hit@5%={tm['hit_rate_top5pct']:.3f}")
    if result.per_cell_line:
        top = list(result.per_cell_line.items())[:5]
        print("[baselines] per-cell-line AUC (top 5 by support):")
        for cl, auc in top:
            print(f"    {cl:20s} AUC={auc:.4f}")
    print(f"[baselines] elapsed={elapsed:.1f}s")

    if args.out:
        out_path = Path(args.out)
    elif args.split == "random":
        # Preserve the original filename so existing dashboards keep working
        out_path = REPORTS_DIR / f"baseline_{args.metal.lower()}_{args.model}.json"
    else:
        out_path = REPORTS_DIR / (
            f"baseline_{args.metal.lower()}_{args.model}_{args.split}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(result.to_json())
    payload["split_strategy"] = args.split
    payload["seed"] = args.seed
    payload["elapsed_seconds"] = elapsed
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[baselines] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
