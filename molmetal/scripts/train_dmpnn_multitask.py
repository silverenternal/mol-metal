"""CLI for the multi-task D-MPNN baseline (TODO/04 C4).

Usage
-----
    source .venv/bin/activate
    python -m molmetal.scripts.train_dmpnn_multitask --metal Ru --split temporal --epochs 30 --alpha 0.5

Trains a dual-head D-MPNN with:

  * pIC50 regression head (MSE)
  * activity classification head (BCE)

Loss:

  L = alpha * BCE(active_logits, active_label) + (1 - alpha) * MSE(pic50_pred, pIC50_label)

Evaluates and writes both classification (ROC-AUC, PR-AUC, Hit@5%) and
regression (MAE, RMSE, Pearson r) metrics to stdout **and** to
``molmetal/reports/baseline_<metal>_dmpnn_multitask_<split>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskBaseline,
    MT_ALPHA,
)
from molmetal.baselines.dmpnn import SPLITTER_FACTORIES

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-task D-MPNN.")
    p.add_argument(
        "--metal",
        default="Ru",
        choices=["Ru", "Ir", "Rh", "Os", "Re", "Pt"],
        help="Metal centre to filter on (default: Ru)",
    )
    p.add_argument(
        "--split",
        default="random",
        choices=sorted(SPLITTER_FACTORIES.keys()),
        help=(
            "Splitting strategy. 'random' (default), 'temporal' (pre/post 2024), "
            "'ligand_dedup', 'scaffold', 'chemical' (Tanimoto-0.7)."
        ),
    )
    p.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    p.add_argument(
        "--alpha",
        type=float,
        default=MT_ALPHA,
        help="Weight on BCE; (1-alpha) goes to MSE (default: 0.5)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: molmetal/reports/baseline_<metal>_dmpnn_multitask_<split>.json",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default uses MT_LR=1e-3)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[dmpnn_multitask] metal={args.metal} split={args.split} "
        f"epochs={args.epochs} alpha={args.alpha} seed={args.seed}"
    )

    splitter = SPLITTER_FACTORIES[args.split](args.seed)
    kwargs = dict(metal=args.metal, seed=args.seed, splitter=splitter,
                  epochs=args.epochs, alpha=args.alpha)
    if args.lr is not None:
        kwargs["lr"] = args.lr
    runner = DMPNNMultiTaskBaseline(**kwargs)

    t0 = time.time()
    result = runner.run()
    elapsed = time.time() - t0

    tm = result.test_metrics
    vm = result.val_metrics
    tpm = result.test_pic50_metrics
    vpm = result.val_pic50_metrics
    print(
        f"[dmpnn_multitask] train/val/test = "
        f"{result.n_train}/{result.n_val}/{result.n_test}"
    )
    print(
        f"[dmpnn_multitask] val  : "
        f"AUC={vm['roc_auc']:.4f} AP={vm['pr_auc']:.4f} "
        f"Hit@5%={vm['hit_rate_top5pct']:.3f} "
        f"MAE={vpm['mae']:.3f} RMSE={vpm['rmse']:.3f} r={vpm['pearson']:.3f}"
    )
    print(
        f"[dmpnn_multitask] test : "
        f"AUC={tm['roc_auc']:.4f} AP={tm['pr_auc']:.4f} "
        f"Hit@5%={tm['hit_rate_top5pct']:.3f} "
        f"MAE={tpm['mae']:.3f} RMSE={tpm['rmse']:.3f} r={tpm['pearson']:.3f}"
    )
    if result.per_cell_line:
        top = list(result.per_cell_line.items())[:5]
        print("[dmpnn_multitask] per-cell-line AUC (top 5 by support):")
        for cl, auc in top:
            print(f"    {cl:20s} AUC={auc:.4f}")
    print(f"[dmpnn_multitask] elapsed={elapsed:.1f}s")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = REPORTS_DIR / (
            f"baseline_{args.metal.lower()}_dmpnn_multitask_{args.split}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(result.to_json())
    payload["split_strategy"] = args.split
    payload["seed"] = args.seed
    payload["epochs"] = args.epochs
    payload["alpha"] = args.alpha
    payload["elapsed_seconds"] = elapsed
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[dmpnn_multitask] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
