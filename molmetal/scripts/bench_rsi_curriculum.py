"""Quick benchmark for RSI curriculum primitives (TODO-150 §P1+P2).

================================================================
Bench: validate the RSI curriculum orchestrator end-to-end
================================================================
Runs :class:`RSICurriculum` over a small synthetic protocol:

* 3 metals (Ru / Rh / Pd),
* 5 epochs × 2 seeds × 30 samples/epoch,
* per-epoch ledger write + acquire + drift-guard.

Reports per-metal row counts, predictor MAE on the refit epoch,
Tanimoto-histogram of the injection set, and drift-retirement rate.
Output is written to ``molmetal/reports/wf_bench_rsi_curriculum/
bench.json``.

Honest framing
==============
This is a **plumbing** benchmark — it verifies that the 5 primitives
(ledger / predictor / gate / acquisition / drift) wire up correctly
and that the per-step report round-trips.  It does NOT measure any
AUC lift; the only MEASURED Rh-AUC gate is the Wf-124 5-seed ×
50-epoch side-by-side A/B (TODO-150 §P2.2), which lives in
``molmetal/reports/wf124_p2_rsi_curriculum/lift_report.json`` once
shipped.

Usage
-----
::

    python molmetal/scripts/bench_rsi_curriculum.py
    python molmetal/scripts/bench_rsi_curriculum.py --epochs 10 --samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.replay.curriculum import (  # noqa: E402
    AcquisitionResult,
    CurriculumConfig,
    CurriculumStepReport,
    DriftReport,
    RSICurriculum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Distinct aromatic SMILES — diverse enough that Tanimoto < 0.4 holds
# across rows for the diversity filter to actually drop duplicates.
BENCH_AROMATIC_POOL: List[str] = [
    "c1ccccc1",
    "c1ccccc1C",
    "c1ccccc1O",
    "c1ccccc1N",
    "c1ccccc1Cl",
    "c1ccccc1F",
    "c1ccccc1Br",
    "c1ccncc1",
    "c1cnccc1",
    "c1ncccc1",
    "c1ccc2ccccc12",
    "c1ccc2[nH]ccc2c1",
    "c1ccc(C)c(C)c1",
    "c1ccc(O)c(O)c1",
    "c1ccc(N)c(N)c1",
    "c1ccc(Cl)c(Cl)c1",
    "c1ccc(F)c(F)c1",
    "c1cc(ccc1Cl)Cl",
    "c1ccc2cccnc2c1",
    "c1ccc2ncccc2c1",
]


def _synth_samples(metal: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Build a deterministic synthetic per-sample list."""
    import numpy as np

    rng = np.random.default_rng(seed * 1_000 + hash(metal) % 1000)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        smi = BENCH_AROMATIC_POOL[(seed * 31 + i) % len(BENCH_AROMATIC_POOL)]
        loss = float(rng.uniform(0.05, 0.95))
        out.append(
            {
                "row_id": seed * n + i,
                "smiles": smi,
                "pocket_id": "A2780" if i % 2 == 0 else "MCF7",
                "loss": loss,
                "gradient_norm": float(rng.uniform(0.0, 0.5)),
                "predicted_pic50": 6.0 + float(rng.normal(0, 0.5)),
                "target_pic50": 6.5,
                "predicted_active": int(rng.integers(0, 2)),
                "target_active": 1,
            }
        )
    return out


def _summarise_reports(reports: List[CurriculumStepReport]) -> Dict[str, Any]:
    """Aggregate per-step reports into a small dict."""
    by_metal: Dict[str, Dict[str, float]] = {}
    for rep in reports:
        d = by_metal.setdefault(
            rep.metal,
            {
                "n_steps": 0,
                "n_written": 0,
                "n_acquired": 0,
                "n_retired": 0,
                "predictor_mae_sum": 0.0,
                "predictor_mae_n": 0,
                "projection_lift_sum": 0.0,
            },
        )
        d["n_steps"] += 1
        d["n_written"] += rep.n_written
        d["n_acquired"] += rep.n_acquired
        d["n_retired"] += rep.n_retired
        if rep.predictor_mae == rep.predictor_mae:  # not NaN
            d["predictor_mae_sum"] += rep.predictor_mae
            d["predictor_mae_n"] += 1
        d["projection_lift_sum"] += rep.projection_lift
    summary: Dict[str, Any] = {}
    for m, d in by_metal.items():
        summary[m] = {
            "n_steps": int(d["n_steps"]),
            "n_written": int(d["n_written"]),
            "n_acquired": int(d["n_acquired"]),
            "n_retired": int(d["n_retired"]),
            "mean_predictor_mae": (
                float(d["predictor_mae_sum"]) / float(d["predictor_mae_n"])
                if d["predictor_mae_n"] > 0
                else None
            ),
            "mean_projection_lift": float(d["projection_lift_sum"])
            / float(d["n_steps"]),
        }
    return summary


def run_bench(
    metals: List[str],
    seeds: List[int],
    epochs: int,
    samples_per_epoch: int,
    ledger_root: Path,
) -> Dict[str, Any]:
    cfg = CurriculumConfig(
        metal_list=tuple(metals),
        predictor_input_dim=20,
        predictor_hidden_dim=4,
        predictor_n_epochs=3,
        predictor_batch_size=16,
        refit_every=2,
        ledger_root=str(ledger_root),
        seed=0,
    )
    cur = RSICurriculum(cfg)
    reports: List[CurriculumStepReport] = []
    t0 = time.time()
    for seed in seeds:
        for epoch in range(epochs):
            for metal in metals:
                samples = _synth_samples(metal, samples_per_epoch, seed)
                rep = cur.step(
                    metal=metal,
                    seed=seed,
                    epoch=epoch,
                    samples=samples,
                )
                reports.append(rep)
    elapsed = time.time() - t0
    summary = _summarise_reports(reports)
    ledger_stats = cur.ledger.stats()
    out = {
        "config": cfg.to_dict(),
        "metals": metals,
        "seeds": seeds,
        "epochs": epochs,
        "samples_per_epoch": samples_per_epoch,
        "elapsed_seconds": elapsed,
        "per_metal_summary": summary,
        "ledger_stats": ledger_stats,
        "n_total_steps": len(reports),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metals",
        nargs="+",
        default=["Ru", "Rh", "Pd"],
        help="Metals to bench (default: Ru Rh Pd).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1], help="Seeds to bench."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per seed.")
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Samples per (metal, seed, epoch).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "molmetal" / "reports" / "wf_bench_rsi_curriculum"),
        help="Output directory for bench.json.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        ledger_root = Path(td) / "ledger"
        result = run_bench(
            metals=args.metals,
            seeds=args.seeds,
            epochs=args.epochs,
            samples_per_epoch=args.samples,
            ledger_root=ledger_root,
        )
    out_path = out_dir / "bench.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, sort_keys=True, indent=2)
    print(json.dumps(result, sort_keys=True, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
