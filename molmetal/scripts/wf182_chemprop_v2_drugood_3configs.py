"""Wf-182 — Extend Wf-174 to 5 seeds × 20 epochs × 3 refined DrugOOD configs.

==============================================================================
Wf-182 ultracode follow-up to Wf-174 chemprop v2 baseline. Runs the same
chemprop v2 retrain on the three lbap_refined_ic50_* configs (assay,
scaffold, size) at full 5-seed × 20-epoch scale and emits a single
aggregate JSON with per-config per-seed ood_test AUC + per-config
mean / std.

Strategy
--------
Wf-174's :mod:`wf174_chemprop_v2_drugood_real` is parameterised at
module level via the :data:`SUBSET` constant.  We monkey-patch it
plus :data:`DEFAULT_OUT` and invoke :func:`wf174_chemprop_v2_drugood_real.main`
once per config.  Output is per-config
``molmetal/reports/wf182_chemprop_v2_drugood_3configs/<config>/summary.json``
plus a top-level ``wf182_summary.json`` aggregating all 3 configs.

Wall-time budget: 4 hr.  Per Wf-174 measurements: 415s / seed ×
5 seeds = ~35 min per config.  3 configs = ~105 min training + ~5 min
featurization per config (cache hits) ≈ 1.85 hr.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# --- Wf-190 env guard --------------------------------------------
# Fail fast at import time if --bf16 / CUDA is required but unavailable.
# Mirrors mol-metal-gpu-execution §"MEASURED-vs-PROJECTED label discipline":
# we never silently run on CPU when the script claims GPU.  Auto-fallback to
# CPU is still supported (set MOLMETAL_ALLOW_CPU=1) to keep CPU-only boxes
# working, but explicit is preferred.  The cuda-availability check itself
# lives in main() (after `import torch`) per the molmetal-gpu-execution
# skill convention.
_MOLMETAL_ALLOW_CPU = os.environ.get("MOLMETAL_ALLOW_CPU", "0") == "1"
# --- end Wf-190 env guard ----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHEMPROP_V2_PATH = PROJECT_ROOT / ".cloned" / "chemprop_v2"
for p in (str(PROJECT_ROOT), str(CHEMPROP_V2_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

CONFIGS = (
    "lbap_refined_ic50_assay",
    "lbap_refined_ic50_scaffold",
    "lbap_refined_ic50_size",
)

DEFAULT_OUT_ROOT = PROJECT_ROOT / "molmetal" / "reports" / "wf182_chemprop_v2_drugood_3configs"


def _log(msg: str) -> None:
    print(f"[wf182] {msg}", flush=True)


def run_one_config(
    subset: str,
    out_dir: Path,
    seeds: List[int],
    epochs: int,
    time_budget_s: float,
    cache_dir: Path,
    reset_cache: bool = False,
) -> Dict:
    """Invoke Wf-174's :func:`main` with SUBSET monkey-patched.

    Returns the per-config summary dict that wf174 writes.
    """
    # Import here so PYTHONPATH / venv is set up before we touch
    # chemprop v2 / molmetal modules.
    wf174 = importlib.import_module("molmetal.scripts.wf174_chemprop_v2_drugood_real")
    # Reload in case of stale state (defensive).
    importlib.reload(wf174)

    # Monkey-patch module-level constants.
    wf174.SUBSET = subset
    wf174.DEFAULT_OUT = out_dir
    wf174.DEFAULT_CACHE_DIR = cache_dir

    # Build the argv to pass to wf174.main() so we get full control
    # over seeds / epochs / time_budget / cache-reset.
    argv = [
        "--seeds", *[str(s) for s in seeds],
        "--epochs", str(epochs),
        "--out", str(out_dir),
        "--cache-dir", str(cache_dir),
        "--time-budget-s", str(time_budget_s),
        # P0 FIX 2026-09-28: Default to 'cuda' now that wf174 supports GPU
        # (was 'cpu' which left Blackwell sm_120 idle at 0% utilization).
        # Falls back to CPU automatically if torch.cuda.is_available() False.
        # Previous 'Wf-182 task spec: CPU only' comment is now stale —
        # GPU concurrent contention resolved by serial per-config launch.
        "--device", "cuda",
    ]
    if reset_cache:
        argv.append("--reset-cache")
    _log(f"running subset={subset}  out={out_dir}  cache_dir={cache_dir}  "
         f"seeds={seeds}  epochs={epochs}")
    rc = wf174.main(argv)
    if rc != 0:
        raise RuntimeError(f"wf174.main returned {rc} for subset={subset}")
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"expected {summary_path} after wf174.main")
    with open(summary_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    return summary


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS),
                   help="DrugOOD lbap_refined_ic50_* subset names to run.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--cache-dir", default=None,
                   help="Shared MolGraphCache directory (re-used across "
                        "configs — saves the ~144s npz-load cost on "
                        "configs 2+3). Defaults to "
                        "<out-root>/shared_cache.")
    p.add_argument("--time-budget-s", type=float, default=14400.0,
                   help="Per-run wall budget (4 hr default — matches session budget).")
    p.add_argument("--reset-cache", action="store_true",
                   help="Delete cache dir before featurizing each config.")
    args = p.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_root / "shared_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _log(f"using shared cache_dir={cache_dir} "
         f"({sum(1 for _ in cache_dir.glob('*.npz'))} npz files)")

    overall_t0 = time.time()
    per_config: Dict[str, Dict] = {}
    failures: Dict[str, str] = {}

    for subset in args.configs:
        config_name = subset.replace("lbap_refined_ic50_", "")
        out_dir = out_root / config_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"=== START config={config_name} subset={subset} ===")
        try:
            summary = run_one_config(
                subset=subset,
                out_dir=out_dir,
                seeds=list(args.seeds),
                epochs=args.epochs,
                time_budget_s=args.time_budget_s,
                cache_dir=cache_dir,
                reset_cache=args.reset_cache,
            )
            per_config[config_name] = summary
            _log(f"=== END config={config_name}  "
                 f"ood_test_mean={summary.get('ood_test_auc_mean'):.4f}  "
                 f"total_wall_min={summary.get('total_wall_min'):.2f} ===")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _log(f"=== FAIL config={config_name}: {exc} ===")
            failures[config_name] = str(exc)
            # Continue with the remaining configs — we still want
            # partial coverage to report honestly.
        # Print inter-config wall so user can monitor.
        _log(f"elapsed since start: {(time.time() - overall_t0) / 60.0:.1f} min")

    total_wall_min = (time.time() - overall_t0) / 60.0

    # ------------------------------------------------------------------
    # Aggregate.
    # ------------------------------------------------------------------
    def _mean(xs: List[float]) -> float:
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return float(sum(xs) / len(xs)) if xs else float("nan")

    def _std(xs: List[float]) -> float:
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return float((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)

    aggregate = {
        "date": time.strftime("%Y-%m-%d"),
        "task": "Wf-182",
        "configs_run": list(per_config.keys()),
        "configs_failed": list(failures.keys()),
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "hidden": 128,
        "depth": 2,
        "head_hidden": 128,
        "device": "cpu",
        "total_wall_min": total_wall_min,
        "per_config_ood_test": {},
        "per_config_ood_test_mean": {},
        "per_config_ood_test_std": {},
        "per_config_n_runs": {},
        "per_config_full_summary": {},
    }
    for config_name, summary in per_config.items():
        aucs = [
            r["ood_test_auc"]
            for r in summary["results"]
            if r["status"] == "ok"
        ]
        aggregate["per_config_ood_test"][config_name] = aucs
        aggregate["per_config_ood_test_mean"][config_name] = _mean(aucs)
        aggregate["per_config_ood_test_std"][config_name] = _std(aucs)
        aggregate["per_config_n_runs"][config_name] = len(aucs)
        aggregate["per_config_full_summary"][config_name] = {
            "ood_test_auc_mean": summary.get("ood_test_auc_mean"),
            "passes_ood_gate": summary.get("passes_ood_gate"),
            "n_seeds_ok": summary.get("n_seeds_ok"),
            "n_seeds_total": summary.get("n_seeds_total"),
            "total_wall_min": summary.get("total_wall_min"),
            "splits_n_train": summary.get("splits", {}).get("train", {}).get("n_valid"),
        }

    aggregate_path = out_root / "wf182_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, default=str))
    _log(f"wrote {aggregate_path}")

    # Console summary.
    print("\n========== Wf-182 SUMMARY ==========")
    for config_name in per_config.keys():
        m = aggregate["per_config_ood_test_mean"][config_name]
        s = aggregate["per_config_ood_test_std"][config_name]
        n = aggregate["per_config_n_runs"][config_name]
        aucs = aggregate["per_config_ood_test"][config_name]
        aucs_str = ", ".join(f"{a:.4f}" for a in aucs)
        print(f"  {config_name:10s}  n={n}  mean={m:.4f}  std={s:.4f}  "
              f"per-seed=[{aucs_str}]")
    if failures:
        print(f"\nFAILED configs: {failures}")
    print(f"\nTotal wall: {total_wall_min:.2f} min")
    print("====================================")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())