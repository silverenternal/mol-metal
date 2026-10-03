"""R18 modules 5-fold cross-validation harness.

================================================================
W2.3 / R18 ultracode dispatch (2026-09-26).

This script runs a **5-fold CV across the 32 R18 ship-modules** using
the same module registry as
:file:`molmetal/scripts/benchmark_r18_platinai.py`.  Where the PlatinAI /
MetalCytoToxDB benchmarks sweep *modules × samples*, this script sweeps
*modules × folds* and reports per-fold accuracy, wall-time, and memory.

Why "CV" on deterministic forward primitives?
---------------------------------------------
The R18 modules are forward-primitive carriers (SE3Geodesic.interpolate,
BergmanKernel(...), TanimotoOODProtocol.is_ood, …) whose outputs are
deterministic given a fixed input batch + seed.  The CV sweep varies the
*seed* (and a small slice of the input batch) across 5 folds so that:

  1. **Variance quantification** — each module's wall-time / memory is
     reported as mean ± std across 5 folds, not as a single point.
  2. **Regression detection** — a single bad fold surfaces in the JSON
     so a CI gate can flag any module whose fold-to-fold std exceeds
     the per-module mean (i.e. CV-stability flag).
  3. **Audit-grade reporting** — the per-module per-fold record is
     the basis for the paper-claim "5-fold CV mean ± std" line.

The folds are produced by slicing the PlatinAI 5K sample batch into 5
disjoint subsets and varying the module random seed across folds.
DiffDock + EquiBind are intentionally import-only (no runner) and
contribute accuracy=0, status="skipped".

Output schema (``molmetal/reports/wf_cv_harness/results.json``)::

    {
      "n_modules": 32,
      "n_folds": 5,
      "n_runs_per_fold": 5,
      "module_groups": {...},
      "per_module_per_fold_metrics": {
        "<module_id>": {
          "folds": [
            {"fold_idx": 0, "seed": 0,
             "n_samples": 100, "n_runs": 5,
             "accuracy": 1.0, "wall_time_ms_median": 0.42,
             "wall_time_ms_mean": 0.43,
             "memory_mb": 0.5, "status": "ok"},
            ...
          ],
          "mean": {"accuracy": 1.0, "wall_time_ms_median": 0.43,
                   "wall_time_ms_mean": 0.44, "memory_mb": 0.5},
          "std":  {"accuracy": 0.0, "wall_time_ms_median": 0.02,
                   "wall_time_ms_mean": 0.03, "memory_mb": 0.05}
        },
        ...
      },
      "mean_metrics": {
        "accuracy_mean": 0.97,
        "wall_time_ms_median_mean": 0.42,
        "wall_time_ms_median_std":  0.04,
        "memory_mb_mean": 0.6,
        "memory_mb_std":  0.1
      },
      "std_metrics": {...},
      "fold_seeds": [0, 1, 2, 3, 4],
      "total_wall_time_s": 12.34,
      "peak_memory_mb_total": 12.34,
      "timestamp_utc": "...",
      "python": "...",
      "torch": "..."
    }

Usage::

    source env.sh && uv run --no-sync python molmetal/scripts/r18_cv_harness.py
    python3 molmetal/scripts/r18_cv_harness.py --n-samples 100 --n-folds 5

Honest framing
--------------
* This is a **CPU-only** harness; the W2.3 contract pins every R18
  module to the CPU backend.
* "Accuracy" here means the fraction of forward calls that returned a
  finite result — it is NOT a downstream task accuracy (e.g. AUC).
  Modules whose forward primitive raises on every call report
  accuracy=0.0 and status="error"; modules with no runner (DiffDock /
  EquiBind) report accuracy=0.0 and status="skipped".
* The PlatinAI 5K sample batch is sliced into ``n_folds`` disjoint
  contiguous subsets; the ``--seed`` controls the RNG inside each
  module, not the fold assignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths — keep absolute so the script can be invoked from any CWD.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_cv_harness"
REPORT_FILE = REPORT_DIR / "results.json"
DEFAULT_CSV = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"

DEFAULT_N_FOLDS = 5
DEFAULT_N_RUNS_PER_FOLD = 5
DEFAULT_N_SAMPLES = 100  # total samples split across folds
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Module registry — reuse the canonical 32-module catalogue.
# ---------------------------------------------------------------------------
def _build_module_registry() -> Dict[str, Callable[[List[str], int], List[float]]]:
    """Compose the 32-module registry from benchmark_r18_platinai.

    The runners return wall-times in seconds; ``n_runs`` calls per
    invocation.  ``None`` runners (DiffDock / EquiBind) are kept in
    the registry so the JSON output preserves the 32-module shape.
    """
    from molmetal.scripts.benchmark_r18_platinai import (  # noqa: WPS433
        _build_module_registry as _plat_registry,
    )

    return _plat_registry()


_MODULE_GROUPS: Dict[str, List[str]] = {
    "math_skills": [
        "math_skills/bass_existence",
        "math_skills/bergman_kernel",
        "math_skills/monge_ampere_regularity",
        "math_skills/multi_parameter_ph",
        "math_skills/ph_hardness_guardrail",
        "math_skills/sb_bass_martingale",
        "math_skills/se3_schrodinger_bridge",
        "math_skills/transport_relaxation",
        "math_skills/wasserstein_gradient_flow",
    ],
    "adapters": [
        "adapters/coordination_gnn",
        "adapters/quotient_space_diffusion",
        "adapters/multi_ph_integration",
        "adapters/se3_sb_integration",
        "adapters/transport_relaxation_integration",
        "adapters/diffdock",
        "adapters/equibind",
    ],
    "lam_chem": [
        "lam_chem/coupling_joint_inference",
        "lam_chem/decoder_m3_sequence",
        "lam_chem/cross_attention_m4",
        "lam_chem/layered_generation",
        "lam_chem/buyable_gate",
        "lam_chem/ood_protocol",
        "lam_chem/macroguide_ph",
        "lam_chem/non_click_medchem_rules",
        "lam_chem/skeleton_jump",
        "lam_chem/md_relaxation_stub",
        "lam_chem/fg_compat_veto",
    ],
    "reward": [
        "reward/hilbert_primal_dual",
    ],
    "search_alg": [
        "search_alg/score_repellent_mc",
        "search_alg/variance_puct_integration",
    ],
    "flow_matching_lipman": [
        "flow_matching_lipman/w_flow",
        "flow_matching_lipman/m5_schedulers",
    ],
}


# ---------------------------------------------------------------------------
# Data loading — same loader as benchmark_r18_platinai.
# ---------------------------------------------------------------------------
def _load_platinai_smiles(csv_path: Path, n: int) -> List[str]:
    """Return the first ``n`` SMILES from the PlatinAI 5K CSV.

    Empty / whitespace SMILES are skipped.  Returns whatever is
    available if the CSV is smaller than ``n``.
    """
    out: List[str] = []
    if not csv_path.is_file():
        return out
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                out.append(smi)
            if len(out) >= n:
                break
    return out


def _make_synthetic_smiles(n: int, seed: int) -> List[str]:
    """Build a fallback list of ``n`` synthetic SMILES if the CSV is missing.

    The synthetic set is a deterministic round-robin over a 10-SMILES
    kitchen-sink so the CV harness can still exercise every module on
    a meaningful sample batch even without the PlatinAI CSV present.
    """
    kitchen = [
        "CCO",
        "CCN",
        "CCC",
        "CC(=O)O",
        "CCN(C)C",
        "c1ccccc1",
        "c1ccncc1",
        "CCNC(=O)C",
        "C1CCCCC1",
        "CCBr",
    ]
    return [kitchen[(i + seed) % len(kitchen)] for i in range(n)]


# ---------------------------------------------------------------------------
# Fold construction — deterministic, contiguous, balanced.
# ---------------------------------------------------------------------------
def _build_folds(
    samples: List[str],
    n_folds: int,
) -> List[List[str]]:
    """Slice ``samples`` into ``n_folds`` contiguous folds.

    Each fold has size ``ceil(len(samples) / n_folds)``; the last fold
    is truncated if the samples don't divide evenly.  Folds are
    returned in fold-index order.  This is the "5-fold CV" layout.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be positive, got {n_folds}")
    n = len(samples)
    if n == 0:
        return [[] for _ in range(n_folds)]
    fold_size = math.ceil(n / n_folds)
    folds: List[List[str]] = []
    for k in range(n_folds):
        start = k * fold_size
        end = min((k + 1) * fold_size, n)
        folds.append(samples[start:end])
    return folds


# ---------------------------------------------------------------------------
# Per-fold evaluation — invoke the module, capture wall-time + memory.
# ---------------------------------------------------------------------------
def _run_fold(
    runner: Callable[[List[str], int], List[float]] | None,
    fold_samples: List[str],
    n_runs: int,
    seed: int,
) -> Dict[str, Any]:
    """Run ``runner`` on ``fold_samples`` and capture fold metrics.

    Returns a dict with:
      - ``fold_idx`` (set by caller),
      - ``seed``,
      - ``n_samples``,
      - ``n_runs``,
      - ``accuracy`` (fraction of forward calls that returned finite result),
      - ``wall_time_ms_median``, ``wall_time_ms_mean``,
      - ``memory_mb`` (tracemalloc delta),
      - ``status`` (``ok`` / ``skipped`` / ``error``).
    """
    if runner is None:
        return {
            "seed": seed,
            "n_samples": len(fold_samples),
            "n_runs": n_runs,
            "accuracy": 0.0,
            "wall_time_ms_median": 0.0,
            "wall_time_ms_mean": 0.0,
            "memory_mb": 0.0,
            "status": "skipped",
        }

    # Seed the RNGs for deterministic forward primitives.
    try:
        import random as _random

        _random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as _np

        _np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
    except Exception:  # noqa: BLE001
        pass

    base = tracemalloc.get_traced_memory()[0]
    t0 = time.perf_counter()
    wall_times: List[float] = []
    status = "ok"
    error_msg = None
    try:
        wt = runner(fold_samples, n_runs)
        wall_times = [float(w) for w in wt if w is not None and math.isfinite(w)]
        if not wall_times:
            status = "error"
            error_msg = "runner returned empty / non-finite wall_times"
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_msg = f"{type(exc).__name__}: {str(exc)[:200]}"
    elapsed = time.perf_counter() - t0
    peak_mem = tracemalloc.get_traced_memory()[1]
    memory_mb = float(peak_mem - base) / (1024 * 1024)

    accuracy = (len(wall_times) / n_runs) if n_runs > 0 else 0.0
    wall_ms_median = statistics.median(wall_times) * 1000.0 if wall_times else 0.0
    wall_ms_mean = statistics.fmean(wall_times) * 1000.0 if wall_times else 0.0

    out: Dict[str, Any] = {
        "seed": int(seed),
        "n_samples": len(fold_samples),
        "n_runs": int(n_runs),
        "accuracy": float(accuracy),
        "wall_time_ms_median": float(wall_ms_median),
        "wall_time_ms_mean": float(wall_ms_mean),
        "memory_mb": float(memory_mb),
        "status": status,
        "wall_time_total_s": float(elapsed),
    }
    if error_msg is not None:
        out["error"] = error_msg
    return out


# ---------------------------------------------------------------------------
# Per-module CV — sweep 5 folds and aggregate.
# ---------------------------------------------------------------------------
def _per_module_cv(
    module_id: str,
    runner: Callable[[List[str], int], List[float]] | None,
    folds: List[List[str]],
    fold_seeds: List[int],
    n_runs_per_fold: int,
) -> Dict[str, Any]:
    """Run ``module_id`` across ``folds`` and aggregate per-fold metrics."""
    fold_records: List[Dict[str, Any]] = []
    for fold_idx, (fold_samples, seed) in enumerate(zip(folds, fold_seeds)):
        rec = _run_fold(runner, fold_samples, n_runs_per_fold, seed)
        rec["fold_idx"] = int(fold_idx)
        fold_records.append(rec)

    mean = _aggregate_mean(fold_records)
    std = _aggregate_std(fold_records, mean)
    return {
        "folds": fold_records,
        "mean": mean,
        "std": std,
    }


def _aggregate_mean(fold_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute the per-module mean over folds for each metric."""
    keys = ("accuracy", "wall_time_ms_median", "wall_time_ms_mean", "memory_mb")
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in fold_records]
        out[k] = float(statistics.fmean(vals)) if vals else 0.0
    return out


def _aggregate_std(
    fold_records: List[Dict[str, Any]],
    mean: Dict[str, float],
) -> Dict[str, float]:
    """Compute the per-module std (sample std, ddof=1) over folds."""
    keys = ("accuracy", "wall_time_ms_median", "wall_time_ms_mean", "memory_mb")
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in fold_records]
        if len(vals) < 2:
            out[k] = 0.0
        else:
            out[k] = float(statistics.stdev(vals))
    return out


# ---------------------------------------------------------------------------
# Cross-module aggregate — mean / std across modules.
# ---------------------------------------------------------------------------
def _cross_module_aggregate(
    per_module: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute mean ± std of per-module (mean, std) across all modules."""
    keys = ("accuracy", "wall_time_ms_median", "wall_time_ms_mean", "memory_mb")
    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}
    if not per_module:
        return mean_metrics, std_metrics

    # Use each module's mean across folds.
    module_means = [per_module[mid]["mean"] for mid in per_module]
    for k in keys:
        vals = [float(m[k]) for m in module_means]
        mean_metrics[f"{k}_mean"] = float(statistics.fmean(vals))
        mean_metrics[f"{k}_min"] = float(min(vals)) if vals else 0.0
        mean_metrics[f"{k}_max"] = float(max(vals)) if vals else 0.0
    # Std of the per-module mean: how much do module means vary?
    for k in keys:
        vals = [float(m[k]) for m in module_means]
        if len(vals) < 2:
            std_metrics[f"{k}_std"] = 0.0
        else:
            std_metrics[f"{k}_std"] = float(statistics.stdev(vals))
    return mean_metrics, std_metrics


# ---------------------------------------------------------------------------
# Top-level harness — drive 5-fold CV across all 32 modules.
# ---------------------------------------------------------------------------
def run_cv(
    csv_path: Path = DEFAULT_CSV,
    n_folds: int = DEFAULT_N_FOLDS,
    n_runs_per_fold: int = DEFAULT_N_RUNS_PER_FOLD,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
    modules: List[str] | None = None,
) -> Dict[str, Any]:
    """Execute the 5-fold CV across R18 modules and return the JSON payload.

    Parameters
    ----------
    csv_path
        PlatinAI 5K CSV path; falls back to a 10-SMILES synthetic
        kitchen-sink if the file is missing.
    n_folds
        Number of CV folds (default 5).
    n_runs_per_fold
        Number of forward calls per (module, fold) cell (default 5).
    n_samples
        Total PlatinAI samples to load; sliced into ``n_folds``
        contiguous subsets (default 100).
    seed
        Master seed; each fold uses ``seed + fold_idx`` as its RNG.
    modules
        Optional subset of module IDs to evaluate (default: all 32).
    """
    _ensure_repo_root_on_path()

    registry = _build_module_registry()
    if modules:
        wanted = set(modules)
        registry = {k: v for k, v in registry.items() if k in wanted}
        if not registry:
            raise RuntimeError(f"no modules matched {modules!r}")

    # Load / build the sample batch.
    samples = _load_platinai_smiles(csv_path, n_samples)
    used_synthetic = False
    if not samples:
        samples = _make_synthetic_smiles(n_samples, seed)
        used_synthetic = True
    folds = _build_folds(samples, n_folds)
    fold_seeds = [seed + k for k in range(n_folds)]

    # Run the CV.
    tracemalloc.start()
    base_mem = tracemalloc.get_traced_memory()[0]
    t_total_0 = time.perf_counter()
    per_module: Dict[str, Dict[str, Any]] = {}
    for mod_id, runner in registry.items():
        per_module[mod_id] = _per_module_cv(
            module_id=mod_id,
            runner=runner,
            folds=folds,
            fold_seeds=fold_seeds,
            n_runs_per_fold=n_runs_per_fold,
        )
    total_wall_s = time.perf_counter() - t_total_0
    peak_mem_total = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    mean_metrics, std_metrics = _cross_module_aggregate(per_module)

    # Build payload.
    try:
        import torch as _torch

        torch_ver = _torch.__version__
    except Exception:  # noqa: BLE001
        torch_ver = "unavailable"

    payload: Dict[str, Any] = {
        "n_modules": int(len(registry)),
        "n_folds": int(n_folds),
        "n_runs_per_fold": int(n_runs_per_fold),
        "n_samples_total": int(len(samples)),
        "csv_path": str(csv_path),
        "used_synthetic_smiles": bool(used_synthetic),
        "module_groups": _MODULE_GROUPS,
        "per_module_per_fold_metrics": per_module,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "fold_seeds": fold_seeds,
        "fold_sizes": [len(f) for f in folds],
        "total_wall_time_s": float(total_wall_s),
        "peak_memory_mb_total": float(peak_mem_total - base_mem) / (1024 * 1024),
        "seed": int(seed),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "torch": torch_ver,
    }
    return payload


def _ensure_repo_root_on_path() -> None:
    """Insert the repo root into ``sys.path`` if missing (idempotent)."""
    rr = str(_REPO_ROOT)
    if rr not in sys.path:
        sys.path.insert(0, rr)


# ---------------------------------------------------------------------------
# Output renderers — text summary + JSON writer.
# ---------------------------------------------------------------------------
def _render_text(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("R18 5-fold CV harness")
    lines.append("=" * 72)
    lines.append(f"timestamp       : {payload['timestamp_utc']}")
    lines.append(f"n_modules       : {payload['n_modules']}")
    lines.append(f"n_folds         : {payload['n_folds']}")
    lines.append(f"n_runs_per_fold : {payload['n_runs_per_fold']}")
    lines.append(f"n_samples_total : {payload['n_samples_total']}")
    lines.append(f"used_synthetic  : {payload['used_synthetic_smiles']}")
    lines.append(f"total wall (s)  : {payload['total_wall_time_s']:.2f}")
    lines.append(f"peak mem (MB)   : {payload['peak_memory_mb_total']:.2f}")
    lines.append("")
    mm = payload["mean_metrics"]
    sm = payload["std_metrics"]
    lines.append("cross-module aggregates (mean / std of per-module mean):")
    lines.append(
        f"  accuracy_mean           = {mm.get('accuracy_mean', 0.0):.4f}  "
        f"(std across modules = {sm.get('accuracy_std', 0.0):.4f})"
    )
    lines.append(
        f"  wall_ms_median_mean     = {mm.get('wall_time_ms_median_mean', 0.0):.4f}  "
        f"(std = {sm.get('wall_time_ms_median_std', 0.0):.4f})"
    )
    lines.append(
        f"  wall_ms_mean_mean       = {mm.get('wall_time_ms_mean_mean', 0.0):.4f}  "
        f"(std = {sm.get('wall_time_ms_mean_std', 0.0):.4f})"
    )
    lines.append(
        f"  memory_mb_mean          = {mm.get('memory_mb_mean', 0.0):.4f}  "
        f"(std = {sm.get('memory_mb_std', 0.0):.4f})"
    )
    lines.append("")
    lines.append("per-module (mean across folds):")
    for mid, rec in payload["per_module_per_fold_metrics"].items():
        m = rec["mean"]
        s = rec["std"]
        lines.append(
            f"  {mid:<48s}  acc={m['accuracy']:.2f} "
            f"±{s['accuracy']:.2f}  "
            f"wall_ms={m['wall_time_ms_median']:.3f} "
            f"±{s['wall_time_ms_median']:.3f}  "
            f"mem_mb={m['memory_mb']:.3f} ±{s['memory_mb']:.3f}"
        )
    return "\n".join(lines)


def _write_json(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R18 5-fold CV harness across 32 modules.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of CV folds (default: 5).",
    )
    parser.add_argument(
        "--n-runs-per-fold",
        type=int,
        default=DEFAULT_N_RUNS_PER_FOLD,
        help="Forward calls per module per fold (default: 5).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Total PlatinAI samples (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Master RNG seed (default: 42).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV),
        help="PlatinAI 5K CSV path (default: molmetal/data/platinai_5k.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPORT_DIR),
        help="Output directory for results.json (default: reports/wf_cv_harness).",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="results.json",
        help="Output JSON file name (default: results.json).",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="",
        help="Comma-separated subset of module IDs (default: all 32).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print JSON only; do not render the text summary.",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)
    modules_arg = (
        [m.strip() for m in args.modules.split(",") if m.strip()]
        if args.modules
        else None
    )

    payload = run_cv(
        csv_path=csv_path,
        n_folds=max(1, int(args.n_folds)),
        n_runs_per_fold=max(1, int(args.n_runs_per_fold)),
        n_samples=max(1, int(args.n_samples)),
        seed=int(args.seed),
        modules=modules_arg,
    )

    out_path = Path(args.out_dir) / args.out_file
    _write_json(payload, out_path)

    if args.json_only:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_render_text(payload))
        print("")
        print(f"JSON written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
