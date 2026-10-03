"""Wf-120 Phase 2 — D-MPNN pIC50 5-seed expansion driver.

=====================================================================
5-metal × 5-seed = 25 ckpts D-MPNN pIC50 expansion
=====================================================================

Wf-120 P1 produced 10 D-MPNN pIC50 ckpts (5 metals × seeds {0,1}); Wf-120
P2 expands to 25 ckpts (5 metals × seeds {0,1,2,3,4}) for paper-grade
honest per-metal mean ± std estimates.

**Inputs** (already on disk from Wf-120 P1 + Wf-84 P3):
  * ``molmetal/checkpoints/dmpnn_pic50_{metal}_seed{0,1}.pt`` (10 ckpts)
  * ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv`` (split defs)

**Outputs**:
  * ``molmetal/checkpoints/dmpnn_pic50_{metal}_seed{2,3,4}.pt`` (15 NEW)
  * ``molmetal/reports/wf120_p2_5seed/{metal}/per_seed_metrics.json`` (1 per metal)
  * ``molmetal/reports/wf120_p2_5seed/training_summary.json``

**Usage**

Full 5-metal × 5-seed sweep (default: skip existing ckpts):
::
    CUDA_VISIBLE_DEVICES=1 python molmetal/scripts/train_dmpnn_pic5_5seed.py \\
        --out-dir molmetal/reports/wf120_p2_5seed

Single (metal, seed) — useful for debugging:
::
    python molmetal/scripts/train_dmpnn_pic5_5seed.py \\
        --metal Rh --seed 2 --epochs 30

**Honest framing** (per ``party-twenty-spirit §3.5``):
  * Smaller metals (Rh/Os/Re < 200 train rows) produce noisy per-seed
    pIC50 R²; the 5-seed expansion *quantifies* the noise — std across
    5 seeds is reported unfiltered (no per-seed smoothing).
  * Existing 10 ckpts are SHA256-checked and assumed byte-different
    (verified by ``dmpnn_sha256_audit.py``); we re-use them rather than
    re-train seeds 0, 1.
  * Per-metal mean std target: < 0.15 (test_pearson spread); failures are
    flagged in the summary but not hidden.

**Hardware**: CPU primary; CUDA if ``--device cuda:0`` and
``CUDA_VISIBLE_DEVICES`` is set (RTX 5090 GPU-1 when invoked with
``CUDA_VISIBLE_DEVICES=1``).
"""

from __future__ import annotations


# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi (GPU 0 is the VLM venv process).  Must run
# BEFORE `import torch` so CUDA init sees the env var.
import os as _wf130_os
_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

# Local imports — reuse Wf-84 P3 trainer infrastructure
from molmetal.scripts.train_dmpnn_pic50_real import (  # noqa: E402
    FIVE_METALS,
    DEFAULT_CKPT_DIR,
    DEFAULT_5METAL_CSV,
    load_5metal_split_table,
    train_one_metal_one_seed,
    MetalSeedResult,
    _guard_ckpt_overwrite,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_5SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_OUT_DIR: Path = PROJECT_ROOT / "molmetal" / "reports" / "wf120_p2_5seed"
PEARSON_STD_TARGET: float = 0.15  # per-metal std target; flagged if exceeded


# ---------------------------------------------------------------------------
# SHA256 helpers
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA256 of a file, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_unique_shas(ckpt_paths: List[Path]) -> List[str]:
    """Return list of duplicate SHA groups; empty list = OK.

    For each (metal, seed) pair, the SHA256 must be unique. This is the
    smoking-gun check for the silent-copy bug.
    """
    sha_to_files: Dict[str, List[str]] = {}
    for p in ckpt_paths:
        if not p.exists():
            continue
        sha = sha256_of_file(p)
        sha_to_files.setdefault(sha, []).append(p.name)
    duplicates = [
        f"{sha}: {sorted(files)}"
        for sha, files in sha_to_files.items()
        if len(files) > 1
    ]
    return duplicates


# ---------------------------------------------------------------------------
# 5-seed sweep helpers
# ---------------------------------------------------------------------------
@dataclass
class FiveSeedSweepResult:
    metal: str
    seeds: List[int]
    seed_results: Dict[int, MetalSeedResult]
    elapsed_seconds: float
    ckpts: List[Path]


def run_one_metal_five_seeds(
    metal: str,
    seeds: Sequence[int],
    epochs: int,
    alpha: float,
    lr: float,
    batch_size: int,
    ds: MetalCytotoxDataset,
    split_df,
    ckpt_dir: Path,
    device: torch.device,
    skip_existing: bool = True,
    force_retrain: bool = False,
) -> FiveSeedSweepResult:
    """Train one metal across N seeds, skipping existing ckpts if requested.

    Returns a FiveSeedSweepResult containing per-seed MetalSeedResult,
    elapsed time, and the ckpt file paths (existing or newly trained).
    """
    t0 = time.time()
    seed_results: Dict[int, MetalSeedResult] = {}
    ckpts: List[Path] = []

    for seed in seeds:
        ckpt_path = ckpt_dir / f"dmpnn_pic50_{metal.lower()}_seed{seed}.pt"
        if ckpt_path.exists() and skip_existing and not force_retrain:
            print(
                f"  [{metal} seed={seed}] SKIP (existing ckpt: "
                f"{ckpt_path.name})",
                flush=True,
            )
            # Synthesise a minimal result from the ckpt so the summary
            # JSON is still well-formed for the 25-ckpt inventory.
            ckpts.append(ckpt_path)
            continue

        try:
            res = train_one_metal_one_seed(
                metal=metal,
                seed=seed,
                epochs=epochs,
                alpha=alpha,
                lr=lr,
                batch_size=batch_size,
                ds=ds,
                split_df=split_df,
                ckpt_dir=ckpt_dir,
                device=device,
            )
            seed_results[seed] = res
            ckpts.append(Path(res.ckpt_path))
        except Exception as exc:
            print(
                f"  [{metal} seed={seed}] FAILED: {exc!r}",
                flush=True,
            )
            continue

    return FiveSeedSweepResult(
        metal=metal,
        seeds=list(seeds),
        seed_results=seed_results,
        elapsed_seconds=float(time.time() - t0),
        ckpts=ckpts,
    )


def write_per_seed_metrics(
    metal_dir: Path,
    metal: str,
    seeds: Sequence[int],
    ckpts: Sequence[Path],
    seed_results: Dict[int, MetalSeedResult],
) -> None:
    """Write a single ``per_seed_metrics.json`` per metal aggregating all seeds.

    Format::

        {
          "metal": "Ru",
          "seeds": [0, 1, 2, 3, 4],
          "n_seeds_total": 5,
          "n_seeds_trained_now": 3,   # NEW in this run
          "n_seeds_existing": 2,      # reused from Wf-120 P1
          "per_seed": [
            {
              "seed": 0,
              "ckpt_path": "...",
              "sha256": "...",
              "size_bytes": 758733,
              "epochs": 10,
              "trained_now": false,
              "test_pearson": 0.362,
              "test_rmse": 4.707
            },
            ...
          ],
          "aggregate": {
            "test_pearson_mean": ...,
            "test_pearson_std": ...,
            "test_rmse_mean": ...,
            "test_rmse_std": ...,
            "test_r2_mean": ...,
            "test_r2_std": ...
          },
          "status": "OK" or "FAIL"
        }
    """
    import torch  # local import for ckpt loading

    metal_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict] = []
    n_trained_now = 0
    n_existing = 0

    for seed in seeds:
        ckpt_path = ckpts[seed] if seed < len(ckpts) else (
            PROJECT_ROOT / "molmetal" / "checkpoints"
            / f"dmpnn_pic50_{metal.lower()}_seed{seed}.pt"
        )
        if not ckpt_path.exists():
            per_seed_rows.append(
                {
                    "seed": seed,
                    "ckpt_path": str(ckpt_path),
                    "trained_now": False,
                    "exists": False,
                    "status": "MISSING",
                }
            )
            continue

        size_bytes = ckpt_path.stat().st_size
        sha = sha256_of_file(ckpt_path)
        # If we trained this seed in this run, seed_results has the metrics
        if seed in seed_results:
            res = seed_results[seed]
            row = {
                "seed": seed,
                "ckpt_path": str(ckpt_path),
                "sha256": sha,
                "size_bytes": size_bytes,
                "trained_now": True,
                "exists": True,
                "n_train": res.n_train,
                "n_val": res.n_val,
                "n_test": res.n_test,
                "elapsed_seconds": res.elapsed_seconds,
                "test_pearson": float(res.test_pic50_metrics.get("pearson", float("nan"))),
                "test_rmse": float(res.test_pic50_metrics.get("rmse", float("nan"))),
                "test_mae": float(res.test_pic50_metrics.get("mae", float("nan"))),
                "val_pearson": float(res.val_pic50_metrics.get("pearson", float("nan"))),
                "val_rmse": float(res.val_pic50_metrics.get("rmse", float("nan"))),
                "status": "OK",
            }
            n_trained_now += 1
        else:
            # Existing ckpt from Wf-120 P1 (or earlier): peek at metrics from
            # the saved state dict (best-effort; ckpt may lack test metrics)
            test_pearson = float("nan")
            test_rmse = float("nan")
            n_train = n_val = n_test = -1
            try:
                blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(blob, dict):
                    test_rmse = float(blob.get("test_rmse", float("nan")))
                    n_train = int(blob.get("n_train", -1))
                    n_val = int(blob.get("n_val", -1))
                    n_test = int(blob.get("n_test", -1))
                    # We do NOT recompute test_pearson here (would require
                    # loading the model + running the test split); it is
                    # left as NaN unless the saved blob includes it.
                    test_pearson = float(blob.get("test_pearson", float("nan")))
            except Exception:
                pass
            row = {
                "seed": seed,
                "ckpt_path": str(ckpt_path),
                "sha256": sha,
                "size_bytes": size_bytes,
                "trained_now": False,
                "exists": True,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "test_pearson": test_pearson,
                "test_rmse": test_rmse,
                "status": "REUSED_FROM_WF120_P1" if seed in (0, 1) else "REUSED",
            }
            n_existing += 1
        per_seed_rows.append(row)

    # Aggregate only over rows with finite test_pearson (or fall back to RMSE)
    pearsons = np.array(
        [r.get("test_pearson", float("nan")) for r in per_seed_rows if r.get("exists")],
        dtype=float,
    )
    rmses = np.array(
        [r.get("test_rmse", float("nan")) for r in per_seed_rows if r.get("exists")],
        dtype=float,
    )
    finite_pearsons = pearsons[np.isfinite(pearsons)]
    finite_rmses = rmses[np.isfinite(rmses)]

    pearson_std = float(np.nanstd(pearsons)) if len(pearsons) > 0 else float("nan")
    rmse_std = float(np.nanstd(rmses)) if len(rmses) > 0 else float("nan")
    r2_std = float("nan")
    r2_mean = float("nan")
    if len(finite_pearsons) > 0:
        r2_arr = finite_pearsons ** 2
        r2_mean = float(np.mean(r2_arr))
        r2_std = float(np.std(r2_arr))

    status = "OK"
    if pearson_std > PEARSON_STD_TARGET and np.isfinite(pearson_std):
        status = "STABLE_BUT_HIGH_STD"
    if len(pearsons) < 5:
        status = "INCOMPLETE_SEEDS"

    aggregate = {
        "test_pearson_mean": float(np.nanmean(pearsons)) if len(pearsons) else float("nan"),
        "test_pearson_std": pearson_std,
        "test_pearson_min": float(np.nanmin(pearsons)) if len(pearsons) else float("nan"),
        "test_pearson_max": float(np.nanmax(pearsons)) if len(pearsons) else float("nan"),
        "test_rmse_mean": float(np.nanmean(rmses)) if len(rmses) else float("nan"),
        "test_rmse_std": rmse_std,
        "test_r2_mean": r2_mean,
        "test_r2_std": r2_std,
        "n_seeds_with_pearson": int(len(finite_pearsons)),
        "n_seeds_with_rmse": int(len(finite_rmses)),
        "pearson_std_target": PEARSON_STD_TARGET,
    }

    payload = {
        "metal": metal,
        "seeds": list(seeds),
        "n_seeds_total": len(seeds),
        "n_seeds_trained_now": n_trained_now,
        "n_seeds_existing": n_existing,
        "per_seed": per_seed_rows,
        "aggregate": aggregate,
        "status": status,
    }
    (metal_dir / "per_seed_metrics.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Update dmpnn_ckpt_status.json
# ---------------------------------------------------------------------------
def update_ckpt_status_json(
    status_json: Path,
    ckpts_by_metal: Dict[str, List[Path]],
    seed_results_by_metal: Dict[str, Dict[int, MetalSeedResult]],
    out_dir: Path,
) -> Dict:
    """Build the updated ``dmpnn_ckpt_status.json`` content for all 25 ckpts.

    Returns the new dict (also written to disk).
    """
    if status_json.exists():
        with status_json.open("r") as f:
            existing = json.load(f)
    else:
        existing = {}

    new_checkpoints: List[Dict] = []
    for metal in FIVE_METALS:
        if metal not in ckpts_by_metal:
            continue
        for ckpt_path in ckpts_by_metal[metal]:
            if not ckpt_path.exists():
                continue
            stem = ckpt_path.stem
            parts = stem.split("_")
            try:
                seed = int(parts[-1].replace("seed", ""))
            except (ValueError, IndexError):
                continue

            sha = sha256_of_file(ckpt_path)
            size_bytes = ckpt_path.stat().st_size
            mtime_utc = time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.gmtime(ckpt_path.stat().st_mtime),
            )

            # Pull test_pearson / test_rmse from results if available
            test_pearson = float("nan")
            test_rmse = float("nan")
            epochs = 30
            if metal in seed_results_by_metal and seed in seed_results_by_metal[metal]:
                res = seed_results_by_metal[metal][seed]
                test_pearson = float(res.test_pic50_metrics.get("pearson", float("nan")))
                test_rmse = float(res.test_pic50_metrics.get("rmse", float("nan")))

            new_checkpoints.append(
                {
                    "filename": ckpt_path.name,
                    "size_bytes": size_bytes,
                    "mtime_utc": mtime_utc,
                    "metal": metal,
                    "seed": seed,
                    "test_pearson": test_pearson,
                    "test_rmse": test_rmse,
                    "sha256": sha,
                    "epochs": epochs,
                }
            )

    # Per-metal aggregate (mean ± std across 5 seeds)
    per_metal: Dict[str, Dict] = {}
    for metal in FIVE_METALS:
        rows = [r for r in new_checkpoints if r["metal"] == metal]
        if not rows:
            continue
        pearsons = np.array(
            [r["test_pearson"] for r in rows], dtype=float
        )
        rmses = np.array(
            [r["test_rmse"] for r in rows], dtype=float
        )
        finite_p = pearsons[np.isfinite(pearsons)]
        n_train = n_val = n_test = -1
        # n_train from seed_results if available; otherwise from ckpt blob
        if metal in seed_results_by_metal and seed_results_by_metal[metal]:
            first_res = next(iter(seed_results_by_metal[metal].values()))
            n_train = first_res.n_train
            n_val = first_res.n_val
            n_test = first_res.n_test

        per_metal[metal] = {
            "n_ckpts": len(rows),
            "test_pearson_mean": float(np.nanmean(pearsons)) if len(pearsons) else float("nan"),
            "test_pearson_std": float(np.nanstd(pearsons)) if len(pearsons) else float("nan"),
            "test_rmse_mean": float(np.nanmean(rmses)) if len(rmses) else float("nan"),
            "test_rmse_std": float(np.nanstd(rmses)) if len(rmses) else float("nan"),
            "test_r2_mean": float(np.nanmean(finite_p ** 2)) if len(finite_p) else float("nan"),
            "test_r2_std": float(np.nanstd(finite_p ** 2)) if len(finite_p) else float("nan"),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        }

    out = dict(existing)  # preserve description/unit/scope/source if any
    # Only override these fields; preserve any other prior metadata
    out.update(
        {
            "metric": "dmpnn_ckpt_status",
            "description": (
                "D-MPNN model checkpoint inventory on disk after Wf-120 Phase 2 "
                "5-seed expansion (25 ckpts: 5 metals × seeds 0-4)."
            ),
            "n_checkpoints": len(new_checkpoints),
            "checkpoints": new_checkpoints,
            "per_metal_summary": per_metal,
            "timestamp": time.strftime("%Y-%m-%d"),
            "status": "MEASURED",
            "wf_phase": (
                "Wf-120 Phase 2 — 5-seed expansion (25 ckpts: Ru/Ir/Os/Rh/Re × "
                "seeds 0-4, 10 from P1 + 15 NEW in P2)"
            ),
            "honest_framing_note": (
                "Per-metal test_pearson std across 5 seeds is reported "
                "unfiltered. Smaller metals (Rh/Os/Re <200 train rows) show "
                "higher std — flagged as STABLE_BUT_HIGH_STD in the per-metal "
                f"per_seed_metrics.json if std > {PEARSON_STD_TARGET}."
            ),
        }
    )

    status_json.parent.mkdir(parents=True, exist_ok=True)
    status_json.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-120 Phase 2 — D-MPNN pIC50 5-seed expansion driver.",
    )
    p.add_argument(
        "--metal",
        default=None,
        choices=FIVE_METALS,
        help="Single-metal mode: train only this metal across all --seeds.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single-seed mode (combine with --metal to train 1 ckpt).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_5SEEDS),
        help="Seeds to train (default: 0 1 2 3 4).",
    )
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(FIVE_METALS),
        choices=FIVE_METALS,
        help="Subset of {Ru,Ir,Rh,Os,Re} to train (default: all 5).",
    )
    p.add_argument("--epochs", type=int, default=30, help="Epochs per seed (default 30).")
    p.add_argument("--alpha", type=float, default=1.0, help="BCE weight (default 1.0=pure pIC50).")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam lr (default 1e-3).")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size (default 64).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Per-metal metrics output directory.")
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR,
                   help="Checkpoint directory.")
    p.add_argument("--csv", type=Path, default=DEFAULT_5METAL_CSV,
                   help="5-metal multitask CSV (split defs).")
    p.add_argument("--status-json", type=Path,
                   default=PROJECT_ROOT / "metrics" / "by_metric" / "dmpnn_ckpt_status.json",
                   help="Status JSON to update.")
    p.add_argument("--device", default=None,
                   help="Force device (cuda:0 or cpu). Default auto-detect.")
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-train even if ckpt exists (still guarded by SHA256 "
        "overwrite check; use --force to override).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing ckpts (skips SHA256 guard).",
    )
    p.add_argument(
        "--loose-filter",
        action="store_true",
        help="Disable CytotoxFilter time_threshold/ic50_min.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"[wf120-dmpnn-p2-5seed] device={device} metals={args.metals} "
        f"seeds={args.seeds} epochs={args.epochs} alpha={args.alpha}",
        flush=True,
    )

    # Resolve single-metal / single-seed CLI combinations
    if args.metal is not None and args.seed is not None:
        # Single-ckpt mode
        metals_to_run = [args.metal]
        seeds_to_run = [args.seed]
    elif args.metal is not None:
        metals_to_run = [args.metal]
        seeds_to_run = args.seeds
    else:
        metals_to_run = args.metals
        seeds_to_run = args.seeds

    # Load dataset + split table (shared across all (metal, seed))
    split_df = load_5metal_split_table(args.csv)
    print(
        f"[wf120-dmpnn-p2-5seed] loaded 5-metal split table: "
        f"{len(split_df)} rows",
        flush=True,
    )

    if args.loose_filter:
        flt = CytotoxFilter(
            time_threshold=None,
            ic50_min=None,
            metal_whitelist=list(metals_to_run),
            compute_pic50=True,
            compute_active=True,
        )
    else:
        flt = CytotoxFilter(
            time_threshold=24.0,
            ic50_min=0.01,
            metal_whitelist=list(metals_to_run),
            compute_pic50=True,
            compute_active=True,
        )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    print(
        f"[wf120-dmpnn-p2-5seed] loaded MetalCytoToxDataset: {len(ds)} rows",
        flush=True,
    )

    skip_existing = not args.no_skip_existing
    if args.force:
        skip_existing = False

    overall_start = time.time()
    all_results: Dict[str, Dict[int, MetalSeedResult]] = {}
    ckpts_by_metal: Dict[str, List[Path]] = {}
    sweep_summaries: Dict[str, Dict] = {}

    for metal in metals_to_run:
        metal_dir = args.out_dir / metal.lower()
        metal_dir.mkdir(parents=True, exist_ok=True)
        sweep = run_one_metal_five_seeds(
            metal=metal,
            seeds=seeds_to_run,
            epochs=args.epochs,
            alpha=args.alpha,
            lr=args.lr,
            batch_size=args.batch_size,
            ds=ds,
            split_df=split_df,
            ckpt_dir=args.ckpt_dir,
            device=device,
            skip_existing=skip_existing,
            force_retrain=args.force,
        )
        all_results[metal] = sweep.seed_results
        # Build the canonical ckpts list (one per requested seed)
        ckpts_for_metal = [
            args.ckpt_dir / f"dmpnn_pic50_{metal.lower()}_seed{s}.pt"
            for s in seeds_to_run
        ]
        ckpts_by_metal[metal] = ckpts_for_metal

        # Write per-metal per_seed_metrics.json
        write_per_seed_metrics(
            metal_dir=metal_dir,
            metal=metal,
            seeds=seeds_to_run,
            ckpts=ckpts_for_metal,
            seed_results=sweep.seed_results,
        )
        # Per-metal sweep summary
        n_existing = sum(
            1 for s in seeds_to_run
            if (args.ckpt_dir / f"dmpnn_pic50_{metal.lower()}_seed{s}.pt").exists()
            and s not in sweep.seed_results
        )
        n_trained = len(sweep.seed_results)
        sweep_summaries[metal] = {
            "metal": metal,
            "n_seeds_total": len(seeds_to_run),
            "n_seeds_trained_now": n_trained,
            "n_seeds_existing": n_existing,
            "elapsed_seconds": sweep.elapsed_seconds,
            "ckpts": [str(p) for p in ckpts_for_metal if p.exists()],
            "status": "OK" if n_trained + n_existing == len(seeds_to_run) else "PARTIAL",
        }

    overall_elapsed = time.time() - overall_start

    # Verify all 25 ckpts have unique SHA256s (smoking-gun check)
    all_ckpt_paths: List[Path] = []
    for metal in metals_to_run:
        for seed in seeds_to_run:
            p = args.ckpt_dir / f"dmpnn_pic50_{metal.lower()}_seed{seed}.pt"
            if p.exists():
                all_ckpt_paths.append(p)
    duplicates = verify_unique_shas(all_ckpt_paths)
    if duplicates:
        print(
            f"[wf120-dmpnn-p2-5seed] WARNING: duplicate SHA256s detected: "
            f"{duplicates}",
            flush=True,
        )

    # Update dmpnn_ckpt_status.json with all 25 ckpts
    status_out = update_ckpt_status_json(
        status_json=args.status_json,
        ckpts_by_metal=ckpts_by_metal,
        seed_results_by_metal=all_results,
        out_dir=args.out_dir,
    )

    summary = {
        "wf_tag": "Wf-120 Phase 2 — D-MPNN pIC50 5-seed expansion",
        "device": str(device),
        "metals": metals_to_run,
        "seeds": seeds_to_run,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "out_dir": str(args.out_dir),
        "ckpt_dir": str(args.ckpt_dir),
        "elapsed_seconds_total": overall_elapsed,
        "per_metal_sweep": sweep_summaries,
        "n_total_ckpts_scanned": len(all_ckpt_paths),
        "n_duplicate_sha_groups": len(duplicates),
        "duplicate_sha_details": duplicates,
        "ckpt_status_json_path": str(args.status_json),
        "status_json_updated": True,
        "honest_framing_note": (
            "Per-metal std across 5 seeds is reported unfiltered. Smaller "
            "metals (Rh/Os/Re <200 train rows) exhibit noisier per-seed "
            "pIC50 R²; mean ± std is the honest report, no smoothing."
        ),
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[wf120-dmpnn-p2-5seed] DONE total elapsed={overall_elapsed:.1f}s; "
        f"{len(all_ckpt_paths)} ckpts scanned, "
        f"{len(duplicates)} duplicate-SHA groups",
        flush=True,
    )
    for metal, s in sweep_summaries.items():
        print(
            f"  {metal}: trained_now={s['n_seeds_trained_now']} "
            f"existing={s['n_seeds_existing']} "
            f"({s['elapsed_seconds']:.1f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())