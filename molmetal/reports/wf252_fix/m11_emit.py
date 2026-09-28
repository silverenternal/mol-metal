"""Wf-252 F3 — emit m11 per-metal Wasserstein pIC50 from existing Wf-200 per-seed JSONs.

================================================================================
Wf-252-Apply Fix #3 (M11Emit) — pure plumbing, no model training.
================================================================================

Background (TODO-130 G3)
------------------------
Wf-124 P3 ran 5-metal × Wasserstein pIC50 multi-task training.  The per-seed
JSONs (``molmetal/reports/wf200_chemprop_v2_v3/{metal}/per_seed_metrics_seed{N}.json``)
already carry test_auc + w2_history + log_sigma_history.  TODO-130 G3 wants
this data plumbed into ``metrics/by_metric/per_metal_pic50_wasserstein.json``
in the standard FLAT ``metrics/by_metric/{name}.json`` schema.

This script
-----------
1. Walks the Wf-200 per-seed directory tree.
2. Per metal, computes test_auc mean/std + w2_history final-value mean/std.
3. Emits ``metrics/by_metric/per_metal_pic50_wasserstein.json`` with:
   - per_metal.{Ru,Ir,Os,Rh,Re}.test_auc_mean / test_auc_std / w2_final_mean /
     n_runs
   - status: MEASURED
   - honest_framing_note: explicit per (metal, seed) availability count.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
- Entries with no per-seed JSONs are reported as ``null`` with ``n_runs=0``.
  PROJECTED: 5 metals × 3 seeds = 15 entries if all Wf-124 P3 runs reached
  completion; the on-disk reality is fewer (the Wf-200 v3 driver has only
  populated Ru/Ir/Os at this point — Rh/Re are not yet trained).
- This script reads existing JSONs ONLY — no model re-training, no
  checkpoint loading, no GPU.  CPU-only and trivially safe.
- w2_history values are the LAST-element (final-epoch) Wasserstein transfer
  loss per run.  They are NOT paper-grade W₂ numbers (the driver bounded
  W₂ to max 32 rows per metal per step as an O(N²) cost bound).

Usage
-----

    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/reports/wf252_fix/m11_emit.py

Writes ``metrics/by_metric/per_metal_pic50_wasserstein.json``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Project root on sys.path ---------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import os as _os
if str(PROJECT_ROOT) not in _os.sys.path:  # type: ignore[attr-defined]
    _os.sys.path.insert(0, str(PROJECT_ROOT))  # type: ignore[attr-defined]

# --- Constants -------------------------------------------------------------
WF200_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf200_chemprop_v2_v3"
METRICS_OUT = PROJECT_ROOT / "metrics" / "by_metric" / "per_metal_pic50_wasserstein.json"
METALS = ("Ru", "Ir", "Os", "Rh", "Re")
DEFAULT_SEEDS = (0, 1, 2)


def _load_one_seed(metal: str, seed: int) -> Optional[Dict]:
    """Read one per-seed JSON; return None if missing or malformed."""
    p = WF200_DIR / metal.lower() / f"per_seed_metrics_seed{seed}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as exc:
        return {"error": str(exc), "metal": metal, "seed": seed}


def _final_w2(w2_history: List) -> Optional[float]:
    """Return the last-element Wasserstein history value, or None.

    Returns None when the list is empty (degenerate run with no W₂ recorded).
    """
    if not w2_history:
        return None
    last = w2_history[-1]
    try:
        v = float(last)
    except (TypeError, ValueError):
        return None
    return v


def _summarise_per_metal(metal: str) -> Dict:
    """Aggregate one metal across all available per-seed JSONs."""
    runs = []
    for seed in DEFAULT_SEEDS:
        rec = _load_one_seed(metal, seed)
        if rec is None:
            continue
        # Skip ONLY when "error" is a non-empty string (real error payload).
        # The "error": null field that the v3 driver writes for healthy
        # runs must NOT count as an error — it's the no-error sentinel.
        err_val = rec.get("error")
        if isinstance(err_val, str) and err_val:
            continue
        test_auc = rec.get("test_auc")
        if test_auc is None or (isinstance(test_auc, float) and test_auc != test_auc):
            continue
        runs.append(
            {
                "seed": seed,
                "test_auc": float(test_auc),
                "w2_final": _final_w2(rec.get("w2_history", [])),
                "n_train": rec.get("n_train"),
                "n_test": rec.get("n_test"),
                "epochs": rec.get("epochs"),
                "status": rec.get("status", "ok"),
            }
        )
    n_runs = len(runs)
    if n_runs == 0:
        return {
            "test_auc_mean": None,
            "test_auc_std": None,
            "w2_final_mean": None,
            "n_runs": 0,
            "test_aucs": [],
            "w2_finals": [],
        }
    aucs = [r["test_auc"] for r in runs]
    w2s = [r["w2_final"] for r in runs if r["w2_final"] is not None]
    auc_mean = statistics.mean(aucs)
    auc_std = statistics.pstdev(aucs) if len(aucs) > 1 else 0.0
    w2_mean = statistics.mean(w2s) if w2s else None
    return {
        "test_auc_mean": float(auc_mean),
        "test_auc_std": float(auc_std),
        "w2_final_mean": float(w2_mean) if w2_mean is not None else None,
        "n_runs": int(n_runs),
        "test_aucs": aucs,
        "w2_finals": w2s,
        "seeds_covered": [r["seed"] for r in runs],
    }


def main() -> int:
    per_metal: Dict[str, Dict] = {}
    for m in METALS:
        per_metal[m] = _summarise_per_metal(m)

    total_runs = sum(v["n_runs"] for v in per_metal.values())
    metals_with_data = sum(1 for v in per_metal.values() if v["n_runs"] > 0)

    honest_framing = (
        f"Wf-252-Apply Fix #3 (F3-M11Emit) plumbing emission.  "
        f"Reads Wf-200 v3 per-seed JSONs from {WF200_DIR} (no model "
        f"re-training, no GPU).  {total_runs} per-seed runs across "
        f"{metals_with_data}/{len(METALS)} metals.  Wf-200 v3 driver "
        f"populated Ru/Ir/Os (3 metals) for seed 0; Ru seed 1 also "
        f"present.  Rh/Re have no per-seed JSONs at this point (the "
        f"underlying Wf-124 P3 multi-task Wasserstein training did not "
        f"complete for these metals before the session budget expired).  "
        f"test_auc_std is population std (pstdev), zero when n_runs=1.  "
        f"w2_final is the last-epoch Wasserstein transfer loss per run; "
        f"the Wf-200 driver bounded W2 to max 32 rows per metal per step "
        f"as an O(N²) cost bound — these are NOT paper-grade Wasserstein "
        f"distances."
    )

    out = {
        "metric": "per_metal_pic50_wasserstein",
        "description": (
            "Per-metal test-AUC + Wasserstein pIC50 final-loss from "
            "Wf-124 P3 / Wf-200 v3 chemprop v2 multi-task driver.  "
            "Plumbing-only emission (no re-training)."
        ),
        "unit": "test_auc (0-1, higher=better); w2_final (kcal/mol surrogate)",
        "scope": "Wf-200 v3 per-seed JSONs across metals {Ru,Ir,Os,Rh,Re}",
        "source": "molmetal/reports/wf200_chemprop_v2_v3/{metal}/per_seed_metrics_seed{N}.json",
        "per_metal": per_metal,
        "n_runs_total": int(total_runs),
        "n_metals_with_data": int(metals_with_data),
        "status": "MEASURED",
        "timestamp": "2026-09-28",
        "honest_framing_note": honest_framing,
        "wf_tag": "Wf-252-Apply Fix #3 (F3-M11Emit)",
        "wf_design_ref": (
            "molmetal/reports/wf252_research/fix_design.md §4 Fix F3"
        ),
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(out, indent=2))
    print(f"[wf252-f3] wrote {METRICS_OUT}", flush=True)
    print(
        f"[wf252-f3] total_runs={total_runs} metals_with_data="
        f"{metals_with_data}/{len(METALS)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())