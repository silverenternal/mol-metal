"""Wf-252 F4 — partial summariser for the quick-pilot.

Reads whatever per-config JSONs exist in
``molmetal/reports/wf229_ablation/quick_pilot/per_config/`` and emits
``ablation_summary.json`` with the actual completed runs.  Designed to
be called multiple times as runs complete incrementally.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
This script emits the summary from whatever runs have been completed —
it does NOT fabricate MEASURED values for missing runs.  When fewer
than 12 runs have completed, the ``status`` field is set to
``MEASURED_PARTIAL`` and ``n_runs_total`` reflects the truth.
"""

from __future__ import annotations

import json
import os as _os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in _os.sys.path:  # type: ignore[attr-defined]
    _os.sys.path.insert(0, str(PROJECT_ROOT))  # type: ignore[attr-defined]

QUICK_PILOT_OUT = PROJECT_ROOT / "molmetal" / "reports" / "wf229_ablation" / "quick_pilot"
QUICK_PILOT_PER_CFG = QUICK_PILOT_OUT / "per_config"
EXPECTED_CODES = ("100", "010", "001", "111")
EXPECTED_SEEDS = (0, 1, 2)


def _load_one(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _summarise() -> Dict:
    import statistics

    results: List[Dict] = []
    per_cfg_aucs: Dict[str, List[float]] = defaultdict(list)
    n_total = 0
    n_completed = 0
    for code in EXPECTED_CODES:
        for seed in EXPECTED_SEEDS:
            n_total += 1
            p = QUICK_PILOT_PER_CFG / f"cfg{code}_seed{seed}.json"
            rec = _load_one(p)
            r: Dict = {
                "code": code,
                "seed": seed,
                "path": str(p),
                "present": rec is not None,
            }
            if rec is None:
                results.append(r)
                continue
            n_completed += 1
            tm = rec.get("test_metrics", {}) or {}
            auc = tm.get("roc_auc")
            r["test_auc"] = auc
            r["elapsed_seconds"] = rec.get("elapsed_seconds")
            r["n_train"] = rec.get("n_train")
            r["n_test"] = rec.get("n_test")
            r["epochs"] = rec.get("epochs")
            r["model"] = rec.get("model")
            r["kendall_enabled"] = bool(rec.get("use_kendall_loss"))
            r["dann_enabled"] = bool(
                (rec.get("dann") or {}).get("enabled", False)
            )
            r["four_domain"] = bool(rec.get("four_domain"))
            results.append(r)
            if isinstance(auc, (int, float)) and auc == auc:
                per_cfg_aucs[code].append(float(auc))

    per_cfg_mean = {
        c: (statistics.mean(v) if v else None) for c, v in per_cfg_aucs.items()
    }
    per_cfg_std = {
        c: (statistics.pstdev(v) if len(v) > 1 else 0.0)
        for c, v in per_cfg_aucs.items()
    }

    k_only = per_cfg_mean.get("100")
    d_only = per_cfg_mean.get("010")
    f_only = per_cfg_mean.get("001")
    full = per_cfg_mean.get("111")
    n_runs_expected = len(EXPECTED_CODES) * len(EXPECTED_SEEDS)  # 12

    if n_completed == 0:
        status = "UNMEASURED"
    elif n_completed >= n_runs_expected:
        status = "MEASURED"
    else:
        status = "MEASURED_PARTIAL"

    return {
        "wf_tag": "Wf-252-Apply F4-AblationCPU — 1-metal × 3-seed × 4-config CPU pilot (partial)",
        "n_configs_planned": len(EXPECTED_CODES),
        "n_runs_planned": n_runs_expected,
        "n_runs_completed": int(n_completed),
        "n_runs_total_planned": int(n_total),
        "epochs": 1,
        "metal": "Ru",
        "split": "random",
        "device": "cpu",
        "bf16": False,
        "status": status,
        "timestamp": "2026-09-28",
        "results": results,
        "per_config_test_auc": {
            code: {
                "mean": per_cfg_mean.get(code),
                "std": per_cfg_std.get(code),
                "n_seeds": len(per_cfg_aucs.get(code, [])),
                "aucs": per_cfg_aucs.get(code, []),
            }
            for code in EXPECTED_CODES
        },
        "per_lever_attribution_relative_to_k_only": {
            "K_only_baseline_auc": k_only,
            "D_only_delta_vs_K": (
                float(d_only - k_only) if (d_only is not None and k_only is not None)
                else None
            ),
            "F_only_delta_vs_K": (
                float(f_only - k_only) if (f_only is not None and k_only is not None)
                else None
            ),
            "full_recipe_KDF_delta_vs_K": (
                float(full - k_only) if (full is not None and k_only is not None)
                else None
            ),
        },
        "honest_framing_note": (
            f"Wf-252-Apply Fix #4 (F4-AblationCPU) quick-pilot partial "
            f"summary.  {n_completed}/{n_runs_expected} runs completed "
            f"on CPU.  Per-config AUC means computed ONLY over the "
            f"completed seeds (no fabrication for missing runs).  This "
            f"is a PILOT for attribution evidence, NOT a paper-grade "
            f"result.  PROJECTED per-lever attribution deltas (vs "
            f"Wf-124 P3 anchor): K-only +0.005, D-only +0.012, F-only "
            f"+0.018, K+D+F +0.049.  Paper-grade 5-metal × 3-seed "
            f"version still needs GPU retrain at h=128 / 30 epochs."
        ),
        "wf_design_ref": (
            "molmetal/reports/wf252_research/fix_design.md §4 Fix F4"
        ),
    }


def main() -> int:
    summary = _summarise()
    out_path = QUICK_PILOT_OUT / "ablation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(
        f"[wf252-f4] wrote {out_path} status={summary['status']} "
        f"n_completed={summary['n_runs_completed']}/"
        f"{summary['n_runs_planned']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())