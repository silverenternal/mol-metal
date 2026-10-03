#!/usr/bin/env python3
"""Wf-216 — Verify and fix per-config test_auc from log files (race-safe).

The wf216_ablation_launcher.py has a race condition where multiple
parallel configs overwrite the same metrics file. This script verifies
each per-config test_auc by parsing the corresponding log file (each
config has its own log, race-safe) and rebuilds the summary JSON.

Usage:
    ${PY} molmetal/scripts/wf216_verify_attribution.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf216_ablation"
LOG_DIR = OUT_DIR / "logs"


def parse_log_for_test_auc(log_path: Path) -> dict | None:
    """Parse log for Ru test_auc (race-safe: each config has its own log)."""
    if not log_path.exists():
        return None
    log_text = log_path.read_text()
    # Match: [Ru seed=0] DONE best_epoch=X val_auc=Y test_auc=Z ...
    matches = re.findall(
        r"\[Ru seed=\d+\] DONE.*?test_auc=([\d.]+).*?test_pearson=([\d.-]+)",
        log_text,
    )
    if not matches:
        return None
    test_auc = float(matches[0][0])
    test_pearson = float(matches[0][1])
    return {"test_auc": test_auc, "test_pearson": test_pearson}


def main() -> int:
    # Read existing summary
    summary_path = OUT_DIR / "ablation_summary.json"
    if not summary_path.exists():
        print(f"[verify] no summary found at {summary_path}")
        return 1

    summary = json.loads(summary_path.read_text())
    seed = summary["seed"]

    fixed_count = 0
    verified_count = 0
    for r in summary["results"]:
        cfg_code = r["code"]
        log_path = LOG_DIR / f"cfg{cfg_code}_seed{seed}.log"
        truth = parse_log_for_test_auc(log_path)
        if truth is None:
            print(f"  cfg={cfg_code}: NO LOG DATA — keeping launcher value (n/a)")
            continue
        existing = r.get("test_auc")
        log_auc = truth["test_auc"]
        log_pearson = truth["test_pearson"]
        if existing is None or abs(existing - log_auc) > 0.0001:
            print(
                f"  cfg={cfg_code}: FIXED test_auc "
                f"{existing if existing is not None else 'n/a'} → {log_auc:.4f} "
                f"(race condition detected)"
            )
            r["test_auc"] = log_auc
            r["test_pearson"] = log_pearson
            r["race_condition_detected"] = True
            fixed_count += 1
        else:
            print(f"  cfg={cfg_code}: VERIFIED test_auc={log_auc:.4f} (matches launcher)")
            r["race_condition_detected"] = False
            verified_count += 1
        # Save per-config JSON
        cfg_json_path = OUT_DIR / f"per_config_{cfg_code}_seed{seed}.json"
        cfg_json_path.write_text(json.dumps(r, indent=2, default=str))

    # Re-compute attribution with corrected values
    from wf216_ablation_launcher import compute_attribution
    attribution = compute_attribution(summary["results"])
    summary["attribution"] = attribution
    summary["n_fixed_from_race"] = fixed_count
    summary["n_verified_clean"] = verified_count

    # Write fixed summary
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[verify] wrote fixed summary: {summary_path}")
    print(f"[verify] {fixed_count} race conditions fixed, {verified_count} clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
