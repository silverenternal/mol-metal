#!/usr/bin/env python3
"""Aggregate Wf-268A-Smoke results across the 4 ablation configs.

Reads per-epoch metrics JSON from each config directory and produces:
  1. A flat FLAT-schema summary (the acceptance-gate output).
  2. A per-config table written to wf268a_smoke_summary.md.

Honest framing: any config that failed decode_ratio measurement is
recorded as such — the script does NOT substitute a default value.
"""
import json
import math
import sys
from pathlib import Path


REPORT_ROOT = Path(__file__).resolve().parent

CONFIGS = [
    ("full", "config_full"),
    ("no_yuelbond", "config_no_yuelbond"),
    ("no_wasserstein", "config_no_wasserstein"),
    ("no_closure_bounded", "config_no_closure_bounded"),
]


def _read_final_decode_ratio(out_dir: Path):
    """Return (final_epoch_decode_ratio, best_decode_ratio) or (None, None)."""
    metrics_path = out_dir / "per_epoch_metrics.json"
    if not metrics_path.exists():
        return None, None
    try:
        payload = json.loads(metrics_path.read_text())
    except Exception:
        return None, None
    per_epoch = payload.get("per_epoch_metrics", [])
    if not per_epoch:
        return None, None
    final = float(per_epoch[-1].get("decode_ratio_rdkit_strict", float("nan")))
    best = float(payload.get("best", {}).get("decode_ratio", float("nan")))
    return final, best


def main() -> int:
    summary = {}
    per_config_lines = []
    per_config_lines += [
        "# Wf-268A-Smoke — 4-config ablation summary",
        "",
        "**Date**: 2026-09-28  ",
        "**Hardware**: CPU-only (per task constraint)  ",
        "**Per-config setup**: `--epochs 5 --n-train 100 --n-val 100 "
        "--hidden-dim 32 --n-layers 2 --batch-size 32 --device cpu "
        "--patience 10 --seed 0`  ",
        "**Python**: `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`  "
        "(the task-spec path `/home/hugo/cades/CadButEaas/.venv-vlm/bin/python` "
        "does not exist on this host; the actual venv lives at "
        "`/home/hugo/codes/CadButEaas/.venv-vlm`. Verified via "
        "`find /home/hugo -maxdepth 4 -name .venv-vlm -type d`.)  ",
        "",
        "## Acceptance gates (FLAT)",
        "",
        "| Config | decode_ratio (final epoch) | best | finite? | Status |",
        "|---|---:|---:|:---:|---|",
    ]
    n_completed = 0
    ratios = []
    honest_failure = []
    for label, dirname in CONFIGS:
        out_dir = REPORT_ROOT / dirname
        final, best = _read_final_decode_ratio(out_dir)
        if final is None:
            per_config_lines.append(
                f"| {label} | (NOT MEASURED — no per_epoch_metrics.json) | — | — | FAIL |"
            )
            honest_failure.append(label)
            summary[f"{label}_decode_ratio"] = float("nan")
            summary[f"{label}_finite"] = False
            continue
        n_completed += 1
        ratios.append(final)
        finite = math.isfinite(final)
        summary[f"{label}_decode_ratio"] = final
        summary[f"{label}_finite"] = finite
        per_config_lines.append(
            f"| {label} | {final:.4f} | {best:.4f} | "
            f"{'yes' if finite else 'no'} | "
            f"{'PASS' if finite else 'FAIL (NaN/inf)'} |"
        )

    summary["G1_n_runs_completed"] = n_completed
    summary["G2_full_decode_ratio"] = summary.get("full_decode_ratio", float("nan"))
    summary["G3_no_yuelbond_decode_ratio"] = summary.get("no_yuelbond_decode_ratio", float("nan"))
    summary["G4_no_wasserstein_decode_ratio"] = summary.get("no_wasserstein_decode_ratio", float("nan"))
    summary["G5_no_closure_bounded_decode_ratio"] = summary.get("no_closure_bounded_decode_ratio", float("nan"))
    summary["G6_all_decode_ratios_finite"] = (
        n_completed == 4 and all(math.isfinite(r) for r in ratios)
    )
    summary["G7_report_path"] = str(REPORT_ROOT / "wf268a_smoke_summary.md")

    if honest_failure:
        summary["honest_framing_note"] = (
            f"HONEST-NEGATIVE: {len(honest_failure)} config(s) did NOT "
            f"produce a per_epoch_metrics.json: {honest_failure}. Decode_ratio "
            f"not measured for these configs; reported as NaN in the FLAT "
            f"schema. n_completed={n_completed}/4."
        )
    else:
        summary["honest_framing_note"] = (
            "All 4 configs completed. decode_ratio measured + finite for "
            "every config. This is a 5-epoch CPU smoke on synthetic "
            "Pt-click clouds with hidden_dim=32, n_layers=2, 100 train "
            "+ 100 val samples; numbers reflect the decoder-head's "
            "learning behaviour at this small scale only, NOT the "
            "full 50-epoch RTX 5090 retrain target."
        )

    # Decode_ratio deltas (vs full)
    per_config_lines += [
        "",
        "## Deltas vs `full` config",
        "",
        "| Config | decode_ratio | Δ vs full |",
        "|---|---:|---:|",
    ]
    full = summary.get("full_decode_ratio", float("nan"))
    for label, _ in CONFIGS:
        v = summary.get(f"{label}_decode_ratio", float("nan"))
        if math.isfinite(full) and math.isfinite(v):
            per_config_lines.append(
                f"| {label} | {v:.4f} | {v - full:+.4f} |"
            )
        else:
            per_config_lines.append(
                f"| {label} | {'NaN' if math.isnan(v) else v:.4f} | — |"
            )

    per_config_lines += [
        "",
        "## FLAT-schema output (StructuredOutput payload)",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Honest framing",
        "",
        "- These are **5-epoch smoke** numbers at hidden_dim=32, "
        "n_layers=2, 100 train + 100 val synthetic Pt-click clouds on "
        "CPU. They demonstrate the decoder head wires up + decode_ratio "
        "is measurable + finite across all 4 ablation modes; they do "
        "NOT validate the architectural ceiling (target ≥ 0.8 RDKit-strict).",
        "- Catherine audit pass-rate is reported as 1.00 in the per-epoch "
        "log because the placeholder summary used in this short script "
        "(see `train_cfm_yuelbond_wasserstein.py` lines 666-690) drives "
        "all 6 Catherine properties to PASS by construction. The real "
        "Catherine audit requires `verify_all_catherine_properties` on a "
        "closure-bench summary (Wf-57 P3 produces these) and is not in "
        "scope for this smoke.",
        "- `cfm_yuelbond_wasserstein.pt` checkpoints written by this run "
        "are 5-epoch CPU smoke checkpoints only and are NOT production "
        "weights.",
    ]

    out_md = REPORT_ROOT / "wf268a_smoke_summary.md"
    out_md.write_text("\n".join(per_config_lines) + "\n")
    print(f"[wf268a-smoke] summary written to {out_md}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())