#!/usr/bin/env python3
"""Wf-268B-Aggregate — aggregate per-seed ood_test AUC from Wf-268B-Run output.

Honest framing per party-twenty-spirit §3.5:
- READ-ONLY on per_seed/seed*.json. We never substitute a default value
  for missing seeds; missing seeds are reported as NaN.
- Std requires n>=2 seeds with a finite value; n=1 → std = NaN.
- Config with 0 seeds measured → mean = NaN; per-config pass flag = False.

v2 baseline (Wf-185 chemprop v2 retrain, 5-seed × 20-ep × h=128/d=2,
ood_test mean across 5 seeds):
    lbap_refined_ic50_assay:    0.7137 ± 0.0079
    lbap_refined_ic50_scaffold: 0.6707 ± 0.0042
    lbap_refined_ic50_size:     0.6757 ± 0.0142

Acceptance gate per config: ood_test_mean >= v2_baseline_mean.

NOTE: Wf-268B-Run was launched with hidden=300 depth=3 head_hidden=300
batch=64 epochs=20 seed_time_budget_s=600 — the chemprop v2 default backbone,
NOT the h=128/d=2 used by Wf-185. This means Wf-268B is a different (larger)
model than Wf-185; the v2 baseline is the comparison anchor specified by
the Wf-268B-Aggregate task brief, not a like-for-like match.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUN_ROOT = PROJECT_ROOT / "molmetal" / "reports" / "wf268b_run"
OUT_ROOT = PROJECT_ROOT / "molmetal" / "reports" / "wf268b_verify"

CONFIGS: List[str] = [
    "lbap_refined_ic50_assay",
    "lbap_refined_ic50_scaffold",
    "lbap_refined_ic50_size",
]
SPLITS: List[str] = ["train", "iid_val", "iid_test", "ood_val", "ood_test"]
SEEDS: List[int] = [0, 1, 2, 3, 4]

# Wf-185 v2 baseline (5-seed × 20-ep × h=128/d=2 chemprop v2 BCE retrain).
V2_BASELINE: Dict[str, float] = {
    "lbap_refined_ic50_assay":    0.7137,
    "lbap_refined_ic50_scaffold": 0.6707,
    "lbap_refined_ic50_size":     0.6757,
}
V2_BASELINE_STD: Dict[str, float] = {
    "lbap_refined_ic50_assay":    0.0079,
    "lbap_refined_ic50_scaffold": 0.0042,
    "lbap_refined_ic50_size":     0.0142,
}


def _finite(xs: List[float]) -> List[float]:
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _mean(xs: List[float]):
    """Return float mean, or None if no finite values."""
    f = _finite(xs)
    return float(sum(f) / len(f)) if f else None


def _std(xs: List[float]):
    """Sample std (ddof=1) — None if n < 2."""
    if len(xs) < 2:
        return None
    f = _finite(xs)
    if len(f) < 2:
        return None
    m = sum(f) / len(f)
    var = sum((x - m) ** 2 for x in f) / (len(f) - 1)
    return math.sqrt(var)


def load_per_seed(cfg: str) -> Dict[int, dict]:
    """Return seed→payload for cfg, only including seeds with a JSON on disk."""
    out: Dict[int, dict] = {}
    cfg_dir = RUN_ROOT / cfg / "per_seed"
    if not cfg_dir.exists():
        return out
    for seed in SEEDS:
        p = cfg_dir / f"seed{seed}.json"
        if not p.exists():
            continue
        try:
            out[seed] = json.loads(p.read_text())
        except Exception:
            continue
    return out


def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt(x) -> str:
    return "NaN" if _is_nan(x) else f"{x:.4f}"


def _fmt_delta(x) -> str:
    return "NaN" if _is_nan(x) else f"{x:+.4f}"


def main() -> int:
    per_cfg: Dict[str, dict] = {}
    for cfg in CONFIGS:
        per_seed = load_per_seed(cfg)
        per_split_values: Dict[str, List[float]] = {sp: [] for sp in SPLITS}
        for seed in SEEDS:
            payload = per_seed.get(seed)
            if payload is None:
                continue
            for sp in SPLITS:
                key = f"{sp}_auc"
                v = payload.get(key)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    per_split_values[sp].append(float(v))
        per_split_mean = {sp: _mean(per_split_values[sp]) for sp in SPLITS}
        per_split_std = {sp: _std(per_split_values[sp]) for sp in SPLITS}
        per_cfg[cfg] = {
            "n_seeds_measured": len(per_seed),
            "n_seeds_expected": len(SEEDS),
            "missing_seeds": [s for s in SEEDS if s not in per_seed],
            "per_split_mean": per_split_mean,
            "per_split_std": per_split_std,
            "ood_test_per_seed": [v for v in per_split_values["ood_test"]],
            "ood_test_mean": per_split_mean["ood_test"],
            "ood_test_std": per_split_std["ood_test"],
            "v2_baseline_mean": V2_BASELINE[cfg],
            "v2_baseline_std": V2_BASELINE_STD[cfg],
            "delta_vs_v2": (
                per_split_mean["ood_test"] - V2_BASELINE[cfg]
                if not _is_nan(per_split_mean["ood_test"])
                else None
            ),
            "passes_v2_baseline": (
                not _is_nan(per_split_mean["ood_test"])
                and per_split_mean["ood_test"] >= V2_BASELINE[cfg]
            ),
        }

    # FLAT-schema acceptance gates.
    g1_assay_mean = per_cfg["lbap_refined_ic50_assay"]["ood_test_mean"]
    g2_scaffold_mean = per_cfg["lbap_refined_ic50_scaffold"]["ood_test_mean"]
    g3_size_mean = per_cfg["lbap_refined_ic50_size"]["ood_test_mean"]
    g4_assay_above = bool(per_cfg["lbap_refined_ic50_assay"]["passes_v2_baseline"])
    g5_scaffold_above = bool(per_cfg["lbap_refined_ic50_scaffold"]["passes_v2_baseline"])
    g6_size_above = bool(per_cfg["lbap_refined_ic50_size"]["passes_v2_baseline"])
    g7_report_path = str(OUT_ROOT / "verify_report.md")

    # Honest framing: only 1 seed completed (assay seed=0); 14 missing.
    n_total_measured = sum(c["n_seeds_measured"] for c in per_cfg.values())
    n_total_expected = sum(c["n_seeds_expected"] for c in per_cfg.values())
    if n_total_measured == 0:
        honest_note = (
            "HONEST-NEGATIVE: Wf-268B-Run produced ZERO per-seed JSON outputs. "
            "All three configs (assay, scaffold, size) report NaN mean. "
            "No v2-baseline comparison is possible; gates G4/G5/G6 = False."
        )
    elif n_total_measured < n_total_expected:
        honest_note = (
            f"HONEST-NEGATIVE: Wf-268B-Run produced only "
            f"{n_total_measured}/{n_total_expected} per-seed JSON outputs. "
            "Std is NaN where n<2; per-config mean over a partial seed set is "
            "NOT comparable to the Wf-185 5-seed v2 baseline. The single "
            "completed seed (lbap_refined_ic50_assay seed=0) reported "
            "ood_test=0.7016, which is below the Wf-185 v2 baseline of 0.7137, "
            "but this is a 1-seed point estimate with seed_time_budget=600s hit "
            "at epoch=8 — NOT a 5-seed ablation. Re-run Wf-268B-Run with a "
            "larger wall budget to complete the remaining seeds before drawing "
            "any v2-vs-v3-fallback conclusion."
        )
    else:
        honest_note = (
            "All 15 seeds measured; mean+std computed across 5 seeds per config."
        )

    summary = {
        "date": "2026-09-29",
        "wf": "wf268b-aggregate",
        "task": "Wf-268B-Aggregate: aggregate per-split AUC + verify against v2 baseline",
        "run_root": str(RUN_ROOT),
        "out_root": str(OUT_ROOT),
        "v2_baseline_source": "Wf-185 chemprop v2 retrain (5-seed × 20-ep × h=128/d=2)",
        "v2_baseline": {
            "lbap_refined_ic50_assay":    {"mean": V2_BASELINE["lbap_refined_ic50_assay"],
                                            "std": V2_BASELINE_STD["lbap_refined_ic50_assay"]},
            "lbap_refined_ic50_scaffold": {"mean": V2_BASELINE["lbap_refined_ic50_scaffold"],
                                            "std": V2_BASELINE_STD["lbap_refined_ic50_scaffold"]},
            "lbap_refined_ic50_size":     {"mean": V2_BASELINE["lbap_refined_ic50_size"],
                                            "std": V2_BASELINE_STD["lbap_refined_ic50_size"]},
        },
        "per_config": per_cfg,
        "n_seeds_total_measured": n_total_measured,
        "n_seeds_total_expected": n_total_expected,
        "acceptance_gates": {
            "G1_assay_mean": g1_assay_mean,
            "G2_scaffold_mean": g2_scaffold_mean,
            "G3_size_mean": g3_size_mean,
            "G4_assay_above_v2": g4_assay_above,
            "G5_scaffold_above_v2": g5_scaffold_above,
            "G6_size_above_v2": g6_size_above,
            "G7_report_path": g7_report_path,
        },
        "honest_framing_note": honest_note,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "per_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Write verify_report.md
    lines: List[str] = [
        "# Wf-268B-Aggregate — per-split AUC aggregation + v2-baseline verify",
        "",
        "**Date:** 2026-09-29  ",
        "**Task:** Aggregate per-seed ood_test AUC from Wf-268B-Run output and "
        "verify against Wf-185 chemprop v2 baseline.  ",
        "**Run output root:** `molmetal/reports/wf268b_run/`  ",
        "**Aggregate root:** `molmetal/reports/wf268b_verify/`  ",
        "",
        "## Wf-268B-Run recipe",
        "",
        "Script: `molmetal/scripts/wf263a_verify_drugood.py`  ",
        "Args used (per `wf268b_run.log` header):  ",
        "- `--configs lbap_refined_ic50_assay lbap_refined_ic50_scaffold lbap_refined_ic50_size`  ",
        "- `--seeds 0 1 2 3 4`  ",
        "- `--epochs 20 --hidden 300 --depth 3 --head-hidden 300 --batch 64 --lr 0.001`  ",
        "- `--device cpu --time-budget-s 10800 --seed-time-budget-s 600 --eval-max-samples 1000`  ",
        "- `--chemprop-mode v3-fallback` (forces chemprop v2 BinaryClassificationFFN head)",
        "",
        "Note: backbone is chemprop v2 DEFAULTS (h=300/d=3), NOT the h=128/d=2 "
        "recipe used in Wf-185. Wf-268B is therefore a different (larger) model "
        "compared against the Wf-185 v2 anchor.",
        "",
        "## Per-config measured seed counts",
        "",
        "| Config | n_seeds_measured / n_expected | missing_seeds |",
        "|---|---:|---|",
    ]
    for cfg in CONFIGS:
        c = per_cfg[cfg]
        miss = ",".join(str(s) for s in c["missing_seeds"]) or "(none)"
        lines.append(
            f"| {cfg} | {c['n_seeds_measured']}/{c['n_seeds_expected']} | {miss} |"
        )
    lines += [
        "",
        f"**Total:** {n_total_measured}/{n_total_expected} per-seed JSONs on disk.",
        "",
        "## Per-config per-split AUC (mean ± std across measured seeds)",
        "",
        "| Config | split | mean | std | n |",
        "|---|---|---:|---:|---:|",
    ]
    for cfg in CONFIGS:
        c = per_cfg[cfg]
        for sp in SPLITS:
            m = c["per_split_mean"][sp]
            s = c["per_split_std"][sp]
            n = len(_finite([
                per_seed_payload.get(f"{sp}_auc")
                for per_seed_payload in load_per_seed(cfg).values()
            ]))
            lines.append(f"| {cfg} | {sp} | {_fmt(m)} | {_fmt(s)} | {n} |")
    lines += [
        "",
        "## v2 baseline (Wf-185) + delta vs Wf-268B",
        "",
        "| Config | v2_mean | v2_baseline (Wf-185) | v3-fallback (Wf-268B) | delta | passes? |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for cfg in CONFIGS:
        c = per_cfg[cfg]
        v2m = c["v2_baseline_mean"]
        v3m = c["ood_test_mean"]
        d = c["delta_vs_v2"]
        v3m_str = _fmt(v3m) if not _is_nan(v3m) else "NaN (insufficient data)"
        ok = "PASS" if c["passes_v2_baseline"] else "FAIL"
        lines.append(
            f"| {cfg} | {v2m:.4f} | {v2m:.4f} | {v3m_str} | {_fmt_delta(d)} | {ok} |"
        )

    lines += [
        "",
        "## Per-seed ood_test AUC (raw)",
        "",
        "| Config | seed | ood_test | epochs_done | status | elapsed_s |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for cfg in CONFIGS:
        per_seed = load_per_seed(cfg)
        for seed in SEEDS:
            payload = per_seed.get(seed)
            if payload is None:
                lines.append(f"| {cfg} | {seed} | (NOT MEASURED) | — | — | — |")
                continue
            lines.append(
                f"| {cfg} | {seed} | {payload['ood_test_auc']:.4f} | "
                f"{payload['epochs_done']} | {payload['status']} | "
                f"{payload['elapsed_s']:.1f} |"
            )

    lines += [
        "",
        "## Acceptance gates (FLAT schema)",
        "",
        "| Gate | Value |",
        "|---|---:|",
        f"| G1_assay_mean | {_fmt(g1_assay_mean)} |",
        f"| G2_scaffold_mean | {_fmt(g2_scaffold_mean)} |",
        f"| G3_size_mean | {_fmt(g3_size_mean)} |",
        f"| G4_assay_above_v2 | {g4_assay_above} |",
        f"| G5_scaffold_above_v2 | {g5_scaffold_above} |",
        f"| G6_size_above_v2 | {g6_size_above} |",
        f"| G7_report_path | {g7_report_path} |",
        "",
        "## Honest framing",
        "",
        honest_note,
        "",
        "### What Wf-268B-Run measured",
        "",
        "- `lbap_refined_ic50_assay` seed=0: ood_test=0.7016 "
        "(8/20 epochs, status=timeout, elapsed=815s). Seed-time-budget=600s "
        "hit at epoch=8; same timeout pattern as Wf-185 (which also aborted "
        "seed=0/1 of the assay config at 6-8 epochs).",
        "- `lbap_refined_ic50_assay` seed=1..4: log shows epoch=1 started "
        "(loss=0.2501, elapsed=66.6s) but no further epoch/FINAL lines — the "
        "run was interrupted before seed=1 completed.",
        "- `lbap_refined_ic50_scaffold` and `lbap_refined_ic50_size`: "
        "0/5 seeds each — neither config was reached before the run was "
        "interrupted.",
        "",
        "### Why Wf-268B-Aggregate cannot pass",
        "",
        "- Task spec requires mean ± std across 5 seeds per config. "
        "With 1 seed measured (assay seed=0) and 0 measured for the other two "
        "configs, we cannot compute a 5-seed mean ± std for ANY config.",
        "- The single 1-seed point estimate for assay is "
        "0.7016 vs Wf-185 v2 baseline 0.7137 → below v2 by 0.0121, BUT this is "
        "a 1-seed point estimate that timed out at 8/20 epochs on a *larger* "
        "backbone (h=300/d=3 vs Wf-185 h=128/d=2). Drawing a v2-vs-v3-fallback "
        "conclusion from one timed-out seed is unsound.",
        "- G4/G5/G6 are therefore False — not because v3-fallback is "
        "demonstrably worse than v2, but because there is not enough measured "
        "data to make the comparison. Re-run Wf-268B-Run with a larger wall "
        "budget (>= 6 h) to complete all 15 seeds before re-running "
        "Wf-268B-Aggregate.",
        "",
        "### Files",
        "",
        "* `per_split_summary.json` — full per-config + FLAT-schema gate payload",
        "* `verify_report.md` — this file",
        "* `aggregate.py` — READ-ONLY aggregator (no result.json writes)",
    ]
    (OUT_ROOT / "verify_report.md").write_text("\n".join(lines) + "\n")

    print(f"[wf268b-aggregate] wrote {summary_path}")
    print(f"[wf268b-aggregate] wrote {OUT_ROOT / 'verify_report.md'}")
    print(json.dumps(summary["acceptance_gates"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())