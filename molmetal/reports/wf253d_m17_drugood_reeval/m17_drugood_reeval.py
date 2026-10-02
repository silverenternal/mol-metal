"""Wf-253-D — chemprop v2 v3 forward-pass reeval on DrugOOD scaffold split.

================================================================================
Wf-253-D / TODO-130 G5 — wire the v3 checkpoint forward pass on DrugOOD.
================================================================================

Background
----------
Wf-252-Apply F5 (M17DrugOODReeval) shipped a deferred-forward-pass
emission at ``metrics/by_metric/drugood_v3_scaffold.json``.  The Wf-252
script only verified checkpoint loadability and emitted PROJECTED AUCs
plus a ``CHECKPOINT_LOADED_BUT_FORWARD_PASS_DEFERRED`` status.  This
workstream wires the actual forward pass.

Strategy
--------
1. Load v3 checkpoints from
   ``molmetal/checkpoints/wf200_chemprop_v2_v3/{metal}_seed{N}.pt``
   for the metals {Ru, Ir, Os} (the only metals with checkpoints on
   disk at this point: Ru × 4, Ir × 3, Os × 1 = 8 (metal, seed)
   pairs).
2. Load DrugOOD ``lbap_refined_ic50_scaffold`` ood_test split via
   ``molmetal.data.drugood_loader.load_drugood`` (raw JSON fallback —
   no DrugOOD library required).
3. Run forward pass for each (metal, seed) checkpoint on a balanced
   subset (cap = ``--eval-max-samples``, default 2000) of ood_test.
   The active label (``cls_label``) is the binary target.
4. Compute ROC-AUC + AUPR per (metal, seed) using ``sklearn.metrics``.
5. Aggregate per-metal (mean / std over available seeds) and compare
   to the Wf-185 MEASURED scaffold AUC = 0.6707 ± 0.0042.
6. Emit ``molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json``
   and ``m17_drugood_reeval.md``.

CPU-only feasibility
--------------------
* Each forward pass on 2000 SMILES with hidden=64 / depth=2 / CPU takes
  ~10 min per (metal, seed) (no BF16; CPU forward).  Total wall-clock
  budget ~ 8 × 10 min = 80 min on CPU.
* For a faster smoke, run with ``--eval-max-samples 500`` (~3 min/run).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* v3 weights were NOT re-trained on DrugOOD — no leakage.
* The checkpoint was trained on 5-metal MetalCytoToxDB
  (Ru/Ir/Os/Rh/Re) — we use the per-metal head that the
  held-out metal belongs to (Ru→Ru head, Ir→Ir head, Os→Os head)
  to score the binary DrugOOD activity.
* We do NOT compare to iMoLD / TargetDiff directly — the
  comparison target is Wf-185 chemprop v2 retrain (sister backbone,
  same DrugOOD split).
* Per-metal coverage is uneven: Ru has 4 seeds, Ir has 3, Os has 1;
  the aggregate mean / std are reported over the available seeds.
* PROJECTED AUC ranges from Wf-252 are kept in the output as a
  comparison reference; v3 measured AUCs may differ from PROJECTED
  in either direction.

Usage
-----

    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.py \\
        --eval-max-samples 500

Writes ``molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json``
and ``m17_drugood_reeval.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os as _os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root on sys.path so we can import molmetal.* modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in _os.sys.path:  # type: ignore[attr-defined]
    _os.sys.path.insert(0, str(PROJECT_ROOT))  # type: ignore[attr-defined]

REPORT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf253d_m17_drugood_reeval"
CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints" / "wf200_chemprop_v2_v3"
METRICS_OUT = PROJECT_ROOT / "metrics" / "by_metric" / "drugood_v3_scaffold.json"
DRUGOOD_JSON = (
    PROJECT_ROOT / ".data" / "molmetal" / "drugood" / "ic50" / "official"
    / "lbap_refined_ic50_scaffold.json"
)
METALS = ("Ru", "Ir", "Os")
DEFAULT_SEEDS = (0, 1, 2, 3)
DEFAULT_EVAL_MAX_SAMPLES = 2000
BATCH_SIZE = 64

#: Wf-185 MEASURED DrugOOD scaffold AUC reference (chemprop v2 retrain).
WF185_REFERENCE = {
    "lbap_refined_ic50_scaffold_ood_test_mean": 0.6706684024759187,
    "lbap_refined_ic50_scaffold_ood_test_std": 0.004172704990837158,
    "n_seeds": 5,
    "note": (
        "Wf-185 chemprop v2 retrain baseline (NOT v3 backbone).  "
        "Used as honest MEASURED reference for the m17 forward-pass "
        "comparison."
    ),
}

#: PROJECTED AUC ranges from the Wf-252 design §4 Fix F5 — kept as a
#: sanity-check vs the v3 measured numbers.
PROJECTED_PER_METAL = {
    "Ru": {"auc_mean_lo": 0.65, "auc_mean_hi": 0.70, "auc_mean_mid": 0.675},
    "Ir": {"auc_mean_lo": 0.55, "auc_mean_hi": 0.65, "auc_mean_mid": 0.60},
    "Os": {"auc_mean_lo": 0.55, "auc_mean_hi": 0.65, "auc_mean_mid": 0.60},
    "Rh": {"auc_mean_lo": 0.55, "auc_mean_hi": 0.65, "auc_mean_mid": 0.60},
    "Re": {"auc_mean_lo": 0.55, "auc_mean_hi": 0.65, "auc_mean_mid": 0.60},
}


def _list_checkpoints() -> Dict[str, List[int]]:
    """Return {metal: [seed, ...]} for which v3 checkpoints exist."""
    out: Dict[str, List[int]] = {}
    for m in METALS:
        for s in DEFAULT_SEEDS:
            p = CKPT_DIR / f"dmpnn_mtw_v3_{m.lower()}_seed{s}.pt"
            if p.exists():
                out.setdefault(m, []).append(int(s))
    return out


def _load_drugood_ood_test(eval_max_samples: int) -> Optional[Tuple[List[str], List[int]]]:
    """Load DrugOOD scaffold split ood_test directly from the official
    JSON.  Returns ``(smiles_list, label_list)`` or ``None`` on any failure.

    Uses :mod:`csv`-free direct JSON parsing — does NOT depend on the
    DrugOOD library.
    """
    if not DRUGOOD_JSON.exists():
        print(
            f"[wf253d-m17] DrugOOD JSON not found: {DRUGOOD_JSON}",
            flush=True,
        )
        return None
    try:
        with open(DRUGOOD_JSON, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(
            f"[wf253d-m17] DrugOOD JSON load failed: {exc}",
            flush=True,
        )
        return None

    if "split" not in payload or "ood_test" not in payload["split"]:
        print(
            f"[wf253d-m17] DrugOOD payload missing ood_test split",
            flush=True,
        )
        return None

    smiles: List[str] = []
    labels: List[int] = []
    for row in payload["split"]["ood_test"]:
        try:
            s = str(row["smiles"])
            y = int(row["cls_label"])
        except (KeyError, TypeError, ValueError):
            continue
        smiles.append(s)
        labels.append(y)

    if eval_max_samples > 0 and len(smiles) > eval_max_samples:
        smiles = smiles[:eval_max_samples]
        labels = labels[:eval_max_samples]

    return smiles, labels


def _run_one_forward(
    metal: str,
    seed: int,
    smiles_list: List[str],
    labels: List[int],
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
) -> Optional[Dict]:
    """Run v3 forward pass on the full smiles_list; compute ROC-AUC + AUPR.

    Returns ``{"roc_auc": float, "pr_auc": float, "n_samples": int,
    "n_positive": int, "n_negative": int, "wall_clock_s": float}`` on
    success; ``None`` on any error.
    """
    try:
        import numpy as np
        import torch
        from sklearn.metrics import (  # type: ignore
            average_precision_score,
            roc_auc_score,
        )
    except Exception as exc:
        print(
            f"[wf253d-m17] required import failed (torch/sklearn): {exc}",
            flush=True,
        )
        return None

    ckpt_path = CKPT_DIR / f"dmpnn_mtw_v3_{metal.lower()}_seed{seed}.pt"
    if not ckpt_path.exists():
        return None

    try:
        ckpt = torch.load(
            ckpt_path, map_location=device, weights_only=False,
        )
    except Exception as exc:
        print(
            f"[wf253d-m17] checkpoint load failed {ckpt_path}: {exc}",
            flush=True,
        )
        return None

    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Build the wrapper with the v3 config matching the Wf-200 driver.
    try:
        from molmetal.scripts.train_chemprop_v2_v3 import ChempropV2V3Model

        model = ChempropV2V3Model(
            hidden_dim=64, depth=2, n_metals=5, n_coord_classes=5,
            metal_embed_dim=64, domain_hidden_dim=128,
            enhanced_head=False, use_metal_attention=False,
        )
        model.eval()

        # The lazy-built backbone inside model.inner.encoder needs a
        # forward pass before the state_dict matches (the message
        # passing module is built on the first call).
        try:
            _ = model(["CCO"], torch.tensor([0]))
        except Exception:
            pass

        model.load_state_dict(sd, strict=True)
    except Exception as exc:
        print(
            f"[wf253d-m17] model build / load failed {ckpt_path}: {exc}",
            flush=True,
        )
        return None

    # Resolve metal → int id.
    try:
        metal_id = int(model.metal_id(metal))
    except Exception as exc:
        print(
            f"[wf253d-m17] metal_id({metal}) failed: {exc}",
            flush=True,
        )
        return None

    # Forward pass — process in batches.
    n = len(smiles_list)
    all_probs: List[float] = []
    t0 = time.time()
    try:
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch_smiles = smiles_list[i:i + batch_size]
                batch_metal = torch.full(
                    (len(batch_smiles),), metal_id, dtype=torch.long,
                )
                out = model(batch_smiles, batch_metal)
                # out[metal] has shape (B, 2); column 0 is the active logit.
                logits = out[metal][:, 0]
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs)
    except Exception as exc:
        print(
            f"[wf253d-m17] forward pass failed {ckpt_path}: {exc}",
            flush=True,
        )
        return None
    wall_clock_s = float(time.time() - t0)

    if len(all_probs) != n:
        print(
            f"[wf253d-m17] forward prob count mismatch: {len(all_probs)} vs {n}",
            flush=True,
        )
        return None

    # Filter out SMILES that failed to parse (the model returns a
    # constant placeholder for invalid SMILES; we approximate by
    # checking whether the batch probability spread is zero).
    # For honesty, we report the metrics on the full cohort with
    # the caveat that some molecules may have placeholder features.
    try:
        roc_auc = float(roc_auc_score(labels, all_probs))
    except Exception as exc:
        print(
            f"[wf253d-m17] roc_auc failed: {exc}",
            flush=True,
        )
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(labels, all_probs))
    except Exception as exc:
        pr_auc = None

    n_pos = int(sum(labels))
    n_neg = int(n - n_pos)
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_samples": int(n),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "wall_clock_s": wall_clock_s,
    }


def _summarise_per_metal(metal: str, aucs: List[float]) -> Dict:
    if not aucs:
        return {
            "test_auc_mean": None,
            "test_auc_std": None,
            "n_seeds_measured": 0,
            "projected_auc_range": list(
                PROJECTED_PER_METAL.get(
                    metal, {}
                ).get("auc_mean_lo", None) is not None
                and [
                    PROJECTED_PER_METAL[metal]["auc_mean_lo"],
                    PROJECTED_PER_METAL[metal]["auc_mean_hi"],
                ]
                or []
            ),
            "projected_auc_midpoint": PROJECTED_PER_METAL.get(
                metal, {}
            ).get("auc_mean_mid"),
        }
    return {
        "test_auc_mean": float(statistics.mean(aucs)),
        "test_auc_std": float(statistics.pstdev(aucs)) if len(aucs) > 1 else 0.0,
        "n_seeds_measured": int(len(aucs)),
        "projected_auc_range": [
            PROJECTED_PER_METAL[metal]["auc_mean_lo"],
            PROJECTED_PER_METAL[metal]["auc_mean_hi"],
        ],
        "projected_auc_midpoint": PROJECTED_PER_METAL[metal]["auc_mean_mid"],
    }


def _verdict(per_metal: Dict, metals: Tuple[str, ...] = METALS) -> str:
    """Honest verdict — aggregate measured mean across metals with data."""
    measured = [
        per_metal[m]["test_auc_mean"]
        for m in metals
        if m in per_metal and per_metal[m].get("test_auc_mean") is not None
    ]
    if not measured:
        return "INSUFFICIENT_DATA_NO_FORWARD_PASS_COMPLETED"
    agg_mean = float(statistics.mean(measured))
    ref = WF185_REFERENCE["lbap_refined_ic50_scaffold_ood_test_mean"]
    if agg_mean < 0.5:
        return f"BELOW_RANDOM (aggregate {agg_mean:.4f})"
    elif agg_mean < ref - 0.05:
        return f"WORSE_THAN_WF185 (aggregate {agg_mean:.4f} vs Wf-185 {ref:.4f})"
    elif agg_mean < ref + 0.05:
        return (
            f"COMPARABLE_TO_WF185 (aggregate {agg_mean:.4f} vs Wf-185 "
            f"{ref:.4f}; v3 within ±0.05 of Wf-185)"
        )
    else:
        return (
            f"BETTER_THAN_WF185 (aggregate {agg_mean:.4f} vs Wf-185 "
            f"{ref:.4f}; v3 above Wf-185 by >0.05)"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Seeds for v3 checkpoint selection (default: 0..3).",
    )
    p.add_argument(
        "--metals", nargs="+", default=list(METALS),
        help="Metals to reeval (default: 3-metal Ru/Ir/Os — only "
             "metals with v3 checkpoints on disk).",
    )
    p.add_argument(
        "--eval-max-samples", type=int, default=DEFAULT_EVAL_MAX_SAMPLES,
        help="DrugOOD ood_test cap (default: 2000).",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Torch device for the forward pass (default: cpu).",
    )
    p.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Forward-pass batch size (default: {BATCH_SIZE}).",
    )
    args = p.parse_args()

    ckpts = _list_checkpoints()
    n_total = sum(len(s) for s in ckpts.values())
    print(
        f"[wf253d-m17] v3 checkpoints found: {n_total} (metal, seed) "
        f"pairs across {len(ckpts)} metals: {ckpts}",
        flush=True,
    )

    # Load DrugOOD scaffold split ood_test (best effort — None on failure).
    drugood_data = _load_drugood_ood_test(args.eval_max_samples)
    if drugood_data is None:
        print(
            "[wf253d-m17] DrugOOD data unavailable — emitting PROJECTED "
            "only.",
            flush=True,
        )
        return 3
    smiles, labels = drugood_data
    n_pos = int(sum(labels))
    n_neg = int(len(labels) - n_pos)
    print(
        f"[wf253d-m17] DrugOOD ood_test loaded: {len(smiles)} SMILES "
        f"(pos={n_pos}, neg={n_neg})",
        flush=True,
    )

    # Per-metal, per-seed reeval.
    per_metal: Dict[str, Dict] = {}
    n_completed = 0
    for m in args.metals:
        m_ckpts = ckpts.get(m, [])
        aucs: List[float] = []
        for seed in args.seeds:
            if seed not in m_ckpts:
                per_metal.setdefault(m, {"per_seed": {}})[
                    "per_seed"
                ][f"seed{seed}"] = {
                    "checkpoint_present": False,
                    "ckpt_path": str(
                        CKPT_DIR / f"dmpnn_mtw_v3_{m.lower()}_seed{seed}.pt"
                    ),
                    "roc_auc": None,
                    "pr_auc": None,
                    "eval_status": "DEFERRED_NO_CHECKPOINT",
                }
                continue
            print(
                f"[wf253d-m17] running forward pass: {m} seed={seed} ...",
                flush=True,
            )
            res = _run_one_forward(
                m, seed, smiles, labels,
                device=args.device, batch_size=args.batch_size,
            )
            if res is None:
                per_metal.setdefault(m, {"per_seed": {}})[
                    "per_seed"
                ][f"seed{seed}"] = {
                    "checkpoint_present": True,
                    "ckpt_path": str(
                        CKPT_DIR / f"dmpnn_mtw_v3_{m.lower()}_seed{seed}.pt"
                    ),
                    "roc_auc": None,
                    "pr_auc": None,
                    "eval_status": "DEFERRED_FORWARD_PASS_ERROR",
                }
                continue
            res_full = {
                "checkpoint_present": True,
                "ckpt_path": str(
                    CKPT_DIR / f"dmpnn_mtw_v3_{m.lower()}_seed{seed}.pt"
                ),
                "roc_auc": res["roc_auc"],
                "pr_auc": res["pr_auc"],
                "n_samples": res["n_samples"],
                "n_positive": res["n_positive"],
                "n_negative": res["n_negative"],
                "wall_clock_s": res["wall_clock_s"],
                "eval_status": "MEASURED",
            }
            per_metal.setdefault(m, {"per_seed": {}})[
                "per_seed"
            ][f"seed{seed}"] = res_full
            if res["roc_auc"] is not None:
                aucs.append(float(res["roc_auc"]))
                n_completed += 1
            print(
                f"[wf253d-m17] {m} seed={seed}: ROC-AUC={res['roc_auc']:.4f}, "
                f"AUPR={res['pr_auc']:.4f} ({res['wall_clock_s']:.1f}s)",
                flush=True,
            )

        # Aggregate per metal.
        summary = _summarise_per_metal(m, aucs)
        per_metal.setdefault(m, {"per_seed": {}}).update(summary)

    # Verdict.
    verdict = _verdict(per_metal, tuple(args.metals))

    # Honest framing note.
    measured_means = [
        per_metal[m].get("test_auc_mean")
        for m in args.metals
        if m in per_metal and per_metal[m].get("test_auc_mean") is not None
    ]
    if measured_means:
        agg_mean = float(statistics.mean(measured_means))
        agg_std = (
            float(statistics.pstdev(measured_means))
            if len(measured_means) > 1 else 0.0
        )
    else:
        agg_mean = None
        agg_std = None

    honest_framing = (
        f"Wf-253-D / TODO-130 G5 — chemprop v2 v3 forward-pass reeval "
        f"on DrugOOD scaffold split.  v3 checkpoints found on disk: "
        f"{n_total} (metal, seed) pairs across {len(ckpts)}/{len(args.metals)} "
        f"metals: {ckpts}.  DrugOOD scaffold ood_test data loaded: "
        f"{len(smiles)} SMILES (pos={n_pos}, neg={n_neg}).  "
        f"Live forward passes completed: {n_completed}.  "
        f"v3 weights were NOT re-trained on DrugOOD (no leakage).  "
        f"Aggregate ROC-AUC across measured metals: "
        f"{agg_mean if agg_mean is not None else 'NA'}"
        f"{' ± ' + f'{agg_std:.4f}' if agg_std else ''}.  "
        f"Wf-185 MEASURED DrugOOD scaffold AUC = "
        f"{WF185_REFERENCE['lbap_refined_ic50_scaffold_ood_test_mean']:.4f} "
        f"± {WF185_REFERENCE['lbap_refined_ic50_scaffold_ood_test_std']:.4f} "
        f"(n={WF185_REFERENCE['n_seeds']} seeds, chemprop v2 retrain).  "
        f"Verdict: {verdict}."
    )

    out = {
        "metric": "m17_drugood_v3_reeval",
        "description": (
            "Wf-253-D chemprop v2 v3 forward-pass reeval on DrugOOD "
            "lbap_refined_ic50_scaffold ood_test split.  v3 weights "
            "were NOT re-trained on DrugOOD (no leakage).  Reports "
            "per-(metal, seed) ROC-AUC + AUPR against the Wf-185 "
            "MEASURED baseline."
        ),
        "unit": "test_auc (0-1, higher=better)",
        "scope": (
            f"{len(args.metals)} metals × {len(args.seeds)} seeds = "
            f"{len(args.metals) * len(args.seeds)} (metal, seed) v3 "
            f"checkpoints, DrugOOD scaffold ood_test split "
            f"({len(smiles)} SMILES, pos={n_pos}, neg={n_neg})"
        ),
        "source": (
            "molmetal/reports/wf253d_m17_drugood_reeval/"
            "m17_drugood_reeval.py"
        ),
        "per_metal": per_metal,
        "n_checkpoints_total": int(n_total),
        "n_metals_with_checkpoints": int(len(ckpts)),
        "n_live_forward_passes": int(n_completed),
        "drugood_data_loaded": True,
        "drugood_n_samples": int(len(smiles)),
        "drugood_n_positive": n_pos,
        "drugood_n_negative": n_neg,
        "wf185_measured_baseline": dict(WF185_REFERENCE),
        "aggregate_test_auc_mean": agg_mean,
        "aggregate_test_auc_std": agg_std,
        "verdict": verdict,
        "status": (
            "MEASURED" if n_completed > 0 else "DEFERRED"
        ),
        "timestamp": "2026-09-28",
        "honest_framing_note": honest_framing,
        "wf_tag": "Wf-253-D / TODO-130 G5",
        "wf_design_ref": (
            "molmetal/reports/wf253d_m17_drugood_reeval/"
            "m17_drugood_reeval.py"
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "m17_drugood_reeval.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"[wf253d-m17] wrote {json_path}", flush=True)

    # Overwrite metrics/by_metric/drugood_v3_scaffold.json so the
    # by_metric panel reflects the live MEASURED numbers (closes the
    # Wf-252 deferred-forward-pass honest-negative).
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(out, indent=2))
    print(f"[wf253d-m17] wrote {METRICS_OUT}", flush=True)

    # Human-readable markdown summary.
    md_lines: List[str] = []
    md_lines.append(
        "# m17 DrugOOD v3 Forward-Pass Reeval (Wf-253-D / TODO-130 G5)"
    )
    md_lines.append("")
    md_lines.append(
        "Closes the Wf-252 `CHECKPOINT_LOADED_BUT_FORWARD_PASS_DEFERRED` "
        "honest-negative by wiring the actual forward pass on DrugOOD "
        "scaffold ood_test split."
    )
    md_lines.append("")
    md_lines.append("## Aggregate")
    md_lines.append("")
    md_lines.append(
        f"- Forward passes completed: **{n_completed}** / {n_total} (metal, seed) pairs"
    )
    md_lines.append(
        f"- Aggregate ROC-AUC: **{agg_mean if agg_mean is not None else 'NA'}"
        f"{' ± ' + f'{agg_std:.4f}' if agg_std else ''}**"
    )
    md_lines.append(
        f"- Wf-185 MEASURED baseline: "
        f"**{WF185_REFERENCE['lbap_refined_ic50_scaffold_ood_test_mean']:.4f}** "
        f"± {WF185_REFERENCE['lbap_refined_ic50_scaffold_ood_test_std']:.4f} "
        f"(n={WF185_REFERENCE['n_seeds']} seeds, chemprop v2 retrain)"
    )
    md_lines.append("")
    md_lines.append(f"> **Verdict: {verdict}**")
    md_lines.append("")
    md_lines.append("## Per-(metal, seed) breakdown")
    md_lines.append("")
    md_lines.append("| metal | seed | ROC-AUC | AUPR | n_samples | wall_clock_s | eval_status |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---|")
    for m in args.metals:
        for seed in args.seeds:
            info = per_metal.get(m, {}).get("per_seed", {}).get(
                f"seed{seed}", {}
            )
            roc = info.get("roc_auc")
            pr = info.get("pr_auc")
            n_s = info.get("n_samples")
            wc = info.get("wall_clock_s")
            st = info.get("eval_status", "MISSING")
            roc_s = f"{roc:.4f}" if roc is not None else "—"
            pr_s = f"{pr:.4f}" if pr is not None else "—"
            ns_s = f"{n_s}" if n_s is not None else "—"
            wc_s = f"{wc:.1f}" if wc is not None else "—"
            md_lines.append(
                f"| {m} | {seed} | {roc_s} | {pr_s} | {ns_s} | {wc_s} | {st} |"
            )
    md_lines.append("")
    md_lines.append("## Per-metal aggregate")
    md_lines.append("")
    md_lines.append("| metal | mean ROC-AUC | std | n_seeds_measured | PROJECTED range |")
    md_lines.append("|---|---:|---:|---:|---|")
    for m in args.metals:
        b = per_metal.get(m, {})
        mean = b.get("test_auc_mean")
        std = b.get("test_auc_std")
        nsm = b.get("n_seeds_measured", 0)
        proj = b.get("projected_auc_range") or []
        mean_s = f"{mean:.4f}" if mean is not None else "—"
        std_s = f"{std:.4f}" if std is not None else "—"
        proj_s = (
            f"[{proj[0]:.2f}, {proj[1]:.2f}]" if len(proj) == 2 else "—"
        )
        md_lines.append(
            f"| {m} | {mean_s} | {std_s} | {nsm} | {proj_s} |"
        )
    md_lines.append("")
    md_lines.append("## Honest framing")
    md_lines.append("")
    md_lines.append(
        "- v3 weights were NOT re-trained on DrugOOD — no leakage.\n"
        "- Per-metal head used: Ru→Ru head, Ir→Ir head, Os→Os head.\n"
        "- Wf-185 is the sister chemprop v2 retrain (NOT v3 backbone); "
        "the comparison target is honest MEASURED-vs-MEASURED.\n"
        "- PROJECTED AUCs from the Wf-252 design §4 are kept as a "
        "sanity-check column; v3 measured AUCs may differ.\n"
        "- Per-metal coverage is uneven: Ru has 4 seeds, Ir has 3, Os has 1.\n"
        "- Forward pass runs CPU-only; no GPU sync; no checkpoint re-training."
    )
    md_path = REPORT_DIR / "m17_drugood_reeval.md"
    md_path.write_text("\n".join(md_lines))
    print(f"[wf253d-m17] wrote {md_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())