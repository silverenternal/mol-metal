#!/usr/bin/env python3
"""Generate the Paper A figures from the on-disk run reports / metric artifacts.

Every panel reads a run report under ``molmetal/reports/`` or a
``metrics/by_metric/*.json`` file and plots its MEASURED values; nothing is
hard-coded, so regenerating after a re-run is one command.

Outputs PDF (vector, for LaTeX) and PNG (for review) into ``paper_a/figures/``.

    .venv/bin/python molmetal/scripts/make_paper_a_figures.py

Main text:  fig1_transfer (collapse), fig2_kshot_curve (FSMA K-curve).
Supplement: si_pretrain (tmQM A/B), si_dedup_tie (tie rule).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = PROJECT_ROOT / "metrics" / "by_metric"
REPORTS = PROJECT_ROOT / "molmetal" / "reports"
OUT = PROJECT_ROOT / "paper_a" / "figures"
METALS = ["Ru", "Ir", "Rh", "Os", "Re"]
COL_WITHIN = "#3b6ea5"
COL_CROSS = "#c0392b"


def _load(name: str) -> dict:
    return json.loads((METRICS / name).read_text())


def _report(label: str) -> dict:
    return json.loads((REPORTS / label / "metrics.json").read_text())


def _within_metal(run: dict) -> dict:
    return {m: run["within_metal"][m]["mean_auc"] for m in METALS}


def _zero_shot(run: dict) -> dict:
    out = {}
    for fold in run["per_fold"]:
        vals = [r["kshot"]["zero_shot_query_auc"] for r in fold["per_seed"].values()
                if r.get("kshot")]
        if vals:
            out[fold["metal"]] = float(np.mean(vals))
    return out


def fig1_transfer() -> None:
    ceiling = _report("l3_within_pf_ceiling")
    cross = _zero_shot(_report("l3_kshot_pf_k80"))
    within = _within_metal(ceiling)
    n_unique = {m: ceiling["within_metal"][m]["n_unique"] for m in METALS}

    x = np.arange(len(METALS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(x - w / 2, [within[m] for m in METALS], w, color=COL_WITHIN,
           label="Within-metal (5-fold)")
    ax.bar(x + w / 2, [cross[m] for m in METALS], w, color=COL_CROSS,
           label="Cross-metal (LOMO)")
    for i, m in enumerate(METALS):
        ax.text(i, 0.02, f"$-$ {within[m] - cross[m]:.3f}", ha="center",
                fontsize=7, color="0.25")
    ax.axhline(0.5, color="0.4", ls="--", lw=1)
    ax.text(len(METALS) - 0.5, 0.505, "chance", color="0.4", fontsize=7, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n$n$={n_unique[m]}" for m in METALS], fontsize=7)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_title("Cross-metal transfer collapses on every metal", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_transfer.{ext}", dpi=200)
    plt.close(fig)


def fig2_kshot_curve() -> None:
    stats = _load("fsma_paper_stats.json")
    runs = {"5": "head_k5", "10": "head_k10", "20": "head_k20",
            "40": "head_k40", "80": "head_k80"}
    ks = [int(k) for k in runs]
    pooled, ci, per_metal = [], [], {m: [] for m in METALS}
    for k in ks:
        arm = stats["per_run"][runs[str(k)]]["arms"]["kshot"]
        p = arm["_pooled_ru_dominated"]
        pooled.append(p["mean"])
        ci.append((p["ci95_ttest"][0], p["ci95_ttest"][1]))
        for m in METALS:
            per_metal[m].append(arm[m]["mean"])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    lo = [pooled[i] - ci[i][0] for i in range(len(ks))]
    hi = [ci[i][1] - pooled[i] for i in range(len(ks))]
    ax.errorbar(ks, pooled, yerr=[lo, hi], fmt="-o", color="k", lw=2,
                capsize=3, label="pooled (Ru-dominated)")
    colors = {"Ru": "#7f8c8d", "Ir": "#2980b9", "Rh": "#27ae60",
              "Os": "#e67e22", "Re": "#8e44ad"}
    for m in METALS:
        ax.plot(ks, per_metal[m], "--o", ms=3, lw=1, color=colors[m],
                alpha=0.85, label=m)
    ax.axhline(0.0, color="0.5", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("support ligands $K$")
    ax.set_ylabel("paired $\\Delta$ ROC-AUC vs zero-shot")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    ax.set_title("FSMA recovers cross-metal AUC, monotonically in $K$", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_kshot_curve.{ext}", dpi=200)
    plt.close(fig)


def si_pretrain() -> None:
    ab = _load("tmqm_pretrain_init_ab_ru.json")
    rnd = ab["arms"]["random"]["per_seed"]
    tmqm = ab["arms"]["tmqm"]["per_seed"]
    n = min(len(rnd), len(tmqm))

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for i in range(n):
        up = tmqm[i] >= rnd[i]
        ax.plot([0, 1], [rnd[i], tmqm[i]], "-o", ms=4, lw=1.2,
                color="#2e7d32" if up else "#c0392b", alpha=0.85)
    ax.plot([0, 1], [np.mean(rnd[:n]), np.mean(tmqm[:n])], "k-", lw=2.2,
            marker="s", ms=6, label="mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["random init", "tmQM init"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("Ru ROC-AUC")
    d = ab["paired_delta"]
    ax.set_title(
        f"paired $\\Delta$ = {d['mean']:+.4f}\n95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]",
        fontsize=8,
    )
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"si_pretrain.{ext}", dpi=200)
    plt.close(fig)


def si_dedup_tie() -> None:
    tr = _load("metal_activity_transfer.json")
    tie = _load("metal_activity_transfer_tie_sensitivity.json")
    dup = tr["duplication"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ax = axes[0]
    x = np.arange(len(METALS))
    w = 0.38
    raw = [dup[m]["n_rows_raw"] for m in METALS]
    uniq = [dup[m]["n_unique"] for m in METALS]
    ax.bar(x - w / 2, raw, w, color="#95a5a6", label="raw rows")
    ax.bar(x + w / 2, uniq, w, color="#2c3e50", label="unique ligands")
    for i, m in enumerate(METALS):
        r = 1 - uniq[i] / raw[i]
        ax.text(i, raw[i] * 1.02, f"{r*100:.0f}%", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(METALS)
    ax.set_ylabel("count")
    ax.set_title("Deduplication is severe and uneven", fontsize=8)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    rules = [("ties_to_ACTIVE_current", "ties$\\to$active (current)", "#c0392b"),
             ("ties_to_INACTIVE_conservative", "ties$\\to$inactive", "#2e7d32")]
    mk = [m for m in METALS if m in tie["within_metal_auc_by_rule"]["ties_to_ACTIVE_current"]]
    x = np.arange(len(mk))
    for j, (key, lab, col) in enumerate(rules):
        block = tie["within_metal_auc_by_rule"][key]
        lo = [block[m]["min"] for m in mk]
        hi = [block[m]["max"] for m in mk]
        mid = [(a + b) / 2 for a, b in zip(lo, hi)]
        ax.errorbar(x + (j - 0.5) * 0.18, mid,
                    yerr=[np.array(mid) - np.array(lo), np.array(hi) - np.array(mid)],
                    fmt="o", ms=4, capsize=3, color=col, label=lab)
    ax.axhline(0.81, color="0.4", ls="--", lw=1)
    ax.text(len(mk) - 0.5, 0.813, "Krasnov Ru 0.81", fontsize=6.5, ha="right", color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels(mk)
    ax.set_ylabel("within-metal ROC-AUC")
    ax.set_title("The tie rule decides the Ru comparison", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"si_dedup_tie.{ext}", dpi=200)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_transfer()
    fig2_kshot_curve()
    si_pretrain()
    si_dedup_tie()
    for f in sorted(OUT.glob("*.pdf")):
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
