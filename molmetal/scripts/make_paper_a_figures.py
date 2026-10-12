#!/usr/bin/env python3
"""Generate the three Paper A figures from the on-disk metric artifacts (BLK-PA8).

Every panel reads a ``metrics/by_metric/*.json`` file and plots its MEASURED
values; nothing is hard-coded, so regenerating after a re-run is one command.
Outputs PDF (vector, for LaTeX) and PNG (for review) into ``paper_a/figures/``.

    .venv/bin/python molmetal/scripts/make_paper_a_figures.py
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
OUT = PROJECT_ROOT / "paper_a" / "figures"
METALS = ["Ru", "Ir", "Rh", "Os", "Re"]
COL_WITHIN = "#3b6ea5"
COL_SCAF = "#7fb3d5"
COL_CROSS = "#c0392b"


def _load(name: str) -> dict:
    return json.loads((METRICS / name).read_text())


def fig1_transfer() -> None:
    stab = _load("metal_activity_transfer_seed_stability.json")
    scaf = _load("metal_activity_transfer_scaffold.json")
    perm = _load("metal_activity_transfer_permutation_null.json")["permutation_null"]

    within = [stab["within_metal_stability"][m]["mean"] for m in METALS]
    wlo = [stab["within_metal_stability"][m]["min"] for m in METALS]
    whi = [stab["within_metal_stability"][m]["max"] for m in METALS]
    scaff = [scaf["net_metal_effect"][m]["within_scaffold_auc"] for m in METALS]
    cross = [scaf["net_metal_effect"][m]["cross_metal_auc"] for m in METALS]

    x = np.arange(len(METALS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - w, within, w, yerr=[np.array(within) - np.array(wlo),
                                   np.array(whi) - np.array(within)],
           capsize=3, color=COL_WITHIN, label="Within-metal (5-fold)")
    ax.bar(x, scaff, w, color=COL_SCAF, label="Within-metal scaffold")
    ax.bar(x + w, cross, w, color=COL_CROSS, label="Cross-metal (LOMO)")
    ax.axhline(0.5, color="0.4", ls="--", lw=1)
    ax.text(len(METALS) - 0.5, 0.505, "chance", color="0.4", fontsize=7, ha="right")

    labels = []
    for m in METALS:
        v = perm.get(m, {})
        verdict = "above\n(sig)" if v.get("above_chance_significant") else "in null"
        labels.append(f"{m}\n$n$={v.get('n_test', '?')}\n{verdict}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.28)
    ax.legend(fontsize=7, loc="upper right", frameon=False, ncol=1)
    ax.set_title("Cross-metal transfer and its scaffold control", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_transfer.{ext}", dpi=200)
    plt.close(fig)


def fig2_pretrain() -> None:
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
        fig.savefig(OUT / f"fig2_pretrain.{ext}", dpi=200)
    plt.close(fig)


def fig3_dedup_tie() -> None:
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
        fig.savefig(OUT / f"fig3_dedup_tie.{ext}", dpi=200)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_transfer()
    fig2_pretrain()
    fig3_dedup_tie()
    for f in sorted(OUT.glob("fig*.pdf")):
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
