"""Generate paper figures from existing JSON / PNG results.

Reads the 32 baseline JSON files in molmetal/reports/, the leakage diagnosis JSON,
and the two FM training-loss PNGs, then writes 6 PNG figures and a README into
molmetal/reports/figures/.

Run:
    source .venv/bin/activate && python -m molmetal.scripts.make_figures
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METALS = ["Ru", "Ir"]
SPLIT_ORDER = ["random", "ligand_dedup", "scaffold", "temporal"]
MODELS_ORDER = ["xgb", "rf", "dmpnn", "dmpnn_attn"]
SPLIT_LABEL = {
    "random": "Random",
    "ligand_dedup": "Ligand dedup",
    "scaffold": "Scaffold",
    "temporal": "Temporal",
}
MODEL_LABEL = {
    "xgb": "XGB (Morgan FP)",
    "rf": "RF (Morgan FP)",
    "dmpnn": "D-MPNN",
    "dmpnn_attn": "Attentive D-MPNN",
}
DPI = 150


def _load_baseline(metal: str, model: str, split: str) -> dict | None:
    """Load a single baseline JSON file if present."""
    if split == "random":
        path = REPORTS_DIR / f"baseline_{metal.lower()}_{model}.json"
    elif model == "dmpnn_attn":
        # Attentive D-MPNN only exists for Ru + temporal split.
        path = REPORTS_DIR / f"baseline_{metal.lower()}_dmpnn_attn_temporal.json"
    else:
        path = REPORTS_DIR / f"baseline_{metal.lower()}_{model}_{split}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _auc(b: dict | None) -> float:
    """Extract test ROC-AUC from a baseline dict, NaN if missing."""
    if b is None:
        return float("nan")
    tm = b.get("test_metrics") or {}
    return float(tm.get("roc_auc", float("nan")))


def fig1_baseline_heatmap() -> Path:
    """fig1_baseline_heatmap.png — 4 splits x 4 models x 2 metals (32 cells)."""
    matrix = np.full((len(SPLIT_ORDER), len(MODELS_ORDER) * len(METALS)), np.nan)
    col_labels: list[str] = []
    for metal in METALS:
        for model in MODELS_ORDER:
            col_labels.append(f"{metal}\n{MODEL_LABEL[model]}")
            for i, split in enumerate(SPLIT_ORDER):
                matrix[i, len(col_labels) - 1] = _auc(_load_baseline(metal, model, split))

    fig, ax = plt.subplots(figsize=(11, 5))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="viridis", aspect="auto", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(SPLIT_ORDER)))
    ax.set_yticklabels([SPLIT_LABEL[s] for s in SPLIT_ORDER])
    ax.set_title("Fig 1. Baseline ROC-AUC across splits / models / metals\n(masked = not run)")
    ax.set_xlabel("Metal & model")
    ax.set_ylabel("Split strictness (low → high)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v < 0.7 else "black",
                )
    fig.colorbar(im, ax=ax, label="Test ROC-AUC")
    fig.tight_layout()
    out = FIGURES_DIR / "fig1_baseline_heatmap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig2_temporal_ooc() -> Path:
    """fig2_temporal_ooc.png — AUC vs split strictness for Ru and Ir (XGB, RF, D-MPNN)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [
        ("Ru — XGB", "xgb", "#1f77b4"),
        ("Ru — RF", "rf", "#2ca02c"),
        ("Ru — D-MPNN", "dmpnn", "#d62728"),
        ("Ir — XGB", "xgb", "#1f77b4"),
        ("Ir — RF", "rf", "#2ca02c"),
        ("Ir — D-MPNN", "dmpnn", "#d62728"),
    ]
    x = np.arange(len(SPLIT_ORDER))
    for label, model, color in series:
        metal = label.split(" — ")[0]
        ys = [_auc(_load_baseline(metal, model, s)) for s in SPLIT_ORDER]
        ls = "--" if "Ir" in label else "-"
        marker = "o"
        ax.plot(x, ys, marker=marker, linestyle=ls, color=color, label=label, linewidth=1.8)

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLIT_ORDER])
    ax.set_xlabel("Split strictness (low → high)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Fig 2. Out-of-distribution degradation\nRandom → Ligand-dedup → Scaffold → Temporal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_temporal_ooc.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig3_leakage_seen_unseen() -> Path:
    """fig3_leakage_seen_unseen.png — AUC on seen vs unseen SMILES."""
    path = REPORTS_DIR / "leakage_diagnosis_data.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        data = json.load(f)

    metals = [m for m in METALS if m in data and "seen_unseen" in data[m]]
    seen = [data[m]["seen_unseen"]["seen"]["roc_auc"] for m in metals]
    unseen = [data[m]["seen_unseen"]["unseen"]["roc_auc"] for m in metals]
    n_seen = [data[m]["seen_unseen"]["n_seen_rows"] for m in metals]
    n_unseen = [data[m]["seen_unseen"]["n_unseen_rows"] for m in metals]

    x = np.arange(len(metals))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    b1 = ax.bar(x - w / 2, seen, w, label=f"Seen (n={n_seen[0] if metals else 0})", color="#4c72b0")
    b2 = ax.bar(
        x + w / 2,
        unseen,
        w,
        label=f"Unseen (n={n_unseen[0] if metals else 0})",
        color="#dd8452",
    )
    for bars, vals in ((b1, seen), (b2, unseen)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metals)
    ax.set_xlabel("Metal")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Fig 3. Seen-vs-unseen SMILES leak diagnosis\n(XGB on the random split, leakage_diagnosis_data.json)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "fig3_leakage_seen_unseen.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig4_attentive_vs_vanilla() -> Path:
    """fig4_attentive_vs_vanilla.png — Ru temporal AUC across the 4 models."""
    models = MODELS_ORDER
    vals = [_auc(_load_baseline("Ru", m, "temporal")) for m in models]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    bars = ax.bar(models, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0.4, 0.7)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_xlabel("Model")
    ax.set_title("Fig 4. Ru temporal-split AUC: vanilla vs attentive D-MPNN\n(Morgan FP baselines shown for reference)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_attentive_vs_vanilla.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig5_dmpnn_wallclock() -> Path:
    """fig5_dmpnn_wallclock.png — scatter (params, s/epoch) for the 4 models.

    Params are approximate (effective-capacity proxies):
        XGB        -> n_estimators=500, max_depth=6  -> 3.0k
        RF         -> n_estimators=500, max_depth=None -> 300k
        D-MPNN     -> hidden=256, depth=3              -> 1.1M
        Attn D-MPNN-> hidden=256, depth=3, attn heads  -> 1.3M
    """
    rows = []
    for m in MODELS_ORDER:
        b = _load_baseline("Ru", m, "temporal")
        if b is None:
            continue
        elapsed = float(b.get("elapsed_seconds", float("nan")))
        rows.append((m, elapsed))

    # Param proxies (effective trainable params, approximate)
    param_proxy_k = {
        "xgb": 3.0,          # ~500 trees * ~6 splits worth of leaves
        "rf": 300.0,         # 500 trees * ~600 leaves
        "dmpnn": 1100.0,     # ~1.1 M params (MolMetal default)
        "dmpnn_attn": 1300.0,
    }
    xs = [param_proxy_k[m] for m, _ in rows]
    ys = [t / 60.0 for _, t in rows]  # convert seconds → minutes per *training run*
    labels = [m for m, _ in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"xgb": "#4c72b0", "rf": "#55a868", "dmpnn": "#c44e52", "dmpnn_attn": "#8172b2"}
    for x, y, lbl in zip(xs, ys, labels):
        ax.scatter(x, y, s=110, color=colors[lbl], label=MODEL_LABEL[lbl], edgecolor="black")
        ax.annotate(lbl, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Approx. effective params (k, log scale)")
    ax.set_ylabel("Wall-clock per training run (min, log scale)")
    ax.set_title("Fig 5. Compute cost vs capacity on Ru temporal split")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "fig5_dmpnn_wallclock.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig6_fm_loss_curves() -> Path:
    """fig6_fm_loss_curves.png — 2-panel: ru FM + pocket FM training loss."""
    ru_path = REPORTS_DIR / "fm_ru_train_loss.png"
    pk_path = REPORTS_DIR / "fm_pocket_train_loss.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, src, title in (
        (axes[0], ru_path, "Ru temporal FM — training loss"),
        (axes[1], pk_path, "Pocket MMP2 FM — training loss"),
    ):
        if src.exists():
            img = mpimg.imread(src)
            ax.imshow(img)
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, f"missing:\n{src.name}", ha="center", va="center")
            ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Fig 6. Flow-matching training-loss curves (re-plotted from saved PNGs)", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "fig6_fm_loss_curves.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


FIGURES = [
    (
        "fig1_baseline_heatmap.png",
        "Heatmap of test ROC-AUC across 4 split strategies × 4 models × 2 metals.",
        "molmetal/reports/baseline_*.json (32 files)",
        fig1_baseline_heatmap,
    ),
    (
        "fig2_temporal_ooc.png",
        "ROC-AUC vs split strictness for Ru and Ir (XGB / RF / D-MPNN).",
        "molmetal/reports/baseline_ru_xgb*.json, baseline_ir_xgb*.json, "
        "baseline_*rf*.json, baseline_*dmpnn*.json",
        fig2_temporal_ooc,
    ),
    (
        "fig3_leakage_seen_unseen.png",
        "Test ROC-AUC on seen vs unseen SMILES per metal (XGB random split).",
        "molmetal/reports/leakage_diagnosis_data.json",
        fig3_leakage_seen_unseen,
    ),
    (
        "fig4_attentive_vs_vanilla.png",
        "Ru temporal-split AUC for XGB / RF / D-MPNN / Attentive D-MPNN.",
        "molmetal/reports/baseline_ru_xgb_temporal.json, "
        "baseline_ru_rf_temporal.json, baseline_ru_dmpnn_temporal.json, "
        "baseline_ru_dmpnn_attn_temporal.json",
        fig4_attentive_vs_vanilla,
    ),
    (
        "fig5_dmpnn_wallclock.png",
        "Wall-clock per training run vs approximate effective params (log–log).",
        "molmetal/reports/baseline_ru_*_temporal.json (elapsed_seconds)",
        fig5_dmpnn_wallclock,
    ),
    (
        "fig6_fm_loss_curves.png",
        "Two-panel flow-matching training-loss curves (Ru + Pocket MMP2).",
        "molmetal/reports/fm_ru_train_loss.png, "
        "molmetal/reports/fm_pocket_train_loss.png",
        fig6_fm_loss_curves,
    ),
]


def write_readme(elapsed: float, paths: list[Path]) -> Path:
    """Write the figures README with one caption per figure and data source."""
    lines = [
        "# Paper Figures",
        "",
        "Generated by `python -m molmetal.scripts.make_figures` from the existing",
        "`molmetal/reports/` JSON / PNG artifacts. All figures use `dpi=150`.",
        "",
        f"Total generation time: **{elapsed:.2f} s**.",
        "",
        "## Figures",
        "",
    ]
    for path, caption, source, _ in FIGURES:
        lines.append(f"### {path}")
        lines.append("")
        lines.append(f"- Caption: {caption}")
        lines.append(f"- Data source: `{source}`")
        lines.append("")
    out = FIGURES_DIR / "README.md"
    out.write_text("\n".join(lines))
    return out


def main() -> list[Path]:
    paths: list[Path] = []
    started = time.time()
    for _, _, _, fn in FIGURES:
        t0 = time.time()
        out = fn()
        dt = time.time() - t0
        print(f"  wrote {out.relative_to(REPORTS_DIR)} ({dt:.2f} s)")
        paths.append(out)
    elapsed = time.time() - started
    paths.append(write_readme(elapsed, paths))
    print(f"\nGenerated {len(paths) - 1} figures + README.md in {elapsed:.2f} s.")
    return paths


if __name__ == "__main__":
    main()
