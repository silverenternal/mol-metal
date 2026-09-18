"""Figure 4: Homotype distance vs Tanimoto distance scatter plot.

Plots the 20 MEASURED (homotype_distance, tanimoto_distance) pairs from
WF-Lambda-2.E (10 isomers + 10 unrelated) on a unit square [0,1]^2 with
quadrant guide-lines at x=0.5 / y=0.5.  Quadrant annotations:

  - Upper-left (low Tanimoto, high homotype):
      "Lambda-discovered-but-Tanimoto-invisible chemistry"
      -- the contribution our paper makes.
  - Lower-right (high Tanimoto, low homotype):
      "Tanimoto home turf (constitutional isomerism)"

Provenance: molmetal/reports/wf_lambda2e_compare/pairs.csv (canonical,
written by molmetal/scripts/wf_lambda2e_compare.py).

Reproducibility:
    uv run paper/figures/fig4_homotype_vs_tanimoto.py
Outputs (in this directory):
    fig4_homotype_vs_tanimoto.png  (600 dpi)
    fig4_homotype_vs_tanimoto.svg  (scalable)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "fig4_homotype_vs_tanimoto.png"
SVG_PATH = OUT_DIR / "fig4_homotype_vs_tanimoto.svg"

# Input -- the canonical CSV produced by molmetal/scripts/wf_lambda2e_compare.py.
PAIRS_CSV = Path(
    "/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda2e_compare/pairs.csv"
)

# ---------------------------------------------------------------------------
# Figure 1-compatible phase-band palette (paper/figures/fig1_mlc_architecture.py).
# We reuse the 4 phase-band hues so the scatter harmonises with the MLC
# architecture diagram (parsing=blue, chemistry=green, search=orange,
# output=red).
# ---------------------------------------------------------------------------
PHASE_EDGE = {
    "parsing":   "#1f4e79",
    "chemistry": "#1e6b3a",
    "search":    "#a3560b",
    "output":    "#8b1a1a",
}
PHASE_FILL = {
    "parsing":   "#cfe2ff",
    "chemistry": "#d4edda",
    "search":    "#ffe5b4",
    "output":    "#f8d7da",
}
# Subset encoding for our 20-pair data:
#   isomers    -> "chemistry" (green)  -- organic constitutional isomers
#   unrelated  -> "output"    (red)    -- across different chemistries
SUBSET_PHASE = {"isomers": "chemistry", "unrelated": "output"}


# ---------------------------------------------------------------------------
# Data loading (stdlib only -- no pandas dependency).
# ---------------------------------------------------------------------------

def load_pairs(csv_path: Path) -> list[dict]:
    """Load pairs.csv into a list of dicts.  Expected columns:
    subset, i, j, smi_i, smi_j, tanimoto_distance, homotype_distance,
    homotype_minus_tanimoto."""
    import csv
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "subset": row["subset"],
                    "i": row["i"],
                    "j": row["j"],
                    "tanimoto_distance": float(row["tanimoto_distance"]),
                    "homotype_distance": float(row["homotype_distance"]),
                }
            )
    return rows


def quadrant_label(x: float, y: float) -> str:
    """2-letter quadrant label: first char = Tanimoto (L|R), second = Homotype (L|U)."""
    return ("L" if x < 0.5 else "R") + ("L" if y < 0.5 else "U")


# Canonical key set matches quadrant_label output: Tanimoto-letter + Homotype-letter.
QUADRANT_KEYS = ("LL", "LU", "RL", "RU")


def quadrant_count_init() -> dict:
    return {k: 0 for k in QUADRANT_KEYS}


def render() -> Tuple[int, dict, dict]:
    """Render the scatter.  Returns (n_pairs, png_size, svg_size)."""
    pairs = load_pairs(PAIRS_CSV)
    n_pairs = len(pairs)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=120)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    # ----- Background quadrant tinting (very faint) ---------------------------
    quad_colors = {
        "LU": "#fdecec",  # upper-left  -- our contribution
        "RU": "#fdecec",  # upper-right -- partial overlap
        "LD": "#ecf6ec",  # lower-left  -- both low
        "RD": "#ecf6ec",  # lower-right -- Tanimoto home turf
    }
    for q, color in quad_colors.items():
        x0 = 0.0 if q[0] == "L" else 0.5
        x1 = 0.5 if q[0] == "L" else 1.0
        y0 = 0.0 if q[1] == "L" else 0.5
        y1 = 0.5 if q[1] == "L" else 1.0
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor=color, edgecolor="none", alpha=0.35, zorder=0,
            )
        )

    # ----- Quadrant guide lines at x=0.5 / y=0.5 ------------------------------
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1.0, zorder=1)

    # ----- Scatter points (phase-coded by subset) -----------------------------
    for p in pairs:
        phase = SUBSET_PHASE.get(p["subset"], "output")
        edge = PHASE_EDGE[phase]
        fill = PHASE_FILL[phase]
        ax.scatter(
            p["tanimoto_distance"],
            p["homotype_distance"],
            s=120,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.6,
            alpha=0.92,
            zorder=3,
        )

    # ----- Per-point id labels (i,j) for traceability ------------------------
    for p in pairs:
        x = p["tanimoto_distance"]
        y = p["homotype_distance"]
        # Offset label depending on quadrant to avoid overlapping the point.
        if x < 0.5:
            dx, ha = 0.012, "left"
        else:
            dx, ha = -0.012, "right"
        if y < 0.5:
            dy, va = -0.014, "top"
        else:
            dy, va = 0.014, "bottom"
        ax.annotate(
            f"({p['i'][:8]},{p['j'][:8]})",
            xy=(x, y),
            xytext=(x + dx, y + dy),
            ha=ha, va=va,
            fontsize=6.2,
            color="#444444",
            zorder=4,
        )

    # ----- Quadrant annotations ----------------------------------------------
    # Upper-left = Lambda-discovered-but-Tanimoto-invisible (OUR contribution)
    ax.text(
        0.02, 0.985,
        "Upper-left -- Lambda-discovered-but-Tanimoto-invisible\n"
        "chemistry (contribution of this paper)",
        ha="left", va="top",
        fontsize=10.0, fontweight="bold",
        color=PHASE_EDGE["output"],
    )
    # Lower-right = Tanimoto home turf (constitutional isomerism)
    ax.text(
        0.98, 0.015,
        "Lower-right -- Tanimoto home turf\n(constitutional isomerism)",
        ha="right", va="bottom",
        fontsize=10.0, fontweight="bold",
        color=PHASE_EDGE["chemistry"],
    )
    # Upper-right neutral note
    ax.text(
        0.98, 0.985,
        "Upper-right -- both metrics\nagree (high distance)",
        ha="right", va="top",
        fontsize=9.0, fontstyle="italic",
        color="#444444",
    )
    # Lower-left neutral note
    ax.text(
        0.02, 0.015,
        "Lower-left -- both metrics\nagree (low distance)",
        ha="left", va="bottom",
        fontsize=9.0, fontstyle="italic",
        color="#444444",
    )

    # ----- Axes / grid / labels ----------------------------------------------
    ax.set_xlabel(
        "Tanimoto distance (1 - Tanimoto similarity on Morgan-2 fp)",
        fontsize=11.0, fontweight="bold",
    )
    ax.set_ylabel(
        "Homotype distance (typed-variable edit distance on homotype_enriched_vocab)",
        fontsize=11.0, fontweight="bold",
    )
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=9.0)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cccccc", zorder=1)
    ax.set_axisbelow(True)

    # ----- Legend (subset = phase band) --------------------------------------
    handles = [
        mpatches.Patch(
            facecolor=PHASE_FILL["chemistry"],
            edgecolor=PHASE_EDGE["chemistry"],
            label="Constitutional isomers (n=10)",
        ),
        mpatches.Patch(
            facecolor=PHASE_FILL["output"],
            edgecolor=PHASE_EDGE["output"],
            label="Unrelated chemistries (n=10)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.18),
        frameon=True, fontsize=9.0,
    )

    # ----- Title + provenance caption ----------------------------------------
    ax.set_title(
        "Homotype vs Tanimoto distance -- 20 pairs from WF-Lambda-2.E",
        fontsize=12.5, fontweight="bold", pad=14,
    )
    fig.text(
        0.5, 0.012,
        ("MEASURED on homotype_enriched_vocab "
         "(molmetal/reports/wf_lambda2e_compare/pairs.csv). "
         "Quadrant guide lines at x=0.5, y=0.5."),
        ha="center", va="bottom",
        fontsize=8.2, fontstyle="italic", color="#555555",
    )

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    fig.savefig(SVG_PATH, format="svg", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)

    # ----- Quadrant counts for downstream reproducibility --------------------
    counts = quadrant_count_init()
    for p in pairs:
        counts[quadrant_label(p["tanimoto_distance"],
                              p["homotype_distance"])] += 1

    return n_pairs, counts, {"png": PNG_PATH, "svg": SVG_PATH}


def main() -> None:
    n_pairs, counts, paths = render()
    png_size = paths["png"].stat().st_size
    svg_size = paths["svg"].stat().st_size
    print(f"[fig4] wrote {paths['png'].name}  "
          f"({round(png_size / 1024.0, 1)} KB, 600 dpi)")
    print(f"[fig4] wrote {paths['svg'].name}  "
          f"({round(svg_size / 1024.0, 1)} KB)")
    print(f"[fig4] n_pairs = {n_pairs}")
    print(f"[fig4] quadrant counts = {counts}")


if __name__ == "__main__":
    main()
