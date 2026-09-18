"""Figure 1: 9-Layer Molecular Lambda Calculus (MLC) Architecture Diagram.

Renders the MLC stack as 9 colour-coded layer boxes with arrows for
data flow. Layer semantics follow the canonical 9-layer MLC
interpretation used in the paper; the metric catalogue
(`molmetal/reports/lambda_layer_metrics.md`) supplies the per-layer
metric counts surfaced in the caption.

Reproducibility:
    uv run paper/figures/fig1_mlc_architecture.py
Outputs (in this directory):
    fig1_mlc_architecture.png  (600 dpi)
    fig1_mlc_architecture.svg  (scalable)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "fig1_mlc_architecture.png"
SVG_PATH = OUT_DIR / "fig1_mlc_architecture.svg"

# ---------------------------------------------------------------------------
# Layer specification (the 9-layer MLC stack; see fig1_caption.md)
# ---------------------------------------------------------------------------

# Phase bands (color-coded):
#   parsing (blue): L1-L3
#   chemistry (green): L4-L6
#   search (orange): L7
#   output gates (red): L8-L9
PHASE_STYLE: dict[str, Tuple[str, str]] = {
    "parsing":   ("#cfe2ff", "#1f4e79"),  # light blue fill, dark blue edge
    "chemistry": ("#d4edda", "#1e6b3a"),  # light green fill, dark green edge
    "search":    ("#ffe5b4", "#a3560b"),  # light orange fill, dark orange edge
    "output":    ("#f8d7da", "#8b1a1a"),  # light red fill, dark red edge
}


@dataclass(frozen=True)
class Layer:
    idx: int
    name: str
    brief: str
    input_t: str
    output_t: str
    invariant: str
    phase: str


LAYERS: list[Layer] = [
    Layer(
        idx=1,
        name="Lexer",
        brief="SMILES -> token stream",
        input_t="SMILES string",
        output_t="Token stream",
        invariant="Lexical completeness",
        phase="parsing",
    ),
    Layer(
        idx=2,
        name="Parser",
        brief="Tokens -> AST",
        input_t="Token stream",
        output_t="Abstract syntax tree",
        invariant="Parse-tree well-formedness",
        phase="parsing",
    ),
    Layer(
        idx=3,
        name="AST Canonicaliser",
        brief="Alpha-equivalence class",
        input_t="Abstract syntax tree",
        output_t="Canonical AST (alpha-NF)",
        invariant="Alpha-NF uniqueness per class",
        phase="parsing",
    ),
    Layer(
        idx=4,
        name="Click Chemistry Rules",
        brief="5 typed reductions",
        input_t="Pair of closed terms",
        output_t="Product closed terms",
        invariant="Mass balance = {}",
        phase="chemistry",
    ),
    Layer(
        idx=5,
        name="Tile Library",
        brief="204 SMARTS-diverse fragments",
        input_t="SMILES + tag list",
        output_t="12-204 Tile instances",
        invariant="Canonical SMILES uniqueness",
        phase="chemistry",
    ),
    Layer(
        idx=6,
        name="Beta-Reduction Engine",
        brief="Lambda-calc interpreter",
        input_t="Reactant pair + rule",
        output_t="Reduced closed term",
        invariant="Confluence (Church-Rosser)",
        phase="chemistry",
    ),
    Layer(
        idx=7,
        name="MCTS Search",
        brief="UCB + VirtualLoss + TT",
        input_t="Seed term + tiles + rules",
        output_t="Top-K candidates + history",
        invariant="Best-score monotone non-dec.",
        phase="search",
    ),
    Layer(
        idx=8,
        name="MetalGeometryPrior",
        brief="CN + geometry prior",
        input_t="Candidate + metal symbol",
        output_t="Geometry-tagged 3D coords",
        invariant="Pt(II)=4; Ru/Ir(III)=6",
        phase="output",
    ),
    Layer(
        idx=9,
        name="Validity + Dedup Gate",
        brief="Validity + alpha-equiv dedup",
        input_t="Candidate SMILES stream",
        output_t="Unique valid SMILES set",
        invariant="RDKit-parseable, dedup OK",
        phase="output",
    ),
]


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 12.0, 12.5  # inches
BOX_W, BOX_H = 7.8, 0.95
LEFT_X = 1.55
ARROW_GAP = 0.12
BOX_TOP = 10.65  # y-coord of topmost box centre


def box_y_centres() -> list[float]:
    """Vertical centres for the 9 layer boxes (top-down)."""
    return [BOX_TOP - i * (BOX_H + ARROW_GAP) for i in range(len(LAYERS))]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_layer_box(ax, layer: Layer, y_centre: float) -> FancyBboxPatch:
    fill, edge = PHASE_STYLE[layer.phase]
    x0 = LEFT_X
    patch = FancyBboxPatch(
        (x0, y_centre - BOX_H / 2),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4,
        facecolor=fill,
        edgecolor=edge,
    )
    ax.add_patch(patch)

    # ---- Left column: name + brief + I/O ----
    pad_l = 0.18
    title = f"L{layer.idx}  |  {layer.name}"
    ax.text(
        x0 + pad_l,
        y_centre + 0.26,
        title,
        ha="left",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x0 + pad_l,
        y_centre + 0.04,
        layer.brief,
        ha="left",
        va="center",
        fontsize=9.2,
        fontstyle="italic",
        color="#222222",
    )
    ax.text(
        x0 + pad_l,
        y_centre - 0.20,
        f"in : {layer.input_t}",
        ha="left",
        va="center",
        fontsize=8.2,
        color="#333333",
    )
    ax.text(
        x0 + pad_l,
        y_centre - 0.36,
        f"out: {layer.output_t}",
        ha="left",
        va="center",
        fontsize=8.2,
        color="#333333",
    )

    # ---- Right column: invariant ----
    pad_r = 0.18
    ax.text(
        x0 + BOX_W - pad_r,
        y_centre + 0.26,
        "INVARIANT",
        ha="right",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x0 + BOX_W - pad_r,
        y_centre + 0.05,
        layer.invariant,
        ha="right",
        va="center",
        fontsize=8.6,
        color="#222222",
        wrap=True,
    )

    # Vertical divider line between left/right column
    ax.plot(
        [x0 + BOX_W * 0.66, x0 + BOX_W * 0.66],
        [y_centre - BOX_H / 2 + 0.06, y_centre + BOX_H / 2 - 0.06],
        color=edge,
        linewidth=0.7,
        alpha=0.55,
    )

    return patch


def draw_arrow(ax, y_top: float, y_bot: float, phase_top: str, phase_bot: str) -> None:
    color = PHASE_STYLE[phase_top][1] if phase_top == phase_bot else "#555555"
    arrow = FancyArrowPatch(
        (LEFT_X + BOX_W / 2, y_top),
        (LEFT_X + BOX_W / 2, y_bot),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.4,
        color=color,
    )
    ax.add_patch(arrow)


def draw_phase_labels(ax, y_centres: list[float]) -> None:
    """Right-side phase band labels spanning the appropriate layers."""
    spans = [
        ("Parsing (L1-L3)",        "parsing",   [0, 1, 2]),
        ("Chemistry (L4-L6)",      "chemistry", [3, 4, 5]),
        ("Search (L7)",            "search",    [6]),
        ("Output gates (L8-L9)",   "output",    [7, 8]),
    ]
    label_x = LEFT_X + BOX_W + 0.55
    for label, phase, idxs in spans:
        y_top = y_centres[idxs[0]] + BOX_H / 2 + 0.04
        y_bot = y_centres[idxs[-1]] - BOX_H / 2 - 0.04
        edge = PHASE_STYLE[phase][1]
        # vertical bracket
        ax.plot([label_x - 0.10, label_x - 0.10], [y_bot, y_top],
                color=edge, linewidth=1.8)
        ax.plot([label_x - 0.16, label_x - 0.10], [y_top, y_top],
                color=edge, linewidth=1.8)
        ax.plot([label_x - 0.16, label_x - 0.10], [y_bot, y_bot],
                color=edge, linewidth=1.8)
        mid_y = (y_top + y_bot) / 2
        ax.text(
            label_x + 0.02,
            mid_y,
            label,
            ha="left",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=edge,
            rotation=90,
        )


def draw_input_output_pills(ax, y_centres: list[float]) -> None:
    """Left-side pill: 'SMILES' feeds L1; right-side pill: 'Valid unique
    SMILES set' is the L9 output."""
    # Input pill (left of L1)
    in_x = LEFT_X - 0.95
    y = y_centres[0]
    ax.add_patch(
        FancyBboxPatch(
            (in_x - 0.65, y - 0.22), 1.30, 0.44,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#f0f0f0", edgecolor="#444444", linewidth=1.0,
        )
    )
    ax.text(in_x, y + 0.06, "SMILES seed", ha="center", va="center",
            fontsize=8.6, fontweight="bold", color="#222222")
    ax.text(in_x, y - 0.08, "(closed term)", ha="center", va="center",
            fontsize=7.6, fontstyle="italic", color="#444444")
    ax.add_patch(
        FancyArrowPatch(
            (in_x + 0.65, y),
            (LEFT_X, y),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.2, color="#444444",
        )
    )

    # Output pill (right of L9, placed BELOW the rotated phase label)
    # The rotated "Output gates (L8-L9)" label sits at LEFT_X+BOX_W+0.55,
    # spanning L8-L9 vertically, so we put the output pill further right
    # and slightly lower (just under the L9 box bottom).
    out_x = LEFT_X + BOX_W + 0.95
    y = y_centres[-1] - BOX_H / 2 - 0.85
    ax.add_patch(
        FancyBboxPatch(
            (out_x - 0.85, y - 0.22), 1.70, 0.44,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#f0f0f0", edgecolor="#444444", linewidth=1.0,
        )
    )
    ax.text(out_x, y + 0.06, "Unique valid", ha="center", va="center",
            fontsize=8.6, fontweight="bold", color="#222222")
    ax.text(out_x, y - 0.08, "SMILES set", ha="center", va="center",
            fontsize=7.6, fontstyle="italic", color="#444444")
    # Arrow from L9 right edge curving down into the pill
    arrow = FancyArrowPatch(
        (LEFT_X + BOX_W + 0.05, y_centres[-1] - BOX_H / 2),
        (out_x - 0.85, y + 0.05),
        connectionstyle="arc3,rad=-0.35",
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color="#444444",
    )
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render() -> Tuple[int, int]:
    """Render the figure. Returns (n_arrows, file_size_bytes_png)."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=120)
    ax.set_xlim(0, FIG_W + 1.0)  # give room for right-side phase labels
    ax.set_ylim(-1.2, FIG_H + 0.4)
    ax.set_aspect("auto")
    ax.axis("off")

    y_centres = box_y_centres()
    n_arrows = 0

    # Draw boxes
    for layer, y in zip(LAYERS, y_centres):
        draw_layer_box(ax, layer, y)

    # Vertical arrows between consecutive boxes
    for i in range(len(LAYERS) - 1):
        y_top = y_centres[i] - BOX_H / 2
        y_bot = y_centres[i + 1] + BOX_H / 2
        draw_arrow(
            ax,
            y_top,
            y_bot,
            LAYERS[i].phase,
            LAYERS[i + 1].phase,
        )
        n_arrows += 1

    # Phase labels on the right
    draw_phase_labels(ax, y_centres)

    # I/O pills
    draw_input_output_pills(ax, y_centres)
    n_arrows += 2  # input pill arrow + output pill arrow

    # Title and subtitle
    ax.text(
        FIG_W / 2,
        FIG_H + 0.25,
        "Molecular Lambda Calculus (MLC) -- 9-Layer Architecture",
        ha="center",
        va="bottom",
        fontsize=15.0,
        fontweight="bold",
        color="#1a1a1a",
    )
    ax.text(
        FIG_W / 2,
        FIG_H + 0.02,
        "parsing  ->  chemistry  ->  search  ->  output gates",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontstyle="italic",
        color="#444444",
    )

    # Phase-band legend (bottom)
    legend_handles = [
        mpatches.Patch(facecolor=PHASE_STYLE["parsing"][0],
                       edgecolor=PHASE_STYLE["parsing"][1],
                       label="Parsing (L1-L3)"),
        mpatches.Patch(facecolor=PHASE_STYLE["chemistry"][0],
                       edgecolor=PHASE_STYLE["chemistry"][1],
                       label="Chemistry (L4-L6)"),
        mpatches.Patch(facecolor=PHASE_STYLE["search"][0],
                       edgecolor=PHASE_STYLE["search"][1],
                       label="Search (L7)"),
        mpatches.Patch(facecolor=PHASE_STYLE["output"][0],
                       edgecolor=PHASE_STYLE["output"][1],
                       label="Output gates (L8-L9)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=4,
        frameon=False,
        fontsize=10.0,
    )

    # Footnote
    ax.text(
        FIG_W / 2,
        -0.95,
        ("Per-layer metric catalogue: "
         "molmetal/reports/lambda_layer_metrics.md  "
         "(L1=5, L2=6, L3=6, L4=7, L5=5, L6=6, L7=5, L8=6, L9=8 metrics)."),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#555555",
        fontstyle="italic",
    )

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    fig.savefig(SVG_PATH, format="svg", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    return n_arrows, PNG_PATH.stat().st_size


def main() -> None:
    n_arrows, size_bytes = render()
    size_kb = round(size_bytes / 1024.0, 1)
    print(f"[fig1] wrote {PNG_PATH.name}  ({size_kb} KB, 600 dpi)")
    print(f"[fig1] wrote {SVG_PATH.name}  "
          f"({round(SVG_PATH.stat().st_size / 1024.0, 1)} KB)")
    print(f"[fig1] n_layers = 9")
    print(f"[fig1] n_arrows = {n_arrows}")


if __name__ == "__main__":
    main()
