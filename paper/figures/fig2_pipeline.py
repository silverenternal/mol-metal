"""Figure 2: End-to-end pipeline diagram.

Renders the full pipeline from "pocket input" through the Lambda MCTS
expansion, parallel validation branches, and reward aggregation.  The
stage inventory is taken from the canonical sweep driver
``molmetal/scripts/r4_c_full_sweep.py`` (the Lambda search orchestrator)
and the multi-channel reward head in
``molmetal/molmetal_lam/search_alg/proof_search.py``
(:class:`RewardAggregator`).

Reproducibility:
    uv run paper/figures/fig2_pipeline.py
Outputs (in this directory):
    fig2_pipeline.png  (600 dpi)
    fig2_pipeline.svg  (scalable)
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
PNG_PATH = OUT_DIR / "fig2_pipeline.png"
SVG_PATH = OUT_DIR / "fig2_pipeline.svg"

# ---------------------------------------------------------------------------
# Phase colour palette
#   input      = blue
#   generation = green
#   validation = orange  (parallel branch)
#   output     = red
# ---------------------------------------------------------------------------
PHASE_STYLE: dict[str, Tuple[str, str]] = {
    "input":      ("#cfe2ff", "#1f4e79"),  # blue
    "generation": ("#d4edda", "#1e6b3a"),  # green
    "validation": ("#ffe5b4", "#a3560b"),  # orange
    "output":     ("#f8d7da", "#8b1a1a"),  # red
}


@dataclass(frozen=True)
class Stage:
    """One pipeline stage.

    ``pos`` is one of "main" (sequential backbone) or "branch"
    (parallel validation branch off the main spine).
    """

    letter: str          # (a), (b), (c), ...
    name: str            # stage title
    brief: str           # one-line description
    phase: str           # "input" | "generation" | "validation" | "output"
    pos: str = "main"    # "main" or "branch"


# Pipeline stages (canonical inventory from r4_c_full_sweep.py +
# proof_search.py's RewardAggregator).
STAGES: list[Stage] = [
    # --- main spine ---------------------------------------------------------
    Stage("(a)", "Pocket input",
          "3D receptor PDB + ligand SMILES", "input"),
    Stage("(b)", "Lambda MCTS expansion",
          "typed-variable substitution + click-rule application + beta-reduction",
          "generation"),
    Stage("(c)", "MetalGeometryPrior check",
          "coordination number + geometry constraint", "generation"),
    Stage("(d)", "RDKit decode",
          "assemble SMILES + sanitise + valence check", "generation"),
    Stage("(i)", "RewardAggregator",
          "7-channel score (qed, sa, binding, novelty, "
          "reinvent4, logp, ring_count)", "output"),
    # --- parallel validation branches --------------------------------------
    Stage("(e)", "QuickVina 2",
          "3D pocket-conditioned binding energy", "validation", "branch"),
    Stage("(f)", "PoseBusters",
          "22 geometric validity rules", "validation", "branch"),
    Stage("(g)", "AdmetAI",
          "drug-likeness + toxicity", "validation", "branch"),
    Stage("(h)", "REINVENT4",
          "r_reinvent4 multiproperty (WF-Extra-2)", "validation", "branch"),
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 14.0, 9.5
MAIN_X0 = 1.2            # left edge of the main spine boxes
MAIN_X1 = 4.6            # right edge of the main spine boxes
MAIN_BOX_W = MAIN_X1 - MAIN_X0
MAIN_BOX_H = 0.78

# Branch band: 4 boxes side-by-side.  Each box gets one quarter of the
# band so the box edges just touch (no overlap, no gap).
BRANCH_X0 = 6.4          # left edge of branch band
BRANCH_X1 = 13.4         # right edge of branch band
BRANCH_BAND_W = BRANCH_X1 - BRANCH_X0
BRANCH_BOX_H = 0.78
N_BRANCH = 4
BRANCH_BOX_W = BRANCH_BAND_W / N_BRANCH

# vertical positions (top-down) for the 5 main-spine stages
MAIN_STAGE_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(i)"]
MAIN_GAP = 0.30
MAIN_TOP = 8.0

# branch stages (left-to-right inside the branch band)
BRANCH_LETTERS = ["(e)", "(f)", "(g)", "(h)"]
BRANCH_Y = 4.6           # vertical centre of the branch row


def main_y_centres() -> dict[str, float]:
    out: dict[str, float] = {}
    for i, letter in enumerate(MAIN_STAGE_LETTERS):
        out[letter] = MAIN_TOP - i * (MAIN_BOX_H + MAIN_GAP)
    return out


def branch_x_centres() -> dict[str, float]:
    """Centre x of each branch box.  Each box gets one slot of width
    BRANCH_BAND_W/N_BRANCH; box centre = slot centre.  This guarantees
    adjacent boxes touch but never overlap."""
    out: dict[str, float] = {}
    for i, letter in enumerate(BRANCH_LETTERS):
        slot_w = BRANCH_BAND_W / N_BRANCH
        slot_x0 = BRANCH_X0 + i * slot_w
        out[letter] = slot_x0 + slot_w / 2.0
    return out


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_box(ax, x0: float, y0: float, w: float, h: float,
              fill: str, edge: str) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, facecolor=fill, edgecolor=edge,
    )
    ax.add_patch(patch)
    return patch


def draw_main_stage(ax, stage: Stage, y_centre: float) -> None:
    fill, edge = PHASE_STYLE[stage.phase]
    y0 = y_centre - MAIN_BOX_H / 2.0
    _draw_box(ax, MAIN_X0, y0, MAIN_BOX_W, MAIN_BOX_H, fill, edge)
    pad_l = 0.18
    ax.text(MAIN_X0 + pad_l, y_centre + 0.18,
            f"{stage.letter}  {stage.name}",
            ha="left", va="center",
            fontsize=11.0, fontweight="bold", color=edge)
    ax.text(MAIN_X0 + pad_l, y_centre - 0.16, stage.brief,
            ha="left", va="center",
            fontsize=8.6, fontstyle="italic", color="#222222")


def draw_branch_stage(ax, stage: Stage, x_centre: float) -> None:
    fill, edge = PHASE_STYLE[stage.phase]
    x0 = x_centre - BRANCH_BOX_W / 2.0
    y0 = BRANCH_Y - BRANCH_BOX_H / 2.0
    _draw_box(ax, x0, y0, BRANCH_BOX_W, BRANCH_BOX_H, fill, edge)
    pad_l = 0.08
    ax.text(x0 + pad_l, BRANCH_Y + 0.18,
            f"{stage.letter} {stage.name}",
            ha="left", va="center",
            fontsize=9.0, fontweight="bold", color=edge)
    # Brief text — wrap manually if it would overflow BRANCH_BOX_W.
    brief = stage.brief
    max_chars = 28  # empirical: fits at fontsize 7.0 within 1.6" wide
    if len(brief) > max_chars:
        # simple two-line wrap on the first space past max_chars // 2
        mid = max_chars // 2
        cut = brief.rfind(" ", 0, max_chars)
        if cut <= mid:
            cut = max_chars
        brief = brief[:cut] + "\n" + brief[cut + 1:]
    ax.text(x0 + pad_l, BRANCH_Y - 0.16, brief,
            ha="left", va="center",
            fontsize=7.0, fontstyle="italic", color="#222222")


def draw_main_arrow(ax, y_top: float, y_bot: float, edge: str) -> None:
    arrow = FancyArrowPatch(
        (MAIN_X0 + MAIN_BOX_W / 2.0, y_top),
        (MAIN_X0 + MAIN_BOX_W / 2.0, y_bot),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=edge,
    )
    ax.add_patch(arrow)


def draw_branch_arrow(ax, x_from: float, x_to: float, y: float,
                      edge: str) -> None:
    arrow = FancyArrowPatch(
        (x_from, y), (x_to, y),
        arrowstyle="-|>", mutation_scale=13,
        linewidth=1.2, color=edge, linestyle="--",
    )
    ax.add_patch(arrow)


def draw_band_label(ax, x0: float, y_top: float, y_bot: float,
                    label: str, edge: str) -> None:
    """Draw a right-side vertical band label (rotated)."""
    label_x = x0 - 0.25
    ax.plot([label_x, label_x], [y_bot, y_top],
            color=edge, linewidth=1.6)
    ax.plot([label_x - 0.06, label_x], [y_top, y_top],
            color=edge, linewidth=1.6)
    ax.plot([label_x - 0.06, label_x], [y_bot, y_bot],
            color=edge, linewidth=1.6)
    mid_y = (y_top + y_bot) / 2.0
    ax.text(label_x - 0.08, mid_y, label, ha="right", va="center",
            fontsize=9.6, fontweight="bold", color=edge, rotation=90)


def draw_phase_legend(ax) -> None:
    handles = [
        mpatches.Patch(facecolor=PHASE_STYLE["input"][0],
                       edgecolor=PHASE_STYLE["input"][1],
                       label="input (pocket + ligand)"),
        mpatches.Patch(facecolor=PHASE_STYLE["generation"][0],
                       edgecolor=PHASE_STYLE["generation"][1],
                       label="generation (Lambda + chemistry)"),
        mpatches.Patch(facecolor=PHASE_STYLE["validation"][0],
                       edgecolor=PHASE_STYLE["validation"][1],
                       label="validation (parallel branches)"),
        mpatches.Patch(facecolor=PHASE_STYLE["output"][0],
                       edgecolor=PHASE_STYLE["output"][1],
                       label="output (reward aggregation)"),
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False,
              fontsize=10.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render() -> Tuple[int, int]:
    """Render the figure. Returns (n_main_stages, n_branches, file_size_png)."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=120)
    ax.set_xlim(-1.4, FIG_W)
    ax.set_ylim(-1.1, FIG_H + 0.6)
    ax.set_aspect("auto")
    ax.axis("off")

    main_y = main_y_centres()
    branch_x = branch_x_centres()

    # ---- main spine boxes + sequential arrows --------------------------
    for letter in MAIN_STAGE_LETTERS:
        stage = next(s for s in STAGES if s.letter == letter)
        draw_main_stage(ax, stage, main_y[letter])

    for i in range(len(MAIN_STAGE_LETTERS) - 1):
        top = main_y[MAIN_STAGE_LETTERS[i]] - MAIN_BOX_H / 2.0
        bot = main_y[MAIN_STAGE_LETTERS[i + 1]] + MAIN_BOX_H / 2.0
        edge = PHASE_STYLE["generation"][1]
        # (d) -> (i) crosses into "output" colour
        if MAIN_STAGE_LETTERS[i] == "(d)":
            edge = PHASE_STYLE["output"][1]
        draw_main_arrow(ax, top, bot, edge)

    # ---- parallel branch boxes ----------------------------------------
    for letter in BRANCH_LETTERS:
        stage = next(s for s in STAGES if s.letter == letter)
        draw_branch_stage(ax, stage, branch_x[letter])

    # ---- branch-to-branch horizontal arrows (left to right) -----------
    # The boxes already touch, so the arrow lives *outside* the boxes,
    # above (or below) them — we draw a small chevron above each
    # inter-box boundary.
    for i in range(len(BRANCH_LETTERS) - 1):
        boundary_x = BRANCH_X0 + (i + 1) * (BRANCH_BAND_W / N_BRANCH)
        ax.add_patch(
            FancyArrowPatch(
                (boundary_x - 0.15, BRANCH_Y + BRANCH_BOX_H / 2.0 + 0.18),
                (boundary_x + 0.15, BRANCH_Y + BRANCH_BOX_H / 2.0 + 0.18),
                arrowstyle="-|>", mutation_scale=12,
                linewidth=1.0, color=PHASE_STYLE["validation"][1],
                linestyle="--",
            )
        )

    # ---- spoke from (d) (RDKit decode) into the branch band -----------
    # (d) is the natural fan-out point: every molecule exiting RDKit
    # decode gets evaluated by the four parallel validation channels.
    # Use a *short* elbow that exits from the right edge of (d) and
    # arrives at the left edge of the branch band.
    x_main_out = MAIN_X0 + MAIN_BOX_W  # right edge of main-spine boxes
    y_d_bottom = main_y["(d)"] - MAIN_BOX_H / 2.0
    # Connect (d) right-edge to branch band left-edge with a right-angle
    # elbow (horizontal segment then vertical drop into BRANCH_Y).
    elbow_x = (x_main_out + BRANCH_X0) / 2.0
    ax.plot([x_main_out, elbow_x], [BRANCH_Y, BRANCH_Y],
            color=PHASE_STYLE["validation"][1],
            linewidth=1.2, linestyle="--")
    ax.plot([elbow_x, elbow_x], [BRANCH_Y, BRANCH_Y],
            color=PHASE_STYLE["validation"][1],
            linewidth=1.2, linestyle="--")
    ax.add_patch(
        FancyArrowPatch(
            (x_main_out, main_y["(d)"]),
            (BRANCH_X0 - 0.05, BRANCH_Y),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.2, color=PHASE_STYLE["validation"][1],
            linestyle="--",
            connectionstyle="angle,angleA=0,angleB=90,rad=0",
        )
    )

    # ---- spoke from the branch band back to the (i) RewardAggregator --
    # All four branch channels feed the aggregator on the main spine.
    # We use a *single* curved spoke from the branch band to the right
    # edge of (i) so the diagram does not get cluttered.
    x_branch_right = BRANCH_X1
    y_i_top = main_y["(i)"] + MAIN_BOX_H / 2.0
    ax.add_patch(
        FancyArrowPatch(
            (x_branch_right, BRANCH_Y),
            (MAIN_X1 + 0.05, y_i_top),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.2, color=PHASE_STYLE["output"][1],
            linestyle="--",
            connectionstyle="arc3,rad=-0.25",
        )
    )

    # ---- phase band labels on the far left of the main spine ----------
    y_top_input = main_y["(a)"] + MAIN_BOX_H / 2.0 + 0.02
    y_bot_input = main_y["(a)"] - MAIN_BOX_H / 2.0 - 0.02
    draw_band_label(ax, MAIN_X0 - 0.15, y_top_input, y_bot_input,
                    "INPUT", PHASE_STYLE["input"][1])
    y_top_gen = main_y["(b)"] + MAIN_BOX_H / 2.0 + 0.02
    y_bot_gen = main_y["(d)"] - MAIN_BOX_H / 2.0 - 0.02
    draw_band_label(ax, MAIN_X0 - 0.15, y_top_gen, y_bot_gen,
                    "GENERATION", PHASE_STYLE["generation"][1])
    y_top_out = main_y["(i)"] + MAIN_BOX_H / 2.0 + 0.02
    y_bot_out = main_y["(i)"] - MAIN_BOX_H / 2.0 - 0.02
    draw_band_label(ax, MAIN_X0 - 0.15, y_top_out, y_bot_out,
                    "OUTPUT", PHASE_STYLE["output"][1])

    # ---- phase band label for the branch row --------------------------
    # Placed *above* the branch band (so it does not collide with the
    # main spine's INPUT/GENERATION/OUTPUT labels).
    ax.text((BRANCH_X0 + BRANCH_X1) / 2.0, BRANCH_Y + BRANCH_BOX_H / 2.0 + 0.35,
            "VALIDATION (parallel branches)",
            ha="center", va="bottom",
            fontsize=10.0, fontweight="bold",
            color=PHASE_STYLE["validation"][1])

    # ---- title + subtitle ---------------------------------------------
    ax.text(FIG_W / 2.0, FIG_H + 0.30,
            "End-to-end pipeline -- pocket-conditioned Lambda search",
            ha="center", va="bottom",
            fontsize=15.0, fontweight="bold", color="#1a1a1a")
    ax.text(FIG_W / 2.0, FIG_H + 0.05,
            "input -> generation (Lambda MCTS + chemistry) -> parallel "
            "validation (Vina / PoseBusters / AdmetAI / REINVENT4) "
            "-> reward aggregation",
            ha="center", va="bottom",
            fontsize=10.0, fontstyle="italic", color="#444444")

    # ---- reward-aggregator channel callout (below main spine) -------
    # Show the seven reward channels aggregated at (i).  Placed below
    # the main spine so it does not collide with the branch band or
    # the branch-to-spine spoke.
    channels_text = (
        "RewardAggregator  --  7 channels\n"
        "  1. qed         (RDKit Descriptors.qed)\n"
        "  2. sa          (RDKit sascorer)\n"
        "  3. binding     (Vina proxy)\n"
        "  4. novelty     (alpha-equiv dedup)\n"
        "  5. reinvent4   (WF-Extra-2 multiproperty)\n"
        "  6. logp        (RDKit Descriptors.MolLogP)\n"
        "  7. ring_count  (RDKit GetRingInfo)"
    )
    callout_x = 0.5
    callout_y_top = 1.7
    callout_y_bot = -0.2
    ax.add_patch(
        FancyBboxPatch(
            (callout_x, callout_y_bot), 4.0, callout_y_top - callout_y_bot,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="#f0f0f0", edgecolor="#444444", linewidth=1.0,
        )
    )
    ax.text(callout_x + 0.14, callout_y_top - 0.22, channels_text,
            ha="left", va="top", fontsize=7.6, color="#222222",
            family="monospace")

    # ---- legend --------------------------------------------------------
    draw_phase_legend(ax)

    # ---- footnote ------------------------------------------------------
    ax.text(FIG_W / 2.0, -0.75,
            ("Stage inventory: "
             "molmetal/scripts/r4_c_full_sweep.py (orchestrator) + "
             "molmetal/molmetal_lam/search_alg/proof_search.py "
             "(:class:`RewardAggregator`)."),
            ha="center", va="bottom",
            fontsize=8.2, color="#555555", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    fig.savefig(SVG_PATH, format="svg", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    return len(MAIN_STAGE_LETTERS), len(BRANCH_LETTERS), PNG_PATH.stat().st_size


def main() -> None:
    n_main, n_branch, size_bytes = render()
    size_kb = round(size_bytes / 1024.0, 1)
    print(f"[fig2] wrote {PNG_PATH.name}  ({size_kb} KB, 600 dpi)")
    print(f"[fig2] wrote {SVG_PATH.name}  "
          f"({round(SVG_PATH.stat().st_size / 1024.0, 1)} KB)")
    print(f"[fig2] n_main_stages       = {n_main}")
    print(f"[fig2] n_parallel_branches = {n_branch}")
    print(f"[fig2] n_total_stages      = {len(STAGES)}")


if __name__ == "__main__":
    main()
