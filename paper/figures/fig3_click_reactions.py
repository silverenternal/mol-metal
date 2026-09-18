"""Figure 3: The 5 click-chemistry reaction rules.

Renders the canonical click set registered in
``molmetal/molmetal_lam/lam_chem/rules.py`` (re-exporting the
:class:`ReactionRule` singletons from
``molmetal/molmetal_lam/reactions/beta_reductions.py``) as a 2x3 grid of
reaction panels — each panel shows the reaction name, SMARTS template,
reactant SMILES, a curved mechanism arrow, and the canonical product
SMILES drawn with RDKit.

The chemistry shown follows the standard textbook presentations of each
reaction class (Sharpless 2001 for CuAAC; Wittig & Krebs / Agard et al.
2004 for SPAAC; Hoyle & Bowman for thiol-ene; Miyaura & Suzuki 1995 for
Suzuki-Miyaura; Montalbetti & Falque 2005 for amide coupling).  The
example SMILES are *illustrative* — chosen to be RDKit-parseable and
small enough to fit a panel — not synthesised by the SMARTS templates
(some template products, e.g. SPAAC, currently trip RDKit's kekule
sanitizer and are drawn here in their chemically expected form).

Reproducibility:
    uv run paper/figures/fig3_click_reactions.py
Outputs (in this directory):
    fig3_click_reactions.png  (600 dpi)
    fig3_click_reactions.svg  (scalable)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "fig3_click_reactions.png"
SVG_PATH = OUT_DIR / "fig3_click_reactions.svg"

# ---------------------------------------------------------------------------
# Per-reaction colour palette
# ---------------------------------------------------------------------------

# fill, edge — CuAAC orange, SPAAC orange (strained variant),
# thiol-ene green, Suzuki blue, amide red.  The two orange variants use
# slightly different shades so SPAAC is visually distinct from CuAAC
# despite sharing the azide-alkyne cycloaddition chemistry.
PALETTE: dict[str, Tuple[str, str]] = {
    "CuAAC":         ("#fde0c5", "#a3560b"),  # light orange / dark orange
    "SPAAC":         ("#ffe5b4", "#cc6f0b"),  # peach / amber
    "ThiolEne":      ("#d4edda", "#1e6b3a"),  # light green / dark green
    "Suzuki":        ("#cfe2ff", "#1f4e79"),  # light blue / dark blue
    "AmideCoupling": ("#f8d7da", "#8b1a1a"),  # light red / dark red
}

# ---------------------------------------------------------------------------
# Reaction specification
#
# Each entry bundles:
#   - name        : reaction name (matches REACTION_RULES keys)
#   - smarts      : the SMARTS template (verbatim from beta_reductions.py)
#   - reactants   : SMILES pair (literature-typical example)
#   - product     : SMILES of the expected product
#   - catalyst    : string label rendered on the curved arrow
#   - chemistry   : one-line chemistry description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClickRxn:
    name: str
    smarts: str
    reactants: Tuple[str, str]
    product: str
    catalyst: str
    chemistry: str


# SMARTS strings verbatim from
# molmetal/molmetal_lam/reactions/beta_reductions.py (lines 563, 627,
# 817, 965, 1033).
CLICK_RXNS: List[ClickRxn] = [
    ClickRxn(
        name="CuAAC",
        smarts=("[N:1]=[N:2]=[N:3].[C:4]#[CH:5]>>"
               "[C:4]1=[C:5][N:3]=[N:2][N:1]1"),
        reactants=("[N-]=[N+]=NCc1ccccc1", "C#CC"),
        product="CC1=CN=[N+](Cc2ccccc2)[N-]1",
        catalyst="Cu(I)",
        chemistry="azide + terminal alkyne -> 1,4-disubstituted 1,2,3-triazole",
    ),
    ClickRxn(
        name="SPAAC",
        smarts=("[N:1]=[N:2]=[N:3].[C:4]#[C:5]>>"
               "[C:4]1=[C:5][N:3]=[N:2][N:1]1"),
        reactants=("[N-]=[N+]=NCc1ccccc1", "C1CCC#CCCCC1"),
        product="c1ccc(C[N]2N=NC=C2C3CCCCCCC3)cc1",
        catalyst="strain (no Cu)",
        chemistry="azide + strained cyclooctyne -> triazole (no metal catalyst)",
    ),
    ClickRxn(
        name="ThiolEne",
        smarts="None  (functional-group rewrite)",
        reactants=("CCS", "C=CC"),
        product="CCSCC",
        catalyst="hν / radical",
        chemistry="thiol + alkene -> thioether (anti-Markovnikov, photoinitiated)",
    ),
    ClickRxn(
        name="Suzuki",
        smarts=("[#6:1][B]([O])[O].[#6:3][F,Cl,Br,I]>>"
               "[#6:1][#6:3]"),
        reactants=("OB(O)c1ccccc1", "Brc1ccccc1"),
        product="c1ccc(-c2ccccc2)cc1",
        catalyst="Pd(0) + base",
        chemistry="aryl-boronic acid + aryl halide -> biaryl",
    ),
    ClickRxn(
        name="AmideCoupling",
        smarts=("[C:1](=[O:2])[OH].[NH2:4]>>"
               "[C:1](=[O:2])[NH:4]"),
        reactants=("CC(=O)O", "CCN"),
        product="CCNC(C)=O",
        catalyst="EDC/HOBt or HATU",
        chemistry="carboxylic acid + primary amine -> amide (-H2O)",
    ),
]


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 16.0, 12.0
N_ROWS, N_COLS = 2, 3  # 5 reactions + 1 empty cell

# A single cell occupies a sub-axes region inside the figure; molecule
# drawings are placed via inset_axes.
CELL_W = 5.0
CELL_H = 4.4
CELL_X0 = 0.35
CELL_Y0 = 1.85  # leave room at top for title + subtitle, bottom for legend

# RDKit molecule image size (pixels)
MOL_IMG_W = 380
MOL_IMG_H = 240


# ---------------------------------------------------------------------------
# RDKit drawing helper
# ---------------------------------------------------------------------------


def render_mol_png(smiles: str, size_px: Tuple[int, int] = (MOL_IMG_W, MOL_IMG_H)
                   ) -> "matplotlib.image.AxesImage | None":
    """Render a single SMILES to an AxesImage via rdMolDraw2D.

    Returns ``None`` if RDKit fails to parse — caller skips the inset.
    """
    from matplotlib.image import imread
    from io import BytesIO

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(size_px[0], size_px[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1.6
    opts.padding = 0.06
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    image = imread(BytesIO(png_bytes), format="png")
    return image


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------


def draw_panel(ax, rxn: ClickRxn, x0: float, y0: float, w: float, h: float) -> None:
    """Draw a single click-reaction panel inside the axes ``ax``."""
    fill, edge = PALETTE[rxn.name]

    # ---- Panel background card ----
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.6,
            facecolor=fill,
            edgecolor=edge,
            alpha=0.30,
        )
    )

    # ---- Title bar (reaction name + catalyst pill) ----
    title_h = 0.45
    ax.add_patch(
        FancyBboxPatch(
            (x0 + 0.05, y0 + h - title_h - 0.05),
            w - 0.10,
            title_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.2,
            facecolor=edge,
            edgecolor=edge,
        )
    )
    ax.text(
        x0 + 0.20,
        y0 + h - title_h / 2 - 0.05,
        rxn.name,
        ha="left",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x0 + w - 0.20,
        y0 + h - title_h / 2 - 0.05,
        f"cat: {rxn.catalyst}",
        ha="right",
        va="center",
        fontsize=9.0,
        fontstyle="italic",
        color="white",
    )

    # ---- Chemistry one-liner ----
    ax.text(
        x0 + w / 2,
        y0 + h - title_h - 0.22,
        rxn.chemistry,
        ha="center",
        va="center",
        fontsize=8.4,
        color="#222222",
        fontstyle="italic",
    )

    # ---- Molecule row: reactant_a  +  reactant_b  -> product ----
    mol_y_top = y0 + h - title_h - 0.78          # top of molecule row
    mol_band_h = 2.55                            # vertical space for images
    mol_y_bot = mol_y_top - mol_band_h

    # Layout: 3 columns inside the panel
    inner_x0 = x0 + 0.18
    inner_w = w - 0.36
    col_w = inner_w / 3.0
    col_centres = [inner_x0 + col_w * (i + 0.5) for i in range(3)]

    # Reactant A
    img_a = render_mol_png(rxn.reactants[0])
    if img_a is not None:
        ax.imshow(
            img_a,
            aspect="equal",
            extent=(
                col_centres[0] - col_w * 0.40,
                col_centres[0] + col_w * 0.40,
                mol_y_bot,
                mol_y_top,
            ),
            zorder=2,
        )
    ax.text(
        col_centres[0],
        mol_y_bot - 0.18,
        rxn.reactants[0],
        ha="center",
        va="top",
        fontsize=7.4,
        family="monospace",
        color="#222222",
    )
    ax.text(
        col_centres[0],
        mol_y_top + 0.05,
        "reactant A",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=edge,
    )

    # Reactant B
    img_b = render_mol_png(rxn.reactants[1])
    if img_b is not None:
        ax.imshow(
            img_b,
            aspect="equal",
            extent=(
                col_centres[1] - col_w * 0.40,
                col_centres[1] + col_w * 0.40,
                mol_y_bot,
                mol_y_top,
            ),
            zorder=2,
        )
    ax.text(
        col_centres[1],
        mol_y_bot - 0.18,
        rxn.reactants[1],
        ha="center",
        va="top",
        fontsize=7.4,
        family="monospace",
        color="#222222",
    )
    ax.text(
        col_centres[1],
        mol_y_top + 0.05,
        "reactant B",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=edge,
    )

    # Curved arrows from A -> B and B -> product
    # Arrow 1: reactant A -> reactant B (above the molecules)
    arrow_y = (mol_y_top + mol_y_bot) / 2 + 0.55
    ax.add_patch(
        FancyArrowPatch(
            (col_centres[0] + col_w * 0.42, arrow_y),
            (col_centres[1] - col_w * 0.42, arrow_y),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.6,
            color="#444444",
            connectionstyle="arc3,rad=-0.18",
        )
    )
    # Plus sign between reactants
    ax.text(
        (col_centres[0] + col_centres[1]) / 2,
        arrow_y + 0.30,
        "+",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#444444",
    )

    # Arrow 2: reactants -> product (below the molecules)
    arrow_y2 = (mol_y_top + mol_y_bot) / 2 - 0.65
    ax.add_patch(
        FancyArrowPatch(
            ((col_centres[0] + col_centres[1]) / 2 + col_w * 0.10, arrow_y2 + 0.20),
            (col_centres[2] - col_w * 0.42, arrow_y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.6,
            color=edge,
            connectionstyle="arc3,rad=0.18",
        )
    )

    # Product
    img_p = render_mol_png(rxn.product)
    if img_p is not None:
        ax.imshow(
            img_p,
            aspect="equal",
            extent=(
                col_centres[2] - col_w * 0.40,
                col_centres[2] + col_w * 0.40,
                mol_y_bot,
                mol_y_top,
            ),
            zorder=2,
        )
    ax.text(
        col_centres[2],
        mol_y_bot - 0.18,
        rxn.product,
        ha="center",
        va="top",
        fontsize=7.4,
        family="monospace",
        color="#222222",
    )
    ax.text(
        col_centres[2],
        mol_y_top + 0.05,
        "product",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=edge,
    )

    # ---- SMARTS template (monospace, very bottom) ----
    smarts_y = y0 + 0.10
    ax.text(
        x0 + w / 2,
        smarts_y,
        f"SMARTS: {rxn.smarts}",
        ha="center",
        va="bottom",
        fontsize=7.4,
        family="monospace",
        color="#444444",
        wrap=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render() -> Tuple[int, int, bool]:
    """Render the figure. Returns (n_panels, file_size_bytes_png, all_valid)."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=120)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H + 1.0)
    ax.set_aspect("auto")
    ax.axis("off")

    n_panels = len(CLICK_RXNS)
    all_valid = True

    for idx, rxn in enumerate(CLICK_RXNS):
        row = idx // N_COLS
        col = idx % N_COLS
        x0 = CELL_X0 + col * (CELL_W + 0.25)
        # row 0 (top) is the first row in CLICK_RXNS
        y0 = CELL_Y0 + (N_ROWS - 1 - row) * (CELL_H + 0.40)

        # Validate SMILES
        for s in (rxn.reactants[0], rxn.reactants[1], rxn.product):
            if Chem.MolFromSmiles(s) is None:
                all_valid = False
        draw_panel(ax, rxn, x0, y0, CELL_W, CELL_H)

    # ---- Figure title + subtitle ----
    ax.text(
        FIG_W / 2,
        FIG_H + 0.55,
        "The 5 Click-Chemistry Reactions in the MLC Rule Set",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#1a1a1a",
    )
    ax.text(
        FIG_W / 2,
        FIG_H + 0.10,
        ("Curated click set registered in molmetal/molmetal_lam/lam_chem/rules.py "
         "(axis A, round-10 click family)."),
        ha="center",
        va="top",
        fontsize=10.5,
        fontstyle="italic",
        color="#444444",
    )

    # ---- Legend (colour key) at the bottom ----
    legend_handles = [
        mpatches.Patch(facecolor=PALETTE[k][0], edgecolor=PALETTE[k][1], label=k)
        for k in ("CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling")
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        fontsize=10.5,
    )

    # ---- Footnote ----
    ax.text(
        FIG_W / 2,
        0.05,
        ("SMARTS templates verbatim from molmetal/molmetal_lam/reactions/"
         "beta_reductions.py (lines 563/627/817/965/1033). "
         "Illustrative example SMILES, not synthesised by the rule engine."),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#555555",
        fontstyle="italic",
    )

    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    fig.savefig(SVG_PATH, format="svg", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    return n_panels, PNG_PATH.stat().st_size, all_valid


def main() -> None:
    n_panels, size_bytes, all_valid = render()
    size_kb = round(size_bytes / 1024.0, 1)
    print(f"[fig3] wrote {PNG_PATH.name}  ({size_kb} KB, 600 dpi)")
    print(f"[fig3] wrote {SVG_PATH.name}  "
          f"({round(SVG_PATH.stat().st_size / 1024.0, 1)} KB)")
    print(f"[fig3] n_reactions = {n_panels}")
    print(f"[fig3] all_smiles_valid = {all_valid}")


if __name__ == "__main__":
    main()
