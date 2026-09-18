# Paper Figures Index

Three core figures for the WF-Paper-2 manuscript, each rendered at 600 dpi PNG + scalable SVG, with a reproducible Python source.

| ID | File | One-line description |
|----|------|----------------------|
| 1 | `fig1_mlc_architecture.{png,svg,py}` | 9-layer Metal-Ligand Constructor (MLC) architecture: typed-variable input, edge-wise bond head, CFM refinement, metal-aware constraints. |
| 2 | `fig2_pipeline.{png,svg,py}` | End-to-end Lambda-as-generator pipeline: MCTS propose -> RDKit/CFM validate -> QVina dock -> RewardAggregator -> pIC50 oracle (5 main stages + 4 parallel branches). |
| 3 | `fig3_click_reactions.{png,svg,py}` | Five CuAAC click-reaction templates with metal-fragment handles and CLICK viability annotations for hybrid ligand assembly. |
| 4 | `fig4_homotype_vs_tanimoto.{png,svg,py}` | 20-pair scatter of (homotype, Tanimoto) distances from WF-Lambda-2.E with quadrant guide lines + 4 phase-banded annotations (upper-left = Lambda-discovered-but-Tanimoto-invisible chemistry, lower-right = Tanimoto home turf). |

All captions live next to the figures as `figN_caption.md` (paper-grade, MEASURED vs PROJECTED clearly labelled).
Every `.py` file is self-contained: `uv run python figN_*.py` regenerates the paired `.svg` and `.png` from canonical data — no hidden state, no random colour injection.
