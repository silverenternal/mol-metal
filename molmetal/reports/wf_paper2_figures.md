# WF-Paper-2 Figures Verification Report

**Date:** 2026-09-14
**Workflow:** WF-Paper-2 (3 core paper figures)
**Hardware/OS:** ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64 (uv-managed Python 3.12)
**Status:** SHIPPED — all 3 figures reproduce cleanly, INDEX.md + report written.

---

## 1. File inventory (paper/figures/)

| File | Type | Bytes | KB |
|------|------|------:|----:|
| fig1_mlc_architecture.py | source | 14,312 | 14.0 |
| fig1_mlc_architecture.svg | vector | 214,550 | 209.5 |
| fig1_mlc_architecture.png | raster 600 dpi | 1,380,259 | 1348.0 |
| fig1_caption.md | caption | 2,018 | 2.0 |
| fig2_pipeline.py | source | 17,692 | 17.3 |
| fig2_pipeline.svg | vector | 191,663 | 187.2 |
| fig2_pipeline.png | raster 600 dpi | 1,119,355 | 1093.1 |
| fig2_caption.md | caption | 3,419 | 3.3 |
| fig3_click_reactions.py | source | 15,727 | 15.4 |
| fig3_click_reactions.svg | vector | 288,017 | 281.3 |
| fig3_click_reactions.png | raster 600 dpi | 1,722,580 | 1682.2 |
| fig3_caption.md | caption | 3,019 | 2.9 |
| INDEX.md | index | (new) | <1 |

PNG dimensions (from `file`):
- fig1: 7164 x 7530 px @ 600 dpi
- fig2: 8364 x 5710 px @ 600 dpi
- fig3: 7028 x 5755 px @ 600 dpi

## 2. Reproducibility confirmation

Each `.py` was invoked via `uv run python figN_*.py` from `/home/hugo/codes/try_triton_on_rocm/paper/figures/`.
All three ran cleanly with no warnings, exited 0, and re-emitted both PNG (600 dpi) and SVG.

### fig1_mlc_architecture.py
- wrote fig1_mlc_architecture.png (1347.9 KB, 600 dpi)
- wrote fig1_mlc_architecture.svg  (209.5 KB)
- n_layers = 9, n_arrows = 10

### fig2_pipeline.py
- wrote fig2_pipeline.png (1093.1 KB, 600 dpi)
- wrote fig2_pipeline.svg  (187.2 KB)
- n_main_stages = 5, n_parallel_branches = 4, n_total_stages = 9

### fig3_click_reactions.py
- wrote fig3_click_reactions.png (1682.2 KB, 600 dpi)
- wrote fig3_click_reactions.svg  (281.3 KB)
- n_reactions = 5, all_smiles_valid = True

All scripts are self-contained: deterministic layout, fixed palette, no `pyplot.show()`, no random seed required.
SVG files begin with the standard `<?xml version="1.0" ...?>` + DOCTYPE header (valid XML).
PNG files have valid PNG magic and the expected RGBA non-interlaced header.

## 3. Honest-framing check

Each caption is paired with its figure and uses MEASURED / PROJECTED language consistently:
- fig1: architectural diagram (no measurements) — labelled as "schematic, not to scale".
- fig2: pipeline diagram with reward-component labels; reward-channel arrows annotate which signals are MEASURED vs PROXY/PROJECTED.
- fig3: 5 click-reaction templates (CuAAC + variants) with M-CLICK viability annotations marking which reaction families are MEASURED (validated against retrosynthetic literature) vs PROPOSED (in-silico extrapolation).

## 4. Existence check (required artefacts)

All 12 required files present and non-empty:
- [x] paper/figures/fig1_mlc_architecture.{png,svg,py}
- [x] paper/figures/fig2_pipeline.{png,svg,py}
- [x] paper/figures/fig3_click_reactions.{png,svg,py}
- [x] paper/figures/{fig1,fig2,fig3}_caption.md
- [x] paper/figures/INDEX.md (newly written)

## 5. Reproduction command (one-liner)

```bash
cd /home/hugo/codes/try_triton_on_rocm/paper/figures && \
  uv run python fig1_mlc_architecture.py && \
  uv run python fig2_pipeline.py && \
  uv run python fig3_click_reactions.py
```

All three exit 0 in this exact sequence on the current environment.

## 6. Metrics

- n_figures: 3
- total_size_kb (png+svg+py+md+INDEX): ~4955
- all_reproducible: true
