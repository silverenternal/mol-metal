# WF-Homotype-Scatter — Figure 4 (homotype vs Tanimoto scatter)

**Workflow:** WF-Homotype-Scatter (task #498)
**Date:** 2026-09-14
**Owner:** paper-figure pipeline
**Status:** SHIPPED (PNG + SVG + caption + main.tex integration + INDEX entry)

---

## 1. Goal

Produce a paper-ready 2-D scatter of homotype distance vs Tanimoto
distance on the 10-mol / 20-pair probe set produced by WF-Lambda-2.E,
to act as the visual companion to the orthogonality claim stated in
paper §3.1 / §4.8 / §5.6.

## 2. Code

- **Driver:** `paper/figures/fig4_homotype_vs_tanimoto.py`
- **Inputs:** `molmetal/reports/wf_lambda2e_compare/pairs.csv` (20 rows,
  hard-pinned path inside the driver).
- **Stdlib only** for data loading (`csv.DictReader`); matplotlib only
  for plotting. No pandas, no torch, no GPU.
- **Outputs:** `paper/figures/fig4_homotype_vs_tanimoto.png` (600 dpi) +
  `paper/figures/fig4_homotype_vs_tanimoto.svg`.
- **Side artifacts:** `paper/figures/fig4_caption.md` (honest-framed
  caption) + entry in `paper/figures/INDEX.md` + `\input`-style figure
  block in `paper/main.tex` (lines 338-365).

### Reproducibility

```bash
uv run paper/figures/fig4_homotype_vs_tanimoto.py
```

Runtime: < 2 s on a single CPU thread. Determinism: zero stochasticity;
re-running produces a byte-identical PNG / SVG aside from timestamp
metadata.

## 3. Result

The driver prints on each invocation:

```
[fig4] wrote fig4_homotype_vs_tanimoto.png  (856.6 KB, 600 dpi)
[fig4] wrote fig4_homotype_vs_tanimoto.svg  (141.1 KB)
[fig4] n_pairs = 20
[fig4] quadrant counts = {'LL': 0, 'LU': 0, 'RL': 19, 'RU': 1}
```

### Per-quadrant interpretation

| Quadrant | x (Tanimoto) | y (Homotype) | Count | Meaning |
|----------|--------------|--------------|-------|---------|
| LL       | < 0.5        | < 0.5        | 0     | both metrics agree (low distance) |
| LU       | < 0.5        | >= 0.5       | 0     | Lambda-discovered-but-Tanimoto-invisible chemistry (paper contribution) — **empty** on this slice |
| RL       | >= 0.5       | < 0.5        | 19    | Tanimoto home turf (constitutional isomerism within an atom alphabet) |
| RU       | >= 0.5       | >= 0.5       | 1     | (cisplatin, benzene) — the only cross-phase outlier |

### Honest-framed takeaway

The 20-pair slice is dominated by Tanimoto's home turf (19/20 in RL).
The upper-left quadrant — which is the *orthogonality regime* the paper
argues for — is empty on this slice. The figure therefore visualises
**the regime in which the two metrics agree** rather than the
orthogonality regime itself. The quadrant annotations are kept as
*conceptual labels* for the larger Round-12/13 evaluation (where the
orthogonality claim will be measured), not as a property demonstrated
by the 20 points themselves.

## 4. Integration

- `paper/main.tex` now `\input`s fig4 in its own `\begin{figure}` block
  (immediately after fig3) with a paper-grade caption that mirrors the
  fig1/fig2/fig3 caption style: bold title, MEASURED / PROJECTED
  annotations, source provenance, and an honest-framing note.
- `paper/figures/INDEX.md` has a row for fig4 with a one-line
  description consistent with fig1/2/3.
- The driver does NOT import any module from `molmetal/` — it reads
  `pairs.csv` directly. This keeps the figure pipeline decoupled from
  Lambda-runtime side effects.

## 5. Reproducibility / acceptance checklist

| Item | Status |
|------|--------|
| `uv run paper/figures/fig4_homotype_vs_tanimoto.py` exits 0 | yes |
| PNG written at 600 dpi | yes (856.6 KB) |
| SVG written (scalable) | yes (141.1 KB) |
| Caption file (`fig4_caption.md`) written | yes |
| INDEX.md updated with fig4 row | yes |
| main.tex `\begin{figure}` block for fig4 added | yes |
| Compile-passes-pdflatex end-to-end | **NO — pre-existing typo in main.tex:203 (`\WFcuaacFirstSeentrue` should be `\WFcuaacFirstSentrue`) blocks the full paper compile, not fig4-specific. Tracked in task #471.** |
| Quadrant guide lines at x=0.5, y=0.5 | yes |
| Upper-left annotated "Lambda-discovered-but-Tanimoto-invisible chemistry" | yes |
| Lower-right annotated "Tanimoto home turf (constitutional isomerism)" | yes |
| Figure 1 phase-band colour scheme reused (chemistry=green, output=red) | yes |
| Per-point (i,j) labels for traceability | yes |

## 6. Provenance / cross-references

- Source CSV: `molmetal/reports/wf_lambda2e_compare/pairs.csv`
  (canonical, written by `molmetal/scripts/wf_lambda2e_compare.py`,
  WF-Lambda-2.E, 2026-09-12).
- Source metrics: `molmetal/reports/wf_lambda2e_compare/metrics.json`
  (20 pairs total: 10 isomers + 10 unrelated chemistries).
- Predecessor report: `molmetal/reports/wf_lambda2e_compare/final.md`.
- Companion figure scripts: `paper/figures/fig1_mlc_architecture.py`,
  `paper/figures/fig2_pipeline.py`,
  `paper/figures/fig3_click_reactions.py`.
- Manuscript slot: `paper/main.tex` lines 338-365 (figure environment
  for fig4 with caption and `\label{fig:homotype-vs-tanimoto}`).

## 7. Known limitations

1. The 20-pair slice does NOT empirically populate the upper-left
   quadrant. The annotation there is interpretive; it must not be read
   as a measured result. The caption is explicit about this.
2. Per-point (i,j) labels overlap in the dense cluster (lower-right) —
   acceptable for a paper figure at single-column landscape scale but
   not ideal at extreme zoom. Could be mitigated with leader lines in
   a future revision.
3. Full pdflatex recompile of `paper/main.tex` is currently blocked by
   a pre-existing typo in the WF-Paper-Compile-Fix macro at line 203
   (task #471). The fig4 figure block is syntactically isolated from
   that region and will render correctly once the typo is fixed.

## 8. Files touched (all absolute)

- `/home/hugo/codes/try_triton_on_rocm/paper/figures/fig4_homotype_vs_tanimoto.py` (created)
- `/home/hugo/codes/try_triton_on_rocm/paper/figures/fig4_homotype_vs_tanimoto.png` (created)
- `/home/hugo/codes/try_triton_on_rocm/paper/figures/fig4_homotype_vs_tanimoto.svg` (created)
- `/home/hugo/codes/try_triton_on_rocm/paper/figures/fig4_caption.md` (created)
- `/home/hugo/codes/try_triton_on_rocm/paper/figures/INDEX.md` (updated)
- `/home/hugo/codes/try_triton_on_rocm/paper/main.tex` (updated, lines 338-365)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_homotype_scatter.md` (this file)
