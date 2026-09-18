# 3D embedding sanity (Ru 78%, Ir 71%, Pt 84%)

**Status:** completed
**Completed:** 2026-09-11
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/b1_3d_embed_sanity.md
**Owner:** (unset)

## What was delivered
Per-metal 3D-coordinate embed success-rate sweep across the three metals that
drive the lambda_clickchem reward signal. For each metal: of N hand-curated
reference geometries, how often does the embedder produce a structure within
0.3 Å RMSD of the reference after equivariance alignment.

## Hard numbers (b1_3d_embed_sanity.md)
- **Pt: 84%** pass rate (best — d8 square-planar is easiest)
- **Ru: 78%** pass rate
- **Ir: 71%** pass rate (worst — mix of 4-/5-/6-coordinate motifs in the test set)
- Aggregate across the three: ~78%
- All three pass the ≥70% gate that the milestone plan requires before
  downstream optimisation is allowed to consume the embedder

## Lessons learned
- Failure mode documented: **square-planar Pt(II) needs UFF fallback when
  MMFF94s rejects** the standard embedding.
- The Ir gap is driven by ambiguous coordination number; combining the
  embedder with the chemprop CN regressor (TODO/completed/05_tmqm_pretraining.md)
  closes ~half the gap.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/b1_3d_embed_sanity.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/b1_3d_embed_ru.json
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/b1_3d_embed_ir.json
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/b1_3d_embed_pt.json