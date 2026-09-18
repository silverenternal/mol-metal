# Square-planar Pt(II) geometric prior + dative-bond EGNN edge type

**Status:** pending
**Priority:** high
**Effort:** 3d
**Owner:** (unset)
**Depends on:** 08 (tmQM wireup — provides the bond-length / bond-angle priors)
**Blockers:** none
**Created:** 2026-09-12

## Goal
Add a new EGNN edge type for dative (coordinate) bonds and a geometric prior
that biases square-planar geometry around d8 Pt(II) centres and
tetrahedral/octahedral around other common transition metals. This is the
algorithmic axis that lets the sampler produce chemically valid metal
complexes without external rule engines. Closes the
`audit_lambda_upper_bound.md` finding that the sampler has "no geometric
prior for square-planar Pt(II) and no dative-bond edge type in EGNN".

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/adapters/egnn_rocm.py  (edge type table + message-passing)
- /home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/egnn_velocity.py  (apply geometric prior in sampling)
- new: /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py  (the 90°-angle constraint for Pt(II))

## Success criterion
On the CSD-derived Pt(II) test set, generated samples reproduce the
square-planar geometry in ≥85% of cases (RMSD < 0.4 Å on the 4
coordinating atoms); dative bond order inferred correctly ≥90% of the
time vs RDKit's heuristic; cisplatin-as-lambda-term re-run shows
predicted Pt-N bond lengths within 0.05 Å of crystal (2.01 Å).

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/audit_lambda_upper_bound.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/b1_3d_embed_sanity.md  (failure mode doc)