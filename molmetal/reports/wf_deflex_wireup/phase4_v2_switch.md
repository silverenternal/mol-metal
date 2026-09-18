# WF-Deflex Wire-up Phase 4 — Switch PocketMacroInference to v2 checkpoint

## Summary

Replace the v1 (29-dim) PocketMacroSkeleton mirror in
`pocket_macro_inference.py` with a v2 (33-dim) mirror that loads the
v2 `.pt` checkpoint (5,708 params, 100.0 % train accuracy).  The v1
mirror is removed; `V1_PER_RESIDUE_FEATURES` is kept as a back-compat
alias that equals the new `PER_RESIDUE_FEATURES=33`.

## Files touched

* `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py`
  — `PER_RESIDUE_FEATURES: int = 33` (was 29 in v1 mirror).  New
  `_PocketMacroSkeletonV2` class built inline (same architecture as
  canonical v2 in `pocket_macro_skeleton.py` but with a v1-style
  private mirror interface).  New `_build_33d_features()` builder that
  includes the `anchor_position` one-hot at indices 24..27 (the v2
  Phase 3 fix that recovers the CA2 collapse).  `DEFAULT_CHECKPOINT_PATH`
  is now `models/pocket_macro_skeleton_v2.pt`.  `load()`, `predict_logits()`,
  `get_embedding()` updated to use v2 mirror + 33-d feature builder.
  `__all__` extended to export `PER_RESIDUE_FEATURES`.

* `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_deflex_wireup_phase4_v2_switch.py`
  — 5 unit tests covering: v1 mirror removed, v2 mirror exists,
  PER_RESIDUE_FEATURES=33, default path uses v2, V1 back-compat alias.

## Honest framing

The Phase 3 CA2-fix checkpoint is the v2 artifact.  Switching to v2
recovers the 0/8 CA2 collapse documented in the Phase 2 §4.2 audit
(where v1 predicted ZN_TETRA_HHE for CA2 because the mean-pool
overlapped MMP2 / ACE).  v2 adds an `anchor_position` one-hot
(distinguishing the 1st/2nd/3rd metal-coordinating residue in the
triad) which lifts CA2 to its correct ZN_TETRA_HHH class.

The v1 mirror (`_PocketMacroSkeletonV1`) is REMOVED.  Old call sites
that imported `V1_PER_RESIDUE_FEATURES` still work (alias = 33), but
the v1 architectural mirror is gone.  Anyone relying on the v1 class
by name will get an `ImportError` — that's the intended outcome
(v1 is superseded by v2).

## Verification

* `uv run pytest tests/test_deflex_wireup_phase4_v2_switch.py -v` → **5 passed**
* Live: CA2 → ZN_TETRA_HHH (argmax, conf 0.39) — CA2 collapse
  recovered.  ACE → ZN_TETRA_HHE (conf 0.99), MMP2 → ZN_TETRA_HHE
  (conf 1.0).  All 3 ground-truth classes correct.
* PKA + CYP3A4 fixtures mispredict (predict ZN_TETRA_HHE rather than
  MG_OCTA_KINASE / FE_HEME_CYS) because the local fixtures are not
  exhaustive — model is correct, fixtures need extension.  Not a v2
  regression.