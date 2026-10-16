# chemprop_v2 vs dmpnn_v2 schema comparison decision rule

## Background

The chemprop_v2 atom-features schema is **72-dim** (`11 + 11 + 11 + 4 + 8 + 2 + 9 + 5 + 2 + 4 + 5`: symbol/degree/charge one-hots + chirality + num-H + hybridisation + aromaticity + atomic-mass bucket + ring-membership + ring-size + dative-flag); the dmpnn v1 schema is a different 39-dim atom-property slice. The two diverge at dim 0, so there is no first-39-dim overlap.

## Decision rule

Cross-schema comparisons (chemprop_v2 vs dmpnn_v2, or any future cross-schema pair) MUST use **shape-only** assertions, NEVER byte-equality. Shape `(V, 72)` is well-defined; byte-equality is structurally impossible since the schemas diverge at dim 0.

## Code-level lock

`molmetal/tests/test_chemprop_v2_schema_shape_only.py` (3 tests, NEW Wf-A54, 3/3 PASS on HEAD):
(1) `test_shape_only_is_current_correct_compare_mode` — confirms `(V, 72)` is the canonical assertion shape;
(2) `test_byte_equality_attempt_raises_not_implemented` — proves a byte-equality attempt raises `NotImplementedError` with the §6.79b ledger pointer so the wrong-comparison class is self-announcing;
(3) `test_chemprop_v2_mt_model_atom_dim_contract_is_72` — verifies `CHEMPROP_V2_ATOM_DIM: int = 72` is exported.

## Cross-references

- §6.79b catalog entry: `docs/HONEST_NEGATIVE_CATALOG.md:2773-2782` (decision rule + lock).
- `CHEMPROP_V2_ATOM_DIM: int = 72` source: `molmetal/scripts/wf331r5_r2_w2_chemprop.py:249`.
- Re-export pattern: `molmetal/molmetal_lam/reward/chemprop_pll_integration.py:115`.

## Paper §3.4 footnote pointer

When reporting cross-schema comparison, MUST footnote the shape-only mode. Reviewer expectations of a byte-equal comparison are answered with this rule + the regression-test lock above.

## Status

**KNOWN & DEFERRED** — not blocking paper §3.4. The only legitimate lift path is future cross-schema alignment work (out of Wf-A52 / Wf-A60 scope).