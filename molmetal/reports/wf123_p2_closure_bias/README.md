# Wf-123 Phase 2 — M2 pocket_closure_bias module

**Date**: 2026-09-27
**Status**: SHIPPED
**Spec**: TODO/pending/126_innovation_strengthen_roadmap.md §3 / M2
**Module**: `molmetal/molmetal_lam/reward/pocket_closure_bias.py`

## Outputs

| Artifact | Path |
|---|---|
| Module | `molmetal/molmetal_lam/reward/pocket_closure_bias.py` |
| Tests | `molmetal/tests/test_pocket_closure_bias.py` |
| Verification JSON | `molmetal/reports/wf123_p2_closure_bias/per_function_verification.json` |

## 4 public functions (TODO-126 §3 / M2)

1. `pocket_specific_closure_bias(pocket_id)` → `PocketBiasVerdict` with
   `bias_dict` summing to 1.0 across 5-click-rule weights (+ metal-coord
   rules for metalloenzyme pockets).
2. `closure_bounded_smiles_generation(pocket_id, n=100)` →
   `BoundedGenerationResult` with Catherine 6/6 filtered candidates.
3. `closure_oversample(target_metal, n=500)` →
   dict wrapper over `ClosureOversampler`; tagged
   `engine="closure_catherine_oversample"`.
4. `per_pocket_pharmacophore_lookup(pocket_id)` → curated descriptor dict
   for the 5 known pockets (1h36 / 3eml / 3ny8 / 4aaw / 4rlu).

## Consumers (4 strengthen TODOs)

| TODO | Weak metric | Function used |
|---:|---|---|
| TODO-119 | m14 decode_ratio | `closure_bounded_smiles_generation` |
| TODO-121 | m19 LOMO (Rh bottleneck) | `closure_oversample` |
| TODO-124 | m5 SA | `closure_bounded_smiles_generation` |
| TODO-125 | per-pocket Vina | `pocket_specific_closure_bias`, `per_pocket_pharmacophore_lookup` |

## MEASURED-vs-PROJECTED (per party-twenty-spirit §3.5)

**MEASURED**:
- All 4 functions return valid output (verified via
  `build_per_function_verification()`).
- Per-pocket bias dicts sum to 1.0 across 5 known pockets.
- Per-pocket closure-BFS generation emits Catherine 6/6 PASS candidates.
- Closure oversample thin-wraps the existing `ClosureOversampler` with
  audit-friendly engine tag.

**PROJECTED** (downstream — NOT this module's job):
- m5 SA lift (TODO-124) is plausible from bias-shift to closure cohort.
- m14 decode_ratio lift (TODO-119) requires YuelBond swap + Wasserstein,
  not just closure-bias.
- m19 LOMO lift (TODO-121) requires multi-task backbone + Wasserstein
  + Rh oversample stacked.
- per-pocket Vina lift (TODO-125) requires per-pocket Wasserstein
  alignment, not just per-pocket bias priors.

## Honest framing

This module is a **bias + candidate-generation primitive** — its outputs
are downstream-friendly priors + verified Catherine-PASS candidates.  It
does NOT score, dock, or guarantee metric lift.  Per TODO-126 §7 every
emitted candidate is tagged with an `engine` field
(`pocket_bias_prior`, `closure_catherine_bounded`,
`closure_catherine_oversample`,
`pharmacophore_descriptor`) so scoring contributions are auditable
rather than silent substitutions.

## Verification run

```
pytest molmetal/tests/test_pocket_closure_bias.py -v
collected 55 items
============================== 55 passed, 1 warning in 6.42s ==============================
```

CPU-only; RDKit + Catherine + closure-theorem BFS only.
