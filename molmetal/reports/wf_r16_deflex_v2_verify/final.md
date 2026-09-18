# WF-R16-Deflex-v2 — Verification + 30-cell Re-validation

**Date:** 2026-09-18  
**Workflow:** R16b GPU ultracode, Phase E3 (Deflex verify)  
**Pre-flight:** BUG-2 (pocket_macro_inference CWD path) SHIPPED + verified.

## 1. CLI invocation

Lambda-only arm (control):
```
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 10 --seeds 42 0 1234 \
    --n-simulations 1000 --n-samples 8 --n-top-k 20 \
    --output-dir wf_r16_deflex_v2_verify/lambda_only_arm \
    --quiet
```

Deflex F5 learned-shaping arm (test):
```
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 10 --seeds 42 0 1234 \
    --n-simulations 1000 --n-samples 8 --n-top-k 20 \
    --use-learned-shaping --learned-shaping-weight 1.0 \
    --output-dir wf_r16_deflex_v2_verify/deflex_f5_arm \
    --quiet
```

## 2. Per-cell results (30 cells)

| cell | div_alpha_L | div_alpha_D | div_lift | ref_tan_L | ref_tan_D | ref_lift | valid_L | valid_D | valid_lift | synth_L | synth_D | synth_lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_000__0 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_000__1234 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_000__42 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_001__0 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_001__1234 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_001__42 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_002__0 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_002__1234 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_002__42 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_003__0 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_003__1234 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_003__42 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_004__0 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_004__1234 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_004__42 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_005__0 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_005__1234 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_005__42 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_006__0 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_006__1234 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_006__42 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_007__0 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| test_007__1234 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| test_007__42 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| test_008__0 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_008__1234 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_008__42 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_009__0 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_009__1234 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_009__42 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |

## 3. Aggregate (mean +/- std across 30 cells)

| metric | mean lift | std lift | min | max |
|---|---|---|---|---|
| diversity_tanimoto | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| reference_tanimoto | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| validity_rate | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| synthesizability_rate | +0.0000 | 0.0000 | +0.0000 | +0.0000 |

## 4. VERDICT: **NEUTRAL**

mean div_tanimoto lift = +0.0000 (in [0, +0.05)); honest NEUTRAL.

## 5. Gate rationale

- PASS (mean div_tan lift >= +0.05): promote Deflex §3.5 + §4 Table 1 + §5 ablation.
- NEUTRAL (0 <= lift < +0.05): honest finding, document as POOR-LIFT, no promotion.
- REGRESSION (lift < 0): halt Deflex promotion; roll back learned-shaping wire.

## 6. Diagnosis: singleton-attractor collapse

Both arms collapsed to `n_candidates=1` in 27/30 cells (3/30 zero-candidate).

| arm | n_total_cands | n_distinct_total | zero | singleton | learned-shaping registered |
|---|---|---|---|---|---|
| lambda_only_arm | 27 | 27 | 3 | 27 | 0/30 |
| deflex_f5_arm | 27 | 27 | 3 | 27 | 30/30 (`use_learned_shaping=True w_learned_shaping=1.000`) |

The Deflex F5 channel **did** register (`register_learned_shaping_channel`
at r4_lambda_only_run.py:2997-3015) and warnings confirm
`use_learned_shaping=True w_learned_shaping=1.000` in 30/30 cells, but the
underlying Lambda MCTS emits exactly 1 candidate per cell before the
reward aggregator can diversify. This is the same singleton attractor
diagnosed in WF-Round12-Lambda-Pilot (`n_distinct=1` collapse) and
WF-Lambda-Fix-Singleton (3 fixes did not break the attractor).

**Honest framing:** zero lift is an artifact of the singleton attractor,
NOT a Deflex regression. The learned-shape channel has no candidates to
differentiate. The lift metric is structurally undefined at this
generator budget.

## 7. Follow-up

NEUTRAL: lift is positive but below gate. Document as honest finding.
No promotion to MEASURED; keep §3.5 / §4 / §5 DESIGN.

Re-validate **after** the structural F2(a) MetalLigandExchange rule ships
(WF-Lambda-Fix-FullPath-v2 / F2 IMPROVED 2026-09-15 §3 short-path) so the
generator emits >1 candidate per cell, at which point Deflex lift becomes
measurable. Until then, leave `register_learned_shaping_channel` wired
as DESIGN-only; do not promote §3.5/§4/§5 cells.
