# TODO-125 — Per-pocket Vina balance strengthen via Catherine closure-bias

**Date**: 2026-09-27
**Status**: 🟢 **SHIP-honest-partial** (Wf-142 P1 ex=32 follow-up: 2/5 pockets clear 0.30 uniform floor — 4aaw 0.786, 1h36 1.000; 1/5 partial — 3ny8 0.286 fails by 0.014; 2/5 honest-negative — 3eml 0.000, 4rlu 0.000 cohort-bound; aggregate rate 0.150 → 0.414 with Δ +0.264 across 5 pockets)
**Priority**: **P1 HIGH** (per-pocket uniformity is essential for paper-grade claim)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Honest model-state assessment (2026-09-27)
**Skill alignment**: `molmetal-sbdd-env` + `mol-metal-click-chemistry`
**Close-out**: `TODO/completed/142_vina_per_pocket_ex32_high_affinity_lift.md` (Wf-142 P1 final report, ≤350 lines, 2026-09-27)
**Gap-deferred**: 3/5 pockets below 0.30 floor (3ny8 by 0.014, 3eml/4rlu cohort-bound) require per-pocket pharmacophore depth + per-seed n=500 — DEFERRED to GPU run.

---

## §0 Why this plan exists

**Current state** (per-pocket vina_high_affinity rate):
- 1h36: **0.95** ✅ (95%)
- 4aaw: **0.65**
- 4rlu: 0.45
- 3ny8: 0.35
- 3eml: **0.05** ❌ (5%)

**Aggregate**: vina_high_affinity (all-pocket) = **0.51** — barely above coin flip

**Why 90% spread**:
- 1h36 receptor is MMP13 with clear hydrophobic pocket — well-suited to small organic
- 3eml receptor has tighter binding site — current generation bias toward generic organics produces poor poses

**Fix path**: use **Catherine closure-bias** to generate metallodrugs targeted at 3eml-like pockets.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU + GPU-0 Blackwell |
| Wall-time | 4 hr CPU + 2 days GPU |
| Output | per-pocket closure-bias generation + per-pocket vina_high_affinity |
| Tests | ~15 new tests |

## §2 Solution path (2 innovations stacked)

1. **Catherine closure-bias per-pocket generation**
   - For each pocket, generate candidates biased toward 5-click productive space + pocket-specific pharmacophore
   - `pocket_specific_closure_bias(pocket_id)` returns candidate bias weights
2. **Per-pocket Wasserstein alignment** (theorem T1)
   - Match candidate distribution to pocket's known ligand distribution

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** pocket_specific_closure_bias function | 4 hr CPU | `molmetal/molmetal_lam/reward/pocket_closure_bias.py` (NEW, ~200 LOC) + 6 tests | per-pocket candidate generation count |
| **P1.2** Wasserstein per-pocket alignment | 1 day GPU-0 | `molmetal/molmetal_lam/reward/wasserstein_pocket.py` (NEW) + 4 tests | Wasserstein distance decay per pocket |
| **P1.3** Per-pocket Vina re-bench | 1 day GPU-0 | `molmetal/reports/wf125_vina_per_pocket_rerun/` | per-pocket vina_high_affinity — 3eml ≥ 0.50 (vs current 0.05) |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: per-pocket closure-bias candidate count + Wasserstein distance + vina_high_affinity per pocket
- **PROJECTED**: 3eml ≥ 0.50 (vs current 0.05), 4rlu ≥ 0.65 (vs 0.45), 3ny8 ≥ 0.55 (vs 0.35)

## §5 Honest framing

The 0.95 vs 0.05 spread is **a real distribution mismatch** between our generation bias and pocket-specific requirements. The fix isn't to make every pocket 0.95 — it's to bias generation per pocket. Per-pocket results may legitimately differ; the goal is **no pocket below 0.30** (uniform floor).

## §6 Cross-references

- `metrics/by_metric/vina_high_affinity.json` (current 0.51)
- `metrics/by_metric/vina_high_affinity_{1h36,3eml,3ny8,4aaw,4rlu}.json`
- `molmetal/math_skills/catherine_properties.py` (closure theorem)
- `molmetal/math_skills/wasserstein_gradient_flow.py` (theorem)
- TODO-119/120/121/122/123/124 (Wasserstein reuse)
- TODO-121 (closure oversampling — same machinery)
- TODO-84 paper rule (paper §4 not blocked)
- LAMBDA_ROADMAP §4.2 m3 (Vina) + m4 (Success Rate) columns
