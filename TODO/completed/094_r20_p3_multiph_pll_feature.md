# TODO-94 — R20-P3 Multi-PH + PLL feature extraction CPU SCAFFOLD expansion

**Date:** 2026-09-30
**Status:** 🚀 **Active — CPU SCAFFOLD expansion via Wf-382-A1**
**Priority:** MEDIUM (R20-P3 ship target; GPU bench deferred per §5.1)
**Owner:** Hugo Lee + Wf-382-A1 ultracode agent
**Parent:** `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 R20-P3 + `TODO/completed/94_r20_p3_multiph_pll_feature.md` (Wf-37 P3 ship surface) + `TODO/completed/129_wf129_p3_multiph_pll_guarded.md` (guard rail)
**Skill alignment:** `mol-metal-multiph-pll` + `mol-metal-multi-ph-pll-feature` (R20-P3 ship axis)

---

## §0 TL;DR

R20-P3 ship target: lift Multi-PH + PLL feature extraction from Wf-37 P3 SHIPPED R² +0.087 (honest-neg, n=200 small-cohort) → **R² ≥ 0.70 on PlatinAI 4-cell + MetalCytoToxDB 5-metal** at 5-seed × 50-epoch × dual-init.

Wf-382-A1 = CPU-only SCAFFOLD expansion: wrap existing
`molmetal/molmetal_lam/lam_chem/pll_signature.py` (Liu & Feng 2026, 684 LOC)
into the `molmetal/adapters/flow_matching_lipman/` adapter layer so the
Multi-PH + PLL feature can be consumed by the CFM trainer without a
direct cross-module dependency on `molmetal_lam`.  Full GPU integration
deferred to TODO-207 Wf-299 launch (USER_OK-gated).

---

## §1 Per-feature status (R20-P3 axis)

| Feature axis | Existing surface | Wf-382-A1 SCAFFOLD | GPU bench |
|---|---|---|---|
| **Multi-PH Fourier 32-bin** | `molmetal/math_skills/multi_parameter_ph.py` (existing, 1185 LOC) | adapter re-export from `molmetal.adapters.flow_matching_lipman.multi_ph_features` (DEFERRED — out of Wf-382-A1 scope) | TODO-207 Wf-299 |
| **PLL signature 128-d** | `molmetal/molmetal_lam/lam_chem/pll_signature.py` (existing, 684 LOC) | **NEW `molmetal/adapters/flow_matching_lipman/pll_features.py`** (Wf-382-A1 SHIPPED ~150 LOC) | TODO-207 Wf-299 |
| **Joint R² lift (Wf-37 P3 SHIPPED)** | Wf-37 R² +0.087 (n=200) + Wf-348-B1 R² +0.6814 (chemprop+PLL on 60-SMILES smoke) + Wf-359-A1 W₂ on R20-P3 features (per-cell MEASURED) | N/A (CPU bench only) | R20-P3 GPU retrain (Wf-41 per LAMBDA_ROADMAP §1.4) |

---

## §2 CPU SCAFFOLD scope (Wf-382-A1 SHIPPED)

### §2.1 New module: `molmetal/adapters/flow_matching_lipman/pll_features.py` (~150 LOC)

Public API (3 functions):
- `persistent_laplacian(mol, k=16, k_nn=8)` — thin wrapper over
  `molmetal.molmetal_lam.lam_chem.pll_signature.compute_pll` that
  exposes only the top-k eigenvalues + their graph metadata
- `pll_eigenvalues(pll_signature)` — extract the descending eigenvalues array
- `pll_to_feature_vector(eigenvalues, top_k=10)` — pad/truncate to fixed-length
  vector suitable for downstream Ridge / chemprop concatenation

### §2.2 New extraction script: `molmetal/scripts/wf382a1_pll_feature_extract.py` (~200 LOC)

- `per_mol_pll_features(mols, top_k=10)` — batch extraction helper
- `aggregate_pll_distribution(cohort)` — distribution analysis (mean/std/quartiles)
- Smoke-cohort run on benzene + methane + cisplatin with PLL feature dump

### §2.3 New test file: `molmetal/tests/test_wf382a1_pll_features.py`

- 1 NEW test: PLL spectrum sanity for benzene + methane (matches within 1e-3)

---

## §3 Expected lift (PROJECTED per party-twenty-spirit §3.5)

- **Wf-37 P3 SHIPPED R² +0.087** on n=200 small-cohort (HONEST_NEGATIVE_PARTIAL)
- **Wf-348-B1 R² +0.6814** chemprop+PLL on 60-SMILES smoke (CEILING_BREAK)
- **Wf-359-A1 W₂ on R20-P3 features** per-cell MEASURED (mcf7=0.4428, a549=0.4940, a2780=0.4292, a2780cis=0.4337)
- **Wf-382-A1 SCAFFOLD**: 0 lift on small cohorts (per Wf-377-A2 diagnostic — n=200 cohorts intrinsically OOD-bound; SCAFFOLD-only impact = feature extraction pipeline ready for GPU retrain)
- **R20-P3 target** (GPU bench, deferred per §5.1): R² ≥ 0.70 on PlatinAI 4-cell + MetalCytoToxDB 5-metal at 5-seed × 50-epoch × dual-init

---

## §4 Test status

- Wf-382-A1 SCAFFOLD test pass: **1/1 NEW** at `molmetal/tests/test_wf382a1_pll_features.py`
- Existing `test_pll_signature.py` (209 LOC, 12+ tests PASS) PRESERVED as math-engine contract test
- Existing `test_multi_ph_pll_integration.py` PRESERVED as integration test
- Existing `test_wf359a1_m6_kappa_w2_r20p3.py` PRESERVED as W₂ on R20-P3 reference

---

## §5 Cross-references

- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 R20-P3 + §4.2 L7 trajectory column
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (Wf-37 P3 SHIP surface)
- `TODO/completed/129_wf129_p3_multiph_pll_guarded.md` (guard rail + Wf-25 R² fix)
- `TODO/completed/32_trackB_mlc_pll_wireup_done.md` (PLL module SHIPPED)
- `TODO/completed/78_hilbert_multiph_real.md` (Hilbert+Multi-PH joint bench)
- `TODO/completed/102_wf79_p2_3way_pll_combined.md` (PLL+3way integration)
- `TODO/completed/335m1_chemprop_pll_full.md` (chemprop+PLL R² +0.6814 CEILING_BREAK)
- `TODO/completed/348b1_*.md` (Wf-348-B1 chemprop+PLL on 60-SMILES smoke)
- `TODO/completed/359a1_*.md` (Wf-359-A1 W₂ on R20-P3 features per-cell MEASURED)
- `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/result.json` (per-cell W₂ reference)
- `TODO/pending/207_wf299_cfm_3wave_launch.md` (USER_OK-gated GPU launch)
- AGENTS.md §5.1 (Standing GPU authorization) + §5.2 (Honest framing) + §5.4 (Tier-2 markers)

---

## §6 Honest framing (party-twenty-spirit §3.5)

- SCAFFOLD-only; full Multi-PH + PLL feature integration deferred to TODO-207 Wf-299 launch.
- Expected impact on small cohorts (n=200): **0 lift** (per Wf-377-A2 diagnostic).
- Pre-flight 89/2/3/94 PRESERVED at Wf-360-C1 baseline; Wf-382-A1 is docs+light-code only, no new pre-flight gates added.
- All speedup / lift claims are PROJECTED, not MEASURED end-to-end on Blackwell sm_120.
- 0 GPU launches performed; 0 `paper/*` touches (per AGENTS.md §5.3 TODO-84 active).
