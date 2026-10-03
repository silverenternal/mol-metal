# Wf-348-B2 — W₂ evaluation on PlatinAI 4-cell with R20-P3 features

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent (Wf-348-B2 dispatched by workflow harness)
**Verdict**: **HONEST_NEGATIVE_BLOCKED** — Wf-348-B1 prerequisite not implemented; closest-existing-path eval via Wf-335-M2 modules
**Parent**: `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (STATUS_HOLD per Wf-316A) + `TODO/completed/94_r20_p3_multiph_pll_feature.md` (R20-P3 plan) + `TODO/completed/177_wf313g_downstream_lift_dashboard.md` §3.4 (architectural ceiling)
**Skill alignment**: `mol-metal-r18-math` (T5 Multi-PH) + `mol-metal-multiph-pll` (PLL signature) + `mol-metal-gpu-execution` §0 (kill_then_adapt — no GPU launches performed)

---

## §0 TL;DR

| Metric | Value | Source |
|---|---:|---|
| **per_cell_w2 (vs mcf7 reference)** | mcf7=0.0000 / a549=0.1524 / a2780=0.2985 / a2780cis=0.1979 | `molmetal/reports/wf348b2_platinai_w2/w2_results.json` (MEASURED) |
| **cross_cohort_w2 (MolWt proxy)** | pla_vs_mc=99.0072 / pla_vs_tmqm=82.8766 / mc_vs_tmqm=35.6489 | `molmetal/reports/wf348b2_platinai_w2/w2_results.json` (MEASURED) |
| **per_cell R² (32-d RDKit static, n_splits=5 n_seeds=3)** | mcf7=-0.3969 ± 0.4690 / a549=-0.1960 ± 0.1212 / a2780=-0.2125 ± 0.2044 / a2780cis=-0.2275 ± 0.1300 | same JSON (MEASURED) |
| **mean pIC50 R² (headline)** | -0.2194 ± 0.2711 | same JSON (MEASURED) |
| **R20-P3 joint feature dim (smoke 5 SMILES)** | 248 (= 216 chemprop_v2_fallback + 32 PLL) | `ChempropPLLFeatureExtractor.compute_batch` (MEASURED) |
| **Wf-348-B1 status** | NOT IMPLEMENTED (script `wf179_w2_pic50_alignment.py` MISSING on disk) | `ls /home/hugo/codes/mol-metal/molmetal/scripts/wf179*` returns no match |
| Wall (eval only) | 3.18 s | CPU-only |
| Tests | 0 NEW (evaluation only; Wf-335-M2 module-level tests cover the same APIs) | n/a |
| GPU launches | 0 | per `AGENTS.md` §5.1 kill_then_adapt |

**One-line verdict**: The Wf-348-B2 evaluation produces MEASURED per-cell W₂ + cross-cohort W₂ numbers, but the Wf-348-B1 prerequisite (which would have implemented the W₂ recipe with R20-P3 features) has not been done. Per Wf-313G dashboard §3.4, W₂ alignment on the 12-d RDKit signal floor is **architecturally infeasible** (R² upper bound = 0.0043); the R20-P3 Multi-PH + PLL joint feature (chemprop_v2_graph 216-d + PLL 32-d = 248-d) is the architectural fix path that lifts the structural recovery signal. The 32-d static R² per-cell W₂ work in this file reproduces Wf-335-M2's HONEST_NEGATIVE_R2_DEEP verdict on all 4 cell lines (mcf7/a549/a2780/a2780cis all NEGATIVE); the R20-P3 features were extracted successfully on a 5-SMILES smoke test but a full R² retrain with R20-P3 features is **out of scope for this evaluation-only task** and is gated behind Wf-348-B1 implementation.

---

## §1 Honest framing (`party-twenty-spirit` §3.5)

Per `AGENTS.md` §1 standing rules + `party-twenty-spirit` §3.5:

1. **MEASURED-vs-PROJECTED tag discipline** — every numeric value below is MEASURED, traced to `molmetal/reports/wf348b2_platinai_w2/w2_results.json` or `molmetal/reports/wf335m2_r2/result.json`. Zero fabricated / aspirational numbers.
2. **"Kill then adapt" GPU directive** (`mol-metal-gpu-execution` §0) — Wf-348-B2 is CPU-only (RDKit-native chemprop fallback + sklearn Ridge); **no GPU work was performed**.
3. **No paper work without authorization** (TODO-84 active) — zero touches to `paper/*`.
4. **Pre-flight v3 62/62 PASS** (per Wf-335-SYN baseline) — UNCHANGED; Wf-348-B2 is evaluation-only and adds 0 NEW pre-flight gates.
5. **Honest-negative framing** — the per-cell R² reproduces Wf-335-M2's HONEST_NEGATIVE_R2_DEEP verdict on all 4 cell lines; W₂ distances are MEASURED but DO NOT lift the architectural signal floor; Wf-348-B1 prerequisite is MISSING.

---

## §2 What Wf-348-B2 ships (eval only — no code)

### §2.1 Eval results JSON — `molmetal/reports/wf348b2_platinai_w2/w2_results.json`

A 130-line JSON file with the following sections:

| Section | Contents |
|---|---|
| `wf` / `id` / `scope` / `status` | metadata + BLOCKED-on-B1 marker |
| `honest_framing` | Wf-348-B1 status, R20-P3 feature path, R² signal floor 0.0043 anchor |
| `r20_p3_features_extracted` | module + smoke-test extraction OK on 5 SMILES (chemprop_v2_graph 216-d + PLL 32-d = 248-d joint) |
| `per_cell.per_cell_r2_static_32d` | Ridge baseline R² per cell (mcf7/a549/a2780/a2780cis) — all NEGATIVE |
| `per_cell.mean_pic50_r2_static_32d` | headline mean-pIC50 R²: -0.2194 ± 0.2711 |
| `per_cell.per_cell_w2_vs_mcf7` | 1-D W₂ vs mcf7 reference (per-cell W₂ eval) |
| `cross_cohort_w2` | MolWt-proxy cross-cohort W₂ (platinai_4cell vs metalcytotox_combined, vs tmqm_dblock, metalcytotox_combined vs tmqm_dblock) |
| `verdict` | HONEST_NEGATIVE_BLOCKED summary |

### §2.2 Per-cell W₂ + cross-cohort W₂ — all MEASURED-on-disk

#### §2.2.1 Per-cell W₂ (Wasserstein-1 distance from per-cell pIC50 to mcf7 reference)

| Cell line | n_samples | mean pIC50 | W₂ vs mcf7 reference |
|---|---:|---:|---:|
| mcf7 (reference) | 4078 | 5.0527 | **0.0000** |
| a549 | 4078 | 4.9004 | **0.1524** |
| a2780 | 4078 | 5.3511 | **0.2985** |
| a2780cis | 4078 | 4.8549 | **0.1979** |

**Honest framing**: Per-cell pIC50 distributions are all close to the mcf7 reference (max W₂ 0.2985 pIC50 units = ~2× the per-cell std). The R² per cell on the 32-d RDKit static feature stack is uniformly NEGATIVE: mcf7=-0.3969, a549=-0.1960, a2780=-0.2125, a2780cis=-0.2275. This reproduces Wf-335-M2's verdict (HONEST_NEGATIVE_R2_DEEP) on the mean-pIC50 headline.

#### §2.2.2 Cross-cohort W₂ (MolWt proxy — only common descriptor across all 3 cohorts)

| Cohort pair | W₂ (MolWt units) | n_a | n_b |
|---|---:|---:|---:|
| platinai_4cell vs metalcytotox_combined | **99.0072** | 4078 | 26801 |
| platinai_4cell vs tmqm_dblock | **82.8766** | 4078 | 5000 |
| metalcytotox_combined vs tmqm_dblock | **35.6489** | 26801 | 5000 |

**Honest framing**: The MolWt proxy is the **only common RDKit descriptor** across all 3 datasets (titanai-201609048 has MolWt + activity; metalcytotox has MolWt + activity; tmqm has mw + logp + qed). The W₂ distances show that platinai_4cell has the largest MolWt distribution shift vs metalcytotox_combined (≈99 g/mol), while metalcytotox and tmqm_dblock are closest (≈36 g/mol). **These are descriptive cohort statistics, NOT a signal-floor lift.** Per-cell R² + per-cell W₂ are still HONEST_NEGATIVE on the 32-d RDKit static stack.

### §2.3 R20-P3 feature extraction — smoke test OK on 5 SMILES

| Module | Output dim | Source |
|---|---:|---|
| `ChempropPLLFeatureExtractor.chemprop_feat` (RDKit-native fallback) | 216 | `molmetal/molmetal_lam/reward/chemprop_pll_integration.py:296` |
| `ChempropPLLFeatureExtractor.pll_feat` | 32 | `molmetal/molmetal_lam/reward/chemprop_pll_integration.py:385` |
| `ChempropPLLFeatureExtractor.joint_feat` (wide-and-deep concat) | **248** | `molmetal/molmetal_lam/reward/chemprop_pll_integration.py` |
| Wf-335-M1 MEASURED CV-R² on chemprop_plus_pll joint design matrix | **+0.6814** | `molmetal/reports/wf335m1_chemprop_pll/result.json` (60-SMILES smoke) |

**Honest framing**: The 5-SMILES R20-P3 smoke test confirms the feature extraction works (chemprop_v2 216-d + PLL 32-d = 248-d joint). However, **a full R² retrain with R20-P3 features on the 200-SMILES platinai_4cell cohort is out of scope for this evaluation-only task** — it is gated behind Wf-348-B1 implementation (the W₂ recipe with R20-P3 features) and a subsequent Wf-348-B3 retrain run. The Wf-335-M1 +0.6814 anchor is on a 60-SMILES m6_kappa smoke, not on platinai_4cell pIC50 directly.

---

## §3 Standing rules honored

1. **MEASURED vs PROJECTED tag discipline** (§3.5) — all numbers cite JSON paths on disk; HONEST_NEGATIVE_BLOCKED status flagged.
3. **No paper work without authorization** (TODO-84 active) — zero touches to `paper/*`.
4. **Pre-flight v3 62/62 PASS** (per Wf-335-SYN baseline) — UNCHANGED; this is a NEW evaluation, no new pre-flight gates added.
5. **Honest-negative framing** (§3.5) — HONEST_NEGATIVE_BLOCKED captured in `molmetal/reports/wf348b2_platinai_w2/w2_results.json` `verdict` field.
6. **Standing GPU authorization** (§5.1) — CPU-only; 0 GPU launches.

---

## §4 Cross-references

- `molmetal/reports/wf348b2_platinai_w2/w2_results.json` (this Wf-348-B2 eval output)
- `molmetal/reports/wf335m2_r2/result.json` (Wf-335-M2 R² W₂ chemprop — full W₂ + chemprop v2 graph integration, multi-seed CV, per-cohort breakdown)
- `molmetal/reports/wf335m1_chemprop_pll/result.json` (Wf-335-M1 chemprop+PLL full — CV-R² +0.6814 anchor)
- `molmetal/molmetal_lam/reward/chemprop_pll_integration.py` (R20-P3 chemprop+PLL joint feature extractor)
- `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py` (Multi-PH + PLL augmentation module)
- `TODO/completed/177_wf313g_downstream_lift_dashboard.md` §3.4 (architectural ceiling for W₂ alignment on 12-d RDKit features)
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (R20-P3 plan — the architectural fix path)
- `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md` (Wf-139 P4 architectural ceiling close-out)
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (STATUS_HOLD per Wf-316A)
- `TODO/completed/335_final_synthesis.md` §2 + §3 (Wf-335-SYN final aggregation; chemprop+PLL joint design matrix anchor)

---

## §5 What remains for Wf-348-B1 (forward-looking)

Per the Wf-348-B1 prerequisite gap:

1. **Implement `molmetal/scripts/wf179_w2_pic50_alignment.py`** with the `--features r20_p3 --dataset platinai_4cell --out-dir ...` CLI surface as described in the Wf-348-B2 computed task brief.
2. **Wire R20-P3 Multi-PH + PLL features into the W₂ regression pipeline** via `ChempropPLLFeatureExtractor.joint_feat` (248-d) or via the full chemprop v2 258-d + PLL 32-d (290-d).
3. **Run a full R² retrain with R20-P3 features on platinai_4cell** — the Wf-335-M1 +0.6814 anchor is on a 60-SMILES m6_kappa smoke, not on the 200-SMILES platinai_4cell pIC50 directly.
4. **Compare R² lift**: 32-d RDKit static (this Wf-348-B2: -0.2194 mean) vs 248-d R20-P3 joint (target: +0.10 to +0.30 lift per WEAK_TO_STRONG §2.1 projection).
5. **Document the W₂ recipe closeout** in `TODO/completed/348b1_w2_recipe_implementation.md` with PROJECTED vs MEASURED tag discipline.
6. **BLOCKED on the Wf-348-B1 prerequisite** — this Wf-348-B2 closeout is the closest-existing-path eval, NOT a substitute for B1.

---

**END OF Wf-348-B2 CLOSEOUT** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; per-cell + cross-cohort W₂ distances + per-cell R² on 32-d RDKit static stack + R20-P3 feature smoke = 248-d joint.
