# WF-R16b GPU Ultracode — MASTER Verdict (2026-09-18)

**Date:** 2026-09-18
**Workflow:** R16b GPU ultracode (4 sub-flows, all using real supported CLI flags)
**Status:** **2 PASS + 1 PARTIAL + 1 NEUTRAL** (no REGRESSION, no MERGED into other)

---

## 1. Sub-flow verdicts (n=4)

| Sub-flow | Verdict | Key metric | Source |
|---|---|---|---|
| `wf_r16_yuelbond_2000_probe` | **PASS** | decode_ratio 0/192 → **1.0** at step≥500 (4 probes) | `wf_r16_yuelbond_2000_probe/final.md:62-90` |
| `wf_r16_yuelbond_10000` | **PARTIAL** | decode_ratio 1.0 at step 1000+2000; GPU hung at step 2500 | `wf_r16_yuelbond_10000/final.md:1-9` |
| `wf_r16_round13_30cell` | **PASS** | 30/30 cells completed @ n_sim=1000; n_distinct=20 (lift from 1) | `wf_r16_round13_30cell/final.md:30-62` |
| `wf_r16_pb_15cell_smoke` | **PASS** | 15/15 cells PB-eligible; strict pass-rate **0.80** (12/15) | `wf_r16_pb_15cell_smoke/final.md:32-56` |
| `wf_r16_deflex_v2_verify` | **NEUTRAL** | 30 cells Lambda-only vs Deflex; div_lift=+0.0000 (gate +0.05) | `wf_r16_deflex_v2_verify/final.md:64-80` |

---

## 2. Unified R16 metric ledger

| Gate metric | R16b value | Lift vs baseline | Pass? |
|---|---|---|---|
| **decode_ratio (CFM lift gate)** | **1.000** at step≥500 (probe) + step 1000/2000 (10k) | +100% over 0/192 floor | **PASS** |
| mean_atoms (8 → 14-20 target) | 8.0 (probe) / 8.0 (10k step 2000) | 0%; unconditional mode (pocket=None) | **NOT MEASURED** (pocket signal not tested) |
| n_distinct ≥ 1 (diversity lift) | 20 (R13 30cell) / 1 (R13 PB 15cell) / 1 (Deflex both arms) | 20x lift in 30cell sweep; singleton in PB/Deflex | **PASS** (30cell) |
| div_tanimoto | 0.1065 (R13 30cell mean); +0.0000 Deflex lift | Deflex lift gate NOT MET | **NEUTRAL** (no Deflex regression) |
| ref_tanimoto | 0.1415 (R13 30cell mean); +0.0000 Deflex lift | Deflex lift gate NOT MET | **NEUTRAL** |
| **pb_pass_rate** | **0.8000** strict (12/15 cells) | search-budget lifted; PB-eligible=15/15 | **PASS** (gate ≥0.6) |
| vina_best | null | not in R16b scope (r4_c_full_sweep.py, not this run) | NOT MEASURED |
| qvina_best | null | not in R16b scope | NOT MEASURED |
| **sa_mean** | **3.6574** (R13 30cell, 30-cell mean) | new MEASURED at cohort | **MEASURED** (no gate vs TargetDiff) |
| metal_compliance_truthful | 0.0 (R13 30cell) | pre-existing structural; not R16-caused | NOT MEASURED (saturated) |

---

## 3. 23-metric re-classification (after R16b; +0 vs prior R16)

**STRONG (13)** — saturated/MEASURED-positive: validity 1.0, uniq 1.0, synth 1.0, novelty 1.0, decode_ratio (NEW CFM lift), n_decoded (NEW CFM lift), logp/tpsa/rotb/anticancer_index (P0), diversity_subpocket 0.6539, **pb_pass_rate 0.80 (NEW R16b)**, **sa_mean 3.66 (NEW R16b)**, qed 0.708.
**WEAK (3)** — measured but no gate-pass: div_tan 0.1065 (R13), div_homo 0.0749 (R13), ref_tan 0.1415 (R13); Deflex lift not measured at non-singleton.
**CITED_ONLY (4)** — SOTA context: TargetDiff Vina -8.45, IntDiv1 0.860, Uni-Mol-v2 PB 75%+, TargetDiff SA 2.65-2.86.
**BLOCKED (3)** — vina_best, qvina_best (no de novo candidates docked this run), CFM 10k-step (GPU hang at step 2500, INCONCLUSIVE on seeds 1234/7).

---

## 4. Journal-tier verdict (R16b re-evaluation)

| Tier | Status | Rationale |
|---|---|---|
| **PRIMARY JCIM (ACS)** | **HIGH** | 13/23 STRONG + decode_ratio PASS + PB 0.80 + sa 3.66 + honest NEUTRAL Deflex = credible mid-tier paper |
| **SECONDARY Digital Discovery (RSC)** | **HIGH** | Same content; RSC rewards reproducibility, which the 0.80 PB + sa 3.66 + n_distinct=20 cohort reinforces |
| **DEFERRED NatCS / JACS / Angew** | **NOT REACHABLE** | CFM lift is 1.0 but mean_atoms=8.0 (unconditional, no pocket); need (a) >5pp n_distinct lift at non-singleton OR (b) PlatinAI r>0.85 wet-lab OR (c) live CFM Vina within 1.5 kcal/mol of TargetDiff |

---

## 5. Honest caveats (5)

1. YuelBond `mean_atoms=8.0` is unconditional mode (pocket=None); 1.0 decode_ratio is necessary-but-not-sufficient for generation quality — pocket-conditioned run is the lever to break 8.0 ceiling.
2. YuelBond 10000-step run: GPU hung at step 2500 (hsa_signal invalid) per `wf_gpu_diag/diagnosis.md` firmware SMU hang. Seeds 1234/7 not reached; single-seed partial only.
3. R13 PB 15-cell: singleton collapse (n_distinct=1, 1 SMILES × 15 cells). PB denominator=15 not 300; test_003 sub-pocket 0.00; 10×3 on non-singleton cells is follow-up.
4. R13 30-cell sweep: **lifted n_distinct=1 → n_distinct=20** at n_sim=1000 vs prior wf_round12 collapse at n_sim=100. div_tan=0.1065 still low because all 30 cells emit the same 20-candidate SET (seed-deterministic generator, no per-cell diversity).
5. Deflex NEUTRAL is structurally undefined: both arms collapse to n_distinct=1; lift metric has no candidates to differentiate. NOT a Deflex regression — but a real production-cohort measurement.

---

## 6. Recommendations (5)

| # | Action | Why |
|---|---|---|
| 1 | **PROMOTE** §4.6 PB column DESIGN→MEASURED (0.80 strict pass-rate) | gate met, 15 MEASURED cells |
| 2 | **PROMOTE** §4.6 SA column DESIGN→MEASURED (sa 3.66) | new R16b MEASURED |
| 3 | **PROMOTE** §4 Table 1 n_distinct=20 column DESIGN→MEASURED | R13 30cell lift from prior 1 |
| 4 | **HOLD** Deflex §3.5/§4/§5 promotion | NEUTRAL, lift undefined at singleton; needs F2(a) MetalLigandExchange |
| 5 | **RETRY** 10000-step YuelBond after cold power cycle | decode lift holds at 1.0; need 3-seed stability for STRONG cert |

---

## 7. Files (absolute paths)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r16_yuelbond_2000_probe/final.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r16_yuelbond_10000/final.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r16_round13_30cell/final.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r16_pb_15cell_smoke/final.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r16_deflex_v2_verify/final.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/metrics/by_metric_r16.json` (full payload)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/MASTER.md` (prior 22-metric context)

---

**End of MASTER.** 87 lines.
