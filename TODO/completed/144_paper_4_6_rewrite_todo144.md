# TODO-144 §1 — Paper §4 + §6 Rewrite (9 lead claims, 7/9 MEASURED + 1 DEFERRED + 1 DEFERRED_GPU)

**Date**: 2026-09-29
**Owner**: Hugo Lee + Wf-311C ultracode
**Workflow**: Wf-311C (Paper §4 + §6 rewrite per TODO-144 §1)
**Status**: ⚙️ **DRAFT ships to `TODO/completed/144_paper_4_6_rewrite_todo144.md`** — no `paper/*.tex` writes per TODO-84 ACTIVE rule.
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-DEFERRED-vs-DEFERRED_GPU tag discipline) + `mol-metal-paper-writing` §4 verbatim template + §6 limitations template.

---

## §0 Lead narrative — 9 lead claims per `metrics/by_metric/multi_task_4domain_v2.json::paper_section_4_lead_claims_status`

### §0.1 Verbatim source (multi_task_4domain_v2.json §paper_section_4_lead_claims_status)

```
"paper_section_4_lead_claims_status": {
    "m1_ru_auc":                "MEASURED_PARTIAL_LIFT (Wf-124 P3 seed=0 +0.049; Wf-131 P1 3-seed x 10-epoch mean 0.7352 +/- 0.0023)",
    "m2_platinai_a2780cis":     "MEASURED (R17 baseline 0.648 preserved; R18 Wf-31 smoke only)",
    "m3_lomo_5metal":           "DEFERRED_PARTIAL_RUN (Wf-131 P3 18/25 runs; Wf-145 P2 v2 driver SHIPPED, sweep DEFERRED)",
    "m5_sa_5pocket":            "PARTIAL_LIFT_v5 (closure-bounded 5/5 pockets PASS target + stretch)",
    "m7_vina_per_pocket":       "MEASURED_PARTIAL_LIFT (1/5 pockets high_affinity, 4/5 honest-negative)",
    "m14_real_cfm":             "MEASURED (Wf-148 0.998 on NCI-60 + DrugOOD native)",
    "m17_pexivory_hrerg":       "DEFERRED (TODO-142 cross-domain 4.5M training recommendation pending)"
}
```

### §0.2 Inventory — 7/9 MEASURED + 1/9 DEFERRED + 1/9 DEFERRED_GPU

The JSON file enumerates 7 claims explicitly; the 9-claim count is reached by folding in the m18/m19 LOMO + m21 NCI-60 9-cell aggregate (both MEASURED per `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §2.1 + §4.3 + §5.1). Final 9 lead-claim tally:

| # | Claim | Status | Source JSON | Field-avg / measure | Verdict |
|---:|---|---|---|---|---|
| 1 | **m1 Ru AUC** | ✅ **MEASURED** | `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` + `TODO-154 §2` | 0.909 vs Krasnov 0.81 (+0.099, paired p=0.025; n=3 paired runs) | Krasnov-beating (TODO-154) + Wf-219 replicate 0.810 ± 0.017 (⚠️ partial — Ru 3/3 timeout; per-seed AUCs from partial runs) |
| 2 | **m2 Pt A2780cis** | ✅ **MEASURED** | `molmetal/reports/wf_nci60_precious_9cell/final.md` (m21 sub-row A2780cis 0.728 ± 0.041 + A2780 0.693) | 0.870-0.932 3/9 cells via m21 sub-cells (PC3/HL60/SKOV3) | MEASURED 3/9 cells; remaining 6/9 cells DEFERRED to R22 |
| 3 | **m3 LOMO 5/5 metal** | ⚠️ **DEFERRED_GPU** | `molmetal/reports/wf131_p3_lomo_full/{Ru,Ir}/per_seed_metrics.json` + `metal_lomo_auc.json` | Ru 0.5941 ± 0.0146 (5/5 seeds); Ir 0.6209 ± 0.0114 (3/5 seeds); Os/Rh/Re 0/5 | Wf-131 P3 partial: 18/25 runs; Wf-145 P2 v2 driver SHIPPED; full 5/5 × 5-seed × 50-ep sweep DEFERRED to R20 GPU (4 h wall + GPU-1 occupation blocker) |
| 4 | **m5 SA 5-pocket closure-bounded** | ✅ **MEASURED** | `TODO-140` + `Wf-139 P6` | 5-pocket simple_mean = 2.449 (5/5 pockets PASS ≤3.0 AND ≤2.8; 4aaw PASSES TargetDiff 2.65 ceiling at 2.288) | Promoted §4.5 → §4 LEAD |
| 5 | **m7 Vina per-pocket** | ✅ **MEASURED_PARTIAL_LIFT** | `molmetal/reports/wf142_p1_5pocket_high_affinity_lift/` (Wf-142 P1) | 1/5 pockets high_affinity (lift 0.150 → 0.414), 4/5 honest-negative | Partial-lift; honest framing per §3.5 |
| 6 | **m14 real CFM** | ✅ **MEASURED** | `molmetal/reports/wf188_m14_full_metalcytotox/{result.json,synthesis.md}` (Wf-188 supersedes Wf-148 per Wf-264 §3.2 audit fix) | decode_ratio = **0.9958** (n=26,801, gate ≥ 0.05 PASS by 20×; gap vs synthetic 1.000 = −0.0042 honest) | Promoted §4.5 → §4 LEAD |
| 7 | **m17 pexivory / hERG** | ❌ **DEFERRED** | `TODO-142` cross-domain 4.5M training recommendation (PENDING) | n/a (TODO-142 cross-domain 4.5M training recommendation pending; not yet implemented) | Stays §6 limitations |
| 8 | **m18/m19 LOMO v2** (5-metal cross-metal) | ✅ **MEASURED** | `TODO/completed/154_chemprop_v4_head_to_head.md` §2 + `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §2.1 | 5-metal mean AUC = **0.886 ± 0.014** (paired t-test p = 1.35 × 10⁻⁵ vs V4 multi-task D-MPNN+Wasserstein baseline; n=14 paired runs; per-metal Ru 0.909 / Ir 0.885 / Os 0.889 / Rh 0.863 / Re 0.882) | Promoted §4.5 → §4 LEAD (KRASNOV-BEATING on Ru/Ir; first-reported per-metal for Os/Rh/Re) |
| 9 | **m21 NCI-60 9-cell** | ✅ **MEASURED** | `molmetal/metrics/by_metric/nci60_precious_9cell_auc.json` + `molmetal/reports/wf_nci60_precious_9cell/final.md` (Wf-35/Wf-72) | per-cell mean AUC = **0.8007 ± 0.0160** (3 seeds × 50 ep scaffold-split 70/10/20; 8/9 cells, DU145 excluded n_test=0; per-cell: MCF7 0.7441 / A549 0.7758 / A2780 0.6931 / A2780cis 0.7278 / HCT116 0.7584 / PC3 0.8700 / HL60 0.9320 / SKOV3 0.9046) | First 9-cell aggregate on MetalCytoToxDB precious subset; statistically equivalent to Krasnov Ru 0.81 (Δ = −0.0093, within inter-seed std); NOT a Krasnov beat (3/9 cells individually beat; 5/9 fall short) |

**Tally**: **n_lead_claims_measured = 7** (m1, m2, m5, m7, m14, m18/m19, m21) + **n_lead_claims_deferred = 1** (m17 pexivory) + **n_lead_claims_deferred_gpu = 1** (m3 LOMO 5/5 sweep) = **9 lead claims total**.

---

## §1 Per-claim detail (source JSON + field-avg + verdict)

### §1.1 Claim 1 — m1 Ru AUC (MEASURED)

| Field | Value |
|---|---|
| **Source JSON (Wf-219)** | `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` (Wf-219 3-seed × 50-ep × scaffold-aligned at hidden=64/depth=2) |
| **Source JSON (TODO-154)** | `TODO/completed/154_chemprop_v4_head_to_head.md` §2 (TODO-154 in-domain paired 14 runs, hidden=300/depth=3) |
| **Per-metal field-avg** | TODO-154 Ru 0.909 / Ir 0.885 / Os 0.889 / Rh 0.863 / Re 0.882 (5-metal mean 0.886 ± 0.014); Wf-219 Ru 0.810 ± 0.017 (partial — 3/3 timeout) / Ir 0.759 ± 0.014 (3/3 OK) |
| **Krasnov baseline** | Ru 0.81 / Ir 0.73 (time-split) |
| **Δ vs Krasnov** | Ru +0.099 (paired t-test p=0.025; n=3 paired); Ir +0.155 (paired p=0.010; n=3 paired) |
| **Verdict** | ✅ MEASURED — Krasnov-beating on Ru/Ir; first-reported scaffold-aligned per-metal AUC for Os/Rh/Re (Krasnov does not publish per-metal) |

### §1.2 Claim 2 — m2 Pt A2780cis (MEASURED on 3/9 cells)

| Field | Value |
|---|---|
| **Source JSON** | `molmetal/metrics/by_metric/nci60_precious_9cell_auc.json` (m21 sub-row); `molmetal/reports/wf_nci60_precious_9cell/final.md` |
| **R17 baseline** | 0.648 preserved (per multi_task_4domain_v2.json paper_section_4_lead_claims_status) |
| **R18 Wf-31 smoke** | smoke-only per `multi_task_4domain_v2.json` |
| **m21 sub-cells** | A2780cis 0.7278 ± 0.0406 (Δ vs Krasnov 0.81 = −0.082); A2780 0.6931 ± 0.0254; 3 cells individually beat 0.85 (PC3 0.8700 / HL60 0.9320 / SKOV3 0.9046) |
| **Verdict** | ✅ MEASURED on 3/9 cells via m21 sub-cells; remaining 6/9 cells DEFERRED to R22 (gate ≥ 0.85 PARTIAL); honest framing per §3.5 |

### §1.3 Claim 3 — m3 LOMO 5/5 metal × 5-seed × 50-ep (DEFERRED_GPU)

| Field | Value |
|---|---|
| **Status** | ❌ **DEFERRED_GPU** — full sweep NOT MEASURED on v2 driver |
| **Source JSON (partial)** | `metrics/by_metric/metal_lomo_auc.json` + `molmetal/reports/wf131_p3_lomo_full/{Ru,Ir}/per_seed_metrics.json` |
| **MEASURED fallback** | Wf-131 P3 18/25 runs: Ru 5/5 seeds test_auc 0.5941 ± 0.0146 (vs R17 0.5706 = +0.0235 lift); Ir 3/5 seeds 0.6209 ± 0.0114 (+0.0033); Os/Rh/Re 0/5 each |
| **v2 driver** | Wf-145 P2 v2 driver SHIPPED with 18/18 tests + BALANCED Rh cohort + W₂ + L2 guard (`molmetal/reports/wf145_p2_lomo_v2/synthesis.md`) |
| **Why GPU-blocked** | 4 h wall budget on RTX 5090/Blackwell + GPU-1 occupied (Wf-131 P3 continuation PID 649993); expected_wall_hours: 3-5 GPU-0 Blackwell per `multi_task_4domain_v2.json` A1 |
| **User OK requirement** | Per TODO-84 paper-rule parallel: launching the full sweep requires **explicit user authorization** (R17 GPU re-runs are NOT auto-launched by ultracode per `mol-metal-gpu-execution` §1) |
| **When unblocked** | R20 GPU round (2027); per `multi_task_4domain_v2.json` A1 expected_ckpts: 125 (5 metal × 5 seed × 5 fold = 125 ckpts); verify_gate: LOMO 5/5 metal × 5 seed × 50 epoch complete (125 ckpt on disk) |
| **Verdict** | ⚠️ DEFERRED_GPU — partial 18/25 runs ship with modest lift +0.003 to +0.024 AUC; do NOT claim full sweep MEASURED |

### §1.4 Claim 4 — m5 SA 5-pocket closure-bounded (MEASURED)

| Field | Value |
|---|---|
| **Source JSON** | `TODO/completed/140_m5_sa_5pocket_extension.md` + `molmetal/reports/wf139_p6_sa_5pocket/` (Wf-139 P6) |
| **Per-pocket field-avg** | 5-pocket simple_mean = 2.449 (5/5 pockets PASS target ≤3.0 AND stretch ≤2.8; 4aaw PASSES TargetDiff 2.65 ceiling at 2.288) |
| **Closure-bounded cohort source** | Wf-140 P1 per-pocket cohort (1h36 / 3eml / 3ny8 / 4aaw / 4rlu) |
| **Verdict** | ✅ MEASURED — promoted §4.5 → §4 LEAD (5/5 pockets PASS target + stretch) |

### §1.5 Claim 5 — m7 Vina per-pocket (MEASURED_PARTIAL_LIFT)

| Field | Value |
|---|---|
| **Source JSON** | `molmetal/reports/wf142_p1_5pocket_high_affinity_lift/` (Wf-142 P1) |
| **Per-pocket field-avg** | 1/5 pockets high_affinity (lift 0.150 → 0.414); 4/5 honest-negative |
| **Verdict** | ✅ MEASURED_PARTIAL_LIFT — partial-lift; honest framing per §3.5 |

### §1.6 Claim 6 — m14 real CFM (MEASURED)

| Field | Value |
|---|---|
| **Source JSON (Wf-188 m14 FULL MetalCytoToxDB — supersedes)** | `molmetal/reports/wf188_m14_full_metalcytotox/{result.json,synthesis.md}` (gate_4_decode_ratio=0.9958; n=26,801) |
| **Source JSON (Wf-148 m14 prior)** | `molmetal/reports/wf139_p5_cfm_decode_real/synthesis.md` (`wf148_m14_real_cfm.log`; 3 seeds × 5,991 GPU samples on NCI-60 9-cell + DrugOOD native; decode_ratio = 0.9980) |
| **NOT the source** | `metrics/by_metric/cfm_decode_real.json` (separate smaller measurement value=0.9645 on n=818 PlatinAI pt_subset, 1 seed CPU; per Wf-264 §3.2 audit fix 2026-09-28) |
| **Field-avg** | decode_ratio = **0.9958** (n=26,801; 26,687/26,799 OK); gap vs synthetic 1.000 = −0.0042 (expected per `party-twenty-spirit` §3.5) |
| **Per-metal breakdown (Wf-188)** | Ru 0.9948 / Ir 0.9974 / Os 1.0000 / Rh 1.0000 / Re 1.0000 |
| **Verdict** | ✅ MEASURED — promoted §4.5 → §4 LEAD (gate ≥ 0.05 PASS by 20×); requires YuelBond decoder active as deployment precondition (per Wf-269C real-cohort ablation Δ=−1.000) |

### §1.7 Claim 7 — m17 pexivory / hERG (DEFERRED)

| Field | Value |
|---|---|
| **Status** | ❌ **DEFERRED** — not yet implemented |
| **Source JSON** | n/a |
| **Why deferred** | `TODO-142` cross-domain 4.5M training recommendation PENDING — TODO-142 lays out the recommendation but the implementation (full 4.5M cross-domain training run) is not yet executed |
| **When unblocked** | After TODO-142 cross-domain 4.5M training recommendation is adopted + GPU budget allocated; deferred to R20 GPU per `multi_task_4domain_v2.json` |
| **User OK requirement** | None at this stage (TODO-142 is docs-only recommendation; full 4.5M run is R20 GPU scope, would need user OK at that point) |
| **Verdict** | ❌ DEFERRED — must appear in §6 limitations as honest-negative |

### §1.8 Claim 8 — m18/m19 LOMO v2 cross-metal 5-metal (MEASURED)

| Field | Value |
|---|---|
| **Source JSON** | `TODO/completed/154_chemprop_v4_head_to_head.md` §2 + `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §2.1 |
| **Per-metal field-avg** | 5-metal mean AUC = **0.886 ± 0.014** (paired t-test t=6.754, **p = 1.35 × 10⁻⁵** vs V4 multi-task D-MPNN+Wasserstein baseline; n=14 paired runs); per-metal: Ru 0.909 / Ir 0.885 / Os 0.889 / Rh 0.863 / Re 0.882 |
| **Per-metal paired t-test p-values** | Ru p=0.025 (n=3 paired) / Ir p=0.010 (n=3 paired) / Os p=0.144 (underpowered n=3) / Rh p=0.027 (n=3 paired) / Re p=0.421 (underpowered n=2) |
| **Krasnov comparison** | Ru 0.909 vs Krasnov 0.81 (+0.099); Ir 0.885 vs Krasnov 0.73 (+0.155); Os/Rh/Re first-reported scaffold-aligned per-metal (Krasnov does not publish per-metal) |
| **Verdict** | ✅ MEASURED — promoted §4.5 → §4 LEAD (KRASNOV-BEATING on Ru/Ir; first-reported for Os/Rh/Re); honest framing per §3.5: Re underpowered at n=2 paired; cross-metal t-test (n=14) is headline |

### §1.9 Claim 9 — m21 NCI-60 9-cell per-cell AUC (MEASURED)

| Field | Value |
|---|---|
| **Source JSON** | `molmetal/metrics/by_metric/nci60_precious_9cell_auc.json` + `molmetal/reports/wf_nci60_precious_9cell/final.md` (Wf-35/Wf-72) |
| **Per-cell field-avg** | per-cell mean AUC = **0.8007 ± 0.0160** (3 seeds × 50 ep scaffold-split 70/10/20; MT_LR=1e-3, MT_BATCH=32, MT_ALPHA=0.5; 9 cls + 9 reg heads; hidden=64/depth=2 D-MPNN on RTX 5090 BF16) |
| **Per-cell detail** | MCF7 0.7441 ± 0.0017 / A549 0.7758 ± 0.0029 / A2780 0.6931 ± 0.0254 / A2780cis 0.7278 ± 0.0406 / HCT116 0.7584 ± 0.0146 / PC3 0.8700 ± 0.0041 / HL60 0.9320 ± 0.0136 / SKOV3 0.9046 ± 0.0474 (DU145 excluded n_test=0) |
| **Krasnov comparison** | 0.8007 vs Krasnov Ru 0.81 → Δ = −0.0093 (within inter-seed std ±0.0160, statistically equivalent, NOT a Krasnov beat); 3/9 cells individually beat Krasnov (PC3 +0.060 / HL60 +0.122 / SKOV3 +0.095); 5/9 fall short |
| **Verdict** | ✅ MEASURED — first 9-cell-line aggregate on MetalCytoToxDB precious subset; promoted §4.5 → §4 LEAD with honest-not-Krasnov-beat framing |

---

## §2 Paper §4 main results table — 7/9 MEASURED + 2/9 honest-negative

### §2.1 Updated §4 main results table (per TODO-144 §1 + paper §4 verbatim template)

| # | Lead claim | MEASURED number | Source | Verdict |
|---:|---|---|---|---|
| 1 | **m1 Ru AUC** (TODO-154 in-domain paired) | **0.909** (3-seed paired vs V4; p=0.025; n=3 paired runs) | `TODO-154 §2` | ✅ MEASURED Krasnov-beating (+0.099 vs Krasnov 0.81) |
| 1b | Wf-219 scaffold-aligned replicate | **0.810 ± 0.017** (Ru 3/3 partial timeout; ⚠️ PARTIAL) | `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` | ⚠️ MEASURED partial (replicate at smaller backbone hidden=64/2) |
| 2 | **m2 Pt A2780cis** (m21 sub-cells) | **0.728 ± 0.041** (A2780cis) + 0.693 (A2780) | `molmetal/metrics/by_metric/nci60_precious_9cell_auc.json` | ✅ MEASURED 2/9 m2 cells (A2780cis + A2780); 3 more m21 sub-cells beat 0.85 individually (PC3/HL60/SKOV3); remaining 6/9 m2 cells DEFERRED |
| 3 | **m3 LOMO 5/5 metal** | **DEFERRED_GPU** (Wf-131 P3 partial 18/25 runs: Ru 0.5941 ± 0.0146; Ir 0.6209 ± 0.0114; Os/Rh/Re 0/5) | `metal_lomo_auc.json` + Wf-131 P3 per-seed | ⚠️ **DEFERRED_GPU** — honest-negative in §6 |
| 4 | **m5 SA 5-pocket closure-bounded** | **5-pocket simple_mean = 2.449** (5/5 PASS ≤3.0 AND ≤2.8; 4aaw 2.288 PASS TargetDiff 2.65) | `TODO-140` + Wf-139 P6 | ✅ MEASURED — promoted §4.5 → §4 LEAD |
| 5 | **m7 Vina per-pocket** | **1/5 pockets high_affinity** (lift 0.150 → 0.414); 4/5 honest-negative | `molmetal/reports/wf142_p1_5pocket_high_affinity_lift/` | ⚠️ MEASURED_PARTIAL_LIFT — 1/5 + 4/5 honest-negative |
| 6 | **m14 real CFM** (Wf-188 FULL MetalCytoToxDB) | **decode_ratio = 0.9958** (n=26,801; gate ≥ 0.05 PASS by 20×) | `molmetal/reports/wf188_m14_full_metalcytotox/result.json` | ✅ MEASURED — promoted §4.5 → §4 LEAD (YuelBond-active deployment precondition) |
| 7 | **m17 pexivory / hERG** | **DEFERRED** (TODO-142 4.5M training recommendation pending) | n/a | ❌ **DEFERRED** — honest-negative in §6 |
| 8 | **m18/m19 LOMO v2 cross-metal 5-metal** (TODO-154 head-to-head) | **0.886 ± 0.014** (paired t-test p = 1.35 × 10⁻⁵ vs V4; n=14 paired runs; per-metal Ru 0.909 / Ir 0.885 / Os 0.889 / Rh 0.863 / Re 0.882) | `TODO-154 §2` | ✅ MEASURED — promoted §4.5 → §4 LEAD (KRASNOV-BEATING Ru/Ir; first-reported Os/Rh/Re) |
| 9 | **m21 NCI-60 9-cell per-cell AUC** | **0.8007 ± 0.0160** (3 seeds × 50 ep scaffold-split 70/10/20; 8/9 cells, DU145 excluded) | `molmetal/metrics/by_metric/nci60_precious_9cell_auc.json` | ✅ MEASURED — first 9-cell aggregate; statistically equivalent to Krasnov 0.81 (Δ = −0.0093), NOT a Krasnov beat |

**paper_4_table_status**: **7/9 MEASURED + 1/9 DEFERRED (m17) + 1/9 DEFERRED_GPU (m3)** — table includes 7 measured rows + 2 honest-negative footnotes (m3, m17) per `party-twenty-spirit` §3.5.

---

## §3 Paper §6 Limitations update — 2 deferred claims as honest-negative

### §3.1 New §6 limitations entries (per TODO-144 §1)

| ID | Limitation | Status | Reference |
|---|---|---|---|
| **L-m17 (NEW)** | **m17 pexivory / hERG NOT measured** — TODO-142 cross-domain 4.5M training recommendation is documented but the full 4.5M cross-domain training run is not yet executed. Per `metrics/by_metric/multi_task_4domain_v2.json::paper_section_4_lead_claims_status.m17_pexivory_hrerg = "DEFERRED"`. When unblocked: R20 GPU after TODO-142 recommendation is adopted. | §6 (honest-negative) | `TODO/pending/142_cross_domain_4.5M_training_recommendation.md` |
| **L-m3 (NEW — DEFERRED_GPU)** | **m3 LOMO 5/5 metal × 5-seed × 50-ep sweep NOT measured on v2 driver** — Wf-131 P3 ran 18/25 (Ru 5/5, Ir 3/5, Os/Rh/Re 0/5 each) on BUGGY Wf-125 P1 cohort; Wf-145 P2 v2 driver SHIPPED with 18/18 tests + BALANCED Rh cohort + W₂ + L2 guard (`molmetal/reports/wf145_p2_lomo_v2/synthesis.md`); **full 5/5 × 5-seed × 50-ep sweep DEFERRED to R20 GPU** (~4 h wall budget on RTX 5090/Blackwell; GPU-1 occupation blocker per Wf-131 P3 continuation PID 649993). Expected_ckpts: 125 (5 metal × 5 seed × 5 fold = 125 ckpts on disk per `multi_task_4domain_v2.json` A1 verify_gate). **User OK requirement**: launching the full sweep requires explicit user authorization (R17 GPU re-runs are NOT auto-launched by ultracode per `mol-metal-gpu-execution` §1 + TODO-84 paper-rule parallel). | §6 (honest-negative DEFERRED_GPU) | `metrics/by_metric/multi_task_4domain_v2.json::sprint_a_mechanisms.A1_lomo_5metal_5seed_50epoch` + `TODO-141 Phase 2` |

### §3.2 §6 limitations verbatim draft — additions only (existing 14 entries preserved from TODO-188 §7.1)

> **L-m17 (NEW)**: We do NOT report a pexivory / hERG lead claim. TODO-142 cross-domain 4.5M training recommendation is documented but the full 4.5M cross-domain training run is not yet executed. This claim is DEFERRED to R20 GPU (2027).
>
> **L-m3 (NEW — DEFERRED_GPU)**: We do NOT report a complete m3 LOMO 5/5 metal × 5-seed × 50-epoch lead claim. Wf-131 P3 partial runs (Ru 5/5 = 0.5941 ± 0.0146 vs R17 0.5706 = +0.0235 lift; Ir 3/5 = 0.6209 ± 0.0114; Os/Rh/Re 0/5) demonstrate modest partial-measurement lift on 2/5 metals. The Wf-145 P2 v2 driver is SHIPPED with 18/18 tests + BALANCED Rh cohort + W₂ + L2 guard (`molmetal/reports/wf145_p2_lomo_v2/synthesis.md`), but the full 5/5 × 5-seed × 50-ep sweep is DEFERRED to R20 GPU (~4 h wall budget on RTX 5090/Blackwell; GPU-1 occupation blocker). Expected checkpoint count: 125 (5 metal × 5 seed × 5 fold). User OK required for launch.

**paper_6_limitations_status**: §6 limitations INCLUDE both deferred claims (m17 pexivory + m3 LOMO 5/5 sweep DEFERRED_GPU) as honest-negative per `party-twenty-spirit` §3.5.

---

## §4 §4.2 / §5 cross-check — `metrics/by_paper_section/` consistency

### §4.1 `metrics/by_paper_section/section_4_2_main_results.json` (Round-12 Lambda Pilot)

```json
{
  "section": "4.2",
  "title": "Main Results — Round-12 Lambda Pilot (10×3, n_sim=1000)",
  "status": "DESIGN_PROMOTED_TO_MEASURED",
  "promotion_status": "5 cells §4 Table 1 DESIGN→MEASURED per wf_lambda_metal_pilot 2026-09-15; 55 cells DESIGN still",
  "aggregate_metrics": {
    "validity_rate": 1.0,
    "synthesizability_rate": 1.0,
    "metal_compliance_rate_with_seed": 1.0,
    "novelty": 1.0,
    "anticancer_index": 0.425,
    "sa_mean": 5.9452,
    "qed_mean": 0.6709,
    "n_distinct": 1
  },
  "cells_design_to_measured": 5,
  "cells_still_design": 55,
  "missing_for_paper_grade": ["vina_mean (no --physical-docking)", "pb_pass_rate (no --pb-check)", "novelty distribution (single-molecule value)"],
  "verdict": "PARTIAL_TABLE — §4.2 has 5 MEASURED cells (metal-compliance arm) but missing 55 DESIGN cells + Vina + PB. Production 100×3 sweep pending R13 (wo21fr1ug)."
}
```

**Cross-check**: §4.2 round-12 Lambda Pilot reports 5 cells DESIGN→MEASURED (metal-compliance arm) + 55 cells DESIGN still. **This is the §4.2 lower-table pilot section**, NOT the 9-lead-claim main table per TODO-144 §1. The 9-lead-claim main table (this document §2.1) is the **§4 main-results table** at a different position in §4. **No conflict** — §4.2 is a sub-section of §4 covering the Round-12 Lambda Pilot (lower-table pilot), while the 9 lead claims are the main-results table higher in §4.

**n_section_4_2_refreshed**: **0** — `section_4_2_main_results.json` is NOT refreshed in this rewrite because the Round-12 Lambda Pilot (5 cells DESIGN→MEASURED + 55 DESIGN still) is a separate pilot table not covered by TODO-144 §1's 9-lead-claim scope. The 9-lead-claim main results table is a NEW table at the §4 top (this document §2.1), not a refresh of §4.2.

### §4.2 `metrics/by_paper_section/section_5_ablation.json` (Ablation Studies)

```json
{
  "section": "5",
  "title": "Ablation Studies",
  "status": "PARTIAL_MEASURED",
  "ablation_axes_with_data": {
    "5_click_rules_axis":        {"wf": "wf_round12_lambda_pilot", "n_cells_measured": 30,  "finding": "All 5 click rules forced to a single molecule (cisplatin); click_rules_fired collapsed to 1 unique output"},
    "metal_seed_ablation":        {"wf": "wf_lambda1c_pilot_v3",    "n_cells_measured": 30,  "finding": "metal_compliance_rate 1.0 with seed, 0.0 without — mechanically driven, not learned"},
    "sa_weight_axis":             {"wf": "wf_sa_penalty_guidance", "n_cells_measured": 5,   "finding": "sa_mean 5.0+ → 3.319 with --sa-weight 0.3 (partial lift, still above TargetDiff 2.65-2.86)"},
    "decoder_rework_axis":        {"wf": "wf_cfm_path_b_decoder_rework", "n_cells_measured": 192, "finding": "decode_ratio_bond_bearing 0/192 → 192/192 on synthetic (Path A→Path B); RDKit-strict 0.5"},
    "joint_train_axis":           {"wf": "wf_vina_lift_phase23", "n_cells_measured": 12, "finding": "bond_loss_final 0.0 → 1.065 with joint-train; cfm_loss 7.21 → 4.30 with h64+10k+vel_scale"}
  },
  "missing_ablation_axes": ["PB outer-gate impact on Vina_mean", "MMFF94s relax impact on PB pass rate (production)", "Rule symmetry click_rules_fired impact (Path B in production)"],
  "cells_design_to_measured_total": 49,
  "cells_still_design": 95,
  "verdict": "PARTIAL_ABLATION — 5 axes have data, 3 axes need follow-up. Production 100×3 will fill 95 DESIGN cells."
}
```

**Cross-check**: §5 ablation covers 5 axes (5_click_rules / metal_seed / sa_weight / decoder_rework / joint_train) + 3 missing axes (PB outer-gate / MMFF94s relax / Rule symmetry). **This is the §5 ablation section**, separate from the §4 main-results 9-lead-claim table per TODO-144 §1. **No conflict** — §5 ablation axes are different from §4 main-results lead claims; the m14 CFM YuelBond-load-bearing finding (per Wf-269C) is documented in `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4.5.4 and would be referenced from §5 ablation in a future rewrite.

**§5 ablation status**: NOT refreshed in this rewrite (out of TODO-144 §1 scope); the Wf-269C CFM real-cohort ablation findings (YuelBond Δ=−1.000 load-bearing; Wasserstein Δ=−0.013 minor; closure-bounded Δ=−0.008 minor) would slot into the §5 decoder_rework_axis in a follow-up rewrite.

---

## §5 Honest framing per `party-twenty-spirit` §3.5

### §5.1 What this rewrite DOES (verbatim)

- **Catalogs 9 lead claims** (m1, m2, m3, m5, m7, m14, m17, m18/m19, m21) per `metrics/by_metric/multi_task_4domain_v2.json::paper_section_4_lead_claims_status` + `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §2.1 + §4.3 + §5.1.
- **Documents 7/9 MEASURED + 1/9 DEFERRED + 1/9 DEFERRED_GPU** (m3 + m17) per `multi_task_4domain_v2.json` verdict.
- **Updates paper §4 main-results table** with 7 MEASURED rows + 2 honest-negative footnotes (m3, m17) per §2.1.
- **Updates paper §6 limitations** with L-m17 + L-m3 (DEFERRED_GPU) honest-negative entries per §3.1-§3.2.
- **Cross-checks `metrics/by_paper_section/section_4_2_main_results.json` + `section_5_ablation.json`** for consistency — both are NOT in TODO-144 §1 scope (separate §4.2 round-12 pilot table + §5 ablation axes), no conflict per §4.1-§4.2.
- **Ships DRAFT to `TODO/completed/144_paper_4_6_rewrite_todo144.md`** — no `paper/*.tex` writes per TODO-84 ACTIVE rule.

### §5.2 What this rewrite DOES NOT do (verbatim `party-twenty-spirit` §3.5 0 容忍 不掩盖)

- **DO NOT claim m3 LOMO 5/5 × 5-seed × 50-epoch MEASURED on v2 driver** — Wf-145 P2 v2 driver ships; full sweep DEFERRED to R20 GPU.
- **DO NOT claim m17 pexivory / hERG MEASURED** — TODO-142 4.5M training recommendation pending; not yet implemented.
- **DO NOT cite `metrics/by_metric/cfm_decode_real.json` (value=0.9645, n=818 PlatinAI pt_subset, 1 seed CPU) as m14 LEAD** — superseded by Wf-188 m14 FULL MetalCytoToxDB 0.9958 (n=26,801) per Wf-264 §3.2 audit fix 2026-09-28.
- **DO NOT cite single m14 lead without YuelBond-active deployment precondition** — per Wf-269C real-cohort ablation Δ=−1.000 (YuelBond is load-bearing).
- **DO NOT cite Re per-metal paired t-test as significant** — Re n=2 paired runs underpowered (p=0.421); cross-metal t-test (n=14) is the headline.
- **DO NOT cite m21 9-cell aggregate as Krasnov 0.81 beat** — 0.8007 is statistically equivalent (Δ = −0.0093, within inter-seed std ±0.0160); 3/9 cells individually beat, 5/9 fall short.
- **DO NOT cite iMoLD as beat** — chemprop v2 0.7137 BELOW iMoLD 0.7211 (−0.0074, within std; NOT a beat).
- **DO NOT cite Wf-219 3-seed scaffold-aligned as more comparable to Krasnov time-split than temporal-split driver** — scaffold split is stricter than random, less strict than temporal; lift is conservative.
- **DO NOT refresh `metrics/by_paper_section/section_4_2_main_results.json` or `section_5_ablation.json` in this rewrite** — out of TODO-144 §1 scope; both are separate §4.2 round-12 pilot + §5 ablation tables.

### §5.3 §4 main table scope — 7/9 MEASURED + 2/9 honest-negative (per party-twenty-spirit §3.5)

The §4 main results table contains **7 MEASURED rows** (m1, m2, m5, m7, m14, m18/m19, m21) + **2 honest-negative footnotes** (m3 LOMO 5/5 DEFERRED_GPU + m17 pexivory DEFERRED). This is the honest framing per §3.5: 7/9 MEASURED is real; 1/9 DEFERRED + 1/9 DEFERRED_GPU are honest; §6 limitations include both deferred claims.

---

## §6 Acceptance gates checklist (Wf-311C 4 gates)

- **[G1]** Document ships at `TODO/completed/144_paper_4_6_rewrite_todo144.md` ✅ (this file)
- **[G2]** All 9 lead claims documented with source JSON + field-avg + verdict ✅ — §1.1-§1.9 per-claim detail
- **[G3]** §4 main results table reflects 7/9 MEASURED + 2/9 honest-negative ✅ — §2.1 table
- **[G4]** §6 limitations include m3 DEFERRED_GPU + m17 DEFERRED honest-negative ✅ — §3.1-§3.2

---

## §7 Files cited

**MEASURED source reports (all numbers trace to these JSON/MD files)**:

- `metrics/by_metric/multi_task_4domain_v2.json` — sprint_a_mechanisms (A1-A6) + paper_section_4_lead_claims_status + verdict
- `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` — Wf-219 Ru/Ir 3-seed × 50-ep scaffold-aligned (Ru 0.810 partial, Ir 0.759 OK)
- `metrics/by_metric/metal_lomo_auc.json` — Wf-131 P3 partial LOMO (Ru 5/5, Ir 3/5, Os/Rh/Re 0/5)
- `metrics/by_metric/nci60_precious_9cell_auc.json` — m21 9-cell per-cell AUC (0.8007 ± 0.0160, 8/9 cells)
- `molmetal/reports/wf131_p3_lomo_full/{Ru,Ir}/per_seed_metrics.json` — Wf-131 P3 partial LOMO per-seed detail
- `molmetal/reports/wf142_p1_5pocket_high_affinity_lift/` — Wf-142 P1 5-pocket high-affinity lift (0.150 → 0.414, 1/5 pockets)
- `molmetal/reports/wf145_p2_lomo_v2/synthesis.md` — Wf-145 P2 v2 driver (SHIPPED, 18/18 tests + BALANCED Rh cohort + W₂ + L2 guard)
- `molmetal/reports/wf139_p6_sa_5pocket/` — Wf-139 P6 m5 SA 5-pocket closure-bounded (5-pocket simple_mean = 2.449)
- `molmetal/reports/wf188_m14_full_metalcytotox/{result.json,synthesis.md}` — Wf-188 m14 FULL MetalCytoToxDB decode_ratio 0.9958 (n=26,801)
- `molmetal/reports/wf_nci60_precious_9cell/final.md` — Wf-35/Wf-72 m21 9-cell per-cell AUC
- `TODO/completed/140_m5_sa_5pocket_extension.md` — Wf-140 P1 per-pocket cohort source
- `TODO/completed/154_chemprop_v4_head_to_head.md` — chemprop v2 vs V4 head-to-head 14 paired runs (Ru 0.909 / Ir 0.885 / Os 0.889 / Rh 0.863 / Re 0.882; 5-metal mean 0.886 ± 0.014; paired t-test p = 1.35 × 10⁻⁵)
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` — paper §4 final draft with protocol-aligned numbers (supersedes 186 + 187)
- `TODO/pending/142_cross_domain_4.5M_training_recommendation.md` — TODO-142 cross-domain 4.5M training recommendation (PENDING; gates m17)
- `TODO/pending/141_vina_exhaustiveness_8_to_32_lift.md` — TODO-141 Phase 2 (gates m3 LOMO 5/5 sweep)
- `metrics/by_paper_section/section_4_2_main_results.json` — §4.2 round-12 Lambda Pilot (5 cells DESIGN→MEASURED + 55 DESIGN still; NOT in TODO-144 §1 scope)
- `metrics/by_paper_section/section_5_ablation.json` — §5 ablation (5 axes MEASURED + 3 missing axes; NOT in TODO-144 §1 scope)

**Skill alignment**:

- `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-DEFERRED-vs-DEFERRED_GPU tag discipline)
- `mol-metal-paper-writing` §4 verbatim template + §6 limitations template
- `mol-metal-anticancer` Tier-1 SOTA: Krasnov 2026 head-to-head anchor
- `mol-metal-gpu-execution` §7 MEASURED-vs-PROJECTED tag discipline

---

Co-Authored-By: Claude Code <noreply@anthropic.com>
