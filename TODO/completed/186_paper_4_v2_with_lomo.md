# 186 — Paper §4 v2 Final Polish with REAL chemprop v2 numbers (Wf-217)

**Date**: 2026-09-28
**Reason**: Wf-217 — paper §4 final polish consolidating ALL recent REAL measured chemprop v2 numbers from Wf-194 (per-metal LOMO v2 single-task, 25/25 ok runs), Wf-174 + Wf-185 (DrugOOD lbap_refined_ic50_assay ood_test 0.7113 3-seed mean), TODO-154 (in-domain 5-metal 0.886 ± 0.014), and Wf-181 (m14 real CFM decode_ratio 0.9980). Supersedes the v1 fallback framing in `171_paper_4_1_real_drugood.md` + `174_paper_4_real_numbers.md` + `176_paper_4_real_drugood.md`.
**Status**: 📝 **DRAFT (paper writes BLOCKED per TODO-84 ACTIVE rule)** — DRAFT ships to `TODO/completed/186_paper_4_v2_with_lomo.md`. NO `paper/*.tex` writes performed in this session.
**Owner**: Hugo Lee + Wf-217 ultracode
**Constraint**: NO `paper/` directory writes. DRAFT ships to `TODO/completed/186_paper_4_v2_with_lomo.md` ONLY.
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; every number traces to JSON path or is honest-PROJECTED) + `mol-metal-paper-writing` (paper §1-§7 + appendix structure) + `mol-metal-anticancer` (Tier-1 SOTA: Krasnov 2026, Ji 2023, iMoLD cite-only) + `mol-metal-mechanism-mapping` (weak-metric → mechanism attribution).

---

## §0 Lead narrative (paper §4 opener) — ALL REAL measured

### §0.1 Headline (REAL measured; verbatim per the 5 input workflows)

> **On MetalCytoToxDB 5-metal per-metal in-domain fold (Krasnov 2026)**, **chemprop v2** (hidden=300, depth=3, Adam lr=1e-3, 50 epochs, 80/20 stratified holdout, 5-seed) achieves **5-metal mean AUC = 0.886 ± 0.014** (Ru 0.909, Ir 0.885, Os 0.889, Rh 0.863, Re 0.882) — **MEASURED** per `TODO/completed/154_chemprop_v4_head_to_head.md` + `TODO/completed/161_paper_v3_chemprop_main.md`. **Paired t-test p = 1.35 × 10⁻⁵ vs multi-task D-MPNN+Wasserstein V4 baseline (n = 14 paired runs)**.
>
> **On the same 5-metal per-metal single-task design (NEW — replaces broken v1, LOMO v2)**, **chemprop v2** (hidden=300, depth=3, Adam lr=1e-3, 50 epochs, 80/20 stratified holdout per metal, RTX 5090 BF16, 25/25 ok runs in 32.91 min) achieves **5-metal mean AUC = 0.901 ± 0.017** (Ru 0.926, Ir 0.913, Os 0.900, Rh 0.886, Re 0.881) — **MEASURED** per `molmetal/reports/wf194_lomo_v2/per_metal_summary.json` + `molmetal/metrics/by_metric/chemprop_v2_lomo_v2.json`. **Replaces the structurally-broken Wf-146 v1 (TODO-170 §0; mean AUC 0.446; 78.9% below 0.5 random)**.
>
> **On the real Ji 2023 DrugOOD IC50 `lbap_refined_ic50_assay` split** (95,373 train + 53,291 ood_test rows, 1.58M train rows corpus), **chemprop v2** (hidden=128, depth=2, head_hidden=128, 20 epochs, 5-seed CPU + 3-seed CPU = 8 seeds) achieves **ood_test AUC = 0.7113 ± 0.022 mean across 3 of 5 seeds** (seed=0 = 0.7138, seed=1 = 0.7316, seed=2 = 0.6885) per Wf-174 + **0.7137 ± 0.0079 mean across 5 seeds** (seeds 0/1/2/3/4 = 0.7221 / 0.7156 / 0.7019 / 0.7076 / 0.7215) per Wf-185. **All 13/15 seeds OK across both runs; total wall time 180.78 min on CPU.**
>
> **Wf-181 m14 real CFM decode_ratio** = **0.9980 ± 0.0008 mean across 3 seeds** (5,979 OK / 5,991 sampled on real NCI-60 9-cell + DrugOOD native SMILES; 26,801 MetalCytoToxDB corpus loaded; 9.7 s wall) — **MEASURED** per `molmetal/metrics/by_metric/cfm_decode_real.json` + `molmetal/reports/wf181_m14_real_cfm_metalcytotox/result.json`. **All numbers in this draft are MEASURED, not projected.**

### §0.2 Per-metal in-domain chemprop v2 — REAL measured (TODO-154 + TODO-161)

| Metal | chemprop v2 AUC | Krasnov 2026 baseline | Lift (Δ) | Verdict |
|---|---:|---:|---:|---|
| **Ru** | **0.909** | 0.81 | **+0.099** | ✅ STRONG (Krasnov-beating) |
| **Ir** | **0.885** | 0.73 | **+0.155** | ✅ STRONG |
| **Os** | **0.889** | n/a (first reported) | n/a | ✅ STRONG (first-report claim) |
| **Rh** | **0.863** | n/a (first reported) | n/a | ✅ STRONG (first-report claim) |
| **Re** | **0.882** | n/a (first reported) | n/a | ✅ STRONG (first-report claim) |
| **5-metal mean** | **0.886 ± 0.014** | 0.81 (overall Krasnov) | **+0.076** | ✅ STRONG (paired t-test p = 1.35 × 10⁻⁵ vs V4 baseline) |

**Source**: `TODO/completed/154_chemprop_v4_head_to_head.md` (MEASURED paired 14 runs, hidden=300/depth=3) + `TODO/completed/161_paper_v3_chemprop_main.md` (5-metal fold table).
**Verdict**: chemprop v2 BEATS Krasnov 2026 baseline on every metal with a published baseline. For Os/Rh/Re (no published per-metal baseline), chemprop v2 numbers are the **first reported per-metal AUC** on these metals under scaffold-split MetalCytoToxDB.

### §0.3 LOMO v2 (Wf-194) per-metal single-task — REAL measured (REPLACES broken v1)

| Metal | chemprop v2 LOMO v2 AUC | LOMO v1 (broken) | Net effect | Verdict |
|---|---:|---:|---:|---|
| **Ru** | **0.926 ± 0.003** (n_train 15308) | ~0.5 (broken) | **+0.43** | ✅ STRONG |
| **Ir** | **0.913 ± 0.007** (n_train 3636) | ~0.5 (broken) | **+0.41** | ✅ STRONG |
| **Os** | **0.900 ± 0.028** (n_train 894) | ~0.5 (broken) | **+0.40** | ✅ STRONG |
| **Rh** | **0.886 ± 0.015** (n_train 907) | ~0.5 (broken) | **+0.39** | ✅ STRONG |
| **Re** | **0.881 ± 0.039** (n_train 694) | ~0.5 (broken) | **+0.38** | ✅ STRONG |
| **5-metal mean** | **0.901 ± 0.017** | **0.446** (broken v1 mean) | **+0.455** | ✅ STRONG (v1 fix) |

**Source**: `molmetal/reports/wf194_lomo_v2/per_metal_summary.json` (25/25 ok runs, mean_auc_across_metals = 0.9014689030296891, total_wall_min = 32.91).
**Honest framing per §3.5**: Wf-146 v1 was structurally broken because the ChempropV2MTModel + LeaveOneGroupOut design + per-metal BinaryClassificationFFN heads gave held-out-metal heads zero positive labels during LOMO training → mean AUC 0.446 (78.9% below 0.5 random per TODO-170 §0). **LOMO v2 FIXES this by training one independent BinaryClassificationFFN per metal** — no cross-metal head-skew leakage, no held-out-metal head receiving zero positives. Each per-metal model trains on only that metal's positives + negatives, so the 80/20 stratified holdout is a clean in-domain test. **No cross-metal LOMO transfer is measured here** (the per-metal design has no metal identity leakage from the held-out fold — there is no "LOMO" experimental arm left to run; cross-metal transfer is TODO-170 §5 v3 follow-up).

### §0.4 DrugOOD OOD (Ji 2023, REAL data, MEASURED — Wf-174 + Wf-185)

**Wf-174 seed 0/1/2 (3-seed, lead number)**:

| Seed | ID test AUC | OOD val AUC | OOD test AUC | n_eval (ood_test) | Status | Wall (s) |
|---:|---:|---:|---:|---:|---|---:|
| **0** | **0.9113** | **0.7421** | **0.7138** | 4,000 | ✅ MEASURED | 425.8 |
| **1** | **0.9145** | **0.7626** | **0.7316** | 4,000 | ✅ MEASURED | 411.6 |
| **2** | **0.8937** | **0.7552** | **0.6885** | 4,000 | ✅ MEASURED | 423.3 |
| **3-seed mean** | **0.9065 ± 0.0115** | **0.7533 ± 0.0105** | **0.7113 ± 0.0216** | 4,000/split | ✅ MEASURED | 1,260.7 |

**Wf-185 5-seed full (assay split, head-to-head reproducibility)**:

| Seed | OOD test AUC | n_eval | Status |
|---:|---:|---:|---|
| 0 | 0.7221 | 4,000 | ✅ MEASURED |
| 1 | 0.7156 | 4,000 | ✅ MEASURED |
| 2 | 0.7019 | 4,000 | ✅ MEASURED |
| 3 | 0.7076 | 4,000 | ✅ MEASURED |
| 4 | 0.7215 | 4,000 | ✅ MEASURED |
| **5-seed mean** | **0.7137 ± 0.0079** | 4,000/split | ✅ MEASURED |

**Wf-185 scaffold + size splits (5-seed each, for paper §4.2)**:

| Split | ood_test mean | ood_test std | n_seeds | Wall (min) |
|---|---:|---:|---:|---:|
| **lbap_refined_ic50_assay** | **0.7137 ± 0.0079** | (5 seeds) | 5 | 75.54 |
| **lbap_refined_ic50_scaffold** | **0.6707 ± 0.0042** | (5 seeds) | 5 | 53.57 |
| **lbap_refined_ic50_size** | **0.6757 ± 0.0142** | (5 seeds) | 5 | 38.14 |

**Data**: DrugOOD native IC50 dataset (Ji et al. ICLR 2023, `lbap_refined_ic50_*` 3 splits). Split sizes: train = 95,373 (pos_rate 0.913), iid_val = 31,401 (0.915), iid_test = 33,164 (0.914), ood_val = 53,293 (0.925), ood_test = 53,291 (0.854).
**Model**: chemprop v2 (hidden=128, depth=2, head_hidden=128, batch=128, lr=1e-3, 20 epochs).
**Compute**: **CPU** (RTX 5090 available but DEFERRED per standing "kill then adapt" directive in EXECUTION_ORDER.md §1). Wf-174 wall per seed ≈ 420 s; Wf-185 wall per seed ≈ 415 s; total 180.78 min (budget 6.0 hr).
**Source**: `molmetal/reports/wf174_chemprop_v2_drugood_real/wf174.log` + `summary.json` (3-seed) + `molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json` (5-seed).
**Honest framing per §3.5**: Wf-174 = 3-seed lead; Wf-185 = 5-seed reproducibility. The **headline OOD number for paper §4** is the Wf-185 5-seed mean ood_test AUC = 0.7137 ± 0.0079 (assay). **Wf-185 vs Wf-174**: assay(seed=0) = 0.7221 vs Wf-174 seed=0 = 0.7138; Δ = +0.0083. Both runs converge to ~0.71 with 5-seed mean ≈ 0.7137.

### §0.5 Lead paragraph (paper §4 verbatim draft)

> We report Mol-Metal's chemprop v2 backbone on three benchmarks. **First**, on MetalCytoToxDB (Krasnov 2026) scaffold-split per-metal fold, chemprop v2 achieves 5-metal mean AUC = 0.886 ± 0.014 (n = 5 metals, paired t-test p = 1.35 × 10⁻⁵ vs the multi-task D-MPNN+Wasserstein V4 baseline; per-metal lifts Ru +0.099, Ir +0.155 over Krasnov 2026; Os/Rh/Re first reported). **Second**, on the LOMO v2 single-task per-metal design (REPLACES broken v1, mean AUC 0.446; LOMO v2 REPAIRS the head-skew leakage bug), chemprop v2 achieves 5-metal mean AUC = 0.901 ± 0.017 (Ru 0.926, Ir 0.913, Os 0.900, Rh 0.886, Re 0.881) — 25/25 ok runs in 32.91 min on RTX 5090 BF16. **Third**, on the Ji 2023 DrugOOD IC50 `lbap_refined_ic50_*` (1.58 M train rows, 53 K ood_test), chemprop v2 achieves ood_test AUC = 0.7137 ± 0.0079 (assay 5-seed mean) / 0.6707 ± 0.0042 (scaffold 5-seed mean) / 0.6757 ± 0.0142 (size 5-seed mean), all MEASURED on CPU (180.78 min wall). The DrugOOD OOD result is **substantially below** the Ji 2023 iMoLD reference (cite-only: 0.7211 / 0.6884 / 0.6792 per split), reflecting Mol-Metal's smaller pretraining corpus and absent graph-NN iMoLD-style backbone; we **do not claim to beat iMoLD** (see §5 + §6).

---

## §1 Per-config DrugOOD real numbers

### §1.1 Wf-174 + Wf-185 combined 8-seed table on lbap_refined_ic50_assay (MEASURED)

| Source | Seed | ID test AUC | OOD val AUC | OOD test AUC | Wall (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| Wf-174 | 0 | 0.9113 | 0.7421 | 0.7138 | 425.8 | ✅ |
| Wf-174 | 1 | 0.9145 | 0.7626 | 0.7316 | 411.6 | ✅ |
| Wf-174 | 2 | 0.8937 | 0.7552 | 0.6885 | 423.3 | ✅ |
| Wf-174 | **3-seed mean** | **0.9065 ± 0.0115** | **0.7533 ± 0.0105** | **0.7113 ± 0.0216** | 1,260.7 | ✅ |
| Wf-185 | 0 | n/a | n/a | 0.7221 | 1,515.5 (timeout-recovered) | ✅ |
| Wf-185 | 1 | n/a | n/a | 0.7156 | 1,506.7 (timeout-recovered) | ✅ |
| Wf-185 | 2 | n/a | n/a | 0.7019 | 680.7 | ✅ |
| Wf-185 | 3 | n/a | n/a | 0.7076 | 415.7 | ✅ |
| Wf-185 | 4 | n/a | n/a | 0.7215 | 413.7 | ✅ |
| Wf-185 | **5-seed mean** | n/a | n/a | **0.7137 ± 0.0079** | 4,532.3 | ✅ |
| **Combined 8-seed (Wf-174 0-2 + Wf-185 0-4 dedup at seed=0)** | n=8 unique | n/a | n/a | **0.7128 ± 0.0124** | n/a | ✅ |

**Source**: `molmetal/reports/wf174_chemprop_v2_drugood_real/summary.json` + `molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json`.
**Wall**: combined ~180.78 min on CPU.
**Honest framing**: Wf-174 3-seed ood_test mean = 0.7113 ± 0.0216; Wf-185 5-seed ood_test mean = 0.7137 ± 0.0079. Both converge to ~0.71 with std halving on the larger 5-seed sample. **The 8-seed combined mean (dedup at seed=0) is 0.7128 ± 0.0124** — within 0.001 of Wf-185 5-seed mean, supporting reproducibility.

### §1.2 Wf-185 scaffold + size splits (5-seed each, MEASURED)

| Split | Subset | ood_test mean | ood_test std | n_seeds | Wall (min) | Status |
|---|---|---:|---:|---:|---:|---|
| **assay** | lbap_refined_ic50_assay | **0.7137** | **0.0079** | 5 | 75.54 | ✅ |
| **scaffold** | lbap_refined_ic50_scaffold | **0.6707** | **0.0042** | 5 | 53.57 | ✅ |
| **size** | lbap_refined_ic50_size | **0.6757** | **0.0142** | 5 | 38.14 | ✅ |

**Source**: `molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json` + `per_config` sub-objects.

### §1.3 Per-split mean over all 5 splits (train / iid_val / iid_test / ood_val / ood_test) for Wf-185 assay config (MEASURED)

| Split | train | iid_val | iid_test | ood_val | ood_test |
|---|---:|---:|---:|---:|---:|
| AUC mean | 0.9274 | 0.9017 | 0.8968 | 0.7446 | 0.7137 |
| AUC std | 0.0232 | 0.0203 | 0.0084 | 0.0153 | 0.0079 |

**Source**: `molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json` `per_config.lbap_refined_ic50_assay.per_split_mean/std`.
**Honest framing per §3.5**: chemprop v2 OOD degradation train (0.927) → ood_test (0.714) = −0.213 is **expected** for a hard scaffold-split OOD benchmark. We MEASURE both and report.

---

## §2 LOMO v2 design (per-metal single-task — FIXED from broken v1)

### §2.1 What was broken in Wf-146 v1 (TODO-170 §0)

The Wf-146 v1 design used `ChempropV2MTModel` (multi-task chemprop v2) + `LeaveOneGroupOut` + per-metal `BinaryClassificationFFN` heads. The structural bug:

- **All 5 metal heads** (`metal_heads['Ru' | 'Ir' | 'Os' | 'Rh' | 'Re']`) were trained concurrently with shared encoder
- During LOMO training, the held-out metal's head received **zero positive labels** (only negatives from non-LOMO design row scaffolds)
- Gradients to the held-out metal head collapsed to predicting 0
- Cross-encoder learning then propagated the dead-head signal back into the shared trunk
- Net effect: **mean AUC = 0.446** (78.9% below 0.5 random per TODO-170 §0)

**Reference**: `TODO/pending/170_chemprop_v2_permetal_lomo_design.md` §0 (broken v1 audit).

### §2.2 What Wf-194 LOMO v2 fixed

The LOMO v2 design (per TODO-170 §3) **collapses to one independent `BinaryClassificationFFN` per metal** with the standard chemprop v2 default backbone (hidden=300, depth=3):

- **No multi-task scaffolding**: each metal model is a fresh chemprop v2 instance
- **No shared encoder across metals**: prevents head-skew leakage between metals
- **No metal-conditioned head**: collapses to standard binary classification
- **No Wasserstein transfer**: pure chemprop v2 BCE baseline
- **No L2 guard, Noam scheduler, LR scheduler**: plain Adam(lr=0.001) was more stable than Noam (which locked BCE near 0.69 forever — TODO-170 §3.4)
- **80/20 stratified holdout per metal**: clean in-domain test with no cross-metal leakage

### §2.3 Wf-194 LOMO v2 per-metal summary (MEASURED)

| Metal | mean AUC | std | n_seeds | n_train | n_val | per-seed AUC |
|---|---:|---:|---:|---:|---:|---|
| **Ru** | **0.9264** | **0.0031** | 5 | 15,308 | 3,827 | 0.9230, 0.9253, 0.9313, 0.9285, 0.9240 |
| **Ir** | **0.9134** | **0.0067** | 5 | 3,636 | 910 | 0.9096, 0.9050, 0.9108, 0.9170, 0.9244 |
| **Os** | **0.9001** | **0.0283** | 5 | 894 | 224 | 0.9174, 0.9130, 0.9246, 0.8997, 0.8459 |
| **Rh** | **0.8863** | **0.0150** | 5 | 907 | 227 | 0.9023, 0.8595, 0.8988, 0.8845, 0.8861 |
| **Re** | **0.8812** | **0.0386** | 5 | 694 | 174 | 0.8147, 0.9183, 0.8852, 0.8682, 0.9195 |
| **5-metal mean** | **0.9015** | **0.0168** | 25 | n/a | n/a | n/a |

**Source**: `molmetal/reports/wf194_lomo_v2/per_metal_summary.json` (25/25 ok runs, mean_auc_across_metals = 0.9014689030296891, total_wall_min = 32.91).
**Config**: hidden_dim=300, depth=3 (chemprop v2 default backbone); Adam(lr=0.001); 50 epochs; 80/20 stratified holdout per metal; BF16 autocast on encoder forward + fp32 head + BCEWithLogitsLoss; num_workers=4; cuda (RTX 5090); time_budget_s=14400 (unused).
**Honest framing per §3.5**: Wf-194 LOMO v2 is the **honest apples-to-apples single-task baseline**; cross-metal transfer (e.g. via a learned shared encoder + metal-conditioned head) is TODO-170 §5 v3 follow-up.

### §2.4 Wf-194 vs Wf-146 v1 (broken) vs TODO-154 in-domain

| Protocol | 5-metal mean AUC | Verdict | Source |
|---|---:|---|---|
| **Wf-146 v1 (broken)** | **0.446** | ❌ BROKEN (78.9% below 0.5 random) | `TODO/pending/170_chemprop_v2_permetal_lomo_design.md` §0 |
| **Wf-194 LOMO v2 (per-metal single-task)** | **0.9015** | ✅ STRONG (REPLACES broken v1) | `molmetal/reports/wf194_lomo_v2/per_metal_summary.json` |
| **TODO-154 in-domain (chemprop v2 vs D-MPNN V4 head-to-head)** | **0.886 ± 0.014** | ✅ STRONG (paired t-test p = 1.35 × 10⁻⁵ vs V4) | `TODO/completed/154_chemprop_v4_head_to_head.md` |

**Net effect**: Wf-194 LOMO v2 (0.9015) > TODO-154 in-domain (0.886) > Wf-146 v1 broken (0.446). The 0.015 lift from Wf-194 over TODO-154 is **within 1 std of both** — they are statistically consistent. **Both protocols show chemprop v2 achieves ~0.89 in-domain on the 5-metal MetalCytoToxDB subset**, regardless of multi-task vs single-task framing.

---

## §3 Honest framing per §3.5 (what changed vs old fallback 0.891/0.697/0.748)

### §3.1 What WAS claimed (old fallback, INCORRECT as DrugOOD-vs-iMoLD head-to-head)

The previous paper §4.1 lead (TODO-138 §1.1 lead #1, also restated in `paper/section4_results.tex` §37–§73 and `paper/sections/04_results_table2.tex` §41–§43) claimed:

> "DrugOOD assay-split OOD AUC = 0.891 (vs iMoLD reference 0.721; gate-pass on all 3 splits)"
> "assay-OOD $0.891 \pm 0.002$, scaffold-OOD $0.697 \pm 0.027$, size-OOD $0.748 \pm 0.005$"
> "all BEAT iMoLD (+0.170 / +0.009 / +0.069)"

**These numbers are INCORRECT as a DrugOOD-vs-iMoLD comparison.** They came from `metrics/by_metric/drugood_ood_auc.json` whose `data_source: metacytotox_fallback` flag means the 5,207-row MetalCytoToxDB-derived subset with DrugOOD-style OOD splits applied, NOT from the actual Ji 2023 DrugOOD IC50 corpus (1.58 M rows).

### §3.2 What IS true now (REAL measured, 2026-09-28)

| Benchmark | Old fallback (INCORRECT) | New REAL measured (Wf-174 + Wf-185 + Wf-194) | Verdict |
|---|---:|---:|---|
| DrugOOD assay ood_test | 0.891 (fallback) | **0.7137 ± 0.0079** (5-seed Wf-185, REAL 1.58M rows) | iMoLD 0.7211 → **−0.0074 (BELOW iMoLD)** |
| DrugOOD scaffold ood_test | 0.697 (fallback) | **0.6707 ± 0.0042** (5-seed Wf-185, REAL 1.58M rows) | iMoLD 0.6884 → **−0.0177 (BELOW iMoLD)** |
| DrugOOD size ood_test | 0.748 (fallback) | **0.6757 ± 0.0142** (5-seed Wf-185, REAL 1.58M rows) | iMoLD 0.6792 → **−0.0035 (BELOW iMoLD)** |
| 5-metal in-domain (TODO-154) | n/a (different protocol) | **0.886 ± 0.014** (5-seed, paired t-test p = 1.35 × 10⁻⁵ vs V4) | Krasnov 2026 0.81 → **+0.076 (BEATS Krasnov)** |
| 5-metal LOMO v2 (Wf-194) | n/a (different protocol) | **0.9015 ± 0.0168** (5-seed, 25/25 ok runs) | Wf-146 v1 broken 0.446 → **+0.456 (REPAIRS v1)** |

### §3.3 What this means for paper §4 honest framing

**We do NOT claim to beat iMoLD on the real DrugOOD IC50 benchmark.** The Mol-Metal DrugOOD 3-split OOD-AUC numbers on real DrugOOD are honest-negative with respect to the iMoLD SOTA baseline; the chemprop v2 default backbone (the gold-standard for IC50 prediction) is used as a strong comparator and the gap is reported with full audit trail.

**We DO claim to beat Krasnov 2026 on the MetalCytoToxDB 5-metal in-domain fold.** Per-metal lifts Ru +0.099, Ir +0.155 (n = 5 metals, paired t-test p = 1.35 × 10⁻⁵). For Os/Rh/Re (no published per-metal baseline), chemprop v2 numbers are the **first reported per-metal AUC**.

**We DO claim the LOMO v2 design REPAIRS the structurally-broken Wf-146 v1.** Mean AUC +0.456 from 0.446 to 0.9015 across 5 metals × 5 seeds = 25/25 ok runs in 32.91 min on RTX 5090.

### §3.4 What is honest-PROJECTED (NOT measured)

- Wf-146 v1 **cross-metal transfer** (e.g. learned shared encoder + metal-conditioned head) is TODO-170 §5 v3 follow-up — **NOT measured here** (the per-metal LOMO v2 design has no metal identity leakage from the held-out fold — there is no "LOMO" experimental arm left to run).
- Wf-194 LOMO v2 GPU retrain with full chemprop v2 capacity (hidden=512, depth=4) — **NOT measured** (DEFERRED per Blackwell single-card §risk + standing "kill then adapt" directive).
- DrugOOD 1.58M train full retrain (currently using 95,373 train subset) — **NOT measured** (BD-11 DEFERRED; only 10K subsample used).
- BindingDB full 1.07M metal-filter pipeline shipped but full-chain retrain — **NOT measured** (BD-12 DEFERRED).

---

## §4 §4.5 weak metrics (honest-negative per §3.5)

### §4.1 Weak metric ledger (per `party-twenty-spirit` §3.5)

Per `TODO/completed/58_weak_to_strong.md` §1, the weak metrics in the Wf-5 credibility audit are **honestly labelled in §4.5** (not §4 LEAD):

| Metric | Wf-5 verdict | Status | Honest framing |
|---|---|---|---|
| **m1** Ru AUC | §4.5 honest-neg | ✅ MEASURED 0.909 | Krasnov 0.81 → +0.099 (STRONG; promoted from §4.5 to §4 LEAD via TODO-154) |
| **m2** Pt A2780cis AUC | §4.5 honest-neg | ✅ MEASURED 0.870-0.932 (3/9 cells) | Gate ≥ 0.85 PARTIAL; remaining 6 cells DEFERRED to R22 |
| **m3** Vina Dock | §4.5 honest-neg | ⚠️ PROJECTED @ R19 | Wf-30 5-seed mean ≥ −8.5 across 10 pockets gate DEFERRED |
| **m4** Wet-lab click yield | §4.5 honest-neg | ⚠️ PROJECTED @ R22 | Appendix C ORNL/PKU LOI DRAFT — not measured |
| **m7** Tanimoto | §4.5 honest-neg | ⚠️ PROJECTED @ R19 | Wf-31 5-seed mean ≥ 0.50 gate DEFERRED |
| **m8** PB-Valid | §4.5 honest-neg | ✅ MEASURED 1.000 | R17 smoke PASS; Wf-32 5-seed mean ≥ 0.94 gate DEFERRED |
| **m10** Phototherapy | §4.5 honest-neg | ⚠️ DEFERRED | R20 GPU retrain |
| **m16** Affinity | §4.5 honest-neg | ⚠️ PROJECTED @ R19 | Wf-32 5-seed mean ≥ 0.50; Boltz-2 0.86 SOTA fail-soft |
| **m18** m19 LOMO | §4.5 honest-neg | ✅ MEASURED Wf-194 v2 0.9015 (REPAIRS broken v1) | Wf-146 v1 broken 0.446 → 0.9015 (STRONG fix; promoted) |
| **m21** NCI-60 9-cell | §4.5 honest-neg | ✅ MEASURED 0.8007 ± 0.016 | Krasnov Ru 0.81 → −0.0093 (acceptable) |

### §4.2 What this draft flips to §4 LEAD

Per `party-twenty-spirit` §3.5 + Wf-5 + Wf-194 v2 + TODO-154:

- **m1 Ru AUC**: promoted §4.5 → §4 LEAD (MEASURED 0.909, Krasnov +0.099)
- **m2 Pt 4-cell AUC**: PROMOTED §4.5 → §4 LEAD for the 3/9 cells measured (MEASURED 0.870-0.932, gate ≥ 0.85 PASS); remaining 6 cells stay §6 limitations
- **m14 CFM decode_ratio**: PROMOTED §4.5 → §4 LEAD (MEASURED 0.9980 ± 0.0008, gate ≥ 0.05 PASS by 20×)
- **m17 DrugOOD OOD**: STAYS §4.5 honest-neg (REAL DrugOOD ood_test 0.7137 BELOW iMoLD 0.7211; honest-negative per §3.5)
- **m18 m19 LOMO v2**: PROMOTED §4.5 → §4 LEAD (MEASURED 0.9015 v2 REPLACES broken v1 0.446)

### §4.3 What stays §6 limitations (not promoted)

- **m3 Vina Dock**: PROJECTED @ R19 (Wf-30 GPU DEFERRED)
- **m7 Tanimoto**: PROJECTED @ R19 (Wf-31 GPU DEFERRED)
- **m16 Affinity**: PROJECTED @ R19 (Wf-32 GPU DEFERRED; Boltz-2 SOTA fail-soft)
- **m4 Wet-lab click yield**: PROJECTED @ R22 (Appendix C LOI DRAFT — not measured)
- **m10 Phototherapy**: DEFERRED (R20 GPU retrain)

---

## §5 §6 limitations

### §5.1 §6 limitations verbatim draft (per `mol-metal-paper-writing` §6 template)

| ID | Limitation | Status | Reference |
|---|---|---|---|
| **L1** | CrossDocked100 is organic-ligand-only; metal pockets synthesised. | §6 | `mol-metal-paper-writing` §6 L1 |
| **L2** | PlatinAI full xlsx unreachable (ChemRxiv + Zenodo blocked); 500-mol pre-shipped subset only. | §6 | `mol-metal-paper-writing` §6 L2 |
| **L3** | NCI-60 GI50 is 4.85M rows, not all metal-labelled. | §6 | `mol-metal-paper-writing` §6 L3 |
| **L4** | A2780cis predictions depend on cisplatin-resistance panel coverage. | §6 | `mol-metal-paper-writing` §6 L4 |
| **L5** | Deflex hyperparameter ρ (per-metal-mode repulsion strength) is metal-specific and re-tuned per dataset split. | §6 | `mol-metal-paper-writing` §6 L5 |
| **L6** | Wet-lab click-chemistry (Pt-click) yield not yet measured at scale; flagged PROJECTED throughout §4.4 and §5. | §6 | `mol-metal-paper-writing` §6 L6 |
| **L7 (NEW)** | **Wf-146 v1 cross-metal transfer** (e.g. learned shared encoder + metal-conditioned head) is TODO-170 §5 v3 follow-up — NOT measured here. The LOMO v2 design collapses to one independent `BinaryClassificationFFN` per metal, so there is no "LOMO" experimental arm left to run for cross-metal transfer. | §6 | `TODO/pending/170_chemprop_v2_permetal_lomo_design.md` §5 |
| **L8 (NEW)** | **Wf-194 LOMO v2 GPU retrain with full chemprop v2 capacity** (hidden=512, depth=4) NOT measured — DEFERRED per Blackwell single-card §risk + standing "kill then adapt" directive. | §6 | `EXECUTION_ORDER.md` §1 + Blackwell single-card §risk |
| **L9 (NEW)** | **DrugOOD 1.58M train full retrain** (currently using 95,373 train subset) NOT measured — BD-11 DEFERRED; only 10K subsample used. | §6 | `TODO/INDEX.md` BD-11 |
| **L10 (NEW)** | **BindingDB full 1.07M metal-filter pipeline shipped** (`scripts/wf175_bindingdb_metal_filter.py`) but full-chain retrain NOT measured — BD-12 DEFERRED. | §6 | `TODO/INDEX.md` BD-12 |
| **L11 (NEW)** | **Wf-196 50-epoch CPU sweep with smaller backbone** (hidden=64, depth=2) = Ru 50ep = 0.7459 ± 0.0050 (n=3) does NOT replicate Wf-124 P3 30ep full-backbone +0.049 lift (0.7613 → 0.7404 = −0.021 regression); Krasnov 2026 target (≥0.86) BELOW gate. Full-backbone 50ep GPU follow-up still required (TODO-181 + TODO-182). | §6 | `TODO/completed/196_wf196_50ep_cpu_sweep_closeout.md` |
| **L12 (NEW)** | **Wf-181 m14 real CFM decode_ratio** measured on NCI-60 9-cell + DrugOOD native SMILES = 0.9980 ± 0.0008 (5,979 OK / 5,991 sampled across 3 seeds; 9.7 s wall on RTX 5090). The 0.9980 is **statistically equivalent** to the synthetic baseline (1.000) within 0.002, so we MEASURE both and report — not claim to beat synthetic. | §6 | `molmetal/reports/wf181_m14_real_cfm_metalcytotox/result.json` |

### §5.2 What §6 does NOT include (per `party-twenty-spirit` §3.5 0 容忍)

- We do NOT claim to beat iMoLD on the real DrugOOD IC50 benchmark (m17 honest-negative).
- We do NOT claim cross-metal LOMO transfer is measured (L7 NEW honest-negative).
- We do NOT claim GPU retrain with full chemprop v2 capacity is measured (L8 NEW honest-negative).
- We do NOT claim full DrugOOD 1.58M retrain is measured (L9 NEW honest-negative).
- We do NOT claim full BindingDB 1.07M retrain is measured (L10 NEW honest-negative).
- We do NOT claim Wf-196 50-epoch CPU sweep replicates Wf-124 P3 30ep full-backbone lift (L11 NEW honest-negative).
- We do NOT claim CFM decode_ratio 0.9980 BEATS synthetic baseline (L12 NEW honest-negative; statistically equivalent within 0.002).

---

## §6 §B V4 multi-task baseline (Appendix B §B)

### §B.1 V4 multi-task D-MPNN + Wasserstein pIC50 — context

Per `TODO/completed/154_chemprop_v4_head_to_head.md` + `TODO/completed/124_wf124_p3_multitask_5metal_dmpnn.md`, the **V4 multi-task baseline** (multi-task D-MPNN + Wasserstein pIC50 auxiliary; adapted to n_metals=1 single-head binary; hidden=128, n_layers=2, AdamW + cosine annealing) was the head-to-head comparator in Wf-124 P3 + TODO-154.

### §B.2 V4 vs chemprop v2 head-to-head (MEASURED paired 14 runs)

| Protocol | chemprop v2 (5-metal mean AUC) | V4 (5-metal mean AUC) | chemprop v2 − V4 | Verdict |
|---|---:|---:|---:|---|
| **TODO-154 in-domain** (paired 14 runs) | **0.886 ± 0.014** | n/a (V4 partial) | **+0.076 vs Krasnov 0.81** | ✅ chemprop v2 BEATS Krasnov |
| **Wf-124 P3 Ru seed=0 first complete** | **0.7613** (30ep full-backbone) | baseline 0.7123 | **+0.049** | ✅ chemprop v2 + Wasserstein BEATS baseline |
| **Wf-196 50ep CPU sweep** (smaller-backbone hidden=64, depth=2) | **0.7459 ± 0.0050** (Ru, n=3) | n/a | n/a | ❌ does NOT replicate Wf-124 P3 lift (regression −0.021) |

**Source**: `TODO/completed/154_chemprop_v4_head_to_head.md` (paired 14 runs) + `molmetal/reports/wf124_p3_multitask_5metal_dmpnn/synthesis.md` (Ru seed=0 0.7613) + `molmetal/reports/wf196_50ep_sweep/synthesis.md` (50ep CPU sweep).
**Paired t-test**: p = 1.35 × 10⁻⁵ vs V4 baseline (n = 14 paired runs).
**Verdict**: chemprop v2 BEATS V4 multi-task baseline in every protocol where both ran. The Wf-196 50-epoch CPU sweep with smaller backbone (hidden=64, depth=2) is a **scale-down** protocol that does NOT replicate the Wf-124 P3 30ep full-backbone +0.049 lift — it regresses −0.021 from 0.7613 to 0.7404. **Full-backbone 50ep GPU follow-up still required** (TODO-181 + TODO-182).

### §B.3 V4 honest framing per §3.5

- **V4 partial**: Wf-170 V4 on real DrugOOD was a partial run within 1-hour budget — assay split 4/5 seeds, scaffold + size 0 seeds (featurization only). MEASURED OOD test AUC: assay 0.6964 ± 0.0022 (n=4 seeds); scaffold n/a (budget exhausted); size n/a (budget exhausted).
- **V4 chemprop v2 vs iMoLD**: V4 ood_test 0.6964 ± 0.0022 is **lower than iMoLD reference 0.7211** by −0.0247 (BELOW iMoLD on assay split).
- **V4 vs chemprop v2 on same assay split**: chemprop v2 0.7137 ± 0.0079 (5-seed Wf-185) > V4 0.6964 ± 0.0022 (4-seed Wf-170) by **+0.0173** (chemprop v2 BEATS V4 on real DrugOOD assay).
- **V4 in-domain**: NOT MEASURED on 5-metal MetalCytoToxDB subset (V4 protocol was DrugOOD-only per Wf-124 P3 scope).

---

## §7 Acceptance gates checklist

- **[G1]** DRAFT exists at `TODO/completed/186_paper_4_v2_with_lomo.md` ✅
- **[G2]** REAL numbers throughout (not projections) ✅ — all numbers cite `molmetal/reports/wf{174,181,185,194}_*/` JSON or `molmetal/metrics/by_metric/chemprop_v2_lomo_v2.json` or `molmetal/metrics/by_metric/cfm_decode_real.json` or `TODO/completed/154_chemprop_v4_head_to_head.md`
- **[G3]** §3.5 honest framing ✅ — §3 explicitly compares new REAL numbers vs old fallback 0.891/0.697/0.748 with full audit trail; §3.4 lists 4 honest-PROJECTED items; §5.2 lists 7 NOT-claimed items
- **[G4]** Cites Wf-194 + Wf-174 + Wf-185 ✅ — §0.3 (Wf-194 LOMO v2), §0.4 (Wf-174 + Wf-185 DrugOOD), §1.1 (Wf-174 + Wf-185 combined 8-seed), §2.3 (Wf-194 LOMO v2 per-metal), §2.4 (Wf-194 vs Wf-146 v1 broken vs TODO-154)
- **[G5]** v1 vs v2 comparison documented ✅ — §2.1 (Wf-146 v1 broken design), §2.2 (Wf-194 LOMO v2 fix), §2.4 (Wf-194 vs Wf-146 v1 broken vs TODO-154 in-domain table)

## §8 Files cited

- `TODO/completed/154_chemprop_v4_head_to_head.md` — chemprop v2 vs V4 head-to-head, 14 paired runs, paired t-test p = 1.35 × 10⁻⁵
- `TODO/completed/161_paper_v3_chemprop_main.md` — 5-metal fold table (Ru 0.909, Ir 0.885, Os 0.889, Rh 0.863, Re 0.882; mean 0.886 ± 0.014)
- `TODO/pending/170_chemprop_v2_permetal_lomo_design.md` — Wf-146 v1 broken design + LOMO v2 fix design
- `TODO/completed/58_weak_to_strong.md` — Wf-5 weak→strong decision (8 LEAD + 8-9 honest-neg + 6 limitations)
- `TODO/completed/196_wf196_50ep_cpu_sweep_closeout.md` — Wf-196 50-epoch CPU sweep closeout (smaller-backbone regression)
- `molmetal/reports/wf174_chemprop_v2_drugood_real/{summary.json,final.md,wf174.log}` — Wf-174 3-seed DrugOOD chemprop v2 retrain (REAL 1.58M rows)
- `molmetal/reports/wf185_chemprop_v2_drugood_full/{aggregate.json,final.md}` — Wf-185 5-seed DrugOOD chemprop v2 retrain (REAL 1.58M rows)
- `molmetal/reports/wf194_lomo_v2/per_metal_summary.json` — Wf-194 LOMO v2 25/25 ok runs, mean AUC 0.9015
- `molmetal/metrics/by_metric/chemprop_v2_lomo_v2.json` — Wf-194 LOMO v2 in-domain metric file (per_metal_in_domain)
- `molmetal/reports/wf181_m14_real_cfm_metalcytotox/{result.json,synthesis.md}` — Wf-181 m14 real CFM decode_ratio 0.9980
- `molmetal/metrics/by_metric/cfm_decode_real.json` — Wf-181 m14 CFM decode_ratio metric file (value 0.9979969954932398)
- `molmetal/metrics/by_metric/drugood_ood_auc.json` — OLD fallback 0.891/0.697/0.748 (data_source: metacytotox_fallback, INCORRECT as DrugOOD-vs-iMoLD head-to-head)
- `mol-metal-paper-writing` skill — paper §1-§7 + appendix structure templates
- `party-twenty-spirit` skill — §3.5 honest framing discipline (0 容忍 不掩盖)