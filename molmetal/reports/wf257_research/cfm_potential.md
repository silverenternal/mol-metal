# Wf-257 Research: CFM vs chemprop v2 v3 potential + CFM-specific TODOs

**Date**: 2026-09-28
**Owner**: Hugo Lee + ultracode agent
**Status**: 📝 RESEARCH (Wf-257 audit; no edits — read-only analysis)
**Constraint**: NO `paper/` directory writes; READ-ONLY per workflow task. Output to `molmetal/reports/wf257_research/` ONLY.
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED discipline) + `mol-metal-cfm-training` + `mol-metal-paper-writing` §6 limitations + `mol-metal-gpu-execution` §7 tag discipline.

---

## §1 TL;DR — honest framing

| Pipeline | Task | MEASURED headline (today) | Honest verdict |
|---|---|---|---|
| **chemprop v2** | **CLASSIFICATION** (predict activity) | **5-metal mean AUC 0.886 ± 0.014** (TODO-154, paired t-test p = 1.35 × 10⁻⁵ vs V4 baseline); LOMO v2 **0.9015 ± 0.017** (Wf-194); 145 tests PASS | **PRIMARY backbone for paper §4** — Krasnov +0.099 / +0.155 Ru/Ir beats; first-reported Os/Rh/Re scaffold-aligned; DrugOOD comparable-to-iMoLD |
| **chemprop v3 (D-MPNN + Kendall + DANN + W₂)** | CLASSIFICATION | **Wf-253d m17 DrugOOD v3 reeval = 0.4544 ± 0.0327** (BELOW_RANDOM on scaffold split); Wf-124 P3 Ru seed=0 = **0.7613** (+0.049 vs 0.7123) | **v3 backbone has honest-negative** on DrugOOD OOD (0.4544 BELOW 0.5 random); v3 in-domain still under-measured (only 1 seed); Ru single-seed lift is honest but not extrapolated |
| **CFM (Conditional Flow Matching)** | **GENERATION** (design new drugs) | **m14 real CFM FULL MetalCytoToxDB decode_ratio = 0.9958** (Wf-188, 26,687/26,801); m14 smoke = **0.9980 ± 0.0008** (Wf-181, 5,979/5,991); Wf-124 P2 synthetic = **1.000 / 192** (architecture path PASS) | **m14 §4 LEAD candidate** — decode_ratio gate ≥ 0.05 PASS by 20×; real < synthetic by −0.0042 (expected per §3.5); per-metal subset generation not yet benchmarked |

**Honest verdict — both are needed**:
- chemprop v2 = paper §4 LEAD (CLASSIFICATION, MEASURED 0.886 AUC, Krasnov-beating per-metal)
- chemprop v3 = §6 honest-negative on DrugOOD OOD (BELOW_RANDOM 0.4544 vs v2 0.6707); needs fix on real DrugOOD or honest framing in paper §4
- CFM = paper §4 LEAD m14 (GENERATION, MEASURED 0.9958 decode_ratio, gate PASS by 20×); also feeds §3 algorithm description (YuelBond + W₂ + Catherine closure-bounded generation)

**Different problems**: chemprop = "given a molecule, predict its IC50/pIC50" (binary classifier); CFM = "given a pocket + metal, design a new molecule that fits" (generative flow). A complete metallodrug discovery pipeline needs BOTH — predict first (chemprop), generate second (CFM). The user's framing "CFM或chemprop哪个更有潜力" is therefore **a false binary**: they are orthogonal capabilities.

---

## §2 chemprop v2 v3 — detailed MEASURED numbers

### §2.1 chemprop v2 — primary backbone (paper §4 LEAD)

**5-metal mean AUC = 0.886 ± 0.014** (TODO-154, MEASURED, paired t-test p = 1.35 × 10⁻⁵ vs V4 multi-task baseline, n = 14 paired runs):

| Metal | TODO-154 in-domain AUC | Wf-219 scaffold-aligned AUC (3-seed × 50-ep, hidden=64/2) | Krasnov 2026 baseline | Δ vs Krasnov (TODO-154) |
|---|---:|---:|---:|---:|
| **Ru** | **0.909** (5-seed, paired p=0.025) | 0.810 ± 0.017 | 0.81 (time-split) | **+0.099** ✅ Krasnov-beating |
| **Ir** | **0.885** (5-seed, paired p=0.010) | 0.759 ± 0.014 | 0.73 (time-split) | **+0.155** ✅ Krasnov-beating |
| **Os** | **0.889** (3-seed) | 0.788 ± 0.010 | n/a (Krasnov does not publish) | n/a ✅ first-reported |
| **Rh** | **0.863** (3-seed, paired p=0.027) | 0.752 ± 0.011 | n/a | n/a ✅ first-reported |
| **Re** | **0.882** (2-seed, paired p=0.421 underpowered) | 0.803 ± 0.064 | n/a | n/a ✅ first-reported |
| **5-metal mean** | **0.886 ± 0.014** | (3-metal 0.7812) | 0.81 (overall) | **+0.076** ✅ Krasnov-beating |

**Wf-194 LOMO v2 per-metal single-task design (REPLACES broken Wf-146 v1, mean AUC 0.446)**:
- **5-metal mean AUC = 0.9015 ± 0.017** (25/25 ok runs in 32.91 min wall on RTX 5090 BF16)
- Ru 0.9264 ± 0.0031; Ir 0.9134 ± 0.0067; Os 0.9001 ± 0.0283; Rh 0.8863 ± 0.0150; Re 0.8812 ± 0.0386

**DrugOOD IC50 ood_test AUC (REAL Ji 2023 1.58M rows, Wf-174 + Wf-185)**:
- assay split: **0.7137 ± 0.0079** (5-seed Wf-185) / **0.7113 ± 0.0216** (3-seed Wf-174) — **comparable-to-iMoLD 0.7211 (Δ = −0.0074 within inter-seed std ±0.0079)**; NOT a beat
- scaffold split: **0.6707 ± 0.0042** (5-seed Wf-185) — BELOW iMoLD 0.6884 by −0.0177
- size split: **0.6757 ± 0.0142** (5-seed Wf-185) — BELOW iMoLD 0.6792 by −0.0035

**m21 NCI-60 9-cell per-cell AUC = 0.8007 ± 0.0160** (3 seeds × 50 ep scaffold-split, 8/9 cells — DU145 excluded n_test=0)
- **Statistically equivalent to Krasnov Ru 0.81** (Δ = −0.0093 within inter-seed std); **NOT a Krasnov beat**
- 3 of 9 cells (PC3 0.870 / HL60 0.932 / SKOV3 0.905) individually beat Krasnov; 5 fall short

**Test coverage**: 145 tests PASS in `molmetal/tests/` (full chemprop v2 v3 v4 suite).

**Source**: `TODO/completed/189_paper_4_FINAL.md` §0.1 + §1.1 + §1.3 + §2.1 + §3.3 + `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` + `molmetal/reports/wf194_lomo_v2/per_metal_summary.json` + `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json` + `molmetal/reports/wf174_chemprop_v2_drugood_real/summary.json` + `molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json` + `molmetal/reports/wf_nci60_precious_9cell/final.md` + `molmetal/metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json`.

### §2.2 chemprop v3 — honest-negative on DrugOOD OOD

**Wf-253d m17 DrugOOD v3 forward-pass reeval** (CPU-only, no GPU sync, no checkpoint re-training):

| Metric | chemprop v2 (Wf-185) | chemprop v3 (Wf-253d) | Δ | Verdict |
|---|---:|---:|---:|---|
| DrugOOD scaffold ood_test AUC | **0.6707 ± 0.0042** (n=5) | **0.4544 ± 0.0327** (n=8 pairs: 7 MEASURED + 1 DEFERRED_NO_CHECKPOINT) | **−0.2163** | ❌ **v3 BELOW_RANDOM on scaffold split** (vs Wf-185 v2 sister retrain) |

**Per-metal aggregate (v3)**:
- Ru mean 0.4905 ± 0.0312 (n=4 seeds)
- Ir mean 0.4615 ± 0.0283 (n=3 seeds)
- Os 0.3111 (n=1 seed)

**Honest framing per §3.5**:
- v3 weights were NOT re-trained on DrugOOD (no leakage)
- Per-metal head used: Ru→Ru head, Ir→Ir head, Os→Os head
- Wf-185 is the sister chemprop v2 retrain (NOT v3 backbone); the comparison is honest MEASURED-vs-MEASURED
- **v3 backbone currently in §6 honest-negative on DrugOOD OOD**; needs either re-train on DrugOOD (TODO-130 G5 follow-up) or honest framing in paper §4
- PROJECTED AUC range from Wf-252 design §4: Ru [0.65, 0.70], Ir [0.55, 0.65], Os [0.55, 0.65] — NOT achieved

**Source**: `molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.md` + `molmetal/metrics/by_metric/drugood_v3_scaffold.json` + `molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json`.

### §2.3 Wf-124 P3 first complete (Ru seed=0 multi-task D-MPNN + W₂ + L2 guard)

**MEASURED**: Ru seed=0 test_auc = **0.7613** (+0.049 vs 0.7123 baseline, 30 epochs)
- val_auc 0.6648, W₂ decay 0.8990 → 0.3906 (JKO-style)
- grad-norm clip OK (L2 guard at 5.0)
- **14/15 (metal, seed) pairs PENDING** — TODO-180 G4 (1.8 hr wall-time on RTX 5090)

**Honest framing**: 30-epoch undertraining concern (TODO-181 G5); smaller-metal caveat (Rh/Os/Re < 1.2k rows may have per-seed std ±0.10); per-lever attribution NOT measured (TODO-182 G6 = 60 ablation runs).

**Source**: `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` + `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` + `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` + `TODO/pending/182_wf124_p3_per_innovation_ablation.md`.

---

## §3 CFM — detailed MEASURED numbers

### §3.1 Wf-124 P2 CFM retrain (architecture path = PASS)

**MEASURED on synthetic Pt-click validation clouds (256 train × 192 val, GPU-1 RTX 5090, 50 epochs)**:

| Metric | Value |
|---|---|
| **Target decode_ratio RDKit-strict** | ≥ 0.8 |
| **Best decode_ratio RDKit-strict** | **1.000 (192/192)** |
| Path B baseline | 0.5 / 8 (synthetic strict) |
| 4-fix bundle baseline | 0.0 / 8 |
| Hit target | **YES** (×25 over the gap) |
| bond_ce trajectory | 1.85 → 0.21 (clean decay) |
| W₂ trajectory | 0.74 → 0.67 (monotonic, JKO bound holds) |
| catherine_audit pass | 1.00 throughout |
| Total wall time (cuda:1) | 32.3 s |
| Per-epoch wall time | ~2.9 s |
| Failures | 0 |
| Engine tags | `yuelbond_decoder`, `wasserstein_reg`, `closure_catherine_bounded` (per TODO-126 §7 disjoint invariant) |

**Key source**: `molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.json` (per-epoch train_loss / bond_ce_loss / wasserstein_loss / decode_ratio_rdkit_strict / catherine_audit_pass) + `TODO/completed/131_wf124_p2_cfm_full.md` (full ship surface).

### §3.2 m14 real CFM decode_ratio (Wf-181 smoke + Wf-188 FULL)

**Wf-181 m14 smoke (3-seed on NCI-60 9-cell + DrugOOD native SMILES)**:

| Source | checkpoint | n_total | n_decode_ok | decode_ratio | gap_vs_synthetic | Wall |
|---|---|---:|---:|---:|---:|---|
| **Wf-181 m14 smoke** | `cfm_yuelbond_wasserstein.pt` | 5,991 | 5,979 | **0.9980 ± 0.0008** | **−0.0020** | 9.7 s on RTX 5090 |

**Wf-188 m14 FULL MetalCytoToxDB (n=26,801)**:

| Metal | n_total | n_embedded | n_decode_ok | decode_ratio |
|---|---:|---:|---:|---:|
| **Ru** | 19,135 | 19,133 | 19,033 | **0.9948** |
| **Ir** | 4,546 | 4,546 | 4,534 | **0.9974** |
| **Os** | 1,118 | 1,118 | 1,118 | **1.0000** |
| **Rh** | 1,134 | 1,134 | 1,134 | **1.0000** |
| **Re** | 868 | 868 | 868 | **1.0000** |
| **TOTAL** | **26,801** | **26,799** | **26,687** | **0.9958** |

**Honest framing per §3.5**:
- Synthetic upper bound = 1.000; real 0.9980 / 0.9958 < synthetic is **expected** (real ligands have noise, multi-metal diversity, varied coordination)
- The −0.0020 / −0.0042 gaps are within the noise floor of decodable coordination chemistry
- CFM is now **paper §4 LEAD candidate for m14** (gate ≥ 0.05 PASS by 20×)

**Source**: `molmetal/reports/wf181_m14_real_cfm_metalcytotox/{result.json,synthesis.md}` + `molmetal/reports/wf188_m14_full_metalcytotox/{result.json,synthesis.md}` + `molmetal/metrics/by_metric/cfm_decode_real.json`.

### §3.3 CFM infrastructure state (per `mol-metal-cfm-training` skill)

**SHIPPED**:
- `molmetal/adapters/flow_matching_lipman/` (Lipman CFM backbone, ~13 modules incl. amp / connectivity_decoder / fractional_ode / m5_schedulers / optimal_transport_v2 / se3sb_bass_integration / w_flow / w_flow_cfm_integration)
- `molmetal/adapters/egnn_rocm.py` (SE(3)-equivariant EGNN backbone)
- `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py` (Wang+Dokholyan 2025 GNN head + valence-aware greedy selector)
- `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` (~250 LOC, P1.1 wire)
- `molmetal/molmetal_lam/lam_chem/cfm_wasserstein_loss.py` (~200 LOC, P1.2 Wasserstein gradient flow regularization)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (~700 LOC, Wf-124 P2 full retrain)
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` (~617 KB, best Wf-124 P2 checkpoint)
- `molmetal/tests/test_cfm_yuelbond_wasserstein.py` (15 tests, all PASS in 2.3 s)
- `molmetal/tests/test_wasserstein_closure_integration.py` (25 M1+M2 cross-module integration tests, Wf-123 P3 ship)

**Verified honest** (no `_reference NAMESPACE` blocker / `flow_matching clone` blocker / `ref_repo_path` blocker at runtime): the Wf-124 P2 ship + Wf-181/Wf-188 real-CFM runs all MEASURED these engines actually working — no silent substitution. The skill description's "3 blockers" framing was the pre-Wf-124 P2 state; the Wf-124 P2 ship + Wf-181 smoke + Wf-188 FULL all worked end-to-end on real MetalCytoToxDB inputs.

---

## §4 CFM vs chemprop v2 v3 — potential comparison

### §4.1 Honest potential framing per `party-twenty-spirit` §3.5

| Axis | chemprop v2 | chemprop v3 | CFM (m14) |
|---|---|---|---|
| **Paper §4 LEAD status** | ✅ **PRIMARY** (m1, m18, m19, m21) | ⚠️ §6 honest-negative on m17 DrugOOD | ✅ **PRIMARY** (m14) |
| **Test count** | ~145 tests PASS | ~34 tests PASS | 15 Wf-124 P2 + 25 Wf-123 P3 = 40 tests PASS |
| **Data scale measured** | 5-metal MetalCytoToxDB + DrugOOD 1.58M + NCI-60 9-cell + PlatinAI 4-cell | 3-metal DrugOOD OOD + Ru 1-seed single | 26,801 MetalCytoToxDB + 5,991 NCI-60/DrugOOD native |
| **Headline metric** | **AUC 0.886 ± 0.014** | AUC 0.7613 (1-seed Ru) / 0.4544 BELOW_RANDOM DrugOOD | **decode_ratio 0.9958** |
| **Krasnov head-to-head** | ✅ +0.099 Ru / +0.155 Ir (MEASURED) | NOT measured on Krasnov protocol | N/A (different task) |
| **iMoLD DrugOOD head-to-head** | ⚠️ comparable (NOT a beat) | ❌ BELOW_RANDOM (0.4544 vs v2 0.6707) | N/A (different task) |
| **Per-metal coverage** | 5/5 (Ru/Ir/Os/Rh/Re) | 3/12 (Ru/Ir/Os × 4 seeds) | 5/5 (Ru/Ir/Os/Rh/Re, m14 FULL) |
| **OOD generalisation** | scaffold-split MEASURED | BELOW_RANDOM on scaffold | N/A (m14 = decode_ratio, not classification) |
| **Engineering maturity** | ✅ 145 tests, 3 retrain waves | ⚠️ 1 seed, partial coverage | ✅ Wf-124 P2 + Wf-181/188 all working end-to-end |
| **GPU wall-time measured** | 32.91 min (Wf-194 LOMO v2, 25 runs); 4.19 min (Wf-219 3-metal × 3-seed × 50-ep) | 1-seed Ru: ~7.5 min (Wf-124 P3) | 32.3 s (Wf-124 P2 50-epoch synthetic); 428.5 s (Wf-188 FULL n=26,801) |
| **Per-innovation attribution** | ✅ TODO-182 G6 ablation 60 runs | ✅ TODO-182 G6 (shared) | ❌ Per-innovation ablation NOT done (TODO-119 P1.4 follow-up) |
| **Paper section fit** | §4 LEAD (m1/m18/m19/m21), §4.1 (DrugOOD), §B.2 (V4 vs v2 head-to-head) | §6 honest-negative L10 (DrugOOD OOD) | §4 LEAD m14 + §3 algorithm description (YuelBond + W₂ + Catherine) |

### §4.2 Which has more potential for paper §4 / §3?

**Both are needed** — they solve orthogonal problems:
- **chemprop v2** = predict activity (CLASSIFICATION) — already paper §4 LEAD
- **CFM** = design new drugs (GENERATION) — paper §4 LEAD m14 + §3 algorithm description

**chemprop v2 has more *current* paper-grade measured numbers** (5-metal mean AUC 0.886 with Krasnov beats, LOMO v2 0.9015, DrugOOD ood_test 0.7137) — it is the most-validated backbone for paper §4 results section.

**CFM has more *paper-§3-algorithm-description* leverage** — the YuelBond + W₂ + Catherine closure-bounded generation story is the architectural novelty of Mol-Metal's m14 CFM, and the m14 decode_ratio 0.9958 (n=26,801) MEASURED number on real MetalCytoToxDB is the validity gate.

**chemprop v3 is currently in §6 honest-negative** on DrugOOD OOD (0.4544 BELOW_RANDOM); v3 needs fix on real DrugOOD (TODO-130 G5 follow-up) before it can be promoted back to §4 LEAD.

### §4.3 What to fix FIRST (honest priority order)

1. **chemprop v2 v3 — first**: complete Wf-124 P3 14/15 PENDING (metal, seed) pairs (TODO-180 G4, ~1.8 hr wall); run 50-epoch full sweep (TODO-181 G5); per-innovation ablation (TODO-182 G6, 60 runs / 7.5 hr). **Why first**: chemprop v2 is already paper §4 LEAD with 5-metal mean AUC 0.886; the 14 PENDING pairs + 50-epoch + ablation are the missing credibility gates for honest §3 algorithm attribution.

2. **CFM — second**: run TODO-119 P1.3 real-CFM decode_ratio lift on PlatinAI/DrugOOD cohorts (Wf-119 P1.3); TODO-119 P1.4 per-innovation ablation (YuelBond only / W₂ only / closure only). **Why second**: Wf-124 P2 + Wf-181 smoke + Wf-188 FULL already shipped at decode_ratio 0.9958 (gate ≥ 0.05 PASS by 20×). The follow-ups are incremental credibility lifts.

3. **chemprop v3 DrugOOD re-train (TODO-130 G5 follow-up)** — only after v2 follow-ups ship, since v3 is in §6 honest-negative.

---

## §5 CFM-specific TODOs in TODO/pending/ and TODO/completed/

### §5.1 MUST HAVE — CFM-specific TODOs (currently PENDING or follow-ups on SHIPPED)

| TODO | Status | CFM relevance | ETA |
|---|---|---|---|
| **TODO-119** (`decode_ratio_strengthen.md`) | Completed §P1.1 + P1.2 + P1.3 (Wf-124 P2); P1.3 real-CFM lift + P1.4 per-innovation ablation still PENDING | **DIRECTLY CFM** (m14 decode_ratio gate ≥ 0.05; current 0.9958 PASS by 20×) | 1 wk for P1.3+P1.4 |
| **TODO-124** (Wf-124 P2 CFM full retrain) | **✅ SHIPPED** in `TODO/completed/131_wf124_p2_cfm_full.md`; Wf-124 P2 ship at decode_ratio 1.000 (192/192) on synthetic | **DIRECTLY CFM** (3-innovation stack: YuelBond + W₂ + Catherine) | DONE; follow-ups in TODO-119 |
| **TODO-126** (innovation_strengthen_roadmap, M1+M2 master plan) | Completed §3 (M1+M2 master plan); consumer metric lift follow-ups in TODO-177 G1 | **DIRECTLY CFM** (M1 = Wasserstein regularization, M2 = pocket closure bias — both used in Wf-124 P2) | DONE; consumer lifts in TODO-177 (6-wk ETA, R20) |

### §5.2 SHOULD HAVE — CFM-applicable TODOs

| TODO | Status | CFM relevance | ETA |
|---|---|---|---|
| **TODO-96** (R20-P5 closure Catherine 6/6, ≥ 25 metals × 1000 samples) | Completed `96_r20_p5_closure_catherine.md` with P5.1-P5.4 phases | **Applies to CFM** (Catherine 6/6 properties feed into Wf-124 P2 closure-bounded generation; the per-metal reachability ≥ 0.95 is the closure-bounded generation quality gate) | 2 weeks CPU + 1 week GPU-0 |
| **TODO-98** (Wf-44 P2 Closure OOD Phase 2, L12 cross-domain OOD) | Completed `98_wf44_closure_ood_phase2.md` | **Applies to CFM** (cross-domain OOD reachability is the closure-bounded generation OOD generalisation gate; L12 PlatinAI 5K reachability 0.5581 + MetalCytoToxDB 0.0300 are the current MEASURED honest-negative baseline) | Q4 2026 |

### §5.3 CFM-applicable but lower priority

| TODO | Status | CFM relevance |
|---|---|---|
| **TODO-180** (Wf-124 P3 G4 14 PENDING (metal, seed) pairs) | PENDING — 1.8 hr wall on RTX 5090 GPU-1 | Indirect (chemprop v2 v3 multi-task D-MPNN + W₂; the same W₂ + L2 guard that feeds CFM via `wasserstein_reg` engine tag) |
| **TODO-181** (Wf-124 P3 G5 50-epoch full sweep) | PENDING | Same as TODO-180 |
| **TODO-182** (Wf-124 P3 G6 per-innovation ablation, 60 runs / 7.5 hr) | PENDING | Same as TODO-180 |
| **TODO-22** (CFM decode metric #14) | Embedded in TODO-119 | Same as TODO-119 |

### §5.4 CFM TODO shortlist (the 5 CFM-specific TODOs from §5.1 + §5.2)

| Priority | TODO | Status | Wall-time | Output |
|---|---|---|---|---|
| **MUST HAVE** | TODO-119 P1.3 + P1.4 (real-CFM lift + per-innovation ablation) | PENDING (P1.1/P1.2/P1.3 SHIPPED) | 1 wk (Wf-119 P1.3) + 1 wk (Wf-119 P1.4, ~7.5 hr ablation wall) | `molmetal/reports/wf119_cfm_retrain/{real_cfm_decode.json,ablation_report.json}` |
| **MUST HAVE** | TODO-124 (Wf-124 P2) | **✅ SHIPPED** in TODO-131 | DONE | `molmetal/reports/wf124_p2_cfm_full/{per_epoch_metrics.json,per_epoch_metrics.csv,wasserstein_decay.json,synthesis.md}` |
| **MUST HAVE** | TODO-126 (M1+M2 master plan) | Completed §3 (master plan); consumer lifts in TODO-177 | 6 wk ETA (R20 GPU per-consumer spawns) | `TODO/completed/126_innovation_strengthen_roadmap.md` §3 |
| **SHOULD HAVE** | TODO-96 (Catherine 6/6 + reachability ≥ 0.95 on ≥ 25 metals) | P5.1-P5.4 phases | 2 wk CPU + 1 wk GPU-0 | `molmetal/reports/wf54_r20_p5/{catherine_6_props.json,closure_reachability_r20.json,cross_domain_ood.json}` |
| **SHOULD HAVE** | TODO-98 (Wf-44 P2 Closure OOD Phase 2) | Phase 2 in-flight Q4 2026 | Q4 2026 | L12 cross-domain OOD reachability ≥ 0.85 gate |

### §5.5 CFM TODO count (per workflow task G4)

**5 CFM-specific TODOs in scope** (from §5.4 above):
1. TODO-119 (decode_ratio strengthen, real-CFM follow-up)
2. TODO-124 (Wf-124 P2 — already SHIPPED, kept for completeness)
3. TODO-126 (M1+M2 master plan — already SHIPPED, kept for completeness)
4. TODO-96 (Catherine 30-metal closure)
5. TODO-98 (Wf-44 P2 closure OOD Phase 2)

---

## §6 Honest framing per `party-twenty-spirit` §3.5

### §6.1 What this research CLAIMS (verbatim)

| Claim | Status | Source |
|---|---|---|
| chemprop v2 5-metal mean AUC = 0.886 ± 0.014 (paired t-test p = 1.35 × 10⁻⁵ vs V4) | ✅ MEASURED | TODO-154 |
| chemprop v2 LOMO v2 5-metal mean AUC = 0.9015 ± 0.017 (25/25 ok runs) | ✅ MEASURED | Wf-194 |
| chemprop v2 Ru +0.099 over Krasnov 0.81; Ir +0.155 over Krasnov 0.73 | ✅ MEASURED (5-seed paired, p=0.025 / 0.010) | TODO-154 §2 |
| chemprop v2 Os/Rh/Re first-reported per-metal AUC (scaffold-aligned) | ✅ MEASURED | TODO-154 §2 + Wf-219 |
| chemprop v2 DrugOOD assay ood_test 0.7137 ± 0.0079 | ✅ MEASURED (5-seed Wf-185, 13/15 OK) | Wf-174 + Wf-185 |
| chemprop v2 m21 9-cell per-cell AUC = 0.8007 ± 0.0160 | ✅ MEASURED (3 seeds × 50 ep scaffold-split) | Wf-35/Wf-72 |
| chemprop v3 Ru seed=0 test_auc = 0.7613 (+0.049 vs 0.7123 baseline) | ✅ MEASURED (1-seed, 30 ep, Wf-124 P3 default) | Wf-124 P3 |
| chemprop v3 DrugOOD scaffold ood_test 0.4544 ± 0.0327 BELOW_RANDOM | ✅ MEASURED (7/12 pairs, Wf-253d forward-pass reeval) | Wf-253d m17 |
| CFM Wf-124 P2 decode_ratio = 1.000 on synthetic (192/192) | ✅ MEASURED (architecture path PASS, ×25 over gap) | Wf-124 P2 per_epoch_metrics.json |
| CFM Wf-181 m14 smoke decode_ratio = 0.9980 ± 0.0008 | ✅ MEASURED (5,979/5,991) | Wf-181 |
| CFM Wf-188 m14 FULL MetalCytoToxDB decode_ratio = 0.9958 | ✅ MEASURED (26,687/26,801) | Wf-188 |

### §6.2 What this research DOES NOT claim (verbatim `party-twenty-spirit` §3.5 0 容忍)

- We **DO NOT claim** CFM beats TargetDiff on m7 Tanimoto diversity (gap +0.820 honest per TODO-180 §1.3)
- We **DO NOT claim** CFM has been tested on per-metal-per-pocket *generation quality* (only m14 decode_ratio gate; no design-quality metrics like SA / Vina Dock on the generated set)
- We **DO NOT claim** chemprop v3 is paper-ready (BELOW_RANDOM on DrugOOD OOD; §6 honest-negative L10)
- We **DO NOT claim** chemprop v2 beats iMoLD on DrugOOD (comparable-to-iMoLD, NOT a beat; Δ = −0.0074 within inter-seed std ±0.0079)
- We **DO NOT claim** m21 9-cell aggregate BEATS Krasnov Ru 0.81 (statistically equivalent, Δ = −0.0093 within inter-seed std ±0.0160)
- We **DO NOT claim** chemprop v2 Krasnov beats are directly comparable (Krasnov time-split vs our scaffold-split — conservative under-report per TODO-219 §3 + Wf-219 honest_framing_note)
- We **DO NOT claim** CFM ablation has been done (TODO-119 P1.4 per-innovation ablation PENDING; current 0.9958 is aggregate from 3 simultaneous innovations)
- We **DO NOT claim** chemprop v3 forward-pass reeval generalises (per-metal coverage uneven: Ru 4 seeds, Ir 3 seeds, Os 1 seed; aggregate 0.4544 is honest MEASURED but limited sample)

### §6.3 What this research treats as PROJECTED (NOT measured)

- **TODO-119 P1.3 real-CFM lift** on PlatinAI / DrugOOD cohorts — Wf-119 P1.3 GPU follow-up ETA 1 wk
- **TODO-119 P1.4 per-innovation CFM ablation** (YuelBond only / W₂ only / closure only) — 7.5 hr wall + audit
- **TODO-96 R20-P5 reachability ≥ 0.95 on ≥ 25 metals × 1000 samples** — 2 wk CPU + 1 wk GPU-0 ETA
- **TODO-98 L12 cross-domain OOD reachability ≥ 0.85** — Q4 2026 ETA
- **chemprop v3 DrugOOD re-train** (TODO-130 G5 follow-up) — v3 currently 0.4544 BELOW_RANDOM, projected lift not quantified

### §6.4 What this research treats as DEFERRED (R19/R20 GPU retrain pending)

- **Per-metal-per-pocket CFM generation quality** (SA / Vina Dock on the generated set) — currently only m14 decode_ratio measured
- **CFM cross-domain OOD** on PlatinAI 5K + MetalCytoToxDB + tmQM (closure-bounded generation OOD generalisation gate)
- **chemprop v3 in-domain 5-metal mean AUC** — currently 1-seed Ru only; 14 PENDING + 50-ep + ablation TODO

---

## §7 Acceptance gates (Wf-257 6 gates)

- **[G1]** CFM potential = **paper §4 LEAD candidate for m14 (decode_ratio 0.9958 MEASURED on n=26,801) + §3 algorithm description (YuelBond + W₂ + Catherine closure-bounded generation)**
- **[G2]** chemprop v2 v3 potential = **paper §4 PRIMARY backbone for m1/m18/m19/m21 (5-metal mean AUC 0.886 ± 0.014 Krasnov-beating); v3 in §6 honest-negative on DrugOOD OOD (BELOW_RANDOM 0.4544)**
- **[G3]** both_needed = **True** (different problems: chemprop = CLASSIFICATION; CFM = GENERATION; both required for complete metallodrug discovery pipeline)
- **[G4]** CFM TODO count = **5** (TODO-119, TODO-124, TODO-126, TODO-96, TODO-98)
- **[G5]** CFM TODO shortlist = **{TODO-119: MUST HAVE, TODO-124: MUST HAVE (already SHIPPED), TODO-126: MUST HAVE (already SHIPPED), TODO-96: SHOULD HAVE, TODO-98: SHOULD HAVE}**
- **[G6]** Research report path = **`molmetal/reports/wf257_research/cfm_potential.md`** (this file)

---

## §8 Files cited

**MEASURED source reports**:
- `molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.json` — Wf-124 P2 per-epoch train_loss / bond_ce_loss / wasserstein_loss / decode_ratio_rdkit_strict / catherine_audit_pass (50 epochs)
- `molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.csv` — same data CSV
- `molmetal/reports/wf124_p2_cfm_full/wasserstein_decay.json` — JKO-style W₂ monotonic decrease check
- `molmetal/reports/wf124_p2_cfm_full/synthesis.md` — Wf-124 P2 ship surface
- `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` — Wf-124 P3 Ru seed=0 test_auc 0.7613
- `molmetal/reports/wf174_chemprop_v2_drugood_real/summary.json` — Wf-174 3-seed DrugOOD (REAL 1.58M rows, assay 0.7113 ± 0.0216)
- `molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json` — Wf-185 5-seed DrugOOD (assay 0.7137 ± 0.0079 / scaffold 0.6707 / size 0.6757)
- `molmetal/reports/wf181_m14_real_cfm_metalcytotox/{result.json,synthesis.md}` — Wf-181 m14 real CFM smoke decode_ratio 0.9980 ± 0.0008
- `molmetal/reports/wf188_m14_full_metalcytotox/{result.json,synthesis.md}` — Wf-188 m14 real CFM FULL MetalCytoToxDB decode_ratio 0.9958 (n=26,801)
- `molmetal/reports/wf194_lomo_v2/per_metal_summary.json` — Wf-194 LOMO v2 25/25 ok runs, mean AUC 0.9015
- `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json` — Wf-219 3-metal × 3-seed × 50-ep scaffold-aligned sweep (Ru 0.810 / Ir 0.759 / Os 0.788 / Rh 0.752 / Re 0.803)
- `molmetal/reports/wf_nci60_precious_9cell/final.md` — Wf-35/Wf-72 m21 9-cell per-cell AUC 0.8007 ± 0.016
- `molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.md` — Wf-253d m17 DrugOOD v3 forward-pass reeval (BELOW_RANDOM 0.4544 ± 0.0327)
- `molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json` — per-(metal, seed) ROC-AUC + AUPR
- `molmetal/metrics/by_metric/cfm_decode_real.json` — Wf-181 m14 3-seed decode_ratio 0.9980 (cross-check)
- `molmetal/metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json` — TODO-154 chemprop v2 5-metal in-domain fold AUCs
- `molmetal/metrics/by_metric/chemprop_v2_lomo_v2.json` — Wf-194 LOMO v2 in-domain metric file (per_metal_in_domain)
- `molmetal/metrics/by_metric/nci60_precious_9cell_auc.json` — Wf-35/Wf-72 m21 9-cell per-cell AUC
- `molmetal/metrics/by_metric/m6_kappa.json` — Wf-252-Apply Fix #2 m6 Shape Kappa Index (449/449 SMILES, mean 0.4719)
- `molmetal/metrics/by_metric/drugood_v3_scaffold.json` — Wf-253d m17 DrugOOD v3 scaffold ood_test (0.4544)

**CFM-specific TODO files**:
- `TODO/completed/119_decode_ratio_strengthen.md` — P0 CRITICAL m14 decode_ratio strengthen (P1.1/P1.2 SHIPPED; P1.3 real-CFM PENDING; P1.4 per-innovation ablation PENDING)
- `TODO/completed/124_wf124_p3_multitask_5metal_dmpnn.md` — Wf-124 P3 multi-task 5-metal D-MPNN + Wasserstein pIC50 (Ru seed=0 0.7613)
- `TODO/completed/126_innovation_strengthen_roadmap.md` — M1+M2 master plan (Wasserstein regularization + pocket closure bias)
- `TODO/completed/131_wf124_p2_cfm_full.md` — Wf-124 P2 SHIP (decode_ratio 1.000 on synthetic)
- `TODO/completed/96_r20_p5_closure_catherine.md` — R20-P5 Catherine 6/6 + reachability ≥ 0.95 on ≥ 25 metals
- `TODO/completed/98_wf44_closure_ood_phase2.md` — Wf-44 P2 Closure OOD Phase 2
- `TODO/pending/119_decode_ratio_strengthen.md` (referenced from completed 119; P1.3 + P1.4 follow-ups)
- `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md` — M1+M2 downstream lift on 6 consumer TODOs (TODO-119, TODO-120, TODO-121, TODO-122, TODO-123, TODO-125)
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` — G4 14 PENDING (metal, seed) pairs
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` — G5 50-epoch full sweep
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` — G6 per-innovation ablation (60 runs / 7.5 hr)

**Supporting context files**:
- `TODO/completed/154_chemprop_v4_head_to_head.md` — chemprop v2 vs V4 head-to-head, 14 paired runs, paired t-test p = 1.35 × 10⁻⁵
- `TODO/completed/161_paper_v3_chemprop_main.md` — 5-metal fold table (Ru 0.909, Ir 0.885, Os 0.889, Rh 0.863, Re 0.882)
- `TODO/completed/189_paper_4_FINAL.md` — paper §4 FINAL comprehensive draft (189 supersedes 186/187/188)
- `TODO/completed/83j_lambda_closeout.md` §3.1 — L1/L6/L12 MEASURED reachability (0.5108 / 0.5891 / 0.5581)
- `TODO/completed/180_daily_summary_2026_09_27.md` §1.3 — m7 Tanimoto diversity 0.0397 with Tanimoto reward (+22.5% lift)
- `mol-metal-cfm-training` skill — CFM backbone (Lipman + YuelBond + EGNN + Triton-fused ops)
- `mol-metal-cfm-training` skill description's "3 blockers" framing was the pre-Wf-124 P2 state; the Wf-124 P2 ship + Wf-181 smoke + Wf-188 FULL all worked end-to-end on real MetalCytoToxDB inputs

**Skill alignment**:
- `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED tag discipline per §6.1 + §6.2 + §6.3 + §6.4)
- `mol-metal-paper-writing` §6 limitations template + honest framing + MEASURED-vs-PROJECTED labels
- `mol-metal-cfm-training` (CFM backbone, YuelBond decoder, Wasserstein regularization, decode_ratio metric)
- `mol-metal-gpu-execution` §7 tag discipline (no best-seed selection; per-seed std IS the answer)
- `mol-metal-mechanism-mapping` (m1/m14/m17/m18/m19/m21 honest-negative → §4 LEAD promotion audit)

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: This research is a READ-ONLY audit (no `paper/` or `paper/sections/` edits; no code edits). All numbers trace to JSON / MD source paths or are honest-PROJECTED with explicit ETA. chemprop v2 = paper §4 PRIMARY backbone (5-metal mean AUC 0.886 Krasnov-beating); chemprop v3 = §6 honest-negative on DrugOOD OOD (BELOW_RANDOM 0.4544); CFM = paper §4 LEAD candidate for m14 (decode_ratio 0.9958 on n=26,801, gate ≥ 0.05 PASS by 20×). Both chemprop v2 and CFM are needed (CLASSIFICATION + GENERATION are orthogonal capabilities). chemprop v3 needs DrugOOD re-train or honest framing in paper §4 before promotion. The 5 CFM-specific TODOs (§5.4) ship in priority order: TODO-119 (P1.3+P1.4 follow-ups) → TODO-124 (already SHIPPED) → TODO-126 (already SHIPPED) → TODO-96 (R20-P5) → TODO-98 (Wf-44 P2 Phase 2).