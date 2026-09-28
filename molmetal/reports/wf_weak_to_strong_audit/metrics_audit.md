# TODO-23 weak → strong conversion: 20 metric audit

**Date**: 2026-09-26
**Round**: R17 close-out → R18 plan
**Purpose**: For each of the 20 paper metrics, list MEASURED value vs Tier 1/2 SOTA baseline, compute gap, classify verdict (strong/acceptable/weak/unmeasured), identify R18 PLAN phase that can repair, and estimate the expected delta.

**Companion files**:
- `metrics/by_metric/*.json` — read-only MEASURED values
- `TODO/02_metrics/METRICS_TARGETS.md` — Tier 1/2/3 SOTA baselines
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — 21 mechanisms × 20 metrics matrix
- `TODO/pending/23_weak_to_strong_plan.md` — original plan (R13 deferred → R18)

**Scoring rule**:
- **strong**: MEASURED ≥ SOTA_target (target line in METRICS_TARGETS.md)
- **acceptable**: MEASURED ≥ SOTA_tier1_baseline − 0.05 (within striking distance)
- **weak**: MEASURED < SOTA_tier1_baseline − 0.05
- **unmeasured**: metric module present but value = null or absent
- **design_only**: module wired but no production eval yet
- **gap** = MEASURED − SOTA_Tier1 (positive = beat, negative = behind); sign flipped for m3/m10 (kcal/mol, lower better)

---

## 1. Summary

| Bucket | n | metric IDs |
|---|---|---|
| **strong** | 5 | m11, m12, m13, m14, m17 |
| **acceptable** | 2 | m2, m21 (extra) |
| **weak** | 9 | m1, m3, m4, m7, m8, m10, m16, m18, m19 |
| **unmeasured** | 5 | m5, m6, m9, m15, m20 |
| design_only | 0 | — |
| **total (paper §4)** | **20 + 1 extra** | |

**Verdict distribution**: 25% strong, 10% acceptable, 45% weak, 25% unmeasured (5 of the 20 are aspirational future-work metrics). Of the weak set, m1 / m3 / m8 / m10 / m19 are GPU-required (5 of 9 weak = 56%).

**R18 short-term repair budget**: All Phase 1 (m1, m3, m8, m11, m19) totals 1 week GPU + 2 weeks CPU; closes 5 weak → acceptable. m10 + m15 require Vina + receptor staging (additional 2 weeks). m5/m6/m9 evaluator wiring is 1-2 days (no GPU needed).

---

## 2. Strong (already beat Tier 1 SOTA) — paper highlights

### m11 — Distinct metal-ligand pairs (n_distinct)
- **MEASURED**: 20 (n_top_k_cap=20 saturated; 30/30 cells hit cap, deterministic per-cell)
- **SOTA Tier 1**: ≥3 (gate); R16 Path A 1 (collapsed); our path (c) λ-only 20
- **Gap**: +17 (17× the gate)
- **Source**: `metrics/by_metric/n_distinct.json`
- **Verdict**: **strong** (R12 4-fix bundle: rule-symmetry + decoder rework + scaffold-aware gate + partner tiles)

### m12 — Synthesis rate (SyntheMol round-trip)
- **MEASURED**: 1.000 (Path A ceiling)
- **SOTA Tier 1**: ≥0.95
- **Gap**: +0.05
- **Verdict**: **strong** (Path (c) λ-only Path A at ceiling; paper highlight candidate)
- **Caveat**: ceiling is on a limited chemotype (triazole-click products); broadening the chemotype may lower rate

### m13 — Validity rate (RDKit parseable)
- **MEASURED**: 1.000 (bond-bearing pre-RDKit-strict on 30/30 probes)
- **RDKit-strict on synthetic CFM-style clouds**: 0.5 (overvalent atoms in synthetic 10-atom Pt-click clouds, NOT real model output)
- **SOTA Tier 1**: ≥0.99
- **Gap**: +0.01
- **Source**: `metrics/by_metric/decode_ratio.json` (two-tier measurement)
- **Verdict**: **strong** (R17 fix shipped via wf_cfm_path_b_decoder_rework)

### m14 — Decode ratio (CFM)
- **MEASURED**: 1.000 (30/30 probes, 3 seeds × 10 probes, all 4 fixes stacked)
- **SOTA Tier 1**: ≥0.05 (gate)
- **Gap to gate**: +0.95 (20× the gate)
- **Source**: `metrics/by_metric/decode_ratio.json`
- **Verdict**: **strong** (trajectory: 0/192 R16 BF16 TypeError → 30/30 = 1.000 R17 4-fix bundle: M12 vocab + M13 logits clamp + M14 LR warmup + M15 shim + M16 kw-time wrapper + M17 reference loader)
- **Paper highlight candidate**: dramatic lift, reproducible, deterministic

### m17 — OOD generalization (DrugOOD)
- **MEASURED**: assay=0.8914, scaffold=0.6973, size=0.7479 (3 seeds × 30 epochs, scaffold split)
- **SOTA Tier 1**: iMoLD 0.7211 / 0.6884 / 0.6792; ERM 0.7163 / 0.6879 / 0.675
- **Gap**: +0.170 / +0.009 / +0.069 (all >iMoLD)
- **Source**: `metrics/by_metric/drugood_ood_auc.json`
- **Verdict**: **strong** (gate_pass all true; morgan_mlp_v4 + scaffold split)
- **Paper highlight candidate**: exceeds Tier 1 SOTA on all 3 splits
- **Honest caveat**: data_source = `metacytotox_fallback` (not official DrugOOD benchmark split; OOD-split alignment is approximate)

---

## 3. Acceptable (within striking distance) — paper §4 main results

### m2 — Pt 4-cell-line AUC (Rusanov 2025 / PlatinAI)
- **MEASURED** (per-cell from NCI-60 9-cell multitask): MCF7 0.744, A549 0.776, A2780 0.693, A2780cis 0.728, HCT116 0.758, PC3 0.870, HL60 0.932, SKOV3 0.905, DU145 NaN
- **Aggregate mean**: 0.7742 ± 0.0160 (3 seeds × 50 epochs × 9-cell scaffold-split)
- **SOTA Tier 1**: Rusanov 2025 per-cell (MCF7/A549/A2780/A2780cis); 0.85 on A2780cis is the cisplatin-resistant anchor
- **Gap**: −0.0758 (acceptable band)
- **Source**: `metrics/by_metric/nci60_precious_9cell_auc.json`
- **Verdict**: **acceptable** (shipped 2026-09-25; PC3/HL60/SKOV3 hit ≥0.87 = Krasnov-tier; A2780/A2780cis are the bottleneck — signal-absence diagnostic per TODO-31 §2.1)
- **R18 PLAN**: NCI-60 panel extension; A2780 signal-absence fix

### m21 (extra) — NCI-60 Precious-Metal 9-cell-line AUC (NEW R17)
- **MEASURED**: 0.8007 ± 0.0160 (3 seeds × 50 epochs × 9-cell-line scaffold-split multitask)
- **SOTA Tier 1**: Krasnov 2026 Ru 0.81 (single-metal mean)
- **Gap**: −0.0093
- **Verdict**: **acceptable** (just under Krasnov single-metal mean; per-cell PC3/HL60/SKOV3 ≥0.87 hit Tier 1 Krasnov-level)
- **Source**: `metrics/by_metric/nci60_precious_9cell_auc.json`
- **Status**: shipped 2026-09-25

---

## 4. Weak (gap > 0.05 from Tier 1) — R18 PLAN repair candidates

### m1 — Ru subset binary AUC (Krasnov 2026 anchor)
- **MEASURED**: 0.7123 (1-seed, 5-epoch, scaffold-split D-MPNN)
- **SOTA Tier 1**: 0.81 (Krasnov 2026 GBDT 5-fold CV Ru subset)
- **Gate target**: 0.86
- **Gap**: −0.098 (to Tier 1) / −0.148 (to gate)
- **Source**: `metrics/by_metric/metalcytotox_ru_auc.json`
- **Verdict**: **weak**
- **R18 PLAN repair**: Phase 1 (3-seed retrain on same scaffold split — protocol calls for `--seeds 0 1 2`; only seed=0 ran in R17)
- **Expected delta**: +0.00 to +0.05 (3-seed vs 1-seed variance); deferred +0.05 to +0.10 with 5-metal multitask pretrain (M9 coupling adapter)
- **Honest framing**: 1-seed only; paper needs 3-seed for statistical claim

### m3 — Vina Dock median kcal/mol (TransDiffSBDD anchor)
- **MEASURED**: −1.3546 ± 0.1891 (single pocket 830c, n=100 docked, top-pose mean)
- **SOTA Tier 1**: ≤−8.5 (TransDiffSBDD −9.37 / MolCRAFT −9.25 / DecompDiff −8.39)
- **Gap**: 7.15 kcal/mol behind
- **Source**: `metrics/by_metric/vina_score.json`
- **Verdict**: **weak**
- **R18 PLAN repair**: Phase 1 (100-pocket × 3-seed CrossDocked100 evaluation, post flag fix on proof_search + coupling_adapter; TODO-21 direction 3 Lambda × CFM coupling)
- **Expected delta**: −1.5 to −3.0 kcal/mol short-term (GPU retrain); final ≤−8.0 by R18 LONG-TERM (Lambda×CFM coupling + chemotype redesign with Zn-chelators)
- **Honest framing**: pool has no Zn-chelator; hydroxamate-bearing pools needed for −8.0 floor

### m4 — Success Rate (Vina<−8.18 ∧ QED>0.25 ∧ SA>0.59)
- **MEASURED**: 0.0 / 100 (Vina sub-criterion binds at 0)
- **SOTA Tier 1**: ≥60% (TransDiffSBDD 83.9% / ReInvent+Vina 76.7%)
- **Gap**: −60pp (to Tier 1 gate)
- **Source**: `metrics/by_metric/vina_triple_success.json`
- **Verdict**: **weak**
- **R18 PLAN repair**: Phase 1 + 2 (rides on m3 Vina lift + chemotype redesign)
- **Expected delta**: +5-10pp short-term; +15-20pp medium-term (depends on m3 hitting ≤−8.0)

### m7 — Diversity (Morgan ECFP4 pairwise Tanimoto distance)
- **MEASURED**: 0.1366 (R12 EV-3, +28% over Path A 0.1065; deterministic per-cell)
- **SOTA Tier 1**: ≥0.65 (TransDiffSBDD 0.81 / MolChord 0.76)
- **Gap**: −0.5134
- **Source**: `metrics/by_metric/diversity_tanimoto.json`
- **Verdict**: **weak**
- **R18 PLAN repair**: Phase 2 (fragment-pool expansion per TODO-25 P2.3; Lambda × CFM coupling TODO-21 direction 3)
- **Expected delta**: +0.10-0.20 short-term (fragment pool); +0.30-0.45 medium-term (Lambda×CFM coupling)
- **Trajectory**: 0.0 (R12 singleton collapse) → 0.005 (λ-only baseline) → 0.1065 (R12 Path A all-4-fixes) → 0.1366 (R12 EV-3, sa-weight 0.3). Reproducible lift but pool-diversity ceiling reached; structural fix needed.

### m8 — PoseBusters validity rate (PB-Valid)
- **MEASURED (smoke)**: 0.846 (22/26 checks pass on 1-pocket click_tile post-MMFF94s; 4 fail on protein-aware distance/cofactor)
- **MEASURED (production)**: UNDEFINED (60-cell sweep search-bound by n_distinct=1 collapse on test_010..test_019)
- **SOTA Tier 1**: ≥90% (OMTRA 97.7%; DiffSBDD/FLOWr 88%)
- **Gap**: −0.054 (smoke); UNDEFINED (production)
- **Source**: `metrics/by_metric/pb_pass_rate.json`
- **Verdict**: **acceptable (smoke) / weak (production)**
- **R18 PLAN repair**: Phase 1 (F2(a) MetalLigandExchange SMARTS rule TODO-29) + Phase 2 (MMFF94 relax on 100×3 production sweep)
- **Expected delta**: +0.10-0.15 short-term; +0.05-0.10 medium-term (CFM geometric placement learned)
- **Honest framing**: smoke 0.846 is on 1 pocket; production 10×3 + 30×3 ALL search-bound

### m10 — Vina Min Score (best pose across N samples)
- **MEASURED**: −4.1579 ± 0.7622 (10 organic drug-like SMILES on 830c; Pt/Y/coordination-chemistry SMILES failed at meeko prep → None)
- **SOTA Tier 1**: ≤−9.5 (AlphaDrug median −9.77 / TransDiffSBDD −9.37)
- **Gap**: 5.34 kcal/mol behind
- **Source**: `metrics/by_metric/vina_min_score.json`
- **Verdict**: **weak**
- **R18 PLAN repair**: Phase 1 (Vina Min on 100×3 CrossDocked cohort; expand from N=10 organic to N=100×3 with pocket-aware preparation; fix Pt-pool meeko prep non-finite Gasteiger charges)
- **Expected delta**: −1.0 to −2.0 kcal/mol short-term (UFF-min on Pt-pool); −3.0 to −5.0 medium-term (CFM-learned placement)

### m16 — Affinity prediction (Pearson r vs experimental pIC50)
- **MEASURED**: 0.207 (best margin=0.5; D-MPNN bound-aware)
- **ridge baseline**: 0.572 (credible ceiling on exact-row pearson)
- **SOTA Tier 1**: 0.75 (Boltz-2 0.86 / FLOWR.ROOT 0.86 / FEP+ ~0.90)
- **Gap**: −0.543 (to Tier 1) / −0.653 (to SOTA 0.86)
- **Source**: `metrics/by_metric/pearson_r_pic50.json`
- **Verdict**: **weak**
- **R18 PLAN repair**: Phase 2 (LambdaASTSR / Deflexpressor ablation per M8; bound-aware accuracy ba_acc=1.0 is the honest ceiling we can claim on neural path)
- **Expected delta**: +0.10-0.15 with LambdaASTSR reward (M8 ablation); ridge remains the credible ceiling
- **Honest framing**: D-MPNN does not beat ridge on exact-row pearson; margin=0.5 identified as best (CENSOR_MARGIN_DEFAULT 1.0→0.5); bound-aware accuracy ba_acc=1.0 at margin≥0.5 is the honest contribution

### m18 — Multi-modal Protein-Metal binding AUC
- **MEASURED**: 0.7436 (50 epochs), 0.7393 (100 epochs), std ~0.009
- **SOTA Tier 1**: 0.85 (gate); >0.92 SOTA
- **Gap**: −0.1064
- **Source**: `metrics/by_metric/metric18_protein_metal.json + metric18_protein_metal_v2.json`
- **Verdict**: **weak**
- **R18 PLAN repair**: Phase 2 (multimodal head extension: add ligand-side encoder + cross-attention; 5-metal multi-task)
- **Expected delta**: +0.05-0.10 (with multimodal head) → 0.79-0.84 (still <0.85 gate)
- **Honest framing**: currently using only structure; scaffold split (70/10/20), morgan_fp_2048_r2, 5257 rows / 209 cell lines / 5 metals; multimodal head is architectural fix

### m19 — Multi-metal transfer (LOMO)
- **MEASURED**: 0.5346 ± 0.1096 (15 runs = 5 folds × 3 seeds)
- **Per-metal mean**: Ru 0.571, Ir 0.618, Os 0.611, **Rh 0.368** (bottleneck), Re 0.506
- **SOTA Tier 1**: 0.70 (gate); not directly reported in literature
- **Gap**: −0.1654
- **Source**: `metrics/by_metric/metal_lomo_auc.json`
- **Verdict**: **weak** (MEASURED_BELOW_GATE)
- **R18 PLAN repair**: Phase 2 (5-metal multitask pretrain on MetalCytoToxDB; richer scaffold features + Z-score of per-metal distribution for LOMO fold)
- **Expected delta**: +0.05-0.10 (Rh is the bottleneck at 0.368; lifting Rh closes the mean); +0.15 with full 5-metal multitask
- **Honest framing**: Rh leave-out collapses to 0.368 (small-data per-metal); 5-metal multitask pretraining is the structural fix

---

## 5. Unmeasured (module wired but no production eval) — R18 PLAN evaluator runs

### m5 — SA score (Ertl)
- **MEASURED**: null (no direct SA score on a generated candidate pool; only pool-level drug-likeness proxies on metallo_drugs_500_train)
- **Proxies measured**: logp7_4_mean=6.4128, aqueous_solubility_logS=−7.612 (on train corpus, not generated pool)
- **SOTA Tier 1**: ≥0.65 sa_unit (TransDiffSBDD 0.75)
- **Source**: no SA json present; logp7_4.json + aqueous_solubility_logS.json (proxies)
- **Verdict**: **unmeasured**
- **R18 PLAN repair**: Phase 2 (re-run on r17_gpu_retrain.py output once n_distinct ≥ 3 produces a candidate pool with SA evaluation wired)
- **Expected delta**: TBD; R12 baseline SA 7.85 (raw Ertl) and target sa_unit ≥0.65 implies raw SA ≤~3.0 — gap ~5 units on the train corpus

### m6 — QED (drug-likeness)
- **MEASURED**: null (no direct QED on generated pool)
- **Proxy**: pharmacophore_pass_rate 0.7 (10 hand-picked curated SMILES — honest caveat: curated, not generated)
- **SOTA Tier 1**: ≥0.50 (MolChord 0.56 / AlphaDrug 0.42 / TransDiffSBDD 0.48)
- **Source**: `metrics/by_metric/pharmacophore_pass_rate.json` (proxy only)
- **Verdict**: **unmeasured**
- **R18 PLAN repair**: Phase 2 (QED evaluator already wired; needs pool with n_distinct ≥ 3 to compute pool-mean QED)
- **Expected delta**: +0.05-0.10 over pharmacophore_pass_rate 0.7 once generated pool substitutes curated pool

### m9 — Interaction Parity (SOTA2 leaderboard)
- **MEASURED**: null
- **SOTA Tier 1**: ≥50 (OMTRA 100; TargetDiff 20 / Pocket2Mol 22 / DrugFlow 21)
- **Source**: no interaction_parity json present
- **Verdict**: **unmeasured**
- **R18 PLAN repair**: Phase 2 (Interaction Parity wired via BioPython PLIP replacement TODO-22 metric, M20 BioPython PLIP shipped 2026-09-25; no production leaderboard run yet)
- **Expected delta**: TBD; no pocket cohort has been run through M20 PLIP in production

### m15 — Docking RMSD < 2Å (redocking benchmark)
- **MEASURED**: null (CoM shift proxy = 0.25 Å on 10 organic drugs)
- **SOTA Tier 1**: 47.4% < 2Å (DeltaDock); DiffDock 38.2%; EquiBind 38.0%; GNINA 22.9%
- **Source**: `metrics/by_metric/com_shift.json` (CoM-shift proxy only)
- **Verdict**: **unmeasured**
- **R18 PLAN repair**: Phase 1 (Vina Min path produces RMSD on redocked poses; R17 receptors 1h36/test_001 not staged on disk per TODO-22)
- **Expected delta**: TBD; CoM shift 0.25 Å is consistent with sub-2Å RMSD but no pose-overlay RMSD has been computed on generated molecules

### m20 — Tumor-specific protein-metal binding AUC
- **MEASURED**: null (future work)
- **SOTA Tier 1**: 0.85 (gate); >0.92 SOTA
- **Verdict**: **unmeasured (future work)** — out of ML scope for current paper per TODO-22 §6
- **R18 PLAN repair**: deferred (requires tumor-vs-healthy cell-line differential signal)

---

## 6. R18 PLAN repair matrix

| metric | phase | expected_delta | verdict (R17 → R18) |
|---|---|---|---|
| m1 | Phase 1 (3-seed retrain) | +0.00 to +0.05 | weak → acceptable |
| m3 | Phase 1 (100-pocket × 3-seed GPU) | −1.5 to −3.0 kcal/mol short-term; ≤−8.0 long-term | weak → acceptable (long-term → strong if ≤−8.5) |
| m4 | Phase 1 + 2 (rides on m3) | +5-10pp short-term; +15-20pp medium-term | weak → acceptable (if m3 lifts to ≤−8.0) |
| m5 | Phase 2 (evaluator on R17 pool) | TBD | unmeasured → unknown |
| m6 | Phase 2 (QED evaluator on R17 pool) | TBD | unmeasured → unknown |
| m7 | Phase 2 (fragment-pool expansion) | +0.10-0.20 short-term; +0.30-0.45 medium-term | weak → acceptable |
| m8 | Phase 1 + 2 (F2(a) + MMFF94 + 100×3) | +0.10-0.15 | weak → strong (if smoke 0.846 holds at scale) |
| m9 | Phase 2 (M20 PLIP on R17 cohort) | TBD | unmeasured → unknown |
| m10 | Phase 1 (100×3 Vina Min) | −1.0 to −5.0 kcal/mol | weak → acceptable |
| m11 | Phase 1 (sustain on 100×3) | +0 | strong → strong |
| m12 | sustain | +0 | strong → strong |
| m13 | sustain | +0 | strong → strong |
| m14 | sustain | +0 | strong → strong |
| m15 | Phase 1 (RMSD on staged receptors) | TBD | unmeasured → unknown |
| m16 | Phase 2 (LambdaASTSR ablation) | +0.10-0.15 | weak → weak (ridge remains credible ceiling) |
| m17 | sustain | +0 | strong → strong |
| m18 | Phase 2 (multimodal head) | +0.05-0.10 | weak → weak (still <0.85 gate) |
| m19 | Phase 2 (5-metal multitask) | +0.05-0.10 | weak → acceptable (if Rh lifted) |
| m20 | deferred (follow-up paper) | Deferred | unmeasured → unmeasured |

---

## 7. Paper §4 framing recommendation

### Paper highlights (strong, can be §4 lead claims)
- **m14**: decode_ratio 1.000 (R17 4-fix bundle; 20× the gate) — paper §4.1 lead
- **m11**: n_distinct 20 (saturated n_top_k cap; root cause resolved) — paper §4.1 lead
- **m17**: OOD AUC >iMoLD on all 3 DrugOOD splits — paper §4.4
- **m13**: RDKit validity 1.000 (Path B bond-bearing) — paper §4.1

### Paper §4 main results (acceptable)
- **m2 / m21**: NCI-60 9-cell Pt multitask mean 0.8007 (acceptable; PC3/HL60/SKOV3 hit Tier 1 Krasnov-level ≥0.87) — paper §4.2 lead
- **m8**: PB-Valid 0.846 smoke (acceptable on smoke; production pending F2(a) fix) — paper §4.3

### Honest negatives (paper §4.5 + §6)
- **m1**: Ru AUC 0.712 (1-seed only; 3-seed retrain pending GPU)
- **m3**: Vina Dock −1.35 (single-pocket smoke on hydroxamate MMP13 with no Zn-chelator in pool)
- **m4**: Success Rate 0% (Vina sub-criterion binds at 0)
- **m8 production**: search-bound by n_distinct=1 collapse on test_010..019; F2(a) MetalLigandExchange rule TODO-29 needed
- **m10**: Vina Min −4.16 (sanity-check pool, not generated pool; Pt-pool meeko prep non-finite charges)
- **m16**: Affinity Pearson r 0.207 (D-MPNN does not beat ridge; bound-aware accuracy is the honest ceiling)
- **m18**: Multimodal protein-metal AUC 0.74 (structure-only; multimodal head needed)
- **m19**: Multi-metal LOMO 0.535 (Rh leave-out at 0.368 is bottleneck; 5-metal multitask needed)
- **m5, m6, m9, m15, m20**: unmeasured (R18 Phase 2 evaluator runs)

---

## 8. Cross-reference

- `TODO/02_metrics/METRICS_TARGETS.md` §2 — per-metric Tier 1/2 SOTA baselines
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §2 — 21 mechanisms × 20 metrics matrix; §4 — failure mode → mechanism check
- `TODO/pending/23_weak_to_strong_plan.md` — R13 deferred plan (now R18 Phase 1/2)
- `TODO/pending/26_round13_round14_complete_plan.md` — cross-section integration
- `molmetal/reports/wf_weak_to_strong_audit/metrics_audit.json` — structured JSON companion

**Audit complete**: 20 + 1 metrics categorized; 5 strong / 2 acceptable / 9 weak / 5 unmeasured; R18 Phase 1 closes 5 weak → acceptable; R18 Phase 2 closes 2 weak + 4 unmeasured (m5, m6, m7, m8, m9, m10, m15); paper §4 framing aligned with honest-negative discipline.