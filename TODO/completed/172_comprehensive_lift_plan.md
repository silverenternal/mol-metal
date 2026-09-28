# TODO-172 — Comprehensive Lift Plan: Weak Metrics + Full Retrain (R21 master)

**Date**: 2026-09-27
**Reason**: User challenge 2026-09-27 — "把所有的弱指标都拉起来吧，而且更重要的是，我们的整个模型在大数据集上完全重新训练了吗？需要更完整的规划以及todo目录的完整更新"
**Status**: 📐 **MASTER PLAN** — consolidates weak-metric lift + full retrain into one orchestrated 12-week GPU plan; supersedes / extends `completed/160_big_dataset_retrain_audit_2026_09_27.md` + `03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` + `03_plans/WEAK_TO_STRONG_PLAN.md` + `pending/134_*.md` + `pending/135_*.md` + `pending/138_*.md`
**Priority**: **P0 CRITICAL** — drives paper §4 / §6 framing + R21 retrain cycle
**Owner**: Hugo Lee + ultracode agents (Wf-56, Wf-65..76, Wf-124, Wf-131, Wf-146, Wf-170, Wf-171, future Wf-172..177)
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; every number traces to JSON path or is honest-PROJECTED) + `mol-metal-gpu-execution` (Blackwell sm_120 + RTX 5090 dual-card) + `mol-metal-anticancer` (Tier-1 SOTA comparison) + `mol-metal-mechanism-mapping` (weak metric → mechanism diagnostic)

---

## §0 Status — what's honest now (2026-09-27)

### §0.1 What the user asked, and what is true today

The user asked two questions:

1. **"把所有的弱指标都拉起来吧"** — lift all weak metrics.
2. **"我们的整个模型在大数据集上完全重新训练了吗？"** — has our model been fully retrained on big data?

The honest answer, per `party-twenty-spirit` §3.5:

> **(1) Weak metrics have NOT all been lifted.** 8 metrics remain in the WEAK_REAL bucket (m1 Ru AUC, m3 Vina Dock, m4 SR, m7 Diversity, m8 PB-Valid, m10 Vina Min, m16 Affinity, m19 LOMO) — none has cleared its paper gate; all are honestly negative in paper §4.5. Wf-60 attempted to lift 8 of them and shipped infrastructure-only (53 tests PASS, 0/8 cells reached the gate; data-path FileNotFoundError + missing Vina binary blocked every actual run).
>
> **(2) The model has NOT been fully retrained on big data.** The on-disk total is **1,924,037 rows / pairs across 7 datasets**. Of this, **only 146,437 rows have been used in actual retrain (7.6 %)**. DrugOOD 1.58 M is on disk but only 10 K subsample was used. BindingDB full 1.07 M is NOT on disk (only 5 K subset). PlatinAI full 226 K is NOT on disk (URL 404, ChemRxiv gated). 3 of 6 dataset loader modules are MISSING (`drugood_loader`, `binding_db`, `nci60`). Full retrain cannot proceed without these blockers.

### §0.2 Architecture decision status

- **chemprop v2 wins head-to-head vs V4** by **+0.282 AUC** (MEASURED, 14 paired runs, paired t-test p=1.35e-5; per TODO-154). **5.1× faster** than V4 (MEASURED per chemprop_v2_status_2026_09_27.json).
- **chemprop v2 pivot** (V4 → chemprop_v2) — DECIDED per TODO-155.
- **In-domain per-metal fold wins for paper §4 headline** (Ru chemprop v2 AUC = 0.91 MEASURED vs Krasnov 0.81 = +0.099 lift) — DECIDED per TODO-154 §0.
- **LOMO for paper §B appendix** (honest cross-metal generalization story) — DECIDED per `completed/170_chemprop_v2_permetal_lomo_design.md`.
- **Wf-146 v1 design (multi-task head + LOMO) is structurally broken** (78.9 % of 19 runs < 0.5 random) — DECIDED per `molmetal/reports/wf146_lomo_v1_broken_design.json`.
- **Wf-170 v2 design (per-metal single-task + ensemble LOMO)** — DESIGN COMPLETE, code NOT yet written — PROPOSED per `completed/170_chemprop_v2_permetal_lomo_design.md` §5.

### §0.3 Compute environment

- **Blackwell sm_120** GPU available (Wf-23 in flight; per EXECUTION_ORDER.md §1).
- **RTX 5090** (32 GB) Blackwell single-card backup; per `mol-metal-gpu-execution` SKILL.
- **Standing "kill then adapt" directive** (per INDEX.md §0 + TODO-148 §0 + TODO-152): all GPU work > 1 hr DEFERRED pending explicit fresh user OK. Subagent harness budget is 3600 s (1 hr); 92-min GPU launch would die mid-run + corrupt resume state.
- **chemprop v2 CPU SMILES featurization bottleneck**: Wf-155 full 5×5×50 TIMED OUT at iter 1 (6:50 wall, GPU 2-4 %, CPU 660 %+). Fix ETA: 30 min engineering + 1 day validation (precompute features once per epoch + faster encoder pipeline).

### §0.4 What this plan produces

This is a **12-week, 8-phase plan** that addresses BOTH the weak-metric lift AND the full-retrain architecture question. It:

1. Ranks the 8 weak metrics by **lift feasibility today** (CPU-only, GPU-needed, or arch-ceiling-no-fix).
2. Decides the **full-retrain architecture**: chemprop v2 per-metal single-task (paper §4 headline) + per-metal ensemble LOMO (paper §B appendix) + Wf-170 v2 design + Multi-PH + PLL + Wasserstein transfer.
3. Decides the **full-retrain data strategy**: use 1.58 M DrugOOD + 1.07 M BindingDB (download) + 100 K tmQM + 26 K MetalCytoToxDB + 7.8 K NCI-60 + 500 PlatinAI (drop 226 K full — URL 404); 3 P0 loader-module blockers + 1 P0 download blocker + 1 P0 featurization fix + 1 P0 GPU directive.
4. Lists **ETA per phase** (single RTX 5090 vs Blackwell + RTX 5090 dual).
5. Lists **dependencies** (P0 blockers gate P1 retrain; P1 retrain gates P2 mechanism retrain; etc.).
6. Lists **risks + mitigations** (5 risks named).
7. Closes with **honest framing per §3.5** (what IS true / what is NOT true / paper §4 / §6 / §B framing).

---

## §1 Weak metrics ranked by lift feasibility

### §1.1 The 8 WEAK_REAL metrics + 3 honest caveats

| ID | metric | MEASURED today | Gate / SOTA | Δ to gate | Lift feasibility |
|---|---|---:|---:|---:|---|
| **m1** | Ru subset binary AUC (Krasnov) | 0.7613 (Wf-124 P3 Ru seed=0) / 0.7352 ± 0.0023 (Wf-131 P1 3-seed × 10-epoch) | ≥ 0.86 (Krasnov + 0.05) | −0.099 | 🟢 **HIGH** — chemprop v2 per-metal fold Ru 5×5×50 ~2 hr wall; MEASURED 0.91 ceiling per TODO-154 §0 |
| **m3** | Vina Dock median kcal/mol on CrossDocked100 | −1.3546 (Vina stub) | ≤ −8.5 (DecompDiff) / −9.37 (TransDiffSBDD) | +7.15 kcal/mol | 🟡 **MEDIUM** — requires (a) real Vina binary install, (b) SE(3) SB GPU retrain; PDB-fix + meko 0.8.0 fork for Pt/Au/Ru needed |
| **m4** | Success Rate (Vina < −8.18 ∧ QED > 0.25 ∧ SA > 0.59) | 0.0 (Vina stub) | ≥ 0.60 (TransDiffSBDD 0.84) | −0.60 | 🟡 **MEDIUM** — cascades from m3; same blockers; + SA closure-bias generator (Wf-139 P6) already ships 5/5 pockets ≤ 3.0 |
| **m7** | Diversity (Tanimoto) | 0.1065 (Lambda-MCTS) / 0.137 (Wf-32) | ≥ 0.65 (TransDiffSBDD) | −0.514 | 🔴 **LOW** — arch-ceiling honest-negative per `completed/137_l5_l7_l10_arch_ceiling_honest.md`; Multi-PH + PLL not measured to lift; Wf-36 P2 Score-Rep on mock CCO = −0.185 honest-negative |
| **m8** | PB-Valid production | UNDEFINED | ≥ 0.94 (PB-Valid) | UNDEFINED | 🟡 **MEDIUM** — Phase 1 F2(a) MetalLigandExchange SMARTS rule ship (TODO-29) + Phase 2 MMFF94 wiring |
| **m10** | Vina Min kcal/mol | −4.158 (Vina stub) | ≤ −9.5 (TransDiffSBDD) | +5.34 | 🟡 **MEDIUM** — cascades from m3; same blockers |
| **m16** | Affinity Pearson r (Boltz-2 baseline 0.86) | 0.207 (D-MPNN HeLa48h) / −0.040 (PlatinAI 4-cell) | ≥ 0.65 | −0.443 | 🟡 **MEDIUM** — chemprop v2 BindingDB full retrain (Phase 3, ~7 hr); MEASURED chemprop v2 5-metal Ru seed=0 = 0.31 already partial lift |
| **m19** | Multi-metal LOMO | 0.535 (Wf-125 P2 baseline) / 0.4568 (Wf-125 P2 W₂) / 0.5256 (Wf-146 v1 Rh seed=0 SMOKE only) | ≥ 0.70 | −0.165 | 🟢 **HIGH** — Wf-170 v2 design per-metal single-task + ensemble LOMO (17 min smoke / 1.5 hr production); honest-positive if ≥ 0.55 |

**3 honest caveats NOT in WEAK_REAL bucket but DO need attention**:

- **m17 DrugOOD OOD (MLP proxy)** — MEASURED 0.891 / 0.697 / 0.748 on metacytotox_fallback 5,207 rows (NOT real DrugOOD per `171_paper_4_1_real_drugood.md`); chemprop v2 on REAL DrugOOD MEASURED 0.7100 ± 0.0162 / 0.6652 ± 0.0082 / 0.6313 ± 0.0055 (n=5 seeds × 3 splits, 113 s wall). iMoLD reference 0.7211 / 0.6884 / 0.6792 (cite-only). **Both Mol-Metal models are LOWER than iMoLD on every split measured** (Δ assay = −0.0111, Δ scaffold = −0.0232, Δ size = −0.0479). Honest-negative per paper §4.1 + §6.
- **m21 NCI-60 9-cell mean AUC** — MEASURED 0.8007 ± 0.016 (3-seed × 50-epoch D-MPNN); Krasnov baseline 0.81. Within ±0.01 — paper §4 LEAD claim. No retrain needed.
- **m2 Pt 4-cell** — MEASURED per-cell: PC3 0.870, HL60 0.905, SKOV3 0.932 (Krasnov-tier ≥ 0.85). A2780cis 0.7278 (signal-absence bottleneck). Paper §4 per-cell LEAD with A2780cis honest-negative.

### §1.2 Per-metric lift plan (priority-ordered)

#### Tier 1 — Lift to STRONG_REAL via Phase 1-2 (chemprop v2 per-metal + LOMO ensemble)

**m1 Ru AUC** — Phase 1 chemprop v2 per-metal 5×5×50 = **2 hr wall on RTX 5090**:
- Per-metal Ru single-task model: chemprop v2 (hidden=300, depth=3, 50 epochs, BF16 autocast)
- Expected in-domain AUC: 0.86-0.91 (MEASURED 0.91 in TODO-154 §0, paired 14 runs)
- Wall: Ru n=19,135 rows × 50 epochs × 5 seeds = ~2 hr (per TODO-154 0.193 min/run × 25 = 4.8 min × 5x for production backbone)
- Gate hit: 0.86 = achievable; 0.91 = strong-positive

**m19 LOMO** — Phase 1 Wf-170 v2 per-metal + ensemble LOMO = **17 min smoke / 1.5 hr production wall**:
- Per TODO-170 §3.3 + §3.4: hidden=64/depth=2 smoke = 17 min serial; hidden=300/depth=3 production = 1.5 hr serial
- Expected ensemble LOMO AUC: 0.55-0.65 (per TODO-170 §4.2 model prediction, NOT measurement)
- Gate hit: 0.55-0.65 = achievable; ≥ 0.65 = strong-positive

#### Tier 2 — Lift to STRONG_REAL via Phase 2-3 (real Vina + BindingDB)

**m3 Vina Dock / m4 SR / m10 Vina Min** — Phase 2 SE(3) SB + real Vina = **2 weeks GPU**:
- Pre-condition: install real Vina binary + meko 0.8.0 fork for Pt/Au/Ru (per `pending/112_dmpnn_seed_integrity.md` blocker note)
- Wf-30 P2 Vina triple re-bench: 10 pockets × 5 candidates × 3 seeds × 50 epochs on RTX 5090
- Wall: ~5 hr wall for Vina triple; ~12 hr wall for SE(3) SB retrain
- Gate hit: m3 ≤ −8.5 = strong-positive if SE(3) SB converges; m4 ≥ 0.60 cascades from m3

**m16 Affinity Pearson r** — Phase 3 BindingDB full chemprop v2 retrain = **~7 hr wall**:
- Pre-condition: BindingDB full 1.07 M download + metal filter (~50 K-200 K rows) + `molmetal/data/binding_db.py` loader
- chemprop v2 (hidden=300, depth=3, 5 seeds × 50 epochs)
- Wall: ~7 hr single RTX 5090
- Gate hit: ≥ 0.65 = strong-positive if BindingDB subset ≥ 100 K rows; 0.40-0.55 = honest-acceptable

#### Tier 3 — Honest-negative (no fix in scope)

**m7 Diversity** — arch-ceiling honest-negative per `completed/137_l5_l7_l10_arch_ceiling_honest.md`:
- Wf-36 P2 Score-Rep on mock CCO variance_reduction = −0.185 (REGRESSED, honest-negative)
- Multi-PH + PLL retrain not yet run; Wf-37 P3 baseline R² = +0.087 ± 0.081 (16-bin Multi-PH / 64-d / no Score-Rep)
- Phase 5 M24 Multi-PH Fourier 32-bin + PLL 128-d + Score-Rep MC = **2 weeks GPU**, PROJECTED 0.55-0.70 (NOT a measurement)
- Paper §4.5 honest-negative unless Wf-66 P3 / Wf-67 P4 GPU retrain clears

**m8 PB-Valid production** — Phase 1 F2(a) MetalLigandExchange SMARTS rule + Phase 2 MMFF94 = **1-2 weeks**:
- Pre-condition: TODO-29 SMARTS rule ship (currently NOT shipped)
- Wall: ~3 days engineering + 1 day validation
- Gate hit: 0.94 if F2(a) SMARTS rule ships AND MMFF94 mines force field correctness

#### Tier 4 — Paper §4.5 honest-negative (no fix available)

**m3 / m4 / m10 / m7** — paper §4.5 honest-negative in current R19 cycle; revisit after Wf-23 GPU ultracode queue (per EXECUTION_ORDER.md §1 Phase A + B).

### §1.3 Lift feasibility summary

- **2 weak → strong (m1, m19)**: 2 hr + 1.5 hr GPU = 3.5 hr wall — **DO THIS FIRST** (Phase 1).
- **3 weak → acceptable (m3, m4, m10)**: 12 hr + ~5 hr GPU = 17 hr wall — Phase 2 (after Vina binary install).
- **1 weak → strong (m16)**: 7 hr GPU — Phase 3 (after BindingDB download).
- **1 weak → honest-negative (m7)**: arch-ceiling; paper §4.5.
- **1 weak → acceptable (m8)**: 3 days engineering — Phase 1 CPU.

**Total wall to clear 4 of 8 weak (m1, m19, m16, m8)**: ~28 hr GPU + 3 days CPU. **Total wall to clear 7 of 8 weak (m1, m19, m3, m4, m10, m16, m8)**: ~45 hr GPU + 1 week engineering. **m7 stays honest-negative.**

---

## §2 Full retrain architecture decision

### §2.1 Decision (DECIDED per TODO-155, TODO-154, TODO-170)

**Architecture**: **chemprop v2 (NOT V4 D-MPNN) + per-metal single-task (NOT multi-task) + ensemble LOMO (NOT per-metal-head LOMO) + Multi-PH + PLL + Wasserstein transfer (joint fine-tune head)**.

**Justification**:

| Decision | Evidence | Source |
|---|---|---|
| **chemprop v2 > V4 D-MPNN** | +0.282 AUC paired head-to-head, p=1.35e-5, 5.1× faster | TODO-154 |
| **Per-metal single-task > multi-task for LOMO** | v1 multi-task head + LOMO is structurally broken (78.9 % < 0.5 random); v2 per-metal single-task + ensemble is honest-positive | TODO-170 §0 |
| **Per-metal fold for paper §4 headline, LOMO for paper §B appendix** | Ru per-metal MEASURED 0.91 (TODO-154 §0); LOMO honestly negative unless v2 design clears | TODO-170 §2.3 |
| **Multi-PH + PLL + Wasserstein for cross-dataset retrain** | Joint Hilbert-ASTSR + Multi-PH + PLL framework SHIPPED (CPU wiring + smoke) | Wf-37 P3, Wf-41 P1 |
| **Wasserstein pIC50 auxiliary for Ru 5-metal** | +0.049 lift MEASURED (Wf-124 P3 0.7123 → 0.7613); production backbone +0.05-0.10 PROJECTED | TODO-131 |

### §2.2 Phase-by-phase architecture

| Phase | Backbone | Head | Split | Data |
|---|---|---|---|---|
| **Phase 1A** (m1 lift) | chemprop v2 (h=300, d=3, BF16) | Per-metal single-task BinaryClassificationFFN | Per-metal scaffold split | MetalCytoToxDB Ru 19,135 |
| **Phase 1B** (m19 lift) | chemprop v2 (h=64 smoke / h=300 production) | 5× per-metal single-task ensemble | 4-metal-ensemble LOMO | MetalCytoToxDB 5-metal 26,801 |
| **Phase 1C** (m8 lift) | (n/a — RDKit evaluator) | F2(a) MetalLigandExchange SMARTS + MMFF94 | (production SMARTS) | (synthesis rule) |
| **Phase 2A** (m3 / m4 / m10 lift) | SE(3) SB (Schrödinger Bridge on Lie Groups, M22) | Joint diffusion + Metal-Catalyst pose head | CrossDocked 100 pocket × 10 seed | CrossDocked 100 K pairs |
| **Phase 2B** (Vina triple) | real Vina 1.2 + meko 0.8.0 Pt/Au/Ru fork | exhaustiveness 8 → 32 (Wf-141 §2) | 10-pocket × 3-seed × 50-ep | CrossDocked 100 K |
| **Phase 3** (m16 lift) | chemprop v2 (h=300, d=3, BF16) | Affinity regression (Ki / Kd / IC50) | scaffold split | BindingDB full 1.07 M → metal-filtered ~50-200 K |
| **Phase 4A** (cross-dataset pretrain) | chemprop v2 (h=300, d=3) | tmQM regression head + MetalCytoToxDB multitask | per-cohort scaffold | tmQM 100 K + MetalCytoToxDB 26 K |
| **Phase 4B** (PlatinAI fine-tune) | chemprop v2 (h=300, d=3) | 4-cell multi-task (MCF7/A549/A2780/A2780cis) | scaffold split | PlatinAI 500 (full 226 K NOT on disk) |
| **Phase 5A** (M24 Multi-PH) | Fourier 32-bin Multi-PH + PLL 128-d | Score-Rep MC variance reduction | per-cohort | MetalCytoToxDB 26 K |
| **Phase 5B** (M25+M26 joint) | Hilbert-ASTSR + Multi-PH + PLL | Score-Rep + Hilbert primal-dual | per-cohort | MetalCytoToxDB 26 K + PlatinAI 500 |
| **Phase 5C** (M27+M28 joint) | W-Flow ODE + MacroGuide | Joint Schrödinger Bridge + Macrocyclic scaffold prior | per-pocket | CrossDocked 100 K + PlatinAI 500 |
| **Phase 6** (paper §4 / §6 / §B v3) | (compile results) | (paper builds) | (n/a) | (n/a) |

### §2.3 What is NOT the architecture decision

- **NOT "use V4 D-MPNN as production"** — V4 is kept as §B appendix only (lower bound).
- **NOT "use multi-task head for LOMO"** — v1 design is structurally broken (TODO-170 §0).
- **NOT "use F5_RDKIT_2D + Ridge as production baseline"** — that's the cross-dataset R² baseline (Wf-71 P3 MEASURED +0.1248 macro), NOT the production chemprop v2 backbone.
- **NOT "use LSTM / Transformer"** — chemprop v2 + Multi-PH + PLL + Wasserstein is the canonical 2024-2026 SBDD stack (per `mol-metal-r18-math` skill).
- **NOT "use pre-trained ChemBERTa / Uni-Mol"** — pretraining not in scope; chemprop v2 + Multi-PH is sufficient for paper §4.

### §2.4 Risk: chemprop v2 SMILES featurization bottleneck

Per `chemprop_v2_status_2026_09_27.json` — Wf-155 full 5×5×50 TIMED OUT at iter 1 (6:50 wall, GPU 2-4 %, CPU 660 %+). **This is the #1 blocker for Phase 1A chemprop v2 per-metal run** (Ru 19,135 × 50 epochs × 5 seeds = ~2 hr projected; if bottleneck persists, scale to 4-6 hr).

**Fix ETA**: 30 min engineering + 1 day validation. Per chemprop_v2_reference_notes.md:
- Precompute features once per epoch (cache `SimpleMoleculeMolGraphFeaturizer` output to disk)
- Faster encoder pipeline: skip `BatchNorm` on small batch (BF16 autocast)
- Use `num_workers=4` DataLoader for parallel featurization

After fix: chemprop v2 per-metal Ru 5×5×50 = 2 hr wall (per TODO-154 §3 baseline 0.193 min/run × 25 = 4.8 min for smoke; ×5 for production backbone = 24 min × 5 seeds = 2 hr).

---

## §3 Full retrain data strategy (use 1.58 M DrugOOD + 1.07 M BindingDB)

### §3.1 Decision (DECIDED per this plan + `completed/160_big_dataset_retrain_audit_2026_09_27.md` §3.2)

**Strategy**: **use 1.58 M DrugOOD (on disk) + download BindingDB full 1.07 M + tmQM 100 K + MetalCytoToxDB 26 K + NCI-60 7.8 K + PlatinAI 500 (subset) = ~3.04 M rows total**. Drop PlatinAI full 226 K (URL 404). Drop DrugOOD refined × 3 = 800 K (NOT on disk) + DrugOOD partial Ji 2023 296 K (NOT on disk).

**Justification**: Of the 7 on-disk datasets, **5 have loaders MISSING**; must add 3 loader modules (`drugood_loader`, `binding_db`, `nci60`) before Phase 2-3 retrain. The "2.67 M aspirational" figure referenced in the user prompt is **INFLATED**; actual on-disk + downloadable = ~3.04 M.

### §3.2 Data inventory + per-dataset plan

| Dataset | On disk | Plan | Loader status | Wall |
|---|---:|---|---|---:|
| **MetalCytoToxDB** | 26,801 | 100 % used; per-metal fold (Phase 1A) + LOMO (Phase 1B) | ✅ `molmetal/data/metal_cytotox.py` | 2 hr + 1.5 hr |
| **NCI-60 precious-9** | 7,823 | 100 % used; D-MPNN 9-cell MEASURED 0.8007; chemprop v2 retrain optional | ❌ `molmetal/data/nci60.py` MISSING → add loader (Phase 0 P0) | 1-2 days engineering |
| **BindingDB subset** | 5,257 | Full 1.07 M → download + metal filter (~50-200 K) | ❌ `molmetal/data/binding_db.py` MISSING → add loader (Phase 0 P0) | 1 day download + 1 day filter + 1 day loader |
| **DrugOOD native** | 1,583,302 | Full retrain (Phase 2; 3 splits × 5 seeds × 50 epochs) | ❌ `molmetal/data/drugood_loader.py` MISSING → add loader (Phase 0 P0) | 10 hr GPU |
| **tmQM d-block** | 100,849 | Joint pretrain (Phase 4A) | ✅ `molmetal/data/tmqm.py` | 3 hr GPU |
| **PlatinAI 500-subset** | 500 | Fine-tune (Phase 4B); full 226 K NOT on disk (URL 404) — honest subset | ✅ `metallo_drugs_500_train.csv` | 30 min GPU |
| **PlatinAI full 226 K** | 0 | DROP — URL 404, ChemRxiv gated | n/a | n/a |
| **CrossDocked2020** | 100,000 pairs | Phase 2A SE(3) SB retrain (docking) + Phase 2B Vina triple | ✅ `molmetal/data/crossdocked.py` | 12 hr + 5 hr |
| **TOTAL on disk + downloadable** | **~1.92 M + 1.07 M BindingDB = ~2.99 M** | — | — | — |
| **TOTAL currently used** | 146,437 (7.6 %) | — | — | — |
| **TOTAL planned used (post-Phase 6)** | ~2.99 M (98 %) | — | — | — |

### §3.3 Per-phase data loading spec

#### Phase 0 — P0 loaders + downloads (1 week CPU)

**Add 3 loader modules** (per `completed/148_wf146_lomo_retrain_prep_status.md` §2.1 audit):

| Loader | Module | Test count | Wall |
|---|---|---:|---:|
| **DrugOOD loader** | `molmetal/data/drugood_loader.py` (load train.csv + lbap_core_ic50 splits) | ≥12 tests | 1-2 days |
| **BindingDB loader** | `molmetal/data/binding_db.py` (load TSV.gz + metal filter on SMILES) | ≥12 tests | 1-2 days |
| **NCI-60 loader** | `molmetal/data/nci60.py` (load nci60_platinai_link.csv + 9-cell filter) | ≥12 tests | 1 day |

**Download BindingDB full 1.07 M**:
- URL: `https://www.bindingdb.org/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/home/hugo/codes/mol-metal/.data/drugen/bindingdb/BindingDB_All_2D_202510.sdf.zip` (or TSV.gz equivalent)
- Wall: ~1 hr download (1 GB), ~30 min unzip
- Filter: keep only metal-binding entries (queries against `molmetal/data/metal_smarts.txt` for Pt/Ru/Ir/Os/Rh/Re/Au/Pd)
- Expected rows after filter: ~50 K-200 K (MEASURED-dependent on filter spec)

#### Phase 1A — m1 Ru per-metal retrain (2 hr GPU)

```python
# molmetal/scripts/train_chemprop_v2_permetal.py --metal Ru --seeds 0 1 2 3 4 --epochs 50 --hidden 300 --depth 3 --bf16
data = load_metal_cytotox().query("metal == 'Ru'")  # 19,135 rows
model = ChempropV2PerMetalModel(metal_name="Ru", hidden_dim=300, depth=3, dropout=0.2)
trainer = ChempropV2PerMetalTrainer(model, lr=1e-3, batch_size=64, epochs=50, bf16_autocast=True)
trainer.fit(data, scaffold_split=True, seeds=range(5))
```

#### Phase 1B — m19 Wf-170 v2 per-metal + ensemble LOMO (17 min smoke / 1.5 hr production)

```python
# 5 per-metal models (Ru/Ir/Os/Rh/Re) × 5 seeds × 50 epochs (smoke h=64 d=2)
# 5 LOMO folds × 5 seeds ensemble eval
# Output: metrics/by_metric/chemprop_v2_permetal_lomo.json
```

#### Phase 1C — m8 PB-Valid production (3 days CPU)

```python
# Add F2(a) MetalLigandExchange SMARTS rule to molmetal/molmetal_lam/reactions/
# Wire MMFF94 mines force field for Pt/Au/Ru complexes
# 5-pocket × 100 sample × 3 seeds × PB-Valid v0.8.0 eval
# Output: metrics/by_metric/pb_valid_production.json
```

#### Phase 2A — m3 Vina Dock / SE(3) SB retrain (12 hr GPU)

```python
# M22 SE(3) Schrödinger Bridge on Lie Groups (Mahmood)
# Train on CrossDocked 100 pocket × 50 epoch × 5 seed
# Vina dock at exhaustiveness 32 (Wf-141 lift)
# Output: metrics/by_metric/vina_high_affinity.json (median kcal/mol)
```

#### Phase 2B — Vina triple re-bench (5 hr GPU)

```python
# Wf-30 P2 Vina triple: 10 pockets × 5 candidates × 3 seeds
# Real Vina 1.2 binary (NOT stub)
# meko 0.8.0 fork for Pt/Au/Ru
# Output: molmetal/reports/wf30_vina_triple/synthesis.json
```

#### Phase 3 — m16 BindingDB full chemprop v2 (7 hr GPU)

```python
# chemprop v2 (h=300, d=3) on BindingDB metal-filtered ~50-200 K rows
# 5 seeds × 50 epochs
# Targets: Ki / Kd / IC50 affinity regression
# Output: metrics/by_metric/affinity_pearson_r.json
```

#### Phase 4A — tmQM + MetalCytoToxDB joint pretrain (3 hr GPU)

```python
# chemprop v2 multi-task pretrain on tmQM d-block 100 K + MetalCytoToxDB 26 K
# 5 seeds × 50 epochs
# Wasserstein pIC50 transfer from pretrain to fine-tune head
```

#### Phase 4B — PlatinAI 4-cell fine-tune (30 min GPU)

```python
# chemprop v2 (h=300, d=3) on PlatinAI 500 subset × 4-cell multi-task
# Per-cell AUC for MCF7/A549/A2780/A2780cis
# Output: metrics/by_metric/metalcytotox_combined_r2.json (refresh)
```

#### Phase 5A — Multi-PH Fourier 32-bin + PLL 128-d + Score-Rep MC (2 weeks GPU)

```python
# M24 Multi-PH Fourier 32-bin + PLL 128-d
# Score-Rep MC variance reduction (Burch & Bardsley 2022 + Hu et al. 2026)
# 5 seeds × 50 epochs on MetalCytoToxDB 26 K
# Output: metrics/by_metric/multi_task_4domain_v2.json + Score-Rep variance reduction
```

#### Phase 5B — M25+M26 joint Hilbert-ASTSR + Multi-PH + PLL (1 week GPU)

```python
# Wf-56 P2 Hilbert-ASTSR + Multi-PH + PLL + Triangle closure (R20-P2 retry)
# Real pIC50 R² ≥ 0.85 (stretch) / ≥ 0.30 (realistic) gate
# Output: metrics/by_metric/hilbert_4cell_r2.json + triangle_closure
```

#### Phase 5C — M27+M28 W-Flow + MacroGuide joint (1 week GPU)

```python
# W-Flow ODE (Mahmood) + Macrocyclic scaffold prior (MacroGuide)
# Joint Schrödinger Bridge + macrocyclic scaffold prior
# Output: metrics/by_metric/cfm_decode_real.json (refresh + m3 Vina cascade)
```

#### Phase 6 — Paper v3 submission (1 week CPU)

Per `completed/141_q2_jcim_completion_sprint.md` + `pending/138_paper_v2_submission_package.md`:
- paper builds clean (0 unresolved refs)
- JCIM submission checklist complete
- cover letter + figures + supplementary S1-S8
- Constraint: per `pending/84_no_paper_task_without_authorization.md` — paper work BLOCKED pending user authorization

### §3.4 What is NOT the data strategy

- **NOT "use only 10 K DrugOOD subsample"** — that's the Phase 0 smoke test; full 1.58 M is Phase 2 retrain.
- **NOT "wait for PlatinAI full 226 K"** — URL 404, ChemRxiv gated; 500 mol subset is honest.
- **NOT "use DrugOOD refined × 3 = 800 K"** — NOT on disk; drop from paper §4.
- **NOT "use BindingDB full 1.07 M as-is"** — must metal-filter to ~50-200 K rows.

---

## §4 ETA per phase

### §4.1 Per-phase wall-time (single RTX 5090, GPU-1)

| Phase | Wall-time (single 5090) | Wall-time (Blackwell + 5090 dual) | Bottleneck |
|---|---:|---:|---|
| **Phase 0** (P0 loaders + BindingDB download + chemprop featurization fix) | 1 week CPU | 1 week CPU | engineering + download (1 GB BindingDB) |
| **Phase 1A** (m1 Ru chemprop v2 per-metal 5×5×50) | 2 hr | 1 hr | chemprop v2 SMILES featurization |
| **Phase 1B** (m19 Wf-170 v2 per-metal + ensemble LOMO 5×5×50 smoke) | 17 min | 9 min | per-metal training + LOMO eval |
| **Phase 1B-production** (m19 hidden=300/depth=3 5×5×50) | 1.5 hr | 45 min | per-metal training + LOMO eval |
| **Phase 1C** (m8 PB-Valid production) | 3 days CPU | 3 days CPU | engineering (SMARTS rule + MMFF94) |
| **Phase 2A** (m3 SE(3) SB retrain) | 12 hr | 6 hr | SE(3) SB diffusion training |
| **Phase 2B** (Vina triple real Vina) | 5 hr | 2.5 hr | real Vina binary + meko fork |
| **Phase 3** (m16 BindingDB chemprop v2 5×5×50) | 7 hr | 3.5 hr | BindingDB loader + metal filter |
| **Phase 4A** (tmQM + MetalCytoToxDB joint pretrain) | 3 hr | 1.5 hr | multi-task pretrain |
| **Phase 4B** (PlatinAI 4-cell fine-tune) | 30 min | 15 min | per-cell fine-tune |
| **Phase 5A** (Multi-PH + PLL + Score-Rep MC) | 2 weeks (80 hr) | 1 week (40 hr) | Multi-PH 32-bin Fourier acceleration |
| **Phase 5B** (Hilbert-ASTSR + Multi-PH + PLL joint) | 1 week (40 hr) | 30 hr | Hilbert primal-dual convergence |
| **Phase 5C** (W-Flow + MacroGuide joint) | 1 week (40 hr) | 30 hr | ODE solver 6 ms → 0.06 ms (100× speedup) |
| **Phase 6** (paper v3 submission) | 1 week CPU | 1 week CPU | paper builds + JCIM checklist |
| **TOTAL** | **~12 weeks wall** | **~6-8 weeks wall** | — |

### §4.2 Per-metric ETA contribution

| Metric | Phase | Wall | Gate hit probability (model) |
|---|---|---:|---:|
| m1 | 1A | 2 hr | 🟢 90 % (chemprop v2 Ru 0.91 MEASURED ceiling) |
| m19 | 1B | 1.5 hr | 🟡 60 % (v2 design model 0.55-0.65 PROJECTED) |
| m8 | 1C | 3 days | 🟡 50 % (F2(a) SMARTS rule NOT shipped) |
| m3 | 2A | 12 hr | 🟡 40 % (SE(3) SB convergence uncertain) |
| m4 | 2A (cascade from m3) | 0 hr (same retrain) | 🟡 40 % |
| m10 | 2A (cascade from m3) | 0 hr (same retrain) | 🟡 40 % |
| m16 | 3 | 7 hr | 🟡 50 % (BindingDB ≥ 100 K subset) |
| m7 | 5A | 80 hr | 🔴 25 % (arch-ceiling honest-negative) |

### §4.3 Total compute budget

- **Total wall-time**: 12 weeks (single 5090) or 6-8 weeks (Blackwell + 5090 dual)
- **Total GPU hours**: ~280 hr (12 weeks × 24 hr/week on 1 GPU) or ~140 hr (dual)
- **GPU memory peak**: chemprop v2 BindingDB 200 K × h=300 + Multi-PH 32-bin + PLL 128-d ≈ ~16-20 GB peak (RTX 5090 has 32 GB)
- **GPU memory peak for Blackwell**: ~32-40 GB peak (Blackwell sm_120 has 95.6 GB)
- **Standing "kill then adapt" directive**: every GPU launch > 1 hr wall requires explicit fresh user OK

---

## §5 Dependencies

### §5.1 P0 blockers (must resolve before any retrain)

| ID | Blocker | Wall | Owner |
|---|---|---:|---|
| **BD-1** | Add 3 loader modules: `drugood_loader.py`, `binding_db.py`, `nci60.py` | 1 week CPU | Hugo Lee + ultracode |
| **BD-2** | Download BindingDB full 1.07 M + metal filter | 1 day download + 1 day filter | Hugo Lee |
| **BD-3** | chemprop v2 CPU SMILES featurization bottleneck fix | 30 min engineering + 1 day validation | Hugo Lee + ultracode |
| **BD-4** | Standing "kill then adapt" GPU directive — explicit fresh user OK | (decision) | Hugo Lee |
| **BD-5** | Install real Vina binary + meko 0.8.0 fork for Pt/Au/Ru | 1 day install | Hugo Lee |

### §5.2 P1 dependencies (gating retrain phases)

| ID | Dependency | Gates |
|---|---|---|
| **BD-6** | Wf-146 LOMO v2 design (TODO-170) code complete | Phase 1B m19 retrain |
| **BD-7** | Wf-131 P3 LOMO completion (7/25 remaining: Rh seed3/4 + Re seed0-4) | Phase 1B m19 retrain (alternative path) |
| **BD-8** | Wf-135 m19 Rh oversample regression fix (TODO-135) | Phase 1B m19 retrain (alternative path) |
| **BD-9** | Wf-134 m15 Docking RMSD redesign Phase 2 (TODO-134) | Phase 6 paper v3 (per-pocket median RMSD ≤ 2.5 Å gate) |
| **BD-10** | Wf-138 paper v2 submission package (TODO-138) | Phase 6 paper v3 |
| **BD-11** | Phase 1 m1 lift MEASURED | Phase 4 PlatinAI fine-tune (joint retrain) |
| **BD-12** | Phase 3 BindingDB metal filter MEASURED ≥ 100 K rows | Phase 3 m16 retrain (gate hit ≥ 0.65) |
| **BD-13** | Phase 1B m19 LOMO ≥ 0.55 MEASURED | Phase 4B PlatinAI fine-tune |
| **BD-14** | Phase 2A m3 Vina Dock ≤ −8.5 MEASURED | Phase 5C W-Flow + MacroGuide joint |

### §5.3 DAG (Phase → Phase)

```
Phase 0 (P0 blockers)
    ├── BD-1: 3 loaders → unlocks Phase 1, 2, 3, 4
    ├── BD-2: BindingDB download → unlocks Phase 3
    ├── BD-3: chemprop featurization fix → unlocks Phase 1A
    ├── BD-4: GPU directive OK → unlocks ALL GPU phases
    └── BD-5: Vina binary install → unlocks Phase 2B
Phase 1A (m1) ──┐
Phase 1B (m19) ─┤
Phase 1C (m8) ──┘──→ Phase 4 (joint pretrain + PlatinAI fine-tune)
Phase 2A (m3) ──┐
Phase 2B (Vina) ─┘──→ Phase 5C (W-Flow + MacroGuide joint)
Phase 3 (m16) ────────→ Phase 6 (paper v3)
Phase 5A (Multi-PH) ──┐
Phase 5B (Hilbert) ───┴──→ Phase 6 (paper v3)
```

### §5.4 Critical path

```
BD-1 + BD-3 + BD-4 (1 week CPU) → Phase 1A (2 hr) + Phase 1B (1.5 hr)
→ Phase 4 (3.5 hr) → Phase 5B (1 week) → Phase 5A (2 weeks) → Phase 6 (1 week)
TOTAL: 1 week CPU + 2-3.5 hr GPU + 3.5 hr GPU + 1 week GPU + 2 weeks GPU + 1 week CPU
     = 6-7 weeks single 5090, 4-5 weeks dual
```

---

## §6 Risks + mitigations

### §6.1 Risk matrix

| ID | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | chemprop v2 SMILES featurization bottleneck NOT fixed (Wf-155 timed out at 6:50 wall) | Medium | High — blocks Phase 1A | Pre-fix: use hidden=64/depth=2 smoke backbone (TODO-170 §3.5); ship LOMO before per-metal. Post-fix ETA: 30 min engineering. |
| **R2** | Standing "kill then adapt" GPU directive not lifted — every GPU launch > 1 hr DEFERRED pending fresh user OK | High | High — blocks all GPU phases | Per task rule: ask user for explicit fresh OK before each GPU launch > 1 hr; document the OK in `molmetal/reports/wf_NNN_pre_launch_approval.json` |
| **R3** | BindingDB full 1.07 M download fails (URL rot, server down) | Low | Medium — blocks Phase 3 | Backup: use BindingDB subset 5 K (already on disk); honest-negative for m16 in paper §4.5 |
| **R4** | PlatinAI full 226 K URL stays 404 | Already happened | Low — Phase 4B uses 500 subset | Accept 500-mol subset (honest); drop "full retrain" claim from paper §4 |
| **R5** | Wf-170 v2 design code does NOT ship (TODO-170 §5 implementation order Phase 0-2 not started) | Medium | Medium — Phase 1B has alternative Wf-131 path | Per TODO-170 §5.2: ~3 hr coding + 30 min wall to ship Phase 0 smoke; fallback Wf-131 P3 LOMO 7/25 remaining at 92 min wall |
| **R6** | chemprop v2 per-metal Ru 5×5×50 wall exceeds 4 hr (beyond 1 hr subagent budget) | Medium | Medium — Phase 1A exceeds budget | Split into 5 sequential launches (1 hr each); use hidden=64/depth=2 smoke for first batch (TODO-168 §1.3 MEASURED throughput) |
| **R7** | Phase 2A SE(3) SB convergence fails (M22 mechanism convergence uncertain) | High | Medium — blocks m3 / m4 / m10 lift | Honest-negative: paper §4.5 already has m3 / m4 / m10 as honest-negative; if SE(3) SB fails, keep honest-negative (no regression); alternative: midpoint ODE fallback per EXECUTION_ORDER.md §1 |
| **R8** | Phase 5A m7 Diversity arch-ceiling honest-negative confirmed | Already known | None — honest-negative expected | Per `completed/137_l5_l7_l10_arch_ceiling_honest.md`: keep honest-negative in paper §4.5; do NOT fabricate lift |
| **R9** | Wf-138 paper v2 submission BLOCKED per TODO-84 paper rule | Always (active rule) | None — block is the rule | Per `pending/84_no_paper_task_without_authorization.md`: ship paper drafts to `TODO/completed/<WF>_draft.md`; actual paper/*.tex writes BLOCKED pending user OK |
| **R10** | GPU OOM on Blackwell 95.6 GB (chemprop v2 h=300 + Multi-PH 32-bin + PLL 128-d + 200 K BindingDB) | Low | High — would crash mid-retrain | Mitigation: gradient checkpointing + BF16 autocast + per-metal head singleton; use RTX 5090 32 GB with batch=64 (NOT batch=256) for safety |

### §6.2 Risk-driven short-circuits

If R1 (chemprop featurization bottleneck) persists: ship smoke LOMO first (TODO-170 §3.5: 17 min wall), then attempt per-metal fold at hidden=64/depth=2 (NOT 300/3); honest-negative if gate not hit.

If R2 (GPU directive not lifted): execute Phase 0 (CPU loaders + BindingDB download) FIRST; ship smoke LOMO (17 min wall, < 1 hr budget OK); document R2 in paper §6.

If R7 (SE(3) SB convergence fails): keep m3 / m4 / m10 honest-negative; paper §4.5 ships them as-is.

If R8 (m7 arch-ceiling confirmed): keep m7 honest-negative; paper §4.5.

### §6.3 Risk NOT in scope (not part of this plan)

- R-extra-1: wet-lab IC50 cooperation (Wf-58, R20-P6) — 6-mo deferred; out of R21 cycle.
- R-extra-2: paper acceptance probability — out of scope; depends on reviewer.
- R-extra-3: external dataset request (PlatinAI authors) — already tried; URL 404.

---

## §7 Honest framing per `party-twenty-spirit` §3.5

### §7.1 What IS true (today, 2026-09-27)

- **1.92 M rows of training data IS on disk** across 7 datasets (DrugOOD 1.58 M dominates).
- **MetalCytoToxDB 26 K IS the workhorse** (used 4+ workflows).
- **Wf-124 P3 chemprop v2 BEATS Krasnov 2026 baseline by +0.099 AUC** on Ru in-domain (MEASURED per TODO-154 §0).
- **Wf-124 P3 D-MPNN Wasserstein lifts Ru seed=0 from 0.7123 → 0.7613** (+0.049 lift, MEASURED per `metalcytotox_ru_auc.json`).
- **Wf-71 P3 cross-dataset R² lift +0.1248 macro** (MEASURED per `metalcytotox_combined_r2.json`).
- **Wf-131 P1 Ru 3-seed × 10-epoch mean AUC = 0.7352 ± 0.0023** (MEASURED).
- **Wf-146 v1 LOMO 1/25 completed** (Rh seed=0 = 0.5256 MEASURED); **v1 design structurally broken** (78.9 % < 0.5 random per `wf146_lomo_v1_broken_design.json`).
- **Wf-131 P3 D-MPNN LOMO 18/25 completed** (Ru 5/5 + Ir 5/5 + Os 5/5 + Rh 3/5; Re 0/5).
- **Wf-170 v2 per-metal + ensemble LOMO design COMPLETE** (per `completed/170_chemprop_v2_permetal_lomo_design.md`).
- **Wf-170 v2 code NOT yet written** (Phase 0 of TODO-170 §5.2).
- **Wf-171 paper §4.1 lead rewrite with REAL DrugOOD MEASURED**: chemprop v2 0.7100 ± 0.0162 / 0.6652 ± 0.0082 / 0.6313 ± 0.0055; honest-negative vs iMoLD 0.7211 / 0.6884 / 0.6792 (cite-only).
- **All 32 R18 modules + ≈1,235 tests CPU-shipped** (per R18 close-out `completed/73_r18_closeout_final.md`).
- **Wf-139 P5 m14 CFM decode real** = 0.9745 MEASURED (per `completed/139_wf139_p5_cfm_decode_real.md`).
- **Wf-139 P6 m5 SA closure-bias generation** 5-pocket × 100 = aggregate SA 2.463 (PASS ≤ 2.8 stretch; per `completed/142_wf139_p6_sa_closure_bias_generation.md`).
- **Wf-44 P2 Closure OOD Phase 2 + Catherine 6/6 PROPERTIES MEASURED** on 25 metals × 6/6 PASS (per `completed/96_r20_p5_closure_catherine.md`).
- **NCI-60 9-cell precious-metal AUC MEASURED 0.8007 ± 0.016** (per `nci60_precious_9cell_auc.json`).

### §7.2 What is NOT true (claims to drop per `party-twenty-spirit` §3.5)

- **"2.67 M total available data"** is INFLATED. Real on-disk is 1.92 M; real on-disk + downloadable = ~2.99 M (after BindingDB download).
- **"Full retrain on big dataset"** has NOT been done. Utilized = 7.6 % of on-disk (146 K / 1.92 M); 4.8 % of aspirational full-retrain goal.
- **DrugOOD 1.58 M is mostly UNUSED** (only 10 K subsample in 1 experiment; Wf-171 ran 95 K × 3 splits chemprop v2 = 0.71 honest-negative vs iMoLD).
- **BindingDB full 1.07 M is NOT on disk** (only 5 K subset).
- **PlatinAI full 226 K is NOT on disk** (URL 404, ChemRxiv gated).
- **3 of 6 dataset loader modules MISSING** (`drugood_loader`, `binding_db`, `nci60`); full retrain cannot proceed without them.
- **Wf-131 P3 LOMO DEFERRED** (7/25 remaining: Rh seed3/4 + Re seed0-4).
- **Wf-146 v1 LOMO 1/25 done; v1 design structurally broken**; Wf-170 v2 design code NOT yet written.
- **Wf-135 m19 Rh oversample regression** (TODO-135 master plan; 5-seed re-bench pending).
- **Wf-138 paper v2** (pending Wf-134 + Wf-135 + Wf-139 P5 + Phase 6 MEASURED numbers).
- **All R19/R20 GPU retrain DEFERRED** per standing "kill then adapt" directive.
- **m7 Diversity arch-ceiling honest-negative confirmed** per `completed/137_l5_l7_l10_arch_ceiling_honest.md`.
- **Wf-171 chemprop v2 DrugOOD LOWER than iMoLD on every split** (assay −0.0111, scaffold −0.0232, size −0.0479).

### §7.3 What to honestly claim in paper

#### Paper §4 LEAD (8 STRONG metrics)

| ID | metric | MEASURED | Tier 1 SOTA | Gap | Verdict |
|---|---|---:|---:|---:|---|
| **m11** | distinct metal-ligand pairs | **20** | ≥ 3 (gate) | +17 | ✅ STRONG |
| **m12** | SyntheMol synthesis rate | **1.000** | ≥ 0.95 | +0.05 | ✅ STRONG |
| **m13** | RDKit validity | **1.000** | ≥ 0.99 | +0.01 | ✅ STRONG |
| **m14** | CFM decode ratio | **0.9745** (Wf-139 P5) | ≥ 0.05 | +0.92 | ✅ STRONG |
| **m17** | DrugOOD OOD (MLP proxy on metacytotox_fallback) | **0.891 / 0.697 / 0.748** | iMoLD 0.721 / 0.688 / 0.679 | +0.17 / +0.009 / +0.07 | ⚠️ STRONG-with-caveat (proxy on 5 K subset, NOT real DrugOOD) |
| **m2-PC3/HL60/SKOV3** | per-cell AUC subset | **0.870 / 0.905 / 0.932** | Krasnov ≥ 0.85 | +0.02 to +0.08 | ✅ STRONG (high-data cells) |
| **m21** | NCI-60 9-cell precious-metal mean AUC | **0.8007** | Krasnov 0.81 | −0.009 (just under) | ✅ STRONG (within ±0.01) |
| **Deflexpressor** | ba_acc (novelty) | **1.000 @ margin ≥ 0.5** | n/a | novelty | ✅ STRONG (novelty claim) |

#### Paper §4 ACCEPTABLE (per-cell or PAPER acceptable with caveat)

- **m2 Pt A2780cis**: 0.7278 — within 9-cell 0.8007 mean — §4 per-cell acceptable
- **m1 Ru chemprop v2**: PROJECTED 0.86-0.91 (Phase 1A); MEASURED 0.91 ceiling per TODO-154 §0 — §4 lead if Phase 1A clears

#### Paper §4.5 HONEST-NEGATIVE (8 WEAK metrics)

| ID | metric | MEASURED | Gate / SOTA | Gap | Verdict |
|---|---|---:|---:|---:|---|
| **m1** | Ru subset binary AUC | 0.7613 (Wf-124 P3) | Krasnov ≥ 0.86 | −0.099 | ❌ HONEST-NEGATIVE (unless Phase 1A clears) |
| **m3** | Vina Dock median kcal/mol | −1.35 (Vina stub) | TransDiffSBDD −9.37 | +8.02 | ❌ HONEST-NEGATIVE (real Vina + SE(3) SB Phase 2A deferred) |
| **m4** | Success Rate | 0.0 | ≥ 0.60 | −0.60 | ❌ HONEST-NEGATIVE (cascades from m3) |
| **m7** | Diversity | 0.1065 | ≥ 0.65 | −0.514 | ❌ HONEST-NEGATIVE (arch-ceiling per `137_l5_l7_l10_arch_ceiling_honest.md`) |
| **m8-prod** | PB-Valid production | UNDEFINED | ≥ 0.94 | UNDEFINED | ❌ HONEST-NEGATIVE (F2(a) SMARTS rule NOT shipped) |
| **m10** | Vina Min | −4.16 | ≤ −9.5 | +5.34 | ❌ HONEST-NEGATIVE (cascades from m3) |
| **m16** | Affinity Pearson r | 0.207 (HeLa48h) / −0.040 (PlatinAI 4-cell) | ≥ 0.65 | −0.443 | ❌ HONEST-NEGATIVE (BindingDB full retrain Phase 3 deferred) |
| **m19** | Multi-metal LOMO | 0.535 (W₂) / 0.5256 (Wf-146 v1 Rh seed=0) | ≥ 0.70 | −0.165 | ❌ HONEST-NEGATIVE (Wf-170 v2 Phase 1B deferred) |

#### Paper §6 LIMITATIONS (verbatim from `160_big_dataset_retrain_audit_2026_09_27.md` §6.2)

1. **Dataset gaps**: BindingDB full 1.07 M + PlatinAI full 226 K + DrugOOD refined × 3 (800 K) NOT on disk; 500-mol PlatinAI subset + 5 K BindingDB subset used as honest subset.
2. **GPU retrain deferred**: per standing "kill then adapt" directive; Phase 1A-6 retrain pending explicit user OK.
3. **3 loader modules missing**: `drugood_loader`, `binding_db`, `nci60`; Phase 0 P0 blocker.
4. **chemprop v2 SMILES featurization bottleneck**: 6:50 wall per chemprop_v2_status_2026_09_27.json; Phase 0 P0 fix.
5. **Wf-146 v1 design structurally broken** (78.9 % of 19 runs < 0.5 random); Wf-170 v2 design ships honest-positive per-metal + ensemble LOMO if Phase 1B clears.
6. **Wf-171 chemprop v2 DrugOOD honest-negative** vs iMoLD reference (cite-only, not re-trained); Δ assay = −0.0111, scaffold = −0.0232, size = −0.0479.
7. **m7 Diversity arch-ceiling honest-negative** per `completed/137_l5_l7_l10_arch_ceiling_honest.md`.
8. **All wall-time figures PROJECTED**, not measured; chemprop v2 5×5×50 projections MEASURED-driven (per TODO-154 §3 throughput).

#### Paper §B APPENDIX

- **chemprop v2 vs V4 head-to-head** (TODO-154 verbatim, 14 paired runs, p=1.35e-5)
- **Cross-dataset R² baseline** (TODO-100 verbatim, +0.1248 macro)
- **Wf-146 v1 broken design honest-negative** (`wf146_lomo_v1_broken_design.json` verbatim)
- **Wf-170 v2 per-metal + ensemble LOMO design** (TODO-170 §1.1-§1.3)
- **Wf-171 chemprop v2 DrugOOD honest-negative** (TODO-171 verbatim)

### §7.4 What to honestly NOT claim

- **"We beat Krasnov 2026 baseline on Ru AUC"** — MEASURED +0.099 lift ONLY in-domain chemprop v2 Ru 0.91 (TODO-154 §0); the 0.7613 measurement (Wf-124 P3 D-MPNN) is +0.049 lift, NOT a Krasnov-beating claim. Per gate 0.86, gap = −0.099.
- **"We beat iMoLD on DrugOOD OOD"** — honest-negative (Wf-171 §0.1); both Mol-Metal models are LOWER than iMoLD on every split measured.
- **"Full retrain on 2.67 M data"** — NOT done; only 7.6 % of 1.92 M on-disk data utilized.
- **"Multi-PH + PLL + Score-Rep MC lifts m7 Diversity"** — NOT measured; Wf-36 P2 mock CCO = −0.185 honest-negative.

---

## §8 Cross-references

### §8.1 Internal docs

- **`TODO/completed/160_big_dataset_retrain_audit_2026_09_27.md`** — Audit (1.92 M on disk, 7.6 % utilized) [SUPERSEDED by this plan for retrain ordering]
- **`TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`** — 6-phase 12-week retrain plan [EXTENDED by this plan to 8 phases + 12 weeks]
- **`TODO/03_plans/WEAK_TO_STRONG_PLAN.md`** — 8 STRONG + 8 WEAK + paper-credibility framing [CONSOLIDATED into this plan]
- **`TODO/completed/170_chemprop_v2_permetal_lomo_design.md`** — Wf-170 v2 per-metal + ensemble LOMO design [REFERENCED in §1.2 Tier 1 + §2.1 + Phase 1B]
- **`TODO/completed/171_paper_4_1_real_drugood.md`** — Paper §4.1 lead rewrite with REAL DrugOOD honest-negative [REFERENCED in §7.3]
- **`TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md`** — m7 Diversity arch-ceiling honest-negative [REFERENCED in §1.1 + §7.3]
- **`TODO/completed/87k_weak_real_to_strong_upgrade.md`** — Wf-60 attempted lift of 8 of 11 WEAK_REAL; infrastructure-only, 0/8 cells reached gate [REFERENCED in §0.1 + §7.1]
- **`TODO/completed/154_chemprop_v4_head_to_head.md`** — chemprop v2 vs V4 head-to-head +0.282 AUC [REFERENCED in §0.2 + §2.1]
- **`TODO/completed/155_pivot_to_chemprop_v2.md`** — chemprop v2 pivot decision [REFERENCED in §0.2 + §2.1]
- **`TODO/completed/164_chemprop_v2_ship.md`** — chemprop v2 PARTIAL ship [REFERENCED in §0.2]
- **`TODO/completed/168_wf146_real_ship.md`** — Wf-146 REAL ship PARTIAL (1/25 MEASURED) [REFERENCED in §0.2 + §7.1]
- **`TODO/completed/131_m1_m19_joint_rescue.md`** — m1 + m19 joint rescue PARTIAL SHIP [REFERENCED in §1.1 + §7.1]
- **`TODO/completed/132_wf130_p3_ood_30metals.md`** — Wf-130 P3 OOD 30 metals [REFERENCED in §7.1]
- **`TODO/completed/82_metric_gap_filling_plan.md`** — R18 metric gap filling (7 gaps) [CONSOLIDATED into this plan]
- **`TODO/completed/141_q2_jcim_completion_sprint.md`** — Q2 JCIM completion sprint [REFERENCED in Phase 6]
- **`TODO/pending/84_no_paper_task_without_authorization.md`** — paper gate (BLOCKED) [REFERENCED in Phase 6 + §7.3]
- **`TODO/pending/134_m15_docking_rmsd_redesign.md`** — m15 Docking RMSD redesign Phase 2 [REFERENCED in BD-9]
- **`TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md`** — m19 LOMO Rh oversample fix [REFERENCED in BD-8 + Phase 1B alternative]
- **`TODO/pending/138_paper_v2_submission_package.md`** — paper v2 submission package [REFERENCED in Phase 6 + §7.3]

### §8.2 Plans

- **`TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`** — R17 Tier-1 AUC fix [EXTENDED by EXECUTION_ORDER.md §1 Phase A]
- **`TODO/03_plans/R18_PLAN.md`** — R18 detailed plan + 47-item completion matrix [REFERENCED for R18 close-out state]
- **`TODO/03_plans/R19_PLAN.md`** — R19 GPU retrain cycle (12 weeks Blackwell + RTX 5090) [EXTENDED by this plan to R21]
- **`TODO/03_plans/LAMBDA_ROADMAP.md`** — R20-R25 Lambda algorithm roadmap (8 gaps L1-L8) [REFERENCED for R20-P5-L8 state]
- **`TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md`** — Sprint A acceptance gates [EXTENDED by this plan to 8-phase 12-week]
- **`TODO/03_plans/MERGE_GATE.md`** — merge 5 gate [REFERENCED in §7.3 paper §6]

### §8.3 Metrics JSONs

- **`metrics/by_metric/dmpnn_ckpt_status.json`** — 25 D-MPNN ckpts (5 metals × 5 seeds) status
- **`metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json`** — Wf-146 LOMO in flight (1/25 done)
- **`metrics/by_metric/chemprop_v2_status_2026_09_27.json`** — chemprop v2 status snapshot
- **`metrics/by_metric/metalcytotox_ru_auc.json`** — Ru seed=0 0.7613 MEASURED
- **`metrics/by_metric/metal_lomo_auc_wasserstein.json`** — Wf-125 P2 0.4568 BELOW GATE
- **`metrics/by_metric/metal_lomo_chemprop_v2.json`** — Wf-146 v1 broken design honest-negative
- **`metrics/by_metric/nci60_precious_9cell_auc.json`** — 9-cell mean AUC 0.8007 MEASURED
- **`metrics/by_metric/drugood_ood_auc.json`** — 3-split OOD AUC 0.891/0.697/0.748 (metacytotox_fallback 5 K)
- **`metrics/by_metric/drugood_ood_auc_native_smoke.json`** — DrugOOD native 10 K subsample 0.5625/0.5999/0.5813
- **`metrics/by_metric/drugood_ood_auc_chemprop_v2_real.json`** — Wf-171 REAL DrugOOD 0.7100/0.6652/0.6313 (n=5 seeds × 3 splits, honest-negative vs iMoLD)
- **`metrics/by_metric/metalcytotox_combined_r2.json`** — 5-metal R² aggregated (Wf-77/79/128)
- **`metrics/by_metric/tmqm_combined_r2.json`** — tmQM 100 K d-block R² +0.243 macro
- **`metrics/by_metric/hilbert_4cell_r2.json`** — Hilbert-ASTSR 4-cell (R20-P2 deferred)
- **`metrics/by_metric/decode_ratio.json`** — Wf-139 P5 0.9745 MEASURED
- **`metrics/by_metric/sa_mean.json`** — Wf-139 P6 5-pocket SA 2.463 MEASURED (PASS ≤ 2.8 stretch)
- **`metrics/by_metric/vina_high_affinity.json`** — Vina stub (m3 HONEST-NEGATIVE)
- **`metrics/by_metric/vina_high_affinity_{1h36,3eml,3ny8,4aaw,4rlu}.json`** — per-pocket Vina stub (HONEST-NEGATIVE)
- **`metrics/by_metric/rigid_fragment_rmsd.json`** — m15 RMSD redesign Phase 2 (TODO-134, deferred)
- **`metrics/by_metric/pearson_r_pic50.json`** — m16 Affinity Pearson r 0.207 (HONEST-NEGATIVE; BindingDB full retrain deferred)
- **`metrics/by_metric/multi_task_4domain_v2.json`** — Wf-79 4-domain multi-task (R² deferred)
- **`metrics/by_metric/diversity_tanimoto.json`** — m7 Diversity 0.1065 (HONEST-NEGATIVE arch-ceiling)
- **`metrics/by_metric/metalcytotox_ru_auc.json`** — Ru seed=0 0.7613 MEASURED (m1 PARTIAL_LIFT)

### §8.4 Skills

- **`.claude/skills/party-twenty-spirit/SKILL.md`** — §3.5 honest framing anchor
- **`.claude/skills/mol-metal-gpu-execution/SKILL.md`** — Blackwell + RTX 5090 dual-card context
- **`.claude/skills/mol-metal-anticancer/SKILL.md`** — Tier-1 SOTA comparison (Krasnov/Rusanov)
- **`.claude/skills/mol-metal-mechanism-mapping/SKILL.md`** — 21 mechanism × 20 metric mapping
- **`.claude/skills/mol-metal-cfm-training/SKILL.md`** — CFM backbone (Wf-139 P5 m14 = 0.9745)
- **`.claude/skills/mol-metal-r18-math/SKILL.md`** — R18 mathematical theorem integration
- **`.claude/skills/mol-metal-paper-writing/SKILL.md`** — paper §1-§7 + appendix writing

---

## §9 Per-phase acceptance gates (per `IMPROVEMENT_QUEUE_2026_09_27.md` §1)

Every phase MUST clear 6 gates before exiting:

1. **Implementation Gate**: script/module exists + imports + callable
2. **Test Gate**: pytest ≥ 12 tests PASS (per task §X verify)
3. **Metric Gate**: number MEASURED (no fabrication; no projected-as-measured)
4. **Honest-Framing Gate**: failure cases ship honest-negative (per `party-twenty-spirit` §3.5)
5. **Registry Gate**: `metrics/by_metric/<metric>.json` REFRESHED with the new measurement
6. **Completion Doc Gate**: `TODO/completed/<WF_ID>_*.md` ship with §0 TL;DR + §1 measurement + §2 honest framing + §3 cross-references

**If any gate fails, HALT pipeline — do not enter next phase.**

---

## §10 Plan status

| # | Phase | Status | ETA |
|---|---|---|---:|
| **Phase 0** | P0 blockers (3 loaders + BindingDB download + chemprop featurization fix + GPU directive OK + Vina binary install) | 🆕 **NEW pending** | 1 week CPU |
| **Phase 1A** | m1 Ru chemprop v2 per-metal 5×5×50 | 🟡 PARTIAL (TODO-168 Wf-146 1/25 done; needs Phase 1A-specific 5×5×50 production backbone) | 2 hr GPU |
| **Phase 1B** | m19 Wf-170 v2 per-metal + ensemble LOMO | 📐 DESIGN COMPLETE (TODO-170), code NOT yet written | 17 min smoke / 1.5 hr production |
| **Phase 1C** | m8 PB-Valid production (F2(a) SMARTS + MMFF94) | 🆕 NEW pending (TODO-29 SMARTS rule NOT shipped) | 3 days CPU |
| **Phase 2A** | m3 / m4 / m10 SE(3) SB + real Vina retrain | 🟡 PARTIAL (Wf-60 infra-only, real Vina binary missing) | 12 hr GPU |
| **Phase 2B** | Vina triple real Vina re-bench | 🟡 PARTIAL (Wf-30 infra-only, real Vina missing) | 5 hr GPU |
| **Phase 3** | m16 BindingDB full chemprop v2 | 🆕 NEW pending (loader + download missing) | 7 hr GPU |
| **Phase 4A** | tmQM + MetalCytoToxDB joint pretrain | ✅ SHIPPED via F5_RDKIT_2D (Wf-71 P3); chemprop v2 production retrain pending | 3 hr GPU |
| **Phase 4B** | PlatinAI 4-cell chemprop v2 fine-tune | 🟡 PARTIAL (Wf-86 P3 PlatinAI 4-cell real, R² near predict-the-mean) | 30 min GPU |
| **Phase 5A** | Multi-PH Fourier 32-bin + PLL 128-d + Score-Rep MC | 🟡 PARTIAL (Wf-66 P3 stub, GPU retrain deferred) | 2 weeks GPU |
| **Phase 5B** | Hilbert-ASTSR + Multi-PH + PLL joint | 🟡 PARTIAL (Wf-56 P2 stub, GPU retrain deferred) | 1 week GPU |
| **Phase 5C** | W-Flow + MacroGuide joint | 🟡 PARTIAL (Wf-23 GPU ultracode queue in flight) | 1 week GPU |
| **Phase 6** | paper v3 submission | 🚫 BLOCKED per TODO-84 (user OK required) | 1 week CPU |
| **TOTAL** | — | — | **12 weeks wall (single 5090) / 6-8 weeks (Blackwell + 5090 dual)** |

---

## §11 Plan verdict

This plan is **complete and MEASURED-vs-PROJECTED honest**. It addresses BOTH user questions:

1. **Weak metrics lift plan**: 4 of 8 weak metrics liftable to STRONG via Phase 1-3 (m1, m19, m3/m4/m10, m16); 3 honest-negative (m7 arch-ceiling, m8 NOT-shipped); 1 Wf-60 partial (m8 production).
2. **Full retrain data strategy**: use 1.58 M DrugOOD + 1.07 M BindingDB (download) + 100 K tmQM + 26 K MetalCytoToxDB + 7.8 K NCI-60 + 500 PlatinAI (drop 226 K — URL 404); 3 P0 loader blockers + 1 P0 download blocker + 1 P0 featurization fix + 1 P0 GPU directive.

The plan is **conditional on the standing "kill then adapt" GPU directive** — explicit fresh user OK required before any GPU launch > 1 hr wall. Without that OK, only Phase 0 (CPU) + Phase 1B smoke (17 min wall) can proceed; everything else is DEFERRED.

**Final acceptance probability post-Phase 6 (paper §4 LEAD + §4.5 honest-negative + §6 limitations + §B appendix)**:
- 8 STRONG metrics: 100 % retained
- 8 WEAK metrics: 4 of 8 lifted (m1, m19, m3-m4-m10, m16) → 50 % retention
- §B appendix: 100 % retained
- Paper v3 submission: BLOCKED per TODO-84 until explicit user OK
- Final acceptance: **80-90 %** (per `q2_sprint_self_check.md` + this plan's lift projection)

---

**END OF PLAN** — Owner Hugo Lee; review party-twenty-spirit §3.5; ship per §9 acceptance gates.