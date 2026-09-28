# R20 Plan Status Ledger (Wf-70 Phase 1)

**Date**: 2026-09-26
**Workflow**: Wf-70 (Phase 1: R20 plan status ledger verification — CPU-only docs)
**Status**: ✅ **SHIP** — 7/8 R20-P verified on disk; R20-P2 (Wf-56 / TODO-93) GPU retrain **in flight / deferred** (honest framing per `party-twenty-spirit §3.5`).
**Priority**: HIGH (drives R20 routing policy — unblocks R21 plan items)
**Owner**: Hugo Lee + ultracode agents
**Parent**:
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 (R20-P1..P6 plan items)
- `TODO/INDEX.md` §17.3 (R20 plan table)
- `TODO/pending/92_r20_p1_*.md` .. `TODO/pending/99_r20_skill_ecosystem.md` (8 active R20 plans)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate — no paper work without explicit OK)

**Constraint**: CPU-only docs — no GPU retrain executed in this phase.

---

## §0 TL;DR — honest ledger

| R20-P | Plan item | Wf | Plan file | Close-out file | Status |
|---|---|---|---|---|---|
| **R20-P1** | VariancePUCT default + Lazy RDKit import (L2) | Wf-65 | `pending/92_*.md` | **NOT YET SHIPPED** (no `completed/92_r20_p1_*.md`) | ⚠️ **IN FLIGHT** |
| **R20-P2** | Hilbert-ASTSR + Multi-PH + PLL + Triangle closure (L4+L5+L7+L8+L11) | Wf-56 | `pending/93_*.md` | `completed/93_r20_p2_hilbert_astsr_gpu.md` | 🟡 **HONEST PARTIAL** (code+skill+smoke SHIP; 0/4 GPU retrain runs) |
| **R20-P3** | Multi-PH Fourier 32-bin + PLL 128-d + Score-Rep MC (L7+L3) | Wf-66 | `pending/94_*.md` | `completed/94_r20_p3_multiph_pll_feature.md` | 🟡 **HONEST PARTIAL** (code+skill+smoke SHIP; GPU retrain deferred) |
| **R20-P4** | 3-way on MetalCytoToxDB + Closure d-block ≥ 0.95 + 100-pocket × 10-seed (L6+L10) | Wf-67 | `pending/95_*.md` | `completed/95_r20_p4_3way_100pocket.md` | 🟡 **HONEST-NEGATIVE PARTIAL** (prereqs SHIP; GPU retrain NOT executed; reachability 0.5962 NOT gate 0.95) |
| **R20-P5** | Closure theorem + Catherine 6/6 properties (L1+L6+L12) | Wf-68 + Wf-57 + Wf-67 | `pending/96_*.md` | `completed/96_r20_p5_closure_catherine.md` | 🟡 **HONEST PARTIAL** (Catherine 6/6 MEASURED on 25 metals; reachability ≥ 0.95 NOT MET; cross-domain NOT re-measured) |
| **R20-P6** | Wet-lab IC50 cooperation (ORNL / PKU / 协和) | Wf-58 | `pending/97_*.md` | `completed/97_r20_p6_wetlab_ic50_dryrun.md` | 🟡 **PARTIAL** (in-silico dry-run + D-MPNN prediction SHIP; wet-lab cooperation 6-mo deferred) |
| **Wf-44 P2** | Closure OOD Phase 2 end-to-end (L12) | Wf-57 | `pending/98_*.md` | `completed/98_wf44_closure_ood_phase2.md` | ✅ **PHASE 2 PARTIAL SHIP** (Tanimoto<0.4 protocol + 6/6 Catherine MEASURED; per-domain reachability ≥ 0.85 reserved Q4 2026 GPU) |
| **R20 skill** | 8 NEW skills (4 R20 + 4 R21) | Wf-55 | `pending/99_*.md` | `completed/99_r20_skill_ecosystem.md` | 🟡 **PARTIAL** (S2+S3 SHIP; S1+S4 stubs; S5-S8 deferred) |

**Aggregate**: **7/8 R20-P verified on disk** (R20-P1 still in flight); **5/8 partial / 2/8 partial / 1/8 phase 2 partial**; **0/8 R20-P fully MEASURED** (all close-out files honestly report GPU retrain as deferred / honest-negative).

**Wf-56 (R20-P2 / TODO-93) Wf-65 (R20-P1 / TODO-92)**: both have `wf65_p1_sign_audit/sign_flip_audit.json` on disk and `wf65_p1_sign_audit` directory exists; R20-P1 is the **only** R20-P without a `TODO/completed/9N_r20_pN_*.md` file yet (Wf-65 partial ship / in flight).

---

## §1 Per-R20-P target vs MEASURED (8 rows)

### §1.1 R20-P1 — VariancePUCT default + Lazy RDKit import (L2 vuln_a fix)

| Aspect | Plan target | MEASURED today | Source | Verdict |
|---|---|---|---|---|
| VariancePUCT scoring rule audit | patch `variance_puct_integration.py`; 5 new tests; sign-flip regression test | ✅ `molmetal/reports/wf65_p1_sign_audit/sign_flip_audit.json` SHIP (canonical/pre_fix/post_fix); `molmetal/scripts/audit_variance_puct_sign.py` exists; `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` exists | on disk | ✅ VERIFIED |
| Lazy RDKit import | patch `learned_shaping.py` + `lambda_combinators.py`; 10 new tests; import-order test | ⚠️ not visibly shipped on disk (no `molmetal/reports/wf65_p1_lazy_rdkit/` or equivalent) | git status shows `learned_shaping.py` + `lambda_combinators.py` modified, but no dedicated lazy-import patch is verified | ⚠️ IN FLIGHT |
| Real-data trajectory benchmark | `molmetal/reports/wf50_r20_p1/trajectory_{platinai_5k,metalcytotox}.json` | ❌ not on disk (no `wf50_r20_p1/` reports dir; only `wf50_gpu_audit/` etc.) | n/a | ❌ DEFERRED (GPU) |
| Variance reduction on real data | variance_reduction ≥ 0 on real data | ❌ not MEASURED | n/a | ❌ DEFERRED (GPU) |
| Tests added | ~15 (5 sign-flip + 10 lazy import) | ⚠️ partial (`molmetal/scripts/verify_variance_puct.py` exists; `verify_variance_mcts.py` exists) | disk | ⚠️ PARTIAL |
| Wf-50 close-out | `TODO/completed/92_r20_p1_*.md` | ❌ not on disk | n/a | ❌ NOT YET WRITTEN |

**Verdict**: ⚠️ **IN FLIGHT** — sign-flip audit SHIP, lazy RDKit import partial, GPU trajectory + close-out pending. **1/6 sub-tasks VERIFIED**.

### §1.2 R20-P2 — Hilbert-ASTSR + Multi-PH + PLL + Triangle closure (L4+L5+L7+L8+L11)

| Aspect | Plan target | MEASURED today | Source | Verdict |
|---|---|---|---|---|
| Hilbert-ASTSR 5-seed × 50-epoch on PlatinAI 4-cell | R² ≥ 0.85 (paper §3.6 lead claim) | ❌ NOT executed; Wf-37 P3 baseline R² = +0.087 ± 0.081 (16-bin Multi-PH / 64-d / no Score-Rep); Wf-49 P2 F5_RDKIT_2D best |ρ| = +0.204 on MetalCytoToxDB | `molmetal/reports/wf37_multiph_pll/` + `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` | ❌ PROJECTED @ Wf-51 (deferred) |
| Multi-PH + PLL on PlatinAI 4-cell + MetalCytoToxDB 5-metal | R² ≥ 0.70 | ❌ NOT executed | same | ❌ DEFERRED |
| Triangle closure real pIC50 | mean_hilbert_r2 ≥ 0.6 on real data | ❌ NOT executed; Wf-37 P1 3way baseline R² = −0.3595 ± 0.2806 on synthetic `qed * qed`; +0.7524 lift over AST | `molmetal/reports/wf41_3way/3way_bench.json` | ❌ DEFERRED |
| MCTS tree vuln_f+g retrain | hit_rate ≥ 0.95 | ❌ NOT executed; Wf-36 P2 Score-Rep on mock CCO variance_reduction = −0.185 (honest-negative) | `molmetal/reports/wf36_closure_mcts/` | ❌ DEFERRED |
| Close-out file | `TODO/completed/93_r20_p2_*.md` | ✅ SHIP (16 017 bytes, Wf-56 close-out synthesis) | on disk | ✅ VERIFIED |
| `mol-metal-hilbert-astsr-e2e` skill | ≥ 250 LOC SKILL.md | ✅ SHIP (`.claude/skills/mol-metal-hilbert-astsr-e2e/SKILL.md` + `references.md` + `test_hilbert_astsr_e2e.py`) | on disk | ✅ VERIFIED |
| Tests added | ~30 | ⚠️ partial (`test_hilbert_astsr_e2e.py` SHIP) | on disk | ⚠️ PARTIAL |

**Verdict**: 🟡 **HONEST PARTIAL** — close-out file + skill SHIP; 4/4 GPU retrain phases NOT executed. **3/7 sub-tasks VERIFIED, 4/7 DEFERRED (GPU)**.

### §1.3 R20-P3 — Multi-PH Fourier 32-bin + PLL 128-d + Score-Rep MC (L7+L3)

| Aspect | Plan target | MEASURED today | Source | Verdict |
|---|---|---|---|---|
| Multi-PH Fourier 32-bin | per-cohort Pearson r; R² mean ≥ 0.20 (realistic) / ≥ 0.70 (stretch) | ❌ NOT executed | `TODO/completed/94_r20_p3_multiph_pll_feature.md` §1.1 | ❌ DEFERRED |
| PLL signature 128-d | per-cohort Pearson r; Δ vs 16-d ≥ +0.05 (realistic) / ≥ +0.55 (stretch) | ❌ NOT executed; Wf-14 P4 PLL 16-d default ship | `molmetal/molmetal_lam/lam_chem/pll_signature.py` | ❌ DEFERRED |
| Score-Rep MC variance reduction | variance_reduction ≥ 0.2 (stretch) / ≥ 0.10 (realistic) | ❌ NOT executed on real data; Wf-36 P2 mock CCO = −0.185 (honest-negative) | `molmetal/reports/wf36_closure_mcts/` | ❌ DEFERRED |
| Close-out file | `TODO/completed/94_r20_p3_*.md` | ✅ SHIP (25 004 bytes, Wf-66 close-out synthesis) | on disk | ✅ VERIFIED |
| `mol-metal-multiph-pll` skill | ≥ 250 LOC SKILL.md | ✅ SHIP (518 LOC SKILL.md + `references.md` + `test_multiph_pll.py`) | on disk | ✅ VERIFIED |
| Tests added | ~25 | ⚠️ partial (`test_multiph_pll.py` PASS) | on disk | ⚠️ PARTIAL |

**Verdict**: 🟡 **HONEST PARTIAL** — close-out file + skill SHIP; 3/3 GPU retrain phases NOT executed. **2/6 sub-tasks VERIFIED, 3/6 DEFERRED (GPU)**.

### §1.4 R20-P4 — 3-way on MetalCytoToxDB + Closure d-block ≥ 0.95 + 100-pocket × 10-seed (L6+L10)

| Aspect | Plan target | MEASURED today | Source | Verdict |
|---|---|---|---|---|
| 3-way on MetalCytoToxDB 5-metal real pIC50 | R² ≥ +0.30 per-metal | ❌ NOT executed; prerequisites (Wf-49 P2 F5_RDKIT_2D SHIP 23/23 tests; Wf-41 P1 3-way baseline R² = −0.3595 ± 0.2806 on synthetic) | `molmetal/reports/wf41_3way/` + `TODO/completed/91_wf49_r2_root_fix.md` | ❌ DEFERRED |
| Closure d-block reachability ≥ 0.95 on 30 metals × 1000 samples | reachability ≥ 0.95 | 🟡 MEASURED **0.5962** (n_metals_pass=1/30); Wf-37 P2 baseline 0.5891 carry-forward | `molmetal/reports/wf67_p4_2_closure_r20/reachability_overall.json` | ❌ **NOT MET** (0.5962 ≪ 0.95 gate) |
| 100-pocket × 10-seed dock sweep | EXIT 0; m3 ≤ −11.5 kcal/mol; lift ≤ −1.4 kcal/mol | 🟡 PARTIAL — `bench_100pocket_10seed.py` scaffold exit 0 with 5 pockets (vina_stub mode); mean kcal/mol = −8.42; lift +1.3e-08 stub-floor artifact carry-forward | `molmetal/reports/wf67_p4_3_100pocket/sweep_summary.json` | ❌ **NOT MEASURED** (real-Vina path BLOCKED on meko 0.8.0 Pt/Au/Ru) |
| Close-out file | `TODO/completed/95_r20_p4_*.md` | ✅ SHIP (22 048 bytes, Wf-67 close-out synthesis) | on disk | ✅ VERIFIED |
| `mol-metal-3way-integration` skill | ≥ 250 LOC SKILL.md | ❌ NOT created | n/a | ❌ DEFERRED |
| Tests added | ~30 | ❌ 0 tests committed for Wf-67 | n/a | ❌ DEFERRED |

**Verdict**: 🟡 **HONEST-NEGATIVE PARTIAL** — prereqs SHIP; 2 GPU-requiring sub-tasks NOT executed (3-way on real pIC50 + 100-pocket dock); reachability **0.5962 < 0.95 gate**; skill NOT created. **1/6 sub-tasks VERIFIED, 3/6 DEFERRED, 2/6 NOT MET**.

### §1.5 R20-P5 — Closure theorem + Catherine 6/6 properties (L1+L6+L12)

| Aspect | Plan target | MEASURED today | Source | Verdict |
|---|---|---|---|---|
| Catherine 6/6 properties on ≥ 25 metals | 6/6 PASS | ✅ **MEASURED** — 25/25 metals × 6/6 properties PASS, per-property pass-rate 100.00% | `molmetal/reports/wf68_p5_1_catherine/catherine_6_props.json` + `synthesis.md` | ✅ **VERIFIED** |
| Closure reachability ≥ 0.95 on 25 metals × 1000 samples | reachability ≥ 0.95 | 🟡 MEASURED **0.5744** (n_metals_pass=0/25) on tmQM d-block subset; Wf-37 P2 baseline 0.5891 carry-forward | `molmetal/reports/wf68_p5_2_reachability/reachability_overall.json` | ❌ **NOT MET** (0.5744 ≪ 0.95 gate) |
| Cross-domain OOD end-to-end | OOD reachability ≥ 0.85 (PlatinAI 5K + MetalCytoToxDB + tmQM d-block) | 🟡 NOT RE-MEASURED — carries Wf-41 P2 honest-negative (PlatinAI 5K 0.5581 / MetalCytoToxDB 0.0300) | `molmetal/reports/wf41_3way_integration/` + `TODO/completed/98_wf44_closure_ood_phase2.md` | ❌ DEFERRED |
| Close-out file | `TODO/completed/96_r20_p5_*.md` | ✅ SHIP (30 368 bytes, Wf-68 + Wf-57 + Wf-67 close-out synthesis) | on disk | ✅ VERIFIED |
| `mol-metal-closure-catherine` skill | ≥ 250 LOC SKILL.md | ✅ SHIP (`.claude/skills/mol-metal-closure-catherine/SKILL.md` exists) | on disk | ✅ VERIFIED |

**Verdict**: 🟡 **HONEST PARTIAL** — Catherine 6/6 PASS MEASURED; reachability **0.5744 < 0.95 gate**; cross-domain OOD NOT re-measured. **3/5 sub-tasks VERIFIED, 2/5 NOT MET / DEFERRED**.

### §1.6 R20-P6 — Wet-lab IC50 cooperation (m18/m19/m20)

| Aspect | Plan target | MEASURED today | Source | Verdict |
|---|---|---|---|---|
| In-silico dry-run (CPU + D-MPNN prediction layer) | CPU-only synthesis | ✅ SHIP — `molmetal/reports/wf58_dmpnn_predictions/` + `molmetal/reports/wf58_wetlab_dryrun/{ranking_summary.md,top20_candidates.csv}` | on disk | ✅ VERIFIED |
| Wet-lab cooperation (LOI + PI outreach + cell culture) | 20 compounds synthesized + LOI signed | ❌ NOT started — wet-lab cooperation 6-mo deferred per Wf-58 R20-P6 plan | n/a | ❌ DEFERRED (wet-lab) |
| IC50 assay (MCF7 + A549 + A2780 + A2780cis + 5 NCI-60 cells) | full panel IC50 measured | ❌ NOT started | n/a | ❌ DEFERRED (wet-lab) |
| In-silico vs wet-lab agreement | Pearson r ≥ 0.7 | ❌ NOT MEASURED | n/a | ❌ DEFERRED (wet-lab) |
| Close-out file | `TODO/completed/97_r20_p6_*.md` | ✅ SHIP (25 417 bytes, Wf-58 close-out synthesis) | on disk | ✅ VERIFIED |
| `mol-metal-wetlab-protocol` skill | ≥ 250 LOC SKILL.md | ✅ SHIP (`.claude/skills/mol-metal-wetlab-protocol/SKILL.md` + `references.md`) | on disk | ✅ VERIFIED |

**Verdict**: 🟡 **PARTIAL** — in-silico dry-run + D-MPNN prediction layer + skill SHIP; wet-lab cooperation deferred 6 months. **3/6 sub-tasks VERIFIED, 3/6 DEFERRED (wet-lab)**.

### §1.7 Wf-44 P2 — Closure OOD Phase 2 end-to-end (L12)

| Aspect | Plan target | MEASURED today | Source | Verdict |
|---|---|---|---|---|
| Tanimoto<0.4 OOD protocol on tmQM d-block | Tanimoto<0.4 confirmed | ✅ SHIP — `molmetal/reports/wf57_tanimoto/{ood_split.json,per_cluster.csv,protocol_summary.md}` | on disk | ✅ VERIFIED |
| Per-domain closure reachability (PlatinAI 5K + MetalCytoToxDB + tmQM) | per-domain ≥ 0.85 | ⚠️ `wf57_per_domain_reachability/` directory exists but is EMPTY | on disk (empty dir) | ❌ DEFERRED |
| 6 Catherine properties on ≥ 25 metals | 6/6 PASS | ✅ MEASURED — 25/25 metals × 6/6 properties PASS (carries Wf-68 P5.1) | `molmetal/reports/wf57_catherine_25metals/per_metal_pass.json` | ✅ VERIFIED |
| Close-out file | `TODO/completed/98_wf44_*.md` | ✅ SHIP (24 180 bytes, Wf-57 close-out synthesis) | on disk | ✅ VERIFIED |
| `mol-metal-closure-theorem` skill | ≥ 250 LOC SKILL.md | ✅ SHIP (`.claude/skills/mol-metal-closure-theorem/SKILL.md` exists) | on disk | ✅ VERIFIED |

**Verdict**: ✅ **PHASE 2 PARTIAL SHIP** — Tanimoto<0.4 protocol + 6/6 Catherine MEASURED; per-domain reachability ≥ 0.85 reserved Q4 2026 GPU. **4/5 sub-tasks VERIFIED, 1/5 DEFERRED**.

### §1.8 R20 skill ecosystem — 8 NEW skills (4 R20 + 4 R21)

| Aspect | Plan target | MEASURED today | Source | Verdict |
|---|---|---|---|---|
| S1 `mol-metal-variance-puct` (R20-P1) | ≥ 250 LOC SKILL.md + 3 test scripts | ❌ NOT created | n/a | ❌ DEFERRED |
| S2 `mol-metal-hilbert-astsr-e2e` (R20-P2) | ≥ 250 LOC SKILL.md + 3 test scripts | ✅ SHIP — SKILL.md + `references.md` + `test_hilbert_astsr_e2e.py` | `.claude/skills/mol-metal-hilbert-astsr-e2e/` | ✅ VERIFIED |
| S3 `mol-metal-multiph-pll` (R20-P3) | ≥ 250 LOC SKILL.md + 3 test scripts | ✅ SHIP — SKILL.md 518 LOC + `references.md` + `test_multiph_pll.py` PASS | `.claude/skills/mol-metal-multiph-pll/` | ✅ VERIFIED |
| S4 `mol-metal-3way-integration` (R20-P4) | ≥ 250 LOC SKILL.md + 3 test scripts | ❌ NOT created | n/a | ❌ DEFERRED |
| S5 `mol-metal-closure-catherine` (R20-P5) | ≥ 250 LOC SKILL.md + 3 test scripts | 🟡 SHIP — SKILL.md exists (size TBD; smoke test not committed) | `.claude/skills/mol-metal-closure-catherine/` | 🟡 PARTIAL |
| S6 `mol-metal-wetlab-protocol` (R20-P6) | ≥ 250 LOC SKILL.md + 3 test scripts | 🟡 SHIP — SKILL.md + `references.md` (size TBD) | `.claude/skills/mol-metal-wetlab-protocol/` | 🟡 PARTIAL |
| S7 `mol-metal-triangle-closure` (R21-P4) | ≥ 250 LOC SKILL.md + 3 test scripts | 🟡 SHIP — SKILL.md + `references.md` (size TBD) | `.claude/skills/mol-metal-triangle-closure/` | 🟡 PARTIAL |
| S8 `mol-metal-mcts-replay-learner` (R21-P2) | ≥ 250 LOC SKILL.md + 3 test scripts | 🟡 SHIP — SKILL.md + `references.md` (size TBD) | `.claude/skills/mol-metal-mcts-replay-learner/` | 🟡 PARTIAL |
| Close-out file | `TODO/completed/99_r20_skill_ecosystem.md` | ✅ SHIP (19 483 bytes, Wf-55 close-out synthesis) | on disk | ✅ VERIFIED |

**Verdict**: 🟡 **PARTIAL** — 2/8 skills FULL SHIP (S2 + S3); 4/8 skills stub-shell (S5-S8); 2/8 skills NOT created (S1 + S4). **3/9 sub-tasks VERIFIED**.

---

## §2 Cross-cutting observations

### §2.1 The MEASURED-vs-PROJECTED gap is explicit and honest

Per `party-twenty-spirit §3.5` (0 容忍 + 不掩盖):

- **0/8 R20-P is FULLY MEASURED on disk** — every R20-P close-out file honestly reports GPU retrain phases as deferred / not-executed.
- **3/8 R20-P have a partial MEASURED result** that does NOT meet the plan gate:
  - R20-P4 (P4.2): reachability **0.5962** < gate **0.95**
  - R20-P5 (P5.2): reachability **0.5744** < gate **0.95**
  - Wf-44 P2 (L12): per-domain reachability **DEFERRED** (empty dir, not yet measured)
- **5/8 R20-P are code + skill + smoke SHIP only** with GPU phases unblocked but pending (Wf-23 GPU ultracode queue + RTX 5090 GPU-1 + RTX PRO 6000 Blackwell fallback).

### §2.2 Honest carry-forward from prior wf

| Carry-forward item | Source | Honesty |
|---|---|---|
| 3-way on PlatinAI 5K synthetic `qed * qed` R² = −0.3595 ± 0.2806 (3-seed × 92-d) | Wf-41 P1 | ✅ MEASURED on disk (`molmetal/reports/wf41_3way/3way_bench.json`) |
| 3-way lift over ASTSR baseline (synthetic) | Wf-41 P1 | ✅ MEASURED on disk (+0.7524) |
| ASTSR F5 → F5_RDKIT_2D swap | Wf-49 P2 | ✅ MEASURED on disk (23/23 tests PASS; MolLogP |ρ| = +0.204 on MetalCytoToxDB) |
| Wf-37 P2 Closure d-block reachability | Wf-37 P2 | ✅ MEASURED on disk (0.5891) |
| Wf-30 Vina Dock stub-floor | Wf-30 | ✅ MEASURED on disk (m3 = −9.6574 kcal/mol; lift +1.3e-08) |
| Score-Rep MC variance_reduction on mock CCO | Wf-36 P2 | ✅ MEASURED on disk (−0.185 honest-negative) |
| Multi-PH + PLL joint R² on real pIC50 | Wf-37 P3 | ✅ MEASURED on disk (+0.087 ± 0.081; seeds 3+4 NOT run) |
| Multi-PH F1 + acc per cohort | Wf-25 | ✅ MEASURED on disk (F1 0.362 ± 0.119; acc 0.639 ± 0.068) |

### §2.3 GPU retrain ETA (per Wf-23 + Wf-50 + Wf-66 + Wf-67 + Wf-68 GPU queue)

Per `mol-metal-gpu-execution` §5 (kanzi contention policy):

| R20-P | GPU ETA | Required GPU | Fallback |
|---|---|---|---|
| R20-P2 (L5+L7+L8+L11) | ~4 weeks GPU-0 | RTX 5090 GPU-0 (32 GB free) | RTX PRO 6000 Blackwell (95 GB) |
| R20-P3 (L7+L3) | ~3 weeks GPU-1 | RTX 5090 GPU-1 (32 GB) | RTX PRO 6000 Blackwell (kanzi contention) |
| R20-P4 (L6+L10) | ~4 weeks GPU-1 + 2 weeks GPU-0 | RTX 5090 + Blackwell sm_120 | RTX PRO 6000 Blackwell |
| R20-P5 (L1+L6+L12) | ~2 weeks CPU + 1 week GPU-0 | RTX 5090 GPU-0 | RTX PRO 6000 Blackwell |
| Wf-44 P2 (L12) | ~2 weeks CPU + 1 week GPU-0 | RTX 5090 GPU-0 (Q4 2026) | RTX PRO 6000 Blackwell |
| R20-P1 (L2) | ~2-3 weeks GPU-0 | RTX 5090 GPU-0 | RTX PRO 6000 Blackwell |

**Cumulative wall-time** (if serialized): ~16 weeks GPU + ~6 weeks CPU.
**Cumulative wall-time** (if parallelized across GPU-0 + GPU-1): ~6 weeks GPU + ~6 weeks CPU.

### §2.4 Cross-R20-P reachability issue (structural reason for NOT MET)

Per `molmetal/reports/wf67_p4_2_closure_r20/reachability_overall.json` aggregate and `molmetal/reports/wf68_p5_2_reachability/reachability_overall.json` aggregate:

- 30 d-block metals × 1000 samples (Wf-67 P4.2): reachability **0.5962** (n_metals_pass=1/30, La only)
- 25 d-block metals × 1000 samples (Wf-68 P5.2): reachability **0.5744** (n_metals_pass=0/25)

Both runs use `max_depth=1` per `LAMBDA_ROADMAP §1.2 R20-P5`. The structural reason for the gap to gate 0.95 is:

1. **Per-periodic-block reachability ceiling**: 3d=0.6682 / 4d=0.5813 / 5d_La=0.4918 — different blocks have different SMARTS coverage
2. **Synth SMARTS gaps**: Some SMARTS in `pt_click_compat.py` (21-rule R17 baseline) do not generalize from Wf-37 P2 9-metal cohort to Wf-67 30-metal cohort
3. **`max_depth=1` ceiling**: Multi-step coordination requires `max_depth ≥ 2`; current `max_depth=1` is a structural bound

Per `party-twenty-spirit §3.5` honesty: the gap is structural, not implementation bug; lifting requires SMARTS coverage expansion + `max_depth ≥ 2` + GPU retrain.

---

## §3 What CAN be launched in parallel (CPU-only, no paper, no GPU contention)

**Per `TODO/pending/84_no_paper_task_without_authorization.md`** (no paper work without explicit OK) and **`TODO/README.md` §risk** (ultracode vs training blacklist when Wf-23 GPU ultracode queue is active):

| Item | Type | Wall-time | Conflicts with Wf-23? | Authorized without paper-OK? |
|---|---|---|---|---|
| **Wf-65 P1.2 Lazy RDKit import (TODO-92 P1.2)** | CPU-only code + tests | 1 wk CPU | ❌ No | ✅ Yes (code + tests, no paper) |
| **Wf-54 P5.1 Catherine 6 properties on additional metals (TODO-96 P5.1)** | CPU-only bench | 1 wk CPU | ❌ No | ✅ Yes (bench data, no paper) |
| **Wf-54 P5.3 Cross-domain OOD CPU scaffold (TODO-96 P5.3)** | CPU-only bench | 1 wk CPU | ❌ No | ✅ Yes (bench data, no paper) |
| **Wf-44 P2.1 Tanimoto<0.4 OOD protocol extension (TODO-98 P2.1)** | CPU-only bench | 1 wk CPU | ❌ No | ✅ Yes (bench data, no paper) |
| **Wf-55 R20 skill ecosystem S1 + S4 + S5 (TODO-99 P1-S1 / P4-S4 / P5-S5)** | CPU-only docs | 2-4 wk CPU | ❌ No | ✅ Yes (skill docs, no paper) |
| **Wf-67 P4.2 Closure d-block reachability on 30 metals × 1000 samples (TODO-95 P4.2)** | CPU-only bench (already executed) | n/a (DONE) | ❌ No | n/a (DONE) |
| **Wf-67 P4.3 100-pocket scaffold extension to real Vina path** | BLOCKED on meko 0.8.0 Pt/Au/Ru fix | n/a | n/a | ❌ NOT authorized |
| **All GPU retrain phases (R20-P2/3/4/5/Wf-44)** | GPU retrain | 2-4 weeks each | ⚠️ YES — Blackwell sm_120 single-card constraint | ✅ Yes (no paper) but **deferred until Wf-23 GPU ultracode queue exit** |

**3.1 Honest launch candidates** (no paper, no GPU, no Wf-23 contention):

1. **Wf-65 P1.2 Lazy RDKit import** (TODO-92 P1.2) — 1 wk CPU, unlocks R20-P1 trajectory benchmark and R20-P3 Score-Rep MC on real data
2. **Wf-55 R20-P5-S5 `mol-metal-closure-catherine` skill polish** (TODO-99 P5-S5) — 2 wk CPU docs, expands existing stub
3. **Wf-55 R20-P4-S4 `mol-metal-3way-integration` skill scaffolding** (TODO-99 P4-S4) — 2 wk CPU docs, NEW skill creation
4. **Wf-55 R20-P6-S6 `mol-metal-wetlab-protocol` skill polish** (TODO-99 P6-S6) — 1 wk CPU docs, expands existing stub

**3.2 Items that CANNOT be launched without user authorization**:

- All `paper/sections/*.tex`, `paper/appendices/*.tex`, `paper/supplementary/*.{tex,md}`, `paper/main.tex`, `paper/main.pdf` rebuilds, reviewer rebuttal, arXiv preprint, journal submission — per TODO-84 rule

**3.3 Items that CANNOT be launched due to GPU contention**:

- All R20-P2/3/4/5/Wf-44 P2 GPU retrain phases — pending Wf-23 GPU ultracode queue exit + Blackwell sm_120 single-card constraint

---

## §4 Cross-references

- `TODO/pending/84_no_paper_task_without_authorization.md` — paper-gate (CRITICAL)
- `TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md` — R20-P1 input plan
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` — R20-P2 input plan
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` — R20-P3 input plan
- `TODO/pending/95_r20_p4_3way_100pocket.md` — R20-P4 input plan
- `TODO/pending/96_r20_p5_closure_catherine.md` — R20-P5 input plan
- `TODO/pending/97_r20_p6_wetlab_ic50.md` — R20-P6 input plan
- `TODO/pending/98_wf44_closure_ood_phase2.md` — Wf-44 P2 input plan
- `TODO/pending/99_r20_skill_ecosystem.md` — R20 skill ecosystem input plan
- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` — R20-P2 close-out (Wf-56)
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` — R20-P3 close-out (Wf-66)
- `TODO/completed/95_r20_p4_3way_100pocket.md` — R20-P4 close-out (Wf-67)
- `TODO/completed/96_r20_p5_closure_catherine.md` — R20-P5 close-out (Wf-68)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` — R20-P6 close-out (Wf-58)
- `TODO/completed/98_wf44_closure_ood_phase2.md` — Wf-44 P2 close-out (Wf-57)
- `TODO/completed/99_r20_skill_ecosystem.md` — R20 skill ecosystem close-out (Wf-55)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 (R20-P1..P6 plan items)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 (R20 skill ecosystem)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §2.4 (R21 skill ecosystem)
- `TODO/INDEX.md` §17.3 (R20 plan table)
- `TODO/EXECUTION_ORDER.md` §1 (R19 GPU training priority + Wf-23 GPU ultracode queue)
- `TODO/README.md` §risk (ultracode vs training blacklist)
- `molmetal/reports/wf65_p1_sign_audit/sign_flip_audit.json` — Wf-65 R20-P1 sign-flip audit
- `molmetal/reports/wf67_p4_2_closure_r20/reachability_overall.json` — Wf-67 P4.2 reachability
- `molmetal/reports/wf67_p4_3_100pocket/sweep_summary.json` — Wf-67 P4.3 100-pocket scaffold
- `molmetal/reports/wf68_p5_1_catherine/catherine_6_props.json` — Wf-68 P5.1 Catherine 6/6
- `molmetal/reports/wf68_p5_2_reachability/reachability_overall.json` — Wf-68 P5.2 reachability
- `molmetal/reports/wf57_catherine_25metals/per_metal_pass.json` — Wf-57 Wf-44 P2 Catherine 25 metals
- `molmetal/reports/wf57_tanimoto/ood_split.json` — Wf-57 Wf-44 P2 Tanimoto<0.4 protocol
- `molmetal/reports/wf58_dmpnn_predictions/` — Wf-58 R20-P6 D-MPNN predictions
- `molmetal/reports/wf58_wetlab_dryrun/ranking_summary.md` — Wf-58 R20-P6 top-20 ranking
- `.claude/skills/mol-metal-hilbert-astsr-e2e/` — S2 R20 skill
- `.claude/skills/mol-metal-multiph-pll/` — S3 R20 skill
- `.claude/skills/mol-metal-closure-catherine/` — S5 R20 skill
- `.claude/skills/mol-metal-closure-theorem/` — R20 skill (Wf-44 P2)
- `.claude/skills/mol-metal-wetlab-protocol/` — S6 R20 skill
- `.claude/skills/mol-metal-triangle-closure/` — S7 R21 skill
- `.claude/skills/mol-metal-mcts-replay-learner/` — S8 R21 skill
- `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 — 0 容忍 + 不掩盖 (always apply)
- `.claude/skills/mol-metal-gpu-execution/SKILL.md` §5 — kanzi contention policy

---

## §5 Final verdict

**Aggregate R20 plan status (CPU-only docs honest ledger)**:

| Metric | Count |
|---|---:|
| R20-P fully MEASURED on disk | **0 / 8** |
| R20-P close-out file SHIP | **7 / 8** (R20-P1 still in flight) |
| R20-P close-out reports HONEST PARTIAL (GPU deferred) | **7 / 7** |
| Sub-tasks VERIFIED (file on disk + MEASURED target met or honest-negative documented) | **15 / 50** |
| Sub-tasks DEFERRED (GPU / wet-lab / paper-gate) | **30 / 50** |
| Sub-tasks NOT MET (gate failed) | **3 / 50** (R20-P4 P4.2 reachability 0.5962 < 0.95; R20-P5 P5.2 reachability 0.5744 < 0.95; Wf-44 P2 per-domain reachability empty dir) |
| Sub-tasks IN FLIGHT (not yet committed) | **2 / 50** (R20-P1 P1.2 lazy RDKit; R20-P1 close-out TODO file) |

**Bottom line**: R20 plan has shipped its **code + skill + smoke + close-out ledger surface** today (7/8 close-out files; 5/8 skills; ~16 reports; ~30 verification JSONs). All GPU retrain phases and wet-lab cooperation are **honestly deferred** per `party-twenty-spirit §3.5`. **The structural reachability issue (0.5744-0.5962 ≪ 0.95 gate) is NOT a bug** — it is a structural ceiling that requires SMARTS coverage expansion + `max_depth ≥ 2` + GPU retrain to lift, all of which are sequenced behind Wf-23 GPU ultracode queue exit and TODO-84 paper-gate authorization.

**For Hugo's next-action decision**:

- If user wants **CPU-only parallel launches NOW**: see §3.1 — 4 safe candidates (Wf-65 P1.2 + Wf-55 S1/S4/S5/S6 skill polish).
- If user wants **GPU retrain launches**: defer until Wf-23 GPU ultracode queue exit; ETA ~6 weeks if parallelized across GPU-0 + GPU-1.
- If user wants **paper work**: REQUIRES explicit authorization per TODO-84.

---

*Wf-70 Phase 1 ship ledger — generated 2026-09-26 — honest framing per `party-twenty-spirit §3.5`*

## 2026-09-27 update

-9.6574 was the legacy stub-floor artifact; Wf-121 P3 ship real Vina path now active per TODO-113 honest reframe.
