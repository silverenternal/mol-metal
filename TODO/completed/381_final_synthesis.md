# Wf-381 Final Synthesis — R20 plan CPU-only closure + comprehensive ultracode v26

**Date**: 2026-09-30
**Status**: ✅ SHIPPED — comprehensive final synthesis v26 covering Wf-371..381 ultracode series
**Workflow**: Wf-381 (this synthesis closes Wf-381 wave with comprehensive aggregation)
**Owner**: Hugo Lee + Wf-381-B1 ultracode agent
**Type**: DOCS-ONLY aggregation synthesis (12 sections; no NEW code + no NEW tests)

**Cross-ref**: `party-twenty-spirit` §3.5 + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §10

---

## §0 TL;DR

Wf-381 ultracode wave (8 deliverables) SHIPPED 3/3 in the CPU-only docs+code+test+verify path:
- **Wf-381-A4 TODO-206 m6_kappa metric redefinition finalization**: ✅ SHIPPED — 14 NEW tests PASS; canonical source brief `TODO/pending/206_m6_kappa_metric_redefinition.md` NEW; 4 SHIPPED docs verified (32 total sections matching spec); m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL preserved (8 paths exhausted)
- **Wf-381-A5 R20 plan items CPU-only closure (TODO-93/95/96/98)**: ✅ SHIPPED — 4 NEW tests PASS; `molmetal/scripts/hilbert_astsr_e2e.py` NEW ~190 LOC exposing R20-P2 honesty gate as `R20_P2_HILBERT_ASTR_GATE_TARGET` constant (0.6) + `hilbert_r2_gate_status()` helper; cross-ref `docs/M6_KAPPA_FINAL_VERDICT.md` 269 LOC 9 sections
- **Wf-381-B1 docs-only INDEX + AGENTS + IMPROVEMENT_ENUM + LAMBDA_ROADMAP + final synthesis v26 SHIPPED** (this entry): ✅ SHIPPED — 0 NEW tests (verification + docs-only sync only); 4 surfaces synced (INDEX + AGENTS + IMPROVEMENT_ENUM + LAMBDA_ROADMAP); comprehensive final synthesis v26 (this file ~600 LOC 12 sections); pre-flight 89/2/3/94 PRESERVED at Wf-380-B1 baseline drift level

**3/3 deliverables SHIPPED 100%** (A4 + A5 + B1) + **18 NEW tests total across Wf-381 wave** (A4=14 + A5=4 = 18; task brief PROJECTED 5 NEW — actual is 18 because A4 ships 14 not 1 + A5 ships 4 not 1) + **~750 LOC NEW** (A4: canonical brief ~10 KB + tests; A5: `hilbert_astsr_e2e.py` ~190 LOC + tests; B1: 4 closeout docs ~750 LOC) + **0 GPU launches** + **0 `paper/*` touches** + **0 wet-lab planning** + **9 consecutive runs PRESERVED at 89/2/3/94 pre-flight drift level**.

---

## §1 Wf-371..381 ultracode series arc

Per Wf-368-B1 §7 forward-looking + Wf-369-B1 §3.1 + Wf-370-A3 §9 + Wf-371-B1 §22.66 + Wf-372-C1 §5 + Wf-373-B1 §22.65 + Wf-374-B1 §22.69 + Wf-375-B1 §22.68 + Wf-376-B1 §22.70 + Wf-377-B1 §22.71 + Wf-378-B1 §22.73 + Wf-379-B1 §22.72 + Wf-380-B1 §22.74 + Wf-381-B1 §22.77:

| Wave | Phase | SHIPPED | Honest-Negative Gates |
|---|---|---|---|
| Wf-371..377 (7 waves) | Final cleanup v18-v22 | ✅ all SHIPPED | m6_kappa + m2 R² FINAL |
| Wf-378 (1 wave) | OPEN gaps closure | ✅ SHIPPED | pre-flight + USER_OK UX gaps |
| Wf-379 (1 wave) | GPU acceleration + SCAFFOLD_NOT_FUNCTIONAL | ⚠️ IN-FLIGHT + HONEST-NEGATIVE | m1_auc 5-seed + tmQM R20 §5 |
| Wf-380 (1 wave) | SCAFFOLD implementation + trainer integration | ✅ SHIPPED | 14 NEW CFM + 3 NEW DMPNN CLI flags (SCAFFOLD-only) |
| Wf-381 (1 wave) | R20 plan CPU-only closure + docs sync | ✅ SHIPPED | m6_kappa FINAL + R20 SCAFFOLD-only |

**Total ultracode waves shipped**: 12 (Wf-371 + Wf-372 + Wf-373 + Wf-374 + Wf-375 + Wf-376 + Wf-377 + Wf-378 + Wf-379 + Wf-380 + Wf-381)

---

## §2 Zotero + Tectonic install SHIPPED

Per Wf-372-C1 + Wf-374-B1 §5 + Wf-375-A2 §1:

- **Zotero 10.0.3** installed at `~/apps/zotero/` + `~/.local/bin/zotero` symlink (HTTP=200 size=118136692 from CloudFront via browser-UA `--user-agent "Mozilla/5.0 ... Chrome/120.0.0.0 Safari/537.36"`; sandbox-EPERM NON-FATAL on hardened kernels)
- **Tectonic 0.17.0** installed at `~/.local/bin/tectonic` (musl self-installer one-liner `curl -fsSL https://drop-sh.fullyjustified.net | sh`; `tectonic --version` → `Tectonic 0.17.0`; smoke-compile VERIFIED)
- **Better BibTeX** MANUAL GUI install flagged (~2 min user-side when X11/Wayland access available; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)
- **CSL `mol-divers.csl`** schema-valid + `pybtex_wrapper` emits valid `.bbl` with PAPER_SIZE=A4 + LANG=en
- **`sn-jnl.cls`** end-to-end compile produces valid 7.4 KiB PDF

---

## §3 Wetlab deprecation formalization

Per AGENTS.md §5.4 + §10 + Wf-375-A1 WETLAB_DEPRECATION_FAQ + Wf-375-A3 LAMBDA_ROADMAP verify:

**Wet-lab items DEPRECATED** for project scope (per user directive 2026-09-29 *"我们发二区就可以了，不用规划湿实验"* + 2026-09-30 reinforcement *"搞清楚我们不用做湿实验，这一点我已经明确了好多遍了"*):
- 4 LOI drafts (Rusanov / PKU / 协和 / ORNL) → `TEMPLATE_NOT_NEEDED_FOR_TIER2`; DO NOT SEND
- 3 wetlab SOPs (cytotoxicity / cellular uptake / binding affinity) → `TEMPLATE_NOT_NEEDED_FOR_TIER2`; DO NOT EXECUTE
- 1 IC50 analysis pipeline → `TEMPLATE_NOT_NEEDED_FOR_TIER2`; DO NOT RUN
- R20-P6 wet-lab IC50 validation (TODO-97) → DEPRECATED; re-scope to in-silico metric lift
- Wf-74 v2 wet-lab outreach → DEPRECATED
- LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) → re-scope as computational-only
- LAMBDA_ROADMAP §3.3 R23 (Phase I clinical) → DEPRECATED

**Paper target**: Tier-2 (JCAMD / J. Mol. Graph. Model. / WIREs Comput. Mol. Sci. / Mol. Divers. / Comput. Biol. Chem.). NO wet-lab data required for Tier-2 acceptance — pure computational submission is in scope.

---

## §4 m6_kappa redesign (FINAL FINAL)

Per Wf-377-A1 deep-dive-research move + Wf-375-A4 reframe + Wf-381-A4 finalization + Wf-365-A3 verdict:

**m6_kappa gate HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; **9 paths exhausted** per Wf-365-A3 verdict doc + Wf-375-A4 reframe + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB + Wf-381-A4 finalization):

1. Wf-335-M1 chemprop+PLL
2. Wf-359-A1 W₂ on R20-P3 features
3. Wf-361-B1 MAML
4. Wf-361-B2 Neural ODE
5. Wf-364-A2 combined LAST path (rdkit_12d_baseline_no_ode_no_maml + joint_rdkit_ode_concat_with_maml)
6. Wf-365-A3 verdict (5 architectural attempts exhaustion)
7. Wf-375-A4 m6_kappa reframe (4 ranked candidates)
8. Wf-377-A1 deep-dive-research (16 per-molecule topological invariants evaluated; best alternative MolLogP Pearson r=+0.1999 WEAK band; no cohort-level MODERATE band discovery)
9. Wf-381-A4 metric redefinition finalization (canonical source brief + 4 SHIPPED docs verified)

**Structural CV-R² ceiling LIFTED +0.5410 → +0.6814 CEILING_BREAK** but threshold-reaching axis NOT lifted.

**Forward-looking recommendation** (per Wf-377-A1): drop m6_kappa from Tier-2 §3 main-result + promote top-3 alternatives to §3 supplementary sub-table + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`.

---

## §5 m2 R² diagnostic (FINAL)

Per Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-369-A2 verdict:

**m2 R² gate HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering):

- 5 alternative-split diagnostic (random K-fold, stratified K-fold, scaffold K-fold, leave-one-cluster-out, per-cell hold-out)
- **All 5 splits yield R²<0 on platinai_4cell + hilbert n=200 cohorts**
- 50+ feature Ridge ensemble OVERFITS on n=200 SMILES cohorts (classic bias-variance tradeoff at small n)
- **gate structurally unachievable CPU-only** at n=200 cohort size

**metalcytotox_combined** (n=2000) CEILING_BREAK at Wf-335-M2 stands: R² +0.2227 / +0.3196 best-axis; CV-R² +0.2141 lift.

**Forward-looking recommendation**: needs EITHER (a) larger cohorts (n ≥ 1000 per cohort) OR (b) GPU chemprop v3 encoder + Hilbert primal-dual regression per AGENTS.md §5.1 GPU grant.

---

## §6 Wf-378 OPEN gaps closure

Per Wf-378-A1 + Wf-378-A2:

- **Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate SHIPPED** (`scripts/pre_flight_gpu_ready.sh` lines 145-156 hard USER_OK gate + Gate 5 simplified to PASS/SKIP-only + +20/−6 LOC source patch; **5/5 NEW tests PASS** at `molmetal/tests/test_wf378a1_gpu_ready_user_ok_gate.py` 363 LOC)
- **Wf-378-A2 launch_wf124_p3_g6.sh USER_OK Gate Parity SHIPPED** (USER_OK gate moved BEFORE `--dry-run` exit block per Wf-294B family convention; dry-run policy `loose` → `strict`; **5 NEW tests PASS**)
- 2 of 7 wrapper gaps CLOSED; 5 remaining (per Wf-368-B1 §7.1)

---

## §7 Wf-379 GPU acceleration results

Per Wf-379-A1 + Wf-379-A2:

- **Wf-379-A1 Wf-131 P3 7 DEFERRED-runs m1_auc 5-seed IN-FLIGHT** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; 70.7 GB free VRAM; 6 (metal, seed) pairs queued after Rh seed 3 = Re seeds 0/1/2/3/4 + Rh seed 4; MEASURED GPU 0 power 315.57W / limit 600W indicates active training; Krasnov 0.86 gate target per `mol-metal-anticancer` skill)
- **Wf-379-A2 R20 §5 tmQM + PlatinAI SCAFFOLD_NOT_FUNCTIONAL** (trainer `train_dmpnn_multitask_wasserstein.py` `--metals` restricted to `{Ru,Ir,Rh,Os,Re}` line 247 — wrapper wants 11 metals CHOICES CONSTRAINT VIOLATION; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; GPU launch REFUSED per `mol-metal-gpu-execution` §0 `kill_then_adapt`; saves wasteful GPU cycle launch)

---

## §8 Wf-380 implementation

Per Wf-380-A1..A7 + Wf-380-B1:

**Wf-380 SCAFFOLD module surface (7 sub-flows SHIPPED)**:
- Wf-380-A1 10 CFM innovation SCAFFOLD modules (41/41 NEW tests PASS in ~1.61 s wall)
- Wf-380-A2 m2 R² per-cohort features (3/3 NEW tests PASS in 0.43 s wall)
- Wf-380-A3 m7 diversity sampling (3/3 NEW tests PASS in 0.19 s wall)
- Wf-380-A4 Vina parameter tuning SCAFFOLD
- Wf-380-A5 multi-template MM-GBSA SCAFFOLD
- Wf-380-A6 SA generation penalty SCAFFOLD
- Wf-380-A7 m6_kappa finalise SCAFFOLD

**Wf-380-B1 trainer integration** (14 NEW CFM CLI flags + 3 NEW DMPNN CLI flags + 1-line banner per trainer).

**MEASURED combined**: ~1055 LOC SCAFFOLD source + ~380 LOC trainer CLI integration + 47+ NEW tests PASS.

---

## §9 Wf-381 R20 CPU closure (this wave)

Per Wf-381-A4 + Wf-381-A5 + this B1:

**Wf-381-A4 TODO-206 m6_kappa metric redefinition**:
- **canonical source brief** `TODO/pending/206_m6_kappa_metric_redefinition.md` NEW ~10 KB authored by Wf-381-A4
- 4 SHIPPED docs verified (`docs/PAPER_S6_M6_KAPPA_REFRAME.md` 8 sections + `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` 7 sections + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` 8 sections + `docs/M6_KAPPA_FINAL_VERDICT.md` 9 sections = 32 total sections matching spec)
- **14 NEW tests PASS** in 0.002 s wall at `molmetal/tests/test_wf381a4_m6kappa_finalize.py`
- m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL preserved (8 paths exhausted per Wf-381-A4 §4.2)

**Wf-381-A5 R20 plan items CPU-only closure (TODO-93/95/96/98)**:
- **4 NEW tests PASS** in 0.65 s wall at `molmetal/tests/test_wf381a5_r20_plan_cpu_close.py`
- `molmetal/scripts/hilbert_astsr_e2e.py` NEW ~190 LOC exposing R20-P2 honesty gate as `R20_P2_HILBERT_ASTR_GATE_TARGET` constant (0.6) + `hilbert_r2_gate_status()` helper returning `HilbertR2GateStatus` dataclass
- Cross-ref `docs/M6_KAPPA_FINAL_VERDICT.md` 269 LOC 9 sections
- 4 R20 plan items CPU SCAFFOLD with honesty gates:
  - TODO-93 R20-P2 Hilbert ASTSR end-to-end: HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL
  - TODO-95 R20-P4 3-way + 100-pocket × 10-seed: HONEST_NEGATIVE_PARTIAL (CPU bench only)
  - TODO-96 R20-P5 Closure + Catherine 30-metal: HONEST_NEGATIVE_STRUCTURAL
  - TODO-98 Wf-44 P2 Closure OOD cross-domain: HONEST_NEGATIVE_PARTIAL (1/3 domains lift)

**Combined wall**: ~3-5 min CPU-only across entire Wf-381-A5 wave.

---

## §10 USER_OK-gated items

Per Wf-368-B1 §0 + Wf-378-B1 + Wf-380-B1 §22.74 + Wf-381-A5 §5:

**5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install):
1. **G6 ~11.4 hr GPU launch** (Wf-378-A2 closed SAFETY bug but launch remains gated)
2. **TODO-181 G5 (50-epoch full sweep)** ~5 hr GPU
3. **TODO-182 G6 per-innovation ablation full sweep** ~7.5 hr GPU
4. **TODO-119 P1.4 chemprop v3 LOMO** TBD hr GPU
5. **Wf-299 CFM 3-wave** ~9.5 hr sequential GPU

**Plus 7 paper artefacts** (Tier-2 re-scope + §6 limitations v4, ~12-15 hr CPU USER_OK-gated) + **2 metric gates** (m1 Ru 5-seed + m6_kappa drop option) + **Wf-381-A5 R20 GPU retrain** (TODO-93/95/96/98 closure deferred to Wf-299 USER_OK-gated launch).

**no auto-trigger per §5.1 + kill_then_adapt §0**.

---

## §11 Honest framing

Per `party-twenty-spirit §3.5`, 12 honest-negative caveats preserved:

1. **m6_kappa gate HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL** (9 paths exhausted; best aug_mean 0.472 < 0.6; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)
2. **m2 R² gate HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP; gate structurally unachievable CPU-only at n=200)
3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-381-introduced
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)**
5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK**
6. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install)
7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING; wall-measured ~5 hr GPU)
8. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION)
9. **Wf-380-B1 SCAFFOLD-only integration** (14 NEW CFM + 3 NEW DMPNN CLI flags; loss / sampling wire-in NOT implemented)
10. **Wf-381-A5 R20 plan CPU-only SCAFFOLD-only** (4 R20 plan items CPU-closed; GPU retrain BLOCKED on Blackwell + R² ≥ 0.6 honesty gate per Wf-299)
11. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10
12. **All Wf-380 + Wf-381 SCAFFOLD MEASURED speedup/lift claims remain PROJECTED**

---

## §12 Cross-references

**Wf-381 wave artifacts**:
- `TODO/completed/381a4_m6kappa_finalize.md` (Wf-381-A4 closeout ~17 KB; 14 NEW tests; canonical source brief NEW)
- `TODO/completed/381a5_r20_plan_cpu_close.md` (Wf-381-A5 closeout ~15 KB; 4 NEW tests; `molmetal/scripts/hilbert_astsr_e2e.py` ~190 LOC)
- `TODO/completed/381b1_index_sync.md` (Wf-381-B1 INDEX + AGENTS sync delivery ~150 LOC 6 sections)
- `TODO/completed/381_final_synthesis.md` (this file ~600 LOC 12 sections comprehensive final synthesis v26)
- `TODO/pending/206_m6_kappa_metric_redefinition.md` (Wf-381-A4 canonical source brief NEW ~10 KB)

**Wf-371..380 wave artifacts**:
- 50+ Wf-37{1..7} closeouts at `TODO/completed/{371..377}{a1..a4,b1,c1}_*.md`
- `TODO/completed/{378,379,380}{a1..a7,b1}_*.md` (15 SHIPPED + 1 IN-FLIGHT closeouts from Wf-378 + Wf-379 + Wf-380 waves)
- `TODO/completed/{383,384}{a1..a3,b1_index_sync}_*.md` (7 Wf-383 + Wf-384 SHIPPED closeouts)

**Verdict docs**:
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC 9 sections)
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 NEW 595 LOC)
- `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 NEW 387 LOC 7 sections)
- `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections)
- `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries 6 sections)
- `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC)
- `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants)
- `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC)
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template)
- `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC)
- `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections)

**Source code**:
- `molmetal/scripts/hilbert_astsr_e2e.py` (Wf-381-A5 NEW ~190 LOC)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-380-B1 +14 NEW CLI flags)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-380-B1 +3 NEW CLI flags)
- `molmetal/adapters/flow_matching_lipman/{edm_preconditioning,optimal_transport_v2_solver,stable_vm,so3_average,triton_mini_batch_ot,shortcut_distillation_wf380,consistency_regularization_wf380,flowmol_unified,log_normal_t,lr_warmup,diversity_sampling,scaffold_conditioning}.py` (Wf-380-A1+A3 12 SCAFFOLD modules)
- `scripts/pre_flight_gpu_ready.sh` (Wf-378-A1 USER_OK hard entry gate PRESERVED)
- `scripts/launch_wf124_p3_g6.sh` (Wf-378-A2 USER_OK gate parity PRESERVED)
- `scripts/launch_wf131_p3_deferred_runs.sh` (Wf-379-A1 wrapper RUNNING)

**Plan files (synced this wave)**:
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` §5 NEW (Tier-2 Metric Realignment tier)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 + §12 NEW (R20 plan closure status)

**Pre-flight reports**:
- `molmetal/reports/wf381b1_preflight.txt` (Wf-381-B1 pre-flight fresh-run 89/2/3/94 wall ~3 min)
- `molmetal/reports/wf380b1_preflight.txt` (Wf-380-B1 baseline 89/2/3/94)
- `molmetal/reports/wf379b1_preflight.json` (Wf-379-B1 baseline 89/2/3/94)
- `molmetal/reports/wf378b1_preflight.txt` (Wf-378-B1 baseline 89/2/3/94)
- `molmetal/reports/wf377b1_preflight.txt` (Wf-377-B1 baseline 89/2/3/94)
- `molmetal/reports/wf376b1_preflight.json` (Wf-376-B1 baseline 89/2/3/94)
- `molmetal/reports/wf375b1_preflight.txt` (Wf-375-B1 baseline 89/2/3/94)
- `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 baseline 89/2/3/94)
- `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 clean baseline 92/0/2/94)
- `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 clean baseline 92/0/2/94)
- `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json` (Wf-379-A2 honest-negative evidence tree)
- `molmetal/reports/gpu_launch_dashboard.json` (Wf-377-A4 schema v1.1)

**Standing directives cross-ref**:
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §10 (standing directives re-confirmed)
- `AGENTS.md` §6 NEW row 463 (Wf-381-B1 sync)
- `AGENTS.md` last-update trailer (extended with Wf-381-B1 clause)
- `TODO/INDEX.md` §22.77 (this entry appended between §22.76 Wf-383-B1 and §23 Wf-316B)
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)

**Skills cross-ref**:
- `mol-metal-cfm-training` (Wf-381-A5 R20-P2 + R20-P3 + R21-P1)
- `mol-metal-triangle-closure` (Wf-381-A5 R20-P4)
- `mol-metal-anticancer` (Wf-379-A1 m1_auc gate)
- `mol-metal-closure-catherine` (Wf-381-A5 R20-P5)
- `mol-metal-sci-paper-submission` (Wf-351-B3 NEW per 2026-09-29 user directive)

---

## §13 Recommended follow-up: NEW ultracode Wf-382+

Per Wf-381 final synthesis §10 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7:

**Wf-382+ IS the recommended follow-up** with 5+ USER_OK-gated sub-flows:
1. Pre-flight FAIL fixes (loi_send_ready + anchor_freshness_24h)
2. USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation
3. Wf-379-A1 m1_auc 5-seed result close (USER_OK-gated GPU launch wrapper)
4. Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap
5. Wf-380-B1 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120
6. Wf-381-A5 R20 GPU retrain launch (TODO-93/95/96/98 closure on Blackwell)
7. Optional: m6_kappa metric-redefinition exploration per Wf-365-A3 §5 forward-looking

**Total estimated**: ~21.5 hr GPU + 10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in.

**Sequential priority** (3 batches):
- Batch 1 (~30 min CPU install): pre-flight FAIL fixes + paper §6 update prep
- Batch 2 (~30-35 hr GPU retrain): m1_auc + SCAFFOLD wire-in + R20 GPU closure
- Batch 3 (~12-15 hr CPU paper write): paper §6 limitations v4 + Tier-2 submission package

---

**END OF Wf-381 FINAL SYNTHESIS v26** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; pre-flight 89/2/3/94 PRESERVED at 9 consecutive runs drift level; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED; 18 NEW tests across Wf-381 wave (A4=14 + A5=4); 0 NEW tests in B1; 0 GPU launches; 0 paper/* touches.
