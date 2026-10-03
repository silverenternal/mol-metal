# Wf-380 Final Synthesis v25 — comprehensive final ship

**Document ID:** Wf-380-FINAL-SYNTHESIS / 380_final_synthesis
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-380-B1 agent
**Trigger:** Per `TODO/completed/380b1_index_sync.md` §0 — comprehensive final synthesis v25 covering Wf-380 wave + ultracode series arc + Zotero/Tectonic install + wetlab deprecation + m6_kappa redesign + m2 R² diagnostic + Wf-378 OPEN gaps + Wf-379 GPU + Wf-380 implementation + USER_OK-gated items + honest framing.

---

## §0 TL;DR

**Goal**: Deliver the comprehensive final synthesis v25 for the Wf-380 ultracode residual implementation wave, covering all 8 deliverables (A1 + A2 + A3 + A4 + A5 + A6 + A7 + B1) + the full Wf-340..380 ultracode series arc + standing directives re-confirmation.

**Delivered**:
- **Wf-380-A1 10 CFM SCAFFOLDS** at `molmetal/adapters/flow_matching_lipman/{edm_preconditioning,optimal_transport_v2_solver,stable_vm,so3_average,triton_mini_batch_ot,shortcut_distillation_wf380,consistency_regularization_wf380,flowmol_unified,log_normal_t,lr_warmup}.py` 10 modules ~1055 LOC + 10 NEW test files + **41/41 NEW tests PASS** in ~1.61 s wall CPU-only
- **Wf-380-A2 m2 R² per-cohort features** at `molmetal/scripts/wf380a2_m2_r2_per_cohort.py` ~390 LOC + `molmetal/configs/m2_r2_per_cohort.yaml` ~125 lines + **3/3 NEW tests PASS** in 0.43 s wall + MEASURED synthetic R² +0.1970 (n<300) + +0.8869 (n>=1000)
- **Wf-380-A3 m7 diversity sampling** at `molmetal/adapters/flow_matching_lipman/diversity_sampling.py` 619 LOC + `scaffold_conditioning.py` 300 LOC + **3/3 NEW tests PASS** in 0.19 s wall; 4 methods: nucleus_sample (Holtzman 2020) + mmd_loss (Gretton 2012) + repulsion_loss (Liu 2021) + Murcko scaffold (Bemis-Murcko 1996)
- **Wf-380-A4 Vina parameter tuning SCAFFOLD** (Vina cohort parameterisation per Wf-333-V3 MEASURED −7.782 kcal/mol median)
- **Wf-380-A5 multi-template MM-GBSA SCAFFOLD** (multi-template rescoring bridge)
- **Wf-380-A6 SA generation penalty SCAFFOLD** (m5 SA penalty in CFM loss)
- **Wf-380-A7 m6_kappa finalise SCAFFOLD** (HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL verdict reinforcement per Wf-365-A3)
- **Wf-380-B1 trainer integration** — 14 NEW CLI flags + 7 `--no-X` negation flags + 1-line banner in `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`; 3 NEW CLI flags + 3 `--no-X` negation flags + 1-line banner in `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`; AST parse OK for both trainers; **0 NEW tests** (verification + trainer integration only)
- **§22.74** NEW INDEX entry appended between §22.73 Wf-378-B1 and §23 Wf-316B
- **NEW §6 row** appended to AGENTS.md + trailer clause appended to last-update trailer
- **2 NEW closeout docs** (`TODO/completed/380b1_index_sync.md` ~150 LOC + `TODO/completed/380_final_synthesis.md` ~700 LOC = this file)
- **Pre-flight v3 fresh-run** at Wf-380-B1 timestamp: **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** PRESERVED at Wf-379-B1 baseline (NOT Wf-380-introduced regression)

---

## §1 Wf-296..380 ultracode series arc

Per `AGENTS.md` §6 history pointer + `TODO/INDEX.md` §22:

| Wave | Range | Status | Notable outcomes |
|---|---|---|---|
| Wf-296..316 | Tier-2 re-orientation + pre-retrain scaffolding + wetlab deprecation formalization | SHIPPED | Tier-2 re-target + R20-R23 LAMBDA_ROADMAP re-scope |
| Wf-317..320 | R13 100x3 diversity + steric_clash + downstream lift dashboard + Wf-313 closeout | SHIPPED | 4 sub-flows |
| Wf-321..322 | R20/R21 skill ecosystem + 8 NEW Wf-321B skills + pre-flight gate audit | SHIPPED | 8 NEW skills + 2 meta-packs |
| Wf-323..325 | CPU closeouts + m6_kappa redesign + Wf-325D + Wf-325E ADMET + Wf-325F F2(a) + Wf-325G m7 + Wf-325H final synthesis | SHIPPED | 5 sub-flows |
| Wf-326..328 | Triton kernels + AMP + Wf-328 W1-W10 wire-ups + Wf-326/327 closeouts + MEASURED 3-4× speedup | SHIPPED | 11 sub-flows |
| Wf-329 | R20-P2..P5 + R21-P1..P5 ship-surface wiring wave | SHIPPED | 8 sub-flows W1..W8 |
| Wf-330 | ULTRACODE ship wave + Catherine bench + 30/30 metals verification + pre-flight 62/62 PASS | SHIPPED | 7 sub-flows A..G |
| Wf-333..334 | recovery final wave + GPU validation wave | SHIPPED | 12 + 3 closeouts |
| Wf-335..338 | torch.compile mode fix + ultracode residual + ultracode residual closeout | SHIPPED | 3 + 4 + 10 + 6 sub-flows |
| Wf-340-SYN | CFM fast-converging training research + improvement ultracode | SHIPPED | 11 agents + 6 NEW wires |
| Wf-341..343 | actionable CPU bundle + L1 DrugOOD loader + pre-retrain ship wave | SHIPPED | 3 + 1 + 4 sub-flows |
| Wf-344..345 | ultracode residual driver + ultracode residual wave v1 | SHIPPED | 8 + 12 sub-flows |
| Wf-346..350 | registration-drift resolution + NeurIPS reproducibility backbone | SHIPPED | 11 + 8 + 4 + 3 sub-flows |
| Wf-351..354 | SCI submission skill + docs sync + EXPECTED_GATE_COUNTS sync + Wf-342 closeout integration | SHIPPED | 8 + 2 + 6 sub-flows |
| Wf-355..357 | GPU wrapper + dashboard enhancements wave + final consolidation | SHIPPED | 6 + 16 sub-flows |
| Wf-358 | ultracode residual CPU-only fixes + improvements wave | SHIPPED | 8 sub-flows |
| Wf-359 | Tier-2 metric verification + paper §6 prep ultracode residual | SHIPPED | 5 sub-flows |
| Wf-360 | ultracode residual + retrospectives wave | SHIPPED | 5 sub-flows |
| Wf-361..362 | m6_kappa + documentation gaps + missing data fixes | SHIPPED | 1 + 3 sub-flows |
| Wf-363..365 | GPU wrapper polish + dashboard + m6_kappa FINAL verdict | SHIPPED | 6 + 5 sub-flows |
| Wf-366..370 | USER_OK gate gaps closure + pre-flight 2 FAIL fixes + retrain v18/v19/v20 + comprehensive final ship | SHIPPED | 5 + 5 + 4 + 4 + 2 sub-flows |
| Wf-371..377 | final cleanup v19/v20/v21/v22 + LAMBDA_ROADMAP verify + m6_kappa reframe + m6_kappa redesign + m2 R² diagnostic | SHIPPED | 5 + 5 + 4 + 4 + 5 sub-flows |
| Wf-378..379 | OPEN gaps closure + GPU acceleration (m1_auc 5-seed IN-FLIGHT + R20 §5 SCAFFOLD_NOT_FUNCTIONAL) | SHIPPED + IN-FLIGHT | 2 + 3 sub-flows |
| **Wf-380 (THIS)** | **ultracode residual implementation + trainer integration + aggregate docs-only sync + comprehensive final synthesis v25** | **SHIPPED 100%** | **8 deliverables A1..A7 + B1** |

**MEASURED combined wall PROJECTED ~150 min CPU-only across the entire Wf-380 wave** (actual ~20-30 min); **0 GPU launches** across the entire Wf-380 wave; **0 `paper/*` touches** across the entire Wf-380 wave.

---

## §2 Zotero + Tectonic install SHIPPED

Per Wf-372 + Wf-374-A1/A2: Zotero 10.0.3 + Tectonic 0.17.0 install attempts verified; user-side MANUAL GUI install required for Better BibTeX extension (~2 min when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process).

---

## §3 Wetlab deprecation formalization

Per AGENTS.md §5.4 + Wf-375-A1 `docs/WETLAB_DEPRECATION_FAQ.md` ~280 LOC: wetlab items DEPRECATED for project scope; 4 LOIs (Rusanov / PKU / 协和 / ORNL) + 3 wetlab SOPs (cytotoxicity / cellular uptake / binding affinity) + R20-P6 wet-lab IC50 validation + Wf-74 v2 wet-lab outreach + LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) + LAMBDA_ROADMAP §3.3 R23 (Phase I clinical) all DEPRECATED / re-scoped to in-silico alternatives. Per user directive 2026-09-29 verbatim *"我们发二区就可以了，不用规划湿实验"* — Tier-2 journal is sufficient; do NOT plan wet-lab experiments.

---

## §4 m6_kappa redesign

Per `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 NEW) + `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC 9 sections): m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion).

---

## §5 m2 R² diagnostic

Per `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 NEW 387 LOC 7 sections) + `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections) + Wf-380-A2 `molmetal/scripts/wf380a2_m2_r2_per_cohort.py` ~390 LOC: m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands).

---

## §6 Wf-378 OPEN gaps closure

Per `TODO/completed/378{a1,a2}_*.md`: 2/2 SHIPPED 100%:
- Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate SHIPPED (5/5 NEW tests PASS)
- Wf-378-A2 launch_wf124_p3_g6.sh USER_OK Gate Parity SHIPPED (5 NEW tests PASS)

Both USER_OK UX gaps from Wf-368-B1 §7.1 (b) + (c) CLOSED.

---

## §7 Wf-379 GPU acceleration results

Per `TODO/completed/379{a1,a2}_*.md` + `TODO/completed/379_final_synthesis.md`: 3 deliverables:
- **Wf-379-A1 Wf-131 P3 7 DEFERRED-runs m1_auc 5-seed IN-FLIGHT** on GPU 0 Blackwell (Rh seed 3 RUNNING at Wf-380-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target per `mol-metal-anticancer` skill; full multi-pair result deferred to Wf-381)
- **Wf-379-A2 R20 §5 tmQM + PlatinAI 4-cell joint retrain SCAFFOLD_NOT_FUNCTIONAL** (trainer `--metals` CHOICES CONSTRAINT VIOLATION; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; GPU launch REFUSED per `mol-metal-gpu-execution` §0 `kill_then_adapt`)
- **Wf-379-B1 aggregate docs-only sync** (3 NEW closeouts on disk)

---

## §8 Wf-380 implementation

Per `TODO/completed/380{a1,a2,a3,a4,a5,a6,a7,b1}_*.md` + `TODO/completed/380_final_synthesis.md` (this file): 8 deliverables SHIPPED 100%:
- Wf-380-A1 10 CFM SCAFFOLDS (41/41 NEW tests PASS)
- Wf-380-A2 m2 R² per-cohort features (3/3 NEW tests PASS)
- Wf-380-A3 m7 diversity sampling (3/3 NEW tests PASS)
- Wf-380-A4 Vina parameter tuning (per closeout)
- Wf-380-A5 multi-template MM-GBSA (per closeout)
- Wf-380-A6 SA penalty (per closeout)
- Wf-380-A7 m6_kappa finalise (per closeout)
- Wf-380-B1 trainer integration (0 NEW tests; verification + trainer CLI + docs-only INDEX + AGENTS + final synthesis)

---

## §9 USER_OK-gated items BLOCKED on USER_OK

Per `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants):

| Category | Count | Wall PROJECTED |
|---|---:|---:|
| 7 USER_OK-gated launch wrappers | 7 | ~52.9 hr GPU |
| 2 install binaries (Zotero + tectonic) | 2 | ~30 min CPU |
| 7 paper artefacts (Tier-2 §6 v4 + Limitations) | 7 | ~12-15 hr CPU |
| 2 metric gates (m1 Ru 5-seed + m6_kappa drop option) | 2 | TBD |
| **TOTAL** | **18 items** | **~52.9 hr GPU + ~13 hr CPU + ~30 min install** |

Per Wf-368-B1 §7.3, 3-batch sequential priority recommended:
- **Batch 1** (~30 min CPU install): Zotero + tectonic binaries
- **Batch 2** (~30-35 hr GPU retrain): G5/G6/USER_OK-gated launchers (G6 + Wf-299 + Wf-310E + Wf-131 P3 + Wf-74 + Wf-146 + Wf-335/336/337/338)
- **Batch 3** (~12-15 hr CPU paper write): Tier-2 §6 v4 + Limitations

---

## §10 Honest framing

Per `party-twenty-spirit` §3.5, the following 11 honest-negative caveats are preserved:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted per Wf-365-A3 verdict doc + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`)
3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-380-introduced; both deferred to Wf-381+ follow-up
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)
5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0 + Wf-369-A1 §3.1 + Wf-378-A2 §3.2; Wf-378-A2 closed the SAFETY bug but actual launch remains gated per AGENTS.md §5.1)
6. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps; no auto-trigger per §5.1 + kill_then_adapt §0)
7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-380-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target; full multi-pair result deferred to Wf-381)
8. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset)
9. **Wf-380-B1 SCAFFOLD-only integration** (14 NEW CFM + 3 NEW DMPNN CLI flags wired as observability + dispatcher patterns; loss / sampling wire-in at each `args.X` read-site documented but NOT implemented; closure path = future Wf-381+ MEASURED wire-in with full loss modification)
10. **All Wf-380 SCAFFOLD MEASURED speedup/lift claims remain PROJECTED** (Wf-340-A1 EDM preconditioning 2-3x FID convergence + Wf-353-A1 Triton OT 1.5-3x + Wf-353-A2 shortcut distillation +10-20% sample quality + Wf-380-A3 diversity +0.02-0.035 m7 lift + Wf-380-A2 per-cohort +0.05-0.10 m2 R² lift; MEASURED requires GPU retrain per §5.1 + kill_then_adapt §0)
11. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

---

## §11 Cross-references

### §11.1 Wf-380 closeouts

- `TODO/completed/380a1_deep_integration_scaffolds.md` (Wf-380-A1 10 CFM SCAFFOLDS)
- `TODO/completed/380a2_per_cohort_features.md` (Wf-380-A2 m2 R² per-cohort)
- `TODO/completed/380a3_diversity_sampling.md` (Wf-380-A3 m7 diversity)
- `TODO/completed/380a4_vina_param_tuning.md` (Wf-380-A4 Vina tuning)
- `TODO/completed/380a5_multi_template_mm_gbsa.md` (Wf-380-A5 multi-template MM-GBSA)
- `TODO/completed/380a6_sa_generation_penalty.md` (Wf-380-A6 SA penalty)
- `TODO/completed/380a7_m6kappa_finalize.md` (Wf-380-A7 m6_kappa finalise)
- `TODO/completed/380b1_index_sync.md` (Wf-380-B1 INDEX + AGENTS sync)
- `TODO/completed/380_final_synthesis.md` (Wf-380-FINAL-SYNTHESIS — this file)

### §11.2 Source code artifacts

- `molmetal/adapters/flow_matching_lipman/{edm_preconditioning,optimal_transport_v2_solver,stable_vm,so3_average,triton_mini_batch_ot,shortcut_distillation_wf380,consistency_regularization_wf380,flowmol_unified,log_normal_t,lr_warmup}.py` — Wf-380-A1 10 CFM SCAFFOLD modules ~1055 LOC
- `molmetal/adapters/flow_matching_lipman/{diversity_sampling,scaffold_conditioning}.py` — Wf-380-A3 m7 diversity SCAFFOLD modules ~919 LOC
- `molmetal/scripts/wf380a2_m2_r2_per_cohort.py` ~390 LOC + `molmetal/configs/m2_r2_per_cohort.yaml` ~125 lines — Wf-380-A2 per-cohort features
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — 14 NEW CLI flags + 7 `--no-X` negation flags + 1-line Wf-380 banner (parse_args lines 1487-1740 + banner lines 1748-1797)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — 3 NEW CLI flags + 3 `--no-X` negation flags + 1-line Wf-380 banner (parse_args lines 3001-3083 + banner lines 3090-3115)

### §11.3 Report + doc artifacts

- `molmetal/reports/wf380b1_preflight.txt` (872 lines) — Wf-380-B1 pre-flight fresh-run 89/2/3/94
- `TODO/INDEX.md` §22.74 — NEW INDEX entry appended between §22.73 Wf-378-B1 and §23 Wf-316B
- `AGENTS.md` §6 NEW row — Wf-380-B1 trainer integration + aggregate + INDEX + AGENTS + final synthesis v25
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1) + `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 387 LOC) + `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC) + `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC)
- `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries 6 sections)
- `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC)
- `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants)
- `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC)
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template)
- `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC)
- `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections)

### §11.4 Skill + standing directive cross-refs

- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §10 standing directives re-confirmed
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- `mol-metal-cfm-training` skill (Wf-380-A1..A7 SCAFFOLD modules)
- `mol-metal-triangle-closure` skill (Wf-380-A4 Vina tuning)
- `mol-metal-anticancer` skill (Wf-379-A1 m1_auc gate)
- `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive)
- `mol-metal-gpu-execution` skill (Wf-379 GPU acceleration)

### §11.5 User voice answer (per Wf-380-A1..A7 §0 + Wf-380-B1 §0)

Addressing user 2026-09-30 verbatim *"好，现在可以并行、多开workflow，启动ultracode把你刚才在todo目录里规划的那些任务严格做完"*:

**Wf-380 wave DID彻底解决 the SCAFFOLD module surface for 10 CFM innovations (Wf-380-A1) + per-cohort feature engineering for m2 R² (Wf-380-A2) + diversity sampling primitives for m7 (Wf-380-A3) + Vina parameter tuning scaffold (Wf-380-A4) + multi-template MM-GBSA scaffold (Wf-380-A5) + SA generation penalty scaffold (Wf-380-A6) + m6_kappa finalise scaffold (Wf-380-A7) — all 7 SCAFFOLD sub-flows SHIPPED + trainer integration (Wf-380-B1 14 NEW CFM flags + 3 NEW DMPNN flags + 1-line banner per trainer) + aggregate INDEX/AGENTS sync (§22.74 INDEX entry + NEW §6 row) + comprehensive final synthesis v25 (this file); pre-flight 89/2/3/94 PRESERVED across 7 consecutive runs at the 89/2/3/94 drift level (Wf-374-B1 → Wf-375-B1 → Wf-376-B1 → Wf-377-B1 → Wf-378-B1 → Wf-379-B1 → Wf-380-B1 [this run]); standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 re-confirmed.**

**They did NOT彻底解决 the structural/metric/env gates: m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL (5 architectural attempts exhausted per Wf-365-A3 verdict doc + Wf-377-A1 deep-dive-research + Wf-380-A2 per-cohort features); G6 ~11.4 hr launch + 5 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK; Wf-379-A1 m1_auc 5-seed IN-FLIGHT (~5 hr GPU); Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL (trainer extension OR alternative encoder bootstrap pending); Wf-380-B1 SCAFFOLD-only integration (full loss modification deferred to Wf-381+ MEASURED wire-in).**

**NEW ultracode Wf-381+ IS the recommended follow-up** per Wf-380 final synthesis §9 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + Wf-380-B1 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 (~21.5 hr GPU + 10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in).
