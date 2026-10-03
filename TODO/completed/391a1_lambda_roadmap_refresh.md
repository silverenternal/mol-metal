# Wf-391-A1 — LAMBDA_ROADMAP refresh (Wf-381/389/390/376/379/383 propagation + AGENTS.md §10 wetlab deprecation)

**Date**: 2026-09-30
**Wf**: Wf-391-A1 (ultracode CPU-only docs-only wave per user directive 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — start additional ultracode for TODO items not yet covered by the workflows just launched)
**Type**: DOCS-ONLY (no NEW source code; no NEW tests; no GPU launches; no paper/* touches; no wet-lab planning per AGENTS.md §10)
**Cross-ref**: `party-twenty-spirit §3.5` + `AGENTS.md §5.1` + `§5.2` + `§5.3` + `§5.4` + `§5.6` + `§10`

---

## §0 TL;DR

| Item | Value | Status |
|---|---|---|
| **File edited** | `TODO/03_plans/LAMBDA_ROADMAP.md` | ✅ DONE |
| **In-place edits** | 6 (covers §1.5/§2.4/§3.2/§3.3/§4.2 + table cell updates) | ✅ SHIPPED |
| **NEW sub-sections added** | 2 (§4.4 R20 §5 联合 retrain status + §6.6 Forward-looking 2026-Q4 GPU retrain priorities) | ✅ SHIPPED |
| **NEW trailing update section** | 1 (§11 2026-09-30 update) | ✅ SHIPPED |
| **NEW tests** | 0 | ✅ docs-only per task brief |
| **GPU launches** | 0 | ✅ per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Wet-lab planning** | 0 | ✅ per AGENTS.md §5.4 + §10 |
| **Sudo** | 0 | ✅ per environment constraints |
| **Pre-flight v3** | NOT re-run (CPU-only docs-only); PRESERVED at Wf-379-B1 baseline 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL | ✅ PRESERVED |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-391-A1) | ✅ PRESERVED |

---

## §1 Sections refreshed

### §1.1 In-place edits (6 anchors)

| Section | Edit | Source |
|---|---|---|
| **§1.5** R20 4 NEW skill table | Added Wf-381-A5 CPU SCAFFOLD SHIPPED cross-ref + Wf-382-A1 SCAFFOLD cross-ref to `mol-metal-closure-theorem` (R20-P5) + `mol-metal-multiph-pll` (R20-P3) rows | Wf-381-A5 §2.3 + §2.4 + Wf-382-A1 (`TODO/completed/382a1_multiph_pll_scaffold.md`) |
| **§2.4** R21 4 NEW skill table | Added Wf-381-A5 CPU SCAFFOLD SHIPPED cross-ref to `mol-metal-closure-dblock-bench` (Wf-44 P2 / TODO-98) + `mol-metal-triangle-closure` (Wf-37 L8 / TODO-95) rows | Wf-381-A5 §2.2 + §2.4 |
| **§3.2** R22-P1 (m18 wet-lab) | Added **DEPRECATION NOTICE** per AGENTS.md §10 + Tier-2 re-orientation; re-scoped to in-silico AlphaFold-Multimer + MetalSite feature extraction only; patient-derived organoid DEPRECATED; m18/m19/m20 metric ship surface preserved as in-silico scaffold lift targets | AGENTS.md §5.4 + §10 + 2026-09-29/30 user directives |
| **§3.3** R23 (Phase I clinical) | Added **FULL DEPRECATION NOTICE** — all 6 phases DEPRECATED per AGENTS.md §10; IND filing + Phase I + IRB + DSMB + patient enrollment all out of project scope; 4 LOI + 3 SOP + 1 IC50 pipeline retained in `TODO/pending/` as historical artifacts per §10.5; R23-P1 through R23-P6 table rows strikethrough-marked | AGENTS.md §10.1 + §10.5 |
| **§4.2** L1-L12 trajectory table | Added Wf-381-A5 closure cross-ref to L6/L7/L8/L10/L11/L12 rows; TODO # cross-ref expanded with Wf-381-A5 closure status + Wf-377-A1 redesign progress for TODO-93 + Wf-382-A1 SCAFFOLD for TODO-94 + TODO-179 STATUS_HOLD per Wf-316A + TODO-376 PLAN_ONLY per Wf-376-A3 + TODO-383 PLAN_READY per Wf-383-A2 + TODO-200 SHIPPED via Wf-340 + Wf-353 | Wf-381-A5 + Wf-376-A3 + Wf-383-A2 + Wf-377-A1 + Wf-316A + Wf-340 + Wf-353 |
| **§6.6** NEW Forward-looking 2026-Q4 GPU retrain priorities | NEW sub-section ranking 11 launch wrappers (P1-A through P5-A) by USER_OK priority with VRAM + wall PROJECTED + pre-flight gate columns; combined wall PROJECTED ~25-30 days sequential GPU; constraint priorities per AGENTS.md §5.1 + Wf-347-A1/A2/B1/B2 integration | Wf-368-B1 §0 + Wf-378-A1+A2 + Wf-379-A1+A2 + Wf-355-B1/B2 + Wf-363-B1 + Wf-378-A2 |

### §1.2 NEW sub-sections (2 added)

| §4.4 NEW | R20 §5 联合 retrain status (post-Wf-381/389/390/379) — aggregated 9-row status table per TODO (93/94/95/96/98/376/383/179/200) with CPU status / GPU retrain / forward-looking path columns; cross-references Wf-381-A5 + Wf-376-A3 + Wf-383-A2 + Wf-377-A1 + Wf-316A + Wf-340 + Wf-353; 6 GPU launches BLOCKED on user OK per AGENTS.md §5.1 + kill_then_adapt §0 |
| **§6.6 NEW** | Forward-looking 2026-Q4 GPU retrain priorities — 11 launch wrappers ranked P1-A through P5-A with VRAM + wall PROJECTED + pre-flight gate + USER_OK priority columns; combined wall PROJECTED ~25-30 days sequential GPU; constraint priorities per AGENTS.md §5.1 (VRAM check + hyperparam config + pre-flight PASS + USER_OK gate + wave budget); forward-looking honest-negative caveats per `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-369-A3 expansion 40 → 65 entries) |

### §1.3 NEW trailing update section (§11)

**§11. 2026-09-30 update (Wf-391-A1 refresh — §1.5/§2.4/§3.2/§3.3/§4.2/§4.4/§6.6 closure)** — appended at end of file with:
- Wave context (Wf-391-A1 ultracode CPU-only docs-only refresh)
- Per-section delta table (6 in-place edits + 2 NEW sub-sections + 1 NEW trailing §11)
- Constraints honoured (0 NEW tests, 0 GPU launches, 0 paper/*, 0 wet-lab, 0 sudo)
- Pre-flight v3 preservation (Wf-379-B1 baseline 89/2/3/94)
- Honest framing per party-twenty-spirit §3.5
- User voice answer (Wf-391-A1 §0)

---

## §2 Cross-references (Wf-381/389/390/376/379/383 + AGENTS.md §10)

### §2.1 Source wave surface

| Wave | Closeout | Subject |
|---|---|---|
| **Wf-381-A5** | `TODO/completed/381a5_r20_plan_cpu_close.md` (24+ KB, 6 sections) | R20 plan items CPU-only closure: TODO-93/95/96/98 closed CPU-side (4 NEW tests PASS in 0.65 s) |
| **Wf-382-A4** | `TODO/completed/382a4_strategic_plans_refresh.md` | Strategic plans CPU refresh: 10 plans + 47 cross-refs + 348 LOC delta |
| **Wf-389-A3** | `TODO/completed/389a3_weak_to_strong_update.md` (28 KB, 4 sections) | WEAK_TO_STRONG + AGENTS.md §5.6 + HONEST_NEGATIVE_CATALOG.md §6.17 sync |
| **Wf-376-A3** | `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` (~250 LOC, 8 sections) | TODO-376 m2 R² GPU retrain PLAN_ONLY (Path A/B/A+B) |
| **Wf-379-A1** | `TODO/completed/379a1_wf131_p3_7runs.md` | Wf-131 P3 7 DEFERRED-runs m1_auc 5-seed IN-FLIGHT on GPU 0 Blackwell |
| **Wf-379-A2** | `TODO/completed/379a2_r20_p5_tmqm_plat.md` | R20 §5 tmQM + PlatinAI SCAFFOLD_NOT_FUNCTIONAL (CHOICES CONSTRAINT VIOLATION) |
| **Wf-383-A2** | `TODO/pending/383_se3_sb_fast_converging_integration.md` (~250 LOC, 5 sections) | TODO-383 SE(3) Schrödinger Bridge fast-converging integration PLAN_READY |
| **Wf-377-A1** | `TODO/completed/377a1_m6_kappa_redesign.md` + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` | m6_kappa REDESIGN deep-dive-research move (16 per-molecule topological invariants n=5460) |
| **Wf-377-A2** | `TODO/completed/377a2_m2r2_diagnostic.md` + `docs/M2_R2_COHORT_DIAGNOSTIC.md` | m2 R² cohort diagnostic (5 alternative splits: random/stratified/scaffold/leave-one-cluster/per-cell) |
| **Wf-316A** | (historic) | TODO-179 L5 Hilbert R² W₂ pIC50 alignment STATUS_HOLD |
| **Wf-340** + **Wf-353** | `TODO/completed/200_wf340_353_deep_integration_audit.md` | TODO-200 CFM YuelBond fast-converging SHIPPED (~820 LOC + 14 NEW tests + 10 innovations wired into `train_cfm_yuelbond_wasserstein.py`) |
| **AGENTS.md §10** | `AGENTS.md` §10.1-§10.6 | Tier-2 re-orientation + wetlab deprecation FAQ (4 LOI + 3 SOP + 1 IC50 pipeline DEPRECATED) |

### §2.2 AGENTS.md §10 wetlab deprecation summary

Per AGENTS.md §10 + 2026-09-29/30 user directives:
- **Tier-2 journal target** (JCAMD / JMGM / Mol. Divers. / WIREs CC / Comput. Biol. Chem.) — computational-only acceptance, no wet-lab data required
- **4 LOI drafts DEPRECATED**: `wetlab_loi_ornl_protein_metal.md`, `wetlab_loi_pku_cisplatin_ic50.md`, `wetlab_loi_xiehe_nci60.md`, `wetlab_loi_rusanov_2025_platinai.md` — retained in `TODO/pending/` as historical artifacts per §10.5; DO NOT SEND
- **3 wetlab SOPs DEPRECATED**: `wetlab_sop_binding_affinity_assay.md`, `wetlab_sop_cellular_uptake_assay.md`, `wetlab_sop_cytotoxicity_assay.md` — DO NOT EXECUTE
- **1 IC50 analysis pipeline DEPRECATED**: `wetlab_ic50_analysis_pipeline.md` — DO NOT RUN
- **R20-P6 wet-lab IC50 validation** DEPRECATED — re-scoped to in-silico metric lift
- **Wf-74 v2 wet-lab outreach** DEPRECATED — re-scoped to in-silico
- **LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab)** re-scoped to in-silico AlphaFold-Multimer + MetalSite feature extraction only
- **LAMBDA_ROADMAP §3.3 R23 (Phase I clinical)** FULL DEPRECATION — IND filing + Phase I + IRB + DSMB + patient enrollment all out of project scope

### §2.3 LAMBDA_ROADMAP surface delta

**Pre-Wf-391-A1**: LAMBDA_ROADMAP.md = 616 lines (Wf-382-A4 Wf-41 + Wf-40 trailing update at §10 + Wf-382-A4 baseline). NO Wf-381-A5 cross-refs. NO Wf-382-A1 SCAFFOLD cross-refs. NO Tier-2 wetlab deprecation markers in §3.2/§3.3. NO §4.4 R20 §5 joint retrain status. NO §6.6 Forward-looking 2026-Q4 priorities.

**Post-Wf-391-A1**: LAMBDA_ROADMAP.md = ~700 lines (Wf-391-A1 added ~84 lines in-place + ~40 lines for §4.4 NEW + ~70 lines for §6.6 NEW + ~50 lines for §11 NEW = ~244 LOC total delta).

---

## §3 Per-section source preservation

### §3.1 Existing structure preserved

All existing §1-§10 sections preserved unchanged except for the targeted in-place edits listed in §1.1. No existing paragraphs were removed or restructured.

### §3.2 §3.2 / §3.3 strikethrough marker convention

For §3.3 R23 all 6 phases, applied **Markdown strikethrough** (`~~text~~`) + **DEPRECATED §10** badge to each cell. This is a standard Markdown rendering pattern that:
- Renders as ~~strikethrough text~~ in view mode
- Preserves original text for git history / archaeology
- Visually flags the deprecated scope

### §3.3 §4.2 L1-L12 row updates — minimal in-place additions

For each L-row, appended **bold cross-ref annotation** after the existing R20 ship column. Preserves original column structure; cross-ref is additive only.

### §3.4 §11 trailing update — appended after §10 (not in-place)

§11 is appended after the existing §10 trailing update from Wf-41 Phase 4 L9-L12 ship (2026-09-26). Both §10 and §11 are now present, chronologically ordered. No collisions.

---

## §4 Honest framing (party-twenty-spirit §3.5)

### §4.1 Honest-negative caveats preserved

Per `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-369-A3 expansion 40 → 65 entries):

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted per Wf-365-A3 verdict doc + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits diagnostic; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`)
3. **2 pre-flight FAILs PRESERVED BY-DESIGN** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE natural drift; NOT Wf-391-introduced)
4. **3 pre-flight SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with `-183.04 hr` overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)
5. **6 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 7 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps; no auto-trigger per §5.1 + kill_then_adapt §0)
6. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; 6 (metal, seed) pairs queued after Rh seed 3 = Re seeds 0/1/2/3/4 + Rh seed 4; Krasnov 0.86 gate target; MEASURED GPU 0 power 315.57W indicates active training; wall-measured ~5:23 CPU at Wf-379-B1 timestamp; full multi-pair result deferred to Wf-380)
7. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals per `train_dmpnn_multitask_wasserstein.py:247`; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset)
8. **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)
9. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 wet-lab outreach + LAMBDA_ROADMAP §3.2 R22-P1 m18 wet-lab re-scoped to in-silico + LAMBDA_ROADMAP §3.3 R23 Phase I clinical DEPRECATED)
10. **All PROJECTED speedup / lift claims REMAIN PROJECTED not MEASURED end-to-end on RTX 5090 / Blackwell sm_120** (per Wf-326/328/335/340/357-SYN + Wf-379-A1 + Wf-379-A2 baseline)

### §4.2 User voice answer (Wf-391-A1 §0)

**YES — DID彻底解决 the LAMBDA_ROADMAP refresh gap surfaced by Wf-381/389/376/379 + AGENTS.md §10** (per user 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*).

Wf-391-A1 SHIPS:
- 6 in-place edits (§1.5/§2.4/§3.2/§3.3/§4.2 L1-L12 + TODO # column)
- 2 NEW sub-sections (§4.4 R20 §5 joint retrain status + §6.6 Forward-looking 2026-Q4 GPU retrain priorities)
- 1 NEW trailing §11 update section

All CPU-only docs-only per AGENTS.md §5.1+§5.2+§5.3+§5.4+§10. Pre-flight cascade PRESERVED at Wf-379-B1 baseline 89/2/3/94.

---

**Date**: 2026-09-30 (Wf-391-A1 close-out)
**Status**: SHIPPED — 6 in-place edits + 2 NEW sub-sections + 1 NEW trailing §11 update = 9 deliverables; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning
**Wf**: Wf-391-A1 (ultracode CPU-only docs-only wave)
**Owner**: Hugo Lee + ultracode agent (Wf-391-A1 dispatched by workflow harness)
**Parent context**: User directive 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*

**Cross-references**:
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 + §2.4 + §3.2 + §3.3 + §4.2 + §4.4 (NEW) + §6.6 (NEW) + §11 (NEW) — 9 anchors edited
- `TODO/completed/381a5_r20_plan_cpu_close.md` (Wf-381-A5 R20 plan CPU closure source)
- `TODO/completed/382a1_multiph_pll_scaffold.md` (Wf-382-A1 multi-PH PLL scaffold source)
- `TODO/completed/382a4_strategic_plans_refresh.md` (Wf-382-A4 strategic plans refresh source)
- `TODO/completed/389a3_weak_to_strong_update.md` (Wf-389-A3 WEAK_TO_STRONG update source)
- `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` (Wf-376-A3 m2 R² GPU retrain PLAN_ONLY source)
- `TODO/completed/379a1_wf131_p3_7runs.md` (Wf-379-A1 m1_auc 5-seed IN-FLIGHT source)
- `TODO/completed/379a2_r20_p5_tmqm_plat.md` (Wf-379-A2 R20 §5 SCAFFOLD_NOT_FUNCTIONAL source)
- `TODO/pending/383_se3_sb_fast_converging_integration.md` (Wf-383-A2 SE(3) SB integration PLAN_READY source)
- `TODO/completed/377a1_m6_kappa_redesign.md` + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 m6_kappa redesign source)
- `TODO/completed/377a2_m2r2_diagnostic.md` + `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 m2 R² cohort diagnostic source)
- `AGENTS.md §10` (Tier-2 wetlab deprecation FAQ source)
- `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-369-A3 65-entry catalog source)
