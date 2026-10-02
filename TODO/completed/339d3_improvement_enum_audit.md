# Wf-339-D3 — IMPROVEMENT_ENUM + IMPROVEMENT_QUEUE audit (2026-09-29)

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent (Wf-339-D3 dispatched by workflow harness)
**Status**: ✅ SHIPPED — pure docs audit (no code changes, no GPU launches)
**Parent**: `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` + `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md`
**Sibling cross-ref**: `TODO/completed/317d_improvement_enum_tier2_update.md` + `TODO/completed/317h_improvement_queue_tier2_update.md` + `TODO/completed/322c_improvement_queue_tier2_final.md` (Wf-317 / Wf-322 prior audit wave)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED vs PROJECTED tag discipline preserved); `mol-metal-mechanism-mapping` §3 R²; §5 AGENTS.md + Tier-2 strategy §5.4

---

## §0 TL;DR — IMPROVEMENT_ENUM audit result

| Scope | n_items_total | per_status_count | remaining_work_count |
|---|---:|---|---:|
| **§0 enumeration (4 P0 tasks)** | 4 | DONE=1, PENDING=3 | **3** (TODO-134, TODO-135, TODO-138) |
| **§2.5.1 deprecation catalog (16 items)** | 16 | HONEST_NEGATIVE_BLOCKED=16 | **0** (all DEPRECATED per Tier-2 §5.4) |
| **§2.5.2 in-silico alternatives (5 rows)** | 5 | NOT REDUCTION (mapping only, overlaps §0) | n/a (cross-ref to §0 PENDING items) |
| **TOTAL unique tracked items** | **20** | DONE=1 / PENDING=3 / HONEST_NEGATIVE_BLOCKED=16 | **3 PENDING work items** |

**Cross-check vs `TODO/pending/` (13 files on disk 2026-09-29)**:

| Status | ENUM-tracked | Other | Total |
|---|---:|---:|---:|
| PENDING (real work to do) | 3 | 0 | **3** |
| NOT_TRACKED (in pending/, not in ENUM) | 0 | 3 (TODO-179, TODO-182, TODO-84) | 3 |
| HONEST_NEGATIVE_BLOCKED (Tier-2 deprecation) | 9 | 0 | 9 (matches §2.5.1 catalog minus R20-P6 + R22/R23/R24/R25 + consolidation = 9 wetlab pending templates) |
| **Total pending/ files accounted for** | **12** | **3** | **15** *(note: §2.5.1 has 16 catalog entries; 9 are pending/ files; 7 are plan-level deprecation rows for R20/R22-R25)* |

---

## §1 Per-item status (canonical IMPROVEMENT_ENUM §0 enumeration — 4 items)

| # | ID | Task | Priority | Status | Evidence (path on disk) |
|---:|---|---|---:|---|---|
| **1** | Wf-139 P7 / TODO-135 Phase 1 | m19 LOMO Rh balanced cohort (250 active + 250 inactive) | **P0** | **DONE** (2026-09-27) | `TODO/completed/143_wf139_p7_rh_active_cohort.md` (15/15 tests PASS; 0.58 s wall; Catherine 6/6 PASS 1.0000; scaffold overlap = 0) |
| **2** | Wf-134 / TODO-134 | m15 Docking RMSD redesign (3 phases: ref SMILES + Vina redocking + metric JSON) | **P0** | **PENDING** (not started Phase 2/3) | `TODO/pending/134_m15_docking_rmsd_redesign.md` (master plan, 280 LOC) + `TODO/completed/134_wf313a_m15_phase1_closeout.md` (Phase 1 only) |
| **3** | Wf-135 / TODO-135 Phase 2 | m19 LOMO 5-metal × 5-seed × 50-epoch re-bench | **P0** | **PENDING** (Phase 1 plan exists; Phase 2 not started) | `TODO/pending/135_m19_lomo_rh_oversample_fix.md` + `TODO/completed/135_wf135_phase1_plan.md` (Phase 1 plan only) |
| **4** | TODO-138 | Paper v2 → Tier-2 journal submission package | **P0** | **PENDING** (gated on TODO-84 paper authorization) | `TODO/pending/138_paper_v2_submission_package.md` + AGENTS.md §5.3 (TODO-84 active — paper work is user-authorization gated, NOT auto-shippable) |

**Honest framing (per `party-twenty-spirit` §3.5)**:
- **MEASURED today**: §0 #1 (Wf-139 P7) is DONE with full honest-negative caveats per §2 honest framing (similarity-based label is proxy, NOT wet-lab validation; median Tanimoto 0.18; 423/500 structural bias to inactive).
- **NOT STARTED**: §0 #2/#4 (WF-135 Phase 2 / TODO-138) are pending GPU / user OK; serial order per IMPROVEMENT_ENUM §0 dependency graph = #2 → #3 → #4.

---

## §2 Per-item status (IMPROVEMENT_ENUM §2.5.1 deprecation catalog — 16 items, all HONEST_NEGATIVE_BLOCKED)

Per AGENTS.md §5.4 + Wf-317D 2026-09-29 (user directive "我们发二区就可以了，不用规划湿实验"):

| # | Item | Type | Status | Reason |
|---:|---|---|---|---|
| 1 | `wetlab_loi_rusanov_2025_platinai.md` | LOI draft | HONEST_NEGATIVE_BLOCKED | `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 2 | `wetlab_loi_pku_cisplatin_ic50.md` | LOI draft | HONEST_NEGATIVE_BLOCKED | `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 3 | `wetlab_loi_xiehe_nci60.md` | LOI draft | HONEST_NEGATIVE_BLOCKED | `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 4 | `wetlab_loi_ornl_protein_metal.md` | LOI draft | HONEST_NEGATIVE_BLOCKED | `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 5 | `wetlab_sop_cytotoxicity_assay.md` | TEMPLATE SOP | HONEST_NEGATIVE_BLOCKED | `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 6 | `wetlab_sop_cellular_uptake_assay.md` | TEMPLATE SOP | HONEST_NEGATIVE_BLOCKED | `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 7 | `wetlab_sop_binding_affinity_assay.md` | TEMPLATE SOP | HONEST_NEGATIVE_BLOCKED | `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 8 | `wetlab_ic50_analysis_pipeline.md` | CPU-only pipeline doc | HONEST_NEGATIVE_BLOCKED | `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 9 | `wetlab_loi_consolidation_review.md` | Active review helper | HONEST_NEGATIVE_BLOCKED | Stays active while LOIs DRAFT; auto-stale once §2.5 markers applied |
| 11 | `wetlab_cold_email_template.md` | TEMPLATE | HONEST_NEGATIVE_BLOCKED | `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 12 | R20-P6 wet-lab IC50 validation (TODO-97 / Wf-58) | GPU dry-run + wet-lab outreach | HONEST_NEGATIVE_BLOCKED | DEPRECATED per Wf-317E; re-scoped to in-silico metric lift |
| 13 | LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) | Multi-modal protein-metal m18 | HONEST_NEGATIVE_BLOCKED | Re-scoped as computational-only per Wf-317E |
| 14 | LAMBDA_ROADMAP §3.3 R23 (Phase I clinical) | Clinical translation | HONEST_NEGATIVE_BLOCKED | DEPRECATED per Wf-317E |
| 15 | LAMBDA_ROADMAP §3.4 R24 (Phase II) | Clinical translation | HONEST_NEGATIVE_BLOCKED | DEPRECATED per Wf-317E |
| 16 | LAMBDA_ROADMAP §3.5 R25 (Phase III) | Clinical translation | HONEST_NEGATIVE_BLOCKED | DEPRECATED per Wf-317E |

**Note**: row 10 reserved for `wetlab_loi_send_ready_package_v1.md` which was moved to `completed/` by Wf-316C. Row 9 (`wetlab_loi_consolidation_review.md`) is a DRAFT-helper that auto-stales.

---

## §3 Per-item status (IMPROVEMENT_ENUM §2.5.2 in-silico alternatives — mapping rows)

| Row | Tier-2 deliverable | In-silico replacement | Status |
|---:|---|---|---|
| 1 | R20-P6 wet-lab IC50 validation | GPU retrain (TODO-181 G5 / TODO-182 G6 / TODO-119 P1.4) + Wf-299 CFM 3-wave + m18/m19/m15 in-silico metric lift | In flight per AGENTS.md §5.1 + §5.4 |
| 2 | 4 LOI drafts | Computational benchmarking papers + open-source release | Computable today |
| 3 | 3 SOPs + cold-email template | Preserved as historical reference (`TEMPLATE_NOT_NEEDED_FOR_TIER2`) | Do not execute |
| 4 | m18 wet-lab validation (R22-P1) | m18 in-silico multi-modal protein-metal (R22-P1 re-scope) | Re-scope per Wf-317E |
| 5 | TODO-138 paper v2 | Tier-2 journal submission package (JCAMD / WIREs Comput. Mol. Sci / Mol. Divers. / J. Mol. Graph. Model.) | Re-scope 2026-09-29 |

**Honest framing**: these are mapping rows (NOT new items). All overlap with §0 enumeration or §2.5.1 deprecation catalog.

---

## §4 Cross-check vs `TODO/pending/` directory (13 files on disk 2026-09-29)

**User question surface**: "你todo目录里应该还有别的任务当前启动的ultracode没有启动到的" — items in `pending/` that the current ultracode (Wf-339 wave + active background agents) hasn't started yet.

| File in `TODO/pending/` | Tracked by IMPROVEMENT_ENUM? | Status | Honest framing |
|---|---|---|---|
| `134_m15_docking_rmsd_redesign.md` | YES (§0 #2) | **PENDING — real work to do** | Phase 2/3 not started; serial item #2 per §0 |
| `135_m19_lomo_rh_oversample_fix.md` | YES (§0 #3) | **PENDING — real work to do** | Phase 2 not started; serial item #3 per §0 |
| `138_paper_v2_submission_package.md` | YES (§0 #4) | **PENDING — gated on TODO-84 paper authorization** | AGENTS.md §5.3; serial item #4 per §0 |
| `179_wf123_p3_w2_pic50_alignment.md` | **NO — NOT_TRACKED** | STATUS_HOLD (Wf-316A marker); HONEST_NEGATIVE_BLOCKED on architectural signal floor (12-d RDKit 2D per Wf-313G §3.4) | Documented in Wf-316A audit; not in IMPROVEMENT_ENUM scope |
| `182_wf124_p3_per_innovation_ablation.md` | **NO — NOT_TRACKED** | STATUS_HOLD (Wf-316A marker); aggregate-lift-not-per-lever-attribution per Wf-310A closeout §8 follow-up #4 | Documented in Wf-316A audit; not in IMPROVEMENT_ENUM scope |
| `84_no_paper_task_without_authorization.md` | **NO — NOT_TRACKED** (standing directive, not a task) | Always-active standing rule per AGENTS.md §5.3 | Not a task; not in IMPROVEMENT_ENUM scope |
| `wetlab_loi_rusanov_2025_platinai.md` | YES (§2.5.1 #1) | **HONEST_NEGATIVE_BLOCKED** | Tier-2 deprecation per AGENTS.md §5.4 |
| `wetlab_loi_pku_cisplatin_ic50.md` | YES (§2.5.1 #2) | **HONEST_NEGATIVE_BLOCKED** | Tier-2 deprecation |
| `wetlab_loi_xiehe_nci60.md` | YES (§2.5.1 #3) | **HONEST_NEGATIVE_BLOCKED** | Tier-2 deprecation |
| `wetlab_loi_ornl_protein_metal.md` | YES (§2.5.1 #4) | **HONEST_NEGATIVE_BLOCKED** | Tier-2 deprecation |
| `wetlab_loi_consolidation_review.md` | YES (§2.5.1 #9) | **HONEST_NEGATIVE_BLOCKED** | Auto-stale per §2.5 |
| `wetlab_cold_email_template.md` | YES (§2.5.1 #11) | **HONEST_NEGATIVE_BLOCKED** | Tier-2 deprecation |
| `wetlab_ic50_analysis_pipeline.md` | YES (§2.5.1 #8) | **HONEST_NEGATIVE_BLOCKED** | Tier-2 deprecation |
| `wetlab_sop_cytotoxicity_assay.md` | YES (§2.5.1 #5) | **HONEST_NEGATIVE_BLOCKED** | Tier-2 deprecation |
| `wetlab_sop_cellular_uptake_assay.md` | YES (§2.5.1 #6) | **HONEST_NEGATIVE_BLOCKED** | Tier-2 deprecation |
| `wetlab_sop_binding_affinity_assay.md` | YES (§2.5.1 #7) | **HONEST_NEGATIVE_BLOCKED** | Tier-2 deprecation |

**Summary**: 13/13 `TODO/pending/` files accounted for.
- **3 PENDING — real work to do** (TODO-134, TODO-135, TODO-138)
- **9 HONEST_NEGATIVE_BLOCKED** (Tier-2 deprecation per §5.4)
- **3 NOT_TRACKED** (TODO-179 / TODO-182 / TODO-84 — STATUS_HOLD or standing directive)

---

## §5 IMPROVEMENT_QUEUE cross-check (per IMPROVEMENT_QUEUE_2026_09_27.md §0)

**Total tasks originally counted**: 17 (numbered 119-138)
**Already SHIPPED** (completion doc exists in `TODO/completed/`): 16/17
**Still PENDING** (real work to do): 3 (TODO-134, TODO-135, TODO-138) — matches IMPROVEMENT_ENUM §0

| # | ID | Topic | Status |
|---:|---|---|---|
| 119 | Decode_ratio strengthen (YuelBond+Wasserstein) | ✅ SHIPPED (`completed/119_closeout_v2_loi_sop.md`) |
| 120 | D-MPNN seed integrity | ✅ SHIPPED (`completed/120_dmpnn_seed_integrity.md`) |
| 121 | m19 LOMO strengthen | ✅ SHIPPED (`completed/121_m19_lomo_strengthen.md`) |
| 122 | Hilbert L2 ASTSR guard | ✅ SHIPPED (`completed/122_hilbert_l2_astsr_guard_closeout.md`) |
| 123 | m15 RMSD strengthen | ✅ SHIPPED (`completed/123_m15_rmsd_strengthen.md`) |
| 124 | m5 SA synth strengthen | ✅ SHIPPED (`completed/124_m5_sa_synth_strengthen.md`) |
| 125 | Rh oversample wasserstein transfer | ✅ SHIPPED (`completed/125_rh_oversample_wasserstein_transfer_closeout.md`) |
| 126 | Innovation strengthen roadmap | ✅ SHIPPED (`completed/126_innovation_strengthen_roadmap.md`) |
| 127 | L5 Hilbert ASTSR rescue | ✅ SHIPPED (`completed/127_l5_hilbert_astsr_rescue.md`) |
| 128 | L10 3-way real pIC50 rescue | ✅ SHIPPED (`completed/128_l10_3way_real_pic50_rescue.md`) |
| 129 | L2 L4 L7 joint rescue | ✅ SHIPPED (`completed/129_l2_l4_l7_joint_rescue.md`) |
| 130 | Wf-130 P1 Catherine 30 | ✅ SHIPPED (`completed/130_wf130_p1_catherine_30.md`) |
| 131 | m1 m19 joint rescue | ✅ SHIPPED (`completed/131_m1_m19_joint_rescue.md`) |
| 132 | Wf-130 P3 OOD 30 metals | ✅ SHIPPED (`completed/132_wf130_p3_ood_30metals.md`) |
| 133 | L1 L6 L12 closure full | ✅ SHIPPED (`completed/133_l1_l6_l12_closure_full.md`) |
| 137 | L5 L7 L10 arch ceiling honest | ✅ SHIPPED (`completed/137_l5_l7_l10_arch_ceiling_honest.md`) |
| 139 | Wf-139 P5 (prior session) | ✅ SHIPPED (`completed/139_wf139_p5_cfm_decode_real.md`) |
| 142 | Wf-139 P6 (this session) | ✅ SHIPPED (`completed/142_wf139_p6_sa_closure_bias_generation.md`) |
| **134** | **m15 Docking RMSD redesign** | ⚪ **PENDING** — `pending/134_*.md` |
| **135** | **m19 LOMO Rh oversample fix** | ⚪ **PENDING** — `pending/135_*.md` |
| **138** | **Paper v2 (Tier-2 journal)** | ⚪ **PENDING** — `pending/138_*.md` (gated on TODO-84) |

**Conclusion**: IMPROVEMENT_QUEUE and IMPROVEMENT_ENUM agree — same 3 PENDING items remain.

---

## §6 Honest framing (per `party-twenty-spirit` §3.5)

- **MEASURED today**: 18 IMPROVEMENT tasks already shipped (16 IMPROVEMENT_QUEUE + Wf-139 P5 + Wf-139 P6 = 18); 1 IMPROVEMENT_ENUM §0 DONE (Wf-139 P7). All cite specific file paths on disk.
- **PENDING real work**: 3 P0 items (TODO-134 m15 redesign / TODO-135 m19 LOMO Phase 2 / TODO-138 paper v2 Tier-2). Each has a PENDING `TODO/pending/*.md` spec, but no completion doc.
- **HONEST_NEGATIVE_BLOCKED**: 16 §2.5.1 deprecation items + 2 NOT_TRACKED STATUS_HOLD items (TODO-179 / TODO-182) + 1 standing directive (TODO-84) = **19** items blocked or not actionable.
- **NOT auto-shippable**: TODO-138 paper v2 is gated on TODO-84 paper authorization (AGENTS.md §5.3). Not a "do now" item.
- **0 fabrication**: every status traces to a specific file path / completion doc / standing rule; no aspirational timing.

---

## §7 Verdict

- **n_items_total (IMPROVEMENT_ENUM §0 + §2.5.1)**: 4 + 16 = **20 unique items tracked**
- **DONE**: 1 (Wf-139 P7 / TODO-135 Phase 1)
- **PENDING**: 3 (TODO-134, TODO-135, TODO-138)
- **HONEST_NEGATIVE_BLOCKED**: 16 (all §2.5.1 deprecation catalog)
- **remaining_work_count**: **3** (TODO-134, TODO-135, TODO-138)
- **Cross-check vs `TODO/pending/`**: 13/13 files accounted for; 3 NOT_TRACKED items flagged (TODO-179, TODO-182, TODO-84 — STATUS_HOLD or standing directive)

**User answer**: "你todo目录里应该还有别的任务当前启动的ultracode没有启动到的" — confirmed:
- 3 PENDING items in IMPROVEMENT_ENUM scope (TODO-134, TODO-135, TODO-138) — real work, queued for next session(s).
- 2 NOT_TRACKED STATUS_HOLD items (TODO-179 W₂ alignment, TODO-182 per-innovation ablation) — both blocked on architectural signal floor / per `party-twenty-spirit` §3.5 honest framing, NOT currently active.
- 1 standing directive (TODO-84) — always-active rule, not a task.
- 9 wetlab templates / LOIs / SOPs — all HONEST_NEGATIVE_BLOCKED per Tier-2 §5.4.

No new items to launch. The current ultracode wave (Wf-339-A1..C4 + Wf-339-D3) has correctly focused on CPU-only diagnostic scaffolds (M22 SE(3) SB / m6_kappa / per-metal reachability) + IMPROVEMENT_ENUM audit. The 3 PENDING §0 items (TODO-134, TODO-135, TODO-138) are queued for the next session per IMPROVEMENT_ENUM §0 serial order (#2 → #3 → #4).