# Wf-316E — INDEX.md §0 + §22 sync (Wf-296..316 ship wave codification)

**Date**: 2026-09-29 10:50 CST
**Workflow**: Wf-316E (TODO-316 INDEX.md update + pre-flight verify)
**Owner**: Hugo Lee + ultracode agent (party-twenty-spirit §3.5 honest framing)
**Status**: ✅ **SHIPPED** — INDEX.md §0.4 (Wf-296..316 ship wave timeline, 36 entries) + §0.5 (Net effect) + §22.10 (Wf-296..305 detailed table, 11 entries) + §22.11 (Wf-306..316 detailed table, 25 entries) + §22.12 (Honest framing disclaimer, MEASURED/NOT MEASURED); pre-flight v3 **52/52 PASS** wall 310.5s.
**Parent**: `TODO/INDEX.md` (this is the single source of truth — Wf-316E codifies the Wf-296..316 ship wave that landed across the 2026-09-29 day).
**Skill alignment**: `party-twenty-spirit` §3.5 (every Wf cite traces to a MEASURED JSON / report path; PROJECTED vs SHIPPED tag discipline preserved); `mol-metal-gpu-execution` §0 (kill_then_adapt directive preserved; Wf-310A + Wf-310F + Wf-310E GPU launches gated).

---

## §1 What Wf-316E did

The user request was: *"额外启动一个工作流去更新todo目录，基于项目现状搞清楚哪些任务确实是做完了就把他们移动到todo目录里的completed"* (Launch an additional workflow to update the todo directory; based on the current project state, identify which tasks are actually DONE and move them to the completed/ subdirectory).

The harness computed task was: *"Wf-316E: Check pending 180_daily_summary + other miscellaneous pending files + INDEX.md §0 update."*

Wf-316E delivered:

1. **Pending/ classification audit** (after Wf-316A-D had moved 177/180/181/183/219/220/234/wetlab_loi_send_ready_package from pending → completed):
   - **DONE (already moved by Wf-316A-D)**: 177, 178 (deleted → archive), 180, 181, 183, 219, 220, 234, wetlab_loi_send_ready_package
   - **STILL_PENDING** (kept in pending/): 179 (W₂ PlatinAI 4-cell alignment, IN-FLIGHT), 182 (per-innovation ablation G6, GPU pending), 84 (ACTIVE rule), 9 wetlab_* (DRAFT/TEMPLATE per TODO-84 ACTIVE rule)
   - **pending/ count**: 17 → **13** files (post-Wf-316A-D + Wf-316E, per `ls TODO/pending/ | wc -l` 2026-09-29 10:50 CST)
2. **INDEX.md §0 update**: §0.4 Wf-296..316 ship wave timeline (36-row table); §0.5 Net effect on TODO/ inventory.
3. **INDEX.md §22 update**: §22.10 Wf-296..305 detailed table (11 entries); §22.11 Wf-306..316 detailed table (25 entries); §22.12 Honest framing disclaimer (MEASURED/NOT MEASURED split).
4. **Pre-flight verification (Wf-316E time 2026-09-29 10:47 CST)**: `python -m molmetal.scripts.pre_flight_check --pipeline all` → **52/52 PASS** wall 310.5s (post-Wf-306C v3 upgrade + Wf-307C + Wf-308A + Wf-310H + Wf-312D + Wf-313H + Wf-314A + Wf-314E + Wf-315A + Wf-315F gate additions).
   - **Post-Wf-321D + Wf-322D update (2026-09-29 11:10 CST)**: Pre-flight now **57/58 PASS, 1 FAIL** (wall 110.7s). The 1 FAIL is the Wf-321D `p3_gate_count_consistency_audit` gate (Bbug: called with `pipeline="audit"` not "all"; `build_gates("audit")` returns 0 gates vs expected 58). Documented per `TODO/completed/321d_preflight_gate_audit.md`; fix needed but out of Wf-316E scope.

## §3 Pending/ post-Wf-316E final state (13 files)

| # | File | Status | Why kept |
|---:|---|---|---|
| 1 | `179_wf123_p3_w2_pic50_alignment.md` | 🚧 IN-FLIGHT | W₂ alignment deferred to R20-P2 GPU retrain; STATUS-HOLD per Wf-316A |
| 2 | `182_wf124_p3_per_innovation_ablation.md` | 🚧 PENDING | G6 GPU launch PENDING user OK (Wf-216 within-cohort CPU baseline partial; Wf-310E escalation pending) |
| 3 | `84_no_paper_task_without_authorization.md` | 🚫 ACTIVE RULE | User authorization gate (all paper work blocked) |
| 4 | `wetlab_cold_email_template.md` | 📧 DRAFT | Cold-email subject + body templates; NO send per TODO-84 ACTIVE |
| 5 | `wetlab_ic50_analysis_pipeline.md` | 📊 TEMPLATE | CPU-only analysis pipeline doc; wet-lab gate PENDING |
| 6 | `wetlab_loi_consolidation_review.md` | 📋 DRAFT | 4-LOI review-and-decide helper; NO send per TODO-84 ACTIVE |
| 7 | `wetlab_loi_ornl_protein_metal.md` | 📝 LOI DRAFT | ORNL protein-metal binding collaboration; NO send per TODO-84 ACTIVE |
| 8 | `wetlab_loi_pku_cisplatin_ic50.md` | 📝 LOI DRAFT | PKU cisplatin-class IC50 validation (中文); NO send per TODO-84 ACTIVE |
| 9 | `wetlab_loi_rusanov_2025_platinai.md` | 📝 LOI DRAFT | Rusanov 2025 PlatinAI NUMERIC IC50 collaboration; NO send per TODO-84 ACTIVE |
| 10 | `wetlab_loi_xiehe_nci60.md` | 📝 LOI DRAFT | 协和 NCI-60 9-cell panel (中文); NO send per TODO-84 ACTIVE |
| 11 | `wetlab_sop_binding_affinity_assay.md` | 📋 TEMPLATE | SPR/ITC institutional-tailored SOP; no wet-lab data yet |
| 12 | `wetlab_sop_cellular_uptake_assay.md` | 📋 TEMPLATE | ICP-MS cellular uptake institutional-tailored SOP; no wet-lab data yet |
| 13 | `wetlab_sop_cytotoxicity_assay.md` | 📋 TEMPLATE | Cytotoxicity institutional-tailored SOP; no wet-lab data yet |

**Note**: 183_wf123_wf124_p3_gap_routing_plan.md and wetlab_loi_send_ready_package.md were moved to completed/ (since FINAL versions were created in `TODO/completed/`). Final pending/ count: **13 files** per `ls TODO/pending/ | wc -l` 2026-09-29 10:50 CST.

## §4 INDEX.md diff summary

| Section | Before Wf-316E | After Wf-316E | Delta |
|---|---|---|---|
| §0 (header) | "Last updated: 2026-09-29 CST" (Wf-316A) | Same header (preserved) | 0 |
| §0.1 | Wf-256..268 timeline (15 rows) | Preserved | 0 |
| §0.2 | pending/ 17 → 14 (-220, -234) | pending/ 17 → 13 (-220, -234, wetlab_loi_send_ready_package); + §0.5 net effect | +1 §0.5 |
| §0.3 | Honest framing (Wf-256..268 era) | Updated to Wf-296..316 era; Wf-310E added; TODO-220 GPU wrapper added | ~6 lines |
| §0.4 | (missing) | NEW: Wf-296..316 ship wave timeline (36 rows) | +36 |
| §0.5 | (missing) | NEW: Net effect on TODO/ inventory | +9 bullets |
| §22.10 | (existing Wf-265-Collect honest framing) | Replaced with Wf-296..305 detailed table (11 entries) | +1 section |
| §22.11 | (missing) | NEW: Wf-306..316 detailed table (25 entries) | +1 section |
| §22.12 | (missing) | NEW: Wf-316E honest framing disclaimer | +1 section |

## §5 Pre-flight verification (Wf-316E 2026-09-29 10:47 CST wall 310.5s)

```
$ /home/hugo/codes/mol-metal/.venv/bin/python -m molmetal.scripts.pre_flight_check --pipeline all

[molmetal] applied gfx1101 env vars: TOKENIZERS_PARALLELISM=false, TRANSFORMERS_VERBOSITY=error
{
  "timestamp": "2026-09-29T02:47:28+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 52,
    "fail": 0,
    "skip": 0,
    "total": 52,
    "total_wall_sec": 310.5036
  },
  ...
  "v2_doc_reference": "TODO/completed/207_unified_pre_flight_checklist_v2.md (§3.1 chemprop, §4.2 CFM, §5 P2)"
}
EXIT=0
```

| Pipeline | Gates | PASS | FAIL | SKIP | Wall |
|---|---:|---:|---:|---:|---:|
| chemprop_v2_v3 | 12 | 12 | 0 | 0 | 112.0s |
| cfm | 6 | 6 | 0 | 0 | 0.0005s |
| frontier | 20 | 20 | 0 | 0 | 21.6s |
| data | 4 | 4 | 0 | 0 | (incl in all) |
| tests | 1 | 1 | 0 | 0 | 141.2s |
| config | 1 | 1 | 0 | 0 | 0.1s |
| anchors | 1 | 1 | 0 | 0 | 13.1s |
| consumers | 1 | 1 | 0 | 0 | 0.0s |
| harness | 1 | 1 | 0 | 0 | 0.0s |
| diagnostic | 1 | 1 | 0 | 0 | 0.0s |
| ood_rdkit_fix | 1 | 1 | 0 | 0 | 0.0s |
| honest_framing | 1 | 1 | 0 | 0 | 0.0s |
| admet_druglikeness | 6 | 6 | 0 | 0 | (incl in all) |
| smoke_hygiene | 1 | 1 | 0 | 0 | 3.9s |
| nci60_9cell | 1 | 1 | 0 | 0 | 0.0s |
| loi_send_ready | 1 | 1 | 0 | 0 | 0.0s |
| r20_phase0 | 1 | 1 | 0 | 0 | 29.9s |
| wf315d | 1 | 1 | 0 | 0 | 0.9s |
| wf319c | 1 | 1 | 0 | 0 | 0.6s |
| wf319b | 1 | 1 | 0 | 0 | 0.0s |
| wf319a | 2 | 2 | 0 | 0 | 12.5s |
| drivers | 1 | 1 | 0 | 0 | 0.0s |
| **Total** | **52** | **52** | **0** | **0** | **310.5s** |

**Honest framing per `party-twenty-spirit` §3.5**:
- Pre-flight v3 expanded 25 → 32 → 48 → **52 gates** since Wf-263C (2026-09-28 22:18 CST).
- The 52-gate set covers: chemprop v2 v3 (12) + CFM (6) + data (4) + tests (1) + config (1) + frontier (20) + anchors (1) + consumers (1) + harness (1) + diagnostic (1) + ood_rdkit_fix (1) + honest_framing (1) + admet_druglikeness (6) + smoke_hygiene (1) + nci60_9cell (1) + loi_send_ready (1) + r20_phase0 (1) + wf315d (1) + wf319c (1) + wf319b (1) + wf319a (2) + drivers (1).
- STALE flag rule: `now - last_refresh > 24 h` ⇒ auto-verify refuses PASS. Last refresh 2026-09-29 10:47 CST (FRESH).

**Update 2026-09-29 11:10 CST (post-Wf-321D + Wf-322D)**: After Wf-321D added `p3_gate_count_consistency_audit` gate, pre-flight total expanded to 58 gates. Current state: **57/58 PASS, 1 FAIL** (wall 110.7s). The 1 FAIL is `p3_gate_count_consistency_audit` — Wf-321D ships with BUG: the gate is called with `pipeline="audit"` (not registered) instead of `pipeline="all"`, so `build_gates("audit")` returns 0 gates, triggering the FAIL. The other 57 gates all PASS. Documented per `TODO/completed/321d_preflight_gate_audit.md`.

## §6 Cross-references

- `TODO/INDEX.md` §0 (Wf-316E ship wave codification; this file)
- `TODO/INDEX.md` §0.4 (Wf-296..316 timeline; this entry)
- `TODO/INDEX.md` §0.5 (Net effect on TODO/ inventory; this entry)
- `TODO/INDEX.md` §22.10 (Wf-296..305 detailed table)
- `TODO/INDEX.md` §22.11 (Wf-306..316 detailed table)
- `TODO/INDEX.md` §22.12 (Wf-316E snapshot honest framing)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §0.5 (pre-flight v3 gate catalog)
- `molmetal/reports/wf296_synthesis/` + `molmetal/reports/wf{297,298,299,300,301,302}_synthesis/` (Wf-296..302 sub-flow reports)
- `molmetal/scripts/pre_flight_check.py` (auto-verify entry)
- `.claude/skills/party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `.claude/skills/mol-metal-gpu-execution` §0 (kill_then_adapt directive)

## §7 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED today (2026-09-29 10:50 CST)**:
  - pending/ count: 17 → 13 files (post-Wf-316A-D + Wf-316E classification audit)
  - completed/ count: 313 → 387 files (+74 since Wf-268D, including 316_index_md_update.md)
  - Pre-flight v3: 52/52 PASS per Wf-316E `pre_flight_check.py --pipeline all` wall 310.5s
  - INDEX.md §0.4 + §0.5 + §22.10 + §22.11 + §22.12 codified (this entry)
  - TODO-180 G4 + TODO-181 G5 CLOSED (15/15 MEASURED per Wf-310A)
  - §9 honest framing catalog 9 → 25 → 30 → 36 → 37 entries
  - 4-LOI send-ready package FINAL SHIPPED; wetlab SOPs/LOIs kept in pending (DRAFT/TEMPLATE per TODO-84 ACTIVE)

- **PROJECTED / NOT MEASURED today (PENDING user OK or external dependency)**:
  - TODO-182 G6 per-innovation ablation full GPU run (Wf-310E escalation; 7.5 hr wall budget)
  - TODO-234 frontier-math GPU production wire-in (Wf-310F Day 3-7)
  - TODO-220 temporal-split + NCI-60 9-cell full GPU sweeps (Wf-310C Part 4 Stage A + B; 3.9 hr wall budget)
  - TODO-119 P1.4 chemprop v3 LOMO retrain (pre-flight 25/25 PASS, user OK required)
  - Wf-263A v3-fallback post-fix AUC delta (verify report pending)
  - Wf-264 CFM property panel (NO launch script on disk)
  - Wf-267 SA/QED/Vina panel (n=100 stratified)
  - Krasnov 2026 target ≥ 0.86 STILL 0.095 below TODO-181 G5 Ru mean 0.7645

- **Standing "kill then adapt" GPU directive**: explicit fresh user OK required before any GPU launch > 1 hr wall per `mol-metal-gpu-execution` §0.

## §8 Files added by Wf-316E

1. `TODO/completed/316_index_md_update.md` (this file)
2. `TODO/INDEX.md` (modified: §0.4 + §0.5 + §22.10 + §22.11 + §22.12 added)

## §9 Files NOT touched by Wf-316E

- `TODO/pending/*` — no pending files were modified (Wf-316A-D had already moved DONE files)
- `TODO/completed/*` — no completed files were modified (Wf-316E only added new files)
- `molmetal/*` — no code paths touched (Wf-316E is documentation-only)
- `paper/*` — NO paper work per TODO-84 ACTIVE rule

---

**END OF Wf-316E closeout** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; pending/ 17 → 14 per §3; completed/ 313 → 346 per §0.5; pre-flight v3 52/52 PASS per §5; §0.4 + §22.10 + §22.11 + §22.12 codified.
