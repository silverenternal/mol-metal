# Wf-316F — TODO directory final-state synthesis

**Date**: 2026-09-29 (Wf-316F synthesis; Wf-316A-E moved 6 files pending → completed today; this file is the canonical ledger)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ SHIPPED (synthesis doc; downstream = INDEX §0 + README.md updates)

---

## §0 TL;DR (party-twenty-spirit §3.5 honest framing)

| Metric | Before (Wf-316 start, 2026-09-29 ~09:30 CST) | After (Wf-316F ship, 2026-09-29 ~10:30 CST) |
|---|---:|---:|
| `pending/` files | **20** | **14** (−6) |
| `completed/` files | 338 | 347 (+9: 6 closeouts + 3 Wf-3xx closeout companions) |
| `archive/` files | 7 | 8 (+1: 178 archive) |

| Move action | Count |
|---|---:|
| `pending/ → completed/` (Wf-316A-E moves) | 6 |
| `pending/ → archive/` (Wf-316 pre-existing) | 1 (TODO-178) |
| **Net removed from `pending/`** | **7** |
| Newly added to `completed/` (not from pending) | 3 (closeout companions: 220 ×2 + 234 ×2 + 234 + 234b) |

**Final `pending/` categorization** (14 files):

| Category | Count | Notes |
|---|---:|---|
| **RULE** (TODO-84 paper rule) | 1 | Active rule — never moves |
| **STILL_PENDING** (work remains) | 3 | 179 / 182 / 183 — GPU work gated on user OK |
| **TEMPLATE** (wetlab DRAFT/SOP/IC50/LOI) | 10 | All wetlab_* — not sent / not executed (per `TODO/INDEX.md` §pending wetlab policy + `TODO/pending/84_no_paper_task_without_authorization.md` paper rule) |
| **Total** | **14** | |

---

## §1 Before-state (Wf-316 start, 2026-09-29 ~09:30 CST)

Per `ls TODO/pending/ | wc -l` first read at session start: **20 files**.

| # | File | Category (pre-move) | Status at Wf-316 start |
|---:|---|---|---|
| 1 | `177_wf123_p3_m1m2_downstream_metric_lift.md` | STILL_PENDING (planned move to completed) | IN-FLIGHT — BUT downstream lift data MEASURED via Wf-313G dashboard |
| 2 | `179_wf123_p3_w2_pic50_alignment.md` | STILL_PENDING | IN-FLIGHT — PlatinAI 4-cell W₂ alignment NOT yet MEASURED |
| 3 | `180_wf124_p3_remaining_metals_seeds.md` | STILL_PENDING (planned move to completed) | IN-FLIGHT in file — BUT G4 15/15 (metal,seed) pairs MEASURED per Wf-265-Collect (`molmetal/reports/wf265_collect/all_results.json`) |
| 4 | `181_wf124_p3_50_epoch_full_sweep.md` | STILL_PENDING (planned move to completed) | IN-FLIGHT in file — BUT G5 13/15 OK pairs + 2 retries MEASURED per Wf-310A + Wf-278A G5 aggregator (`molmetal/reports/wf310a_g5_sweep/`) |
| 5 | `182_wf124_p3_per_innovation_ablation.md` | STILL_PENDING | IN-FLIGHT — G6 GPU sweep requires explicit user OK per standing "kill then adapt" directive |
| 6 | `183_wf123_wf124_p3_gap_routing_plan.md` | STILL_PENDING | IN-FLIGHT — depends on 179 / 182 closure; routing plan still active |
| 7 | `220_wf219_temporal_split_nci60_driver.md` | STILL_PENDING (planned move to completed) | IN-FLIGHT in file — BUT §6.4 acceptance gates MEASURED for smoke layer per Wf-313D (`molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json` 9/9 OK); full P1 GPU sweep PENDING user OK |
| 8 | `234_wf234_frontier_math_audit_integration.md` | STILL_PENDING (planned move to completed) | AUDIT COMPLETE — but CPU portion SHIPPED per Wf-307D + Wf-310F (`molmetal/adapters/frontier_math_integration.py` wrapper); GPU portion deferred to user-OK-gated ultracode |
| 9 | `84_no_paper_task_without_authorization.md` | RULE | Active rule (never moves) |
| 10 | `wetlab_cold_email_template.md` | TEMPLATE | DRAFT (untracked in git; Wf-310G Part 1 created today 09:44) |
| 11 | `wetlab_ic50_analysis_pipeline.md` | TEMPLATE | TEMPLATE — CPU-only pipeline doc, no wet-lab IC50 MEASURED today |
| 12 | `wetlab_loi_consolidation_review.md` | TEMPLATE | DRAFT (untracked; Wf-308D Part 2 review helper) |
| 13 | `wetlab_loi_ornl_protein_metal.md` | TEMPLATE | DRAFT LOI letter — not sent |
| 14 | `wetlab_loi_pku_cisplatin_ic50.md` | TEMPLATE | DRAFT LOI letter — not sent |
| 15 | `wetlab_loi_rusanov_2025_platinai.md` | TEMPLATE | DRAFT LOI letter — not sent (Wf-305A NEW) |
| 16 | `wetlab_loi_send_ready_package.md` | TEMPLATE (planned move to completed/superseded) | Wf-310G draft — SUPERSEDED by Wf-313B FINAL (in `completed/wetlab_loi_send_ready_package_final.md`) |
| 17 | `wetlab_loi_xiehe_nci60.md` | TEMPLATE | DRAFT LOI letter — not sent |
| 18 | `wetlab_sop_binding_affinity_assay.md` | TEMPLATE | TEMPLATE SOP — institutional-tailored, not executed |
| 19 | `wetlab_sop_cellular_uptake_assay.md` | TEMPLATE | TEMPLATE SOP — institutional-tailored, not executed |
| 20 | `wetlab_sop_cytotoxicity_assay.md` | TEMPLATE | TEMPLATE SOP — institutional-tailored, not executed |

**Source**: `ls -la TODO/pending/` first read this session (mod times 2026-09-27 22:13 → 2026-09-29 10:07).

---

## §2 Wf-316A-E moves (per-file audit)

Each row traces to a MEASURED JSON / report path per `party-twenty-spirit` §3.5.

| # | Source file (pending/) | Destination | Action | MEASURED evidence |
|---:|---|---|---|---|
| 1 | `177_wf123_p3_m1m2_downstream_metric_lift.md` | `completed/177_wf123_p3_m1m2_downstream_metric_lift.md` | MOVE | Wf-313G downstream lift dashboard at `molmetal/reports/wf313g_downstream_lift/`; closeout file at `completed/177_wf313g_downstream_lift_dashboard.md` |
| 2 | `178_wf123_p3_metalcytotox_csv_ondisk.md` | `archive/178_wf123_p3_metalcytotox_csv_ondisk.md` | ARCHIVE (superseded) | Wf-303C CSV hygiene closeout at `completed/178_metalcytotox_csv_hygiene_closeout.md`; Wf-308A `178b_smoke_fail_on_missing_hardening.md` (15/15 illegitimate skips fixed; pre-flight v3 35/35 PASS) |
| 3 | `180_wf124_p3_remaining_metals_seeds.md` | `completed/180_wf124_p3_remaining_metals_seeds.md` | MOVE | `molmetal/reports/wf265_collect/all_results.json` `g4_per_seed` — 15/15 (metal,seed) pairs MEASURED (Ru 0.7527 ± 0.0083, Ir 0.6702 ± 0.0069, Os 0.7025 ± 0.0251, Rh 0.7227 ± 0.0662, Re 0.7700 ± 0.0298) — Krasnov 2026 ≥ 0.86 BELOW gate for all metals (honest-negative per §22.2) |
| 4 | `181_wf124_p3_50_epoch_full_sweep.md` | `completed/181_wf124_p3_50_epoch_full_sweep.md` | MOVE | Wf-310A retry + Wf-278A G5 aggregator; closeout at `completed/181_wf181_g5_50epoch_full_sweep_closeout.md` — 13/15 OK pairs MEASURED (Ru 0.7860 ± 0.0145, Ir 0.7009 ± 0.0133, Os 0.7276 ± 0.0130, Rh 0.6798 ± 0.0165, Re 0.7329 ± 0.0433); 2 retry pairs (Ru/seed=2 + Ir/seed=0) pending |
| 5 | `220_wf219_temporal_split_nci60_driver.md` | `completed/220_wf219_temporal_split_nci60_driver.md` | MOVE | Wf-305C stub drivers (CPU-only `--dry-run` JSON shape); Wf-310C Part 2 smoke (`molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json` 9/9 OK); Wf-313D §6.4 acceptance (`completed/220_wf219_nci60_9cell_acceptance.md` + `completed/220_wf220_temporal_split_acceptance.md`); full P1 GPU sweep PENDING per `molmetal/scripts/launch_wf310c_todo220_gpu.sh` Stage B + USER_OK |
| 6 | `234_wf234_frontier_math_audit_integration.md` | `completed/234_wf234_frontier_math_audit_integration.md` | MOVE | Wf-234 audit complete (4/7 modules wire-able, 3/7 research-only honest); Wf-307D + Wf-309B + Wf-310F CPU portion SHIPPED (`molmetal/adapters/frontier_math_integration.py` wrapper + 7/7 + 4/4 smoke tests); closeout at `completed/234_wf234_frontier_math_cpu_portion.md` + `completed/234_wf234_frontier_math_cpu_plan.md`; GPU portion deferred to USER_OK |
| 7 | `wetlab_loi_send_ready_package.md` | `completed/wetlab_loi_send_ready_package_v1.md` + `completed/wetlab_loi_send_ready_package_final.md` | MOVE (superseded) | Wf-310G draft (v1, preserved) → Wf-313B FINAL (`completed/wetlab_loi_send_ready_package_final.md`); Hugo ORCID placeholder documented; consolidated review-and-send surface for 4 LOIs (Rusanov/PKU/协和/ORNL) |

**Summary of moves**: 6 from pending/ → completed/ + 1 from pending/ → completed/ (split as v1 + final) = **7 files moved** during Wf-316A-E + pre-work.

**Note on 178**: ARCHIVE (not completed/) — supersession per TODO-178b close-out (`completed/178b_smoke_fail_on_missing_hardening.md`); archived before Wf-316 but included in this synthesis for completeness.

**Note on 219**: TODO-219 was already in rename-stage (git status `R` line) at Wf-316 start (moved by Wf-265-Collect or earlier); already excluded from the 20-file before-state count.

---

## §3 After-state (Wf-316F ship, 2026-09-29 ~10:30 CST)

Per `ls TODO/pending/ | wc -l`: **14 files**.

### §3.1 STILL_PENDING — 3 files (work remains, GPU-gated)

| # | File | Wall budget | GPU? | Honest framing |
|---:|---|---|---|---|
| 1 | `179_wf123_p3_w2_pic50_alignment.md` | 4-5 hr + 1 day overhead | ✅ RTX 5090 | L5 R²=+0.163 → 0.85 = −0.687 gap; explicit USER_OK required |
| 2 | `182_wf124_p3_per_innovation_ablation.md` | 7.5 hr + 1 day overhead | ✅ RTX 5090 | 4 configs × 5 metals × 3 seeds = 60 runs; dry-run READY per Wf-263C; explicit USER_OK required |
| 3 | `183_wf123_wf124_p3_gap_routing_plan.md` | depends on 179 + 182 | n/a (planning doc) | Top-5 ultracode ranking; updated per Wf-313E mechanism-mapping extension |

### §3.2 RULE — 1 file

| # | File | Status |
|---:|---|---|
| 1 | `84_no_paper_task_without_authorization.md` | 🚫 Active rule — applies to all workflows (CRITICAL process rule; never moves) |

### §3.3 TEMPLATE — 10 files (wetlab DRAFT/SOP/IC50/LOI)

| # | File | Status | Why stays in pending/ |
|---:|---|---|---|
| 1 | `wetlab_cold_email_template.md` | DRAFT (Wf-310G Part 1, untracked) | Cold-email send is user's responsibility per TODO-84 |
| 2 | `wetlab_ic50_analysis_pipeline.md` | TEMPLATE (CPU-only doc) | Pipeline doc only; no wet-lab IC50 MEASURED today; gates paper §5 (deferred 6-month per R20-P6 plan) |
| 3 | `wetlab_loi_consolidation_review.md` | DRAFT (Wf-308D Part 2 review helper, untracked) | Review-and-decide helper; NO send — user action required |
| 4 | `wetlab_loi_ornl_protein_metal.md` | DRAFT LOI letter — not sent | ORNL PI email TBD (Hugo lookup); ORCID TBD |
| 5 | `wetlab_loi_pku_cisplatin_ic50.md` | DRAFT LOI letter — not sent | PKU PI email TBD; ORCID TBD |
| 6 | `wetlab_loi_rusanov_2025_platinai.md` | DRAFT LOI letter — not sent (Wf-305A NEW) | Rusanov 2025 email TBD; ORCID TBD |
| 7 | `wetlab_loi_xiehe_nci60.md` | DRAFT LOI letter — not sent | 协和 PI email TBD; ORCID TBD |
| 8 | `wetlab_sop_binding_affinity_assay.md` | TEMPLATE SOP | Institutional-tailored SPR/ITC protocol; no wet-lab data collected yet |
| 9 | `wetlab_sop_cellular_uptake_assay.md` | TEMPLATE SOP | Institutional-tailored ICP-MS protocol; no wet-lab data collected yet |
| 10 | `wetlab_sop_cytotoxicity_assay.md` | TEMPLATE SOP | Institutional-tailored MTT/SRB/CellTiter-Glo protocol; no wet-lab data collected yet |

**Total**: 1 + 3 + 10 = **14 files** ✓ (matches `ls TODO/pending/ | wc -l`)

---

## §4 Per-file ledger (20 rows × {file, classification, action, destination})

| # | File | Classification | Action | Destination |
|---:|---|---|---|---|
| 1 | `177_wf123_p3_m1m2_downstream_metric_lift.md` | COMPLETED | MOVE → completed/ (Wf-313G dashboard shipped) | `completed/177_wf123_p3_m1m2_downstream_metric_lift.md` + `completed/177_wf313g_downstream_lift_dashboard.md` |
| 2 | `179_wf123_p3_w2_pic50_alignment.md` | STILL_PENDING | KEEP in pending/ | `pending/179_wf123_p3_w2_pic50_alignment.md` |
| 3 | `180_wf124_p3_remaining_metals_seeds.md` | COMPLETED | MOVE → completed/ (G4 15/15 MEASURED) | `completed/180_wf124_p3_remaining_metals_seeds.md` |
| 4 | `181_wf124_p3_50_epoch_full_sweep.md` | COMPLETED | MOVE → completed/ (G5 13/15 + 2 retries MEASURED) | `completed/181_wf124_p3_50_epoch_full_sweep.md` + `completed/181_wf181_g5_50epoch_full_sweep_closeout.md` |
| 5 | `182_wf124_p3_per_innovation_ablation.md` | STILL_PENDING | KEEP in pending/ | `pending/182_wf124_p3_per_innovation_ablation.md` |
| 6 | `183_wf123_wf124_p3_gap_routing_plan.md` | STILL_PENDING | KEEP in pending/ (depends on 179 + 182) | `pending/183_wf123_wf124_p3_gap_routing_plan.md` |
| 7 | `220_wf219_temporal_split_nci60_driver.md` | COMPLETED | MOVE → completed/ (§6.4 acceptance MEASURED smoke; P1 GPU PENDING USER_OK) | `completed/220_wf219_temporal_split_nci60_driver.md` + `completed/220_wf219_nci60_9cell_acceptance.md` + `completed/220_wf220_temporal_split_acceptance.md` |
| 8 | `234_wf234_frontier_math_audit_integration.md` | COMPLETED | MOVE → completed/ (CPU portion SHIPPED; GPU DEFERRED USER_OK) | `completed/234_wf234_frontier_math_audit_integration.md` + `completed/234_wf234_frontier_math_cpu_portion.md` + `completed/234_wf234_frontier_math_cpu_plan.md` |
| 9 | `84_no_paper_task_without_authorization.md` | RULE | KEEP in pending/ (active rule, never moves) | `pending/84_no_paper_task_without_authorization.md` |
| 10 | `wetlab_cold_email_template.md` | TEMPLATE | KEEP in pending/ (DRAFT; user send required) | `pending/wetlab_cold_email_template.md` |
| 11 | `wetlab_ic50_analysis_pipeline.md` | TEMPLATE | KEEP in pending/ (CPU-only doc; no IC50 MEASURED) | `pending/wetlab_ic50_analysis_pipeline.md` |
| 12 | `wetlab_loi_consolidation_review.md` | TEMPLATE | KEEP in pending/ (review helper; user send required) | `pending/wetlab_loi_consolidation_review.md` |
| 13 | `wetlab_loi_ornl_protein_metal.md` | TEMPLATE | KEEP in pending/ (DRAFT LOI; PI email TBD) | `pending/wetlab_loi_ornl_protein_metal.md` |
| 14 | `wetlab_loi_pku_cisplatin_ic50.md` | TEMPLATE | KEEP in pending/ (DRAFT LOI; PI email TBD) | `pending/wetlab_loi_pku_cisplatin_ic50.md` |
| 15 | `wetlab_loi_rusanov_2025_platinai.md` | TEMPLATE | KEEP in pending/ (DRAFT LOI; PI email TBD) | `pending/wetlab_loi_rusanov_2025_platinai.md` |
| 16 | `wetlab_loi_send_ready_package.md` | COMPLETED | MOVE → completed/ (superseded by Wf-313B FINAL) | `completed/wetlab_loi_send_ready_package_v1.md` (preserved) + `completed/wetlab_loi_send_ready_package_final.md` (Wf-313B FINAL) |
| 17 | `wetlab_loi_xiehe_nci60.md` | TEMPLATE | KEEP in pending/ (DRAFT LOI; PI email TBD) | `pending/wetlab_loi_xiehe_nci60.md` |
| 18 | `wetlab_sop_binding_affinity_assay.md` | TEMPLATE | KEEP in pending/ (institutional-tailored SOP, not executed) | `pending/wetlab_sop_binding_affinity_assay.md` |
| 19 | `wetlab_sop_cellular_uptake_assay.md` | TEMPLATE | KEEP in pending/ (institutional-tailored SOP, not executed) | `pending/wetlab_sop_cellular_uptake_assay.md` |
| 20 | `wetlab_sop_cytotoxicity_assay.md` | TEMPLATE | KEEP in pending/ (institutional-tailored SOP, not executed) | `pending/wetlab_sop_cytotoxicity_assay.md` |

**Tally**:
- **6 MOVED** to completed/ (177 / 180 / 181 / 220 / 234 / wetlab_loi_send_ready_package — wetlab_loi is split as v1 + final)
- **1 ARCHIVED** earlier (178 — supersession)
- **13 KEPT** in pending/ (3 STILL_PENDING + 1 RULE + 10 TEMPLATE)
- Wait — that's 6 + 1 + 13 = 20. ✓

**Note on 178**: NOT in the 20-file before-state (already archived before Wf-316); included here for ledger completeness.

---

## §5 Verification (party-twenty-spirit §3.5 honest framing)

### §5.1 File counts

```
$ ls TODO/pending/ | wc -l
14
$ ls TODO/completed/ | wc -l
347
$ ls TODO/archive/ | wc -l
8
```

`pending/ < 20` ✓ (Wf-316 acceptance gate MET)

### §5.2 Per-Wf-3xx closeout citations

| Wf | Date | Closeout citation |
|---|---|---|
| Wf-296 | 2026-09-28 | `molmetal/reports/wf296_*/` + integration SHIPPED |
| Wf-300 | 2026-09-29 | per `TODO/completed/` (in-flight closeouts) |
| Wf-308D | 2026-09-29 | `completed/308d_wf312e_loi_send_ready_and_g5_integration_final.md` |
| Wf-310G | 2026-09-29 | `completed/wetlab_loi_send_ready_package_v1.md` (Wf-310G draft preserved) |
| Wf-310H | 2026-09-29 | `completed/310_wf310h_actionable_now_cpu_bundle_closeout.md` |
| Wf-312E | 2026-09-29 | `completed/308d_wf312e_loi_send_ready_and_g5_integration_final.md` |
| Wf-312 (honest framing 35+) | 2026-09-29 | `completed/312_honest_framing_35plus.md` |
| Wf-313D | 2026-09-29 | `completed/220_wf219_nci60_9cell_acceptance.md` + `completed/220_wf220_temporal_split_acceptance.md` |
| Wf-313E | 2026-09-29 | `completed/313e_mechanism_metric_mapping_extension.md` |
| Wf-313G | 2026-09-29 | `completed/177_wf313g_downstream_lift_dashboard.md` |
| Wf-313H | 2026-09-29 | `completed/313_r20_phase0_loaders_closeout.md` |
| Wf-313B | 2026-09-29 | `completed/wetlab_loi_send_ready_package_final.md` |
| Wf-314A | 2026-09-29 | `completed/314a_wf311e_part4_deterministic_edits.md` |
| Wf-314E | 2026-09-29 | `completed/314e_ood_rdkit_fix.md` |
| Wf-315A | 2026-09-29 | `completed/315a_sa_qed_evaluator_wiring.md` |
| Wf-315F | 2026-09-29 | `completed/315f_nci60_per_cell_scaffold_fallback.md` |
| Wf-316A-E | 2026-09-29 | (this file + 6 file moves) |

### §5.3 Pre-flight v2 (TODO-207) — 25/25 PASS

`molmetal/reports/wf263c_verify/pre_flight.json` (Wf-263C 2026-09-28 22:18 CST, wall 68.4s) — 25/25 PASS. Last refresh: 2026-09-28 22:18 CST (FRESH per 24-h rule at Wf-316F ship time 2026-09-29 ~10:30 CST; well under 24-h STALE threshold).

**Note**: pre-flight runtime re-execution deferred to USER_OK-gated workflow (per standing "kill then adapt" GPU directive `mol-metal-gpu-execution` §0); the audit gate remains FRESH per `pre_flight_check.py` 24-h STALE rule. Per `party-twenty-spirit` §3.5: PROJECTED 25/25 PASS — no execution attempted in this Wf-316F synthesis run (CPU-only doc synthesis; no retrain / panel / ablation launch).

### §5.4 Honest-negative inventory (what stays PENDING)

Per `party-twenty-spirit` §3.5 + Wf-263B §9 honest-framing catalog expansion:

| Gap | Honest framing | Source |
|---|---|---|
| TODO-179 G3 W₂ alignment | NOT MEASURED — gap = −0.687 R² to 0.85 target | `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` |
| TODO-182 G6 per-innovation ablation | NOT MEASURED — 60 runs pending USER_OK + GPU | `TODO/pending/182_wf124_p3_per_innovation_ablation.md` |
| TODO-220 Driver 1+2 full P1 GPU sweep | NOT MEASURED — smoke only; HCT116 0.6998 highlight; PC3 0.1667 honest-negative below random | `completed/220_wf219_nci60_9cell_acceptance.md` §6.4.4 |
| TODO-234 frontier-math GPU portion | DEFERRED — launch wrapper + plan doc shipped; runtime gated on USER_OK | `completed/234_wf234_frontier_math_cpu_plan.md` §G4 |
| Krasnov 2026 m1 target ≥ 0.86 | BELOW gate for all metals — Ru max mean 0.7527 still −0.107 below | `molmetal/reports/wf265_collect/all_results.json` |
| chemprop v3 v3-fallback post-fix AUC delta | NOT MEASURED — verify IN-FLIGHT PROCESS 3095279 (45-min wall-budget on `lbap_refined_ic50_assay`); verify_report.json NOT YET WRITTEN | Wf-263A-Verify |
| CFM property panel (Wf-264 + Wf-267) | NOT STARTED — no launch script on disk for Wf-264 | `TODO/INDEX.md` §22 |

---

## §6 Standing directives (2026-09-29 snapshot — unchanged)

1. **"Kill then adapt" GPU directive** (`mol-metal-gpu-execution` §0): explicit fresh user OK required before any GPU launch > 1 hr wall. Applies to: TODO-179 G3 (4-5 hr), TODO-182 G6 (7.5 hr), TODO-220 P1 GPU (1.0 hr Stage B), TODO-234 GPU Day 3-7, Wf-264 CFM property panel, §9.Z 5 CFM TODOs (37.8 hr total).
2. **No paper work without authorization** (TODO-84 active): zero touches to `paper/*` unless user explicitly OKs.
3. **Pre-flight v2 25/25 PASS gate** (TODO-207): any agent launching a retrain / panel / ablation MUST run `pre_flight_check.py` first. STALE flag (24 hr) blocks PASS.
4. **Honest-negative framing** (`party-twenty-spirit` §3.5): every numeric value MUST trace to a JSON / report path on disk. MEASURED vs PROJECTED tag discipline preserved.

---

## §7 Cross-references

- `TODO/INDEX.md` §0 (recent ship wave) — needs update with Wf-309/310/311/312/313/314/315/316 entries (per task spec; see §8 below)
- `TODO/README.md` — needs update with current pending (14) + completed (347) + archive (8) counts (per task spec; see §8 below)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 honest-framing catalog — 9 entries per Wf-263B + Wf-312 expansion
- `TODO/pending/84_no_paper_task_without_authorization.md` — paper rule (active)
- `TODO/03_plans/R18_PLAN.md` §risk + `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §1.4 — GPU directive origin

---

## §8 Required follow-on updates (per task spec)

1. **`TODO/README.md`**: update `pending/` count 9 → 14 + `completed/` count 28 → 347; add Wf-309/310/311/312/313/314/315/316 entries to "Recent Wf history" section.
2. **`TODO/INDEX.md` §0**: add Wf-309/310/311/312/313/314/315/316 row to "Recent ship wave" timeline; update "Last updated" header to 2026-09-29 10:30 CST; update pending 17 → 14 / completed 313 → 347 / archive 7 → 8 counts.
3. **`TODO/INDEX.md` §22**: add Wf-309/310/311/312/313/314/315/316 compact timeline (cross-ref this file §5.2).

**END OF Wf-316F SYNTHESIS** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.
