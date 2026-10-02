# Wf-377-A4: GPU launch dashboard final state + final pre-flight verification

**Status**: SHIPPED 2026-09-29T16:48 UTC wall ~5 min CPU-only (docs-only + JSON refresh + 1 NEW test)
**Wave**: Wf-377 (m6_kappa redesign + m2 R² diagnostic + final consolidation)
**Origin**: Per Wf-360-A3 + Wf-363-B1 + Wf-363-B2 + Wf-364-B2 + Wf-365-B1 + Wf-368-B1 dashboard evolution

## §1 Task

Per computed task brief:

> **Wf-377-A4: GPU launch dashboard final state + final pre-flight verification (READ + minimal refresh).**
>
> USE Bash + Read + Bash (curl + jq).
>
> ENVIRONMENT CONSTRAINTS (per user directive 2026-09-30):
> - NO GPU
> - NO paper/* writes
> - NO sudo
> - NO wet-lab planning

5 sub-steps:
1. Read `scripts/gpu_launch_dashboard.sh` + `molmetal/scripts/gpu_launch_aggregator.py` (existing infrastructure)
2. Run final pre-flight via `python molmetal/scripts/pre_flight_check.py --pipeline all`
3. Read final dashboard status at `molmetal/reports/gpu_launch_dashboard.json`
4. Refresh dashboard JSON with Wf-372 INSTALL + Wf-371+374+375+376 SHIPPED + pre-flight 89/2/3/94 + 7 USER_OK-gated launches
5. Add 1 NEW test verifying dashboard JSON exists + has per_pair_progress + budget_status + pre-flight state recorded correctly

## §2 Existing infrastructure (READ + verified)

- `scripts/gpu_launch_dashboard.sh` (44511 bytes / 787 LOC bash+python-heredoc) — READ-ONLY aggregator, never launches GPU job, USER_OK-gated per Wf-364-B2
- `molmetal/scripts/gpu_launch_aggregator.py` (27214 bytes / 245 LOC) — JSON aggregator with `--status-summary` + `--activate` + `--budget-summary` flags per Wf-363-B1 + Wf-363-B2 + Wf-364-B2
- `molmetal/reports/gpu_launch_dashboard.json` — output JSON file (READ + refresh)

## §3 Final pre-flight verification

**Run**: `PYTHONPATH=. python molmetal/scripts/pre_flight_check.py --pipeline all` at 2026-09-29T16:47:42+0000 UTC
**Wall**: 148.2856 s
**Output**: `molmetal/reports/wf377a4_preflight.txt`

**Result**: **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** PRESERVED at Wf-353-B1 baseline
- Per-pipeline sub-counts: `chemprop_v2_v3=14/14` + `cfm=6/6` + `frontier=3/3` + `all=94`
- 2 FAILs are pre-existing per Wf-350-A2 §5 + Wf-345-D1 §3 honest-negative baseline (NOT Wf-377-A4-introduced)
- 3 SKIPs are BY-DESIGN: `p3_wf343c4_wave_budget_remaining_nonneg` (per Wf-262 §7.1 with −183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency) + 1 wave-budget gate structural SKIP

## §4 Dashboard refresh

**Schema version**: 1.0 → **1.1** (Wf-377-A4 bumped)
**Timestamp**: 2026-09-29T16:47:58 UTC

### §4.1 install_state (NEW field — Wf-372)

- **zotero**: installed=True, version=`140.15.0esr`, path=`/home/hugo/.local/bin/zotero`
- **tectonic**: installed=True, version=`0.17.0`, path=`/home/hugo/.local/bin/tectonic`
- **csl_integration**: Wf-372-B1 SHIPPED at `TODO/completed/372b1_zotero_csl_integration.md`
- **sn_jnl_template**: Wf-372-B2 SHIPPED at `TODO/completed/372b2_tectonic_sn_jnl.md`

### §4.2 recent_wf_shipped (NEW field — Wf-371..377 inventory)

- **Wf-371**: 6 closeouts on disk (m6_kappa degenerate + 365a2 + 368c1 standalone authored + b1 final synthesis + index_sync + 371_final_synthesis)
- **Wf-372**: 6 closeouts (zotero + tectonic install + CSL integration + sn-jnl template + index_sync + 372_final_synthesis)
- **Wf-373**: 5 closeouts (closeout_inventory + Wf-371 xverify + Wf-372 xverify + index_sync + 373_final_synthesis)
- **Wf-374**: 5 closeouts (zotero_linux_standalone + tectonic + 371b1_final_synthesis_authored + index_sync + 374_final_synthesis)
- **Wf-375**: 4 closeouts (catalog_wetlab + lambda_roadmap_verify + index_sync + 375_final_synthesis)
- **Wf-376**: 4 closeouts (m1_multiseed_scaffold + index_sync + 376_final_synthesis + 1 inline wetlab deprecation edit wave)
- **Wf-377**: 1 closeout (377a1_m6kappa_redesign SHIPPED 2026-09-30)

### §4.3 preflight_state (NEW field — Wf-377-A4 pre-flight)

- **timestamp**: 2026-09-29T16:47:42+0000 UTC
- **pass/fail/skip/total**: 89/2/3/94
- **wall_sec**: 148.2856
- **expected_gate_counts_all**: 94 PRESERVED at Wf-353-B1 baseline
- **per_pipeline**: `{chemprop_v2_v3: 14, cfm: 6, frontier: 3, all: 94}`
- **verdict**: PRESERVED at Wf-353-B1 baseline
- **fails_preserved**: pre-existing Wf-350-A2 §5 + Wf-345-D1 §3 honest-negative baseline

### §4.4 7 USER_OK-gated launchers (UNCHANGED — all still USER_OK-gated)

| Launcher | Status | Wall_proj | Notes |
|---|---|---:|---|
| wf355_a1 | DRY_RUN_OK | 10.0h | DrugOOD 1.58M chemprop_v2 |
| wf355_a2 | DRY_RUN_OK | 7.0h | BindingDB 1.07M |
| wf355_a3 | DRY_RUN_OK | **12.0h OVER** | tmQM+PlatinAI joint retrain |
| wf355_b1 | PLAN_ONLY | 5.0h | Wf-131 P3 7 DEFERRED runs |
| wf355_b2 | DRY_RUN_OK | 3.0h | Wf-146 chemprop LOMO |
| g5_retrain | PLAN_ONLY | 5.0h | 15-pair full sweep |
| g6_recover | PARTIAL | **11.4h OVER** | 31 COMPLETED + 29 FAILED |

**Total projected**: 53.4h vs 10.0h session cap = -43.4h total remaining
**2 launchers OVER** (wf355_a3 + g6_recover): need cross-session scheduling or USER_OK waiver per AGENTS.md §5.1

## §5 NEW test: `molmetal/tests/test_wf377a4_dashboard_final.py`

**5 tests PASS** (in 0.10 s wall CPU-only):

1. `test_dashboard_json_exists_with_per_pair_progress` — dashboard JSON exists + has per_pair_progress per launcher
2. `test_budget_status_field_present` — budget_status with session_cap_hr=10.0 + per_launcher + total_projected_hr + total_remaining_hr + honest_framing
3. `test_preflight_state_recorded_correctly` — pre-flight 89/2/3/94 + wall_sec < 300 + per_pipeline counts
4. `test_install_state_zotero_tectonic` — Zotero 140 + tectonic 0.17 INSTALLED
5. `test_dashboard_activation_gated_user_ok_required` — activation_gated=True + user_ok_required=True + dashboard_mode=READ-ONLY_SCAFFOLD + activation_flag_set=False

## §6 Honest framing per party-twenty-spirit §3.5

This wave is **READ + refresh + 1 NEW test**:

- **NO GPU launches** performed across the entire Wf-377-A4 task (per AGENTS.md §5.1 + kill_then_adapt §0)
- **NO `paper/*` touches** across the entire Wf-377-A4 task (per AGENTS.md §5.3 TODO-84 active)
- **NO sudo** invoked (only `which zotero tectonic` + `zotero --version` + `tectonic --version` read-only probes)
- **NO wet-lab planning** added (per AGENTS.md §5.4 Tier-2 markers UNCHANGED)

## §7 Verification verdict

- Dashboard JSON refreshed at `molmetal/reports/gpu_launch_dashboard.json` (schema v1.1)
- 5 NEW tests PASS in 0.10 s wall CPU-only at `molmetal/tests/test_wf377a4_dashboard_final.py`
- Pre-flight 89/2/3/94 PRESERVED at Wf-353-B1 baseline
- 7 USER_OK-gated launchers all status preserved (DRY_RUN_OK ×4 + PLAN_ONLY ×2 + PARTIAL ×1)
- 2 launchers OVER session cap (wf355_a3 + g6_recover) — need cross-session scheduling or USER_OK waiver
- Zotero 140.15.0 + Tectonic 0.17.0 INSTALLED per Wf-372 (with Wf-374-A1 zotero linux standalone follow-up)
- Wf-371 + Wf-372 + Wf-373 + Wf-374 + Wf-375 + Wf-376 + Wf-377-A1 closeouts all SHIPPED on disk (30 closeouts total)
- 1 SHIPPED sub-flow (this A4) + 0 GPU launches + 0 paper/* touches + 5 NEW tests + ~150 LOC NEW source + 1 dashboard JSON refresh

**End of Wf-377-A4 closeout** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; READ + refresh wave for final-state confirmation.