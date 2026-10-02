# Wf-293 Synthesis — followup_summary

**Date:** 2026-09-29
**Wave:** Wf-293 (Wf-288D hygiene nits N1/N2 + Wf-279C/D retries)
**Owner:** Hugo Lee

---

## §0 TL;DR

| Agent | Task | Status | Evidence |
|---|---|---|---|
| Wf-293A | nit_1 yaml flag status (Wf-288D N1) | **COMPLETED** | `configs/wf124_p3_per_innovation_ablation_g6.yaml` lines 103+113 now show `status: "verified-by-wf283b+wf285b"` (was `"verify-flag-before-launch"`); grep for stale flag returns zero hits |
| Wf-293B | nit_2 --no-enforce-budget (Wf-288D N2) | **COMPLETED** | `wf124_p3_50epoch_full_sweep_g5.yaml` line 222 + `wf124_p3_per_innovation_ablation_g6.yaml` line 252 both carry `--no-enforce-budget` inside `entry.command` block (after `--ckpt-dir`) |
| Wf-293C | Wf-279D retry — launcher safety trio verify | **COMPLETED** | `molmetal/reports/wf293c_launcher_safety/verify_report.md` §5: `guard_present_before = TRUE`, `guard_present_after = TRUE`, `dry_run_verdict = PASS`; P0 BLOCKER (Wf-288B) cleared |
| Wf-293D | Wf-279C retry — paper_dashboard.py refresh | **COMPLETED** | `molmetal/reports/wf293d_dashboard_refresh/refresh_report.md` §0: 34 claims (+1), 18 honest-negatives (+1), 7 cohorts (+1), 80.9 KB HTML (+4.0 KB); new `claim_wf286b_ecfp4_axis_ruled_out` + `hn_wf286a_5_seed_drugood_failed` + `wf286b_ecfp4_direct` cohort |

**4/4 agents completed. 0 failed. 0 partial.**

---

## §1 Per-agent summary

### §1.1 Wf-293A — nit_1 yaml flag status

- **Ref:** Wf-288D hygiene nit N1; Wf-290-I item #9
- **Scope:** Update `configs/wf124_p3_per_innovation_ablation_g6.yaml` lines 103+113 status fields from `"verify-flag-before-launch"` → verified form
- **Verification:**
  - Read `configs/wf124_p3_per_innovation_ablation_g6.yaml` lines 90-115: both `--no-kendall` (line 103) and `--no-dann` (line 113) configs now carry `status: "verified-by-wf283b+wf285b"`
  - Repo-wide grep for `"verify-flag-before-launch"` returns zero matches (stale string fully purged)
- **Verdict:** FIX APPLIED — yaml flag status reflects post-Wf-283B + Wf-285B verification reality

### §1.2 Wf-293B — nit_2 --no-enforce-budget CLI flag

- **Ref:** Wf-288D hygiene nit N2; Wf-280A no-enforce-budget directive
- **Scope:** Append `--no-enforce-budget` to entry.command blocks of `wf124_p3_50epoch_full_sweep_g5.yaml` (lines 208-221) + `wf124_p3_per_innovation_ablation_g6.yaml` (lines 239-251)
- **Verification:**
  - G5 line 222 (immediately after --ckpt-dir) carries `--no-enforce-budget` as the final command-line token
  - G6 line 252 (immediately after --ckpt-dir) carries `--no-enforce-budget` as the final command-line token
- **Verdict:** FIX APPLIED — both G5/G6 launch configs honor user 2026-09-28 instruction "时间预算没关系的"

### §1.3 Wf-293C — Wf-279D retry launcher safety trio

- **Ref:** Wf-288B P0 BLOCKER; Wf-279D §5 cross-cutting bug; Wf-283C guard wire; Wf-280D recommendation #1
- **Scope:** Verify all 3 Wf-124 P3 launchers (G4/G5/G6) wire the Wf-262 pre-launch safety trio (pre_flight_check + per_wave_checklist + wave_budget_tracker)
- **Verification:**
  - G4 (117 lines, bash -n OK): guard block lines 62-66, all 3 trio calls present
  - G5 (157 lines, bash -n OK): guard block lines 81-85, all 3 trio calls present
  - G6 (186 lines, bash -n OK): guard block lines 111-115, all 3 trio calls present
  - Wf-262 trio scripts (pre_flight_check.py 32179 B, per_wave_checklist.py 36180 B, wave_budget_tracker.py 42824 B) all present + executable
  - G5 dry-run: PASS (15 metal×seed pairs, CSV 26633 rows × 11 cols OK)
  - G6 dry-run: PASS (60 config×metal×seed tuples, 30 CONFIRMED + 30 UNVERIFIED-skip-if-real)
  - G4 dry-run: N/A by design (no --dry-run mode; trio runs unconditionally before Wave A)
- **Verdict:** GUARD INTACT — zero edits required; P0 BLOCKER cleared

### §1.4 Wf-293D — Wf-279C retry paper_dashboard.py refresh

- **Ref:** Wf-279C baseline; Wf-286A/B/D post-Wf-285/286/287/288 evidence
- **Scope:** Refresh paper dashboard catalog on top of Wf-280C baseline (33/17/6) with new Wf-286A/B/D claims + honest-negatives + cohort
- **Verification:**
  - Schema 280.v1 preserved (no breaking changes)
  - Counter delta: 33→34 claims (+1), 17→18 honest-negatives (+1), 6→7 cohorts (+1)
  - HTML delta: 76.9 KB → 80.9 KB (+4.0 KB)
  - New claim: `claim_wf286b_ecfp4_axis_ruled_out` (§6.3, MEASURED, test_auc=0.8820 Δ=+0.2113 vs Wf-185 0.6707 baseline)
  - New honest-negative: `hn_wf286a_5_seed_drugood_failed` (§9.AK, gate_1_all_ran=false, gate_2_total_runs=0)
  - New cohort: `wf286b_ecfp4_direct` (ECFP4 + chemprop v2 BinaryClassificationFFN, 1-seed × 5-ep, CPU, wall=190.9s)
  - Note: `claim_wf269c_yuelbond_load_bearing` was ALREADY in catalog from Wf-279C (lines 393-400 of pre-edit); Wf-293D-specific delta is +1, not +2
- **Verdict:** REFRESH SUCCEEDED — dashboard now reflects Wf-286A/B evidence + remains backward-compatible

---

## §2 Critical remaining

After Wf-293 wave, the residual critical backlog is:

| # | Item | Severity | Source | Estimated effort |
|---|---|---|---|---|
| 1 | Wf-286B-Full (5-seed × 20-ep ECFP4 + BinaryClassificationFFN) on `lbap_core_ic50_assay` to pin Δ ± pooled-std | HIGH | Wf-286 synthesis §4 P2; Wf-293D §8 next-step | ~3-4 hr GPU |
| 2 | chemprop v2 MPNN baseline on `lbap_core_ic50_assay` matching-subset to Wf-286B-Full | HIGH | Wf-286 synthesis §4 P2 | ~3-4 hr GPU |
| 3 | TODO-130 G5 sign-flip audit update after Wf-286B-Full MEASURED | MEDIUM | Wf-293D §8 | ~30 min |
| 4 | Dashboard re-refresh after Wf-286B-Full to add 5-seed cohort row | LOW | Wf-293D §8 | ~10 min |

---

## §3 Next priorities

1. **Wf-294 + Wf-295**: launch Wf-286B-Full (5-seed × 20-ep) + chemprop v2 matching-subset baseline on `lbap_core_ic50_assay` in parallel on GPU-1 (kanzi). Both target Δ-vs-Wf-185-baseline with pooled-std, will pin down remaining hypotheses (chemprop v3 MPNN backbone vs v3-fallback path config vs DrugOOD JSON splits).
2. **Wf-296**: post-Wf-286B-Full MEASURED, run TODO-130 G5 sign-flip audit on chemprop_v2 + chemprop_v3 + ECFP4 axes; refresh dashboard with 5-seed cohort row.
3. **Wf-297** (paper-table gate): after Wf-296 sign-flip clears, fold final chemprop-v2 + ECFP4 5-seed numbers into paper §6.3 / §9.AK tables.
4. **Wf-298** (final draft signoff): the Wf-289 signoff pattern — re-run TODO/03_plans/LAMBDA_ROADMAP.md §5/§7 checkpoints; if all green, move paper §3 + §4 + §6 to arxiv-ready.

---

## §4 Aggregate ledger

```
Wf-293 wave:  4/4 completed,  0 failed,  0 partial
Critical remaining: 4 items (2 HIGH, 1 MEDIUM, 1 LOW)
Next wave:    Wf-294 + Wf-295 (parallel GPU retrain)
```

*Generated by Wf-293E (2026-09-29). Read-only synthesis. No edits made.*