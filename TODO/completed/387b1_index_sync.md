# Wf-387-B1 — INDEX + AGENTS + final synthesis sync (this wave's B1 closeout)

**Date:** 2026-09-30 (Wf-387-B1 timestamp)
**Wave:** Wf-387 ultracode CPU-only docs+aggregation wave (3 sub-flows A1+A2+A3 SHIPPED 100%)
**Sub-flow:** Wf-387-B1 (docs-only INDEX + AGENTS + final synthesis sync)
**Task reference:** User voice relay from Wf-389-B1: *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* (Wf-387 wave parent directive — same ultracode family as Wf-330 / Wf-335 / Wf-336 / Wf-341 / Wf-347 / Wf-348 / Wf-349 / Wf-354 / Wf-357 / Wf-359 / Wf-360 / Wf-362 / Wf-363 / Wf-364 / Wf-365 / Wf-366 / Wf-367 / Wf-368 / Wf-369 / Wf-370 / Wf-371 / Wf-372 / Wf-373 / Wf-374 / Wf-375 / Wf-376 / Wf-377 / Wf-378 / Wf-379 / Wf-380 / Wf-381 / Wf-383 / Wf-384 / Wf-385 / Wf-386 / Wf-388 / Wf-389 ultracode).

---

## §0 TL;DR

Wf-387-B1 is a docs-only sync that aggregates the 3 SHIPPED Wf-387 sub-flow closeouts
(Wf-387-A1 + Wf-387-A2 + Wf-387-A3) into the canonical AGENTS.md + TODO/INDEX.md project
state. This wave DID彻底解决 three internal-metric-improvement gaps per user directive 2026-09-30
verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*:

1. **m1 AUC GPU aggregation fix** (Wf-387-A1) — the canonical Tier-2 catalog
   `metrics/by_metric/metal_lomo_auc_wasserstein.json` was carrying the WRONG n=1 single-fold
   Re-only summary (value=0.6220 from Wf-176 restore). Wf-387-A1 fixes this by computing the
   correct 6-run aggregate across all completed (metal, seed) pairs from the Wf-379-A1 GPU
   launch: value **0.7640** (6 runs, 2 metals Rh+Re, std 0.0995). Per-fold: Rh 0.8798 ± 0.0115
   (PASS Krasnov 0.86) + Re 0.7061 ± 0.0552 (BELOW Krasnov 0.86).

2. **TODO-135 BALANCED Rh cohort from REAL MetalCytoToxDB** (Wf-387-A2) — a balanced 500-compound
   Rh cohort (250 active + 250 inactive) sourced from REAL MetalCytoToxDB compounds under the
   Wf-261 strict scaffold split, with 0 scaffold leak to Rh val or Rh test, with structural
   attestations (`is_synthetic=False`, `wf139_p7_carryover=False`) on every entry. This is the
   verification anchor for the Wf-125 P1 val_auc 0.7613 lift (the lift may be artifact of the
   Wf-139 P7 synthetic 500 injection, not real bioactivity).

3. **Balanced Rh benchmark scaffold** (Wf-387-A3) — `molmetal/scripts/wf387a3_balanced_rh_bench.py`
   674 LOC + `molmetal/tests/test_wf387a3_balanced_rh_bench.py` 220 LOC + smoke verdict
   `ARTIFACT_LIFT` on inline synthetic 60-SMILES cohort (AUC=0.5 random — pipeline-validation PASS,
   real-cohort verdict PENDING Wf-387-A2 balanced Rh cohort + USER_OK-gated GPU launch). 8/8 NEW
   tests PASS in ~0.02 s wall CPU-only.

**Paper §4 verdict (per Wf-387-A1 §3)**:
- **Rh fold MEASURED_PASS_KRASNOV** (0.8798 vs 0.86 gate; gap +0.020) — paper §4 LEAD-claim candidate
- **Re fold MEASURED_BELOW_KRASNOV** (0.7061 vs 0.86 gate; gap −0.154) — paper §6 honest-negative preserved
- **Overall m1_auc 0.7640 BELOW_KRASNOV_PARTIAL** — Rh PASS, Re BELOW

---

## §1 Pre-flight verification

Pre-flight v3 fresh-run was NOT re-run by Wf-387-B1 (CPU-only docs+aggregation wave per
AGENTS.md §5.1 + kill_then_adapt §0). Pre-flight cascade PRESERVED at Wf-385-B1 baseline
**89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** per the prior Wf-389-B1 sync (preserved at
Wf-385-B1 baseline 2026-09-29T17:17:10+0000 UTC wall ~3 min). HONEST-NEGATIVE DRIFT
PRESERVED at 89/2/3/94 matches Wf-374-B1 through Wf-389-B1 baseline at same drift level
(15+ consecutive runs PRESERVED at 89/2/3/94).

2 FAILs preserved (NOT Wf-387-introduced):
- `p3_loi_send_ready` FAIL — Tier-2 reclassification cascade from 2026-09-30 wetlab
  deprecation inline edits (per AGENTS.md §5.4)
- `p3_wf343c4_anchor_freshness_24h` FAIL — `molmetal/reports/wf269c_report.json` age >
  24h STALE — natural drift across Wf-374..389 wave activity

3 SKIPs preserved (BY-DESIGN):
- `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2)
- `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun)
- `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)

`gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-recomputed via Wf-345-B3)`.

EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight
gates added by Wf-387 — 3 sub-flows + 8 NEW tests + 1 INDEX sync only).

---

## §2 INDEX.md sync

Appended **§22.77** Wf-387 entry to `TODO/INDEX.md` after `### §22.5.4 Cross-link` block
(Wf-389 entry). The §22.77 entry captures the 3 SHIPPED sub-flows (Wf-387-A1 + Wf-387-A2 +
Wf-387-A3) + this docs-only B1 sync + cross-refs to:

- `metrics/by_metric/metal_lomo_auc_wasserstein.json` (Wf-387-A1 UPDATED canonical Tier-2
  catalog with 6-run aggregate)
- `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` (Wf-387-A2 NEW
  balanced 500-cohort artifact)
- `molmetal/scripts/wf387a3_balanced_rh_bench.py` (Wf-387-A3 NEW benchmark scaffold 674 LOC)
- `molmetal/tests/test_wf387a3_balanced_rh_bench.py` (Wf-387-A3 NEW 8-test suite)
- `molmetal/reports/wf387a3_balanced_rh_smoke/smoke_result.json` (Wf-387-A3 MEASURED smoke
  output, AUC=0.5 random ARTIFACT_LIFT verdict on inline synthetic cohort)

---

## §3 AGENTS.md sync

Appended NEW §6 row to AGENTS.md (between Wf-386 row at line 545 and Wf-385 trailer at
line 547) for Wf-387:

- Format: Wf-387 GPU aggregation fix + balanced Rh cohort + balanced Rh CPU bench + docs sync
- Pre-flight 89/2/3/94 PRESERVED (NOT re-run by Wf-387 — CPU-only docs+aggregation)
- 0 NEW pre-flight gates added by Wf-387
- 8 NEW tests added by Wf-387-A3 (per `molmetal/tests/test_wf387a3_balanced_rh_bench.py`)
- 3 metric JSONs / artifact files updated (canonical m1_auc catalog + balanced_rh_cohort.json
  + smoke_result.json)
- 1 backup file created (`metal_lomo_auc_wasserstein.json.bak.wf387a1`)
- 2 NEW source code files (`molmetal/scripts/wf387a3_balanced_rh_bench.py` 674 LOC +
  `molmetal/tests/test_wf387a3_balanced_rh_bench.py` 220 LOC)

**last-update trailer extended** with Wf-387-B1 sync clause (following the existing
pattern from Wf-374-B1 + Wf-375-B1 + ... + Wf-389-B1 prior syncs).

---

## §4 IMPROVEMENT_ENUM + LAMBDA_ROADMAP sync

Neither `IMPROVEMENT_ENUM_2026_09_27.md` nor `LAMBDA_ROADMAP.md` required edits by
Wf-387. The Wf-387 wave is an **internal-metric-improvement** wave (m1_auc canonical catalog
fix + TODO-135 balanced Rh cohort + benchmark scaffold), not a research/planning layer
improvement, so it does not appear in either plan file directly.

However, the Wf-387-A2 balanced Rh cohort + Wf-387-A3 benchmark scaffold collectively
provide the **computational verification path** for the Wf-125 P1 / Wf-139 P7 honest-negative
entry that was DEFERRED per `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-358-C2 expansion). The
verification path requires USER_OK=1 + Wf-388 follow-up GPU retrain (Wf-125 P1 with Wf-387-A2
real 500 substituted for Wf-139 P7 synthetic 500 + READOUT to Rh val).

Cross-references:
- `AGENTS.md` §6 NEW row (this sync)
- `TODO/INDEX.md` §22.77 NEW entry (this sync)
- `metrics/by_metric/metal_lomo_auc_wasserstein.json` UPDATED (Wf-387-A1)
- `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` NEW (Wf-387-A2)
- `molmetal/scripts/wf387a3_balanced_rh_bench.py` NEW (Wf-387-A3)
- `molmetal/tests/test_wf387a3_balanced_rh_bench.py` NEW (Wf-387-A3)

---

## §5 Honest framing (per party-twenty-spirit §3.5)

### §5.1 What Wf-387 DID彻底解决 (m1 AUC GPU aggregation + TODO-135 balanced Rh + paper §4 gate verdict)

1. **The Wf-379-A1 GPU aggregation fix** is closed by Wf-387-A1 — the canonical Tier-2 catalog
   `metrics/by_metric/metal_lomo_auc_wasserstein.json` previously showed `value=0.6220 n_runs=1`
   (Wf-176 Re seed=4 only). Wf-379-A1 Re seed=3 trainer wrote `value=0.6295 n_runs=1` (its own
   data only), but this was overwritten back to the Wf-176 value by a later restore. Wf-387-A1
   rewrites with correct 6-run aggregate covering all MEASURED pairs.

2. **The TODO-135 balanced Rh cohort** is closed by Wf-387-A2 — a balanced 500-cohort Rh
   augmentation (250 active + 250 inactive) sourced from REAL MetalCytoToxDB compounds under
   the Wf-261 strict scaffold split. The artifact is the structural fix for the Wf-125 P1 /
   Wf-139 P7 synthetic-500 honest-negative entry. The 3-run CONTROLD PAIR per Wf-387-A2 §5.1
   (real 500 vs synthetic 500 vs no augmentation) is the verification protocol.

3. **The balanced Rh benchmark scaffold + smoke validation** is closed by Wf-387-A3 — the
   pipeline produces valid output (ARTIFACT_LIFT verdict on inline synthetic cohort; AUC=0.5
   random confirms pipeline doesn't crash; correct verdict taxonomy + honest_framing block
   populated). 8/8 NEW tests PASS in ~0.02 s wall CPU-only.

### §5.2 What Wf-387 did NOT彻底解决 (preserved honest-negatives)

1. **Re seed=4 NOT_RUN** — GPU wrapper terminated after Re seed=3; Re seed=4 not launched.
   Closure path = follow-up launch per Wf-379-A1 `launch.log` final state (~80 min on RTX 5090).

2. **Wrapper-format `per_seed_metrics.json` files have SEED-FIELD-SWAP BUG** — 4 of 6 wrapper
   files (Rh/seed3, Rh/seed4, Re/seed0, Re/seed1) have seed-field labels SWAPPED between paired
   directories. The flat `per_seed_metrics_seed{n}.json` files are canonical and labels are
   correct, but the wrapper files are misleading. **Recommendation**: delete or correct the
   wrapper-format files; emit only the flat files going forward. Deferred to follow-up Wf.

3. **Re fold MEASURED_BELOW_KRASNOV** — Re fold mean 0.7061 ± 0.0552 is below Krasnov 0.86 by
   −0.154. Re cohort is the smallest (n=868 test set). Per-fold R² ceiling on this small
   cohort is the structural constraint.

4. **m1_auc overall 0.7640 BELOW_KRASNOV_PARTIAL** — Rh PASS but Re BELOW; paper §4 LEAD-claim
   candidate is Rh fold only (per Wf-387-A1 §3).

5. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-381-A4 +
   Wf-386-A1 cross-poll + Wf-377-A1 deep-dive-research move; 9 paths exhausted; best aug_mean
   MEASURED 0.472 vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but
   threshold-reaching axis NOT lifted). Wf-387-A1 + A2 + A3 do NOT touch m6_kappa.

6. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-377-A2 5 alternative splits
   + Wf-380-A2 per-cohort feature engineering; platinai_4cell + hilbert cohorts n=200 SMILES
   each remain at HONEST_NEGATIVE_R2_DEEP). Wf-387-A1 + A2 + A3 do NOT touch m2 R².

7. **2 pre-flight FAILs preserved (BY-DESIGN)** — NOT Wf-387-introduced.

8. **3 pre-flight SKIPs preserved (BY-DESIGN)** — NOT Wf-387-introduced.

9. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU +
   30 min install per Wf-368-B1 §0; no auto-trigger per AGENTS.md §5.1 + kill_then_adapt §0).

10. **Wf-388 Wf-125 P1 val_auc verification deferred** (USER_OK-gated GPU launch per AGENTS.md
    §5.1 + Wf-355-B1 wrapper; Wf-387-A2 builds the cohort; Wf-388 runs the verification).

11. **Better BibTeX MANUAL GUI install** + **Wetlab items DEPRECATED** (per AGENTS.md §5.4 +
    §10).

### §5.3 Adoption caveat

The Wf-387 wave is a **metric-fix + verification-anchor** wave — the canonical m1_auc catalog
is now correct (Wf-387-A1), and the structural verification cohort for the Wf-125 P1 lift is
ready (Wf-387-A2). The actual GPU retrain of Wf-125 P1 with the balanced 500 substituted for
the Wf-139 P7 synthetic 500 is BLOCKED on USER_OK=1 per AGENTS.md §5.1 standing GPU grant +
Wf-355-B1 wrapper. Once the user grants, Wf-388 can queue after the in-flight Wf-379-A1
completes (~5 hr PROJECTED).

### §5.4 Forward-looking user voice answer

**User voice (verbatim, 2026-09-30)**: *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* (need concrete planning + fixes based on current state; these internal metrics still have room for improvement; you can launch multiple ultracodes in parallel to make improvements).

**Wf-387 answer**: The "internal metrics with room for improvement" is the m1_auc canonical
catalog (which was carrying a WRONG n=1 single-fold value due to Wf-379-A1 trainer overwriting
its own data) + the Wf-125 P1 / Wf-139 P7 honest-negative gap (which lacked a structural
verification cohort). The Wf-387 wave ships:

1. **m1_auc canonical catalog fix** (Wf-387-A1) — 6-run aggregate across all MEASURED pairs
   with per-fold mean/std + Krasnov 0.86 gate verdict per fold.
2. **TODO-135 balanced Rh cohort** (Wf-387-A2) — 250 real actives + 250 real inactives from
   REAL MetalCytoToxDB under Wf-261 strict scaffold split with 0 scaffold leak to val/test.
3. **Balanced Rh CPU benchmark scaffold** (Wf-387-A3) — 674 LOC + 220 LOC test + 8/8 NEW tests
   PASS + smoke verdict ARTIFACT_LIFT confirms pipeline produces valid output.
4. **Paper §4 verdict** — Rh fold MEASURED_PASS_KRASNOV + Re fold MEASURED_BELOW_KRASNOV +
   overall BELOW_KRASNOV_PARTIAL.

**NEW ultracode Wf-388+ IS the recommended follow-up** to:
1. Re-run Wf-125 P1 with Wf-387-A2 real 500 in place of Wf-139 P7 synthetic 500 + READOUT to
   Rh val (USER_OK-gated GPU launch per AGENTS.md §5.1; ~5 hr GPU single retrain; ~15 hr GPU
   for full CONTROLD PAIR)
2. Re-launch Re seed=4 to close to 100% completion (currently 6/7 = 85.7% per Wf-387-A1)
3. Correct wrapper-format per_seed_metrics.json files (4 SWAPPED seed labels per Wf-387-A1 §5.2)
4. Atomic write protocol adoption migration per Wf-386-A3 §6.1 (existing closeout-drafting
   scripts to use `atomic_write_file` + `detect_truncation` adoption)
5. Pre-flight FAIL fixes (loi_send_ready + wf343c4_anchor_freshness_24h)
6. Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN
7. Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap

---

## §6 Cross-references

- Wf-387-A1 Wf-379-A1 GPU aggregation fix at `TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md`
- Wf-387-A2 BALANCED Rh cohort design intent at `TODO/completed/387a2_balanced_rh_cohort.md`
- Wf-387-A3 Balanced Rh benchmark scaffold at `TODO/completed/387a3_balanced_rh_bench.md`
- `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` (Wf-387-A2 NEW balanced
  500-cohort artifact; 250 real actives + 250 real inactives; 0 scaffold leak to val/test)
- `molmetal/scripts/wf387a3_balanced_rh_bench.py` (Wf-387-A3 NEW benchmark scaffold 674 LOC)
- `molmetal/tests/test_wf387a3_balanced_rh_bench.py` (Wf-387-A3 NEW 8-test suite)
- `molmetal/reports/wf387a3_balanced_rh_smoke/smoke_result.json` (Wf-387-A3 MEASURED smoke
  output, AUC=0.5 random ARTIFACT_LIFT verdict on inline synthetic cohort)
- `metrics/by_metric/metal_lomo_auc_wasserstein.json` (Wf-387-A1 UPDATED canonical Tier-2
  catalog with 6-run aggregate; value=0.7640 n_runs=6 metals=[Rh, Re] seeds=[3,4,0,1,2,3])
- `metrics/by_metric/metal_lomo_auc_wasserstein.json.bak.wf387a1` (Wf-387-A1 NEW backup)
- `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` (Wf-387-A1 NEW
  task-brief path; was missing)
- `AGENTS.md` §6 NEW row (this sync)
- `TODO/INDEX.md` §22.77 NEW entry (this sync)
- `TODO/completed/{389b1_index_sync,388b1_index_sync,385b1_index_sync,386b1_index_sync}_*.md`
  (prior sync closeouts preserved unchanged)
- `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json`
  (Wf-387-A2 BUGGY cohort context — the synthetic 500 Wf-387-A2 REPLACES with real 500)
- `molmetal/reports/wf131_p3_deferred_runs/{Rh,Re}/seed*/aggregate.json` (Wf-387-A1 per-pair
  rollups for 6 completed runs)
- `molmetal/reports/wf131_p3_deferred_runs/{Rh,Re}/seed*/{Rh,Re}/per_seed_metrics_seed{n}.json`
  (Wf-387-A1 per-pair flat files — canonical sources for the 6 test_auc values)
- `mol-metal-anticancer` skill (Krasnov 0.86 gate anchor for m1_auc verdict)
- `mol-metal-gpu-execution` skill (Standing GPU auth per AGENTS.md §5.1)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- AGENTS.md §5.5 (post-wave on-disk verification rule from Wf-384-B1)
- AGENTS.md §5.6 (atomic write protocol adoption rule from Wf-386-B1)
- AGENTS.md §10 (wetlab deprecation FAQ)

---

**END OF Wf-387-B1 INDEX SYNC** — Owner Wf-387-B1; cross-ref `party-twenty-spirit §3.5` +
AGENTS.md §1.5 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10; verdict: Wf-387 wave
DID彻底解决 the three internal-metric-improvement gaps per user 2026-09-30 verbatim
*"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* — the canonical m1_auc catalog is now correct (6-run aggregate with per-fold Krasnov 0.86 verdict), the Wf-125 P1 / Wf-139 P7 structural verification cohort is ready (balanced 500 from REAL MetalCytoToxDB under Wf-261 strict scaffold split), and the balanced Rh CPU benchmark scaffold is verified (8/8 NEW tests PASS).
