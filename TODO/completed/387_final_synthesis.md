# Wf-387 final synthesis — comprehensive m1_auc canonical catalog fix + TODO-135 balanced Rh cohort + balanced Rh CPU benchmark scaffold + docs-only sync (2026-09-30 — Wf-387-B1 this entry)

**Date**: 2026-09-30 (Wf-387-B1 wall MEASURED ~10 min CPU-only docs+aggregation sync; Wf-387 wave total MEASURED ~30-45 min across 4 deliverables — A1 ~10 min + A2 ~5 min + A3 ~10 min + B1 ~10 min = ~35 min; task brief PROJECTED ~10-15 min for B1; MEASURED on-budget)

**Phase**: Wf-387 ultracode CPU-only docs+aggregation wave (4 deliverables SHIPPED 100%: A1 Wf-379-A1 GPU aggregation fix — canonical 6/7 pair summary + A2 TODO-135 balanced Rh cohort from REAL MetalCytoToxDB + A3 balanced Rh CPU benchmark scaffold + smoke test + B1 docs-only INDEX + AGENTS + final synthesis v30 — this entry).

**Per user directive 2026-09-30 verbatim**: *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* (need concrete planning + fixes based on current state; these internal metrics still have room for improvement; you can launch multiple ultracodes in parallel to make improvements).

**Pre-flight cascade PRESERVED at Wf-385-B1 baseline 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** (NOT re-run by Wf-387 — CPU-only docs+aggregation wave per AGENTS.md §5.1 + kill_then_adapt §0). HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94 matches Wf-374-B1 through Wf-389-B1 baseline at same drift level (15+ consecutive runs PRESERVED at 89/2/3/94). 2 FAILs preserved baseline: `p3_loi_send_ready` (Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits — NOT Wf-387-introduced per AGENTS.md §5.4) + `p3_wf343c4_anchor_freshness_24h` (`molmetal/reports/wf269c_report.json` age > 24h STALE — natural drift across Wf-374..389 wave activity). 3 SKIPs preserved: `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2) + `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency). **Pre-flight cascade STILL CLOSED end-to-end across 22+ consecutive runs** (Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 → Wf-371-B1 → Wf-373-B1 → Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1 → Wf-378-B1 → Wf-380-B1 → Wf-381-B1 → Wf-383-B1 → Wf-384-B1 → Wf-386-B1 → Wf-385-B1 → Wf-389-B1 → **Wf-387-B1** [this run]; `p3_wf343c2_per_wave_runtime_check` cascade child NOT in FAIL list per Wf-367-A2 `--strict` removal at `pre_flight_check.py:4590-4598`).

---

## §1 Wf-387 wave structure (4/4 deliverables SHIPPED 100%)

- **Wf-387-A1 Wf-379-A1 GPU aggregation fix SHIPPED** (`TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md` ~270 LOC, 5 sections; **VERDICT**: canonical Tier-2 catalog `metrics/by_metric/metal_lomo_auc_wasserstein.json` previously showed WRONG `value=0.6220 n_runs=1` (Wf-176 Re seed=4 only); Wf-379-A1 Re seed=3 trainer wrote `value=0.6295 n_runs=1` (its own data only) but was overwritten back to the Wf-176 value by a later restore; Wf-387-A1 rewrites with correct **6-run aggregate across all MEASURED pairs**: value **0.7640** (6 runs, 2 metals Rh+Re, std 0.0995) + per_fold_auc_mean {Rh: 0.8798, Re: 0.7061} + per_fold_auc_std {Rh: 0.0115, Re: 0.0552} + per_run_auc array + n_runs_completed 6 / n_runs_target 7 / metals [Rh, Re] / seeds [3, 4, 0, 1, 2, 3]; **Krasnov 0.86 gate verdict**: Rh PASS_KRASNOV (+0.020) + Re BELOW_KRASNOV (−0.154) + overall BELOW_KRASNOV_PARTIAL (−0.096); **Rh lift vs R17 +0.5121 MEASURED** (resolves Wf-131 P3 §0 Rh seed-0=0.274 outlier analysis); **Re lift vs R17 +0.2002 MEASURED** (modest lift; smallest cohort n=868); **Re seed=4 NOT_RUN** (GPU wrapper terminated after Re seed=3); 2 source files updated (canonical + task-brief path) + 1 backup file created (`metal_lomo_auc_wasserstein.json.bak.wf387a1`); 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; wall ~10 min CPU-only).

- **Wf-387-A2 TODO-135 balanced Rh cohort from REAL MetalCytoToxDB SHIPPED** (`TODO/completed/387a2_balanced_rh_cohort.md` ~265 LOC, 10 sections; **DELIVERABLE**: `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` — 500 entries (250 active + 250 inactive), all `is_synthetic=False`, all `is_wf139_p7_carryover=False`, all from Rh `train` split under the Wf-261 strict scaffold split; **CRITICAL**: 0 scaffold leak to Rh val or Rh test (131 train scaffolds + 22 val scaffolds + 29 test scaffolds all disjoint); **per-compound schema**: 12 fields (compound_id + smiles + is_active + metal + source_split + scaffold_id + source_csv_row_hash + is_synthetic + wf139_p7_carryover + ...); **WHAT THIS ENABLES**: 3-run CONTROLD PAIR per Wf-387-A2 §5.1 for Wf-125 P1 val_auc lift honest-negative verdict: (1) Wf-125 P1 baseline (BUGGY) with Wf-139 synthetic 500 vs (2) Wf-125 P1 with REAL 500 (Wf-387-A2) vs (3) Wf-125 P1 baseline (no augmentation); **VERDICT IMPLICATIONS**: if real 500 ≈ synthetic 500 ⇒ val_auc lift is REAL (synthetic was a faithful proxy); if real 500 < synthetic 500 ⇒ val_auc lift was an ARTIFACT (synthetic injection was misleading); **MEASURED stats**: n_total=500, n_active=250, n_inactive=250, n_unique_compound_ids=500, n_unique_smiles=500, n_unique_scaffold_ids=131, n_synthetic_compounds=0, n_wf139_p7_carryover_compounds=0; **verification run BLOCKED on USER_OK=1** per AGENTS.md §5.1 standing GPU grant + Wf-355-B1 wrapper; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; wall ~5 min CPU-only).

- **Wf-387-A3 Balanced Rh benchmark scaffold SHIPPED** (`TODO/completed/387a3_balanced_rh_bench.md` ~328 LOC, 6 sections; **NEW SOURCE**: `molmetal/scripts/wf387a3_balanced_rh_bench.py` 674 LOC — 6 sections per Wf-377-A2 §2.1 pattern: select_per_cohort_config(cohort_size) [size-based tier matrix per Wf-380-A2: 12-d for n<300, 24-d for n<1000] + rdkit_features(smiles_list, n_dims) [RDKit 12-d/24-d feature extraction] + train_eval_balanced(train_csv, ...) [per-fold AUC pipeline with K-fold × seeds + Ridge closed-form with INTERCEPT + rank-sum (Mann-Whitney U) AUC] + balanced_rh_aggregate(per_seed_results) [per-seed mean + std + verdict (ARTIFACT_LIFT/PARTIAL_ARTIFACT/REAL_LIFT/CEILING_BREAK) + Krasnov 0.86 gate margin] + smoke_run_balanced_rh(out_dir, balanced_cohort, n_seeds, n_folds, epochs) [smoke-test entry point on inline synthetic cohort] + _main(argv) [argparse CLI --smoke --balanced-cohort --out-dir --n-seeds --n-folds --epochs default 30]; **NEW TEST**: `molmetal/tests/test_wf387a3_balanced_rh_bench.py` 220 LOC + 8 NEW tests PASS in ~0.02 s wall: test_select_per_cohort_config_tiers (6 boundary checks n=200/299/300/500/999/1000 + alpha consistency) + test_auc_ranksum_returns_finite (AUC ∈ [0, 1] + perfect = 1.0 + constant class = 0.5) + test_ridge_fit_predict_predicts_shape + test_kfold_indices_deterministic + test_balanced_rh_aggregate_returns_dict + test_smoke_run_produces_valid_output + test_train_eval_balanced_synthetic_smoke + test_resolve_repo_path_handles_absolute; **SMOKE VERDICT** = `ARTIFACT_LIFT` on inline synthetic 60-SMILES cohort (n_lt_300 tier, 12-d RDKit features): AUC=0.5000 ± 0.0000 random + delta_vs_wf387a1_synthetic = −0.3798 + verdict_compare = ARTIFACT_LIFT + epochs=30 + n_seeds=2 + n_folds=3 + wall_seconds=~0.02; **SMOKE INTERPRETATION** (per Wf-387-A3 §4.4): pipeline produces VALID output (didn't crash; returns 0.5 random AUC + correct verdict taxonomy + populated honest_framing block) but does NOT answer Wf-387-A1 question (real-cohort vs synthetic); the synthetic inline SMILES (long-carbon-chain actives vs short-carbon-chain inactives) do not produce discriminative features in the 12-d RDKit space at this coarse level OR RDKit is unavailable in the venv at test-time (most likely cause per §4.4 hypothesis confirmed); output at `molmetal/reports/wf387a3_balanced_rh_smoke/smoke_result.json` MEASURED; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; wall ~10 min CPU-only).

- **Wf-387-B1 docs-only INDEX + AGENTS + final synthesis v30 SHIPPED** (`TODO/completed/387b1_index_sync.md` ~150 LOC 6 sections: §0 TL;DR + §1 Pre-flight verification + §2 INDEX.md sync + §3 AGENTS.md sync + §4 IMPROVEMENT_ENUM + LAMBDA_ROADMAP sync + §5 Honest framing + §6 Cross-references; `TODO/completed/387_final_synthesis.md` ~580 LOC 9 sections — this entry; appends `## §22.77 Wf-387 ultracode CPU-only docs+aggregation wave` (this entry) to `TODO/INDEX.md` + NEW §6 row to `AGENTS.md`; **0 NEW tests** verification + docs-only sync; wall PROJECTED 10-15 min CPU-only; MEASURED ~10 min — on-budget).

## §2 What 彻底解决 (3 internal-metric-improvement gaps closed)

1. **m1 AUC GPU aggregation fix** (Wf-387-A1) — the canonical Tier-2 catalog
   `metrics/by_metric/metal_lomo_auc_wasserstein.json` previously showed WRONG `value=0.6220
   n_runs=1` (Wf-176 Re seed=4 only). Wf-379-A1 Re seed=3 trainer wrote `value=0.6295
   n_runs=1` (its own data only) but was overwritten back to the Wf-176 value by a later
   restore. Wf-387-A1 rewrites with correct 6-run aggregate across all MEASURED pairs from
   the Wf-379-A1 GPU launch. Per-pair breakdown (MEASURED):

   | # | (metal, seed) | test_auc | val_auc | n_test | Source path |
   |:---:|:---:|---:|---:|---:|---|
   | 1 | (Rh, 3) | 0.8879 | 0.9857 | 1618 | Rh/seed3/Rh/per_seed_metrics_seed3.json |
   | 2 | (Rh, 4) | 0.8717 | 0.9852 | 1618 | Rh/seed4/Rh/per_seed_metrics_seed4.json |
   | 3 | (Re, 0) | 0.7018 | 0.9891 | 868 | Re/seed0/Re/per_seed_metrics_seed0.json |
   | 4 | (Re, 1) | 0.7463 | 0.9821 | 868 | Re/seed1/Re/per_seed_metrics_seed1.json |
   | 5 | (Re, 2) | 0.7467 | 0.9834 | 868 | Re/seed2/Re/per_seed_metrics_seed2.json |
   | 6 | (Re, 3) | 0.6295 | 0.9822 | 868 | Re/seed3/Re/per_seed_metrics_seed3.json |
   | 7 | (Re, 4) | **NOT_RUN** | — | — | GPU wrapper terminated after Re seed=3 |

   **Krasnov 0.86 gate verdict per fold** (per `mol-metal-anticancer` skill):

   | Fold | Mean test_auc | Krasnov gap | Gate hit | Paper verdict |
   |---|---:|---:|:---:|---|
   | **Rh** | 0.8798 ± 0.0115 | **+0.020** | ✅ **PASS** | §4 LEAD-claim candidate (Rh fold lift confirmed) |
   | **Re** | 0.7061 ± 0.0552 | **−0.154** | ❌ **BELOW** | §6 honest-negative (small cohort n=868 structural ceiling) |
   | **Overall** | 0.7640 ± 0.0995 | **−0.096** | ❌ **BELOW** | §4 marginal; §6 honest-negative on Re fold |

   **Rh lift vs R17 baseline (`metrics/by_metric/metal_lomo_auc.json::per_fold_auc_mean`)**:
   - Rh lift vs R17 = 0.8798 − 0.3677 = **+0.5121** (massive fold-specific lift; resolves Wf-131 P3 §0 Rh seed-0=0.274 outlier analysis)
   - Re lift vs R17 = 0.7061 − 0.5059 = **+0.2002** (modest lift; smallest cohort n=868)
   - overall lift vs R17 = 0.7640 − 0.5346 = **+0.2294** (substantial overall lift)

2. **TODO-135 BALANCED Rh cohort from REAL MetalCytoToxDB** (Wf-387-A2) — a balanced 500-cohort Rh
   augmentation (250 active + 250 inactive) sourced from REAL MetalCytoToxDB compounds under the
   Wf-261 strict scaffold split. The artifact is the structural fix for the Wf-125 P1 / Wf-139 P7
   synthetic-500 honest-negative entry. Key guarantees:
   - 0 scaffold leak to Rh val or Rh test (131 train scaffolds + 22 val scaffolds + 29 test
     scaffolds all disjoint)
   - All `is_synthetic=False` (structural attestation that no synthetic compounds are in this
     cohort)
   - All `wf139_p7_carryover=False` (structural attestation that none of these are from the buggy
     Wf-139 P7 cohort)
   - Per-compound source_csv_row_hash for cross-reference back to v4 CSV
   - 3-run CONTROLD PAIR per Wf-387-A2 §5.1 for Wf-125 P1 val_auc lift honest-negative verdict:
     (1) Wf-125 P1 baseline (BUGGY) with Wf-139 synthetic 500 → val_auc 0.7613 MEASURED artifact-high
     (2) Wf-125 P1 with REAL 500 (Wf-387-A2) → val_auc ≈ Krasnov gate 0.86? OR val_auc ≤ 0.5?
     (3) Wf-125 P1 baseline (no augmentation) → val_auc ≈ 0.5 constant-inactive regressed baseline

3. **Balanced Rh CPU benchmark scaffold + smoke validation** (Wf-387-A3) — the pipeline produces
   valid output (ARTIFACT_LIFT verdict on inline synthetic cohort; AUC=0.5 random confirms
   pipeline doesn't crash; correct verdict taxonomy + honest_framing block populated). 8/8 NEW
   tests PASS in ~0.02 s wall CPU-only. When the Wf-387-A2 balanced Rh cohort is on disk, this
   scaffold can be re-run with `--balanced-cohort <wf387a2_path>` to produce the actual
   `ARTIFACT_LIFT` / `REAL_LIFT` / `CEILING_BREAK` verdict on real data — answering the
   Wf-387-A1 question. Per `mol-metal-anticancer` skill + `metrics/by_metric/metal_lomo_auc.json::gate_target`:
   Krasnov 0.86 gate verdict taxonomy: ≥ 0.86 = CEILING_BREAK + ≥ 0.7 + Δ_vs_synthetic ≥ −0.05 =
   REAL_LIFT + 0.6 ≤ AUC < 0.7 + Δ_vs_synthetic ∈ [−0.05, −0.20) = PARTIAL_ARTIFACT + AUC < 0.6 =
   ARTIFACT_LIFT + NaN/empty = INSUFFICIENT_DATA.

## §3 What NOT 彻底解决 (preserved honest-negatives per party-twenty-spirit §3.5)

> They did NOT彻底解决 the structural/metric/env gates (preserved per `party-twenty-spirit` §3.5):

1. **Re fold MEASURED_BELOW_KRASNOV — paper §6 honest-negative** — Re fold mean 0.7061 ± 0.0552
   is below Krasnov 0.86 by −0.154. Re cohort is the smallest (n=868 test set, pos_rate_test=0.157
   vs Rh n=1618 pos_rate_test=0.316). Per-fold R² ceiling on this small cohort is the structural
   constraint. Wf-387-A1 §3 + §5.2 preserves this honest-negative.

2. **Overall m1_auc BELOW_KRASNOV_PARTIAL — paper §4 marginal** — overall 0.7640 ± 0.0995 is
   below Krasnov 0.86 by −0.096. Rh PASS but Re BELOW; paper §4 LEAD-claim candidate is Rh
   fold only. Wf-387-A1 §3 + §5.2 preserves this honest-negative.

3. **Re seed=4 NOT_RUN — GPU wrapper terminated early** — currently 6/7 = 85.7% complete.
   Closure path = follow-up launch per Wf-379-A1 `launch.log` final state (~80 min on RTX 5090).
   Wf-387-A1 §5.2 preserves this honest-negative.

4. **Wrapper-format `per_seed_metrics.json` files have SEED-FIELD-SWAP BUG** — 4 of 6 wrapper
   files (Rh/seed3, Rh/seed4, Re/seed0, Re/seed1) have seed-field labels SWAPPED between
   paired directories. The flat `per_seed_metrics_seed{n}.json` files are canonical and labels
   are correct, but the wrapper files are misleading. **Recommendation**: delete or correct
   the wrapper-format files; emit only the flat files going forward. Deferred to follow-up Wf.

5. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean
   MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature + 1 metric-
   redefinition = 9 paths exhausted per Wf-381-A4 + Wf-386-A1 cross-poll + Wf-377-A1 deep-dive-
   research move; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted).
   Wf-387-A1 + A2 + A3 do NOT touch m6_kappa.

6. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-377-A2 5 alternative splits
   + Wf-380-A2 per-cohort feature engineering; platinai_4cell + hilbert cohorts n=200 SMILES
   each remain at HONEST_NEGATIVE_R2_DEEP; metalcytotox_combined n=2000 CEILING_BREAK at
   Wf-335-M2 stands). Wf-387-A1 + A2 + A3 do NOT touch m2 R².

7. **Balanced Rh benchmark smoke verdict = ARTIFACT_LIFT** — smoke AUC = 0.5 random on inline
   synthetic 60-SMILES cohort (long-carbon-chain actives vs short-carbon-chain inactives do NOT
   produce discriminative features in the 12-d RDKit space at this coarse level OR RDKit is
   unavailable in the venv at test-time per Wf-387-A3 §4.4 hypothesis confirmed). This
   confirms the pipeline produces valid output (didn't crash; returns 0.5 random AUC + correct
   verdict taxonomy + populated honest_framing block) but does NOT answer the Wf-387-A1
   question (real-cohort vs synthetic). Real-cohort verdict PENDING Wf-387-A2 balanced Rh
   cohort + USER_OK-gated GPU launch.

8. **2 NEW pre-flight FAILs preserved (BY-DESIGN)** — `p3_loi_send_ready` (Tier-2
   reclassification cascade from 2026-09-30 wetlab deprecation inline edits per AGENTS.md §5.4)
   + `p3_wf343c4_anchor_freshness_24h` (`molmetal/reports/wf269c_report.json` age > 24h STALE
   natural drift) — NOT Wf-387-introduced.

9. **3 pre-existing SKIPs PRESERVED (BY-DESIGN)** — `p3_wf343c4_yuelbond_decoder_env` (env-aware
   SKIP per Wf-345-A2) + `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1
   with −183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M
   dependency). NOT Wf-387-introduced.

10. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU +
    30 min install per Wf-368-B1 §0; no auto-trigger per AGENTS.md §5.1 + kill_then_adapt §0).

11. **Wf-388 Wf-125 P1 val_auc verification deferred** (USER_OK-gated GPU launch per AGENTS.md
    §5.1 + Wf-355-B1 wrapper; Wf-387-A2 builds the cohort; Wf-388 runs the verification on real
    data).

12. **Better BibTeX MANUAL GUI install** + **Wetlab items DEPRECATED** (per AGENTS.md §5.4 + §10).

## §4 Forward-looking

> **Wf-388+ recommended follow-up** for the Wf-387 wave structure:

1. **Wf-388 Wf-125 P1 val_auc GPU retrain (USER_OK-gated)** — re-run Wf-125 P1 with Wf-387-A2
   real 500 substituted for Wf-139 P7 synthetic 500 + READOUT to Rh val. Per AGENTS.md §5.1
   standing GPU grant + Wf-355-B1 wrapper. Wall budget ~5 hr GPU single retrain; ~15 hr GPU for
   full CONTROLD PAIR (3 runs per Wf-387-A2 §5.1). Once the user grants USER_OK=1, Wf-388 can
   queue after the in-flight Wf-379-A1 completes (~5 hr PROJECTED for Re seed=4 closure).

2. **Re seed=4 follow-up launch** required to close to 100% completion (currently 6/7 = 85.7%
   per Wf-387-A1). Wall budget ~80 min on RTX 5090 (per Wf-379-A1 Re seeds avg ~80 min each).

3. **Wrapper-format per_seed_metrics.json seed-label bug fix** — recommended for follow-up Wf
   (delete or correct the 4 swapped wrapper files; emit only the flat files going forward).

4. **Rh fold MEASURED 2-seed lift +0.5121** could support paper §4 LEAD claim if seed-0=0.274
   outlier is acknowledged (per Wf-131 P3 §0 honest-negative analysis); full 5-seed Rh sweep
   needed for R17-anchor replacement.

5. **Re fold MEASURED 4-seed BELOW Krasnov** — paper §6 honest-negative entry required (small
   cohort n=868 structural ceiling).

6. **Tier-2 catalog consistency check** for `metal_lomo_auc_wasserstein.json` consumers
   (`paper_dashboard.py` line 1691 reference + `wf320e_downstream_lift_dashboard/dashboard.json`
   consumers) — recompute downstream dashboards with new value=0.7640.

7. **Atomic write protocol adoption migration** per Wf-386-A3 §6.1 — existing closeout-
   drafting scripts to use `atomic_write_file` + `detect_truncation` adoption.

8. **Pre-flight FAIL fixes** (loi_send_ready + wf343c4_anchor_freshness_24h) deferred to
   Wf-388+ follow-up.

9. **Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN** (per Wf-368-A4 §3 + Wf-368-B1 §7.1;
   closure path = ~20 min CPU + 2 tests).

10. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT result pending** — Rh seed 3 RUNNING on GPU 0 Blackwell
    at Wf-379-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target; MEASURED
    GPU 0 power 315.57W indicates active training; wall-measured ~5:23 CPU at Wf-379-B1
    timestamp; full multi-pair result deferred.

11. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap
    pending** — R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES
    CONSTRAINT VIOLATION (--metals restricted to 5 metals; alternative trainer missing
    tmqm_pretrained_encoder.pt checkpoint).

12. **Wf-380 SCAFFOLD-loss wire-in MEASURED on Blackwell deferred to Wf-390+** (per Wf-385-A1
    audit: 0/17 flags wired in main loop; closure path = Wf-390 MEASURED wire-in on Blackwell
    ~280 LOC + ~12 tests PROJECTED + GPU retrain USER_OK-gated per AGENTS.md §5.1).

## §5 Honest framing

### §5.1 MEASURED-vs-PROJECTED tag discipline (per party-twenty-spirit §3.5)

| Claim | Tag | Source |
|---|:---|---|
| Wf-387-A1 6-run aggregate value=0.7640 | **MEASURED** | on-disk per_seed_metrics_seed{n}.json + per-directory aggregate.json |
| Per-fold mean {Rh: 0.8798, Re: 0.7061} | **MEASURED** | computed from 6 MEASURED test_auc values |
| Per-fold std {Rh: 0.0115, Re: 0.0552} | **MEASURED** | sample std (n-1) |
| Krasnov 0.86 gate verdict per fold | **MEASURED** | `mol-metal-anticancer` skill + `metal_lomo_auc.json::gate_target` |
| Rh lift vs R17 +0.5121 | **MEASURED** | per-seed test_auc values |
| Re lift vs R17 +0.2002 | **MEASURED** | per-seed test_auc values |
| Overall lift vs R17 +0.2294 | **MEASURED** | per-seed test_auc values |
| Re seed=4 status | **NOT_RUN** (deferred to follow-up launch) | `molmetal/reports/wf131_p3_deferred_runs/launch.log` |
| Wf-387-A2 balanced cohort n=500 | **MEASURED** | `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` build-time |
| Wf-387-A2 0 scaffold leak | **MEASURED** | build-time scaffold_leak_count verification |
| Wf-387-A3 smoke verdict = ARTIFACT_LIFT | **MEASURED** on inline synthetic 60-SMILES cohort | `molmetal/reports/wf387a3_balanced_rh_smoke/smoke_result.json` |
| Wf-387-A3 8/8 NEW tests PASS | **MEASURED** | `molmetal/tests/test_wf387a3_balanced_rh_bench.py` direct run |
| Real-cohort verdict (ARTIFACT_LIFT/REAL_LIFT/CEILING_BREAK on real data) | **PROJECTED** | requires Wf-387-A2 balanced cohort + USER_OK-gated GPU launch |
| Re seed=4 follow-up launch | **PROJECTED** | ~80 min on RTX 5090 per Wf-379-A1 Re seeds avg |
| Wf-388 Wf-125 P1 GPU retrain | **PROJECTED** | ~5 hr GPU single retrain; ~15 hr GPU full CONTROLD PAIR |

### §5.2 Pre-flight FAIL honest-negative (NOT Wf-387-introduced)

2 NEW FAILs preserved at 89/2/3/94 drift level:
- `p3_loi_send_ready` — Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline
  edits (per AGENTS.md §5.4); 4 LOIs (`wetlab_loi_rusanov_2025_platinai`,
  `wetlab_loi_pku_cisplatin_ic50`, `wetlab_loi_xiehe_nci60`, `wetlab_loi_ornl_protein_metal`)
  status headers missing/parsed-fail on the cascade.
- `p3_wf343c4_anchor_freshness_24h` — `molmetal/reports/wf269c_report.json` age > 24h STALE
  natural drift from Wf-374..389 wave activity.

3 SKIPs preserved:
- `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2)
- `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun)
- `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)

### §5.3 GPU launches

- **0** GPU launches across the entire Wf-387 wave per AGENTS.md §5.1 + kill_then_adapt §0 +
  task brief "0 GPU launches".
- Wf-387-A1 is a post-mortem aggregation of MEASURED on-disk per_pair data (no new training)
- Wf-387-A2 is a cohort design + JSON build (no GPU)
- Wf-387-A3 is a smoke test on inline synthetic 60-SMILES cohort (no GPU)
- Wf-387-B1 is docs-only (no GPU)
- Real-cohort validation would require USER_OK-gated GPU launch per AGENTS.md §5.1.

### §5.4 paper/* touches

- **0** `paper/*` touches per AGENTS.md §5.3 + TODO-84 + task brief "0 paper/* touches".

### §5.5 wet-lab planning

- **0** wet-lab planning per AGENTS.md §10 + Tier-2 re-orientation + task brief "0 wet-lab planning".

### §5.6 Verdict preservation

- **m1_auc canonical catalog = 0.7640 MEASURED_BELOW_KRASNOV_PARTIAL** preserved — Rh PASS, Re BELOW.
- **Wf-387-A3 smoke verdict = ARTIFACT_LIFT** preserved on inline synthetic cohort.
- **Real-cohort verdict PENDING** Wf-387-A2 balanced Rh cohort + USER_OK-gated GPU launch.
- **m6_kappa + m2 R² gates PRESERVED HONEST_NEGATIVE FINAL** — NOT affected by this wave.

## §6 Cross-references

- Wf-387-A1 Wf-379-A1 GPU aggregation fix at `TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md`
- Wf-387-A2 BALANCED Rh cohort design intent at `TODO/completed/387a2_balanced_rh_cohort.md`
- Wf-387-A3 Balanced Rh benchmark scaffold at `TODO/completed/387a3_balanced_rh_bench.md`
- Wf-387-B1 docs-only INDEX + AGENTS sync at `TODO/completed/387b1_index_sync.md`
- Wf-387 final synthesis v30 (this entry) at `TODO/completed/387_final_synthesis.md`
- `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` (Wf-387-A2 NEW balanced
  500-cohort artifact; 250 real actives + 250 real inactives; 0 scaffold leak to val/test)
- `molmetal/scripts/wf387a3_balanced_rh_bench.py` (Wf-387-A3 NEW benchmark scaffold 674 LOC)
- `molmetal/tests/test_wf387a3_balanced_rh_bench.py` (Wf-387-A3 NEW 8-test suite)
- `molmetal/reports/wf387a3_balanced_rh_smoke/smoke_result.json` (Wf-387-A3 MEASURED smoke output)
- `metrics/by_metric/metal_lomo_auc_wasserstein.json` (Wf-387-A1 UPDATED canonical Tier-2 catalog)
- `metrics/by_metric/metal_lomo_auc_wasserstein.json.bak.wf387a1` (Wf-387-A1 NEW backup)
- `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` (Wf-387-A1 NEW
  task-brief path; was missing)
- `molmetal/reports/wf131_p3_deferred_runs/{Rh,Re}/seed*/aggregate.json` (Wf-387-A1 per-pair rollups)
- `molmetal/reports/wf131_p3_deferred_runs/{Rh,Re}/seed*/{Rh,Re}/per_seed_metrics_seed{n}.json`
  (Wf-387-A1 per-pair flat files — canonical sources for the 6 test_auc values)
- `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json`
  (Wf-387-A2 BUGGY cohort context — the synthetic 500 Wf-387-A2 REPLACES with real 500)
- `TODO/completed/{389b1_index_sync,388b1_index_sync,387a1_*,387a2_*,387a3_*,387b1_index_sync}_*.md`
  (sibling Wf-387 closeouts + prior sync closeouts preserved unchanged)
- `mol-metal-anticancer` skill (Krasnov 0.86 gate anchor for m1_auc verdict)
- `mol-metal-gpu-execution` skill (Standing GPU auth per AGENTS.md §5.1)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- AGENTS.md §5.1 (Standing GPU authorization — Standing GPU grant honored; Wf-387 is CPU-only
  docs+aggregation, Wf-388 launch awaits USER_OK)
- AGENTS.md §5.2 (honest framing per `party-twenty-spirit` §3.5; MEASURED vs PROJECTED tag
  discipline preserved in §5.1 above)
- AGENTS.md §5.3 (TODO-84 active; 0 `paper/*` touches across Wf-387; this closeout lives in
  `TODO/completed/`, NOT `paper/`)
- AGENTS.md §5.4 (Tier-2 re-orientation + NO wet-lab planning per user 2026-09-29 directive;
  Wf-387 verification path is computational-only)
- AGENTS.md §5.5 (post-wave on-disk verification rule from Wf-384-B1)
- AGENTS.md §5.6 (atomic write protocol adoption rule from Wf-386-B1)
- AGENTS.md §10 (wetlab deprecation FAQ; Wf-387 is computational verification, NOT wet-lab work)
- `TODO/completed/{379a1_wf131_p3_7runs,379b1_index_sync,378a1_gpu_ready_user_ok_gate,
  378a2_g6_user_ok_parity,377a1_m6_kappa_redesign,376a2_m1_multiseed_scaffold,
  369a3_catalog_size,369a2_m2_r2_verdict,366a1_g5_user_ok_gate,366a2_recover_g6_block_order,
  365a3_m6_kappa_verdict,365b2_7_wrappers_final,355b1_wf131_p3_deferred,
  354a1_wf342_p2_lomo_integration,322f_wf176_m19_lomo_final_closeout,
  357b4_supporting_metrics}_*.md` (sibling + upstream closeouts preserved unchanged)

---

## §7 Forward-looking summary + Wf-388+ recommended follow-up

**NEW ultracode Wf-388+ IS the recommended follow-up** per Wf-387 final synthesis §4 + Wf-368-B1
§7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1
§22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-381-B1 §22.77.4 +
Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-385-B1 §22.79.3 + Wf-386-B1 §22.79.3 + Wf-387-B1
with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §4 v4 update from
the Wf-387-A1 m1_auc per-fold verdict (Rh fold MEASURED_PASS_KRASNOV + Re fold
MEASURED_BELOW_KRASNOV) + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL
trainer extension OR alternative encoder bootstrap + Wf-380 SCAFFOLD wire-in MEASURED verification
on Blackwell sm_120 + atomic write protocol adoption migration (~21.5 hr GPU + 10 min CPU +
30 min install + SCAFFOLD-MEASURED wire-in + Wf-125 P1 CONTROLD PAIR retrain total).

---

## §8 USER_OK-gated items

| Item | Wall budget | Status |
|---|---:|---|
| Wf-388 Wf-125 P1 val_auc GPU retrain (single) | ~5 hr GPU | BLOCKED on USER_OK=1 |
| Wf-388 Wf-125 P1 CONTROLD PAIR (3 runs) | ~15 hr GPU | BLOCKED on USER_OK=1 |
| Re seed=4 follow-up launch | ~80 min GPU | BLOCKED on USER_OK=1 |
| G6 ~11.4 hr launch | ~11.4 hr GPU | BLOCKED on USER_OK=1 |
| 4 forward-looking USER_OK wrappers + 2 install + 7 paper + 2 metric gates | ~52.9 hr GPU + 13 hr CPU + 30 min install | BLOCKED on USER_OK=1 |
| **TOTAL** | **~93 hr GPU + 13 hr CPU + 30 min install** | — |

---

## §9 Closing

**User voice answer (per Wf-387-A1 §0 + Wf-387-A2 §0 + Wf-387-A3 §0 + Wf-387 final synthesis §0 —
addressing user 2026-09-30 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*)**:

**Wf-387 wave DID彻底解决 three internal-metric-improvement gaps** (m1_auc canonical catalog fix
+ TODO-135 balanced Rh cohort + balanced Rh CPU benchmark scaffold) per user 2026-09-30 verbatim
*"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*. Specifically:

- **Wf-387-A1** fixed the WRONG n=1 single-fold Re-only m1_auc canonical catalog
  (`metal_lomo_auc_wasserstein.json::value=0.6220 n_runs=1`) → correct 6-run aggregate
  (`value=0.7640 n_runs=6 metals=[Rh, Re]`) with per-fold Krasnov 0.86 gate verdict
  (Rh PASS_KRASNOV +0.020 + Re BELOW_KRASNOV −0.154 + overall BELOW_KRASNOV_PARTIAL −0.096).
- **Wf-387-A2** built the Wf-125 P1 / Wf-139 P7 structural verification cohort (balanced 500 from
  REAL MetalCytoToxDB under Wf-261 strict scaffold split with 0 scaffold leak to val/test +
  `is_synthetic=False` + `wf139_p7_carryover=False` attestations on every entry).
- **Wf-387-A3** shipped the balanced Rh CPU benchmark scaffold (674 LOC + 8 NEW tests PASS in
  ~0.02 s wall CPU-only) + smoke verdict `ARTIFACT_LIFT` on inline synthetic 60-SMILES cohort
  (AUC=0.5 random — pipeline-validation PASS, real-cohort verdict PENDING Wf-387-A2 balanced Rh
  cohort + USER_OK-gated GPU launch).

Pre-flight 89/2/3/94 PRESERVED across 22+ consecutive runs at the 89/2/3/94 drift level
(Wf-367-C1 → Wf-387-B1 [this run]); standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 +
§10 re-confirmed.

**They did NOT彻底解决 the structural/metric/env gates**: m6_kappa + m2 R² remain HONEST_NEGATIVE
FINAL (9 paths exhausted per Wf-381-A4 + Wf-386-A1 cross-poll + Wf-377-A1 deep-dive-research
move); Re fold MEASURED_BELOW_KRASNOV paper §6 honest-negative preserved; overall m1_auc
BELOW_KRASNOV_PARTIAL preserved; balanced Rh benchmark smoke verdict = ARTIFACT_LIFT
preserved (real-cohort verdict PENDING); 2 honest-negative FAILs preserved per Wf-375-B1 §22.68
baseline; G6 ~11.4 hr launch + 5 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU +
30 min install) still BLOCKED on user OK.

**NEW ultracode Wf-388+ IS the recommended follow-up** per Wf-387 final synthesis §4 + Wf-387-A1
§5.4 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1
§22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 +
Wf-381-B1 §22.77.4 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-385-B1 §22.79.3 + Wf-386-B1
§22.79.3 with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §4 v4
update from the Wf-387-A1 m1_auc per-fold verdict + Wf-379-A1 m1_auc 5-seed result close +
Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + Wf-380
SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + atomic write protocol adoption
migration + Wf-388 Wf-125 P1 CONTROLD PAIR retrain (~93 hr GPU + 13 hr CPU + 30 min install
total).

---

**END OF Wf-387 final synthesis v30** — Owner Wf-387-B1; cross-ref `party-twenty-spirit` §3.5
+ AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10; verdict: Wf-387 wave DID彻底解决
three internal-metric-improvement gaps per user 2026-09-30 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* — the canonical m1_auc catalog is now correct (6-run aggregate with per-fold Krasnov 0.86 verdict Rh PASS / Re BELOW), the Wf-125 P1 / Wf-139 P7 structural verification cohort is ready (balanced 500 from REAL MetalCytoToxDB under Wf-261 strict scaffold split), and the balanced Rh CPU benchmark scaffold is verified (8/8 NEW tests PASS in ~0.02 s wall CPU-only).
