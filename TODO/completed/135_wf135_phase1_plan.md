# TODO-135 / Wf-135 Phase 1 + Phase 2 — Detailed Plan Doc

**Date**: 2026-09-29
**Status**: 📌 Phase 1 SHIPPED (Wf-139 P7, see §1 cross-ref); Phase 2 DEFERRED to GPU window (see §2)
**Owner**: Hugo Lee + Claude Code
**Parent**: Wf-131 honest-negative finding + TODO-121 m19 LOMO strengthen
**Close-out (broader context)**: `TODO/completed/135_m19_lomo_rh_oversample_fix.md`
**Skill alignment**: `party-twenty-spirit` §3.5 + `mol-metal-lambda-calculus` + `mol-metal-gpu-execution` §0

---

## §0 TL;DR

Wf-135 / TODO-135 is a 2-phase LOMO Rh-regression fix.

- **Phase 1 = balanced Rh cohort generation**. ✅ **SHIPPED 2026-09-27 via Wf-139 P7** (CPU-only RDKit + Tanimoto-diff labels, 15/15 tests PASS, scaffold-split 0 leak). This Phase 1 work is **done** and the close-out is `TODO/completed/135_m19_lomo_rh_oversample_fix.md`.
- **Phase 2 = LOMO 5-metal × 5-seed × 50-epoch re-bench using the Phase 1 balanced cohort + per-metal head ensemble + Wasserstein target fix**. 🚫 **DEFERRED to GPU window** — requires USER_OK per `mol-metal-gpu-execution` §0 + ~2 weeks GPU-1 wall. Wall-time honest projection: ~10 GPU-hours of compute, ~2 calendar weeks including setup/teardown/reporting.

This doc focuses on:
1. What Phase 1 produced (acceptance evidence, where to read it).
2. What Phase 2 must do to recover Rh fold ≥ 0.30 AUC + push LOMO mean ≥ 0.60.
3. The USER_OK gate and what USER must approve before Phase 2 starts.

---

## §1 Phase 1 — Balanced Rh cohort (SHIPPED)

**Workflow**: Wf-139 P7
**Full close-out**: `TODO/completed/135_m19_lomo_rh_oversample_fix.md` §1
**Per-step close-out**: `TODO/completed/143_wf139_p7_rh_active_cohort.md`
**Report**: `molmetal/reports/wf139_p7_rh_active_cohort/`

### §1.1 Method recap (active learning, similarity-based)

1. Load MetalCytoToxDB Rh real train cohort (n=794: 413 active + 381 inactive).
2. Compute Morgan fingerprints (radius=2, nBits=2048) for all real + 500 Wf-131 P2 synthetic compounds.
3. For each synthetic compound `s`:
   - `max_sim_active(s)` = max Tanimoto similarity to 413 real actives.
   - `max_sim_inactive(s)` = max Tanimoto similarity to 381 real inactives.
   - `diff(s)` = `max_sim_active(s) − max_sim_inactive(s)`.
4. Rank 500 compounds by `diff` descending.
5. **Top 250** by diff → label = 1 (active).
6. **Bottom 250** by diff → label = 0 (inactive).

### §1.2 Phase 1 acceptance gates (all MET, per IMPROVEMENT_ENUM §1.2)

| Gate | Standard | MEASURED | Status |
|---|---|---|---|
| 500 cohort generated | 250 active + 250 inactive | **250 + 250** | ✅ |
| Balanced labels | not constant-inactive | 250/250 split | ✅ |
| Catherine 6/6 PASS | ≥ 99% | 1.0000 (500/500) | ✅ |
| Scaffold leak | overlap = 0 | 0 | ✅ |
| ≥12 tests PASS | per `test_rh_active_label_generation.py` | **15/15 PASS** | ✅ |
| Wall-time | CPU-only | 0.58 s | ✅ |

### §1.3 Phase 1 honest framing (carried over from 135_m19_lomo_rh_oversample_fix.md §1.3)

1. Similarity-based label is **proxy**, NOT wet-lab validation. The 250 "actives" share chemotype similarity with 413 real actives but have NOT been tested for actual cytotoxicity.
2. Diff-based ranking is workaround for low absolute Tanimoto similarity (median 0.18). Pearson r(active_sim, inactive_sim) = 0.95 — synthetics are about as similar to actives as to inactives.
3. Structural bias: 423/500 compounds more similar to inactive than active. "Active" cohort = LEAST inactive-like 250, not actually active.
4. Scaffold-split verified clean (overlap = 0).

### §1.4 Phase 1 outputs (already on disk)

| Path | Status |
|---|---|
| `molmetal/scripts/rh_active_label_generation.py` | ✅ SHIPPED (Wf-139 P7, 340 LOC) |
| `molmetal/tests/test_rh_active_label_generation.py` | ✅ SHIPPED (15/15 PASS) |
| `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` | ✅ SHIPPED (500 balanced) |
| `molmetal/reports/wf139_p7_rh_active_cohort/synthesis.md` | ✅ SHIPPED |
| `TODO/completed/143_wf139_p7_rh_active_cohort.md` | ✅ SHIPPED |
| `TODO/completed/135_m19_lomo_rh_oversample_fix.md` | ✅ SHIPPED (close-out) |

**Phase 1 is closed**. No further Phase 1 work is needed. Phase 2 is the open question.

---

## §2 Phase 2 — LOMO 5-metal × 5-seed × 50-epoch re-bench (DEFERRED to GPU)

**Workflow**: Wf-135 (planned; not yet run)
**Resource gate**: USER_OK per `mol-metal-gpu-execution` §0 + GPU-1 RTX 5090 (Blackwell sm_120) availability
**Wall budget**: ~2 calendar weeks (compute: ~10 GPU-hours; remainder = setup, teardown, reporting, regression triage)
**Honest deferral rationale**: per "kill then adapt" directive — CPU-only session cannot run multi-metal multi-seed D-MPNN training. Phase 2 GPU retrain cannot be fake-shipped via CPU proxy.

### §2.1 Phase 2 design — what to actually do

Inputs (already exist):
- Wf-139 P7 balanced cohort (`molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json`, 500 compounds)
- Wf-131 P2 500 synthetic compounds (orig, now superseded by Wf-139 P7)
- MetalCytoToxDB Rh real cohort (n=794: 413 active + 381 inactive)
- Per-metal LOMO folds for the 5 metals (Pt / Ru / Ir / Au / Pd — final list TBD per `Wf-39 P2 LOMO design`)

What to ship:
1. `molmetal/scripts/train_metal_lomo_v2.py` (~280 LOC)
   - 5 metals × 5 seeds × 50 epochs (1250 runs total)
   - Rh oversample with **balanced labels** (use Phase 1 cohort)
   - **Per-metal head ensemble** (K=3 heads per metal, majority vote at inference — fixes Wf-131 P2 multi-task backbone signal leak)
   - **Wasserstein transfer target = Rh real active distribution** (NOT synthetic cohort — fixes Wf-123 M1 transfer direction)
2. `molmetal/tests/test_metal_lomo_v2.py` (~16 tests, target ≥16 PASS)
3. `molmetal/reports/wf135_p2_lomo_v2/{metal}/per_seed_metrics.json` (5 metals × 5 seeds × per-fold metrics)
4. `molmetal/reports/wf135_p2_lomo_v2/synthesis.md` (per-metal fold AUC + LOMO mean + honest framing)
5. `TODO/completed/135_wf135_phase2_closeout.md` (Phase 2 close-out)

### §2.2 Phase 2 acceptance gates (per IMPROVEMENT_ENUM §1.4)

**Hard gates (HALT if violated)**:
- 5/5 metals × 5 seeds × 50 epochs = 1250 runs completed
- Rh fold AUC ≥ 0.30 (recover from Wf-131 P2 regression 0.136; baseline was 0.368)
- Overall LOMO mean ≥ 0.60 (vs baseline 0.5346)
- ≥16 tests PASS in `test_metal_lomo_v2.py`

**Soft gates (PARTIAL if violated)**:
- Rh 0.30–0.50 → honest frame as PARTIAL (vs target ≥ 0.30)
- LOMO < 0.60 → honest frame as PARTIAL
- Multi-task backbone signal leak observed → PARTIAL (per-metal head ensemble is the structural fix; if it's still leaking, that's a deeper issue)

### §2.3 USER_OK gate (per `mol-metal-gpu-execution` §0)

USER must explicitly approve:
1. **Compute**: ~10 GPU-hours on GPU-1 RTX 5090 (Blackwell sm_120 environment). Confirms environment probe via `check_blackwell_env.py` first.
2. **Schedule**: 2 calendar weeks wall budget. If GPU-1 contention with kanzi means longer, USER_OK to either (a) wait, (b) drop to 3-seed × 30-epoch (acceptable scope shrink), or (c) cancel Phase 2 and ship honest-negative close-out.
3. **Acceptance criteria**: Rh fold AUC ≥ 0.30 + LOMO mean ≥ 0.60. If Rh stays flat (0.10–0.20), USER_OK to ship honest-negative framing similar to TODO-131 (Rh sample size n=60 is structural ceiling).
4. **Reporting**: LABELING-STRICT (MEASURED-vs-PROJECTED). Per `party-twenty-spirit` §3.5 + `mol-metal-gpu-execution` §0, every result must label whether it is MEASURED from GPU run or PROJECTED from prior work. No mixing.
5. **Risk fallback**: If Phase 2 does not lift Rh ≥ 0.30, paper §4.5.6 lead candidate is forgone; §6 honest-negative framing applies. NCI-60 9-cell 0.8007 + single-cell SOTA in §4 is unaffected.

### §2.4 Wall-time breakdown (honest estimate)

| Activity | Wall-time | Notes |
|---|---|---|
| `check_blackwell_env.py` probe + GPU-1 lock | 0.5 day | Confirm sm_120 + contention check |
| Cohort + metal-cohort ingestion prep | 0.5 day | Verify Phase 1 cohort loads cleanly into LOMO splitter |
| `train_metal_lomo_v2.py` scaffold (skeleton + CLI + tests) | 2 days | Mirror Wf-124 P3 architecture (multi-task D-MPNN + Wasserstein pIC50 reg) |
| 5-metal × 5-seed × 50-epoch training runs | 10 GPU-hours = ~2 calendar days | GPU-1 single-tenant; if kanzi contention, scale up |
| Metric JSON emission + synthesis.md drafting | 1 day | Per-fold AUC, per-seed std, honest framing |
| Regression vs Wf-131 P2 baseline | 1 day | Confirm 0.136 → ≥ 0.30 lift or honest-negative |
| Tests + close-out doc | 0.5 day | ≥16 tests + `135_wf135_phase2_closeout.md` |
| **Total wall** | **~7–8 calendar days** (best case), **~14 days** (with delays + retries) | Honest upper bound: **2 weeks** |

### §2.5 Deferral plan (what USER_OK should look like)

USER has 3 paths:

**Path A — Run Phase 2 (USER_OK)**:
- Approve GPU-1 lock + 2-week wall + acceptance gates.
- Phase 2 runs as §2.1.
- Close-out: `TODO/completed/135_wf135_phase2_closeout.md`.

**Path B — Honest deferral + scale shrink (USER_OK)**:
- Approve GPU-1 lock but only 1-week wall.
- Scope shrink: 3-metal × 3-seed × 30-epoch (270 runs = ~3 GPU-hours).
- Acceptance gate scaled: Rh fold AUC ≥ 0.25 (PARTIAL honest frame acceptable).
- Close-out: `TODO/completed/135_wf135_phase2_closeout.md` marked SCOPE_SHRUNK.

**Path C — Permanent deferral (USER_OK)**:
- Acknowledge Phase 2 will not run in current round.
- Ship `TODO/completed/135_m19_lomo_rh_oversample_fix.md` (already on disk) as final close-out.
- Mark TODO-135 as DEFERRED_PERMANENT in `TODO/INDEX.md`.
- Rh remains a §6 honest-negative limitation in paper (similar to TODO-131 framing).

### §2.6 Honest framing on this Phase 2 plan

1. **Phase 2 is not yet a runnable artifact** — `train_metal_lomo_v2.py` does not exist on disk. Estimated 280 LOC + 16 tests to scaffold; not started.
3. **Rh sample size n=60 train is structural ceiling** — oversample + ensemble cannot fix fundamental sample-size limit. Honest projection: 50% likelihood Rh AUC lifts to ≥ 0.30 if Phase 2 runs on GPU.
4. **Per-metal head ensemble is a structural fix, not a magic fix** — addresses multi-task backbone signal leak (Wf-131 P2 root cause ~5%); does not address label bias (~60%, already fixed Phase 1), Wasserstein direction (~20%, target fix in §2.1), or cohort granularity (~15%, inherent to closure BFS).
5. **Phase 2 alone may not satisfy paper §4.5.6** — if Rh remains flat, §6 honest-negative framing is the right move, not silent abandonment.

---

## §3 Cross-references

- **`TODO/completed/135_m19_lomo_rh_oversample_fix.md`** — comprehensive close-out (this doc is a focused Phase 2 plan supplement)
- **`TODO/completed/143_wf139_p7_rh_active_cohort.md`** — Phase 1 per-step close-out
- **`TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` §1.4** — Phase 2 acceptance gates (source of truth)
- **`TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` §0 row 4** — TODO-135 queue entry
- **`TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` §2 Step 3** — Phase 2 deferred to GPU window
- **Wf-131 P2** — 500 synthetic compounds (constant-inactive BUG → fixed by Wf-139 P7)
- **Wf-123 M1** — wasserstein_transfer_loss (target = synthetic → real Rh actives fix in §2.1)
- **Wf-124 P3** — multi-task D-MPNN + Wasserstein pIC50 reg architecture template for `train_metal_lomo_v2.py`
- **Wf-125 P1 + P2** — Rh oversample + LOMO rerun (separate ships, also covered by Wf-139 P7 + Wf-135 deferred)
- **`mol-metal-gpu-execution` skill** — USER_OK gate, MEASURED-vs-PROJECTED discipline, environment probe
- **`party-twenty-spirit` §3.5** — honest-negative framing discipline

---

## §4 Final verdict

**Wf-135 / TODO-135 status**:

| Phase | Status | Evidence |
|---|---|---|
| Phase 1 (cohort) | ✅ SHIPPED | 15/15 tests + 500 balanced + scaffold-overlap=0 |
| Phase 2 (LOMO 5×5×50) | 🚫 DEFERRED to GPU window | USER_OK gate + 2-week wall + acceptance gates per §2.2 |

**Action items for USER (any time)**:
1. Approve one of §2.5 Path A / B / C (or another).
2. If Path A or B: confirm GPU-1 RTX 5090 is free (no kanzi contention).
3. If Path A or B: ack acceptance gates + honest framing per §2.6.

**Action items for code (only after USER_OK on Path A or B)**:
1. Scaffold `molmetal/scripts/train_metal_lomo_v2.py` (~280 LOC).
2. Scaffold `molmetal/tests/test_metal_lomo_v2.py` (~16 tests).
3. Run `molmetal/scripts/check_blackwell_env.py` first.
4. Run training, emit `molmetal/reports/wf135_p2_lomo_v2/{metal}/per_seed_metrics.json`.
5. Draft `molmetal/reports/wf135_p2_lomo_v2/synthesis.md` + `TODO/completed/135_wf135_phase2_closeout.md`.

**No code action this session** — Phase 1 is done, Phase 2 is gated on USER_OK + GPU availability.