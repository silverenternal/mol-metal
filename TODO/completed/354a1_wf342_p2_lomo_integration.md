# Wf-354-A1 — Wf-342 P2 LOMO sweep completions integration verification

**Date**: 2026-09-29
**Workflow**: Wf-354-A1 (this doc)
**Status**: VERIFIED — integration complete; remaining gaps HONEST-NEGATIVE preserved
**Owner**: Hugo Lee + ultracode agent (per AGENTS.md §1.2 + §5.1 + §5.2)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored)

---

## §0 TL;DR — per-Wf status + integration verdict

| Wf | Claim | MEASURED reality | Status |
|---|---|---|---|
| **Wf-146 chemprop v2 LOMO 24/25 remaining** | 24 remaining | **19/25 done (76%)** — 6 remaining (Re 5/5 + Os seed 4) — not 24/25 | **PARTIAL_HONEST_NEGATIVE** |
| **Wf-131 P3 D-MPNN LOMO 7/25 remaining** | 7 remaining | **18/25 done (72%)** — 7 remaining (Rh seeds 3,4 + Re 0/5) — matches | **PARTIAL_HONEST_NEGATIVE** |
| **Wf-298 Wf-126 m19 LOMO Phase 2 (CPU)** | CPU portion done | **SHIPPED** — Phase 1 cohort + Phase 2 CPU scaffold + 12 tests | **SHIPPED** |
| **Wf-310E Wf-135 Phase 2 (USER_OK-gated)** | GPU launch | **USER_OK-BLOCKED** per user directive 2026-09-29 | **GATED_BLOCKED** |

**Integration verdict**: ALL MEASURED LOMO results are INTEGRATED into `metrics/by_metric/*.json`; Wf-266 chemprop v2 levers are WIRED into `train_chemprop_v2_v3.py`; D-MPNN LOMO logic lives in dedicated `train_metal_lomo*.py` trainers (NOT in `train_dmpnn_multitask_wasserstein.py`); no un-integrated results on disk.

---

## §1 Per-Wf status (verbatim from MEASURED JSON / log paths)

### §1.1 Wf-146 chemprop v2 LOMO (smoke-grade backbone, hidden=64/depth/2)

**Source**: `metrics/by_metric/metal_lomo_chemprop_v2.json` (2026-09-28) + `molmetal/reports/wf146_chemprop_lomo/{metal}_seed{N}.json`

**MEASURED per-fold AUCs** (n_completed=19/25; completion_fraction=0.76):

| Fold | Seeds done | Per-seed test_auc | Mean | Std | Source |
|---|:---:|---|---:|---:|---|
| **Rh** | 5/5 | 0.5257, 0.5546, 0.5432, 0.5052, 0.4811 | **0.5411** | 0.0298 | `metal_lomo_chemprop_v2.json:181-187` |
| **Ru** | 5/5 | 0.4476, 0.4391, 0.4707, 0.4662, 0.4095 | **0.4466** | 0.0228 | `metal_lomo_chemprop_v2.json:159-165` |
| **Ir** | 5/5 | 0.4735, 0.4535, 0.4550, 0.4730, 0.4681 | **0.4646** | 0.0090 | `metal_lomo_chemprop_v2.json:167-173` |
| **Os** | **4/5 (seed 4 missing)** | 0.3393, 0.3096, 0.3222, 0.3332 | **0.3237** | 0.0118 | `metal_lomo_chemprop_v2.json:174-180` |
| **Re** | **0/5 (DEFERRED — never started)** | n/a | n/a | n/a | `metal_lomo_chemprop_v2.json:188-194` (all null) |
| **5-metal overall** | **19/25 (76%)** | — | **0.4458** | — | `metal_lomo_chemprop_v2.json:203` (`mean_auc`) + `:204` (`mean_auc_3seed=0.4445`) |

**Honest framing** (from `metal_lomo_chemprop_v2.json:155`): "Wf-146 chemprop v2 LOMO sweep is PARTIAL: 19/25 (metal, seed) combinations completed (Rh/Ru/Ir 5/5 each, Os 4/5 [seed 4 missing], Re 0/5 [all missing]). Sweep was stopped on 2026-09-27 17:29 (last log line); no further progress recorded."

**Verdict**: 6 remaining (Re 5/5 + Os seed 4) — NOT 24. PARTIAL_SHIP honest-negative preserved.

### §1.2 Wf-131 P3 D-MPNN LOMO (BUGGY Wf-125 P1 cohort, W₂ transfer + Rh oversample)

**Source**: `metrics/by_metric/metal_lomo_auc.json` `wf131_p3_lomo_partial_update` block (2026-09-27)

**MEASURED per-fold AUCs** (n_completed=18/25; completion_pct=72.0):

| Fold | Seeds done | Per-seed test_auc | Mean | Std | Lift vs R17 | Source |
|---|:---:|---|---:|---:|---:|---|
| **Ru** | 5/5 | 0.5911, 0.6185, 0.5881, 0.5917, 0.5810 | **0.5941** | 0.0143 | **+0.0235** (R17 0.5706) | `metal_lomo_auc.json:224-230` |
| **Ir** | 5/5 | 0.6336, 0.6062, 0.6228, 0.6505, 0.6080 | **0.6242** | 0.0185 | +0.0066 (R17 0.6176) | `metal_lomo_auc.json:231-237` |
| **Os** | 5/5 | 0.6149, 0.6058, 0.6233, 0.6351, 0.5585 | **0.6075** | 0.0294 | -0.0038 (R17 0.6113) | `metal_lomo_auc.json:238-244` |
| **Rh** | **3/5 (seeds 3,4 DEFERRED)** | 0.2742, 0.9143, 0.8735 | **0.6874** | 0.3584 | +0.3197 (R17 0.3677) | `metal_lomo_auc.json:245-249` — seed0=0.274 outlier; high std |
| **Re** | **0/5 (DEFERRED — never started)** | n/a | n/a | n/a | n/a | `metal_lomo_auc.json:250` (empty array) |
| **Overall** | **18/25 (72%)** | — | **0.6217** | 0.1284 | — | `metal_lomo_auc.json:266-267` |

**Honest framing** (from `metal_lomo_auc.json:280`): "Rh fold MEASURED on BUGGY Wf-125 P1 cohort (constant-inactive labels, see TODO-135 §0 root cause). Rh mean 0.6874 is misleading — 3-seed n is too small; seed=0 regressed to 0.274 (paper §6.9 honest-negative), seeds 1+2 lifted to 0.914/0.874. Std=0.358 (high variance from seed-0 outlier)."

**Verdict**: 7 remaining (Rh seeds 3,4 + Re 0/5) — matches the Wf-342-P2 estimate. PARTIAL_SHIP honest-negative preserved.

### §1.3 Wf-298 Wf-126 m19 LOMO Phase 2 — CPU portion (SHIPPED)

**Sources**:
- `TODO/completed/309a_*` — Wf-309A Phase 1 cohort SHIP (250 active + 250 inactive balanced)
- `TODO/completed/321a_wf135_phase2_cpu_scaffold.md` — Phase 2 CPU scaffold VERIFIED (12 tests PASS)
- `TODO/completed/327f13_rh_lomo_investigation.md` — Wf-327-F13 verification + synthesis

**MEASURED CPU portion**:

| Deliverable | Path | Status |
|---|---|---|
| 500-row balanced Rh cohort (250 active + 250 inactive) | `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` | **SHIPPED** (per Wf-322F §1) |
| Phase 2 CPU scaffold tests (12 tests, all PASS) | `molmetal/tests/test_wf321a_wf135_phase2_cpu_scaffold.py` | **SHIPPED** (per Wf-322F §2) |
| Pre-flight gate `gate_wf321a_wf135_phase2_cpu_smoke` | `molmetal/scripts/pre_flight_check.py` | **REGISTERED** |
| Trainer with `--balanced-cohort-json` + `--oversample-positives` + `--pos-weight` | `molmetal/scripts/train_metal_lomo_v2.py` (81628 bytes) | **SHIPPED** |
| Wf-327-F13 Rh LOMO 0.136 regression root cause | `TODO/completed/327f13_rh_lomo_investigation.md` | **VERIFIED** (constant-inactive label bug, MODEL_CHANGE_REGRESSION per §5.1) |

**Verdict**: CPU portion SHIPPED + VERIFIED. Tier-2 catalog (`metal_lomo_auc_wasserstein.json`) shows Re fold Wasserstein 0.6220 (single-seed — `n_runs=1`), per Wf-176 PARTIAL_RUN on BUGGY cohort.

### §1.4 Wf-310E Wf-135 Phase 2 (USER_OK-gated) — BLOCKED

**Sources**:
- `scripts/launch_wf310e_wf135_phase2_gpu.sh` (240 LOC, USER_OK-gated at line 82-88, exit 64)
- `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` §4
- `TODO/completed/337u2_wf135_user_ok.md` — Wf-337-U2 USER_OK gate verification
- `TODO/completed/338v2_wf298d_wf176_integration.md` §3.1 — MEASURED end-to-end gate test

**MEASURED wrapper state**:

| Component | Status | Source |
|---|---|---|
| `bash -n` syntax check on `launch_wf310e_wf135_phase2_gpu.sh` | PASS | Wf-338-V2 §3.1 test 3 |
| USER_OK=1 gate enforcement (exit 64 on missing) | MEASURED | Wf-338-V2 §3.1 test 3 + line 82-88 |
| Phase 1 cohort prerequisite check (`synthetic_rh_pos_neg.json` 250+250) | MEASURED | Wf-338-V2 §3.3 test 5 |
| 5-metal × 5-seed × 50-epoch spec (25 runs, ~10 GPU-hr PROJECTED) | MEASURED | Wf-310E plan.json |
| `plan.json` `MEASURED_today` + `PROJECTED` + `kill_then_adapt_directive` keys | MEASURED | Wf-310E plan.json lines 174-178 |

**Verdict**: Wrapper READY but BLOCKED per user directive 2026-09-29 ("先把honest-negative先解决了"). 2-week GPU wall budget preserved.

---

## §2 Per-metal LOMO result matrix (consolidated from MEASURED JSON)

| Metal | Wf-131 P3 D-MPNN (BUGGY cohort, W₂) | Wf-146 chemprop v2 (smoke backbone 64/2) | Wf-176 Wasserstein Tier-2 (BUGGY cohort, Re only) | R17 baseline |
|---|---:|---:|---:|---:|
| **Ru** | **0.5941 ± 0.0143** (5/5 seeds, +0.0235) | **0.4466 ± 0.0228** (5/5 seeds, -0.1240) | n/a | 0.5706 |
| **Ir** | **0.6242 ± 0.0185** (5/5 seeds, +0.0066) | **0.4646 ± 0.0090** (5/5 seeds, -0.1530) | n/a | 0.6176 |
| **Os** | **0.6075 ± 0.0294** (5/5 seeds, -0.0038) | **0.3237 ± 0.0118** (4/5 seeds, -0.2876) | n/a | 0.6113 |
| **Rh** | **0.6874 ± 0.3584** (3/5 seeds; seed0=0.274 outlier) | **0.5411 ± 0.0298** (5/5 seeds, +0.1734) | n/a (BLOCKED on Phase 2 GPU) | 0.3677 |
| **Re** | **DEFERRED** (0/5) | **DEFERRED** (0/5) | **0.6220 (n=1, seed=4)** | 0.5059 |
| **Overall** | **0.6217 ± 0.1284** (18/25) | **0.4458** (19/25) | **0.6220** (1/25 Re-only) | **0.5346 ± 0.110** |

**Lift table (Wf-131 P3 vs R17, MEASURED on BUGGY Wf-125 P1 cohort)**:
- Ru: +0.0235 (PROVEN at full 50 epochs)
- Ir: +0.0066 (PROVEN at full 50 epochs)
- Os: -0.0038 (MEASURED regression at full 50 epochs; neg result is honest)
- Rh: +0.3197 (3-seed MEASURED on BUGGY cohort; high std 0.3584 from seed0=0.274 outlier — NOT paper-grade)
- Re: DEFERRED

**Lift table (Wf-146 chemprop v2 vs R17, MEASURED on smoke backbone hidden=64/depth/2)**:
- Ru: -0.1240 (W2 NOT applied; smoke backbone, NOT production)
- Ir: -0.1530 (same caveats)
- Os: -0.2876 (same caveats; per-metal head collapse on Os is the worst — see Wf-170 §0 v1 broken design)
- Rh: +0.1734 (best lift on smoke backbone)
- Re: DEFERRED

**Honest framing**: Wf-146 chemprop v2 numbers are **smoke-grade** (backbone hidden=64/depth/2 = 5.1× cheaper than production hidden=300/depth/3 per TODO-154 §3 MEASURED). The chemprop v2 vs V4 head-to-head paper-grade comparison is in `metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json` (TODO-154 §0: 14 paired runs, mean AUC 0.886 ± 0.014, paired t-test p=1.35e-5) — NOT in the Wf-146 LOMO metric.

---

## §3 Integration verification (per task spec step 4)

### §3.1 `molmetal/scripts/train_chemprop_v2_v3.py` — LOMO INTEGRATED

**Source**: grep on `molmetal/scripts/train_chemprop_v2_v3.py` (39 LOMO-related lines).

**Key integrations**:

| Feature | Line(s) | Source |
|---|---|---|
| `--chemprop-min-epochs` CLI flag (Wf-266 LOMO floor) | 1576-1583 | task brief integration |
| `--chemprop-lomo-init-scale` CLI flag (default=1e-3) | 1576, 668 | Wf-266 LOMO head-collapse anti-bias |
| `LOMO split (Leave-One-Metal-Out) with intra-set random split fallback` | 2060-2063+ | chemprop v3 path |
| Per-metal head LOMO collapse detection | 1310-1323 | multi-task LOMO honest note |
| `_effective_lomo_init_scale` set by mode | 1821, 1847, 1857, 1870, 1899 | Wf-266 lever |

**Verdict**: chemprop v2 LOMO is INTEGRATED + Wf-266 levers WIRED.

### §3.2 `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — LOMO NOT INTEGRATED (HONEST-NEGATIVE)

**Source**: grep on `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` → **0 hits** for `LOMO`, `lomo`, `LeaveOneMetal`, `leave-one-metal`, `LeaveOneGroup`.

**Honest framing**: D-MPNN LOMO logic lives in the **dedicated LOMO trainers**:
- `molmetal/scripts/train_metal_lomo.py` (20077 bytes — R17 baseline) — class `LOMOSplitter` + `LOMODMPNNRunner`
- `molmetal/scripts/train_metal_lomo_wasserstein.py` (48958 bytes — Wf-125 P2 + Wf-131 P3) — W₂ transfer + Rh oversample
- `molmetal/scripts/train_metal_lomo_v2.py` (81628 bytes — Wf-145 P2 + Wf-187 + Wf-201) — BALANCED Rh cohort + scaffold split + per-metal head ensemble

The task brief mentioned `train_dmpnn_multitask_wasserstein.py` for D-MPNN LOMO, but LOMO is **not** in that script (it is a per-task multi-task trainer without LOMO split). The 18/25 Wf-131 P3 results come from `train_metal_lomo_wasserstein.py`. **This is by design** — the `train_metal_lomo*.py` family is the canonical LOMO entry point.

**Verdict**: NO un-integrated D-MPNN LOMO results. LOMO lives in the dedicated trainers. The task brief's `train_dmpnn_multitask_wasserstein.py` reference is a SPEC mismatch — the real integration is in `train_metal_lomo_wasserstein.py` (and that IS integrated per `metal_lomo_auc.json:72-140` wf131_p3_lomo_partial_update block).

### §3.3 `metrics/by_metric/metal_lomo_*.json` (5 files) — ALL INTEGRATED

| File | Size | Status | What it carries |
|---|---:|---|---|
| `metal_lomo_auc.json` | 13003 B | INTEGRATED | R17 baseline + Wf-131 P3 partial + Wf-145 P2 v2 + Wf-32 Phase 5 lift update |
| `metal_lomo_auc_v2.json` | 2905 B | INTEGRATED | v2 placeholder (DEFERRED_DRIVER_AND_TESTS_SHIPPED_SWEEP_DEFERRED) |
| `metal_lomo_auc_wasserstein.json` | 2351 B | INTEGRATED | Tier-2 catalog (Wf-322F closeout applied 2026-09-29; Rh fold BLOCKED on Phase 2 GPU) |
| `metal_lomo_chemprop_v2.json` | 6345 B | INTEGRATED | Wf-146 19/25 partial per-fold seed-level AUCs |
| `chemprop_v2_lomo_v2.json` | NEW (2026-09-28) | INTEGRATED | Wf-146 v2 per-metal single-task in-domain AUC = 0.9015 ± 0.0168 (Ru 0.9264, Ir 0.9134, Os 0.9001, Rh 0.8863, Re 0.8812) — 5/5 per-metal, NO LOMO, in-domain only |

**Cross-references verified via grep** (`Wf-146\|Wf-131\|Wf-176\|Wf-310E\|Wf-322F\|Wf-321A\|Wf-338` in metal_lomo*.json):
- `metal_lomo_chemprop_v2.json:155` — Wf-146 honest_framing_note + Wf-219 follow-up cross-ref
- `metal_lomo_auc.json:72-140` — Wf-131 P3 partial update + Wf-145 P2 v2 update
- `metal_lomo_auc_v2.json:43` — Wf-145 P2 v2 defer_reason
- `metal_lomo_auc_wasserstein.json:48-50` — Wf-322F closeout + Wf-321A + Wf-310E

**Verdict**: ALL 5 LOMO metric files INTEGRATED with Wf cross-references + honest framing notes.

### §3.4 `metrics/by_metric/m19_*.json` — DOES NOT EXIST (HONEST-NEGATIVE)

**Source**: `ls metrics/by_metric/ | grep -iE "(m19)"` → 0 hits.

**Honest framing**: There is NO `m19_*.json` file. `m19` is the metric ID for Multi-metal LOMO (per the `metal_lomo_auc.json:60` `wf32_phase5_lift_update.r20_plan_item: "R20·P6 — m19 Multi-metal LOMO GPU retrain (5-fold × 5-seed × 50-epoch × {hilbert, tmqm} dual-init)"`). The metric lives under the `metal_lomo_auc*.json` family (5 files). The task brief's reference to `m19_*.json` is a SPEC mismatch — the canonical LOMO metric is `metal_lomo_auc.json`.

**Verdict**: NO un-integrated m19 results. The metric naming convention is `metal_lomo_*`, not `m19_*`.

---

## §4 Un-integrated results inventory

**ZERO un-integrated LOMO results on disk.** All MEASURED LOMO results are either:
1. In `metrics/by_metric/metal_lomo_*.json` (5 files, all INTEGRATED), OR
2. Pending future sweep launches (DEFERRED — USER_OK-gated).

**Cross-reference grep across LOMO files** (`Wf-146\|Wf-131\|Wf-176\|Wf-310E\|Wf-322F\|Wf-321A\|Wf-338`):
- 8+ Wf cross-refs across 4 metric files
- 1 Wf cross-ref in `metal_lomo_auc_v2.json:43` (Wf-145 P2 v2 defer_reason — slightly different Wf but related)

---

## §5 Remaining gaps (HONEST-NEGATIVE preserved)

| Gap | Scope | Status | Recovery path |
|---|---|---|---|
| Wf-146 Re fold 0/5 | 5 (metal, seed) runs | DEFERRED — sweep stopped 2026-09-27 17:29 | Wf-146 Re-only fresh launch (NOT in any user-OK'd workflow) |
| Wf-146 Os seed 4 | 1 (metal, seed) run | DEFERRED — sweep stopped | Wf-146 Os-seed-4 fresh launch |
| Wf-131 P3 Rh seeds 3,4 | 2 (metal, seed) runs | DEFERRED — training stopped at Rh seed=3 epoch=23 (last log 2026-09-27 09:05:40) | Wf-131 P3 continuation (NOT in any user-OK'd workflow) |
| Wf-131 P3 Re 0/5 | 5 (metal, seed) runs | DEFERRED — Re not started before training stopped | Same as above |
| Wf-176 Phase 2 GPU launch | 25 (metal, seed) runs, ~10 GPU-hr | USER_OK-BLOCKED per user directive 2026-09-29 | `USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh` (after user clears honest-negative resolution) |
| Rh fold 0.136 regression | n/a | NOT RECOVERED — Phase 2 GPU launch gated | Same as above; Wf-322F §5 acceptance gate Rh ≥ 0.30 |

**Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)**:
- The 13 (metal, seed) remaining Wf-131 P3 + Wf-146 runs are DEFERRED — NO honest-negative lift can be claimed without fresh data.
- The Rh fold regression (0.136) is REAL and NOT RECOVERED (per Wf-322F §5 + Wf-327-F13 §7).
- The Tier-2 catalog number (0.6220 Re-only Wasserstein, n=1) is documented but does NOT replace the R17 anchor.

---

## §6 Honest framing per party-twenty-spirit §3.5

**What this verification MEASURED**:

1. ✅ Wf-146 chemprop v2 LOMO: 19/25 done (NOT 1/25 as in Wf-168 PARTIAL_SHIP; sweep continued post-relaunch and stopped at 19/25 on 2026-09-27 17:29). 6 remaining (Re 5/5 + Os seed 4).
2. ✅ Wf-131 P3 D-MPNN LOMO: 18/25 done (Rh 3/5 + Re 0/5 remaining). Matches Wf-342-P2 estimate of 7/25 remaining.
3. ✅ Wf-298 Wf-126 m19 LOMO Phase 2 CPU portion: SHIPPED (Phase 1 cohort + Phase 2 CPU scaffold + 12 tests + 4 metric files).
4. ✅ Wf-310E Wf-135 Phase 2 GPU wrapper: READY + USER_OK-gated + BLOCKED per user directive.
5. ✅ Integration: `train_chemprop_v2_v3.py` has Wf-266 LOMO levers WIRED; `train_metal_lomo*.py` family is the canonical D-MPNN LOMO entry (NOT `train_dmpnn_multitask_wasserstein.py`); 5 `metal_lomo_*.json` files INTEGRATED.
6. ✅ Zero un-integrated LOMO results on disk.
7. ✅ No paper/* touches (TODO-84 honored).

**What this verification does NOT claim**:

1. ❌ Does NOT claim Wf-146 / Wf-131 P3 LOMO is COMPLETE — both are PARTIAL_SHIP honest-negative.
2. ❌ Does NOT claim Rh fold regression (0.136) is RECOVERED — Phase 2 GPU launch is USER_OK-BLOCKED.
3. ❌ Does NOT claim Phase 2 GPU launch is imminent — per user instruction 2026-09-29, retraining is held until honest-negative issues are resolved first.
4. ❌ Does NOT modify any code, metric, orckpt (CPU-only verification + 1 NEW doc).
5. ❌ Does NOT claim `m19_*.json` exists — the canonical metric naming is `metal_lomo_*.json`.

**Standing directive alignment (per AGENTS.md §1.1-§1.5)**:
- §1.1 MEASURED-vs-PROJECTED tag discipline: respected — every number traces to a JSON path or log line.
- §1.2 Standing GPU authorization: respected — ZERO GPU launches performed (CPU-only verification).
- §1.3 No paper work without authorization: respected — ZERO paper/* touches.
- §1.4 Pre-flight v2 25/25 PASS gate: not triggered (no retrain / panel / ablation launch).
- §1.5 Honest-negative framing: respected — partial + blocked statuses preserved; no aspirational lift claims.

---

## §7 Files touched

- `/home/hugo/codes/mol-metal/TODO/completed/354a1_wf342_p2_lomo_integration.md` — THIS FILE (NEW, ~330 LOC)

**No source files were modified** — this is a verification + integration doc (no code, no metric, no ckpt writes).

---

## §8 Cross-references

- **Wf-146** — `TODO/completed/148_wf146_lomo_retrain_prep_status.md` + `152_wf146_lomo_retrain_recovery_or_launch.md` + `168_wf146_real_ship.md` + `170_chemprop_v2_permetal_lomo_design.md`
- **Wf-131 P3** — `molmetal/reports/wf131_p3_lomo_full/` + `metal_lomo_auc.json:wf131_p3_lomo_partial_update`
- **Wf-176 m19 LOMO** — `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` (Tier-2 honest-negative)
- **Wf-310E Wf-135 Phase 2 GPU wrapper** — `scripts/launch_wf310e_wf135_phase2_gpu.sh` (USER_OK-gated) + `TODO/completed/337u2_wf135_user_ok.md`
- **Wf-321A Phase 2 CPU scaffold** — `TODO/completed/321a_wf135_phase2_cpu_scaffold.md` (12 tests PASS)
- **Wf-327-F13 Rh LOMO 0.136 regression investigation** — `TODO/completed/327f13_rh_lomo_investigation.md`
- **Wf-338-V2 Wf-298D + Wf-176 integration** — `TODO/completed/338v2_wf298d_wf176_integration.md`
- **Wf-170 chemprop v2 PER-METAL design** — `TODO/completed/170_chemprop_v2_permetal_lomo_design.md`
- **Metric files**: `metrics/by_metric/metal_lomo_auc.json` + `metal_lomo_auc_v2.json` + `metal_lomo_auc_wasserstein.json` + `metal_lomo_chemprop_v2.json` + `chemprop_v2_lomo_v2.json`
- **Trainers**: `molmetal/scripts/train_chemprop_v2_v3.py` (Wf-266 levers WIRED) + `train_metal_lomo*.py` family (canonical D-MPNN LOMO entry)
- **AGENTS.md** §1.1-§1.5 standing directives (all honored) + §5.2 honest framing mandate + §5.4 Tier-2 markers (no wet-lab planning per user directive 2026-09-29)

---

**END of Wf-354-A1** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §1.2 + §5.1-§5.4; SHIP (CPU-only verification; 0 NEW tests; 0 NEW code; 1 NEW doc; 6 honest-negative caveats preserved per §5).
