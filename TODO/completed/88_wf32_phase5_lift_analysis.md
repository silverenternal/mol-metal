# TODO-88 — Wf-32 Phase 5 m16/m19/m8 MEASURED vs R17 baseline lift close-out

**Date**: 2026-09-26
**Workflow**: Wf-32 (Phase 5 — lift analysis close-out)
**Status**: ✅ **SHIP — full honest-negative; 0/3 lift vs R17 baseline**
**Priority**: HIGH (closes the Wf-32 deliverable loop on the metric side; unblocks R20 plan creation)
**Owner**: Hugo Lee + ultracode agent
**Parent**: `Wf-32` (Phase 5 of multi-phase rollout; Phase 4 was the deferred GPU sweep)

---

## §0 TL;DR

Wf-32 Phase 5 produces the MEASURED-vs-R17-baseline lift comparison for the 3 metrics in scope (m16 / m19 / m8). The honest verdict is **full honest-negative across the board**:

| Metric | R17 baseline | R18 MEASURED | Lift | Verdict | R20 plan |
|---|---:|---:|---:|:---:|:---:|
| m16 Affinity Pearson r | 0.207 | −0.040 (smoke) | **−0.247** | ❌ HONEST-NEGATIVE | R20·P5 |
| m19 Multi-metal LOMO | 0.5346 ± 0.110 | DEFER (no data) | n/a | ⏸️ DEFERRED-NO-DATA | R20·P6 |
| m8 PB-Valid | 0.846 (1p×26c×26check) | 1.000 (1p×1c×12check smoke) | +0.154 (artifact) | ❌ HONEST-NEGATIVE-on-PRODUCTION | R20·P7 (revised) |

R17 baselines are preserved as canonical values for paper §4 Table 2. All 3 metrics become R20 plan items. No fabricated headline values exist anywhere in this Phase 5 deliverable.

---

## §1 Pre-condition (Phase 4 ship)

Per the Phase 4 ship report (`TODO/completed/87_wf32_phase4_honest_negative.md`), Phase 4 of Wf-32 was **deferred** because GPU 1 (RTX 5090, 32 GB) was occupied by a concurrent Wf-31 m1 Ru-AUC SE(3) SB run (PID 3853986) and immediately before by a Wf-29 train_se3sb.py smoke (PID 3802039). The 5-seed × 30/50-epoch GPU 1 sweep would have collided.

Off-GPU scaffolding was fully shipped:
- `molmetal/scripts/train_affinity_regression.py` (Wf-32 Phase 1, m16 trainer)
- `molmetal/scripts/train_multimetal_lomo.py` (Wf-32 Phase 2, m19 trainer)
- `molmetal/scripts/bench_pb_valid.py` (Wf-32 Phase 3, m8 bench)
- 3 verify scripts (all green) + 18 unit tests (5 pass, 13 skip awaiting full Phase 4 outputs)
- 6 placeholder JSONs + 3 summary JSONs in `molmetal/reports/wf32_m{16,19,8}/`

The on-disk R18 MEASURED numbers we have today come from **smoke-scale runs only** (not the full Phase 4 sweep).

---

## §2 What Phase 5 delivered

### §2.1 Lift analysis JSON

`molmetal/reports/wf32_m16_m19_m8_lift_analysis.json` — single comprehensive JSON covering all 3 metrics:

- **Top-level keys**: metric / description / generated_utc / workflow / phase / predecessor_phase / predecessor_summaries / data_sources / metrics / aggregate_lift_summary / honest_negative_provenance / r20_plan_summary / ship_ledger
- **`metrics[]`**: 3 entries (m16, m19, m8), each with `r17_baseline` / `r18_wf32_measured` / `lift_r18_minus_r17` / `pass_target_r18` / `sota_target` / `ship_gate` / `ship_gate_passed` / `verdict` / `verdict_detail` / `r20_plan_item`
- **`aggregate_lift_summary`**: 0/3 lift above baseline, 0/3 passed ship gate, 3/3 honest-negative, R17 baselines unchanged and retained as canonical

### §2.2 Lift summary MD

`molmetal/reports/wf32_m16_m19_m8_lift_summary.md` — Markdown close-out (this sibling JSON's narrative counterpart):

- §1 m16 detail (R17 0.207 / Wf-32 smoke −0.040 / lift −0.247)
- §2 m19 detail (R17 0.5346 / Wf-32 DEFER / lift n/a)
- §3 m8 detail (R17 0.846 / Wf-32 smoke 1.000 12-check / lift +0.154 smoke-to-smoke artifact of reduced check scope)
- §4 Aggregate verdict (full honest-negative)
- §5 R20 plan items (R20·P5 / R20·P6 / R20·P7 revised)
- §6 Honest framing (per party-twenty-spirit §3.5)
- §7 Cross-reference (8 entries)

### §2.3 Metrics JSON updates (3 files, append-only)

**`metrics/by_metric/pearson_r_pic50.json`** — appended 5th measurement (`wf32_phase4_smoke`) + top-level `wf32_phase5_lift_update` (verdict HONEST-NEGATIVE, R17 baseline 0.207 retained, R20·P5 referenced). The R17 baseline (`aggregate.best_neural = 0.207`) is preserved.

**`metrics/by_metric/metal_lomo_auc.json`** — appended top-level `wf32_phase5_lift_update` (verdict DEFERRED-NO-DATA, R17 baseline 0.5346 retained, R20·P6 referenced, GPU 1 occupancy defer reason). The R17 baseline (`value = 0.5346340730251975`) is preserved.

**`metrics/by_metric/pb_pass_rate.json`** — appended top-level `wf32_phase5_lift_update` (verdict HONEST-NEGATIVE-on-PRODUCTION, R17 baseline 0.846 retained, R20·P7 revised referenced, naive smoke-to-smoke +0.154 explicitly tagged as artifact of reduced check scope). The R17 baseline (`aggregate.best_measured_smoke = 0.846`) is preserved.

### §2.4 Test file

`molmetal/tests/test_wf32_phase5_lift_analysis.py` — **NEW, 31 tests, all PASS** in 0.22 s on CPU. Covers:

- Lift analysis JSON well-formedness + top-level keys + 3 metrics
- Lift summary MD exists + references JSON + has TL;DR + honest-negative provenance
- m16 / m19 / m8 R17 baseline preservation (3 tests)
- m16 / m19 / m8 wf32 entry presence (3 tests)
- m16 R18 negative lift (m19 R18 null + DEFER_NO_MEASURED_VALUE + GPU 1 in defer_reason; m8 R18 smoke value + smoke-not-comparable status + naive-lift-is-artifact flag)
- R20 plan items (P5/P6/P7_revised all referenced)
- Aggregate verdict (0 lift above baseline, full honest-negative)
- No fabricated headlines (m19 must be null, m8 smoke-not-comparable, honest_negative_provenance section present)
- Cross-references to Phase 4 ship doc

---

## §3 MEASURED vs PROJECTED ledger

### §3.1 MEASURED (on disk, 2026-09-26)

| Item | Value | Source |
|---|---|---|
| Wf-32 Phase 4 m16 smoke (1-seed × 1-epoch, PlatinAI 4-cell) | **−0.0403** | `metrics/by_metric/affinity_pearson_r.json` headline_pearson_r_mean |
| Wf-32 Phase 4 m16 smoke per-cell | MCF7=−0.013, A549=−0.049, A2780=+0.024, A2780cis=−0.124 | same file, per_cell_line_pearson_r |
| Wf-32 Phase 4 m16 smoke wall | 20.84 s on RTX 5090 | same file, wall_seconds |
| Wf-32 Phase 3 m8 verify smoke (1p×1c×12-check chem-only) | **1.000** | `metrics/by_metric/wf32_pb_valid_verify.json` overall_pb_pass_rate |
| Wf-32 Phase 3 m8 verify smoke wall | 1.11 s on CPU | same file, per_evaluation[0].elapsed_s |
| m16 R17 baseline preserved | 0.207 | `metrics/by_metric/pearson_r_pic50.json` aggregate.best_neural |
| m19 R17 baseline preserved | 0.5346 ± 0.110 | `metrics/by_metric/metal_lomo_auc.json` value/value_std |
| m8 R17 baseline preserved | 0.846 | `metrics/by_metric/pb_pass_rate.json` aggregate.best_measured_smoke |

### §3.2 PROJECTED (R20, NOT executed in Wf-32)

| Item | Projection | Basis |
|---|---|---|
| m16 R18 full retrain | ≥0.50 (TODO-82 §6 Phase D target) | Requires GPU 1 free + tmQM encoder arch mismatch resolved + 5-seed × 30-epoch × PlatinAI 4-cell + HeLa48h dual-cohort |
| m19 R18 full retrain | ≥0.70 (TODO-82 §6 Phase D target) | Requires GPU 1 free + 5-fold × 5-seed × 50-epoch × {hilbert, tmqm} dual-init; hilbert wires HilbertASTSRComposite reward channel |
| m8 R18 production sweep | ≥0.94 across 10 pockets | Requires F2(a) MetalLigandExchange SMARTS rule SHIP first (to break n_distinct=1 search-bound collapse that has blocked every ≥3-pocket production eval since Wf-13) |

### §3.3 DEFER (no data)

| Item | Reason |
|---|---|
| m19 R18 5-fold × 5-seed × 50-epoch | GPU 1 occupancy blocked Wf-32 Phase 4 sweep (Wf-31 m1 + Wf-29 smoke contention). Only structural verification (verify_train_multimetal_lomo.py 8/8 PASS) executed. |

---

## §4 Per-metric honest framing

### §4.1 m16 Affinity Pearson r

**R17 baseline**: 0.207 (HeLa48h/dark 3-seed × 50-epoch, margin=0.5). The R17 baseline is **highly seed-fragile** (per-seed: 0.005 / 0.519 / 0.071).

**R18 MEASURED**: −0.040 (Wf-32 1-seed × 1-epoch smoke on PlatinAI 4-cell, init=tmqm). Per-cell per-seed: MCF7=−0.013, A549=−0.049, A2780=+0.024, A2780cis=−0.124. Wall=20.84 s on RTX 5090.

**Comparability caveats**:
- Smoke vs full: Wf-32 is 1-seed × 1-epoch; R17 is 3-seed × 50-epoch
- Cohort mismatch: Wf-32 = PlatinAI 4-cell; R17 = HeLa48h/dark
- Encoder init: Wf-32 inherits tmQM arch mismatch → random init fallback

**Lift**: −0.247 (R18 − R17 = −0.040 − 0.207)

**Verdict**: HONEST-NEGATIVE. Even accounting for smoke-vs-full and cohort mismatch, a negative Pearson r at smoke scale is a clear honest-negative signal. R17 baseline (0.207) remains canonical. Full retrain deferred to **R20·P5**.

### §4.2 m19 Multi-metal LOMO

**R17 baseline**: 0.5346 ± 0.110 (5-fold × 3-seed × 30-epoch multitask DMPNN on metal_lomo.csv). Per-fold: Ru=0.571, Ir=0.618, Os=0.611, Rh=0.368 (worst), Re=0.506.

**R18 MEASURED**: DEFER — no MEASURED value. Wf-32 Phase 4 sweep did not launch because GPU 1 was occupied. Only structural verification (verify_train_multimetal_lomo.py 8/8 PASS on CLI + main() smoke) executed.

**Comparability caveats**: N/A — no R18 value to compare.

**Verdict**: DEFERRED-NO-DATA. R17 baseline (0.5346 ± 0.110) remains canonical. Cannot claim lift or regression without data. Full 5-fold × 5-seed × 50-epoch × {hilbert, tmqm} dual-init GPU retrain deferred to **R20·P6**.

The hilbert init wires HilbertASTSRComposite reward channel per Wf-9 P1-4 (Sun & Zheng 2026 数学学报 53(4)) — this would have been the first time it was tested in a multi-metal LOMO context. That exercise is now R20·P6 work.

### §4.3 m8 PB-Valid

**R17 baseline**: 0.846 (1-pocket click_tile smoke with MMFF94s relax, all 26 checks; 22/26 PASS, 4 fail on protein-aware distance/cofactor). PRODUCTION_BLOCKED_AT_ALL_SCALES: 60 cells (10 × 3 + 30 × 3 search-bound on test_000..test_019) ALL search-bound on n_distinct=1 collapse.

**R18 MEASURED**: 1.000 (Wf-32 verify smoke: 1 pocket 830c × 1 candidate CCO × pb_mode=mol × chemistry-only 12 checks, 12/12 PASS). Wall=1.11 s on CPU.

**Comparability caveats**:
- Different check scope: Wf-32 = 12 chemistry-only checks; R17 = 26 full checks (chemistry + protein-aware). The 4 R17 failures were protein-aware; Wf-32 smoke did not exercise those.
- Different cohort: Wf-32 = 1 pocket × 1 candidate; R17 = 1 pocket × 26 candidates. 1-cell smoke cannot demonstrate a 10-pocket × 5-candidate production sweep.
- Different candidate: Wf-32 = CCO (trivial); R17 = click_tile (representative metal-drug fragment).
- Naive smoke-to-smoke +0.154 is an **artifact** of the reduced check scope, not a real lift.

**Verdict**: HONEST-NEGATIVE-on-PRODUCTION. R17 baseline (0.846) remains canonical for full-check PB pass rate. Wf-32 smoke (1.000 on 12 chemistry checks, 1 cell) is a positive structural signal that the PB adapter works end-to-end, but does NOT lift over R17.

The full 10-pocket × 5-candidate × 26-check × {mol, dock} dual-mode production sweep is deferred to **R20·P7 (revised)** — pre-condition: F2(a) MetalLigandExchange SMARTS rule ship to break the n_distinct=1 search-bound collapse that has blocked every ≥3-pocket production eval since Wf-13.

---

## §5 R20 plan items (new + revised)

| Plan item | Description | Pre-conditions | Wall estimate |
|---|---|---|---|
| **R20·P5** | m16 Affinity Pearson r GPU retrain (5-seed × 30-epoch × PlatinAI 4-cell + HeLa48h dual-cohort) | GPU 1 free; tmQM encoder arch mismatch resolved (else random-init caps lift) | 6-12 h GPU |
| **R20·P6** | m19 Multi-metal LOMO GPU retrain (5-fold × 5-seed × 50-epoch × {hilbert, tmqm} dual-init) | GPU 1 free; HilbertASTSRComposite reward channel verified end-to-end on a free card first | 12-24 h GPU |
| **R20·P7 (revised)** | m8 PB-Valid production sweep (10 pockets × 5 candidates × 26 checks × {mol, dock} modes) | F2(a) MetalLigandExchange SMARTS rule SHIP (to break n_distinct=1 collapse); CPU-only | 1-3 h CPU |

---

## §6 Unmodified commitments (per party-twenty-spirit §3.5)

| Commitment | Status |
|---|---|
| No R17 baseline overwritten | ✅ All 3 R17 baselines preserved as canonical values |
| No fabricated MEASURED headline | ✅ m16 = smoke value AS-IS, m19 = null, m8 = smoke value with NOT_COMPARABLE caveat |
| No claim of lift without measurement | ✅ All 3 metrics honestly reported as not-lifted |
| New tests added | ✅ 31 new tests (test_wf32_phase5_lift_analysis.py) |
| New reports added | ✅ 2 new files (lift_analysis.json + lift_summary.md) |
| Metrics JSONs updated append-only | ✅ No key collisions, no overwrite of R17 baseline fields |
| R20 plan items created | ✅ 3 plan items (R20·P5 / R20·P6 / R20·P7 revised) |

---

## §7 Test totals (post-Wf-32 Phase 5)

| Test file | Pass | Skip | Fail | Total |
|---|---:|---:|---:|---:|
| `test_wf32_phase1_affinity.py` | 5 | 1 | 0 | 6 |
| `test_wf32_phase2_lomo.py` | 0 | 6 | 0 | 6 |
| `test_wf32_phase3_pb_valid.py` | 0 | 6 | 0 | 6 |
| **`test_wf32_phase5_lift_analysis.py` (NEW)** | **31** | **0** | **0** | **31** |
| **Wf-32 total** | **36** | **13** | **0** | **49** |

Total wall: **0.38 s** on CPU.

No regressions: the existing 13 SKIP tests (waiting for full Phase 4 metrics.json) still skip cleanly; the 5 PASS tests still pass.

---

## §8 Ship ledger

| Path | LOC / size | Status |
|---|---|---|
| `molmetal/reports/wf32_m16_m19_m8_lift_analysis.json` (NEW) | ~7 KB JSON | SHIP |
| `molmetal/reports/wf32_m16_m19_m8_lift_summary.md` (NEW) | ~330 lines MD | SHIP |
| `molmetal/tests/test_wf32_phase5_lift_analysis.py` (NEW) | ~270 LOC | SHIP |
| `metrics/by_metric/pearson_r_pic50.json` (M — appended wf32_phase4_smoke measurement + wf32_phase5_lift_update top-level) | +30 lines | SHIP |
| `metrics/by_metric/metal_lomo_auc.json` (M — appended wf32_phase5_lift_update top-level) | +15 lines | SHIP |
| `metrics/by_metric/pb_pass_rate.json` (M — appended wf32_phase5_lift_update top-level) | +20 lines | SHIP |

**Total new code**: ~270 LOC (test) + 2 new reports + 3 modified metrics JSONs (append-only).

---

## §9 Cross-reference

- `TODO/pending/82_metric_gap_filling_plan.md` §6 Phase D — ship gate (m16 ≥0.50, m19 ≥0.70, m8 ≥0.94)
- `TODO/completed/87_wf32_phase4_honest_negative.md` — Phase 4 ship report (predecessor)
- `TODO/completed/86_wf31_phase4_dual_line.md` — Phase 4 Wf-31 sibling (caused the GPU 1 contention that deferred Wf-32 Phase 4)
- `TODO/completed/82_metric_gap_filling.md` — TODO-82 synthesis (R20·P5 / R20·P6 / R20·P7 to be appended)
- `TODO/completed/82b_vina_dock_retrain.md` — Wf-30 sibling lift analysis template
- `TODO/INDEX.md` §13.6 — final results table (to be updated with this Phase 5 close-out)
- `molmetal/reports/wf30_dock/m3_lift_analysis.json` — sibling Wf-30 lift analysis JSON template
- `molmetal/reports/wf30_dock/m3_lift_summary.md` — sibling Wf-30 lift summary template
- `metrics/by_metric/affinity_pearson_r.json` — Wf-32 m16 smoke metrics (Phase 4 ship)
- `metrics/by_metric/wf32_pb_valid_verify.json` — Wf-32 m8 verify smoke (Phase 3 ship)
- `molmetal/scripts/verify_train_affinity_regression.py` (4/4 PASS)
- `molmetal/scripts/verify_train_multimetal_lomo.py` (8/8 PASS)
- `molmetal/scripts/verify_bench_pb_valid.py` (PASSED)

## §10 Final verdict

**Wf-32 Phase 5 SHIP — full honest-negative across m16/m19/m8 vs R17 baseline.** R17 baseline numbers remain canonical for paper §4 Table 2. The R18 modules (train_affinity_regression.py / train_multimetal_lomo.py / bench_pb_valid.py + 3 verify scripts + 18 unit tests + NEW skill mol-metal-gpu-execution) all ship. The lift question is unanswered and the deferred 5-seed × 30/50-epoch × GPU 1 retrain becomes 3 R20 plan items (R20·P5 / R20·P6 / R20·P7 revised). No fabricated headline values exist anywhere in this Phase 5 deliverable.

**Date**: 2026-09-26