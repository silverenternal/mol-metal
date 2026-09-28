# Wf-32 Phase 5 — m16/m19/m8 MEASURED vs R17 baseline lift summary

**Generated**: 2026-09-26
**Workflow**: Wf-32 (Phase 5 — lift analysis close-out)
**Status**: ✅ SHIP — all 3 metrics honestly reported; **0/3 lift vs R17 baseline**; all 3 become R20 plan items
**Companion JSON**: `molmetal/reports/wf32_m16_m19_m8_lift_analysis.json`

---

## TL;DR — honest-negative across the board

| Metric | R17 baseline | R18 PROJECTED | R18 MEASURED | Lift | Verdict | R20 plan |
|---|---:|---:|---:|---:|:---:|:---:|
| **m16** Affinity Pearson r (PlatinAI 4-cell mean) | **0.207** (HeLa48h/dark 3-seed × 50-epoch, margin=0.5) | ≥0.50 | **−0.040** (Wf-32 1-seed × 1-epoch smoke on PlatinAI 4-cell, init=tmqm) | **−0.247** | ❌ HONEST-NEGATIVE | R20·P5 |
| **m19** Multi-metal LOMO mean AUC (5 folds × N seeds) | **0.5346 ± 0.110** (5-fold × 3-seed × 30-epoch multitask DMPNN) | ≥0.70 | **DEFER** (no MEASURED value — GPU 1 occupancy blocked Wf-32 Phase 4 sweep) | n/a | ⏸️ DEFERRED-NO-DATA | R20·P6 |
| **m8** PB-Valid pass rate (10 pockets × 5 candidates, mol + dock modes) | **0.846** (R17 1-pocket × 26-candidate × 26-check smoke; production BLOCKED) | ≥0.94 across 10 pockets | **1.000** (Wf-32 1-pocket × 1-candidate × 12-check chemistry-only smoke) | **+0.154 (smoke-to-smoke, NOT comparable)** | ❌ HONEST-NEGATIVE-on-PRODUCTION | R20·P7 (revised) |

**Aggregate**: 0/3 metrics lifted over R17 baseline at this workflow's measurement scale. R17 baseline numbers remain the canonical MEASURED values for paper §4 Table 2. The R18 modules ship (train_affinity_regression.py, train_multimetal_lomo.py, bench_pb_valid.py + verify scripts) but require the deferred 5-seed × 30/50-epoch GPU 1 retrain to produce a real lift.

---

## §1 — m16 Affinity Pearson r

### R17 baseline (canonical)
- **0.207** (HeLa48h/dark cohort, 3-seed × 50-epoch, margin=0.5)
- Seed-std: 0.005 / 0.519 / 0.071 (highly seed-fragile)
- Source: `metrics/by_metric/pearson_r_pic50.json` aggregate.best_neural
- Date: 2026-09-14

### R18 MEASURED (Wf-32 Phase 4 smoke)
- **−0.040** (Wf-32 1-seed × 1-epoch smoke on PlatinAI 4-cell, init=tmqm, hidden=300, depth=3, batch=128)
- Per-cell per-seed: MCF7=−0.013, A549=−0.049, A2780=+0.024, A2780cis=−0.124
- Wall: 20.84 s on RTX 5090
- Source: `metrics/by_metric/affinity_pearson_r.json` headline_pearson_r_mean
- Date: 2026-09-26

### Comparability caveat
- **Smoke-vs-full**: Wf-32 is 1-seed × 1-epoch; R17 is 3-seed × 50-epoch. The smoke is structurally unable to reach the R17 number even by chance.
- **Cohort mismatch**: Wf-32 used PlatinAI 4-cell cohort; R17 used HeLa48h/dark cohort. These are different test sets.
- **Encoder init**: Wf-32 inherits the tmQM encoder arch mismatch fallback to random init (per Wf-29 §2 finding).

### Lift
- **−0.247** (R18 − R17 = −0.040 − 0.207)
- Even accounting for the smoke-vs-full gap and the cohort mismatch, a negative Pearson r at smoke scale is a clear honest-negative signal.

### Verdict
**HONEST-NEGATIVE**. R17 baseline (0.207) remains the canonical number. The Wf-32 trainer produces Pearson r < 0 at smoke scale; the full 5-seed × 30-epoch × PlatinAI 4-cell + HeLa48h dual-cohort retrain is deferred to **R20·P5**.

---

## §2 — m19 Multi-metal LOMO

### R17 baseline (canonical)
- **0.5346 ± 0.110** (5-fold × 3-seed × 30-epoch multitask DMPNN on metal_lomo.csv)
- Per-fold: Ru=0.571, Ir=0.618, Os=0.611, **Rh=0.368** (worst fold, leaves-Rh-out), Re=0.506
- n_runs = 15 (5 folds × 3 seeds)
- Source: `metrics/by_metric/metal_lomo_auc.json` value/value_std/per_fold_auc_mean
- Date: 2026-09-25

### R18 MEASURED (Wf-32 Phase 4)
- **DEFER — no MEASURED value**
- The Wf-32 m19 trainer (`molmetal/scripts/train_multimetal_lomo.py`) and verify script (`verify_train_multimetal_lomo.py` 8/8 PASS on CLI + main() smoke) ship, but no actual training run completed.
- Reason: GPU 1 (RTX 5090, 32 GB) was occupied at Wf-32 Phase 4 launch by Wf-31 m1 Ru-AUC SE(3) SB run (PID 3853986) and immediately before by Wf-29 train_se3sb.py smoke (PID 3802039). The 5 folds × 5 seeds × 50 epochs × 2 init modes (hilbert, tmqm) = 250 epoch-runs would have collided on the same card.

### Comparability caveat
- **N/A — no R18 value to compare**. Cannot claim lift or regression without data.

### Verdict
**DEFERRED-NO-DATA**. R17 baseline (0.5346 ± 0.110) remains the canonical number. The full 5-fold × 5-seed × 50-epoch × {hilbert, tmqm} dual-init GPU retrain is deferred to **R20·P6**.

The hilbert init wires HilbertASTSRComposite reward channel per Wf-9 P1-4 (Sun & Zheng 2026 数学学报 53(4)) and is the new mechanism that has never been tested in a multi-metal LOMO context. The Wf-32 Phase 4 sweep is the first time it would have been exercised; that exercise is now R20·P6 work.

---

## §3 — m8 PB-Valid pass rate

### R17 baseline (canonical)
- **0.846** (1-pocket click_tile smoke with MMFF94s relax, all 26 checks, 22/26 PASS, 4 fail on protein-aware distance/cofactor)
- **PRODUCTION_BLOCKED_AT_ALL_SCALES**: 60 cells (10 × 3 + 30 × 3 search-bound on test_000..test_019) ALL search-bound on n_distinct=1 collapse. Production 100×3 sweep killed before completion.
- Lit target: **0.94** (TargetDiff, Guan 2023 ICLR)
- Source: `metrics/by_metric/pb_pass_rate.json` aggregate.best_measured_smoke
- Date: 2026-09-15

### R18 MEASURED (Wf-32 Phase 3 verify smoke)
- **1.000** (Wf-32 verify smoke: 1 pocket (830c) × 1 candidate (CCO) × pb_mode=mol × chemistry-only 12 checks, 12/12 PASS)
- Wall: 1.11 s on CPU
- Source: `metrics/by_metric/wf32_pb_valid_verify.json` overall_pb_pass_rate
- Date: 2026-09-26

### Comparability caveat
- **Different check scope**: Wf-32 smoke = 12 chemistry-only checks; R17 = 26 full checks (chemistry + protein-aware). The 4 R17 failures were protein-aware distance/cofactor checks; Wf-32 smoke did not exercise those at all.
- **Different cohort**: Wf-32 = 1 pocket × 1 candidate; R17 = 1 pocket × 26 candidates. 1-cell smoke cannot demonstrate a 10-pocket × 5-candidate production sweep.
- **Different candidate**: Wf-32 = CCO (trivial); R17 = click_tile (representative metal-drug fragment).
- **Naive smoke-to-smoke lift (+0.154) is an artifact of the reduced check scope**, not a real lift.

### Verdict
**HONEST-NEGATIVE-on-PRODUCTION**. R17 baseline (0.846) remains the canonical number for full-check PB pass rate. Wf-32 smoke (1.000 on 12 chemistry checks) is a positive structural signal that the PB adapter works end-to-end, but does not lift over R17 because the full-check + production-cohort sweep was not executed.

The 10-pocket × 5-candidate × 26-check × {mol, dock} dual-mode production sweep is deferred to **R20·P7 (revised)** — pre-condition: F2(a) MetalLigandExchange SMARTS rule ship to break the n_distinct=1 search-bound collapse that has blocked every ≥3-pocket production eval since Wf-13.

---

## §4 — Aggregate verdict

```
n_metrics_compared      = 3
n_metrics_with_measured = 2   (m16 + m8 smoke; m19 = no data)
n_metrics_passed_r17    = 0
n_metrics_passed_target = 0
n_honest_negative       = 3   (m16 NEG, m19 DEFER-NO-DATA, m8 NEG-on-PROD)
n_deferred_no_data      = 1   (m19)
n_lift_above_baseline   = 0
```

**Full honest-negative**: All 3 Wf-32 metrics are honest-negative vs the R17 baseline at this workflow's measurement scale. R17 baseline numbers remain the canonical MEASURED values for paper §4 Table 2. The R18 modules (train_affinity_regression.py / train_multimetal_lomo.py / bench_pb_valid.py + 3 verify scripts + 18 unit tests + NEW skill mol-metal-gpu-execution) all ship — the lift question is unanswered and stays deferred.

---

## §5 — R20 plan items (new + revised)

| Plan item | Description | Pre-conditions | Wall estimate |
|---|---|---|---|
| **R20·P5** | m16 Affinity Pearson r GPU retrain (5-seed × 30-epoch × PlatinAI 4-cell + HeLa48h dual-cohort) | GPU 1 free; tmQM encoder arch mismatch resolved (else random-init caps lift) | 6-12 h GPU |
| **R20·P6** | m19 Multi-metal LOMO GPU retrain (5-fold × 5-seed × 50-epoch × {hilbert, tmqm} dual-init) | GPU 1 free; HilbertASTSRComposite reward channel verified end-to-end on a free card first | 12-24 h GPU |
| **R20·P7 (revised)** | m8 PB-Valid production sweep (10 pockets × 5 candidates × 26 checks × {mol, dock} modes) | F2(a) MetalLigandExchange SMARTS rule SHIP (to break n_distinct=1 collapse); CPU-only | 1-3 h CPU |

---

## §6 — Honest framing (per party-twenty-spirit §3.5)

> **MEASURED 数字 > PROJECTED 数字. wf failure ≠ 真活 failure, 真实磁盘文件 + 测试通过 才是 MEASURED.**

What this Phase 5 deliverable does NOT do:
- Does NOT claim m16 lift over R17.
- Does NOT claim m19 lift over R17 (no MEASURED value exists).
- Does NOT claim m8 lift over R17 (smoke-to-smoke +0.154 is an artifact of reduced check scope).
- Does NOT overwrite the R17 baselines in `metrics/by_metric/{pearson_r_pic50,metal_lomo_auc,pb_pass_rate}.json`.
- Does NOT fabricate any headline value.

What this Phase 5 deliverable DOES:
- Honestly reports the Wf-32 smoke numbers AS-IS with full comparability caveats.
- Preserves R17 baselines as canonical MEASURED values.
- Creates new Wf-32 Phase 5 entries appended to the `measurements` lists of the 3 metrics JSONs (no overwrite).
- Creates the lift analysis JSON + this summary MD on disk.
- Generates R20 plan items for the 3 Wf-32 metrics (R20·P5, R20·P6, R20·P7 revised).
- Updates TODO/INDEX.md §13.6 with the Phase 5 close-out verdict (lift = 0).

---

## §7 — Cross-reference

- `molmetal/reports/wf32_m16_summary.json` — Phase 4 m16 honest-negative summary
- `molmetal/reports/wf32_m19_summary.json` — Phase 4 m19 honest-negative summary
- `molmetal/reports/wf32_m8_summary.json` — Phase 4 m8 partial summary
- `molmetal/reports/wf32_m16_m19_m8_lift_analysis.json` — Phase 5 JSON companion
- `molmetal/reports/wf30_dock/m3_lift_analysis.json` — sibling Wf-30 lift analysis (template)
- `molmetal/reports/wf30_dock/m3_lift_summary.md` — sibling Wf-30 lift summary (template)
- `metrics/by_metric/affinity_pearson_r.json` — Wf-32 smoke metrics
- `metrics/by_metric/wf32_pb_valid_verify.json` — Wf-32 m8 verify smoke
- `TODO/pending/82_metric_gap_filling_plan.md` §6 Phase D ship gate (m16 ≥0.50, m19 ≥0.70, m8 ≥0.94)
- `TODO/completed/87_wf32_phase4_honest_negative.md` — Phase 4 ship report
- `TODO/completed/86_wf31_phase4_dual_line.md` — Phase 4 Wf-31 sibling (caused the GPU 1 contention)
- `TODO/INDEX.md` §13.6 — final results table (to be updated with this Phase 5 entry)

## §8 — Final verdict

**Wf-32 Phase 5 SHIP — full honest-negative**. All 3 metrics (m16 / m19 / m8) are honestly reported as not-lifted vs R17 baseline. R17 baselines remain canonical. The R20 plan absorbs the deferred work. The R18 module + verify + test scaffolding for all 3 metrics is on disk and ready for the deferred GPU cycle. No fabricated headline values exist anywhere in this Phase 5 deliverable.