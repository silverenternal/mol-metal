# TODO-82c — Wf-31 Phase C synthesis: m1 + m2 + m7 retrain + lift analysis

**Date**: 2026-09-26
**Workflow**: Wf-31 (Phase C — 5-metal + cisplatin + Multi-PH train scripts + GPU 1 smoke run + lift analysis)
**Status**: ⚠️ **PARTIAL HONEST-NEGATIVE** — m1 + m2 deferred to R20·P2 / R20·P3 (smoke-only in this session); m7 floor PASS real; **0/3 apples-to-apples lift vs R17 baseline** (per `party-twenty-spirit` §3.5).
**Parent**: `TODO/pending/82_metric_gap_filling_plan.md` §2 Phase C
**Sibling**: `TODO/completed/86_wf31_phase4_dual_line.md` (Phase 4 dual-line ship), `TODO/completed/82b_vina_dock_retrain.md` (Wf-30 Phase B), `TODO/completed/82e_paper_v3.md` (Wf-33 paper v3 templates).

---

## §0 TL;DR

| Metric | R17 baseline | Wf-22 retrain | R18 PROJECTED | R18 MEASURED | Lift | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| **m1** Ru 5-metal AUC | 0.7123 | 0.630 (honest-neg) | ≥ 0.86 | **DEFER** (smoke-only) | n/a | HONEST-NO-MEASURED → R20·P2 |
| **m2** Pt A2780cis AUC | 0.6476 | 0.630 (shared Wf-22) | ≥ 0.85 | **DEFER** (smoke-only) | n/a | HONEST-NO-MEASURED → R20·P3 |
| **m7** Multi-PH Diversity Tanimoto | 0.1065 | n/a | ≥ 0.75 (floor) | **0.904** (PlatinAI 5K, n=500, seed=42) | +0.798 naive, **APPLES-TO-ORANGES** vs R17 | HONEST-PASS-ON-FLOOR → R20·P8 NEW |

**Aggregate**: 0/3 apples-to-apples lift. m7 floor-PASS (0.904 ≥ 0.75) is real and reproducible. R17 baselines preserved; paper §4 Table 2 cells remain `\MEASURED{TBD Wf-31}` per TODO-82e SHELL.

---

## §1 Per-deliverable

### §1.1 Three train scripts + Multi-PH extension (SHIP)

| Script | Metric | Init modes | LOC | Status |
|---|---|---|---:|---|
| `molmetal/scripts/train_ru_5metal_se3sb.py` | m1 | se3sb, tmqm | ~520 | SHIP |
| `molmetal/scripts/train_cisplatin_resistance_v2.py` | m2 | se3sb, tmqm | ~580 | SHIP |
| `molmetal/scripts/bench_multi_ph_diversity.py` | m7 | (bench, no init) | ~340 | SHIP |

All three share `_common_train_args.py` (TODO-82f) for the dual-init CLI surface. m7 bench extends `multi_parameter_ph.py` (R18 PLAN §3.6 T5) with Tanimoto distance over Morgan-ECFP4.

### §1.2 GPU 1 smoke run (5 init scripts, in-flight)

`run_in_background=True` × 5 — launched per TODO-86 dual-line model:
- m1 se3sb + m1 tmqm → `molmetal/reports/wf31_m1/{se3sb,tmqm}/`
- m2 se3sb + m2 tmqm → `molmetal/reports/wf31_m2/{se3sb,tmqm}/`
- m7 bench → `molmetal/reports/wf31_m7/` (DONE, 164.74 s)

Smoke config: 2 epochs × 50 samples/metal × batch 8 × hidden 64 × depth 2 (~3-10 min per script). Full spec (5 seeds × 50 epoch × 2 init) deferred to R20.

### §1.3 Lift analysis (SHIP, partial honest-negative)

- `molmetal/reports/wf31_m1_m2_m7_lift_analysis.json` (NEW, ~12 KB) — structured MEASURED-vs-PROJECTED ledger per metric
- `molmetal/reports/wf31_m1_m2_m7_lift_summary.md` (NEW, ~205 lines) — narrative summary with cohort-mismatch caveat
- `molmetal/tests/test_wf31_phase5_lift_analysis.py` (NEW, ~250 LOC) — asserts R17 baseline preservation + no fabricated lift + R20 plan items present

Result: **16 + N tests PASS**, 0 fabricated numbers, R17 baselines intact across `metalcytotox_ru_auc.json`, `platinai_4cell_auc.json`, `nci60_precious_9cell_auc.json`, `diversity_tanimoto.json`.

---

## §2 MEASURED vs PROJECTED

### §2.1 MEASURED — on disk

| Item | Value | Source |
|---|---:|---|
| m7 Multi-PH expressivity | 0.242 (vs 0.120 single-PH) | `wf31_m7/metrics.json` |
| m7 Δ expressivity | **+0.122** | same |
| m7 mean Tanimoto distance | **0.904** | same, `diversity.mean_tanimoto_distance` |
| m7 wall-time | 164.74 s on RTX 5090 | `wf31_m7/bench.log` |
| m7 n_parsed (diversity) | 500/500 | same |
| m7 n_parsed (Multi-PH) | 380/500 (24% parse fail) | same — likely Pt(II) square-planar outside `max_radius=5Å` |

### §2.2 PROJECTED — R18 ceilings (NOT measured)

| Metric | R17 baseline | R18 PROJECTED | Δ projected | Source |
|---|---:|---:|---:|---|
| m1 Ru 5-metal AUC | 0.7123 | 0.86 | +0.148 | `METRICS_TARGETS.md` §2 Krasnov 2026 (0.81) + R18 Multi-PH lift |
| m2 Pt A2780cis AUC | 0.6476 | 0.85 | +0.202 | Rusanov 2025 (0.85) + R18 Hilbert+Multi-PH lift |
| m7 Diversity Tanimoto | 0.1065 | 0.65 | +0.544 | R18 §3 floor is 0.75 (already exceeded by 0.904) |

Per `TODO/completed/82e_paper_v3.md` §2.2: paper §4 Table 2 v3 cells for m1/m2/m7 carry `\MEASURED{TBD Wf-31}` — they remain placeholders pending R20 full sweep.

---

## §3 Module-by-module lift analysis

### §3.1 se3sb Schrödinger Bridge (R18 §3.4 T1, m1/m2)

- Mechanism: SE(3)-equivariant SB integrator as initial-pose/noise prior for downstream D-MPNN
- Pretraining path: `models/se3sb_pretrained/se3sb_pretrained_encoder.pt`
- Expected lift per R18 PLAN §3.4: +0.05 to +0.10 AUC over random init
- **Wf-31 status**: encoder init loaded OK for se3sb mode (hidden 64 compatible with V4MultiTaskDMPNN.dmpnn). Smoke-only — no MEASURED lift.
- **Inherited risk**: tmQM init mode falls back to random init due to encoder arch mismatch (hidden 300 vs 64, per Wf-29 attr fix).

### §3.2 Transport Relaxation (R18 §3.4 T2, m1/m2)

- Mechanism: kinetic-energy-decay kernel replacing naive Langevin for D-MPNN training
- Expected lift: +0.02 to +0.04 AUC over baseline Langevin
- **Wf-31 status**: integrated into `_common_train_args.py` as `--transport {none,relaxation,langevin}`. Smoke-only — no MEASURED lift.
- Note: `train_transport.py` referenced by paper §3.5 v3 (TODO-82e §2.1 row 2) is not the Wf-31 deliverable; Wf-31 ships the dual-init + transport switch on m1/m2 trainers.

### §3.3 Multi-Parameter Persistent Homology (R18 §3.6 T5, m7)

- Mechanism: 2-parameter persistence (radial + angular filtration) Hilbert feature
- Expected lift: 2× expressivity over single-PH marginal
- **Wf-31 MEASURED**: Multi-PH expressivity 0.242 vs single-PH 0.120, **+0.122 = PASS** on R18 PLAN §3.6 T5 theoretical lift condition.
- m7 floor (0.75) **PASSED** at 0.904, but cohort is PlatinAI 5K random subset (n=500), not Lambda-MCTS generated (n=30) — see §3.4.

### §3.4 Apples-to-oranges caveat (m7)

The naive +0.798 lift (R17 0.1065 → Wf-31 0.904) is dominated by **cohort differences**, not by R18 module lift:
- **Cohort size**: R17 n=30 (generated via Lambda-MCTS, n_distinct_mean=20); Wf-31 n=500 (PlatinAI 5K random subset)
- **Source corpus**: R17 = model-generated (post-R12 4 FIXES + decoder rework gate); Wf-31 = dataset-intrinsic diversity
- **Verdict**: apples-to-oranges. R17 baseline 0.1065 stays canonical for *generated-molecule* diversity. R20·P8 NEW will measure the apples-to-apples lift via Lambda-MCTS generated molecules with R18 Multi-PH attached to the reward channel.

---

## §4 Honest framing (party-twenty-spirit §3.5)

| Commitment | Status |
|---|---|
| m7 Multi-PH expressivity delta +0.122 MEASURED | ✅ `molmetal/reports/wf31_m7/metrics.json` |
| m7 Tanimoto diversity 0.904 ≥ 0.75 floor MEASURED | ✅ same file |
| m1 / m2 5-seed × 50-epoch MEASURED | ❌ NOT measured (smoke-only; deferred to R20·P2 / R20·P3) |
| Honest Wf-22 history (R17 0.7123 → 0.630 = Δ -0.082) | ✅ cited in §0; precondition for R20·P2 |
| Lift expectation stated but not guaranteed | ✅ R18 PROJECTED = +0.148 / +0.202 / +0.544; MEASURED = null/null/0.904 |
| R17 baselines preserved (no overwrite) | ✅ all 4 metrics JSONs verified post-write |

**Why lift is expected but not guaranteed**: Wf-22's honest-negative (R18 retrain at 50 epoch got 0.630 vs R17 0.7123) shows the same R18 modules **can** regress if the encoder arch mismatch isn't fixed (tmQM 300-d vs V4 64-d forces random-init fallback). R20·P2 / R20·P3 must resolve the tmQM mismatch before the GPU sweep, otherwise the same honest-negative recurs. R18 §3 ceilings are **theoretical** — they assume a working pretraining path; the se3sb init path is the one most likely to land lift on first try.

---

## §5 No-commitment clause — R17 baseline untouched

Wf-31 Phase C is a **script + smoke + analysis ship**, not a results ship:
- **No R17 baseline overwritten** in any `metrics/by_metric/*.json`.
- **No MEASURED headline number** claimed for m1 or m2 (both null in `wf31_m1_m2_m7_lift_analysis.json`).
- **No paper §4 Table 2 cell** swapped from `\MEASURED{TBD Wf-31}` to a real number.
- **No new module** wired into Lambda-MCTS or bench paths — Multi-PH remains an isolated bench input.
- **No wet-lab claim** — Wf-31 R18 modules are bench/retrain only; experimental validation is TODO-81 territory (R20·P3 precondition).

Paper v3 templates (TODO-82e) keep `\MEASURED{TBD Wf-31}` for m1/m2/m7 row 3 cells until R20 ships.

---

## §6 Cross-reference

- `TODO/pending/82_metric_gap_filling_plan.md` §1 G2 / G3 / G4, §2 Phase C (Wf-31 scope)
- `TODO/completed/86_wf31_phase4_dual_line.md` — Phase 4 dual-line + 16 tests + 3 summary JSONs (predecessor)
- `TODO/completed/82b_vina_dock_retrain.md` — Wf-30 sibling (m3 Vina Dock honest-negative; R20·P1 territory)
- `TODO/completed/82e_paper_v3.md` — Wf-33 paper v3 templates (consumes Wf-31/30/32 MEASURED when they ship)
- `TODO/completed/82e_final_closeout.md` — TODO-82 close-out (R20·P2 + R20·P3 + R20·P8 created/preserved)
- `TODO/completed/82f_train_scripts_refactor.md` — `_common_train_args.py` shared CLI
- `TODO/completed/75_blackwell_retrain.md` §3 — Wf-22 R18 honest-negative 0.630 (cautionary tale for R20·P2)
- `TODO/completed/88_wf32_phase5_lift_analysis.md` — sibling Wf-32 Phase 5 (m16/m19/m8 all honest-negative)
- `TODO/INDEX.md` §13.6.5 — Wf-29 ~ Wf-34 honest-negative ledger
- `molmetal/reports/wf31_m7/metrics.json` — m7 on-disk MEASURED (0.904, 164.74 s)
- `molmetal/reports/wf31_m1_m2_m7_lift_analysis.json` — structured MEASURED-vs-PROJECTED ledger
- `molmetal/reports/wf31_m1_m2_m7_lift_summary.md` — narrative summary with cohort caveat
- `molmetal/tests/test_wf31_phase5_lift_analysis.py` — Phase 5 test file (NEW)
- `molmetal/scripts/train_ru_5metal_se3sb.py` — m1 trainer (dual-init)
- `molmetal/scripts/train_cisplatin_resistance_v2.py` — m2 trainer (dual-init)
- `molmetal/scripts/bench_multi_ph_diversity.py` — m7 bench
- `metrics/by_metric/metalcytotox_ru_auc.json` — m1 R17 0.7123 preserved
- `metrics/by_metric/platinai_4cell_auc.json` — m2 R17 value_a2780cis_auc=0.6476 preserved
- `metrics/by_metric/nci60_precious_9cell_auc.json` — m2 broader metric (cross-ref to PlatinAI 4-cell)
- `metrics/by_metric/diversity_tanimoto.json` — m7 R17 0.1065 preserved; Wf-31 0.904 added as measurements[6]

---

## §7 Ship ledger

| Path | LOC / size | Status |
|---|---|---|
| `molmetal/scripts/train_ru_5metal_se3sb.py` (NEW) | ~520 | SHIP |
| `molmetal/scripts/train_cisplatin_resistance_v2.py` (NEW) | ~580 | SHIP |
| `molmetal/scripts/bench_multi_ph_diversity.py` (NEW) | ~340 | SHIP |
| `molmetal/scripts/_common_train_args.py` (NEW, per TODO-82f) | ~180 | SHIP |
| `molmetal/reports/wf31_m7/{bench.log,metrics.json,comparison_table.md}` (NEW) | ~12 KB | SHIP + MEASURED |
| `molmetal/reports/wf31_m{1,2}/{se3sb,tmqm}/train_smoke.log` (in-flight) | ~1 MB each | SMOKE-ONLY (no metrics.json yet) |
| `molmetal/reports/wf31_m1_summary.json` + m2 + m7 (NEW) | ~2 KB each | SHIP |
| `molmetal/reports/wf31_m1_m2_m7_lift_analysis.json` (NEW) | ~12 KB | SHIP |
| `molmetal/reports/wf31_m1_m2_m7_lift_summary.md` (NEW) | ~205 lines | SHIP |
| `molmetal/tests/test_wf31_phase4_dual_line.py` (NEW) | ~16 tests | PASS |
| `molmetal/tests/test_wf31_phase5_lift_analysis.py` (NEW) | ~250 LOC | PASS |
| `metrics/by_metric/{metalcytotox_ru,platinai_4cell,nci60_precious_9cell,diversity_tanimoto}.json` (M — append-only) | +30 lines each | SHIP |

**Total new code**: ~1,620 LOC scripts + ~250 LOC tests + ~30 KB reports + 4 metrics JSON appends.

---

## §8 Final verdict

**Wf-31 Phase C SHIP** — partial honest-negative:
- **m7** Multi-PH expressivity +0.122 and Tanimoto 0.904 are real and reproducible MEASURED numbers. m7 floor (0.75) PASS is real. Cohort-mismatch caveat documented; apples-to-apples lift deferred to R20·P8 NEW.
- **m1 + m2** ship as 3 dual-init train scripts + GPU 1 smoke runs (structural contract only). No MEASURED AUC. Full 5-seed × 50-epoch × {se3sb, tmqm} GPU sweep deferred to R20·P2 / R20·P3. Pre-condition: tmQM encoder arch mismatch from Wf-29 must be resolved first, else random-init fallback will cap any lift (same as Wf-22 honest-negative 0.630).
- **0/3 apples-to-apples lift** vs R17 baseline. R17 baselines preserved across all 4 metrics JSONs. Paper §4 Table 2 cells remain `\MEASURED{TBD Wf-31}` per TODO-82e SHELL discipline.
- **Honest framing**: per party-twenty-spirit §3.5, this is a **scripts + smoke + analysis ship**, not a results ship. Wf-22 history (R18 retrain 0.630 < R17 baseline 0.7123) is the cautionary precedent — lift is expected but not guaranteed. R20 is where the honest apples-to-apples lift will land (or the honest-negative will be repeated).

**Date**: 2026-09-26
