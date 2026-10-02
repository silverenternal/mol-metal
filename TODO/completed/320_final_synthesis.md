# Wf-320I — Final synthesis (Wf-320 closeout aggregation)

**Date**: 2026-09-29
**Wf**: Wf-320I (aggregation agent, READ-only audit + INDEX.md sync)
**Status**: ✅ SHIPPED (INDEX.md §0 + §22 sync + this synthesis doc)
**Skill alignment**: `party-twenty-spirit` §3.5 (every numeric value traces to a JSON / report path on disk; MEASURED vs PROJECTED tag discipline preserved) + `AGENTS.md` §1.1-§1.5 (5 standing rules).

---

## §0 One-line summary

All 7 Wf-320 sub-flows (B/C/D/E/F/G/H) **SHIPPED** in CPU-only / verification scope.
n_completed = 7, n_partial = 0, n_honest_negative_dashboard_rows = 4 (downstream lift
consumers, NOT workflow outcomes), remaining_blockers = 5 GPU launches + 1 SOTA Phase 2
benchmark + 1 m15 metric JSON refresh.

---

## §1 Per-agent summary

| # | Wf | Surface | Status | MEASURED evidence |
|---:|---|---|:---:|---|
| 1 | **Wf-320B** | Wf-291F harness auto-fallback 真修 (3rd line of defence) | ✅ SHIPPED (CPU-only) | 13/13 tests PASS in 0.53 s; CFM pre-flight 6/6 PASS, wall 0.0006 s |
| 2 | **Wf-320C** | TODO-220 stub driver full impl verification | ✅ SHIPPED (verification, not re-impl) | 10/10 integration tests PASS in 25.17 s; temporal smoke 5/5 ok + NCI-60 smoke 9/9 ok; HCT116 highlight 0.6998; PC3 honest-negative 0.1667 |
| 3 | **Wf-320D** | chemprop v2 featurization fix (BatchMolGraph cache + opt-in fast featurizer) | ✅ SHIPPED (CPU-only engineering gate) | 12/12 tests PASS in 4.68 s; 1-epoch smoke Ru seed=0 test_auc 0.7452; cache hit_rate 0.00 (1-epoch expected) |
| 4 | **Wf-320E** | Wf-123 P3 M1+M2 downstream lift dashboard | ✅ SHIPPED (aggregation-only) | 12/12 dashboard tests PASS in 0.43 s; pre-flight CFM 6/6 PASS; 6 consumers tracked (2 MEASURED + 4 HONEST_NEGATIVE) |
| 5 | **Wf-320F** | R20 Phase 0 closeout ACTUAL (independent re-verification) | ✅ SHIPPED | 37/37 new tests PASS (12 BindingDB + 12 NCI-60 + 13 DrugOOD) in 10.72 s; combined with Wf-313H 42 = 79 total tests; pre-flight `p3_r20_phase0_loaders_complete` PASS |
| 6 | **Wf-320G** | MECHANISM_METRIC_MAPPING §13/§14 plan-level registry (23 mechanisms M22-M44) | ✅ SHIPPED (CPU-only documentation) | 107/107 tests PASS; Wf-313E regression 33/33 PASS; 12 CPU-ONLY-SKELETON + 2 PLAN-ONLY + 2 PROJECTED @ R20 + 5 HONEST-NEGATIVE + 2 MEASURED |
| 7 | **Wf-320H** | TODO-134 Phase 2 Vina attempt (pipeline proof) | ✅ PIPELINE PROOF SHIP | 6/6 pockets green; pre-flight 23/23 PASS; recovered Vina 1.2.7 via CadButEaas `.venv-vlm` Python binding; SOTA redocking benchmark remains FUTURE WORK |

**n_completed = 7**, **n_partial = 0**, **n_failed = 0**.

---

## §2 Per-Wf honest-framing notes

### §2.1 Wf-320B (Wf-291F harness auto-fallback 真修)

- 3rd line of defence against Wf-291C pipeline regression class; bounded `max_retries=3`.
- Auto-fallback **reduces** future pipeline regression frequency but does NOT **eliminate**
  the Wf-291C class (root cause lives in parent schema-extraction layer).
- All 13 contract tests CPU-only, deterministic, < 1 sec.
- Chemprop v2 v3 pre-flight wall timeout is **pre-existing issue** (unrelated to Wf-320B;
  per Wf-320B §6 honest framing).

### §2.2 Wf-320C (TODO-220 stub driver verification)

- Wf-320C is **verification**, NOT re-implementation (per Wf-320C §4).
- `--execute` paths shipped by Wf-308C; smoke JSONs shipped by Wf-310C; acceptance gate
  shipped by Wf-313D. Wf-320C adds 10 integration tests + verification matrix.
- Smoke MEASURED (CPU 1-epoch 1-seed); GPU 50-ep × 3-seed full sweep **PROJECTED PENDING
  USER_OK + 3.9 hr wall**.
- PC3=0.1667 honest-negative preserved (NOT paraphrased via MultiTaskLoss head).
- HCT116=0.6998 highlight preserved (above 0.5 random, cited in test 8).

### §2.3 Wf-320D (chemprop v2 featurization fix)

- Cache hit_rate **MEASURED 0.00 on 1-epoch smoke** (expected — single epoch does not
  exercise cross-epoch reuse).
- Full 50-ep projected cache hit_rate ≈ 0.98 with 19 s CPU savings (~0.1% of total wall).
- **GPU paper-grade verification deferred to R20-P1 (TODO-120)**; explicit fresh user OK
  required per `mol-metal-gpu-execution` §0 standing "kill then adapt" directive.
- 5.1× per-run speedup cited in TODO-155 §1.3 came from chemprop-v2 vs V4 pivot
  (different model entirely) — NOT double-counted by Wf-320D.
- Canonical chemprop v2 driver (`train_chemprop_v2_permetal_singletask.py`) **UNCHANGED**;
  fix is opt-in via new smoke runner.

### §2.4 Wf-320E (M1+M2 downstream lift dashboard)

- 6 consumers tracked; **n_me_badge = 2** (TODO-120 m1 Ru AUC +10.35%; TODO-123 m15 RMSD
  −56.15% lower-is-better); **n_honest_negative_badge = 4** (TODO-119 methodology
  change, TODO-121 BELOW_RANDOM, TODO-122 architectural ceiling, TODO-125 STUB_FALLBACK).
- Dashboard is **aggregation-only**, NOT new measurement.
- 4 GPU follow-ups totaling **19.0 hr wall**: TODO-119 6hr + TODO-121 3hr + TODO-122 4hr +
  TODO-125 6hr — all BLOCKED on explicit fresh user OK.

### §2.5 Wf-320F (R20 Phase 0 closeout ACTUAL)

- Wf-313H (Sep 28) was original ship (42 tests class-grouped); Wf-320F (Sep 29) is
  **independent re-verification** (37 tests flat-functions) — **double-anchor** for the
  loader contract.
- 3/3 loaders Phase 0 ready: BindingDB 5,257 rows / NCI-60 7,823 rows / DrugOOD 95,373 rows.
- GATE-6 canonical schema (smiles, metal, cell_line, protein, pIC50, active) PASS for all
  3 loaders; cross-loader concat-compatible.
- Pre-flight `p3_r20_phase0_loaders_complete` now checks **6 test files** (3 Wf-313H + 3
  Wf-320F) and 12 artefacts (3 modules + 3 docs + 6 test files); 79/79 PASS.
- Phase 1+ retrain remains BLOCKED on GPU user authorization.

### §2.6 Wf-320G (MECHANISM_METRIC_MAPPING M22-M44)

- §13 + §14 added as plan-level cross-walk; §1-§12 implementation-level entries
  **PRESERVED** (no destruction of Wf-1/Wf-2/Wf-7/Wf-10/Wf-11/Wf-12/Wf-14/Wf-15 ship work).
- 23 mechanism status distribution: **12 CPU-ONLY-SKELETON + 2 PLAN-ONLY + 2 PROJECTED @ R20
  + 5 HONEST-NEGATIVE + 2 MEASURED (M43, M44) = 23**.
- **M40 (Wet-lab IC50 validation) DEPRECATED** per Wf-317 Tier-2; replaced by
  `mol-metal-wetlab-protocol` skill.
- M36 (Hilbert-ASTSR R² lift) and M38 (3-way MetalCytoToxDB) cite AGENTS.md §5.1
  kill_then_adapt directive as PROJECTED @ R20.

### §2.7 Wf-320H (TODO-134 Phase 2 Vina attempt)

- **PIPELINE PROOF**, NOT SOTA redocking benchmark.
- 6/6 pockets green via CadButEaas `.venv-vlm` Vina 1.2.7 Python binding (recovery path
  for Wf-313A BLOCKED on `which vina → not found`).
- RMSD measures **self-redocking consistency** (RDKit ETKDGv3 start → Vina docked pose
  vs Wf-30 staged PDBQT receptor), NOT crystal-ligand-vs-docked accuracy.
- Per Wf-317 Tier-2: m15 `docked_pose_rmsd.json` stays `NOT_MEASURED` for paper §4; cite
  Wf-320H as **pipeline readiness** evidence in §6 honest limitations.
- SOTA Phase 2 redocking benchmark (n_poses=5 + seeds 0,1,2 + crystal pose pre-alignment)
  remains **FUTURE WORK**.

---

## §3 Categorical counts

| Category | Count | Items |
|---|---:|---|
| **n_completed** | 7 | Wf-320B / C / D / E / F / G / H |
| **n_partial** | 0 | (all in-scope work fully shipped) |
| **n_failed** | 0 | (no failure-mode triggered) |
| **n_honest_negative_dashboard_rows** | 4 | TODO-119, TODO-121, TODO-122, TODO-125 (per Wf-320E dashboard) |
| **n_honest_negative_mechanisms** | 5 | M33, M34, M40, M41, M42 (per Wf-320G §13) |
| **n_projected_mechanisms** | 2 | M36, M38 (PROJECTED @ R20 per Wf-320G §13) |
| **n_deprecated** | 1 | M40 wet-lab (DEPRECATED per Wf-317 Tier-2; per Wf-320G) |

---

## §4 Remaining blockers (post-Wf-320 closeout)

| # | Blocker | Type | Wall (hr) | Wall budget status | GPU/CPU |
|---:|---|---|---:|---|---|
| 1 | TODO-119 P1.4 chemprop v3 LOMO | GPU retrain | 6.0 | kill_then_adapt §0 (user OK required) | GPU |
| 2 | TODO-121 m19 LOMO Rh | GPU retrain | 3.0 | kill_then_adapt §0 (user OK required) | GPU |
| 3 | TODO-122 L5 Hilbert R² | GPU retrain (Wf-94 R20-P3) | 4.0 | kill_then_adapt §0 (user OK required) | GPU |
| 4 | TODO-125 L12 per-pocket Vina | GPU retrain (Wf-130 P3) | 6.0 | kill_then_adapt §0 + TODO-29 F2(a) GPU unblock | GPU |
| 5 | TODO-220 temporal + NCI-60 9-cell full GPU sweep | GPU retrain | 3.9 | kill_then_adapt §0 (user OK required); Wf-310C Part 4 wrapper READY | GPU |
| 6 | TODO-134 Phase 2 SOTA redocking benchmark | CPU+GPU SOTA bench | TBD | future work (per Wf-320H §7) | CPU + GPU |
| 7 | m15 `docked_pose_rmsd.json` refresh | metric JSON | n/a | stays `NOT_MEASURED` per Wf-317 Tier-2; cite Wf-320H as pipeline readiness in paper §6 | n/a |
| **Total GPU** | | | **22.9 hr** | | |

(Plus the existing standing TODO-181 G5 5 hr + TODO-182 G6 7.5 hr + Wf-264 CFM panel +
§9.Z 5 CFM TODOs 37.8 hr per Wf-258 §2.1. The full 2027 GPU backlog remains ~60.7 hr
excluding the Wf-320 follow-ups, or ~83.6 hr including all 5 Wf-320 GPU blockers.)

---

## §5 INDEX.md sync status

| Update | Status |
|---|---|
| Header "Last updated" line | ✅ UPDATED (Wf-320I timestamp appended) |
| §0.6 Wf-317..320 timeline | ✅ ADDED (7 Wf-320 sub-flows + Wf-317..319 ship wave) |
| §0.7 Net effect (post-Wf-320) | ✅ ADDED (completed/ 387 → 397; pre-flight v3 status maintained) |
| §22.13 Wf-320 detailed table | ✅ ADDED (7 sub-flows verbatim per-agent summary) |
| §22.14 Honest framing disclaimer (Wf-320 snapshot) | ✅ ADDED (MEASURED vs NOT MEASURED split) |
| §22 END annotation | ✅ UPDATED (Wf-320 closeout codification) |
| Pre-flight v3 52/52 PASS | ✅ PRESERVED (no Wf-320 sub-flow touched the gate) |

---

## §6 Cross-references

- `TODO/completed/320b_wf291f_harness_autofallback_fix.md` — Wf-320B closeout (13 tests)
- `TODO/completed/320c_todo220_full_impl_closeout.md` — Wf-320C closeout (10 tests)
- `TODO/completed/320d_chemprop_featurization_fix.md` — Wf-320D closeout (12 tests)
- `TODO/completed/320e_downstream_lift_dashboard.md` — Wf-320E closeout (12 tests)
- `TODO/completed/320_r20_phase0_loaders_closeout.md` — Wf-320F closeout (37 tests)
- `TODO/completed/320g_mechanism_metric_mapping_extension.md` — Wf-320G closeout (107 tests)
- `TODO/completed/320h_todo134_phase2_attempt.md` — Wf-320H closeout (6/6 pockets)
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §13 + §14 — Wf-320G plan-level registry
- `molmetal/reports/wf320h_vina_phase2/summary.json` — Wf-320H 6-pocket MEASURED
- `molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json` — Wf-320E 6 consumers
- `molmetal/scripts/wf320b_harness_autofallback.py` — Wf-320B harness module
- `molmetal/scripts/wf320d_featurization_fix.py` — Wf-320D cache module
- `molmetal/scripts/train_chemprop_v2_v2_fastfeaturize.py` — Wf-320D smoke runner
- `molmetal/scripts/m6_wf320h_vina_phase2.py` — Wf-320H Vina driver
- `molmetal/tests/test_wf320b_harness_autofallback.py` — 13 Wf-320B tests
- `molmetal/tests/test_wf320c_todo220_full_impl.py` — 10 Wf-320C tests
- `molmetal/tests/test_wf320d_chemprop_featurization_fix.py` — 12 Wf-320D tests
- `molmetal/tests/test_wf320e_downstream_lift_dashboard.py` — 12 Wf-320E tests
- `molmetal/tests/test_wf320f_*_loader.py` (3 files) — 37 Wf-320F tests
- `molmetal/tests/test_wf320g_mechanism_metric_mappings.py` — 107 Wf-320G tests
- `AGENTS.md` §1.1-§1.5 — 5 standing project rules
- `AGENTS.md` §5.1 — "kill then adapt" GPU directive
- `AGENTS.md` §5.2 — honest framing mandate
- `AGENTS.md` §5.3 — no paper work without authorization
- `party-twenty-spirit` §3.5 — MEASURED vs PROJECTED tag discipline
- `mol-metal-gpu-execution` §0 — kill_then_adapt directive

---

**END OF Wf-320I FINAL SYNTHESIS** — Owner: Wf-320I aggregation subagent;
cross-ref `TODO/completed/{320b,320c,320d,320e,320g,320h,320_r20_phase0}*.md` +
`TODO/INDEX.md` §0.6/§22.13/§22.14; last update 2026-09-29.
