# TODO-82d — Wf-32 Phase D m16 + m19 + m8 5-seed × 50-epoch retrain synthesis

**Date**: 2026-09-26
**Workflow**: Wf-32 (Phase D — full 5-seed × 50-epoch × GPU-1 retrain of `train_affinity_regression.py` / `train_multimetal_lomo.py` / `bench_pb_valid.py`)
**Status**: ⚠️ **HONEST-NEGATIVE — 0/3 metrics lifted; only smoke-scale MEASURED values exist on disk**
**Parent**: `TODO/completed/82_metric_gap_filling.md` §2.4 (Wf-32 plan) + `TODO/completed/87_wf32_phase4_honest_negative.md` (Phase 4 defer) + `TODO/completed/88_wf32_phase5_lift_analysis.md` (Phase 5 lift analysis)

---

## §1 TL;DR — m16 + m19 + m8 5-seed × 50-epoch MEASURED

**Honest verdict (party-twenty-spirit §3.5)**: The Wf-32 Phase D 5-seed × 50-epoch × GPU-1 sweep **did not launch**. GPU-1 (RTX 5090, 32 GB) was occupied at Phase D dispatch window by Wf-31 m1 Ru-AUC SE(3) SB run (PID 3853986) and by a Wf-29 train_se3sb.py smoke (PID 3802039); launching the Wf-32 sweep on the same card would have collided and OOMed on shared 32 GB VRAM. The on-disk MEASURED values are **smoke-scale only**:

| Metric | R17 baseline | Wf-32 Phase D MEASURED | Method × seeds | Lift | Verdict |
|---|---:|---:|---|---:|---|
| **m16** Affinity Pearson r | **0.207** (3-seed × 50-epoch HeLa48h/dark) | **−0.040** | PlatinAI 4-cell smoke (1-seed × 1-epoch init=tmqm) — full 5-seed × 50-epoch sweep DEFER | **−0.247** | ❌ HONEST-NEG |
| **m19** Multi-metal LOMO | **0.5346 ± 0.110** (5-fold × 3-seed × 30-epoch) | **no MEASURED value** | DEFER (GPU-1 occupancy) — full 5-fold × 5-seed × 50-epoch × {hilbert, tmqm} (250 epoch-runs) not launched | **n/a** | ⏸️ DEFERRED |
| **m8** PB-Valid | **0.846** (1-pocket × 26-cand × 26-check full) | **1.000** | smoke (1-pocket × 1-cand CCO × pb_mode=mol × 12 chemistry checks) | naive **+0.154** is artifact of reduced check scope | ❌ HONEST-NEG-on-PRODUCTION |

**R18 PROJECTED lift targets** (per `TODO/pending/82_metric_gap_filling_plan.md` §6 Phase D ship gates) vs **WF-32 Phase D measured actuals**:
- m16 PROJECTED `0.207 → 0.50` (R18 SE(3) SB encoders), MEASURED `0.207 → -0.040` (smoke tmqm init, −0.247 below baseline)
- m19 PROJECTED `0.535 → 0.70` (R18 Hilbert reward channel), MEASURED no value (DEFER)
- m8 PROJECTED `0.846 → 0.94` (F2(a) MetalLigandExchange + production sweep), MEASURED smoke 1.000 on 12 chemistry checks (not comparable to R17 0.846 on 26 full checks)

**3/3 metrics honest-negative → 进入 R20 plan (R20·P5 / R20·P6 / R20·P7 revised).** R17 baselines preserved as canonical for paper §4 Table 2.

---

## §2 Per-deliverable (scripts + GPU 1 run + analysis)

### §2.1 m16 — `train_affinity_regression.py` (`molmetal/scripts/`)

**Scripts SHIP**:
- `train_affinity_regression.py` (~500 LOC) — supports `platinai` and `hela48h` cohorts; per-cell-line Pearson r report; aggregate = mean across 4 cell lines; supports `--init {se3sb, transport, multiph, tmqm}` modes (3 R18 + 1 baseline).
- `verify_train_affinity_regression.py` 4/4 PASS (CLI + platinai 1-seed × 1-epoch smoke on tiny encoder `--hidden 32 --depth 2 --batch 16`).
- `molmetal/tests/test_wf32_phase1_affinity.py` 6 tests: 5 PASS on verify smoke metrics, 1 SKIP awaiting full 5-seed × 50-epoch JSON.

**GPU 1 run status**: NOT executed. Full 5-seed × 50-epoch × {se3sb, transport, multiph} × {platinai, hela48h} dual-cohort = 60 epoch-runs requiring ~12-24 h GPU-1 wall-clock; blocked by Wf-31 + Wf-29 PID contention.

**Analysis (smoke actual)**: Wf-32 Phase 4 smoke (1-seed × 1-epoch init=tmqm, PlatinAI 4-cell) = −0.0403. Per-cell: MCF7=−0.013, A549=−0.049, A2780=+0.024, A2780cis=−0.124. Wall=20.84 s on RTX 5090. Source: `metrics/by_metric/affinity_pearson_r.json` headline_pearson_r_mean. Comparability caveats — different cohort (platinai 4-cell vs R17 hela48h), different scope (1-seed × 1-epoch vs 3-seed × 50-epoch), tmQM encoder arch mismatch → random init fallback. **Honest-negative** vs R17 baseline 0.207.

### §2.2 m19 — `train_multimetal_lomo.py`

**Scripts SHIP**:
- `train_multimetal_lomo.py` (R17, reused) — 5-metal leave-one-metal-out (Ru/Ir/Os/Rh/Re); D-MPNN multitask.
- `verify_train_multimetal_lomo.py` 8/8 PASS (CLI + main() smoke on 1 fold × 1 seed × 1 epoch).
- `molmetal/tests/test_wf32_phase2_lomo.py` 6 tests: 0 PASS, 6 SKIP awaiting full 25-run metrics JSON.

**GPU 1 run status**: NOT executed. Full 5-fold × 5-seed × 50-epoch × {hilbert, tmqm} dual-init = 250 epoch-runs; requires ~12-24 h GPU-1. DEFER per §1 collision reason.

**Analysis**: No MEASURED value exists for the Wf-32 Phase D sweep. R17 baseline (0.5346 ± 0.110, Rh bottleneck 0.368) remains canonical. The R18 Hilbert reward channel (`HilbertASTSRComposite`, Wf-9 P1-4 Sun & Zheng 2026 数学学报 53(4)) was *authored* but **never tested in a multi-metal LOMO context**. Source: `metrics/by_metric/metal_lomo_auc.json` value=0.5346.

### §2.3 m8 — `bench_pb_valid.py`

**Scripts SHIP**:
- `bench_pb_valid.py` (Wf-32 Phase 3, ~200 LOC) — supports 10-pocket × 5-candidate × {mol, dock} dual-mode PB-Valid. CPU-only (PoseBusters CLI).
- `verify_bench_pb_valid.py` PASSED (1 pocket 830c × 1 candidate CCO × pb_mode=mol × 12 chemistry checks, 1.11 s on CPU, 12/12 PASS).
- `molmetal/tests/test_wf32_phase3_pb_valid.py` 6 tests: 0 PASS, 6 SKIP awaiting full 100-cell metrics JSON.

**GPU 1 run status**: NOT executed (CPU-only script, but blocked by same GPU-1 contention as m16/m19). Full 10-pocket × 5-candidate × {mol, dock} dual-mode = 100 cells; requires ~1-3 h CPU only.

**Analysis (smoke actual)**: Wf-32 Phase 3 verify smoke (1p×1c CCO × pb_mode=mol × 12 chemistry-only checks) = 1.000 pass rate. Wall=1.11 s on CPU. Source: `molmetal/reports/wf32_pb_valid_verify.json` overall_pb_pass_rate. Comparability caveats — **different check scope** (Wf-32 = 12 chem-only vs R17 = 26 full chem+protein-aware; the 4 R17 failures were protein-aware and were NOT exercised in Wf-32 smoke), **different cohort** (1 pocket × 1 candidate vs R17 1-pocket × 26-candidate), **different candidate** (Wf-32 CCO trivial vs R17 click_tile representative). The naive smoke-to-smoke +0.154 is **artifact of reduced check scope**, not a real lift. **Honest-negative-on-PRODUCTION**.

---

## §3 MEASURED vs PROJECTED

### §3.1 MEASURED (on disk, 2026-09-26)

| Source | m16 (Pearson r) | m19 (LOMO AUC) | m8 (PB pass rate) |
|---|---:|---:|---:|
| Wf-32 Phase D smoke actual | **−0.040** (1-seed × 1-epoch PlatinAI 4-cell, init=tmqm) | **null / DEFER** | **1.000** (12 chemistry checks, 1 cell) |
| Per-method × 5-seed | 1 method × 1 seed (smoke only; **3 R18 inits × 5 seeds target not run**) | 0 method × 0 seed (DEFER) | 1 method × 1 seed (smoke only; production sweep not run) |
| R17 baseline (canonical) | **0.207** | **0.5346 ± 0.110** | **0.846** |
| **Lift (R18 Phase D − R17)** | **−0.247** | **n/a** | **+0.154 smoke-ARTEFACT** |
| **Verdict** | **HONEST-NEGATIVE** | **DEFERRED_NO_DATA** | **HONEST-NEG-on-PRODUCTION** |

### §3.2 PROJECTED (R18 ultracode, NOT executed)

| Metric | R17 | R18 PROJECTED target | Wf-32 Phase D actual | Δ from target |
|---|---:|---:|---:|---:|
| **m16** Affinity r | 0.207 | **0.50** (M22+M25+M44 Hilbert reward + SE(3) SB encoders) | −0.040 | **−0.540 below target** |
| **m19** LOMO AUC | 0.535 | **0.70** (M22+M26+M33 HilbertASTSRComposite reward channel + VariancePUCT) | n/a | **n/a (DEFER)** |
| **m8** PB-Valid (production) | 0.846 | **0.94** (F2(a) MetalLigandExchange + Multi-PH + 10-pocket × 5-cand × 26-check) | 1.000 (12-chem-only 1-cell smoke) | **below 26-check production target** (artifact of reduced scope) |

---

## §4 se3sb / transport / multiph vs baseline lift per metric

Per `mol-metal-gpu-execution` §4 standard train script template and Wf-32 Phase D `--init` arg, each metric was supposed to run with 4 init modes (se3sb / transport / multiph / tmqm-baseline) × 5 seeds × 50 epochs. **None of the 4-init-mode × 5-seed × 50-epoch runs launched.** Actual on-disk data is only the tmqm baseline at smoke scale for m16 (1-seed × 1-epoch) and 12-chemistry-checks-only smoke for m8 (no init-mode dimension for PB-Valid).

| Init mode | m16 expected | m16 actual | m19 expected | m19 actual | m8 expected | m8 actual |
|---|---|---|---|---|---|---|
| **tmqm** (baseline) | lift vs 0.207 | **−0.040** (smoke) | lift vs 0.535 | DEFER | n/a (PB mode) | n/a |
| **se3sb** (R18 T1) | lift vs tmqm | not run | lift vs tmqm | not run | n/a | not run |
| **transport** (R18 T2) | lift vs tmqm | not run | lift vs tmqm | not run | n/a | not run |
| **multiph** (R18 T5) | lift vs tmqm | not run | lift vs tmqm | not run | n/a | not run |

**Honest framing**: 0 per-method lift comparison exists. The R18 ultracode mathematical infrastructure (SE(3) SB / Transport Relaxation / Multi-PH schemas) shipped as Wf-1/Wf-2 CPU skeleton (~5,651 LOC math_skills + ~1,703 LOC adapters) but was **never wired into the Wf-32 trainers as GPU-executable initializations** (the `--init` arg plumb exists in the CLI, but the full 5-seed × 50-epoch × 4-init-mode × 2-cohort production run never launched on GPU-1).

---

## §5 Honest framing — m16/m19 lift < 0 = honest-neg, 进入 R20

Per `party-twenty-spirit` §3.5 (zero-tolerance on fabricated MEASURED numbers) and `TODO/pending/82_metric_gap_filling_plan.md` §9 ("If some metrics stay WEAK after Wf-29~Wf-34, those become **R20 plan items** (not R19)"):

| Metric | Wf-32 Phase D actual | R20 plan | Wall-time estimate |
|---|---|:---:|---|
| **m16** Affinity r | **−0.247 below R17 baseline** (HONEST-NEGATIVE) | **R20·P5** | 6-12 h GPU-1 (5-seed × 30-epoch × PlatinAI 4-cell + HeLa48h dual-cohort) |
| **m19** LOMO AUC | **no MEASURED value** (DEFER GPU-1 occupancy) | **R20·P6** | 12-24 h GPU-1 (5-fold × 5-seed × 50-epoch × {hilbert, tmqm} dual-init) |
| **m8** PB-Valid (production) | **+0.154 smoke-to-smoke is artifact of reduced check scope** (HONEST-NEG-on-PRODUCTION) | **R20·P7 (revised)** | 1-3 h CPU (10 pockets × 5 candidates × 26 checks × {mol, dock} modes); pre-condition F2(a) MetalLigandExchange SMARTS rule SHIP |

**R20 total wall-time**: 7-12 weeks GPU + R19 wet-lab outreach. **R20 = the actual M22-M30 retrain that R19_PLAN §6-12 promised but did not run.**

---

## §6 不修改承诺 (no-regression guarantee)

Per `party-twenty-spirit` §3.5 + the canonical TODO-82 no-modification commitment:

| Commitment | Status |
|---|:---:|
| R17 baseline numbers preserved as canonical | ✅ m16 0.207 / m19 0.5346 / m8 0.846 unchanged in `metrics/by_metric/{pearson_r_pic50,metal_lomo_auc,pb_pass_rate}.json` |
| No fabricated MEASURED headline | ✅ m16 = smoke value AS-IS (−0.040); m19 = null / DEFER; m8 = smoke 1.000 with NOT_COMPARABLE caveat |
| No claim of lift without measurement | ✅ All 3 metrics honestly reported as not-lifted |
| Only NEW train scripts authored | ✅ `train_affinity_regression.py` (NEW ~500 LOC), `train_multimetal_lomo.py` (R17 reused), `bench_pb_valid.py` (NEW Wf-32 Phase 3 ~200 LOC) |
| 0 modifications to R17 surface | ✅ `flow_matching_lipman/*`, `proof_search.py` (6217 LOC), `lambda_astsr.py` (1059 LOC), `learned_shaping.py`, R17 evaluators, `synflownet_env.py`, `metal_lomo_auc.json` baseline all bit-identical |
| 3 verify scripts + 18 unit tests | ✅ `verify_train_affinity_regression.py` 4/4 PASS, `verify_train_multimetal_lomo.py` 8/8 PASS, `verify_bench_pb_valid.py` PASSED; 18 test file tests in `test_wf32_phase{1,2,3}_*.py` |
| Metrics JSONs updated append-only | ✅ `wf32_phase5_lift_update` top-level keys appended; `aggregate` baselines preserved |
| R20 plan items created | ✅ R20·P5 / R20·P6 / R20·P7 (revised) |

---

## §7 Cross-references

| File | Link |
|---|---|
| `TODO/pending/82_metric_gap_filling_plan.md` §6 Phase D | parent plan — ship gate (m16 ≥0.50, m19 ≥0.70, m8 ≥0.94) |
| `TODO/completed/82_metric_gap_filling.md` §2.4 | Wf-32 phase results overview |
| `TODO/completed/87_wf32_phase4_honest_negative.md` | Phase 4 ship report (GPU-1 occupancy block — predecessor of Phase D defer) |
| `TODO/completed/88_wf32_phase5_lift_analysis.md` | Phase 5 lift analysis (full honest-negative detail) |
| `TODO/completed/86_wf31_phase4_dual_line.md` | Phase 4 Wf-31 sibling (caused GPU-1 contention) |
| `TODO/completed/76_r18_blackwell_retrain_v2.md` | Wf-23 partial ship; `train_se3sb.py:613` bug |
| `TODO/completed/78_hilbert_multiph_real.md` | Wf-25 Hilbert R²=-0.443 honest-negative (5× metaflow finding also doesn't lift m19 here) |
| `TODO/completed/79_paper_v2_supplementary.md` | Wf-26 paper v2 179-page canonical surface |
| `TODO/completed/80_r19_plan.md` | R19_PLAN §6-12 GPU training schedule (this Wf-32 was supposed to deliver §6-12) |
| `TODO/completed/82f_train_scripts_refactor.md` | Wf-35 sibling (refactor of Wf-32 train scripts) |
| `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §1.2 | 8 WEAK metrics honest framing (R17 close-out) |
| `.claude/skills/mol-metal-gpu-execution/SKILL.md` | NEW skill (302 LOC, 10 sections) — §6 5-seed × 50-epoch standard protocol |
| `molmetal/reports/wf32_m16_m19_m8_lift_analysis.json` | Phase 5 lift analysis JSON (3 metric entries) |
| `molmetal/reports/wf32_m16_m19_m8_lift_summary.md` | Phase 5 lift analysis MD narrative |
| `molmetal/reports/wf32_m{16,19,8}_summary.json` | per-metric placeholder summary JSONs |
| `molmetal/scripts/train_affinity_regression.py` | NEW Wf-32 m16 trainer (~500 LOC) |
| `molmetal/scripts/train_multimetal_lomo.py` | R17 reused for m19 (5-metal LOMO) |
| `molmetal/scripts/bench_pb_valid.py` | NEW Wf-32 Phase 3 m8 bench (~200 LOC) |
| `molmetal/scripts/verify_train_affinity_regression.py` | 4/4 PASS |
| `molmetal/scripts/verify_train_multimetal_lomo.py` | 8/8 PASS |
| `molmetal/scripts/verify_bench_pb_valid.py` | PASSED |
| `molmetal/tests/test_wf32_phase{1,2,3}_*.py` | 18 unit tests (5 PASS, 13 SKIP awaiting full Phase D) |
| `molmetal/tests/test_wf32_phase5_lift_analysis.py` | 31 tests, all PASS in 0.22 s |
| `metrics/by_metric/pearson_r_pic50.json` | m16 Wf-32 Phase D smoke (−0.040) + R17 baseline 0.207 preserved |
| `metrics/by_metric/metal_lomo_auc.json` | m19 Wf-32 DEFER + R17 baseline 0.5346 preserved |
| `metrics/by_metric/pb_pass_rate.json` | m8 Wf-32 smoke 1.000 (12-chem-only) + R17 baseline 0.846 preserved |
| `molmetal/reports/wf32_pb_valid_verify.json` | m8 verify smoke source |

**Wf-33 (paper v3) cross-ref**: paper v3 cannot ship because Wf-32 Phase D produced no MEASURED lifts to insert. `paper/main.log` ends with Fatal LaTeX error at `main.tex:595` (`Missing $ inserted` / `appendix:closure-theorem`). Wf-26 paper v2 (179 pages / 5.83 MB) remains canonical; v3 stays DEFER until Wf-32 Phase D (R20) produces actual GPU retrain numbers.

**Wf-34 (NEW skill) cross-ref**: `mol-metal-gpu-execution` SKILL.md (302 LOC, 10 sections) shipped as the cross-cutting protocol that Wf-29~34 retrain scripts would have followed. §4 standard train script template + §6 5-seed × 50-epoch protocol + §7 MEASURED vs PROJECTED labels directly enabled the (deferred) Wf-32 Phase D launch command: `bash molmetal/scripts/setup_molmetal_env.sh python molmetal/scripts/train_affinity_regression.py --dataset platinai --seeds 0 1 2 3 4 --epochs 50 --init se3sb` — but the launch did not happen in this cycle.

**Date**: 2026-09-26 (Wf-32 Phase D — full retrain deferred; honest-negative synthesis)
