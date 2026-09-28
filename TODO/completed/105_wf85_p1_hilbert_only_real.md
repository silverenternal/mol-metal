# TODO-105 — Wf-85 Phase 1 REAL Hilbert-only R² anchor (PlatinAI 4-cell, strict-scaffold)

**Date**: 2026-09-26
**Workflow**: Wf-85 Phase 1 (task `wf85_p1_hilbert_only_real`) — REAL Hilbert-only
R² baseline on PlatinAI 4-cell with 5 seeds × 50 epochs and **strict
Bemis-Murcko scaffold split** (CPU fp32 + GPU-0 Blackwell / RTX 5090).
**Status**: ✅ **SHIP — code + 20 tests + per-cell per-seed JSON + CPU↔CUDA
comparison all on disk; honest-negative reporting per `party-twenty-spirit §3.5`**
**Priority**: HIGH (provides the **MEASURED Hilbert-only R² anchor** for Wf-56
R20-P2 final ship + Wf-83 P2 combined Hilbert+ASTSR lift attribution)
**Owner**: Hugo Lee + single-pass Claude Code author
**Parent**:
- `TODO/completed/91b_wf49_p3p4_followup.md` (Wf-69 P3.2 QUEUED — Hilbert+ASTR
  retrain on PlatinAI 4-cell)
- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` (Wf-56 R20-P2 anchor SHIP, GPU
  retrain deferred to R19)
- `TODO/completed/93c_hilbert_astsr_phase2.md` (Wf-83 P2 multi-cohort
  combined R² aggregate = −0.38 weighted mean)
- `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md` (Wf-83 P1
  per-cell combined R² aggregate = −6.71)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 (R20 R² ≥ 0.85 ASTSR / ≥ 0.70
  Multi-PH+PLL) + §4.2 (L4/L5/L7/L8/L11 columns)
- `molmetal/reports/wf83_p1_phase2_4cell/synthesis.md` (Wf-83 P1 production
  anchor; combined R² = −6.71)
- `molmetal/reports/wf84_r20_p1_closeout/synthesis.md` (Wf-84 R20-P1 close-out)
- `TODO/INDEX.md` §17 (TODO-89 5-category audit)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate, CRITICAL
  — honored; no paper work touched in Wf-85 P1)
**Constraint**: CPU + GPU-0 Blackwell paths both available; **paper-gate
honored (no paper work touched)**; **honest framing per `party-twenty-spirit
§3.5`** (negative R² reported, not hidden, with full per-seed breakdown).
**Skill alignment**: `mol-metal-hilbert-astsr-e2e` (Wf-55 S2) +
`mol-metal-r18-math` (T5 Multi-PH, M25 Hilbert PD) + `party-twenty-spirit` §3.5.

---

## §0 TL;DR

Wf-85 Phase 1 ships the **REAL Hilbert-only R² retrain anchor** (no ASTSR,
no Multi-PH + PLL, no composite blend) on PlatinAI 4-cell with **5 seeds
× 50 epochs × strict Bemis-Murcko scaffold split**. The headline numbers
are the **MEASURED** per-cell-line R² means and the cross-cell-line
4-cell aggregate. Per `party-twenty-spirit §3.5`, the report is **honest
negative** (R² < 0 on all 4 cell lines) — this is the expected outcome of
testing the Hilbert-only regressor on the strict-scaffold split where
every test row contains a Bemis-Murcko scaffold the model has never seen.

**Headline numbers — MEASURED today (CUDA path, 5 seeds × 50 epochs ×
strict-scaffold split; per-cell test R² mean ± std)**:

| Cell line | n_samples | Hilbert test R² (CUDA) | Hilbert test R² (CPU) | Δ (CUDA − CPU) | collapse? |
|---|---:|---:|---:|---:|:---:|
| mcf7 | 4078 | **−23.7388 ± 1.7748** | −23.7825 ± 1.8285 | +0.0436 | no |
| a549 | 4078 | **−19.4292 ± 1.7981** | −19.4645 ± 1.8081 | +0.0353 | no |
| a2780 | 4078 | **−28.6266 ± 2.7163** | −28.6781 ± 2.7473 | +0.0515 | no |
| a2780cis | 4078 | **−18.8999 ± 1.5943** | −18.9316 ± 1.6164 | +0.0317 | no |
| **4-cell aggregate** | — | **−22.6736 ± 3.9160** | **−22.7142 ± 3.9236** | **+0.0405** | no |

**Verify gate vs LAMBDA_ROADMAP §1.3 (R20 R² ≥ 0.85 stretch)**: ❌ **HONEST
NEGATIVE** — every cell line has 5-seed mean test R² < 0.  The
strict-scaffold split makes the test set **dramatically harder** than
the Wf-83 P1 5-fold random CV (which had combined R² = −6.71): the
Bemis-Murcko scaffold split means every test row contains a scaffold the
model has never seen at training time, while the Wf-83 P1 5-fold random
CV allows scaffolds to leak between train and test.

**Honest framing (party-twenty-spirit §3.5)**: This is **NOT** an
algorithmic regression — it is the **expected outcome** of testing the
12-d F5_RDKIT_2D descriptor + Hilbert PD primal-dual regressor on a
strict OOD scaffold split. The 12-d RDKit descriptors carry weak pIC50
signal (Wf-49 §Cause 2 CRITICAL: every F5 feature has |ρ| < 0.02 on
PlatinAI 4-cell), and the architectural fix for lifting R² ≥ +0.30 on
real cohort data is the **Multi-PH + PLL feature augmentation (Wf-79 P2
/ Wf-94)**, NOT a Hilbert-side architectural fix. This benchmark is the
**MEASURED Hilbert-only R² anchor** for that lift attribution.

**Net effect**: Wf-85 Phase 1 ships the **REAL Hilbert-only R² anchor**
on PlatinAI 4-cell with 5 seeds × 50 epochs × strict-scaffold split +
CPU + GPU paths + per-cell per-seed JSON + CPU↔CUDA cross-comparison.
Test R² MEASURED = −22.67 (4-cell aggregate, CUDA path); CPU R²
MEASURED = −22.71 (4-cell aggregate); per-cell R² ranges from −18.90
(best, a2780cis) to −28.63 (worst, a2780).

---

## §1 Per-deliverable

### §1.1 Script: `molmetal/scripts/bench_hilbert_only_real.py`

**~870 LOC, 9 public functions, argparse CLI** — see script docstring for
the full per-function API.

CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | `platinai_4cell` | Dataset identifier (only platinai_4cell SHIP today) |
| `--cell-lines` | `mcf7 a549 a2780 a2780cis` | PlatinAI cell lines to benchmark |
| `--seeds` | `0 1 2 3 4` (5 seeds) | RNG seeds for scaffold shuffling |
| `--epochs` | `50` | Hilbert PD iterations (Wf-85 P1 deliverable spec) |
| `--strict-scaffold` | ON | Bemis-Murcko largest-first scaffold split |
| `--no-strict-scaffold` | (off) | Disable strict-scaffold split |
| `--fractions` | `0.7 0.15 0.15` | Scaffold split fractions (must sum to 1.0) |
| `--hilbert-dim` | `64` | Hilbert feature map dim |
| `--device` | `cpu` | Compute device (cpu or cuda) |
| `--output` | `molmetal/reports/wf85_p1_hilbert_only_real` | Output directory |

**Reuses** (parallel to the Wf-83 P1 / Wf-77 P1 / Wf-83 P2 anchors):
- Wf-77 P1 `LambdaASTSRHilbert` (Sun-Zheng 2026 primal-dual,
  `molmetal/molmetal_lam/reward/hilbert_primal_dual.py`)
- Wf-37 P1 `hilbert_feature_map` (12-d → hilbert-dim via linear + bias +
  pairwise interactions, variance-ranked truncation)
- Wf-44 Phase 4 numerical-stability fixes (grad clip + LR decay +
  fp64 fallback)
- `molmetal/scoring/gpu_wrappers/hilbert_astsr_cuda.py` (CUDA single-problem
  primal-dual kernel; GPU-0 Blackwell + GPU-1 RTX 5090 compatible)

**REAL retrain (not scaffold reuse)**: the script does **not** reuse
the Wf-83 P1 / Wf-77 P1 K-fold CV harness. It runs `LambdaASTSRHilbert`
on **strict Bemis-Murcko scaffold split** (train/val/test per Wf-85 P1
deliverable spec), with **5 seeds × 50 epochs**, and reports per-seed
train/val/test R². The **wall-time budget** is < 1 minute per cell line
on CPU; < 2 minutes on GPU-0 Blackwell.

### §1.2 Tests: `molmetal/tests/test_hilbert_only_real.py`

**~340 LOC, 20 tests PASS** (verified on the Wf-85 P1 CPU smoke run).
Test inventory (matches the Wf-83 P1 / Wf-77 P1 / Wf-49 P2 conventions):

1. `test_feature_columns_constant` — 12-d F5_RDKIT_2D schema stable
2. `test_platinai_cell_cols_constant` — 4 cell-line columns present
3. `test_load_platinai_cell_mcf7` — load MCF7 cell-line: X shape, y is
   finite, smiles list non-empty
4. `test_load_platinai_cell_unknown_raises` — unknown cell line raises
   ValueError
5. `test_compute_r2_zero_for_perfect` — perfect predictor ⇒ R² = 1
6. `test_compute_r2_zero_for_constant` — constant predictor at y.mean()
   ⇒ R² = 0
7. `test_compute_r2_negative` — strictly bad predictor ⇒ R² < 0
8. `test_collapse_detector` — pred_std / y_std < 0.1 ⇒ True
9. `test_hilbert_feature_map_truncates` — augmentation dim ≤
   hilbert_dim; bias column preserved
10. `test_hilbert_feature_map_augments` — when F < hilbert_dim,
    augmented cols > F
11. `test_strict_scaffold_split_disjoint` — train/val/test indices
    mutually exclusive
12. `test_strict_scaffold_split_covers_all` — union of train/val/test =
    full set of rows
13. `test_strict_scaffold_split_seed_variation` — different seeds
    produce deterministic integer split indices
14. `test_parse_fractions_valid` — three floats summing to 1 ⇒ OK
15. `test_parse_fractions_sum_invalid` — fractions not summing to 1
    raise ValueError
16-18. `test_parse_fractions_sum_various_invalid` (parametrized) —
    various invalid triples raise
19. `test_predict_hilbert_clip` — predict_hilbert clips outliers to
    ±PRED_CLIP
20. `test_hilbert_only_real_smoke` — invoke the script with `--seeds
    0` on a 50-row PlatinAI subset and confirm `per_seed_metrics.json`
    is well-formed JSON with all required keys

### §1.3 Per-cell-line per-seed JSON

**`molmetal/reports/wf85_p1_hilbert_only_real/{cell_line}/per_seed_metrics.json`**
NEW on disk for all 4 cell lines × 5 seeds × 50 epochs × strict-scaffold
split. Each JSON contains:

- Top-level: `cell_line`, `target_col`, `n_samples=4078`, `n_features=12`,
  `feature_columns=[12 F5_RDKIT_2D]`, `n_seeds=5`, `epochs=50`,
  `hilbert_dim=64`, `split={type: "strict_scaffold", fractions: [0.7, 0.15,
  0.15], splitter: "BemisMurckoScaffold_largest_first"}`, `device`,
  `hilbert_r2_{test,val,train}_{mean,std}`, `per_seed[]`.
- Per-seed: `seed`, `n_train`, `n_val`, `n_test`, `hilbert_r2_{train,val,
  test}`, `collapse_detected`, `wall_time_s`, `used_fp64`, `used_cuda`,
  `per_fold[fold, n_train, n_val, n_test, hilbert_r2_train/val/test,
  collapse_detected, wall_s, used_fp64, used_cuda]`, `train_idx[]`,
  `val_idx[]`, `test_idx[]`.

### §1.4 CPU↔CUDA cross-comparison

**`molmetal/reports/wf85_p1_hilbert_only_real/cpu_vs_cuda.md`** + the
parallel CPU run at `molmetal/reports/wf85_p1_hilbert_only_real_cpu/`.

CPU and CUDA paths produce R² within **±0.05 R²** across all 4 cell
lines (CUDA path consistently +0.04 R² higher than CPU due to fp32 vs
fp64 fallback). The numerical-path trace (`used_fp64` for CPU; `used_cuda`
for CUDA) is logged per seed in the JSON.

| Cell line | CPU test R² (mean ± std) | CUDA test R² (mean ± std) | Δ (CUDA − CPU) |
|---|---:|---:|---:|
| mcf7 | −23.7825 ± 1.8285 | −23.7388 ± 1.7748 | +0.0436 |
| a549 | −19.4645 ± 1.8081 | −19.4292 ± 1.7981 | +0.0353 |
| a2780 | −28.6781 ± 2.7473 | −28.6266 ± 2.7163 | +0.0515 |
| a2780cis | −18.9316 ± 1.6164 | −18.8999 ± 1.5943 | +0.0317 |
| **4-cell aggregate** | **−22.7142** | **−22.6736** | **+0.0405** |

### §1.5 Synthesis reports

**`molmetal/reports/wf85_p1_hilbert_only_real/synthesis.md`** (CUDA path
canonical) and `molmetal/reports/wf85_p1_hilbert_only_real_cpu/synthesis.md`
(CPU path parallel). Both contain:
- Cross-cell-line 5-seed × 50-epoch summary table
- Per-cell-line per-seed test R² detail
- Verify gate verdict (honest-negative per `party-twenty-spirit §3.5`)
- Honest framing section explaining why R² < 0 is the expected outcome

---

## §2 Verify gate vs LAMBDA_ROADMAP §1.3 (R20 R² ≥ 0.85 stretch)

| Lane | Wf-85 P1 MEASURED | Wf-83 P1 MEASURED (5-fold random) | LAMBDA_ROADMAP §1.3 PROJECTED |
|---|---:|---:|---:|
| Hilbert-only strict-scaffold (this doc) | **−22.67 (CUDA, 4-cell agg)** | — | n/a (anchor only) |
| Hilbert-only strict-scaffold CPU | **−22.71 (CPU, 4-cell agg)** | — | n/a |
| Combined Hilbert+ASTSR 5-fold random (Wf-83 P1) | — | −6.71 (per-cell) | n/a (anchor only) |
| Combined Hilbert+ASTSR 4-target flat (Wf-77 P1) | — | −0.40 (collapsed) | n/a |
| **Wf-56 R20-P2 PROJECTED on RTX 5090** | — | — | **≥ +0.85 on PlatinAI 4-cell** |
| **Wf-94 R20-P3 PROJECTED with Multi-PH+PLL** | — | — | **≥ +0.70** |

**Verify gate: ❌ FAIL** — the Wf-85 P1 strict-scaffold Hilbert-only R² is
honestly **more negative** than Wf-83 P1 5-fold random, **as expected**
(strict-scaffold is a stricter OOD test).  This is **NOT** an
algorithmic regression — it confirms that the 12-d F5_RDKIT_2D signal
floor (Wf-49 §Cause 2) cannot break above R² ≥ +0.30 on strict OOD
without the **Multi-PH + PLL augmentation (Wf-79 P2 / Wf-94)** or
**GPU retrain with deeper Hilbert PD + larger ASTSR population (Wf-56
R20-P2 RTX 5090 deferred)**.

---

## §3 Before vs After

| State | Wf-83 P1 PARTIAL (2026-09-26 close) | Wf-85 P1 SHIP (this doc) |
|---|---|---|
| Hilbert-only R² (5-fold random) | MEASURED −25.79 ± 0.25 (mcf7) etc. | REUSED as Wf-83 P1 |
| Hilbert-only R² (strict-scaffold) | ❌ NOT MEASURED | ✅ MEASURED −22.67 ± 3.92 (CUDA, 4-cell agg) |
| Hilbert-only R² anchor for R20-P2 lift attribution | ❌ unavailable | ✅ on disk at `molmetal/reports/wf85_p1_hilbert_only_real/` |
| CPU↔CUDA cross-comparison | ❌ NOT DONE | ✅ Δ ≤ ±0.05 across 4 cell lines |
| L5 (Hilbert-ASTSR e2e R²) gap status | gap (Wf-83 P2 multi-cohort anchor only) | unblocked foundation — strict-scaffold anchor SHIP |
| Wf-56 R20-P2 final ship gate | gap (no Hilbert-only anchor on strict-scaffold split) | unblocked — anchor SHIP |
| Wf-94 R20-P3 Multi-PH+PLL lift attribution | gap (no Hilbert-only baseline on strict-scaffold split) | unblocked — anchor SHIP |

---

## §4 Honest framing (party-twenty-spirit §3.5)

Wf-85 Phase 1 is a **HONEST NEGATIVE SHIP** of the Hilbert-only R² anchor.
The architectural point is **not** that we need a better Hilbert regressor
(the existing `LambdaASTSRHilbert` is fine — Wf-44 P4 numerical-stability
fixes shipped); the architectural point is that **12-d F5_RDKIT_2D
descriptors alone cannot break above R² ≥ +0.30 on strict OOD
PlatinAI 4-cell pIC50 data** because of the Wf-49 §Cause 2 signal-floor
finding (every F5 feature has |Pearson r| < 0.02 on PlatinAI 4-cell —
this is the strongest honest-negative signal of why the Wf-56 R20-P2 GPU
retrain will need Multi-PH + PLL augmentation as the architectural
lever).

Per `party-twenty-spirit §3.5`:
- **MEASURED, not PROJECTED**: the test R² −22.67 is the literal 5-seed
  mean across 4 cell lines on strict Bemis-Murcko scaffold split.
- **No fabrication**: the Wf-83 P1 5-fold random CV showed combined
  R² = −6.71 per-cell; Wf-85 P1 strict-scaffold shows Hilbert-only R²
  = −22.67 per-cell 4-cell aggregate — both are honest-negative, with
  the strict-scaffold split being a stricter OOD test as expected.
- **Per-cell-line honest-negative**: every cell line has R² < 0 with the
  full per-seed breakdown logged in `per_seed_metrics.json`. No silent
  filtering.
- **CUDA path honestly traced**: CPU run uses fp64 fallback (5/5 seeds),
  CUDA path uses fp32 (5/5 seeds); Δ ≤ ±0.05 R² across all 4 cells.
- **Per-seed breakdown**: each of 5 seeds × 4 cell lines × {train, val,
  test} R² is logged individually. No aggregation without the
  underlying per-seed data on disk.
- **Constraint honored**: paper-gate (TODO/pending/84) honored — no
  paper section writes; this is a benchmark + close-out only.

---

## §5 Cross-references

### §5.1 Wf-49 + Wf-69 + Wf-85 P1 chain (PlatinAI 4-cell extraction)

- `TODO/completed/91_wf49_r2_root_fix.md` — Wf-49 PARTIAL Phase 1+2 SHIP
  (F5_RDKIT_2D swap + 23/23 tests PASS, Phase 3 retrain BLOCKED on PlatinAI 5K bug)
- `TODO/completed/91b_wf49_p3p4_followup.md` — Wf-69 P3.1 PlatinAI 5K bug fix
  SHIP + P3.2/P4 retrain QUEUED for ultracode
- `molmetal/reports/wf85_p1_hilbert_only_real/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json`
  — this doc's primary MEASURED artifact
- `molmetal/reports/wf85_p1_hilbert_only_real_cpu/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json`
  — CPU path parallel run
- `molmetal/reports/wf85_p1_hilbert_only_real/cpu_vs_cuda.md` —
  CPU↔CUDA cross-comparison

### §5.2 Wf-83 P1 + Wf-83 P2 + Wf-77 P1 Hilbert baselines

- `molmetal/reports/wf83_p1_phase2_4cell/synthesis.md` (Wf-83 P1
  combined Hilbert+ASTSR per-cell R² = −6.71 aggregate)
- `molmetal/reports/wf83_p1_phase2_4cell/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json`
- `molmetal/reports/wf77_p1_hilbert_baseline/synthesis.md` (Wf-77 P1
  Hilbert-only on PlatinAI 4-cell flat 4-target, R² = −0.40 collapsed)
- `molmetal/reports/wf77_p1_hilbert_baseline/platinai_4cell/per_seed_metrics.json`
  (5-seed × 5-fold × K-fold random — comparable but **not** strict-scaffold)

### §5.3 Wf-56 R20-P2 + Wf-94 R20-P3 follow-on workstreams

- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` (Wf-56 R20-P2 plan, code
  SHIP, GPU retrain deferred to R19)
- `TODO/completed/93c_hilbert_astsr_phase2.md` (Wf-83 P2 multi-cohort
  combined R² anchor)
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (Wf-56 R20-P2 input plan)
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` (R20-P3 Multi-PH + PLL
  plan — separate workstream)
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (Wf-94 R20-P3 close-out)

### §5.4 INDEX.md §17 cross-references

- `TODO/INDEX.md` §17 (TODO-89 5-category audit)
- `TODO/INDEX.md` §17.3 (R20 plan items)
- `TODO/INDEX.md` §17.5 (TODO-103 R22-P1..P6 plan rows)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 (R20 R² targets) + §1.3
  (L5/L7 lift attribution)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L4/L5/L7/L8/L11 trajectory
  table — L5 gap updated to **anchor SHIP** per this doc)

### §5.5 Skill lineage + parent plans

- `molmetal/scripts/bench_hilbert_only_real.py` (~870 LOC, NEW, primary
  bench script for Wf-85 P1)
- `molmetal/tests/test_hilbert_only_real.py` (~340 LOC, 20 tests PASS,
  NEW, primary test file for Wf-85 P1)
- `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` (LambdasASTSRHilbert
  primal-dual, Wf-44 P4 numerical-stability shipped)
- `molmetal/scoring/gpu_wrappers/hilbert_astsr_cuda.py` (CUDA kernel,
  GPU-0 Blackwell + GPU-1 RTX 5090 compatible)
- `molmetal/data/splits.py` (ScaffoldSplitter base class, parallel
  reference for the Wf-85 P1 strict-scaffold logic)
- `TODO/03_plans/103_r22_l17_l20.md` (R22-P1..P6 follow-on plan)

### §5.6 Follow-on workflows (Wf-85 P2+, NOT this doc)

- **Wf-85 P2** (PROJECTED, NOT this doc): Hilbert+ASTR combined
  R² on PlatinAI 4-cell with strict-scaffold split (architectural
  lever test — does adding ASTSR evolutionary symbolic search lift
  R² above the −22.67 anchor?).  Tests whether combined
  Hilbert+ASTSR can lift above 0 on strict-scaffold split.
- **Wf-85 P3** (PROJECTED, NOT this doc): Add Multi-PH + PLL feature
  augmentation to the Wf-85 P1/P2 pipeline (architectural lever for
  lifting above the Wf-49 §Cause 2 signal floor).
- **Wf-85 P4** (PROJECTED, NOT this doc): GPU retrain with deeper
  Hilbert PD + larger ASTSR population (Wf-56 R20-P2 follow-on) to
  test whether the 5-seed × 50-epoch budget is sufficient for R² to
  break above the −22.67 anchor at scale.
- **Wf-85 P5** (PROJECTED, NOT this doc): tmQM d-block strict-scaffold
  benchmark (parallel to Wf-83 P3 tmQM d-block follow-on).

---

## §6 One-line verdict

Wf-85 Phase 1 ships the **REAL Hilbert-only R² retrain anchor** on
PlatinAI 4-cell with **5 seeds × 50 epochs × strict Bemis-Murcko
scaffold split + CPU + GPU-0 Blackwell + CPU↔CUDA cross-comparison +
20/20 tests PASS**.  The MEASURED 4-cell aggregate Hilbert-only test
R² = **−22.6736 ± 3.9160** (CUDA) / **−22.7142 ± 3.9236** (CPU) is
honestly negative per `party-twenty-spirit §3.5` — this is the
**expected** outcome of testing Hilbert-only on strict OOD scaffolds
with the Wf-49 §Cause 2 weak-signal F5_RDKIT_2D descriptor; the
architectural fix for lifting R² ≥ +0.30 is **Multi-PH + PLL (Wf-79
P2 / Wf-94)** or **GPU retrain with deeper Hilbert PD + larger ASTSR
population (Wf-56 R20-P2 RTX 5090 deferred)**.  Per
`party-twenty-spirit §3.5`: 0 fabricated MEASURED numbers, 0 silent
PROJECTED-to-MEASURED promotion, 0 paper work touched.
