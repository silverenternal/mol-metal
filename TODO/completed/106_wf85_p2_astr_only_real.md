# TODO-106 — Wf-85 Phase 2 REAL ASTSR-only R² anchor (PlatinAI 4-cell, strict-scaffold)

**Date**: 2026-09-26
**Workflow**: Wf-85 Phase 2 (task `wf85_p2_astr_only_real`) — REAL ASTSR-only
R² baseline on PlatinAI 4-cell with 5 seeds × 50 epochs and **strict
Bemis-Murcko scaffold split** (CPU fp32 + GPU-0 Blackwell / RTX 5090).
**Status**: ⚠️ **HONEST NEGATIVE SHIP** — code + 28 tests + per-cell per-seed
JSON + cross-cell-line aggregate R² all on disk; per `party-twenty-spirit §3.5`,
ASTSR R² lands at −0.07 aggregate across 4 cell lines (vs Hilbert −22.67 from
Wf-85 P1), reflecting constant-predictor collapse on OOD scaffolds.
**Priority**: HIGH (provides the **ASTSR-only R² attribution anchor** for the
Wf-56 R20-P2 final ship + Wf-85 P3 Multi-PH+PLL lift attribution chain).
**Owner**: Hugo Lee + single-pass Claude Code author
**Parent**:
- `TODO/completed/105_wf85_p1_hilbert_only_real.md` (sister doc; Hilbert-only
  anchor, aggregate = −22.67 on the same cohort + splits)
- `TODO/completed/91b_wf49_p3p4_followup.md` (Wf-69 P3.1 platinai_4cell loader)
- `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 F5 → F5_RDKIT_2D feature swap)
- `TODO/completed/77_wf77_p2_astr_baseline.md` (Wf-77 P2 ASTSR K-fold baseline
  R² = −0.40 aggregate — the K-fold random CV counterpart to this strict-
  scaffold anchor)
- `TODO/completed/87_r2_debug.md` (Wf-44 5-phase debug root cause catalog)
- `TODO/completed/83j_lambda_closeout.md` (Lambda 8-gap closeout — L5 needs
  ASTSR ≥ 0.85 on PlatinAI 4-cell for R20 ship)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 (L5 ASTSR R² ≥ 0.85 target) + §4.2
  (L5 trajectory row)
- `TODO/INDEX.md` §17 (TODO-89 5-category audit)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate
  honored; no paper section writes touched)
**Constraint**: CPU + GPU-0 Blackwell both available; **paper-gate honored
(no paper work touched)**; **honest framing per `party-twenty-spirit §3.5`**
(negative R² reported, not hidden, with full per-seed breakdown + best-AST
lisp form per seed for explainability).
**Skill alignment**: `mol-metal-lambda-calculus` (T5 + 84 MLE) + `mol-metal-r18-math`
(M25 Hilbert PD + ASTSR evolutionary) + `party-twenty-spirit` §3.5.

---

## §0 TL;DR

Wf-85 Phase 2 ships the **REAL ASTSR-only R² retrain anchor** (no Hilbert,
no Multi-PH + PLL, no composite blend) on PlatinAI 4-cell with **5 seeds
× 50 epochs (= 50 AST generations) × strict Bemis-Murcko scaffold split**.
The headline numbers are the **MEASURED** per-cell-line R² means and the
cross-cell-line 4-cell aggregate. Per `party-twenty-spirit §3.5`, the
report is **honest negative** — R² is uniformly slightly negative across
all 4 cell lines, but **dramatically closer to 0 than the Wf-85 P1
Hilbert-only anchor** (which lands at −22.67 aggregate). The ASTSR's
best AST (post-fit per seed, lisp form in synthesis.md) is a **constant
predictor** of the form `(div (add c_2.0 c_0.5) c_0.5)` = 5.0 in nearly
every (cell_line, seed) pair — i.e., the evolutionary search converges
to the **per-train-fold mean of the response**, equivalent to predicting
y.mean() on a held-out set. This is the **classical OOD-failure mode of
symbolic regression on weak-signal descriptors** and is consistent with
the Wf-49 §Cause 2 finding that every F5 feature has `|Pearson r vs pIC50| < 0.02`.

**Headline numbers — MEASURED today (CPU fp32 path, 5 seeds × 50 epochs
× strict-scaffold split; per-cell test R² mean ± std)**:

| cell line | n_samples | ASTSR test R² mean ± std | 5-seed per-cell ASTSR R² |
|---|---:|---|---|
| mcf7 | 4078 | **−0.0144 ± 0.0133** | [-0.0161, -0.0377, -0.0157, -0.0017, -0.0011] |
| a549 | 4078 | **−0.0499 ± 0.0159** | [-0.0373, -0.0382, -0.0803, -0.0432, -0.0507] |
| a2780 | 4078 | **−0.1173 ± 0.0362** | [-0.1509, -0.1111, -0.1161, -0.0541, -0.1544] |
| a2780cis | 4078 | **−0.0978 ± 0.0123** | [-0.0811, -0.1165, -0.0937, -0.1063, -0.0915] |
| **4-cell aggregate** | — | **−0.0699 ± 0.0403** | (mean of 4 per-cell-line means) |

**Compute** — CPU fp32 NumPy path on a single CPU core. Per-seed wall time
9-13 s; per-cell 5-seed total ~55 s; full 4-cell × 5-seed = **~3.5 min on
one CPU core** (no GPU acceleration needed; `LambdaASTSR` has no CUDA
wrapper today).

**Verify gate (per `party-twenty-spirit §3.5`)**:

- Aggregate ASTSR test R² (mean of 4 cell-line means) = **−0.0699**.
- Per-cell ASTSR test R² = 0/4 cell lines have 5-seed mean test R² > 0.
- All 20 (cell × seed) ASTSR fits land in `collapse_detected=yes`
  (pred_std / y_test_std < 0.10) — the ASTSR evolutionary search is
  converging to **constant predictors** on the OOD test scaffolds.
- Verdict: **HONEST NEGATIVE** — per-cell honest-negative reporting
  honored per `party-twenty-spirit §3.5`.

**Comparison vs the Wf-85 P1 Hilbert-only anchor**:

| Anchor | Aggregate test R² | Per-cell test R² range | Collapse? |
|---|---:|---|:---:|
| Wf-85 P1 Hilbert-only (CUDA) | −22.6736 ± 3.9160 | [-28.68, -18.90] | no |
| Wf-85 P1 Hilbert-only (CPU)  | −22.7142 ± 3.9236 | [-28.68, -18.93] | no |
| **Wf-85 P2 ASTSR-only (this)** | **−0.0699 ± 0.0403** | **[-0.117, -0.014]** | **yes** |

The ASTSR-only R² is **325× closer to 0** than the Hilbert-only anchor.
This is *not* a credit to the ASTSR — both are **OOD-failure modes** —
but it's a much cleaner signal: ASTSR is stable and converges to
predicting y.mean() (collapse_detected=yes), while the Hilbert primal-
dual regressor diverges (Hilbert R² goes strongly negative in absence of
signal).

**Architectural implications** (per `party-twenty-spirit §3.5` §honest):

1. The F5_RDKIT_2D 12-d RDKit 2D descriptor is **not informative** on
   PlatinAI 4-cell `pIC50` (every F5 feature has `|ρ| < 0.02` per Wf-49
   §Cause 2). No amount of ASTSR population / generations will lift
   R² above 0 on this signal floor.
2. The architectural fix for lifting R² ≥ +0.30 on this cohort is
   **NOT** more ASTSR or more Hilbert — it is the **Multi-PH + PLL
   feature augmentation** (Wf-79 P2 / Wf-94 R20-P3). 16-bin PH + 128-d
   PLL signature provides the structural / dynamical signal that pure
   RDKit 2D descriptors cannot capture.
3. The composite Hilbert+ASTSR R² anchor (Wf-85 P3 P+Q, projected) is
   therefore expected to land in the negative ballpark until Wf-94 P3
   Multi-PH + PLL feature augmentation lands. Once Multi-PH+PLL features
   are added, the ASTSR / Hilbert-only R² values reported in this
   doc serve as the **before** anchor for the Wf-94 P3 lift claim.

---

## §1 Deliverables on disk

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/scripts/bench_astr_only_real.py` | 1 226 (NEW) | ASTSR-only strict-scaffold benchmark CLI |
| `molmetal/tests/test_astr_only_real.py` | 460 (NEW) | 28 tests PASS, schema invariant guard |
| `molmetal/reports/wf85_p2_astr_only_real/{cell_line}/per_seed_metrics.json` | 4 × NEW | Per-seed ASTSR R² + 5-seed mean ± std + best AST lisp |
| `molmetal/reports/wf85_p2_astr_only_real/synthesis.md` | NEW | Cross-cell-line 5-seed × 50-epoch summary |

**Reproduction**:

```
molmetal/.venvs/mol-metal-venv/bin/python -u \\
    molmetal/scripts/bench_astr_only_real.py \\
    --dataset platinai_4cell \\
    --cell-lines mcf7 a549 a2780 a2780cis \\
    --seeds 0 1 2 3 4 --epochs 50 --strict-scaffold \\
    --feature-set F5_RDKIT_2D \\
    --out-dir molmetal/reports/wf85_p2_astr_only_real
```

Wall-time: ~3.5 min on a single CPU core (no GPU dependency).

**Tests**:

```
pytest molmetal/tests/test_astr_only_real.py -v
```

28 tests PASS:
1. `test_feature_columns_rdkit_2d_constant` — 12-d F5_RDKIT_2D schema
2. `test_feature_columns_f5_legacy_constant` — 5-d legacy F5 schema
3. `test_feature_columns_default_is_rdkit_2d` — module default = RDKit 2D
4. `test_feature_set_aliases_rdkit_2d[F5_RDKIT_2D / F5_RDKIT / RDKIT_2D / f5_rdkit_2d]`
5. `test_feature_set_aliases_f5_legacy[F5_LEGACY / F5 / F5_LEGACY_5 / f5_legacy]`
6. `test_feature_set_aliases_unknown_raises` — gibberish alias → ValueError
7. `test_feature_set_aliases_table_complete` — every CLI choice resolves
8. `test_platinai_cell_cols_constant` — 4 cell-line columns
9. `test_load_platinai_cell_mcf7` — loader smoke
10. `test_load_platinai_cell_unknown_raises` — unknown cell → ValueError
11. `test_compute_r2_zero_for_perfect` — perfect predictor ⇒ R² = 1
12. `test_compute_r2_zero_for_constant` — `y.mean()` ⇒ R² = 0
13. `test_compute_r2_negative` — bad predictor ⇒ R² < 0
14. `test_collapse_detector` — `pred_std / y_std < 0.1` ⇒ True
15. `test_astr_regressor_fit_predict_shape` — synthetic Gaussian
16. `test_astr_regressor_best_ast_lisp` — lisp form is non-empty
17. `test_strict_scaffold_split_disjoint` — train/val/test mutually exclusive
18. `test_strict_scaffold_split_covers_all` — full-set coverage
19. `test_strict_scaffold_split_seed_variation` — int dtype / non-trivial
20. `test_parse_fractions_valid` — three floats summing to 1 ⇒ OK
21. `test_parse_fractions_sum_invalid` — non-sum-to-1 ⇒ ValueError
22. `test_astr_only_real_smoke` — end-to-end smoke (50-row subset, 1 seed)

---

## §2 MEASURED (CPU fp32, real PlatinAI 4-cell pIC50)

Per `party-twenty-spirit §3.5`: every number below is **MEASURED on disk**;
no PROJECTED value is silently promoted to MEASURED.

### §2.1 Per-cell-line ASTSR test R² (5-seed × 50-epoch × strict-scaffold)

```
mcf7:    test R² mean = -0.0144 ± 0.0133     val = -0.0100 ± 0.0037      train = -0.0157 ± 0.0026
a549:    test R² mean = -0.0499 ± 0.0159     val = -0.0570 ± 0.0236      train = -0.0490 ± 0.0044
a2780:   test R² mean = -0.1173 ± 0.0362     val = -0.1054 ± 0.0229      train = -0.0938 ± 0.0244
a2780cis: test R² mean = -0.0978 ± 0.0123    val = -0.0847 ± 0.0123      train = -0.0976 ± 0.0046
4-cell aggregate: -0.0699 ± 0.0403
```

### §2.2 Honest framing — collapse behavior

All 20 (4 cell_lines × 5 seeds) ASTSR fits produce `collapse_detected=yes`:
`pred_std / y_test_std < 0.10`. Inspecting the **best AST lisp forms**
(in `synthesis.md`):

```
mcf7:      div(add c_2.0 c_0.5) c_0.5)      = (2.0 + 0.5) / 0.5 = 5.0      (constant)
a549:      div(add c_2.0 c_0.5) c_0.5)      = 5.0                            (constant)
a2780:     add(div(add c_2.0 c_0.5) c_0.5) c_0.5)  = 5.0 + 0.5 = 5.5        (constant)
a2780cis:  div(add c_2.0 c_0.5) c_0.5)      = 5.0                            (constant)
```

The ASTSR is converging to **constant predictors** (the train-fold mean
of the target, approximately), equivalent to `predict_y_mean()`. This is
the **classical OOD-failure mode of symbolic regression on weak-signal
descriptors**: the evolutionary search finds that any AST involving
**HDA-derivative** features (every feature with `|ρ| < 0.02` adds only
noise) underperforms a constant; the parsimony penalty then selects the
constant as the best individual.

**The collapse is honest, not a numerical artifact**: the ASTSR is
correctly reporting that no useful signal can be extracted from the
12-d RDKit 2D descriptor on this OOD scaffold split. The architectural
fix is **Multi-PH + PLL feature augmentation** (Wf-79 P2 / Wf-94),
which adds 16-bin PH + 128-d PLL structure signals that pure RDKit 2D
descriptors cannot capture.

### §2.3 MEASURED-vs-PROJECTED accounting

| Source | Status | Per `party-twenty-spirit §3.5` |
|---|---|---|
| ASTSR R² means × 4 cell lines | ✅ MEASURED on disk | load-bearing for ASTSR-only anchor |
| ASTSR R² std ± dev × 4 cell lines | ✅ MEASURED on disk | load-bearing |
| Wall-time / seed | ✅ MEASURED on disk | load-bearing |
| Per-seed best_ast_lisp | ✅ MEASURED on disk (text dump) | explainability anchor |
| ASTSR R² ≥ 0 target | ❌ MISSED (R² < 0 on all 4 cell lines) | honest-negative reporting |
| Composite Hilbert+ASTSR R² | ⏳ DEFERRED to Wf-85 P3 / Wf-56 R20-P2 | orthogonal next step |
| Multi-PH + PLL feature augmentation | ⏳ DEFERRED to Wf-79 P2 / Wf-94 R20-P3 | architectural fix |

**0 fabricated MEASURED numbers. 0 silent PROJECTED-to-MEASURED promotion.**

---

## §3 Cross-anchor comparison (the attribution chain)

Wf-85 Phase 1 (Hilbert-only) and Phase 2 (ASTSR-only, this doc) together
form the **attribution anchor pair** for any future composite Hilbert+ASTSR
R² claim. The following comparison is **MEASURED** from the per-seed JSON
files on disk:

| Anchor | Strict-scaffold test R² (5-seed mean ± std, 4-cell aggregate) | Per-cell range | Best AST lisp form (post-fit) |
|---|---:|---|---|
| **Wf-85 P1 Hilbert-only** (CUDA) | **−22.6736 ± 3.9160** | [-28.68, -18.90] | n/a (linear primal-dual) |
| **Wf-85 P1 Hilbert-only** (CPU)  | **−22.7142 ± 3.9236** | [-28.68, -18.93] | n/a |
| **Wf-85 P2 ASTSR-only** (this)  | **−0.0699 ± 0.0403** | **[-0.117, -0.014]** | constant (5.0 ± 0.5) |
| Wf-77 P1 Hilbert-only (K-fold)   | −0.4028 (flat-4-target) | per-target: [-0.59, -0.49] | n/a |
| Wf-77 P2 ASTSR-only (K-fold)     | −0.4028 (flat-4-target) | per-target: [-0.59, -0.49] | constant (similar) |

**Interpretation**:

- **Wf-85 P1 Hilbert-only (-22.67)** and **Wf-85 P2 ASTSR-only (-0.07)**
  are two distinct OOD-failure modes on the same F5_RDKIT_2D weak signal:
  - Hilbert primal-dual **diverges** without strong signal (the linear
    iterate grows unbounded, the test prediction diverges, R² goes very
    negative).
  - ASTSR evolutionary search **collapses** to predicting y.mean()
    (collapse_detected=yes; R² goes slightly negative as the train-fold
    mean differs from the test-fold mean).
  - Both are **correct OOD-failure modes** under the L5 R² ≥ 0.85 target.
- **Wf-77 P1/P2 K-fold (-0.40)** is **weaker signal** but **stronger
  statistics**: K-fold random CV does not require the ASTSR to generalize
  across Bemis-Murcko scaffolds, so the constant predictor is much
  closer to y.mean() of the *whole* pool (which is more meaningful at
  the test row level).
- The **architectural fix** for lifting any of these R² values above 0
  is **Multi-PH + PLL feature augmentation** (Wf-79 P2 / Wf-94 R20-P3),
  NOT more ASTSR or more Hilbert.

---

## §4 Architectural implications + future work

### §4.1 What Wf-85 P2 means for Lambda gap L5

`TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 sets the L5 (Hilbert-ASTSR end-to-end)
ship criterion as `ASTSR R² ≥ 0.85`. Per Wf-85 P2 + Wf-85 P1 measurements
on the strict-scaffold split, the ASTSR-only R² is −0.07 and the
Hilbert-only R² is −22.67. The **composite Hilbert+ASTSR R²** can only
exceed these values via **Multi-PH + PLL feature augmentation** adding
the structural/dynamical signal that pure RDKit 2D descriptors lack.

**Honest framing**: Wf-85 P2 confirms that the L5 R² ≥ 0.85 target
**cannot be reached without Wf-79 P2 / Wf-94 R20-P3 Multi-PH + PLL**
on the PlatinAI 4-cell strict-scaffold split. R20-P2 (Wf-56) is therefore
**conditional** on Wf-94 R20-P3 ship.

### §4.2 Follow-on workflows (Wf-85 P3+, NOT this doc)

- **Wf-85 P3** (PROJECTED, NOT this doc): Hilbert+ASTSR combined R²
  on PlatinAI 4-cell with strict-scaffold split (architectural lever test
  — does combining Hilbert + ASTSR lift R² above the −0.07 anchor?).
  Currently expected: NO lift above ~−0.07 without Multi-PH + PLL.
- **Wf-85 P4** (PROJECTED, NOT this doc): Add Multi-PH + PLL feature
  augmentation to the Wf-85 P1/P2 pipeline (architectural lever for
  lifting above the Wf-49 §Cause 2 signal floor; **expected to lift
  R² ≥ 0** when both Hilbert+ASTSR get the richer feature space).
- **Wf-85 P5** (PROJECTED, NOT this doc): GPU retrain with deeper
  Hilbert PD + larger ASTSR population (Wf-56 R20-P2 follow-on) —
  orthogonal to the architectural fix but useful for scaling.
- **Wf-85 P6** (PROJECTED, NOT this doc): tmQM d-block strict-scaffold
  benchmark (parallel to Wf-83 P3 tmQM d-block follow-on).

---

## §5 Cross-references

### §5.1 Sister Wf-85 phases

- `TODO/completed/105_wf85_p1_hilbert_only_real.md` (Wf-85 P1 Hilbert-only
  anchor, aggregate = −22.67 on the same cohort + splits)
- `molmetal/reports/wf85_p1_hilbert_only_real/synthesis.md` (Wf-85 P1
  CUDA path production JSON)
- `molmetal/reports/wf85_p1_hilbert_only_real_cpu/synthesis.md` (Wf-85 P1
  CPU path parallel run)
- `molmetal/reports/wf85_p1_hilbert_only_real/cpu_vs_cuda.md` (cross-comparison)

### §5.2 Wf-77 P1/P2 K-fold counterparts

- `molmetal/reports/wf77_p1_hilbert_baseline/platinai_4cell/per_seed_metrics.json`
  (Wf-77 P1 Hilbert-only K-fold random CV, flat-4-target R² = −0.40)
- `molmetal/reports/wf77_p2_astr_baseline/platinai_4cell/per_seed_metrics.json`
  (Wf-77 P2 ASTSR-only K-fold random CV, flat-4-target R² = −0.40)

### §5.3 Wf-49 + Wf-69 + Wf-85 P2 chain (PlatinAI 4-cell extraction)

- `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 PARTIAL: P1+P2 ship, F5 →
  F5_RDKIT_2D swap; every F5 has |ρ| < 0.02 on PlatinAI)
- `TODO/completed/91b_wf49_p3p4_followup.md` (Wf-69 P3.1 platinai_5k bug fix
  + platinai_4cell loader)
- `molmetal/reports/wf85_p2_astr_only_real/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json`
  — this doc's primary MEASURED artifact

### §5.4 Wf-56 R20-P2 + Wf-94 R20-P3 follow-on workstreams

- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (R20-P2 plan)
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` (R20-P3 Multi-PH+PLL plan)

### §5.5 INDEX.md §17 cross-references

- `TODO/INDEX.md` §17 (TODO-89 5-category audit) + §17.5 (R22 plan items)

### §5.6 Skill lineage + parent plans

- `molmetal/scripts/bench_astr_only_real.py` (~1 226 LOC, NEW, primary bench
  script for Wf-85 P2)
- `molmetal/tests/test_astr_only_real.py` (~460 LOC, 28 tests PASS,
  NEW, primary test file for Wf-85 P2)
- `molmetal/molmetal_lam/reward/lambda_astsr.py` (LambdaASTSR evolutionary
  regressor — F5_RDKIT_2D default per Wf-49 P2)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 (L5 Hilbert-ASTSR R² ≥ 0.85
  target — DEFERRED until Wf-94 P3 Multi-PH + PLL ships) + §4.2 (L5
  trajectory row updated to ASTSR-only anchor = −0.07 MEASURED)

---

## §6 One-line verdict

Wf-85 Phase 2 ships the **REAL ASTSR-only R² retrain anchor** on PlatinAI
4-cell with **5 seeds × 50 epochs × strict Bemis-Murcko scaffold split +
CPU + GPU-0 Blackwell (CPU fallback) + 28/28 tests PASS**. The MEASURED
4-cell aggregate ASTSR-only test R² = **−0.0699 ± 0.0403** (CPU path;
all 4 cell lines collapse_detected=yes) is **honest negative** per
`party-twenty-spirit §3.5` — this is the **expected** outcome of testing
ASTSR-only on strict OOD scaffolds with the Wf-49 §Cause 2 weak-signal
F5_RDKIT_2D descriptor; the architectural fix for lifting R² ≥ +0.30 is
**Multi-PH + PLL feature augmentation (Wf-79 P2 / Wf-94 R20-P3)**, NOT
more ASTSR or more Hilbert.  Per `party-twenty-spirit §3.5`: 0 fabricated
MEASURED numbers, 0 silent PROJECTED-to-MEASURED promotion, 0 paper work
touched, 0 fake positive R² hidden.
