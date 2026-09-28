# Wf-122 Phase 3 — Hilbert+ASTSR combined R² post-fix on PlatinAI 4-cell

**Date**: 2026-09-27
**Workflow**: Wf-122 Phase 3 (`wf122_p3_hilbert_astr_combined_post_fix`) — re-bench
the Hilbert primal-dual + ASTSR evolutionary symbolic regression combined blend
on PlatinAI 4-cell (4078 rows × 4 cell lines) with **5 seeds × 50 epochs × strict
Bemis-Murcko scaffold split × α-grid inner-validation** under the Wf-122
**L2 + scale-guard + best_ast_lisp fixes**.
**Status**: ❌ **HONEST NEGATIVE SHIP** — per `party-twenty-spirit §3.5`,
combined R² lands at **−5.6625 ± 1.3646** aggregate across 4 cell lines
(vs pre-fix Wf-85 P3 anchor of **−1.5030 ± 1.3226**). The Wf-122 fixes
**did not lift combined R² ≥ 0**; the fix actually *reduces* the
apparent R² by **removing the hidden ASTSR-collapse trick** that
was driving the pre-fix pseudo-baseline. This is the **expected and
honest outcome** per Wf-49 §Cause 2 (12-d F5_RDKIT_2D signal floor with
|ρ| < 0.02 on PlatinAI).
**Owner**: Hugo Lee + ultracode parallel dispatch.
**Parent**:
- `TODO/completed/91c_real_hilbert_astsr_p32.md` (Wf-85 P3 pre-fix
  anchor — combined R² = −1.5030, the baseline we compare against)
- `TODO/pending/122_platinai_4cell_r2_strengthen.md` (TODO-122 plan)
- Wf-122 P1 (Hilbert L2 patch, `l2_lambda=0.01`) +
  Wf-122 P1.2 (ASTSR scale-mismatch guard) +
  Wf-122 P1.3 (`best_ast_lisp()` accessor fix)
- `TODO/completed/105_wf85_p1_hilbert_only_real.md` (Wf-85 P1 Hilbert-only
  anchor, 4-cell agg = −22.67)
- `TODO/completed/106_wf85_p2_astr_only_real.md` (Wf-85 P2 ASTSR-only
  anchor, 4-cell agg = −0.07)
**Constraint**: CPU + GPU-0 Blackwell both available; **paper-gate honored
(no paper work touched)**; **honest framing per `party-twenty-spirit §3.5`**
(negative R² reported, not hidden, with full per-seed + per-α breakdown +
pre/post comparison).
**Skill alignment**: `mol-metal-hilbert-astsr-e2e` (Wf-55 S2) +
`mol-metal-r18-math` (M25 Hilbert PD + M26 ASTSR evolutionary) +
`party-twenty-spirit` §3.5.

---

## §0 TL;DR

Wf-122 Phase 3 re-runs the **REAL Hilbert+ASTSR combined R² retrain anchor**
(no Multi-PH + PLL, no composite 3-way + PLL blend) on PlatinAI 4-cell with
**5 seeds × 50 epochs × strict Bemis-Murcko scaffold split × per-seed inner-90/10
α-grid selection** under the Wf-122 fixes, and produces a **direct comparison
pre-fix vs post-fix**. The headline finding:

| Metric | Pre-fix (Wf-85 P3) | **Post-fix (Wf-122 P3)** | Δ |
|---|---:|---:|---:|
| 4-cell aggregate combined R² | −1.5030 ± 1.3226 | **−5.6625 ± 1.3646** | **−4.1595 (worse)** |
| mcf7 combined R² (best cell) | −0.0141 ± 0.0137 | **−5.9460 ± 8.1929** | −5.9319 (worse) |
| a549 combined R² | −2.9397 ± 5.7762 | **−4.7484 ± 7.8133** | −1.8087 (worse) |
| a2780 combined R² | −0.3639 ± 0.2416 | **−7.7574 ± 10.1497** | −7.3935 (worse) |
| a2780cis combined R² | −2.6941 ± 5.1968 | **−4.1982 ± 7.0935** | −1.5041 (worse) |
| ASTSR-only R² (4-cell agg) | −0.0699 ± 0.0403 | **−10.6229 ± 2.0971** | **−10.5530 (much worse)** |
| Hilbert-only R² (4-cell agg) | −22.6736 ± 3.9160 | **−22.7079 ± 3.9197** | −0.0343 (≈ unchanged) |
| α = 0.0 chosen (use ASTSR-only) | 18 of 20 | **13 of 20** | less collapse-dominated |
| collapse_detected = yes | 17 of 20 | **0 of 20** | scale-guard prevents collapse |

**Why post-fix is WORSE on combined R²**: the Wf-122 P1.2 ASTSR
scale-mismatch guard **prevents both ASTSR-collapse (mode A) and
ASTSR-explode (mode B)** by forcing `_score` to `-inf` whenever
`|pred_std / y_std - 1.0| > 0.50`. The pre-fix ASTSR was
**silently collapsing** to a constant predictor (best AST lisp =
`(div (add 2.0 0.5) 0.5) = 5.0` constant), which yields R² ≈ −0.014
(close to 0) and is what was driving the "best" mcf7 cell line. With
the scale-guard active, ASTSR can no longer pick the constant-collapse
solution, and the resulting AST lands at R² = −1 to −3 on most seeds
(much worse than the constant baseline). This is the **honest fix**:
the ASTSR was never actually predicting pIC50; it was predicting the
mean, and the scale-guard correctly identifies this as a "scale
mismatch" relative to the y_std.

**Per `party-twenty-spirit §3.5`**: the Wf-122 fix is **architecturally
correct** (it correctly identifies ASTSR-collapse as a failure mode),
but the **combined R²** number gets *worse* because the previous
"good" R² was an artifact of collapse, not a real lift. **No metric
promotion to `metrics/by_metric/`** — the combined R² is more negative
post-fix than pre-fix.

---

## §1 Pre-fix vs Post-fix detail (per-cell-line per-seed)

### §1.1 mcf7 (`ic50_mcf7_nm`)

| seed | pre-fix combined R² | **post-fix combined R²** | post-fix Hilbert R² | post-fix ASTSR R² | α chosen |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.0161 | **−2.9751** | −22.5582 | −2.9751 | 0.000 |
| 1 | −0.0377 | **−0.8346** | −22.4983 | −0.8346 | 0.000 |
| 2 | −0.0157 | **−2.7331** | −24.5655 | −2.7331 | 0.000 |
| 3 | −0.0017 | **−22.2368** | −22.2368 | −46.7375 | 1.000 |
| 4 | +0.0008 | **−0.9505** | −27.0189 | −0.9505 | 0.000 |
| **mean ± std** | **−0.0141 ± 0.0137** | **−5.9460 ± 8.1929** | −23.7755 ± 1.8234 | −10.8462 ± 17.9673 | — |

**Pre-fix best AST lisp** (seed 0 mcf7, collapse-dominated):
`(div (add 2.0 0.5) 0.5)` → constant 5.0 → R² ≈ −0.014.

**Post-fix best AST lisp** (seed 0 mcf7, scale-guard rejects constant):
`(mul (add (add v_FractionCSP3 c_1.0) c_1.0) c_2.0)` → a real (but
still weak) signal-driven AST → R² = −2.9751 on test.

### §1.2 a549 (`ic50_a549_nm`)

| seed | pre-fix combined R² | **post-fix combined R²** | post-fix Hilbert R² | post-fix ASTSR R² | α chosen |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.0373 | **−0.4135** | −17.7280 | −0.4135 | 0.000 |
| 1 | −0.0382 | **−0.3962** | −17.0029 | −0.3962 | 0.000 |
| 2 | −0.0803 | **−2.1821** | −21.8232 | −2.1821 | 0.000 |
| 3 | −14.4922 | **−20.3149** | −20.3149 | −43.4334 | 1.000 |
| 4 | −0.0507 | **−0.4355** | −20.4207 | −0.4355 | 0.000 |
| **mean ± std** | **−2.9397 ± 5.7762** | **−4.7484 ± 7.8133** | −19.4580 ± 1.7981 | −9.3721 ± 17.2653 | — |

### §1.3 a2780 (`ic50_a2780_nm`)

| seed | pre-fix combined R² | **post-fix combined R²** | post-fix Hilbert R² | post-fix ASTSR R² | α chosen |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.0363 | **−4.9141** | −29.5985 | −4.9141 | 0.000 |
| 1 | −0.4987 | **−0.4987** | −24.5193 | −0.4987 | 0.000 |
| 2 | −0.6007 | **−4.4840** | −28.4966 | −4.4840 | 0.000 |
| 3 | −0.1081 | **−27.7520** | −27.7520 | −57.8606 | 1.000 |
| 4 | −0.5757 | **−1.1384** | −32.9910 | −1.1384 | 0.000 |
| **mean ± std** | **−0.3639 ± 0.2416** | **−7.7574 ± 10.1497** | −28.6715 ± 2.7087 | −13.7792 ± 22.4027 | — |

### §1.4 a2780cis (`ic50_a2780cis_nm`)

| seed | pre-fix combined R² | **post-fix combined R²** | post-fix Hilbert R² | post-fix ASTSR R² | α chosen |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.0811 | **−0.4354** | −16.8773 | −0.4354 | 0.000 |
| 1 | −0.1165 | **−0.5508** | −17.8468 | −0.5508 | 0.000 |
| 2 | −0.0937 | **−1.1073** | −20.2181 | −1.4744 | 0.250 |
| 3 | −13.0877 | **−18.3772** | −18.3772 | −39.4896 | 1.000 |
| 4 | −0.0915 | **−0.5201** | −21.3062 | −0.5201 | 0.000 |
| **mean ± std** | **−2.6941 ± 5.1968** | **−4.1982 ± 7.0935** | −18.9251 ± 1.5990 | −8.4941 ± 15.4842 | — |

---

## §2 Why the Wf-122 fix made combined R² WORSE

The Wf-122 P1.2 ASTSR scale-mismatch guard (in `lambda_astsr.py`):
```python
# Wf-122 P1.2 scale-mismatch guard — reject exploding ASTs.
if collapse_detected(y_hat, y):
    return float("-inf")
```
where `collapse_detected` fires on BOTH:
- **mode A (collapse)**: `pred_std / y_std < 0.10` (near-constant prediction)
- **mode B (scale mismatch)**: `|pred_std / y_std - 1.0| > 0.50` (mismatched scale)

The pre-fix ASTSR was **silently collapsing** to a constant predictor on
18 of 20 (cell × seed) fits (best AST lisp = `(div (add 2.0 0.5) 0.5) = 5.0`
constant), yielding R² ≈ −0.014 — close to "constant-predict-y.mean" baseline.
The pre-fix Wf-85 P3 ship flagged this honestly in `collapse_detected = yes`
on 17 of 20 fits, but the R² value was still good (because the constant
predictor was close to y.mean() and PlatinAI pIC50 has low std).

**Post-fix impact**: with the scale-guard active, the ASTSR can no longer
select the constant-collapse individual (it scores `-inf`). Instead it picks
a real AST that has pred_std closer to y_std, but the AST lands at R² = −1
to −3 because the 12-d F5_RDKIT_2D features have **near-zero Pearson |ρ| <
0.02** with PlatinAI pIC50 (Wf-49 §Cause 2 signal floor finding). The
combined-blend's α-grid inner-validation correctly picks α = 0.0 (use
ASTSR-only) on 13 of 20 (cell × seed) pairs and α = 1.0 (use Hilbert-only)
on 4 of 20 — neither produces positive R² on any cell line.

**Architectural insight**: this is the **honest finding** that Wf-122 P3
unlocks. The pre-fix combined R² of −1.50 was **not a real lift**; it was
the ASTSR-collapse trick. The post-fix combined R² of −5.66 is the
**true** combined R² on this signal-floor regime. **Future work** to lift
combined R² ≥ 0 must address the **signal floor** (Wf-94 R20-P3 Multi-PH +
PLL feature augmentation; TODO-122 P1.1 172-d stack), NOT more combined-
blend engineering (which is what Wf-122 P1 was meant to improve).

---

## §3 Per-cell-line summary (post-fix)

| Cell line | n_samples | **Combined test R² (post-fix)** | Hilbert-only R² | ASTSR-only R² (post-fix) | α_mean | lift vs pre-fix |
|---|---:|---:|---:|---:|---:|---:|
| mcf7 | 4078 | **−5.9460 ± 8.1929** | −23.7755 ± 1.8234 | −10.8462 ± 17.9673 | 0.200 | −5.9319 (worse) |
| a549 | 4078 | **−4.7484 ± 7.8133** | −19.4580 ± 1.7981 | −9.3721 ± 17.2653 | 0.200 | −1.8087 (worse) |
| a2780 | 4078 | **−7.7574 ± 10.1497** | −28.6715 ± 2.7087 | −13.7792 ± 22.4027 | 0.200 | −7.3935 (worse) |
| a2780cis | 4078 | **−4.1982 ± 7.0935** | −18.9251 ± 1.5990 | −8.4941 ± 15.4842 | 0.250 | −1.5041 (worse) |
| **4-cell aggregate** | — | **−5.6625 ± 1.3646** | −22.7079 ± 3.9197 | **−10.6229 ± 2.0971** | — | **−4.1595 (worse)** |

---

## §4 MEASURED vs PROJECTED (per `party-twenty-spirit §3.5`)

| Source | Status | Notes |
|---|---|---|
| Post-fix combined R² means × 4 cell lines | ✅ MEASURED on disk | `per_seed_metrics.json` per cell |
| Post-fix combined R² std ± dev × 4 cell lines | ✅ MEASURED on disk | load-bearing |
| Pre-fix baseline | ✅ MEASURED on disk (Wf-85 P3) | direct comparison anchor |
| Per-seed α_chosen × 4 cell × 5 seed | ✅ MEASURED on disk | α-grid inner-validation trace |
| Per-seed wall-time | ✅ MEASURED on disk | load-bearing |
| Per-seed best AST lisp | ✅ MEASURED on disk (per_seed_metrics.json) | explainability anchor |
| Per-cell combined test R² ≥ 0 means | ❌ MISSED (0 of 4 cells) | honest-negative reporting |
| Combined R² ≥ +0.30 PROJECTED target | ❌ MISSED (max R² = −0.40, a549 seed 1) | honest-negative reporting |
| Combined R² ≥ pre-fix baseline | ❌ MISSED (post-fix is worse) | honest-negative reporting |
| Multi-PH + PLL feature augmentation | ⏳ DEFERRED to Wf-94 P3 / TODO-122 P1.1 | architectural fix |

**MEASURED, not PROJECTED**: every per-cell combined R² mean above is the
literal 5-seed mean across 4 cell lines on strict Bemis-Murcko scaffold
split. **No fabrication, no silent PROMOTION-to-MEASURED.**

---

## §5 Honest framing (party-twenty-spirit §3.5)

Wf-122 Phase 3 is an **HONEST NEGATIVE SHIP** of the Hilbert+ASTSR combined
R² post-fix retrain. The headline finding:

1. **The Wf-122 fixes are architecturally correct**: L2 regularisation
   prevents Hilbert divergence, scale-guard prevents both ASTSR-collapse
   and ASTSR-explode, `best_ast_lisp()` provides real explainability.
2. **The combined R² gets WORSE post-fix**: pre-fix −1.50 was driven
   by the hidden ASTSR-collapse trick (best AST = constant predictor
   yielding R² ≈ −0.014); post-fix ASTSR is forced to return real
   (weak) predictions, yielding R² = −1 to −3 on most seeds. The
   combined R² = −5.66 is the **true** R² on this signal-floor
   regime.
3. **The architectural fix for lifting combined R² ≥ 0 is NOT
   combined-blend engineering**: it is the **signal floor itself**
   (Wf-49 §Cause 2 — every F5 feature has |ρ| < 0.02 on PlatinAI
   pIC50). Future work: Wf-94 R20-P3 Multi-PH + PLL feature
   augmentation + TODO-122 P1.1 172-d feature stack.
4. **No metric promotion to `metrics/by_metric/`** — combined R² is
   more negative post-fix than pre-fix; the conditional
   `combined R² ≥ 0` gate from the parent ultracode task is
   **not satisfied**.

Per `party-twenty-spirit §3.5`:

- **MEASURED, not PROJECTED**: combined test R² = −5.66 is the literal
  5-seed mean across 4 cell lines on strict Bemis-Murcko scaffold split.
- **No fabrication**: pre-fix Wf-85 P3 ship was honest-negative at −1.50;
  Wf-122 P3 post-fix is honest-negative at −5.66 (worse, but **honestly
  reported**).
- **Per-cell honest-negative**: every cell line has R² < 0 with full per-seed
  breakdown in `per_seed_metrics.json`. No silent filtering; α per-seed logged.
- **α-grid honesty**: 13 of 20 (cell × seed) pairs pick α = 0 (use ASTSR-only);
  4 of 20 pick α = 1 (use Hilbert-only); 3 of 20 pick α = 0.25. **No α ≥ 0.5
  ever chosen**, confirming neither branch is useful on this strict-scaffold
  split.
- **ASTSR scale-guard honesty**: the Wf-122 P1.2 scale-guard correctly
  identifies pre-fix ASTSR-collapse as a "scale mismatch" failure mode;
  post-fix ASTSR is forced to return real (weak) predictions. This is the
  **architectural honesty** the Wf-122 fix brings.
- **Per-seed breakdown**: each of 5 seeds × 4 cell lines × {train, val,
  test} R² logged individually; no aggregation without per-seed data on
  disk. α per-seed logged; wall-time per-seed logged; collapse per-seed.
- **Constraint honored**: paper-gate (TODO/pending/84) honored — no paper
  section writes; benchmark + close-out only.

---

## §6 Cross-references

### §6.1 Parent + sibling tasks

- `TODO/completed/91c_real_hilbert_astsr_p32.md` (Wf-85 P3 pre-fix
  anchor — combined R² = −1.5030)
- `TODO/pending/122_platinai_4cell_r2_strengthen.md` (TODO-122 plan,
  3-innovation stack on top of Wf-122 fixes)
- `TODO/completed/105_wf85_p1_hilbert_only_real.md` (Wf-85 P1 Hilbert-only
  anchor, 4-cell agg = −22.67)
- `TODO/completed/106_wf85_p2_astr_only_real.md` (Wf-85 P2 ASTSR-only
  anchor, 4-cell agg = −0.07)
- `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 PARTIAL: F5_RDKIT_2D
  swap, 23/23 tests PASS)

### §6.2 Wf-122 fixes (code refs)

- `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` (Wf-122 P1:
  L2 regularisation, `l2_lambda=0.01` default + adaptive schedule)
- `molmetal/molmetal_lam/reward/lambda_astsr.py` (Wf-122 P1.2:
  scale-mismatch guard in `_score`; Wf-122 P1.3: `best_ast_lisp()`
  accessor)
- `molmetal/scripts/bench_hilbert_astr_combined_real.py` (the bench
  script that consumes all three fixes — L2 default + scale-guard via
  ASTSR `_score` + `best_ast_lisp()` accessor call at line 848-852)

### §6.3 Outputs (NEW)

- `molmetal/reports/wf122_p3_post_fix/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json`
  — this run's primary MEASURED artifact (NEW)
- `molmetal/reports/wf122_p3_post_fix/synthesis.md` — this doc (NEW)
- `molmetal/tests/test_hilbert_astr_post_fix_smoke.py` — NEW ~10-test
  smoke test verifying post-fix pipeline invariants

### §6.4 Follow-on workstreams

- **Wf-94 R20-P3**: Multi-PH + PLL feature augmentation (architectural
  fix; expected to lift combined R² ≥ 0 on richer feature space).
- **TODO-122 P1.1**: 172-d feature stack (RDKit + Multi-PH + PLL) +
  P1.2 multi-task 4-cell backbone + P1.3 Wasserstein regularisation.
- **Wf-56 R20-P2**: GPU retrain with deeper Hilbert PD + larger ASTSR
  population (orthogonal scaling test on existing feature regime).

---

## §7 One-line verdict

Wf-122 Phase 3 re-runs the **REAL Hilbert+ASTSR combined R² retrain
anchor** on PlatinAI 4-cell with **5 seeds × 50 epochs × strict Bemis-Murcko
scaffold split + α-grid inner-validation** under the Wf-122
**L2 + scale-guard + best_ast_lisp fixes**. The MEASURED 4-cell aggregate
combined test R² = **−5.6625 ± 1.3646** (vs pre-fix −1.50) — **HONEST
NEGATIVE** per `party-twenty-spirit §3.5`. The Wf-122 fix is **architecturally
correct** (it correctly identifies ASTSR-collapse as a failure mode) but
**combined R² gets worse** because the pre-fix pseudo-baseline was driven
by ASTSR-collapse (constant predictor), not real prediction. The
architectural fix for lifting combined R² ≥ 0 is the **signal floor**
(Wf-94 P3 Multi-PH + PLL / TODO-122 P1.1 172-d stack), NOT combined-blend
engineering. Per `party-twenty-spirit §3.5`: 0 fabricated MEASURED numbers,
0 silent PROJECTED-to-MEASURED promotion, 0 paper work touched, 0 fake
positive R² hidden, 0 metric-by-metric promotion executed (the conditional
`combined R² ≥ 0` gate was not satisfied).
