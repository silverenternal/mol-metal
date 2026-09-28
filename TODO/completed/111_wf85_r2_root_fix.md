# TODO-111 — Wf-85 R² = -6.71 honest-negative root cause fix

**Date**: 2026-09-27
**Status**: 🚀 **Active — fixes Hilbert-ASTR collapse + ASTSR dual-mode failure**
**Priority**: **P0 CRITICAL** (paper-blocking — combined R² is the central L4/L5/L7/L8/L11 Lambda claim)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Agent 2 Wf-85 root cause investigation (2026-09-27)
**Lambda gap(s)**: L4 (Hilbert primal-dual) + L5 (ASTSR evolutionary)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math`

---

## §0 Why this plan exists

Agent 2 root cause analysis (2026-09-27) confirmed the -6.71 R² (Wf-83 P1 5-fold random CV) and -1.50 R² (Wf-85 P3 strict-scaffold) failures decompose into:

| Root cause | Layer | Severity | Evidence |
|---|---|---|---|
| 12-d RDKit features \|ρ\|<0.02 on PlatinAI pIC50 | **DATA (primary)** | HIGH | Wf-49 §Cause 2 — pIC50 std=0.45, all 12 features have near-zero Pearson |
| Hilbert primal-dual diverges (no L2 regularization) | **MODEL (secondary)** | MED | `hilbert_primal_dual.py:601-613` plain MSE + no L2; train R² ≈ -21 (diverges during optimization) |
| ASTSR dual-mode collapse / explosion | **MODEL (secondary)** | MED | `lambda_astsr.py:931-938` mode A constant (R²≈-0.014) or mode B exploding AST (R²=-8 to -27) |
| `best_ast_lisp` logging bug | **LOGGING (minor)** | LOW | `bench_hilbert_astr_combined_real.py:846` checks `best_ast_lisp` attribute that doesn't exist on `LambdaASTSR` |
| **Scoring/pipeline/splits** | **OK** | NONE | R² formula correct, alpha-blend math correct, no leakage, strict-scaffold correctly applied |

**Scoring is NOT the bug.** The bug is upstream (data signal floor + optimization regularization).

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU + GPU-0 Blackwell |
| Wall-time | 1 week CPU + 2-3 days GPU |
| Output | patched `hilbert_primal_dual.py` + `lambda_astsr.py` + new feature augmentation module |
| Tests | ~25 new tests |

## §2 Solution path (3 fixes + 1 augmentation)

1. **Add L2 regularization to Hilbert primal-dual** — fix `hilbert_primal_dual.py:601-613` plain MSE objective. Add ridge term `0.5 * lambda * ||x||²` with adaptive `lambda` from inner-val.
2. **Fix ASTSR dual-mode guard** — `lambda_astsr.py:931-938` already has `collapse_detected` flag but only checks `pred_std/y_test_std < 0.10`. Add **scale-mismatch guard**: `|pred_std/y_test_std - 1.0| > 0.5` → reject.
3. **Fix `best_ast_lisp` logging** — change `hasattr(reg, "best_ast_lisp")` to `hasattr(reg, "best_ast_") and reg.best_ast_ is not None` then dump via `reg.best_ast_.to_string()`.
4. **(Longer) Multi-PH + PLL feature augmentation** — per TODO-94 R20-P3 ship (Wf-66). Already done in Wf-66 — apply to PlatinAI 4-cell as `F5_RDKIT_2D + Multi-PH-Fourier-32 + PLL-128` = 172-d feature stack.

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Hilbert L2 regularization | 1 day CPU | patched `hilbert_primal_dual.py` + 8 new tests | train R² ≥ -0.5 (no divergence) on probe |
| **P1.2** ASTSR scale-mismatch guard | 1 day CPU | patched `lambda_astsr.py` + 6 new tests | ASTSR R² std across seeds < 0.3 on PlatinAI 4-cell |
| **P1.3** `best_ast_lisp` logging fix | 2 hr CPU | patched `bench_hilbert_astr_combined_real.py:846` | log shows actual AST string |
| **P1.4** Multi-PH + PLL feature augmentation on PlatinAI 4-cell | 2 days GPU-0 | `bench_hilbert_astr_aug_features.py` (NEW) | combined R² > 0 on PlatinAI 4-cell |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: post-fix Hilbert train R² + post-fix ASTSR R² std + `best_ast_lisp` log content
- **PROJECTED**: combined R² > 0 on PlatinAI 4-cell after L2 + scale guard + feature augmentation

## §5 Honest framing

The honest baseline is **-1.50 to -6.71 R² (worse than constant-mean)**. Even with P1.1 + P1.2 + P1.3 fixes, the combined R² may still be negative — the 12-d feature signal floor is the structural ceiling. P1.4 (Multi-PH + PLL augmentation) is the only path to lift R² > 0 on PlatinAI 4-cell without changing the data.

**Crucial scoring-related note**: Agent 2 verified the R² formula and alpha-blend math are CORRECT. The negative R² is not a fabrication — it's the real result of the learned predictor being worse than `y.mean()` on these features. Fixing the model (P1.1-P1.3) and the features (P1.4) is the path to positive R².

## §6 Cross-references

- Agent 2 Wf-85 root cause report (2026-09-27)
- TODO/completed/91c_real_hilbert_astsr_p32.md (Wf-85 close-out — the -1.50 result)
- TODO/completed/91_wf49_r2_root_fix.md (Wf-49 PARTIAL — Cause 2 signal floor)
- TODO/completed/87_r2_debug.md (Wf-44 5-phase R² debug)
- TODO/pending/94_r20_p3_multiph_pll_feature.md (Wf-66 — already shipped Multi-PH + PLL)
- `molmetal/molmetal_lam/reward/hilbert_primal_dual.py:601-700` (the Hilbert code)
- `molmetal/molmetal_lam/reward/lambda_astsr.py:830-1052` (the ASTSR code)
- `molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py` (Wf-83 P1 driver)
- `molmetal/scripts/bench_hilbert_astr_combined_real.py` (Wf-85 P3 driver)
- `molmetal/scripts/bench_hilbert_astr_aug_features.py` (NEW for P1.4)
- LAMBDA_ROADMAP §4.2 L4/L5/L7/L8/L11 columns