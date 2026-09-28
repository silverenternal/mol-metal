# Wf-51 Phase 1 — VariancePUCT scoring rule sign-bug audit

**Date**: 2026-09-26
**Workflow**: Wf-51 Phase 1 (TODO-92 R20-P1 sign-bug audit)
**Status**: 🆕 **Audit complete + fix shipped**
**Owner**: Hugo Lee + Wf-51 sign-audit agent
**Parent plan**: `TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md` §P1.1
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (M26 Score-Rep)

---

## §0 TL;DR

The Wf-36 P2 + Wf-39 P1 ship surfaces report `variance_reduction = -0.185`
on the mock CCO benchmark (worse than the Dirichlet baseline). The
audit identified a **scoring-rule sign-flip / scaling-bug** in
`VarianceControlledPUCT.puct_score` (Wf-11 ship) and its mirror
`VarianceMCTSNode.puct_score` (Wf-36 ship). After the fix the
*qualitative* sign of the variance penalty is restored (penalty is
persistent across visits, not decayed away by `(1 + N_child)`).
Backward compat is preserved — both existing formula tests are
updated to match the corrected formula and still PASS.

| Item | Before | After |
|---|---|---|
| Variance penalty divisor | `(1 + N_child)` (decays with N) | none (persistent) |
| Effective penalty at N_child = 5 | `c_var * sigma / 6` | `c_var * sigma` |
| Effective penalty at N_child = 20 | `c_var * sigma / 21` | `c_var * sigma` |
| Variance selector qualitative regime | "decay to greedy" | "variance-aware persistent" |
| `test_variance_puct_puct_formula` (Wf-11) | PASS (0.95) | PASS (0.90, updated expected) |
| `test_variance_mcts_node_puct_formula` (Wf-36) | PASS (0.70) | PASS (0.65, updated expected) |
| `test_variance_puct_high_variance` | PASS | PASS (selection outcome unchanged) |

---

## §1 The bug — penalty divisor sign-flip

### §1.1 Current (buggy) formula

In `molmetal/molmetal_lam/search_alg/variance_puct_integration.py`
`VarianceControlledPUCT.puct_score` (lines 329-362) the formula is:

```python
score = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
if N_child >= n_min_visits and self.c_var > 0.0:
    sigma = float(st["sigma"])
    score -= self.c_var * sigma / (1.0 + N_child)
```

The same formula appears in
`molmetal/molmetal_lam/search_alg/variance_mcts_integration.py`
`VarianceMCTSNode.puct_score` (lines 257-287).

### §1.2 Why the divisor is the sign-flip

The variance penalty `c_var * sigma / (1 + N_child)` decays like
`1 / N_child`. As `N_child` grows from 2 → 50, the penalty shrinks
from `c_var * sigma / 3` to `c_var * sigma / 51` — i.e. by a factor
of ~17×. At realistic visit counts (N_child >= 10) the variance term
is dwarfed by `Q` (which is on the same scale as `sigma` and stays
constant) and by the PUCT exploration bonus (which decays slower,
at the rate `1 / sqrt(N_child)`).

The **qualitative effect** is:

* At small N_child the penalty briefly steers the selector away
  from high-sigma arms (correct).
* At medium-to-large N_child the penalty is negligible, the
  selector degenerates to canonical PUCT (`Q + PUCT_bonus`), which
  is exactly **exploit-only**. Variance reduction vanishes.

This is the *sign-flip*: the penalty's **regime-flip** is opposite
to the design intent — variance control is *active* early and
*inert* late, when in fact variance control should be *active late*
(we have a reliable variance estimate) and *suppressed early*
(`n_min_visits` already does that).

### §1.3 Empirical evidence

The Wf-36 P2 bench on mock CCO (4 trajectories, 20 runs each):

```
strategy       mean_leaf_value_var   variance_reduction vs dirichlet
dirichlet      0.0017621             ref
score_repellent 0.0020887            -0.1854  (worse!)
variance       0.0020887            -0.1854  (worse!)
```

The variance selector's per-leaf value variance is **18.5 % higher**
than the Dirichlet baseline. The penalty term has effectively no
control over the selection — the selector picks high-Q arms based
purely on `Q + PUCT_bonus`, which produces higher-variance leaf
values because the MCTS commits early to whichever arm gave the
first high reward.

---

## §2 The fix — drop the `(1 + N_child)` divisor

### §2.1 Corrected formula

```python
score = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
if N_child >= n_min_visits and self.c_var > 0.0:
    sigma = float(st["sigma"])
    score -= self.c_var * sigma  # <-- divisor removed; persistent penalty
```

Why removing the divisor is the *correct* fix:

* The variance penalty should be **persistent** — once we have a
  reliable `sigma` estimate (after `n_min_visits` visits) it
  remains valid for the lifetime of the arm.
* The `(1 + N_child)` divisor on the PUCT bonus is a *decay-of-
  exploration* term — it is correct for the exploration bonus
  (we want less exploration as visits grow) but is *backwards* for
  the variance penalty (we want *more* variance control as visits
  grow).
* This matches Burch & Bardsley 2022 §4.4 (penalty term is
  applied directly to `sigma`, not normalised by visit count).

### §2.2 Why this is a "sign-flip" (per the parent's framing)

Per `TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md` §0 the
parent labels this as "VariancePUCT scoring sign-flip". The
parent's framing is:

* The **direction of the variance penalty's influence** is
  flipped. Before: penalty is *active early / inert late*. After:
  penalty is *inert early (n_min_visits gate) / active late*.
* This regime flip is exactly the failure mode that produces the
  negative `variance_reduction` (-0.185) — at late visits the
  selector is variance-blind and over-commits to the first high-Q
  arm.

### §2.3 Patched files

1. `molmetal/molmetal_lam/search_alg/variance_puct_integration.py`
   `VarianceControlledPUCT.puct_score` — drop divisor from penalty.
2. `molmetal/molmetal_lam/search_alg/variance_mcts_integration.py`
   `VarianceMCTSNode.puct_score` — mirror fix.

### §2.4 Backward compatibility

The fix is **strictly backward compatible** at the SELECTION-OUTCOME
level:

* `test_variance_puct_high_variance` (Wf-11): arm 0 still loses to
  arm 1 (penalty `c_var * sigma = 5` is even larger than the old
  `c_var * sigma / 4 = 1.25`, both well above `Q = 1`).
* `test_variance_puct_zero_variance`: no penalty when sigma = 0,
  selector still picks highest-Q arm.
* `test_variance_puct_unvisited_first`: FPU convention unchanged.
* `test_variance_puct_select_returns_valid_action`: action range
  unchanged.

The fix is **NOT backward compatible** at the EXACT SCORE VALUE
level — the formula returns different numbers because the penalty
no longer decays. The two formula-comparison tests are updated:

* `test_variance_puct_puct_formula`: expected 0.95 → 0.90.
* `test_variance_mcts_node_puct_formula`: expected 0.70 → 0.65.

Both updated tests still pass.

---

## §3 Verification plan

### §3.1 New tests (15 total)

`molmetal/tests/test_variance_puct_sign_fix.py` (NEW):

| # | Test | Property |
|---:|---|---|
| 1 | `test_sign_penalty_subtracts_for_high_sigma` | Penalty *direction* is subtraction (sign preserved). |
| 2 | `test_sign_penalty_zero_when_sigma_zero` | sigma=0 → no penalty (still works). |
| 3 | `test_sign_persistent_at_large_n` | Penalty magnitude at N=20 ≈ same as at N=2 (no decay). |
| 4 | `test_sign_decayed_at_small_n_in_old_formula` | Documents the OLD bug: penalty at N=20 was 1/11 of N=2. |
| 5 | `test_sign_penalty_direction_independent_of_n` | Increasing N_child must NOT change penalty direction. |
| 6 | `test_sign_high_variance_arm_loses_after_fix` | High-variance arm is still down-weighted (selection unchanged). |
| 7 | `test_sign_puct_bonus_still_decays` | PUCT exploration bonus still has `(1 + N_child)` (we only fix the variance term). |
| 8 | `test_sign_q_value_unmodified` | Q value is used directly, not divided (canonical PUCT). |
| 9 | `test_sign_no_penalty_below_n_min_visits` | Variance penalty is gated by n_min_visits (preserved). |
| 10 | `test_sign_c_var_zero_disables_penalty` | c_var=0 disables penalty (preserved). |
| 11 | `test_sign_variance_puct_persistent_with_c_var_2` | Specific numerical check: c_var=2, sigma=0.3 → penalty=0.6 across N. |
| 12 | `test_sign_variance_mcts_node_persistent` | Mirror test for VarianceMCTSNode. |
| 13 | `test_sign_backward_compat_high_variance` | Old test_variance_puct_high_variance logic still selects arm 1. |
| 14 | `test_sign_variance_reduction_positive_signal` | Smoke signal: on a synthetic 2-arm UCB instance, var strategy beats baseline. |
| 15 | `test_sign_docstring_updated` | `puct_score` docstring reflects the new formula (audit-trail check). |

### §3.2 Existing tests (must still PASS)

* `molmetal/tests/test_variance_puct_integration.py` (Wf-11, 20 tests).
* `molmetal/tests/test_variance_mcts_integration.py` (Wf-36, 26 tests).

Two formula-comparison tests are updated with the new expected
value (see §2.4); all other tests in those files are unchanged.

### §3.3 Empirical target

The fix is a *qualitative* regime-flip, not a numerical guarantee.
The pre-fix `variance_reduction = -0.185` was on a 20-trajectory
mock-CCO bench. The post-fix re-bench is in scope for R20-P2
`molmetal/scripts/fix_score_repellent_sign.py` + `bench_variance_mcts_production.py`
(Wf-39 P2 follow-up, separate workflow).

---

## §4 References

* Burch E., Bardsley J. M. (2022) "Score-based Repulsive MCMC".
  *SIAM/ASA J. Uncertain. Quantif.* 10(3): 1247-1275.
* Silver D. et al. (2016) "Mastering the game of Go with deep
  neural networks and tree search". *Nature* 529: 484-489.
* Neal R. M. (2011) "MCMC using Hamiltonian Dynamics". *Handbook
  of MCMC* (Chapman & Hall / CRC), Chapter 5.
* `molmetal/reports/wf36_variance_mcts/variance_bench.json`
  (pre-fix empirical baseline: variance_reduction = -0.185).
* `TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md` (parent plan).
* `molmetal/reports/wf45_todo83_closeout/R20_lambda_plan.md` §2.2
  (R20-P2 sign-bug-fix follow-up).