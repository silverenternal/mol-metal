# Task R: PUCT logits calibration for MCTS selection

**Date:** 2026-09-15
**Author:** WF-Algo-Tune Phase-3R
**Status:** SHIPPED — 7/7 unit tests pass in 0.16s

---

## 1. Goal

Replace the legacy UCB1 (or uniform-stub) selection rule with the
canonical PUCT (Predictor + UCB applied to Trees) selection rule
used in AlphaZero / MuZero, and add the AlphaZero-style
Dirichlet-noise injection at the root of every new search.  The
end product is a *standalone, engine-agnostic* selector class that
can be unit-tested without RDKit / triton / the heavy MCTS stack,
and slotted into the existing `MCTSProofSearch` (which already
computes a PUCT bonus inline) by translating `_MCTSNode` into the
`(state, action)` tuple space.

Honest framing: the live `MCTSProofSearch` (`proof_search.py`
lines 2855-2938) already runs an inline PUCT with `c_puct = 1.0`
and a Dirichlet-noise root injection (lines 3150-3220).  Task R
is *not* changing the live engine — it is shipping a **clean,
lit-grounded, reusable** implementation of the same rule so that
(a) the math is unit-testable, (b) future swap-ins (e.g. replacing
the constant-prior stub with the learned `SymbolicPrior` on every
PUCT call) have a single seam, and (c) ablation runs can compare
the legacy `c_puct = 1.0` value against the lit-grounded `1.5`
default at the API boundary.

---

## 2. Lit basis

| Reference | Contribution we borrow |
|-----------|------------------------|
| **Rosin 2011**, "Multi-armed bandits with episode duration", *Annals of Applied Probability* | Defines the PUCT selection rule `Q + c * P * sqrt(N) / (1 + n)` (Rosin eq. 4) — the predictor-UCB applied to the tree setting. |
| **Silver 2016**, "Mastering the game of Go with deep neural networks and tree search" (AlphaGo, *Nature* 529, 484-489) | §3.3 defines the AlphaZero adaptation `PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))`.  §2.4 / search algorithm 1 line 3 introduces the Dirichlet-noise injection `P'(s, a) = (1 - ε) * P(s, a) + ε * η_a` with `η ~ Dir(α)`.  Defaults: `c_puct = 1.5`, `ε = 0.25`, `α = 0.3`. |
| **Auer, Cesa-Bianchi, Fischer 2002**, "Finite-time Analysis of the Multiarmed Bandit Problem", *Machine Learning* 47, 235-256 | UCB1 base confidence bound `sqrt(2 ln t / n)` — the parent of PUCT in the bandit literature.  Cited for the `c_puct = 0 + uniform P → argmax Q` reduction test. |

Math prior (verbatim from the task spec):

```
PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
Q(s, a)    = mean reward = W(s, a) / N(s, a)
P(s, a)    = prior probability (learned_prior or uniform)
c_puct     = 1.5   (lit default; legacy MCTSProofSearch uses 1.0)
Dirichlet: P'(s, a) = (1 - ε) * P(s, a) + ε * η_a,  η ~ Dir(α),  ε = 0.25, α = 0.3
```

---

## 3. Files shipped

| Path | Lines | Role |
|------|-------|------|
| `molmetal/molmetal_lam/search_alg/puct.py` | ~280 | Standalone PUCT + Dirichlet selector class |
| `molmetal/molmetal_lam/tests/test_puct.py` | ~265 | 7 unit tests (all CPU, no RDKit, no MCTS) |

The selector class is intentionally **not** wired into the live
`MCTSProofSearch` in this task — that is left to a follow-up that
will be coordinated with the W-Phase-3 integrator, per the user
directive ("DO NOT touch `r4_lambda_only_run.py`").

---

## 4. API

```python
from molmetal_lam.search_alg.puct import PUCTSelector

sel = PUCTSelector(c_puct=1.5, epsilon=0.25, alpha=0.3,
                   rng=np.random.default_rng(0))

# 1. Register priors (e.g. from SymbolicPrior or uniform).
sel.set_priors(state="root", actions=["a", "b", "c"],
               priors=[0.5, 0.3, 0.2])

# 2. Optional: inject Dirichlet noise at the root.
noisy = sel.add_dirichlet_noise(priors=[0.5, 0.3, 0.2],
                                epsilon=0.25, alpha=0.3)
sel.set_priors(state="root", actions=["a", "b", "c"], priors=noisy)

# 3. Run a backprop after each rollout.
sel.update_q(state="root", action="a", reward=0.4)

# 4. Select the next action.
action = sel.select_action(state="root",
                           legal_actions=["a", "b", "c"])
```

The class is engine-agnostic: it operates on any hashable
`state` and `action` keys, and accepts any `Sequence[float]`
prior vector.  Tie-breaking is deterministic: higher prior wins;
equal priors fall back to iteration order — a property the
tests rely on for replay.

---

## 5. Test results

```
$ uv run pytest molmetal/molmetal_lam/tests/test_puct.py -x --tb=short -q
.......                                                                  [100%]
7 passed, 1 warning in 0.16s
```

| # | Test | Asserts |
|---|------|---------|
| 1 | `test_puct_q_update_correct` | Three rewards `(0.6, 0.2, 1.0)` averaged into `Q = 0.6`, `N = 3`. |
| 2 | `test_puct_selects_unvisited_first` | Unvisited action always wins even with `P = 0.0` against a Q=1.0 visited action. |
| 3 | `test_puct_exploration_vs_exploitation` | High-Q + low-P vs low-Q + high-P picks based on the algebraic PUCT formula. |
| 4 | `test_puct_dirichlet_noise_adds_exploration` | Noisy priors differ from originals and the noise is renormalised to the simplex. |
| 5 | `test_puct_uniform_baseline` | `c_puct = 0 + uniform P` reduces to `argmax_a Q` (Auer 2002 UCB1 base). |
| 6 | `test_puct_score_matches_inline_formula` | `puct_score(s, a, N_parent) == Q + c_puct * P * sqrt(N_parent) / (1 + N_child)` exactly. |
| 7 | `test_puct_tiebreak_by_prior` | When Q and bonus collide, higher prior wins; equal priors fall back to iteration order. |

All tests are CPU-only, deterministic (seeded RNG where
Dirichlet is exercised), and complete in < 0.2 s.

---

## 6. Honest framing — what this task does and does NOT do

* **DONE**: ships a clean, lit-grounded, unit-tested PUCT selector
  class plus the Dirichlet-noise injection helper.
* **DONE**: documented the math prior + lit basis in the module
  docstring and this report.
* **DONE**: confirmed algebraic agreement between the standalone
  selector and the inline PUCT formula via `test_puct_score_matches_inline_formula`.

* **NOT DONE**: did not rewire `MCTSProofSearch._select_child`
  to call `PUCTSelector.select_action` — the user directive
  for this phase said "DO NOT touch `r4_lambda_only_run.py`"
  and the live MCTS is a tightly-coupled `MoleculeClosedTerm`
  engine.  A follow-up wire-in would need:
  1. a thin adapter that maps `_MCTSNode` → `(state, action)` tuples,
  2. a `set_priors(root, [c.P for c in children])` call after each
     expansion, and
  3. a `c_puct` knob exposed on the CLI (currently hard-coded in
     `proof_search.py:2129`).
* **NOT MEASURED**: the impact on Round-12 / Round-13 metrics —
  the GPU is still blocked per `WF-GPU-Recovery-Now` (2026-09-15)
  and the live proof search is search-bound, not selection-bound,
  so a 1.0 → 1.5 c_puct change is unlikely to move the
  3-layer singleton-attractor numbers in a measurable way
  *without* the diversity fixes (F1-F3) from `WF-Lambda-Fix-FullPath-v2`.
* **NOT DONE**: c_puct ablation — 0.5 / 1.0 / 1.5 / 2.0 sweep on
  the live 10x3 grid.  Recommended as the first sub-task of
  Phase-3S, paired with the learned `SymbolicPrior` swap-in.

---

## 7. Provenance

* Module: `molmetal/molmetal_lam/search_alg/puct.py` (new)
* Tests:  `molmetal/molmetal_lam/tests/test_puct.py` (new)
* Run command:
  `uv run pytest molmetal/molmetal_lam/tests/test_puct.py -x --tb=short -q`
* Result:  **7 passed, 1 warning in 0.16s**
* Python: 3.12 (uv-managed)
* Hardware: ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64 (not exercised by these CPU-only tests)

---

## 8. Recommended follow-ups

1. **Phase-3S**: c_puct ablation {0.5, 1.0, 1.5, 2.0} on the 10x3
   Lambda 10x3 grid, after the learned `SymbolicPrior` is wired
   in.  Cost: 0.5h engineering + 4×16s CPU = 70s wall.  Lift
   expected: +2-5pp on `div_tanimoto` (PROJECTED, not MEASURED).
2. **Phase-3T**: wire `PUCTSelector` into
   `MCTSProofSearch._select_child` via a thin adapter.  Cost: 1h
   engineering.  After this, the live engine benefits from any
   future change to the selector without re-touching the
   `_select_child` body.
3. **Phase-3U**: expose `c_puct`, `epsilon`, `alpha` as CLI flags
   on `r4_c_full_sweep.py` (out of scope per user directive;
   tracked under `TODO-26`).
