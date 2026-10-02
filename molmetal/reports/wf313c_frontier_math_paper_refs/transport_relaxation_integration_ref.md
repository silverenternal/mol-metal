# `transport_relaxation_integration` Adapter — Paper Reference (Wf-313C)

**Adapter:** `molmetal/adapters/transport_relaxation_integration.py`
**Ship status:** W2.2 (CPU smoke shipped; reward-head drop-in pending)
**Owner:** Hugo Lee (audit) — multi-surf integration per TODO-234 §6

---

## 1. Paper citation

> Jiang, X., Xu, Y. & Zhang, Z. (2026). **Schrödinger Bridge with
> Transport Relaxation.** arXiv:2602.08118.

The adapter implements the ε-relaxed endpoint KL penalty from §3 of
the paper. Reference cite key: `jiang_xu_zhang2026sb_transport_relaxation`.

Supporting citations:
* Dayan 1997 — *Theoretically Grounded Reward Shaping* (potential-based
  shaping form `F(s, s') = γ Φ(s') - Φ(s)`).
* McAllester 1999 — PAC-Bayes bound (the original F5 derivation).
* Yu / Yang / Ren / Zhao 2026 — Deflexpressor λ-AST hypothesis class.

---

## 2. API surface (per Wf-309B honest framing)

| Symbol | Type | Purpose |
|---|---|---|
| `RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)` | class | Soft endpoint OT penalty: `λ · KL(p_T ‖ q_T(ε))` |
| `.kl(trajectory: (B, T, D) or (B, D), energies: (B,) or (B, T)) -> Tensor` | method | Compute the non-negative KL scalar |
| `.forward(trajectory, energies) -> Tensor` | method | Compute `λ_relax · KL(p_T ‖ q_T(ε))` |
| `.endpoint_geometry(trajectory) -> Tensor` | method | Return the terminal slice `(B, D)` |
| `.empirical_distribution(energies) -> Tensor` | method | Return `q_T = softmax(-E / ε)` |
| `augment_learned_shaping(base_shaping_output, trajectory, energies, lambda_relax=1e-3, eps=0.1) -> Tensor` | function | Additive wrapper around `LearnedShaping` / ASTSR |
| `LambdaRelaxSweep(base_shaping=None, lambdas=(1e-4, 1e-3, 1e-2), eps=0.1)` | class | Grid-search sweep over `λ_relax` |
| `.best_lambda(val_trajectories, val_energies, val_features=None) -> float` | method | Return the λ that minimises the augmented reward |
| `.evaluate_grid(val_trajectories, val_energies, val_features=None) -> List[dict]` | method | Per-λ evaluation rows for logging |

**Math invariants** (pinned in the adapter module):
* `_EPS_EPS_FLOOR = 1e-6` (numerical floor for `ε` to keep the
  softmax well-conditioned)
* `_EPS_LAMBDA_FLOOR = 0.0` (relaxation weight must be ≥ 0)
* KL is **always ≥ 0** by Gibbs' inequality (test contract pinned by
  `test_relaxed_endpoint_kl_non_negative`)
* When `λ_relax = 0` the penalty is identically zero (test contract
  pinned by `test_relaxed_endpoint_zero_lambda`)
* When all energies are equal, the KL is exactly 0 (`q_T` is uniform)

---

## 3. Honest reproduction caveats

* **CPU-only.** All operations use `torch.log_softmax` + `torch.softmax`
  on the CPU backend.  No CUDA / Triton kernels required for the W2.2
  contract.
* **No modification of `learned_shaping.py` / `lambda_astsr.py`.** The
  adapter is *additive*: `augment_learned_shaping` consumes the output
  of the existing F5 / ASTSR shaping module and adds the relaxation
  penalty.  The Dayan 1997 potential-based shaping form
  `F(s, s') = γ Φ(s') - Φ(s)` is preserved up to an ε-dependent
  constant.
* **KL is fp32-stable.** The textbook `p_i (log p_i - log q_i)` form
  overflows on `log q_i` for sharp `q`.  The adapter uses the
  equivalent form `KL(p ‖ q) = -log B - (1/B) sum_i log q_i` which
  is numerically stable in fp32 for `B <= 4096`.
* **`trajectory` and `energies` shapes.** The adapter accepts
  `(B, T, D)` trajectory + `(B, T)` energies (time-averaged) or
  `(B, D)` trajectory + `(B,)` energies (terminal).  Both shapes
  are auto-detected.  NaN / ±inf energy entries are replaced with
  the batch mean so the softmax remains well-defined.
* **KL is *batch-level*, not per-trajectory.** The relaxation
  regularises the empirical terminal distribution across the
  minibatch, not individual trajectories.  Per-trajectory
  relaxation would require an additional grouping key.
* **`eps → ∞` lowers the KL.** As `ε` grows, `softmax(-E/ε)`
  approaches uniform and the KL collapses to 0.  Conversely,
  `ε → 0` sharpens `q_T` toward a delta on the lowest-energy
  trajectory and inflates the KL.  The sweep grid
  `λ ∈ {1e-4, 1e-3, 1e-2}` is the W2.2 spec's 1-decade grid.

---

## 4. Drop-in integration points

| Wire-in | Where in the pipeline | Replacement? |
|---|---|---|
| `RelaxedEndpointTerm.forward` | Adds `λ · KL(p_T ‖ q_T)` to the shaping term | **Augments** the existing F5 / ASTSR shaping output |
| `augment_learned_shaping` | Drop-in wrapper around `LearnedShaping.shape_reward` | **Augments** — additive only, base shaping unchanged |
| `LambdaRelaxSweep.best_lambda` | Hyper-parameter sweep harness | **Diagnostic only** — returns the optimal `λ` for the validation set |

**Honest gap:** the adapter does NOT replace the reward head — it
augments alongside.  W2.2 recommends wrapping the F5 / ASTSR shaping
output via `augment_learned_shaping` so the existing reward contract
(``reward' = reward_learned + λ · KL``) is preserved bit-exact at
`λ_relax = 0`.

**Production readiness:** W2.2 smoke ✅; reward-head production
wire-in ❌ (the F5 / ASTSR slot is not yet wrapped via
`augment_learned_shaping` in the training loop — left for R18-P3).

---

## 5. Cross-references

* Wf-310F Day 1-2 build-only tests: `molmetal/tests/test_wf310f_frontier_math_integration.py`
* Wf-313C Day 3-5 paper-grade tests: `molmetal/tests/test_wf313c_frontier_math_integration_e2e.py::test_transport_relaxation_endpoint_term`
* TODO-234 §6 Day 2-3 execution plan: `TODO/pending/234_wf234_frontier_math_audit_integration.md`
* Adapter module map: `molmetal/adapters/transport_relaxation_integration.py` §module map
* Companion reward modules: `molmetal/molmetal_lam/reward/learned_shaping.py`,
  `molmetal/molmetal_lam/reward/lambda_astsr.py`

---

**END OF REF** — Owner Hugo Lee; MEASURED + W2.2 status; cite `jiang_xu_zhang2026sb_transport_relaxation`.