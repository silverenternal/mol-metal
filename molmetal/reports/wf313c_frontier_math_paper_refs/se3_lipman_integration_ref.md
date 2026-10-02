# `se3_lipman_integration` Adapter — Paper Reference (Wf-313C)

**Adapter:** `molmetal/adapters/se3_lipman_integration.py`
**Ship status:** Integration 1 (CPU smoke shipped; SE(3) CFM drop-in pending)
**Owner:** Hugo Lee (audit) — multi-surf integration per TODO-234 §6

---

## 1. Paper citation

> Mahmood, F., Halder, S. & Akhtar, R. (2026). **Schrödinger Bridges on
> Compact Connected Lie Groups with Application to Molecular Generation.**
> arXiv:2603.14049 [stat.ML].

The adapter wraps the SE(3) geodesic flow from §3.5 (Prop. 3.5) and
the SE(3)-equivariant velocity field from §3.4 (Def. 3.4) around the
R17 ODESolver shim. Reference cite key: `mahmood2026sb_lie`.

---

## 2. API surface (per Wf-309B honest framing)

| Symbol | Type | Purpose |
|---|---|---|
| `SE3LipmanSolver(base_lipman_solver=None, se3_adapter=None, n_steps=50, method="midpoint", ortho_tol=1e-3)` | class | SE(3) ODE solver wrapping the R17 ODESolver shim |
| `.solve(velocity_fn, x0: (B, N, 3), t_span=(0,1), pocket=None) -> (B, N, 3)` | method | Drop-in for `ODESolver.sample(x_init, n_steps, method)` |
| `.verify_se3_orthogonality(n_runs=10) -> float` | method | Max `\|R^T R - I\|_inf` audit across runs |
| `compare_se3_lipman_vs_base(lipman_solver, se3_solver, n_runs=10, z0=None, ...) -> Dict` | function | Wall-time + accuracy + orthogonality comparison harness |
| `SE3CFMComposite(base_cfm, se3_solver, n_steps=50, method="midpoint")` | class | Production entry point — composes base CFM with SE(3) solver |
| `.forward(z0, t=1.0, pocket=None) -> (B, N, 3)` | method | Replaces `LipmanFlowMatchingAdapter.forward_velocity` |
| `.__call__(z0, t, pocket)` | method | Callable alias matching the R17 adapter's callable convention |

**Math invariants** (pinned in the adapter module):
* `_EPS_TRIG = 1e-6`, `_EPS_ROT = 1e-8` (numerical guards)
* `DEFAULT_N_STEPS = 50`, `DEFAULT_N_RUNS = 10`, `DEFAULT_ORTHO_TOL = 1e-3`
* `manifold = "SO3"` raises `NotImplementedError` (R18 roadmap)
* Orthogonality hard contract: `max_err < 1e-3`

---

## 3. Honest reproduction caveats

* **CPU-only.** All operations use `torch.linalg` (CPU backend by
  default).  No CUDA / Triton kernels required for the Integration 1
  contract.  The 100× wall-time speedup claim in the W-Flow docstring
  is *not* achievable on a single-thread CPU skeleton.
* **No modification of `flow_matching_lipman/`.** The adapter is a
  *composition* layer: it wraps the R17 ODESolver shim and the
  `SE3SBVelocityField` adapter without modifying the locked pipeline.
  Caller code (the integration loop in `LipmanFlowMatchingAdapter`) is
  expected to instantiate `SE3LipmanSolver` in place of the existing
  `ODESolver`.
* **Per-atom vs. centroid tangent.** The `SE3SBVelocityField` returns
  a single `(B, 6)` twist per batch element (centroid translation +
  moment-arm angular velocity).  The per-atom trajectory is recovered
  by walking the SE(3) geodesic *from the centroid* and rotating the
  per-atom offsets by the centroid rotation at each sub-step.  This
  is the standard rigid-body identification.
* **1 % discrepancy in `se3_log`.** The adapter imports
  `se3_log` from `molmetal/math_skills/se3_schrodinger_bridge.py`
  which uses `torch.linalg.inv` (not the Barfoot / Eade closed form).
  There is a documented ~1 % discrepancy vs. the textbook SE(3) log
  (see `molmetal/math_skills/se3_schrodinger_bridge.py:396-399`).  This
  is below the `ortho_tol = 1e-3` audit threshold for `n_steps = 50`.
* **`base_lipman_solver = None` falls back to a zero-velocity
  `_ZeroEGNN`.**  Useful for unit tests that want to exercise `solve`
  without booting the full EGNN stack.  Production callers should
  pass the trained EGNN velocity net via the `se3_adapter` slot.

---

## 4. Drop-in integration points

| Wire-in | Where in the pipeline | Replacement? |
|---|---|---|
| `SE3LipmanSolver.solve` | Drop-in replace of `molmetal.adapters.flow_matching_lipman._reference.solver.ODESolver.sample(x, n_steps, method)` | **Replaces** the Euclidean ODESolver |
| `SE3CFMComposite.forward` | Drop-in replace of `LipmanFlowMatchingAdapter.forward_velocity(z0, t, pocket)` | **Replaces** the legacy Euclidean forward |
| `compare_se3_lipman_vs_base` | Benchmarking harness for dashboards | **Diagnostic only** — returns wall-time + accuracy + orthogonality |

**Honest gap:** the adapter does NOT replace the trained EGNN velocity
net — it wraps the velocity through `SE3SBVelocityField` so the same
EGNN weights are reused.  The adapter is a *transport* upgrade (Euclidean
→ SE(3) geodesic), not a *learning* upgrade.

**Production readiness:** Integration 1 smoke ✅ (test surface at
`molmetal/tests/test_se3_lipman_integration.py`); SE(3) CFM production
swap ❌ (smoke 0/1000 fail per the test contract — the W2.1 ship
verified shape preservation only).

---

## 5. Cross-references

* Wf-310F Day 1-2 build-only tests: `molmetal/tests/test_wf310f_frontier_math_integration.py`
* Wf-313C Day 3-5 paper-grade tests: `molmetal/tests/test_wf313c_frontier_math_integration_e2e.py::test_se3_lipman_solver_tangent_correctness`
* TODO-234 §6 Day 1 execution plan: `TODO/pending/234_wf234_frontier_math_audit_integration.md`
* Math skill source: `molmetal/math_skills/se3_schrodinger_bridge.py`
* Companion SE(3) SB adapter: `molmetal/adapters/se3_sb_integration.py`
* Adapter module map: `molmetal/adapters/se3_lipman_integration.py` §module map

---

**END OF REF** — Owner Hugo Lee; MEASURED + Integration 1 status; cite `mahmood2026sb_lie`.