# `se3_sb_integration` Adapter — Paper Reference (Wf-313C)

**Adapter:** `molmetal/adapters/se3_sb_integration.py`
**Ship status:** W2.1 (CPU smoke shipped; SE(3) SB decoder drop-in pending)
**Owner:** Hugo Lee (audit) — multi-surf integration per TODO-234 §6

---

## 1. Paper citation

> Mahmood, F., Halder, S. & Akhtar, R. (2026). **Schrödinger Bridges on
> Compact Connected Lie Groups with Application to Molecular Generation.**
> arXiv:2603.14049 [stat.ML].

The adapter implements the SE(3) Schrödinger-Bridge decoder from §4
(Th. 4.1) and the SE(3) geodesic decoder head from §3.5 (Prop. 3.5).
Reference cite key: `mahmood2026sb_lie`.

---

## 2. API surface (per Wf-309B honest framing)

| Symbol | Type | Purpose |
|---|---|---|
| `SE3SBVelocityField(base_velocity_net: nn.Module, manifold="SE3", eps=1e-6)` | `nn.Module` | Wraps EGNN → `(B, 6)` SE(3) twist |
| `.forward(x_T: (B, N, 3), t: Tensor, pocket=None) -> (B, 6)` | method | Per-atom Euclidean velocity → body-frame twist |
| `SE3SBDecoder(eps=0.1, n_iters=50, use_sinkhorn=True, seed=None)` | `nn.Module` | SE(3) geodesic + Sinkhorn SB coupling decoder head |
| `.decode(z0: (B, 4, 4), z1: (B, 4, 4), pocket=None, t_values=None) -> (B, n_t, 4, 4)` | method | Walks the SE(3) geodesic at each `t_value` |
| `.decode_at(z: (..., 4, 4), t: float) -> (..., 4, 4)` | method | Single-t decoder with self-coupling |
| `integrate_with_cfm(samples, velocity_field, n_steps=100, manifold="SE3", method="midpoint", eps=1e-6, t0=0.0, t1=1.0) -> (B, N, 3)` | function | Euler / midpoint SE(3) ODE integrator |

**Math invariants** (pinned in the adapter module):
* `_EPS_TRIG = 1e-6`, `_EPS_ROT = 1e-8` (numerical guards)
* `manifold = "SO3"` raises `NotImplementedError` (R18 roadmap)
* `use_sinkhorn = False` for bit-exact testing
* `n_steps = 100` matches the R17 CFM ODE solver default
* Pairwise SB cost = squared Euclidean distance in twist space (eq. 12)

---

## 3. Honest reproduction caveats

* **CPU-only.** All operations use `torch.linalg` (CPU backend by
  default).  No CUDA / Triton kernels required for the W2.1 contract.
* **No modification of `flow_matching_lipman/`.** The adapter is a
  *decoder head* replacement; the existing Euclidean lerp decoder
  ``X_t = (1 - t) X_0 + t X_1`` is replaced by the SE(3) geodesic.
  The R17 ODESolver and EGNN velocity net are untouched.
* **Centroid-only SE(3) twist.** `SE3SBVelocityField.forward` returns
  a single `(B, 6)` twist per batch element (centroid translation +
  moment-arm angular velocity).  The per-atom trajectory is recovered
  by walking the SE(3) geodesic *from the centroid* and rotating the
  per-atom offsets by the centroid rotation at each sub-step.
  Equivariance is per Definition 3.4 of `mahmood2026sb_lie`.
* **`use_sinkhorn = True` adds B² pairwise cost entries.** The SB
  coupling is computed via `LogSinkhornLoss` with a `(B, B)` cost
  matrix in twist space.  Inference cost is `O(B²)` per `t_value` —
  fine for inference (`B ≤ 64`), expensive for training (`B ≥ 128`).
  Callers that need training-time Sinkhorn should pre-compute the
  coupling once and reuse it across `t_values`.
* **Decoder adapts to handle `pocket` kwarg for signature parity**
  but the *decoding* is currently pocket-agnostic (the pocket signal
  enters through the velocity field, not the geodesic interpolation).
* **Greedy hard assignment** of `P.argmax(dim=-1)` recovers the SB
  plan in the `eps → 0` limit (Th. 4.1 of `mahmood2026sb_lie`).
  For `eps > 0` the hard assignment is an approximation to the
  soft plan.

---

## 4. Drop-in integration points

| Wire-in | Where in the pipeline | Replacement? |
|---|---|---|
| `SE3SBVelocityField.forward` | Drop-in replace of `EGNNVelocityField.forward(x, t, pocket)` | **Replaces** the Euclidean velocity field wrapper |
| `SE3SBDecoder.decode` | Drop-in replace of the Euclidean lerp decoder ``X_t = (1-t) X_0 + t X_1`` | **Replaces** the Euclidean decoder head |
| `integrate_with_cfm` | Drop-in wrapper for `flow_matching_lipman._reference.solver.ODESolver.sample` | **Replaces** the ODESolver shim (signature compatible) |

**Honest gap:** the adapter does NOT replace the trained EGNN velocity
net — it wraps the velocity through `SE3SBVelocityField` so the same
EGNN weights are reused.  The adapter is a *decoder transport* upgrade
(Euclidean lerp → SE(3) geodesic + SB coupling), not a *learning*
upgrade.

**Production readiness:** W2.1 smoke ✅ (25 tests in
`test_se3_sb_integration.py`); SE(3) SB production swap ❌ (smoke
0/1000 fail per the test contract — the W2.1 ship verified shape
preservation only).

---

## 5. Cross-references

* Wf-310F Day 1-2 build-only tests: `molmetal/tests/test_wf310f_frontier_math_integration.py`
* Wf-313C Day 3-5 paper-grade tests: `molmetal/tests/test_wf313c_frontier_math_integration_e2e.py::test_se3_sb_decoder_endpoint_distribution`
* TODO-234 §6 Day 1 execution plan: `TODO/pending/234_wf234_frontier_math_audit_integration.md`
* Math skill source: `molmetal/math_skills/se3_schrodinger_bridge.py`
* Companion SE(3) Lipman adapter: `molmetal/adapters/se3_lipman_integration.py`
* Adapter module map: `molmetal/adapters/se3_sb_integration.py` §module map

---

**END OF REF** — Owner Hugo Lee; MEASURED + W2.1 status; cite `mahmood2026sb_lie`.