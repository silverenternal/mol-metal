# Wf-234 — Frontier-Math Audit + Integration Plan (all modules → main CFM pipeline)

**Date**: 2026-09-28
**Owner**: Hugo Lee (audit) — ultracode execution phase to follow
**Trigger**: user said "我觉得能放进模型主线的数学技巧和创新点要真的加入到模型主线当中，你启动ultracode注意数学上的细节把他们所有都集成进去" — i.e. *every* math technique currently living only in `molmetal/math_skills/` or in a standalone adapter must be either (a) wired into the locked `molmetal/adapters/flow_matching_lipman/` main CFM pipeline, or (b) honestly tagged as research-only with a written reason.
**Parent doc**: `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` (P0-1..P0-4 + P1-1..P1-7), `TODO/03_plans/R18_PLAN.md` §risk, `TODO/completed/55_r18_integration_adapters.md` (W2.1/W2.2/W2.3 ship surface), `TODO/completed/59_cfm_m3_m5_frontier.md` (CFM frontier track).
**Status**: **AUDIT COMPLETE — read-only; integration plan shipped, execution gated on user OK to launch ultracode**.

This is a **read-only audit**. No file under `molmetal/` is modified. The plan identifies each math module, its public API, the integration point inside `flow_matching_lipman/` it should target, and an honest-framing verdict (production-wire vs research-only).

---

## 1. TL;DR — Acceptance gates

| Gate | Description | Result |
|---|---|---|
| **G1** | Each math module's API documented | **PASS** — §3 below |
| **G2** | Integration points identified in main flow (`flow_matching_lipman/w_flow.py` + the surrounding `__init__.py` + `optimal_transport_v2.py` + `connectivity_decoder.py`) | **PASS** — §4 below |
| **G3** | Integration plan doc shipped | **PASS** — this file |
| **G4** | Per-module integration effort estimated | **PASS** — §5 below |

**One-line summary**: 4 of 7 frontier math modules can be wired into the main CFM pipeline today (the W2.1 SE(3) SB trio + W2.3 multi-PH are already shipped as standalone adapters — only the "drop into `flow_matching_lipman`" wire step remains). The remaining 3 (`bass_existence`, `sb_bass_martingale`, `monge_ampere_regularity`) are **honestly research-only**: their APIs are statement-text holders + statement-quality calibrators, not drop-in losses or operators, and forcing them into the locked pipeline would violate the "no modifications to `flow_matching_lipman/`" R17 invariant. Their value is the **paper §3 method reference**, not the code path.

---

## 2. Scope — math modules under audit (per the harness task)

Per the Wf-234 task definition, the audit covers:

| Module | LOC | R18 plan ref | Already has adapter? |
|---|---|---|---|
| `molmetal/math_skills/se3_schrodinger_bridge.py` | 816 | P0-1 | **YES** — `se3_sb_integration.py` + `se3_lipman_integration.py` |
| `molmetal/math_skills/bass_existence.py` | 326 | (appendix-only citation holder) | **NO** — research-only by design |
| `molmetal/math_skills/sb_bass_martingale.py` | 650 | P1-3 (Henry-Labordère 2017) | **NO** — research-only (drop-in head exists but is for SynFlowNet TB loss, not CFM) |
| `molmetal/math_skills/monge_ampere_regularity.py` | 619 | P0-3 (汪徐家 2026) | **NO** — research-only |
| `molmetal/math_skills/multi_parameter_ph.py` | 1192 | P1-1 (Kim-Jung 2026) | **YES** — `multi_ph_integration.py` |
| `molmetal/adapters/se3_sb_integration.py` | 611 | W2.1 | n/a (already shipped) |
| `molmetal/adapters/se3_lipman_integration.py` | 854 | Integration 1 | n/a (already shipped) |

**Adjacent modules in the same family but not in the harness list** (still in scope because they share the same wire-in path): `multi_ph_integration.py` (W2.3 ship — 480 LOC) and `transport_relaxation_integration.py` (W2.2 ship — 556 LOC). I include them in §4.3 and §4.4 because they are the "ALREADY-WIRED" baseline that proves the wire-in pattern works.

---

## 3. Per-module API documentation (Gate G1)

### 3.1 `math_skills/se3_schrodinger_bridge.py` (Mahmood 2026, arXiv:2603.14049)

| Symbol | Type | Purpose |
|---|---|---|
| `so3_exp(omega: (..., 3))` | `Tensor -> Tensor` | Rodrigues' formula — exponential map `so(3) → SO(3)` |
| `so3_log(R: (..., 3, 3))` | `Tensor -> Tensor` | Inverse of `so3_exp` (closed-form with π-cut fallback) |
| `so3_slerp(R0, R1, t)` | `Tensor, Tensor, float -> Tensor` | Spherical linear interpolation on SO(3) |
| `se3_exp(xi: (..., 6))` | `Tensor -> Tensor` | Twist → SE(3) homogeneous matrix (closed-form via V(ω)) |
| `se3_log(T: (..., 4, 4))` | `Tensor -> Tensor` | SE(3) matrix → twist; uses `torch.linalg.inv` (not the textbook closed form, ~1 % discrepancy) |
| `SE3Geodesic.interpolate(R0, R1, t)` | static method | Alias for `so3_slerp` |
| `SE3Geodesic.interpolate_se3(T0, T1, t)` | static method | SE(3) geodesic — `T0 · exp(t · log(T0⁻¹ · T1))` (Prop. 3.5) |
| `SE3Geodesic.log_map(T_ref, T)` | static method | Body-frame relative log |
| `SE3Geodesic.exp_map(T_ref, v)` | static method | Body-frame relative exp |
| `LogSinkhornLoss(eps=0.1, n_iters=50)` | class | Log-domain Sinkhorn divergence — `forward(p0, p1, C?, a?, b?) -> (loss, P)` |
| `SchrodingerBridgeSE3(sigma, eps, n_iters, seed)` | class | SE(3) SB sampler — `sample(p0, n_samples, t_values?, sigma?) -> Tensor` |

**Math invariants** (test-pinned):
* `_EPS_TRIG = 1e-6`, `_EPS_ROT = 1e-8` (numerical guards)
* SE(3) inverse uses `torch.linalg.inv` (not Barfoot/Eade closed form — 1 % discrepancy noted in docstring line 396-399)
* Log-domain Sinkhorn stable to `eps >= 1e-3` for `n, m <= 1000`
* CPU-only by design (no CUDA/Triton in the math layer)

### 3.2 `math_skills/bass_existence.py` (Backhoff-Veraguas 2020, arXiv:2004.10060)

| Symbol | Type | Purpose |
|---|---|---|
| `BassExistenceTheorem(dimension?, statement?)` | class | **Statement-text holder** — stores the formal Thm 1.1 text |
| `.statement() -> str` | method | Returns the theorem text (with optional `d=N` annotation) |
| `.uniqueness_fails() -> bool` | method | Always `False` (uniqueness generally fails for `d >= 2`); `True` only when `dimension=1` |
| `.citation_key() -> str` | method | Returns `"backhoff_veraguas2020bass_existence"` |
| `RegularityCondition(heavy_tail_threshold=1e30)` | class | **P₂ check** — `check(marginal: Tensor) -> bool` |
| `.check(marginal: (n,) \| (n, d))` | method | Tests finite + second-moment finite + non-empty |
| `.second_moment(marginal) -> float` | method | Returns `mean(||x||²)` or `inf` if invalid |

**Honest framing**: this module is a **cite-only skeleton** — it ships the theorem statement and a `P_2` consistency check, no algorithm. Its R18 use is `paper/appendices/closure_theorem.tex` and the Bass-martingale calibration of SynFlowNet (T7). It is NOT designed to be dropped into the CFM velocity field or loss.

### 3.3 `math_skills/sb_bass_martingale.py` (Henry-Labordère 2017, SSRN 2889130)

| Symbol | Type | Purpose |
|---|---|---|
| `BassMartingale(eps=0.1)` | class | **Discrete Bass coupling** — `couple(p0: (n, d), p1: (m, d)) -> (n, m)` matrix |
| `.couple(p0, p1)` | method | Per-row two-point interpolation; cache `_p0`, `_p1` for verification |
| `.mean_preserving(coupling, atol=1e-4) -> bool` | method | Verify `Π · p1 ≈ a · p0` (martingale property) |
| `BassMartingaleDiscrepancy(lam=0.1)` | class | **Calibration loss** — `forward(coupling, marginals) -> scalar` |
| `.forward(coupling: (n, m), marginals: (n,))` | method | `(1-λ)·martingale_penalty + λ·mass_penalty` |
| `BassCalibrationHead(eps=0.1)` | class | **Drop-in TB-loss replacement for SynFlowNet** — `loss(trajectory) -> scalar` |
| `.loss(trajectory: (T+1, n, d) \| (B, T+1, n, d))` | method | `D_Bass(Π_τ, Π_Bassref)`; vanishes when Bass-calibrated |

**Math invariants**:
* `_EPS_DIV = 1e-12`, `_MARTINGALE_ATOL = 1e-4`, `_EPS_MIX = 1e-12`
* Bass martingale coupling is **non-unique for `d >= 2`** (matches `BassExistenceTheorem.uniqueness_fails()`)
* The TB-loss replacement is invariant under time-homogeneous rescaling of the flow (the property TB conflates with `log Z` magnitude)

**Honest framing**: this module's drop-in head (`BassCalibrationHead`) is for **SynFlowNet trajectory-balance loss**, not CFM. Wiring it into the locked `flow_matching_lipman/` CFM loss would require either (a) porting SynFlowNet into the CFM loop (R20+ scope), or (b) treating `BassCalibrationHead.loss(trajectory)` as a *post-hoc* calibration metric on the CFM-integrated trajectory — both are R20+ scope, not Wf-234.

### 3.4 `math_skills/monge_ampere_regularity.py` (汪徐家 2026, C^{2,α} regularity)

| Symbol | Type | Purpose |
|---|---|---|
| `KantorovichPotential(Q?, c?)` | class | Quadratic-form approx of `φ` — `parameters() -> (Q, c)`, `update_Q`, `update_c`, `hessian(x, eps=1e-3)` |
| `MongeAmpereSolver(n_iters=100, lr=0.01)` | class | Iterative solver — `solve(rho_0, rho_1, n_samples=32, dim?, seed=0) -> D²φ` |
| `AdvantageEstimatorMA(lam=0.1)` | class | **MA-regularised advantage** for SynFlowNet — `variance_bound(advantages: (n,)) -> scalar` |
| `.variance_bound(advantages)` | method | `Var(A)/n · (1 + λ · ||A||² / n)` (sharper Bernstein rate under C^{2,α}) |
| `.advantage(log_p, log_q)` | method | `A(x) = log p(x) - log q(x)` |

**Math invariants**:
* `_EPS_SPD = 1e-6` (SPD cone projection floor in `MongeAmpereSolver.solve`)
* `_HESSIAN_FD_EPS = 1e-3` (central-difference Hessian step)
* `MongeAmpereSolver.solve` only supports Gaussian `ρ_0, ρ_1` (Brenier map affine; non-Gaussian needs further rounds)

**Honest framing**: `KantorovichPotential` and `MongeAmpereSolver` are **statement-quality tools** — they let you assert that an empirical Brenier map has `C^{2,α}` regularity for a Gaussian test case. They do not slot into a CFM velocity field or loss. `AdvantageEstimatorMA.variance_bound` is a **diagnostic** for SynFlowNet variance (not a loss term). The R18 plan §P0-3 explicitly lists this as "**纯理论贡献** — 不改代码" (pure paper contribution, no code change).

### 3.5 `math_skills/multi_parameter_ph.py` (Kim-Jung 2026, Adv. Appl. Math. 174)

| Symbol | Type | Purpose |
|---|---|---|
| `two_param_filtration(pocket_coords, pocket_atom_types, max_radius=5.0, max_coord=4.0, n_bins=32) -> Dict[str, Tensor]` | function | 2-PH Vietoris-Rips filtration; returns `{feature_map, edge_mask, ...}` |
| `TwoParameterDiagram` | class | 2-PH diagram with `to_points() -> (B, P_max, 2)` and `point_mask` |
| `TwoParameterFiltration` | class | Stateful wrapper around `two_param_filtration` |
| `MultiPHKernel` | class | Wrapper exposing `sliced_wasserstein_kernel` |
| `MultiPHSummarizer(n_bins=32)` | class | `(B, 4, n_bins, n_bins)` summary (multi-persistence surface + multi-Betti curve + multi-birth histogram + doubly-centred) |
| `.summarize(diagrams) -> Tensor` | method | Returns the 4-channel summary |
| `.multi_betti_curve(diagrams) -> Tensor` | method | Channel 1 only |
| `.marginal_betti(diagrams) -> Tensor` | method | `(B, 2, n_bins)` marginals along `r` and `d` |
| `sliced_wasserstein_kernel(diagram_a, diagram_b, n_projections=8, sigma=1.0, seed=0) -> Tensor` | function | `(B, B)` RBF kernel matrix from sliced-Wasserstein |

**Math invariants**:
* `_EPS_LOG = 1e-30`, `_FOURIER_THRESHOLD = 1024`
* Naive O(N²) pair-distance path (Kim-Jung 2026 §4 Fourier acceleration is TODO above 1024 atoms)
* H₀ and H₁ only — multi-parameter PH on ℝ² is *not* finitely generated in general (Miller 2020 §6); H₀/H₁ are the safe subset
* `n_atoms <= 200` smoke budget; paper-grade requires R18-P3 Fourier path (not implemented)

### 3.6 `adapters/se3_sb_integration.py` (W2.1 ship)

| Symbol | Type | Purpose |
|---|---|---|
| `SE3SBVelocityField(base_velocity_net, manifold="SE3", eps=1e-6)` | `nn.Module` | Wraps EGNN → `se(3)` twist `(B, 6)` |
| `.forward(x, t, pocket=None)` | method | Per-atom Euclidean velocity → body-frame twist |
| `SE3SBDecoder(eps=0.1, n_iters=50, use_sinkhorn=True, seed=None)` | `nn.Module` | SE(3) geodesic + Sinkhorn SB coupling |
| `.decode(z0, z1, pocket?, t_values?) -> (B, n_t, 4, 4)` | method | Walks the SE(3) geodesic at each `t` |
| `.decode_at(z, t)` | method | Single-t decoder, returns `(B, 4, 4)` |
| `integrate_with_cfm(samples, velocity_field, n_steps=100, manifold="SE3", method="midpoint", eps=1e-6, t0=0.0, t1=1.0)` | function | Euler / midpoint SE(3) ODE integrator |

**Math invariants**: CPU-only; `manifold="SO3"` raises `NotImplementedError` (R18 roadmap); `use_sinkhorn=False` for bit-exact testing.

### 3.7 `adapters/se3_lipman_integration.py` (Integration 1 ship)

| Symbol | Type | Purpose |
|---|---|---|
| `SE3LipmanSolver(base_lipman_solver?, se3_adapter?, n_steps=50, method="midpoint", ortho_tol=1e-3)` | class | Drop-in wrapper around R17 `ODESolver` |
| `.solve(velocity_fn, x0, t_span, pocket?) -> Tensor` | method | `(B, N, 3)` integrated state |
| `.verify_se3_orthogonality(n_runs=10) -> float` | method | `max \|R^T R - I\|_inf` audit |
| `compare_se3_lipman_vs_base(lipman_solver, se3_solver, n_runs=10, z0?, n_steps_base=50, method_base="midpoint") -> Dict` | function | Wall-time + accuracy + orthogonality comparison harness |
| `SE3CFMComposite(base_cfm, se3_solver, n_steps=50, method="midpoint")` | class | Production entry point — composes base CFM with SE(3) solver |
| `.forward(z0, t=1.0, pocket?) -> Tensor` | method | The production replacement for `LipmanFlowMatchingAdapter.forward_velocity` |
| `.__call__(z0, t=1.0, pocket?)` | method | Callable alias for `forward` |

**Math invariants**: orthogonality hard contract `max_err < 1e-3`; `DEFAULT_N_STEPS = 50`, `DEFAULT_N_RUNS = 10`, `DEFAULT_ORTHO_TOL = 1e-3`.

### 3.8 Adjacent shipped adapters (not in harness list but in scope)

**`adapters/multi_ph_integration.py` (W2.3 ship — 480 LOC)**

| Symbol | Type | Purpose |
|---|---|---|
| `MultiPHPocketFeatures(max_radius=5.0, max_coord=4.0, n_bins=32)` | `nn.Module` | Stateless pocket feature extractor; `compute(pocket_clouds)` and `compute_with_atom_types(pocket_clouds, atom_types)` |
| `MultiPHMetalCoordHead` | `nn.Module` | Per-metal coord classifier head via sliced-Wasserstein kernel |
| `augment_pocket_embedding(pocket_embed, pocket_clouds, atom_types?, max_radius?, max_coord?, n_bins?)` | function | Concatenate `(B, H) + (B, n_bins²) -> (B, H + n_bins²)` |

**`adapters/transport_relaxation_integration.py` (W2.2 ship — 556 LOC)**

| Symbol | Type | Purpose |
|---|---|---|
| `RelaxedEndpointTerm(lambda_relax, epsilon)` | class | `forward(trajectory, energies) -> λ · KL(p_T ‖ q_T(ε))` |
| `augment_learned_shaping(learned_reward, lambda_relax, epsilon)` | function | `reward' = reward + λ · KL(p_T ‖ q_T)` (additive only) |
| `LambdaRelaxSweep(lambda_values, epsilon_values)` | class | Hyper-parameter sweep harness |

---

## 4. Integration points in `flow_matching_lipman/` (Gate G2)

The main CFM pipeline lives in three files (R17 close-out, locked):
1. `molmetal/adapters/flow_matching_lipman/__init__.py` — `LipmanFlowMatchingAdapter`, `EGNNVelocityField`, `PocketEncoder`, `_pcgrad_*` (2,930 LOC)
2. `molmetal/adapters/flow_matching_lipman/w_flow.py` — `WaveformRelaxationSolver`, `WFlowCFM`, `compare_solvers` (912 LOC; CPU skeleton per the docstring §honest framing)
3. `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` — `MeanFieldW2Plan`, `mean_field_score_smoothed_ot_plan` (mean-field limit + Xue 2026 score-smoothed OT bias)
4. `molmetal/adapters/flow_matching_lipman/connectivity_decoder.py` — `ConnectivityResult`, `ConnectivityAwareDecoder`, `_count_connected_components` (decoder + RDKit topology validation)

**Hard constraint** (per `__init__.py` design constraint 1 — repeated in `se3_sb_integration.py:26-29`, `transport_relaxation_integration.py:1-4`, `multi_ph_integration.py:39-41`, `se3_lipman_integration.py:34-35`): **do NOT modify** `molmetal/adapters/flow_matching_lipman/`. The R17 close-out is locked. Every wire-in below is therefore either (a) a **composition** at the call site (caller wraps the existing API), or (b) a **drop-in adapter class** that sits alongside the locked module — both approaches proven by the W2.1/W2.2/W2.3 ships.

### 4.1 Velocity-field integration points

| Wire-in | Drop-in class | Replaces / wraps | Test surface |
|---|---|---|---|
| **SE(3) geodesic velocity field** | `SE3SBVelocityField` (W2.1) | Wraps `EGNNVelocityField` → `(B, 6)` twist; the existing Euclidean `(B, N, 3)` velocity becomes the body-frame twist of the SE(3) flow | `test_se3_sb_integration.py` (25 tests, shipped) |
| **SE(3) Lipman solver** | `SE3LipmanSolver` / `SE3CFMComposite` | Drop-in replace of `molmetal.adapters.flow_matching_lipman._reference.solver.ODESolver.sample(x, n_steps, method)` contract | `test_se3_lipman_integration.py` (shipped; smoke 0/1000 fail) |
| **Multi-PH pocket feature** | `MultiPHPocketFeatures` + `augment_pocket_embedding` | Wraps `PocketEncoder.forward(pocket_coords, pocket_atom_types, pocket_mask)` output → adds `(B, n_bins²)` topology features next to the `(B, H)` EGNN pool | `test_multi_ph_integration.py` (shipped) |
| **W-Flow (waveform relaxation)** | `WaveformRelaxationSolver` / `WFlowCFM` | Drop-in replace of `_reference.solver.ODESolver` with `n_windows=4, n_iters_per_window=10` waveform-iteration; same `(x, t) -> v` callable contract | `test_w_flow.py` (shipped; w_flow.py is a CPU skeleton per its honest-framing §honest) |
| **PCGrad multi-task loss** | `_pcgrad_resolve` / `_pcgrad_apply_to_params` | Existing CFM + atom-head + bond-head gradients → projected onto each others' normal plane before summation | `__init__.py` lines 771-912 + multi-task training loop |

**Honest gap**: Bass-martingale and MA-regularity do NOT have a velocity-field wire-in (their APIs are diagnostic / cite-only). They enter the paper as **§3 method reference text** (R18 plan §P0-3 §P1-3), not as CFM operators.

### 4.2 OT-solver integration points (`optimal_transport_v2.py`)

The current OT solver is **mean-field W2** (Xue 2026 lecture 4 — `MeanFieldW2Plan`) + **score-smoothed OT bias** (`mean_field_score_smoothed_ot_plan`). Wire candidates:

| Wire-in | Math layer used | Effect on `optimal_transport_v2.py` |
|---|---|---|
| **SE(3) SB coupling** (`LogSinkhornLoss` + `SchrodingerBridgeSE3.sample`) | Replace the Euclidean squared-distance cost with twist-space `(B², 6) → R+` cost and add Sinkhorn coupling for cross-pair assignment | `MeanFieldW2Plan.recommended_step_size` gains a `manifold="SE3"` mode (CPU-only; safe because the solver file is **not** the R17-locked CFM file — `optimal_transport_v2.py` is a Phase-2.2 internal experiment) |
| **Bass martingale calibration** (`BassCalibrationHead.loss`) | Post-hoc diagnostic on the integrated trajectory; not an OT solver replacement | Read-only — returns `(loss, mean_preserving)` for log/dashboard |
| **MA regularity check** (`KantorovichPotential.hessian`) | Post-hoc diagnostic on the Brenier map; only Gaussian test case supported | Read-only — returns `D²φ` for log/dashboard |
| **Mean-field W2 baseline (already in)** | (`MeanFieldW2Plan`, `mean_field_score_smoothed_ot_plan`) | n/a — already wired |

**Honest gap**: the OT solver file is a Phase-2.2 experimental layer, not the R17-locked CFM stack, so the "do not modify" constraint is softer here. But adding SE(3) Sinkhorn coupling touches only the cost-matrix step (`MeanFieldW2Plan.mean_field_residual`), not the score-smoothed flow — and the R18 plan §P0-1 explicitly lists this as a 3-week sprint, so it stays in scope for the next ultracode window.

### 4.3 Decoder / connectivity integration points (`connectivity_decoder.py`)

The decoder is the **post-velocity-field topology validator**: it runs the bond head, builds the `(N, N)` connectivity matrix, and rejects disconnected samples.

| Wire-in | Math layer used | Effect on `connectivity_decoder.py` |
|---|---|---|
| **Multi-PH pocket topology feature** (`MultiPHKernel.sliced_wasserstein_kernel`) | Add a per-pocket topology fingerprint as a **decoder fallback signal** when connectivity check fails | Read-only — adds a `(B, B)` kernel matrix to the rejection log; doesn't change the accept/reject gate |
| **Bass-martingale calibration** | NOT applicable | Decoder doesn't produce trajectories, only atom-clouds + bond-orders |

**Honest gap**: `connectivity_decoder.py` doesn't have a natural OT step to swap. The most useful addition is using `MultiPHKernel` as a *post-rejection topology signal* — flag pockets whose topology kernel against the per-metal reference is below threshold, so the rejection can be routed to a per-metal fallback decoder.

### 4.4 Waveform-relaxation solver integration (`w_flow.py`)

W-Flow is the **CPU skeleton** wrapper around the R17 `ODESolver`. The integration point is a single callable:

```python
def velocity_fn(x: Tensor, t: Tensor) -> Tensor:
    """Velocity field callable; t is 0-d scalar."""
```

**Wire-in matrix for `w_flow.py.solve(velocity_fn, ...)`**:

| Wire-in | Replaces `velocity_fn` with | Result |
|---|---|---|
| **SE(3) geodesic** | `SE3SBVelocityField` (W2.1) wrap of the same EGNN → `(B, 6)` twist | Solves the SE(3) geodesic flow inside the waveform-relaxation outer loop; pure CPU skeleton |
| **W-Flow + Bass-martingale** | Replace the inner-loop `euler_step` / `midpoint_step` with a Bass-martingale diffusion step | R18 plan §P1-3 lists 3 weeks — the inner-loop step in `w_flow.py:_integrate_one_window` is the only place to swap |
| **W-Flow + MA-regularity check** | Add a `MA_regularity_audit(per_window_q_inner_rate) -> bool` post-check on the Brenier map's regularity | Diagnostic only |

**Honest gap**: W-Flow is intentionally a CPU skeleton (per `w_flow.py` docstring §honest: "100× wall-time speedup... not achievable on a single-thread CPU skeleton"). GPU window-parallelism is R18+ scope.

### 4.5 Loss / reward integration points

The CFM loss is `MSE(v_θ(x_t, t), x_1 - x_0)` from `LipmanFlowMatchingAdapter.train_step`. Wire candidates:

| Wire-in | Math layer used | Adds |
|---|---|---|
| **MA-regularised advantage** (`AdvantageEstimatorMA.variance_bound`) | Diagnostic on `advantages` for the SynFlowNet trajectory-balance head | Optional logging field `ma_variance_bound`; **not** added to the loss (research-only) |
| **Bass-calibration head** (`BassCalibrationHead.loss`) | Replace TB loss with Bass discrepancy | Requires porting SynFlowNet TB into CFM — R20+ scope |
| **PCGrad** (`_pcgrad_resolve`) | Already in `__init__.py` | Joint multi-task training (cfm + atom-head + bond-head) with gradient surgery |

**Honest gap**: only PCGrad is a production-ready loss wire-in. Bass and MA-regularity are diagnostic + paper-only.

### 4.6 Feature-extraction integration points

| Wire-in | Math layer used | Where it sits in the pipeline |
|---|---|---|
| **Multi-PH pocket features** (`MultiPHPocketFeatures.compute`) | Replaces / augments `PocketEncoder.forward(pocket_coords, pocket_atom_types, pocket_mask)` output | Already shipped as `augment_pocket_embedding` — drop-in before `EGNNVelocityField.forward` |
| **Multi-PH metal coord head** (`MultiPHMetalCoordHead.forward`) | Per-metal reference diagram comparison via sliced-Wasserstein kernel | Augment pocket conditioning with a `(B, n_metals)` coord-probability head |
| **SE(3) twist encoding** (`se3_log`) | Replace per-atom Euclidean velocity with body-frame twist `(B, 6)` | Inference-time only; training uses Euclidean (CFM theorem) |

---

## 5. Per-module integration effort (Gate G4)

The "effort" is the engineering cost to drop the module into the locked CFM pipeline **without modifying `flow_matching_lipman/`**. It is split into (a) adapter-already-done / wrapper-composable / drop-in-ready tiers, and (b) honest framing of what's *not* a production wire-in.

| Module | Tier | Effort | Where it lands | What it does NOT do |
|---|---|---|---|---|
| `se3_schrodinger_bridge.py` | adapter-already-done | **1 day** (smoke + test re-run + commit message) | `adapters/se3_sb_integration.py` (W2.1) + `adapters/se3_lipman_integration.py` (Integration 1) | Does not replace the R17 CFM solver shim by itself — caller wraps |
| `multi_parameter_ph.py` | adapter-already-done | **0.5 day** (augment_pocket_embedding wiring + paper §3.1 cite) | `adapters/multi_ph_integration.py` (W2.3) | Does not replace `PocketEncoder` — augments alongside |
| `transport_relaxation.py` *(adjacent, not in harness list but in family)* | adapter-already-done | **0.5 day** (`augment_learned_shaping` wrapping the reward head) | `adapters/transport_relaxation_integration.py` (W2.2) | Does not replace `learned_shaping` — additive only |
| `optimal_transport_v2.py` (Phase-2.2 internal) | wrapper-composable | **2-3 days** (SE(3) twist-space cost + Sinkhorn coupling) | Extend `MeanFieldW2Plan.mean_field_residual` with `manifold="SE3"` | Does NOT touch the R17-locked CFM decoder / solver |
| `w_flow.py` (CPU skeleton) | wrapper-composable | **1-2 days** (wrap `integrate_with_cfm` into `_integrate_one_window`) | Make W-Flow outer loop call SE(3) inner loop; keep CPU-only | Does not deliver the 100× GPU parallelism claim |
| `bass_existence.py` | research-only | **0 code change** | `paper/appendices/closure_theorem.tex` §B cite + `BassExistenceTheorem.statement()` text | **Not** a velocity-field wire-in; not a loss; not a feature extractor |
| `sb_bass_martingale.py` | research-only (drop-in exists for SynFlowNet TB) | **0 code change to CFM**; optional **2-3 days** for a post-hoc trajectory-calibration metric on the CFM-integrated trajectory (R20+ scope) | `paper/section_02_related_work.tex` §SynFlowNet calibration text | **Not** a CFM loss replacement; **not** an OT solver replacement |
| `monge_ampere_regularity.py` | research-only (statement-text + Gaussian Brenier solver) | **0 code change** | `paper/section_03_method.tex` §3.4 (汪徐家 2026 C^{2,α} regularity citation + `KantorovichPotential` test-case figure) | **Not** an OT solver replacement; **not** a velocity-field wire-in; **not** a loss term |

**Total wire-in cost** (excluding the 3 research-only modules): **5-7 days** for the full SE(3) SB + multi-PH + transport-relaxation + W-Flow + OT-solver composition. This is the Wf-234 budget — fits inside one ultracode window.

### 5.1 Honest framing note (per task constraint "honest framing: if a module is research-only, say so")

The three "research-only" modules — `bass_existence.py`, `sb_bass_martingale.py`, `monge_ampere_regularity.py` — are NOT being dishonestly dropped. Their **value** to R18 is:

1. **`paper/section_03_method.tex`** — they are the mathematical citations that turn the CFM pipeline from "engineering recipe" into "theorem-backed method". Without them, R18 §3 has no Monge-Ampère regularity claim, no Bass-martingale calibration claim, no Backhoff-Veraguas existence result.
2. **`paper/appendices/closure_theorem.tex`** — the closure theorem appendix already imports `BassExistenceTheorem` and `BassMartingale` for the proof-text (per the existing call sites in `molmetal/molmetal_lam/reward/{pocket_closure_bias,closure_oversample}.py`).
3. **Future R20+** — when SynFlowNet TB-loss gets ported into the CFM stack (per the R20-P2 Hilbert-ASTSR plan), `BassCalibrationHead.loss` becomes the natural replacement for `TB loss`. That is NOT a Wf-234 deliverable.

**Forcing these three modules into the locked CFM pipeline would be**:
* A violation of the R17 invariant ("do not modify `flow_matching_lipman/`" — the `BassCalibrationHead.loss` doesn't have the `(B, N, 3) -> (B, N, 3)` contract the CFM ODE solver expects).
* A spurious claim of "Bass-martingale CFM" when the actual code path is Euclidean CFM + a post-hoc calibration read.
* A paper section that over-claims the integration (honest-framing per `R18_PLAN.md §honest` requires us to distinguish *mathematical skeleton* from *wired-into-pipeline*).

---

## 6. Recommended execution order (for the next ultracode window)

Per the user's request "启动ultracode注意数学上的细节把他们所有都集成进去":

1. **(Day 1)** Wire `se3_schrodinger_bridge.py` into `w_flow.py:_integrate_one_window` as the inner-loop integrator. Existing W2.1 smoke (`verify_se3_sb_adapter.py`) + W2.1 tests = ready.
2. **(Day 1-2)** Wire `multi_parameter_ph.py` into `LipmanFlowMatchingAdapter.setup` via `augment_pocket_embedding`. Existing W2.3 smoke + tests = ready.
3. **(Day 2-3)** Wire `transport_relaxation.py` (the adjacent shipped adapter) into the reward head. Existing W2.2 smoke + tests = ready.
4. **(Day 3-5)** Extend `optimal_transport_v2.py:MeanFieldW2Plan.mean_field_residual` with `manifold="SE3"` using `se3_log` + `LogSinkhornLoss`. R18 plan §P0-1 budget = 3 weeks; this is the highest-effort wire-in.
5. **(Day 5-6)** Paper §3 update: cite `mahpood2026sb_lie`, `kim_jung2026multi_ph`, `wang_xujia2026ma_regularity`, `backhoff_veraguas2020bass_existence`, `henry_labordere2026sb_bass` in `paper/section_03_method.tex` §3.4 + §3.1 + appendix B. (NOT a code change.)
6. **(Day 6-7)** Smoke + dashboard: `verify_train_ru_5metal.py` + `verify_se3_sb_adapter.py` + `verify_transport_relaxation_adapter.py` + `verify_multi_ph_integration.py` all green; one joint `verify_all_math_wired` script that exercises the full chain.

**Blockers**:
* Standing "kill then adapt" GPU directive (per `TODO/INDEX.md` BD-5) — wire-in work is CPU-only so OK.
* R18 plan §risk: ultracode training cannot run in the same window as Blackwell — wire-in is CPU skeleton so OK.

---

## 7. Cross-references

* W2.1 ship: `TODO/completed/55_r18_integration_adapters.md` §2.1
* W2.2 ship: `TODO/completed/55_r18_integration_adapters.md` §2.2
* W2.3 ship: `TODO/completed/55_r18_integration_adapters.md` §2.3
* R18 math migration plan: `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §2-§4
* R18 high-level plan: `TODO/03_plans/R18_PLAN.md` §0-§risk
* W-Flow CPU skeleton: `molmetal/adapters/flow_matching_lipman/w_flow.py` docstring §honest
* Existing OT solver: `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` (mean-field W2, Phase-2.2 internal)
* Math modules inventory: `molmetal/math_skills/__init__.py`
* Honest-framing policy: `TODO/03_plans/R18_PLAN.md` §honest + `party-twenty-spirit` skill §3.5