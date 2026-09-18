# Audit: flow_matching (facebookresearch/flow_matching)

Path: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/flow_matching/`
Our hand-rolled code: `LipmanFlowMatchingAdapter` in
`/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py`
(the `interpolate`, `lipman_loss`, `sample_path` helpers at ~lines 350–700
plus the Euler integrator at ~lines 1700–1900)

## (1) What we already have hand-rolled that is redundant

- `AffineProbPath` (`flow_matching/path/affine.py`): we re-implement the
  conditional-OT interpolation `x_t = (1-t)·x_0 + t·x_1 + σ·t·(1-t)·ε` in
  our `interpolate(...)` helper. facebookresearch ships exactly this with
  vectorised batched ops and 5 schedulers (CondOT, Cosine, VPSDE,
  LinearVPSDE, PolynomialConvex).
- `CondOTScheduler` (`flow_matching/path/scheduler/scheduler.py`): we hand-roll
  `alpha(t)`, `beta(t)`, `d_alpha_dt` for the linear interpolation; they
  expose `alpha_t`, `beta_t`, `d_alpha_t`, `d_beta_t` out of the box.
- `ODESolver` (`flow_matching/solver/ode_solver.py`): we run plain Euler;
  facebookresearch provides `ODESolver` with Euler / midpoint / RK4 +
  `MixtureDiscreteEulerSolver` for discrete flows. We can also use their
  `ModelWrapper` to plug any `nn.Module` without rewriting the call site.
- `MixturePathGeneralizedKL` (`flow_matching/loss/`): a correct CFM loss
  for mixture paths; ours is hard-coded for CondOT only.
- `AffineProbPath.sample(x_0, x_1)` returns a typed `PathSample` with
  `.x_t`, `.dx_t`, `.t`, `.x_0`, `.x_1`; ours returns a tuple of dict.

## (2) What is actually different / better in our hand-rolled code

- **3D rigid-body SE(3) decomposition**: facebookresearch is generic
  Euclidean / Riemannian / discrete; we project velocity into (translation,
  rotation SO(3), torsion dihedral) and ODE-integrate separately.
  Importing their `ODESolver` for this requires a custom `ModelWrapper` —
  feasible but not free.
- **Discrete atom-type + bond-tensor pair**: `MixtureDiscreteEulerSolver`
  handles the *discrete* part, but our bond-order CE loss is separate and
  trained jointly with the CFM loss; facebookresearch treats discrete
  flows in a separate example, not as a multi-task head.
- **Joint decode-and-bond**: facebookresearch expects you to give it
  `x_1_hat` from `sample`; we run the bond-head AFTER the CFM integration
  to round out bond orders. Their API doesn't model this.
- **Midpoint solver smoke**: we already adopted midpoint via TOP FIX #2
  (task #799) — so the solver gap is partly closed.

## (3) Concrete 3-line patch plan (file + lines + import)

```
# adapters/flow_matching_lipman/__init__.py:1700-1900 (the Euler block)
from flow_matching.solver import ODESolver
from flow_matching.path import AffineProbPath, CondOTScheduler
# then replace the manual Euler with:
path = AffineProbPath(scheduler=CondOTScheduler()); sample = path.sample(x0, x1)
solver = ODESolver(velocity_field=velocity_net)  # midpoint by default
traj = solver.sample(x_init=sample.x_t, step_size=1/N, method='midpoint')
```
Net effect: ~150 LOC removed from our adapter; midpoint solver becomes
free; we keep our 3D SE(3) projection wrapper at the velocity-field boundary.

## Verdict

**REUSE YES — replace the path/scheduler/solver half of our adapter.**
License is CC BY-NC (problematic for commercial, fine for academic).
Import `AffineProbPath` + `CondOTScheduler` + `ODESolver`; keep our
3D-projection wrapper, bond-head, joint training loop. Highest-ROI swap
in the entire audit.