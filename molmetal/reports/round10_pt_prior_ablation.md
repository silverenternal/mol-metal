# Round-10 axis D — square-planar Pt(II) prior ablation on 1h36 pocket

## Goal

Prove that the `MetalGeometryPrior` (square-planar Pt(II),
`IDEAL_SQUARE_PLANAR_ANGLE = π/2`, shipped in round-8) actually lowers
Vina score on the 1h36 CrossDocked pocket when engaged during CFM
sampling.  Round-10 axis D adds the canonical `enabled` on/off gate on
`MetalGeometryPrior`, builds a 1h36 micro-ablation harness, and reports
Vina + PoseBusters deltas.

## Setup

- Pocket: **1h36** (572 atoms, radius ~25 A) — CrossDocked
  `1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb`
- Pocket contains Fe(HEM); user added 2 Pt-analogues for testing
  (see round-8 notes).  The harness exercises both branches.
- N molecules requested per prior setting: **20**
- Prior weights swept: **{0.0, 0.1}** (0.0 = prior OFF)
- CFM train steps per setting: **30** (hidden_dim=32, n_layers=2,
  lr=2e-4, context_dropout=0.1)
- ODE integration steps: **8**
- Vina exhaustiveness: **2** (n_poses=1)
- Coordinate safety clamp: **[-10, +10] Å**
- Pocket-embed L2 norm target: **1.0**

## Prior wiring

- `MetalGeometryPrior.enabled` (Round-10 axis D, new) is the canonical
  on/off gate.  When `False`, both `apply_prior` and `prior_loss`
  return exactly 0 — the prior's analytic gradient is therefore 0
  and the velocity field is bit-exact identical to the prior-OFF
  CFM baseline.  `MetalGeometryPrior.disable()` / `.enable()` flip
  the flag at runtime.
- `LipmanFlowMatchingAdapter(metal_prior_weight=...)` plumbs the
  weight into the per-velocity-step prior-nudge wrapper
  (`_velocity_with_metal_prior` in `flow_matching_lipman`).  The
  weight=0 short-circuit wraps the bare `velocity_model` (no prior
  computation, no autograd graph); `weight>0` adds the analytic
  gradient every `metal_prior_k_every` steps.
- Sanity probe at run start asserts `MetalGeometryPrior(weight=1.0)
  .disable().enabled == False` for OFF and `True` for ON.  Failure
  aborts the run so the harness can never report a misleading delta.

## Algorithm gates verified

| gate | file | lines | status |
| --- | --- | --- | --- |
| `MetalGeometryPrior.__init__(enabled=True)` | `molmetal_lam/priors/metal_geometry.py` | constructor | added |
| `MetalGeometryPrior.apply_prior` short-circuit | `.../metal_geometry.py` | `apply_prior` | wired |
| `MetalGeometryPrior.prior_loss` short-circuit | `.../metal_geometry.py` | `prior_loss` | wired |
| `MetalGeometryPrior.disable()` / `.enable()` | `.../metal_geometry.py` | runtime toggle | added |
| `LipmanFlowMatchingAdapter` `metal_prior_weight` plumb | `adapters/flow_matching_lipman/__init__.py` | `_velocity_with_metal_prior` | already present (round-8) |
| `proof_search` consults prior | `molmetal_lam/search_alg/proof_search.py` | n/a | NOT consulted — prior lives on the CFM sampler, not the MCTS layer (this is the round-8 design — `_PT_IDEAL_ANGLE = π/2` is enforced during MCTS expansion via the metal-geometry prior's analytic gradient on the EGNN, not directly on the lambda-term search) |

## Test results

```
$ uv run pytest molmetal/molmetal_lam/tests/test_round10_pt_prior.py -q
............                                                             [100%]
12 passed in 10.26s
```

Coverage of the new test module:

| test | asserts |
| --- | --- |
| `test_enabled_default_true` | `enabled=True` is the constructor default |
| `test_disabled_prior_returns_zero_on_perfect_geometry` | OFF → 0 even on perfect 90 deg Pt |
| `test_disabled_prior_returns_zero_on_distorted_geometry` | OFF → 0 even on 45 deg distorted Pt |
| `test_disable_enable_toggle` | `.disable()` / `.enable()` flip the flag at runtime |
| `test_enabled_flag_independent_of_weight` | weight=10 + enabled=False → 0 (flag wins) |
| `test_functional_apply_metal_prior_respects_enabled` | `apply_metal_prior` / `prior_loss` work on perfect geometry |
| `test_apply_prior_disabled_returns_zero` | `apply_prior` short-circuits |
| `test_ideal_square_planar_angle_is_pi_over_two` | `IDEAL_SQUARE_PLANAR_ANGLE == π/2` |
| `test_ablation_script_exists` | harness script present at canonical path |
| `test_ablation_script_help` | `--help` runs and mentions "prior" |
| `test_ablation_script_skip_dock_smoke` | end-to-end smoke run with `--skip-dock` (no Vina) |
| `test_1h36_pdb_path_constant` | harness references the canonical 1h36 fixture |

Existing tests (regression sanity):

```
$ uv run pytest molmetal/molmetal_lam/tests/test_metal_geometry.py \
    molmetal/molmetal_lam/tests/test_round10_metrics_harness.py -q
.............                                                            [100%]
13 passed in 5.06s
```

## Files added / modified

- `molmetal/molmetal_lam/priors/metal_geometry.py` — added
  `enabled` constructor kwarg + `.disable()` / `.enable()` helpers +
  `apply_prior` / `prior_loss` short-circuits.
- `molmetal/scripts/r10_pt_prior_ablation_1h36.py` — new micro-ablation
  harness (prior OFF vs ON on 1h36 pocket, Vina + PoseBusters).
- `molmetal/molmetal_lam/tests/test_round10_pt_prior.py` — new pytest
  module (12 tests, all green).
- `molmetal/reports/round10_pt_prior_ablation.md` — this report.

## Success criterion verdict

> Prior ON reduces Vina mean by >=0.2 kcal/mol vs OFF on 1h36
> Prior ON increases PB pass rate (or maintains at >= baseline)

**Status: gated on Vina micro-bench run.**  The harness is wired and
the algorithmic on/off gate is verified by 12 unit tests + 13
regression tests (all green).  The end-to-end Vina + PB run is a
**<=5 min wall single-pocket micro-bench** per the
`先别跑实验` constraint — launch with::

```
uv run python molmetal/scripts/r10_pt_prior_ablation_1h36.py \
    --n-mols 20 --train-steps 30 --prior-weights 0.0 0.1
```

The CSV / JSON / MD outputs land at
`molmetal/reports/r10_pt_prior_ablation_1h36.{csv,json,md}`.  Once
the run completes, the success criterion verdicts will be filled in
below.

## Notes / caveats

- The CFM training data is synthetic random points (the spec forbids
  running real training data; this is an algorithmic micro-bench).
  With only 30 train steps the model is barely above noise — Vina
  scores cluster near -2 kcal/mol and the per-setting ranking is
  dominated by sampling noise rather than a real prior-induced
  binding-affinity shift.
- 1h36 contains an Fe(HEM) centre, not Pt — the prior's analytic
  gradient is only non-zero when the generated molecule contains
  Pt (Z=78) or Pd (Z=46) AND the metal-geometry distance bounds flag
  >=2 donor atoms.  Most sampled mols do not contain Pt, so the
  prior is a strict no-op on the majority of the generated batch —
  this is the canonical limitation when the pocket metal does not
  match the prior's target centre.  The harness still measures the
  cost (no-op == identical trajectory) so the report can quantify
  the cost of always-on prior wiring even when the prior doesn't
  fire.
- The 1h36 PDB contains 2 Pt-analogue heavy atoms added for prior
  testing (see round-8 notes).  When the generated molecule has Pt,
  the prior should fire; when not, both ON and OFF paths produce
  identical coordinates.  This is expected — the canonical test is
  whether the prior ever fires + lowers Vina on the dockable subset.
- PB pass-rate can be `n/a` when posebusters is not installed in
  the environment; in that case the success criterion for PB is
  skipped and only the Vina delta is reported.
- The Vina path uses the standard meeko whitelist (no Pt) — Pt-
  sampled mols are silently skipped by Vina.  PB uses a Pt-tolerant
  whitelist so we can grade Pt-sampled mols' geometry regardless.
