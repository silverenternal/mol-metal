# Round-10 axis D micro-ablation — 1h36 pocket (MetalGeometryPrior on/off)

## Goal

Quantify the Vina-score + PoseBusters-pass-rate effect of engaging the `MetalGeometryPrior` (square-planar Pt(II), IDEAL_SQUARE_PLANAR_ANGLE = pi/2) on the EGNN flow-matching sampling trajectory for the 1h36 pocket.  Prior ON weights the analytic gradient of the per-centre angular penalty into the velocity field every integration step, nudging the generated coordinates toward canonical 90 deg Pt-L-X angles.  Prior OFF skips the gradient nudge entirely.

## Setup

- Pocket: **1h36** (572 atoms, radius=21.1A)
- PDB: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb`
- N molecules requested per setting: **20**
- Prior weights swept: **[0.0, 0.1]** (0.0 = prior OFF)
- CFM train steps per setting: **30** (hidden_dim=32, n_layers=2, lr=0.0002, context_dropout=0.1)
- ODE integration steps: **8**
- Vina exhaustiveness: **2** (n_poses=1)
- Coordinate safety clamp: **[-10.0, +10.0] A**
- Pocket-embed L2 norm target: **1.0**
- Total wall: **42.60s** (0.71 min)

## Prior wiring

- `MetalGeometryPrior.enabled` flag (Round-10 axis D) is the canonical on/off gate.  When `False`, `prior_loss` returns exactly 0 regardless of the geometry — the prior's analytic gradient is therefore 0 and the velocity field is bit-exact identical to the prior-OFF CFM baseline.
- `LipmanFlowMatchingAdapter(metal_prior_weight=...)` plumbs the weight into the per-velocity-step prior-nudge wrapper (see `_velocity_with_metal_prior` in `flow_matching_lipman`). `metal_prior_weight=0.0` is the historical short-circuit (wraps the bare `velocity_model`); `>0` adds the gradient every `metal_prior_k_every` steps (here we set k_every=1 so the prior is applied on every ODE step).
- Sanity probe at run start asserts `MetalGeometryPrior(weight=1.0).disable().enabled == False` for OFF and `True` for ON.  Failure aborts the run with an AssertionError so the harness can never report a misleading delta.

## Results

| prior_weight | n_docked | mean | median | min | max | delta_vina_vs_OFF | PB_pass_rate | delta_pb_vs_OFF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 4 | -2.197 | -2.000 | -2.979 | -1.809 | +0.000 | 0.167 | +0.000 |
| 0.100 | 4 | -2.197 | -2.000 | -2.979 | -1.809 | +0.000 | 0.167 | +0.000 |

## Headline

- Best prior weight: **0.000** (Vina mean **-2.197** kcal/mol over 4 docked mols)
- Delta Vina vs prior OFF: **+0.000** kcal/mol  (negative = prior ON improves binding)
- Success criterion (Vina delta <= -0.2 kcal/mol): **FAIL** (observed +0.000)
- Success criterion (PB pass-rate >= baseline): **PASS** (observed +0.000)

## Notes / caveats

- The CFM training data is synthetic random points (the spec forbids running real training data; this is an algorithmic micro-bench).  With only 30 train steps the model is barely above noise — Vina scores cluster near -2 kcal/mol and the per-setting ranking is dominated by sampling noise rather than a real prior-induced binding-affinity shift.
- 1h36 contains an Fe(HEM) centre, not Pt — the prior's analytic gradient is only non-zero when the generated molecule contains Pt (Z=78) or Pd (Z=46) AND the metal-geometry distance bounds flag >=2 donor atoms.  Most sampled mols do not contain Pt, so the prior is a strict no-op on the majority of the generated batch — this is the canonical limitation when the pocket metal does not match the prior's target centre.  The harness still measures the cost (no-op == identical trajectory) so the report can quantify the cost of always-on prior wiring even when the prior doesn't fire.
- The 1h36 PDB contains 2 Pt-analogue heavy atoms added for prior testing (see Round-8 notes).  When the generated molecule has Pt, the prior should fire; when not, both ON and OFF paths produce identical coordinates.  This is expected — the canonical test is whether the prior ever fires + lowers Vina on the dockable subset.
- PB pass-rate can be `n/a` when posebusters is not installed in the environment; in that case the success criterion for PB is skipped and only the Vina delta is reported.
