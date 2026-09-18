# Sweep guidance implementation evidence — 2026-09-13

The runner can now consume an explicitly serialized **linear descriptor prior**
and a separately configured synthesis gate. This does **not** complete the
original PySR symbolic regression milestone or establish a learned
retrosynthesis backend on this machine.

## API

`run_one_pocket` accepts these additional keyword arguments:

- `prior_state=None`: JSON replay/coefficient state supplied by the caller.
- `prior_mode="frozen"`: default; neither adds held-out observations nor refits.
- `prior_data_split="test"`: split label recorded by the caller.
- `synthesis_config_path=None`: explicit AiZynth configuration and model/stock paths.

`symbolic_prior=True` (legacy switch name), or the explicit string
`"linear_descriptor_prior"`, selects the linear backend. The result reports
`pysr_symbolic_regression=False`. A fitted `SymbolicPrior` wrapper actually
enters the MCTS `_prior` PUCT dispatcher before the reward heuristic. Unfitted
states retain the reward heuristic; no synthetic fit is performed to make a
switch appear active.

`prior_mode="train"` requires `prior_data_split="train"` or `"development"`.
The caller must ensure these really identify its input dataset; the runner
cannot infer dataset membership from arbitrary path names. Mutable replay
states are separated by experiment seed. At least eight distinct measured
candidate observations with nonconstant descriptor features and reward values
are required. `symbolic_prior_refit_every` counts completed pockets, not MCTS
simulations. Unchanged observations do not trigger another fit. Labels are
actual **search reward values**, not measured docking affinity, biological
potency, or wet-lab outcomes. Candidate-only replay has selection bias.

An explicit `prior_mode="transductive"` can update from evaluation observations,
but records that fact and its resulting state is rejected as frozen pretrained
input. The driver controls persistence and which independent seed's state is
passed to the next job. The output includes `prior_state`, `prior_report`, and
`synthesis_report`; failure branches preserve state for deterministic resume.
No process-global replay cache is used.

## Synthesis gate

`True` or `"aizynthfinder"` requires a real loaded AiZynth configuration. No
configuration was found in the inspected `/mnt/storage/data/molmetal` or
`/mnt/storage/models` listings. Missing configuration/models are explicitly
`not_applied`; RDKit SA is never substituted for a learned route oracle.
If a configured AiZynth adapter silently returns a SMARTS fallback at runtime,
that result is rejected rather than accepted as an AiZynth route.

`"smarts"` explicitly enables the existing non-learned SMARTS heuristic. This
is a real post-search candidate filter with per-candidate reports, but it does
not prove purchasable starting materials or a chemically executable route.
Disabled `False` / `0` settings are labelled `disabled`, not unsupported.

## Validation

55 focused tests passed in 9.05 seconds on the project's `uv run` Python 3.12.
Coverage includes frozen-state identity, strict split/seed isolation,
insufficient-data warmup, interval scheduling, JSON round-trip coefficients,
real PUCT invocation, actual candidate descriptor-reward measurements,
explicit SMARTS filtering, rejection of learned-backend fallback, failed-job
state preservation, real click-tile generation, configuration dispatch, and
coordinate-preserving PoseBusters checks. No long sweep was run by this task.

The implementation tests use small development fixtures with RDKit QED labels.
They establish behavior, not a production pretrained model or empirical prior
improvement. A production training/development dataset and a frozen trained
state remain necessary before evaluating learned-prior gains.

Runtime code frozen for the parent driver's next independent physical smoke:

- `molmetal/scripts/lambda_100pocket_sweep.py`: `4154e74d694ecb7af7c9ef06c50a79d8f52b29fdc7e8b4a22d9bacd1be18c262`
- `molmetal/molmetal_lam/search_alg/sweep_guidance.py`: `f7e34588ffb795e8d04c8110df4b897dddacdf071409b285a88dfcb4dc076ad4`
