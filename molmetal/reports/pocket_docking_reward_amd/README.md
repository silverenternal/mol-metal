# Actual pocket-conditioned docking reward provider (2026-09-13)

## Goal

Provide real docking energy to an MCTS reward callback using the exact receptor, explicit box and seed, with canonical-SMILES caching, a hard evaluation budget and saved pose/device evidence. Reference ligand coordinates may define the box; its chemistry and reference score must not supply the reward.

## Outcome

**PASS for the independent provider on real CrossDocked test_001.** QuickVina2-GPU 2.1 uses the strictly prepared original receptor, 1,000 OpenCL search lanes, depth 1, seed 42 and one pose. Both uncached searches complete kernel1 and kernel2 on gfx1101 and save their actual docked conformers.

| Callback input | Result | New docking attempt |
|---|---|---:|
| CCO | −2.3 kcal/mol | yes |
| OCC | −2.3 kcal/mol, canonical cache hit | no |
| c1ccccc1 | −3.7 kcal/mol | yes |
| CCC | explicit budget-exhausted exception; energy null | no |

The report records `n_attempted=2`, `n_docked=2`, `n_cache_hits=1`, `n_budget_exhausted=1` and all four calls. Both scored candidates retain actual GPU trace/log/config provenance, requested/effective seed, the selected pose, all returned pose records and their SHA256 hashes. No reference molecule is docked or scored in this provider. `search_docking_reward.json` contains the complete audit; `integration.json` records the smoke's acceptance checks.

Seven focused tests pass: canonical/state callback cache behavior, hard-budget exceptions, failed-attempt accounting and cached failure, invalid-input preservation, finite scores rejected without GPU trace or matching pose chemistry, and refusing cached reuse when input receptor/reference files change.

## Interface and semantics

```python
from molmetal.molmetal_lam.sbdd_env.pocket_docking_reward import PocketDockingReward

provider = PocketDockingReward(
    receptor_path, ligand_path, output_dir,
    seed=42, engine="quickvina2-gpu",
    gpu_config="molmetal/configs/amd_gpu_docking.json",
    max_evaluations=16, exhaustiveness=1, n_poses=1,
)
raw_kcal_mol = provider(state)  # accepts SMILES or state.canonical_smiles()
audit = provider.report()
```

The callable returns the lowest finite **actual** pose energy, with its raw sign and units. It raises `PocketDockingRewardError` on invalid input, receptor/engine/pose failure, changed input files or budget exhaustion. Energy remains null for failures. The caller owns reward sign/weight and must not label exception-to-zero aggregation as a measured docking energy. This module does not modify or register itself with any runner or reward aggregator.

Receptor preparation is lazy and uses the established `prepare_receptor_for_evaluation` helper. The box exactly follows `evaluate_generated_poses`: heavy-atom centroid of the paired ligand, cube side `max(12, 2*max_abs_deviation + 8)` Å. Its source hash and geometric role are explicit; no ligand fingerprint/chemistry, native score or reference-score threshold enters the reward. Missing receptor atoms remain a preparation failure unless a separately labelled modeled receptor is explicitly supplied.

Canonical isomeric SMILES identifies cache entries within one fixed receptor/config provider. Successes and failures are cached. The budget counts actual calls to the docking engine, including failed calls; cache hits, invalid inputs and receptor preparation failures consume no additional docking attempts. Original source hashes are checked before cached reuse. `report()` provides the parent-requested top-level counters and `candidates`, with per-call status/latency and detailed budget accounting. JSON is checkpointed atomically before preparation/docking and after each call so an interrupted process leaves evidence.

GPU mode requires the configured trace library and verified completed GPU kernels; there is no CPU fallback. Explicit native-engine choices are supported through the existing native adapter, but this smoke exercises the AMD path. GPU lane/depth budgets are distinct from the accepted but unused native `exhaustiveness` argument.

## Caveats

This two-molecule smoke validates the provider, caching and budget behavior. It does not by itself prove that an MCTS run invoked the callback or improved molecules; the parent runner must report nonzero actual attempts/docks for such a claim. PoseBusters is not part of this energy callback, so saved poses still need the separate physical-evaluation protocol for validity claims. OpenCL event timings are not used for a speed claim. The cache is in-memory and bound to one provider instance; use a fresh output directory per experiment, and do not treat its audit file as a resume cache.

## Reproduce

```bash
uv run pytest -q molmetal/tests/test_pocket_docking_reward.py
HIP_VISIBLE_DEVICES=0 uv run python - <<'PY'
from pathlib import Path
from molmetal.scripts.prepare_crossdocked_receptor import manifest_pair
from molmetal.molmetal_lam.sbdd_env.pocket_docking_reward import PocketDockingReward, PocketDockingRewardError

receptor, ligand = manifest_pair(Path('molmetal/data/crossdocked100_manifest.csv'), 'test_001')
reward = PocketDockingReward(
    receptor, ligand, 'molmetal/reports/NEW_POCKET_REWARD_PREFIX', seed=42,
    gpu_config='molmetal/configs/amd_gpu_docking.json', max_evaluations=2,
    n_poses=1, exhaustiveness=1,
)
for smiles in ['CCO', 'OCC', 'c1ccccc1', 'CCC']:
    try:
        print(smiles, reward(smiles))
    except PocketDockingRewardError as exc:
        print(smiles, 'FAILED', str(exc))
print(reward.report()['summary'])
PY
```

Provider source/config/input hashes and per-engine binary/kernel/input/output hashes accompany the report. All native receptor records remain unchanged.
