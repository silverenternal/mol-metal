# Audit: REINVENT4 (MolecularAI 2024, J. Cheminformatics)

Path: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/REINVENT4/`
Our existing wrapper: `molmetal/adapters/reinvent4_subprocess_adapter.py`
(subprocess wrapper around `reinvent -l sampling.log sampling.toml`)

## (1) What we already have hand-rolled that is redundant

- **Multiproperty score composition**: we compose
  QED / SA / MolWt / reward-platinai in Python in
  `molmetal/molmetal_lam/reward/reward_aggregator.py`. REINVENT4 does
  exactly this via the scoring framework (`reinvent/scoring/`) with
  aggregators (`aggregators/`) and per-component transforms
  (`transforms/`).
- **Reinvent model load + sampling**: we shell out to a TOML config and
  parse the CSV output. REINVENT4 has a direct Python API:
  `reinvent.runmodes.create_adapter.create_adapter(...)` returns
  `(adapter, model_dict)` and `reinvent.runmodes.RL.reward.run_reinvent(...)`
  is callable from Python.
- **Scoring component framework**: `reinvent/scoring/scorer.py:Scorer` is
  a plug-in registry for `MolWt`, `QED`, `SAscore`, `Tanimoto`,
  `custom_alerts`, etc. — we duplicate most of these as plain functions.

## (2) What is actually different / better in our hand-rolled code

- **Subprocess isolation**: invoking `reinvent` as a subprocess avoids
  contaminating our `molmetal_lam` process state — REINVENT4's main
  installs a CLI entrypoint that imports ~80 sub-modules and reads
  `prior_registry.py` (registry-driven). In-process reuse would require
  us to vendor this.
- **Offline graceful fallback**: our adapter already passes
  `graceful_offline=True` (per WF-Extra-2 task #404) so subprocess
  failures don't crash RewardAggregator; in-process API calls would
  raise and break the test loop.
- **5-SMILES batch smoke + μ_wall_clock=4.06s per SMILES** (task #414):
  our subprocess harness is the measured baseline. Replacing it requires
  re-measurement; not free.
- **Multiproperty TOML**: `molmetal/configs/reinvent_multiproperty_amd.json`
  is a verified config with ROCm-friendly weights; in-process reuse
  requires rebuilding this config in REINVENT4's TOML schema.

## (3) Concrete 3-line patch plan (file + lines + import)

```
# molmetal/adapters/reinvent4_subprocess_adapter.py:1-50 (subprocess entry)
# OPTION A (minimal): keep subprocess, but call REINVENT4 Python API for the
#                     scoring side ONLY (avoids TOML parsing overhead):
import sys; sys.path.insert(0, ".../references/REINVENT4")
from reinvent.scoring import Scorer
from reinvent.scoring.config import ScoringConfig
scorer = Scorer(scoring_config=ScoringConfig.from_dict(...))
# then in score_batch(): return scorer([mols]) instead of subprocess.
# OPTION B (full API): from reinvent.runmodes import setup_sampler; setup_sampler(...).sample(...)
```
Risk: REINVENT4 imports `torch>=2.1` and `rdkit>=2024.3`; we already have
both, so no version conflict.

## Verdict

**PARTIAL REUSE — extract the scoring component.** Keep the subprocess
wrapper as the verified batch path (task #414), but factor out the
multiproperty composition into REINVENT4's `Scorer` API for clarity and
to inherit the registry of standard transforms (e.g. `RunningStats`,
`ParallelScore`). Do not rewrite the sampling side — that's where the
TOML + checkpoint dependencies live and we don't gain much over our
4.06 s/SMILES baseline.