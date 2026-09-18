# Audit: RxnFlow (Seo et al. ICLR 2024, arXiv:2410.04542)

Path: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/RxnFlow/`
Our relevant code: `molmetal/adapters/flow_matching_lipman/__init__.py` (joint
bond-head + atom-type path-(b) plan, currently path-(c) λ-only as default)

## (1) What we already have hand-rolled that is redundant

- **Synthesizable generative model over a building-block + reaction-template
  action space**: RxnFlow (`src/rxnflow/`) is exactly this. They ship:
  - `src/rxnflow/base/task.py::BaseTask` — class to inject custom reward;
  - `src/rxnflow/base/trainer.py::RxnFlowTrainer` — GFlowNet online trainer;
  - `src/rxnflow/base/generator.py::RxnFlowSampler` — sampling;
  - `src/rxnflow/algo/synthetic_path_sampling.py` — synthetic pathway
    forward sampling from {building block + reaction template} pairs;
  - 13 uni-molecular + 58 bi-molecular templates from Cretu et al;
  - ZINC22 fragment library + Enamine REAL library.
- Our hand-rolled path-(b) (joint atom-type+bond-tensor FM) would NOT be
  synthesizable out-of-the-box — RxnFlow explicitly builds 1M+ building
  blocks × 100+ reactions → discrete trajectories and is what gives
  their synthesizability guarantee.
- Our `closure.py` (closure-theorem) is a different concern: it
  proves reachability of *cisplatin + 5-click reactions* under typed
  reductions; RxnFlow solves the same problem operationally with templates.

## (2) What is actually different / better in our hand-rolled code

- **Conditional 3D FM**: RxnFlow does NOT condition on a 3D pocket; it's a
  sequence-of-actions GFlowNet over SMILES building blocks, then a separate
  UniDock docking step. No equivariant 3D conditioning.
- **Per-pocket conditioning**: our `reference_ligand_resolver` plus
  pocket-conditioned `velocity_net` (with EGNN pocket embedding) is
  structurally different — it is 3D geometry-aware.
- **Metal coordination prior**: RxnFlow has no concept of metal seed;
  we have `MetalGeometryPrior` + `AquaExchange` + `MetalLigandExchange`
  SMARTS.
- **Decision tree**: per TODO-21 (Lambda × CFM coupling, deferred) +
  TODO-24 (CFM path-b), our path-(c) λ-only is the current default,
  NOT path-(b). RxnFlow adoption would be a *third* path — a hybrid
  λ-action + GFlowNet reward channel, not a replacement of our path-(b).

## (3) Concrete 3-line patch plan (file + lines + import)

```
# DO NOT wholesale-replace our path-(b).
# INSTEAD, use RxnFlow as an *oracle* for synthesizability scoring:
import sys; sys.path.insert(0, ".../references/RxnFlow/src")
from rxnflow.base import BaseTask, RxnFlowSampler
from rxnflow.tasks import QEDTask, UnidockVinaTask  # templates
# Then in molmetal/reward/synth_oracle.py (new), wire BaseTask.compute_obj_properties()
# to consume our pocket-conditioned candidate SMILES and return a synthesizability score.
```
Caveat: RxnFlow requires `torch>=2.5`, `torch_geometric`, and UniDock
binding for Vina — heavy install. We don't have UniDock on RX 7800 XT
and our SMILES-only path wouldn't even need docking.

## Verdict

**PARTIAL REUSE — adopt RxnFlow as a synthesizability oracle.** Their
GFlowNet over reaction templates is genuinely orthogonal to our work —
they answer "is this synthesizable?" while we answer "is this 3D-pocket
compatible + metal-geometry correct?". Wire RxnFlow as a 6th oracle
column (r_rxnflow channel) in `RewardAggregator`, not as a replacement
for path-(b) or path-(c). Path-(b) is structurally different (3D FM
with bond-head); RxnFlow is 2D graph GFlowNet. The two could compose
later as a hybrid λ-prior × GFlowNet-action sampler.