# `sbdd_env/` — Structure-Based Drug Design evaluators

Wraps every chemistry-quality evaluator used by the MLC pipeline: Vina docking
(CPU + GPU), SA score (Ertl-Schuffenhauer), QED (Bickerton), PoseBusters
geometric filter, REINVENT4 multi-property scorer, retrosynthesis gates,
and competitive baselines (TargetDiff, Pocket2Mol, DiffDock, FlowDock,
REINVENT4, AiZynthfinder, SynFlowNet, SyntheMol). Also handles pocket
voxelization and 3D pose chemistry checks.

## Key files

| File | Role |
|------|------|
| `vina_adapter.py` | CPU Vina docking (m3). |
| `vina_gpu_adapter.py` | ROCm/GPU Vina backend. |
| `sa_score.py` | Ertl-Schuffenhauer SA score (M18, m5). |
| `qed_scorer.py` | Bickerton QED (m6). |
| `posebusters_adapter.py` | PoseBusters geometric filter (M19, m8). |
| `reinvent_wrapper.py` | REINVENT4Scorer + voxelization (m5/m6/m7). |
| `retrosynthesis.py`, `synthesis_gate.py` | Retro + synthesizability gate. |
| `pocket_docking_reward.py` | Combined pocket-docking reward channel. |
| `voxelization.py` | `pocket_to_spatial_tiles` — PDB → 3D voxel grid. |
| `pic50_predictor.py` | LightGBM pIC50 oracle (calibrated). |
| `protein_metal_binding.py` | BioPython NeighborSearch (M20). |
| `synflownet_env.py`, `syntemol_reactions.py` | Baseline generation envs. |
| `targetdiff_adapter.py`, `pocket2mol_adapter.py`, `diffdock_adapter.py`, `flowdock_adapter.py`, `reinvent4_adapter.py`, `aizynth_adapter.py`, `flowr_adapter.py` | SOTA-baseline adapters. |
| `metrics_v2.py`, `per_residue_diversity.py` | SBDD metrics (m3/m4/m7/m8/m9). |

## Usage

```python
from molmetal_lam.sbdd_env import REINVENT4Scorer
scorer = REINVENT4Scorer(weights={"qed": 0.4, "sa": 0.3, "binding": 0.3})
scores = scorer.batch_score(["CCO", "c1ccccc1O"])  # → np.ndarray[B]
```

```python
from molmetal_lam.sbdd_env.voxelization import pocket_to_spatial_tiles
tiles = pocket_to_spatial_tiles("pocket.pdb", grid_size=32)  # → (32,32,32,C)
```

## Common pitfalls

* Vina needs a **prepared receptor** (`prepare_receptor`); raw PDB fails.
* SA score is **1-10 (lower=better)** — convert to `sa_unit = (10 - sa)/9`.
* PoseBusters requires **RDKKit `MMFF94s` cleanup**; disabled it diverges ~12%.
* `protein_metal_binding` uses BioPython `NeighborSearch` (PLIP replacement).
* Mechanism ↔ metric cross-refs: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`
  for **M18** (SA), **M19** (PB), **M20** (PLIP→BioPython).
* Tests: `pytest molmetal/tests/test_vina_adapter.py test_posebusters_adapter.py -q`.