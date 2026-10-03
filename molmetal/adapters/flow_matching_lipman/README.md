# `flow_matching_lipman/` — Lipman CFM + 6 shim fixes

Wraps facebookresearch/flow_matching (Lipman et al., ICLR 2023). Provides the
**conditional OT path** (`AffineProbPath` + `CondOTScheduler`) and the
**midpoint ODE solver** for sampling 3D molecular geometries. The velocity
field is our own `EGNNVelocityField` reusing the QM9 EGNN from `models/`.
The pocket-conditioning encoder (`PocketEncoder`) injects per-pocket atom
features into v_θ — the **only** change from the canonical SBDD-FM
(TargetDiff / DiffSBDD / Pocket2Mol do the same).

The sub-package ships **6 R17 shim fixes** that harden the upstream clone:
logits clamp (M13), LR warmup integration, `ModelWrapper` keyword fallback
(M16), flat-clone loader auto-detect (M17), reference solver Euler/midpoint
fallback (M15), and AMP BF16 autocast.

## Key files

| File | Role |
|------|------|
| `__init__.py` | Main adapter — `EGNNVelocityField`, `PocketEncoder`, shim fixes. |
| `amp.py` | AMP BF16 autocast shim (VRAM fix). |
| `connectivity_decoder.py` | Decode bonds from atomic coords + atom types. |
| `_reference/` | Vendored facebookresearch/flow_matching (CC-by-NC). |
| `_reference_loader.py` | Flat-clone auto-detect (M17). |

## Usage

```python
from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
adapter = LipmanFlowMatchingAdapter.from_pretrained("checkpoints/egnn_qm9.pt")
mol = adapter.sample(num_samples=64, pocket="pocket.pdb", n_steps=200)
```

```python
from molmetal.adapters.flow_matching_lipman import PocketEncoder
enc = PocketEncoder(hidden_dim=128, n_layers=2)
bias = enc(pocket_coords, pocket_atomic_numbers)  # (B, hidden_dim)
```

## Common pitfalls

* The adapter assumes **midpoint ODE**; Euler-only sampling diverges on
  metallo-ligands (M15).
* `from_pretrained` picks the **flat-clone** checkpoint format only when
  `_reference_loader.py` finds the marker file (M17).
* Atom-type logits must be **clamped to ±50** before softmax (M13) — without
  it the sampler NaNs at step ~30.
* `PocketEncoder` uses **distance features**, not raw coords — preserves
  SE(3) equivariance.
* Mechanism ↔ metric cross-refs: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`
  for **M6** (Lipman CFM), **M12** (vocab), **M13** (logits clamp),
  **M15-M17** (shim fixes).
* Tests: `pytest molmetal/tests/test_lipman_spatial_contract.py test_pocket_conditioned_lipman.py -q`.
