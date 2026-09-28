# `models/` — EGNN + YuelBond + velocity net

The geometric + cheminformatic backbone. Houses the SE(3)-equivariant
`EGNNPredictor` (Layer 5 of the MLC framework), the D-MPNN encoder
(`DirectedMPNN`), the YuelBond learning-based bond head (replaces the
naïve distance-threshold decoder), the pocket-conditioned velocity field
(`MetalHybridV4`), and the multi-task loss for MetalCytoToxDB.

## Key files

| File | Role |
|------|------|
| `egnn_predict.py` | `EGNNPredictor`, `EGNNPredictorConfig` (M5, SE(3)-equivariant). |
| `velocity_net.py` | Velocity field used by Lipman CFM (M5). |
| `dmpnn.py` | `DirectedMPNN` — message-passing on directed bonds (D-MPNN). |
| `bond_head.py` | YuelBond learning-based bond decoder (M7). |
| `connectivity_gumbel.py` | Differentiable bond sampler (Gumbel-Softmax). |
| `cross_attention_fusion.py`, `cross_attention_fusion_v3.py` | Pocket × ligand cross-attn. |
| `fusion.py` | `FusionMLP` — late fusion of D-MPNN + EGNN features. |
| `metal_hybrid.py`, `metal_hybrid_v2.py` … `_v4.py` | Hybrid model lineage (T1-T3). |
| `loss.py` | `MetalCytotoxLoss` — multi-task regression loss. |
| `_scatter.py` | Triton-accelerated `scatter_sum`. |
| `symbolic_reward/` | SR backend for `reward/`. |

## Usage

```python
from molmetal.models import DirectedMPNN, EGNNPredictor, MetalHybridModel
mpnn = DirectedMPNN(MPNNConfig(hidden_dim=256, depth=3))
egnn = EGNNPredictor(EGNNPredictorConfig(hidden_dim=128, n_layers=4))
model = MetalHybridModel(mpnn=mpnn, egnn=egnn, fusion_dim=64)
y_hat = model(smiles_batch, coord_batch)   # → (B,) pIC50 prediction
```

```python
from molmetal.models.bond_head import YuelBondHead
head = YuelBondHead(in_dim=128, n_bond_types=4)
bond_logits = head(node_feats, edge_index)  # (E, n_bond_types)
```

## Common pitfalls

* EGNN updates use **distance features only** — passing raw coords breaks
  equivariance.
* `YuelBondHead` must be paired with **connectivity_gumbel** for
  differentiable bond sampling during training.
* `MetalHybridV4` is the only version that supports **temporal holdout**;
  v1-v3 silently leak split info.
* `scatter_sum` falls back to torch_scatter when Triton is unavailable.
* Mechanism ↔ metric cross-refs: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`
  for **M5** (EGNN), **M7** (YuelBond).
* Tests: `pytest molmetal/tests/test_egnn_predictor_device.py test_metal_hybrid_v4.py -q`.