# EGNN Critique — `metal_hybrid_v3.py` + `egnn_predict.py`

## 1. Is `_CoordRefineHead` really EGNN?
**No.** It borrows the *shape* of EGNN's coord update
(`x_i' = x_i + Σ (x_j − x_i) · φ(h_i,h_j,r_ij)`) but breaks two of its
defining properties: (a) the gradient is **detached** from the encoder
(EGNN's whole point is co-evolving `h` and `x`), and (b) outputs are
clipped to **±0.3 Å** (a hard prior, not learned equivariance).
The real Satorras 2021 EGNN layer is `h ← h + Σ_j MLP([h_i,h_j,r_ij])`,
`x ← x + Σ_j (x_j − x_i) · φ_e([h_i,h_j,r_ij²])` — fully differentiable,
no clipping. The bottom layer in `egnn_rocm.py:EquivariantGraphConv`
already does this correctly; the V3 head is just a stripped-down
*post-hoc* regressor that reuses EGNN-flavored notation.

## 2. Is `EGNNPredictor` really EGNN & equivariant?
**Substantially yes** at the layer level — `EquivariantGraphConv`
follows Satorras Eq.3 with radial basis on raw distance (only 1-D,
not Gaussian-expanded). It is **not** invariant: it returns per-atom
features (equivariant w.r.t. permutation, equivariant w.r.t. SE(3)
for `x`, but `h` is invariant by design — that's correct). To prove
rotation invariance on `(Σ_atoms h_i, x_mean)` pool, run a random
SO(3) test (assert `|f(Rx) − f(x)| < 1e-5` after sum-pool).

## 3–6. Concrete upgrades
- **Replace `_CoordRefineHead`** with the `EGNNLayer` below (drop-in,
  no detach, no clip).
- **Angle basis** (SchNet/GemNet style): add Gaussian expansion
  `rbf(d) = [exp(-(d-μ_k)²/σ²)]_{k=1..K}` and 3-body angle features
  `cos θ_ijk = (x_i−x_j)·(x_k−x_j) / (|…|·|…|)` via k-NN triplets.
- **Swish/SiLU** — already used inside `EquivariantGraphConv`, replace
  remaining `nn.ReLU` in `output_proj`/`fusion_gate`/`head_mlp`.
- **Soft constraint** — multiply the coord update by `tanh(·)` (still
  equivariant — `tanh` is a scalar acting on invariant magnitudes).

## Files touched: `metal_hybrid_v3.py`, `egnn_predict.py`,
`adapters/egnn_rocm.py`.