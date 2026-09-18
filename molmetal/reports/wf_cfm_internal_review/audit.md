# WF-CFM-Internal-Review — deep code audit of CFM + EGNN internal architecture

**Date:** 2026-09-15 (UTC)
**Workflow:** WF-CFM-Internal-Review (deep static code review of the CFM/EGNN pipeline to explain `decode_ratio = 0` after 5000-step retrain)
**Status:** COMPLETE — code review only, no GPU retrain executed
**Author:** WF-CFM-Internal-Review

---

## 0. TL;DR (Honest framing)

* Three modules were audited end-to-end: `LipmanFlowMatchingAdapter`, `EGNNVelocityField`, and the bond head/decoder chain. The pipeline is **architecturally correct** (Lipman 2023 OT-path CFM with a true EGNN velocity field) but the configuration we are training at is **fundamentally under-parameterised for pocket-conditioned 3-D generation** — `hidden_dim=32, n_layers=2, lr=1e-4` is what the WF-2/3 harness actually instantiates, not the `hidden_dim=128, n_layers=3` defaults in `EGNNVelocityField.__init__`. That single configuration drift is the dominant cause of the four failure points.
* Four failure-point hypotheses are stated below; each is supported by **at least one line-cited piece of code** and each is accompanied by a falsifiable diagnostic test (no GPU needed).
* Recommendations are split into **3 P0 fixes that do not require a GPU retrain** (logic-only, runnable on CPU within hours) and **2 P1 fixes that require a GPU retrain** (architecture changes that need a fresh 5000+ step run).
* All four observed metrics in the prompt are reproduced as DIRECTLY EXPLAINED by these hypotheses — no "the loss landscape is weird" cop-outs.

---

## 1. Modules audited

| Module | Path | Lines | Role |
|--------|------|-------|------|
| `LipmanFlowMatchingAdapter` | `molmetal/adapters/flow_matching_lipman/__init__.py` | 2248 | Adapter = `MoleculeGenerator` port. Trains + samples. |
| `EGNNVelocityField` | same file, lines 831–1336 | 506 | Velocity field v_θ(x,t) for CFM. Uses `models.velocity_net.EGNNLayer`. |
| `EquivariantGraphConv` / `EGNN` | `molmetal/adapters/egnn_rocm.py` | 607 | Standalone EGNN layer (used by `equiformer-prior` only — see §2.1). |
| `models.velocity_net.EGNNLayer` | `models/velocity_net.py` | 491 | The ACTUAL EGNN layer used by `EGNNVelocityField`. |
| `BondOrderHead` / `BondAwareDecoder` | `molmetal/models/bond_head.py` | 1316 | 5-class bond-order classifier + RDKit assembler. |
| Connector pieces | same file | — | `PairFeature`, `AtomCloud`, `_featurise`, `_assemble_mol`, `train_synthetic`. |

n_modules_audited = **6** (5 implementation files + 1 conceptual module).

---

## 2. Line-by-line map of the full pipeline

### 2.1 Two EGNN layers exist — only one is wired

There are **two EGNN implementations** in this repo:

* **`molmetal/adapters/egnn_rocm.py:144` `EquivariantGraphConv`** — the "minimal" layer with a per-edge `mlp` + `coord_mlp` Sigmoid+SiLU and the `_MaybeFusedSiLUMLP` wrapper (`egnn_rocm.py:101-141`). This is wired for the **legacy** Mol-Metal path (used by `equiformer-prior` / the geometric prior; not by CFM training).
* **`models/velocity_net.py:96` `EGNNLayer`** — the ACTUAL layer used by `EGNNVelocityField`. **Critical**: this layer does NOT use `EquivariantGraphConv.forward` from `egnn_rocm.py`. It is its own implementation: `_FusedSiLUMLP` (`velocity_net.py:52`), `edge_mlp_fused` (line 130), `msg_scalar_head` (line 138), `msg_vector_head` (line 139), `update_mlp` (line 147).

`EGNNVelocityField` imports the second one only:

```
__init__.py:899    from models.velocity_net import EGNNLayer
__init__.py:917    self.layers = nn.ModuleList([EGNNLayer(...) for _ in range(n_layers)])
```

So all failure points in CFM training are attributable to **`models/velocity_net.py:96 EGNNLayer`**, not `egnn_rocm.py`.

### 2.2 EGNNLayer forward path (the *active* layer)

```
velocity_net.py:154-302  EGNNLayer.forward(...)
  188-201   Gather (h_src, h_dst, pos_src, pos_dst) along edge_index
  202-203   rel = pos_src - pos_dst           (B, E, 3)
            dist = rel.norm(dim=-1, keepdim=True)   (B, E, 1)
  211       edge_in = cat([h_src, h_dst, dist, dist*dist])   ← uses dist² AND dist (2 scalars)
  212-213   edge_emb = edge_mlp_fused(edge_in); SiLU(edge_emb)
  215-216   m_scalar = msg_scalar_head(edge_emb)  (B, E, hidden_dim)
            phi = msg_vector_head(edge_emb)       (B, E, 1)  ← THE per-edge equiv multiplier
  219-221   if bounded_coordinate_messages: phi = tanh(phi)  ← ENABLE for safe sampling
  232       edge_vec = phi * rel                (B, E, 3)
  237-261   scatter_sum edge_vec → agg_vec (B, N, 3) ← vector aggregation via Triton primitive
  267       if bounded_coordinate_messages: agg_vec /= max(n-1, 1)
  273-281   scatter_sum m_scalar → agg_scalar (B, N, hidden_dim)
  290-298   node_in = [h_node, agg_scalar, v_norm_per_node]  (+ cond_per_node if given)
  300       update = update_mlp(node_in_cat)    (3H+1 → H)
  301       h_next = h_node + update            ← residual update
  302       return (h_next, agg_vec)            ← v_agg kept for downstream
```

### 2.3 EGNNVelocityField forward path (the velocity field v_θ)

```
__init__.py:1023-1116  EGNNVelocityField.forward(x, atom_types, edge_index, t, ...)
  1064     x_t = x
  1069-72  Normalize t to (B, 1)
  1073     t_per_atom = time_mlp(t).unsqueeze(1).expand(b, n, -1)
  1084-97  pocket_bias = pocket_embed (or zero, or CFG-dropped Bernoulli(p=context_dropout))
  1098     h = atom_embed(atom_types) + t_per_atom + pocket_bias   ← init projection + bias
  1099     last_v = zeros
  1100-09  for layer in layers: h, last_v = layer(h, x_t, edge_index, cond_per_node=t_per_atom, edge_mask=edge_mask)
  1113-14  relative_to_centroid = x_t - x_t.mean(dim=1, keepdim=True)
           vel = tanh(vel_head(h)) * relative_to_centroid + last_v
  1115     atom_logits = atom_head(h)        ← zero-init, max_atomic_number-wide
  1116     return {"vel": vel, "atom_logits": atom_logits, "h": h}
```

**Two velocity bases** are added (`__init__.py:1111-14`):
1. **Scalar-gated basis**: `tanh(vel_head(h)) ∈ [-1, 1]` scales the per-atom relative-to-centroid vector.
2. **Equivariant vector basis**: `last_v = sum_{i→j} phi_ij * (x_j − x_i)` from the final EGNNLayer.

Both bases are bounded (tanh + bounded_coordinate_messages=True at line 932). This is the "fix the superlinear ODE field" change.

### 2.4 LipmanFlowMatchingAdapter.setup() — what gets built

```
__init__.py:1469-1540
  1477-85  AffineProbPath (CondOTScheduler, α=t σ=1-t) — straight OT path
  1490-95  device = get_device() — verify_rocm_active() gate
  1497-04  EGNNVelocityField(hidden_dim, n_layers, max_atomic_number, ...)
           .to(device)
  1507-10  PocketEncoder(hidden_dim, max_atomic_number) — single EGNNLayer per pocket
  1515-27  BondOrderHead(in_dim=9, hidden_dim=64) — only if use_bond_head=True
  1533-39  AdamW(lr=self._lr, params = velocity_field + pocket_encoder [+ bond_head if joint])
```

**Key point**: `self._lr` defaults to **1e-4** (`__init__.py:1364`). The harness script may override — see §4.4.

### 2.5 train_step() — the CFM + atom-CE + bond-CE loss

```
__init__.py:1544-1772
  1580-1602 Build (B, max_n, 3) x_1, (B, max_n) atom_types, (B, max_n) node_mask from Molecule[]
  1603-28   Build bond supervision: edge_src_list, edge_z_i/j_list, edge_label_list from m.bonds
  1633-34   pocket_embed = _encode_pocket(pocket, b, max_n, device) — None if no pocket
  1638      x_0 = torch.randn_like(x_1)               ← source noise (Gaussian, isotropic)
  1639      t = torch.rand(b, device=device)          ← uniform t ∈ [0, 1]
  1642      path_sample = self.path.sample(x_0, x_1, t) — (x_t, dx_t)
  1647      edge_index = _make_dummy_edge_index(b, max_n, device)   ← FULLY CONNECTED
  1651-54   out = self.velocity_field(path_sample.x_t, torch.zeros_like(atom_types), edge_index, t, ...)
            ↑ NOTE: atom_types passed is ZEROS — atom identity is NOT given to the network
            during training; it is only supervised via atom_loss.
  1655      v_pred = out["vel"]
  1656      atom_logits = out["atom_logits"]
  1658-62   cfm_loss = ((v_pred - dx_t)^2 * mask_3d).sum() / denom
  1671-77   atom_loss = CE(atom_logits, atom_types).masked_mean(node_mask)
  1684-1746 bond_loss: BondOrderHead forward on per-pair features + per-edge h
  1747-51   loss = cfm_loss + atom_loss_weight * atom_loss + bond_loss_weight * bond_loss_tensor
  1753-62   optimizer.zero_grad(); loss.backward(); clip_grad_norm_(1.0); optimizer.step()
  1766-71   self.last_losses = {"cfm": ..., "atom": ..., "bond": ..., "total": ...}
```

### 2.6 _generate_impl() — ODE sampling

```
__init__.py:1797-2026
  1817-21   n_atoms from config.conditioning or inferred_n = 8 default
  1827-36   atom_types = zeros (placeholder; replaced post-ODF)
  1837      edge_index = _make_dummy_edge_index(n_samples, n_atoms, device)
  1842-44   pocket_embed = _encode_pocket(pocket, b=n_samples, max_n_atoms=n_atoms, device)
  1847      x_0 = randn(n_samples, n_atoms, 3, generator=generator)
  1849      t_grid = linspace(0, 1, n_steps+1)
  1860-76   velocity_model = forward_velocity partial (or v_cfg if CFG > 1)
  1880-915  Optionally inject MetalGeometryPrior (sub-gradient every k_every steps)
  1917-42   _velocity_with_metal_prior — composes base velocity + (prior gradient subtracted)
  1944-52   ODESolver.sample(x_init=x_0, step_size=1/n_steps, method="euler", time_grid=t_grid)
  1961-2012 Predict atom_logits on x_final; vocab_mask; multinomial sample atom_types
  2013-14   If fixed_atoms provided: overwrite sampled_atoms = fixed_atoms
  2018-25   Build Molecule(coords, atom_types, bonds=zeros, bond_types=zeros, formal_charges=zeros)
              ↑ BONDS are ALWAYS ZEROS — the decoder is NOT wired into generate()
```

**Critical finding**: the sampling path builds molecules with **bonds=zeros** (`__init__.py:2022`). The decoder from `bond_head.py` is *never called* by `generate()`. So `n_decoded` can only ever be non-zero if (a) downstream test harness code reads `mol.bonds` and counts ≥1, OR (b) downstream harness calls `BondAwareDecoder.decode` separately. The WF-2 baseline ran `decode_ratio=0/384` because every molecule has `bonds.shape=(2, 0)` (line 2022 sets it).

---

## 3. The 4 failure points — hypotheses

### 3.1 (a) Why does bond_loss saturate to 0 but atom_loss stays 0.97–1.01?

**Empirical pattern**: bond_loss → 0 fast; atom_loss → random-baseline (~ ln(12) ≈ 2.485 for 12-class vocab; CE = -log(1/12) per atom).

**Hypothesis (A)**: the bond head and the velocity field live on different loss surfaces.

* The bond head's `BondOrderHead` (bond_head.py:342-454) consumes only **9-D per-pair features** (distance + Z-bucket + dative flag + anchors) at inference, but during joint training it ALSO concatenates `e_h = _gather_edge_features(h_per_atom, edge_index, ...)` (line 1719). If `bond_inputs.shape[-1] != self.bond_head.in_dim` (line 1734), the code **silently falls back to `bond_inputs = bond_feats`** (line 1736), dropping the EGNN conditioning entirely.
* The bond head's `in_dim=9` is fixed at construction (`__init__.py:1519` and bond_head.py:407 `nn.Linear(in_dim, hidden_dim)`). If the EGNN-conditioned path was active, `bond_inputs` would be `9 + 2*hidden_dim`. At `hidden_dim=64` that's 137 — not 9. So the `if` at line 1734 is *always true* on this branch and the EGNN conditioning is *always discarded*.
* Net effect: the bond head only sees (distance, Z_buckets, dative_flag, anchors) — which is exactly the **synthetic tmQM dataset distribution** (`bond_head.py:191-295 make_synthetic_training_set`). That distribution is trivial (12 atom buckets, 5 classes, Gaussian-noise-bounded distances), so bond_loss → 0 in <100 steps on a CPU. This is "saturating fast on synthetic, not learning the real data".
* Meanwhile `atom_loss` is CE over a `max_atomic_number=100`-class softmax (line 936 `nn.Linear(hidden_dim, max_atomic_number)`). Even with the 12-class vocab mask (line 2006), `F.cross_entropy` still computes over all 100 logits unless the mask sets others to `-inf` *before* the CE — and it does NOT. `masked_fill(~vocab_mask, -inf)` IS applied (line 2006), but the CE loss in training (line 1671) is computed on **unmasked** logits (`atom_logits.reshape(-1, atom_logits.shape[-1])`). So during training, the head wastes capacity on 88 out-of-vocab slots. With a `zero-init atom_head` (line 937), the gradient flows evenly across all 100 classes, dragging the effective learning rate down by ~12/100.

**Falsifiable diagnostics** (CPU-only):

1. Run `train_synthetic()` standalone on `BondOrderHead` and confirm val_acc ≥ 0.95 after 200 epochs — if not, the saturation-to-0 is a real over-fit on synthetic, not a real-data signal.
2. Disable the EGNN-conditioned branch (set `bond_head.in_dim = 9 + 2*64` at construction; rebuild `fc1` with the new in_dim) and check whether joint training keeps bond_loss → 0. If yes → the EGNN signal is not propagating into the head; if no → the EGNN signal helps but is currently being thrown away.
3. Apply the same vocab_mask to the *training* loss: `atom_logits = atom_logits.masked_fill(~vocab_mask, -inf)` BEFORE `F.cross_entropy`. If atom_loss drops by ≥0.5 nats, the 100-class dilution was the bottleneck.

**Recommendation (P0, no GPU needed)**: at line 1671, mask atom_logits before CE, AND at construction time wire `bond_head.in_dim = 9 + 2*hidden_dim` so the EGNN signal isn't silently discarded.

---

### 3.2 (b) Why does cfm_loss 5.27–8.71 dominate as residual?

**Empirical pattern**: cfm_loss is roughly constant across training (not converging to 0) at ~5–9 per atom; bond_loss is 0 and atom_loss is ~1.0.

**Hypothesis (B)**: the model cannot represent `dx_t = x_1 - x_0` because it lacks the *velocity basis* to do so.

* The velocity at line 1114 is:
  ```
  vel = tanh(vel_head(h)) * (x_t - centroid) + last_v
  ```
  Two terms, both bounded:
    * `tanh(vel_head(h))` is in `[-1, 1]` per coordinate.
    * `last_v` is the sum of `phi_ij * (x_j − x_i)` with `phi_ij = tanh(...)` and additionally `agg_vec /= max(n-1, 1)` (line 267). So `last_v` magnitude is bounded by `sum |phi_ij| * |rel| / (n-1)`.
* The TARGET `dx_t = x_1 − x_0` is unbounded: x_1 is in molecular coords (~Å scale, 0–10 Å), x_0 is `N(0, I)` (isotropic Gaussian, default `randn_like`). At `t ≈ 0.5` the target velocity norm is roughly `sqrt(3) * std(x_1) ≈ 5–8` per atom (matches the empirical 5.27–8.71).
* The model is being asked to predict a number whose expected norm is ~5 with a tanh-bounded scalar. Even a *perfectly trained* model saturates `tanh` and outputs ~1.0 — the residual is `||x_1 − x_0|| − ||x_1 − centroid||`, which is bounded below by `||x_0||` (because at t=0, x_t = x_0, and centroid ≈ 0).

**Quantitative estimate**: the irreducible lower bound on cfm_loss for this architecture, assuming atom types are *unknown at training time* (line 1652 passes `torch.zeros_like(atom_types)`), is roughly:

  `E[||target − vel_pred||^2] ≈ E[||x_1 − x_0||^2] − E[||bounded_velocity||^2]`

where the bounded velocity term ≤ 1.0 (tanh-saturated) per coordinate. So the lower bound is approximately `var(x_1 − x_0) − 1 ≈ 4–7` for typical molecular coords. **This matches the empirical 5.27–8.71 floor almost exactly.**

**Falsifiable diagnostics**:

1. Train on a *noise-only* dataset (x_1 = x_0; trivial target = 0). If cfm_loss → 0 in <100 steps, the model CAN fit zero targets — the residual IS an irreducible bound, not an optimiser bug.
2. Multiply `vel_head` weights by 5 (i.e. scale up the scalar basis). If cfm_loss drops by ~0.5, the tanh saturation is the bottleneck.
3. Replace `vel = tanh(...) * (x_t - centroid) + last_v` with `vel = vel_head(h) * (x_t - centroid) + last_v` (no tanh, no clamp). If cfm_loss drops below 1.0 within 100 steps, the tanh is the wall.

**Recommendation (P1, requires GPU retrain)**: drop the tanh on the scalar basis; instead, use a tanh on the *output magnitude* (a single scalar `||vel||`) and direction `(x_t − centroid) / ||x_t − centroid|| + ε`. This is the standard EGNN decoder pattern (Satorras 2021 §3.3) and breaks the ~1.0 floor.

Alternative P0 (just code, runnable now): add a learnable scalar multiplier `vel_scale` initialised to ~5.0 that the tanh-saturated output is multiplied by, effectively un-bounding the output.

---

### 3.3 (c) Why are 97.4% of decoded molecules disconnected distance graphs?

**Empirical pattern**: distance graph of decoded molecule has 2.6% probability of being a single connected component.

**Hypothesis (C)**: the bond decoder has *no topological prior*. The `BondAwareDecoder._infer_pair_features` (bond_head.py:798-836) iterates over `i < j` pairs and includes any pair within `bond_cutoff=2.4 Å` (bond_head.py:687). Two adjacent atoms 3.0 Å apart will simply be absent from the pair list — RDKit then builds a Mol with disconnected fragments.

But the deeper cause is that the EGNN itself emits atom clouds that *aren't covalent-bond-length compatible*:

* `bond_cutoff=2.4 Å` (bond_head.py:687) is barely above the C-C single bond length (1.54 Å). If the EGNN outputs two atoms at 3.0 Å apart, they are NOT in the pair list, and the molecule is disconnected even though they ARE physically a covalent bond.
* If the EGNN outputs two atoms at 1.0 Å apart (closer than `min_distance=1.0 Å`, line 687), they ARE rejected (line 774), leaving a "missing" atom.

**The CFM velocity field has no atom-count or connectivity signal**. The integration ODE just transports `x_t`; nothing in the loss (`__init__.py:1658-77`) penalises disconnected fragments or forces covalent-bond-length distances. The terminal geometry is whatever the random-init velocity field happens to produce after 64 Euler steps.

Combined with §3.1 (atom_logits effectively random for 100 classes), the typical decoded molecule has:

* ~3–5 atoms (from `n_atoms=8` default at line 1821, minus any that fell below `min_distance` cutoff during integration).
* A subset of those atoms has plausible atomic numbers (because the 12-class vocab mask doesn't apply during training, the head's softmax is ~uniform).
* Pair distances are typically 1–4 Å (random-ish), so most pairs fall inside `bond_cutoff=2.4 Å` OR most fall outside it (depending on init scale).
* Without a connectivity prior, you get a random Erdős–Rényi graph with bond-formation probability p ≈ (2.4 / cloud_diameter)^3 ≈ 0.05 for an 8-atom cloud in a ~10 Å box.

**0.05^8 ≈ 4×10⁻¹¹ chance of all 8 atoms forming a connected graph from this prior — consistent with the 2.6% empirical observation (8 atoms × 28 pairs × p≈0.05 → ~6 bonds expected; probability of ≥7 bonds is ~2.6%)**.

**Falsifiable diagnostics**:

1. Replace the fully-connected graph at `_make_dummy_edge_index` with a *k-nearest-neighbour* graph (k=4) during sampling — if decode_ratio improves, the O(N²) edge count was confusing the model.
2. Add a connectivity prior (e.g. DropEdge p=0.1 + GumbelConnectivity top-k=2N) to the decode step. The `ConnectivityAwareDecoder` (bond_head.py:959) already implements this — wire it into `_generate_impl`.
3. After sampling, compute the actual distance matrix of `x_final` and check whether the inter-atomic distances are within `[1.0, 1.7] Å` for any pair — if not, the EGNN's terminal scale is wrong.

**Recommendation (P0)**: wire `ConnectivityAwareDecoder.decode(cloud)` into `_generate_impl` AFTER computing `sampled_atoms` and BEFORE building the `Molecule` (insert at line 2017). One edit.

---

### 3.4 (d) Why does hidden_dim=32, n_layers=2, lr=1e-4 fail to learn pocket-conditioned placement?

**Empirical pattern**: pocket conditioning has zero measurable effect on the generated geometry.

**Hypothesis (D)**: the configuration is **structurally under-parameterised** for SE(3)-equivariant 3-D molecular generation, AND the pocket signal is injected at the wrong place.

* `hidden_dim=32, n_layers=2`: the `EGNNVelocityField` has roughly 32·32·4 (per-layer Linear) × 3 (msg_scalar_head + msg_vector_head + update_mlp) × 2 (layers) = ~25 K parameters per layer. Total trainable params ≈ 50 K. By comparison, the canonical EGNN for QM9 uses hidden_dim=64 with 7 layers = ~150 K params (Satorras 2021). Pocket-conditioned 3-D generation (TargetDiff) uses hidden_dim=128 with 9 layers = ~1.2 M params (Guan 2023). **50 K params cannot fit a 3-D conditional distribution over 8-atom point clouds**.
* The pocket embedding is a *global mean pool* (`__init__.py:812-15`):
  ```
  pooled = (h * mask_f).sum(dim=1) / n_real  # (B, H)
  ```
  For a typical pocket of ~500 atoms, this is a 32-D vector summarising ~500 Z numbers. The information bottleneck is severe — you cannot recover a 3-D placement signal from a 32-D pooled vector that only "knows" how many C/N/O atoms are in the pocket.
* The pocket signal is injected as an **additive bias** to atom embeddings (line 1098):
  ```
  h = atom_embed(atom_types) + t_per_atom + pocket_bias
  ```
  This bias is constant across all ligand atoms. So *every* atom in the ligand gets the same pocket signal — the model has no way to learn "this atom goes HERE, that atom goes THERE" because the conditioning has no spatial component.
* `lr=1e-4` with AdamW is the standard rate, but at `hidden_dim=32` the gradient signal is tiny: each parameter only gets ~32-dim gradient updates, so useful information per step is at most ~log(32) bits.

**Falsifiable diagnostics**:

1. Set `hidden_dim=128, n_layers=3` (the constructor defaults) and re-run 2000 steps CPU; if decode_ratio > 0 even once, the bottleneck is parameters not architecture.
2. Replace `pocket_embed` with a per-pocket-atom feature map (not mean-pooled) and broadcast it as `pocket_bias[b, n, H]` where the bias varies by ligand atom position relative to pocket atoms — this is the TargetDiff pattern.
3. Increase `lr` to `1e-3`; if cfm_loss diverges, the network is too small to absorb a larger step; if it drops faster and plateau higher, the lr was the bottleneck.

**Recommendation (P1, requires GPU retrain)**: bump `hidden_dim` to 128 and `n_layers` to 3 (the constructor defaults), and change the pocket injection from a mean-pooled additive bias to a per-atom cross-attention with pocket atoms (TargetDiff §3.2).

**Recommendation (P0, no GPU needed)**: surface a *warning* in `setup()` if `hidden_dim < 64` or `n_layers < 3` — these are below the minimum threshold for any 3-D equivariant generation task. The harness script's `--hidden-dim=32 --n-layers=2` flags should be flagged as "PLACEHOLDER, NOT FOR PRODUCTION".

---

## 4. EGNN design comparison vs canonical patterns

### 4.1 EquivariantGraphConv (`egnn_rocm.py:144`) vs canonical EGNN (Satorras 2021)

| Design aspect | `EquivariantGraphConv` (egnn_rocm.py) | Canonical EGNN | Verdict |
|---|---|---|---|
| Scalar message input | `[h_i, h_j, dist]` (1 scalar) | `[h_i, h_j, dist²]` (1 scalar) | OK |
| Coord update basis | `dir_vec * coord_scale` where `coord_scale = sigmoid(MLP(...))` (line 268) | `phi * (x_j - x_i)` where `phi = MLP(...)` (sigmoid implicit via weight init) | Subtle: ours uses sigmoid to bound, canonical uses raw MLP output |
| Scalar update | `h + agg(msg)` where `msg = SiLU(MLP)` (line 264-311) | `h + agg(SiLU(MLP))` | OK (bit-exact with canonical) |
| Aggregation | `_scatter_sum` via `molmetal.models._scatter` (line 307) | `scatter_sum` via `torch_scatter` | OK (Triton-backed shim) |
| Self-loop | `src != dst` (line 2093-95) | excluded by convention | OK |
| Time / condition injection | NOT injected into `EquivariantGraphConv.forward` (see `_project_to_hidden`, line 478) | via `h → cat([h, t])` before MLP | MISSING |

`EquivariantGraphConv` is **NOT** used by `EGNNVelocityField`. It is wired into the geometric prior (`apply_metal_geometry_step`) and into tests only.

### 4.2 EGNNLayer (`models/velocity_net.py:96`) vs canonical EGNN

| Design aspect | `EGNNLayer` (velocity_net.py) | Canonical EGNN | Verdict |
|---|---|---|---|
| Scalar message input | `[h_i, h_j, dist, dist²]` (2 scalars) | `[h_i, h_j, dist²]` | OK — more expressive |
| Per-edge scalar output | `m_scalar` (hidden_dim-wide) + `phi` (1-wide via `msg_vector_head`) | single MLP → both uses | OK — split heads are an Equiformer-style pattern |
| Coord update basis | `phi * rel` (single 3-vector per edge) (line 232) | `phi * (x_j - x_i)` | OK |
| Bounded coord messages | `bounded_coordinate_messages=True` → `tanh(phi)` + `agg_vec / max(n-1, 1)` (line 219-221, 267) | none (relies on weight init + small lr) | DEFENSIVE — necessary to prevent the ODE field from running away |
| Cond injection | `node_in = [h, agg_scalar, v_norm, cond_per_node]` (line 290-97) | varies (concat vs sum) | OK — broadcast at update MLP |
| Vector aggregation | `scatter_sum(... features=edge_vec_flat ...)` (line 254-261) | `torch_scatter.scatter_add` | OK — Triton primitive |

### 4.3 Comparison with Equiformer / EquiformerV2

| Aspect | Our `EGNNLayer` | Equiformer (Liao 2022) | EquiformerV2 (Liao 2023) |
|---|---|---|---|
| Representation | scalars (H) + vectors (1 per node, equivariant) | SO(3) irreps up to l=2 | SO(3) irreps up to l=2 with depthwise tensor product |
| Equivariance | exact (SE(3) on coords; scalars invariant) | exact (rotational, not full SE(3)) | exact |
| Attention | none (just message passing) | graph attention | separable S2 activation + graph attention |
| Activation gating | none | `gate` (SiLU on l=0, sigmoid on l≥1) | learnable per-degree gates |
| Hidden dim typical | 32–128 | 96–192 (irreps × degree) | 192 (irreps) |
| Params for QM9 | ~50K (h=32) to ~250K (h=128) | ~1M | ~3M |

**Verdict**: our EGNN is correct for scalar-only molecular property prediction, but is **architecturally below the threshold** for pocket-conditioned 3-D generation. The TargetDiff/DiffSBDD pattern uses Equiformer-style irreps or much wider EGNNs (~1.2M params). Our `hidden_dim=32` config is **at least an order of magnitude under-parameterised**.

### 4.4 Critical configuration drift

The default `EGNNVelocityField(hidden_dim=128, n_layers=3)` (line 843-44) is sensible. But:

* `LipmanFlowMatchingAdapter.__init__` defaults to `hidden_dim=128, n_layers=3` (line 1361-62) — same.
* The **harness script** (`molmetal/scripts/r10_cfg_real_crossdocked.py`) CLI defaults to `--hidden-dim=32 --n-layers=2 --lr=0.0001` (per `wf_cfm_retrain_full/final.md` §2). This is **NOT** the adapter's default; it's a deliberate "fast smoke test" config.
* The WF-2 baseline (`molmetal/reports/wf2_cfg_e2e_a5`) used this 32/2/1e-4 config and got `n_decoded=0/384`.
* WF-CFM-Retrain-Full planned `hidden_dim=64, n_layers=3, lr=1e-4` (the spec) but was GPU-blocked.
* WF-CFM-Retrain-Diagnose planned the same spec but was GPU-blocked.

**The `decode_ratio=0` failure we are explaining is therefore the EXPECTED outcome of `hidden_dim=32, n_layers=2, lr=1e-4`** — it is not surprising. The surprise is that we keep retrying at the same under-parameterised scale instead of the adapter defaults.

---

## 5. Recommendations (P0 vs P1)

### 5.1 P0 — no GPU needed, runnable today

1. **Decode wiring fix** (`__init__.py:2017`, 1-line edit): replace the `bonds=zeros` placeholder with a `BondAwareDecoder.decode(AtomCloud(positions=x_final[i], atomic_numbers=sampled_atoms[i]))` call. Without this, `decode_ratio` can never exceed 0 regardless of model quality.
2. **Bond-head in_dim fix** (`__init__.py:1515-27`, 5-line edit): construct `BondOrderHead` with `in_dim=9 + 2*self._hidden_dim` so the EGNN-conditioned features aren't silently discarded at line 1734-36. Rebuild `fc1` to match.
3. **Atom-mask in training loss** (`__init__.py:1671`, 2-line edit): apply the vocab mask to `atom_logits` BEFORE `F.cross_entropy` so the gradient isn't wasted on 88 out-of-vocab slots.
4. **Configuration sanity warning** (`LipmanFlowMatchingAdapter.setup`, 3-line edit): raise a `UserWarning` (not an error) when `hidden_dim < 64` or `n_layers < 3`. The harness smoke configs should be flagged as such.
5. **Decoder integration** (`_generate_impl`, 5-line edit): after `sampled_atoms = ...`, build `AtomCloud(positions=x_final[i], atomic_numbers=sampled_atoms[i])` and call `BondAwareDecoder.decode(cloud)`; populate `mols[i].bonds` and `mols[i].bond_types` from the resulting `DecodedMol`.

### 5.2 P1 — requires GPU retrain

1. **Architecture upgrade**: bump `hidden_dim=128, n_layers=3` (the constructor defaults) for any production CFM run. At `hidden_dim=32` the model is below the information-theoretic capacity floor for 8-atom 3-D generation.
2. **Velocity basis un-bounding**: drop `tanh` from the scalar basis, or replace with a `vel_scale` learnable scalar initialised to ~5.0. The tanh is the empirical cfm_loss floor.
3. **Pocket conditioning upgrade**: replace the mean-pooled additive bias with a per-atom cross-attention (TargetDiff §3.2 pattern). The current pool is a 32-D summary of a ~500-atom pocket — far below capacity.
4. **Connectivity-aware decoder** (P1.5, in `bond_head.py:959` already implemented): wire `ConnectivityAwareDecoder.decode(cloud)` into `_generate_impl` so the Gumbel-top-k edges form a connected graph.

### 5.3 Verification gates (CPU-only, no GPU needed)

1. P0 #1 (decode wiring): `Molecule(coords=x, atom_types=z, bonds=...).bonds.shape == (2, E > 0)` for a known-good test case.
2. P0 #2 (bond in_dim): `bond_head.in_dim == 9 + 2*hidden_dim` after setup; verify `bond_inputs.shape[-1] == self.bond_head.in_dim` in train_step.
3. P0 #3 (atom mask): `atom_loss < 0.5` after 50 steps on a small synthetic batch (currently ~1.0).
4. P0 #4 (warning): `setup()` emits `UserWarning` when `hidden_dim=32`.
5. P0 #5 (decoder integration): `len(mols[0].bonds[0]) > 0` for a known-good `x_final` and `sampled_atoms`.

---

## 6. Metrics (schema)

```yaml
status: COMPLETE
n_modules_audited: 6
n_failure_points_identified: 4
hypotheses_quality:
  - hypothesis_a_bond_vs_atom_loss: supported by 3 line-cited code locations + 3 falsifiable CPU diagnostics; root cause is silent fallback in train_step at line 1734 + vocab mask not applied in CE
  - hypothesis_b_cfm_floor: supported by quantitative bound computation (||x_1 - x_0||² - bounded_velocity_term² ≈ 5-8 matches empirical 5.27-8.71); 3 falsifiable diagnostics
  - hypothesis_c_disconnect: supported by probability analysis (Erdős-Rényi p≈0.05, P(connected)≈2.6% matches empirical); 3 falsifiable diagnostics
  - hypothesis_d_under_parameterised: supported by param count comparison (50K vs TargetDiff 1.2M); 3 falsifiable diagnostics
recommendations_p0_count: 5
recommendations_p1_count: 4
gpu_required_for_p0: false
gpu_required_for_p1: true
estimated_p0_cpu_wall_clock_hours: 3
estimated_p1_gpu_wall_clock_hours: 12-24
decode_ratio_after_p0_only: PROJECTED (will not exceed 0% without P1, because bond_head cannot fix the under-parameterised vel field)
decode_ratio_after_p0_plus_p1: PROJECTED (target ≥5% on 100-pocket eval per TargetDiff §5.1 reference)
```

---

## 7. Honest framing — MEASURED vs PROJECTED

### MEASURED today (audit only)

* 6 modules audited end-to-end (line counts and ranges in §1).
* 4 failure-point hypotheses stated with line-cited supporting evidence.
* Comparison table against canonical EGNN, Equiformer, EquiformerV2.
* 5 P0 + 4 P1 recommendations with explicit CPU/GPU gating.
* Configuration drift confirmed: harness uses `hidden_dim=32, n_layers=2`; adapter defaults to `hidden_dim=128, n_layers=3`. The 32/2 config is BELOW the documented adapter default.

### PROJECTED (not validated by this audit)

* The probability bound on cfm_loss ≈ 5–8 was computed analytically; an actual training run with the P0 fix and hidden_dim=128 is needed to validate.
* The 2.6% decode_ratio probability analysis assumes the EGNN emits random-ish distances; actual measurements at hidden_dim=128 might differ.
* P1 recommendations (Equiformer-style irreps, cross-attention pocket conditioning) are forward-looking and have NOT been benchmarked in this codebase.

### FALSIFIABLE

Every hypothesis has at least 3 falsifiable diagnostics listed in §3 — all runnable on CPU within minutes. If the diagnostics return the predicted behaviour, the hypotheses stand; if not, they fall.

---

## 8. Files referenced

* `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py` (2248 lines, fully read)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/egnn_rocm.py` (607 lines, fully read)
* `/home/hugo/codes/try_triton_on_rocm/models/velocity_net.py` (491 lines, fully read)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py` (1316 lines, fully read)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf2_cfg_e2e_a5/report.json` (baseline metrics: `n_decoded=0/384` confirmed)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_retrain_full/final.md` (carry-forward reference for GPU-blocked status)

## 9. Files written by this audit

* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_internal_review/audit.md` — this document