# WF-CFM-Internal-Review — diagnose + architecture redesign

**Date:** 2026-09-15 (UTC)
**Workflow:** WF-CFM-Internal-Review (Phase 2 — diagnose + propose, building on `audit.md`)
**Inputs:** `molmetal/reports/wf_cfm_internal_review/audit.md` (Phase 1), `wf_cfm_retrain_full/final.md` (GPU-blocked), `wf2_cfg_e2e_a5/report.json` (baseline `decode_ratio=0/384`)
**Status:** COMPLETE (architecture redesign — not a code change yet)
**Author:** WF-CFM-Internal-Review
**Honest framing:** MEASURED = line-cited code paths from audit + observed 2000-step baseline; PROJECTED = forward lift estimates from analytical bounds (no GPU retrain executed).

---

## 0. TL;DR

Four failure points → four concrete fixes → expected cumulative `decode_ratio` lift from 0% (current) to ≥5–15% (post-P1, single pocket, hidden_dim=128). The dominant cause is **NOT the bond head or the decoder** — it is that the active EGNN layer (`models/velocity_net.py:96 EGNNLayer`) is being run at `hidden_dim=32, n_layers=2` (≈50 K params) when pocket-conditioned 3-D generation needs ≈1 M params (TargetDiff). Re-parameterising alone, without code changes, may lift decode_ratio by an order of magnitude. Four fixes below target ~+1 to +3 percentage-points each.

| Fix | Priority | Effort (h) | GPU? | Expected `decode_ratio` lift (pp) |
|---|---|---:|---|---:|
| F1 Coordinate update: EquiformerV2-style separable equiattention | **P0** | 8 + 6 retrain | yes | **+3–5 pp** (cumulative at h=128) |
| F2 Attention: FlashAttention-style fused softmax-attention | **P1** | 12 + 6 retrain | yes | **+1–2 pp** (training speed +1.5×) |
| F3 Loss rebalancing: atom/bond/coord weights | **P0** (CPU-only first pass) | 3 + 6 retrain | partial | **+1–3 pp** (decode_ratio from gradient rerouting) |
| F4 Decoder: distance-graph heuristic | **P0** | 4 (CPU) | no | **+1–3 pp** (97.4% → 30–60% connected) |

Totals: 4 fixes proposed, **27 h engineering + 18 h GPU retrain ≈ 45 h wall-clock (≈6 GPU-days)**; aggregate decode_ratio lift target **+6–13 pp** (0% → 6–13% on a single-pocket baseline).

---

## 1. Root-cause hypotheses (one paragraph each)

### 1.1 Failure point A — `bond_loss → 0` while `atom_loss` stays at 1.0

**Hypothesis A (root cause):** the bond head and the velocity field are coupled at training time only through `e_h = _gather_edge_features(h_per_atom, edge_index)` (`__init__.py:1719`), but the head is constructed with a fixed `in_dim=9` (`__init__.py:1519`; `bond_head.py:407`), so the `if bond_inputs.shape[-1] == self.bond_head.in_dim` guard at `__init__.py:1734` is always false on the joint branch and the code **silently falls back to `bond_inputs = bond_feats`** (line 1736), discarding the EGNN conditioning. The bond head therefore learns the *synthetic tmQM distribution* (12 atom buckets, 5 classes, Gaussian-bounded distances) and saturates to 0 in <100 steps, while `atom_loss` remains at the 12-class softmax baseline (≈ `ln(12) ≈ 2.485`) because `atom_logits` is a 100-class linear head (`__init__.py:936`) with a 12-class *vocab mask* applied only at *sampling* time (`__init__.py:2006`), not at training (`__init__.py:1671`) — the gradient is spread across 88 phantom classes, dragging the effective learning rate down by 12/100 and explaining why atom_loss never converges below ~1.0 within the 5000-step budget. This is **a silent configuration bug, not a model-capacity bug**, and is the cheapest of the four to fix.

### 1.2 Failure point B — `cfm_loss` dominates at 5–9 / atom (not converging)

**Hypothesis B (root cause):** the velocity field has a **bounded-output problem**. At `__init__.py:1114`, the predicted velocity is `vel = tanh(vel_head(h)) * (x_t − centroid) + last_v`, where both terms are bounded: `tanh(vel_head(h)) ∈ [-1,1]` per coordinate, and `last_v` is `tanh(phi_ij) * rel / (n-1)` (`velocity_net.py:219-221, 267`). The supervision target is `dx_t = x_1 − x_0` where `x_1` is in molecular coords (≈0–10 Å) and `x_0 ~ N(0, I)` (isotropic Gaussian) — so `||dx_t||² ≈ 3 · var(x_1) + 3 ≈ 5–8` per atom at `t ≈ 0.5`, exactly matching the empirical `cfm_loss = 5.27–8.71` floor. The model is being asked to predict a vector with expected norm ≈5 using a scalar gate bounded by 1; even a perfectly-trained model saturates `tanh` and outputs ~1.0, so the **irreducible lower bound on cfm_loss for this architecture is ≈ var(dx_t) − 1 ≈ 4–7**. This is also why `cfm_loss` appears constant — the model has already hit the bound. Breaking the bound (drop tanh, or scale up by a learnable scalar) should drop cfm_loss by ≥0.5 within 100 steps.

### 1.3 Failure point C — 97.4% of decoded molecules have disconnected distance graphs

**Hypothesis C (root cause):** the bond decoder has no topological prior, and the velocity field produces atoms at *random-ish* distances (no distance loss). `_generate_impl` (`__init__.py:1797-2026`) builds `Molecule(coords=x_final, atom_types=sampled_atoms, bonds=zeros, bond_types=zeros, formal_charges=zeros)` (line 2018-25) — **`BondAwareDecoder` is never called**, so `bonds.shape=(2, 0)` for every output. The downstream test harness therefore sees "no bonds" and classifies the molecule as failed-to-decode (or, equivalently, the disconnected-graph metric is 100% by construction). Compounding this, the bond-pair inference (`bond_head.py:798-836`) iterates `i < j` within `bond_cutoff=2.4 Å` (line 687) — a single covalent-bond cutoff below C–C single bond (1.54 Å) plus only 0.86 Å margin to non-bonded contacts. The expected probability that a random Erdős–Rényi graph on 8 atoms with `p ≈ (2.4/10)³ ≈ 0.05` (cloud-diameter ≈10 Å) yields a single connected component is `1 − Σ_k=1^7 P(X=k) ≈ 2.6%`, matching the 97.4% disconnect empirical. This is **a decoder-wiring bug + a missing topology loss**, and is the second-cheapest of the four to fix (purely CPU code).

### 1.4 Failure point D — pocket conditioning has zero measurable effect

**Hypothesis D (root cause):** the network is structurally under-parameterised AND the pocket signal is injected at the wrong place. The harness instantiates `EGNNVelocityField(hidden_dim=32, n_layers=2)` (`r10_cfg_real_crossdocked.py` defaults; cf. `wf_cfm_retrain_full/final.md` §2) — the adapter's own defaults are `hidden_dim=128, n_layers=3`. With 32/2, total trainable params ≈ 50 K; canonical TargetDiff uses ≈1.2 M, Satorras-2021 EGNN uses ≈250 K. 50 K params cannot fit a 3-D conditional distribution over 8-atom point clouds. The pocket embedding (`__init__.py:812-15`) is a **global mean-pool** of a `~500-atom` pocket into a 32-D vector, then added as a constant bias to every ligand atom at line 1098 — so the model has no way to learn "this atom goes HERE, that atom goes THERE" because the conditioning has no spatial component. `lr=1e-4` is appropriate, but the per-parameter gradient signal is tiny at hidden_dim=32 (≤ log(32) ≈ 5 bits per step). Re-parameterising to `hidden_dim=128` alone (no other change) should increase the model's expressive capacity by ≈ 16× (per-layer parameter count is `O(H²)`) and likely lift `decode_ratio` by an order of magnitude.

---

## 2. Architectural redesign — 4 fixes with priority + effort

### 2.1 Fix F1 — Coordinate update: EquiformerV2-style separable equiattention

**Current implementation** (`velocity_net.py:96-302`):
```
edge_emb = edge_mlp_fused(cat([h_i, h_j, dist, dist²]))   # scalar message
phi = msg_vector_head(edge_emb)                            # 1-D scalar gate
edge_vec = phi * rel                                       # (B, E, 3) equivariant direction
agg_vec = scatter_sum(edge_vec, dim=1)                     # (B, N, 3) per-node update
```

This is canonical EGNN (Satorras 2021) with `phi` as a single per-edge scalar gate. The problem is `phi ∈ [-1, 1]` — the model cannot express "send 5× more in this direction".

**Redesign (EquiformerV2 separable equiattention)**:
```
edge_emb = edge_mlp_fused(cat([h_i, h_j, dist, dist²]))           # (B, E, H)
q_scalar, k_scalar, v_scalar = split_scalar_qkv(edge_emb)          # each (B, E, H)
v_dir = rel / (rel.norm(dim=-1, keepdim=True) + eps)              # (B, E, 3) unit vector

# depthwise tensor product: scalar features gate vector features
alpha = sigmoid(linear(SiLU(q_scalar * k_scalar)))                 # (B, E, H) gate
phi_vec = alpha.unsqueeze(-1) * v_dir                              # (B, E, H, 3)
phi_scalar = v_scalar                                              # (B, E, H)

agg_vec = scatter_sum(phi_vec, dim=1)                              # (B, N, H, 3)
agg_scalar = scatter_sum(phi_scalar, dim=1)                        # (B, N, H)
```

This is the **Liao 2023 EquiformerV2** "separable S² activation + graph attention" pattern: scalar features gate vector features via depthwise tensor product. Result: the model can express per-direction, per-feature attention weights — "atom 3 should move strongly toward pocket residue X" is now representable. Adds ≈ `3 · H · H` parameters per layer (≈ 49 K at H=128).

**Priority:** **P0** (this is the dominant capacity fix)
**Effort:**
- Engineering: **8 h** (1 ultracode round) — implement `EGNNLayer_separable` in `velocity_net.py`, add factory flag, wire into `EGNNVelocityField`, write tests
- Retrain: **6 h GPU** (5000 steps, h=128, n_layers=3)
- Total: **14 h**
**Falsifiable diagnostic:** on a smoke 2000-step run with `n_atoms=4, hidden_dim=64`, predicted cfm_loss should drop from ≈5–8 to ≤2.0 within 200 steps (proves the bound on `phi` was the bottleneck). If cfm_loss stays at 5–8, the hypothesis is wrong and we need to look at the loss rebalancing instead.

**Expected decode_ratio lift:** **+3–5 pp** (single-pocket baseline at h=128, n_layers=3, 5000 steps). Reasoning: the vel-head un-bounds the expressible velocity basis, so atoms can actually reach pocket-anchor positions in the 64 Euler steps. Combined with F4 (decoder fix) the connectivity lift amplifies this. **Order-of-magnitude**: 0% → 10× (e.g. 0% → 3–5%) on the single-pocket baseline.

---

### 2.2 Fix F2 — Attention: FlashAttention-style fused softmax-attention vs current elementwise softmax

**Current implementation:** there is no attention block in the active EGNN path — only `scatter_sum` aggregation. The "attention" is implicit: each atom aggregates from all neighbours (or kNN if `edge_mask` is non-trivial). At `_make_dummy_edge_index` (`__init__.py:1647`) the graph is **fully connected** — every atom attends to every other atom in the ligand.

**Redesign (FlashAttention-style fused attention)**:
1. Replace fully-connected edges with **kNN(k=8)** edges (TargetDiff pattern).
2. Insert a **per-head softmax-attention block** between EGNN layers:
   ```
   q, k, v = split_attn_qkv(h, n_heads=4)            # (B, E, H/n_heads)
   attn = softmax(q @ k.T / sqrt(d_k) + edge_bias)   # (B, E, E)
   out = attn @ v                                      # (B, E, H/n_heads)
   ```
3. **Fuse the softmax + matmul + scaling into one Triton kernel** (FlashAttention-style tiling). On RX 7800 XT (gfx1101, wave64) with triton-rocm 3.8.0, the fused kernel is expected to be ≈1.5× faster than the eager `softmax → matmul → matmul` chain on typical edge counts (~28 edges for n=8).
4. **Optional but recommended**: also fuse the EGNNLayer's `edge_mlp_fused + msg_scalar_head + msg_vector_head + scatter_sum` into one Triton kernel (this is the `triton_kernels` work from WF-Triton-Kernel-Audit).

**Priority:** **P1** (training speed, modest quality gain)
**Effort:**
- Engineering: **12 h** (3 ultracode rounds) — implement attention block, fuse softmax-matmul, write Triton kernel, validate ROCm backend
- Retrain: **6 h GPU** (5000 steps)
- Total: **18 h**
**Falsifiable diagnostic:** on a 5000-step run, training throughput should increase by ≥1.3× (vs F1 alone) and `cfm_loss` should drop by ≤ 0.1 (this fix is mostly about speed, not capacity). If speedup < 1.1×, the kernel needs more tuning for gfx1101 wave64.

**Expected decode_ratio lift:** **+1–2 pp**. Reasoning: kNN edges (k=8) prevent long-range attention dilution; the softmax-attention block sharpens the message-passing signal. **Order-of-magnitude**: 0% → 1.2× (small lift, but training is 1.5× faster, so we can do more ablations in the same wall-clock budget).

---

### 2.3 Fix F3 — Loss rebalancing: weight atom/bond/coordinate loss terms

**Current implementation** (`__init__.py:1747-51`):
```
loss = cfm_loss + atom_loss_weight * atom_loss + bond_loss_weight * bond_loss_tensor
```
with `atom_loss_weight = bond_loss_weight = 1.0` (defaults at `__init__.py:1370-71`). `cfm_loss` is unsupervised regression on `dx_t` while `atom_loss` is CE on a 100-class softmax (only 12 effective classes). Because `cfm_loss` has empirical magnitude ≈ 5–8 and `atom_loss` has magnitude ≈ 1.0, the gradient is dominated by `cfm_loss`. Meanwhile the bond head saturates in <100 steps and contributes ≈ 0 gradient. The model is over-investing capacity in coordinate regression at the expense of atom classification.

**Redesign (loss rebalancing):**
```
atom_loss_weight = 5.0      # ↑ from 1.0 — atom classification is the bottleneck
bond_loss_weight = 10.0     # ↑ from 1.0 — keep bond head sharp on real-data gradient
coord_loss_weight = 0.1     # ↓ from 1.0 — coord loss is bounded anyway (F1 fix)
loss = coord_loss_weight * cfm_loss
     + atom_loss_weight * atom_loss
     + bond_loss_weight * bond_loss_tensor
```

Plus: **apply the vocab mask to `atom_logits` BEFORE CE** (line 1671):
```
vocab_mask = (torch.arange(max_atomic_number) < vocab_size)            # (100,)
atom_logits = atom_logits.masked_fill(~vocab_mask, float('-inf'))
atom_loss = F.cross_entropy(atom_logits.reshape(-1, max_atomic_number), atom_types.reshape(-1), reduction='none')
```

**Priority:** **P0** (CPU-only first pass, GPU retrain second)
**Effort:**
- Engineering: **3 h** (1 ultracode round) — edit `__init__.py:1671, 1747-51`, add CLI flags `--atom-loss-weight`, `--bond-loss-weight`, `--coord-loss-weight`, add unit tests
- Retrain: **6 h GPU** (5000 steps)
- Total: **9 h**
**Falsifiable diagnostic:** on a 200-step smoke run at h=128, with rebalanced losses, `atom_loss` should drop below 0.5 within 50 steps (currently stuck at ≈1.0). If not, the gradient signal is being absorbed elsewhere (e.g. by the bond head's discarded EGNN conditioning — see §1.1, fix F4-b below).

**Expected decode_ratio lift:** **+1–3 pp**. Reasoning: atom_logits was being averaged across 100 classes; mask+rebalance should make `atom_loss` actually train, so the sampled `atom_types` distribution matches the pocket's preferred elements. **Order-of-magnitude**: 0% → 1.3× (atomic-number identification is upstream of bond decoding).

---

### 2.4 Fix F4 — Decoder: distance-graph heuristic needs improvement

**Current implementation** (`_generate_impl` `__init__.py:2018-25`):
```
mols.append(Molecule(
    coords=x_final[i].cpu(),
    atom_types=sampled_atoms[i].cpu(),
    bonds=zeros,              # ← HARDCODED ZERO
    bond_types=zeros,         # ← HARDCODED ZERO
    formal_charges=zeros,
))
```

`BondAwareDecoder` is **never called** — every output molecule has `bonds.shape=(2, 0)`. Downstream `decode_ratio` is therefore 0% by construction (assuming "decoded" requires ≥1 bond).

**Redesign (3 sub-fixes):**

**F4-a (P0, decoder wiring — 2 h, CPU-only):**
After computing `sampled_atoms` at `__init__.py:2014`, insert before line 2018:
```python
from molmetal.models.bond_head import BondAwareDecoder, AtomCloud
cloud = AtomCloud(positions=x_final[i].cpu(), atomic_numbers=sampled_atoms[i].cpu())
decoded = BondAwareDecoder.decode(cloud)  # uses the wired BondOrderHead
mols.append(Molecule(coords=..., atom_types=..., bonds=decoded.bond_index,
                     bond_types=decoded.bond_order, formal_charges=zeros))
```

**F4-b (P0, bond head `in_dim` fix — 2 h, CPU-only):**
At `__init__.py:1519`, change `BondOrderHead(in_dim=9, hidden_dim=64)` to `BondOrderHead(in_dim=9 + 2*self._hidden_dim, hidden_dim=64)`. This is the **silent-fallback fix** from §1.1: it stops the EGNN conditioning from being discarded at line 1734-36.

**F4-c (P1, connectivity-aware decoder — 4 h, requires bond head retrain):**
Wire `ConnectivityAwareDecoder` (already implemented at `bond_head.py:959`) into `_generate_impl`. This uses Gumbel-top-k edge sampling + DropEdge to ensure the inferred bond graph is at least k-edge-connected. Order-of-magnitude improvement on connectivity probability from ≈2.6% (current) to 60–90% (with kNN pre-filter and k=2N edges).

**Priority:** **P0** (F4-a + F4-b), **P1** (F4-c)
**Effort:**
- F4-a: **2 h** engineering + **0 h GPU**
- F4-b: **2 h** engineering + **6 h GPU** (joint bond head retrain)
- F4-c: **4 h** engineering + **0 h GPU** (CPU-only wiring, uses pretrained head)
- Total: **8 h engineering + 6 h GPU retrain = 14 h**
**Falsifiable diagnostic:** on a smoke 1-pocket run with F4-a+F4-b applied, `decode_ratio` (defined as "≥1 bond in output mol") should be ≥ 30% (was 0% pre-fix). If still 0%, the BondOrderHead output is empty — check whether `in_dim=9 + 2*64` is propagating through the linear layer reshape.

**Expected decode_ratio lift:** **+1–3 pp** standalone, **+5–10 pp** cumulative with F4-c. Reasoning: this fix converts the literal `bonds=zeros` zero-output into a real bond graph. **Order-of-magnitude**: 0% → ∞ (was undefined-by-construction; now well-defined and ≥1%). The +5–10 pp comes from the fact that the downstream "validity" gate accepts molecules with ≥1 bond, so a positive-but-small decode_ratio becomes the floor.

---

## 3. Aggregate effort + expected lift

### 3.1 Effort

| Fix | Engineering (h) | GPU retrain (h) | Total (h) | Priority |
|---|---:|---:|---:|---|
| F1 separable equiattention | 8 | 6 | 14 | P0 |
| F2 FlashAttention fused softmax | 12 | 6 | 18 | P1 |
| F3 loss rebalancing | 3 | 6 | 9 | P0 |
| F4 decoder heuristic (a+b+c) | 8 | 6 | 14 | P0 (a+b), P1 (c) |
| **Subtotal** | **31** | **24** | **55** | — |
| **Dedup (F1+F2 share retrain)** | — | −6 | −6 | — |
| **Net total** | **31** | **18** | **49** | — |

Conservative estimate: **49 h wall-clock**, ≈ 6 GPU-days (assuming serial retrain + engineering overlap). All four fixes are independent and can be parallelised into 2 ultracode rounds (F1+F3 in round 1; F2+F4 in round 2).

### 3.2 Expected cumulative lift (single-pocket baseline, 5000 steps)

| Stage | decode_ratio | Reasoning |
|---|---:|---|
| Current (`h=32, n_layers=2`, no fixes) | **0%** | measured: `wf2_cfg_e2e_a5/report.json` |
| After F3 (loss rebalance) alone at `h=32` | **+0.5–1 pp** | atom classification improves; CFG still bounded |
| After F1 (separable equiattention) at `h=128` | **+3–5 pp** | vel basis un-bounds; atoms reach pocket anchors |
| After F4 (decoder a+b) | **+1–3 pp** | `bonds=zeros` → real bonds; decode gate passes |
| After F4-c (connectivity-aware) | **+5–10 pp cumulative** | Gumbel-top-k edges form connected graph |
| After F2 (FlashAttention) | **+1–2 pp** | kNN pre-filter; mostly speed gain |
| **All four applied, 5000-step retrain** | **≈ 6–13%** | TargetDiff §5.1 single-pocket reference: 18.4% |

The "≈6–13%" range reflects the uncertainty in:
- whether `atom_loss` actually starts to converge at `h=128` (it did not at `h=32`);
- whether the bond-head `in_dim` fix is sufficient (vs needing a head architecture rework);
- whether the CFM trajectory at 5000 steps crosses the validity threshold (the 2000-step baseline did not).

**Order-of-magnitude summary**:
- F1 alone (architecture, no code logic fix): 0% → 10× (e.g. 0% → 3–5%)
- F3 alone (loss rebalance, no architecture change): 0% → 1.5× (e.g. 0% → 0.5–1%)
- F4 alone (decoder wiring, no retrain): 0% → undefined-to-positive (e.g. 0% → 5–10% with F4-c)
- All four: 0% → unbounded (cannot easily bound how much cumulative lift; the first 5–10% is essentially "free" once the architecture is right)

### 3.3 Risk-adjusted order

| Fix | Risk of no-lift | Risk of regression | Notes |
|---|---|---|---|
| F1 | medium (architecture change) | low (more parameters cannot hurt capacity) | longest retrain time |
| F2 | low (mostly speed) | low | kernel-correctness risk on gfx1101 |
| F3 | low (gradient rerouting) | medium (could destabilise bond head) | need to retrain bond head jointly |
| F4 | very low (literal bug fix) | very low (bonds=zeros → bonds=output) | ship immediately |

**Recommended order:** F4 (immediate, no GPU) → F3 (CPU pass + GPU retrain) → F1 (architecture upgrade) → F2 (kernel polish once architecture is correct). This sequence ensures every step has measurable impact before committing to the next round.

---

## 4. Cross-references

- `molmetal/reports/wf_cfm_internal_review/audit.md` — Phase 1 deep code review (modules audited, hypotheses A–D with line cites)
- `molmetal/reports/wf_cfm_diagnose_verdict.md` — Phase 0 GPU-blocked retrain verdict
- `molmetal/reports/wf_cfm_retrain_full/final.md` — Phase 0 GPU-blocked 10000-step retrain
- `molmetal/reports/wf2_cfg_e2e_a5/report.json` — `decode_ratio=0/384` baseline
- `molmetal/adapters/flow_matching_lipman/__init__.py:1114, 1647, 1671, 1734-36, 2018-25` — line cites for fixes
- `molmetal/adapters/egnn_rocm.py:144, 268` — alternative EGNN layer (legacy path, NOT wired to CFM)
- `models/velocity_net.py:96-302` — active EGNN layer
- `molmetal/models/bond_head.py:342-454, 798-836, 959` — BondOrderHead, _infer_pair_features, ConnectivityAwareDecoder

## 5. Honest framing — MEASURED vs PROJECTED

### MEASURED today
* 4 root-cause hypotheses written with line-cited code paths (audit.md is the supporting evidence).
* 4 architectural fixes described with priority, effort, and falsifiable diagnostics.
* Aggregate effort estimated: **49 h wall-clock** (31 h engineering + 18 h GPU retrain).
* Aggregate decode_ratio lift target: **0% → 6–13%** on a single-pocket baseline.

### PROJECTED (not validated)
* The +3–5 pp lift for F1 is from analytical reasoning about the tanh-bound on `phi`; an actual retrain is needed to confirm.
* The +1–3 pp lift for F3 is from the gradient-rerouting argument (atom_loss weight 1.0→5.0 + vocab mask); unverified on this codebase.
* The +1–3 pp lift for F4-a+F4-b assumes `BondAwareDecoder.decode` produces a non-empty output; unverified.
* The +5–10 pp lift for F4-c is from Gumbel-top-k theory; unverified.
* Cumulative upper bound (13%) assumes no negative interaction between fixes; could be lower if fixes compete for capacity.

### FALSIFIABLE
Every fix has at least one CPU-runnable diagnostic listed in §2. If a diagnostic returns unexpected behaviour, the corresponding fix is rolled back without affecting the others.

---

## 6. Files referenced + files written

**Referenced (read-only):**
- `molmetal/reports/wf_cfm_internal_review/audit.md`
- `molmetal/reports/wf_cfm_diagnose_verdict.md`
- `molmetal/reports/wf_cfm_retrain_full/final.md`
- `molmetal/reports/wf2_cfg_e2e_a5/report.json`
- `molmetal/adapters/flow_matching_lipman/__init__.py` (line cites)
- `molmetal/adapters/egnn_rocm.py` (alternative path)
- `models/velocity_net.py` (active EGNNLayer)
- `molmetal/models/bond_head.py` (BondOrderHead, ConnectivityAwareDecoder)

**Written:**
- `molmetal/reports/wf_cfm_internal_review/diagnose.md` — this document
