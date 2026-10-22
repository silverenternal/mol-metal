# Wf-A108 — What a 3D velocity head can express, and what the frontier does about it

**Status:** survey + design note for Wf-A108 (2026-10-22). Read before changing
`EGNNVelocityField`'s coordinate head.
**Why it exists:** `docs/HONEST_NEGATIVE_CATALOG.md:§6.170` MEASURED that the current
velocity field cannot fit one 8-cloud batch (MSE 10.529 against a 10.652 `v = 0`
baseline and a 5.783 radial-only ceiling; |v| 0.298 vs |target| 5.159; cosine 0.128).
This note asks whether that is an implementation defect or the expressivity class of
the head, what the 2024-2026 literature does instead, and which of those choices are
affordable and worth it for **precious-metal anticancer complexes**.

Every number tagged MEASURED below traces to a JSON under `metrics/by_metric/`.

---

## 1. The current head, stated as a function class

`molmetal/adapters/flow_matching_lipman/__init__.py:2164`:

```
vel_i = tanh(vel_head(h_i)) * vel_scale * (x_i - x_centroid) + last_v_i
```

`last_v_i` is the last `EGNNLayer`'s vector message,
`models/velocity_net.py:287-300`:

```
last_v_i = (1 / max(N-1, 1)) * Σ_j tanh(φ_ij) * (x_j - x_i)
```

So the head is a sum of exactly two terms:

| term | direction | magnitude |
|---|---|---|
| radial | the ray `x_i - x_centroid` | `|vel_scale| * |x_i - x_centroid|`, `vel_scale` free |
| edge-mean | a convex-ish combination of `(x_j - x_i)` | `≤ mean_pair_distance` (each coefficient is in `(-1, 1)` **and** divided by `N-1`) |

Two properties follow, and both are structural rather than optimisation artefacts:

1. **The only term with a free magnitude is radial.** The edge term's coefficients
   are `tanh`-bounded *and* degree-normalised, so it is an average, not a sum: it
   cannot exceed the mean pair distance no matter what the weights are.
2. **The edge term is a single channel.** One scalar `φ_ij` scales one 3-vector per
   edge, so the layer has one "direction slot" per edge. It can reweight neighbour
   directions, but it cannot carry *several independent* vector hypotheses per atom
   the way a channel dimension does.

This is the same limitation the original EGNN paper accepts: EGNN was designed for
**invariant** scalar outputs (energy, property) with an optional coordinate update,
not for regressing a vector field whose target has O(10×) the prior's scale.

## 2. What the frontier actually changes

| work | coordinate head | parameterisation | takeaway for us |
|---|---|---|---|
| EGNN — Satorras et al., ICML 2021, arXiv:2102.09844 | `Σ_j φ(m_ij)(x_j - x_i)` | — | our `last_v`; fine for scalars, bounded for vectors |
| TFN — Thomas et al., arXiv:1802.08219 | tensor products up to `ℓ_max` | — | completeness needs `ℓ ≥ 1` mixing; expensive |
| GVP — Jing et al., ICML 2021, arXiv:2009.03092 | per-channel scalar gate on vector features | — | the cheap, proven way to get `C` vector channels |
| MiDi — Vignac et al., ICLR 2023, arXiv:2303.04791 | `d_equi` coordinate sets, per-atom RMS normalised | — | normalising the *channels* (not the coordinates) is what keeps deep stacks stable |
| SemlaFlow — Irwin et al., AISTATS 2025, arXiv:2406.07266 | attention-weighted sum of **unit** relative directions over `d_equi = 64` channels + outer-product feature mixing | **x₁-prediction**; `u = (x̂₁ - x_t)/(1-t)` | coordinates are *not special*: treat `x` as one more equivariant channel |
| FlowMol3 — Dunn & Koes, arXiv:2508.12629 | `x_i ← x_i + ψ_x(s_i, v_i)` (3 chained GVPs, final one emits 1 vector) | **x₁-prediction**, loss `w(t)‖x̂₁-x₁‖²`, `w(t)=min(max(0.005, t/(1-t)), 1.5)` | 6M params beats 40-150M models; the endpoint loss is the whole trick |
| FrameFlow — Yim et al., arXiv:2310.17082 | per-residue **local frame**, update = linear combination of frame vectors | rigid-body FM | local frames buy expressivity *and* interpretability |
| TMCgen — Schaufelberger & Jorner, arXiv:2606.00666 | E3NN GNN emitting tangent vectors on `S² × SO(3) × T^m` | manifold diffusion over **coordination angles + ligand rotations + torsions**, M–L bond lengths **fixed** | for TMCs the informative DOFs are metal-centric, not Cartesian-global |

Three independent conclusions, and they are not the same conclusion:

**(a) The direction problem is solved by vector *channels*, not by a bigger MLP.**
SemlaFlow, FlowMol3 and MiDi all keep `C` independent 3-vectors per atom and let a
learned linear map combine them. The coordinate update stops being "a scalar times a
fixed ray" and becomes "a projection of a learned vector basis". This is exactly the
missing degree of freedom in `last_v`.

**(b) The magnitude problem is mostly a *parameterisation* problem.**
Both SemlaFlow and FlowMol3 predict `x₁` and derive the velocity algebraically,
`u = (x̂₁ - x_t)/(1-t)`. The network then regresses a target of the same scale as the
data (Å), instead of `x₁ - x₀` whose scale is set by the prior (≈2.5× the data at
`fm_noise_sigma` 2.485). Their clamped weight `w(t)` is the exact change-of-variables
factor `(1-t)^{-2}`, capped for stability.

**(c) For metal complexes, the informative degrees of freedom are metal-centric.**
TMCgen's argument is data efficiency: 37M organic conformers vs 61k experimental TMC
structures, so spending capacity on global Cartesian DOFs is wasteful when the
chemistry lives in coordination angles. It also documents the failure mode we care
about — torsional-only models "keep all bond lengths and angles fixed, including the
coordination bonds and angles of ligands around a metal center" and therefore cannot
generate coordination environments at all.

## 3. What we adopt, and in what order

The order matters: one factor at a time, each with its own gate, because §6.166 and
§6.170 both showed that a two-factor change produces a number nobody can attribute.

### Wf-A108a — vector-channel coordinate head (architecture only)

`velocity_head: vector_channels` (default `radial`, bit-exact).

* Per atom keep `V ∈ R^{C×3}`, `C = 8` by default.
* Init: channel 0 = `x_i - x_centroid`; other channels zero. (Channel 0 is the old
  radial basis, so the head can *recover* the current behaviour but is not limited to it.)
* Per layer, per edge:
  `m_ij^c = gate_ij^c · V_j^c + φ_ij^c · (x_j - x_i)`, with
  `gate, φ ∈ R^C` from the edge MLP. Aggregate by destination, divide by `√deg`.
  This is GVP's scalar gating (Jing 2021) with SemlaFlow's per-channel radial
  injection (Irwin 2025, Eq. 4/10-13), at `O(E·C)` rather than Semla's `O(E·C²)`.
* After aggregation: shared channel mixing `V ← V + V W` (`W` zero-init) and a
  per-atom RMS normalisation with a learnable per-channel gain (MiDi's `_CoordNorm`).
* Output: `u_i = Σ_c w_c(h_i) · V_i^c` with `w(h)` zero-init, so the field starts at
  zero velocity exactly like `vel_head` does today.

Why this is the right first step: it changes the *function class* and nothing else.
The loss, the sampler, the data, and every other head are untouched, so a difference
in the gate is attributable to the architecture.

### Wf-A108b — x₁ parameterisation (loss + sampler only)

`fm_parameterisation: x1` (default `velocity`).

* Head output stays a vector; interpret it as `x̂₁ = x_t + (1-t)·u`.
* Loss: `w(t)·‖x̂₁ - x₁‖²`, `w(t) = min(max(0.005, t/(1-t)), 1.5)` (FlowMol3 Eq. 6-7).
* Sampler: `u = (x̂₁ - x_t)/max(1-t, 1e-3)` inside the ODE wrapper, so `ODESolver`
  still integrates a velocity and no solver code changes.
* Record `fm_parameterisation` and the realised `w(t)` mean in the artifact.

Why separate: if 108a already fixes the fit, 108b is a refinement; if 108b is what
fixes it, we learn that the head was adequate and the target scale was not.

### Wf-A108c — metal-centred channels

`metal_frame_channels: true` (default off).

* Channel 1 := `x_i - x_M(i)` for the nearest metal in the same cloud (zero if the
  cloud has no metal). Channel 2 := `x_i - centroid(donors of that metal)`.
* The edge MLP gains a binary `is_metal_edge` feature (edge incident to a metal
  atom), so dative edges are distinguishable without a new input tensor — the FM
  path has no dative flags today and adding one would change every call site.
* No new loss. The coordination geometry prior (`metal_geometry_loss`, T9) stays
  where it is; this only gives the *velocity* a metal-relative basis.

Why this and not TMCgen's manifold: switching to `S² × SO(3) × T^m` would mean
generating ligands as rigid fragments with fixed internal geometry. Our corpus is
metal-cytotoxicity complexes whose ligands are flexible and whose decode path needs
Cartesian coordinates for RDKit. A product-manifold model is a different project.
Giving the Cartesian field a metal-centric basis captures the same insight — the
metal-ligand DOFs are the ones that matter — at one channel's cost.

**Status: the metal-frame channels did not earn their keep on the capacity gate**
(§5.1: 1.065 with vs 1.157 without at 400 steps, seed 0; 0.092 vs 3.034 at 800
steps). The `is_metal_edge` scalar is separate and always on for this head, so the
metal *topology* is visible either way; only the two seeded channels are in question.
They stay opt-in and are **not** enabled in `configs/wfa108_vector_channels.yaml`'s
shipped default recommendation until the transport arm separates them.

## 4. What we deliberately do not adopt

* **Attention over `d_equi = 64` channels (SemlaFlow).** The `O(E·C²)` channel-mixing
  is the expensive part; at `C = 8` and our 14-72 atom clouds the gated form covers
  the same function class for the vector field. Revisit only if 108a's gate passes
  but transport is still poor.
* **Cross-product GVPs (FlowMol3).** They exist to make the model SE(3)- rather than
  E(3)-equivariant so stereoisomers get different likelihoods. That is a real concern
  for Δ/Λ Ru(II) complexes, but it is a *chirality* feature, not a *transport*
  feature, and it should not ride along with a capacity fix.
* **Self-conditioning / fake atoms / geometry distortion.** FlowMol3's ablations say
  these are worth 1-4 points each and interact. We already have `fm_p_distort`
  (FlowMol3's distortion, `--fm-p-distort`). Self-conditioning is the next
  self-correction feature after the head works, not before.
* **Torsion-only or internal-coordinate flow.** See §3c: it cannot generate
  coordination environments at all, which is the thing we are trying to generate.

## 5. Gate (must pass before any budget run)

### 5.1 Results so far (MEASURED, 2026-10-22)

**Capacity — overfit one 8-cloud val batch, 400 AdamW steps, lr 1e-3, t = 0.5,
`--seed 0`.** Artifacts: `metrics/by_metric/wfa108h_*capacity*.json`.

| arm | reached MSE | % of `v = 0` baseline | % of radial ceiling (5.783) | \|v\| (target 5.159) | cosine |
|---|---:|---:|---:|---:|---:|
| radial (trained ckpt, unmasked centroid) | 10.529 | 98.8 % | 182 % | 0.288 | 0.130 |
| vector channels, no metal frames | 1.157 | 10.9 % | 20.0 % | 4.797 | 0.944 |
| vector channels + metal frames | 1.065 | 10.0 % | 18.4 % | 4.879 | 0.949 |

At 800 steps the same seed gives 0.092 (0.9 %) with metal frames and 3.034 (28.5 %)
without. **One seed cannot separate "metal frames help" from optimisation luck** —
at 400 steps the two arms differ by 0.09 MSE and at 800 by 2.9. What the gate does
establish is that the *architecture* moved the reachable loss from 98.8 % of the
baseline to 10 %, i.e. from "cannot fit" to "fits", and that the metal channels are
never worse than neutral at 400 steps.

An earlier unseeded version of this test produced 0.548 / 1.060 / 1.352 for the same
config, which is why `--seed` now exists: the init variance was wider than the effect
being measured. Numbers from that run are not quotable.

### 5.2 The centroid defect this exposed

`fm_velocity` pads every cloud in an FM batch to the **batch maximum** atom count
(`train_cfm_yuelbond_wasserstein.py:2370`), and the radial head then computes
`x - x.mean(dim=1)`. MEASURED on the 199-cloud train cohort at B = 8 (2,000 draws):
the resulting centroid is scaled by `s / n_max` with **mean 0.528, median 0.481,
p05 0.148**, and 52 % of samples below 0.5. For half the batch the "radial" term is
radial about a point halfway to the origin, and the shrink factor changes from batch
to batch.

This is why the A/B needs three arms, not two: `radial_centroid_masked` is a separate
opt-in (`--radial-centroid-masked`, default off so the legacy forward stays
bit-exact). The confound control is
`configs/wfa108_radial_masked_centroid.yaml`.

### 5.3 Remaining checks

1. **Bit-exact default.** `velocity_head: radial` + `radial_centroid_masked: false`
   produces the same `state_dict` keys and the same forward as the pre-Wf-A108 code
   (`test_default_head_matches_the_legacy_formula` recomputes the closed form).
2. **Equivariance.** `u(Rx + b) = R u(x)` for a random rotation/translation, to
   `2e-5` in float32 (`test_vector_head_is_se3_equivariant`).
3. **Capacity.** PASS at 10.0 % of the `v = 0` baseline (the radial head reached
   98.8 %).
4. **Transport.** 5-epoch A/B at `fm_noise_sigma: 2.485` (Wf-A107g protocol):
   **FAILED for both arms.** Generated mean pair distance 5.67 Å for the control *and*
   for the vector head, against the prior's 5.59 Å and the truth's 5.00 Å; pairs-in-cutoff
   0.197 vs 0.203 vs 0.412. The two samplers differ by 0.25 Å per coordinate and the
   head's effect on every decoded metric is exactly 0.0000. `fm_term` is flat across the
   five epochs in both arms. **Reason: 120 optimizer steps at lr 1e-4, against the 400
   steps at lr 1e-3 the capacity gate needed.** The head is no longer the binding
   constraint; the budget is.
5. **No regression on the default path**: 50-step CPU smoke exit 0 (§5.7) with the new
   head; pre-flight 115 PASS / 0 FAIL / 10 SKIP.

Failure at 4 is an honest negative and gets a catalog entry (§6.173), not a retry with
more epochs.

### 5.4 What this licenses, and what it does not

It licenses a **budget run**: the target is now inside the head's function class, so more
optimizer steps can in principle reduce the loss. It does **not** license a claim that the
c7 corpus is enough — that is exactly what the budget run measures. It also does not
establish the metal-frame channels (neutral on one seed) or the confound control
(prepared, unrun; at 5 epochs both arms are flat so it could not separate anything).

## 6. References

* Satorras, Hoogeboom, Welling. *E(n) Equivariant Graph Neural Networks.* ICML 2021. arXiv:2102.09844.
* Thomas et al. *Tensor Field Networks.* arXiv:1802.08219.
* Jing et al. *Learning from Protein Structure with Geometric Vector Perceptrons.* ICML 2021. arXiv:2009.03092.
* Vignac et al. *MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation.* ICLR 2023. arXiv:2303.04791.
* Irwin, Tibo, Janet, Olsson. *SemlaFlow — Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching.* AISTATS 2025. arXiv:2406.07266.
* Dunn, Koes. *FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation.* arXiv:2508.12629.
* Yim et al. *Fast protein backbone generation with SE(3) flow matching.* arXiv:2310.17082.
* Schaufelberger, Jorner. *Manifold Diffusion for Structure Generation of Transition Metal Complexes.* arXiv:2606.00666.
* Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023. arXiv:2210.02747.
