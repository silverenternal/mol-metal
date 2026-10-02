# `multi_ph_integration` Adapter — Paper Reference (Wf-313C)

**Adapter:** `molmetal/adapters/multi_ph_integration.py`
**Ship status:** W2.3 (CPU smoke shipped; R18-P3 paper-grade pending)
**Owner:** Hugo Lee (audit) — multi-surf integration per TODO-234 §6

---

## 1. Paper citation

> Kim, H. & Jung, J. (2026). **Fourier-Accelerated Computation of Multi-Parameter
> Persistent Homology for Molecular Shape Analysis.** *Advances in Applied
> Mathematics* **174**, 103023. doi:10.1016/j.aam.2025.103023.

The adapter uses the `CountGrid` statistic from §4.1 (the multi-persistence
analogue of the single-parameter Betti curve) and the 2-Wasserstein stability
bound from Th.1. Reference cite key: `kim_jung2026multi_ph`.

---

## 2. API surface (per Wf-309B honest framing)

| Symbol | Type | Purpose |
|---|---|---|
| `MultiPHPocketFeatures(max_radius=5.0, max_coord=4.0, n_bins=32)` | `nn.Module` | Stateless pocket feature extractor; no learnable params |
| `.compute(pocket_clouds: (B, P, 3)) -> (B, n_bins, n_bins)` | method | Feature map; default atom-types are 1-hot fallback |
| `.compute_with_atom_types(pocket_clouds, pocket_atom_types) -> (B, n_bins, n_bins)` | method | Same contract as `PocketEncoder.forward` |
| `.forward(pocket_clouds) -> (B, n_bins, n_bins)` | method | `nn.Module`-style alias for `compute` |
| `MultiPHMetalCoordHead(n_metals=12, kernel="sliced_wasserstein", sigma=1.0, n_projections=8, ...)` | `nn.Module` | Per-metal coord classifier via sliced-Wasserstein kernel |
| `.set_reference(reference_coords, reference_atom_types)` | method | Load per-metal reference diagrams as buffer |
| `.forward(pocket_features: (B, n_bins, n_bins)) -> (B, n_metals)` | method | Returns *logits* (not probabilities) |
| `augment_pocket_embedding(base_pocket_emb: (B, H), multi_ph_features: (B, n_bins, n_bins)) -> (B, H + n_bins**2)` | function | Drop-in additive augmentation of base `PocketEncoder` output |

**Math invariants** (pinned by `molmetal/math_skills/multi_parameter_ph.py`):
* `_EPS_LOG = 1e-30`, `_FOURIER_THRESHOLD = 1024`
* Naive O(N²) pair-distance path — Fourier-accelerated path (R18-P3) is
  TODO above 1024 atoms
* H₀ and H₁ only — multi-parameter PH on ℝ² is *not* finitely generated
  in general (Miller 2020 §6)
* W2.3 smoke budget: `N <= 200` atoms per pocket

---

## 3. Honest reproduction caveats

* **CPU-only.** The 2-PH filtration uses `torch.cdist` + `torch.linalg` on
  the CPU backend; no CUDA / Triton kernels are required for the W2.3
  smoke contract. Production-scale pockets (`N > 200`) need the R18-P3
  Fourier path (NOT implemented in the adapter).
* **No modification of `flow_matching_lipman/`.** The adapter is
  *additive*: `augment_pocket_embedding` concatenates the multi-PH
  feature map to the base `PocketEncoder` output.  The downstream
  velocity field can either consume the augmented embedding as a
  replacement for `pocket_embed` or ignore it entirely.
* **Naive 2-Wasserstein stability.** The adapter uses sliced-Wasserstein
  (Rabin 2011) with `n_projections=8`; the *true* 2-Wasserstein cost is
  approximate (variance ~O(1/K)).  The Kim-Jung 2026 stability bound is
  the upper envelope — empirical metrics may be sharper.
* **`max_coord = 4.0` clips the heavy tail.** Atoms with
  `|z_i - z_j| > 4` (e.g. Mn vs. H) fall into the last coordination bin.
  This is intentional and matches the W2.3 smoke budget — the clip
  reduces the feature map size at the cost of some discrimination on the
  transition-metal block.
* **`use_sinkhorn=False` for bit-exact testing.** The SB coupling is
  optional (`MultiPHMetalCoordHead.kernel = "sliced_wasserstein"`) and
  does NOT use `LogSinkhornLoss`; tests that need bit-exact output should
  set `sigma` to a fixed value and `seed=0` to fix the random projections.

---

## 4. Drop-in integration points

| Wire-in | Where in the pipeline | Replacement? |
|---|---|---|
| `MultiPHPocketFeatures.compute` | Augments `PocketEncoder.forward(pocket_coords, pocket_atom_types, pocket_mask)` output | **Augments** — concatenates `(B, n_bins**2)` to the `(B, H)` base embedding |
| `MultiPHMetalCoordHead.forward` | Per-metal reference diagram comparison | **Augments** — adds `(B, n_metals)` coord logits next to the existing head |
| `augment_pocket_embedding` | Drop-in replacement for the `(B, H)` pocket embedding | **Augments** — returns `(B, H + n_bins**2)` tensor |

**Honest gap:** the adapter does NOT replace `PocketEncoder` — it augments
alongside.  W2.3 recommends adapting the velocity field to accept
`hidden_dim = H + n_bins**2` so the augmented embedding can be used
end-to-end; that refactor is left for the R18-P3 integration step.

**Production readiness:** W2.3 smoke ✅; R18-P3 paper-grade ❌ (Fourier
acceleration + n_atoms>200 path are TODO above the W2.3 budget).

---

## 5. Cross-references

* Wf-310F Day 1-2 build-only tests: `molmetal/tests/test_wf310f_frontier_math_integration.py`
* Wf-313C Day 3-5 paper-grade tests: `molmetal/tests/test_wf313c_frontier_math_integration_e2e.py::test_multi_ph_pocket_augmentation`
* TODO-234 §6 Day 1-2 execution plan: `TODO/pending/234_wf234_frontier_math_audit_integration.md`
* Math skill source: `molmetal/math_skills/multi_parameter_ph.py`
* Adapter module map: `molmetal/adapters/multi_ph_integration.py` §module map

---

**END OF REF** — Owner Hugo Lee; MEASURED + W2.3 status; cite `kim_jung2026multi_ph`.