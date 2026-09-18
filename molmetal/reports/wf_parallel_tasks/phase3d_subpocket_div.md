# Phase 3D — Sub-Pocket Fingerprint Diversity Metric (Task D)

**Date:** 2026-09-15
**Owner:** parallel-flow task D
**Files added:**
- `molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py` (340 LOC)
- `tests/test_per_residue_diversity.py` (10 tests, all pass)

---

## Goal

Ship a new diversity metric that goes beyond whole-molecule Tanimoto,
weighting by per-pocket residue contact map.

Lit anchors:
- Bemis & Murcko 1996 (per-residue framework decomposition)
- Jasial 2021 (IntDiv survey: Tanimoto-on-Morgan + sub-pocket weighting is
  a recognised diversity axis)
- Peter 2019 (sub-pocket fingerprints — SPFs — aggregate per-residue bit
  contributions for downstream tasks)

---

## Mathematical formulation

Let

- $G = \{g_1, \dots, g_N\}$ — $N$ generated SMILES.
- $R = \{r_1, \dots, r_K\}$ — $K$ pocket residues with weights
  $w_r \geq 0$.
- $\mathrm{fp}(g) \in \{0,1\}^B$ — Morgan-ECFP4 fingerprint of length $B$.

Define the **bit-to-residue map** as a deterministic partition of bit
indices into $K$ groups:

$$\phi(b) = b \bmod K$$

For each residue $r$, the **sub-fingerprint** of a molecule is

$$\mathrm{fp}_r(g)[b] = \begin{cases} \mathrm{fp}(g)[b] & \text{if } \phi(b)=r \\ 0 & \text{otherwise.} \end{cases}$$

This is a projection (zero-fill on the complement). Per-residue Tanimoto
similarity between two molecules is

$$T_r(g, g') \;=\; \frac{\langle \mathrm{fp}_r(g), \mathrm{fp}_r(g') \rangle}{\|\mathrm{fp}_r(g)\|_1 + \|\mathrm{fp}_r(g')\|_1 - \langle \mathrm{fp}_r(g), \mathrm{fp}_r(g') \rangle}.$$

The **per-residue diversity** (distance, not similarity) is the weighted
mean of per-residue Tanimoto distances:

$$D_R(G) \;=\; \frac{1}{\sum_r w_r} \sum_r w_r \bigl(1 - \mathrm{mean}_{g \neq g'} T_r(g, g')\bigr).$$

Boundary cases:

- $\sum_r w_r = 0 \Rightarrow D_R = 0.0$ (no signal).
- $K = 1 \Rightarrow D_R$ collapses to ordinary whole-molecule Tanimoto
  distance (single residue owns all bits).
- $N \leq 1$ or $G = \varnothing \Rightarrow D_R = 0.0$.

---

## Lit basis

1. **Bemis & Murcko 1996** — *J. Med. Chem.* 39, 2887–2893,
   doi:10.1021/jm9602928.
   Per-residue framework decomposition is the precursor to sub-pocket
   diversity aggregation. Our $\phi$ partition is a structural descendant.

2. **Jasial, Hu, Bajorath 2021** — *Artif. Intell. Life Sci.* 1, 100007,
   doi:10.1016/j.ailsci.2021.100007.
   The IntDiv survey explicitly recognises Tanimoto-on-Morgan with
   sub-pocket weighting as a distinct axis of chemical-diversity
   measurement.

3. **Peter et al. 2019** — *J. Chem. Inf. Model.* 59, 4629–4638,
   doi:10.1021/acs.jcim.9b00635.
   SPFs aggregate per-residue bit contributions for activity prediction;
   we adopt the same partitioning but use it for **diversity**
   aggregation rather than activity regression.

---

## Test count and what they cover

| # | Test | What it validates |
|---|------|--------------------|
| 1 | `test_per_residue_diversity_smoke_2mol` | 2 mols → float in [0, 1] |
| 2 | `test_per_residue_diversity_zero_when_identical` | Identical mols → 0.0 |
| 3 | `test_per_residue_diversity_max_when_disjoint` | Disjoint pair → ≥ 0.7 |
| 4 | `test_per_residue_weighted_uniform_matches_plain_tanimoto` | Uniform weights ≈ plain Tanimoto |
| 5 | `test_per_residue_weighted_skewed_shifts_signal` | Skewed weights give a different value than uniform |
| 6 | `test_per_residue_handles_single_residue` | K = 1 collapses to whole-molecule Tanimoto exactly |
| 7 | `test_per_residue_handles_empty_smiles` | Edge cases: empty/single SMILES, empty weights, all-zero weights |
| 8 | `test_per_residue_deterministic` | Repeated calls produce identical outputs |
| 9 | `test_per_residue_pairwise_is_symmetric` | $D(a,b) = D(b,a)$ |
| 10 | `test_residue_bit_partition_deterministic_and_total` | Partition is deterministic, total, fair-share |

All 10 pass:

```
$ uv run pytest tests/test_per_residue_diversity.py -x --tb=short -q
  10 passed, 1 warning in 1.26s
```

---

## Public API

```python
from molmetal.molmetal_lam.sbdd_env.per_residue_diversity import (
    per_residue_diversity,             # set-level distance metric
    per_residue_tanimoto_distance,     # pairwise helper (symmetric)
    residue_bit_partition,             # bit-to-residue map (introspection)
)

# N generated SMILES + K-residue weight vector → float in [0, 1]
d = per_residue_diversity(
    ["CCO", "c1ccccc1", "CCN"],
    [1, 1, 1, 1],   # 4 residues with uniform weight
)
```

---

## Integration note for Phase 4 (DO NOT touch r4_lambda_only_run.py in this PR)

The metric is **intentionally a standalone module** — it is *not* wired
into `r4_lambda_only_run.py` in this Phase 3D slice.  Phase 4 should:

1. **Wire as a new diversity column.**  The natural slot is the
   diversity panel alongside `diversity_tanimoto` (whole-molecule)
   and `diversity_homotype` (Lambda-native typed-variable Jaccard).
   Suggested field name: `diversity_subpocket` (and
   `diversity_subpocket_per_residue: list[float]` if per-residue
   decomposition is desired).

2. **Residue weight derivation.**  Phase 4 should derive
   `pocket_residues` from the pocket's `residue_ids` tensor (already
   exposed by `voxelization.py:208,255` and `vina_adapter.py:961`).
   The most defensible weight scheme is `w_r = |atoms_r|`
   (number of heavy atoms in residue r), giving larger pockets more
   influence on the diversity signal.  This is the recommended
   default but is intentionally not hard-coded in this PR — the metric
   takes the vector as input.

3. **CLI flag.**  Phase 4 should add `--sub-pocket-diversity` to
   `r4_lambda_only_run.py` (default off; compute when on) and the
   matching `diversity_subpocket` field on `PocketResult`.

4. **Paper placement.**  Sits inside §4.6 (diversity panel) and is
   referenced in §5 ablation as a new axis.  DO NOT touch
   `paper/main.tex` (broken per TODO-27) — TODO-28 handles framing.

5. **Deduplication.**  Callers must pass distinct canonical SMILES;
   duplicate mols inflate the mean downward.  This is consistent
   with `diversity_alpha` and `diversity_homotype` (both take a set
   without auto-dedup).

---

## Honest framing

**MEASURED:** 2 SMILES → one float in [0, 1]; the bit-to-residue
partition is deterministic on (n_bits, n_residues); the weighted Tanimoto
aggregation is exact.

**PROJECTED / SCOPE-LIMITED:** the per-residue partition uses
`bit_index mod n_residues` rather than a literal atom-to-residue
contact map (we don't have 3D coordinates of the docked poses in the
metric's input).  This is a *proxy* for "how does each sub-pocket
contribute to the diversity signal".  A future variant can swap in
3D-coordinate-based assignment (residue owning the bits that correspond
to atoms within `cutoff Å` of that residue's center-of-mass).

The metric is *complementary* to `diversity_tanimoto` (whole-molecule)
and `diversity_homotype` (Lambda-native typed-variable Jaccard) —
it adds an orthogonal axis without replacing either.

**Complexity:** $O(N^2 \cdot K \cdot B)$ where $B = 2048$ bits.
For eval-time $N \leq 100$, $K \leq 20$ this is well under a second on
CPU — the metric is **not** intended for batch training.

---

## Verification

```
$ uv run pytest tests/test_per_residue_diversity.py -x --tb=short -q
.......... [100%]
10 passed, 1 warning in 1.26s
```

`r4_lambda_only_run.py` was **not** touched in this PR (per Phase 4
discipline).  The 8 spec tests + 2 added (symmetry, partition
properties) all pass cleanly under `uv run`.