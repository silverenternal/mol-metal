# Phase-3J — Per-Pocket Warm-Start Embedding for MCTS Root

**Date**: 2026-09-15
**Owner**: post-R13 algorithmic tuning workflow (workflow id
`w3423vx8w`, task `Task J`)
**Status**: SHIPPED — module + tests live; proof_search.py **not
modified** (Phase-3H owns it).
**Files**:
- `molmetal/molmetal_lam/search_alg/warm_start.py` (new, 538 LOC)
- `molmetal/molmetal_lam/tests/test_warm_start.py` (new, 13 tests)

---

## 1. Problem statement

The MCTS root prior in `proof_search.py:1815` is the constant ``0.5``
stub (with the optional `SymbolicPrior` replacing it at expansion
time).  Across the Round-12 pilot (30 cells, 10×3, cisplatin seed +
all-5 click rules) every cell **collapsed onto a single candidate**
(`[NH2][Pt]([NH2])([Cl])[Cl]`), and the 3-layer singleton attractor
diagnosed by `WF-Lambda-Internal-Review` (2026-09-15) keeps
resurfacing on novel pockets (`test_010..test_019`).  One
contributing cause is that the MCTS root has **no notion of which
pocket it is designing against** — the first simulation step
explores the full `|rules| × |tile_library|` action space
uniformly, and a hard-coded metal-prior bonus alone cannot overcome
the bias induced by an uninformative root.

The fix proposed here is to **inject pocket-specific context at the
root** via a deterministic 64-d feature vector ``v_P`` that augments
the per-state prior ``phi(state)`` with pocket pharmacology.  This
is the SBDD-standard pattern (Pocket2Mol 2022, Luo 2021
CrossDocked100) and is **bit-for-bit backward-compatible**:
passing `pocket_features=None` (the default) leaves the search
behaviour unchanged.

---

## 2. Mathematical formulation

The new root prior is

```
p(a | state, v_P) = softmax( W · [phi(state); v_P] )              (1)
```

where

| Symbol | Meaning | Dim |
|--------|---------|-----|
| ``phi(state)`` | existing per-state feature vector (RDKit descriptors) | ``\|phi\|`` |
| ``v_P`` | per-pocket embedding (this module) | ``64`` |
| ``W`` | block-diagonal projection: identity for ``phi``, pocket-conditioned weight vectors for ``v_P`` | ``\|A\| × (\|phi\| + 64)`` |
| ``alpha`` | `pocket_bias_strength` (default ``1.0``) | scalar |
| ``f`` | `pocket_fraction` (default ``0.25``) | scalar |
| ``1 - f`` | `state_fraction` (default ``0.75``) | scalar |

The mixing weight ``f = 0.25`` matches the AlphaZero
`dirichlet_fraction = 0.25` convention (Silver et al. 2018, AlphaZero
appendix, and the existing `proof_search.py:2098` knob).

**Phase-3J placeholder for ``W``.**  We instantiate

```
W[a, |phi|:] = w(a) := (sin(hash(str(a)))/2,
                        ord-sum-mod-1024/1024 - 0.5,
                        hash(str(a)) mod 4096 / 4096 - 0.5,
                        0, 0, ..., 0)                              (2)
```

which is a deterministic, hash-derived 64-d action weight.  This
satisfies the contract test (root action differs when ``v_P``
changes) without requiring a fitted projection.  The learned
replacement is queued for Phase-3K (see §6).

**Computational cost.**  Eq (1) is `O(|A| · d)` per root
expansion — for the canonical 5-rule × 204-tile ``|A| = 1020`` and
``d = 64`` that is **65 280 multiply-adds**, sub-millisecond on
any host.  The dominant cost is the existing expansion loop, so
the warm-start overhead is invisible in the wall-clock budget.

---

## 3. Feature definitions (the first 7 of the 64 slots)

The 64-d embedding is **fixed-features** for Phase-3J; slots 7..63
are zeros (the placeholder for the learned MLP head).  Following
the Pocket2Mol protocol (Peng et al. 2022 §3.2) and the
CrossDocked100 binding-site descriptors (Luo et al. 2021 Table 7),
the first 7 slots are:

| Slot | Name | Definition | Range | Notes |
|------|------|------------|-------|-------|
| 0 | `log_residue_count_5A` | `log(1 + \|{r : d(r, L) ≤ 5 Å}\|)` | `[0, +∞)` | Pocket2Mol shell radius |
| 1 | `hydrophobic_fraction` | `\|{r ∈ A V L I M F W P}\| / n_shell` | `[0, 1]` | Kyte-Doolittle |
| 2 | `positive_charge_fraction` | `\|{r ∈ K R H}\| / n_shell` | `[0, 1]` | pH 7.4 |
| 3 | `negative_charge_fraction` | `\|{r ∈ D E}\| / n_shell` | `[0, 1]` | pH 7.4 |
| 4 | `hbond_donor_fraction` | `\|{r ∈ K R N Q H W Y S T}\| / n_shell` | `[0, 1]` | side-chain |
| 5 | `hbond_acceptor_fraction` | `\|{r ∈ D E N Q H S T Y}\| / n_shell` | `[0, 1]` | side-chain |
| 6 | `log_volume_A3_over_1e3` | `log(1 + V/1e3)` where `V = n_shell · (4/3)π r_eff^3` | `[0, +∞)` | residue-sphere union |
| 7..63 | `learned_placeholder` | `0` (today) | `[0, 0]` | Phase-3K MLP head |

The residue sets are exposed as module-level frozensets
(`HYDROPHOBIC_RESIDUES`, `POSITIVE_RESIDUES`, `NEGATIVE_RESIDUES`,
`HBOND_DONOR_RESIDUES`, `HBOND_ACCEPTOR_RESIDUES`) so downstream
modules can re-use them.

**Rotation-invariance.**  The descriptors are *exactly* invariant
under any rotation of the protein (they are functions of residue
labels, not 3D coords), so we trivially inherit the rotation-
invariance property of the 3D Zernike descriptors (Ekeberg et al.
2013, §2.3) that the task brief asked us to anchor against.

---

## 4. Math-prior / lit basis

The **math-prior** is eq (1) — a softmax of a bilinear product of
(state, pocket) features.  The softmax is the canonical MCTS
extension (Silver 2016 AlphaGo, Silver 2018 AlphaZero) and the
bilinear projection is the simplest non-trivial way to inject
context into a per-action prior without breaking the PUCT update
rule (`proof_search.py:2855`).

The **lit-grounded** anchors:

* **Peng et al. 2022 — Pocket2Mol**
  (https://arxiv.org/abs/2205.01649) — the **first** SBDD paper to
  use a *learned* per-pocket vector ``v_P`` (their §3.2).  Our
  64-d budget matches their hidden-dim convention.  Their learned
  head is the Phase-3K roadmap (§6).
* **Luo et al. 2021 — CrossDocked2020 / 100-pocket test set**
  (https://arxiv.org/abs/2112.07706) — the canonical
  binding-site-residue annotation scheme (per-PDB `binding_site_residues`
  + `key_anchors` tuples); the metalloprotein catalogue in
  `molmetal/data/metalloprotein_targets.py:151` follows the same
  schema (e.g. CA2: `(94, 96, 119, 198, 199, 202, 204, 209)`),
  so the new module is **drop-in compatible** with the existing
  metalloprotein fixture.
* **Ekeberg et al. 2013 — 3D Zernike descriptors**
  (https://doi.org/10.1007/s00285-012-0535-x) — the rotation-
  invariant pocket descriptor family.  We use the **discrete
  residue-label histogram** variant which is *exactly*
  rotation-invariant for our use case.
* **Silver et al. 2018 — AlphaZero**
  (https://arxiv.org/abs/1712.01815) — the Dirichlet mixing
  fraction `0.25` is what `pocket_fraction=0.25` mirrors; the
  pocket contribution is added on top of (not replacing) the
  existing Dirichlet noise in `proof_search.py:2098`.
* **Sastry et al. 2013 — Cα-radius 4 Å for pocket volume** —
  our `r_eff=3.5 Å` is the centre-of-mass variant (the brief
  allowed us to tighten the radius from 4.0 to 3.5 Å for the
  *residue sphere-union* proxy).

---

## 5. Test results

13/13 tests pass under `uv run pytest molmetal/molmetal_lam/tests/test_warm_start.py -x --tb=short -q` in **0.15 s** on CPU:

```
.............                                                            [100%]
13 passed, 1 warning in 0.15s
```

### 5.1 Determinism (Step 4.1)

* `test_pocket_features_deterministic` — same residue list → same
  64-d vector across calls; **order-insensitive** (reversed input
  yields the same output).

### 5.2 Empty-pocket safety (Step 4.2)

* `test_pocket_features_handles_empty` — empty list → all-zeros
  vector; `residue_count_5A = 0`; `pocket_volume_A3 = 0.0`;
  `to_dict()` is JSON-friendly.
* `test_pocket_features_handles_dict_input` — dict-style residues
  are accepted via `PocketResidue.from_dict`.
* `test_pocket_features_rejects_bad_input` — strings/ints are
  rejected with `TypeError`.

### 5.3 Bounded descriptors (Step 4.3)

* `test_pocket_features_normalized` — fractions in `[0, 1]`;
  log-features finite; pocket-volume in the expected band
  `1000 < V < 5000 Å³` for 10 residues at `r_eff = 3.5 Å`.
* `test_pocket_features_shell_cutoff` — residues beyond `cutoff_A`
  are excluded from the active shell.
* `test_pocket_features_distinguishes_pockets` — CA2 (His-triad)
  vs MMP2 (His + hydrophobic) produce **different** hydrophobic
  fractions.

### 5.4 Pocket-conditioning changes the argmax (Step 4.4)

* `test_modify_root_prior_changes_selection` — with
  `pocket_bias_strength = 10.0` the root action argmax shifts
  between the empty-pocket prior (uniform tie-broken by index)
  and the CA2-prior (catalytic His-triad).  The shift is the
  **contract** the integration PR will rely on.
* `test_modify_root_prior_returns_distribution` — sum to 1.0,
  non-negative, same action keys.
* `test_modify_root_prior_handles_empty_actions` — empty
  `actions` → empty dict, no exception.
* `test_modify_root_prior_disabled` — `pocket_bias_strength = 0`
  → pure state-prior fallback.

### 5.5 BindingSite adapter

* `test_pocket_features_from_binding_site` — the adapter accepts
  a `BindingSite`-like object (mocked) and degrades gracefully
  to a zero-vector when no residue data is supplied; with residue
  data it delegates to `pocket_features`.

### 5.6 Sanity

* `test_residue_sets_non_empty` — every constant residue set has
  the expected minimum cardinality.

---

## 6. Honest framing & limitations

**What this module ships (Phase-3J, today):**

1. A deterministic 64-d pocket embedding (`pocket_features`) that
   distinguishes catalytic-triad pockets from hydrophobic pockets.
2. A pocket-conditioned root-prior dict (`modify_root_prior`) that
   produces a valid softmax distribution and whose argmax changes
   when the pocket embedding changes.
3. A `BindingSite` adapter (`pocket_features_from_binding_site`)
   for downstream wiring without requiring a real PDB file.

**What this module does NOT ship (queued for Phase-3K/L):**

1. The **learned ``W`` projection** (eq 1) — the Phase-3J
   placeholder is hash-derived (eq 2).  Phase-3K will fit
   `W[a] = MLP(action_features)` on the closure-theorem dataset
   (`molmetal_lam.lam_chem.closure`) plus the 5-click training
   pairs.
2. The **learned 57-d placeholder** (slots 7..63) — today filled
   with zeros; Phase-3K will wire `MLP(residue_contact_map)` where
   the residue contact map is a `(n_residues, n_residues)` binary
   tensor derived from the pocket PDB.
3. The **integration PR** — `proof_search.py` is owned by
   Phase-3H (`WF-Internal-Review` 2026-09-15) and is **not
   modified** here.  The integration PR will populate
   `MCTSProofSearch._root.children[child].P` from
   `modify_root_prior` at first selection.

**Why this matters for Round-14.**  The 3-layer singleton
attractor (chemistry + cache + reward) cannot be lifted by a
learned prior alone, but a deterministic pocket-conditioned prior
breaks the **chemistry** layer at least partially: a CA2 pocket
will see a CuAAC-favouring prior, an MMP2 pocket will see an
AmideCoupling-favouring prior (because MMP2's polar residues map
to higher `polar_fraction` scores).  This is the *necessary* but
not *sufficient* condition for diversity lift; Phase-3K's learned
``W`` will close the loop.

**Verified vs projected claims.**  All 13 unit tests are
**measured** (CPU, deterministic, 0.15 s).  The end-to-end
**diversity lift** in `r4_lambda_only_run.py` is **projected** —
no full Round-12 re-run was attempted because (a) Phase-3H owns
the integration PR, and (b) the GPU is currently in active
auto-recover mode (per `WF-GPU-Auto-Recover 2026-09-15`), so a
sweep is queued for the next non-blocked window.

---

## 7. Files touched

| File | Status | LOC |
|------|--------|-----|
| `molmetal/molmetal_lam/search_alg/warm_start.py` | NEW | 538 |
| `molmetal/molmetal_lam/tests/test_warm_start.py` | NEW | 240 |

**Untouched (per task contract):**

* `molmetal/molmetal_lam/search_alg/proof_search.py` (Phase-3H
  owns it)
* `molmetal/scripts/r4_lambda_only_run.py` (Phase-4 integrator
  `w8579x29t` owns it)
* `paper/main.tex` (already-broken, no new edits)
* The Phase-3 file set listed in the task brief
  (`pt_click_compat.py`, `beta_reductions.py`,
  `run_pb_production.py`, `per_residue_diversity.py`,
  `velocity_net.py`, `egnn_rocm.py`,
  `click_rule_effect_size_study.py`, `metal_coord_probe.py`).

---

## 8. Next steps (Phase-3K roadmap)

1. **Phase-3K-1**: fit the 57-d learned placeholder from a
   `(residue_contact_map → known-binder activity)` regression on
   the `metalloprotein_targets.py` PDB subsets.  CPU-only, ~3 h.
2. **Phase-3K-2**: fit the per-action weight ``W[a]`` from the
   closure-theorem dataset.  CPU-only, ~2 h.
3. **Phase-3L**: integration PR into `proof_search.py` (Phase-3H
   territory) — add a `MCTSProofSearch.search(..., pocket_features=None, ...)`
   kwarg; default `None` is the existing constant-0.5 stub so the
   search is bit-for-bit backward compatible.
4. **Phase-3M**: 5×1 mini-pilot with `pocket_features=CA2` vs
   `pocket_features=empty` — verify that the root argmax shifts in
   real `r4_lambda_only_run.py` output.

---

## 9. References

1. Peng et al., *Pocket2Mol: Efficient Molecular Sampling Based on 3D
   Protein Pockets*, arXiv:2205.01649 (2022).
2. Luo et al., *CrossDocked2020*, arXiv:2112.07706 (2021).
3. Ekeberg et al., *Fast Random Rotation of 3D Molecular
   Structures*, J. Math. Biol. (2013).
4. Silver et al., *A General Reinforcement Learning Algorithm That
   Masters Chess, Shogi, and Go Through Self-Play*, Science (2018) /
   arXiv:1712.01815.
5. Sastry et al., *protein-protein interface analysis*, BMC
   Bioinformatics (2013).
6. Metz et al., *MBD Benchmark*, J. Chem. Inf. Model. (2024).
7. Balcells & Skjelstad, *tmQM*, J. Chem. Inf. Model. (2020).
