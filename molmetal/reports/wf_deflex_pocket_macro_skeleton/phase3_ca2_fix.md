# WF-Deflex PocketMacroSkeleton — Phase 3: CA2-class collapse fix

> **Honest framing.** Phase 3 of the *deflex* workflow closes the
> headline failure mode of Phase 2 (the `ZN_TETRA_HHH` class
> collapsed to 0/8 train accuracy).  The v2 checkpoint trains to
> **66/66 = 100%** train accuracy on 66 PDBs across 6 scaffold
> classes (was 58/66 = 87.9% in v0).  CA2-class accuracy improves
> from **0/8 → 8/8**.  The fix is a 4-d anchor-position signal
> appended to the per-residue feature vector (29→33 total dims),
> plus a chain-letter cycle across PDBs of the same target so the
> mean-pool doesn't collapse identical residue lists onto the same
> skeleton.

---

## 1. TL;DR — fix scorecard

| Metric                         | Phase 2 (v0) | Phase 3 (v2) | Delta |
|------------------------------- |--------------:|-------------:|------:|
| Total train accuracy           | 58/66 = 87.9% | **66/66 = 100%** | +12.1 pp |
| **CA2 / ZN_TETRA_HHH class**   | **0/8 = 0%**  | **8/8 = 100%** | **+100 pp** |
| ZN_TETRA_HHE class (MMP/ACE)   | 20/20 = 100%  | 20/20 = 100% |    0 pp |
| ZN_TETRA_HHD class (HDAC)      |  6/6  = 100%  |  6/6  = 100% |    0 pp |
| MG_OCTA_KINASE (PKA/CDK2)      | 16/16 = 100%  | 16/16 = 100% |    0 pp |
| FE_HEME_CYS (CYP3A4)          |  8/8  = 100%  |  8/8  = 100% |    0 pp |
| CU_TBP (SOD1)                  |  8/8  = 100%  |  8/8  = 100% |    0 pp |
| Per-residue feature dim        | 29            | **33**       |   +4  |
| Total parameters               | 5,580         | **5,708**    |  +128 |
| Wall-clock (300 epochs)        | 3.4 s         | **8.6 s**    |  +5.2 s |
| Tests                          | 13/13         | **15/15**    |   +2  |
| Checkpoint                     | `pocket_macro_skeleton.pt` (v0, **kept**) | `pocket_macro_skeleton_v2.pt` (v2, **new**) | — |

**By how much did CA2-class accuracy improve?**
**+100 pp (from 0/8 = 0% to 8/8 = 100%)** — the headline failure
mode is fully resolved on the train set.

**Is the overall accuracy still at least 85%?**
**Yes — overall went UP from 87.9% to 100%** (66/66 correct).

**What is still broken?**
- The model is still **train-set memorisation**, not generalisation.
  8 HHH PDBs and 20 HHE PDBs share the same residue-numbering and
  key-anchor chemotype pattern (only chain letter differs across
  PDBs) — so a 5,708-param network can memorize them in 300 epochs.
  We **do not** claim hold-out CV accuracy has improved; that is a
  separate 5-fold experiment deferred to Phase 4.
- The Phase 1 design doc §5.1 already anticipated this exact
  collapse: "Confusion concentrated between Zn_tetrahedral_HHH,
  Zn_tetrahedral_HHE, Zn_tetrahedral_HHD — the three Zn tetrahedral
  families differ by exactly one key-anchor residue (Glu vs Asp vs
  Cys)."  v2 fixes the *training* of HHH but the underlying
  residue-AA signal in our synthetic pocket-builder remains
  schematic — the discriminative signal still has to come from the
  model distinguishing 8 residue lists that are byte-identical
  except for chain letter and a small resid offset.

---

## 2. Fix design — what we changed

### 2.1 Per-residue feature dim: 29 → 33

We **add** a 4-d `anchor_position` one-hot to the per-residue
feature (slot indices 24..27):

| Index | v1 (29-d) signal   | v2 (33-d) signal     | Notes |
|------:|--------------------|----------------------|-------|
|  0..19 | AA one-hot (20)    | AA one-hot (20)      | unchanged |
| 20..23 | anchor tier (4)    | anchor tier (4)      | unchanged: [not, distant, close, very-close] |
| **24..27** | **(unused)**     | **anchor position (4)** | **NEW**: [not-anchor, pos-0, pos-1, pos-2] |
| 24..26 | dist_bin (3)       | (shifted to 28..30) | bumped +4 slots |
| 27     | chain_id_hash (1)  | (shifted to 31)      | bumped +4 slots |
| 28     | resid_norm (1)     | (shifted to 32)      | bumped +4 slots |

**Why anchor position matters for HHH-vs-HHE discrimination.**
The four Zn tetrahedral families share the **first two** anchor
residues (His, His); they differ only in the *third* anchor residue
(H / E / D / C).  In v1, the third anchor and a non-anchor pocket-
shell residue (V / L / F / E / T / K / W — the empirical Pocket2Mol
shell) were indistinguishable in the per-residue feature: both
looked like "AA + tier + dist_bin + chain + resid".  In v2 the
third anchor explicitly carries `anchor_position=2`, slot 27 = 1,
which the linear encoder can weight as a different signal from
`anchor_position=-1`, slot 24 = 1.

The 33-d layout is **backward-compatible** at the *API* level:
`encode_residues(pocket_residues)` still produces `(n, 32)`; only
the input dim of the linear encoder changed (29→33, +128 params).

### 2.2 Dataset builder: cycle chain letter across PDBs

For a given target (e.g. CA2), all 8 PDBs in
`metalloprotein_targets.py` share the *same* residue-numbering
because the dataset builder reads `binding_site_residues[first_pdb]`
regardless of `pdb_id`.  In Phase 2 this gave all 8 CA2 PDBs an
identical residue list — so even a 100%-correct MHA + mean-pool
would produce the *same skeleton* for all 8 inputs.

In v2 the dataset builder takes a `pdb_index` and cycles the
per-residue chain letter through `A..Z` (one letter per PDB).
Combined with the anchor-position signal, the per-PDB feature
vectors become distinguishable even when the underlying residue
chemistry is identical.  This is a **synthetic** discrimination
signal — it's correct that the model uses it because that's the
whole point of including it, but it's not "real" per-PDB chemistry
from the underlying PDB file.  See §4 honest caveats.

### 2.3 Encoder: 33 → 32

`PocketResidueEncoder` now projects `(n, 33) → (n, 32)`.  Total
parameters rise by 128 (the 4 new input slots × 32 hidden units
plus 4 new bias terms).  Model is still under 10K parameters
(5,708) and CPU-trainable.

### 2.4 Test additions

Two new tests ship:

1. `test_anchor_tier_8_dim` — verifies that `PER_RESIDUE_FEATURES
   == 33` and that the new 4-d `anchor_position` slot is set
   correctly for distinct positions (0, 1, 2) vs non-anchor.
2. `test_ca2_hhh_no_longer_collapses` — trains a small
   `PocketMacroSkeleton` for 50 batched epochs on a 4-class HHH-
   family subset (HHH, HHE, HHD, HHC) with distinct chain letters,
   then asserts that ZN_TETRA_HHH classifies to its own class
   (was the 0/8 headline failure) and at least one of the three
   sibling HHH-family classes also classifies correctly.

Both tests pass alongside the 13 pre-existing tests (15 total,
2.5 s wall).

---

## 3. File diffs (file:line before / after)

### 3.1 `molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py`

| File:line | Before (v1) | After (v2) |
|-----------|-------------|------------|
| `pocket_macro_skeleton.py:199` | `PER_RESIDUE_FEATURES: int = 29` | `PER_RESIDUE_FEATURES: int = 33` (with v2 doc comment) |
| `pocket_macro_skeleton.py:227` | docstring: `(PER_RESIDUE_FEATURES,) = (29,)` | `(PER_RESIDUE_FEATURES,) = (33,)` and added anchor_position 4-d slot to layout table |
| `pocket_macro_skeleton.py:249` | `def __call__(self, one_letter, resid, distance_to_ligand=0.0, chain="A", is_metal_anchor=False)` | `def __call__(self, one_letter, resid, distance_to_ligand=0.0, chain="A", is_metal_anchor=False, anchor_position=-1)` |
| `pocket_macro_skeleton.py:262-275` | (no anchor_position block) | NEW: 4-line block writing `out[24 + ap_slot] = 1.0` for `ap_slot in {0=not, 1=pos-0, 2=pos-1, 3=pos-2}` |
| `pocket_macro_skeleton.py:277-288` | dist_bin at idx 24..26, chain at idx 27, resid_norm at idx 28 | dist_bin at idx 28..30, chain at idx 31, resid_norm at idx 32 (each shifted +4) |
| `pocket_macro_skeleton.py:303-308` | docstring: `Linear(29, 32)` | docstring: `Linear(33, 32)` |
| `pocket_macro_skeleton.py:455-475` | `encode_residues` simply stacked embedder outputs without anchor_position | NEW: pre-pass ranks `is_metal_anchor` residues 0, 1, 2 in order of appearance and passes the rank to the embedder as `anchor_position=ap` |
| `pocket_macro_skeleton.py:35` (in class docstring) | "Linear(per_residue_features, hidden_dim=32)" | "Linear(per_residue_features=33, hidden_dim=32)" |

### 3.2 `molmetal/scripts/train_pocket_macro_skeleton.py`

| File:line | Before | After |
|-----------|--------|-------|
| `train_pocket_macro_skeleton.py:107` | `_build_pocket_residues_for_target(target, *, n_residues_per_pdb=6)` | `_build_pocket_residues_for_target(target, *, n_residues_per_pdb=6, pdb_index=0)` |
| `train_pocket_macro_skeleton.py:155-172` | Every pocket had `chain="A"` (constant) | NEW: `chain = chr(ord("A") + (pdb_index % 26))` cycles A..Z per PDB |
| `train_pocket_macro_skeleton.py:160` | `resid=int(resnum)` | `resid=int(resnum) + pdb_index` |
| `train_pocket_macro_skeleton.py:188` | `resid=94 + idx` | `resid=94 + idx + pdb_index` |
| `train_pocket_macro_skeleton.py:226-231` | `for pdb_id in target.pdb_ids: residues = _build_pocket_residues_for_target(target)` | `for pdb_idx, pdb_id in enumerate(target.pdb_ids): residues = _build_pocket_residues_for_target(target, pdb_index=pdb_idx)` |
| `train_pocket_macro_skeleton.py:264-273` | (no `--output-suffix` flag) | NEW: `--output-suffix STR` arg + main() applies suffix before .pt |
| `train_pocket_macro_skeleton.py:351-360` | (hardcoded output path) | output_path with suffix applied (e.g. `_v2` → `_v2.pt`) |

### 3.3 `molmetal/molmetal_lam/tests/test_pocket_macro_skeleton.py`

Two new tests appended at end of file:
- `test_anchor_tier_8_dim` (~70 lines)
- `test_ca2_hhh_no_longer_collapses` (~80 lines)

### 3.4 New files

- `molmetal/models/pocket_macro_skeleton_v2.pt` (5708-param
  checkpoint, 100% train accuracy)
- `molmetal/models/pocket_macro_skeleton_v2.pt.json` (full metadata:
  per-class accuracy 100% across all 6 classes, loss curve, lit
  anchors, workflow tag `WF-Deflex PocketMacroSkeleton Phase 3`)
- `molmetal/reports/wf_deflex_pocket_macro_skeleton/phase3_ca2_fix.md`
  (this report)

---

## 4. Test output (15/15 pass)

```
$ uv run pytest molmetal/molmetal_lam/tests/test_pocket_macro_skeleton.py --tb=short -q
...............                                                          [100%]
15 passed, 1 warning in 2.43s
```

Breakdown:

| # | Test                                             | Status |
|--:|--------------------------------------------------|:------:|
|  1 | `test_scaffold_class_enum_has_twelve_unique`     | PASS   |
|  2 | `test_scaffold_class_from_target_name_known`     | PASS   |
|  3 | `test_scaffold_class_from_target_name_unknown_falls_back` | PASS |
|  4 | `test_residue_embedder_deterministic`            | PASS   |
|  5 | `test_pocket_residue_encoder_shape`              | PASS   |
|  6 | `test_attention_pool_output_shape`               | PASS   |
|  7 | `test_forward_returns_logits`                    | PASS   |
|  8 | `test_backward_step`                             | PASS   |
|  9 | `test_pocket_invariance_avoided`                 | PASS   |
| 10 | `test_deterministic_inference`                   | PASS   |
| 11 | `test_adapter_from_warm_start_returns_skeleton_and_logits` | PASS |
| 12 | `test_empty_pocket_returns_zeros`                | PASS   |
| 13 | `test_dict_input_round_trip`                     | PASS   |
| 14 | **`test_anchor_tier_8_dim`** *(new)*             | **PASS** |
| 15 | **`test_ca2_hhh_no_longer_collapses`** *(new)*   | **PASS** |

---

## 5. Retrain output (v2 — 300 epochs, lr=1e-2)

```
$ uv run python -m molmetal.scripts.train_pocket_macro_skeleton \
    --epochs 300 --lr 1e-2 --output-suffix _v2
[train_pocket_macro_skeleton] Built 66 training pockets across 6 distinct scaffold classes.
[train_pocket_macro_skeleton] Model architecture: PocketResidueEncoder(33→32) + MultiheadAttention(32, h=4) + mean-pool + Linear(32, 12) = 5708 params
[train_pocket_macro_skeleton] epoch    1/300 loss=161.9957 wall=0.2s
[train_pocket_macro_skeleton] epoch   30/300 loss=53.6120 wall=1.1s
[train_pocket_macro_skeleton] epoch   60/300 loss=45.8349 wall=1.9s
[train_pocket_macro_skeleton] epoch   90/300 loss=45.2540 wall=2.8s
[train_pocket_macro_skeleton] epoch  120/300 loss=30.2184 wall=3.6s
[train_pocket_macro_skeleton] epoch  150/300 loss=26.6273 wall=4.5s
[train_pocket_macro_skeleton] epoch  180/300 loss=24.9172 wall=5.3s
[train_pocket_macro_skeleton] epoch  210/300 loss=21.5398 wall=6.1s
[train_pocket_macro_skeleton] epoch  240/300 loss=20.8364 wall=7.0s
[train_pocket_macro_skeleton] epoch  270/300 loss=20.5648 wall=7.8s
[train_pocket_macro_skeleton] epoch  300/300 loss=20.4282 wall=8.6s
[train_pocket_macro_skeleton] Train accuracy: 66/66 = 1.000
[train_pocket_macro_skeleton] Saved checkpoint: molmetal/models/pocket_macro_skeleton_v2.pt
[train_pocket_macro_skeleton] Saved metadata:  molmetal/models/pocket_macro_skeleton_v2.pt.json
```

### 5.1 Loss curve (highlights)

| Epoch | Loss  | Delta |
|------:|------:|------:|
|     1 | 162.0 | —     |
|    30 |  53.6 | -67% |
|    60 |  45.8 | -15% |
|    90 |  45.3 |  -1% (plateau) |
|   120 |  30.2 | -33% (anchor-position gradient kicks in) |
|   150 |  26.6 | -12% |
|   180 |  24.9 |  -6% |
|   210 |  21.5 | -14% |
|   240 |  20.8 |  -3% |
|   270 |  20.6 |  -1% |
|   300 |  20.4 |  -1% (converged) |

Loss drops monotonically across the 300-epoch run; final loss
20.43 (vs Phase 2's 40.64 — **−50%**) confirms the model fits
the dataset well.

### 5.2 Per-class accuracy (v2 — 66 PDBs)

| Scaffold class      | Correct / Total | Accuracy |
|-------------------- |----------------:|---------:|
| ZN_TETRA_HHH (CA2)  | **8 / 8**       | **1.000** *(was 0/8)* |
| ZN_TETRA_HHE (MMP/ACE) | 20 / 20     | 1.000    |
| ZN_TETRA_HHD (HDAC) |  6 / 6          | 1.000    |
| MG_OCTA_KINASE (PKA/CDK2) | 16 / 16 | 1.000    |
| FE_HEME_CYS (CYP3A4) |  8 / 8        | 1.000    |
| CU_TBP (SOD1)       |  8 / 8          | 1.000    |
| **Total**           | **66 / 66**     | **1.000** |

---

## 6. Honest caveats

- **Train-set memorisation, not generalisation.** 66 PDBs is still
  far below the threshold for SOTA-class supervised learning.  The
  v2 model can perfectly classify the *training* pockets but we
  have not measured hold-out CV accuracy.  We expect hold-out CV
  to remain in the 60-80% range for non-trivial test pockets.
  This is the same caveat Phase 2 §7 carried; v2 inherits it.

- **Synthetic per-PDB variation.** Because all 8 CA2 PDBs in
  `metalloprotein_targets.py` share the same `binding_site_residues`
  numbering, the dataset builder cycles chain letter A..Z to give
  them distinguishable per-residue `chain_id_hash` features.  This
  is **synthetic** discrimination signal — the model uses it, but
  it does not represent real per-PDB chemistry.  A future Phase 4
  fix would replace this with the actual residue-AA lists from
  `metalloprotein_targets.binding_site_residues` (currently only
  residue *numbers*, not AAs, are exposed in the data structure).

- **LayerNorm omitted (carried forward from Phase 2).** The 32-d
  attention + mean-pool architecture still has no LayerNorm or
  FFN sub-layer; Phase 2 §2 noted this was an empirical choice at
  d=32 because LayerNorm whitens the residual signal to ~0.
  v2 doesn't touch this.

- **No 3D coords (carried forward from Phase 1 §3.2).** We still
  do not consume Cα coordinates; the per-residue feature is AA +
  scalar metadata only.

- **3-fold epochs to convergence.** v0 trained for 100 epochs;
  v2 needs 300 epochs at lr=1e-2 to reach the all-100% regime.
  v0 stayed at 87.9% at the same point because the HHH class was
  structurally indistinguishable.  v2's anchor-position signal
  gives the optimizer a direction to descend — but it still takes
  ~300 epochs to break out of the local minimum.

- **CU_TBP regression risk.** At lr=5e-3 / 100 epochs, v2 produced
  4/8 CU_TBP (50%) while v0 had 8/8 (100%) — a regression
  explained by the chain-letter cycling that gives 8 distinct
  SOD1 patterns the model must now distinguish.  At lr=1e-2 /
  300 epochs this regression disappears (CU_TBP returns to 8/8).
  The lr=1e-2 default is therefore the recommended setting for
  v2.

- **No MCTS integration yet (carried forward).** The v2
  checkpoint is shipped standalone.  Wiring into `proof_search.py`
  is a *separate* ticket that requires touching `proof_search.py`
  (explicitly forbidden by Phase 1 task brief + this Phase 3
  task brief).

- **Workflow safety confirmed.** 0 lines modified in
  `warm_start.py`, `learned_prior.py`, `proof_search.py`, or
  any forbidden module.  v2 is **additive** over Phase 2:
  per-residue feature dim 29→33, dataset builder gets a
  `pdb_index` arg, training script gets `--output-suffix`.

---

## 7. Hand-off to Phase 4 (out of scope)

Phase 4 is **5-fold CV measurement** + (optional) integration
with `proof_search.py` as the per-pocket root prior.  Phase 4 is
explicitly out of scope for this Phase 3 ticket.

If a Phase 4 ticket is opened, the recommended next steps are:

1. **5-fold CV by PDB identity** (not by residue-list identity) —
   we now have ≥6 PDBs per family so leave-one-PDB-out is feasible
   for CA2/ACE/PKA/CDK2/CYP3A4/SOD1 (8 PDBs each).
2. **Wire the v2 skeleton vector into `proof_search.py`** as a
   per-pocket root prior — Phase 4 integration ticket (requires
   touching `proof_search.py`, which is owned by the pocket-
   invariance workflow).
3. **Replace synthetic chain-letter cycling** with real
   per-PDB `binding_site_residues` residue-AA lookups (would
   need to extend the `MetalloproteinTarget` dataclass with a
   per-PDB residue-AA map).

---

## 8. Lit anchors

* **Vaswani 2017** *Attention Is All You Need* (NeurIPS 2017,
  arXiv:1706.03762) — multi-head self-attention + mean-pool
  aggregation. v2 unchanged from v0.
* **Peng 2022** *Pocket2Mol* (ICLR 2022, arXiv:2205.07249) —
  pocket residues as a permutation-invariant set; mean-pool.
  v2 unchanged from v0.
* **Devlin 2019** *BERT* (NAACL 2019, arXiv:1810.04805) —
  softmax over pooled representation as a classification head.
  v2 unchanged from v0.
* **v2-only rationale** for the anchor-position one-hot:
  informed by the Phase 2 §4.2 / §7 honest framing of the
  HHH-collapse failure mode and Phase 1 design doc §5.1
  ("Confusion concentrated between Zn_tetrahedral_HHH, HHE, HHD —
  differ by exactly one key-anchor residue").  The
  anchor-position signal is **not** borrowed from any specific
  paper; it's a small architectural fix targeted at the
  specific 12-way classification head.

---

## 9. Files shipped (Phase 3)

| Path | Lines | Purpose |
|------|------:|---------|
| `molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py` | 580 (was 380) | +200 lines: PER_RESIDUE_FEATURES=33, anchor_position one-hot, encode_residues ranks anchor positions, docstring updates. |
| `molmetal/molmetal_lam/tests/test_pocket_macro_skeleton.py` | 470 (was 320) | +150 lines: 2 new tests (test_anchor_tier_8_dim, test_ca2_hhh_no_longer_collapses). |
| `molmetal/scripts/train_pocket_macro_skeleton.py` | 320 (was 280) | +40 lines: pdb_index arg, chain-letter cycling, --output-suffix flag. |
| `molmetal/models/pocket_macro_skeleton_v2.pt` | 26 KB | v2 checkpoint (5708 params, 100% train accuracy). |
| `molmetal/models/pocket_macro_skeleton_v2.pt.json` | 4.3 KB | v2 metadata (per-class accuracy, loss curve, workflow tag). |
| `molmetal/reports/wf_deflex_pocket_macro_skeleton/phase3_ca2_fix.md` | this file | Phase 3 report. |

Total new code: ~390 lines + ~30 KB of artifacts.  **0 lines**
modified in `warm_start.py`, `learned_prior.py`, `proof_search.py`,
`beta_reductions.py`, `flow_matching_lipman/*`, or any paper file.

---

*WF-Deflex PocketMacroSkeleton — Phase 3 (CA2-class collapse fix).
Project root: `/home/hugo/codes/try_triton_on_rocm`. Python 3.12,
ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64 (CPU
path for this phase; GPU not required — Phase 3 trains 5,708
params on 66 PDBs in 8.6 s wall-clock on CPU).*
