# WF-1 A1 — Learned Bond-Order Head

| field | value |
|---|---|
| date | 2026-09-14 |
| round | 10 (decoder audit follow-up) |
| scope | `molmetal/models/bond_head.py` (NEW), `molmetal/molmetal_lam/tests/test_bond_head.py` (NEW) |
| runtime | uv-managed Python 3.12; PyTorch ≤ 2.14; CPU seed 0 |
| author | WF-1 (round-10 decoder audit) |
| status | SHIPPED — 16 / 16 tests pass |

---

## 1. Spec

A learned bond-order head that maps per-pair atomic features to a
5-class bond-order distribution and assembles the resulting bond
graph into an `rdkit.Chem.Mol`.  It replaces the SMARTS-only decoder
path in the round-9 `molmetal_lam.lam_chem` decoder (which fails on
~20 % of SBDD-relevant motifs: dative Pd-N, mixed Fe clusters,
strained bicyclics).

### 1.1 Inputs

Per pair `(i, j)`:

| name | type | meaning |
|------|------|---------|
| `distance` | float (Å) | Euclidean distance between atoms i, j |
| `z_i` | int | atomic number of atom i |
| `z_j` | int | atomic number of atom j |
| `angle_to_metal` | float (rad) | deviation from ideal donor–metal angle registered in `molmetal_lam.priors.metal_geometry.DEFAULT_METAL_GEOMETRY`; 0.0 when no metal centre present |
| `is_dative_candidate` | bool | pair has (donor, metal) Z roles at < 2.5 Å |

### 1.2 Outputs

Logits over 5 classes:

| idx | label |
|-----|-------|
| 0 | no bond |
| 1 | single (covalent) — also used for dative |
| 2 | double (covalent) |
| 3 | triple (covalent) |
| 4 | aromatic (covalent) |

Dative bonds reuse class 1 (`single`) with a post-hoc sidecar
(`DecodedMol.dative_bonds`) so downstream consumers can render the
dative character differently.

### 1.3 Architecture

`BondOrderHead` — 2-hidden-layer ReLU MLP, 64 units per layer,
0.10 dropout.  Input dimension is 9 (distance, Z-bucket(i),
Z-bucket(j), angle, dative flag, Z_i raw, Z_j raw, bucket-product,
distance-gap-to-single-bond-anchor).  ~120 LOC including docstrings.

`BondAwareDecoder` — wraps `BondOrderHead` with a candidate-pair
selector (distance ∈ [1.0, 2.4] Å by default), an RDKit assembler
(`Chem.RWMol` → `AddAtom` / `AddBond` → `Chem.SanitizeMol`), and a
graceful fallback that catches every exception and returns
`DecodedMol(error=...)` rather than raising.

### 1.4 Training

Closed-form supervised pass on synthetic tmQM-flavoured patterns
(amide, imine, azide, alkyne, nitrile, ether, amine, C=O, C=C,
c1ccccc1, dative N→Pt/Cu).  Implemented in
`make_synthetic_training_set(seed)` + `train_synthetic(seed, epochs)`.

The head is intentionally SE(3)-unaware — distances and angle
deviations are pre-computed — so it composes cleanly with the EGNN
backbone without breaking equivariance.

---

## 2. MEASURED test results

`uv run pytest -v molmetal/molmetal_lam/tests/test_bond_head.py`

```
molmetal/molmetal_lam/tests/test_bond_head.py::test_head_forward_shape PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_training_script_runs PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_synthetic_dataset_size_and_distribution PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_recovers_cuaac_triazole_pattern PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_recovers_amide_pattern PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_rejects_impossible_valence PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_single_pair_single_bond PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_multi_candidate_best_ranked PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_aromatic_ring_closure PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_dative_bond_to_metal PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_dative_bond_short_distance_bias PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_graceful_fallback_empty_cloud PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_graceful_fallback_invalid_atomic_number PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_graceful_fallback_misaligned_tensors PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_pair_feature_construction PASSED
molmetal/molmetal_lam/tests/test_bond_head.py::test_decoder_accepts_explicit_pair_features PASSED
======================== 16 passed, 1 warning in 2.49s ========================
```

| metric | value |
|--------|-------|
| n_tests | 16 |
| n_passed | 16 |
| synthetic train_acc | **96.03 %** (MEASURED, 600 epochs, seed 0) |
| synthetic val_acc | **95.19 %** (MEASURED, 832 train / 208 val pairs) |
| final loss | 0.120 (MEASURED) |

The head reaches 95 %+ top-1 on the held-out synthetic split.  The
class-weighted cross-entropy + cosine LR schedule keeps the rare
classes (triple, aromatic, no-bond) above chance.

---

## 3. Demo — MEASURED on three canonical motifs

Decoder = `BondAwareDecoder(BondOrderHead(... default-trained ...))`.
Atoms supplied as `(N, 3)` coords + atomic numbers; no explicit
bond list given to the decoder.

### 3.1 CuAAC alkyne end

Input: 2 carbons at 1.20 Å (alkyne C≡C anchor).

```
bond_orders: [(0, 1, BOND_TRIPLE)]
SMILES:     'C#C'
sanitized:  True
```

### 3.2 Amide `C-C(=O)-N`

Input: 4 atoms (C-C-O-N) at tmQM empirical distances (1.52, 1.21, 1.47 Å).

```
bond_orders: [(0, 1, BOND_SINGLE), (1, 2, BOND_DOUBLE), (1, 3, BOND_SINGLE)]
SMILES:     'CC(N)=O'
sanitized:  True
```

(The SMILES is RDKit's canonical rotation `CC(N)=O`, equivalent to
`C(=O)N` — both forms are accepted in the test.)

### 3.3 Benzene ring

Input: 6 carbons at hexagonal vertices, C-C distance = 1.40 Å.

```
bond_orders: 6 aromatic bonds (0,5), (0,1), (1,2), (2,3), (3,4), (4,5)
SMILES:     'c1ccccc1'
sanitized:  True
```

### 3.4 Dative `N → Pt` (bonus)

Input: N at (0,0,0), Pt at (2.0, 0, 0).  Decoder auto-flags the
dative bond via the geometry prior (`DEFAULT_METAL_GEOMETRY[78] ==
square_planar` + N is a known donor).

```
bond_orders:  [(0, 1, BOND_SINGLE)]
dative_bonds: [(0, 1)]          # donor=N (idx 0), metal=Pt (idx 1)
SMILES:      '[NH2][Pt]'
sanitized:   True
```

---

## 4. PROJECTED follow-ups

These are NOT yet validated — they are listed so the WF-2 / WF-3
work can hook into them without re-reading the code.

1. **Train on tmQM-21k** — swap `make_synthetic_training_set` for a
   real-data loader.  PROJECTED 92-94 % top-1 on a held-out tmQM test
   split, given the synthetic baseline is already 96 %.
2. **Add SE(3)-aware features** — the head currently consumes
   pre-computed distances / angles.  An equivariant variant would
   accept EGNN edge embeddings directly and could be > 2 % better
   on noisy coordinates.
4. **Bond aromaticity perception** — the head does not yet model
   6π / 4n+2 rule.  Aromatic bonds are recovered as `class 4` only
   because tmQM averages say "C-C at 1.40 Å is usually aromatic".
   For polycyclic / heterocyclic systems a downstream RDKit
   `Chem.SetAromaticity` cleanup is needed.
5. **Dative bond polarity** — current model returns the donor as the
   first atom in the tuple.  If ligand-resolved electron bookkeeping
   is required, replace `_infer_pair_features` with a polarity-aware
   classifier.

---

## 5. Honest framing

* MEASURED: `n_tests=16`, `n_passed=16`, `train_acc=96.03 %`,
  `val_acc=95.19 %`, `loss_final=0.120`, four demo SMILES on CPU
  (`uv run python` invocation logged in this conversation).
* PROJECTED: tmQM-21k top-1 ≥ 92 %, equivariant variant > 2 %
  improvement, aromaticity perception extension.  None of these
  have been measured yet.
* Out of scope: training on the real tmQM corpus, integration into
  the flow-matching decoder loop, batched CUDA inference.  These
  are WF-2 / WF-3 follow-ups.

---

## 6. Files

* NEW — `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py`
* NEW — `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_bond_head.py`
* Reused — `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py` (`DEFAULT_METAL_GEOMETRY`)
* Reused — `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/egnn_rocm.py` (EGNN backbone for future WF-2 wiring)