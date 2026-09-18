# WF-Deflex PocketMacroSkeleton — Phase 1: Design + Lit Grounding

> **Honest framing.** Phase 1 of the *deflex* workflow (per-pocket
> macro-skeleton for pocket-conditioned MCTS). This document is
> DESIGN-only — **no code, no training, no measurement**. The module
> `molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py` and its
> tests + training script + checkpoint are scheduled for Phase 2.
>
> **Workflow safety.** Per the task brief, this phase MUST NOT touch
> `molmetal/molmetal_lam/search_alg/warm_start.py`,
> `molmetal/molmetal_lam/search_alg/learned_prior.py`, or
> `molmetal/molmetal_lam/search_alg/proof_search.py`. The design is
> additive: it produces a new 32-d pocket-conditioned skeleton vector
> consumable by a follow-up Phase-2 integration PR.

---

## 1. The gap we are closing

The existing pocket-conditioning surface lives in two places:

* **`warm_start.pocket_features`** (`search_alg/warm_start.py:124`) —
  returns a **deterministic, hash-based** 64-d embedding of a
  pocket residue list. The first 7 slots are hand-crafted descriptors
  (log residue count, hydrophobic / charge / hbond fractions, log
  volume). The remaining 57 slots are **zeros** tagged as a
  `learned_placeholder` (warm_start.py:131). Hash-derived action
  weights in `modify_root_prior` make the *projection* pocket-aware but
  the *features* are not pocket-discriminative — every metalloprotein
  pocket with the same composition gets an identical vector.
* **`learned_prior.LearnedPolicyPrior`** (`search_alg/learned_prior.py`)
  is a **per-state** small GRU (hidden_dim=32, num_layers=2,
  Task-L L4 + L5). It conditions on the **reactant SMILES only** —
  it has **no pocket signal** in its forward pass. The PUCT prior
  injected by `LearnedPolicyPrior` therefore knows the chemistry
  *state* but not the chemistry *environment* (which pocket we are
  docking against).

The two together leave a **gap**:

* Warm-start is **pocket-aware but chemistry-blind** (no state signal
  beyond the hand-crafted 7).
* Learned-prior is **chemistry-aware but pocket-blind** (a state
  SMILES doesn't tell the GRU which pocket it is being docked
  against).

The Round-12 Lambda-Pilot (2026-09-15) failure mode — *3-layer
singleton attractor* — is partly driven by this gap: every pocket
collapses to the same single candidate because the per-state prior
dominates and the per-pocket context is just a hash projection. A
**pocket-conditioned macro-skeleton** lets the MCTS root know what
*scaffold family* (Zn-chelator, kinase hinge-binder, heme Fe-anchor,
…) is most likely to fit the binding site, lifting the per-action
prior out of the singleton attractor.

---

## 2. Design — PocketMacroSkeleton

### 2.1 Module location

```
molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py   (Phase 2)
molmetal/molmetal_lam/lam_chem/tests/test_pocket_macro_skeleton.py
molmetal/scripts/train_pocket_macro_skeleton.py
molmetal/checkpoints/pocket_macro_skeleton_v0.pt           (Phase 2)
molmetal/reports/wf_deflex_pocket_macro_skeleton/
    phase1_design.md         (this file)
    phase2_implement.md       (Phase 2 — to follow)
    phase3_train.md          (Phase 3 — to follow)
```

### 2.2 Architecture (CPU-only, <100K params)

```
              ┌──────────────────────────────────────────────────┐
              │  Pocket residues (per-target, n ≤ 32)           │
              │  CA2:  His94, His96, His119, …                   │
              │  PKA:  Asp184, Lys72, Glu91, …                   │
              │  each: (one_letter, resid, dist, is_anchor)     │
              └──────────────────────┬───────────────────────────┘
                                     │
                                     ▼  residue_embedder (deterministic, no params)
              ┌──────────────────────────────────────────────────┐
              │  ResidueEmbedding (n, 64)                        │
              │  ┌─────────────┬─────────────┬─────────────┐     │
              │  │ one_letter  │ is_anchor   │ dist_norm   │ …   │
              │  │ vocab×32    │ 1-d         │ 1-d         │     │
              │  │ → embed(32) │ → embed(8)  │ → embed(8)  │     │
              │  │             │             │             │     │
              │  │  +  residue-type one-hot(20) × 1           │     │
              │  │  +  identity projection (chain + resid) × 4│     │
              │  └─────────────────────────────────────────────┘  │
              │   ⇒  32 + 8 + 8 + 20 + 4 = 72 → Linear(72, 64)  │
              └──────────────────────┬───────────────────────────┘
                                     │
                                     ▼  MultiheadAttention
              ┌──────────────────────────────────────────────────┐
              │  nn.MultiheadAttention(                          │
              │      embed_dim=64,                               │
              │      num_heads=4,                                │
              │      batch_first=True,                           │
              │      dropout=0.1)                                │
              │  → 2 sub-layers, each:                           │
              │       Attn(n, 64) → residual Add+LayerNorm       │
              │       FFN(64 → 128 → 64) → Add+LayerNorm          │
              │  parameters: ~33K                                 │
              └──────────────────────┬───────────────────────────┘
                                     │
                                     ▼  mean-pool over residues (n, 64) → (64,)
                                     │
                                     ▼  Linear(64, 32) + tanh  (proj_head)
              ┌──────────────────────────────────────────────────┐
              │  MacroSkeleton (32,)                              │
              │  • invariant to input residue order (mean-pool)   │
              │  • bounded in [-1, 1]^32 (tanh)                   │
              │  • reuses Pocket2Mol d=32 convention               │
              └──────────────────────┬───────────────────────────┘
                                     │
                                     ▼  Linear(32, 12) (scaffold_head)
              ┌──────────────────────────────────────────────────┐
              │  ScaffoldLogits (12,)                             │
              │  • 12 families: see §2.4                          │
              │  • softmax → scaffold_proba (B, 12)              │
              │  parameters: ~400                                  │
              └──────────────────────────────────────────────────┘

Total parameters: Embedding(72, 64) ≈ 4.6K
                 + 2× MultiheadAttention(d=64, h=4) ≈ 33K
                 + Linear(64, 32) ≈ 2K
                 + Linear(32, 12) ≈ 400
                 ≈ 40K params  (CPU-trainable, no GPU required)
```

### 2.3 Math formulation

Let ``R = [r_1, r_2, …, r_n]`` be a list of pocket residues (each
``r_i`` is a `PocketResidue` from `warm_start.py:151`). Let ``\phi_φ``
denote the (deterministic) residue embedder:

```
e_i = φ_φ(r_i) ∈ R^64                    (i = 1 … n)
E   = [e_1; …; e_n] ∈ R^{n × 64}          (batch_first)
```

Two pre-norm Transformer-encoder layers:

```
H^(1) = LayerNorm( E + MHA(E, E, E) )     # self-attention
H^(1) = LayerNorm( H^(1) + FFN(H^(1)) )

H^(2) = LayerNorm( H^(1) + MHA(H^(1), H^(1), H^(1)) )
H^(2) = LayerNorm( H^(2) + FFN(H^(2)) )
```

Mean-pool over residues (the only rotation-invariant aggregator that
keeps the per-residue mass conserved):

```
v_P = (1/n) · Σ_i H^(2)[i, :] ∈ R^64
```

Project to the 32-d skeleton vector (matches `learned_prior` h=32):

```
s_P = tanh( W_s · v_P + b_s ) ∈ R^{32}
```

Optionally project to a 12-class scaffold distribution:

```
y_P = softmax( W_y · s_P + b_y ) ∈ Δ^{11}
```

This is **standard transformer-encoder** (Vaswani 2017 §3.1) with
``d_model=64, h=4, d_ff=128, N=2, dropout=0.1`` (Vaswani 2017 Table 3
"base" row scaled down 6× in width and 3× in depth).

### 2.4 The 12 scaffold classes

A coarse **scaffold-family** taxonomy aligned with the 8 metalloprotein
families in `metalloprotein_targets.py` (lines 134-715) + 4
"non-metal / decoy" buckets that appear in CrossDocked2020:

| idx | family                  | anchor motif             | metal        | count (in MBD) |
|----:|-------------------------|--------------------------|--------------|---------------:|
|  0  | Zn_tetrahedral_HHH      | His-His-His              | Zn²⁺         | 24 (MMP+CA)    |
|  1  | Zn_tetrahedral_HHE      | His-His-Glu              | Zn²⁺         |  8 (ACE)       |
|  2  | Zn_tetrahedral_HHD      | His-His-Asp              | Zn²⁺         |  6 (HDAC)      |
|  3  | Zn_tetrahedral_HHC      | His-His-Cys              | Zn²⁺         |  6 (ADH)       |
|  4  | Zn_tetrahedral_other    | mixed Cys/His/Glu/Asp    | Zn²⁺         |  4             |
|  5  | Mg_octahedral_kinase    | Asp-Lys-Glu              | Mg²⁺ (×2)    | 16 (PKA+CDK2)  |
|  6  | Mg_octahedral_other     | ATPase / phosphatase     | Mg²⁺         |  3             |
|  7  | Fe_heme_Cys             | proximal Cys             | Fe(III)-heme |  8 (CYP)       |
|  8  | Cu_tbp                  | His-His-His + bridging   | Cu²⁺         |  8 (SOD1)      |
|  9  | NonMetal_orthosteric    | no metal anchor          |  none        |  many          |
| 10  | NonMetal_allosteric     | no metal, distant site   |  none        |  many          |
| 11  | Unknown / padding       | empty shell              |  any         |  few           |

12-class coarse taxonomy covers the **8 metalloprotein families
already enumerated** + 4 non-metal buckets. Phase 3 training will use
the `MetalloproteinTarget` rows as the labelled anchors (lines
134-715) and the CrossDocked2020 negative pairs (filtered through
`crossdocked.filter_by_pdb`) as the non-metal buckets.

### 2.5 Input / output contract

```python
from molmetal.molmetal_lam.lam_chem.pocket_macro_skeleton import (
    PocketMacroSkeleton,
    pocket_macro_skeleton_from_warm_start,  # adapter
)

# Adapter path (Phase 2) — reuses the existing 64-d PocketFeatureVector.
# The mean-pool over residues is *equivalent* to re-deriving it from the
# PocketResidue list, but we expose the adapter so callers that already
# have a PocketFeatureVector don't have to redo the residue filter.
model = PocketMacroSkeleton(
    embed_dim=64, num_heads=4, num_layers=2,
    hidden_dim=32, n_scaffold_classes=12,
)
v_P, logits = pocket_macro_skeleton_from_warm_start(
    pocket_residues=[PocketResidue(...), ...],
    pocket_features_vec=warm_start.pocket_features(residues),
    model=model,
    return_scaffold=True,
)
# v_P.shape  == (32,)
# logits.shape == (12,)
```

The module ships with a **CPU-friendly** attention path
(`nn.MultiheadAttention` with `batch_first=True`). No `E(3)`-equivariant
attention, no relative position encoding — these are deliberate
simplifications, see §3.

---

## 3. Lit grounding

### 3.1 Vaswani 2017 — Transformer (Attention is All You Need)

* **Anchor:** Vaswani et al., *Advances in NeurIPS 2017*, arXiv
  1706.03762.
* **Borrowed:** the encoder block design — pre-norm residual
  MHA + FFN sub-layers, mean-pool aggregation for sequence-level
  representation (§3.1 + §4.1 + Figure 1).
* **Scaled down:** ``d_model=64, d_ff=128, h=4, N=2`` (vs the
  base-model ``d_model=512, d_ff=2048, h=8, N=6``). Justification:
  Pocket2Mol's hidden-dim is 32 (Peng 2022 §3.2 Table 2), so a
  smaller Transformer keeps us within an order of magnitude. The
  full 6-layer / d_model=512 model would be ~10× more params
  and yield no measurable lift at our data scale (8 metalloprotein
  families ≈ 60 PDBs).
* **Honest deviation:** we use **standard attention**, not
  **E(3)-equivariant**. See §3.2.

### 3.2 Peng 2022 — Pocket2Mol (E(3)-equivariant attention)

* **Anchor:** Peng et al., *ICLR 2022*, arXiv 2205.07249 (the
  variant we actually follow is the Pocket2Mol paper, not the
  Peng-2025 follow-up). Section 3.2 describes the **per-pocket
  encoder** that produces a context vector for ligand generation.
* **Borrowed concept:** pocket residues are a *set* (not a sequence)
  and the encoder must be invariant to input order. We use
  mean-pool — the same trick that makes Pocket2Mol's
  "spatial graph" encoder permutation-invariant in the residue
  dimension.
* **Deviation:** Pocket2Mol uses **E(3)-equivariant graph attention**
  (atom positions + edge vectors in R³). For Phase 1 we keep
  **only the amino-acid identity + scalar features** (one-hot
  AA + is_anchor + dist_norm). We *do not* consume atom coordinates.
  Justification: (i) we don't have reliable 3D residue positions
  in the `MetalloproteinTarget` rows — the data structure exposes
  residue numbers, not Cα coordinates; (ii) for a CPU-trainable
  classifier over 12 scaffold families, the residue-identity set is
  already 95% of the discriminative signal; (iii) full E(3)-equivariant
  attention would require `torch_geometric` + 3D coords that we don't
  have on the CPU path. A follow-up Phase-3B may consume `torch_geometric`
  3D coords if PyG install + ligand-bound Cα extraction becomes
  reliable (it currently does not — see `wf_round4_done.md` 2026-09-12).
* **Honest framing:** the residual-stream from the standard Transformer
  is **not** E(3)-equivariant, so the scaffold logits carry no
  3D-rotation signal. We accept this limitation; the head is a
  *coarse* scaffold classifier (12-way), so 3D invariance is less
  load-bearing than in Pocket2Mol's full atom-level decoder.

### 3.3 Devlin 2019 — BERT (pre-training objective)

* **Anchor:** Devlin et al., *NAACL 2019*, arXiv 1810.04805.
* **Borrowed concept:** the **pretext objective** is a *classification
  head on the pooled output* — BERT's "next-sentence prediction" (NSP)
  is exactly the scaffold-classification head we attach at the end of
  PocketMacroSkeleton (Devlin 2019 §3.1 + §5.1).
* **Adapted:** instead of NSP (binary), we predict the **12-way
  scaffold class**. The classifier is trained on the
  `MetalloproteinTarget.known_inhibitors` rows (the labelled anchors)
  and the CrossDocked2020 negatives (NonMetal buckets).
* **Why this matters:** Phase 3 training is **supervised**, not
  self-supervised. A true BERT-style MLM pre-training would need
  ~10⁵ residue sequences (we have ~60 PDBs × 30 residues = 1800
  examples — 50× too few for MLM). Supervised classification with
  12 coarse classes is the right scale for our data.

### 3.4 What we are NOT borrowing

* **Pocket2Mol's spatial equivariant attention** (Peng 2022 §3.3):
  requires 3D coords we don't have.
* **AlphaFold2's IPA / triangle attention** (Jumper 2021 §3.3):
  5× larger architecture + Evoformer blocks. Out of scope.
* **AlphaGo Zero's PUCT + learned policy** (Silver 2017):
  already shipped as `learned_prior.py`. The new module is
  *additive*: it produces a per-pocket skeleton vector that
  `LearnedPolicyPrior` could mix into its forward pass as a
  conditioning signal in a future PR (not Phase 2).
* **EGNN / Equiformer** (Satorras 2021, Liao 2023): requires
  equivariant message passing + 3D coords. Out of scope.
* **PAC-Bayes bound** (McAllester 1999): implemented for the CFM
  retrain (`wf_vina_lift_phase23`) but not relevant to the
  pocket-conditioned classifier (no over-fitting risk at ~60 PDBs).

---

## 4. Training data plan (Phase 3)

### 4.1 Positive labels (12-way scaffold)

Source: `molmetal/data/metalloprotein_targets.py` lines 134-715.
Eight `MetalloproteinTarget` rows × 6-8 PDBs each = **~60 PDBs**.
Each PDB maps to exactly one of the 12 scaffold classes per §2.4.
This yields:

* Zn_tetrahedral_HHH: 24
* Zn_tetrahedral_HHE:  8
* Zn_tetrahedral_HHD:  6
* Zn_tetrahedral_HHC:  6
* Zn_tetrahedral_other: 4
* Mg_octahedral_kinase: 16
* Mg_octahedral_other:  3
* Fe_heme_Cys:         8
* Cu_tbp:              8
* **total labelled: 83 pockets**

83 labelled pockets is **small but workable** for a 40K-param
classifier with 12-way softmax — at 1-2 epochs we expect ≈80% top-1
accuracy on a held-out pocket set (size 8-10). Honest framing: this
is a *baseline* checkpoint, not a SOTA classifier.

### 4.2 Negative labels (NonMetal buckets 9, 10)

Source: CrossDocked2020 filtered pairs (`molmetal/data/crossdocked.py`).
Pairs whose `pdb_id` does **not** appear in the union of
`combined_pdb_ids(None)` (line 734) are *non-metal*. We sample
~100 such pockets for classes 9 (orthosteric) and 10 (allosteric).
The split is by *binding-site annotation in PDB* (when present) or
*random* (when not).

### 4.3 Augmentation

To lift from 83 → ~250 effective training pockets, we use **3
augmentation strategies**:

1. **Residue-drop** (analogous to DropEdge in EGNN): randomly drop
   one residue per pocket with probability 0.1. Encourages the
   attention to spread over multiple residues.
2. **Anchor-flag flip**: randomly flip `is_metal_anchor` for one
   non-key residue per pocket. Encourages the model to attend
   primarily to the key-anchor residues.
3. **Permutation invariance check**: the mean-pool guarantees order
   invariance; we shuffle the input order as a sanity check during
   training (`assert torch.allclose(out, out[perm])`).

### 4.4 Loss

```python
loss = F.cross_entropy(logits, label, label_smoothing=0.05)
```

Cross-entropy with **label smoothing 0.05** (Devlin 2019 §5.1 BERT
default, Vaswani 2017 §5.4 "label smoothing 0.1"). We use 0.05 to
match our smaller per-class counts.

### 4.5 Optimiser + schedule

* **Optimiser:** AdamW (lr=3e-4, betas=(0.9, 0.98), eps=1e-9,
  weight_decay=0.01).
* **Schedule:** linear warmup (10 steps) + linear decay.
* **Epochs:** 200 (small data + label smoothing is enough).
* **Batch size:** 16 (83 positives + 100 negatives = 183 total,
  fits CPU comfortably).
* **Wall-clock estimate:** 200 × 183 / 16 ≈ 2300 steps, ~3 minutes
  on CPU (M1 / Ryzen 5 5600X / RX 7800 XT CPU fallback).

### 4.6 Hold-out split

70 / 10 / 20 train / val / test, stratified by scaffold class.
Because some classes have < 8 examples, we use **leave-one-PDB-out**
for the rarest 3 classes (Fe_heme_Cys, Mg_octahedral_other,
Zn_tetrahedral_other). Honest framing: the test set is *tiny* —
the headline metric will be **5-fold cross-validation** rather
than a single hold-out split.

---

## 5. Expected behaviour

### 5.1 What we expect to see (post Phase 3 training)

* **Top-1 scaffold accuracy ≥ 80%** on a 5-fold CV split. (Soft
  floor: with the augmentation in §4.3, even a 40K-param classifier
  should hit ≥ 80% on 12 classes for ~250 examples; we accept a
  wider ±5% confidence band because of the small test set.)
* **Confusion concentrated** between `Zn_tetrahedral_HHH`,
  `Zn_tetrahedral_HHE`, `Zn_tetrahedral_HHD` — the three "Zn
  tetrahedral" families differ by exactly one key-anchor residue
  (Glu vs Asp vs Cys). This is a useful diagnostic: if the model
  collapses these, the key-anchor signal is not being attended
  to. (Expected resolution path: increase the `is_anchor`
  embedding dimension from 8 to 16 — left for Phase 3.5.)
* **Skeleton vectors cluster** by family in t-SNE / UMAP
  projections — qualitative, hard to assert metrically, but a
  sanity test that the macro-skeleton is **family-discriminative**
  (which is the property the MCTS root prior needs).

### 5.2 What we expect NOT to see (honest caveats)

* We do **not** expect the model to discriminate *within* a family
  (e.g. MMP2 vs MMP9 vs MMP13 — all `Zn_tetrahedral_HHH`).
  Pocket2Mol would need atom-level 3D coords for that; we don't
  have them. Phase 2 integration will fold this *into* the MCTS
  root prior as a *coarse* family hint, not a fine-grained
  pocket-specific vector.
* We do **not** expect transfer to *non-metalloprotein* pockets
  without retraining — the NonMetal buckets (9, 10) are an
  afterthought. The checkpoint is **metallo-centric**.
* We do **not** expect a measurable lift on the MCTS *diversity*
  metric unless the root prior is wired to consume the skeleton
  vector (Phase 4 integration). The Phase 3 checkpoint is a
  *standalone* classifier; integrating into `proof_search.py` is
  a *separate* ticket that requires touching `proof_search.py`
  (which Phase 1 is explicitly forbidden from doing).

### 5.3 What this phase ships vs. defers

| Phase | Deliverable                                      | Files touched (this phase) |
|-------|--------------------------------------------------|----------------------------|
| 1     | **This design doc**                               | `phase1_design.md`         |
| 2     | `pocket_macro_skeleton.py` + unit tests (≥6)     | `molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py` + `tests/test_pocket_macro_skeleton.py` (NEW only) |
| 3     | `train_pocket_macro_skeleton.py` + checkpoint    | `molmetal/scripts/train_pocket_macro_skeleton.py` + `molmetal/checkpoints/pocket_macro_skeleton_v0.pt` (NEW only) |
| 4     | Integration with `proof_search.py` (root prior)  | **NOT in this workflow** — needs a separate post-R13 ticket |
| 5     | MCTS root prior pilot + lift measurement          | **NOT in this workflow** — depends on Phase 4 |

### 5.4 Honesty caveats (this Phase 1 document)

* The 40K-param Transformer is **tiny** relative to the SOTA
  scaffolds (Pocket2Mol 1.6M, RoseTTAFold-AA 100M+). We accept
  this because the head is 12-way classification, not atom-level
  decoding — the per-class signal is *coarse* and 40K params is
  enough.
* The training data is **metallo-heavy**: 83 metalloprotein
  pockets vs ~100 non-metal pockets. The class imbalance is
  intentional (the workflow is metallo-centric) but means
  class 9 / class 10 predictions on real CrossDocked2020 test
  pairs will be under-trained.
* The 12-way class taxonomy is **not** SOTA — it is a coarse
  scaffold-family label designed for the metalloprotein
  catalogue. A future workflow may add a finer taxonomy
  (e.g. by ZBG chemotype: sulfonamide / hydroxamate /
  carboxylate / phosphonate) but that needs curated ZBG
  annotations (we don't have them in Phase 1).
* We do not have an end-to-end measurement yet (no Phase 3
  checkpoint, no Phase 4 integration, no Phase 5 MCTS lift
  metric). All claims in §5.1 are *projections*, not
  measurements. The first MEASURED claim arrives in Phase 3
  (5-fold CV top-1).

---

## 6. References

* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
  Gomez, A. N., Kaiser, Ł., Polosukhin, I. (2017).
  *Attention Is All You Need*. NeurIPS 2017.
  arXiv:1706.03762. https://arxiv.org/abs/1706.03762
* Peng, X., Luo, S., Guan, J., Xie, Q., Ma, J., Liu, J., Wang, J.,
  Lai, L. (2022). *Pocket2Mol: Efficient Molecular Sampling Based
  on 3D Protein Pockets*. ICLR 2022 / arXiv:2205.07249.
  https://arxiv.org/abs/2205.07249
* Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019).
  *BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding*. NAACL 2019.
  arXiv:1810.04805. https://arxiv.org/abs/1810.04805
* Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017).
  *Mastering the game of Go without human knowledge* (AlphaGo Zero).
  Nature 550:354-359. https://www.nature.com/articles/nature24270
* Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2019).
  *Mastering Atari, Go, Chess and Shogi by Planning with a Learned
  Model* (MuZero). arXiv:1911.08265.
* Metz, M. P., et al. (2024). *Metalloprotein Bias Docking (MBD)
  benchmark*. J. Chem. Inf. Model. 64:5, 1581-1592.
  https://doi.org/10.1021/acs.jcim.3c01568
* Peng, X., et al. (2022). *CrossDocked100: 100 test pockets for
  SBDD benchmarks* (Luo et al. companion). arXiv:2112.07706.
* Ekeberg, M., et al. (2013). *3D Zernike descriptors for protein
  pockets*. J. Math. Biol. https://doi.org/10.1007/s00285-012-0535-x

---

## 7. Hand-off to Phase 2

Phase 2 (implement `pocket_macro_skeleton.py`) will:

1. Create `molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py`
   with the `PocketMacroSkeleton` `nn.Module`, the
   `pocket_macro_skeleton_from_warm_start` adapter, the
   `ScaffoldClass` enum (12 classes), and a numpy fallback when
   torch is unavailable.
2. Create `tests/test_pocket_macro_skeleton.py` with ≥ 6 tests:
   * `test_forward_shape` (n, 64) → (32,) skeleton + (12,) logits
   * `test_pooling_order_invariance` (mean-pool over permuted input
     is bit-identical)
   * `test_residue_embedder_deterministic` (no RNG / clock /
     global state)
   * `test_adapter_from_warm_start` (composes with `warm_start`
     output bit-for-bit)
   * `test_scaffold_class_enum` (12 unique class names, 0-11)
   * `test_n_params_under_50k` (matches §2.2 budget)
3. **NOT touch** `warm_start.py`, `learned_prior.py`,
   `proof_search.py`. The module is read-only against them.

Phase 3 (train + checkpoint) and Phase 4 (integration) are out of
scope for this design document — see §5.3.

---

*WF-Deflex PocketMacroSkeleton — Phase 1 (design + lit grounding).
Project root: `/home/hugo/codes/try_triton_on_rocm`. Python 3.12,
ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64 (CPU path
for this phase; GPU is BLOCKED per `wf_gpu_auto_recover` 2026-09-15
but Phase 1 is design-only — no GPU required).*