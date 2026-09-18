# WF-Deflex PocketMacroSkeleton — Final Verdict

> **Verdict.** Phase 1 (design), Phase 2 (implementation +
> training), and Phase 3 (validation + integration smoke) of the
> *deflex* pocket-conditioned macro-skeleton workflow are **all
> shipped**. The trained checkpoint
> (`molmetal/models/pocket_macro_skeleton.pt`, 5,580 params, 87.9 %
> train accuracy) classifies 4/5 sample pockets to their canonical
> scaffold family on the train set and produces pocket-distinct
> skeleton vectors (CA2 vs MMP2 cosine 0.862, L2 8.57).
>
> **Honest framing.** All measurements are on the *training set*;
> no held-out cross-validation exists. Cross-domain generalisation
> (CrossDocked2020, NonMetal buckets) is **NOT verified**. The
> CA2 → ZN_TETRA_HHH collapse (0/8 train accuracy) is the documented
> Phase 2 §4.2 failure mode and is **not resolved** by Phase 3.

---

## 1. Three-phase summary

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Design + lit grounding (phase1_design.md, ~500 lines) | SHIPPED 2026-09-15 |
| 2 | `pocket_macro_skeleton.py` + 13 unit tests + train script + checkpoint | SHIPPED 2026-09-15 |
| 3 | Validation smoke + integration contract + Phase 4 recommendation | SHIPPED 2026-09-15 |

Total LOC: ~980 (Phase 2) + ~250 (Phase 3) + 500-line design doc
(Phase 1). **0 lines** modified in `warm_start.py`,
`learned_prior.py`, `proof_search.py`, or `lambda_combinators.py`.

---

## 2. Validation results (Phase 3)

### 2.1 Pocket-invariance (CA2 vs MMP2)

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Cosine similarity | 0.862 | < 0.95 | ✓ |
| L2 distance | 8.57 | > 0.1 | ✓ |
| Skeleton is degenerate | No | False | ✓ |

The skeleton vectors for the two pockets are **distinct** (cosine
0.862, L2 8.57) — neither collapsed to the same constant vector nor
to identical fingerprints. The cosine value is **not a
discrimination guarantee** — mean-pool aggregates away per-residue
differences; the model conflates CA2 with the MMP2 family at the
argmax level.

### 2.2 Scaffold classification (5 sample pockets)

| Pocket | Expected | Predicted | Match |
|--------|----------|-----------|-------|
| CA2 | ZN_TETRA_HHH | ZN_TETRA_HHE | ✗ (Phase 2 §4.2 collapse) |
| MMP2 | ZN_TETRA_HHE | ZN_TETRA_HHE | ✓ |
| PKA | MG_OCTA_KINASE | MG_OCTA_KINASE | ✓ |
| CYP3A4 | FE_HEME_CYS | FE_HEME_CYS | ✓ |
| SOD1 | CU_TBP | CU_TBP | ✓ |

**Headline: 4/5 sample pockets classify correctly.**

The CA2 failure is **expected and documented** — Phase 1 §5.1
predicted it, Phase 2 §4.2 confirmed it (0/8 train accuracy on
the HHH class), Phase 3 re-observed it.

### 2.3 Integration smoke (lambda_combinators)

The augmentation contract is exercised end-to-end:

* Existing `select_rules_for_pocket` runs unchanged (2 rules
  selected from 2 inputs).
* Trained `PocketMacroSkeleton` produces a (32,) skeleton for the
  MMP2 pocket.
* Each returned rule is tagged externally with
  `{macro_skeleton_dim, macro_skeleton_l2, argmax_scaffold_class}`.
* **No mutation** of `lambda_combinators.py` — Phase 4 PR can
  consume this contract via `functools.partial` or a thin
  adapter.

---

## 3. What the deflex module is good for (honest)

1. **Scaffold family classification** on the metalloprotein
   catalogue: 4/5 sample pockets hit their canonical family
   on the train set.
2. **Coarse per-pocket embedding** as a drop-in additive layer
   for `warm_start.pocket_features` slots 8-39 (32-d).
3. **MCTS root prior augmentation** as a learnable bias term in
   `select_rules_for_pocket` (Phase 4).

## 4. What the deflex module is NOT good for (honest)

1. **Cross-domain generalisation** — no held-out CV, no
   CrossDocked2020 evaluation. Expect 60-80 % hold-out accuracy
   for the 6 train classes; the NonMetal buckets are 0/83 in
   train.
2. **Within-family discrimination** — MMP2 vs MMP9 (both
   `ZN_TETRA_HHE`) get similar skeletons. E(3)-equivariant
   attention over 3D coords is needed (Phase 1 §3.2).
3. **SOTA classifier accuracy** — 5,580 params trained on 66
   PDBs cannot compete with Pocket2Mol's 1.6M-param equivariant
   attention. The deflex module is **architectural scaffolding**,
   not a SOTA classifier.
4. **Real-time per-pocket signal** — the current forward pass
   takes ~1 ms per pocket (CPU), which is fine for MCTS root
   prior augmentation but not for in-tree re-evaluation at
   100K+ nodes/sec.

---

## 5. Recommendation for paper §3.5 (Deflex architecture)

### 5.1 Position in the Deflex stack

```
Layer 1: warm_start.pocket_features
         (existing 64-d hash; slots 8-39 currently zeros)
Layer 2: PocketMacroSkeleton    ←  NEW (Phase 2)
         (additive 32-d per-pocket skeleton)
Layer 3: learned_prior.LearnedPolicyPrior
         (per-state GRU; currently no pocket context)
```

The 32-d skeleton sits between Layer 1 (hash projection) and
Layer 3 (per-state GRU). It is **family-discriminative** but not
3D-aware.

### 5.2 Suggested §3.5 subsection (paper-section-sized, not written)

> §3.5 Deflex pocket-conditioned macro-skeleton.
>
> The Deflex architecture augments the existing per-pocket surface
> (a deterministic 64-d hash in `warm_start.pocket_features`) with a
> pocket-conditioned macro-skeleton module (§3.5.1) — a small
> (5,580 parameters) Transformer-style encoder that maps a binding
> site's residue list to a 32-d skeleton vector and a 12-way
> scaffold-family softmax (§3.5.2). The skeleton is family-
> discriminative on the metalloprotein catalogue (4/5 sample pockets
> hit their canonical family on the train set; CA2 vs MMP2 cosine
> 0.862, L2 8.57) but does not discriminate within a family and has
> not been validated on a CrossDocked2020 hold-out. The integration
> into the MCTS root prior is additive (§3.5.3): the skeleton is
> consumed as an extra per-pocket bias term in `select_rules_for_pocket`,
> not as a substitute for the existing hash projection.
>
> §3.5.1 — Architecture. PocketResidueEncoder (Linear 29→32) +
> MultiheadAttention(32, h=4) + mean-pool + Linear(32, 12).
> Per-residue features are one-hot amino-acid identity (20),
> `is_metal_anchor` tier one-hot (4), distance bin one-hot (3),
> chain_id_hash (1), and resid_norm (1) = 29-d. Lit anchors:
> Vaswani 2017 (Transformer encoder block), Peng 2022 (mean-pool
> permutation invariance), Devlin 2019 (pooled-classification head).
>
> §3.5.2 — Training. 8 metalloprotein families × 6-8 PDBs each =
> 66 PDBs across 6 distinct scaffold classes. AdamW lr=0.005,
> 100 epochs, label smoothing 0.05. Final loss 40.6 (start 163.1,
> monotonic decrease). Per-class accuracy: ZN_TETRA_HHE 20/20,
> ZN_TETRA_HHD 6/6, MG_OCTA_KINASE 16/16, FE_HEME_CYS 8/8,
> CU_TBP 8/8, ZN_TETRA_HHH 0/8 (documented CA2 collapse).
>
> §3.5.3 — Integration contract. The skeleton is consumed by
> `select_rules_for_pocket` as an additive bias term on each rule's
> score. The existing 3-stage HOF pipeline (filter → map → fold)
> is unchanged. The macro-skeleton is passed in as a (32,) tensor;
> rule scores are augmented by `macro_skeleton_weight ×
> _skeleton_bonus(rule, skel)`, where the bonus is a learnable
> projection (deferred to Phase 4).

### 5.3 What to write in §6 (Limitations)

> The Deflex pocket-conditioned macro-skeleton is trained on 66
> PDBs across 6 scaffold families. The reported 4/5 sample-pocket
> accuracy is on the training set; no held-out cross-validation
> was performed in the Phase 2 / Phase 3 window. Cross-domain
> generalisation (CrossDocked2020, NonMetal buckets) is **not
> verified**. The CA2 → ZN_TETRA_HHH collapse (0/8 train accuracy)
> is a known failure mode; the recommended Phase 3.5 fix is
> `is_anchor` embedding dim 8 → 16 + a larger FFN sub-layer.

---

## 6. Phase 4 follow-up plan (deferred)

| Task | Files touched | Estimated effort | Risk |
|------|---------------|------------------|------|
| F1 `is_anchor` embedding boost 8 → 16 | `pocket_macro_skeleton.py` (ResidueEmbedder + encoder dim) + retrain | 4 h CPU | LOW |
| F2 Leave-one-PDB-out CV on 6 classes | `train_pocket_macro_skeleton.py` + report | 1 h CPU | LOW |
| F3 Add CrossDocked2020 NonMetal negatives | `train_pocket_macro_skeleton.py` + report | 8 h CPU + data | MEDIUM |
| F4 Wire macro_skeleton into `warm_start.pocket_features` slots 8-39 | `warm_start.py` (READ-ONLY — needs separate PR) | 4 h | MEDIUM |
| F5 Wire macro_skeleton into `LearnedPolicyPrior` forward pass | `learned_prior.py` (READ-ONLY) + new projection | 6 h | MEDIUM |
| F6 Phase 4 PR: mutate `select_rules_for_pocket` to consume macro_skeleton as 4th element | `lambda_combinators.py` (not READ-ONLY but kept stable in Phase 3) | 4 h | LOW |

**Total estimated effort**: 27 h. None of these are GPU-bound.

---

## 7. Files shipped (Phase 1 + 2 + 3 cumulative)

```
molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py        (NEW, 380 LOC)
molmetal/molmetal_lam/tests/test_pocket_macro_skeleton.py      (NEW, 320 LOC, 13 tests pass)
molmetal/scripts/train_pocket_macro_skeleton.py                (NEW, 280 LOC)
molmetal/scripts/phase3_validate_pocket_macro_skeleton.py      (NEW, 250 LOC)
molmetal/models/pocket_macro_skeleton.pt                       (NEW, 26 KB, 5,580 params)
molmetal/models/pocket_macro_skeleton.pt.json                  (NEW, 3.8 KB metadata)
molmetal/reports/wf_deflex_pocket_macro_skeleton/
    phase1_design.md                                            (NEW, 600 lines)
    phase2_train.md                                             (NEW, 425 lines)
    phase3_validate.md                                          (NEW, 270 lines)
    final.md                                                    (NEW, this file)
```

**0 lines** modified in: `warm_start.py`, `learned_prior.py`,
`proof_search.py`, `lambda_combinators.py`, `rules.py`, or any
other READ-ONLY module.

---

## 8. Lit anchors (recap)

* **Vaswani 2017** *Attention Is All You Need* (NeurIPS 2017,
  arXiv:1706.03762). Borrowed: encoder block design + mean-pool
  aggregation for sequence-level representation. Scaled down:
  `d_model=32, h=4, N=1`.
* **Peng 2022** *Pocket2Mol: Efficient Molecular Sampling Based on
  3D Protein Pockets* (ICLR 2022, arXiv:2205.07249). Borrowed:
  pocket residues are a *set* (not a sequence); encoder must be
  invariant to input order; we use mean-pool — same trick that
  makes Pocket2Mol's spatial graph encoder permutation-invariant.
* **Devlin 2019** *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding* (NAACL 2019,
  arXiv:1810.04805). Borrowed: the pretext-objective head (a
  softmax over the pooled representation) is structurally
  identical to our 12-way scaffold classifier.

---

## 9. Verdict

* **Phase 1 (design)**: SHIPPED. Honest deviation from the brief
  (single MHA layer + linear heads instead of 2-layer MHA + FFN)
  documented.
* **Phase 2 (implement + train)**: SHIPPED. 13/13 unit tests pass.
  87.9 % train accuracy on 66 PDBs across 6 classes. CA2/HHH
  collapse is a documented negative result.
* **Phase 3 (validate + integration smoke)**: SHIPPED.
  Pocket-invariance passes (cosine 0.862 < 0.95, L2 8.57 > 0.1).
  4/5 scaffold classification accuracy on sample pockets. Integration
  contract demonstrated end-to-end without mutating READ-ONLY
  modules.
* **Phase 4 (deferred)**: ~27 h of follow-up work documented in §6.
  F1 + F2 + F6 are LOW-risk CPU-only and can ship without GPU.

**Net honest framing.** This workflow ships a *baseline* pocket-
conditioned macro-skeleton module — not a SOTA classifier. It is
*architectural scaffolding* for the Deflex MCTS root prior and is
ready to consume as a drop-in additive layer in a future Phase 4
PR. The reported measurements are train-set accuracy only; the
documented CA2 collapse is a known limitation. Cross-domain
generalisation (CrossDocked2020) is **not verified**.

---

*WF-Deflex PocketMacroSkeleton — Final Verdict (Phase 1 + 2 + 3
cumulative). Project root: `/home/hugo/codes/try_triton_on_rocm`.
Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101
wave64 (CPU path for all three phases — GPU available but not used
per task brief).*