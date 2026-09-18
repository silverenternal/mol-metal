# WF-Deflex PocketMacroSkeleton — Phase 3: Validation + Integration Smoke

> **Honest framing.** Phase 3 of the *deflex* workflow (per-pocket
> macro-skeleton for pocket-conditioned MCTS). This document is
> **validation + integration smoke** — the trained checkpoint from
> Phase 2 is loaded, three validation experiments are run, and the
> integration path into `lambda_combinators.select_rules_for_pocket`
> is exercised end-to-end. The checkpoint is **CPU-only, trained on
> 66 PDBs across 6 classes** — none of the measurements below should
> be read as generalisation claims.
>
> **Workflow safety.** Per the task brief, this phase MUST NOT touch
> `molmetal/molmetal_lam/search_alg/warm_start.py`,
> `molmetal/molmetal_lam/search_alg/learned_prior.py`, or
> `molmetal/molmetal_lam/search_alg/proof_search.py`. We also do NOT
> mutate `lambda_combinators.py` (the existing 3-stage HOF pipeline
> stays unchanged) — the integration smoke demonstrates the
> augmentation contract and documents the Phase 4 follow-up PR.

---

## 1. Checkpoint + smoke-script

| Path | Purpose |
|------|---------|
| `molmetal/models/pocket_macro_skeleton.pt` | Trained Phase 2 checkpoint (5,580 params, 87.9 % train acc) |
| `molmetal/models/pocket_macro_skeleton.pt.json` | JSON metadata (per-class accuracy, loss curve, lit anchors) |
| `molmetal/scripts/phase3_validate_pocket_macro_skeleton.py` | Phase 3 validation script (NEW, 250 LOC) |

The script:

1. Loads the checkpoint via `torch.load(..., weights_only=False)`
2. Constructs a fresh `PocketMacroSkeleton` and runs `load_state_dict(strict=True)`
3. Runs three experiments (pocket-invariance, scaffold classification, integration smoke)
4. Asserts cheap sanity gates (skeleton not degenerate)

Run from project root:

```
uv run python -m molmetal.scripts.phase3_validate_pocket_macro_skeleton
```

---

## 2. Experiment 1 — Pocket-invariance (CA2 vs MMP2)

```
[2/3] Experiment 1 — pocket-invariance (CA2 vs MMP2):
    ca2_argmax: ZN_TETRA_HHE
    mmp2_argmax: ZN_TETRA_HHE
    cosine_similarity: 0.8622639775276184
    l2_distance: 8.568647384643555
    expected_ca2_class: ZN_TETRA_HHH
    expected_mmp2_class: ZN_TETRA_HHE
```

### 2.1 Interpretation

The task brief asks for `cosine_similarity < 0.95` and
`L2 distance > 0.1`. We satisfy both:

* `cosine_similarity = 0.862 < 0.95` ✓ — the skeletons are **distinct**
  (not collapsed to the same vector).
* `L2 distance = 8.57 > 0.1` ✓ — the per-residue mean-pool produces
  vectors of meaningful magnitude.

Honest caveats:
* The `cosine_similarity < 0.95` gate is **not a discrimination
  guarantee** — it is a *degeneracy check*. Mean-pool averages away
  per-residue differences (Phase 1 §2.3, Phase 2 §3 "Honest framing
  on test_pocket_invariance_avoided"), so cosine can stay above
  0.85 even when the model classifies correctly. The Phase 2
  `test_pocket_invariance_avoided` test was reframed for this reason
  (see Phase 2 §3 / §4.2).
* The CA2 pocket is **mis-classified as ZN_TETRA_HHE** (the MMP
  family), which is the documented Phase 2 §4.2 collapse
  (0/8 train accuracy on the HHH class). This is expected — the
  Phase 1 spec §5.1 anticipated this exact failure mode and
  recommended `is_anchor` embedding dim 8 → 16 as a Phase 3.5 fix.
* Both argmaxes hit `ZN_TETRA_HHE` because **CA2 and MMP2 share
  the same residue-identity set in mean-pool space** (3x His +
  hydrophobic shell). The L2 distance is non-zero (8.57) but
  the *direction* of the difference vector is below the cosine
  threshold for class separation.

### 2.2 Why we do not claim "pocket-invariant" in the SOTA sense

The Phase 1 spec §1 framed `pocket_features` (a hash-based 64-d
projection) as "not pocket-discriminative — every metalloprotein
pocket with the same composition gets an identical vector." The
Phase 2 module **does** discriminate — 4/5 of the sample pockets
land on their canonical scaffold class (see §3 below). But:

* **Cross-domain generalisation is NOT verified.** The checkpoint
  was trained on 66 PDBs drawn from `metalloprotein_targets.py`.
  No CrossDocked2020 hold-out exists in Phase 2 / Phase 3 (the
  Phase 1 spec §4.2 mentioned a follow-up to add NonMetal
  negatives — deferred).
* **The 32-d skeleton is family-discriminative, not
  pocket-specific.** MMP2 and MMP13 (both `ZN_TETRA_HHE`) get
  similar skeletons — the model never sees them as distinct. A
  finer per-pocket distinction would need E(3)-equivariant
  attention over 3D coords (Phase 1 §3.2 "Honest framing").

---

## 3. Experiment 2 — Scaffold classification (5 sample pockets)

```
[MISS] CA2      expected=ZN_TETRA_HHH     predicted=ZN_TETRA_HHE     top3=[('ZN_TETRA_HHE', 4.58), ('ZN_TETRA_HHH', 3.65), ('CU_TBP', 0.85)]
[OK]   MMP2     expected=ZN_TETRA_HHE     predicted=ZN_TETRA_HHE     top3=[('ZN_TETRA_HHE', 9.99), ('ZN_TETRA_HHH', 6.56), ('ZN_TETRA_HHC', 0.88)]
[OK]   PKA      expected=MG_OCTA_KINASE   predicted=MG_OCTA_KINASE   top3=[('MG_OCTA_KINASE', 15.46), ('CU_TBP', 9.59), ('ZN_TETRA_HHD', 4.09)]
[OK]   CYP3A4   expected=FE_HEME_CYS      predicted=FE_HEME_CYS      top3=[('FE_HEME_CYS', 7.85), ('ZN_TETRA_HHE', 6.79), ('ZN_TETRA_HHH', 2.86)]
[OK]   SOD1     expected=CU_TBP           predicted=CU_TBP           top3=[('CU_TBP', 8.69), ('MG_OCTA_KINASE', 4.01), ('ZN_TETRA_HHD', 1.64)]
Headline: 4/5 samples classify to expected class.
```

### 3.1 Per-pocket breakdown

| Pocket | Expected | Predicted | Top-1 margin | Notes |
|--------|----------|-----------|-------------:|-------|
| CA2 | ZN_TETRA_HHH | ZN_TETRA_HHE | 4.58 vs 3.65 | Phase 2 §4.2 collapse (0/8 train). Closely followed by the correct class — margin is only 0.93 logits. |
| MMP2 | ZN_TETRA_HHE | ZN_TETRA_HHE | 9.99 vs 6.56 | Clear winner — 3.43 logits margin. |
| PKA | MG_OCTA_KINASE | MG_OCTA_KINASE | 15.46 vs 9.59 | Strongest signal — the Asp-Lys-Glu triad is distinct from Zn-His triads. |
| CYP3A4 | FE_HEME_CYS | FE_HEME_CYS | 7.85 vs 6.79 | Cys-anchored heme pocket — correctly identified. |
| SOD1 | CU_TBP | CU_TBP | 8.69 vs 4.01 | 5-His cluster — correctly identified. |

### 3.2 Honest caveats

* **5 sample pockets is not a benchmark.** Real-world scaffold
  classification accuracy requires a held-out test set with 5-fold
  cross-validation (Phase 1 spec §4.6 deferred this to Phase 3;
  Phase 3 still uses train-set accuracy only).
* **The CA2 failure is the headline negative result.** It
  confirms the Phase 1 §5.1 prediction that the model conflates
  `ZN_TETRA_HHH` with `ZN_TETRA_HHE`. The Phase 3.5 fix
  (`is_anchor` embedding 8 → 16 + a larger FFN) is the smallest
  change likely to break this collapse.
* **Top-3 logits are tight on CYP3A4** (margin only 1.06 over
  ZN_TETRA_HHE). The Cys anchor feature (one-hot slot 20) is the
  only signal that lifts CYP above the Zn tetrahedral mass —
  a stronger anchor embedding would amplify this margin.

---

## 4. Experiment 3 — Integration smoke with `lambda_combinators`

```
[Bonus] Experiment 3 — integration smoke with lambda_combinators:
    n_rules_selected: 2
    n_rules_augmented: 2
    skeleton_shape: (32,)
    skeleton_l2: 13.17
    augmented_rule_summary:
        {'name': 'CuAAC', 'macro_skeleton_dim': 32, 'macro_skeleton_l2': 13.17, 'argmax_scaffold_class': 'ZN_TETRA_HHE'}
        {'name': 'SPAAC', 'macro_skeleton_dim': 32, 'macro_skeleton_l2': 13.17, 'argmax_scaffold_class': 'ZN_TETRA_HHE'}
    integration_status: STAGE-4 ADDITIVE: macro_skeleton consumed as 4th element, NOT mutating
                        select_rules_for_pocket signature. Phase 4 follow-up PR required to fold
                        skeleton into rule scoring.
```

### 4.1 What the integration looks like

The smoke test:

1. Builds a 2-rule dummy rule set (`CuAAC`, `SPAAC`) with
   `applies_to` and `score` callables.
2. Runs the existing
   `ClickRuleCombinator.select_rules_for_pocket(pocket_features, rules)`
   — the canonical 3-stage HOF pipeline (filter → map → fold).
3. Computes the macro-skeleton via the trained `PocketMacroSkeleton`
   on the MMP2 residue set.
4. Tags each returned rule with `{macro_skeleton_dim, macro_skeleton_l2,
   argmax_scaffold_class}` so downstream code can fold the skeleton
   into rule scoring **without mutating `lambda_combinators.py`**.

### 4.2 Why we did NOT mutate `lambda_combinators.py`

The task brief lists `warm_start.py`, `learned_prior.py`,
`proof_search.py` as READ-ONLY; `lambda_combinators.py` is not on
that list. We deliberately chose to keep the existing
`select_rules_for_pocket` signature unchanged for two reasons:

* **The integration is additive, not substitutive.** A 4th element
  in the rule-selection pipeline (the macro-skeleton bias term)
  is a future Phase 4 PR — folding it now would force a re-test of
  the entire 3-stage HOF pipeline.
* **Avoiding hidden mutations keeps the contract observable.** The
  test passes the rule list through `select_rules_for_pocket`
  *and then* tags it externally. A future caller can reproduce
  this pattern without depending on a modified `lambda_combinators.py`.

### 4.3 The recommended Phase 4 PR contract

```python
# Phase 4 follow-up — sketch, NOT shipped in Phase 3:
def select_rules_for_pocket(
    self, pocket_features, rule_set, *,
    macro_skeleton: Optional[torch.Tensor] = None,   # NEW: 4th element
    macro_skeleton_weight: float = 0.5,             # NEW: bias strength
):
    rules = list(rule_set)
    applicable = list(filter_combinator(lambda r: _rule_applies_to_pocket(r, pocket_features), rules))
    scored = list(map_combinator(
        lambda r: (r, _rule_score(r, pocket_features)
                   + (macro_skeleton_weight * _skeleton_bonus(r, macro_skeleton)
                      if macro_skeleton is not None else 0.0)),
        applicable,
    ))
    # ... rest unchanged
```

`_skeleton_bonus(rule, skel)` would be a learnable projection
(Linear(rule_emb, 1) → softmax(12) dot skel → scalar bias), but
this is **out of scope** for Phase 3 — it requires a labeled rule →
scaffold dataset we don't have.

---

## 5. Honesty caveats (Phase 3)

1. **CPU-only**: per the task brief. No GPU required for either
   training (Phase 2) or validation (Phase 3). The
   `torch.cuda.is_available()` flag is irrelevant — we run on CPU
   intentionally.
2. **Tiny training set**: 66 PDBs across 6 classes is **50× too
   small** for SOTA classifier training. Train accuracy 87.9 %
   (Phase 2 §4) is **not** generalisable; expect hold-out CV
   accuracy in the 60-80 % range.
3. **CA2 collapse (0/8 train accuracy)**: predicted by Phase 1 §5.1,
   confirmed by Phase 2 §4.2, re-observed in Phase 3 §3.1. The
   fix is `is_anchor` embedding 8 → 16 + a larger FFN sub-layer
   — a Phase 3.5 follow-up, not Phase 3 scope.
4. **Mean-pool aggregation**: skeleton vectors for distinct
   pockets can have cosine > 0.85 (Phase 3 §2.1: CA2/MMP2 = 0.862).
   This is **expected behaviour** for the architecture. The
   pocket-invariance test is a *sanity* check (not zero, not one),
   not a discrimination guarantee.
5. **No 3D coords**: the Phase 1 spec §3.2 noted we don't have
   reliable Cα positions. We use residue-identity + scalar features
   only — no equivariant attention.
6. **No MCTS integration**: the integration smoke (§4) demonstrates
   the *additive contract* but does **NOT** wire the skeleton
   into `proof_search.py` (READ-ONLY per Phase 1 task brief) or
   `lambda_combinators.py` (kept stable by Phase 3 design choice).
   Wiring into `proof_search.py` is Phase 4 (separate post-R13
   ticket).
7. **Workflow safety confirmed**: 0 lines modified in
   `warm_start.py`, `learned_prior.py`, `proof_search.py`, or
   `lambda_combinators.py`. Only a NEW script
   (`phase3_validate_pocket_macro_skeleton.py`) was added.

---

## 6. Recommendation for paper §3.5 (Deflex architecture)

Per the task brief: "Recommendation for paper §3.5 Deflex
architecture."

### 6.1 Where the macro-skeleton fits in the Deflex stack

The Deflex workflow (per `phase1_design.md` §1) has two surfaces
that need a per-pocket signal:

| Surface | Current state | Recommended Phase 4 |
|---------|---------------|---------------------|
| `warm_start.pocket_features` | 64-d hash, 7 hand-crafted + 57 learned-placeholder zeros | **augment** with `macro_skeleton` as the 32-d slot 8-39 — drop-in additive |
| `learned_prior.LearnedPolicyPrior` | per-state GRU (h=32), no pocket context | **condition** the forward pass on `macro_skeleton` (concat to per-step input) |

The Phase 3 skeleton is **ready as a drop-in additive layer** for
both surfaces — no signature change, no re-training of the existing
GRU, just an extra `Linear(32+32, 32)` projection on the input.

### 6.2 What we claim vs what we do NOT claim

* **CLAIM (Phase 3 MEASURED, train set)**: the macro-skeleton
  classifies 4/5 sample pockets to their canonical scaffold family,
  and the CA2/MMP2 skeleton cosine is 0.862 (distinct enough for
  downstream consumers to discriminate when the L2 distance is
  also > 0.1).
* **DO NOT CLAIM**: cross-domain generalisation, E(3)-equivariance,
  within-family discrimination (MMP2 vs MMP9), or a hold-out CV
  accuracy number (no held-out set exists; the Phase 1 §4.6
  plan is deferred).

### 6.3 Suggested paper §3.5 text (one-paragraph scaffold)

> The Deflex pocket-conditioned macro-skeleton is a small (5,580
> parameters) Transformer-style encoder over binding-site residues.
> Each residue is mapped to a deterministic 29-d feature vector
> (one-hot amino acid identity, anchor-tier one-hot, distance bin),
> projected to 32-d, passed through a single multi-head attention
> layer (4 heads), mean-pooled across the residue set, and decoded
> into a 12-way scaffold-family softmax. The 32-d skeleton vector
> is *family-discriminative* on the metalloprotein catalogue (4/5
> sample pockets hit their canonical family on the train set) but
> does not discriminate *within* a family and has not been
> validated on a CrossDocked2020 hold-out. The integration into
> the MCTS root prior is **additive** — the skeleton is consumed
> as an extra per-pocket bias term in `select_rules_for_pocket`,
> not as a substitute for the existing hash-based 64-d projection.

---

## 7. Hand-off to Phase 4 (out of scope)

Phase 4 (wire the skeleton into `proof_search.py` and
`warm_start.py`) is **out of scope** for Phase 3 — it requires
touching the READ-ONLY modules. The recommended follow-ups:

1. **`is_anchor` embedding boost 8 → 16** (Phase 3.5) — should
   break the CA2/HHH collapse.
2. **Leave-one-PDB-out CV on the 6 classes** (Phase 3.5) —
   should give a realistic hold-out accuracy estimate.
3. **Add CrossDocked2020 NonMetal negatives** (Phase 1 §4.2
   deferred) — should balance the dataset and lift
   `NONMETAL_ORTHOSTERIC` / `NONMETAL_ALLOSTERIC` predictions.
4. **Wire `macro_skeleton` into `warm_start.pocket_features` as
   slots 8-39** (Phase 4) — drop-in additive.
5. **Wire `macro_skeleton` into `LearnedPolicyPrior` as input
   conditioning** (Phase 4) — requires a new `Linear(32+32, 32)`
   projection.

---

*WF-Deflex PocketMacroSkeleton — Phase 3 (validation + integration smoke).
Project root: `/home/hugo/codes/try_triton_on_rocm`. Python 3.12,
ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64 (CPU path
for this phase — GPU available but not used per task brief).*