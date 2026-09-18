# WF-Deflex PocketMacroSkeleton — Phase 2 inference smoke

**Date:** 2026-09-15
**Author:** integration agent (delegated from main orchestrator)
**Workflow:** WF-Deflex Follow-up Phase 2
**Checkpoint owner:** wyyy283ck (PocketMacroSkeleton v2 retrain + is_anchor tier 4→8 fix)
**Integration scope:** inference adapter + 5-pocket smoke + tests (NOT the retrain — that owns wyyy283ck)

---

## TL;DR

| metric                                  | value                                    |
|-----------------------------------------|------------------------------------------|
| Checkpoint exists                       | YES (`pocket_macro_skeleton.pt`, 26 KB)  |
| Strict load (`strict=True`)             | OK — 5,580 params, no missing keys      |
| Per-pocket ground-truth matches         | **4/5** (CA2 is the documented collapse) |
| Embedding dim                           | 32 (matches `HIDDEN_DIM`)                |
| Tests                                   | **10/10 pass** in 1.1 s                  |
| Smoke wall                              | <2 s (CPU only, no GPU required)         |

The shipped artefacts (`pocket_macro_inference.py` + `eval_pocket_macro_skeleton.py` + `test_pocket_macro_inference.py`) load the trained checkpoint via a **v1 mirror** module (29-d input — matching the saved state_dict exactly). The v2 module (`pocket_macro_skeleton.py`) was NOT modified per the constraints.

---

## 1. Checkpoint audit

```
molmetal/models/pocket_macro_skeleton.pt        26 KB  (state_dict)
molmetal/models/pocket_macro_skeleton.pt.json   3.8 KB (metadata)
```

State_dict contents:

| key                                  | shape           |
|--------------------------------------|-----------------|
| `residue_encoder.linear.weight`      | `(32, 29)`      |
| `residue_encoder.linear.bias`        | `(32,)`         |
| `attention.mha.in_proj_weight`       | `(96, 32)`      |
| `attention.mha.in_proj_bias`         | `(96,)`         |
| `attention.mha.out_proj.weight`      | `(32, 32)`      |
| `attention.mha.out_proj.bias`        | `(32,)`         |
| `attention.head.weight`              | `(12, 32)`      |
| `attention.head.bias`                | `(12,)`         |

**Total parameters: 5,580** (matches `.pt.json` "n_params": 5580).

**Architectural note.** The current `pocket_macro_skeleton.py` is the **v2** variant with `PER_RESIDUE_FEATURES = 33` (the Phase 3 CA2-fix adds the `anchor_position` one-hot). The trained checkpoint is the **v1** variant with `PER_RESIDUE_FEATURES = 29`. Loading the v1 state_dict into the v2 module produces `size mismatch for residue_encoder.linear.weight: copying a param with shape torch.Size([32, 29]) from checkpoint, the shape in current model is torch.Size([32, 33]).`

The integration agent does **not** have authority to modify `pocket_macro_skeleton.py` (that file is owned by wyyy283ck for the v2 retrain). The inference adapter therefore constructs a **v1 mirror module** inline (`molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py:_PocketMacroSkeletonV1`) that:

1. Bypasses `PocketResidueEncoder.__init__`'s hard-coded `PER_RESIDUE_FEATURES=33` check by attaching the `Linear` directly via `nn.Module()`.
2. Loads the state_dict with `strict=True` — any future architectural drift surfaces immediately (no silent fallback).
3. Preserves the v2 `PocketMacroSkeletonAttention` (MHA + mean-pool + Linear head) so the softmax head shape `(12, 32)` matches the saved checkpoint exactly.

This is documented in the inference module's module docstring (search for "v1 mirror").

---

## 2. Per-pocket predictions vs ground truth

| pocket | PDB id | ground-truth class | predicted class | confidence | source | match |
|--------|--------|--------------------|-----------------|------------|--------|-------|
| CA2    | 1AKL   | ZN_TETRA_HHH       | ZN_TETRA_HHE    | 0.684      | model  | MISS (documented collapse) |
| ACE    | 1O86   | ZN_TETRA_HHE       | ZN_TETRA_HHE    | 0.982      | model  | OK |
| MMP2   | 1HOV   | ZN_TETRA_HHE       | ZN_TETRA_HHE    | 0.968      | model  | OK |
| PKA    | 1ATP   | MG_OCTA_KINASE     | MG_OCTA_KINASE  | 0.997      | model  | OK |
| CYP3A4 | 1W0G   | FE_HEME_CYS        | FE_HEME_CYS     | 0.728      | model  | OK |

**Headline accuracy: 4/5 (80%).**

### Honest framing

* **CA2 (1AKL) → ZN_TETRA_HHH collapse.** The model predicts ZN_TETRA_HHE for CA2 with softmax mass 0.684. This is the documented Phase 2 §4.2 failure mode (train accuracy 0/8 on the ZN_TETRA_HHH class). The v1 mean-pool aggregator cannot distinguish "His at anchor-position 0,1,2" (CA2 / ZN_TETRA_HHH) from "His at all three positions with one Glu present" (MMP / ZN_TETRA_HHE). The v2 fix (extending the `anchor_position` one-hot to break this conflation) is owned by wyyy283ck and not yet trained — the v2 checkpoint is the **next** deliverable.

* **PKA → MG_OCTA_KINASE.** 100 % confidence (0.997). The kinase Asp-Lys-Glu triad is chemically very different from any Zn²⁺ triad — the model has learned this distinction cleanly.

* **MMP2 / ACE → ZN_TETRA_HHE.** 96-98 % confidence. Both have His-His-Glu anchor triads so the model converges on the correct class with high mass.

* **CYP3A4 → FE_HEME_CYS.** 72.8 % confidence. The CYP class has only 8 training PDBs and a smaller residue fixture (5 residues vs 7-10 for the kinase/Zn²⁺ pockets) so the softmax is less peaked. Still well above the 0.50 confidence floor.

* **ACE↔MMP2 cosine similarity = 0.999.** Both are ZN_TETRA_HHE so this is *expected* — the embeddings are discriminative across metalloprotein families (CA2 ↔ PKA cosine = −0.793) but NOT within a single train class.

---

## 3. 32-d embedding bridge for downstream integration

The `PocketMacroInference.get_embedding(target_name)` method returns a 32-d `np.float32` vector per pocket. These are the **bridge** between the trained PocketMacroSkeleton checkpoint and the downstream `warm_start.py` / `learned_prior.py` integration owned by the **pocket-invariance** workflow.

Per-pocket L2 norms:

| pocket | L2 norm  |
|--------|----------|
| CA2    | 5.98     |
| ACE    | 14.61    |
| MMP2   | 13.17    |
| PKA    | 18.74    |
| CYP3A4 | 17.00    |

Pairwise cosine similarity matrix (off-diagonal entries span `[−0.97, +1.0]`):

```
            CA2     ACE    MMP2     PKA    CYP3A4
CA2        1.000   0.843   0.862  -0.793   0.323
ACE        0.843   1.000   0.999  -0.976   0.745
MMP2       0.862   0.999   1.000  -0.970   0.725
PKA       -0.793  -0.976  -0.970   1.000  -0.693
CYP3A4     0.323   0.745   0.725  -0.693   1.000
```

### Honest framing for downstream consumers

1. **PKA is the most-distinct pocket** (cosine ~ ±0.8 to ±1.0 vs the Zn²⁺ families). This is consistent with the kinase pocket having a different metal coordination chemistry (Mg²⁺ octahedral + ATP competitive).

2. **ACE ↔ MMP2 collapse** (cosine = 0.999). They share the same train class (`ZN_TETRA_HHE`) and the same His-His-Glu anchor triad. The pocket-invariance workflow that wants to *break* pocket-invariance via the root prior cannot rely on the v1 embedding alone for these two targets.

3. **CA2 is the smallest-norm embedding** (L2 = 5.98). This is the documented mean-pool collapse — CA2's input signal is averaged down to a near-uniform distribution that sits close to the model's "average pocket" baseline.

4. **The v2 retrain (wyyy283ck) will replace these vectors.** When the v2 checkpoint ships (with the `anchor_position` one-hot fix), the inference adapter will need a parallel `V2_PER_RESIDUE_FEATURES = 33` mirror module. The public API (`predict_scaffold_class`, `get_embedding`, etc.) can stay unchanged.

---

## 4. Files shipped

| file                                                                                          | LOC  | purpose                                              |
|-----------------------------------------------------------------------------------------------|------|------------------------------------------------------|
| `molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py`                                    | 380  | v1 mirror + PocketMacroInference adapter             |
| `molmetal/scripts/eval_pocket_macro_skeleton.py`                                              | 150  | 5-pocket smoke + embedding L2 / cosine matrix         |
| `molmetal/molmetal_lam/tests/test_pocket_macro_inference.py`                                  | 230  | 10 tests (load / predict / embedding / missing-cp)    |
| `molmetal/reports/wf_deflex_followup/phase2_pocket_macro_smoke.md`                            | this | integration report + honest framing                   |

**Total: ~760 LOC** across 4 files (within the 30-min wall budget).

---

## 5. Constraints respected

| file                                                                            | status                |
|---------------------------------------------------------------------------------|-----------------------|
| `molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py` (wyyy283ck owns)      | UNTOUCHED             |
| `molmetal/molmetal_lam/search_alg/proof_search.py` (pocket-invariance owns)    | UNTOUCHED             |
| `molmetal/molmetal_lam/search_alg/warm_start.py` (pocket-invariance owns)       | UNTOUCHED             |
| `molmetal/molmetal_lam/search_alg/learned_prior.py` (pocket-invariance owns)   | UNTOUCHED             |
| `molmetal/scripts/r4_lambda_only_run.py` (pocket-invariance owns)               | UNTOUCHED             |
| `molmetal/adapters/flow_matching_lipman/*` (CFM frontier owns)                 | UNTOUCHED             |
| `molmetal/molmetal_lam/reactions/beta_reductions.py` (parallel-fixes owns)    | UNTOUCHED             |
| `molmetal/molmetal_lam/lam_chem/lambda_combinators.py` (parallel-fixes owns)   | UNTOUCHED             |
| `paper/*`                                                                       | UNTOUCHED             |

All new code lives under `molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py` (NEW) + `molmetal/scripts/eval_pocket_macro_skeleton.py` (NEW) + `molmetal/molmetal_lam/tests/test_pocket_macro_inference.py` (NEW) + the report above.

---

## 6. Tests (10/10 pass, 1.1 s)

```
uv run pytest molmetal/molmetal_lam/tests/test_pocket_macro_inference.py --tb=short -q
.......... [100%]
10 passed, 1 warning in 1.09s
```

| # | test                                  | asserts                                                      |
|---|---------------------------------------|--------------------------------------------------------------|
| 1 | `test_load_checkpoint`                | `_loaded is True`, 5,580 trainable params                    |
| 2 | `test_is_available`                   | `is_available() is True` after init                          |
| 3 | `test_get_embedding_dim`              | `(32,) float32` for CA2/MMP2/PKA/CYP3A4                       |
| 4 | `test_predict_pka`                    | MG_OCTA_KINASE, conf ≥ 0.80                                   |
| 5 | `test_predict_mmp2`                   | ZN_TETRA_HHE, conf ≥ 0.80                                     |
| 6 | `test_predict_cyp3a4`                 | FE_HEME_CYS, conf ≥ 0.50 (above floor)                       |
| 7 | `test_predict_ca2_honest_collapse`    | raw argmax != ZN_TETRA_HHH, floored stays ZN_TETRA_HHE       |
| 8 | `test_handles_missing_checkpoint`     | UNKNOWN fallback, zeros embedding                            |
| 9 | `test_unknown_target_returns_unknown` | UNKNOWN for unknown name, source=fallback                    |
| 10| `test_metadata_sidecar_loaded`        | per-class accuracy loaded (ZN_TETRA_HHH=0.0, MG_OCTA_KINASE=1.0) |

---

## 7. Smoke output (full)

```
========================================================================
WF-Deflex PocketMacroSkeleton — Phase 2 inference smoke
========================================================================

[1/4] Loaded checkpoint: molmetal/models/pocket_macro_skeleton.pt
      Architecture:    v1 mirror (PER_RESIDUE_FEATURES=29, hidden_dim=32, n_classes=12)
      Trainable params: 5580 (expected 5,580)

[2/4] Per-pocket scaffold-class predictions (5 pockets):
    [MISS] CA2      expected=ZN_TETRA_HHH       predicted=ZN_TETRA_HHE       confidence=+0.684 source=model
    [OK] ACE      expected=ZN_TETRA_HHE       predicted=ZN_TETRA_HHE       confidence=+0.982 source=model
    [OK] MMP2     expected=ZN_TETRA_HHE       predicted=ZN_TETRA_HHE       confidence=+0.968 source=model
    [OK] PKA      expected=MG_OCTA_KINASE     predicted=MG_OCTA_KINASE     confidence=+0.997 source=model
    [OK] CYP3A4   expected=FE_HEME_CYS        predicted=FE_HEME_CYS        confidence=+0.728 source=model

      Headline: 4/5 pockets classify to ground truth.

[3/4] 32-d embedding L2 norms (downstream bridge for warm_start.py):
      CA2       L2=+5.9820  shape=(32,)
      ACE       L2=+14.6100  shape=(32,)
      MMP2      L2=+13.1732  shape=(32,)
      PKA       L2=+18.7404  shape=(32,)
      CYP3A4    L2=+17.0047  shape=(32,)
      Pairwise cosine similarity (pocket-distinct?):
                    CA2      ACE     MMP2      PKA   CYP3A4
      CA2        +1.000   +0.843   +0.862   -0.793   +0.323
      ACE        +0.843   +1.000   +0.999   -0.976   +0.745
      MMP2       +0.862   +0.999   +1.000   -0.970   +0.725
      PKA        -0.793   -0.976   -0.970   +1.000   -0.693
      CYP3A4     +0.323   +0.745   +0.725   -0.693   +1.000

[4/4] Training metadata (per-class accuracy):
      [OK  ] CU_TBP             correct=  8/8   acc=1.000
      [OK  ] FE_HEME_CYS        correct=  8/8   acc=1.000
      [OK  ] MG_OCTA_KINASE     correct= 16/16  acc=1.000
      [OK  ] ZN_TETRA_HHD       correct=  6/6   acc=1.000
      [OK  ] ZN_TETRA_HHE       correct= 20/20  acc=1.000
      [MISS] ZN_TETRA_HHH       correct=  0/8   acc=0.000

========================================================================
Phase 2 inference smoke COMPLETE — 4/5 ground-truth matches.
========================================================================
```

---

## 8. Honest caveats

1. **The shipped adapter only loads the v1 checkpoint.** When wyyy283ck ships the v2 retrain, the inference adapter will need a parallel `V2_PER_RESIDUE_FEATURES = 33` mirror. The public API is forward-compatible — only the internal `_PocketMacroSkeletonV1` will need a `_PocketMacroSkeletonV2` sibling.

2. **The CA2 collapse is a structural failure, not a numerical one.** The v1 mean-pool cannot distinguish position-specific His-triad patterns. No amount of additional training of the v1 architecture will fix this — only the v2 `anchor_position` feature will. Honest reporting of CA2 → ZN_TETRA_HHE is therefore the correct, reproducible measurement.

3. **The 32-d embeddings are NOT pocket-invariant.** Two pockets in the same train class (ACE / MMP2) have cosine similarity 0.999. The downstream `pocket-invariance` workflow that wants *more* pocket-specificity per state needs to use the embedding **additively** (mixed with other pocket signals), not as a sole signal.

4. **No real PocketResidue loading from PDB files.** The fixture residues are hand-curated from the canonical binding-site definitions in `metalloprotein_targets.py`. A production version would call `warm_start.pocket_features(pdb_id)` to populate them — out of scope here per the constraints.

5. **No CFM integration.** The downstream `learned_prior.LearnedPolicyPrior` is chemistry-aware but pocket-blind; the 32-d embedding is the gap-filler. Folding the embedding into the per-state RNN is owned by the pocket-invariance workflow, not this PR.

---

## 9. Recommended next actions

| priority | action                                                                  | owner                |
|----------|-------------------------------------------------------------------------|----------------------|
| HIGH     | Train v2 checkpoint (extend anchor tier 4→8 + retrain)                  | wyyy283ck            |
| HIGH     | Add `_PocketMacroSkeletonV2` mirror when v2 checkpoint ships            | integration agent    |
| MEDIUM   | Wire `get_embedding()` into `warm_start.pocket_features(target_name)`   | pocket-invariance    |
| MEDIUM   | Wire 32-d pocket embedding into `learned_prior.LearnedPolicyPrior`      | pocket-invariance    |
| LOW      | Add `is_metal_anchor` 4→8 tier to v2 + retrain (Phase 3 follow-up)      | wyyy283ck            |
| LOW      | Expand residue fixtures to all 8 metalloprotein families in metaltargets| integration agent    |

---

## 10. Run commands

```bash
# Tests
uv run pytest molmetal/molmetal_lam/tests/test_pocket_macro_inference.py --tb=short -q

# Smoke
uv run python molmetal/scripts/eval_pocket_macro_skeleton.py
```

Both run on CPU in <2 s. No GPU required. No external dependencies beyond PyTorch + numpy.

---

**End of report.**
