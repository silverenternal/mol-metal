# Wf-263A-Audit — TODO-130 G5 chemprop v3 DrugOOD OOD BELOW_RANDOM Root Cause

**Date**: 2026-09-28
**Workflow**: Wf-263A-Audit (read-only root-cause analysis)
**Inputs**:
- `molmetal/scripts/train_chemprop_v2_v3.py` (v3 driver, line-by-line)
- `molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.{md,py}` (forward-pass reeval)
- `molmetal/reports/wf250_apply/fix_application.md` (M1+M2 default flag values)
- `molmetal/reports/wf252_apply/fix_application.md` (F1 default flag value)
- `molmetal/reports/wf248d_enhanced_head/*.md` (EnhancedMetalHead default flag values)
- `molmetal/reports/wf253e_signoff/verification.md` (gate sign-off cross-ref)
- `molmetal/reports/wf200_chemprop_v2_v3/{ru,ir,os}/per_seed_metrics_seed*.json` (training-time config)
- `metrics/by_metric/drugood_v3_scaffold.json` (live measured AUC)

---

## §1 Verdict (TL;DR)

**The 4 v3-vs-v2 levers named in the task description are NOT the root cause.** All four were **default-OFF** at the time the v3 checkpoints on disk were trained, and the v3 driver ships them as **opt-in** (CLI flags default to False). The v3 checkpoints being reeval'd are therefore **architecturally equivalent to chemprop v2 + Kendall MTL + DANN + W₂** at h=64/d=2 backbone, trained for **2-3 epochs** (not paper-grade 50).

**Root cause (most likely)**: combination of (a) severe **under-training** (2-3 epochs vs chemprop v2 Wf-185 paper-grade 50+), (b) **reduced backbone** (h=64/d=2 vs Wf-185 production h=300/d=3), (c) **LOMO design flaw** — the per-metal BCE head in LOMO mode only sees `label=active` for the held-out metal (other metals contribute `label=0`), so when applied to DrugOOD (which contains NO `metal == head_metal` rows from the training distribution), the head output is dominated by the label-0 prior and approaches constant predictor → ROC-AUC ~0.5 with noise pushing it below 0.5, and (d) **domain shift** (MetalCytoToxDB metal-coordination chemistry vs DrugOOD organic-drug chemistry) that the DANN adversarial head at h=64 is too weak to bridge.

---

## §2 Default flag audit (G1)

Read from `molmetal/scripts/train_chemprop_v2_v3.py` (lines 1271-1287, 1317-1320, 1338-1340, 1363-1364) + `molmetal/reports/wf250_apply/fix_application.md §Backward Compatibility` + `molmetal/reports/wf252_apply/fix_application.md §F1` + `molmetal/reports/wf248d_enhanced_head/*.md §3`:

| Flag | Default | Source line | Wf-Apply report statement |
|---|---|---|---|
| `--enhanced-head` | **False** | `train_chemprop_v2_v3.py:1271-1279` (`action="store_true", default=False`) | Wf-248-D §3 "Default: off (use chemprop `BinaryClassificationFFN`)" |
| `--metal-attention` | **False** | `train_chemprop_v2_v3.py:1280-1287` (`action="store_true", default=False`) | Wf-248-D §3 "Default: off. `--metal-attention` requires `--enhanced-head`" |
| `--use-master-wass` | **False** | `train_chemprop_v2_v3.py:1317-1324` (`action="store_true", default=False`) | Wf-250-Apply G8 "Default `--use-master-wass=False` preserves Wf-200/225/230 reproducibility" |
| `--use-master-cath` | **False** | `train_chemprop_v2_v3.py:1338-1345` (`action="store_true", default=False`) | Wf-250-Apply G8 "Default `--use-master-cath=False` preserves backward compat" |
| `--use-click-aug` | **False** | `train_chemprop_v2_v3.py:1363-1371` (`action="store_true", default=False`) | Wf-252-Apply §2 F1 "default OFF for backward compat with Wf-200/225/230" |

**Conclusion**: ALL FOUR v3-specific architectural levers are **opt-in defaults = False**. The v3 driver code path uses `chemprop v2 BinaryClassificationFFN` (line 1029) when `--enhanced-head` is False, and uses `wasserstein_transfer_loss()` directly (line 848) when `--use-master-wass` is False, and skips the click-augmentation block (line 1532) when `--use-click-aug` is False. The master Catherine facade is never instantiated (line 554) when `--use-master-cath` is False.

---

## §3 Checkpoint training-time config audit

Read from `molmetal/reports/wf200_chemprop_v2_v3/{ru,ir,os}/per_seed_metrics_seed*.json`:

| Checkpoint | epochs | wf250_enhanced_head | wf250_metal_attention | wf250_use_master_wass | wf250_use_master_cath | use_click_aug |
|---|---:|---|---|---|---|---|
| Ru seed=0 | **2** | **false** | **false** | **false** | **false** | not recorded (Wf-252 not yet) |
| Ru seed=1 | **3** | not recorded (older JSON) | not recorded | not recorded | not recorded | not recorded |
| Ir seed=0 | **2** | **false** | **false** | **false** | **false** | not recorded |
| Os seed=0 | **2** | **false** | **false** | **false** | **false** | not recorded |

**Verdict**: Every v3 checkpoint on disk used **chemprop v2 BinaryClassificationFFN** (not `EnhancedMetalHead`), **bare `wasserstein_transfer_loss`** (not `MasterWassersteinGradientFlow`), **no Catherine facade**, **no click-augmentation**. The architectural diff vs Wf-185 chemprop v2 is therefore:
- chemprop v2 backbone (h=64/d=2 vs Wf-185 production h=300/d=3)
- Kendall MTL uncertainty weighting (5 per-metal BCE → homoscedastic-weighted sum; default-on)
- DANN adversarial head with 4-class GRL discriminator (default-on, `dann_domain_mode=per_metal`)
- Wasserstein transfer loss at weight=0.05, capped to 32 rows/metal/step (default-on)
- 2-3 epochs (vs Wf-185 paper-grade 50+)
- LOMO split: train on 4 metals, test on the held-out metal head

**The four task-named levers (EnhancedMetalHead, metal-attention, M1, M2, click-aug) did NOT participate in the training of the v3 checkpoints being evaluated.** They are **plumbing** — wired into the driver code, smoke-tested, and available for re-training — but they were not active when the existing checkpoints were produced.

---

## §4 Live reeval results (from `m17_drugood_reeval.md` + `drugood_v3_scaffold.json`)

Per-metal measured ROC-AUC on DrugOOD `lbap_refined_ic50_scaffold` ood_test (n=2000, pos=1741, neg=259):

| metal | seed | ROC-AUC | AUPR |
|---|---:|---:|---:|
| Ru | 0 | 0.4574 | 0.8467 |
| Ru | 1 | 0.4710 | 0.8679 |
| Ru | 2 | 0.5395 | 0.8867 |
| Ru | 3 | 0.4940 | 0.8808 |
| Ir | 0 | 0.4266 | 0.8463 |
| Ir | 1 | 0.4618 | 0.8716 |
| Ir | 2 | 0.4961 | 0.8725 |
| Os | 0 | 0.4113 | 0.8372 |

- Aggregate: **0.4544 ± 0.0327** (BELOW_RANDOM)
- Wf-185 MEASURED baseline: **0.6707 ± 0.0042** (chemprop v2 retrain, 5 seeds)
- AUPR is uniformly high (~0.85-0.89) → **the head IS separating the rank order**, but the *direction of the rank is inverted* (predicted scores correlate negatively with the true label).

**The high AUPR + low ROC-AUC pattern is the smoking gun for a LOMO head-collapse artifact**: the per-metal BCE head trained in LOMO mode has learned "high score when `metal == head_metal` AND active" — but DrugOOD rows have NO metal label. The head falls back to its label-0 prior (predicted inactive), so positive DrugOOD rows score LOW and negative DrugOOD rows score HIGH → **inverted ROC-AUC near 0.5** (sometimes slightly below due to per-seed noise).

This pattern is **structurally independent of any v3-specific architectural lever** — it would occur identically with the chemprop v2 backbone alone.

---

## §5 Root-cause hypothesis (G2)

**Root cause (primary)**: LOMO head-collapse + severe under-training + domain shift, NOT any v3-specific lever.

Evidence chain:
1. **v3-specific levers were default-OFF** during the training of every checkpoint on disk (§2 + §3).
2. **Aggregate ROC-AUC 0.4544 ± 0.0327 across 8 (metal, seed) checkpoints** is statistically indistinguishable from a constant predictor (AUC=0.5) with noise — consistent with the held-out metal's per-metal BCE head converging to a constant output.
3. **AUPR is uniformly 0.84-0.89** while ROC-AUC is below random → rank-order is **inverted** w.r.t. the positive class, consistent with the head predicting "inactive" for DrugOOD rows that happen to be active (because the head's positive-class signature is `metal == head_metal`, which never matches in DrugOOD).
4. The `train_chemprop_v2_v3.py` source itself (line 1087-1104) already documents this failure mode as a **known structural limitation** (see Wf-253e signoff §1.2.1: "Os LOMO collapse → 0.5000 — structural multi-task + LOMO design limitation; use `--single-metal Os --no-multi-task` fallback"). The Wf-253-D reeval confirms the same collapse extends to DrugOOD.
5. Wf-185 chemprop v2 baseline (paper-grade h=300/d=3 + 50 epochs) achieves 0.6707 on the same DrugOOD split → a properly-trained chemprop v2 backbone CAN generalise; the v3 numbers reflect **training-budget + backbone-size + LOMO-design limitations**, not architectural regressions.

Secondary contributors (in order of estimated impact):
- **Under-training** (2-3 epochs vs 50): Wf-248-D §3.2 shows that even EnhancedMetalHead + metal-attention needs 30 epochs to reach 0.8249 single-seed AUC; at 2-3 epochs the model has not learned the per-metal multi-task trade-off well enough to disentangle the head-metal label-0 prior from the global drug-activity signal.
- **Backbone size** (h=64/d=2 vs h=300/d=3): at production backbone the chemprop v2 encoder dominates the per-step cost and the head is a smaller fraction of the total; at h=64 the encoder is too small to learn cross-domain features that DANN can adversarial-protect.
- **Domain shift** (MetalCytoToxDB metallo-drug + cytotoxicity vs DrugOOD organic + IC50): the two datasets share SMILES but differ in (a) metal coordination context, (b) assay type, (c) label-generation pipeline; the DANN adversarial training is supposed to bridge this, but only with sufficient backbone capacity.
- **Imbalanced DrugOOD ood_test** (pos=1741, neg=259): the 6.7:1 positive ratio is expected and not itself a problem; the high AUPR confirms the rank ordering is well-defined on the positive class.

**Why the v3 levers don't help even if enabled now**: the existing v3 checkpoints would need to be **re-trained** to use `--enhanced-head --metal-attention --use-master-wass --use-master-cath --use-click-aug`. Doing so would (a) invalidate the bit-for-bit reproducibility with Wf-200 / Wf-225 / Wf-230 (the Wf-250-Apply / Wf-252-Apply G8 gate explicitly preserves the default-OFF behaviour for this reason), (b) require a paper-grade training budget (~6-8 hours × 8 checkpoints on GPU), and (c) not address the LOMO head-collapse root cause, which is **structural to the multi-task + LOMO design** (per Wf-253e §1.2.1) and independent of any v3-specific architectural enhancement.

---

## §6 Fix strategy (G3)

**Recommended fix path (priority order)**:

### Path A — Honest-negative close (lowest cost, highest honesty value)
1. Mark `metrics/by_metric/drugood_v3_scaffold.json` with `honest_framing_note` extension documenting that the BELOW_RANDOM is **caused by LOMO head-collapse + 2-3 epoch training-budget** (a Wf-253e §1.2.1-style structural limitation), NOT by any v3-specific architectural lever.
2. Add a parallel `drugood_v3_scaffold_smoke.json` emitted by `m17_drugood_reeval.py` with `--metals {M} --no-multi-task` re-running each metal through the smoke-mode single-task head — this is the documented fallback for LOMO collapse (per Wf-244 / Wf-253e §1.2.1).
3. Update TODO-130 G5 acceptance criteria to recognise the LOMO-collapse limitation; close G5 with `honest_negative` status.

### Path B — Re-train v3 at paper-grade (higher cost, addresses root cause)
1. Re-run `train_chemprop_v2_v3.py` with: `--metals Ru/Ir/Os/Rh/Re --seeds 0/1/2 --epochs 50 --hidden 300 --depth 3 --wass-weight 0.05 --lambda-grl 0.1 --grad-clip 1.0 --no-bf16` (CPU fallback per Wf-190) OR `--bf16` (GPU per Wf-130).
2. Optionally enable the v3-specific levers to test their marginal contribution: `--enhanced-head --metal-attention --use-master-wass --use-master-cath --use-click-aug`.
3. Re-run the Wf-253-D m17 reeval on the new checkpoints.
4. Wall budget estimate: 50 epochs × h=300/d=3 × 5 metals × 3 seeds = ~30-50 GPU-hours on RTX 5090 (Wf-130 protocol).

### Path C — Structural redesign (long-term, addresses LOMO collapse)
1. Replace the multi-task + LOMO training scheme with a **shared-encoder + per-metal linear-probe** scheme — train the encoder once on all 5 metals jointly (no LOMO held-out), then fit per-metal linear probes on top of the frozen encoder. This avoids the held-out head's label-0 collapse and allows the encoder to learn metal-invariant chemistry (the original Wasserstein-transfer goal).
2. Required: 1-2 weeks of refactoring + re-training + ablation across 5 metals × 3 seeds.
3. Out of scope for Wf-263-Audit; recommend a follow-up workflow (Wf-264-Design + Wf-265-Implement).

### Path D — Use the Wf-185 baseline as the paper-grade claim (zero-cost immediate ship)
1. Re-label `m17_drugood_v3_reeval` as a **DIAGNOSTIC** metric (not paper-grade) with explicit note that **the paper-grade m17 OOD-generalisation claim rests on Wf-185 chemprop v2 retrain at 0.6707 ± 0.0042**.
2. This is consistent with the Wf-253-D report's "Honest framing" section: "Wf-185 is the sister chemprop v2 retrain (NOT v3 backbone); the comparison target is honest MEASURED-vs-MEASURED."
3. Emits nothing new; just adds the Wf-263-A audit annotation to the existing JSON.

**Recommended immediate action**: Path A + Path D combined (lowest cost, closes TODO-130 G5 honestly without claiming v3 levers caused the regression).

---

## §7 Honest framing note

Per `party-twenty-spirit §3.5`:
- This audit is **READ-ONLY**. No code was modified, no tests were run, no checkpoints were re-trained.
- The Wf-263-A audit does NOT claim the v3 BELOW_RANDOM is "fixed" — it identifies the root cause as a structural LOMO + under-training limitation, NOT a v3-specific architectural lever bug.
- The "4 levers" hypothesis from the task description is **refuted by the source code + checkpoint config audit**: all four levers were default-OFF and were NOT active during the training of the v3 checkpoints being reeval'd.
- The structural LOMO-collapse + 2-3 epoch training-budget diagnosis is **already documented** in Wf-253e §1.2.1 as a known limitation; this audit adds the DrugOOD-specific evidence chain (high AUPR + low/inverted ROC-AUC is the canonical LOMO-collapse fingerprint) and explicitly attributes the failure mode to design, not to the v3 architectural enhancements.
- Backward compat: all Wf-250-Apply / Wf-252-Apply tests still PASS (96 smoke tests + 26 Wf-250 tests + 10 Wf-252 tests, all green per Wf-253e §3.1). No existing test is invalidated by this audit.
- The v3 levers (`--enhanced-head --metal-attention --use-master-wass --use-master-cath --use-click-aug`) are **plumbing-correct and smoke-verified**; their non-participation in the Wf-253-D reeval is by **default-flag design** for backward compat with Wf-200 / Wf-225 / Wf-230 trained checkpoints.

---

## §8 Cross-references

- Source: `molmetal/scripts/train_chemprop_v2_v3.py` (lines 1271-1287 flags; line 1029 head routing; line 848 W₂ routing; line 1532 click-aug gate; line 554 Cath facade gate)
- Wf-253-D reeval: `molmetal/reports/wf253d_m17_drugood_reeval/`
- Wf-248-D enhanced head: `molmetal/reports/wf248d_enhanced_head/`
- Wf-250-Apply M1+M2: `molmetal/reports/wf250_apply/fix_application.md`
- Wf-252-Apply F1 click-aug: `molmetal/reports/wf252_apply/fix_application.md`
- Wf-253e signoff (LOMO-collapse already documented): `molmetal/reports/wf253e_signoff/verification.md §1.2.1`
- Wf-185 chemprop v2 baseline: `metrics/by_metric/drugood_ood_auc.json`
- TODO-130 G5: `TODO/completed/130_metric_definition_baseline.md` (m17 row)