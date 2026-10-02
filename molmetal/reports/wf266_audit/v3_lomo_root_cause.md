# Wf-266-Audit — chemprop v3 LOMO Head-Collapse Root Cause

**Date**: 2026-09-28
**Workflow**: Wf-266-Audit (READ-ONLY root-cause deep-dive)
**Inputs**:
- `molmetal/scripts/train_chemprop_v2_v3.py` (v3 driver, line-by-line)
- `molmetal/reports/wf263a_audit/v3_drugood_root_cause.md` (prior root-cause audit, TODO-130 G5 close)
- `molmetal/reports/wf253e_signoff/verification.md` (sign-off, structural LOMO-limitation note §1.2.1)
- `molmetal/reports/wf200_chemprop_v2_v3/{ru,ir,os}/per_seed_metrics_seed*.json` (training-time config + run logs)
- `molmetal/models/chemprop_v2_mt_model.py` (multi-task backbone — referenced)

---

## §1 Verdict (TL;DR)

The chemprop v3 BELOW_RANDOM on DrugOOD (aggregate ROC-AUC 0.4544 ± 0.0327) is **not** caused by the v3-specific architectural levers (`--enhanced-head`, `--metal-attention`, `--use-master-wass`, `--use-master-cath`, `--use-click-aug`). All five were **default-OFF** at training time (verified in Wf-263A-Audit §2-§3). The root cause is a **structural LOMO design flaw** in the multi-task + LOMO recipe, amplified by a **reduced backbone** (h=64/d=2 vs Wf-185 production h=300/d=3) and **2-3 epoch under-training** (vs Wf-185 paper-grade 50). This audit adds three new findings on top of Wf-263A-Audit:

1. The head-collapse mechanism is **deterministic given LOMO + label construction** — not a numerical accident. Once `train_idx` excludes all rows of the held-out metal, the held-out head's BCE label tensor is structurally all-zeros and the BinaryClassificationFFN converges to a constant predictor (predicted logit → -∞, predicted probability → 0).
2. The default script parameters (`--hidden=64`, `--depth=2`, `--epochs=50`) are **honest configuration** but the **operational default** used in every v3 checkpoint on disk was the smoke-mode floor (`--epochs=2-3`, no flag override for `--hidden`/`--depth`). This was a **time-budget constraint**, not an architectural choice — and it should be reflected in the metric files' honest-framing notes.
3. The fix is **structural** (Path C — shared encoder + per-metal linear probes); the current v3 driver **cannot** be salvaged by flipping the v3-specific architectural levers. The Wf-263A recommended path (Path A + Path D) remains valid for closing TODO-130 G5 without claiming v3 levers caused the regression.

---

## §2 LOMO head-collapse mechanism (G1)

### §2.1 Mechanism: label construction + held-out head's BCE loss

In LOMO (Leave-One-Metal-Out) mode, the v3 driver partitions the corpus as follows (lines 1706-1718 of `train_chemprop_v2_v3.py`):

```python
for target_metal in args.metals:
    target_rows = np.where(metals_all == target_metal)[0]
    if lomo_eligible:
        train_mask = metals_all != target_metal   # <-- exclude target
        test_mask  = metals_all == target_metal
        train_idx  = np.where(train_mask)[0]
        test_idx   = np.where(test_mask)[0]
```

The held-out metal's training labels are then constructed (lines 681-687):

```python
y_true_per_metal: Dict[str, np.ndarray] = {}
for m in model.metals:
    # Per-row binary target = active if row's metal == m else 0.
    y_true_per_metal[m] = np.where(
        metals_all == m, active_all.astype(np.float32), 0.0
    ).astype(np.float32)
```

**Critical observation**: when `m` equals the held-out metal, `metals_all == m` is **always False** for rows in `train_idx` (because those rows are filtered out by `train_mask = metals_all != target_metal`). Therefore `y_true_per_metal[held_out_metal]` is **all zeros** on the training data.

During training, the per-metal BCE loop (lines 814-817) computes:

```python
for m in active_metals_for_loss:
    logit_1 = logits_dict[m]  # (B, 1)
    tgt     = y_true_b[m].unsqueeze(-1).float()  # (B, 1) — all zeros for held-out metal
    per_metal_losses.append(bce_per_metal(logit_1, tgt))
```

For the held-out metal's BinaryClassificationFFN head, `bce_per_metal(logit, zeros)` is **minimised** by sending `logit → -∞` (sigmoid → 0). The gradient on the held-out head's weights is therefore a **single, monotonic "push toward more negative logits"** signal — the head never sees a positive example and cannot learn anything other than "always predict inactive".

At eval time (lines 1113-1122), the constant-prediction guard fires:

```python
if scores and float(np.ptp(scores)) == 0.0:
    logger.warning(
        "[metal=%s seed=%d] held-out predictions are CONSTANT "
        "(score=%.6f for all %d test rows); test_auc=0.5000 "
        "indicates the head collapsed.  Honest-negative: the "
        "multi-task BCE in LOMO trains the held-out head on "
        "label=0 only (no positive labels exist for the held-out "
        "metal in train_idx).  Structural fix deferred to R20+.",
        ...
    )
```

For DrugOOD reeval (which is not part of the MetalCytoToxDB corpus at all), the held-out head is applied to **completely out-of-distribution** SMILES — but the underlying failure mode is identical: the head was trained to output ~0 regardless of input, so on DrugOOD it produces uniform scores near 0, and the ROC-AUC drifts to ~0.5 (random) with per-seed noise pushing it slightly below.

### §2.2 Why this is **structural** (independent of v3 levers)

The collapse depends only on:
- `y_true_per_metal[m]` for held-out `m` being all-zero in `train_idx`
- `BCEWithLogitsLoss` being used as the per-head loss
- The held-out head being trained at all

None of these depend on `EnhancedMetalHead`, `metal-attention`, `MasterWassersteinGradientFlow`, `MasterCatherineClosureBias`, or click-augmentation. Switching `--chemprop-mode` to `v3` (which permits `EnhancedMetalHead`) does not change the label construction, so the collapse would persist identically. This is consistent with Wf-263A-Audit §2's finding that all five v3-specific levers were default-OFF and cannot have caused the regression.

### §2.3 Why DrugOOD exhibits BELOW_RANDOM (not just ~0.5)

Per Wf-263A-Audit §4:
- AUPR is uniformly 0.84-0.89 (rank-order is preserved on the positive class).
- ROC-AUC is 0.41-0.54 with aggregate 0.4544 ± 0.0327 (rank-order is **inverted** w.r.t. the positive class).

The inversion arises because the held-out head learned to fire **only** when `metal == head_metal AND active` — its positive-class signature requires a metal-context it never sees at DrugOOD inference. DrugOOD rows that are *actually* active therefore score LOW (because the head cannot fire its positive signature on out-of-distribution SMILES), and DrugOOD rows that are actually inactive score HIGH only when the head's negative prior happens to dominate (which is most of the time at the head's "always inactive" asymptote). The result is anti-correlated rank order → ROC-AUC < 0.5.

---

## §3 Default epoch floor (G2)

The script-level default is `DEFAULT_EPOCHS = 50` (line 210). However, the **operational default** that produced the v3 checkpoints on disk was 2-3 epochs (per Wf-263A-Audit §3 verification of per-seed JSON files):

| Checkpoint | epochs (training-time) | Source |
|---|---:|---|
| Ru seed=0 | **2** | `molmetal/reports/wf200_chemprop_v2_v3/ru/per_seed_metrics_seed0.json` |
| Ru seed=1 | **3** | `molmetal/reports/wf200_chemprop_v2_v3/ru/per_seed_metrics_seed1.json` |
| Ir seed=0 | **2** | `molmetal/reports/wf200_chemprop_v2_v3/ir/per_seed_metrics_seed0.json` |
| Os seed=0 | **2** | `molmetal/reports/wf200_chemprop_v2_v3/os/per_seed_metrics_seed0.json` |

The `--quick` flag (lines 1436-1438 + 1465-1469) sets epochs to 2; the unflagged default of 50 is overridden by the time-budget-driven sweep launcher (e.g., Wf-243 + Wf-225 + Wf-230 ran with a 7200s budget divided across many (metal, seed) combinations, forcing epochs → 2-3 in practice). **Effective floor for LOMO collapse diagnosis**: 2 epochs is enough for the held-out head to converge to "always inactive" — that convergence is **fast** because the gradient signal is monotonic (single class, all-zero labels).

**Recommendation**: report `cfg_epochs = 50` as the script default but `operational_default_epochs = 2-3` as the **actual epoch budget** used for the v3 checkpoints on disk. Future honest-framing notes should distinguish between the two.

---

## §4 Default backbone (G3)

- `DEFAULT_HIDDEN = 64` (line 211)
- `DEFAULT_DEPTH = 2` (line 212)

The script's own module-level docstring (lines 33-37) flags this as **not production-grade**:

> The chemprop v2 backbone is **smaller** than production here
> (`hidden_dim=64, depth=2`) — matches the Wf-156 smoke config.
> Production chemprop v2 uses 300/3; per-step cost is ~5-7x lower
> at 64/2, so the AUC ceiling is lower.

This is honest reporting at the script level. However, the Wf-253-D reeval did **not** note that the LOMO head-collapse mechanism is **orthogonal to backbone size** — even at h=300/d=3, the held-out head's BCE on all-zero labels would still converge to "always inactive". Increasing the backbone buys more headroom for the **non-held-out** heads but does not rescue the held-out head. The Wf-185 chemprop v2 baseline at h=300/d=3 (0.6707 ± 0.0042 on the same DrugOOD split) is a **single-task + no-LOMO** baseline, which is why it avoids the collapse by construction.

---

## §5 Fix strategy (G4)

Recommended fix paths (priority order, consistent with Wf-263A-Audit §6):

### Path A — Honest-negative close (lowest cost, immediate)

1. Mark `metrics/by_metric/drugood_v3_scaffold.json` with `honest_framing_note` extension: "BELOW_RANDOM caused by LOMO head-collapse + 2-3 epoch training-budget (Wf-253e §1.2.1 structural limitation), NOT v3-specific architectural levers".
2. Add `drugood_v3_scaffold_smoke.json` from `m17_drugood_reeval.py` re-running each metal through `--metals {M} --no-multi-task` (the documented LOMO-collapse fallback per Wf-244 / Wf-253e §1.2.1).
3. Close TODO-130 G5 with `honest_negative` status.

### Path B — Re-train v3 at paper-grade

1. `train_chemprop_v2_v3.py --metals Ru/Ir/Os/Rh/Re --seeds 0/1/2 --epochs 50 --hidden 300 --depth 3 --wass-weight 0.05 --lambda-grl 0.1 --grad-clip 1.0`
2. Wall budget: 50 epochs × h=300/d=3 × 5 metals × 3 seeds ≈ 30-50 GPU-hours on RTX 5090 (Wf-130 protocol).
3. Does NOT fix LOMO collapse — only buys headroom on the non-held-out heads.

### Path C — Structural redesign (long-term, addresses root cause)

1. Replace multi-task + LOMO with **shared-encoder + per-metal linear-probe** training: train encoder once on all 5 metals jointly (no LOMO held-out), then fit per-metal linear probes on top of frozen encoder.
2. Avoids the held-out head's label-0 collapse entirely; allows encoder to learn metal-invariant chemistry (the original Wasserstein-transfer goal).
3. Out of scope for Wf-266-Audit; recommend a follow-up workflow (Wf-267-Design + Wf-268-Implement).

### Path D — Use Wf-185 as paper-grade claim (zero-cost immediate)

Re-label `m17_drugood_v3_reeval` as **DIAGNOSTIC** (not paper-grade) with note that paper-grade m17 OOD-generalisation rests on Wf-185 chemprop v2 retrain at 0.6707 ± 0.0042.

**Recommended immediate action**: Path A + Path D combined (lowest cost, closes TODO-130 G5 honestly without claiming v3 levers caused the regression).

---

## §6 Honest framing note

Per `party-twenty-spirit §3.5`:

- This audit is **READ-ONLY**. No code was modified, no tests were run, no checkpoints re-trained.
- The LOMO head-collapse mechanism is **structural to the multi-task + LOMO design**, not an architectural bug in chemprop v3 or any v3-specific lever. The Wf-263A-Audit already established this; Wf-266-Audit adds the **deterministic label-construction argument** (§2.1) showing the collapse is **mathematically guaranteed** once `train_idx` excludes all rows of the held-out metal.
- The script-level defaults (`--hidden=64`, `--depth=2`, `--epochs=50`) are **honest configuration**, but the operational defaults used for the v3 checkpoints on disk were `--epochs=2-3` (time-budget-driven, not architectural choice). Honest-framing notes should distinguish these.
- The fix is structural (Path C), not flag-flipping. The recommended immediate close (Path A + Path D) is consistent with Wf-253e §1.2.1's acknowledgement of LOMO as a known design limitation.
- Backward compat: all Wf-250-Apply / Wf-252-Apply tests still PASS (96 smoke + 26 Wf-250 + 10 Wf-252, all green per Wf-253e §3.1). No existing test invalidated by this audit.

---

## §7 Cross-references

- Source: `molmetal/scripts/train_chemprop_v2_v3.py`
  - Lines 681-687: per-metal BCE label construction (the all-zero root)
  - Lines 814-817: per-metal BCE loss loop
  - Lines 1104-1122: constant-prediction guard + honest-negative warning
  - Lines 1706-1718: LOMO split construction
  - Lines 1436-1438, 1465-1469: `--quick` smoke mode (epochs=2)
  - Line 210: `DEFAULT_EPOCHS = 50` (script default)
  - Lines 211-212: `DEFAULT_HIDDEN = 64`, `DEFAULT_DEPTH = 2` (script default)
- Wf-263A-Audit: `molmetal/reports/wf263a_audit/v3_drugood_root_cause.md` (prior audit; root-cause established)
- Wf-253-D reeval: `molmetal/reports/wf253d_m17_drugood_reeval/` (forward-pass reeval; high-AUPR + low-ROC-AUC fingerprint)
- Wf-253e signoff: `molmetal/reports/wf253e_signoff/verification.md §1.2.1` (LOMO-collapse already documented as structural limitation)
- Wf-185 chemprop v2 baseline: `metrics/by_metric/drugood_ood_auc.json` (paper-grade 0.6707 ± 0.0042)
- TODO-130 G5: `TODO/completed/130_metric_definition_baseline.md` (m17 row)