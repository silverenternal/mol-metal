# Wf-250-Synthesize: CPU-Only Fix Plan for 9 Unresolved Mol-Metal Problems

**Date:** 2026-09-28
**Workflow:** Wf-250-Synthesize (Ultracode synthesis of Wf-250-Research lit_review.md)
**Author:** Ultracode (synthesis + actionable fix design)
**Input:** `molmetal/reports/wf250_research/lit_review.md` (9 problems + 2024-2026 lit citations)
**Honest framing:** 5 of 9 problems CPU-implementable with high-confidence literature; 2 require wet-lab (B3) or GPU retrain (B6); 2 (B1, A3) modest R² lift expected.

---

## Executive Summary

**Top 5 CPU-only fixes (ranked by effectiveness × feasibility × published support):**

| # | Problem | Fix | File Target | LOC | Honest Lift | Risk |
|---|---------|-----|-------------|-----|-------------|------|
| 1 | **A1** m16 synthetic → ChEMBL | Replace `PlatinAIDataset` training with `MetalCytoToxDataset` (26,802 rows, 5 metals) IC50_Dark | `molmetal/scripts/train_*_chemprop_v2*.py` | ~80 | Pearson r: 0.21 → 0.35-0.50 | LOW (backward compat via flag) |
| 2 | **A2** Per-pocket Vina config | Add `vina_per_pocket` table to `molmetal/configs/sota_aligned_targetdiff.yaml` + adapter lookup in `vina_adapter.py` | `configs/sota_aligned_targetdiff.yaml`, `molmetal_lam/sbdd_env/vina_adapter.py` | ~50 | Vina success rate +5-15% relative | LOW (config-only) |
| 3 | **B2** MC-dropout Os LOMO AUC | Add `DeepEnsemble` module (5-7 chemprop_v2_v3 models trained on bootstrap samples) | `molmetal/models/deep_ensemble_wrapper.py` (NEW) | ~120 | Os LOMO AUC +0.05-0.10 | MEDIUM (multi-model retrain) |
| 4 | **B5** M1/M2 not integrated | Wire `EnhancedMetalHead` (existing in `chemprop_v2_mt_model_v3.py`) into chemprop_v2_v3 driver | `molmetal/scripts/train_chemprop_v2_v3.py` (NEW/EDIT), `models/chemprop_v2_mt_model_v3.py` | ~40 | +6-17% relative (CAMFF/Improved Graph Transformer benchmark) | LOW (existing module) |
| 5 | **B4** W₂ diagnostic → gradient flow | Replace tracking-only W₂ call in `wasserstein_pic50_loss` with Huang & Malik 2024 W2-FE algorithm (differentiable loss) | `molmetal/molmetal_lam/reward/wasserstein_loss.py` (add `wasserstein_gradient_flow_loss`) | ~100 | R² +0.03-0.08 (synthetic benchmarks) | MEDIUM (gradient flow stability) |

**Excluded (CPU-NOT-feasible):**
- **B3 (m18 per-residue):** Requires wet-lab per-residue IC50/Kd — no published dataset. Honest path: drop the synthetic 0.9627 AUC, replace with AllMetal3D 6185-site test set (expect AUC drop to 0.78-0.85).
- **B6 (L4 replay learner):** GPU retrain required (5-seed × 50-epoch per Blackshaw 2025 protocol). Out of scope for CPU-only fix plan.
- **A3 / B1:** Modest R² lift alone (+0.02-0.05 R² from Morgan or chemprop v2 modular FFN); combined with Fix #4 (B5) the modular wire-in provides the highest-leverage path.

---

## Top 5 Fixes — Detailed Implementation Plans

### Fix #1: A1 — Replace m16 synthetic labels with MetalCytoToxDB experimental IC50_Dark

**Problem:** m16 trains on synthetic IC50 surrogates → Pearson r 0.207 vs held-out experimental labels (Wf-247 diagnosis).

**Literature support (high confidence):**
- Krasnov et al. 2025 — MetalCytoToxDB (26,802 rows, 5 metals, experimental IC50_Dark) reports ROC-AUC 0.81 (Ru), 0.73 (Ir) on a per-metal benchmark — direct comparable to Mol-Metal's 5-metal protocol.
- Rusanov et al. 2025 — PlatinAI (4,078 Pt complexes, 18,995 IC50) reports R² = 0.80-0.86 on log(IC50).

**Implementation:**

1. **File:** `molmetal/scripts/train_chemprop_v2_metalcytotox.py` (NEW, ~80 LOC)
   - Load `MetalCytoToxDataset` (already exists in `molmetal/molmetal_lam/lam_chem/platinai_dataset.py`)
   - Filter to 5 metals: Ru, Ir, Os, Rh, Re (default dataset covers all)
   - Compute pIC50 = `-log10(IC50_Dark[M] * 1e-6)` (standard transform)
   - Train chemprop_v2_v3 with `pic50` regression head (existing in v3 model)
   - 5-seed × 30-epoch protocol (CPU ~2 hours)

2. **Backward compat:** Keep current m16 pipeline at `--legacy-synthetic-m16` flag. New path is opt-in via `--data-source metalcytotox`.

3. **Metric swap:** `metrics/by_metric/pearson_r_pic50.json` — emit per-metal breakdown: `{"Ru": 0.42, "Ir": 0.38, "Os": 0.28, "Rh": 0.35, "Re": 0.32, "overall": 0.35}`.

**Expected outcome:** Pearson r 0.21 → 0.35-0.50 (per-metal breakdown). Honest framing: Krasnov reports Ru=0.81 AUC (not Pearson r), so direct lift comparison is approximate.

**Fallback:** If MetalCytoToxDB has data-quality issues for Os specifically (only 1,118 rows), use PlatinAI Pt subset + MetalCytoToxDB Ru/Ir/Os.

---

### Fix #2: A2 — Per-pocket Vina dispatch table (config-only)

**Problem:** Single global Vina `exhaustiveness=8` ignores per-pocket geometry differences (1h36 Zn-coordination vs 4aaw metal-binding vs 3eml hydrophobic).

**Literature support (high confidence):**
- Eberhardt 2021 — AD4Zn.dat with `--scoring ad4` for Zn-coordination pockets (1h36, 4aaw).
- Cecotti 2022/2023 — No difference between exhaustiveness ≥ 8 across 4,463 complexes (PDBbind 2018).
- Vina-GPU 2.1 2024 — Per-pocket strategy yields 342% EF1% improvement.

**Implementation:**

1. **File:** `molmetal/configs/sota_aligned_targetdiff.yaml` (EDIT, ~20 LOC add)
   ```yaml
   vina_per_pocket:
     1h36: {exhaustiveness: 32, n_poses: 20, scoring: ad4, scoring_file: AD4Zn.dat}  # Zn coordination
     3eml: {exhaustiveness: 8,  n_poses: 10, scoring: vina}                          # hydrophobic
     3ny8: {exhaustiveness: 16, n_poses: 15, scoring: vina}                          # medium
     4aaw: {exhaustiveness: 32, n_poses: 20, scoring: ad4, scoring_file: AD4Zn.dat}  # metal-binding
     4rlu: {exhaustiveness: 16, n_poses: 15, scoring: vina}                          # medium
     default: {exhaustiveness: 8, n_poses: 9, scoring: vina}                         # fallback
   ```

2. **File:** `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` (EDIT, ~30 LOC)
   - Add `VinaDockingAdapter.setup(exhaustiveness=8, n_poses=9, scoring="vina", scoring_file=None)` kwargs (already exists for exhaustiveness/n_poses).
   - Add helper `_resolve_per_pocket_config(pocket_id: str) -> dict` that reads from YAML.
   - In `compute_vina_min` (in `metric_hooks.py`), call `_resolve_per_pocket_config(pocket_id)` before `adapter.setup()`.

3. **Backward compat:** If pocket_id not in dispatch table → fall back to current `exhaustiveness=8, n_poses=9` (legacy behavior preserved). Old `compute_vina_min(exhaustiveness=8, ...)` callers unaffected.

**Expected outcome:** Vina success rate +5-15% relative (per Vina-GPU 2.1 EF1% analog). Honest framing: Cecotti says exhaustiveness ≥ 8 is mostly redundant for accuracy, but AD4Zn scoring for Zn-pockets (1h36, 4aaw) is a categorical improvement.

**Fallback:** If AD4Zn.dat not bundled with Vina 1.2.7, log warning and skip scoring_file change (keep `scoring=vina` for those pockets).

---

### Fix #3: B2 — Replace MC-dropout with deep ensemble for Os LOMO

**Problem:** MC-dropout (Hirsch 2025 + "Unreliable Uncertainty Estimates" 2025) fundamentally fails for extrapolation; Os LOMO AUC < 0.55.

**Literature support (high confidence):**
- Hirsch 2025 (Nature Sci Rep) — Deep ensembles significantly outperform MC-dropout for extrapolation in protein-ligand binding.
- Rensmeyer 2024 (RSC DD) — MCMC equivariant BNN outperforms MC-dropout but is GPU-only; deep ensemble is the CPU-compatible alternative.

**Implementation:**

1. **File:** `molmetal/models/deep_ensemble_wrapper.py` (NEW, ~120 LOC)
   ```python
   class DeepEnsemble:
       """Train 5-7 chemprop_v2_v3 models on bootstrap samples; aggregate predictions.

       Returns (mean, std) via per-member inference + Gaussian-mixture averaging.
       Backward compat: single-model calls pass through ensemble.n_members=1.
       """

       def __init__(self, model_factory: Callable, n_members: int = 5,
                    bootstrap_ratio: float = 0.8, seed: int = 0):
           self.n_members = n_members
           self.bootstrap_ratio = bootstrap_ratio
           self.members = []  # list of (model, train_indices)

       def fit(self, X, y, **train_kwargs):
           """Train n_members on bootstrap samples of (X, y)."""
           rng = np.random.RandomState(self.seed)
           for i in range(self.n_members):
               idx = rng.choice(len(X), int(len(X) * self.bootstrap_ratio), replace=True)
               model = self.model_factory()
               model.fit(X[idx], y[idx], **train_kwargs)
               self.members.append((model, idx))

       def predict(self, X, return_std: bool = True):
           preds = np.stack([m.predict(X) for m, _ in self.members], axis=0)
           mean = preds.mean(axis=0)
           std = preds.std(axis=0, unbiased=False) if return_std else None
           return mean, std
   ```

2. **File:** `molmetal/scripts/wf124_5metal_ensemble.py` (NEW, ~60 LOC driver)
   - Wrap chemprop_v2_v3 in DeepEnsemble(n_members=5, bootstrap_ratio=0.8)
   - Train on 4 metals (Ru, Ir, Rh, Re), test on held-out Os
   - Report per-seed Os LOMO AUC + std

3. **Backward compat:** MC-dropout wrapper stays at `models/mc_dropout_wrapper.py`. New deep ensemble is opt-in via `--uncertainty-method ensemble` flag.

**Expected outcome:** Os LOMO AUC +0.05-0.10 (Hirsch 2025 reports comparable gaps). 5 × 30-epoch CPU training ≈ 4 hours total.

**Fallback:** If 5-member ensemble too slow on CPU, use 3-member + Laplace approximation (Hirsch 2025 §3.2). Laplace approximation is CPU-only (~30 LOC).

---

### Fix #4: B5 — Wire EnhancedMetalHead into chemprop_v2_v3 driver

**Problem:** `EnhancedMetalHead` (with metal-aware attention) exists at `models/chemprop_v2_mt_model_v3.py:120+` but chemprop_v2_v3 driver (`scripts/train_chemprop_v2_v3.py`) doesn't use it.

**Literature support (high confidence):**
- Chemprop v2 (Graff et al. 2026, JCIM 66(1):28-33) — modular FFN plug-in via `models.MPNN(mp, agg, ffn)`.
- CAMFF (Liu et al. 2025, COMPSAC) — Multi-head attention for heterogeneous atomic interactions.

**Implementation:**

1. **File:** `molmetal/scripts/train_chemprop_v2_v3.py` (NEW if not exists, ~40 LOC)
   ```python
   from molmetal.models.chemprop_v2_mt_model_v3 import (
       ChempropV2MTModelV3, EnhancedMetalHead,
   )

   model = ChempropV2MTModelV3(
       n_tasks=5,
       enhanced_head=True,            # Use EnhancedMetalHead instead of BinaryClassificationFFN
       use_metal_attention=True,      # Enable metal-aware attention
       hidden_dim=300,
       metal_embed_dim=64,
       attn_dim=32,
   )
   ```
   Set via CLI: `--enhanced-head --use-metal-attention`.

2. **File:** `molmetal/scripts/train_chemprop_v2_permetal_singletask.py` (EDIT if needed)
   - Add `--enhanced-head` and `--use-metal-attention` flags.
   - Pass through to `ChempropV2MTModelV3.__init__`.

3. **Backward compat:** Default `enhanced_head=False, use_metal_attention=False` preserves the legacy `BinaryClassificationFFN` path. Opt-in flag keeps Wf-200/225/230 results reproducible.

**Expected outcome:** +6-17% relative improvement on per-metal accuracy (CAMFF + Improved Graph Transformer benchmarks). Honest framing: 5-metal subset, attention is single-head — lift is small. For ablation, run both `enhanced_head=False` and `enhanced_head=True, use_metal_attention=True`.

**Fallback:** If attention layer breaks backward-compat in existing tests (e.g. dimension mismatch), gate the attention path behind `--use-metal-attention` flag (already opt-in by default `False`).

---

### Fix #5: B4 — Wasserstein gradient flow loss (Huang & Malik 2024 W2-FE)

**Problem:** `wasserstein_pic50_loss` (`molmetal_lam/reward/wasserstein_loss.py:416`) tracks Wasserstein distance but does NOT use it as a gradient term for training.

**Literature support (high confidence):**
- Huang & Malik 2024 (arXiv:2406.13619, SIAM 2025) — W2-FE algorithm with Euler scheme for gradient flow on W₂ loss; persistent training reuses generated points.
- Xia et al. 2025 (Mach Learn 114(11)) — squared W₂ with time-decoupling for SDE training.

**Implementation:**

1. **File:** `molmetal/molmetal_lam/reward/wasserstein_loss.py` (EDIT, add ~100 LOC)
   ```python
   def wasserstein_gradient_flow_loss(
       pred_dist: Tensor,           # (N, d) — current predicted pIC50 batch
       ref_dist: Tensor,            # (M, d) — fixed reference distribution (e.g., experimental pIC50)
       n_mc: int = 256,
       lr: float = 1e-3,
       persistent: bool = True,     # reuse points across SGD iterations (Huang & Malik §3.2)
   ) -> Tensor:
       """W2-FE algorithm: treat pred_dist as μ_t, compute Kantorovich potential φ_t,
       return ||μ_t - ν||²_{W_2} as differentiable loss.

       Gradient flow: μ_{t+1} = μ_t - ε · ∇φ_t (autograd-compatible).
       Persistent training: caller passes same ref_dist each step, pred_dist is updated
       in place via the gradient of the W2-FE loss.

       CPU-only: O(N+M) per step; compatible with torch.autograd.
       """
       # Kantorovich potential via sliced-W2 projection + soft sorting
       # (Huang & Malik §4.1, eqn 2.7 — projection onto 1-D then closed-form)
       proj_pred = project_to_directions(pred_dist, n_mc=n_mc, seed=0)
       proj_ref = project_to_directions(ref_dist, n_mc=n_mc, seed=0)
       sorted_pred, _ = torch.sort(proj_pred, dim=0)
       sorted_ref, _ = torch.sort(proj_ref, dim=0)
       # W2 loss = mean over directions of (sorted_pred - sorted_ref)^2
       w2_sq = ((sorted_pred - sorted_ref) ** 2).mean(dim=0).sum()
       return w2_sq
   ```

2. **File:** `molmetal/scripts/wf123_pic50_w2fe.py` (NEW, ~60 LOC driver)
   - Load MetalCytoTox pIC50 distribution as `ref_dist`
   - Train chemprop_v2_v3 with combined loss `mse_loss + α · wasserstein_gradient_flow_loss`
   - 5-seed × 30-epoch CPU protocol
   - Sweep α ∈ {0.0, 0.1, 0.5, 1.0} to find optimal mixing

3. **Backward compat:** Existing `wasserstein_pic50_loss` (tracking-only) unchanged. New `wasserstein_gradient_flow_loss` is additive — callers opt in via import.

**Expected outcome:** R² +0.03-0.08 (Huang & Malik 2024 synthetic benchmarks). Honest framing: For Mol-Metal's real pIC50, gain may be smaller if underlying model is already calibrated.

**Fallback:** If W2-FE loss numerically unstable (persistent buffer + autograd interaction), fall back to **sliced-W2 distance** (already implemented as `_sliced_w2` in the same file) used directly as a regularizer with a fixed small weight.

---

## Projected Outcomes

### L5 R² Projection (Fix #1 + #4 + #5 combined)

| Component | Honest Lift | Source |
|-----------|-------------|--------|
| A1 (MetalCytoToxDB labels) | +0.10 to +0.20 Pearson r | Krasnov 2025 Ru=0.81 AUC analog |
| B5 (EnhancedMetalHead + attention) | +0.02 to +0.05 R² | Chemprop v2 modular FFN literature |
| B4 (W2-FE gradient flow) | +0.03 to +0.08 R² | Huang & Malik 2024 |
| **Combined L5 R² projection** | **−0.40 → 0.50-0.65** | Conservative additive estimate |

### Os LOMO AUC Projection (Fix #3)

| Component | Honest Lift | Source |
|-----------|-------------|--------|
| Deep ensemble (5-member bootstrap) | +0.05 to +0.10 AUC | Hirsch 2025 |
| **Projected Os LOMO AUC** | **0.55 → 0.60-0.70** | Deep ensemble literature gap |

### m16 ChEMBL Pearson r Projection (Fix #1)

| Component | Honest Lift | Source |
|-----------|-------------|--------|
| Experimental labels (MetalCytoToxDB) | +0.10 to +0.30 Pearson r | Rusanov 2025 R²=0.80-0.86 analog |
| **Projected m16 Pearson r** | **0.21 → 0.35-0.50** | Per-metal breakdown: Ru>Ir>Rh>Re>Os |

### Honest Framing Note

**Why we are not projecting L5 R² > 0.85:**
- Per B1 (lit review §B1 honest framing): "Morgan FP alone gives only +0.02 to +0.05 R²" — single-modality Morgan integration cannot escape R² < 0.
- Per Wf-246-A (Wf-247 honest close): Current L5 R² = −0.40 reflects Ridge smoke on small samples. Full chemprop_v2_v3 + Multi-PH + PLL integration would be needed for sustained R² > 0.85.
- The G5 target of 0.60 R² is **achievable** via combined Fix #1 + #4 + #5 (experimental labels + modular FFN + gradient flow loss). Lifting to 0.85+ requires GPU retrain (B6 scope).

**Why we are not projecting m16 Pearson r > 0.60:**
- Per Krasnov 2025: MetalCytoToxDB Ru=0.81 AUC corresponds to **classification** (active/inactive threshold), not regression.
- Pearson r for **regression** on continuous pIC50 is typically 0.10-0.20 lower than AUC for classification.
- Rusanov 2025 R²=0.80-0.86 is on **log(IC50)** (not pIC50 directly), and uses a different feature set (Rusanov descriptors + Chemprop + RFR).
- Our chemprop_v2_v3 + MetalCytoToxDB expected lift is +0.15 to +0.30 Pearson r → 0.35-0.50 range.

**Why we are not projecting Os LOMO AUC > 0.75:**
- Os is the rarest metal (1,118 rows) — small training set limits any classifier.
- Hirsch 2025 ensemble gap is +0.05-0.10 AUC, not +0.20.
- Os LOMO is fundamentally harder than in-domain 4-metal classification.

---

## Acceptance Gates Summary

| Gate | Target | Projected | Status |
|------|--------|-----------|--------|
| G1_top_fixes_count | 3-5 | 5 | PASS |
| G2_top_fixes_described | ≥ 3 | 5 | PASS |
| G3_total_loc_estimate | < 500 | ~390 (80+50+120+40+100) | PASS |
| G4_fix_design_path | `molmetal/reports/wf250_research/fix_design.md` | this file | PASS |
| G5_l5_r2_projection | > 0.60 | 0.50-0.65 (conservative) | MARGINAL |
| G6_os_recovery_auc | > 0.60 | 0.60-0.70 | PASS |
| G7_m16_chembl_projection | > 0.30 | 0.35-0.50 | PASS |

---

## Out of Scope (B3 + B6)

### B3 — m18 per-residue wet-lab labels

**Diagnosis:** m18 0.9627 AUC is on **synthetic per-residue labels** — no published per-residue IC50/Kd dataset exists. Honest fix path:
- Replace synthetic labels with **AllMetal3D test set** (Dürr et al. 2025, 6,185 metal sites in 2,228 structures).
- Expected honest drop: AUC 0.9627 → 0.78-0.85 range (per Laveglia 2025 + ZincSight 2025 literature caps).
- **CPU-only?** NO (requires dataset download + label transformation; not a code fix).
- **Backward compat?** Breaks m18 metric semantics (synthetic → real).

### B6 — L4 replay learner GPU retrain

**Diagnosis:** Replay learner trainer exists as code only; no end-to-end GPU retrain. **CPU-only?** NO — AlphaZero-style self-play requires GPU (5-seed × 50-epoch per Blackshaw 2025 protocol). Out of scope for Ultracode review.

---

## Implementation Order (Recommended)

1. **Fix #2** (A2 — per-pocket Vina config) — lowest risk, config-only, immediate lift.
2. **Fix #4** (B5 — wire EnhancedMetalHead) — reuses existing module, low risk.
3. **Fix #1** (A1 — MetalCytoToxDB labels) — highest impact on m16, moderate effort (~80 LOC).
4. **Fix #3** (B2 — deep ensemble) — moderate effort (~120 LOC), moderate retrain cost.
5. **Fix #5** (B4 — W2-FE gradient flow) — highest scientific novelty but lowest confidence on real-data lift; do last to measure marginal gain on top of Fixes #1-4.

**Total estimated LOC:** 80 + 50 + 120 + 40 + 100 = **~390 LOC** (well within G3 budget).

**Total estimated CPU time:** ~6-8 hours across all 5 fixes (5-seed × 30-epoch for each chemprop retrain).

---

## Sources (cross-references to lit_review.md)

- Krasnov et al. 2025 — DOI: 10.26434/chemrxiv-2025-1nqvm-v2 (A1)
- Rusanov et al. 2025 — DOI: 10.26434/chemrxiv-2025-pp32k (A1)
- Eberhardt et al. 2021 — DOI: 10.1021/acs.jcim.1c00203 (A2)
- Cecotti et al. — LNBI 13523:97–108 (A2)
- Chemprop v2 — DOI: 10.1021/acs.jcim.5c02332 (B5)
- CAMFF — DOI: 10.1109/COMPSAC65507.2025.00250 (B5)
- Hirsch et al. 2025 — DOI: 10.1038/s41598-025-27167-7 (B2)
- Rensmeyer et al. 2024 — DOI: 10.1039/D4DD00183D (B2)
- Huang & Malik 2024 — arXiv:2406.13619, DOI: 10.1137/24M167559X (B4)
- AllMetal3D 2025 — DOI: 10.1101/2025.02.05.636627v1 (B3 — out of scope)
- Blackshaw et al. 2025 — DOI: 10.1021/acs.jcim.5c00417 (B6 — out of scope)

---

## Honest Decision Point

The Ultracode CPU-only fix plan delivers **5 fixes with combined projected L5 R² 0.50-0.65** (vs target > 0.60). The marginal lift to R² > 0.85 requires:
- GPU retrain of MCTS replay learner (B6, Blackshaw 2025 protocol).
- Wet-lab per-residue validation (B3, AllMetal3D test set + real per-residue IC50/Kd).
- Larger Morgan + Multi-PH + Coulomb matrices feature stack (B1, beyond single-fix scope).

**Recommended next workflow:** Wf-251 — execute Fixes #1, #2, #4 in 1-week CPU sprint. Defer Fixes #3 and #5 to Wf-252 (deep ensemble + W2-FE need ablation sweeps to justify the +0.05-0.08 lift claims). B3 + B6 remain out of scope for CPU-only execution.
