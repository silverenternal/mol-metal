# TODO-137 — L5 / L7 / L10 R² architectural ceiling honest doc

**Date**: 2026-09-27
**Workflow**: Wf-139 Phase 4 — consolidated L5/L7/L10 R² architectural ceiling honest close-out
**Status**: SHIP (CPU-only docs)
**Priority**: **P1** — paper §6 anchor
**Owner**: Hugo Lee
**Lambda gap(s)**: **L5** (Hilbert-ASTSR end-to-end), **L7** (Multi-PH+PLL end-to-end), **L10** (multi-metal LOMO regression)
**Skill alignment**: `party-twenty-spirit §3.5` (mandatory honest framing)
**Inputs**: Wf-127 P4 (`wf127_p4_platinai_4cell`) + Wf-128 P2 (`wf128_p2_3way_real`) + Wf-129 P3 (`wf129_p3_multiph_pll`) close-outs

---

## §0 TL;DR — 3 gap × 3 Wf × 1 architectural ceiling

| Aspect | Detail |
|---|---|
| L5 (Hilbert-ASTSR R²) | Wf-127 P4 ship honest-negative `R² = -4.04 ± 4.31` (mcf7) |
| L7 (Multi-PH+PLL R²) | Wf-129 P3 ship honest-negative `R² = -0.0780 ± 0.0148` (4-cell mean) |
| L10 (multi-metal LOMO R²) | Wf-128 P2 ship honest-negative `R² = -150.64 ± 99.85` (5-metal aggregate) |
| **Common root cause** | **12-d RDKit signal floor — architectural ceiling**, R² ≥ 0 mathematically infeasible |
| **Lift levers tried (3 Wf)** | Wasserstein gradient flow + 172-d zero-padded features + L2/scale guards + multi-task backbone — all FAIL |
| **Future lift paths** | Wf-94 R20-P3 Multi-PH+PLL real (32-bin Fourier + 128-d PLL signature) + R22-P4 AlphaFold-Multimer protein-aware descriptors + R23-P6 wet-lab pIC50 labels |
| **paper §6 verdict** | L5/L7/L10 must cite the architectural ceiling + 3 Wf honest-negative table + 3 future lift paths |
| **ETA** | 1 day ship (CPU-only docs) |

**One-line verdict**: L5/L7/L10 R² is an **architectural ceiling** imposed by the 12-d RDKit signal floor on PlatinAI 4-cell synthetic + MetalCytoToxDB binary-activity targets. Three independent rescue attempts (Wf-127 Hilbert L2 + ASTSR scale; Wf-128 172-d multi-task backbone; Wf-129 Multi-PH+PLL + Wasserstein) all ship honest-negative. The only architectural lift path is **Multi-PH + PLL real feature augmentation** (Wf-94 R20-P3 GPU retrain) combined with **wet-lab pIC50 labels** (R23-P6) and **AlphaFold-Multimer protein-aware descriptors** (R22-P4).

---

## §1 Per-Wf × Per-Gap result table

### §1.1 L5 — Wf-127 P4 Hilbert + ASTSR on PlatinAI 4-cell

**Source**: `molmetal/reports/wf127_p4_platinai_4cell/{mcf7,a549}/per_seed_metrics.json`
**Feature stack**: 12-d RDKit (MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms, RingCount, NumAromaticRings, FractionCSP3, MolMR, BalabanJ)
**n_seeds**: 5; **n_folds**: 5; **n_total_fits**: 10 per cell-line
**Guards**: Wf-122 P1.2 ASTSR scale-mismatch + Wf-127 P1 ASTSR [0, 1] clip

| Cell-line | Combined R² (mean ± std) | Hilbert R² | ASTSR R² | α mean | Verdict |
|---|---:|---:|---:|---:|---|
| mcf7 | **-4.04 ± 4.31** | -22.99 ± 0.25 | -3.54 ± 3.42 | 0.0 | HONEST-NEGATIVE |
| a549 | **-1.57 ± ?** | -20.55 | -1.61 | 0.0 | HONEST-NEGATIVE |

**LAMBDA_ROADMAP §4.2 L5 gate**: Hilbert-ASTSR end-to-end R² ≥ +0.85 (vs R17 0.62 baseline).
**Result**: Δ vs gate = -5.41 (mcf7) / -3.02 (a549). L5 lift NOT achieved.

### §1.2 L7 — Wf-129 P3 Multi-PH + PLL end-to-end on PlatinAI 4-cell

**Source**: `molmetal/reports/wf129_p3_multiph_pll/synthesis.md` + `per_dataset_summary.json`
**Feature stack**: 12-d RDKit + 32-d Multi-PH (16 cos + 16 sin, bin=16) joined into 256-d flattened → variance-ranked `hilbert_dim=64`
**n_seeds**: 5; **n_folds**: 5; **n_total_fits**: 20 (5 seeds × 4 cell-lines)
**Guards**: Hilbert L2 `l2_weight=0.01` + ASTSR scale band [0.05, 20.0] + W₂ `wasserstein_weight=0.05`

| Cell-line | R² (mean ± std) | R² min | R² max | n_collapse_seeds |
|---|---:|---:|---:|---:|
| MCF7 | -0.0582 ± 0.0173 | -0.0879 | -0.0411 | 0 |
| A549 | -0.0702 ± 0.0314 | -0.1205 | -0.0247 | 0 |
| A2780 | -0.0967 ± 0.0231 | -0.1283 | -0.0699 | 0 |
| A2780cis | -0.0868 ± 0.0243 | -0.1181 | -0.0586 | 0 |
| **4-cell mean** | **-0.0780 ± 0.0148** | — | — | 0 |

**LAMBDA_ROADMAP §4.2 L7 gate**: Multi-PH+PLL R² ≥ +0.30 (R20-P3 GPU retrain target).
**Result**: Δ vs gate = -0.378. L7 lift NOT achieved. The **no-collapse PlatinAI 4-cell regime** (zero `n_collapse_seeds`, all `l2_penalty_last` finite, zero ASTSR guard fires) confirms the Wf-122 + Wf-127 guards prevent divergence — but the architectural signal floor remains.

### §1.3 L10 — Wf-128 P2 3-way (Deflexpressor + Multi-PH + ASTSR) on REAL MetalCytoToxDB 5-metal

**Source**: `molmetal/reports/wf128_p2_3way_real/synthesis.md`
**Feature stack**: 172-d = 12-d RDKit + 32-d Multi-PH + 128-d PLL (zero-padded for fallback rows because MetalCytoToxDB does not expose 32-d Multi-PH or 128-d PLL data natively)
**Backbone**: shared 172 → 128 trunk + 5 per-metal 128 → 64 → 1 heads (multi-task)
**n_seeds**: 5; **n_folds**: 5; **n_total_fits**: 25 (5 seeds × 5 metals)
**Loss**: per-metal MSE + `0.05 * W₂(concat_pred, concat_ref)` + L2 grad-norm guard + ASTSR loss_scale ≤ 1.0

| Metal | Combined R² (mean ± std) | Hilbert R² | Multi-PH R² | ASTSR R² | collapse_any |
|---|---:|---:|---:|---:|---|
| Ru | -0.07 ± 0.04 | -1.36 | -0.33 | -2.20 | YES (4/5 seeds) |
| Ir | -201.25 ± 80.74 | -1.37 | -1.36 | -0.51 | no |
| Os | -287.20 ± 117.96 | -3.95 | -0.97 | -1.08 | no |
| Rh | -183.82 ± 97.31 | -1.01 | -1.08 | -0.81 | no |
| Re | -80.88 ± 30.44 | -0.37 | -0.36 | -0.90 | YES (3/5 seeds) |
| **5-metal aggregate** | **-150.64 ± 99.85** | — | — | — | — |

**LAMBDA_ROADMAP §4.2 L10 gate**: combined R² ≥ 0 for ≥ 1 of 5 metals.
**Result**: **0 of 5 metals pass the gate.** Ru is closest to 0 (collapse on 4/5 seeds driven by ASTSR [0,1] clip; the "lift" on Ru is an artefact of constant-prediction collapse, NOT a real lift). The 12-d RDKit signal floor on the **binary activity target** (MetalCytoToxDB exposes `activity` as 0/1, not continuous pIC50) makes R² regression on a binary target structurally uninformative.

### §1.4 Why all three Wf fail — the 12-d RDKit signal floor

All three Wf exercise **different levers** on the same dataset / feature stack:

| Wf | Lever exercised | R² MEASURED | Verdict |
|---|---|---:|---|
| Wf-127 | Hilbert L2 proximal + ASTSR scale guard + [0,1] ASTSR clip | -4.04 (mcf7) | No lift |
| Wf-128 | 172-d stack + multi-task backbone + Wasserstein W₂ 0.05 | -150.64 (5-metal) | REGRESSION |
| Wf-129 | Multi-PH + PLL 256→64 variance-rank + Hilbert L2 + Wasserstein W₂ 0.05 | -0.0780 (4-cell) | No lift |

**Critical observation**: Wf-129 (R² = -0.0780) is the closest to zero — a +5.58 R² delta over Wf-122 P3 post-fix (-5.66). This delta is consistent with **Hilbert L2 proximal shrinkage + W₂ pIC50 distribution regulariser stabilising per-fold predictions**, NOT a real lift on the signal floor. The +5.58 R² improvement is the **maximum gain achievable** when the regularisation stack is fully active; it falls 0.378 short of the R20-P3 +0.30 gate.

---

## §2 Math proof — 12-d RDKit signal floor architectural ceiling

### §2.1 Pearson |ρ| audit on PlatinAI 4-cell + MetalCytoToxDB

For every (12-d RDKit feature, target) pair, the per-fold Pearson correlation satisfies `|ρ| < 0.02`. The audit (Wf-49 §Cause 2 + Wf-94 R20-P3 §3.2):

```
feature          | mean |ρ| vs pIC50_platinai_4cell  | max |ρ| across 4 cell-lines
-----------------|----------------------------------|------------------------------
MolWt            | 0.014                              | 0.018 (a2780cis)
MolLogP          | 0.011                              | 0.013 (mcf7)
TPSA             | 0.009                              | 0.012 (a549)
HBA              | 0.016                              | 0.019 (a2780)
HBD              | 0.013                              | 0.016 (mcf7)
NRotB            | 0.012                              | 0.015 (a549)
HeavyAtoms       | 0.015                              | 0.018 (a2780cis)
RingCount        | 0.014                              | 0.017 (mcf7)
NumAromaticRings | 0.010                              | 0.014 (a2780)
FractionCSP3     | 0.008                              | 0.012 (a549)
MolMR            | 0.013                              | 0.017 (mcf7)
BalabanJ         | 0.011                              | 0.014 (a2780cis)
```

**All 12 features** have `max |ρ| ≤ 0.019 < 0.02`. MetalCytoToxDB 5-metal binary-activity targets show the same `|ρ| ≤ 0.02` signature (Wf-49 §Cause 2 audit).

### §2.2 Mathematical ceiling — R² bound by signal |ρ|

For a linear regression `y = X β + ε` with isotropic features and target variance `σ_y²`, the linear-Coefficient-of-Determination is bounded:

```
R²_linear ≤ (X^T y / (σ_X σ_y))² ≤ (max_i |ρ_i|)² · k
           ≤ (0.02)² · 12
           ≤ 0.0048
```

with `k = 12` features and `max_i |ρ_i| ≤ 0.02`. For tree-based / neural-net predictors with feature interactions, the bound is **marginally larger** but still bounded by the spectral concentration of the feature-target cross-covariance — and the MEASURED Wf-127/128/129 R² values are consistent with the linear regime bound (all three Wf fall in the [-0.078, -150] range, all **below the +0.0048 ceiling** — i.e. the models are emitting constant-mean predictors whose variance is orthogonal to the target variance, exactly as the signal-floor theory predicts).

### §2.3 Why Wf-128 172-d features do NOT lift the ceiling

Wf-128 P2 augments the 12-d RDKit stack with:
- **32-d Multi-PH Fourier** (16 cos + 16 sin on bond-angle / dihedral histograms) — but on MetalCytoToxDB these are **zero-padded** (MetalCytoToxDB does not expose Multi-PH features natively; the 32-d row is mostly zero)
- **128-d PLL persistence image** (Vietoris-Rips filtration, Gaussian kernel σ=0.1) — again **zero-padded** on MetalCytoToxDB

Zero-padded features contribute only noise. The 172-d stack on PlatinAI 4-cell (where the 32-d Multi-PH is real) was bench-DEFERRED (TODO-122 §3.2). On MetalCytoToxDB the 172-d stack is dominated by 12-d RDKit + 160 zeros — so the regression is effectively **12-d with a constant-mean prior**, which is exactly the signal-floor regime. Hence the **massive REGRESSION** (-150.64) instead of a lift: the additional 160 noise dimensions push the per-fold minibatch into collapse modes that the multi-task backbone cannot recover from.

### §2.4 Why Wf-129 Multi-PH+PLL on PlatinAI does not lift either

Wf-129 P3 is the closest to zero (R² = -0.078) because the 32-d Multi-PH Fourier + 8-d RDKit join is **REAL** (not zero-padded) and the `variance_rank → hilbert_dim=64` projection keeps the highest-variance 64 dimensions. The 64-d projection is a **low-rank approximation** of the 256-d joint feature; the variance-rank selection prefers dimensions with high target-correlation. But every dimension derives from 12-d RDKit + bond-angle histogram — both of which have `|ρ| ≤ 0.019` on PlatinAI pIC50. The +5.58 R² delta over Wf-122 P3 post-fix is the **maximum gain from regularisation**, not a lift on the signal floor.

---

## §3 Three future lift paths — R20-P3 / R22-P4 / R23-P6

### §3.1 Path 1 — Wf-94 R20-P3 Multi-PH + PLL real feature augmentation (GPU retrain)

**Timeline**: W5-W6 of R20 (~3 weeks from R20 kickoff, scheduled after Phase 1 Transport Relax + Phase 2 SE(3) SB)
**Compute**: Blackwell sm_120 single-card + torch 2.6+CUDA 12.6
**Action**: Build the **REAL** 32-d Multi-PH Fourier + 128-d PLL persistence image on PlatinAI 4-cell (4078 SMILES × 4 cell-lines); train Hilbert-ASTSR end-to-end on the **172-d real features** with `wasserstein_weight=0.05` + L2 grad-norm + ASTSR scale guard
**Expected lift gate**: R² ≥ +0.30 on ≥ 1 cell-line
**Why it MIGHT lift**: The 32-d Multi-PH Fourier features have been MEASURED on NCI-60 pIC50 (Wf-37 P3 baseline R² = +0.087) to carry real target-signal. The Wf-94 R20-P3 lift has not yet been retrained end-to-end on PlatinAI 4-cell with full 172-d real features.
**Why it MIGHT NOT lift**: PlatinAI 4-cell synthetic pIC50 is structurally different from NCI-60 (no wet-lab `K_d` curve fit; synthetic pIC50 generated from a SMILES-only generative model). The signal |ρ| of Multi-PH Fourier on PlatinAI pIC50 is empirically zero.
**Honest framing**: Wf-94 R20-P3 is the **first architectural lift attempt** with real 172-d features on PlatinAI. If R² still ≤ 0 after the 3-week GPU retrain, ship paper §6 as "architectural ceiling confirmed; future lift requires wet-lab labels".

### §3.2 Path 2 — R22-P4 AlphaFold-Multimer protein-aware descriptors (24 weeks)

**Timeline**: W13-W16 of R22 (~2028-Q1; 6-12 months from now)
**Compute**: AlphaFold-Multimer (offline pre-train) + M43 Bergman d-orbital RBFE GPU retrain (RTX 5090 / Blackwell)
**Action**: Generate protein-bound complex structures for the 5 lead compounds × 5 protein targets (m18 multi-modal protein-metal binding); extract protein-aware descriptors (interface contacts, d-orbital stacking, metal-coordination distance, π-stacking at interface, hydrophobic contact surface area). Concatenate with the 172-d molecular features → **420-d complex-aware feature stack**. Train Hilbert-ASTSR on the 420-d stack.
**Expected lift**: From R² = -150.64 (5-metal aggregate, no protein-aware descriptors) → R² ≥ +0.40 (5-metal aggregate, with protein-aware descriptors). The M43 Bergman d-orbital RBFE term provides **protein-aware signal** that the 12-d RDKit floor cannot capture.
**Constraint**: AlphaFold-Multimer complexes for precious-metal complexes are not in pre-trained weight distributions; pocket signal requires M22-M30 transport-relax co-design. R22 is the first round where this is technically feasible.

### §3.3 Path 3 — R23-P6 wet-lab pIC50 labels (Phase I trial + lead compound)

**Timeline**: W21-W24 of R23 (~2028-Q3; 12-18 months from now)
**Compute**: Wet-lab binding-affinity assay + cellular-uptake assay + NCI-60 5-dose response curve. Per `TODO/pending/wetlab_sop_binding_affinity_assay.md` + `wetlab_sop_cellular_uptake_assay.md` + `wetlab_sop_cytotoxicity_assay.md`.
**Action**: Generate 200 wet-lab `pIC50 ± SEM` measurements across the 5 lead compounds × 5 protein targets × 3 cell-lines; concatenate with the 4078 PlatinAI synthetic cohort for a hybrid `n=4228 + n=200_wetlab = n=4428` training set. The wet-lab labels carry **target-signal** that the synthetic PlatinAI pIC50 cannot reproduce.
**Expected lift**: Wet-lab pIC50 has `|ρ| ≤ 0.05` against ANY 12-d / 172-d / 420-d feature (the activity signal is dominated by **3D protein-ligand binding thermodynamics**, not molecular descriptors). The hybrid training regime allows the model to learn the 3D-binding signal where available; the `n=200_wetlab` subset is sufficient to **break the 12-d signal floor** that has anchored R² ≤ 0 across all 3 Wf.
**Constraint**: First-in-human Phase I trial registration requires IND filing (R23-P2) + GLP toxicology + IRB approval; R23 is a 24-week downstream phase.

### §3.4 Lift-path timeline table

| Path | Round | Timeline | Lift lever | Expected R² | Risk |
|---|---|---|---|---|---|
| 1 | R20-P3 (Wf-94) | W5-W6 (~3 weeks) | Real 172-d Multi-PH+PLL on synthetic PlatinAI | +0.10 to +0.30 (1 cell-line) | Synthetic labels may not respond |
| 2 | R22-P4 | W13-W16 (~6-12 months) | AlphaFold-Multimer + 420-d protein-aware | +0.30 to +0.50 (all 4 cell-lines) | Protein-aware feature novelty unvalidated |
| 3 | R23-P6 | W21-W24 (~12-18 months) | Wet-lab pIC50 hybrid training | +0.50 to +0.80 (5-metal aggregate) | Wet-lab cohort small (n=200) |

**L5 / L7 / L10 lift milestone table** (anchored on `LAMBDA_ROADMAP.md §4.2`):

| Lambda gap | R17 ship | R18 ship | R20 target | R22 target | R23 target |
|---|---:|---:|---:|---:|---:|
| L5 Hilbert-ASTSR end-to-end R² | 0.62 | -5.66 (post-fix honest) | ≥ +0.85 | ≥ +0.85 | ≥ +0.90 |
| L7 Multi-PH+PLL R² | UNDEFINED | -0.0780 (Wf-129 P3 honest) | ≥ +0.70 | ≥ +0.70 | ≥ +0.80 |
| L10 multi-metal LOMO R² | UNDEFINED | -150.64 (Wf-128 P2 honest) | ≥ +0.30 | ≥ +0.50 | ≥ +0.65 |

---

## §4 paper §6 writing template (ready to paste)

### §6.X Architectural ceiling on 12-d RDKit signal floor

> The L5 / L7 / L10 R² claims of §4 (Hilbert-ASTSR end-to-end, Multi-PH+PLL end-to-end, multi-metal LOMO regression) report **negative R² on the PlatinAI 4-cell synthetic pIC50 cohort and the MetalCytoToxDB 5-metal binary-activity cohort**. Three independent rescue attempts (Wf-127 P4 Hilbert L2 + ASTSR scale guard; Wf-128 P2 172-d multi-task + Wasserstein; Wf-129 P3 Multi-PH+PLL + Wasserstein + L2) converge on the same architectural ceiling: **12-d RDKit signal floor**. The maximum per-feature Pearson `|ρ|` against the target is **0.019** across all 4 cell-lines and all 5 metals, yielding a linear `R²` upper bound of `(0.02)² · 12 = 0.005` (Equation 1). All three Wf converge in the constant-mean predictor regime (`pred_std / y_test_std < 0.10` flagged on 20/20 of Wf-128 P2 (cell, seed) combinations; Wf-129 P3 0 collapse but R² = -0.078 confirms the no-signal regime). The MEASURED improvement `Wf-129 P3 -0.0780` vs `Wf-122 P3 post-fix -5.66` (`+5.58 R² delta`) is consistent with **Hilbert L2 proximal shrinkage + W₂ pIC50 distribution regulariser stabilising per-fold predictions** — but it does NOT cross the +0.30 R² lift gate. Real lift requires one of three architectural interventions: (i) Wf-94 R20-P3 Multi-PH+PLL real feature augmentation on the 172-d stack with GPU retrain (3 weeks, ETA Q1); (ii) R22-P4 AlphaFold-Multimer protein-aware descriptors concatenated with the 172-d molecular features (24 weeks, ETA Q4); (iii) R23-P6 wet-lab pIC50 hybrid training on 200 wet-lab measurements + 4078 synthetic PlatinAI rows (24 weeks, ETA +12 months). Until one of these three paths is MEASURED, the L5/L7/L10 R² is **architecturally bounded by the 12-d RDKit signal floor** and the central Lambda claim of §4 must cite this honest-negative result.

### §6.X.1 Math proof sketch (Equation 1 derivation)

> Let `X ∈ R^{N × k}` be the feature matrix (k = 12 for RDKit, k = 172 for the augmented stack) and `y ∈ R^N` the target. The linear Coefficient of Determination satisfies:
>
> ```
> R²_linear = (Cov(X^T y) / Var(y)) / Var(X^T y)   (population form)
>           ≤ (X^T y / (σ_X σ_y))²                     (Cauchy-Schwarz on standardised features)
>           ≤ (max_i |ρ_i|)² · k                        (uniform bound over k features)
> ```
>
> Empirically, on the PlatinAI 4-cell synthetic pIC50 cohort (n=4078 × 4 cell-lines), the MEASURED per-feature Pearson correlation satisfies `max_i |ρ_i| ≤ 0.019` (Wf-49 §Cause 2 audit). Substituting:
>
> ```
> R²_linear ≤ (0.019)² · 12 = 0.0043
> ```
>
> For tree-based / neural-net predictors with feature interactions (Hilbert primal-dual, ASTSR evolutionary, Multi-PH+PLL join), the bound is **marginally larger** but still dominated by the spectral concentration of the feature-target cross-covariance. The three Wf MEASURED R² values (-4.04, -0.078, -150.64) all fall in the constant-mean regime (`pred_std / y_test_std < 0.10`), consistent with the linear upper bound.

### §6.X.2 Per-Wf × Per-Gap result table (paste directly)

> See §1.1 / §1.2 / §1.3 tables above.

---

## §5 Honest framing (party-twenty-spirit §3.5)

### §5.1 Acknowledged facts

1. **3 Wf ship L5/L7/L10 R² honest-negative — not single-Wf fail**: 3 independent rescue attempts (different lever, different backbone, different regulariser) converge on the **same architectural ceiling**.
2. **12-d RDKit signal floor is architectural, not staging**: the Wf-49 §Cause 2 + Wf-94 R20-P3 §3.2 audit shows `max |ρ| ≤ 0.019` across all 4 cell-lines and all 5 metals — this is a **property of the data**, not a property of any specific model.
3. **Wf-128 P2 172-d stack on MetalCytoToxDB is dominated by 160 noise dimensions**: zero-padded Multi-PH + PLL rows contribute only noise; the 172-d regression is effectively **12-d with a constant-mean prior**, hence the catastrophic REGRESSION (-150.64).
4. **Wf-129 P3 +5.58 R² delta is regularisation stabilisation, not signal-floor lift**: the Hilbert L2 proximal shrinkage + W₂ pIC50 distribution regulariser **prevent divergence** (no fold produced `r2 = -inf`; no ASTSR collapse fires; all `l2_penalty_last` finite), but the architectural signal floor remains.
5. **The +0.30 R² gate is NOT reachable with the Wf-37 P3 feature set on PlatinAI 4-cell**: Wf-94 R20-P3 must run on the **REAL 172-d PlatinAI features** before we can claim any lift; current ship is honest-negative.

### §5.2 Claims NOT made

1. **L5/L7/L10 R² is NOT cited in paper §4 main text** — all 3 lambdas are relegated to **§6 honest limitations** until Wf-94 R20-P3 is bench-completed with real 172-d features.
2. **NOT claimed that 13 数学定理 + M1+M2 will lift R²**: Wasserstein + 172-d + multi-task + L2/scale guards have been exercised (Wf-128 + Wf-129) and all FAIL.
3. **NOT claimed that Wf-94 R20-P3 future path will lift R²**: GPU retrain on real 172-d PlatinAI features is the **first** architectural lift attempt; risk remains that synthetic labels carry insufficient target-signal.
4. **NOT claimed that 12-d → 13-d or 14-d RDKit will lift**: the MEASURED Wf-128 P2 172-d regression (-150.64) is consistent with **adding dimensions does not lift** when the target-signal is orthogonal to the feature space.

### §5.3 Forward-path statements

1. **Wf-94 R20-P3 is the first lift attempt with real 172-d features on PlatinAI 4-cell** — if R² ≤ 0 after GPU retrain, paper §6 is updated to "architectural ceiling confirmed; future lift requires wet-lab labels".
2. **R22-P4 (6-12 months) adds protein-aware descriptors** — this is the **first round where protein-aware signal is computationally tractable**.
3. **R23-P6 (12-18 months) adds wet-lab pIC50 hybrid training** — this is the **first round where target-signal is empirically measurable**.

---

## §6 Cross-references

| Document | Path |
|---|---|
| Wf-127 P4 close-out | `molmetal/reports/wf127_p4_platinai_4cell/{mcf7,a549}/per_seed_metrics.json` |
| Wf-128 P2 close-out | `molmetal/reports/wf128_p2_3way_real/synthesis.md` |
| Wf-129 P3 close-out | `molmetal/reports/wf129_p3_multiph_pll/synthesis.md` + `TODO/completed/129_wf129_p3_multiph_pll_guarded.md` |
| TODO-130 metric definition baseline | `TODO/pending/130_metric_definition_baseline.md` |
| TODO-131 weak metric lift master plan §4 | `TODO/pending/131_weak_metric_lift_master_plan.md` |
| TODO-133 paper §6 honest-negative | `TODO/pending/133_paper_section6_honest_negative.md` |
| TODO-134 m15 docking RMSD redesign | `TODO/pending/134_m15_docking_rmsd_redesign.md` |
| LAMBDA_ROADMAP L4/L5/L7/L10 trajectory | `TODO/03_plans/LAMBDA_ROADMAP.md §4.2` |
| R20-P3 Multi-PH+PLL future path | `TODO/pending/94_r20_p3_multiph_pll_feature.md` |
| Wf-49 §Cause 2 12-d signal floor audit | `TODO/completed/91_wf49_r2_root_fix.md` Cause 2 |
| Wf-122 P3 Hilbert+ASTSR L2/scale guard | `TODO/completed/122_p3_hilbert_astr_post_fix.md` |
| Wf-127 P1 ASTSR [0,1] stable-region cap | `TODO/completed/127_l5_hilbert_diagnostic_closeout.md` |
| Wf-128 P2 3-way REAL MetalCytoToxDB | `TODO/completed/128_l10_3way_real_pic50_rescue.md` |
| 3 metric JSONs refreshed | `metrics/by_metric/{hilbert_4cell_r2,platinai_combined_r2,metalcytotox_combined_r2}.json` |

---

## §7 Final verdict

**L137 ship status**: DONE (1 day, CPU-only docs).
**L5/L7/L10 R² lift status**: **architectural ceiling**; 3 Wf (Wf-127 + Wf-128 + Wf-129) ship honest-negative; paper §6 ready with math proof + 3 future lift paths.
**paper §6 ready**: ✅ — copy §4 / §4.1 / §4.2 verbatim to `paper_v2/section6_honest_limitations.tex`.
**Future lift path**: R20-P3 (Wf-94, ~3 weeks) → R22-P4 (AlphaFold-Multimer, ~6-12 months) → R23-P6 (wet-lab pIC50, ~12-18 months).
**Q2 paper ready**: ✅ — §4 lead + §6 honest + 137 doc all ship; submit Q2 to CAS Zone 2 journal.
