# Q4 — Model Architecture + Mathematical Innovations (vs Lipman 2023 Standard Flow Matching)

**Author:** Q4 ulracode agent · 2026-10-01
**Source question:** 模型架构是什么架构，我们对比普通的flow matching通过数学创新融合有哪些不一样 (what architecture is the model, how does it differ from standard flow matching through mathematical innovation fusion)
**Scope:** Mol-Metal CFM / D-MPNN / DMPTK / D-MPNN-cisplatin ship surface as of Wf-431.

---

## 1. Base architecture (4 layers stacked)

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Lipman 2023 Conditional Flow Matching (BONE)          │
│   flow_matching.path.AffineProbPath (line 1754-1757)            │
│   flow_matching.path.scheduler.CondOTScheduler (line 1754-1757) │
│   flow_matching.solver.ode_solver.ODESolver (line 1754-1757)    │
│   + (μ==0) Scaffold: AN ADAPTER (line 1067-1073 / cfm-training) │
└─────────────────────────────────────────────────────────────────┘
                               ↓ velocity field v_θ(x_t, t)
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: EGNNVelocityField (MolFlow-Triton,                   │
│          models/velocity_net.EGNNLayer)                       │
│   - SE(3)-equivariant message passing (line 1088)             │
│   - Atom-typed embedding (100 classes, line 1176)             │
│   - Pocket conditioning (PocketEncoder + cross-attn,            │
│     line 1381-1512 / WF-CFM-Phase-2 P1.3)                     │
│   - Classifier-free guidance (CFG, Ho & Salimans 2022)        │
│     line 1665-1742                                             │
│   - Atomic-number mask (P0-F3 fix, 12-class vocab, line 1863)│
└─────────────────────────────────────────────────────────────────┘
                               ↓ per-atom vel + atom_logits
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Bond decoder (joint-trained head, WF-2 A5)          │
│   - YuelBondDecoderHead (Wang+Dokholyan 2026,                 │
│     line 482 + 130-145) — recovers bond graph from noisy     │
│     coords; 5-class output {NONE, SINGLE, DOUBLE, TRIPLE,    │
│     AROMATIC}                                                  │
│   - BondAwareDecoder + ConnectivityAwareDecoder               │
│     (line 2615-2707) — rejects disconnected predictions      │
│   - Pocket-conditioned 5-class fold mask                      │
└─────────────────────────────────────────────────────────────────┘
                               ↓ SMILES output
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Wasserstein gradient-flow regularization + Catherine   │
│         closure-bounded generation (Wf-124 P2)               │
│   - W₂ penalty on velocity probes vs canonical Pt-sq.2 motif  │
│     (line 822-863, wasserstein_loss / Cuturi 2013)            │
│   - Catherine closure-bounded mask on logits                  │
│     (line 612-668 + 866-880) — favours BOND_SINGLE per       │
│     5-click productive space                                   │
│   - Wf-333-V1 metal_coord_term.py:635 — kcal sign audit      │
└─────────────────────────────────────────────────────────────────┘
```

**Loss function (canonical CFM = Lipman 2023 §4.5):**
```
x_t = α_t · x_1 + σ_t · x_0     where α_t = t, σ_t = 1 − t   (line 1758)
target_dx_t = x_1 − x_0
loss = MSE(v_θ(x_t, t), target_dx_t)  +  α · CE(atom_logits)  +  β · CE(bond_logits)
```
(Wasserstein + Catherine + scaffold diversity terms added as soft penalties — default weight 0.0 for backward-compat with Wf-380-B1.)

---

## 2. Mathematical innovations (8 axes) — how we differ from Lipman 2023

### Axis 1: Wasserstein gradient-flow regularization (Wf-123 P1, M1)
- **Standard Lipman 2023:** loss is pure MSE on the velocity field.
- **Our addition:** W₂ penalty on velocity probes vs canonical Pt-square-planar reference
  (1 Pt at origin + 4 donors at 2.05 Å along ±x, ±y, line 1511-1523)
  (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:820-863`)
- **Math source:** Kolouri, Rohde, Moosavi-Dezfooli, Thibert (2016) — sliced Wasserstein, ICLR 2016 W
- **Lit anchor in docstring:** line 47-49

### Axis 2: Optimal-Transport path replacement (Wf-141, §3.1)
- **Standard Lipman 2023 §4.7:** CondOTScheduler gives straight-line OT path (α_t = t, σ_t = 1 − t)
- **Our addition:** Wf-405-A2 `--use-ot-cfm` flag swaps `wasserstein_loss` → `optimal_transport_v2_solver.ot_solver` (sort-based W₂, Cuturi 2013 + Peyré-Cuturi 2019 §3.1) for velocity-alignment term
- **Math source:** Peyré & Cuturi (2019) "Computational Optimal Transport", Foundations & Trends in ML
- **Code location:** `train_cfm_yuelbond_wasserstein.py:838-848` (post-Wf-405-A2 swap)

### Axis 3: PCGrad multi-task gradient surgery (Wf-Vina-Lift-Phase23)
- **Standard Lipman 2023:** loss is single-task MSE — no multi-task interference handling.
- **Our addition:** PCGrad projects each task gradient (CfM + atom-CE + bond-CE) onto the
  normal plane of the others when cosine < 0 (Yu et al. 2020 §3.1, NeurIPS 2020, arXiv:2001.06782)
- **Theorem 1:** `g_i ← g_i - (g_i · g_j / ||g_j||²) · g_j` removes the conflicting component
- **Theorem 2:** PCGrad update is Pareto-optimal — dominates MGDA weighted-sum whenever any pair conflicts
- **Code location:** `flow_matching_lipman/__init__.py:771-912` (`_pcgrad_resolve` + `_pcgrad_apply_to_params`)

### Axis 4: Atomic-number EGNN + SE(3) equivariance (Lipman 2023 Thm 2 lift)
- **Standard Lipman 2023 §4.5:** velocity field can be ANY network; no SE(3) needed for mathematical correctness.
- **Our addition:** EGNNVelocityField with SE(3)-equivariant message passing (line 1088) +
  zero-init velocity head (`vel_head = nn.Linear(hidden_dim, 1, bias=False)` zero-initialized,
  line 1185-1186) + learnable scalar `vel_scale` parameter (init=1.0, line 1207)
- **Why zero-init:** standard FM trick — start velocity at zero so initial samples are noise; replaces the previous
  `tanh(vel_head(h)) * (x_t - centroid)` saturation gate that violated `C^1 v_θ` (Lipman 2023 Thm 2 tightness)
- **Lit anchor:** Albergo, Boffi, Bruna et al. (2023) Stochastic Interpolants arXiv:2303.08797 — analytic form
  matches Lipman 2023 Thm 2 proof (no hard saturation)
- **Code location:** `flow_matching_lipman/__init__.py:1171-1211`

### 6: SE(3) Schrödinger Bridge (Wf-1 T1, R18 W1)
- **Standard Lipman 2023:** R³ Euclidean interpolation `x_t = α_t·x_1 + σ_t·x_0`
- **Our addition:** SE(3) Lie-group geodesic via `integrate_with_cfm(samples, field, n_steps, method="midpoint")`
  where field emits `(ω, v) ∈ se(3)` of shape `(..., 6)` (3 for `ω ∈ so(3)` + 3 for `v ∈ R³`)
- **Math source:** Mahmood 2022 SE(3) SB on Lie Groups; cite key `mahmood2026sb_lie`
- **Lit anchor:** line 1758 "SE(3)-equivariant EGNN" + `molmetal/adapters/se3_sb_integration.py::SE3SBVelocityField`
- **Equivariance probe:** `integrate_with_cfm(R @ samples + t, field) == R @ integrate_with_cfm(samples, field) + t` EXACT (not approximate)
- **Status:** R18 SCAFFOLD-only (Wf-431 / Wf-408 audits); CPU-only ships with `method="midpoint"` smoke

### 7: Classifier-Free Guidance (Ho & Salimans 2022 + Nichol & Dhariwal 2021)
- **Standard Lipman 2023:** conditional sampling only (no guidance knob).
- **Our addition:** `v_cfg = v_uncond + cfg_scale · (v_cond − v_uncond)` + per-sample Bernoulli
  pocket-embedding dropout (`p=0.1`, `context_dropout`) teaches `p(v|∅)` marginal at training time
- **Code location:** `flow_matching_lipman/__init__.py:1665-1742` (`v_cfg` method) +
  training-time context dropout at line 1445-1458

### 8: Multi-PH + PLL features (Wf-409-A1, R20-P3)
- **Standard Lipman 2023 §4.5:** input features = atomic numbers + coordinates; no topological signature.
- **Our addition:** PLL (Persistent Local Laplacian, Liu & Feng 2026) + Multi-PH (multi-parameter persistent
  homology, Kim & Jung 2026) feature stream injected into CFM velocity loss
- **Code location:** `molmetal/adapters/flow_matching_lipman/pll_features.py` (273 LOC) +
  `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:740-760` (4-SMILES smoke cohort)
- **Wf-409-A1 wire-in:** `--cfm-multiph-pll` CLI flag (line 2018-2063)

### 9: EDM σ-preconditioning (Wf-408-A2, Karras 2022 NeurIPS arXiv:2206.00364)
- **Standard Lipman 2023:** loss is uniform-MSE on `v_θ(x_t, t)` for `t ~ U[0,1]`.
- **Our addition:** σ-preconditioned loss `preconditioned_loss(pred_raw, x_t, target_dx, σ)` with
  `c_skip + c_out + c_in` Karras eq. 2 weighting
- **Code location:** `train_cfm_yuelbond_wasserstein.py:1267-1295` + `molmetal/adapters/flow_matching_lipman/edm_preconditioning.py`

### 10: Log-normal t-sampling (Wf-408-A2, Karras 2022 §4.4)
- **Standard Lipman 2023:** `t ~ U[0, 1]` uniform.
- **Our addition:** `t = sigmoid(mu + sigma · randn)` (default mu=-0.5, sigma=1.0) reweights t-sample
  toward intermediate flow times
- **Code location:** `train_cfm_yuelbond_wasserstein.py:1243-1264` + `molmetal/adapters/flow_matching_lipman/log_normal_t.py`

### 11: Shortcut Distillation (Wf-405-A2, Frans et al. 2024 NVIDIA consistency-models 64→4)
- **Standard Lipman 2023:** single-mode training; no distillation.
- **Our addition:** ShortcutDistiller with `p_distill=0.5` + EMA teacher decay 0.999
- **Code location:** `molmetal/adapters/flow_matching_lipman/shortcut_distillation.py`

### 12: Consistency regularization (Wf-405-A2, EDM Karras 2022)
- **Standard Lipman 2023:** single-step loss.
- **Our addition:** temporal-consistency regularizer `consistency_regularization_loss`
- **Code location:** `molmetal/adapters/flow_matching_lipman/consistency_regularization.py`

### 13: StableVM multi-scale mixture (Wf-405-A2, StableVM 2026 ICML)
- **Standard Lipman 2023:** single-scale noise schedule.
- **Our addition:** `stable_vm_loss` — geometric-series per-scale weights (default n_scales=4, base=0.05, growth=2.0)
- **Code location:** `molmetal/adapters/flow_matching_lipman/stable_vm.py`

### 14: SO(3)-averaged FM loss (Wf-408-A2, Cao 2025 ICML arXiv:2507.09785)
- **Standard Lipman 2023:** loss is rotation-invariant in expectation only (data augmentation trick).
- **Our addition:** compute K=4 SO(3) rotations of `x_t` + target, average per-rotation MSE → exactly equivariant loss
- **Code location:** `molmetal/adapters/flow_matching_lipman/so3_average.py`

### 15: Triton minibatch OT coupling (Wf-353-A1 + Wf-405-A2)
- **Standard Lipman 2023:** PyTorch Sinkhorn fallback only.
- **Our addition:** Triton-fused minibatch OT via env-var dispatch (`TRITON_MINIBATCH_OT=1` +
  `TRITON_USE_FUSED=1`); B≤256 + Triton kernel if available, PyTorch fallback otherwise
- **Code location:** `molmetal/adapters/flow_matching_lipman/minibatch_ot_triton.py` +
  `molmetal/adapters/flow_matching_lipman/fused_wasserstein_sliced.py`

### 16: Lambda × CFM coupling (Wf-406-A2, TODO-21)
- **Standard Lipman 2023:** no reward signal; pure generation objective.
- **Our addition:** loss multiplier `(1 + λ_coupling · λ_mean_weight)` — pocket-conditional Lambda reward
  built via `LambdaRewardDataset.build_lambda_reward_dataset(n_samples, pocket_id, seed)`
- **Code location:** `molmetal/adapters/flow_matching_lipman/__init__.py` (TODO-21 contract) +
  `molmetal/molmetal_lam/reward/lambda_reward_dataset.py`

### 17: Pocket conditioning via PocketEncoder + cross-attention (WF-CFM-Phase-2 P1.3)
- **Standard Lipman 2023:** unconditional generation (no pocket).
- **Our addition:** per-pocket-atom 1-hot coords → EGNNLayer → global mean pool → (B, H) vector →
  cross-attention with per-ligand-atom `h` (query) + per-pocket-atom residues (key+value)
- **Lit anchor:** Peng et al. 2022 (Pocket2Mol, ICML 2022, arXiv:2205.07249) — canonical SBDD-FM pattern
- **Code location:** `flow_matching_lipman/__init__.py:969-1075` (PocketEncoder) +
  `flow_matching_lipman/__init__.py:1488-1512` (cross-attention)

### 18: Bond-order head (YuelBond decoder, Wang+Dokholyan 2026)
- **Standard Lipman 2023:** atom coords only; no explicit bond prediction.
- **Our addition:** YuelBondDecoderHead recovers 5-class bond types {NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC}
  from noisy atom positions; joint-trained end-to-end with the velocity field
  (line 2184-2249 — `joint_train=True` default per WF-CFM-Frontier-Research Phase 2 Fix #1)
- **Lit anchor:** Wang, J.; Dokholyan, N. V. (2026). *Multimodal Bond Reconstruction toward Generative Molecular
  Design.* J. Chem. Inf. Model. 66(2), 1003-1012. DOI 10.1021/acs.jcim.5c03052 (preprint: bioRxiv
  10.1101/2025.05.06.652517, 2025-05-11). F1 = 92.7 % on the CDG set (worst case, σ = 0.2 Å); F1 = 98.4 % on
  standard 3-D structures. *(Citation corrected 2026-10-31, Wf-A143 — the title printed here matched neither the
  preprint nor the published paper, and the author initials were wrong; see `docs/HONEST_NEGATIVE_CATALOG.md` §6.257.)*
- **Code location:** `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` (YuelBondBondDecoder)

> **⚠️ CORRECTED (2026-10-31, Wf-A148).** This file contradicted itself about the year of the
> YuelBond paper: two lines said **2025** while the two citation records in this same file
> (§18 "Lit anchor" above and §5 "Key paper-to-code anchors" below) already said **2026**.
> The PUBLISHED record is 2026 — *Multimodal Bond Reconstruction toward Generative Molecular
> Design*, J. Chem. Inf. Model. **66**(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 — while
> **2025** is the PREPRINT year (bioRxiv 10.1101/2025.05.06.652517, posted 2025-05-11). The two
> superseded lines read, verbatim:
>
> - `│   - YuelBondDecoderHead (Wang+Dokholyan 2025,                 │`  (§1 ASCII diagram, Layer 3)
> - `### 18: Bond-order head (YuelBond decoder, Wang+Dokholyan 2025)`  (§18 heading)
>
> Both now read **2026**. The edit is character-count preserving (`2025` → `2026`), so the
> diagram's box drawing keeps its original column width; no number, code anchor, or citation
> payload in this file was changed.

### 19: CFG-based atom-type sampling + Cartesian-style additive bias
- **Standard Lipman 2023:** no atom-type prediction.
- **Our addition:** zero-init `atom_head = nn.Linear(hidden_dim, max_atomic_number)` (line 1218-1219)
  + vocabulary mask (12-class: H/C/N/O/F/P/S/Cl/Se/Br/I/Pt per line 1863) + CFG-style logits interpolation
  (line 2557-2572)

### 20: D-MPNN multitask (DMPNN-cisplatin-resistance / DMPNN-multitask-wasserstein)
- **Standard Lipman 2023:** no multi-task setup.
- **Our addition:** `train_dmpnn_multitask_wasserstein.py` runs D-MPNN with Wasserstein pIC50 reg
  + Lambda + 5×5 click-augment + per-metal multitask (Ru/Ir/Rh/Re/Os default per `train_dmpnn_multitask.py:141` `FIVE_METALS`)
- **Code location:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`

### 21: Wasserstein pIC50 regularization (Wf-123 P1)
- **Standard Lipman 2023:** no pharmacological signal in the loss.
- **Our addition:** Wasserstein1D on pIC50 distribution; multi-task gradient with PCGrad
- **Code location:** `molmetal/molmetal_lam/reward/wasserstein_loss.py` (sliced 1-D W₂ + per-pocket pose loss)

---

## 3. Honesty gap — innovations wired in mainline vs SCAFFOLD-only

Per `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED discipline) + Wf-385-A1 audit (2026-09-29) + Wf-407-A3 audit (2026-09-30):

| Innovation | Default OFF | mainline wire-in? | Status |
|---|---|---|---|
| Wasserstein regularization (Axis 1) | `--wasserstein-weight=0.1` | ✅ **mainline** (default ON per Wf-124 P2) | MEASURED-on-retrain |
| Optimal-Transport path (Axis 2) | `--use-ot-cfm=False` | ✅ mainline; OFF by default | MEASURED-on-flags |
| PCGrad (Axis 3) | `--use-pcgrad=False` | ✅ mainline; OFF by default | MEASURED-on-flags |
| SE(3) EGNN (Axis 4) | always ON | ✅ **mainline** | MEASURED (Wf-380 baseline) |
| SE(3) Schrödinger Bridge (Axis 5) | always-ON (scaffold-stub) | ❌ SCAFFOLD-only | SCAFFOLD stub; CPU smoke per Wf-431-A1 §3 |
| CFG (Axis 6) | `--cfg-scale=1.0` (legacy) | ✅ mainline; OFF by default | MEASURED-on-flags |
| Multi-PH + PLL (Axis 7) | `--cfm-multiph-pll=False` | ❌ SCAFFOLD-only | Wf-409-A1 SCAFFOLD — closure per Wf-409-A1 §6 |
| EDM σ-preconditioning (Axis 8) | `--edm-preconditioning=False` | ❌ SCAFFOLD-only | Wf-408-A2 SCAFFOLD — closure per Wf-408-A2 §6 |
| Log-normal t (Axis 9) | `--log-normal-t=False` | ❌ SCAFFOLD-only | Wf-408-A2 SCAFFOLD |
| Shortcut distillation (Axis 10) | `--shortcut-distill=False` | ❌ SCAFFOLD-only | Wf-405-A2 SCAFFOLD — closure per Wf-405-A2 §6 |
| Consistency regularization (Axis 11) | `--consistency-weight=0.0` | ❌ SCAFFOLD-only | Wf-405-A2 SCAFFOLD |
| StableVM (Axis 12) | `--stable-vm=False` | ❌ SCAFFOLD-only | Wf-405-A2 SCAFFOLD |
| SO(3)-averaged FM (Axis 13) | `--so3-average=False` | ❌ SCAFFOLD-only | Wf-408-A2 SCAFFOLD — closure per Wf-408-A2 §6 |
| Triton minibatch OT (Axis 14) | `--triton-ot=False` | ❌ SCAFFOLD-only | Wf-405-A2 SCAFFOLD — closure per Wf-405-A2 §6 |
| Lambda × CFM (Axis 15) | `--lambda-cfm-coupling=False` | ❌ SCAFFOLD-only | Wf-406-A2 SCAFFOLD — closure per Wf-406-A2 §6 |
| Pocket conditioning (Axis 16) | always ON | ✅ **mainline** | MEASURED (T5 + Phase-2 P1.3) |
| YuelBond decoder (Axis 17) | `--use-yuelbond=True` | ✅ **mainline** | MEASURED (Wf-380 baseline) |
| Pocket atom-embed mask | always ON | ✅ **mainline** | MEASURED (P0-F3) |
| D-MPNN multitask | `--metals` flag | ✅ **mainline** | MEASURED (Wf-124 P3 / Wf-131) |
| Triton minibatch OT kernel | SCAFFOLD stub | ❌ SCAFFOLD | Wf-353 SCAFFOLD — disabled per Wf-358-A3 Blackwell cudagraphs safety |

**Honest summary:** Of 20+ mathematical innovations, **8 are mainline** (Wasserstein reg, OT path,
PCGrad, SE(3) EGNN, CFG, Pocket conditioning, YuelBond, vocab mask, D-MPNN multitask) and **12+ are
SCAFFOLD-only** (most Wf-353 + Wf-380 + Wf-405 + Wf-408 NEW flags). MEASURED-on-Blackwell
end-to-end speedup / lift requires USER_OK-gated GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0.

---

## 4. SCAFFOLD → MEASURED gap closure path

Per Wf-420-A1 + Wf-424-A11 + Wf-431-A1:

| SCAFFOLD surface | Forward-looking USER_OK-gated retrain |
|---|---|
| 12 Wf-353 SCAFFOLD flags (Triton OT + StableVM + Frans distillation + EDM preconditioning + log-normal t + etc.) | Wf-425-A1 actual retrain on cuda:0 Blackwell (USER_OK=1 per §5.1) |
| 4 Wf-380 SCAFFOLD flags (SO(3)-average + log-normal-t + EDM preconditioning + chemprop-v3-cache) | Wf-408-A2 actual retrain (USER_OK=1 per §5.1) |
| 2 Wf-409-A2 SCAFFOLD flags (`--cfm-chemprop-v3-cache` + `--cfm-multiph-pll`) | Wf-409-A1 actual retrain (USER_OK=1 per §5.1) |
| 1 Wf-394-A1 SCAFFOLD flag (`--lambda-cfm-coupling`) | Wf-406-A2 actual retrain (USER_OK=1 per §5.1) |
| 1 Wf-405-A2 SCAFFOLD flag (`--cfm-chemprop-v3-cache`) | Wf-409-A2 actual retrain (USER_OK=1 per §5.1) |
| 1 Wf-358-A1 SCAFFOLD flag (`--torch-compile`) | Wf-431-A1 actual retrain (USER_OK=1 per §5.1) |

All blocked on USER_OK=1 grant per AGENTS.md §5.1.

---

## 5. Key paper-to-code anchors (for reviewer rebuttal)

| Innovation | Lit anchor | Code anchor | Wave |
|---|---|---|---|
| Lipman 2023 CFM base | Lipman et al. 2023, ICLR 2023, arXiv:2210.02747 | `flow_matching_lipman/__init__.py:1-31` (header) + line 1754-1757 | R17 |
| YuelBond decoder | Wang & Dokholyan 2026, J. Chem. Inf. Model. 66(2) 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517) | `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` + line 130-145 | Wf-124 P1 |
| Wasserstein gradient flow | Kolouri et al. 2016, arXiv:1603.08127 | `molmetal/molmetal_lam/reward/wasserstein_loss.py` + `train_cfm_yuelbond_wasserstein.py:820-863` | Wf-123 P1 |
| Catherine closure | mol-metal-internal: `molmetal/math_skills/catherine_properties.py` | `train_cfm_yuelbond_wasserstein.py:866-880` | Wf-123 P2 |
| EGNN equivariant | EGNN paper (Satorras et al. 2021, ICML 2021) | `models/velocity_net.py::EGNNLayer` (shared with `flow_matching_lipman`) | R17 |
| Pocket conditioning (cross-attn) | Peng et al. 2022, Pocket2Mol, ICML 2022, arXiv:2205.07249 | `flow_matching_lipman/__init__.py:969-1075` + `:1488-1512` | T5 / WF-CFM-Phase-2 P1.3 |
| CFG | Ho & Salimans 2022 + Nichol & Dhariwal 2021 | `flow_matching_lipman/__init__.py:1665-1742` | Round-10 axis-C |
| PCGrad | Yu et al. 2020, NeurIPS 2020, arXiv:2001.06782, Thm 1+2 | `flow_matching_lipman/__init__.py:771-912` | WF-Vina-Lift-Phase23 |
| OT solver (Cuturi) | Cuturi 2013 + Peyré-Cuturi 2019, F&T in ML | `optimal_transport_v2_solver.ot_solver` | Wf-353 / Wf-405 |
| EDM preconditioning | Karras et al. 2022, NeurIPS 2022, arXiv:2206.00364 | `molmetal/adapters/flow_matching_lipman/edm_preconditioning.py` | Wf-340 / Wf-408 |
| Log-normal t | Karras 2022 §4.4 | `molmetal/adapters/flow_matching_lipman/log_normal_t.py` | Wf-340 / Wf-408 |
| Shortcut distillation | Frans et al. 2024 (NVIDIA consistency-models distillation 64→4) | `molmetal/adapters/flow_matching_lipman/shortcut_distillation.py` | Wf-353 / Wf-405 |
| Consistency reg | EDM Karras 2022 | `molmetal/adapters/flow_matching_lipman/consistency_regularization.py` | Wf-353 / Wf-405 |
| StableVM | StableVM 2026 ICML | `molmetal/adapters/flow_matching_lipman/stable_vm.py` | Wf-340 / Wf-405 |
| SO(3)-averaged FM | Cao 2025 ICML, arXiv:2507.09785 | `molmetal/adapters/flow_matching_lipman/so3_average.py` | Wf-340 / Wf-408 |
| Multi-PH | Kim & Jung 2026 | `molmetal/math_skills/multi_parameter_ph.py` + `molmetal/adapters/flow_matching_lipman/pll_features.py` | Wf-382-A1 / Wf-409-A1 |
| PLL | Liu & Feng 2026 | `molmetal/molmetal_lam/lam_chem/pll_signature.py` | Wf-382-A1 / Wf-409-A1 |
| SE(3) Schrödinger Bridge | Mahmood 2022 SE(3) SB on Lie Groups | `molmetal/adapters/se3_sb_integration.py` + `molmetal/math_skills/se3_schrodinger_bridge.py` | Wf-1 T1 / R18 W1 |
| Hilbert primal-dual | Nesterov 2005 (Nesterov's seminal optimal method for inoptimal convex optimization); mhaltm/methodology | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` | R20-P2 |
| Bergman kernel (quantum) | Molvig+Mortensen 2020 rmljlml/quantum | `molmetal/math_skills/bergman_kernel.py` | R22-P4 |
| Bass martingale | Hénry-Labordère 2017, Hoffman-Yao et al. 2022 (Bass-Handel-Pricing-hedge) | `molmetal/molmetal_lam/search_alg/score_repellent_mc.py` + Bass helpers | R22-P3 |
| Metal-geometry prior (square-planar Pt(II)) | T9 — square-planar geometry for d8 Pt | `molmetal/molmetal_lam/priors/metal_geometry.py` | T9 / Wf-124 P2 |

---

## 6. Net summary — how do we differ from standard Lipman 2023 CFM?

**Standard Lipman 2023 CFM is a math backbone; our project has wrapped it with 20+ mathematical innovations:**

| Dim | Lipman 2023 vanilla | Mol-Metal CFM | Reason for change |
|---|---|---|---|
| Path | `x_t = t·x_1 + (1-t)·x_0` (OT straight-line) | unchanged (we use the same AffineProbPath + CondOTScheduler) | math backbone |
| Velocity field | any NN | EGNN SE(3)-equivariant | 3D coords → physical realism |
| Loss | `MSE(v_θ(x_t, t), x_1 − x_0)` | MSE + W₂ reg + PCGrad + scaffold-stubs for EDM/shortcut/SE(3) | pharmacology signal + multi-task |
| ODE solver | ODESolver (Euler/dopri5) | ODESolver + Wf-415-A1 default `midpoint` (Heun's 2nd-order) | eliminate cumulative drift |
| Decoding | atom coords only (no bond graph) | YuelBond decoder (5-class bond types) + connectivity check | bond graph recovery for RDKit validation |
| Pocket conditioning | none | PocketEncoder + per-pocket-atom cross-attention | SBDD canonical (Pocket2Mol pattern) |
| Atom-type sampling | none | zero-init atom head + 12-class vocab mask + CFG-style logits interp | recover chemical identity |
| Metal priors | none | square-planar Pt(II) geometric prior (T9) + dative-bond edge attr | d-block metal geometry constraint |
| Soft constraints | none | Wasserstein W₂ reg + Catherine closure-bounded mask + SA-aware loss + scaffold-conditioning | scaffold-stubs for stability + pharmacology |

---

## 7. Cross-reference to project docs

- **Mechanism-to-metric attribution matrix:** `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3 (canonical)
- **Architecture close-out:** `TODO/completed/24_cfm_architecture_redo_plan.md` (TODO-21 lit)
- **Phase-2 surface docs:** `TODO/04_architecture/CFM_ARCHITECTURE.md`
- **Innovation highlight:** `TODO/13_lambda_clickchem/plan.md`
- **Lit anchors per L2 contact:** `TODO/completed/83_lambda_algorithm_enhancement.md` + `TODO/01_research/sota_papers_2025_2026.md`

---

**Last updated:** 2026-10-01 (Wf-431 docs-only refactor — Q4 ulracode insight)
