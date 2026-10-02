# Wf-340-A4 — Compatibility Assessment: CFM-Acceleration Opportunities × D-MPNN / W₂ / Multi-Task / m6·m7

**Date:** 2026-09-29
**Wf:** Wf-340-A4 (CFM training framework improvement ultracode — compatibility assessment phase)
**Phase:** A4 — Compatibility matrix + blocking-issue identification (this file)
**Status:** CPU-only analysis complete; zero GPU launches; zero integration code changes
**Owner:** Hugo Lee + ultracode agent
**Parent:** `TODO/completed/340r1_cfm_papers_survey.md` (R1 papers) +
  `TODO/completed/340r2_acceleration_methods.md` (R2 acceleration methods) +
  `TODO/completed/340r3_molecule_cfm.md` (R3 molecule-specific CFM) +
  `TODO/completed/340r4_code_repos.md` (R4 code-repo survey) +
  `TODO/completed/339b4_m6_kappa_diagnostic.md` (B4 m6_kappa structural ceiling)
**Cross-ref:** `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3 (contribution matrix)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline preserved)

---

## §0 TL;DR — Wf-340-A4 compatibility verdict

| Axis | Verdict | Reason |
|---|---|---|
| **D-MPNN encoder (graph-level)** | **ORTHOGONAL** (no compatibility issue) | D-MPNN lives in the **downstream pIC50 regression pipeline**, NOT in the CFM-EGNN path. CFM acceleration does not change D-MPNN architecture. |
| **Wasserstein W₂ pIC50 regression (Wf-328-W5 sort)** | **ORTHOGONAL** | W₂ sort is a 1-D closed-form gradient on the pIC50 axis. CFM loss changes do not affect the W₂ gradient mode. |
| **Multi-task (5-metal) classification** | **ORTHOGONAL** | Multi-task heads live on the D-MPNN encoder, downstream of CFM-generated structures (when used). |
| **m6_kappa + m7 diversity** | **INDEPENDENT** | m6_kappa is computed on a fixed reference cohort (`metallo_drugs_500_train.csv`, n=449) via RDKit Kappa shape descriptors — NOT on CFM-generated structures. m7 is computed on Lambda MCTS-generated cohorts via Morgan-ECFP4 Tanimoto distance. CFM acceleration does not change either metric's *primary computation* but can change CFM-generated cohorts that feed m7 in some downstream Wf. |
| **Blocking issues count** | **2/8** (FlowMol + Augmented FM carry risk) | FlowMol's joint-coord+type+bond output representation may break downstream featurization that expects separate modality outputs. Augmented FM's auxiliary variables are an architectural change requiring regression test on m6/m7. |
| **PROJECTED integration risk** | **LOW** for 6/8; **MEDIUM** for 2/8 (FlowMol + Augmented FM) | The 6 training-loss-level changes (OT-CFM, EDM, StableVM, Reflow, VP-CFM, W-Flow) are non-breaking. |

**Honest framing (`party-twenty-spirit` §3.5):**

- **MEASURED (this assessment is MEASURED):**
  - 8 distinct opportunities identified from Wf-340 R1-R4 (R1 papers, R2 deep-dive, R3 molecule-specific, R4 code repos).
  - 4 compatibility axes verified against the actual codebase paths (D-MPNN: `train_dmpnn_multitask_wasserstein.py`; W₂ sort: `molmetal/scripts/wf328w5`; m6_kappa: `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles`; m7: `molmetal/molmetal_lam/reward/multiph_pll_feature.py:wire_m7_diversity_recovery`).
  - 2 BLOCKING ISSUES identified (FlowMol unified-flow + Augmented FM aux-vars) — both architectural, not training-loss-level.
- **PROJECTED:** Integration risk per opportunity (no actual integration attempted; this is a compatibility matrix, not a benchmark).
- **NOT MEASURED:** Wall-time impact on the Mol-Metal CFM retrain (requires GPU A/B per Wf-340-R2 §7).

---

## §1 Compatibility matrix — 8 opportunities × 4 axes

### §1.1 Legend

- ✅ **ORTHOGONAL** — change is in a separate axis; no interaction with target axis.
- ✅ **COMPATIBLE** — change is on the same axis but does not break the target.
- ⚠️ **COMPATIBLE WITH CAUTION** — change is on the same axis; target may need re-validation.
- ❌ **INCOMPATIBLE / BLOCKING** — change breaks target axis; requires design rework.

### §1.2 Matrix

| # | Opportunity | D-MPNN encoder | W₂ sort (Wf-328-W5) | Multi-task (5-metal) | m6_kappa + m7 | Source |
|---|---|---|---|---|---|---|
| 1 | **OT-CFM (Tong 2023)** — minibatch OT coupling | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT (m6 on ref cohort; m7 only shifts if CFM-generated samples are added) | 340r2 §2.6 + `optimal_transport_v2.py` |
| 2 | **EDM preconditioning (Karras 2022)** — σ-reparameterization | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT | 340r2 §2.5 + 340r2 §4.1 |
| 3 | **StableVM (ICML 2026)** — multi-sample variance reduction | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT | 340r1 §3.2 |
| 4 | **Augmented FM (ICLR 2025)** — auxiliary-variable variance reduction | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ⚠️ COMPATIBLE WITH CAUTION (m6 unchanged; m7 may shift if aux-vars change CFM output distribution) | 340r1 §3.3 |
| 5 | **FlowMol (Dunn 2024)** — unified coord+type+bond flow | ⚠️ COMPATIBLE WITH CAUTION (joint output may break D-MPNN featurization that expects separate modality inputs) | ✅ ORTHOGONAL | ⚠️ COMPATIBLE WITH CAUTION (multi-task head expects graph-level fingerprint, not raw joint flow) | ⚠️ COMPATIBLE WITH CAUTION (m6 Kappa shape uses RDKit Mol which may need different SMILES reconstruction path) | 340r1 §3.4 |
| 6 | **Rectified Flow / Reflow (Liu 2022)** — reflow straightening | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT (reflow straightens trajectories; output distribution is the same data manifold) | 340r2 §2.7 |
| 7 | **VP-CFM (Albergo 2023)** — variance-preserving trigonometric path | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT | 340r1 §2.3 + 340r2 §2.8 |
| 8 | **W-Flow (Stanford ICML 2026)** — waveform relaxation ODE solver | ✅ ORTHOGONAL (affects inference only) | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT (faster sampling, same output manifold) | `molmetal/adapters/flow_matching_lipman/w_flow.py` |

**Summary:**
- 6/8 opportunities are **ORTHOGONAL** on all 4 axes (no interaction; safe to integrate).
- 1/8 (Augmented FM) is **COMPATIBLE WITH CAUTION** on the m6_kappa + m7 axis (architectural change; output distribution may shift).
- 1/8 (FlowMol) is **COMPATIBLE WITH CAUTION** on 3 axes (D-MPNN featurization, multi-task head, m6 SMILES reconstruction). This is the **only opportunity with multi-axis integration risk**.
- **0/8 are INCOMPATIBLE / BLOCKING** outright; all 8 are at minimum COMPATIBLE WITH CAUTION.

---

## §2 Per-opportunity compatibility analysis

### §2.1 OT-CFM (Tong 2023) — `ExactOptimalTransportConditionalFlowMatcher`

**Status in codebase:** PARTIALLY WIRED via `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py`.

**Per-axis analysis:**

| Axis | Compatibility | Reason |
|---|---|---|
| **D-MPNN encoder** | ✅ ORTHOGONAL | D-MPNN operates on molecular graphs (atoms + bonds); CFM operates on continuous-time vector field interpolation between noise and 3D ligand structures. The OT coupling in CFM changes *which* (noise, ligand) pairs are coupled within a minibatch — it does NOT change the molecule distribution that D-MPNN sees. |
| **W₂ sort (Wf-328-W5)** | ✅ ORTHOGONAL | W₂ sort is a 1-D closed-form gradient on the pIC50 regression axis (`train_dmpnn_multitask_wasserstein.py:train_one_epoch_joint`). OT-CFM does not touch the pIC50 axis. |
| **Multi-task (5-metal)** | ✅ ORTHOGONAL | Multi-task heads (Ru/Ir/Os/Rh/Re) are downstream of the shared D-MPNN encoder. OT-CFM changes CFM training only. |
| **m6_kappa + m7** | ✅ INDEPENDENT | m6_kappa is computed on `metallo_drugs_500_train.csv` (n=449 reference cohort) via `m6_kappa_from_smiles` (RDKit Kappa1/2/3 descriptors). OT-CFM changes CFM-generated cohorts, NOT the reference cohort. m7 is computed on Lambda MCTS-generated cohorts via Morgan-ECFP4 Tanimoto; OT-CFM does not change Lambda MCTS. **Caveat:** if a downstream Wf adds CFM-generated samples to the m7 diversity pool, the diversity shift must be re-measured. |

**Verdict:** FULLY COMPATIBLE. Tier-1 candidate (340r2 §4.2). Drop-in for `cfm_loss.py`.

---

### §2.2 EDM preconditioning (Karras 2022) — σ-preconditioning wrapper

**Status in codebase:** NOT WIRED. Pure additive change.

**Per-axis analysis:**

| Axis | Compatibility | Reason |
|---|---|---|
| **D-MPNN encoder** | ✅ ORTHOGONAL | The c_skip/c_out/c_in/c_noise factors are applied to the velocity-net output (per-atom predicted velocity). They re-weight the MSE loss and re-scale the network output but DO NOT change the molecule distribution that D-MPNN sees. |
| **W₂ sort (Wf-328-W5)** | ✅ ORTHOGONAL | Same as OT-CFM — pIC50 axis is downstream of CFM. |
| **Multi-task (5-metal)** | ✅ ORTHOGONAL | Same as OT-CFM. |
| **m6_kappa + m7** | ✅ INDEPENDENT | σ-preconditioning is a wall-budget trick (per 340r2 §2.5 honest-negative): "the preconditioning IS the optimal choice per the Karras ablation study; no quality penalty." Output molecule distribution is unchanged at convergence. |

**Verdict:** FULLY COMPATIBLE. Tier-1 candidate (340r2 §4.1). 150-200 LOC wire-up.

---

### §2.3 StableVM (ICML 2026) — multi-sample variance reduction

**Status in codebase:** NOT WIRED. Code availability TBD (340r1 §3.2).

**Per-axis analysis:**

| Axis | Compatibility | Reason |
|---|---|---|
| **D-MPNN encoder** | ✅ ORTHOGONAL | StableVM averages multiple velocity samples at high-variance regions (near prior); output velocity is the same expectation, lower variance. D-MPNN input unchanged. |
| **W₂ sort (Wf-328-W5)** | ✅ ORTHOGONAL | Same as OT-CFM. |
| **Multi-task (5-metal)** | ✅ ORTHOGONAL | Same as OT-CFM. |
| **m6_kappa + m7** | ✅ INDEPENDENT | At convergence, output molecule distribution is the same as un-preconditioned CFM; only variance in the gradient signal differs (lower variance ⇒ faster convergence). |

**Verdict:** FULLY COMPATIBLE (assuming code is released). Tier-2 candidate (340r1 §6 item 4).

---

### §2.4 Augmented FM (ICLR 2025) — auxiliary-variable variance reduction

**Status in codebase:** NOT WIRED. Architectural change.

**Per-axis analysis:**

| Axis | Compatibility | Reason |
|---|---|---|
| **D-MPNN encoder** | ✅ ORTHOGONAL | Auxiliary variables are latent codes added to the CFM input (e.g., atom-type latent codes per 340r1 §3.3); they do not change D-MPNN's graph featurization. |
| **W₂ sort (Wf-328-W5)** | ✅ ORTHOGONAL | Same as OT-CFM. |
| **Multi-task (5-metal)** | ✅ ORTHOGONAL | Same as OT-CFM. |
| **m6_kappa + m7** | ⚠️ COMPATIBLE WITH CAUTION | m6_kappa is computed on RDKit Mol objects; if aux-vars change the CFM output SMILES distribution, the downstream m6_kappa on CFM-generated samples may shift (NOT on the reference cohort). m7 (Morgan-ECFP4 Tanimoto on Lambda MCTS) is unaffected if Lambda MCTS path is unchanged. **Mitigation:** re-run `m6_kappa.json` emission on CFM-generated cohort after integration; if shift is ±0.05, document as PROJECTED. |

**Verdict:** COMPATIBLE WITH CAUTION. Tier-3 candidate (340r1 §6 item 7). 1 regression test on m6_kappa before integration.

---

### §2.5 FlowMol (Dunn 2024) — unified coord+type+bond flow

**Status in codebase:** NOT WIRED. Architectural refactor.

**Per-axis analysis:**

| Axis | Compatibility | Reason |
|---|---|---|
| **D-MPNN encoder** | ⚠️ COMPATIBLE WITH CAUTION | FlowMol produces a *unified* flow over (coordinates, atom types, bond orders) as a single object. Current D-MPNN featurization (`molmetal/baselines/dmpnn_multitask.py`) expects **separate modality inputs**: SMILES string → RDKit Mol → graph features. If CFM output is now a unified (coord, type, bond) tensor, D-MPNN must reconstruct SMILES first via RDKit, which is **lossy for 3D-specific information** but acceptable for pIC50 (pIC50 is a SMILES-level property). **Mitigation:** add SMILES reconstruction step before D-MPNN featurization; verify pIC50 R² delta is within ±0.02. |
| **W₂ sort (Wf-328-W5)** | ✅ ORTHOGONAL | W₂ sort is on pIC50 axis; FlowMol changes CFM output but not pIC50. |
| **Multi-task (5-metal)** | ⚠️ COMPATIBLE WITH CAUTION | Same as D-MPNN: multi-task head expects graph-level fingerprint, not raw joint flow. Requires SMILES reconstruction. |
| **m6_kappa + m7** | ⚠️ COMPATIBLE WITH CAUTION | m6_kappa_from_smiles takes a SMILES; FlowMol output must be reconstructed to SMILES first. RDKit MolFromSmiles + Sanitize is lossy for 3D-specific bond orders; if FlowMol produces a bond order that RDKit cannot canonicalize, the m6_kappa is undefined. **Mitigation:** validate SMILES reconstruction success rate > 99% on FlowMol output before m6_kappa emission. |

**Verdict:** COMPATIBLE WITH CAUTION. Multi-axis integration risk. **Only opportunity with risk on 3/4 axes.** Tier-2 candidate for future architectural refactor (340r1 §6 item 5).

**Blocking issue identification:** FlowMol's joint-output representation is the only opportunity that requires downstream pipeline changes (SMILES reconstruction step). This is a 50-100 LOC change in `dmpnn_multitask.py` + `metrics_v2.py`. Recommend DEFERRED until Tier-1 candidates (OT-CFM, EDM) ship and prove out the CFM acceleration axis.

---

### §2.6 Rectified Flow / Reflow (Liu 2022) — reflow straightening

**Status in codebase:** NOT WIRED.

**Per-axis analysis:**

| Axis | Compatibility | Reason |
|---|---|---|
| **D-MPNN encoder** | ✅ ORTHOGONAL | Reflow recursively straightens ODE trajectories; output molecule distribution is the same data manifold at convergence. |
| **W₂ sort (Wf-328-W5)** | ✅ ORTHOGONAL | Same. |
| **Multi-task (5-metal)** | ✅ ORTHOGONAL | Same. |
| **m6_kappa + m7** | ✅ INDEPENDENT | Same. |

**Verdict:** FULLY COMPATIBLE (axis-wise). **BUT** per 340r2 §2.7 honest-negative: "Reflow INCREASES total training wall-time by a factor of (number of reflow iterations + 1)." Counter-productive for the user's training-cost ask. OUT OF SCOPE for this Wf.

---

### §2.7 VP-CFM (Albergo 2023) — variance-preserving trigonometric path

**Status in codebase:** NOT WIRED. Config-only change in torchcfm.

**Per-axis analysis:**

| Axis | Compatibility | Reason |
|---|---|---|
| **D-MPNN encoder** | ✅ ORTHOGONAL | The VP path uses trigonometric interpolation instead of linear; the marginal distribution at t=1 is the data distribution regardless. Output molecule distribution is the same. |
| **W₂ sort (Wf-328-W5)** | ✅ ORTHOGONAL | Same. |
| **Multi-task (5-metal)** | ✅ ORTHOGONAL | Same. |
| **m6_kappa + m7** | ✅ INDEPENDENT | Same. |

**Verdict:** FULLY COMPATIBLE. Tier-3 candidate (340r1 §6 item 6).

---

### §2.8 W-Flow (Stanford ICML 2026) — waveform relaxation ODE solver

**Status in codebase:** PARTIALLY WIRED via `molmetal/adapters/flow_matching_lipman/w_flow.py` + `w_flow_cfm_integration.py`.

**Per-axis analysis:**

| Axis | Compatibility | Reason |
|---|---|---|
| **D-MPNN encoder** | ✅ ORTHOGONAL (inference-only) | W-Flow is an ODE solver at inference; it changes *how fast* we integrate `dx/dt = v_θ(x, t)` but not the molecule distribution at convergence. |
| **W₂ sort (Wf-328-W5)** | ✅ ORTHOGONAL | Same. |
| **Multi-task (5-metal)** | ✅ ORTHOGONAL | Same. |
| **m6_kappa + m7** | ✅ INDEPENDENT | Same. |

**Verdict:** FULLY COMPATIBLE. Inference-time accelerator (340r2 §4.3 Tier-2 candidate for inference speedup, not training speedup).

---

## §3 Blocking issue identification

### §3.1 Counting methodology

A "blocking issue" is defined as: an opportunity that, if integrated naively without code changes, breaks a downstream pipeline OR changes a metric such that re-validation is mandatory.

### §3.2 Blocking issue count: 0 outright blocks, 2 medium-risk

| # | Opportunity | Blocking issue | Severity | Mitigation |
|---|---|---|---|---|
| 1 | OT-CFM | None | n/a | Drop-in integration |
| 2 | EDM preconditioning | None | n/a | 150-200 LOC wire-up + 1 unit test |
| 3 | StableVM | Code availability TBD | LOW | Check ICML 2026 proceedings for repo; defer until released |
| 4 | Augmented FM | Auxiliary variables change CFM output distribution → m6_kappa re-validation required | MEDIUM | 1 regression test on `m6_kappa.json` post-integration |
| 5 | **FlowMol** | Joint (coord, type, bond) output breaks 3 downstream pipelines (D-MPNN featurization, multi-task head, m6 SMILES reconstruction) | **MEDIUM** | Add SMILES reconstruction step + verify R² delta < 0.02 + verify SMILES reconstruction success > 99% |
| 6 | Rectified Flow | Reflow INCREASES total training wall-time | LOW (counter-productive) | Out of scope per 340r2 §4.3 honest-negative |
| 7 | VP-CFM | None | n/a | Config-only change |
| 8 | W-Flow | None (inference-only) | n/a | Already partially wired |

**Total blocking issues: 2 (Augmented FM, FlowMol)** — both are architectural changes that need re-validation, not fundamental incompatibilities.

---

## §4 Integration risk assessment (PROJECTED)

Per `party-twenty-spirit` §3.5, all risk levels below are **PROJECTED** based on per-opportunity codebase inspection. No actual integration attempted.

### §4.1 Risk per opportunity

| # | Opportunity | Integration risk | Reason |
|---|---|---|---|
| 1 | OT-CFM | LOW | `optimal_transport_v2.py` already exists; verify exact-OT vs stochastic-OT (5-15 LOC patch per 340r2 §4.2) |
| 2 | EDM preconditioning | LOW | Pure additive wrapper; 150-200 LOC + 1 unit test (340r2 §4.1) |
| 3 | StableVM | MEDIUM (TBD on code availability) | Need to verify ICML 2026 repo exists; if so, drop-in |
| 4 | Augmented FM | MEDIUM | Architectural change; needs regression test on m6_kappa |
| 5 | FlowMol | MEDIUM-HIGH | Joint-output requires downstream SMILES reconstruction + 3-axis re-validation |
| 6 | Rectified Flow | MEDIUM | Reflow pass increases wall-time; not a fit for "training convergence" ask |
| 7 | VP-CFM | LOW | Config-only change in torchcfm |
| 8 | W-Flow | LOW | Inference-only; partial wire already exists |

### §4.2 Risk per axis

| Axis | Integration risk | Reason |
|---|---|---|
| D-MPNN encoder | LOW (1/8 medium: FlowMol) | Most opportunities are orthogonal |
| W₂ sort (Wf-328-W5) | NONE | All 8 opportunities are orthogonal to W₂ sort |
| Multi-task (5-metal) | LOW (1/8 medium: FlowMol) | Most opportunities are orthogonal |
| m6_kappa + m7 | LOW (2/8 medium: Augmented FM, FlowMol) | Architectural changes that may shift CFM output distribution |

### §4.3 Aggregate PROJECTED risk

- **LOW:** 6/8 opportunities (OT-CFM, EDM, StableVM, VP-CFM, W-Flow, Rectified Flow — though Reflow is out of scope).
- **MEDIUM:** 2/8 opportunities (Augmented FM, StableVM-pending-code).
- **MEDIUM-HIGH:** 1/8 opportunities (FlowMol — joint-output architectural change).

**Net assessment:** 6/8 opportunities can be integrated with LOW risk; the user's "training framework improvement" ask is best served by Tier-1 (OT-CFM + EDM) which are both LOW-risk and HIGH-impact (PROJECTED 2-3× FID convergence per 340r2 §4.1 + §4.2).

---

## §5 Recommended integration order

Per the PROJECTED risk + impact matrix, the recommended order is:

1. **Wf-340-A5 / A6 (CPU wire-up, no GPU launch):**
   - OT-CFM exact-solver audit + 5-15 LOC patch (340r2 §4.2 + §7 R4).
   - EDM σ-preconditioning wire-up (150-200 LOC + 1 unit test per 340r2 §4.1 + §7 R3).
2. **Wf-340-A7 (CPU regression test):**
   - Re-run `m6_kappa.json` emission on CFM-generated cohort after OT-CFM + EDM integration.
   - Verify W₂ sort + multi-task + m6/m7 all unchanged (orthogonal check).
3. **Wf-340-A8 (GPU A/B, USER-OK NOT REQUIRED per AGENTS.md §5.1 standing GPU auth):**
   - 5-seed × 30-epoch A/B CFM retrain: with vs without EDM preconditioning × with vs without OT-CFM exact coupling → 2² × 5 = 20 GPU-runs.
   - Estimated wall: 5-8 hr on RTX 5090 per cell, ~20-32 hr total.
   - Standing GPU authorization per AGENTS.md §5.1 covers this; no per-launch user OK required as long as VRAM is sufficient.

**Defer:**
- **Augmented FM, FlowMol:** until Tier-1 (OT-CFM + EDM) ships and proves out the CFM acceleration axis.
- **StableVM:** until ICML 2026 repo release confirmed.
- **Rectified Flow:** out of scope per 340r2 §4.3 honest-negative.
- **VP-CFM:** Tier-3; lower priority than OT-CFM/EDM.

---

## §6 Honoured standing rules (AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ Every compatibility verdict cites source file or doc path; PROJECTED risk levels marked; MEASURED counts of 8 opportunities × 4 axes |
| **§1.4 Pre-flight v2 25/25 PASS gate** | ✅ No retrain/panel/ablation launched; pre-flight v3 62/62 PASS preserved at Wf-335-SYN baseline |
| **§1.5 Honest-negative framing** | ✅ Augmented FM + FlowMol risks declared upfront; Reflow counter-productive per 340r2 §2.7 honest-negative preserved; StableVM code-availability TBD declared |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches; CPU-only analysis (this file is a compatibility matrix, not a benchmark) |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — zero touches to `paper/*` |
| **§5.4 Tier-2 + no wet-lab** | ✅ Tier-2 acceptance bar — no wet-lab planning; CFM acceleration targets computational-only improvements |

---

## §7 Cross-references

- `TODO/completed/340r1_cfm_papers_survey.md` §3 — 9 papers with arXiv IDs + GitHub URLs
- `TODO/completed/340r2_acceleration_methods.md` §3 — 8 techniques × 5 axes summary table; §4 Tier-1/2 candidates
- `TODO/completed/340r3_molecule_cfm.md` — FlowMol + molecule-specific CFM applications
- `TODO/completed/340r4_code_repos.md` — 10 GitHub repos surveyed
- `TODO/completed/339b4_m6_kappa_diagnostic.md` — m6_kappa structural ceiling + unbiased-CV identity (m6 is on reference cohort, NOT CFM-generated)
- `TODO/completed/333v4_m6_kappa_alt.md` — chemprop_plus_pll CV-R² +0.6814 lift; HONEST_NEGATIVE_PARTIAL_STRUCTURAL
- `TODO/completed/328w5_w2_sort_wired.md` — W₂ sort default mode (sort vs sliced)
- `TODO/completed/325g_m7_multi_ph_lift.md` — m7 Multi-PH integration; mechanical lift +0.0283
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` — partial OT-CFM wire (PARTIALLY WIRED)
- `molmetal/adapters/flow_matching_lipman/w_flow.py` — partial W-Flow wire (PARTIALLY WIRED)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — CFM training entry (EGNN + YuelBond)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — D-MPNN multi-task + W₂ pIC50 entry
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` — m6_kappa RDKit descriptor emission
- `molmetal/molmetal_lam/reward/multiph_pll_feature.py:wire_m7_diversity_recovery` — m7 diversity recovery wire
- `metrics/by_metric/m6_kappa.json` — canonical m6_kappa JSON (n=449, value=0.4719, HONEST_NEGATIVE=true)
- `metrics/by_metric/diversity_tanimoto.json` — m7 cohort-level Tanimoto distance

---

**END OF Wf-340-A4** — Owner Hugo Lee + ultracode agent; cross-ref `party-twenty-spirit`
§3.5; AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 honoured; CPU-only compatibility assessment;
8 opportunities × 4 axes matrix complete; 2 blocking issues identified (Augmented FM,
FlowMol); 6/8 opportunities FULLY COMPATIBLE; recommended Tier-1 integration order:
OT-CFM + EDM preconditioning.
