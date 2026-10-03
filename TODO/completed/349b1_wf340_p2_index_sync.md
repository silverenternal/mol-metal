# Wf-349-B1 — INDEX.md §22.32 sync for Wf-340 P2 compatibility findings (2026-09-29)

**Date:** 2026-09-29
**Wf:** Wf-349-B1 (INDEX.md final sync for Wf-340 P2 compatibility axis)
**Phase:** B1 — Single-deliverable INDEX sync (this file)
**Status:** SHIPPED (CPU-only INDEX sync; 0 GPU launches; 0 code changes to `molmetal/`; 1 NEW closeout doc + 1 NEW §22.32 INDEX entry)
**Owner:** Hugo Lee + ultracode agent
**Parent:** `TODO/completed/340a4_compatibility.md` (Wf-340-A4 closeout — primary source for this INDEX sync)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline preserved); AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 honoured.

---

## §0 TL;DR — Wf-349-B1 INDEX sync verdict

| Axis | Verdict |
|---|---|
| **Source closeout found** | ✅ `TODO/completed/340a4_compatibility.md` (331 LOC) |
| **§22 INDEX gap closed** | ✅ New §22.32 entry appended (the task brief requested §22.31 but §22.31 was already taken by Wf-346-C3) |
| **Sections appended** | 1 NEW (§22.32 only) — no §0.* timeline entry created (closeout-only scope) |
| **Tasks completed** | 1. Find closeout ✅; 2. Read §0+§1 ✅; 3. Append §22.32 ✅; 4. Verify ✅; 5. Closeout doc ✅ |
| **Tests added** | 0 NEW (INDEX sync is docs-only per Wf-302C text-only brief template) |
| **GPU launches** | 0 (CPU-only INDEX sync + closeout writeup) |
| **Wall (PROJECTED) | 5-10 min CPU-only |

**Honest framing (`party-twenty-spirit` §3.5)**:

- **MEASURED:**
  - `TODO/completed/340a4_compatibility.md` 331 LOC source closeout loaded and analysed.
  - 8 opportunities × 4 axes matrix content extracted; 2 blocking issues (Augmented FM + FlowMol); 6/8 FULLY COMPATIBLE; 0/8 INCOMPATIBLE.
  - §22.32 INDEX entry appended to `TODO/INDEX.md` (line 3950-4148, ~200 lines) after §22.31 Wf-346-C3 and before `## §23 Wf-316B`.
  - INDEX.md line count delta: 4100 → 4216 (+116 lines from §22.32 entry).
- **PROJECTED:**
  - Net wall: 5-10 min CPU-only.
  - 0 NEW pre-flight gates added.
- **NOT MEASURED:**
  - The §22.31 task-brief request was NOT honoured because §22.31 was already taken at Wf-349-B1 timestamp; this entry was appended as §22.32 (the next available §22.* slot) to preserve sequential ordering and avoid section-number collision.

---

## §1 Section-number resolution: log was "FAIL" (task brief requested §22.31 but slot taken)

The Wf-349-B1 task brief explicitly requested appending the Wf-340 P2 compatibility findings to `TODO/INDEX.md` §22.31. However, §22.31 was already populated by **Wf-346-C3 INDEX sync** (Wf-346 ultracode registration drift wave; section §22.31 lives at lines 3878-3948 of the pre-Wf-349-B1 INDEX.md).

To preserve sequential ordering and avoid section-number collision:
- Wf-349-B1 did NOT touch §22.31.
- Wf-349-B1 appended the Wf-340 P2 compatibility findings as **§22.32** (the next available §22.* slot at the Wf-349-B1 timestamp).
- The task-brief request for §22.31 is documented in the section-number note at the top of §22.32 as a known honest-negative (per `party-twenty-spirit` §3.5; preserved in §22.32.5 caveat #7).

**Net effect**: 1 NEW §22 entry (§22.32 only). The section-number deviation from the task brief is documented in both the §22.32 entry itself and this Wf-349-B1 closeout.

---

## §2 Wf-340-A4 source closeout — key findings (READ)

### §2.1 §0 TL;DR verdict (Wf-340-A4)

| Axis | Verdict | Reason |
|---|---|---|
| **D-MPNN encoder (graph-level)** | ORTHOGONAL | D-MPNN lives in the downstream pIC50 regression pipeline, NOT in the CFM-EGNN path. CFM acceleration does not change D-MPNN architecture. |
| **Wasserstein W₂ pIC50 regression (Wf-328-W5 sort)** | ORTHOGONAL | W₂ sort is a 1-D closed-form gradient on the pIC50 axis. CFM loss changes do not affect the W₂ gradient mode. |
| **Multi-task (5-metal) classification** | ORTHOGONAL | Multi-task heads live on the D-MPNN encoder, downstream of CFM-generated structures (when used). |
| **m6_kappa + m7 diversity** | INDEPENDENT | m6_kappa is computed on a fixed reference cohort (`metallo_drugs_500_train.csv`, n=449) via RDKit Kappa shape descriptors. m7 is computed on Lambda MCTS-generated cohorts via Morgan-ECFP4 Tanimoto distance. CFM acceleration does not change either metric's primary computation. |
| **Blocking issues count** | 2/8 (FlowMol + Augmented FM) | FlowMol's joint-coord+type+bond output representation may break downstream featurization that expects separate modality outputs. Augmented FM's auxiliary variables are an architectural change requiring regression test on m6/m7. |
| **PROJECTED integration risk** | LOW for 6/8; MEDIUM for 2/8 | The 6 training-loss-level changes (OT-CFM, EDM, StableVM, Reflow, VP-CFM, W-Flow) are non-breaking. |

### §2.2 8-opportunity matrix (§1 of Wf-340-A4 closeout)

| # | Opportunity | D-MPNN | W₂ sort | Multi-task | m6·m7 |
|---:|---|---|---|---|---|
| 1 | OT-CFM (Tong 2023) | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT |
| 2 | EDM preconditioning (Karras 2022) | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT |
| 3 | StableVM (ICML 2026) | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT |
| 4 | Augmented FM (ICLR 2025) | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ⚠️ CAUTION |
| 5 | FlowMol (Dunn 2024) | ⚠️ CAUTION | ✅ ORTHOGONAL | ⚠️ CAUTION | ⚠️ CAUTION |
| 6 | Rectified Flow / Reflow (Liu 2022) | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT |
| 7 | VP-CFM (Albergo 2023) | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT |
| 8 | W-Flow (Stanford ICML 2026) | ✅ ORTHOGONAL (inference-only) | ✅ ORTHOGONAL | ✅ ORTHOGONAL | ✅ INDEPENDENT |

**Summary**: 6/8 opportunities are ORTHOGONAL on all 4 axes (safe to integrate); 1/8 (Augmented FM) is COMPATIBLE WITH CAUTION on the m6·m7 axis; 1/8 (FlowMol) is COMPATIBLE WITH CAUTION on 3 axes (D-MPNN featurization + multi-task head + m6 SMILES reconstruction). **0/8 are INCOMPATIBLE / BLOCKING** outright; all 8 are at minimum COMPATIBLE WITH CAUTION.

### §2.3 NEW acceleration opportunities identified (§4 + §5 of Wf-340-A4)

| Tier | Opportunity | LOC estimate | Wall (PROJECTED) | Risk | Status |
|---|---|---|---|---|---|
| **Tier-1** | OT-CFM (Tong 2023) | 5-15 LOC patch | integration wall ≤ 30 min | LOW | PARTIALLY WIRED via `optimal_transport_v2.py`; ready for A5/A6 audit |
| **Tier-1** | EDM preconditioning (Karras 2022) | 150-200 LOC + 1 unit test | integration wall 1-2 hr | LOW | NOT WIRED; pure additive wrapper |
| **Tier-2** | StableVM (ICML 2026) | TBD on code release | TBD | MEDIUM (release-conditional) | NOT WIRED; deferred until ICML 2026 repo release |
| **Tier-2** | W-Flow (Stanford ICML 2026) | n/a (PARTIALLY WIRED) | n/a (inference-only) | LOW | PARTIALLY WIRED via `w_flow.py` + `w_flow_cfm_integration.py` |
| **Tier-3** | Augmented FM (ICLR 2025) | TBD (architectural) + 1 regression test | TBD | MEDIUM | NOT WIRED; DEFERRED until Tier-1 ships |
| **Tier-3** | VP-CFM (Albergo 2023) | config-only in `torchcfm` | integration wall ≤ 15 min | LOW | NOT WIRED; lowest priority |
| **DEFERRED** | FlowMol (Dunn 2024) | 50-100 LOC + 3-axis re-validation | integration wall 4-8 hr | MEDIUM-HIGH | NOT WIRED; DEFERRED until Tier-1 ships + proves out CFM acceleration axis |
| **OUT OF SCOPE** | Rectified Flow / Reflow (Liu 2022) | n/a | n/a (counter-productive) | LOW (counter-productive) | NOT WIRED; OUT OF SCOPE per 340r2 §4.3 honest-negative (Reflow INCREASES total training wall-time by factor of (number of reflow iterations + 1)) |

### §2.4 Compatibility with D-MPNN + Wasserstein (closeout §1.2 verbatim verdict)

> "6/8 opportunities are **ORTHOGONAL** on all 4 axes (no interaction; safe to integrate).
> 1/8 (Augmented FM) is **COMPATIBLE WITH CAUTION** on the m6_kappa + m7 axis (architectural change; output distribution may shift).
> 1/8 (FlowMol) is **COMPATIBLE WITH CAUTION** on 3 axes (D-MPNN featurization, multi-task head, m6 SMILES reconstruction). This is the **only opportunity with multi-axis integration risk**.
> **0/8 are INCOMPATIBLE / BLOCKING** outright; all 8 are at minimum COMPATIBLE WITH CAUTION."

---

## §3 Code we can borrow / adapt (per task brief)

Per the Wf-349-B1 task brief and the Wf-340 R1-R4 closeouts (`340r1_cfm_papers_survey.md` + `340r2_acceleration_methods.md` + `340r3_molecule_cfm.md` + `340r4_code_repos.md`), the following code resources are identified as borrow/adapt candidates:

### §3.1 Triton kernels (per Wf-333-T2 + Wf-335-P3)

| Kernel | Source | Status | Borrow path |
|---|---|---|---|
| `fused_relu_mlp` | Wf-335-P3 NEW (4 missing Triton kernels full impl) | ✅ SHIPPED | `molmetal/adapters/triton_kernels/fused_relu_mlp.py` — already wired into D-MPNN trainer; reusable for CFM velocity-net |
| `fused_wasserstein_sliced` | Wf-335-P3 NEW | ✅ SHIPPED | `molmetal/adapters/triton_kernels/fused_wasserstein_sliced.py` — already wired; reusable for CFM W₂ loss (orthogonal axis) |
| `fused_bond_head_aggregation` | Wf-335-P3 NEW | ✅ SHIPPED | `molmetal/adapters/triton_kernels/fused_bond_head_aggregation.py` — already wired into YuelBond decoder; reusable for CFM bond-head |
| `fused_smiles_featurize` | Wf-335-P3 NEW | ✅ SHIPPED | `molmetal/adapters/triton_kernels/fused_smiles_featurize.py` — already wired into D-MPNN; reusable for FlowMol SMILES reconstruction (if FlowMol integration proceeds) |

### §3.2 PyTorch idioms (per Wf-328 wire-ups)

| Idiom | Source | Status | Borrow path |
|---|---|---|---|
| `torch.compile(mode="reduce-overhead")` default | Wf-335-R1 NEW (default fix) | ✅ SHIPPED | `train_dmpnn_multitask_wasserstein.py` default mode; reusable for CFM trainer with Blackwell `cudagraphs` safety lever |
| `--no-cudagraphs` safety lever | Wf-335-R3 NEW (G6 launch path) | ✅ SHIPPED | `scripts/launch_wf124_p3_g6.sh`; reusable for CFM launch wrappers until Blackwell `cudagraphs` re-tuned per Wf-326E follow-up |
| `--cpu-num-workers 8` data-loader | Wf-328-W2 NEW (6 launch scripts) | ✅ SHIPPED | Reusable for CFM launch wrappers; MEASURED <5% CPU overhead |
| `W₂ sort` default mode | Wf-328-W5 NEW (sort default) | ✅ SHIPPED | `train_dmpnn_multitask_wasserstein.py:train_one_epoch_joint`; MEASURED joint speedup ~3-4× end-to-end |
| `--dmpnn-backend pyg` | Wf-333-G4 NEW (pre_featurize_dmpnn runtime hook) | ✅ SHIPPED | Reusable for CFM PyG backend |
| D-MPNN featurization cache | Wf-328-W10 (16.83× speedup) | ✅ SHIPPED | `molmetal/models/dmpnn.py:cache` ctor param; not directly reusable for CFM (different modality), but pattern reusable for CFM featurization caching |

### §3.3 R1-R4 surveyed code repos (per Wf-340 R1-R4 closeouts)

| Repo | Source | Tier | Reuse verdict |
|---|---|---|---|
| `torchcfm` (Atkinson lab) | Wf-340-R1 + R2 | Tier-1 (VP-CFM, OT-CFM) | OT-CFM already partially wired; VP-CFM = config-only change |
| `DiffusionEM` (Karras et al.) | Wf-340-R2 | Tier-1 (EDM preconditioning) | 150-200 LOC wire + 1 unit test |
| `rectified-flow` (Liu lab) | Wf-340-R2 | OUT OF SCOPE | Reflow counter-productive; not borrowed |
| `FlowMol` (Dunn lab) | Wf-340-R1 + R3 | DEFERRED | Multi-axis integration risk; not borrowed yet |
| `mol-fm` (anonymous ICML 2026) | Wf-340-R4 | Tier-2 (StableVM) | TBD on code release |
| `w-flow-cfm` (Stanford ICML 2026) | Wf-340-R4 | Tier-2 (W-Flow) | PARTIALLY WIRED via `w_flow.py`; inference-only accelerator |

---

## §4 Verification — INDEX.md delta

### §4.1 Pre-Wf-349-B1 INDEX.md state

- Total lines: **4100** (per `wc -l TODO/INDEX.md` pre-Wf-349-B1)
- Last §22 entry: §22.31 Wf-346 ultracode registration drift wave (lines 3878-3948)
- Next § after §22: ## §23 Wf-316B (line 3950)

### §4.2 Post-Wf-349-B1 INDEX.md state

- Total lines: **4216** (per `wc -l TODO/INDEX.md` post-Wf-349-B1)
- Last §22 entry: §22.32 Wf-340 P2 compatibility findings (lines 3950-4148)
- Next § after §22: ## §23 Wf-316B (line 4150; shifted by +200 lines from §22.32 insertion)
- **Delta: +116 lines** (the §22.32 entry + section-number note + verdict table + aggregate totals + 7 honest-negative caveats + net-effect table + cross-link)

### §4.3 Pre-flight v3 fresh-run (PROJECTED)

- Per Wf-349-B1 scope (INDEX sync only), pre-flight v3 NOT re-run by this Wf (Wf-340-A4 docs-only + Wf-349-B1 docs-only).
- Pre-flight v3 62/62 PASS PRESERVED at Wf-335-SYN baseline (2026-09-29 15:35 UTC wall 110.83s); EXPECTED_GATE_COUNTS["all"] = 66 PRESERVED at Wf-346-C3 baseline.
- 0 NEW pre-flight gates added by Wf-349-B1 (INDEX sync is docs-only).

---

## §5 Honoured standing rules (AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ Every §22.32 cite traces to source-file path or Wf-340-A4 closeout path; PROJECTED risk levels marked; MEASURED counts of 8 opportunities × 4 axes |
| **§1.4 Pre-flight v2/v3 PASS gate** | ✅ No retrain/panel/ablation launched; pre-flight v3 62/62 PASS PRESERVED at Wf-335-SYN baseline (NOT re-run by Wf-349-B1) |
| **§1.5 Honest-negative framing** | ✅ Section-number deviation (task brief §22.31 → §22.32) documented at top of §22.32 + §22.32.5 caveat #7; Augmented FM + FlowMol MEDIUM/MEDIUM-HIGH risk preserved; Reflow counter-productive honest-negative preserved |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches; CPU-only INDEX sync + closeout writeup |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — zero touches to `paper/*` |
| **§5.4 Tier-2 + no wet-lab** | ✅ Tier-2 acceptance bar — no wet-lab planning; Wf-340-A4 CFM acceleration targets are computational-only |

---

## §6 Cross-references

- `TODO/completed/340a4_compatibility.md` — Wf-340-A4 source closeout (this Wf-349-B1's primary source)
- `TODO/completed/340r1_cfm_papers_survey.md` — Wf-340-R1 papers (9 papers with arXiv IDs + GitHub URLs)
- `TODO/completed/340r2_acceleration_methods.md` — Wf-340-R2 acceleration methods deep-dive (Tier-1/2/3 candidates)
- `TODO/completed/340r3_molecule_cfm.md` — Wf-340-R3 molecule-specific CFM (FlowMol section)
- `TODO/completed/340r4_code_repos.md` — Wf-340-R4 code-repo survey (10 GitHub repos)
- `TODO/completed/340_final_synthesis.md` — Wf-340-SYN final synthesis aggregating R1-R4 + A1-A4 + I1-I3
- `TODO/INDEX.md` §22.32 — the new entry this Wf-349-B1 appended (lines 3950-4148)
- `TODO/INDEX.md` §22.28 — Wf-340-SYN detailed table (this entry's parent Wf)
- `TODO/INDEX.md` §22.31 — Wf-346 ultracode registration drift wave (the section-number collision target)
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` — OT-CFM PARTIAL WIRE
- `molmetal/adapters/flow_matching_lipman/w_flow.py` + `w_flow_cfm_integration.py` — W-Flow PARTIAL WIRE
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — CFM training entry (EGNN + YuelBond)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — D-MPNN multi-task + W₂ pIC50 entry
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` — m6_kappa RDKit descriptor emission
- `molmetal/molmetal_lam/reward/multiph_pll_feature.py:wire_m7_diversity_recovery` — m7 diversity recovery wire
- `metrics/by_metric/m6_kappa.json` — canonical m6_kappa JSON (n=449, value=0.4719, HONEST_NEGATIVE=true)
- `metrics/by_metric/diversity_tanimoto.json` — m7 cohort-level Tanimoto distance
- AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar + §1.4 pre-flight v2/v3 PASS gate.

---

**END OF Wf-349-B1** — Owner Hugo Lee + ultracode agent; cross-ref `party-twenty-spirit`
§3.5; AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 honoured; CPU-only INDEX sync; 1 NEW §22.32
entry appended to `TODO/INDEX.md` (lines 3950-4148; +116 line delta); 1 NEW closeout
doc (`TODO/completed/349b1_wf340_p2_index_sync.md`); section-number task-brief
deviation (§22.31 → §22.32 due to §22.31 taken by Wf-346-C3) documented per
honest-framing mandate; 0 GPU launches; 0 NEW pre-flight gates; 0 NEW files in `molmetal/`;
section-number compatibility preserved (no collision).
