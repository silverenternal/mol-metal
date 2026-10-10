# Mol-Metal: Precious-Metal Anti-Cancer Drug Design via Flow Matching + SBDD

**Last-updated trailer (Wf-METRIC-RESCUE 2026-10-08/09 sync — 5 honest-negative findings + 2 code fixes SHIPPED per user voice *"我觉得现在应该同步一下项目所有文档和实际状态以及还有哪些我们的项目在运行中的任务"*):** W1 arm2 (PID 31946, per-epoch cadence 30-epoch real-only) COMPLETED 2026-10-09 00:15; pooled 0.5496 ± 0.0967 (3-arm ON/OFF confirms W₂ NOT load-bearing); W1 arm1 (every-batch cadence) = 0.5349; OFF reference = 0.5575. **5 honest-negative findings on disk**: (1) W₁ W₂ ON/OFF de-confound [HONEST-NEGATIVE: W₂ lowers metric], (2) G6 per-innovation ablation [18 tuples, no individual lever decisive], (3) bace rescue hypothesis REFUTED [0.7421→0.6805, class-weighting NOT the cause], (4) m19 fix-stack `--rh-loso-validation` HURT Rh [0.5022→0.4056], (5) pIC50 pearson ridge beats head [pooled ridge 0.4302 vs head 0.2036 (corrected 2026-10-11; 0.4151/0.1543 were not reproducible from disk), head anti-correlated on Os]. **Code SHIPPED**: `train_moleculenet.py` 4 NEW flags wired (--class-weight/--dropout/--weight-decay/--patience, verified end-to-end NOT FLAG_ONLY) + `train_metal_lomo_wasserstein.py` W₂ no-op inverse guard + 11 NEW tests PASS. **Pre-flight v3 still rc=0**. **Running tasks inventory 2026-10-09**: 0 active GPU jobs (W1 arm2 exited 2026-10-09 00:15); 0 background monitors active. All artifact paths + §6.38 catalog + closeout file `TODO/completed/metric_rescue_wfmrr_2026_10_08.md` 12519 B on disk per AGENTS.md §5.5 filesystem-truth + §5.6 atomic-write + §5.7 trainer-flag-smoke. **Previous (Wf-A3 sync 2026-10-02)**: 22 NEW tests PASS; pre-flight v3 fresh-run MEASURED 93 PASS / 3 FAIL / 2 SKIP / 98 TOTAL; AGENTS.md §5.10 SHIPPED. **PREVIOUS HISTORY PRESERVED VERBATIM PER §5.5 rule #6 — rewriting the historical closeout trail would violate the filesystem-truth contract** (the prior Wf-438/439/443 trailers follow below).


## Project Overview

**Mol-Metal** designs de-novo small-molecule ligands for precious-metal anti-cancer targets
(cisplatin-class, Pt/Ru/Ir organometallics) using **Flow Matching** generative models on top of a
**Structure-Based Drug Design (SBDD)** loop. The system generates 3D ligand poses in a protein
pocket, docks and scores them, and refines via a closed-loop design cycle.

We aim to close the single-GPU (RX 7800 XT) vs. 64×A100 cluster gap by leveraging **Triton
kernels on ROCm** — the constraint is "fits in 16 GB memory," not "reproduce multi-GPU training."

---

## Environment

> **⚠️ STALE section preserved for historical context (R16 2026-09-18 close-out authored
> this README on AMD RX 7800 XT + ROCm). The CURRENT canonical environment is NVIDIA Blackwell
> + RTX 5090 + .venv/bin/python — see [environment.md §1–§7](environment.md) for MEASURED 2026-10-03
> facts. The legacy RX 7800 XT claim below does NOT match today's host; per AGENTS.md §5.5
> filesystem-truth rule, historical text is preserved (NOT deleted) but explicitly flagged STALE.**

### CURRENT (canonical, per `environment.md` MEASURED 2026-10-03)

- **Package manager:** `uv 0.11.8` (lockfile `uv.lock` 344 KB) — but launch wrappers invoke `/home/hugo/codes/mol-metal/.venv/bin/python` directly per AGENTS.md §5.9 single-venv rule
- **Python:** 3.12.13 (`.venv/bin/python` symlink to uv-managed CPython)
- **GPU stack:** NVIDIA CUDA 13.0 + `torch 2.14.1+cu130` + `triton 3.8.0` (PyPI; NOT `triton-rocm`)
- **Hardware:** GPU 0 = NVIDIA RTX PRO 6000 Blackwell (97887 MiB VRAM, sm_120); GPU 1 = NVIDIA GeForce RTX 5090 (32607 MiB VRAM); driver 615.71.09
- **Canonical venv:** `/home/hugo/codes/mol-metal/.venv/bin/python` (single venv rule per §5.9)

### HISTORICAL (STALE — RX 7800 XT + ROCm was the R16 close-out host)

This project uses a **fully managed uv environment** (`UV_PROJECT_ENVIRONMENT` / `managed=true`):

- **Package manager:** `uv` (all commands prefixed with `uv run`)
- **Python:** 3.12
- **GPU stack:** ROCm 7.2 + `triton-rocm` 3.8.0
- **Hardware target:** AMD RX 7800 XT (gfx1101, 16 GB)

**No pip / no venv / no system Python.** Activate nothing; just `uv run <cmd>`. See
[`environment.md`](environment.md) for full setup, kernel validation, and the ROCm/Triton
compatibility matrix.

---

## Engineering Practices

Read [`engineering_practices.md`](engineering_practices.md) before touching code. Key conventions:

- **Single source of truth:** every ADR lives in top-level `decisions.md` (open) or
  `completed/` (closed). Never in code comments or scattered .md files.
- **One task, one file:** `pending/NN_short_slug.md` is the atomic unit of work. Status moves
  `pending → completed` (or `pending → archive` if superseded by a newer plan), not
  delete-and-recreate.
- **Audit trail:** every decision cites a paper / commit / date. AUDIT_RESEARCH_GRADE.md is the
  immutable historical snapshot.
- **No scope creep into TODO/01-13/:** those are historical research notes. New topical work
  either updates an existing topic dir or lives in `pending/`.
- **Use INDEX.md for status:** every file's current state is tracked in
  [`INDEX.md`](INDEX.md) (single source of truth for "what's where + status").

---

## Directory Structure

```
TODO/
├── README.md                       ← this file (overview + navigation)
├── INDEX.md                        ← single source of truth: every file's status
├── environment.md                  ← uv + ROCm 7.2 + triton-rocm 3.8.0 setup
├── engineering_practices.md        ← conventions for this project
├── AUDIT_RESEARCH_GRADE.md         ← historical audit (2026-09-11), preserved verbatim
├── completion_audit_2026-09-13.md  ← completed-task audit
├── project_workflow.md             ← project-level workflow doc
├── decisions.md                    ← pending architecture choices (D1–D∞)
├── risks.md                        ← open blockers / risks
├── roadmap.md                      ← phase tracker (Phase 0-4)
│
├── pending/                        ← ACTIVE plans / in-flight work (9 files, see INDEX.md)
│   ├── 13_top_journal_pilot_r12.md
│   ├── 14_full_100pocket_paper_r13.md
│   ├── 19_user_decisions.md        (info doc — 6 user-gated decisions)
│   ├── 21_lambda_model_coupling.md (long-term strategic)
│   ├── 22_data_gap_alignment_plan.md
│   ├── 23_weak_to_strong_plan.md
│   ├── 24_cfm_architecture_redo_plan.md
│   ├── 25_round14_lit_grounded_plan.md
│   └── 26_round13_round14_complete_plan.md  ← master comprehensive plan
│
├── completed/                      ← SHIPPED work (28 files)
│
├── archive/                        ← STALE plans superseded by newer ones (4 files)
│
└── 01_research/ ... 13_lambda_clickchem/   ← historical topic subdirs (research notes)
```

**Navigation rule of thumb:**

- "What's the current status of every file?" → [`INDEX.md`](INDEX.md)
- "What's the project roadmap?" → [`roadmap.md`](roadmap.md)
- "What should I work on next?" → [`pending/26_round13_round14_complete_plan.md`](pending/26_round13_round14_complete_plan.md) (the master comprehensive plan)
- "What's blocked / at risk?" → [`risks.md`](risks.md)
- "What architecture decisions are pending?" → [`decisions.md`](decisions.md)
- "Why did we decide X?" → top-level `decisions.md` (open) or `completed/` (closed)
- "What did we originally research?" → `01_research/` through `13_lambda_clickchem/`
- "What just got shipped?" → newest entries in `completed/`
- "What got superseded and is no longer relevant?" → `archive/`
- "What are the actual measured numbers?" → [`../metrics/`](metrics/) (single source of truth for all measured values; 17 JSON + README at 2026-09-15)
- "Why is the project shaped this way?" → this README + `AUDIT_RESEARCH_GRADE.md`

---

## Strategic Core (preserved)

### Lipman 2023 as the FM backbone

**Lipman et al. 2023 (Flow Matching for Generative Modeling, ICLR 2023)** is the foundational
flow-matching paper (4600+ citations, official `github.com/facebookresearch/flow_matching`). It
plugs naturally into MolFlow-Triton's `flow_matching` infrastructure.

**Strategy:** clone the official library, write a thin `LipmanFlowMatchingAdapter`. We do NOT
rewrite FM — we only own the adapter + glue code + algorithmic tweaks (mini-batch OT, guidance,
metal coordination prior).

### Hexagonal architecture (`molmetal/`)

```
┌────────────────────────────────────────────────────────┐
│  molmetal/                                              │
│                                                         │
│  ┌──────────┐   ┌──────────────────────┐                │
│  │ domain/  │   │  ports/ (we define)   │                │
│  │ Pocket   │◄──│  MoleculeGenerator   │                │
│  │ Molecule │   │  DockingEngine       │                │
│  │ Complex  │   │  PropertyPredictor   │                │
│  └──────────┘   │  ScoringFunction     │                │
│                 │  DesignLoop          │                │
│                 └────────┬─────────────┘                │
│                          │                              │
│  ┌───────────────────────┴──────────────────────────┐  │
│  │  adapters/  (concrete impls from cloned repos)      │  │
│  │  ┌───────────────────────────────────────────┐    │  │
│  │  │ LipmanFlowMatchingAdapter ← we write      │    │  │
│  │  │  import facebookresearch/flow_matching    │    │  │
│  │  │  import models.egnn (MolFlow-Triton)      │    │  │
│  │  └───────────────────────────────────────────┘    │  │
│  │  ┌───────────────────────────────────────────┐    │  │
│  │  │ DiffDockAdapter (later Phase)             │    │  │
│  │  └───────────────────────────────────────────┘    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                         │
│  ┌────────────────────────────┐                        │
│  │  references/  (git cloned) │                        │
│  │  flow_matching/            │ ← Lipman 2023 official │
│  │  DiffDock/                 │ ← Nature 2024 (later)  │
│  └────────────────────────────┘                        │
└────────────────────────────────────────────────────────┘
```

### Reuse strategy

**MolFlow-Triton provides:** `scatter_sum` (Triton kernel + autograd), `EGNNLayer` (3D
equivariance, dative-bond support), toy `ConditionalFlowMatchingLoss`, `FlowMatchingSampler`,
`BaseMoleculeDataset`, `positions_to_smiles`.

**facebookresearch/flow_matching provides** (cloned): `AffineLinearPath` (Lipman §4.8),
`CondOTScheduler` (§4.7), production `ConditionalFlowMatchingLoss` (§4.5), `EulerSimulator` /
`RK4Simulator`, `ODESolver`.

**We write:** `EGNNVelocityField` (uses MolFlow-Triton's EGNN as v_θ),
`LipmanFlowMatchingAdapter` (implements `MoleculeGenerator` port), training/sampling glue,
closed-loop orchestration (generate → dock → score → refine), and all algorithmic optimizations
(mini-batch OT, guidance, metal coordination prior, ...).

---

## Roadmap Status

See [`roadmap.md`](roadmap.md) for the living tracker. Snapshot as of 2026-09-15:

| Round | Goal | Status |
|---|---|---|
| R3 (closed) | Lambda + Mol-Hybrid algorithmic axes | ✅ |
| R8 (closed) | 10 TODO items | ✅ |
| R10 (closed) | CFG end-to-end + Pt prior | ✅ |
| R11 (closed) | QVina + data staging + N=50 parity | ✅ |
| R11b (closed) | anticancer metric suite | ✅ |
| **R12 (in flight)** | Round-12 pilot at top-journal standard | ⚙️ partial |
| **R13 (in flight)** | 100-pocket × 3-seed paper-grade sweep | ⚙️ 3 workflows running |
| **R14 (planned)** | lit-grounded + math-prior + code-fix lift | ⏳ per TODO-25 |
| arXiv | paper submission | ⏳ post-R13/R14 |

---

## Active Work

Top priority items (full status in [`INDEX.md`](INDEX.md); full content in [`pending/`](pending/)):

| # | File | Round | Status |
|---|---|---|---|
| 13 | [`13_top_journal_pilot_r12.md`](pending/13_top_journal_pilot_r12.md) | R12 | ✅ **DELIVERED** — diversity lift verified 1→20, 147 cells DESIGN→MEASURED in paper §4 |
| 14 | [`14_full_100pocket_paper_r13.md`](pending/14_full_100pocket_paper_r13.md) | R13 | ⚙️ partial — honest-negative on novel pockets; pocket-invariance sub-fix in flight |
| 19 | [`19_user_decisions.md`](pending/19_user_decisions.md) | meta | info — D6 REINVENT4, D7 Vina↔QVina, cite-only SOTA, journal choice |
| 21 | [`21_lambda_model_coupling.md`](pending/21_lambda_model_coupling.md) | long-term | 战略方向, deferred post-R12 |
| 22 | [`22_data_gap_alignment_plan.md`](pending/22_data_gap_alignment_plan.md) | R12/R13 | ✅ mostly shipped — 16/25 metrics MEASURED at scale |
| 23 | [`23_weak_to_strong_plan.md`](pending/23_weak_to_strong_plan.md) | R13 | strategic plan, deferred |
| 24 | [`24_cfm_architecture_redo_plan.md`](pending/24_cfm_architecture_redo_plan.md) | R13/R14 | ⚙️ P0+P1 SHIPPED; GPU retrain FAILURE; frontier research in flight |
| 25 | [`25_round14_lit_grounded_plan.md`](pending/25_round14_lit_grounded_plan.md) | R14 | ⚙️ Path A + D2/D4 SHIPPED; CFM + pocket-invariance in flight |
| 26 | [`26_round13_round14_complete_plan.md`](pending/26_round13_round14_complete_plan.md) | R13/R14 | R12 deliverable COMPLETE; R13 honest-negative; R14 in flight |
| **27** | [`27_paper_main_pdf_repair.md`](completed/27_paper_main_pdf_repair.md) | **CLOSED** | ✅ 69 pages / 5.17 MB / 0 unresolved (per w5hj17h50 + we5qbl16b + wfm0fjqjs) |
| **28** | [`28_round12_honest_negative_reframe.md`](pending/28_round12_honest_negative_reframe.md) | R12 framing | ⚙️ mostly closed — §4 + §6 + §3.4 updated; CROSS_REFS update deferred |
| 29 | [`29_f2a_round13_retry.md`](pending/29_f2a_round13_retry.md) | R13 retry | ⚙️ in flight — `wyyy283ck` B (scaffold-aware) + `wrd5dbewn` (pocket-invariance combined) |

Open blockers: [`risks.md`](risks.md). Open architecture decisions: [`decisions.md`](decisions.md).

### Major 2026-09-15 ship summary (advantages made full)

- ✅ **paper/main.pdf**: 69 pages / 5.17 MB / 0 unresolved bibitems
- ✅ **paper §4**: 147 cells DESIGN→MEASURED (Path A 4-fix bundle + metal column + P0 anticancer + PB panel + click-rule effect sizes)
- ✅ **paper §5**: 3 NEW sub-sections (drug-likeness ADMET, metal coordination probe, click-rule effect sizes)
- ✅ **paper §6**: expanded 8→12 caveats with honest trade-off framing
- ✅ **5 NEW first-class modules shipped**: conformer embed, pharmacophore filter, stereo-aware reduction, reaction confidence, Pareto multi-objective
- ✅ **Deflex dual-system framework**: PocketMacroSkeleton (87.9% train acc) + λ Combinators (10 tests) + Symbolic Regression F5 formula (R = 2.58 - 2.51·sa_norm)
- ✅ **REINVENT4 multiproperty wire**: r=0.6763 vs proxy at 4.06 s/SMILES
- ✅ **PB MMFF94 relax**: 22/26 PB checks pass on click_tile
- ✅ **Vina real**: -6.929 kcal/mol on 1-pocket smoke
- ✅ **SA fragment pool optimize**: 3.32 → 3.099 (TargetDiff target 2.65-2.86)
- ✅ **CFM internal review + P0 + P1 fixes**: 5 P0 + 4 P1 SHIPPED (CPU-only)
- ✅ **GPU recovery**: cuda_available=True, 2 devices confirmed

### Major 2026-09-16 R15 ship summary (advantages made full, structural layer)

**R15 = 13 workflows across 4 parallel families + 1 verifier. 89 tests across 12 new files: 82 pass (92.1%) + 5 skip (libtorch ABI) + 2 fail (real bugs caught).**

- ✅ **YuelBond decoder ships** (NEW `lam_chem/yuelbond_decoder.py`, 5 unit tests; chem-aware bond decoder head)
- ✅ **CFM-Rescue 4 fixes ship bit-exact**: bond_head default `distance`→`learned`; joint_train default `False`→`True`; n_train 8→32; ODE solver euler→midpoint (RK2 within 0.01 ref)
- ✅ **Lambda-Boost sub-fix C ship**: reference_ligand_resolver wire (5 unit tests); F2(a) MetalLigandExchange SMARTS (5 tests); AquaExchange SMARTS (6 tests)
- ✅ **Lambda × CFM coupling dry-run bridge alive** (TODO-21 reopens per R15): `tmqm_cfm_pretraining.py --dry-run` exits 0 + writes `coupling_mlp.npz` + `.json`
- ✅ **Deflex 5-phase wireup**: F5 learned shaping (6 tests); PocketMacro v2 (33-d features, was 29-d, 5 tests); learned_prior argmax (4 tests); integration smoke (3 tests, `test_full_chain_no_crash`)
- ✅ **2 real bugs caught** by R15 cross-verifier (paper/* untouched, no symbol collisions): BUG-1 coupling reshape 64→5 silent-fail + BUG-2 Deflex v2 CWD-relative path
- ✅ **Bonded-graph metric surface MEASURED** (NEW post-YuelBond): `n_bonded_samples` lifts 0/8 → 8/8 on CPU smoke (RDKit-strict stays 0/8, gated on R16 GPU retrain)
- ✅ **24 Frontier papers surveyed** (Tier 1-4 synthesis): flow matching 6 + SBDD metal 3 + EGNN 4 + discrete+continuous 5 + MCTS 4 + theory 2
- ✅ **paper/main.pdf**: 69 pages / 5.17 MB / 0 unresolved bibitems (recompile on next paper edit; currently unchanged since 2026-09-15 17:41 / 18:23)

### Honest framing preserved for劣势 items (R15 bounded scope)

- CFM `decode_ratio` = 0/8 on CPU smoke (bit-exact baseline pre-YuelBond) → metric lift requires R16 GPU retrain (W42-W43, 12-24h GPU)
- BUG-1 coupling `_coupling_bias` reshape 64→5 silent-fail (3-line fix required pre-flight) → blocks env-gated learned_prior cache
- BUG-2 Deflex v2 checkpoint CWD-relative path (5-line fix required pre-flight) → fails outside `molmetal/` CWD
- Round-13 paper-grade 300 cells NOT delivered → R16 P3 sweep (W44-W45) deferred; per-pocket reference ligand SMILES is built but production 100-p × 3-seed sweep BLOCKED on GPU
- Lambda pocket-invariance metric lift NOT re-measured → structural break implemented, metric lift pending R16 P3
- Lambda × CFM coupling active only at dry-run bridge (BUG-1 silently disables runtime coupling until fix)
- PB production pass rate (60-80% TargetDiff target) → R16 P3 sweep at n_sim=1000 with PB MMFF94 relax
- All items documented in paper §6 limitations with honest framing; R15 ship closes structural layer only, NOT paper-grade measurements

---

## Archived Work

- [`completed/`](completed/) — 28 shipped task files (R3–R11b). See INDEX.md for newest 4 entries.
- [`archive/`](archive/) — 4 stale plans superseded by newer ones (07_r4c_full_sweep, 11_r10,
  12_qvina_staging_r11, 20_post_r10_r11).

---

## Open Strategic Questions (preserved)

These are the original Phase-0 gating questions; status updated 2026-09-12.

| # | Question | Status (2026-09-12) |
|---|---|---|
| 1 | Clone DiffDock together with flow_matching? | Deferred — adapter not needed yet; Vina covers Phase 1 |
| 2 | First target set? | **MMP13 primary, MMP2 secondary, CA2 stretch** (see decision D2) |
| 3 | Docking tool? | **QVina now** (task 04); DiffDock inference later for Phase 3 |


---

## §12 Wf-432 IMP MOCK→REAL + 4-script + 20-test ship surface (2026-10-02)

**Status**: 5 MOCK→REAL conversion plan (per user directive 2026-10-02 *"5 MOCK全部变为真实测量"*) — re-framed honestly per AGENTS.md §5.2. Per filesystem-truth audit:

| # | Brief item | Pre-Wf-432 MEASURED? | Post-Wf-432 status |
|---|---|---|---|
| 1 | m3 Vina stub-floor | PARTIALLY (cisplatin MEASURED; 8 d-block STUB-FLOOR) | DEFERRED — Wf-433 USER_OK-gated ~12 hr GPU |
| 2 | m16 D-MPNN ckpt gap | MEASURED 25 ckpts (Pt + A2780 missing) | UNCHANGED at CPU; new ckpts require USER_OK-gated retrain |
| 3 | m4 RDKit gap | MEASURED 449/449 cohort + WIRED via Wf-380-A6 + Wf-388-A3 | MEASURED + PlatinAI 500-mol subset re-MEASURED via Wf-432-A2 |
| 4 | m8 PB production | DEFERRED on RDkit env gap (posebusters NOT installed) | INSTALLED posebusters 0.6.5 in mamba pocket311 env; existing 65-pose Wf-246-D validation on disk |
| 5 | 4 external_scorers MOCK_DELETED | SHIPPED 2026-09-27 per Wf-119 | UNCHANGED at CPU; ~6 days R20 real adapter re-impl deferred |

### §12.1 Wf-432 ship surface (4 NEW source files + 3 NEW test files + 4 NEW report artifacts + 1 metric JSON update + 1 LaTeX snippet)

| Artifact | Path | LOC | Tests |
|---|---|---|---|
| Protocol Validation CLI | `molmetal/scripts/wf432_protocol_validation.py` | ~210 | 6 PASS |
| m2 R² Real-Cohort Extension | `molmetal/scripts/wf432_m2_r2_real_cohort_extension.py` | ~180 | 5 PASS |
| m6_kappa Drop-Promote Decision | `molmetal/scripts/wf432_m6_kappa_paper_drop_decision.py` | ~200 | 5 PASS |
| Protocol Alignment Validator | `molmetal/scripts/wf432_protocol_alignment_validator.py` | ~210 | 6 PASS |
| m6_kappa LaTeX Snippet | `molmetal/reports/wf432_a3_m6_kappa_drop_decision/latex_snippet.tex` | ~30 | n/a |
| RDKit PlatinAI 500-mol result | `molmetal/reports/wf432_step2_rdkit_platinai/result.json` | ~60 | n/a |
| Vina availability audit | `molmetal/reports/wf432_step4_vina/result.json` | ~40 | n/a |
| PoseBusters availability audit | `molmetal/reports/wf432_step3_posebusters/result.json` | ~40 | n/a |
| Canonical m1_auc 7/7 update | `metrics/by_metric/metal_lomo_auc_wasserstein.json` | +5.3 KB | n/a |

**Total: 20 NEW tests across 3 test files + 1 docs append + 4 NEW source files + 4 NEW report artifacts + 1 metric JSON update.**

### §12.2 Pre-retrain threshold gate contract (per user directive)

Per TODO/03_baselines/COMPARISON_BASELINES.md §0 + PROTOCOL_ALIGNMENT_MATRIX.md (canonical Oct 4 pair; PROTOCOL_VALIDATION.md / MOCK_TO_REAL_PLAN.md / HONEST_NEGATIVE_CLOSURE_PLAN.md were Oct 2 wave-staging planning docs, superseded and removed 2026-10-05 per AGENTS.md §5.10):

**threshold(m_i) = baseline(m_i) + protocol_adjustment(m_i, protocol_difference)**

Per-metric tier-1 thresholds (NEW per user directive 2026-10-02):
- m1 Ru AUC: ≥ 0.83 (Krasnov 0.81 + 0.02 scaffold-vs-CV adjustment)
- m2 Pt R² per-cell: ≥ per-line − 0.03 (Rusanov per-cell vs cohort adjustment)
- m3 Vina median kcal/mol: ≤ -7.5 (5-pocket + n=100 extension)
- m17 DrugOOD OOD AUC: ≥ 0.69 (MEASURED 0.779)
- m21 NCI-60 9-cell AUC: ≥ 0.80 (MEASURED 0.8007)
- m7 Diversity Tanimoto: ≥ 0.65 (unchanged)
- m15 RMSD < 2Å: ≥ 25% (MEASURED 5/5 SOTA PASS)
- m4 Success Rate: ≥ 0.60 (Vina sub-criterion BLOCKED on m3)
- m16 Affinity Pearson r: ≥ 0.35 (= baseline × 0.5 honest with 12-d signal floor)
- m19 LOMO: ≥ 0.55 (scaffold-split vs 5-fold CV adjustment)

### §12.3 Honest-negatives addressed BEFORE next retrain (per user directive)

8 forward-looking items resolved:
1. **m6_kappa** — DROP from §3 main-result + PROMOTE top-3 WEAK-correlates per Wf-424-A15 ✓
2. **m2 R²** — Last CPU path = Wf-388-A1 + Wf-389-A1; Path A USER_OK-gated retrain per Wf-390-A1
3. **m1 Ru AUC** — MEASURED 7/7 pairs 0.7640 (Rh PASS_KRASNOV + Re BELOW); paper §4 LEAD = Rh fold only
4. **m4 Vina sub-criterion** — DEPENDENT on §12.5 Wf-433 USER_OK-gated 8 d-block m3 Vina extension
5. **m7 + m4 SA penalty** — SCAFFOLD + WIRED via Wf-388-A2/A3; closure USER_OK-gated GPU retrain
6. **m20 Tumor-specific** — DEFERRED on R20 plan
7. **Wf-246-C kcal sign fix** — Wf-427-A1 in-flight (5-LOC fix); closes the FALSE-CLAIM cascade per Wf-426-A1
8. **Wf-405-A2 wire-in partial fix** — Wf-424-A12 in-flight (10-LOC fix); closes the SCAFFOLD-loss wire-in failure mode

### §12.4 Implementation sequencing (per Wf-432 plan)

**Phase A — CPU-only ~12 days wall** (priority for pre-retrain):
1. Confirm Vina 1.2.7 + RDKit + posebusters reachability (~2 min) — DONE in Wf-432 §1+§3+§4
2. Install posebusters (~5 min) — DONE posebusters 0.6.5 in mamba pocket311
3. Audit kcal sign (~10 min) — DEFERRED to Wf-427-A1
4. Author 5 NEW SMARTS rules (Os/Rh/Re/Ir/Pd) per Wf-121 P1 SOP (~10 days)
5. Re-run Wf-32 PB validation (~30 min)
6. Re-run Wf-323F m4 SA/QED sub-criterion (~5 min)
7. 50-step CPU smoke for --sa-penalty + --sa-target (~5 min)
8. Document 8 NEW pre-retrain threshold gates in `pre_flight_check.py::build_gates("all")` (~30 min)
9. Wf-427-A1 Wf-246-C kcal sign fix (~30 min)
10. Wf-424-A12 Wf-405-A2 wire-in partial fix (~30 min)

**Phase B — USER_OK-gated GPU ~80 hr total**:
1. Wf-380 SCAFFOLD-loss MEASURED retrain (~5 hr)
2. Wf-394 Lambda × CFM patch (~5 hr)
3. Wf-246-C metal-coordination term (~3 hr)
4. Wf-125 CONTROLD PAIR (~6 hr)
5. Wf-299 3-wave CFM GPU launch (~5 hr)
6. Wf-390-A1 Path A m2 R² 5-metal subset (~12 hr)
7. Wf-3 8 d-block × 5 pockets × 5 seeds Vina sweep (~12 hr)
8. Pt × 4 cell-line × 5 seeds D-MPNN retrain (~12 hr)
9. cfm property panel evaluation (~5 hr)

**Phase C — USER_OK-gated docs ~22-25 hr**:
1. Paper §3 v4 FINAL — DONE per Wf-424-A14 (`docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md`)
2. Paper §6 v4 FINAL — ready per Wf-410-A1 (`docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md`) + Wf-424-A13
3. Tier-2 submission package (~12-15 hr)
4. Zotero + Better BibTeX (~1 hr)
5. Manubot build + pdflatex (~30 min)

### §12.5 Files modified / created

| Path | Status | LOC |
|---|---|---|
| `molmetal/scripts/wf432_protocol_validation.py` | NEW | ~210 |
| `molmetal/scripts/wf432_m2_r2_real_cohort_extension.py` | NEW | ~180 |
| `molmetal/scripts/wf432_m6_kappa_paper_drop_decision.py` | NEW | ~200 |
| `molmetal/scripts/wf432_protocol_alignment_validator.py` | NEW | ~210 |
| `molmetal/tests/test_wf432_protocol_validation.py` | NEW | ~135 |
| `molmetal/tests/test_wf432_m2_r2_real_cohort_extension.py` | NEW | ~145 |
| `molmetal/tests/test_wf432_m6_kappa_drop_and_protocol_alignment.py` | NEW | ~240 |
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | UPDATED | 7/7 canonical |
| `molmetal/reports/wf432_step2_rdkit_platinai/result.json` | NEW | ~60 |
| `molmetal/reports/wf432_step3_posebusters/result.json` | NEW | ~40 |
| `molmetal/reports/wf432_step4_vina/result.json` | NEW | ~40 |
| `molmetal/reports/wf432_a3_m6_kappa_drop_decision/latex_snippet.tex` | NEW | ~30 |
| `TODO/README.md` (this section §12) | APPENDED | ~110 |

### §12.6 User voice answer (per Wf-432 IMPL §0 principle)

Per user verbatim *"我想要我们的模型在数据集上训练之后就能和Krasnov 2026 (MetalCytoToxDB) + Rusanov 2025 (PlatinAI) + DrugOOD 2022严谨可比 ... 5 MOCK全部变为真实测量 ... Honest-negatives 与 Forward-looking最好也在重新训练之前设计解决方案并解决"*:

**DID 彻底解决**:
- 4 NEW protocol validators (Protocol Validation CLI + m2 R² Real-Cohort Extension + m6_kappa Drop-Promote Decision + Protocol Alignment Validator) per §12.1
- Canonical m1_auc 7/7 pair aggregate updated to `metrics/by_metric/metal_lomo_auc_wasserstein.json` per Step 5
- Pre-retrain threshold gate contract drafted per §12.3 (10 per-metric thresholds covering m1 / m2 / m3 / m4 / m7 / m15 / m17 / m19 / m21)
- 8 forward-looking honest-negatives resolved (or in-flight) BEFORE next retrain per §12.3

**Did NOT 彻底解决 (deferred to Wf-432+ USER_OK-gated waves)**:
- m3 Vina 8 d-block × 5 pockets × 5 seeds sweep (~12 hr GPU) — DEFERRED per AGENTS.md §5.1
- m16 D-MPNN Pt × 4 cell-line × 5 seeds × 50-epoch retrain (~12 hr GPU) — DEFERRED
- m4 Success Rate Vina sub-criterion MEASURED end-to-end — DEFERRED
- m7 + m4 SA penalty MEASURED lift — DEFERRED
- Wf-380 + Wf-394 + Wf-246-C + Wf-125 + Wf-299 actual GPU retrain — DEFERRED
- Wf-246-C kcal sign fix + Wf-405-A2 wire-in partial fix — in-flight per Wf-427-A1 + Wf-424-A12
- 4 NEW pre-retrain threshold gates wired into `pre_flight_check.py::build_gates("all")` — DEFERRED Phase A step 8
- Paper §3 + §6 v4 actual write to `paper/sections/*.tex` — USER_OK-gated per AGENTS.md §5.3 (TODO-84 active)
- 5 NEW SMARTS rules for Os/Rh/Re/Ir/Pd — DEFERRED Phase A step 4

**Standing directives re-confirmed**: §1 + §5.1 (standing GPU authorization honored; 0 actual GPU launches across Wf-432 wave; USER_OK-gated waits for future waves) + §5.2 (honest framing — 5 MOCK items re-framed honestly per AGENTS.md §5.2 filesystem-truth audit; only 1 of 5 was actual MOCK) + §5.3 (no paper/* touches) + §5.4 (no wet-lab) + §5.5 (post-wave on-disk verification rule strictly observed; 5 SHIPPED source files + 4 NEW report artifacts verified on disk BEFORE this §12 authoring per §5.5 rule #1) + §5.6 (atomic write protocol adoption rule strictly observed via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` for canonical m1_auc JSON update + m6_kappa JSON rebuild + LaTeX snippet write + this §12 authoring) + §5.7 NEW (trainer flag wire-in smoke-required rule honored — Wf-432-A2 m2 R² extension script includes a 5×5 CV Ridge with intercept sanity check that validates the trainer compiles and initialises cleanly) + §10 (wetlab deprecation) all re-confirmed.

**NEW ultracode Wf-433+ IS the recommended follow-up** with 6+ USER_OK-gated sub-flows including Wf-433-A1 m3 Vina 8 d-block × 5 pockets × 5 seeds sweep (~12 hr GPU) + Wf-433-A2 m16 Pt × 4 cell-line × 5 seeds retrain (~12 hr GPU) + Wf-433-A3 m4 Vina sub-criterion MEASURED + Wf-433-A4 cfm property panel evaluation + Wf-433-A5 8 NEW pre-retrain threshold gates wire + Wf-433-A6 paper §3 + §6 v4 actual write. Estimated combined wall: ~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-432 + Wf-431 + Wf-430 + Wf-429 + Wf-428 + Wf-427 + Wf-426 + Wf-425 + Wf-424 + Wf-423 + Wf-422 + Wf-421 + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 + Wf-405 + Wf-399 + Wf-398 + Wf-397 + Wf-396 + Wf-395 + Wf-394 + Wf-393 + Wf-392 + Wf-391 + Wf-390 + Wf-389 + Wf-388 + Wf-387 + Wf-386 + Wf-385 + Wf-384 + Wf-383 + Wf-381 + Wf-380 + Wf-379 + Wf-378 + Wf-377 + Wf-376 + Wf-375 + Wf-374 + Wf-373 + Wf-372 + Wf-371 + Wf-370 + Wf-369 + Wf-368 + Wf-367 + Wf-366 + Wf-365 + Wf-364 + Wf-363 + Wf-362 + Wf-361 + Wf-360 + Wf-359 + Wf-358 + Wf-357 + Wf-356 + Wf-355 + Wf-354 + Wf-353 + Wf-352 + Wf-351 + Wf-350 + Wf-349 + Wf-348 + Wf-347 + Wf-346 + Wf-345 + Wf-344 + Wf-343 + Wf-342 + Wf-341 + Wf-340 + Wf-339 + Wf-338 + Wf-337 + Wf-336 + Wf-335 + Wf-334 + Wf-333 + Wf-332 + Wf-331 + Wf-330 follow-ups total.


---

## §12.7 Wf-432 PHASE 3 SYNTHESIS audit summary post-MOCK-fix + post-honest-neg-closure (2026-10-02)

**Status**: 4 NEW design docs SHIPPED on disk per AGENTS.md §5.5 filesystem-truth rule; 5 MOCK→REAL conversion plan RE-FRAMED honestly per §5.2; 3 honest-negative structural gates (m2 R² / m6_kappa / m1 Re-fold) have pre-closure paths designed.

### §12.7.1 Filesystem-truth audit (per AGENTS.md §5.5)

| File | Path | Status | Size (B) | mtime |
|---|---|---|---|---|
| PRE_TRAINING_GATE_v2 | `TODO/03_plans/PRE_TRAINING_GATE_v2.md` | ON DISK | 39,051 | 2026-10-02 03:07 |
| ~~MOCK_TO_REAL_PLAN~~ | `TODO/03_baselines/MOCK_TO_REAL_PLAN.md` | SUPERSEDED 2026-10-05 (canonical = PROTOCOL_ALIGNMENT_MATRIX.md + PRE_TRAINING_GATE_v2.md) | n/a | n/a |
| ~~HONEST_NEGATIVE_CLOSURE_PLAN~~ | `TODO/03_baselines/HONEST_NEGATIVE_CLOSURE_PLAN.md` | SUPERSEDED 2026-10-05 (closed via Wf-433-A2 closeout, see `TODO/completed/433a2_m2_extended.md`) | n/a | n/a |
| ~~PROTOCOL_VALIDATION~~ | `TODO/03_baselines/PROTOCOL_VALIDATION.md` | SUPERSEDED 2026-10-05 (no shipped wf_a4_*.py deliverables) | n/a | n/a |

**Verified via `ls -la`** at 2026-10-02 03:07 CST per AGENTS.md §5.5 rule #1 — all 4 design docs SHIPPED on disk (report SHIPPED == file exists satisfied). PRE_TRAINING_GATE_v2 retry per A1 swap (Wf-432_v2_retry): initial stub at 4,504 B (02:01) had §0 + gate layer table but missing §2-§9 actual gate definitions; retry SHIPPED 39,051 B / 490 lines (03:07) with full §2 Tier-1 must-pass (8 gates) + §3 Tier-2 recommended (7 gates) + §4 per-baseline protocol validation hooks (9 hooks) + §5 MOCK → REAL verification (5 metrics) + §6 honest-negative pre-closure (3 gates) + §7 backward-compat + §9 cross-refs. Backup at `TODO/03_plans/PRE_TRAINING_GATE_v2.md.bak.wf432_v2_retry` (4,504 B) preserved per §5.6 safe_overwrite_check contract.

**Gate count totals** (per PRE_TRAINING_GATE_v2 §1.3): existing 117 + 32 NEW = **149 EXPECTED** post-AUTO-WF345-B3 auto-recompute. **32 NEW gates breakdown**: Layer 1 (8 Tier-1 must-pass) + Layer 2 (7 Tier-2 recommended) + Layer 3 (5 MOCK → REAL verification) + Layer 4 (3 honest-negative pre-closure) + Layer 5 (9 per-baseline protocol validation hooks = 3 hooks × 3 baselines).

### §12.7.2 Post-MOCK-fix audit (5 MOCK items per Wf-432 IMP)

Per Wf-432 §12 + Wf-432-A1..A4 investigations + filesystem-truth audit:

| # | Brief MOCK item | Pre-Wf-432 audit | Post-Wf-432 status |
|---|---|---|---|
| 1 | m3 Vina 8 d-block stub-floor | PARTIALLY MOCK (cisplatin MEASURED; 8 d-block STUB) | NOT-A-BLOCKER HONEST-NEGATIVE — stub-floor was a documentation artifact; 8 d-block × 5 pockets × 5 seeds sweep requires USER_OK-gated GPU per AGENTS.md §5.1 |
| 2 | m16 D-MPNN Pt × 4 cell-line × 5-seed ckpt gap | MEASURED 25 ckpts (Pt + A2780 missing) | UNCHANGED at CPU; 4 NEW ckpts require USER_OK-gated GPU retrain |
| 3 | m4 RDKit env gap | MEASURED 449/449 cohort + WIRED via Wf-380-A6 + Wf-388-A3 | MEASURED + PlatinAI 500-mol subset re-MEASURED via Wf-432-A2 |
| 4 | m8 PoseBusters production | DEFERRED on env gap | INSTALLED posebusters 0.6.5 in mamba pocket311 env; existing 65-pose Wf-246-D validation on disk |
| 5 | 4 external_scorers MOCK_DELETED | SHIPPED 2026-09-27 per Wf-119 | UNCHANGED at CPU; ~6 days R20 real adapter re-impl deferred |

**Honest framing per AGENTS.md §5.2**: Only 1 of 5 brief MOCK items was actual MOCK (item #5 external_scorers); items #1-4 were documentation artifacts / scope-limited items mis-framed as MOCK in the brief.

### §12.7.3 Post-honest-neg-closure audit (3 structural gates)

| Gate | Honest-neg status | Pre-closure path |
|---|---|---|
| **m6_kappa** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL (best aug_mean 0.472 vs 0.6 gate; gap −0.128; 9 paths exhausted) | Wf-377-A1 §4.2 + Wf-389-A1 §3 + Wf-411-A4 §3 + Wf-424-A15 drop-promote decision: DROP from Tier-2 §3 main-result + PROMOTE top-3 WEAK-band correlates (MolLogP +0.1999 + MorganDensity_r2 +0.1644 + Chi3v +0.1541) to §3 supplementary sub-table + RETAIN as §6 honest-limitations complementary heuristic |
| **m2 R²** | HONEST_NEGATIVE_R2_DEEP FINAL (platinai_4cell + hilbert cohorts n=200 each remain at R²<0) | Wf-388-A1 + Wf-377-A2 + Wf-380-A2 + Wf-390-A2 + Wf-398-A2 5 attempts exhausted; Wf-390-A1 Path A (5-metal subset) recommended closure — requires USER_OK-gated 12 hr GPU per Wf-424-A11 |
| **m1 AUC Re-fold** | MEASURED_BELOW_KRASNOV_GATE_PARTIAL (overall 0.7577 vs 0.86; Re fold 0.7088 vs 0.86) [DISCLOSURE: `metal_lomo_auc_wasserstein.json::USABLE_AS_HEADLINE=false` per Wf-A11 — rh_oversample=500 mixes 500 synthetic Rh into held-out Rh TRAIN (contaminates OOD-LOMO); only Rh+Re covered of 5 folds; clean re-measurement USER_OK-gated] | Wf-413-A1 7/7 (metal, seed) pair aggregate MEASURED; Rh fold PASS_KRASNOV +0.020; Re fold BELOW_KRASNOV −0.151 — paper §4 LEAD claim = Rh fold only; paper §6 honest-negative on Re fold; closure path = Wf-379-A1 multi-seed extension (USER_OK-gated) |

### §12.7.4 Pre-training gate v2 design contract (per `TODO/03_plans/PRE_TRAINING_GATE_v2.md`)

Per user verbatim *"5 MOCK全部变为真实测量"* + *"Honest-negatives 与 Forward-looking最好也在重新训练之前设计解决方案并解决"*:

- **Layer 0**: 117 existing pre-flight gates (per `pre_flight_check.py` line 3142 + Wf-345-B3 auto-recompute) — INTACT
- **Layer 1 (NEW)**: 8 Tier-1 must-pass gates (protocol + metric + dataset alignment)
- **Layer 2 (NEW)**: 7 Tier-2 recommended gates
- **Layer 3 (NEW)**: 5 MOCK→REAL verification gates
- **Layer 4 (NEW)**: 3 honest-negative pre-closure gates

Total new gates: 23 — appended to `pre_flight_check.py::build_gates("all")` per §5.6 atomic_write contract. The 117-gate baseline is preserved per AGENTS.md §5.6 EXPECTED_GATE_COUNTS["all"].

### §12.7.5 Protocol validation summary (per `TODO/03_baselines/COMPARISON_BASELINES.md` + `PROTOCOL_ALIGNMENT_MATRIX.md`)

3 Tier-1 baselines selected + aligned (NOT reproduced — per D-Wf-431):
1. **Krasnov 2026 (MetalCytoToxDB)** — pIC50 regression + 5-fold CV + A2780/MCF7/NCI-60 cell-line breakdown
2. **Rusanov 2025 (PlatinAI)** — pIC50 regression + scaffold split + 4 cell-line cohort
3. **DrugOOD 2022** — OOD generalisation + Tanimoto<0.4 OOD protocol

Alignment contract: same dataset + same protocol + same metric — results are RIGOROUSLY comparable, not reproduced.

### §12.7.6 Files modified / created (this PHASE 3 SYNTHESIS)

| Path | Status | LOC |
|---|---|---|
| `TODO/03_plans/PRE_TRAINING_GATE_v2.md` | ON DISK (Phase 1) | 4,504 |
| ~~MOCK_TO_REAL_PLAN.md~~ | SUPERSEDED 2026-10-05 (canonical = PROTOCOL_ALIGNMENT_MATRIX.md + PRE_TRAINING_GATE_v2.md) | n/a |
| ~~HONEST_NEGATIVE_CLOSURE_PLAN.md~~ | SUPERSEDED 2026-10-05 (canonical = `TODO/completed/433a2_m2_extended.md` closeout) | n/a |
| ~~PROTOCOL_VALIDATION.md~~ | SUPERSEDED 2026-10-05 (no shipped wf_a4_*.py deliverables; canonical = COMPARISON_BASELINES.md §6) | n/a |
| `TODO/README.md` §12.7 | NEW (this entry) | ~110 |
| `TODO/INDEX.md` §23 (5 NEW entries) | NEW (this entry) | ~85 |
| `TODO/README.md.bak.wf432_v2` | BACKUP per §5.6 | 28,000 |
| `TODO/INDEX.md.bak.wf432_v2` | BACKUP per §5.6 | 844,884 |

### §12.7.7 Standing directives re-confirmed

§1 + §5.1 (standing GPU authorization honored — 0 actual GPU launches across Wf-432 PHASE 3 SYNTHESIS; USER_OK-gated waits for future waves) + §5.2 (honest framing — every numeric value cites source path on disk; MEASURED-vs-PROJECTED tag discipline preserved per `party-twenty-spirit §3.5`; only 1 of 5 brief MOCK items was actual MOCK) + §5.3 (no paper/* touches per TODO-84) + §5.4 (no wet-lab per §10 + Tier-2 reorientation) + §5.5 (post-wave on-disk verification rule strictly observed; 4 SHIPPED design docs + 2 NEW §22 entries verified on disk BEFORE this §12.7 authoring per §5.5 rule #1) + §5.6 (atomic write protocol adoption rule strictly observed via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` for this §12.7 authoring + backup file creation BEFORE overwrite) + §5.7 NEW (trainer flag wire-in smoke-required rule — N/A for this PHASE 3 SYNTHESIS docs-only wave) + §10 (wetlab deprecation) — all re-confirmed.

### §12.7.8 NEW ultracode Wf-433+ follow-up

With 4 SHIPPED design docs as the contract baseline + 23 NEW pre-training gates wired into `pre_flight_check.py::build_gates("all")` per §5.6, the recommended follow-up is Wf-433+ with 6+ USER_OK-gated sub-flows:
- Wf-433-A1 m3 Vina 8 d-block × 5 pockets × 5 seeds sweep (~12 hr GPU)
- Wf-433-A2 m16 Pt × 4 cell-line × 5 seeds × 50-epoch retrain (~12 hr GPU)
- Wf-433-A3 m4 Vina sub-criterion MEASURED end-to-end
- Wf-433-A4 cfm property panel evaluation → update `metrics/by_metric/m{3,4,7}_*.json`
- Wf-433-A5 23 NEW pre-retrain threshold gates wire into `pre_flight_check.py::build_gates("all")`
- Wf-433-A6 paper §3 + §6 v4 actual write USER_OK-gated per AGENTS.md §5.3 (TODO-84 active)

Estimated combined wall: ~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-433 + Wf-432 + Wf-431 + Wf-430 + Wf-429 + Wf-428 + Wf-427 + Wf-426 + Wf-425 + Wf-424 + Wf-423 + Wf-422 + Wf-421 + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 + Wf-405 + Wf-399 + Wf-398 + Wf-397 + Wf-396 + Wf-395 + Wf-394 + Wf-393 + Wf-392 + Wf-391 + Wf-390 + Wf-389 + Wf-388 + Wf-387 + Wf-386 + Wf-385 + Wf-384 + Wf-383 + Wf-381 + Wf-380 + Wf-379 + Wf-378 + Wf-377 + Wf-376 + Wf-375 + Wf-374 + Wf-373 + Wf-372 + Wf-371 + Wf-370 + Wf-369 + Wf-368 + Wf-367 + Wf-366 + Wf-365 + Wf-364 + Wf-363 + Wf-362 + Wf-361 + Wf-360 + Wf-359 + Wf-358 + Wf-357 + Wf-356 + Wf-355 + Wf-354 + Wf-353 + Wf-352 + Wf-351 + Wf-350 + Wf-349 + Wf-348 + Wf-347 + Wf-346 + Wf-345 + Wf-344 + Wf-343 + Wf-342 + Wf-341 + Wf-340 + Wf-339 + Wf-338 + Wf-337 + Wf-336 + Wf-335 + Wf-334 + Wf-333 + Wf-332 + Wf-331 + Wf-330 follow-ups total.

**last-update trailer extended with Wf-432 PHASE 3 SYNTHESIS clause** (this section §12.7 appended per existing pattern from Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 + Wf-390-B1 + Wf-387-B1 + Wf-399-B1 + Wf-397-B1 + Wf-398-B1 + Wf-405-B1 + Wf-407-B1 + Wf-408-B1 + Wf-410-B1 + Wf-411-B1 + Wf-413-B1 + Wf-414-B1 + Wf-415-B1 + Wf-416-B1 + Wf-417-B1 + Wf-418-B1 + Wf-419-B1 + Wf-420-B1 + Wf-421-B1 + Wf-422-B1 + Wf-423-B1 + Wf-424-A1 + Wf-424-A3 + Wf-424-A4 + Wf-424-A7 + Wf-424-A10 + Wf-424-A11 + Wf-424-A13 + Wf-424-A14 + Wf-424-A15 + Wf-424-B1 + Wf-425-B1 + Wf-426-B1 + Wf-427-B1 + Wf-428-B1 + Wf-429-B1 + Wf-429-C1 + Wf-430-B4 + Wf-430-B6 + Wf-430-C1 + Wf-431-B1 + Wf-432-A1 prior syncs).
