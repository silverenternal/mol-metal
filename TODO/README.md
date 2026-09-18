# Mol-Metal: Precious-Metal Anti-Cancer Drug Design via Flow Matching + SBDD

**Last updated:** 2026-09-16 (R15 ship summary block + honest framing preserved block added)

## Project Overview

**Mol-Metal** designs de-novo small-molecule ligands for precious-metal anti-cancer targets
(cisplatin-class, Pt/Ru/Ir organometallics) using **Flow Matching** generative models on top of a
**Structure-Based Drug Design (SBDD)** loop. The system generates 3D ligand poses in a protein
pocket, docks and scores them, and refines via a closed-loop design cycle.

We aim to close the single-GPU (RX 7800 XT) vs. 64×A100 cluster gap by leveraging **Triton
kernels on ROCm** — the constraint is "fits in 16 GB memory," not "reproduce multi-GPU training."

---

## Environment

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
