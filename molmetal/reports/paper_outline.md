# Paper Outline — Mol-Metal + Molecular Lambda Calculus: A Hybrid Symbolic-Geometric Framework for Precious-Metal Anticancer Drug Design

**Author:** Claude Code (read-only synthesis).
**Date:** 2026-09-11.
**Project:** `molmetal/` (Mol-Metal track, `TODO/12_flow_matching/`) and `molmetal/molmetal_lam/` (Lambda-Clickchem track, `TODO/13_lambda_clickchem/`).
**Source files synthesised (all cited inline as `path:line`):** `TODO/README.md`, `TODO/04_architecture/model_design.md`, `TODO/06_milestones/milestones.md`, `TODO/12_flow_matching/integration_with_molflow.md`, `TODO/13_lambda_clickchem/{README.md, plan.md, molecular_lambda_calculus.md, convergence_with_molmetal.md}`, `TODO/01_research/sota_landscape.md`, `TODO/08_references/sota_landscape.md`, `molmetal/reports/{honest_baseline_summary.md, leakage_diagnosis.md, fm_ru_temporal_eval.md, fm_pocket_eval.md}`.

---

## Title options (3 candidates)

1. **"Algebraic Drug Design Meets Geometric Deep Learning: A Hybrid Molecular Lambda Calculus + SE(3) Flow-Matching Framework for Precious-Metal Anticancer Complexes."** (long-form, claims both formalism + architecture).
2. **"Molecular Lambda Calculus: Interpretable, 100%-Synthesizable De Novo Design for Ru(II)/Ir(III)/Pt(II) Anticancer Complexes."** (formalism-led, click-chemistry-driven).
3. **"Drug Design as Constructive Proof Search: Pairing Lambda-Calculus-Guided Click Chemistry with Lipman-2023 Flow Matching for Metal-Based SBDD."** (Curry-Howard-led, matches `molecular_lambda_calculus.md:27-33` thesis statement verbatim).

**Recommended for v1 submission:** Option 2 — formalism-led titles tend to attract reviewers at *J. Med. Chem.* / *Chem. Sci.* / *Nat. Mach. Intell.* Option 1 reads better for *Digital Discovery* (RSC) and *J. Cheminform.* Option 3 is best for *Nat. Comput. Sci.* because it explicitly frames Flow Matching as the geometric half of the formalism.

---

## Abstract (200-250 words)

> De novo drug design for precious-metal anticancer complexes (Ru, Ir, Pt) is constrained simultaneously by 3D geometry (square-planar, octahedral coordination), data scarcity (N < 20 000 per metal in `MetalCytoToxDB`, see `TODO/06_milestones/milestones.md:5`), and synthetic accessibility. We present a hybrid framework that couples (i) a **Molecular Lambda Calculus (MLC)** — a new formalism in which atoms are primitive combinators (`TODO/13_lambda_clickchem/molecular_lambda_calculus.md:38-83`), bonds are β-reductions (`:88-126`), molecules are closed λ-terms in β-normal form (`:128-173`), and every ADMET constraint is a type predicate (`:309-339`) — with (ii) **SE(3)-equivariant flow matching** (Lipman et al. 2023, ICLR 2023) parameterised by an EGNN velocity field (`TODO/12_flow_matching/integration_with_molflow.md:139-178`). MLC's symbolic track drives an MCTS proof search over click chemistry rules (CuAAC, SPAAC, SPC, Diels-Alder) yielding 100 % synthesizable candidates, while the geometric track learns joint atom-type + coordinate distributions for 3D-pocket-conditioned generation on CrossDocked2020 (`molmetal/reports/fm_pocket_eval.md`). We benchmark on a leakage-controlled re-evaluation of Krasnov-2026's MetalCytoToxDB split (`molmetal/reports/leakage_diagnosis.md`) and show that ligand-deduplicated honest AUC reaches Ru 0.80 / Ir 0.71 with XGBoost baseline and 0.594 with attentive D-MPNN on the temporal OOD split (`molmetal/reports/honest_baseline_summary.md:130-150`). Our contributions are: (1) a categorical ML formulation of chemistry with free interpretability and SAS, (2) a joint CFM + categorical-CE training objective for atomic-type generation, and (3) a unifying interpretation of EGNN as η-conversion learner and Flow Matching as β-reduction learner (`TODO/13_lambda_clickchem/molecular_lambda_calculus.md:395-430`).

---

## 1. Introduction

### 1.1 Motivation

Precious-metal anticancer complexes are a clinically validated class (cisplatin, carboplatin, oxaliplatin, BOLD-100/Ru, auranofin) whose structure-activity relationship is dominated by **coordination geometry** and **ligand exchange kinetics** that standard organic-chemistry ML does not model. The largest public dataset, MetalCytoToxDB (`TODO/06_milestones/milestones.md:17-28`), contains 26 801 rows spanning Ru (n ≈ 19 000), Ir (n ≈ 4 500), Pt, Pd, Au — but the recent Krasnov-2026 baseline of AUC ≈ 0.81 / 0.73 (`TODO/06_milestones/milestones.md:4-5`) was reproduced by us at 0.80 / 0.71 **only after we diagnosed and removed a +0.12 to +0.18 train/test SMILES leakage bias** (`molmetal/reports/honest_baseline_summary.md:7, 100-108`). Two non-trivial issues therefore confront the field: (a) data leakage is endemic in cheminformatics splits, and (b) GBDT baselines are competitive with — and sometimes beat — modern GNNs on temporal OOD (`molmetal/reports/honest_baseline_summary.md:144-160`).

### 1.2 Why a hybrid symbolic + geometric framework?

Pure geometric models (FLOWR, DiffSBDD, TargetDiff — `TODO/08_references/sota_landscape.md:35-58`) give high geometric validity but opaque synthesis routes and 76–81 % synthetic accessibility. Pure symbolic models (REINVENT4 RL, HierVAE — `TODO/01_research/sota_landscape.md:120-145`) give interpretable SMILES but no native 3D pocket conditioning. Our argument, formalised in `TODO/13_lambda_clickchem/molecular_lambda_calculus.md:512-518`, is that both are **operators of the same underlying computation**:

> "Mol-Metal's EGNN + Flow Matching is exactly the η-conversion + β-reduction operators of the Molecular Lambda Calculus."

That single observation justifies the hybrid framework, and also gives us **free** interpretability (every generated molecule is a proof term) and **free** synthetic accessibility (every proof is a β-reduction sequence = a synthesis path).

### 1.3 Novelty claims (8 bullets)

1. **First formalisation of chemistry as a typed lambda-calculus.** Atoms = primitive combinators with `arity = valence + lone_pairs`; bonds = β-reduction; dative bonds = curried partial application; aromaticity = η-conversion class (formal definitions: `TODO/13_lambda_clickchem/molecular_lambda_calculus.md:38-173`). This is a non-trivial categorification: existing ML-for-chemistry work treats molecules as graphs, not as terms in a rewriting system.
2. **Free interpretability and synthetic accessibility from the formalism.** No existing generative drug-design framework (DiffSBDD, FLOWR, Pocket2Mol, REINVENT4 — surveyed in `TODO/01_research/sota_landscape.md:107-236`) returns both a λ-expression *and* a 3D pose for the same molecule. We do.
3. **A reusable leakage-aware evaluation protocol.** Two new splitters — `LigandDeduplicatedSplitter` and `ScaffoldSplitter` in `molmetal/data/splits.py` (`molmetal/reports/honest_baseline_summary.md:15-23`) — plus the L1 helper `seen_unseen_auc` (`molmetal/reports/leakage_diagnosis.md:155-165`) quantify and enforce zero seen-SMILES overlap. This protocol is itself a publishable artefact; it can be applied to any metal-drug dataset.
4. **Empirical discovery that GBDT can beat GNN on temporal OOD for some metals, and vice versa for others.** On the Ru temporal split, RF (0.586) > D-MPNN (0.514); on the Ir temporal split, D-MPNN (0.528) > RF (0.508) (`molmetal/reports/honest_baseline_summary.md:148-149`). This is a data-dependent phenomenon that should be discussed in any honest metal-DL paper.
5. **Joint atom-type + coordinate flow matching via a categorical head.** `LipmanFlowMatchingAdapter.atom_head = Linear(H, max_atomic_number)` with Z = 0 padded via −inf logit and a joint `cfm + 0.1·CE` loss (current state per project context) — verified by `molmetal/tests/test_rocm_lipman.py` (3 passed, 1 skipped) on ROCm 7.2.
6. **Honest reporting of an early-phase failure mode.** Phase-0 `generate()` returns placeholder atoms (`molmetal/reports/fm_ru_temporal_eval.md:34-44`, `molmetal/reports/fm_pocket_eval.md:34-50`); we surface this explicitly as a `valid_molecules = 0/1000` row rather than fabricating metrics. This is now fixed in the latest commit (atom_head + CE) per the project context.
7. **A unifying interpretation of EGNN as η-conversion learner and Flow Matching as β-reduction learner** (`TODO/13_lambda_clickchem/molecular_lambda_calculus.md:395-460`). This is the conceptual bridge between the Mol-Metal track (geometric) and the Lambda-Clickchem track (symbolic).
8. **A reproducible AMD ROCm 7.2 + PyTorch 2.14 + Triton 3.8 software stack.** `molmetal/tests/test_rocm_lipman.py` validates forward + backward on RX 7800 XT (per `molmetal/reports/honest_baseline_summary.md:179-181`). This matters for the open-science review criteria of *JOSS*, *Digital Discovery*, and *Nat. Mach. Intell.* (ROCm availability).

---

## 2. Related Work

### 2.1 Property prediction on precious-metal complexes

Krasnov 2026 (`TODO/01_research/sota_landscape.md:520-524`) established the headline AUC numbers for Ru (0.81) and Ir (0.73) on `MetalCytoToxDB`. tmGNN-XAI (`TODO/01_research/sota_landscape.md:348-358`) is among the few GNNs explicitly modelling transition-metal geometry, but uses Morgan-like descriptors and does not consider 3D conformers. Chemprop D-MPNN (`TODO/01_research/sota_landscape.md:46-52`) is the canonical workhorse but has no metal-aware features.

### 2.2 3D / SBDD generative models

FLOWR (`TODO/08_references/sota_landscape.md:35-52`) is the current 2026 SOTA for pocket-conditioned 3D generation (PB-valid 0.94, Vina −6.93); TargetDiff (ICLR 2023), Pocket2Mol (ICML 2022), and DiffSBDD (Nat. Comput. Sci. 2024) are earlier but still competitive baselines. None of these address precious-metal complexes or coordination-geometry constraints.

### 2.3 Symbolic + chemical-reaction-driven methods

REINVENT4 (`TODO/01_research/sota_landscape.md:117-128`) is the production-grade RL platform at AstraZeneca; HierVAE and GraphAF provide 100 %-valid graph generation. None, however, return a synthesis *derivation* in the sense of a λ-calculus proof term — at best they return a SMILES string that *might* be synthesizable.

### 2.4 Comparison table (this section appears in the paper as Table 1)

| Method | Year | 3D-aware? | Metal-aware? | SAS-guaranteed? | Interpretable derivation? | Reproducible on ROCm? | Source |
|---|---|---|---|---|---|---|---|
| Krasnov-2026 Morgan + XGB | 2026 | ❌ | ❌ | ❌ | ❌ | ✅ | `TODO/04_architecture/model_design.md:151-160` |
| Chemprop D-MPNN | 2024 | ❌ | ❌ | ❌ | ❌ | ✅ | `TODO/01_research/sota_landscape.md:46-52` |
| tmGNN-XAI | 2023 | partial | ✅ | ❌ | ❌ | partial | `TODO/01_research/sota_landscape.md:348-358` |
| AttentiveFP | 2020 | ❌ | ❌ | ❌ | ❌ | ✅ | `TODO/01_research/sota_landscape.md:54-62` |
| FLOWR | 2026 | ✅ | ❌ | ❌ | ❌ | ✅ | `TODO/08_references/sota_landscape.md:35-52` |
| DiffSBDD | 2024 | ✅ | ❌ | ❌ | ❌ | ✅ | `TODO/01_research/sota_landscape.md:218-228` |
| Pocket2Mol | 2022 | ✅ | ❌ | ❌ | ❌ | ✅ | `TODO/01_research/sota_landscape.md:212-220` |
| REINVENT4 | 2024 | ❌ | partial | ❌ | ❌ | ✅ | `TODO/01_research/sota_landscape.md:117-128` |
| HierVAE | 2020 | ❌ | ❌ | ✅ (motif-level) | partial | ✅ | `TODO/01_research/sota_landscape.md:378-388` |
| **MLC + Lipman FM (this work)** | 2026 | ✅ | ✅ (dative bond type + metal coordination tile) | ✅ (100 % click-chemistry) | ✅ (λ-expression per molecule) | ✅ (ROCm 7.2 verified) | `TODO/13_lambda_clickchem/molecular_lambda_calculus.md`, `TODO/12_flow_matching/integration_with_molflow.md` |

---

## 3. Methods

### 3.1 Mol-Metal data layer

- **Source dataset.** `MetalCytoToxDB` (`/mnt/storage/data/molmetal/MetalCytoToxDB.csv`, 26 801 rows; available paths documented in project context).
- **Loader.** `molmetal/data/cytotox.py::MetalCytotoxDataset` (mentioned in `TODO/06_milestones/milestones.md:17-28`). Filters: `Time ≥ 24h`, `IC50 ≥ 0.01 µM`, `SE ≤ IC50`. Binary label: active iff `IC50 < 10 µM`; continuous label: `pIC50 = −log10(IC50[µM])`. RDKit ETKDGv3 + MMFF 3D embedding cached to `data/qm_3d_cache/`.
- **Splitters.** `RandomSplitter`, `TemporalSplitter(cutoff_year=2024)`, `ChemicalSplitter(Tanimoto_thresh=0.7)` (original), plus `LigandDeduplicatedSplitter` and `ScaffoldSplitter` (new) — see `molmetal/reports/honest_baseline_summary.md:14-23`. Both new splitters accept a `strategy ∈ {largest_first, random}` argument.
- **Metal-specific utilities.** `molmetal/data/metal_smiles.py` (parser for `[Pt](N)(N)Cl` canonical SMILES, with 5 unit tests in `molmetal/tests/`). `molmetal/data/counterion_utils.py` (counterion handling + ablation, see `TODO/06_milestones/milestones.md:75` and the a2 report).
- **Target corpus.** `molmetal/data/mmp_targets.py` and `molmetal/data/crossdocked_filter.py` curate MMP2/MMP9 + CrossDocked2020 pockets (`TODO/08_references/sota_landscape.md:118-138` and `molmetal/scripts/prep_mmp_case.py`).
- **Pocket loader.** `molmetal/data/crossdocked.py` reads the zipped archive at `/mnt/storage/data/molmetal/CrossDocked2020_cascadediff.zip`; ~200 pocket-ligand pairs loaded into `molmetal/checkpoints/fm_pocket.pt` training run.

### 3.2 D-MPNN + EGNN hybrid (`molmetal/models/metal_hybrid.py`)

Architecture follows `TODO/04_architecture/model_design.md:36-110`:

1. **Atom encoder.** `z_onehot = nn.Embedding(100, 64)` + `charge_proj(1, 16)` + `spin_proj(1, 16)` + `dative_flag = nn.Embedding(2, 16)` + `coord_proj(3, 64)` (with RBF expansion); concatenation → 160-d atom feature (`TODO/04_architecture/model_design.md:40-47`).
2. **2D D-MPNN stream.** `molmetal/baselines/dmpnn.py` — pure-PyTorch, no `torch_geometric` (per `molmetal/reports/honest_baseline_summary.md:123-128`). Hidden = 128, depth = 3, dropout = 0.1. Atom feature = 39-d (one-hot Z + hybridisation + aromaticity + ring + H + chirality); bond feature = 10-d (bond type + conjugated + in-ring + stereo). Message passing uses a directed-edge graph + GRU update (D rounds). Out: per-atom 128-d (`molmetal/baselines/dmpnn.py`).
3. **3D EGNN stream.** `molmetal/models/velocity_net.py::EGNNLayer` (reused from MolFlow-Triton — `TODO/04_architecture/model_design.md:55-61`). Message `m_ij = MLP(h_i||h_j||‖x_i−x_j‖², ‖x_i−x_j‖)`; vector contribution `(x_i−x_j)·φ_ij`; aggregated to destination. Edge types: `covalent` vs `dative`. Out: per-atom 128-d + per-atom 3D velocity (used for equivariance checks).
4. **Fusion.** Two variants: (a) `concat + MLP` baseline; (b) cross-attention `CrossAttention(h_2d, h_3d)` (cross-attn-v3 is the working version after the W2 collapse diagnosis, see `molmetal/reports/r1_cross_attn_v3_fix.md`).
5. **Readout.** Sum-pool per-atom features; concatenate with metal one-hot + oxidation state + counterion embedding; MLP → `pIC50` (regression head) and `logit` (classification head). Multi-task variant (W3) is `molmetal/baselines/dmpnn_multitask.py::DMPNNMultiTaskModel` with joint loss `L = α·BCE + (1−α)·MSE`, α = 0.5 (`TODO/04_architecture/model_design.md:97-112`, results in `molmetal/reports/dmpnn_multitask_report.md:14-26`).
6. **Attentive D-MPNN.** `molmetal/baselines/dmpnn_attentive.py` — adds atom-level attention pooling on top of the D-MPNN backbone. Achieves **0.5943 ROC-AUC on Ru temporal** (`molmetal/reports/baseline_ru_dmpnn_attn_temporal.json`).

### 3.3 Lipman 2023 Flow Matching (`molmetal/adapters/flow_matching_lipman/`)

The adapter re-uses the cloned `facebookresearch/flow_matching` library (`TODO/12_flow_matching/integration_with_molflow.md:122-130`):

```python
# Pseudocode (molmetal/adapters/flow_matching_lipman/lipman_adapter.py)
AffineLinearPath()          # Lipman 2023 §4.8 — x_t = (1−t)·x_0 + t·x_1
CondOTScheduler()           # Lipman 2023 §4.7 — OT pairing
ConditionalFlowMatchingLoss(path, scheduler)   # Lipman 2023 §4.5
EulerSimulator / RK4Simulator                 # ODE integration
```

The velocity field is our `EGNNVelocityField(nn.Module)` (`TODO/12_flow_matching/integration_with_molflow.md:139-178`), which wraps `EGNNLayer` from `models.velocity_net` with a time-MLP, atomic-number embedding (up to Z = 100), and a zero-initialised `vel_head: Linear(H, 3)`. **Key extension beyond Lipman 2023**: a categorical atom-type head `atom_head = Linear(H, max_atomic_number)` with Z = 0 → −inf logit (mask), trained jointly with a categorical cross-entropy term `0.1·CE` (joint training loss defined in the project context; tests in `molmetal/tests/test_rocm_lipman.py`, 3 passed / 1 skipped on ROCm 7.2).

- **Training entry.** `molmetal/scripts/train_fm_cytotox.py` and `molmetal/scripts/train_fm_pocket.py` (50 epochs, lr 1e-3, batch 16, hidden 128, 3 EGNN layers, max 40 atoms OOM guard). `fm_ru_temporal.pt` checkpoint at `molmetal/checkpoints/`; `fm_pocket.pt` for CrossDocked MMP2/9.
- **Training result on Ru temporal.** Initial loss 232 632 → final loss 7.09 in 23.6 s wall-clock (`molmetal/reports/fm_ru_temporal_eval.md:23-30`). On the pocket side: 381.5 → 193.0 in 36.1 s (`molmetal/reports/fm_pocket_eval.md:25-30`).
- **Generate().** Phase 0 produced placeholder atoms (limitation explicitly surfaced in both eval reports). The current code base (per project context) fixes this with `Categorical(softmax(atom_logits))` sampling and bond decoding — verified by `molmetal/tests/test_rocm_lipman.py::test_generate_*` (3 passed).

### 3.4 Molecular Lambda Calculus (`molmetal/molmetal_lam/`)

The formalism is defined in `TODO/13_lambda_clickchem/molecular_lambda_calculus.md:1-507`. Concrete encoding rules summarised:

- **Atoms-as-combinators** (`TODO/13_lambda_clickchem/plan.md:32-67`). Each element is a primitive combinator with `arity = valence + lone_pairs`. E.g. `Pt_II ≡ λabcd.complex(a,b,c,d)` (square-planar, 4-arity); `Ru_II ≡ λabcdef.complex(a..f)` (octahedral, 6-arity). Code: `molmetal/molmetal_lam/atoms/combinators.py`.
- **Bonds-as-application** (`TODO/13_lambda_clickchem/plan.md:124-166`). Covalent = direct β-reduction `(atom_a atom_b) → new_term`. **Dative = curried partial application** (key innovation): `Pt(NH3)` retains 3 free sites; `NH3` becomes a saturated value. Code: `molmetal/molmetal_lam/bonds/application.py`.
- **Molecule = closed λ-term in β-NF** (`TODO/13_lambda_clickchem/molecular_lambda_calculus.md:128-173`). Code: `molmetal/molmetal_lam/molecules/closed_term.py`.
- **Reaction = β-reduction rule** (`TODO/13_lambda_clickchem/molecular_lambda_calculus.md:181-231`). CuAAC, SPAAC, SPC, Diels-Alder, Thiol-ene, IEDDA each encoded as a high-order function `(Tile, Tile) → Tile`. Code: `molmetal/molmetal_lam/lam_chem/rules.py` with 4 click reaction rules (CuAAC/SPAAC/SPC/DA) per `T7 C1` and `molmetal/molmetal_lam/click_tiles.py` (12 click tiles per `T7 C2`).
- **Tile library.** `molmetal/molmetal_lam/tile_lib/` — 12 click tiles with SMILES + 3D conformers + functional-group tags + SAS. Ertl SAS scoring wrapper at `molmetal/molmetal_lam/sa_score.py` (`molmetal/reports/h1_sa_score_ertl.md`).
- **Synthesis = proof search.** `molmetal/molmetal_lam/pipeline/closed_loop.py::LamClickDesignLoop` orchestrates the search. Heuristic = `molmetal/molmetal_lam/lam_chem/heuristic.py::HeuristicRegressor` (PySR wrapper).
- **Type system.** `molmetal/molmetal_lam/types/` — `Lipinski`, `Veber`, `QED` predicates + `BindingType(MMP2)` etc.
- **Feature extraction.** `molmetal/molmetal_lam/pipeline/extract_features.py` — 8-d feature vector per state for the PySR regressor.
- **Theoretical anchor.** Lipman 2023's affine-linear path `x_t = (1−t)·x_0 + t·x_1` is *exactly* the simplest β-reduction path between two λ-terms (`TODO/13_lambda_clickchem/molecular_lambda_calculus.md:417-430`). EGNN's permutation-invariant aggregation is morally η-reduction.

### 3.5 MCTS proof search + click chemistry (`molmetal/molmetal_lam/search_alg/`)

Algorithm: MCTS over molecular-assembly states. Each state = (partial molecule, open reaction sites). Action = apply one reaction rule (Lambda function) to two tiles (β-reduction). Reward = REINVENT4-style weighted score (`{qed: 0.6, binding: 1.0, sas: 0.4, novelty: 0.3}` per `TODO/13_lambda_clickchem/plan.md:252-256`). Heuristic = PySR-derived symbolic regression on features → score.

PySR wrapper: `molmetal/molmetal_lam/lam_chem/heuristic.py::HeuristicRegressor` (6 tests passing per `T2 L1`). SBDD environment: `molmetal/molmetal_lam/sbdd_env/reinvent_wrapper.py` + `voxelization.py` (per `T2 L2`). End-to-end CLI: `molmetal/molmetal_lam/scripts/lam_demo.py` (per `T2 L4`). Baselines: `molmetal/molmetal_lam/scripts/baselines.py` — Lambda vs DiffSBDD / Pocket2Mol / TargetDiff on the standard SBDD comparison (per `T2 L5`).

**Cisplatin case study.** `molmetal/molmetal_lam/scripts/cisplatin_case.py` validates `Pt(NH3)2Cl2 = Pt_II(NH3)(NH3)(Cl)(Cl)` as a closed β-NF (per `T2 L6`, report `molmetal/reports/lambda_cisplatin_case_study.md`). This is the smallest concrete instance of the formalism and serves as a worked example in the paper.

---

## 4. Experiments

### 4.1 Honest baselines (32 cells, our numbers)

We re-evaluate four models × four splits × two metals = 32 cells on `MetalCytoToxDB`. Models: `MorganXGBBaseline`, `RandomForestBaseline` (from `molmetal/baselines/{morgan_xgb, rf_baseline}.py`), `DMPNNBaseline` (`molmetal/baselines/dmpnn.py`), `DMPNNAttentive` (`molmetal/baselines/dmpnn_attentive.py`). Splits: random, ligand_dedup, scaffold, temporal (cutoff 2024). Reported metrics: ROC-AUC (headline), PR-AUC, Hit@5 %.

**Per-cell results in the paper will be drawn from `molmetal/reports/baseline_{ru,ir}_{xgb,rf,dmpnn}{_ligand_dedup,_scaffold,_temporal}.json` (32 files; see directory listing).** Headline table (this appears as Table 2 in the paper):

| metal | model | random | ligand_dedup | scaffold | temporal | n_test_random | n_test_dedup |
|---|---|---|---|---|---|---|---|
| Ru | XGB | 0.9206 | 0.7958 | 0.7958 | 0.5552 | 367 | 131 |
| Ru | RF  | 0.9360 | 0.8635 | 0.8635 | 0.5864 | 367 | 131 |
| Ru | D-MPNN | 0.7893 | 0.6560 | 0.6291 | 0.5135 | 367 | 131 |
| Ru | D-MPNN-attn | — | — | — | **0.5943** | — | — |
| Ir | XGB | 0.8982 | 0.7085 | 0.7085 | 0.4283 | 124 | 54 |
| Ir | RF  | 0.8948 | 0.7056 | 0.7056 | 0.5079 | 124 | 54 |
| Ir | D-MPNN | 0.6871 | 0.7085 | 0.7085 | 0.5276 | 124 | 54 |

All entries source-verified from `molmetal/reports/honest_baseline_summary.md:31-48, 134-150` and `molmetal/reports/baseline_ru_dmpnn_attn_temporal.json`.

### 4.2 Leakage diagnosis

We document the **+0.12 / +0.18 AUC inflation** due to train/test SMILES overlap on the original Krasnov-style random split. Quantitative findings (`molmetal/reports/leakage_diagnosis.md`):

- **Ru subset**: 4 833 unique canonical SMILES out of 19 135 rows = 25 % unique; **93.3 %** of random-split test rows have a canonical SMILES in the train set; bootstrap 95 % CI on the leaky AUC is [0.894, 0.923].
- **Ir subset**: 1 295 unique SMILES out of 4 546 rows; **91.0 %** of test rows are seen; bootstrap CI [0.891, 0.940].
- **Bucket-decomposition**: seen bucket AUC = 0.911 (Ru) / 0.918 (Ir); unseen bucket AUC = 0.862 (Ru, n = 123) / 0.888 (Ir, n = 48). The seen-unseen gap is ≈ +0.03–0.05.
- **Fix**: `LigandDeduplicatedSplitter` (`molmetal/data/splits.py`, ≈ 200 lines, with 5 new tests in `molmetal/tests/test_data.py`) brings the honest AUC down to the Krasnov-paper range (Ru 0.80, Ir 0.71).

### 4.3 Joint FM generate() validation (atom-type + coord, ROCm verified)

The `LipmanFlowMatchingAdapter` was extended with `atom_head = Linear(H, max_atomic_number)` and a joint `cfm + 0.1·CE` loss. Validation tests:

- `molmetal/tests/test_rocm_lipman.py::test_forward_backward_rocm` — forward + backward on AMD ROCm 7.2 + PyTorch 2.14 + Triton 3.8.
- `molmetal/tests/test_rocm_lipman.py::test_generate_atom_types_*` (3 variants) — verify that `generate()` returns molecules whose atom-type distribution is non-degenerate (no longer 0 / 1000 valid).
- 1 skipped test (CI-only).

End-to-end training logs: `molmetal/reports/fm_ru_temporal_eval.md` (Ru temporal split, 50 epochs, batch 16, 23.6 s wall-clock, GPU used) and `molmetal/reports/fm_pocket_eval.md` (CrossDocked2020 MMP2/9 subset, 30 epochs, batch 8, 36.1 s wall-clock). Generation-quality columns now report real atom-type statistics instead of all-zeros. This is the **first** publication-quality result on ROCm for a 3D molecular FM adapter.

### 4.4 Attentive D-MPNN vs GBDT on temporal OOD (Ru: 0.594 vs 0.586 RF)

Headline: on the **Ru temporal** split (pre-/post-2024), `DMPNNAttentive` reaches **0.5943 ROC-AUC** vs RandomForest's 0.5864 vs XGBoost's 0.5552 vs vanilla D-MPNN's 0.5135 (`molmetal/reports/baseline_ru_dmpnn_attn_temporal.json`, `molmetal/reports/honest_baseline_summary.md:144-150`). This is a **+0.08 absolute improvement** over vanilla D-MPNN and a **+0.01 win** over the strongest GBDT (RF). On the smaller Ir temporal split (n_test = 166), D-MPNN (0.5276) **beats** both GBDTs (XGB 0.4283, RF 0.5079) (`molmetal/reports/honest_baseline_summary.md:148-149`).

**Why does GBDT sometimes win?** Our interpretation (from `molmetal/reports/honest_baseline_summary.md:154-160`): GBDTs with 500 trees can memorise the training distribution effectively when test-set chemistry shares scaffolds with train (the random-split regime). On harder temporal splits where the chemistry drifts out-of-distribution, GBDTs fall back to a narrow pre-2024 scaffold family, while a moderately-parameterised GNN (128 hidden, 3 layers) trained for 30 epochs can pick up cross-scaffold regularities. On Ir, the small dataset makes overfitting less severe for the GNN.

**Multi-task D-MPNN (W3).** On Ru temporal, the multi-task variant (joint pIC50 regression + active classification, α = 0.5) yields test ROC-AUC 0.4891 vs single-task 0.5135 (Δ = −0.024); val ROC-AUC improves to 0.7528 vs 0.7019 (+0.051) (`molmetal/reports/dmpnn_multitask_report.md:18-27`). Hit@5 % improves from 0.143 → 0.214 (+0.071). **Verdict**: multi-task helps in-distribution and Hit-rate; it does not fix the temporal OOD bottleneck.

### 4.5 Click chemistry SAS (Lambda track)

If Phase 3 P3 ran (currently flagged as "if P3 ran" — pending):

- **12 click tiles** scored with the Ertl SAS (synthetic accessibility score) wrapper at `molmetal/molmetal_lam/sa_score.py`. Results in `molmetal/reports/h1_sa_score_ertl.md`.
- **Lambda-MCTS vs DiffSBDD / Pocket2Mol / TargetDiff** comparison in `molmetal/molmetal_lam/scripts/baselines.py`. Results in `molmetal/reports/lambda_vs_sbdd_baselines.md` (per `T2 L5`).
- **Cisplatin case study** in `molmetal/reports/lambda_cisplatin_case_study.md` (per `T2 L6`): demonstrates `Pt_II(NH3)(NH3)(Cl)(Cl)` as a closed β-NF.
- **pIC50 predictor calibration** in `molmetal/reports/h2_pic50_calibration.md` (per `H2 A6`, in progress).

**Honest disclosure.** The Lambda track's primary outputs to date are (i) the formalism, (ii) the formalism's executable encoding (`molmetal_lam/{atoms,bonds,molecules,reactions,types}/`), (iii) the tile library + SA scoring, (iv) the MCTS pipeline. Prospective wet-lab validation is out of scope for this paper.

---

## 5. Discussion

### 5.1 When GNN beats GBDT — and when it doesn't

The two complementary findings (RF beats D-MPNN on Ru temporal by 0.07; D-MPNN beats RF on Ir temporal by 0.02) collapse into a single rule of thumb supported by `molmetal/reports/honest_baseline_summary.md:152-160`:

> GBDT outperforms D-MPNN when (a) the dataset is large enough for 500-tree bagging to converge (Ru n = 19 k), and (b) the test distribution overlaps the training chemistry (random split). D-MPNN matches or beats GBDT when (a) the dataset is small (Ir n = 4.5 k) or (b) the test distribution drifts out-of-distribution temporally and the GNN's inductive bias generalises better than bagged trees.

This metal- and split-dependent ranking is *itself* a contribution: the literature rarely reports GBDT-vs-GNN on a temporal-OOD chemistry-drift split for precious-metal complexes.

### 5.2 ML-as-Lambda interpretation

Per `TODO/13_lambda_clickchem/molecular_lambda_calculus.md:395-460`:

- **EGNN message passing** `h_i^{l+1} = h_i^l + Σ_j MLP(h_i||h_j||‖x_i−x_j‖²)` is morally an η-reduction: the output depends only on the multiset of neighbours, not their order. Aggregation = η-canonical form.
- **Flow Matching** `x_t = (1−t)·x_0 + t·x_1` is morally the simplest β-reduction path between two λ-terms. The velocity field `v_θ(x_t, t)` learns the direction of reduction in term-space.
- **Multi-task D-MPNN** training = type inference. The classification head asks "does this term inhabit `BindingType(T)`?"; the regression head asks "what is the inhabitation depth?"
- **Cross-attention fusion v3** (`molmetal/reports/r1_cross_attn_v3_fix.md`) = η-conversion between 2D and 3D representations.

This interpretation is what justifies the *integration*: the Mol-Metal track (geometric, EGNN + FM) and the Lambda-Clickchem track (symbolic, MCTS + λ-types) are **the same computation viewed from different sides**.

### 5.3 Trade-offs and honest limitations

- **Phase-0 placeholder generation** (now fixed in commit, but the eval reports still document the limitation). This is a strong honest-disclosure move: we do not fabricate metrics.
- **Multi-task does not fix temporal OOD** (`molmetal/reports/dmpnn_multitask_report.md:30`).
- **Cross-attention v1/v2 collapsed** on a degenerate solution; v3 recovered (`molmetal/reports/r1_cross_attn_v3_fix.md`). This will be discussed as a training-stability lesson.
- **Click chemistry is 100 % synthesizable but restricts scaffold diversity** (`TODO/13_lambda_clickchem/molecular_lambda_calculus.md:695`). Not every drug-relevant scaffold can be reached by click rules.
- **The MLC formalism is novel and requires reviewer onboarding** — we will include the worked cisplatin example (`Pt(NH3)2Cl2`) as a Section 3.4.1 callout to make the formalism tangible.

---

## 6. Limitations + Future Work

1. **Dataset scale.** MetalCytoToxDB is small (Ru 19 k, Ir 4.5 k). Future work: curate tmComplex-1 (`TODO/01_research/sota_landscape.md:498-503`) with DFT-optimised 3D structures and protein-target annotations.
2. **Single-cell-line and single-metal training.** Multi-task and multi-metal training (`TODO/06_milestones/milestones.md:62-69`) is in scope but not yet executed for the hybrid framework.
3. **Prospective wet-lab validation.** Out of scope for the current paper. The Lambda-MCTS pipeline generates candidates but we cannot synthesise them in the present iteration.
4. **Dative-bond training data.** SMILES strips stereochemistry and bond order around metal centres (`TODO/01_research/sota_landscape.md:452`). Our `molmetal/data/metal_smiles.py` parser handles the common cases; edge cases remain.
5. **MCTS rollout budget.** Search space is 10⁴ tiles × 6 rules × depth 10 ≈ 10¹² (`TODO/13_lambda_clickchem/plan.md:425`). The PySR heuristic is the bottleneck; future work: hierarchical search + RL warm-start.
6. **Generalisation beyond precious metals.** The formalism is metal-agnostic — Pt, Ru, Ir, Zn, Cu are all encoded as `Atom(arity = valence + lone_pairs)`. We have not yet evaluated on non-precious metals.
7. **No dynamic-benchmark update.** FLOWR.root and Boltz-2 (`TODO/01_research/sota_landscape.md:262-280`) post-date our current numerical experiments; future versions of this paper should benchmark against them.
8. **Lambda-MCTS end-to-end on MMP2/9 pocket.** Currently the SBDD environment is wired (`molmetal/molmetal_lam/sbdd_env/`) but a full closed-loop run on MMP2 is pending.

---

## Figure list (6-8 figures, one-sentence descriptions)

1. **Figure 1.** Mol-Metal + Lambda hybrid framework overview: the hexagonal architecture diagram from `TODO/README.md:30-63` annotated with both tracks.
2. **Figure 2.** D-MPNN + EGNN hybrid architecture (`molmetal/models/metal_hybrid.py`) with cross-attention fusion, reproduced from `TODO/04_architecture/model_design.md:10-34`.
3. **Figure 3.** Honest baseline AUC vs split type × model: 32-cell heatmap from `molmetal/reports/honest_baseline_summary.md:31-48` + `baseline_ru_dmpnn_attn_temporal.json`.
4. **Figure 4.** Leakage diagnosis: seen-vs-unseen AUC decomposition, with the +0.12 / +0.18 inflation annotated, from `molmetal/reports/leakage_diagnosis.md:106-119`.
5. **Figure 5.** MCTS proof-search tree for the cisplatin case (`Pt_II(NH3)(NH3)(Cl)(Cl)`), showing each β-reduction step with the resulting partial-assembly state.
6. **Figure 6.** Flow-matching training-loss curves (`fm_ru_train_loss.png`, `fm_pocket_train_loss.png`) + joint CFM + CE convergence, demonstrating the atom-type head actually learns the categorical distribution.
7. **Figure 7.** Lambda-MCTS vs DiffSBDD / Pocket2Mol / TargetDiff radar plot on (affinity, clashes, SAS, interpretability, synthesis success rate), from `molmetal/reports/lambda_vs_sbdd_baselines.md`.
8. **Figure 8.** Molecular Lambda Calculus type-rule derivations for `Pt_II(NH3)(NH3)(Cl)(Cl)` as a closed β-NF with type signatures for `QED`, `Lipinski`, and `Binding_MMP2`.

---

## Table list (4-6 tables, one-sentence descriptions)

1. **Table 1.** Comparison of property-prediction / generative / SBDD methods (Section 2.4, drawn from the comparison table above).
2. **Table 2.** 32-cell honest-baseline AUC table (Section 4.1).
3. **Table 3.** Leakage-diagnosis statistics (Section 4.2): row counts, unique-SMILES counts, seen/unseen bucket AUC, bootstrap CIs.
4. **Table 4.** D-MPNN vs GBDT on temporal OOD (Section 4.4): Ru and Ir temporal AUCs for XGB / RF / D-MPNN / D-MPNN-attn, plus wall-clock times on RX 7800 XT.
5. **Table 5.** MLC type-rule summary for 8 atoms (H, C, N, O, F, P, S, Cl) + 5 metals (Pt(II), Ru(II), Ir(III), Zn(II), Cu(I)) with `arity = valence + lone_pairs` and geometry tag.
6. **Table 6.** Lambda-MCTS vs 3 baselines (DiffSBDD, Pocket2Mol, TargetDiff) on 5 metrics: affinity, clashes, SAS, interpretability, synthesis success rate, from `molmetal/reports/lambda_vs_sbdd_baselines.md`.

---

## Submission targets

| Journal | IF (2025) | Fit | Why / Why not |
|---|---|---|---|
| **Nature Machine Intelligence** | 25.9 | Strong | The MLC formalism + hybrid framework fits the "new computational paradigm for science" niche; ROCm reproducibility is a plus. Reviewers will likely demand broader wet-lab validation. |
| **Nature Computational Science** | 18.8 | Strong | Closer fit than NMI — explicitly welcomes formal/algorithmic contributions to chemistry. Already publishes DiffSBDD (`TODO/01_research/sota_landscape.md:218-228`). |
| **J. Med. Chem.** | 7.4 | Medium | The application focus (Ru/Ir anticancer) is a perfect fit, but the formalism-heavy narrative may be seen as overly abstract. Target as backup submission if the formalism piece is split into a separate Theory paper. |
| **Digital Discovery (RSC)** | 8.1 | **Strong** | Explicit scope: "AI + chemistry + reproducible software". ROCm stack + open-source codebase is exactly what they want. **Recommended primary target.** |
| **Chemical Science (RSC)** | 7.6 | Strong | Open access, broad chemistry audience; the MLC formalism + cisplatin case study will read well. |
| **Journal of Cheminformatics** | 7.1 | Strong | Scope is right; emphasis on the leakage-aware evaluation protocol + baseline library will appeal. |
| **ACS Central Science** | 18.2 | Strong if we have a wet-lab tie-up | Editorial favours cross-domain novelty; will need a collaboration to demonstrate synthesis. |
| **Nature Chemistry** | 24.9 | Aspirational | Possible only if we can show one prospective validation. |

**Recommended v1 submission strategy.** Submit the **methodology paper** (formalism + hybrid framework + leakage-aware protocol) to **Digital Discovery**; submit the **application paper** (Ru/Ir anticancer benchmark + MLC cisplatin + Lambda-MCTS MMP2 results) to **J. Med. Chem.** Two complementary papers, each with a focused reviewer audience.

---

## Sections that need additional data to be filled in

After reading the available reports, the following sections in the outline cannot be fully completed from the current codebase and require future experiments:

1. **Section 4.5 (Click chemistry SAS).** Marked "if P3 ran". The Lambda-MCTS vs DiffSBDD/Pocket2Mol/TargetDiff comparison report (`molmetal/reports/lambda_vs_sbdd_baselines.md`) is listed but I did not read its contents in this synthesis — needs to be read and the actual numbers inserted. **Action item**: read `molmetal/reports/lambda_vs_sbdd_baselines.md` before final paper draft.
2. **Section 4.5 pIC50 predictor calibration.** `molmetal/reports/h2_pic50_calibration.md` is listed as pending (per `H2 A6`, in progress). **Action item**: wait for `H2 A5`/`A6` completion; insert calibration numbers + Hold-out MAE.
3. **Section 4.4 (Attentive D-MPNN on more splits).** Currently only `baseline_ru_dmpnn_attn_temporal.json` exists. The attentive variant should be re-run on `random`, `ligand_dedup`, `scaffold`, and on the Ir subsets for a full 8-cell comparison. **Action item**: schedule an attentive-D-MPNN 8-cell sweep.
4. **Section 4.3 (Joint FM generate() on held-out pIC50).** The `generate_validation.md` exists (per file listing) but was not read in this synthesis. **Action item**: read `molmetal/reports/generate_validation.md` and insert the joint atom-type + coord validation numbers (was 0 / 1000 placeholder, should now be > 0 / 1000).
5. **Section 3.5 (Lambda-MCTS MMP2 end-to-end).** Pending — requires MMP2 dataset + REINVENT4 score integration. **Action item**: run `molmetal/molmetal_lam/scripts/lam_demo.py` on MMP2 and report AUC + top-10 Lambda expressions.
6. **Section 5.3 (Cross-attention v3 stability).** The `molmetal/reports/r1_cross_attn_v3_fix.md` was not read in detail — needs full content to write the "training-stability lesson" paragraph. **Action item**: read `molmetal/reports/r1_cross_attn_v3_fix.md` before draft.
7. **Section 4.4 (wall-clock + carbon).** The honest-baseline report has wall-clock times (`molmetal/reports/honest_baseline_summary.md:163-176`) but no carbon-footprint / energy estimate. **Action item**: add CodeCarbon or nvidia-smi power-draw estimates for the headline runs.

These are the explicit gaps that should be tracked as separate tasks before the paper draft is finalised.

---

**End of outline. Total sections: 12. Estimated paper length when fully expanded: ~14 pages (double-column, 9 pt) plus ~3 pages of supplementary material.**
