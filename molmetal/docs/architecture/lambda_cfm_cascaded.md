# Strategy 3 — Cascaded Lambda → CFM-refine

**Document type:** Future-work architecture design
**Status:** DESIGN ONLY — no code changes, no measurements
**Author:** generated from TODO-21 (Strategy 3 spec, lines 81-110)
**Date:** 2026-09-17
**SBDD paper lineage:** TargetDiff §4.3, FLOWr §3.4, Pocket2Mol §4.2

---

## 1. Overview

Mol-Metal currently has **two independent generators**:

- **Lambda MCTS** (`molmetal/molmetal_lam/search_alg/proof_search.py`, 6217 LOC) — emits β-NF typed-variable SMILES over discrete synthesis reductions (5 click rules × metal-ligand exchanges × MCTS search). Ships today as the λ-only baseline (Round-12 N=10×3, 78/78 spot-check tests green, MEASURED cells in paper §4 Table 1).
- **CFM** (`molmetal/adapters/flow_matching_lipman/__init__.py`, 2888 LOC) — joint atom-type + coordinate generative model via EGNN + Lipman flow matching + bond head. Currently BLOCKED at `decode_ratio = 0/192` on the h=128 5000-step checkpoint per `wf_cfm_frontier_research/final.md` 2026-09-15.

**Strategy 3** is the **pragmatic middle** between Strategy 1 (Lambda-as-reward, 1 week, CPU-verifiable) and Strategy 2 (joint training, 5-10 d, GPU-gated). It treats Lambda and CFM as a **two-stage cascade**: Lambda proposes chemically valid scaffolds; CFM refines their 3D coordinates inside the binding pocket.

Concretely:

1. **Stage A — Lambda MCTS** emits 1-50 β-NF candidates per pocket (synth-valid, click-rule-compliant, metal-prior-respecting, PB-passing) — already produces these in 11-50 s per pocket per the Round-12 pilot.
2. **Stage B — CFM refine** takes each candidate's SMILES, embeds a rough 3D conformer via RDKit `AllChem.EmbedMolecule`, then runs **100-200 ODE steps** of the trained CFM velocity field conditioned on `(atom_types, init_coords, pocket_embedding)`. The CFM output is `(refined_coords, refined_bonds)` that respects both the Lambda pathway AND the pocket geometry.
3. **Selection** — rank refined candidates by Vina score + PB-pass; keep top-K (D12 user-decision: top-K is the recommended mode).

This is the **most well-trodden SBDD pipeline in the literature** — every published 3D-pocket-conditioned generator uses some form of "init from SMILES → refine with neural field". The novelty contribution of Mol-Metal is not the cascade itself but the **typed-variable path constraint** from Lambda into the CFM initial state.

---

## 2. Architecture diagram

```
        pocket.pdb  ────────────────────────────────┐
                                                      │
        target SMILES hint (optional) ──────────┐    │
                                                │    │
                                                ▼    ▼
   ┌─────────────────────────────────────────────────────────┐
   │  STAGE A: Lambda MCTS  (proof_search.py:3216 search)    │
   │  ──────────────────────────────────────────────────     │
   │  • Typed-variable β-NF initial state (SMILES + tags)     │
   │  • Click-rule reductions (CuAAC / SPAAC / ThiolEne /    │
   │    Suzuki / AmideCoupling via pt_click_compat.py)       │
   │  • MCTSProofSearch with UCB + transposition table       │
   │    + virtual loss + 5-click prior + SA penalty          │
   │  • RewardAggregator (QED / SA / PB / Vina / pIC50 /     │
   │    PlatinAI oracle / REINVENT4 multiprop)                │
   │  ──────────────────────────────────────────────────     │
   │  Output: top-K β-NF candidate SMILES                     │
   │          (K=1 for D12=refine-1, K=5-10 for top-K)        │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              │  candidate SMILES × K
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  BRIDGE: SMILES → 3D conformer  (RDKit AllChem)          │
   │  ──────────────────────────────────────────────────     │
   │  For each candidate SMILES s_i:                          │
   │    mol_i = Chem.MolFromSmiles(s_i)                       │
   │    mol_i = Chem.AddHs(mol_i)                             │
   │    AllChem.EmbedMolecule(mol_i,                          │
   │                          randomCoords=5,                 │
   │                          numZeroFail=10)                │
   │    AllChem.MMFFOptimizeMolecule(mol_i)                   │
   │    coords_i = mol_i.GetConformer().GetPositions()        │
   │    atom_types_i = [atom.GetAtomicNum() for ...]          │
   │  ──────────────────────────────────────────────────     │
   │  Output: (atom_types, init_coords) tuples, K            │
   │  Edge: bond topology comes from Lambda's typed path     │
   │        (carried through as the bond-prior mask)         │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              │  (atom_types, init_coords, pocket) × K
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  STAGE B: CFM refine  (flow_matching_lipman/__init__.py)│
   │  ──────────────────────────────────────────────────     │
   │  • PocketEncoder (one-shot, K-fold batched)              │
   │    → pocket_embed ∈ R^(K × pocket_hidden)               │
   │  • For each candidate k=1..K:                            │
   │      x_init = init_coords_k (3 × N_k tensor)             │
   │      AtomCloud(z=atom_types_k, x=x_init)                 │
   │      AffineProbPath.sample(x_0, x_1, t)                 │
   │      ── ODE integration (100-200 steps) ──               │
   │      ODESolver(velocity_model=wrapped_EGNN,             │
   │                step_size=1.0 / n_steps,                  │
   │                method='midpoint')                        │
   │      x_refined_k = solver.sample(...)                    │
   │  • BondAwareDecoder.decode(z, x_refined_k)               │
   │      → bonds_k (with Path B chem-aware soft prior)      │
   │  • Chem.SanitizeMol + PB checks                         │
   │  ──────────────────────────────────────────────────     │
   │  Output: refined (coords, bonds) × K                    │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              │  refined candidates × K
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  SELECTION                                               │
   │  ──────────────────────────────────────────────────     │
   │  • Vina dock each (CPU Vina / QVina — both engines      │
   │    per D7 default)                                       │
   │  • PB pass-rate filter (chemistry-only + protein-aware  │
   │    per wf_pb_dock_mode_wire)                            │
   │  • Rank by combined score (Vina + SA + QED + novelty)    │
   │  • Return top-N (paper-grade: N=10)                      │
   └─────────────────────────────────────────────────────────┘
```

**Data flow summary:**

- Stage A emits **discrete typed-variable SMILES** (one per MCTS leaf).
- Bridge converts to **3D `(atom_types, init_coords)`** via RDKit (deterministic, <1 s per SMILES on CPU).
- Stage B runs the **trained CFM velocity field as a refining ODE**, starting from `init_coords` and conditioning on the pocket embedding + the typed-variable bond prior.
- Selection runs **physical docking + chemistry validation**, returning the top-N refined mols.

---

## 3. Two-stage implementation roadmap

### Stage A — Lambda MCTS emit (already shipped)

| Component | Status | Location |
|---|---|---|
| `MCTSProofSearch.search` | SHIPPED (6217 LOC) | `molmetal/molmetal_lam/search_alg/proof_search.py:3216` |
| 5-click SMARTS rules | SHIPPED | `molmetal/molmetal_lam/lam_chem/decoder_rework.py:419-465` |
| 5×5 Pt-click compat matrix | SHIPPED | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` |
| `RewardAggregator` (8+ channels) | SHIPPED | `molmetal/molmetal_lam/search_alg/proof_search.py:566` |
| `--metal-seed cisplatin` | SHIPPED | `molmetal/scripts/r4_lambda_only_run.py:1466` |
| `top-K` selector | SHIPPED | `molmetal/scripts/r4_lambda_only_run.py` candidate sort |

**No new code needed for Stage A.** Top-K = 1 (D12 mode A) or top-K = 5-10 (D12 mode B) Lambda candidates per pocket are already available.

### Bridge — SMILES → 3D conformer (1-line addition)

| Component | Status | Plan |
|---|---|---|
| RDKit `EmbedMolecule` + `MMFFOptimizeMolecule` | SHIPPED (used in `r4_lambda_only_run.py:_embed_3d_for_rmsd`) | Add helper `embed_candidate_3d(smiles) → (atom_types, coords)` in `molmetal/molmetal_lam/lam_chem/conformer_embed.py` (NEW, ~30 LOC). |
| Bond topology preservation | SHIPPED | Pass Lambda's typed-variable bond mask as a soft prior into Stage B (Path B decoder rework already accepts this). |

### Stage B — CFM refine (2-3 d engineering effort)

| Step | Effort | GPU? | Description |
|---|---|---|---|
| B.1 Add `--cascade-mode` CLI flag | 0.5 h | no | New flag on `r4_lambda_only_run.py` to dispatch to bridge + refine path. |
| B.2 Wire `embed_candidate_3d` helper | 0.5 h | no | One-line wiring into the orchestrator after Lambda emit. |
| B.3 Wire CFM `refine_from_init(z, x_init, pocket_embed, n_steps=200)` method | 4 h | no (CPU-debug) | Modify `_generate_impl` to accept `x_init` arg; when present, ODE starts from `x_init` instead of random Gaussian. |
| B.4 Wire bond-prior mask from Lambda typed variables | 2 h | no | Pass the typed-variable bond mask into `BondAwareDecoder.decode` as `bond_prior_mask` arg (Path B already accepts priors). |
| B.5 200-step ODE refine (CPU smoke, no GPU) | 4 h | yes (1 h smoke) | Smoke on 5 SMILES × 200 steps; verify refine produces coherent coords (no NaN, no diverged positions). |
| B.6 Cascade e2e test | 2 h | no | New `test_lambda_cfm_cascade.py` (5 tests): SMILES → embed → refine → RDKit sanitize → bond consistency vs Lambda path. |
| B.7 Vina lift measurement | 8 h | yes (6-12 h GPU) | 10×3 sweep with --cascade-mode ON vs OFF; compare Vina lift. |

**Total effort: 2-3 days** (1 d adapter + 1 d 200-step ODE refine + 1 d verify), per TODO-21 spec line 90.

### Stage C — Selection + paper integration (1 d)

| Step | Effort | Description |
|---|---|---|
| C.1 Vina + PB scoring of refined candidates | 1 h | Reuse `molmetal/scripts/r4_c_full_sweep.py` Vina dispatch (D7 both engines default). |
| C.2 Top-K selector with diversity bonus | 2 h | Combine Vina + homotype + Tanimoto per Round-9 P0 metrics. |
| C.3 Paper §3.5 subsection "Cascade inference" | 4 h | New subsection in `paper/sections/03_method.tex` describing the cascade. |
| C.4 Paper §4 Table 1 + §4.6 new cascade column | 2 h | Add cascade mode cells to paper tables. |

---

## 4. Known blockers

### 4.1 Hard gate — `decode_ratio > 0` on real CFM

The Strategy 3 cascade **requires the CFM forward pass to produce sane outputs at all**. As of 2026-09-15:

- `wf_cfm_frontier_research/final.md` reports `decode_ratio = 0/192` on the h=128 5000-step P1-fixed checkpoint (decode failures: 97.4% `disconnected_distance_graph`).
- The Path B chem-aware soft prior (`wf_cfm_path_b_decoder_rework/final.md`) lifts decode to **192/192 bond-bearing** on synthetic CFM-style clouds, but the real-CFM lift is **NOT MEASURED** (gated on GPU retrain).
- Per `wf_cfm_internal_review/diagnose.md`, root causes include: (A) BondOrderHead fixed in_dim=9 vs needed 9+2·hidden_dim, (B) tanh-bounded velocity scalar gate has irreducible floor 4-7 vs target 5-8, (C) bonds=zeros placeholder in `_generate_impl:2018` means BondAwareDecoder NEVER called at training time, (D) hidden_dim=32/n_layers=2 vs TargetDiff 1.2M params (10× under-parameterised).

**Resolution path:** All 5 P0 fixes shipped (`wf_cfm_p0_fixes` 2026-09-15). The 5000-step diagnostic at h=128 with stacked P0+P1 fixes completed (`wf_gpu_recovery_now` 2026-09-15) but `decode_ratio = 0/192` persists at this budget. **Strategy 3 is BLOCKED until either (a) full 10000-step GPU retrain completes and `decode_ratio > 0` on real CFM output, OR (b) a CPU-only fallback path exists.**

### 4.2 GPU flakiness

Per `wf_gpu_diag_fix` 2026-09-14, the SMU-hang root cause on RX 7800 XT is **firmware-level** and requires a cold power cycle. The 2026-09-15 one-shot recovery is not durable. The 6-12 h GPU retrain budget for Strategy 3 verification is **at risk** if the GPU hangs mid-run.

**Mitigation:** checkpoint every 500 steps; resume from last checkpoint on re-failure. This is the same protocol as `wf_cfm_retrain_full` Phase 2.

### 4.3 Cascade inference cost

Two-stage pipeline doubles wall time per candidate:
- Stage A: 11-50 s per pocket (Round-12 pilot at n_sim=1000)
- Bridge: <1 s per candidate (RDKit embed)
- Stage B: 2-5 s per candidate at 200 ODE steps on GPU (projected; not measured)
- Selection: 5-30 s per candidate (Vina docking)

**Wall-clock per pocket:** 11-50s + 5×(1s + 5s + 30s) ≈ 4-6 min for K=5 with Vina selection. Acceptable for paper-grade 100×3 = 300 evaluations ≈ 20-30 h wall on a single GPU.

**Mitigation:** cache Stage A output across all Stage B attempts (Lambda emit is the expensive part; once emitted, K candidates can be re-refined with different CFM noise seeds cheaply).

### 4.4 No feedback to Lambda

Strategy 3 is one-way: Lambda emits → CFM refines → done. Lambda does NOT learn from CFM refine failures (unlike Strategy 2 joint training). For paper-grade this is acceptable; for long-term improvement it is suboptimal.

**Mitigation:** future-work direction in paper §7. Cite Strategy 2 (joint training) as the path forward.

---

## 5. Honest framing

### 5.1 Refine is well-known SBDD technique

The Lambda → CFM cascade is **not novel architecture** — it is the standard 3D-pocket-conditioned generation pipeline used by every published SBDD model:

| Paper | Init | Refine | Citation |
|---|---|---|---|
| **TargetDiff** (Guan ICLR 2023) | Random Gaussian on pocket atom-clouds | 1000-step DDPM iterative denoising | TargetDiff §4.3, arXiv:2303.03543 |
| **FLOWr** (Bose ICLR 2024) | Random Gaussian (rectified flow) | 100-200 ODE steps | FLOWr §3.4 |
| **Pocket2Mol** (Peng ICML 2022) | Empty atom cloud + bond template | Iterative atom/bond sampling via GEOM | Pocket2Mol §4.2 |
| **DiffSBDD** (Schneuing NeurIPS 2023) | Random Gaussian | DDPM 1000 steps + bond decoder | DiffSBDD §3 |
| **DecompDiff** (Guan ICLR 2024) | Decomposed fragment coords | DDPM fragment-by-fragment | DecompDiff §3, arXiv:2303.14246 |

**Mol-Metal's contribution is not the cascade itself.** The contribution is the **typed-variable β-NF path constraint from Lambda MCTS into the CFM initial state** — the bond-prior mask carries the synthetic pathway information forward, ensuring the refined coords respect Lambda's 5-click rule chain (CuAAC / SPAAC / ThiolEne / Suzuki / AmideCoupling) AND the metal coordination (Pt_II / Pt_IV / chelating).

### 5.2 What is NOT measured

This document is **design only**. No Strategy 3 code has been written, no measurements taken, no GPU retrain consumed. Specifically:

- Refined Vina lift: PROJECTED +1 to +2 kcal/mol (TODO-21 line 92). NOT MEASURED.
- Refined decode_ratio lift: PROJECTED +0.10 to +0.30 (TODO-21 line 92). NOT MEASURED.
- Cascade wall-clock: PROJECTED 4-6 min per pocket at K=5. NOT MEASURED.
- Bond-prior mask effectiveness on real CFM output: NOT MEASURED.

Per project policy (`paper/sections/06_limitations.tex`), these will be marked `\SEARCHONLY{}` or `\PROJECTED{}` in the paper, not `\MEASURED{}`, until the cascade is verified end-to-end.

### 5.3 When to start Strategy 3

Per TODO-21 spec, Strategy 3 is **deferred** until:

1. **decode_ratio > 0** gate lifted on real CFM output (`wf_cfm_frontier_research` gate)
2. **GPU stable** for 12+ hour runs (currently flaky)
3. **Round-13 100×3 λ-only column ships** (so the cascade has a baseline to compare against)

Recommended sequencing:

1. Strategy 1 (Lambda-as-reward) ships first (CPU-only, 1 week) — gains the reward-shaping lift without GPU dependency.
2. GPU retrain with P0+P1 fixes + λ-reward shaping completes; if `decode_ratio > 0`, promote CFM to Round-14 column.
3. Strategy 3 cascade becomes viable; 2-3 d engineering effort + 1 d paper integration.

---

## 6. Reference implementation — pseudo-code (~30 lines)

```python
# molmetal/docs/architecture/lambda_cfm_cascaded.md §6 — PSEUDO-CODE ONLY
# This is a design sketch. No production code has been written.

from typing import List, Tuple
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from molmetal.molmetal_lam.search_alg.proof_search import MCTSProofSearch
from molmetal.adapters.flow_matching_lipman import FlowMatchingAdapter
from molmetal.adapters.vina_adapter import VinaDockingAdapter


def lambda_cfm_cascade(
    pocket_pdb: str,
    cfm: FlowMatchingAdapter,
    vina: VinaDockingAdapter,
    mcts: MCTSProofSearch,
    k_top: int = 5,
    n_ode_steps: int = 200,
) -> List[Tuple[str, float, object]]:
    """Two-stage Lambda → CFM-refine cascade.

    Stage A: Lambda MCTS emits top-K β-NF candidate SMILES.
    Bridge:  RDKit embeds 3D conformers.
    Stage B: CFM refines coords via 200-step ODE conditioned on pocket.
    Select:  Vina-dock refined candidates, return top-scoring.

    Returns: list of (smiles, vina_score, refined_mol) tuples.
    """
    # Stage A — Lambda MCTS (already ships; ~11-50 s per pocket)
    candidates = mcts.search(pocket_pdb=pocket_pdb, n_top=k_top)

    # Bridge — SMILES → 3D conformers via RDKit
    init_states = []
    for smiles in candidates:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomCoords=5, numZeroFail=10) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol)
        coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        atom_types = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
        bond_mask = build_bond_prior_mask(mol, smiles)  # typed-variable bond prior
        init_states.append((smiles, atom_types, coords, bond_mask))

    # Stage B — CFM refine (the new bit)
    pocket_embed = cfm.encode_pocket(pocket_pdb)  # (pocket_hidden,) — one-shot
    refined = []
    for smiles, atom_types, coords, bond_mask in init_states:
        refined_mol = cfm.refine_from_init(
            atom_types=atom_types,
            init_coords=coords,
            pocket_embed=pocket_embed,
            bond_prior_mask=bond_mask,
            n_steps=n_ode_steps,           # 100-200 ODE steps
            method="midpoint",             # ODE solver method
        )
        refined.append((smiles, refined_mol))

    # Selection — Vina dock + rank
    scored = []
    for smiles, mol in refined:
        vina_score = vina.dock(mol, pocket_pdb)  # kcal/mol, lower is better
        scored.append((smiles, vina_score, mol))
    scored.sort(key=lambda x: x[1])

    return scored[:k_top]


def build_bond_prior_mask(mol, smiles: str) -> torch.Tensor:
    """Build soft bond prior from Lambda typed-variable annotations.

    The Lambda MCTS emits typed SMILES with bond annotations that encode
    the 5-click rule chain + metal coordination. This function projects
    those annotations onto the mol's bond topology as a soft prior for
    the Path B chem-aware bond decoder.

    Implementation note: depends on decoder_rework.py:bond_prior_mask
    (Path B already accepts bond priors; see wf_cfm_path_b_decoder_rework).
    """
    raise NotImplementedError("DESIGN ONLY — see TODO-21 Strategy 3 spec")
```

---

## 7. References

- TODO-21 — Lambda × CFM coupling strategy memo, lines 81-110 (Strategy 3 spec): `TODO/pending/21_lambda_model_coupling.md`
- TargetDiff (Guan ICLR 2023) — DDPM refine on pocket atom-clouds: arXiv:2303.03543 §4.3
- FLOWr (Bose ICLR 2024) — rectified flow ODE 100-200 steps: §3.4
- Pocket2Mol (Peng ICML 2022) — GEOM-based iterative sampling: §4.2
- DiffSBDD (Schneuing NeurIPS 2023) — DDPM + bond decoder: §3
- DecompDiff (Guan ICLR 2024) — fragment decomposition: §3, arXiv:2303.14246
- Mol-Metal Path B decoder rework: `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md`
- Mol-Metal CFM internal review: `molmetal/reports/wf_cfm_internal_review/diagnose.md`
- Mol-Metal GPU recovery verdict: `molmetal/reports/wf_gpu_recovery_now/final.md`
- Mol-Metal Pt-click compat matrix: `molmetal/molmetal_lam/lam_chem/pt_click_compat.py`

---

**End of design doc. No code changes; no measurements. Status: DESIGN ONLY.**
