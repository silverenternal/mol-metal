# RxnFlow (ICLR 2024) Reuse Verdict — WF-SOTA-Reuse R3

**Author:** subagent (claude-code)
**Date:** 2026-09-16
**Repo:** /home/hugo/codes/try_triton_on_rocm/molmetal/references/RxnFlow/ (127 .py files, ~3.5k LOC)
**Paper:** Seo et al. ICLR 2024, arXiv:2410.04542 — "Generative Flows on Synthetic Pathway for Drug Design"
**Source task:** TODO-24 §5 path-(b) — joint atom-type + bond-tensor FM (5-10d architect work)
**Verdict:** **REJECTED for joint-3D-FM reuse.** See §4 fallback.

---

## 1. What RxnFlow actually does

RxnFlow is a **2D GFlowNet over a discrete action space** of {building block × reaction template}. Concretely:

- **State:** partial SMILES string (built incrementally via RDKit reactions).
- **Action space:** one of `Stop`, `FirstBlock`, `UniRxn` (uni-molecular template), `BiRxn` (bi-molecular template × fragment from 1M-block library).
- **Reaction templates:** 109 SMARTS-based reactions from Enamine REAL (`templates/real.txt`); bonds/atoms in the *output* are dictated deterministically by the chosen template applied to the input SMILES via RDKit's `ReactionFromSmarts`.
- **Model:** `src/rxnflow/models/gfn.py::RxnFlow` — a GraphTransformer over the current partial molecular graph. Outputs categorical logits over (protocol, block) pairs. No 3D coords, no equivariant layers, no joint atom-type+bond-tensor head.
- **3D handling:** None for the ligand. The only "3D" code is `src/rxnflow/appl/pocket_conditional/pocket/data.py` which encodes *protein* 3D coords (GVP embedding) for pocket conditioning — the ligand itself is still sampled in 2D SMILES, then docked *externally* via UniDock/Vina as a post-hoc reward signal.
- **Atom vocab:** hard-coded `["B","C","N","O","F","P","S","Cl","Br","I"]` (env_context.py:17) — **no Pt, no Au, no Cu, no Pd, no transition metals.**
- **Bond types:** `[SINGLE, DOUBLE, TRIPLE, AROMATIC]` — no coordination bonds.

## 2. Direct answers to task questions

**Q1.** Does RxnFlow do joint atom-type + bond-tensor + 3D coord generation conditioned on reaction templates?
**A1. NO.** Joint atom-type + bond-tensor → yes (via RDKit reaction application, not learned FM). 3D coord generation → NO. Pocket-conditioned (via GVP protein embedding) → yes for *conditioning vector*, not for joint 3D generation of the ligand.

**Q2.** Could a 30-line wrapper use RxnFlow's training loop + sampling?
**A2. Yes, mechanically — but it would NOT be a joint-3D-FM model.** Wrapper would:
- `from rxnflow.base import RxnFlowTrainer, RxnFlowSampler` (already exist, well-factored)
- `SynthesisEnv(env_dir=...)` with `env_context.num_cond_dim = pocket_dim`
- `class MetallodrugTask(BaseTask): compute_obj_properties(mols) -> PlatinAI reward`
- `--engine rxnflow` could be a CLI flag, but the engine would emit *2D SMILES*, then Vina docks them. This is **identical** in scope to our existing DiffDock/RoseTTAFold/BindNet adapters (already wired per `WF-Wire-Clone-Scoring`). Adds no new capability vs DiffDock (R2).

**Q3.** Hours to (a) wire + (b) train on PlatinAI+tmQM 500 + (c) verify decode_ratio > 0.5?
**A3.**
- (a) Wire: ~4-6h (env_dir setup, BaseTask subclass, CLI flag, 1-pocket smoke). Wrapper as written: ≤30 lines (`rxnflow_adapter.py`). But required infra to *run* it (PyG 2.5.1+cu121, UniDock conda env, 109 reaction template SMARTS curation) is the real cost — already partly done in R1/R2.
- (b) Train: paper uses 50k iterations × 64 mol/iter = 3.2M sampled mols. CPU: 1000 iter/min → ~50h. GPU: 8-12h wall (paper claims).
- (c) Verify decode_ratio > 0.5: **this metric is undefined for RxnFlow.** RxnFlow does not have a "decode" step (no flow, no ODE solve); every sampled SMILES is by construction *chemically valid* (output of RDKit `Reaction.RunReactants`). The relevant metric is *synthesizability rate* (already ~1.0 by construction) or *pocket affinity* (Vina). The user's `decode_ratio` gate doesn't apply.

## 3. Why RxnFlow is the wrong tool for path-(b)

| Requirement (TODO-24 §5 path-b) | RxnFlow has it? |
|---|---|
| Joint atom-type + bond-tensor generation (single model head) | Partial — atom types and bonds are *jointly produced* by RDKit template application, not by a learnable FM head. There is no per-atom/per-bond probability distribution. |
| 3D coord generation (EGNN/Equiformer style) | NO — ligand stays 2D; only protein has 3D |
| Conditional on reaction templates | YES — templates *are* the action space |
| Metal coordination chemistry (Pt/Au) | NO — atom vocab excludes transition metals; no coordination bonds; no oxidation-state machinery |
| PlatinAI 500-mol / tmQM training | NO — GFlowNet needs ≥10k sampled mols for trajectory-balance to converge; tmQM (500) is too small to learn reward distribution |
| Our existing `metal_geometry_prior_bonus` | NO — no concept of metal prior in RxnFlow's GFN loss |
| Pocket-conditional 3D | NO — only protein embedding for context; ligand stays 2D |

## 4. Verdict and fallback

**Verdict: REJECT for path-(b).** RxnFlow is a 2D discrete-action GFlowNet — it does not and cannot produce 3D coordinates, has no metal coordination chemistry, and has no joint learned bond-tensor head. Wiring it as `--engine rxnflow` would add a 5th SOTA-style 2D-then-dock adapter that is strictly less informative than the already-wired DiffDock adapter (R2). The user's `decode_ratio` gate is non-applicable.

**Recommended fallback (no code change needed — TODO-24 §5 path-b is correctly deferred per `WF-Lambda-CFM-coupling (DEFERRED)`):**

1. **Keep path-(c) λ-only as Round-12 default** (already ship, 0 GPU, `r4_lambda_only_run.py --metal-seed cisplatin`).
2. **Hold TODO-24 §5 path-(b) until GPU recovers** — when GPU available, the right tool is **either**:
   - **FlowMol** (Seonghwan Seo et al. 2024, Flow Matching for 3D molecule generation) — true 3D + joint atom+bond FM, naturally fits TODO-24 §5 architecture; OR
   - **DiffDock + our path-(b) joint bond-head CFM** (current plan; P0+P1 fixes ship per `WF-CFM-P0-Fixes` + `WF-Triton-Kernel-Audit`).
3. **RxnFlow is best leveraged as a *synthesizability oracle*, not a generator.** The 109-template set + `MultiRetroSyntheticAnalyzer` (`envs/retrosynthesis.py`) is a strong test for whether our path-(c) λ-only outputs are synthetically accessible. Worth a 2-day sub-project (`WF-RxnFlow-Retrosynth-Oracle`) where RxnFlow's retro-analyzer scores λ-only outputs as an independent synthesizability check — orthogonal to Vina/dock.

**Files reviewed:**
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/RxnFlow/src/rxnflow/models/gfn.py` (256 LOC, GraphTransformer + categorical policy)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/RxnFlow/src/rxnflow/envs/env.py` (205 LOC, SynthesisEnv — RDKit-reaction-driven state transitions)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/RxnFlow/src/rxnflow/envs/env_context.py` (302 LOC, atom/bond vocab DEFAULT_ATOMS = [B,C,N,O,F,P,S,Cl,Br,I] line 17; no 3D in ligand)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/RxnFlow/src/rxnflow/appl/pocket_conditional/model.py` (134 LOC, RxnFlow_PocketConditional — only protein has GVP 3D embedding)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/RxnFlow/scripts/train_pocket_conditional.py` (CLI: 50k iter default)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/RxnFlow/data/templates/real.txt` (109 SMARTS reaction templates, no Pt/Au)

**Cross-refs:**
- TODO-24 (CFM path-(b) architecture redo plan)
- TODO-21 (Lambda × CFM coupling, deferred)
- `molmetal/reports/wf_sota_reuse/audit_RxnFlow.md` (pre-existing audit, conclusion aligns)
- `molmetal/reports/wf_sota_reuse/r2_reinvent4.md` (R2 REINVENT4 verdict, similar reject pattern for structural mismatch)
- `molmetal/scripts/r10_cfg_real_crossdocked.py:978` (current `--engine quickvina2-gpu`; no --engine rxnflow needed)
