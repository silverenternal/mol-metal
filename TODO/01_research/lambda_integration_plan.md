# Lambda Line — SOTA Integration Plan

**Date:** 2026-09-12
**Storage:** `/home/hugo/codes/try_triton_on_rocm/molmetal/references/`
**Goal:** Absorb reusable code from 6 cloned SOTA repos into Lambda's reaction-rules MCTS pipeline, expanding from 5 click rules + 14 tiles → 13+ reactions + 1k+ building blocks.

## Inventory (cloned 2026-09-12)

| Repo | Size | GitHub | Key reusable modules for Lambda |
|---|---:|---|---|
| **SyntheMol** | 21M | swansonk14/SyntheMol (Nature MI 2024) | `synthemol/reactions/{reaction.py, real.py, query_mol.py}` — RDKit reaction template wrapper identical format to Lambda's reaction_rules. **13 REAL SMARTS reactions** ready to import. Building-block sanitizer. |
| **SynFlowNet** | 22M | mirunacrt/synflownet (ICLR 2025) | `synflownet/envs/synthesis_building_env.py` — ReactionTemplateEnv (forward+backward stepping). `synflownet/utils/synthesis_utils.py` — RDKit Reaction wrapper. |
| **RxnFlow** | 9.7M | SeonghwanSeo/RxnFlow | Synthesis-oriented flow matching. Code-reuseable scoring (SeH binding proxy + Vina). |
| **BioLM-Score** | 17M | SZU-ADDG/BioLM-Score | **Protein-ligand scoring oracle** — drop-in upgrade for Lambda's r_vina_proxy. Language-prior-conditioned probabilistic geometric potentials. |
| **MLM-Scaling** | 14M | SZU-ADDG/MLM-Scaling | FRAGPT-like molecular LM scaling laws. Backup if Lambda's proof-search doesn't scale. |
| **SoftMol** | 73M | SZU-ADDG/SoftMol | Block-diffusion alternative to FRAGPT-MCTS. |

**Total:** 6 repos / ~157 MB. SynFlowNet + SyntheMol are the **highest priority** — their reaction abstractions are direct fits to Lambda's RDKit-SMARTS architecture.

---

## Why these 6 (and what we skipped)

| Skipped | Why |
|---|---|
| Chemlambda / AlChemy / chemSKI | **Molecular computing** literature (Ackermann function etc.), not drug design — wrong field |
| Llamole / SynLlama / mCLM | **LLM-based** generation, not MCTS-compatible — incompatible architecture with our proof_search |
| DiffSBDD / TargetDiff / DecompDiff / TransDiffSBDD / MolCRAFT | 3D diffusion models — already have refs (`DiffDock`, `Pocket2Mol`, `targetdiff`, `FlowDock`, `FLOWR`); Lambda needs 2D-first, not 3D-first |
| ClickGen, MolDrug | Specialized to single reaction types — narrower than SyntheMol's 13 |
| 3D-MCTS (VGAE-MCTS) | Graph-generative, not reaction-rule-based; wrong abstraction layer |

---

## Integration phases (5 phases, all using ultracode workflows)

### Phase 1 — SyntheMol Reaction + QueryMol (P0, ~3 h)
**Import:** `synthemol/reactions/reaction.py` (Reaction class), `query_mol.py` (QueryMol class), `real.py` (13 SMARTS reactions).
**Adapter:** `molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py` — wraps Reaction + QueryMol with `MoleculeClosedTerm` integration.
**Wiring:** Extend Lambda's `REACTION_RULES` (currently 5 click rules) with the 13 SyntheMol REAL SMARTS. Each gets a SMARTS + a building-block sanitizer hook.
**Verify:** pytest — 13 reactions fire on real SMILES, building blocks embed successfully, can_apply returns True for valid pairs.
**Report:** `molmetal/reports/lambda_syntemol_reactions_integration.md`.

**Expected gain:** Reaction rule count **5 → 18** (3.6×), branching factor **1020 → ~3700** per pocket.

### Phase 2 — SynFlowNet ReactionTemplateEnv (P0, ~4 h)
**Import:** `synflownet/envs/synthesis_building_env.py` (ReactionTemplateEnv) + `utils/synthesis_utils.py` (Reaction wrapper).
**Adapter:** `molmetal/molmetal_lam/sbdd_env/synflownet_env.py` — adapter taking `MoleculeClosedTerm` and applying forward/backward reaction steps.
**Cross-verify:** Same reaction on same SMILES in both SyntheMol wrapper (Phase 1) and SynFlowNet wrapper (Phase 2) should produce identical products. If not, document the divergence.
**Verify:** pytest 5+ — forward step, backward step, no-match, multiple products, post-reactions.
**Report:** `molmetal/reports/lambda_synflownet_env_integration.md`.

**Expected gain:** Backward planning (retrosynthesis from target) — Lambda currently only goes forward.

### Phase 3 — BioLM-Score as r_vina_proxy (P1, ~1 day)
**Import:** BioLM-Score's probabilistic geometric potential model.
**Adapter:** `molmetal/molmetal_lam/sbdd_env/biolm_score_adapter.py` — wraps BioLM-Score for protein-ligand scoring.
**Wiring:** Replace r_vina_proxy in `RewardAggregator` with r_biolm_score. Adds language-prior conditioning (better than plain Vina scoring for novel chemotypes).
**Verify:** Score 5 SMILES against 1 pocket, verify gradient non-zero, deterministic.
**Report:** `molmetal/reports/lambda_biolm_score_integration.md`.

**Expected gain:** r_vina_proxy is currently a 0-value placeholder (Phase 1 of round-2). BioLM-Score is real SOTA-grade protein-ligand scoring → real reward gradient → real PUCT signal.

### Phase 4 — 100-pocket sweep with 13 reactions (P0, ~24 h)
With Phases 1-3 done, re-run R4-C. Target:
- Vina mean ≥ −7 kcal/mol (match Pocket2Mol −7.15)
- Success rate ≥ 24% (match Pocket2Mol 24.4%)
- SA mean < 3 (better than SOTA 2.51-2.86)
- Synthesizability ≥ 95% (SyntheMol paper claim ~99%)
- 100 pockets × 1000 sims × L-3 204-tile library × 13 reactions

**Report:** `molmetal/reports/lambda_round3_full_sweep.md`.

### Phase 5 — Comparative benchmark (P0, ~1 week)
Three-axis comparison vs SOTA:
1. **Synthesizability** — feed generated candidates through AiZynthFinder retrosynthesis, report % with valid synthesis paths.
2. **Binding** — dock top-100 candidates against CrossDocked targets via Vina (or AutoDock-koto, since it's already at `szuaddg.com/research/`).
3. **QED/SA distribution** — generate 1000 candidates, compute mean + std.

**Targets:**
- Synthesizability ≥ 95% (SyntheMol claim ~99%)
- Vina within 1 kcal/mol of TransDiffSBDD
- SA < 3, QED > 0.55

**Report:** `molmetal/reports/lambda_sota_comparison_round3.md`.

---

## What's NOT in scope (deferred)

- **MLM-Scaling / SoftMol**: backup architectures only — invoke only if Phase 4 reveals Lambda's MCTS doesn't scale to 100-pocket.
- **RxnFlow**: flow-matching alternative — invoke only if Phase 4's success rate is below 24% and we need a different generative backbone.
- **3D coordinate generation**: would require RDKit ETKDGv3 confs + RMSD-to-crystal-pose comparison — too big for round-3; defer to round-4.
- **DiffDock / EquiBind / TankBind (already in refs)**: only invoke if BioLM-Score adapter fails (Phase 3 fallback).

---

## Constraint compliance

- uv pip only (NO plain pip)
- ROCm 7.2 + Triton 3.8 must remain available (syntemol_reactions module is pure RDKit, no torch dep change)
- NO commits, NO running other people's weights
- SyntheMol's Chemprop model weights are NOT loaded (we use its Reaction class only)

---

*Last updated: 2026-09-12. See `molmetal/references/{SyntheMol,SynFlowNet,RxnFlow,BioLM-Score,MLM-Scaling,SoftMol}/` for the actual code.*