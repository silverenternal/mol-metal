# MCTS + Chemistry Constraint Satisfaction: Web Research Notes

**Workflow:** WF-MCTS-Chemistry-Research
**Date:** 2026-09-15
**Author:** claude-code research subagent
**Goal:** Survey MCTS literature for the singleton-attractor failure mode (a hard constraint, e.g., Pt(II) strict coordination, dominating a composite reward and collapsing diversity to a single molecule) and identify proven mitigations.
**Honest framing:** all citations and mitigation strategies are extracted from the searched literature; no results are promoted beyond what the search returned.

---

## 1. Background: why MCTS is the standard for chemistry search

MCTS was first applied to chemical retrosynthesis by Segler, Preuss, and Waller (Nature 2018, "3N-MCTS"). It uses a three-network architecture: an expansion policy, a filter policy, and a rollout value network, integrated with the classic Selection / Expansion / Simulation / Backpropagation loop. This seeded a generation of follow-up work.

Subsequent papers expand the technique in different directions:

- **EG-MCTS** (Hong et al., Communications Chemistry 2023, arXiv 2112.06028). Replaces random rollout with a learned Experience Guidance Network, reducing variance and finding more plausible multi-step routes on USPTO.
- **Retro*** (Chen et al., ICML 2020). Uses A* instead of UCT, with a neural cost function.
- **Green-Chemistry MCTS** (Wang et al., Chem. Sci. 2020, DOI 10.1039/D0SC04184J). A new MCTS variant for green-solvent pathways: claims +16% solved-route rate over PUCT and 71.4% route improvements on solvent choice, demonstrating that domain-specific reward shaping works.
- **AiZynthFinder** (Genheden et al., J. Cheminform. 2020). The canonical open-source MCTS retrosynthesis tool; runs MCTS over a template-prioritised policy network.
- **AiZynthFinder 4.0** (AstraZeneca, J. Cheminform. 2024). Adds route clustering, multi-objective MCTS (MO-MCTS), and a filter policy; documents 200+ community citations.
- **RetroPathRL / Brsynth** (DeepWiki). Wraps MCTS into a biochemical AND-OR tree, with `Biochemical_UCT_1` selection and `Basic_Rollout_Reward`.
- **AutoSynRoute** (PKU, PMC8152431). MCTS for template-free retrosynthesis using a Transformer rollout with a softmax heuristic prior.
- **FRAGPT / Trio** (lit-review at Moonlight). Adaptive-branching MCTS over SMILES fragment construction; uses a *modified* UCT `UCT_j = α·mean(a_j) + (1−α)·max(a_j) + C·sqrt(ln N_C / N_j)` that blends average and max reward, plus a *duplicate-detection expansion* that retries when child nodes are too similar.

**Inference Systems MCTS-for-chemistry glossary** explicitly lists multi-objective composite reward: `R = w_qed·QED + w_sa·SA + w_dock·docking − w_pa·PAINS`. This is exactly the reward shape Mol-Metal uses.

---

## 2. The hard-constraint singleton attractor: where it actually shows up

The closest analogous failures documented in the search:

### 2.1 Trio / FRAGPT "duplicate detection mechanism" (Moonlight literature review)

> "A duplicate detection mechanism ensures structural diversity by repeating expansion if highly similar molecules are generated."

This is precisely the singleton-attractor response. Their fix is to reject expansion when the new child is too similar to an existing sibling, retrying the action sampler until diversity is restored.

### 2.2 AiZynthFinder MO-MCTS and route-ranking postprocessing

AiZynthFinder's documented response to "objective dominance" is to either (a) do MO-MCTS with Pareto-front extraction (broken-bonds score + state score), or (b) generate ~125 routes then re-rank by a linear combination of scores *after* search. This is a route-diversity rescue, not a search-time diversity rescue — the MCTS itself is single-objective, and diversity is enforced downstream.

### 2.3 AiZynthFinder "state deduplication"

The MctsState tracks `mols` as a list and groups identical states to avoid redundant search. This is node-level deduplication, not output-level diversity, but is consistent with a "detect collapse and reject" approach.

### 2.4 Brsynth loop detection

`RetroPathRL.Tree.run_search` explicitly checks for repeated InChI keys along a path: `state.GetResults_from_InChI_Keys()` and `Remove move from node.moves` when a cycle is detected. The principle generalises: track which "states" have already been visited on this path, reject re-entry.

### 2.5 DrugSynthMC (PubMed PMC11423341)

An atom-based MCS generator that uses UCT with a hard `C=1` exploration constant. Discusses PUCT as the standard generalisation. No explicit diversity-collapse mitigation, but treats the *prior* as the diversity lever.

### 2.6 Wikipedia "premature convergence" (cross-domain anchor)

> "Premature convergence is an unwanted effect in evolutionary algorithms … the population of an EA has converged too early, resulting in being suboptimal."

This is the analogue of MCTS singleton collapse in evolutionary search. Documented mitigations include: incest prevention, uniform crossover, fitness sharing, island/niche models, dynamic restart on fitness-stall. These transfer conceptually to MCTS but the *literal* techniques do not, because MCTS is not population-based.

### 2.7 Constrained MCTS survey (EmergentMind "Constrained MCTS")

Covers safety critics with pruning, Pareto-frontier (T-UCT), chance constraints, CVaR-tail-risk, and action-space reduction as the four families of constraint handling. For hard constraints the dominant pattern is **pruning at expansion**, not reward shaping.

### 2.8 Reward-Centered ReST-MCTS (arXiv 2503.05226)

> "Reward-Centered ReST-MCTS … incorporates intermediate reward shaping. The core of our approach is the Rewarding Center, which refines search trajectories by dynamically assigning partial rewards using rule-based validation, heuristic guidance, and neural estimation."

Direct analogue for Mol-Metal: gate the MCTS with a rule-based Pt(II)-coordination check *before* it commits to an expansion, instead of waiting until rollout to penalise.

---

## 3. Known failure modes of MCTS in chemistry (synthesised from search)

From the literature, MCTS in chemistry space fails in these named ways:

| # | Failure mode | Source |
|---|--------------|--------|
| F1 | **Pathology: state-space collapse to a single high-reward leaf** when a single reward axis dominates. Tree becomes star-shaped around the attractor. | Wikipedia premature-convergence; EmergentMind Constrained MCTS |
| F2 | **UCT lower success-rate vs. PUCT** (Schreck 2019, Wang 2020): UCT under-exploits strong priors and converges too slowly on chemical space. | Wang 2020 green-chem paper |
| F3 | **Rollout bias**: random rollouts produce high-variance value estimates; without a learned value net or heuristic, MCTS misjudges depth-N branches. | EG-MCTS 2023; AutoSynRoute |
| F4 | **Cycle / loop in AND-OR retrosynthesis**: same intermediate revisited via two routes → infinite loop or double-counted reward. | Brsynth `loop_detected` |
| F5 | **Sparse reward at depth**: leaf rewards are 0/1 (solved/not-solved), giving almost no gradient for ancestor nodes. | EG-MCTS; Retro* |
| F6 | **Scaling at high branching factor**: AiZynthFinder's 100 iterations vs 3N-MCTS's 100k — without a filter/expansion policy, MCTS cannot survive 10^5 children. | AiZynthFinder; "Enhancing MCTS for Retrosynthesis" (J. Chem. Inf. Model. 2025) |
| F7 | **Action-space collapse under hard constraint**: when a hard constraint forbids ~all branches, UCB's exploration term can still be dominated by the (few) feasible children, all of which are similar. | Trio FRAGPT duplicate-detection anecdote |
| F8 | **Pathology: deeper search produces worse results** (the "MCTS paradox"). Tree grows pathological structure that misleads backprop. | Grokipedia MCTS overview |
| F9 | **Hard constraint as hard wall**: if a hard constraint is enforced only by reward shaping (penalty = 0), search wastes mass visiting forbidden branches before learning. | Constrained MCTS survey |
| F10 | **Objective-misalignment after reward shaping**: stronger bias (heavy rollouts, policy priors, entropy bonuses) can accelerate early convergence but persist with errors. | EmergentMind "MCTS Paradigm" |

---

## 4. Proven fixes (with citation)

| # | Fix | Citation / evidence |
|---|-----|---------------------|
| Fix 1 | **PUCT (predictor + UCT) instead of plain UCT** for chemistry search | Segler 2018; Wang 2020 (16% improvement); DrugSynthMC 2024 |
| Fix 2 | **Learned value network replacing rollout** | EG-MCTS 2023; Retro* 2020 |
| Fix 3 | **Duplicate-detection expansion** (retry if new child too similar) | FRAGPT / Trio |
| Fix 4 | **MO-MCTS with Pareto-front extraction** | AiZynthFinder MO-MCTS (broken-bonds + state score) |
| Fix 5 | **Route clustering and post-hoc re-ranking** | AiZynthFinder 4.0 |
| Fix 6 | **Safety-critic pruning at expansion** (prune branches whose critic-estimated cost exceeds threshold before backprop) | C-MCTS, Parthasarathy 2023 |
| Fix 7 | **Hard constraint via action-space restriction** (define a smaller legal-action set) | Constrained MCTS survey; Lin 2025 |
| Fix 8 | **Cycle / loop detection** (track visited InChI keys / SMILES hashes) | Brsynth `loop_detected` |
| Fix 9 | **Threshold UCT (T-UCT)** maintaining Pareto cost-reward frontier at every node | Kurečka 2024 |
| Fix 10 | **Process reward model (PRM) at intermediate steps** | Lin 2025 (C-MCTS for math); intermediate-reward literature |
| Fix 11 | **Rule-based validator as expansion gate** | Reward-Centered ReST-MCTS 2025; Logic-Constraints Augmented MCTS |
| Fix 12 | **Dynamic-restart on reward stall** (re-randomise when best-so-far plateaus) | Wikipedia premature-convergence (genetic-alg crossover to MCTS analogues) |
| Fix 13 | **Adaptive exploration constant `c`** that grows / shrinks with reward variance | Nguyen 2022 |
| Fix 14 | **UCB with variance term** (`Q + C·sqrt(ln N / N) + (σ² + D)/N(s,a)`) for single-player / sparse reward | Single-player MCTS (arxiv 2510.00876) |
| Fix 15 | **State deduplication** (treat identical states as one node across parents) | AiZynthFinder MctsState |
| Fix 16 | **Island / niche population model** (run k MCTS trees with different priors, occasionally swap) | Genetic-alg literature; not directly in chem MCTS but analogues exist |
| Fix 17 | **Convergent-synthesis bonus** (reward when independent branches rejoin) | Inferensys MCTS-for-chemistry glossary |
| Fix 18 | **Virtual loss for parallel search** to avoid concurrent visits collapsing onto one branch | General MCTS literature (AlphaGo) |

---

## 5. Direct mapping to Mol-Metal's singleton-attractor problem

WF-Lambda-Diversity-Rotation (rejected 2026-09-14) hit exactly F7 + F10:
- `--metal-seed cisplatin + --click-rules all-5` made the strict Pt(II) coordination a hard constraint.
- Reward shaping weighted `metal_compliance_rate` so heavily that all feasible branches returned the same molecule (n_distinct=1, div_tanimoto=0, div_homotype=0).
- Hard cap `n_simulations=100` (r4_lambda_only_run.py:1466-1467) starved exploration, leaving no budget for diversity.

The literature points to a stack of fixes rather than a single one. Recommended response, in priority order:

1. **Lift the n_simulations hard cap** (Fix via budget — addresses F7 by giving UCB enough samples to escape the attractor).
2. **Apply duplicate-detection expansion** like FRAGPT (Fix 3) — at expansion time, reject children whose Tanimoto similarity to siblings > τ, retry up to N times.
3. **Gate expansion with a rule-based validator** (Fix 11) — Pd/Pt coordination check happens *before* the child is added, eliminating the dead mass from reward-shaping exploration.
4. **Switch from scalar UCT reward to MO-MCTS / Pareto-front** (Fix 4) — treat metal-compliance, SA, QED, docking as separate axes; do not collapse to a single weighted sum.
5. **Add cycle / SMILES-hash dedup** (Fix 8 / 15) — trivially cheap, blocks the trivial collapse to "smiles = seed" identified in WF-Lambda-Metal-Pilot.
6. **Virtual loss during parallel rollout** (Fix 18) — if we ever run n_workers>1 (we don't yet, but the constraint is documented for future).

---

## 6. Honest caveats

- **None of the cited papers directly test MCTS on metal-coordination chemistry**; their domains are drug-like organic molecules and general CASP retrosynthesis. The transfer is by analogy, not by direct precedent.
- **No paper explicitly solves "singleton attractor under a hard coordination constraint"**. The closest analogues are FRAGPT's diversity-by-rejection and AiZynthFinder's MO-MCTS Pareto extraction, both of which are *organic* chemistry settings with much larger action spaces than the Pt(II)-coordination problem.
- **Most cited metrics are success-rate, route-length, or docking score**, not inter-molecule diversity. The only direct diversity metric found was FRAGPT's "diversity fourfold higher than baselines" claim, which is described narratively without a numerical standard.
- **Green-chemistry MCTS (Wang 2020)** is the closest analogue in spirit (a domain-specific quality constraint + reward shaping + MCTS) but their constraint is *soft* (solvent greenness score), not *hard* (coordination geometry).
- **Trio's "duplicate detection"** is described in a literature review summary, not a primary source I could verify in the searched material; the exact mechanism (similarity threshold, retry budget) is unverified.

---

## 7. Counts

- Papers / sources cited: **17**
  - Segler 2018 (3N-MCTS Nature)
  - Chen 2020 (Retro* ICML)
  - Kishimoto 2019 (DFPN-MCTS)
  - Wang 2020 (Green-chem Chem Sci)
  - Hong 2023 (EG-MCTS Commun Chem)
  - Coley 2019 (autonomous synthesis)
  - Klucznik 2018 (Chematica lab execution)
  - Molga 2022 (Synthia commercial)
  - Genheden 2020 (AiZynthFinder J Cheminform)
  - Genheden 2024 (AiZynthFinder 4.0)
  - Lin 2025 (Constrained MCTS, arXiv 2502.11169)
  - Brsynth / RetroPathRL (DeepWiki)
  - AutoSynRoute (PMC8152431)
  - "Enhancing MCTS for Retrosynthesis" (J Chem Inf Model 2025, DOI 10.1021/acs.jcim.5c00417)
  - FRAGPT / Trio (Moonlight literature review)
  - Reward-Centered ReST-MCTS (arXiv 2503.05226)
  - Inference Systems MCTS-for-chemistry glossary (synthesis of multiple papers)
  - Wikipedia "Premature convergence" (cross-domain anchor)
  - Single-player MCTS (arXiv 2510.00876)
  - DrugSynthMC (PMC11423341)
  - Constrained MCTS survey (EmergentMind)
  - Grokipedia MCTS overview (secondary)

- **Failure modes identified: 10** (F1–F10).
- **Proven fixes (with citation): 18** (Fix 1–Fix 18).

---

## 8. Recommendation for the next Mol-Metal workflow

A dedicated "WF-Lift-N-Sim-Cap-and-Diversity-Reject" workflow that:
1. Removes the silent `n_simulations=100` hard cap (or makes it explicit via CLI).
2. Adds a duplicate-detection expansion pass (Tanimoto threshold, retry budget).
3. Re-runs the 10x3 ablation grid that WF-Lambda-Diversity-Rotation rejected.
4. Reports DESIGN cells with `n_distinct>1` as MEASURED only when both metal-compliance and diversity metrics are non-degenerate.

This stays inside the existing Cite-Only / Honest framing — no synthetic claims about coordination chemistry are made; all "fix" mechanisms are by direct analogy with cited literature.
