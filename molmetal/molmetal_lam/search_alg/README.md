# `search_alg/` — MCTS proof search over the Molecular Lambda Calculus

Layer 8 of the MLC framework. Drug-design-as-proof-search: each β-reduction
step = one reaction rule applied to two tiles. The central class is
`MCTSProofSearch`, an AlphaGo-style **PUCT** searcher (PUCT + virtual loss
+ Dirichlet noise + FPU + policy pruning) over closed λ-terms.

The module is **dependency-light at import time** — RDKit and torch are
imported lazily so the module loads in any environment.

## Key files

| File | Role |
|------|------|
| `proof_search.py` | `MCTSProofSearch` — main UCB-tree driver (M3). |
| `puct.py` | PUCT selection rule, virtual loss, FPU reduction. |
| `learned_prior.py` | `SymbolicPrior` — wraps `HeuristicRegressor` so P(a\|s) is learned. |
| `pll_heuristic.py` | PLL heuristic for MCTS rollouts (M4). |
| `pareto.py` | Pareto-front tracking across objectives (QED, SA, dock). |
| `sweep_guidance.py` | Parameter sweep orchestration (cfg, OT, prior). |
| `warm_start.py` | Replay-buffer warm start for the MCTS tree. |
| `priors/` | Pre-trained symbolic priors. |
| `_batch_reward_worker.py` | Multiprocess reward evaluation. |

## Usage

```python
from molmetal_lam.search_alg import MCTSProofSearch
searcher = MCTSProofSearch(initial_term=closed_T, target_predicates=[Lipinski, QED],
                            binding_site=pocket, scorer=vina_qed_sa)
best_mol = searcher.search(n_simulations=1000, c_puct=1.4)
```

```python
from molmetal_lam.search_alg.puct import select_child
node = select_child(root, c_puct=1.4)  # PUCT + virtual loss
```

## Common pitfalls

* `MCTSProofSearch.search` returns the **highest-scoring** term, not the
  most-visited one — pass `return_mode="visits"` for the AlphaZero default.
* Use **persistent tree** across calls (`warm_start.persistent_tree`) to
  amortize cost over R17 sweeps.
* Mechanism ↔ metric cross-refs: see `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`
  for **M3** (PUCT/MCTS) and **M4** (PLL heuristic).
* Tests: `pytest molmetal/tests/test_proof_search_strengthened.py -q`.