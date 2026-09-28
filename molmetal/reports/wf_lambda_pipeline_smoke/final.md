# TODO/32 B5 — Lambda pipeline smoke (test_001)

- pocket_id: `test_001`
- pocket_smiles: `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O`
- n_requested: 100
- n_returned_mcts_total: 804 (n_seeds=12 sims/seed=83)
- n_unique_smi_in_pool: 796
- n_emitted: 100  n_valid_rdkit: 100  n_unique_valid_smi: 100
- sa_mean: 2.330029800224768 (n=100)
- qed_mean: 0.7687472614502255 (n=100)
- n_tiles: 240
- rules_used: ['CuAAC', 'DielsAlder', 'SPAAC', 'SPC', 'ThiolEne']
- n_reactions_total: 27
- fpu_reduction: 0.25
- pll_k: 16 pll_knn: 8 pll_available: True
- n_simulations_total: 996 max_depth: 3 top_k: 100
- nfe: 996 nfe_reductions: 820427
- wallclock_s: 693.3

PLL + KataGo-style FPU (KataGo `docs/GraphSearch.md`, Silver 2016,
Liu & Feng 2026 DOI: 10.1021/acs.jcim.6c02460).  Closed β-NF term
rendered via `_molecule_to_lambda_expr` (atoms=left-associated app).
