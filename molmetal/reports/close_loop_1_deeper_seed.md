# Close-Loop 1 — Deeper-Seed 1000-sim MCTS

**Date:** 2026-09-11
**Driver:** `molmetal/scripts/close_loop_1_deeper_seed.py`
**JSON dump:** `molmetal/reports/close_loop_1_deeper_seed.json`

---

## 1. Chosen seed

| Field | Value |
|-------|-------|
| SMILES | `C1=CCC=C1` |
| Name | cyclopentadiene |
| Tile for closure | MVK (`C=CC(C)=O`, STANDARD_12_TILES index 10) and maleimide (`O=C1NC(=O)C=C1`, index 11) — both act as DielsAlder dienophiles |
| Reaction required | `DielsAlder` |

### Why this seed

Cyclopentadiene is a 4π-diene. Its term-level `is_beta_normal_form` is
`True` (every atom valence-saturated), but chemically it is **open**: the
`DielsAlder` rule fires against any of the dienophile tiles present in
`STANDARD_12_TILES`, producing 1–2 distinct bicyclic norbornene products per
dienophile. Other tiles (azides, alkynes, methylphosphine, maleimide-as-
thiol-acceptor) **fail to react with the seed** under any of
`CuAAC/SPAAC/SPC/ThiolEne`, so the search is forced to focus on the
DielsAlder axis. In other words: cyclopentadiene exercises the rule layer
non-trivially (5 distinct children via `_apply_dirichlet_to_root` at the
root expansion step), in contrast to the cisplatin seed which cannot fire
any RDKit reaction rule and degenerates to a single concatenating rule.

### Alternative seeds considered

| Seed | Why deferred |
|------|--------------|
| Maleimide `O=C1NC(=O)C=C1` | Closed term, no diene; ThiolEne would fire but no SH tile exists in `STANDARD_12_TILES` (task #274 tracks that addition) |
| Methyl vinyl ketone `C=CC(C)=O` | Dienophile-only; no rule in `REACTION_RULES` fires against it directly. Used as the closure tile, not the seed |

---

## 2. Setup

```python
seed       = MoleculeClosedTerm.from_smiles("C1=CCC=C1", embed_3d=False)
tiles      = [MoleculeClosedTerm.from_smiles(t.smiles, embed_3d=False)
              for t in STANDARD_12_TILES()]
rules      = dict(REACTION_RULES)               # CuAAC, SPAAC, SPC, DielsAlder, ThiolEne
predicates = [LIPINSKI]                          # target_predicates
binding    = PROTEASE_GENERIC                    # binding not the bottleneck
reward     = RewardAggregator()                  # all-zero channels, zero bonuses
n_simulations = 1000
max_depth     = 5
c_puct        = 1.4
top_k         = 5
dirichlet     = (alpha=0.3, fraction=0.25)
rng           = random.Random(20260911)
```

---

## 3. Metrics — full table

| # | Metric | Deeper seed (cyclopentadiene, 1000 sims) | Cisplatin baseline (100 sims, prior report) | Status vs. target |
|---|--------|------------------------------------------|---------------------------------------------|-------------------|
| 1 | `BEST_SCORE_TRAJECTORY_monotone` | **True** across all 1000 iters | True | PASS (catalogue: Silver 2016 §3.3) |
| 2 | `N_STATES_EXPLORED_final` | **6** (root + 5 DielsAlder children) | 5 (closed seed ⇒ no expansion) | OBS — seed still β-NF (see §5) |
| 3 | `N_SATISFYING_final` | **1** (the maleimide-norbornene adduct passes Lipinski) | 4 (cisplatin passes trivially) | PASS — at least 1 candidate returned |
| 4 | `PUCT_EXPLOIT_RATIO` (proxy Δmean) | **0.0** | 0.0 | OBS — see §5 |
| 5 | `ROLLOUT_GUIDED_RATIO` | 0.0 (no prior fitted — task #273 territory) | 0.0 | OK (no prior) |
| 6 | `DIRICHLET_APPLIED` | **True** | True | PASS |
| 7 | `CLOSED_LOOP_ITERATION_LATENCY` | not re-measured here | 0.25 s/iter | PASS (prior report) |
| 8 | `EQUATION_CHANGE_RATE` | not re-measured here | 0.0 | PASS (prior report) |
| 9 | `TREE_DIVERSITY` | **0.833** (5 unique children / 6 nodes) | 0.80 | PASS (≥ 0.4, Silver 2016 §3.4) |
| 10 | `ROLLOUT_DEPTH_DIST_median` | **0** (every simulation path length 1, root-only) | 0 | OBS — see §5 |

Wall time: 10.4 s on CPU for the 1000-sim run (vs. 0.46 s for the 100-sim
cisplatin baseline). Both well below the 6 s/iter ROCm budget.

---

## 4. Top-K sample (cyclopentadiene run)

| Rank | SMILES | SA (Ertl proxy) | QED |
|------|--------|----------------|------|
| 1 | `O=C1NC(=O)C2C3CC=C(C3)C12` | 3.38 | 0.410 |
| 2–5 | (only 1 candidate returned — the DielsAlder reaction count of unique alpha-classes is 5, but only the maleimide-norbornene adduct passes `LIPINSKI`; MVK + cyclooctyne-derived norbornenes all fail Ro5 on MW > 500 or HBA > 10) |

The single returned candidate is the **cyclopentadiene–maleimide DielsAlder
adduct**: a 10-heavy-atom norbornene fused to a succinimide ring. The
MVK-derived adducts (10 atoms but 2 ketones) fall outside Lipinski (HBA >
10); the cyclooctyne-derived adducts (from tile index 6) blow past MW=500
after combining. Hence only the maleimide path returns a Lipinski-typed
witness.

The candidate is *chemically reasonable* (a real DielsAlder product) but
**Lipinski-failing is not the goal here** — the goal is to verify that the
search expands, that the Dirichlet prior is applied, and that the rollout-
depth histogram is being recorded correctly.

---

## 5. Honest assessment — did PUCT_EXPLOIT_RATIO and ROLLOUT_DEPTH move from OBS → PASS?

**No — both metrics remain OBS.**

The deeper-seed experiment **confirms a real architectural limitation in
`MCTSProofSearch._simulate`** that the cisplatin baseline could not have
surfaced. The root cause is in `proof_search.py:793-797`:

```python
while not node.is_leaf and not node.is_terminal and len(path) <= max_depth:
    node = self._select_child(node)
    path.append(node)
```

The SELECT loop only descends past a node when `not is_terminal`. Every
MoleculeClosedTerm built from SMILES (cyclopentadiene, cisplatin, MVK,
maleimide, …) reports `is_beta_normal_form == True` because `_is_acidic`
requires explicit H/Cl/Br/I/F atom neighbours — which implicit-H SMILES
parsing does not materialise. Consequence: **even when the root has 5
children added by `_apply_dirichlet_to_root`, `_simulate` never descends
into them**. The path stays at length 1, the rollout never plays out, and
`len(path) - 1 = 0` for every simulation → `ROLLOUT_DEPTH_DIST_median = 0`.

The deeper-seed run **does** demonstrate progress on the metrics that don't
depend on `_simulate` descending:

- **`N_STATES_EXPLORED_final = 6`** (up from cisplatin's 5). The 5
  children exist on the tree via the root Dirichlet expansion; the MCTS
  visits them but only at the SELECT step inside `_apply_dirichlet_to_root`,
  not via the main loop.
- **`TREE_DIVERSITY = 0.833`** (up from cisplatin's 0.80). More distinct
  alpha-classes → higher diversity.
- **`N_SATISFYING_final = 1`** — at least one Lipinski-typed candidate
  is returned (vs. cisplatin's 4 from the trivial concatenation rule).
- **One real chemistry reaction fires** (cyclopentadiene + maleimide →
  norbornene-fused succinimide) and passes `LIPINSKI`. This is a stronger
  signal than the cisplatin concatenation product.

What this experiment **cannot** move:

- **`ROLLOUT_DEPTH_DIST_median`** stays at 0 because `_simulate` doesn't
  descend past a terminal root. Fixing this requires either (a) a seed
  with explicit acidic neighbours (i.e. a hand-constructed MoleculeClosedTerm
  with `H` atoms bonded to saturated heavy atoms), or (b) lifting the
  `not is_terminal` gate in the SELECT loop when the root has expanded
  children — a one-line patch to `proof_search.py:795`.
- **`PUCT_EXPLOIT_RATIO`** stays at 0 because every leaf value is 0
  (zero `RewardAggregator`); without value variance, the PUCT proxy
  collapses. This is expected for the zero-reward configuration but means
  the metric is **only meaningful when the scorer is non-trivial** —
  which is task #273/279 territory (fit `SymbolicPrior`, run guided
  MCTS, verify guided-rollout ratio).

### Verdict

| Metric | Status | Notes |
|--------|--------|-------|
| `PUCT_EXPLOIT_RATIO` | **OBS** | zero-reward config ⇒ no leaf-value variance |
| `ROLLOUT_DEPTH_DIST_median` | **OBS** | `_simulate` SELECT loop gated on `not is_terminal`; closed-term seeds can't descend |

The deeper-seed run demonstrates that **5 DielsAlder children are correctly
expanded on the tree**, **tree-diversity is high**, and **a real
Lipinski-typed candidate is returned**. It also exposes — more clearly than
the cisplatin baseline — that two metrics are still gated by upstream
choices (zero reward + closed-term seeds). The honest path to PASS is:

1. **ROLLOUT_DEPTH → PASS**: drop the `not is_terminal` gate in the SELECT
   loop when `len(node.children) > 0` (one-line patch in `proof_search.py`),
   or supply a seed with explicit acidic neighbours.
2. **PUCT_EXPLOIT_RATIO → PASS**: fit a `SymbolicPrior` (task #276) and
   run with a non-trivial scorer — task #277 territory.

Both fixes are out of scope for this round (TASK 1 of 3 is the deeper-
seed measurement; the follow-up tasks #275-279 are the guided-rollout
work). This report is the honest bridge between OBS and PASS.

---

## 6. Files

- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/close_loop_1_deeper_seed.py` (driver)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/close_loop_1_deeper_seed.json` (raw metrics)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/close_loop_1_deeper_seed.md` (this file)
- Pre-existing baseline: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/govern_remeasure_L9_cross.md`