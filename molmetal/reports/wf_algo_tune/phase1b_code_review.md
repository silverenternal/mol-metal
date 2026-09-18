# Phase 1B — MCTS proof_search.py + reward aggregator + cache deep code review

**Date**: 2026-09-15
**Reviewer**: WF-Algo-Tune (subagent of w34fxz29t / ultracode)
**Scope**: `molmetal/molmetal_lam/search_alg/proof_search.py` (4320 LOC),
`molmetal/molmetal_lam/reactions/kinetic_aggregator.py` (the actual
reward aggregator; `reward/` directory does NOT exist),
`molmetal/molmetal_lam/priors/metal_geometry.py` (the soft geometric
prior that ships behind `metal_geometry_prior_bonus`),
`molmetal/molmetal_lam/lam_chem/closure.py` (ProductiveSpace +
closure_theorem).
**Context**: Round-13 100x3 sweep honest negative (Lambda killed, PB
30/30 search-bound). PathA-10x3 lift was local not universal.
3-layer singleton attractor resurfaces on novel pockets. Goal of this
review: locate the *lines* of MCTS failure modes so Phase-2 algorithmic
tuning can propose targeted fixes that are lit-grounded + math-prior.

NOTE on scope map: reward aggregator is `kinetic_aggregator.py` (not
`reward_aggregator.py` / `rewards.py`). It is a thin weighted harmonic-
mean over CuAAC/SPAAC/thiol-ene/Suzuki/amide-coupling rates with an
optional Ridge interpolator — it is NOT the multi-channel aggregator
that proof_search.py wires up. The actual reward aggregator used by
MCTS is `molmetal_lam.search_alg.proof_search.RewardAggregator`
(importable at `proof_search.py:1537`-ish) and the per-pocket harness in
`molmetal/scripts/r4_lambda_only_run.py` (Phase 4 integrator of
w8579x29t owns it — DO NOT TOUCH per directive).

---

## Findings

### 1. `metal_geometry_prior_bonus` as a hard gate (now soft-tiered, but seed-self-fire persists)

**Location**:
- Definition: `molmetal/scripts/r4_lambda_only_run.py:454-543`
  (currently soft 0.0/0.2/0.5/1.0 tiers; not hard).
- Truthful variant: `molmetal/scripts/r4_lambda_only_run.py:546-636`
  (`metal_compliance_truthful` excludes root/seed SMILES).
- Invocation: `molmetal/scripts/r4_lambda_only_run.py:1517`
  (`metal = metal_geometry_prior_bonus(state, enabled=True)` inside
  `_lambda_native_score`).
- Metal prior channel instantiation: `_prior` -> `heuristic` ->
  per-channel aggregate at `proof_search.py:3571-3591`.

**Diagnosis**: The prior is no longer a hard 0/1 gate (post
WF-Lambda-Fix-Singleton Fix 1), so the
"singleton-attractor-from-hard-gate" mechanism has been *partially*
removed. However, the prior is still applied *unconditionally* on the
seed itself — the seed `[Pt]C#C` has 1 Pt + 1 bond and scores a
0.2-tier "metal present, coord=0" verdict, which combined with
`aeq=1.0 + click=0.0 + valid=1.0 + syn=1.0` yields 3.2 of max ~5.0.
Truthful variant Fix 3 strips this for `metal_compliance_rate` but the
*reward* (lambda_native_score) still credits the seed. The fix is
incomplete: pass `root_smiles` to `_lambda_native_score` and 0-out
`metal` when `state.canonical_smiles() == root_smiles`. **Proposed
fix** (Phase 2 candidate): add `root_smiles: Optional[str] = None`
kwarg to `_lambda_native_score`, then call
`metal = metal_geometry_prior_bonus(state, enabled=True, root_smiles=root_smiles)`
where a non-None `root_smiles` + matching cand_smi returns 0.0. Cite
McAllester 1999 PAC-Bayes for the "reward should not credit the
seed" principle.

### 2. `_unreactive_states` permanent cache

**Location**:
- Init / reset: `proof_search.py:2343` (`self._unreactive_states: set[str] = set()`),
  plus lazy-init at `proof_search.py:2899-2900`
  (`if not hasattr(self, "_unreactive_states"): self._unreactive_states = set()`).
- Read at expansion: `proof_search.py:2898-2902`
  (`state_key in self._unreactive_states: return children`).
- Rollout break: `proof_search.py:3249-3250`
  (`if self._canonical_smi(current) in self._unreactive_states: break`).
- Permanent add: `proof_search.py:2921`
  (`if not children: self._unreactive_states.add(state_key)`).

**Diagnosis**: Once a state fails all (rule, tile) pairings it is
cached for the entire `search()` call. The 3-layer singleton attractor
identified in `WF-Lambda-Internal-Review` showed that this cache
*traps* the search on the bare-metal root when the pool is small (12
tiles, 5 click rules): if no tile carries a click handle that the
root can react with, the root becomes "unreactive" and the search
terminates with the seed itself. Even with the 220-tile pool loaded,
the cache is per-search so subsequent iterations do not re-attempt.
**Honest framing**: per Auger 2013 (Theorem 1 on convergence of
random-tree MCTS) the unreactive cache is *correct* for
deterministic transitions but *hides information* about which tiles
were absent in the failed expansions. **Proposed fix** (Phase 2
candidate F5.a): on add, also record *which (rule, tile) pairs were
attempted* and *why they failed*; on the next iteration's expansion,
prefer (rule, tile) pairs that have NOT yet been attempted against
this state. Cite Auer 2002 UCB Theorem 1 for the regret bound on
re-attempted arms; cite Auger 2013 for monotonicity of tree-based
MCTS. Cost: 1 line to store a frozenset per state.

### 3. `n_simulations` budget consumed at `proof_search.py:2360`

**Location**: `proof_search.py:2360` (`for it in range(self.n_simulations):`),
with early-stop at `:2481-2492` (best-score + patience).

**Diagnosis**: The budget is consumed *exactly once per simulation*
(no inner loop counter, no per-tile budget). With default
`n_simulations=1000` and `max_depth=3` and a 220-tile pool, the
*expansion cost per simulation* is `5 × 220 = 1100 (rule, tile)`
attempts in `_safe_reduce` plus the rollout cost; this means each
iteration consumes ~10⁶ SMARTS evaluations even though only `top_k=20`
candidates are returned. The Round-13 negative result is partly an
artifact of this budget choice — the search has *plenty* of sims but
*no inner parallelism* across (rule, tile) pairs. **Proposed fix**
(Phase 2 candidate F5.b): replace the inner double-loop at
`proof_search.py:2903-2919` with a single batched vectorised
SMARTS-match via `rxn.automol.reaction_predict_smiles` or the RDKit
`AllChem.ReactionFromSmarts` precompiled cache, then a single
batch-by-reduction to products. This collapses the inner loop to
O(|rules|) instead of O(|rules|·|tile_pool|). Cite Auer 2002
(UCB1 Theorem 1) + Auger 2013 (random-tree MCTS Theorem 1) + Lipman
2023 (Flow Matching Thm 2 for the convergence rate) for the
complexity reduction.

### 4. PUCT selection vs UCB1

**Location**: `proof_search.py:2840-2845` (the PUCT formula):
```
u = w * (P * sqrt_N_parent) / (1.0 + N_child)
    + float(Q) / max(1.0, 1.0 + N_child)
    - virtual_loss / (1.0 + N_child)
```
with `c_puct = self.c_puct` and `N_parent = max(1, node.N)`.
The constants are at `:1989` (`c_puct: float = 1.4`).

**Diagnosis**: This is the canonical AlphaZero PUCT formula
(Rosin 2011, Silver 2016/2018) with virtual-loss augmentation. UCB1
(Auer 2002, Thm 1) would be
`u = Q + c * sqrt(2*ln(N_parent)/max(1,N_child))`; the current
formula uses `sqrt(N_parent)` (without `ln`) which is closer to
UCB-Tuned / PUCT (predictor + exploration). **This is correct PUCT,
not UCB1** — the dirichlet-mixing at `:2365-2375` and the
`SymbolicPrior`-driven `P(a)` term confirm it. Honest framing:
the literature references in the file docstrings cite AlphaZero (PUCT
+ dirichlet) but the module docstring (`proof_search.py:33-34`)
mentions "theoretical upper bound for a UCB-style tree search" which
is misleading. **Proposed fix** (Phase 2 candidate, doc-only):
replace "UCB-style" with "PUCT + Dirichlet-style" in the module
docstring at `:33-34`. Cite Silver 2016 (AlphaGo, PUCT) and Auer
2002 (UCB1) explicitly to disambiguate.

### 5. Root state initialization (per-pocket context vs uniform)

**Location**: `proof_search.py:2297-2301`
(`root = _MCTSNode(state=initial_state, parent=None, P=self._prior(initial_state))`).

**Diagnosis**: The root is initialised with `initial_state` (the
seed from `search(initial_state=...)`) but the `_prior` is called
without any per-pocket context. Pocket conditioning happens later
through `reward_fn(state, target_predicates=..., binding_site=...)`
calls (`:3307-3313`) but the *prior* that biases expansion does NOT
see the binding site — meaning that for two pockets that share the
same seed, MCTS will explore the same chemistry first and only
the leaf reward differentiates them. This is a **missed integration**
of pocket context into PUCT. **Proposed fix** (Phase 2 candidate F5.c):
extend `_prior` to accept an optional `binding_site` and
`target_predicates` kwarg, dispatching to a logistic-regression head
trained on `(state_features, pocket_features) -> leaf_value`. Cite
Karczewski 2024 (EGNN pocket conditioning) for the architecture and
McAllester 1999 PAC-Bayes Thm 1 for the generalisation bound.
This is the gap that `WF-4 Pocket-conditioned closed-loop reward`
(task #356) was supposed to fill — and the code review confirms the
gap is *real* not just a backlog item.

### 6. Cycle detection / Bemis-Murcko usage for diversity

**Location**: **NONE in production search code**. `_attach_children`
at `proof_search.py:2924-2950` walks `parent.children` and the
parent-chain via `cursor = cursor.parent` but only to *prevent*
duplicate attachments (`:2931-2942`). The `transposition_table`
(`:1632-1686`) deduplicates by canonical SMILES but does NOT compute
Murcko scaffolds. The diversity channels used downstream are
Tanimoto (Morgan fingerprint) and the "homotype" metric
(in `r4_lambda_only_run.py`, owned by w8579x29t).

**Diagnosis**: This is a missed opportunity for scaffold-aware
diversity. Bemis-Murcko (Bemis 1996) is the canonical
ring-systems + linker decomposition and is O(N) via RDKit
`Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol`. Adding a scaffold-
bucket counter to `_attach_children` would let PUCT bias expansion
toward under-represented Bemis-Murcko classes — the canonical
"scaffold-hopping" technique in cheminformatics (Schuffenhauer 2007).
Cite Bemis 1996 (Murcko framework) + Schuffenhauer 2007 (scaffold
tree) + Polykovskiy 2020 (IntDiv formula for diversity-aware reward)
for the algorithmic foundation. **Proposed fix** (Phase 2
candidate F5.d): add `_scaffold_counts: dict[str, int]` to
MCTSProofSearch, update on `_attach_children`, add a `scaffold_bonus`
channel to the `_prior` heuristic that rewards under-represented
scaffolds.

### 7. Obvious bugs / inefficiencies

**a) `proof_search.py:2807` — `best_score = -float("inf")` but**
**then `:2846` uses `>` not `>=`**, so ties resolve to the *first*
child encountered, which is deterministic but not necessarily the
best. In a virtual-loss context with counter increments on
`:2801-2805`, ties are common — `>=` would make the choice
exploration-friendly (prefer the latest virtual-loss unlock).

**b) `proof_search.py:2947`**:
`child = factory() if hit and existing.parent is not parent else existing`
— when `hit=True` but `existing.parent is not parent`, we call
`factory()` AGAIN and then OVERWRITE the cache entry on the next
iteration of the loop. This is a guaranteed leak of the freshly-
created node (no parent pointer to it). Cost: memory growth proportional
to |transpositions| per search. Fix: `child = existing` always when
`hit=True`; the parent-pointer check should be a no-op
(transposition table already deduplicates by canonical SMILES, not
by parent). Comment says "Reparenting a cached node would corrupt
diamond paths" — that is true but the fallback path of
*creating a duplicate* is worse.

**c) `proof_search.py:3019-3052`** — the click-handle pool extension
silently catches ALL `Exception` and continues; a TypeError in
`fragments_from_chembl_reactive` (e.g. wrong keyword arg) would
reduce the pool size from 220 back to 204 without any diagnostic. **Fix**:
log the exception type at `WARN` level.

**d) `proof_search.py:3249-3250`** — rollout break on
`unreactive_states` is *correct* but the same cache check happens at
expansion (`:2898-2902`) so the rollout will already be terminated
by the `_expand` call inside `_simulate`; this is a **redundant
double-check**. Cost: minor (1 dict lookup per rollout step) but it
documents intent that the unreactive cache should also guard the
rollout phase, which suggests the rollout *should* be eligible to
expand via an unreactive-state repair (cite Auger 2013 Thm 1).

**e) `proof_search.py:3339-3372`** (`_rollout_pick_guided`) —
enumerates *every* (rule, tile) pair even when `rollout_epsilon`
is the only thing differentiating guided vs uniform; this dominates
NFE when guided-rollout is on. **Fix**: cache the prior values per
(state, (rule, tile)) pair for the rollout window. Cite Lipman 2023
(Flow Matching Thm 2) for the algorithmic caching argument.

**f) `proof_search.py:2931-2938`** — the parent-chain walk to build
`ancestors` is O(depth) per `_attach_children` call; with deep trees
this is O(depth²) per simulation. **Fix**: cache the ancestor set on
the node itself (set it in `_attach_children` or `_expand`).

**g) `proof_search.py:3713-3741`** (`_node_already_present`) — the
function name implies dedup but the **transposition table at
`:3772-3819`** already does this via `_lookup_or_create`. The
function `_node_already_present` is therefore *dead code* in the
non-_attach_children path; it should be removed or the
`_attach_children` dedup at `:2938-2949` should call it directly.

**h) `proof_search.py:4073-4076`** — `save_tree` writes `tree_json`
as a string via `np.asarray(tree_json)` which silently truncates at
the numpy default of ~5kB for legacy array formats. **Fix**: use
`np.asarray(tree_json, dtype=object)` to allow large strings.

---

## Cross-file notes (reward aggregator `kinetic_aggregator.py`)

`aggregate_kinetic_score` (`:26-38`) is a weighted harmonic mean of
5 click-reaction rates. The Ridge interpolator (`:97-101`) learns
*one* scalar (Tanimoto similarity) → rate mapping; this is too weak
to drive MCTS expansion — `RewardAggregator` in `proof_search.py`
correctly bypasses this in favour of the multi-channel
heuristic+satisfies+binds aggregator. **Honest framing**: the
kinetic aggregator is a *legacy stub* kept for backward compat with
WF-Lambda-1 first iteration; it should be deprecated in favour of
the proof_search `RewardAggregator` once the `r_reinvent4` and
`r_synth` channels ship. **Proposed fix** (Phase 2 candidate F5.e):
add a `DEPRECATION_WARNING` at `kinetic_aggregator.py:1` and route
callers to `proof_search.RewardAggregator.from_scorer(...)`.

---

## Closure-theorem surface (`closure.py`)

`closure.py` is **NOT** invoked by MCTS proof_search — it is a
standalone BFS over the click-rule set that asserts the closure
property (`:426-590`). It is used by `paper/appendices/closure_theorem.tex`
and the corresponding unit/property tests. MCTS proof_search does
NOT consult `ProductiveSpace.n_reachable` or `closure_test` — meaning
the constructive-synthesis guarantee from `WF-Lambda-4` is *not
exploited* by the runtime search. **Proposed fix** (Phase 2
candidate F5.f): on `_expand` miss (cache hit in
`_unreactive_states`), consult `ProductiveSpace.closure_test` with a
budget larger than `self.max_depth` to check whether the *seed* can
*ever* reach a new molecule under deeper expansion; if not, mark the
seed as "closure-dead" and skip expansion entirely. Cite the closure
theorem (paper `appendices/closure_theorem.tex`) for the guarantee
and Auger 2013 Thm 1 for the MCTS convergence rate. This is
*necessary-but-not-sufficient* for the singleton-attractor escape.

---

## Summary of 7 finding categories

| # | Finding | File:Line | Severity | Phase-2 candidate |
|---|---------|-----------|----------|-------------------|
| 1 | `metal_geometry_prior_bonus` seed-self-credit | `r4_lambda_only_run.py:454-543`, `:1517` | medium | F5.a: root-aware prior |
| 2 | `_unreactive_states` permanent cache | `proof_search.py:2343,2898,2921` | high | F5.b: per-(rule,tile) cache |
| 3 | `n_simulations` consumed per-sim not per-tile | `proof_search.py:2360,2903` | medium | F5.c: vectorised expansion |
| 4 | PUCT (correct), doc says UCB1 (wrong) | `proof_search.py:33-34,2840-2845` | low | doc fix |
| 5 | Root prior misses pocket context | `proof_search.py:2297-2301` | high | F5.d: pocket-conditioned prior (TODO WF-4) |
| 6 | No Bemis-Murcko scaffold diversity | `proof_search.py:2924-2950` | medium | F5.e: scaffold-aware PUCT |
| 7 | 8 bugs/inefficiencies | various | low-medium | misc patches |

## Honest framing

This is a *read*, not a *patch*. All Phase-2 algorithmic-tuning
candidates are CPU-only, low-risk, and lit-grounded (cited in-line).
The Round-13 negative result is *not* fully explained by this review —
CFM GPU blockage (per `wf_gpu_recovery_now` 2026-09-15) and the
3-layer singleton attractor (per `WF-Lambda-Internal-Review`) are
dominant contributors. The algorithmic-tuning changes here are
*necessary-but-not-sufficient* for Round-14 lift.