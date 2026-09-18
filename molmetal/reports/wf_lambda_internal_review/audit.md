# WF-Lambda-Internal-Review — Lambda MCTS Deep Audit (Round-12 Failure Modes)

> **Honest-framing**: this audit reads the *actual* code paths in
> `molmetal/molmetal_lam/search_alg/proof_search.py` (4095 lines,
> `MCTSProofSearch` dataclass) and
> `molmetal/scripts/r4_lambda_only_run.py` (2122 lines, the harness
> that drives `MCTSProofSearch` on CrossDocked pockets with
> `--metal-seed`). The 4 failure points surfaced by Round-12 Lambda
> Pilot (final.md §5, 2026-09-14) are mapped to concrete code lines
> and ranked by root-cause quality.

---

## 0. Scope and inputs

| input | path / lines | purpose |
|---|---|---|
| MCTS core | `molmetal/molmetal_lam/search_alg/proof_search.py:1865-4081` | `MCTSProofSearch` dataclass |
| MCTS harness | `molmetal/scripts/r4_lambda_only_run.py:1-2122` | `r4_lambda_only_run.py` |
| Click reaction rules | `molmetal/molmetal_lam/reactions/beta_reductions.py:545-1100` | `CuAAC` / `SPAAC` / `ThiolEne` / `Suzuki` / `AmideCoupling` |
| Metal pilot artefact | `molmetal/reports/wf_lambda_metal_pilot/final.md` | round-12 evidence |
| Lambda-only mini pilot | `molmetal/reports/wf_lambda_only_mini_pilot/final.md` | reference (no metal seed) |
| Diversity-rotation rejection | `molmetal/reports/wf_lambda_div_rotation/final.md` | n_simulations=1000 follow-up |

---

## 1. Line-by-line map of `MCTSProofSearch`

The search has 6 internal phases. Each phase is annotated with the
exact file lines and its round-12 contribution.

### 1.1 `__post_init__` / config resolution (proof_search.py:2118-2145)

```
_resolved_reward()  → RewardAggregator.from_scorer or zero aggregator
reward_channels{}   → setattr overrides
agg._leaf_oracle_call_top_k_only = bool(self.leaf_oracle_call_top_k_only)
agg._oracle_top_k = int(self.oracle_top_k)
agg._owner_search = self
```

The aggregator carries the leaf-oracle preference so the search loop
can dispatch to the real docking oracle on the top-K leaves at
iteration end without dragging the flag through every call.
**No metal-specific hook exists here.** The metal_seed is *not*
represented in the aggregator; it lives only on the *root state*
in the harness.

### 1.2 `search(initial_state, max_depth)` (proof_search.py:2151-2462)

Public entry-point. The full pipeline:

```
1. Initialise root = _MCTSNode(state=initial_state, P=self._prior(initial_state))
2. Initialise leaves_by_smi, _leaf_registry_nodes, history
3. Reset nfe / nfe_reductions / nfe_oracle
4. Reset early-stop bookkeeping (_best_score_so_far, _best_score_iter)
5. Reset _leaf_value_history
6. Reset per-search D2 transient (_selection_signature, _sim_counter,
   _t_hits, _t_misses, _t_table, _unreactive_states)
7. Pre-compute root_children_cache for Dirichlet injection
8. for it in range(n_simulations):
     - apply Dirichlet noise to root (line 2255-2265)
     - path = self._simulate(root, max_depth, root_children_cache)
     - leaves_by_smi = self._collect_leaves(root, leaves_by_smi)
     - compute per-iter scores + history entry (line 2270-2351)
     - early-stop check (line 2361-2382)
9. Filter candidates: _satisfies_predicates AND _binds_target
   (line 2388-2397)
10. Top-K leaf oracle call (line 2405-2433)
11. tree_diversity = len(leaves_by_smi) / max(1, _count_nodes(root))
    (line 2437)
12. leaf_value_var / puct_exploit_ratio_var (line 2442-2452)
13. self._root = root  (L-2 persistent-tree state)
14. self._accumulate_leaf_pairs(leaves_by_smi)
15. return candidates[:top_k]
```

**Round-12 critical lines**: 2388-2397. The candidate filter is
the *only* place where the search decides "which leaves survive
into `top_k`". Since the filter does not gate on
`metal_compliance`, `diversity_tanimoto`, or `n_distinct`, none of
those metrics are visible to the search — they are reported *after*
the fact in the harness.

### 1.3 `_simulate(root, max_depth, root_children_cache)` (proof_search.py:2468-2569)

The 4 phases per simulation:

```
1) SELECT   — PUCT traversal while node is non-leaf + non-terminal + len(path) ≤ max_depth
              node = self._select_child(node)              # proof_search.py:2571-2668
              Lock: node.virtual_loss.value += VIRTUAL_LOSS_VALUE  # line 2499
                    self.virtual_loss.apply(id(node))              # line 2509
2) EXPAND   — if not node.is_terminal and len(path) ≤ max_depth:
              children = (root_children_cache if root + cache else _expand(node.state))
              self._attach_children(node, children)        # proof_search.py:2729-2755
              if node.children: node = self.rng.choice(node.children); path.append(node)
3) ROLLOUT  — depth_remaining = max(0, max_depth - (len(path)-1))
              value = self._rollout(node.state, depth_remaining, reward_fn)
                                                     # proof_search.py:3026-3137
4) BACKPROP — self._backprop(path, value)               # proof_search.py:3179-3206
```

`is_terminal` is defined at line 1676-1683 as
`expansion_complete AND not children` — a node is terminal *only*
after `_expand` exhaustively proves no (rule, tile) fires.

**Round-12 critical line**: 2519-2524. The root's children cache
(`root_children_cache`) is computed *once* during the first
iteration (line 2260) and reused for all subsequent simulations.
This is the Dirichlet-injection optimization (AlphaZero
convention). **It also means the root's children set is frozen
after the first expansion** — if a later simulation wants to
expand the root a second time, it sees the same children. This is
correct MCTS but it interacts with `_attach_children`'s
"ancestor + sibling dedup" rule (proof_search.py:2729-2755) which
*rejects* children whose canonical SMILES already appears in the
ancestor chain or among siblings. For a metal seed with no
plausible (rule, tile) reducible product, the root's children set
collapses to empty → `expansion_complete=True` → `is_terminal=True`
on line 1683.

### 1.4 `_select_child(node)` (proof_search.py:2571-2668)

Canonical PUCT + virtual-loss:

```
u(child) = c_puct * P(child) * sqrt(N_parent) / (1 + N_child)
         + Q(child) / (1 + N_child)
         - virtual_loss(child) / (1 + N_child)
```

For each child the formula reads:

- `P(child)` — `SymbolicPrior.predict_proba(state)` if fitted,
  else `self._prior(state)` (proof_search.py:3421) which dispatches
  to `self.heuristic(state)` (proof_search.py:3292). For the
  Lambda-only harness the per-channel aggregator returns 0 for
  every channel except `r_vina_proxy`, so the
  RewardAggregator-based heuristic computes
  `1.0 + 0 + 0 + 0 + 0 + 0 = 1.0` *when the lambda-native score
  returns nonzero* (see `build_lambda_only_aggregator`,
  r4_lambda_only_run.py:1295-1300). The sigmoid squashes 1.0 →
  ≈0.731. **The prior is biased toward high-λ-native-score
  children, i.e. toward children that satisfy
  `metal_geometry_prior_bonus`.**

- `Q(child) = W/N` — mean of backpropagated leaf values.
  Backpropagated value at line 3179 is *one* float for the whole
  simulation (whatever `_rollout` returned). When the rollout
  collapses onto the seed (no firing redex), `_rollout` returns
  the seed's lambda-native score → all children get identical
  Q → PUCT becomes **pure-prior-driven**.

- `virtual_loss` — 0.0 in single-threaded search.

**Round-12 critical implication**: when all Q values are equal
(rollback to seed), PUCT is *purely* prior-driven, and the prior
favors children with high `metal_compliance_rate`. This biases
the search *back to the seed* rather than *away* from it.

### 1.5 `_expand(state)` (proof_search.py:2670-2727)

```
for rule_name, rule in self.rules.items():
    for tile in tile_pool:                  # 204-tile pool
        products = self._safe_reduce(rule, state, tile)
        if not products: continue
        for prod in products:
            children.append((prod, rule_name, tile))
if not children: self._unreactive_states.add(state_key)
```

This is where the **singleton attractor** lives. The cycle is:

```
root = cisplatin seed (canonical SMILES: [NH3][Pt]([NH3])(Cl)Cl)
   ↓
for each of 5 click rules × 204 tiles = 1020 (rule, tile) pairs
   ↓
_safe_reduce(rule, root, tile):
   - rule.pattern_smiles encodes e.g. "[N:1]=[N:2]=[N:3].[C:4]#[CH:5]>>[C:4]1=..."
   - RDKit RunReactants(azide.to_rdkit(), alkyne.to_rdkit())
   - Returns [] when no redex substructure is found
   ↓
For Pt_II seed [NH3][Pt]([NH3])(Cl)Cl:
   - No N=N=N substructure → CuAAC returns []
   - No N=N=N → SPAAC returns []
   - No [SH] + C=C → ThiolEne returns []
   - No boronic acid → Suzuki returns []
   - No COOH + NH2 → AmideCoupling returns []
```

When every (rule, tile) fires zero products, `children == []` and
`_unreactive_states.add(state_key)` (line 2726) **permanently
caches the seed as unreactive**. From the next simulation
onwards, `len(children)==0`, `is_terminal=True`, and the MCTS
loop *short-circuits* (proof_search.py:2516-2527:

```
if not node.is_terminal and len(path) <= max_depth:
    children = ...
    self._attach_children(node, children)
    if node.children:
        node = self.rng.choice(node.children)
```

When `node.is_terminal` is True, the EXPAND step is skipped, and
the rollout starts from this node. The rollout itself (line
3026-3137) only samples (rule, tile) pairs from the *union* of
`self.rules` and `tile_pool` — it does **not** know about
`_unreactive_states`. The `_rollout` does consult
`_unreactive_states` *before* sampling (line 3054):

```
if self._canonical_smi(current) in getattr(self, "_unreactive_states", set()):
    break
```

But this only fires when the *current* (rollout) state is in
`_unreactive_states`. After a single expansion of cisplatin at
root, `_unreactive_states` contains the cisplatin canonical SMILES
— and any rollout that re-enters cisplatin (which is *all* of
them, since no firing reaction exists) breaks immediately.

**This is the singleton attractor mechanism.** The MCTS has no
escape valve: every cycle either re-enters cisplatin (and
short-circuits) or attempts the same 1020 (rule, tile) pairs
that already failed.

### 1.6 `_attach_children(parent, children)` (proof_search.py:2729-2755)

```
ancestors = {ancestor's canonical_smi}
sibling_keys = {existing child canonical SMILES}
for child_state, rule_name, tile in children:
    child_key = self._canonical_smi(child_state)
    if child_key in ancestors or child_key in sibling_keys:
        continue                                # dedup
    existing, hit = self._lookup_or_create(child_key, factory)
    child = factory() if hit and existing.parent is not parent else existing
    parent.children.append(child)
    sibling_keys.add(child_key)
parent.expansion_complete = True
```

This is where diamond-paths and α-equivalent molecules are
de-duplicated. For a metal seed like cisplatin that already
*is* the root, no (rule, tile) ever produces a child whose
canonical SMILES differs from cisplatin. The result is:

```
parent.children == []
parent.expansion_complete == True
```

and `is_terminal == True` for the root node.

### 1.7 `_rollout(state, depth, reward_fn)` (proof_search.py:3026-3137)

```
while cur_depth < depth:
    if _canonical_smi(current) in _unreactive_states: break
    rule_name = self.rng.choice(list(self.rules.keys()))
    tile = self.rng.choice(tile_lib)
    products = self._safe_reduce(rule, current, tile)
    if not products:
        cur_depth += 1; continue
    current = self.rng.choice(products)
    cur_depth += 1

value = reward_fn(current, ..., satisfies_typed=sat, binds_target=binds)
```

When the root is cisplatin and `_unreactive_states` already
contains the cisplatin canonical SMILES, the loop breaks
immediately and the rollout *always* evaluates cisplatin itself.
The leaf reward is then `lambda_native_score(cisplatin)`:

```
aeq = 1.0 if cisplatin.canonical_smiles() else 0.0  → 1.0
click = click_rule_match_bonus(cisplatin, rules)    → 0.0  (no azide/alkyne/thiol/...)
metal = metal_geometry_prior_bonus(cisplatin, enabled=True)
       = 1.0 if (has_metal and matches_target)      → 1.0
valid = rdkit_validity_score(cisplatin_smi)         → 1.0
syn = synthesizability_via_lambda_paths(cisplatin, ...)
       = 1.0 (cisplatin is a single atom, BNF trivially satisfied)
       → return 1.0 + 0.0 + 1.0 + 1.0 + 1.0 = 4.0
```

The leaf value is a *constant* `4.0` across all rollouts.

### 1.8 `_backprop(path, value)` (proof_search.py:3179-3206)

```
for node in path:
    node.N += 1
    node.W += value
    node.virtual_loss.value -= VIRTUAL_LOSS_VALUE
    self.virtual_loss.release(id(node))
```

Identical `value` (= 4.0) at every node means `Q = W / N` is
constant, PUCT is purely prior-driven, and the prior selects the
seed as the dominant root.

### 1.9 top-K selection (proof_search.py:2388-2397)

```
candidates = []
for node in leaves_by_smi.values():
    state = node.state
    if not self._satisfies_predicates(state, self.target_predicates): continue
    if not self._binds_target(state): continue
    candidates.append((self.score_final(state), state))
candidates.sort(key=lambda p: p[0], reverse=True)
```

`target_predicates` is **empty** in the lambda-only harness
(r4_lambda_only_run.py:1492: `target_predicates = []`).
`_satisfies_predicates(state, [])` returns `True` for every state
(line 3223-3231: the `for pred in predicates: ...` loop is a
no-op). `_binds_target(state)` calls `typecheck(state,
binding_site, leaf_oracle_call=False, oracle=None)` (line 3233-3251).
With a no-op `BindingSite(name=f"lambda_only::{pocket_id}")` and
no docking oracle, `typecheck` falls through to the cheap
fingerprint stub (binding/types.py).

**Every cisplatin leaf passes both gates.** `score_final(state)`
returns the same `4.0` for all leaves → `candidates.sort(...)`
keeps the canonical-SMILES ordering of `leaves_by_smi` →
`top_k` returns the first 20 (or fewer) leaves → since
`leaves_by_smi` is a dict keyed by canonical SMILES and the
seed is the only key, **the top_k slice has length 1**.

### 1.10 `search()` return (proof_search.py:2462)

```
return [state for _, state in candidates[: self.top_k]]
```

`self.top_k = 20` (proof_search.py:1968). With 1 candidate, this
returns a list of length 1. The harness then computes
`n_distinct = len(set(smis)) = 1`
(r4_lambda_only_run.py:1535: `cell.n_distinct = len(set(smis))`).

---

## 2. metal-seed handling in `r4_lambda_only_run.py`

### 2.1 Metal-seed SMILES registry (r4_lambda_only_run.py:190-203)

```
METAL_SEED_SMILES: Dict[str, str] = {
    "cisplatin":  "[NH3][Pt]([NH3])(Cl)Cl",
    "ru_arene":   "[Ru](c1ccccc1)(c1ccccc1)(Cl)(Cl)(N)N",
    "ir_cp_star": "[Ir](C1C(C)=C(C)C(C)=C1C)(Cl)(N)N",
}
```

Three seeds, each carrying a fully-coordinated metal centre:
- cisplatin: Pt + 2 NH3 + 2 Cl → coordination 4 (square-planar)
- ru_arene: Ru + 2 Ph + 2 Cl + 2 NH3 → coordination 6 (octahedral)
- ir_cp_star: Ir + Cp* + Cl + 2 NH3 → coordination 4-5

### 2.2 Resolution order (r4_lambda_only_run.py:1430-1463)

```
if metal_seed is not None:
    seed_smi = METAL_SEED_SMILES.get(metal_seed)
    ...
    root = MoleculeClosedTerm.from_smiles(seed_smi, embed_3d=False)
if root is None and reference_smiles:
    root = MoleculeClosedTerm.from_smiles(reference_smiles, embed_3d=False)
if root is None:
    root = MoleculeClosedTerm.from_smiles("Cl[Pt]Cl", embed_3d=False)
```

The metal seed is **always the root**, never a sibling. The
search never sees the metal seed as one of N alternatives — it
*starts* there.

### 2.3 `build_lambda_only_aggregator` (r4_lambda_only_run.py:1240-1300)

```
r_vina_proxy = _lambda_native_score:
    aeq   = 1.0 if smi else 0.0
    click = click_rule_match_bonus(state, rules)
    metal = metal_geometry_prior_bonus(state, enabled=True)    ← always-on
    valid = rdkit_validity_score(smi)
    syn   = synthesizability_via_lambda_paths(state, ...)
    return aeq + click + metal + valid + syn         # ∈ [0, 5]
```

`metal_geometry_prior_bonus` (r4_lambda_only_run.py:330-384)
returns 1.0 iff the molecule has a metal centre AND the
coordination number matches the Pt/Ru/Ir target. **For cisplatin,
this fires at every leaf** (coordination 4).

### 2.4 The "ROOT-only" vs "ROOT+HARD" distinction

The metal-seed currently does **two** things:

1. Sets the **root** of the search (initial_state in
   proof_search.py:2187).
2. Adds a **hard prior** in the reward aggregator (the `metal`
   channel returns 1.0 iff `coord_number == target`).

These are conceptually different:

- (1) is correct: the metal seed should be the *starting* typed-
  variable, like any root.
- (2) is **the singleton attractor**: every leaf must carry a
  Pt/Ru/Ir centre to score nonzero, so the search collapses to
  cisplatin itself.

The expected MCTS exploration behaviour is **(1) only** — the
metal seed is the *starting* root, not a *forced* attractor. The
prior should bias toward, not force, metal-containing children.

---

## 3. The 4 failure points — root causes

### 3.1 Failure (a): n_distinct=1 when metal-seed is set

**Root cause** — proof_search.py:2720-2727 (`_expand`):

```
for rule_name, rule in self.rules.items():
    for tile in tile_pool:
        products = self._safe_reduce(rule, state, tile)
        if not products: continue
        ...
if not children: self._unreactive_states.add(state_key)
```

The cisplatin seed carries *no* click-reaction warhead
substructure (no `N=N=N`, no `C#CH`, no `[SH]`, no boronic acid,
no `COOH`). Every (rule, tile) pair returns `[]` because
`RunReactants` on `[NH3][Pt]([NH3])(Cl)Cl` + a 204-tile pool
yields zero valid redex matches. The fallback path:
- `_unreactive_states` permanently caches cisplatin as unreactive.
- `_expand` short-circuits to `[]` on every subsequent simulation.
- `_attach_children` sets `parent.expansion_complete=True` with
  zero children → `is_terminal=True`.

**Result**: `n_distinct = 1` (cisplatin is the only leaf).

**Quality of root cause**: **high**. The mechanism is unambiguous:
the click SMARTS patterns do not match the cisplatin substructure,
the cache prevents re-attempts, and `_attach_children` finalises
the empty children set.

### 3.2 Failure (b): 5-click NOT reducible against Pt_II motif

**Root cause** — beta_reductions.py:545-1100 (click SMARTS patterns):

- CuAAC: `[N:1]=[N:2]=[N:3].[C:4]#[CH:5]>>[C:4]1=[C:5][N:3]=[N:2][N:1]1`
- SPAAC: `[N:1]=[N:2]=[N:3].[C:4]#[C:5]>>[C:4]1=[C:5][N:3]=[N:2][N:1]1`
- ThiolEne: pattern is structural (callable predicate), checks
  for `[SH]` + alkene.
- Suzuki: requires boronic acid `B(O)` + aryl halide.
- AmideCoupling: requires `COOH` + `NH2`.

None of these patterns recognise the Pt_II motif
`[NH3][Pt]([NH3])(Cl)Cl` as a *reactant*. The 5 reactions are
*organic* click reactions — they require C/H/N/O/S warheads. The
Pt_II complex is a *metal complex*, not an organic substrate.

The chemoinformatics reality:

- The 204-tile fragment pool (FRAGMENT_LIBRARY_200_TILES) is
  ChEMBL/ZINC-style organic fragment tiles. None of them carry a
  Pt_II / Ru / Ir centre.
- The 5 click reactions do *not* know about transmetalation,
  ligand substitution, or oxidative addition — i.e. the
  *inorganic* chemistry that would actually expand a metal
  complex.
- No rule exists for "Pt–Cl + NH3 → Pt–NH3 + HCl" (a basic
  aquation / ammination step). Such a rule would be the natural
  way to grow a Pt_II complex — but it is not in the registry.

**Quality of root cause**: **high**. The failure is structural:
the click reaction set is incomplete for metal chemistry. There
is no *cheminformatics* error — the system is faithfully reporting
that the 5-click rules do not reduce metal complexes.

### 3.3 Failure (c): diversity_tanimoto 0.005 → 0.000 with metal-seed

**Root cause** — interaction of 1) the singleton attractor
(3.1) and 2) the diversity metric definition:

`diversity_tanimoto_mean` in
r4_lambda_only_run.py is computed over the **candidates list**
returned by the search. With n_distinct=1 (only cisplatin in the
candidate set), the mean pairwise Tanimoto between identical
SMILES is 1.0, and the *diversity* metric
(typically `1 - mean_tanimoto`) is 0.0.

In the no-metal-seed baseline
(wf_lambda_only_mini_pilot/final.md, 2026-09-14):
- 5/5 cells returned valid candidates with at least 2 distinct
  SMILES per cell
- diversity_tanimoto ≈ 0.005 (low but nonzero)

In the metal-seed arm (wf_lambda_metal_pilot/final.md,
2026-09-14):
- 5/5 cells returned exactly 1 candidate (cisplatin) per cell
- diversity_tanimoto = 0.000 (the only SMILES pair is identical)

**Quality of root cause**: **high**. The mechanism is a direct
*consequence* of 3.1 — when `n_distinct=1`, diversity is
trivially 0. Fixing 3.1 fixes 3.3.

### 3.4 Failure (d): metal_compliance=1.0 is trivially satisfied by the seed itself

**Root cause** — r4_lambda_only_run.py:1556-1563:

```
# 7) metal_compliance_rate
cell.metal_compliance_rate = sum(
    1 for c in candidates
    if any(a.symbol in {"Pt", "Ru", "Ir"} for a in parse_atoms(c))
) / max(1, cell.n_candidates)
```

The metric counts *candidates* with a metal centre. With the
metal seed forced into the root, every leaf IS the metal seed →
100% of candidates carry the metal centre → metal_compliance=1.0
is *trivially true* because the *only* candidate IS the seed.

This is a **metric definition error**, not a measurement error.
The metric should count *non-seed candidates* with a metal
centre (i.e. test whether MCTS *generated* new metal-bearing
molecules, not whether the seed itself has a metal).

**Quality of root cause**: **high**. The bug is purely in the
metric definition: `metal_compliance_rate` confuses "the search
produced a metal-bearing molecule" with "the seed has a metal".
A fix is one-line:

```python
seed_smi = cell.reference_smiles or ""  # or seed
non_seed = [c for c in candidates if c != seed_smi]
metal_count = sum(1 for c in non_seed if has_metal(c))
cell.metal_compliance_rate = metal_count / max(1, len(non_seed))
```

---

## 4. Where the chemistry path forces the singleton attractor

The attractor is built across **three** layers:

| layer | file:line | mechanism |
|---|---|---|
| chemistry | beta_reductions.py:545-1100 | click SMARTS ignore Pt_II / Ru / Ir motifs |
| MCTS cache | proof_search.py:2706-2707 | `_unreactive_states.add(state_key)` is permanent |
| reward prior | r4_lambda_only_run.py:1269 | `metal_geometry_prior_bonus` always-on forces 1.0 only for already-metal-bearing molecules |

Each layer reinforces the others:

1. **chemistry**: cisplatin is a *metal complex*, not an
   organic substrate → 1020 (rule, tile) pairs all return [].
2. **MCTS cache**: `_unreactive_states` permanently blocks
   re-attempts → no future simulation can re-expand cisplatin.
3. **reward prior**: `metal_geometry_prior_bonus` returns 1.0
   only for molecules with a *correctly-coordinated* metal centre,
   so the only way to score nonzero is to be cisplatin or
   cisplatin-like.

The three layers together make the MCTS a *deterministic*
singleton attractor with no escape hatch.

---

## 5. Expected MCTS exploration behaviour

The AlphaZero convention is that the **root** is the only place
where the prior is "pinned" — the prior biases *exploration* but
does not *force* selection. In the current implementation:

- Dirichlet noise is mixed into the root priors (proof_search.py:2866-2935,
  `dirichlet_alpha=0.3, dirichlet_fraction=0.25`). This is the
  standard mechanism for root-level exploration.
- However, with **zero children** at the root, Dirichlet noise
  has nothing to perturb. The PUCT formula degenerates to the
  root being its own parent (a degenerate cycle), and every
  simulation collapses to the seed leaf.

The correct behaviour is: the metal seed should be the **root
only**, and the search should explore from there. The metal
seed's coordination geometry (Pt_II square-planar) should be a
*soft* prior, not a *hard* attractor.

---

## 6. Recommendations

### 6.1 Fix the singleton attractor (3.1)

**Option A — short-term** (5-line patch in
r4_lambda_only_run.py:1448-1451):

When `metal_seed is not None`, append a *click-handle fragment*
(e.g. an azide-bearing arm `N(CC[NH3][Pt]([NH3])(Cl)Cl)N=[N+]=[N-]`)
to the root before passing it to MCTS. This gives the MCTS a
valid redex on the first simulation and breaks the singleton.

**Option B — medium-term** (new rule in beta_reductions.py):

Add a `MetalLigandExchange` rule:
- SMARTS: `[Pt:1]([*:2])([*:3])([*:4])([*:5]).[N:6]>>[Pt:1]([N:6])([*:3])([*:4])([*:5]).[*:2]`
  (substitute one monodentate ligand on Pt_II with a primary
  amine from the tile pool).
- This gives the MCTS a *real* reaction path to grow a Pt_II
  complex from cisplatin + an aminotile.

### 6.2 Fix metal_compliance metric (3.4)

In r4_lambda_only_run.py:1556-1563, change to:

```python
seed_smi = ... # canonical seed SMILES
non_seed_candidates = [c for c in candidates if c != seed_smi]
cell.metal_compliance_rate = sum(
    1 for c in non_seed_candidates
    if any(a.symbol in {"Pt", "Ru", "Ir"} for a in parse_atoms(c))
) / max(1, len(non_seed_candidates))
```

This makes `metal_compliance_rate` measure *whether MCTS
generated new metal-bearing molecules*, not whether the seed has
a metal.

### 6.3 Soften the metal prior (1 + 2)

Make `metal_geometry_prior_bonus` return a *partial* credit for
near-misses:

- coord == target: 1.0
- coord == target ± 1: 0.5 (penalty for off-by-one)
- no metal: 0.2 (small bonus for "potentially add a metal")
- coord > target + 2: 0.0 (too far off)

This keeps the bias toward metal chemistry without forcing
collapse.

### 6.4 Lift the `_unreactive_states` cache

In proof_search.py:2706-2707, change the cache from
*permanent* to *per-simulation*:

```python
# Was: self._unreactive_states.add(state_key)
# Now: only cache within a single simulation
state_unreactive_this_sim = set()
```

This lets MCTS re-attempt previously-unreactive states across
different simulations, which is the AlphaZero convention for
state-space exploration.

### 6.5 Honest framing in §5.7 ablation

Until the fixes ship, the 5-click ablation in paper §5.7 should
*explicitly* note that the all-5 column is degenerate due to the
singleton attractor. The honest framing is already in
wf_lambda_metal_pilot/final.md §5 — propagate it to the paper.

---

## 7. Honest caveats

- This audit reads the *actual* code paths but does **not**
  execute the MCTS to verify the predicted behaviour. The
  singleton-attractor theory is consistent with the evidence
  (final.md §3 + §5 of wf_lambda_metal_pilot) but a fresh
  smoke test post-fix is needed to confirm.
- The `_TranspositionTable` (proof_search.py:1552-1605) and
  `_VirtualLoss` (1499-1547) are round-3 upgrades that *do not*
  affect the singleton attractor — they operate on already-
  expanded nodes.
- The "metal_seed + Pt_II not reducible" finding (3.2) is
  **structural** to the current click reaction registry and
  is *not* a code bug. The fix is to extend the registry, not
  patch the MCTS.

---

## 8. Cross-references

- Round-12 evidence: `molmetal/reports/wf_lambda_metal_pilot/final.md`
- Round-12 mini pilot (no metal-seed): `molmetal/reports/wf_lambda_only_mini_pilot/final.md`
- Diversity-rotation rejection: `molmetal/reports/wf_lambda_div_rotation/final.md`
- n_simulations hard-cap lift: `molmetal/reports/wf_lift_n_sim_cap/`
- BNF predicate fix: `WF-Lambda-1c` (TODO-pending row)
- Closure-theorem proof: `molmetal/molmetal_lam/lam_chem/closure.py`
- Paper §5.7 ablation: `paper/sections/05_ablation.tex` (will need honest framing update after fix)

---

**Audit quality**:
- 4 failure points identified: YES
- root cause quality: HIGH for (a) (b) (d); HIGH for (c) via
  reduction to (a)
- n_lines_read: 6217 (proof_search.py 4095 + r4_lambda_only_run.py 2122)
- recommendations: 5 (one for each failure point + one for the
  metric definition + one for honest framing)
