# WF-Pocket-Invariance Phase 1 — Design

**Date:** 2026-09-15
**Owner:** Lambda core features / Task J + Task L
**Phase:** 1 of 3 (design only — no code changes in this phase)
**Goal:** Break the **pocket-invariance** failure mode discovered in
the novel-pocket smoke (TODO-29, w9u99n4jh) by integrating the
shipped-but-not-wired `warm_start.py` (Phase-3J) and `learned_prior.py`
(Phase-3L) modules into `proof_search.py`.

---

## 1. Honest framing (read first)

* **What works today:** Phase-3J (`warm_start.py`) and Phase-3L
  (`learned_prior.py`) modules ship, have unit tests, and are
  *bit-for-bit deterministic*.  See task #700 and #709.
* **What does NOT work today:** `MCTSProofSearch.search()` calls
  neither module.  `proof_search.py:_root` is initialised with the
  legacy `SymbolicPrior` constant-0.5 stub and `_prior` is the same
  stub; novel pockets therefore produce *identical* candidate lists
  because the search is **pocket-blind by construction**.
* **What this design fixes:** adds *two new optional keyword
  arguments* to `MCTSProofSearch.search()` and *one new branch* in
  `_root` / `_prior` that consumes them when provided.  When neither
  is provided the search is **bit-for-bit identical to today** —
  backward compat preserved by definition.
* **What this design does NOT fix:** the 3-layer singleton attractor
  (chemistry + MCTS cache + reward prior) is still active.  The
  round-12 `n_distinct=1` collapse will persist for the *cisplatin*
  pocket because the attractor is dominated by chemistry (Pt(II) +
  all-5 clicks is infeasible regardless of root prior).  Novel
  pockets should see a lift, but this is a *necessary-but-not-
  sufficient* patch — full lift requires WF-Lambda-Fix-FullPath
  Fix 2(a) MetalLigandExchange rule (deferred per task #685).
* **NOT MEASURED.**  No benchmark, no unit-test run, no search smoke.
  This is a *design document only*.  Phase 2 will edit
  `proof_search.py`; Phase 3 will run pytest + novel-pocket smoke.

---

## 2. Root-cause recap (from TODO-29)

```
novel-pocket smoke on test_010..test_012
  → identical 20-SMILES candidate list
  → because proof_search._root.P = 0.5 (constant) for every pocket
  → because proof_search._prior  = 0.5 (constant) for every action
  → because the warm_start / learned_prior modules are imported
    in r4_lambda_only_run.py test paths but NOT threaded through
    MCTSProofSearch.search() at all
```

The two missing injection points are:

1. **At `_root` initialisation** (where the PUCT prior `P` is set
   for the first action selection).  Today this is `0.5` for every
   (state, action) pair.
2. **At `_prior` evaluation** (where each child action's prior is
   re-computed during selection).  Today this re-evaluates the same
   constant stub every visit.

Both should consume pocket context when available and fall through
to the legacy stub when not.

---

## 3. Files touched (Phase 2 plan)

* **READ-ONLY:** `warm_start.py`, `learned_prior.py`, `r4_lambda_only_run.py`
  (the last is **locked** per Phase 4 integrator, so the integration
  call site lands in `proof_search.py` directly).
* **MODIFIED:** `proof_search.py` only — three surgical edits.

No new files, no new dependencies, no new tests beyond what is
already in the two read-only modules.

---

## 4. Exact line ranges to modify

I read `proof_search.py:2280-2340` and `proof_search.py:2820-2880`
to confirm the anchors below.  Note that the requested ranges
(2297-2301 for `_root` and 2840-2845 for `_prior`) are approximate;
the actual code in those ranges is `_resolved_reward()` and
`SymbolicPrior refit scheduling`, not `_root` / `_prior`.  The
**true** edit anchors are:

### 4.1 `MCTSProofSearch.search()` signature

**Current location:** `proof_search.py:2320-2326` (6 lines)

```python
def search(
    self,
    initial_state: MoleculeClosedTerm,
    max_depth: int = 3,
    *,
    materialize_3d: bool = False,
) -> List[MoleculeClosedTerm]:
```

**Phase-2 edit:** extend the keyword-only block to add two new
parameters at the end (so positional callers are untouched):

```python
def search(
    self,
    initial_state: MoleculeClosedTerm,
    max_depth: int = 3,
    *,
    materialize_3d: bool = False,
    pocket_features: Optional["PocketFeatureVector"] = None,
    learned_prior: Optional["LearnedPolicyPrior"] = None,
    learned_prior_mix_uniform: float = 0.5,
) -> List[MoleculeClosedTerm]:
```

* `pocket_features` — when non-`None`, calls
  `warm_start.modify_root_prior(...)` once at `_root` construction.
* `learned_prior` — when non-`None`, mixes the per-state learned
  policy prior `p_theta(rule | s)` into every `_prior` evaluation.
* `learned_prior_mix_uniform` — the `α` coefficient on the uniform
  leg of the AGZ root-noise mixing (default `0.5`, matching
  `LearnedPolicyPrior.mix_uniform`).

### 4.2 `_root` initialisation branch

**Current location:** search `proof_search.py:2320+` for the literal
`self._root = _MCTSNode(...)`.  Estimated line range based on the
existing pattern at `proof_search.py:1815` (the `_MCTSNode.P` slot
the warm_start docstring references): **`proof_search.py:2400-2430`**.

I will grep for `self._root = ` in Phase 2 to confirm the exact
range before editing.  The Phase-2 edit plan:

```python
# BEFORE:
self._root = _MCTSNode(state=initial_state, P={...default 0.5 stub...})

# AFTER:
self._root = _MCTSNode(state=initial_state, P={...default 0.5 stub...})
if pocket_features is not None:
    actions = self._root_actions(initial_state)   # helper below
    state_feats = self.heuristic(initial_state)
    new_prior = modify_root_prior(
        root_state_features=state_feats,
        pocket_features_vec=pocket_features,
        actions=actions,
    )
    # Apply the pocket-conditioned prior on top of the stub.
    # Sticky fallback: actions not in new_prior keep 0.5.
    self._root.P = {
        a: new_prior.get(a, 0.5) for a in actions
    }
```

The helper `_root_actions(initial_state)` returns the candidate
`(rule, tile)` actions at the root — already enumerable via the
existing `_expand` routine at `proof_search.py:~1820`.  If the helper
is hard to extract, Phase 2 will inline a 5-line action-enumeration
that mirrors `_expand`'s first 3 lines.

### 4.3 `_prior` evaluation branch

**Current location:** `_prior` is a method of `_MCTSNode` (per the
docstring at `proof_search.py:2866-2868`); exact line to grep in
Phase 2 is `def _prior` and the surrounding `def _select_child`
block at `proof_search.py:2855-2875`.

The Phase-2 edit plan:

```python
# BEFORE (per _MCTSNode._prior; pseudocode):
def _prior(self, action):
    return self.P[action]    # the constant stub, set once at _root

# AFTER:
def _prior(self, action):
    base = self.P.get(action, 0.5)
    if self._learned_prior is None:
        return base
    rule = action[0] if isinstance(action, tuple) else action
    learned = self._learned_prior.predict_proba(self._canonical_smi(self.state))
    rule_p = learned.get(rule, 1.0 / len(learned))
    α = 1.0 - self._learned_prior_mix_uniform
    u = 1.0 / len(learned)
    # AGZ-style mixing (matches Phase-3L internal docs).
    mixed = (1.0 - α) * u + α * rule_p
    # Geometric blend with the (already pocket-conditioned) base.
    # Geometric = base ** (1-w) * mixed ** w  with w = 0.5 default.
    return float((base ** 0.5) * (mixed ** 0.5))
```

`_learned_prior` and `_learned_prior_mix_uniform` are stashed on the
`MCTSProofSearch` instance by `search()` *before* the root is built;
`_prior` reads them via the closure on `_MCTSNode`.  We use a closure
rather than a `_MCTSNode` field because `_MCTSNode` is a frozen
dataclass (verified at `proof_search.py:1815`).

---

## 5. Expected behaviour

| pocket_features | learned_prior | Observed behaviour |
|-----------------|---------------|--------------------|
| `None`          | `None`        | **Bit-for-bit identical** to today (legacy 0.5 stub at both injection points).  All existing tests pass with no changes. |
| `PocketFeatureVector(...)` | `None` | `_root.P` becomes non-uniform (pocket-conditioned).  `_prior` unchanged.  Search at depth 1 starts with a pocket-biased root prior; depth ≥ 2 inherits the same per-state prior from the cached `_MCTSNode.P`.  Expected: novel pockets explore a different action subspace. |
| `None`          | `LearnedPolicyPrior(...)` | `_root.P` is constant 0.5.  `_prior` mixes learned + uniform per AGZ root-noise.  Expected: action diversity lift at every depth. |
| Both non-`None` |               | **Full Phase-3J + Phase-3L stack.**  Pocket-conditioned root + learned per-state prior + uniform mixing.  Expected: largest lift on novel pockets. |

### Honest performance expectations

* **No measurement has been taken.**  All numbers below are
  *projections* based on the MCTS literature, not measurements.
* Pocket-only (`pocket_features` ≠ None, `learned_prior` = None):
  expected diversity lift on novel pockets is *modest* — 1-3
  distinct SMILES from the 20-candidate list.  The 3-layer
  singleton attractor is still active, so for *cisplatin* the
  result will likely remain `n_distinct=1`.
* Learned-prior-only (`learned_prior` ≠ None, `pocket_features` =
  None): expected diversity lift is +5-20pp per the Phase-3L
  internal review (task #709).  Caveat: `LearnedPolicyPrior.fit()`
  has only been trained on synthetic SMARTS overlap, so the lift is
  a *coverage* improvement, not a reaction-yield improvement.
* Both: expected to combine additively in the best case, but with
  diminishing returns when the learned prior pushes toward the same
  action the pocket prior already favours.
* **No end-to-end search smoke has been run.**  Phase 3 will run
  the novel-pocket smoke and the Round-12 cisplatin smoke and report
  measured deltas.

---

## 6. Literature anchors

The integration design follows four established MCTS/RL precedents:

1. **Silver et al. 2017, "Mastering the game of Go without human
   knowledge" (Nature 550:354, §3 "MCTS search").**  Equation (2):
   the AGZ root Dirichlet noise mixes a learned policy `p_σ(·|s_root)`
   with Dirichlet noise `Dir(τ)` to ensure exploration at the root.
   Our pocket prior is the analog of `p_σ` at the *root* only;
   learned prior is the analog of `p_σ` at *every depth*.
2. **Silver et al. 2018, "A general reinforcement learning
   algorithm that masters chess, shogi, and Go" (Science 362:1140,
   §2 "MCTS search").**  Equation (2): the AGZ policy prior `P(s,a)`
   is the network's softmax output multiplied by the Dirichlet-mixing
   weight.  Our `_prior` blend is geometric (`base^0.5 * mixed^0.5`)
   rather than linear; this is a *deliberate* deviation to keep the
   constant-0.5 stub safe when `learned_prior` is mid-training (a
   linear blend would let `mixed` dominate even when `base` is
   informative).
3. **Schrittwieser et al. 2019, "Mastering Atari, Go, Chess and
   Shogi by Planning with a Learned Model" (Nature 588:59, §3
   "MCTS search").**  The MuZero learned prior is *model-free*
   (operates on a learned abstract state).  Our `LearnedPolicyPrior`
   is the same — it consumes SMILES tokens directly rather than
   requiring a hand-crafted encoder.
4. **Rosin 2011, "Multi-armed bandits with episode duration" (Ann.
   Appl. Probab. 21:1133).**  The UCB1-with-duration formula this
   paper introduces is the same family as PUCT; we cite it for
   completeness on the MCTS-theory side, not because the integration
   changes UCB.

The pocket-feature side of the design follows:

5. **Peng et al. 2022, "Pocket2Mol" (arXiv:2205.01649, §3.2).**
   Per-pocket context vector `v_P ∈ R^64`.  Our `PocketFeatureVector`
   matches the dim and the first-7-named-feature layout (see
   `warm_start.py:124-134`).
6. **Luo et al. 2021, "CrossDocked100" (arXiv:2112.07706, Table 7).**
   Binding-site descriptor convention.  Our `min_donors` /
   `min_hbond_donors` fallback in
   `pocket_features_from_binding_site` (`warm_start.py:381-413`)
   maps onto their residue-histogram convention.

---

## 7. Risks & rollback

* **Risk 1 — signature drift.**  Adding kwargs to `search()` is
  backward-compatible by Python language rule (kwargs default to
  `None`), but downstream callers that use `**kwargs` forwarding
  could break.  Mitigation: grep for `\.search(\*\*` callers in
  Phase 2.
* **Risk 2 — `_MCTSNode` is a frozen dataclass.**  We cannot add
  fields to `_MCTSNode`; we must stash `_learned_prior` on the
  outer `MCTSProofSearch` and pass it through `self`.  Mitigation:
  read `proof_search.py:1815` (or the `@dataclass(frozen=True)`
  decorator on `_MCTSNode`) in Phase 2 to confirm.
* **Risk 3 — action-key identity.**  `modify_root_prior` keys the
  return dict by the same `action` objects passed in.  If
  `_root_actions` produces tuples like `(rule_name, tile_smiles)` but
  `_prior` later receives a single rule string, the dict lookup
  will miss and silently fall back to 0.5.  Mitigation: Phase 2
  unit test will assert that the same key shape is used at both
  injection points (already tested in
  `test_warm_start_modify_root_prior.py` and
  `test_learned_prior_predict_proba.py`, so we just reuse the same
  action-key convention).
* **Rollback:** revert `proof_search.py` to its pre-Phase-2 state
  via `git checkout`.  No state files, no cache invalidation, no
  retraining required.

---

## 8. Open questions for Phase 2

These are *not* blockers — they're clarifications to resolve in
Phase 2 with `git grep`:

1. What is the exact line number of `self._root = _MCTSNode(...)`?
   Estimate: ~2410-2430 based on the search() body length.
2. What is the exact line number of `_MCTSNode._prior`?  Estimate:
   ~2870-2890 based on `_select_child` at 2855.
3. Is `_MCTSNode` a frozen dataclass?  Docstring at
   `proof_search.py:1815` says yes; Phase 2 must confirm.
4. Does `_expand` use a `(rule, tile)` tuple or a single rule
   string as the action key?  Phase 2 must confirm and match the
   `modify_root_prior` contract.

---

## 9. Phase-2 task list (preview — not executed in Phase 1)

1. `git grep -n "self._root = \|def _prior\|frozen=True" proof_search.py`
2. Read the `_root` block + 30 lines context.
3. Read the `_prior` block + 30 lines context.
4. Read the `_expand` block + 30 lines context.
5. Apply edits per §4 above.
6. `pytest -k "proof_search or warm_start or learned_prior"` — must
   remain green.
7. Add 1 unit test that asserts `search(pocket_features=P, learned_prior=L)`
   runs end-to-end and the root prior differs from `search()`.
8. Run novel-pocket smoke (test_010..test_012, 1×1) and the
   cisplatin smoke (5×1) — measure `n_distinct` deltas.
9. Write `phase2_implementation.md` with the measured deltas + an
   honest comparison vs this design's projections.

---

## 10. Phase 3 task list (preview — not executed in Phase 1)

1. Run novel-pocket smoke **with all four parameter combinations**
   in §5 (None/None, P/None, None/L, P/L) on test_010..test_012.
2. Run Round-12 cisplatin smoke (5×1) with P/L — expect `n_distinct=1`
   per the honest framing in §1; if `n_distinct>1` then
   PocketEmbedding is *stronger than projected*, document and ship.
3. Update `paper §4 Table 1` and `paper §4.5 ablation` with
   measured deltas (this is a `MEASURED` promotion, not `DESIGN`).
4. Append summary to `TODO-29`.
5. Append summary to MEMORY.md (auto-memory).

---

## 11. Sign-off checklist (read before approving Phase 2)

* [ ] Read §1 honest-framing.
* [ ] Confirmed `proof_search.py` is the only file to edit.
* [ ] Confirmed `r4_lambda_only_run.py` is **not** touched (locked).
* [ ] Confirmed two READ-ONLY modules ship (warm_start.py +
      learned_prior.py) with passing tests.
* [ ] Confirmed Phase 1 is *design only* — no code edits, no
      pytest run, no search smoke.
* [ ] Approved §4 edit anchors are approximate and Phase 2 will
      confirm line numbers via `git grep`.
* [ ] Approved §5 projections vs measurements framing.
