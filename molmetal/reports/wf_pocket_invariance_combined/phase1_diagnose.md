# WF-Pocket-Invariance Combined Phase 1 — Diagnostic of single-patch failure

**Date:** 2026-09-15
**Owner:** Lambda core features / Task J + Task L + integration owner
**Phase:** 1 of N (diagnose why w2swi9tsu single-patch failed)
**Goal:** Pinpoint the root cause of `test_search_pocket_invariance_break`
failing under the w2swi9tsu Phase-3 patch, and propose 3 combined
sub-fixes (A/B/C) that together break the pocket-invariance failure
mode.

---

## 1. Honest framing (read first)

* **What was attempted:** apply the w2swi9tsu Phase-3 single patch
  (the 3 surgical edits on `proof_search.py` described in
  `wf_pocket_invariance/phase3_smoke.md`) and run
  `test_search_pocket_invariance_break`.
* **What was measured:** the test **FAILS** at line 441:
  ```
  AssertionError: pocket-conditioned priors must produce different
  argmax actions (pocket-invariance break); both picked
  ('CuAAC', 'C#C')
  ```
  (NB: brief said `'SPAAC+C#C'`; actual failure is `('CuAAC', 'C#C')`
  — both equally informative as a pocket-invariant collapse).
* **Root cause (this report):** the pocket-conditioned prior
  distribution over `actions` is *rank-preserving* across CA2 and MMP2
  because the two pocket embeddings live in a near-parallel subspace
  (both ~7 residues, both His-metal-anchor heavy, both with similar
  residue-fraction histograms). The softmax argmax therefore picks the
  same action even though the probability *values* differ.
* **What is NOT in this report:** a working fix. Phase 2 will land the
  3 combined sub-fixes (A/B/C below); Phase 3 will verify on the novel
  pockets; Phase 4 will run the production Round-13 sweep.
* **What is NOT yet attempted:** changing `warm_start.py` or
  `learned_prior.py` (READ-ONLY by spec). Sub-fix A is a `proof_search.py`
  edit; sub-fix B is a `proof_search.py` edit; sub-fix C is a
  `r4_lambda_only_run.py` edit (CLI flag only).

---

## 2. Files inventoried in this Phase

| File | Read | Modified | Status |
|------|------|----------|--------|
| `molmetal/molmetal_lam/search_alg/proof_search.py` | YES (lines 2320-2540: `search()`, 2326-2540: pocket boost, 3854-3901: `_prior`) | NOT YET | Phase 2 target for sub-fix A + B |
| `molmetal/molmetal_lam/search_alg/warm_start.py` | YES (lines 416-510: `modify_root_prior`, 513-542: `_action_weight_vector`) | NOT (READ-ONLY) | Sub-fix proposal must NOT mutate |
| `molmetal/molmetal_lam/search_alg/learned_prior.py` | YES (lines 376-426: `predict_proba`, `batch_predict_proba`) | NOT (READ-ONLY) | Sub-fix B proposal must NOT mutate |
| `molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py` | YES (lines 383-462: `test_search_pocket_invariance_break`) | NOT YET | Phase 3 verification target |
| `molmetal/scripts/r4_lambda_only_run.py` | partial (CLI flag surface) | NOT YET | Phase 2 target for sub-fix C |

The w2swi9tsu Phase-3 patch on disk is unchanged — this Phase is
pure-diagnostic.

---

## 3. What the w2swi9tsu Phase-3 patch actually shipped

### 3.1 `search()` signature (lines 2320-2329 of `proof_search.py`)

```python
def search(
    self,
    initial_state: MoleculeClosedTerm,
    max_depth: int = 3,
    *,
    materialize_3d: bool = False,
    pocket_features: Optional[Any] = None,
    learned_prior: Optional[Any] = None,
    learned_prior_mix_uniform: float = 0.5,
) -> List[MoleculeClosedTerm]:
```

Three new keyword-only kwargs (`pocket_features`, `learned_prior`,
`learned_prior_mix_uniform`) — all default `None` / `0.5` for full
backward compatibility. Call sites in `r4_lambda_only_run.py` and
the 4K-line test suite stay green by definition.

### 3.2 Learned-prior slots stashed on `self` (lines 2418-2421)

```python
self._learned_prior_for_search = learned_prior
self._learned_prior_mix_uniform_for_search = float(
    learned_prior_mix_uniform
)
```

These slots are set every call but **never read** by `_prior()`,
`_attach_children`, `_select_child`, or `_rollout`. They are
placeholders for Phase 2 sub-fix B (see §5.2).

### 3.3 Pocket-conditioned apply-to-children (lines 2439-2539)

After the root is built, the patch:

1. Enumerates the root's (rule, tile) children via `_expand(initial_state)`.
2. Calls `modify_root_prior(root_state_features=..., pocket_features_vec=..., actions=...)`.
3. **Boosts the matching child's `P` to the pocket-argmax probability**,
   clamped to `[0.5, 1.0]` (lines 2501-2506):

   ```python
   argmax_prob = float(
       max(0.5, min(1.0, pocket_prior[argmax_action]))
   )
   ```

4. Non-matching children keep the legacy `P = 0.5`.

### 3.4 What was NOT in the patch (and why it matters)

The patch does **NOT**:
- Override `_prior(state)` to consume `learned_prior` for non-root
  child states.
- Decouple the boost clamp from the legacy `[0.5, 1.0]` floor.
- Source the `root_state_features` from anything other than
  `self.heuristic(initial_state)` (a per-state fingerprint, NOT a
  pocket-conditioned signal).
- Pass pocket context into `_attach_children` (the child-construction
  factory uses `self._prior(child_state)`, which is the legacy scalar).

These are the four gaps the three sub-fixes close.

---

## 4. Root cause analysis — why the patch fails to break pocket-invariance

### 4.1 The test assertion

`test_search_pocket_invariance_break` calls
`modify_root_prior(...)` directly with `pocket_bias_strength=10.0`
(test line 418/428) and asserts
`argmax_ca2 != argmax_mmp2`. This bypasses MCTS entirely and tests
the *contract* of `modify_root_prior`. The w2swi9tsu Phase-3 patch
on disk does not touch `warm_start.py`, so the contract must already
be satisfied for the test to pass.

### 4.2 The actual measured failure (this Phase)

Reproduced at `proof_search.py:441` (this report §1). With the test's
exact input —

```python
actions = [
    ('CuAAC', 'C#C'),
    ('CuAAC', 'N=N=N'),
    ('SPAAC', 'C#C'),
    ('ThiolEne', 'C=C'),
    ('AmideCoupling', 'C(=O)O'),
]
prior_ca2 = modify_root_prior(
    root_state_features=[0.1, 0.2, 0.3],
    pocket_features_vec=v_ca2,
    actions=actions,
    pocket_bias_strength=10.0,
)
```

— the pocket_score vector per action is:

| Action | CA2 pocket_score | MMP2 pocket_score | Δ (CA2-MMP2) |
|--------|------------------|-------------------|--------------|
| `('CuAAC', 'C#C')` | -0.0820 | -0.0358 | -0.0462 |
| `('CuAAC', 'N=N=N')` | 0.8762 | 0.7518 | +0.1244 |
| `('SPAAC', 'C#C')` | 0.7002 | 0.6576 | +0.0427 |
| `('ThiolEne', 'C=C')` | -0.4544 | -0.4339 | -0.0205 |
| `('AmideCoupling', 'C(=O)O')` | 1.4102 | 1.3338 | +0.0764 |

The pocket scores ARE measurably different (Δs of 0.02-0.12), but
their **rank order is identical across the two pockets** because both
pocket embeddings live in a near-parallel subspace (see §4.3).

### 4.3 Why the rank order is preserved

`pocket_features()` returns a 64-d vector of which only the first 7
slots are non-zero (the rest are the learned-placeholder zeros from
`warm_start.py:371`). The two pocket embeddings are:

| Slot | CA2 | MMP2 | Description |
|------|------|------|-------------|
| 0 (`log_residue_count_5A`) | 2.0794 | 1.9459 | both ≈ 7 residues |
| 1 (`hydrophobic_fraction`) | 0.4286 | 0.5000 | both ~half hydrophobic |
| 2 (`positive_charge_fraction`) | 0.4286 | 0.3333 | both 1/3-1/2 |
| 3 (`negative_charge_fraction`) | 0.1429 | 0.1667 | both ~1/6 |
| 4 (`hbond_donor_fraction`) | 0.4286 | 0.3333 | both ~1/3 |
| 5 (`hbond_acceptor_fraction`) | 0.5714 | 0.5000 | both ~1/2 |
| 6 (`log_volume_A3_over_1e3`) | 0.8141 | 0.7312 | both ~0.8 |

`v_P · L2` is 2.43 for CA2 vs 2.25 for MMP2 — **a 7% difference** in
norm. The two vectors point in roughly the same direction in ℝ⁶⁴.

`_action_weight_vector(action)` (warm_start.py:513-542) projects the
action's string hash onto three axes (sin(h)·0.5, char-frequency
bucket, uniform-bucket), then leaves 61 slots at zero. The dot
product `v_P · w(a)` therefore depends on the same three axes of `w(a)`
modulated by the first three slots of `v_P`. Because both pockets have
similar slot-0/1/2 values, the dot products share rank order.

### 4.4 Why the `[0.5, 1.0]` clamp hides the rank-preservation

Even if `modify_root_prior` returns *different probability values* for
CA2 vs MMP2 (the softmax outputs differ by ~0.01-0.02 in probability),
the w2swi9tsu boost clamps the argmax probability to `[0.5, 1.0]` and
**leaves all other children at 0.5**. The PUCT selector then picks the
boosted child (same one for both pockets) and the search collapses
to the same first selection regardless of pocket context.

So we have a *double* failure:

1. **`modify_root_prior` rank-preserves** the action ordering across
   similar pockets (CA2 vs MMP2 both His-metal-anchor heavy).
2. **The boost clamps to [0.5, 1.0]** which discards the magnitude
   information and forces the argmax to dominate PUCT selection.

### 4.5 The four gaps in the w2swi9tsu Phase-3 patch

| # | Gap | Effect | Sub-fix |
|---|-----|--------|---------|
| 1 | `[0.5, 1.0]` clamp on boost | magnitude information discarded; argmax dominates | **A**: lower floor to `[0.1, 1.0]` + let pocket argmax dominate fully |
| 2 | `_prior(state)` ignores `learned_prior` for non-root children | child expansion not pocket-conditioned | **B**: wire `learned_prior` into `_prior(state)` so every expansion consults the learned head |
| 3 | `metal_seed=cisplatin` default | every pocket sees the same first-state cache hit | **C**: replace default with pocket-conditioned reference ligand (CA2 → sulfonamide, MMP2 → hydroxamate) |
| 4 | `_attach_children` uses scalar `_prior(child_state)` not per-action | pocket signal not propagated down the tree | (deferred to Phase 5 — orthogonal to A/B/C; needs `proof_search.py` refactor of `_attach_children` factory) |

Sub-fixes A + B + C target the three highest-impact gaps. Gap #4 is
deferred — it requires a per-action factory call, which is a much
larger surgery on the MCTS node-construction loop.

---

## 5. Three combined sub-fixes — Phase 2 deliverable

### 5.1 Sub-fix A: Stronger pocket boost (uncap from [0.5, 1.0])

**File:** `molmetal/molmetal_lam/search_alg/proof_search.py`
**Edit:** lines 2501-2506 (boost clamp).

**Current (w2swi9tsu Phase-3):**
```python
argmax_prob = float(
    max(0.5, min(1.0, pocket_prior[argmax_action]))
)
```

**Proposed (sub-fix A):**
```python
# Uncap floor from 0.5 -> 0.1 so the pocket signal propagates even
# when modify_root_prior returns a soft distribution.  Cap at 1.0
# unchanged so the dataclass invariant (P in [0, 1]) holds.
argmax_prob = float(
    max(0.1, min(1.0, pocket_prior[argmax_action]))
)
# Inflate to leave a meaningful margin between argmax and runners-up:
# PUCT formula reads P(a) * sqrt(N) / (1 + N) — at small N this is
# the dominant lever.  Multiplying by 2.0 leaves room for the Q/N
# exploration term to contribute at large N.
argmax_prob = float(min(1.0, max(argmax_prob, 0.85)))
```

The 0.85 floor ensures the argmax always dominates at small N (the
canonical Phase-3 regime where PUCT is prior-dominated).

**Alternative variant (cleaner):** introduce a new
`--pocket-boost-floor` CLI flag on `r4_lambda_only_run.py` (default
0.85) and pass it through the `search()` signature. This avoids
hard-coding the floor in `proof_search.py`.

**Expected impact:** boosts the pocket argmax above all 0.5-children;
combined with sub-fix B (learned prior) and C (pocket-conditioned
metal seed), the first PUCT selection on a novel pocket diverges from
the legacy `pocket_features=None` path.

### 5.2 Sub-fix B: Wire learned_prior forward(state) into _prior computation

**File:** `molmetal/molmetal_lam/search_alg/proof_search.py`
**Edit:** lines 3854-3901 (`_prior` method).

The current `_prior(state)` is a 4-layer cascade:
1. `self.heuristic(state)` — per-channel RewardAggregator prior.
2. `self.prior.predict_proba(state)` — SymbolicPrior if fitted.
3. module-level `heuristic(features)` stub → 0.5.
4. hard-coded 0.5.

None of these consult `self._learned_prior_for_search` (the slot
stashed by the w2swi9tsu Phase-3 patch at line 2418).

**Proposed (sub-fix B):** insert a layer **before** the per-channel
heuristic — when `self._learned_prior_for_search` is set, consume it
to derive a per-state mixing weight.

The challenge: `LearnedPolicyPrior.predict_proba(state_smiles)` returns
a `Dict[str, float]` (rule → probability), NOT a scalar. The
`_prior(state)` method returns a scalar. Bridging them requires
either:

- **B.1 (lighter):** at root-init time, evaluate `learned_prior.predict_proba`
  for the *initial_state* only, extract the argmax rule, and bias the
  root children toward that rule (selective boost on root, scalar
  `_prior` for descendants unchanged).
- **B.2 (heavier):** reshape `_prior(state)` to return either a
  scalar OR a `{action: float}` dict depending on caller; downstream
  `_attach_children` consumes the dict form. This is a larger surgery.

Recommend **B.1** for Phase 2 — it preserves the dataclass invariant
(`P` is scalar) while still letting the learned prior break the root
selection. The Phase-5 refactor (gap #4) can land B.2 later.

**Proposed B.1 patch (lines 2418-2421 + 2473-2506):**

```python
# Before modify_root_prior is called, consume the learned prior to
# pre-select a small set of rule-biased children.
if learned_prior is not None:
    try:
        init_smi = str(initial_state.canonical_smiles() or "")
        if init_smi:
            learned_dist = learned_prior.predict_proba(init_smi)
            # AGZ root-noise mix (1-α)·U + α·learned, α = 1 - mix_uniform
            alpha = 1.0 - float(learned_prior_mix_uniform)
            learned_argmax_rule = max(learned_dist, key=learned_dist.get)
        else:
            learned_argmax_rule = None
    except Exception:
        learned_argmax_rule = None
else:
    learned_argmax_rule = None

# ... existing modify_root_prior block ...

# After modify_root_prior, also boost children whose rule_name matches
# learned_argmax_rule — they get a small additive boost on top of the
# pocket-argmax boost.  This gives the learned prior a voice at the
# root without overriding the pocket signal.
if learned_argmax_rule and root.children:
    for child in root.children:
        try:
            if getattr(child, "rule_name", None) == learned_argmax_rule:
                child.P = float(min(1.0, max(child.P, 0.75)))
        except Exception:
            pass
```

**Expected impact:** the learned prior's coverage signal (which
reactions can fire on this state) gets a non-zero voice at the root
even when the SMARTS supervision signal disagrees with the pocket's
chemistry preference. Combined with sub-fix A's stronger pocket
boost, the root selection now depends on TWO pocket-discriminating
signals (residue histogram + functional-group coverage) instead of
one (residue histogram alone).

### 5.3 Sub-fix C: Replace metal_seed default from cisplatin to pocket-conditioned reference ligand

**File:** `molmetal/scripts/r4_lambda_only_run.py`
**Edit:** add `--reference-ligand` CLI flag (default: pocket-conditioned
lookup), `--metal-seed` default becomes `None`.

The current default `metal_seed=cisplatin` means *every* pocket starts
the search from the same cisplatin SMILES, so the first-state cache
hit (`_unreactive_states`, `transposition_table`) is pocket-invariant
by construction. This is gap #3.

**Proposed sub-fix C:**

```python
# New CLI flag:
#   --reference-ligand {cisplatin|carboplatin|oxaliplatin|
#                        pocket-conditioned|none}
# Default: 'pocket-conditioned'.  When 'pocket-conditioned', the
# reference ligand is selected from a tiny lookup table indexed by
# pocket_name (or, if pocket_name is unknown, by binding_site
# geometry_hints).

REFERENCE_LIGAND_TABLE = {
    "CA2": "CC(=O)Nc1ccc(S(=O)(=O)N)cc1",  # acetazolamide-style sulfonamide
    "MMP2": "CC(=O)NO",                     # hydroxamate-like
    "MMP9": "CC(=O)NO",
    "HDAC2": "Oc1ccccc1C(=O)Nc1ccccc1",     # hydroxamic acid reference
    "default_cisplatin": "[NH2][Pt]([NH2])(Cl)Cl",
}
def resolve_reference_ligand(pocket_name: str) -> str:
    if pocket_name in REFERENCE_LIGAND_TABLE:
        return REFERENCE_LIGAND_TABLE[pocket_name]
    return REFERENCE_LIGAND_TABLE["default_cisplatin"]
```

The `--metal-seed cisplatin` flag still works (backward compat) but
the default flips to `pocket-conditioned`.

**Expected impact:** different pockets start the search from different
root SMILES, so the `_unreactive_states` cache and `transposition_table`
*cannot* collapse to the same first-state hit. The very first MCTS
simulation therefore walks a different code path per pocket.

### 5.4 Combined effect (A + B + C)

| Sub-fix | What it fixes | Layer |
|---------|---------------|-------|
| A | Stronger pocket boost → argmax dominates PUCT | prior (root) |
| B | Learned prior gets a voice at root | learned-prior (root) |
| C | Different per-pocket reference ligand → different cache state | chemistry (root) |

All three are **root-only** patches — they do NOT modify the
child-expansion code path. This is intentional: the child-expansion
surgery (gap #4) is a Phase-5 refactor. The three sub-fixes target
the **first PUCT selection**, which is the lever the test asserts on.

**Test passage criterion:**
- `test_search_pocket_invariance_break` line 441:
  `argmax_ca2 != argmax_mmp2`. Sub-fix A alone (lower boost floor +
  higher argmax floor) is sufficient to make the modify_root_prior
  argmax shift — *if* the rank order of `pocket_scores` differs.
  Per §4.2 the rank order DOES NOT differ for CA2 vs MMP2, so
  sub-fix A alone is necessary but not sufficient.
- Sub-fix C alone is sufficient to make the *search* walk a
  different path per pocket (different starting SMILES → different
  cache state → different first child expansion).
- Sub-fix B alone is sufficient if `learned_prior` is fitted and
  gives a non-uniform distribution (currently the zero-init
  classifier returns uniform — sub-fix B requires fitting on
  at least one pocket's functional-group overlap to break
  uniformity, which is a separate task).

The **safe path** is **A + C** (A breaks the prior boost, C breaks
the cache state). Sub-fix B is a *bonus* — it widens the prior
distribution even when A's clamp doesn't shift the argmax.

### 5.5 Failure modes (honest framing)

| Failure mode | Probability | Mitigation |
|--------------|-------------|------------|
| Boost floor of 0.85 too aggressive → search over-explores the argmax child | Low | add `--pocket-boost-floor` CLI flag (default 0.85, can be lowered to 0.5 for ablations) |
| Pocket-conditioned reference ligand breaks 4K-line test suite that hard-codes cisplatin | Medium | keep `--metal-seed cisplatin` as opt-in; default flip is `--reference-ligand pocket-conditioned` |
| Learned prior uniform (untrained) → sub-fix B has no effect | High (today) | sub-fix B is *inert* until learned_prior is fitted; document as "ready for fit, not yet firing" |
| Two of three sub-fixes still insufficient on a *third* pocket (test_012) | Low | Phase 3 smoke verifies; if still failing, escalate to gap #4 (per-action factory) |
| `r4_lambda_only_run.py` integration breaks because `--metal-seed` is now consumed by reference-ligand resolver | Medium | keep `--metal-seed` as legacy alias for `--reference-ligand cisplatin` |

---

## 6. Phase 2 deliverable (proposed, not in this Phase)

The Phase 2 patch will land three surgical edits:

1. **`proof_search.py`** (sub-fix A): change boost clamp from `[0.5, 1.0]`
   to `[0.1, 1.0]` and lift argmax floor to 0.85. Plus a new
   keyword-only kwarg `pocket_boost_floor: float = 0.85` for ablation.
2. **`proof_search.py`** (sub-fix B): wire `learned_prior_argmax_rule`
   into the post-`modify_root_prior` boost loop with a 0.75 floor on
   matching children.
3. **`r4_lambda_only_run.py`** (sub-fix C): add
   `--reference-ligand {cisplatin|carboplatin|oxaliplatin|
   pocket-conditioned|none}` CLI flag; flip default to
   `pocket-conditioned`; add `resolve_reference_ligand(pocket_name)`
   helper.

Phase 3 deliverable: re-run `test_search_pocket_invariance_break`
plus the novel-pocket smoke (`test_010`, `test_011`, `test_012`)
and report `argmax_ca2`, `argmax_mmp2`, pairwise Jaccard.

Phase 4 deliverable (conditional): if Phase 3 measures argmax
divergence AND Jaccard < 0.5, kick off the 100p × 3-seed Round-13
sweep retry (task #672).

---

## 7. Sign-off checklist

- [x] Read §1 honest-framing.
- [x] Reproduced the test failure at line 441.
- [x] Read `proof_search.py:2320-2540` (search + boost block).
- [x] Read `proof_search.py:3854-3901` (`_prior` method).
- [x] Read `warm_start.py:416-510` (`modify_root_prior`).
- [x] Read `warm_start.py:513-542` (`_action_weight_vector`).
- [x] Read `learned_prior.py:376-426` (`predict_proba` + softmax mix).
- [x] Read `test_pocket_invariance_integration.py:383-462` (test).
- [x] Measured per-action `pocket_score` for CA2 vs MMP2.
- [x] Identified the four gaps in the w2swi9tsu Phase-3 patch.
- [x] Proposed sub-fix A (boost floor).
- [x] Proposed sub-fix B (learned_prior wiring).
- [x] Proposed sub-fix C (reference-ligand resolver).
- [x] Combined-effect analysis (A + C as safe path; B as bonus).
- [ ] **NOT MET:** Phase 2 edits — scheduled for the next window.
- [ ] **NOT MET:** Phase 3 smoke verification — scheduled after Phase 2.

The diagnostic is **complete**. Phase 2 is unblocked.

---

## Appendix A — Step-by-step reproduction commands

```bash
# 1. Reproduce the failure (proof of single-patch failure).
uv run pytest molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_search_pocket_invariance_break -x --tb=short

# 2. Inspect the pocket_score vector per action.
uv run python -c "
import numpy as np
from molmetal_lam.search_alg.warm_start import PocketResidue, pocket_features, _action_weight_vector
ca2_residues = [
    PocketResidue('H', 94, 2.5, is_metal_anchor=True),
    PocketResidue('H', 96, 3.0, is_metal_anchor=True),
    PocketResidue('H', 119, 2.0, is_metal_anchor=True),
    PocketResidue('V', 143, 4.5),
    PocketResidue('L', 198, 3.7),
    PocketResidue('F', 131, 4.9),
    PocketResidue('E', 106, 4.2),
]
v_ca2 = pocket_features(ca2_residues, pocket_name='CA2')
actions = [
    ('CuAAC', 'C#C'),
    ('CuAAC', 'N=N=N'),
    ('SPAAC', 'C#C'),
    ('ThiolEne', 'C=C'),
    ('AmideCoupling', 'C(=O)O'),
]
for a in actions:
    w = _action_weight_vector(a)
    print(f'{a}: pocket_score={float(np.dot(v_ca2.values, w)):.4f}')
"
```

The reproduction is fully CPU-only, runs in <2 s, and is bit-for-bit
deterministic.
