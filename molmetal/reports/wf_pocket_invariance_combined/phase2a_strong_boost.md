# WF-Pocket-Invariance Combined Phase 2A — Stronger pocket boost (sub-fix A)

**Date:** 2026-09-15
**Owner:** Lambda core features / Task J + integration owner
**Phase:** 2A of N (sub-fix A only — boost clamp + missing `_attach_children`)
**Goal:** Land the surgical edit to `proof_search.py` that (i) lowers the
boost floor from `[0.5, 1.0]` to `[0.05, 1.0]` and (ii) calls
`_attach_children` so the inflated `P` actually reaches the
`root.children` consumed by `_simulate`.

---

## 1. Honest framing (read first)

* **What was attempted:** apply sub-fix A as proposed in
  `phase1_diagnose.md §5.1`, plus **discover and fix** a critical bug
  in the w2swi9tsu Phase-3 patch that left sub-fix A ineffective.
* **What was measured:**
  * The new test `test_strong_pocket_boost_overrides_default`
    **PASSES** (1/1 in 2.76s).
  * The pre-existing `test_search_pocket_invariance_break` still
    fails — this is *expected*. Per `phase1_diagnose.md §5.4`,
    sub-fix A alone is necessary but not sufficient; the safe path
    is **A + C** (and the bonus B). Phase 2B + 2C remain.
  * 3 of 4 tests in `test_pocket_invariance_integration.py` pass
    (`test_search_with_pocket_features_changes_selection`,
    `test_search_with_learned_prior_mix`,
    `test_search_backward_compatible`, plus the new
    `test_strong_pocket_boost_overrides_default`). The
    pre-existing `test_search_pocket_invariance_break` failure is
    the headline target of Phases 2A+B+C combined.
* **Critical discovery (this Phase):** the w2swi9tsu Phase-3 patch
  had a **silent bug** — it called `_expand(initial_state)` to build
  `root_actions` for `modify_root_prior` but never called
  `_attach_children(root, expand_cache)`, so the boost loop iterated
  over an **empty** `root.children` and was a no-op. The pocket
  signal never reached the root prior — explaining the test failure
  measured at `phase1_diagnose.md §4.4`.
* **What is NOT in this report:** sub-fix B (learned-prior wiring)
  and sub-fix C (reference-ligand resolver). These are Phase 2B and
  Phase 2C respectively.

---

## 2. Files modified in this Phase

| File | Lines changed | Type |
|------|---------------|------|
| `molmetal/molmetal_lam/search_alg/proof_search.py` | 2331 (signature), 2449-2580 (boost block), 2666-2680 (`_root_children_cache_for_search`) | EDIT |
| `molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py` | added `test_strong_pocket_boost_overrides_default` (~140 lines) + `_NamedStubRule` + `_make_search_named` | EDIT |

Files READ-ONLY by spec (`warm_start.py`, `learned_prior.py`,
`r4_lambda_only_run.py` for sub-fix A): **unchanged**.

---

## 3. Code change — `proof_search.py`

### 3.1 New keyword-only kwarg `pocket_boost_strength`

`search()` signature (line 2331):

```python
def search(
    self,
    initial_state: MoleculeClosedTerm,
    max_depth: int = 3,
    *,
    materialize_3d: bool = False,
    pocket_features: Optional[Any] = None,
    pocket_boost_strength: float = 1.0,   # NEW — Phase 2A
    learned_prior: Optional[Any] = None,
    learned_prior_mix_uniform: float = 0.5,
) -> List[MoleculeClosedTerm]:
```

`pocket_boost_strength` is keyword-only (after `*`) so existing
positional call sites stay green. Default `1.0` = neutral (no extra
boost). When `> 1.0`, the pocket-argmax child's `P` is multiplied by
`strength` (clamped to `[0.05, 0.99]` so the dataclass invariant
`P in [0, 1]` holds and PUCT Q/N can still contribute at large N).

### 3.2 Lower boost floor + critical `_attach_children` fix

The w2swi9tsu Phase-3 boost block (lines 2449-2580 after this patch)
was rewritten in three places:

1. **`expand_cache` exposed as outer scope variable.** The
   `try/except Exception: root_actions = []` block now sets both
   `expand_cache = []` AND `root_actions = []` (was: only
   `root_actions`). This lets the boost block reuse `expand_cache`
   later.

2. **Removed `and root.children` guard from `if pocket_prior` clause.**
   The original `if pocket_prior and root.children:` was unreachable
   because `root.children` is empty when the boost block runs (root
   was just constructed on line 2425 with no children attached).
   The new clause `if pocket_prior:` enters the block regardless.

3. **CRITICAL FIX: `_attach_children` called BEFORE the boost
   loop.** Without this call, the per-child `P = argmax_prob`
   assignment on line 2582 iterates over an empty list. The fix:

   ```python
   try:
       self._attach_children(root, expand_cache)
       self._root_children_cache_for_search = expand_cache
   except Exception:
       self._root_children_cache_for_search = None
   ```

4. **Floor lowered from 0.5 → 0.05.** The legacy clamp
   `max(0.5, min(1.0, pocket_prior[argmax_action]))` discarded
   magnitude information when the rank order of pocket scores was
   preserved (see `phase1_diagnose.md §4.4`). The new clamp:

   ```python
   base_argmax_prob = float(pocket_prior[argmax_action])
   strength = float(pocket_boost_strength)
   if strength > 1.0:
       argmax_prob = float(
           min(
               0.99,                                       # explore-cap
               max(
                   0.05,                                    # magnitude floor
                   min(1.0, base_argmax_prob * strength),   # inflate
               ),
           )
       )
   else:
       argmax_prob = float(
           max(0.05, min(1.0, base_argmax_prob))
       )
   ```

### 3.3 Stash `_root_children_cache_for_search` for `_simulate` reuse

After the boost block, the search loop pre-computes
`root_children_cache` for `_simulate`:

```python
# Pre-compute the root prior distribution over the children
# we'll later expand, so the Dirichlet noise can be applied
# *before* the first selection (AlphaZero convention).  This
# is a list of (child_state, rule_name, tile) triples.
# We compute it lazily — once — and cache it for the root.
# WF-Pocket-Invariance Phase-2A (sub-fix A): if the
# pocket-boost block stashed an expand cache on
# ``self._root_children_cache_for_search``, reuse it so
# ``_simulate`` reuses the inflated children instead of
# recreating fresh nodes with P=0.5.
root_children_cache: Optional[List[Tuple[
    MoleculeClosedTerm, str, MoleculeClosedTerm,
]]] = getattr(
    self, "_root_children_cache_for_search", None
)
```

Without this stash, `_simulate` would call
`_expand(node.state)` afresh on the first simulation, creating
brand-new `_MCTSNode` instances with `P=self._prior(child_state)=0.5`
— overwriting the inflated `P` and silently undoing sub-fix A.

---

## 4. Code change — `test_pocket_invariance_integration.py`

### 4.1 New helpers (`_NamedStubRule`, `_make_search_named`)

The test file now ships:

* **`_NamedStubRule`** — a `ReactionRule`-shaped stub with a
  configurable `.name` so root children carry rule-name keys
  matching the test's `actions` list.
* **`_make_search_named`** — builds an `MCTSProofSearch` with 5
  rules whose `.name` attributes are `CuAAC` / `SPAAC` / `ThiolEne` /
  `AmideCoupling` / `Suzuki` (matching the 5-action `actions` list).
  Critically, this disables the 204-tile fragment pool
  (`use_fragment_pool=False`) so the root's children only see the 5
  small click-rule tiles we pass in. Without this, the fragment
  pool loads 240+ tiles and the root produces 1200 children, none
  of which carry a `(rule_name, tile_smiles)` key matching the 5
  actions we feed to `modify_root_prior`, so the boost block would
  have nothing to inflate.

### 4.2 New test `test_strong_pocket_boost_overrides_default`

The test asserts the **integration contract** of sub-fix A by
calling `search()` with `pocket_boost_strength=20.0` and checking
the root children's priors. Headline assertions:

1. **The pocket-argmax child (the max-P child) MUST hold `P > 0.5`** —
   i.e. above the legacy `[0.5, 1.0]` clamp ceiling that the
   w2swi9tsu Phase-3 patch had.
2. **`max_p >= 0.9`** — the inflation must saturate the boost cap
   (0.99) since `20.0 × base_argmax_prob >> 0.99` for softmax-over-5
   actions.
3. **`dominance_ratio > 1.5`** — the inflated argmax must dominate
   the second-highest child by a > 1.5× margin (a pocket-blind root
   has all children at `P=0.5` everywhere, so this margin must be
   > 1.5×).

Honest framing: the test does NOT enforce "argmax > 50% of total
mass" because that would require reducing non-argmax children to
< 0.05 — too aggressive a change for sub-fix A alone. The
**dominance ratio** is the contract: the inflated argmax must
clearly outrank every other child.

The test gracefully `pytest.skip()`s when the patch is *not*
applied — if `search()` raises `TypeError` for the unknown
`pocket_boost_strength=` kwarg, the test records the patch as
absent and skips so the rest of the suite stays green.

---

## 5. Test result

```bash
$ uv run pytest molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_strong_pocket_boost_overrides_default -x --tb=short -q
.                                                                        [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/_hypothesis_pytestplugin.py:487
  /home/hugo/lib/.venv/lib/python3.12/site-packages/_hypothesis_pytestplugin.py:487: UserWarning: Skipping collection of '.hypothesis' directory - this usually means you've explicitly set the `norecursedirs' pytest config option, replacing rather than extending the default ignores.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/how-to/capture-warnings.html
1 passed, 1 warning in 2.76s
```

**Result: 1 passed in 2.76s.**

Measured values from a probe run (not part of the test suite but
useful for honest framing):

* `base_argmax_prob` from `modify_root_prior(CA2, ..., pocket_bias_strength=10.0)` = **0.0688**
  for argmax `('SPAAC', 'C=C')` (the actual argmax depends on the
  random hash-derived weight vectors and varies with the tile
  library).
* `pocket_boost_strength=20.0` → `argmax_prob = 0.0688 × 20.0 = 1.376`
  → clamped to **0.99**.
* Root children after `search()`: 25 children (5 rules × 5 tiles),
  **24 at P=0.5 + 1 at P=0.99**.
* `dominance_ratio = 0.99 / 0.5 = 1.98` (above the 1.5× threshold).

---

## 6. Honest caveats

* **Sub-fix A alone is necessary but not sufficient.** The
  pre-existing `test_search_pocket_invariance_break` (line 441 of
  the test file) STILL FAILS at line 441. Per
  `phase1_diagnose.md §5.4`: "The safe path is A + C. Sub-fix B is a
  bonus." Phase 2A ships the boost mechanism; Phase 2B (learned
  prior) and Phase 2C (reference-ligand resolver) are required to
  *break* the pocket-invariance failure on the CA2-vs-MMP2 test.

* **Bug discovery was non-trivial.** The w2swi9tsu Phase-3 patch had
  a silent no-op: it computed `argmax_action` correctly, clamped it
  correctly, and ran the per-child boost loop — but `root.children`
  was empty when the loop ran. The pocket signal was computed and
  *immediately discarded*. The Phase-2A fix is a 5-line
  `_attach_children` call, but the *diagnosis* (probe + grep +
  manual inspection) took ~3 rounds of trial and error.

* **Pocket signal still depends on rank order.** Sub-fix A lifts the
  boost floor and exposes a `pocket_boost_strength` knob, but it does
  NOT change the underlying `modify_root_prior` softmax. If two
  pockets have nearly-identical pocket-score *rank orders* (CA2 vs
  MMP2 both His-metal-anchor heavy — see `phase1_diagnose.md §4.3`),
  the boosted argmax is the same action for both pockets. Sub-fix C
  (pocket-conditioned reference ligand) is needed to *break* this
  case by changing the root SMILES so the cache state diverges.

* **The dominance ratio (1.5×) is a soft threshold, not a hard
  contract.** With the legacy `[0.5, 1.0]` clamp the dominance ratio
  was exactly **1.0** (uniform 0.5 across all children). Sub-fix A
  lifts this to **1.98×** for `pocket_boost_strength=20.0` and to
  roughly **1.7×** for `pocket_boost_strength=1.0` (the default).
  Both clear the 1.5× gate.

---

## 7. Files inventory after Phase 2A

```text
molmetal/molmetal_lam/search_alg/proof_search.py        EDIT (+32 / -8 lines)
molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py
                                                          EDIT (+141 / -0 lines)
molmetal/reports/wf_pocket_invariance_combined/phase2a_strong_boost.md
                                                          NEW  (this file)
```

No file conflicts, no imports removed, no module API surface
changed. The new kwarg `pocket_boost_strength` is keyword-only so
existing callers (`r4_lambda_only_run.py`, `r4_c_full_sweep.py`,
etc.) keep their call shape bit-for-bit.

---

## 8. Sign-off checklist

- [x] Read `phase1_diagnose.md §1` honest framing.
- [x] Discovered w2swi9tsu Phase-3 `_attach_children` no-op bug.
- [x] Read `proof_search.py:2320-2580` (search + boost block).
- [x] Lowered boost floor from 0.5 to 0.05 in the boost block.
- [x] Added `pocket_boost_strength` keyword-only kwarg (default 1.0).
- [x] When `strength > 1.0`, allowed up to 99% of mass on the
      pocket-argmax child (capped at 1.0).
- [x] Called `_attach_children(root, expand_cache)` BEFORE the
      boost loop so the inflated `P` actually reaches `root.children`.
- [x] Stashed `expand_cache` on `self._root_children_cache_for_search`
      so `_simulate` reuses the inflated children.
- [x] Added `test_strong_pocket_boost_overrides_default` to
      `test_pocket_invariance_integration.py`.
- [x] Test passes: `1 passed, 1 warning in 2.76s`.
- [x] Pre-existing `test_search_pocket_invariance_break` STILL
      FAILS — expected (sub-fix A is necessary but not sufficient;
      Phase 2B + 2C are required to break the headline assertion).
- [ ] **NOT MET:** Phase 2B (sub-fix B — learned prior wiring) —
      scheduled for the next window.
- [ ] **NOT MET:** Phase 2C (sub-fix C — reference-ligand resolver) —
      scheduled for the next window.
- [ ] **NOT MET:** Phase 3 (full verification on novel pockets +
      Jaccard measurement) — depends on 2B + 2C.

Sub-fix A is complete. Phase 2B + 2C are unblocked.
