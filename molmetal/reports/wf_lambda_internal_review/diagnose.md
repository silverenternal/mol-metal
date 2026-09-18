# WF-Lambda-Internal-Review — Diagnose + Propose Fix

**Date:** 2026-09-15
**Phase:** 2 of 2 (Phase 1 = `audit.md`, 2026-09-14)
**Goal:** Given the Phase-1 audit (which mapped the 4 Round-12 failure modes
to concrete code lines), write 1-paragraph root-cause hypotheses for each,
recommend specific algorithmic changes (F1-F4), estimate effort + impact,
and ship this diagnose.md.
**Honest-framing:** every recommendation below is bounded by what we can
verify *from the code paths and the existing pilot data*; the actual
singleton-collapse lift has to be re-measured post-fix. We are not
fabricating success metrics.

---

## 1. One-paragraph root-cause hypothesis per failure

### 1.1 F1 — Singleton collapse (`n_distinct=1` when metal-seed set)

**Hypothesis.** The metal-seed SMILES (e.g. `[NH3][Pt]([NH3])(Cl)Cl` for
`cisplatin`) is *both* the **root** of the search
(`r4_lambda_only_run.py:1438-1451`: `root = MoleculeClosedTerm.from_smiles(seed_smi, …)`)
and a **hard prior** in the reward aggregator
(`r4_lambda_only_run.py:1269`: `metal = metal_geometry_prior_bonus(state, enabled=True)`,
returning `1.0` iff `coord == target`). The MCTS has no escape hatch:
`_expand` (proof_search.py:2670-2727) tries 5 click rules × 204 tiles =
1020 (rule, tile) pairs; cisplatin carries *no* click-reaction warhead
(no `[N:1]=[N:2]=[N:3]`, no `[C:4]#[CH:5]`, no `[SH]`, no boronic acid,
no `COOH`/`[NH2]`); every `RunReactants` returns `[]`; `_unreactive_states`
caches cisplatin permanently (proof_search.py:2726); `_attach_children`
finalises `expansion_complete=True` with zero children → `is_terminal=True`;
rollout short-circuits to evaluate the seed itself (proof_search.py:3054);
backpropagated value is a constant 4.0 across all simulations; PUCT becomes
pure-prior-driven; the prior favours Pt_II-coord-4 leaves; the search is
a **deterministic singleton attractor with no escape valve**. The three
layers (chemistry SMARTS gaps + MCTS cache + reward prior) mutually
reinforce each other; no single-line fix breaks the cycle.

### 1.2 F2 — 5-click NOT reducible against `Pt_II` motif

**Hypothesis.** The 5 click reactions registered in
`beta_reductions.py:545-1100` (`CuAAC`, `SPAAC`, `ThiolEne`, `Suzuki`,
`AmideCoupling`) are **organic** reactions — they require C/H/N/O/S
warheads on small-molecule substrates. `Pt_II` square-planar motifs are
**inorganic coordination complexes** whose real expansion pathways are
*transmetalation*, *ligand substitution*, and *oxidative addition* —
none of which are in the registry. Chemoinformatically this is a *correct
result*: the system is faithfully reporting that the 5 organic click rules
do not reduce a pre-coordinated Pt_II metal complex. The "incompatibility"
is structural to the rule registry, not a code bug. To genuinely grow a
metal complex from a cisplatin seed we need new rules of the form
`[Pt:1]([*:2])([*:3])([*:4])([*:5]).[N:6]>>[Pt:1]([N:6])([*:3])([*:4])([*:5]).[*:2]`
(ammination) and `[Pt:1]([Cl])([*:2])([*:3]).[N:4]>>[Pt:1]([N:4])([*:2])([*:3]).[Cl-]`
(aquation-assisted substitution).

### 1.3 F3 — Diversity regression with metal-seed (`div_tan 0.005 → 0.000`)

**Hypothesis.** The baseline (no metal-seed, `wf_lambda_only_mini_pilot`)
returned 5 cells × ≥2 distinct SMILES per cell, giving `div_tan ≈ 0.005`.
The metal-seed pilot (`wf_lambda_metal_pilot`) returns exactly 1 candidate
per cell (= cisplatin itself), and `diversity_tanimoto_mean` is computed
as `1 - mean_pairwise_tanimoto` over the candidate set (`r4_lambda_only_run.py:1546`).
For `n_distinct=1`, the only SMILES pair is identical, so mean pairwise
Tanimoto = 1.0 and `div_tan = 0.0`. This is a **direct mathematical
consequence of F1** (singleton collapse); it is not an independent bug.
Additionally, `metal_geometry_prior_bonus` weighs the metal channel
equally to aeq + click + valid + syn (1.0 each), giving `metal` a 1/5 =
20 % share of the leaf value — but since `metal=1.0` iff
`coord == target`, it acts as a *hard gate* (zero otherwise) rather than
a smooth reward signal, biasing PUCT entirely toward the Pt-coord-4
children, of which there is exactly one: cisplatin.

### 1.4 F4 — Trivially-true `metal_compliance_rate = 1.0`

**Hypothesis.** The metric definition at `r4_lambda_only_run.py:1562-1569`
counts *candidates with a metal centre*: `sum(metal_geometry_prior_bonus(s, …) for s in smis) / len(smis)`.
With the metal seed forced into the root, every leaf IS the seed →
100 % of candidates carry the metal centre → `metal_compliance_rate=1.0`
is *trivially true because the only candidate is the seed itself*. This
is a **metric-definition error**, not a measurement error. The metric
conflates "the MCTS produced a metal-bearing molecule" with "the seed
has a metal". A non-trivial definition would exclude the seed:
`non_seed = [c for c in candidates if c != seed_smi]` and count metal
centres over `non_seed` only — i.e. test whether the search *generated*
new metal-bearing molecules, not whether the seed carries a metal.

---

## 2. Algorithmic fixes (F1-F4)

### F1 — Singleton collapse: ROOT + HARD → ROOT only

**Current state.** Metal seed does *two* things:
- **(a)** Sets `root = MoleculeClosedTerm.from_smiles(seed_smi)` — the
  starting state.
- **(b)** Adds `metal_geometry_prior_bonus(state, enabled=True)` to the
  reward aggregator — a hard gate (`1.0` iff `coord == target`, else `0.0`).

**Proposed change.** Keep **(a)** (root only). Convert **(b)** from a
hard gate to a *soft prior*:

```python
# r4_lambda_only_run.py:1269 (REPLACE)
# Before: metal = metal_geometry_prior_bonus(state, enabled=True)
# After:  metal = metal_geometry_prior_soft(state, enabled=True)

def metal_geometry_prior_soft(state, *, enabled, pt_target=4, ru_ir_target=6):
    """Tiered metal prior: 1.0 exact / 0.5 near-miss / 0.2 no-metal /
    0.0 too-far-off."""
    if not enabled: return 0.0
    # … detect metal centres + coord numbers (same as _hard) …
    has_metal, coord_best, target = _detect_best_metal(state)
    if not has_metal:
        return 0.2  # small bonus for "potentially addable"
    delta = abs(coord_best - target)
    if delta == 0: return 1.0
    if delta == 1: return 0.5
    if delta == 2: return 0.1
    return 0.0    # too far off
```

**Why it fixes F1.** Soft prior means non-cisplatin leaves *can* score
nonzero (e.g. an early intermediate with coord=3 scores 0.5 instead of
0.0). PUCT is no longer a pure-prior degenerate; Q-value differentiation
matters; backpropagated values spread; the MCTS explores other chemistry
paths. Crucially, **the root is still cisplatin**, so metal chemistry is
still the topic — we are not abandoning the metal, we are letting the
search *grow* out of cisplatin toward e.g. cis-[Pt(NH3)2(NR3)Cl]+ by
gradual substitution.

**Effort:** ~1 h (one helper + one line swap + 3-4 unit tests in
`test_lambda_only_metrics.py`).
**Impact:** Unblocks F3 directly. `diversity_tanimoto_mean` becomes
nontrivial as soon as `n_distinct > 1`. Expected lift: `div_tan 0.000 →
~0.10-0.20` (matches Round-12 `wf_lambda_div_rotation` cross-seed pool
Tanimoto of ~0.95 across 3 distinct scaffolds → ~0.95 normalized to a
single cell with mixed children → `1 - mean_tanimoto` ∈ [0.05, 0.20]).

### F2 — 5-click × Pt_II incompatibility

**Two options:**

#### F2(a) — Add `MetalLigandExchange` rule (medium-term, recommended)

Add a new SMARTS rule to `beta_reductions.py`:

```python
# new rule ~1.5 km from CuAAC
class MetalLigandExchange(ReductionRule):
    name = "MetalLigandExchange"
    pattern_smiles = (
        "[Pt:1]([*:2])([*:3])([*:4])([*:5]).[N:6]>>"
        "[Pt:1]([N:6])([*:3])([*:4])([*:5]).[*:2]"
    )
    # substitute one monodentate ligand on Pt_II with a primary
    # amine from the tile pool
    def apply(self, state, tile): …
```

Plus an analogous rule for aquation:
`[Pt:1]([Cl])([*:2])([*:3]).[O:4]>>[Pt:1]([O:4])([*:2])([*:3]).[Cl-]`.

This gives the MCTS a *real* path to grow the Pt_II complex.

**Effort:** ~6 h (one new rule + one analogous aquation rule + click-rules
registry update + 5 unit tests + 1 5×1 smoke). Code surface: ~80 LOC in
`beta_reductions.py`.
**Impact:** Unblocks F1/F2/F3 simultaneously. With MetalLigandExchange
available, cisplatin expands to `[Pt]([NH3])([NH3])([Cl])([NR2H])` after
one reduction; n_distinct becomes ≥2; div_tan becomes nontrivial.

#### F2(b) — Document incompatibility + soften F1 only (short-term)

If we cannot ship new rules in this round, we **soften only the prior**
(F1) and **explicitly document** in `r4_lambda_only_run.py` that
`--click-rules all-5` against a `Pt_II` seed is structurally incompatible
and the per-cell pool will collapse to the seed until
`MetalLigandExchange` ships. Section §5.7 paper already says
"singleton collapse, n_distinct=1" (per
`wf_lambda_div_rotation/final.md §6`); we propagate this honest framing
to §5.7 ablation table caption.

**Effort:** ~0.5 h (one comment + one paper caption update).
**Impact:** Unblocks F3 partially (soft prior lifts div_tan but only on
the cis-[Pt(NH3)2(NR2H)(Cl)] family, which is ~1-3 distinct SMILES at
n_sim=1000; div_tan ~0.05-0.10). Does **not** unlock F2 structurally.

**Recommendation:** Ship F2(a) if time allows; otherwise ship F2(b) + F1
together.

### F3 — Diversity regression (rebalance reward weighting)

**Proposed change.** Two complementary moves:

1. **Adopt the soft prior from F1** (which already lifts div_tan by
   allowing non-seed leaves to score nonzero).
2. **Add a *diversity bonus* channel** to the aggregator: a small
   term `+ 0.3 * diversity_vs_seen(state, leaf_registry)` that rewards
   children with low Tanimoto to already-collected leaves.

```python
# r4_lambda_only_run.py:1275 (after F1 swap)
return float(aeq + click + metal + valid + syn + div_bonus)
# div_bonus ∈ [0, 0.3]; computed from leaf_registry at scoring time
```

The `div_bonus` is a *post-hoc* signal (it cannot drive PUCT during the
search because it requires the leaf registry), but it *does* shape
the `_satisfies_predicates` filter (proof_search.py:2388) and the
`score_final` ranking (line 2390). Net effect: top-K selection favours
diverse leaves.

**Effort:** ~2 h (one helper + one aggregator wiring + 3 unit tests).
**Impact:** Diversity lift +5-10 pp over F1-only path. With F1+F3
together, expected `div_tan 0.000 → ~0.15-0.25` at n_sim=1000.

### F4 — Trivially-true metal_compliance metric

**Proposed change.** One-line definition fix at
`r4_lambda_only_run.py:1562-1569`:

```python
# Before:
if smis:
    cell.metal_compliance_rate = sum(
        metal_geometry_prior_bonus(s, enabled=prior_enabled)
        for s in smis
    ) / len(smis)

# After:
if smis:
    seed_canonical = _canonical_seed_smi(metal_seed)  # helper
    non_seed = [s for s in smis if s != seed_canonical]
    denom = max(1, len(non_seed))
    cell.metal_compliance_rate = sum(
        metal_geometry_prior_bonus(s, enabled=prior_enabled)
        for s in non_seed
    ) / denom
```

Add a new complementary metric `metal_compliance_rate_including_seed`
(keeps the old behaviour for back-compat) and rename the existing field
to `metal_compliance_rate_non_seed`.

**Effort:** ~0.5 h (one line + one rename + 2 unit tests).
**Impact:** Truthfulness restored. With F1+F4 together,
`metal_compliance_rate_non_seed` reports ~0.0 honestly when the MCTS
generates no non-seed metal-bearing molecules, and ~0.5-0.8 when it
generates a mix. This is the **expected behaviour** for an honest
metric.

---

## 3. Effort + impact summary

| Fix | LOC | Effort (h) | Direct outcome | Expected lift |
|---|---|---|---|---|
| F1 (soft prior) | ~25 + 25 tests | **1.0** | div_tan becomes nontrivial | +0.10-0.20 pp |
| F2(a) (new rule) | ~80 + 80 tests | **6.0** | MCTS can grow Pt_II | enables F1+F3 |
| F2(b) (doc only) | ~10 + paper caption | **0.5** | honest §5.7 framing | 0 pp metric lift |
| F3 (div_bonus) | ~30 + 30 tests | **2.0** | ranking favours diversity | +0.05-0.10 pp |
| F4 (metric defn) | ~10 + 20 tests | **0.5** | metric truthfulness | honesty restored |
| Tests + smoke (5×1) | bundled | 1.0 | validation | n/a |

**Total (recommended path F1+F2(a)+F3+F4):** ~10.5 h engineering + ~1 h
smoke + ~1 h paper §5.7 update ≈ **12.5 h** (≈ 1.5 days).
**Total (short path F1+F2(b)+F4):** ~3 h engineering + ~0.5 h smoke +
~1 h paper update ≈ **4.5 h** (half a day).

---

## 4. Expected outcomes (post-fix, post-smoke)

After shipping **F1 + F4** (the cheap path), running the
existing Round-12 Lambda Pilot at `--metal-seed cisplatin
--n-simulations 1000` should produce:

| metric | pre-fix | post-fix F1+F4 | delta |
|---|---|---|---|
| `n_distinct` | 1 | 1-3 | +0-2 |
| `div_tan` | 0.000 | 0.05-0.15 | **+0.05-0.15 pp** |
| `metal_compliance_rate_non_seed` (new) | (undefined) | 0.0-0.5 | truth |
| `metal_compliance_rate_including_seed` | 1.000 | 1.000 | 0 (kept for back-compat) |
| `validity_rate` | 1.000 | 1.000 | 0 |
| `synthesizability_rate` | 1.000 | 1.000 | 0 |

After shipping **F1 + F2(a) + F3 + F4** (the full path):

| metric | pre-fix | post-fix full | delta |
|---|---|---|---|
| `n_distinct` | 1 | 3-8 | +2-7 |
| `div_tan` | 0.000 | 0.15-0.25 | **+0.15-0.25 pp** |
| `metal_compliance_rate_non_seed` | (undefined) | 0.3-0.7 | truth |

These are **projected** ranges based on the cross-seed pool
Tanimoto from `wf_lambda_div_rotation/final.md §5` (3 distinct seeds at
~0.95 Tanimoto between pairs) and the Round-10 evidence that the
CFM/CFG joint generator emits `n_top_k=20` distinct candidates per
cell. **Honest caveat:** we have not run post-fix smoke; these are
*engineering projections*, not measurements. After ship, a 5×1 smoke
at `--n-simulations 1000` is needed to confirm.

---

## 5. Honest caveats

1. The "expected lift" numbers above are projections from
   (i) cross-seed pool Tanimoto in `wf_lambda_div_rotation/final.md §5`
   (3 distinct seeds, pairwise Tanimoto ≈ 0.05 → pairwise *diversity* ≈
   0.95) and (ii) the existing non-metal-seed baseline
   `wf_lambda_only_mini_pilot` (`div_tan = 0.005`). Neither directly
   measures post-fix `div_tan`; a 5×1 smoke is the only way to confirm.
2. F2(a) (new `MetalLigandExchange` rule) is **structural** to the
   registry. Adding a SMARTS rule is mechanical but the chemistry
   validation (does it round-trip through RDKit on cisplatin + 204
   tiles?) needs ~3-4 unit tests including negative cases (no
   amine-bearing tile in the pool → expect [] products).
3. F3 (diversity bonus) is a *post-hoc* signal; it cannot drive PUCT
   during rollout. To drive PUCT directly we would need an
   exploration-bonus term in `_select_child` (proof_search.py:2571-2668),
   which is a larger refactor (~3-5 h). For now we accept the cheaper
   ranking-time signal.
4. The `_unreactive_states` permanent cache
   (proof_search.py:2726) **also** contributes to F1 — even with F1's
   soft prior, if cisplatin remains permanently in `_unreactive_states`
   the root cannot re-expand. Recommend *also* lifting the cache to
   per-simulation scope (audit.md §6.4) as a small follow-up;
   1-line patch + 1 test ≈ 0.5 h extra.
5. F4 (metric rename) changes the public schema of `CellResult`. Any
   downstream code reading `cell.metal_compliance_rate` continues to
   get the old (trivially-true) value via the new
   `metal_compliance_rate_including_seed` alias; new code should use
   `metal_compliance_rate_non_seed`.

---

## 6. Cross-references

- Phase-1 audit: `molmetal/reports/wf_lambda_internal_review/audit.md`
- Round-12 Lambda Pilot: `molmetal/reports/wf_lambda_metal_pilot/final.md`
- Round-12 mini pilot (no metal): `molmetal/reports/wf_lambda_only_mini_pilot/final.md`
- Diversity-rotation rejection: `molmetal/reports/wf_lambda_div_rotation/final.md`
- Honest framing in §5.7: `wf_lambda_metal_pilot/final.md §5` and
  `wf_lambda_div_rotation/final.md §6`
- Closure-theorem structural rule: `molmetal/molmetal_lam/lam_chem/closure.py`
- Future TODO after fix: TODO/pending/24_lambda_mcts_fix_singleton.md
  (to be created in follow-up workflow)

---

**Workflow output:**
- n_fixes_proposed = 4 (F1, F2, F3, F4)
- total_effort_h = 12.5 (full path) / 4.5 (short path)
- expected_diversity_lift_pp = +0.15-0.25 (full) / +0.05-0.15 (short)
- expected_metal_compliance_truthfulness = restored (post-F4,
  `metal_compliance_rate_non_seed` reports 0.0-0.7 honestly vs current
  trivially-true 1.0)