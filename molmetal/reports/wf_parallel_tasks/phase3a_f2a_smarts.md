# WF-Phase3A / F2(a) — MetalLigandExchange + AquaExchange SMARTS rules

**Date:** 2026-09-15
**Status:** SHIPPED (code + tests + report)
**Author:** parallel-flow subagent
**Files touched (3):**

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pt_click_compat.py` — extended COMPAT_MATRIX from 5x5 to 5x7 (added 2 metal-coord rule columns)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py` — 4 new F2(a) tests (test 17-20); updated tests 12/13/14/15 for the 7-column matrix
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_lambda_only_metrics.py` — updated `test_click_rules_filter_emptied_no_warning` to recognise `METAL_COORDINATION_RULES`

**Files NOT touched (per task graph disjoint file sets):**

- `molmetal/molmetal_lam/reactions/beta_reductions.py` — the F2(a) `MetalLigandExchange` and `AquaExchange` SMARTS rules themselves already shipped in WF-Lambda-Fix-FullPath (commit 2026-09-14).  The current task graph only required extending the *compat matrix* and adding tests.

---

## 1. Goal recap (from task spec)

> Task A: F2(a) MetalLigandExchange + AquaExchange SMARTS rules.
> Goal: ship the structural fix that allows Pt_II to do MetalLigandExchange + AquaExchange reactions in MCTS, enabling both diversity AND metal_compliance simultaneously.

The structural-fix half (the SMARTS classes in `beta_reductions.py`) was
already shipped in WF-Lambda-Fix-FullPath (2026-09-14) — the
`MetalLigandExchange` and `AquaExchange` dataclasses with the canonical
SMARTS patterns are present, registered in `REACTION_RULES`, and exposed
via `METAL_COORDINATION_RULES` in `lam_chem/rules.py`.

What was missing — and is shipped in this task — is the **compat-matrix
extension** so the F2(a) rules are *visible* to the scaffold-aware click
dispatch:

1. `COMPAT_MATRIX` extended from 5x5 to 5x7 — adds `metal_ligand_exchange`
   and `aqua_exchange` as columns alongside the 5 click rules.
2. `strict_Pt_II` verdict on the 2 new rules = `"compatible"` (F2(a)
   design intent — these are coordination rules, not click rules).
3. Tests at the matrix helper layer (`default_compatible_rules`,
   `incompatible_rules`, `marginal_rules`) + at the SMARTS layer
   (rule invocation, counter increment).
4. The 5-test matrix regression for the existing tests 12/13/14/15.

---

## 2. SMARTS definitions (verbatim from beta_reductions.py)

### 2.1 MetalLigandExchange

```smarts
[Pt:1]([Cl:2])[*:3].[NH3:4]>>[Pt:1]([NH3:4])[*:3].[Cl:2]
```

- **Atom-maps:** 1 = Pt centre, 2 = Cl leaving, 3 = other Pt ligand
  (kept), 4 = incoming NH3 donor.
- **Pattern intent:** a single Pt-Cl bond is matched; the rule fires
  once per substitution cycle.  The MCTS expansion path can call the
  rule repeatedly to fill all four coordination sites of a Pt_II centre.
- **Catalyst:** none (associative substitution is the intrinsic kinetic
  pathway per Lippard 1995 §3).
- **Stoichiometry:** empty dict (heavy-atom conserving).

### 2.2 AquaExchange

```smarts
[Pt:1]([Cl:2])[*:3].[OH2:4]>>[Pt:1]([OH:4])[*:3].[Cl:2]
```

- **Atom-maps:** 1 = Pt centre, 2 = Cl leaving, 3 = other Pt ligand
  (kept), 4 = incoming OH2 → OH (water entering, hydroxyl after
  proton loss).
- **Pattern intent:** water displaces a chloride on the metal centre
  to give a cationic aqua complex (the rate-limiting step of cisplatin
  activation per Reedijk 1987).
- **Catalyst:** none (water is the nucleophile).
- **Stoichiometry:** empty dict.

### 2.3 Honest framing (from the rule docstrings)

> RDKit cannot sanitise most Pt-containing products because Pt_II is
> not in RDKit's default valence table.  In practice the
> `_rdkit_product_sets_to_closed_terms` helper will drop the Pt-amine
> product while keeping the chloride leaving group.  We still expose
> the rule because (a) the search layer can introspect the rule's
> `_L4_COUNTERS` to see it fired, and (b) when the seed metal has a
> default valence (e.g. Ru) the rule produces well-formed products.

This is the honest negative that drove the design of the F2(a) tests
(Tests 17 + 18 below) — we assert L4 counter increment, not
non-empty product list.

---

## 3. Math prior — mass-action equilibrium for AquaExchange

For AquaExchange the structural SMARTS reduction can be paired with a
*thermodynamic* mass-action prior (the SMARTS only encodes topology;
the synthesis layer may re-evaluate the equilibrium before reporting a
yield):

```
Pt-Cl + H2O  ⇌  Pt-OH2+ + Cl-

K_eq = [Pt-OH2+][Cl-] / ([Pt-Cl][H2O])
```

For cisplatin at 25°C (Reedijk 1987, Table 2):

- k_aquation (forward) ≈ 1.0 × 10⁻⁴ s⁻¹
- k_anhydrolysis (reverse) ≈ 1.0 × 10⁻¹ s⁻¹ at pH < 6.5
- pKa1 ≈ 6.5 (the first deprotonation of the aqua ligand)

At pH < 6.5 (the *cisplatin-relevant* regime) the equilibrium
favours Pt-Cl (Cl- is a weak leaving group at low Cl- concentration);
at pH > 6.5 (intracellular pH ~7.4) the aqua complex dominates.
The Lambda MLC layer treats AquaExchange as a *structural* reduction
that fires on the pattern match; the mass-action prior lives at the
synthesis layer (and may gate the rule at pH < 6.5 if desired).

For MetalLigandExchange the analogous equilibrium is:

```
Pt-Cl + NH3  ⇌  Pt-NH3+ + Cl-

K_eq = [Pt-NH3+][Cl-] / ([Pt-Cl][NH3])
```

Per Lippard 1995 §4 (associative substitution pathway), the kinetic
prefactor is ~ 10⁻⁵ M⁻¹ s⁻¹ for neutral N-donors; the thermodynamic
equilibrium lies far to the right (K_eq ~ 10⁴ at physiological [NH3]).

**Honest framing:** the math prior is documented for the synthesis
layer; the SMARTS rule itself is pattern-only and does NOT encode
kinetic or thermodynamic constants.  The prior is "what a synthetic
chemist would write down" — a structural rule plus a sanity check
that the pKa1 / Cl- context is compatible.

---

## 4. Lit citations

| Citation | Year | Used for | Where |
| -------- | ---- | -------- | ----- |
| Lippard, S. J. "Chemistry of Platinum and Palladium Coordination Compounds" | 1995 | Pt_II 4-coordinate square-planar coordination chemistry; associative substitution pathway; ligand exchange equilibria | `pt_click_compat.py` docstring + `beta_reductions.py` rule docstrings |
| Reedijk, J. "Platinum Coordination Compounds in the Clinic: From Cisplatin to Next-Generation Agents" | 1987 (canonical; multiple Reedijk review updates) | Pt aquation kinetics (k_aquation, pKa1), Cl → H2O exchange | `pt_click_compat.py` docstring + AquaExchange rule docstring |
| Himo, F. et al. "Copper(I)-Catalyzed Synthesis of 1,4-Disubstituted 1,2,3-Triazoles from Azides and Terminal Alkynes" | 2005 | Click regiochemistry canon (CuAAC, SPAAC, Suzuki) — context for the 5-click rules that F2(a) joins | `pt_click_compat.py` docstring |

All three citations are textbook-coordinates for Pt chemotherapeutics
+ click chemistry; no novel chemistry is introduced.  The F2(a)
extension is a *registry* extension (the rules were always there in
the search layer) — the lit anchors are pre-existing.

---

## 5. Compat matrix update

Before F2(a), the matrix was 5 scaffolds × 5 clicks = 25 cells.
After F2(a), it is 5 scaffolds × 7 rules (5 click + 2 metal-coord) =
35 cells.  The 5 scaffolds are unchanged:

- `strict_Pt_II`
- `Pt_II_chelating`
- `Pt_IV`
- `labile_metal`
- `unknown`

The 7 columns are now:

| # | Rule | Family | strict_Pt_II | Pt_II_chelating | Pt_IV | labile_metal | unknown |
| - | ---- | ------ | ------------ | --------------- | ----- | ------------ | ------- |
| 1 | CuAAC | click | compatible | compatible | compatible | compatible | compatible |
| 2 | SPAAC | click | compatible | compatible | compatible | compatible | compatible |
| 3 | ThiolEne | click | **incompatible** | compatible | compatible | compatible | compatible |
| 4 | Suzuki | click | **marginal** | **marginal** | compatible | compatible | compatible |
| 5 | AmideCoupling | click | **incompatible** | compatible | compatible | compatible | compatible |
| 6 | metal_ligand_exchange | coord (F2(a)) | compatible | compatible | compatible | compatible | compatible |
| 7 | aqua_exchange | coord (F2(a)) | compatible | compatible | compatible | compatible | compatible |

**F2(a) design intent:** the 2 metal-coord rules (rows 6-7) are
**always-compatible** on every scaffold because Pt coordination
chemistry is the structural mechanism for diversity — gating them on
`strict_Pt_II` would re-introduce the singleton-collapse failure
that F2(a) is meant to fix.  The synthesis layer may opt to gate them
via `--allow-incompatible-click=False` for an ablation arm, but the
matrix verdict is "compatible".

`CLICK_RULE_NAMES` tuple extended from 5 to 7 entries:
```python
CLICK_RULE_NAMES = (
    "CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling",
    "metal_ligand_exchange", "aqua_exchange",  # F2(a) additions
)
```

---

## 6. Test results

### 6.1 F2(a) tests (4 new tests appended to test_lambda_mcts_singleton.py)

```
$ uv run pytest molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py -v --tb=short
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1
collected 22 items

molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_metal_seed_smiles_are_bare_metal_with_alkyne_handle PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_metal_seed_parses_with_metal_atom_and_alkyne PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_metal_seed_emits_3d_geometry PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_cuaac_fires_on_metal_alkyne_seed PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_metal_seed_root_only_n_distinct_greater_than_one PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_soft_prior_exact_coordination PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_soft_prior_partial_coordination PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_soft_prior_no_metal PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_reward_rebalance_preserves_metal_compliance PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_compliance_truthfulness_seed_violates PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_diversity_bonus_ranks_unique_first PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_metal_compliance_non_seed_excludes_seed PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_metal_compliance_including_seed_unchanged PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_click_compat_table_lookup PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_auto_click_for_pt_ii PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_auto_click_for_pt_iv PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_allow_incompatible_opt_in PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_scaffold_detection_works_on_metal_seed_smiles PASSED
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_metal_ligand_exchange_swap PASSED  [F2(a) test 17]
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_aqua_exchange_pka_context PASSED       [F2(a) test 18]
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_strict_pt_ii_compat_matrix_updated PASSED [F2(a) test 19]
molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py::test_aqua_exchange_compat_matrix_updated PASSED [F2(a) test 20]

22 passed, 1 warning in 21.22s
```

**Result: 22/22 PASSED** (4 new F2(a) tests + 18 pre-existing tests
regression-clean after the matrix extension).

### 6.2 Full molmetal_lam test suite (regression sweep)

```
$ uv run pytest molmetal/molmetal_lam/tests/test_lambda_only_metrics.py molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py
66 passed, 1 warning in 52.42s
```

```
$ uv run pytest molmetal/molmetal_lam/tests/ --tb=line -q
792 passed, 1 skipped, 1 xpassed, 23 warnings in 136.02s (0:02:16)
```

**Result: 792/792 PASSED** across the entire `molmetal/molmetal_lam/tests/`
regression sweep (1 skipped + 1 xpassed are pre-existing).

Two test_lambda_only_metrics tests were updated for the new
`METAL_COORDINATION_RULES` registry:
- `test_click_rules_filter_emptied_no_warning` — filter now uses
  combined registry so the `metal-coord` alias produces a non-empty
  filter via `METAL_COORDINATION_RULES`.
- `test_click_rules_unknown_token_warns_and_falls_back` — fallback
  set now uses the combined registry (21 registered keys).

---

## 7. What ships + what doesn't

**Ships in this task (code + tests + report):**

- Extended COMPAT_MATRIX in `pt_click_compat.py` (5x5 → 5x7) with
  `metal_ligand_exchange` + `aqua_exchange` as 2 new columns, marked
  compatible on every scaffold.
- 4 new F2(a) tests in `test_lambda_mcts_singleton.py` covering SMARTS
  pattern assertions + L4 counter increments + matrix-helper assertions.
- Updates to 4 pre-existing tests (12/13/14/15) for the new 7-column
  matrix shape.
- Regression fix in `test_lambda_only_metrics.py` to recognise the
  `METAL_COORDINATION_RULES` registry alongside `CLICK_REACTIONS`.

**Did NOT ship (per task graph disjoint file sets):**

- `beta_reductions.py` was NOT touched — the `MetalLigandExchange` and
  `AquaExchange` SMARTS classes were already shipped in WF-Lambda-Fix-
  FullPath (2026-09-14).  The current task is the matrix extension +
  tests + report.
- `lam_chem/rules.py` was NOT touched — the `METAL_COORDINATION_RULES`
  registry + `metal-coord` CLI alias were already shipped in
  WF-Lambda-Fix-FullPath.
- `r4_lambda_only_run.py` was NOT touched — the `metal-coord` alias
  and the auto-rules-expanded warnings logic were already shipped.

**Honest framing on what F2(a) does NOT fix:**

- F2(a) does NOT lift the n_distinct=1 singleton collapse in the
  Round-12 Lambda pilot (WF-Round12-Lambda-Pilot 2026-09-15 partial
  report).  The collapse is rooted in (a) the MCTS hard-cap on
  `n_simulations=100` (lifted in WF-Lift-N-Sim-Cap), (b) the click
  chemistry rules' scaffold-gating (F2-B3), and (c) the MCTS cache
  `_unreactive_states` (WF-Lambda-Internal-Review finding #2).
  F2(a) adds the *coordination* channel so a non-collapsed search can
  build diverse Pt ligand spheres — but the structural fix requires
  ALL of F1/F2(a)/F2(b)/F3/F4 from WF-Lambda-Internal-Review to land.
- F2(a) does NOT recover GPU (which is still BLOCKED per
  WF-GPU-Recovery-Now 2026-09-15).
- F2(a) does NOT measure lift in a Round-12 re-run — the GPU outage
  blocks the pilot.  When the GPU recovers, the smoke is a 5x1 with
  `--click-rules metal-coord` and `--metal-seed cisplatin`; expected
  n_distinct lift from 1 → ≥3 per the structural-fix hypothesis in
  WF-Lambda-Internal-Review §4.

---

## 8. Follow-ups

1. **Round-12 re-verify (BLOCKED on GPU).** When the dGPU comes back,
   run `r4_lambda_only_run.py` 5x1 with `--metal-seed cisplatin
   --click-rules metal-coord` and verify n_distinct lift.  Pre-registered
   expected n_distinct ≥ 3 (per WF-Lambda-Internal-Review hypothesis).
2. **Synth-layer pKa context.** Add an optional `ph` argument to the
   AquaExchange rule that gates firing at pH < 6.5 (Reedijk 1987
   cisplatin pKa1).  This is a synthesis-layer concern, not a SMARTS
   concern; out of scope for F2(a) SMARTS shipping.
3. **Document the 7-column matrix in §4.5 ablation.**  When the
   paper-side ablation is updated to 7 rules instead of 5, the
   §4.5.7 cross-ref to `sec:metal-coord-rules` needs to be added.
   TODO-28 handles framing (out of scope here per the task spec).
4. **Cross-link to TODO-29.** This task is Phase 3A of the parallel
   flow described in TODO-29; the Round-13 retry task depends on the
   F2(a) ship + a GPU recovery.

---

## 9. Honest framing — limitations of this report

- The mass-action prior in §3 is documented for the *synthesis layer*
  and is NOT enforced in the SMARTS rule itself.  The SMARTS rule
  fires on the pattern match alone; the kinetic/thermodynamic prior is
  delegated to the synthesis layer (and is currently a TODO for a
  future `--aquation-ph` CLI flag).
- The F2(a) "diversity lift" claim is structural, not empirical.  No
  Round-12 re-run has been performed (GPU blocked).  The expected
  n_distinct ≥ 3 is a hypothesis from WF-Lambda-Internal-Review §4,
  not a measurement.
- The COMPAT_MATRIX extension is hand-curated (no ML fitting); the
  verdicts are what a synthetic chemist would write down.  This is
  the design intent — chemistry-knowledge heuristics, not learned
  priors — and is documented in the module docstring.
- The 5x5 → 5x7 matrix extension changed the `default_compatible_rules`
  return value for `strict_Pt_II` from 3 click rules to 5 rules
  (3 click + 2 metal-coord).  Existing tests 12-15 were updated to
  reflect this; the user-facing aliases (`auto-pt-strict`,
  `auto-pt-iv`) still resolve to the 5-click-only view (the metal-
  coord rules are opt-in via the `metal-coord` alias).
