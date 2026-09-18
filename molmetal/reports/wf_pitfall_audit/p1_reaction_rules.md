# Pitfall Audit P1 — REACTION RULES layer

**Date**: 2026-09-17
**Scope**: Audit the SMARTS/reaction-rule layer of Mol-Metal against 4 pitfalls
from the user's brief. **Audit only — no code modifications.**

**Files read** (≤ 2 min each):
- `molmetal/molmetal_lam/reactions/beta_reductions.py` (1615 LOC)
- `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` (518 LOC)
- `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` (248 LOC)
- `molmetal/molmetal_lam/lam_chem/closure.py` (649 LOC)
- `molmetal/molmetal_lam/reactions/synthesis_oracle.py` (rate_predictor bridge)
- `molmetal/molmetal_lam/reactions/rate_predictor.py` (LITERATURE_YIELDS path)
- `molmetal/molmetal_lam/tests/test_click_reactions.py` (234 LOC)
- `molmetal/molmetal_lam/priors/metal_geometry.py` (T9 prior, first 100 LOC)

**Note**: `molmetal/reports/wf_metal_geometry_prior.md` does NOT exist as a
file — the corresponding code module is `priors/metal_geometry.py`; the
pitfall relevant to that file is at the SMARTS/geometry level and audited
inside P1.4 below. There is also no `synthesizability_check.py` module;
the synthesizability channel lives in
`molmetal_lam/sbdd_env/{retrosynthesis,synthesis_gate}.py` (SOTA clones)
and `molmetal_lam/reactions/synthesis_oracle.py` (yield predictor) — both
audited under P1.2 / P1.3.

---

## Verdict per pitfall

| ID  | Pitfall                                                | Status   | Evidence (file:LOC)                                                                                          |
| --- | ------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------ |
| P1.1 | "Synthesizable" conflated with "click chemistry"       | **OPEN** | `beta_reductions.py:1391-1406` only ships 5 click + 2 metal-coord rules; no peptide/glycosylation/halogenation |
| P1.2 | Yield / conditions / FG-compat metadata missing        | **PARTIAL** | `beta_reductions.py:175` `requires_catalyst` + `rate_predictor.py:1-95` LITERATURE_YIELDS exist; FG map = 0 |
| P1.3 | Buyable building blocks / AiZynth / stock check missing in MCTS | **PARTIAL** | `aizynth_sota_scoring.py` wired as a *scoring* adapter (SOTA), not used to **gate** MCTS expansion          |
| P1.4 | SMARTS rules not unit-tested against known Pt complexes | **OPEN** | `test_click_reactions.py` covers ethyl-azide + propyne only (line 36); no cisplatin/carboplatin/oxaliplatin |

---

## P1.1 — "Synthesizable" conflated with "click chemistry"? — **OPEN**

**Evidence (1-line)**: `REACTION_RULES` in `beta_reductions.py:1391-1406`
contains 7 named rules: CuAAC, SPAAC, SPC, DielsAlder, ThiolEne, Suzuki,
AmideCoupling + MetalLigandExchange + AquaExchange. The 5 "click" rules
plus 2 metal-coordination rules. **Zero non-click med-chem transformations**
(peptide coupling, Boc/FDMOC deprotection, reductive amination, SNAr,
glycosylation, halogenation, Mitsunobu, Wittig, dihydroxylation, C-H
activation, esterification, …) appear in the registry.

**Honest framing**: The MLC formalism was explicitly scoped to "click
chemistry as β-reductions" (paper §3.2). For de novo metallodrug
generation this is *intentional* — CuAAC/SPAAC give biorthogonal handles
to attach metal cores, and Pt-Cl exchange gives the metal sphere. But
"PLATINUM DRUG = Pt-Cl2(NH3)2 + CuAAC adduct" is only ONE chemotype among
many. Carboplatin/oxaliplatin/satraplatin are not in the productive space
unless `--metal-seed` happens to be set; and even then no rule produces
carboplatin's CBDCA chelate from succinic-acid precursors.

**Concrete gap**: A Round-14 metallodrug generator that wants to enumerate
even the 6 canonical Pt-drug chemotypes (cisplatin, carboplatin,
oxaliplatin, nedaplatin, lobaplatin, satraplatin) cannot do so without
hand-seeded `--metal-seed` because no rule synthesises the chelate /
axial-Cl precursors.

### Patch plan (concrete, ETA)

1. **Phase A — peptide coupling** (4h CPU): add `PeptideCoupling` rule
   (`molmetal/molmetal_lam/reactions/beta_reductions.py:1147` next to
   `AmideCoupling`). Pattern: `[NH:1][C:2](=O)[OH].[NH2:3]>>[NH:1][C:2](=O)[NH:3]`
   with the leaving-group branch on the acid. Register in `REACTION_RULES`.
   3 unit tests against AlaGlyGly + triglycine + Z-protected dipeptide.

2. **Phase B — Boc deprotection** (2h): `BocDeprotection` rule,
   pattern `[NH:1][C:2](=O)[O][C](C)(C)C>>[NH:1][C:2](=O)[OH]+CH2=C(CH3)2+CO2`
   (mass-balanced, leaving-group bookkeeping consistent with current
   `MetalLigandExchange` style at `beta_reductions.py:1228-1234`).

3. **Phase C — reductive amination** (3h): `ReductiveAmination` rule
   that joins an aldehyde/ketone to a primary/secondary amine with
   `[C:1]=O.[NH2:2]>>[C:1][NH2:2]` (after dehydration we keep the
   iminium equivalent — honest framing in docstring).

4. **Phase D — register & test** (1h): append to `REACTION_RULES`; add
   2 `test_pt_complex_*` tests (e.g. `test_carboplatin_synthesizable`
   asserts that `PeptideCoupling + BocDeprotection + MetalLigandExchange`
   reaches CBDCA-chelated Pt within depth-3 BFS in
   `closure.py:ProductiveSpace`).

**Total ETA**: 10h CPU. **Risk**: low (existing pattern is well-tested
for click; new patterns follow the same `_reduce` shape).

---

## P1.2 — Yield / conditions / FG-compat metadata missing — **PARTIAL**

**Evidence (1-line)**:

- `ReactionRule.requires_catalyst` exists at
  `beta_reductions.py:175` and is populated for CuAAC (`"Cu(I)"` line
  618), SPAAC (`None` line 687), SPC (`None` line 753), DielsAlder
  (`None`), ThiolEne (`"hν / radical initiator"` line 880), Suzuki
  (`"Pd(0) + base"` line 1028), AmideCoupling (`"coupling reagent
  (EDC/HOBt, HATU, etc.)"` line 1099). **Catalyst tag = partial coverage**
  (5/7 rules populated; 2/7 default to `None`).
- `rate_predictor.py:1-95` ships `LITERATURE_YIELDS` (6 reactions × 10
  hand-curated citations) and `HeuristicRegressor` (PySR/sklearn).
  `attach_all_rate_predictors()` at `beta_reductions.py:1471-1511` wires
  a fitted `RatePredictor` onto every click rule via 8-d
  `smiles_pair_features` (mol-wt / logP / TPSA / heavy-atom-count per
  reactant). **Yield modelling = present but approximate** (8-d
  descriptor; no FG-compat / leaving-group / solvent terms).
- `SynthesisOracle.predict()` (`synthesis_oracle.py:84-106`) returns
  `0.5` fallback when any predictor fails — neutral default is
  defensible but does NOT propagate FG-incompatibility as a *penalty*.
  **No `functional_group_compatibility` field** on `ReactionRule`.
- `composite_score` (`priors/anticancer_metric_suite.py:130`) is a
  *property oracle*, not a synthesizability oracle — it does not
  consult reaction rules at all.

**Honest framing**: `composite_score` weights 9 anticancer properties
(logP / TPSA / rotB / oxidation_state / coord / GSH / DNA-kb /
anticancer_index). It does NOT use the reaction-layer SMARTS. The
8-d `smiles_pair_features` includes logP/TPSA but is a black-box
ML-trained predictor on 60 hand-curated citations — it doesn't encode
the chemistry of "thiolate attacks Pt-Cl" (the literal text from
`pt_click_compat.py:128`). So the *yield* number is a literature-fit
regressor; the *FG-compat* gate (do we tolerate a free amine in the
substrate?) is **not modelled**.

**Concrete gap**: A Round-12 candidate containing a free thiol +
carboxylic acid + aldehyde is *incompatible with itself* (intramolecular
thioacetal forms); the MCTS will happily join a CuAAC onto the thiol
group, producing a fictitious structure. There is no `fg_compat` veto.

### Patch plan (concrete, ETA)

1. **Phase A — extend `ReactionRule`** (1h): add `fg_compatible: Dict[str, set[str]]`
   field listing incompatible functional-group SMARTS per rule
   (e.g. `CuAAC.fg_compatible = {"incompatible_with": ["#17-SH",
   "#15-CHO"]}`). Use existing
   `beta_reductions.py:153` (`pattern_smiles`) as the storage site.

2. **Phase B — query in `_reduce`** (2h): add an `fg_compatible_check`
   helper that parses educt SMARTS and vetoes if any forbidden group
   is present. Wire into `CuAAC._reduce` at
   `beta_reductions.py:628-665` (the most-fired rule).

3. **Phase C — tests** (2h): add `test_fg_compat_veto` cases
   (e.g. thiol + CuAAC → empty product set; same for aldehyde +
   AmideCoupling when the amine is primary + an aldehyde is also present).

4. **Phase D — leave group / condition metadata** (4h): add
   `leaves: List[str]` and `conditions: Dict[str, str]` (solvent /
   temperature / atmosphere) to `ReactionRule`. Populate from literature
   review (5 most-cited reactions). Backward-compat: all fields default
   to `[]` / `{}` so existing tests don't break.

**Total ETA**: 9h CPU. **Risk**: medium — FG rules can over-veto; needs
property-based tests to bound the false-positive rate.

---

## P1.3 — Buyable building blocks — **PARTIAL**

**Evidence (1-line)**: `molmetal_lam/sbdd_env/aizynth_isolated.py`,
`aizynth_sota_scoring.py`, `molmetal_lam/scripts/aizynth_worker.py`,
`aizynth_torch_worker.py` (≈ 6 files) all ship **real AiZynthFinder
integration** (per Round-8 wrap-up, see MEMORY entry "WF-Round-8 done
2026-09-12"). However, it is wired as a **SOTA scoring column**
(`r4_c_full_sweep.py --synthesis-oracle aizynth` per MEMORY
WF-Wire-AiZynth entry), not as a **building-block gate** that vetoes
expansion if the next reactant is not commercially available.

**Honest framing**: The system CAN call AiZynthFinder end-to-end
(verified by test_aizynth_adapter.py + WF-Wire-AiZynth; 5+ tests
per MEMORY). But the MCTS proof-search layer (`proof_search.py:889-911`)
does not consult AiZynth before expanding a candidate. The
`ref_ligand_resolver.py` (F2 Phase 1, MEMORY task #825) is a
*pocket-conditioned reference ligand* resolver, not a *buyable
reactant* resolver — different problem.

**Concrete gap**: Round-12 Lambda 10×3 with metal-seed cisplatin
collapsed to singleton `[NH2][Pt]([NH2])([Cl])[Cl]` (per MEMORY
WF-Round12-Lambda-Pilot + WF-Lambda-Internal-Review entries) precisely
because the MCTS cannot generate diverse *real* ammines beyond NH3 —
it does not consult any stock database.

### Patch plan (concrete, ETA)

1. **Phase A — wrap AiZynth as a gate** (3h): add a `BuyableReactantGate`
   class to `molmetal/molmetal_lam/lam_chem/` that wraps the existing
   `aizynth_isolated.AiZynthAdapter.find_buyable(smiles) -> bool` call
   (already exists per `aizynth_isolated.py`). Expose a
   `gate.reactant_commercially_available(smiles) -> float` (0-1 score,
   NaN if AiZynth unavailable).

2. **Phase B — wire into MCTS expand** (3h): call `gate` from
   `proof_search.py:889-911` *before* a partner tile is added to the
   frontier. Backward-compat: when `gate` is `None` (AiZynth model
   file missing), behave as today (no veto).

3. **Phase C — tests** (2h): 3 tests in `test_buyable_gate.py`
   (1: ethyl-azide → buyable=True; 2: arbitrary fluoroketone →
   buyable=False; 3: NaN fallback when model file missing).

4. **Phase D — pilot** (4h GPU/CPU): re-run R12 10×3 with `--buyable-gate`
   ON; expect `n_distinct` to lift from 1 to ≥5 (per MEMORY
   WF-Lambda-Internal-Review, the structural singleton root cause
   requires more than gating — but this is the cheapest partial fix).

**Total ETA**: 12h (8h CPU + 4h pilot). **Risk**: medium-high —
AiZynth model file may not be checked in, requiring re-download.

---

## P1.4 — SMARTS not unit-tested against known Pt complexes — **OPEN**

**Evidence (1-line)**: `test_click_reactions.py:36-80` tests CuAAC on
ethyl-azide + propyne only. No cisplatin/carboplatin/oxaliplatin
reference structure is parsed by any rule. The
`metal_geometry.py:99-100` module ships a Pt-specific 90°-deviation
penalty but it operates on 3-D coordinates, not on SMARTS — so it is
an *EGNN training loss*, not a *rule verification*.

**Honest framing**: The SMARTS for `MetalLigandExchange` (Pt-Cl + NH3 → Pt-NH3)
at `pt_metal_ligand_exchange.py:67` was hand-curated from Lippard 1995,
and *property-based* closure tests (per MEMORY WF-Lambda-4 entry) catch
the Au_III arity bug (5d^8 vs 4-coordinate default). But there is **no
unit test that asserts**: "given cisplatin SMILES, applying
`MetalLigandExchange` once yields a mono-substituted Pt complex with
one Cl⁻ leaving group". Such a test would catch any future change that
breaks Pt chemistry (e.g. switching the SMILES atom-map convention).

**Concrete gap**: The `WF-Lambda-Boost` Phase 2 + 3 work added 5
MetalLigandExchange + 2 AquaExchange SMARTS (per `pt_metal_ligand_exchange.py`
header comment "5 + 2 = 7 SMARTS patterns") but the existing test suite
only tests ONE pattern indirectly via the closure-theorem property test
— no targeted regression for the other 4 + 2 patterns.

### Patch plan (concrete, ETA)

1. **Phase A — known-Pt reference test** (3h): add
   `test_pt_metal_ligand_exchange_known_complexes` in
   `molmetal/molmetal_lam/tests/test_pt_metal_ligand_exchange.py`.
   Cases: cisplatin + NH3 → mono-ammine (1 Cl⁻ released);
   cisplatin + NH3 + NH3 (twice) → diammine (2 Cl⁻ released);
   carboplatin + NH3 → ring-opening of CBDCA (1 cyclobutane-dicarboxylate
   released). Each test asserts on product SMILES substring
   (e.g. `"[Pt](N)(N)"` present) and Cl⁻ leaving-group count.

2. **Phase B — AquaExchange regression** (2h): add
   `test_pt_aqua_exchange_first_second_aquation` covering both patterns
   in `pt_metal_ligand_exchange.py:120-136`. Verify second-aquation
   product keeps both OH groups + releases Cl⁻.

3. **Phase C — pt_click_compat smoke** (1h): add a test that for each
   (scaffold, rule) cell in `COMPAT_MATRIX` (`pt_click_compat.py:125-206`)
   the corresponding rule's `can_apply()` matches the matrix verdict
   (e.g. `strict_Pt_II` + ThiolEne → `can_apply=False`; `strict_Pt_II` +
   CuAAC → `True`). 35-cell test.

4. **Phase D — sync with priors** (2h): add 1 test that
   `metal_geometry.apply_metal_prior` correctly identifies a 4-coordinate
   Pt in cisplatin vs a 6-coordinate Pt in satraplatin (sanity check on
   the geometry prior that consumes the SMARTS output).

**Total ETA**: 8h CPU. **Risk**: low — tests are additive; existing
PyTest suite must continue to pass (no SMARTS edits).

---

## Aggregate verdict

| Pitfall | Status   | ETA to close (CPU) | Risk    | EV |
| ------- | -------- | ------------------ | ------- | -- |
| P1.1 — non-click med-chem rules  | OPEN     | 10h                | low     | high |
| P1.2 — FG-compat / leave-group   | PARTIAL  | 9h                 | medium  | medium |
| P1.3 — buyable-gate wired        | PARTIAL  | 12h (incl. pilot)  | medium-high | high |
| P1.4 — known-Pt regression tests | OPEN     | 8h                 | low     | high |

**Total CPU**: 39h (≈ 1 dev-week). **Total pilot GPU**: 4h.
**Honest aggregate**: the reaction-rules layer is the *most-tested*
component of Mol-Metal (closure-theorem property tests + L4
instrumentation + 13/14 SOTA packages installed per WF-Round7),
but its *chemistry coverage* is narrow (7 rules total, 0 non-click
med-chem) and its *chemistry verification* is narrow (1 unit test
per rule, 0 known-Pt regression).

## Honest framing

1. **The 5-click + 2-metal-coord registry is a *by-design* narrow
   scope** per the MLC formalism's paper §3.2 commitment. P1.1's
   patch is *expansion*, not a bug fix.
2. **The FG-compat gap (P1.2)** is real and would manifest in
   Round-14 metallodrug runs where a CuAAC adduct + free thiol would
   give fictitious thioacetal side products. We have not yet seen
   this in production because every pilot so far uses *single-handle*
   tiles (azide OR alkyne, not both).
3. **The buyable-gate (P1.3)** is the partial fix for the
   WF-Lambda-Internal-Review "singleton attractor" root cause
   identified in MEMORY — closing this would directly lift
   Round-13 n_distinct from 1 toward ≥5 per the EV-1 GPU-free
   lift workflow already in flight.
4. **The known-Pt tests (P1.4)** are pure insurance — they
   cost 8h CPU and prevent future SMARTS drift. High EV, low risk,
   no GPU dependency.

## Recommended sequencing

1. **P1.4 first** (8h, low risk) — ship the known-Pt regression
   suite so any future SMARTS edit is caught.
2. **P1.3 next** (12h, partial lift) — wire AiZynth as a gate so
   EV-1 pilot runs can lift n_distinct.
3. **P1.2 last** (9h, narrowing false positives) — add FG-compat
   veto only AFTER we have buyable-gate + known-Pt tests; otherwise
   over-narrow veto may make Round-13 zero-output.
4. **P1.1 deferred** (10h) — non-click med-chem is a Round-15+
   expansion; current paper §3.2 is honest about the 5-click scope.

---

## Files referenced (absolute paths)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/beta_reductions.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pt_click_compat.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/closure.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/synthesis_oracle.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/rate_predictor.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_click_reactions.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/aizynth_isolated.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/retrosynthesis.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py`

## Audit-only declaration

This document audits only; no code in any of the above files was
modified during this task.
