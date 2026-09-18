# Pitfall Audit P5 — EXPERIMENTAL VALIDATION layer

**Date**: 2026-09-17
**Scope**: Audit the *experimental-validation* layer of Mol-Metal against 3 pitfalls from the user's brief:
- **P5.1**: Reagent availability, steric hindrance, purification difficulty, scale-up (does the paper claim wet-lab validation?)
- **P5.2**: Closed-loop wet feedback (is there a plan for periodic wet-lab recalibration of the reward model?)
- **P5.3**: Narrow start, extensible interface (do we start with 5 click rules but have extensibility for adding reactions?)

**Audit only — no code modifications.**

**Files read** (≤ 2 min each):
- `TODO/decisions.md` — user-decision history (D1-D7)
- `TODO/risks.md` — R1-R8 risk register
- `TODO/pending/19_user_decisions.md` — 6 pending user-gated decisions incl. cite-only SOTA column
- `molmetal/reports/wf_pic50_margin_sweep_verdict.md` — AI-trained pIC50 (margin=0.5 best, honest negative pearson_r=0.195 ≤ ridge 0.572)
- `molmetal/reports/reinvent4_learned_smoke/README.md` + `report.json` — REINVENT4 official CLI learned model PASS on CUDA + ROCm
- `molmetal/molmetal_lam/reactions/beta_reductions.py:1387-1406, 175` — `REACTION_RULES` registry (5 click + 2 metal-coord = 7 named rules, plus `_register_synthemol_rules` lazy-loading 13 SyntheMol REAL reactions)
- `molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py` — SyntheMol 13 REAL reaction adapter (`from_syntemol_to_lambda`, `load_real_reactions`)
- `molmetal/molmetal_lam/reactions/rate_predictor.py:1-95` — `LITERATURE_YIELDS` (6 reactions × 10 citations) + `HeuristicRegressor` 8-d smiles_pair_features
- `molmetal/reports/wf_pitfall_audit/p1_reaction_rules.md` — upstream P1 audit (defines `REACTION_RULES` 7-rule scope, FG-compat gap)
- `molmetal/reports/wf_pitfall_audit/p4_eval_novelty.md` — upstream P4 audit (wet-lab IC50 listed as out-of-scope metric F in TargetDiff gap)
- `paper/section_06_limitations.tex:19-22` — explicit admission: "wet-lab cytotoxicity are not measured by the present pipeline and require dedicated predictors or experiments"
- `paper/section_07_future_work.tex:14` — explicit admission: "ultimately wet-lab validation are required before therapeutic claims can be made"
- `paper/section_02_related_work.tex:22` — claim of "exhaustive experimental enumeration is impossible" (paraphrased)
- `molmetal/reports/reinvent4_learned_smoke/rocm_traced/report.json` — REINVENT4 ROCm run on gfx1101 (PASS)
- `molmetal/reports/anticancer_vs_general_metrics_survey.md:352` — explicit: "Less emphasis on wet-lab validation"

---

## Verdict per pitfall

| ID  | Pitfall                                                                  | Status    | Evidence (file:LOC)                                                                                                                                            |
| --- | ------------------------------------------------------------------------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P5.1 | "Synthesizable" / reagent / steric / purification / scale-up — wet-lab claim | **AVOIDED** | `section_06_limitations.tex:19-22` + `section_07_future_work.tex:14` explicit two-paper-claim of wet-lab-only-out-of-scope; `anticancer_vs_general_metrics_survey.md:352` Less-emphasis-on-wet-lab framing; zero molecules claimed synthesised-or-tested.    |
| P5.2 | Closed-loop wet feedback / reward model recalibration              | **OPEN**   | `proof_search.py:614-706` RewardAggregator 14 channels (Vina / SA / QED / PB / pIC50 / PlatinAI / RxnFlow) — **all 14 are in-silico**; no wet-lab feedback channel; no `recalibrate_from_assay()` hook; no `wetlab_hit_rate` column; no PfWetLab claim queue.    |
| P5.3 | Narrow start (5 rules) but extensible interface                  | **PARTIAL**| `REACTION_RULES` registry (5 click + 2 metal-coord = 7) + `_register_synthemol_rules()` lazy-loads **13 SyntheMol REAL reactions** (`sbdd_env/syntemol_reactions.py:67-149`) via `load_real_reactions` → `from_syntemol_to_lambda`. Wrappers expose `name`, `can_apply`, `fire` — drop-in to existing callables. **Gap: GUI/CI gate to vet new rule before it joins the search expansion set; SyntheMol rules opt-in, not auto-registered.**     |

**1 AVOIDED, 1 PARTIAL, 1 OPEN.** The paper is **honest** about
wet-lab scope (P5.1) and ships an extensible reaction registry (P5.3),
but has **no wet-lab feedback loop** (P5.2) — the reward model is
calibrated from in-silico oracles only and cannot re-train from a
hypothetical NCI-60 / MetalCytoToxDB / in-house panel of wet-lab
hits.

---

## P5.1 — Reagent availability / steric / purification / scale-up / wet-lab claim — **AVOIDED**

### What we ship

**Honest textual framing**:

| Location | Exact claim |
|---|---|
| `paper/section_06_limitations.tex:19-22` | "the anticancer metric suite provides chemical proxies. hERG, CYP450, protein binding, DNA binding, reduction potential, and wet-lab **cytotoxicity are not measured by the present pipeline and require dedicated predictors or experiments.**" |
| `paper/section_07_future_work.tex:14` | "Broader metal coverage, explicit DNA-fragment docking, redox and aquation predictors, and **ultimately wet-lab validation are required before therapeutic claims can be made.**" |
| `paper/section_02_related_work.tex:22` | "This design is especially useful for sparse metal-complex data, where **exhaustive experimental enumeration is impossible**" |
| `molmetal/reports/anticancer_vs_general_metrics_survey.md:352` | "Less emphasis on wet-lab validation" (re: NCI-60 / MetalCytoToxDB as gold standard but inaccessible) |
| `molmetal/reports/anticancer_vs_general_metrics_survey.md:437` | "In silico Mol-Metal cannot generate true NCI-60 GI50/TGI data without experimental collaborators. **Any pIC50 predictions should be flagged as computational estimates, not wet-lab measurements.**" |

**No metric claims wet-lab validation**:

- Paper §4 Table 1 reports `pIC50` column with `pearson_r=0.181 ± 0.234` and
  `bound_aware_acc=1.000`. The §4.6 cell explicitly frames it as
  `HeLa48h/dark cohort, n=1451/715/251 censored` and
  `WF-pIC50-Margin-Sweep` verdict (`wf_pic50_margin_sweep_verdict.md`
  line 88-99) calls it `Sub-baseline pearson_r ... well below the
  conditioned ridge baseline (0.572 on the same cohort, same splits)`.
  **It is honest-negative MEASURED-but-not-wet-lab.**
- REINVENT4 channel is
  `r_reinvent4 = SMILES sequence NLL under the prior` (per
  `reinvent4_learned_smoke/README.md:35`). It is a sequence
  likelihood, not activity. Explicit framing: *"This experiment does
  not turn the existing project proxy bridge into a learned
  multiproperty scoring plugin."*
- PlatinAI oracle (`r_platinai`) is **literature-data-derived kNN
  Tanimoto** to a corpus of known Pt complexes. Not wet-lab, but
  literature-curated. Verified by `wf_pivot_followup/F2`
  (per MEMORY `WF-Pivot-Followup MASTER`).

**Yield / conditions / FG-compat metadata partially shipped** (per P1.2
upstream audit):

- `ReactionRule.requires_catalyst` populated for 5/7 named rules
  (`beta_reductions.py:618, 687, 753, 880, 1028, 1099`). Two of seven
  default to `None`. Documents the catalyst for the chemist but does
  NOT model purification or scale-up.
- `rate_predictor.py:1-95` ships `LITERATURE_YIELDS` (6 reactions ×
  10 hand-curated DOI citations) and `HeuristicRegressor` (PySR /
  sklearn fallback). 8-d `smiles_pair_features` (mol-wt / logP / TPSA
  / heavy-atom-count per reactant). **Yield-modelling = approximate,
  purification / scale-up NOT modelled.**

### Why this is AVOIDED (not PARTIAL)

1. **Paper never claims any molecule has been synthesised or
   tested in vitro.** All numeric cells are DESIGN / SEARCHONLY /
   MEASURED-but-not-wet-lab per the `STATUS = [NAME]` markers
   (`DESIGN`, `MEASURED`, `PROJECTED`, `CITEDONLY`, `SEARCHONLY`)
   shipped across paper sections.
2. **Limitations section explicitly names wet-lab as out-of-scope**
   and future work explicitly conditions therapeutic claims on it.
3. **The closest surrogate to wet-lab** (PlatinAI literature kNN
   oracle, NCI-60 columns in supplementary table) is honestly labelled
   as ML-generated / literature-derived / not-wet-lab per
   `paper/supplementary_data_table.tex:142-167`.
5. **"Honest framing per TODO-19"** is recorded in the user's brief;
in `TODO/pending/19_user_decisions.md:46` (cite-only SOTA column) and
`TODO/decisions.md:38-43` (D2 — MMP13 / CA2 / MMP2 evaluation scope)
the pattern is *"label every row as
'not re-run by Mol-Metal; cited from <ref>'"*. Same pattern applies
to wet-lab.

### Honest framing

The pipeline is *not* claiming synthesis capability. It ships:

- **14 in-silico reward channels** (Vina / SA / QED / PB / pIC50 /
  PlatinAI / RxnFlow / 7 more).
- **0 wet-lab channels** — by-design, because the host has no
  chemistry bench.
- **Honest framing**: §6 Limitations and §7 Future Work explicitly
  condition therapeutic claims on future wet-lab validation.

**The biggest risk is silent inflation, not explicit over-claim.**
P5.1 is the *one pitfall where the system is already doing the right
thing* and the audit confirms it.

### Patch plan (none required)

No patch required for P5.1. The honest framing is in place.
Recommended **defensive additions** (no urgency, all cite-only
safeguards):

1. **Add `STATUS` marker `WETLAB_PROPOSED`** for the 5-cell wet-lab
   proposal table (paper §7 future work). Already implied by
   "ultimately wet-lab validation are required before therapeutic
   claims can be made" (line 14) but a tag would let reviewers
   grep for it. Cost: 1 line in `paper/main.aux` + 5 cell updates.
2. **Cite-only `wet_lab_estimate_panel.md`** — list the 3-5 most
   relevant NCI-60 / MetalCytoToxDB experiments a wet-lab
   collaborator would need to run (GI50 at 5 doses on OVCAR-3 /
   A2780 / A549 cell lines, 72h MTT assay per Keepers 1981). Cost:
   2h to draft from the relevant literature.
3. **Add a "no synthesised / no wet-tested" badge** to the §4 Table 1
   caption so reviewers do not mistake the in-silico metrics for
   experimental measurements. Cost: 1 caption line.

None of these are required to claim AVOIDED status. They are
defensive hygiene against reviewer misreading.

---

## P5.2 — Closed-loop wet feedback / reward model recalibration — **OPEN**

### What we ship

**All 14 reward channels are in-silico** (per P3.1 upstream audit):

| Channel | Source | Wet-lab? |
|---|---|---|
| `r_vina` | AutoDock-Vina (kcal/mol, negated) | No — physics-based scoring |
| `r_qvina` | QuickVina 2 (kcal/mol) | No — physics-based scoring |
| `r_sa` | Ertl SA score | No — heuristic graph complexity |
| `r_qed` | QED drug-likeness | No — Bickerton 2012 trained on PubChem |
| `r_vina_proxy` | Cheap Vina regression | No — regression model |
| `r_posebusters` | PB pass-rate | No — chemistry-aware checks |
| `r_pb_valid` | PB outer-gate | No |
| `r_pic50` | D-MPNN retrain (TODO-18) | **No** — it is a neural net trained on HeLa48h literature pIC50; the *training data* is wet-lab but the channel output is not; honest pearson_r=0.181 |
| `r_retro` | Retrosynthesis feasibility | No — graph-pattern matching |
| `r_reinvent4` | REINVENT4 multiproperty scorer | No — sequence NLL / learned proxy |
| `r_synth` | Synthesis-yield oracle | No — literature-cited yields |
| `r_admet` | Lipinski desirability | No — rule-based |
| `r_anticancer_index` | TODO-15 equal-weight | No — heuristic composite |
| `r_metal_geom_soft` | soft Pt-CN prior | No — geometry prior |
| `r_platinai` | PlatinAI kNN Tanimoto | No — literature Pt complexes |
| `r_rxnflow` | RxnFlow template match | No — template matching |
| `r_diversity` (rank-time) | canonical-SMILES uniqueness | No |

**Zero channels** are derived from a wet-lab assay feedback loop.
The closest surrogate is the **PlatinAI oracle** (literature kNN Tanimoto to
known Pt drugs), which is *literature-derived*, not *experiments-in-this-lab*.

**No `recalibrate_from_assay()` hook exists**. A grep for
`recalibrat`, `assay`, `wetlab`, `experimental_feedback` over
`molmetal/` returns 0 hits in the production code. Only `paper/sections/`,
`molmetal/reports/`, and `TODO/` mention wet-lab; the code does not.

### Why this is OPEN (not PARTIAL)

A *closed-loop* wet-feedback architecture has 4 necessary
components. We have **none**:

| Component | Where required | Status |
|---|---|---|
| **Wet-lab assay queue** (`assays.csv` / `wetlab_plate_X.csv`) | input | **MISSING** |
| **Reward channel that consumes assay hits** (`r_wetlab = -mean(GI50)`) | training-time | **MISSING** |
| **`recalibrate_from_assay()`** hook that takes assay hits and refits the reward weights | retraining-time | **MISSING** |
| **Periodic re-train cadence** (e.g. quarterly, gated on ≥N new hits) | scheduling | **MISSING** |

The *closest* design pattern is the **RxnFlow cite-only oracle channel**
(`WF-T24 RxnFlow cite-only oracle channel wire`, MEMORY entry),
which is the architectural placeholder for "learned oracle" — but it
is still in-silico.

### Concrete gap

Mol-Metal today produces a list of candidate Pt complexes with
in-silico metrics. The reward weights are fixed at construction
(`proof_search.py:707-708` `w_vina=1.0, w_sa=1.0` etc). No mechanism
exists for:

> "We tested candidates X₁..X₁₀ in vitro on OVCAR-3 cells;
> GI50(X) values are now known. Re-train MCTC weights to
> increase GI50-prediction accuracy."

Without this, the reward model is **frozen at training time** and
**any in-silico / wet-lab drift is invisible to the generator**.
This is the textbook limitation of *static* reward models in
RL-from-scratch.

### Patch plan (concrete, ETA)

**Tier 1 — Add the *plumbing* without wet-lab execution** (6h CPU):

1. **Add `Assay` dataclass** to
   `molmetal/molmetal_lam/reactions/wetlab_protocol.py` (NEW FILE):
   `cell_line: str`, `target_protein: str`, `outcome_metric: str,
   outcome_value: float, outcome_unit: str, smiles: str, assay_id: str,
   collaborator: str, date: str`. CSV read/write helper
   (`load_assays`, `append_assay`). Pure data shape — no ML.

2. **Add `r_wetlab` channel** to `RewardAggregator`
   (`proof_search.py:614-706`):
   `r_wetlab(smiles) = mean(outcome_value) for smiles in assay_db,
   0.0 if smiles not yet assayed`. Wired with weight `w_wetlab`
   defaulting to 0.0 (opt-in, like `r_synth`).

3. **Add `recalibrate_from_assay()` method** to `RewardAggregator`:
   `given N new assays, fit a 2-d descriptor (mw, logP) regressor on
   (smiles, outcome_value) and blend it as 50/50 with the existing
   static oracle`. Cost: 10 lines.

4. **Add `--recalibrate-from-assay path/to/assays.csv`** CLI flag to
   `r4_lambda_only_run.py`. Backward-compat: flag defaults to
   `None` (no recalibration).

**Tier 2 — Seed a synthetic wet-lab assay queue** (4h CPU):

5. **Synthesise a synthetic `assays.csv`** by perturbing the PlatinAI
   literature corpus (Tanimoto ≥ 0.7 → "tested", random Gaussian noise
   on log(GI50) → "outcome_value"). This is *synthetic wet-lab data*
   and must be honestly labelled as such. Cost: 4h to draft +
   5 unit tests.

7. **Wire `assays.csv → r_wetlab` end-to-end**: a candidate that
   matches a synthetic "wet-tested" SMILES gets a non-zero
   `r_wetlab` value; the CLI run produces a `r_wetlab_nonzero_rate`
   column in the per-cell summary.

**Tier 3 — Real wet-lab integration** (out of scope, requires collaborator):

8. **Outreach plan** (document only) for 3 wet-lab collaborators
   (NCI-60 GI50 panel, MetalCytoToxDB cell-line panel, in-house
   OVCAR-3 / A2780 72h MTT). Cost: 1-page outreach plan as a
   `TODO/pending/31_wetlab_collaborator_outreach.md` (NEW FILE).
   Honest framing: out-of-scope for this PR.

**Total ETA for Tier 1+2**: 10h (CPU-only, no GPU).
**Tier 3 ETA**: undefined, requires chemistry facility.

### Honest framing

Mol-Metal is a **CPU-only, single-GPU, no-wet-lab-host** pipeline.
Wet-lab feedback cannot be implemented without (i) a chemistry
facility or (ii) a wet-lab collaborator with a published
high-throughput assay panel. The patch above adds the **plumbing**
(Tier 1) and a **synthetic wet-lab surrogate** (Tier 2) so that
when real wet-lab data arrives, the integration cost is hours, not
months.

This is the **load-bearing OPEN pitfall** in the experimental-
validation layer. Until Tier 1 ships, the *paper cannot make any
either* — but it correctly does not (per P5.1).

---

## P5.3 — Narrow start (5 click rules) but extensible interface — **PARTIAL**

### What we ship

**5-click core + 2 metal-coord = 7 named rules** (per P1.1 upstream
audit + `beta_reductions.py:1387-1406`):

```python
REACTION_RULES: Dict[str, object] = {
    "CuAAC":         CuAAC(),
    "SPAAC":         SPAAC(),
    "SPC":           SPC(),
    "DielsAlder":    DielsAlder(),
    "ThiolEne":      ThiolEne(),
    "Suzuki":        Suzuki(),          # R10 axis A extension
    "AmideCoupling": AmideCoupling(),   # R10 axis A extension
    "MetalLigandExchange": MetalLigandExchange(),  # F2(a)
    "AquaExchange":  AquaExchange(),     # F2(a)
}
```

**+ 13 SyntheMol REAL reactions** (lazy-loaded via
`_register_synthemol_rules`):

- `molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py:67-149`
  ships `SyntheMolRuleError`, `SyntheMolReactionRule`,
  `from_syntemol_to_lambda`, `load_real_reactions`,
  `SYNTHEMOL_REAL_REACTIONS`.
- Per the module docstring: *"SyntheMol (https://github.com/insitro/SyntheMol)
  ships a curated set of 'REAL' reaction SMARTS — bond-forming
  transformations drawn from the Enamine REAL database"*. 13
  reactions are loaded on first call.
- The wrappers expose `name`, `can_apply(List[str]) -> bool`, and
  `fire(List[str]) -> List[str]` — the same interface the synthesis
  layer expects from click-chemistry rules.

**Public-facing 5 rules** (`lam_chem.rules`):

- The user's brief says *"do we start with 5 click rules (narrow)
  but have extensibility for adding reactions?"*. The default
  expansion set is `rules` from `lam_chem.rules` (5 rules). The
  SyntheMol 13 are opt-in.

### Why this is PARTIAL (not AVOIDED)

**What works**:

1. **Registry is dict-based** (`REACTION_RULES: Dict[str, object]`)
   — adding a new rule is a 1-line entry into the dict.
2. **`ReactionRule` dataclass** (`beta_reductions.py:172-204`) is
   the single interface contract; `requires_catalyst`,
   `stoichiometry`, `_rdkit_reaction` are well-typed fields.
3. **SyntheMol adapter exists** as a reference implementation for
   "import an external 13-reaction library without polluting the
   namespace".
4. **Rate predictor / catalyst tag** is wired
   (`beta_reductions.py:1455-1480`) — new rules inherit the
   metadata infrastructure automatically.

**What's missing**:

1. **No `register_custom_rule(rule)` public API**. Today adding a
   rule requires editing `beta_reductions.py:1387-1406` directly.
   The dict is mutable, but the codebase has no convention or
   helper for *pluggable* registration.
2. **No vetting gate** for a new rule. The SyntheMol 13 are
   loaded but **NOT** consulted by the MCTS expansion set (the
   user's brief asks for "extensibility for adding reactions",
   implying a workflow, not just a registry). Per P1.1 upstream
   audit, the productive space ships 7 rules + 13 opt-in SyntheMol,
   for a *narrow* productive space.
3. **No regression test that exercises the 13 SyntheMol rules**.
   The 13 are imported lazily, but no test asserts each one can
   `fire` on a hand-curated substrate (cf. P1.4 upstream audit
   "known-Pt regression tests" gap).
4. **The `lam_chem.rules` 5-rule subset is hard-coded** as the
   public-facing set. Changing to 6 or 7 requires a separate file
   edit. The SyntheMol 13 are a *parallel* registry, not an
   *additive* one.

### Concrete patch plan (P5.3 PARTIAL → AVOIDED)

**Phase A — Public registration API** (2h CPU):

1. **Add `register_rule(name: str, rule: ReactionRule)`** to
   `molmetal/molmetal_lam/reactions/beta_reductions.py:1556`
   (next to the existing `register_syntemol_rules`). Idempotent
   (warns if `name` already in `REACTION_RULES`).

2. **Add `deregister_rule(name: str)`** for symmetry (used in
   test cleanup).

3. **Add `get_rule_names() -> List[str]`** for the MCTS expansion
   set to enumerate the *currently active* rules without consulting
   the registry directly.

4. **Add 3 unit tests** in `test_click_reactions.py`:
   `test_register_rule_idempotent`, `test_deregister_rule_restores`,
   `test_get_rule_names_includes_lazy_loaded`.

**Phase B — Vetting gate for new rules** (4h CPU):

5. **Add `vet_rule(rule: ReactionRule) -> VetReport`** to
   `molmetal/molmetal_lam/reactions/vet.py` (NEW FILE):
   - Asserts `pattern_smiles` is non-empty
   - Asserts `can_apply(substrate_smiles)` returns True for at
     least one of [cisplatin, carboplatin, oxaliplatin,
     nedaplatin, satraplatin] reference set
   - Asserts the rule's `stoichiometry` is mass-balanced
   - Asserts `predict_yield` is callable and returns finite (0,1)
     for the hand-curated reactant pair
   - Returns a `VetReport(status=PASS/FAIL, details=List[str])`.

6. **Wire `vet_rule` into `register_rule`** as a default-on
   pre-condition (skip via `vet=False` flag).

7. **Add 5 unit tests** in `test_vet.py`:
   - `vet_cisplatin_metalligandexchange` → PASS
   - `vet_amateur_pat_smarts_unbalanced` → FAIL
   - `vet_unknown_yield_predictor` → FAIL
   - `vet_amateur_reaction_no_catalyst_tag` → PASS-with-warning
   - `vet_full_syntemol_real_set` → 13/13 PASS

**Phase C — Document the public API** (1h CPU):

8. **Add a `docs/adding_a_new_reaction.md`** (NEW FILE) showing
   a 3-line example: write a `ReactionRule` subclass, call
   `register_rule("MyReaction", MyReaction())`, run `vet_rule`.
   Cost: 30 LOC doc + 1 worked example.

**Phase D — Per-rule regression test sweep** (4h CPU):

9. **Add a `test_all_rules_substrate_regression.py`** that, for
   each rule in the registry (5 click + 2 metal-coord + 13
   SyntheMol), applies the rule to a hand-curated substrate and
   asserts the product SMILES contains a known substructure. This
   is the P1.4 upstream audit patch (8h) re-scoped to all 20 rules
   instead of just Pt complexes.

**Total ETA**: 11h CPU (4h Phase A+B, 1h Phase C, 4h Phase D,
plus 2h for SyntheMol × 13 substrate curation).

### Honest framing

The current extensibility is **structural** (the dict accepts any
`ReactionRule`), but lacks a **convention** (the public registration
API + vetting gate + regression test sweep). Adding a 6th click rule
is *physically possible* in 5 lines; adding a 21st reaction from a
new external library (e.g. AiZynth template library) requires
navigating the registry convention without a vetting gate, which is
the same anti-pattern that P1.1 (no peptide-coupling rule) suffers
from.

The 13 SyntheMol rules *exist* in the codebase (per `syntemol_reactions.py:67-149`)
but are **not** consulted by the MCTS expansion set. They are a
*registry* not a *productive space*. To close P5.3, the expansion
set needs to be *opt-in* via the new `register_rule` API +
`vet_rule` gate.

---

## Aggregate verdict

| Pitfall                                          | Status    | ETA to close           | Risk    | EV   |
| ------------------------------------------------ | --------- | ---------------------- | ------- | ---- |
| P5.1 — wet-lab claim                             | AVOIDED   | 0h (none required)     | none    | high (negative — already correct) |
| P5.2 — closed-loop wet feedback                  | OPEN      | 10h CPU (Tier 1+2)     | medium  | high |
| P5.3 — narrow/extensible interface               | PARTIAL   | 11h CPU                | low     | medium |

**Total CPU**: 21h to close P5.2 + P5.3. **Pilot GPU**: 0h. **Wet-lab**: 0h
(plumbing-only; real wet-lab requires collaborator).

---

## Honest aggregate framing

1. **P5.1 is the easy win** — the paper already honestly disclaims
   wet-lab scope (section_06_limitations.tex:19-22 +
   section_07_future_work.tex:14 + anticancer_vs_general_metrics_survey.md:352).
   No engineering required. The defensive hygiene items
   (WETLAB_PROPOSED tag, badge in §4 caption, cite-only
   `wet_lab_estimate_panel.md`) are 3-5h optional polish.
2. **P5.2 is the load-bearing OPEN** — without `r_wetlab` + `recalibrate_from_assay()`,
   the reward model is a static function of in-silico oracles and
   cannot learn from any wet-lab collaborator's NCI-60 / OVCAR-3
   panel. Tier 1 (6h plumbing) is the highest-EV next step — it
   unlocks the entire wet-lab feedback loop *if* a collaborator
   ever provides data.
3. **P5.3 is structural completeness** — the registry accepts any
   `ReactionRule`, but lacks the `register_rule` /
   `vet_rule` / `test_all_rules_substrate_regression` trio that
   would let external contributors add rules safely. Tier 1 + 2 of
   the patch plan (6h) closes the gap.
4. **No single molecule is claimed synthesised or tested in vitro**
   anywhere in the paper. The honest framing is consistent across
   limitations, future work, supplementary table, and the
   anticancer-vs-general-metrics survey.
5. **The most realistic next steps** are: Tier 1 P5.2 plumbing
   (6h CPU) + Tier 1+2 P5.3 pluggable registration (6h CPU) for a
   combined 12h. Both are pre-wet-lab engineering that pays off
   if/when a wet-lab collaborator comes online.

---

## Recommended sequencing

1. **P5.2 Tier 1 first** (6h, low risk) — ship the `r_wetlab`
   channel + `recalibrate_from_assay()` hook + `assays.csv` plumbing
   + `--recalibrate-from-assay` CLI flag. This unblocks wet-lab
   integration *if* data arrives.
2. **P5.3 Tier 1+2 next** (6h, low risk) — ship `register_rule` /
   `deregister_rule` / `get_rule_names` + `vet_rule` +
   `vetting_gate`. Closes the extensibility convention.
3. **P5.3 Tier 3 deferred** (4h, optional) — substrate-regression
   sweep across the full 20-rule registry (5 click + 2 metal + 13
   SyntheMol).
4. **P5.1 defensive polish** (3-5h, optional) — WETLAB_PROPOSED tag
   + §4 caption badge + `wet_lab_estimate_panel.md`. Not required
   for AVOIDED status.
5. **P5.2 Tier 3 wet-lab outreach** (1 page) — draft
   `TODO/pending/31_wetlab_collaborator_outreach.md` listing 3
   potential collaborators (NCI-60, MetalCytoToxDB, in-house OVCAR-3).
   Cost: 1h. Out-of-scope for the engineering PR.

---

## Files referenced (absolute paths)

- /home/hugo/codes/try_triton_on_rocm/TODO/decisions.md
- /home/hugo/codes/try_triton_on_rocm/TODO/risks.md
- /home/hugo/codes/try_triton_on_rocm/TODO/pending/19_user_decisions.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pic50_margin_sweep_verdict.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/reinvent4_learned_smoke/README.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/reinvent4_learned_smoke/rocm_traced/report.json
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pitfall_audit/p1_reaction_rules.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pitfall_audit/p4_eval_novelty.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/anticancer_vs_general_metrics_survey.md
- /home/hugo/codes/try_triton_on_rocm/paper/section_06_limitations.tex
- /home/hugo/codes/try_triton_on_rocm/paper/section_07_future_work.tex
- /home/hugo/codes/try_triton_on_rocm/paper/section_02_related_work.tex
- /home/hugo/codes/try_triton_on_rocm/paper/supplementary_data_table.tex
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/beta_reductions.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/rate_predictor.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/synthesis_oracle.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/rules.py (referenced — the public-facing 5-rule subset)
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py (lines 614-706, 707-787, 1482-1524 — RewardAggregator channel inventory)

---

## Audit-only declaration

This document audits only; no code in any of the above files was
modified during this task.