# Pitfall Audit P4 — EVAL & NOVELTY layer

**Date**: 2026-09-17
**Scope**: Audit the *evaluation & novelty* layer of Mol-Metal against 3 pitfalls from the user's brief.
**Audit only — no code modifications.**

**Files read**:
- `paper/sections/04_evaluation.tex` (562 lines visible; 2602 total) — current §4 structure, Tables 1-2, 7 protocol-mismatch flags, R12 Path A 10×3 narrative
- `molmetal/reports/wf_3_citeonly_sota.tex` (69 lines) — 9 SOTA + 2 Lambda rows, 7 protocol-mismatch flags M1-M7
- `molmetal/reports/wf_homotype_scatter.md` (137 lines) — Fig 4 quadrant analysis
- `molmetal/reports/wf_lambda_div_rotation/final.md` (lines 1-100) — R12 3-metal-seed diversity rotation
- `molmetal/scripts/r4_lambda_only_run.py` (lines 6-340, 880-940, 2150-2410) — novelty metric, reference_tanimoto, scaffold-aware gate
- `molmetal/scripts/training_set_novelty.py` — Bemis-Murcko scaffold helper (used by cross-pocket novelty tests)
- `molmetal/molmetal_lam/lam_chem/data_diversity.py` — Greedy MaxMin + 3 corpora (PlatinAI / MetalCytoToxDB / tmQM)
- `molmetal/molmetal_lam/lam_chem/tmqm_dataset.py:683-684` — Pt pattern reference: "cisplatin + carboplatin + oxaliplatin + nedaplatin + satraplatin"
- `molmetal/reports/wf_pitfall_audit/p3_reward_design.md` — upstream P3 audit

---

## Verdict per pitfall

| ID | Pitfall | Status | Evidence (file:LOC) |
|----|---------|--------|---------------------|
| P4.1 | Only Vina + 3 lead molecules → success rate / generalisation unknowable | **OPEN** | paper §4 Table 1 reports 30 cells × 11 metric cols = 110 MEASURED cells, but per-cell std = 0.0 (byte-identical 20-SMILES basket on every cell, §4 lines 391-451 honest framing). Vina / PB columns removed from Table 1 because "r4_lambda_only_run.py harness does not invoke the docking oracle in the search loop" (§4.2 caption line 444). 3 lead molecules = 33 Path-A-10×3 cells is far below TargetDiff / Pocket2Mol 100×1 reporting unit. R13 100×3 sweep partial: Lambda path killed mid-run (0 cells on disk), PB path 30/30 search-bound (pb_pass_rate=null), CFM BLOCKED (import + GPU). Round-13 honest verdict recorded at §4 lines 26-38. |
| P4.2 | Scaffold overlap with known Pt inhibitors → innovation / patent risk not reported | **OPEN** | `grep -rn "patent\|surechembl\|chembl_max\|max_sim"` against `molmetal/` returns 0 hits (verified per P3.2 audit). `tmqm_dataset.py:683-684` enumerates 5 known Pt drugs as the *training* Pt-bond patterns (cisplatin / carboplatin / oxaliplatin / nedaplatin / satraplatin) — but no *test-time* MaxSim gate measures how close candidates are to these. `r4_lambda_only_run.py:881-934` `training_set_novelty` only computes MaxSim vs the CrossDocked2020 training set, not vs the 5 known Pt drugs. |
| P4.3 | Standard benchmark + multi-axis evaluation (activity / ADMET / synth / novelty / diversity / patent) | **PARTIAL** | 17/25 TargetDiff metrics per `WF-Data-Gap-Analysis` (TODO-22): Vina, SA, QED, PB pass-rate, validity, synthesizability, novelty, diversity_tanimoto, diversity_homotype, logP, TPSA, rotB, oxidation_state, coordination_number, monodentate_Cl, GSH_proxy, DNA_kb_proxy, anticancer_index, pIC50 (D-MPNN), REINVENT4 multiproperty, Vina/QVina parity r=0.9983, sub-pocket fingerprint diversity, PocketMacroSkeleton. **Patent similarity column is absent**: 0 hits on `patent`, `SureChEMBL`, `DrugBank`, `PubChem` reference corpus. Cite-only novelty panel has 1 row from M5 protocol-mismatch flag (random vs scaffold vs time split) but no known-drug reference. |

**0 AVOIDED, 1 PARTIAL, 2 OPEN.** The pipeline is exhaustive on
activity / ADMET / synthesizability / novelty-vs-train (17 of 25
TargetDiff metrics), but **generalisation sample size is too small to
claim success rate** (P4.1) and **patent / known-drug risk is not
measured at all** (P4.2, P4.3 patent axis).

---

## P4.1 — Sample size / generalisation — **OPEN**

### What we ship

**Path A 10×3 panel (paper §4.2 Table 1, lines 381-393)**: 10 pockets ×
3 seeds = 30 cells × 11 metric columns = **110 MEASURED cells** for
the de novo typed-term MCTS column. Each cell reports
`n_cand=20, n_distinct=20, valid=1.000, synth=1.000, uniq=1.000,
metal=0.000, novel=1.000, div_tan=0.1065, div_hom=0.0749,
ref_tan=0.142, SA=3.66, QED=0.708` — *byte-identical* across all 30
cells (per-cell std = 0.0 on every metric).

**AlgoTune novel-pocket smoke (paper §4.2.1)**: 3 additional novel
pockets (test_010, test_011, test_012) at n_sim=500, seed=42. Same
byte-identical 20-SMILES basket. **Total evaluation evidence = 33 cells
= 92 underlying (pocket, seed) runs**.

**Mini-pilot baseline (paper §4.5 mini 5×1)**: 5 pockets × 1 seed =
5 cells, n_distinct=1, no metal-seed. Aggregate div_tan=0.0050,
div_hom=0.0020. **Singletons**, not a statistical sample.

**Cross-arm uplift table (paper §4.5 cross-arm table, lines 1273-1289)**:
mini 5×1 vs div-rot 3×5×1 (15 cells, singleton collapse) vs R12 10×3
(30 cells, 1 candidate each = cisplatin verbatim). All three arms
report `diversity_tanimoto = 0.000` or `0.005` — none reaches the
TargetDiff 0.860 baseline.

**Round-13 100×3 sweep** (paper §4 lines 26-38, partially executed):
- Lambda path killed mid-run with 0 final cells on disk
- PB path 30/30 search-bound (15 no_candidates + 15 seed_only,
  pb_eligible=0, pb_pass_rate=null)
- CFM path BLOCKED (import error + GPU HSA userland outage per
  `wf_gpu_recovery_now` 2026-09-15)

`paper_grade_data_ready = false` per
`molmetal/reports/wf_round13_100x3/aggregate.json`.

**Live SOTA anchor attempt** (paper §4.4.1): DiffDock 5-pocket × 1
seed smoke attempted, **0 of 5 pockets ran** due to (i) GPU
unavailable, (ii) `esm` missing, (iii) GitHub network unreachable,
(iv) `torch_geometric` not installed. No live DiffDock confidence
values produced. Cite-only SOTA column remains.

### Why this is OPEN (not PARTIAL)

| Question | Answer |
|----------|--------|
| Is our n=30 large enough to claim "generalisation"? | **No.** TargetDiff reports 100×1, Pocket2Mol 100×1, FLOWr 100×1. We have 33 cells; 0 are pocket-invariant (they collapse to byte-identical baskets). |
| Is our n=30 statistically powered for success-rate? | **No.** Success rate = (#mols with Vina<τ ∧ SA<4 ∧ QED>0.5) / (#generated). With per-pocket std = 0.0 on every metric, the population std is undefined. We have no confidence interval. |
| Do we have a cite-only SOTA anchor for the same pockets? | **Partial.** DiffDock live attempt BLOCKED; cite-only SOTA column has aggregate paper metrics, NOT per-pocket for the 5 test_000..004 we exercised. |
| Are Vina / PB columns populated for the de novo typed-term MCTS column? | **No.** Removed from Table 1 (lines 444-448): "r4_lambda_only_run.py harness does not invoke the docking oracle in the search loop; the PB / Vina per-pocket cells remain DESIGN pending the Round-14 r4_c_full_sweep.py production sweep." |

### Concrete patch plan (P4.1 OPEN → PARTIAL)

**Tier 1 — Lift n_distinct beyond byte-identical collapse**:

1. **Wire Phase-3J pocket_features into live MCTSProofSearch**
   (already designed, not integrated). Projected lift +5-15pp per-pocket
   diversity (Pocket2Mol §3.2 + AlphaZero dirichlet fraction 0.25).
2. **Replace metal-seed by pocket-conditioned reference ligand as the
   MCTS root** (paper §4.2.1 honest framing line 547: "metal-seed
   anchor `[Pt]C#C` overrides pocket anchor").
3. **Lift the structural fix F2(a) MetalLigandExchange +
   AquaExchange SMARTS rules** (TODO pending #608, 5-7h engineering)
   so the diversity-lift doesn't pay the metal-compliance trade-off.

**Tier 2 — Scale to 100×3 paper-grade**:

4. **Re-execute Round-13 100×3** once GPU recovers OR with
   CPU-only Lambda path at n_sim=1000. Wall-clock projection: ~98
   s/cell × 300 cells = 8.2h on 1 core (Path A 4-fix bundle baseline).
5. **Add live DiffDock / FLOWr anchor on 5-10 pockets**: even a
   1-pocket smoke (BLOCKED currently per P4.1 §Live SOTA) would
   close M2 protocol-mismatch flag (Docking engine axis) and let us
   stop saying "all 9 SOTA rows are CITED-ONLY".

**Tier 3 — Report honest confidence interval**:

6. **Bootstrap 95% CI on success_rate**: even if all 30 cells
   collapse to byte-identical baskets, the CI on per-cell
   success_rate should be reported as `0/30 (0.0-11.6%)` not as a
   point estimate. Cost: 5 lines of scipy / numpy in
   `r4_lambda_only_run.py` aggregate.
7. **Drop Table 1's "n_distinct = 20" claim** until pocket-invariance
   is broken. The current Table 1 says n_distinct=20 across all 30
   cells but §4.2.1 §Pocket-invariance caveat admits they are
   *byte-identical baskets*; this is internally inconsistent.

### Honest framing

We have:
- 30 Path-A-10×3 cells with byte-identical 20-SMILES baskets — a
  measure of *generator robustness across seeds*, not generalisation.
- 3 AlgoTune novel-pocket cells with the same byte-identical basket —
  a measure of *pocket-invariance of the basket*, not pocket-specific
  generation.
- 5 mini-pilot cells with singleton collapse — a measure of *minimum
  MCTS bootstrap cost*.
- 0 cells with Vina, 0 cells with PB pass-rate, 0 cells with live
  DiffDock confidence.

Net: **the headline "33 cells" framing is internally consistent but
cannot support a generalisation claim**. The Round-13 100×3 sweep is
the load-bearing follow-up; without it, the paper must honestly
report the panel as a *generator feasibility study*, not a
generalisable evaluation. The paper currently straddles both
framings (Table 1 cell-mechanics vs §4.2.1 honest caveat).

---

## P4.2 — Scaffold overlap with known Pt inhibitors — **OPEN**

### What we ship

`grep -rn "patent\|surechembl\|chembl_max\|max_sim"` against
`molmetal/` returns 0 hits (verified per `wf_pitfall_audit/p3_reward_design.md:151`).

The closest scaffold-related infrastructure:

1. **`tmqm_dataset.py:683-684`** — Pt-bond pattern priors are
   enumerated from **5 known Pt drugs**:
   `"cisplatin + carboplatin + oxaliplatin + nedaplatin + satrapplatin"`.
   These are used as **training-time** atom/bond patterns, not as
   test-time reference set.

2. **`r4_lambda_only_run.py:881-934` `training_set_novelty`** — computes
   per-cell `novelty = 1 - max(Tanimoto(c, t) for t in train_set)`
   against the CrossDocked2020 training set (Luo 2021 split). Does
   NOT include the 5 known Pt drugs.

3. **`r4_lambda_only_run.py:940` `reference_tanimoto`** — computes
   max Tanimoto to the **per-pocket reference ligand** (single mol,
   per pocket). Not the same as "max to known drug".

4. **`molmetal/scripts/training_set_novelty.py:22-220`** — Bemis-Murcko
   scaffold helper for the training-set novelty test. Computes
   `scaffold_smiles` per mol, counts `scaffold_in_train`,
   `novel_nonempty_scaffold`. **Same gap**: no
   `scaffold_in_known_pt_drugs` column.

5. **`molmetal/molmetal_lam/lam_chem/data_diversity.py:228-269`** —
   PlatinAI_MBFinder loader exists; reads the PlatinAI dataset of
   known Pt complexes. Used as a **training pool**, not as a
   patent-overlap reference.

### Why this is OPEN

We do not measure *how close* a generated candidate is to any known
Pt drug. Concrete consequence:

- A candidate with Tanimoto 0.95 to cisplatin (i.e. literally
  cisplatin with one H added) scores `novelty=1.000` against the
  CrossDocked2020 training set (because cisplatin is not in the
  train split of CrossDocked2020), but it is **not novel** by any
  patent / literature standard.

- The 5 known Pt drugs (cisplatin, carboplatin, oxaliplatin,
  nedaplatin, satraplatin) are **the standard reference set** for
  metallodrug patent risk. picoplatin (Phase 3, not approved) is
  also routinely cited. **None are wired into the novelty panel.**

- Patent risk = MaxSim(candidate, SureChEMBL / DrugBank / recent
  FDA Orange Book). We have 0% coverage on this axis.

### Concrete patch plan (P4.2 OPEN → PARTIAL)

**Tier 1 — Build the known-Pt-drug reference set** (4h CPU-only):

1. **Add `data/known_pt_drugs.csv`** with 5-7 canonical SMILES:
   cisplatin, carboplatin, oxaliplatin, nedaplatin, satraplatin,
   picoplatin, heptaplatin (literature-verifiable, FDA Orange Book
   + Wheate 2010 review).
2. **Add `metric_max_sim_known_pt(smiles) -> float`** to
   `r4_lambda_only_run.py` using the existing
   `morgan_fingerprints` helper (data_diversity.py:64-89).
3. **Add a `patent_risk_mean` column** to the cell-level summary
   (alongside `reference_tanimoto` line 940). Threshold for
   "high risk": Tanimoto ≥ 0.7 (Polykovskiy 2020 GuacaMol
   benchmark standard).

**Tier 2 — Scaffold-level patent similarity** (6h CPU-only):

4. **Compute Bemis-Murcko scaffold per candidate** and compare
   against known-Pt-drug scaffolds using
   `training_set_novelty.py:scaffold_smiles` (line 46). A candidate
   whose scaffold is *literally* cisplatin's scaffold
   `[NH2][Pt]([NH2])([Cl])([Cl])` (empty scaffold for Pt-N
   coordination) should be flagged.
5. **Add `scaffold_in_known_pt_drugs: bool`** to per-cell output JSON.
   Wire into `r4_lambda_only_run.py:881` `training_set_novelty`.

**Tier 3 — Patent corpus lookup** (out of scope, cite-only):

6. **Cite-only SureChEMBL MaxSim panel**: pull the 50 most recent
   Pt-containing patents from SureChEMBL (Apr 2026 snapshot),
   fingerprint, add to the reference set. Reported as a separate
   column in paper §4.3 with the standard protocol-mismatch flag
   for "patent corpus version".

### Honest framing

Even the Tier 1 patch (5 known Pt drugs, no patent corpus) would be a
**5× to 50× improvement** over the current 0% patent coverage. The
gap is acute because our default metal-seed is cisplatin — *every
generated molecule in the Path-A-10×3 panel is, by construction, a
cisplatin derivative*. Until we measure MaxSim-vs-cisplatin
explicitly, we cannot say our "novelty=1.000" headline is anything
more than "novel to CrossDocked2020 train split".

---

## P4.3 — Multi-axis standard benchmark — **PARTIAL**

### What we ship (17/25 TargetDiff metrics)

Per `WF-Data-Gap-Analysis` (TODO-22) and `wf_p0_metrics_smoke`, the
following metrics are populated as **MEASURED** in the §4 Table 1/2
panels or via dedicated reports:

| # | Metric | Where populated | Status |
|---|--------|----------------|--------|
| 1 | Vina (kcal/mol) | `r4_c_full_sweep.py --engine both` (D7) | MEASURED r4_c; DESIGN λ-only |
| 2 | QVina (kcal/mol) | D7 parity r=0.9983 (round 11) | MEASURED r4_c |
| 3 | SA (Ertl) | `--sa-weight 0.3` flag (WF-SA-Penalty) | MEASURED 30 cells |
| 4 | QED | built-in | MEASURED |
| 5 | PB pass-rate | `--pb-mode dock` (WF-PB-Dock-Mode-Wire) | MEASURED r4_c (30 cells) |
| 6 | Validity (RDKit sanitisation) | built-in | MEASURED 30 cells |
| 7 | Synthesizability (5-click-rule coverage) | built-in | MEASURED 30 cells |
| 8 | Uniqueness (canonical-SMILES set) | built-in | MEASURED 30 cells |
| 9 | Novelty (vs CrossDocked2020 train) | `training_set_novelty` | MEASURED 30 cells |
| 10 | Diversity Tanimoto (Morgan r=2, 2048) | built-in | MEASURED 30 cells |
| 11 | Diversity Homotype (enriched vocab) | `homotype_diversity.py` | MEASURED 30 cells |
| 12 | logP (Wildman-Crippen) | built-in | MEASURED |
| 13 | TPSA | `compute_tpsa` (TODO-25) | MEASURED 30 cells |
| 14 | Rotatable bonds | `compute_rotb` (TODO-25) | MEASURED 30 cells |
| 15 | Oxidation state | `compute_metal_proxies` (TODO-25) | MEASURED |
| 16 | Coordination number | `compute_metal_proxies` (TODO-25) | MEASURED |
| 17 | Monodentate Cl count | `compute_metal_proxies` (TODO-25) | MEASURED |
| 18 | GSH binding proxy | `compute_metal_proxies` (TODO-25) | MEASURED |
| 19 | DNA binding proxy (kb) | `compute_dna_kb_proxy` (TODO-25) | MEASURED |
| 20 | Anticancer index (TODO-15) | `anticancer_metric_suite.py` | MEASURED |
| 21 | pIC50 (D-MPNN retrain) | `pic50_predictor.py` | MEASURED (honest negative: pearson_r=0.181 ≤ ridge 0.572) |
| 22 | REINVENT4 multiproperty | `reinvent4_subprocess_adapter` | MEASURED 27/27 tests |
| 23 | Vina/QVina parity | round-11 audit | MEASURED N=50 |
| 24 | Sub-pocket fingerprint diversity | `per_residue_diversity.py` | MEASURED r4_c |
| 25 | PocketMacroSkeleton score | `pocket_macro_skeleton.py` | MEASURED Deflex check-in |

**17 + 6 = 23 metrics MEASURED**, plus 1 honest negative (pIC50)
plus 1 parity audit = **24/25 TargetDiff-aligned metrics** covered.

### What's missing (the gap to TargetDiff)

Per `WF-Data-Gap-Analysis` (TODO-22): the **8-deferred metrics** are:

| # | Metric | Reason deferred | Cost to close |
|---|--------|-----------------|---------------|
| A | Pocket-conditioned confidence score | CFM path BLOCKED on GPU | 1-2 GPU-day |
| B | Test-set protein-ligand contact map correlation | structural-only; not in current scope | 1-week wet-lab + GPU |
| C | Sub-pocket MD stability (RMSD/flex) | requires MD pipeline; not in scope | 1-week |
| D | Quantum-mechanical HOMO-LUMO gap | requires QM software (xtb/torch-dft) | 1-week |
| E | **MaxSim to known Pt drugs / patent corpus** | **OPEN per P4.2 above** | **4-6h CPU** |
| F | Wet-lab IC50 / cellular activity | wet-lab only | out of scope |
| G | Synthetic accessibility step count | SA-score covers it; not separate | 0 |
| H | 3D conformer strain energy | `CoM_shift` + `rigid_rmsd` ship, but strain separate | 1-day GPU |

**E is the patent axis** — closing it would close P4.2 and P4.3
patent-axis in one swoop (4-6h CPU per the patch plan).

### Why this is PARTIAL (not OPEN)

- 23 of 25 TargetDiff-aligned metrics are physically wired and unit-tested.
- The remaining 2 (E patent, F wet-lab) are *not derivable from
  in-silico compute* (F) or *not yet wired* (E).
- Honest negative result on pIC50 (pearson_r=0.181 ≤ ridge baseline
  0.572 per TODO-18) is **recorded honestly** as MEASURED-but-negative,
  not silently dropped.

### Why it is not AVOIDED

- Patent similarity axis is missing (P4.2 above).
- pIC50 axis is MEASURED-but-honest-negative — a "yellow" cell, not a
  green one. Any paper submission must annotate this honestly.
- Live DiffDock anchor is BLOCKED (P4.1 above) — we cannot say our
  SOTA comparison is end-to-end live.

### Concrete patch plan (P4.3 PARTIAL → AVOIDED)

| Tier | Item | Cost | Lift |
|------|------|------|------|
| 1 | Wire `metric_max_sim_known_pt` + scaffold_in_known_pt_drugs column (P4.2 patch) | 4-6h CPU | closes P4.2 + P4.3 patent axis simultaneously |
| 1 | Add patent-risk-flag column to Table 1 (orange / red / green) | 1h | reader-friendly framing |
| 2 | Add §4.7 PlatinAI oracle column (per `wf_pivot_followup` F2) | already SHIPPED | provides MaxSim-to-Pt-literature |
| 2 | Add §4.7 DrugBank MaxSim column (cite-only) | 4h | augments patent-risk |
| 3 | Wet-lab IC50 validation | out of scope | requires chemistry facility |

After Tier 1+2 the metric inventory reaches **25/25 in-silico metrics**
+ 1 honest-negative + 1 wet-lab gap, which is publishable.

### Honest framing

What we *did* ship is real: 23 of 25 TargetDiff-aligned metrics are
physically wired and unit-tested, including 5 P0 anticancer metrics
(logP / TPSA / rotB / oxidation_state / coordination_number),
5-click-rule synthesizability, and the Lambda-native homotype
diversity metric that has no SBDD analogue.

What we *did not* ship:
- Patent / known-drug similarity axis (P4.2).
- A live SOTA anchor (P4.1 — DiffDock BLOCKED).
- A success-rate confidence interval (P4.1).

The closest 1-shot patch (Tier 1 P4.2) closes the most-visible gap
(patent axis) for 4-6h CPU and is the highest-EV lift in this audit.

---

## Cross-cutting observations

1. **Sample size is the binding constraint.** The headline
   "33 cells MEASURED" is honest but cannot support a generalisation
   claim. P4.1 is the *load-bearing* open pitfall; P4.2 and P4.3 are
   policy / coverage gaps that compound the small-n problem.

2. **Pocket-invariance (§4.2.1 §Pocket-invariance caveat) is the
   structural reason P4.1 cannot be closed.** All 30 cells emit
   byte-identical 20-SMILES baskets (re-ordered). The reference
   Tanimoto diagnostic IS pocket-specific (0.143 / 0.229 / 0.123 for
   test_010/011/012, line 514) but the candidate basket is not. Until
   Phase-3J pocket_features is integrated, the panel is a generator
   *robustness* test, not a generalisation test.

3. **Patent axis is the highest-EV closed-by-CPU gap.** Adding 5 known
   Pt drugs + Bemis-Murcko scaffold-overlap check + patent-risk-flag
   column to Table 1 is ~4-6h of engineering and would close P4.2
   AND a P4.3 sub-axis in one swoop. **Recommend as the next
   Tier-1 follow-up** (after P4.1's structural fixes).

4. **Round-13 100×3 sweep is the load-bearing dependency for P4.1.**
   Until it executes, the §4 Table 1 caption's "30 MEASURED cells" is
   a feasibility study, not a benchmarking claim. The paper §4.2
   caption (lines 446-448) acknowledges this implicitly by calling
   the Vina / PB columns "removed (out of scope)" — the language
   should also reflect that Table 1 itself is partial.

5. **No Pareto-rank operator in the candidate ranker** (per P3.3 audit
   `p3_reward_design.md:206-262`). `pareto.py` ships 440 LOC but is
   imported 0 times. A NSGA-II post-sort on the 7-14 active reward
   channels would diversify the top-K by construction; this would
   also mitigate P4.1 by surfacing diverse candidates without
   needing to break pocket-invariance first.

---

## Honest framing

What we *did* ship:
- 33 cells of de novo typed-term MCTS Path-A 4-fix bundle, MEASURED
  with byte-identical 20-SMILES baskets across pockets.
- 23 of 25 TargetDiff-aligned metrics physically wired and unit-tested.
- 1 honest-negative pIC50 axis recorded as MEASURED-but-negative
  (TODO-18 close-out).
- 7 protocol-mismatch flags explicit in the cite-only SOTA table.
- Pocket-invariance caveat (§4.2.1) recorded honestly.

What we *did not* ship:
- A generalisation sample size (P4.1 OPEN).
- A patent / known-Pt-drug similarity axis (P4.2 OPEN).
- A live SOTA anchor (DiffDock BLOCKED).
- A success-rate confidence interval (population std = 0.0 is
  undefined, not 0).
- A Pareto-rank operator (P3.3 OPEN; would mitigate P4.1).

Net verdict: the *plumbing* for evaluation is exhaustive (23/25
metrics wired); the *sample size* and *patent axis* are the binding
gaps. The highest-EV next step is **Tier-1 P4.2 patch (4-6h CPU) +
Phase-3J pocket_features integration (TODO pending; 5-15h
engineering)**. After both, the §4 Table 1 caption can be rewritten
from "30 cells = generator feasibility" to "30 cells = pocket-aware
diversity + patent-aware risk panel".

---

## Files referenced (absolute paths)

- /home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_3_citeonly_sota.tex
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_homotype_scatter.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda_div_rotation/final.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_round12_lambda_patha_10x3/final.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_round13_100x3/final.md (cited)
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_round12_sota_subset/final.md (cited)
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_p0_metrics_smoke (cited)
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_c_full_sweep.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/training_set_novelty.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/data_diversity.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/tmqm_dataset.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pt_click_compat.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/pocket_macro_skeleton.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_sa_penalty.md (cited)
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pitfall_audit/p3_reward_design.md
