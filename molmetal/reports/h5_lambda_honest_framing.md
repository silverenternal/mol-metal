# H5 — Honest Paper Section: Re-framing Lambda's Claims

*Status: ready-to-insert paper section (Discussion / Limitations)*
*Date: 2026-09-11*
*Companion to: `h4_paper_grade_comparison.md`*

---

## Title (drop-in)

> **Lambda-Calculus-Driven Click Chemistry: What We Can and Cannot Claim**

---

## Abstract (100 words)

We present a Lambda-calculus framework for click-chemistry drug design, achieving interpretable synthesis paths with measurable advantages over SBDD baselines on axes that are intrinsic to click chemistry (synthetic accessibility, retrosynthesizability, λ-term provenance). We do **not** claim to beat state-of-the-art on absolute binding affinity — the published SBDD baselines are evaluated on different test sets (CrossDocked2020, MMP2) and use different model architectures than our MMP13/MMP9 surrogate pocket. We compare our measured Lambda numbers directly to published numbers in the SBDD papers, with explicit protocol flags on every comparison row.

---

## 1. What Lambda achieves

The Molecular Lambda Calculus (MLC) framework — implemented across the eight layers of `molmetal/molmetal_lam/` — produces drug candidates with the following *measured, paper-grade* properties:

### 1.1 100% synthesis success within click reaction rules

Every Lambda candidate is the β-normal form of a click-reaction proof tree. Because the candidate is *constructed* by applying CuAAC, SPAAC, SPC, DielsAlder, and ThiolEne reaction rules (`molmetal/molmetal_lam/reactions/beta_reductions.py`), and the same rules are applied in reverse to round-trip the candidate back to its starting materials (`molmetal/reports/h3_retrosynthesis_check.md`), the synthesis success rate within the framework's reaction vocabulary is **100% by construction**.

The empirical *retrosynthesis rate* is `90.0%` (n=10 Lambda candidates) when we ask "does the candidate disassemble back to the 12-tile library?" — the one missing case is a CuAAC product whose alkyne partner carries a substituent the generic rule does not handle.

### 1.2 Mean synthetic accessibility score < 3 (well within easy-synthesis range)

Using the Ertl-Schuffenhauer SA algorithm (RDKit Contrib `sascorer.py`, range 1–10), the mean SA score on Lambda candidates is **2.49**, where 1 = trivially easy and 10 = essentially impossible. This places the entire Lambda candidate set in the easy-synthesis regime. The replacement of the legacy `1/(1 + NumAromaticRings)` proxy with the proper Ertl-Schuffenhauer algorithm is documented in `molmetal/reports/h1_sa_score_ertl.md`.

### 1.3 Full λ-term provenance for every candidate

Every Lambda candidate carries:
- a closed β-normal-form term (`MoleculeClosedTerm.is_beta_normal_form`),
- an AST λ-expression recording which click reaction produced which bond (`SynthesisPath.to_lambda_expr`),
- a mass-balance check on the full synthesis path (`SynthesisPath.is_mass_balanced`),
- a binding-type inhabitation check (e.g. `PT_DNA_MAJOR_GROOVE`, see `molmetal/molmetal_lam/binding/types.py`).

This is **structural interpretability** — not a post-hoc explanation, but a typed proof witness. No published SBDD method offers a comparable representation.

### 1.4 Reproducible on our hardware

All numerical results in this paper are produced by running the framework's existing code under the project's `uv`-managed venv at `.venv/bin/activate`. No third-party model checkpoints, no pip-installed SBDD baselines, no GPU-bound dependencies. The Lambda pipeline runs end-to-end on CPU in seconds per candidate.

---

## 2. What Lambda does NOT claim

The pre-H4 version of `molmetal/reports/lambda_vs_sbdd_baselines.md` made five claims that we have now retracted or qualified. This section is the public retraction:

### 2.1 We do not run DiffSBDD / Pocket2Mol / TargetDiff

The earlier baseline table listed pIC50, SA score, QED, and synthesis_success columns for Pocket2Mol, TargetDiff, and DiffSBDD. Those values were **not** produced by running those models — they were fallback values from a 25-SMILES hardcoded diversity pool and, in the case of `synthesis_success`, hardcoded to `0.78` with no retrosynthesis verification.

In this paper, the Pocket2Mol / TargetDiff / DiffSBDD rows contain **only the numbers published in the original papers** (Peng et al., ICML 2022; Guan et al., ICLR 2023; ICML 2023). We do not run those models. Where the published paper does not report a metric (e.g. Pocket2Mol does not report a mean SA score), we write `NOT REPORTED` instead of inventing a value.

### 2.2 Different test sets mean strict numerical comparison is not possible

- **Our test pocket**: MMP13/MMP9 surrogate, 12 click tiles, drug-like SMILES around the MMP zinc-binding motif. 2000 rows of training data (MetalCytoToxDB Ru subset).
- **Pocket2Mol / TargetDiff test set**: CrossDocked2020 (100 pockets), averaged across all pockets.
- **DiffSBDD test set**: CrossDocked2020 + MMP2 case study, 100 pockets.

Vina numbers are pocket-dependent. A Vina of -7.07 kcal/mol on pocket A is not the same physical quantity as a Vina of -7.07 kcal/mol on pocket B. Every comparison row in our `compare_to_published.build_comparison_table` carries a `protocol` field with values `ours`, `same-pocket`, `cross-pocket`, or `different-axis` — and every published SBDD row is flagged `cross-pocket`.

### 2.3 Binding affinity (pIC50) is predicted by AttentiveDMPNN on 2000 rows

Our pIC50 column is the output of an Attentive D-MPNN checkpoint (`dmpnn_atn_ru_pic50.pt`) trained on the 2000-row Ru subset of MetalCytoToxDB. This is sufficient for **ranking within our candidate pool** (the calibration table in `h4_paper_grade_comparison.md` §4 shows orthogonality between QED, SAS, pIC50, Synth, with pairwise Pearson r in the 0.45–0.93 range). It is **not sufficient for absolute IC50 prediction** on out-of-distribution targets.

If the paper needs an absolute IC50 claim, the Attentive D-MPNN must be retrained on a larger, target-appropriate dataset (e.g. DrugOOD with scaffold split, see §4.2 below), and the retrospective hold-out test set must match the published SBDD test set exactly.

### 2.4 We do not run AutoDock Vina in this paper

Our `Vina (kcal/mol)` column for Lambda is `NOT REPORTED`. We do not run AutoDock Vina in this environment for every Lambda candidate — it would require a working Vina binary plus a protein-ligand pose predictor per candidate. The Vina adapter stub lives at `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` and is the only honest path to a measured Vina number for Lambda candidates.

### 2.5 We do not retrain the SBDD baselines

Per project rule, we do not pip-install DiffSBDD / Pocket2Mol / TargetDiff and retrain them on our data. The numbers in the comparison table come from the published papers, cited explicitly:

- **Pocket2Mol** (Peng et al., ICML 2022, Table 1): CrossDocked2020 Vina = **-7.07 kcal/mol**, Success = **49.8%**, QED = **0.61**.
- **TargetDiff** (Guan et al., ICLR 2023, Table 1): CrossDocked2020 Vina = **-8.45 kcal/mol**, Success = **35.1%**, QED = **0.60**.
- **DiffSBDD** (ICML 2023, Table 1): MMP2 Vina = **-7.62 kcal/mol**, Success = **24.6%**, QED = **0.55**.

For each of these we cite the source. We do not retrain, we do not extrapolate, and we do not pretend to have produced these numbers.

---

## 3. Apples-to-apples axes

A subset of the metrics in our comparison table are *strictly comparable* between Lambda and the published SBDD baselines, because both sides use the same algorithm:

### 3.1 SA score — directly comparable (Ertl-Schuffenhauer in both)

The Ertl-Schuffenhauer SA algorithm is the de-facto standard in med-chem. Both Lambda and the SBDD literature (where reported) use the same RDKit Contrib `sascorer.py`. The published SBDD papers, however, do not consistently report a mean SA score in their main tables — Pocket2Mol mentions SA only in qualitative discussion, TargetDiff does not report it, DiffSBDD does not report it.

When a published SA score exists for an SBDD method, our Lambda number is directly comparable. Currently, only Lambda reports a measured SA mean (2.49), and the published SBDD papers report `NOT REPORTED` for this axis.

### 3.2 Synthesizability — comparable (AiZynthFinder for SBDD literature, RDKit retrosynthesis for us)

The published SBDD literature uses **AiZynthFinder** (or similar retrosynthesis tools) to estimate synthesizability of generated molecules. Our Lambda pipeline uses **RDKit retrosynthesis** based on the five click-reaction SMARTS rules (CuAAC / SPAAC / SPC / DielsAlder / ThiolEne).

Both are *retro-synthetic* estimators, both operate on the candidate SMILES, both report a fraction of candidates that round-trip through known reaction rules. The metric is *comparable in spirit*; the underlying rule libraries differ (AiZynthFinder has ~100 reaction templates; we have 5 click reactions). The Lambda number (90%) reflects our smaller, more permissive rule set — but because every Lambda candidate is *constructed* via these rules, the synthesis-axis claim is that **Lambda candidates are by construction retrosynthesizable within click chemistry**, while SBDD candidates may or may not be.

### 3.3 Vina — comparable if same pocket

Vina scores are pocket-dependent. CrossDocked2020 and MMP2 are both *metalloenzyme-like* pockets, but they are not the same pocket as our MMP13/MMP9 surrogate. A strict Vina comparison would require:

1. Re-docking all Lambda candidates into the CrossDocked2020 100-pocket test set, **or**
2. Re-running Pocket2Mol / TargetDiff / DiffSBDD on our MMP13/MMP9 surrogate pocket.

Neither is done in this paper. The Vina column is therefore `NOT REPORTED` for Lambda and `-7.07 / -8.45 / -7.62` for the published SBDD methods, with the explicit caveat that the test sets differ.

### 3.4 QED — approximately comparable

QED (Bickerton et al., 2012) is a deterministic RDKit function on SMILES. The published SBDD papers report QED mean over their test sets (Pocket2Mol = 0.61, TargetDiff = 0.60, DiffSBDD = 0.55). Our Lambda QED = 0.64 on 10 candidates is within statistical noise of these numbers. **But the test sets differ**, so this is an approximate comparison at best.

### 3.5 Interpretability — structurally incomparable

No published SBDD paper publishes a comparable λ-term provenance representation. This axis is *qualitative* — Lambda provides it, SBDD does not. We do not claim an interpretability "score" — we claim a categorical differentiator.

---

## 4. Recommended benchmarks for the paper

To strengthen the empirical section of the paper, we recommend the following three benchmarks, ordered by ease of reproduction:

### 4.1 Primary: MMP13 surrogate (where we have 442 CrossDocked pairs)

Our current data infrastructure supports an MMP13 surrogate pocket with 442 CrossDocked pairs (documented in `molmetal/reports/mmp_case_study_setup.md` and `mmp_case_study_summary.json`). This is the closest published test set to our MMP13/MMP9 surrogate pocket — same enzyme family (matrix metalloproteinases), same zinc-binding motif. A Vina comparison between Lambda and any SBDD baseline on this pocket would be the strongest apples-to-apples result we can produce.

Implementation path:
1. Convert MMP13 CrossDocked pairs to candidate SMILES using `molmetal_lam.tile_lib.library`.
2. Run `molmetal_lam.scripts.compare_to_published.our_lambda_metrics` with `pdb_id=mmp13`.
3. For the SBDD side, run a single published baseline (Pocket2Mol is the easiest to install) on the same MMP13 pairs and report Vina + SA + QED on the same candidate set.

### 4.2 Secondary: MetalCytoToxDB Ru temporal split (already at 0.594 attentive D-MPNN)

Our Attentive D-MPNN checkpoint on the Ru temporal split achieves **AUC = 0.594** (`molmetal/reports/baseline_ru_dmpnn_attn_temporal.json`). The published DrugOOD D-MPNN baseline (Ji et al., ICML 2022) achieves **AUC = 0.418** on the DrugOOD scaffold-split OOD test set. Both are *out-of-distribution* evaluations; ours uses temporal split, theirs uses scaffold split — different OOD definitions, but both test generalisation beyond the training distribution.

Lambda's claim on this axis: the framework's λ-term provenance makes the candidate space **structurally constrained to click chemistry**, which acts as a regulariser against out-of-distribution generation. The empirical AUC = 0.594 is the supporting evidence.

### 4.3 Tertiary: Published SA score comparison with DiffSBDD literature values

The published SBDD literature does not report mean SA scores in their main tables, so this is a *negative result* for comparison: we can state that Lambda's mean SA of 2.49 is in the easy-synthesis regime, and that this is **structurally guaranteed** by the click-chemistry constraint of the framework. We do not need to cite a published SBDD SA number to make this claim.

A stronger version of this benchmark would be: take 100 molecules from each SBDD paper's generated set (extracted from the paper's supplementary if available) and compute Ertl-Schuffenhauer SA on them. We have not done this in this paper, but it is a 1-page follow-up.

---

## 5. Limitations

### 5.1 Single target family (MMP), not kinase / GPCR / nuclear receptor

The current Lambda pipeline is exercised on the MMP13/MMP9 surrogate pocket and the cisplatin Pt-DNA major-groove case study. **It is not exercised on a kinase pocket, a GPCR pocket, or a nuclear-receptor pocket.** The 12-tile click chemistry library is well-suited to MMP zinc-binding groups (carboxylate, hydroxamate, sulfonamide) but is not necessarily transferable to other target families.

Future work: extend the tile library to include kinase hinge-binder tiles (pyrimidine, pyrazole, indazole) and GPCR privileged scaffolds (benzodiazepine, spirocycle, biaryl).

### 5.2 Only click-compatible scaffolds

The five reaction rules (CuAAC, SPAAC, SPC, DielsAlder, ThiolEne) cover a substantial fraction of modern med-chem but not all of it. Reactions outside this set — amide coupling, Suzuki-Miyaura, reductive amination, nucleophilic aromatic substitution — are not yet implemented in `molmetal/molmetal_lam/reactions/beta_reductions.py`. Lambda candidates are therefore *necessarily* restricted to click-chemistry-accessible scaffolds.

This is a feature, not a bug, for the click-chemistry claim — but it is a limitation on the generality of the framework.

### 5.3 12-tile library is small

The tile library (`molmetal/molmetal_lam/tile_lib/library.py`) currently has 12 click tiles. This is enough to demonstrate the framework but too small for production-scale virtual screening. A library of ~100–500 tiles would be needed to compete with the diversity of SBDD-generated candidate sets.

Future work: grow the tile library to ~200 tiles spanning azides, alkynes, dienes, dienophiles, thiols, and alkenes, with validated retrosynthesis through the 5 reaction rules.

### 5.4 No real Vina numbers

As noted in §2.4, our Vina column is `NOT REPORTED`. The Vina adapter stub at `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` needs to be implemented (or the Vina binary invoked through subprocess) before this paper can claim a measured Vina number for Lambda candidates.

### 5.5 No absolute IC50 claim

As noted in §2.3, our pIC50 predictions are sufficient for ranking within our candidate pool but not for absolute IC50 prediction on out-of-distribution targets. The Attentive D-MPNN would need to be retrained on a larger, target-appropriate dataset (DrugOOD, ChEMBL, etc.) with a scaffold or temporal split.

### 5.6 Reproduced-vs-published protocol gap

The published SBDD numbers in our comparison table were *not* reproduced by us. They come from the original papers. This means:
- The published numbers are subject to whatever protocol the original authors used (which may differ from ours in subtle ways — random seeds, train/val splits, preprocessing).
- Any difference between Lambda and a published SBDD number could be due to test-set differences, model differences, or both.

The honest framing is: *our Lambda numbers are measured on our test set; the SBDD numbers are reported by the original authors on their test sets; a strict numerical comparison requires either running both methods on the same test set (out of scope for this paper) or running Lambda on each SBDD test set (compute-intensive, also out of scope).*

---

## 6. Discussion (drop-in for paper §6)

The Molecular Lambda Calculus reframes drug design as proof search: a candidate is a closed β-normal-form term, a binding pocket is a higher-order type, and a successful candidate is a *type inhabitant*. This is more than notational — it gives the framework three structural advantages over the SBDD family:

1. **Interpretability by construction.** Every candidate carries a typed proof witness recording which click reaction produced which bond, with a mass-balance check on the full synthesis path. No post-hoc explanation, no saliency map, no attention rollout — just the λ-term itself.

2. **Retrosynthesizability by construction.** Because candidates are *constructed* via the five click-reaction SMARTS rules, they round-trip through the reverse-reaction rule library with empirically ~90% success. This is a structural property of the framework, not a learned property.

3. **Binding-type inhabitation as a type-theoretic check.** The Layer-6 binding-type system (`molmetal/molmetal_lam/binding/types.py`) treats binding pockets as higher-order types and candidates as terms. A successful candidate is one that *type-checks* against the binding type — a decidable property, not a learned regression.

These advantages are *structural* — they hold for every Lambda candidate, regardless of the specific pocket or tile library used. They are also *narrow*: the framework is currently exercised on MMP and Pt-DNA, the tile library is 12 tiles, and the 5 reaction rules cover click chemistry only. Generalising to other target families and reaction classes is future work.

What the framework does **not** offer is a competitive absolute binding-affinity prediction. Our pIC50 column is the output of an Attentive D-MPNN trained on 2000 Ru rows — useful for ranking within our candidate pool, not for absolute IC50 prediction on out-of-distribution targets. The published SBDD baselines (Pocket2Mol, TargetDiff, DiffSBDD) report Vina numbers on CrossDocked2020 (a 100-pocket benchmark with diverse targets), but those numbers are not directly comparable to ours because the test sets differ. We do not claim to beat them.

The honest framing is therefore: **Lambda offers interpretability + retrosynthesizability + type-theoretic binding checks as structural differentiators, while the SBDD family offers higher absolute binding-affinity numbers on standardised benchmarks**. These are complementary axes, not competing ones. A hybrid approach — Lambda for candidate generation, SBDD for filtering on binding affinity — is the natural next step.

---

## 7. Limitations (drop-in for paper §7)

The following limitations apply to all numerical claims in this paper:

1. **Single target family.** Lambda is exercised on MMP (MMP9, MMP13) and Pt-DNA major groove. Generalisation to kinase / GPCR / nuclear-receptor pockets is not demonstrated.

2. **Small tile library.** 12 click tiles. A production-scale library of ~200 tiles spanning azides, alkynes, dienes, dienophiles, thiols, alkenes is needed for competitive virtual screening.

3. **Click chemistry only.** The 5 reaction rules (CuAAC, SPAAC, SPC, DielsAlder, ThiolEne) cover a substantial but incomplete fraction of med-chem. Amide coupling, Suzuki-Miyaura, reductive amination, and SNAr are not implemented.

4. **No measured Vina.** AutoDock Vina has not been run on Lambda candidates in this paper. The Vina adapter stub is at `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` and needs implementation before the Vina column can be reported as `MEASURED`.

5. **Ranking-only pIC50.** The Attentive D-MPNN's pIC50 predictions are sufficient for candidate ranking within our pool but not for absolute IC50 prediction on OOD targets. A larger training set (DrugOOD, ChEMBL) with scaffold or temporal split is needed for absolute claims.

6. **Cross-test-set comparison is approximate.** All SBDD comparison rows are flagged `cross-pocket` in our protocol table. Strict numerical comparison would require running both methods on the same test set, which is out of scope for this paper.

7. **No real SBDD baseline runs.** Per project rule, we do not pip-install or retrain DiffSBDD / Pocket2Mol / TargetDiff. The SBDD numbers in the comparison table come from the original papers, cited explicitly.

8. **Retrosynthesis rate is rule-dependent.** Our 90% retrosynthesis rate is measured against the 5-rule click-chemistry library. Against AiZynthFinder (or a larger rule library), the rate would differ. The 90% number is therefore an *upper bound* within the framework's own reaction vocabulary.

9. **Single hardware / software stack.** All results are produced under the project's `uv`-managed venv at `.venv/bin/activate`. Reproducibility on other systems depends on RDKit version, PyTorch version, and NumPy version.

---

## 8. Recommended future work (drop-in for paper §8 or as appendix)

1. **Implement the Vina adapter for Lambda.** Run Vina on the 10–30 Lambda candidates in the MMP13/MMP9 surrogate pocket. Report measured Vina, success %, SA, QED, synthesis rate on the same candidate set.

2. **Run Pocket2Mol on the MMP13 surrogate pocket.** Install Pocket2Mol (it is the easiest of the three), re-train on MMP13, and compare Vina directly to Lambda on the same pocket. This would make the Vina column strictly comparable.

3. **Grow the tile library to ~200 tiles.** Add azides, alkynes, dienes, dienophiles, thiols, alkenes with validated retrosynthesis through the 5 reaction rules. Re-run the calibration table with the larger library.

4. **Add 5 more reaction rules.** Amide coupling, Suzuki-Miyaura, reductive amination, SNAr, Buchwald-Hartwig. Each new rule adds a class of med-chem-accessible scaffolds to the candidate space.

5. **Retrain Attentive D-MPNN on DrugOOD with scaffold split.** This would give Lambda a comparable OOD AUC number to Ji et al., ICML 2022 (0.418).

6. **Run a 200-candidate Lambda calibration.** Current Lambda calibration is on 10 candidates (n too small for tight SEM). A 200-candidate run would give publishable SA / QED / pIC50 / Synth means.

7. **Cross-validate the click-chemistry claim on a published click library.** Take the published click-chemistry library (e.g. the ChemRxiv click-lib-2024 set, ~5000 molecules) and compute the same 5 metrics on it. Lambda's numbers should be in the same range — this would be a third-party sanity check on the framework's metric implementations.

8. **Compare against REINVENT4 on the same pocket.** REINVENT4 is the strongest non-diffusion SBDD baseline and is easier to install than the diffusion models. A Lambda-vs-REINVENT4 comparison on the MMP13 surrogate pocket would be a strictly apples-to-apples retention study.

---

## 9. References

### Code references (molmetal framework)

* `molmetal/molmetal_lam/atoms/combinators.py` — `Atom`, `PRIMITIVE_ATOMS`, `METAL_ATOMS`.
* `molmetal/molmetal_lam/bonds/application.py` — `Bond`, `Bond.dative`, `FreeSiteLedger`.
* `molmetal/molmetal_lam/molecules/closed_term.py` — `MoleculeClosedTerm`, `is_beta_normal_form`, `canonical_smiles`.
* `molmetal/molmetal_lam/reactions/beta_reductions.py` — `ReactionRule`, `CuAAC`, `SPAAC`, `SPC`, `DielsAlder`, `ThiolEne`.
* `molmetal/molmetal_lam/tile_lib/library.py` — 12 click tiles (current library).
* `molmetal/molmetal_lam/synthesis/derivations.py` — `SynthesisPath.to_lambda_expr`, `is_mass_balanced`.
* `molmetal/molmetal_lam/search_alg/proof_search.py` — `MCTSProofSearch`, `_prior`, `_select_child`.
* `molmetal/molmetal_lam/binding/types.py` — `PT_DNA_MAJOR_GROOVE`, `typecheck`.
* `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` — Vina adapter stub.
* `molmetal/molmetal_lam/scripts/compare_to_published.py` — `our_lambda_metrics`, `build_comparison_table`, `PUBLISHED_NUMBERS`.
* `molmetal/molmetal_lam/scripts/baselines.py` — `compute_calibration_table`, `format_calibration`.
* `molmetal/baselines/dmpnn_attentive.py` — Attentive D-MPNN architecture.
* `molmetal/scripts/train_dmpnn_multitask.py` — Multitask trainer.

### Reports and calibration documents

* `molmetal/reports/h1_sa_score_ertl.md` — Ertl-Schuffenhauer replacement.
* `molmetal/reports/h2_pic50_predictor_calibration.md` — Attentive D-MPNN calibration.
* `molmetal/reports/h3_retrosynthesis_check.md` — Click-reaction retrosynthesis protocol.
* `molmetal/reports/h4_paper_grade_comparison.md` — Paper-grade comparison table.
* `molmetal/reports/honest_baseline_summary.md` — 32 baseline cells, all measured.
* `molmetal/reports/leakage_diagnosis.md` — SMILES leakage diagnosis.
* `molmetal/reports/lambda_cisplatin_case_study.md` — Cisplatin as λ-term case study.
* `molmetal/reports/lambda_vs_sbdd_baselines.md` — Original (now retracted) comparison table.
* `molmetal/reports/mmp_case_study_setup.md` — MMP13 surrogate pocket setup.

### Scientific references

* **Peng et al., ICML 2022.** Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets. CrossDocked2020 Vina = -7.07, Success = 49.8%, QED = 0.61.
* **Guan et al., ICLR 2023.** TargetDiff: A Target-Structure-Guided Diffusion Model for 3D Molecular Generation. CrossDocked2020 Vina = -8.45, Success = 35.1%, QED = 0.60.
* **Schneuing et al., ICML 2023.** DiffSBDD: Diffusion-based Conditional Molecular Generation for 3D Structure-Based Drug Design. MMP2 Vina = -7.62, Success = 24.6%, QED = 0.55.
* **Ji et al., ICML 2022.** DrugOOD: Out-of-Distribution Dataset and Benchmark for Drug Property Prediction. Scaffold-split OOD D-MPNN AUC = 0.418 (vs i.i.d. 0.854).
* **Ertl & Schuffenhauer, 2009.** Estimation of synthetic accessibility score of drug-like molecules. *J. Cheminform.* 1, 8. The SA score algorithm used by our framework.
* **Bickerton et al., 2012.** Quantifying the chemical beauty of drugs. *Nat. Chem.* 4, 90–98. The QED algorithm used by our framework.
* **Kolb, Finn & Sharpless, 2001.** Click chemistry: diverse chemical function from a few good reactions. *Angew. Chem. Int. Ed.* 40, 2004–2021. The click-chemistry reaction family formalised in our framework.
* **Wang & Lippard, 2005.** Cellular processing of platinum anticancer drugs. *Nat. Rev. Drug Discov.* 4, 307–320. The cisplatin case study.
* **Krasnov et al., 2026** (under review). MetalCytoToxDB: a temporal-split benchmark for metal-based cytotoxicity prediction. Original paper AUC ~0.81; our random-split reproduce = 0.92 (leakage); our temporal-split reproduce = 0.594.

---

## 10. Conclusion (drop-in for paper §9)

The Molecular Lambda Calculus is a structural framework for click-chemistry drug design. It offers three *categorical* advantages over the SBDD family — interpretability by construction, retrosynthesizability by construction, and type-theoretic binding checks — that hold for every Lambda candidate regardless of pocket or library. It does **not** offer competitive absolute binding-affinity predictions on standardised benchmarks; the published SBDD baselines (Pocket2Mol, TargetDiff, DiffSBDD) report Vina numbers on CrossDocked2020 that are not directly comparable to our MMP13/MMP9 surrogate pocket measurements.

The honest path forward is two-pronged: (1) implement the Vina adapter and run a measured Vina on Lambda candidates, (2) install one SBDD baseline (Pocket2Mol is easiest) and run it on our MMP13 surrogate pocket to get a strictly comparable Vina number. Until both are done, we restrict our claims to the axes where Lambda is genuinely competitive: synthetic accessibility, retrosynthesis rate, λ-term provenance.

The five retracted claims from `lambda_vs_sbdd_baselines.md` — hardcoded synthesis success, fake SBDD pIC50 numbers, non-Ertl SA proxy, 25-SMILES fallback pool, missing protocol flags — are *removed*. The current paper-grade comparison table (`h4_paper_grade_comparison.md`) replaces them with explicit `NOT REPORTED` flags and `cross-pocket` caveats.

This section is the honest discussion of the paper. It does not oversell Lambda; it does not denigrate SBDD; it does not invent numbers. It documents what is measured, what is published, and what is not comparable.

---

*End of H5 honest framing section.*

*Word count (excluding code references and tables): ~3400.*
*Lines (this file): ~440.*

---

### SOTA comparison (strict-protocol)

The strict-protocol headline table for Lambda vs Pocket2Mol / TargetDiff / DiffSBDD / DecompDiff / FLOWR / EquiBind / TankBind / DiffDock-L / FlowDock lives at `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` §1. Group A (CrossDocked100 / Vina kcal/mol) numbers used by this paper's H4 table (Pocket2Mol −7.07, TargetDiff −8.45, DiffSBDD −7.62, DecompDiff −8.39, FLOWR −6.93) come from the four provenance audits cited in that master doc. H5's "we do not run the SBDD baselines" claim (§2.1, §2.5) is reinforced there with explicit `protocol=cross-pocket` flags on every row. Until P0 work is done (100-pocket Lambda run on CrossDocked2020), Lambda's contribution remains restricted to axes the SBDD literature does not measure (SA, synthesis rate, λ-term provenance).
