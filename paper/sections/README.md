# Paper — Section List & Cross-Reference Map

Paper target: **Digital Discovery** (RSC) Q1 IF 8.5 (primary), with **J. Chem. Inf. Model.** (ACS) Q1 IF 5.6 as fallback.
Style: IMRaD, 9pt body, single-column for §1–§2 then double-column for §3–§7 (Digital Discovery layout).

---

## Section inventory

| # | File | Title | Status | Owner workflow | Cross-refs |
|---|---|---|---|---|---|
| 1 | `01_intro.tex` | Introduction | SHIPPED (this PR) | WF-Paper-1 (§1) | §2, §3, §4, §5, §6, §7 |
| 2 | `section_02_related_work.tex` | Related Work | SHIPPED (pre-existing) | WF-Paper-1 (§2) | §3 (MLC vs SBDD), §4 (cite-only SOTA), §6 (gap analysis) |
| 3 | `03_method.tex` (master) + 4 sub-section fragments: `03_1_mlc_formalism.tex` (MLC formalism, 9-layer architecture, strong-normalisation claim), `03_2_click_chemistry.tex` (5 click reactions as typed reductions, verbatim SMARTS), `03_3_metal_geometry_prior.tex` (MetalGeometryPrior: first-class constraint, 5 metals registered, dative-vs-covalent), `03_4_mcts_search.tex` (MCTS over typed-term beta-NF space, UCB + VirtualLoss + TT, WF-Lambda-1/1c pilot evidence) | Method: MLC + Click + MGP + MCTS (4 sub-sections) | SHIPPED (this workflow, WF-Lambda-3) | WF-Lambda-3 (§3) | §1 (motivation), §2 (MLC vs SBDD), §4 (eval of method components), §5 (ablations), App. (closure-theorem), Fig 1 (MLC arch), Fig 2 (pipeline), Fig 3 (click reactions) |
| 4 | `04_evaluation.tex` | Evaluation (Round-13 100-pocket × 3-seed) | SHIPPED (this workflow, WF-Paper-Section-04) | WF-Paper-Section-04 (§4) | §3 (each method component), §5 (which knobs get ablated), §6 (limitations surfaced by this section), Fig 2 (pipeline), wf_3_citeonly_sota_table.csv |
| 5 | `05_ablation.tex` | Ablation Studies | SHIPPED (this workflow, WF-Paper-Section-05) | WF-Paper-1 (§5) | §3.2 (click rules, axis 4), §3.3 (MGP, axis 3), §3.4 (MCTS branching/top_k/CFG, axes 1+2+6); reports `wf_lambda1_build.md` (click ablation), `wf_lambda2e_compare/final.md` (homotype ablation), `wf2_final.md` (CFM ablation); `wf1_recon_cfm_train.md` (CFG endpoints) |
| 6 | `section_06_limitations.tex` | Limitations | SHIPPED (pre-existing) | WF-Paper-1 (§6, this workflow) | §1 (claim scope), §3 (architectural limits), §4 (empirical limits), §5 (untested knobs) |
| 7 | `section_07_future_work.tex` | Future Work | SHIPPED (pre-existing) | WF-Paper-1 (§7, this workflow) | §1 (open questions), §3 (Lambda × CFM coupling deferred), §6 (timeline) |
| App. | `appendix_betanf_semantics.tex` | Appendix: β-NF Operational Semantics | SHIPPED (pre-existing) | WF-Paper-3 (Appendix) | §3 (formal layer), §5 (closure theorem sketch) |

---

## Cross-reference map

### §1 → §3 (Method)
- §1 mentions the 9-layer MLC formalism → §3 §3.2 (the nine layers enumerated).
- §1 mentions 5-click rule set → §3 §3.4 (click registry table).
- §1 mentions MetalGeometryPrior → §3 §3.5 (geometry prior; Pt=4, Ru/Ir=6).
- §1 mentions MCTS with virtual loss + TT → §3 §3.6 (MCTS driver spec).

### §1 → §4 (Evaluation)
- §1 §"Empirical evidence" lists five MEASURED claims → §4 sub-sections mirror them:
  - §4.1 (Top-journal protocol) — three columns (hybrid, $\Lambda$-only, cite-only SOTA) wired.
  - §4.2 (Per-pocket metrics, Round-12 pilot, 10 rows) — DESIGN pending Round-12 pilot data.
  - §4.3 (Aggregate metrics across 10 × 3 cells) — DESIGN pending; cite-only column CITEDONLY.
  - §4.4 (Cite-only SOTA comparison) — wf_3_citeonly_sota_table.csv, 7 protocol-mismatch flags.
  - §4.5 (Hybrid vs $\Lambda$-only ablation) — wf_lambda1c_pilot_v3 5×3×2 cells.
  - §4.6 (Anticancer-specific metrics) — LogP/TPSA/RotB target ranges, Pt/Ru/Ir oxidation states.
  - §4.7 (5-click-rule vs CuAAC-only ablation) — wf_lambda1_build N=10 split 80/20.
  - §4.8 (Homotype vs Tanimoto orthogonality) — wf_lambda2e_compare 10-mol panel, MEASURED.
  - §4.9 (Failure-case analysis) — DESIGN pending Round-12 output.
  - §4.10 (Reporting protocol and honest framing summary) — 4 rules copied from wf_3_citeonly_sota.md §4.

### §1 → §6 (Limitations)
- §1 mentions the Pt-prior $\Delta = 0$ as an honest null → §6 §6.3 (Pt prior empirical non-claim).
- §1 mentions CFG decoder $0/96$ at unconstrained architecture → §6 §6.2 (architectural limit of CFM decoder).
- §1 mentions the 100-pocket sweep not yet executed → §6 §6.4 (computational scale limit).

### §1 → §7 (Future Work)
- §1 mentions Lambda × model-side coupling as deferred → §7 §7.1 (TODO-21 verbatim, four coupling directions).
- §1 mentions closure-theorem proof for 5-click productive space → §7 §7.2 (open theorem).
- §1 mentions fused-silu-MLP on gfx1101 wave64 → §7 §7.3 (engineering question).

### §2 (Related Work) → §1 / §3 / §6
- §2 catalogues the geometric SBDD family (DiffSBDD, Pocket2Mol, TargetDiff, MolDiff, DecompDiff, FLOWr) → §1 §"Existing approaches" enumerates the same.
- §2 surfaces seven protocol-mismatch flags → §6 §6.1 (where the mismatches become limitations for us).
- §2 cite-only SOTA column → §4 §4.6 (SOTA comparison table).

### §5 (Ablation) → §3
- §5 ablates each 9-layer component → §3 §3.2–§3.6 (component-by-component).
- §5 ablates click-rule set → §3 §3.4.
- §5 ablates MetalGeometryPrior weights → §3 §3.5.

### Appendix → §3 / §5
- Appendix (β-NF semantics) is referenced by §3 §3.3 (formal layer) and §5 §5.4 (closure-theorem sketch).

---

## Per-section citation footnotes (Digital Discovery style)

Per the journal's convention, supplementary references are URLs in footnotes rather than inline citations. Footnote labels in `01_intro.tex` are:

| Footnote | Source | Status |
|---|---|---|
| `footnote:cisp_history` | historical cisplatin FDA approval (1978) | citation placeholder |
| `footnote:metallodrug_review` | metallodrug clinical pipeline review | citation placeholder |
| `footnote:qsar_review` | QSAR methodology review | citation placeholder |
| `footnote:fep_metallo` | FEP for metallodrugs case study | citation placeholder |
| ~~`footnote:targetdiff`~~ | ~~TargetDiff (geometric SBDD)~~ | **REMOVED 2026-09-16** (WF-Pivot-A-Phase-2) |
| ~~`footnote:diffsbdd`~~ | ~~DiffSBDD (geometric SBDD)~~ | **REMOVED 2026-09-16** (WF-Pivot-A-Phase-2) |
| ~~`footnote:pocket2mol`~~ | ~~Pocket2Mol (geometric SBDD)~~ | **REMOVED 2026-09-16** (WF-Pivot-A-Phase-2) |
| ~~`footnote:moldiff`~~ | ~~MolDiff (geometric SBDD)~~ | **REMOVED 2026-09-16** (WF-Pivot-A-Phase-2) |
| ~~`footnote:decompdiff`~~ | ~~DecompDiff (geometric SBDD)~~ | **REMOVED 2026-09-16** (WF-Pivot-A-Phase-2) |
| ~~`footnote:flowr`~~ | ~~FLOWr (geometric SBDD)~~ | **REMOVED 2026-09-16** (WF-Pivot-A-Phase-2) |
| `footnote:wf_lambda1c` | `molmetal/reports/wf_lambda1c_pilot_v3/final.md` | URL — MEASURED |
| `footnote:wf_lambda1` | `molmetal/reports/wf_lambda1_pilot_v1/final.md` + CuAAC-only ablation | URL — MEASURED |
| `footnote:wf_lambda2e` | `molmetal/reports/wf_lambda2e_compare/final.md` | URL — MEASURED |
| `footnote:round11_parity` | `molmetal/reports/round11_engine_parity_n50.md` | URL — MEASURED |
| `footnote:wf_extra2` | `molmetal/reports/wf_extra2_batch/final.md` (REINVENT4 multiproperty) | URL — MEASURED |
| `footnote:round10_e2e` | `molmetal/reports/round10_e2e_pt_cfg_vina.md` | URL — MEASURED (honest null) |
| `footnote:lambda_coupling` | `TODO/pending/21_lambda_model_coupling.md` | URL — DEFERRED |

Footnote `bibitem` resolution will be handled in a separate pass once the bibliography is finalised.

---

## Bundle policy

- Each section is a complete LaTeX fragment intended to be `\input{}`-ed into `paper/main.tex`.
- No section depends on code that has not yet been written; §4 and §5 are SHIPPED as protocol/cell-scaffold documents; their data cells are \DESIGN{} pending the Round-12 pilot run, after which the cells will be filled in without re-tagging.
- All MEASURED-vs-PROJECTED labels are preserved verbatim from the upstream report that produced the number.
- Honest-framing is mandatory: any number cited in §1 must be traceable to a report under `molmetal/reports/` or a TODO under `TODO/pending/`; the per-cell traceability table is in `paper/traceability.csv` (to be generated by the §4 workflow).
