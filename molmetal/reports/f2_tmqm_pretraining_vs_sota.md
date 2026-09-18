# F2 tmQM pre-training versus SOTA baselines

**Purpose.** Paper-grade, cite-only comparison of the Lambda-tmQM coordination
pre-training result with named 2021–2026 baselines. This table is intended for
review and manuscript drafting; it does not claim a head-to-head re-run.

**Primary sources.** Lambda metrics are transcribed from
`/home/hugo/codes/try_triton_on_rocm/molmetal/reports/f2_tmqm_pretraining.md`.
SOTA facts are transcribed from
`/home/hugo/codes/try_triton_on_rocm/TODO/01_research/sota_papers_2025_2026.md`.
Protocol guidance follows
`/home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_protocol_aligned.md`.

## §1 Lambda-tmQM headline numbers

The pre-training corpus contains **21,617 Pt/Ru/Ir complexes** (7,854 Pt,
9,068 Ru, and 4,695 Ir before the two unusable-SMILES exclusions). The model is
a metal-aware DirectedMPNN with a coordination-number (CN/MND) head and a
metal Wiberg-bond-order head. The reported validation split is random 90/10,
seed 42, with 2,161 held-out complexes.

| Task | MAE | RMSE | R² | Evaluation set |
|---|---:|---:|---:|---|
| Coordination number (MND) | **0.132** | **0.206** | **0.9862** | 2,161 held-out Pt/Ru/Ir complexes |
| Total metal Wiberg bond order | 0.176 | 0.254 | 0.9239 | 2,161 held-out Pt/Ru/Ir complexes |

CN rounding gives 97.0% exact-match accuracy on the same held-out set. The
mean-predictor CN MAE is 1.365, so the reported MAE is a large improvement over
that reference, although CN is partly recoverable from graph degree. The source
report explicitly describes the CN target as close to a graph invariant and
identifies Wiberg bond order as the more chemically informative auxiliary task.

### §1.1 Scope of the headline

These numbers measure auxiliary-task prediction inside tmQM-style data. They
are evidence that the encoder learns a useful metal-centred representation;
they are not downstream cytotoxicity AUC, docking score, or de-novo-design
performance. Pt/Ru/Ir coverage is valuable for transfer, but MetalCytoToxDB has
Ru/Ir/Rh/Re/Os and no Pt, so Pt transfer remains an extrapolation.

## §2 SOTA baseline table

“NR” means the named source does not report a directly comparable CN-regression
metric. A dash in **metal classes** means that the source is not a
metal-specific coordination benchmark. Published task metrics are retained in
the notes column so reviewers can see what each baseline actually measures.

| paper | dataset | metal-specific? | CN MAE | R² | metal classes |
|---|---|---|---:|---:|---|
| Schütt et al. 2021, SchNet | QM9; molecular-property benchmark | **No** (QM9, flag) | NR | NR | none; organic QM9 |
| Hu et al. 2021, Equiformer | cited molecular 3-D/equivariant benchmark | No evidence of metal-specific CN pre-training | NR | NR | not reported |
| Rusanov et al. 2025, PlatinAI | 4,078 Pt complexes; 18,995 IC50 values; 4 cell lines | **Yes, Pt activity**; no CN pre-training metric | NR | NR | Pt |
| Krasnov et al. 2026, MetalCytoToxDB | 26,500 IC50 values; 7,050 complexes; 5 metals | **Yes, activity**; no CN pre-training metric | NR | NR | Ru, Ir, Rh, Re, Os |
| Tsinghua/Microsoft 2025, TransDiffSBDD | Uni-Mol 209M molecules + CrossDocked2020 100,000 complexes | No metal-specific pre-training reported | NR | NR | not reported; protein-pocket SBDD |

### §2.1 What is and is not comparable

SchNet’s QM9 row is deliberately flagged **NOT metal-specific**: QM9 does not
provide a Pt/Ru/Ir coordination-number task. Equiformer is included because it
is a cited 3-D/equivariant baseline, but the tracker does not provide a CN MAE
or R² that can be audited against Lambda. PlatinAI reports activity/design
outcomes, including superiority to cisplatin in resistant cell lines, rather
than CN regression. MetalCytoToxDB reports Ru ROC-AUC 0.81 and Ir ROC-AUC 0.73
for binary activity classification. TransDiffSBDD reports median Vina docking
of −9.37 kcal/mol and 83.9% success on its SBDD protocol; it is not a
metal-centred CN pre-training study.

### §2.2 Metric map for manuscript use

| Claim | Lambda evidence | Baseline evidence | Safe wording |
|---|---|---|---|
| CN prediction | MAE 0.132; RMSE 0.206; R² 0.9862 | No directly comparable CN values located | “Lambda reports a within-corpus CN result; cross-paper CN ranking is unavailable.” |
| Metal specificity | Pt/Ru/Ir, 21,617 complexes | Only PlatinAI and MetalCytoToxDB are metal-specific, but activity-focused | “Lambda is the only row here with explicit metal-centred CN supervision.” |
| Activity prediction | Not measured in F2 | MetalCytoToxDB Ru/Ir AUC 0.81/0.73; PlatinAI per-cell-line AUC | “F2 does not establish cytotoxicity performance.” |
| Structure generation | Not measured in F2 | TransDiffSBDD Vina −9.37; success 83.9% | “Docking-generation metrics are a separate task.” |

### §2.3 Source-path citation map

The following absolute paths are the audit trail for every repository-backed
claim in this comparison: Lambda/tmQM metrics —
`/home/hugo/codes/try_triton_on_rocm/molmetal/reports/f2_tmqm_pretraining.md`;
Schütt/SchNet, Hu/Equiformer, PlatinAI, MetalCytoToxDB, and TransDiffSBDD
tracker entries —
`/home/hugo/codes/try_triton_on_rocm/TODO/01_research/sota_papers_2025_2026.md`;
seven-flag protocol language —
`/home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_protocol_aligned.md`.

## §3 Honest caveats

- **Different data splits:** Lambda uses a random 90/10 split (seed 42), while
  PlatinAI uses scaffold splitting and MetalCytoToxDB uses a time-based split.
- **Different test sets:** Lambda tests 2,161 tmQM complexes; the activity and
  SBDD papers test cell-line records or protein pockets from unrelated corpora.
- **Different evaluation protocols:** MAE/RMSE/R² for continuous CN cannot be
  ordered against ROC-AUC, IC50 hit rate, Vina score, or SBDD success rate.
- **Metal coverage differs:** Lambda is metal-specific (Pt/Ru/Ir); several
  comparator rows are organic-only or do not disclose metal classes.
- **Model architectures differ:** Lambda uses a metal-aware DirectedMPNN;
  SchNet, Equiformer, gradient boosting, GNN fragment assembly, and diffusion
  generators have different inductive biases and outputs.
- **Small-data caveat:** 21,617 complexes is substantial for a focused corpus
  but small relative to Uni-Mol’s 209M-molecule pre-training; one random seed
  does not quantify uncertainty or transfer gains.

## §4 Cite-only protocol

All values in §1–§2 are **cited or transcribed**, not re-measured from external
checkpoints. The protocol uses the same seven-mismatch-flag style as
`/home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_protocol_aligned.md`.

1. **Task mismatch:** CN regression is not activity classification or docking.
2. **Corpus mismatch:** tmQM Pt/Ru/Ir is not QM9, MetalCytoToxDB, PlatinAI, or
   CrossDocked2020.
3. **Split mismatch:** random, scaffold, temporal, and pocket-held-out splits
   must remain labelled exactly as reported.
4. **Metric mismatch:** report MAE, RMSE, and R² only for continuous CN/BO;
   retain ROC-AUC, IC50, Vina, and success rate in separate columns or notes.
5. **Model-not-rerun:** baseline numbers are cite-only unless an identical
   implementation, checkpoint, and evaluation set are rerun by Mol-Metal.
6. **Architecture mismatch:** do not attribute differences to pre-training
   alone when the encoder, head, features, or objective also change.
7. **Sample-size/uncertainty mismatch:** include n, split seed, and error bars
   when available; do not compare a single seed to a multi-seed mean as equal.

### §4.1 Citation format requirements

- Use an author–year label in prose, e.g. **Balcells & Skjelstad (2020)**,
  **Rusanov et al. (2025)**, or **Krasnov et al. (2026)**.
- Follow each table or paragraph with an absolute repository source path in
  backticks, then the paper DOI/arXiv identifier when the tracker supplies one.
- Preserve the source’s units, sign convention, split name, and test-set size.
- Write **NR (not reported)** when a metric is absent; never back-calculate or
  fill a missing CN MAE from a different task.

### §4.2 Preprint flag requirements

- Mark ChemRxiv or arXiv-only evidence as **PREPRINT** beside the citation.
- Keep “version of record” and preprint dates distinct when both are supplied.
- Do not call a preprint result SOTA without naming its benchmark and protocol.
- For PlatinAI, label the ChemRxiv 2025 record **PREPRINT** and identify the
  four cell lines and scaffold split.
- For TransDiffSBDD, label the March 2025 record **PREPRINT** and identify
  CrossDocked2020/Uni-Mol provenance before quoting Vina or success rate.

### §4.3 Reproducibility checklist

Before promoting any comparison to a headline claim, record dataset version,
metal filter, number of usable SMILES, train/validation/test counts, random seed,
feature representation, model checkpoint, metric implementation, and rounding.
For Lambda, these fields are available in the F2 report and its companion JSON
artifacts; for SOTA rows, retain “cite-only” status until the same fields are
auditable.

**Bottom-line manuscript sentence:** Lambda-tmQM establishes a strong,
metal-specific auxiliary CN result (MAE 0.132, RMSE 0.206, R² 0.9862) on
21,617 Pt/Ru/Ir complexes, while the named SOTA papers do not report directly
comparable CN metrics; downstream activity and SBDD claims therefore require
separate, protocol-aligned experiments.
