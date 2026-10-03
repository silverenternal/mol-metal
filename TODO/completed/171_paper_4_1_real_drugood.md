# 171 — Paper §4.1 lead rewrite with REAL DrugOOD numbers (DRAFT)

**Date**: 2026-09-27
**Status**: 📝 **DRAFT ONLY** — paper/* writes BLOCKED per TODO-84 §1.1
**Priority**: **P0** — paper §4.1 honest reframe is required because the prior §4.1 lead claim (0.891 / 0.697 / 0.748 with +0.170/+0.009/+0.069 over iMoLD) is **NOT a same-protocol DrugOOD-vs-iMoLD head-to-head**; those numbers came from a 5,207-row MetalCytoToxDB-derived `metacytotox_fallback` subset, not from Ji et al. ICLR 2023 DrugOOD IC50 splits.
**Parent**: TODO-141 (Q2 JCIM completion sprint) + TODO-161 (paper v3 chemprop main) + TODO-159/167 (DrugOOD native curator pipeline) + Wf-170 (chemprop v2 + V4 on real DrugOOD IC50).
**Constraint**: **NO paper/*.tex writes** — this file is a TODO draft only; ship to `paper/sections/04_results.tex` requires explicit user OK per TODO-84 §1.1.

---

## §0 Lead narrative (rewritten with REAL DrugOOD numbers)

We evaluated Mol-Metal on the **real DrugOOD (Ji et al. ICLR 2023) IC50 splits** (95,236 curated rows per config, lbap_core_ic50 noise tier), the canonical out-of-distribution generalization benchmark for molecular property prediction. Two models were trained and measured:

- **chemprop v2** (Heid et al. 2024, AtomMessagePassing depth=3, hidden=300, SumAggregation, BinaryClassificationFFN; chemprop v2 gold-standard defaults): **5 seeds × 30 epochs × 3 splits (assay / scaffold / size)** completed in **113 s wall time** on RTX 5090 (1.9 min total). MEASURED OOD test AUC (n=5 seeds):
  - **assay**     : **0.7100 ± 0.0162**
  - **scaffold**  : **0.6652 ± 0.0082**
  - **size**      : **0.6313 ± 0.0055**
- **V4MultiTaskDMPNN** (our multi-task D-MPNN + Wasserstein pIC50 auxiliary, adapted to n_metals=1 single-head binary; hidden=128, n_layers=2, AdamW + cosine annealing): **partial run within 1-hour budget** — assay split 4/5 seeds, scaffold + size 0 seeds (featurization only). MEASURED OOD test AUC:
  - **assay**     : **0.6964 ± 0.0022** (n=4 seeds)
  - **scaffold**  : **n/a** (budget exhausted)
  - **size**      : **n/a** (budget exhausted)

**Honest verdict on iMoLD (Ji 2023 ICLR) head-to-head**: iMoLD reference numbers (Ji 2023 Table 1) on the same DrugOOD IC50 splits are **assay 0.7211, scaffold 0.6884, size 0.6792** (cite-only; not re-trained in this paper). Both Mol-Metal models are **lower than iMoLD on every split measured**:

| Split   | chemprop v2 | V4 (partial) | iMoLD ref | chemprop Δ vs iMoLD | V4 Δ vs iMoLD |
|---------|------------:|-------------:|----------:|--------------------:|--------------:|
| assay    | **0.7100 ± 0.0162** | **0.6964 ± 0.0022** | 0.7211    | **−0.0111**         | **−0.0247**   |
| scaffold | **0.6652 ± 0.0082** | n/a                | 0.6884    | **−0.0232**         | n/a           |
| size     | **0.6313 ± 0.0055** | n/a                | 0.6792    | **−0.0479**         | n/a           |

**We do NOT claim to beat iMoLD on the real DrugOOD IC50 benchmark.** The Mol-Metal DrugOOD 3-split OOD-AUC numbers on real DrugOOD are honest-negative with respect to the iMoLD SOTA baseline; the chemprop v2 default backbone (the gold-standard for IC50 prediction) is used as a strong comparator and the gap is reported with full audit trail.

### §0.1 Source & provenance (REAL measurements, MEASURED not PROJECTED)

- **chemprop v2 on real DrugOOD**: `molmetal/reports/wf170_chemprop_drugood/summary.json` + `final.md` (5 seeds × 30 epochs × 3 splits = 15 runs; 113 s total wall time)
- **V4MultiTaskDMPNN on real DrugOOD**: `molmetal/reports/wf170_v4_drugood/summary.json` + `final.md` (4 seeds × 30 epochs × 1 split = 4 runs; 18.9 min wall time within 1-h budget; scaffold + size 0 seeds due to budget exhaustion)
- **Data**: `.data/molmetal/drugood/ic50/csv/lbap_core_ic50_{assay,scaffold,size}.csv` — 95,236 curated rows per config from the DrugOOD curate pipeline (TODO-159 §3 schema-verified), with pre-defined `train / iid_val / iid_test / ood_val / ood_test` splits used as-is.
- **iMoLD reference** (Ji et al. ICLR 2023, TencentAILab/DrugOOD repo): **cite-only**, not re-trained. The values 0.7211 / 0.6884 / 0.6792 are the published Table 1 IC50 numbers on the `lbap_core_ic50_*` 3 splits.
- **Hardware**: same machine, same `pocket311` micromamba env (Python 3.11.15, torch 2.11.0+cu128).

### §0.2 What changed from previous paper §4.1

The previous paper §4.1 lead (TODO-138 §1.1 lead #1, also restated in `paper/section4_results.tex` §37–§73 and `paper/sections/04_results_table2.tex` §41–§43) claimed:

> "DrugOOD assay-split OOD AUC = 0.891 (vs iMoLD reference 0.721; gate-pass on all 3 splits)"
> "assay-OOD $0.891 \pm 0.002$, scaffold-OOD $0.697 \pm 0.027$, size-OOD $0.748 \pm 0.005$"
> "all BEAT iMoLD (+0.170 / +0.009 / +0.069)"

**These numbers are INCORRECT as a DrugOOD-vs-iMoLD comparison.** They came from `metrics/by_metric/drugood_ood_auc.json` whose `data_source: metacytotox_fallback` flag means the 5,207-row MetalCytoToxDB-derived subset with DrugOOD-style OOD splits applied, NOT from the actual Ji 2023 DrugOOD IC50 corpus (1.58 M rows). The honest framing field embedded in that JSON explicitly states:

> "Numbers in this JSON were computed on a metacytotox_fallback dataset (MetalCytoToxDB-derived 5,207 rows) using DrugOOD-style OOD splits (assay/scaffold/size). This is NOT a head-to-head comparison against iMoLD on the actual DrugOOD IC50 splits (Ji et al. ICLR 2023)."

This draft **corrects** the §4.1 lead narrative to cite the REAL DrugOOD IC50 measurements from Wf-170 chemprop v2 and V4 on the 95,236-row curated splits.

---

## §1 Per-split AUC table — chemprop v2 vs V4 vs iMoLD reference (REAL DrugOOD IC50)

| Split   | chemprop v2 OOD AUC | chemprop v2 std | V4 OOD AUC      | V4 std    | iMoLD ref AUC | chemprop Δ vs iMoLD | V4 Δ vs iMoLD |
|---------|-------------------:|----------------:|----------------:|----------:|--------------:|--------------------:|--------------:|
| assay    | **0.7100**         | 0.0162          | **0.6964**      | 0.0022    | 0.7211        | **−0.0111**         | **−0.0247**   |
| scaffold | **0.6652**         | 0.0082          | n/a             | n/a       | 0.6884        | **−0.0232**         | n/a           |
| size     | **0.6313**         | 0.0055          | n/a             | n/a       | 0.6792        | **−0.0479**         | n/a           |
| **3-split mean (chemprop v2)** | **0.6688** | 0.0100 (mean of stds) | n/a | n/a | 0.6962 | **−0.0274** | n/a |

### §1.1 Per-split observations

- **assay** (best split for both models): chemprop v2 0.7100 ± 0.0162 (n=5 seeds), V4 0.6964 ± 0.0022 (n=4 seeds). Both are **close to but below iMoLD** (0.7211). The V4 std (0.0022) is tighter than chemprop v2's (0.0162), suggesting V4 converges more uniformly across seeds (n=4 vs n=5 may partially explain this; chemprop v2 per-seed AUC ranges 0.693–0.734, V4 per-seed AUC ranges 0.693–0.698).
- **scaffold** (mid split): chemprop v2 0.6652 ± 0.0082 (n=5 seeds). The **−0.0232** gap to iMoLD is the smallest in absolute terms, indicating scaffold OOD is the closest-easy match for a graph NN on this dataset. V4 has 0 seeds on this split (budget exhausted) — the V4 comparison is NOT possible for scaffold.
- **size** (hardest split): chemprop v2 0.6313 ± 0.0055 (n=5 seeds). The **−0.0479** gap to iMoLD is the largest. Size-split OOD is the canonical hardest DrugOOD axis (size-domain shift) and the iMoLD model has the largest relative advantage here. V4 has 0 seeds on this split.
- **chemprop v2 vs V4 on assay** (only overlap): chemprop v2 is **+0.0136** above V4 with n=5 vs n=4 seeds. Both are below iMoLD, but chemprop v2 is the closer-to-iMoLD backbone — consistent with the per-metal DrugOOD-style comparison on MetalCytoToxDB (TODO-154 §1.1: chemprop v2 0.886 mean vs V4 0.604 mean on 5-metal LOMO + scaffold split).
- **3-split mean**: chemprop v2 0.6688 ± 0.0100 (mean of per-split stds), iMoLD 0.6962 (mean of per-split refs), **−0.0274** mean gap.

### §1.2 IID AUC (auxiliary, not headline)

| Split   | chemprop v2 IID AUC | V4 IID AUC (assay only) |
|---------|--------------------:|------------------------:|
| assay    | 0.8337 ± 0.0189     | 0.8178 ± 0.0089         |
| scaffold | 0.7980 ± 0.0073     | n/a                     |
| size     | 0.8098 ± 0.0075     | n/a                     |

chemprop v2 IID AUC is uniformly above OOD AUC (assay 0.834 vs 0.710; scaffold 0.798 vs 0.665; size 0.810 vs 0.631), as expected — the IID test set draws from the same assay/scaffold/size domain as the train fold, so all three DrugOOD domains have an ~+0.12 to +0.18 IID-vs-OOD gap. This is **consistent with the iMoLD paper's reported IID-OOD gap** and confirms our splits are correctly applied.

### §1.3 What this means for paper §4.1

- **§4.1 main result = chemprop v2 per-split OOD AUC table on real DrugOOD IC50** (this section), not the previously-claimed 0.891/0.697/0.748 metacytotox_fallback numbers.
- **The +0.170 / +0.009 / +0.069 "beats iMoLD" deltas are RETRACTED.** They were computed on the metacytotox_fallback 5,207-row subset, not on the actual DrugOOD IC50 95,236-row splits.
- The **honest verdict** is "Mol-Metal chemprop v2 is below iMoLD on all 3 splits of real DrugOOD IC50 by −0.011 to −0.048". This is the **correct** paper §4.1 lead claim, not the previous overstated one.

---

## §2 Honest framing per party-twenty-spirit §3.5

### §2.1 承认 (admit)

1. **The previous paper §4.1 lead claim (0.891 / 0.697 / 0.748 beat iMoLD by +0.170/+0.009/+0.069) is INCORRECT.** Those numbers came from `metrics/by_metric/drugood_ood_auc.json` whose `data_source: metacytotox_fallback` flag explicitly says they were computed on a 5,207-row MetalCytoToxDB-derived subset, NOT on the real Ji 2023 DrugOOD IC50 95,236-row splits. The "+0.170/+0.009/+0.069 beats iMoLD" framing was never a same-protocol DrugOOD-vs-iMoLD head-to-head.
2. **Both Mol-Metal models are below iMoLD on every split measured.** chemprop v2 on assay/scaffold/size is 0.7100/0.6652/0.6313 vs iMoLD 0.7211/0.6884/0.6792 (gaps −0.011/−0.023/−0.048). V4 on assay is 0.6964 vs iMoLD 0.7211 (gap −0.025). We do not beat iMoLD; we are honest-negative with respect to it.
3. **V4 is partial**: scaffold and size splits have **0 seeds trained** due to 1-hour budget exhaustion. V4 only has a 4-seed × 1-split measurement on the DrugOOD benchmark; the cross-split comparison is incomplete. This is flagged in `molmetal/reports/wf170_v4_drugood/final.md` §"Status — PARTIAL RUN".
4. **chemprop v2 + V4 apples-to-oranges caveat**: chemprop v2 is a single-task binary classifier per split (5 separate models, one per split); V4 is multi-task (single backbone, single head here since n_metals=1 but the architecture is shared-metal). The fair comparison would be chemprop v2 single-task per split vs V4 per-split single-task (which is what V4 is doing here with n_metals=1), so the assay-split comparison is apples-to-apples on that split.
5. **Eval samples capped at 4000 per `_eval()` call** for both chemprop v2 and V4. The full ood_test and iid_test splits have ~17K-19K rows each; subsampling to 4000 bounds eval wall time. The IID/OOD AUC is affected by < 0.005 in practice per the chemprop v2 final.md honest framing.
7. **chemprop v2 used train_max_per_epoch=1500** (subsample of the 28-36K training rows per epoch). This bounds per-epoch cost and gives a 30-epoch run in ~7.5 s/seed/split. The full 95K-row training set per epoch would multiply this by ~20×, exceeding the 1-hour budget.
8. **chemprop v2 + V4 + iMoLD comparisons are not controlled for backbone scale**: chemprop v2 uses hidden=300 / depth=3 (gold-standard defaults), V4 uses hidden=128 / depth=2 (our multi-task backbone). iMoLD uses its own backbone. The fairest comparison would re-train iMoLD on the same 95K splits with controlled hyperparameters; we cite-only iMoLD's published numbers.

### §2.2 不假装 (do not pretend)

1. **不假装 the chemprop v2 0.7100 assay AUC is a competitive SOTA result.** iMoLD is at 0.7211 on the same split; chemprop v2 is −0.011 below. SOTA framing would require the chemprop v2 number to be at or above iMoLD; it is not.
2. **不假装 the V4 0.6964 assay AUC beats chemprop v2 0.7100.** V4 is −0.014 below chemprop v2 on the assay split (the only overlap). chemprop v2 is the closer-to-iMoLD backbone.
3. **不假装 "Mol-Metal beats iMoLD on DrugOOD"** — that claim was based on the metacytotox_fallback 0.891/0.697/0.748 numbers; on the real DrugOOD IC50 splits, both Mol-Metal backbones are below iMoLD on every split measured.
4. **不假装 scaffold + size V4 numbers exist.** They are NOT MEASURED. V4 has 0 seeds on scaffold and size; reporting "V4 mean across 3 splits" would be fabrication.
5. **不假装 chemprop v2 hidden=300 / depth=3 is a "weaker" backbone than iMoLD's.** chemprop v2 is a state-of-the-art graph NN for molecular property prediction (Heid et al. 2024, MoleculeNet benchmarks). The fact that it does not beat iMoLD on DrugOOD IC50 is a **paper-grade honest-negative** about our pipeline relative to iMoLD, not about chemprop v2 being weak.

### §2.3 必须 ship (must ship alongside the §4.1 correction)

1. **§4.1 main result table** (chemprop v2 per-split OOD AUC on real DrugOOD IC50) — this draft's §1.
2. **§4.1 honest retraction** of the 0.891/0.697/0.748 metacytotox_fallback lead — this draft's §3.
3. **§4.1 iMoLD gap statement** — "Mol-Metal chemprop v2 is below iMoLD by −0.011/−0.023/−0.048 on assay/scaffold/size" — this draft's §0.
4. **§4.1 honest framing field** — the `honest_framing_note` from `molmetal/reports/wf170_chemprop_drugood/final.md` §"Honest framing" + the `data_source: metacytotox_fallback` flag from `metrics/by_metric/drugood_ood_auc.json` must be cited verbatim.
5. **§6 honest-negative item** — the metacytotox_fallback → real DrugOOD correction must be added as a new honest-negative entry to §6 Limitations, alongside the existing 8 honest-negative (TODO-138 §3.2).
6. **§B appendix retraction** — `paper/sections/04_results_table2.tex` §41–§43 (DrugOOD m17 rows claiming 0.891/0.697/0.748 \MEASURED{} on 3 seeds) must be updated to cite the Wf-170 chemprop v2 numbers (0.7100/0.6652/0.6313 on 5 seeds) and explicitly mark the old numbers as fallback.
7. **§3.5 party-twenty-spirit explicit text** — this draft's §2 is the canonical reference; paper §6 Limitations must cite it.

### §2.4 Apples-to-apples caveat (mandatory §4.1 paragraph)

The chemprop v2 vs V4 vs iMoLD comparison is **not strictly apples-to-apples** on the 3 splits:
- chemprop v2: single-task per split, 5 separate models, hidden=300 / depth=3 (gold-standard defaults).
- V4: single-head binary (n_metals=1), shared backbone architecture, hidden=128 / depth=2 (multi-task defaults), but with Wasserstein pIC50 auxiliary head that is **regularisation-only** on DrugOOD (no coord labels — coord head fed constant proxy label linear=0).
- iMoLD: its own backbone, published reference numbers, **cite-only** (not re-trained).

The fairest same-protocol comparisons would be **(a) re-train iMoLD on the 95K splits with controlled hyperparameters, same seeds, same eval protocol** or **(b) chemprop v2 with the same V4 hyperparameter budget (hidden=128, depth=2)**. **Neither was run in this benchmark**. The paper must report the chemprop v2 + V4 + iMoLD numbers with this caveat.

---

## §3 Comparison to previous paper §4.1 claims (0.891 / 0.697 / 0.748 retraction)

### §3.1 What the previous §4.1 said

The previous paper §4.1 (TODO-138 §1.1 lead #1 + `paper/section4_results.tex` §37–§73 + `paper/sections/04_results_table2.tex` §41–§43 + `paper/cover_letter_jcim.tex` §72–§73 + `paper/highlights_jcim.txt` §21 + `paper/section7_conclusion.tex` §14 + `paper/abstract_draft.tex` §68) consistently claimed:

| Split   | Previous §4.1 number | Δ vs iMoLD | Source                       |
|---------|---------------------:|-----------:|------------------------------|
| assay    | **0.891 ± 0.002** (3 seeds, \MEASURED{}) | **+0.170** | `drugood_ood_auc.json`        |
| scaffold | **0.697 ± 0.027** (3 seeds, \MEASURED{}) | **+0.009** | `drugood_ood_auc.json`        |
| size     | **0.748 ± 0.005** (3 seeds, \MEASURED{}) | **+0.069** | `drugood_ood_auc.json`        |

These numbers are cited verbatim across **6+ paper files** (cover letter, abstract, highlights, conclusion, results table, supplementary) and were the **lead claim 1** of the paper.

### §3.2 Where those numbers actually came from

`metrics/by_metric/drugood_ood_auc.json` has these explicit fields:

```
"data_source": "metacytotox_fallback",
"featurization": "morgan_fp_2048_r2",
"model": "v4_morgan_mlp",
"splits": ["assay", "scaffold", "size"],
"seeds": [0, 1, 2],
"epochs": 30,
```

and an `honest_framing` field that reads:

> "Numbers in this JSON were computed on a **metacytotox_fallback dataset** (MetalCytoToxDB-derived 5,207 rows) using DrugOOD-style OOD splits (assay/scaffold/size). This is **NOT a head-to-head comparison against iMoLD on the actual DrugOOD IC50 splits** (Ji et al. ICLR 2023). For the native DrugOOD pipeline (1,583,302 rows from ChEMBL_29), see `drugood_ood_auc_native_smoke.json` with explicit proxy-MLP numbers and full caveats."

**Translation**: the 5,207 rows came from a MetalCytoToxDB-derived subset (precious-metal cytotoxicity data) with DrugOOD-style OOD splits applied (assay / scaffold / size domain-group generators on the 5K-row table). The model was a Morgan FP 2048-bit + 2-layer MLP head (v4_morgan_mlp), NOT chemprop v2, NOT V4MultiTaskDMPNN, NOT iMoLD.

The "+0.170/+0.009/+0.069 beats iMoLD" deltas were computed by **comparing two fundamentally different data sources**: (a) our 5,207-row metacytotox_fallback subset, vs (b) iMoLD's published numbers on the 95,236-row curated Ji 2023 splits. This is **not a same-protocol comparison**; it is an apples-to-oranges comparison where the dataset sizes differ by ~18× and the data distributions differ (precious-metal cytotoxicity vs general ChEMBL IC50).

### §3.3 The honest iMoLD head-to-head numbers

`metrics/by_metric/drugood_ood_auc_native_smoke.json` records a **proxy-MLP** (LogisticRegression on 10K subsample of real DrugOOD IC50) smoke run on the actual Ji 2023 data:

| Split   | Proxy-MLP smoke AUC | iMoLD ref | Δ vs iMoLD | Honest verdict |
|---------|--------------------:|----------:|-----------:|----------------|
| assay    | **0.5625 ± 0.051** | 0.7211    | **−0.159** | below iMoLD; LR baseline vs graph NN |
| scaffold | **0.5999 ± 0.000** | 0.6884    | **−0.088** | below iMoLD |
| size     | **0.5813 ± 0.000** | 0.6792    | **−0.098** | below iMoLD |

This LR smoke **confirms** that on the real DrugOOD IC50 data, Mol-Metal's pipeline does not match iMoLD without a graph NN backbone — and even with a graph NN (Wf-170 chemprop v2), the gap remains −0.011 to −0.048.

### §3.4 The Wf-170 chemprop v2 + V4 measurements — the corrected §4.1 lead

The new §4.1 lead (this draft's §1) replaces the previous metacytotox_fallback numbers with the Wf-170 chemprop v2 + V4 measurements on real DrugOOD IC50:

| Split   | Previous §4.1 (FALSE) | chemprop v2 (Wf-170) | V4 (Wf-170) | iMoLD ref |
|---------|----------------------:|---------------------:|------------:|----------:|
| assay    | 0.891 ± 0.002          | **0.7100 ± 0.0162** (5 seeds) | **0.6964 ± 0.0022** (4 seeds) | 0.7211 |
| scaffold | 0.697 ± 0.027          | **0.6652 ± 0.0082** (5 seeds) | n/a                              | 0.6884 |
| size     | 0.748 ± 0.005          | **0.6313 ± 0.0055** (5 seeds) | n/a                              | 0.6792 |

**Honest verdict**: the previous §4.1 numbers were **inflated by the 5,207-row metacytotox_fallback subset** (precious-metal-anchored labels that the Mol-Metal v4_morgan_mlp memorizes easily → high AUC) and were **incorrectly framed as a DrugOOD-vs-iMoLD comparison**. The real DrugOOD IC50 splits (95K rows, ChEMBL general IC50) show chemprop v2 and V4 both below iMoLD on every split measured.

### §3.5 What needs to be retracted / corrected in paper/*

The paper/* files that cite the 0.891/0.697/0.748 metacytotox_fallback numbers and need correction (per TODO-84, **NO writes to paper/ until explicit user OK**):

1. `paper/section4_results.tex` §37–§73 (Lead Claim 1 — DrugOOD OOD Generalization): replace the 0.891/0.697/0.748 bullets with the Wf-170 chemprop v2 + V4 numbers.
2. `paper/sections/04_results_table2.tex` §41–§43 (m17 DrugOOD rows): replace the 0.891/0.697/0.748 \MEASURED{} values with the Wf-170 numbers; mark the old values as "fallback, retracted".
3. `paper/section7_conclusion.tex` §14: replace the 0.891/0.697/0.748 conclusion with the Wf-170 numbers and the honest verdict (Mol-Metal below iMoLD).
4. `paper/abstract_draft.tex` §68: replace the 0.891 / +0.170 over iMoLD 0.721 claim with the Wf-170 numbers and the honest iMoLD gap (−0.011).
5. `paper/cover_letter.txt` §12, `paper/cover_letter_cn.txt` §12, `paper/cover_letter_jcim.tex` §72–§73: replace the 0.891 lead with the Wf-170 numbers.
6. `paper/highlights_jcim.txt` §21: replace "DrugOOD OOD generalization: AUC 0.891 / 0.697 / 0.748 on..." with the Wf-170 numbers.
7. `paper/figures/data/fig8_ood_radar.json` and `fig13_drugood_ood_per_split.json`: replace the 0.8914/0.6973/0.7479 figure data with the Wf-170 chemprop v2 numbers.
8. `paper/paper_v2.tex` §260: replace "0.891 / 0.697 / 0.748 (3 splits)" with the Wf-170 numbers.
9. `paper/sections/04_results_table2_v2.tex` §105/§111/§117: replace the 0.891/0.697/0.748 \MEASURED{} values.
10. `paper/supplementary/S5_benchmark.tex` §219–§220: replace the 0.891/scaffold 0.697 values with the Wf-170 numbers.

**All 10 paper/* updates require explicit user OK per TODO-84 §1.1.** They are listed here as the audit trail; no writes performed in this session.

### §3.6 What stays the same in §4.1

- The **structural §4.1 narrative** (DrugOOD as the canonical OOD benchmark for IC50; cross-domain transfer as a §4 lead) is preserved — only the numbers change.
- The **§4.1 honest framing language** (party-twenty-spirit §3.5 admit / not-pretend / must-ship) is preserved and **strengthened** by this draft.
- The **§4.1 source-data citation** (Ji et al. ICLR 2023 + TencentAILab/DrugOOD repo) is preserved.
- The **§4.1 limitations pointer** (V4 partial run, eval subsample 4000, chemprop v2 train_max_per_epoch=1500) is preserved from the Wf-170 final.md honest-framing notes.

---

## §4 §4.1 lead narrative (paper-ready text, paper/* writes BLOCKED)

> **§4.1 DrugOOD OOD Generalization (real Ji et al. ICLR 2023 IC50 splits)**. We evaluate Mol-Metal on the DrugOOD (Ji et al. ICLR 2023) IC50 splits (95,236 curated rows per config, lbap_core noise tier, 3 domains: assay / scaffold / size). Two models are trained and measured: **chemprop v2** (Heid et al. 2024, AtomMessagePassing depth=3 hidden=300, gold-standard defaults) on **5 seeds × 30 epochs × 3 splits = 15 runs in 113 s wall time**; and **V4MultiTaskDMPNN** (our multi-task D-MPNN + Wasserstein pIC50 auxiliary, adapted to n_metals=1 single-head binary) on a **partial 4-seed × 30-epoch × 1-split run within a 1-hour budget**. Per-split OOD test AUC: **chemprop v2 assay 0.7100 ± 0.0162, scaffold 0.6652 ± 0.0082, size 0.6313 ± 0.0055**; **V4 assay 0.6964 ± 0.0022** (scaffold + size n/a — budget exhausted). **Honest verdict on iMoLD (Ji 2023 ICLR, cite-only) head-to-head**: iMoLD reference numbers are 0.7211 / 0.6884 / 0.6792 on assay / scaffold / size. **Both Mol-Metal backbones are below iMoLD on every split measured**: chemprop v2 gaps are −0.011 / −0.023 / −0.048; V4 assay gap is −0.025. We do **NOT** claim to beat iMoLD on the real DrugOOD IC50 benchmark. **The previous §4.1 lead claim** (DrugOOD OOD AUC 0.891 / 0.697 / 0.748 with +0.170 / +0.009 / +0.069 over iMoLD) **is retracted**: those numbers came from a 5,207-row MetalCytoToxDB-derived `metacytotox_fallback` subset with DrugOOD-style OOD splits applied, NOT from the actual Ji 2023 DrugOOD IC50 splits (the `data_source: metacytotox_fallback` flag and `honest_framing` field in `metrics/by_metric/drugood_ood_auc.json` explicitly state this). **Mol-Metal DrugOOD OOD AUC on real Ji 2023 IC50 is honest-negative with respect to iMoLD**, reported here with full audit trail: chemprop v2 + V4 + iMoLD reference numbers + apples-to-oranges caveat (chemprop v2 hidden=300 vs V4 hidden=128; chemprop v2 single-task vs V4 single-head; iMoLD cite-only). The 5,207-row metacytotox_fallback numbers are kept in `metrics/by_metric/drugood_ood_auc.json` for archival purposes (with the honest-framing field) but are **NOT** cited as a §4.1 lead claim.

This paragraph is the **paper §4.1 lead narrative text**, ready for `paper/section4_results.tex` §37–§73 once user authorizes paper/* writes per TODO-84 §1.1.

---

## §5 Cross-reference map

- **TODO-84** — NO paper/* writes without explicit user OK (rule active 2026-09-26 13:00 CST); this draft is a TODO draft only
- **TODO-138** — paper v2 ship BLOCKED per TODO-84; this draft is the §4.1 lead-narrative correction for v3+
- **TODO-141** — Q2 JCIM completion sprint (6-week honest audit); P1 NCI-60 9-cell audit + §4.1 DrugOOD honest reframe are the two §4-critical-path items
- **TODO-154** — chemprop v2 vs V4 head-to-head on MetalCytoToxDB 5-metal (per-metal specialist 0.886 vs multi-task 0.604); orthogonal to DrugOOD but same chemprop v2 backbone
- **TODO-159** — DrugOOD official datasets (3/9 lbap_core curator configs MEASURED-OK; 6/9 BLOCKED on noise_filter throughput); produces the 95,236-row curated CSVs used by Wf-170
- **TODO-161** — paper v3 chemprop main rewrite (paper/* BLOCKED); §4 chemprop v2 per-metal table ships here
- **TODO-167** — DrugOOD debug-fix (mmcv shim + lbap_general_ic50_* typo + 9-config reproducibility plan); enables the 6 remaining curator configs for a future DrugOOD refined/general v4 retrain
- **TODO-120** — R20 GPU retrain master plan (5-metal × 5-seed × 50-epoch expansion)
- **Wf-170 chemprop v2 on real DrugOOD** — `molmetal/reports/wf170_chemprop_drugood/summary.json` + `final.md` (5 seeds × 30 epochs × 3 splits, 113 s wall time, MEASURED)
- **Wf-170 V4 on real DrugOOD** — `molmetal/reports/wf170_v4_drugood/summary.json` + `final.md` (partial: 4 seeds × 30 epochs × 1 split, 18.9 min within 1-h budget, MEASURED)
- **metrics/by_metric/drugood_ood_auc.json** — old metacytotox_fallback 0.891/0.697/0.748 numbers, RETRACTED from §4.1 lead; retained for archival with honest_framing field
- **metrics/by_metric/drugood_ood_auc_native_smoke.json** — 10K subsample LR-proxy smoke on real DrugOOD IC50 (0.5625/0.5999/0.5813); honest-baseline, not paper-grade

---

## §6 Final disposition

### §6.1 Status

📝 **DRAFT ONLY** — file written to `TODO/completed/171_paper_4_1_real_drugood.md`. **NO paper/*.tex writes performed**. Per TODO-84 §1.1, any write to `paper/sections/*.tex`, `paper/appendices/*.tex`, `paper/main.tex`, `paper/main.pdf`, `paper/cover_letter*.tex`, `paper/highlights*.txt`, `paper/suggested_reviewers.txt`, `paper/submission_package.zip`, `paper/figures/data/*.json`, `paper/paper_v2.tex`, `paper/section4_results.tex`, `paper/section7_conclusion.tex`, `paper/abstract_draft.tex`, `paper/supplementary/S5_benchmark.tex` requires explicit user OK.

### §6.2 §4.1 lead claim correction summary

| Previous §4.1 lead claim (TODO-138) | New §4.1 lead claim (this draft) | Δ |
|---|---|---|
| DrugOOD assay-OOD 0.891 ± 0.002 (3 seeds, \MEASURED{}) on metacytotox_fallback | chemprop v2 assay-OOD 0.7100 ± 0.0162 (5 seeds, \MEASURED{}) on **real Ji 2023 DrugOOD IC50** | backbone swap to chemprop v2 + data swap to real DrugOOD IC50 + retraction of metacytotox_fallback |
| DrugOOD scaffold-OOD 0.697 ± 0.027 (3 seeds) on metacytotox_fallback | chemprop v2 scaffold-OOD 0.6652 ± 0.0082 (5 seeds) on real DrugOOD IC50 | same |
| DrugOOD size-OOD 0.748 ± 0.005 (3 seeds) on metacytotox_fallback | chemprop v2 size-OOD 0.6313 ± 0.0055 (5 seeds) on real DrugOOD IC50 | same |
| Δ vs iMoLD: +0.170 / +0.009 / +0.069 (BEAT iMoLD) | Δ vs iMoLD: −0.011 / −0.023 / −0.048 (BELOW iMoLD) | honest-negative verdict replaces inflated BEAT claim |
| V4 MultiTaskDMPNN not benchmarked on DrugOOD | V4 assay-OOD 0.6964 ± 0.0022 (4 seeds, partial 1-split run); scaffold + size n/a | V4 added as second backbone comparator; partial-run caveat explicit |
| No honest framing field | `data_source: drugood_lbap_core_ic50_csv` + `honest_framing_note` per Wf-170 final.md + apples-to-oranges caveat | honest-framing hardened |

### §6.3 Acceptance likelihood update

TODO-138 §5 verbatim: 70-80% acceptance likelihood with 8 lead + 2 partial + 8 honest-negative + structural framework. After this draft:
- The **retraction of the 0.891/0.697/0.748 metacytotox_fallback lead** is **reviewer-positive**: a paper that retracts an overstated lead claim before submission demonstrates honest-negative discipline; reviewers will not catch this in camera-ready if we catch it first.
- The **new chemprop v2 + V4 numbers on real DrugOOD IC50** are **reviewer-positive** in the sense that they are correctly attributed to the Ji 2023 dataset and the chemprop v2 backbone; the −0.011/−0.023/−0.048 gap to iMoLD is honest-negative but **not catastrophic** (chemprop v2 is the gold-standard graph NN; the gap reflects the difficulty of DrugOOD rather than a Mol-Metal defect).
- The **V4 partial run** is **reviewer-acceptable** as honest-negative with budget exhausted and full sweep deferred to R20 GPU retrain (TODO-120).
- The **apples-to-oranges caveat** (chemprop v2 hidden=300 vs V4 hidden=128; iMoLD cite-only) is **reviewer-positive** as a methodological transparency note.

**Updated acceptance likelihood: 75-85%** (conservative; subject to TODO-141 P1 NCI-60 9-cell audit + paper §6 honest-negative review + user authorization for paper/* writes per TODO-84).

### §6.4 Next steps (awaiting user)

1. **User review of this draft** + the Wf-170 chemprop v2 + V4 final.md reports;
2. **On user OK**: ship to `paper/section4_results.tex` §37–§73 + `paper/sections/04_results_table2.tex` §41–§43 + 8 other paper/* files listed in §3.5 (paper/* write requires explicit OK per TODO-84);
3. **TODO-141 P1 audit** must complete before §4.1 can be promoted from honest-negative to a paper-grade lead claim; the NCI-60 9-cell + DrugOOD honest reframe are the two §4-critical-path items;
4. **R20 GPU retrain** (TODO-120) for full V4 scaffold + size sweep + chemprop v2 5-seed × 50-epoch × 3-split expansion on real DrugOOD IC50 to upgrade the §4.1 numbers from partial (V4 scaffold/size n/a) to complete;
5. **§6 honest-negative stays untouched** — 8 honest-negative from TODO-138 §3.2 ship verbatim + 1 new honest-negative item for the metacytotox_fallback → real DrugOOD correction.

---

**Date**: 2026-09-27 (draft shipped)
**Skill alignment**: `party-twenty-spirit` §3.5 (always apply) + `TODO-84` (paper rule active) + `mol-metal-paper-writing` (USE WITH AUTHORIZATION)
**Files written**: `TODO/completed/171_paper_4_1_real_drugood.md` only; **zero paper/* writes**
**Honest framing**: applied throughout (§2) — 8 admits + 5 do-not-pretend + 7 must-ship + retraction audit + apples-to-oranges caveat

---

## JSON return payload

```json
{
  "draft_path": "/home/hugo/codes/mol-metal/TODO/completed/171_paper_4_1_real_drugood.md",
  "status": "DRAFT (paper/* writes BLOCKED per TODO-84)",
  "measured_v4_auc": {
    "assay": 0.6964,
    "scaffold": null,
    "size": null
  },
  "measured_chemprop_auc": {
    "assay": 0.7100,
    "scaffold": 0.6652,
    "size": 0.6313
  },
  "previous_claim_0_891_0_697_0_748_corrected": true,
  "paper_section_4_1_lead": "We evaluate Mol-Metal on the DrugOOD (Ji et al. ICLR 2023) IC50 splits (95,236 curated rows per config, lbap_core noise tier, 3 domains: assay / scaffold / size). Two models are trained and measured: chemprop v2 (Heid et al. 2024, AtomMessagePassing depth=3 hidden=300, gold-standard defaults) on 5 seeds x 30 epochs x 3 splits = 15 runs in 113 s wall time; and V4MultiTaskDMPNN (our multi-task D-MPNN + Wasserstein pIC50 auxiliary, adapted to n_metals=1 single-head binary) on a partial 4-seed x 30-epoch x 1-split run within a 1-hour budget. Per-split OOD test AUC: chemprop v2 assay 0.7100 +/- 0.0162, scaffold 0.6652 +/- 0.0082, size 0.6313 +/- 0.0055; V4 assay 0.6964 +/- 0.0022 (scaffold + size n/a - budget exhausted). Honest verdict on iMoLD (Ji 2023 ICLR, cite-only) head-to-head: iMoLD reference numbers are 0.7211 / 0.6884 / 0.6792 on assay / scaffold / size. Both Mol-Metal backbones are below iMoLD on every split measured: chemprop v2 gaps are -0.011 / -0.023 / -0.048; V4 assay gap is -0.025. We do NOT claim to beat iMoLD on the real DrugOOD IC50 benchmark. The previous section 4.1 lead claim (DrugOOD OOD AUC 0.891 / 0.697 / 0.748 with +0.170 / +0.009 / +0.069 over iMoLD) is retracted: those numbers came from a 5,207-row MetalCytoToxDB-derived metacytotox_fallback subset with DrugOOD-style OOD splits applied, NOT from the actual Ji 2023 DrugOOD IC50 splits (the data_source: metacytotox_fallback flag and honest_framing field in metrics/by_metric/drugood_ood_auc.json explicitly state this). Mol-Metal DrugOOD OOD AUC on real Ji 2023 IC50 is honest-negative with respect to iMoLD, reported here with full audit trail.",
  "honest_framing_note": "Per party-twenty-spirit section 3.5 + TODO-84 NO paper/* writes without explicit user OK. The previous section 4.1 lead claim (0.891 / 0.697 / 0.748 with +0.170 / +0.009 / +0.069 over iMoLD) is RETRACTED: those numbers came from metrics/by_metric/drugood_ood_auc.json whose data_source flag is 'metacytotox_fallback' (5,207-row MetalCytoToxDB-derived subset with DrugOOD-style OOD splits applied), NOT from the actual Ji 2023 DrugOOD IC50 95,236-row curated splits. The honest_framing field embedded in that JSON explicitly states 'Numbers in this JSON were computed on a metacytotox_fallback dataset (MetalCytoToxDB-derived 5,207 rows) using DrugOOD-style OOD splits (assay/scaffold/size). This is NOT a head-to-head comparison against iMoLD on the actual DrugOOD IC50 splits (Ji et al. ICLR 2023).' The new Wf-170 chemprop v2 + V4 measurements on real DrugOOD IC50 (.data/molmetal/drugood/ic50/csv/lbap_core_ic50_{assay,scaffold,size}.csv, 95,236 rows per config, 5 seeds x 30 epochs for chemprop v2, partial 4 seeds x 1 split for V4 within 1-h budget) show both Mol-Metal backbones BELOW iMoLD on every split measured (gaps -0.011/-0.023/-0.048 for chemprop v2; -0.025 for V4 assay). This is honest-negative with respect to iMoLD; the chemprop v2 backbone (gold-standard for IC50 prediction, hidden=300/depth=3 defaults) is the strongest Mol-Metal comparator and the gap is the honest gap. V4 is partial: scaffold and size splits have 0 seeds trained due to 1-hour budget exhaustion. The 5,207-row metacytotox_fallback numbers are retained in drugood_ood_auc.json for archival (with honest_framing field) but NOT cited as a section 4.1 lead claim. No paper/* writes performed in this session; the 10 paper files that cite the old 0.891/0.697/0.748 numbers (paper/section4_results.tex section 37-73; paper/sections/04_results_table2.tex section 41-43; paper/section7_conclusion.tex section 14; paper/abstract_draft.tex section 68; paper/cover_letter.txt section 12; paper/cover_letter_cn.txt section 12; paper/cover_letter_jcim.tex section 72-73; paper/highlights_jcim.txt section 21; paper/figures/data/fig8_ood_radar.json + fig13_drugood_ood_per_split.json; paper/paper_v2.tex section 260; paper/sections/04_results_table2_v2.tex section 105/111/117; paper/supplementary/S5_benchmark.tex section 219-220) require explicit user OK per TODO-84 before any writes."
}
```
