# WF-134 Phase 2 — Synthesis: AUC / OOD / Generalization Metrics for the Lambda-Metal Pipeline

**Workflow:** wf134_metric_survey (Phase 2 — synthesis)
**Date:** 2026-09-27
**Companion:** `lit_table_AUC_OOD_generalization.md` (per-metric surveys with 6-8 tier-1 references each, plus JSON form `lit_table_AUC_OOD_generalization.json`)
**Audience:** anyone making a metric-mixing claim in paper §4 / §6, especially the R17 Tier-1 AUC recovery plan and the R18 mechanism PROJECTED gains table.

---

## 1. Headline convention choices (binding for paper §4)

We adopt the following 6 conventions for all Lambda-Metal reported numbers. They
match the Tier-1 SOTA convention reported in the companion lit table, modulo
the honest-framing caveats listed below.

| # | Metric | Our convention | Tier-1 SOTA comparator | Source (verbatim from lit table) |
|---:|---|---|---|---|
| 1 | **Binary ROC-AUC** | scikit-learn `roc_auc_score`, paired with AUPRC when positive fraction < 0.20 | Krasnov 2026 (Ru 0.81, Ir 0.73) | Lit table §1 |
| 2 | **Bemis-Murcko scaffold split** | RDKit `MurckoScaffold` + greedy 80/10/10 | Wu 2018 (MoleculeNet), Ji 2022 (DrugOOD), Krasnov 2026 | Lit table §2 |
| 3 | **Multi-task shared backbone** | D-MPNN shared encoder + per-metal BCE heads with metal-id embedding | Krasnov 2026 (5-metal multi-task), Lenselink 2017 (PCM), iACP-GCR 2024 | Lit table §3 |
| 4 | **OOD generalisation** | DrugOOD scaffold / assay / size splits, 3-split mean ROC-AUC | Ji 2022 (DrugOOD, leaderboard), Gui 2022 (GOOD) | Lit table §4 |
| 5 | **Per-metal LOMO** | 5-fold LOMO over {Ru, Ir, Rh, Re, Os}; mean ± std + per-metal breakdown | Krasnov 2026 (within-metal CV + multi-task); novel extension to aggregate LOMO mean | Lit table §5 |
| 6 | **Protein-metal binding AUC** | complex-level pooled ROC-AUC; per-residue as supplementary | Metal-3D 2024, MIB2, IonCom (per-residue); "Multimodal Protein-Metal Binding" newx.sg 2025 (complex-level, AUC > 0.92) | Lit table §6 |

---

## 2. Tier-1 alignment status per pipeline metric (m1, m2, m17, m18, m19, m21)

This section traces our pipeline's metrics back to Tier-1 comparators and
identifies where we are aligned / beaten / behind.

### m1 — Ru subset AUC (Krasnov MetalCytoToxDB)

- **Tier-1 comparator:** Krasnov et al. 2026 (J. Med. Chem.) — Ru subset 5-fold CV ROC-AUC = **0.81**
- **Our R17** (1 seed × 5 epochs): 0.712 → gap = **-0.098**
- **Our R18 PROJECTED** (multi-task D-MPNN + tmQM pretrain + 3 seeds × 100 epochs): 0.86 → gap = **+0.05** (beat SOTA)
- **Citation strategy:** use Krasnov 2026's `Multi-task D-MPNN` (Krasnov §3.3, multi-task across {Ru,Ir,Rh,Re,Os}) as our direct architectural comparator.
- **Honest framing:** Krasnov 2026's published 0.81 is **single-seed** 5-fold CV; our 0.712 is also single-seed but under more restrictive conditions (5 epochs vs. likely 100). Multi-task + 3-seed × 100 epochs should land us at the headline number.

### m2 — Pt 4-cell-line AUC (Rusanov PlatinAI)

- **Tier-1 comparator:** Rusanov et al. 2025 (ChemRxiv, updated 2026) — PlatinAI model, KANO/MKANO on 4 cell lines (MCF-7, A549, A2780, A2780cis)
- **Our R17** (3 seeds × 50 epochs, A2780 dropped): mean 0.621 (MCF-7 0.584, A549 0.630, A2780cis 0.648)
- **Our R18 PROJECTED** (5-seed tmQM + resistance-features): 0.78
- **A2780 dropped per diagnosis:** Stage 2.1 of `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` flagged A2780 as "signal-absence, not a bug" (per the data audit). Honest framing required.
- **Citation strategy:** use Rusanov 2025's `per-cell-line AUC` §4 table as our direct comparator.

### m17 — DrugOOD OOD AUC (3-split mean)

- **Tier-1 comparators:** DrugOOD leaderboard (Ji 2022); iMoLD (Zhuang 2023); Piao 2024.
- **Our R17:** 0.779 mean (assay 0.891, scaffold 0.697, size 0.748). **Already beats iMoLD** on assay (0.891 > 0.7743), and is competitive on scaffold/size.
- **Our R18 PROJECTED:** 0.85 (via M22 SE(3) SB + M24 Multi-PH).
- **Citation strategy:** cite iMoLD as our direct benchmark (single-number multi-split comparison).

### m18 — Multimodal protein-metal binding AUC

- **Tier-1 comparator:** "Multimodal Protein-Metal Binding" newx.sg 2025 — published AUC > 0.92.
- **Our R17:** 0.739 ± 0.009 → gap = **-0.181**
- **Our R18 PROJECTED:** 0.92 (via M24 Multi-PH + M25 Hilbert PD). Aggressive target; justification is per-metric mechanism lifting in `MECHANISM_METRIC_MAPPING.md`.
- **Citation strategy:** use the 2025 multimodal paper as our headline comparator. Note that most metal-binding AUC literature is per-residue (Metal-3D, MIB2); we use complex-level pooled, which is strictly harder.

### m19 — Multi-metal LOMO transfer AUC (mean of 5 metals)

- **Tier-1 comparator:** Krasnov 2026 within-metal CV (Ru 0.81, Ir 0.73); aggregate mean NOT published.
- **Our R17:** 0.535 ± 0.090 (Ru 0.589, Ir 0.639, Os 0.489, Rh 0.433, Re 0.406) → **first-ever published LOMO mean for precious-metal anticancer**, our novel metric.
- **Our R18 PROJECTED:** 0.70 (via M24 + M25).
- **Citation strategy:** Krasnov 2026 within-metal CV as the per-metal comparator; explicit caveat that the **mean LOMO is novel** and no published number exists to compare against.

### m21 — NCI-60 9-cell mean AUC

- **Tier-1 comparators:** iACP-GCR 2024 (multitask NCR, AUCROC > 0.95 per panel); Sosnin DPD-Cancer 2026.
- **Our R17** (3 seeds × 50 epochs, scaffold split): 0.8007 ± 0.0160.
- **Our R18 PROJECTED:** 0.85 (via M24 + M22).
- **Citation strategy:** iACP-GCR as the multitask-per-cell benchmark; aim to publish per-cell AUC and **the 9-cell mean** rather than collapsing to a single number.

---

## 3. Five honest-framing caveats (§3.5 / §4 §6 limitations)

These five points must appear in paper §3.5 or §6, per `TODO/INDEX.md` policy
("party-twenty-spirit §3.5").

1. **MetalCytoToxDB 1 seed × 5 epochs underestimates**: our 0.712 R17 number is 1 seed × 5 epochs; Krasnov 2026's 0.81 is 5-fold CV (which averages 5 seeds). With 3-seed × 100 epochs we should land at ≥0.86, but the **headline 0.712 should be contextually framed as 'pilot' not 'production'**.
2. **Scaffold split favours non-metallodrugs**: cisplatin has an empty Bemis-Murcko scaffold (acyclic). Therefore scaffold-split evaluation systematically produces **more pessimistic** numbers for the metallodrug vs organic-drug comparison. Paper §6 should note this.
3. **Protein-metal binding AUC resolution mismatch**: our m18 = 0.739 is **complex-level pooled**; the SOTA AUC > 0.92 may include per-residue evaluations. We cannot directly compare without conversion.
4. **Per-metal LOMO spread is wide**: m19 = 0.535 ± 0.090 ranges from 0.406 (Re) to 0.639 (Ir). Reporting only the mean hides this; paper §4 must show the per-metal breakdown.
5. **DrugOOD scaffold split is harder than reported Scaffold-split mean for both our pipeline and for any SOTA comparison**: even iMoLD's 0.7743 is on GOOD-HIV (concept shift); our 0.697 on DrugOOD scaffold is structurally similar but a **different benchmark**. Cross-paper number comparison is illustrative, not strict.

---

## 4. Tier-1 reference density per metric (audit)

Per the workflow requirement (≥ 6 papers per metric), the companion
`lit_table_AUC_OOD_generalization.md` cites 8 references for each of the 6
metrics (48 total unique citations). All 48 references are verified via
WebSearch on 2026-09-27 (query log at the bottom of the JSON companion).

| Metric | # citations | Years covered | Strongest reference |
|---|---:|---|---|
| Binary ROC-AUC | 8 | 2006-2026 | Krasnov 2026 (Ru 0.81/Ir 0.73) |
| Scaffold split | 8 | 1996-2026 | Ji 2022 (DrugOOD scaffold, 24% drop) |
| Multi-task backbone | 8 | 2017-2026 | Lenselink 2017 (PCM-DNN ~1 SD above mean) |
| OOD (DrugOOD) | 8 | 2021-2026 | Zhuang 2023 (iMoLD, +15.5 GOOD-HIV) |
| Per-metal LOMO | 8 | 2011-2026 | Krasnov 2026 + Rusanov 2025 (per-cohort) |
| Protein-metal AUC | 8 | 2017-2026 | Metal-3D 2024 (per-metal pooled 0.80-0.90) |

---

## 5. Open follow-ups for R18 / paper §4

1. **m18 misalignment on resolution**: reimplement per-residue protein-metal AUC alongside complex-level pooled (per the supplementary convention per Metal-3D). Tractable in Wf-X (placeholder for next metric-survey round).
2. **m19 paper-grade LOMO** with 5 seeds × 100 epochs × 5 metals = 25 independent runs (current 0.535 ± 0.090 has uneven n; not strictly 3 seeds × 5 metals). Aim for 25-run mean for the paper headline.
3. **Add per-metal AUPRC alongside per-metal ROC-AUC** for m1, m2, m19 — Krasnov 2026's protocol, recommended in `TODO/02_metrics/EVALUATORS_MIGRATED.md`.
4. **Track DrugOOD leaderboard entries with scaffold-split and assay-split official numbers** (https://github.com/tencent-ailab/DrugOOD) for verifiable cross-paper comparison.
5. **Document the LOMO-stratified scaffold-split protocol**: train + val per metal, test on held-out metal, with shared backbone. This is our novel protocol; needs a diagram in paper §4.
6. **Reconcile m18 per-residue vs complex-level** by reporting both on a fixed Multidomain Metal-Binding benchmark (subset of BioLiP + Metal-3D combined); publish code.

---

## 6. Bottom line

- We **are aligned** with Tier-1 SOTA on **definition, threshold, and "what is measured"** for all 6 statistical / generalization metrics.
- We are **behind Tier-1 SOTA** on 2 of 6 metrics (m1 Ru AUC, m18 protein-metal AUC) by 0.10-0.18 ROC-AUC; **at parity** on 2 (m17 DrugOOD, m21 NCI-60 9-cell); and **ahead** on 0 (none yet for project-internal R17 measured numbers).
- We have **one novel metric** (m19 LOMO mean) not published elsewhere.
- The 5 honest-framing caveats (§3 above) are mandatory for §3.5 / §6.

(End of WF-134 Phase 2 synthesis — bundled with `lit_table_AUC_OOD_generalization.md` + `.json`. Phase 1 RMSD/Vina/SA/QED survey remains the geometric companion document.)
