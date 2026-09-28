# WF-wetlab-plan — Prospective wet-lab validation protocol (DRAFT)

**Status**: DRAFT (companion to paper/appendices/W_wetlab_protocol.tex)
**Date**: 2026-09-25
**TODO anchor**: §7 (future work, item 7.4) + §6 (limitation 6.4 "no wet-lab")
**Constraint**: NO actual lab work executed; this is a planned protocol only

---

## TL;DR

The prospective wet-lab validation plan for the top-$k{=}10$
candidates is fully specified in the Appendix W LaTeX file
(`paper/appendices/W_wetlab_protocol.tex`).  This report is the
companion ML-side / project-side summary covering:

- selection criteria (Vina Dock $\leq -8.5$, SA $\geq 0.65$, QED $\geq 0.50$, PB-valid $\geq 90\%$, Morgan ECFP4 Tanimoto $\leq 0.65$)
- per-candidate reaction route (18 SyntheMol rules + 84-MLE matrix), scaffold-aware via `pt_click_compat.py:detect_scaffold()`
- 3-batch procurement plan (¥18k + ¥12k + ¥8k; 5+3+2 day lead times)
- 4-cell-line PlatinAI panel (MCF-7 / A549 / A2780 / A2780cis)
- CellTiter-Glo$^{\circledR}$ 6-point dose-response, 3 replicates, 72 h
- 6-week wall-clock timeline (synthesis 2 w + assay 4 w)
- ¥70{-}130k budget envelope (median ¥100k)

All numbers are **PROJECTED**, not **MEASURED**.  The Tier-1 wet-lab
plumbing is SHIPPED (`wf_t30_wetlab_plumbing/final.md`,
2026-09-17); Tier-3 (collaborator outreach + IRB + plate-reader
integration) is the work this plan operationalises.

---

## 1. Why this plan

### 1.1 §6.4 (no wet-lab) is a hard limitation

The §6.4 limitation reads:

> This paper is in silico only; we report computational Vina /
> PoseBusters / SA / QED / anticancer-index / pIC$_{50}$ predictions
> but do not include synthesised candidates, in-vitro GI$_{50}$
> measurements, or cellular-uptake assays.

The §7.4 future-work item (§\ref{sec:future:wetlab-validation})
references Rusanov 2025 PlatinAI as the prospective-validation
anchor:

> PlatinAI synthesised 20/20 predictions with strong
> predicted-vs-observed agreement and validated a bis-carbene Pt(II)
> complex superior to cisplatin in A2780cis-resistant lines; the
> same protocol applied to our top-5 closes the
> in-silico-to-experimental loop.

This appendix + plan operationalises the §7.4 future-work item in
the form a chemistry collaborator can pick up and execute.

### 1.2 What the plan closes

| Limitation / Future-work item | What this plan does |
|---|---|
| §6.4 no wet-lab | Synthesise + assay $k{=}10$ candidates |
| Metric 2 (Pt 4-cell-line AUC) | Per-cell-line pIC$_{50}$ on MCF-7/A549/A2780/A2780cis |
| Metric 5 (SA-score) | Per-candidate yield (the SA metric's bench-side proxy) |
| Metric 11 ($n_{\text{distinct}}$) | Independent characterisation verifies the 20-chemotype basket |
| §6.4 honest caveat (Pearson $r = 0.198 \pm 0.228$ vs ridge $r = 0.572$) | Measured-vs-predicted Pearson on $n{=}40$ (10 candidates × 4 cell lines) |

### 1.3 Honest framing

None of the chemistry / biology has been performed.  Cost / time /
success criteria are **PROJECTED** from PlatinAI's published
protocol + NCI-60 standard operating procedures + internal
metallo-drug lab experience (consultations with 2 academic
chemistry groups, 2026-Q2).

---

## 2. Top-k candidate selection (§W.1)

### 2.1 Source list

Round-13 100-pocket $\times$ 3-seed sweep output (the
`wf_lambda1c/final.md` Tier-3 deliverable) produces one candidate
list per pocket.  We pool these lists and apply a 4-stage filter:

| Filter | Threshold | Anchor |
|---|---|---|
| Vina Dock score | $\leq -8.5$ kcal/mol | DecompDiff 2024 Tier-2 SOTA |
| SA-unit score | $\geq 0.65$ | Metric 5 Tier-2 gate |
| QED | $\geq 0.50$ | Metric 6 Tier-2 gate |
| PoseBusters validity | $\geq 90\%$ | Metric 8 Tier-2 gate |

After the 4-stage filter the pool contains 15{-}30 candidates (10
pockets $\times$ 3 seeds produces $\sim$30 rows, of which 15{-}30
survive the filters).

### 2.2 De-duplication

Morgan ECFP4 pairwise Tanimoto $\leq 0.65$ (Metric 7 threshold)
selects $k{=}10$ structurally-distinct candidates.  Ties broken
by Vina Dock score.

### 2.3 Metal-seed provenance

Each candidate is labelled with its scaffold category via
`pt_click_compat.detect_scaffold()`:

- `strict_Pt_II` (cisplatin-like)
- `Pt_II_chelating` (carboplatin / oxaliplatin-like)
- `Pt_IV` (satraplatin-like)
- `labile_metal` (Cu/Zn/Fe/Mn coordination)

The label selects the per-candidate reaction protocol (§W.2).

---

## 3. Per-candidate synthesis protocol (§W.2)

### 3.1 18-rule SyntheMol pathway

The default synthetic route for a typical candidate is the
3-step sequence **CuAAC (WM-1) → sulfonamide (WM-13) → amide
coupling (WM-5)**:

- **WM-1 CuAAC** --- 2 working days (alkyne + azide + CuSO$_4$ +
  sodium ascorbate, H$_{2}$O/$^{t}$BuOH, RT → 60 °C, 12 h;
  Himo 2005 regiochemistry canon).
- **WM-13 Sulfonamide** --- 1 working day (sulfonyl chloride +
  amine, Et$_3$N, CH$_2$Cl$_2$, 0 °C → RT, 4 h).
- **WM-5 Amide coupling** --- 1 working day (PyBOP + R-COOH +
  R-NH$_2$, DIPEA, DMF, RT, 8 h; Pt_II_chelating requires
  the SPAAC variant to avoid Cu contamination).

Pt_II-chelating candidates substitute **SPAAC (WM-2) → amide
coupling (WM-5)** to avoid Cu(I) contamination of the
platinum-chloride scaffold.

The 13 SyntheMol heterocyclic / cross-coupling / condensation
rules (WM-6 … WM-18) are exposed in
`molmetal/molmetal_lam/reactions/syntemol_reactions.py`.  The
selection per candidate is driven by the §W.1 scaffold label.

### 3.2 84-MLE metal-ligand exchange pathway

For candidates whose final coordination sphere requires a
ligand-exchange step (the dominant pathway for cisplatin-class
analogues), the reaction is drawn from the 12-metals × 7-ligands
= 84 matrix in `pt_metal_ligand_exchange.py`:

| Index | Reaction | Scaffold | Bench-time | Lit anchor |
|---|---|---|---|---|
| MLE[Pt_II, Cl → NH3] | cisplatin-Cl → cisplatin-NH3 | strict_Pt_II | 1 d | Lippard 1995 §5.3 |
| MLE[Pt_II, Cl → H2O] | cisplatin aquation | strict_Pt_II | 0.5 d | Reedijk 1987 |
| MLE[Pt_II, Cl → CN⁻] | cyanide bridge | strict_Pt_II | 1 d | standard |
| MLE[Pt_IV, axial-OH → carboxylate] | Pt(IV) prodrug activation | Pt_IV | 1.5 d | standard |

### 3.3 Building-block procurement (3 batches)

| Batch | Week | Contents | SKUs | Cost | Lead time |
|---|---|---|---|---|---|
| Batch 1 | week 1 | azides / alkynes / aryl-borons / amines | 28 | ¥18k | 5 d |
| Batch 2 | week 2 | Pt precursors (K$_2$PtCl$_4$, cisplatin, carboplatin, Pt(IV) SM) | 4 | ¥12k | 3 d |
| Batch 3 | week 2 | auxiliary reagents (PyBOP, EDC, CuSO$_4$, sodium ascorbate, TBTA) | 6 | ¥8k | 2 d |

Per-candidate bulk cost is ¥1.5{-}3.0k (10-candidate bulk discount
≈ ¥1.5k / candidate).

### 3.4 Per-candidate estimated cost / time

| Component | Cost / candidate (¥) |
|---|---|
| Building blocks | 1.5k |
| Reagents / solvents | 0.8{-}1.2k |
| Pt precursor | 1.0{-}3.0k |
| NMR / MS / HPLC characterisation | 0.5{-}1.0k |
| Single-crystal XRD (1 representative) | 5.0k |
| **Total / candidate** | **4{-}8k** (median 5k) |
| **Total $k{=}10$** | **¥50{-}80k** |

Per-candidate bench time: $2$ w synthesis + $1$ w characterisation
= $3$ w serial; $1$ w / candidate parallel across 3 fume hoods.
Full $k{=}10$ programme in **6 weeks wall-clock**.

---

## 4. In-vitro assay protocol (§W.3)

### 4.1 4-cell-line PlatinAI panel

The PlatinAI 4-cell-line panel is the reference; we replicate it
for direct head-to-head comparison with their published per-line
AUC numbers (Metric 2 gate).  The panel:

| Cell line | Origin | ATCC / ECACC SKU | Role |
|---|---|---|---|
| MCF-7 | breast adenocarcinoma, ER+ | ATCC HTB-22 | canonical breast line |
| A549 | lung adeno, KRAS-mut | ATCC CCL-185 | weak Pt-sensitivity baseline |
| A2780 | ovarian, cisplatin-sens | ECACC 93112519 | gold-standard sens. control |
| A2780cis | ovarian, cisplatin-res | ECACC 95028834 | **the key readout** |

The §\ref{sec:evaluation:metallodrug-vertical} 9-line cohort is a
Tier-2 follow-up (§\ref{sec:future:nci60}); the Tier-1 wet-lab
programme stays on the PlatinAI 4-line panel.

### 4.2 Assay format

96-well plate, 6-point dose-response (1:3 dilution, top 100 μM),
3 biological replicates, 72 h endpoint, CellTiter-Glo$^{\circledR}$
luminescent ATP assay.  Per-plate calibrator: cisplatin (0.1{-}100
μM dose range).

### 4.3 Quantitative readouts

For each candidate × cell-line:

- **IC$_{50}$ (μM)** --- 4-parameter logistic fit, Prism 10.
- **pIC$_{50}$** --- $-\log_{10}$(IC$_{50}$ / 1M), the canonical
  unit for in-silico oracle comparison.
- **Efficacy ratio** --- IC$_{50}$(candidate) / IC$_{50}$(cisplatin)
  on each line; "smaller is superior to cisplatin" gate.
- **Resistance-reversal ratio** --- IC$_{50}$(A2780cis) /
  IC$_{50}$(A2780); the smaller-the-better metric PlatinAI used
  for bis-carbene superiority ($\leq 2.0$ target; cisplatin
  baseline 5{-}10).

### 4.4 Quality control

Each plate batch includes vehicle (DMSO ≤ 0.5%), positive controls
(cisplatin + oxaliplatin + carboplatin), negative (medium-only),
Z'-factor ≥ 0.5 (plates below threshold re-run).

---

## 5. Timeline (§W.4)

6-week best-case; 8{-}10 w realistic:

| Week | Activity |
|---|---|
| 1 | Procurement Batch 1 (azides / alkynes / amines); NMR / MS instrument booking |
| 2 | Procurement Batches 2 (Pt) + 3 (auxiliary); candidates 1{-}2 start synthesis |
| 3 | Synthesis of candidates 3{-}7 (5-candidate bulk); on-track for 3-week window |
| 4 | Synthesis of candidates 8{-}10; candidates 1{-}4 move to assay |
| 5 | Cell-viability assay on full 4-cell-line panel (~50 plates × 6 doses; 1-week bench) |
| 6 | Final IC$_{50}$ fits, qa, write-up of measured-vs-predicted agreement |

2-week buffer for synthesis retries (typical Pt-IV prodrug yield
drop, pH-sensitive CuAAC catalysts) + mycoplasma contamination
screening.

---

## 6. Cost estimate (§W.5)

### 6.1 Itemised budget (¥ CNY)

| Category | Amount | Sub-breakdown |
|---|---|---|
| Synthesis | ¥50{-}100k | BB ¥18k + Pt ¥12k + aux ¥8k + solvents ¥5{-}10k + NMR/MS ¥5{-}15k + bench ¥10{-}20k |
| Assay | ¥20{-}30k | cell-line licences ¥8{-}10k + culture ¥4{-}6k + CTG kit ¥4{-}6k + controls ¥2{-}3k + plate-reader ¥2{-}5k |
| **Total** | **¥70{-}130k (median ¥100k)** |

### 6.2 Industry benchmarks

- Single CRO quote for 4-cell-line × 10-candidate IC$_{50}$ panel:
  ¥120{-}150k (WuXi AppTec 2026 / ChemPartner 2026 quote rates).
  In-house is ~30% cheaper in exchange for bench-time commitment.
- PhD-year of consumables at a typical Chinese R1 university:
  ¥80{-}120k.  Wet-lab programme is equivalent of one academic year
  compressed into 6{-}8 weeks.
- PlatinAI's published budget is not in the public ChemRxiv
  supplementary; we estimate their full 20-candidate × 4-cell-line
  programme at ¥300{-}500k based on synthesis count and Pt
  consumption.

### 6.3 Funding sources

- Single NSFC 青年基金 (¥300k) covers the programme with ample room
  for the §\ref{sec:future:nci60} Tier-2 follow-up.
- Provincial science foundation grant (¥50{-}100k) covers the
  bare programme.
- Collaboration with one of the existing metallo-drug labs
  (e.g.\ the Sun / Wang / Chen groups in Pt-cytoxicity synthesis)
  shares bench-time overhead; cash outlay drops to ~¥40{-}60k.

---

## 7. Comparison to PlatinAI 2025 (§W.6)

### 7.1 PlatinAI's protocol (the reference)

PlatinAI synthesised 20/20 predictions with a 4-cell-line panel
and obtained strong predicted-vs-observed agreement (their
reported bis-carbene Pt(II) was superior to cisplatin in A2780cis).
Three of their methodological choices inform ours:

1. **Reference compound on every plate** --- cisplatin per-plate
   calibrator (we replicate).
2. **Resistance-reversal readout** ---
   IC$_{50}$(A2780cis) / IC$_{50}$(A2780) (target $\leq 2.0$).
3. **Per-candidate $^{1}$H-NMR / ESI-MS / HPLC purity $\geq 95$\%**
   (we replicate).

### 7.2 Adaptations for Mol-Metal

| | PlatinAI | Mol-Metal |
|---|---|---|
| Number of candidates | 20 | 10 |
| Cell-line panel | MCF-7, A549, A2780, A2780cis | same |
| Reaction routes | single (bis-carbene Pt(II)) | 18 SyntheMol + 84-MLE matrix, scaffold-aware |
| Building-block procurement | sequential, ~30 SKUs | 3-batch parallel (28 + 4 + 6 SKUs) |
| Tier-2 cohort | none | HeLa / HCT-116 / OVCAR-3 follow-on |
| Oracle validation | external (Rusanov's lab) | in-house + wetlab_protocol.py additive reward |
| Plumbed into ML? | no | yes (recalibrate_from_assay.py) |

### 7.3 Success criteria for the plan

| Criterion | Target | Source |
|---|---|---|
| Measured pIC$_{50}$ on $k{=}10$ candidates | $n{=}40$ rows | §W.3 |
| Measured-vs-predicted Pearson $r$ | $\geq 0.50$ | §6.4 baseline $0.198 \pm 0.228$; PlatinAI analogue ~0.6{-}0.7 |
| Resistance reversal on at least 1 candidate | $\leq 2.0$ on A2780cis/A2780 | PlatinAI-style "superior to cisplatin" |
| Per-candidate yield | $\geq 40\%$ average | SA $\geq 0.65$ gate wet-side proxy |
| SA-Vina-PB consistency | every §W.1-filter candidate synthesised $\leq 6$ steps + measured Vina-Dock re-docking RMSD $\leq 2.0$ Å | §W.6 sanity check |

### 7.4 Out-of-scope

- In-vivo mouse xenograft (4{-}8 weeks + animal-ethics approval;
  Tier-2 §\ref{sec:future:nci60} follow-up)
- Pharmacokinetics (plasma protein binding, CYP, hERG; standard
  CRO panel; Tier-3 §\ref{sec:future:pk-panel} follow-up)
- DNA-binding constant Kb by electrophoretic mobility shift assay

### 7.5 Reference citations

- `paper/refs.bib @rusanov2025platinai` (PlatinAI 4-cell-line protocol)
- `paper/refs.bib @krasnov2026metalcytotoxdb` (5-metal cytotoxicity database)
- NCI-60 screening protocol (Tier-2 anchor)
- Lippard 1995 + Reedijk 1987 (Pt-Cl aquation kinetics)
- Himo 2005 (CuAAC regiochemistry canon)
- Ertl-Scharan-Filter 2009 (SA-score)
- Bickerton 2012 (QED)

---

## 8. Data plumbing integration

### 8.1 Tier-1 wet-lab API (SHIPPED)

The Tier-1 wet-lab plumbing is already SHIPPED at
`wf_t30_wetlab_plumbing/final.md` (2026-09-17):

- `molmetal/molmetal_lam/lam_chem/wetlab_protocol.py` —
  `Assay` dataclass + `load_assays(path)` + `append_assay()` +
  `assays_to_dict()` (TSV → `{smiles: pIC50}` mapping).
- `molmetal/molmetal_lam/lam_chem/wetlab_reward_channel.py` —
  signed-error channel (returns $-$|predicted $-$ measured|, ∈
  [-3, 0], additive on existing `RewardAggregator`, default
  `w_wetlab=0.0` opt-in).
- `molmetal/scripts/recalibrate_from_assay.py` — CLI for the
  bias-diagnostic report (Pearson r / Spearman rho / RMSE /
  within-margin count).
- `molmetal/molmetal_lam/search_alg/proof_search.py` —
  `register_wetlab_channel(measurement_dict, *, weight=0.0, predicted_lookup=None)`
  method on the search-aggregator dataclass.
- `molmetal/tests/test_wetlab_channel.py` — 29 tests; 21 pass
  on CPU (8 deselected on the host-level libtorch issue, NOT a
  code issue).

### 8.2 Drop-in usage after Tier-3 wet-lab data lands

After Tier-3 produces measured pIC$_{50}$ rows, the integration
is a one-line TSV drop:

```python
# 1. Write assays as TSV:
#   smiles\tpIC50\tcell_line\tdate\tcollaborator
#   [Pt](N)(N)(N)(N)Cl...   6.5    A2780cis   2026-12-15  ChenLab
#   ... 39 more rows

# 2. Drop into the CLI:
$ python molmetal/scripts/recalibrate_from_assay.py \
    --wetlab-input data/assays_2026_Q4.tsv \
    --drylab-input data/cytotox_predicted.tsv \
    --output reports/wf_wetlab_q4_calibration.json
#   -> Pearson r, Spearman rho, RMSE, bias diagnostics
#      + recommendation (OK / RE-TRAIN with margin / etc.).

# 3. Re-register on the search aggregator:
from molmetal.molmetal_lam.lam_chem.wetlab_reward_channel import (
    register_wetlab_channel, )
from molmetal.molmetal_lam.lam_chem.wetlab_protocol import (
    load_assays, assays_to_dict, )
meas = assays_to_dict(load_assays("data/assays_2026_Q4.tsv"))
register_wetlab_channel(agg, meas, weight=0.1)
#   -> subsequent Lambda+CFM rounds see the wet-lab signal
#      in their MCTS leaf reward; no code change required.
```

### 8.3 What this means

The plan in this report + the §W appendix is **operational**
rather than aspirational: the wet-lab run, when it happens,
immediately re-calibrates the in-silico oracle for the next
Lambda+CFM round.  This is the structural fix for the §6.4
no-wet-lab limitation.

---

## 9. Files shipped

### NEW

- `paper/appendices/W_wetlab_protocol.tex` (the prospective
  protocol appendix; 6 sub-sections §W.1-§W.6, ~300 lines,
  ready for inclusion in paper supplementary or §7 expansion).

### MODIFIED

- None (this workflow is a planning / reporting deliverable).

---

## 10. Cross-references

- §7.4 future-work item
- §6.4 no-wet-lab limitation
- Metric 2 (Pt 4-cell-line AUC)
- Metric 5 (SA score)
- Metric 11 ($n_{\text{distinct}}$)
- `molmetal/reports/wf_t30_wetlab_plumbing/final.md` (Tier-1
  plumbing ship)
- `paper/refs.bib @rusanov2025platinai` (PlatinAI reference)

---

## 11. Next-step actions (for the wet-lab programme owner)

1. **Identify chemistry collaborator** --- one PI + 2 postdocs
   in metallo-drug synthesis; contact options listed in
   TODO/pending/ (Sun / Wang / Chen groups, CN-domestic
   platinum-cytoxicity labs).

2. **Identify biology collaborator / CRO** --- in-house 4-cell-line
   capability, or CRO quote (WuXi AppTec, ChemPartner); target
   quote ≤ ¥120k.

3. **IRB + biosafety paperwork** --- typically 4{-}6 weeks at a
   Chinese R1 university; budget for parallel filing.

4. **Wet-lab run start** --- after IRB + procurement settled;
   target Q1-2027 start.

5. **Data integration** --- drop assays as TSV into the
   `wetlab_protocol.load_assays` API; run
   `recalibrate_from_assay.py`; commit results to
   `wf_wetlab_q4_calibration.json`; re-register the channel
   on the next Lambda+CFM round.

---

**Status**: DRAFT --- pending PlatinAI 2025 full-data download
(Metric 2 prerequisite; cf. §\ref{sec:future:platinai-download})
+ chemistry / biology collaborator confirmation.
