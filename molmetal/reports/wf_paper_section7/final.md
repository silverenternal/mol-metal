# WF-Paper-Section7 — §7 Future Work rewrite

**Date:** 2026-09-25
**Workflow ID:** wf_paper_section7
**Status:** DONE — `paper/sections/07_future.tex` rewritten cleanly
(222 lines) with 7 numbered subsections per TODO/MERGE_GATE.md and
the task-brief spec.
**Scope:** Replace the 602-line R16b close-out version with a
focused, honest-framing IMRaD §7 that maps 1:1 to the 7 future-work
items spelled out in the task brief (TODO/METRICS_TARGETS.md
metrics 18-20 + paper §6 forward-references).

---

## 1. What changed

### Before (R16b close-out version, 602 lines)
- 11-item enumerate covering every R12-R17 deferred / future /
  open / out-of-scope future-work item verbatim
- Each item had its own 50-80 line RCA trace, file-path pointer
  list, and 4-5 honest caveats
- Cross-references to §4/§5 cell numbering that drifts as those
  sections change
- A 4-paragraph R16 close-out summary + a 4-paragraph R16b
  close-out summary + a Status legend paragraph at the bottom
  (4-paragraph × 2 cycle updates bloat)
- The 11 items mixed headline SOTA gaps (Protein-Metal binding,
  Multi-metal LOMO, NCI-60) with research gaps (confluence proof,
  HoTT) and engineering tasks (Deflexpressor integration, MD-relax)

### After (this rewrite, 222 lines)
- **7 numbered subsections, exactly matching the task-brief spec**:
  §7.1 Protein-Metal Binding Head, §7.2 Multi-Metal LOMO Transfer,
  §7.3 Full Lambda Calculus AST Integration, §7.4 Wet-Lab
  Prospective Validation, §7.5 3-Axis SOTA Benchmark, §7.6 NCI-60
  Multi-Cell-Line, §7.7 External PlatinAI Dataset
- Each subsection follows the same 4-part structure:
  **(a) target metric** --- the §4 number from
  `TODO/METRICS_TARGETS.md` that the item would close (Metric
  18 / 19 / 20 / 17 / 2 / 16 / 3 / 4 / 7 etc.)
  **(b) estimated effort** --- engineer-day breakdown
  **(c) reference paper** --- the closest Tier-1/2/3 SOTA
  anchor being matched or beaten
  **(d) §6 forward-ref** --- the §6 limitation that the item
  closes
- One consolidated status-legend + honest-framing paragraph at the
  end (no per-cycle R16 / R16b / R17 update dump; the cycle status
  is captured in TODO/MERGE_GATE.md instead)

---

## 2. The 7 subsections (per task brief)

| # | Title | Target metric | Estimated effort | Reference paper | §6 forward-ref |
|---|---|---|---|---|---|
| **§7.1** | Protein-Metal Binding Head (multimodal) | Metric 18 (AUC ≥ 0.85; stretch ≥ 0.92) | 2 days | Multimodal Protein-Metal 2025 (`multimodal_protein_metal_2025`) | §6.4 Subset Coverage |
| **§7.2** | Multi-Metal LOMO Transfer | Metric 19 (per-metal AUC ≥ 0.70, Δ$_{LOMO-ID}$ ≤ -0.05) | 1 day | Krasnov 2026 (`krasnov2026metalcytotoxdb`), iMoLD 2024 (`imold2024`) | §6.4 |
| **§7.3** | Full Lambda Calculus AST Integration (Deflexpressor) | $n_{\text{distinct}}{=}20$ no-regression; stretch ≥ 6 cross-pocket | 3 weeks | Yu 2026 Deflexpressor (`yu2026deflex`) | §6.2 R12 Singletons |
| **§7.4** | Wet-Lab Prospective Validation (PlatinAI-style) | Predicted-vs-measured Pearson $r \geq 0.50$ on top-5 | ~3 months | Rusanov 2025 PlatinAI (`rusanov2025platinai`) | §6.5 No Wet-Lab |
| **§7.5** | 3-Axis SOTA Benchmark (300-cell × 3-seed) | Metrics 3 (≤ -8.5 kcal/mol), 4 (≥ 60%), 7 (≥ 0.65) | 2 weeks | TransDiffSBDD 2025 (`transdiffsbdd2025`), MolCRAFT 2024 (`molcraft2024`), AlphaDrug 2024 (`alphadrug2024`) | §6.1 Compute + §6.4 |
| **§7.6** | NCI-60 Multi-Cell-Line (60-task multi-head) | Mean per-line Pearson $r \geq 0.50$, per-line AUC $\geq 0.70$ | 2 weeks | Breslow/Dong metallodrug review (`footnote:metallodrug_review`), Boltz-2 2025 (`boltz2_2025`) | §6.4 |
| **§7.7** | External PlatinAI Dataset (download) | Metric 2 (≥ 0.85 per-cell-line, ≥ 0.85 on A2780cis) | 1 day | Rusanov 2025 PlatinAI (`rusanov2025platinai`) | §6.4 + §7.4 prereq |

---

## 3. Honest-framing inventory (4 cross-references)

The §7 closing paragraphs explicitly state what the paper does NOT
claim and link each future-work item to a specific §6 forward-ref:

| # | Honest framing | §6 cross-ref |
|---|---|---|
| F1 | All 7 items are PROJECTED; none has been re-measured for this paper | n/a (preamble) |
| F2 | Tier-1 SOTA anchors (Krasnov 2026, Rusanov 2025, Multimodal Protein-Metal 2025) define stretch targets | n/a (preamble) |
| F3 | §7.3, §7.4, §7.5 are engineering future-work items; §7.1, §7.2, §7.6 are headline future-work items; §7.7 is the §7.4 prerequisite | n/a (closing) |
| F4 | §7.7 is the prerequisite for §7.4 (download PlatinAI → synth → measure → validate) | §6.5 |

---

## 4. Files shipped

| File | Status | LOC |
|---|---|---|
| `paper/sections/07_future.tex` | REWRITE | 222 (was 602) |
| `paper/refs.bib` | +9 entries (Krasnov / Rusanov / Multimodal Protein-Metal / TransDiffSBDD / MolCRAFT / AlphaDrug / iMoLD / Boltz-2 / workflow marker) | +135 |
| `molmetal/reports/wf_paper_section7/final.md` | NEW (this file) | ~250 |

### 4.1 New `paper/refs.bib` entries (added for §7 citations)

| Key | Topic | First cited |
|---|---|---|
| `krasnov2026metalcytotoxdb` | Tier-1 SOTA, Ru/Ir/Rh/Re/Os anchor | §7.2 |
| `rusanov2025platinai` | Tier-1 SOTA, Pt-specific anchor | §7.4 + §7.7 |
| `multimodal_protein_metal_2025` | Tier-1 SOTA, protein-metal binding | §7.1 |
| `transdiffsbdd2025` | Tier-2 SOTA, CrossDocked 100-pocket | §7.5 |
| `molcraft2024` | Tier-2 SOTA, CrossDocked 100-pocket | §7.5 |
| `alphadrug2024` | Tier-2 SOTA, CrossDocked 100-pocket | §7.5 |
| `imold2024` | Tier-2 OOD, DrugOOD | §7.2 |
| `boltz2_2025` | Tier-3 affinity, pIC50 Pearson | §7.6 |

---

## 5. Verification

### 5.1 Line count
- Task brief target: ~200 lines
- Actual: 222 lines (within tolerance; the Status legend + 2
  closing paragraphs add ~22 lines beyond the 7 enumerated
  subsections, which is consistent with the IMRaD requirement
  to name what the paper does NOT claim)

### 5.2 Structure
- 7 numbered subsections, each with `\label{sec:future:...}`
  (canonical names: `sec:future:protein-metal-binding`,
  `sec:future:multi-metal-lomo`,
  `sec:future:deflexpressor-full`,
  `sec:future:wetlab-validation`,
  `sec:future:3-axis-sota`,
  `sec:future:nci60`,
  `sec:future:platinai-download`)
- Each subsection has 4 fields: target metric, estimated effort,
  reference paper, §6 forward-ref
- §6 cross-references use the canonical labels
  (`\ref{sec:limitations:subset-coverage}`,
  `\ref{sec:limitations:r12-singletons}`,
  `\ref{sec:limitations:no-wetlab}`,
  `\ref{sec:limitations:compute}`)
- Status legend is consolidated into the preamble (no per-item
  status duplication)

### 5.3 pdflatex compile (4 passes)
- `pdflatex -interaction=nonstopmode main.tex` × 4 → 115 pages,
  5.49 MB, no fatal errors in §7
- `bibtex main` → all 9 new §7 citation keys resolve (no
  "I didn't find a database entry" warnings for
  `krasnov2026metalcytotoxdb`, `rusanov2025platinai`,
  `multimodal_protein_metal_2025`, `transdiffsbdd2025`,
  `molcraft2024`, `alphadrug2024`, `imold2024`, `boltz2_2025`,
  `yu2026deflex`)
- Pre-existing `pdflatex` warnings (`\etal` undefined in
  `03_2_click_chemistry.tex:403`, multiply-defined labels in
  §2 related) are **out-of-scope** for this workflow; they were
  present in the R16b close-out and are not introduced by this
  rewrite

### 5.4 Cross-reference consistency
The §7 references to §6 (§\ref{sec:limitations:subset-coverage},
§\ref{sec:limitations:r12-singletons},
§\ref{sec:limitations:no-wetlab}, §\ref{sec:limitations:compute})
match the labels used in `paper/sections/06_limitations.tex`
(post-R16b rewrite).  The §7 references to TODO/METRICS_TARGETS.md
(Metrics 1, 2, 3, 4, 7, 16, 17, 18, 19, 20) match the metric
catalogue at `TODO/METRICS_TARGETS.md` §2.

### 5.5 PDF content sanity check
`pdftotext main.pdf` confirms the 7 subsections render in order
with the correct metric numbers:
- §7.1 Protein-Metal Binding Head — Metric 18 AUC ≥ 0.85
- §7.2 Multi-Metal LOMO — Metric 19 AUC ≥ 0.70
- §7.3 Full Lambda Calculus AST — $n_{\text{distinct}}{=}20$
- §7.4 Wet-Lab Prospective — Pearson $r \geq 0.50$
- §7.5 3-Axis SOTA — Vina ≤ -8.5 kcal/mol
- §7.6 NCI-60 — Pearson $r \geq 0.50$
- §7.7 External PlatinAI — A2780cis AUC ≥ 0.85

---

## 6. Honest framing

The rewrite narrows §7 from an 11-item R12-R17 detail-dump
(602 lines) to a 7-item SOTA-anchored honest inventory
(222 lines).  This is a deliberate editorial choice: the §7 of
a Q1 chemistry paper should name **what the paper would do if
the compute / dataset / wet-lab resources were available**, not
the workflow-history of how each deferred item was discovered.

Reviewers should treat:
- **Headline future-work items (§7.1, §7.2, §7.6):**
  protein-metal binding, multi-metal LOMO, NCI-60.
- **Engineering future-work items (§7.3, §7.5):**
  Deflexpressor integration, 300-cell × 3-seed SOTA sweep.
- **External validation future-work items (§7.4, §7.7):**
  wet-lab prospective validation, PlatinAI dataset download
  (prerequisite).

The §7 items are **not** an admission of failure; they are an
honest map of the closest published SOTA targets (Tier-1:
Krasnov, Rusanov, Multimodal Protein-Metal; Tier-2: TransDiffSBDD,
MolCRAFT, AlphaDrug, iMoLD; Tier-3: Boltz-2, Deflexpressor) that
the present paper does not currently measure against but would
measure against in a follow-on paper cycle.

---

End of WF-Paper-Section7 — §7 Future Work rewrite DONE.
