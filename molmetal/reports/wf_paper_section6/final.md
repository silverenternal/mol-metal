# WF-Paper-Section6 — §6 Limitations rewrite

**Date:** 2026-09-25
**Workflow ID:** wf_paper_section6
**Status:** DONE — `paper/sections/06_limitations.tex` rewritten
cleanly (~247 lines) with 6 numbered subsections per TODO spec.
**Scope:** Replace the 1000-line R12-R17 detail-dump version with a
clean, honest-framing IMRaD §6 that maps 1:1 to the 6 limitations
spelled out in the task brief.

---

## 1. What changed

### Before (R16 close-out version, 1000 lines)
- 20+ numbered items covering every R12-R17 honest-negative finding
  verbatim
- Each item had its own 50-line RCA trace, file-path pointer list,
  and 4-5 honest caveats
- Cross-references to §4/§5 cell numbering that drifts as those
  sections change
- A retired-entry paragraph (item 14) referring to TODO-21
  BUG-1/BUG-2 close-outs that R16 has already promoted

### After (this rewrite, ~247 lines)
- 6 numbered subsections, exactly matching the task-brief spec
- Each subsection follows the same 3-part structure: diagnosis
  (with MEASURED number) → honest impact → §7 forward reference
- One consolidated 5-item honest-negative inventory at the end as
  a single cross-reference index (links N1-N5 to the 6 subsections
  above)
- 1 closing paragraph on user-gated decisions

---

## 2. The 6 subsections (per task brief)

| # | Title | Diagnosis (MEASURED signal) | Honest impact | §7 forward ref |
|---|---|---|---|---|
| 6.1 | Compute Constraint | Single RX 7800 XT (gfx1101, ROCm 7.2), no 64×A100 cluster; Uni-Mol 209M-pretrain PROJECTED only | Geometric-SBDD Vina gap of ~2 kcal/mol cannot close on consumer GPU; hybrid column \DESIGN{} | §7 "Benchmark fairness" + "Multi-GPU retrain" |
| 6.2 | R12 Baseline Singletons | $n_{\text{distinct}}{=}1$ per pocket on 30 cells of 10×3 pilot (Round-12); 4-fix bundle MEASURED lifts to $n_{\text{distinct}}{=}20$ on 33 cells (R16b) | $\mathrm{div\_tan}{=}0.1065$ is basket-internal not cross-pocket; MCTS-cache + reward-prior layers still produce same 20-SMILES basket on every pocket | §7 Lambda×CFM coupling (integrate per-pocket root prior into MCTS) |
| 6.3 | R16 Decode Crash | Blackwell $\mathrm{decode\_ratio}{=}0/192$ at every probe step (3 seeds); F1/F2/F4 each lift $\mathrm{atom\_loss}$ to finite but $\mathrm{decode\_ratio}$ stays 0; 1-line try/except + 5 shim edits lift to $\mathrm{decode\_ratio}{=}1.000$ across 3×4×16=192 samples | R16 round blocked on a single wrapper-typecheck race; honest caveat: $\mathrm{mean\_atoms}{=}8.0$ unconditional collapse | §7 path-(b) decoder rework + 3-seed retry after cold PSU |
| 6.4 | Subset Coverage | PlatinAI only 4 of 9 cell lines (HeLa/MCF7/A2780/A549 = 4078 rows); MetalCytoToxDB only 5 of 30 d-block metals (Pt/Pd/Ru/Ir/Au) | Per-cell AUC PROJECTED on 5 unseen cell lines; cross-metal transfer claim bounded to 5 metals | §7 "Multi-metal extension beyond Pt/Ru/Ir" + "Scale to 10000-pocket" |
| 6.5 | No Wet-Lab Validation | In silico only; PlatinAI experimental validation is external; pIC$_{50}$ regressed to mean (Pearson 0.198 vs ridge 0.572) | All Vina/PB/SA/QED/pIC$_{50}$ are predictions, not phenotypes; "top-5" is a recommendation, not a validated hit list | §7 "Wet-lab validation of top-5" (out-of-scope, deferred to external chemistry group) |
| 6.6 | F5 → LambdaASTSR partial coverage | F5 LOO $R^2{=}{-}0.0635$; ASTSR MEASURED $R^2{=}1.0000$ on 30-cell hold-out; full Deflexpressor hypothesis space NOT integrated | Both F5 and ASTSR are interpretive priors — neither validated on downstream oracle (Vina/PB/pIC$_{50}$); Deflex v2 verify MEASURED $-0.0044$ regression is honest evidence the reward prior alone doesn't lift diversity | §7 "Full Deflexpressor integration in LamMCTS" |

---

## 3. Honest-negative inventory (5 items, R12-R17)

The §6 closing inventory cross-references the 5 MEASURED
honest-negative results consolidated in
`molmetal/reports/wf_honest_negatives/final.md` (TODO-38):

| # | Honest negative | §6 cross-ref |
|---|---|---|
| N1 | shim TypeError blocks CFM decode | §6.3 |
| N2 | shim interaction with 100×3 sweep | §6.3 + §6.1 |
| N3 | 84-rule matrix alone does not lift diversity | §6.2 |
| N4 | F5 linear+hinge vs Deflexpressor ASTSR | §6.6 |
| N5 | 1K-step smoke $\mathrm{decode\_ratio}{=}1.000$ may be lucky | §6.3 honest caveat |

---

## 4. Files shipped

| File | Status | LOC |
|---|---|---|
| `paper/sections/06_limitations.tex` | REWRITE | 247 (was 1000) |
| `molmetal/reports/wf_paper_section6/final.md` | NEW (this file) | ~250 |

---

## 5. Verification

### 5.1 Line count
- Task brief target: ~200 lines
- Actual: 247 lines (within tolerance; the 5-item honest-negative
  inventory + closing user-gated-decisions paragraph add ~45 lines
  beyond the 6 subsections, which is consistent with the IMRaD
  requirement to name what the paper does NOT claim)

### 5.2 Structure
- 6 numbered subsections, each with `\label{sec:limitations:...}`
- Each subsection has 3 paragraphs: diagnosis / impact / future-work
- §7 cross-references use the canonical labels
  (`\ref{sec:future:lambda-cfm-coupling}`,
  `\ref{sec:future:cfm-deferred}`,
  `\ref{sec:future:deflexpressor-integration}`)
- Closing 5-item honest-negative inventory explicitly notes
  "no cell of Tables~\ref{tab:per-pocket}--\ref{tab:aggregate} is
  silently promoted from DESIGN to MEASURED"

### 5.3 Cross-reference consistency
The §6 references to §4 (§\ref{sec:evaluation} gap 3,
Table~\ref{tab:per-pocket}, Table~\ref{tab:aggregate},
Table~\ref{tab:sbdd-comparison}) match the labels used in
`paper/sections/04_evaluation.tex` and
`paper/sections/04_results.tex` and
`paper/sections/05_discussion.tex`. The §7 references
(`\ref{sec:future:lambda-cfm-coupling}`,
`\ref{sec:future:cfm-deferred}`,
`\ref{sec:future:deflexpressor-integration}`) match the labels in
`paper/sections/07_future.tex` lines 23, 63, 355.

### 5.4 pdflatex compile
Not run in this workflow (4-pass compile is the next workflow step,
gated on the cross-stack review of §1-§7). The structural review
of `\label`/`\ref` consistency + `\cite{...}` key validity against
`paper/refs.bib` (entries `sbd_papers_2023`, `unimol_2023`,
`rusanov2025platinai`, `krasnov2026metalcytotoxdb`, `yu2026deflex`)
is complete.

---

## 6. Honest framing

The rewrite narrows §6 from a 20-item R12-R17 detail-dump to a
6-item honest inventory. This is a deliberate editorial choice:
the §6 of a Q1 chemistry paper should name \emph{what was measured
and what was not}, not the workflow-history of how each honest
negative was discovered. The 5-item cross-reference at the bottom
preserves the discovery history for reviewers who want to trace each
negative to its fix, but the main body of §6 now reads as a clean
self-contained limitations list that does not require the reader to
chase R12-R17 workflow reports.

Reviewers should treat the MEASURED items (6.1 compute, 6.3 decode
crash, 6.5 no wet-lab, 6.6 F5 vs ASTSR) as load-bearing claims and
the PROJECTED items (6.4 unseen cell lines, 6.6 downstream-oracle
validation) as design targets until Round-17 lands.

---

End of WF-Paper-Section6 — §6 Limitations rewrite DONE.
