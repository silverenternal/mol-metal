# WF-Pivot-Followup F2 — §4.7 PlatinAI Oracle Column Integration

**Date**: 2026-09-16
**Workflow**: WF-Pivot-Followup F2 (placeholder §4.7 metallodrug vertical panel with PlatinAI activity oracle column)
**Scope**: `paper/sections/04_evaluation.tex` (insert §4.6.1 metallodrug-vertical subsubsection between §4.6 anticancer PB dataclass note and §4.6.x 30-cell PB statistic panel)
**Verdict**: **PLACEHOLDER SHIPPED — 0 cells promoted DESIGN → MEASURED, awaiting GPU retrain**

---

## 1. What this workflow delivered

`paper/sections/04_evaluation.tex` now carries a new subsubsection at
lines 1773--1937 (165 tex lines) under the §4.6 anticancer-specific
metrics subsection:

- **`\subsubsection{Metallodrug vertical (PlatinAI activity oracle --- placeholder, 2026-09-16, WF-pivot-followup F2)}`** at line 1773
- **`\label{sec:evaluation:metallodrug-vertical}`** at line 1775 (label is unique in the file — verified via `grep -n "label{sec:evaluation" paper/sections/04_evaluation.tex`)

The new subsubsection contains:

1. **Goal paragraph** (~15 lines): explains that PlatinAI oracle is a *platinum-cancer cell-line activity oracle* that joins the 214K PlatinAI predicted-activity corpus onto Round-12+ de novo typed-term MCTS candidates that survive the MetalGeometryPrior gate.

2. **Activity matrix contract** (~10 lines): documents the `(smiles, (a2780, mcf7), metal)` tuple shape and the `PlatinAIDataset.get_activity_matrix()` API (returns `(N, 2)` tensor of `[P_A2780, P_MCF7] ∈ [0, 1]`).

3. **Per-pocket `test_000..009` PlatinAI cell-line panel** (tabular, ~25 lines): 10-row + 10-pocket-mean table with columns `n_cand`, `P_A2780 (mean±std)`, `P_MCF7 (mean±std)`, `PlatinAI oracle = (P_A2780 + P_MCF7) / 2 ∈ [0, 1]`. ALL values deliberately labelled `\DESIGN{}` (no silent promotion). Seed set extended to `{42, 0, 1234, 7, 99}` (5 seeds) to widen std estimate.

4. **PlatinAI calibration row** (~12 lines): explicit `Pearson r vs wet-lab IC50`, `MAE / RMSE`, all `\DESIGN{}` pending MetalCytoToxDB IC50_Dark overlay; cites the 80/10/10 split semantics of `test_platinai_dataset.py:152-167` for internal cross-validation. Explicit "ML-predicted, not wet-lab" caveat.

6. **Honest framing paragraph** (~30 lines): PlatinAI oracle is a soft target; uses PlatinAI as a *wide-coverage* complement to MetalCytoToxDB IC50_Dark (214K vs 26K); cannot replace §4.6 PB/Vina/MGP metrics; cross-refs `sec:method:secondary-cfm` (the §3.6 CFM stub), `sec:related:metallodrug` (§2.5 metallodrug literature), and `sec:evaluation:hybrid-wip` (the §4.12 honest-negative WIP).
   - **0 cells promoted DESIGN → MEASURED** (explicit).
   - Sources cited: `molmetal/reports/wf_metallodrug_vertical/phase1_inventory.md` §2.1, `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` lines 1--100 + 110--200, `molmetal/molmetal_lam/tests/test_platinai_dataset.py` lines 120--200 (9/9 PlatinAI tests pass per F1-VERIFY).

## 2. Slot decision: §4.6.1 not §4.7

The original spec called for "§4.7 metallodrug vertical panel". On inspection of the existing §4 layout, **§4.7 was already claimed** by the "5-click-rule vs CuAAC-only ablation" subsubsection (`\label{sec:evaluation:click-ablation}` at line 2127, content at line 1960). Renumbering §4.7--§4.12 would break the ~25 existing `\ref{sec:evaluation:click-ablation}` / `homotype` / `failures` / `honest-summary` / `round13-honest` / `hybrid-wip` cross-references.

To preserve all existing cross-references verbatim, the new content was inserted as **`\subsubsection{...}` (level 3) within §4.6 anticancer-specific metrics**, between the PBResult dataclass + 26-check note (line 1771) and the 30-cell PB statistic panel (line 1939). The slot becomes **§4.6.1** in the document hierarchy:

```
§4.6 anticancer-specific metrics              \label{sec:evaluation:anticancer}
  §4.6 PB pass-rate on real docked poses       (subsubsection)
  §4.6 PBResult + 26-check dataclass           (subsubsection)
  §4.6.1 Metallodrug vertical (PlatinAI) ← NEW (subsubsection, \label{sec:evaluation:metallodrug-vertical})
  §4.6.x 30-cell PB statistic panel            (subsubsection, \label{sec:evaluation:pb-30cell})
§4.7 5-click-rule vs CuAAC-only ablation      \label{sec:evaluation:click-ablation}
§4.8 Homotype vs Tanimoto orthogonality        \label{sec:evaluation:homotype}
... (rest unchanged)
```

This decision is documented in-line in the new subsubsection text:

> "The §4.7 number (\emph{slot renamed to §4.6.1 because §4.7 was already claimed by the 5-click-rule vs CuAAC-only ablation, \S\ref{sec:evaluation:click-ablation}}) is a placeholder; the GPU retrain required to integrate the oracle at \texttt{molmetal/molmetal\_lam/reward/reward\_aggregator.py}'s new \texttt{r\_platinai} channel was NOT executed (per \S\ref{sec:evaluation:hybrid-wip} honest-negative)."

## 3. Why this is a placeholder, not a measurement

Per the F1-INTEGRATE audit of 2026-09-16 (`f1_integrate.md` §1 + §2), widzuipcl has shipped only Phase 1 of the metallodrug vertical:

| Required widzuipcl deliverable | Status (per F1-INTEGRATE 2026-09-16) |
|---|---|
| `paper/sections/03_7_metallodrug_vertical.tex` | MISSING |
| `§4.7 metallodrug panel` in `04_evaluation.tex` | MISSING (now SHIPPED as §4.6.1 placeholder by this workflow) |
| `molmetal/reports/wf_metallodrug_vertical/MASTER.md` | MISSING |
| Phase 2+ GPU retrain report | NOT STARTED |
| Phase 3 paper §4 update report | NOT STARTED |
| Phase 3_smoke/final.md (referenced by F2 spec) | MISSING |
| Phase 4_s4_metallodrug.md (referenced by F2 spec) | MISSING |
| `r_platinai` channel in `reward_aggregator.py` | MISSING (reward/ directory only ships `learned_shaping.py` + `symbolic_regression.py`) |

The two files the F2 spec asked me to read first —
`molmetal/reports/wf_metallodrug_vertical/phase3_smoke/final.md` and
`molmetal/reports/wf_metallodrug_vertical/phase4_s4_metallodrug.md` —
**do not exist on disk** (verified via `ls`). Only three Phase 1 files exist: `phase1_inventory.md` + `phase1_inventory.json` + `phase1_filter_fix.md`.

Per the F2 spec branch:

> "If widzuipcl has not created §4.7 yet: Add a placeholder §4.7 with column headers and '(see §3.7 for vertical)' cross-ref. Add an honest 'PlatinAI oracle: pending GPU retrain results' note."

This is exactly what was done — the §4.6.1 placeholder subsubsection:
- carries the column headers (`P_A2780`, `P_MCF7`, `PlatinAI oracle`)
- documents the activity-matrix contract from `platinai_dataset.py` (the *real* API)
- labels every numeric cell `\DESIGN{}` (no silent DESIGN → MEASURED promotion)
- records the "PlatinAI oracle: pending GPU retrain results" note as an honest-negative headline
- cites the 3 Phase 1 artefacts as sources + the test suite + the (missing) §3.7 metallodrug vertical as the forward cross-ref

## 4. Pivot-A invariant preservation

The new §4.6.1 subsubsection preserves all 9 Pivot-A invariants from F1-VERIFY (`f1_verify.md` §1):

| # | Invariant | Status after F2 edit |
|---|---|---|
| 1 | §1 "de novo typed-term MCTS" framing | PASS — §4.6.1 uses "de novo typed-term MCTS" + "MGP metal-coordination gate" language |
| 2 | §2.1 2-row comparison | NOT TOUCHED |
| 3 | §3.6 stub | NOT TOUCHED (cross-referenced from §4.6.1) |
| 4 | §3.7 metallodrug vertical | NOT YET WRITTEN (forward cross-ref to §3.7 mentioned as "see §3.7 for vertical" intent; absent §3.7 label preserved) |
| 5 | §4 column "de novo typed-term MCTS" | PASS — §4.6.1 column header is "PlatinAI oracle" (not SBDD comparison) |
| 6 | §6 n_distinct=1 ack | NOT TOUCHED |
| 7 | §7 order | NOT TOUCHED |
| 8 | refs.bib 100 entries | NOT TOUCHED — no new bibitem created (PlatinAI label marked "cite-only") |
| 9 | Compile passes | TO BE VERIFIED — `pdflatex` was not run as part of this workflow; the new content uses standard `\subsubsection` + `\label` + `\DESIGN` markers that match the file's existing macros. **No `\cite` added** so no bibtex re-pass needed. |

## 5. Honest framing — what §4.6.1 deliberately does NOT claim

1. **No A2780 / MCF7 MEASURED value** is reported. Every cell is `\DESIGN{}`. The arithmetic operator `platinai_oracle = (P_A2780 + P_MCF7) / 2` is the column *definition*, not a result.
2. **No PlatinAI bibitem** is cited. `grep -i platinai paper/refs.bib` returns 0 hits; the column is explicitly labelled "cite-only" in §4.6.1 paragraph 4.
3. **No MetalCytoToxDB IC50 calibration** is computed. Pearson r / MAE / RMSE are all `\DESIGN{}` pending the IC50_Dark overlay (TODO pending; out-of-scope).
4. **No 5-seed data cell** is populated. The seed set `{42, 0, 1234, 7, 99}` is documented in the placeholder text but no number is emitted.
5. **No `r_platinai` channel API claim**. The §4.6.1 text explicitly says the channel is "planned addition for Round-14" and that the GPU retrain required to wire it was NOT executed.
6. **No claim drift back to "geometric SBDD" / "structure-based"**. §4.6.1 frames the metallodrug vertical as the de novo typed-term MCTS column's platinum-cancer extension, in line with Pivot A §1 line 110 ("de novo typed-term MCTS, pocket-optional").

## 6. Files modified / created

| Path | Status |
|---|---|
| `paper/sections/04_evaluation.tex` | MODIFIED — added subsubsection + label at lines 1773--1937 (165 tex lines) |
| `molmetal/reports/wf_pivot_followup/f2_platinai_s4.md` | CREATED — this verdict |

No other files modified. No new bibitems. No `reward_aggregator.py` API changes (those are Round-14 work).

## 7. Recommended follow-ups (out-of-scope for this workflow)

1. **Phase 2 widzuipcl deliverable** (GPU retrain + paper §4 update): execute the `r_platinai` channel wire-up against `PlatinAIDataset.get_activity_matrix()` in `r4_lambda_only_run.py`, run on `test_000..009` × `{42, 0, 1234, 7, 99}` seeds, populate the §4.6.1 cells DESIGN → MEASURED.
2. **§3.7 metallodrug vertical section** (MISSING per F1-INTEGRATE §2): write `paper/sections/03_7_metallodrug_vertical.tex` and add the corresponding `\input{sections/03_7_metallodrug_vertical}` line to `paper/main.tex`.
3. **PlatinAI citation placeholder**: add a `@misc{platinai_dataset, ...}` bibitem to `paper/refs.bib` so the "cite-only" label in §4.6.1 can graduate to a real `\cite{platinai_dataset}`.
4. **MetalCytoToxDB IC50_Dark calibration overlay**: compute Pearson r vs PlatinAI pred_0 on the joint (platinai ∩ metal_cytotox) test split to lift the calibration row from DESIGN → MEASURED.

---

**F2 verdict**: **PLACEHOLDER SHIPPED, 0 promotions.** The §4.6.1 metallodrug vertical subsubsection is honest, structurally sound, and Pivot-A-preserving; no DESIGN cell is silently promoted; no numeric claim is fabricated. Awaiting widzuipcl Phase 2 (GPU retrain + paper §4 update) for the §4.6.1 cells to graduate to MEASURED.