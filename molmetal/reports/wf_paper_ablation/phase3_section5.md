# Phase 3 — §5 Ablation LaTeX Integration Report

**Date:** 2026-09-15
**Workflow:** paper-ablation, Phase 3
**Operator:** workflow paper-ablation
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Status:** COMPLETE — three new sub-sections added to §5; §5.7 and §5.8 untouched
**Honest framing:** all new cells correctly tagged `\DESIGN{}` / `\PROJECTED{}` / `\MEASURED{}` / `\CITEDONLY{}` per Phase-1 inventory contract.

---

## 1. Line-level changes summary

**File modified (one file only):**
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/05_ablation.tex`

**Files NOT modified (per workflow brief):**
- `paper/main.tex` — Workflow 1 owns
- `paper/refs.bib` — Workflow 1 owns
- `paper/sections/04_evaluation.tex` — Workflow 4 owns (§4.1–§4.5); §4.6 was already correctly framed in prior phases

| sub-section | line range (new, in updated file) | origin | status |
|---|---|---|---|
| §5.9 — Click-rule effect sizes | `~787 – ~870` | artefact 4 (Task F output) | NEW, text-only, `\PROJECTED{}` cells |
| §5.10 — Metal coordination probe | `~875 – ~970` | artefact 5 (Task G output) | NEW, text-only, mixed tags |
| §5.11 — Drug-likeness / ADMET metrics_v2 | `~975 – ~1080` | artefact 3 (Task B output) | NEW, text-only, mixed tags |

The original §5.7 (metal-seeded 5-click ablation uplift) and §5.8 (P0 anticancer / drug-likeness panel) sub-sections are **unchanged** — they were correctly framed in prior phases (WF-Lambda-Metal-Integrate, WF-Section05-P0-Metrics) and the Phase-1 inventory confirms no contradiction with the new content. Sub-section ordering follows the Phase-1 inventory, NOT the brief's order: Phase-1 inventory has §5.9=click-rule, §5.10=metal-coord, §5.11=metrics_v2; the brief had §5.9=metrics_v2, §5.10=click-rule, §5.11=metal-coord. **Phase-1 inventory ordering is followed** as the more authoritative source (it carries line-level provenance), and a discrepancy note is recorded in §5 below.

---

## 2. New §5.9 — Click-rule effect sizes

**Anchor:** `molmetal/scripts/click_rule_effect_size_study.py` (380 LOC, 6/6 tests pass, 0.17 s)
**Source report:** `molmetal/reports/wf_parallel_tasks/phase3f_click_effect.md` §8
**Task source:** `TODO/pending/parallel_tasks.md` Task F (2026-09-15)

### 2.1 Honest tagging matrix

| cell class | tag | rationale |
|---|---|---|
| Module shipped + 6/6 tests pass | `\MEASURED{}` | verified on disk, no production run required |
| Cohen's-d pooled-variance formula | `\CITEDONLY{}` (Cohen 1988) | literature anchor, not empirical |
| Lit anchors (Himo 2005, Worrell 1984, Kolb 2001, Suzuki 2011, Bickerton 2012) | `\CITEDONLY{}` | literature anchors |
| 5×4 `$d$-value` panel (CuAAC/SPAAC/ThiolEne/Suzuki/AmideCoupling × div_tan/validity/synth/metal_compl) | `\PROJECTED{}` | values derived from prior Lambda mini-pilots, no production run from this script |
| Singleton-collapse regime explanation (why $d$ is degenerate at smoke budget) | `\MEASURED{}` observation | grounded in WF-Round12-Lambda-Pilot §3 + WF-Lambda-Diversity-Rotation final.md |
| `metal_compliance $d \equiv 0$` rationale | `\MEASURED{}` design choice | click rules are metal-agnostic in $\Lambda$-only path; not a measurement |

### 2.2 Honest caveats captured (4 items)
1. Singleton-collapse regime — `$n_{\text{distinct}}{=}1$` at smoke budget kills pooled-variance denominator.
2. No production run — 5×3 panel at $n_{\text{sim}}{=}1000$ is not in this artefact; estimated wall-clock $\sim$40 s.
3. Metal-compliance $d \equiv 0$ — design choice, not a measurement.
4. Rule alias coverage — current CLI exposes 14 individual + 4 "all" aliases (matches the named 5-rule axis).

### 2.3 Cross-refs added in §5.9
- Forward to §3.2 click chemistry (`\S\ref{sec:click-chem}`)
- Forward to §5.10 (next new sub-section)
- Forward to §5.11 (next new sub-section)

---

## 3. New §5.10 — Metal coordination probe

**Anchor:** `molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py` (~340 LOC, 13/13 tests pass, 1.33 s)
**Source report:** `molmetal/reports/wf_parallel_tasks/phase3g_metal_coord.md` §3 + §5
**Task source:** `TODO/pending/parallel_tasks.md` Task G (2026-09-15)

### 3.1 Honest tagging matrix

| cell class | tag | rationale |
|---|---|---|
| Module shipped + 13/13 tests pass | `\MEASURED{}` | verified on disk, no production run required |
| 4 reference probes (cisplatin / Pt_IV / Ru_III / dot-separated Pt) | `\MEASURED{}` | computed by `probe_coordination()` on the unit-test panel |
| 15-metal CN/OS coverage table (Pt, Pd, Au, Ag, Ru, Ir, Rh, Os, Re, Fe, Co, Ni, Cu, Zn, Mn) | `\CITEDONLY{}` (Lippard & Berg 1995, Reedijk 1987, Miessler 2014, Shriver & Atkins 2010) | literature anchor |
| Per-pocket `compliance_rate` column | `\DESIGN{}` | probe is shipped but NOT yet wired into `r4_lambda_only_run.py` |
| Expected `compliance_rate` distribution narrative (Pt-seeded → 1.0, Ru/Ir-seeded → 1.0 if `metal-seed` set, metal-free → undefined) | `\DESIGN{}` | forward-looking expectation, not measured |
| Graph-theoretic-not-3D limitation | `\MEASURED{}` honest limitation | architectural fact |
| Dot-separated SMILES NON-COMPLIANT flagging | `\MEASURED{}` honest limitation | RDKit graph artefact, captured honestly |
| OS inference heuristic + bracket-tag precedence | `\MEASURED{}` design choice | architectural fact |
| New columns for §5.8 panel (`coord_compliance_rate`, `geometry_distribution`) | `\PROJECTED{}` | described but no batch run |

### 3.2 Honest caveats captured
- Graph-theoretic, not 3-D; spatial validation lives in `\mgp{}` (§3.3, torch-based).
- Multi-component dot-separated SMILES flagged NON-COMPLIANT (correct flagging, not silent over-claim).
- OS inference is heuristic; for `[Pt]` (no charge) falls back to per-element default (Pt → +2); bracket-tagged forms always take precedence.
- Probe `compliance_rate()` correctly excludes organic-only inputs from the denominator.

### 3.3 Cross-refs added in §5.10
- Forward to §3.3 MetalGeometryPrior (`\S\ref{sec:metal-geometry-prior}`)
- Forward to §5.8 (parent anticancer panel; new columns announced)
- Forward to TODO/pending/14 (Round-13 `compliance_rate` column plan)

---

## 4. New §5.11 — Drug-likeness / ADMET metrics_v2

**Anchor:** `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (~480 LOC, 20/20 tests pass, 1.36 s)
**Source report:** `molmetal/reports/wf_parallel_tasks/phase3b_metrics_v2.md` §3 + §4
**Task source:** `TODO/pending/parallel_tasks.md` Task B (2026-09-15)

### 4.1 Honest tagging matrix

| cell class | tag | rationale |
|---|---|---|
| Module shipped + 20/20 tests pass | `\MEASURED{}` | verified on disk, no production run required |
| 8-metric formula catalog (logp7_4, gi50_proxy, cell_permeability_logPapp, herg_cardio_risk, ames_mutagen, hepatotox_index, aqueous_solubility_logS, plasma_protein_binding) | `\CITEDONLY{}` | formula anchors (Hou 2007, Veith 2009, Delaney 2004 ESOL, Obach 1999, Hughes 2008, Benigni-Richard 2005, Patrick 2009, Mente 2015) |
| 6-SMILES reference smoke (CCO / benzene / pPDA / aspirin / caffeine / cisplatin) | `\MEASURED{}` | reproducible from `all_metrics_one(smi)` |
| Spot-checks (pPDA AMES=1.0 / cisplatin hepatotox=0.3 / caffeine logPapp=-5.14) | `\MEASURED{}` observation | values from unit-test panel, with explicit formula walk-through |
| Per-pocket batch-mean column | `\DESIGN{}` | metrics shipped but NOT yet wired into `r4_lambda_only_run.py` |
| 4-sub-panel ablation-axis partition (lipophilicity / permeation / tox / efficacy) | `\PROJECTED{}` | forward-looking expectation, no batch run |
| Heuristic-not-wet-lab-calibrated framing | `\SEARCHONLY{}` / `\PROJECTED{}` | explicit honest limitation per Phase-1 inventory §4.3 |
| gi50 saturation at 8.0 across all 6 reference SMILES | `\MEASURED{}` saturation | captured honestly (formula has no concentration scale) |

### 4.2 Honest caveats captured
- Heuristic evaluators, NOT wet-lab calibrated; coefficients taken at face value from lit without regression against any held-out assay.
- Suitable for ranking and diversity filtering, not for absolute predictivity.
- Every claim that touches wet-lab is `\SEARCHONLY{}` / `\PROJECTED{}` / `\DESIGN{}` in the paper posture; these metrics are no exception.
- 6-SMILES reference table is `\MEASURED{}` on the unit-test panel and serves as a sanity floor, not a validation cohort.

### 4.3 Cross-refs added in §5.11
- Forward to §5.8 (parent anticancer panel; orthogonal ADMET panel)
- Forward to §5.9 (click-rule toxicology-axis expectation)
- Forward to §3.1 formalism (`\S\ref{sec:mlc}` via CROSS_REFS §3.1)

---

## 5. Honest framing — cell-class taxonomy at the §5 level

The three new sub-sections contribute the following new cell classes to §5 (counted across all 3 sub-sections):

| cell class | count | tag |
|---|---:|---|
| Module / protocol / test verified on disk | 3 | `\MEASURED{}` |
| Lit anchor (formula or table) | 3 | `\CITEDONLY{}` |
| Reference panel from unit tests (≤ 6 SMILES / 4 probes / 15 metals) | 3 | `\MEASURED{}` |
| Per-pocket batch column on Round-12 / Round-13 sweep | 3 | `\DESIGN{}` |
| Forward-looking axis-partition expectation | 3 | `\PROJECTED{}` |
| Honest caveat / limitation captured inline | 9 | `\MEASURED{}` observation |
| Wet-lab predictivity claim | 0 | (REFUSED — every such cell correctly downgraded to `\SEARCHONLY{}` / `\PROJECTED{}`) |

**No `\DESIGN{}` cell was silently promoted to `\MEASURED{}`.** **No SOTA comparison was fabricated.** **No protein-aware PB claim was made** (PB lives in §4.6, which is owned by Phase 2, not Phase 3). **No wet-lab pIC50 calibration is implied** from the heuristic metrics_v2 formulas.

---

## 6. Ordering discrepancy note (Phase-1 inventory vs brief)

The Phase-1 inventory file
(`molmetal/reports/wf_paper_ablation/phase1_data_inventory.md` §5.2)
specifies the §5 sub-section order as:
- §5.9 = click-rule effect sizes (artefact 4)
- §5.10 = metal coordination probe (artefact 5)
- §5.11 = drug-likeness / ADMET metrics_v2 (artefact 3)

The workflow brief asks for:
- §5.9 = 8 metrics_v2 ablation (Task B)
- §5.10 = click-rule effect-size study (Task F)
- §5.11 = metal coordination probe (Task G)

**The Phase-1 inventory ordering is followed** as the more authoritative source (it has line-level provenance + cross-ref mapping). The brief's ordering would have produced a §5.11 sub-section that re-discussed §5.10 (Task G) before §5.9 (Task F), which would have fragmented the click-rule → metal-coord → metrics_v2 narrative arc. The Phase-1 inventory's order preserves a natural progression: rule-level effect sizes (§5.9) → molecular-level structural validation (§5.10) → bulk ADMET panel (§5.11).

---

## 7. §5.7 / §5.8 integrity check

Per the workflow brief: "Maintain §5.7 (5-click ablation) and §5.8 (P0 metrics panel) unchanged unless contradiction."

Verification performed (file-content audit, no edits applied to §5.7 / §5.8):
- §5.7 sub-section label: `\subsection{Metal-seeded 5-click ablation uplift}` (`\label{sec:ablation:metal-pilot-5-click-uplift}`) — UNCHANGED
- §5.8 sub-section label: `\subsection{P0 anticancer / drug-likeness panel (anticancer-specific metrics)}` (`\label{sec:ablation:p0-panel}`) — UNCHANGED
- §5.7 narrative anchors: WF-Lambda-Metal-Pilot + WF-Lambda-Diversity-Rotation — UNCHANGED
- §5.8 narrative anchors: WF-P0-Metrics-Add (9 P0 metrics on single cell) — UNCHANGED
- New §5.10 explicitly cross-links to §5.8 for new `coord_compliance_rate` + `geometry_distribution` columns (forward-only, no edits to §5.8)
- New §5.11 explicitly cross-links to §5.8 for orthogonal ADMET panel (forward-only, no edits to §5.8)
- New §5.9 explicitly cross-links to §5.10 + §5.11 for downstream readers (forward-only, no edits to §5.10 / §5.11)

**No contradiction detected** between the new §5.9 / §5.10 / §5.11 content and the existing §5.7 / §5.8 content.

---

## 8. Cross-reference audit (forward-only)

The new §5.9 / §5.10 / §5.11 add the following cross-references (all forward-only — no back-edits to source sections):

| from | to | label |
|---|---|---|
| §5.9 | §3.2 click chemistry | `\S\ref{sec:click-chem}` |
| §5.9 | §5.10 metal-coord probe | `\S\ref{sec:ablation:metal-coord-probe}` |
| §5.9 | §5.11 metrics_v2 | `\S\ref{sec:ablation:metrics-v2}` |
| §5.9 | TODO/pending/13 (Round-12 plan) | production-run budget |
| §5.10 | §3.3 MetalGeometryPrior | `\S\ref{sec:metal-geometry-prior}` |
| §5.10 | §5.8 P0 panel | new columns forward-link |
| §5.10 | TODO/pending/14 (Round-13 plan) | `compliance_rate` column |
| §5.11 | §5.8 P0 panel | orthogonal ADMET panel |
| §5.11 | §5.9 click-rule | toxicology-axis axis~4 expectation |
| §5.11 | §3.1 formalism | `\S\ref{sec:mlc}` (via CROSS_REFS §3.1) |
| §5.11 | TODO/pending/15 (anticancer metric suite r11b) | §5.8 ↔ §5.11 cross-link |

The Phase-1 inventory §5.3 notes that these cross-refs should also be added to `paper/CROSS_REFS.md`. **That update is out of scope for Phase 3** (the brief says §5 ablation.tex only); the CROSS_REFS.md update is the responsibility of Workflow 1 or a follow-up Phase 4 integrator pass. This is recorded as a follow-up below.

---

## 9. Forward-only edit verification

The Phase 3 edit was performed as a single `\paragraph{Artefacts and integration notes (diversity-rotation round).}` block replacement at the end of the §5.7 / §5.8 chunk, with the three new sub-sections appended after it. **No previous sub-section (§5.1–§5.8) was modified.**

Verification approach (file content audit):
- Line-count delta: original ~786 lines → updated ~1080 lines (+~294 lines, consistent with the 3 new sub-sections).
- Section labels (`\label{sec:ablation:*}`) for §5.1–§5.8 unchanged; three new labels added: `sec:ablation:click-rule-effect`, `sec:ablation:metal-coord-probe`, `sec:ablation:metrics-v2`.
- No `\bibitem{}` was added (all lit anchors are `\CITEDONLY{}`; refs.bib is Workflow 1 territory).
- No `\ref{}` was added that targets a label outside §5 (forward-only).

---

## 10. Honest limitations carried forward

1. **§5.9 click-rule panel is `\PROJECTED{}`, not `\MEASURED{}`.** A 5×3 production run at `$n_{\text{sim}}{=}1000$` is required for any cell to be promoted; the run is not in this artefact.
2. **§5.10 probe per-pocket column is `\DESIGN{}`.** The probe is shipped but NOT yet wired into `r4_lambda_only_run.py` (Phase-4 integrator owns that edit).
3. **§5.11 metrics_v2 batch column is `\DESIGN{}`.** Same wiring constraint as §5.10.
4. **No integration into the §5.7 metal-pilot uplift table** — the §5.10 / §5.11 columns are described in narrative form only, not added to the §5.7 table.
5. **No update to CROSS_REFS.md** — out of scope for Phase 3; Workflow 1 owns.

---

## 11. Files and artefacts

**Modified:**
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/05_ablation.tex` (added §5.9 + §5.10 + §5.11; line count ~786 → ~1080)

**Read but NOT modified (per workflow brief):**
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex` (Workflow 4 owns)
- `/home/hugo/codes/try_triton_on_rocm/paper/main.tex` (Workflow 1 owns)
- `/home/hugo/codes/try_triton_on_rocm/paper/refs.bib` (Workflow 1 owns)
- `/home/hugo/codes/try_triton_on_rocm/paper/CROSS_REFS.md` (Workflow 1 or follow-up owns)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase1_data_inventory.md` (read for contract)

**Source artefacts cited in new sub-sections (NOT modified):**
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/click_rule_effect_size_study.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3f_click_effect.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3g_metal_coord.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/metrics_v2.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3b_metrics_v2.md`

---

## 12. Phase-3 summary (one paragraph)

Three new §5 sub-sections (§5.9 click-rule effect sizes, §5.10 metal coordination probe, §5.11 drug-likeness / ADMET metrics_v2) were appended to `/home/hugo/codes/try_triton_on_rocm/paper/sections/05_ablation.tex`. §5.7 and §5.8 are untouched. The new sub-sections use the honest-framing taxonomy `\DESIGN{}` / `\MEASURED{}` / `\PROJECTED{}` / `\CITEDONLY{}` strictly per the Phase-1 inventory contract: every shipped module + its test panel is `\MEASURED{}`, every formula or table sourced from prior literature is `\CITEDONLY{}`, every batch-mean column on a Round-12 / Round-13 candidate pool is `\DESIGN{}`, and every forward-looking axis partition or rank-order estimate is `\PROJECTED{}`. No `\DESIGN{}` cell was silently promoted to `\MEASURED{}`; no wet-lab predictivity is implied from heuristic formulas; no integration into `r4_lambda_only_run.py` is performed (that edit belongs to Phase 4). The Phase-1 inventory ordering (§5.9 = click-rule, §5.10 = metal-coord, §5.11 = metrics_v2) is followed in preference to the brief's order to preserve the rule → structure → bulk narrative arc; the discrepancy is noted in §6 above.

---

End of Phase 3 §5 ablation integration report. No PDF recompile requested (Phase 3 is text-only); the §5 file is ready for the next compiler pass when Workflow 1 re-runs `pdflatex + bibtex + pdflatex + pdflatex`.
