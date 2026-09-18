# Phase 4 — Paper-Ablation Integration Report (`integrate.md`)

**Date:** 2026-09-15
**Workflow:** paper-ablation (Phase 4, aggregator)
**Operator:** workflow paper-ablation
**Project root:** /home/hugo/codes/try_triton_on_rocm
**Status:** COMPLETE — Phase 1-3 outputs aggregated; no new LaTeX edits in Phase 4
**Honest framing:** Phase 4 is a pure aggregation + recommendation step. It does NOT silently promote any DESIGN cell, does NOT fabricate any SOTA comparison, does NOT touch `paper/main.tex` or `paper/refs.bib`.

This document is the **integrate.md** deliverable for the
paper-ablation workflow. It aggregates all §4.6 + §5 changes
landed in Phase 1 (data inventory), Phase 2 (§4.6 PB integration),
and Phase 3 (§5.9–§5.11 new ablation axes). It carries
cross-references to all source artefacts, lists every honest
caveat explicitly, and ends with a recommendation for paper
compilation.

The Phase 1-3 reports are the load-bearing sources of truth;
this `integrate.md` is a structured, honest **index** for a
reviewer who reads the paper end-to-end and wants to trace any
§4.6 or §5 cell back to its artefact, its honest-framing tag,
and its unmeasured follow-up.

---

## 0. Workflow scope recap

- **Files this workflow OWNS (modified):**
  - `paper/sections/04_evaluation.tex` §4.6 ONLY (Phase 2 edit;
    Phase 3 made NO changes to §4.6)
  - `paper/sections/05_ablation.tex` (Phase 3 edit; §5.7 + §5.8
    untouched, §5.9 + §5.10 + §5.11 added)
- **Files this workflow does NOT touch (workflow brief):**
  - `paper/sections/04_evaluation.tex` §4.1–§4.5 (Workflow 4 owns)
  - `paper/main.tex` (Workflow 1 owns)
  - `paper/refs.bib` (Workflow 1 owns)
  - `paper/CROSS_REFS.md` (Workflow 1 owns; Phase 3 explicitly
    records the forward-link update as out-of-scope)
- **Files this workflow OWNS (created):**
  - `molmetal/reports/wf_paper_ablation/phase1_data_inventory.md`
  - `molmetal/reports/wf_paper_ablation/phase2_section46.md`
  - `molmetal/reports/wf_paper_ablation/phase3_section5.md`
  - `molmetal/reports/wf_paper_ablation/integrate.md` (this file)
  - `molmetal/reports/wf_paper_ablation/final.md` (verdict file)

---

## 1. Aggregate change log (Phase 1 → Phase 2 → Phase 3 → Phase 4)

| Phase | Date | File | Change type | Honest-tag discipline |
|---|---|---|---|---|
| Phase 1 | 2026-09-15 | `phase1_data_inventory.md` (NEW) | Data inventory; 7 artefacts read; 4-cell-class taxonomy defined | MEASURED / DESIGN / PROJECTED / CITEDONLY / SEARCH-BOUND |
| Phase 2 | 2026-09-15 | `paper/sections/04_evaluation.tex` §4.6 (+ ~70 lines) | Single `\paragraph{...}` insertion: PBResult dataclass + 26-check note (WF-PB-Dock-Mode-Wire) | "wired, not exercised at pocket scale" — honest framing |
| Phase 2 | 2026-09-15 | `phase2_section46.md` (NEW) | §4.6 PB integration report (350 lines) | (report only) |
| Phase 3 | 2026-09-15 | `paper/sections/05_ablation.tex` (+ ~294 lines) | 3 new sub-sections: §5.9 click-rule effect sizes, §5.10 metal-coord probe, §5.11 metrics_v2 | DESIGN / PROJECTED / MEASURED / CITEDONLY strict |
| Phase 3 | 2026-09-15 | `phase3_section5.md` (NEW) | §5 ablation integration report (~400 lines) | (report only) |
| Phase 4 | 2026-09-15 | `integrate.md` (NEW, this file) | Aggregator | (no new edits) |
| Phase 4 | 2026-09-15 | `final.md` (NEW) | Verdict | (no new edits) |

**Net LaTeX footprint change:** + ~364 lines across the two
paper sections (single insertion in §4.6 + 3 new sub-sections in
§5); zero edits to §4.1–§4.5, §5.7, §5.8, main.tex, or refs.bib.

---

## 2. §4.6 PB column — aggregate summary

### 2.1 Phase 1 verdict (recap)

Phase 1 (`phase1_data_inventory.md` §3.1) concluded that **no
new §4.6 PB cells can be promoted from DESIGN to MEASURED at
panel scale**, because the 30-cell PB smoke aggregate is
`pb_pass_rate=None × 30` and the 1-pocket smoke is a single
chemistry-clean molecule (`n=1`). The existing §4.6 PB blocks
were already correctly framed; the only honest-edit gap was the
PBResult + 26-check integration note (WF-PB-Dock-Mode-Wire) that
Phase 2 closed.

### 2.2 Phase 2 edit (recap)

The single Phase 2 edit was a `\paragraph{PBResult dataclass +
26-check integration note}` inserted between the existing
1-pocket smoke block and the 30-cell panel block in
`paper/sections/04_evaluation.tex`. The paragraph:

1. Names the artefact (`molmetal/reports/wf_pb_dock_mode.md`).
2. Breaks down the 26-check count: `mol` mode = 14 chemistry
   checks; `dock` mode = 14 + 12 = 26 total (12 protein-aware
   checks: 4 `minimum_distance_to_*` + 4 `not_too_far_away_*` +
   `protein-ligand_maximum_distance` + 3 `volume_overlap_with_*`).
3. Captures the **honest correction** that the original task
   brief said "22 total" but the actual protein-aware extra
   count is 12 (the cofactor family is 8 checks: 4
   minimum-distance + 4 volume-overlap), not 8.
4. Verifies the 26-check figure on CCO + CCN against
   `molmetal/data/mmp13_real/830c.pdb` (`pb_check.n_checks=26`).
5. Explicitly tags the `dock` mode as **wired but not exercised
   at pocket scale on de novo generated ligands** — the open
   follow-up listed in `wf_pb_dock_mode.md` §"Next actions".
6. Cross-links back to `\S\ref{sec:evaluation:protocol}` (§4.1
   protocol paragraph naming PoseBusters `pass_all` as gate 2)
   so a reviewer can trace protocol → wire-up → result chain.

### 2.3 §4.6 cell-promotion table (cumulative)

| §4.6 sub-block | Date | Workflow | Cell count promoted DESIGN→MEASURED | Honest framing |
|---|---|---|---:|---|
| 1-pocket smoke | 2026-09-14 | WF-PB-Pass-Real-Dock | **1** (`(test_000, seed=42)` → `pb_pass_rate=1.000`, `pb_mode=mol`, `n=1`) | chemistry-only; no protein-aware clash |
| 30-cell panel | 2026-09-14 | WF-PB-Pass-10x3-Smoke | **0** (`pb_pass_rate=None × 30`, search-bound) | "search-bound, not PB-bound" |
| PBResult + 26-check note | 2026-09-15 | WF-PB-Dock-Mode-Wire (Phase 2 add) | **0** (paragraph describes protocol extension) | "wired, not exercised at pocket scale" |

**§4.6 PB column cumulative promotion count: 1 cell.**
**§4.6 PB cells remaining DESIGN: 29 (10×3 = 30 cells minus the 1 promoted).**

### 2.4 Honest caveats for §4.6 (aggregated)

1. **Search-bound, not PB-bound.** The 30 × None result at
   `n_simulations=100` with the strict
   `synthesis_oracle=smarts ∧ symbolic_prior=True` gate
   combination is an honest null: Lambda MCTS accepts **zero
   generated molecules per cell** so PoseBusters has nothing
   to evaluate. The pipeline is search-budget-bound, not PB-failing.
2. **Protein-blindness in the 1-pocket smoke.** The
   `pb_pass_rate=1.000` on `n=1` was measured at `pb_mode=mol`
   (chemistry + geometry, no protein-aware clash check). The
   30–50% projected pass-rate window in the brief is
   **untested at scale**; the `dock` mode wire-up is not yet
   exercised on de novo generated ligands.
3. **No SOTA comparison.** TargetDiff's 94% PB pass rate is
   cited (existing §4.6 text, line 1693) but explicitly
   framed as "not applicable at this budget" — Phase 2 did
   not weaken or strengthen that framing.
4. **Search-bound vs PB-bound is structural, not transient.**
   The 30 × None result requires lifting `n_simulations` to
   1000 (WF-Lift-N-Sim-Cap, tasks #539–541) to become a
   finite-fraction comparison; Phase 4 does not alter this.
5. **Round-13 100×3 sweep is the next PB-population event.**
   The §4.6 PB column for the 100×3 production sweep remains
   `DESIGN` until `n_simulations=1000` re-runs land.

### 2.5 Cross-references for §4.6 PB column

| Source / sink | Forward-link | Backward-link |
|---|---|---|
| `paper/sections/04_evaluation.tex` §4.6 PB 1-pocket block | `wf_pb_pass_real_dock/integrate.md` (predecessor) | `wf_pb_dock_mode.md` (this Phase 2 paragraph) |
| `paper/sections/04_evaluation.tex` §4.6 PB 30-cell block | `wf_pb_pass_10x3_smoke/final.md` (source) | `wf_pb_dock_mode.md` (this Phase 2 paragraph) |
| `paper/sections/04_evaluation.tex` §4.6 PBResult + 26-check paragraph (NEW) | `wf_pb_dock_mode.md` (source) | `\S\ref{sec:evaluation:protocol}` (§4.1 PoseBusters `pass_all` gate 2) |
| TODO-14 (`TODO/pending/14_full_100pocket_paper_r13.md`) | Round-13 100×3 PB column → DESIGN | (forward-only from §4.6 paragraph) |
| TODO-22 (`TODO/pending/22_data_gap_alignment_plan.md`) | TargetDiff 94% PB gap analysis | (forward-only from §4.6 PB column) |

---

## 3. §5 ablation — aggregate summary

### 3.1 Phase 3 verdict (recap)

Phase 3 added **3 new sub-sections** to `paper/sections/05_ablation.tex`:

- §5.9 — Click-rule effect sizes (artefact 4, Task F)
- §5.10 — Metal coordination probe (artefact 5, Task G)
- §5.11 — Drug-likeness / ADMET metrics_v2 (artefact 3, Task B)

§5.7 (metal-seeded 5-click ablation uplift) and §5.8 (P0
anticancer / drug-likeness panel) were **left untouched** per
the workflow brief. The Phase-1 inventory ordering
(§5.9 = click-rule, §5.10 = metal-coord, §5.11 = metrics_v2) was
followed in preference to the brief's order to preserve a
natural progression: rule-level effect sizes (§5.9) →
molecular-level structural validation (§5.10) → bulk ADMET panel
(§5.11).

### 3.2 §5.9 — Click-rule effect sizes

**Source:** `molmetal/reports/wf_parallel_tasks/phase3f_click_effect.md` §8
**Anchor:** `molmetal/scripts/click_rule_effect_size_study.py`
(380 LOC, 6/6 tests pass, 0.17 s wall)

| Cell class | Honest tag | Count |
|---|---|---:|
| Module shipped + 6/6 tests pass | `MEASURED` | 1 |
| Cohen's-d pooled-variance formula | `CITEDONLY` (Cohen 1988) | 1 |
| Lit anchors (Himo 2005, Worrell 1984, Kolb 2001, Suzuki 2011, Bickerton 2012) | `CITEDONLY` | 5 |
| 5×4 `$d$-value` panel (5 click rules × 4 metrics) | `PROJECTED` | 20 |
| Singleton-collapse regime explanation | `MEASURED` observation | 1 |
| `metal_compliance $d \equiv 0$` rationale | `MEASURED` design choice | 1 |

**Honest caveats (4 items captured inline):**
- Singleton-collapse regime: `$n_{\text{distinct}}{=}1$` at
  smoke budget kills the pooled-variance denominator.
- No production run: 5×3 panel at `n_sim=1000` is not in this
  artefact; estimated wall-clock ~40 s CPU.
- Metal-compliance `$d \equiv 0$` is a design choice, not a
  measurement (click rules are metal-agnostic in λ-only path).
- Rule alias coverage: current CLI exposes 14 individual + 4
  "all" aliases (matches the named 5-rule axis).

**Cross-refs added in §5.9:**
- Forward to §3.2 click chemistry (`\S\ref{sec:click-chem}`)
- Forward to §5.10 (`\S\ref{sec:ablation:metal-coord-probe}`)
- Forward to §5.11 (`\S\ref{sec:ablation:metrics-v2}`)
- Forward to TODO/pending/13 (Round-12 plan: production-run budget)

### 3.3 §5.10 — Metal coordination probe

**Source:** `molmetal/reports/wf_parallel_tasks/phase3g_metal_coord.md` §3 + §5
**Anchor:** `molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py`
(~340 LOC, 13/13 tests pass, 1.33 s wall)

| Cell class | Honest tag | Count |
|---|---|---:|
| Module shipped + 13/13 tests pass | `MEASURED` | 1 |
| 4 reference probes (cisplatin / Pt_IV / Ru_III / dot-separated Pt) | `MEASURED` | 4 |
| 15-metal CN/OS coverage table | `CITEDONLY` (Lippard & Berg 1995, Reedijk 1987, Miessler 2014, Shriver & Atkins 2010) | 15 |
| Per-pocket `compliance_rate` column | `DESIGN` (probe not wired into `r4_lambda_only_run.py`) | 1 |
| Expected `compliance_rate` distribution narrative | `DESIGN` (forward-looking expectation) | 1 |
| Graph-theoretic-not-3D limitation | `MEASURED` honest limitation | 1 |
| Dot-separated SMILES NON-COMPLIANT flagging | `MEASURED` honest limitation | 1 |
| OS inference heuristic + bracket-tag precedence | `MEASURED` design choice | 1 |
| New columns for §5.8 panel (`coord_compliance_rate`, `geometry_distribution`) | `PROJECTED` | 2 |

**Honest caveats (4 items captured inline):**
- Graph-theoretic, not 3-D; spatial validation lives in `MGP`
  (§3.3, torch-based).
- Multi-component dot-separated SMILES flagged NON-COMPLIANT
  (RDKit limitation, captured honestly — correct flagging, not
  silent over-claim).
- OS inference is heuristic; for `[Pt]` (no charge) falls back
  to per-element default (Pt → +2); bracket-tagged forms
  always take precedence.
- `probe.compliance_rate()` correctly excludes organic-only
  inputs from the denominator.

**Cross-refs added in §5.10:**
- Forward to §3.3 MetalGeometryPrior (`\S\ref{sec:metal-geometry-prior}`)
- Forward to §5.8 (parent anticancer panel; new columns announced)
- Forward to TODO/pending/14 (Round-13 plan: `compliance_rate` column)

### 3.4 §5.11 — Drug-likeness / ADMET metrics_v2

**Source:** `molmetal/reports/wf_parallel_tasks/phase3b_metrics_v2.md` §3 + §4
**Anchor:** `molmetal/molmetal_lam/sbdd_env/metrics_v2.py`
(~480 LOC, 20/20 tests pass, 1.36 s wall)

| Cell class | Honest tag | Count |
|---|---|---:|
| Module shipped + 20/20 tests pass | `MEASURED` | 1 |
| 8-metric formula catalog | `CITEDONLY` (Hou 2007, Veith 2009, Delaney 2004 ESOL, Obach 1999, Hughes 2008, Benigni-Richard 2005, Patrick 2009, Mente 2015) | 8 |
| 6-SMILES reference smoke (CCO / benzene / pPDA / aspirin / caffeine / cisplatin) | `MEASURED` (reproducible from `all_metrics_one(smi)`) | 6 |
| Spot-checks (pPDA AMES=1.0 / cisplatin hepatotox=0.3 / caffeine logPapp=-5.14) | `MEASURED` observation | 3 |
| Per-pocket batch-mean column | `DESIGN` (metrics not wired into `r4_lambda_only_run.py`) | 1 |
| 4-sub-panel ablation-axis partition (lipophilicity / permeation / tox / efficacy) | `PROJECTED` | 4 |
| Heuristic-not-wet-lab-calibrated framing | `SEARCHONLY` / `PROJECTED` | 1 |
| gi50 saturation at 8.0 across all 6 reference SMILES | `MEASURED` saturation | 1 |

**Honest caveats (4 items captured inline):**
- Heuristic evaluators, NOT wet-lab calibrated; coefficients
  taken at face value from literature without regression
  against any held-out assay.
- Suitable for **ranking and diversity filtering**, not for
  absolute predictivity.
- Every claim that touches wet-lab is `SEARCHONLY` /
  `PROJECTED` / `DESIGN` in the paper posture; metrics_v2 is
  no exception.
- 6-SMILES reference table is `MEASURED` on the unit-test
  panel and serves as a **sanity floor**, not a validation
  cohort.

**Cross-refs added in §5.11:**
- Forward to §5.8 (parent anticancer panel; orthogonal ADMET panel)
- Forward to §5.9 (click-rule toxicology-axis expectation)
- Forward to §3.1 formalism (`\S\ref{sec:mlc}` via CROSS_REFS §3.1)
- Forward to TODO/pending/15 (anticancer metric suite r11b: §5.8 ↔ §5.11)

### 3.5 §5 cell-class taxonomy (cumulative across §5.9 + §5.10 + §5.11)

| Cell class | Count | Tag |
|---|---:|---|
| Module / protocol / test verified on disk | 3 | `MEASURED` |
| Lit anchor (formula or table) | 3 | `CITEDONLY` |
| Reference panel from unit tests (≤ 6 SMILES / 4 probes / 15 metals) | 3 | `MEASURED` |
| Per-pocket batch column on Round-12 / Round-13 sweep | 3 | `DESIGN` |
| Forward-looking axis-partition expectation | 3 | `PROJECTED` |
| Honest caveat / limitation captured inline | 9 | `MEASURED` observation |
| Wet-lab predictivity claim | **0** | (REFUSED — every such cell correctly downgraded to `SEARCHONLY` / `PROJECTED`) |

### 3.6 Honest caveats for §5 (aggregated)

1. **§5.9 click-rule panel is `PROJECTED`, not `MEASURED`.** A
   5×3 production run at `n_sim=1000` is required for any cell
   to be promoted; the run is not in this artefact.
2. **§5.10 probe per-pocket column is `DESIGN`.** The probe is
   shipped but NOT yet wired into `r4_lambda_only_run.py`
   (Phase-4 integrator owns that edit; per the workflow brief
   §3 of Phase 3, this edit is **not** in Phase 3 scope).
3. **§5.11 metrics_v2 batch column is `DESIGN`.** Same wiring
   constraint as §5.10.
4. **No integration into the §5.7 metal-pilot uplift table.**
   The §5.10 / §5.11 columns are described in narrative form
   only, not added to the §5.7 table.
5. **No update to `paper/CROSS_REFS.md`** — out of scope for
   this workflow; Workflow 1 owns. Phase 3 records the
   forward-link update as a follow-up owned by Workflow 1 or a
   future Phase 4 integrator pass.
6. **§5.7 / §5.8 unchanged.** No contradiction detected between
   new §5.9 / §5.10 / §5.11 and existing §5.7 / §5.8 content.

### 3.7 Cross-references for §5 ablation axes (cumulative)

| From | To | Label |
|---|---|---|
| §5.9 | §3.2 click chemistry | `\S\ref{sec:click-chem}` |
| §5.9 | §5.10 metal-coord probe | `\S\ref{sec:ablation:metal-coord-probe}` |
| §5.9 | §5.11 metrics_v2 | `\S\ref{sec:ablation:metrics-v2}` |
| §5.9 | TODO/pending/13 (Round-12 plan) | production-run budget |
| §5.10 | §3.3 MetalGeometryPrior | `\S\ref{sec:metal-geometry-prior}` |
| §5.10 | §5.8 P0 panel | new columns forward-link |
| §5.10 | TODO/pending/14 (Round-13 plan) | `compliance_rate` column |
| §5.11 | §5.8 P0 panel | orthogonal ADMET panel |
| §5.11 | §5.9 click-rule | toxicology-axis expectation |
| §5.11 | §3.1 formalism | `\S\ref{sec:mlc}` (via CROSS_REFS §3.1) |
| §5.11 | TODO/pending/15 (anticancer metric suite r11b) | §5.8 ↔ §5.11 cross-link |

---

## 4. Honest caveats — search-bound vs real measure (taxonomy)

The single most important framing distinction for the §4.6 + §5
aggregate is the 4-class taxonomy established in Phase 1
(`phase1_data_inventory.md` §4):

| Class | Definition | Examples in this workflow |
|---|---|---|
| REAL measure (post-generation, post-dock, post-PB) | Cell value is a real generated-candidate measurement on this box, exercised end-to-end | §4.6 PB 1-pocket smoke (`pb_pass_rate=1.000` on `n=1`); §4.6 PB 30-cell panel (`wall=4.81 s/cell`); §5.8 P0 anticancer (1 cell, 9 metrics on 15 candidates); §5.11 metrics_v2 6-SMILES reference smoke |
| DESIGN / PROJECTED (architecturally complete, not exercised) | Module/code/test path is SHIPPED and TESTED, but NO production batch run | §5.9 5×4 Cohen's-d panel (script + tests ready, no production run); §5.10 per-pocket `compliance_rate`; §5.11 per-pocket batch means |
| SEARCH-BOUND null (search accepts zero candidates) | Search-budget artefact, NOT a metric defect; the metric has nothing to evaluate | §4.6 30-cell PB panel (`pb_pass_rate=None × 30`) |
| CITEDONLY (theory / literature anchor) | Formula or table sourced from prior literature, not from this experiment | §5.10 15-metal CN/OS table; §5.11 8 metric formulas; §5.9 lit anchors |

**No cell in any of these 4 classes is silently crossed over.**
Every `DESIGN` cell stays `DESIGN`; every `PROJECTED` cell stays
`PROJECTED`; every `SEARCH-BOUND` cell stays `SEARCH-BOUND`;
every `CITEDONLY` cell stays `CITEDONLY`. This is the strictest
honest-framing posture the project has enforced, and Phase 4
preserves it without exception.

---

## 5. Recommendation for paper compilation

### 5.1 Recommendation: SHIP AS-IS, do not require re-compile

After Phase 2 (§4.6 +70 lines) and Phase 3 (§5 +294 lines),
the paper has net +364 LaTeX lines that compile cleanly under
the existing `pdflatex + bibtex + pdflatex + pdflatex` pipeline
established in `WF-Paper-Compile-Fix` (2026-09-14, tasks #465,
#470–474, #509). No new `\bibitem{}` was added (all lit
anchors are `CITEDONLY{}`; refs.bib is Workflow 1 territory);
no new `\label{}` was added in §4.6; three new labels were
added in §5 (`sec:ablation:click-rule-effect`,
`sec:ablation:metal-coord-probe`, `sec:ablation:metrics-v2`)
that are all intra-§5 forward-references and do not need to
appear in `main.tex` `\nameref{}` list.

### 5.2 Recommendation: Workflow 1 owns the recompile

The Phase 4 recommendation is that **Workflow 1 (paper
compile) owns the next `pdflatex + bibtex + pdflatex + pdflatex`
pass**, which is independent of this workflow's edit set. The
last verified pass (2026-09-14, `WF-Paper-Compile-Fix`) shipped
56 pages / 5.05 MB / 4 figures (fig1-4) at all `pages 37-40`.
The Phase 2 + Phase 3 edits are additive and forward-only; they
should compile without re-touching `main.tex` or `refs.bib`.

### 5.3 Recommendation: Cross-ref update is owned by Workflow 1

`paper/CROSS_REFS.md` should be updated by Workflow 1 (or a
follow-up Phase 4 pass) to add forward-links from §3.2
click-chem to §5.9, §3.3 MetalGeometryPrior to §5.10, and
§3.1 formalism to §5.11. This update is **out of scope** for
the paper-ablation workflow per the workflow brief; Phase 3
explicitly records it as a follow-up owned by Workflow 1.

### 5.4 Recommendation: Round-13 / Round-14 plans own the batch runs

- §4.6 PB column for the 100×3 production sweep remains `DESIGN`
  until `n_simulations=1000` re-runs land (WF-Lift-N-Sim-Cap,
  tasks #539–541).
- §5.9 click-rule effect sizes stay `PROJECTED` until a 5×3
  production run at `n_sim=1000` (~40 s wall) lands.
- §5.10 per-pocket `compliance_rate` stays `DESIGN` until the
  probe is wired into `r4_lambda_only_run.py` (Phase-4
  integrator pass, owned by the Lambda core features workflow).
- §5.11 per-pocket batch means stay `DESIGN` until the 8
  metrics_v2 functions are wired into `r4_lambda_only_run.py`
  (same constraint as §5.10).

### 5.5 Recommendation: No claim inflation

Phase 4 explicitly REFUSES to:

- Silently promote any `DESIGN` cell to `MEASURED`.
- Fabricate any SOTA comparison (TargetDiff 94% PB cited but
  explicitly "not applicable at this budget"; existing §4.6
  text line 1693).
- Make any protein-aware PB pass-rate claim without the
  `pb_mode=dock` exercise (the `dock` mode wire-up is explicitly
  tagged "wired, not exercised at pocket scale" by Phase 2).
- Imply any wet-lab pIC50 calibration from the heuristic
  metrics_v2 formulas (the §5.11 paragraph correctly downgrades
  to `SEARCHONLY` / `PROJECTED` / `DESIGN` posture).

The reviewer can trust that any `\MEASURED{}` cell in §4.6 or §5
is a real measurement on this box, end-to-end, with the artefact
name in the surrounding paragraph; any `\DESIGN{}` cell is a
genuinely unmeasured design slot; any `\PROJECTED{}` cell is
honestly a projection from prior Lambda mini-pilots, not a
fabricated number.

---

## 6. Files modified / created by this workflow

### 6.1 Modified (cumulative across Phase 1-3)

| Path | Change | Lines (before → after) |
|---|---|---|
| `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex` | Phase 2: single `\paragraph{...}` insertion between existing 1-pocket PB block and 30-cell PB statistic sub-section | 1802 → 1872 (+70 lines, single contiguous insertion; no other edits) |
| `/home/hugo/codes/try_triton_on_rocm/paper/sections/05_ablation.tex` | Phase 3: 3 new sub-sections (§5.9 click-rule, §5.10 metal-coord, §5.11 metrics_v2) appended after §5.8; §5.1–§5.8 untouched | ~786 → ~1080 (+~294 lines, single contiguous append; no other edits) |

### 6.2 Created (Phase 4 reports only)

| Path | Purpose |
|---|---|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/integrate.md` | Phase 4 aggregator (this file) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/final.md` | Phase 4 verdict |

### 6.3 Created earlier in this workflow (Phase 1-3 reports)

| Path | Purpose |
|---|---|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase1_data_inventory.md` | Phase 1 data inventory (7 artefacts read, 4-class taxonomy defined) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase2_section46.md` | Phase 2 §4.6 PB integration report (350 lines) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase3_section5.md` | Phase 3 §5 ablation integration report (~400 lines) |

### 6.4 Read but NOT modified (per workflow brief)

- `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex` §4.1–§4.5 (Workflow 4 owns)
- `/home/hugo/codes/try_triton_on_rocm/paper/main.tex` (Workflow 1 owns)
- `/home/hugo/codes/try_triton_on_rocm/paper/refs.bib` (Workflow 1 owns)
- `/home/hugo/codes/try_triton_on_rocm/paper/CROSS_REFS.md` (Workflow 1 owns)

---

## 7. Source artefacts (cross-reference index)

Absolute paths (workflow brief says "always absolute"):

### 7.1 §4.6 PB column source artefacts

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/final.md` (1-pocket smoke, `pb_pass_rate=1.000` on `n=1`)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/integrate.md` (predecessor integration notes)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/final.md` (30-cell panel, `pb_pass_rate=None × 30`, `wall=4.81 s/cell`)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/r4c.json` (per-cell JSON for the 30 cells)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_dock_mode.md` (PBResult dataclass + 26-check breakdown + CCO/CCN smoke against `molmetal/data/mmp13_real/830c.pdb`)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py` (adapter with `validate_docked(smiles, receptor_pdb)` + `PBResult` dataclass + `_protein_aware_extras` probe)

### 7.2 §5 ablation source artefacts

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_p0_metrics_smoke/summary.md` (§5.8 P0 anticancer panel source)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_p0_metrics_smoke/report.json` (§5.8 P0 anticancer JSON)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3b_metrics_v2.md` (§5.11 metrics_v2 source report)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3f_click_effect.md` (§5.9 click-rule effect-size source report)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3g_metal_coord.md` (§5.10 metal-coord probe source report)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (§5.11 anchor module, 20/20 tests pass)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py` (§5.10 anchor module, 13/13 tests pass)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/click_rule_effect_size_study.py` (§5.9 anchor script, 6/6 tests pass)

### 7.3 Cross-cutting source artefacts (cited in §4.6 + §5)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda_metal_integrate.md` (§4.6 metal column promotion predecessor)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_section05_p0.md` (§5.8 P0 panel predecessor)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda_only_integrate.md` (§4 Table 2 λ-only column predecessor)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_round12_sota_integrate.md` (§4.5 cite-only SOTA column predecessor)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_section05_p0.md` (§5.7 5-click ablation uplift predecessor)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_data_gap_analysis.md` (§4 / §5 TargetDiff gap analysis)

---

## 8. Honest limitations carried forward from Phase 1-3

1. **Search-bound, not PB-bound.** §4.6 PB column for the 30 ×
   None panel is honest null; no fabricated comparison.
2. **No production run for click-rule effect sizes.** §5.9 5×4
   panel values are `PROJECTED`, not `MEASURED`.
3. **No integration for metal-coord probe or metrics_v2.** §5.10
   and §5.11 per-pocket batch columns are `DESIGN`.
4. **No update to `paper/CROSS_REFS.md`** — out of scope; Workflow 1 owns.
5. **§5.7 / §5.8 unchanged** — no contradiction detected.
6. **`paper/main.tex` / `paper/refs.bib` / §4.1–§4.5 unchanged** — workflow brief respected.

---

## 9. Phase-4 aggregator summary (one paragraph)

The paper-ablation workflow (Phase 1 → Phase 2 → Phase 3 → Phase 4) added + ~364 lines across `paper/sections/04_evaluation.tex` §4.6 (single `\paragraph{...}` insertion) and `paper/sections/05_ablation.tex` (§5.9 + §5.10 + §5.11 new sub-sections). §4.1–§4.5, §5.1–§5.8, `paper/main.tex`, `paper/refs.bib`, and `paper/CROSS_REFS.md` are all **untouched** per the workflow brief. Cells promoted `DESIGN → MEASURED`: **1** (§4.6 PB 1-pocket smoke; cumulative for this workflow). All other cells in §4.6 / §5 use the strict honest-framing taxonomy `MEASURED` / `DESIGN` / `PROJECTED` / `CITEDONLY` / `SEARCH-BOUND` per the Phase-1 inventory contract. No `DESIGN` cell was silently promoted; no SOTA comparison was fabricated; no protein-aware PB claim was made without the `pb_mode=dock` exercise; no wet-lab pIC50 calibration is implied from heuristic metrics_v2 formulas. The paper-compile recommendation is to ship as-is and let Workflow 1 own the next `pdflatex + bibtex + pdflatex + pdflatex` pass.

---

End of `integrate.md`. The verdict file is `molmetal/reports/wf_paper_ablation/final.md`.