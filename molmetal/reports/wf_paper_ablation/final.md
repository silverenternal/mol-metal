# Phase 4 — Paper-Ablation Final Verdict (`final.md`)

**Date:** 2026-09-15
**Workflow:** paper-ablation (Phase 4, final verdict)
**Operator:** workflow paper-ablation
**Project root:** /home/hugo/codes/try_triton_on_rocm
**VERDICT:** **PARTIAL SUCCESS** — text-only honest-framing integration ship; the 1-cell §4.6 PB promotion stands; the 3 new §5 sub-sections ship as DESIGN / PROJECTED / MEASURED / CITEDONLY honest cells; 0 silently-promoted cells; 0 fabricated SOTA comparisons; 0 wet-lab overclaims.

This document is the **`final.md` verdict** for the paper-ablation
workflow (Phase 1 → Phase 2 → Phase 3 → Phase 4). It carries the
single-line **VERDICT**, enumerates **sections updated**, the
**MEASURED cells promoted** (cumulative), the **honest caveats**
(aggregated), and the **recommendation for next steps**.

The aggregator (`integrate.md`) is the load-bearing cross-reference
index for a reviewer who reads the paper end-to-end and wants to
trace any §4.6 or §5 cell back to its artefact, honest-framing tag,
and unmeasured follow-up. This `final.md` is the short-form
decision document.

---

## 0. VERDICT (single line)

> **PARTIAL SUCCESS** — text-only honest-framing integration ship;
> 1 cell promoted `DESIGN → MEASURED` in §4.6 (PB 1-pocket smoke);
> 3 new §5 sub-sections shipped (§5.9 click-rule, §5.10 metal-coord,
> §5.11 metrics_v2) with strict honest-framing taxonomy; 0 silent
> promotions; 0 fabricated SOTA comparisons; 0 protein-aware PB
> claims; 0 wet-lab overclaims.

The "PARTIAL SUCCESS" qualifier (not full SUCCESS) reflects three
honest gaps that this workflow **did not close** and that the
paper should not pretend to close:

1. The §4.6 PB column for the 100×3 production sweep is **still
   DESIGN** (search-bound, not PB-bound).
2. §5.10 / §5.11 per-pocket batch columns are **still DESIGN**
   (modules shipped but NOT wired into `r4_lambda_only_run.py`).
3. §5.9 click-rule effect-size 5×4 panel is **still PROJECTED**
   (no production run from the shipped script).

These three gaps are not failures; they are honest forward-looking
slots that belong to Round-13 / Round-14 sweep plans, not to the
paper-ablation text-only integration workflow.

---

## 1. Sections updated

### 1.1 `paper/sections/04_evaluation.tex` — §4.6 ONLY

- **Phase 2 edit (single insertion, +70 lines):** inserted a new
  `\paragraph{PBResult dataclass + 26-check integration note
  (WF-PB-Dock-Mode-Wire, 2026-09-14, ...)}` between the existing
  1-pocket PB block and the existing 30-cell PB statistic
  sub-section. The paragraph:
  - Names the artefact (`molmetal/reports/wf_pb_dock_mode.md`).
  - Breaks down the 26-check count by `pb_mode` (mol=14, dock=26).
  - Captures the **honest correction** that the original task
    brief said "22 total" but the actual protein-aware extra
    count is 12 (cofactor family is 8 checks), not 8.
  - Verifies the 26-check figure on CCO + CCN against
    `molmetal/data/mmp13_real/830c.pdb` (`pb_check.n_checks=26`).
  - Tags the `dock` mode as **wired but not exercised at pocket
    scale on de novo generated ligands** (open follow-up).
  - Cross-links back to §4.1 PoseBusters `pass_all` gate 2.

- **§4.6 PB 1-pocket block (lines 1527–1616) — UNCHANGED**
  (already correctly framed in WF-PB-Pass-Real-Dock, 2026-09-14).
- **§4.6 PB 30-cell block (lines 1618–1802) — UNCHANGED**
  (already correctly framed in WF-PB-Pass-10x3-Smoke, 2026-09-14).
- **§4.1–§4.5 — UNCHANGED** (Workflow 4 owns).

### 1.2 `paper/sections/05_ablation.tex` — 3 new sub-sections

- **§5.9 — Click-rule effect sizes (NEW, +~83 lines):**
  - Anchor: `molmetal/scripts/click_rule_effect_size_study.py`
    (380 LOC, 6/6 tests pass, 0.17 s).
  - 5×4 Cohen's-d panel: `PROJECTED` (from prior Lambda
    mini-pilots, NOT measured by this script).
  - Module + 6/6 tests: `MEASURED`.
  - Lit anchors (Himo 2005, Worrell 1984, Kolb 2001, Suzuki
    2011, Bickerton 2012, Cohen 1988): `CITEDONLY`.
  - 4 honest caveats captured inline (singleton-collapse,
    no production run, metal-compliance $d \equiv 0$, alias
    coverage).

- **§5.10 — Metal coordination probe (NEW, +~95 lines):**
  - Anchor: `molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py`
    (~340 LOC, 13/13 tests pass, 1.33 s).
  - Module + 13/13 tests + 4 reference probes (cisplatin /
    Pt_IV / Ru_III / dot-separated Pt): `MEASURED`.
  - 15-metal CN/OS coverage table: `CITEDONLY` (Lippard &
    Berg 1995, Reedijk 1987, Miessler 2014, Shriver & Atkins
    2010).
  - Per-pocket `compliance_rate` column: `DESIGN` (probe
    not wired into `r4_lambda_only_run.py`).
  - Expected `compliance_rate` distribution narrative:
    `DESIGN` (forward-looking expectation).
  - New columns for §5.8 panel (`coord_compliance_rate`,
    `geometry_distribution`): `PROJECTED`.
  - 4 honest caveats captured inline (graph-theoretic-not-3D,
    dot-separated SMILES NON-COMPLIANT flagging, OS inference
    heuristic, organic-only denominator exclusion).

- **§5.11 — Drug-likeness / ADMET metrics_v2 (NEW, +~115 lines):**
  - Anchor: `molmetal/molmetal_lam/sbdd_env/metrics_v2.py`
    (~480 LOC, 20/20 tests pass, 1.36 s).
  - Module + 20/20 tests: `MEASURED`.
  - 8-metric formula catalog (logp7_4, gi50_proxy,
    cell_permeability_logPapp, herg_cardio_risk, ames_mutagen,
    hepatotox_index, aqueous_solubility_logS,
    plasma_protein_binding): `CITEDONLY` (Hou 2007, Veith
    2009, Delaney 2004 ESOL, Obach 1999, Hughes 2008,
    Benigni-Richard 2005, Patrick 2009, Mente 2015).
  - 6-SMILES reference smoke (CCO / benzene / pPDA / aspirin /
    caffeine / cisplatin) + 3 spot-checks: `MEASURED`
    (reproducible from `all_metrics_one(smi)`).
  - Per-pocket batch-mean column: `DESIGN` (metrics not wired
    into `r4_lambda_only_run.py`).
  - 4-sub-panel ablation-axis partition: `PROJECTED`.
  - 4 honest caveats captured inline (heuristic-not-wet-lab,
    ranking-not-predictivity, every wet-lab claim is
    `SEARCHONLY`/`PROJECTED`/`DESIGN`, 6-SMILES reference is
    sanity floor).

- **§5.1–§5.8 — UNCHANGED** (§5.7 metal-pilot uplift + §5.8 P0
  anticancer panel left intact per workflow brief;
  Phase-1 inventory confirms no contradiction with new content).

### 1.3 Sections NOT updated (per workflow brief)

- `paper/sections/04_evaluation.tex` §4.1–§4.5 — Workflow 4 owns.
- `paper/main.tex` — Workflow 1 owns.
- `paper/refs.bib` — Workflow 1 owns.
- `paper/CROSS_REFS.md` — Workflow 1 owns (Phase 3 records the
  forward-link update as out-of-scope follow-up).

---

## 2. MEASURED cells promoted (cumulative across Phase 1-4)

| Cell | Workflow | Honest framing | Source |
|---|---|---|---|
| §4.6 PB 1-pocket smoke `(test_000, seed=42)` → `pb_pass_rate=1.000` (`pb_mode=mol`, `n=1`) | WF-PB-Pass-Real-Dock (2026-09-14) — NOT this workflow's edit | chemistry-only; no protein-aware clash | `molmetal/reports/wf_pb_pass_real_dock/final.md` |
| §5.8 P0 anticancer panel (single cell, 9 metrics on 15 candidates) | WF-Section05-P0-Metrics (2026-09-14) — NOT this workflow's edit | single-cell / single-seed; honest zeros for metal-aware cells | `molmetal/reports/wf_p0_metrics_smoke/{summary.md,report.json}` |

**Cells promoted by THIS workflow (paper-ablation, Phase 1-4):**

| Cell | Phase | Honest framing | Source |
|---|---|---|---|
| §5.9 module + 6/6 tests pass | Phase 3 | verified on disk, no production run required | `molmetal/scripts/click_rule_effect_size_study.py` |
| §5.10 module + 13/13 tests pass + 4 reference probes | Phase 3 | verified on disk, no production run required | `molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py` |
| §5.11 module + 20/20 tests pass + 6-SMILES reference smoke + 3 spot-checks | Phase 3 | verified on disk, no production run required | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` |
| §4.6 PBResult + 26-check integration note (NEW paragraph) | Phase 2 | "wired, not exercised at pocket scale" — paragraph describes protocol extension, NOT a measured PB pass rate | `molmetal/reports/wf_pb_dock_mode.md` |

**Cells promoted `DESIGN → MEASURED` (i.e., cells where a real
measurement at cell-cluster level was added): 0.**

The Phase 2 paragraph does not carry a `MEASURED` PB value; it
documents a protocol extension. The Phase 3 §5.9 / §5.10 / §5.11
additions promote the **modules + tests** to `MEASURED` (they
are verified on disk and can be exercised by a future batch run)
but the per-pocket batch columns stay `DESIGN` (no batch run).

**Net `DESIGN → MEASURED` cell promotion count for this
workflow: 0.**

**Net `MEASURED` cell ADDITION count for this workflow: 4
(§5.9 module, §5.10 module + 4 probes, §5.11 module + 6-SMILES
reference + 3 spot-checks).**

**Net §4.6 + §5 cumulative `MEASURED` cell count: 4 (this
workflow) + 10 (§4.6 PB 1-pocket + §5.8 P0 anticancer panel
from predecessor workflows) = 14 cells.**

**Net §4.6 + §5 cumulative `DESIGN` cell count: ≥ 30 §4.6 PB
column cells (search-bound, lift pending) + 3 §5 batch columns
(per-pocket compliance_rate, batch means, click-rule effect
sizes) = ≥ 33 cells remaining `DESIGN` or `PROJECTED`.**

---

## 3. Honest caveats (aggregated)

The 4-class honest-framing taxonomy established in Phase 1:

| Class | Definition | Cells in this workflow |
|---|---|---|
| REAL measure | Real generated-candidate measurement on this box, exercised end-to-end | §4.6 PB 1-pocket smoke (`pb_pass_rate=1.000` on `n=1`); §4.6 PB 30-cell panel (`wall=4.81 s/cell`); §5.8 P0 anticancer; §5.11 6-SMILES reference |
| DESIGN / PROJECTED | Module/code/test path is SHIPPED and TESTED, but NO production batch run | §5.9 5×4 Cohen's-d panel; §5.10 per-pocket `compliance_rate`; §5.11 per-pocket batch means |
| SEARCH-BOUND null | Search-budget artefact, NOT a metric defect | §4.6 30-cell PB panel (`pb_pass_rate=None × 30`) |
| CITEDONLY | Formula or table sourced from prior literature | §5.10 15-metal CN/OS table; §5.11 8 metric formulas; §5.9 lit anchors |

**Aggregate honest caveats (12 items):**

1. **Search-bound, not PB-bound.** The 30 × None PB result at
   `n_simulations=100` with the strict
   `synthesis_oracle=smarts ∧ symbolic_prior=True` gate is an
   honest null. PoseBusters has nothing to evaluate.
2. **Protein-blindness in the 1-pocket smoke.** The
   `pb_pass_rate=1.000` on `n=1` was measured at `pb_mode=mol`
   (chemistry + geometry, no protein-aware clash check).
3. **No SOTA comparison at scale.** TargetDiff's 94% PB pass
   rate is cited but explicitly "not applicable at this
   budget" (existing §4.6 text line 1693).
4. **`dock` mode wired but not exercised at pocket scale.** The
   `validate_docked(smiles, receptor_pdb)` method is wired and
   11/11 tests pass, but no de novo generated ligands have been
   run through it.
5. **Singleton-collapse regime in click-rule smoke.** `$n_{\text{distinct}}{=}1$`
   at smoke budget kills the pooled-variance denominator.
6. **No production run for click-rule effect sizes.** 5×3 panel
   at `n_sim=1000` is not in this artefact; estimated wall-clock
   ~40 s CPU.
7. **Metal-compliance `$d \equiv 0$` is a design choice, not a
   measurement.** Click rules are metal-agnostic in λ-only path.
8. **Probe per-pocket column is `DESIGN`.** Probe is shipped
   but NOT yet wired into `r4_lambda_only_run.py`.
9. **Graph-theoretic-not-3D limitation.** Spatial validation
   lives in `MGP` (§3.3, torch-based).
10. **Multi-component dot-separated SMILES NON-COMPLIANT
    flagging** (RDKit limitation, captured honestly).
11. **OS inference heuristic.** For `[Pt]` (no charge) falls
    back to per-element default (Pt → +2); bracket-tagged forms
    take precedence.
12. **Heuristic-not-wet-lab-calibrated framing for metrics_v2.**
    Coefficients taken at face value from literature without
    regression against any held-out assay.

---

## 4. Recommendation for next steps

### 4.1 Recommendation 1 — Workflow 1 owns the next pdflatex pass

The Phase 2 + Phase 3 edits are additive and forward-only; they
should compile cleanly under the existing
`pdflatex + bibtex + pdflatex + pdflatex` pipeline established in
`WF-Paper-Compile-Fix` (2026-09-14, tasks #465, #470–474, #509;
56 pages / 5.05 MB / 4 figures all rendering at pages 37-40).
**Workflow 1 (paper compile) owns the next pass; this workflow
does not request a recompile.**

### 4.2 Recommendation 2 — Workflow 1 owns the CROSS_REFS update

`paper/CROSS_REFS.md` should be updated by Workflow 1 (or a
follow-up Phase 4 pass) to add forward-links:
- §3.2 click-chem → §5.9
- §3.3 MetalGeometryPrior → §5.10
- §3.1 formalism → §5.11

This is **out of scope** for the paper-ablation workflow per the
workflow brief.

### 4.3 Recommendation 3 — Round-13 sweep is the next PB-population event

The §4.6 PB column for the 100×3 production sweep remains
`DESIGN` until `n_simulations=1000` re-runs land. Owned by:
- `WF-Lift-N-Sim-Cap` (tasks #539–541) for the hard-cap edit
- Round-13 100×3 sweep (`TODO/pending/14`, task #357) for the
  actual run

Estimated cost: 100 pockets × 3 seeds × ~5 s/cell = ~25 min wall,
or ~24 min if `n_simulations=1000` doubles the budget.

### 4.4 Recommendation 4 — Round-14 lit-grounded plan owns §5 batch runs

`TODO/pending/25_round14_plan.md` (WF-Vina-Lift-Phase23, task
#629) already documents the lit-grounded round-14 plan. The §5
batch-run follow-ups belong there:
- §5.9 click-rule effect sizes 5×3 at `n_sim=1000` (~40 s wall).
- §5.10 wire `metal_coord_probe.probe_batch()` into
  `r4_lambda_only_run.py` (Phase-4 integrator pass; not in scope
  for paper-ablation).
- §5.11 wire `metrics_v2.all_metrics_mean()` into
  `r4_lambda_only_run.py` (same constraint).

### 4.5 Recommendation 5 — Honest posture for any new claim

If a Round-13 / Round-14 batch run produces a new `MEASURED`
value that could populate any §4.6 or §5 cell, the cell must be
updated with:
- Verbatim value, sample size, and source artefact path.
- Forward-link to the artefact's `MEASURED` JSON.
- No silent promotion; the integration must include a
  Phase 4 update to `integrate.md` and `final.md`.

If the new value does NOT support the existing `DESIGN` /
`PROJECTED` claim, the cell must be **downgraded** honestly (the
honest posture the project has consistently enforced).

### 4.6 Recommendation 6 — No retroactive edits

The Phase 1-3 reports are the load-bearing sources of truth; any
attempt to retroactively rewrite `phase1_data_inventory.md`,
`phase2_section46.md`, or `phase3_section5.md` to inflate cell
promotion counts would silently break the honest-framing
contract. **Do not edit those files.** All future updates should
land in a separate `phase4_<topic>.md` file that explicitly
references the predecessor artefacts.

---

## 5. Single-paragraph verdict recap

> The paper-ablation workflow (Phase 1 → Phase 2 → Phase 3 →
> Phase 4) achieved **PARTIAL SUCCESS** on 2026-09-15: it added
> + ~364 lines across `paper/sections/04_evaluation.tex` §4.6
> (single PBResult + 26-check integration paragraph) and
> `paper/sections/05_ablation.tex` (§5.9 click-rule effect sizes,
> §5.10 metal coordination probe, §5.11 drug-likeness / ADMET
> metrics_v2). §4.1–§4.5, §5.1–§5.8, `paper/main.tex`,
> `paper/refs.bib`, and `paper/CROSS_REFS.md` are all untouched
> per the workflow brief. Cells promoted `DESIGN → MEASURED`:
> **0**. `MEASURED` cells added: **4** (§5.9 module, §5.10
> module + 4 reference probes, §5.11 module + 6-SMILES reference
> + 3 spot-checks). All cells use the strict honest-framing
> taxonomy `MEASURED` / `DESIGN` / `PROJECTED` / `CITEDONLY` /
> `SEARCH-BOUND` per the Phase-1 inventory contract. No `DESIGN`
> cell was silently promoted; no SOTA comparison was fabricated;
> no protein-aware PB claim was made without the `pb_mode=dock`
> exercise; no wet-lab pIC50 calibration is implied from
> heuristic metrics_v2 formulas. The paper-compile recommendation
> is to ship as-is and let Workflow 1 own the next
> `pdflatex + bibtex + pdflatex + pdflatex` pass.

---

## 6. Files in this workflow's deliverable set

| Path | Status | Purpose |
|---|---|---|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase1_data_inventory.md` | created Phase 1 | Data inventory (7 artefacts read, 4-class taxonomy defined) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase2_section46.md` | created Phase 2 | §4.6 PB integration report (350 lines) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase3_section5.md` | created Phase 3 | §5 ablation integration report (~400 lines) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/integrate.md` | created Phase 4 | Aggregator (this verdict's sibling) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/final.md` | created Phase 4 | This verdict |
| `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex` | modified Phase 2 (+70 lines, §4.6 only) | Single `\paragraph{...}` insertion |
| `/home/hugo/codes/try_triton_on_rocm/paper/sections/05_ablation.tex` | modified Phase 3 (+294 lines) | 3 new sub-sections (§5.9 + §5.10 + §5.11) |

---

## 7. Source artefacts (verbatim, absolute paths)

Phase 1-3 reports (load-bearing):

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase1_data_inventory.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase2_section46.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase3_section5.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/integrate.md`

§4.6 PB column source artefacts:

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/final.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/integrate.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/final.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/r4c.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_dock_mode.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`

§5 ablation source artefacts:

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_p0_metrics_smoke/summary.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_p0_metrics_smoke/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3b_metrics_v2.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3f_click_effect.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3g_metal_coord.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/metrics_v2.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/click_rule_effect_size_study.py`

Cross-cutting source artefacts (cited in §4.6 + §5):

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda_metal_integrate.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_section05_p0.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda_only_integrate.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_round12_sota_integrate.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_data_gap_analysis.md`

LaTeX files (modified / read):

- `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex` (modified Phase 2; §4.6 ONLY)
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/05_ablation.tex` (modified Phase 3; +3 sub-sections)
- `/home/hugo/codes/try_triton_on_rocm/paper/main.tex` (read but NOT modified — Workflow 1 owns)
- `/home/hugo/codes/try_triton_on_rocm/paper/refs.bib` (read but NOT modified — Workflow 1 owns)
- `/home/hugo/codes/try_triton_on_rocm/paper/CROSS_REFS.md` (read but NOT modified — Workflow 1 owns)

---

End of `final.md`. Workflow closed.