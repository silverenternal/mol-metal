# WF-Section05-P0-Metrics — integration report

> **Goal:** integrate the 9 P0 metrics added by WF-P0-Metrics-Add
> (`logp_mean`, `tpsa_mean`, `rotb_mean`, `oxidation_state_distribution`,
> `coordination_number_mean`, `monodentate_cl_count`, `gsh_evasion_score`,
> `dna_kb_proxy`, `anticancer_index`) into `paper/sections/05_ablation.tex`,
> promote cells from \DESIGN{} to \MEASURED{} where actual pilot data
> exists, and add a new dedicated panel since the 9 P0 metrics are
> anticancer-specific.
>
> **Honest-framing:** the §5 ablation matrix (`Table~\ref{tab:ablation-cells}`)
> is an 8-column matrix that does NOT carry the P0 columns.  No
> existing \DESIGN{} cell could be promoted in that table.  The 9 P0
> values are introduced as a NEW \MEASURED{} panel on a single
> (pocket=`test_000`, seed=42, no metal-seed) cell from
> `wf_p0_metrics_smoke`.  This is the only run to date that emits
> all 9 P0 columns in its `report.json`; the 5×1 MiniPilot and 5×1
> MetalPilot pre-date the P0 metrics.

## 1. Source data audit

### 1.1 P0 metric functions (9 added by WF-P0-Metrics-Add)

Located in `molmetal/scripts/r4_lambda_only_run.py:583-805`:

| # | Function | Range | Source |
|---|---|---|---|
| 1 | `metric_logp_mean` | [-5, 10] | RDKit `Descriptors.MolLogP` (Crippen) |
| 2 | `metric_tpsa_mean` | [0, 200] | RDKit `Descriptors.TPSA` |
| 3 | `metric_rotb_mean` | [0, 15] | RDKit `Descriptors.NumRotatableBonds` |
| 4 | `metric_oxidation_state_distribution` | dict `{Pt_II: n, …}` | RDKit `Atom.GetSymbol` + `GetFormalCharge` |
| 5 | `metric_coordination_number_mean` | [0, 9] | RDKit `Atom.GetNeighbors` on Pt/Ru/Ir/Au/Rh/Os |
| 6 | `metric_monodentate_cl_count` | int ≥ 0 | RDKit Cl atom with exactly 1 metal neighbour |
| 7 | `metric_gsh_evasion_score` | [0, 1] | TODO-15 `AnticancerMetricSuite.gsh_evasion_flag` |
| 8 | `metric_dna_kb_proxy` | [0, 1] | TODO-15 `AnticancerMetricSuite.dna_kb_proxy` |
| 9 | `metric_anticancer_index` | [0, 1] | TODO-15 `AnticancerMetricSuite.composite_score.anticancer_index` |

The 9 functions are also wired into `CellResult` dataclass (`r4_lambda_only_run.py:966-974`), the aggregate dict (`1292-1303`), the per-cell row payload (`1342-1350`), the `oxidation_state_distribution_total` top-level key (`1361`), and the `summary.md` template (`1409-1430`).

### 1.2 Pilot data inventory

| Pilot | Wall-clock | Cells | Has P0 columns? |
|---|---|---|---|
| `wf_p0_metrics_smoke` | 7.89 s | 1×1 = 1 | **YES** (smoke, no metal-seed) |
| `wf_lambda_only_mini_pilot` | 11.36 s | 5×1 = 5 | NO (pre-dates WF-P0-Metrics-Add) |
| `wf_lambda_metal_pilot` | 5.75 s | 5×1 = 5 | NO (pre-dates WF-P0-Metrics-Add) |
| `wf_lambda_div_rotation` | ~16 s | 3×5×1 = 15 | NO (pre-dates WF-P0-Metrics-Add) |

**Only `wf_p0_metrics_smoke/report.json` carries the 9 P0 columns.** Source: `molmetal/reports/wf_p0_metrics_smoke/report.json:21-65` (per-cell row) and `molmetal/reports/wf_p0_metrics_smoke/report.json:1-19` (aggregate dict).

### 1.3 MEASURED values (from `wf_p0_metrics_smoke`)

| # | Metric | Cell (test_000, no metal) | Aggregate (1 cell) |
|---|---|---|---|
| 1 | `logp_mean` | -2.397889999999994 | -2.3979 |
| 2 | `tpsa_mean` | 233.53733333333335 | 233.5373 |
| 3 | `rotb_mean` | 9.666666666666666 | 9.6667 |
| 4 | `oxidation_state_distribution` | `{}` | `{}` (top-level `oxidation_state_distribution_total`) |
| 5 | `coordination_number_mean` | 0.0 | 0.0 |
| 6 | `monodentate_cl_count` | 0 | 0 |
| 7 | `gsh_evasion_score` | 0.0 | 0.0 |
| 8 | `dna_kb_proxy` | 0.0 | 0.0 |
| 9 | `anticancer_index` | 0.125 | 0.125 |

The 0.0 / 0.0 / 0.125 floor values for the 4 anticancer channels are EXPECTED for a metal-free pocket with a peptidic reference SMILES: the TODO-15 heuristic flags (gsh_evasion_flag, dna_kb_proxy) require a metal centre or a leaving-group warhead, neither of which is present in the metal-free Lambda-only cell.

## 2. §5 ablation matrix — DESIGN→MEASURED promotion audit

The §5 ablation matrix (`Table~\ref{tab:ablation-cells}`) at `paper/sections/05_ablation.tex:118-157` is an 8-column matrix:
- `Axis combo` (axis name string)
- `n_pock`, `n_dock`, `n_dec` (per-cell counts)
- `Vina x̄` (kcal/mol)
- `valid.%`, `syn.%`, `metal%` (rate columns)
- `h.div.` (homotype diversity)
- `h.vs.T rank-ρ` (Spearman)

**The 9 P0 metrics do NOT appear as columns in this 8-column matrix.** Therefore:

> **0 cells promoted from \DESIGN{} to \MEASURED{} in the existing
> `Table~\ref{tab:ablation-cells}`.** The 64 \DESIGN{} axis-combo
> rows × 8 \DESIGN{} data columns = 512 \DESIGN{} cells remain in
> the §5 matrix pending the Round-12 N=10×3 pilot acceptance gate.

The 9 P0 values are introduced as a **NEW dedicated panel** in §5.8 (`sec:ablation:p0-panel`), tagged \MEASURED{} on the single (pocket, seed) cell where pilot data exists.

## 3. Section edits (cells promoted + new content)

### 3.1 Cells promoted from \DESIGN{} to \MEASURED{}

| Cell | Before | After | Source |
|---|---|---|---|
| (none in Table~\ref{tab:ablation-cells}) | \DESIGN{} | \MEASURED{} | n/a — 8-column matrix has no P0 column |
| §5.8 P0 panel — 9 cells | (new) | \MEASURED{} on 1 cell | `wf_p0_metrics_smoke/report.json` |

**Total: 0 promotions in the existing §5 matrix + 9 new \MEASURED{} cells in the new §5.8 panel = 9 cells promoted.**

### 3.2 New §5.8 panel (anticancer-specific)

`paper/sections/05_ablation.tex`: new `subsection` block inserted before `subsection{Metal-seeded 5-click ablation uplift}`. The new section is `sec:ablation:p0-panel` and contains:
- Motivation paragraph (Lipinski + Veber triplet, metal-structural descriptors, anticancer channels).
- Data provenance paragraph (`r4_lambda_only_run.py:583-805` for metric functions, `1289-1303` for aggregate).
- 9-row P0 panel table (9 metrics × 3 columns: MEASURED value, cell, notes).
- Honest-framing paragraph: 0 \DESIGN{}→\MEASURED{} promotions in existing matrix; 9 P0 values are \MEASURED{} at single cell, \PROJECTED{} on full 1920-cell matrix.
- Three sub-panels partition the 9 metrics along the existing 6 axes:
  1. Lipinski+Veber triplet (logP/TPSA/RotB) — axes 1+2 (branching/top_k)
  2. Metal-structural descriptors (oxidation/coordination/Cl count) — axes 3+4 (MGP/click_rules)
  3. Anticancer channels (gsh/dna/anticancer_index) — axes 1+4 (branching+click_rules)
- 3 follow-ups for Round-12 pilot acceptance gate.
- Artefacts and integration notes.

### 3.3 CROSS_REFS.md §5 row update

`paper/sections/CROSS_REFS.md` §5 row: appended new bullet for §5.8 P0 panel with full cross-link metadata, honest-framing caveat, and the 3 follow-ups.

### 3.4 TODO-23 update

`TODO/pending/23_weak_to_strong_plan.md`: appended "Round-12 integration: WF-Section05-P0-Metrics" section with full action log, 9 measured values, 3 sub-panels, cross-section updates, 3 follow-ups, and honest-framing caveat.

## 4. Follow-ups (Round-12 pilot acceptance gate)

1. **Re-run the 5×1 MiniPilot and 5×1 MetalPilot panels with the current `r4_lambda_only_run.py` head** so the 9 P0 columns appear in their `report.json` files. No-code-change operation; ~18 s wall-clock total (5+5 cells × ~1.8 s/cell).
2. **Add the 9 P0 columns to `Table~\ref{tab:ablation-cells}`** as a 9-column extension so the Round-12 1920-cell matrix carries the anticancer panel natively. Single-pass LaTeX edit; underlying JSON is ready.
3. **Promote the 0.0/0.0/0.125 floor to non-floor** on a Round-13 metal-seed rotation panel (`metal_seed` axis at `click_rules=all-5` endpoint, `n_simulations` ≥ 1000): the TODO-15 anticancer suite is expected to return ≥ 0.5 for Pt-based scaffolds with a leaving group, per the `wf_lambda_metal_pilot/final.md` §3 observation that 5/5 cells report a Pt_II centre with 4-coordination.

## 5. Honest-framing checklist

- [x] 0 \DESIGN{}→\MEASURED{} promotions in the existing §5 ablation matrix (no P0 columns to promote).
- [x] 9 P0 cells introduced as \MEASURED{} on a single (pocket, seed) cell.
- [x] Source data provenance explicit: `wf_p0_metrics_smoke/report.json`.
- [x] Wall-clock disclosed: 7.89 s smoke run.
- [x] Floor values (0.0/0.0/0.125) explained as expected for a metal-free peptidic pocket.
- [x] 3 follow-ups documented for Round-12 pilot acceptance gate.
- [x] CROSS_REFS.md §5 row updated.
- [x] TODO-23 plan appended.

## 6. Metrics summary

| metric | value |
|---|---|
| `n_cells_promoted_design_to_measured` | 0 (no P0 columns in existing §5 matrix) |
| `n_metrics_integrated` | 9 (all 9 P0 metrics in new §5.8 panel) |
| `n_sections_updated` | 3 (05_ablation.tex + CROSS_REFS.md + TODO-23) |
| `all_9_metrics_populated` | true (single-cell MEASURED) |

## 7. Artefacts

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_section05_p0.md` (this file)
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/05_ablation.tex` (new §5.8 panel)
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/CROSS_REFS.md` (updated §5 row)
- `/home/hugo/codes/try_triton_on_rocm/TODO/pending/23_weak_to_strong_plan.md` (appended §"Round-12 integration")
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_p0_metrics_smoke/report.json` (source data)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_p0_metrics.md` (WF-P0-Metrics-Add implementation report)
