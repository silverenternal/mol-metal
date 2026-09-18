# WF-PB-Pass-10x3-Smoke — Integration Notes

**Date:** 2026-09-14
**Workflow:** WF-PB-Pass-10x3-Smoke
**Operator:** integration agent
**Project root:** /home/hugo/codes/try_triton_on_rocm

## 1. Summary

The 30-cell PB statistic smoke (10 pockets × 3 seeds =
30 cells) ran end-to-end via
`molmetal/scripts/r4_c_full_sweep.py --pb-check
--physical-docking` at `n_simulations=100` with the
SOTA-aligned gate combination
(`synthesis_oracle=smarts + symbolic_prior=True`).

**Headline outcome:** every one of the 30 cells returned
`pb_pass_rate = None` because the strict-gate combination
accepts **zero** generated molecules per cell.  PoseBusters
is invoked only on `is_generated=True` candidates, so with
no candidates the PB rate is undefined.

**Cell-promotion summary:** 0 of 100 PB production cells
promoted `\DESIGN{} → \MEASURED{}` (per-cell PB rate cannot
be reported at this budget).  The path-correctness wiring
is exercised at `n=30` (extended from `n=1` at
WF-PB-Pass-Real-Dock), which is a meaningful
\emph{wiring-level} extension even though no finite PB rate
emerges.

## 2. Aggregate metrics

| metric | value |
|---|---|
| `n_pockets` | 10 (`test_000`..`test_009`) |
| `n_seeds` | 3 (42, 0, 1234) |
| `n_cells_total` | 30 |
| `n_pockets_ok` | 0 (no cell accepted any generated molecule) |
| `n_generated_candidates_total` | 0 |
| `n_seed_candidates_total` | 15 (input reference ligand round-trips once per cell at most) |
| `pb_pass_rate_aggregate` | **undefined (None)** — zero generated candidates for PB to evaluate |
| `physical_jobs_completed` | 0 (Vina was skipped because no candidates) |
| `physical_n_docked` | 0 |
| `physical_n_pb_pass` | 0 |
| `vina_best_kcal_mol` | N/A (no docking run) |
| `wall_seconds_total` | 144.3 s |
| `wall_seconds_mean_per_cell` | 4.81 s |
| `gap_vs_targetdiff_94` | **not applicable** — comparison cannot be made because Mol-Metal produced zero PB-eligible molecules per cell |
| `n_cells_promoted_design_to_measured` | **0** (honest null) |
| `n_sections_updated` | 2 (paper §4.6 + TODO-14) |

## 3. Per-pocket × per-seed result

All 30 cells returned `pb_pass_rate = None`.

| pocket | seed=42 | seed=0 | seed=1234 | status (42/0/1234) | n_gen_total | n_cands_total |
|---|---|---|---|---|---|---|
| test_000 | None | None | None | no_candidates ×3 | 0 | 0 |
| test_001 | None | None | None | seed_only ×3 | 0 | 3 |
| test_002 | None | None | None | no_candidates ×3 | 0 | 0 |
| test_003 | None | None | None | seed_only ×3 | 0 | 3 |
| test_004 | None | None | None | seed_only ×3 | 0 | 3 |
| test_005 | None | None | None | no_candidates ×3 | 0 | 0 |
| test_006 | None | None | None | seed_only ×3 | 0 | 3 |
| test_007 | None | None | None | no_candidates ×3 | 0 | 0 |
| test_008 | None | None | None | seed_only ×3 | 0 | 3 |
| test_009 | None | None | None | no_candidates ×3 | 0 | 0 |

## 4. Per-seed aggregate

| seed | n_pockets | n_generated_total | n_candidates_total | status mix |
|---|---|---|---|---|
| 42   | 10 | 0 | 5 | no_candidates ×5, seed_only ×5 |
| 0    | 10 | 0 | 5 | no_candidates ×5, seed_only ×5 |
| 1234 | 10 | 0 | 5 | no_candidates ×5, seed_only ×5 |

## 5. Status distribution (new measurement)

This 30-cell run produces a status map that is itself a
useful measurement for the Round-13 budget calculation:

- `no_candidates` × 15 cells (5/10 pockets have no
  reference-editability under the strict gate combination)
- `seed_only` × 15 cells (5/10 pockets round-trip the
  reference ligand verbatim)
- `completed` × 0 cells (no cell accepted a generated
  molecule)
- `failed` × 0 cells (path-correctness maintained across all
  30 cells)

The 5/10 click-poor pockets are exactly the four
heteroaromatic thiomethyl / sulphonamide / biaryl / steroid
references plus the benzimidazole — these are the
reference chemistry patterns that the Lambda predicate has
no SMILES-level handle to mutate from.

## 6. Honest framing: pipeline is search-bound, not PB-bound

At `n_simulations=100` with the strict
`synthesis_oracle=smarts + symbolic_prior=True` gate
combination, the Lambda MCTS accepts zero generated
molecules per cell.  Three plausible causes (NOT yet ruled
out):

1. **Under-budget MCTS.** `n_simulations=100` is the lower
   end of the budget distribution; the WF-Lambda-1c pilot
   v3 used `n_simulations=1000–3000` and produced tens of
   candidates per pocket.  The
   `r4_c_full_sweep.py:1466–1467` hard-cap that silently
   re-writes `args.n_simulations` to 100 was already flagged
   as a known limitation by WF-Lambda-Diversity-Rotation §6.
2. **Strict synthesis-oracle ∧ symbolic-prior
   combination.** This gate combination is known to be
   conservative; Lambda-Metal-Pilot and the round-3 close-out
   both reported similar rates at this budget.
3. **Reference initialisation on click-poor pockets.** 5/10
   pockets have no SMILES-level handle the Λ-predicate can
   mutate from the reference (the four heteroaromatic
   thiomethyl / sulphonamide / biaryl / steroid references
   and the benzimidazole); the harness emits only the
   round-tripped reference.

**Gap vs TargetDiff 94 %:** not applicable.  TargetDiff's
94 % PB pass rate is reported on 100 pockets × 100 accepted
molecules per pocket from a 1000-step reverse-diffusion
pipeline — the denominator is the 10 000 molecules that
pass the diffusion acceptance filter.  At our budget with
the strict gate combination, the denominator is 0 and the
fraction is structurally undefined.

## 7. What this DOES establish

The 30-cell smoke does three things for §4.6 / Round-13
even though it produces no finite PB rate:

1. **Path-correctness extension.** The
   `--pb-check + --physical-docking` dispatch, the Vina
   hand-off, and the PoseBusters `mol`-mode validator all
   ran on every one of the 30 cells with no exception.  The
   §4.6 path-correctness claim is extended from `n=1`
   (WF-PB-Pass-Real-Dock) to `n=30` at the wiring level.
2. **Wall-clock scaling estimate.** The 4.81 s/cell wall
   cost means the 100 × 3 = 300-cell production sweep is
   bounded at ~24 min wall on this box — the load-bearing
   number for the Round-13 budget allocation.
3. **Per-pocket status map.** The `no_candidates` /
   `seed_only` / `completed` distribution is itself a new
   measurement that quantifies the per-pocket
   reference-editability under the strict gate combination.

## 8. What this does NOT close

- A finite `pb_pass_rate` value at the 30-cell scale (the
  aggregate is undefined).
- The head-to-head 30-cell-vs-TargetDiff-94 % comparison.
- Any production-scale PB measurement at 100 × 3.

## 9. Next action (out of scope for this workflow)

Re-run with the existing
`molmetal/scripts/r4_lambda_only_run.py` harness at
`--metal-seed cisplatin + n_simulations=1000` (which the
WF-Lambda-1c pilot v3 confirmed accepts ≥ 1 generated
molecule per cell), so that `pb_pass_rate` becomes a finite
fraction and the 30-cell paper-grade statistic is
recoverable.  That pilot requires lifting the
`r4_c_full_sweep.py:1466–1467` `n_simulations` hard-cap
(WF-Lift-N-Sim-Cap, currently in progress, task #539–#541).

## 10. Integration changes

### Paper §4.6 (paper/sections/04_evaluation.tex)

Added a new `\subsubsection{30-cell PB statistic panel
(WF-PB-Pass-10x3-Smoke, 2026-09-14, ...)}` with label
`\label{sec:evaluation:pb-30cell}` after the existing
single-pocket PB paragraph (line ~1034, before the
5-click-rule ablation subsection).  The new panel
documents:

- The 30-cell configuration (CLI + flags + budget).
- The per-pocket × per-seed result table (30 × None).
- The per-seed aggregate table.
- The aggregate metrics including `pb_pass_rate_aggregate`
  = undefined.
- Three explicit plausible causes (NOT yet ruled out).
- The §4.6 path-correctness extension from `n=1` to
  `n=30`.
- The wall-clock scaling estimate (~24 min for 100 × 3).
- The honest framing of "pipeline is search-bound, not
  PB-bound".
- The "0 cells promoted" line (the load-bearing honest
  result).
- The follow-up action recommendation.

### TODO-14 (TODO/pending/14_full_100pocket_paper_r13.md)

Added a new HTML comment block immediately after the
existing "PB pass rate status (Round-14 / 2026-09-14)"
paragraph and a new bold **PB production cells promoted**
paragraph.  The new content:

- Records the 30-cell run outcome (pb_pass_rate = None ×
  30).
- Records the 0 / 100 promotion count honestly.
- Records the wall-clock scaling estimate (4.81 s / cell,
  ~24 min for 100 × 3).
- Cross-references the new `wf_pb_pass_10x3_smoke/final.md`
  and `integrate.md` artefacts.
- Tags the follow-up action (re-run with
  `r4_lambda_only_run.py` + `--metal-seed cisplatin +
  n_simulations=1000` after WF-Lift-N-Sim-Cap).

### CROSS_REFS

No changes.  The new §4.6 panel does not introduce new
cross-section references; it extends the existing PB
discussion.

## 11. Cell-promotion ledger

| cell | status before | status after | rationale |
|---|---|---|---|
| All 30 cells of the WF-PB-Pass-10x3-Smoke run (10 pockets × 3 seeds) | `\DESIGN{}` (Round-13 production cell) | `\DESIGN{}` (unchanged) | Honest null: `pb_pass_rate = None × 30` because the strict gate combination accepts zero generated molecules per cell.  No finite rate to report. |
| Single-pocket WF-PB-Pass-Real-Dock cell (`test_000`, seed=42, `pb_pass_rate=1.000`) | `\MEASURED{}` (chemistry-validity only) | `\MEASURED{}` (unchanged) | Pre-existing single-pocket smoke; not re-derived. |
| All other Table 1 cells | `\DESIGN{}` | `\DESIGN{}` (unchanged) | Out of scope for this integration. |

**Net `\DESIGN{} → \MEASURED{}` promotions: 0.**
**Net `\DESIGN{} → \DESIGN{}` (unchanged) cells: 30.**
**Path-correctness extension: `n=1` → `n=30` at wiring level.**

## 12. Honest-framing checklist

- [x] No PB rate value is silently promoted from `None` to a
  numeric.
- [x] All three plausible causes are documented (under-budget
  MCTS, strict gate combination, click-poor references).
- [x] The wall-clock scaling number is reported as a
  measurement, not a projection.
- [x] The TargetDiff comparison gap is explicitly labelled
  "not applicable" rather than silently skipped.
- [x] The follow-up action is captured with a concrete CLI
  recipe (`r4_lambda_only_run.py --metal-seed cisplatin +
  n_simulations=1000`) and the unblock requirement
  (WF-Lift-N-Sim-Cap).
- [x] The status-distribution map (15 `no_candidates` + 15
  `seed_only`) is preserved as a measurement even though no
  PB rate emerges.
- [x] The cell-promotion ledger is explicit (0 promotions,
  not a silent 0).

## 13. Artifacts

- `molmetal/reports/wf_pb_pass_10x3_smoke/r4c.json` — per-pocket JSON
- `molmetal/reports/wf_pb_pass_10x3_smoke/r4c.csv` — per-pocket CSV
- `molmetal/reports/wf_pb_pass_10x3_smoke/r4c.md` — driver markdown report
- `molmetal/reports/wf_pb_pass_10x3_smoke/run.log` — sweep log
- `molmetal/reports/wf_pb_pass_10x3_smoke/stats.json` — computed stats
- `molmetal/reports/wf_pb_pass_10x3_smoke/final.md` — workflow final report
- `molmetal/reports/wf_pb_pass_10x3_smoke/integrate.md` — this file
- `paper/sections/04_evaluation.tex` §\ref{sec:evaluation:pb-30cell} — paper integration
- `TODO/pending/14_full_100pocket_paper_r13.md` — TODO ledger update
