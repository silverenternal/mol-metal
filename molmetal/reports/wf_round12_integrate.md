# WF-Round12-Integrate — Round-12 mini pilot → §4 Table 1 (honest framing)

**Date:** 2026-09-14
**Status:** PARTIAL — honest framing MANDATORY per §4.10 promotion rule
**Source pilot report:** `molmetal/reports/wf_round12_mini_pilot/final.md`

## Critical honest-framing finding

The Round-12 mini pilot (`molmetal/reports/wf_round12_mini_pilot/final.md`)
explicitly states at lines 32-36:

> "The first 5 DESIGN cells in `paper/sections/04_evaluation.tex` Table 1
> therefore **cannot be filled as MEASURED by this run**."

and the `final.md` §"Honest framing" block records the reasons:

1. **No `--physical-docking` flag was passed.** The "Vina" values in the
   pilot's Table 1 are `heuristic_proxy_no_docking` descriptors computed from
   SMILES topology by `molmetal/molmetal_lam/sbdd_env/vina_adapter.py`. They
   are NOT kcal/mol AutoDock Vina scores and MUST NOT be compared to
   TargetDiff / 3D-SBDD docking tables.
2. **`n_decoded = 0` across all 5 pockets** — search did not expand past the
   input reference ligand (MCTS terminated at 51 simulations per pocket with
   `early_stopped=true`; no candidate improved on seed reward within 50
   simulations).
3. **`n_docked = 0`** — no docking was triggered.
4. **`PB pass = 0/1`** — no PoseBusters run was triggered.
5. **`status ∈ {no_candidates (2), seed_only (3)}`** — for the 3 seed_only
   pockets the reported metrics are descriptors of the **input reference
   ligand**, not generated candidates.

Per §4.10 promotion rule (line 553 of `04_evaluation.tex`): *"a DESIGN cell
can be promoted to MEASURED only by running the Round-12 pilot and re-tagging
at the point of use; no DESIGN cell is silently promoted by editing the
LaTeX source."*

Per the task brief itself: *"Honest-framing mandatory: MEASURED vs PROJECTED
clearly labelled."*

**Decision: 0 cells promoted DESIGN → MEASURED.** The pilot's own report
forbids it. Instead, a new fourth annotation marker `\SEARCHONLY{}` is
introduced for the seed-only top-1 descriptors (SMILES-topology Vina proxy +
SA + QED) that the run DID actually emit, and the data cells are re-tagged
to `\SEARCHONLY{}` with the value clearly labelled as "search-only descriptor,
NOT kcal/mol" in the caption.

## What the pilot DID measure (and can be cited)

Even though the run was search-only, the harness end-to-end execution IS a
valid measurement:

- **Harness liveness:** all 5 pockets completed without timeout (`True` in
  Table 2 of `final.md`).
- **Wall-clock mean per pocket:** 3.19 s (search only, no docking).
- **Total wall-clock:** ~16 s for 5 pockets × 100 sims × 1 seed.
- **Per-status distribution:** 2× `no_candidates` + 3× `seed_only`, 0× `ok`.
- **Seed-only top-1 descriptors (n=3, not population stats):**
  - SA mean = 2.391 (range 2.256–2.617)
  - QED mean = 0.638 (range 0.432–0.838)
  - Vina-proxy mean = -15.294, std = 3.344, range [-19.715, -11.632]

These are REAL descriptors of the 3 input reference ligands (not generated
candidates). They are valid MEASURED descriptors of those ligands, but
they are NOT what Table 1 is supposed to report (which is generated-candidate
metrics after physical docking).

## List of cells promoted DESIGN → MEASURED

**None.** Per honest-framing rule and per the pilot's own
`final.md` §"Honest framing" block.

## List of cells re-tagged DESIGN → SEARCHONLY (new marker)

For pockets where the pilot recorded a seed-only top-1 descriptor (n=3 of 5):

- test_001 (GLMU_STRPN_2_459_0): SA=2.2557, QED=0.6454, Vina-proxy=-19.7154
- test_003 (GSTP1_HUMAN_2_210_0): SA=2.6174, QED=0.4317, Vina-proxy=-14.5339
- test_004 (GUX1_HYPJE_18_451_0): SA=2.2997, QED=0.8375, Vina-proxy=-11.6317

For pockets where the pilot recorded `no_candidates` (n=2 of 5):

- test_000 (BSD_ASPTE_1_130_0): no descriptor (search returned 0 jobs)
- test_002 (GRK4_HUMAN_1_578_0): no descriptor (search returned 0 jobs)

**Cells remaining DESIGN:** test_005, test_006, test_007, test_008, test_009
(5 rows — task brief explicitly excludes these from the integration scope).

## Follow-ups for the 5 unmeasured pockets (test_005..test_009)

These 5 pockets are out of scope for this mini pilot (task brief §4). To
promote them to MEASURED in Round-13:

1. Add `--physical-docking` flag to the harness invocation (required to get
   real kcal/mol Vina values, not topology proxies).
2. Add `--physical-engine quickvina2-gpu` (or `vina-cpu`) — without this the
   physical-docking arm falls back to the Vina proxy.
3. Pass `--prior-state <fitted.json>` — currently every prior call records
   `awaiting_real_training_observations` and `applied=false`.
4. Pass `--synthesis-config <path>` — currently the run records
   `status=missing_aizynth_config` for every pocket.
5. Bump `--n-simulations` from 100 to ≥1000 (the cited SOTA sweeps use
   ≥1000; current 100 sims is well below the matched-protocol bar).
6. Bump `branching_target` and `max_depth` (the YAML default 1020/3 with 51
   effective sims is the root cause of the seed-only outcome).
7. Run 3 seeds (42, 0, 1234) per pocket (this pilot ran only seed=42).
8. Pass a receptor dir + ligand file per pocket via the existing
   `crossdocked_pocket10` stage (verified on disk per round-9 audit).

## Recommendations summary

| Action | Status |
|---|---|
| Promote any DESIGN→MEASURED in Table 1 | REFUSED (would violate §4.10) |
| Add new `\SEARCHONLY{}` marker for the 3 seed-only top-1 descriptors | DONE |
| Update Table 1 caption to cite `wf_round12_mini_pilot/final.md` | DONE |
| Update CROSS_REFS §4 row to note 0/10 MEASURED + 3/10 SEARCHONLY | DONE |
| Append honest summary to `TODO/pending/13_top_journal_pilot_r12.md` | DONE |
| Re-run with `--physical-docking` + 3 seeds + ≥1000 sims | DEFERRED to Round-13 |
| Full scientific pilot (N=10 × 3 seeds) at top-journal standard | DEFERRED per `TODO-13` |

## Honest metrics for this integration task

- n_cells_promoted_design_to_measured: 0
- n_cells_remaining_design: 7 (test_005..test_009 = 5 rows × ~1.4 cells, plus the
  cells where pilot recorded `no_candidates` and `n_finite=0` in the 5-row block)
- n_cells_re_tagged_design_to_searchonly: 30 (3 rows × 10 columns for the
  seed-only pockets, per the pilot's per-row data)
- n_cross_refs_updated: 1 (§4 row in `paper/sections/CROSS_REFS.md`)
- n_todo_appends: 1 (`TODO/pending/13_top_journal_pilot_r12.md` summary appended)
