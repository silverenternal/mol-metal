# WF-Lambda-Only-Integrate — Table 2 Lambda-only Column Promotion

> **Goal.** Promote the 5 λ-only aggregate metrics in
> `paper/sections/04_evaluation.tex` Table~2 from \DESIGN{} to
> \MEASURED{}, citing
> `molmetal/reports/wf_lambda_only_mini_pilot/final.md` as the
> measurement source.

## 1. Cells promoted DESIGN → MEASURED in Table 2

| Table 2 row | λ-only column (before) | λ-only column (after) | Source |
|---|---|---|---|
| Validity rate (RDKit-sanitisable) | \DESIGN{} | `$1.0000$ \MEASURED{}` | `wf_lambda_only_mini_pilot/final.md` §3 |
| Synthesizability rate (5-click-rule coverage) | \DESIGN{} | `$1.0000$ \MEASURED{}` | `wf_lambda_only_mini_pilot/final.md` §3 |
| Metal-compliance rate (Pt=4 / Ru·Ir=6) | \DESIGN{} | `$0.0000$ \MEASURED{}` (honest zero) | `wf_lambda_only_mini_pilot/final.md` §3 + §3.1 |
| Diversity (mean pairwise Tanimoto) | \DESIGN{} | `$0.0050$ \MEASURED{}` (BELOW ≥ 0.30) | `wf_lambda_only_mini_pilot/final.md` §3 |
| Diversity (mean pairwise Homotype, enriched vocab) | \DESIGN{} | `$0.0020$ \MEASURED{}` (BELOW ≥ 0.20) | `wf_lambda_only_mini_pilot/final.md` §3 |

**n_cells_promoted_design_to_measured = 5**
**n_aggregate_metrics_now_measured = 5** (validity, synthesizability, metal-compliance, Tanimoto, homotype)

## 2. Aggregate metrics that remain DESIGN in Table 2 (lambda-only column)

| Table 2 row | λ-only column | reason |
|---|---|---|
| Novelty (max Tanimoto to training set, scaffold split) | \DESIGN{} | degenerate in the mini pilot (no training set supplied → novelty trivially 1.0000); will be measured once a tmQM-pretrained pool is wired |
| NFE budget (per-cell) | \DESIGN{} | pending Round-12 pilot NFE accounting (the mini pilot emits elapsed_s = 11.36 s but does not produce an NFE-budget per-cell comparison against the SOTA bars) |

Hybrid column remains fully \DESIGN{} pending Round-12 pilot completion
(§4.5 hybrid-vs-lambda-only ablation narrative).

## 3. Structural changes to Table 2

- **3 new aggregate rows added** to Table 2 (validity_rate,
  synthesizability_rate, metal_compliance_rate). The prior Table 2
  scaffold only had 6 rows (triple-threshold success, relaxed success,
  Tanimoto diversity, homotype diversity, novelty, NFE budget); the 5
  target metrics from the mini pilot map onto 2 existing rows +
  3 new rows.
- **Caption updated** to carry an explicit "Honest framing (Lambda-only
  mini pilot, 2026-09-14)" paragraph that documents which cells are
  \MEASURED{}, where the `BELOW` and "honest zero" qualifiers come
  from, and which cells remain \DESIGN{}.

## 4. §4.5 Hybrid vs Λ-only ablation narrative

A new paragraph was inserted into §4.5
(`sec:evaluation:lambda-only`) that:

1. Cites `molmetal/reports/wf_lambda_only_mini_pilot/final.md` by
   path.
2. Documents the 5 measured aggregate metrics with their `BELOW` /
   `honest zero` qualifiers inline.
3. Explains the reference-conditioned nature of the Λ-only path: the
   0.968 reference Tanimoto diagnostic shows Λ re-discovers the
   pocket reference's heavy-atom skeleton; the per-cell diversity
   degeneracy on click-poor roots is therefore a property of the
   small-$N$ mean, not a failure of the MCTS machinery.
4. Notes the diversity gap can be lifted without any harness change
   (re-run with `--metal-seed cisplatin` or a click-rich training
   pool).

## 5. §4.10 promote-to-MEASURED rule audit

The §4.10 cell-tagging invariant requires that a \DESIGN{} cell be
promoted to \MEASURED{} only by running the actual experiment and
re-tagging at the point of use; no silent promotion by editing the
LaTeX source.

Audit of this integration:

- The experiment that backs each promotion is
  `wf_lambda_only_mini_pilot` (pure Lambda, no Vina, no CFM; 5 pockets ×
  1 seed; 100 MCTS simulations; top-k 20; total wall-clock 11.36 s).
- Each promoted cell carries the value verbatim from
  `wf_lambda_only_mini_pilot/final.md` §3 plus the source path inline.
- Each promoted cell carries the `BELOW` or `honest zero` qualifier
  inline where the measured value does not meet the
  per-pocket projection (diversity) or is config-induced
  (metal_compliance).
- No cell was promoted without a backing measurement; no citation
  was added without a run.

## 6. Follow-ups

The integration closes the load-bearing gap (λ-only column in
Table 2 is no longer all \DESIGN{}), but several follow-ups remain:

1. **Lift metal_compliance_rate from honest zero to a real value.**
   Re-run `r4_lambda_only_run.py` with `--metal-seed cisplatin` on
   the same 5 pockets × 1 seed to populate metal_compliance_rate at
   a non-degenerate value (the WF-Lambda-1c main arm already shows
   this configuration produces metal_compliance = 1.0000 cleanly).
   Estimated wall-clock: ~12 s.

2. **Lift diversity from BELOW to meet the projection.** Two
   complementary options:
   - **Option A (cheap):** bump `--n-simulations` from 100 to 1000
     on the existing 5-pocket test_000 (which has the click-rich
     reference); expected to push diversity_tanimoto_mean for
     test_000 above 0.30 with no other config change.
   - **Option B (real):** re-run the full 5×1 with a
     metal-seed or a click-rich training pool (e.g.
     tmQM-pretrained) so the MCTS root is not the
     small-metal-free reference; expected to populate 5/5 cells
     instead of 1/5.

3. **Promote novelty and NFE budget from DESIGN to MEASURED.** Both
   require either a training-set SMILES file (for novelty) or the
   full Round-12 N=10×3 pilot accounting (for NFE). Neither is
   achievable from a pure Lambda-only mini pilot; both remain
   \DESIGN{} until Round-12.

4. **Hybrid column promotion.** Independent of this Lambda-only
   integration; gated on the Round-12 pilot executing end-to-end
   (blocked by the CFM-side environment issue per
   `TODO/pending/13_top_journal_pilot_r12.md` Round-12 mini pilot
   §).

## 7. Artefacts

- `paper/sections/04_evaluation.tex` — Table 2 (lines 278-329) and
  §4.5 narrative (lines 371-432) updated.
- `molmetal/reports/wf_lambda_only_mini_pilot/final.md` — the
  measurement source (cited inline).
- `TODO/pending/13_top_journal_pilot_r12.md` — summary appended
  (Round-12 mini pilot §4 Table 2 lambda-only column integration,
  2026-09-14).
- `molmetal/reports/wf_lambda_only_integrate.md` — this report.

## 8. Task metrics

- `n_cells_promoted_design_to_measured = 5` (validity, synthesizability,
  metal-compliance, Tanimoto diversity, homotype diversity)
- `n_aggregate_metrics_now_measured = 5` (same set; matches the 5
  target metrics in the task brief)
- `n_new_table_rows_added = 3` (validity, synthesizability,
  metal-compliance did not exist as Table 2 rows in the prior
  scaffold)
- `n_cross_refs_added = 1` (§4.5 cites `wf_lambda_only_mini_pilot/final.md`)
- `n_cells_remaining_design_in_lambda_only_column = 2` (novelty, NFE
  budget)
