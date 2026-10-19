# docs/ index

Generated 2026-10-14 by scanning the directory. 40 files.

**If a document contradicts what is on disk, the disk wins.** Several
entries below are marked SUPERSEDED because a later decision replaced
them; they are kept, not deleted, so a claim made at the time can still
be audited (AGENTS.md section 5.5).

## Where is the CFM line right now?

- `CFM_SIX_ROUND_CONCLUSION_2026_10_13.md` — Six rounds, what was learned, and the optimisation plan. START HERE for CFM.
- `CFM_STATUS_2026_10_12.md` — Running CFM status log, newest sections appended. Long but current.

## What are the project rules?

- `../AGENTS.md` — Agent entry point. Sections 1-10.
- `STANDING_DIRECTIVES.md` — Full text of sections 5.4-5.10.
- `WETLAB_DEPRECATION_FAQ.md` — Hard mandate: no wet-lab work.
- `ATOMIC_WRITE_PROTOCOL.md` — How to write without truncating.
- `00_MODEL_NARRATIVE.md` — PINNED: how the two models and the algorithm integrate.

## What may the paper say?

- `PAPER_SCOPE.md` — Paper A vs Paper B scope, shared-asset manifest, open contradictions.
- `PAPER_SECTION_3_METRIC_TABLE_FINAL.md` — Paper section 3 metric table (2026-10-01).
- `PAPER_SECTION_6_v4_FINAL.md` — Paper section 6 limitations (2026-10-06).
- `SUBMISSION_PLAN.md` — Submission plan (2026-10-12).
- `HONEST_NEGATIVE_CATALOG.md` — Every honest-negative, 1300+ lines. The source for any negative claim.

## What is the architecture?

- `ARCHITECTURE.md` — Architecture as of 2026-10-02. Check against code before trusting.
- `DESIGN_BOUNDARY.md` — Where the two lines meet.
- `WORKFLOW_DEPENDENCY_GRAPH.md` — Wave dependency graph.
- `POCKET_ENCODER_PIPELINE.md` — Pocket encoder architecture (Wf-A31/A32): PocketGen + P2Rank + platiny_pocket_pool + pocket_pool_unified + PerLayerPocketInjector + honest-framing table.

## What is the overall status?

- `STATUS.md` — Cross-line status (2026-10-11).
- `../README.md` — Top-level status (repo root).
- `ALIGNED_COMPARISON.md` — Same-protocol metric comparison across models.
- `GENERATION_OBJECTIVE_METRIC_DESIGN.md` — Why the generation metrics are designed as they are.

## What data do we train and test on?

- `DATA_INVENTORY.md` — Every training/eval dataset with measured row counts, role, and status (ACTIVE / SUPERSEDED / DO-NOT-TRAIN). Regenerate with `molmetal/scripts/wf_a50_data_inventory.py`; do not hand-edit.
- `DATA_LEAKAGE_AUDIT.md` — Measured train/val/test leakage audit for every active dataset (molecule-level, scaffold-level, cross-dataset). Regenerate with `molmetal/scripts/wf_a51_leakage_audit.py`; do not hand-edit.

## Superseded but retained

These are earlier drafts or intermediate decisions. A later document
supersedes them; read the later one first.

- `M2_R2_COHORT_DIAGNOSTIC.md` — M2_R2_FINAL_VERDICT.md is the decision that followed it.
- `M6_KAPPA_REDESIGN_PROPOSAL.md` — M6_KAPPA_DROP_DECISION.md is the outcome.
- `M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` — M6_KAPPA_FINAL_VERDICT.md is the verdict.
- `PAPER_S6_M6_KAPPA_REFRAME.md` — PAPER_SECTION_6_v4_FINAL.md supersedes it.
- `PAPER_S6_M2_R2_LIMITATIONS.md` — PAPER_SECTION_6_v4_FINAL.md supersedes it.
- `PAPER_SECTION_6_v4_LIMITATIONS_READY.md` — PAPER_SECTION_6_v4_FINAL.md is the final.
- `SECTION_3_METRIC_REFRAME_TEMPLATE.md` — PAPER_SECTION_3_METRIC_TABLE_FINAL.md is the filled result.
- `SECTION_6_LIMITATIONS_TEMPLATE.md` — PAPER_SECTION_6_v4_FINAL.md is the filled result.
- `USER_OK_FINAL_CHECKLIST.md` — USER_OK_OUTSTANDING_REPORT.md and PAPER_SCOPE.md are newer.
- `WF296_TO_WF359_RETROSPECTIVE.md` — WF_HISTORY.md carries the wave history forward.

## Not indexed above

Present in the directory but not routed to a question above. If one of
these should be canonical, say so and it gets an entry.

- `M2_R2_FINAL_VERDICT.md` (last modified 2026-09-29)
- `M6_KAPPA_DROP_DECISION.md` (last modified 2026-10-06)
- `M6_KAPPA_FINAL_VERDICT.md` (last modified 2026-10-03)
- `PREREG_TEMPLATE.md` (last modified 2026-10-03)
- `PROJECT_ARCHITECTURE_INSIGHTS_Q4.md` (last modified 2026-10-03)
- `RESOURCE_SCALING.md` (last modified 2026-10-03)
- `SCI_PAPER_SETUP.md` (last modified 2026-10-03)
- `SESSION_FINAL_SYNTHESIS.md` (last modified 2026-10-06)
- `USER_OK_OUTSTANDING_REPORT.md` (last modified 2026-10-03)
- `WF_HISTORY.md` (last modified 2026-10-06)

## Regenerating

Run `gen_docs_index.py`. Do not hand-edit: hand edits are how a
document index goes stale, which is the problem it exists to solve.
