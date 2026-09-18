# Anticancer metric suite

- Status: local implementation complete; full paper evaluation pending
- Priority: high
- Effort: 1 day (1 codex task)
- Owner: codex
- Goal: build metal-anticancer adjusted metric suite (logP, TPSA, RotB, hERG proxy + composite_score).
- Scope: as specified in anticancer_vs_general_metrics_survey.md §6.
- Success criterion: pytest green, composite_score wired into reward_aggregator, TODO file created.
- Related: anticancer_vs_general_metrics_survey.md §6, metal_hydration.py, round-12 TODO/pending/13.

## Local completion update (2026-09-13)

The executable local portion is complete: `RewardAggregator` canonicalizes
object states, registers logP/TPSA/rotatable-bond/hERG channels, and exposes
`composite_score` through reward and metrics output. Regression coverage passes
with the proof-search prior tests. Full paper evaluation remains pending.
