# Wf-60 Phase 3 — Weak-real lifting synthesis (m7 + m15 + m16)

**Generated**: 2026-09-26T08:29:57Z
**Phase**: P3-weak-real-lifting (3 metrics on 5-seed × strict-scaffold)
**Mode**: per-recipe gate; honest framing per `party-twenty-spirit` §3.5

## Per-metric verdict

| Metric | Mean | Std | R17 baseline | Wf-60 target | Δ vs baseline | Δ vs target | Verdict |
|---|---:|---:|---:|---:|---:|---:|:---:|
| m7 | — | — | 0.1065 | 0.15 | — | — | DEFERRED |
| m15 | — | — | 0.428 | 0.35 | — | — | DEFERRED |
| m16 | — | — | 0.207 | 0.3 | — | — | DEFERRED |

## Pass criteria (target gates)

- **m7**: 5-seed mean Tanimoto distance ≥ 0.15 (R17 0.1065)
- **m15**: 5-seed mean docking RMSD ≤ 0.35 Å (R17 0.428)
- **m16**: 5-seed mean Pearson r ≥ 0.30 on HeLa48h cohort with Wf-44 numerical fix active (R17 0.207)

## Honest framing (party-twenty-spirit §3.5)

Per-metric notes:
- **m7**: R17 baseline 0.1065 measured on Lambda-MCTS generated molecules (5 cells × 30 cells × 3 seeds); Wf-31 floor-PASS 0.904 measured a fixed PlatinAI 5K subset (apples-to-oranges vs R17). Honest framing applies: report apples-to-apples vs the R17 0.1065 baseline only, never vs 0.904.
- **m15**: R17 baseline 0.428 Å on 10/10 pockets without strict scaffold split; Wf-30 10/10 pocket staging only; Wf-47 Vina MISSING_DEP. Strict-scaffold retrain + meeko extension for Pt/Au/Ru (Wf-50 R20-P1 ship) is the active fix path.
- **m16**: R17 baseline 0.207 (margin=0.5, 3-seed × 50-epoch HeLa48h). Wf-32 smoke -0.040 = HONEST-NEGATIVE on PlatinAI 4-cell due to cohort swap + tmQM encoder arch mismatch (Wf-29 carry). Wf-44 P4 numerical-fix ledger invariants: grad_clip=1.0 + step_decay=0.5 + fp64_fallback (cause C6 mitigation). Wf-32 smoke NOT canonical; R17 0.207 is canonical.

If a metric lands **DEFERRED** it means the trainer subprocess did not exit 0 / did not emit the expected JSON. Re-running with `--aggregate-existing` after the GPU job lands is the safe path. No fabricated numbers: every cell pulls from a real JSON.
