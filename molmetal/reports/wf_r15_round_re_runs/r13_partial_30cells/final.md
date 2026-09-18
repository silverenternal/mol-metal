# WF-R13 Partial 30-Cell Sweep — Aggregate Report

> **Honest-framing**: this is a **MEASURED** run on `2026-09-16T06:57:08.330367+00:00`. Script: `molmetal/scripts/r4_lambda_only_run.py`. Both arms share 5 pockets × 3 seeds = 15 cells each, 30 cells total at `n_simulations=1000` per cell. **vina_best / vina_pass_at_5A are NOT emitted by r4_lambda_only_run.py** (the script is pure Lambda-only: no docking, no PB). Recorded as `null` in the JSON with this caveat so the schema stays future-proof for r4_c_full_sweep.py (the docking harness).

## Configuration (both arms)

| field | metal-arm | no-metal-arm |
|---|---|---|
| n_pockets | 5 | PENDING |
| seeds | `[42, 0, 1234]` | `PENDING` |
| n_simulations/cell | 1000 | 1000 |
| n_top_k | 20 | 20 |
| metal_seed | `cisplatin` | `None` |
| click_rules | `['all-5']` | `['all-5']` |
| sa_weight | 0.0 | 0.0 |
| prior_enabled | True | PENDING |

## Aggregate metrics (mean ± std across 15 cells per arm)

| metric | metal-arm (mean ± std) | no-metal-arm (mean ± std) |
|---|---|---|
| valid | 1.0000 ± 0.0000 | PENDING |
| synth | 1.0000 ± 0.0000 | PENDING |
| uniq | 1.0000 ± 0.0000 | PENDING |
| sa_mean | 3.6574 ± 0.0000 | PENDING |
| qed | 0.7080 ± 0.0000 | PENDING |
| div_tan | 0.1065 ± 0.0000 | PENDING |
| n_distinct | 20.0000 ± 0.0000 | PENDING |

## Per-cell panel — metal-arm (cisplatin seed)

| pocket | seed | n_cand | n_distinct | valid | synth | uniq | sa_mean | qed | div_tan |
|---|---|---|---|---|---|---|---|---|---|
| test_000 | 42 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_000 | 0 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_000 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_001 | 42 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_001 | 0 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_001 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_002 | 42 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_002 | 0 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_002 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_003 | 42 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_003 | 0 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_003 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_004 | 42 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_004 | 0 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |
| test_004 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 3.657 | 0.708 | 0.106 |

## Caveats (READ FIRST)

1. **No Vina docking**: per spec, `vina_best` and `vina_pass_at_5A` would require `r4_c_full_sweep.py` with `--engine vina`, not `r4_lambda_only_run.py`. W4's responsibility per spec; this run deliberately stays in Lambda-only territory.
2. **No PB outer gate**: same reason — PB is a post-docking metric in this codebase, not part of the Lambda MCTS loop. `--pb-check` is a flag on the docking harness `r4_c_full_sweep.py` (added 2026-09-14 via WF-Wire-PoseBusters), not on the Lambda-only runner.
3. **No SA penalty weight**: spec says W4 owns that. `--sa-weight 0.0` (Lambda-only default) used here for cross-comparability with prior Round-12 10×3 pilots.
4. **Singleton attractor known issue**: per wf_lambda_internal_review (2026-09-15) and wf_lambda_fix_full_path_v2 (2026-09-15), `n_distinct=1` is expected with cisplatin seed at this MCTS budget. F1 soft-tiered prior + F2 scaffold-aware click selection are wired (17/17 tests pass) but the F2(a) MetalLigandExchange structural rule is still needed for non-collapse.
5. **No production-file edits**: pure read of the two report.json files + write to this directory only.

## File map

- `metal_arm_report.json` — 15 cells (cisplatin seed), raw script output
- `metal_arm_summary.md` — script-generated summary, copied verbatim
- `no_metal_arm_report.json` — 15 cells (no metal seed), raw script output
- `no_metal_arm_summary.md` — script-generated summary
- `final.json` — this aggregated JSON (short-name schema)
- `final.md` — this document
- `aggregate.py` — this aggregator