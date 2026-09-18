# Metrics Registry — Single Source of Truth for Measured Numbers

**Last updated:** 2026-09-15 (smoke cleanup appended)
**Owner:** metrics-registry

This folder centralizes all measured numerical results from `molmetal/reports/wf_*/final.md` and the various one-off pilot JSON files. **Every paper table cell, every ablation axis, every honest-negative claim must trace back to a JSON in this folder.**

> **Smoke section cleaned up — 2026-09-15.** All `*smoke*` JSON / reports / CSV / MD artefacts under `molmetal/reports/` that were NOT load-bearing for paper §3 / §4 / §5 were deleted (33 items: 16 dirs + 17 files, ~3.37 MB). The 5 KEEP paths are documented in `molmetal/reports/wf_remove_smoke/phase3g_reports_done.md`. Remaining `*smoke*` paths under `molmetal/reports/`:
> - `wf_p0_metrics_smoke/` — load-bearing for paper §5.8 P0 MEASURED panel
> - `wf_pb_pass_10x3_smoke/` — load-bearing for paper §4 PB panel (30-cell)
> - `wf_cfm_path_b_decoder_rework/smoke/` — load-bearing for §3.3 + §4.6 bond-decoder MEASURED
> - `wf_lambda_rule_symmetry_smoke/` — integration report smoke baseline (kept for paper trail)
> - `reinvent4_learned_smoke/` — TODO-05 on-host REAL execution artefact
>
> Final report: `molmetal/reports/wf_remove_smoke/final.md`. Pytest result: 1596 passed, 7 skipped, 1 xpassed, 10 failed (8 CFM-GPU + 2 metal_coord_probe; pre-existing, NOT smoke-related).

## Why this exists

Before this folder (2026-09-14 and earlier):
- 38 `wf_*/final.md` reports scattered across `molmetal/reports/`
- 40+ ad-hoc `*.json` and `*.csv` files in `molmetal/reports/` (baseline_*.json, *_ablation*.csv, *_pair*.csv, *_standard12.csv, etc.)
- No central registry of which metric was measured where, on what input distribution, with what status
- Task tracker and memory sometimes recorded different status than the wf final.md (e.g., "ship" vs. "synthetic-only lift")

After this folder:
- One JSON per run in `by_round/` — full config + aggregate + verdict
- One JSON per metric in `by_metric/` — every measurement of that metric across all runs + aggregate verdict
- One JSON per paper section in `by_paper_section/` — promotion status from DESIGN→MEASURED
- One JSON per SOTA paper in `sota_comparisons/` — headline metrics + Mol-Metal comparison
- `INDEX.md` at this folder root = single source of truth

## Status taxonomy (every JSON file uses one of these)

| Status | Meaning | Example |
|---|---|---|
| **MEASURED** | Actually ran the experiment, recorded the number | wf_round12_lambda_pilot 10×3 sa_mean=5.9452 |
| **DESIGN** | Claim in paper spec / plan, but no experiment yet | §4 Table 1 cells with no wf behind them |
| **PROJECTED** | Code change complete, lift claimed but on different/smaller input | Path B decode_ratio 192/192 on synthetic Pt-click clouds (not trained CFM) |
| **CITEDONLY** | SOTA number from another paper, not Mol-Metal | TargetDiff Vina -8.45 |
| **UNMEASURED** | No data yet — flag explicitly | CFM Vina_mean on trained-CFM coordinates |
| **NEGATIVE_RESULT_HONEST** | Experiment ran, lift was 0 / negative, framed honestly | Round-12 diversity_tanimoto=0.0 |
| **CONTROL_VERIFIED** | Experiment ran, demonstrated controlled behavior | metal_compliance=1.0 with seed vs 0.0 without |
| **PARTIAL_LIFT** | Some axes lifted, others didn't | SA lift 5.0 → 3.099 (TargetDiff target 2.65-2.86 not reached) |
| **FAILURE_PER_SPEC_GATE** | Experiment ran, result below gate, spec says "do not proceed" | CFM 5000-step decode_ratio=0 |

## Layout

```
metrics/
├── README.md                       ← this file
├── INDEX.md                        ← single source of truth (TODO: write)
├── by_round/                       ← one JSON per wf run
│   ├── r12_lambda_pilot.json
│   ├── r12_lambda_pathb.json
│   ├── r12_pilot_1c.json
│   ├── cfm_path_b_decoder_rework.json
│   ├── cfm_gpu_5000step_retrain.json
│   ├── cfm_vina_lift_phase23.json
│   ├── pb_pass_real_dock.json
│   ├── pb_mmff94_relax.json
│   ├── sa_fragment_pool_optimize.json
│   ├── sa_penalty_guidance.json
│   ├── pic50_margin_sweep.json
│   └── reinvent4_multiproperty.json
├── by_metric/                      ← one JSON per metric, all measurements aggregated
│   ├── decode_ratio.json
│   ├── diversity_tanimoto.json
│   ├── metal_compliance.json
│   ├── vina_kcal_per_mol.json
│   ├── pb_pass_rate.json
│   ├── sa_mean.json
│   ├── n_distinct.json
│   └── pearson_r_pic50.json
├── by_paper_section/               ← promotion status for paper cells
│   ├── section_4_2_main_results.json
│   ├── section_4_6_pb_panel.json
│   └── section_5_ablation.json
└── sota_comparisons/               ← cite-only SOTA numbers for paper §2 + §6
    ├── targetdiff_guan_2023.json
    ├── unimol_v2.json
    └── diffsbdd_pocket2mol.json
```

## How to use

**Before writing a paper cell:**
1. Open `metrics/by_paper_section/section_X.json`
2. Check the cell's `status` and `cells_design_to_measured` count
3. If `MEASURED`, find source JSON in `by_round/` and cite it
4. If `DESIGN`, mark as `DESIGN` in paper + cite TODO-14/22/23

**Before claiming a lift:**
1. Open `metrics/by_metric/<metric>.json`
2. Find every `status: MEASURED` measurement
3. Compute `best_measured` and `gap_to_lit_target`
4. If `status: PROJECTED`, label in paper as "design claim, projected on synthetic"

**Before deciding what to do next:**
1. Read all `by_metric/<metric>.json` `verdict` fields
2. Find metrics with `FAILURE_PER_SPEC_GATE` or `NEGATIVE_RESULT_HONEST`
3. These are the highest-ROI investigation targets

## Update protocol

When a new wf report ships → write `by_round/<wf_name>.json`.
When new data lands → update the corresponding `by_metric/<metric>.json`.
When paper cells promote DESIGN→MEASURED → update `by_paper_section/<section>.json`.
When a SOTA number is published/cited → add to `sota_comparisons/<paper>.json`.

**Do not write raw numbers anywhere else.** All measurements must be in this folder first.

## What's still missing (TODO)

- `by_round/` for wf_round13_100x3 (pending workflow `wo21fr1ug`)
- `by_round/` for wf_round12_lambda_patha_10x3 (pending workflow `wgvpkwmvb`)
- `by_metric/anticancer_index.json`
- `by_metric/validity_synth_novel.json`
- `INDEX.md` master cross-reference table
