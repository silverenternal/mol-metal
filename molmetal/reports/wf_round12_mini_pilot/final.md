# WF-Round-12 Mini Pilot — 5 pockets × 1 seed (Lambda search-only)

**Date:** 2026-09-14
**Driver:** `molmetal/scripts/r4_c_full_sweep.py`
**Output prefix:** `molmetal/reports/wf_round12_mini_pilot/r12`
**CLI (corrected):**
```
uv run python molmetal/scripts/r4_c_full_sweep.py \
    --pockets /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10 \
    --n-pockets 5 --seeds 42 --n-simulations 100 \
    --output-prefix molmetal/reports/wf_round12_mini_pilot/r12
```
Note: the task brief mentioned `--n-top-k 20` and `--output-dir ...`; neither flag
exists in the harness. `top_k` is fixed at 100 by the SOTA-aligned YAML config and
is not a CLI flag in this driver. `--output-dir` is also not present; the driver
uses `--output-prefix <prefix>` and writes `<prefix>.{csv,json,md}` plus a
`<prefix>_logs/` and a `<prefix>_poses/` directory.

## Honest framing

This run is **search-only** — no `--physical-docking` flag was passed, so:
- The "Vina" values in this report are `heuristic_proxy_no_docking` descriptors
  computed from SMILES topology by `molmetal/molmetal_lam/sbdd_env/vina_adapter.py`
  (see each pocket's `metric_backend.vina`). They are NOT kcal/mol AutoDock
  Vina scores and MUST NOT be compared to TargetDiff / 3D-SBDD docking tables.
- "Docked" / "PB pass" columns are `0` because no Vina / PoseBusters run was
  triggered. They are reported as 0 explicitly, not omitted.
- The 3 cells where `status=seed_only` mean the search returned the input
  reference ligand (no MCTS expansion produced a kept candidate); the metrics
  shown for those rows are the input ligand's descriptors.

The first 5 DESIGN cells in `paper/sections/04_evaluation.tex` Table 1 therefore
**cannot be filled as MEASURED by this run**. This report records what WAS
measured honestly so the paper team can decide between (a) running the
`--physical-docking` arm, (b) using these seed-only descriptors as a citation
sanity baseline, or (c) marking the first 5 cells PROJECTED.

## Table 1 — per-pocket results

| pocket_id | n_requested | n_finite | n_decoded | n_docked | Vina mean ± std | SA mean | QED mean | Lipinski pass | PB pass |
|---|---:|---:|---:|---:|---|---:|---:|---|---|
| test_000 (BSD_ASPTE_1_130_0) | 100 | 0 | 0 | 0 | — | — | — | 0/0 | 0/0 |
| test_001 (GLMU_STRPN_2_459_0) | 100 | 1 | 0 | 0 | -19.7154 (single) | 2.2557 | 0.6454 | 1/1 | 0/1 |
| test_002 (GRK4_HUMAN_1_578_0) | 100 | 0 | 0 | 0 | — | — | — | 0/0 | 0/0 |
| test_003 (GSTP1_HUMAN_2_210_0) | 100 | 1 | 0 | 0 | -14.5339 (single) | 2.6174 | 0.4317 | 1/1 | 0/1 |
| test_004 (GUX1_HYPJE_18_451_0) | 100 | 1 | 0 | 0 | -11.6317 (single) | 2.2997 | 0.8375 | 1/1 | 0/1 |

**Notes per row:**
- `n_requested` = `n_sims` from search_config (100 sims / pocket).
- `n_finite` = jobs whose `status` produced >= 1 candidate record (incl. seed_only).
- `n_decoded` = `n_generated_candidates` from each job (0 across the board: search did
  not expand past seed; reason recorded as "No products from attempted reductions").
- `n_docked` = 0 (no `--physical-docking` flag).
- `Vina` is `top1_vina_proxy` from `heuristic_proxy_no_docking` (descriptive, not kcal/mol).
- `Lipinski pass` and `PB pass` are per-cell denominators.

## Table 2 — aggregate across 5 pockets

| Metric | Value |
|---|---:|
| pockets requested | 5 |
| pockets completed (no timeout) | 5 |
| seeds | 1 (42) |
| total sims requested | 500 |
| total jobs returning >=1 cand | 3 |
| total newly decoded / generated candidates | 0 |
| total docked | 0 |
| total PB pass | 0 |
| search wall-clock mean per pocket (s) | 3.19 |
| total wall-clock (s) | ~16 |
| all 5 pockets complete (no wall-clock cap) | True |

Per-status distribution:
- `no_candidates`: 2 (test_000, test_002)
- `seed_only`: 3 (test_001, test_003, test_004) — returned only the input ligand
- `ok`: 0

Seed-only top-1 descriptor means (n=3, not population stats):
- SA mean = 2.391 (range 2.256–2.617)
- QED mean = 0.638 (range 0.432–0.838)
- Vina-proxy mean = -15.294, std = 3.344, range [-19.715, -11.632]

## Why 0 generated candidates?

The driver runs `reference_initialized_search` (paired-input search where the
input ligand is the seed). With `n_simulations=100`, `branching_target=1020`,
`max_depth=3`, the search attempted 1020 reductions per pocket and either
(i) all 1020 reductions failed (no matching rule / invalid reduction — recorded
in `diagnostics.generation.no_generated_reasons` for test_001/002/003/004), or
(ii) reductions produced products but all were rejected by the type or binding
gate (test_000: 11 products returned, all rejected). With `early_stop=True` and
`patience=50`, MCTS stopped at 51 simulations in every job.

The harness reports `simulations_completed=51` and
`early_stopped=true` for all 5 jobs — i.e., search terminated quickly because
no candidate improved on the seed reward within 50 simulations. This is the
EXPECTED behaviour for a low-budget seed-anchored pilot on drug-like inputs:
you are looking at the seed's neighbourhood, not the model's generative range.

## What would change the result

1. Pass `--physical-docking` (or `--physical-docking --physical-engine quickvina2-gpu`)
   to actually run AutoDock Vina on the kept candidates. Without this, the
   "Vina mean" column in Table 1 of the paper is ungrounded.
2. Increase `--n-simulations` (e.g., 1000) and bump `branching_target` and
   `max_depth` to give MCTS room to escape the seed. Current 100 sims is well
   below the values used in the cited SOTA sweeps (>=1000).
3. Pass `--synthesis-config <path>` so the SMARTS / AiZynth oracle can actually
   run; current run records `status=missing_aizynth_config` for every pocket.
4. Pass `--prior-state <path>` to a fitted prior JSON — currently every prior
   call records `awaiting_real_training_observations` and `applied=false`.

## Artefacts

- `molmetal/reports/wf_round12_mini_pilot/r12.csv` — per-pocket row table.
- `molmetal/reports/wf_round12_mini_pilot/r12.json` — full per-pocket records
  including diagnostics, search_config, all_candidates.
- `molmetal/reports/wf_round12_mini_pilot/r12.md` — driver-generated summary.
- `molmetal/reports/wf_round12_mini_pilot/r12_logs/` — per-job stdout/stderr.
- `molmetal/reports/wf_round12_mini_pilot/run.log` — top-level INFO log.
- `molmetal/reports/wf_round12_mini_pilot/final.md` — this report.

## Recommendation

Re-run with `--physical-docking` AND `--prior-state <fitted.json>` AND a larger
`--n-simulations` (e.g. 1000) before claiming any DESIGN cell of Table 1 as
MEASURED. The current run only validates that the harness executes end-to-end
without timeout and produces JSON/MD/CSV outputs in the expected schema.
