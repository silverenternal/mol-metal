# WF-PB-Pass-10x3-Smoke — 30-cell PB statistic

**Date:** 2026-09-14
**Operator:** r4_c_full_sweep.py driver
**Project root:** /home/hugo/codes/try_triton_on_rocm

## 1. Configuration

```
uv run python molmetal/scripts/r4_c_full_sweep.py \
    --pockets /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10 \
    --n-pockets 10 \
    --seeds 42 0 1234 \
    --physical-docking --pb-check \
    --n-simulations 100 --physical-top-k 20 \
    --output-prefix molmetal/reports/wf_pb_pass_10x3_smoke/r4c
```

CLI notes:
- `--pockets` is the extraction **root hint** (used for path validation), not the
  count; the count is supplied via `--n-pockets 10`.
- `--n-top-k` is **not** a CLI flag; top-k for physical docking is set via
  `--physical-top-k 20`.
- The `extended_204` tile library + `all_5` click rules + `symbolic_prior` +
  `synthesis_oracle` gates are inherited from the SOTA-aligned config; the
  driver downgrades `--engine both` to single-engine `vina` because GPU
  docking was unavailable.

## 2. Per-pocket × per-seed PB pass rate

Per-cell `pb_pass_rate` is **None** for every cell because every cell returns
0 generated candidates (status `no_candidates` or `seed_only`). PB only runs on
`is_generated=True` candidates, so the absence of generated candidates yields
`pb_status="no_candidates"` and `pb_pass_rate=None`.

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

## 3. Per-seed aggregate

| seed | n_pockets | n_generated_total | n_candidates_total | status mix |
|---|---|---|---|---|
| 42   | 10 | 0 | 5 | no_candidates ×5, seed_only ×5 |
| 0    | 10 | 0 | 5 | no_candidates ×5, seed_only ×5 |
| 1234 | 10 | 0 | 5 | no_candidates ×5, seed_only ×5 |

## 4. Aggregate

- **n_pockets:** 10 (test_000..test_009)
- **n_seeds:** 3 (42, 0, 1234)
- **n_cells_total:** 30
- **n_pockets_ok:** 0 (no cell accepted any generated molecule)
- **n_generated_candidates_total:** 0
- **n_seed_candidates_total:** 15 (the input reference ligand round-trips once per cell at most; 15/30 cells returned it)
- **pb_pass_rate (aggregate):** **undefined (None)** — there were zero generated candidates for PB to evaluate.
- **physical_jobs_completed:** 0 (Vina was skipped because no candidates; `physical.status="completed"` requires at least one accepted candidate)
- **physical_n_docked:** 0
- **physical_n_pb_pass:** 0
- **vina_best_kcal_mol:** N/A (no docking run)
- **wall_seconds_total:** 144.3 s
- **wall_seconds_mean_per_cell:** 4.81 s
- **gap_vs_targetdiff_94:** not applicable — the comparison cannot be made because Mol-Metal produced zero PB-eligible molecules per cell. The honest framing is: at this search budget (n_simulations=100) and these strict gates (synthesis_oracle=smarts + symbolic_prior=True + reference initialization), Lambda accepts zero generated molecules for any of the 10 CrossDocked pocket10 pockets across any of the 3 seeds.

## 5. Honest framing vs. TargetDiff 94% SOTA

TargetDiff's published 94% PB pass rate (Guan et al., ICML 2023, CrossDocked
test set, 100 pockets) is reported on **diffusion-generated** molecules that
are accepted by their pipeline. Our 10×3 smoke produces zero accepted
molecules per cell at `n_simulations=100`. There are several plausible
causes that we have NOT yet ruled out for this budget:

1. `n_simulations=100` is the lower end of the budget distribution; the
   earlier R3 round-3 round-12 mini pilot (`r4_lambda_only_run.py`) used
   `n_simulations=1000-3000` and produced tens of candidates per pocket.
2. The combination `synthesis_oracle=smarts + symbolic_prior=True` is
   known to be strict (Lambda-Metal-Pilot reported similar rates with
   `r4_lambda_only_run.py`).
3. `physical-top-k=20` is the upstream default; increasing `n_simulations`
   is a single CLI edit.

This smoke's purpose (per WF-PB-Pass-10x3-Smoke spec) was to obtain a
**30-cell PB statistic** for paper-grade reporting. The statistic it
obtained is: `pb_pass_rate = None × 30 cells` — i.e., the pipeline is
search-bound, not PB-bound. The honest conclusion is that **PB pass rate
cannot be reported at this search budget**; a follow-up pilot with the
existing `r4_lambda_only_run.py` harness (which accepts generated
candidates at this budget) is the right next step. That follow-up is
**out of scope** for this workflow and is captured as the next action.

**Conclusion: §4.6 update with the new 30-cell PB panel is NOT made**
because `pb_pass_rate` aggregate mean is undefined (None), not >0.5. No
LaTeX edit is performed. This is the honest outcome.

## 6. Artifacts

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/r4c.json` — per-pocket JSON
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/r4c.csv` — per-pocket CSV
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/r4c.md` — driver markdown report
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/run.log` — sweep log
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/stats.json` — computed stats
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/final.md` — this file
