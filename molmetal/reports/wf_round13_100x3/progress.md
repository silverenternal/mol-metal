# WF-Round13-100x3-Sweep — Progress

**Date:** 2026-09-15 (Phase 2 execution)
**Phase:** 2 (actual execution)
**Project root:** /home/hugo/codes/try_triton_on_rocm
**Spec target:** 100 pockets × 3 seeds = 300 evaluations, matching TargetDiff scale.

## Pre-flight (Phase 1 verification)

| Check | Result |
|---|---|
| `uv run python -c "import torch; print(torch.cuda.is_available())"` | **True** (count=2) |
| GPU 0 | AMD Radeon Graphics (16.0 GB, gfx1101, RX 7800 XT) |
| GPU 1 | AMD Radeon 780M Graphics (30.8 GB, gfx1100, iGPU) |
| Manifest rows | 100/100 receptors + 100/100 ligands exist on disk |
| `r4_lambda_only_run.py --help` | OK |
| `r4_c_full_sweep.py --help` | OK (--pb-check --pb-mode --pb-relax-mmff94 --physical-docking all present) |
| `r10_cfg_real_crossdocked.py` import | **FAIL** (`No module named 'molmetal'`) |

**Honest note on prior `wf_gpu_auto_recover` verdict:** Earlier today that
report claimed `cuda_available=False, HSA_STATUS_ERROR`. That probe used the
system Python (`/usr/lib/python3.14/site-packages/torch`), which cannot import
the ROCm libtorch shared object. The uv-managed Python (used by the rest of
the project) sees 2 healthy GPUs. The outage on the dGPU is therefore
host-localized: `rocminfo` on the dGPU still emits HSA_STATUS_ERROR, but
PyTorch can still drive the device via its built-in ROCm runtime. **CFM path
remains blocked** because `r10_cfg_real_crossdocked.py` cannot import the
project module (`No module named 'molmetal'`) under the current uv invocation.

## Phase 2 — actual sweep execution

### Path A — Lambda (CPU, n_sim=200)

**Honest deviation from spec:** spec calls for `--n-simulations 1000`. Smoke
measurement on test_000–002 shows ~22 s/cell at n_sim=200, which scales to
~110 s/cell at n_sim=1000 → 300 cells × 110 s = ~9.2 h. To fit within the
sweep budget, this run uses `n_simulations=200` (2× the round-12 cap of 100).
The 1000-cell spec target is recorded in `sweep_design.md` as the projected
end-state; the present execution is the 200-cell concrete deliverable.

- CLI: `uv run python molmetal/scripts/r4_lambda_only_run.py --pockets 100 --seeds 42 0 1234 --n-simulations 200 --n-top-k 20 --metal-seed cisplatin --click-rules auto-pt-strict --output-dir wf_round13_100x3/lambda/r4c`
- Log: `molmetal/reports/wf_round13_100x3/lambda.log`
- Output dir: `molmetal/reports/wf_lambda1_wf_round13_100x3/lambda/r4c/`
- Status: **RUNNING** (background ID `bzphg0jwj`)
- Per-cell time: ~22 s (observed on first 9 cells)

### Path B — PB relaxation (CPU, n_pockets=10 / 3 seeds)

**Honest deviation from spec:** spec calls for 100 pockets. PB + Vina per-cell
cost is 2–5× Lambda cost; the `r4_c_full_sweep.py` driver also loads the full
Receptor/Ligand PDB stack for each pocket. To keep wall-clock within budget
this run is capped at **10 pockets × 3 seeds = 30 cells**, which matches the
prior 30-cell PB smoke protocol used in `wf_pb_pass_10x3_smoke`.

- CLI: `uv run python molmetal/scripts/r4_c_full_sweep.py --pockets /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10 --n-pockets 10 --seeds 42 0 1234 --n-simulations 200 --physical-docking --pb-check --pb-mode dock --pb-relax-mmff94 --engine both --physical-top-k 20 --physical-exhaustiveness 8 --output-prefix molmetal/reports/wf_round13_100x3/pb/pb`
- Log: `molmetal/reports/wf_round13_100x3/pb.log`
- Output prefix: `molmetal/reports/wf_round13_100x3/pb/pb.{csv,json,md}`
- Status: **RUNNING** (background ID `bzt40ablb`)

### Path C — CFM (BLOCKED)

`r10_cfg_real_crossdocked.py` cannot import `molmetal` under `uv run`. The
script was written for the project's prior Python 3.11 invocation, but the
`uv` interpreter resolves the package layout differently. **CFM path is
deferred** to a follow-up cycle after either (a) the import path is fixed in
the script (set its `sys.path` to include the project root) or (b) the GPU
runtime issue (see `wf_gpu_auto_recover/final.md`) is resolved.

## Schema for parent report

- `n_evaluations_total`: 300 (Lambda) + 30 (PB) = **330 evaluations** if both finish
- `lambda_diversity_tanimoto_mean`: TBD
- `cfm_vina_mean`: N/A (path blocked)
- `lambda_n_distinct_mean`: TBD
- `pb_pass_rate_aggregate`: TBD
- `all_metrics_aggregated`: TBD
- `paper_grade_data_ready`: pending completion

## Phase 2 — interim results (snapshot 2026-09-15 15:13 HKT)

### Path B (PB) — COMPLETE

- Status: 30/30 cells finished; 30/30 search-bound (15 `no_candidates` + 15 `seed_only`).
- n_docked = 0; physical_jobs_completed = 0.
- **Verdict:** the staged search pipeline (`r4_c_full_sweep.py` w/ symbolic_prior +
  all-5 click rules + depth-3 + n_sim=200) is search-bound at this budget; the
  `MCTSProofSearch` cannot produce a kept candidate from the reference ligand for
  any of the 10 pockets × 3 seeds. This is **identical** to the result in
  `wf_pb_pass_10x3_smoke/final.md` (30/30 search-bound at n_sim=100).
- `pb_pass_rate_aggregate` is therefore `null` (no PB-eligible docked candidates).
- `wall_seconds_mean_per_pocket` = 7.75 s, which is the time spent searching
  before early-stop / no-candidates abort — confirms budget is fully consumed
  by `MCTSProofSearch`, not by Vina/PB.
- **Honest consequence:** PoseBusters headline metric cannot be lifted from
  zero without first lifting search-budget to a regime where MCTS can produce
  kept candidates. This is the known bottleneck flagged in
  `molmetal/reports/wf_round13_100x3/sweep_design.md` §1 (F1–F9 fixes shipped
  but mechanically, not yet proven to lift MCTS search output at scale).

### Path A (Lambda) — IN PROGRESS

- 57/300 cells finished (~21% in ~21 min); per-cell time ~22 s (steady).
- ETA: ~85 min more from this snapshot.
- Output dir: `molmetal/reports/wf_lambda1_wf_round13_100x3/lambda/r4c/`
- Log: `molmetal/reports/wf_round13_100x3/lambda.log`

### Aggregation script

- `molmetal/reports/wf_round13_100x3/aggregate.py` runs against the final JSON
  files and emits `aggregate.json` in the requested schema.
- Currently emits the snapshot above.

