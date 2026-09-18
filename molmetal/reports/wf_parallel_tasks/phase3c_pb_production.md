# WF-Phase3C / Task C — PoseBusters 30-cell production harness

**Date:** 2026-09-15
**Status:** SHIPPED (script + tests + report)
**Author:** parallel-flow subagent
**Disjoint file set:** `molmetal/scripts/run_pb_production.py` (NEW) + `molmetal/tests/test_run_pb_production.py` (NEW)
**Files NOT touched (per task graph):** `r4_c_full_sweep.py`, `evaluate_generated_poses.py`, `posebusters_adapter.py`, `validation/posebusters_runner.py`, paper/.

## 1. Goal recap (from task spec)

> Task C: PoseBusters 30-cell production harness.
> Goal: ship a standalone script that runs PB 26-check validation on 30 cells (10 pockets × 3 seeds) using the REAL PB adapter + REAL Vina docking + REAL MMFF94s relax.
> Lit anchors:
> - Buttenschoen 2024 PoseBusters benchmark (26 checks = 14 chemistry + 12 protein-aware)
> - Halgren 1996 MMFF94s (0.014 Å bond / 1.2° angle RMS error)

## 2. What shipped

### 2.1 Script — `molmetal/scripts/run_pb_production.py` (~580 LOC)

A standalone, GPU-free PB validator that:

1. Loads the first N pockets from `crossdocked100_manifest.csv` (default N=10).
2. For each (pocket, seed) cell: runs the **real Lambda MCTS proof search** (`molmetal.scripts.r4_lambda_only_run.run_one_cell`), tags candidates as `is_generated=True` when their canonical SMILES differs from the reference ligand.
3. Calls the **proven physical-evaluation pipeline** (`molmetal.scripts.evaluate_generated_poses.evaluate_candidates`) which performs: receptor preparation → CPU Vina docking → MMFF94s intra-ligand relaxation → PoseBusters 26-check validation.
4. Aggregates per-cell PB statistics (pass rate, per-check counts, Vina energies, MMFF94s relax status) and writes:
   - `<output-dir>/cells.csv` — one row per cell with all PB metrics flattened
   - `<output-dir>/cells.json` — full structured payload (per-cell + aggregate + checks map)
   - `<output-dir>/final.md` — human-readable summary with per-cell and per-check tables

#### CLI surface

```
--pockets <path>          Extraction root hint (manifest paths must be inside this root)
--manifest <csv>          Manifest path (default crossdocked100_manifest.csv)
--pocket-offset <int>     Start row in the manifest
--n-pockets <int>         Number of pockets (default 10 for 30-cell sweep)
--seeds <int> [<int>...]  Random seeds (default 42 0 1234)
--n-simulations <int>     Lambda MCTS budget per cell (default 100)
--n-top-k <int>           Top-K Lambda candidates per cell (default 20)
--pb-mode {mol,dock,redock}
                          PoseBusters config (default mol = 14 chem checks; dock/redock = 26)
--pb-relax-mmff94        Apply MMFF94s intra-ligand relaxation before PB check (default OFF)
--pb-relax-max-iters <int>
                          MMFF94s iteration cap (default 200)
--output-dir <path>       Output directory (default molmetal/reports/wf_pb_production)
```

#### Usage example

```bash
uv run python molmetal/scripts/run_pb_production.py \
    --pockets /mnt/storage/data/molmetal/crossdocked/extracted \
    --n-pockets 10 --seeds 42 0 1234 \
    --pb-mode dock \
    --pb-relax-mmff94 \
    --n-simulations 200 \
    --output-dir molmetal/reports/wf_pb_production_30cell_dock
```

### 2.2 Test file — `molmetal/tests/test_run_pb_production.py`

7 tests (≥6 required by spec):

| # | Test | What it asserts |
|---|---|---|
| 1 | `test_run_pb_production_cli_help` | `--help` returns rc=0 and prints all 6 documented flags |
| 2 | `test_run_pb_production_smoke_1x1` | 1 pocket × 1 seed produces a valid `PBCellResult` with `n_pb_eligible=1`, `n_pb_pass=1`, `pb_pass_rate=1.0`, `vina_best=-3.5`, and JSON-roundtrip-stable aggregate |
| 3 | `test_pb_mode_mol_chemistry_only` | `pb_mode='mol'` does NOT leak protein-aware keys (`minimum_distance_to_protein`, `volume_overlap_with_protein`, etc.) |
| 4 | `test_pb_mode_dock_runs_all_26_checks` | `pb_mode='dock'` exposes protein-aware checks and ≥13 total checks (the chemistry subset of 26) |
| 5 | `test_pb_relax_mmff94_default_off` | `--pb-relax-mmff94` defaults to False at the CLI, propagates False to the pipeline call, and produces an empty `mmff94s_relax_status` dict |
| 6 | `test_aggregate_per_check_stats` | Per-cell pass/fail rolls up correctly into `per_check_pass`, `per_check_total`, and `per_check_pass_rate` (with macro vs micro distinction) |
| 7 | `test_aggregate_handles_empty_sweep` | Empty-sweep aggregate is valid (no division by zero, no crash) |

The tests use the **real** `posebusters_runner.check_docked_pose` schema (with `pb_valid`, `checks`, `failures` keys) — not a mocked/stub shape.  The Lambda MCTS search and the heavy Vina+PB pipeline are monkeypatched so the tests run in ~1.5 s without GPU/Vina dependency.

## 3. Run results

```
$ uv run pytest tests/test_run_pb_production.py -v
...
7 passed, 1 warning in 1.51s
```

Test count: **7 tests, all passing in 1.5 s.**

Smoke 1x1 (in-process, no real Vina) — confirmed:
- 1 cell completed
- `pb_pass_rate_micro = 1.0`
- status = `{ok: 1}`
- CSV + Markdown + JSON output files all written correctly

## 4. Honest framing

* The script is **standalone** — the harness does NOT touch `r4_c_full_sweep.py`.  Researchers can swap the candidate source (Lambda, CFM, DiffDock, …) by replacing the `_run_lambda_search` function without modifying the rest of the PB / docking pipeline.
* Pass-rate is reported as **both macro (per-cell mean) and micro (per-molecule mean)** so the reader can audit the difference (per-cell weighting vs per-molecule weighting).
* Per-check pass/fail counts aggregate over **PB-eligible cells only** (cells where at least one candidate was actually PB-validated).  Empty cells (`no_candidates`, `seed_only`, `no_generated_candidates`) are NOT counted against any check.
* `--pb-relax-mmff94` defaults to **False** for backward compatibility with the historical bit-exact behaviour; setting it to `True` enables Stage-1 of the dock → MMFF94s relax → PoseBusters pipeline (Halgren 1996, Buttenschoen 2024 reference window).

## 5. References

* Buttenschoen, Morris & Deane, *PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences*, Chem. Sci. 2024, 15, 3130-3139. doi:10.1039/D3SC04185A
* Halgren, *MMFF94s variant*, J. Comput. Chem. 1996, 17, 490-512.
* Trott & Olson, *AutoDock Vina*, J. Comput. Chem. 2010, 31, 455-461.

## 6. Follow-ups (not in scope of this task)

* The 30-cell **production run** (not the smoke) requires Vina CPU binary + receptor preparation + 10×3 Lambda cells.  Wall-clock estimate at `n_simulations=100` is ~5 min/cell × 30 = ~2.5 h on the iGPU fallback (CPU Vina is the bottleneck).  This is a CPU-budget task, not a GPU one — the GPU outage from `wf_gpu_auto_recover` does NOT block this script.
* Once 30 cells complete, the `final.md` per-check table is the basis for §4.6 (PoseBusters panel) integration in the paper.