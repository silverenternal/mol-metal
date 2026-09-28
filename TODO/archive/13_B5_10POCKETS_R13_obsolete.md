# TODO-32 B5 — Lambda pipeline smoke, 10-pocket expansion — IN PROGRESS

> **LEGACY** — R13-era in-progress tracker, OBSOLETE.
> Superseded by: `TODO/completed/36_lambda_b1_coupling_deflex.md` (10/10 pocket pilot shipped; 3-seed 10×3 PASS the TODO-25 §3 ≥3 distinct metal-ligand-pair gate).
> Moved to `archive/` on 2026-09-25 — see `TODO/01_strategy/IMPROVEMENT_BACKLOG.md` §A5.

**Status:** ⚙️ IN PROGRESS — wrapper script `wf_lambda_pipeline_10pockets.py`
launched at 2026-09-25 15:38 CST against `test_010..test_019`.
Each pocket ≈ 10-12 min CPU → 10 pockets ≈ 100-120 min wall (sequential).
Live log: `logs/lambda_pipeline_smoke_10pockets_runner.log`.
Per-pocket outputs: `logs/lambda_pipeline_smoke_10pockets/<pocket_id>/`.
Aggregate + final report will land in:
- `logs/lambda_pipeline_smoke_10pockets/aggregate.json`
- `molmetal/reports/wf_lambda_pipeline_10pockets/final.md`

This entry will be promoted to `TODO/completed/` with the actual
numbers once the run completes.

---

## What was launched

| Item | Value |
|---|---|
| Wrapper script | `molmetal/scripts/wf_lambda_pipeline_10pockets.py` |
| Manifest | `molmetal/data/test_010_019_subset_manifest.csv` (10 rows) |
| Pocket IDs | `test_010, test_011, ..., test_019` |
| Per-pocket budget | 1000 MCTS sims × 12 seeds × ~83 sims/seed, top_k=100, max_depth=3 |
| FPU reduction | 0.25 (KataGo canonical) |
| PLL prior | k=16, k_nn=8 (Liu & Feng 2026 fingerprint) |
| 5 click rules | CuAAC, DielsAlder, SPAAC, SPC, ThiolEne |
| Multi-seed diversity | seed per pocket = pocket index (0..9) |

Wall-clock budget: ~100-120 min sequential; should fit within the
ultracode session timebox.

---

## Files added (this session)

| File | Change |
|---|---|
| `molmetal/scripts/wf_lambda_pipeline_10pockets.py` | NEW — wrapper that loops the existing single-pocket smoke over 10 pockets and aggregates the per-pocket stats into `aggregate.json` + `final.md`. |
| `molmetal/molmetal_lam/atoms/combinators.py` | METAL_ATOMS extended from 6 → 30 entries (UC-B1). All 24 new metals use the `valence=0, lone_pairs=CN` convention with tmQM-canonical geometries. `_EXPECTED_METAL_CN` extended in lockstep. |
| `molmetal/tests/test_lambda_combinators_expansion.py` | 4 new test funcs + 60 parametrize cases — METAL_ATOMS 30-entry contract. **69 passed.** |
| `TODO/completed/32_trackB_uc_b1_metal_atoms_30.md` | NEW — UC-B1 (Track B-剩余) "DONE" entry. |

---

## Parallel work (Track A UC-A1)

| Item | Value |
|---|---|
| `molmetal/references/flow_matching` | Symlinked (already present from prior session) |
| `molmetal/references/FLOWR` | Newly cloned from `jule-c/flowr` (Cremer 2025/2026 strong baseline) |
| Adapter落点 | Verified — `molmetal/adapters/flow_matching_lipman/{amp,connectivity_decoder,_reference,_reference_loader,__init__}.py` all present |
| Log | `logs/trackA_uc_a1_clone.log` |

---

## Verification plan (when smoke completes)

1. `python -m pytest molmetal/tests/test_lambda_combinators_expansion.py` —
   ensure 69 tests still pass after smoke modifications.
2. Validate `aggregate.json` schema:
   `{n_pockets, pocket_ids, per_pocket[10 entries], aggregate:{...}}`.
3. Validate `final.md` per-pocket table renders all 10 rows.
4. Per-pocket `summary.json` has 28 keys each (matching the workflow H
   summary on test_001).
5. Move this entry to `TODO/completed/32_trackB_b5_10pocket_smoke_done.md`
   with actual measured numbers.

---

End of in-progress status.