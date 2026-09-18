# R10 Pt(II) Prior Harness — Recon (2026-09-14)

**Status:** ON-DISK, RUNNABLE (untested at run-time per `先别跑实验` constraint — this recon is read-only).

## Harness surface (MEASURED)

- Script: `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_pt_prior_ablation_1h36.py` (33 KB, 782 lines, mtime 2026-09-13).
- CLI surface (`argparse`, lines 467-512): `--prior-weights` (default `[0.0, 0.1]`), `--n-mols 20`, `--train-steps 30`, `--n-steps 8`, `--exhaustiveness 2`, `--hidden-dim 32`, `--n-layers 2`, `--lr 2e-4`, `--seed 0`, `--device` (cuda/cpu auto), `--context-dropout 0.1`, `--coord-clip 10.0`, `--pocket-embed-norm-target 1.0`, `--output-prefix` (`molmetal/reports/r10_pt_prior_ablation_1h36`), `--skip-dock` (smoke flag).
- Internal gate (line 319): asserts `MetalGeometryPrior.enabled == (metal_prior_weight > 0)` so the harness cannot emit a misleading delta if the flag drifts.
- Outputs (per report §Outputs): CSV + JSON + MD, all prefixed `molmetal/reports/r10_pt_prior_ablation_1h36.{csv,json,md}`.

## Reported harness location + prior-weights protocol (CITED from round10 report)

- `molmetal/reports/round10_pt_prior_ablation.md` documents the exact launch command (lines 110-115):
  `uv run python molmetal/scripts/r10_pt_prior_ablation_1h36.py --n-mols 20 --train-steps 30 --prior-weights 0.0 0.1`
- Prior-weights protocol: `LipmanFlowMatchingAdapter(metal_prior_weight=...)` plumbs into `_velocity_with_metal_prior`; `weight=0` short-circuits to bare velocity (no autograd graph), `weight>0` adds the analytic gradient every `metal_prior_k_every` steps.
- Gate table (lines 47-54) lists all 6 algorithm gates as wired; 12/12 unit tests + 13/13 regression tests green per the report.

## Tests touching this harness (MEASURED)

- `molmetal/molmetal_lam/tests/test_round10_pt_prior.py` — dedicated 12-test module (covers `enabled` default, disable/enable toggle, weight×enabled independence, `apply_prior`/`prior_loss` short-circuits, π/2 ideal angle, harness-existence, `--help`, `--skip-dock` smoke, canonical 1h36 PDB constant).
- No matching tests under `molmetal/tests/` (top-level tests directory) — all harness coverage lives under `molmetal_lam/tests/`.

## Pocket 1h36 receptor path (MEASURED vs path-in-question)

- The script does NOT consume `/mnt/storage/data/molmetal/crossdocked_pocket10/` — that path does not exist (MEASURED: `ls` → ENOENT).
- Actual path consumed: `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb` (45 KB, hard-coded in `EXAMPLE_PDB`, lines 58-60 of the harness). This file exists.
- The `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/` directory DOES exist (CrossDocked 100-pocket sweep root used by `lambda_100pocket_sweep.py`), but it contains 100 directories named by `{PDB}_{REC}_{RES}_{NUM}_{RECID}` — NOT a `1h36` flat file. The single-pocket micro-bench deliberately uses the bundled TargetDiff fixture (45 KB, hand-checked), not the 100-pocket tarball.

## Path list (load-bearing)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_pt_prior_ablation_1h36.py` — harness (782 lines, CLI surface above).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/round10_pt_prior_ablation.md` — round-10 axis D report (documents the launch command + algorithm-gate table).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_round10_pt_prior.py` — 12-test pytest module covering the harness.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb` — actual 1h36 PDB consumed by the harness (NOT under `/mnt/storage/data/molmetal/`).
- `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/` — alternate CrossDocked 100-pocket root (NOT used by R10 harness; only by lambda_100pocket_sweep).
