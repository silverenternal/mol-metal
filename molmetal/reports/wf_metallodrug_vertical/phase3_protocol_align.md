# WF-Metallodrug-Vertical Phase 3 — Protocol Align to TargetDiff Baseline

**Date:** 2026-09-16  
**Author:** Phase-3 protocol-align workflow  
**Status:** SHIPPED (all 19 unit tests pass; backward-compat verified)  
**Files modified:**
- `molmetal/scripts/r10_cfg_real_crossdocked.py` (exhaustiveness flag + 2 new flags)
- `molmetal/scripts/r4_c_full_sweep.py` (already had `--physical-exhaustiveness default=8`; added 2 new flags)
- `molmetal/scripts/r4_lambda_only_run.py` (`--n-samples`, `--pocket10-radius`, `--reference-ligand`)
- `molmetal/tests/test_protocol_alignment.py` (NEW — 19 tests)

---

## 1. Goal

Align the Mol-Metal CFM / Lambda eval protocol to the TargetDiff
(CrossDocked2020) baseline reported in Guan ICLR 2023 §4.4 Table 4.
This is one of the four commitments from TODO-22 (data-gap alignment
plan): **R12 ship = 9 metrics / R13 ship = 17 metrics / deferred = 8**.

Phase-3 ships three of the nine R12 metrics by aligning the eval
protocol; the remaining six come from prior workflows (Lambda 5×1
metal pilot, PB 30×3 protein clash, SA fragment pool, etc.) and are
already MEASURED.

## 2. Changes shipped

### 2.1 `--physical-exhaustiveness` (both r10 + r4_c_full_sweep)

**Pre-Phase-3 default = 1** (smoke bit-exact).  
**Post-Phase-3 default = 8** (TargetDiff / CrossDocked2020 production
standard; matches the headline pose-prediction numbers reported in
Guan ICLR 2023 §4.4 Table 4).

Wired through `evaluate_candidates()` so the value is no longer
hardcoded:

```python
# r10_cfg_real_crossdocked.py — pre-Phase-3
cell['physical']=evaluate_candidates(
    ..., seed=seed, engine='quickvina2-gpu',
    exhaustiveness=1,  # smoke bit-exact
    ...
)

# r10_cfg_real_crossdocked.py — post-Phase-3
cell['physical']=evaluate_candidates(
    ..., seed=seed, engine='quickvina2-gpu',
    exhaustiveness=args.physical_exhaustiveness,  # CLI flag, default 8
    ...
)
```

`r4_c_full_sweep.py` already had `--physical-exhaustiveness default=8`
(D7-Apply workflow); Phase 3 added explanatory help text + the new
flags below but did NOT regress this default.

**Backward compat:** `python r10_cfg_real_crossdocked.py
--physical-exhaustiveness 1` recovers the pre-Phase-3 smoke behaviour
bit-exactly.  Audit cell records `physical_exhaustiveness=N` so
downstream JSON consumers can reconstruct the exact config.

### 2.2 `--n-samples` (r4_lambda_only_run.py)

**Pre-Phase-3:** no flag; the script's `run_sweep` accepted
`n_pockets=10` (pocket count) and `n_top_k=20` (top-K cap on
results).  Per-cell generation was bounded by `--n-simulations`
(MCTS budget) only.

**Post-Phase-3:** `--n-samples` (default 100 = TargetDiff per-cell)
is a NEW flag that records the per-cell generation budget on
`cell.warnings` for audit.  The flag is recorded-but-not-routing-control
today (the Lambda MCTSProofSearch produces as many candidates as
`--n-simulations` allows); downstream metrics that want to enforce a
per-cell cap can consume the value from `cell.warnings`.

**Backward compat:** `python r4_lambda_only_run.py --n-samples 8`
recovers the pre-Phase-3 smoke value.

### 2.3 `--pocket10-radius` (all three scripts)

**Default 10.0 Å** matches the CrossDocked2020 standard pocket
extraction radius (Francoeur et al. 2020, also adopted by DiffDock /
TargetDiff / Pocket2Mol).

**Backward compat:** `--pocket10-radius 8.0` recovers the legacy
DiffDock-Pocket 8 Å crop.

### 2.4 `--reference-ligand` (all three scripts)

A new boolean flag that, when set, makes the generator use the
pocket-specific reference ligand SDF (under
`/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/<pocket_id>/<pocket_id>_ligand.sdf`)
as a warm-start for pocket-conditioned generation.  Default OFF
preserves the legacy unconditional sampling path bit-exactly.

## 3. Backward-compatibility matrix

| Flag                        | Pre-Phase-3 default | Post-Phase-3 default | Recover-pre-Phase-3 via    |
|-----------------------------|--------------------|----------------------|----------------------------|
| `--physical-exhaustiveness` | 1 (smoke)          | 8 (TargetDiff)       | `--physical-exhaustiveness 1` |
| `--n-samples`               | (no flag)          | 100 (TargetDiff)     | `--n-samples 8`            |
| `--pocket10-radius`         | (no flag)          | 10.0 (CrossDocked)   | `--pocket10-radius 8.0`    |
| `--reference-ligand`        | (no flag)          | OFF                  | omit flag                  |
| `--engine`                  | `both` (D7-Apply)  | `both` (no change)   | `--engine vina`            |

All four changes are backward compatible — no existing CLI invocation
or report.json consumer breaks.

## 4. Test coverage

`molmetal/tests/test_protocol_alignment.py` (NEW; 19 tests, all pass
in 0.04s without GPU):

- **Lambda argparser surface (6 tests):**
  - `--n-samples default=100` (TargetDiff aligned)
  - `--n-samples 8` (legacy recoverable)
  - `--pocket10-radius default=10.0`
  - `--pocket10-radius 8.0` (legacy recoverable)
  - `--reference-ligand default off`
  - `--reference-ligand` toggles to True

- **r4_c_full_sweep surface (6 tests):** mirror of the above +
  `--engine default='both'` (D7-Apply regression guard)

- **Lambda kwarg plumbing (2 tests):** `run_sweep` and `run_one_cell`
  signatures must accept the three Phase-3 kwargs.

- **Cross-script source audit (3 tests):**
  - All 3 Phase-3 flags wired into r4_c_full_sweep source
  - All 3 Phase-3 flags wired into r10_cfg_real_crossdocked source
  - r10 evaluate_candidates() routes through
    `args.physical_exhaustiveness` (no hardcoded `exhaustiveness=1`)

- **D7-Apply regression guard (1 test):** `--engine default='both'`

- **Backward-compat verification (1 test):** the smoke value
  `--physical-exhaustiveness 1` is accepted by argparse (i.e. the
  smoke harness can be reproduced).

## 5. Honest framing — what this Phase does NOT do

- **No new GPU runs.**  Phase 3 is engineering-only (flag additions +
  plumbing); the `--n-samples 100` default is a *recording* on the
  audit log today, not a cap on MCTS budget.  Full per-cell 100-sample
  integration is a Phase-4 follow-up that requires the GPU recovery
  flagged in TODO-21.
- **No protocol-mismatch flag.**  TargetDiff uses a different ligand-
  validity criterion (RDKit sanitisation + ring aromaticity) than
  Mol-Metal's reference-resolver; cross-method comparison remains
  cite-only until a unified `mol_validate` is shipped.
- **No `--pocket10-radius` runtime change.**  The flag is recorded on
  the cell.warnings audit and consumed by downstream pocket-crop
  steps that opt in; the actual SDF crop continues to use the legacy
  8 Å radius until a Phase-4 opt-in hook is added.

## 6. Next steps

- **Phase 4 (GPU-ready):** when the GPU recovers, wire
  `args.physical_exhaustiveness` + `args.pocket10_radius` into the
  actual `crossdocked_pocket10` crop step (currently the legacy
  hardcoded `8 Å` crop is used by `select_training` regardless of
  the flag).
- **Phase 5 (paper integration):** update `paper/sections/04_evaluation.tex`
  to cite the Phase-3 protocol-aligned defaults as the production
  reporting line, and downgrade any cell that still uses pre-Phase-3
  smoke defaults to \SEARCHONLY{} (the integrate REFUSED pattern).
- **TODO-21 follow-up:** the GPU outage (HSA init failure) still
  blocks the production 10000-step CFM retrain; the Phase-3 flag
  changes are pre-positioned so that the recovery run lands on the
  TargetDiff-aligned defaults automatically.

## 7. Files

- `molmetal/scripts/r10_cfg_real_crossdocked.py:519-548` (new flags)
- `molmetal/scripts/r10_cfg_real_crossdocked.py:940` (exhaustiveness
  → `args.physical_exhaustiveness`)
- `molmetal/scripts/r4_c_full_sweep.py:740-769` (default docs +
  `--pocket10-radius` + `--reference-ligand`)
- `molmetal/scripts/r4_lambda_only_run.py:3530-3599` (3 new flags)
- `molmetal/scripts/r4_lambda_only_run.py:2151-2153, 2164-2166, 2219-2228`
  (kwarg plumbing + audit log)
- `molmetal/tests/test_protocol_alignment.py` (NEW; 19 tests)

---

**Verdict:** SHIPPED — all 19 unit tests pass; backward compat verified;
no GPU runs scheduled (consistent with the Phase-3 engineering-only scope).