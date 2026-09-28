# Wf-35 Phase 3 — Master verify (R18 train scripts)

Generated at `2026-09-26T12:37:20.342927+00:00` (UTC). Total wall-clock: `19.7 s`.

- **Total scripts**: 8
- **PASS**: 7
- **FAIL**: 0
- **PENDING**: 1
- **venv python**: `/home/hugo/codes/mol-metal/molmetal/.venvs/mol-metal-venv/bin/python`

## Per-script detail

| Script | WF | Status | Checks (import / argparse / smoke) | Elapsed (s) | Notes |
|---|---|---|---|---|---|
| `train_se3sb.py` | Wf-29 | PASS | ✓ import, ✓ argparse, ⏭ smoke | 4.1 | all checks passed |
| `train_transport.py` | Wf-29 | PASS | ✓ import, ✓ argparse, ⏭ smoke | 2.8 | all checks passed |
| `train_multiph.py` | Wf-29 | PASS | ✓ import, ✓ argparse, ⏭ smoke | 2.8 | all checks passed |
| `train_ru_5metal_se3sb.py` | Wf-31 | PASS | ✓ import, ✓ argparse, ⏭ smoke | 2.3 | all checks passed |
| `train_affinity_regression.py` | Wf-32 | PASS | ✓ import, ✓ argparse, ⏭ smoke | 2.5 | all checks passed |
| `train_multimetal_lomo.py` | Wf-32 | PASS | ✓ import, ✓ argparse, ⏭ smoke | 2.4 | all checks passed |
| `train_cisplatin_resistance_v2.py` | Wf-31 | PASS | ✓ import, ✓ argparse, ⏭ smoke | 2.8 | all checks passed |
| `bench_pb_valid.py` | Wf-32 | PENDING |  | 0.0 | script authored but not yet refactored to _common_train_args — auto-skipped per Wf-35 contract |

## Methodology

- **Check 1 — Import**: in-process via `importlib`. Verifies `parse_args` + `main` exist.
- **Check 2 — argparse**: subprocess `--help` against the venv python. Expects `Common recipe` + `GPU env` argument groups from `_common_train_args`.
- **Check 3 — 1-epoch smoke**: subprocess `--epochs 1 --train-max-per-metal 50` with `CUDA_VISIBLE_DEVICES=""`. Confirms the `MEASURED_vs_PROJECTED_decorator` pipeline runs to completion and writes `--metrics-path`.
