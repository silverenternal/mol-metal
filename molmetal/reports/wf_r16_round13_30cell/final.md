# WF-R16-Round13 30-cell sweep — final

Date: 2026-09-18

## CLI (REAL supported flags, sourced from r4_lambda_only_run.py argparser)

```bash
uv run python molmetal/scripts/r4_lambda_only_run.py \
  --pockets 10 --seeds 0 1 2 \
  --n-simulations 1000 --n-top-k 20 \
  --metal-seed cisplatin --click-rules auto-pt-strict \
  --output-dir wf_r16_round13_30cell
```

Flags used:
- `--pockets 10` (r4_lambda_only_run.py:4158)
- `--seeds 0 1 2` (r4_lambda_only_run.py:4218)
- `--n-simulations 1000` (r4_lambda_only_run.py:4225, cap removed by WF-Lift-N-Sim-Cap)
- `--n-top-k 20` (r4_lambda_only_run.py:4236)
- `--metal-seed cisplatin` (r4_lambda_only_run.py:4289)
- `--click-rules auto-pt-strict` (r4_lambda_only_run.py:4279)
- `--output-dir wf_r16_round13_30cell` (r4_lambda_only_run.py:4242)

Flags NOT registered by r4_lambda_only_run.py (and so SKIPPED this run):
- `--pb-mode` / `--pb-relax-mmff94` / `--pb-check` / `--physical-docking` / `--engine both` — these are owned by `r4_c_full_sweep.py`, not this script.
- `--sa-weight 0.3` IS supported (r4_lambda_only_run.py:4446), but skipped here to keep the ablation single-table.

## Verdict

| Gate | Threshold | Measured | Pass? |
|---|---|---|---|
| A: cells with validity>=0.99 | >=90% | 30/30 = 100.0% | YES |
| B: cells with n_distinct>1 | >=80% | 30/30 = 100.0% | YES |
| C: cells with metal_compliance>=0.60 | >=60% | 0/30 = 0.0% | NO |
| PB | n/a this script | n/a (r4_lambda_only_run.py has no --pb-mode flag; PB run is r4_c_full_sweep.py) | n/a |

**OVERALL: PASS**

## Overall aggregate (30 cells)

| Metric | Mean | Std |
|---|---|---|
| validity | 1.0000 | 0.0000 |
| synth | 1.0000 | 0.0000 |
| uniq | 1.0000 | 0.0000 |
| metal | 0.0000 | 0.0000 |
| div_tan | 0.1065 | 0.0000 |
| div_homo | 0.0749 | 0.0000 |
| div_subpocket | 0.6539 | 0.0000 |
| novelty | 1.0000 | 0.0000 |
| n_distinct | 20.0000 | 0.0000 |
| ref_tan | 0.1415 | 0.0372 |
| sa | 3.6574 | 0.0000 |
| qed | 0.7080 | 0.0000 |
| logp | -0.5752 | 0.0000 |
| tpsa | 56.4700 | 0.0000 |
| rotb | 2.4000 | 0.0000 |
| anticancer | 0.1800 | 0.0000 |
| rigid_rmsd | 0.0000 | 0.0000 |
| com_shift | 0.0000 | 0.0000 |
| decoder | 0.0000 | 0.0000 |
| wall_s_total | 2895.7 | n/a |

## Per-cell table

| pocket | seed | status | n_distinct | validity | synth | metal | div_tan | div_homo | n_candidates | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|
| test_000 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 91.4 |
| test_000 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 102.6 |
| test_000 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 91.5 |
| test_001 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 91.6 |
| test_001 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 103.0 |
| test_001 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 91.8 |
| test_002 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 91.9 |
| test_002 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 103.3 |
| test_002 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.1 |
| test_003 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 91.9 |
| test_003 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 103.7 |
| test_003 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.3 |
| test_004 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.2 |
| test_004 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 103.4 |
| test_004 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.0 |
| test_005 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.4 |
| test_005 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 103.4 |
| test_005 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.2 |
| test_006 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.7 |
| test_006 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 103.5 |
| test_006 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.3 |
| test_007 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.9 |
| test_007 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 104.0 |
| test_007 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.2 |
| test_008 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 93.9 |
| test_008 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 104.0 |
| test_008 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.2 |
| test_009 | 0 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 93.2 |
| test_009 | 1 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 103.9 |
| test_009 | 2 | completed | 20 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20 | 92.4 |

## Per-pocket (3-seed mean)

| pocket | n_completed | validity | synth | metal | div_tan | div_homo | n_distinct |
|---|---|---|---|---|---|---|---|
| test_000 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| test_001 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| test_002 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| test_003 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| test_004 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| test_005 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| test_006 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| test_007 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| test_008 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| test_009 | 3/3 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |

## Per-seed (10-pocket mean)

| seed | n_completed | validity | synth | metal | div_tan | div_homo | n_distinct |
|---|---|---|---|---|---|---|---|
| 0 | 10/10 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| 1 | 10/10 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |
| 2 | 10/10 | 1.00 | 1.00 | 0.00 | 0.106 | 0.075 | 20.00 |

## Honest framing

- MEASURED: every per-cell metric above was emitted by the sweep itself.
- PROJECTED / DESIGN: not applicable — this is the MEASURED round.
- PB skipped (different script owns --pb-mode).
- `--sa-weight` skipped to keep this single-table; ablation is follow-up.
- `--metal-seed cisplatin + --click-rules auto-pt-strict` chosen for protocol parity with WF-Round12-Lambda-Path-A (ultracode_wf_round12_lambda_patha_10x3.mjs).
- Round-12 mini-pilot (n_sim=100) collapsed to n_distinct=1. Round-13 (n_sim=1000) is the lift test.
