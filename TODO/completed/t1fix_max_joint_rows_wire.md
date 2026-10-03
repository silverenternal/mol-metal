# T1-FIX — `--max-joint-rows` FLAG_ONLY wire-in (§5.7 smoke PASS)

**Date:** 2026-10-03 · **Lane:** trainer + tests + TODO/completed only

**Choice: WIRE (not remove).** `--metals` is not a substitute — for single-metal smoke
shapes the joint pool is the full metal train set (Ir = 3444 rows), so no existing knob
scaled the pool. Removal would also break 3 test assertions (`test_wf334u1_gpu_smoke.py`
L167 + L218-275; `test_wf335r2_wireup_reverify.py` L104/160) and leave no pool cap.

## Code change — `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
1. `train_one_metal_one_seed()` gains `max_joint_rows: Optional[int] = None`.
2. After the `<50` floor check and before `train_pool_mids`/featurisation (one
   assembly site, L1450-1465):
   ```python
   if max_joint_rows and 0 < max_joint_rows < len(train_pool_idx):
       _cap_rng = np.random.default_rng(seed)
       train_pool_idx = train_pool_idx[
           _cap_rng.permutation(len(train_pool_idx))[:max_joint_rows]
       ]
   ```
   Seeded permutation — NOT head-truncation: the pool is a metal-blocked concatenate
   in FIVE_METALS order, so a head cap would keep only the first metal's rows.
3. main() call site threads `max_joint_rows=getattr(args, "max_joint_rows", None)`.
4. Help text documents the cap semantics. Note: the `n_train` field in the DONE
   summary is target-train (L1858) and does NOT move; the cap is visible via
   `n_joint_train` (ckpt, L1859) and via the featurise/epoch wall-time drop.

## §5.7 smoke — CPU, 2 epochs, `--metals Ir --seeds 0 --hidden 64 --depth 2`
Invocation = 330b2 §1.1 measured recipe, `--num-workers` replaced by
`--cpu-num-workers 2` (the former no longer exists in this argparse), epochs 1->2.

| run | n_joint_train (ckpt) | n_train | epoch1 loss | epoch2 loss |
|---|---|---|---|---|
| no flag | **3444** | 3444 | 1.3477 | 0.9751 |
| `--max-joint-rows 200` | **200** | 3444 | 4.1992 | 2.5745 |

- (1) parse_args accepts the flag — `--help` + in-process round-trip test PASS.
- (2) trainer initialises — both runs exit 0.
- (3) row count ACTUALLY CHANGES — 3444 -> 200 (read from saved checkpoints).
- (4) loss decreases — 1.3477 -> 0.9751 and 4.1992 -> 2.5745.
- decode_ratio: **N/A** — D-MPNN classifier, no decoder (not faked).
- Corroborating: featurise 5.6 s -> 1.7 s; epoch 7.7 s -> 0.7 s.

Logs (scratch, not committed): `/tmp/t1fix_smoke_nocap.log`, `/tmp/t1fix_smoke_cap.log`.
Data: `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (MEASURED, real labels).

## Regression guard
`molmetal/tests/test_t1fix_max_joint_rows_wire.py` (NEW, 3 tests PASS in 1.34 s):
live-read count >= 4, cap expression + seeded permutation present, parse round-trip,
kwarg default None.

## Evidence (ls -la, current disk state)
```
-rw-r--r-- 1 hugo hugo 142014 Oct  3 15:02 molmetal/scripts/train_dmpnn_multitask_wasserstein.py
-rw-r--r-- 1 hugo hugo   2300 Oct  3 15:03 molmetal/tests/test_t1fix_max_joint_rows_wire.py
```

## Honest framing
- Capped-run losses are on a 200-row subsample — smoke-only, NOT a quality claim.
- Real runs MUST omit the flag: default None = full pool, bit-for-bit unchanged
  (proven by the nocap run n_joint_train=3444).
- No GPU launch; no commit (per task HARD RULES).
- `pre_flight_check.py` L7880 still narrates the flag as FLAG_ONLY with a `--metals Re`
  workaround — that file is owned by another workflow and was not touched; its
  workaround remains valid but is no longer required.
