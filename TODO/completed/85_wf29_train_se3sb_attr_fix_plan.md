# TODO-85 — Wf-29 train_se3sb.py AttributeError fix (Phase 1)

**Date**: 2026-09-26
**Status**: 🚀 **Active — Phase 1 COMPLETE**
**Priority**: HIGH (blocks all W2.1 SE(3) SB training runs)
**Owner**: Hugo Lee + ultracode agent
**Parent**: `Wf-29` (workflow) — Phase 1 of multi-phase rollout

## Bug summary

`molmetal/scripts/train_se3sb.py` line 613 (pre-fix) referenced
`args.se3_sinkhorn` — a non-existent argparse attribute. The flag
is `--se3-use-sinkhorn` which lands as `args.se3_use_sinkhorn`.

First call to `main()` raised:

```
AttributeError: 'Namespace' object has no attribute 'se3_sinkhorn'
```

BEFORE any training started (the decoder is built for JSON metadata
on line 610-614 of the pre-fix script).

## Fix (1 line)

`molmetal/scripts/train_se3sb.py` line 620 (post-fix):

```diff
 decoder = SE3SBDecoder(
     eps=args.se3_sinkhorn_eps,
     n_iters=args.se3_sinkhorn_iters,
-    use_sinkhorn=args.se3_sinkhorn,
+    use_sinkhorn=args.se3_use_sinkhorn,
 )
```

No other lines touched. Recipe defaults (`RECIPE_SE3_USE_SINKHORN = True`)
and flag definition (`--se3-use-sinkhorn`, `store_true`) unchanged.

## Verification chain

1. **`--help` exits 0** — argparse contract preserved.
2. **Smoke run** — `--seeds 0 --epochs 1 --train-max-per-metal 50` exits 0
   on CPU. No `AttributeError` in stdout/stderr.
3. **Regression test suite** — new file
   `molmetal/tests/test_train_se3sb_smoke.py` (11 tests):
   - `test_help_exits_zero` — argparse well-formed
   - `test_parse_args_se3_attribute_names` — `se3_use_sinkhorn` exposed, NOT `se3_sinkhorn`
   - `test_se3sb_decoder_constructs_from_parse_args` — literal 1-line regression
   - `test_main_smoke_one_seed_one_epoch` — full `main()` exits 0
   - `test_metrics_json_after_smoke` — JSON has recipe name
   - `test_se3_init_meta_records_flag` — `use_sinkhorn` flips with flag
   - `test_se3_integration_smoke_function_is_truthy` — SE(3) path live
   - `test_ckpt_file_written` — `se3sb_seed0.pt` written
   - `test_summary_json_contains_recipe` — summary JSON contract
   - `test_se3_use_sinkhorn_is_store_true` — CLI contract
   - `test_module_imports_cleanly` — no import side effects

## Constraint check

- Only 1 line modified (the bug at line 620).
- All 11 regression tests pass on CPU.
- venv `mol-metal-venv` has `numpy 2.5.3` + `torch 2.14.0+cu130` —
  no baseline-env workaround required.

## Files

- `molmetal/scripts/train_se3sb.py` (1 line, line 620)
- `molmetal/tests/test_train_se3sb_smoke.py` (NEW, 11 tests)
- `molmetal/reports/wf29_fix_train_se3sb/fix_log.md` (NEW, summary)

## Status

**Phase 1 COMPLETE**. W2.1 SE(3) Schrödinger-Bridge training unblocked.

## Next

- Phase 2 (deferred): W2.1 5-seed × 50-epoch × 5-metal multitask training run
  on CUDA host (out of scope for CPU-only fix verification).
