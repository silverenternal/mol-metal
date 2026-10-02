# Wf-298B — wf278a_g4_aggregate.py multi-root patch

## Task

Per Wf-294D, the existing `wf278a_g4_aggregate.py` only accepted a single
`--input-dir`, so it could not merge G4 (30-epoch) + G5 (50-epoch) sweep
results when a `(metal, seed)` pair appears under both roots (e.g.
Ru/seed0). This patch adds `--input-dirs` (multi-root) with last-wins-
by-mtime dedup and bumps the expected pair count from 14 to 15.

## Diff summary

### `molmetal/scripts/wf278a_g4_aggregate.py`

1. Added `collect_seed_metrics_multi_root(roots: List[Path])` that walks
   multiple roots, dedups by `(metal_dir_lower, seed_n)` tuple, and keeps
   the file with the largest `st_mtime` (ties → first-encountered wins,
   which is stable because argparse preserves CLI arg order).
2. Refactored `collect_seed_metrics_nested(root)` to be a thin wrapper
   over `collect_seed_metrics_multi_root([root])` (no behaviour change
   for the single-root path).
3. `--input-dir` is now optional; `--input-dirs <PATH> [<PATH> ...]`
   accepts one or more roots. At least one must be passed (else exit 2).
4. Added `--expected-pairs` CLI flag with `default=15` (5 metals ×
   3 seeds). The legacy hard-coded `"expected_pairs": 14` in the JSON
   payload is now wired through `args.expected_pairs`.
5. Summary JSON now records `input_dirs` (list) + `cascade_mode: bool`
   in addition to the legacy `input_dir` (single-root backwards compat).

### `molmetal/tests/test_wf278a_multi_root.py` (new, 3 tests)

1. `test_multi_root_merge` — G4 (Ru/seed0, Ir/seed1) + G5 (Ru/seed0
   overlap, Os/seed2 new) → 3 unique pairs, NOT 4.
2. `test_last_wins_dedup` — Both roots have Ru/seed0; G5 has newer mtime
   + distinctive `roc_auc=0.9999`. The picked payload must have
   `epochs=50` and `roc_auc=0.9999` (G5 wins on mtime).
3. `test_expected_pairs_15` — Source-level check that
   `--expected-pairs default=15` is set and the legacy
   `"expected_pairs": 14` hard-code is gone.

## Cascade run (G4 + G5)

```
$ python molmetal/scripts/wf278a_g4_aggregate.py \
    --input-dirs results/wf124_p3_remaining_metals_g4 \
                  results/wf124_p3_50epoch_full_sweep_g5 \
    --output /tmp/wf298b_cascade_training_summary.json
```

Output JSON:
- `n_pairs_total = 15`
- `expected_pairs = 15`
- `cascade_mode = True`
- `input_dirs = [results/wf124_p3_remaining_metals_g4, results/wf124_p3_50epoch_full_sweep_g5]`

5-metal aggregate: `test_auc = 0.7243 ± 0.0460` (n=15).

Per-metal (n=3 each):
- Ru: 0.7561 ± 0.0112
- Ir: 0.6702 ± 0.0056
- Os: 0.7025 ± 0.0205
- Rh: 0.7227 ± 0.0540
- Re: 0.7700 ± 0.0244

## Test run

```
$ python -m pytest molmetal/tests/test_wf278a_multi_root.py -v
============================= 3 passed, 1 warning in 0.45s ==============================
```

## Stats

- `n_lines_added`: ~90 lines in `wf278a_g4_aggregate.py` (multi_root
  helper + CLI changes) + 161 lines in `test_wf278a_multi_root.py`.
- `cascade_pairs`: 15
- `expected_pairs`: 15
- `n_tests_pass`: 3