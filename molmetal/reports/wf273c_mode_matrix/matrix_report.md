# Wf-273C: chemprop v3 mode test matrix coverage

## Summary

Wf-273C surveys the chemprop v3 driver
(`molmetal/scripts/train_chemprop_v2_v3.py`) to enumerate every
`--chemprop-mode` value exposed in argparse, verify each mode has a
distinct code path, smoke-test each mode end-to-end on CPU, and audit
the existing unit-test coverage. This report satisfies TODO-130 G5
follow-up — the Wf-263A + Wf-266 + Wf-269A mode contract is fully
covered and reproducible.

## G1 — modes accepted by `--chemprop-mode`

The argparse `--chemprop-mode` flag exposes **5** choices
(`molmetal/scripts/train_chemprop_v2_v3.py:1523-1524`):

```
choices=("v2", "v3", "v3-fallback", "v3-fix", "shared-encoder"),
default="v3-fallback",
```

Five modes are accepted; **shared-encoder IS implemented** (per
Wf-269A-Implement Path C structural fix in the same file). The
implementation is on lines 287-337 (`_LinearProbeHead` class) +
lines 1883-1908 (mode dispatch).

## G2 — modes list

| # | Mode           | Audit source                                                  | Effective enhanced_head | Effective metal_attention | _effective_min_epochs | _effective_lomo_init_scale |
|---|----------------|---------------------------------------------------------------|-------------------------|---------------------------|-----------------------|----------------------------|
| 1 | v2             | Wf-263A-Implement (line 1848-1857)                            | False                   | False                     | 1                     | 1.0                        |
| 2 | v3             | Wf-263A-Implement (line 1858-1870)                            | user-supplied           | user-supplied             | 1                     | 1.0                        |
| 3 | v3-fallback    | Wf-263A-Implement default (line 1835-1847)                    | False                   | False                     | 1                     | 1.0                        |
| 4 | v3-fix         | Wf-266-Implement (line 1803-1834)                             | False                   | False                     | user-supplied (10)    | user-supplied (1e-3)       |
| 5 | shared-encoder | Wf-269A-Implement Path C (line 1883-1908)                     | False                   | False                     | user-supplied (10)    | 1.0 (Path C replaces)      |

## G3 — smoke test outcomes (CPU, 1-epoch, h=64, d=2)

The smoke command prescribed in the task invoked `--csv` / `--data-csv`
flags that the script does not expose (data is loaded via the
`load_metal_cytotox()` resolver in `molmetal/data/metal_cytotox.py`).
The corrected command dropped those flags and capped rows at 200 to
keep wall-time cheap. Run command:

```
MOLMETAL_ALLOW_CPU=1 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_chemprop_v2_v3.py \
    --chemprop-mode MODE --metals Ru --seeds 0 --epochs 1 \
    --hidden 64 --depth 2 --device cpu --no-bf16 \
    --allow-single-seed --out-dir /tmp/wf273c_MODE \
    --time-budget-s 300 --max-rows 200 [--enhanced-head]
```

| Mode           | test_auc (Ru, seed=0) | Epochs actually trained | status    | Notes |
|----------------|-----------------------|-------------------------|-----------|-------|
| v2             | 0.2509                | 1                       | ok        | Single Ru row, 200-cap; below random expected with 1 epoch + LOMO. |
| v3-fallback    | 0.4122                | 1                       | ok        | Same default path as v2; identical head selection, different code path name. |
| v3             | 0.3082                | 1                       | ok        | --enhanced-head required to actually swap the head. |
| v3-fix         | 0.5000                | 10 (bumped from 1)      | ok        | --epochs auto-bumped to --chemprop-min-epochs=10. |
| shared-encoder | 0.4570                | 10 (bumped from 1)      | ok        | --epochs auto-bumped to --chemprop-min-epochs=10; Path C active. |

All 5 smoke runs completed with `status=ok` and emitted a valid
`training_summary.json`. AUC values are below the 0.7 honest-negative
gate; this is expected because (a) 1-epoch + 200-row cap is a
deliberate smoke budget and (b) the LOMO head-collapse behaviour
documented in `molmetal/reports/wf263a_audit/v3_drugood_root_cause.md`
predicts below-random AUC on a single held-out metal when the
encoder backbone is h=64/d=2 (production = 300/3). Smoke success is
defined as the script running cleanly through the
pre-featurize → train → eval → summary-emit pipeline.

## G4 — existing test coverage (29 tests across 3 files)

| Mode           | Test file                                       | Test names (sample)                                                                                                | Coverage status |
|----------------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|-----------------|
| v2             | test_wf263a_chemprop_v3_fix.py                  | test_chemprop_mode_v2_forces_v2_head, test_argparse_chemprop_mode_choices, test_help_with_chemprop_mode_flag        | covered         |
| v3-fallback    | test_wf263a_chemprop_v3_fix.py                  | test_default_v3_fallback_disables_enhanced_head, test_default_v3_fallback_disables_metal_attention, test_argparse_chemprop_mode_default_v3_fallback, test_help_with_chemprop_mode_flag | covered         |
| v3             | test_wf263a_chemprop_v3_fix.py                  | test_chemprop_mode_v3_honours_enhanced_head_true, test_chemprop_mode_v3_honours_metal_attention, test_user_cli_intent_preserved_after_dispatch | covered         |
| v3-fix         | test_wf266_v3_lomo_fix.py                       | test_min_epochs_default_10, test_lomo_init_scale_default_1e_3, test_v3_fix_mode_combines_all, test_min_epochs_blocks_premature_collapse | covered         |
| shared-encoder | test_wf269a_chemprop_path_c.py                  | test_argparse_chemprop_mode_includes_shared_encoder, test_argparse_metal_probe_init_scale_default, test_linear_probe_head_emit_shape, test_linear_probe_head_init_scale_zero_logit_at_init, test_linear_probe_head_init_scale_controls_weight_std, test_linear_probe_head_bias_zero_at_init, test_shared_encoder_model_uses_linear_probes, test_default_model_preserves_binary_classification_ffn, test_shared_encoder_forward_and_train_step, test_shared_encoder_dispatch_forces_v2_off, test_backward_compat_other_modes_unchanged, test_help_with_shared_encoder_flag | covered         |

Every mode is already covered by at least one dedicated test. The
combined mode-dispatch tests are exhaustive: `test_argparse_chemprop_mode_choices`
(wf-263A) iterates v2/v3/v3-fallback; `test_backward_compat_other_modes_unchanged`
(wf-269A) iterates v3-fallback/v2/v3/v3-fix; and
`test_argparse_chemprop_mode_includes_shared_encoder` (wf-269A)
verifies shared-encoder.

### Test run summary

```
$ pytest molmetal/tests/test_wf263a_chemprop_v3_fix.py \
          molmetal/tests/test_wf266_v3_lomo_fix.py \
          molmetal/tests/test_wf269a_chemprop_path_c.py -v
============================= 29 passed, 2 warnings in 3.49s ==============================
```

All 29 existing tests pass.

## G5 — new tests added

0. Every mode already has dedicated test coverage; no new tests are
needed.

## Honest framing note (per party-twenty-spirit §3.5)

* **Task command had stale flags**: the prescribed smoke command in
  the task spec invoked `--csv` / `--data-csv`, which are not exposed
  by the driver (data loads via `load_metal_cytotox()` resolver).
  The corrected command drops those flags.
* **Smoke AUC is below random on purpose**: 1-epoch + 200-row + h=64/d=2
  + LOMO single-metal is a deliberately cheap smoke budget. AUC below
  0.5 on a single seed does NOT contradict mode correctness; it
  matches the Wf-263A-Audit §1.2.1 root-cause #1 prediction
  (LOMO head-collapse on a held-out metal). Smoke success is defined
  as the script completing with `status=ok` and emitting a valid
  summary JSON.
* **`shared-encoder` IS implemented** (Wf-269A-Implement Path C
  shipped). The task spec hedged that it "may not exist if Wf-269A
  failed"; on the current branch it ships with the
  `_LinearProbeHead` class (`molmetal/scripts/train_chemprop_v2_v3.py:287-337`)
  and the mode dispatch (lines 1883-1908). Verified by
  `test_wf269a_chemprop_path_c.py::test_shared_encoder_forward_and_train_step`
  (CPU smoke through the Linear probe).
* **v3-fix and shared-encoder auto-bumped --epochs from 1 to 10**:
  the `--chemprop-min-epochs` floor (default 10) is enforced by the
  driver in both modes. Smoke runs reported
  `epochs=10` in the JSON even though the CLI passed `--epochs 1`.
  This is the expected Wf-266 / Wf-269A contract, not a smoke bug.
* **v3 mode requires --enhanced-head to swap the head**: a bare
  `--chemprop-mode v3` smoke (no `--enhanced-head`) is
  binary-identical to v3-fallback because both routes through the
  chemprop v2 BinaryClassificationFFN when the enhanced flag is
  False. The smoke for v3 above used `--enhanced-head` to actually
  exercise the v3 architectural lever.
* **No new tests added**: G6=0 because every mode was already
  covered by Wf-263A + Wf-266 + Wf-269A tests. G7=True because all
  29 existing tests pass (`29 passed in 3.49s`).
* **Production chemprop v2 is hidden_dim=300, depth=3**: the smoke
  budget h=64/d=2 matches the Wf-156 smoke recipe (5-7x cheaper per
  step) but caps the achievable AUC. This is intentional for the
  smoke budget; production retrain would lift the cap but was not
  requested for Wf-273C (smoke only).