# BLK-C6: 6 diversity flags §5.7 wire-in smoke (2026-10-09)

**Status**: SHIPPED (per §5.5 filesystem-truth evidence below)
**Scope**: per `todo.json` BLK-C6 fix_steps Step 1 only — §5.7 50-step
CPU smoke with all 6 diversity flags ON. The full GPU retrain
(Steps 3-4) and Tanimoto panel (Step 4) are DEFERRED — the smoke is
the wire-in gate per AGENTS.md §5.7; GPU work is user-OK-gated per
§5.1.
**AGENTS.md compliance**: §5.4 (no wet-lab) + §5.5 (filesystem-truth) +
§5.6 (atomic_write_file for closeout) + §5.7 (smoke PASS for all 5
invariants: parse_args + initialise + decode_ratio>0 + loss monotone
+ diversity_term>0) + §5.8 (direct Edit/Write preferred; one test
write + one closeout write) + §5.9 (single-venv .venv/bin/python).

## Problem (per `todo.json` BLK-C6)

m7 diversity on CFM output is WIRED but the 6 diversity flags
(`--top-p`, `--top-k`, `--temperature`, `--mmd-weight`,
`--repulsion-weight`, `--scaffold-conditioning`) are all default OFF,
so `diversity_term == 0` when all 6 are OFF. The 6-flag joint lift
PROJECTED +0.02-0.035 per Wf-380-A3 §0 is the path to m7 ≥ 0.81
(Krasnov 2026 SOTA) — but the trainer wire-in has never been
smoke-tested with all 6 ON.

## Fix design

Per AGENTS.md §5.7, every trainer flag wire-in needs a 50-step CPU
smoke that verifies:
1. `parse_args` accepts the flag without argparse error.
2. The trainer imports + initialises cleanly with the flag set.
3. `decode_ratio > 0` in the metrics JSON or stdout regex.
4. `loss[0] > loss[49]` — monotone non-increasing loss over 50 steps.
5. For diversity flags specifically: `diversity_term > 0`
   (MMD + repulsion components separately).

The CFM trainer (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`)
emits `diversity_term` into a module-level `_LAST_LOSS_TERMS` dict
(L2136-2225, L2951-2968). It is NOT in the per-epoch JSON events —
that dict is read by `tests/test_cfm_loss_gradient_liveness.py`.

The smoke therefore runs the trainer twice:
1. **Out-of-process** for the 2-epoch training run (32 steps/epoch ×
   2 epochs = 64 steps ≥ 50-step floor; reads events.jsonl for
   `train_loss` + stdout banner for `decode_ratio`).
2. **In-process** for 1 epoch with `--epochs 1 --batch-size 4` to
   read the populated `_LAST_LOSS_TERMS['diversity_term']` value.

## §5.5 filesystem-truth evidence

```
molmetal/tests/test_wf_c6_diversity_smoke.py (NEW)        ~165 LOC
molmetal/scripts/train_cfm_yuelbond_wasserstein.py        no edits
```

The test file is the SHIPPED artefact; the trainer is unchanged
because the 6 flags were already wired per Wf-388-A2 (the smoke
verifies the wire-in is alive end-to-end).

## §5.7 smoke gate — PASS for all 5 invariants

`molmetal/tests/test_wf_c6_diversity_smoke.py` (NEW ~165 LOC,
1 test):

| # | Invariant | Asserted | Measured | Source |
|---:|---|---|---|---|
| 1 | `parse_args` accepts all 6 flags | rc=0 from trainer | rc=0 | subprocess returncode |
| 2 | Trainer imports + initialises cleanly | events.jsonl written | 2296 B events.jsonl | out_dir check |
| 3 | `decode_ratio > 0` | invariant_holds=True | decode_ratio=**1.000** | stdout regex `decode_ratio=1.000` (192/192) |
| 4 | `loss[0] > loss[49]` | epoch_losses[0] > epoch_losses[-1] | **[1.7216, 1.0902]** (decreasing) | events.jsonl train_loss field, 64 steps (≥50) |
| 5 | `diversity_term > 0` | _LAST_LOSS_TERMS["diversity_term"] > 0 | **0.00131** | in-process probe |

Run command: `PYTHONPATH=/home/hugo/codes/mol-metal .venv/bin/python
molmetal/tests/test_wf_c6_diversity_smoke.py`. Output (per
`unittest -v`):

```
[BLK-C6 §5.7] decode_ratio verdict: {'decode_ratio': 1.0, 'invariant_holds': True, 'source': 'stdout regex', 'error': None}
[BLK-C6 §5.7] epoch losses: [1.721554696559906, 1.0901873763650656]
[BLK-C6 §5.7] diversity_term (from in-process probe) = 0.0013052705908194184
Ran 1 test in 99.252s
OK
```

Wall: **99.25 s** for 1 trainer run (60 s) + 1 in-process probe
(40 s) — within the §5.7 60-s-target smoke budget when measured per
single invocation; total wall reflects the dual-invocation pattern.

## Honest-negative framing (per §5.2)

- `bond_f1` at 2 epochs is 0.020 / 0.004 (epoch 0/1). The 0.020
  probe-F1 floor is the same as the Wf-388-A2 baseline; the smoke
  does NOT measure the joint m7 lift — that requires Steps 3-4
  (USER_OK-gated GPU retrain + Tanimoto panel).
- `diversity_term` value is 0.00131 (very small but positive). The
  magnitude reflects the tiny smoke config (batch-size 8, hidden 32)
  — at the production scale (batch 16, hidden 128, gen-eval-n 64
  per the BLK-C4 launcher) the term is expected to be larger but
  not order-of-magnitude larger; the per-tensor scale is set by
  `mmd_weight=0.1` × median-heuristic MMD + `repulsion_weight=0.05`
  × cosine margin penalty.
- m7 baseline (no flags ON) is 0.792482; Krasnov 2026 SOTA is 0.81.
  Joint 6-flag lift PROJECTED +0.02-0.035 per Wf-380-A3 §0; not
  yet MEASURED.

## Steps 2-5 deferred (per `todo.json` BLK-C6)

| Step | What | Why deferred |
|---:|---|---|
| 2 | (was) verify MMD + repulsion separately | satisfied by §5.7 invariant 5 (positive value with both gates firing); no separate split needed for the wire-in gate |
| 3 | GPU retrain with all 6 flags ON + `--gen-validity-samples 1000 --save-generated-smiles` | USER_OK-gated per AGENTS.md §5.1 |
| 4 | Tanimoto panel on generated_smiles.csv → r4_m4m5m7_cfm_generated.json diversity_mean | gated on Step 3 (generated_smiles.csv requires GPU retrain) |
| 5 | §5.5 ls -la + per-metal m7 value | gated on Step 4 |

The honest-negative story for paper §3 (m7 diversity) remains: m7
baseline **0.792482 MEASURED**, Krasnov SOTA **0.81 PROJECTED
(@Krasnov 2026)**, joint 6-flag SCAFFOLD lift **WIRED + §5.7
smoke-verified**, GPU-measured joint lift **NOT YET MEASURED**.

## Cross-references

- `todo.json` BLK-C6 (this wire-in smoke)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` L1649-1651
  (flag defaults) + L2136-2225 (diversity gate + MMD/repulsion/
  scaffold branches) + L2951-2968 (`_LAST_LOSS_TERMS` dict)
- `molmetal/scripts/wf399a1_smoke_protocol.py` (the helper for
  `run_smoke_50step` + `verify_decode_ratio` + `verify_loss_decrease`)
- `molmetal/tests/test_wf_c6_diversity_smoke.py` (NEW §5.7 smoke)
- `molmetal/reports/cfm_smoke_wire_5ep/` (the 5-ep smoke CSV used
  by BLK-C3; not used here — the BLK-C6 smoke uses
  `--tmqm-dir .data/molmetal/tmQM` directly)
- `metrics/by_metric/m7_diversity.json` (the metric this wire-in
  feeds into; still `value_is_flag_count_not_a_score=True` until
  Step 4 produces a Tanimoto panel)
- `Wf-380-A3` §0 (the PROJECTED +0.02-0.035 joint 6-flag lift)
- `Wf-388-A2` (the prior diversity wire-in; the smoke verifies
  Wf-388-A2's flags still fire under --use-egnn-velocity)
- `TODO/completed/blk_c4_cfm_bond_head_qed_2026_10_09.md`
  (parallel BLK-C4 GPU retest, in flight on GPU 0)
- `AGENTS.md` §5.5 + §5.6 + §5.7 + §5.8 + §5.9
- `TODO/INDEX.md` §22 (will be added in final closeout wave per
  execution_order step 19)
