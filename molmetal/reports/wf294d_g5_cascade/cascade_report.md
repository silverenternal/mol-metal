# Wf-294D — G5 epoch-50 G4 Cascade Readiness Report

**Date**: 2026-09-29 00:55:44 CST
**Author**: Wf-294D agent
**Trigger**: Wf-289A + Wf-283A — G5 Ru seed=0 emit test_auc=0.7716 (50 epochs)

---

## §1 current_status (verified at 2026-09-29 00:55:44 CST)

### G5 Ru seed=1 progress

- **PID 1015907** — alive, etime 26:26 (1586 s)
- **cmd**: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metals Ru --seeds 1 ... --epochs 50 --device cuda:0 --cuda-visible-devices 1`
- **Latest logged epoch**: 26 / 50, loss=0.6981 val_auc=0.7155 val_pearson=0.274 (60.8 s)
- **Epoch times last 5**: 48.9, 39.7, 39.8, 60.8, 66.1 → mean 51.1 s/epoch
- **Overall mean so far**: 1586 s / 26 epochs = 61.0 s/epoch
- **Ru seed=0 reference**: DONE in 2850 s total (47.5 min), test_auc = 0.7716 (best_epoch=47, val_auc=0.8177)
- **G5 master log**: only Ru seed=0/1 launched (master.log shows `=== 23:41:04 START Ru seed=0 ===`, `=== 00:28:52 END Ru seed=0 ===`, `=== 00:28:52 START Ru seed=1 ===`); waves 2-15 not yet enqueued
- **G4 training_summary.json**: n_pairs_total = 15, expected_pairs = 14 (off-by-one in script default — needs update or note), aggregate_5metal_test_auc_mean = 0.7243

### G5 Ru seed=1 directory state

```
results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/
├── ckpts/             (empty so far)
├── ru/                (empty so far — per_seed_metrics_seed1.json not yet emitted)
└── run.log            (3265 bytes, 26 epoch lines)
```

### G5 Ru seed=0 directory state

```
results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/
├── ckpts/
├── ru/
│   ├── per_seed_metrics.json
│   └── per_seed_metrics_seed0.json   ← already emitted (DONE)
├── run.log
└── training_summary.json
```

---

## §2 expected_completion

**Estimate when G5 Ru seed=1 hits epoch 50**:

| Method | Calculation | ETA from 00:55:44 CST |
|---|---|---|
| Best-case (last-5 avg 51.1 s/ep) | 24 epochs × 51.1 s = 1226 s | ~01:16:10 CST |
| Mid-case (overall mean 61.0 s/ep) | 24 epochs × 61.0 s = 1464 s | ~01:20:08 CST |
| Worst-case (Ru seed=0 wall-clock 35.6 s/ep amortized over 50) | 24 × 35.6 = 854 s | ~01:10:00 |

**Best mid-range estimate**: epoch 50 around **01:15 – 01:20 CST** (≈ 20–25 min from now).

After Ru seed=1 finishes, master.log will queue the next 13 pairs (Ru seed=2 + Ir/Os/Rh/Re × 3 seeds); ETA for full G5 cohort completion ≈ 4-5 hr from now, but the **n=14→15 cascade closes at the first pair's completion**, not the cohort's.

---

## §3 cascade_ready

### §3.1 wf278a_g4_aggregate.py inventory

- **Location**: `molmetal/scripts/wf278a_g4_aggregate.py` (276 LOC, ready)
- **Inputs**: nested path glob `{root}/**/per_seed_metrics_seed*.json`
- **Outputs**: `{output}/training_summary.json` with per-metal n_seeds + test_auc mean ± std + 5-metal aggregate
- **De-dup logic**: keeps most-specific (deepest) match per seed number; bare `per_seed_metrics.json` aggregator wrapper is correctly filtered out (line 49)
- **Validation**: filters `status != "OK"` (line 78) and emits payload/path mismatch WARN

### §3.2 Cascade wiring verification

The cascade intent is: **G4 (30-epoch, 14 OK pairs) + G5 (50-epoch, Ru seed=0 DONE) = 15 OK pairs at canonical path → n_pairs_total 14→15 → TODO-180 close**.

**Status: PARTIALLY READY** — three blockers observed:

1. **`expected_pairs: 14` mismatch**: `wf278a_g4_aggregate.py:216` hard-codes `expected_pairs: 14` but currently reports `n_pairs_total: 15`. This is because the script already collected Ru seed=0 from G4's `A/Ru/seed0/ru/per_seed_metrics_seed0.json` (30-epoch). For an honest cascade, either:
   - Update script to scan both G4 + G5 directories and only count *one* Ru seed=0 (de-dup by `(metal, seed)` pair across roots), OR
   - Add a `--exclude-pair` flag to drop the G4 30-epoch Ru seed=0 when ingesting G5 50-epoch Ru seed=0.

2. **Cross-directory cascade not yet wired**: `wf278a_g4_aggregate.py` accepts a single `--input-dir`. To merge G4 (30-epoch) + G5 (50-epoch) for the 14→15 cascade, either:
   - Symlink/copy `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics_seed0.json` → `results/wf124_p3_remaining_metals_g4/C/Ru/seed0/ru/per_seed_metrics_seed0.json` to overwrite the 30-epoch version (NOT RECOMMENDED — loses 30-epoch audit trail), OR
   - Add `--input-dirs G4,G5` multi-root support to the aggregator (PREFERRED), OR
   - Run aggregator twice and merge JSON (workaround).

3. **Ru seed=1 not yet emitted**: `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/ru/per_seed_metrics_seed1.json` does NOT exist yet. ETA: §2 above.

### §3.3 Cascade trigger conditions (when to fire)

Trigger wf278a aggregation cascade **IMMEDIATELY upon** all three:

- [ ] `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/ru/per_seed_metrics_seed1.json` exists (eta §2)
- [ ] Process PID 1015907 has exited (cascade should not race with a half-written JSON)
- [ ] master.log shows `=== END Ru seed=1 ===` marker (the trainer prints this on clean exit)

Pre-cascade sanity:

```bash
test -s results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/ru/per_seed_metrics_seed1.json
ps -p 1015907  # expect "No such process"
grep -c "END Ru seed=1" results/wf124_p3_50epoch_full_sweep_g5/master.log
```

---

## §4 action_items

### Immediate (next 5 min)

- [ ] Decide cascade wiring strategy: **(A)** symlink/copy overwrite, **(B)** add multi-root `--input-dirs` to wf278a aggregator, or **(C)** run twice + JSON merge. **Recommended: B (most auditable, preserves 30-epoch artifact)**.
- [ ] If (A): schedule the symlink after Ru seed=1 emits its file (ETA §2).
- [ ] If (B): open a 30-LOC patch to `wf278a_g4_aggregate.py` adding multi-root glob + last-wins-by-mtime de-dup; verify on dry-run before Ru seed=1 ETA.
- [ ] If (C): write a 15-LOC merge script (`wf294d_cascade_merge.py`) that walks both roots, de-dups by (metal, seed) preferring G5 (higher epoch-count), and emits a single training_summary.json.

### After Ru seed=1 epoch 50 (eta §2)

- [ ] Verify `per_seed_metrics_seed1.json` exists, has `status: "OK"`, and `test_metrics.roc_auc` is finite.
- [ ] Verify ckpt file `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/ckpts/dmpnn_mtw_ru_seed1.pt` exists.
- [ ] Verify W₂ trajectory in `wasserstein_history` is non-increasing on average (JKO-style per TODO-181 §3.2).
- [ ] Verify L2 guard grad-norm bounded (no clip-trip warnings in run.log).
- [ ] Fire cascade: run aggregator with merged roots → write `results/wf124_p3_50epoch_full_sweep_g5/cascade_training_summary.json`.
- [ ] Cross-check: `n_pairs_total` = 15 (after Ru seed=1 completes), aggregate 5-metal test_auc mean compared against G4 30-epoch aggregate (0.7243) for honest lift/no-lift framing.

### TODO-180 close

- [ ] After cascade emits summary with n=15 and 50-epoch G5 baseline, mark `[G4.1]–[G4.6]` in `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` §4 acceptance gate.
- [ ] If 50-epoch ≥ 30-epoch lift > 0.01 absolute: paper §4 LEAD-claim candidate; else §6 honest-negative entry.
- [ ] Move `TODO/pending/180_…` → `TODO/completed/180_…`.

### Risk callouts

- **Race condition**: do NOT run cascade while PID 1015907 is alive — JSON may be partially written. Use ps + grep master.log trigger.
- **Epoch-count mixing**: G4 is 30-epoch, G5 is 50-epoch. Cascade must clearly label which epoch-count the 15-pair aggregate refers to. If mixing without disclosure = `party-twenty-spirit` §3.5 violation.
- **wf278a `expected_pairs: 14` is stale** — update to 15 in next pass or add comment noting G4 had 14 + G5 brought it to 15.

---

**Honest framing per `party-twenty-spirit` §3.5**: cascade is READY but with one orchestration gap (single-root glob) and one data gap (Ru seed=1 not yet emitted). ETA for cascade-firing completion: ~25 min from this report.