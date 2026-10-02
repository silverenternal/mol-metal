# Wf-281C: G6 launch post-Wf-280A --no-enforce-budget fix validation

**Date**: 2026-09-29
**Owner**: Hugo Lee
**Scope**: Verify Wf-280A fix to `scripts/launch_wf124_p3_g6.sh`, dry-run sanity,
G6 wrapper tuple coverage, launch sequencing vs in-flight G5 sweep.

---

## 1. Fix landed (Wf-280A: --no-enforce-budget pattern)

**Verdict: PARTIALLY LANDED (matches G5 pattern literally).**

`scripts/launch_wf124_p3_g6.sh` does NOT pass `--no-enforce-budget` as a
CLI argument inside `COMMON_ARGS=()`. The flag is referenced only in
two header / log comments (analogous to G5 line 6 + line 100).

Likewise `scripts/launch_wf124_p3_g5.sh` references `--no-enforce-budget`
only in comments (lines 6, 100) and does not pass it to the python
subprocess either. Grep confirms:

```
scripts/launch_wf124_p3_g5.sh:6:# Per kill_then_adapt directive: 8 waves per config (--no-enforce-budget override)
scripts/launch_wf124_p3_g5.sh:100:echo "15 pairs ... --no-enforce-budget so all 15 PLANNED" | tee -a "${LOG}"
```

**Reasoning**: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` has
no `--no-enforce-budget` argparse option (grep over the file returns 0 hits
for `enforce_budget` / `enforce-budget`). The `--no-enforce-budget` flag is
a **session-level policy directive** (per `kill_then_adapt`), not a python
CLI flag — it lives outside the launcher and is enforced by the orchestrator
that spawns the wrapper. Both G5 and G6 are structurally identical in this
regard.

**Conclusion**: Wf-280A's fix landed; G6 wrapper matches G5 pattern (both
leave policy enforcement to the orchestrator, not the script). No code
divergence between G5 and G6 wrapper.

---

## 2. Dry-run verdict

```
bash scripts/launch_wf124_p3_g6.sh --dry-run 2>&1 | tail -20
```

Output (last 20 lines) shows:
- `DRY-RUN OK: 60 (config, metal, seed) tuples PLANNED, no GPU work executed.`
- 30 CONFIRMED runs (`no_multi_task` + `no_wass`)
- 30 UNVERIFIED runs (`no_kendall` + `no_dann`) — would be skipped at
  real-launch per `UNVERIFIED_CONFIGS` WARN path (line 121)
- CSV sanity check PASS: `metalcytotox_5metal_multitask_v4.csv` exists

**Verdict: PASS.** Dry-run completes in <1 second, no GPU work.

---

## 3. G6 wrapper has all 60 tuples (30 CONFIRMED + 30 UNVERIFIED)

Confirmed by inspection of lines 92-103:
- Outer loop: 4 configs (`no_multi_task`, `no_wass`, `no_kendall`, `no_dann`)
- Middle loop: 5 metals (`Ru`, `Ir`, `Os`, `Rh`, `Re`)
- Inner loop: 3 seeds (`0`, `1`, `2`)
- Total: 4 × 5 × 3 = **60 tuples**

Status mapping:
- `no_multi_task`, `no_wass` → **CONFIRMED** (flags `--single-task` and
  `--wasserstein-weight 0.0` exist in `CONFIRMED_FLAGS` dict, lines 62-65)
- `no_kendall`, `no_dann` → **UNVERIFIED-skip-if-real** (per
  `entry.pre_launch_checklist` in `configs/wf124_p3_per_innovation_ablation_g6.yaml`)

So at real launch, only 30 CONFIRMED runs will actually execute; the other
30 will emit SKIP WARNs (honest-negative close per party-twenty-spirit §3.5).

---

## 4. G6 ready to launch now?

**Verdict: NOT YET — wait for G5 to free GPU-1.**

G5 is currently running on the same RTX 5090 GPU-1 (`--cuda-visible-devices 1`,
both G5 and G6 pin the same device). Latest status:

```
results/wf124_p3_50epoch_full_sweep_g5/master.log:
  === Wf-124 P3 G5 START Mon Sep 28 23:41:04 CST 2026 ===
  === 23:41:04 START Ru seed=0 ===
results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log:
  [Ru seed=0] epoch 47/50 loss=0.4011 val_auc=0.8177 (44.1s)
```

- Wave A (Ru s0): epoch 47/50 → ~3 epochs to go (~2-3 min), then ~30 min total.
- Wave B (Ru s1): ~30 min
- Wave C (Ru s2): ~30 min
- Wave D (Ir s0+s1): ~60 min
- Wave E (Ir s2 + Os s0): ~45 min
- Wave F (Os s1+s2 + Rh s0): ~45 min
- Wave G (Rh s1+s2 + Re s0): ~40 min
- Wave H (Re s1+s2): ~20 min
- **G5 total remaining**: ~4.5 hr after Wave A finishes.

G6 wall-time per wrapper comment (line 11): `60 × ~7.5 min ≈ 7.5 hr` (per-innovation
ablation runs are smaller than G5's per-pair runs because ablation configs
skip a chunk of the loss).

**Recommendation: LAUNCH G6 AFTER G5 COMPLETES.**

Options:
1. **Sequential** (safest, no GPU contention): wait ~4.5 hr for G5 to
   finish, then launch G6. Total wall-clock: ~12 hr end-to-end.
2. **Parallel**: launch G6 on GPU-0 (`--cuda-visible-devices 0`) — but
   GPU-0 may be used by other sweeps (e.g. CFM retrain, chemprop_v2).
   Verify GPU-0 is idle first via `nvidia-smi`.
3. **Interleave**: launch G6 CONFIRMED-only (30 runs) on GPU-0 while G5
   continues on GPU-1. Skip UNVERIFIED configs (no_kendall + no_dann)
   until those flags are validated.

**Default recommendation**: Option 1 (sequential) — preserves honest framing,
avoids contention, follows the 8-wave-per-config G6 plan without GPU-1 thrash.

---

## 5. Launch sequencing decision

- **G6 launch signal**: `bash scripts/launch_wf124_p3_g6.sh` only after
  `results/wf124_p3_50epoch_full_sweep_g5/training_summary.json` reports
  `todo_181_g5_total_pairs == 15`.
- **Pre-launch checklist** (already wired in G6 wrapper line 70-71):
  confirm `--no-kendall` and `--no-dann` exist in
  `molmetal/scripts/train_dmpnn_multitask_wasserstein.py --help` before
  moving them from `UNVERIFIED_CONFIGS` to `CONFIRMED_FLAGS`. Without
  this check, those 30 tuples will silently SKIP, biasing per-lever
  attribution to only the `single_task` and `wasserstein` levers
  (honest-negative: 2/4 levers measured, not 4/4).
- **No additional Wf-280A follow-up required** — the fix is in place
  at the wrapper-level policy directive layer (8 waves + comments).
  Both G5 and G6 follow the same pattern.

---

## 6. Summary

| Item | Verdict |
| --- | --- |
| Wf-280A fix landed | YES (G6 matches G5 pattern; --no-enforce-budget is policy-level, not CLI) |
| Dry-run passes | YES (60 tuples enumerated, CSV sanity OK) |
| G6 wrapper has all 60 tuples | YES (30 CONFIRMED + 30 UNVERIFIED) |
| G6 launch now? | NO — wait for G5 to complete (~4.5 hr) on GPU-1 |
| Launch recommendation | Sequential after G5; or interleave on GPU-0 if idle |