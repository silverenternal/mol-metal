# Wf-276B — G6 Launch Orchestration Plan (post-G5)

Date: 2026-09-29
Owner: Hugo Lee + Claude
Parent: TODO-182 G6 (Wf-124 P3 per-innovation ablation)

---

## 1. Current ground-truth state (MEASURED 2026-09-29 00:18 CST)

| item | value | source |
|---|---|---|
| G5 wrapper PID | 609543 | `ps aux` (RTX 5090 GPU-1, 1810 MiB) |
| G5 phase | Ru seed=0, ~epoch 36/50, val_auc=0.7621 | task context + task capture |
| G5 remaining | ~12 hr GPU-1 wall (14 pairs × ~50 min) | Wf-272D MEASURED 63.2 s/epoch × 50 × 15/3600 = 13.17 hr; 1 pair already in flight, ~12.5 hr remaining |
| GPU-0 (Blackwell RTX PRO 6000) | **idle**, 4 MiB, 0% util | `nvidia-smi` |
| GPU-1 (RTX 5090) | busy w/ G5 PID 609543 | `nvidia-smi` |

## 2. G6 readiness verification (per task items 4-6)

### 2.1 cuda:0 fix (Wf-266) — PASS

`scripts/launch_wf124_p3_g6.sh:33` sets `GPU="cuda:0"` and line 126 invokes with
`CUDA_VISIBLE_DEVICES=1 ${PY} ${SCRIPT} ... --device cuda:0`. Matches the
Wf-131/Wf-266/272D cuda:0 trick (env exposes GPU-1 as cuda:0). G6 wrapper is
cuda:0-fix-clean.

### 2.2 v4 CSV reference (Wf-263C) — PASS

`scripts/launch_wf124_p3_g6.sh:34` →
`/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv`
Same path as G5. `configs/wf124_p3_per_innovation_ablation_g6.yaml:27` also
references v4. No deviation.

### 2.3 --no-enforce-budget flag — **MISSING from wrapper** (honest-negative)

`wave_budget_tracker.py:763` supports `--no-enforce-budget` per user 2026-09-28
"时间预算没关系的" standing directive. **G6 wrapper does NOT pass this flag** —
it must be added at invocation time:

```
python molmetal/scripts/wave_budget_tracker.py \
    --budget-hr 13 --no-enforce-budget \
    --report-json molmetal/reports/wf262_wave_budget/g6_pre_launch.json
```

This is a passive projection only (per `wave_budget_tracker.py:81` honest framing);
not a launch gate. Pre-launch emission should still happen for the
orchestration log.

### 2.4 Configs 3 + 4 (--no-kendall / --no-dann) flag verification — **NOT YET DONE**

`launch_wf124_p3_g6.sh:69-71` declares `VERIFIED_CONFIGS="no_multi_task no_wass"`
and `UNVERIFIED_CONFIGS="no_kendall no_dann"`. Per `configs/...g6.yaml:103,113`:
"verify-flag-before-launch" status. Wrapper will **SKIP** unverified tuples
(line 121 WARN). Honest-negative close: only 30 CONFIRMED runs (no_multi_task +
no_wass × 5 metals × 3 seeds) will execute unless user upgrades
`UNVERIFIED_CONFIGS` after flag check.

## 3. Wrapper semantics — kill_then_adapt is NOT auto-advance (honest correction)

The task brief says "G5 wrapper reads master.log when a pair finishes (looks
for `test_auc=` line at end)" and "once Ru seed=0 test_auc is emitted, wrapper
automatically moves to next". **This is INCORRECT.**

Both `launch_wf124_p3_g5.sh` (PID 609543) and `launch_wf124_p3_g6.sh` are plain
**sequential bash for-loops**. There is no master.log polling, no
auto-advance. The single bash process runs pairs/tuples serially via
`run_pair` / `run_tuple` functions. The "kill_then_adapt" directive refers to
the **budget discipline** (--no-enforce-budget), NOT inter-pair auto-launch.

Implication for orchestration: G5 PID 609543 will continue to Ru seed=1, s2,
Ir s0, ..., Re s2 uninterrupted. There is no intermediate hook point between
pairs to insert "now launch G6". We must either:

(a) Wait for G5 wrapper PID 609543 to exit fully (~12 hr from now),
then launch G6.

(b) Launch G6 **in parallel** on GPU-0 (Blackwell, idle), via a modified
wrapper that uses `CUDA_VISIBLE_DEVICES=0` + `--device cuda:0` (the Blackwell
ordinal trick: GPU-0 maps to cuda:0 natively, no env trick needed).

(c) Pre-flight `--dry-run` G6 now (CPU verify, no GPU) so we have
verification + budget projection ready the moment G5 exits.

## 4. GPU strategy options

| option | wall-time saving | risk | GPU load | user action |
|---|---|---|---|---|
| **A. Serialize on GPU-1** | none (12.5 hr wait + G6 ≈ 15-25 hr) | none | GPU-1 only | queue G6 with `nohup` after G5 exit |
| **B. Parallel on GPU-0 (Blackwell)** | **saves ~12 hr** | sm_120 + D-MPNN untested on Blackwell; would need smoke test first | GPU-0 + GPU-1 | modify wrapper to `CUDA_VISIBLE_DEVICES=0` + run 3-metal smoke on GPU-0 first |
| **C. Hybrid: pre-flight G6 dry-run now; queue G6 serial after G5; reserve GPU-0 for CFM TODO-119 P1.3 follow-up** | dry-run = 0 GPU hr; full G6 = same as A | none | GPU-1 only | submit G6 with `--dry-run` now + add `wait-for-g5-exit-then-launch` cron hook |

## 5. Recommendation

**Recommend Option C** (hybrid dry-run + serial queue + reserve GPU-0 for CFM):

1. **Now** (00:18 CST, 12 min after capture): run
   `bash scripts/launch_wf124_p3_g6.sh --dry-run` to emit the 60-tuple
   enumeration + flag-verification status into `master.log`. Zero GPU cost,
   confirms CUDA path + v4 CSV reachability while G5 keeps running.
2. **Now**: emit wave-budget report for G6 via
   `python molmetal/scripts/wave_budget_tracker.py --no-enforce-budget
   --report-json molmetal/reports/wf262_wave_budget/g6_pre_launch.json` so the
   G6 budget projection is captured against the MEASURED G5 actuals
   (TODO_181_G5_ACTUAL_HR=13.17 hr).
3. **After G5 wrapper PID 609543 exits** (~12 hr): launch G6 real-run via
   `nohup bash scripts/launch_wf124_p3_g6.sh > g6_nohup.out 2>&1 &`. Expected
   wall-time on RTX 5090 GPU-1 for the 30 CONFIRMED runs (--single-task +
   --wasserstein-weight 0.0 ablations; smaller multi-task overhead ⇒ estimate
   ~30-40 min/pair × 30 ≈ **15-20 hr**, vs Wf-258 PROJECTED 7.5 hr — gap is the
   honest Wf-258 underestimate, same pattern as G5's 13.17 vs 5.0).
4. **GPU-0 (Blackwell) is reserved** for the next-priority item once the
   unverified flags (--no-kendall / --no-dann) are confirmed: that means a
   small (3-metal) Blackwell smoke first (Wf-269A pattern), then CFM
   TODO-119 P1.3 follow-up (Wf-258 §1.1) since GPU-0 has 96 GB vs GPU-1's
   32 GB and the 16 hr PROJECTED for CFM decode-lift benefits from the bigger
   memory pool.

## 6. Honest framing (party-twenty-spirit §3.5)

* G5 + G6 do **NOT** serialize automatically on GPU-1 — the wrappers are
  sequential bash, no master.log polling, no auto-advance. (Task premise
  corrected.)
* G6 wrapper will SKIP 30/60 tuples (--no-kendall + --no-dann) unless
  pre-launch verification adds those flags. Only `no_multi_task` + `no_wass`
  (30 CONFIRMED runs) will execute.
* Wf-258 PROJECTED 7.5 hr for G6 is likely 2-3× low based on the Wf-272D
  G5 actual (PROJECTED 5.0 hr → MEASURED 13.17 hr). Plan for ~15-20 hr
  serial on RTX 5090.
* Per-lever attribution % is PROJECTED, NOT MEASURED, until all 30 CONFIRMED
  runs complete. Per TODO-182 §6: PROJECTED attribution is "multi-task
  60-80%, W2 10-30%, L2 guard ~5%"; interaction term may be >30%. Do NOT
  cite as MEASURED.

## 7. Pre-launch checklist (per `configs/...g6.yaml:256-259`)

- [ ] Verify `--no-kendall` flag exists in
      `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (or set fallback
      to uniform Kendall weights)
- [ ] Verify `--no-dann` flag exists (or fallback to `--dann-domain-mode=off`)
- [ ] Confirm `--single-task` replaces `--no-multi-task` (already CONFIRMED
      per launch_wf124_p3_g6.sh:63)
- [ ] Emit `--no-enforce-budget` wave-budget report before real launch
- [ ] Confirm G5 wrapper PID 609543 has exited before launching G6 real-run

---

End of plan.
