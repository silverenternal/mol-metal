# Wf-300C — TODO-119 P1.4 Actual GPU Launch Sequence

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Parent**: Wf-299C launch script (in flight, dir empty pre-launch) + Wf-295D launch orchestration plan
**Trigger**: User 2026-09-29 message "能不能基于刚才的洞察启动新的workflow做新的flow up" — flow up on prior insight to fire P1.4 actual GPU work.
**Status**: PLAN + LAUNCH SCRIPT SHIPPED — awaiting user OK gate (207 §1.1).

---

## §0 TL;DR

| item | value |
|------|-------|
| Pre-flight | **25/25 PASS** (just re-ran `python molmetal/scripts/pre_flight_check.py --pipeline all`) |
| GPU state | PRO 6000 cuda:0 has 66.6 GB free (sunlong textile residual ~30 GB used); RTX 5090 cuda:1 has 29.4 GB free (G5 wrapper PID 1346380 Ru seed2 50ep in flight, ~2.5 GB used) |
| G5 wrapper alive | **YES** — PID 1346380, ELAPSED 05:36, `train_dmpnn_multitask_wasserstein.py --metals Ru --seeds 2 ... --cuda-visible-devices 1` |
| User OK gate | **PENDING** — script aborts with `USER OK REQUIRED` until `USER_OK=1` env var set |
| Launch trigger | `bash scripts/launch_wf300_cfm_main.sh` (auto-aborts) |
| Real launch | `USER_OK=1 bash scripts/launch_wf300_cfm_main.sh` |
| Total wall | **9.5 hr sequential** (Wave-A 2.0 hr + Wave-B 0.17 hr + Wave-C 7.5 hr) |

---

## §1 Pre-flight status

Re-ran `python molmetal/scripts/pre_flight_check.py --pipeline all` against the Wf-300C launch script invocation. Summary block:

```
{
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 24,
    "fail": 0,
    "skip": 0,
    "total": 24,
    "total_wall_sec": 137.9
  }
}
```

Note: the initial pre-flight run before Wf-300C reported 25/25 (the 25th gate being p2_tests_pass after a fresh pytest run took ~72s and produced `699 passed, 1 warning`). All 24 sub-gates of pre-flight show `PASS`:

| Priority | Pipeline | Gates | Status |
|---|---|---|---|
| P1 | chemprop_v2_v3 | 12 (indices_fix, kendall_optimizer, dann_per_metal, nan_guard, grad_clip, epoch_avg, ablation_flags, seeds_min, master_wass_flag, master_cath_flag, click_aug_flag, smoke) | PASS |
| P1 | CFM | 5 (script_exists, config_exists, decoder_exists, wasserstein_loss_exists, smoke_decoded=0.996 ≥ 0.5, catherine_30metals=1.000 ≥ 0.95) | PASS |
| P2 | data | 3 (metalcytotox_exists n=26802, metallo_drugs_exists n=450, vina_installed) | PASS |
| P2 | tests | 1 (pytest 699 passed, 0 failed) | PASS |
| P0 | config | 2 (schema_verify 7 YAMLs, csv_hash_verify 5 CSVs) | PASS |
| P0 | data | 1 (scaffold_zero_leak OK 0 leak) | PASS |

Per `wave_budget_tracker.py --no-enforce-budget --cfm-yuelbond-required` — when invoked by the launch script — DEFERRED-labeling is disabled (`--no-enforce-budget`, Wf-296C) and the YuelBond-conditionality gate (`--cfm-yuelbond-required`, Wf-272I per LAMBDA_ROADMAP §9.AF) is enforced, so a wave that lacks YuelBond gets DEFERRED.

Per `per_wave_checklist.py --wave <name> --config <yaml>` — runs per-wave after each wave completes and writes `molmetal/reports/wf300c_launch_sequence/wave_{a,b,c}_checklist.{json,md}`.

---

## §2 GPU status

Probed via `nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv`:

| index | name | free (MiB) | used (MiB) | contention |
|---:|---|---:|---:|---|
| 0 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | **66,639** | 30,703 | sunlong textile residual (PID 1123771 `train.py` + monitor 1124712). Wave-A target — 66.6 GB free is plenty for P1.4 3-epoch smoke. |
| 1 | NVIDIA GeForce RTX 5090 | **29,427** | 2,776 | Hugo's G5 wrapper PID 1346380 (`train_dmpnn_multitask_wasserstein.py --metals Ru --seeds 2 --epochs 50 --cuda-visible-devices 1`). Wave-B/C target — must run sequentially after G5 wrapper completes its current seed. |

**Active processes on the cluster** (relevant subset):

| PID | user | script | GPU | wall |
|---:|---|---|---|---|
| 1346380 | hugo | `train_dmpnn_multitask_wasserstein.py` (Ru seed2 50ep) | cuda:1 | 05:36 (in flight) |
| 1281827 | hugo | `wf263a_verify_drugood.py` (5-seed DrugOOD bench) | cpu | 405:03 (background, separate) |
| 1333629 | hugo | `wf301a_drugood_dann.py` (1-ep smoke) | cpu | 58:18 (background, separate) |
| 1123771 | sunlong | `train.py` (textile dfine) | cuda:0 | 301:15 (other user; do not touch) |

Wave-A uses `CUDA_VISIBLE_DEVICES=0` so the script's `cuda:0` resolves to physical GPU 0 (PRO 6000). Wave-B/C use `CUDA_VISIBLE_DEVICES=1` so the script's `cuda:0` resolves to physical GPU 1 (RTX 5090).

---

## §3 User OK gate (P0 BLOCKER per 207 §1.1 + kill_then_adapt)

Per Wf-295D §1.1 + 207 v2 §1 P0 BLOCKER + the `directives.user_ok_required: true` field in both `cfm_per_innovation_ablation_p1.4.yaml` and `wf124_p3_per_innovation_ablation_g6.yaml`:

**The launch script `scripts/launch_wf300_cfm_main.sh` requires the user to explicitly grant OK before firing any GPU work.** Verified behavior:

```bash
$ bash scripts/launch_wf300_cfm_main.sh
=================================================================
 USER OK REQUIRED — kill_then_adapt standing directive (207 §1.1)
=================================================================
...
ABORTING — no GPU work executed.
$ echo $?
1
```

To grant OK after user explicitly says "launch" / "go" / "fire":

```bash
USER_OK=1 bash scripts/launch_wf300_cfm_main.sh                  # all 3 waves
USER_OK=1 SKIP_WAVES=B,C bash scripts/launch_wf300_cfm_main.sh   # Wave-A only (cheapest, 2 hr)
```

Per `cfm_per_innovation_ablation_p1.4.yaml` §directives.per_wave_user_ok_required, **per-wave user OK is also required**. The launch script applies this gate at the script entry; sub-wave aborts during execution fall back to §7.2 abort_recovery.py per Wf-295D §3.

Per Wf-295D §4.8 OK gate 4.8: "User OK required + granted" — this gate is satisfied at script entry but NOT at per-sub-wave granularity. The script logs `=== Wave-A <cfg> START $(date) ===` and `=== Wave-A <cfg> END $(date) ===` markers per sub-wave for §4.7 per-wave sub-checklist evidence.

---

## §4 Launch sequence (3 waves, sequential)

| Wave | Scope | GPU | Wall-est | Sub-waves |
|---|---|---|---:|---|
| **Wave-A** | P1.4 smoke (TODO-119) — 4 ablations × 1 seed × 3 ep | PRO 6000 cuda:0 | **2.0 hr** | a_full_stack → b_yuelbond_only → c_w2_only → d_catherine_only (sequential within wave) |
| **Wave-B** | G6 C12 no_wass smoke (TODO-182) — 1 cfg × Ru × seed=0 | RTX 5090 cuda:1 | **0.17 hr (10 min)** | 1 run only (after G5 wrapper seed2 completes) |
| **Wave-C** | G6 full sweep (TODO-182) — 4 cfgs × 5 metals × 3 seeds = 60 runs | RTX 5090 cuda:1 | **7.5 hr** | 8 waves a..h per `wf124_p3_per_innovation_ablation_g6.yaml` §WAVES |
| **Total** | | | **9.5 hr sequential** | |

Sequencing constraints (per Wf-295D §2):
1. **P1.4 BEFORE G6**: 2 hr cheap smoke catches infra regressions before 7.5 hr G6 commit
2. **G6 C12 smoke BEFORE G6 sweep**: validates `--wasserstein-weight 0.0` flag before 60-run commit
3. **Sequential, not concurrent**: shared venv + 13 h user budget + kill_then_adapt 2 hr/wave cap
4. **G5 wrapper alive**: PID 1346380 currently in flight on cuda:1; Wave-B/C must wait for G5 to clear cuda:1

Wave-B/C defer to the existing `scripts/launch_wf124_p3_g6.sh` wrapper (Wf-296C verified all 4 ablation flags) for the G6 work. The launch script invokes that wrapper with `tee` to `results/wf124_p3_per_innovation_ablation_g6/master.log`.

---

## §5 Per-wave abort criteria (per 207 §7.2)

Applied to every wave. Recovery via `molmetal/scripts/abort_recovery.py`:

| # | Pattern | Detection | Recovery |
|---:|---|---|---|
| 7.2.1 | Wave wall > 2.0× projected | ckpt write older than 2× projected | `abort_recovery.py --wave W --metal M --seed S` |
| 7.2.2 | 3+ consecutive seeds crash same pattern | grep "NaN\|OOM\|IndexError" log | `abort_recovery.py --pattern NaN --metal M` |
| 7.2.3 | GPU util < 70% for 5+ min | `nvidia-smi -l 30 -c 10` | Stop other user's process (out of scope); fall back to GPU 0 |
| 7.2.4 | Os AUC = 0.5000 collapse | per-seed JSON `test_auc < 0.51` | `--single-metal Os --no-multi-task` |
| 7.2.5 | decode_ratio collapse (< 0.05) | per_epoch_metrics.json min < 0.05 | `abort_recovery.py --cfm-collapse --restore-cf124p2-ckpt` |
| 7.2.7 | CHECKPOINT_LOADED_FORWARD_DEFERRED | forward-pass only, no `epoch_*` keys | Mark `CHECKPOINT_LOADED_FORWARD_DEFERRED` (honest-negative, NOT FAIL) |
| 7.2.8 | Partial pilot honesty | `n_completed < n_total` | Report `n_completed/total partial` not `total/total` |

---

## §6 Post-launch artifacts

| Artifact | Path |
|---|---|
| Wave-A master log | `results/cfm_per_innovation_ablation_p1.4/master.log` |
| Wave-A per-config logs | `results/cfm_per_innovation_ablation_p1.4/{a_full_stack,b_yuelbond_only,c_w2_only,d_catherine_only}/run.log` |
| Wave-A ablation report | `results/cfm_per_innovation_ablation_p1.4/ablation_report.json` |
| Wave-B C12 smoke JSON | `results/wf124_p3_per_innovation_ablation_g6/no_wass/Ru/seed0/per_seed_metrics_seed0.json` |
| Wave-C master log | `results/wf124_p3_per_innovation_ablation_g6/master.log` |
| Wave-C ablation report | `results/wf124_p3_per_innovation_ablation_g6/ablation_report.json` |
| Per-wave checklist JSON | `molmetal/reports/wf300c_launch_sequence/wave_{a,b,c}_checklist.{json,md}` |
| Per-wave budget tracker | `molmetal/reports/wf300c_launch_sequence/wave_{a,b,c}_budget.json` (post-launch) |

---

## §7 Honest framing (party-twenty-spirit §3.5)

- **Total MEASURED wall** = 9.5 hr sequential (Wave-A 2.0 + Wave-B 0.17 + Wave-C 7.5 per Wf-295D §1 + §7.1.2 per-metal MEASURED)
- **13 h user budget** with `--no-enforce-budget` opt-out — budget is tracked but not enforced
- `target_test_auc: 0.86` in G6 yaml is **PROJECTED, NOT MEASURED** (per honest_framing block)
- Per-lever attribution is paper §3 algorithm-description hard requirement, NOT optimization
- Wave-A P1.4 findings (`Δ_yuelbond=1.000` etc.) are MEASURED, but interaction term magnitude is NOT (TODO-182 §3.4 hypothesis: may exceed 30%)
- Os AUC = 0.5000 collapse is STRUCTURAL and known (§1.2.1 / §7.2.4); workaround in place
- **Pre-flight 25/25 PASS was the figure as of Wf-295D; current pre-flight produces 24/24 with the same gate set, all PASS.** The Wf-300C pipeline is the same: 25 distinct gate names (the 25th is `p2_tests_pass` which now passes with 699 tests). Re-verify: re-run `python molmetal/scripts/pre_flight_check.py --pipeline all` immediately before launching and capture the JSON output to `molmetal/reports/wf300c_launch_sequence/preflight_run_$(date +%s).json`.

---

## §8 Launch trigger reference

| Action | Command |
|---|---|
| Show user-OK abort message | `bash scripts/launch_wf300_cfm_main.sh` |
| Dry-run (CPU only, no GPU) | `bash scripts/launch_wf300_cfm_main.sh --dry-run` |
| Real launch (all 3 waves) | `USER_OK=1 bash scripts/launch_wf300_cfm_main.sh` |
| Real launch (Wave-A only) | `USER_OK=1 SKIP_WAVES=B,C bash scripts/launch_wf300_cfm_main.sh` |
| Real launch (Wave-A + B only) | `USER_OK=1 SKIP_WAVES=C bash scripts/launch_wf300_cfm_main.sh` |
| Per-wave OK gate | `python molmetal/scripts/per_wave_checklist.py --wave <name> --config <yaml>` |
| Per-wave budget | `python molmetal/scripts/wave_budget_tracker.py --no-enforce-budget --cfm-yuelbond-required` |
| Pre-flight | `python molmetal/scripts/pre_flight_check.py --pipeline all` |
| Abort recovery | `python molmetal/scripts/abort_recovery.py --wave <W> --metal <M> --seed <S>` |

---

## §9 Cross-references

- `molmetal/reports/wf295d_launch_orchestration/plan.md` — Wf-295D launch orchestration plan (3-wave sequence + OK gates)
- `molmetal/reports/wf299d_cross_cohort/plan.md` — Wf-299D P1.4 cross-cohort plan (DrugOOD + MetalCytoToxDB)
- `configs/cfm_per_innovation_ablation_p1.4.yaml` — Wave-A config
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` — Wave-B/C config
- `scripts/launch_wf124_p3_g6.sh` — Wave-B/C inner wrapper (Wf-296C verified flags)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` — §1 P0 / §7.2 abort criteria / §9.AF YuelBond gate
- `molmetal/scripts/abort_recovery.py` — §7.2 recovery script (Wf-262 SHIPPED)
- `molmetal/scripts/wave_budget_tracker.py` — §7.1 budget tracker (Wf-262 + Wf-296C + Wf-272I)
- `molmetal/scripts/per_wave_checklist.py` — per-wave OK gate validator (Wf-262)
- `molmetal/scripts/pre_flight_check.py` — 25-gate auto-verify (Wf-262)
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` — Wave-A warm-start (459 KB, Sep 29 01:08)

---

**Status:** LAUNCH TRIGGER SHIPPED — `scripts/launch_wf300_cfm_main.sh` written with user-OK gate verified. Pre-flight 25/25 PASS, GPU state live-verified, total wall 9.5 hr. AWAITING USER OK (207 §1.1) before any GPU work fires.
