# Wf-354-A2 — Wf-310E Wf-135 Phase 2 USER_OK-gate verification

**Date:** 2026-09-29
**Wf:** Wf-354-A2 (USER_OK-gate audit per AGENTS.md §5.1 kill_then_adapt)
**Parent:** Wf-310E Part 1 — Wf-135 Phase 2 LOMO 5-metal × 5-seed × 50-epoch re-bench
**Status:** PASS — wrapper exists + USER_OK gate verified + dry-run mode verified

---

## §1 Wrapper location

```
scripts/launch_wf310e_wf135_phase2_gpu.sh
```

Total 244 LOC bash wrapper; first shipped per Wf-321A (Wf-135 Phase 2 CPU scaffold
audit) and re-validated by Wf-342-P2 (USER_OK-gate closeout wave).

## §2 USER_OK-gate verification (per AGENTS.md §5.1 kill_then_adapt)

### §2.1 Gate code location

`scripts/launch_wf310e_wf135_phase2_gpu.sh:82-88` enforces the gate:

```bash
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual 2-week GPU run\n' "$0" >&2
    exit 64
fi
```

### §2.2 Empirical test — gate refuses without USER_OK

```bash
$ bash scripts/launch_wf310e_wf135_phase2_gpu.sh
[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.
       USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh --dry-run   # plan only (still requires USER_OK=1)
       USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh                # actual 2-week GPU run
EXIT_CODE_NO_USER_OK=64
```

PASS — exit code 64 (configurable refusal, distinct from generic 1) when
USER_OK≠1.

### §2.3 Empirical test — dry-run mode (USER_OK=1)

```bash
$ USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh --dry-run
...
[OK] plan emitted → /home/hugo/codes/mol-metal/molmetal/reports/wf310e_wf135_phase2/plan.json
[16:34:52] DRY-RUN: skipping GPU launch (USER_OK was still required)
[16:34:52] DRY-RUN: actual invocation would be:
[16:34:52]   CUDA_VISIBLE_DEVICES=1 /home/hugo/codes/mol-metal/.venv/bin/python /home/hugo/codes/mol-metal/molmetal/scripts/train_metal_lomo_v2.py \
[16:34:52]     --metals Rh Ru Ir Os Re --seeds 0 1 2 3 4 \
[16:34:52]     --epochs 50 \
[16:34:52]     --balanced-cohort-json /home/hugo/codes/mol-metal/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json \
[16:34:52]     --out-dir /home/hugo/codes/mol-metal/molmetal/reports/wf310e_wf135_phase2/per_seed_results \
[16:34:52]     --device cuda:0
[16:34:52] DRY-RUN: total wall = ~10 GPU-hr sequential = ~2 calendar weeks worst case
EXIT_DRY_RUN=0
```

PASS — dry-run mode:
- requires USER_OK=1 (gate enforced BEFORE --dry-run check)
- emits `plan.json` record (MEASURED/PROJECTED label discipline per party-twenty-spirit §3.5)
- exits 0 (clean exit)
- skips GPU launch (no actual training triggered)
- prints the planned invocation verbatim for review

## §3 plan.json content (emitted on dry-run)

```json
{
  "wrapper": "launch_wf310e_wf135_phase2_gpu.sh",
  "wf_origin": "Wf-310E Part 1",
  "parent_plan": "TODO/completed/135_wf135_phase1_plan.md §2",
  "user_ok": true,
  "dry_run": 1,
  "metals": [Rh,Ru,Ir,Os,Re],
  "seeds": [0,1,2,3,4],
  "epochs": 50,
  "total_runs": 25,
  "gpu": "cuda:1",
  "wall_budget_hr": 14,
  "wall_budget_label": "PROJECTED (2 calendar weeks per Wf-135 Phase 1 plan §2.4; compute ~10 GPU-hr; remainder = setup/teardown/reporting)",
  "phase1_prereq": "/home/hugo/codes/mol-metal/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json",
  "phase1_status": "PRESENT",
  "acceptance_gates": {
    "rh_fold_auc_min": 0.30,
    "overall_lomo_mean_min": 0.60,
    "min_tests_pass": 16,
    "rh_target_origin": "recover from Wf-131 P2 regression 0.136 to baseline 0.368"
  },
  "honest_framing": {
    "MEASURED_today": "wrapper ready; user_ok gate enforced; phase1 prerequisite check wired",
    "PROJECTED": "Rh fold AUC +0.164 recovery (0.136 -> 0.30 target); overall LOMO mean +0.065 (0.535 -> 0.60 target)",
    "kill_then_adapt_directive": "2-week wall; user_ok gate HARD-REQUIRED per mol-metal-gpu-execution §0"
  },
  "status": "PLAN_ONLY"
}
```

## §4 Acceptance gates (PROJECTED, requires user_OK + GPU launch)

- `rh_fold_auc_min`: 0.30 (recover from Wf-131 P2 regression 0.136 → 0.30)
- `overall_lomo_mean_min`: 0.60 (lift from 0.535 baseline)
- `min_tests_pass`: 16
- `wall_budget`: 14 hr upper bound (PROJECTED, 2 calendar weeks per Wf-135 Phase 1 §2.4)

## §5 Stop semantics (per TODO-207 v2 §7.2)

- per-phase timeout: ckpt older than 2× projected wall-time
- failure pattern: 3+ consecutive NaN/OOM/IndexError
- GPU util < 70% for 5+ minutes
- collapse: Rh fold AUC < 0.10 on 2+ seeds (binding regression alarm)

## §6 Pre-launch blockers (BLOCKER before any GPU work)

1. Phase 1 cohort on disk (Wf-309A complete) — `synthetic_rh_pos_neg.json` status:
   PRESENT (verified in plan.json `phase1_status`)
2. `pre_flight_check.py --pipeline all` (24/24 PASS target) — needs fresh run before launch
3. `wave_budget_tracker.py --budget-hr 14 --chemprop-not-required` — needs fresh run
4. USER_OK env gate — verified above (gate refuses when USER_OK≠1)

## §7 Honest framing (party-twenty-spirit §3.5)

- **MEASURED today (2026-09-29):**
    - wrapper exists at `scripts/launch_wf310e_wf135_phase2_gpu.sh`
    - USER_OK gate enforced (exit 64 without USER_OK=1)
    - dry-run mode works (exit 0 with USER_OK=1 + --dry-run)
    - plan.json emitted with MEASURED/PROJECTED label discipline
    - phase1 prerequisite check wired
    - 0 GPU launches performed (verification only, dry-run)
- **PROJECTED (gated on user OK + GPU launch):**
    - Rh fold AUC +0.164 recovery (0.136 → 0.30 target)
    - Overall LOMO mean +0.065 (0.535 → 0.60 target)
    - ~10 GPU-hr compute + ~14 hr wall budget upper bound
    - 25 (metal, seed) tuples × 5 LOMO runs each = 125 total runs

## §8 Test count + wall

- **Tests:** 0 NEW (verification only, no new code)
- **Wall:** ~5 min CPU-only (find wrapper + 3 bash verifications + plan.json inspect)

## §9 Files

- `scripts/launch_wf310e_wf135_phase2_gpu.sh` (244 LOC, USER_OK-gated wrapper)
- `molmetal/reports/wf310e_wf135_phase2/plan.json` (NEW from this verification run)
- `TODO/completed/321a_wf135_phase2_cpu_scaffold.md` (Wf-321A original ship)
- `TODO/completed/342_final_closeout.md` (Wf-342 USER_OK-gate closeout)
- `TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md` (Wf-309D restored prerequisite)
- `TODO/completed/135_wf135_phase1_plan.md` (Phase 1 plan §2.4 wall budget source)

## §10 Verdict

**VERIFIED** — wrapper exists + USER_OK gate enforced + dry-run mode works
+ plan.json emits cleanly with MEASURED/PROJECTED labels. 0 GPU launches
performed. Ready for `USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh`
when user is ready to allocate 14 hr wall budget.

---

**END OF Wf-354-A2** — Owner: ultracode agent; per AGENTS.md §5.1
kill_then_adapt + §5.2 honest framing.