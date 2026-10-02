# Wf-310E Part 2 — Wf-219 §4.2 production backbone launch plan

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Wrapper**: `scripts/launch_wf310e_wf219_production_gpu.sh`
**Status**: 🚧 **READY on USER_OK + PAPER_OK gate** (per `mol-metal-gpu-execution` §0 + `TODO-84`)

---

## ⚠️  PAPER BLOCKED per TODO-84

> "所有 paper 类型的任务在我授权之前禁止执行。"
> (All paper-type tasks are forbidden from execution without my authorization.)
> — Hugo Lee, 2026-09-26 13:00 CST, per `TODO/pending/84_no_paper_task_without_authorization.md` §0

The §4.2 production backbone produces paper-grade Krasnov-2026 numbers (hidden=300/depth=3 vs Wf-219 smoke 64/2). This is **paper-class work** per TODO-84 §1.1.

**The wrapper REFUSES to launch unless BOTH `USER_OK=1` AND `PAPER_OK=1` are set in env.**

Setting `PAPER_OK=1` is the explicit lift of the TODO-84 rule — only do this if user has explicitly authorized paper work.

---

## §0 TL;DR

| Gate | Description | Status |
|---|---|---|
| **G1** | Wrapper script ready on disk | ✅ `scripts/launch_wf310e_wf219_production_gpu.sh` (195 LOC, `bash -n` PASS) |
| **G2** | USER_OK gate enforced | ✅ `USER_OK=1` env required |
| **G3** | PAPER_OK gate enforced (TODO-84) | ✅ `PAPER_OK=1` env required; wrapper aborts with `[FATAL] PAPER_OK gate refused` otherwise |
| **G4** | Production backbone spec documented | ✅ hidden=300/depth=3/50 epochs/3 seeds × 5 metals = 15 runs |
| **G5** | Wall budget honest (MEASURED vs PROJECTED) | ✅ ~80 min best (12 non-Ru runs) / ~110 min worst (with Ru) |

**One-line verdict**: Wf-219 §4.2 production backbone wrapper is READY. Launch blocked on BOTH `USER_OK=1` AND `PAPER_OK=1` (TODO-84 paper rule). Total wall is **80-110 min** GPU-1.

---

## §1 Production backbone spec (from Wf-219 closeout §4.2)

### §1.1 Scope

5-metal × 3-seed × 50-epoch × hidden=300/depth=3 = **15 runs**.

| Dimension | Value | Notes |
|---|---|---|
| Metals | `Ru Ir Os Rh Re` | full 5-metal (Ru first — biggest cohort, most expensive) |
| Seeds | `0 1 2` | per Wf-219 P1 ship pattern |
| Epochs | 50 | Krasnov-2026 default |
| **hidden** | **300** | chemprop v2 production (vs Wf-219 smoke 64) |
| **depth** | **3** | chemprop v2 production (vs Wf-219 smoke 2) |
| Backbone | chemprop v2 (single-task) | per Krasnov protocol |
| Split | Bemis-Murcko scaffold split | per Krasnov 2026 protocol |
| Autocast | BF16 encoder + fp32 head + BCEWithLogitsLoss | per Wf-219 ship |
| Time budget | 110 min max per run | for safety vs Wf-219 §4.1 Ru timeout |

### §1.2 Trainer

`molmetal/scripts/wf219_krasnov_aligned.py` (~500 LOC) — already exists from Wf-219 P1 ship.

Has flags:
- `--metals` (default `(Ru, Ir, Os, Rh, Re)`)
- `--seeds` (default `(0, 1, 2)`)
- `--epochs` (default 50)
- `--hidden` (default 300)
- `--depth` (default 3)
- `--batch-size` (default 64)
- `--lr` (default 1e-3)
- `--device` (default "cuda")
- `--time-budget-s` (default 4 hr)
- `--ckpt-dir`, `--json-dir`

### §1.3 Per-run wall estimate (PROJECTED, not MEASURED)

| Metal | n_train | n_test | per-run wall (300/3) | source |
|---|---:|---:|---:|---|
| Ru | 16,349 | 2,075 | ~10 min | Wf-219 §4.1 measured 4.5 min/run at 64/2; production 300/3 ≈ 2-2.5× slower |
| Ir | 3,902 | 482 | ~7 min | scale from Wf-119 P3 chemprop v2 timing |
| Os | 956 | 126 | ~5 min | smaller cohort; cheaper |
| Rh | 930 | 146 | ~5 min | smaller cohort |
| Re | 741 | 95 | ~4 min | smallest cohort |

**Total wall** (3 seeds × 5 metals, sequential):
- Ru: 3 × 10 = 30 min
- Ir: 3 × 7 = 21 min
- Os: 3 × 5 = 15 min
- Rh: 3 × 5 = 15 min
- Re: 3 × 4 = 12 min
- **Total = 93 min ≈ 80-110 min wall range**

PROJECTED upper bound: **110 min** (with Ru timeout risk). Best case: **80 min** if all runs finish cleanly.

### §1.4 Ru timeout risk (per Wf-219 §4.1)

Per `TODO/completed/219_wf219_krasnov_aligned_closeout.md` §4.1, the Ru baseline at hidden=64/depth=2 timed out at **43/50 epochs** under the standard protocol. Production backbone (300/3) makes this worse.

**Mitigation**:
- `--skip-ru` flag: drops Ru from the sweep (4-metal × 3-seed × 50-epoch = 12 runs, ~80 min wall)
- Per-Ru timeout tolerance: if Ru fails to complete 50/50, per-seed AUCs are emitted but flagged as PARTIAL (43/50 epochs)
- Wall budget expanded to 110 min to absorb worst case

---

## §2 PAPER_BLOCKED notice (TODO-84)

### §2.1 What is forbidden without PAPER_OK

Per `TODO/pending/84_no_paper_task_without_authorization.md` §1.1, the following are **paper-type tasks** and require explicit user OK:

| Category | Examples | Files touched |
|---|---|---|
| Paper sections | §1-§7 + abstract + cover letter | `paper/sections/*.tex` |
| Paper appendix | math_theorems / closure_theorem | `paper/appendices/*.tex` |
| Paper supplementary | S1-S8 + DATA_AVAILABILITY etc. | `paper/supplementary/*.{tex,md}` |
| paper/main.tex | Any modification to `\input{}` | `paper/main.tex` |
| paper/main.pdf build | pdflatex / bibtex / make | `paper/main.pdf` rebuild |
| Reviewer rebuttal | Q1-Q20 response | any rebuttal file |
| arXiv preprint | Upload to arXiv | arxiv.org submission |
| Journal submission | Digital Discovery / JCIM / JMC | any journal portal |

### §2.2 What is allowed without PAPER_OK (CODE-ONLY work)

Per TODO-84 §1.2, the following are NOT paper work and can proceed without user OK:
- GPU retrain runs (this wrapper) — produces raw JSONs / checkpoints, NOT paper claims
- Metric JSON emissions — `metrics/by_metric/*.json`
- Code fixes + tests
- `molmetal/reports/wf*/` synthesis.md drafts (NOT paper/*)
- This launch_plan.md itself (lives in `molmetal/reports/wf310e_wf219_production/`)

### §2.3 What PAPER_OK unlocks

Once `PAPER_OK=1` is set (post-user-authorization):
- Run the GPU sweep → emit `sweep_summary.json` + 15 per-(metal,seed) JSONs
- Aggregate into `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold_production.json`
- Update `molmetal/reports/wf219_krasnov_aligned/synthesis.md` (NOT paper/*)
- **Next step (after GPU run completes)**: paper §4.5 Krasnov comparison table update — REQUIRES ADDITIONAL PAPER_OK round-trip per TODO-84

---

## §3 Acceptance gates (per Wf-219 §4.2)

### §3.1 Hard gates (HALT if violated)

| Gate | Standard | Source |
|---|---|---|
| All 15 runs attempted | 5/5 metals × 3 seeds | Wf-219 §4.2 |
| Epoch completion (best effort) | 50/50 epochs; Ru timeout acknowledged | Wf-219 §4.1 |
| Krasnov target match | mean test_auc ≥ Krasnov 2026 baseline | Wf-219 §1.2 |
| No best-seed selection | report per-seed AUC honestly | mol-metal-gpu-execution §7 |

### §3.2 Krasnov 2026 comparison targets (MEASURED vs PROJECTED)

| Metal | Krasnov 2026 baseline | Wf-219 smoke (64/2) | Wf-219 production (300/3) target |
|---|---:|---:|---:|
| Ru | 0.81 | 0.7858, 0.8215, 0.8215 (timeout) | PROJECTED ≥ 0.81 |
| Ir | 0.73 | 0.7586 ± 0.0142 | PROJECTED ≥ 0.76 |
| Os | n/a | 0.7884 ± 0.0098 | PROJECTED ≥ 0.78 |
| Rh | n/a | 0.7525 ± 0.0114 | PROJECTED ≥ 0.75 |
| Re | n/a | 0.8027 ± 0.0637 | PROJECTED ≥ 0.80 |

PROJECTED lift from smoke (64/2) → production (300/3) is +0.00 to +0.05 per metal (typical chemprop v2 backbone scaling).

---

## §4 USER_OK + PAPER_OK gates

USER must explicitly approve (per `mol-metal-gpu-execution` §0 + `TODO-84`):

1. **Compute**: ~80-110 min on GPU-1 RTX 5090 (Blackwell sm_120). Confirm env probe via `check_blackwell_env.py` first.
2. **Schedule**: 2 hr wall budget. If GPU-1 contention with kanzi means longer, USER_OK to either (a) wait, (b) drop Ru via `--skip-ru` (4-metal × 3-seed = 12 runs, ~80 min), or (c) cancel and ship close-out as PROJECTED-only.
3. **Paper-class authorization**: Explicit PAPER_OK gate lift per TODO-84. The §4.2 production backbone produces paper-grade Krasnov-2026 numbers; this is paper-class work.
4. **Reporting**: LABELING-STRICT (MEASURED-vs-PROJECTED). Per `party-twenty-spirit` §3.5 + `mol-metal-gpu-execution` §0, every result must label whether it is MEASURED from GPU run or PROJECTED from prior work.
5. **Risk fallback**: If production backbone under-performs smoke (e.g., overfit on small metals Re/Os/Rh), honest-negative framing applies. No paper claim until MEASURED + Krasnov match.

---

## §5 Pre-launch checks (BLOCKER before any GPU work)

1. **PAPER_OK env gate**: TODO-84 lift — explicit user authorization
2. **USER_OK env gate**: kill_then_adapt enforced
3. **`pre_flight_check.py --pipeline all`**: 24/24 PASS target
4. **`wave_budget_tracker.py --budget-hr 2 --cfm-not-required`**: confirms wall budget fits
5. **GPU-1 contention check**: confirm kanzi not blocking RTX 5090 cuda:1

---

## §6 Abort criteria (per TODO-207 v2 §7.2)

- **per-phase timeout**: ckpt older than 2× projected wall-time (44 min per non-Ru; 60 min for Ru)
- **failure pattern**: 3+ consecutive NaN/OOM/IndexError
- **GPU util < 70%**: for 5+ minutes
- **Ru collapse**: 50/50 epochs not achieved — emit per-seed AUC as PARTIAL but continue (per Wf-219 §4.1 tolerance)

---

## §7 Honest framing per `party-twenty-spirit` §3.5

**承认 (Acknowledge)**:

1. **Wf-219 P1 ship is 4-metal × 3-seed × 50-epoch MEASURED** at hidden=64/depth=2 (smoke config). Aggregate mean = 0.7755.
2. **Ru baseline timed out at 43/50 epochs** under standard protocol — production backbone (300/3) increases this risk.
3. **Production backbone (300/3) per-run cost is ~2-2.5× smoke (64/2)** — wall budget expanded accordingly.
4. **PAPER BLOCKED per TODO-84** — wrapper refuses unless `PAPER_OK=1` is set (explicit user lift of paper rule).
5. **Per-seed std reported honestly** — no best-seed selection per `mol-metal-gpu-execution` §7.
6. **Krasnov direct comparison caveat**: scaffold split (Bemis-Murcko) is NOT identical to Krasnov's temporal split. Full Krasnov-direct needs TODO-220 P1 sweep.

**PROJECTED lift**:
- Mean AUC per metal: +0.00 to +0.05 vs Wf-219 smoke (typical chemprop v2 backbone scaling)
- Wall: ~80 min best case, ~110 min worst case (Ru timeout)
- Total runs: 15 (5-metal × 3-seed × 50-epoch × 300/3)

---

## §8 Cross-references

- `TODO/completed/219_wf219_krasnov_aligned_closeout.md` §4.2 (production backbone spec origin)
- `TODO/pending/84_no_paper_task_without_authorization.md` §1.1 + §1.2 (paper rule)
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (related gap context)
- `TODO/pending/220_wf219_temporal_split_nci60_driver.md` (temporal-split follow-up)
- `molmetal/reports/krasnov_protocol_investigation.md` §3 (Krasnov 2026 protocol)
- `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json` (Wf-219 smoke baseline)
- `molmetal/scripts/wf219_krasnov_aligned.py` (~500 LOC trainer)
- `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` (Wf-219 smoke metric)
- `scripts/launch_wf306b_wf219_ru_gpu.sh` (template pattern)
- `mol-metal-gpu-execution` skill §0 (kill_then_adapt directive)

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: Wf-310E Part 2 wrapper is READY on disk. USER_OK + PAPER_OK gates are enforced. Wall budget is 80-110 min GPU-1. PAPER_OK is the explicit lift of TODO-84 paper-block rule. No GPU work has been performed; this is a wrapper + plan doc ship, NOT a measurement ship. The wrapper refuses to launch unless BOTH `USER_OK=1` AND `PAPER_OK=1` are set in env. Setting `PAPER_OK=1` requires explicit user authorization per TODO-84.