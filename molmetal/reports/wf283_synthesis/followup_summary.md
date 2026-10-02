# Wf-283E — Follow-up Synthesis (Aggregation of Wf-283A/B/C/D)

**Date**: 2026-09-29
**Wf**: Wf-283E (aggregate Wf-283A-D follow-up outcomes)
**Owner**: Hugo Lee + ultracode agents
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §6 honest-negative + `mol-metal-paper-writing`
**Mode**: Read-only aggregation; no source files / launchers / configs modified.

---

## §0 TL;DR (FLAT schema verdict)

| Gate | Value | Verdict |
|---|---|---|
| **G1_n_upstream_waves** | 4 (Wf-283A / B / C / D) | — |
| **G2_n_completed** | **3** (Wf-283A ✅ + Wf-283B ✅ + Wf-283C ✅) | ⚠️ PARTIAL — Wf-283D NOT YET PRODUCED on disk |
| **G3_n_failed_or_undefined** | **1** (Wf-283D §9.AG + §9.5 wording fix MISSING) | ⚠️ D-missing |
| **G4_critical_honest_negatives_remaining** | **6 items** (see §3) | ⚠️ NOT MET — carried from Wf-280E with Wf-283B/C closures logged |
| **G5_next_priorities** | 6 items (see §4) — **3 are parallelizable CPU-only ultracode repair tasks** | — |
| **G6_report_path** | `molmetal/reports/wf283_synthesis/followup_summary.md` | PASS — this file |
| **honest_framing_note** | "Wf-283 closed 3 of 4 upstream meta-pipeline gaps: (A) verified G5 wrapper self-recovered — Ru seed=0 actually completed with ckpt saved, so HN-1 from Wf-280E is DOWNGRADED from HIGH to LOW (wrapper is alive, just slow); (B) added `--no-kendall` / `--no-dann` CLI surface so G6 ablation gating is no longer CLI-blocked; (C) wired Wf-262 pre-flight trio into all 3 Wf-124 P3 launchers, closing Wf-279D §5 cross-cutting bug. Wf-283D paper §9.AG + §9.5 wording fix is NOT YET ON DISK — gap from Wf-280E HN-5 carried forward. Domain honest-negative ledger otherwise UNCHANGED: 15/17 Cf-258 waves PENDING, 14/15 G5 pairs still PENDING (Ru seed=0 done, seed=1 running, 13 to go), G6 60/60 PENDING, CFM ablation P1.3/P1.4 PENDING, Wf-283B §7 explicit non-goal: flag→loss wiring NOT done (still needs Wf-284 C10-C17 wave execution)." | per `party-twenty-spirit` §3.5 |

**Headline**:
- **3 of 4 upstream waves delivered MEASURED signal**: Wf-283A (G5 recovery verification, Ru seed=0 ACTUALLY completed), Wf-283B (--no-kendall/--no-dann flags + 8 tests), Wf-283C (Wf-262 trio wired into G4/G5/G6 launchers, 27 LOC).
- **Wf-283D NOT produced** — paper §9.AG + §9.5 wording fix missing on disk; the honest gap from Wf-280E HN-5 (paper_dashboard YuelBond stale) is STILL UNRESOLVED at the wording layer.
- **Wf-283A DOWNGRADED HN-1**: G5 wrapper self-recovered. Ru seed=0 completed at 00:28 with test_auc=0.7716 + 11.3 MB ckpt saved. Wf-280D misread was a stale-read, not a real abort.
- **Wf-283C CLOSED HN-3**: Wf-262 pre-flight trio (pre_flight_check + per_wave_checklist + wave_budget_tracker) now fires in all 3 Wf-124 P3 launchers BEFORE any GPU work.
- **Wf-283B CLOSED the CLI-surface half of HN-2**: `--no-kendall` / `--no-dann` flags now parse correctly and 8 unit tests pass; the downstream loss/model wiring is EXPLICITLY OUT OF SCOPE per Wf-283B §7 (deferred to Wf-284 C10-C17 wave execution).

---

## §1 Per-wave aggregation (MEASURED vs MISSING)

### §1.1 Wf-283A — G5 wrapper abort RECOVERY verification — **MEASURED SUCCESS** (HN-1 DOWNGRADED HIGH → LOW)

**File on disk**: `molmetal/reports/wf283a_g5_recovery/recovery_report.md` (5,628 bytes, 2026-09-29 00:30 CST).

**Findings (verbatim per recovery_report.md §1.1–1.5)**:

> The old wrapper PID 609543 is no longer running, but a NEW wrapper (609528) launched a fresh `launch_wf124_p3_g5.sh` that has been running since Sep 28 23:41:04. The new wrapper is the live G5 sweep.

> [Ru seed=0] epoch 47/50 loss=0.4011 ... val_auc=0.8177 ... (44.1s)
> [Ru seed=0] epoch 48/50 loss=0.3867 ... val_auc=0.8059 ... (66.6s)
> [Ru seed=0] epoch 49/50 loss=0.3628 ... val_auc=0.7481 ... (68.4s)
> [Ru seed=0] epoch 50/50 loss=0.3490 ... val_auc=0.8165 ... (80.5s)
> [Ru seed=0] DONE best_epoch=47 val_auc=0.8177 test_auc=0.7716 test_pearson=0.548 (2850.0s, featurise=0.5s)
> [wf131-mtw] DONE total elapsed=2864.2s
> Ru: n_train=12471 n_val=2166 n_test=4350 test_auc=0.7716±0.0000 test_pearson=0.548

> Checkpoint `dmpnn_mtw_ru_seed0.pt` (11.3 MB) saved at 00:28 — final-epoch ckpt, status OK.

**Verdict**: **MEASURED PASS — Ru seed=0 ACTUALLY COMPLETED with full 50 epochs, ckpt saved, test_auc=0.7716 emitted.** Wf-280D's "cut at epoch 47/50, no ckpt" claim was a **stale-read** (report generated while seed=0 was still running OR before new wrapper finished re-run). The wrapper PID 609528 self-recovered by `rm -rf`-ing the results dir and re-launching at 23:41:04.

**State snapshot at 00:30 CST**: 1/15 done (Ru seed=0), 1/15 running (Ru seed=1 epoch 1/50 done, ETA ~67 min), 13/15 pending. **Total ETA to G5 completion: ~7-8 hr wall-clock** (Ru seed=1 + 12 more cohorts).

**Critical learning (carry to Wf-280D postmortem)**: add a "wait until train finishes + check last_epoch >= target_epochs" guard before declaring "abort". The current Wf-280D heuristic ("no ckpt at file path → abort") would have correctly fired on epoch 47 *intermediate state*, but missed that the wrapper was about to save the final ckpt 30 seconds later.

### §1.2 Wf-283B — G6 flag gap fix (`--no-kendall` / `--no-dann`) — **MEASURED SUCCESS (CLI-surface only; downstream wiring DEFERRED)**

**File on disk**: `molmetal/reports/wf283b_g6_flags_fix/fix_report.md` (4,978 bytes, 2026-09-29).

**Findings (verbatim per fix_report.md §1–4)**:

> The strings "kendall" and "dann" were **TRULY MISSING** from the script source — not present-but-undocumented. Both features already exist in the loss / model code (cf. `molmetal/reports/wf241_launch.log` already prints `kendall=True dann=True`) but had no CLI surface.

> Added to `parse_args` (immediately after `--cpu-num-workers`):
> ```python
> # Wf-283B — C10-C17 wave gating flags.
> p.add_argument("--no-kendall", dest="kendall", action="store_false", default=True, ...)
> p.add_argument("--no-dann",    dest="dann",    action="store_false", default=True, ...)
> ```

> Replaced `15-30% util` with `15-30 percent util` on line 1190. This restores `--help` to a printable state.

> Wrote `molmetal/tests/test_no_kendall_no_dann_flags.py` with **8 tests**:
> 1. test_no_kendall_flag_toggles_false — `--no-kendall` → `args.kendall=False`
> 2. test_no_dann_flag_toggles_false — `--no-dann` → `args.dann=False`
> 3. test_defaults_kendall_and_dann_are_true — empty argv → both True
> 4. test_combined_no_kendall_no_dann_smoke — combined toggle parses
> 5. test_help_lists_no_kendall_flag — help text contains `--no-kendall`
> 6. test_help_lists_no_dann_flag — help text contains `--no-dann`
> 7. test_subprocess_help_lists_new_flags — end-to-end `--help` in venv
> 8. test_subprocess_no_kendall_no_dann_smoke_runs — combined smoke
>
> 8 passed, 1 warning in 10.56s.

**Verdict**: **MEASURED PARTIAL — CLI surface fixed, 8 tests pass, `--help` crash resolved as bonus fix. EXPLICIT NON-GOAL (per fix_report.md §7): wire `args.kendall` / `args.dann` into the loss and model — currently the flags parse but downstream `mtl_loss()` and `DANN_GRL()` calls do not yet consult them. That wiring belongs to Wf-284 (C10-C17 wave execution).**

**Files touched**: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (added 2 flags ≈28 LOC + fixed `15-30%` → `15-30 percent` 1 LOC); `molmetal/tests/test_no_kendall_no_dann_flags.py` (new, 8 tests ≈200 LOC).

### §1.3 Wf-283C — launcher × wave-tool wiring — **MEASURED SUCCESS (HN-3 CLOSED)**

**File on disk**: `molmetal/reports/wf283c_launcher_wiring/wiring_report.md` (4,047 bytes, 2026-09-29).

**Findings (verbatim per wiring_report.md)**:

> Added a wave-tool guard block to all 3 Wf-124 P3 launchers (G4/G5/G6). The block wires in:
> 1. `pre_flight_check.py --pipeline all` — auto-verify TODO-207 v2 pre-flight checklist (P1 + P2 gates, exit 0 = all PASS, exit 1 = any FAIL).
> 2. `per_wave_checklist.py --wave "Wf-124-P3-G{4,5,6}"` — per-wave sub-checklist per TODO-208 / Wf-262.
> 3. `wave_budget_tracker.py` — live wave budget tracker per 207 v2 §7.1 (default: --enforce-budget; emits DEFERRED labels for over-budget waves).
>
> The guard is placed AFTER the existing `--dry-run` early-exit on G5 and G6 so dry-run mode still bypasses the guards. G4 has no dry-run handling so the guard fires unconditionally.

> Total LOC added: 21 lines × 3 wrappers = 21 wrapper-lines + 6 comment/blank = ~27 net wrapper LOC (below the 45-LOC budget; comment-heavy per the Wf-280D recommendation).

**Verification**:
- `bash -n` syntax check: G4/G5/G6 all OK.
- G5 `--dry-run`: guard correctly SKIPPED (dry-run exits at line 79 before reaching new guards).
- G6 `--dry-run`: guard correctly SKIPPED; output enumerates 60 tuples, 30 CONFIRMED + 30 UNVERIFIED.
- G4 (no `--dry-run` handler): guards fire and main loop proceeds (`pre_flight_check.py PASSED` + `per_wave_checklist.py PASSED` + `wave_budget_tracker.py PASSED`).

**Verdict**: **MEASURED PASS — all 3 launchers are now wave-tool aware. The Wf-279D §5 cross-cutting bug (G4/G5/G6 skip pre_flight_check / per_wave_checklist / wave_budget_tracker) is FIXED.** Any future real-launch invocation will be gated by all 3 wave-tools before any GPU work begins, enforcing the "verify before launch" invariant per TODO-207 v2 §7.1.

**Files modified**: `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g4.sh`, `launch_wf124_p3_g5.sh`, `launch_wf124_p3_g6.sh`.

### §1.4 Wf-283D — §9.AG + §9.5 wording fix — **NOT YET PRODUCED ON DISK**

**File on disk**: **NONE** — no `wf283d_*/` directory under `molmetal/reports/`. Searched with `find` + `grep -ril` for any `wf283d` reference in the reports tree; zero matches.

**Implication**: The honest gap from Wf-280E HN-5 (paper_dashboard stale on Wf-269C YuelBond load-bearing badge + Wf-269C signal wording) is **STILL UNRESOLVED at the wording layer**. This gap remains as a CRITICAL carry-over.

**Status**: MISSING — must be produced by a subsequent workflow round (Wf-283D itself or Wf-284D).

---

## §2 Aggregate statistics

| Metric | Value |
|---|---|
| Upstream waves dispatched | 4 |
| Upstream waves with on-disk MEASURED output | **3** (Wf-283A + Wf-283B + Wf-283C) |
| Upstream waves MISSING (no on-disk report) | **1** (Wf-283D) |
| MEASURED CRITICAL honest-negatives CLOSED | **2** (HN-3 via Wf-283C; HN-1 DOWNGRADED HIGH→LOW via Wf-283A) |
| MEASURED PARTIAL honest-negatives delivered | **1** (Wf-283B CLI surface only; downstream loss/model wiring deferred to Wf-284) |
| Total follow-up work MEASURED-completed | **3 of 4 attempted** = 75% |
| Total follow-up work MEASURED-failed (or pending) | **1 of 4** (Wf-283D not produced) — **0 hard failures** (no on-disk FAIL signal) |

**Per `party-twenty-spirit` §3.5**: Wf-283 is the SECOND consecutive meta-pipeline-clean round (Wf-280 broke the Wf-270/271/272/274/279 5-occurrence streak; Wf-283 maintained it with 3/4 delivered). The Wf-283A finding has a notable **inversion** of the Wf-280D audit: Wf-280D reported "Ru seed=0 cut at epoch 47/50, no ckpt" as a HIGH-severity blocker (HN-1), but Wf-283A verified that Ru seed=0 ACTUALLY COMPLETED at 00:28 with ckpt saved. This is a **stale-read regression** rather than an actual abort — the Wf-280D heuristic was wrong, not the wrapper.

**Domain honest-negative gap**:
- G5 14/15 pairs PENDING (Ru seed=0 done at 00:28, seed=1 running, 13 to go; ETA ~7-8 hr total wall-clock)
- G6 60/60 runs PENDING (wrapper NEVER launched; CLI flags now exist but downstream wiring DEFERRED to Wf-284)
- CFM ablation P1.3/P1.4 PENDING on 16-hr Blackwell pre-launch scope
- Cf-258 ledger overall: 15/17 waves PENDING (UNCHANGED from Wf-280E)
- Wf-283D §9.AG + §9.5 wording: MISSING (Wf-280E HN-5 STILL UNRESOLVED at wording layer)
- Workflow harness upstream-pipeline regression root cause (Wf-280E HN-6): STILL UNRESOLVED (no Wf-283 wave targeted it)

---

## §3 Critical honest-negatives remaining

| # | Item | Severity | Status vs Wf-280E | Evidence | Owner |
|---|---|---|---|---|---|
| **HN-1** | **Cf-258 G5 wrapper abort** (C6–C9, 4 waves, 15 pairs) — Ru seed=0 mid-run at epoch 47/50, no ckpt, no per_seed_metrics.json | **LOW** (DOWNGRADED from HIGH) | ⚠️ PARTIALLY CLOSED via Wf-283A — Ru seed=0 actually completed at 00:28 with ckpt; 13 pairs still PENDING on wrapper continuation | `molmetal/reports/wf283a_g5_recovery/recovery_report.md` §1.3 + `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` (DONE line + ckpt) | ultracode agent (GPU runner) — live |
| **HN-2** | **Cf-258 G6 wrapper NEVER launched + `--no-kendall` / `--no-dann` flags UNVERIFIED** (C10–C17, 8 waves, 60 runs) | **HIGH** (UNCHANGED) | ⚠️ CLI-surface closed via Wf-283B; downstream `mtl_loss()` / `DANN_GRL()` wiring DEFERRED to Wf-284 per Wf-283B §7 explicit non-goal | `molmetal/reports/wf283b_g6_flags_fix/fix_report.md` §7 + `results/wf124_p3_per_innovation_ablation_g6/` empty | ultracode agent (Python CLI + loss author) — Wf-284 |
| **HN-3** | **Wf-262 pre-launch safety trio NOT wired into any Wf-124 P3 launcher** (carry-over from Wf-279D) | **CLOSED** | ✅ RESOLVED via Wf-283C — guard block fires in G4/G5/G6 before any GPU work | `molmetal/reports/wf283c_launcher_wiring/wiring_report.md` (G4/G5/G6 all pass `bash -n`; dry-run smoke OK) | ultracode agent (Bash + smoke verify) — done |
| **HN-4** | **CFM ablation P1.3 + P1.4 NOT run on PlatinAI / DrugOOD cohorts** (C3–C5, 3 waves) | **HIGH** (UNCHANGED) | UNCHANGED — no Wf-283 wave targeted this | `molmetal/reports/wf280d_c258_waves/audit_report.md` §1.2 + Wf-276 §3 honest-negative | ultracode agent (Blackwell GPU-0) |
| **HN-5** | **Wf-283D §9.AG + §9.5 wording fix NOT produced** + paper_dashboard.py YuelBond badge stale (carry-over from Wf-279 §1.3 + Wf-280E §3 HN-5) | **MEDIUM** (UNCHANGED + Wf-283D missing) | UNCHANGED — Wf-283D not on disk | `find /home/hugo/codes/mol-metal -name "*wf283d*"` returns empty | Hugo Lee (paper authority) |
| **HN-6** | **Workflow harness upstream-pipeline regression class root cause STILL UNRESOLVED** (5-occurrence streak; Wf-280 broke it for MEASURED but root cause not identified) | **SYSTEMIC** (UNCHANGED) | UNCHANGED — Wf-283A surfaced a STALE-READ sub-class of the same regression (Wf-280D report generated before wrapper finished re-run) | Wf-280E §3 HN-6 + Wf-283A §4 (action item: "wait until train finishes + check last_epoch >= target_epochs guard before declaring abort") | Hugo Lee (workflow author) |

**Note**: HN-1 status updated from Wf-280E HIGH → LOW based on Wf-283A verified completion. HN-2 PARTIALLY CLOSED on CLI surface only; downstream wiring still needed. HN-3 fully closed. HN-4, HN-5, HN-6 unchanged.

---

## §4 Next follow-up priorities (ordered by ROI × urgency × parallelizability)

### P1.1 — Produce Wf-283D §9.AG + §9.5 wording fix on disk (≤1 hr CPU + paper edit)

**Evidence**: HN-5 (carry-over from Wf-280E §3 + Wf-279 §1.3) — `find` confirms zero `wf283d_*/` directories on disk; Wf-283D was dispatched in the harness task text but no agent output landed.
**Action**: Edit `paper/main.tex` (or applicable section file per `mol-metal-paper-writing` skill) to fix §9.AG + §9.5 wording for Wf-269C YuelBond load-bearing context. Update `paper_dashboard.py` if YuelBond badge needs refresh. Emit `molmetal/reports/wf283d_wording_fix/wording_report.md` with verbatim diffs.
**Owner**: Hugo Lee (paper authority). **PARALLELIZABLE**: CPU-only paper edit, can run with P1.2 + P1.3.
**Why P1**: Closes the only on-disk-MISSING wave from Wf-283; ≤1 hr; zero GPU cost.

### P1.2 — Wire `args.kendall` / `args.dann` into `mtl_loss()` + `DANN_GRL()` (~2 hr CPU + smoke verify)

**Evidence**: HN-2 + `molmetal/reports/wf283b_g6_flags_fix/fix_report.md` §7 EXPLICIT NON-GOAL — "wire `args.kendall` / `args.dann` into the loss and model ... currently the flags parse but downstream `mtl_loss()` and `DANN_GRL()` calls do not yet consult them. That wiring belongs to Wf-284 (C10-C17 wave execution)."
**Action**: Edit `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` to consult `args.kendall` in the MTL loss construction and `args.dann` in the DANN Gradient-Reversal Layer instantiation. Add 2 unit tests in `molmetal/tests/test_train_dmpnn_multitask_wasserstein_cli.py` verifying the flags actually disable the relevant loss terms (not just parse). Confirm `molmetal/reports/wf241_launch.log` line still shows `kendall=True dann=True` under default invocation.
**Owner**: ultracode agent (Python loss author). **PARALLELIZABLE**: CPU-only, can run with P1.1 + P1.3.
**Why P1**: HIGH-severity honest-negative; unblocks 8 G6 waves (30/60 runs at minimum with the now-parseable flags); ≤2 hr; zero GPU cost.

### P1.3 — Add `--no-enforce-budget` CLI flag to `train_dmpnn_multitask_wasserstein.py` for parity with `wave_budget_tracker.py` (~1 hr CPU)

**Evidence**: Wf-280E §4 P1.3 + Wf-281C §1 — the `--no-enforce-budget` policy directive is at the orchestrator layer, but the python CLI doesn't have the flag. Adding the flag at the CLI layer would make `--no-enforce-budget` greppable from the python file (closing the Wf-281C honest caveat) AND would let G5/G6 wrappers actually pass it (closing the Wf-280A PARTIAL gap from Wf-280E).
**Action**: Add 1 argparse option to `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:
```python
parser.add_argument("--no-enforce-budget", action="store_true", help="disable wave budget enforcement (kill_then_adapt session-level policy)")
```
plus 1 unit test verifying the flag parses and the `wave_budget_tracker` check is skipped.
**Owner**: ultracode agent (Python CLI author). **PARALLELIZABLE**: CPU-only, can run with P1.1 + P1.2.
**Why P1**: Closes the Wf-280A PARTIAL gap; ≤1 hr; zero GPU cost.

### P1.4 — Add "wait until train finishes + check last_epoch >= target_epochs" guard to wrapper abort heuristic (~1 hr CPU + read)

**Evidence**: HN-6 stale-read sub-class + `molmetal/reports/wf283a_g5_recovery/recovery_report.md` §4 EXPLICIT ACTION ITEM — "add a 'wait until train finishes + check last_epoch >= target_epochs' guard before declaring 'abort'. The current Wf-280D heuristic ('no ckpt at file path → abort') would have correctly fired on epoch 47 *intermediate state*, but missed that the wrapper was about to save the final ckpt 30 seconds later."
**Action**: Patch `molmetal/scripts/wf229_ablation_launcher.py` (or whichever wrapper-class owns the abort heuristic) to (a) poll `run.log` for the `DONE` line + check `last_epoch >= target_epochs` before declaring abort, (b) emit a structured `[ABORT-VERIFIED]` vs `[ABORT-STALE-READ]` tag, (c) re-run Wf-280D retrospective on the recovered data to confirm no actual abort occurred.
**Owner**: ultracode agent (Bash + wrapper author). **PARALLELIZABLE**: CPU-only investigation + patch.
**Why P1**: SYSTEMIC; addresses the stale-read sub-class of the 5-occurrence upstream-pipeline regression; ≤1 hr; zero GPU cost.

### P2.1 — Investigate workflow harness upstream-pipeline regression root cause (~4 hr)

**Evidence**: HN-6 + 5 consecutive occurrences (Wf-270, Wf-271, Wf-272, Wf-274, Wf-279) — Wf-280 + Wf-283 broke the streak for MEASURED-output-on-disk but the root cause is still unidentified. Wf-283A surfaced a STALE-READ sub-class that Wf-280D triggered.
**Action**: Read `TODO/completed/207_unified_pre_flight_checklist_v2.md` + the synthesis report series (Wf-270/271/272/274/279/280/283) to identify the common upstream-pipeline regression class. Likely root cause: harness not awaiting subagent stdout/stderr capture, or subagents returning too quickly to the parent before writing to disk, or synthesis agents reading stale files (the Wf-283A sub-class).
**Owner**: Hugo Lee (workflow author). **NOT parallelizable**: requires reading + synthesis.
**Why P2**: SYSTEMIC; addresses the historical follow-up-failure rate; out-of-scope for ≤2 hr window but high-value.

### P3.1 — CFM ablation P1.3 + P1.4 on PlatinAI / DrugOOD cohorts (GPU, ~16 hr Blackwell)

**Evidence**: HN-4 + `molmetal/reports/wf280d_c258_waves/audit_report.md` §1.2 — C3–C5 PENDING on 16-hr Blackwell pre-launch scope (PlatinAI 4-cell + DrugOOD per-cohort assays not wired).
**Action**: Wire PlatinAI 4-cell + DrugOOD per-cohort assays into the CFM ablation harness per `configs/cfm_per_innovation_ablation_p1.4.yaml`. Launch 3 waves × ~5 hr each on Blackwell GPU-0.
**Owner**: ultracode agent (Blackwell GPU-0). **NOT parallelizable**: needs Blackwell GPU + Vina CPU.
**Why P3**: HIGH-severity honest-negative; out-of-scope for ≤2 hr window; ~16 hr GPU wall-clock.

---

## §5 Parallelizability matrix (for ultracode dispatch)

| Priority | Wall | GPU? | Parallelizable? | Depends on |
|---|---|---|---|---|
| **P1.1** (produce Wf-283D wording fix) | ~1 hr | NO | **YES** — CPU only, can run with P1.2 + P1.3 + P1.4 | paper/main.tex (already on disk) |
| **P1.2** (wire kendall/dann into loss) | ~2 hr | NO | **YES** — CPU only, can run with P1.1 + P1.3 + P1.4 | Wf-283B's flag surface (already on disk) |
| **P1.3** (add --no-enforce-budget CLI) | ~1 hr | NO | **YES** — CPU only, can run with P1.1 + P1.2 + P1.4 | train_dmpnn_multitask_wasserstein.py source (already on disk) |
| **P1.4** (stale-read guard patch) | ~1 hr | NO | **YES** — CPU only, can run with P1.1 + P1.2 + P1.3 | wf283a recovery_report.md §4 action item |
| **P2.1** (workflow harness root cause) | ~4 hr | NO | NO — requires reading + synthesis | All prior synthesis reports (already on disk) |
| **P3.1** (CFM ablation P1.3 + P1.4) | ~16 hr | YES (Blackwell GPU-0) | NO — needs GPU + Vina | `configs/cfm_per_innovation_ablation_p1.4.yaml` (already on disk) |

**Recommended ultracode parallelization** (4 agents in parallel, all CPU-only):
1. **ultracode-283E-A**: P1.1 (produce Wf-283D wording fix, ~1 hr)
2. **ultracode-283E-B**: P1.2 (wire kendall/dann into loss, ~2 hr)
3. **ultracode-283E-C**: P1.3 (add --no-enforce-budget CLI, ~1 hr)
4. **ultracode-283E-D**: P1.4 (stale-read guard patch, ~1 hr)

**Total wall** (4 agents parallel): **~2 hr** (limited by P1.2, the longest)
**Total wall** (sequential): **~5 hr** (P1.1 + P1.2 + P1.3 + P1.4)
**Speedup**: 2.5× via parallel dispatch.

**GPU-only follow-ups** (post-CPU, sequential after G5 completes):
- P3.1 (CFM ablation P1.3 + P1.4, ~16 hr Blackwell GPU-0)
- Live G5 wrapper continuation (Ru seed=1 + 12 more cohorts, ~7-8 hr GPU-1, no action needed)

---

## §6 Files referenced

- `/home/hugo/codes/mol-metal/molmetal/reports/wf283a_g5_recovery/recovery_report.md` (Wf-283A G5 recovery, MEASURED)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf283b_g6_flags_fix/fix_report.md` (Wf-283B flag gap fix, MEASURED PARTIAL)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf283c_launcher_wiring/wiring_report.md` (Wf-283C Wf-262 trio wiring, MEASURED)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf283_synthesis/followup_summary.md` (this file, Wf-283E synthesis)
- `/home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` (Ru seed=0 DONE line, ckpt saved)
- `/home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ckpts/dmpnn_mtw_ru_seed0.pt` (11.3 MB, saved 00:28)
- `/home/hugo/codes/mol-metal/results/wf124_p3_per_innovation_ablation_g6/` (EMPTY — G6 wrapper never launched, HN-2 carry-over)
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (now has --no-kendall / --no-dann; still missing --no-enforce-budget + downstream wiring)
- `/home/hugo/codes/mol-metal/molmetal/tests/test_no_kendall_no_dann_flags.py` (new, 8 tests, all PASS)
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g4.sh`, `launch_wf124_p3_g5.sh`, `launch_wf124_p3_g6.sh` (Wf-283C wave-tool guards wired in)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf280_synthesis/followup_summary.md` (Wf-280E prior synthesis)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf280d_c258_waves/audit_report.md` (Wf-280D Cf-258 audit, MEASURED — basis for Wf-283A inversion)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf279d_launcher_integration/integration_report.md` (Wf-279D launcher bypass, MEASURED — HN-3 origin)
- `/home/hugo/codes/mol-metal/TODO/completed/207_unified_pre_flight_checklist_v2.md` (Wf-262 pre-flight checklist, gated by Wf-283C)
- `/home/hugo/codes/mol-metal/paper/main.tex` (target for Wf-283D §9.AG + §9.5 wording fix)

---

**End of Wf-283E synthesis.**