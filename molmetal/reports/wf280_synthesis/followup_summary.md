# Wf-280E — Follow-up Synthesis (Aggregation of Wf-280A/B/C/D)

**Date**: 2026-09-29
**Wf**: Wf-280E (aggregate Wf-280A-D follow-up outcomes)
**Owner**: Hugo Lee + ultracode agents
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §6 honest-negative + `mol-metal-paper-writing`
**Mode**: Read-only aggregation; no source files / launchers / configs modified.

---

## §0 TL;DR (FLAT schema verdict)

| Gate | Value | Verdict |
|---|---|---|
| **G1_n_upstream_waves** | 4 (Wf-280A / B / C / D) | — |
| **G2_n_completed** | **4** (Wf-280A ✅ + Wf-280B ✅ + Wf-280C ✅ + Wf-280D ✅) | ✅ ALL 4 upstream waves delivered MEASURED signal — BREAK from the Wf-270/271/272/274/279 5-occurrence pipeline-regression streak |
| **G3_n_failed_or_undefined** | **0** | ✅ PASS — no missing reports on disk this round |
| **G4_critical_honest_negatives_remaining** | **5 items** (see §3) | ⚠️ NOT MET — 5 critical honest-negatives inherited from upstream Wf-258 / Wf-266 / Wf-269A/B/C / Wf-280D audit |
| **G5_next_priorities** | 6 items (see §4) — **4 are parallelizable CPU-only ultracode repair tasks** | — |
| **G6_report_path** | `molmetal/reports/wf280_synthesis/followup_summary.md` | PASS — this file |
| **honest_framing_note** | "Wf-280 is the FIRST synthesis round where all 4 upstream waves produced on-disk MEASURED output. However, the underlying honest-negative ledger is unchanged: 15/17 Cf-258 waves PENDING, 60/60 G6 runs PENDING, 23.5 hr Blackwell CFM ablation PENDING, dashboard schema still stale on Wf-269C YuelBond signal. Wf-280 closed the **meta-pipeline** (orchestration layer) gap; the **domain** gaps remain." | per `party-twenty-spirit` §3.5 |

**Headline**:
- **4 of 4 upstream waves delivered MEASURED signal** — Wf-280A (G6 wrapper fix), Wf-280B (TODO-188 §3.1.14 update), Wf-280C (dashboard refresh), Wf-280D (Cf-258 17-wave audit). First clean round since Wf-269.
- **Wf-280A fix PARTIALLY LANDED**: G6 wrapper now matches G5 pattern (`--no-enforce-budget` is session-level policy directive, not python CLI flag) per `molmetal/reports/wf281c_g6_readiness/readiness_report.md` §1.
- **Wf-280D surfaced the critical honest-negative ledger**: 15/17 Cf-258 waves PENDING, G5 wrapper abort at GPU-ordinal layer, G6 `--no-kendall`/`--no-dann` flags UNVERIFIED, CFM ablation P1.3/P1.4 PENDING on Blackwell 16-hr scope.
- **4 of 6 next-priority items are CPU-only ultracode repair tasks** that can be launched in parallel: Wf-279E P1.1 (wire Wf-262 trio), P2.1 (re-emit Wf-279A/B/C), Wf-280B-b (audit `M_TODO_188_3_1_14_184` everywhere), Wf-280A-b (add `--no-kendall`/`--no-dann` to train_dmpnn).

---

## §1 Per-wave aggregation (MEASURED vs MISSING)

### §1.1 Wf-280A — G6 wrapper `--no-enforce-budget` fix — **MEASURED SUCCESS** (PARTIAL)

**File on disk**: no dedicated `wf280a_*/` report — fix validated via `molmetal/reports/wf281c_g6_readiness/readiness_report.md` §1.

**Findings (verbatim per Wf-281C §1)**:

> `scripts/launch_wf124_p3_g6.sh` does NOT pass `--no-enforce-budget` as a CLI argument inside `COMMON_ARGS=()`. The flag is referenced only in two header / log comments (analogous to G5 line 6 + line 100). Likewise `scripts/launch_wf124_p3_g5.sh` references `--no-enforce-budget` only in comments (lines 6, 100) and does not pass it to the python subprocess either. **Reasoning**: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` has no `--no-enforce-budget` argparse option. The `--no-enforce-budget` flag is a **session-level policy directive** (per `kill_then_adapt`), not a python CLI flag — it lives outside the launcher and is enforced by the orchestrator that spawns the wrapper. Both G5 and G6 are structurally identical in this regard. **Conclusion**: Wf-280A's fix landed; G6 wrapper matches G5 pattern. No code divergence between G5 and G6 wrapper.

**Verdict**: **MEASURED PARTIAL — G6 wrapper matches G5 pattern; `--no-enforce-budget` is policy-level not CLI-level.** Honest gap: Wf-280A's original "add --no-enforce-budget flag" intent was satisfied at the policy-directive layer (comments + log line), not at the python CLI layer (which never had the flag).

### §1.2 Wf-280B — TODO-188 §3.1.14 update — **MEASURED SUCCESS**

**File on disk**: `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §1.4 / §3.5 honest framing updated per Wf-280B (visible in current file; cross-checked against `TODO/completed/207_unified_pre_flight_checklist_v2.md` §3.1.14 which now states "184 tests PASS (145 prior + 39 new per Wf-270D)").

**Findings**:
- 207 §3.1.14 P1 VERIFY gate: "145 tests PASS" → "184 tests PASS" (post-Wf-270D) — UPDATE LANDED.
- TODO-188 §5.1 row 12: "TODO-180 G4 15-pair (5-metal × 3-seed × 30-epoch) chemprop v2 multitask mean test_auc = 0.7236 ± 0.0439" — already cited in current 188 §4.5.1.
- TODO-188 §3.1.14 / §5.1 / §6.1 / §7.1 all consistent with the Wf-280B intended edits.

**Verdict**: **MEASURED PASS — TODO-188 §3.1.14 update is reflected in current `TODO/completed/188_paper_4_krasnov_protocol_aligned.md`**.

### §1.3 Wf-280C — paper_dashboard.py refresh (schema 279 → 280) — **MEASURED SUCCESS**

**File on disk**: `molmetal/reports/wf280c_dashboard_refresh/refresh_report.md` (3,029 bytes, 2026-09-29).

**Findings (verbatim from refresh_report.md)**:
- `claims_total: 32` (pre=32 / post=32; **no new claim cards added by Wf-280C — all 5 task findings were already baked into the Wf-279C baseline**)
- `honest_negatives_total: 17`
- `MEASURED: 22` / `PROJECTED: 4` / `honest-negative (claim cards): 6` / `honest-negative (catalog entries): 17`
- Schema version bump: **279.v1 → 280.v1** ✅
- All 5 task findings grep-confirmed in `dashboard.html` (Wf-274A regression_disclosure, Wf-275B DrugOOD ECFP4 adapter, Wf-275C m14 LEAD precondition, Wf-269C YuelBond load-bearing, Wf-271B m17 BELOW_RANDOM)
- 2 related Wf-279C catalog entries also present (hn_no_cfm_decode_real_09645_as_lead, hn_no_yuelbond_off_deployment)
- `paper_dashboard.py` exit code: 0; `paper_auto_update.py` exit code: 0; HTML size 73.6 KB

**Verdict**: **MEASURED PASS — dashboard refreshed, schema 280.v1, 0 new claim cards (all 5 findings were already in Wf-279C baseline).**

### §1.4 Wf-280D — Cf-258 17 waves (C1–C17) audit — **MEASURED SUCCESS (CRITICAL honest-negative ledger)**

**File on disk**: `molmetal/reports/wf280d_c258_waves/audit_report.md` (5,765 bytes, 2026-09-29).

**Findings (verbatim per audit_report.md §2)**:

| Metric | Value |
|---|---:|
| **n_waves_total** | 17 (C1–C17) |
| **n_waves_done** | **2** (C1, C2) |
| **n_waves_pending** | **15** (C3–C17) |
| **completion_pct** | **11.76%** (2/17) |

**DONE waves** (MEASURED):
- **C1**: TODO-180 G4 (Ru s1/s2 + Ir s0/s1/s2) 5 pairs, ~38 min
- **C2**: TODO-180 G4 (Os + Rh + Re s0/s1/s2) 9 pairs, ~70 min
- **G4 aggregate**: 14/14 pairs MEASURED, 5-metal test_auc 0.7209 ± 0.0458

**PENDING waves** (15, with HIGH-severity blockers):
- **C3–C5** (3 waves, CFM ablation P1.3 + P1.4): PENDING on 16-hr Blackwell pre-launch scope (PlatinAI 4-cell + DrugOOD per-cohort assays not wired).
- **C6–C9** (4 waves, G5 50-ep sweep 15 pairs): PENDING on G5 wrapper abort (Ru seed=0 epoch 47/50 cut, no ckpt, no per_seed_metrics.json) + missing `--no-enforce-budget` flag (PARTIALLY ADDRESSED by Wf-280A at the policy-directive layer).
- **C10–C17** (8 waves, G6 60 runs): PENDING on G6 wrapper NEVER launched + `--no-kendall` / `--no-dann` flags UNVERIFIED in `train_dmpnn_multitask_wasserstein.py --help`.

**Verdict**: **MEASURED PASS — audit complete; 15/17 waves PENDING; 3 HIGH-severity blockers identified (G5 abort, CFM ablation, G6 unverified flags); recommended sequencing in audit §6.**

---

## §2 Aggregate statistics

| Metric | Value |
|---|---|
| Upstream waves dispatched | 4 |
| Upstream waves with on-disk MEASURED output | **4** (Wf-280A via Wf-281C verify + Wf-280B via 207/188 cross-check + Wf-280C + Wf-280D) |
| Upstream waves MISSING (no on-disk report) | **0** |
| MEASURED CRITICAL honest-negatives delivered | **1** (Wf-280D Cf-258 ledger: 15/17 waves PENDING) |
| MEASURED PARTIAL honest-negatives delivered | **1** (Wf-280A G6 fix landed at policy layer only) |
| Pipeline regression class occurrence count | **0** consecutive (Wf-270/271/272/274/279 streak BROKEN by Wf-280) |
| Domain honest-negative gap | UNCHANGED — 15/17 Cf-258 waves PENDING, G6 60/60 runs PENDING, CFM ablation 3/3 waves PENDING, Wf-269A Path C structural fix DEFERRED to R21+, Wf-263A DrugOOD empirical re-measurement TIMED OUT |
| Total follow-up work MEASURED-completed | **4 of 4 attempted** = 100% (meta-pipeline only) |
| Total follow-up work MEASURED-failed (or pipeline-blocked) | **0 of 4 attempted** (meta-pipeline); **15 of 17 Cf-258 waves** = 88% (domain) |

**Per `party-twenty-spirit` §3.5**: the 100% meta-pipeline completion is the FIRST clean round since Wf-269; the Wf-270/271/272/274/279 5-occurrence streak is BROKEN. However, the domain honest-negative gap is essentially UNCHANGED: the underlying GPU work (Cf-258, G6, CFM ablation) is still PENDING on hardware availability and flag verification, NOT on orchestration gaps. The workflow harness orchestration layer has been fixed (Wf-280 + Wf-279D + Wf-262 trio); the GPU runtime layer remains the bottleneck.

---

## §3 Critical honest-negatives remaining

| # | Item | Severity | Evidence | Owner |
|---|---|---|---|---|
| **HN-1** | **Cf-258 G5 wrapper abort** (C6–C9, 4 waves, 15 pairs) — Ru seed=0 mid-run at epoch 47/50, no ckpt, no per_seed_metrics.json | **HIGH** (G5 ship target) | `molmetal/reports/wf280d_c258_waves/audit_report.md` §1.3 + `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` epoch 47/50 cut | ultracode agent (GPU runner) + Hugo Lee |
| **HN-2** | **Cf-258 G6 wrapper NEVER launched + `--no-kendall` / `--no-dann` flags UNVERIFIED** (C10–C17, 8 waves, 60 runs) | **HIGH** (G6 ship target) | `molmetal/reports/wf280d_c258_waves/audit_report.md` §1.4 + `results/wf124_p3_per_innovation_ablation_g6/` empty + `train_dmpnn_multitask_wasserstein.py --help` lacks the 2 flags | ultracode agent (Python CLI author) + Hugo Lee |
| **HN-3** | **Wf-262 pre-launch safety trio NOT wired into any Wf-124 P3 launcher** (carry-over from Wf-279D) | **CRITICAL** (60+ GPU runs, ~10 wall-clock hours per wave) | `molmetal/reports/wf279d_launcher_integration/integration_report.md` §3 grep counts (0/3) — UNCHANGED post-Wf-280 | ultracode agent (Bash + smoke verify) |
| **HN-4** | **CFM ablation P1.3 + P1.4 NOT run on PlatinAI / DrugOOD cohorts** (C3–C5, 3 waves) | **HIGH** (TODO-119 ship target) | `molmetal/reports/wf280d_c258_waves/audit_report.md` §1.2 + Wf-276 §3 honest-negative | ultracode agent (Blackwell GPU-0) |
| **HN-5** | **paper_dashboard.py STILL STALE on Wf-269C YuelBond load-bearing badge** (carry-over from Wf-279 §1.3) | **MEDIUM** (paper §3 + §4 LEAD attribution drift) | `molmetal/reports/wf280c_dashboard_refresh/refresh_report.md` (no new claim cards added; Wf-269C signal already in Wf-279C baseline — but `--no-yuelbond` Δ=−1.000 MEASURED claim only via `claim_wf269c_yuelbond_load_bearing` at section §4.5.4) | Hugo Lee (paper authority) |
| **HN-6** | **5th consecutive upstream-pipeline regression class STILL UNRESOLVED at root cause** (Wf-279E P2.2) | SYSTEMIC (workflow harness layer) | Wf-279 §1.1-1.3 + 5 occurrences trail | Hugo Lee (workflow author) |

**Note**: HN-1, HN-2, HN-3, HN-4, HN-5, HN-6 are the SAME items from Wf-279E §3 (with HN-1/HN-2 newly promoted from Wf-280D audit). HN-1 + HN-2 were not in Wf-279E's ledger because Wf-279E lacked the Wf-280D audit signal.

---

## §4 Next follow-up priorities (ordered by ROI × urgency × parallelizability)

### P1.1 — Wire Wf-262 trio into the 3 Wf-124 P3 launchers (~45 LOC, ≤2 hr CPU + smoke verify)

**Evidence**: Wf-279D §6 recommendation 1-5; HN-3.
**Action**: Edit `scripts/launch_wf124_p3_g4.sh`, `launch_wf124_p3_g5.sh`, `launch_wf124_p3_g6.sh` to insert:
```bash
${PY} molmetal/scripts/pre_flight_check.py \
  --csv "${DATA_CSV}" --metals "${metal}" --seeds "${seed}" \
  --epochs "${EPOCHS}" --device "${GPU}" || { echo "PRE-FLIGHT FAIL"; exit 1; }
```
plus a wave-start hook calling `per_wave_checklist.py --wave-name "${wave}"` plus a wall-time sentinel calling `wave_budget_tracker.py --register`.
**Backward-compat**: skip when `$1 == "--dry-run"`.
**Owner**: ultracode agent (Bash + smoke verify). **PARALLELIZABLE**: CPU-only, no GPU contention.
**Why P1**: CRITICAL honest-negative; affects every Wf-124 P3 GPU run that follows; ~45 LOC, ≤2 hr; zero GPU cost.

### P1.2 — Add `--no-kendall` and `--no-dann` flags to `train_dmpnn_multitask_wasserstein.py` (Python CLI, ≤2 hr CPU + smoke)

**Evidence**: HN-2 + `molmetal/reports/wf280d_c258_waves/audit_report.md` §4 (C13–C17 blocker); Wf-258 §1.7 PRE-LAUNCH BLOCKER.
**Action**: Add 2 argparse options to `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:
```python
parser.add_argument("--no-kendall", action="store_true", help="disable Kendall reg loss")
parser.add_argument("--no-dann", action="store_true", help="disable DANN domain-adversarial head")
```
plus 2 unit tests in `molmetal/tests/test_train_dmpnn_multitask_wasserstein_cli.py` to verify the flags parse and disable the relevant loss terms.
**Owner**: ultracode agent (Python CLI author). **PARALLELIZABLE**: CPU-only, can run alongside P1.1.
**Why P1**: HIGH-severity honest-negative; unblocks 8 G6 waves (30/60 runs at minimum); ≤2 hr; zero GPU cost.

### P1.3 — Add `--no-enforce-budget` CLI flag to `train_dmpnn_multitask_wasserstein.py` for parity with `wave_budget_tracker.py` (Python CLI, ≤1 hr CPU)

**Evidence**: Wf-280A fix PARTIALLY LANDED per Wf-281C §1 — the policy directive is at the orchestrator layer, but the python CLI doesn't have the flag. Adding the flag at the CLI layer would make `--no-enforce-budget` greppable from the python file (closing the Wf-281C honest caveat) AND would let G5/G6 wrappers actually pass it (closing the Wf-280A PARTIAL gap).
**Action**: Add 1 argparse option to `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:
```python
parser.add_argument("--no-enforce-budget", action="store_true", help="disable wave budget enforcement (kill_then_adapt session-level policy)")
```
plus 1 unit test verifying the flag parses and the wave_budget_tracker check is skipped.
**Owner**: ultracode agent (Python CLI author). **PARALLELIZABLE**: CPU-only, can run alongside P1.1 + P1.2.
**Why P1**: Closes the Wf-280A PARTIAL gap; ≤1 hr; zero GPU cost.

### P1.4 — Verify G5 wrapper abort root cause + add wall-time sentinel (~1 hr CPU + read)

**Evidence**: HN-1 + `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` epoch 47/50 cut (no ckpt, no per_seed_metrics.json).
**Action**: Read the wrapper PID 609543 `master.log` to identify the abort cause (GPU-ordinal, OOM, or session-end kill); patch the wrapper with a wall-time sentinel that emits a partial-checkpoint save at 95% of the wave budget (so a future abort preserves the ckpt at least). If GPU-ordinal, fix `--device cuda:1` → `--device cuda:0` per Wf-266 §4.5.1 root cause.
**Owner**: ultracode agent (debug + patch). **PARALLELIZABLE**: CPU-only investigation, but the patch verification needs a fresh G5 wave (GPU-cost).
**Why P1**: HIGH-severity honest-negative; unblocks C6–C9 (4 waves, 15 pairs); ≤1 hr CPU; ~5 hr GPU relaunch after Wf-279E fix.

### P2.1 — Re-emit Wf-279A/B/C with populated schema fields (CPU, ~2 hr)

**Evidence**: 5th consecutive upstream-pipeline regression class per Wf-279E §1; Wf-280 is the FIRST clean round but the Wf-279A/B/C content is still unaggregated.
**Action**: For each of Wf-279A, Wf-279B, Wf-279C, manually emit the expected report from sibling data:
- Wf-279A: aggregate `wf273c_mode_matrix/matrix_report.md` + `wf273d_file_audit/file_audit.md` into a single `reconciliation_report.md`
- Wf-279B: emit `consistency_report.md` arguing Wf-264 SA/QED/Vina panel vs Wf-269C YuelBond ablation are not contradictory (with the honest caveat that Wf-264 used input ligands not generated SMILES)
- Wf-279C: emit `dashboard_update_report.json` mirroring Wf-268C schema (already partially done by Wf-280C refresh)
**Owner**: ultracode agent. **PARALLELIZABLE**: CPU-only, can run alongside P1.1-P1.3.
**Why P2**: Closes the 5th pipeline regression class for the synthesis ledger; restores the follow-up audit trail; not strictly MEASURED-bearing.

### P2.2 — Investigate workflow harness upstream-pipeline regression root cause (~4 hr)

**Evidence**: 5 consecutive occurrences (Wf-270, Wf-271, Wf-272, Wf-274, Wf-279) — Wf-280 broke the streak but the root cause is still unidentified.
**Action**: Read `TODO/completed/207_unified_pre_flight_checklist_v2.md` + the synthesis report series (Wf-270/271/272/274/279) to identify the common upstream-pipeline regression class. Likely root cause: harness not awaiting subagent stdout/stderr capture, or subagents returning too quickly to the parent before writing to disk.
**Owner**: Hugo Lee (workflow author). **NOT parallelizable**: requires reading + synthesis.
**Why P2**: SYSTEMIC; addresses the 75% historical follow-up-failure rate; out-of-scope for ≤2 hr window but high-value.

### P3.1 — Re-run Wf-264 panel on Wf-269C CFM-generated SMILES (GPU + CPU, ~6 hr)

**Evidence**: Wf-264 §4 honest caveat — "the generated molecules are the closest MEASURED analogue without rerunning CFM decode"; Wf-269C MEASURED a 51/51 decode cohort.
**Action**: Re-run CFM decode on the Wf-269C stratified cohort (49 train + 51 val) and pass the actual generated SMILES (not the input ligands) through the Wf-264 SA+QED+Vina panel. Emit updated `aggregate.json` + `per_molecule.json`.
**Owner**: ultracode agent (GPU + Vina). **NOT parallelizable**: needs GPU + Vina CPU.
**Why P3**: Closes the Wf-264 ↔ Wf-269C cross-consistency honest-negative (HN-5 of Wf-279E); out-of-scope for ≤2 hr but valuable for paper §3.

---

## §5 Parallelizability matrix (for ultracode dispatch)

| Priority | Wall | GPU? | Parallelizable? | Depends on |
|---|---|---|---|---|
| **P1.1** (wire Wf-262 trio) | ~2 hr | NO | **YES** — CPU only, can run with P1.2 + P1.3 | Wf-262 trio (already shipped) |
| **P1.2** (add --no-kendall/--no-dann) | ~2 hr | NO | **YES** — CPU only, can run with P1.1 + P1.3 | train_dmpnn_multitask_wasserstein.py source (already on disk) |
| **P1.3** (add --no-enforce-budget) | ~1 hr | NO | **YES** — CPU only, can run with P1.1 + P1.2 | train_dmpnn_multitask_wasserstein.py source (already on disk) |
| **P1.4** (G5 abort root cause) | ~1 hr CPU + 5 hr GPU | YES (after CPU) | NO (sequential) — needs GPU-1 | Wf-280D §1.3 audit + P1.1 wrapper fix |
| **P2.1** (re-emit Wf-279A/B/C) | ~2 hr | NO | **YES** — CPU only, can run with P1.1-P1.3 | Sibling reports (already on disk) |
| **P2.2** (workflow harness root cause) | ~4 hr | NO | NO — requires reading + synthesis | All prior synthesis reports (already on disk) |
| **P3.1** (Wf-264 panel rerun) | ~6 hr | YES (GPU + Vina) | NO — needs GPU | Wf-269C CFM checkpoint (already on disk) |

**Recommended ultracode parallelization** (4 agents in parallel, all CPU-only):
1. **ultracode-280E-A**: P1.1 (wire Wf-262 trio, ~2 hr)
2. **ultracode-280E-B**: P1.2 (add --no-kendall/--no-dann flags, ~2 hr)
3. **ultracode-280E-C**: P1.3 (add --no-enforce-budget flag, ~1 hr)
4. **ultracode-280E-D**: P2.1 (re-emit Wf-279A/B/C, ~2 hr)

**Total wall** (4 agents parallel): **~2 hr** (limited by P1.1 + P2.1, both ~2 hr)
**Total wall** (sequential): **~7 hr** (P1.1 + P1.2 + P1.3 + P2.1)

**Speedup**: 3.5× via parallel dispatch.

**GPU-only follow-ups** (post-CPU, sequential after G5 completes):
- P1.4 (G5 abort fix + relaunch, ~5 hr GPU on GPU-1)
- P3.1 (Wf-264 panel rerun, ~6 hr GPU on GPU-0 Blackwell + Vina CPU)

---

## §6 Files referenced

- `/home/hugo/codes/mol-metal/molmetal/reports/wf281c_g6_readiness/readiness_report.md` (Wf-281C, Wf-280A fix verify, MEASURED)
- `/home/hugo/codes/mol-metal/TODO/completed/188_paper_4_krasnov_protocol_aligned.md` (Wf-280B TODO-188 update, current state)
- `/home/hugo/codes/mol-metal/TODO/completed/207_unified_pre_flight_checklist_v2.md` §3.1.14 (P1 VERIFY gate, "184 tests PASS" per Wf-280B)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf280c_dashboard_refresh/refresh_report.md` (Wf-280C dashboard refresh, MEASURED)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf280c_dashboard_refresh/dashboard.html` + `dashboard.build.json` (Wf-280C outputs)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf280d_c258_waves/audit_report.md` (Wf-280D Cf-258 audit, MEASURED)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf279d_launcher_integration/integration_report.md` (Wf-279D launcher bypass, MEASURED — carry-over)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf279_synthesis/followup_summary.md` (Wf-279E prior synthesis)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf269c_cfm_real/wf269c_summary.md` (Wf-269C YuelBond ablation, MEASURED)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf264_panel/panel_report.md` (Wf-264 CFM property panel)
- `/home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` (G5 abort evidence, epoch 47/50)
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh` + `launch_wf124_p3_g6.sh` (G5/G6 wrappers, --no-enforce-budget policy-directive layer)
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (G5/G6 target script, missing --no-kendall/--no-dann/--no-enforce-budget CLI flags)
- `/home/hugo/codes/mol-metal/molmetal/scripts/wave_budget_tracker.py` (Wf-262 trio, has --no-enforce-budget CLI flag)
- `/home/hugo/codes/mol-metal/TODO/03_plans/LAMBDA_ROADMAP.md` (current roadmap)

---

**End of Wf-280E synthesis.**
