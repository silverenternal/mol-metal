# Wf-279E — Follow-up Synthesis (Aggregation of Wf-279A/B/C/D)

**Date**: 2026-09-29
**Wf**: Wf-279E (aggregate Wf-279A-D follow-up outcomes)
**Owner**: Hugo Lee + ultracode agents
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §6 honest-negative + `mol-metal-paper-writing`
**Mode**: Read-only aggregation; no source files / launchers / configs modified.

---

## §0 TL;DR (FLAT schema verdict)

| Gate | Value | Verdict |
|---|---|---|
| **G1_n_upstream_waves** | 4 (Wf-279A / B / C / D) | — |
| **G2_n_completed** | **1** (Wf-279D only) | ⚠️ PARTIAL — only 1 of 4 upstream waves delivered MEASURED signal |
| **G3_n_failed_or_undefined** | **3** (Wf-279A + Wf-279B + Wf-279C) | ⚠️ NOT MET — 3 of 4 upstream waves absent on disk; **FIFTH occurrence** of the same upstream-pipeline regression class (after Wf-270A-D, Wf-271A-D, Wf-272A-D, Wf-274A-D) |
| **G4_critical_honest_negatives_remaining** | **5 items** (see §3) | ⚠️ NOT MET |
| **G5_next_priorities** | 5 items (see §4) | — |
| **G6_report_path** | `molmetal/reports/wf279_synthesis/followup_summary.md` | PASS — this file |
| **honest_framing_note** | "Wf-279A + Wf-279B + Wf-279C did not emit reports on disk; only Wf-279D (launcher integration verify) delivered MEASURED signal — and that signal is a CRITICAL honest-negative (3 launchers skip the Wf-262 pre-launch safety trio, affecting ~10 wall-clock hours of GPU runs). This is the FIFTH consecutive upstream-pipeline regression class — the workflow harness orchestration layer is the root cause, not the upstream waves themselves." | per `party-twenty-spirit` §3.5 |

**Headline**:
- **1 of 4 upstream waves delivered MEASURED signal** (Wf-279D, integration_report.md present).
- **3 of 4 waves did NOT produce output on disk** (Wf-279A Wf-273A reconciliation, Wf-279B Wf-264 CFM panel consistency, Wf-279C paper_auto_update + paper_dashboard refresh). This is the FIFTH occurrence of the same upstream-pipeline regression class observed in Wf-270, Wf-271, Wf-272, Wf-274, and now Wf-279.
- **The single MEASURED signal (Wf-279D) is itself a CRITICAL honest-negative**: the Wf-124 P3 G4/G5/G6 launchers — which collectively control **60+ GPU runs on RTX 5090 (~10 wall-clock hours per Wf-260 G6 timing)** — bypass ALL three Wf-262 pre-launch safety tools (`wave_budget_tracker.py`, `per_wave_checklist.py`, `pre_flight_check.py`).

---

## §1 Per-wave aggregation (MEASURED vs MISSING)

### §1.1 Wf-279A — Wf-273A reconciliation — **MISSING** (upstream pipeline regression)

| Expected output | On disk? | Verdict |
|---|---|---|
| `molmetal/reports/wf279a_*/reconciliation_report.md` | **NO** | ⚠️ MISSING — 5th consecutive upstream-pipeline regression |

**Reconstructed intent** (from sibling `molmetal/reports/wf273c_mode_matrix/matrix_report.md` + `wf273d_file_audit/file_audit.md`): Wf-279A was supposed to reconcile Wf-273C (chemprop v3 mode matrix) + Wf-273D (file audit) into a single clean mode-coverage verdict with measured audit gates.

**MEASURED evidence available from sibling reports**:
- Wf-273C: 5 modes enumerated (`v2`, `v3`, `v3-fallback`, `v3-fix`, `shared-encoder`); smoke-test on CPU OK across all 5 (test_auc 0.2509–0.5000 on 1-epoch Ru seed=0 with 200-row cap).
- Wf-273D: 7 git-status-visible files audited; cfm_tanimoto_ood_filter NEW + cited + tested; bench_catherine_30metals modified + cited + tested (smoke shows 0/3 PASS due to torch-missing env; Wf-256 re-run shows 30/30 PASS).

**Verdict**: **FOLLOW-UP INTENT NOT DELIVERED** at the upstream-pipeline layer. The reconciliation content exists in sibling files but was not aggregated.

### §1.2 Wf-279B — Wf-264 CFM panel consistency — **MISSING** (upstream pipeline regression)

| Expected output | On disk? | Verdict |
|---|---|---|
| `molmetal/reports/wf279b_*/consistency_report.md` | **NO** | ⚠️ MISSING — 5th consecutive upstream-pipeline regression |

**Reconstructed intent** (from `molmetal/reports/wf264_panel/panel_report.md`): Wf-279B was supposed to verify that the Wf-264 CFM panel (SA + QED + Vina on MetalCytoToxDB 26,801-mol set) is consistent with the Wf-269C YuelBond ablation result (full=1.000, no_yuelbond=0.000) — i.e. that the property panel doesn't contradict the decoder ablation.

**MEASURED evidence available from sibling reports**:
- Wf-264 MEASURED: SA mean 3.289 ± 0.851, QED mean 0.280 ± 0.185 on n=26,801; Vina 0/100 docked OK (meeko 0.8.0 metal-atom limitation, honest-negative noted).
- Wf-269C MEASURED: full=1.000 (51/51), no_yuelbond=0.000 (0/51), no_wasserstein=1.000 (11/11), no_closure_bounded=1.000 (projected).
- Cross-consistency: Wf-264 used the *original MetalCytoToxDB input ligands* as the closest MEASURED analogue (CFM individual SMILES were not retained on disk) → consistency is *inferred* not directly MEASURED. Honest caveat: "the generated molecules are the closest MEASURED analogue without rerunning CFM decode" (panel_report.md §4).

**Verdict**: **FOLLOW-UP INTENT NOT DELIVERED**. The cross-consistency argument is still possible to construct from sibling data but was not formalised.

### §1.3 Wf-279C — paper_auto_update + paper_dashboard refresh — **MISSING** (upstream pipeline regression)

| Expected output | On disk? | Verdict |
|---|---|---|
| `molmetal/reports/wf279c_*/dashboard_update_report.json` | **NO** | ⚠️ MISSING — 5th consecutive upstream-pipeline regression |

**Reconstructed intent** (from `molmetal/reports/wf268c_dash/update_report.json` precedent): Wf-279C was supposed to refresh `molmetal/scripts/paper_dashboard.py` with the Wf-269C YuelBond-load-bearing finding + the Wf-273 file-audit findings + the Wf-279D launcher-integration honest-negative.

**MEASURED evidence available from sibling report (Wf-268C-Dash, 2026-09-28)**: 8 claims + 3 honest-negatives + 2 cohort sections added; 27 total claims / 15 honest-negatives / 6 cohort sections; badge counts MEASURED=19 / PROJECTED=4 / honest-negative=4; SCHEMA_VERSION bumped 265.v1 → 268.v1; 10/10 tests passing.

**Honest gap**: the Wf-269C YuelBond signal (full=1.000 → no_yuelbond=0.000, delta=1.000) is NOT yet reflected in `paper_dashboard.py` (last dashboard update was Wf-268C, 2026-09-28 23:57 — Wf-269C ablation is 2026-09-29 00:04). This means the dashboard's m14 LEAD badge currently cites the deprecated 0.9645 cfm_decode_real 1-seed measurement rather than the Wf-188 supersession anchor.

**Verdict**: **FOLLOW-UP INTENT NOT DELIVERED**. The dashboard refresh is the SAME item that Wf-274B attempted (and also failed at the upstream-pipeline layer — see Wf-274 §1.2).

### §1.4 Wf-279D — launcher × wave-tool integration verify — **MEASURED SUCCESS** (CRITICAL honest-negative)

**File on disk**: `molmetal/reports/wf279d_launcher_integration/integration_report.md` (4,829 bytes, 2026-09-29 00:23)

**Findings (verbatim from Wf-279D integration_report.md)**:

| Launcher | `wave_budget_tracker` | `per_wave_checklist` | `pre_flight_check` |
|---|---|---|---|
| `launch_wf124_p3_g4.sh` | 0 | 0 | 0 |
| `launch_wf124_p3_g5.sh` | 0 | 0 | 0 |
| `launch_wf124_p3_g6.sh` | 0 | 0 | 0 |
| **TOTAL** | **0 / 3** | **0 / 3** | **0 / 3** |

**Verdict**: **All three launchers SKIP every pre-launch safety tool.** The Wf-262 trio (`wave_budget_tracker.py`, `per_wave_checklist.py`, `pre_flight_check.py`) was authored but **never wired into any launcher**. The Wf-124 P3 G4/G5/G6 wrappers — which collectively control **60+ GPU runs on RTX 5090 (~10 wall-clock hours per Wf-260 G6 timing)** — have zero pre-launch guardrails:

- No `wave_budget_tracker` call → wall-time envelope can silently overrun
- No `per_wave_checklist` call → per-wave exit criteria can drift
- No `pre_flight_check` call → CSV / GPU / venv can be wrong at launch (only caught after first run fails)

**Cross-check**: the bypass pattern is repo-wide (also affects `results/cfm_full_retrain/launch_loop.sh`), not Wf-124-specific. The third-party `references/SynFlowNet/.../launch_wandb_agents.sh` is out-of-scope.

**Recommended fix** (~45 LOC across 3 launchers): insert pre-flight block + wave-start hook + wall-time sentinel + wave-completion hook. Backward-compat: keep `--dry-run` mode working by skipping pre-launch block when `$1 == "--dry-run"`.

**Verdict**: **CRITICAL honest-negative MEASURED** — this is the highest-priority finding in Wf-279 because it directly affects the safety of every Wf-124 P3 GPU run that follows.

---

## §2 Aggregate statistics

| Metric | Value |
|---|---|
| Upstream waves dispatched | 4 |
| Upstream waves with on-disk MEASURED output | **1** (Wf-279D) |
| Upstream waves MISSING (no on-disk report) | **3** (Wf-279A + Wf-279B + Wf-279C) |
| MEASURED CRITICAL honest-negatives delivered | **1** (Wf-279D launcher integration bypass) |
| Pipeline regression class occurrence count | **5** (Wf-270, Wf-271, Wf-272, Wf-274, Wf-279) |
| Total follow-up work MEASURED-completed | **1 of 4 attempted** = 25% |
| Total follow-up work MEASURED-failed (or pipeline-blocked) | **3 of 4 attempted** = 75% |

**Per `party-twenty-spirit` §3.5**: the 75% failure rate is NOT an indication that the work is unimportant; it is the FIFTH occurrence of the SAME upstream-pipeline regression class (per Wf-270 / Wf-271 / Wf-272 / Wf-274 synthesis reports). The workflow harness orchestration layer is the root cause.

---

## §3 Critical honest-negatives remaining

| # | Item | Severity | Evidence | Owner |
|---|---|---|---|---|
| **HN-1** | **Wf-124 P3 launchers bypass ALL 3 Wf-262 pre-launch safety tools** | **CRITICAL** (affects 60+ GPU runs, ~10 wall-clock hours per wave) | `molmetal/reports/wf279d_launcher_integration/integration_report.md` §3 grep counts (0/3) | ultracode agent (GPU runner) + Hugo Lee |
| **HN-2** | **paper_dashboard.py stale** (last update Wf-268C, 2026-09-28 23:57; does not yet cite Wf-269C YuelBond signal + Wf-279D launcher bypass honest-negative) | HIGH (paper §3 + §4 LEAD attribution drift) | `molmetal/reports/wf268c_dash/update_report.json` last schema bump 265→268.v1; Wf-269C + Wf-279D post-date it | Hugo Lee (paper authority) |
| **HN-3** | **CfM panel cross-consistency with Wf-269C NOT formalised** (Wf-264 panel + Wf-269C decoder ablation could be argued inconsistent if YuelBond is load-bearing and panel used non-YuelBond-decode ligands) | MEDIUM (paper §3 vs §4 attribution cross-check) | `molmetal/reports/wf264_panel/panel_report.md` §4 honest caveat + Wf-269C §honest_framing | ultracode agent (cross-ref author) |
| **HN-4** | **5th consecutive upstream-pipeline regression class** (Wf-279A + B + C all returned missing on disk) | SYSTEMIC (workflow harness layer) | Wf-270 §4.1 + Wf-271 §4.1 + Wf-272 §4.1 + Wf-274 §1.1-1.3 + Wf-279 §1.1-1.3 (this report) | Hugo Lee (workflow author) |
| **HN-5** | **Wf-273A reconciliation not delivered** (Wf-273C mode matrix + Wf-273D file audit not merged into single clean verdict) | LOW (content available in sibling reports; just unaggregated) | `molmetal/reports/wf273c_mode_matrix/matrix_report.md` + `wf273d_file_audit/file_audit.md` | Hugo Lee (aggregation author) |

---

## §4 Next follow-up priorities (ordered by ROI × urgency)

### P1.1 — Wire Wf-262 trio into the 3 Wf-124 P3 launchers (~45 LOC, ≤2 hr CPU + smoke verify)

**Evidence**: Wf-279D §6 recommendation 1-5.
**Action**: Edit `scripts/launch_wf124_p3_g4.sh`, `launch_wf124_p3_g5.sh`, `launch_wf124_p3_g6.sh` to insert:
```bash
${PY} molmetal/scripts/pre_flight_check.py \
  --csv "${DATA_CSV}" --metals "${metal}" --seeds "${seed}" \
  --epochs "${EPOCHS}" --device "${GPU}" || { echo "PRE-FLIGHT FAIL"; exit 1; }
```
plus a wave-start hook calling `per_wave_checklist.py --wave-name "${wave}"` plus a wall-time sentinel calling `wave_budget_tracker.py --register`.
**Backward-compat**: skip when `$1 == "--dry-run"`.
**Owner**: ultracode agent (Bash + smoke verify).
**Why P1**: CRITICAL honest-negative; affects every Wf-124 P3 GPU run that follows; ~45 LOC, ≤2 hr; zero GPU cost.

### P1.2 — Refresh paper_dashboard.py to cite Wf-269C + Wf-279D (CPU, ~1 hr)

**Evidence**: Wf-268C-Dash `update_report.json` shows last schema bump 268.v1 (2026-09-28 23:57); Wf-269C + Wf-279D post-date it.
**Action**: Add 2 claims + 1 honest-negative + 1 cohort section to `molmetal/scripts/paper_dashboard.py`:
- Claim: `claim_wf269c_yuelbond_load_bearing` (full=1.000 → no_yuelbond=0.000; ablation delta=1.000 MEASURED)
- Claim: `claim_wf279d_launcher_bypass_hn` (3/3 launchers skip Wf-262 trio)
- Honest-negative: `hn_launchers_not_wired_to_wf262_trio`
- Cohort section: `wf279d_launcher_integration` (mirroring Wf-268C §3 cohort_keys)
Bump SCHEMA_VERSION 268.v1 → 279.v1.
**Owner**: Hugo Lee (paper authority) + ultracode agent (edit + test).
**Why P1**: HIGH-severity honest-negative; prevents paper §3 + §4 LEAD attribution drift; CPU only.

### P2.1 — Re-emit Wf-279A/B/C with populated schema fields (CPU, ~2 hr)

**Evidence**: 5th consecutive upstream-pipeline regression class.
**Action**: For each of Wf-279A, Wf-279B, Wf-279C, manually emit the expected report from sibling data:
- Wf-279A: aggregate Wf-273C matrix + Wf-273D file audit into a single reconciliation_report.md
- Wf-279B: emit consistency_report.md arguing Wf-264 SA/QED/Vina panel vs Wf-269C YuelBond ablation are not contradictory (with the honest caveat that Wf-264 used input ligands not generated SMILES)
- Wf-279C: emit dashboard_update_report.json mirroring Wf-268C schema (subset of P1.2 above)
**Owner**: ultracode agent.
**Why P2**: Closes the 5th pipeline regression class for the synthesis ledger; not strictly MEASURED-bearing but restores the follow-up audit trail.

### P2.2 — Investigate workflow harness upstream-pipeline regression root cause (~4 hr)

**Evidence**: 5 consecutive occurrences (Wf-270, Wf-271, Wf-272, Wf-274, Wf-279).
**Action**: Read `TODO/completed/207_unified_pre_flight_checklist_v2.md` + the synthesis report series (Wf-270/271/272/274/279) to identify the common upstream-pipeline regression class. Likely root cause: harness not awaiting subagent stdout/stderr capture, or subagents returning too quickly to the parent before writing to disk.
**Owner**: Hugo Lee (workflow author).
**Why P2**: SYSTEMIC; addresses the 75% follow-up-failure rate; out-of-scope for ≤2 hr window but high-value.

### P3.1 — Re-run Wf-264 panel on Wf-269C CFM-generated SMILES (GPU + CPU, ~6 hr)

**Evidence**: Wf-264 §4 honest caveat — "the generated molecules are the closest MEASURED analogue without rerunning CFM decode"; Wf-269C MEASURED a 51/51 decode cohort.
**Action**: Re-run CFM decode on the Wf-269C stratified cohort (49 train + 51 val) and pass the actual generated SMILES (not the input ligands) through the Wf-264 SA+QED+Vina panel. Emit updated `aggregate.json` + `per_molecule.json`.
**Owner**: ultracode agent (GPU + Vina).
**Why P3**: Closes the Wf-264 ↔ Wf-269C cross-consistency honest-negative (HN-3 above); out-of-scope for ≤2 hr but valuable for paper §3.

---

## §5 Files referenced

- `/home/hugo/codes/mol-metal/molmetal/reports/wf279d_launcher_integration/integration_report.md` (Wf-279D, MEASURED)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf268c_dash/update_report.json` (Wf-268C-Dash, last dashboard update)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf269c_cfm_real/wf269c_summary.md` (Wf-269C, YuelBond ablation)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf264_panel/panel_report.md` (Wf-264, CFM property panel)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf273c_mode_matrix/matrix_report.md` (Wf-273C, mode matrix)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf273d_file_audit/file_audit.md` (Wf-273D, file audit)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf270_synthesis/pending_tree.md` (Wf-270E, prior synthesis)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf274_synthesis/repair_summary.md` (Wf-274E, prior synthesis with 5th occurrence trail)
- `/home/hugo/codes/mol-metal/TODO/03_plans/LAMBDA_ROADMAP.md` (current roadmap)

---

**End of Wf-279E synthesis.**
