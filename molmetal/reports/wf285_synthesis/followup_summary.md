# Wf-285E — Synthesis: Wf-285 batch outcomes + follow-up priorities

**Date**: 2026-09-29
**Wf**: Wf-285E (aggregate Wf-285A/B/C/D follow-up outcomes)
**Owner**: Hugo Lee + workflow harness
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §6 honest-negative
**Mode**: Read-only aggregation; no source files / launchers / configs modified.

---

## §0 TL;DR

| Gate | Value | Verdict |
|---|---|---|
| **G1_n_upstream_waves** | 4 dispatched (Wf-285A / B / C / D) | — |
| **G2_n_with_on_disk_artefact** | **2** (Wf-285B ✅ + Wf-285C ✅) | ⚠️ 2/4 with reports; Wf-285A + Wf-285D NO on-disk artefact |
| **G3_n_failed** | **2** (Wf-285A + Wf-285D did not produce a report) | ⚠️ classified as MISSING-DELIVERABLE, not crash |
| **G4_critical_honest_negatives_remaining** | **3 items** (see §3) — Wf-285A redo not done, Wf-285D guard not in place, plus carry-over §9.AH delta mismatch | ⚠️ 3 HNs open |
| **G5_next_priorities** | 4 items (see §4) — 2 ultracode doc edits + 1 small code patch + 1 re-dispatch | — |
| **G6_report_path** | `molmetal/reports/wf285_synthesis/followup_summary.md` | PASS — this file |

**Honest framing (party-twenty-spirit §3.5)**: Wf-285B (Kendall UW + DANN GRL wiring) and Wf-285C (`--no-enforce-budget` CLI flag) both produced **MEASURED deliverables** with their own test suites green. Wf-285A and Wf-285D did NOT produce on-disk reports under `molmetal/reports/wf285a_*` or `molmetal/reports/wf285d_*` — this is a **MISSING-DELIVERABLE** honest-negative, not a hidden crash. The synthesis is built only from what is on disk; the two missing waves are re-dispatched as follow-ups.

---

## §1 Per-Agent Summary

### §1.1 Wf-285A — §9.AH wrong values redo — **MISSING-DELIVERABLE**

**Expected artefact**: `molmetal/reports/wf285a_section9_ah_redo/` or similar
**Observed**: directory does not exist on disk (verified via `find /home/hugo/codes/mol-metal -path "*wf285a*"` — only `wf285b_kendall_dann_wiring` and `wf285c_no_enforce_budget` directories present).
**Cross-ref**: §9.AH was the §9 row whose values were flagged as wrong in Wf-284D §10 audit retry; the redo was supposed to overwrite §9.AH with the corrected row but no report + no doc patch landed.
**Verdict**: ❌ MISSING-DELIVERABLE — agent did not deposit an on-disk artefact; the §9.AH wrong-value row remains in the checklist as it stood post-Wf-284D.

### §1.2 Wf-285B — Kendall UW + DANN GRL wiring — **MEASURED PASS (3/3 tests)**

**File on disk**: `molmetal/reports/wf285b_kendall_dann_wiring/wiring_report.md` (≈200 lines)
**Findings**:
- `MultiTaskLoss(use_kendall_loss=...)` and `MultiTaskWassersteinModel(use_dann=..., num_domains=..., domain_head_dropout=...)` now thread the toggles all the way through `train_one_metal_one_seed` and `train_one_epoch_joint`.
- DANN GRL block runs `ReverseLayerF.apply(..., grl_alpha)` + `model.domain_head(rev)` + cross-entropy against constant `domain_id`; bypassed when `use_dann=False` so Wf-131 P1 baseline bit-for-bit preserved.
- Four new optional CLI flags (`--domain-id`, `--lambda-grl`, `--grl-alpha`, `--num-domains`) added with sensible defaults.
- 3/3 new tests pass (`molmetal/tests/test_kendall_dann_wiring.py`); 8/8 existing CLI-flag tests still pass (`test_no_kendall_no_dann_flags.py`).
- Smoke (Ru seed=0, 1 epoch, CPU): `--no-kendall --no-dann` 65.2s `test_auc=0.5849`; defaults 108.0s `test_auc=0.6012` — both end-to-end without NaN/shape/backward-graph errors. DANN ON ~1.65× slower (expected).
**Verdict**: ✅ MEASURED — lever is now real, the Wf-283B honest-gap (flags parsed but not wired) is closed.

### §1.3 Wf-285C — `--no-enforce-budget` CLI flag — **MEASURED PASS (31/31 tests)**

**File on disk**: `molmetal/reports/wf285c_no_enforce_budget/flag_report.md` (≈120 lines)
**Findings**:
- `parse_argv()` exposes `--no-enforce-budget` (action=`store_true`, default=False); `main()` forwards `enforce_budget = not args.no_enforce_budget` to `compute_wave_status(...)` and `summarise(...)`.
- 3 new tests added under Test 13 in `molmetal/tests/test_wave_budget_tracker.py`: `test_no_enforce_budget_flag`, `test_budget_enforced_when_flag_off`, `test_no_enforce_budget_default_false`.
- Full suite: **31 passed, 1 warning in 104.50s** (0:01:44 wall). `--help` exits cleanly (rc=0).
- Honest framing preserved: `summarise()` branch with `enforce_budget=False` still reports "10.0h budget exceeded by 4.0h (--no-enforce-budget; user opted out of DEFERRED labeling); all 4 wave(s) PLANNED." — the over-budget is logged, only the DEFERRED label is suppressed (Wf-262 §0.5 honest-gap).
**Verdict**: ✅ MEASURED — user 2026-09-28 "时间预算没关系的" opt-out directive is now an explicit CLI flag, with the budget figure still visible for honest ledger.

### §1.4 Wf-285D — stale-read guard — **MISSING-DELIVERABLE**

**Expected artefact**: `molmetal/reports/wf285d_stale_read_guard/` or similar
**Observed**: directory does not exist on disk (verified via `find /home/hugo/codes/mol-metal -path "*wf285d*"` — zero hits).
**Cross-ref**: a stale-read guard was proposed to defend `m14_anchor.json` / `cfm_decode_real.json` style JSON ledger files against re-using a cached value after the source-of-truth changed (Wf-271B + Wf-287A m14 LEAD swap pattern); no code or test landed in this wave.
**Verdict**: ❌ MISSING-DELIVERABLE — the guard was specified but no agent produced the on-disk implementation. Carry-over HN.

---

## §2 Aggregate Counts

| Metric | Value |
|---|---|
| Upstream waves dispatched | 4 |
| Upstream waves with on-disk MEASURED output | **2** (Wf-285B ✅ + Wf-285C ✅) |
| Upstream waves MISSING (no on-disk report) | **2** (Wf-285A + Wf-285D) |
| MEASURED SHIPPED deliverables | **2** (Wf-285B 3/3 wiring tests + smoke; Wf-285C 31/31 budget tracker tests) |
| MEASURED HONEST-NEGATIVES delivered | **0** (Wf-285A + Wf-285D did not deliver a negative-result report either — they did not deliver at all) |
| Tests added (net) | **6** (3 Wf-285B + 3 Wf-285C) |
| Tests passing post-wave | **31/31** in Wf-285C suite; Wf-285B suite not re-counted in this aggregate |
| Inconsistencies detected → fixed on disk | **0** this wave |
| Total work MEASURED-completed | **2 of 4 attempted** = **50%** |
| Total work NOT LAUNCHED (wave count) | **0** (all 4 were launched; 2 didn't deposit artefacts) |
| Total work failed (hard) | **0** (no crash reports; missing ≠ failed) |
| Pipeline regression class occurrence count | **0** |

---

## §3 Critical honest-negatives remaining

1. **Wf-285A §9.AH wrong-values redo missing** — §9.AH row in `molmetal/TODO/completed/207_unified_pre_flight_checklist_v2.md` still carries the pre-Wf-284D values flagged as wrong. No redo landed; no on-disk patch to a §9 row. This is the highest-priority doc-edit HN.
2. **Wf-285D stale-read guard missing** — `m14_anchor.json` / `cfm_decode_real.json` / `per_metal_pass.json` reads downstream of those JSON files remain vulnerable to the Wf-271B / Wf-287A re-use pattern (anchor swapped, downstream still references the older value). No `if_version` / mtime-staleness wrapper landed in this wave.
3. **§9.AH carry-over delta** — even if Wf-285A is re-dispatched, the corrected row must align with the cross-ref entries Wf-284D §10.3 path-naming fix and Wf-277A §9.AF verbatim Δ (`per_lever.json` `no_w Δ=-0.013`, `no_cb Δ=-0.008`, NOT `Δ=0.000`) so the §9 row is internally consistent before §10 cross-ref audit is rerun.
4. **Wf-285D guard interacts with Wf-287A m14 LEAD 0.9958 anchor** — the anchor-supersession path (0.9645 → 0.9958) was handled by `silent_regression_methodology_change_not_disclosed` block, but a downstream consumer that re-reads `m14_anchor.json` outside that explicit block could still pick up the older 0.9645 value. The guard is the durable fix.

---

## §4 Next follow-up priorities (ranked)

| Pri | Task | Source | Effort | Parallel-safe |
|---|---|---|---|---|
| **P1** | Re-dispatch Wf-285A: replace §9.AH row in `207_unified_pre_flight_checklist_v2.md` with corrected values; cross-check against Wf-284D §10.3 path-naming fix + Wf-277A §9.AF `per_lever.json` Δ alignment | §3 HN-1 + HN-3 | ~10 min doc edit | **YES — ultracode-suitable (explicit doc-only patch, single file)** |
| **P2** | Re-dispatch Wf-285D: implement stale-read guard wrapper (e.g. `read_anchor_json(path, max_age_sec=86400, required_version=...)`) for `m14_anchor.json`, `cfm_decode_real.json`, `per_metal_pass.json`, `per_lever.json`; add 2-3 tests under `molmetal/tests/test_stale_read_guard.py` | §3 HN-2 + HN-4 | ~30 min code + tests | **YES — CPU-only, isolated module** |
| **P3** | Wf-285B follow-up: 4-cell `use_dann × use_kendall ∈ {0,1}²` ablation grid (CPU 1-seed smoke first, then GPU 5-seed) — turns the lever into paper §6 ablation table entry | Wf-285B §7 | ~10 min CPU smoke + ~30 min GPU wall | NO — GPU part is kanzi-contention-gated |
| **P4** | Wf-285B follow-up: per-cell-line `domain_id` labelling (4-domain mode across MetalCytoToxDB + DrugOOD + NCI-60 + BindingDB rows) + multi-domain training pool | Wf-285B §7 | ~45 min code | NO — depends on P2 guard + needs a 5-metal CSV column |

**P1 and P2 are the two must-ship items** (close Wf-285A + Wf-285D missing-deliverable HNs). They are both CPU-only / single-file, ultracode-safe, and orthogonal to the GPU retrain queue (Wf-277B + Wf-289 series). P3 and P4 unlock the paper §6 ablation table that the Wf-285B wiring enables.

---

## §5 Honest framing (party-twenty-spirit §3.5)

- 2 of 4 Wf-285 agents shipped with on-disk MEASURED artefacts (Wf-285B wiring, Wf-285C budget flag) — both green tests, both honest in their reports.
- 2 of 4 agents (Wf-285A §9.AH redo + Wf-285D stale-read guard) did NOT deposit on-disk artefacts; this is a **MISSING-DELIVERABLE**, not a hard failure, and is the dominant honest-negative left from Wf-285.
- Wf-285's 50% MEASURED completion is a regression from Wf-282E / Wf-287E / Wf-284E's 4-of-4 streak — but the missing 2 are tractable doc/code tasks (P1 + P2), not architectural blockers.
- No §9.N entry needs to be created for Wf-285A or Wf-285D this turn; both are tracked as Wf-285E follow-up priorities instead.
- Wf-285B's `--no-kendall --no-dann` lever is now real; Wf-251-F3 ablation grid is unblocked at the trainer level (P3 + P4 turn it into paper-ship numbers).

---

**End of Wf-285E followup_summary. Aggregated n_agents=4, n_shipped=2, n_failed=0, n_missing_deliverable=2, n_remaining_doc_edits=1 (P1 ultracode), n_remaining_code_patches=1 (P2 stale-read guard), n_remaining_gpu_walls=1 (P3 ablation grid), n_remaining_data_pipelines=1 (P4 4-domain pool).**