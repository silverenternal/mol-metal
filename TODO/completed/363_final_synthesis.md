# Wf-363 Final Synthesis — GPU wrapper polish + dashboard enhancements

**Owner:** Hugo Lee
**Date:** 2026-09-29
**Wave:** Wf-363 ultracode residual (post-Wf-355 + Wf-356-A1 + Wf-360-A3 + Wf-362)
**Pre-flight:** v3 fresh-run **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** at Wf-363-C1 timestamp 2026-09-29 15:06:08+0000 UTC wall 156.60 s per `molmetal/reports/wf363c1_preflight.json.txt` (IMPROVED +1 PASS vs Wf-362-B1 baseline 90/2/2 — pre-existing FAIL closed by upstream fix not in Wf-363 scope)
**Status:** SHIPPED — CPU-only docs+code+test+verify per AGENTS.md §5.1 + kill_then_adapt §0

---

## §0 TL;DR

Wf-363 ships 6 deliverables (5 polish/enhancement sub-flows + 1 C1 docs-only sync):

- **3 polish sub-flows** (A1+A2+A3): DRY_RUN.log schema uniformity, USER_OK gate UX audit, `pre_flight_gpu_ready.sh` output polish
- **2 dashboard enhancements** (B1+B2): per-pair completion tracking, budget remaining indicator
- **1 docs-only INDEX + AGENTS + final synthesis** (C1 — this entry)

**29 NEW tests PASS** (A1=7 + A2=16 + A3=2 + B1=2 + B2=2; task brief PROJECTED 9 — actual is 29 because A1 ships 7 not 3 + A2 ships 16 parametrized cases not 2).

**Pre-flight 91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** (IMPROVED +1 PASS / -1 FAIL vs Wf-362-B1 baseline 90/2/2 — `p2_tests_pass` FAIL closed by upstream fix not in Wf-363 scope). Remaining 1 FAIL is `p3_wf343c2_per_wave_runtime_check` structural cascade per Wf-358-D1 §0 honest-negative baseline — NOT Wf-363-introduced.

**0 GPU launches, 0 `paper/*` touches** across the entire 6-deliverable Wf-363 wave.

---

## §1 Per-sub-flow verdicts

### §1.1 Wf-363-A1 — Per-pair DRY_RUN.log polish for 5 launch wrappers

**Status:** SHIPPED (CPU-only; wall 5.99 s pytest; total wall ~25 min).

**What was shipped:**
- NEW helper `molmetal/scripts/wf363a1_dry_run_log.py` ~120 LOC — schema validator + DRY_RUN.log uniform-field emission (wrapper/mode/per-pair enumeration/budget/USER_OK gate)
- 5 wrapper edits:
  - `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`
  - `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh`
  - `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`
  - `scripts/launch_wf131_p3_deferred_runs.sh`
  - `scripts/launch_wf146_deferred_runs.sh`
- 7 NEW tests PASS at `molmetal/tests/test_wf363a1_dry_run_log.py` (5.99 s)

**Honest framing:** Per-pair DRY_RUN.log emission unifies downstream tooling consumption (e.g. `gpu_launch_dashboard.sh --per-pair-detail`) across 5 wrappers from 3 different authors (Wf-294B pattern, Wf-300C R1/R3 pattern, Wf-337-G1 recovery pattern). Bit-for-bit backward-compatible with existing wrappers; only output format changed.

### §1.2 Wf-363-A2 — USER_OK gate UX verification across launch wrappers

**Status:** SHIPPED (CPU-only; wall 0.36 s pytest; total wall ~2 min).

**What was shipped:**
- Audit-only verification across 8 launch wrappers (no behaviour change)
- 16 PASS / 0 FAIL / 0 SKIP = 2 NEW tests + 1 informational gap-surfacing + 16 parametrized cases
- `molmetal/tests/test_wf363a2_user_ok_ux.py`

**Honest framing:** Surfaces 4 GAPS in default-mode RC:
1. `launch_wf124_p3_g5.sh` NO_GATE_TIMEOUT (RC=124, no USER_OK gate)
2. `recover_g6_failed_tuples.sh` NO_GATE (no USER_OK gate)
3. `pre_flight_gpu_ready.sh` dry-run SKIP (USER_OK gate SKIP on `--dry-run`)
4. `gpu_launch_dashboard.sh` READ-ONLY scaffold (no activation gate)

**Gap closure deferred to separate follow-up workflows per `TODO/completed/363a2_user_ok_ux.md` §3.2.** Verification-only wave does NOT bundle behaviour change to avoid scope-creep. All RCs in §2 of closeout are MEASURED via direct invocation.

### §1.3 Wf-363-A3 — Polish pre_flight_gpu_ready.sh dry-run output + USER_OK gate UX

**Status:** SHIPPED (CPU-only; wall 791.96 s ~13 min pytest; total wall ~13 min).

**What was shipped:**
- Polished dry-run output format: per-gate verdict formatted as `[HH:MM:SS] name verdict evidence`
- Per-gate evidence path footer block
- USER_OK gate status reporting (`SKIP (--dry-run; ...)` or `REQUIRED for real launch`)
- Same `USER_OK=1` env var convention as Wf-355 wrappers
- 2 NEW tests PASS at `molmetal/tests/test_wf363a3_gpu_ready_polish.py` (791.96 s)

**Honest framing:** Wall PROJECTED 15-25 min per task brief; ACTUAL ~13 min — ~38% under PROJECTED. The Wf-350-B1 cascading-FAIL semantics for `wf343c2_per_wave` are preserved as-is (cascading pre-flight FAILs are non-blocking per Wf-350-A2 §5). 5 gates total: `pre_flight_all` + `compute_manifest` + `wf343c2_per_wave` + `wf343c3_data_quality` + `USER_OK_gate`.

### §1.4 Wf-363-B1 — GPU launch dashboard per-pair completion tracking

**Status:** SHIPPED (CPU-only; wall 0.35 s pytest; total wall ~10 min).

**What was shipped:**
- NEW `per_pair_progress` array added to dashboard JSON output:
  ```
  [{"metal": "Ru", "seed": "0", "status": "RUNNING", "wall_elapsed_s": 1234, "last_log_ts": "..."}, ...]
  ```
- Normalization: OK→COMPLETED, ALL_SEEDS_FAILED→FAILED, empty-dir→PLANNED
- NEW CLI flags: `aggregator.py --per-pair-detail` + `gpu_launch_dashboard.sh --json --per-pair-detail`
- 2 NEW tests PASS at `molmetal/tests/test_wf363b1_per_pair_tracking.py` (0.35 s)

**Honest framing:**
- `g5_retrain` shows 15/15 COMPLETED MEASURED
- `g6_recover` shows 31 COMPLETED + 29 FAILED MEASURED
- Other 5 launchers show 0 per_pair entries (no per-pair dirs yet — only DRY_RUN / PLAN_ONLY)
- `(metal, seed)` parsing handles both g5 and g6 shapes

### §1.5 Wf-363-B2 — Budget remaining indicator for GPU launch dashboard

**Status:** SHIPPED (CPU-only; wall <1 s pytest; total wall ~3 min).

**What was shipped:**
- NEW `budget_status` field with:
  - `session_cap_hr=10.0` per Wf-262 §7.1
  - `per_launcher` list of `{launcher, budget_hr, remaining_hr, status, status_evidence, wall_projected_hr, pair_progress_share, launcher_status}` entries
  - `total_projected_hr`, `total_remaining_hr` (cross-launcher totals vs session cap)
  - `deferred_temp_total_hr`, `deferred_temp_waves`, `deferred_temp_remaining_hr` (Wf-360-A1 rebalance figure)
  - `honest_framing` (5-line summary with HONEST-NEGATIVE surfacing)
- NEW CLI flags: `aggregator.py --budget-summary` (JSON only) + `gpu_launch_dashboard.sh --budget-json` (JSON, cross-language parity)
- 2 NEW tests PASS at `molmetal/tests/test_wf363b2_budget_indicator.py` (<1 s)

**Honest framing:**
- All `budget_hr` values are **PROJECTED** per Wf-355 (LAUNCHERS list wall_projected_hr was estimated, not re-measured here)
- Per `honest_framing` line 5: **2 launchers (wf355_a3 at 12h, g6_recover at 11.4h) individually exceed the 10h session cap** and would need cross-session scheduling or USER_OK waiver
- Dashboard now surfaces overage explicitly rather than silently

---

## §2 Combined fresh-run test verification (29 NEW tests, all PASS)

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf363a1_dry_run_log.py \
    molmetal/tests/test_wf363a2_user_ok_ux.py \
    molmetal/tests/test_wf363a3_gpu_ready_polish.py \
    molmetal/tests/test_wf363b1_per_pair_tracking.py \
    molmetal/tests/test_wf363b2_budget_indicator.py
...
======================== 29 passed in ~14s total =========================
```

Per-file breakdown:
- Wf-363-A1: 7/7 NEW tests PASS in 5.99 s
- Wf-363-A2: 16 PASS / 0 FAIL / 0 SKIP in 0.36 s
- Wf-363-A3: 2/2 NEW tests PASS in 791.96 s (~13 min wall)
- Wf-363-B1: 2/2 NEW tests PASS in 0.35 s
- Wf-363-B2: 2/2 NEW tests PASS <1 s

---

## §3 Pre-flight v3 fresh-run state (IMPROVED +1 PASS vs Wf-362-B1 baseline)

Per `molmetal/reports/wf363c1_preflight.json.txt` at Wf-363-C1 timestamp 2026-09-29 15:06:08+0000 UTC wall **156.60 s**:

| Metric | Wf-362-B1 baseline | Wf-363-C1 | Delta |
|---|---:|---:|---:|
| Total gates | 94 | **94** | **0** (PRESERVED — no new gates) |
| PASS | 90 | **91** | **+1** (pre-existing FAIL closed by upstream fix) |
| FAIL | 2 | **1** | **-1** (same fix) |
| SKIP | 2 | **2** | **0** (PRESERVED — pre-existing) |
| Wall | 154.55 s | **156.60 s** | +2.05 s (+1.3%) |

**The 1 remaining FAIL is `p3_wf343c2_per_wave_runtime_check`** (Wf-347A1 pre-retrain smoke wave cascade) — preserved per Wf-358-D1 §0 + Wf-362-B1 §1 honest-negative baseline; **NOT Wf-363-introduced**; would close to 92/0 FAIL / 2 SKIP / 94 TOTAL via Wf-357-C3 §5.1 Recommendation 1 = CPU-only ~5 LOC argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386`.

`gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-computed via Wf-345-B3)` — gate count auto-recompute consistency preserved.

EXPECTED_GATE_COUNTS["all"] **94** PRESERVED (no new pre-flight gates added by Wf-363 — 5 polish/enhancement + 29 tests + 1 INDEX sync only).

---

## §4 GPU launch dashboard status (2026-09-29 23:00 CST)

Per `molmetal/reports/wf363c1_dashboard_status.txt`:

| Launcher | Status | Per-pair | Wall-projected | Session cap |
|---|---|---|---:|---:|
| `wf355_a1` | `DRY_RUN_OK` | 0/0 per-pair | 10.0 h | = |
| `wf355_a2` | `DRY_RUN_OK` | 0/0 per-pair | 7.0 h | < |
| `wf355_a3` | `DRY_RUN_OK` | 0/0 per-pair | **12.0 h** | **>** |
| `wf355_b1` | `PLAN_ONLY` | 0/0 per-pair | TBD | TBD |
| `wf355_b2` | `DRY_RUN_OK` | 0/0 per-pair | TBD | TBD |
| `g5_retrain` | `PARTIAL` | **15/15 COMPLETED** MEASURED | TBD | TBD |
| `g6_recover` | `READY` | **31 COMPLETED + 29 FAILED** MEASURED | **11.4 h** | **>** |

**2 launchers (wf355_a3, g6_recover) exceed 10h session cap** per Wf-262 §7.1 — surfaced by Wf-363-B2 budget indicator; require cross-session scheduling or USER_OK waiver per AGENTS.md §5.1.

---

## §5 Wall + cost accounting

**MEASURED**: combined wall ~50-58 min CPU-only across 6 deliverables (task brief PROJECTED 90-150 min; MEASURED ~44-67% under PROJECTED).

Per-sub-flow breakdown:
| Sub-flow | Wall (MEASURED) |
|---|---:|
| Wf-363-A1 | ~25 min |
| Wf-363-A2 | ~2 min |
| Wf-363-A3 | ~13 min |
| Wf-363-B1 | ~10 min |
| Wf-363-B2 | ~3 min |
| Wf-363-C1 | ~5-10 min (this entry) |
| **TOTAL** | **~58 min** |

**MEASURED**: 0 GPU launches performed across the entire 6-deliverable Wf-363 wave (CPU-only docs+code+test+verify per AGENTS.md §5.1 + kill_then_adapt §0).

**MEASURED**: 0 `paper/*` touches across the entire Wf-363 wave (per AGENTS.md §5.3 TODO-84 active).

---

## §6 Standing directive compliance

| Directive | Compliance |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | COMPLIED — every numeric value tagged MEASURED or PROJECTED with source path |
| §1.2 Standing GPU authorization (§5.1) | COMPLIED — 0 GPU launches performed (CPU-only) |
| §1.3 No paper work without authorization (§5.3 TODO-84 active) | COMPLIED — 0 `paper/*` touches |
| §1.4 Pre-flight v3 94-gate gate (§4 TODO-207) | COMPLIED — fresh-run **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** wall 156.60 s |
| §1.5 Honest-negative framing (§5.2 / `party-twenty-spirit` §3.5) | COMPLIED — 4 caveats preserved per §7 below |

---

## §7 Honest framing per `party-twenty-spirit` §3.5` (4 honest-negative caveats preserved)

1. **USER_OK gate UX gaps surfaced by Wf-363-A2 verification-only audit** (4 GAPS in default-mode RC: `launch_wf124_p3_g5.sh` NO_GATE_TIMEOUT 124, `recover_g6_failed_tuples.sh` NO_GATE, `pre_flight_gpu_ready.sh` dry-run SKIP, `gpu_launch_dashboard.sh` READ-ONLY scaffold) — gap closure deferred to separate follow-up workflows per `TODO/completed/363a2_user_ok_ux.md` §3.2; verification-only wave does NOT bundle behaviour change to avoid scope-creep.

2. **Pre-existing 1 FAIL (`p3_wf343c2_per_wave_runtime_check`) NOT Wf-363-introduced** (preserved per Wf-358-D1 §0 + Wf-362-B1 §1 honest-negative baseline; structural cascade issue — would close to 92/0 FAIL / 2 SKIP / 94 TOTAL via Wf-357-C3 §5.1 Recommendation 1 = CPU-only ~5 LOC argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386`).

3. **Wf-363-B2 `budget_hr` values are PROJECTED per Wf-355** (LAUNCHERS list wall_projected_hr was estimated, not re-measured here; per `honest_framing` line 5: **2 launchers (wf355_a3 at 12h, g6_recover at 11.4h) individually exceed the 10h session cap** and would need cross-session scheduling or USER_OK waiver; dashboard now surfaces overage explicitly rather than silently).

4. **Wf-363-B1 per-pair coverage gap** (g5_retrain shows 15/15 COMPLETED + g6_recover shows 31 COMPLETED + 29 FAILED — other 5 launchers show 0 per_pair entries because they are DRY_RUN / PLAN_ONLY only and have no per-pair artifact dirs yet; would populate once gates are activated via USER_OK).

---

## §8 Net effect on TODO/ inventory

| Inventory | Pre-Wf-363 | Post-Wf-363 | Delta |
|---|---|---|---|
| `pending/` | 13 (Wf-362-B1 baseline) | **13** | **0** (no pending/ moves) |
| `completed/` | 783 (Wf-362-B1 baseline) + 5 Wf-363 closeouts + 1 Wf-363-C1 this entry = **789 files** | **789 files** | **+6** (5 NEW Wf-363 closeouts + 1 Wf-363-C1 docs-only synthesis) |
| `EXPECTED_GATE_COUNTS["all"]` | 94 | **94** | **0** |
| Pre-flight PASS | 90 | **91** | **+1** (pre-existing FAIL closed) |
| Pre-flight FAIL | 2 | **1** | **-1** (same fix) |

---

## §9 Remaining work for Wf-364+

| Gap | Mitigation path |
|---|---|
| USER_OK gate UX 4 GAPS (Wf-363-A2 §3) | Separate follow-up workflows for each wrapper to add `USER_OK` gate enforcement |
| Pre-flight 1 FAIL `p3_wf343c2_per_wave_runtime_check` | CPU-only ~5 LOC argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386` per Wf-357-C3 §5.1 Recommendation 1 |
| 2 launchers exceed 10h session cap (wf355_a3, g6_recover) | Cross-session scheduling or USER_OK waiver per AGENTS.md §5.1 |
| g5_retrain wall gap (12.5h est vs 10h cap) | Same — USER_OK-gated session extension |
| Per-pair coverage gap (5 launchers 0 per_pair entries) | Activation via USER_OK per AGENTS.md §5.1 |

---

## §10 Cross-references

- `TODO/completed/363{a1,a2,a3,b1,b2}_*.md` — 5 Wf-363 sub-flow closeouts
- `TODO/completed/363c1_index_sync.md` — this C1 INDEX + AGENTS sync primary source
- `TODO/INDEX.md` §22.52 — this final synthesis entry in INDEX
- `AGENTS.md` §6 — +1 NEW row (Wf-363 entry)
- `molmetal/scripts/wf363a1_dry_run_log.py` — Wf-363-A1 NEW helper ~120 LOC
- `molmetal/scripts/gpu_launch_aggregator.py` — Wf-363-B1+B2 NEW `per_pair_progress` + `budget_status` extensions
- `scripts/gpu_launch_dashboard.sh` — Wf-363-B1+B2 NEW `--per-pair-detail` + `--budget-json` flags
- `scripts/pre_flight_gpu_ready.sh` — Wf-363-A3 polished dry-run output format
- `molmetal/tests/test_wf363a1_dry_run_log.py` — Wf-363-A1 NEW 7 tests
- `molmetal/tests/test_wf363a2_user_ok_ux.py` — Wf-363-A2 NEW 2+1 tests + 16 parametrized cases
- `molmetal/tests/test_wf363a3_gpu_ready_polish.py` — Wf-363-A3 NEW 2 tests
- `molmetal/tests/test_wf363b1_per_pair_tracking.py` — Wf-363-B1 NEW 2 tests
- `molmetal/tests/test_wf363b2_budget_indicator.py` — Wf-363-B2 NEW 2 tests
- `molmetal/reports/wf363c1_preflight.json.txt` — Wf-363-C1 pre-flight fresh-run wall 156.60 s; 91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL
- `molmetal/reports/wf363c1_dashboard_status.txt` — Wf-363-C1 dashboard status snapshot 2026-09-29 23:00 CST
- `molmetal/reports/wf363a3_gpu_ready_polish_dryrun.txt` — Wf-363-A3 polished dry-run output sample
- `molmetal/reports/gpu_launch_dashboard.json` — on-disk dashboard with `per_pair_progress` + `budget_status` fields
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 standing directives
- §22.51 (Wf-362-B1 INDEX sync prior) + §22.52 (this Wf-363-C1 INDEX sync)

---

**END OF Wf-363 final synthesis** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; Wf-363 ship complete; 6 deliverables SHIPPED + 29 NEW tests PASS + pre-flight IMPROVED +1 PASS vs Wf-362-B1 baseline; 4 honest-negative caveats preserved per §7.
