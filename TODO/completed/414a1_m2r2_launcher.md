# Wf-414-A1 — m2 R² Path A launch wrapper SHIPPED + 10 NEW tests PASS

**Date:** 2026-09-30
**Status:** **CLOSED — wrapper + tests + plan move all SHIPPED; Wf-411-A2 honest-negative
CLOSED end-to-end.**
**Workflow:** Wf-414-A1 (sub-flow of Wf-414 ultracode CPU-only docs+code+test wave —
"tasks not yet covered" parent for Wf-411-A2 HONEST-NEGATIVE closure)

---

## §0 TL;DR

Wf-414-A1 closes the **Wf-411-A2 HONEST-NEGATIVE** that surfaced the missing
`scripts/launch_wf390a1_m2_r2_path_a.sh` wrapper. The plan lives at
`TODO/pending/376_m2_r2_gpu_retrain_plan.md` (Wf-390-A1) but the wrapper was
NOT on disk before this wave — leaving the plan with no concrete invocation
path. Wf-414-A1 SHIPPED:

| Deliverable | Path | LOC | Status |
|---|---|---:|---|
| Launch wrapper | `scripts/launch_wf390a1_m2_r2_path_a.sh` | ~270 | SHIPPED |
| Test suite | `molmetal/tests/test_wf414a1_m2r2_launcher.py` | ~310 | SHIPPED 10/10 PASS |
| Plan move | `TODO/pending/376_*.md` → `TODO/completed/376_*.md` | n/a | SHIPPED + SCAFFOLD header updated |
| Closeout | `TODO/completed/414a1_m2r2_launcher.md` | ~280 | SHIPPED (this file) |

**Combined wall:** ~5 min CPU-only (test suite already SHIPPED at 10/10 PASS).
**0 GPU launches, 0 paper/* touches (TODO-84), 0 wet-lab planning (AGENTS.md §10).**

---

## §1 Wf-411-A2 HONEST-NEGATIVE close-out

Per Wf-390-A1 + Wf-390-B1 §22.7 + Wf-411-A2:

> **HONEST-NEGATIVE Wf-411-A2:** `scripts/launch_wf390a1_m2_r2_path_a.sh` MISSING
> on disk. The plan (`TODO/pending/376_m2_r2_gpu_retrain_plan.md`) is
> complete but the actual launch wrapper was never authored.

Wf-414-A1 SHIPS the wrapper following the canonical pattern of `launch_wf124_p3_g5.sh`
(Wf-366-A1 USER_OK gate fix) + `launch_wf131_p3_deferred_runs.sh` (Wf-355-B1
USER_OK-gated LOMO) + Wf-294B family convention (exit 64 on USER_OK refusal).

**Wrapper features:**
- USER_OK gate (exit 64 on refusal) — kill_then_adapt safety contract
- --dry-run path emits Wf-363-A1 DRY_RUN.log + plan.json
- Wf-390-A1 SCAFFOLD smoke (`wf390a1_path_a_smoke.py`) wired BEFORE the DRY_RUN
  branch — verifies trainer CLI accepts the 5-metal subset + emits per-cohort
  config JSON without GPU launch (catches Wf-379-A2 type regressions early)
- 15 (metal, seed) pair enumeration (5 metals × 3 seeds × 50 epochs on
  metalcytotox_combined n=2000)
- --per-cohort-features (Wf-405-A2 + Wf-388-A1 SCAFFOLD/APPLIED pattern)
- --no-cuda-graphs safety lever (Wf-335-R3 + Wf-330-B4 Blackwell RuntimeError fix)
- --chemprop-v3-cache flag APPENDED only when TRAINER=*chemprop_v2_v3* (per
  Wf-392-A2 — flag scoping; D-MPNN trainer does NOT accept this flag)
- Per-wave runtime check (Wf-343c2 7-check surface) wired into run_pair() via
  --skip-live for CPU-only dry-run (Wf-347-B1 + Wf-347-B2 hardening)
- Per-pair budget tracking (PAIR_BUDGET_START / PAIR_BUDGET_END + elapsed_s)
- Wf-302B guard: wait_until_epoch_reached polls pair log for epoch=50
- Aggregation script: collects per_seed_metrics_seed*.json into training_summary.json

---

## §2 Test suite (10/10 PASS)

`molmetal/tests/test_wf414a1_m2r2_launcher.py` — 10 NEW tests, all PASS in
~4.25 s wall CPU-only.

| Test | Verifies |
|---|---|
| `test_launcher_syntax` | bash -n parses cleanly (Wf-411-A2 closure: file on disk + valid syntax) |
| `test_user_ok_gate_REFUSED_default` | USER_OK unset → exit 64 (kill_then_adapt safety) |
| `test_user_ok_gate_REFUSED_zero` | USER_OK=0 → exit 64 (only USER_OK=1 exact match satisfies) |
| `test_user_ok_gate_PASSED_one_dry_run` | USER_OK=1 + --dry-run → SCAFFOLD smoke runs + DRY_RUN.log + plan.json emitted + 15 pairs enumerated |
| `test_dry_run_emission_schema` | DRY_RUN.log has Wf-363-A1 schema (wrapper/mode/trainer/pair/flags/gpu/budget/user_ok/ts/per_pair × 15) |
| `test_plan_json_emission` | plan.json is valid JSON with Wf-414-A1 canonical schema (user_ok, dry_run, n_pairs=15, gpu=cuda:0, wall_budget_hr=12, honest_framing.wf411a2_honest_negative) |
| `test_scaffold_smoke_invocation` | SCAFFOLD smoke (Wf-390-A1) runs BEFORE DRY_RUN branch (catches Wf-379-A2 type regressions) |
| `test_chemprop_v3_cache_conditional` | --chemprop-v3-cache APPENDED only when TRAINER=*chemprop_v2_v3* (per Wf-392-A2 flag scoping) |
| `test_per_cohort_features_present` | --per-cohort-features is INSIDE COMMON_ARGS=() array (per Wf-388-A1 + Wf-405-A2) |
| `test_no_cuda_graphs_safety_lever` | --no-cuda-graphs is INSIDE COMMON_ARGS=() array (per Wf-335-R3 + Wf-330-B4 RuntimeError fix) |

**Test invocation:**
```
PYTHONPATH=/home/hugo/codes/mol-metal \
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
-m pytest molmetal/tests/test_wf414a1_m2r2_launcher.py \
--noconftest \
--tb=short
```
**Result:** `10 passed in 4.25 s` wall CPU-only.

Note: the project's `molmetal/tests/conftest.py` (Wf-235 wire-in shim) has
a known issue with the `_reference/path` auto-detect in the current venv;
tests are invoked with `--noconftest` to bypass. Same workaround used by
the Wf-358 / Wf-380 / Wf-385 + Wf-388 wave test suites.

---

## §3 TODO-376 plan move + SCAFFOLD status update

**Before:** `TODO/pending/376_m2_r2_gpu_retrain_plan.md` (PLAN ONLY — wrapper MISSING)
**After:** `TODO/completed/376_m2_r2_gpu_retrain_plan.md` (SCAFFOLD SHIPPED —
launcher + tests PASS; actual GPU retrain awaits USER_OK=1)

Updated header to reflect SCAFFOLD status:
```
# Wf-390-A1 — m2 R² Path A 5-metal subset GPU retrain plan (USER_OK-gated)

**Date:** 2026-09-30 (plan authored; Wf-414-A1 wrapper SHIPPED 2026-09-30)
**Status:** **SCAFFOLD SHIPPED — launcher + 10/10 tests PASS; awaiting `USER_OK=1` grant
for actual GPU retrain.**
```

The plan body (Wf-390-A1 §0 through §8) is preserved verbatim as historical
baseline; only the status header was updated to reflect Wf-414-A1 closure.

---

## §4 Honest framing (party-twenty-spirit §3.5 + AGENTS.md §5.2)

| Caveat | Verdict |
|---|---|
| **`--chemprop-v3-cache` flag brief mismatch** | The task brief listed `--chemprop-v3-cache` alongside the D-MPNN trainer. Per Wf-392-A2, this flag exists on `train_chemprop_v2_v3.py` but NOT on `train_dmpnn_multitask_wasserstein.py`. Including it unconditionally would cause argparse error on the D-MPNN trainer. Wf-414-A1 ships it with a conditional guard: `if [[ "${TRAINER}" == *"chemprop_v2_v3"* ]]; then COMMON_ARGS+=( --chemprop-v3-cache ); fi` — so D-MPNN launches skip it, chemprop v3 launches include it. Honest documentation in the wrapper header (lines 56-63) + test 8. |
| **MEASURED vs PROJECTED** | MEASURED: bash syntax + USER_OK gate behavior + DRY_RUN.log schema + plan.json schema + COMMON_ARGS composition + SCAFFOLD smoke invocation. PROJECTED: ~12 hr GPU wall (per Wf-260 G6 timing on RTX 5090 / Blackwell sm_120); m2 R² lift on metalcytotox_combined (n=2000) CEILING_BREAK axis (per Wf-390-A1 §0 table). |
| **n=200 cohort ceiling preserved** | Rh (n=1.2k) + Re (n=800) per-metal singlet subsweeps PROJECTED R² remains near 0 or negative per Wf-316A 12-d RDKit signal floor + Wf-367-B1 multi-stream overfit. The 15-pair sweep targets metalcytotox_combined (n=2000) CEILING_BREAK axis only. |
| **5-metal × 3-seed × 50-epoch vs Wf-390-A1 §2.2 plan** | The task brief calls for "5-metal × 3-seed × 50-epoch". Wf-390-A1 §2.2 plans 3 sub-launches (A.1 = 5-metal × 5-seed × 50-ep on n=2000; A.2 = 5-metal × 5-seed × 50-ep on n=26.8k; A.3 = per-metal singlets × 1 seed). Wf-414-A1 implements A.1 scope (5-metal × 3-seed × 50-ep = 15 pairs) per the brief literally; A.2 + A.3 deferred to follow-up launch wrappers if needed. |
| **`--pair "M 0"` runtime subset override** | Wrapper accepts --pair "M S" overrides for single-pair debug runs (e.g., `--pair "Ru 0"`). Defaults to all 15 pairs. |
| **GPU launch BLOCKED** | Actual GPU retrain remains gated on USER_OK=1 + VRAM ≥ 8 GB + pre-flight 89/2/3/94 + wave budget USER_OK waiver. Wrapper exits 64 without USER_OK=1 in env (kill_then_adapt safety contract preserved). |

---

## §5 Forward-looking user-OK-gated follow-ups

Per Wf-368-B1 §7 + Wf-390-B1 §22.7.3 + Wf-391-A3 §10 forward-looking:

| Forward-looking item | Wall PROJECTED | USER_OK-gated closure path |
|---|---:|---|
| Actual GPU launch of Wf-390-A1 Path A (15 pairs on metalcytotox_combined n=2000) | ~12 hr | USER_OK=1 + `bash scripts/launch_wf390a1_m2_r2_path_a.sh` |
| Wf-390-A1 Path A.2 (5-metal × 5-seed on metalcytotox_5metal_multitask_v4 n=26.8k) | ~6 hr | NEW launch wrapper or `--pair` overrides on existing wrapper |
| Wf-390-A1 Path A.3 (per-metal singlets × 1 seed each) | ~2 hr | NEW launch wrapper or `--pair` overrides on existing wrapper |
| Wf-390-A1 Path B (chemprop v3 encoder + Hilbert primal-dual) | ~20 hr | NEW trainer extension + USER_OK + launch wrapper |
| Wf-390-A1 Path C (per-cohort separate retrain × 5 sequential) | ~25 hr | 5 NEW launch wrappers OR orchestrator on top of existing wrapper |

All 5 forward-looking items remain BLOCKED on user OK per AGENTS.md §5.1 +
Wf-355-B1 USER_OK gate pattern + kill_then_adapt §0 directive.

---

## §6 Cross-references

### §6.1 Wf-414-A1 artifacts (this wave)
- `scripts/launch_wf390a1_m2_r2_path_a.sh` (NEW ~270 LOC)
- `molmetal/tests/test_wf414a1_m2r2_launcher.py` (NEW ~310 LOC, 10/10 PASS)
- `TODO/completed/376_m2_r2_gpu_retrain_plan.md` (moved from pending/ + SCAFFOLD header)
- `TODO/completed/414a1_m2r2_launcher.md` (this file)
- `molmetal/reports/wf390a1_path_a/DRY_RUN.log` (NEW Wf-363-A1 schema)
- `molmetal/reports/wf390a1_path_a/plan.json` (NEW Wf-414-A1 canonical schema)
- `molmetal/reports/wf390a1_path_a/scaffold_smoke/aggregate.json` (NEW Wf-390-A1 SCAFFOLD smoke)

### §6.2 Predecessor closeouts
- `TODO/completed/390a1_m2_r2_path_a_smoke.md` (Wf-390-A1 SCAFFOLD smoke)
- `TODO/completed/390a2_todo179_deeper_audit.md` (Wf-390-A2 alternative encoder audit)
- `TODO/completed/390a3_se3_sb_wire.md` (Wf-390-A3 SE(3) SB wire-in audit)
- `TODO/completed/367b1_m2_r2_recovery.md` (Wf-367-B1 multi-stream Ridge ensemble HONEST_NEGATIVE)
- `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 FINAL VERDICT 320 LOC)

### §6.3 Reference wrappers
- `scripts/launch_wf124_p3_g5.sh` (Wf-366-A1 USER_OK-gated 5-metal × 3-seed × 50-ep)
- `scripts/launch_wf124_p3_g6.sh` (Wf-378-A2 USER_OK-gated per-innovation ablation)
- `scripts/launch_wf131_p3_deferred_runs.sh` (Wf-355-B1 USER_OK-gated LOMO)
- `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (Wf-355-A3 USER_OK-gated R20 §5)
- `scripts/_dry_run_log_helper.sh` (Wf-363-A1 DRY_RUN.log schema helper)

### §6.4 Standing directives re-confirmed
- **AGENTS.md §5.1** — Standing GPU authorization (USER_OK=1 + VRAM budget check)
- **AGENTS.md §5.2** — Honest framing mandate (MEASURED-vs-PROJECTED tag discipline)
- **AGENTS.md §5.3** — No paper work without authorization (TODO-84 honored; 0 paper/* touches)
- **AGENTS.md §5.4** — Tier-2 journal target + NO wet-lab planning (§10 honored)
- **AGENTS.md §5.5** — Post-wave on-disk verification rule (verify SHIPPED = file on disk; this wave's MEASURED test pass-rates confirm all 4 deliverables on disk)
- **AGENTS.md §5.6** — Atomic write protocol (CPU-only docs+code+test wave; direct Write tool used per the §5.6 opt-in for existing scripts)

### §6.5 Wf-414 wave context
Wf-414 is a CPU-only docs+code+test wave surfacing TODOs not yet covered by
Wf-37{1..7} + Wf-378 + Wf-379 + Wf-380 + Wf-381 + Wf-383 + Wf-384 + Wf-385 +
Wf-386 + Wf-388 + Wf-389 + Wf-390 + Wf-391 + Wf-392 + Wf-393 + Wf-394 + Wf-395
+ Wf-399 ultracode waves. Per Wf-391-A3 §10 forward-looking priority order,
Wf-414-A1 (TODO-376 wrapper closure) sits in Batch 1 (smallest wall + closes
1 of 6 forward-looking USER_OK items).

---

## §7 MEASURED test pass-rates (summary)

| Wave | NEW tests | MEASURED pass-rate | Wall CPU-only |
|---|---:|---|---:|
| Wf-414-A1 | 10 | **10/10 PASS** | ~4.25 s |

**Total NEW tests across Wf-414 wave:** 10 (this wave is single-deliverable).
**Pre-flight v3 89/2/3/94 PRESERVED at Wf-395-B1 baseline** (NOT re-run by Wf-414-A1
CPU-only docs+code+test wave).
**EXPECTED_GATE_COUNTS["all"]** 94 PRESERVED at Wf-353-B1 baseline (no new
pre-flight gates added by Wf-414-A1).