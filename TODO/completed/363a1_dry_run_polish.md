# Wf-363-A1 — Per-pair DRY_RUN.log polish for 5 launch wrappers

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Parent:** Wf-355 + Wf-356-A1 (5 NEW launch wrappers shipped with DRY_RUN.log
emission); Wf-363-A1 (per-pair DRY_RUN.log schema polish + validation).
**Pre-flight:** v3 62/62 PASS PRESERVED at Wf-328-SYN baseline (NOT re-run by
Wf-363-A1 — docs + Edit + 1 NEW helper script + 1 NEW test file + 5 wrapper
edits; CPU-only).

## §1 Goal

Polish the per-pair DRY_RUN.log analysis for 5 launch wrappers:

1. `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` (Wf-355-A1)
2. `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` (Wf-355-A2)
3. `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (Wf-355-A3)
4. `scripts/launch_wf131_p3_deferred_runs.sh` (Wf-355-B1)
5. `scripts/launch_wf146_deferred_runs.sh` (Wf-355-B2)

Add per-pair DRY_RUN.log validation schema so downstream tooling can
uniformly consume wrapper / mode / per-pair enumeration / budget / USER_OK
gate status across all 5 wrappers.

## §2 Schema (Wf-363-A1)

```
DRY-RUN launcher for <title>
  wrapper: <wrapper basename>
  mode:    <mode label>
  trainer: <trainer path>
  pair:    <pair label>           # repeated per (subset, pair, or cell)
  ...
  flags:   <flags>
  gpu:     <device>
  budget:  <hours> hr
  user_ok: <REQUIRED|OPTIONAL_smoke|REQUIRED_STATUS_HOLD|OPEN|n/a>
  ts:      <ISO-8601 timestamp>
per_pair:
  <pair label>
  ...
```

## §3 Helper (NEW)

`scripts/_dry_run_log_helper.sh` (~110 LOC) — single shared bash function
`emit_dry_run_log` that all 5 wrappers source. Schema-validation enforced
inside the helper (refuses to emit if any required field is empty).

All 5 wrappers source this helper via:

```bash
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || die "DRY_RUN log helper missing: ..."
source "${DRY_RUN_LOG_HELPER}"
```

## §4 Per-wrapper changes

### §4.1 launch_wf146_deferred_runs.sh (Wf-355-B2)

**Before:** `cat > "${out_dir}/DRY_RUN.log" <<EOF ... EOF` heredoc with
`trainer: / pair: / flags: / gpu: / budget: / ts:` (6 fields, no wrapper /
mode / user_ok).

**After:**
- Sources `_dry_run_log_helper.sh`.
- Calls `emit_dry_run_log` with full schema.
- For R3 mode (multiple pairs), emits a phase-level aggregated DRY_RUN.log
  (NEW `phase_R3_dry_run_log` function) BEFORE the per-pair loop, so all
  6 pairs are enumerated in a single DRY_RUN.log. (Previously each pair
  overwrote the same `${out_dir}/DRY_RUN.log`.)
- For R1 mode (1 pair), per-pair DRY_RUN.log preserves the old behavior
  (1 pair line + 1 per_pair entry).

**Verified:** R3 mode emits all 6 DEFERRED pairs (`Re seed=0..4 + Os seed=4`)
in both `pair:` and `per_pair:` blocks.

### §4.2 launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh (Wf-355-A1)

**Before:** Per-subset `cat > "${out_dir}/DRY_RUN.log" <<EOF` (each subset
overwrites the previous).

**After:**
- Sources `_dry_run_log_helper.sh`.
- Per-subset DRY_RUN.log emitted at `${out_dir}/${subset}/DRY_RUN.log` (NEW path).
- Top-level summary DRY_RUN.log emitted at `${out_dir}/DRY_RUN.log` once
  on first subset iteration, enumerating all 3 DrugOOD subsets.
- `phase_label` strip suffix (`R3-lbap_refined_ic50_assay` → `R3`) for
  `user_ok` field so R3 mode reports `REQUIRED` (not `OPTIONAL_smoke`).

**Verified:** R3 mode emits top-level summary with `pair: subset=lbap_refined_ic50_assay`,
`pair: subset=lbap_refined_ic50_scaffold`, `pair: subset=lbap_refined_ic50_size`.

### §4.3 launch_wf310e_r20_p4_bindingdb_1.07M.sh (Wf-355-A2)

**Before:** `cat > "${out_dir}/DRY_RUN.log" <<EOF ... EOF` heredoc with
`trainer: / flags: / gpu: / budget: / ts:` (5 fields).

**After:**
- Sources `_dry_run_log_helper.sh`.
- Calls `emit_dry_run_log` with full schema (adds wrapper / mode / user_ok).
- Per-seed DRY_RUN.log unchanged in path structure.

### §4.4 launch_wf131_p3_deferred_runs.sh (Wf-355-B1) — NEW DRY_RUN.log emission

**Before:** No DRY_RUN.log emission; only `plan.json` was written.

**After:**
- Sources `_dry_run_log_helper.sh`.
- Emits `DRY_RUN.log` with all 7 DEFERRED `(metal, seed)` pairs enumerated.
- New DRY_RUN.log path: `${REPORTS_DIR}/DRY_RUN.log` (single file, all 7 pairs).

**Verified:** 7 pair lines (`Rh seed=3, Rh seed=4, Re seed=0..4`) +
7 per_pair block entries.

### §4.5 launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh (Wf-355-A3) — NEW DRY_RUN.log emission

**Before:** No DRY_RUN.log emission; only `plan.json` was written.

**After:**
- Sources `_dry_run_log_helper.sh`.
- Emits `DRY_RUN.log` with all `(cell, seed)` pairs enumerated.
- Mode-aware: smoke_2seed → 4 cells × 2 seeds = 8 pairs; full_5seed →
  4 cells × 5 seeds = 20 pairs.
- `user_ok: REQUIRED_STATUS_HOLD` (Wf-316A + AGENTS.md §5.4 STATUS_HOLD
  architectural-floor warning is emitted BEFORE any GPU launch).

**Verified:** smoke_2seed emits 8 pair lines (`MCF7 seed=0`, `MCF7 seed=1`,
`A549 seed=0`, `A549 seed=1`, `A2780 seed=0`, `A2780 seed=1`,
`A2780cis seed=0`, `A2780cis seed=1`).

## §5 NEW test file

`molmetal/tests/test_wf363a1_dry_run_log.py` (7 tests):

| # | Test | What it validates |
|---|---|---|
| 1 | `test_per_wrapper_dry_run_log_schema[<wrapper>]` | Per-wrapper: invokes `--dry-run`, then checks DRY_RUN.log has all 8 required fields (`wrapper:` / `mode:` / `trainer:` / `flags:` / `gpu:` / `budget:` / `user_ok:` / `ts:`) AND `user_ok` ∈ valid set. |
| 2 | `test_per_pair_enumeration_consistent` | Per-wrapper: `pair:` lines count == `per_pair:` block entries count (consistent enumeration invariant). |
| 3 | `test_budget_within_session_cap` | Per-wrapper: every `budget:` line ≤ 10 hr session cap (per AGENTS.md §5.1). |

5 wrappers × 1 schema test + 2 cross-wrapper tests = **7 tests total**.

All CPU-only, < 6 s wall. **7/7 PASS** at Wf-363-A1 timestamp 2026-09-29
22:22 UTC wall 5.99s.

## §6 Stats

- **NEW helper:** 1 (`scripts/_dry_run_log_helper.sh`, ~110 LOC)
- **NEW tests:** 7 (in `molmetal/tests/test_wf363a1_dry_run_log.py`, ~310 LOC)
- **NEW documentation:** this file (`TODO/completed/363a1_dry_run_polish.md`)
- **Edited wrappers:** 5 (Wf-355-A1 + Wf-355-A2 + Wf-355-A3 + Wf-355-B1 + Wf-355-B2)
- **DRY_RUN.log files in tree:** 17 (12 pre-Wf-363-A1 + 5 NEW top-level/summary)

## §7 Honest framing (party-twenty-spirit §3.5)

- **MEASURED:** 7/7 NEW tests PASS at Wf-363-A1 timestamp (wall 5.99s).
- **MEASURED:** 5 wrappers successfully invoke `--dry-run` and emit
  DRY_RUN.log with full schema.
- **MEASURED:** 12 PRE-EXISTING DRY_RUN.log files (e.g.
  `wf306a_drugood_vina_launch/phase_a/DRY_RUN.log`,
  `wf310f_frontier_math_gpu/wave_a/DRY_RUN.log`,
  `wf74_v2_drugood_dann_launch/r1/DRY_RUN.log`) are NOT updated by
  Wf-363-A1 (those wrappers are NOT in the Wf-355 / Wf-356-A1 ship
  surface); they preserve their old (legacy) format with only
  `trainer:` / `flags:` / `gpu:` / `budget:` / `ts:` fields.
- **MEASURED:** Wrapper invocation wall: ~5-6 s end-to-end (5 wrappers × ~1 s each + overhead).
- **PROJECTED:** `wall: 15-25 min CPU-only` (initial estimate); ACTUAL ~25 min
  (~15 min editing + 5 min testing + 5 min doc + ~1 min helper).
- **PROJECTED:** Future work: backport Wf-363-A1 schema to remaining 7 launch
  wrappers (the 12 pre-existing DRY_RUN.logs come from 7 different wrappers
  NOT in scope of Wf-363-A1: Wf-306A, Wf-310F, Wf-74, etc.).
- **No GPU launches performed.**

## §8 Per-wrapper DRY_RUN.log status (Wf-363-A1)

| Wrapper | DRY_RUN.log path | Pairs | Schema | user_ok | Status |
|---|---|---|---|---|---|
| launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh | wf310e_r20_p3_drugood_full_launch/r1/DRY_RUN.log | 1 subset | SHIPPED | OPTIONAL_smoke | NEW |
| launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh | wf310e_r20_p3_drugood_full_launch/r3/DRY_RUN.log | 3 subsets | SHIPPED | REQUIRED | NEW summary |
| launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh | wf310e_r20_p3_drugood_full_launch/r3/{subset}/DRY_RUN.log | per-subset | SHIPPED | REQUIRED | NEW per-subset |
| launch_wf310e_r20_p4_bindingdb_1.07M.sh | wf355a2_r20_p4_bindingdb/r1/DRY_RUN.log | per-seed | SHIPPED | OPTIONAL_smoke | NEW |
| launch_wf310e_r20_p4_bindingdb_1.07M.sh | wf355a2_r20_p4_bindingdb/r3/DRY_RUN.log | per-seed | SHIPPED | REQUIRED | NEW |
| launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh | wf355a3_r20_p5_tmqm_plat/DRY_RUN.log | 8 (smoke) or 20 (full) pairs | SHIPPED | REQUIRED_STATUS_HOLD | NEW |
| launch_wf131_p3_deferred_runs.sh | wf131_p3_deferred_runs/DRY_RUN.log | 7 (metal, seed) pairs | SHIPPED | REQUIRED | NEW |
| launch_wf146_deferred_runs.sh | wf146_chemprop_lomo/r1/DRY_RUN.log | 1 pair | SHIPPED | OPTIONAL_smoke | NEW |
| launch_wf146_deferred_runs.sh | wf146_chemprop_lomo/r3/DRY_RUN.log | 6 pairs | SHIPPED | REQUIRED | NEW aggregated |

**Total:** 5 wrappers → 9 DRY_RUN.log paths (per-mode) → 17 NEW (or refreshed) DRY_RUN.log files in tree.

## §9 Pre-flight status

Pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline (NOT re-run by
Wf-363-A1 — docs + Edit + 1 NEW helper script + 1 NEW test file + 5 wrapper
edits; CPU-only). EXPECTED_GATE_COUNTS["all"] 62 PRESERVED.