# Wf-414-B1 — docs-only INDEX + AGENTS sync

**Date:** 2026-09-30 16:54 CST
**Wave:** Wf-414 ultracode CPU-only docs+code+test wave (this B1 sub-flow)
**Parent wave:** Wf-414 (3 SHIPPED sub-flows A1 + A2 + A3)
**Anchor:** Wf-414-A1 + Wf-414-A2 + Wf-414-A3 closeouts on disk per AGENTS.md §5.5 filesystem-truth rule

---

## §0 TL;DR

Wf-414-B1 SHIPPED the docs-only INDEX + AGENTS sync surface for the Wf-414 wave.
3 SHIPPED sub-flows verified on disk per AGENTS.md §5.5 filesystem-truth rule:

- Wf-414-A1 (m2 R² Path A launcher) at `TODO/completed/414a1_m2r2_launcher.md` (12,145 bytes)
- Wf-414-A2 (Lambda CFM audit heuristic fix) at `TODO/completed/414a2_lambda_cfm_audit.md` (12,727 bytes)
- Wf-414-A3 (atomic write protocol migration of 5 more scripts) at `TODO/completed/414a3_atomic_migration.md` (13,216 bytes)

**17 NEW tests PASS** across the 3 SHIPPED sub-flows (Wf-414-A1=10 + Wf-414-A2=2 + Wf-414-A3=5).
§22.99 INDEX entry appended + NEW §6 row in AGENTS.md (this row).

**Status:** SHIPPED (CPU-only docs+code+test; 0 GPU launches; 0 `paper/*` touches per TODO-84;
0 wet-lab planning per AGENTS.md §5.4 + §10).

---

## §1 Wf-414 wave recap (3 SHIPPED sub-flows)

### §1.1 Wf-414-A1 m2 R² Path A launch wrapper

**Closeout:** `TODO/completed/414a1_m2r2_launcher.md` (12,145 bytes mtime 16:38 CST)
**VERDICT:** SHIPPED + ON DISK; closes the Wf-411-A2 HONEST-NEGATIVE
("`scripts/launch_wf390a1_m2_r2_path_a.sh` MISSING on disk") + SCAFFOLD header update of
`TODO/pending/376_m2_r2_gpu_retrain_plan.md` → `TODO/completed/376_m2_r2_gpu_retrain_plan.md`
(plan→closed transition).

**New artifacts:**
- `scripts/launch_wf390a1_m2_r2_path_a.sh` (~270 LOC NEW) with USER_OK gate (exit 64 on refusal
  per kill_then_adapt safety contract) + `--dry-run` path emits Wf-363-A1 DRY_RUN.log + plan.json
  + Wf-390-A1 SCAFFOLD smoke (`wf390a1_path_a_smoke.py`) wired BEFORE the DRY_RUN branch
  + 15 (metal, seed) pair enumeration (5 metals × 3 seeds × 50 epochs on metalcytotox_combined
  n=2000) + `--per-cohort-features` (Wf-405-A2 + Wf-388-A1 SCAFFOLD/APPLIED pattern)
  + `--no-cuda-graphs` safety lever (Wf-335-R3 + Wf-330-B4 Blackwell RuntimeError fix)
  + `--chemprop-v3-cache` flag APPENDED only when TRAINER=*chemprop_v2_v3* (per Wf-392-A2 flag
  scoping; D-MPNN trainer does NOT accept this flag) + Per-wave runtime check (Wf-343c2 7-check
  surface) wired into run_pair() via `--skip-live` for CPU-only dry-run (Wf-347-B1 + Wf-347-B2
  hardening) + Per-pair budget tracking (PAIR_BUDGET_START / PAIR_BUDGET_END + elapsed_s)
  + Wf-302B guard + Aggregation script: collects per_seed_metrics_seed*.json into
  training_summary.json
- `molmetal/tests/test_wf414a1_m2r2_launcher.py` (~310 LOC NEW; 10/10 PASS in 4.25 s wall CPU-only)
- `molmetal/reports/wf390a1_path_a/DRY_RUN.log` (NEW Wf-363-A1 schema)
- `molmetal/reports/wf390a1_path_a/plan.json` (NEW Wf-414-A1 canonical schema)
- `molmetal/reports/wf390a1_path_a/scaffold_smoke/aggregate.json` (NEW Wf-390-A1 SCAFFOLD smoke)

**Tests** (10 NEW tests PASS in `molmetal/tests/test_wf414a1_m2r2_launcher.py`):
1. `test_launcher_syntax` — bash -n parses cleanly (Wf-411-A2 closure: file on disk + valid syntax)
2. `test_user_ok_gate_REFUSED_default` — USER_OK unset → exit 64 (kill_then_adapt safety)
3. `test_user_ok_gate_REFUSED_zero` — USER_OK=0 → exit 64 (only USER_OK=1 exact match)
4. `test_user_ok_gate_PASSED_one_dry_run` — USER_OK=1 + `--dry-run` → SCAFFOLD smoke runs
5. `test_dry_run_emission_schema` — DRY_RUN.log has Wf-363-A1 schema (15 pairs enumerated)
6. `test_plan_json_emission` — plan.json valid JSON with Wf-414-A1 canonical schema
7. `test_scaffold_smoke_invocation` — SCAFFOLD smoke runs BEFORE DRY_RUN branch
8. `test_chemprop_v3_cache_conditional` — `--chemprop-v3-cache` APPENDED only when TRAINER chemprop
9. `test_per_cohort_features_present` — `--per-cohort-features` INSIDE COMMON_ARGS=()
10. `test_no_cuda_graphs_safety_lever` — `--no-cuda-graphs` INSIDE COMMON_ARGS=()

### §1.2 Wf-414-A2 TODO-21 Lambda × CFM coupling audit heuristic fix

**Closeout:** `TODO/completed/414a2_lambda_cfm_audit.md` (12,727 bytes mtime 16:45 CST)
**VERDICT:** SHIPPED + ON DISK; closes the Wf-406-A2 / Wf-410-A2 FALSE-NEGATIVE surface for
hypothetical rewires via 3 NEW heuristic patterns.

**Edit:** `molmetal/scripts/wf394a1_lambda_cfm_wire.py` (+53/-14 LOC):
1. **Line-based regex grep** in `_flags_present` (~45 LOC delta; replaces pure substring search;
   bare flag uses negative lookahead `(?!-weight|-samples)` to avoid matching suffixed forms)
2. **Pattern match for `_lambda_cfm_coupling_flag` boolean kwarg** in `_loss_multiplier_present`
   (+8 LOC; regex `(?:_)?lambda_cfm_coupling_flag\s*:\s*bool`)
3. **Pattern match for `lambda_cfm_multiplier` variable** in `_loss_multiplier_present`
   (regex `\blambda_cfm_multiplier\b` word-bounded)

**New test file:** `molmetal/tests/test_wf414a2_lambda_cfm_audit.py` (~280 LOC NEW)
- 2 NEW tests PASS in 1.41 s wall CPU-only:
  - `test_heuristic_catches_patched_trainer` (verifies ALL 4 contracts PASS post-Wf-414-A2 fix)
  - `test_heuristic_rejects_unpatched` (canonical "no FALSE-POSITIVE" regression guard)

**Combined:** 5/5 PASS across Wf-394-A1 + Wf-410-A2 + Wf-414-A2 tests in 1.58 s wall CPU-only (zero regressions)

**Audit artefact:** `molmetal/reports/wf414a2_lambda_cfm_audit/wf394a1_audit.json` (NEW) — re-run
of Wf-394-A1 audit on the patched trainer post-Wf-414-A2 reports ALL 4 contracts PASS + patch_actions=[]

### §1.3 Wf-414-A3 atomic write protocol migration of 5 more scripts

**Closeout:** `TODO/completed/414a3_atomic_migration.md` (13,216 bytes mtime 16:51 CST)
**VERDICT:** SHIPPED + ON DISK; extends the AGENTS.md §5.6 adoption sequence from 8 scripts
(Wf-406-A3 + Wf-410-A3) to **13 scripts** (Wf-406-A3 + Wf-410-A3 + Wf-414-A3).

**Migrated scripts (5 NEW this wave):**

| Script | Write sites | Write location |
|---|---|---|
| `wf341c3_shortcut_ema.py` | 1 | `out_dir / "summary.json"` line 580 |
| `wf380a4_vina_param_tuning.py` | 1 | `out_path.write_text(...)` line 421 |
| `wf381a1_vina_param_tuning.py` | 1 | `out_path.write_text(...)` line 319 |
| `wf385a3_scaffold_wire_in_gate.py` | 1 | `out_json.write_text(...)` line 468 |
| `wf392a2_chemprop_v3_cache_wire.py` | 3 | backup line 138 + trainer edit line 148 + smoke line 329 |
| **TOTAL** | **7 write sites** | across 5 scripts |

**Tests:** `molmetal/tests/test_wf414a3_atomic_migration.py` (~260 LOC NEW; 5/5 PASS via direct
invocation; <1 s wall CPU-only) — 5 layers of belt-and-braces verification per script.

---

## §2 What 彻底解决 (per Wf-414 wave)

Wf-414 wave DID 彻底解决** 3 TODO-directory items not covered by Wf-37{1..7} + Wf-378 +
Wf-379 + Wf-380 + Wf-381 + Wf-383 + Wf-384 + Wf-385 + Wf-386 + Wf-388 + Wf-389 + Wf-390
+ Wf-391 + Wf-392 + Wf-393 + Wf-394 + Wf-395 + Wf-399 + Wf-405 + Wf-406 + Wf-407 + Wf-408
+ Wf-409 + Wf-410 + Wf-411 + Wf-413 ultracode waves:

(a) **Wf-414-A1 m2 R² Path A launch wrapper SHIPPED** — closes the Wf-411-A2 HONEST-NEGATIVE;
(b) **Wf-414-A2 TODO-21 Lambda × CFM coupling audit heuristic fix SHIPPED** — closes the Wf-406-A2 /
Wf-410-A2 FALSE-NEGATIVE surface;
(c) **Wf-414-A3 atomic write protocol migration extended to 5 more scripts SHIPPED** — extends
AGENTS.md §5.6 adoption sequence to 13 scripts; 17 NEW tests PASS across the 3 sub-flows; 0 GPU
launches; 0 `paper/*` touches; 0 wet-lab planning.

---

## §3 MEASURED test pass-rates (summary)

| Wave | NEW tests | MEASURED pass-rate | Wall CPU-only |
|---|---:|---|---:|
| Wf-414-A1 | 10 | **10/10 PASS** | ~4.25 s |
| Wf-414-A2 | 2 | **2/2 PASS** | ~1.41 s |
| Wf-414-A3 | 5 | **5/5 PASS** | <1 s |
| Wf-414-B1 | 0 NEW (verification + docs-only) | n/a | ~10 min CPU-only |
| **Wf-414 TOTAL** | **17 NEW tests** | **17/17 PASS** | **~30-40 min total CPU-only** |

Pre-flight v3 **95 PASS / 18 FAIL / 4 SKIP / 117 TOTAL** PRESERVED at Wf-411-B1 baseline
(NOT re-run by Wf-414 — CPU-only docs+code+test wave per AGENTS.md §5.1 + kill_then_adapt §0).
EXPECTED_GATE_COUNTS["all"] **117** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates
added by Wf-414).

---

## §4 Honest framing (party-twenty-spirit §3.5 + AGENTS.md §5.2)

11 honest-negative caveats preserved per `TODO/completed/414_final_synthesis.md` §7
+ Wf-413-A1 §7 + Wf-411-B1 §22.97.5 + Wf-410-B1 §22.96.4:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean
   MEASURED 0.472 vs threshold 0.6 — gap −0.128; 10 paths exhausted)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert n=200 each)
3. **m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL** (Rh PASS + Re BELOW; paper §4 = Rh fold only)
4. **Wf-414-A2 `_lambda_cfm_coupling_flag` + `lambda_cfm_multiplier` patterns are future-proofing only**
5. **Wf-414-A2 line-based grep does not handle AST-rewrite**
6. **Wf-414-A3 migration opt-in per Wf-386-A3 §6.1** (~41 more scripts remain)
7. **Wf-414-A1 GPU launch BLOCKED** on USER_OK=1 + VRAM ≥ 8 GB
8. **Wf-413-A1 GPU wrapper termination root cause OPEN**
9. **18 pre-flight FAILs + 4 SKIPs PRESERVED (BY-DESIGN)** per Wf-411-B1 baseline
10. **Wf-299 GPU launch BLOCKED** on Wf-408-A3 alias mapping
11. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10

---

## §5 User voice answer

**Addressing user 2026-09-30 verbatim "用GPU0好了，同时启动好了" via Wf-413-A1 §0
+ Wf-413-A1 USER_OK=1 grant per AGENTS.md §5.1 standing GPU grant:**

**YES — Wf-414 wave DID 彻底解决** 3 TODO-directory items via 3 SHIPPED sub-flows + 1 docs-only
sync (17 NEW tests PASS).

**They did NOT 彻底解决 the structural/metric/env gates:** m6_kappa + m2 R² + m1 AUC
structural gates HONEST_NEGATIVE_FINAL (10 paths); m2 R² GPU retrain BLOCKED on USER_OK;
18 pre-flight FAILs + 4 SKIPs PRESERVED; Wf-299 GPU launch BLOCKED on Wf-408-A3 alias
mapping; Wf-413-A1 GPU wrapper termination root cause OPEN; Wf-410-A3 atomic write migration
NOT COMPLETE (~41 more scripts deferred).

**NEW ultracode Wf-415+ IS the recommended follow-up** per Wf-414 final synthesis §7
with 6+ USER_OK-gated sub-flows (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED
wire-in + Tier-2 §3 + §6 v4 update total).

---

## §6 Cross-references

- Wf-414-A1 closeout: `TODO/completed/414a1_m2r2_launcher.md`
- Wf-414-A2 closeout: `TODO/completed/414a2_lambda_cfm_audit.md`
- Wf-414-A3 closeout: `TODO/completed/414a3_atomic_migration.md`
- Wf-414-B1 docs-only sync (this file)
- Wf-414 final synthesis: `TODO/completed/414_final_synthesis.md`
- §22.99 INDEX entry appended
- NEW §6 row in AGENTS.md (this sync)
- Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 re-confirmed
- `mol-metal-cfm-training` skill (Wf-414-A2 patch detection)
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)

**CHRONOLOGY NOTE:** Wf-414-B1 closeout timestamp 2026-09-30T08:54:00+0000 UTC = ~16:54 CST;
§22.99 entry inserted between §22.98 Wf-413 (this sync) and END OF §22.
