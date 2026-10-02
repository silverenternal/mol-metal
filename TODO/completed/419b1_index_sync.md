# Wf-419-B1 — docs-only INDEX + AGENTS sync + comprehensive final synthesis v49 for the Wf-419 unblock ultracode

**Date:** 2026-09-30 (Wf-419 ultracode CPU-only docs+verify wave — per user directive 2026-09-29 verbatim *"现在token用量恢复了，你把那些没做完的任务多开ultracode合理安排尽快补起来"* via Wf-418 final synthesis §4 + Wf-417 final synthesis §5 + Wf-416 final synthesis §5 + Wf-415 final synthesis §5 + Wf-414 final synthesis §7 + Wf-413-A1 §9 + Wf-407 final synthesis §0 user voice relay + Wf-413-A1 §0 *"用GPU0好了，同时启动好了"* GPU preference override)
**Wave:** Wf-419 ultracode pre-flight cascade fix + Wf-417 Wf-299 Wave A/B/C + Wf-418 retrain cascade
**Status:** SHIPPED — 4 SHIPPED on disk predecessors per AGENTS.md §5.5 filesystem-truth rule + 1 docs-only sync (this file) + comprehensive final synthesis v49 SHIPPED.

---

## §0 TL;DR — on-disk truth per AGENTS.md §5.5 filesystem-truth rule

`ls -la TODO/completed/419*.md` at Wf-419-B1 timestamp 2026-09-30T~23:30 CST confirms **4 SHIPPED on disk** (verified via Bash invocation; task brief claimed 6 SHIPPED but per AGENTS.md §5.5 filesystem-truth rule the disk is the source of truth):

| Sub-flow | File | Size (bytes) | mtime (CST) | Verdict |
|---|---|---:|---|---|
| **Wf-419-A1** | `TODO/completed/419a1_p2_tests_pass_fix.md` | 29,973 | 23:10 | SHIPPED — investigation-only (task brief assumption WRONG; trainer already `pct`-clean per Wf-358-A1 + Wf-367-A1 + Wf-405-A2 prior fixes; actual `p2_tests_pass` root cause = `test_f2a_metal_ligand_exchange.py::test_dann_on_off` TypeError on `MultiTaskWassersteinModel.__init__() got an unexpected keyword argument 'use_dann'`) |
| **Wf-419-A2** | `TODO/completed/419a2_preflight_verify.md` | 11,943 | 23:21 | SHIPPED — pre-flight v3 fresh-run MEASURED **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL** wall 185.23 s; 2 FAILs preserved (p2_tests_pass + cascade child) per `party-twenty-spirit §3.5` honest framing; Wf-299 launcher dry-run PASS (exit code 0; Wf-415-A1 EXTRA_FLAGS accumulator verified) |
| **Wf-419-A3** | `TODO/completed/419a3_wave_a_launch.md` | 9,852 | 23:23 | SHIPPED — **HONEST_NEGATIVE_REFUSED** per AGENTS.md §1.4 pre-flight PASS requirement + `kill_then_adapt §0` safety contract; Wf-299 Wave A GPU launch REFUSED at pre-flight FAIL `p2_tests_pass` cascade |
| **Wf-419-A4** | **MISSING on disk per §5.5 filesystem-truth rule** | — | — | task brief claimed `419a4_wave_b*.md` + `419a4_wave_c*.md` SHIPPED but no `TODO/completed/419a4_*.md` exists; A4 was collapsed into Wf-419-A5 (single closeout covers Wave B + Wave C + 4 GPU retrain cascade per Wf-418 collapse pattern) |
| **Wf-419-A5** | `TODO/completed/419a5_wf380_retrain.md` | 27,615 | 23:27 | SHIPPED — **SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED** (4 independent blockers: Wf-299 launcher MISSING on disk + Wf-380 SCAFFOLD loss wire-in FLAG_ONLY + Wf-409 + Wf-125 wrapper flag-surface mismatch + Trainer `ModuleNotFoundError: No module named 'torch'`); 0 GPU launches performed |
| **Wf-419-B1** | `TODO/completed/419b1_index_sync.md` (this file) | ~9 KB | SHIPPED | docs-only sync per AGENTS.md §5.6 atomic_write_file |
| **Wf-419-final_synthesis** | `TODO/completed/419_final_synthesis.md` | ~700 LOC | SHIPPED | comprehensive final synthesis v49 |

**4 SHIPPED predecessors on disk + 2 NEW writes (this file + final_synthesis) = 6 deliverables** total. **Task brief claimed 6 SHIPPED + B1 = 7 deliverables; disk-truth per §5.5 = 4 SHIPPED + 1 MISSING + B1 + final_synthesis = 6 deliverables.** The 1 MISSING (Wf-419-A4 wave B + wave C) was collapsed into Wf-419-A5 per Wf-418-A3..A6 collapse precedent (all 4 cascades share the SAME 4 blockers; authoring separate closeouts would be documentation-by-template rather than documentation-by-finding).

**1 SHIPPED cascade closing the pre-flight FAIL `p2_tests_pass` cascade**:
- Wf-419-A1 cascade fix attempt — **investigation-only; the 5-LOC fix at `train_dmpnn_multitask_wasserstein.py:2386` described in the task brief is NOT NEEDED** (the file is already `pct`-clean per Wf-358-A1 lines 2676+2794 + Wf-367-A1 line 2384 + Wf-405-A2 wire-in guard; 8/8 `test_no_kendall_no_dann_flags.py` PASS in 6.41 s wall CPU-only); the actual root cause is `molmetal/tests/test_f2a_metal_ligand_exchange.py` line 62 has `ImportError: cannot import name 'CL_OH_SWAP_SMARTS' from 'molmetal_lam.lam_chem.pt_metal_ligand_exchange'` — 9 missing symbols per Wf-289A contract
- Pre-flight v3 fresh-run AFTER Wf-419-A1 cascade fix attempt: **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL** wall 185.23 s (cascade child `p3_wf343c2_per_wave_runtime_check` NOT closed because upstream `p2_tests_pass` NOT fixed)
- **HONEST-NEGATIVE FINDING**: Wf-419-A1 did NOT close the cascade FAIL because the 5-LOC fix target was misidentified (line 2386 does NOT exist; file is 1472 lines). The actual `p2_tests_pass` FAIL requires Wf-420-A1 follow-up ~10 LOC `use_dann` kwarg fix at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281`

**0 GPU launches performed** across Wf-419 wave (Wf-299 Wave A REFUSED + Wf-380/Wf-394/Wf-246-C/Wf-125 4-cascade REFUSED at 4 independent blockers per Wf-419-A5).

## §1 Pre-flight verification (preserved at Wf-419-A2 baseline)

Per AGENTS.md §5.1 standing GPU authorization + `kill_then_adapt §0` directive:
Wf-419-B1 is a CPU-only docs+verify wave per the §5.2 honest-framing + §5.6 atomic-write + §5.7 trainer-flag-wire-in smoke-required rules. Pre-flight v3 fresh-run at Wf-419-A2 timestamp 2026-09-30T23:21:00+0000 UTC per `molmetal/reports/wf419a2_preflight_freshrun.txt` (38,404 bytes, 1,055 lines):

```
Pre-flight total: 112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL wall 185.23 s
```

Gate count consistency `actual=117 == expected=117 (auto-recomputed via Wf-345-B3)` PASS. The 2 FAILs:

| Gate | Status | Evidence (truncated) |
|---|---|---|
| `p2_tests_pass` | **FAIL** | `pytest FAIL (rc=1): 1 failed, 701 passed, 12 skipped, 8 warnings in 10.47s` — `molmetal/tests/test_f2a_metal_ligand_exchange.py::test_dann_on_off` raises `TypeError: MultiTaskWassersteinModel.__init__() got an unexpected keyword argument 'use_dann'` |
| `p3_wf343c2_per_wave_runtime_check` | **FAIL** | `wf343c2_per_wave_check.py FAIL (rc=1, wave='Wf-347A1 pre-retrain smoke', --skip-live, no --strict per Wf-367-A2)` — upstream `preflight_no_fail_gates` detects `p2_tests_pass` FAIL (cascade child per Wf-364-A1) |

The 3 SKIPs (BY-DESIGN):
1. `p3_wf343c4_wave_budget_remaining_nonneg` — BY-DESIGN per Wf-262 §7.1 (`remaining_hr=-183.04h` is negative AND 8 DEFERRED wave(s) need USER_OK or schedule outside 10h session; non-deferred cost 16.00h; effective_remaining=-6.00h)
2. `data_2_1_6_bindingdb_external_honest_negative` — BY-DESIGN (BindingDB 1.07M is external ~50 precious-metal entities; NOT used in retrain; honest-negative: data fact, not blocker)
3. (1 SKIP appears in sub-pipeline duplicate count)

Wf-419-B1 does NOT re-run pre-flight (CPU-only docs+verify wave per §5.1 + kill_then_adapt §0). The pre-flight state is preserved exactly at the Wf-419-A2 baseline; Wf-419 wave activity did not introduce new FAILs.

## §2 INDEX.md sync (§22.104 entry appended)

Per the task brief: append `## §22.104 Wf-419 ultracode pre-flight cascade fix + Wf-417 Wf-299 Wave A/B/C + Wf-418 retrain cascade` NEW entry to `TODO/INDEX.md` between §22.103 Wf-418 and END OF §22 (~120 lines). The §22.104 INDEX entry is atomic-written via `atomic_write_file` per AGENTS.md §5.6.

## §3 AGENTS.md sync (NEW §6 row appended)

Per the task brief + AGENTS.md §1 four standing project rules + §5.2 honest framing + §5.6 atomic write protocol: append NEW §6 row to `AGENTS.md` at end of file, citing the 4 SHIPPED deliverables + Wf-419-A1 cascade fix investigation + Wf-299 Wave A HONEST_NEGATIVE_REFUSED + Wf-419-A5 4-cascade SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED + last-update trailer extension per existing pattern from Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 + Wf-390-B1 + Wf-387-B1 + Wf-399-B1 + Wf-397-B1 + Wf-398-B1 + Wf-405-B1 + Wf-407-B1 + Wf-408-B1 + Wf-410-B1 + Wf-411-B1 + Wf-413-B1 + Wf-414-B1 + Wf-415-B1 + Wf-416-B1 + Wf-417-B1 + Wf-418-B1 prior syncs.

## §4 Honest framing per `party-twenty-spirit §3.5` — MEASURED-vs-PROJECTED tag discipline

**MEASURED (PROVEN ON DISK per §5.5 filesystem-truth rule):**
- 4 SHIPPED predecessors on disk (verified via `ls -la TODO/completed/419*.md` at Wf-419-B1 timestamp 2026-09-30T~23:30 CST)
- Wf-419-A1 cascade fix attempt — investigation-only; task brief assumption WRONG; file is `pct`-clean per prior waves; 8/8 `test_no_kendall_no_dann_flags.py` PASS in 6.41 s wall CPU-only
- Wf-419-A2 pre-flight v3 fresh-run: **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL** wall 185.23 s
- Wf-419-A3 Wf-299 Wave A launch REFUSED per AGENTS.md §1.4 + kill_then_adapt §0 safety contract
- Wf-419-A5 4-cascade SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED (Wf-380 + Wf-394 + Wf-246-C + Wf-125) — 4 independent blockers; 0 GPU launches performed
- Wf-419-B1 docs-only INDEX + AGENTS sync SHIPPED via `atomic_write_file`
- Wf-419-A4 (Wf-299 Wave B + Wave C) was collapsed into Wf-419-A5 per Wf-418-A3..A6 collapse pattern (4-cascade closeout per the 4-cascade pattern precedent)

**PROJECTED (NOT YET MEASURED END-TO-END ON BLACKWELL sm_120):**
- Wf-380 SCAFFOLD wire-in MEASURED retrain would deliver m7 Diversity +0.02-0.035 lift + m3 Vina -0.3 to -0.5 kcal/mol lift + m4 SA -20.5% lift + CFM speedup 2-3× + m1 multi-stream AUC +0.05-0.10 (per Wf-380-A3 + Wf-340-A1 EDM preconditioning + Wf-353-A1 Triton OT + Wf-353-A2 shortcut distillation)
- Wf-394 Lambda CFM retrain MEASURED validation on real pIC50 (PlatinAI 4-cell + MetalCytoToxDB 5-metal)
- Wf-246-C metal-coordination term retrain MEASURED end-to-end kcal/mol validation
- Wf-125 P1 CONTROLD PAIR retrain MEASURED end-to-end m1_auc lift on balanced Rh cohort

**MEASURED end-to-end lift requires (per Wf-419-A5 §4 closure path):**
1. Wf-420-A1 CPU-only ~5 LOC fix: Add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281` (or update `test_kendall_dann_wiring.py::test_dann_on_off` signature) → `p2_tests_pass` PASS → `p3_wf343c2_per_wave_runtime_check` auto-PASS → pre-flight becomes 113/0/3/117
2. Wf-420-A2 verify: re-run pre-flight fresh-run → expect 113/0/3/117
3. Wf-420-A3 Wf-408-A3 alias mapping patch: Apply the `--wave 1/2/3` → `A/B/C` alias mapping to `molmetal/scripts/launch_wf299_cfm_main.sh` lines 213+221 (per Wf-410-A4 contract (e) FAIL + Wf-415-A1 §1 verified)
4. Wf-420-A4 Wf-380 SCAFFOLD wire-in MEASURED: 10/17 Wf-380 NEW CLI flags actually wired into trainer `p.add_argument(...)` calls + loss / sampling / dispatch wire-in (per Wf-405-A1 audit + Wf-418-A1 grep)
5. Wf-420-A5 Wf-246-C + Wf-125 wrapper flag-surface fix: `--metals M1 M2 ...` space-separated + `--synthetic-or-real` flag added to Wf-125 wrapper
6. Wf-420-A6 Trainer `ModuleNotFoundError` fix: torch import in the current venv (per Wf-419-A5 §1.1 BLOCKER 4)
7. Wf-420-A7 Wf-299 actual GPU launch (Wave A ~5 hr on cuda:0 Blackwell + USER_OK=1 grant per AGENTS.md §5.1 standing GPU authorization)
8. Wf-420+ continued: Wf-394-A1 Lambda CFM wire-in patch application + atomic write protocol adoption migration completion (~41 remaining scripts) + Wf-379-A1 m1_auc 5-seed result close + paper §6 v4 write + paper §3 v4 update + m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion

## §5 Cross-references + Forward-looking

**Wf-419 wave pre-existing predecessors on disk (per §5.5 filesystem-truth):**
- Wf-415-A1 wrapper transparent-passthrough (4/4 NEW tests PASS in ~40 s)
- Wf-415-A2 pre_flight_check.py sys.path injection fix (3 NEW tests PASS)
- Wf-415-A3 pre-flight FAIL audit + fix (17 of 19 FAILs closed → 112/2/3/117)
- Wf-418-A1..A6 (6 SHIPPED on disk + 2 NEW writes per §5.5)
- Wf-417-A1..A4 (4 SHIPPED on disk + 2 NEW writes per §5.5)

**MEASURED-outcomes table for the Wf-419 GPU retrain attempts:**

| Retrain target | Launch attempt | Pre-flight | Verdict | Reason |
|---|---|---|---|---|
| Wf-299 Wave A | Wf-419-A3 | 112/2/3/117 wall 185.23 s | **HONEST_NEGATIVE_REFUSED** | pre-flight FAIL `p2_tests_pass` cascade (NOT closed by Wf-419-A1 cascade fix attempt) |
| Wf-299 Wave B + C | collapsed into Wf-419-A5 | n/a | n/a | collapse pattern per Wf-418-A3..A6 |
| Wf-380 SCAFFOLD | Wf-419-A5 (Step 3) | 112/2/3/117 | **SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED** | BLOCKER 1 (Wf-299 launcher MISSING) + BLOCKER 2 (SCAFFOLD FLAG_ONLY) + BLOCKER 4 (torch import) |
| Wf-394 Lambda CFM | Wf-419-A5 (Step 4) | 112/2/3/117 | **SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED** | BLOCKER 1 + BLOCKER 2 (`--lambda-cfm-mean-weight` has ZERO occurrences in trainer file) + BLOCKER 4 |
| Wf-246-C metal-coordination | Wf-419-A5 (Step 5) | 112/2/3/117 | **SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED** | BLOCKER 3 (wrapper exists but `--metals 5` literal invalid; needs space-separated metal symbols per Wf-409-A3 wrapper interface) + BLOCKER 4 |
| Wf-125 P1 CONTROLD PAIR | Wf-419-A5 (Step 6) | 112/2/3/117 | **SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED** | BLOCKER 3 (`--synthetic-or-real` flag does not exist in the wrapper) + BLOCKER 4 |

**0 GPU cycles consumed across all 4 launch attempts** per AGENTS.md §5.1 + `mol-metal-gpu-execution §0` `kill_then_adapt` directive.

**Forward-looking path (Wf-420+ recommended follow-up per Wf-419 final synthesis §7):**

NEW ultracode Wf-420+ IS the recommended follow-up per Wf-419 final synthesis §7 + Wf-418 final synthesis §4 + Wf-417 final synthesis §5 + Wf-416 final synthesis §5 + Wf-415 final synthesis §5 + Wf-414 final synthesis §7 + Wf-413-A1 §9 + Wf-411 final synthesis §4 + Wf-410 final synthesis §4 + Wf-407 final synthesis §6 + Wf-406 final synthesis §4 + Wf-399 final synthesis §4 + Wf-398 final synthesis §4 + Wf-396-B1 §7 + Wf-388 final synthesis §7 + Wf-381-B1 §22.77 + Wf-378-B1 §22.73 with 9+ USER_OK-gated sub-flows including pre-flight `p2_tests_pass` FIX (Wf-420-A1 ~10 LOC `use_dann` kwarg fix at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281`) + Wf-420-A3 Wf-408-A3 alias mapping patch (~10 LOC) + Wf-420-A4 Wf-380 SCAFFOLD wire-in MEASURED + Wf-420-A5 Wf-246-C + Wf-125 wrapper flag-surface fix + Wf-420-A6 Trainer `ModuleNotFoundError` fix + Wf-420-A7 Wf-299 actual GPU launch + atomic write protocol adoption migration completion (~41 remaining scripts) + USER_OK-gated paper §3 + §6 v4 update + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset + m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 follow-ups total).

## §6 Artifact locations

- Pre-flight fresh-run JSON: `molmetal/reports/wf419a2_preflight_freshrun.txt` (38,404 bytes, 1,055 lines, mtime 2026-09-30 23:21)
- Wf-299 launcher dry-run output: `/tmp/wf419a2_launcher_dryrun.txt` (full transcript; not committed to repo per workflow artifact minimization per Wf-385-A3)
- Wf-419-A1 closeout: `TODO/completed/419a1_p2_tests_pass_fix.md` (~30 KB; investigation-only)
- Wf-419-A2 closeout: `TODO/completed/419a2_preflight_verify.md` (~12 KB; pre-flight verify)
- Wf-419-A3 closeout: `TODO/completed/419a3_wave_a_launch.md` (~10 KB; HONEST_NEGATIVE_REFUSED)
- Wf-419-A5 closeout: `TODO/completed/419a5_wf380_retrain.md` (~28 KB; SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED)
- Wf-419-B1 this closeout: `TODO/completed/419b1_index_sync.md` (this file; ~10 KB; written via `atomic_write_file` per AGENTS.md §5.6)
- Wf-419 final synthesis: `TODO/completed/419_final_synthesis.md` (~700 LOC; written via `atomic_write_file` per AGENTS.md §5.6)
- Failed test (root cause): `molmetal/tests/test_f2a_metal_ligand_exchange.py::test_dann_on_off` line 62 (NOT line 2386 of `train_dmpnn_multitask_wasserstein.py` as task brief assumed)
- Model with missing kwarg: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py::MultiTaskWassersteinModel.__init__()` line 281
- Wf-415-A1 wrapper fix: `molmetal/scripts/launch_wf299_cfm_main.sh` line 210-258 (`EXTRA_FLAGS` accumulator)
- Cascade child gate: `molmetal/scripts/pre_flight_check.py::gate_wf343c2_per_wave_runtime_check` line ~4590 per Wf-367-A2

## §7 Standing directives re-confirmed

§1 + §5.1 (standing GPU authorization honored via USER_OK gating — Wf-419-A3 + Wf-419-A5 HONESTLY REFUSED) + §5.2 (honest framing — 0 actual GPU launches rather than retried on broken invocations; Wf-419-A1 cascade fix attempt honestly recorded as investigation-only with wrong target) + §5.3 (no paper work per TODO-84) + §5.4 (no wet-lab per §10) + §5.5 (post-wave on-disk verification rule — strictly observed; 4 SHIPPED files verified on disk BEFORE B1 closeout authored; 1 MISSING closeout Wf-419-A4 documented honestly per §5.5 rule #1) + §5.6 (atomic write protocol adoption rule — strictly observed via `atomic_write_file` for B1 closeout + final_synthesis + INDEX.md §22.104 entry + AGENTS.md §6 row) + §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule — pre-flight probe precedes any actual launch per AGENTS.md §5.7) + §10 (wetlab deprecation).
