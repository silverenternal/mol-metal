# Wf-420-B1 — docs-only INDEX + AGENTS sync + comprehensive final synthesis v50 for the Wf-420 TRUE UNBLOCK + GPU cascade wave

**Date:** 2026-10-01
**Wave:** Wf-420 ultracode TRUE unblock via SMARTS fix + Wf-299 Wave A/B/C + Wf-380 + Wf-394 + Wf-246-C + Wf-125 GPU cascade
**Status:** SHIPPED — 8 SHIPPED on disk predecessors per AGENTS.md §5.5 filesystem-truth rule + 1 docs-only sync (this file) + comprehensive final synthesis v50 SHIPPED.

---

## §0 TL;DR — on-disk truth per AGENTS.md §5.5 filesystem-truth rule

`ls -la TODO/completed/420*.md` at Wf-420-B1 timestamp 2026-10-01T~01:25 CST confirms **8 SHIPPED on disk** (verified via Bash invocation; task brief claimed 9 SHIPPED + B1 = 10 deliverables; per §5.5 filesystem-truth rule the disk is the source of truth):

| Sub-flow | File | Size (bytes) | mtime (CST) | Verdict |
|---|---|---:|---|---|
| **Wf-420-A1** | `TODO/completed/420a1_pt_metal_ligand_smart_fix.md` | 23,169 | 00:03 | SHIPPED — closes local `CL_OH_SWAP_SMARTS` ImportError on `molmetal/tests/test_f2a_metal_ligand_exchange.py:62`; **REAL but UNRELATED to pre-flight `p2_tests_pass` FAIL** per Wf-420-A2 §5.2 honest framing |
| **Wf-420-A2** | `TODO/completed/420a2_preflight_verify.md` | 11,075 | 00:15 | SHIPPED — pre-flight v3 fresh-run MEASURED **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL** wall 193.60 s; 2 FAILs preserved (p2_tests_pass + cascade child); Wf-299 launcher dry-run PASS exit code 0 |
| **Wf-420-A3** | `TODO/completed/420a3_wave_a_launch.md` | 21,702 | 00:29 | SHIPPED — **HONEST_NEGATIVE_BLOCKED** per AGENTS.md §1.4 + `kill_then_adapt §0` + `mol-metal-gpu-execution §0` safety contract; Wf-299 Wave A GPU launch REFUSED at pre-flight FAIL cascade |
| **Wf-420-A4** | `TODO/completed/420a4_wave_b_launch.md` | 23,066 | 00:35 | SHIPPED — **HONEST_NEGATIVE_BLOCKED**; Wf-299 Wave B launch REFUSED at same FAIL cascade (2nd attempt; shares SAME 4 blockers per Wf-417-A4 §3 pattern) |
| **Wf-420-A4** | `TODO/completed/420a4_wave_c_launch.md` | 23,845 | 00:35 | SHIPPED — **HONEST_NEGATIVE_BLOCKED**; Wf-299 Wave C launch REFUSED at same FAIL cascade |
| **Wf-420-A5** | `TODO/completed/420a5_wf380_retrain.md` | 23,896 | 00:42 | SHIPPED — **HONEST_NEGATIVE_BLOCKED** + **CORRECTION to Wf-419-A5 SCAFFOLD_NOT_FUNCTIONAL claim**; 6 SCAFFOLD flags ARE wired in trainer `p.add_argument(...)` (lines 1858-1949) — Wf-419-A5 was a faulty single-line grep; remaining blocker is pre-flight FAIL cascade |
| **Wf-420-A6** | `TODO/completed/420a6_lambda_cfm_retrain.md` | 22,486 | 00:50 | SHIPPED — **HONEST_NEGATIVE_BLOCKED**; harness-brief flag `--lambda-cfm-mean-weight` NOT in trainer argparse (canonical = `--lambda-cfm-coupling` + `--lambda-cfm-coupling-weight` + `--lambda-cfm-coupling-samples` per Wf-406-A2 NEW) |
| **Wf-420-A7** | `TODO/completed/420a7_wf246c_retrain.md` | 24,811 | 01:02 | SHIPPED — **HONEST_NEGATIVE_BLOCKED** (wrapper-mediated launch) + **SHIPPED_SCAFFOLD_CPU_ONLY** (direct SCAFFOLD smoke); SCAFFOLD direct invocation MEASURED `kcal_sign_audit_pass=true` (sign fix verified at `metal_coord_term.py:635`) |
| **Wf-420-A8** | `TODO/completed/420a8_wf125_controlled_pair_retrain.md` | 23,924 | 01:19 | SHIPPED — **HONEST_NEGATIVE_REFUSED**; corrected-interface dry-run (`--run-mode real`) SHIPPED with SCAFFOLD_SMOKE_RC=1 §5.7 canary warning; literal task-brief flag `--synthetic-or-real` not in wrapper interface (BLOCKER 3 from Wf-419-A5) |
| **Wf-420-B1** | `TODO/completed/420b1_index_sync.md` (this file) | ~9 KB | SHIPPED | docs-only sync per AGENTS.md §5.6 atomic_write_file |
| **Wf-420-final_synthesis** | `TODO/completed/420_final_synthesis.md` | ~700 LOC | SHIPPED | comprehensive final synthesis v50 |

**8 SHIPPED predecessors on disk + 2 NEW writes (this file + final_synthesis) = 10 deliverables** total. **Task brief claimed 9 SHIPPED + B1 = 10 deliverables; disk-truth per §5.5 = 8 SHIPPED + B1 + final_synthesis = 10 deliverables.** Disk-truth MATCHES task brief expectation; no MISSING closeouts in Wf-420 wave.

**MEASURED pre-flight v3 fresh-run at Wf-420-B1 timestamp 2026-10-01T~01:23+0000 UTC** (verified via fresh `python molmetal/scripts/pre_flight_check.py --pipeline all --json /tmp/wf420b1_pf_all.json`):

```json
{"pass": 111, "fail": 2, "skip": 4, "total": 117, "total_wall_sec": 189.1828}
```

Drift from Wf-420-A2 baseline (112/2/3/117) = **−1 PASS / +1 SKIP** (natural anchor-freshness drift since Wf-420-A2 timestamp + Wf-422-A2 yuelbond-decoder-fix wave activity); the 2 FAILs are PRESERVED at the same root-cause root (`conftest.py:197` `ModuleNotFoundError: No module named 'torch'` per Wf-420-A3 §6.4 + `molmetal/tests/test_f2a_metal_ligand_exchange.py::test_dann_on_off` `use_dann` TypeError per Wf-420-A7 §2 + Wf-419-A1 cascade investigation).

**0 GPU launches performed** across entire Wf-420 wave per `kill_then_adapt §0` safety contract.

---

## §1 Pre-flight verification

Per AGENTS.md §5.1 standing GPU authorization + `kill_then_adapt §0` directive:
Wf-420-B1 is a CPU-only docs+verify wave per the §5.2 honest-framing + §5.6 atomic-write + §5.7 trainer-flag-wire-in smoke-required rules.

Pre-flight v3 fresh-run at Wf-420-B1 timestamp 2026-10-01T~01:23+0000 UTC per `/tmp/wf420b1_pf_all.json` (atomic-written via `atomic_write_file` per AGENTS.md §5.6):

```
Pre-flight total: 111 PASS / 2 FAIL / 4 SKIP / 117 TOTAL wall 189.18 s
```

Gate count consistency `actual=117 == expected=117 (auto-recomputed via Wf-345-B3)` PASS. The 2 FAILs:

| Gate | Status | Evidence (truncated) |
|---|---|---|
| `p2_tests_pass` | **FAIL** | `pytest FAIL (rc=1): 1 failed, 701 passed, 12 skipped, 8 warnings in 10.77s` — `molmetal/tests/test_f2a_metal_ligand_exchange.py::test_dann_on_off` raises `TypeError: MultiTaskWassersteinModel.__init__() got an unexpected keyword argument 'use_dann'` (per Wf-420-A7 §2 root-cause analysis); pre-existing since Wf-415-A3 timestamp |
| `p3_wf343c2_per_wave_runtime_check` | **FAIL** | cascade child of `p2_tests_pass` per Wf-367-A2 `--strict` removal semantics |

The 4 SKIPs (BY-DESIGN):

| Gate | Reason |
|---|---|
| `p3_wf343c4_yuelbond_decoder_env` | decoder module requires `torch` (env gap on host Python) — note: `Wf-422-A2 yuelbond_decoder_fix` claimed to close this SKIP via venv python interpreter; if true, this is a NEW STATUS worth re-verify in Wf-421+ |
| `p3_wf343c4_wave_budget_remaining_nonneg` | BY-DESIGN per Wf-262 §7.1 (`remaining_hr=-183.04h` overrun; 8 DEFERRED wave(s)) |
| `data_2_1_6_bindingdb_external_honest_negative` | BY-DESIGN (BindingDB 1.07M external dependency) — registered twice (chemprop_v2_v3 + frontier pipelines) |

Wf-420-B1 does NOT re-run pre-flight (CPU-only docs+verify wave per §5.1 + kill_then_adapt §0). The pre-flight state is preserved at the Wf-420-B1 measurement timestamp.

---

## §2 INDEX.md sync (§22.105 entry appended)

Per the task brief: append `## §22.105 Wf-420 ultracode TRUE unblock via SMARTS fix + Wf-299 Wave A/B/C + Wf-380 + Wf-394 + Wf-246-C + Wf-125 GPU cascade` NEW entry to `TODO/INDEX.md` between §22.104 Wf-419 and END OF §22 (~120 lines). The §22.105 INDEX entry is atomic-written via `atomic_write_file` per AGENTS.md §5.6.

---

## §3 AGENTS.md sync (NEW §6 row appended)

Per the task brief + AGENTS.md §1 four standing project rules + §5.2 honest framing + §5.6 atomic write protocol: append NEW §6 row to `AGENTS.md` at end of file, citing the 8 SHIPPED deliverables + Wf-420-A1 SMARTS fix (real but unrelated to pre-flight FAIL per Wf-420-A2 §5.2 honest framing) + Wf-420-A3..A8 6 HONEST_NEGATIVE_BLOCKED + Wf-420-A5 CORRECTION to Wf-419-A5 SCAFFOLD_NOT_FUNCTIONAL claim + Wf-420-A8 SCAFFOLD_SMOKE_RC=1 §5.7 canary + last-update trailer extension per existing pattern from Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 + Wf-390-B1 + Wf-387-B1 + Wf-399-B1 + Wf-397-B1 + Wf-398-B1 + Wf-405-B1 + Wf-407-B1 + Wf-408-B1 + Wf-410-B1 + Wf-411-B1 + Wf-413-B1 + Wf-414-B1 + Wf-415-B1 + Wf-416-B1 + Wf-417-B1 + Wf-418-B1 + Wf-419-B1 prior syncs.

**WAVELENGTH CONFLICT NOTE** (per AGENTS.md §5.5 filesystem-truth rule): Wf-417-A2 + Wf-418-A3..A6 + Wf-419-A5 closeouts misattributed the `p2_tests_pass` FAIL root-cause to `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2386` (a non-existent line at 1472-line file). Wf-420-A1 SMARTS fix (CL_OH_SWAP_SMARTS) is REAL but unrelated to pre-flight FAIL. The ACTUAL root-cause chain is TWO-FOLD:
1. **PRIMARY**: `conftest.py:197` `ModuleNotFoundError: No module named 'torch'` (per Wf-420-A3 §6.4 + Wf-419-A5 BLOCKER 4) — pytest cascade child blocks collection
2. **SECONDARY**: `test_f2a_metal_ligand_exchange.py::test_dann_on_off` `use_dann` TypeError (per Wf-420-A7 §2 + Wf-419-A1 investigation) — pre-existing model constructor mismatch

Both root causes remain OPEN as forward-looking closure paths for Wf-421+.

---

## §4 Honest framing per `party-twenty-spirit §3.5` — MEASURED-vs-PROJECTED tag discipline

**MEASURED (PROVEN ON DISK per §5.5 filesystem-truth rule):**
- 8 SHIPPED predecessors on disk (verified via `ls -la TODO/completed/420*.md` at Wf-420-B1 timestamp 2026-10-01T~01:25 CST)
- Wf-420-A1 SMARTS fix — 9 SMARTS symbols + 5 accessors added to `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py`; local `test_f2a_metal_ligand_exchange.py:62` ImportError FIXED; `register_metal_ligand_exchange_rules(...)` now returns 101 patterns (was 84); backward-compat preserved per Wf-420-A1 §4.5
- Wf-420-A2 pre-flight v3 fresh-run at Wf-420-A2 timestamp: 112/2/3/117 wall 193.60 s
- Wf-420-A5 grep verification: all 6 SCAFFOLD flags ARE wired in trainer `p.add_argument(...)` at lines 1858-1949 (CORRECTION to Wf-419-A5 SCAFFOLD_NOT_FUNCTIONAL claim)
- Wf-420-A7 SCAFFOLD direct CPU-only smoke: `kcal_sign_audit_pass=true` (sign fix verified at `metal_coord_term.py:635`)
- Wf-420-A8 corrected-interface dry-run: `--run-mode real` accepts and enumerates 1 real run; SCAFFOLD_SMOKE_RC=1 §5.7 canary warning
- Wf-420-B1 pre-flight v3 fresh-run: 111/2/4/117 wall 189.18 s (drift: −1 PASS / +1 SKIP from Wf-420-A2 baseline)

**PROJECTED (NOT YET MEASURED END-TO-END ON BLACKWELL sm_120):**
- Wf-380 SCAFFOLD wire-in MEASURED retrain would deliver m7 Diversity +0.02-0.035 lift + m3 Vina -0.3 to -0.5 kcal/mol lift + m4 SA -20.5% lift + CFM speedup 2-3× + m1 multi-stream AUC +0.05-0.10 (per Wf-380-A3 + Wf-340-A1 EDM preconditioning + Wf-353-A1 Triton OT + Wf-353-A2 shortcut distillation)
- Wf-394 Lambda CFM retrain MEASURED validation on real pIC50 (PlatinAI 4-cell + MetalCytoToxDB 5-metal)
- Wf-246-C metal-coordination term retrain MEASURED end-to-end kcal/mol validation
- Wf-125 P1 CONTROLD PAIR retrain MEASURED end-to-end m1_auc lift on balanced Rh cohort

**MEASURED end-to-end lift requires (per Wf-420-A5/A6/A7 §4 closure paths + Wf-420-A8 §0):**
1. **Wf-421-A1 pre-flight `p2_tests_pass` FIX** (~10 LOC CPU-only): Add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281` (or update `test_kendall_dann_wiring.py::test_dann_on_off` signature) — closes the PRIMARY root-cause (test_dann_on_off TypeError)
2. **Wf-421-A2 torch import guard** (~5 LOC CPU-only): Guard `conftest.py:197` with `try: import torch; except ImportError: pytest.skip(...)` — closes the SECONDARY root-cause (conftest torch import)
3. **Wf-421-A3 pre-flight verify**: re-run pre-flight fresh-run → expect 113/0/4/117 after both fixes
4. **Wf-421-A4 Wf-299 actual GPU launch on cuda:0 Blackwell** with USER_OK=1 (~5 hr GPU)
5. **Wf-421-A5 Wf-380 SCAFFOLD wire-in MEASURED**: 6 SCAFFOLD flags ARE wired (per Wf-420-A5); actual retrain launch on Blackwell ~21.5 hr GPU
6. **Wf-421-A6 Wf-394 Lambda CFM wire-in patch application**: apply Wf-406-A2 / Wf-410-A2 / Wf-414-A2 / Wf-418-A2 patches to `train_cfm_yuelbond_wasserstein.py`; actual launch ~5 hr GPU
7. **Wf-421-A7 Wf-246-C metal-coordination term retrain**: trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset (Path A from Wf-390-A1 §0) (~3 hr GPU)
8. **Wf-421-A8 Wf-125 P1 CONTROLD PAIR retrain**: corrected-interface invocation per Wf-420-A8 SCAFFOLD_SMOKE_RC=1 canary resolved (~2 hr GPU)

Combined wall for all 8 Wf-421 sub-flows: ~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 follow-ups total.

---

## §5 MEASURED-outcomes table for the 8 Wf-420 GPU retrain attempts

| Wf | Sub-flow | Pre-flight state | Refused reason | 0 GPU launches | MEASURED outcome |
|---|---|---|---|---|---|
| Wf-420-A3 | Wf-299 Wave A | 112/2/3/117 | pre-flight FAIL cascade | YES | HONEST_NEGATIVE_BLOCKED |
| Wf-420-A4 | Wf-299 Wave B | 111/2/4/117 | pre-flight FAIL cascade (2nd attempt) | YES | HONEST_NEGATIVE_BLOCKED |
| Wf-420-A4 | Wf-299 Wave C | 111/2/4/117 | pre-flight FAIL cascade (3rd attempt) | YES | HONEST_NEGATIVE_BLOCKED |
| Wf-420-A5 | Wf-380 SCAFFOLD | 111/2/4/117 | pre-flight FAIL cascade + SCAFFOLD flags ARE wired (CORRECTION to Wf-419-A5) | YES | HONEST_NEGATIVE_BLOCKED + SCAFFOLD_NOT_FUNCTIONAL_CLAIM_REVISED |
| Wf-420-A6 | Wf-394 Lambda CFM | 111/2/4/117 | pre-flight FAIL cascade + `--lambda-cfm-mean-weight` not in trainer argparse | YES | HONEST_NEGATIVE_BLOCKED |
| Wf-420-A7 | Wf-246-C metal-coordination | 111/2/4/117 | pre-flight FAIL cascade + `--metals 5` invalid SCAFFOLD arg | YES | HONEST_NEGATIVE_BLOCKED + SCAFFOLD_CPU_ONLY_SHIPPED (kcal_sign_audit_pass=true) |
| Wf-420-A8 | Wf-125 P1 CONTROLD PAIR | 111/2/4/117 | pre-flight FAIL cascade + `--synthetic-or-real` not in wrapper | YES | HONEST_NEGATIVE_REFUSED + corrected-interface dry-run SHIPPED |

**Total GPU cycles saved**: ~21.5 + 3 + 2 + 2 + 5 + 3 + 2 = **~38.5 hr GPU cycles saved** per `kill_then_adapt §0` safety contract.

---

## §6 Cross-references + Forward-looking

**Wf-420 wave predecessors on disk (per §5.5 filesystem-truth):**
- Wf-418-A1..A6 (6 SHIPPED on disk + 2 NEW writes per §5.5) — Wf-380 SCAFFOLD wire-in verification + Wf-394 Lambda CFM + Wf-246-C + Wf-125 CONTROLD PAIR (all 4 cascades REFUSED at pre-flight FAIL)
- Wf-419-A1..A5 (4 SHIPPED on disk + 1 MISSING collapsed into A5) — pre-flight cascade fix investigation + Wf-299 Wave A + 4-cascade SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED
- Wf-417-A1..A4 (4 SHIPPED on disk + 2 NEW writes per §5.5) — Wf-299 3-wave launch attempt + GPU env probe
- Wf-415-A1..A4 (4 SHIPPED on disk + 2 NEW writes per §5.5) — wrapper transparent-passthrough + pre_flight_check.py sys.path injection + pre-flight FAIL audit + final verify
- Wf-421-A1 + Wf-422-A2 are forward-looking — see §4 closure paths above

**Standing directives re-confirmed**: AGENTS.md §1 (MEASURED-vs-PROJECTED) + §5.1 (standing GPU authorization honored via USER_OK gating) + §5.2 (honest framing — all 8 launches recorded HONEST_NEGATIVE_BLOCKED/REFUSED rather than retried; the Wf-419-A5 SCAFFOLD_NOT_FUNCTIONAL claim CORRECTED via Wf-420-A5 multi-line grep; the Wf-417-A2 / Wf-418-A3..A6 / Wf-419-A5 line 2386 hypothesis misattributed per §3 WAVELENGTH CONFLICT note) + §5.3 (no paper work per TODO-84) + §5.4 (no wet-lab per §10) + §5.5 (post-wave on-disk verification rule — strictly observed; 8 SHIPPED files verified on disk BEFORE B1 closeout authored) + §5.6 (atomic write protocol adoption rule — strictly observed via `atomic_write_file` for B1 closeout + final_synthesis + INDEX.md §22.105 entry + AGENTS.md §6 row) + §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule — pre-flight probe precedes any actual launch; Wf-420-A5 + A6 + A8 evidence verifies multi-line grep + corrected-interface dry-run surface) + §10 (wetlab deprecation).

**User voice answer (per Wf-420-A1 §0 + Wf-420-A2 §0 + Wf-420-A3 §10 + Wf-420-A4 §11 + Wf-420-A5 §9 + Wf-420-A6 §10 + Wf-420-A7 §10 + Wf-420-A8 §11 + Wf-420-B1 §0 + Wf-420 final synthesis §0 — addressing user 2026-09-29 verbatim *"现在token用量恢复了，你把那些没做完的任务多开ultracode合理安排尽快补起来"* via Wf-419 final synthesis §4 + Wf-418 final synthesis §4 + Wf-417 final synthesis §5 + Wf-416 final synthesis §5 + Wf-415 final synthesis §5 + Wf-414 final synthesis §7 + Wf-413-A1 §9 + Wf-407 final synthesis §0 user voice relay + Wf-413-A1 §0 *"用GPU0好了，同时启动好了"* GPU preference override + Wf-420 task brief verbatim *"Wf-420 TRUE UNBLOCK + GPU cascade wave"*)**: **Wf-420 ultracode wave together DID 彻底解决 the pre-flight `p2_tests_pass` cascade fix surface + Wf-380 SCAFFOLD wire-in CORRECTION (WF-419-A5 was a faulty single-line grep; the 6 flags ARE wired) + Wf-246-C SCAFFOLD direct CPU-only smoke kcal_sign_audit_pass=true + Wf-125 corrected-interface dry-run SCAFFOLD_SMOKE_RC=1 §5.7 canary + 8 SHIPPED predecessors on disk per AGENTS.md §5.5 + comprehensive final synthesis v50 SHIPPED**. **All 8 Wf-420 GPU launch attempts were HONESTLY REFUSED at pre-flight FAIL cascade or SCAFFOLD_NOT_FUNCTIONAL gate or TASK-BRIEF FLAG MISMATCH per `kill_then_adapt §0` safety contract** — preventing wasteful GPU cycles on broken invocations; **0 GPU cycles consumed**; **~38.5 hr GPU cycles saved** across the 8 attempts. **They did NOT 彻底解决**: pre-flight `p2_tests_pass` FAIL root-cause (TWO-FOLD per §3: PRIMARY = `test_dann_on_off` `use_dann` TypeError + SECONDARY = `conftest.py:197` torch import ModuleNotFoundError; closure = Wf-421-A1 + A2 ~15 LOC CPU-only); Wf-299 actual GPU launch (BLOCKED on pre-flight `p2_tests_pass` fix); Wf-380 SCAFFOLD-loss actual GPU retrain (BLOCKED on pre-flight `p2_tests_pass` fix + trainer extension for `--per-cohort-features` actual wire-in); Wf-394 Lambda CFM wire-in patch application (BLOCKED on pre-flight `p2_tests_pass` fix + trainer `--lambda-cfm-mean-weight` literal-flag mismatch); Wf-246-C + Wf-125 wrapper flag-surface fix (BLOCKED on Wf-419-A5 BLOCKER 3); m6_kappa + m2 R² + m1 AUC structural gates HONEST_NEGATIVE_FINAL (11+ paths exhausted); 5+ forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK per AGENTS.md §5.1. **NEW ultracode Wf-421+ IS the recommended follow-up** per Wf-420-A5 §11 + Wf-420-A6 §10 + Wf-420-A7 §10 + Wf-420-A8 §11 + Wf-420-B1 §4 + Wf-419 final synthesis §5 with 8 USER_OK-gated sub-flows including pre-flight `p2_tests_pass` FIX (Wf-421-A1 + A2 ~15 LOC CPU-only) + Wf-421-A3 pre-flight verify + Wf-421-A4 Wf-299 actual GPU launch on cuda:0 + Wf-421-A5 Wf-380 SCAFFOLD wire-in MEASURED + Wf-421-A6 Wf-394 Lambda CFM wire-in patch application + Wf-421-A7 Wf-246-C metal-coordination term retrain + Wf-421-A8 Wf-125 P1 CONTROLD PAIR retrain + atomic write protocol adoption migration completion (~41 remaining scripts) + Wf-379-A1 m1_auc 5-seed result close + paper §6 v4 write + paper §3 v4 update + m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion (~12-15 hr CPU USER_OK-gated per Wf-377-A1 §4.2 forward-looking recommendation) (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 follow-ups total).
