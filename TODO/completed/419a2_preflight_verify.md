# Wf-419-A2 Pre-flight Fresh-run Verify (post-Wf-419-A1 fix)

**Last updated:** 2026-09-30T15:18:11+0000 UTC
**Wall:** ~185.23 s (CPU-only on RTX PRO 6000 Blackwell + RTX 5090 host ZhaoLab)
**Author:** Wf-419-A2 (per AGENTS.md §5.5 filesystem-truth rule)
**Refs:** Wf-419-A1 fix attempt (cascade child NOT closed) + Wf-415-A1 wrapper fix (launcher dry-run PASS) + AGENTS.md §5.1/§5.2/§5.3/§5.5 + Wf-345-B3 dynamic gate count + Wf-364-A1 cascade self-reference fix + Wf-367-A2 wf343c2 --strict removal.

## §0 TL;DR

Pre-flight v3 fresh-run on `pipeline='all'` --strict: **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL**. Gate count consistency `actual=117 == expected=117 (auto-recomputed via Wf-345-B3)` PASS. Wf-299 launcher dry-run with Wf-415-A1 wrapper fix: **exit code 0**; 4 ablation configs logged; NO pre-flight FAIL block.

**HONEST-NEGATIVE finding**: Task brief claim "`p2_tests_pass` cascade child FAIL should be CLOSED" is **INCORRECT** per AGENTS.md §5.5 filesystem-truth rule. The 2 FAILs are:

1. `p2_tests_pass` (P2, tests) — pytest FAIL (rc=1): 1 failed, 701 passed, 12 skipped, 8 warnings in 10.47 s
2. `p3_wf343c2_per_wave_runtime_check` (P3, wf347a1) — wf343c2_per_wave_check.py FAIL (rc=1): upstream `preflight_no_fail_gates` detects `p2_tests_pass` FAIL

**Root cause of p2_tests_pass FAIL**: `molmetal/tests/test_kendall_dann_wiring.py::test_dann_on_off` raises `TypeError: MultiTaskWassersteinModel.__init__() got an unexpected keyword argument 'use_dann'`. This is a PRE-EXISTING upstream bug (NOT Wf-419-A1-introduced, NOT Wf-419-A1-closed); the test expects `use_dann` kwarg in the model constructor but the model does not expose it.

**Wf-299 launcher dry-run**: Wf-415-A1 `EXTRA_FLAGS` accumulator working as designed — unknown flags transparently passthrough to the trainer. The Wf-343C2 advisory (WARN, not FAIL) is informational.

## §1 Fresh-run Results Table

| Gate | Status | Evidence (truncated) |
|---|---|---|
| `p2_tests_pass` | **FAIL** | `pytest FAIL (rc=1): 1 failed, 701 passed, 12 skipped, 8 warnings in 10.47s` |
| `p3_wf343c2_per_wave_runtime_check` | **FAIL** | `wf343c2_per_wave_check.py FAIL (rc=1, wave='Wf-347A1 pre-retrain smoke', --skip-live, no --strict per Wf-367-A2)` |
| `p3_wf343c4_wave_budget_remaining_nonneg` | SKIP (BY-DESIGN) | `remaining_hr=-183.04h is negative AND 8 DEFERRED wave(s) (Wave-cfm:cfm_real_decode_lift_p1.3, Wave-cfm:cfm_per_innovation_ablation_p1.4, Wave-dmpnn:wf124_p3_per_innovation_ablation_g6, ...)` |
| `data_2_1_6_bindingdb_external_honest_negative` | SKIP (BY-DESIGN) | `BindingDB 1.07M is external (~50 precious-metal entities); NOT used in retrain per parent doc §2.1.6. Honest-negative: data fact, not blocker.` (appears 2× — once in `all` pipeline, once in `chemprop_v2_v3` sub-pipeline) |
| `p3_gate_count_consistency_audit` | PASS | `gate count consistent for pipeline='all': actual=117 == expected=117 (auto-recomputed via Wf-345-B3)` |
| `p3_loi_send_ready` | PASS | `4/4 LOI drafts ready-to-send + review helper present (status DRAFT/SENT headers + ≥16 checklist boxes)` |
| `p3_wf343c4_anchor_freshness_24h` | PASS | `all 11/11 paper-grade anchors fresh (<= 24h; Wf-302A / Wf-304A AnchorGuard pre-retrain freshness contract)` |
| `p3_wf385_scaffold_loss_wire_in_verified` | PASS | `wf385a3_scaffold_wire_in_gate.py PASS (rc=0, verdict=PASS): all 11 SCAFFOLD modules export expected callable + 50-step smoke PASSED: loss[0]=0.2008 → loss[-1]=0.1645 (Δ=18.1% decrease), decode_ratio=1.000` |
| `p3_wf343c4_yuelbond_decoder_env` | PASS | `cfm_decoder_yuelbond.py (23792B) exports YuelBondDecoder (per Wf-272I / Wf-269C YuelBond-conditionality structural check; --no-yuelbond → decode_ratio 0.000 MEASURED on real cohort)` |
| `p1_cfm_smoke_decoded` | PASS | `/home/hugo/codes/mol-metal/molmetal/reports/wf188_m14_full_metalcytotox/result.json: value=0.995821 >= 0.5` |
| (Other 104 PASS gates) | PASS | (see `molmetal/reports/wf419a2_preflight_freshrun.txt` for full evidence) |

**Gate counts**: 112 PASS + 2 FAIL + 3 SKIP = 117 TOTAL ✓ (gate_count_consistency_audit PASS).

**SKIP summary** (2 unique gates, 3 records due to sub-pipeline duplication):
1. `p3_wf343c4_wave_budget_remaining_nonneg` — BY-DESIGN (Wf-262 §7.1: -183.04h overrun; 8 DEFERRED wave(s) need USER_OK or schedule outside 10h session; non-deferred cost 16.00h; effective_remaining=-6.00h).
2. `data_2_1_6_bindingdb_external_honest_negative` — BY-DESIGN (external 1.07M entity BindingDB; HONEST_NEGATIVE: data fact, not blocker).

**FAIL root-cause analysis**:
- `p2_tests_pass`: `molmetal/tests/test_kendall_dann_wiring.py::test_dann_on_off` raises `TypeError: MultiTaskWassersteinModel.__init__() got an unexpected keyword argument 'use_dann'`. The test instantiates `MultiTaskWassersteinModel(n_metals=2, hidden=32, depth=2, dropout=0.1, use_dann=False)` but the model's `__init__()` does not declare a `use_dann` parameter. This is a PRE-EXISTING upstream bug; Wf-419-A1 did NOT fix it (the Wf-419-A1 fix attempted a different surface — likely a rewire of `use_dann` to `no_dann` flag, but the test still uses the OLD positional signature). Closure path: either (a) add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()`, or (b) update test to use `no_dann=True` instead of `use_dann=False`. ~5 LOC CPU-only fix; closure = Wf-420+ follow-up.
- `p3_wf343c2_per_wave_runtime_check`: This gate is in Wf-364-A1 cascade-self-reference cycle — it checks `preflight_no_fail_gates` which excludes self but finds `p2_tests_pass` FAIL. Once `p2_tests_pass` is fixed, this gate will auto-pass.

## §2 Wf-299 Launcher Dry-run Verification (Wf-415-A1 wrapper fix)

**Command**: `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A --dry-run`

**Exit code**: 0 ✓

**Key evidence**:
- `[OK] Wave A all 4 ablation configs complete` — 4 invocation patterns (full_stack + yuelbond_only + w2_only + catherine_only) all logged successfully.
- `[OK] Wf-299C launch script done — wave=A` — exit 0.
- **NO pre-flight FAIL block** — only WARN advisories: `[WARN] DRY-RUN: skipping pre_flight_check.py GPU/live gates` + `[WARN] Wf-343C2 wave-A returned non-zero (chain_status=0); advisory` (NOT FAIL — just informational; the Wf-343C2 chain itself reports exit 1 due to upstream p2_tests_pass FAIL but the launcher treats it as advisory in DRY_RUN mode).
- **Wf-415-A1 `EXTRA_FLAGS` accumulator works as designed**: the dry-run emitted all 16 NEW trainer flags (`--use-ot-cfm --stable-vm --so3-average --log-normal-t --triton-ot --shortcut-distill --consistency-weight 1.0 --top-p 0.9 --top-k 50 --temperature 1.0 --mmd-weight 0.1 --repulsion-weight 0.05 --per-cohort-features --scaffold-conditioning --sa-penalty 0.0 --sa-target 2.8 --use-yuelbond`) without any wrapper-side rejection.
- **Wave A plan**: 4 ablation configs × 1 seed × 3 epochs (~2 hr, cuda:0).
- **Reports**: `/home/hugo/codes/mol-metal/molmetal/reports/wf299_cfm_main/wave_a` per Wf-363-A1 DRY_RUN.log schema.
- **Honest framing**: `PROJECTED wall-times; MEASURED after each wave runs.` + `Per TODO-119 P1.4 §directives: per_wave_user_ok_required = true.`

**Implication for Wf-299 actual launch**: GPU launch remains BLOCKED on USER_OK=1 per AGENTS.md §5.1 + `kill_then_adapt §0` directive. The Wf-415-A1 wrapper fix is in place and the dry-run is functional. The pre-flight cascade failure is a SEPARATE concern from the Wf-299 launcher; the launcher works correctly in DRY_RUN mode regardless of upstream pre-flight status.

## §3 Cross-references + Wave-lineage

- **AGENTS.md** §1.4 (pre-flight PASS requirement) + §5.1 (USER_OK gating) + §5.2 (honest framing) + §5.3 (no paper/* touches) + §5.5 (filesystem-truth rule — applied here: the 2 FAILs are recorded honestly, NOT paraphrased) + §5.6 (atomic write protocol — applied via `atomic_write_file` for `wf419a2_preflight_freshrun.txt` + `419a2_preflight_verify.md`)
- **Wf-345-B3** dynamic gate count auto-compute — preserved consistency: `actual=117 == expected=117`.
- **Wf-364-A1** cascade self-reference cycle fix — `p3_wf343c2_per_wave_runtime_check` excludes self when counting upstream FAILs.
- **Wf-367-A2** `--strict` removal from gate invocation (preserves `--skip-live` semantics; 3 SKIPs not FAILs).
- **Wf-415-A1** `EXTRA_FLAGS` accumulator wrapper fix — verified working at Wf-419-A2 timestamp.
- **Wf-345-A1 + Wf-345-A2 + Wf-345-A3 + Wf-345-B1 + Wf-345-B3 + Wf-345-C1..C4** — anchor freshness, hyperparams, yuelbond decoder env-aware SKIP, dynamic count — all preserved at PASS.
- **Wf-385-A3** SCAFFOLD-loss wire-in gate — PASS (`all 11 SCAFFOLD modules export expected callable + 50-step smoke PASSED: loss[0]=0.2008 → loss[-1]=0.1645 (Δ=18.1% decrease), decode_ratio=1.000`).
- **Wf-405-A3** pre-flight FAIL fixes — closed regex + anchor STALE — preserved (anchor freshness now PASS at 11/11 fresh).
- **Wf-411-B1 baseline** (95/18/4/117) — drift +23 NEW gates from Wf-410 + Wf-411 wave activity; current state 112/2/3/117 = +17 PASS -16 FAIL (i.e. 16 FAILs closed by Wf-405-A3 fixes, 2 FAILs remain: p2_tests_pass + cascade child).

## §4 Honest Framing per AGENTS.md §5.2 + party-twenty-spirit §3.5

- **Pre-flight cascade child FAIL: NOT closed by Wf-419-A1 fix.** The 2 FAILs are PRE-EXISTING upstream test bugs (specifically `test_kendall_dann_wiring.py::test_dann_on_off` model constructor mismatch) that Wf-419-A1's fix did not address. Closure requires Wf-420-A1 (5 LOC CPU-only fix: add `use_dann` kwarg to `MultiTaskWassersteinModel.__init__()` OR update test signature).
- **Wf-299 launcher wrapper: WORKING correctly** — exit code 0; 4 ablation configs logged; Wf-415-A1 EXTRA_FLAGS accumulator verified.
- **AGENTS.md §5.5 filesystem-truth rule: APPLIED** — this closeout records the on-disk pre-flight JSON state at Wf-419-A2 timestamp (112/2/3/117) NOT the harness-claimed target state (113/0/3/117).
- **No `paper/*` touches** per AGENTS.md §5.3 + TODO-84.
- **No wet-lab planning** per AGENTS.md §5.4 + §10.

## §5 Next-step recommendations

1. **Wf-420-A1** (CPU-only, ~5 LOC): Add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` in `molmetal/molmetal_lam/sbdd_env/training_utils.py` (or update test_kendall_dann_wiring.py signature). This will close `p2_tests_pass` → which auto-closes `p3_wf343c2_per_wave_runtime_check` cascade → pre-flight becomes 113/0/3/117 (PRESERVING the 3 BY-DESIGN SKIPs per §1 above).
2. **Wf-420-A2**: After Wf-420-A1 fix, re-run pre-flight fresh-run → expect 113/0/3/117.
3. **Wf-420-A3** (USER_OK-gated): Wf-299 actual GPU launch on cuda:0 (~12 hr wall per Wf-415-A1 + Wf-407-A3 closure path).
4. **Wf-420+ continued**: Wf-394-A1 Lambda CFM wire-in patch application + Wf-246-C metal-coordination term retrain + atomic write protocol adoption migration completion (~41 remaining scripts) + Wf-379-A1 m1_auc 5-seed result close + paper §6 v4 write + paper §3 v4 update.

## §6 Artifact locations

- Pre-flight fresh-run JSON: `molmetal/reports/wf419a2_preflight_freshrun.txt` (38,404 bytes, 1,055 lines, mtime 2026-09-30 23:21)
- Wf-299 launcher dry-run output: `/tmp/wf419a2_launcher_dryrun.txt` (full transcript; not committed to repo per workflow artifact minimization per Wf-385-A3)
- This closeout doc: `TODO/completed/419a2_preflight_verify.md` (this file; ~13 KB; written via `write_closeout` per AGENTS.md §5.6)
- Failed test (root cause): `molmetal/tests/test_kendall_dann_wiring.py::test_dann_on_off` line 137
- Model with missing kwarg: `molmetal/molmetal_lam/sbdd_env/training_utils.py::MultiTaskWassersteinModel.__init__()` (or wherever the model class is defined)
- Cascade child gate (depends on p2_tests_pass): `molmetal/scripts/pre_flight_check.py::gate_wf343c2_per_wave_runtime_check` (line ~4590 per Wf-367-A2)
- Wf-415-A1 wrapper fix: `molmetal/scripts/launch_wf299_cfm_main.sh` line 210-258 (`EXTRA_FLAGS` accumulator)
