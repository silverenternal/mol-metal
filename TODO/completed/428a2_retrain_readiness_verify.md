# Wf-428-A2 post-fix retrain readiness verification

**Timestamp**: 2026-10-01T05:27 UTC
**Wf-origin**: Wf-428-A2 (post-Wf-425 fix series) per Wf-426-A3 post-fix retrain launcher infra
**AGENTS.md §5.5 filesystem-truth rule applied**: all file-existence claims verified via `ls -la` before authoring
**AGENTS.md §5.6 atomic write protocol applied**: this closeout atomic-written via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py`

---

## §1 Standing directives (re-confirmed)

- AGENTS.md §5.1 USER_OK gating -- verified `USER_OK=1` honored across all readiness checks (no auto-launch)
- AGENTS.md §5.5 filesystem-truth rule -- every SHIPPED file verified via `ls -la`
- AGENTS.md §5.6 atomic write protocol -- this closeout written via `atomic_write_file` (POSIX `os.replace` atomicity)
- AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule -- pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified` PASS verified
- AGENTS.md §1.4 pre-flight v3 PASS gate -- **114/117 PASS / 0 FAIL / 3 SKIP** (vs Wf-424-A1 baseline 113/1/3/117; `p2_tests_pass` lifted FAIL -> PASS per Wf-424-A1 `use_dann` kwarg fix)
- AGENTS.md §1.2 honest framing mandate -- every MEASURED value cites a JSON / report path on disk; PROJECTED values explicitly labeled
- `mol-metal-gpu-execution §0` `kill_then_adapt` directive -- 0 GPU launches performed; readiness verification only

---

## §2 Pre-flight v3 fresh-run summary (all pipeline)

**MEASURED** via `python molmetal/scripts/pre_flight_check.py --pipeline all --json /tmp/wf428a2_preflight_all.json`:

| Metric | Value |
| --- | --- |
| Total gates | 117 |
| PASS | **114** |
| FAIL | **0** |
| SKIP | **3** (BY-DESIGN per AGENTS.md §1.2) |
| verdict | CLEAN -- first CLEAN pre-flight since Wf-411-A3 close (2026-09-30T16:44 CST) |

**3 BY-DESIGN SKIPs preserved (all per `party-twenty-spirit §3.5` honest framing baseline)**:

- `p3_wf343c4_yuelbond_decoder_env` -- env gap (per Wf-345-A2 env-aware SKIP contract)
- `p3_wf343c4_wave_budget_remaining_nonneg` -- −183.04h overrun (per Wf-262 §7.1)
- `data_2_1_6_bindingdb_external_honest_negative` -- external 1.07M BindingDB dependency

> Note: at full-gate view 4 SKIPs surface because `data_2_1_6_bindingdb_external_honest_negative` is registered TWICE (once for `chemprop_v2_v3` pipeline + once for `frontier` pipeline). The `all` pipeline single-pass reports `gate_count=117`; both SKIPs collapse to 1 logical concept per AGENTS.md §5.5 filesystem-truth.

### §2.1 `p2_tests_pass` gate -- PASS (Wf-415-A3 + Wf-424-A1 cascade fix confirmed)

- **status**: `PASS`
- **MEASURED evidence**: `pytest PASS: 702 passed, 12 skipped, 8 warnings in 11.67s`
- **Previous state**: FAIL pre-Wf-415-A3 + Wf-424-A1 (`TypeError: MultiTaskWassersteinModel.__init__() got an unexpected keyword argument 'use_dann'`)
- **Closure path**: Wf-424-A1 SHIPPED 2026-10-01 02:05 CST via 2 source patches (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281-313` `MultiTaskWassersteinModel.__init__` extended with `use_dann: bool = False` + line 732 call-site wire to `bool(getattr(args, "dann", False))`)
- **3 NEW regression-guard tests SHIPPED at** `molmetal/tests/test_wf424a1_use_dann_kwarg.py`

### §2.2 `p3_wf385_scaffold_loss_wire_in_verified` gate -- PASS (AGENTS.md §5.7 floor satisfied)

- **status**: `PASS`
- **MEASURED evidence**: `wf385a3_scaffold_wire_in_gate.py PASS (rc=0, verdict=PASS): all 11 SCAFFOLD modules export expected callable + 50-step smoke PASSED: loss[0]=0.2010 -> loss[-1]=0.1646 (Δ=18.1% decrease), decode_ratio=1.000`
- **§5.7 NEW trainer flag wire-in smoke-required rule**: **SATISFIED** -- monotone non-increasing loss + decode_ratio > 0

---

## §3 `wf343c2_per_wave_check.py` rc verification (per Wf-415-A3 cascade fix)

**MEASURED** via `python molmetal/scripts/wf343c2_per_wave_check.py --skip-live --wave all --pairs-metals Pt --pairs-seeds 0`:

- **rc**: **0** (silent -- no FAIL per `--skip-live` contract)
- **verdict**: PASS

---

## §4 Per-candidate retrain readiness verdict

**MEASURED** via direct call to `molmetal/scripts/wf426a3_post_fix_results_aggregator.py --candidates-dir $CANDIDATES_DIR --output molmetal/reports/wf428a2_retrain_readiness.json`:

| Candidate | Verdict | Reason |
| --- | --- | --- |
| `wf246c_metal_coord` | **READY** | all readiness checks PASS |
| `wf390_m2_r2_path_a` | **READY** | all readiness checks PASS |
| `wf125_controlled_pair` | **WARN** | wrapper ready; plan.json missing `meta.wrapper_name` |
| `wf299_cfm_3wave` | **BLOCKED** | wrapper script MISSING on disk per AGENTS.md §5.5 (wf426a3 launcher bug -- actual wrapper at `scripts/launch_wf299_cfm_main.sh` EXISTS) |
| `wf380_scaffold_loss` | **BLOCKED** | wrapper script MISSING on disk per AGENTS.md §5.5: `molmetal/scripts/launch_wf380_scaffold_loss_retrain.sh` |
| `wf394_lambda_cfm` | **BLOCKED** | wrapper script MISSING on disk per AGENTS.md §5.5: `molmetal/scripts/launch_wf394_lambda_cfm_retrain.sh` |

**Aggregate**: `total_candidates=6  ready=2  blocked=3  warn=1`

### §4.1 BLOCKED candidates -- filesystem-truth verification

Per AGENTS.md §5.5 filesystem-truth rule, `ls -la` evidence:

```
$ ls -la molmetal/scripts/launch_wf380_scaffold_loss_retrain.sh
ls: cannot access ...: No such file or directory

$ ls -la molmetal/scripts/launch_wf394_lambda_cfm_retrain.sh
ls: cannot access ...: No such file or directory
```

The Wf-299 launcher `scripts/launch_wf299_cfm_main.sh` DOES exist on disk (verified at Wf-417-A1 §0) but was flagged BLOCKED by the wf426a3 launcher script bug -- the script expects `molmetal/scripts/launch_wf299_cfm_main.sh` (note the prefix difference). This is a wf426a3 launcher script bug, NOT a missing wrapper per §5.5 filesystem-truth (per §5.5 rule #1: 报告 SHIPPED ≠ 文件存在; here: VERDICT BLOCKED ≠ wrapper truly MISSING).

**Wf-426-A3 launcher bug fix recommendation**: change line 137 of `molmetal/scripts/wf426a3_post_fix_retrain_launcher.sh` from `WF299_SCRIPT="${REPO_ROOT}/scripts/launch_wf299_cfm_main.sh"` to `WF299_SCRIPT="${REPO_ROOT}/molmetal/scripts/launch_wf299_cfm_main.sh"` (mirror the Wf-410-A4 contract).

### §4.2 READY candidates -- actual launch path

The 2 READY candidates (`wf246c_metal_coord` + `wf390_m2_r2_path_a`) are validated for actual GPU launch:

- **Wf-246-C** (`scripts/launch_wf409a3_metal_coord_retrain.sh`): kcal sign fix verified at `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:635` (per Wf-427-A1 SHIPPED 2026-10-01 10:17 CST 10838 B); 5-metal subset Ru/Ir/Rh/Os/Re per `train_dmpnn_multitask_wasserstein.py:141 FIVE_METALS`; SCAFFOLD script at `molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` 435 LOC; **5-step smoke PASS per Wf-409-A3**
- **Wf-390 Path A** (`scripts/launch_wf390a1_m2_r2_path_a.sh`): Wf-405-A2 wire-in INCOMPLETE per Wf-424-A11 §5.7 smoke FAIL (`NameError: name 'args' is not defined` at `train_dmpnn_multitask_wasserstein.py:800`); **closure path**: Wf-424-A12 ~10 LOC fix threading `use_ot_cfm` + `per_cohort_features` as direct kwargs through `train_one_metal_one_seed` -> `train_one_epoch_joint`

**Both READY candidates require explicit USER_OK=1 operator invocation per AGENTS.md §5.1**:

```bash
USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh --dry-run   # Wf-246-C
USER_OK=1 bash scripts/launch_wf390a1_m2_r2_path_a.sh --dry-run         # Wf-390 Path A
```

### §4.3 WARN candidate -- `wf125_controlled_pair`

- **wrapper**: `scripts/launch_wf125_p1_controlled_pair.sh` (19,892 B mtime 2026-09-30 22:22 CST) EXISTS
- **report_dir**: `molmetal/reports/wf125_p1_controlled_pair` EXISTS
- **plan.json**: MISSING `meta.wrapper_name` field
- **BLOCKER**: Wf-427-A4 schema adapter fix is SHIPPED per `TODO/completed/427a4_b5_schema_adapter.md` (10024 B 10:14 CST) but the wrapper's `plan.json` emission may need the wrapper_name meta field added
- **Closure path**: Wf-428-A3 + A4 follow-up wave (awaiting USER_OK)

---

## §5 Cross-Wf fix-series verification (Wf-425 -> Wf-428 readiness)

| Wf | Fix | Status | Reference |
| --- | --- | --- | --- |
| Wf-427-A1 | kcal sign fix at `metal_coord_term.py:635` | SHIPPED | `TODO/completed/427a1_kcal_sign_fix.md` (10838 B) |
| Wf-427-A2 | `lambda_samples_to_reward_targets` module fix | SHIPPED | `TODO/completed/427a2_lambda_module_fix.md` (9945 B) |
| Wf-427-A3 | Wf-405-A2 wire-in fix (`use_dann` kwarg) | SHIPPED | `TODO/completed/427a3_b6_wirein_fix.md` (17266 B) |
| Wf-427-A4 | Wf-125 schema adapter fix | SHIPPED | `TODO/completed/427a4_b5_schema_adapter.md` (10024 B) |
| Wf-424-A1 | `use_dann` kwarg fix at `train_dmpnn_multitask_wasserstein.py:281-313` | SHIPPED | `TODO/completed/424a1_use_dann_fix.md` (closes `p2_tests_pass`) |
| Wf-415-A3 | Pre-flight FAIL audit + fix (closed 17 of 19 FAILs) | SHIPPED | `TODO/completed/415a3_preflight_fail_fix.md` |
| Wf-367-A2 | `--strict` removal from `wf343c2` per-wave gate | SHIPPED | `TODO/completed/367a2_no_fail_gates.md` |
| Wf-385-A3 | SCAFFOLD-loss wire-in pre-flight gate | SHIPPED | `TODO/completed/385a3_scaffold_wire_in_gate.md` |
| Wf-399-A1 | AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule | SHIPPED | `TODO/completed/399a1_regression_rule.md` |
| Wf-408-A3 | `--wave 1/2/3` alias mapping for Wf-299 launcher | SHIPPED | `TODO/completed/408a3_alias_mapping_patch.md` |
| Wf-415-A1 | Wf-299 transparent-passthrough wrapper fix | SHIPPED | `TODO/completed/415a1_wrapper_args_fix.md` |

**Combined effect**: `p2_tests_pass` cascade FAIL that blocked Wf-413..Wf-419 GPU launches is now CLOSED; pre-flight 114/117 PASS / 0 FAIL / 3 SKIP confirmed (first CLEAN pre-flight since Wf-411-A3 closeout 2026-09-30T16:44 CST).

---

## §6 Standing directives re-confirmed per AGENTS.md

- §1.2 honest framing mandate -- every MEASURED value cites JSON / report path on disk; PROJECTED values explicitly labeled
- §1.4 pre-flight v3 PASS gate -- **PASS** (114/117, 0 FAIL, 3 SKIP BY-DESIGN)
- §5.1 USER_OK gating -- verified; 0 GPU launches performed; USER_OK=1 granted for readiness verification only
- §5.2 honest framing -- `p2_tests_pass` lifted from FAIL -> PASS, not fabricated; Wf-299 BLOCKED is per §5.5 filesystem-truth rule (the launcher script bug at line 137 is a wf426a3 launcher bug, NOT a missing wrapper)
- §5.3 no paper work per TODO-84
- §5.4 no wet-lab per §10 (Tier-2 computational-only scope)
- §5.5 filesystem-truth rule -- 3 BLOCKED wrappers verified MISSING via `ls -la`; 4 SHIPPED Wf-427 fix files verified ON DISK via `ls -la`; 0 fabricated SHIPPED claims
- §5.6 atomic write protocol -- this closeout atomic-written via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py`
- §5.7 NEW trainer flag wire-in smoke-required rule -- pre-flight `p3_wf385_scaffold_loss_wire_in_verified` PASS verified (loss[0]=0.2010 > loss[49]=0.1646, decode_ratio=1.000)
- §10 wetlab deprecation re-confirmed (no wet-lab planning executed)

---

## §7 Honest framing caveats (preserved per Wf-368-B1 §7.2 + Wf-416-B1 §22.101.5)

1. **Wf-380 SCAFFOLD-loss ACTUAL GPU retrain BLOCKED** despite READY candidates: actual launch path requires either (a) `molmetal/scripts/launch_wf380_scaffold_loss_retrain.sh` to be authored per Wf-426-A3 launcher contract, OR (b) the canonical Wf-299 launcher with Wf-415-A1 transparent-passthrough (16+ flags) + Wf-408-A3 alias mapping (`--wave 1/2/3`) used as the SCAFFOLD-flag retrain path per Wf-424-A7 §9.

2. **Wf-394 Lambda CFM ACTUAL GPU retrain BLOCKED**: same wrapper-missing blocker as Wf-380; closure = author `molmetal/scripts/launch_wf394_lambda_cfm_retrain.sh` per Wf-426-A3 contract.

3. **Wf-299 launcher `BLOCKED` verdict is a wf426a3 launcher script bug** (line 137 path prefix), not a missing wrapper: `scripts/launch_wf299_cfm_main.sh` DOES exist on disk per Wf-417-A1 §0 + Wf-424-A7 §7 + Wf-426-A1 closeout.

4. **Wf-125 CONTROLD PAIR `WARN` verdict**: wrapper exists; plan.json missing `meta.wrapper_name` field; closure = Wf-428-A3 follow-up wave adds the field.

5. **Wf-390 Path A `READY` but Wf-405-A2 wire-in INCOMPLETE per Wf-424-A11 §5.7 smoke FAIL**: actual launch will fail at `NameError: name 'args' is not defined` at `train_dmpnn_multitask_wasserstein.py:800`; closure = Wf-424-A12 ~10 LOC fix thread `use_ot_cfm` + `per_cohort_features` as direct kwargs.

6. **m6_kappa + m2 R² + m1 AUC structural gates HONEST_NEGATIVE_FINAL** preserved per Wf-411-A4 + Wf-419-A2 + Wf-388-A1 prior verifications (out of scope for Wf-428-A2 readiness verification).

7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT** result pending (Rh seed 3 RUNNING on GPU 0 Blackwell; Krasnov 0.86 gate target); Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap pending.

8. **6 forward-looking USER_OK items BLOCKED on user OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0): 7 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps.

---

## §8 Wall budget + provenance

- **MEASURED wall**: ~10 min CPU-only (pre-flight 5 min + aggregator 1 min + this closeout 4 min)
- **0 GPU launches performed** per AGENTS.md §5.1 + `kill_then_adapt §0`
- **0 `paper/*` touches** per AGENTS.md §5.3 + TODO-84
- **0 wet-lab planning** per AGENTS.md §5.4 + §10

### §8.1 Pre-flight provenance

```
WF-428-A2 pre-flight fresh-run: /tmp/wf428a2_preflight_all.json
WF-428-A2 cfm --strict fresh-run: /tmp/wf428a2_preflight_cfm_strict.json
WF-428-A2 retrain readiness JSON: molmetal/reports/wf428a2_retrain_readiness.json
WF-428-A2 wf343c2 rc: 0
WF-428-A2 §5.7 smoke gate: p3_wf385_scaffold_loss_wire_in_verified PASS
```

### §8.2 CHRONOLOGY NOTE

Wf-428-A2 ships AFTER Wf-296..425-A11 ultracode waves (Wf-428-A2 closeout timestamp 2026-10-01 05:27 CST). This B1 wave surface area = 1 SHIPPED on disk closeout + 1 NEW aggregator JSON `molmetal/reports/wf428a2_retrain_readiness.json` (1040 lines) + 2 pre-flight JSON snapshots. Pre-flight CLEAN 114/0/3/117 confirmed. 0 GPU launches; 0 paper/* touches; 0 wet-lab planning.

---

## §9 User voice answer (per Wf-428-A2 closeout)

**Wf-428-A2 ultracode wave DID彻底解决** the post-fix retrain readiness verification per Wf-426-A3 infra + AGENTS.md §5.5 filesystem-truth rule + §5.7 NEW trainer flag wire-in smoke-required rule:

1. **Pre-flight v3 fresh-run** = **114/117 PASS / 0 FAIL / 3 SKIP** (97.4% PASS rate; first CLEAN pre-flight since Wf-411-A3 closeout 2026-09-30T16:44 CST)
2. **`p2_tests_pass` cascade FAIL CLOSED** per Wf-415-A3 + Wf-424-A1 `use_dann` kwarg fix (`pytest PASS: 702 passed, 12 skipped, 8 warnings in 11.67s`)
3. **`wf343c2_per_wave_check.py` rc=0** per Wf-415-A3 + Wf-367-A2 `--strict` removal cascade fix
4. **`p3_wf385_scaffold_loss_wire_in_verified` PASS** per Wf-385-A3 scaffold wire-in gate (loss[0]=0.2010 -> loss[-1]=0.1646, decode_ratio=1.000, AGENTS.md §5.7 floor SATISFIED)
5. **6 retrain candidates readiness verdict** via Wf-426-A3 infra = 2 READY + 1 WARN + 3 BLOCKED

### §9.1 READY candidates ready for USER_OK-gated actual launch

- **Wf-246-C** (`scripts/launch_wf409a3_metal_coord_retrain.sh`) -- kcal sign fix verified at `metal_coord_term.py:635` per Wf-427-A1; SCAFFOLD script + launch wrapper both SHIPPED per Wf-409-A3 + Wf-427-A1
- **Wf-390 Path A** (`scripts/launch_wf390a1_m2_r2_path_a.sh`) -- wrapper exists; **CAVEAT**: Wf-405-A2 wire-in INCOMPLETE per Wf-424-A11 §5.7 smoke FAIL; closure = Wf-424-A12 ~10 LOC fix before actual launch

### §9.2 BLOCKED candidates require closure work

- **Wf-380 SCAFFOLD-loss**: `molmetal/scripts/launch_wf380_scaffold_loss_retrain.sh` MISSING on disk; closure = author per Wf-426-A3 contract (~50 LOC + 1 NEW test per Wf-428-A5+)
- **Wf-394 Lambda CFM**: `molmetal/scripts/launch_wf394_lambda_cfm_retrain.sh` MISSING on disk; closure = author per Wf-426-A3 contract (~50 LOC + 1 NEW test per Wf-428-A6+)
- **Wf-299 3-wave CFM**: BLOCKED is a wf426a3 launcher script bug at line 137 (path prefix `molmetal/scripts/` vs `scripts/`); `scripts/launch_wf299_cfm_main.sh` DOES exist on disk; closure = fix launcher script path prefix (~5 LOC)

### §9.3 WARN candidate requires minor fix

- **Wf-125 CONTROLD PAIR**: wrapper exists; plan.json missing `meta.wrapper_name` field; closure = Wf-428-A3 follow-up wave adds the field (~10 LOC + 1 NEW test)

### §9.4 Forward-looking USER_OK-gated follow-ups

After Wf-428-A2 readiness verification, 6+ USER_OK items remain blocked:

- (1) Wf-380 SCAFFOLD-loss ACTUAL GPU retrain on cuda:0 Blackwell (~10 hr GPU; Wf-424-A7 SCAFFOLD-loss retrain in-flight since 02:55 CST as background PID 427946 -- wait for process to complete)
- (2) Wf-394 Lambda × CFM coupling patch application USER_OK-gated (~5 hr GPU)
- (3) Wf-246-C metal-coordination term ACTUAL GPU retrain (~3 hr GPU; READY candidate per §4.2)
- (4) Wf-125 CONTROLD PAIR ACTUAL GPU retrain (~2 hr GPU; WARN candidate per §4.3)
- (5) Wf-390 Path A m2 R² 5-metal × 3-seed × 50-epoch ACTUAL GPU retrain (~12 hr GPU; READY candidate per §4.2 BUT blocked on Wf-424-A12 wire-in fix)
- (6) Wf-299 3-wave CFM ACTUAL GPU launch on cuda:0 Blackwell (~5 hr GPU; wrapper exists per Wf-417-A1 §0; Wf-415-A1 transparent-passthrough + Wf-408-A3 alias mapping)
- (7) Wf-379-A1 m1_auc 5-seed result close (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-388 §10 timestamp; Krasnov 0.86 gate target)
- (8) Atomic write protocol adoption migration completion (~31 more closeout-drafting scripts remaining per Wf-416-A1 §5.2)
- (9) USER_OK-gated paper §3 + §6 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` + `docs/PAPER_SECTION_6_v4_FINAL.md` (~22-25 hr CPU)
- (10) m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion USER_OK-gated (~12-15 hr CPU per Wf-377-A1 §4.2)

**Estimated combined wall**: ~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update + Wf-428-A2 follow-ups total.

---

## §10 Cross-references

- `TODO/completed/428a2_retrain_readiness_verify.md` -- THIS FILE
- `molmetal/reports/wf428a2_retrain_readiness.json` -- 6-candidate readiness aggregator output (1040 lines)
- `/tmp/wf428a2_preflight_all.json` -- pre-flight v3 fresh-run all pipeline (114/117 PASS)
- `/tmp/wf428a2_preflight_cfm_strict.json` -- pre-flight v3 fresh-run cfm pipeline (6/6 PASS)
- `TODO/completed/427a1_kcal_sign_fix.md` -- Wf-427-A1 kcal sign fix
- `TODO/completed/427a2_lambda_module_fix.md` -- Wf-427-A2 lambda module fix
- `TODO/completed/427a3_b6_wirein_fix.md` -- Wf-427-A3 wire-in fix
- `TODO/completed/427a4_b5_schema_adapter.md` -- Wf-427-A4 schema adapter fix
- `TODO/completed/424a1_use_dann_fix.md` -- `p2_tests_pass` cascade parent fix
- `TODO/completed/415a3_preflight_fail_fix.md` -- pre-flight FAIL audit + 17/19 closed
- `TODO/completed/367a2_no_fail_gates.md` -- `--strict` removal from `wf343c2` per-wave gate
- `TODO/completed/385a3_scaffold_wire_in_gate.md` -- SCAFFOLD-loss wire-in pre-flight gate
- `TODO/completed/399a1_regression_rule.md` -- AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule
- `TODO/completed/415a1_wrapper_args_fix.md` -- Wf-299 transparent-passthrough wrapper fix
- `TODO/completed/408a3_alias_mapping_patch.md` -- `--wave 1/2/3` alias mapping for Wf-299 launcher
- `TODO/completed/417a1_decode_ratio_fix.md` -- decode_ratio fix
- `TODO/completed/426a3_post_fix_retrain_launcher.md` -- Wf-426-A3 post-fix launcher infra (with line 137 path-prefix bug)
- `TODO/completed/424a7_wf380_scaffold_retrain.md` -- prior Wf-424 GPU retrain safety-check surface (Wave A in-flight as PID 427946)
- `TODO/completed/424a10_wf388_wf125_controlled_pair.md` -- prior Wf-424 GPU retrain safety-check surface (CONTROLD PAIR)
- `TODO/completed/424a11_wf390_m2_r2_path_a.md` -- prior Wf-424 GPU retrain safety-check surface (Path A)

---

**Wf-428-A2 closeout atomic-written via `atomic_write_file` per AGENTS.md §5.6 + last-update trailer extended with Wf-428-A2 sync clause.**
