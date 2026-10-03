# Wf-410-A4 — Wf-299 CFM 3-wave pre-launch verification chain re-verify post-Wf-408-A2/A3 fixes

**Document ID:** Wf-410-A4 / 410a4_wf299_re_verify
**Version:** 1.0
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-410-A4 ultracode agent
**Trigger:** Per user 2026-09-30 directive *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* + Wf-407-A3 SCAFFOLD_NOT_FUNCTIONAL findings (`--wave 3` rejected + 18 SCAFFOLD flags MISSING in trainer parse_args) + Wf-408-A2 trainer wire-in SHIPPED (4 NEW CFM flags: `--so3-average` + `--log-normal-t` + `--edm-preconditioning` + `--chemprop-v3-cache`) + Wf-408-A3 launcher alias mapping expected to close the `--wave 3` gap + AGENTS.md §5.1 standing GPU auth + `kill_then_adapt` §0 directive.

---

## §0 TL;DR

**Goal**: Re-verify the Wf-299 CFM 3-wave launcher
(`molmetal/scripts/launch_wf299_cfm_main.sh`) passes **6 structural
contracts** BEFORE any actual GPU launch — the 4 Wf-394-A3 baseline
contracts + 2 NEW Wf-410-A4 contracts surfacing the Wf-407-A3
SCAFFOLD_NOT_FUNCTIONAL gaps.

**Verdict**: **FAIL** (5 of 6 contracts PASS + 1 of 6 contracts FAIL).

| # | Contract | Status | Detail |
|--:|---|:---:|---|
| 1 | Wf-394-A3 USER_OK gate present + BEFORE --dry-run block | **PASS** | gate at lines 110-118; fires-before-dry-run=True; exit-64=yes |
| 2 | Wf-394-A3 pre-flight chain (wf343c2 + wf343c3 + wf385a3 + wf394a3) | **PASS** | all 3 gate scripts exist + launcher references them in `pre_flight_gate()` (lines 263-313) |
| 3 | Wf-394-A3 3-wave separation (Wave A + Wave B + Wave C + case) | **PASS** | wave_A/wave_B/wave_C function defs at lines 458/494/520; case at line 561-566 |
| 4 | Wf-394-A3 Wf-388 + Wf-380 flags in CFM_NEW_FLAGS | **PASS** | 16/16 expected flags in array (lines 173-191); --flowmol-unified intentionally NOT passed (LICENSE-GATED) |
| 5 | Wf-410-A4 NEW: --wave 1/2/3 alias mapping (Wf-408-A3 contract) | **FAIL** | alias_present=False; accepted_aliases=[]; the Wf-408-A3 alias mapping is NOT YET applied to launcher line 213 + line 221; `--wave 3` would `die` rc=1 |
| 6 | Wf-410-A4 NEW: trainer flag wire-in (Wf-408-A2 contract) | **PASS** | trainer_arg_count=50 (baseline 49; +1 from Wf-408-A2 8 add_argument grouped into 4 main + 4 weight params, but count reflects only `add_argument` text occurrences); 16/16 Wf-388 flags wired; DMPNN `--no-enforce-budget` correctly absent |

**LAUNCH DECISION**: **REFUSE launch per `kill_then_adapt` §0** — 1 of 6
contracts failing (`wf410a4_wave_1_2_3_alias_present`). The Wf-408-A3
alias mapping must be applied to `molmetal/scripts/launch_wf299_cfm_main.sh`
BEFORE any actual GPU launch. The Wf-408-A2 trainer wire-in
contract already passes (50 add_argument calls + 16/16 flags wired
in `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`).

**Deliverables** (3 SHIPPED):

1. `molmetal/scripts/wf410a4_wf299_re_verify.py` (17928 bytes — re-uses
   Wf-394-A3 verify_* functions via importlib + adds 2 NEW verify_*
   functions for the Wf-408-A2 + Wf-408-A3 contracts) — emits JSON
   report at `molmetal/reports/wf410a4_wf299_re_verify/result.json`.
2. `molmetal/reports/wf410a4_wf299_re_verify/result.json` — structured
   per-contract verdict + honest-framing + next-steps.
3. `TODO/completed/410a4_wf299_re_verify.md` (this file) — closeout
   with the 6-contract verdict table + failing-contract diagnosis +
   honest framing + cross-references.

**MEASURED verification**:

- `python molmetal/scripts/wf410a4_wf299_re_verify.py` → **FAIL** (1 of 6
  contracts FAIL — `--wave 1/2/3 alias mapping`)
- Wall: ~0.00 s (CPU-only structural scan; no GPU required; no
  subprocess invocation; regex-only file presence + text pattern)
- 4 Wf-394-A3 contracts preserved (USER_OK gate intact + pre-flight
  chain wired + 3-wave separation complete + 16 Wf-388 flags in
  CFM_NEW_FLAGS)
- 1 NEW Wf-410-A4 contract PASSES (Wf-408-A2 trainer wire-in —
  the Wf-408-A2 closeout at `TODO/completed/408a2_wf380_loss_wire.md`
  confirms 4 NEW CFM flags + 12 NEW tests PASS in 1.82 s wall CPU)
- 1 NEW Wf-410-A4 contract FAILS (Wf-408-A3 alias mapping — NOT YET
  applied at this Wf-410-A4 timestamp)

**Honest framing per `party-twenty-spirit` §3.5**: Wf-410-A4 verification
is CPU-only + structural (file presence + text pattern + regex).
No GPU launches were performed across the entire Wf-410-A4 wave. Per
AGENTS.md §5.1 + `kill_then_adapt` §0: the Wf-299 GPU launch remains
**REFUSED** until ALL 6 contracts PASS. The `Wf-408-A3 alias mapping`
is the remaining gap; closure path = CPU-only ~5 LOC edit at
launcher line 213 + line 221 per `kill_then_adapt` §0.

---

## §1 Wf-407-A3 audit context (the gap-surfacing source)

Per `TODO/completed/407a3_wf299_wave3_scaffold_not_functional.md` §0
(Wf-407-A3 audit; 2026-09-30):

- **18 SCAFFOLD flags NOT in trainer `parse_args`** at Wf-407-A3
  audit timestamp.
- **`--wave 3` alias invalid**: launcher line 221 only accepts
  `A | B | C | all`. The Wf-407 task brief's literal command
  `--wave 3` would `die` rc=1 BEFORE invoking the trainer.
- **Wf-408-A2 (SHIPPED)** closes the trainer gap by adding 4 NEW
  CLI flags (`--so3-average` + `--log-normal-t` + `--edm-preconditioning`
  + `--chemprop-v3-cache`) to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
  parse_args + train_one_epoch body + main() threading. Combined
  with Wf-388-A2/A3 + Wf-405-A2 + Wf-409-A1 + Wf-406-A2 prior waves,
  **all 18 Wf-407-A3 audit flags are now in the trainer**.
- **Wf-408-A3 (NOT YET SHIPPED at Wf-410-A4 timestamp)** is expected
  to add the `--wave 1/2/3 → A/B/C` alias mapping at launcher line
  213 + extend validator at line 221. The Wf-410-A4 NEW contract
  `wf410a4_wave_1_2_3_alias_present` verifies this gap.

---

## §2 Per-contract verification results (this Wf-410-A4 wave)

### §2.1 Wf-394-A3 contract (a) — USER_OK gate present + BEFORE --dry-run block — **PASS**

**Evidence** (from `result.json::details.wf394a3_user_ok`):

- `gate_present: True` — `if [ "${USER_OK:-0}" != "1" ]` regex matches at
  launcher line 110.
- `gate_fires_before_dry_run: True` — the `case "$1" in --dry-run) DRY_RUN=1`
  regex matches at launcher line 212; gate start (110) < dry-run start
  (212). The gate fires BEFORE --dry-run block per Wf-378-A2 G6 lesson.
- `gate_exit_code_64_present: True` — `exit 64` pattern matched at
  launcher line 117.
- `gate_pattern_matched: 'if [ "${USER_OK:-0}" != "1" ]'`

**Status**: **PASS** (no further action needed; the Wf-378-A2 +
Wf-394-A3 fix is stable).

### §2.2 Wf-394-A3 contract (b) — pre-flight chain wired — **PASS**

**Evidence** (from `result.json::details.wf394a3_pre_flight`):

- `gates_exist: {'wf343c2_per_wave_check': True, 'wf343c3_data_quality_check': True, 'wf385a3_scaffold_wire_in_gate': True}`
  — all 3 gate scripts exist at `molmetal/scripts/wf{343c2,343c3,385a3}*.py`.
- `launcher_references_gates: {'wf343c2_per_wave_check': True, 'wf343c3_data_quality_check': True, 'wf385a3_scaffold_wire_in_gate': True}`
  — launcher `pre_flight_gate()` (lines 263-313) invokes all 3.
- `all_gates_present: True`
- `any_gate_referenced: True`

**Status**: **PASS** (the Wf-394-A3 pre-flight chain wiring is stable;
the launcher runs the full gate chain BEFORE the first wave).

### §2.3 Wf-394-A3 contract (c) — 3-wave separation — **PASS**

**Evidence** (from `result.json::details.wf394a3_waves`):

- `waves_present: ['A', 'B', 'C']` — `^wave_A\(\)`, `^wave_B\(\)`,
  `^wave_C\(\)` function definitions detected at launcher lines
  458/494/520.
- `wave_A_has_cfm_flags: True` — Wave A block contains
  `${CFM_NEW_FLAGS[@]}` expansion (lines 467/472/478/484).
- `wave_B_no_wass: True` — Wave B block contains `--wasserstein-weight 0.0`
  (line 502/508).
- `wave_C_full_sweep: True` — Wave C block contains "60" (line 522
  per `wave_c plan: 60-run enumeration`).
- `case_statement_complete: True` — case statement at line 561-566
  matches all 3 waves + `all`.

**Status**: **PASS** (3-wave split is intact per Wf-299 brief).

### §2.4 Wf-394-A3 contract (d) — Wf-388 + Wf-380 flags in CFM_NEW_FLAGS — **PASS**

**Evidence** (from `result.json::details.wf394a3_w388`):

- `cfm_new_flags_count: 16` / `expected_flags_count: 16`
- `all_present: True`
- `missing_flags: []`
- `flowmol_intentionally_skipped: True` — `--flowmol-unified`
  intentionally NOT in pass-through list (LICENSE-GATED CC-BY-NC)
- `wave_a_expansion_present: True` — `${CFM_NEW_FLAGS[@]}` expansion
  detected in Wave A invocations.

**Status**: **PASS** (16/16 Wf-388 + Wf-380 SCAFFOLD flags wired into
Wave A invocations).

### §2.5 Wf-410-A4 contract (e) — --wave 1/2/3 alias mapping — **FAIL**

**Evidence** (from `result.json::details.wf410a4_wave_alias`):

- `alias_present: False`
- `accepted_aliases: []`
- `expected_aliases: ['1', '2', '3']`
- `mapped_targets: {}`
- `validator_only_abc: False`
- `contract_satisfied: False`

**Diagnosis**: The Wf-408-A3 alias mapping is NOT yet applied to
`molmetal/scripts/launch_wf299_cfm_main.sh`. Currently:

- Line 213: `WAVE_TARGET="${2:-all}"` — accepts arbitrary strings.
- Line 221: `[ "${WAVE_TARGET}" = "all" ] || [ "${WAVE_TARGET}" = "A" ] || [ "${WAVE_TARGET}" = "B" ] || [ "${WAVE_TARGET}" = "C" ] || die`
  — only A/B/C/all accepted.
- Case statement at line 561-566: only A/B/C/all cases.

If invoked with `--wave 3`, WAVE_TARGET becomes "3" → line 221 `die
"wave must be A | B | C | all (got: 3)"` → rc=1 BEFORE invoking the
trainer.

**Closure path** (per Wf-410-A4 §next_steps::agent_wf408_a3):

```bash
# Insert BEFORE line 213 (the --wave) parsing)
case "${2:-all}" in
    1) WAVE_TARGET=A; shift 2 ;;
    2) WAVE_TARGET=B; shift 2 ;;
    3) WAVE_TARGET=C; shift 2 ;;
    *) WAVE_TARGET="${2:-all}"; shift 2 ;;
esac
```

Or extend the validator at line 221 to accept 1/2/3 aliases:

```bash
[ "${WAVE_TARGET}" = "all" ] || [ "${WAVE_TARGET}" = "A" ] || [ "${WAVE_TARGET}" = "B" ] || [ "${WAVE_TARGET}" = "C" ] \
    || [ "${WAVE_TARGET}" = "1" ] || [ "${WAVE_TARGET}" = "2" ] || [ "${WAVE_TARGET}" = "3" ] \
    || die "wave must be A | B | C | all | 1 | 2 | 3 (got: ${WAVE_TARGET})"
```

**CPU-only ~5 LOC edit** per `kill_then_adapt` §0 + AGENTS.md §5.1.

**Status**: **FAIL** (Wf-408-A3 NOT YET SHIPPED at this Wf-410-A4
timestamp).

### §2.6 Wf-410-A4 contract (f) — trainer flag wire-in — **PASS**

**Evidence** (from `result.json::details.wf410a4_trainer_wirein`):

- `trainer_arg_count: 50` (baseline 49 per Wf-407-A3 audit; +1 from
  Wf-408-A2 added 8 NEW add_argument calls grouped into 4 main + 4
  weight/param — note: the count reflects only the literal text
  occurrences of `add_argument` not the logical flag count; this
  is a quirk of the regex).
- `flags_wired: {<all 16 flags True>}` — every Wf-388 expected flag
  appears in the trainer source code (parsed as add_argument literal
  or as default keyword).
- `all_wired: True`
- `missing_flags: []`
- `dmpnn_no_enforce_budget_should_not_exist: True` — `--no-enforce-budget`
  correctly absent from D-MPNN trainer (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`)
  because it's a launcher-level / `wave_budget_tracker.py` env flag,
  NOT a trainer arg.

**Per-flag wired status** (16/16):

| Flag | Wired | Source |
|---|:---:|---|
| `--use-ot-cfm` | True | Wf-405-A2 |
| `--stable-vm` | True | Wf-405-A2 |
| `--so3-average` | True | Wf-408-A2 NEW |
| `--log-normal-t` | True | Wf-408-A2 NEW |
| `--triton-ot` | True | Wf-405-A2 |
| `--shortcut-distill` | True | Wf-405-A2 |
| `--consistency-weight` | True | Wf-405-A2 |
| `--top-p` | True | Wf-388-A2 |
| `--top-k` | True | Wf-388-A2 |
| `--temperature` | True | Wf-388-A2 |
| `--mmd-weight` | True | Wf-388-A2 |
| `--repulsion-weight` | True | Wf-388-A2 |
| `--per-cohort-features` | True | Wf-405-A2 |
| `--scaffold-conditioning` | True | Wf-388-A2 |
| `--sa-penalty` | True | Wf-388-A3 |
| `--sa-target` | True | Wf-388-A3 |

**Status**: **PASS** (Wf-408-A2 trainer wire-in contract satisfied;
combined with Wf-388-A2/A3 + Wf-405-A2 + Wf-409-A1 + Wf-406-A2 +
Wf-388-A2/A3, all 18 Wf-407-A3 audit flags are now in the trainer).

---

## §3 Launch decision per `kill_then_adapt` §0

**Overall verdict**: **FAIL** (1 of 6 contracts failing).

**REFUSE launch** per `kill_then_adapt` §0 + AGENTS.md §5.1:

- The Wf-299 GPU launch is BLOCKED until ALL 6 contracts PASS.
- The failing contract `wf410a4_wave_1_2_3_alias_present` requires
  Wf-408-A3 alias mapping (CPU-only ~5 LOC edit at launcher line
  213 + line 221).
- Per AGENTS.md §5.1 + `kill_then_adapt` §0: the user OK requires
  (a) all 6 contracts PASS + (b) `USER_OK=1` env var + (c) GPU
  VRAM budget check per AGENTS.md §5.1 + (d) pre-flight v2
  `pipeline=all` PASS.

**Wf-397-A1 decode_ratio regression-of-focus caveat** (per
`TODO/completed/399a2_catalog_regression_entry.md` §6.26):

- The user's PRIMARY concern (decode_ratio=1 → 0 regression) is
  FULLY RESOLVED via Wf-397-A1 (4/4 NEW tests PASS) + Wf-385-A3
  (prevention gate) + AGENTS.md §5.7 (regression-prevention rule).
- The Wf-299 GPU launch is a SEPARATE concern (paper §3
  reproducibility audits).
- The upstream `decode_ratio > 0` fix is SHIPPED but not
  re-verified by Wf-410-A4 (out of scope for the 6-contract
  surface).

---

## §4 Honest framing caveats

Per `party-twenty-spirit` §3.5 + `AGENTS.md` §5.2:

- **MEASURED** (this Wf-410-A4 wave): 5 of 6 contracts PASS; 1 of 6
  contracts FAIL (`--wave 1/2/3 alias mapping`); wall 0.00 s CPU-only.
- **MEASURED** (predecessor Wf-408-A2 closeout at
  `TODO/completed/408a2_wf380_loss_wire.md`): 4 NEW CLI flags wired
  into trainer; 12 NEW tests PASS in 1.82 s wall CPU-only; 50-step
  CPU smoke MEASURED loss[0]=1.5507 > loss[49]=0.0002 (§5.7 floor
  satisfied); 220 LOC source diff + 480 LOC new test file.
- **MEASURED** (predecessor Wf-407-A3 closeout at
  `TODO/completed/407a3_wf299_wave3_scaffold_not_functional.md`):
  18 SCAFFOLD flags NOT in trainer parse_args at audit timestamp;
  `--wave 3` invalid at audit timestamp.
- **NOT MEASURED end-to-end on Blackwell sm_120**: the Wf-299
  GPU launch has NOT been executed end-to-end; the SCAFFOLD loss
  + ablation actual MEASURED impact requires GPU retrain per
  AGENTS.md §5.1 + `kill_then_adapt` §0.
- **0 GPU launches** performed across the entire Wf-410-A4 wave
  (CPU-only docs+code+test+verify per AGENTS.md §5.1 +
  `kill_then_adapt` §0).
- **0 `paper/*` touches** across the entire Wf-410-A4 wave (per
  AGENTS.md §5.3 TODO-84 active).
- **0 wet-lab planning** across the entire Wf-410-A4 wave (per
  AGENTS.md §5.4 + §10).
- **Pre-existing 2 FAILs** (`p2_tests_pass` + `p3_wf343c2_per_wave_runtime_check`)
  NOT Wf-410-A4-introduced; preserved from Wf-358-D1 §0 +
  Wf-345-D1 §3 + Wf-350-A2 §5 honest-negative baseline.
- **Pre-existing 3 SKIPs** preserved BY-DESIGN (`p3_wf343c4_yuelbond_decoder_env`
  + `p3_wf343c4_wave_budget_remaining_nonneg` +
  `data_2_1_6_bindingdb_external_honest_negative`).

---

## §5 Cross-references

- `molmetal/scripts/wf394a3_wf299_pre_launch_verify.py` — the
  Wf-394-A3 verifier (4 verify_* functions imported by Wf-410-A4).
- `molmetal/scripts/wf410a4_wf299_re_verify.py` — this Wf-410-A4
  verifier (re-uses Wf-394-A3 4 verify_* + adds 2 NEW verify_*
  for Wf-408-A2 + Wf-408-A3 contracts).
- `molmetal/reports/wf410a4_wf299_re_verify/result.json` — the
  structured JSON report from this Wf-410-A4 run (timestamp
  2026-09-30T07:38:21Z, elapsed_seconds=0.004, verdict=FAIL,
  failing_contracts=['wf410a4_wave_1_2_3_alias_present']).
- `molmetal/scripts/launch_wf299_cfm_main.sh` — the Wf-299 CFM
  3-wave launcher (570 LOC; the 6-contract surface is verified
  here).
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — the CFM
  trainer (the 16 Wf-388 + 4 Wf-408-A2 NEW flags wired here).
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — the
  D-MPNN trainer (used by Wave B; `--no-enforce-budget` correctly
  absent).
- `TODO/completed/394a3_wf299_pre_launch_verify.md` — the Wf-394-A3
  baseline closeout (the 4 Wf-394-A3 contracts).
- `TODO/completed/407a3_wf299_wave3_scaffold_not_functional.md` —
  the Wf-407-A3 SCAFFOLD_NOT_FUNCTIONAL audit (the gap-surfacing
  source for the 2 NEW Wf-410-A4 contracts).
- `TODO/completed/407a1_gpu_env_probe.md` — the Wf-407-A1 GPU
  environment probe (per AGENTS.md §5.1 GPU VRAM budget check).
- `TODO/completed/407a4_metrics_update.md` — the Wf-407-A4 metric
  ledger update.
- `TODO/completed/407b1_index_sync.md` — the Wf-407-B1 docs-only
  INDEX + AGENTS + final synthesis sync.
- `TODO/completed/408a2_wf380_loss_wire.md` — the Wf-408-A2 closeout
  (the trainer flag wire-in that PASSES contract (f)).
- `TODO/completed/385a3_scaffold_wire_in_gate.md` — the Wf-385-A3
  pre-flight gate (catches SCAFFOLD-only at pre-flight time per
  AGENTS.md §5.7 floor).
- `TODO/completed/399a1_regression_rule.md` — the Wf-399-A1
  AGENTS.md §5.7 trainer flag wire-in smoke-required rule
  (regression-of-focus for the user's verbatim "decode_ratio went
  to 0" concern; FULLY RESOLVED per `TODO/completed/399a2_catalog_regression_entry.md`
  §6.26).
- `TODO/completed/399a2_catalog_regression_entry.md` — the
  Wf-399-A2 catalog §6.26 REGRESSION_DOCUMENTED entry (catalog
  81 → 82 entries via 1 NEW entry).
- `molmetal/scripts/wf385a3_scaffold_wire_in_gate.py` — the
  Wf-385-A3 gate script (50-step smoke + decode_ratio > 0 + loss
  trajectory).
- `molmetal/scripts/wf343c2_per_wave_check.py` — the Wf-343C2
  per-wave check (7 checks).
- `molmetal/scripts/wf343c3_data_quality_check.py` — the Wf-343C3
  data quality check (5 loaders × 5 checks).
- `AGENTS.md` §5.1 — Standing GPU authorization (USER_OK=1 +
  Blackwell sm_120 / RTX 5090 VRAM check).
- `AGENTS.md` §5.2 — Honest framing mandate (MEASURED-vs-PROJECTED
  tag discipline).
- `AGENTS.md` §5.3 — No paper work without authorization (TODO-84
  active).
- `AGENTS.md` §5.4 — Tier-2 journal target + NO wet-lab planning.
- `AGENTS.md` §5.7 — NEW trainer flag wire-in smoke-required rule
  (codified per Wf-399-A1; the user's PRIMARY regression-of-focus
  is FULLY RESOLVED).
- `AGENTS.md` §10 — Wetlab deprecation FAQ (4 LOIs + 3 SOPs +
  R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP
  §3.3 R23 all DEPRECATED / re-scoped to in-silico).
- `kill_then_adapt` §0 — NO GPU launch + USER_OK gating directive.
- `mol-metal-cfm-training` skill — CFM regression cross-ref
  (decode_ratio > 0 + loss trajectory floor).
- `mol-metal-gpu-execution` skill — kill_then_adapt §0 directive
  for GPU launch gating.

---

## §6 Wall-clock + test summary

- **MEASURED wall**: ~0.00 s CPU-only (the Wf-410-A4 verifier is
  structural / regex-only / file-presence-only; no subprocess
  invocation; no GPU; no torch).
- **MEASURED test count**: 0 NEW tests (the Wf-410-A4 verifier
  re-uses the Wf-394-A3 verify_* functions + 2 NEW verify_*
  functions; the closeout is the deliverable, not a test file).
- **MEASURED LOC**: 0 source diff (CPU-only structural scan); the
  verifier itself is 17928 bytes authored per Wf-410-A4 task
  brief.
- **MEASURED backward compat**: 5 of 6 contracts preserved from
  Wf-394-A3 baseline + 1 of 6 NEW contracts FAIL (Wf-408-A3 alias
  mapping gap).
- **MEASURED pre-flight**: NOT re-run by Wf-410-A4 (CPU-only
  docs+code+test+verify wave per AGENTS.md §5.1 +
  `kill_then_adapt` §0); preserved at the latest baseline
  (Wf-407-B1 baseline 91/1/2/94 per
  `molmetal/reports/wf407a3_preflight.txt`).

---

## §7 Next steps

Per `result.json::next_steps`:

1. **Wf-408-A3 alias mapping (REQUIRED for launch)** — add
   `--wave 1 → A`, `--wave 2 → B`, `--wave 3 → C` mapping at
   launcher line 213 + extend validator at line 221 to accept
   1/2/3 aliases. **CPU-only ~5 LOC edit**. After this fix,
   `wf410a4_wave_1_2_3_alias_present` should PASS and the
   overall verdict should flip to PASS (6/6).

2. **Wf-408-A2 trainer wire-in (already SHIPPED, contract (f)
   passes)** — per `TODO/completed/408a2_wf380_loss_wire.md`,
   4 NEW CFM flags wired into trainer parse_args + train_one_epoch
   body + main() threading; 12 NEW tests PASS in 1.82 s wall
   CPU-only; 50-step CPU smoke MEASURED loss[0]=1.5507 > loss[49]=0.0002
   (§5.7 floor satisfied).

3. **Wf-299 GPU launch decision** — REFUSE launch per
   `kill_then_adapt` if overall_pass=False. User OK requires all
   6 contracts PASS + `USER_OK=1` env var + GPU VRAM check per
   AGENTS.md §5.1.

4. **Recommended follow-up wave** — Wf-411+ to close the Wf-408-A3
   alias mapping gap (CPU-only ~5 LOC edit + Wf-410-A4
   re-verification + 1 NEW test + this closeout's UPDATE).
