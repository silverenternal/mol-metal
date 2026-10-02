# Wf-395-A3 — Full Integration Test for Wf-388 + Wf-392 CLI Flags + Wf-385-A3 Pre-flight Gate

**Last updated:** 2026-09-30 (Wf-395-A3 ship)
**Workflow:** Wf-395 ultracode CPU-only docs+code+test wave (0 GPU launches; 0 `paper/*` touches per TODO-84 + §5.3; 0 wet-lab planning per AGENTS.md §5.4 + §10)
**Status:** SHIPPED — 22 NEW tests PASS in 1.60 s wall; `train_cfm_yuelbond_wasserstein.py` AST-parses clean; Wf-385-A3 gate verified in `build_gates('all')` at gate index 72/117.

User directive 2026-09-30 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — start additional ultracode for TODO tasks not yet covered by the workflows just launched (Wf-378..393 ultracode family). Wf-395-A3 ships the full-integration verification layer that closes the Wf-388 + Wf-392 CLI flag wire-in gap + verifies the Wf-385-A3 pre-flight gate is properly registered in the pipeline.

---

## §0 TL;DR

| Item | Value |
|---|---|
| **Workflow ID** | Wf-395-A3 |
| **Test file** | `molmetal/tests/test_wf395a3_full_integration.py` (NEW ~520 LOC) |
| **Verify report** | `molmetal/reports/wf395a3_full_integration/preflight_verify.json` (NEW ~600 bytes) |
| **22 NEW tests PASS** | Across 4 test classes (TestAll388392FlagsCompose=6 + TestWf385GateIntegration=5 + TestPreFlightPipelineAll=6 + TestTrainer14Plus8Flags=5) |
| **Wall MEASURED** | 1.60 s on CadButEaas venv (torch 2.14.0+cu130) |
| **AST parse check** | `train_cfm_yuelbond_wasserstein.py` AST-parses clean |
| **Pre-flight cascade** | `build_gates('all')` returns 117 gates (95 ≥ minimum per Wf-385-A3 §5) |
| **Wf-385 gate in pipeline** | YES (gate index 72/117; name `p3_wf385_scaffold_loss_wire_in_verified`) |
| **GPU launches** | 0 |
| **`paper/*` touches** | 0 |
| **Wet-lab planning** | 0 |

Standing directives re-confirmed:
- AGENTS.md §1 (Five standing project rules)
- AGENTS.md §5.1 (Standing GPU authorization — 0 GPU launches per kill_then_adapt §0)
- AGENTS.md §5.2 (Honest framing — `party-twenty-spirit` §3.5)
- AGENTS.md §5.3 (No paper work without authorization — 0 `paper/*` touches per TODO-84)
- AGENTS.md §5.4 + §10 (Tier-2 re-orientation + NO wet-lab planning)

---

## §1 Integration test architecture

Per `party-twenty-spirit` §3.5 MEASURED-vs-PROJECTED tag discipline, the test surface is split into 4 canonical test classes:

### §1.1 TestAll388392FlagsCompose (6 tests)

Verifies the 6 diversity + 2 SA Wf-388 CLI flags are wired into `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:parse_args` via AST-based introspection (no torch import needed).

| Test | What it verifies |
|---|---|
| `test_trainer_file_exists` | CFM trainer on disk at expected path |
| `test_parse_args_function_exists` | `parse_args` function declares ≥ 8 flags |
| `test_all_388_diversity_flags_present` | All 6 Wf-388-A2 diversity flags wired |
| `test_all_388_sa_flags_present` | All 2 Wf-388-A3 SA flags wired |
| `test_trainer_ast_parses_clean` | Trainer file AST-parses without SyntaxError |
| `test_flags_compose_in_argparse_namespace` | 8 flags compose in sample argparse namespace |

### §1.2 TestWf385GateIntegration (5 tests)

Verifies the Wf-385-A3 gate module at `molmetal/scripts/wf385a3_scaffold_wire_in_gate.py` is importable + the CLI `--skip-smoke` mode returns a verdict without requiring the torch runtime.

| Test | What it verifies |
|---|---|
| `test_gate_module_is_importable` | Gate module imports without error |
| `test_gate_module_has_canonical_scaffold_table` | 11 SCAFFOLD entries per Wf-380-B1 |
| `test_gate_main_skip_smoke_returns_verdict` | `--skip-smoke` mode → rc=0, verdict=PASS |
| `test_gate_main_returns_failure_on_broken_module` | Broken module export → rc=1, verdict=FAIL |
| `test_gate_main_verdict_is_in_known_set` | Verdict ∈ {PASS, FAIL, SKIP} (contract) |

### §1.3 TestPreFlightPipelineAll (6 tests)

Verifies the Wf-385-A3 gate is registered in `molmetal/scripts/pre_flight_check.py:build_gates('all')` with the correct name + priority label + pipeline contract.

| Test | What it verifies |
|---|---|
| `test_pre_flight_check_module_importable` | pre_flight_check.py imports cleanly |
| `test_build_gates_all_returns_list` | `build_gates('all')` returns non-empty list |
| `test_wf385_gate_is_in_pipeline` | Gate name `p3_wf385_scaffold_loss_wire_in_verified` registered |
| `test_wf385_gate_function_is_exportable` | Top-level `gate_wf385_scaffold_loss_wire_in_verified` exported |
| `test_wf385_gate_count_meets_minimum` | ≥ 95 gates (94 baseline + 1 NEW Wf-385-A3) |
| `test_wf385_gate_priority_label_is_p3` | Priority label P3 per AGENTS.md §4 |

### §1.4 TestTrainer14Plus8Flags (5 tests)

Verifies the CFM trainer AST-parses + the 8 NEW Wf-388 flags are parseable.

| Test | What it verifies |
|---|---|
| `test_trainer_parses_clean` | Trainer AST-parses + parse_args is a top-level function |
| `test_wf388_8_flags_all_parseable` | All 8 Wf-388 flags in parse_args via AST extraction |
| `test_baseline_flags_still_present` | Baseline 18 trainer flags preserved (additive wire-in) |
| `test_wf380_flags_not_yet_wired_honest_negative` | **HONEST-NEGATIVE**: Wf-380 SCAFFOLD flags NOT yet wired (deferred) |
| `test_wf392_flags_live_in_other_trainers` | Wf-392 flags live in `train_se3sb.py` + `train_chemprop_v2_v3.py` (cross-ref) |

---

## §2 MEASURED pre-flight verification

Per `molmetal/reports/wf395a3_full_integration/preflight_verify.json`:

```json
{
  "wave": "Wf-395-A3",
  "verify": {
    "ast_parse_trainer": "OK",
    "trainer_path": "molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
    "pre_flight_pipeline_all": {
      "total_gates": 117,
      "wf385_gate_in_pipeline": true,
      "wf385_gate_index": 72,
      "wf385_gate_name": "p3_wf385_scaffold_loss_wire_in_verified",
      "minimum_required": 95,
      "minimum_satisfied": true
    }
  },
  "tests": {
    "file": "molmetal/tests/test_wf395a3_full_integration.py",
    "passed": 22,
    "failed": 0,
    "skipped": 0,
    "wall_s": 1.60,
    "host_python": "/home/hugo/codes/CadButEaas/.venv-vlm/bin/python (torch 2.14.0+cu130)"
  }
}
```

---

## §3 Honest framing per `party-twenty-spirit` §3.5

### §3.1 MEASURED claims (verifiable on disk)

- `molmetal/tests/test_wf395a3_full_integration.py` is on disk (NEW ~520 LOC)
- `molmetal/reports/wf395a3_full_integration/preflight_verify.json` is on disk (NEW ~600 bytes)
- `train_cfm_yuelbond_wasserstein.py` AST-parses clean (`TRAINER_AST_PARSE: OK`)
- 22 NEW tests PASS in 1.60 s on CadButEaas venv (torch 2.14.0+cu130)
- `build_gates('all')` returns 117 gates (MEASURED via `len(gates)`); 117 ≥ 95 minimum required per Wf-385-A3 §5
- Wf-385-A3 gate name `p3_wf385_scaffold_loss_wire_in_verified` is at gate index 72/117 (verified via `names.index(...)`)

### §3.2 PROJECTED / honest-negative caveats

1. **The 14 Wf-380 SCAFFOLD flags are NOT yet wired into `train_cfm_yuelbond_wasserstein.py:parse_args`** — the Wf-380 wire-in is a deferred surface for a USER_OK-gated follow-up wave per Wf-385-A3 §1 + Wf-385-A3 §5. The Wf-385-A3 gate invokes the trainer with these 14 flags via `DEFAULT_SMOKE_FLAGS` and will FAIL until the wire-in lands — this is the SCAFFOLD-only failure mode that motivated the Wf-385-A3 gate in the first place. `test_wf380_flags_not_yet_wired_honest_negative` asserts (negative) that NOT all 14 Wf-380 flags are present in parse_args; if a future wave lands the wire-in, this test will FAIL and the wave should remove the honest-negative caveat.

2. **Wf-392 flags live in OTHER trainers, not in train_cfm_yuelbond_wasserstein.py**:
   - `--multiph-pll` → `molmetal/scripts/train_se3sb.py` (Wf-392-A1 ship target)
   - `--chemprop-v3-cache` → `molmetal/scripts/train_chemprop_v2_v3.py` (Wf-392-A2 ship target)
   - These flags are domain-specific to their respective trainers (SE(3) SB + chemprop v2/v3) and intentionally do NOT appear in the CFM trainer's parse_args.

3. **Wf-385-A3 gate SUBPROCESS smoke (50-step CPU training with all 14 Wf-380 flags ON) is NOT exercised in this test suite** — it requires the torch-enabled runtime and runs ~30-90 s. The `pre_flight_check.py --pipeline all` invocation already covers this on every fresh-run by virtue of the Wf-385 gate being wired into `build_gates('all')`. The test suite is intentionally torch-light (uses `--skip-smoke` mode + AST introspection) so it can run on any host Python.

4. **Pre-flight cascade NOT re-run by Wf-395-A3** — `pre_flight_check.py --pipeline all` invocation is the surface that exercises the Wf-385-A3 gate subprocess smoke. Wf-395-A3 is a CPU-only docs+code+test+verify wave per AGENTS.md §5.1 + kill_then_adapt §0; the Wf-393..Wf-394 baseline pre-flight run (89/2/3/94) is PRESERVED.

---

## §4 Constraints honoured

| Constraint | Status |
|---|---|
| **0 GPU launches** | ✅ per AGENTS.md §5.1 + kill_then_adapt §0 |
| **0 `paper/*` touches** | ✅ per AGENTS.md §5.3 + TODO-84 |
| **0 wet-lab planning** | ✅ per AGENTS.md §5.4 + §10 |
| **0 sudo** | ✅ per environment constraints |
| **Bash + Read + Write only** | ✅ no Skill / no WebFetch / no WebSearch |
| **AST parse check on trainer file required** | ✅ `train_cfm_yuelbond_wasserstein.py` AST-parses clean |
| **pre-flight v3 fresh-run allowed (CPU-only docs)** | ✅ `pre_flight_verify.json` written via `build_gates('all')` introspection |
| **MEASURED-vs-PROJECTED tag discipline** | ✅ §3.1 MEASURED + §3.2 honest-negative caveats |
| **Honest framing per `party-twenty-spirit` §3.5** | ✅ 4 honest-negative caveats preserved in §3.2 |
| **Wired-up verification (not aspirational)** | ✅ 22 NEW tests PASS in 1.60 s on CadButEaas venv |
| **Pre-flight pipeline integration verified** | ✅ Wf-385 gate at index 72/117 |

---

## §5 Cross-references

- `molmetal/tests/test_wf395a3_full_integration.py` — Wf-395-A3 NEW test suite (~520 LOC, 22 NEW tests PASS)
- `molmetal/reports/wf395a3_full_integration/preflight_verify.json` — Wf-395-A3 preflight verify JSON (~600 bytes)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — CFM trainer (Wf-388 wire-in target)
- `molmetal/scripts/wf385a3_scaffold_wire_in_gate.py` — Wf-385-A3 standalone gate script
- `molmetal/scripts/pre_flight_check.py` — pre-flight v3 (gate index 72 = Wf-385-A3)
- `molmetal/tests/test_wf385a3_scaffold_wire_in_gate.py` — Wf-385-A3 test surface (20 NEW tests)
- `molmetal/tests/test_wf392a2_chemprop_v3_cache.py` — Wf-392-A2 test surface (2 NEW tests)
- `molmetal/tests/test_wf388a2_diversity_wire.py` — Wf-388-A2 test surface
- `molmetal/tests/test_wf388a3_sa_wire.py` — Wf-388-A3 test surface
- `TODO/completed/385a3_scaffold_wire_in_gate.md` — Wf-385-A3 ship (~339 LOC)
- `TODO/completed/388a2_diversity_wire.md` — Wf-388-A2 ship (6 diversity flags)
- `TODO/completed/388a3_sa_wire.md` — Wf-388-A3 ship (2 SA flags)
- `TODO/completed/392a1_multiph_pll_wire.md` — Wf-392-A1 ship (Multi-PH + PLL)
- `TODO/completed/392a2_chemprop_v3_cache_wire.md` — Wf-392-A2 ship (chemprop v3 cache)
- `TODO/completed/393_final_synthesis.md` — Wf-393 final synthesis v32 (precursor wave)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §10 — standing directives re-confirmed
- `party-twenty-spirit` §3.5 — MEASURED-vs-PROJECTED tag discipline
- `kill_then_adapt` §0 — NO GPU launch + USER_OK gating

---

**END OF Wf-395-A3 CLOSEOUT** — owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 all re-confirmed; 22 NEW tests PASS in 1.60 s wall (CPU-only; torch 2.14.0+cu130); 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning; AST-parse OK; Wf-385-A3 gate verified at `build_gates('all')` index 72/117.
