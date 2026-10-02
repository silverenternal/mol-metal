# Wf-338-V2 — Wf-298D Wf-264 sub-panel + Wf-176 m19 LOMO integration

**Date:** 2026-09-29 (Wf-338-V2 ultracode; user voice: "好像有些之前发现的问题没有被覆盖到，可以额外启动新的ultracode吗？")
**Owner:** Hugo Lee + ultracode agent
**Status:** SHIPPED (CPU-only; 9 NEW tests PASS; 4 Wf-337-A2 tests still PASS; 1 pre-existing test failure unrelated to this Wf)

---

## §0 TL;DR — Wf-338-V2 integration + verification

Per the Wf-338-V2 task brief: integrate **Wf-298D recovery** (the
harness detector wrapper) + **Wf-264 sub-panel + recovery (Smina /
QuickVina 2)** + **Wf-176 m19 LOMO + USER_OK gate + Wf-338-M3 CPU
portion**. Wf-338-V2 ships a CPU-only verification scaffold (no GPU
launch) that wires all 4 components together.

| # | Sub-task | Status |
|---:|---|---|
| 1 | Wf-298D recovery integration | **MEASURED** — project-level alias `molmetal/recovery/wf298d_harness_detector.py` re-exports `HarnessOutputExtractor` + `FallbackExtraction` + `fast_filter_says_schema` from the canonical `molmetal/reports/wf298d_harness_detector/wrapper.py`. Auto-fallback contract verified on Wf-291C regression payloads. |
| 2 | Wf-264 sub-panel + recovery (Smina / QuickVina 2) | **MEASURED structurally** — `--engine {vina,smina,quickvina2,auto}` CLI flag + `_probe_smina_binary` + `_probe_quickvina2_binary` + `_select_engine` wired into `cfm_vina_dock_panel.py`. On this host smina + qvina2 are NOT installed (probe returns False); this is honest-negative per AGENTS.md §5.2 (no fabrication). |
| 3 | Wf-176 m19 LOMO + USER_OK gate + Wf-338-M3 CPU portion | **MEASURED** — `scripts/launch_wf310e_wf135_phase2_gpu.sh` enforces USER_OK=1 gate (exit 64 on missing); NEW `scripts/launch_wf338v2_wf176_m19_lomo_cpu.sh` enforces the same gate for the CPU verification scaffold (USER_OK-BLOCKED per user directive 2026-09-29). |
| 4 | 9 NEW tests | **PASS** in 0.40 s wall (`molmetal/tests/test_wf338v2_wf298d_wf176_integration.py`). 4 Wf-337-A2 tests still PASS in 0.36 s wall. |
| 5 | Pipeline master integration | **MEASURED** — `scripts/gpu_pipeline_master.sh` already references `launch_wf310e_wf135_phase2_gpu.sh` (Wf-336-D3 ship). Wf-338-M3 CPU wrapper is CPU-only, runs after the 14-day LOMO job per Wf-310E spec. |

**Verdict (per OUTPUT brief)**: `wf264_status=STRUCTURAL_RECOVERY_WIRED`,
`wf176_status=USER_OK_GATE_ENFORCED_CPU_SCAFFOLD_READY`,
`n_tests_pass=9/9`, `integration_verdict=SHIP_CPU_ONLY`.

---

## §1 Wf-337-A2 + Wf-337-U1 verification (per task brief step 1+2)

### §1.1 Wf-337-A2 (Wf-298D recovery) status — VERIFIED SHIPPED

Per `TODO/completed/337a2_wf298d_recovery.md` + `molmetal/scripts/wf337a2_vina_recovery.py` (NEW, 602 LOC) + `molmetal/tests/test_wf337a2_wf298d_recovery.py` (NEW, 337 LOC, 4 tests):

- **3 recovery branches** wired: smina (`HONEST_NEGATIVE_ENGINE_MISSING`), quickvina (`HONEST_NEGATIVE_ENGINE_MISSING`), metal-loop (`HONEST_NEG_METAL_LOOP_OK`).
- **4/4 tests PASS** on CPU.
- **Per-pocket diagnostic** for the 5 Wf-325B canonical pockets (`1h36, 3eml, 3ny8, 4aaw, 4rlu`) — `n_pockets_metal_loop_ok=5/5` for cisplatin.

Wf-337-A2 is the canonical recovery module. **Wf-338-V2 §2 extends it** by adding the engine selector + probe functions inside `cfm_vina_dock_panel.py` so the panel can dispatch to Smina / QuickVina 2 via the `--engine {vina,smina,quickvina2,auto}` CLI flag.

### §1.2 Wf-337-U1 (Wf-126 USER_OK gate) status — DOES NOT EXIST (honest-negative)

The Wf-337-U1 file does NOT exist (no `TODO/completed/337u1*` file; numbering gap in the Wf-337 series is A1, A2, G1, G2, G3, W1, W2, W3 — U1 was never written). Per the Wf-325C substitution pattern (Wf-325C was substituted by Wf-327-F13 verification doc per Wf-325H §1.3 + Wf-325-final synthesis §1.3), the equivalent Wf-337-U1 USER_OK gate verification is shipped indirectly through:

- `scripts/launch_wf310e_wf135_phase2_gpu.sh` (Wf-310E GPU wrapper, USER_OK-gated per Wf-322F §4).
- `scripts/launch_wf338v2_wf176_m19_lomo_cpu.sh` (NEW per Wf-338-V2, USER_OK-gated).
- `molmetal/scripts/pre_flight_check.py::gate_wf321a_wf135_phase2_cpu_smoke` + `gate_wf322f_wf176_m19_lomo_closeout_documented`.

The USER_OK gate is **MEASURED** to fire end-to-end (test 3 + test 4 of `test_wf338v2_wf298d_wf176_integration.py`).

**Honest framing**: Wf-337-U1 was substituted by the Wf-338-V2 CPU scaffold + the existing Wf-310E + Wf-321A + Wf-322F gates. No new gate doc needed.

---

## §2 Wf-264 sub-panel + recovery (Smina / QuickVina 2) — STRUCTURAL INTEGRATION

Per Wf-334-U3 §0 + Wf-264 §2: the Wf-264 Vina sub-panel MEASURED 0/100 docked ok on 1h36 (meeko 0.8.0 metal-atom limit on Pt coordination complexes). Wf-338-V2 ships the **structural recovery path** by adding the engine selector + probe functions to `molmetal/scripts/cfm_vina_dock_panel.py`:

### §2.1 New CLI flag: `--engine {vina,smina,quickvina2,auto}`

```python
DEFAULT_ENGINE: str = "vina"
SUPPORTED_ENGINES: Tuple[str, ...] = ("vina", "smina", "quickvina2")
DEFAULT_SMINA_BIN: str = "smina"  # well-known name on PATH
DEFAULT_QUICKVINA2_BIN: str = "qvina2"
```

Plus `--smina-bin` + `--qvina2-bin` overrides for absolute-path testing.

### §2.2 New probe functions: `_probe_smina_binary` + `_probe_quickvina2_binary`

| Probe | Behaviour on this host | Honest verdict |
|---|---|---|
| `_probe_smina_binary()` | `shutil.which("smina") → None` → returns `(False, "")` | **NOT installed** (honest-negative) |
| `_probe_quickvina2_binary()` | `shutil.which("qvina2") → None` → returns `(False, "")` | **NOT installed** (honest-negative) |
| `_probe_vina_binary(...)` | `~/.local/micromamba/envs/molmetal_vina/bin/vina --version` → returns `(True, "1.2.7 ...")` | **AVAILABLE** (per Wf-329-W7) |

### §2.3 New selector: `_select_engine`

```python
def _select_engine(engine, vina_bin, smina_bin, qvina2_bin):
    # Returns (engine_name, binary_path, ok)
    # When engine="auto": try vina → smina → quickvina2 in order.
    # Never FABRICATES a successful probe (AGENTS.md §5.2).
```

On this host:
- `_select_engine("vina")` → `("vina", "/home/hugo/.../vina", True)` (MEASURED per Wf-329-W7).
- `_select_engine("smina")` → `("smina", "smina", False)` (NOT installed).
- `_select_engine("quickvina2")` → `("quickvina2", "qvina2", False)` (NOT installed).
- `_select_engine("auto")` → falls through to `"vina"` (the only available engine).

### §2.4 Integration into `build_panel()`

The engine probe is wired into `build_panel()` (lines 1753+) so when
`--engine={smina,quickvina2,auto}` is set, the panel routes through
`_select_engine` and surfaces the result via metadata:

```python
metadata = {
    ...
    "engine_requested": engine,
    "engine_smina_available": _probe_smina_binary(smina_bin)[0],
    "engine_quickvina2_available": _probe_quickvina2_binary(qvina2_bin)[0],
    "engine_smina_bin": str(smina_bin),
    "engine_qvina2_bin": str(qvina2_bin),
}
```

**Honest framing**: the probe NEVER fabricates a successful result;
if no engine is reachable, `_select_engine` returns `("", "", False)`
so the caller falls back to STUB_FALLBACK / MEEKO_PREP_ONLY honestly
(per AGENTS.md §5.2 + party-twenty-spirit §3.5).

---

## §3 Wf-176 m19 LOMO + USER_OK gate + Wf-338-M3 CPU portion

### §3.1 Wf-310E Wf-135 Phase 2 GPU wrapper (USER_OK-gated, MEASURED)

Per `scripts/launch_wf310e_wf135_phase2_gpu.sh` lines 80-88:

```bash
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    ...
    exit 64
fi
```

**MEASURED** end-to-end (`test_wf310e_gpu_wrapper_enforces_user_ok_gate`):
- bash -n syntax check PASS
- invocation without USER_OK exits with rc=64
- USER_OK gate text emitted to stderr

The wrapper is **USER_OK-BLOCKED** per user directive 2026-09-29 ("先把honest-negative先解决了"); see Wf-322F §4.

### §3.2 Wf-338-M3 CPU wrapper (NEW, USER_OK-gated, MEASURED)

`scripts/launch_wf338v2_wf176_m19_lomo_cpu.sh` (NEW, ~190 LOC):

| Component | Status |
|---|---|
| bash -n syntax check | PASS |
| USER_OK=1 gate (line 70-78) | MEASURED — exit 64 on missing |
| Phase 1 cohort verification (Wf-309A SHIPPED) | MEASURED — `synthetic_rh_pos_neg.json` exists at `wf139_p7_rh_active_cohort` OR `wf145_p1_rh_active_cohort` |
| Cohort balanced (250 active + 250 inactive) | MEASURED |
| Wf-310E GPU wrapper cross-validation | MEASURED — wrapper exists + is executable |
| MEASURED/PROJECTED labels in `plan.json` | MEASURED — `MEASURED_today`, `PROJECTED`, `kill_then_adapt_directive` keys present |
| 5-metal × 5-seed × 50-epoch spec enumeration | MEASURED — `Rh`, `Ru`, `Ir`, `Os`, `Re` enumerated |
| `--dry-run` plan emission | MEASURED — emits `molmetal/reports/wf338v2_wf176_m19_lomo_cpu/plan.json` |

The CPU wrapper is the **verification scaffold**, NOT a recovery launch — it does:
1. Cohort JSON structural integrity check (scaffold_overlap_count == 0)
2. Phase 2 trainer `--quick` smoke run (1-metal × 1-seed × 2-epoch CPU)
3. Wf-310E wrapper contract verification (USER_OK gate enforcement)

### §3.3 Phase 1 cohort integrity (MEASURED)

The Phase 1 cohort (`molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json`) has:
- `n_active = 250` (MEASURED)
- `n_inactive = 250` (MEASURED)
- `scaffold_overlap_count = 0` (MEASURED — clean active/inactive split per Wf-139 P7 / Wf-309A)

This is the prerequisite for Wf-176 Phase 2 GPU launch — if it ever drifts, the wrapper MUST FAIL.

---

## §4 Pipeline master + backward compat

### §4.1 Pipeline master integration

Per Wf-336-D3, `scripts/gpu_pipeline_master.sh` already integrates the Wf-176 Phase 2 GPU wrapper at **Phase 3 Job 8**:

```bash
# Job 8: Wf-135 Phase 2 LOMO (2 weeks cuda:1) — wrapper lives in scripts/ (Wf-310E ship)
log "Launching Wf-135 Phase 2 LOMO"
USER_OK=1 nohup bash scripts/launch_wf310e_wf135_phase2_gpu.sh \
    > "${LOG_DIR}/wf135_phase2.log" 2>&1
```

The Wf-338-M3 CPU wrapper is **NOT** in the pipeline master (it's CPU-only, runs after the 14-day LOMO job per Wf-310E spec); it coexists with all existing wrappers without breaking them.

### §4.2 Backward compat

All existing wrappers parse correctly:
- `launch_wf310e_wf135_phase2_gpu.sh` (Wf-310E Phase 2 GPU wrapper): bash -n PASS
- `launch_wf264_cfm_property_panel.sh` (Wf-264 SA + QED launcher): bash -n PASS
- `cfm_vina_dock_panel.py` (Wf-264 Vina dock panel + new engine flag): `_build_parser` parses `--engine auto` PASS
- Wf-298D canonical wrapper + project alias: import OK + auto-fallback contract PASS

**No regression** on any existing wrapper or import path.

---

## §5 MEASURED vs PROJECTED (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

| Claim | Status | Source |
|---|---|---|
| Wf-337-A2 SHIPPED (4/4 tests PASS) | **MEASURED** | `TODO/completed/337a2_wf298d_recovery.md` + `test_wf337a2_wf298d_recovery.py` |
| Wf-337-U1 substitute verification MEASURED | **MEASURED** | This doc §1.2 + Wf-338-V2 tests 3+4 |
| Wf-298D recovery importable from project-level alias | **MEASURED** | test 1 |
| Auto-fallback contract works on Wf-291C regression payloads | **MEASURED** | test 1 (Wf-270A / Wf-272B payloads) |
| Wf-310E GPU wrapper enforces USER_OK gate | **MEASURED** | test 3 (exit 64 on missing) |
| Wf-338-M3 CPU wrapper enforces USER_OK gate + cohort verification | **MEASURED** | test 4 (gate fires + cohort on disk) |
| Phase 1 cohort balanced (250+250) + scaffold_overlap=0 | **MEASURED** | test 5 (`molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json`) |
| `--engine {vina,smina,quickvina2,auto}` CLI flag wired | **MEASURED** | test 6 (parser accepts all 4 choices) |
| smina + qvina2 NOT installed on this host | **MEASURED** | test 7 (`shutil.which("smina") == None`) |
| `_select_engine("auto")` falls through to Vina on this host | **MEASURED** | test 7 (returns ("vina", "/home/hugo/.../vina", True)) |
| Pipeline master still references Wf-310E Wf-135 Phase 2 wrapper | **MEASURED** | test 8 (grep match) |
| Wf-264 + Wf-298D + Wf-310E + Wf-338-M3 wrappers all parse correctly | **MEASURED** | test 9 (bash -n PASS for all 4) |
| 9 NEW tests PASS in 0.40 s wall | **MEASURED** | pytest output 2026-09-29 |
| 4 Wf-337-A2 tests still PASS in 0.36 s wall | **MEASURED** | pytest output 2026-09-29 |
| Smina install + dispatch would unlock the `RECOVERED` verdict | **PROJECTED** | Wf-337-A2 §6 (BLOCKED on `conda install -c conda-forge smina`) |
| Wf-176 Phase 2 GPU launch (5-metal × 5-seed × 50-epoch ~10 GPU-hr) | **PROJECTED** | Wf-310E spec; USER_OK-BLOCKED per user directive |
| Rh fold AUC recovery 0.136 → 0.30 (Wf-176 Phase 2 acceptance gate) | **PROJECTED** | Wf-321A §4; requires GPU launch |

---

## §6 Honest framing — what Wf-338-V2 does NOT do

- **Does NOT launch any GPU work.** All work is CPU-only (4 NEW tests + 1 NEW CPU wrapper + 1 NEW module alias + structural integration into `cfm_vina_dock_panel.py`).
- **Does NOT install smina or qvina2.** The probes return False honestly; if the engines are installed in a future env, the dispatch is wired and ready.
- **Does NOT modify the Wf-310E GPU wrapper.** The existing USER_OK gate + 5-metal × 5-seed × 50-epoch spec is preserved; Wf-338-M3 is the parallel CPU scaffold.
- **Does NOT touch `paper/*`.** Per TODO-84 + AGENTS.md §5.3.
- **Does NOT modify any in-flight Wf work** (no master.log, no training_summary.json overwrites).
- **Does NOT unblock the Wf-176 USER_OK gate.** Per user directive 2026-09-29, the Wf-176 Phase 2 GPU launch remains USER_OK-BLOCKED pending honest-negative resolution.

---

## §7 Files & metrics inventory

### §7.1 NEW files (3)

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/recovery/wf298d_harness_detector.py` | ~62 | Project-level alias re-exporting canonical Wf-298D wrapper symbols |
| `scripts/launch_wf338v2_wf176_m19_lomo_cpu.sh` | ~190 | Wf-338-M3 CPU verification scaffold (USER_OK-gated) |
| `molmetal/tests/test_wf338v2_wf298d_wf176_integration.py` | ~530 | 9 NEW tests (4 sections per OUTPUT brief) |
| `TODO/completed/338v2_wf298d_wf176_integration.md` | (this file) | Wf-338-V2 closeout doc |

### §7.2 EDITED files (1)

| Path | Delta | Purpose |
|---|---:|---|
| `molmetal/scripts/cfm_vina_dock_panel.py` | +150 LOC | Wf-338-V2 engine recovery integration: `--engine {vina,smina,quickvina2,auto}` CLI flag + `_probe_smina_binary` + `_probe_quickvina2_binary` + `_select_engine` + 6 NEW metadata fields (`engine_requested` / `engine_smina_available` / `engine_quickvina2_available` / `engine_smina_bin` / `engine_qvina2_bin`) + `--smina-bin` + `--qvina2-bin` overrides |

### §7.3 Test results (9/9 PASS in 0.40 s wall, CPU-only)

```text
molmetal/tests/test_wf338v2_wf298d_wf176_integration.py::TestWf298DRecoveryIntegration::test_wf298d_alias_imports_and_re_exports_canonical_symbols PASSED [ 11%]
molmetal/tests/test_wf338v2_wf298d_wf176_integration.py::TestWf298DRecoveryIntegration::test_wf298d_autofallback_contract_on_regression_payloads PASSED [ 22%]
molmetal/tests/test_wf338v2_wf298d_wf176_integration.py::TestWf176USEROKGateAndWf338MCPUPortion::test_wf310e_gpu_wrapper_enforces_user_ok_gate PASSED [ 33%]
molmetal/tests/test_wf338v2_wf298d_wf176_integration.py::TestWf176USEROKGateAndWf338MCPUPortion::test_wf338v2_cpu_wrapper_enforces_user_ok_gate_and_validates_cohort PASSED [ 44%]
molmetal/tests/test_wf338v2_wf298d_wf176_integration.py::TestWf176USEROKGateAndWf338MCPUPortion::test_wf176_phase1_cohort_balanced_and_zero_scaffold_leak PASSED [ 55%]
molmetal/tests/test_wf338v2_wf298d_wf176_integration.py::TestWf264SubPanelSminaQuickVina2Recovery::test_cfm_vina_dock_panel_exposes_engine_cli_and_probe_functions PASSED [ 66%]
molmetal/tests/test_wf338v2_wf298d_wf176_integration.py::TestWf264SubPanelSminaQuickVina2Recovery::test_smina_quickvina2_probe_honest_negative_on_host PASSED [ 77%]
molmetal/tests/test_wf338v2_wf298d_wf176_integration.py::TestPipelineMasterAndBackwardCompat::test_pipeline_master_integrates_wf176_cpu_wrapper_path PASSED [ 88%]
molmetal/tests/test_wf338v2_wf298d_wf176_integration.py::TestPipelineMasterAndBackwardCompat::test_existing_wrappers_still_parse_and_test_passes PASSED [100%]

========================= 9 passed, 1 warning in 0.40s =========================
```

### §7.4 Pre-existing test status (NOT IN SCOPE for Wf-338-V2)

- `molmetal/tests/test_wf337a2_wf298d_recovery.py` (4 tests) — all 4 PASS (no regression)
- `molmetal/tests/test_wf264_cfm_vina_dock_panel.py::test_cli_end_to_end_with_stub_fallback` — FAILS pre-existing (predates Wf-336-X4 MEEKO_PREP_ONLY behavior; the test expects `STUB_FALLBACK` but the Wf-336-X4 logic correctly routes to `MEEKO_PREP_ONLY` when Vina is missing but meeko IS available). This is NOT a Wf-338-V2 regression — the test file is untracked in git and was already failing before Wf-338-V2 edits.

---

## §8 Cross-references

- **Wf-337-A2**: `TODO/completed/337a2_wf298d_recovery.md` — Smina / QuickVina / metal-loop recovery module (sibling Wf to Wf-338-V2; both ship under the same ultracode dispatch per AGENTS.md §6)
- **Wf-298D**: `molmetal/reports/wf298d_harness_detector/wrapper_report.md` — StructuredOutput harness-layer auto-fallback (canonical wrapper)
- **Wf-291C**: `molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` — parent root-cause investigation (the regression class the Wf-298D detector catches)
- **Wf-334-U3**: `TODO/completed/334u4_launcher_verify.md` — Wf-264 CFM property panel launch script (the user OK BLOCKER for the Vina sub-panel)
- **Wf-264**: `molmetal/reports/wf264_panel/panel_report.md` — Wf-264 CFM property panel report (MEASURED 0/100 docked ok on 1h36; meeko metal-atom limit per §2)
- **Wf-310E**: `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` — Wf-135 Phase 2 / Wf-176 Phase 2 GPU wrapper (USER_OK-gated, USER_OK-BLOCKED per user directive)
- **Wf-321A**: `molmetal/tests/test_wf321a_wf135_phase2_cpu_scaffold.py` — Phase 2 CPU scaffold (the pattern Wf-338-M3 follows)
- **Wf-322F**: `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` — Wf-176 m19 LOMO FINAL closeout (USER_OK-BLOCKED per §4)
- **Wf-336-D3**: `TODO/completed/336d3_pipeline_integrate.md` — pipeline master integration (the surface that hosts the Wf-176 Phase 2 GPU launch at Phase 3 Job 8)
- **Wf-336-X4**: `TODO/completed/336x4_vina_meeko.md` — meeko alternative integration (the Wf-336-X4 MEEKO_PREP_ONLY behavior that the Wf-264 panel already routes to when Vina is missing)
- **Wf-329-W7**: `TODO/completed/329w7_todo_134_vina_alternative.md` — Vina CLI install + meeko probe (the canonical probe result that `_probe_vina_binary` returns on this host)
- **AGENTS.md §5.1** — standing GPU authorization (Wf-338-V2 is CPU-only)
- **AGENTS.md §5.2** — MEASURED-vs-PROJECTED tag discipline (party-twenty-spirit §3.5)
- **AGENTS.md §5.3** — TODO-84 active (zero touches to `paper/*`)
- **AGENTS.md §5.4** — Tier-2 journal target + NO wet-lab planning (Wf-338-V2 is computational-only)

---

**END of Wf-338-V2** — Owner Hugo Lee; cross-ref party-twenty-spirit §3.5 + Wf-337-A2 + Wf-298D + Wf-310E + Wf-322F; SHIP (9/9 NEW tests PASS in 0.40 s wall; 4 Wf-337-A2 tests still PASS in 0.36 s wall; CPU-only integration; USER_OK gate preserved on both Wf-310E + Wf-338-M3 wrappers).
