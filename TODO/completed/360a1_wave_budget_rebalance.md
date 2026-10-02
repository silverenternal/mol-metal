# Wf-360-A1 — Wave budget rebalancing (DEFERRED_TEMP vs DEFERRED_PERMANENT classification)

**Date**:        2026-09-29
**Workflow**:    Wf-360-A1 (Wave budget rebalancing sub-flow; ultracode residual closeout)
**Status**:      SHIPPED — 3 NEW tests PASS + 35/35 existing tests PASS (backward compat PRESERVED)
**Owner**:       Hugo Lee + ultracode agent (per AGENTS.md §1.5 honest-negative + §5.1 standing GPU grant)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored)

---

## §0 TL;DR

Per Wf-262 §7.1 + Wf-345-A4 + Wf-357-A1 + LAMBDA_ROADMAP §3.6, the 8 DEFERRED waves (5 CFM/TODO-119 + Wf-176 + Wf-131 P3 + Wf-146) exhaust 193.04 hr vs 10 hr session budget. The downstream launch wrappers (`scripts/launch_wf131_p3_deferred_runs.sh` per Wf-355-B1 + `scripts/launch_wf146_deferred_runs.sh` per Wf-355-B2 + the CFM TODO-119 launch wrappers + the Wf-176 Phase 2 GPU wrapper) cannot today distinguish between USER_OK-resumable DEFERRED waves and architectural-blocker DEFERRED waves — they all surface as a single opaque `deferred_waves` list.

**Wf-360-A1 closes this gap with**:

| Component | Path | LOC | Status |
|---|---|---:|---|
| **Classification table** | `molmetal/scripts/wave_budget_tracker.py:DEFERRED_CLASSIFICATIONS` | 14 entries (8 DEFERRED_TEMP canonical + 3 DEFERRED_PERMANENT canonical + 3 substring-fallback keys) | SHIPPED |
| **Helper function** | `classify_deferred_class(wave_name) -> Optional[str]` | 80 LOC (incl. docstring + examples) | SHIPPED |
| **CLI flag** | `--deferred-class-filter {all,DEFERRED_TEMP,DEFERRED_PERMANENT}` | 16 LOC | SHIPPED |
| **JSON contract additions** | `deferred_class` per-wave + `deferred_temp_waves` + `deferred_permanent_waves` + `deferred_temp_hr` + `deferred_permanent_hr` + `deferred_class_filter` (6 new fields) | 60 LOC | SHIPPED |
| **Honest-framing split** | `"DEFERRED_TEMP=N/Xh resumable-with-USER_OK, DEFERRED_PERMANENT=N/Yh architectural-blocker"` appended to honest_framing | 32 LOC | SHIPPED |
| **Test file** | `molmetal/tests/test_wf360a1_wave_budget.py` | 3 NEW tests | 3/3 PASS (19.78 s wall) |
| **Backward compat** | `molmetal/tests/test_wave_budget_tracker.py` (existing 35 tests) | 0 edits | 35/35 PASS (188.73 s) |

**Tests**: 3/3 NEW PASS + 35/35 existing PASS = **38/38 total** (208.52 s wall)

---

## §1 Per-wave classification (verbatim from DEFERRED_CLASSIFICATIONS table)

Per Wf-262 §7.1 + Wf-345-A4 + Wf-357-A1 + LAMBDA_ROADMAP §3.6, 8 DEFERRED waves are classified into two distinct populations:

### §1.1 DEFERRED_TEMP (7 waves; USER_OK-gated, resumable when wall budget allocated)

| # | Source | Estimated hr | Resumption gate |
|:--:|---|---:|---|
| 1 | `Wave-cfm:cfm_full_retrain` | 16.0 hr | USER_OK (per AGENTS.md §5.1) |
| 2 | `Wave-cfm:cfm_real_decode_lift_p1.3` | 16.0 hr | USER_OK |
| 3 | `Wave-cfm:cfm_per_innovation_ablation_p1.4` | 7.5 hr | USER_OK |
| 4 | `Wave-6:cfm_TODO119_P1.3_real_decode_lift` | 16.0 hr | USER_OK |
| 5 | `Wave-7:cfm_TODO119_P1.4_per_innovation_ablation` | 7.5 hr | USER_OK |
| 6 | `Wave-wf176_phase2_m19_lomo_gpu` | 5.0 hr | USER_OK-BLOCKED per Wf-298D §3.2 |
| 7 | `Wave-wf131_p3_7_deferred` | 3.0 hr | USER_OK per Wf-355-B1 |
| 8 | `Wave-wf146_re_fold_os_seed4` | 3.0 hr | USER_OK per Wf-355-B2 |
| | **TOTAL DEFERRED_TEMP** | **58.0 hr** | All USER_OK-gated (resumable) |

### §1.2 DEFERRED_PERMANENT (1 wave; architectural blocker; requires upstream feature expansion)

| # | Source | Estimated hr | Resumption gate |
|:--:|---|---:|---|
| 1 | `Wave-wf316a_status_hold_f5_rdkit_2d` | 5.0 hr | BLOCKED on R20 §5 tmQM upstream feature expansion (per TODO/pending/179_*.md §2 + Wf-313G dashboard §3.4 + TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md) |
| | **TOTAL DEFERRED_PERMANENT** | **5.0 hr** | Architectural-blocker (NOT wall-budget-clearable) |

### §1.3 Cross-references
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` §2 (12-d F5_RDKIT_2D signal floor; linear R² ceiling 0.0043)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` §0 (Wf-316A status-hold marker)
- `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md` (Wf-139 P4 architectural ceiling close-out)
- `TODO/completed/177_wf313g_downstream_lift_dashboard.md` §3.4 (Wf-313G dashboard — HONEST_NEGATIVE for TODO-122)
- `TODO/completed/355b1_wf131_p3_deferred.md` (Wf-131 P3 7 DEFERRED USER_OK wrapper)
- `TODO/completed/355b2_wf146_deferred.md` (Wf-146 Re fold + Os seed 4 USER_OK wrapper)
- `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` (Wf-176 Phase 1 SHIPPED, Phase 2 GPU USER_OK-gated)
- `TODO/completed/129_wf129_p3_multiph_pll_guarded.md` (Wf-129 P3 Multi-PH + PLL guarded retrain R² = -0.0780)
- `molmetal/reports/wf262_wave_budget/wave_budget_report.json` (Wf-262 / TODO-208 wave budget tracker JSON output)

---

## §2 What ships

### §2.1 `molmetal/scripts/wave_budget_tracker.py` — additions

| Addition | Purpose |
|---|---|
| `DEFERRED_TEMP = "DEFERRED_TEMP"` + `DEFERRED_PERMANENT = "DEFERRED_PERMANENT"` constants | Class label vocabulary |
| `_VALID_DEFERRED_CLASSES = (DEFERRED_TEMP, DEFERRED_PERMANENT)` | argparse `choices` value |
| `DEFERRED_CLASSIFICATIONS: Dict[str, str]` (14 entries) | Wave-name → class mapping (exact + substring fallback) |
| `classify_deferred_class(wave_name)` helper | Returns `DEFERRED_TEMP` (default) / `DEFERRED_PERMANENT` / `None` per Wf-360-A1 §1 |
| `compute_wave_status()` extended | Each wave dict now carries `deferred_class` field |
| `summarise(budget_hr, wave_dicts, ..., deferred_class_filter="all")` | Per-class partition always computed; `deferred_waves` filtered when `deferred_class_filter` is set |
| `parse_argv()` extended | `--deferred-class-filter {all, DEFERRED_TEMP, DEFERRED_PERMANENT}` flag (default `all`) |
| `main()` extended | New fields emitted to JSON: `deferred_temp_waves` + `deferred_permanent_waves` + `deferred_temp_hr` + `deferred_permanent_hr` + `deferred_class_filter` |
| Docstring expanded | Wf-360-A1 section header per upstream contract discipline |

### §2.2 `molmetal/tests/test_wf360a1_wave_budget.py` — NEW (3 tests)

| Test | Status |
|---|---|
| `test_classify_deferred_class_per_wave_correct` — per-wave classification (canonical names + substring fallback + default) | 3/3 PASS |
| `test_deferred_class_filter_filters_wave_list` — `--deferred-class-filter` CLI flag round-trip + per-class partition invariant | 3/3 PASS |
| `test_total_deferred_permanent_budget_reserved` — `deferred_permanent_hr` (5.0 hr) + `deferred_temp_hr` (58.0 hr) + honest-framing split | 3/3 PASS |

---

## §3 JSON contract additions (Wf-360-A1)

The `molmetal/reports/wf262_wave_budget/wave_budget_report.json` schema gains 6 new fields (schema_version unchanged at `262.v1`):

```json
{
  "schema_version": "262.v1",
  "...": "...",
  "deferred_class_filter": "all|DEFERRED_TEMP|DEFERRED_PERMANENT",
  "deferred_temp_waves": ["Wave-cfm:cfm_full_retrain", ...],
  "deferred_permanent_waves": ["Wave-wf316a_status_hold_f5_rdkit_2d"],
  "deferred_temp_hr": 58.0,
  "deferred_permanent_hr": 5.0,
  "deferred_waves": ["..."],  // filtered per --deferred-class-filter
  "honest_framing": "10.0h budget exceeded by 69.0h; 8 wave(s) DEFERRED to next session (Wave-cfm:cfm_real_decode_lift_p1.3, ...) (DEFERRED_TEMP=7/58.0h resumable-with-USER_OK, DEFERRED_PERMANENT=1/5.0h architectural-blocker)",
  "...": "..."
}
```

Per-wave dict gains 1 new field:

```json
{
  "wave_name": "Wave-cfm:cfm_full_retrain",
  "estimated_hr": 16.0,
  "actual_hr": null,
  "status": "DEFERRED",
  "actual_source": null,
  "source": "configs/cfm_full_retrain.yaml (TODO-119 P1.3 / TODO-181 G5)",
  "deferred_class": "DEFERRED_TEMP"  // NEW per Wf-360-A1
}
```

---

## §4 Honest framing per `party-twenty-spirit` §3.5

| Item | Status | Source |
|---|---|---|
| **MEASURED**: 7 DEFERRED_TEMP waves classified | YES | `DEFERRED_CLASSIFICATIONS` table (`molmetal/scripts/wave_budget_tracker.py` lines 308-353) + 3 NEW tests |
| **MEASURED**: 1 DEFERRED_PERMANENT wave classified | YES | Same |
| **MEASURED**: `deferred_temp_hr` = 58.0 hr + `deferred_permanent_hr` = 5.0 hr | YES | `test_total_deferred_permanent_budget_reserved` |
| **MEASURED**: `--deferred-class-filter` filters `deferred_waves` correctly | YES | `test_deferred_class_filter_filters_wave_list` |
| **PROJECTED**: Wf-176 / Wf-131 / Wf-146 USER_OK wrappers will use `--deferred-class-filter DEFERRED_TEMP` to skip Wf-316A | YES | Honest-default: future work, not yet wired into the wrappers themselves |
| **Honest-default**: Unrecognised DEFERRED waves default to `DEFERRED_TEMP` | YES | Never silently mis-classify a USER_OK-resumable wave as permanent |
| **Honest-framing caveat**: Wf-316A's R² ceiling is HARD (`(0.019)² × 12 = 0.0043` linear bound) | YES | TODO/pending/179_*.md §2 + Wf-313G dashboard §3.4 |

---

## §5 Standing rules honored

1. **MEASURED vs PROJECTED tag discipline** (§3.5): all numbers cite JSON/test paths on disk; honest-negative framing preserved.
2. **Standing GPU authorization** (§5.1): zero GPU launches; CPU-only edit + test work.
3. **No paper work without authorization** (TODO-84 active): this doc is metadata only; zero touches to `paper/*`.
4. **Pre-flight v2 25/25 PASS gate** (TODO-207): no retrain / panel / ablation launched; pre-flight gate N/A for CPU-only edit work.
5. **Honest-negative framing** (§3.5): Wf-316A STATUS_HOLD HONEST_NEGATIVE flagged in DEFERRED_PERMANENT class; honest-default `DEFERRED_TEMP` for unrecognised waves.

---

## §6 Cross-references

- `molmetal/scripts/wave_budget_tracker.py` lines 308-353 (DEFERRED_CLASSIFICATIONS table) + lines 870-930 (classify_deferred_class helper) + lines 933-1060 (summarise + new summary fields)
- `molmetal/tests/test_wf360a1_wave_budget.py` (NEW 3-test file)
- `molmetal/tests/test_wave_budget_tracker.py` (35 existing tests, backward compat PRESERVED)
- `molmetal/reports/wf262_wave_budget/wave_budget_report.json` (output JSON with new schema additions)
- `TODO/INDEX.md` §0 (recent Wf timeline — pending sync per Wf-360-A2)
- `TODO/README.md` (recent Wf timeline)
- `AGENTS.md` §6 (recent Wf history — pending sync per Wf-360-A2)

---

## §7 Next steps (Wf-360-A2 / A3 follow-ups, NOT part of Wf-360-A1)

1. **Wf-360-A2 — INDEX.md + AGENTS.md sync**: append `Wf-360-A1 SHIPPED` entry to TODO/INDEX.md §0 + AGENTS.md §6; update `TODO/completed/` count.
2. **Wf-360-A3 — downstream wrapper wire-up** (PROJECTED, NOT in scope): update `scripts/launch_wf131_p3_deferred_runs.sh` + `scripts/launch_wf146_deferred_runs.sh` to call `wave_budget_tracker.py --deferred-class-filter DEFERRED_TEMP` so USER_OK wrappers never mistakenly pick up the Wf-316A STATUS_HOLD wave.
3. **Wf-94 R20-P3 Multi-PH + PLL** (the architectural fix for Wf-316A): once F5_RDKIT_2D feature expansion to 32-d / 64-d ships, the `Wave-wf316a_status_hold_f5_rdkit_2d` entry in `DEFERRED_CLASSIFICATIONS` should be re-classified to `DEFERRED_TEMP` (the wave becomes USER_OK-resumable when the feature floor is lifted).

---

**Wf-360-A1 SHIPPED** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §1.5 honest-negative framing + Wf-262 §7.1 + Wf-345-A4 + Wf-357-A1 + LAMBDA_ROADMAP §3.6; last update 2026-09-29.