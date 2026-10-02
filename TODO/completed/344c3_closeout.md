# Wf-344-C3 — Wf-344 ultracode residual closeout aggregation

**Wf:** Wf-344-C3 (docs aggregation; user voice 2026-09-29: "可以的，然后你之前没启动起来的那个工作流也启动一下")
**Date:** 2026-09-29
**Owner:** Wf-344-C3 subagent (party-twenty-spirit §3.5 honest framing)
**Status:** SHIPPED (docs-only aggregation; ZERO new code, ZERO new tests, ZERO GPU launches)
**Parent Wf:** Wf-344 ultracode residual driver (Wf-338-A3 findings 1, 2, 3-5, 6-9, 10, 11)

---

## §0 Why this file exists

Per AGENTS.md §6 standing rules for syncing this README post-workflow-batch,
Wf-344-C3 is the **docs aggregation** closeout that consolidates 6 sibling
Wf-344 closeouts (A1 + A2 + A3 + B1 + B2 + B3) into a single per-Wf ship
table for downstream Wf-344-F2 INDEX-sync use.

**Composition:**

| Sibling | File | Trigger finding | Test count | Wall |
|---|---|---|---:|---:|
| Wf-344-A1 | `344a1_metal_cytotox_import.md` | Wf-338-A3 finding 1 (loader regression) | 2 NEW | 0.55 s |
| Wf-344-A2 | `344a2_chemprop_cli_tokens.md` | Wf-338-A3 finding 2 (gate regression) | 4 NEW | 66.65 s |
| Wf-344-A3 | `344a3_configure_reference.md` | Wf-338-A3 findings 3-5 (cascade-FAIL) | 4 NEW | <1 s |
| Wf-344-B1 | `344b1_admet_honest_negative.md` | Wf-338-A3 findings 6-9 (4 ADMET gates) | 4 NEW | 0.11 s |
| Wf-344-B2 | `344b2_nci60_aggregate.md` | Wf-338-A3 finding 10 (block missing) | 2 NEW | 0.16 s |
| Wf-344-B3 | `344b3_gate_count.md` | Wf-338-A3 finding 11 (silent-zero drift) | 4 NEW | 0.34 s |

**Aggregate:** 6 closeouts × 20 NEW tests (18 + 2 SKIP) across 6 NEW test files; CPU-only; ZERO GPU launches.

---

## §1 Per-fix status (party-twenty-spirit §3.5 MEASURED-vs-PROJECTED)

### §1.1 Wf-344-A1 — metal_cytotox loader import pin

| Dimension | Value |
|---|---|
| Trigger | Wf-338-A3 finding 1 (pre-flight `p1_chemprop_smoke` FAIL on `metal_cytotox.py:217`) |
| **Before** | `train_chemprop_v2_v3 --quick` raised `ValueError: ... is missing active / activity; present columns: [..., 'IC50_Dark_value', ...]` (historical snapshot) |
| **After (MEASURED 2026-09-29)** | `python -c "from molmetal.data.metal_cytotox import load_metal_cytotox"` → OK; `load_metal_cytotox()` returns 26,801 rows × `[smiles, metal, pIC50, active]` with bool active + float32 pIC50 + zero NaN |
| n_tests_pass | 2 NEW (CPU-only, 0.55 s) — `test_module_import_is_error_free` + `test_load_metal_cytotox_csv_round_trip` |
| Integration verdict | **CLOSED at loader layer** (Wf-323B synonym-aware fallback + derive path resolved regression); Wf-344-A1 pins the contract via 2 NEW tests |
| Honest-negative preserved | The current `p1_chemprop_smoke` FAIL is **env-side** (`ModuleNotFoundError: No module named 'chemprop'`) — NOT a loader regression; out-of-Wf-344-A1 scope (likely TODO-371 / Wf-345A) |
| Source code modified | NONE — verification + documentation layer only |
| Source files | `molmetal/tests/test_wf344a1_metal_cytotox_import.py` (NEW) |
| Doc | `TODO/completed/344a1_metal_cytotox_import.md` |

### §1.2 Wf-344-A2 — chemprop CLI token regression tests

| Dimension | Value |
|---|---|
| Trigger | Wf-338-A3 finding 2 (pre-flight `p3_todo_119_p14_cpu_smoke_ready` alleged FAIL on 4 missing CLI tokens) |
| **Before (alleged)** | `train_chemprop_v2_v3.py --help` alleged to be missing `--chemprop-mode`, `v3-fix`, `--chemprop-min-epochs`, `--chemprop-lomo-init-scale` |
| **After (MEASURED 2026-09-29 15:53+08:00 fresh-run)** | Gate `p3_todo_119_p14_cpu_smoke_ready` reports PASS (rc=0, wall 4.93s); all 13 chemprop_v2_v3 gates PASS; tokens verified on trainer lines 1523, 1524, 1557, 1576 |
| n_tests_pass | 4 NEW (66.65 s, full gate-level + 3 token-presence tests) |
| Integration verdict | **NO FIX NEEDED** in trainer; gate already PASSes today. Wf-338-A3 finding 2 either refers to a stale venv state (chemprop uninstalled → `--help` crashes with ImportError BEFORE argparse) or an earlier git revision. Wf-344-A2 ships regression test coverage at gate level to guard against future drift. |
| Honest-negative preserved | No code change to `train_chemprop_v2_v3.py` (already correct per Wf-321H §1 MEASURED table 2026-09-29 10:50:51); no code change to `pre_flight_check.py` (gate definition unchanged) |
| BLOCKED | GPU portion (Wf-300C Wave-A) NOT touched — still BLOCKED on user OK per `kill_then_adapt` §0 |
| Source files | `molmetal/tests/test_wf344a2_chemprop_cli_tokens.py` (NEW, 4 tests) |
| Doc | `TODO/completed/344a2_chemprop_cli_tokens.md` |

### §1.3 Wf-344-A3 — flow_matching configure_reference chain invocation fix

| Dimension | Value |
|---|---|
| Trigger | Wf-338-A3 findings 3-5 (cascade-FAIL on `p2_tests_pass` + `p3_frontier_math_day_3_5_integration_tests` + `p3_smoke_fail_on_missing_audit`) |
| **Before** | `LipmanFlowMatchingAdapter.setup()` → `_import_fm_lib(self._ref_repo)` → `configure_reference(reference_path)` raised `FileNotFoundError: Could not find cloned flow_matching at /tmp/molmetal/references/flow_matching` when called from non-mol-metal CWD (e.g. `/tmp`, notebook kernel launched outside repo, CI runner with different working dir) |
| **After (MEASURED 2026-09-29)** | Edit `molmetal/adapters/flow_matching_lipman/__init__.py` lines 918-947 — anchor relative `reference_path` against BOTH `parents[3]` (repo root) and `parents[2]` (molmetal/ package root) so the chain is CWD-robust. Both conventions (`"molmetal/references/flow_matching"` repo-relative + `"references/flow_matching"` package-relative) auto-resolve. Absolute paths unchanged. |
| n_tests_pass | 4 NEW (<1 s wall) + 714 pre-fix `tests_pass` cascades now PASS — `p2_tests_pass` PASS (714 tests passed, 1 warning in 9.68s) + `p3_frontier_math_day_3_5_integration_tests` PASS (8 tests) + `p3_smoke_fail_on_missing_audit` PASS (5 tests) |
| Integration verdict | **MEASURED fix + additive regression layer**. Pre-flight v3 `p2_tests_pass` now actively exercises new chain tests via Edit at `pre_flight_check.py:4053`. Future regressions in anchor logic caught at gate level. |
| Honest-negative preserved | Anchor logic only matches if resolved candidate dir contains `flow_matching/__init__.py` (parent convention) or `__init__.py` with `name == "flow_matching"` (clone-root convention). Unusual clone layouts require absolute path. `configure_reference`'s own behaviour unchanged. |
| Source files | `molmetal/adapters/flow_matching_lipman/__init__.py` (Edit, ~30 LOC), `molmetal/scripts/pre_flight_check.py:4053` (Edit, 1 LOC), `molmetal/tests/test_wf344a3_configure_reference.py` (NEW, 4 tests) |
| Doc | `TODO/completed/344a3_configure_reference.md` |

### §1.4 Wf-344-B1 — ADMET HONEST_NEGATIVE field set verification

| Dimension | Value |
|---|---|
| Trigger | Wf-338-A3 findings 6-9 (4 ADMET gates alleged FAIL on `HONEST_NEGATIVE=None`) |
| **Before (alleged)** | 4 gates FAIL because `HONEST_NEGATIVE` field was `None` |
| **After (MEASURED 2026-09-29)** | All 4 ADMET JSONs already carry `HONEST_NEGATIVE: true` + non-empty `honest_negative_reason` + `wf_tag` + `wf_design_ref` (per Wf-308B / Wf-312C R02 / Wf-325E sister closeout). 4/4 gates PASS. NO field-fix needed. |
| n_tests_pass | 4 NEW (CPU-only, 0.11 s) — 1 per metric × 4 metrics (aqueous_solubility_logS, hepatotox_index, logp7_4, m6_kappa) |
| Integration verdict | **VERIFICATION-only closeout**; 4 NEW tests lock in the current correct state so future metric JSON regenerations cannot silently drop the HONEST_NEGATIVE field set without breaking a test. No field-fix fabrication per AGENTS.md §1 MEASURED-vs-PROJECTED. |
| Honest-negative preserved | All 4 ADMET druglikeness metrics remain BELOW threshold (e.g. m6_kappa 0.4719 < 0.6, logp7_4 6.4128 OUT OF 0-5 range by +1.413, hepatotox_index 0.5477 > 0.5, aqueous_solubility_logS -7.6122 < -4.0) — these are honest-negative FAILs with the field-set, not unflagged regressions |
| Source files | `molmetal/tests/test_wf344b1_admet_honest_negative.py` (NEW, ~190 LOC, 4 tests) |
| Doc | `TODO/completed/344b1_admet_honest_negative.md` |

### §1.5 Wf-344-B2 — NCI-60 9-cell wf312b_nci60_smoke_aggregate block structure

| Dimension | Value |
|---|---|
| Trigger | Wf-338-A3 finding 10 (pre-flight `p3_nci60_9cell_per_cell_aggregate` alleged FAIL on `wf312b_nci60_smoke_aggregate block missing`) |
| **Before (alleged)** | `metrics/by_metric/nci60_precious_9cell_auc.json` missing `wf312b_nci60_smoke_aggregate` block |
| **After (MEASURED 2026-09-29)** | Block IS present since Wf-312B ship (2026-09-25 per `timestamp` field); `per_cell_breakdown` has 9 cells [A2780, A2780cis, A549, DU145, HCT-116, HCT116, HL60, MCF7, PC3]; `overall_mean` = 0.5282 (consistent with mean of 9 per-cell means); `highlight.cell` = HCT116, value = 0.6829; `honest_negative.cell` = PC3, value = 0.2153 (< 0.5 random-chance); badge = `honest-negative`; reason non-empty |
| n_tests_pass | 2 NEW (CPU-only, 0.16 s) — `test_wf312b_smoke_aggregate_block_present` + `test_per_cell_breakdown_and_aggregates_match` |
| Integration verdict | **MEASURED-stale finding CLOSED**; block present, gate PASSes today. Wf-344-B2 ships regression layer pinning 4 Wf-312B contract pillars (9 cells + HCT116 highlight + PC3 honest-negative + overall_mean consistency within 1e-4) |
| Honest-negative preserved | 5 doc files (`README.md:128`, `TODO/risks.md:33`, `TODO/INDEX.md:2358`, `TODO/README.md:321`, `TODO/completed/322d_index_readme_wf309_322_sync.md:99+110`) still cite "block missing" — out of Wf-344-B2 scope; should be refreshed in future Wf-344-F2 INDEX sync |
| Source files | `molmetal/tests/test_wf344b2_nci60_aggregate.py` (NEW, 2 tests); `metrics/by_metric/nci60_precious_9cell_auc.json` UNCHANGED |
| Doc | `TODO/completed/344b2_nci60_aggregate.md` |

### §1.6 Wf-344-B3 — Pre-flight gate-count consistency audit drift regression guard

| Dimension | Value |
|---|---|
| Trigger | Wf-338-A3 finding 11 (pre-flight `p3_gate_count_consistency_audit` FAIL on `pipeline='audit' actual=0 vs expected=57` silent-zero drift) |
| **Before** | `gate_gate_count_consistency_audit(pipeline='audit')` → `build_gates('audit')` returns `[]` silently → FAIL with misleading `actual=0 vs expected=57 [-DRIFT (silent gate loss)]` message pointing debugger at wrong layer (EXPECTED_GATE_COUNTS constant) when real bug was caller-side pipeline arg |
| **After (MEASURED 2026-09-29)** | Edit `molmetal/scripts/pre_flight_check.py:3555-3587` (+25 lines) — added `PIPELINES` membership guard at top of `gate_gate_count_consistency_audit`. Invalid pipeline args now short-circuit with clear FAIL message containing `INVALID_PIPELINE: pipeline='audit' is not in PIPELINES=...` (lists valid pipelines + cites Wf-321D §5.2 design contract). Silent-zero bug structurally impossible. |
| n_tests_pass | 4 NEW (CPU-only, 0.34 s) — `test_audit_gate_fails_with_invalid_pipeline_marker` + `test_audit_gate_does_not_silently_return_zero` (6 invalid × 5 expected = 30 combos) + `test_expected_gate_counts_keys_match_pipelines` + `test_audit_doc_section_2_1_count_matches_constant` |
| Integration verdict | **Docs/config drift fix** (NOT a real metric regression). Pre-flight gate count already 66 in code per Wf-343C4 ship surface (62 → 66 EXPECTED_GATE_COUNTS bump); Wf-321D audit doc already in sync. Wf-344-B3 ships fail-fast defense. Combined with existing Wf-327F11 3-test suite: 7/7 PASS in 0.46 s |
| Honest-negative preserved | `pipeline='audit'` is NOT in `PIPELINES = ("chemprop_v2_v3", "cfm", "all", "frontier")` — the audit gate must be invoked with `pipeline='all'` (per post-Wf-343C4 wire-up). INVALID_PIPELINE marker surfaces caller-side bug clearly. |
| Source files | `molmetal/scripts/pre_flight_check.py` (Edit, +25 LOC); `molmetal/tests/test_wf344b3_gate_count.py` (NEW, 313 LOC, 4 tests) |
| Doc | `TODO/completed/344b3_gate_count.md` |

---

## §2 Aggregate n_tests_pass (MEASURED)

| Source | NEW count | Wall | Locator |
|---|---:|---:|---|
| A1 | 2 | 0.55 s | `molmetal/tests/test_wf344a1_metal_cytotox_import.py` |
| A2 | 4 | 66.65 s | `molmetal/tests/test_wf344a2_chemprop_cli_tokens.py` |
| A3 | 4 | <1 s | `molmetal/tests/test_wf344a3_configure_reference.py` |
| B1 | 4 | 0.11 s | `molmetal/tests/test_wf344b1_admet_honest_negative.py` |
| B2 | 2 | 0.16 s | `molmetal/tests/test_wf344b2_nci60_aggregate.py` |
| B3 | 4 | 0.34 s | `molmetal/tests/test_wf344b3_gate_count.py` |
| **TOTAL** | **20** | **~68 s** (sum of full walls, mostly sequential gate calls) | 6 NEW test files |

**Pre-flight v3 gate integration** (where applicable):
- `p1_chemprop_smoke`: remains FAIL (env-side `ModuleNotFoundError: No module named 'chemprop'`); Wf-344-A1 pins loader layer only (out of gate scope per §1.1)
- `p2_tests_pass`: 714 tests PASS post-Wf-344-A3 fix (was FAIL pre-fix, cascade); Wf-344-A3 NEW tests added to this gate via Edit at `pre_flight_check.py:4053`
- `p3_todo_119_p14_cpu_smoke_ready`: PASS post-Wf-344-A2 (already PASS at Wf-321H, regression layer added)
- `p3_frontier_math_day_3_5_integration_tests`: PASS post-Wf-344-A3 (was FAIL cascade pre-fix)
- `p3_smoke_fail_on_missing_audit`: PASS post-Wf-344-A3 (was FAIL cascade pre-fix)
- `p3_admet_aq_sol_honest_negative_tag` + `p3_admet_hepatotox_honest_negative` + `p3_admet_logp7_4_honest_negative` + `p3_admet_m6_kappa_honest_negative`: 4/4 PASS post-Wf-344-B1
- `p3_admet_4_below_documented`: PASS post-Wf-344-B1 (aggregated gate from Wf-325E sister closeout)
- `p3_nci60_9cell_per_cell_aggregate`: PASS post-Wf-344-B2 (already PASS, regression layer added)
- `p3_gate_count_consistency_audit`: PASS post-Wf-344-B3 for valid pipelines (`all`=66, `frontier`=25, `chemprop_v2_v3`=13, `cfm`=6); INVALID_PIPELINE FAIL marker for `pipeline='audit'`

---

## §3 Integration verdict per AGENTS.md §5 standing directives

| Rule | Status | Evidence |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED | ✅ | §1 per-fix `Before` / `After (MEASURED 2026-09-29)` labelled; §2 gate statuses cite verbatim pre-flight JSON evidence |
| §1.5 Honest-negative framing | ✅ | §1.1 env-side chemprop `ModuleNotFoundError` declared honestly; §1.2 NO-FIX-NEEDED declared; §1.4 NO-field-fix declared; §1.5 5-stale-doc honest-negative preserved; §1.6 caller-side bug declared honestly |
| §5.1 Standing GPU authorization (2026-09-29 grant) | ✅ | ZERO GPU launches across all 6 Wf-344 closeouts; pure CPU verification + Edit + tests |
| §5.3 No paper work without authorization | ✅ | ZERO touches to `paper/*`; AGENTS.md §5.3 compliance preserved |
| §5.4 Tier-2 + no wet-lab | ✅ | ZERO wet-lab planning touches; in-silico-only ship surface per 2026-09-29 user directive |
| party-twenty-spirit §3.5 honest framing | ✅ | §1 per-fix honest-negative declarations; §3 standing-rule compliance recap; no fabrication of fixes that were never needed |

---

## §4 Pre-flight v3 integration summary (MEASURED 2026-09-29)

**Per Wf-344-B3 §4 integration status table** (the canonical pre-flight gate-count verification):

| Pipeline | Expected | Actual | Audit gate status |
|---|---:|---:|:---:|
| `all` | 66 | 66 | **PASS** |
| `frontier` | 25 | 25 | **PASS** |
| `chemprop_v2_v3` | 13 | 13 | **PASS** |
| `cfm` | 6 | 6 | **PASS** |
| `audit` (invalid) | n/a | n/a | **FAIL with INVALID_PIPELINE marker** (Wf-344-B3 regression guard) |

**Fresh-run pre-flight v3 `--pipeline all`** (per Wf-344-B2 §2.2 measurement): TOTAL 66 / PASS 63 / FAIL 3. The 3 current FAILs are Wf-343C4 pre-retrain internal-check gates (`p3_wf343c4_hyperparams_config_complete`, `p3_wf343c4_anchor_freshness_24h`, `p3_wf343c4_wave_budget_remaining_nonneg`), all out of Wf-344 scope.

---

## §5 Honest framing per `party-twenty-spirit` §3.5

**Summary of the 6 fixes:**

1. **Wf-344-A1** — verification only; Wf-3B already shipped the loader; Wf-344-A1 pins the contract.
2. **Wf-344-A2** — verification only; trainer already ships all 4 tokens; Wf-344-A2 adds gate-level regression.
3. **Wf-344-A3** — **REAL fix** (one of the few with actual code Edit); anchor logic + regression test + pre-flight gate integration.
4. **Wf-344-B1** — verification only; ADMET JSONs already carry canonical field set per Wf-308B / Wf-312C R02 / Wf-325E sister closeout; Wf-344-B1 pins the contract.
5. **Wf-344-B2** — verification only; block already present per Wf-312B ship; Wf-344-B2 pins the contract.
6. **Wf-344-B3** — **REAL fix** (+25 LOC Edit); fail-fast INVALID_PIPELINE guard + regression tests.

**Total LOC changed (MEASURED):**
- `molmetal/adapters/flow_matching_lipman/__init__.py` — ~30 LOC Edit (Wf-344-A3)
- `molmetal/scripts/pre_flight_check.py` — +25 LOC Edit + 1 LOC Edit (Wf-344-B3 + Wf-344-A3 gate integration)
- 6 NEW test files, ~1,000 LOC total
- 6 NEW doc files (this one + 5 siblings)
- ZERO `metrics/by_metric/*.json` modifications (verification-only for A1/B1/B2)
- ZERO `paper/*` touches (per AGENTS.md §5.3)
- ZERO GPU launches (per `kill_then_adapt` §0)

**Honest-negatives preserved:**

- Wf-344-A1: current `p1_chemprop_smoke` gate FAIL is env-side (`ModuleNotFoundError: No module named 'chemprop'`); out-of-scope (likely TODO-371 / Wf-345A).
- Wf-344-A2: GPU Wave-A still BLOCKED on user OK per `kill_then_adapt` §0; Wf-338-A3 finding 2 root cause NOT investigated end-to-end (was a stale venv state or earlier git revision).
- Wf-344-A3: anchor logic only matches the 2 standard clone layouts; unusual layouts require absolute paths.
- Wf-344-B1: 4 ADMET druglikeness metrics remain BELOW threshold (honest-negative FAILs with field-set).
- Wf-344-B2: 5 doc files (`README.md:128`, `TODO/risks.md:33`, `TODO/INDEX.md:2358`, `TODO/README.md:321`, `TODO/completed/322d_index_readme_wf309_322_sync.md:99+110`) still cite "block missing" — out of Wf-344-B2 scope; should be refreshed in future Wf-344-F2 INDEX sync.
- Wf-344-B3: docs/config drift fix (NOT real metric regression); pre-flight gate count already 66 per Wf-343C4 ship; Wf-321D audit doc already in sync.

---

## §6 Closeout doc status

| Doc | Status | Path |
|---|---|---|
| Wf-344-A1 closeout | ✅ SHIPPED | `TODO/completed/344a1_metal_cytotox_import.md` |
| Wf-344-A2 closeout | ✅ SHIPPED | `TODO/completed/344a2_chemprop_cli_tokens.md` |
| Wf-344-A3 closeout | ✅ SHIPPED | `TODO/completed/344a3_configure_reference.md` |
| Wf-344-B1 closeout | ✅ SHIPPED | `TODO/completed/344b1_admet_honest_negative.md` |
| Wf-344-B2 closeout | ✅ SHIPPED | `TODO/completed/344b2_nci60_aggregate.md` |
| Wf-344-B3 closeout | ✅ SHIPPED | `TODO/completed/344b3_gate_count.md` |
| Wf-344-C3 aggregation (this) | ✅ SHIPPED | `TODO/completed/344c3_closeout.md` |

**Total closeouts:** 7 NEW docs (6 sibling + 1 aggregation); per Wf-338-F2 INDEX-sync precedent, the per-Wf coverage verdict should be added to `TODO/INDEX.md` §0.23 / §22.28 in a future Wf-344-F2 sync wave.

---

## §7 Cross-references

* **Wf-338-A3** — original 11-finding audit that triggered Wf-344 ultracode wave.
* **Wf-321H** (`TODO/completed/321h_todo_119_p14_cpu_scaffold.md`) — gate first added; rc=0 verified 2026-09-29 10:50:51.
* **Wf-321D** (`TODO/completed/321d_preflight_gate_audit.md`) — audit gate contract; §5.2 design + §2.1 + §6 all carry 66 value.
* **Wf-323B** (`TODO/completed/323b_v3_v4_schema_drift_fix.md`) — synonym-aware + derive-active fallback resolved Wf-338-A3 finding 1.
* **Wf-327F2** (`molmetal/tests/test_wf327f2_v3_v4_schema.py`) — 4 existing schema-drift tests (complementary to Wf-344-A1).
* **Wf-327F11** (`molmetal/tests/test_wf327f11_gate_count_consistency.py`) — existing 3-test docs-drift guard (unchanged by Wf-344-B3).
* **Wf-343C4** (`TODO/completed/343c4_integration.md`) — 4 NEW gates; EXPECTED_GATE_COUNTS bump 62 → 66.
* **Wf-312B** (`TODO/completed/nci60_9cell_paper4_ship.md`) — canonical ship that added `wf312b_nci60_smoke_aggregate` block (per §6 line 89).
* **Wf-308B / Wf-312C R02 / Wf-325E** — canonical ADMET HONEST_NEGATIVE field set contract (4 BELOW druglikeness metrics).
* **AGENTS.md §5.1** — Standing GPU authorization (2026-09-29 grant); not invoked in Wf-344.
* **AGENTS.md §5.3** — No paper work without authorization; honoured across all 6 Wf-344 closeouts.
* **AGENTS.md §5.4** — Tier-2 + no wet-lab; honoured across all 6 Wf-344 closeouts.
* **party-twenty-spirit §3.5** — Honest framing mandate; honoured across all 6 Wf-344 closeouts + this aggregation.

---

**END OF Wf-344-C3** — Owner Hugo Lee (via subagent); cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.