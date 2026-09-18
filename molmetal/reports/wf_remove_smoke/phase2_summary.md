# Phase 2 Refactor Plan: Remove Smoke + Route to Real Evaluators

**Generated**: 2026-09-15
**Phase**: 2 of `ultracode: remove smoke + route to real evaluators`
**Sources**: 4 phase-1 inventories (tests, scripts, reports, flags) cross-checked against paper sections.

---

## 1. Scope Summary

After aggregating phase1a (15 smoke tests), phase1b (6 smoke scripts), phase1c (38 smoke-tagged reports/JSON), and phase1d (16 smoke-flag/comment strings across 2 main eval scripts), the refactor plan splits 71 candidate items into 3 lists:

- **DELETE (29 items)**: synthetic-data smoke scripts, mock-only subprocess tests, legacy Round-1/2 reports superseded by Round-12/13 final.md, build-time config dumps, CLI smoke flags.
- **ROUTE_TO_REAL (10 items)**: subprocess-mock tests for SOTA scoring adapters (DiffDock/FlowDock/BioLM/PoseBusters/AiZynth/Vina) plus the `--smoke` flag on `retrain_pic50_neural.py` and `smoke_no_dataset` mode string on `finetune_tmqm_metacytotox.py`. Each routed item specifies the real evaluator that replaces it.
- **KEEP (17 items)**: 8 pytest tests that call real evaluators on real (small-scale) inputs (RDKit MMFF, XGBoost on Ru subset, RandomForest on Ru subset, real forward passes for MetalHybridV4), 5 paper-load-bearing report folders (wf_p0_metrics_smoke, wf_pb_pass_10x3_smoke, wf_cfm_path_b_decoder_rework/smoke, wf_lambda_rule_symmetry_smoke, reinvent4_learned_smoke), and 4 cosmetic smoke strings inside otherwise-clean scripts that don't short-circuit real evaluators.

## 2. Cross-Check Against Paper

All 14 paper source files (sections 01-07 + CROSS_REFS + main.tex + supplementary.tex) were grepped for "smoke". Result:

- **5 KEEP_AS_TEST paths are paper-load-bearing** (see `cross_check_paper_sections.matches_in_paper_for_keep_paths` in phase2_refactor_plan.json).
- **0 DELETE candidates are referenced** by any paper section.
- **0 ROUTE candidates are referenced** by any paper section.

The 5 KEEP paths stay despite their "smoke" name because:
- `wf_p0_metrics_smoke/` provides the 9 P0 MEASURED values for paper §5.8 + §4.6.
- `wf_pb_pass_10x3_smoke/` provides the 30-cell PB panel for paper §4 evaluation.
- `wf_cfm_path_b_decoder_rework/smoke/` provides the 0/192 → 192/192 decode_ratio lift for paper §3.3 + §4.6.
- `wf_lambda_rule_symmetry_smoke/` is the smoke baseline cited by 2 wf_*/final.md integration reports for §3.4.
- `reinvent4_learned_smoke/` is the on-host real execution evidence for the r_reinvent4 channel wire-up.

## 3. Risk Assessment

**Overall risk: LOW.**

- **Test dependencies to smoke symbols**: 0 (grep `molmetal/tests/` for smoke imports returns only internal `quick_smoke()` fixtures on `molmetal.baselines.morgan_xgb` + `molmetal.baselines.dmpnn` which stay).
- **Paper section impact**: 0 (cross-check above).
- **Imports from smoke scripts**: 0 (no other module imports any of the 6 smoke scripts).
- **Output JSON consumed by other scripts**: 0 (smoke JSON are write-only).

**Blocked real evaluators** (not refactor-blocked, environment-blocked):
- DiffDock real: GPU outage + fair-esm missing + diffdock_models.zip unreachable.
- BioLM real: biomlm package not installed.
- CFM full retrain: GPU recovered but decode_ratio=0/192 → path (c) lambda-only stays as Round-12 default.

These blockers do NOT affect the Phase 3 refactor; they affect the new pytest tests that replace the mocks (which become CLI-plumbing parity tests rather than upstream-invocation tests).

## 4. Execution Order (5 phases)

| Phase | Name | Tasks | Blast Radius |
|-------|------|-------|--------------|
| **3A** | Add new pytest tests | Convert 3 verification-helper scripts → `test_vina_3engine_parity.py` + `test_aizynth_real_backend.py` + `test_gpu_vina_smoke.py`; replace test_poseb_backers_wire.py::test_pb_mode_dock_smoke with real PB; convert 3 subprocess-mock SOTA adapter tests into CLI-plumbing parity tests | low (additions only) |
| **3B** | Delete obsolete smoke items | Remove 3 mock-only tests + 6 scripts + 22 reports/JSON + wf_cfm_p0_fixes/smoke/ | medium (verify no imports) |
| **3C** | Strip --smoke flag | Delete `--smoke` flag in retrain_pic50_neural.py (lines 615, 636, 673, 793, 846); rename `smoke_no_dataset`/`smoke_1_epoch` mode strings in finetune_tmqm_metacytotox.py; clean comments in lambda_100pocket_sweep.py | low (no test depends) |
| **3D** | Cosmetic renames (optional) | Rename 8 KEEP_AS_TEST test functions + 5 report folders to drop misleading "smoke" labels | low (cosmetic) |
| **3E** | Verify | Run full pytest + re-compile paper + grep for stale smoke paths | low (verification) |

## 5. Honest Framing Caveats

1. **The 5 KEEP paths are NOT really smoke** despite their names. They are paper-grade MEASURED artefacts that happen to have "smoke" in their folder name from earlier rounds. Phase 3D cosmetic rename recommended.
2. **3 of the ROUTE items (DiffDock/FlowDock/BioLM) cannot truly call the real upstream** because of environment blockers (GPU outage, missing packages). They are converted to CLI-plumbing parity tests that assert the subprocess call would be made correctly — NOT that the upstream actually runs. This is honest: we are not silently swapping mocks for fakes.
3. **The new `test_vina_3engine_parity.py` is the only conversion that PRESERVES the original behaviour** — it ports the spawn-per-engine pattern from `verify_docking_adapter_smoke.py` into pytest. The other 2 verification-helper conversions (`aizynth_learned_smoke.py`, `smoke_vina_gpu_generated.py`) preserve bounded protocols + status assertions.
4. **The `wf_p0_metrics_smoke/` folder is single-source-of-truth for 9 paper MEASURED values**. Renaming it (Phase 3D) requires updating `paper/sections/CROSS_REFS.md:117,124` + `paper/sections/05_ablation.tex:475,476,542,572-574` + `molmetal/reports/wf_section05_p0.md:16,44,49,51,93,129,151-152` + `TODO/pending/23_weak_to_strong_plan.md:150` + `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py:921`. Cosmetic rename should be deferred until after Phase 3E verifies the paper still compiles.
5. **Phase 3D is OPTIONAL**. The "smoke" naming in KEEP paths is cosmetic and does not affect any downstream consumer. Skipping Phase 3D preserves all existing cross-references; doing it requires the grep+replace above.

## 6. What the Refactor Does NOT Touch

- `molmetal/baselines/morgan_xgb.py:quick_smoke()` and `molmetal/baselines/dmpnn.py:quick_smoke()` — these are internal pytest fixtures used by `test_baselines.py`, `test_dmpnn.py`, `test_dmpnn_attn.py`. Phase 1D explicitly noted: "Tests use quick_smoke() helpers ... DO need to stay (they train a tiny D-MPNN/XGB on 200 rows for fast pytest)."
- `lambda_100pocket_sweep.py:500` comment — already at "12-tile library for backward compat / smoke tests" wording. Optional cosmetic rewrite only.
- `wf_round13_100x3/`, `wf_pb_30x3_protein_clash/`, `wf_round12_lambda_pilot/`, `wf_path_b_gpu_retrain/`, `wf_round12_sota_subset/`, `wf_round12_sota_integrate/` — these are the canonical real-evaluator reports that REPLACE the smoke reports. They are NOT in the delete list.

## 7. Files Written

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_remove_smoke/phase2_refactor_plan.json` — full JSON plan with 29 DELETE + 10 ROUTE + 17 KEEP entries, cross-check table, 5-step execution order, risk assessment.
- This file (`phase2_summary.md`).

---

## 8. Phase 3 Next Steps (Not Phase 2 Scope)

1. Run `uv run pytest molmetal/tests/ -x --tb=short` to capture baseline (before any deletions).
2. Phase 3A: write 3 new pytest tests + 2 PB real-DB replacements. Confirm pytest pass.
3. Phase 3B: delete in order (tests → scripts → reports). Confirm pytest pass after each step.
4. Phase 3C: strip --smoke flag. Confirm pytest pass + retrain_pic50_neural.py still works on real D-MPNN.
5. Phase 3E: re-compile paper. Confirm 0 errors.
6. Phase 3D: optional cosmetic renames. Defer until after step 5.
