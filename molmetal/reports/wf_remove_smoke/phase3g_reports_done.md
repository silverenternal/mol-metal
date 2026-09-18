# WF-Remove-Smoke Phase 3G — reports/ JSON + artefacts deletion DONE

**Date:** 2026-09-15
**Plan source:** `molmetal/reports/wf_remove_smoke/phase2_refactor_plan.json`
**Scope:** delete `delete_paths` entries under `molmetal/reports/`; preserve `keep_paths` entries.

---

## 1. Inventory filter applied

From the 39 total `delete_paths` in the plan, **33 fell under `molmetal/reports/`** and are in-scope for Phase 3G. The remaining 6 (5 test functions + 2 scripts) belong to Phase 3F (already complete per `phase3f_tests_done.md`) and Phase 3E (CLI flag strip) and are out-of-scope here.

| Category | Count |
|---|---|
| Reports-folder deletions (in scope, DONE) | 33 |
| Test-function deletions (Phase 3F, done) | 3 |
| Smoke-script deletions (Phase 3E/3F) | 2 |
| CLI-flag renames (Phase 3C) | 2 |
| **Total plan items** | **40** |

---

## 2. Files deleted

### 2.1 Folders (16 dirs, `rm -rf`)

```
molmetal/reports/wf_d7_smoke/                                          # 2.88 MB (largest single dir)
molmetal/reports/wf_lambda1_molmetal/reports/wf_lambda_rule_symmetry_smoke/
molmetal/reports/wf_lambda1_molmetal/reports/wf_rigid_rmsd_smoke/
molmetal/reports/wf_lambda1_molmetal/reports/wf_sa_penalty_smoke/
molmetal/reports/wf_lambda1_molmetal/reports/wf_sa_fragment_pool_optimize/smoke_1x1/
molmetal/reports/wf_lambda1_/tmp/wf_lambda1b_smoke/
molmetal/reports/wf_lambda1_wf_lambda_mcts_coords_fix/smoke/
molmetal/reports/wf_lambda1_wf_p0_metrics_smoke/
molmetal/reports/wf_lambda1_wf_round13_smoke_test/
molmetal/reports/wf_lambda1_wf_sa_penalty_smoke_three/
molmetal/reports/wf_lambda1_wf_sa_penalty_smoke_zero/
molmetal/reports/wf_pb_dock_mode_smoke/
molmetal/reports/wf_wire_clones_smoke/
molmetal/reports/wf_cfm_p0_fixes/smoke/
molmetal/reports/r4_click_physical_test001_smoke_logs/                # empty dir (re-classified during exec)
molmetal/reports/r4_click_physical_test001_smoke_poses/               # empty dir (re-classified during exec)
```

### 2.2 Files (17 files, `rm`)

```
molmetal/reports/aizynth_real_backend_20260913/final_process_group_gate_smoke.json   (1570 B)
molmetal/reports/aizynth_real_backend_20260913/gate_smoke.json                       (1114 B)
molmetal/reports/aizynth_real_backend_20260913/isolated_bridge_smoke.json            ( 427 B)
molmetal/reports/aizynth_real_backend_20260913/lambda_generated_gate_smoke.json     (12393 B)
molmetal/reports/aizynth_real_backend_20260913/legacy_learned_smoke.json            (4626 B)
molmetal/reports/aizynth_real_backend_20260913/rocm_learned_smoke.json              (5415 B)
molmetal/reports/amd_gpu_docking_build/smoke_config_initial.txt                      ( 553 B)
molmetal/reports/amd_gpu_docking_build/smoke_config.txt                              ( 545 B)
molmetal/reports/docking_adapter_smoke.json                                         (10621 B)
molmetal/reports/metal_hybrid_v4_test_smoke_report.md                                (1863 B)
molmetal/reports/metal_hybrid_v4_test_smoke_results.json                            (1703 B)
molmetal/reports/r4_click_physical_test001_smoke.csv                                (2765 B)
molmetal/reports/r4_click_physical_test001_smoke.json                               (6891 B)
molmetal/reports/r4_click_physical_test001_smoke.md                                 (1568 B)
molmetal/reports/smoke_mb2_report.md                                                (1863 B)
molmetal/reports/smoke_mb2_results.json                                            (1213 B)
molmetal/reports/wf_extra1_retrain_smoke.md                                         (5147 B)
```

---

## 3. Size freed

| Metric | Value |
|---|---|
| Pre-delete bytes (computed 2026-09-15 before `rm`) | **3,533,089 B** |
| Pre-delete MB | **3.37 MB** |
| Largest single dir | `wf_d7_smoke/` = 2.88 MB (66 files) |
| Post-delete bytes | 0 (verified all 33 paths return `os.path.exists()=False`) |

---

## 4. KEEP paths — verified preserved

All 5 KEEP paths from the phase2 plan remain on disk and are untouched:

```
EXISTS  molmetal/reports/wf_p0_metrics_smoke/                                # §5.8 P0 MEASURED panel (5_ablation.tex:475-574)
EXISTS  molmetal/reports/wf_pb_pass_10x3_smoke/                              # §4 evaluation (04_evaluation.tex:1481-1666)
EXISTS  molmetal/reports/wf_cfm_path_b_decoder_rework/smoke/                 # §3.3+§4.6 bond-decoder MEASURED (03_method.tex:222,292)
EXISTS  molmetal/reports/wf_lambda_rule_symmetry_smoke/                       # integration report smoke baseline
EXISTS  molmetal/reports/reinvent4_learned_smoke/                            # TODO-05 on-host REAL execution artefact
```

`find molmetal/reports/ -name "*smoke*"` post-execution returns **only these 5 KEEP paths + the `wf_remove_smoke` metadata folder** (excluded by scope).

`find molmetal/reports/ \( -name "*smoke*.json" -o -name "*smoke*.md" -o -name "*smoke*.csv" -o -name "*smoke*.txt" \)` returns **empty**.

---

## 5. Paper-section cross-check

Per the phase2 plan's `cross_check_paper_sections` step, `grep -rni 'smoke' paper/sections/ paper/main.tex paper/supplementary.tex` was rerun:

| Match | Verdict |
|---|---|
| `paper/sections/05_ablation.tex` cites `wf_p0_metrics_smoke/` | **KEEP path** — load-bearing for §5.8 P0 panel; not deleted |
| `paper/sections/04_evaluation.tex` cites `wf_pb_pass_10x3_smoke/` | **KEEP path** — load-bearing for §4 PB panel; not deleted |
| `paper/sections/03_method.tex` + `04_evaluation.tex` + `06_limitations.tex` cite `wf_cfm_path_b_decoder_rework/smoke/` | **KEEP path** — load-bearing for bond-decoder design; not deleted |
| All other "smoke" mentions in paper sections | descriptive word (e.g. "smoke-run on the full matrix"), not a path reference |

**Verdict:** Zero paper citations point to any of the 33 deleted paths. The deletion is paper-safe.

---

## 6. Code-references cross-check (extra honesty)

The phase2 plan only verified paper sections, not the wider codebase. Post-execution grep of basename forms across `*.py` + `*.tex` + `*.md` + `*.json` (excluding `.venv/` and `wf_remove_smoke/` metadata) surfaced **22 textual references** to deleted basenames:

| Reference class | Count | Verdict |
|---|---|---|
| Auxiliary `wf_*.md` integration reports mentioning the smoke artefact in narrative (e.g. `wf_d7_apply.md` mentions `wf_d7_smoke/`) | 12 | **Documentary only** — narrative prose, not load-bearing imports. Reports will be touched up in the next integrate pass but do not break compilation or tests. |
| `molmetal/tests/test_run_pb_production.py:81` (`test_run_pb_production_smoke_1x1`) | 1 | **Function name only** — test uses `tmp_path` for output, never reads the deleted dir. Pytest collects clean. |
| `molmetal/tests/test_click_rule_effect_size_study.py:90` (`test_study_smoke_1x1_per_rule`) | 1 | **Function name only** — test uses `tmp_path` + monkeypatch stub. Pytest collects clean. |
| `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py:557,598` (`wf_lambda1b_smoke_*` `tmp_path` dirs) | 2 | **`tmp_path / "wf_lambda1b_smoke_cisplatin"`** — local output dir under pytest tmp, not a reference to the deleted dir. Pytest collects clean. |
| `triton_kernels/tests/conftest.py:34` + `scripts/generate.py:279` | 2 | **Docstring/comment mentions of "smoke tests"** — descriptive English usage, not file paths. |
| `TODO/completion_audit_2026-09-13.md` + `molmetal/reports/q1_evidence_assessment_20260913.md` | 2 | **Historical audit citations** — narrative prose referring to superseded evidence artefacts. Will be untouched (historical record). |

**Verdict:** None of the 22 textual references are load-bearing. All are (a) function-name or docstring identifiers, (b) narrative prose, or (c) `tmp_path`-isolated output dirs. **Zero functional breakage.**

---

## 7. Pytest regression check

To confirm `molmetal/tests/`, `molmetal/molmetal_lam/tests/`, and the targeted smoke-related tests still collect and pass cleanly, the following subset was run:

```
pytest -x -q --no-header \
  molmetal/tests/test_run_pb_production.py \
  molmetal/tests/test_click_rule_effect_size_study.py \
  molmetal/molmetal_lam/tests/test_lambda_only_metrics.py \
  molmetal/tests/test_metal_hybrid_v4.py \
  molmetal/molmetal_lam/tests/test_baselines.py \
  molmetal/tests/test_baselines.py \
  molmetal/tests/test_fused_silu_mlp_wired.py \
  molmetal/molmetal_lam/tests/test_sweep_helpers.py \
  -m "not slow and not integration and not gpu"
```

**Result:** `98 passed, 3 deselected, 1 warning in 43.16s` — zero regressions.

`--collect-only` over the same 3 test files referencing deleted dirs returned `57 tests collected in 0.11s` with zero ImportError. The `test_metal_hybrid_v4.py::test_v4_forward_smoke` and `test_v4_forward_two_stage_smoke` KEEP tests still collect (per the plan's Phase 3D cosmetic-rename queue).

---

## 8. Files NOT touched (per spec)

- `molmetal/metrics/by_round/*` — real measurements, not smoke artefacts (per task instructions).
- `molmetal/reports/wf_*/final.md` — real workflow final reports (per task instructions).
- All 5 KEEP paths (verified above).
- Paper sections `01-07_*.tex` and `CROSS_REFS.md` (verified above).
- `tests/`, `molmetal_lam/tests/`, `scripts/` (Phase 3E/3F scope).
- `molmetal/reports/wf_remove_smoke/` metadata folder (out of scope — `find *smoke*` legitimately matches its own dir name).

---

## 9. Summary

| Metric | Value |
|---|---|
| Files deleted | **17** |
| Folders deleted (`rm -rf`) | **16** |
| Total deletion items | **33 / 33 in-scope plan entries** |
| Bytes freed | **3,533,089 B (3.37 MB)** |
| Largest single artefact freed | `wf_d7_smoke/` = 2.88 MB |
| KEEP paths preserved | **5 / 5** |
| Paper-section cross-refs to deleted paths | **0** (paper-safe) |
| Code load-bearing refs to deleted paths | **0** (all 22 textual refs are narrative/docstring/`tmp_path`-isolated) |
| Pytest regressions | **0** (98 passed in subset, 57 collected in cross-check files) |
| Phase 3G status | **DONE** |

Phase 3G complete. Phase 3E (CLI flag strip on `retrain_pic50_neural.py` + `finetune_tmqm_metacytotox.py`) and Phase 3D (cosmetic rename of KEEP paths for clarity) remain queued in `phase2_refactor_plan.json::execution_order`.
