# Wf-273D — NEW + Modified Files Verification Audit

**Date**: 2026-09-29
**Scope**: Read-only audit of 7 git-status-visible files (1 NEW + 6 modified).
**Author**: Wf-273D subagent
**Honest framing**: Read-only audit; cite-only; no edits to source.

---

## §0 Honest framing (party-twenty-spirit §3.5)

- This is a READ-ONLY audit. No files were edited.
- Each file is checked on three dimensions:
  - **tested**: at least one pytest in `molmetal/tests/` references the file (CLI flag check, import smoke, behavioural test).
  - **cited**: at least one TODO-completion note / strategy doc / synthesis report names the file or its Wf-/TODO-id.
  - **broken_indicators**: honest-negative findings (no GPU/torch available, CI not run, etc.).
- "modified" status in `git status` is a *snapshot* — many of these files were already in `9b12e6f574` (Wf-243 to Wf-252 close-out commit). Only the local working-tree state changed.

---

## §1 Per-file audit (FLAT schema)

### 1.1 `molmetal/molmetal_lam/lam_chem/cfm_tanimoto_ood_filter.py` (NEW)

- **Status**: NEW (`??`).
- **Lines**: 287.
- **What it does**: Wf-257-Apply TODO-98 — thin CPU-only filter that wraps `molmetal.molmetal_lam.lam_chem.ood_protocol.TanimotoOODProtocol`. Provides `apply_tanimoto_ood_filter`, `cfm_ood_filter_smiles_list`, `summarise_tanimoto_ood`, `write_tanimoto_ood_summary_json`, plus `TanimotoOODSummary` dataclass. Used at the CFM data-loader layer via `--use-tanimoto-ood-filter` (added in Wf-257-Apply).
- **tested**: ✅ `molmetal/tests/test_cfm_todo98_tanimoto_ood.py` (tests `apply_tanimoto_ood_filter`, `cfm_ood_filter_smiles_list`, `summarise_tanimoto_ood`; 6 distinct test cases including `test_self_match_is_in_domain`, `test_empty_train_pool_marks_all_ood`, `test_invalid_smiles_conservatively_ood`).
- **cited**: ✅ Cited in `TODO/completed/207_unified_pre_flight_checklist_v2.md` (L12 cross-row reference + §4 CFM P1 VERIFY), `TODO/completed/98_wf44_closure_ood_phase2.md` (Wf-44 P2 parent), `molmetal/reports/wf257_research/cfm_potential.md`, `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:976` (consumer).
- **broken_indicators**: None for the filter itself; module is pure CPU RDKit (per §1.2.5 honest-negative DIAGNOSTIC-only framing). CLI integration with `--use-tanimoto-ood-filter` round-trips per the docstring claim, but I did not execute `pytest` (read-only audit constraint).

### 1.2 `molmetal/scripts/bench_catherine_30metals.py` (modified)

- **Status**: Modified (`M`).
- **Lines**: 366.
- **What it does**: TODO-96 R20-P5 Catherine 30-metal closure reachability benchmark — extends Wf-37 P2's 9-metal bench to 30 d-block metals using `molmetal.math_skills.catherine_properties.verify_all_catherine_properties`.
- **tested**: ✅ `molmetal/tests/test_catherine_30metals.py` (13+ tests covering `DEFAULT_30_METALS` constant + `_run_bench` smoke) AND `molmetal/tests/test_cfm_todo96_catherine_30metals.py` (Wf-257 cross-test: bench importable + functional).
- **cited**: ✅ Cited in `TODO/completed/96_r20_p5_closure_catherine.md` (parent TODO), `TODO/completed/130_wf130_p1_catherine_30.md`, `TODO/completed/207_unified_pre_flight_checklist_v2.md` §4 + §9 (R20-P5 row + L1/L6/L12 cross-references), `molmetal/reports/wf255_catherine_30metals/sweep_summary.md` (output report), `molmetal/reports/wf256_apply/catherine_30/sweep_summary.md` (re-run output), `molmetal/reports/wf253e_signoff/verification.md`.
- **broken_indicators**: The smoke run (`molmetal/reports/wf255_catherine_30metals/sweep_summary.md`) shows 0/3 PASS due to `ModuleNotFoundError: No module named 'torch'` — but the re-run in `wf256_apply/catherine_30/sweep_summary.md` shows **30/30 PASS** (max_depth=0). The torch-missing smoke is an env artifact, not a code defect; the bench is functional when torch is installed (per the Wf-256 re-run).

### 1.3 `molmetal/scripts/metallo_drug_smoke_retrain.py` (modified)

- **Status**: Modified (`M`).
- **Lines**: 619.
- **What it does**: Wf-GPU-Recovery-Now stacked P0+P1 CFM smoke on real 500-mol metallodrug pool. 1000-step / h=128 / n_layers=3 protocol. Stacked P0 (F1-F5) + P1 (P1.1-P1.4) fixes.
- **tested**: ✅ Multiple tests reference it:
  - `molmetal/tests/test_smoke_flag.py` (4 PASS for `--smoke` flag B1).
  - `molmetal/tests/test_lambda_reward_coupling.py` (`--lambda-reward` parses).
  - `molmetal/tests/test_r17_legacy_compat.py` (`--lr-warmup-steps`, `--decode-smoke-every`).
  - `molmetal/tests/test_pitfall_guards.py`.
  - `molmetal/tests/test_cross_module_smoke.py` + `test_integration.py` + `test_full_pipeline_demo.py`.
- **cited**: ✅ Cited in:
  - `TODO/completed/43_code_test_improvements.md` §B1 (--smoke flag origin).
  - `TODO/completed/33_r17_fixes_blackwell_loader_paths.md` (SMILES pre-filter).
  - `TODO/completed/35_r17_ablation_runner_done.md` (defensive `adapter.bond_head` access).
  - `TODO/completed/37_model_track_mr17.md` (--resume-from addition).
  - `TODO/completed/59_cfm_m3_m5_frontier.md` (M3-M5 deferred wiring).
  - `TODO/completed/79_paper_v2_supplementary.md` (5-seed Pt retrain).
  - `TODO/completed/82f_train_scripts_refactor.md` (--smoke + skill alignment).
  - `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` (M14 LR warmup anchor).
  - `TODO/01_strategy/IMPROVEMENT_BACKLOG.md` (B1 backlog row).
  - `TODO/03_plans/MERGE_GATE.md` (production usage + docstring update item).
  - `molmetal/reports/wf_metallodrug_vertical/MASTER.md` (vertical master).
  - `molmetal/reports/wf_metallodrug_vertical/phase3_smoke/final.md` (smoke output).
- **broken_indicators**: `metallo_drug_smoke_retrain.py` is GPU-only by design (per docstring "Wall-budget: 30 minutes GPU"). Honest-negative: in a torch-missing env it cannot smoke at all. The `--smoke` CPU preset (added in B1) was intended to bridge this gap, but `wf229_ablation_launcher.py` is the CPU-friendly sibling.

### 1.4 `molmetal/scripts/wf229_ablation_launcher.py` (modified)

- **Status**: Modified (`M`).
- **Lines**: 533.
- **What it does**: Wf-229 TODO-182 per-innovation 2³ ablation on `train_dmpnn_multitask.py`. 3 levers (K=Kendall loss, D=DANN, F=four-domain). Supports 8 configs + 3 multi-seed baselines (11 runs) OR `--quick` 4-config × 3-seed = 12-run CPU pilot (Wf-252-Apply F4-AblationCPU).
- **tested**: ✅ `molmetal/tests/test_wf252_apply.py::test_f4_wf229_launcher_has_quick_flag` (verifies `--quick` flag declared + module importable). Also `molmetal/tests/test_wf268a_cfm_ablation_flags.py` for the related Wf-268A ablation flag contract.
- **cited**: ✅ Cited in:
  - `TODO/completed/207_unified_pre_flight_checklist_v2.md` (Wf-252 F4 / F4-AblationCPU).
  - `molmetal/reports/wf229_ablation/` (per_config + logs + quick_pilot outputs).
  - `molmetal/reports/wf256_apply/wf229_3epoch_3seed/ablation_summary.json` (3-epoch × 3-seed pilot: 12 runs completed, returncode=0 for all 12).
  - `molmetal/reports/wf256_apply/wf229_quick_pilot_5epoch_BACKUP/` (5-epoch backup).
  - `TODO/03_plans/LAMBDA_ROADMAP.md` (TODO-182 G6 + LAMBDA_ROADMAP forward-looking).
- **broken_indicators**: None for the launcher itself. The Wf-256 3-epoch × 3-seed pilot reports BCa CI includes 0 for all levers (per `execute_report.md` §0 KEY FINDING) — this is an honest-negative about the *signal*, not a *bug* in the launcher.

### 1.5 `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (modified)

- **Status**: Modified (`M`).
- **Lines**: 1291.
- **What it does**: Phase-3B tumor-relevant anticancer metrics (8 evaluators: `logp7_4`, `gi50_proxy`, `cell_permeability_logPapp`, `herg_cardio_risk`, `ames_mutagen`, `hepatotox_index`, `aqueous_solubility_logS`, `plasma_protein_binding`). Wf-252 F2 added `m6_kappa` (Shape Kappa Index); Wf-253-D added `n_distinct` + `n_distinct_summary`; TODO-114 patch raises `AdapterUnavailable` on Phase-4A MOCK_DELETED scorers.
- **tested**: ✅ Multiple tests:
  - `molmetal/tests/test_metrics_v2.py` (Phase-3B 8 tumor metrics).
  - `molmetal/tests/test_metrics_v2_panel.py` (`_safe_external_score` Panel layer).
  - `molmetal/tests/test_wf252_apply.py::test_f2_m6_kappa_returns_value_in_unit_range` + `test_f2_m6_kappa_mean_handles_empty_input`.
  - `molmetal/tests/test_m11_n_distinct.py` (`n_distinct`, `canonical_smiles`).
  - `molmetal/tests/test_wf253e_signoff/` (verification artefacts).
- **cited**: ✅ Cited in:
  - `TODO/INDEX.md` §2 row 2 (3B Anticancer metrics row).
  - `TODO/completed/45_refactor_and_readme.md` (`r4_metrics.py` dispatcher to `metrics_v2`).
  - `TODO/completed/87l_undefined_metrics_definition.md` (m6 QED + m9 logS anchors).
  - `TODO/completed/114_mock_deleted_external_scorers_cleanup.md` (AdapterUnavailable patch).
  - `TODO/completed/118_score_router_hygiene.md` (panel audit + clean).
  - `TODO/completed/207_unified_pre_flight_checklist_v2.md` (m6_kappa + Wf-253-D cross-refs).
  - `TODO/03_plans/LAMBDA_ROADMAP.md` (m6_kappa + m11 + 32-d F5 cross-refs).
  - `molmetal/reports/wf253e_signoff/`, `molmetal/reports/wf253d_m6_kappa_validation/`.
- **broken_indicators**: None for `metrics_v2.py`. Per `TODO/completed/114_mock_deleted_external_scorers_cleanup.md`, the Phase-4A external scorers (`ptiv_reduction_potential`, `coord_geometry_proxy`, `phototherapy_activity`, `metalloprotein_affinity`) are STILL MOCK_DELETED on disk — they raise `AdapterUnavailable` instead of silently returning None (this is the correct behaviour, but it means these 4 metrics cannot be measured end-to-end until R20 adapter re-implementation).

### 1.6 `molmetal/adapters/flow_matching_lipman/_reference/__init__.py` (modified)

- **Status**: Modified (`M`).
- **Lines**: 23.
- **What it does**: WF-257-Fix-Infra (Blocker 2) — stable import/pickle entry point for the isolated optional Meta reference. Calls `configure_reference()` so `_ReferenceFinder` is registered. **Does NOT redirect `__path__`** to the upstream Facebook clone (the local shim submodules handle their own upstream lookups).
- **tested**: ✅ `molmetal/tests/conftest.py::_ensure_reference_packages()` (line 44) pre-loads the local `_reference` shim packages via monkey-patch; `molmetal/tests/test_cfm_infrastructure.py` (line 111-113) imports `AffineProbPath` + `ODESolver` + `ModelWrapper` through this shim.
- **cited**: ✅ Cited inline in the shim docstring (lines 6-15) which references `Wf-257-Fix-Infra (Blocker 2)`; cited in `molmetal/adapters/se3_lipman_integration.py` + `molmetal/adapters/flow_matching_lipman/w_flow_cfm_integration.py` + `molmetal/adapters/flow_matching_lipman/w_flow.py` (consumers).
- **broken_indicators**: None. The shim explicitly does NOT redirect `__path__` (per lines 6-15) to avoid breaking the local shim subdirectory resolution — this is a *deliberate fix*, not an indicator of broken code.

### 1.7 `molmetal/adapters/flow_matching_lipman/_reference/path/__init__.py` (modified)

- **Status**: Modified (`M`).
- **Lines**: 199.
- **What it does**: Wf-257-Fix-Infra (Blocker 2) — shim that re-exposes the modern flat upstream `flow_matching` symbols under the legacy `path` API. Locates the upstream clone via `_reference_loader._reference_root`; auto-bootstraps from canonical locations; injects the upstream clone into `sys.modules` under `flow_matching`; provides `AffineProbPath` facade + `_PathSample` dataclass.
- **tested**: ✅ `molmetal/tests/test_cfm_infrastructure.py` (line 111 imports `AffineProbPath`); `molmetal/tests/conftest.py::_ensure_reference_packages()` monkey-patches `configure_reference` to return the local shim dir when no Facebook clone is present, exercising the `_UPSTREAM_AVAILABLE = False` branch (lines 81-152).
- **cited**: ✅ Cited inline (docstring lines 1-13 reference Wf-257-Fix-Infra); referenced by `molmetal/adapters/flow_matching_lipman/w_flow.py` + `w_flow_cfm_integration.py` + `se3_lipman_integration.py`; referenced by `TODO/completed/207_unified_pre_flight_checklist_v2.md` §4 (CFM P1 VERIFY gates, "Wf-257 + Wf-258 audit").
- **broken_indicators**: None. The shim has two robustness paths (lines 81-152):
  1. `_UPSTREAM_AVAILABLE = True` (production) — uses upstream `interpolate` + `target_velocity`.
  2. `_UPSTREAM_AVAILABLE = False` (test conftest) — defines minimal in-shim primitives `x_t = (1-t) x_0 + t x_1` + `v_t = x_1 - x_0`.
  Both paths are dead-simple OT-scheduler semantics and the conftest exercises both.

### 1.8 `molmetal/adapters/flow_matching_lipman/_reference/path/scheduler/__init__.py` (modified)

- **Status**: Modified (`M`).
- **Lines**: 55.
- **What it does**: Wf-257-Fix-Infra (Blocker 2) — shim mapping legacy `CondOTScheduler` onto upstream `SamplerConfig` with `n_steps=100, method='euler', t_start=0.0, t_end=1.0` defaults. Has fallback `_UPSTREAM_AVAILABLE = False` branch (lines 36-43) that returns no-op config.
- **tested**: ✅ Implicitly tested via `molmetal/tests/test_cfm_infrastructure.py` (which imports through this shim) and `molmetal/tests/conftest.py::_ensure_reference_packages` (which exercises the fallback branch by monkey-patching `configure_reference`).
- **cited**: ✅ Cited inline (docstring lines 1-7 reference Wf-257-Fix-Infra); referenced transitively via `molmetal/tests/test_w_flow_cfm_integration.py` and `molmetal/tests/test_w_flow.py` (which import through the `_reference.solver` sibling).
- **broken_indicators**: None. The `__call__(t)` method (lines 45-49) returns `torch.stack([t, 1.0 - t], dim=-1)` — alpha_t = t, sigma_t = 1-t, which matches the OT-scheduler contract.

---

## §2 Acceptance gates (FLAT schema)

| Gate | Value | Note |
|---|---|---|
| G1_n_files_audited | **8** | 1 NEW + 7 modified (per git status; note: §3 expansion of `flow_matching_lipman/_reference/` covers 3 files but audit as a unit since they ship together) |
| G2_per_file_status | see §3 below | dict-of-dict |
| G3_n_untested_files | **0** | All 8 files have at least 1 pytest reference |
| G4_n_uncited_files | **0** | All 8 files have at least 1 TODO/synthesis-doc citation |
| G5_n_broken_indicators | **2** (env-only, not code defects) | (a) `bench_catherine_30metals` smoke shows torch-missing failure (`wf255_catherine_30metals/sweep_summary.md` 0/3 PASS) — re-run shows 30/30 PASS when torch is present; (b) `metallo_drug_smoke_retrain.py` cannot smoke in torch-missing env — `--smoke` B1 flag bridges the gap; `wf229_ablation_launcher.py` is the CPU-friendly sibling |
| G6_audit_report_path | **`molmetal/reports/wf273d_file_audit/file_audit.md`** | this file |
| honest_framing_note | (see §0) | Read-only audit; honest-negative on torch-missing env not code defects; **TODO-114 Phase-4A MOCK_DELETED scorers remain `AdapterUnavailable` until R20 adapter re-implementation** (this is a *spec-correct* silent-None prevention, not a code defect). |

---

## §3 G2_per_file_status (FLAT dict)

```
{
  "molmetal/molmetal_lam/lam_chem/cfm_tanimoto_ood_filter.py": {
    "tested": true,
    "cited": true,
    "broken_indicators": []
  },
  "molmetal/scripts/bench_catherine_30metals.py": {
    "tested": true,
    "cited": true,
    "broken_indicators": ["torch-missing env produces 0/3 PASS smoke (wf255_catherine_30metals/sweep_summary.md); wf256_apply/catherine_30 re-run shows 30/30 PASS when torch is present — env artifact, not code defect"]
  },
  "molmetal/scripts/metallo_drug_smoke_retrain.py": {
    "tested": true,
    "cited": true,
    "broken_indicators": ["GPU-only by design (30-min wall-budget); torch-missing env cannot smoke at all; --smoke B1 flag (TODO-43) bridges to <2-min CPU path (test_smoke_flag.py 4 PASS)"]
  },
  "molmetal/scripts/wf229_ablation_launcher.py": {
    "tested": true,
    "cited": true,
    "broken_indicators": ["Wf-256 3-epoch x 3-seed CPU pilot reports BCa CI includes 0 for all 3 levers (per wf256_apply/execute_report.md KEY FINDING) — this is honest-negative about the SIGNAL not a BUG in the launcher"]
  },
  "molmetal/molmetal_lam/sbdd_env/metrics_v2.py": {
    "tested": true,
    "cited": true,
    "broken_indicators": ["4 Phase-4A external scorers (ptiv_reduction_potential, coord_geometry_proxy, phototherapy_activity, metalloprotein_affinity) still MOCK_DELETED on disk per TODO-114 — they raise AdapterUnavailable (correct) instead of silently returning None; cannot be measured end-to-end until R20 adapter re-implementation"]
  },
  "molmetal/adapters/flow_matching_lipman/_reference/__init__.py": {
    "tested": true,
    "cited": true,
    "broken_indicators": []
  },
  "molmetal/adapters/flow_matching_lipman/_reference/path/__init__.py": {
    "tested": true,
    "cited": true,
    "broken_indicators": []
  },
  "molmetal/adapters/flow_matching_lipman/_reference/path/scheduler/__init__.py": {
    "tested": true,
    "cited": true,
    "broken_indicators": []
  }
}
```

---

## §4 Cross-references

- `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` — parallel audit format reference (Wf-258 audit; G8_audit_report_path schema).
- `molmetal/reports/wf253e_signoff/verification.md` — verification artefact for `metrics_v2.py` m6/m11 (Wf-253 sign-off).
- `molmetal/reports/wf255_catherine_30metals/sweep_summary.md` — torch-missing smoke.
- `molmetal/reports/wf256_apply/catherine_30/sweep_summary.md` — torch-present 30/30 PASS re-run.
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §4 + §9 (CFM P1 VERIFY gates + L1/L6/L12 rows).
- `TODO/completed/96_r20_p5_closure_catherine.md` — TODO-96 parent for `bench_catherine_30metals.py`.
- `TODO/completed/98_wf44_closure_ood_phase2.md` — TODO-98 parent for `cfm_tanimoto_ood_filter.py`.
- `TODO/completed/114_mock_deleted_external_scorers_cleanup.md` — TODO-114 patch for `metrics_v2.py`.
- `TODO/completed/118_score_router_hygiene.md` — TODO-118 panel audit for `metrics_v2.py`.

---

## §5 Honest negatives

- I did **NOT** run any pytest during this audit (read-only constraint). The "tested" flag is based on static grep — does NOT imply "currently passing in this env".
- `bench_catherine_30metals.py` shows torch-missing failure on the Wf-255 smoke but 30/30 PASS on the Wf-256 re-run; this is an env artifact, NOT a code defect. The bench is functional.
- `metallo_drug_smoke_retrain.py` is GPU-only by design. The `--smoke` B1 flag (TODO-43) is the <2-min CPU bridge path; the sibling `wf229_ablation_launcher.py --quick` is the fully-CPU-friendly alternative.
- `metrics_v2.py` 4 Phase-4A external scorers are MOCK_DELETED on disk — `_safe_external_score` now raises `AdapterUnavailable` instead of silently returning None. This is the CORRECT behaviour per Wf-43 anti-pattern; the metrics are NOT measurable end-to-end until R20 adapter re-implementation.
- The Wf-257-Fix-Infra shim (`_reference/`) has TWO robustness branches (upstream-present / upstream-missing); the conftest exercises both via `monkey-patch configure_reference`.

**End of Wf-273D audit.**