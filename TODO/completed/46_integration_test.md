# TODO-46 — Integration Test + Smoke + Measurement Hook (I1 + I2 + I3)

**Date**: 2026-09-25 · **Status**: ✅ **SHIP** · **Source plan**: `TODO/01_strategy/IMPROVEMENT_BACKLOG.md` §I (extension)
**Implements**: I1 (5 cross-module integration tests) + I2 (4 subsystem smoke scripts + `all.sh`) + I3 (3 measurement hooks + helper). Each ships independently; the three gates share no production code paths.

Honest framing: this PR is *purely additive* — zero production code was modified. The integration tests, smoke scripts, and metric hooks all reach into the existing public API surface; the 7/7 pitfall guards and 14/14 evaluators stay bit-stable.

---

## 1. Executive summary

| Gate | What ships | Path | Δ LOC | Test count |
|---|---|---|---:|---|
| **I1** | 5 cross-module integration tests (Lipman↔Vina / MCTS↔SA+QED / LambdaASTSR↔CouplingAdapter / 84-rule MLE↔pt_click_compat / end-to-end smoke) | `molmetal/tests/test_integration.py` (NEW, ~520) + `molmetal/reports/wf_integration_tests/final.md` | +~720 | **+5** |
| **I2** | 4 subsystem smoke scripts (`smoke_cfm.sh` / `smoke_lambda.sh` / `smoke_evaluators.sh` / `smoke_pitfalls.sh`) + aggregate `all.sh` + 10-param pytest guard | `molmetal/scripts/smoke/{4 scripts + all.sh}` (NEW, ~475) + `molmetal/tests/test_smoke_scripts.py` (NEW, ~140) + `molmetal/reports/wf_smoke_scripts/final.md` | +~615 | **+10** |
| **I3** | 3 measurement hooks (`compute_vina_min` / `compute_multi_cell_auc` / `compute_decode_ratio`) + helper `resolve_pocket` + 11 pytest guards | `molmetal/molmetal_lam/sbdd_env/metric_hooks.py` (NEW, ~340) + `molmetal/tests/test_metric_hooks.py` (NEW, ~210) + `molmetal/reports/wf_metric_hooks/final.md` | +~550 | **+11** |
| **Net LOC delta** | | | **+~1885** (additive only) | **+26 PASS** |

**Verdict**: ✅ all 3 gates PASS. Canonical CI guard surface unchanged: `eval_all_evaluators.py` still reports **14/14 evaluators PASS**, `wf_pitfall_guard/check_all.py` still reports **7/7 PASS / 0 FAIL**. The integration-test gate runs in ≤ 90 s CPU, smoke gate in ≤ 60 s CPU, hook gate in ≤ 3 s CPU.

---

## 2. I1 — Integration tests (5 new tests, 5/5 PASS)

`molmetal/tests/test_integration.py` (NEW, 520 LOC) wraps two production modules per test through their public APIs and asserts on the **downstream observable** (finite score / registered-rule count / R² / ckpt-presence) — not on either module's internal state. This catches contract drift that the unit-test layer silently misses (mismatched dataclass fields, default-kw drift, import-path collisions, paired-module contract changes).

| # | Test | Modules wired | Pass criterion | Wall |
|---|---|---|---|---|
| 1 | `test_001_lipman_to_vina_e2e` | `LipmanFlowMatchingAdapter` + `VinaDockingAdapter` | 10 docked candidates, all `vina_score` finite, `∈ [-30, +5]` kcal/mol | ~17 s |
| 2 | `test_002_mcts_sa_qed_pipeline` | `MCTSProofSearch` + `sa_score` + QED scorer | 100 MCTS iters → 5 candidates, ≥3 with `SA<5` AND `QED>0.4` | ~4 s |
| 3 | `test_003_lambda_astsr_coupling_adapter` | `LambdaASTSR` + `CouplingAdapter` (Lambda×CFM bridge) | Fit on 50-row synthetic, `R² > 0.5`; 64-d `embed_features` finite for all rows | ~2 s |
| 4 | `test_004_mle_matrix_click_compat` | 84-rule MLE matrix + `pt_click_compat` | 84 rules registered in `metal-coord-matrix` scope; ≥3 of 5 Pt complexes yield valid (scaffold, click_rule) matrices | ~2 s |
| 5 | `test_005_end_to_end_smoke` | `metallo_drug_smoke_retrain.py` (whole pipeline as subprocess) | `--steps 100 --batch-size 4 --seed 0 --hidden-dim 64 --n-layers 2`; `ckpt.pt` saved with finite `velocity_field` tensor | ~53 s |
| **Total** | | | | **~71 s** |

**Run output** (`pytest molmetal/tests/test_integration.py -v`):
```
5 passed, 3 warnings in 70.99s (0:01:10)
```

**Honest framing**: `test_001` uses a curated 10-SMILES list as a proxy for the learned Lipman→Vina handoff (no trained checkpoint exists); `test_002` uses an identity reaction rule (chemistry realism covered by `test_proof_search_strengthened.py`); `test_001`/`test_005` require `vina` + `meeko` + `mk_prepare_receptor.py` to be on PATH (env.sh provides them). `test_004` deliberately uses the top-level `molmetal_lam` shim path so registration + introspection land in the same module instance (avoids Python's per-import module-identity quirk splitting `_REGISTRY`).

---

## 3. I2 — Subsystem smoke scripts (4 new scripts + `all.sh`)

`molmetal/scripts/smoke/` (NEW directory) holds 4 CPU-only smoke scripts + 1 aggregate entry point. Each prints a `<NAME>_SMOKE PASS|FAIL …` line on stdout and exits 0/1.

| Script | Subsystem | Pass criterion | Wall (CPU) |
|---|---|---|---|
| `smoke_cfm.sh` | `LipmanFlowMatchingAdapter` | 200 train steps + 4×4 decode probes → `decode_ratio == 1.0` (lenient atom-based contract) | 5–10 s |
| `smoke_lambda.sh` | Lambda MCTS pilot (10 pockets × 1 seed × 20 sims) | every cell reports `n_distinct ≥ 1`; total `≥ 1` | 10–20 s |
| `smoke_evaluators.sh` | 14-evaluator sanity (SA×3 / QED×2 / Vina×2 / Meeko / PoseBusters / Bio.PDB / OpenBabel / ProDy / cisplatin SA+QED×2) | parses trailing `SUMMARY: X/14 evaluators PASS` ⇒ X == 14 | 3–5 s |
| `smoke_pitfalls.sh` | 7 R12-R17 pitfall guards (P1 OOV vocab / P2 SFN shim / P3 84-rule MLE / P4 BF16 clamp / P5 SAFETY_MAX / P6 ref clone / P7 ModelWrapper kwonly-t) | parses `PASS=N FAIL=M SKIP=K` ⇒ N == 7 AND M == 0 | 0.5–2 s |
| `all.sh` | aggregate (runs all 4 sequentially, 180 s per-script timeout) | `ALL_SMOKE TOTAL_PASS=4/4` | <60 s |

**Pytest gate** (`molmetal/tests/test_smoke_scripts.py`, NEW, 140 LOC, 10/10 PASS):
- 1 × `test_smoke_directory_present`
- 4 × `test_smoke_script_exits_zero_with_pass_token[smoke_*.sh-<TOKEN>]`
- 4 × `test_smoke_script_completes_under_120s[smoke_*.sh]`
- 1 × `test_all_sh_runs_all_4_smokes_and_reports_4_of_4`

**Run output** (`pytest molmetal/tests/test_smoke_scripts.py -v`):
```
10 passed, 1 warning in 69.22s (0:01:09)
```

**Honest framing**: the CFM smoke uses the *lenient* atom-based decode contract (`coords` finite + ≥2 atoms) — the strict `bonds.shape[-1] > 0` contract is reserved for post-bond-head-training runs (would currently fail by design on a cold-start adapter). All scripts are `chmod +x` so the dispatch harness can invoke them directly as `bash <script>`.

---

## 4. I3 — Measurement hooks (3 new functions)

`molmetal/molmetal_lam/sbdd_env/metric_hooks.py` (NEW, 340 LOC) wraps 3 headline metrics behind a single import. Each hook is **self-contained**, **JSON-serializable** (round-trips through `json.dumps(default=str)`), **side-effect-safe** (writes `metrics/by_metric/<name>.json` via `.tmp` + rename), and **never raises** (failures surface as `{"error": …, "trace": …}` in the envelope).

| Hook | Signature | Returns |
|---|---|---|
| `compute_vina_min(smiles_list, pocket_id, n_samples=9, exhaustiveness=4, …)` |  | `{"hook": "vina_min", "best_score": -6.42, "scores": […], "pocket_id": …, "n_samples": 3, "n_docked": 3, "errors": [], "wall_clock_s": 12.34}` |
| `compute_multi_cell_auc(model, smiles_list, cell_lines, y_true=None, …)` |  | `{"hook": "multi_cell_auc", "n_smiles": 3, "n_cell_lines": 3, "per_cell_auc": {"MCF-7": 1.0, "A549": 0.5, "A2780": 1.0}, "model_class": "DMPNNMultiTaskModel", "wall_clock_s": 0.05}` |
| `compute_decode_ratio(checkpoint_path, n_samples=16, …)` |  | `{"hook": "decode_ratio", "checkpoint_path": "…", "n_samples": 16, "n_decoded": 16, "decode_ratio": 1.0, "checkpoint_loaded": True, "wall_clock_s": 3.21}` |
| `resolve_pocket(pocket_id, …)` | helper | `"synthetic:<name>"` / `/path/to/receptor.pdb` / plain PDB ID → unified `Pocket` |

**Pytest gate** (`molmetal/tests/test_metric_hooks.py`, NEW, 210 LOC, 11/11 PASS):

```
TestComputeVinaMin::test_vina_min_returns_serializable_dict           PASSED
TestComputeVinaMin::test_vina_min_best_score_is_float_or_nan          PASSED
TestComputeVinaMin::test_vina_min_writes_metric_file                  PASSED
TestComputeMultiCellAUC::test_multi_cell_auc_returns_serializable_dict PASSED
TestComputeMultiCellAUC::test_multi_cell_auc_per_cell_dict_is_float_only PASSED
TestComputeMultiCellAUC::test_multi_cell_auc_writes_metric_file       PASSED
TestComputeDecodeRatio::test_decode_ratio_returns_serializable_dict    PASSED
TestComputeDecodeRatio::test_decode_ratio_value_is_in_unit_interval   PASSED
TestComputeDecodeRatio::test_decode_ratio_writes_metric_file          PASSED
TestResolvePocket::test_synthetic_pocket_has_correct_id               PASSED
TestResolvePocket::test_unknown_pocket_falls_back_to_synthetic        PASSED
======================== 11 passed, 4 warnings in 2.58s ========================
```

**Honest framing**: `compute_vina_min` on synthetic pockets returns NaN if `mk_prepare_receptor.py` rejects the random receptor (real PDB IDs always work); `compute_decode_ratio` on CPU-only hosts catches the Triton kernel error and returns `decode_ratio=0.0` (use a CUDA host for real values). Hooks wrap inference only — callers must train/load their own model.

---

## 5. Per-phase pass/fail + test count delta

| Phase | Pass/Fail | New tests | New files | Total LOC |
|---|---|---:|---|---:|
| **I1** Integration tests | **PASS** 5/5 | +5 | `molmetal/tests/test_integration.py` + `molmetal/reports/wf_integration_tests/final.md` | +~720 |
| **I2** Smoke scripts | **PASS** 4/4 + 10/10 pytest | +10 | `molmetal/scripts/smoke/{4 scripts + all.sh}` + `molmetal/tests/test_smoke_scripts.py` + `molmetal/reports/wf_smoke_scripts/final.md` | +~615 |
| **I3** Measurement hooks | **PASS** 3/3 + 11/11 pytest | +11 | `molmetal/molmetal_lam/sbdd_env/metric_hooks.py` + `molmetal/tests/test_metric_hooks.py` + `molmetal/reports/wf_metric_hooks/final.md` | +~550 |
| **Aggregate** | **PASS** all 3 phases | **+26** | 8 new files + 3 reports | **+~1885** |

**Total wall-time** (CPU, sequential): ~71 s (I1) + ~70 s (I2) + ~3 s (I3) ≈ 2 min 24 s. Parallelisable per track to ~71 s wall.

**Pre-existing CI surface — unchanged**:
- `molmetal/scripts/eval_all_evaluators.py` → **14/14 evaluators PASS** (verified 2026-09-25)
- `molmetal/scripts/wf_pitfall_guard/check_all.py` → **7/7 PASS / 0 FAIL** (verified 2026-09-25)
- ~1604-pre-existing-passed fast unit suite → unchanged

---

## 6. Cross-reference to canonical CI gates

### 6.1 `molmetal/scripts/eval_all_evaluators.py` — still 14/14 PASS

The I2 `smoke_evaluators.sh` script invokes this exact file as a subprocess and parses the trailing `SUMMARY: X/14 evaluators PASS` line. On the R17 host (RTX PRO 6000 Blackwell, env.sh provisioned venv), it reports `14/14 evaluators PASS` — bit-stable since R17 close-out (per `molmetal/reports/wf_ci_guard/final.md` §4.1).

The 14 evaluators are: SA score × 3 (CCO / aspirin / taxol) + QED × 2 (CCO / ibuprofen) + Vina adapter instantiation + Vina Python binding import + Meeko import + PoseBusters adapter instantiation + Bio.PDB PDBParser + OpenBabel import + ProDy import + cisplatin SA + cisplatin QED (2 reference sanity). No regressions introduced by I1/I2/I3.

### 6.2 `molmetal/scripts/wf_pitfall_guard/check_all.py` — still 7/7 PASS

The I2 `smoke_pitfalls.sh` script invokes this exact file as a subprocess and parses the trailing `PASS=N FAIL=M SKIP=K` line. On the R17 host it reports `PASS=7 FAIL=0 SKIP=0 MISS=0` — bit-stable since R17 close-out.

The 7 R12-R17 guards are:

| Pitfall | Fix | Test ID |
|---|---|---|
| **P1** | OOV atoms → `vocab_mask` + 20-element default vocab | `test_p1_oov_atoms_*` |
| **P2** | SynFlowNet shim TypeError → per-rule try/except | `test_p2_synflownet_shim_*` |
| **P3** | MetalLigandExchange 84-rule matrix + cisplatin seed default | `test_p3_metal_ligand_exchange_*` |
| **P4** | BF16 `atom_logits` clamp ±50 + FP32 AMP fallback | `test_p4_bf16_amp_*` |
| **P5** | `r4_lambda_only_run` SAFETY_MAX ≥ 10000 floor | `test_p5_safety_max_*` |
| **P6** | Reference clone location auto-detect (parent vs clone root) | `test_p6_ref_clone_*` |
| **P7** | ModelWrapper keyword-only `t` fallback (`functools.partial` fix) | `test_p7_model_wrapper_*` |

No guard was touched by I1/I2/I3 — all 7 still PASS in 0.5–2 s on CPU.

---

## 7. Definition of Done (this PR)

1. ✅ `molmetal/tests/test_integration.py` exists with 5 tests; all 5 PASS in ≤ 90 s CPU.
2. ✅ `molmetal/scripts/smoke/{smoke_cfm,smoke_lambda,smoke_evaluators,smoke_pitfalls,all}.sh` exist (5 scripts, executable); `all.sh` reports `TOTAL_PASS=4/4`.
3. ✅ `molmetal/tests/test_smoke_scripts.py` exists with 10 guards; all 10 PASS.
4. ✅ `molmetal/molmetal_lam/sbdd_env/metric_hooks.py` exists with 3 hooks + 1 helper; each is JSON-serializable, side-effect-safe, never raises.
5. ✅ `molmetal/tests/test_metric_hooks.py` exists with 11 guards; all 11 PASS in 2.58 s CPU.
6. ✅ Canonical CI gates unchanged: `eval_all_evaluators.py` 14/14 PASS, `wf_pitfall_guard/check_all.py` 7/7 PASS.
7. ✅ Zero production-code modifications — purely additive.

---

## 8. References

- `TODO/01_strategy/IMPROVEMENT_BACKLOG.md` §I (extension: I1 + I2 + I3)
- `molmetal/reports/wf_integration_tests/final.md` — I1 detail + per-test design rationale
- `molmetal/reports/wf_smoke_scripts/final.md` — I2 detail + pytest gate output
- `molmetal/reports/wf_metric_hooks/final.md` — I3 detail + JSON-serialization contract
- `molmetal/reports/wf_ci_guard/final.md` §4.1 — `eval_all_evaluators.py` 14/14 PASS canonical gate
- `molmetal/scripts/wf_pitfall_guard/check_all.py` — 7/7 PASS canonical gate

---

End of report.