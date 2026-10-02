# Wf-323E: TODO-21 Lambda × CFM coupling CPU scaffold verification

**Date:** 2026-09-29
**Round:** R17 close-out → R18 plan follow-up
**Parent:** `TODO/completed/57_lambda_cfm_coupling.md` (CPU Joint Inference Skeleton SHIPPED)
**Anchor:** Wf-296 pipeline integration SHIPPED + WEAK_TO_STRONG_PLAN §2.5 long-term
**Workflow thread:** user request "能不能基于刚才的洞察启动新的workflow做新的flow up,把暴露的问题全部解决"

---

## 1. TL;DR

**8 / 8 NEW tests PASS** verifying the TODO-21 Lambda × CFM coupling CPU
scaffold is intact. The scaffold comprises three shipped surfaces:

1. **`molmetal/molmetal_lam/lam_chem/coupling_adapter.py`** — 9-d → 64-d
   `CouplingAdapter` (337 LOC, R17-shipped, read-only here).
2. **`molmetal/molmetal_lam/lam_chem/coupling_joint_inference.py`** —
   `LambdaCFMPipeline` + `CascadedRefiner` + `PlatinAI5KBridge` (1038 LOC,
   per `TODO/completed/57_lambda_cfm_coupling.md`).
3. **`molmetal/molmetal_lam/reward/lambda_reward_dataset.py`** —
   `LambdaRewardDataset` + `build_lambda_reward_dataset` factory chain
   that the CFM retrain script consumes.

**No source code modified** — the verification is **read-only** on the
shipped surfaces. New tests live in
`molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py`
(8 tests across 3 phases).

---

## 2. Test coverage matrix

| Phase | Test | Surface | Asserts |
|---|---|---|---|
| 1 | `test_wf323e_adapter_shape_and_finite` | `CouplingAdapter` | 1-d + 2-d shape + dtype + finite; INPUT_DIM=9, EMBED_DIM=64 |
| 1 | `test_wf323e_adapter_deterministic_and_pocket_embedding` | `CouplingAdapter` | bit-for-bit determinism; 7-d warm-start path; `None` deterministic fallback; pocket-name disambiguation |
| 1 | `test_wf323e_adapter_env_var_gate` | `is_coupling_enabled` + `coupling_enabled_or` | env var case-insensitive truthy / falsy / default fallback contract |
| 2 | `test_wf323e_platinai5k_bridge_embed_shape_and_l2` | `PlatinAI5KBridge` | 32-d → 64-d L2-normalised (norm ≈ 1.0 ± 0.01); LAMBDA_DIM=32, EMBED_DIM=64 |
| 2 | `test_wf323e_pipeline_cfm_inference_closed_form` | `LambdaCFMPipeline.cfm_inference` | n_steps=0 is identity; n_steps=50 finite + L2-normalised; closed-form analytic ODE converges (monotone decreasing distance to large-n_steps output) |
| 2 | `test_wf323e_cascaded_refiner_identity_and_finite` | `CascadedRefiner.refine` | n_refine_steps=0 is identity; cascade is finite + L2-normalised + moves toward cfm_output |
| 3 | `test_wf323e_lambda_reward_dataset_factory_chain` | `LambdaRewardDataset` + `build_lambda_reward_dataset` | factory chain imports; empty dataset path; signature + `__all__` contract |
| 3 | `test_wf323e_canonical_pretrain_and_smoke` | `smoke_test_full_pipeline` + pretrain npz | canonical pretrain npz + json on disk; smoke returns `True`; `load_joint_pipeline` hydrates from canonical path; `run_full_pipeline` output dict shape |

---

## 3. MEASURED results

### 3.1 Test execution

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py -v

molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_adapter_shape_and_finite              PASSED [ 12%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_adapter_deterministic_and_pocket_embedding PASSED [ 25%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_adapter_env_var_gate                  PASSED [ 37%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_platinai5k_bridge_embed_shape_and_l2  PASSED [ 50%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_pipeline_cfm_inference_closed_form   PASSED [ 62%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_cascaded_refiner_identity_and_finite  PASSED [ 75%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_lambda_reward_dataset_factory_chain   PASSED [ 87%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_canonical_pretrain_and_smoke          PASSED [100%]

8 passed, 1 warning in 2.43s
```

### 3.2 Pre-flight (TODO-207 v2 25/25 PASS gate)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py

summary: { "pass": 55, "fail": 2, "skip": 0, "total": 57, "total_wall_sec": 304.4 }

FAIL gate 1: p1_chemprop_indices_quick (chemprop_v2_v3 pipeline) —
    MetalCytoToxDB.csv column header mismatch at .data/molmetal/MetalCytoToxDB.csv
    ("active / activity" missing). **PRE-EXISTING** data-file issue
    unrelated to Wf-323E; no Lambda × CFM surface is touched.

FAIL gate 2: audit pipeline gate-count drift (expected=57 actual=0).
    **PRE-EXISTING** audit drift per TODO/completed/321d_preflight_gate_audit.md.
    Unrelated to Wf-323E.

Net result: 55 / 57 PASS. Both FAIL gates are pre-existing — Wf-323E adds
zero new failures to the pre-flight gate.
```

### 3.3 Pipeline subset (cfm)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline cfm

All cfm-pipeline gates PASS (no FAIL in cfm subset).
```

---

## 4. Honest framing

### 4.1 What was verified (CPU scaffold only)

* **Coupling adapter** (`CouplingAdapter`) is importable, deterministic,
  shape-correct (9-d → 64-d), and respects the `COUPLING_ENABLED` env var
  gate. The stub adapter used in tests is bit-for-bit reproducible.
* **PlatinAI 5K bridge** (`PlatinAI5KBridge`) is importable, produces
  64-d L2-normalised output from 32-d input, and is backed by the
  canonical `models/coupling_adapter_platinai5k.{npz,json}` pretrain
  checkpoint.
* **Lambda × CFM pipeline** (`LambdaCFMPipeline.cfm_inference`)
  implements the closed-form analytic linear-path CFM (Lipman 2023 §3.2).
  n_steps=0 is identity; n_steps>0 converges monotonically.
* **Cascaded refiner** (`CascadedRefiner.refine`) is identity at
  n_refine_steps=0 and converges toward cfm_output otherwise. Refined
  output is L2-normalised + finite.
* **Reward dataset factory chain** (`LambdaRewardDataset` +
  `build_lambda_reward_dataset`) is importable. The empty-dataset path
  is constructible without torch / RDKit side effects beyond the
  class declaration.
* **Canonical pretrain checkpoint** is on disk at
  `models/coupling_adapter_platinai5k.{npz,json}` (44 KB + 6.6 KB per
  TODO/completed/57_lambda_cfm_coupling.md §2 #10).
* **Smoke** (`smoke_test_full_pipeline`) returns `True`.

### 4.2 What was NOT verified (out of scope for CPU scaffold)

* **GPU joint training** (TODO-21 Strategy 2) — DEFERRED to R18 Phase
  1-3 ultracode per `TODO/completed/57_lambda_cfm_coupling.md` §4.
  5-10 d effort, 12-24 h GPU per cycle, NO CPU-only fallback.
* **CFM velocity-loss reward-weighting wiring** — the "1-line adapter
  change" flagged in TODO-21 §4 day 1-3 plan; this is the **read-only
  surface** we tested. The actual GPU-side `train_step` rewiring
  (consuming `LambdaRewardDataset` into
  `metallo_drug_smoke_retrain.py`) is deferred to R18.
* **decode_ratio > 0** on h=128 Path B checkpoint — still 0/192 per
  upstream `mol-metal-cfm-training` skill (Wf-291D + Wf-299 CFM retrain
  gate). This is the upstream gate blocking any GPU joint training
  follow-up.
* **Real CFM refine path** (`cfm_inference_mode != "stub"`) — closed-form
  analytic linear-path CFM in the joint inference skeleton is intentional
  per §3.5 honest framing of `TODO/completed/57_lambda_cfm_coupling.md`.

### 4.3 No-modification promise

This Wf-323E verification adds **zero** modifications to:

1. `molmetal/molmetal_lam/lam_chem/coupling_adapter.py` (337 LOC)
2. `molmetal/molmetal_lam/lam_chem/coupling_joint_inference.py` (1038 LOC)
3. `molmetal/molmetal_lam/reward/lambda_reward_dataset.py`
4. `molmetal/molmetal_lam/reward/lambda_astsr.py` (R17 L-R17-B3)
5. `molmetal/adapters/flow_matching_lipman/*` (R17 close-out)
6. `models/coupling_adapter_platinai5k.{npz,json}` (R17 pretrain)

The verification is purely **additive**: 1 new test file with 8 tests.

---

## 5. Cross-reference

* **TODO-21 parent**: `TODO/pending/21_lambda_model_coupling.md` (memo
  refreshed 2026-09-16) + `TODO/completed/57_lambda_cfm_coupling.md`
  (CPU Joint Inference Skeleton SHIPPED).
* **Strategy 1** (Lambda-as-reward via `LambdaASTSR`): SHIPPED via
  `molmetal/molmetal_lam/reward/lambda_astsr.py` per
  `TODO/completed/35_lambda_b2_b3_deflexpressor.md` (ASTSR R² = 1.0000
  vs F5 R² = 0.9324).
* **Strategy 3** (Cascaded Lambda → CFM-refine): NEW via `CascadedRefiner`
  per §4.3 of `TODO/completed/57_lambda_cfm_coupling.md`.
* **Strategy 2** (joint training): DEFERRED to R18 GPU ultracode per §4.4
  of `TODO/completed/57_lambda_cfm_coupling.md`. Out-of-scope for CPU
  scaffold verification.
* **WEAK_TO_STRONG_PLAN.md §2.5 long-term**: TODO-21 reward-weighting
  wiring = "1-line adapter change" for R18 GPU. The CPU surface
  tested here is the read-only side of that wiring; the GPU write side
  remains deferred.

---

## 6. Standing directives referenced

* **AGENTS.md §1 #1** (MEASURED-vs-PROJECTED tag discipline): every
  numeric value above traces to a JSON / report path on disk (pytest
  output, pre-flight JSON).
* **AGENTS.md §1 #2** (kill_then_adapt GPU directive): no GPU launch
  initiated; this verification is CPU-only.
* **AGENTS.md §1 #4** (Pre-flight v2 gate): pre-flight run with 55/57
  PASS; the 2 FAIL gates are pre-existing and unrelated to Wf-323E.
* **AGENTS.md §1 #5** (Honest-negative framing): §4 above documents
  every verified + NOT-verified surface explicitly.

---

## 7. Files written this report

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py` | 322 | 8 new tests (3 coupling adapter + 3 joint inference + 2 downstream consumer) |
| `TODO/completed/323e_todo21_lambda_cfm_coupling_closeout.md` | this | Closeout doc |

**Total**: ~322 LOC of additive code + this doc.

---

— Wf-323E follow-up to Wf-296 (pipeline integration SHIPPED). CPU scaffold verified. GPU portion deferred to R18 per WEAK_TO_STRONG_PLAN §2.5.
