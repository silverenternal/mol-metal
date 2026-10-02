# Wf-331-R6: TODO-21 Lambda × CFM coupling — re-verification (SKIP + VERIFIED)

**Date:** 2026-09-29
**Round:** R17 ultracode follow-up (no-op confirmation)
**Parent:** Wf-323E (CPU scaffold SHIPPED) + Wf-329-W6 (1-line adapter change SHIPPED)
**Anchor:** `TODO/completed/329w6_todo21_lambda_cfm_verified.md` + WEAK_TO_STRONG_PLAN §2.5
**Workflow thread:** Wf-331-R6 reconciliation pass — confirms prior ship is intact.

---

## 1. TL;DR

**12 / 12 re-verified tests PASS** + **cfm pre-flight 6 / 6 PASS / 0 FAIL**.

Wf-323E + Wf-329-W6 already shipped the TODO-21 Lambda × CFM coupling
wiring. Wf-331-R6 was a reconciliation re-verification; **NO new code
changes were required**. The "1-line adapter change" the task brief
asked for is already in
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py` at lines 809-811.

**Status: VERIFIED + SKIP** (per Wf-331-R6 brief: "If both already done:
confirm + skip; otherwise execute.")

---

## 2. What Wf-323E + Wf-329-W6 already shipped (read-only confirmation)

### 2.1 Wf-323E CPU scaffold (8 tests)

* `molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py`
  (322 LOC, 8 tests, all PASS).
* Surfaces verified:
  - `CouplingAdapter` (9-d → 64-d) shape, determinism, `COUPLING_ENABLED`
    env var gate
  - `PlatinAI5KBridge` 32-d → 64-d L2-normalised
  - `LambdaCFMPipeline.cfm_inference` closed-form analytic linear-path
    CFM (Lipman 2023 §3.2)
  - `CascadedRefiner.refine` identity-at-0 + finite + L2-normalised
  - `LambdaRewardDataset` factory chain
  - canonical `models/coupling_adapter_platinai5k.{npz,json}` pretrain
    on disk

### 2.2 Wf-329-W6 1-line adapter change (4 tests)

* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (+38 LOC):
  - **CLI flags added** (`parse_args`, lines 1092, 1100, 1105):
    - `--lambda-cfm-coupling` (default `False`, action="store_true")
    - `--lambda-cfm-coupling-weight` (default `0.1`, float)
    - `--lambda-cfm-coupling-samples` (default `200`, int)
  - **Dataset build block** (main, lines 1142-1156) that wires
    `build_lambda_reward_dataset` into the CFM training entry.
  - **The 1-line adapter change** (`train_one_epoch`, lines 809-811):
    ```python
    if lambda_cfm_coupling_weight > 0.0 and lambda_reward_dataset is not None:
        _rw_mean = float(getattr(lambda_reward_dataset, "mean_weight", 0.0))
        loss = loss * (1.0 + lambda_cfm_coupling_weight * _rw_mean)
    ```
  - **EpochMetrics extension** (line 355-357): `lambda_cfm_coupling`,
    `lambda_cfm_coupling_weight`, `lambda_reward_dataset_len`.
* `molmetal/tests/test_wf329w6_lambda_cfm_coupling.py` (~280 LOC,
  4 tests, all PASS).

---

## 3. MEASURED results (re-verification 2026-09-29)

### 3.1 Test re-execution

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py \
             molmetal/tests/test_wf329w6_lambda_cfm_coupling.py -v

molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_adapter_shape_and_finite              PASSED [  8%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_adapter_deterministic_and_pocket_embedding PASSED [ 16%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_adapter_env_var_gate                  PASSED [ 25%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_platinai5k_bridge_embed_shape_and_l2  PASSED [ 33%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_pipeline_cfm_inference_closed_form   PASSED [ 41%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_cascaded_refiner_identity_and_finite  PASSED [ 50%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_lambda_reward_dataset_factory_chain   PASSED [ 58%]
molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py::test_wf323e_canonical_pretrain_and_smoke          PASSED [ 66%]
molmetal/tests/test_wf329w6_lambda_cfm_coupling.py::test_wf329w6_lambda_cfm_coupling_flag_wired        PASSED [ 75%]
molmetal/tests/test_wf329w6_lambda_cfm_coupling.py::test_wf329w6_reward_weighting_wired_into_loss     PASSED [ 83%]
molmetal/tests/test_wf329w6_lambda_cfm_coupling.py::test_wf329w6_joint_inference_skeleton_callable    PASSED [ 91%]
molmetal/tests/test_wf329w6_lambda_cfm_coupling.py::test_wf329w6_downstream_consumer_integration     PASSED [100%]

12 passed, 1 warning in 3.84s
```

**Net result: 12 / 12 PASS.** Both prior closeout test suites still pass
bit-for-bit on the current `master` branch tip.

### 3.2 Pre-flight (TODO-207 v2 — `cfm` pipeline subset)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline cfm

summary: { "pass": 6, "fail": 0, "total": 6, "total_wall_sec": 0.0005 }
```

**Net result: 6 / 6 PASS, 0 FAIL.** Zero new failures introduced by
Wf-331-R6 (no changes made).

---

## 4. Why Wf-331-R6 was a no-op confirmation

The Wf-331-R6 brief asked:

> 5. Apply the 1-line change: Wire reward_weighting into CFM velocity loss
> 6. Add --lambda-cfm-coupling CLI flag (default False for safety)
> 7. Write 2 NEW tests: molmetal/tests/test_wf331r6_lambda_cfm_coupling.py

Items 5, 6, 7 are **already shipped** by Wf-329-W6 (lines 1092-1105 for
the CLI flag, line 809-811 for the 1-line loss multiplier, and 4 new
tests in `test_wf329w6_lambda_cfm_coupling.py`). Per the brief's
"otherwise execute" branch we should NOT duplicate those surfaces; the
right action is to mark VERIFIED + SKIP and write this reconciliation
document.

The Wf-331-R6 task brief also pointed at
`molmetal/molmetal_lam/reward/coupling_adapter.py` — that path does not
exist; the coupling adapter is at
`molmetal/molmetal_lam/lam_chem/coupling_adapter.py` (per the Wf-323E
closeout). This is a path-name typo in the brief; the actual ship
surface is correct.

---

## 5. Honest framing

### 5.1 What Wf-331-R6 confirmed

* Wf-323E closeout is intact: 8/8 tests PASS, no source drift.
* Wf-329-W6 closeout is intact: 4/4 tests PASS, 1-line adapter change
  in `train_cfm_yuelbond_wasserstein.py:809-811` is live, 3 CLI flags
  are wired, `LambdaRewardDataset` plumbing is functional.
* cfm pre-flight 6/6 PASS, 0 FAIL — no regression from prior ships.

### 5.2 What remains deferred to R18 (out of scope for Wf-331-R6)

* **GPU joint training** (TODO-21 Strategy 2) — DEFERRED to R18
  Phase 1-3 ultracode per `TODO/completed/57_lambda_cfm_coupling.md`
  §4. 5-10 d effort, 12-24 h GPU per cycle, NO CPU-only fallback.
* **Real reward-weighting impact on the CFM velocity field** — this is
  the actual R18 GPU retrain. The Wf-329-W6 wiring is the read-side
  scaffolding that the GPU retrain will plug into.
* **decode_ratio > 0** on h=128 Path B checkpoint — still 0/192 per
  upstream `mol-metal-cfm-training` skill (Wf-291D + Wf-299 CFM
  retrain gate). This is the upstream gate blocking any GPU joint
  training follow-up.

### 5.3 No-modification promise

Wf-331-R6 modifies **zero** source files. The reconciliation is purely
**read-only**:

* No changes to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
* No changes to `molmetal/molmetal_lam/lam_chem/coupling_adapter.py`
* No changes to `molmetal/molmetal_lam/lam_chem/coupling_joint_inference.py`
* No changes to `molmetal/molmetal_lam/reward/lambda_reward_dataset.py`
* No new tests written (12 prior tests already cover all 4 axes:
  CLI flag, reward-weighting, joint inference skeleton, downstream
  consumer integration)

---

## 6. Standing directives referenced

* **AGENTS.md §1 #1** (MEASURED-vs-PROJECTED tag discipline): every
  numeric value above traces to pytest output + pre-flight JSON.
* **AGENTS.md §1 #2** (kill_then_adapt GPU directive): no GPU launch
  initiated; this reconciliation is CPU-only.
* **AGENTS.md §1 #4** (Pre-flight v2 gate): cfm-pipeline 6/6 PASS.
* **AGENTS.md §1 #5** (Honest-negative framing): §5 above documents
  every verified + NOT-verified surface explicitly.

---

## 7. Files written this report

| Path | LOC | Purpose |
|---|---:|---|
| `TODO/completed/331r6_lambda_cfm_verified.md` | this | Reconciliation doc (no-op VERIFIED + SKIP) |

**Total**: 0 LOC of source-code change + this doc.

---

— Wf-331-R6 reconciliation. Wf-323E + Wf-329-W6 ships intact. TODO-21 Lambda × CFM coupling is VERIFIED on CPU. GPU retrain remains deferred to R18 per WEAK_TO_STRONG_PLAN §2.5.
