# Wf-329-W6: TODO-21 Lambda × CFM 1-line adapter change — CPU-safe wiring SHIPPED

**Date:** 2026-09-29
**Round:** R17 ultracode follow-up
**Parent:** Wf-323E (CPU scaffold SHIPPED) + `TODO/completed/57_lambda_cfm_coupling.md` §4 (R18 deferred)
**Anchor:** WEAK_TO_STRONG_PLAN §2.5 long-term "1-line adapter change"
**Workflow thread:** user request "基于刚才的洞察，可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"

---

## 1. TL;DR

**4 / 4 NEW tests PASS** verifying the TODO-21 Lambda × CFM 1-line
adapter change is wired into the CFM training entry script. The
adapter change is **CPU-safe scaffolding**: when
`--lambda-cfm-coupling` is OFF (default), the loss is preserved
bit-for-bit; when ON, the bond-CE + W₂ + closure-bias loss is
multiplied by `1 + w * mean_weight` from `LambdaRewardDataset`.

The actual GPU-side `train_step` rewiring (consuming
`LambdaRewardDataset` into a real retrain) remains deferred to R18
per Wf-323E §4.2.

---

## 2. Source changes (1-line adapter change + plumbing)

| File | Change | LOC |
|---|---|---:|
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | 3 CLI flags + dataset build + 1-line loss multiplier | +38 |

### 2.1 CLI flags added (parse_args, line ~1071)

```python
# Wf-329-W6 — TODO-21 Lambda × CFM reward-weighting wiring
# (1-line adapter change; CPU-safe scaffolding; GPU retrain remains
# deferred per Wf-323E §4.2).  Off by default for backward-compat.
p.add_argument("--lambda-cfm-coupling", action="store_true", default=False, ...)
p.add_argument("--lambda-cfm-coupling-weight", type=float, default=0.1, ...)
p.add_argument("--lambda-cfm-coupling-samples", type=int, default=200, ...)
```

### 2.2 Dataset build (main, line ~1099)

```python
# Wf-329-W6 — TODO-21 Lambda × CFM coupling scaffolding (CPU-only).
lambda_reward_dataset = None
if getattr(args, "lambda_cfm_coupling", False):
    try:
        from molmetal.molmetal_lam.reward.lambda_reward_dataset import (
            build_lambda_reward_dataset,
        )
        lambda_reward_dataset = build_lambda_reward_dataset(
            n_samples=int(getattr(args, "lambda_cfm_coupling_samples", 200)),
            pocket_id=args.pocket_id,
            seed=args.seed,
        )
        print(
            f"[Wf-329-W6] --lambda-cfm-coupling ENABLED: "
            f"LambdaRewardDataset len={len(lambda_reward_dataset)} "
            f"mean_weight={lambda_reward_dataset.mean_weight:.4f} "
            f"max_weight={lambda_reward_dataset.max_weight:.4f}",
            flush=True,
        )
    except Exception as exc:
        print(f"[Wf-329-W6][WARNING] build failed: {exc}; "
              f"reward-weighting term collapses to zero.", flush=True)
        lambda_reward_dataset = None
```

### 2.3 The 1-line adapter change (train_one_epoch loss line ~795)

```python
loss = bond_ce + wasserstein_weight * w2_val + 0.05 * cb_loss
# Wf-329-W6 — TODO-21 1-line adapter change:
if lambda_cfm_coupling_weight > 0.0 and lambda_reward_dataset is not None:
    _rw_mean = float(getattr(lambda_reward_dataset, "mean_weight", 0.0))
    loss = loss * (1.0 + lambda_cfm_coupling_weight * _rw_mean)
loss.backward()
```

When the dataset is empty / unavailable / coupling is off, the term
is exactly zero and the original loss is preserved bit-for-bit
(CPU-safe scaffolding per Wf-323E).

### 2.4 EpochMetrics extension (line ~349)

```python
# Wf-329-W6 — TODO-21 Lambda × CFM reward-weighting (1-line adapter
# change; CPU-safe scaffold; GPU retrain deferred per Wf-323E §4.2).
lambda_cfm_coupling: bool = False
lambda_cfm_coupling_weight: float = 0.0
lambda_reward_dataset_len: int = 0
```

---

## 3. MEASURED results

### 3.1 Test execution

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf329w6_lambda_cfm_coupling.py -v

molmetal/tests/test_wf329w6_lambda_cfm_coupling.py::test_wf329w6_lambda_cfm_coupling_flag_wired        PASSED [ 25%]
molmetal/tests/test_wf329w6_lambda_cfm_coupling.py::test_wf329w6_reward_weighting_wired_into_loss     PASSED [ 50%]
molmetal/tests/test_wf329w6_lambda_cfm_coupling.py::test_wf329w6_joint_inference_skeleton_callable    PASSED [ 75%]
molmetal/tests/test_wf329w6_lambda_cfm_coupling.py::test_wf329w6_downstream_consumer_integration     PASSED [100%]

4 passed, 1 warning in 5.28s
```

### 3.2 Pre-flight (TODO-207 v2 — `cfm` pipeline subset)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline cfm

summary: { "pass": 6, "fail": 0, "skip": 0, "total": 6, "total_wall_sec": 0.0005 }

All 6 cfm-pipeline gates PASS. Zero new failures introduced by Wf-329-W6.
```

---

## 4. Honest framing

### 4.1 What was verified (CPU-safe scaffolding)

* **CLI flags** `--lambda-cfm-coupling`, `--lambda-cfm-coupling-weight`,
  `--lambda-cfm-coupling-samples` are wired and appear in `--help`.
* **Default safety**: `--lambda-cfm-coupling` defaults to `False`,
  preserving the Wf-124 P2 baseline behavior bit-for-bit.
* **1-line loss multiplier** is correctly applied when both
  `lambda_cfm_coupling_weight > 0` AND `LambdaRewardDataset` is
  non-empty. The loss becomes `base_loss * (1 + w * mean_weight)`.
* **Empty-dataset fallback**: when the dataset is empty
  (`mean_weight = 0`), the loss is preserved.
* **LambdaCFMPipeline** from `coupling_joint_inference` is still
  importable + callable end-to-end (Wf-323E surface unchanged).
* **`LambdaRewardDataset`** factory chain returns a usable dataset
  with non-negative `mean_weight`, respecting the `COUPLING_ENABLED`
  env var gate.

### 4.2 What was NOT verified (out of scope per Wf-323E §4.2)

* **GPU joint training** (TODO-21 Strategy 2) — DEFERRED to R18
  Phase 1-3 ultracode per `TODO/completed/57_lambda_cfm_coupling.md`
  §4. 5-10 d effort, 12-24 h GPU per cycle, NO CPU-only fallback.
* **Real reward-weighting impact on the CFM velocity field** — this
  is the actual R18 GPU retrain. The Wf-329-W6 wiring is the
  read-side scaffolding that the GPU retrain will plug into.
* **decode_ratio > 0** on h=128 Path B checkpoint — still 0/192 per
  upstream `mol-metal-cfm-training` skill (Wf-291D + Wf-299 CFM
  retrain gate). This is the upstream gate blocking any GPU joint
  training follow-up.

### 4.3 No-modification promise (carry-over from Wf-323E)

Wf-329-W6 modifies **zero** files in:

1. `molmetal/molmetal_lam/lam_chem/coupling_adapter.py` (337 LOC)
2. `molmetal/molmetal_lam/lam_chem/coupling_joint_inference.py` (1038 LOC)
3. `molmetal/molmetal_lam/reward/lambda_reward_dataset.py`
4. `molmetal/molmetal_lam/reward/lambda_astsr.py` (R17 L-R17-B3)
5. `molmetal/adapters/flow_matching_lipman/*` (R17 close-out)
6. `models/coupling_adapter_platinai5k.{npz,json}` (R17 pretrain)

The change is **additive**: 3 CLI flags + 1 dataset-build block +
1-line loss multiplier + 3 EpochMetrics fields + 1 new test file
(4 tests).

---

## 5. Cross-reference

* **TODO-21 parent**: `TODO/pending/21_lambda_model_coupling.md`
  + `TODO/completed/57_lambda_cfm_coupling.md` (CPU Joint
  Inference Skeleton SHIPPED).
* **Wf-323E parent**: `TODO/completed/323e_todo21_lambda_cfm_coupling_closeout.md`
  (CPU scaffold verified, 8/8 tests PASS).
* **WEAK_TO_STRONG_PLAN §2.5 long-term**: TODO-21 reward-weighting
  wiring = "1-line adapter change" for R18 GPU. Wf-329-W6 is the
  CPU-safe scaffolding on the read-side of that wiring.
* **AGENTS.md §1 #1** (MEASURED-vs-PROJECTED tag discipline): every
  numeric value above traces to pytest output + pre-flight JSON.
* **AGENTS.md §1 #2** (kill_then_adapt GPU directive): no GPU
  launch initiated; this verification is CPU-only.
* **AGENTS.md §1 #4** (Pre-flight v2 gate): cfm-pipeline 6/6 PASS.
* **AGENTS.md §1 #5** (Honest-negative framing): §4 above documents
  every verified + NOT-verified surface explicitly.

---

## 6. Files written this report

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | +38 | 3 CLI flags + dataset build + 1-line loss multiplier + 3 EpochMetrics fields |
| `molmetal/tests/test_wf329w6_lambda_cfm_coupling.py` | ~280 | 4 new tests (CLI flag + reward-weighting + joint inference + downstream consumer) |
| `TODO/completed/329w6_todo21_lambda_cfm_verified.md` | this | Closeout doc |

**Total**: ~318 LOC of additive code + this doc.

---

— Wf-329-W6 follow-up to Wf-323E + Wf-296. CPU-safe 1-line adapter change SHIPPED. GPU retrain remains deferred to R18 per WEAK_TO_STRONG_PLAN §2.5.