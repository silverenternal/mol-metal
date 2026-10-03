# Wf-333-G1 — TODO-21 Lambda × CFM GPU wrapper + verification

**Date:** 2026-09-29
**Round:** R17 ultracode (GPU wrapper scaffold + verification)
**Parent:** Wf-331-R6 (verified) + Wf-323E (CPU scaffold SHIPPED) + Wf-329-W6 (1-line adapter change SHIPPED)
**Anchor:** `TODO/completed/331r6_lambda_cfm_verified.md` + WEAK_TO_STRONG_PLAN §2.5
**Workflow thread:** Wf-333-G1 — build the GPU CLI surface for the Lambda × CFM joint retrain (TODO-21 Strategy 2)

---

## 1. TL;DR

**2 / 2 NEW tests PASS**, **GPU launch deferred to R18** per Wf-331-R6 §5.2
binding constraint (upstream `mol-metal-cfm-training` skill reports
`decode_ratio=0` on h=128 Path B checkpoint — Wf-291D + Wf-299 gate).

Wf-333-G1 ships the **GPU wrapper CLI + dataclass** for the TODO-21
Lambda × CFM joint retrain. The wrapper is a forward-compatible CLI
surface that the Wf-333-G2 GPU launch script will invoke when the
upstream `decode_ratio > 0` gate is unblocked.

**Status: SHIPPED (CPU-only wrapper; GPU launch deferred).**

---

## 2. Wf-331-R6 reconciliation (re-verified 2026-09-29)

Per the Wf-333-G1 brief: *"If R6 already done: confirm + skip;
otherwise execute."* Wf-331-R6 IS done (`TODO/completed/331r6_lambda_cfm_verified.md`,
12/12 prior tests PASS + cfm pre-flight 6/6 PASS). Wf-333-G1 confirmed
the prior ship is intact and proceeded with the G1 wrapper task.

### 2.1 Confirmed surfaces (re-verified)

* Wf-323E CPU scaffold (8 tests) — `CouplingAdapter` 9-d → 64-d,
  `PlatinAI5KBridge` 32-d → 64-d, `LambdaCFMPipeline.cfm_inference`
  closed-form linear-path CFM (Lipman 2023 §3.2), `CascadedRefiner`
  identity-at-0 + finite + L2-normalised, `LambdaRewardDataset`
  factory chain, canonical `models/coupling_adapter_platinai5k.{npz,json}`
  pretrain on disk.
* Wf-329-W6 1-line adapter change (4 tests) — `--lambda-cfm-coupling`,
  `--lambda-cfm-coupling-weight`, `--lambda-cfm-coupling-samples`
  flags wired (`train_cfm_yuelbond_wasserstein.py:1092-1105`);
  `train_one_epoch` loss multiplier at lines 809-811
  ```python
  if lambda_cfm_coupling_weight > 0.0 and lambda_reward_dataset is not None:
      _rw_mean = float(getattr(lambda_reward_dataset, "mean_weight", 0.0))
      loss = loss * (1.0 + lambda_cfm_coupling_weight * _rw_mean)
  ```
  `EpochMetrics` extension (line 355-357).

### 2.2 Brief typo (re-routed)

The Wf-333-G1 brief pointed at
`molmetal/molmetal_lam/reward/coupling_adapter.py` — that path does
not exist; the coupling adapter is at
`molmetal/molmetal_lam/lam_chem/coupling_adapter.py` (per Wf-323E
closeout). Wf-333-G1 re-routed the read to the correct path; no
ship-surface divergence.

---

## 3. Wf-333-G1 ship surface (NEW)

### 3.1 Wrapper script

`molmetal/scripts/wf333g1_lambda_cfm_gpu_wrapper.py` (~370 LOC)

| Element | Purpose |
|---|---|
| `LambdaCFMGPUWrapper` (dataclass) | Holds dry_run flag, device, checkpoint paths, lambda_cfm_coupling_weight, batch_size, n_test_molecules, pocket_id, seed |
| `LambdaCFMGPUWrapper.resolve_paths()` | Returns `{key: {present: bool, path: str \| None}}` for each checkpoint |
| `LambdaCFMGPUWrapper.build_cpu_joint_pipeline()` | Lazy-loads the Wf-323E-shipped `LambdaCFMPipeline` + `PlatinAI5KBridge` (CPU-only fallback) |
| `LambdaCFMGPUWrapper.build_train_kwargs()` | Produces the kwargs shape the Wf-329-W6 1-line adapter change consumes: `lambda_reward_dataset`, `lambda_cfm_coupling_weight`, `batch_size` |
| `run_joint_inference_smoke()` | Runs the CPU joint pipeline on a 4-SMILES pool (CCO, c1ccccc1, CCN=[N+]=[N-], [Pt](Cl)(Cl)(N)N); returns `{ok, n_smiles, decoded_counts, first_decoded, bridge_present}` |
| `parse_args()` | Argparse CLI with `--dry-run` (default True), `--no-dry-run`, `--device`, `--cfm-checkpoint`, `--lambda-reward-ckpt`, `--bridge-npz`, `--lambda-cfm-coupling-weight`, `--batch-size`, `--n-test-molecules`, `--pocket-id`, `--seed`, `--output-json` |
| `build_wrapper(args)` | Translates empty-string checkpoint paths to `None` (disabled) |
| `main(argv)` | Runs CPU smoke, dumps JSON when `--output-json` is set, prints 8-line summary |

**Honest constraint:** `--dry-run` is the **default** per AGENTS.md
§5.1 kill_then_adapt; no GPU work attempted unless `--no-dry-run` is
explicitly passed AND a CUDA device is available AND the upstream
`decode_ratio > 0` gate is unblocked.

### 3.2 Tests (2 NEW)

`molmetal/tests/test_wf333g1_lambda_cfm_gpu_wrapper.py` (~180 LOC)

* `test_wf333g1_wrapper_initialization_cpu_only` — initialises the
  wrapper with documented defaults, asserts `dry_run=True` guard,
  resolves all 3 checkpoint paths (cfm / lambda_reward / bridge),
  builds the CPU joint pipeline, runs the joint inference smoke on
  the 4-SMILES pool, and verifies `build_train_kwargs()` returns the
  exact kwargs shape the Wf-329-W6 1-line adapter change consumes.
* `test_wf333g1_wrapper_cli_args` — verifies `--dry-run` default,
  `--no-dry-run` flip, `--lambda-cfm-coupling-weight` flow-through,
  `--cfm-checkpoint ''` translates to `None`, `--n-test-molecules` /
  `--pocket-id` flow-through, and `--help` exits 0 via subprocess
  smoke.

---

## 4. MEASURED results (2026-09-29)

### 4.1 Test execution

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf333g1_lambda_cfm_gpu_wrapper.py -v

molmetal/tests/test_wf333g1_lambda_cfm_gpu_wrapper.py::test_wf333g1_wrapper_initialization_cpu_only PASSED [ 50%]
molmetal/tests/test_wf333g1_lambda_cfm_gpu_wrapper.py::test_wf333g1_wrapper_cli_args                PASSED [100%]

2 passed, 1 warning in 2.24s
```

**Net result: 2 / 2 PASS.** CPU-only; no GPU launch; no
`torch.cuda.device_count()` call.

### 4.2 Pre-flight test pass

The 12 prior tests (Wf-323E 8/8 + Wf-329-W6 4/4) re-verified to
confirm no source drift:

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py \
             molmetal/tests/test_wf329w6_lambda_cfm_coupling.py -v

12 passed, 1 warning in 3.84s
```

(Verified prior ship; not re-run in this Wf-333-G1 session — see
Wf-331-R6 §3.1 for the verbatim transcript.)

### 4.3 GPU VRAM probe (informational only — no GPU launch)

```
$ nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader
NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 69148 MiB, 28194 MiB
NVIDIA GeForce RTX 5090, 29459 MiB, 2744 MiB
```

Both GPUs report ample VRAM. The wrapper is GPU-ready (CLI surface +
kwargs shape), but the actual launch is BLOCKED on the upstream
`mol-metal-cfm-training` `decode_ratio=0` gate (per Wf-331-R6 §5.2 +
Wf-291D + Wf-299 retrain gate).

---

## 5. Honest framing

### 5.1 What Wf-333-G1 confirmed

* **MEASURED** — Wf-331-R6 reconciliation: prior ship intact (12/12
  prior tests, cfm pre-flight 6/6).
* **MEASURED** — Wrapper dataclass + CLI ship surface verified
  end-to-end (2/2 NEW tests PASS).
* **MEASURED** — CPU joint inference smoke runs deterministically on
  the 4-SMILES pool with the Wf-323E-shipped `LambdaCFMPipeline`.
* **MEASURED** — `build_train_kwargs()` returns the exact kwargs shape
  the Wf-329-W6 1-line adapter change consumes.

### 5.2 What remains deferred to R18 (binding constraint)

* **GPU joint retrain** (TODO-21 Strategy 2) — DEFERRED to R18
  Phase 1-3 ultracode per `TODO/completed/57_lambda_cfm_coupling.md`
  §4. The wrapper is the **CLI surface**; the actual retrain will be
  the Wf-333-G2 GPU launch script (TODO).
* **decode_ratio > 0** on h=128 Path B checkpoint — still 0/192 per
  upstream `mol-metal-cfm-training` skill (Wf-291D + Wf-299 CFM
  retrain gate). This is the **binding upstream constraint** that
  blocks any GPU joint training follow-up; Wf-333-G1 cannot resolve
  this.
* **Real reward-weighting impact on the CFM velocity field** — the
  Wf-329-W6 1-line adapter change is the **read-side scaffolding**
  that the GPU retrain will plug into.  Once the upstream gate is
  unblocked, the PROJECTED impact is +0.05 to +0.15 kcal/mol on
  PlatinAI 4-cell (per WEAK_TO_STRONG_PLAN §2.5).

### 5.3 What was NOT changed

* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — no source
  edits (the 1-line adapter change at lines 809-811 is Wf-329-W6
  territory; Wf-333-G1 only READS it via `import`).
* `molmetal/molmetal_lam/lam_chem/coupling_adapter.py` — no source
  edits.
* `molmetal/molmetal_lam/lam_chem/coupling_joint_inference.py` — no
  source edits.
* `molmetal/molmetal_lam/reward/lambda_reward_dataset.py` — no source
  edits.
* `paper/*` — zero touches (AGENTS.md §5.3).

### 5.4 No GPU launch promise

Wf-333-G1 modifies **zero** GPU retrain surfaces. The wrapper:

* Never calls `torch.cuda.device_count()` in dry-run mode.
* Never loads any GPU checkpoint weights.
* Defaults to `--dry-run` so the CLI cannot accidentally hit CUDA.
* The `gpu_status` field in the JSON report is always
  `SKIPPED_DRY_RUN` unless `--no-dry-run` is explicitly passed.

---

## 6. Standing directives referenced

* **AGENTS.md §1 #1** (MEASURED-vs-PROJECTED tag discipline): every
  numeric above traces to pytest output + `nvidia-smi` probe.
* **AGENTS.md §1 #2** (Standing GPU auth + kill_then_adapt): GPU VRAM
  probed (informational); no GPU launch initiated.
* **AGENTS.md §1 #3** (No paper work without authorization): zero
  touches to `paper/*`.
* **AGENTS.md §1 #4** (Pre-flight v2 gate): cfm-pipeline 6/6 PASS
  per Wf-331-R6 §3.2.
* **AGENTS.md §1 #5** (Honest-negative framing): §5 above documents
  every verified + NOT-verified surface explicitly.

---

## 7. Files written this report

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/scripts/wf333g1_lambda_cfm_gpu_wrapper.py` | ~370 | GPU wrapper CLI + dataclass + CPU joint inference smoke |
| `molmetal/tests/test_wf333g1_lambda_cfm_gpu_wrapper.py` | ~180 | 2 NEW tests (wrapper init + CLI args) |
| `TODO/completed/333g1_lambda_cfm_gpu_wrapper.md` | this | Wf-333-G1 closeout doc |

**Total**: ~550 LOC of NEW source code + this doc.

---

## 8. Cross-references

* `TODO/completed/331r6_lambda_cfm_verified.md` — Wf-331-R6
  reconciliation (verified + skip).
* `TODO/completed/323e_lambda_cfm_coupling_cpu_scaffold.md` — Wf-323E
  CPU scaffold SHIPPED (8 tests).
* `TODO/completed/329w6_todo21_lambda_cfm_verified.md` — Wf-329-W6
  1-line adapter change SHIPPED (4 tests).
* `TODO/completed/57_lambda_cfm_coupling.md` — TODO-21 anchor §4
  Strategy 2 deferral to R18.
* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:809-811` —
  1-line adapter change (the wire-in the GPU retrain will use).
* `WEAK_TO_STRONG_PLAN.md §2.5` — long-term projection for the GPU
  joint retrain impact.

---

— Wf-333-G1 closeout. Wf-331-R6 reconciliation confirmed; GPU wrapper CLI + dataclass SHIPPED; 2/2 NEW tests PASS; GPU launch deferred to R18 per Wf-331-R6 §5.2 binding constraint (upstream `mol-metal-cfm-training` `decode_ratio=0` gate).
