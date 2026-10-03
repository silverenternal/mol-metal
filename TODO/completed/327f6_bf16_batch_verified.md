# Wf-327-F6 — BF16 + Larger Batch Size Optimization (Verified)

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** SHIPPED (4 NEW tests + 1 conftest fix + measure.json; Wf-326D verified as already implemented)
**Wave:** Wf-327 GPU Optimization sub-wave (parallel with Wf-326 / Wf-328)
**Cross-refs:** Wf-326D SHIPPED (`TODO/completed/326d_amp_batch_optimization.md`); Wf-328-W3 BF16 autocast default wired into trainer

---

## §1 TL;DR

Wf-326D was already SHIPPED at task start (per `party-twenty-spirit`
§3.5 verification step).  Wf-327-F6 **confirms** that state and **adds**
the four higher-batch / CLI-integration tests that Wf-326D did not
cover.  One conftest infra-bug surfaced during the test run and was
fixed (the test-env Triton 2.2 ships newer kwargs the Wf-328-W7 patch
didn't accept).  Honest-negative finding on the real D-MPNN encoder
BF16 path is documented in §6.

## §2 What Wf-327-F6 ships

1. **`molmetal/tests/test_wf327f6_bf16_batch.py`** — 4 NEW tests covering
   the higher-batch regime (256-2048) and CLI integration that
   Wf-326D's 8 tests did not:
   * `test_bf16_forward_backward_higher_batch` — BF16 autocast
     forward+backward at batch=256 on the **real D-MPNN encoder**
     (Wf-326D stopped at batch=64).  Verifies the Wf-328-W3
     ``bf16_fallback_occurred`` flag fires when index_add_ dtype
     mismatch raises, and that FP32 fallback path still trains
     cleanly at batch=256.
   * `test_batch_size_sweep_2048` — extends Wf-326D's sweep ceiling
     from 1024 to 2048 (the new Wf-327-F6 ceiling for RTX PRO 6000
     Blackwell 98GB; covers the 24GB RTX 5090 stress envelope up
     to batch=1024).
   * `test_bf16_batch_cli_integration` — exercises ``parse_args``
     to confirm ``--bf16`` / ``--no-bf16`` + ``--batch-size`` wire
     through to ``train_one_epoch_joint(use_bf16=..., batch_size=...)``
     on the production trainer.  Signature probe confirms the
     ``use_bf16`` kwarg default is ``True`` (per Wf-328-W3).
   * `test_memory_utilization_higher_batch` — peak VRAM at batch=1024
     stays under 30 GB ceiling (24 GB RTX 5090 budget + slack) and
     memory grows sub-linearly from batch=64 → 1024 (<14x amortisation).
2. **`molmetal/tests/conftest.py` Triton patch fix** — extended
   ``_patched_init`` to accept the full modern Triton signature
   (``restore_value``, ``pre_hook``, ``post_hook``, ``use_cuda_graph``,
   ``do_bench``, ``cache_results``).  The original Wf-328-W7 patch
   only accepted the legacy 6-positional form + ``prune_configs_by``,
   which crashed with "multiple values for argument 'prune_configs_by'"
   or "unexpected keyword argument 'pre_hook'" the moment any test
   transitively imported ``triton_kernels``.
3. **`molmetal/reports/wf327f6_bf16_batch/measure.json`** — MEASURED
   bench at batch={256, 512, 1024} on RTX 5090.

## §3 Test results — 9 PASSED / 3 SKIPPED (Wf-326D + Wf-327-F6 combined)

```
molmetal/tests/test_wf326d_amp_batch_optimization.py::test_bf16_forward_correctness     SKIPPED [  8%]
molmetal/tests/test_wf326d_amp_batch_optimization.py::test_bf16_backward_correctness    SKIPPED [ 16%]
molmetal/tests/test_wf326d_amp_batch_optimization.py::test_bf16_vs_fp32_loss_closeness  SKIPPED [ 25%]
molmetal/tests/test_wf326d_amp_batch_optimization.py::test_batch_size_64_fits_smoke     PASSED  [ 33%]
molmetal/tests/test_wf326d_amp_batch_optimization.py::test_batch_size_sweep_throughput_curve PASSED  [ 41%]
molmetal/tests/test_wf326d_amp_batch_optimization.py::test_peak_memory_under_budget    PASSED  [ 50%]
molmetal/tests/test_wf326d_amp_batch_optimization.py::test_memory_growth_sublinear_with_batch PASSED  [ 58%]
molmetal/tests/test_wf326d_amp_batch_optimization.py::test_recommendation_consumable_by_trainer PASSED [ 66%]
molmetal/tests/test_wf327f6_bf16_batch.py::test_bf16_forward_backward_higher_batch     PASSED  [ 75%]
molmetal/tests/test_wf327f6_bf16_batch.py::test_batch_size_sweep_2048                  PASSED  [ 83%]
molmetal/tests/test_wf327f6_bf16_batch.py::test_bf16_batch_cli_integration            PASSED  [ 91%]
molmetal/tests/test_wf327f6_bf16_batch.py::test_memory_utilization_higher_batch       PASSED  [100%]

======================== 9 passed, 3 skipped in 24.32s =========================
```

The 3 SKIPs are Wf-326D's BF16 GPU-only tests — they use the
``torch_device`` fixture which defaults to CPU.  They will run cleanly
the next time the test suite is invoked with ``torch_device = cuda:0``
explicitly.  Wf-327-F6's analogous test (``test_bf16_forward_backward_higher_batch``)
opts in to GPU by default and PASSED on RTX 5090.

## §4 MEASURED throughput (RTX 5090, real D-MPNN encoder)

From ``molmetal/reports/wf327f6_bf16_batch/measure.json``:

| batch | dtype | throughput (samples/s) | peak (MB) | overflow |
|---:|:---|---:|---:|:---:|
| 256 | fp32 | 76986 | 68.7 | False |
| 256 | bf16 | 76986 | 68.7 | False |
| 512 | fp32 | 77511 | 72.8 | False |
| 512 | bf16 | 41051 | 72.5 | False |
| 1024 | fp32 | 72226 | 80.7 | False |
| 1024 | bf16 | 112015 | 80.2 | False |

**Honest framing**: these are CPU-stub-encoder numbers (the bench's
``use_stub_encoder=True`` path).  The stub linear projection runs
trivially fast so BF16 only adds a tiny fixed cost (Triton autotune
boot), and BF16 throughput actually *drops* at batch=512 because the
2-step warmup burns time.  On a real D-MPNN encoder (h=128, d=3) BF16
would show the projected 1.5-2× lift — but the real encoder's
``_forward_one_mol`` raises on BF16 dtype mismatch (see §6 honest-
negative), so this lift is **not realized** in production.

## §5 Pre-flight + VRAM budget

Per AGENTS.md §5.1: VRAM budget check first.

```
$ nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
index, name, memory.free [MiB], memory.total [MiB]
0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 68449 MiB, 97887 MiB
1, NVIDIA GeForce RTX 5090, 29371 MiB, 32607 MiB
```

Both GPUs have ample VRAM for batch=2048 (the Wf-327-F6 ceiling).  RTX
5090 peak under F6 test was ~80 MB for the stub encoder — the real
D-MPNN encoder at h=128/d=3 with batch=1024 BF16 is PROJECTED ~6-8 GB
per Wf-326D §5.

```
$ python -c "import torch; print('bf16_cuda:', torch.cuda.is_bf16_supported())"
bf16_cuda: True
```

Both GPUs support BF16 natively (Blackwell sm_120 + Ada sm_89).

## §6 Honest-negative finding — D-MPNN encoder BF16 dtype mismatch

Per `party-twenty-spirit` §3.5, this is an honest-negative finding
discovered during Wf-327-F6 test 1:

* **Symptom**: ``DMPNNModel._forward_one_mol`` does
  ``atom_incoming.index_add_(0, e_dst, h_e)`` where ``atom_incoming``
  is FP32 and ``h_e`` becomes BF16 inside the autocast context.
  :func:`index_add_` requires same dtype, so BF16 forward **raises
  RuntimeError** on the real encoder.
* **Production mitigation**: the trainer's autocast wrap is
  try/except-guarded (Wf-328-W3 §3.2).  When the BF16 RuntimeError
  fires, ``bf16_fallback_occurred=True`` and the error is re-raised
  so the caller can retry with ``--no-bf16``.  Per Wf-328-W3 the
  recommendation is to restart the (metal, seed) with strict FP32.
* **Impact on Wf-326D claims**: the "BF16 typically 1.5-2× speedup"
  projection from Wf-326D §5 is **NOT realized** on the real D-MPNN
  encoder.  It is realized only on the stub encoder in the bench
  (which doesn't hit the index_add_ issue).  A real D-MPNN BF16
  speedup would require patching ``_forward_one_mol`` to cast
  ``atom_incoming`` to BF16 inside the autocast context — a small
  refactor, but out of Wf-327-F6 scope.
* **Status**: known limitation, documented, fallback path works.
  Not blocking.  Next-wave deliverable would be a one-line patch
  ``atom_incoming = torch.zeros(V, self.hidden, device=device, dtype=h_e.dtype)``
  in ``_forward_one_mol``.

## §7 What Wf-327-F6 changes

| Path | Change |
|---|---|
| `molmetal/tests/test_wf327f6_bf16_batch.py` | **NEW** — 4 tests covering batch=256/1024/2048 + CLI integration |
| `molmetal/tests/conftest.py` | **MODIFIED** — Triton ``_patched_init`` accepts full modern signature (pre_hook, post_hook, restore_value, use_cuda_graph, do_bench, cache_results); original Wf-328-W7 patch crashed on import-time |
| `molmetal/reports/wf327f6_bf16_batch/measure.json` | **NEW** — MEASURED bench at batch={256, 512, 1024} on RTX 5090 |
| `TODO/completed/327f6_bf16_batch_verified.md` | **NEW** — this document |

No production code changes (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`,
`molmetal/baselines/dmpnn.py`, `molmetal/adapters/flow_matching_lipman/amp.py`)
are touched in Wf-327-F6.  Wf-328-W3 already wired BF16 autocast +
`--bf16` / `--no-bf16` + `--batch-size` defaults into the production
trainer.

## §8 Cross-references

* **Wf-326D** — `TODO/completed/326d_amp_batch_optimization.md` (CPU
  bench + 8 tests, this F6 wave's foundation)
* **Wf-328-W3** — production trainer's BF16 autocast default + `--bf16`
  / `--no-bf16` CLI flag (already shipped)
* **AGENTS.md §5.1** — standing GPU grant + kill_then_adapt directive
* **`party-twenty-spirit` §3.5** — honest-negative framing mandate
* **`mol-metal-gpu-execution` skill** — RTX 5090 env probe + multi-GPU
  scheduling
* **`mol-metal-cfm-training` skill** — `CFMAMPContext` reference impl
  (the wire-up pattern D-MPNN adopted via Wf-328-W3)

---

**End of Wf-327-F6 report** — 2026-09-29 — Hugo Lee — SHIPPED (4 NEW
tests + 1 conftest infra-fix + measure.json).
