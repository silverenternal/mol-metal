# Wf-328-W1 — torch.compile wired into production trainer

**Date:** 2026-09-29
**Wf:** Wf-328 (Wave 3 quick-win ship surface for the Wf-326 batch)
**Ship surface:** production trainer port of the Wf-326C
``torch.compile(mode='reduce-overhead')`` optimization
**File:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`

## Summary

Per Wf-326C MEASURED, applying ``torch.compile(mode='reduce-overhead')`` to
the D-MPNN encoder forward + the W₂ closed-form gives **2.72× encoder +
2.41× W₂ step speedup** on the RTX PRO 6000 Blackwell sm_120 GPU (see
`molmetal/tests/test_wf326c_torch_compile.py` for the bench suite).  The
optimization lived in `molmetal/scripts/wf326c_torch_compile_optimized.py`
(Wf-326C optimized trainer) but **was NOT wired into the production
trainer** `train_dmpnn_multitask_wasserstein.py` (Wf-131 P1 / Wf-181 G5
/ Wf-182 G6).

This Wf ports the Wf-326C optimisations into the production trainer via
two wrapper helpers + a monkey-patch onto the encoder's
``_forward_one_mol``.  The port preserves backward compatibility (no
behavior change with ``--no-torch-compile``) and threads through the
existing CLI / DataLoader / BF16 / DANN / Kendall UW lever stack.

## Integration points

| Component | Implementation | File location |
|---|---|---|
| Wrapper helper 1 | ``_make_compiled_encoder_forward`` | `train_dmpnn_multitask_wasserstein.py:163` |
| Wrapper helper 2 | ``_make_compiled_w2_penalty`` | `train_dmpnn_multitask_wasserstein.py:204` |
| ``train_one_epoch_joint`` parameter | ``compiled_w2_penalty: Optional[object] = None`` | `train_dmpnn_multitask_wasserstein.py:642` |
| ``train_one_epoch_joint`` W₂ call site (sort mode, both paths) | DataLoader + manual loops use compiled wrapper when provided | `train_dmpnn_multitask_wasserstein.py:790-806` and `train_dmpnn_multitask_wasserstein.py:920-931` |
| ``train_one_metal_one_seed`` parameter | ``use_compile``, ``torch_compile_mode`` | `train_dmpnn_multitask_wasserstein.py:1098-1099` |
| Encoder monkey-patch | ``model.encoder._forward_one_mol = compiled_encoder_forward`` | `train_dmpnn_multitask_wasserstein.py:1278-1279` |
| Encoder restore | ``finally:`` block in ``train_one_metal_one_seed`` | `train_dmpnn_multitask_wasserstein.py:1424-1429` |
| CLI flags | ``--torch-compile`` (default True), ``--no-torch-compile``, ``--torch-compile-mode {default,reduce-overhead,max-autotune,max-autotune-no-cudagraphs}`` | `train_dmpnn_multitask_wasserstein.py:1881-1918` |
| Summary JSON | ``torch_compile_enabled``, ``torch_compile_mode`` | `train_dmpnn_multitask_wasserstein.py:2252-2253` |

**2 integration points** identified: encoder forward + W₂ penalty.

## Tests

`molmetal/tests/test_wf328w1_torch_compile_production.py` — **4 tests, all PASS**:

1. ``test_torch_compile_flag_passed`` — CLI flag wiring (default ON,
   ``--no-torch-compile`` flips, ``--torch-compile-mode`` accepts all
   four Inductor modes, invalid mode rejected)
2. ``test_model_encoder_wrapped_with_compile`` — encoder forward
   monkey-patched with compiled wrapper preserves output within FP32
   tolerance across first + second calls (compile cache hit)
3. ``test_w2_loss_wrapped_with_compile`` — compiled W₂ wrapper
   preserves W₂ value + gradient within FP32 tolerance vs eager
   :func:`wasserstein_pic50_gradient`
4. ``test_compile_integration_with_trainer`` — end-to-end: encoder
   compile + W₂ compile wrap cleanly into
   :func:`train_one_epoch_joint`; autograd graph flows through
   encoder weights after backward (||grad|| > 1e-6)

All tests are CPU-only by design (matching Wf-326C); harness
auto-fallback (Wf-320B) keeps GPU contention at zero so the Inductor
compile latency (~30s first run) does not perturb in-flight runs.

## Pre-flight gate

``molmetal/scripts/pre_flight_check.py`` exit code 0 (PASS gate
preserved; the new torch.compile wiring does not add a regression
test to the v2 25/25 checklist — the Wf-326C bench already covers the
compile-correctness + speedup axes).

## Honest framing (party-twenty-spirit §3.5)

* **MEASURED 2.72× encoder + 2.41× W₂ step speedup** on RTX PRO 6000
  Blackwell sm_120 (per Wf-326C bench, MEASURED + JSON on disk at
  `molmetal/tests/test_wf326c_torch_compile.py`).
* **Projected production speedup**: ~1.4-1.7× end-to-end step
  throughput (the encoder + W₂ are 2 of ~6 ops in the training step;
  the other 4 — optimizer step, data loader, BF16 autocast, loss
  computation — are not torch.compile-traced and bound the ceiling).
  This projection follows from the Wf-326A quick-wins analysis + the
  Wf-326C bench; the actual production speedup will be re-measured
  end-to-end on the Wf-181 G5 RTX 5090 retrain once that run is
  launched.
* **torch.compile compile latency**: ~30s first run; amortised after
  ~100 forward passes on static shape.  The encoder wrapper uses
  ``dynamic=True`` because per-mol V/E varies across the cohort —
  Inductor re-compiles per shape but the dispatch + cache hit amortises
  after the first epoch.
* **mode='reduce-overhead'** enables CUDA Graphs capture inside
  Inductor; on CPU hosts this is a no-op (no CUDA Graphs available).
  ``--torch-compile-mode max-autotune-no-cudagraphs`` is the
  CPU-safe mode.
* **CPU-only test** (compile correctness verifiable on CPU).  GPU
  bench lives in Wf-326C.

## Usage

::

    # Default — torch.compile ON, mode='reduce-overhead'
    python molmetal/scripts/train_dmpnn_multitask_wasserstein.py \\
        --metals Ru --seeds 0 --epochs 5

    # Disable torch.compile (eager baseline, paper §6 reproducibility)
    python molmetal/scripts/train_dmpnn_multitask_wasserstein.py \\
        --metals Ru --seeds 0 --epochs 5 --no-torch-compile

    # CPU-safe mode (no CUDA Graphs)
    python molmetal/scripts/train_dmpnn_multitask_wasserstein.py \\
        --metals Ru --seeds 0 --epochs 5 \\
        --torch-compile-mode max-autotune-no-cudagraphs

## Cross-references

* Wf-326C bench: `molmetal/scripts/wf326c_torch_compile_optimized.py`,
  `molmetal/tests/test_wf326c_torch_compile.py`
* Wf-328 W3 (BF16): `TODO/completed/328w3_bf16_wired.md`
* Wf-328 W4 (TF32): `TODO/completed/328w4_tf32_wired.md`
* Wf-328 W5 (W₂ sort default): `TODO/completed/328w5_w2_sort_wired.md`
* Wf-131 P1 (production trainer):
  `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
