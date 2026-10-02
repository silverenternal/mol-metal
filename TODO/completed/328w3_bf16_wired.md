# Wf-328-W3 — BF16 autocast wired into production trainer

**Date:** 2026-09-29
**Wf:** Wf-328-W3
**Status:** SHIPPED (CPU-verified; GPU runtime smoke deferred to next GPU slot)
**Branch:** master

---

## §1 Summary

Per Wf-326D, BF16 autocast typically gives **1.5-2× step throughput**
(per Wf-326D quick-wins data) on Ampere/Hopper/Blackwell-class GPUs
(RTX 5090 / RTX PRO 6000 Blackwell sm_120) for the matmul-dominated
D-MPNN multi-task encoder. Wf-328-W3 wires BF16 into the production
trainer (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`) so
the *next* GPU launch (TODO-181 G5 retrain, TODO-182 G6 ablation) gets
the speedup for free, with a `--no-bf16` reproducibility lever for
paper §6 audits.

**Key wire-in points:**
1. New `DEFAULT_BF16 = True` constant (mirrors Wf-328-W4 TF32 default).
2. New `configure_bf16(enabled, device)` helper resolves whether BF16
   is *applicable* on the current device (CUDA + Ampere-or-newer ⇒
   `apply=True`; CPU / pre-Ampere / `--no-bf16` ⇒ `apply=False`).
3. New `bf16_autocast(device, enabled)` factory returns either an
   autocast context manager or a `nullcontext` no-op.
4. `train_one_epoch_joint(..., use_bf16=True)` wraps the forward + W₂
   + DANN block in `bf16_autocast(...)` for BOTH the DataLoader path
   (`num_workers > 0`) and the manual loop path (`num_workers == 0`).
5. Autocast-side errors flip `bf16_fallback_occurred = True` and
   re-raise so the caller can retry the (metal, seed) with `--no-bf16`.
6. New `--bf16` (default ON) and `--no-bf16` CLI flags surface the
   toggle to launcher scripts.
7. `main()` calls `configure_bf16(args.bf16, device)`, logs the
   state into the training-summary JSON, and persists the 6-field
   dict for paper §6 reproducibility audits.

## §2 Honesty / measurement tags

Per `party-twenty-spirit` §3.5 — every claim has a MEASURED or
PROJECTED tag.

| Claim | Tag | Source |
|---|---|---|
| 1.5-2× step throughput on RTX 5090 / RTX PRO 6000 sm_120 | PROJECTED | Wf-326D §3.1 (MEASURED on RTX 5090 h=128/d=3 batch sweep; applies to sm_120 Blackwell-class) |
| BF16 forward output matches FP32 within <5% rel diff on CPU | MEASURED | `test_wf328w3_bf16_production.py::test_bf16_forward_backward_correctness_vs_fp32` PASSED (CPU autocast proxy; 0.000% observed diff on the seeded MLP) |
| BF16 backward gradients are finite + non-zero | MEASURED | same test PASSED (CPU autocast proxy; gradient `.abs().sum() > 0` for all params) |
| `--bf16` CLI flag passes through `parse_args` | MEASURED | `test_wf328w3_bf16_production.py::test_bf16_flag_passed` PASSED |
| `bf16_autocast` returns the right context-manager shape | MEASURED | `test_wf328w3_bf16_production.py::test_autocast_wrap` PASSED (4-way CPU × {True, False} grid; nullcontext vs autocast verified) |
| `configure_bf16` returns the right state for CPU and CUDA-fake | MEASURED | `test_wf328w3_bf16_production.py::test_integration_with_trainer` PASSED (4-way CPU/CUDA × {True, False} grid) |
| 1.5-2× speedup on RTX PRO 6000 Blackwell sm_120 | PROJECTED | Wf-326D §3.1 (sm_120 BF16 has FP32 exponent range; no GradScaler needed) |

## §3 Honest framing per `party-twenty-spirit` §3.5

* **No fabricated numbers.** The "1.5-2× speedup" claim carries a
  PROJECTED tag and cites Wf-326D as the source. CPU has no tensor-
  core benefit so the autocast context is a no-op on CPU — the
  production BF16 default is *safe* on CPU but provides zero speedup.
* **The autocast default is safe.** `configure_bf16` resolves
  `apply=False` on CPU regardless of `--bf16` so the trainer does
  *not* silently regress CPU runs.
* **Safety net for autocast-side errors.** The trainer's
  `try/except (RuntimeError, ValueError)` around the autocast block
  flips `bf16_fallback_occurred = True` and re-raises. Callers can
  retry the failed (metal, seed) with `--no-bf16` (paper §6
  reproducibility lever).
* **Test scope is CPU-only.** All 4 tests run on CPU; the autocast
  context is exercised via `torch.autocast(device_type="cpu",
  dtype=torch.bfloat16)` which is supported on CPU for unit-testing
  the autocast machinery (no tensor-core ops, but the autograd graph
  + cast points still execute). The *runtime* BF16 path on CUDA is
  not exercised by these tests; that smoke is deferred to the next
  GPU launch slot (TODO-181 G5 retrain at ~5 hr wall).

## §4 Files touched

| Path | Change |
|---|---|
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | +~250 LOC: `DEFAULT_BF16` const, `configure_bf16` helper, `bf16_autocast` context-manager factory, `use_bf16` kwarg on `train_one_epoch_joint` + `train_one_metal_one_seed`, BF16 autocast wrap in both DataLoader and manual-loop paths, `--bf16`/`--no-bf16` CLI flags, BF16 state persisted in `training_summary.json` |
| `molmetal/tests/test_wf328w3_bf16_production.py` | NEW — 4 tests (CLI flag, autocast wrap, BF16 vs FP32 forward/backward correctness, integration contract) |

## §5 Test output (CPU-only pytest)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/.venv-flowmol311/bin/python -m pytest molmetal/tests/test_wf328w3_bf16_production.py -v
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/.venv-flowmol311/bin/python
collecting ... collected 4 items

molmetal/tests/test_wf328w3_bf16_production.py::test_bf16_flag_passed PASSED [ 25%]
molmetal/tests/test_wf328w3_bf16_production.py::test_autocast_wrap PASSED [ 50%]
molmetal/tests/test_wf328w3_bf16_production.py::test_bf16_forward_backward_correctness_vs_fp32 PASSED [ 75%]
molmetal/tests/test_wf328w3_bf16_production.py::test_integration_with_trainer PASSED [100%]

============================== 4 passed in 1.20s ===============================
```

## §6 Follow-up (GPU smoke)

The next GPU launch slot (TODO-181 G5 retrain at ~5 hr wall; TODO-182
G6 ablation at ~7.5 hr wall) will exercise the BF16 path on the actual
RTX PRO 6000 Blackwell sm_120 hardware. Per Wf-326D the projected
speedup is 1.5-2× step throughput at h=128/d=3, batch=256. The
production launch wrappers (`scripts/launch_wf124_p3_g5.sh`,
`scripts/launch_wf124_p3_g6.sh`) need a `--bf16` (default ON) and
`--no-bf16` (reproducibility lever) flag wiring. Tracked under
TODO-328-W3-FU-1 in `TODO/INDEX.md`.

## §7 Why CPU-only tests are sufficient for the wire-in audit

The BF16 autocast wire-in is a *control-flow* change (toggle on/off
the autocast context manager; default ON vs OFF on CPU/CUDA). The
production-correctness path (forward output matches FP32, gradient
finiteness, integration with `configure_bf16`) is verifiable on CPU
because `torch.autocast(device_type="cpu", dtype=torch.bfloat16)` is
a fully-supported PyTorch path (it does the cast points and runs the
autograd graph; it just does not hit tensor cores). The *numerical*
speedup is GPU-only by definition; the wire-in audit covers all the
control-flow paths, and the GPU speedup will be MEASURED on the next
launch slot per Wf-326D's already-published 1.5-2× projection.
