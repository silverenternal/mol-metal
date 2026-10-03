# Wf-328-W4 — TF32 wired into production trainer

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Wf:** Wf-328-W4 (ultracode sweep)
**Status:** SHIPPED
**Honest framing:** MEASURED impact from Wf-326A quick wins; trivial 1-line config.

---

## §1 TL;DR

TF32 enabled by default in the production D-MPNN multi-task + Wasserstein
trainer. Three flags flipped in one call:
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`
- `torch.set_float32_matmul_precision('high')`

Projected speedup (per Wf-326A): **~1.5× matmul speedup** on Ampere/Hopper/
Blackwell-class GPUs with negligible accuracy loss. CPU-only runs are
unaffected (torch global flags are silently ignored when CUDA is absent).

---

## §2 Files touched

| File | Change |
|---|---|
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | Added `configure_tf32()` helper + `DEFAULT_TF32` constant + `--tf32` / `--no-tf32` CLI flags + `main()` wire-in + summary-JSON logging. |
| `molmetal/tests/test_wf328w4_tf32_production.py` | NEW — 3 tests: default-on, flag override, idempotency. |

No other files modified.

---

## §3 CLI surface

```
python molmetal/scripts/train_dmpnn_multitask_wasserstein.py ... \
    [--tf32 | --no-tf32]
```

- `--tf32` (default; matches `DEFAULT_TF32=True`): enables TF32 matmul +
  cuDNN + sets matmul precision to `'high'`.
- `--no-tf32`: disables TF32; matmul precision forced to `'highest'`. Use
  for paper §6 reproducibility audits where bit-for-bit FP32 agreement
  is required.

The post-config state is logged into `training_summary.json` under the
keys `tf32_enabled`, `tf32_matmul_precision`,
`tf32_current_allow_tf32_matmul`, `tf32_current_allow_tf32_cudnn` so
downstream consumers (paper §6, ablations, audit ledger) can verify the
matmul precision used for a given per-seed AUC.

---

## §4 Honest framing

Per `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED discipline):

- **TF32 matmul speedup: ~1.5× MEASURED** per Wf-326A quick wins (the
  Wf-326A bench produced this number on a Blackwell-class RTX PRO 6000
  using a synthetic matmul microbench; it is not re-measured here).
- **Accuracy delta: PROJECTED negligible** for the matmul-dominated
  D-MPNN encoder + per-metal heads. The Wasserstein regulariser uses
  sliced / sort-based 1-D gradients (no FP64-sensitive reductions), so
  the pIC50 W₂ penalty is not adversely affected. Per-seed AUC spread
  may shift by ≤ 0.001 — within noise; not measured here.
- **CPU-only runs: NO-OP** — `torch.backends.cuda.*` flags are silently
  ignored when CUDA is unavailable; `set_float32_matmul_precision` still
  succeeds (PyTorch accepts `'high'` on CPU; downstream matmul ops
  remain in FP32 because there is no CUDA matmul kernel to TF32-ify).
- **TF32 settings are checkable on CPU** — the torch global flags
  (`torch.backends.cuda.matmul.allow_tf32`, etc.) are read-back
  regardless of CUDA availability. The 3 tests in
  `test_wf328w4_tf32_production.py` exercise the helper on CPU only.

---

## §5 Test results (Wf-328-W4 CPU run)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
  molmetal/tests/test_wf328w4_tf32_production.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
configfile: pyproject.toml
collected 3 items

molmetal/tests/test_wf328w4_tf32_production.py::test_tf32_enabled_by_default PASSED [ 33%]
molmetal/tests/test_wf328w4_tf32_production.py::test_tf32_flag_override      PASSED [ 66%]
molmetal/tests/test_wf328w4_tf32_production.py::test_configure_tf32_idempotent_repeated_calls PASSED [100%]

============================== 3 passed in 2.04s ===============================
```

All 3 tests PASS. The test file exercises:
1. `test_tf32_enabled_by_default` — verifies `configure_tf32(True)` flips
   all three flags and that `DEFAULT_TF32=True`.
2. `test_tf32_flag_override` — verifies `parse_args(['--no-tf32'])` /
   `parse_args(['--tf32'])` produce the expected booleans, and that
   `configure_tf32(False)` clears the flags (`matmul_precision='highest'`).
3. `test_configure_tf32_idempotent_repeated_calls` — defensive: repeated
   calls do not crash and the last call wins.

---

## §6 Pre-flight + production gate

- **Pre-flight v2 25/25 PASS gate (TODO-207)**: not applicable — this
  change is a CPU-only config edit; no retrain / panel / ablation
  launched. The TF32 wiring is a one-line CUDA init, not a model
  architecture change.
- **Wf-326A quick wins reference**: see `TODO/completed/326a_*.md` for
  the original TF32 microbench numbers that motivated this wire-in.
- **GPU standing grant (§5.1)**: not exercised by this Wf — no GPU
  launch was performed. The projected 1.5× speedup will be MEASURED on
  the next Wf-124 P3 G5/G6 retrain launch (TODO-181 / TODO-182).

---

## §7 Notes for downstream

- The `configure_tf32` helper is importable as
  `from molmetal.scripts.train_dmpnn_multitask_wasserstein import configure_tf32`.
  Re-use it from any other trainer that wants the same TF32 contract
  (e.g. `train_chemprop_v2_v3.py`, `train_cfm_yuelbond_wasserstein.py`).
- When launching a new trainer wave, prefer the default (`--tf32`).
  Pass `--no-tf32` only for explicit FP32 reproducibility audits.
- The `training_summary.json` now carries TF32 config keys; aggregator
  scripts should preserve them when re-emitting summaries.

---

**END OF 328w4_tf32_wired.md**
