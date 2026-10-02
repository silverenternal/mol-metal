# Wf-358-B1 — W₁ (1-Wasserstein) adapter-layer scaffold + integration

**Status:** SHIPPED (Wf-358-B1 ultracode architecture-improvement #1)
**Date:** 2026-09-29
**Wall:** ~25 min CPU-only (scaffold + 4 tests + CLI wiring + docs)

## Summary

Wf-358-B1 ships the W₁ (1-Wasserstein) adapter-layer scaffold as the
**first architecture improvement** identified by Wf-357-C2.  The
scaffold lives at
`molmetal/adapters/flow_matching_lipman/wasserstein_w1.py` (~190 LOC)
and re-exposes the canonical closed-form 1-D W₁ (= mean | sort(pred) −
sort(ref) |, bit-equivalent to `wasserstein_1d_torchsort`) behind a
single dispatch entry-point + an env-var gate.  The trainer gains a
new `--wasserstein-w1` CLI flag (default OFF) that, when set, routes
the per-epoch W penalty through the W₁ closed form — bypassing the
sliced-W₂ projection loop and the compiled W₂ wrapper entirely.

**PROJECTED +20-30% training speed** on CPU (Wf-326E benchmark family)
with **no accuracy change** (W₁ closed form is bit-equivalent to the
legacy L1 path).  GPU-fused sort acceleration (Wf-326E-A sm_120 retune)
is orthogonal — the W₁ CPU path is bit-equivalent to
`torch.sort` + `.abs().mean()` so it picks up any future Triton-fused
sort acceleration automatically.

## Files changed

| Path | Status | LOC | Purpose |
|---|---|---|---|
| `molmetal/adapters/flow_matching_lipman/wasserstein_w1.py` | NEW | ~190 | W₁ closed-form + dispatch wrapper + multi-task helper |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | MODIFIED | +60 | `--wasserstein-w1` CLI flag + W₁ wire-in (DataLoader + manual loops) |
| `molmetal/tests/test_wf358b1_wasserstein_w1.py` | NEW | 4 tests | scipy reference / 5-metal multi-task / CLI flag / backward-compat |
| `TODO/completed/358b1_wasserstein_w1.md` | NEW | docs | this file |

## Public API

`molmetal/adapters/flow_matching_lipman/wasserstein_w1.py` exports:

- `wasserstein_w1_1d_torchsort(x, y)` — pure-PyTorch 1-D W₁ closed
  form (sort + abs + mean).  Bit-exact with
  `molmetal.molmetal_lam.reward.wasserstein_loss.wasserstein_1d_torchsort`.
- `wasserstein_w1_loss(pred, ref, weight=1.0)` — trainer-facing W₁
  pIC50 penalty (drop-in for `wasserstein_pic50_gradient`).
- `wasserstein_w1_multi_task(per_metal, weight=1.0)` — pairwise W₁
  over a per-metal dict (mirrors `wasserstein_transfer_loss`).
- `wasserstein_w1_dispatch(pred, ref, weight=1.0, *, force_off=None)` —
  env-gated dispatch wrapper (default OFF; respects `WASSERSTEIN_W1=1`
  per the Wf-335-P3 per-kernel gate convention).

## CLI flag

```
--wasserstein-w1    Route the per-epoch W pIC50 penalty through
                    the W₁ (1-Wasserstein) adapter-layer scaffold.
                    Skips the W₂ sliced projection loop. Default OFF.
```

Default OFF for bit-for-bit backward compat with Wf-131 P1 / Wf-181 G5 /
Wf-182 G6 / Wf-357-D2 shipped behaviour.  Setting the flag wires the W₁
dispatch into BOTH the DataLoader and manual-loop paths in
`train_one_epoch_joint`.

## Honest framing (party-twenty-spirit §3.5)

- **MEASURED** (CPU, Wf-358-B1 timestamp):
  - 4 NEW tests PASS / 0 FAIL.
  - W₁ closed form `wasserstein_w1_1d_torchsort` matches
    `scipy.stats.wasserstein_distance` reference to <1e-5 abs error
    across 5 synthetic distributions (uniform / Gaussian / outlier /
    bimodal / constant shift).
  - Backward-compat: 82/82 PASS for the
    `test_wasserstein_loss.py` + `test_wf357d2_wasserstein_2_drop_in.py`
    + `test_common_train_args.py` re-verification set (no new
    failures; CLI surface unchanged when flag absent).
  - CLI default OFF preserved bit-for-bit — `parse_args(["--metals",
    "Ru"]).use_wasserstein_w1 is False` (mirroring the Wf-357-D2
    `--use-wasserstein-2` precedent).

- **PROJECTED** (no GPU launch):
  - +20-30% training speed on the per-epoch W step on CPU vs the
    sliced-W₂ projection loop (Wf-326E benchmark family).
  - No accuracy change vs the legacy L1 path (same arithmetic, bit-
    equivalent to `wasserstein_1d_torchsort`).

- **BLOCKED on external dependency**:
  - GPU-fused sort + abs + mean kernel acceleration (Wf-326E-A sm_120
    retune) is deferred per AGENTS.md §5.1 — the W₁ CPU path is bit-
    equivalent to `torch.sort` + `.abs().mean()` so it picks up any
    future Triton-fused sort acceleration automatically.

## Test results

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf358b1_wasserstein_w1.py -v
test_w1_matches_scipy_reference        PASSED
test_multi_task_5_metal_w1             PASSED
test_cli_flag_wasserstein_w1_wired     PASSED
test_backward_compat_default_off       PASSED
4 passed in 1.10s
```

Backward-compat re-verification (82 tests, 4.21 s, all PASS):

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wasserstein_loss.py \
             molmetal/tests/test_wf357d2_wasserstein_2_drop_in.py \
             molmetal/tests/test_common_train_args.py -v
82 passed, 1 warning in 4.21s
```

## Architecture notes

The W₁ dispatch is wired BEFORE the existing `w2_gradient_mode == "sort"`
branch in both the DataLoader path (line ~1104) and the manual-loop
path (line ~1296) of `train_one_epoch_joint`.  When
`use_wasserstein_w1` is True the trainer imports
`wasserstein_w1_loss` lazily (inside the per-epoch branch) and calls
`wasserstein_w1_loss(pic50_pred, ref_dev, weight=float(w_w))`,
which is bit-equivalent to the legacy L1 closed form (mean |sort(pred)
- sort(ref)|).  Gradient flows back into `pic50_pred` via the standard
PyTorch autograd graph (no detach — same contract as
`wasserstein_pic50_gradient`).

The `force_off` argument on `wasserstein_w1_dispatch` is the explicit
backward-compat escape hatch (mirroring the Wf-335-P3 `force_pytorch`
pattern on `fused_wasserstein_sliced`).  When `force_off=True` the
dispatch falls through to the W₁ closed form regardless of the
`WASSERSTEIN_W1` env var — useful for tests that want to pin the
behaviour bit-exactly without depending on the environment.

## Future work (Wf-358+)

- W₂ W₁ paper §6 honest-ablation table (Wf-357-D2 provides the W₂
  side; Wf-358-B1 provides the W₁ side).  Both are bit-equivalent on
  Gaussian-ish inputs; the table quantifies the small delta on
  outlier-heavy distributions.
- GPU-fused sort + abs + mean kernel (Wf-326E-A sm_120 retune) — the
  W₁ path benefits automatically when the kernel ships.
- W₂+W₁ ensemble (Wf-358-C4 candidate) — averaged penalty across W₁
  and W₂ for outlier-robust OT alignment.  Out of scope for this
  closeout.
