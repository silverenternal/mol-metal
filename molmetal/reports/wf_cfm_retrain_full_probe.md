# WF-CFM-Retrain-Full — Phase 1 GPU Probe Report

**Date:** 2026-09-14
**Workflow:** WF-CFM-Retrain-Full (full CFM retrain @ 10000-step + hidden-dim 64)
**Project root:** /home/hugo/codes/try_triton_on_rocm
**Python:** uv-managed Python 3.12
**Stack claimed:** ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64

---

## 1. GPU availability probe (mandatory gate)

Command executed:

```bash
uv run python -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())"
```

Raw output:

```
(null): No such file or directory
torch.cuda.is_available(): False
device_count: 0
```

Interpretation:

- `torch.cuda.is_available()` is **False**.
- `device_count` is **0**.
- The `(null): No such file or directory` line above is the conda/uv stub complaining about a missing module path (cosmetic — irrelevant to the CUDA answer; PyTorch still imported and answered the query).
- ROCm exposes devices through the same `torch.cuda` API in `torch==2.x` built with `+ROCM`; the False answer means either (a) `libhsa-runtime` / ROCm userspace is not visible to the uv venv, or (b) the build of PyTorch in the venv is a CPU-only wheel.

**Verdict: GPU is not visible to the uv-managed Python 3.12 environment.** This is exactly the failure mode that R10 / R11 ran into (see `molmetal_post_r10_r11.md` — Pt e2e INCONCLUSIVE, CFG 0/96 decoder-bound; root cause was hardware/ROCm visibility, not the model code).

---

## 2. CLI signature confirmation

Script: `molmetal/scripts/r10_cfg_real_crossdocked.py`

Grep over the script's argparse block confirmed both flags are present and `int`-typed:

- `p.add_argument('--train-steps', type=int, default=200)` — supports `--train-steps 10000` (current default 200).
- `p.add_argument('--hidden-dim', type=int, default=16)` — supports `--hidden-dim 64` (current default 16).

So the CLI surface is ready for the retrain command; the gate is purely hardware.

---

## 3. Budget confirmation

Planned command (for the record; **not run**):

```
uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --train-steps 10000 --hidden-dim 64 \
    --seeds 42 0 1234 7 21 99 \
    --n-samples 16 \
    ... (two pockets, e.g. 1h36 + 830c)
```

Compute estimate:

- Per-seed CFM training at hidden_dim=64, batch=2 (as set in the script's protocol), 10000 steps: roughly 1.0–2.0 h single-card on gfx1101, dominated by per-step CPU↔GPU sync on the small batches.
- 6 seeds × 2 pockets = 12 training runs.
- 12 × ~1–2 h = **~12–24 h single-card** wall-clock.
- Plus sampling/inference (16 samples × 12 runs) and PoseBusters/mmff scoring overhead — round to **~6–12 h best case, 12–24 h realistic**.

The "~6–12 h" headline in the spec is the optimistic end of the range; on a 7800 XT with the CPU-sync bottleneck observed in R11, **realistic budget is closer to 12–24 h single-card**. Honest framing: this is an overnight-to-multi-day job, not a coffee break.

---

## 4. Decision

`can_proceed = False`.

Per the workflow's explicit Phase-1 gate ("If torch.cuda.is_available() is False: return immediately with status GPU_BLOCKED — do NOT proceed to Phase 2"), this workflow stops here.

**Recommendation:** Defer WF-CFM-Retrain-Full to the **GPU-recovery worktree**. The right next move is the ROCm-visibility triage already on the round-9/10 roadmap (separate worktree, separate context, root-cause the `(null): No such file or directory` + `cuda.is_available()=False` pair before spending 12–24 h of compute that will silently fall back to CPU).

Concretely the recovery worktree should investigate:

1. Which `torch` wheel is actually installed in the uv venv (is it `+ROCM` or `+cpu`?).
2. Whether `ROCM_PATH`, `HSA_OVERRIDE_GFX_VERSION`, and `LD_LIBRARY_PATH` are exported into the uv subprocess environment.
3. Whether `libhsa-runtime64.so.1` and `libamdhip64.so` are findable via `ldconfig`.
4. Whether the gfx1101 kernel module is loaded (`lsmod | grep amdgpu`, `rocminfo`).
5. Whether the prior `devoluciones/mm/python` env that worked in earlier rounds still exists and can be re-attached.

Until that triage returns a True on `torch.cuda.is_available()`, **WF-CFM-Retrain-Full stays parked**.

---

## Metrics

| field | value |
|---|---|
| torch_cuda_available | False |
| device_count | 0 |
| budget_hours | 12–24 (single-card, realistic); 6–12 (optimistic headline) |
| can_proceed | False |
| status | GPU_BLOCKED |