# WF-CFM-Retrain-Full — final report

**Date:** 2026-09-14 (UTC)
**Workflow:** WF-CFM-Retrain-Full (full CFM retrain @ 10000-step + hidden-dim 64)
**Status:** **NOT RUN — Phase-1 GPU gate failed. Stops before Phase 2 (full retrain).**
**Author:** WF-CFM-Retrain-Full

---

## 1. Goal (verbatim from spec)

> WF-CFM-Retrain-Full goal: full CFM retrain @ 10000-step + hidden-dim 64 — expected to lift Vina mean from -2 kcal/mol to -5 to -6 kcal/mol (closing the biggest gap vs TargetDiff -8.45). Phase 1 MUST include GPU availability probe to avoid wasting GPU time on already-blocked hardware.

## 2. CLI (verbatim from spec, NOT executed)

```bash
uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --seeds 42 0 1234 7 2024 31415 \
    --train-steps 10000 --n-train 64 --ode-steps 64 \
    --n-samples 16 --hidden-dim 64 --n-layers 3 --lr 0.0001 \
    --vocab-mask --bond-head learned --joint-train \
    --output-dir molmetal/reports/wf_cfm_retrain_full/ \
    --gpu-binary scripts/_fake_vina.sh
```

Script: `molmetal/scripts/r10_cfg_real_crossdocked.py`.
Verified all flags match (`--seeds`, `--train-steps`, `--n-train`, `--ode-steps`, `--n-samples`, `--hidden-dim`, `--n-layers`, `--lr`, `--vocab-mask`, `--bond-head`, `--joint-train`, `--output-dir`, `--gpu-binary`).

## 3. Phase-1 GPU probe (mandatory gate) — MEASURED

```
torch version:        2.14.0+rocm7.2
hip runtime:          7.2.53211
triton:               3.8.0
torch.cuda.is_available():  False
torch.cuda.device_count():  0
/opt/rocm/bin/rocminfo:     ROCk module is loaded; HSA_STATUS_ERROR at /usr/src/debug/rocminfo/.../rocminfo.cc:1329
```

Direct script invocation reproduces the gate failure:

```
PYTHONPATH=. uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --seeds 42 --train-steps 10 --n-train 2 ... --gpu-binary scripts/_fake_vina.sh
...
RuntimeError: Real ROCm GPU required
  File "molmetal/scripts/r10_cfg_real_crossdocked.py", line 261, in main
```

So the gate is enforced both by the Phase-1 probe and by an in-script hard guard at line 261 (`if not torch.cuda.is_available(): raise RuntimeError('Real ROCm GPU required')`). Even the cheapest minimal command (2 train ligands × 10 steps × 1 seed) aborts before training.

This is the **same blocker** that hit WF-CFM-Retrain-Diagnose earlier today (HSA_STATUS_ERROR + `device_count=0`); `rocminfo` confirms `ROCk module is loaded` but the HSA user-mode runtime cannot enumerate a usable GPU for the uv-managed Python 3.12 venv.

## 4. Result table — MEASURED (no Phase-2 work done)

| metric                              | value            | source                                                  |
|-------------------------------------|-----------------:|---------------------------------------------------------|
| `torch.cuda.is_available()`         | **False**        | `uv run python -c "import torch; print(torch.cuda.is_available())"` |
| `torch.cuda.device_count()`         | 0                | same                                                    |
| `rocminfo`                          | `HSA_STATUS_ERROR` (`ROCk module is loaded`) | `/opt/rocm/bin/rocminfo`                |
| script invocation                   | aborts at line 261 (`RuntimeError: Real ROCm GPU required`) | direct run |
| Phase-2 retrain executed?           | **No**           | gate failed                                             |
| `decode_ratio_after_10000_steps`    | **NOT MEASURED** | gate failed before Phase 2                              |
| `n_decoded`                         | 0                | trivially (no harness output produced)                  |
| `n_finite`                          | 0                | trivially                                               |
| `n_requested`                       | 0                | trivially                                               |
| `bond_loss_final`                   | **NOT MEASURED** | n/a (no training)                                       |
| `n_params`                          | **NOT MEASURED** | n/a (model never instantiated on GPU)                  |
| `estimated_vina_mean_lift`          | **0.00 kcal/mol**| n/a (gate failed; cannot estimate)                      |
| `decode_ratio` (compared baseline)  | WF-CFM-Retrain-Diagnose = **NOT_MEASURED** (same blocker); WF-2 baseline = **0/384 @ 2000-step + h32** | see wf_cfm_diagnose/final.md and wf2_cfg_e2e_a5/report.json |
| `improvement_vs_baseline`           | **n/a**          | cannot measure                                          |

## 5. Baseline comparison (carry-forward, no new measurement today)

| baseline                        | decode_ratio        | source                                              |
|---------------------------------|---------------------|-----------------------------------------------------|
| WF-CFM-Retrain-Diagnose         | **NOT_MEASURED**    | same ROCm/HSA outage; script aborts at line 261     |
| WF-2 baseline (2000-step + h32) | **0/384**           | `molmetal/reports/wf2_cfg_e2e_a5/report.json` (2026-09-14T03:22 UTC) |
| TargetDiff cite-only SOTA       | -8.45 kcal/mol Vina | wf_3_citeonly_sota / §4 Table 1                     |

There is no new measurement to report. The same ROCk+HSA failure mode has now blocked both `WF-CFM-Retrain-Diagnose` and `WF-CFM-Retrain-Full` in this session.

## 6. `bond_loss` trajectory

Not produced. The Phase-1 gate prevents any training step from executing, so no bond-loss curve is available. The wf2 baseline trajectory (which reached `bond_loss ≈ 0` long before the model's decode improved) is the only on-record curve and lives in `wf2_cfg_e2e_a5/report.json`; that curve is the carry-forward reference, not a fresh measurement.

## 7. Honest framing — MEASURED vs PROJECTED

### MEASURED today
* GPU is not visible to the uv-managed Python 3.12 venv (`torch.cuda.is_available() = False`, `device_count = 0`).
* `rocminfo` reports `ROCk module is loaded` but the HSA user-mode runtime returns `HSA_STATUS_ERROR`.
* The CFM retrain script enforces the gate in two layers (probe + line-261 guard), so neither the diagnostic nor the full retrain can start.
* Estimated Vina-mean lift is **0.00 kcal/mol** — no compute was performed, so no lift can be claimed.

### PROJECTED (NOT measured)
* When GPU is back: at hidden_dim=64, batch=2, 10000 steps, the expected CFM convergence is roughly an order of magnitude more gradient updates than the 2000-step WF-2 baseline (5×). Memory hints that this could plausibly lift decode_ratio off zero (see wf2_final.md §3), but the only honest source of that number is the next run after GPU recovery.
* No honest estimate of Vina mean lift can be claimed before at least one measured `decode_ratio_after_10000_steps > 0`.

## 8. Recommendation

**Defer WF-CFM-Retrain-Full to the GPU-recovery worktree.** Spending 12–24 h of compute on a 6-seed × 64-pocket × 10000-step retrain when `torch.cuda.is_available() == False` would silently fall back to CPU and produce numbers that look plausible but are not what the workflow intends.

Concrete recovery work (carried forward from `wf_cfm_retrain_full_probe.md` §5):

1. Determine which `torch` wheel the uv venv actually has (`+ROCM` vs `+cpu`).
2. Export `ROCM_PATH`, `HSA_OVERRIDE_GFX_VERSION`, and `LD_LIBRARY_PATH` into the uv subprocess environment.
3. Verify `libhsa-runtime64.so.1` and `libamdhip64.so` are findable via `ldconfig`.
4. Verify the gfx1101 kernel module is loaded (`lsmod | grep amdgpu`); `rocminfo` should enumerate devices.
5. Check whether the prior `devoluciones/mm/python` env that worked in earlier rounds can be re-attached.

Until `torch.cuda.is_available()` returns True, **WF-CFM-Retrain-Full stays parked**.

## 9. Files written

* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_retrain_full/final.md` — this report.
* Phase-1 probe output already lives in `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_retrain_full_probe.md` (no new probe artefacts needed; the answer is unchanged from earlier today).
* No `run.log` / no `report.json` produced (script never started past line 261).

## 10. Metrics (schema)

```yaml
status: GPU_BLOCKED
decode_ratio_after_10000_steps: NOT_MEASURED
n_decoded: 0
n_finite: 0
n_requested: 0
bond_loss_final: NOT_MEASURED
n_params: NOT_MEASURED
estimated_vina_mean_lift: 0.00
baseline_comparison:
  wf_cfm_retrain_diagnose: NOT_MEASURED
  wf2_2000step_h32: 0/384
  targetdiff_cite_only: -8.45 kcal/mol
can_proceed: false
recommendation: defer to GPU-recovery worktree
```
