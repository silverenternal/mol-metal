# WF-CFM-Retrain-Diagnose — Verdict

**Date:** 2026-09-14 (UTC)
**Verdict:** **FAILURE (provisional — GPU-side blocker, not CFM-side)**
**Author:** WF-CFM-Retrain-Diagnose verify pass
**Honest framing:** MEASURED = current blocker; PROJECTED = forward plan only.

---

## 1. Diagnostic intent (recap)

| question | metric | baseline (2000-step) | goal (5000-step) |
|---|---|---:|---:|
| does CFM learn pocket-conditioned placement with more steps? | `decode_ratio = n_decoded / n_requested` | 0 / 384 | > 0.5 → SUCCESS; ∈ [0, 0.5] → PARTIAL; = 0 → FAILURE |
| is bond loss still decreasing? | `bond_loss_trajectory_slope` at 5000 steps | not measured at 2000-step (inconclusive) | slope ≠ 0 → more training helps |

Spec: `--train-steps 5000 --hidden-dim 32` vs original `--train-steps 2000 --hidden-dim 32` on the same CrossDocked pocket, 6 seeds × 2 pockets × 2 cfg × 16 samples = 384 samples.

## 2. MEASURED — decode_ratio at 5000-step

| metric | value | source |
|---|---:|---|
| `decode_ratio_after_5000_steps` | **NOT MEASURED** | script aborts at line 261 (`RuntimeError('Real ROCm GPU required')`) |
| `torch.cuda.is_available()` | **False** | `uv run python -c "import torch; print(torch.cuda.is_available())"` (2026-09-14) |
| `torch.cuda.device_count()` | 0 | same |
| `rocminfo` | `HSA_STATUS_ERROR` | `/opt/rocm/bin/rocminfo` (ROCk module loaded but HSA init fails) |
| `rocm-smi` | device 0 = 0x747e low-power N/A; device 1 = 0x1900 perf=auto 61°C 37W 85% VRAM (compute idle) | `/opt/rocm/bin/rocm-smi` |
| `n_decoded` (this run) | 0 (trivially — harness never produced samples) | not measured |
| `n_finite` (this run) | 0 | not measured |
| `n_requested` (this run) | 0 | not measured |
| `decode_ratio` comparison vs baseline 2000-step | **n/a** — same 0/0 vs 0/384 trivially; the 5000-step sample production was never reached | — |
| `bond_loss_initial` (this run) | **NOT MEASURED** | n/a (no GPU) |
| `bond_loss_final` (this run) | **NOT MEASURED** | n/a (no GPU) |
| `bond_loss_slope` | **NOT MEASURED** | n/a |

**Reference datapoint (MEASURED earlier today):** baseline 2000-step in `molmetal/reports/wf2_cfg_e2e_a5/report.json` gave `decode_ratio = 0/384`, `total loss` trajectory 7.37 → 5.82 (decreasing but decode still 0), exit 0 at `2026-09-14T03:22:11Z`. The same hardware worked 4 hours earlier and is non-functional now — system rebooted since, no sudo, cannot reset ROCk module.

## 3. Verdict — provisional FAILURE with high uncertainty

Strictly applying the spec (`if decode_ratio = 0: FAILURE — confirm CFM geometric generator is structurally broken`), today's result reads FAILURE because the run produced no samples at all (0/0 trivially).

**But this is a GPU-side failure, not a CFM-side failure.** The 5000-step retrain was never executed; the CFM hypothesis ("does more training lift decode_ratio?") is **untested**, not falsified. Strictly speaking, with no measurements we cannot decide between Round-12 paths (a) full retrain, (b) architectural rework, (c) lambda-only column.

| Round-12 path | today's evidence | can ship now? |
|---|---|---|
| (a) full retrain @ 10000-step + hidden-dim 64 | **UNTESTED** | NO — needs GPU; re-run 5000-step diagnostic first |
| (b) CFM architectural rework | **UNTESTED** | NO — depends on 5000-step result landing ∈ [0, 0.5] |
| (c) lambda-only column for Round-12 | **UNBLOCKED by GPU** | YES — λ-only path does not need GPU; ready per WF-Lambda-1/1b/1c/2/3/4 |

## 4. bond_loss slope — NOT MEASURED

The 2000-step baseline shows loss decreasing (7.37 → 5.82, ~20% drop) but decode_ratio remained 0/384. From the 2000-step trajectory alone, the loss is **still decreasing at 2000 steps** — strong evidence that **more training would help geometric quality**. But the *bridge from low loss → successful decode* is the unknown: at the 2000-step horizon, lower loss did not translate to any decoded molecule. We cannot measure the 5000-step slope today.

**PROJECTED** (not measured): if the 5000-step diagnostic could run, the loss curve is expected to extend the 7.37→5.82 descent (plateau not yet visible at 2000 steps). The decode_ratio remains the critical datapoint, not the loss slope.

## 5. Recommendation

1. **Default to (c) lambda-only column for Round-12 today** — no GPU needed; λ-only path is verified (WF-Lambda-1/1b/1c/2/3/4), produces validity=90% signals, ships without ROCm. This matches wf2_final.md §"next steps".
2. **Defer (a) and (b) until GPU is back** — both depend on a successful 5000-step diagnostic, which itself is blocked by HSA init. The 10000-step full retrain (path a) and architectural rework (path b) are *incompatible* with the current hardware state.
3. **When GPU returns, re-run this exact 5000-step diagnostic first** (smallest experiment that can answer the CFM question). Same `uv run` command from `molmetal/reports/wf_cfm_diagnose/final.md` §2 should work. Then escalate to path (a) at 10000-step + hidden-dim 64 if `decode_ratio > 0.5`; otherwise path (b) architectural rework.
4. **Bond-loss interpretation must be honest** — the 2000-step loss decreased 21% but decode still 0. More training may be *necessary but not sufficient*. After the 5000-step run, the bond_loss_slope vs decode_ratio relationship becomes the deciding evidence for path (a) vs (b).

## 6. Follow-up budget (PROJECTED, not measured)

| path | GPU hours | calendar time | blocker |
|---|---:|---:|---|
| re-run 5000-step diagnostic when GPU is back | 0.5 h | ≤1 h wall | HSA init — see preflight.log |
| (a) full retrain 10000-step + hidden-dim 64 (single pocket) | 6-12 h | 1 day | GPU + decision from 5000-step diagnostic |
| (b) CFM architectural rework | 5-10 d effort (1 ultracode round) + retrain | 1-2 weeks | 5000-step decode_ratio ∈ [0, 0.5] |
| (c) λ-only column (current default) | 0 h GPU | already shipped | none |

**Default for Round-12: (c) λ-only column, 0 GPU hours needed.** Path (a) and (b) deferred until GPU recovers.

## 7. What changed since `wf_cfm_diagnose/final.md`

- `final.md` was the WF body run report. This `verdict.md` is the *decision artifact*:
  - **verdict** = FAILURE (provisional, GPU-side)
  - **recommendation** = (c) λ-only column for Round-12, defer (a)+(b) until GPU returns
  - **follow-up budget** = 0 h GPU now; 0.5 h diagnostic on GPU recovery; 6-12 h if (a) is escalated
  - **next experiment** = the same 5000-step `uv run` command, gated on `torch.cuda.is_available() == True`

The diagnostic question — "does CFM learn pocket-conditioned placement at higher step count?" — **remains open**. It can only be answered with a working GPU. The lambda-only fallback path is the responsible default in the meantime.

## 8. Cross-references

- `molmetal/reports/wf_cfm_diagnose/final.md` — WF body run report (NOT RUN status)
- `molmetal/reports/wf_cfm_diagnose/preflight.log` — GPU/HSA runtime probe
- `molmetal/reports/wf2_cfg_e2e_a5/report.json` — 2000-step baseline (decode_ratio = 0/384)
- `TODO/pending/21_lambda_model_coupling.md` — Lambda × CFM coupling TODO (to be updated)
- `TODO/pending/11_algorithm_strengthening_r10.md` — Round-10 axes C (CFG) and D (Pt prior) (to be appended)
