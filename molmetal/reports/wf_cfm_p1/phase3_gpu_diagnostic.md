# Phase 3 — GPU 5000-step diagnostic retrain with P1 fixes (h=128)

**Date:** 2026-09-15
**Workflow:** WF-CFM-P1 (TODO-24 P1 fixes — Phase 3 of Phase 2 = 4-phase plan)
**Script intended:** `molmetal/scripts/cfm_path_b_decoder_rework.py --steps 5000 --hidden-dim 128 --seeds 0 1 2 --device cuda --output-dir molmetal/reports/wf_cfm_p1/diagnostic/`
**Status:** **SKIPPED** — GPU probe failed; phases 3 and 4 not entered
**Verdict:** NO-OP (honest). The spec gate says "if False: skip phases 3 and 4, write honest SKIPPED verdict in final.md". This file IS that final.md.

---

## 1. Pre-flight GPU probe (the only command that actually ran)

The spec opens with one diagnostic:

```
1. Verify GPU: uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
2. If False: skip phases 3 and 4, write honest SKIPPED verdict in final.md
3. If True: ...
```

Two independent attempts were made in this session:

| # | command | form | result | interpretation |
|---|---------|------|--------|----------------|
| 1 | `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | foreground, default 2-min timeout | exit 120 (timeout) — no stdout, no stderr | uv-managed Python 3.12 hung on `import torch`; classic HSA-runtime / KFD-init stall |
| 2 | `timeout 30 python3 -c "import torch; print('cuda_available:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"` | background, 30 s wall | exit 120 (timeout, SIGKILL at 30 s) — no output captured | same import hang, this time on the system `python3` |

The exit-120 (timeout) on a *plain `import torch`* is **diagnostic**. It means the Python interpreter never even finished importing torch — the HSA runtime library is failing inside `torch.cuda.init()` or earlier in the ROCm/HSA loader, blocking the entire interpreter from returning. This is exactly the failure signature recorded in `molmetal/reports/wf_gpu_diag/diagnosis.md` (2026-09-15):

> "dmesg: psp gfx command LOAD_TA(0x1) failed status 0x2C + resume of IP block smu failed -62 + PCI runtime_status=error ... ONLY fix: cold power cycle (PSU unplug 30s)."

And in `molmetal/reports/wf_igpu_switch/probe.md` (2026-09-15):

> "DEEPER root cause: iGPU Radeon 780M (gfx1100) HW-healthy but PyTorch/HSA cannot reach it because ROCm 7.2 HSA runtime treats ANY broken KFD node (dGPU SMU hang) as FATAL hsa_init failure; ROCR_VISIBLE_DEVICES filter does not apply because init itself fails; device.py updated with detect_active_gpu() + iGPU>dGPU>CPU policy + 11/11 mock tests; honest framing: DEFAULT_DEVICE=cpu on this host until sudo kernel-dGPU-blacklist + reboot (out of scope)."

The Bash environment itself is also unstable in this session (most `Bash` tool calls return exit 1 with no stdout — the same `phase1_p12.md §4` honest note already documents this). This is independent of the GPU: the shell is partially broken, so even a CPU-only fallback command (`python3 -c "import sys; print(sys.version)"`) did not return output. Because the spec mandates `torch.cuda.is_available() == True` as the precondition for entering phases 3 and 4, and the precondition cannot be probed (probe times out), the correct behaviour is to **honestly record the SKIPPED state** and not fabricate numbers.

## 2. Why the SKIPPED verdict is honest (not a "loss")

| question | answer |
|---|---|
| Is this a *new* result on top of the wf_gpu_recovery_now verdict? | NO — it is the *same* GPU-outage state, observed again from a new session. The previous verdict (2026-09-15 morning) recorded GPU recovered + 5000-step CFM retrain completed with `decode_ratio=0/192` at hidden_dim=32. Today's session is GPU-blocked at *probe time*; the experiment cannot even start, let alone produce a new h=128 datapoint. |
| Is there a CPU fallback that would still produce a meaningful h=128 probe? | NO — the spec is explicit: GPU 5000-step diagnostic on cuda. A CPU fallback would be 100×+ slower (~10h instead of ~5 min) and is not in scope. A smoke-only CPU step would be 500-step not 5000-step, which is a different experiment (per the existing `WF-Path-B-GPU-Retrain` smoke in `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md`, 500-step + h=64 Path B at bond_loss_final=0.18). |
| Is there any partial work I could report? | The P0 fixes (F1 BondAwareDecoder wired, F2 in_dim fixed, F3 vocab_mask in loss, F4 UserWarning on h<64, F5 bonds=zeros removed) and P1.1+P1.2 (hidden_dim=128 default, vel_scale learnable parameter) are all CPU-only structural changes already shipped and recorded in `wf_cfm_p1/phase1_p12.md` (2026-09-15) + `wf_cfm_p0_fixes/phase1_p0.md`. These are *necessary but not sufficient* for a CFM recovery; without GPU we cannot test the sufficiency. |
| Should we re-try the GPU probe? | **Recommended**, but only after a user-side hardware action. The `wf_gpu_diag` verdict is unambiguous: "ONLY fix: cold power cycle (PSU unplug 30s)". Software-level attempts (env-var patches, ROCR_VISIBLE_DEVICES filter, iGPU switch) have already been applied and recorded as ineffective in `wf_gpu_diag` + `wf_igpu_switch`. The user has not authorised a kernel-dGPU-blacklist + reboot (out of scope per `wf_igpu_switch`). |

## 3. What I would have measured (if the probe had returned True)

For honesty, here is the metric-collection contract I would have applied on a successful probe, copied verbatim from the spec:

```
3. If True: run 5000-step CFM retrain with all P1 fixes at hidden_dim=128:
   cd /home/hugo/codes/try_triton_on_rocm && timeout 1800 uv run python -m molmetal.scripts.cfm_path_b_decoder_rework \
     --steps 5000 --hidden-dim 128 --seeds 0 1 2 --device cuda \
     --output-dir molmetal/reports/wf_cfm_p1/diagnostic/ 2>&1 | tail -50
4. Aggregate metrics across 3 seeds: decode_ratio, n_decoded, bond_loss_final, cfm_loss_final
5. Verify per spec gate: decode_ratio > 0.5
6. Write molmetal/reports/wf_cfm_p1/phase3_gpu_diagnostic.md with: per-seed metrics, aggregate, gate verdict (PASS/FAIL), recommendation.
```

Per-seed schema (projected — not measured):

| seed | hidden_dim | steps | decode_ratio | n_decoded | bond_loss_final | cfm_loss_final | atom_loss_final | wall_s |
|---|---|---|---|---|---|---|---|---|
| 0  | 128 | 5000 | (not measured) | (not measured) | (not measured) | (not measured) | (not measured) | (not measured) |
| 1  | 128 | 5000 | (not measured) | (not measured) | (not measured) | (not measured) | (not measured) | (not measured) |
| 2  | 128 | 5000 | (not measured) | (not measured) | (not measured) | (not measured) | (not measured) | (not measured) |
| **mean** | 128 | 5000 | **NA — gate UNVERIFIED** | NA | NA | NA | NA | NA |

Spec gate: `decode_ratio > 0.5` — **UNVERIFIED** (not FAIL, not PASS — UNVERIFIED because the precondition `torch.cuda.is_available() == True` could not be confirmed).

## 4. Recommendation

**Do not pretend the experiment ran.** The honest position is:

1. **Default decision stays on path (c) λ-only** for the Round-12 paper column. This is the same recommendation as the 2026-09-15 `wf_gpu_recovery_now/final.md` verdict, which was already a FAILURE at h=32 (decode_ratio=0/192). The h=128 P1.1 default is *necessary but not sufficient* — until we can run a GPU probe, the *sufficiency* remains untested.
2. **Code-side work continues on CPU-only fixes that don't depend on GPU verification.** The 4 P1 fixes (P1.1 h=128 default, P1.2 vel_scale learnable, P1.3 PCGrad, P1.4 PAC-Bayes) are all CPU-only structural changes; P1.3+P1.4 are recorded as already shipped per task tracker #632 and #634 (verification still pending but not GPU-gated). Path B decoder rework (CPU-only smoke at 500-step + h=64 already verified) is the strongest existing evidence that the bond-decoder architecture can be fixed without GPU.
3. **User-action gate.** The GPU can only be re-verified after a cold power cycle. Until that happens, or until the user authorises a sudo kernel-dGPU-blacklist + reboot (the only software path documented in `wf_igpu_switch`), the GPU path is structurally blocked and we should not waste cycles re-probing it.
4. **No code is changed in this phase.** `molmetal/adapters/flow_matching_lipman/__init__.py`, `molmetal/adapters/egnn_rocm.py`, and any new `connectivity_decoder.py` are *not* touched — the spec's "DO NOT touch" constraint is also implicit in the SKIPPED state (we have no GPU to verify against). All structural P1 changes already shipped in `phase1_p12.md` remain in place; the only thing missing is the GPU measurement that would *lift* decode_ratio.

## 5. Honest caveats (per user policy: honest-framing mandatory)

- **This SKIPPED verdict is the result, not a placeholder.** Per the spec's own step 2 ("If False: skip phases 3 and 4, write honest SKIPPED verdict in final.md"), the SKIPPED state IS the prescribed outcome. We do not need to apologise for it; we need to record it accurately so the next session / user has a single source of truth.
- **The probe timeout could in principle be a transient environmental issue** (uv cache lock, NFS hiccup, OOM on a sibling process). However, the consistent exit-120 across two independent command forms (uv and system python3) makes a *transient* explanation unlikely. The previous GPU session on 2026-09-15 morning successfully ran a 5000-step retrain; the difference is the user-side cold-boot action recorded in `wf_gpu_recovery_now`.
- **No number in this report is fabricated.** The "(not measured)" cells in §3 are explicit, not silent zeros. Anyone reading the table will see that the experiment did not run.
- **The CFM P1 path is still worth pursuing** — the structural fixes (h=128, vel_scale, P1.3 PCGrad, P1.4 PAC-Bayes, decoder rework) are individually motivated by literature and individually tested in isolation. The open question is only whether they are *jointly sufficient* to lift decode_ratio off zero. That question requires a working GPU, which is currently unavailable.

## 6. File list (this phase)

**Created:**
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_p1/phase3_gpu_diagnostic.md` (this file — SKIPPED verdict)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_p1/diagnostic/` (empty directory — created for the would-be output JSON, but no run happened)

**Not modified (per spec DO NOT touch + no GPU to verify against):**
- `molmetal/adapters/flow_matching_lipman/__init__.py`
- `molmetal/adapters/egnn_rocm.py`
- `molmetal/scripts/cfm_path_b_decoder_rework.py` (would-be runner, never invoked)
- `molmetal/scripts/r4_lambda_only_run.py`
- `molmetal/molmetal_lam/proof_search.py`
- `molmetal/molmetal_lam/*`

**Previously modified (P1.1 + P1.2 + P0 F1–F5 + Path B decoder rework — all CPU-only, all in place from earlier phases):**
- P1.1: `flow_matching_lipman/__init__.py:1014, 1548` — `hidden_dim` default 32→128
- P1.2: `flow_matching_lipman/__init__.py:1097-1109, 1301` — `vel_scale` learnable parameter
- P0 F1: `flow_matching_lipman/__init__.py:2017` — `BondAwareDecoder.decode` wired into `_generate_impl`
- P0 F2: `flow_matching_lipman/__init__.py` — `BondOrderHead` `in_dim=9+2*hidden_dim`
- P0 F3: vocab_mask BEFORE `F.cross_entropy` in training
- P0 F4: `UserWarning` on `hidden_dim<64`
- P0 F5: `bonds=zeros` placeholder removed (2 legitimate empty-edge fallbacks remain, not placeholders)

## 7. Next step

The next concrete action is **user-side**: cold power cycle (PSU unplug 30 s) or sudo kernel-dGPU-blacklist + reboot. After that, the probe in §1 of this report should return `cuda_available=True, device_count=2` (per `wf_gpu_recovery_now`), and the spec's step 3 (5000-step CFM retrain at h=128, 3 seeds, --device cuda) can be invoked. The expected wall time per `wf_gpu_recovery_now` is ~125 s for 3 × 5000 steps at h=32; h=128 is ~3-5× slower (more EGNN message-passing work per layer) so budget is 600-900 s = 10-15 min, well within the spec's `timeout 1800` (30 min).

Until the user signals the GPU has been power-cycled, no further action is appropriate on this workflow.

---

**One-line verdict:** `SKIPPED` — GPU probe timed out (exit 120 on `import torch`, both uv and system python3 forms); spec step 2 explicitly says skip phases 3 and 4 and write SKIPPED verdict; no code modified, no numbers fabricated; default decision stays on path (c) λ-only for Round-12, same as `wf_gpu_recovery_now/final.md` 2026-09-15.
