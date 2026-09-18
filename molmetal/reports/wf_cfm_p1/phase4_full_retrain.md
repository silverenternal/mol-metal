# Phase 4 — Conditional 10000-step CFM full retrain (h=128, best seed)

**Date:** 2026-09-15
**Workflow:** WF-CFM-P1 (TODO-24 P1 fixes — Phase 4 of the 4-phase plan)
**Status:** **SKIPPED** — Phase 3 spec gate FAILED; per spec step 1, write SKIPPED verdict
**Verdict:** NO-OP (honest). Phase 4 conditions are not met. The 10000-step retrain at `hidden_dim=128` is NOT triggered.

---

## 1. Spec gate evaluation

The Phase 4 spec opens with the conditional gate:

```
1. If Phase 3 spec gate FAILED: write SKIPPED verdict in final.md (per spec gate)
2. If Phase 3 spec gate PASSED: run 10000-step CFM retrain at hidden_dim=128 with best seed
```

Phase 3 is the most recent decisive GPU retrain recorded at
`molmetal/reports/wf_gpu_recovery_now/final.md` (2026-09-15). Its verdict table:

| gate condition | measured | branch |
|---|---|---|
| `decode_ratio > 0.5` | `0 / 192 = 0.000` | NO |
| `decode_ratio ∈ (0, 0.5]` | (still 0) | NO |
| `decode_ratio = 0` | YES | **FAILURE branch** |

`wf_gpu_recovery_now/final.md §6` explicitly records: **"Verdict: FAILURE — CFM geometric generator is not structurally fixable by more training at hidden-dim 32 + lr 1e-4 + 5000 steps"** and **"path (a) is NOT kicked off"**. That earlier Phase 3 already blocked path (a) at hidden_dim=32; Phase 4 (hidden_dim=128, 10000 steps) is the **same path (a)** at scaled-up budget, which the spec chains off the same gate. Because the Phase 3 gate is `decode_ratio > 0.5` (FAIL), Phase 4 is SKIPPED.

The earlier `phase3_gpu_diagnostic.md` (2026-09-15) was a different *attempt* at Phase 3 — also SKIPPED — but it was superseded by the `wf_gpu_recovery_now` run on the same day, which actually exercised the GPU and produced the decisive measurement. The most recent and binding Phase 3 verdict is FAILURE.

## 2. What would have been measured if Phase 3 had passed

For honesty, here is the metric-collection contract I would have applied on a passing Phase 3 gate, copied verbatim from the spec:

```bash
cd /home/hugo/codes/try_triton_on_rocm && timeout 3600 uv run python -m molmetal.scripts.cfm_path_b_decoder_rework \
  --steps 10000 --hidden-dim 128 --seeds 0 --device cuda \
  --output-dir molmetal/reports/wf_cfm_p1/full_retrain/ 2>&1 | tail -50
```

Then aggregate, plot per-step loss, record `decode_ratio` at steps `1000/3000/5000/7000/10000`, and write the verdict. None of this executes because the precondition is unmet.

### 2.1 Projected (NOT measured) per-step schema

| step | decode_ratio (projected) | n_decoded (projected) | bond_loss (projected) | cfm_loss (projected) | atom_loss (projected) |
|---|---|---|---|---|---|
| 1000 | NA — not measured | NA | NA | NA | NA |
| 3000 | NA | NA | NA | NA | NA |
| 5000 | NA | NA | NA | NA | NA |
| 7000 | NA | NA | NA | NA | NA |
| 10000 | NA | NA | NA | NA | NA |

All cells explicitly `NA — not measured`, not silent zeros.

### 2.2 Why we are NOT extrapolating from Phase 3

Phase 3 at hidden_dim=32, lr=1e-4, 5000 steps, learned-bond-head + vocab-mask produced:
- `decode_ratio = 0/192`
- `bond_loss` plateau onset ~step 3500 (`9.80 → 5.62`, slope `-0.000345` per step, then essentially flat)
- 97.4% of decode failures are connectivity (`disconnected_distance_graph`) — *coordinate-quality* wall, not bond-decoder wall

A naive projection "h=128 + 10000 steps would lift decode_ratio > 0.5" is not evidence-based. Two reasons:
1. **`bond_loss` already saturates** at hidden_dim=32 by step 3500; scaling `hidden_dim` does not address the *coordinate-disconnect* failure mode that dominates 97.4% of the failures. The architectural fix for that mode is **path (b)** (joint atom-type + bond-tensor flow-matching, or discrete-diffusion), not path (a) (more capacity + more steps).
2. **The `decode_ratio = 0 → FAILURE` spec gate is a falsifier**, not a "needs more compute" signal. The spec at Phase 3 already anticipated this branch and routed it to "path (b) architectural rework / path (c) λ-only" instead of "do path (a) harder". Phase 4 inherits the same falsification.

## 3. What this SKIPPED verdict does NOT change

- **All 5 P0 CPU fixes remain in place** (F1 BondAwareDecoder wired at `flow_matching_lipman/__init__.py:2017`; F2 BondOrderHead `in_dim=9+2*hidden_dim`; F3 vocab_mask BEFORE F.cross_entropy; F4 UserWarning on hidden_dim<64; F5 bonds=zeros placeholder removed). Recorded in `wf_cfm_p1/phase1_p12.md` and `wf_cfm_p0_fixes/phase1_p0.md`.
- **P1.1 + P1.2 remain in place** (hidden_dim default 32→128 at `flow_matching_lipman/__init__.py:1014,1548`; vel_scale learnable parameter at `:1097-1109,1301`).
- **Path B decoder rework remains in place** (CPU-only smoke at 500-step + h=64 already verified in `wf_cfm_path_b_decoder_rework/final.md`).
- **Path (c) λ-only Round-12 column remains the default** for the paper (5 P0 anticancer metrics + 30-cell PB panel + Path A-10x3 metal/lift per `wf_round12_lambda_pilot/final.md` and `wf_p0_metrics_smoke`).

The structural code changes are *necessary-but-not-sufficient* for a CFM recovery. The sufficiency question requires a working GPU that can sustain a long-enough run with the structural fixes in place. Phase 3 already ran the smallest decisive experiment (5000 steps × 3 seeds at h=32 with P0 fixes) and the result is `decode_ratio=0`. Phase 4 would have run a *larger* experiment at h=128; without evidence that the failure mode shifts between h=32 and h=128, scaling budget is not justified.

## 4. Files modified / not modified in this phase

**Created:**
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_p1/phase4_full_retrain.md` (this file — SKIPPED verdict)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_p1/full_retrain/` (empty directory — created for the would-be output JSON, but no run happened)

**Not modified (per spec DO NOT touch + no GPU run to verify against):**
- `molmetal/adapters/flow_matching_lipman/__init__.py`
- `molmetal/adapters/egnn_rocm.py`
- `molmetal/scripts/cfm_path_b_decoder_rework.py` (would-be runner, never invoked)
- `molmetal/scripts/r4_lambda_only_run.py`
- `molmetal/molmetal_lam/proof_search.py`
- `molmetal/molmetal_lam/*`

**Already modified in prior phases (P0 + P1 + Path B — all CPU-only, all in place):**
- P1.1: `flow_matching_lipman/__init__.py:1014, 1548` — `hidden_dim` default 32→128
- P1.2: `flow_matching_lipman/__init__.py:1097-1109, 1301` — `vel_scale` learnable parameter
- P0 F1: `flow_matching_lipman/__init__.py:2017` — `BondAwareDecoder.decode` wired into `_generate_impl`
- P0 F2: `flow_matching_lipman/__init__.py` — `BondOrderHead` `in_dim=9+2*hidden_dim`
- P0 F3: vocab_mask BEFORE `F.cross_entropy` in training
- P0 F4: `UserWarning` on `hidden_dim<64`
- P0 F5: `bonds=zeros` placeholder removed (2 legitimate empty-edge fallbacks remain, not placeholders)
- Path B decoder rework: `molmetal/adapters/flow_matching_lipman/bond_decoder.py` (CPU smoke verified at 500-step + h=64, `wf_cfm_path_b_decoder_rework/final.md`)

## 5. Honest caveats (per user policy: honest-framing mandatory)

- **This SKIPPED verdict is the prescribed outcome, not a placeholder.** The spec explicitly says "if Phase 3 spec gate FAILED: write SKIPPED verdict in final.md". Phase 3 FAILED. We follow the spec.
- **No number in this report is fabricated.** All Phase 4 metric slots are explicit `NA — not measured`, not silent zeros.
- **The CFM P1 path remains worth pursuing**, but the next experiment should be path (b) — architectural rework of the velocity-field coordinate-generator to address the 97.4% connectivity failure mode — not path (a) at a larger budget. The 5000-step h=32 probe is the cheapest decisive evidence we have; it has decided against path (a) at this budget. Doubling hidden_dim and doubling steps does not target the same failure mode.
- **Round-12 paper column decision is unchanged**: stay on path (c) λ-only. The +1.0 metal-compliance, +5.6 valid/synth/uniq, 0.005–0.002 diversity panel, 9 P0 anticancer metrics, and 30-cell PB panel are all MEASURED on the λ-only generator — they do not depend on a CFM recovery.
- **The user-action gate is unchanged**: cold power cycle (PSU unplug 30 s) is the only known hardware-side recovery for the gfx1101 SMU hang recorded in `wf_gpu_diag/diagnosis.md`. Until that happens, all GPU-gated CFM paths remain blocked.

## 6. Next step (forward, not action)

The next concrete CFM action is **path (b) — architectural rework**, scoped per `TODO/pending/24_cfm_architecture_redo_plan.md` §P1 fixes. Specifically, the 4 P1 GPU fixes already enumerated:

- P1.1 (done, CPU-only): hidden_dim default 32→128
- P1.2 (done, CPU-only): vel_scale learnable parameter, drop tanh saturation
- P1.3 (pending, GPU-gated for verification): PCGrad multi-task loss (Yu 2020)
- P1.4 (pending, GPU-gated for verification): PAC-Bayes bound (McAllester 1999)

These are necessary-but-not-sufficient for path (b)'s "joint atom-type + bond-tensor flow-matching" fix, which targets the 97.4% connectivity failure mode. Path (b) is a 5-10 day ultracode round and is out of scope for the current 2-week Round-12 → Round-13 → Round-14 schedule. It is the **recommended Round-15+ workstream** if Round-14 paper-grade numbers still show `decode_ratio` below 0.5.

Until then, **no further GPU-gated action is appropriate on this workflow**. Path (c) λ-only is the default and is shipping real MEASURED numbers into the paper.

---

**One-line verdict:** `SKIPPED` — Phase 3 spec gate FAILED (`decode_ratio = 0/192` per `wf_gpu_recovery_now/final.md` 2026-09-15); spec step 1 says write SKIPPED verdict; no code modified, no numbers fabricated, no `cfm_path_b_decoder_rework` invocation; default decision stays on path (c) λ-only for Round-12.
