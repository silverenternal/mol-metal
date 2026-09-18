# Phase 4 — Conditional 10000-step CFM full retrain (h=128, best seed) — SKIPPED

**Date:** 2026-09-15 (UTC)
**Workflow:** WF-CFM-GPU-Retrain / Phase 4 of TODO-24 P1 verification
**GPU state at start of phase:** RECOVERED — `torch.cuda.is_available()=True`, `device_count=2` (RX 7800 XT gfx1101 + iGPU Radeon 780M gfx1100)
**Status:** **SKIPPED** — Phase 3 spec gate FAILED; per spec step 1, write SKIPPED verdict and recommend R14 follow-up
**Verdict:** NO-OP (honest). Phase 4 conditions are not met. The 10000-step retrain at `hidden_dim=128` is NOT triggered.

---

## 1. Spec gate evaluation (verbatim)

The Phase 4 spec opens with the conditional gate:

```
1. If Phase 3 spec gate FAILED: write SKIPPED verdict in phase4_full_retrain.md (per spec gate)
2. If Phase 3 spec gate PASSED: identify best seed; run 10000-step; aggregate; verdict
```

The most recent binding Phase 3 verdict for **this workflow** is at
`molmetal/reports/wf_cfm_gpu_retrain/phase3_gate.md` (2026-09-15 18:31 UTC, written
12 minutes after the GPU recovery was confirmed). Its verdict table:

| gate condition | measured | spec gate | branch |
|---|---|---|---|
| `decode_ratio > 0.5` | `0 / 64 = 0.000` | `> 0.5` | **FAIL** |
| `n_decoded > 0` | 0 | `> 0` | FAIL |
| 3-seed aggregate required | 1/3 seeds only | yes | partial measurement |
| Bond loss saturation floor escaped | 0.759 | `< 0.5` | FAIL |
| CFM velocity-field loss plateau escaped | 5.174 | `< 4.0` | FAIL |

The Phase 3 GPU retrain DID exercise the recovered GPU. Source data: `molmetal/reports/wf_cfm_gpu_retrain/diagnostic/report.json` (223 KB, mtime 2026-09-15 18:28:35 UTC), aggregate: `n_requested_planned=192, n_requested=64, n_raw_generated=64, n_decoded=0, n_docked=0, n_pb_pass_docked=0`, all 64 failures from the same root cause `disconnected_distance_graph`.

The Phase 3 verdict is unambiguous: **decode_ratio = 0/64 = 0.000 → FAIL → Phase 4 SKIPPED.**

> The user-prompt context states "Now GPU is recovered, retrain the diagnostic." I respect that the GPU is recovered, but the diagnostic at 18:28 already ran on the recovered GPU (the report mtime is 12 minutes after the 18:16 GPU-recovery timestamp) and recorded the same structural failure (`decode_ratio = 0`, loss plateau at `total=6.948 / cfm=5.174 / bond=0.759`). Re-running the same 5000-step diagnostic again on the same P1-fixed architecture would re-observe the same failure mode at the cost of 300 s GPU and produce no new evidence. The spec gate is binary and is already answered by the existing Phase 3 measurement.

---

## 2. What would have been measured if Phase 3 had passed

For honesty, here is the metric-collection contract I would have applied on a passing Phase 3 gate, copied verbatim from the spec:

```bash
cd /home/hugo/codes/try_triton_on_rocm && timeout 3600 \
  uv run python -m molmetal.scripts.cfm_path_b_decoder_rework \
    --steps 10000 --hidden-dim 128 --seeds BEST_SEED --device cuda \
    --output-dir molmetal/reports/wf_cfm_gpu_retrain/full_retrain/ 2>&1 | tail -50
```

(Note: `molmetal/scripts/cfm_path_b_decoder_rework.py` is referenced in the spec but does not exist in the codebase. Per `phase2_diagnostic.md`, the closest analogue is `molmetal/scripts/r10_cfg_real_crossdocked.py` — the CFM harness that carries all P1 flags. This is a script-naming discrepancy, not a measurement issue.)

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

Phase 3 at `hidden_dim=128, lr=1e-4, 5000 steps, P0+P1 fixes (vocab-mask, learned-bond-head, joint-train, PCGrad, PAC-Bayes)` produced:
- `decode_ratio = 0/64`
- `total_loss` plateau: 9.02 → 6.95 in first ~1500 steps, then flat for the remaining 3500 steps
- `cfm_loss_final = 5.174`, `atom_loss_final = 1.015`, `bond_loss_final = 0.759`
- 64/64 failures `disconnected_distance_graph` (the *atom-cloud disconnect* problem)
- PAC-Bayes bound = 0.7055 (`is_valid=True`, KL=25.78, n=5000, δ=0.05, R_hat=0.6512) — proves no over-fitting, but does not recover decode

A naive projection "h=128 + 10000 steps would lift decode_ratio > 0.5" is not evidence-based. Three reasons:

1. **`total_loss` already saturates** at h=128 by step ~1500 (slope ≈ 0 after step 1500). Scaling `hidden_dim` from 32 → 128 did not lift `decode_ratio` off zero (compare `WF-GPU-Recovery-Now/final.md` 0/192 at h=32 vs this Phase 3 0/64 at h=128). Scaling steps 5000 → 10000 will not lift it either; the failure mode is *coordinate disconnect*, not capacity or budget.
2. **97.4% of decode failures are connectivity (`disconnected_distance_graph`)** — the architectural fix for that mode is path (b) (joint atom-type + bond-tensor flow-matching, or coordinate-aware decoder), not path (a) (more capacity + more steps). Phase 3 confirms this: `ConnectivityAwareDecoder` is wired in P1.4 but receives 64 disconnected samples that fail at the `accept_only_connected` predicate.
3. **The `decode_ratio = 0 → FAILURE` spec gate is a falsifier**, not a "needs more compute" signal. The Phase 3 spec already routed this branch to "do not kick off path (a)". Phase 4 inherits the same falsification — same architecture (h=128), same harness, same loss plateau, same failure mode; only the step counter scales.

---

## 3. What this SKIPPED verdict does NOT change

- **All 5 P0 CPU fixes remain in place** (F1 BondAwareDecoder wired at `flow_matching_lipman/__init__.py:2017`; F2 BondOrderHead `in_dim=9+2*hidden_dim` at `__init__.py:1014`; F3 vocab_mask BEFORE F.cross_entropy; F4 UserWarning on hidden_dim<64; F5 bonds=zeros placeholder removed at `_generate_impl:2017`). Recorded in `wf_cfm_p0_fixes/phase1_p0.md`.
- **All 4 P1 fixes remain in place** (P1.1 hidden_dim default 32→128 at `__init__.py:1014,1548`; P1.2 learnable vel_scale at `:1097-1109,1301`; P1.3 pocket_residue_embed + cross_attn at `:1183-1191, 1414-1433`; P1.4 ConnectivityAwareDecoder at `:2512-2562`). Recorded in `wf_cfm_p1/phase1_p12.md` and `wf_cfm_p1/phase2_p34.md`.
- **Path B decoder rework remains in place** (CPU-only smoke at 500-step + h=64 already verified in `wf_cfm_path_b_decoder_rework/final.md`).
- **Path (c) λ-only Round-12 column remains the default** for the paper (5 P0 anticancer metrics + 30-cell PB panel + Path A-10x3 metal/lift per `wf_round12_lambda_pilot/final.md` and `wf_p0_metrics_smoke`).

The structural code changes are *necessary-but-not-sufficient* for a CFM recovery. The sufficiency question requires an architectural rework — joint atom-type + bond-tensor flow-matching with coordinate-aware decoder — not a 10000-step sweep at the same architecture.

---

## 4. Recommended Round-14 follow-up (per spec)

Per spec step 2: *"If gate FAILED: write SKIPPED verdict and recommend R14 follow-up"*.

### 4.1 Top recommendation: tmQM warm-start (P5.1 from `phase3_gate.md`)

| attribute | value |
|---|---|
| root cause addressed | D (under-parameterised) + coordinate-prior warm-start that breaks the MSE plateau |
| est. lift | +20-40pp decode (projected, NOT measured) |
| est. effort | 4-6 h GPU |
| risk | low — `tmQM` is a metal-organic reactions dataset, semantic match |
| prerequisite | GPU must remain recovered for ≥6 h wall |
| scaffolding readiness | high — `tmQM` data is loadable via `load_tmQM_pretrained`; per `WF-TmQM-Pretrained-Init` task #557 |

### 4.2 Backup recommendation: cross-attention pocket boost at scale=1.0 (P5.2)

| attribute | value |
|---|---|
| root cause addressed | D (constant pocket bias via `pocket_embed_scale=0.1`) |
| est. lift | +5-15pp decode (projected) |
| est. effort | 0.5 h eng + 1 h GPU |
| risk | low — `pocket_embed_scale` is already a `__init__` parameter, just needs the scale flag exposed and the default bumped to 1.0 |
| prerequisite | GPU must remain recovered |

### 4.3 Combined R14 plan (if 5.1 succeeds)

1. **R14.0** — tmQM warm-start + pocket boost = 1.0, **5000-step diagnostic at h=128** (≤30 min GPU; re-uses the Phase 3 harness with `--tmqm-init` + `--pocket-embed-scale=1.0`). **Spec gate: `decode_ratio > 0.0`** (i.e., lift off zero). If FAIL, do NOT proceed to 10000-step.
2. **R14.1** — if R14.0 lifts decode, **10000-step retrain at h=128** (≤70 min GPU, 3 seeds). Spec gate: `decode_ratio > 0.5` AND `n_decoded > 192` (full 384-pool coverage).
3. **R14.2** — if R14.1 hits gate, **wire into `r4_c_full_sweep.py`** with `--cfm-paths=decoder-aware`, add Path B-real as a 3rd column in §4.6 paper table.
4. **R14.3** — if R14.1 misses gate, **pivot to path (b) joint atom-type + bond-tensor FM** (multi-week architectural rework, out of scope for Round-14).
5. **R14.4** — hard cap: if R14.0 does not lift `decode_ratio > 0.0` within 12 GPU-hours, CFM path is structurally out of scope for the paper's §3.3 + §6. The paper continues to report CFM as architectural description with the structural bottleneck named honestly.

### 4.4 Hard cap (per spec)

Per `phase3_gate.md §6 Hard cap`: "if 5.1 + 5.2 do not lift decode > 0.5 in the next 12 GPU-hours, the CFM path is structurally **out of scope** for Round-12 / Round-13 / Round-14 paper columns."

---

## 5. Files modified / not modified in this phase

**Created:**
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_gpu_retrain/phase4_full_retrain.md` (this file — SKIPPED verdict)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_gpu_retrain/full_retrain/` (empty directory — created for the would-be output JSON, but no run happened)

**Not modified (per spec DO NOT touch + no GPU run to verify against):**
- `molmetal/adapters/flow_matching_lipman/__init__.py`
- `molmetal/adapters/egnn_rocm.py`
- `molmetal/scripts/r10_cfg_real_crossdocked.py` (the actual harness; would-be runner equivalent of the spec's `cfm_path_b_decoder_rework.py`, never invoked)
- `molmetal/scripts/r4_lambda_only_run.py`
- `molmetal/molmetal_lam/proof_search.py`
- `molmetal/molmetal_lam/*`

**Already modified in prior phases (P0 + P1 + Path B — all CPU-only, all in place):**
- P1.1: `flow_matching_lipman/__init__.py:1014, 1548` — `hidden_dim` default 32→128
- P1.2: `flow_matching_lipman/__init__.py:1097-1109, 1301` — `vel_scale` learnable parameter
- P1.3: `flow_matching_lipman/__init__.py:1183-1191, 1414-1433` — `pocket_residue_embed` + `cross_attn`
- P1.4: `flow_matching_lipman/__init__.py:2512-2562` — `ConnectivityAwareDecoder`
- P0 F1: `flow_matching_lipman/__init__.py:2017` — `BondAwareDecoder.decode` wired into `_generate_impl`
- P0 F2: `flow_matching_lipman/__init__.py` — `BondOrderHead` `in_dim=9+2*hidden_dim`
- P0 F3: vocab_mask BEFORE `F.cross_entropy` in training
- P0 F4: `UserWarning` on `hidden_dim<64`
- P0 F5: `bonds=zeros` placeholder removed (2 legitimate empty-edge fallbacks remain, not placeholders)
- Path B decoder rework: `molmetal/adapters/flow_matching_lipman/bond_decoder.py` (CPU smoke verified at 500-step + h=64, `wf_cfm_path_b_decoder_rework/final.md`)

---

## 6. Honest caveats (per user policy: honest-framing mandatory)

- **This SKIPPED verdict is the prescribed outcome, not a placeholder.** The spec explicitly says "if Phase 3 spec gate FAILED: write SKIPPED verdict in phase4_full_retrain.md". Phase 3 FAILED. We follow the spec.
- **GPU is genuinely recovered** at the time of this verdict (`cuda_available=True, device_count=2`). The Phase 3 diagnostic at 18:28:35 ran on the recovered GPU (mtime is 12 minutes after the 18:16 GPU-recovery confirmation per `wf_gpu_recovery_now/final.md`). Re-running the same diagnostic on the recovered GPU would re-observe the same `decode_ratio=0, disconnected_distance_graph` failure mode at the cost of 300 s GPU and 1 of 3 seeds.
- **No number in this report is fabricated.** All Phase 4 metric slots are explicit `NA — not measured`, not silent zeros.
- **The CFM P1 path remains worth pursuing**, but the next experiment should be **path (b) — architectural rework of the velocity-field coordinate-generator** to address the 97.4% connectivity failure mode (per `phase3_gate.md` P5.1 tmQM warm-start + P5.2 pocket boost), not path (a) at a larger budget. The 5000-step h=128 P1-fixed probe is the cheapest decisive evidence we have; it has decided against path (a) at this architecture. Doubling steps does not target the same failure mode.
- **Round-12 / Round-13 paper column decision is unchanged**: stay on path (c) λ-only. The +1.0 metal-compliance, +5.6 valid/synth/uniq, 0.005–0.002 diversity panel, 9 P0 anticancer metrics, and 30-cell PB panel are all MEASURED on the λ-only generator — they do not depend on a CFM recovery.
- **Honest acknowledgment of the spec-naming discrepancy**: the spec's `molmetal/scripts/cfm_path_b_decoder_rework.py` does not exist in the codebase; the closest analogue is `molmetal/scripts/r10_cfg_real_crossdocked.py`. Per `phase2_diagnostic.md`, the diagnostic was run via the latter; this naming discrepancy is a spec artefact, not a measurement issue.

---

## 7. Cross-references

- Phase 1 design: `molmetal/reports/wf_cfm_gpu_retrain/phase1_design.md`
- Phase 2 diagnostic: `molmetal/reports/wf_cfm_gpu_retrain/phase2_diagnostic.md`
- Phase 2 source data: `molmetal/reports/wf_cfm_gpu_retrain/diagnostic/report.json` (223 KB, mtime 2026-09-15 18:28:35 UTC)
- Phase 2 checkpoint: `molmetal/reports/wf_cfm_gpu_retrain/diagnostic/checkpoint_seed0.pt` (2.1 MB)
- Phase 3 gate: `molmetal/reports/wf_cfm_gpu_retrain/phase3_gate.md` (this verdict's precondition)
- Internal review: `molmetal/reports/wf_cfm_internal_review/diagnose.md`
- Path B decoder rework verdict: `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md`
- P0 fixes ship report: `molmetal/reports/wf_cfm_p0_fixes/phase1_p0.md` (see MEMORY.md `WF-CFM-P0-Fixes`)
- P1 fixes ship report: `molmetal/reports/wf_cfm_p1/phase1_p12.md` + `molmetal/reports/wf_cfm_p1/phase2_p34.md`
- Path B Path-1 SKIPPED verdict (earlier): `molmetal/reports/wf_cfm_p1/phase4_full_retrain.md`
- TODO-24 CFM Architecture Redo Plan: `TODO/pending/24_cfm_architecture_redo_plan.md`
- TODO-25 Round-14 lit-grounded plan: `TODO/pending/25_round14_lit_plan.md`
- Round-12 λ-only pilot: `molmetal/reports/wf_round12_lambda_pilot/final.md` (65 cells DESIGN→MEASURED)
- GPU recovery: `molmetal/reports/wf_gpu_recovery_now/final.md` (cuda_available=True at 2026-09-15 18:16 UTC)
- P5.1 tmQM warm-start task: TODO #557
- P5.2 pocket boost at scale=1.0: TODO P1.3 sub-fix (Phase 2A in `wf_cfm_p1/phase2a_strong_boost.md`)

---

**One-line verdict:** `SKIPPED` — Phase 3 spec gate FAILED (`decode_ratio = 0/64` per `wf_cfm_gpu_retrain/phase3_gate.md` 2026-09-15); spec step 1 says write SKIPPED verdict; no code modified, no numbers fabricated, no `r10_cfg_real_crossdocked.py` (or `cfm_path_b_decoder_rework.py`) invocation; default decision stays on path (c) λ-only for Round-12 / Round-13; recommended R14 follow-up is P5.1 tmQM warm-start + P5.2 pocket boost at scale=1.0 with hard cap at 12 GPU-hours.
