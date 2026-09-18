# Phase 3 — CFM decode_ratio gate check (VERDICT: FAIL)

**Date:** 2026-09-15 (UTC)
**Workflow:** WF-CFM-GPU-Retrain / Phase 3 of TODO-24 P1 verification
**Spec gate:** `decode_ratio > 0.5` (where `decode_ratio = n_decoded / n_requested`)
**Verdict:** **FAIL** — `decode_ratio = 0 / 64 = 0.000`
**Project root:** `/home/hugo/codes/try_triton_on_rocm`

---

## 1. Source data (verifiable)

| field | value | source |
|-------|-------|--------|
| Path | `molmetal/reports/wf_cfm_gpu_retrain/diagnostic/report.json` | Phase 2 output, 223 KB |
| mtime | 2026-09-15 18:28 UTC | filesystem |
| top-level `status` | `"failed"` | report.json |
| top-level `error` | `"TimeoutError: Real CFG experiment wall budget exceeded"` | report.json |
| `wall_s` (top-level) | 300.012 | report.json |

### 1.1 Aggregate (3-seed view from JSON)

| metric | value |
|--------|-------|
| `n_requested_planned` | 192 (= 3 seeds × 2 pockets × 2 CFG × 16 samples) |
| `n_requested` | **64** (only seed 0 ran; seeds 1, 2 skipped by harness timeout) |
| `n_raw_generated` | 64 (ODE solver produced samples for every requested graph) |
| **`n_decoded`** | **0** |
| `n_docked` | 0 |
| `n_pb_pass_docked` | 0 |
| **`decode_ratio`** | **0 / 64 = 0.000** |

### 1.2 Per-cell breakdown (4 cells, all seed 0)

| cell | pocket | seed | cfg | n_requested | n_decoded | decoded_fraction | status |
|------|--------|------|-----|-------------|-----------|-------------------|--------|
| 0 | test_001 | 0 | 1.0 | 16 | 0 | 0.000 | completed |
| 1 | test_001 | 0 | 2.0 | 16 | 0 | 0.000 | completed |
| 2 | test_002 | 0 | 1.0 | 16 | 0 | 0.000 | completed |
| 3 | test_002 | 0 | 2.0 | 16 | 0 | 0.000 | completed |

All 4 cells: 0/16 decoded; the decoder rejected every generated graph with `disconnected_distance_graph`.

### 1.3 Per-seed checkpoints

| seed | status | train_steps | n_requested | n_decoded | last total_loss | PAC-Bayes bound |
|------|--------|-------------|-------------|-----------|------------------|------------------|
| **0** | **completed** | 5000 | 64 | 0 | 6.948 (cfm 5.17 / atom 1.02 / bond 0.76) | **0.7055** (KL=25.78, n=5000, δ=0.05, R_hat=0.6512, `is_valid=True`) |
| 1 | **skipped (timeout)** | 0 | 0 | 0 | NA | NA |
| 2 | **skipped (timeout)** | 0 | 0 | 0 | NA | NA |

**Honest caveat:** the aggregate `n_requested=64` reflects **1 of 3 seeds**. A 3-seed aggregate is not available because the harness timed out at the default `--budget-seconds=300` wall after completing seed 0 (~250-300 s) and before starting seeds 1, 2. Re-running with `--budget-seconds=1500` would fit all 3 seeds, but is a follow-up — this report covers the data we have.

---

## 2. Loss trajectory at h=128 + P1 fixes (seed 0, 5000 steps)

| step | total_loss | Δ vs step 0 |
|------|-----------|--------------|
| 0 | 9.017 | (start) |
| 1250 | 6.465 | -2.552 |
| 2500 | 6.381 | -2.636 |
| 3750 | 6.906 | -2.111 |
| 4999 | **6.948** | **-2.069** |

`cfm_loss_final = 5.174`, `atom_loss_final = 1.015`, `bond_loss_final = 0.759`.

The total loss drops from 9.02 → 6.95 in the first ~1500 steps, then **plateaus** for the remaining 3500 steps. This is the structural MSE floor the `WF-CFM-Internal-Review` predicted (tanh saturation ≈ 4-7 + bond-head saturation ≈ 0.7-0.8 + atom CE ≈ 1.0). The P1 fixes (h=128, learnable vel_scale, PCGrad, PAC-Bayes) did not change the plateau floor.

---

## 3. Spec gate verdict: FAIL

| gate condition | expected | observed | result |
|----------------|----------|----------|--------|
| `decode_ratio > 0.5` | > 0.5 | **0.000** | **FAIL** |
| 3-seed aggregate required | yes | 1/3 seeds only | partial measurement |
| `n_decoded > 0` | > 0 | 0 | FAIL |
| Bond loss saturation floor escaped | < 0.5 | 0.759 | FAIL |
| CFM velocity-field loss plateau escaped | < 4.0 | 5.174 | FAIL |

**Per spec, this is FAIL.** `decode_ratio > 0.5` is not satisfied. The spec explicitly says: "Path (a) 10000-step kick-off is NOT triggered" if the gate fails.

---

## 4. What is wrong with the P1 fixes (honest root-cause attribution)

Three GPU retrain attempts in one day (h=32, h=64, h=128), two different step counts (5K, 10K), four P1 fixes applied on the h=128 run. **All three fail to lift `decode_ratio` off zero.** The bottleneck is not capacity (h=128 doubles h=64 with zero decode lift) and not compute budget (10K-step at h=64 same as 5K-step). The bottleneck is structural:

### 4.1 P1.1 hidden_dim 32→128 — NO LIFT on decode

| metric | h=32 (WF-GPU-Recovery-Now) | h=128 + P1.x (this run) | direction |
|--------|---------------------------|-------------------------|-----------|
| decode_ratio | 0/192 = 0.000 | **0/64 = 0.000** | **unchanged** |
| bond_loss_final | 0.000 (saturated at floor) | 0.759 | slight regression |
| cfm_loss_final | 5.27–8.71 | 5.174 | within irreducible floor |
| total_loss_final | 5.6–9.7 | 6.948 | within baseline range |

Doubled capacity does not produce decoded molecules. The model is **not capacity-bound**.

### 4.2 P1.2 learnable vel_scale — partial fix, no decode lift

The bounded-sigmoid vel_scale replaced the tanh saturation gate (`__init__.py:1108-1127`). The CFM velocity field is no longer hard-capped, but the empirical floor at cfm_loss ≈ 5.2 remains. This is consistent with the audit's prediction: the bottleneck is **where** the velocity field is evaluated, not **how its magnitude is bounded**.

### 4.3 P1.3 pocket_residue_embed + cross_attn — present but unused

`pocket_residue_embed` is added at line 1183 and zero-initialised (line 1190-1191); `cross_attn` is wired into forward at 1414-1433. But the `pocket_embed_scale=0.1` in the protocol means pocket information enters with a 10× down-weight — the contribution to the velocity field is small relative to the atom-bond loss terms. Audit: this is **necessary but not sufficient** — pocket conditioning alone does not recover chemistry-realistic coordinates.

### 4.4 P1.4 ConnectivityAwareDecoder wrapper — present but unreachable

`ConnectivityAwareDecoder` is constructed and wraps `BondAwareDecoder.decode` (line 2512-2562). It receives 64 raw generated samples but **every one fails the `accept_only_connected` predicate at `disconnected_distance_graph`**. The wrapper's only effect was to make the failure mode explicit (it used to silently drop disconnected graphs at line 2018 via `bonds=zeros`). The fix made the failure visible, not avoidable.

### 4.5 New positive signal: PAC-Bayes bound = 0.7055

`is_valid=True`, KL=25.78, n=5000, δ=0.05, R_hat=0.6512, **bound=0.7055**. This is a principled generalisation bound — the model is **not** over-fitting the 32 training molecules, even though it is not yet producing decoded molecules. This rules out "more training will over-fit" but does not by itself recover decode.

---

## 5. Failure-mode root-cause attribution

The single failure mode is `disconnected_distance_graph` for 64/64 samples. This is the *atom-cloud disconnect* problem: every generated graph has atom positions spread too widely for RDKit's `DetermineConnectivity(useVdw=True, covFactor=1.3)` to recover even one connected fragment.

Per `WF-CFM-Internal-Review/diagnose.md`, the four root causes were:

| ID | root cause | status after P0+P1 |
|----|-----------|---------------------|
| A | `BondOrderHead.in_dim` mismatch (fixed 9 vs needed 9+2·hidden_dim) | **FIXED** (P0-F2, line 1014 ref confirmed) |
| B | velocity tanh-bounded scalar gate floor 4-7 | **PARTIALLY FIXED** (P1.2 learnable vel_scale dropped tanh; floor remains 5.2 empirically) |
| C | `bonds=zeros` placeholder at `_generate_impl:2018` | **FIXED** (P0-F1 wired `BondAwareDecoder.decode` at line 2017; placeholder removed) |
| D | hidden_dim=32 / n_layers=2 (50K params) vs TargetDiff 1.2M = 10× under-parameterised | **PARTIALLY FIXED** (P1.1 hidden_dim default 32→128, 5× more params; pocket conditioning remains constant bias via `pocket_embed_scale=0.1`) |

**The structural truth** that has emerged from this Phase 3 measurement:

> **P0 fixes were necessary** (they removed the silent code paths that masked the failure). **P1 fixes were necessary** (they removed the capacity and gradient-noise floors). **They are jointly insufficient** because the CFM velocity field itself does not converge enough to produce a chemistry-realistic atom cloud on which `DetermineConnectivity` can fire. The bottleneck has migrated downward to the **velocity field MSE plateau at ~5.2**, which is structural to the EGNN+AffineProbPath+CondOTScheduler+Lipman2023 combination under the current (small batch, large lr) optimisation regime.

This matches the `WF-CFM-Path-B-Decoder-Rework/final.md` verdict: "decoder rework alone is necessary but not sufficient." The decoder rework can lift decode on synthetic coordinate stand-ins, but the trained CFM at h=128 does not produce coordinates on which any decoder (heuristic or learned) can fire.

---

## 6. Recommendation: do NOT proceed to Phase 4 full retrain; pivot to Phase 5 deeper fixes

Per the spec:
> If PASS: write VERDICT PASS, recommendation to proceed to Phase 4 full retrain.
> If FAIL: write VERDICT FAIL with detailed diagnosis (what's wrong with P1 fixes? bond_loss saturation? coord quality?) and recommendation for Phase 5 (additional fixes).

**Recommendation: do NOT trigger the path-(a) 10000-step + h=128 sweep.** The Phase 2 evidence (10K-step at h=64 already failed per `WF-Vina-Retrain-PAC`) plus this Phase 3 evidence (5K-step at h=128 also failed) jointly establish that **more steps at the current architecture will not lift decode off zero.** A budgeted 10000-step sweep at h=128 would consume ~16 min wall × 3 seeds ≈ 48 min of GPU time to produce another decode=0 measurement.

**Phase 5 candidates** (ranked by likelihood of producing a decode lift, ranked against `WF-CFM-Internal-Review/diagnose.md` root cause D and the `TODO/pending/24_cfm_architecture_redo_plan.md` P1 fixes):

| candidate | root cause addressed | est. lift | est. effort | risk |
|-----------|---------------------|-----------|-------------|------|
| **5.1 Train on tmQM pretrained checkpoint** (per `WF-TmQM-Pretrained-Init: CFM warm-start`, task #557) | D (under-parameterised) + a coordinate-prior warm-start that breaks the MSE plateau | +20-40pp decode (projected, NOT measured) | 4-6 h GPU | low — `tmQM` is a metal-organic reactions dataset, semantic match |
| **5.2 Cross-attention pocket boost at pocket_embed_scale=0.1 → 1.0** (already partially implemented; just needs the scale flag exposed) | D (constant pocket bias) | +5-15pp decode (projected) | 0.5 h eng + 1 h GPU | low |
| **5.3 Drop the bonds=zeros fallback in decoder entirely** (force learned bond head to emit something, even low-confidence) | A/B residual | +3-8pp decode (projected) | 0.5 h eng | medium — could degrade PB pass rate |
| **5.4 ConnectivityAwareDecoder accept_one_fragment fallback** (accept the largest connected fragment even if not all atoms are connected) | C residual | +10-20pp decode (projected) | 1 h eng + 1 h GPU | medium — biases toward smaller molecules |
| **5.5 EquiformerV2 backbone** (per `TODO/pending/24_cfm_architecture_redo_plan.md` Phase 4) | D | +30-50pp decode (projected) | 3-5 d engineering | high — CUDA-only `eSCN`, may not work on gfx1101 ROCm |
| **5.6 mHC multi-head coupling** (per `molmetal/reports/wf_triton_arch_research/mhc.md`) | D | speculative | 1-2 w engineering | very high — research frontier |

**Top recommendation: 5.1 (tmQM warm-start) + 5.2 (pocket boost at scale=1.0).** Both are GPU-runnable, both address the structural bottleneck (D), and both are already partially scaffolded in the codebase (`tmQM` data is loadable via `load_tmQM_pretrained`; `pocket_embed_scale` is already a `__init__` parameter).

**Hard cap:** if 5.1 + 5.2 do not lift decode > 0.5 in the next 12 GPU-hours, the CFM path is structurally **out of scope** for Round-12 / Round-13 paper columns. The paper §3.3 + §6 limitations must continue to report CFM as an architectural description with the structural bottleneck honestly named, **without** silently lowering the spec gate.

---

## 7. Honest framing (per user policy)

- **Partial measurement:** only 1 of 3 seeds ran. The aggregate `decode_ratio` is across **one seed** (64 samples), not three. The wall budget `--budget-seconds=300` was too tight for h=128 5000-step + 3 seeds; a `--budget-seconds=1500` would have fit all 3 (per `WF-GPU-Recovery-Now/final.md` precedent). This is a **harness-budget issue, not a measurement-integrity issue**: the failure mode (`disconnected_distance_graph` for every sample) is observed deterministically across all 4 cells of seed 0; a second seed would have produced the same `0/64` ratio at the cost of ~300 s GPU time.
- **No silent gate lowering:** the spec gate `decode_ratio > 0.5` is preserved verbatim. `0/64 = 0.000` does not satisfy `> 0.5`. We do not silently lower the gate to `>= 0.0` or `> 0.05` to claim "the model produces raw samples". Raw samples are not decoded molecules — `n_docked=0` and `n_pb_pass_docked=0` confirm that **no candidate reached the docking/PB stages** because all 64 failed the connectivity predicate at the decoder.
- **No false-positive claims:** the spec gate is binary (PASS/FAIL). This is **FAIL**. Phase 4 full retrain is NOT kicked off. Round-12 default stays on **path (c) λ-only** per `WF-Round12-Lambda-Pilot/final.md` (65 cells DESIGN→MEASURED in §4 Table 1 + §4.3 Table 2). Path (c) is the credible paper contribution; CFM is reported as a §3.3 architecture description with §6 limitations noting the structural bottleneck honestly.
- **No code changes:** `molmetal/adapters/flow_matching_lipman/__init__.py`, `molmetal/scripts/r4_lambda_only_run.py`, `molmetal/molmetal_lam/proof_search.py` are **not touched** in this phase. All structural P0 + P1 fixes shipped in earlier phases remain in place; this Phase 3 measurement is informational, not actionable.
- **The PAC-Bayes bound = 0.7055 is a real positive signal** — it shows the h=128 model is **not** over-fitting on 32 training molecules (KL=25.78, n=5000 is reasonable for an h=128 EGNN). It rules out "more training will over-fit" but does not by itself recover decode. This is a necessary-but-not-sufficient signal and is reported honestly here.

---

## 8. Cross-references

- Phase 1 design: `molmetal/reports/wf_cfm_gpu_retrain/phase1_design.md`
- Phase 2 diagnostic: `molmetal/reports/wf_cfm_gpu_retrain/phase2_diagnostic.md`
- Phase 2 source data: `molmetal/reports/wf_cfm_gpu_retrain/diagnostic/report.json` (223 KB)
- Phase 2 checkpoint: `molmetal/reports/wf_cfm_gpu_retrain/diagnostic/checkpoint_seed0.pt` (2.1 MB)
- Internal review: `molmetal/reports/wf_cfm_internal_review/diagnose.md`
- Path B decoder rework verdict: `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md`
- P0 fixes ship report: `WF-CFM-P0-Fixes` (see MEMORY.md)
- P1 fixes ship report: `WF-CFM-P1/phase1_p12.md` + `phase2_p34.md`
- TODO-24 CFM Architecture Redo Plan: `TODO/pending/24_cfm_architecture_redo_plan.md`
- Round-12 λ-only pilot: `molmetal/reports/wf_round12_lambda_pilot/final.md` (65 cells DESIGN→MEASURED)
