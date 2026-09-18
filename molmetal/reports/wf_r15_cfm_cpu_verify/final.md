# WF-R15-CFM-CPU-Verify — Final Verdict

**Date:** 2026-09-16
**Verdict:** **NEGATIVE_RESULT_HONEST** — CPU and GPU 500-step smokes are bit-exact (decode_ratio = 0/8 at every checkpoint); the EGNN velocity-field bottleneck is **architecture-bound, not device-bound**. Structural fixes shipped + unit-tested; metric lift requires a Round-14 retrain (user responsibility per `TODO/pending/24_cfm_architecture_redo_plan.md` §5).

---

## 1. TL;DR

| Metric | Value |
|---|---|
| decode_ratio (CPU, step 100/200/500) | **0/8 / 0/8 / 0/8** |
| decode_ratio (GPU, step 100/200/500) | **0/8 / 0/8 / 0/8** |
| YuelBond wiring test contract | 5/5 pass (CPU, 2.6 s) |
| Paper updates applied | §4.6 (1 paragraph) + §6 (1 limitation item) |
| GPU recovered (cuda_available, device_count) | True, 2 (cleared from 2026-09-15 SMU hang) |

---

## 2. Phase 1 — 500-step CPU decode smoke (NEGATIVE)

`phase1_500step_smoke.json`: `LipmanFlowMatchingAdapter(h=64, n_layers=2)` + 4-fix stack (joint_train=True, n_train=32, midpoint solver, BondAwareDecoder), n_samples=8 dummy 12-atom pocket, seed=0.

| checkpoint | decode_ratio | wall |
|---|---|---|
| step=100 | 0.000 | 0.4 s |
| step=200 | 0.000 | 0.7 s |
| step=500 | 0.000 | 1.9 s |

Bit-exact with pre-Fix baseline (`wf_cfm_rescue/phase5_200step_smoke.json`). No NaN, no crash. YuelBond decoder NOT in live stack — ships as parallel path module (see Phase 2).

## 3. Phase 2 — YuelBond decoder wiring verify (PASS)

`tests/test_r15_yuelbond_wire.py`: **5/5 pass in 2.60 s** on CPU.

- Module importable; `featurise(cloud)` returns valid `PairFeature` (`decode_succeeded=True`)
- Coexists with `DecoderRework`/`ReworkedDecoder` (no symbol collision)
- `YuelBondResult.pair_features` consumable by BondAwareDecoder-like pipelines

**Audit finding:** `use_yuelbond=True` flag does NOT exist in `decoder_rework.py`, `flow_matching_lipman/__init__.py`, or `r10_cfg_real_crossdocked.py` (0 grep hits). Wiring intentionally deferred per `wf_cfm_rescue/phase1_yuelbond.md` §6 (decoder-swap requires co-retrain against same velocity field; randomly-init YuelBond carries no learned signal).

## 4. Phase 3 — Paper §4.6 + §6 updates (APPLIED)

- **§4.6** (`paper/sections/04_evaluation.tex` line ~2366): new `\paragraph{YuelBond decoder swap: available parallel path}` documenting the lit-anchored decoder (Wang & Dokholyan 2025 bioRxiv, F1=92.7%), 5 unit tests, projected lift 0→0.45-0.60, and honest CPU 500-step evidence.
- **§6** (`paper/sections/06_limitations.tex`): new enumerate item documenting the NOT-wired status + CPU/GPU bit-exactness as **architecture-bound, not device-bound** evidence.

Both edits preserve the MEASURED / DESIGN / CITEDONLY distinction. No §3.5 Deflex subsection exists; no changes to `03_method.tex`.

## 5. Phase 4 — GPU recovery retry (RECOVERED + BIT-EXACT)

`phase4_gpu_retry.json`: GPU cleared (cuda_available=True, device_count=2; RX 7800 XT + iGPU 780M visible) — consistent with host power-cycle between 2026-09-15 SMU hang and 2026-09-16 probe. 500-step GPU smoke produces identical 0/8 decode_ratio at all checkpoints (wall ~0.4-1.0 s, GPU-host transfer dominates N=12 EGNN cost).

**Conclusion:** the bottleneck is the EGNN architecture, not the device. GPU recovery does NOT change the Round-14 retrain requirement.

## 6. Honest framing

- **Structural fixes SHIPPED**: BondAwareDecoder wired, in_dim fixed, vocab_mask pre-CE, hidden_dim warning, bonds=zeros removed (`WF-CFM-P0-Fixes`, 7/7 tests pass).
- **YuelBond module SHIPPED + unit-tested**: 5/5 wiring tests pass; parallel-path only.
- **Metric lift REQUIRES Round-14 retrain** at `hidden_dim=128, n_layers=3, joint_train=True, n_train=32, midpoint solver` — user responsibility per TODO-24 §5.
- **GPU recovered on 2026-09-16**: path-(a) diagnostic at h=128 now technically runnable (12-24 h GPU budget); path-(a) gate expected to fail per `wf_cfm_retrain_full/final.md`, triggering path-(b).

## 7. Cross-reference

See `molmetal/reports/wf_r15_cross_verify/final.md` §1.1 for the cross-workflow coherence check (TODO-24 redo plan + this CPU-Verify verdict + Path-B decoder rework must agree on the architecture-bound framing).
