# TODO 179 — Wf-123 P3 Gap: W₂ alignment on PlatinAI 4-cell probe (TODO-122 P1.3 deferred)

**Date**: 2026-09-27 (Wf-123 P3 gap analysis per `party-twenty-spirit` §3.5)
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚧 P1 — IN-FLIGHT (Wf-123 P3 §7 explicitly flags W₂ alignment as the lift mechanism)
**Parent**: `TODO/pending/122_platinai_4cell_r2_strengthen.md` §3 P1.3 + `molmetal/reports/wf123_p3_integration/README.md` §7
**Sibling**: `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md` (G1) + `TODO/pending/178_wf123_p3_metalcytotox_csv_ondisk.md` (G2)

---

## §1 Gap description

**Wf-123 P3 §7 honest framing** explicitly states:

> "The signal-floor R² in the verification report is the **honest measure** of the integration layer — it is ≤ 0 by construction for a constant mean predictor against a different cohort. **The W₂ alignment (TODO-122 P1.3) is what would lift R² above the floor.**"

**Root cause**: The PlatinAI 4-cell probe (4078 Pt complexes × 4 cell lines = ~16k rows) is a **different cohort** from the training cohort (MetalCytoToxDB 5-metal, ~5-8k rows). A constant mean predictor evaluated on this OOD cohort naturally yields R² ≤ 0. The W₂ alignment is the mechanism that aligns the predicted pIC50 distribution to the reference training distribution, which **would** lift R² above the signal-floor.

**Why deferred**:
- TODO-122 P1.3 was deferred to R20-P2 GPU retrain (per `TODO/completed/126_innovation_strengthen_roadmap.md` §3)
- The integration layer SHIP (Wf-123 P3) is the foundation primitive; the W₂ alignment lift is the consumer exercise
- Wf-127 P3 (joint sweep) is the planned spawn but is also in-flight (per `TODO/completed/127_l5_hilbert_diagnostic_closeout.md`)

---

## §2 Current MEASURED state

| Item | Value | Source |
|---|---:|---|
| PlatinAI 4-cell probe R² (signal-floor) | ≤ 0 by construction | `molmetal/reports/wf123_p3_integration/cross_module_verification.json` |
| L5 Hilbert+ASTSR R² (R17 base) | 0.62 | `metrics/by_metric/hilbert_r2.json` |
| L5 Hilbert+ASTSR R² (Wf-25 honest-neg) | -0.443 | `molmetal/reports/wf25_hilbert_l2_astsr/` |
| L5 Hilbert+ASTSR R² (Wf-37 P1 fix) | +0.163 MEASURED (bug fix, NOT algorithmic lift) | `molmetal/reports/wf37_l5_hilbert_astsr/` |
| L5 R² target | ≥ 0.85 R20-P2 / ≥ 0.95 R25 | `LAMBDA_ROADMAP.md` §4.2 L5 |
| PlatinAI 4-cell probe n_rows | 4078 Pt complexes × 4 cell lines = ~16k | Wf-123 P3 §1 |
| Wasserstein penalty decay (Wf-124 P3 Ru seed=0) | 0.8990 → 0.3906 (clear JKO-style convergence) | `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` |

**Gap (MEASURED → TARGET)**:
- L5 R²: +0.163 (Wf-37 P1 fix) → 0.85 → −0.687 gap remaining
- This is the largest single-metric gap in the L1-L12 Lambda trajectory per `LAMBDA_ROADMAP.md` §4.2

---

## §3 What's needed

### §3.1 W₂ alignment recipe

The Wasserstein pIC50 alignment (`wasserstein_pic50_loss` from `molmetal/molmetal_lam/reward/wasserstein_loss.py`) must be wired into the L5 Hilbert+ASTSR trainer with:

1. **Reference distribution**: PlatinAI 4-cell probe (16k rows, real pIC50)
2. **Predicted distribution**: per-batch Hilbert+ASTSR predicted pIC50 (current R²=+0.163 on this OOD cohort)
3. **W₂ penalty**: `sliced_w2(predicted, ref, n_projections=256)` per Wf-123 P1 module
4. **Weight schedule**: `w_w = 0.05` baseline → ramp to `0.20` over 10 epochs (avoid early-epoch divergence per Wf-124 P3 §5 honest caveat)
5. **L2 guard**: grad-norm clip at 5.0 (already in Wf-124 P3 recipe)

### §3.2 Trainer modifications

`molmetal/scripts/train_hilbert_astsr_end_to_end.py` (Wf-37 P1 ship) needs:

- Add `--wass-weight` argparse flag (default 0.05)
- Add `--ref-distribution` argparse flag (default `platinai_4cell`)
- Wire `wasserstein_pic50_loss` into the joint loss: `L = MSE_pIC50 + w_w · W₂(predicted, ref)`
- 5-seed × 50-epoch protocol per `mol-metal-gpu-execution` §6

### §3.3 Per-seed reporting

- Per-seed mean ± std R² (5 seeds)
- Per-seed W₂ decay curve (verify JKO-style monotonic decrease)
- Per-cell breakdown (4 cell lines: A2780, A2780cis, MCF7, etc.)
- MEASURED vs PROJECTED label per `mol-metal-gpu-execution` §7 tag discipline

---

## §4 Acceptance gate

- [G3.1] `wasserstein_pic50_loss` wired into `train_hilbert_astsr_end_to_end.py`
- [G3.2] 5-seed × 50-epoch run on PlatinAI 4-cell probe completed
- [G3.3] Per-seed mean R² ≥ 0.70 (PROJECTED; below gate = §6 limitations)
- [G3.4] W₂ decay curve monotonic (JKO-style)
- [G3.5] Per-cell breakdown reported (no aggregation hiding per-cell noise)
- [G3.6] Engine tag `wasserstein_reg` on every per-sample output (per TODO-126 §7)

---

## §5 ETA + dependencies

**ETA**: 1 - 2 weeks (depending on GPU availability + 5-seed × 50-epoch wall-time ~5 hr on RTX 5090)

**Dependencies**:
- Wf-127 P3 joint sweep (L5 R² — Hilbert + ASTSR + W₂) — in-flight, depends on Wf-56 R20-P2 GPU retrain
- `mol-metal-gpu-execution` §6 5-seed × 50-epoch protocol
- Blackwell + RTX 5090 dual-GPU per standing "kill then adapt" directive (need fresh user OK)
- Wf-122 L2 patch (already SHIPPED per TODO-122 close-out)
- `molmetal/reports/wf127_l5_hilbert_diagnostic/` (Wf-127 P2 ship — sweep diagnostic + L2 guard)

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: L5 R² = +0.163 (Wf-37 P1 bug fix, NOT algorithmic lift); Wf-25 honest-neg R² = -0.443
- **PROJECTED**: L5 R² ≥ 0.85 R20-P2 with W₂ alignment — but this is a **strong claim**; honest ceiling per Wf-127 §5 root cause (12-d signal floor) suggests 0.40-0.60 ceiling without feature engineering beyond RDKit 2D
- **The 12-d signal floor caveat** (per Wf-127 §Cause 2): the 12-d RDKit descriptor features have a hard ceiling on what R² can be reached without additional feature engineering (e.g., F5_RDKIT_2D expansion to 32-d or 64-d)
- **No fabrication**: do NOT cite L5 R² ≥ 0.85 in paper §4 LEAD until 5-seed MEASURED — current +0.163 is the honest ship number
- **W₂ may slow early convergence** (per Wf-124 P3 §5): predicted pIC50 forced to track reference before encoder has learned anything; JKO-style CFL gate satisfied but lift may require >10 epochs
- **Per-cell breakdown required** (no aggregation hiding per-cell noise per `party-twenty-spirit` §3.5)

---

## §7 Cross-references

- `TODO/pending/122_platinai_4cell_r2_strengthen.md` §3 P1.3 (parent strengthen TODO — L5 R² ≥ 0.85)
- `molmetal/reports/wf123_p3_integration/README.md` §7 (W₂ alignment cite)
- `molmetal/reports/wf123_p3_integration/cross_module_verification.json` (signal-floor probe)
- `molmetal/molmetal_lam/reward/wasserstein_loss.py` (~530 LOC, 6 functions — M1 module)
- `molmetal/scripts/train_hilbert_astsr_end_to_end.py` (Wf-37 P1 ship — trainer to modify)
- `molmetal/reports/wf37_l5_hilbert_astsr/` (Wf-37 P1 baseline R²=+0.163)
- `molmetal/reports/wf25_hilbert_l2_astsr/` (Wf-25 honest-neg R²=-0.443)
- `molmetal/reports/wf127_l5_hilbert_diagnostic/` (Wf-127 P2 — sweep diagnostic + L2 guard)
- `TODO/completed/127_l5_hilbert_diagnostic_closeout.md` (Wf-127 close-out — 12-d signal floor root cause)
- `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md` (G1 sibling — L5 R² lift is one of 6 consumer metrics)

---

## §8 Follow-ups

- Once G3 closed: G1 (TODO-177) consumer metric lifts table row 4 (L5 R²) updates from "NOT measured" → "MEASURED with W₂ alignment"
- If L5 R² < 0.85 after W₂ alignment: feature engineering expansion (F5_RDKIT_2D → 32-d or 64-d) is the next escalation path
- Paper §6 must cite Wf-25 honest-neg R²=-0.443 unconditionally; do not pretend Wf-37 P1 bug fix is an algorithmic lift

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: W₂ alignment is the MECHANISM that could lift L5 R² above the signal-floor. But the 12-d RDKit 2D feature floor (per Wf-127 §Cause 2) suggests 0.40-0.60 ceiling is the honest projection without F5_RDKIT_2D expansion. Do NOT cite L5 R² ≥ 0.85 as MEASURED until 5-seed × 50-epoch completes with W₂ alignment + (optionally) F5 expansion.
