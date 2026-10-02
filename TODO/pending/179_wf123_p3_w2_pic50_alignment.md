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

## §5 Deeper audit (after Wf-379-A2 verdict)

**Trigger:** Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL honest-negative close at `TODO/completed/379a2_r20_p5_tmqm_plat.md` §0 surfaced three blockers on the R20-P5 tmQM + PlatinAI 4-cell joint retrain wrapper. Although TODO-179 targets `train_hilbert_astsr_end_to_end.py` (NOT the D-MPNN trainer), the deeper audit confirms the same architectural signal-floor risk applies to L5 R² lift on PlatinAI 4-cell probe.

**Cross-link:** the W₂ alignment lift mechanism in §3.1 is the consumer exercise that depends on an **encoder that lifts the signal floor**. If the encoder can't lift R² from the −0.443 Wf-25 honest-neg → +0.163 Wf-37 P1 bug-fix baseline, then W₂ alignment is a downstream optimization on a stale device. This is the same Wf-127 §Cause 2 (12-d RDKit 2D feature floor) finding restated in Wf-379-A2 terms.

**Audit method:** READ-only inspection of:
- `molmetal/scripts/train_hilbert_astsr_end_to_end.py` (the trainer to modify per §3.2)
- `molmetal/models/chemprop_v2_mt_model_v3.py::ChempropV2MTModelV3` (alternative encoder candidate)
- `molmetal/models/dmpnn.py::DMPNN` (alternative encoder candidate)
- `molmetal/data/rdkit_2d_extended.py` (F5_RDKIT_2D 32-d expansion module already SHIPPED per Wf-252)

**Audit verdict:** the deeper audit at Wf-390-A2 confirms:
- Wf-127 §Cause 2 (R² signal floor on 12-d RDKit 2D) is the structural root cause, NOT W₂ misalignment.
- W₂ alignment is a **necessary but not sufficient** lift mechanism; even perfect W₂ alignment can't lift R² above the encoder's intrinsic representational ceiling.
- The honest projection per §6 below remains 0.40-0.60 R² ceiling WITHOUT F5_RDKIT_2D expansion (32-d) OR alternative encoder bootstrap (chemprop v3 / D-MPNN cross-encoder ensemble).

---

## §6 Three blockers documented (from Wf-379-A2 + Wf-390-A2 audit)

### §6.1 Blocker 1 — Encoder selection / fusion NOT-WRITTEN

The W₂ alignment recipe in §3.1 is correct for **distribution alignment**, but the current Hilbert+ASTSR encoder uses a 12-d RDKit 2D feature stack (per `train_hilbert_astsr_end_to_end.py::load_f5_32d_csv` — note the 32-d loader is already imported but is NOT the default path used by the trainer; default uses 12-d per `_featurize_smiles` at line 149). The W₂ penalty can only align distributions within the encoder's representational capacity — it cannot create signal out of nothing.

**Status:** encoder integration as a **standalone W₂-aware encoder** is NOT-WRITTEN.

### §6.2 Blocker 2 — Architectural signal floor at R²=0.0043

Per `molmetal/reports/wf127_l5_hilbert_diagnostic/` (Wf-127 P2 close-out) — the 12-d RDKit 2D feature stack reaches R² ≤ 0.0043 on the PlatinAI 4-cell probe OOD cohort. This is the **honest signal floor**: a Ridge regression on 12-d RDKit 2D features against a 16k-row PlatinAI 4-cell probe pIC50 distribution reaches essentially zero predictive power.

**Cross-link:** Wf-25 R²=−0.443 (5-fold CV) is the **anti-signal** — worse than predicting mean — because the 12-d features fit the MetalCytoToxDB training cohort but extrapolate inversely on PlatinAI OOD cohort. Wf-37 P1 R²=+0.163 is the bug-fix baseline, NOT an algorithmic lift.

### §6.3 Blocker 3 — Pretrained encoder checkpoint MISSING

Per Wf-379-A2 §1.4 — `molmetal/models/tmqm_pretrained/tmqm_pretrained_encoder.pt` does NOT exist on disk (only `featurise_stats.json` + `pretrain_metrics.json` present). This blocks the alternative encoder bootstrap path that would re-use the tmQM pretraining transcript.

**Note:** tmQM pretraining is **not** required for L5 R² lift on PlatinAI 4-cell — the alternative encoder bootstrap in §7 below uses chemprop v3 + D-MPNN cross-encoder ensemble WITHOUT a pretrained checkpoint.

---

## §7 Alternative encoder bootstrap

**Recommendation:** investigate a chemprop v3 + D-MPNN cross-encoder ensemble as the lift mechanism for the L5 signal floor BEFORE adding W₂ alignment.

### §7.1 Why this works

The 12-d RDKit 2D stack has a structural signal ceiling because it captures **only 2D molecular descriptors** — no 3D geometry, no graph-message-passing, no equivariant features. chemprop v3 (`molmetal/models/chemprop_v2_mt_model_v3.py::ChempropV2MTModelV3` class) and D-MPNN (`molmetal/models/dmpnn.py::DMPNN`) both add graph-message-passing layers that capture atom-bond neighborhood information beyond 2D descriptors.

**Cross-encoder ensemble**: concatenate the chemprop v3 hidden_dim output (default 300-d) + D-MPNN hidden_dim output (default 300-d) + F5_RDKIT_2D 32-d as the input to Hilbert+ASTSR joint regression. This expands the encoder from 12-d → 632-d feature stack, lifting the signal floor.

### §7.2 Why the cross-encoder ensemble is NOT a one-shot**: the existing chemprop_v2_mt_model_v3 trainer is restricted to 5 metals per `train_chemprop_v2_v3.py:190 METALS_5: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")`. To extend to 11 metals (Cr/Mn/Cu/Zn/Ru/Rh/Pd/Ag/Pt/Ir/Au) requires either (a) extending the trainer's `--metals` choices per Wf-379-A2 §1.4, OR (b) bootstrapping a fresh chemprop v3 + D-MPNN ensemble trainer for L5 use case specifically.

### §7.3 Smoke scaffold (CPU-only, USER_OK-gated)

`molmetal/scripts/wf390a2_alt_encoder_audit.py` (~200 LOC, CPU-only) provides:
- `audit_alt_encoders()` — checks for chemprop v3 + D-MPNN encoder availability in `molmetal/models/` (READ-only, no GPU launch)
- `smoke_alt_encoder()` — runs 30-epoch smoke on small cohort with chemprop v3 encoder (CPU-only, requires USER_OK=1 per AGENTS.md §5.1 if `--device cuda` is passed)

Test module: `molmetal/tests/test_wf390a2_alt_encoder.py` (~100 LOC, 2 NEW tests PASS):
- `test_audit_alt_encoders_present` — verifies chemprop v3 + D-MPNN + F5_RDKIT_2D modules are importable
- `test_smoke_alt_encoder_cpu_only` — verifies CPU-mode smoke produces plausible n_features_in >= 300 vs 12-d RDKit 2D baseline

---

## §8 Recommended next step

Two paths forward per Wf-369-A2 verdict + Wf-376-A3 plan:

### §8.1 Path A — 5-metal subset (cheapest)

Re-scope `train_hilbert_astsr_end_to_end.py` to operate on the existing 5-metal subset `{Ru, Ir, Rh, Os, Re}` from `train_dmpnn_multitask_wasserstein.py::FIVE_METALS`. This avoids the trainer extension problem entirely.

**Pros:**
- No trainer extension needed
- Re-uses existing chemprop_v2_mt_model_v3 5-metal head
- Wall: ~5 hr GPU on RTX 5090 per Wf-260 G6 timing

**Cons:**
- Excludes 6 metals (Cr, Mn, Cu, Zn, Pd, Ag, Pt, Au) from L5 R² lift
- PlatinAI 4-cell probe is Pt-only (4078 Pt complexes); L5 R² lift on Pt is the real consumer, NOT multi-metal
- Therefore the 5-metal subset does NOT actually constrain the L5 R² lift — the consumer cohort is Pt-only

**Verdict:** **Path A is the cheapest path for L5 R² lift on PlatinAI 4-cell probe** because the consumer cohort is Pt-only. The 11-metal wrapper ambition is a separate paper §3 R² claim, not a L5 strengthening TODO.

### §8.2 Path B — Alternative encoder bootstrap (more comprehensive)

Build the chemprop v3 + D-MPNN cross-encoder ensemble per §7.1 above. Use F5_RDKIT_2D 32-d (already SHIPPED per Wf-252) + chemprop v3 hidden_dim + D-MPNN hidden_dim as the input to Hilbert+ASTSR joint regression.

**Pros:**
- Lifts the 12-d signal floor structurally
- Re-uses existing chemprop v3 + D-MPNN modules
- Forward-compatible with W₂ alignment (which becomes the downstream consumer exercise)

**Cons:**
- Requires trainer extension to ingest 632-d feature stack
- Wall: ~10 hr GPU (encoder bootstrap + 5-seed × 50-epoch joint retrain)
- Adds ~200 LOC of trainer glue code per Wf-379-A2 honest-negative closure path

**Verdict:** **Path B is the recommended path for paper §3 R² ≥ 0.85 claim** because it lifts the structural signal floor.

### §8.3 Decision matrix

| Axis | Path A (5-metal subset) | Path B (alt encoder bootstrap) |
|---|---|---|
| Wall (GPU) | ~5 hr | ~10 hr |
| LOC | 0 (no extension) | ~200 LOC |
| L5 R² ceiling PROJECTED | +0.40 to +0.60 | +0.60 to +0.85 |
| PlatinAI 4-cell coverage | Pt-only (consumer cohort) | Pt-only + 4 other metals |
| W₂ alignment dependency | downstream | downstream |
| Forward-compatible | NO (paper §3 11-metal claim blocked) | YES |
| USER_OK required | yes (5-seed × 50-epoch on RTX 5090) | yes (encoder bootstrap + 5-seed × 50-epoch on RTX 5090) |

### §8.4 Recommended: Path A first, Path B if Path A ≤ +0.40

Per `party-twenty-spirit` §3.5 honest framing discipline:
1. **Launch Path A first** (cheapest, USER_OK=1 + 5 hr GPU on RTX 5090). If L5 R² ≥ +0.60 ceiling holds, paper §4 can use the 0.163 → +0.60 lift as the honest ceiling.
2. **If Path A R² < +0.40**: launch Path B (alt encoder bootstrap, USER_OK=1 + 10 hr GPU). Document the structural floor lift as the upgrade path.
3. **In either case**: W₂ alignment per §3.1 is the downstream consumer exercise applied on top of the encoder lift.

---

## §9 ETA + dependencies

**ETA**: 1 - 2 weeks (depending on GPU availability + 5-seed × 50-epoch wall-time ~5 hr on RTX 5090)

**Dependencies**:
- Wf-127 P3 joint sweep (L5 R² — Hilbert + ASTSR + W₂) — in-flight, depends on Wf-56 R20-P2 GPU retrain
- `mol-metal-gpu-execution` §6 5-seed × 50-epoch protocol
- Blackwell + RTX 5090 dual-GPU per standing "kill then adapt" directive (need fresh user OK)
- Wf-122 L2 patch (already SHIPPED per TODO-122 close-out)
- `molmetal/reports/wf127_l5_hilbert_diagnostic/` (Wf-127 P2 ship — sweep diagnostic + L2 guard)

---

## §10 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: L5 R² = +0.163 (Wf-37 P1 bug fix, NOT algorithmic lift); Wf-25 honest-neg R² = -0.443
- **PROJECTED**: L5 R² ≥ 0.85 R20-P2 with W₂ alignment — but this is a **strong claim**; honest ceiling per Wf-127 §5 root cause (12-d signal floor) suggests 0.40-0.60 ceiling without feature engineering beyond RDKit 2D
- **The 12-d signal floor caveat** (per Wf-127 §Cause 2): the 12-d RDKit descriptor features have a hard ceiling on what R² can be reached without additional feature engineering (e.g., F5_RDKIT_2D expansion to 32-d or 64-d)
- **No fabrication**: do NOT cite L5 R² ≥ 0.85 in paper §4 LEAD until 5-seed MEASURED — current +0.163 is the honest ship number
- **W₂ may slow early convergence** (per Wf-124 P3 §5): predicted pIC50 forced to track reference before encoder has learned anything; JKO-style CFL gate satisfied but lift may require >10 epochs
- **Per-cell breakdown required** (no aggregation hiding per-cell noise per `party-twenty-spirit` §3.5)

---

## §11 Cross-references

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

## §12 Follow-ups

- Once G3 closed: G1 (TODO-177) consumer metric lifts table row 4 (L5 R²) updates from "NOT measured" → "MEASURED with W₂ alignment"
- If L5 R² < 0.85 after W₂ alignment: feature engineering expansion (F5_RDKIT_2D → 32-d or 64-d) is the next escalation path
- Paper §6 must cite Wf-25 honest-neg R²=-0.443 unconditionally; do not pretend Wf-37 P1 bug fix is an algorithmic lift

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: W₂ alignment is the MECHANISM that could lift L5 R² above the signal-floor. But the 12-d RDKit 2D feature floor (per Wf-127 §Cause 2) suggests 0.40-0.60 ceiling is the honest projection without F5_RDKIT_2D expansion. Do NOT cite L5 R² ≥ 0.85 as MEASURED until 5-seed × 50-epoch completes with W₂ alignment + (optionally) F5 expansion.
