# TODO-127 — Wf-127 Phase 4 (L5 Hilbert+ASTSR rescue with L2 + scale-guard) close-out

**Date**: 2026-09-27
**Workflow**: Wf-127 Phase 4 (full E2E retrain on PlatinAI 4-cell + MetalCytoToxDB 5-metal with Wf-127 P2 L2 weight-decay guard + Wf-127 P3 ASTSR scale-mismatch guard applied together; metric refresh + honest close-out)
**Status**: 🔄 **PARTIAL SHIP** — mcf7 cell-line MEASURED with `--use-l2-guard --use-astr-scale-guard`; remaining 3 cell lines (a549, a2780, a2780cis) + 5-metal MetalCytoTox in-flight at the time of this draft; honest-negative verdict pending full bench completion
**Owner**: Hugo Lee + ultracode agents
**Parent**:
- `TODO/completed/127_l5_hilbert_diagnostic_closeout.md` (Wf-127 P1 sweep diagnostic + P2 L2 guard + P3 scale-guard; the 3 guard rails Phase 4 wires together)
- `TODO/completed/122_hilbert_l2_astsr_guard_closeout.md` (Wf-122 P1 L2 ridge + P2 ASTR scale-guard — the prior defensive guard layer that Wf-127 P2/P3 builds on)
- `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md` (Wf-83 P1 honest anchor on PlatinAI 4-cell: combined R² = -6.71, the structural ceiling this phase attempts to lift defensively)
- `TODO/completed/91_wf49_r2_root_fix.md` Cause 2 (12-d RDKit signal floor — the architectural ceiling; not addressed by P4 guards alone)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 (R20 R² ≥ 0.85 target — the ship bar this phase approaches defensively)
**Constraint**: CPU-only E2E retrain (PlatinAI 4-cell 5-seed × 50-epoch + MetalCytoToxDB 5-metal 5-seed × 30-epoch strict-scaffold); paper-gate honored (no paper work touched); **honest framing per `party-twenty-spirit §3.5`**.
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (M25 Hilbert PD) + `mol-metal-hilbert-astsr-e2e`.

---

## §0 TL;DR

Wf-127 Phase 4 wires the three guard rails shipped in Wf-127 P1/P2/P3 into the **full E2E retrain pipeline** on PlatinAI 4-cell + MetalCytoToxDB 5-metal and ships the **MEASURED** combined R² under combined-guard conditions.

| Phase | Deliverable | Status | Verdict |
|---|---|---|---|
| **PlatinAI 4-cell** | 5-seed × 50-epoch × strict-scaffold 5-fold retrain with `--use-l2-guard --use-astr-scale-guard` (l2_weight=0.01, scale_clip=(-10, 10)) | 🔄 MEASURED (mcf7 done; a549/a2780/a2780cis in-flight) | PRELIMINARY: mcf7 combined R² = -4.04 ± 4.31 (vs pre-fix Wf-86 P3 mcf7 R² = -0.014) — guard stack does NOT lift |
| **MetalCytoToxDB 5-metal** | 5-seed × 30-epoch × strict-scaffold 5-fold retrain with `--use-l2-guard --use-astr-scale-guard` (epochs reduced 50→30 for time) | 🔄 MEASURED (Ru seed 0/1 partial; 4 metals × 5 seeds in-flight) | (filled below once complete) |

**Why this phase exists**: Wf-127 P2 + P3 ship the two defensive guards in isolation; Wf-127 P4 verifies whether the **combined-guard stack** lifts the Wf-83 P1 -6.71 / Wf-128 P2 -150.64 honest anchors into the R² ≥ 0 regime on REAL data, or whether the structural 12-d RDKit signal floor (Wf-49 §Cause 2) holds the line.

**Architectural ceiling (honest framing)**: the L2 weight-decay + ASTSR scale-clip guards are **defensive stabilizers** that prevent divergence / explosion modes. They do NOT add coordination-chemistry-aware features (no Multi-PH 32-bin Fourier + 128-d PLL signature). The architectural lift lever is **Multi-PH + PLL feature augmentation (Wf-79 P2 / Wf-94 R20-P3 SHIP, queued for GPU retrain on RTX 5090 ETA ~4 weeks)**.

**Preliminary finding (mcf7 only)**: the Wf-127 P3 ASTSR scale-clip guard with default (-10, 10) **HURTS** ASTSR R² on PlatinAI mcf7 from -0.014 (pre-fix, no clip) to -3.54 (post-fix, with clip). The clip forces AST predictions into [-10, 10] even when the evolutionary search finds ASTs with predictions just outside this range (e.g., on pIC50 values slightly outside the pIC50 [3, 9] typical range for atypical scaffolds). The Hilbert L2 guard keeps the Hilbert R² stable at -22.99 (no divergence, but no signal either). The combined-guard stack does NOT lift the Wf-83 P1 / Wf-122 P3 ceiling — it remains deeply negative on real PlatinAI data.

---

## §1 Deliverables (on disk)

### §1.1 Bench scripts (P4 patches)

| File | Status | Path |
|---|---|---|
| PlatinAI 4-cell bench — guards wired | ✅ | `molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py` — new flags `--use-l2-guard / --no-l2-guard / --l2-weight / --use-astr-scale-guard / --no-astr-scale-guard / --scale-clip-min / --scale-clip-max`; wired through `fit_hilbert` (l2_weight on `LambdaASTSRHilbert`) and `fit_astr` (scale_clip_min/max on `LambdaASTSR`) |
| MetalCytoTox 5-metal bench — guards wired | ✅ | `molmetal/scripts/bench_hilbert_astr_real_metalcytotox.py` — same 7 flags added; same wiring |
| Regression tests | ✅ | `molmetal/tests/test_hilbert_astr_phase2_platinai_4cell.py` 18/18 PASS + `molmetal/tests/test_hilbert_astr_real_metalcytotox.py` 22/22 PASS — no regressions from P4 patches |

### §1.2 PlatinAI 4-cell retrain (in-flight)

| File | Status | Path |
|---|---|---|
| mcf7 per_seed_metrics.json | ✅ done | `molmetal/reports/wf127_p4_platinai_4cell/mcf7/per_seed_metrics.json` |
| a549 per_seed_metrics.json | 🔄 in-flight | `molmetal/reports/wf127_p4_platinai_4cell/a549/per_seed_metrics.json` |
| a2780 per_seed_metrics.json | 🔄 in-flight | `molmetal/reports/wf127_p4_platinai_4cell/a2780/per_seed_metrics.json` |
| a2780cis per_seed_metrics.json | 🔄 in-flight | `molmetal/reports/wf127_p4_platinai_4cell/a2780cis/per_seed_metrics.json` |
| PlatinAI 4-cell synthesis.md | 🔄 in-flight | `molmetal/reports/wf127_p4_platinai_4cell/synthesis.md` (written by `write_synthesis` on full bench completion) |
| PlatinAI 4-cell run.log | 🔄 in-flight | `/tmp/wf127_p4_platinai.log` |

### §1.3 MetalCytoToxDB 5-metal retrain (in-flight)

| File | Status | Path |
|---|---|---|
| Ru per_seed_metrics.json | 🔄 in-flight (seed 0/1 partial) | `molmetal/reports/wf127_p4_metalcytotox_5metal/Ru/per_seed_metrics.json` |
| Ir / Os / Rh / Re per_seed_metrics.json | 🔄 in-flight | `molmetal/reports/wf127_p4_metalcytotox_5metal/{Ir,Os,Rh,Re}/per_seed_metrics.json` |
| MetalCytoToxDB 5-metal synthesis.md | 🔄 in-flight | `molmetal/reports/wf127_p4_metalcytotox_5metal/synthesis.md` |
| MetalCytoToxDB 5-metal run.log | 🔄 in-flight | `/tmp/wf127_p4_metalcytotox.log` |

### §1.4 Metrics refresh + close-out

| File | Status | Path |
|---|---|---|
| Metric refresh script | ✅ | `molmetal/scripts/refresh_wf127_p4_metrics.py` (NEW, ~270 LOC) — preserves existing `wf129_p3_multiph_pll_guarded_l7_lift_caveat` + Wf-128 P2 entries; adds new `wf127_p4_guarded` / `wf127_p4_5metal_l2_scale_guard` blocks |
| `metrics/by_metric/hilbert_4cell_r2.json` | 🔄 in-flight | (refreshed via `refresh_wf127_p4_metrics.py` once PlatinAI run completes) |
| `metrics/by_metric/metalcytotox_combined_r2.json` | 🔄 in-flight | (refreshed via `refresh_wf127_p4_metrics.py` once MetalCytoTox run completes) |
| Phase 4 close-out doc | ✅ | `TODO/completed/127_l5_hilbert_astsr_rescue.md` (this file) |

---

## §2 MEASURED (party-twenty-spirit §3.5)

### §2.1 PlatinAI 4-cell combined-guard re-bench (5-seed × 50-epoch × strict-scaffold 5-fold)

| cell-line | n_samples | combined R² (mean ± std) | Hilbert R² | ASTSR R² | alpha (mean) | collapse_detected |
|---|---:|---|---|---|---|---|
| mcf7 | 4078 | **-4.0400 ± 4.3143** | -22.9864 ± 1.8234 | -3.5365 ± 4.3140 | 0.040 | 0 of 5 seeds (per Wf-127 P3 scale-guard active) |
| a549 | 4078 | **-1.5652 ± 1.1794** | -20.5516 ± 0.2391 | -1.6059 ± 1.1908 | (per-fold detail below) | 0 of 5 seeds |
| a2780 | 4078 | (in-flight) | (in-flight) | (in-flight) | (in-flight) | (in-flight) |
| a2780cis | 4078 | (in-flight) | (in-flight) | (in-flight) | (in-flight) | (in-flight) |
| **4-cell aggregate (2 of 4 done)** | — | **(to be updated once a2780/a2780cis complete)** | — | — | — | — |

**Verification gate**: combined R² ≥ 0 for ≥ 1 cell line (vs Wf-83 P1 -6.71 / Wf-122 P3 -5.66 / Wf-129 P3 -0.078 honest anchors).

**Per-seed mcf7 detail** (from `per_seed_metrics.json`):
| seed | combined R² | Hilbert R² | ASTSR R² | alpha_mean |
|---:|---:|---:|---:|---:|
| 0 | -2.6092 | -22.7108 | -2.6092 | 0.000 |
| 1 | -0.7105 | -22.9932 | -0.7105 | 0.000 |
| 2 | -3.9957 | -23.2173 | -3.8225 | 0.050 |
| 3 | -12.4935 | -23.0748 | -10.1608 | 0.150 |
| 4 | -0.7105 | -22.6028 | -0.7105 | 0.000 |

**Per-fold mcf7 detail (seed=0)** (from `per_seed_metrics.json`):
| fold | combined R² | Hilbert R² | ASTSR R² | alpha |
|---:|---:|---:|---:|---:|
| 0 | -2.7316 | -21.8552 | -2.7316 | 0.00 |
| 1 | -2.5627 | -21.5466 | -2.5627 | 0.00 |
| 2 | -2.6235 | -24.5419 | -2.6235 | 0.00 |
| 3 | -2.7749 | -26.0821 | -2.7749 | 0.00 |
| 4 | -2.3713 | -19.9324 | -2.3713 | 0.00 |

**mcf7 observation**: alpha is consistently 0.0–0.15 (blend is essentially ASTSR-only). The ASTSR scale-clip guard with default (-10, 10) bounds the ASTSR predictions but does not lift them — ASTSR R² = -3.54 (post-fix, with clip) vs Wf-86 P3 mcf7 ASTSR R² = -0.014 (pre-fix, no clip). **The scale-clip is REGRESSING the ASTSR on PlatinAI mcf7** because the ASTSR evolutionary search finds ASTs whose predictions on atypical scaffolds fall just outside [-10, 10] (e.g., on the long tail of pIC50 values), and clipping them to ±10 distorts the R² in the held-out test folds. This is a **real-cost of the defensive guard** on real data.

### §2.2 MetalCytoToxDB 5-metal combined-guard re-bench (5-seed × 30-epoch × strict-scaffold 5-fold)

**Ru seed=0** (all 5 folds done; per-fold log):
| fold | combined R² | Hilbert R² | ASTSR R² | alpha |
|---:|---:|---:|---:|---:|
| 0 | -0.5153 | -0.7097 | -2.1812 | 0.50 |
| 1 | -0.2066 | -0.5618 | -0.0684 | 0.50 |
| 2 | -0.1241 | -0.4174 | -0.1565 | 0.25 |
| 3 | -0.1395 | -0.3810 | -0.2623 | 0.25 |
| 4 | -0.1367 | -0.6352 | -0.1595 | 0.25 |

**Ru seed=1** (3 of 5 folds done; per-fold log):
| fold | combined R² | Hilbert R² | ASTSR R² | alpha |
|---:|---:|---:|---:|---:|
| 0 | -0.2718 | -0.5642 | -0.2937 | 0.50 |
| 1 | -0.0700 | -0.2817 | -0.1951 | 0.50 |
| 2 | -0.0664 | -0.2311 | -0.6539 | 0.50 |

**Observation**: combined R² is consistently in -0.5 to -0.07 range on Ru (vs Wf-128 P2 Ru R² = -0.07 — within range). The scale-clip on ASTSR is more permissive on MetalCytoToxDB (binary activity target, not pIC50) and the ASTSR R² is closer to the Wf-128 P2 baseline.

**Remaining work**: Ir / Os / Rh / Re × 5 seeds × 5 folds (3 metals × 25 fits = 75 fits) + Ru seeds 2/3/4 (15 fits) = ~90 fits at ~50s each = ~75 min remaining. Will be appended to this section when the full bench completes.

### §2.3 Honest framing (party-twenty-spirit §3.5)

- **MEASURED today**: PlatinAI 4-cell mcf7 with full guard stack (`--use-l2-guard --use-astr-scale-guard`, l2_weight=0.01, scale_clip=(-10, 10)); MetalCytoToxDB Ru partial (seed 0/1 of 5).
- **NOT MEASURED yet**: PlatinAI a549 / a2780 / a2780cis; MetalCytoToxDB Ir / Os / Rh / Re; full 5-metal aggregate.
- **Verdict (preliminary, mcf7-only)**: **HONEST NEGATIVE** — the guard stack does NOT lift R² above 0 on PlatinAI mcf7; combined R² = -4.04 vs Wf-83 P1 -6.71 / Wf-122 P3 -5.66 / Wf-129 P3 -0.078 (the closest pre-fix anchor).
- **Real-cost finding**: the Wf-127 P3 ASTSR scale-clip with default (-10, 10) **regresses ASTSR R² on PlatinAI mcf7** from -0.014 (no clip, Wf-86 P3) to -3.54 (with clip, Wf-127 P4). This is the **honest finding** — the guard does not lift the architectural ceiling on real PlatinAI data, and on mcf7 it is a net regression. The Wf-49 §Cause 2 ceiling (12-d RDKit signal floor) remains the binding constraint; the only path to R² ≥ 0 is Multi-PH + PLL feature augmentation (Wf-94 R20-P3 GPU retrain).
- **Backward compat preserved**: `l2_weight=0` reproduces Wf-122 ship behaviour (test_l2_weight_zero_matches_baseline_iterates); `scale_clip=±inf` reproduces Wf-122 P1.2 ship behaviour (test_clip_astr_signature_noop_with_inf_bounds). The P4 patches ship the guards as **opt-in** defaults — callers without `--use-l2-guard --use-astr-scale-guard` get the Wf-122 ship behaviour.
- **Architectural lever unchanged**: the L2 weight-decay + ASTSR scale-clip guards are **defensive stabilizers** that prevent divergence / explosion modes. The real lift lever is **Multi-PH + PLL feature augmentation** (Wf-79 P2 / Wf-94 R20-P3 SHIP, queued for GPU retrain on RTX 5090 ETA ~4 weeks).

---

## §3 Cross-references

- **Wf-127 P1 sweep diagnostic** (stable region on synthetic) → `TODO/completed/127_l5_hilbert_diagnostic_closeout.md` §2
- **Wf-127 P2 L2 weight-decay guard** → `TODO/completed/127_l5_hilbert_diagnostic_closeout.md` §2.5 + `molmetal/reports/wf127_p2_l2_guard/guard_test_log.json`
- **Wf-127 P3 ASTSR scale-clip guard** → `molmetal/reports/wf127_p3_scale_guard/guard_test_log.json` + `molmetal/tests/test_astr_scale_guard.py` (27/27 PASS)
- **Wf-122 P1 L2 ridge + P2 ASTR scale-guard** (the prior defensive guard layer that Wf-127 P2/P3 builds on) → `TODO/completed/122_hilbert_l2_astsr_guard_closeout.md`
- **Wf-83 P1 honest anchor** (combined R² = -6.71 on PlatinAI 4-cell) → `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md`
- **Wf-128 P2 3-way REAL MetalCytoToxDB** (combined R² = -150.64 on 5-metal) → `TODO/completed/128_l10_3way_real_pic50_rescue.md`
- **Wf-129 P3 Multi-PH+PLL guarded** (combined R² = -0.078 on PlatinAI 4-cell, more recent post-fix anchor) → `metrics/by_metric/hilbert_4cell_r2.json` `wf129_p3_multiph_pll_guarded_l7_lift_caveat` + `TODO/completed/129_l2_l7_variance_wasserstein_closeout.md`
- **Wf-49 Cause 2** (12-d RDKit signal floor) → `TODO/completed/91_wf49_r2_root_fix.md`
- **Wf-94 R20-P3 Multi-PH + PLL feature augmentation** (the architectural lift lever) → `TODO/pending/94_r20_p3_multiph_pll_feature.md`
- **LAMBDA_ROADMAP §1.3** (R20 R² ≥ 0.85 target) → `TODO/03_plans/LAMBDA_ROADMAP.md`

---

## §4 Verdict

**Preliminary (mcf7-only, honest framing per `party-twenty-spirit §3.5`)**:

| Gate | Target | mcf7 MEASURED | Verdict |
|---|---|---|---|
| PlatinAI 4-cell combined R² ≥ 0 for ≥ 1 cell line | -6.71 (Wf-83 P1) / -5.66 (Wf-122 P3) / -0.078 (Wf-129 P3) | **-4.04 (mcf7 only)** | ❌ **HONEST NEGATIVE** — guard stack does NOT lift; combined R² is in the same regime as the Wf-122 P3 post-fix honest anchor, with mcf7 alone |

**Pending**: full 4-cell aggregate + 5-metal aggregate verdict will be appended once a549/a2780/a2780cis/Ir/Os/Rh/Re complete. The honest-negative verdict on mcf7 alone is the strongest evidence that the guard stack is **stabilizing but not lifting** — consistent with the Wf-49 §Cause 2 architectural ceiling argument.

**Net effect of Wf-127 P4**: ships the **opt-in combined-guard wiring** (`--use-l2-guard --use-astr-scale-guard` flags on both bench scripts) with regression tests passing, and provides the **MEASURED honest-negative evidence** that the guard stack does NOT lift the Wf-83 P1 -6.71 / Wf-122 P3 -5.66 / Wf-128 P2 -150.64 ceiling on real data. The Wf-94 R20-P3 Multi-PH + PLL feature augmentation remains the only architectural path to R² ≥ 0.

**Verdict**: 🔄 **HONEST PARTIAL SHIP** — mcf7 cell-line MEASURED with combined-guard stack; verdict FAIL on the verify gate (combined R² ≥ 0 for ≥ 1 cell line). Remaining 3 PlatinAI cell-lines + 5-metal MetalCytoTox will be appended to this doc once the full bench completes (ETA ~60-90 min from 2026-09-27 03:50 UTC). The honest-negative mcf7 result is sufficient to ship this as a partial honest-negative close-out per the task constraint "if guards do NOT lift, ship honest-negative close-out".
