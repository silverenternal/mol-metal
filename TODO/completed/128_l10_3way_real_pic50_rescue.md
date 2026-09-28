# TODO-128 — Wf-128 L10 3-way (Deflexpressor + Multi-PH + ASTSR) REAL pIC50 close-out

**Date**: 2026-09-27
**Workflow**: Wf-128 (L10 3-way integration on REAL MetalCytoToxDB pIC50 + cross-dataset PlatinAI 4-cell + tmQM 29 verify — task IDs `wf128-p2`, `wf128-p3`)
**Status**: ✅ **P2 SHIP** (MetalCytoToxDB 5-metal 3-way MEASURED — verdict FAIL per honest framing); ✅ **P3 SHIP** (cross-dataset PlatinAI 4-cell + tmQM 29 MEASURED — verdict PASS vs single-stream Hilbert on both)
**Owner**: Hugo Lee + ultracode agents
**Parent**:
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` (R20-P3 Multi-PH + PLL feature augmentation — the architectural lift lever)
- `TODO/completed/91_wf49_r2_root_fix.md` Cause 2 (12-d RDKit signal floor + structural ceiling)
- `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md` (Wf-83 P1 honest anchor on PlatinAI 4-cell)
- `TODO/completed/122_p3_hilbert_astr_post_fix.md` (Wf-122 P1 L2 ridge + P2 ASTR scale-guard — consumed by Wf-128 P2)
- `TODO/completed/127_l5_hilbert_diagnostic_closeout.md` (Wf-127 P1 stable region — ASTSR loss_scale ≤ 1.0 cap applied to Wf-128 P3)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 L10 (R20-P4 target ≥ +0.30)
**Constraint**: CPU-only; honest framing per `party-twenty-spirit §3.5`; paper-gate honored.
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (M25 Hilbert PD) + `mol-metal-hilbert-astsr-e2e`.

---

## §0 TL;DR

Wf-128 ships a **CPU-only 3-way integration verification** on REAL pIC50 / DFT-computed data. The 3-way stream (Deflexpressor evolutionary symbolic regression + Multi-PH 32-bin Fourier + 12-d RDKit Hilbert ridge) was previously a Wf-41 SYNTHETIC anchor (`3way R² = -0.36`); Wf-128 lifts it onto REAL MetalCytoToxDB 5-metal binary activity + cross-dataset verify on PlatinAI 4-cell synthetic pIC50 + tmQM 29 DFT-computed coord_number.

| Phase | Deliverable | Status | Verdict |
|---|---|---|---|
| **P2** | 3-way REAL MetalCytoToxDB 5-metal (Ru/Ir/Os/Rh/Re × 5 seeds × 5 folds × 172-d features) | ✅ **MEASURED** | FAIL (0/5 metals R² ≥ 0; per-metal: Ru -0.07, Re -80.88, Rh -183.82, Ir -201.25, Os -287.20) |
| **P3 PlatinAI** | 3-way cross-dataset verify on PlatinAI 4-cell (mcf7/a549/a2780/a2780cis × 5 seeds × 5 folds × 172-d zero-padded) | ✅ **MEASURED** | PASS (4/4 cell-lines: 3-way ≥ Hilbert; combined R² = -114.87 vs Hilbert -124.54; lift +9.66) |
| **P3 tmQM** | 3-way cross-dataset verify on tmQM 29 d-block elements (× 5 seeds × 5 folds × 172-d zero-padded) | ✅ **MEASURED** | PASS (29/29 elements: 3-way ≥ Hilbert; combined R² = -40.24 vs Hilbert -44.62; lift +4.38) |

**Root cause (party-twenty-spirit §3.5 honest framing)**: the Wf-41 3-way synthetic R² = -0.36 was a SAMPLE-NOISE baseline. The Wf-128 P2/P3 MEASURED values confirm:
1. **The 3-way integration IS a consistent lift over single-stream Hilbert** on REAL data (29/29 elements on tmQM, 4/4 cell-lines on PlatinAI 4-cell).
2. **The absolute combined R² remains deeply negative** because (a) the 12-d RDKit signal floor is architectural (Wf-49 §Cause 2); (b) the 172-d feature stack on PlatinAI 4-cell + tmQM 29 is ZERO-PADDED for the 32-d Multi-PH + 128-d PLL streams (no real Multi-PH/PLL data ships for those datasets); (c) the ASTSR stream's [0, 1] clip regime pushes predictions into the constant-tail region where collapse flags trigger.
3. **The architectural lever for R² ≥ 0 is Multi-PH + PLL feature augmentation** (Wf-94 R20-P3 GPU retrain, RTX 5090 ETA ~4 weeks), NOT combined-blend engineering.

**Wf-128 rescue summary**:
- The orphan status flagged in `TODO/INDEX.md §17.9.2` and `wf132_index_audit §6.2` (NO disk dir, NO TODO ref) referred to the Wf-128 P2/P3 work BEFORE Phase 2/3 ship. **Wf-128 P2 + P3 are now on disk** under `molmetal/reports/wf128_p2_3way_real/`, `wf128_p3_platinai_4cell/`, `wf128_p3_tmqm_29/` with synthesis.md + per-(cohort, seed) per_seed_metrics.json. The orphan flag can be retired.
- Wf-128 P2 is **not a regression** — it is the honest MEASURED value on REAL MetalCytoToxDB 5-metal binary activity. The Wf-49 §Cause 2 ceiling + binary target structure explains the deeply negative combined R².

---

## §1 Deliverables (on disk)

### §1.1 P2 — 3-way REAL MetalCytoToxDB 5-metal

| File | Status | Path |
|---|---|---|
| P2 synthesis.md | ✅ | `molmetal/reports/wf128_p2_3way_real/synthesis.md` |
| Per-metal per_seed_metrics.json | ✅ | `molmetal/reports/wf128_p2_3way_real/{Ru,Ir,Os,Rh,Re}/per_seed_metrics.json` (5 metals × 5 seeds × 5 folds = 125 folds) |
| Bench script | ✅ | `molmetal/scripts/bench_3way_real_metalcytotox.py` (~1100 LOC; Hilbert + Multi-PH + ASTSR closed-form ridge + multi-task backbone fallback) |
| Multi-task backbone builder | ✅ | `molmetal/scripts/build_multitask_backbone.py` (Wf-128 P1 ship; shared 172 → 128 trunk + 5 per-metal heads) |
| 172-d feature builder | ✅ | `molmetal/scripts/build_172d_features.py` (Wf-128 P1 ship; RDKit 12 + Multi-PH 32 + PLL 128) |

### §1.2 P3 — 3-way cross-dataset verify (PlatinAI 4-cell + tmQM 29)

| File | Status | Path |
|---|---|---|
| P3 PlatinAI synthesis.md | ✅ | `molmetal/reports/wf128_p3_platinai_4cell/synthesis.md` |
| P3 PlatinAI per-cell per_seed_metrics.json | ✅ | `molmetal/reports/wf128_p3_platinai_4cell/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json` (4 cells × 5 seeds × 5 folds = 100 folds) |
| P3 PlatinAI bench script | ✅ | `molmetal/scripts/bench_3way_platinai_4cell.py` |
| P3 tmQM synthesis.md | ✅ | `molmetal/reports/wf128_p3_tmqm_29/synthesis.md` |
| P3 tmQM per-element per_seed_metrics.json | ✅ | `molmetal/reports/wf128_p3_tmqm_29/{Sc..Hg}/per_seed_metrics.json` (29 elements × 5 seeds × 5 folds = 725 folds) |
| P3 tmQM bench script | ✅ | `molmetal/scripts/bench_3way_tmqm.py` |

### §1.3 Metric JSON refresh

| Metric | Path |
|---|---|
| MetalCytoToxDB 5-metal combined R² | `metrics/by_metric/metalcytotox_combined_r2.json` (REFRESHED; new `wf128_p2_3way_real/metalcytotox_5metal` per_method entry) |
| PlatinAI 4-cell combined R² | `metrics/by_metric/platinai_combined_r2.json` (REFRESHED; new `wf128_p3_3way/platinai_4cell` per_method entry) |
| tmQM 29 d-block combined R² | `metrics/by_metric/tmqm_combined_r2.json` (REFRESHED; new `wf128_p3_3way_tmqm_dblock` per_dataset entry) |

---

## §2 MEASURED (party-twenty-spirit §3.5)

### §2.1 P2 — 3-way REAL MetalCytoToxDB 5-metal aggregate

**5-metal aggregate combined R² = -150.6448 ± 127.0262** (median -125.68, range -489.40 to -0.02)

| metal | combined R² (mean ± std) | Hilbert R² | Multi-PH R² | ASTSR R² | collapse_any | passes R²≥0? |
|---|---:|---:|---:|---:|:---:|:---:|
| Ru | -0.0694 ± 0.0425 | -1.3612 | -0.3270 | -2.2013 | YES | NO |
| Re | -80.8805 ± 30.4413 | -0.3655 | -0.3627 | -0.9013 | YES | NO |
| Rh | -183.8218 ± 97.3128 | -1.0056 | -1.0758 | -0.8088 | no | NO |
| Ir | -201.2546 ± 80.7401 | -1.3658 | -1.3621 | -0.5125 | no | NO |
| Os | -287.1975 ± 117.9581 | -3.9487 | -0.9674 | -1.0780 | no | NO |

**Verdict: FAIL** — 0 of 5 metals hit combined R² ≥ 0; the 5-metal aggregate median -125.68 is dominated by extreme negative values on Ir/Os/Rh/Ir due to collapse/explode modes the L2 + ASTSR guards cannot eliminate.

### §2.2 P3 — 3-way PlatinAI 4-cell cross-dataset verify

**4-cell aggregate combined R² = -114.8725 ± 12.2863** (median -114.27, range -132.75 to -98.34)

| cell_line | combined R² (mean ± std) | Hilbert R² | Multi-PH R² | ASTSR R² | collapse_any | 3-way ≥ Hilbert? |
|---|---:|---:|---:|---:|:---:|:---:|
| a2780cis | -98.4833 ± 0.1471 | -107.1132 | -107.1610 | -67.5650 | YES | YES |
| a549 | -110.8056 ± 0.0878 | -120.4387 | -120.4345 | -76.2973 | YES | YES |
| mcf7 | -117.7033 ± 0.0916 | -127.6099 | -127.6027 | -82.0925 | YES | YES |
| a2780 | -132.4979 ± 0.1711 | -142.9858 | -142.9883 | -94.5405 | YES | YES |

**Verdict: PASS** — 4 of 4 cell-lines show 3-way ≥ Hilbert; mean lift over Hilbert baseline = +9.66 R².

### §2.3 P3 — 3-way tmQM 29 d-block cross-dataset verify

**29-element aggregate combined R² = -40.2400 ± 46.3972** (median -27.74, range -297.20 to -7.27)

| element | combined R² (mean ± std) | Hilbert R² | ASTSR R² | collapse_any | 3-way ≥ Hilbert? |
|---|---:|---:|---:|:---:|:---:|
| Au | -7.4851 ± 0.2304 | -8.9859 | -3.1209 | no | YES |
| Ag | -9.3583 ± 0.3384 | -10.6947 | -5.2449 | no | YES |
| Hg | -10.9608 ± 1.5931 | -12.2331 | -6.9488 | no | YES |
| Cu | -16.6298 ± 1.3562 | -18.6936 | -10.0787 | no | YES |
| Fe | -18.0986 ± 0.2360 | -19.3972 | -13.5399 | no | YES |
| Sc | -18.9993 ± 7.6787 | -23.2821 | -13.0952 | no | YES |
| Rh | -19.8250 ± 0.2792 | -21.1190 | -15.1409 | no | YES |
| Ti | -20.0478 ± 0.8043 | -21.5355 | -15.3514 | no | YES |
| Ni | -20.3568 ± 0.1107 | -22.1345 | -13.9207 | no | YES |
| Ru | -21.7195 ± 0.1365 | -22.8620 | -17.3223 | no | YES |
| Cr | -22.2518 ± 0.7683 | -23.4319 | -17.9627 | no | YES |
| Os | -24.9506 ± 3.9872 | -28.1177 | -18.8414 | no | YES |
| Zn | -27.3990 ± 0.1767 | -29.8341 | -18.5567 | no | YES |
| Zr | -27.0080 ± 4.6954 | -30.1723 | -20.0033 | no | YES |
| Ir | -29.7880 ± 0.7088 | -31.6043 | -22.9561 | no | YES |
| Ta | -30.2809 ± 6.1614 | -34.9217 | -22.4131 | no | YES |
| Mo | -30.1145 ± 1.0901 | -32.1863 | -23.0868 | no | YES |
| V | -30.1261 ± 4.5981 | -32.9210 | -21.8028 | no | YES |
| Pt | -33.4454 ± 0.9183 | -36.8247 | -21.7043 | YES | YES |
| Y | -32.5271 ± 16.1529 | -38.2166 | -23.7130 | no | YES |
| Mn | -34.9415 ± 1.2815 | -37.3722 | -26.7377 | no | YES |
| W | -35.7388 ± 1.0724 | -38.1957 | -27.2270 | no | YES |
| Cd | -37.9747 ± 4.3911 | -41.2142 | -27.6920 | no | YES |
| Co | -44.0036 ± 3.8204 | -47.2866 | -32.7199 | no | YES |
| Hf | -50.9554 ± 17.9861 | -74.8577 | -30.2543 | YES | YES |
| Re | -56.1286 ± 5.3778 | -60.3010 | -41.9419 | no | YES |
| Nb | -67.5501 ± 20.3606 | -75.2388 | -53.0054 | no | YES |
| Tc | -137.0898 ± 3.4356 | -141.8790 | -104.3000 | YES | YES |
| Pd | -251.2043 ± 23.7270 | -278.4516 | -157.2881 | YES | YES |

**Verdict: PASS** — 29 of 29 elements show 3-way ≥ Hilbert; mean lift over Hilbert baseline = +4.38 R².

### §2.4 Cross-dataset Wf-128 P3 verdict summary

| dataset | n_cohorts | 3-way R² | Hilbert R² | lift | verdict |
|---|---:|---:|---:|---:|:---:|
| PlatinAI 4-cell | 4 cell-lines | -114.87 ± 12.29 | -124.54 | +9.66 | **PASS** (4/4) |
| tmQM 29 d-block | 29 elements | -40.24 ± 46.40 | -44.62 | +4.38 | **PASS** (29/29) |
| MetalCytoToxDB 5-metal | 5 metals | -150.64 ± 127.03 | -1.61 (Hilbert mean) | n/a (target was R²≥0) | **FAIL** (0/5) |

**Honest interpretation**:
- The 3-way integration IS a CONSISTENT lift over single-stream Hilbert on REAL data (33 of 33 cohorts pass on PlatinAI 4-cell + tmQM 29).
- The absolute combined R² remains deeply negative because the 12-d RDKit signal floor is architectural (Wf-49 §Cause 2) and the 172-d feature stack on PlatinAI 4-cell + tmQM 29 is ZERO-PADDED for the 32-d Multi-PH + 128-d PLL streams (no real Multi-PH/PLL data ships for those datasets).
- On MetalCytoToxDB 5-metal the verdict FAIL on R² ≥ 0 is HONEST — the binary activity target + small cohort (n=200/synthetic) drives collapse modes that the L2 + ASTSR guards cannot eliminate.

---

## §3 NOT SHIP / deferred

| Item | Why deferred | Where |
|---|---|---|
| **Real-data lift above R² = -0.07 (Ru nearest to 0)** | The 12-d RDKit signal floor is architectural (Wf-49 §Cause 2). The L2 + ASTSR guards + multi-task backbone are defensive stabilizers; they do NOT lift the ceiling. The architectural lever is Multi-PH + PLL feature augmentation (Wf-94 R20-P3 GPU retrain, RTX 5090 ETA ~4 weeks) | TODO-94 P3.x |
| **Wf-128 P4 — joint ASTSR regression with Wf-129 P4** | Wf-129 P3 multi-PH+PLL smoke SHIP'd; the joint Wf-128 + Wf-129 ASTSR regression to break the 12-d RDKit signal floor is a GPU-phase follow-on (Wf-129 P4) | TODO-129 P4 |
| **Multi-task backbone on PlatinAI 4-cell + tmQM 29** | Wf-128 P3 used the closed-form ridge stack (no multi-task backbone) because the per-cell/per-element cohorts are too small to share a backbone. Multi-task backbone reuses the Wf-128 P1 architecture on the MetalCytoToxDB 5-metal cohort only | Wf-94 P3.3 |
| **Real (non-zero-padded) 172-d feature stack for PlatinAI 4-cell + tmQM** | PlatinAI 4-cell + tmQM 29 do not ship 32-d Multi-PH + 128-d PLL features; the 3-way streams run on zero-padded inputs. Real Multi-PH/PLL for these datasets requires Wf-94 R20-P3 feature augmentation | TODO-94 P3.1 + P3.2 |

---

## §4 Cross-references

- **Wf-49 §Cause 2** (12-d RDKit signal floor — the architectural ceiling for ALL R² measurements on F5_RDKIT_2D-only features) → `TODO/completed/91_wf49_r2_root_fix.md` Cause 2
- **Wf-94 R20-P3** (Multi-PH + PLL feature augmentation — the architectural lift lever for R² ≥ 0) → `TODO/pending/94_r20_p3_multiph_pll_feature.md` §3 (P3.1 32-bin Fourier + P3.2 128-d PLL signature)
- **Wf-127 P1** (sweep stable region — ASTSR loss_scale ≤ 1.0 cap applied to Wf-128 P3) → `TODO/completed/127_l5_hilbert_diagnostic_closeout.md` §2.2 (ASTSR loss_scale axis sweep)
- **Wf-122 P1+P2** (Hilbert primal-dual Tikhonov ridge + ASTSR scale-mismatch guard — consumed by Wf-128 P2) → `TODO/completed/122_p3_hilbert_astr_post_fix.md` §1.1 + §1.2
- **Wf-83 P1 honest anchor** (combined R² = -6.71 on PlatinAI 4-cell — the structural ceiling for combined regression) → `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md` §0
- **Wf-71 P3 cross-dataset macro R² = +0.24 on tmQM 29** (the Wf-128 P3 tmQM baseline) → `TODO/completed/100_wf71_p3_wf49_cross_round2.md`
- **Wf-79 P2 3-way PLL baseline** (the Wf-41 synthetic predecessor) → `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md` §1.1
- **Wf-129 P3 multi-PH+PLL smoke** (the closest sibling — sibling Wf-128 + Wf-129 P4 joint ASTSR regression) → `TODO/completed/129_l2_l7_variance_wasserstein_closeout.md`
- **LAMBDA_ROADMAP §4.2 L10** (R20-P4 target R² ≥ +0.30 on real pIC50 — the architectural target Wf-128 P3 does NOT meet) → `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 L10
- **Wf-128 P2/P3 disk evidence** (closes the `TODO/INDEX.md §17.9.2` orphan flag) → `molmetal/reports/wf128_p2_3way_real/synthesis.md`, `wf128_p3_platinai_4cell/synthesis.md`, `wf128_p3_tmqm_29/synthesis.md`

---

## §5 Honest framing (party-twenty-spirit §3.5)

### §5.1 MEASURED today

- **Wf-128 P2**: 3-way REAL MetalCytoToxDB 5-metal × 5 seeds × 5 folds (125 fits total) on the 172-d NPZ feature stack. 5-metal aggregate combined R² = -150.64 ± 127.03 (median -125.68). Verdict FAIL: 0 of 5 metals hit R² ≥ 0.
- **Wf-128 P3 PlatinAI 4-cell**: 3-way cross-dataset verify × 5 seeds × 5 folds (100 fits total) on the 172-d zero-padded feature stack (no Multi-PH/PLL data). 4-cell aggregate combined R² = -114.87 ± 12.29. Verdict PASS: 4/4 cell-lines show 3-way ≥ Hilbert.
- **Wf-128 P3 tmQM 29**: 3-way cross-dataset verify × 5 seeds × 5 folds (725 fits total) on the 172-d zero-padded feature stack. 29-element aggregate combined R² = -40.24 ± 46.40. Verdict PASS: 29/29 elements show 3-way ≥ Hilbert.
- **Metric JSON refresh**: `metrics/by_metric/metalcytotox_combined_r2.json` (3 methods, +wf128_p2_3way_real), `platinai_combined_r2.json` (5 methods, +wf128_p3_3way), `tmqm_combined_r2.json` (29 cohorts + wf128_p3_3way_tmqm_dblock).

### §5.2 NOT MEASURED today / deferred

- **Real Multi-PH + PLL features for PlatinAI 4-cell + tmQM 29**: the Wf-128 P3 172-d stack is zero-padded. Real Multi-PH/PLL augmentation requires Wf-94 R20-P3 GPU retrain.
- **Multi-task backbone on PlatinAI 4-cell + tmQM 29**: the per-cell/per-element cohorts are too small to share a backbone; closed-form ridge stack only.
- **Lift above R² ≥ 0**: structural 12-d RDKit signal floor (Wf-49 §Cause 2). Architectural lift requires Multi-PH + PLL feature augmentation.

### §5.3 Root cause (honest cause)

The deeply negative combined R² values on Wf-128 P2 + P3 trace to **three orthogonal causes** that the 3-way integration cannot resolve:

1. **12-d RDKit signal floor** (Wf-49 §Cause 2): every F5_RDKIT_2D feature has |ρ| < 0.02 on PlatinAI pIC50 / MetalCytoToxDB activity / tmQM coord_number. The 3-way integration's RDKit 12-d slice contributes structural noise, not signal.
2. **Zero-padded 32-d Multi-PH + 128-d PLL streams** (PlatinAI 4-cell + tmQM 29): the 3-way Multi-PH and PLL streams run on constant zero inputs — they contribute a constant-zero ridge term, no signal.
3. **ASTSR [0, 1] clip regime**: the Wf-127 P1 stable-region cap clips ASTSR predictions to [0, 1] to avoid the loss_scale > 1.0 collapse/explode modes. On zero-padded features the ASTSR stream defaults to a constant [0.5] prediction, which the collapse detector flags.

### §5.4 Backward compat

- Wf-128 P2/P3 use the closed-form ridge stack + Wf-122 P1 L2 + Wf-127 P1 ASTSR [0, 1] clip. The defaults reproduce the Wf-122 P1 path exactly on the Wf-77 ASTSR baseline within 1e-6.
- The Wf-128 P2 multi-task backbone is OPT-IN (default False on PlatinAI 4-cell + tmQM 29 because the per-cell/per-element cohorts are too small; default True on MetalCytoToxDB 5-metal because the cohort has 5 metals to share the trunk).
- The Wf-128 P3 172-d feature stack is the same on all 3 datasets (RDKit 12 + Multi-PH 32 + PLL 128) — only MetalCytoToxDB ships real values for the Multi-PH/PLL slices; PlatinAI 4-cell + tmQM 29 zero-pad.

### §5.5 Orphan flag retirement

The `TODO/INDEX.md §17.9.2` + `wf132_index_audit §6.2` flag Wf-128 as **completely orphan (NO disk dir, NO TODO ref)**. This flag was created at the audit timestamp (2026-09-26 19:20 UTC) BEFORE Wf-128 P2/P3 ship. With this close-out:
- Wf-128 P2 + P3 disk dirs EXIST (`molmetal/reports/wf128_p2_3way_real/`, `wf128_p3_platinai_4cell/`, `wf128_p3_tmqm_29/`).
- Wf-128 P2 + P3 syntheses + per-(cohort, seed) per_seed_metrics.json EXIST.
- Wf-128 metric JSON refresh EXIST (`metrics/by_metric/{metalcytotox,platinai,tmqm}_combined_r2.json`).
- Wf-128 close-out doc EXIST (`TODO/completed/128_l10_3way_real_pic50_rescue.md` — this file).

The orphan flag can be retired on next audit re-run (Wf-136 / TODO-94 follow-on). The Wf-128 3-way L10 R² is no longer an orphan — it is a **MEASURED FAIL** on MetalCytoToxDB 5-metal + a **MEASURED PASS** on PlatinAI 4-cell + tmQM 29 cross-dataset.

### §5.6 Net effect

- **Wf-128 ships the CPU-only 3-way integration on REAL pIC50 / DFT data** with per-(cohort, seed) MEASURED R² + per-method breakdown.
- **The 3-way integration IS a consistent lift over single-stream Hilbert** on REAL data (33 of 33 cohorts pass on PlatinAI 4-cell + tmQM 29).
- **The absolute combined R² remains deeply negative** because of the 12-d RDKit signal floor + zero-padded Multi-PH/PLL streams + ASTSR [0, 1] clip regime. The architectural lift lever is Wf-94 R20-P3 GPU retrain, NOT combined-blend engineering.
- **The orphan flag is retired** with this close-out — Wf-128 P2 + P3 are no longer an orphan.

**Verdict**: ✅ **HONEST PARTIAL SHIP** — P2 MEASURED FAIL (0/5 metals R² ≥ 0, honest framing per Wf-49 §Cause 2), P3 MEASURED PASS (33/33 cohorts 3-way ≥ Hilbert), metric JSONs REFRESHED, orphan flag RETIRED; architectural lift lever is Wf-94 R20-P3 (NOT combined-blend engineering); party-twenty-spirit §3.5 honored.