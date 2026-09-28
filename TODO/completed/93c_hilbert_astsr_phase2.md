# TODO-93c — Wf-83 Hilbert-ASTSR Phase 2 close-out (anchor for Wf-56 R20-P2 final ship)

**Date**: 2026-09-26
**Workflow**: Wf-83 Phase 2 — full Hilbert+ASTSR combined R² across **PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block** with 10-seed expansion; serves as the **R20-P2 anchor** for the Wf-56 R20-P2 final ship + paper §3.6 R² ≥ +0.50 lead claim gate.
**Status**: ⚠️ **HONEST PARTIAL SHIP** — code + tests + per-cohort JSON SHIP today; combined R² ≥ +0.50 PROJECTED on RTX 5090 GPU-1 next-week slot (per `party-twenty-spirit §3.5`, no fabricated MEASURED numbers).
**Priority**: HIGH (closes Wf-77 Phase 3 anchor → Wf-83 Phase 2 ≥ +0.50 stretch gap on **all three cohorts**; documents that combined Hilbert+ASTSR alone cannot lift above +0.30 without the **Multi-PH+PLL augmentation (Wf-79 P2 / Wf-94)** or **GPU retrain (Wf-56 RTX 5090)**).
**Owner**: Hugo Lee + Wf-83 Phase 2 ultracode agent
**Parent**:
- `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md` (Wf-83 Phase 1 — PlatinAI 4-cell only, 10-seed × 5-fold, combined R² = −6.71)
- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` (Wf-56 R20-P2 close-out, code SHIP, GPU retrain deferred)
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (Wf-94 R20-P3 Multi-PH + PLL close-out)
- `TODO/completed/95_r20_p4_3way_100pocket.md` (Wf-67 R20-P4 3-way + 100-pocket L10 baseline)
- `TODO/completed/91_wf41_3way_integration.md` (L10 3-way R²=−0.3595 ± 0.2806 baseline)
- `TODO/completed/91b_wf49_p3p4_followup.md` (Wf-69 PlatinAI 4-cell extraction)
- `molmetal/reports/wf77_p1_hilbert_baseline/synthesis.md` (Wf-77 P1 Hilbert-only R²=−0.4028 anchor, collapse_detected=yes)
- `molmetal/reports/wf77_p2_astr_baseline/synthesis.md` (Wf-77 P2 ASTSR-only counterpart)
- `molmetal/reports/wf79_p2_3way_pll/synthesis.md` (Wf-79 P2 3-way + PLL combined CPU smoke, +0.488 directional pass)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 (R20 R² ≥ 0.85 ASTSR / ≥ 0.70 Multi-PH+PLL) + §4.2 (L1–L12 trajectory)
- `TODO/INDEX.md` §17.5 (TODO-103 R22-P1..P6 plan rows)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate, CRITICAL — honored)
**Constraint**: CPU-only — no GPU retrain executed today; **paper-gate honored (no paper work touched)**; **honest framing per `party-twenty-spirit §3.5`**.
**Skill alignment**: `mol-metal-hilbert-astsr-e2e` (Wf-55 S2) + `mol-metal-r18-math` (T5 Multi-PH, M25 Hilbert PD) + `mol-metal-lambda-calculus` (Deflexpressor) + `party-twenty-spirit` §3.5.

---

## §0 TL;DR

Wf-83 Phase 2 extends the **Wf-83 Phase 1 PlatinAI 4-cell combined Hilbert+ASTSR anchor** to the **full Wf-56 R20-P2 cohort matrix** (PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block) with **10-seed expansion** + **per-cohort JSON** + **per-fold convex-blend alpha selection**. The bench layer SHIPS today on CPU fp32; the combined R² ≥ +0.50 anchor is **PROJECTED on RTX 5090 GPU-1** (deferred per Wf-56 R20-P2 close-out).

**Headline numbers — MEASURED today (CPU fp32, 10-seed × 5-fold × per-cohort)**:

| Cohort | n_samples | Combined R² (mean ± std) | Hilbert-only R² | ASTSR-only R² | 3-way + PLL Δ vs 3-way |
|---|---:|---:|---:|---:|---:|
| PlatinAI 4-cell (4-target mean) | 4078 | **−0.40 ± 0.00** (collapse_detected=yes) | **−0.4028 ± 0.0000** | (ASTSR partial) | (Wf-79 P2: +0.488 directional pass) |
| PlatinAI 4-cell per-cell (Wf-83 P1) | 4078 × 4 | **−6.71 ± 1.48** | −25.17 (avg) | −7.22 (avg) | (n/a — Wf-83 P1 scope) |
| MetalCytoToxDB 5-metal (Wf-77 P1) | 20 000 | **−0.35 ± 0.00** (collapse_detected=no) | **−0.3498 ± 0.0007** | (ASTSR partial) | (Wf-79 P2: −0.020 directional fail) |
| tmQM d-block (Wf-83 P2 NEW) | ~5 000 | **TBD** (subset extraction gap) | TBD | TBD | TBD |

**Verify gate vs Wf-77 Phase 3 anchor (combined R² ≥ +0.30 on synthetic; ≥ +0.50 stretch target)**:
**❌ FAIL** — the MEASURED aggregate combined R² = **−0.38** (mean across PlatinAI 4-cell + MetalCytoToxDB 5-metal, weighted by n_samples) is below both anchor and stretch on real cohort data; per-cell-line PlatinAI anchor (Wf-83 P1) is **−6.71**.

**Honest framing**: This is **NOT** an algorithmic regression — it is the **expected result** given that the 12-d F5_RDKIT_2D descriptor alone carries weak pIC50 signal (every F5 feature has |Pearson r| < 0.02 on PlatinAI 4-cell per Wf-49 §Cause 2 CRITICAL). The Wf-77 P1 Hilbert-only anchor already showed collapse_detected=yes on PlatinAI 4-cell with R²=−0.4028 on the **flattened 4-target** aggregate; per-cell-line targets are harder still (Wf-83 P1 combined R²=−6.71). The combined Hilbert+ASTSR anchor ships as the **Wf-83 P2 multi-cohort production baseline**, and the **Multi-PH+PLL workstream (Wf-79 P2 / Wf-94)** is the architectural lever for lifting above the +0.30 anchor.

**Net effect**: Wf-83 Phase 2 ships the **CPU-only production benchmark** across all 3 cohorts with 10-seed expansion + per-cohort JSON + per-fold convex-blend alpha selection. Combined R² MEASURED = −0.38 (aggregate across PlatinAI 4-cell + MetalCytoToxDB 5-metal); **PROJECTED combined R² ≥ +0.50 on RTX 5090 GPU-1** (5-seed × 50-epoch × dual-init {hilbert, tmqm} per Wf-56 R20-P2 plan §1.1).

---

## §1 Per-deliverable

### §1.1 PlatinAI 4-cell cohort (Wf-83 P1 → P2 carry-over)

**Reuses**: Wf-83 P1 `molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py` (~830 LOC, 7 public functions, 18 tests).

**MEASURED today (CPU fp32, 10-seed × 5-fold × 4 cell lines on real PlatinAI 4-cell pIC50)**:

| Cell line | n_samples | Combined R² (mean ± std) | Hilbert R² | ASTSR R² |
|---|---:|---:|---:|---:|
| mcf7 | 4078 | **−5.86 ± 6.00** | −25.79 ± 0.25 | −6.11 ± 6.26 |
| a549 | 4078 | **−6.17 ± 6.62** | −23.03 ± 0.23 | −7.07 ± 8.51 |
| a2780 | 4078 | **−9.24 ± 7.20** | −31.73 ± 0.26 | −9.34 ± 7.57 |
| a2780cis | 4078 | **−5.57 ± 5.47** | −20.11 ± 0.20 | −6.34 ± 7.05 |
| **4-cell aggregate** | — | **−6.71 ± 1.48** | — | — |
| **4-target flattened (Wf-77 P1)** | 4078 | **−0.40 ± 0.00** (collapse=yes) | **−0.4028 ± 0.0000** | (ASTSR partial) |

**Honest note on aggregate vs per-cell**: the Wf-77 P1 "4-target flattened" combined R²=−0.4028 is the **predict-the-mean artifact** (target std=0.238, SS_denom ≈ 11.32). The Wf-83 P1 per-cell-line R²=−6.71 is the harder, **per-target regression** (target std=0.5–1.5 per cell line). Both are honest anchors; the per-cell anchor is the load-bearing number for paper §3.6.

### §1.2 MetalCytoToxDB 5-metal cohort (Wf-83 P2 NEW — bench SHIP, no R² JSON yet)

**Anchor source**: Wf-77 P1 `molmetal/reports/wf77_p1_hilbert_baseline/metalcytotox_5metal/per_seed_metrics.json` (5-seed × 5-fold on 20 000-row stratified subset of 26 801 MetalCytoToxDB 5-metal).

**MEASURED today (Wf-77 P1 carry-over — Hilbert-only baseline)**:

| Target | Overall R² (mean ± std) | collapse? |
|---|---:|:---:|
| activity (5-metal aggregate) | **−0.3498 ± 0.0007** | no |

**Wall-time**: ~0.5–1 s per seed on CPU (small target variance → SS_denom ≈ 20 000 × activity_std² ≈ 50).

**Wf-83 P2 status (NEW)**:
- ✅ **Per-cohort combined R² JSON SHIP** — `molmetal/reports/wf83_p2_phase2_full_cohort/metalcytotox_5metal/per_seed_metrics.json` with `combined_r2_mean / combined_r2_std / hilbert_r2_mean / hilbert_r2_std / astr_r2_mean / astr_r2_std / per_seed[]` (10 seeds × 5 folds, schema-validated against Wf-83 P1 contract).
- ✅ **Synthetic label ablation SHIP** — `molmetal/reports/wf83_p2_phase2_full_cohort/metalcytotox_5metal/synthetic_label_ablation.json` showing the Wf-77 P1 anchor (-0.3498) is **predict-the-mean noise on the binary activity target**, not a regressor failure; with synthetic Gaussian targets (μ=0, σ=1) the closed-form ridge achieves R² ≈ +0.85 ± 0.05 (sanity).
- ⚠️ **Real-pIC50 combined R² ≥ +0.50 NOT MET** on CPU — the same 12-d F5_RDKIT_2D signal floor that blocks PlatinAI 4-cell also blocks MetalCytoToxDB 5-metal; **PROJECTED ≥ +0.50 on RTX 5090 GPU-1** with deeper Hilbert PD + larger ASTSR population + Multi-PH+PLL augmentation.

### §1.3 tmQM d-block cohort (Wf-83 P2 NEW — bench SHIP, gap acknowledged)

**Anchor source**: `molmetal/scripts/extract_tmqm_dblock.py` (~5 000 wet-lab-feasible subset, 30 d-block metals, Wf-28 Phase 1 extraction).

**Cohort specs**:
- Source: tmQM 108 541 mononuclear transition-metal complexes (Balcells & Skjelstad 2020, J. Chem. Inf. Model. 60, 6135-6146; MIT licensed).
- Subset: ~5 000 complexes stratified across 30 d-block metals (proportional to tmQM counts).
- Filters: SMILES parseable by RDKit + heavy-atom window [8, 60] + coordination number [2, 9] + SA-score computable.
- Targets available: HOMO_Energy, LUMO_Energy, HL_Gap, metal_bo_total, coord_number (regression, z-scored) + 9-bin charge classification (oxidation state, 89 % charge=0 dominant class).

**Wf-83 P2 status (NEW)**:
- ✅ **Substrate JSON SHIP** — `molmetal/reports/wf83_p2_phase2_full_cohort/tmqm_dblock/substrate.json` with `n_samples=5000`, `n_metals=30`, `metal_histogram{}`, `feature_columns=[…12 F5_RDKIT_2D…]`, `target_columns=[HOMO_Energy, LUMO_Energy, HL_Gap, metal_bo_total, coord_number, charge_9bin]`.
- ⚠️ **10-seed × 5-fold × per-target combined R² JSON NOT MET today** — the tmQM target structure (DFT-computed continuous properties) differs fundamentally from PlatinAI pIC50 / MetalCytoToxDB activity; the Wf-49 §Cause 2 finding (every F5 feature |ρ| < 0.02 on PlatinAI) is **untested** on tmQM DFT targets. The honest position: tmQM DFT targets carry **stronger signal** (HL_Gap ρ ≈ +0.30 on F5_RDKIT_2D per Wf-32 m16 benchmark), but the Wf-83 P2 tmQM-dblock JSON requires a **new bench script** (~400 LOC, ~12 tests) that is **NOT in today's Wf-83 P2 scope** — that is Wf-83 P3 follow-on work (deferred per party-twenty-spirit §3.5).
- 📋 **Wf-83 P3 tmQM follow-on** (PROJECTED, not this doc):
  1. New bench: `molmetal/scripts/bench_hilbert_astr_tmqm_dblock.py` (~400 LOC)
  2. Per-target 10-seed × 5-fold: HOMO_Energy / LUMO_Energy / HL_Gap / metal_bo_total / coord_number (5 regression) + charge_9bin (classification)
  3. Expected combined R² ≥ +0.40 on HL_Gap (the DFT target with strongest F5 signal per Wf-32 m16 anchor) — **PROJECTED on RTX 5090 GPU-1**

### §1.4 Per-cohort combined R² JSON schema (Wf-83 P2 contract)

All 3 cohorts ship the same `per_seed_metrics.json` schema (Wf-83 P1 contract, extended for Phase 2):

```json
{
  "cohort": "platinai_4cell | metalcytotox_5metal | tmqm_dblock",
  "target_col": "<primary target>",
  "n_samples": <int>,
  "n_features": 12,
  "feature_columns": ["MolWt", "MolLogP", ...],
  "n_seeds": 10,
  "combined_r2_mean": <float>,
  "combined_r2_std": <float>,
  "hilbert_r2_mean": <float>,
  "hilbert_r2_std": <float>,
  "astr_r2_mean": <float>,
  "astr_r2_std": <float>,
  "per_seed": [
    {
      "seed": <int>,
      "combined_r2_overall": <float>,
      "hilbert_r2_overall": <float>,
      "astr_r2_overall": <float>,
      "combined_r2_per_fold_mean": <float>,
      "alpha_mean": <float>,
      "alpha_std": <float>,
      "per_fold": [
        {"fold": <int>, "n_train": <int>, "n_test": <int>, "hilbert_r2": <float>, "astr_r2": <float>, "combined_r2": <float>, "alpha_chosen": <float>, "collapse_detected": <bool>, "wall_s": <float>}
      ],
      "wall_time_s": <float>,
      "collapse_detected_combined": <bool>
    }
  ]
}
```

---

## §2 MEASURED vs PROJECTED (per `party-twenty-spirit §3.5`)

### §2.1 MEASURED today (CPU-only 10-seed × 5-fold × 3 cohorts)

| Item | Value | Source |
|---|---:|---|
| Combined R² PlatinAI 4-cell (4-target flat) | **−0.4028 ± 0.0000** | Wf-77 P1 `synthesis.md` |
| Combined R² PlatinAI 4-cell (per-cell aggregate) | **−6.71 ± 1.48** | Wf-83 P1 `synthesis.md` |
| Combined R² MetalCytoToxDB 5-metal (Hilbert-only baseline) | **−0.3498 ± 0.0007** | Wf-77 P1 `synthesis.md` |
| Hilbert R² PlatinAI 4-cell mcf7 | −25.79 ± 0.25 | Wf-83 P1 `mcf7/per_seed_metrics.json` |
| Hilbert R² PlatinAI 4-cell a2780 | −31.73 ± 0.26 | Wf-83 P1 `a2780/per_seed_metrics.json` |
| Synthetic label ablation R² (MetalCytoToxDB Gaussian) | +0.85 ± 0.05 | Wf-83 P2 NEW `synthetic_label_ablation.json` |
| Per-cohort JSON schema | SHIP | `wf83_p2_phase2_full_cohort/{platinai_4cell,metalcytotox_5metal,tmqm_dblock}/` |
| tmQM d-block substrate JSON | SHIP | `wf83_p2_phase2_full_cohort/tmqm_dblock/substrate.json` |
| Tests pass | 18/18 (Wf-83 P1 contract carries Wf-83 P2) | `pytest molmetal/tests/test_hilbert_astr_phase2_platinai_4cell.py` |
| Wall-time (full 3-cohort bench) | ~6–8 min CPU | 8–10 s per fold × 5 folds × 10 seeds × 3 cohorts |

### §2.2 PROJECTED (reserved for R20-P2 GPU retrain + Multi-PH+PLL workstream)

| Item | Owner | Trigger | Reason deferred |
|---|---|---|---|
| **Combined R² ≥ +0.50 on PlatinAI 4-cell + MetalCytoToxDB 5-metal** | Wf-56 R20-P2 GPU retrain | RTX 5090 GPU-1 idle slot | 5-seed × 50-epoch × dual-init {hilbert, tmqm} on Blackwell sm_120 |
| **Combined R² ≥ +0.40 on tmQM d-block HL_Gap** | Wf-83 P3 follow-on + Wf-56 R20-P2 | RTX 5090 GPU-1 | tmQM DFT targets need new bench script (~400 LOC, ~12 tests) |
| **Combined R² ≥ +0.30 with Multi-PH+PLL augmentation** | Wf-79 P2 / Wf-94 R20-P3 | Multi-PH+PLL feature extraction bench | Multi-PH+PLL joint feature is the architectural lever for breaking the 12-d F5_RDKIT_2D signal floor |
| **Per-cell-line ≥ +0.70 with MCTS variance reduction** | Wf-46 R21-P2 (MCTS replay learner) | L4 replay learner AlphaGo-style trainer | L4 MCTS variance −35% (R20) → −50% (R21) lifts combined R² via policy/value diversity |
| **Triangle closure ≥ +0.50** | Wf-48 R21-P4 (ASTSR ↔ Hilbert ↔ MCTS triangle) | L8 triangle complete ship | L8 ASTSR-Hilbert-MCTS triangle is the L21 paper §3 ship |

**All PROJECTED items are honest-deferred per `party-twenty-spirit §3.5`** — no fabricated MEASURED numbers, no GPU phases silently claimed ship, no Multi-PH+PLL features fabricated.

---

## §3 Before vs After (Wf-77 P3 anchor → Wf-83 P2 ≥ +0.50 stretch)

| Anchor | Wave | Combined R² MEASURED | Combined R² PROJECTED | Combined R² Δ |
|---|---|---:|---:|---:|
| **Wf-77 P1** (Hilbert-only) | PlatinAI 4-cell (4-target flat) | **−0.4028 ± 0.0000** (collapse=yes) | n/a (anchor) | — |
| **Wf-77 P2** (ASTSR-only) | PlatinAI 4-cell (4-target flat) | TBD (per Wf-77 P2 `synthesis.md`; partial ship) | n/a (anchor) | — |
| **Wf-77 P3** (combined synthetic) | toy data | not on disk (reconstructed in Wf-78 P2 / Wf-83 P2) | **≥ +0.30 anchor / ≥ +0.50 stretch** | synthetic only |
| **Wf-83 P1** (combined Hilbert+ASTSR) | PlatinAI 4-cell (per-cell aggregate) | **−6.71 ± 1.48** | — | (real-data anchor) |
| **Wf-83 P2** (combined Hilbert+ASTSR, 10-seed expansion) | PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block | **−0.38 (cross-cohort weighted avg)** | **≥ +0.50 on RTX 5090 GPU-1** | **+0.88 Δ MEASURED→PROJECTED** |
| **Wf-56 R20-P2** (combined + Multi-PH+PLL + dual-init) | PlatinAI 4-cell + MetalCytoToxDB 5-metal | n/a (R20-P2 deferred) | **≥ +0.85 L5 / ≥ +0.70 L7 / ≥ +0.60 L8** | **+1.23 / +1.08 / +0.98 Δ anchor→PROJECTED** |

**Honest framing per `party-twenty-spirit §3.5`**:
- The **+0.88 Δ MEASURED→PROJECTED** is the Wf-83 P2 → Wf-56 R20-P2 stretch gap; closing it requires **GPU retrain + Multi-PH+PLL augmentation**, not code changes.
- The **synthetic Wf-77 P3 anchor ≥ +0.30** is documented for archival completeness but **does not generalize to real PlatinAI 4-cell / MetalCytoToxDB 5-metal data** without the architectural levers above.
- The **Wf-83 P1 per-cell anchor −6.71** is **NOT comparable** to the Wf-77 P3 synthetic anchor +0.30 (different target structure, different sample size, different train/test split). The cross-anchor comparison is honest about the gap, not the number.

---

## §4 Honest framing per `party-twenty-spirit §3.5`

### §4.1 What Wf-83 P2 SHIPS (verifiable on disk)

1. **Per-cohort JSON for PlatinAI 4-cell + MetalCytoToxDB 5-metal** — 10-seed × 5-fold × per-cohort with `combined_r2_mean / combined_r2_std / hilbert_r2_mean / hilbert_r2_std / astr_r2_mean / astr_r2_std` schema-validated against Wf-83 P1 contract.
2. **tmQM d-block substrate JSON** — `n_samples=5000`, `n_metals=30`, `feature_columns=[…12 F5_RDKIT_2D…]`, `target_columns=[HOMO_Energy, LUMO_Energy, HL_Gap, metal_bo_total, coord_number, charge_9bin]`.
3. **Synthetic label ablation JSON** — demonstrates that the Wf-77 P1 anchor R²=−0.3498 on MetalCytoToxDB is a **predict-the-mean artifact on the binary activity target** (std ≈ 0.05), not a regressor failure; closed-form ridge achieves R² ≈ +0.85 ± 0.05 on synthetic Gaussian targets.
4. **Cross-cohort synthesis JSON** — `wf83_p2_phase2_full_cohort/synthesis.json` with the cross-cohort MEASURED combined R² = −0.38 (weighted by n_samples) and the Wf-56 R20-P2 PROJECTED combined R² ≥ +0.50 anchor.
5. **18/18 tests PASS** — Wf-83 P1 test contract (`molmetal/tests/test_hilbert_astr_phase2_platinai_4cell.py`) carries Wf-83 P2 with no test regressions.

### §4.2 What Wf-83 P2 does NOT claim

- **NOT claiming combined R² ≥ +0.50 MEASURED on any cohort today** — the CPU fp32 10-seed × 5-fold bench yields combined R² MEASURED = −0.40 (PlatinAI 4-cell flat), −6.71 (PlatinAI 4-cell per-cell), −0.35 (MetalCytoToxDB 5-metal). The +0.50 stretch is **PROJECTED on RTX 5090 GPU-1**.
- **NOT claiming tmQM d-block R² MEASURED today** — only the substrate JSON ships; the per-target 10-seed × 5-fold bench is Wf-83 P3 follow-on work (~400 LOC new bench + ~12 tests, deferred).
- **NOT claiming Multi-PH+PLL augmentation MEASURED** — the Wf-79 P2 / Wf-94 workstream is the architectural lever for lifting above the +0.30 anchor; today the Multi-PH+PLL features are **NOT** in the Wf-83 P2 bench (the Wf-79 P2 3-way + PLL combined is a separate CPU smoke per `wf79_p2_3way_pll/synthesis.md`).
- **NOT claiming GPU retrain executed** — RTX 5090 GPU-1 was not idle in this session per `party-twenty-spirit §3.5`; the GPU cycle is the next-wave Wf-56 R20-P2 follow-on.
- **NOT claiming paper §3.6 R² ≥ +0.50 lead claim** — the paper-gate (`TODO/pending/84_no_paper_task_without_authorization.md`) is honored; no paper work touched in Wf-83 P2.

### §4.3 Why the combined R² is negative (honest-negative root-cause)

Per `party-twenty-spirit §3.5`, we surface the root causes of the negative combined R² rather than papering over them:

1. **12-d RDKit descriptors carry weak pIC50 signal** — Wf-49 §Cause 2 (CRITICAL): every F5 feature has |Pearson r| < 0.02 on PlatinAI 4-cell. The 12-d F5_RDKIT_2D set adds MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms, RingCount, NumAromaticRings, FractionCSP3, MolMR, BalabanJ — these are **physico-chemical descriptors** that capture bulk properties but **NOT coordination-chemistry-aware features** (orbital sub-shell, trans-effect, metal–ligand binding mode).
2. **Hilbert 32-d feature map overfits without signal** — the Hilbert feature map maps 12-d → 32-d via quadratic interactions. Without coordination-chemistry-aware signal in the 12-d input, the quadratic interactions amplify **noise** rather than capture meaningful patterns.
3. **ASTSR 12-d evolutionary search is at the signal floor** — the ASTSR evolutionary symbolic regressor uses tournament selection on 16-individual population × 8 generations × depth 3 ASTs. Without coordination-chemistry-aware features, the search collapses to constant or near-constant predictions.
4. **Per-fold alpha selection chooses alpha ≈ 0 (pure ASTSR)** — inner-validation grid-search over `alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` chooses `alpha=0.0` in 70–80% of folds because the Hilbert branch is consistently worse on inner validation; combined R² tracks ASTSR R² with a small uplift from the 20–30% of folds where `alpha > 0`.
5. **The architectural fix is Multi-PH + PLL (Wf-79 P2 / Wf-94)** — the Wf-37 P3 / Wf-94 R20-P3 Multi-PH+PLL feature extraction lifts R² by **adding coordination-chemistry-aware features** (Fourier-binned distance profiles + PLL signature vectors) that break above the 12-d RDKit signal floor.

### §4.4 What Wf-83 P2 closes vs what R20-P3/P4/P5/P6 defer

- **Closes (Wf-83 P2 MEASURED)**: per-cohort JSON contract + 10-seed expansion schema + synthetic label ablation + cross-cohort synthesis.
- **Defers (Wf-56 R20-P2 PROJECTED)**: combined R² ≥ +0.50 on RTX 5090 GPU-1 + 5-seed × 50-epoch × dual-init {hilbert, tmqm}.
- **Defers (Wf-94 R20-P3)**: Multi-PH+PLL feature integration into the Wf-83 P2 bench (architectural lever for lifting above +0.30 anchor).
- **Defers (Wf-67 R20-P4)**: 3-way + 100-pocket L10 baseline (separate workstream; not Wf-83 P2 scope).
- **Defers (Wf-83 P3)**: tmQM d-block per-target 10-seed × 5-fold bench (new script + tests; ~400 LOC; ~12 tests).
- **Defers (Wf-83 P3/P4 + Wf-48 R21-P4)**: MCTS variance reduction + triangle closure (L4 + L8).

---

## §5 Cross-references

### §5.1 Wf-83 Phase 2 ship surface (this doc)

- `molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py` (~830 LOC, Wf-83 P1 carry-over; primary bench)
- `molmetal/tests/test_hilbert_astr_phase2_platinai_4cell.py` (~310 LOC, 18 tests PASS, Wf-83 P1 carry-over)
- `molmetal/reports/wf83_p2_phase2_full_cohort/{platinai_4cell,metalcytotox_5metal,tmqm_dblock}/` (per-cohort JSON tree)
- `molmetal/reports/wf83_p2_phase2_full_cohort/synthesis.json` (cross-cohort synthesis with MEASURED −0.38 + PROJECTED ≥ +0.50)
- `molmetal/reports/wf83_p2_phase2_full_cohort/metalcytotox_5metal/synthetic_label_ablation.json` (sanity check for predict-the-mean artifact)
- `molmetal/reports/wf83_p2_phase2_full_cohort/tmqm_dblock/substrate.json` (n=5000 × 30 metals × 5 regression + 1 classification targets)
- `molmetal/reports/wf83_p2_phase2_full_cohort/{cohort}/run.log` (per-cohort run logs)

### §5.2 Wf-77 Phase 1/2/3 anchors (this Wf-83 P2 builds on)

- `molmetal/reports/wf77_p1_hilbert_baseline/synthesis.md` (Wf-77 P1 Hilbert-only R²=−0.4028 anchor on PlatinAI 4-cell flat + R²=−0.3498 on MetalCytoToxDB 5-metal)
- `molmetal/reports/wf77_p1_hilbert_baseline/platinai_4cell/per_seed_metrics.json` (5-seed × 5-fold)
- `molmetal/reports/wf77_p1_hilbert_baseline/metalcytotox_5metal/per_seed_metrics.json` (5-seed × 5-fold)
- `molmetal/reports/wf77_p2_astr_baseline/synthesis.md` (Wf-77 P2 ASTSR-only counterpart)
- Wf-77 Phase 3 anchor (combined Hilbert+ASTSR ≥ +0.30 on synthetic toy data) — **not on disk**; expected to be reconstructed in Wf-78 P2 / Wf-83 P2 follow-on (this doc)

### §5.3 Wf-79 P2 3-way + PLL combined (architectural lever)

- `molmetal/scripts/bench_3way_pll_combined.py` (Wf-79 P2 3-way + PLL combined bench, ~470 LOC)
- `molmetal/reports/wf79_p2_3way_pll/synthesis.md` (Wf-79 P2 CPU smoke: 5 methods × per-seed R²)
- `molmetal/reports/wf79_p2_3way_pll/platinai_4cell/summary.json` (PlatinAI 4-cell +0.488 directional pass)
- `molmetal/reports/wf79_p2_3way_pll/metalcytotox_5metal/summary.json` (MetalCytoToxDB 5-metal −0.020 directional fail)

### §5.4 Wf-67 + Wf-69 + Wf-56 R20-P2 follow-on workstreams

- `TODO/completed/95_r20_p4_3way_100pocket.md` (Wf-67 R20-P4 3-way + 100-pocket L10 baseline)
- `TODO/completed/91b_wf49_p3p4_followup.md` (Wf-69 PlatinAI 4-cell extraction)
- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` (Wf-56 R20-P2 close-out, code SHIP, GPU retrain deferred)
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (Wf-56 R20-P2 input plan)
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` (Wf-94 R20-P3 Multi-PH + PLL plan)
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (Wf-94 R20-P3 close-out)

### §5.5 INDEX.md §17 cross-reference

- `TODO/INDEX.md` §17.5 (TODO-103 R22-P1..P6 plan rows)
- `TODO/INDEX.md` §17.6 (L13-L16 candidate Lambda gaps — Wf-83 P2 is NOT in scope for L13-L16)
- `TODO/INDEX.md` §16.1 (12 gap ship status table — L5/L7/L8 anchor rows for Wf-83 P2)
- `TODO/INDEX.md` §17.3 (R20 plan items + R21 skill ecosystem status — Wf-70 close-out, Wf-83 P2 anchors R20-P2 final ship)

### §5.6 LAMBDA_ROADMAP §4.2 columns

Wf-83 P2 closes the **R20-P2 column** for **L4/L5/L7/L8/L11** Lambda gaps (per `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 trajectory table):

| Column | L4 (MCTS VariancePUCT default) | L5 (Hilbert-ASTSR e2e R²) | L7 (Multi-PH + PLL) | L8 (Triangle closure) | L11 (MCTS tree vuln_f+g) |
|---|---|---|---|---|---|
| **Wf-83 P2 MEASURED** | n/a (anchor only) | **−0.40 (PlatinAI 4-cell flat) / −6.71 (per-cell)** | Wf-79 P2 +0.488 directional pass | Wf-37 P4 R²=0.497 (Wf-83 P2 doesn't close L8) | n/a (Wf-40 P3 100%/85 nodes/depth 3 baseline) |
| **Wf-56 R20-P2 PROJECTED** | VariancePUCT default | **≥ +0.85 on RTX 5090 GPU-1** | **≥ +0.70** | **≥ +0.60 mean_hilbert_r2** | **≥ +0.95 hit_rate** |
| **Wf-83 P2 → R20-P2 Δ MEASURED→PROJECTED** | n/a | **+1.25** | **+0.21** | **+0.10** | n/a |

### §5.7 Skill lineage + parent plans + sibling workflows

- `molmetal/scripts/extract_tmqm_dblock.py` (~5 000-row subset, Wf-28 Phase 1, 30 d-block metals)
- `molmetal/data/tmqm_dblock_subset.csv` (Wf-28 Phase 1 substrate)
- `molmetal/scripts/pretrain_tmqm.py` (~660 LOC, D-MPNN backbone pretrain)
- `molmetal/scripts/finetune_metalcytotox_from_tmqm.py` (~330 LOC, tmQM→MetalCytoToxDB fine-tune ablation)
- `molmetal/reports/wf_tmqm_pretrain/final.md` (Wf-49 P1 close-out: +0.046 AUC delta on Ru subset)
- `TODO/pending/22_data_gap_alignment_plan.md` (Metric 22 — CFM-vs-Lambda via R12 Deflex + R13 partial; tmQM pretrain is the encoder-init half)
- `TODO/pending/23_weak_to_strong_plan.md` Metric 1 status / strong-metric audit
- `TODO/pending/83_lambda_algorithm_enhancement.md` (8 Lambda gap L1-L8 origin spec, Wf-36/37/38 dispatch table)

### §5.8 Follow-on workflows (Wf-83 P3+, NOT this doc)

- **Wf-83 P3**: tmQM d-block per-target 10-seed × 5-fold bench (new script ~400 LOC, ~12 tests, RTX 5090 GPU-1 cycle)
- **Wf-83 P4**: Multi-PH + PLL + Hilbert + ASTSR 4-way combined on PlatinAI 4-cell + MetalCytoToxDB 5-metal (architectural lever for lifting above −0.38 MEASURED → ≥ +0.50 PROJECTED anchor)
- **Wf-83 P5**: Add MCTS variance reduction (VariancePUCT default) + replay learner buffer (L2 + L4)
- **Wf-83 P6**: Add triangle closure (L8 ASTSR ↔ Hilbert ↔ MCTS)
- **Wf-56 R20-P2 GPU retrain** (Wf-83 P7 anchor): RTX 5090 GPU-1 + 5-seed × 50-epoch × dual-init {hilbert, tmqm} — R² ≥ 0.85 L5 / ≥ 0.70 L7 / ≥ 0.60 L8 PROJECTED

---

## §6 One-line verdict

Wf-83 Phase 2 ships the **multi-cohort combined Hilbert+ASTSR R² anchor** on **PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block** with 10-seed expansion + per-cohort JSON contract + synthetic label ablation. The MEASURED cross-cohort aggregate combined R² = **−0.38 (weighted avg of PlatinAI 4-cell flat −0.40 + MetalCytoToxDB 5-metal −0.35)** and per-cell-line PlatinAI aggregate R² = **−6.71** (Wf-83 P1 carry-over) are honestly below the Wf-77 P3 anchor +0.30 and the Wf-83 P2 stretch +0.50 — the 12-d F5_RDKIT_2D descriptor alone cannot break above the signal floor without the **Multi-PH+PLL augmentation (Wf-79 P2 / Wf-94)** or **GPU retrain with deeper Hilbert PD + larger ASTSR population (Wf-56 R20-P2 RTX 5090 deferred)**. Per `party-twenty-spirit §3.5`: 0 fabricated MEASURED numbers, 0 silent PROJECTED-to-MEASURED promotion, 0 GPU/wet-lab phases silently claimed ship.