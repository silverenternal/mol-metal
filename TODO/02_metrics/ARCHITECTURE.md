# Architecture — TODO/02_metrics/

**Last updated:** 2026-10-02 (per Wf-438 simplification + Wf-432 protocol validation closure + Wf-424 paper §3 v4 FINAL)

Per-topic documentation for the metrics + mechanism mapping surface. Replaces ad-hoc READMEs scattered in `metrics/by_metric/` + `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` + `METRICS_TARGETS.md`.

## 21 Mechanisms (M1-M21) organized into 4 phases

### Phase 0: Foundation (m1-m3 — main-result candidates)
- **M1** [`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`](../../molmetal/scripts/train_cfm_yuelbond_wasserstein.py) — Flow Matching backbone (Lipman 2023 + YuelBond decoder + EGNN). Wire: `python main.py --metal Pt --cuiddcer scaffold_split`. ~2450 LOC.
- **M2** [`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`](../../molmetal/scripts/train_dmpnn_multitask_wasserstein.py) — Multi-task D-MPNN Wasserstein per-metal heads (Ru/Ir/Rh/Os/Re). Wire: `python molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metals Ru Ir Rh Os Re --epochs 50`. ~1470 LOC.
- **M3** [`molmetal/scripts/train_metal_lomo_wasserstein.py`](../../molmetal/scripts/train_metal_lomo_wasserstein.py) — Metal LOMO (Leave-One-Metal-Out) cross-validation. Wire: `python molmetal/scripts/train_metal_lomo_wasserstein.py --epochs 50 --seeds 0 1 2 3 4`. ~510 LOC.

### Phase 1: Augmented baselines (m4-m11 — recency + protocol-validation-grade)
- **M4** [`molmetal/scripts/cfm_property_panel.py`](../../molmetal/scripts/cfm_property_panel.py) — SA / QED / property panel per-epoch CFM evaluation. Wire: `python molmetal/scripts/cfm_property_panel.py --cohort metalcytotox_combined`.
- **M5** [`molmetal/adapters/flow_matching_lipman/`](../../molmetal/adapters/flow_matching_lipman/) — Multi-PH + PLL + Lambda + Hilbert + ASTSR + YuelBond + Confidence adapters. ~1100 LOC across 8 files.
- **M6** [`molmetal/molmetal_lam/sbdd_env/metal_coord_term.py`](../../molmetal/molmetal_lam/sbdd_env/metal_coord_term.py) — Metal-coordination term Vina scoring. Wire: per-epoch `kcal_sign_audit_pass=True`.
- **M7** [`molmetal/scripts/metallo_drug_smoke_retrain.py`](../../molmetal/scripts/metallo_drug_smoke_retrain.py) — Diversity sampling primitives (Holtzman 2020 nucleus + Gretton 2012 MMD + Liu 2021 repulsion).
- **M8** [`molmetal/scripts/train_pocket_invariance.py`](../../molmetal/scripts/train_pocket_invariance.py) — Pocket-invariance structural break (chemical features → molecular shape → composition pocket-conditioned per-cell-line).
- **M9** [`molmetal/adapters/click_chemistry/`](../../molmetal/adapters/click_chemistry/) — Click chemistry 5-reaction library (CuAAC / SPAAC / ThiolEne / Suzuki / AmideCoupling).
- **M10** [`molmetal/adapters/score_repellent_mc.py`](../../molmetal/adapters/score_repellent_mc.py) — Score-Repellent MC variance reduction + PH approximation guardrail.
- **M11** [`molmetal/adapters/flow_matching_lipman/multiph_pll.py`](../../molmetal/adapters/flow_matching_lipman/multiph_pll.py) — Multi-parameter persistent homology feature extraction.

### Phase 2: Research extensions (m12-m22 — forward-looking + Tier-2 reorientation)
- **M12** [`molmetal/math_skills/catherine_properties.py`](../../molmetal/math_skills/catherine_properties.py) — Catherine closure properties (soundness / completeness / local-confluence / etc.) on 30 d-block metals.
- **M13** [`molmetal/molmetal_lam/lam_chem/closure.py`](../../molmetal/molmetal_lam/lam_chem/closure.py) — Molecular Lambda calculus × β-NF closure operator.
- **M14** [`molmetal/molmetal/molmetal_lam/sbdd_env/yuelbond_decoder.py`](../../molmetal/molmetal_lam/sbdd_env/yuelbond_decoder.py) — YuelBond bond-decoder head (chem-aware).
- **M15** [`molmetal/molmetal_lam/sbdd_env/posebusters_validator.py`](../../molmetal/molmetal_lam/sbdd_env/posebusters_validator.py) — PoseBusters production pass rate (22/26 PB checks).
- **M16** [`molmetal/scripts/train_metal_cytotox.py`](../../molmetal/scripts/train_metal_cytotox.py) — MetalCytoToxDB regression + 5-fold CV + A2780/MCF7/NCI-60 cell-line breakdown.
- **M17** [`molmetal/scripts/train_drugood.py`](../../molmetal/scripts/train_drugood.py) — DrugOOD OOD generalisation + Tanimoto<0.4 OOD protocol.
- **M18** [`molmetal/molmetal_lam/reward/platinai_dataset.py`](../../molmetal/molmetal_lam/reward/platinai_dataset.py) — PlatinAI pIC50 regression + scaffold split + 4 cell-line cohort.
- **M19** [`molmetal/scripts/train_platinai_4cell.py`](../../molmetal/scripts/train_platinai_4cell.py) — PlatinAI 4-cell pIC50 LOMO cross-validation.
- **M20** [`molmetal/scripts/bench_lambda_triangle_end_to_end.py`](../../molmetal/scripts/bench_lambda_triangle_end_to_end.py) — Triangle closure (ASTSR ↔ Hilbert ↔ MCTS).
- **M21** [`molmetal/scripts/train_se3sb.py`](../../molmetal/scripts/train_se3sb.py) — SE(3) Schrödinger Bridge trainer.

## Phase diagram (how mechanisms compose)

```
Phase 0 (m1-m3) ──┬── m1 CFM (YuelBond) ──┬── m4 property panel
                   ├── m2 D-MPNN (5-metal heads) ──┼── m15 PB
                   └── m3 LOMO (cross-val) ──┴── m16/m17/m18/m19 baselines
                                              │
Phase 1 (m4-m11) ──┬── m5 adapters (PH/PLL/AGS/Hilbert) ──┬── m6 metal coord
                   ├── m7 diversity (MMD/MSD) ──┼── m20 triangle
                   ├── m8 pocket invariance ──┼── m19 LOMO+
                   ├── m10-21 multi-click ──┴── m14 YuelBond
                   │
Phase 2 (m12-m22) ──┬── m12 Catherine + m13 closure ──┬── Tier-2 §3 main-result
                     └── m15 SE(3+SB) ──┴── Tier-2 §6 limitations
```

## Files in this directory

| File | Status | LOC |
|---|---|---|
| `METRICS_TARGETS.md` | SHIPPED (canonical 22-metric Tier-1/2 target table) | ~43k |
| `EVALUATORS_MIGRATED.md` | SHIPPED (22/22 evaluators MEASURED-OK) | ~19k |
| `MECHANISM_METRIC_MAPPING.md` | SHIPPED (mechanism ↔ metric contribution matrix) | ~67k |
| `ARCHITECTURE.md` | SHIPPED (this file — per-topic architecture overview) | (new) |

## Cross-links

- Top-level [../../REPRODUCE.md](../../REPRODUCE.md) — 1-command reproducibility
- [`../03_baselines/ARCHITECTURE.md`](../03_baselines/ARCHITECTURE.md) — 3 Tier-1 baselines + protocol alignment
- [`../03_plans/INDEX.md`](../03_plans/INDEX.md) — R17-R20 strategic plans
- `metrics/by_metric/*.json` — single source of truth for every measured value
