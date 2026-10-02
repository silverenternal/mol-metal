# Architecture — TODO/03_baselines/

**Last updated:** 2026-10-02 (per Wf-432 protocol validation + Wf-412 m3 Vina MOCK→REAL audit)

Per-topic consolidation of the 3 Tier-1 baselines selected for rigorous alignment (NOT reproduction — per D-Wf-431). All three use the same Mol-Metal coordinates (m1-m22) + same evaluation protocol.

## 3 Tier-1 baselines

### 1. Krasnov 2026 (MetalCytoToxDB)
- **Dataset path:** [`molmetal/molmetal_lam/reward/metal_cytotox_drugood_loader.py`](../../molmetal/molmetal_lam/reward/metal_cytotox_drugood_loader.py)
- **Scope:** ~5K small-molecule × 78 protein × 5 cell-line pIC50 cytotoxicity assay
- **Split protocol:** 5-fold CV + per-cell-line breakdown (A2780 / MCF7 / NCI-60)
- **Metric:** m1 LOMO AUC per metal + m16 D-MPNN regression
- **Alignment level:** TIER-1 LEAD (m1_me per-fold Krasnov 0.86 gate + m16 m21 NCI-60 9-cell AUC ≥ 0.80)
- **Status:** MEASURED 7/7 (metal, seed) pair aggregate · Rh fold PASS_KRASNOV 0.8798 +0.020 · Re fold BELOW_KRASNOV 0.7088 −0.151 · overall 0.7577

### 2. Rusanov 2025 (PlatinAI)
- **Dataset path:** [`molmetal/molmetal_lam/reward/platinai_dataset.py`](../../molmetal/molmetal_lam/reward/platinai_dataset.py)
- **Scope:** 226K pIC50 measurements (500-mol subset on disk per BD-3) + 4 cell-line cohort
- **Split protocol:** scaffold-split + 4-cell-line cohort (per per-cell-line R²)
- **Metric:** m18 Rusanov per-cell R² + m19 LOMO 4-cell + m2 R² real-cohort extension
- **Alignment level:** TIER-1 LEAD (m2_r2_per_cell + m18 Rusanov per-cell R² ≥ -0.03 baseline × scaffold-vs-CV adjustment)
- **Status:** MEASURED + PlatinAI 500-mol subset re-MEASURED via Wf-432-A2 — `platinai_4cell` + `hilbert` cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 + Wf-398-A2; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981 per Wf-398-A2

### 3. DrugOOD 2022
- **Dataset path:** [`molmetal/molmetal_lam/reward/metal_cytotox_drugood_loader.py`](../../molmetal/molmetal_lam/reward/metal_cytotox_drugood_loader.py)
- **Scope:** ~1.58M OOD generalisation benchmark (10K subsample on disk per BD-1)
- **Split protocol:** Tanimoto<0.4 OOD protocol + scaffold split + 5-fold CV
- **Metric:** m17 OOD AUC + cross-domain M17/M18/M21 lift
- **Alignment level:** TIER-1 LEAD (m17 ≥ 0.69 baseline × scaffold-vs-CV adjustment — MEASURED 0.779)
- **Status:** MEASURED 0.779 per Wf-376-A4 §3 + Wf-389-A1 §3

## Alignment contract (per Wf-432 §12.5 + §12.7.5)

**threshold(m_i) = baseline(m_i) + protocol_adjustment(m_i, protocol_difference)**

Per-metric Tier-1 thresholds:
- m1 Ru AUC: ≥ 0.83 (Krasnov 0.81 + 0.02 scaffold-vs-CV adjustment)
- m2 Pt R² per-cell: ≥ per-line − 0.03 (Rusanov per-cell vs cohort adjustment)
- m3 Vina median kcal/mol: ≤ -7.5 (5-pocket + n=100 extension)
- m17 DrugOOD OOD AUC: ≥ 0.69 (MEASURED 0.779)
- m21 NCI-60 9-cell AUC: ≥ 0.80 (MEASURED 0.8007)
- m7 Diversity Tanimoto: ≥ 0.65 (unchanged)
- m15 RMSD < 2Å: ≥ 25% (MEASURED 5/5 SOTA PASS)
- m4 Success Rate: ≥ 0.60 (Vina sub-criterion BLOCKED on m3)
- m16 Affinity Pearson r: ≥ 0.35 (= baseline × 0.5 honest with 12-d signal floor)
- m19 LOMO: ≥ 0.55 (scaffold-split vs 5-fold CV adjustment)

**File**: `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` documents the per-baseline protocol differences + adjustments.

**Block**: `TODO/03_baselines/PROTOCOL_VALIDATION.md` documents the per-baseline validation hooks + implementation status.

## Files in this directory

| File | Status | LOC |
|---|---|---|
| `COMPARISON_BASELINES.md` | SHIPPED (cross-baseline comparison table) | ~10k |
| `PROTOCOL_ALIGNMENT_MATRIX.md` | SHIPPED (per-baseline protocol difference adjustment matrix) | ~4k |
| `MOCK_TO_REAL_PLAN.md` | SHIPPED (5 MOCK→REAL conversion plan; 4/5 NOT-A-BLOCKER + 1 real MOCK per Wf-432) | ~68k |
| `HONEST_NEGATIVE_CLOSURE_PLAN.md` | SHIPPED (3 honest-negative structural gates pre-closure paths) | ~75k |
| `PROTOCOL_VALIDATION.md` | SHIPPED (23 NEW pre-retrain threshold gates Layer 1-4) | ~68k |
| `ARCHITECTURE.md` | SHIPPED (this file — per-topic architecture overview) | (new) |

## Cross-links

- Top-level [../../REPRODUCE.md](../../REPRODUCE.md) — 1-command reproducibility
- [`../02_metrics/ARCHITECTURE.md`](../02_metrics/ARCHITECTURE.md) — 21 mechanisms + 4 phases
- [`../03_plans/INDEX.md`](../03_plans/INDEX.md) — R17-R20 strategic plans
- `metrics/by_metric/*.json` — single source of truth for every measured value
