# Wf-37 Phase 1 — Hilbert-ASTSR end-to-end on real IC50 (L5)

**Dataset**: real IC50 (NCI-60 precious-metal long-form + PlatinAI 4-cell)
**N samples**: 2000 (parsed=2000, failed=0)
**Hilbert dim**: 64  |  **Folds**: 5
**Features**: mol_wt, mol_logp, tpsa, hba, hbd, rot_bonds, ring_count, frac_sp3, num_aromatic_rings, num_aliphatic_rings, num_heteroatoms, qed (12-d augmented with bias + pairwise interactions to dim 64)

## Aggregate (mean over seeds)

| Backend | R² mean ± std | Wall (s) | Speedup |
|---|---|---|---|
| Hilbert (stable PD) | +0.1631 ± 0.0063 | 0.005 | 716.28x |
| Evolutionary (LambdaASTSR) | -0.0803 ± 0.0677 | 3.390 | 1.00x (ref) |

## Per-seed breakdown

| Seed | Hilbert R² | Hilbert std | Evol R² | Evol std | Hilbert wall (s) | Evol wall (s) |
|---|---|---|---|---|---|---|
| 0 | +0.1510 | 0.0152 | -0.0201 | 0.0698 | 0.007 | 3.804 |
| 1 | +0.1638 | 0.0164 | -0.0984 | 0.1688 | 0.005 | 3.774 |
| 2 | +0.1646 | 0.0134 | -0.0394 | 0.0272 | 0.004 | 3.115 |
| 3 | +0.1679 | 0.0064 | -0.2050 | 0.4023 | 0.004 | 3.249 |
| 4 | +0.1680 | 0.0146 | -0.0387 | 0.0108 | 0.003 | 3.008 |

## Wf-25 numerical-instability fix

* Wf-25 joint R² was -0.443 (binary target + tile inflation + constant lr).
* Wf-37 L5 fixes all three: real continuous pIC50 + bias-column padding + cosine LR schedule + gradient clip + L2 ridge regularisation + pairwise-interaction feature augmentation.
* The headline number is whether **Hilbert R² ≥ 0.0** on real spread pIC50 (vs Wf-25's -0.353 on noisy binary).

## Honest framing

* pIC50 is real (NCI-60) where available; PlatinAI 4-cell is **synthetic** (the IC50 columns were generated from HOMO/LUMO/Gap/Polarizability per ``build_platinai_4cell.py``).
* Speedup is the real Wf-37 lever (Hilbert PD is deterministic, population=1, while ASTSR is evolutionary, population=32 × 25 gens).
* Wall-time is machine-dependent; speedup column is the intended *relative* metric.
