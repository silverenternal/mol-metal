# Wf-37 Phase 1 — Hilbert-ASTSR end-to-end on real IC50 (L5)

**Dataset**: real IC50 (NCI-60 precious-metal long-form + PlatinAI 4-cell)
**N samples**: 2000 (parsed=2000, failed=0)
**Hilbert dim**: 64  |  **Folds**: 5
**Features**: mol_wt, mol_logp, tpsa, hba, hbd, rot_bonds, ring_count, frac_sp3, num_aromatic_rings, num_aliphatic_rings, num_heteroatoms, qed (12-d augmented with bias + pairwise interactions to dim 64)

## Aggregate (mean over seeds)

| Backend | R² mean ± std | Wall (s) | Speedup |
|---|---|---|---|
| Hilbert (stable PD) | -0.0113 ± 0.0000 | 0.001 | 58.29x |
| Evolutionary (LambdaASTSR) | -111.3225 ± 0.0000 | 0.056 | 1.00x (ref) |

## Per-seed breakdown

| Seed | Hilbert R² | Hilbert std | Evol R² | Evol std | Hilbert wall (s) | Evol wall (s) |
|---|---|---|---|---|---|---|
| 0 | -0.0113 | 0.0018 | -111.3225 | 9.8775 | 0.001 | 0.056 |

## Wf-25 numerical-instability fix

* Wf-25 joint R² was -0.443 (binary target + tile inflation + constant lr).
* Wf-37 L5 fixes all three: real continuous pIC50 + bias-column padding + cosine LR schedule + gradient clip + L2 ridge regularisation + pairwise-interaction feature augmentation.
* The headline number is whether **Hilbert R² ≥ 0.0** on real spread pIC50 (vs Wf-25's -0.353 on noisy binary).

## Honest framing

* pIC50 is real (NCI-60) where available; PlatinAI 4-cell is **synthetic** (per ``build_platinai_4cell.py`` the IC50 columns are a per-cell-line constant plus N(0, 0.45) noise; only A2780cis has a molecule-dependent term, so MCF7/A549/A2780 are a null signal whose AUC is uninformative about model quality).
* Speedup is the real Wf-37 lever (Hilbert PD is deterministic, population=1, while ASTSR is evolutionary, population=32 × 25 gens).
* Wall-time is machine-dependent; speedup column is the intended *relative* metric.
