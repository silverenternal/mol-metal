# Round-10 axis C micro-ablation — 1h36 pocket (CFG)

## Goal

Quantify the Vina-score effect of classifier-free guidance (CFG)
on the EGNN velocity field used by the Lipman flow-matching
adapter.  CFG is the canonical trick that pushes conditional
generation toward the conditioning signal; here the conditioning
is the CrossDocked 1h36 pocket (a real protein binding site).

## Setup

- Pocket: **1h36** (572 atoms, radius=21.1A)
- PDB: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb`
- N molecules requested per cfg: **20**
- CFG scales swept: **[1.0, 1.5, 2.0, 3.0]**
- CFM train steps per cfg: **50** (hidden_dim=32, n_layers=2, lr=0.0002, context_dropout=0.1)
- ODE integration steps: **8**
- Vina exhaustiveness: **2** (n_poses=1)
- Coordinate safety clamp: **[-3.0, +3.0] Å** (per-velocity-step clamp prevents CFG>1 ODE divergence)
- Pocket-embed L2 norm target: **1.0** (1h36's 572-atom pocket produces raw embed norm ~178, normalised to 1.0)
- Total wall: **274.23s** (4.57 min)

## CFG implementation

- `EGNNVelocityField.context_dropout` (default 0.1) randomly zeros the `pocket_embed` per training batch so the model learns both `p(v | c)` and `p(v | ∅)`.
- `EGNNVelocityField.v_cfg(...)` calls `forward` twice and returns `v_uncond + cfg_scale · (v_cond − v_uncond)`.  `cfg_scale=1.0` short-circuits to the conditional forward (bit-exact legacy).
- `LipmanFlowMatchingAdapter(..., cfg_scale=...)` plumbs the guidance scale into `generate()`; the same scale is applied to the post-ODE atom-type logits for consistency.

## Vina score distribution per cfg

| cfg_scale | n_docked | mean | median | min | max | delta_vs_cfg=1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 7 | -2.046 | -2.064 | -2.883 | -1.386 | +0.000 |
| 1.50 | 6 | -2.254 | -1.988 | -3.504 | -1.686 | -0.208 |
| 2.00 | 4 | -2.151 | -2.120 | -3.615 | -0.750 | -0.105 |
| 3.00 | 4 | -1.682 | -1.916 | -2.147 | -0.750 | +0.364 |

## Headline

- Best cfg_scale: **1.50** (Vina mean **-2.254** kcal/mol over 6 docked mols)
- Delta vs cfg=1.0: **-0.208** kcal/mol  (negative = CFG improves binding)
- Success criterion (Δ ≤ -0.3 kcal/mol): **FAIL**

## Notes / caveats

- The CFM training data in this micro-bench is synthetic random points (the spec forbids running real training data; this is an algorithmic micro-bench).  With only 50 train steps the model is barely above noise — Vina scores cluster near -2 kcal/mol regardless of cfg_scale, and the per-cfg ranking is dominated by sampling noise rather than a real CFG-induced binding-affinity shift.
- Docking rates vary because (a) the EGNN samples exotic metals (Pt/Ir/Zn) at random which Vina can't parameterise and (b) random-noise geometries often produce multi-fragment SMILES that meeko rejects.  Only single-fragment, C/N/O/S/P/F/Cl/Br/I molecules are docked.
- Two of the four cfg_scales sometimes produce 0/20 dockings in a given run — this is consistent with the high variance of a 50-step-trained model on a single pocket and highlights the importance of training-to-convergence (and a larger N) for production CFG ablations.  The micro-bench is intended to validate the CFG wiring end-to-end, not to produce a converged Vina ranking.
- In an earlier dry-run with seed=0, cfg=2.0 produced Vina mean **-2.614** kcal/mol vs cfg=1.0 mean **-2.102** (Δ = **-0.512**, success criterion PASS).  That result is not reproduced at the current seed — the ranking is below the noise floor of this micro-bench.
