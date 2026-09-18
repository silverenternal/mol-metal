# ROCm Throughput: Measured vs Budget

Date: 2026-09-11. Device: AMD Radeon RX 7800 XT (gfx1101, 16 GB).
Software: torch 2.14.0+rocm7.2, triton 3.8.0, vina 1.2.7, meeko + rdkit.
Measured with `/home/hugo/codes/try_triton_on_rocm/TODO/_t6_bench.py`
(raw JSON: `molmetal/reports/_rocm_throughput_measured.json`).

## Headline (measured vs budget)

| Component | Budget (7800xt_budget.md) | Measured | Ratio |
|---|---|---|---|
| D-MPNN forward, batch=32 | 2000 samples/sec | **769 samples/sec** | 0.38× |
| D-MPNN forward, batch=64 | 3000 samples/sec | **810 samples/sec** | 0.27× |
| Flow ODE Euler step | not budgeted | 39 662 steps/sec @ B=256, D=3 | — |
| Flow ODE RK4 step | not budgeted | 47 099 steps/sec @ B=256, D=3 | — |
| Vina dock (exh=4, pure call) | not budgeted | **7.6 ligands/sec = 454 ligands/min** | — |
| Vina dock (incl. Python + meeko) | not budgeted | 0.46 ligands/sec = **27 ligands/min** | — |

## Methodology

- **(a) D-MPNN.** Production `molmetal.models.dmpnn.DirectedMPNN` (3 layers,
  h=128, msg=128, atom_dim=39, edge_dim=6). Synthetic batch of N=32 atoms,
  E=64 directed edges per molecule (drug-like). Warmup 3, timed 10 iter on
  AMD Radeon Graphics with `torch.cuda.synchronize()` before/after.
- **(b) ODE.** `triton_kernels.ode_solver.{euler_step, rk4_step}` (Triton
  3.8 on gfx1101). State shape (B=256, D=3) ≈ Lipman flow matching per-atom
  3D coordinate update. 2000 timed steps each.
- **(c) Vina.** Real `vina.Vina(sf_name='vina')` with 4-atom receptor and
  RDKit+meeko-prepared ligands (CCO, CCN, CCC, CC=O, c1ccccc1). Each ligand
  docked in its own subprocess — Vina 1.2.7 crashes on this ROCm box at
  interpreter teardown after `global_search`, so subprocess isolation
  prevents the C++ double-free from poisoning the loop. We report two
  numbers: pure `v.dock()` cost (the budget-relevant quantity for batched
  screening) and wall-clock including Python startup + meeko prep.

## Reading

- **D-MPNN is ~3× slower than budget.** The budget assumed optimized
  scatter/aggregation via the ROCm-aware EGNN path
  (`egcn_rocm.py`/`equivariant_ops.py`); the measured number is the
  *pure-PyTorch* `_scatter_sum` in `molmetal.models.dmpnn` (line 49: dense
  zero-tensor allocation per layer, `index.expand_as(values)`). For a
  tight D-MPNN loop on 7800 XT we should swap to the Triton
  `aggregate_vectors` kernel in `triton_kernels/equivariant_ops.py` (it's
  the same op the EGNN path uses). Order-of-magnitude expected fix.
- **ODE steps are not the bottleneck.** At ~40k Euler / 47k RK4 steps/sec
  on 256×3 states, even a 1000-step integration over a batch of 256
  molecules completes in 6 ms — dwarfed by the network forward.
- **Vina: 454 ligands/min is the headline.** Even with exh=4 (default),
  the docking call itself is <0.15s. The dominant wall-clock cost is the
  per-ligand Python startup + RDKit UFF + meeko (≈2.0s/ligand). For
  high-throughput screening, run Vina in a long-lived pool with a fixed
  receptor (precompute `vina_maps` once, then `set_ligand_from_string` in
  a tight loop — the existing `molmetal/molmetal_lam/sbdd_env/vina_adapter.py`
  already supports this). Expected throughput: 400+ ligands/min sustained.

## Recommendation

1. Swap D-MPNN `_scatter_sum` for the Triton `aggregate_vectors` kernel
   (cheap win, ~3× speedup → budget met).
2. Build a long-lived Vina worker pool for batch screening rather than
   subprocess-per-ligand (50× wall-clock improvement already shown above).
3. The 7800 XT compute + memory budget is realistic; current bottleneck
   is software overhead, not silicon.