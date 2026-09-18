# Pocket2Mol vs Lambda — head-to-head on 1h36

Direct comparison of two SBDD methods on PDB 1h36, 100 ligands each, Vina exhaustiveness=8.

Pocket PDB: `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb` (572 atoms, centre=[34.930572509765625, 52.59131622314453, 44.115543365478516], radius=19.7 Å).

## Head-to-head table

| method | n_ligands | mean_vina (kcal/mol) | sa_mean | qed_mean | success_rate |
|---|---|---|---|---|---|
| Pocket2Mol | 95 | -5.951 | 1.841 | 0.535 | 0.189 |
| Lambda | 98 | -5.923 | 1.870 | 0.548 | 0.194 |

## Backend notes

* **Pocket2Mol backend**: `smarts_fallback`. Pretrained weights (`pretrained_Pocket2Mol.pt`, ~165 MB) are **not** present in `molmetal/references/Pocket2Mol/ckpt/` — only the README linking the Google Drive folder is shipped. Task policy forbids pulling external pretrained weights, so the live Pocket2Mol inference path is not exercised; instead the adapter falls back to a built-in SMARTS-only pool of ~120 drug-like SMILES sampled without replacement. Numbers reported here are therefore a **baseline** — to match the published Pocket2Mol numbers, drop `pretrained_Pocket2Mol.pt` into the `ckpt/` folder and rerun (the live backend is auto-selected).

* **Lambda backend**: `Lambda_MCTSProofSearch_v1` with `n_simulations=1000`, `max_depth=3`, `target_predicates=[LIPINSKI]`, binding site = 1h36 (stub BindingSite with H-bond donor/acceptor hints). 8 standard-12 CuAAC tile roots searched in parallel.

## Sample timings

| method | sample_seconds |
|---|---|
| Pocket2Mol | 0.009 |
| Lambda | 1.568 |

## Sample SMILES (top 5 by Vina score per method)

### Pocket2Mol

* `O=C(NCc1ccccc1)c1ccccc1` — Vina -8.626 kcal/mol
* `c1ccc2c(c1)[nH]c1ccccc12` — Vina -8.359 kcal/mol
* `c1ccc(-c2ccccc2)cc1` — Vina -8.047 kcal/mol
* `O=C(O)c1ccc(C(F)(F)F)cc1` — Vina -7.931 kcal/mol
* `Cc1ccc(C(=O)NO)cc1` — Vina -7.909 kcal/mol

### Lambda

* `c1ccc(Cc2cnnn2-c2ccccc2)cc1` — Vina -8.662 kcal/mol
* `O=C(NCc1ccccc1)c1ccccc1` — Vina -8.631 kcal/mol
* `c1ccc2c(c1)oc1ccccc12` — Vina -8.295 kcal/mol
* `O=c1ccc2ccccc2o1` — Vina -7.811 kcal/mol
* `Cc1ccc(C(=O)NO)cc1` — Vina -7.775 kcal/mol

## Caveats

1. **Pocket2Mol numbers are a fallback baseline**, not the published Pocket2Mol numbers (which require the pretrained `.pt` checkpoint). To get a true head-to-head, download `pretrained_Pocket2Mol.pt` into `molmetal/references/Pocket2Mol/ckpt/` and rerun — the adapter auto-switches to the live model.
2. The 1h36 pocket here is a **synthetic bounding-box** extracted from the reference ligand pocket10 PDB. Real Pocket2Mol numbers on CrossDocked2020 are computed on the 100-pockets test set; our 1h36-only number is a single-pocket comparison point, not directly comparable to the paper's CrossDocked mean.
3. The Lambda search runs `MCTSProofSearch` with the same `LIPINSKI` predicate the task specifies, but the binding test uses the **stub** `BindingSite.typecheck` (not real Vina) so candidates are filtered on ADMET + geometric hints only. Real binding is then measured by Vina on the candidate pool.
4. **success_rate** = fraction of docked ligands with Vina ≤ −7.0 kcal/mol (CrossDocked2020 success threshold).
