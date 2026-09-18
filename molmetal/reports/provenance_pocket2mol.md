# Pocket2Mol Provenance & Protocol Audit**  _Peng et al., ICML 2022 · arXiv:2205.07249v2 · https://doi.org/10.48550/arXiv.2205.07249_

## 1. Dataset & Split
- **Name**: CrossDocked2020 (Francoeur et al., 2020), 22.5 M protein–ligand structures.
- **Split file**: `data/crossdocked_pocket10/` with `data/split_by_name.pt` (Luo et al., 2021 / AR convention). Config `configs/sample.yml` points to `dataset.name: pl` and `path: ./data/crossdocked_pocket10`.
- **n_test (verified)**: `data/test_list.tsv` = **100 pocket–ligand pairs** (header `Protein\tLigand` + 100 data rows; `wc -l` = 101 incl. header; README: "i should be between 0 and 99").
- **Pocket def (test set)**: pre-extracted 10 Å sub-pocket around the co-crystal ligand (filename suffix `_pocket10.pdb`).

## 2. Pocket Radius / Center Extraction
- **Default bounding box**: 23 Å cube (code: `sample_for_pdb.py` → `--bbox_size 23.0`; README §"Sampling for PDB pockets": "side length of the bounding box (default: 23Å)").
- **Atom filtering**: drop H; reject atoms with `max(|pos − center|) > bbox_size/2` (axis-aligned cube clip).
- **Center**: user-supplied `--center x,y,z`; for the test set the reference ligand's bounding-box midpoint `(pos.max + pos.min)/2` is used by `QVinaDockingTask` (no protein-residue based pocket centroid).

## 3. Sampling / NFE
From `configs/sample.yml` and `sample.py`:
- `num_samples = 100` (per pocket)
- `beam_size = 500`
- `max_steps = 50`
- `mask_init = True`, focal 0.5 / pos 0.25 / element 0.3 / hasatom 0.6 / bond 0.4
- `seed = 2020`, single-CPU `taskset -c 0` recommended.
- NFE is *not* a fixed count — autoregressive mask-fill, ≤ 50 atom-addition steps × masked-position prediction; no MCMC.

## 4. Vina / Docking Protocol (`evaluation/docking.py`)
- **Engine**: QuickVina 2 (`qvina2`) inside `conda env adt`.
- **Box**: `--size_x 20 --size_y 20 --size_z 20` (Å).
- **Default exhaustiveness = 16** (`QVinaDockingTask.run(exhaustiveness=16)`); `n_poses` defaults to qvina2 top-9, parses `REMARK` lines for affinity + RMSD.
- **Pre-dock prep**: UFF-optimize generated ligand 3D (`UFFOptimizeMolecule`), then `obabel` to PDBQT, `prepare_receptor4.py` for receptor.

## 5. SA-score & Ranges
- **SA**: `utils/sascorer.py` is the Ertl & Schuffenhauer 2009 RDKit SA score, normalized to **[1 (easy) – 10 (hard)]** then mapped to **[0, 1] inverted** so ↑ = easier. Paper reports Pocket2Mol SA = 0.765 ± 0.13.
- **QED** (Bickerton 2012) range [0, 1] ↑.
- **Lipinski** = count of Ro5 passes (0–5) ↑.
- **Diversity** = 1 − mean pairwise Tanimoto on ECFP4 of pocket's 100 mols.
- **Sim.Train** = 1 − max Tanimoto vs. training-set ligands (↓).

## 6. Success-Rate Definition
"**High Affinity**" = fraction of pockets for which Pocket2Mol's best-of-100 docked score is **≤ (better than) the co-crystal ligand's docked Vina score**. Paper Table 1: Pocket2Mol = **0.542 ± 0.32** (vs. AR 0.267, CVAE 0.238).

## 7. Paper Table 1 (Test / CVAE / AR / Pocket2Mol, n=100 × 100 mols)
| Metric | TestSet | CVAE | AR | **Pocket2Mol** |
|---|---|---|---|---|
| Vina Score (kcal/mol ↓) | −7.158±2.10 | −6.144±1.57 | −6.215±1.54 | **−7.288±2.53** |
| High Affinity (↑) | — | 0.238±0.28 | 0.267±0.31 | **0.542±0.32** |
| QED (↑) | 0.484±0.21 | 0.369±0.22 | 0.502±0.17 | **0.563±0.16** |
| SA (↑) | 0.732±0.14 | 0.590±0.15 | 0.675±0.14 | **0.765±0.13** |
| LogP | 0.947±2.65 | −0.140±2.73 | 0.257±2.01 | **1.586±1.82** |
| Lipinski (↑) | 4.367±1.14 | 4.027±1.38 | 4.787±0.50 | **4.902±0.42** |
| Sim.Train (↓) | — | 0.460±0.18 | 0.409±0.19 | **0.376±0.22** |
| Diversity (↑) | — | 0.654±0.12 | 0.742±0.09 | **0.688±0.14** |
| Time (s, ↓ per pocket) | — | — | 19 658.56±14 704 | **2 503.51±2 207** (single V100) |

Time @ ~25 s/mol × 100 mols × 100 pockets ≈ 69.5 GPU-h / **~7 CPU-h equivalent for our 1h36 pocket run × ~30× speed-up**.

## 8. Protocol Deltas vs. Our Lambda 1h36-only Single-Pocket Run
- **n_test**: paper averages 100 pockets; we report **n = 1** (single Lambda pocket) — variance/CI not estimable.
- **Pocket source**: paper uses CrossDocked `_pocket10.pdb` (10 Å around co-crystal ligand); Lambda pocket comes from our prepared `.pdb` + manual `--center`, so ligand-conditioned residue context differs.
- **100 mols/pocket**: paper; we likely sub-sample → low effective N for Top-1/Top-3 stats.
- **Exhaustiveness 16 / box 20 Å** is the same default — keep it for direct comparability of Vina kcal/mol, QED, SA, Diversity; flag the n=1 caveat in any cross-paper claim.

## TL;DR
Pocket2Mol (ICML 2022, arXiv 2205.07249): E(3)-equivariant auto-regressive mask-fill; 23 Å box; 100 mols/pocket × 100 CrossDocked pockets; QVina2 (exh=16, box 20 Å); SA via Ertl, QED via Bickerton; success = best-of-100 ≤ co-crystal Vina. Table-1 headline: Vina −7.29, QED 0.563, SA 0.765, High-Affinity 54.2 %, Diversity 0.688, 2 504 s/pocket (V100). Our Lambda run shares Vina/SA/QED protocol but n_test = 1 and a non-CrossDocked pocket — direct success-rate comparison invalid.

## 5-Line Core Numbers (Pocket2Mol, n=100 pockets × 100 mols)
1. Vina Score −7.288 ± 2.53 kcal/mol
2. QED 0.563 ± 0.16
3. SA 0.765 ± 0.13
4. High Affinity (success rate) 54.2 ± 32 %
(Diversity 0.688 ± 0.14; Time 2 503.51 ± 2 207 s / pocket on a single V100)

**Citation**: Peng X., Luo S., Guan J., Xie Q., Peng J., Ma J. *Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets.* ICML 2022. arXiv:2205.07249.