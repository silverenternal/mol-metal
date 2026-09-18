# Provenance — TargetDiff (Guan et al., ICLR 2023)

## Sources
- Cloned code: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/` (commit `142f1eb`, Jul 13 2023).
- arXiv preprint: **arXiv:2303.03543** (v1, 6 Mar 2023; q-bio.BM).
- OpenReview: **kJqXEPXMsE0** (ICLR 2023, "3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction").

## Dataset + split (CrossDocked2020 / pocket10)
- CrossDocked2020 v1.1, filtered to poses with RMSD < 1.0 Å, 30% sequence-identity split (`split_by_name.pt`, shared with AR / Pocket2Mol).
- Pocket = 10 Å region around the reference ligand (`scripts/data_preparation/extract_pockets.py`).
- Training set ≈ 100 K complexes; **test set = 100 held-out pockets** (README: `i should be between 0 and 99`; `notebooks/summary.ipynb` aggregates over 100).

## Sampling config (`configs/sampling.yml`)
- `num_samples: 100`, **`num_steps: 1000`** (DDPM, full chain), `seed: 2021`, `center_pos_mode: protein`, `sample_num_atoms: prior`.
- NFE budget per pocket = 1000 steps × 100 samples = **100 000 NFE**. Across 100 pockets → **10 M NFE total** for the full paper benchmark.

## Vina docking (`scripts/dock_testset.py` defaults)
- Backend: `vina==1.2.2` (Python bindings), `meeko==0.1.dev3`, AutoDockTools_py3 for receptor prep (`README` install section).
- Search box: built via `QVinaDockingTask.from_original_data` (exhaustiveness follows QVina default = **8**; `size_factor=1.2`, `use_uff=True` for UFF torsion pruning).
- Modes: `qvina | vina_score | vina_dock | none` (`evaluate_diffusion.py`).

## Paper Table 1/2/3 numbers (TargetDiff, 100 pockets)
| metric | value | source |
|---|---|---|
| Vina Dock (Avg / Med, ↓ kcal/mol) | **−7.80 / −7.91** | Table 1 |
| Vina Score (Avg / Med) | −5.47 / −6.30 | Table 1 |
| Vina Min (Avg / Med) | −6.64 / −6.83 | Table 1 |
| High Affinity (Vina Dock < ref, ↑) | **58.1% / 59.1%** | Table 1 |
| **QED** (↑) | 0.48 / 0.48 | Table 1 |
| **SA** (↑) | **0.58 / 0.58** | Table 1 |
| Diversity (Tanimoto, ↑) | 0.72 / 0.71 | Table 1 |
| Success Rate (QED>0.25 ∧ SA>0.59 ∧ Vina<−8.18) | **10.5%** (other sources cite 35.1% for relaxed thresholds; paper reports 10.5%) | Table 1 / TransDiffSBDD Table 1 |
| Best-binding pockets (median Vina) | **57%** of 100 | Fig. 4 |
| Affinity > reference (per molecule) | **58.1%** | §5.2 |
| CoM shift vs reference | **1.45 Å** (AR 1.79 Å) | §5.2 |

NB: "Success rate" definition differs between reproductions (QED>0.25 ∧ SA>0.59 ∧ Vina<−8.18 → 10.5%; with looser drug-like thresholds → up to 35.1%). Use the strict definition unless otherwise stated.

## Protocol differences vs Lambda `1h36-only`
- **Pocket corpus**: TargetDiff averages over **100 held-out CrossDocked2020 pockets** with a pre-registered split; Lambda's metalloprotein branch evaluates a **single pocket (`1h36` chain A, HisP1H/HEM)**. TargetDiff's numbers are population means — Lambda's are a single-case value (no error bar).
- **Pocket def**: TargetDiff clips 10 Å around the bound ligand via `extract_pockets.py`; Lambda uses the crystallographic `1h36` biological assembly + `extract_pocket.py` with metal-aware radius.
- **Docking**: TargetDiff uses **QVina (exh=8)**, `--use_uff --size_factor 1.2`; Lambda uses **AutoDock Vina 1.2.x + Meeko + ADFRsuite** with `--exhaustiveness 16-32` (paper-grade) and writes pdbqt through `prepare_receptor4.py`. Higher exhaustiveness tightens the score floor (~0.3–0.6 kcal/mol).
- **Sample budget**: TargetDiff 100 samples × 1000 DDPM steps = 100k NFE/pocket; Lambda's SBDD lane is diffusion-free (GNN + MCTS), so NFE is defined only for the baseline. TargetDiff's reported success rates assume this budget; Lambda's "metalloprotein" protocol is single-pocket so no NFE aggregation.
- **Reference ligand**: TargetDiff compares to the bound ligand in the same pocket ("High Affinity" = fraction beating reference Vina Dock); Lambda compares to either HEM (cofactor) or a known inhibitor (e.g. staurosporine analog) — not the same statistical frame.
- **Metal handling**: TargetDiff is **metal-agnostic** (no metal atom type in the discrete vocab beyond `K` categories of element); Lambda explicitly models Fe/Zn/Cu via metal-aware atom types and metal-coordination losses. Hence Vina Dock on Lambda's metalloprotein pockets is not directly comparable to TargetDiff's Table 1.
- **Evaluation scope**: TargetDiff reports Vina Score / Min / Dock + High Affinity + QED + SA + Diversity + Success Rate (7 metrics); Lambda's `1h36` run additionally reports geometry RMSD to bound ligand, metal-coordination fidelity, and poseCheck clashes — strictly richer but not directly commutable.

## TL;DR
TargetDiff (Guan et al., ICLR 2023, arXiv:2303.03543) reports 58.1% high-affinity hits, Vina Dock = −7.80 kcal/mol, QED = 0.48, SA = 0.58, Diversity = 0.72 over 100 CrossDocked2020 pockets at 1000 DDPM steps × 100 samples = 100k NFE/pocket. Lambda's `1h36`-only single-pocket metalloprotein protocol is a different statistical frame: not averaged, not metal-aware, Vina exhaustiveness 16–32 vs QVina default 8.

## 5-line headline
1. CrossDocked2020 pocket10, 100 held-out test pockets (same split as AR / Pocket2Mol).
2. DDPM 1000 steps × 100 samples = **100 000 NFE per pocket** (10 M across the benchmark).
3. **Vina Dock −7.80 / −7.91 kcal/mol**, **High Affinity 58.1%**, **QED 0.48**, **SA 0.58**, **Diversity 0.72**.
4. Vina docking via QVina (`vina==1.2.2`, exhaustiveness 8, UFF torsion pruning, `size_factor=1.2`).
5. Lambda `1h36-only` = single-pocket, metal-aware, Vina exh ≥ 16, plus extra metal-coordination / RMSD metrics — **not** commensurable with TargetDiff's Table 1.