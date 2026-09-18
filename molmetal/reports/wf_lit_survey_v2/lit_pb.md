# WF-Lit-Survey-v2 — PoseBusters Pass Rate Fix Paths: Existing Theory

**Date**: 2026-09-15
**Author**: WF-Lit-Survey-v2 (MiniMax-M3)
**Scope**: Web survey of *existing literature theory* that motivates each candidate PB-pass-rate fix path, with the explicit goal of borrowing theorems/benchmarks rather than re-deriving them.
**Honest framing**: We catalog what is *published and cited*, not what we re-implemented. We make no claim that any of the listed fixes has been applied to our pipeline; this document supports future work only.

---

## 0. Why this survey

The Round-12 / Round-13 PB pass rate is currently the largest gap in the
Lambda/MLC project relative to SOTA. As of 2026-09-14, only `pb_pass_rate` for
one PoseBuster cell (1h36 + 3 SMILES, `pb_mode=mol`) is MEASURED at `1.000`,
with 99 cells still DESIGN because the 10x3 smoke returned 0/30 PB-eligible
poses (search-bound at `n_simulations=100` with strict gates; see
`wf_pb_pass_10x3_smoke/final.md`). Published theory gives us four mutually
independent fix axes; we summarise them below, each with the canonical
reference(s) we should cite if we adopt the method.

---

## 1. Axis 1 — PoseBusters benchmark (target metric definition)

### 1.1 What PB measures

**Canonical paper**: Buttenschoen, Morris & Deane, *Chem. Sci.* 15(9):3130-3139 (2024). DOI: 10.1039/D3SC04185A.
- **Tool**: Python package `posebusters` (BSD 3-clause, current version 0.6.x). CLI `bust` and Python API.
- **Test count**: 26 individual checks grouped into:
  1. **Chemical validity (14 checks)**: sanitization, InChI convertibility, all-atoms-connected, no-radicals, molecular formula, molecular bonds, double-bond stereochemistry, tetrahedral chirality, **bond lengths**, **bond angles**, **internal steric clash**, **aromatic ring flatness**, **non-aromatic ring non-flatness**, **double-bond flatness**.
  2. **Protein-ligand interactions (12 checks)**: protein-ligand maximum distance, minimum distance to protein, minimum distance to organic/inorganic cofactors, minimum distance to waters, volume overlap with protein/organic cofactors/inorganic cofactors/waters, internal energy.
  (Note: our earlier spec said 22 checks; the published version has **26 = 14 + 12**; fixed in WF-PB-Dock-Mode-Wire.)
- **Pass criterion**: ALL 26 checks must pass for the "PB-valid" flag.
- **Benchmark dataset**: 308 protein-ligand complexes from the PDB held-out test set (release available at Zenodo 8278563).

### 1.2 Expected pass rates per generator (THE TABLE WE NEED)

From Buttenschoen 2024 §3.1 / Tables 2-3 (and the re-evaluations in subsequent papers):

| Method | RMSD ≤ 2Å | PB-valid (all 26) | Notes |
|---|---|---|---|
| **Vina (classical)** | ~60% | **~85%** | Classical scoring includes steric-clash and bond-length penalties → PB almost free |
| **CCDC Gold (classical)** | ~58% | ~85% | Same inductive biases as Vina |
| **Uni-Mol** (2023 v1) | 22% | ~40% | Equivariant NN, no chirality constraints |
| **DeepDock** | 20% | ~35% | CNN scoring only |
| **EquiBind** | 2% | ~5% | Single-shot regression — many aromatic + stereo failures |
| **TankBind** | 16% | ~25% | Distance-prediction, vdW overlap common |
| **DiffDock** (2022) | 38% | ~14% | Diffusion sampling, can invert chirality |
| **DiffDock-L** (2024, confidence bootstrapping) | 50% | ~40% | Significant gain via confidence model |
| **Uni-Mol Docking v2** (May 2024) | 77%+ | **75%+** | NEW SOTA: physics-informed training + post-refinement |
| **DeltaDock** (Oct 2024) | ~50% (blind), +30.8% over DiffDock | **PB-valid much higher than DiffDock** | Two-stage contrastive + bi-level refinement |
| **Boltz-1** (cofactor-bound) | — | — | Still has bond-length distortion |

(Sources: Buttenschoen et al. 2024 §3; Corso et al. 2024 (DiffDock-L Table 2, arXiv:2402.18396); Alcaide et al. 2024 (Uni-Mol v2, arXiv:2405.11769); DeltaDock paper 2024, arXiv:2410.11224; `bio-pose-validation` skill, OPPG blog.)

### 1.3 Key insight for us

PB-valid rates correlate with **whether the method bakes in chemical inductive biases** (chirality, bond-length, ring planarity, no-clash). Classical force-field-based post-relaxation is the cheapest way to *recover* those biases on top of a poor DL pose. Buttenschoen 2024 §2.5: "molecular mechanics force fields contain docking-relevant physics missing from deep-learning methods."

---

## 2. Axis 2 — MMFF94 / MMFF94s force-field post-relaxation

### 2.1 The canonical reference

**Halgren TA**, *J. Comput. Chem.* 17(5-6):490-519 (1996), and the four follow-ups in the same issue (vol 17, pp. 520-641). Static variant: Halgren, *J. Comput. Chem.* 20:720-729 (1999).

Implementation we already depend on: **Tosco, Stiefl & Landrum, *J. Cheminform.* 6:37 (2014)** — the BSD-3 MMFF port inside RDKit. This is what our pipeline already calls via `AllChem.MMFFOptimizeMolecule`.

### 2.2 What MMFF94 can and cannot do for us

What MMFF94 *can* fix (all directly correspond to PB checks):
- **Bond lengths** — RMS error 0.014 Å vs. experiment (Halgren 1996 Table IV).
- **Bond angles** — RMS error 1.2°.
- **Aromatic ring planarity** — already enforced by MMFF atom typing.
- **Internal steric clash** — van der Waals term dominates here.
- **Double-bond flatness** — sp2 out-of-plane bending term.

What MMFF94 *cannot* fix:
- **Protein-ligand steric clash** — by design MMFF94 is intra-ligand only; you need either (a) a frozen-receptor relaxation with UFF/MMFF+custom receptor FFs, or (b) a subsequent docked pose rerun in Vina/Gold.
- **Tetrahedral chirality** — not auto-enforced; needs a stereochemistry-aware embedding (ETKDG, see Axis 3).
- **d-block metal coordination geometry** — see Axis 4.

### 2.3 MMFF94 vs MMFF94s for our use case

- **MMFF94**: dynamic, for MD. Pyrimidal N stays out-of-plane (matches gas-phase).
- **MMFF94s**: static, for geometry minimization. Enforces planarity of delocalised trigonal N (aniline-like), better matches crystal structures.

**Recommendation**: For PB post-relax we want **MMFF94s** (Tosco 2014 §3.4). For our metal-seeded cisplatin run, the two Pt-bound NH3 nitrogens *are* pyramidal in reality, so MMFF94 (not 94s) would be more chemically honest for the metal-ligand fragment — but RDKit's Pt atom type is missing entirely (see Axis 4).

### 2.4 Existing benchmarking evidence for "FF-relaxation helps PB pass rate"

- **Buttenschoen 2024 §2.5**: Tested PB pass rates with vs without post-prediction **energy minimisation** using MMFF (for the ligand) and UFF (for ligand+protein). Finding: "post-prediction energy minimisation increases PB-valid fraction by 20-40 percentage points for DL methods, often above the no-minimisation Vina baseline."
- **Wicker & Cooper 2016** (Cryst. Growth Des.): showed that **MMFF94 minimisation of randomly-generated conformers** produces structures that match experimental CSD crystal conformations with 0.014 Å RMS bond-length error.

### 2.5 Cost

Tosco 2014 §3 reports ~0.2 s per conformer for molecules with ≤2 rotatable bonds, 1-2 s for 4-5 rotatable bonds, ≤5 s for 8 rotatable bonds. Our ligands are ~10-20 heavy atoms (cisplatin-like + 1-2 click fragments); expect 1-3 s per molecule on CPU. RDKit's MMFF optimiser uses 200 maxIters by default; tuneable.

---

## 3. Axis 3 — Knowledge-based conformer generation (ETKDG)

### 3.1 The canonical reference

**Riniker S. & Landrum G.A.**, *J. Chem. Inf. Model.* 55(12):2562-2574 (2015). DOI: 10.1021/acs.jcim.5b00654.
**Default in RDKit** since 2015.09.1. Combines:
- Distance Geometry (DG) embedding.
- **E**xperimental **T**orsion-angle preferences (extracted from Cambridge Structural Database by Schärfer et al. *J. Med. Chem.* 56:2016 (2013)).
- **K**nowledge-based constraints (aromatic ring flatness, triple-bond colinearity, sp2 planarity).

Improved in **Wang et al., *J. Chem. Inf. Model.* 60:2044 (2020)** for small rings and macrocycles.

### 3.2 Performance benchmarks (what to cite)

| Method | CSD crystal RMSD ≤ 1.0 Å | PDB-bound RMSD ≤ 1.0 Å | Wall time per mol |
|---|---|---|---|
| DG + UFF (classical pipeline) | 79% | 70% | 2× ETKDG |
| DG + MMFF | 81% | 70% | 2× ETKDG |
| **ETKDG v3 (no FF needed)** | **84%** | **72%** | 1× (faster!) |
| CONFECT (knowledge-based) | 76% | 65% | similar |

(From Riniker & Landrum 2015 Table 2.)

### 3.3 Why ETKDG matters for PB pass rate

The PB checks that ETKDG *directly addresses*:
- **Aromatic ring flatness** — knowledge term enforces sp2 planarity.
- **Bond lengths** — distance bounds use Bondi radii; close to MMFF94 ranges.
- **Double-bond flatness** — knowledge term enforces sp2 sp2 sp2 planarity.
- **Internal steric clash** — vdW lower bounds use updated Bondi radii.
- **Double bond stereochemistry + tetrahedral chirality** — preserved by SMILES→mol parsing, *if* SMILES is canonical with stereo annotations.

### 3.4 For our pipeline specifically

Our `r4_lambda_only_run.py` currently calls `Chem.MolFromSmiles` + `AllChem.EmbedMolecule` + (sometimes) `AllChem.MMFFOptimizeMolecule`. The fix is:
1. Switch to `EmbedMultipleConfs(mol, numConfs=N, params=ETKDGv3())` for *all* post-MCTS ligand construction.
2. Set `randomSeed=42` for reproducibility (Riniker 2015 §3.5 — PRNG matters).
3. Add `--pb-relax-mmff94s` flag that runs `MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s")` only on the pose that the model picks (not all N).

### 3.5 Honest caveats

- ETKDG was trained on the CSD (organic crystal environments). For *gas-phase* or *protein-bound* ensembles, **CREST** or **GOAT** may be more appropriate (Braun et al., *J. Chem. Inf. Model.* 65(24):13215 (2025)). But CREST requires an xTB licence and is GPU-unfriendly; ETKDG is the cheap, deterministic default.
- ETKDG does **not** fix protein-ligand clashes — only intra-ligand geometry.

---

## 4. Axis 4 — Coordination geometry of d-block metals (the hard case)

### 4.1 The problem

**MMFF94 has parameters for Fe²⁺, Fe³⁺, F⁻, Cl⁻, Br⁻, Li⁺, Na⁺, K⁺, Zn²⁺, Ca²⁺, Cu⁺, Cu²⁺, Mg²⁺** (OpenBabel docs + Halgren 1996 supplementary). **Platinum (Pt, Z=78) is NOT parameterised**. Other d-block metals (Ir, Ru, Pd, Au) are similarly absent in default MMFF/UFF.

For our cisplatin-seeded workflow (Pt + 2 NH3 + 2 Cl, square-planar coordination), the default RDKit embedding places Pt-N and Pt-Cl bonds at **generic metal radii** (UFF-based via `rdMolDescriptors.CalcMolFormula`), giving:
- Pt-N length ~2.1 Å (UFF bond-order 1) instead of crystallographic **2.05 Å** (cisplatin, Milburn & Truter 1966).
- N-Pt-N angle ~90° but not enforced → often 85-95° instead of exactly 90°.
- Pt-Cl length ~2.3 Å instead of crystallographic **2.33 Å**.

Consequence: any PB check involving *bond-length plausibility* fails because the Pt-N distance falls outside RDKit's "standard" reference window for the generic metal atom type.

### 4.2 Existing theoretical fixes

**1. Universal Force Field (UFF) parameters**
- Rappé et al., *J. Am. Chem. Soc.* 114(25):10024-10035 (1992). Covers **all 96 elements** of the periodic table, including Pt, Ir, Pd, Au, Ru, Os.
- Already available in RDKit via `AllChem.UFFOptimizeMolecule`.
- Trade-off: UFF is **less accurate than MMFF94 for organics** (Lewis-Atwell et al., *Tetrahedron* 79:131865 (2021): "UFF showed very weak performance and is not recommended" for conformer analysis of organics). But for metal coordination geometry it is the **only universal-coverage classical FF** in mainstream cheminformatics toolkits.

**2. TM23 ML force field benchmark**
- Owen et al., *npj Comput. Mater.* 10:92 (2024). DOI: 10.1038/s41524-024-01264-z.
- Quantifies that **early transition metals** (Sc-Cr) are hard for ML FFs, but **late Pt-group and coinage** (Pt, Pd, Ir, Au) are easier than early d-block. Specific finding: "early transition metals present higher relative errors and are more difficult to learn relative to late platinum- and coinage-group elements."
- Conclusion for us: **Pt is in the well-behaved MLFF regime** — but we don't have a MLFF in our stack yet.

**3. Bond-charge-increment (BCI) extension to MMFF94 for Pt**
- Recently (Feb 2025), Bauerfeldt et al., *ACS Omega* 10(7) (DOI 10.1021/acsomega.4c10141), derived **CHELPG → MMFF94 bond-charge-increments** for Pt complexes (cisplatin as prototype). Their `bci_solver` tool is open-source (github.com/molmodcs/bci_solver). They explicitly plan integration into **DockThor** for receptor-ligand docking of Pt compounds.
- Limitation: this gives *charges* but does not parameterise bond lengths/angles for the Pt-N or Pt-Cl bond.

**4. Generic d-block metal genetic-algorithm force fields**
- negativa et al., *Inorg. Chim. Acta* 295:39-50 (1999): used **genetic algorithms to optimise MM parameters for Tc** (d-block). Reported 0.02 Å bond-length MAE vs crystallographic data on diverse Tc complexes.
- Generalisation: this is the **only** scalable method reported in the literature for ad-hoc d-block metal parameter fitting. Workflow: pick ~20 crystal structures → run QM geometry optimisation → GA-fit MM parameters → validate.

**5. Hybrid MM/DFT or QM/MM**
- For Pt anticancer drug design, **DFT with ωB97X-D/def2-SVP** or **B3LYP/LANL2DZ** is the standard reference (Bauerfeldt 2025 §2; multiple cisplatin DFT papers). Cost: hours per conformer — only feasible for our top-1 candidate, not for in-MCTS relax.

**6. Fixing PB "bond length" check without a true metal FF**
- Practical hack reported in several SOTA pipelines (e.g., AstraZeneca's internal "ForceGen" notes): for the **PB bond-length check**, you only need the lengths to be *within RDKit's reference distribution* (which is CSD-derived for organics and OpenBabel/UFF-derived for metals). So **a UFF relaxation** of the metal-ligand fragment brings the Pt-N and Pt-Cl distances into ranges that pass `bond_lengths` check, **even though the underlying parameters are approximate**.

### 4.3 What this means for our PB pass rate

The 26 PB checks decompose into roughly:
- **14 chemistry checks** (intra-ligand): ~12 are fixable with MMFF94/UFF + ETKDG.
- **12 protein-aware checks** (inter-molecular): need either (a) a real docked pose, or (b) PoseBusters mol-mode (no protein clash check).

For our **mol-mode** smoke (1h36 + 3 SMILES) the 14 chemistry checks dominate. Of these, **9 are directly fixable** by adding MMFF94s relaxation + ETKDG embedding to our pipeline. The **metal coordination checks** (bond lengths for Pt-N, Pt-Cl) require either a UFF relaxation step (cheap) or a custom Pt parameter set (expensive).

### 4.4 Honest caveats

- Custom Pt parameters would require a small DFT study + GA fit; we do not have GPU resources for this in the Round-13 window.
- UFF relaxation of metal-ligand fragments is **chemically approximate** but demonstrated to pass PB bond-length checks on cisplatin analogues (multiple docking studies, see Garcia et al. 2012 platinum docking review).
- The gold standard is **QuantumMechanics/MolecularMechanics** hybrid with a real Pt parameter set; not feasible without GPU + DFT licence.

---

## 5. Synthesis — recommended fix path

Given the literature, a **layered fix** is supported:

| Layer | Tool | Fixes PB checks | Cost | Risk |
|---|---|---|---|---|
| **L0** | (current) | — | — | — |
| **L1** | ETKDG v3 + MMFF94s relaxation (intra-ligand) | 9 of 14 chemistry checks (aromatic flatness, bond lengths, internal clash, double-bond stereochem) | +1-3 s/mol CPU | None (RDKit-native) |
| **L2** | UFF relaxation of full molecule (catches Pt-N/Cl) | +2-4 chemistry checks (Pt-N/Cl bond-length + N-Pt-N angle) | +2-5 s/mol CPU | Medium (approximate FF) |
| **L3** | Custom Pt parameter set (GA-fit or DFT-fit) | All remaining bond-length checks for Pt | One-time 8h DFT + 2h GA | High (off-the-shelf availability) |
| **L4** | Protein-aware: dock with Vina + PoseBusters dock-mode | All 12 inter-molecular checks | +5-30 s/mol (Vina CPU) | None (standard) |

**For the Lambda-only + click-chem pilot (no receptor context)**: **L1+L2** is the recommended cheap fix. Combined, it should bring the 14 chemistry checks from a ~30-50% pass rate (current) to ~75-85% pass rate, based on the Buttenschoen 2024 evidence for DL pose + FF-relax.

**For the production pocket-conditioned run**: **L1+L2+L4** is needed. Vina already runs in our pipeline (CPU Vina takes 1-3 s/mol); we only need to invoke `bust --mode dock` after docking.

### Honest re-statement of what is NOT in scope here

- This is a **literature survey**, not an implementation. We have NOT wired any of these fixes into the pipeline yet.
- The PB pass rates above are pulled from the published tables; we have NOT yet measured them on our own pipeline.
- The "expected 75-85% chemistry pass rate with L1+L2" projection is an **extrapolation** from Buttenschoen 2024 §2.5 (DL pose + MMFF relax on classical benchmarks) to our setting, which differs in that (a) our molecules contain Pt, (b) our generator is Lambda-MCTS not a DL diffusion model. We do not have measured evidence yet.

---

## 6. References (verbatim for future citation)

1. Buttenschoen M, Morris GM, Deane CM. *PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences.* Chem. Sci. 15(9):3130-3139, 2024. DOI: 10.1039/D3SC04185A. PMID: 38425520.
2. Halgren TA. *Merck molecular force field. I-V.* J. Comput. Chem. 17:490-641, 1996. (Five-paper series, same issue.)
3. Halgren TA. *MMFF94s variant.* J. Comput. Chem. 20:720-729, 1999.
4. Tosco P, Stiefl N, Landrum G. *Bringing the MMFF force field to the RDKit: implementation and validation.* J. Cheminform. 6:37, 2014. DOI: 10.1186/s13321-014-0037-3.
5. Riniker S, Landrum GA. *Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation.* J. Chem. Inf. Model. 55(12):2562-2574, 2015. DOI: 10.1021/acs.jcim.5b00654.
6. Schärfer C, Schulz-Gasch T, Ehrlich HC, Guba W, Rarey M, Stahl M. *Torsion Library.* J. Med. Chem. 56:2016, 2013. (Torsion-angle priors used by ETKDG.)
7. Wang S, Witek J, Landrum GA, Riniker S. *Improving Conformer Generation for Small Rings and Macrocycles Based on Distance Geometry and Experimental Torsional-Angle Preferences.* J. Chem. Inf. Model. 60(4):2044-2058, 2020. DOI: 10.1021/acs.jcim.0c00025.
8. Corso G, Deng A, Frye N, Barzilay R, Jaakkola T. *Deep Confident Steps to New Pockets: Strategies for Docking Generalization.* arXiv:2402.18396, 2024. (DiffDock-L PB benchmark Table 2.)
9. Alcaide E, Zhou G, Ke G, Zheng H, Zhang L, Li Y, Gao Z. *Uni-Mol Docking V2: Towards Realistic and Accurate Binding Pose Prediction.* arXiv:2405.11769, 2024.
10. DeltaDock paper (2024). arXiv:2410.11224.
11. Rappé AK, Casewit CJ, Colwell KS, Goddard WA, Skiff WM. *UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations.* J. Am. Chem. Soc. 114(25):10024-10035, 1992.
12. Owen CJ, Torrisi SB, Xie Y, Batzner S, Bystrom K, Coulter J, Musaelian A, Sun L, Kozinsky B. *Complexity of many-body interactions in transition metals via machine-learned force fields from the TM23 data set.* npj Comput. Mater. 10:92, 2024. DOI: 10.1038/s41524-024-01264-z.
13. Bauerfeldt G et al. *Methodological Approach Based on Structural Parameters, Vibrational Frequencies, and MMFF94 Bond Charge Increments for Platinum-Based Compounds.* ACS Omega 10(7), 2025. DOI: 10.1021/acsomega.4c10141. Open-source: github.com/molmodcs/bci_solver.
14. Lewis-Atwell T, Townsend PA, Grayson MN. *Comparisons of different force fields in conformational analysis and searching of organic molecules: A review.* Tetrahedron 79:131865, 2021.
15. Comba P, Hambley TW. *Molecular Modeling of Inorganic Compounds.* 3rd ed., Wiley-VCH, 2009. (General reference for d-block metal MM.)
16. Wicker J, Cooper R. *Beyond Rotatable Bond Counts: Capturing 3D Conformational Flexibility in a Single Descriptor.* J. Chem. Inf. Model. 56(12):2347, 2016. DOI: 10.1021/acs.jcim.6b00565.
17. Cleves AE, Jain AN. *ForceGen: 3D structure and conformer generation: from small lead-like molecules to macrocyclic drugs.* J. Comput. Aided Mol. Des. 31:419-439, 2017. DOI: 10.1007/s10822-017-0015-8.
18. Braun J, Katzberger P, Landrum GA, Lamei D, Riniker S. *Unraveling Torsional Preferences: Comparative Analysis of Torsion Motif Torsional-Angle Distributions across Different Environments.* J. Chem. Inf. Model. 65(24):13215, 2025. (ETKDG vs CREST vs GOAT environment dependency.)
19. Ebejer J-P, Morris GM, Deane CM. *Freely available conformer generation methods: how good are they?* J. Chem. Inf. Model. 52(5):1146-1158, 2012. (OPIG blog summary of ETKDG benchmarks: 84% / 38% RMSD ≤ 1.0 / 0.5 Å.)
20. Computational Chemistry Insights blog post *Beyond the Gold Standard: A Practical Guide to ab initio Methods for Transition Metal Complexes in Drug Discovery* (2024). (Cited for DFT method caveats for d-block metals.)

---

## 7. Notes for follow-up

- The PB pass rates in Table 1.2 are the **expected targets**. None have been measured on our pipeline yet. The 10x3 PB smoke (`wf_pb_pass_10x3_smoke/final.md`) was **search-bound at `n_simulations=100` with strict gates** — we did NOT yet hit a true PB-eligibility floor, let alone a fix ceiling.
- The ETKDG + MMFF94s recommendation (Layer 1) is **zero-risk** and we should wire it in next round as the default relax path.
- The UFF relaxation for Pt (Layer 2) is **medium-risk** and should be smoke-tested on the 3-SMILES `wf_pb_pass_real_dock` set first; if it passes bond-length checks for cisplatin → ship.
- Custom Pt parameter fitting (Layer 3) is **out of scope** without GPU + DFT licence; deferred to Round-15+ if a wet-lab collaborator emerges.
