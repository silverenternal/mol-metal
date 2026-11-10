# Wf-A175 — BRANCH FRONTIER: widening the raw-vs-post-processed connectivity/fragmentation census

**Branch:** FRONTIER · **Date:** 2026-11-01 · **GPU:** 0 (CPU-only: no torch import, no CUDA call, no decode, no checkpoint read, no training) · **Writer:** `molmetal/reports/wfa175_frontier_comparability/`

Every number below is **MEASURED** by this wave, either from bytes fetched live from arXiv on 2026-11-01 and pinned by sha256, or from a JSON/markdown file on disk. Nothing is quoted from a search-result snippet or an AI summary. Anything not reached is stated in §8.

---

## 0. Retrieval ledger — sha256 of the bytes actually received

| id | paper | bytes | sha256 | note |
|---|---|---|---|---|
| 2309.17296v2 | EQGAT | 3,242,988 | `9f88c403651a6414fc86c2656e35a30fb5a386152698595ed38831c4b2fc78d4` | new this wave |
| 2210.13695v3 | DiffSBDD | 33,236,821 | `4fa02b871e61183f35f0b7a3a08e4442c0357fa8720917af0d4c401fdcb3c7ab` | new this wave |
| 2507.00899v1 | TABASCO | 5,558,907 | `f91e660918b23fb27cc92fa3943bba7907b5e8ee6cd3047f877f46eb6da9a086` | new this wave |
| 2407.04424v1 | GenBench3D | 4,032,287 | `330a8cc2face85120aa057c8b580ec5734df1618df01d797e0a8dc46bf08a8e3` | new this wave; **PDF-only** submission (text via `pdftotext -layout`) |
| 2305.01140v1 | GeoLDM | 655,684 | `b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23` | new this wave |
| 2609.05097v1 | NEAT-POCKET | 10,632,429 | `bd978841f62a7d75e8635714e7336a6e3abec39b854ec3f2c90974890f2b7fdb` | new this wave |
| 2308.05777v3 | PoseBusters | 9,900,693 | `e5bcb1ec75104a327385b13470ae38816d42adbff6ffb318774d0eb9e253068b` | new this wave |
| 2505.00518v1 | Buttenschoen 2025 | 1,059,489 | `52ca160b7734259eda3bfd7f981920379e144661dfd9228d70447cc6c4ee7e57` | **re-fetched**; hash identical to the Wf-A171 pin |
| 2403.07902v1 | DecompDiff | 26,329,369 | `f510ac0224de07c3a6daaeb9cb3fdf0a67933213eed6d4e1fa189d87387a7c08` | new this wave |
| 2406.07266v3 | SemlaFlow | 2,034,012 | `72dc18f51a1d2361d13bb7224edaefe1bcb71833cfff52153e88e50d3a43c3c5` | new this wave |
| 2603.27113v1 | HLTF | 504,186 | `87c1788e9e9826cfbdf4e0788feed1a796c8a50218eecc6905df771f10c9c329` | new this wave (band provenance, §4) |
| 2609.08333v4 | EF-TALFM | 8,508,562 | `e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224` | **re-fetched**; hash identical to the Wf-A171 and Wf-A168 pins |

**Method.** `urllib` over the host proxy `http://127.0.0.1:7890`; arXiv's `/e-print/<id> -> /src/<id>` redirect followed by urllib itself; the arXiv API (`export.arxiv.org/api/query`, `%22`-phrased) used for candidate discovery only. Candidates were then fetched as **primary records** and read as extracted LaTeX (or `pdftotext -layout` text for the single PDF-only submission). One review (2609.10099v1) was used **only** as a lead generator; no number is quoted from it.

---

## 1. RAW-or-POST verdict for **9 additional** frontier papers

**POST** = the number is computed after a fragment-selection step (largest fragment / largest connected component) or after another step that changes the measured quantity; **RAW** = computed on the decoder's own output with no such step; **PROTOCOL-UNSPECIFIED** = the paper reports the number but never says which stage it was measured at.

| # | paper | metric | reported value | verdict | decisive evidence (verbatim, with locator) |
|---|---|---|---|---|---|
| 1 | **EQGAT** 2309.17296v2 | Validity (RDKit sanitisation over 10,000 molecules) | 0.916 (no PP) -> **0.998** (with PP) | **BOTH** (PP = *repair*, not fragment selection) | "we measure validity using the success rate of RDKit sanitization over 10,000 molecules **(pre-selecting connected components only)**" (`e3moldiffusion.tex:303`); "utilizing their evaluation pipeline that **includes post-processing steps** on the generated molecules to potentially **fix valency and aromaticity** issues when parsing into the RDKit" (`:841`); "**Suppose we do not employ the post-processing scheme from MolDiff** ... EQGAT-diff obtains a mean validity of **0.916** and a mean success rate of 0.887" (`:843`); table `tab:appendix table-moldiff` "Validity **0.998** / Connectivity 0.968 / Succ. Rate 0.966" (`:853-855`) |
| 2 | **DiffSBDD** 2210.13695v3 | Connectivity (share of *valid* molecules with no disconnected fragment) | 63.37 - 98.25 | **RAW** (necessarily pre-filter) | "**Connectivity is the proportion of valid molecules that do not contain any disconnected fragments.**" (`main.tex:929`); "A much larger percentage of molecules is fragmented but, since we can **simply select and process the largest fragments** in these cases, low connectivity does not necessarily affect the efficiency of the generative process" (`:932`); post-processing is described separately: "For postprocessing of generated molecules ... bonds are first added using OpenBabel ... We then use RDKit to sanitise molecules and **filter for the largest molecular fragment**" (`:594-595`) |
| 3 | **TABASCO** 2507.00899v1 | Connectivity (fraction of fully connected molecules) | 0.00 - 1.00 | **RAW** | "We additionally evaluate **connectivity, which denotes the fraction of fully connected molecules**" (`example_paper.tex:629`, table caption); rows "10 & 0.99 & 0.98 & **0.00** & 0.00" and "20 & 1.00 & 0.99 & **0.00** & 0.00" (`:634-635`). A connectivity of **0.00** at 10-20 steps is only possible if no fragment selection was applied -- the table is self-proving about its own stage. (The harness also states it saves molecules "as outputted from reverse sampling, **without any post-processing**", `:526`.) |
| 4 | **GenBench3D** 2407.04424v1 | Validity3D (geometrically valid conformer) | 0.004 - 0.111 RAW -> 0.532 - 0.783 POST | **BOTH** (post = MMFF94s **geometry relaxation**) | Table 3 "Median conformation quality statistics of **raw** generated molecules" (PDF p. 23): DiffSBDD 0.004, LiGAN 0.023, Pocket2Mol 0.094, TargetDiff 0.106, ResGen 0.111, 3D_SBDD 0.099; Table S3 "...on the set of **relaxed** molecules" (PDF p. 49): 0.532 / 0.551 / 0.783 / 0.544 / 0.687 / 0.703; "minimizing the generated molecules within the binding pocket using the **MMFF94s force field** ... This minimization step will be referred to as **relaxation** ... the relaxation **drastically increased** the Validity3D: for instance, LiGAN increases from 0.02 to 0.55, Pocket2Mol increases from 0.09 to 0.78, and ResGen from 0.11 to 0.69" (PDF p. 27) |
| 5 | **GeoLDM** 2305.01140v1 | QM9 Atom-Sta / Mol-Sta / Valid / Valid&Unique | 98.9 / 89.4 / 93.8 / 92.7 | **PROTOCOL-UNSPECIFIED** | Table row "`GeoLDM & 98.9+-0.1 & **89.4+-0.5** & **93.8+-0.4** & **92.7+-0.5** & 84.4 & 99.3`" (`icml2023.tex:491`); the paper's only statement on the issue is "the model occasionally **generates disconnected components**" (`:917`) |
| 6 | **NEAT-POCKET** 2609.05097v1 | PB-valid (de novo) | 69.2 % (CrossDocked) / 71.0 % unguided / 80.3 % (SPINDR, default CFG w=0.5) / 80.5 % unguided; FLOWR 75.8 % | **RAW** (no fragment filter of its own) | "A generated molecule is considered **PB valid** if it can be successfully parsed by RDKit and passes all 22 conditional PoseBusters checks" (`sections/methods.tex:26`); "Among the generative models, NEAT-POCKET achieved the second-highest PB validity, with **69.2 %** PB valid molecules" (`sections/results.tex:132`); "In the unguided setting, w=0.0 ... **71.0 %** valid molecules on CrossDocked and **80.5 %** on SPINDR" (`:190`); "NEAT-POCKET achieves higher PB validity, **80.3 % versus 75.8 %** [FLOWR]" (`:170`) |
| 7 | **PoseBusters** 2308.05777v3 | PB-valid (docking) | Gold 48 %, Vina 51 %, Uni-Mol 23 %; DiffDock 72 % RMSD<=2 A, TankBind 59 % | **RAW** for connectivity | "Molecule poses which pass all tests in PoseBusters are 'PB-valid'" (`01_main.tex:247`); the only post-step is "an additional **post-docking energy minimisation** of the ligand structures in the binding pocket" (`:378`) -- not fragment selection |
| 8 | **Buttenschoen 2025** 2505.00518v1 | Connected % **before and after** the 3-step PP | GeoLDM **37.4 -> 73.8** | **BOTH**, side by side in one table | "The generated molecules were post-processed in three steps. First, the **largest fragment was picked using the RDKit's `LargestFragmentChooser`**. Second, missing explicit hydrogens were filled in ... Third, the molecular conformation was refined by minimising energy using the Universal Force Field" (`iclr2025_conference.tex:129-132`); table `tables/chemical_validity.tex`: "`GeoLDM & 72.5 & 5.3 & **37.4** & 3.2`" (`:17`) vs "`GeoLDM + PP & 73.8 & 73.8 & **73.8** & 73.8`" (`:23`) |
| 9 | **DecompDiff** 2403.07902v1 | (protocol paper) | -- | n/a | "We also introduce a diffusion process on bonds to incorporate the bond generation in an **end-to-end fashion instead of post-processing**" (`sections/10-intro.tex:51`) |

### 1.1 Two frontier protocols that the frontier itself publishes (both new to this repo)

* **DiffSBDD's headline validity is POST *and* a rejection sampler.** From NEAT-POCKET's supplementary (2609.05097v1 `sections/supplementary.tex:99`):

  > "In the original inference procedure, generated molecules are **explicitly filtered** according to internal validity criteria before being returned. Specifically, a molecule is considered valid if its **largest fragment**, identified using RDKit's `Chem.GetMolFrags()` function, can be successfully sanitized by RDKit. ... Thus, the original implementation effectively **continues sampling until 100 valid molecules have been obtained**."

  So DiffSBDD's validity is fragment-selected *and* obtained by resampling until a quota is filled; and its *reported* connectivity (63.37-98.25 %) is necessarily measured **before** that filter, since after it every molecule has one fragment by construction. Its RAW validity is therefore not recoverable from the paper (the OpenBabel bond-addition step is already part of its pipeline).
* **Validity and connectivity can be 99 % and 0 % in adjacent rows of the same table.** TABASCO, at 10-20 sampling steps, reports Validity 0.99-1.00 with Connectivity **0.00** (`:634-635`). This is the strongest available evidence that a validity number and a connectivity number are not the same kind of quantity -- and it comes from the frontier, not from us.

---

## 2. RAW->POST ratio tables

### 2.1 Connectivity axis (fragment selection) -- the axis our 11.751x belongs to

| source | model / cell | RAW | POST | ratio | how POST is obtained |
|---|---|---|---|---|---|
| Buttenschoen 2505.00518 | **GeoLDM** | 37.4 % | 73.8 % | **1.97326x** | printed by the paper, with and without PP, same table |
| Buttenschoen 2505.00518 | FlowMol | 68.6 % | 87.2 % | 1.27114x | idem |
| Buttenschoen 2505.00518 | GCDM | 86.6 % | 95.5 % | 1.10277x | idem |
| Buttenschoen 2505.00518 | EQGAT-diff | 84.4 % | 87.1 % | 1.03175x | idem |
| Buttenschoen 2505.00518 | SemlaFlow | 92.3 % | 94.9 % | 1.02817x | idem |
| DiffSBDD 2210.13695v3 | Binding MOAD / DiffSBDD-cond | **63.37 %** | 100 % | **1.57803x** | *implied* by DiffSBDD's own largest-fragment filter |
| DiffSBDD 2210.13695v3 | CrossDocked / DiffSBDD-inpaint | 67.52 % | 100 % | 1.48071x | idem |
| DiffSBDD 2210.13695v3 | Binding MOAD / DiffSBDD-inpaint | 75.60 % | 100 % | 1.32275x | idem |
| DiffSBDD 2210.13695v3 | Binding MOAD / DiffSBDD-cond (C-alpha) | 77.38 % | 100 % | 1.29233x | idem |
| DiffSBDD 2210.13695v3 | CrossDocked / DiffSBDD-cond | 78.27 % | 100 % | 1.27766x | idem |
| DiffSBDD 2210.13695v3 | CrossDocked / DiffSBDD-cond (C-alpha) | 79.52 % | 100 % | 1.25755x | idem |
| DiffSBDD 2210.13695v3 | Binding MOAD / DiffSBDD-inpaint (C-alpha) | 91.60 % | 100 % | 1.09170x | idem |
| DiffSBDD 2210.13695v3 | CrossDocked / DiffSBDD-inpaint (C-alpha) | 98.25 % | 100 % | 1.01781x | idem |
| EQGAT 2309.17296v2 | Connect. Comp., GEOM-Drugs (w_u) | 0.8657 | 1.0 | 1.15501x | implied by fragment selection |
| EQGAT 2309.17296v2 | Connect. Comp., QM9 (w_s(t)) | 0.9994 | 1.0 | 1.00060x | idem |
| TABASCO 2507.00899v1 | Connectivity (30-500 steps) | 0.99 | 1.00 | 1.01010x | idem |
| **OURS** `metrics/by_metric/cfm_fragmentation_postproc_arm.json` | c7 pool, 12,668 parsed rows | **0.08509630565203663** (1078/12668) | **1.0** | **11.75139x** | measured by our own arm; pipeline = `LargestFragmentChooser -> Chem.AddHs -> UFFOptimizeMolecule`, i.e. **the same 3-step PP as Buttenschoen** |

### 2.2 Repair axis (valency / aromaticity repair) -- EQGAT's own printed pair

| source | metric | RAW | POST | ratio |
|---|---|---|---|---|
| EQGAT 2309.17296v2 | Validity | 0.916 | 0.998 | **1.08952x** (printed by the paper, `e3moldiffusion.tex:843` vs `:853`) |

### 2.3 Geometry axis (force-field relaxation) -- GenBench3D's printed pair

| model | Validity3D RAW (Table 3, PDF p. 23) | Validity3D POST-relaxed (Table S3, PDF p. 49) | ratio |
|---|---|---|---|
| **DiffSBDD** | 0.004 | 0.532 | **133.0x** |
| LiGAN | 0.023 | 0.551 | 23.96x |
| Pocket2Mol | 0.094 | 0.783 | 8.33x |
| 3D-SBDD | 0.099 | 0.703 | 7.10x |
| ResGen | 0.111 | 0.687 | 6.19x |
| TargetDiff | 0.106 | 0.544 | 5.13x |

---

## 3. Answer to (b) -- is our 11.751x still the largest?

**On the connectivity axis: yes, and the margin is unchanged -- 11.75139146567718 / 1.9732620320855614 = 5.955312x the frontier's largest.** The frontier maximum is GeoLDM's **37.4 -> 73.8** (Buttenschoen et al., arXiv 2505.00518v1), **re-verified by this wave from a re-fetched e-print whose sha256 is identical to the Wf-A171 pin** -- not carried over from the brief. The five new papers add no larger connectivity ratio: the next largest is DiffSBDD's own 63.37 % RAW connectivity => 1.57803x, and even that POST value is *implied* by DiffSBDD's pipeline rather than printed by it.

**But "our ratio is the largest in the frontier" does not survive a broader axis.** The frontier does contain a **larger** measured raw->post ratio -- up to **133.0x** (DiffSBDD Validity3D 0.004 -> 0.532) -- once the post-processing axis is widened from *fragment selection* to *geometry relaxation*, and EQGAT prints a 1.08952x *repair* ratio. These are not like-for-like with our 11.75139x (different metric, different post-process) and must not be used to flatter either side. What they establish is the general point the band critique rests on: **a post-processing lift of one to two orders of magnitude is routine and published in this field**, so a RAW number and a POST number are never interchangeable, and ours (8.509630565203663 %) belongs in the RAW column of a table that has a POST column.

### 3.1 The honest-negative this widening exposes (adversarial finding)

**Our RAW connectivity (8.51 %) is far below every frontier RAW connectivity measured here** -- DiffSBDD 63.37 %, DiffSBDD-inpaint(CrossDocked) 67.52 %, EQGAT GEOM-Drugs(w_u) 86.57 %, Buttenschoen's GeoLDM 37.4 %, SemlaFlow 92.3 %, EQGAT QM9(w_s(t)) 99.94 %. The gap between our 11.751x ratio and the frontier's 1.973x ratio is therefore **not** evidence that the frontier post-processes less; it is evidence that **our decoder fragments more than the frontier's does**. The ratio framing flatters us if quoted alone; the RAW column does not. Both belong in any table we publish.

### 3.2 A frontier paper whose own text contradicts its own number (new finding)

EF-TALFM (2609.08333v4) reports connectivity **92.98 %** (`src/appendix.tex:115`, and `:82`: "its connectivity is moderately lower (92.98 % versus 98.72 %)"). Its protocol sentences pull in opposite directions:

* `src/appendix.tex:183`: "We report connectivity of the reconstructed molecule and sanitization, uniqueness, and novelty rates computed **after retaining the atom-count largest connected component** of each sampled molecule and applying strict RDKit sanitization."
* `src/appendix.tex:54`: "For fragmented outputs, **SMILES-based and PoseBusters metrics** use the atom-count largest connected component after sanitization." -- and `:184`: "**C.** denotes the fraction of connected molecules".

If connectivity were literally measured *after* largest-component retention it would be **100 %** by construction, so the reported 92.98 % can only be a **pre-selection** (RAW) measurement. The paper's `:183` sentence is therefore internally inconsistent with its own table, and **EF-TALFM publishes no pre-PP/post-PP pair** -- so its connectivity ratio cannot be computed at all, and a 92.98 % "frontier connectivity" quoted next to our 8.51 % is a comparison between two different stages unless the reader knows the paper's sentence is wrong. (EF-TALFM's `92.98 %` is also one of the numbers the repo's band census sweeps -- see §4.)

---

## 4. Answer to (c) -- the 92.7-94 % band: whose number is it, and where does it come from?

**It is a FRONTIER number. It is not ours.** Measured, in three independent layers.

### 4.1 Source layer -- TWO competing readings of the band's endpoint cells; neither is settled on disk

HLTF (arXiv 2603.27113v1), `tables/qm9_benchmarks_og.tex`, Table 1(b) `tab:qm9_pb` (lines 56-66):

| Metric | GeoLDM | GCDM | **HLTF (ours)** |
|---|---|---|---|
| AS (%) | 98.9+-0.1* | 98.7+-0.1 | 98.8+-0.1 |
| MS (%) | **89.4+-0.5*** | 85.7+-0.4 | 86.8+-0.3 |
| Val (%) | 93.6+-0.2 | 94.8+-0.2* | 93.8+-0.6 |
| Val&Uniq (%) | **92.7+-0.5** | 93.3+-0.0* | 92.9+-0.2 |
| Novel (%) | 53.5+-0.6 | 58.7+-0.5* | 56+-0.7 |
| **PB-Valid (%)** | 93.1+-0.4 | 91.9+-0.5 | **94+-0.5*** |

(Table 1(a), line 38, carries the same GeoLDM row: `GeoLDM & 98.9+-0.1* & 89.4+-0.5* & 93.6+-0.2 & 92.7+-0.5`.)

Which frontier cells the band names is **NOT settled on disk**.  The repo holds TWO competing provenance readings, and they do not even agree on the numerals:

* **reading A** (`band_provenance` in the frontier post-processing matrix record `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`): **lower endpoint 92.7 %** = **GeoLDM's Val&Uniq** (a *chemical* QM9 graph metric) and **upper endpoint 94 %** = **HLTF's own PB-Valid** (a *3D PoseBusters* metric, a different test set) -- two cells of HLTF's Table 1(b), **both `post_processed: false`**; YuelBond's per-bond-type classification F1 92.7 % is a third candidate for the same numeral;
* **reading B** (`sources`.`EF-TALFM`): EF-TALFM arXiv:2609.08333v4 Table 3(a) -- RAW connectivity `C. = 92.98 %`, counted **BEFORE** the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell -- so the frontier arm is **MIXED** under this reading.

Under **BOTH** readings the band is **EXTERNAL**, is **NOT one homogeneous quantity**, and is **NOT on our scale** -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.  Neither reading may be published as the band's settled provenance.  The other members the repo swept -- 92.98 (EF-TALFM connectivity), 94.33 (EF-TALFM yield), 93.13 (EF-TALFM all-PoseBusters-checks), 93.1 (GeoLDM PB-Valid), 93.3 (GCDM Val&Uniq), 92.9 (HLTF Val&Uniq) -- are additional **QUOTED FRONTIER CELLS** of the same heterogeneous family, not settled band endpoints; the EF-TALFM ones are confirmed in its primary record at `src/appendix.tex:82`, `:89`, `:92`, `:115`, `:127`.

### 4.2 Repository layer -- where the band first appears here

A UTF-8-safe scan (repo scanner `molmetal/scripts/wfa156_text_scan.py`, no `os.walk`, no PATH `grep`) over **10,060** files finds the band pattern in **38 working-tree files**. The full census (with the per-file hit lists) is at `molmetal/reports/wfa175_frontier_comparability/band_census.json`.

* **Committed revisions: 0.** Every tracked blob at `HEAD` was read straight out of `git show` and scanned: **0 of 10,060** contain the band. The band exists **only in the 2026-11-01 working tree**.
* **`paper/`: 0 of 134.** 123 tracked files plus 11 gitignored LaTeX build artifacts (`main.pdf`, `main.log`, `main.bbl`, ...) were scanned: **0** carry the band. A separate sweep of the *bare scalar* `92.7` finds 8 files under `paper/`, but every substantive hit is a **different quantity** -- the YuelBond decoder's F1 (`paper/sections/04_evaluation.tex` -> `\paragraph{YuelBond decoder swap: \textsc{available} parallel path (WF-CFM-Rescue, 2026-09-16).}` "bioRxiv 10.1101/2025.05.06.652517, F1=92.7 % on distorted molecules"; `paper/sections/06_limitations.tex` -> `\section{Limitations}\label{sec:limitations}` -> the `\item` "YuelBond decoder swap not wired; CPU + GPU 500-step smokes bit-exact with BondAwareDecoder baseline" "F1=92.7 \%") -- plus build logs and SVG path data. The scalars `92.98`, `94.0`, `94.33` are **0 of 134** in `paper/`.
* **Earliest working-tree renderings, by last-write mtime** (mtime is a LAST-WRITE stamp, so it bounds first introduction from above, not below; the stamps below were **RE-MEASURED on 2026-11-02**, and the stamps this bullet originally carried survive VERBATIM in section F): `TODO/README.md` (2026-11-01T19:13:41) and `TODO/INDEX.md` (2026-11-01T19:13:41) -- each at its **"Last-updated trailer (Wf-A153 census-republish + artifact-path + guard-blind-spot + frontier-comparability wave -- 2026-11-01 -- ...)"** line, a durable TEXT anchor rather than a line number -- both already say the band is **"NOT our band"**; then `metrics/by_metric/wfa155_metal_coordination_sphere.json` (2026-11-01T21:58:52) -> JSON PATH `comparability_vs_frontier_band.not_a_frontier_comparison` and `summary`, whose `comparability_vs_frontier_band.not_a_frontier_comparison` reads *"This is NOT comparable to the 92.7-94 % frontier band -- see comparability_statement for the four independent breaks."* and whose `summary` reads *"This metric is NOT comparable to the published 92.7-94 % frontier band, and no number in this artifact may be presented as comparable to it."*
* **The single mis-attribution, preserved verbatim (§5.5 rule 6).** `molmetal/reports/wfa161_frontier_labelling/report.md` -> SECTION HEADING `## Wf-A176 CORRECTION (2026-11-01) -- §"Comparability breaks" item 2 is WRONG` -> SUB-HEADING `### The superseded sentence (verbatim, line 139 of this file)` (the superseded line number and the 2026-11-01 last-write mtime stamp survive VERBATIM in section F):

  > Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale."

  That rendering is **superseded**, not deleted, by `docs/HONEST_NEGATIVE_CATALOG.md:§6.264` (2026-11-01): *"THE PUBLISHED 92.7-94 % VALIDITY BAND IS **NOT OUR BAND**"* -- and by the Wf-A168 / Wf-A171 audits.
* A sibling branch of this same wave, `molmetal/reports/wfa175_band_provenance/`, covers the band's in-repo provenance in depth (349 lines, with a per-site file:line table and a value-sweep of 92.7 / 92.98 / 94.0 / 94.33). **My §4.2 numbers are an independent re-measurement with a different (broader) regex and a HEAD-blob census**; where the two agree, the number is 0 in `paper/` and 0 at HEAD.

**Conclusion (c):** the band is a frontier composite whose endpoint cells are NOT settled on disk -- the repo holds TWO competing provenance readings (92.7/94.0 under `band_provenance` vs 92.98/94.33 under `sources`.`EF-TALFM`) whose numerals disagree, and under BOTH readings the band is external, is not one homogeneous quantity, and is not on our scale; it entered this repository only in the 2026-11-01 working tree (0 committed renderings), was labelled as the frontier's from its first metric artifact, was briefly mis-labelled "our band" in the Wf-A161 report, and has never reached `paper/` in any rendering.

---

## 5. Answer to (d) -- "89.4" vs "87.9": confirmed independently

**The primary record contains 87.9 %, not 89.4 %.** Re-fetched by this wave; the sha256 is identical to the Wf-A171 and Wf-A168 pins, so the byte stream is stable across three independent fetches:

| string | EF-TALFM e-print (2609.08333v4) | EF-TALFM abs page | count |
|---|---|---|---|
| `89.4` | **0** | **0** | absent |
| `87.9` | **2** | **3** | present |
| `92.98` / `94.33` / `93.13` | 3 / 3 / 2 | 0 / 0 / 0 | present (e-print) |

The 87.9 % is EF-TALFM's **PoseBusters-verified novel yield**, in its own words (`sections/results_unconditional.tex:35`):

> "EF-TALFM achieves the highest PoseBusters-verified novel yield among the evaluated methods, **$87.9\%$**, compared with $75.2\%$ for UAE-3D and $69.6\%$ for FlowMol."

**Where the stray "89.4" comes from -- an adjacent-cell migration, not a rounding.** HLTF's Table 1 carries GeoLDM's **MS (molecule stability) = 89.4 +- 0.5** cell (line 38 in Table 1(a), line 62 in Table 1(b)) in the **same row** as Val 93.6 / Val&Uniq 92.7 / PB-Valid 93.1. A summary that reads one row of that table can carry "89.4" away as if it were a validity figure. The same 89.4 also appears in GeoLDM's own paper (`icml2023.tex:491`, QM9 Mol-Sta 89.4 +- 0.5), so the number is unambiguously GeoLDM's -- and it is a *stability* number, not a validity number. (This also explains the band's endpoints: the same table that supplies 89.4 supplies 92.7 and 94.)

---

## 6. New comparability findings this wave adds (all MEASURED)

1. **A frontier paper prints its own RAW and POST validity side by side** (EQGAT 0.916 -> 0.998) -- a convention directly adoptable in our table.
2. **A frontier paper prints RAW and POST connectivity side by side for five models** (Buttenschoen's `chemical_validity` table) -- and our own post-processing pipeline is the same 3-step recipe, so our ratio is directly comparable to that column.
3. **Frontier RAW connectivity is published as low as 63.37 %** (DiffSBDD, Binding MOAD) -- fragmentation in the raw stream is normal in this literature; ours is simply worse (§3.1).
4. **Frontier "validity" can hide a fragment filter *and* a rejection sampler** (DiffSBDD via NEAT-POCKET's supplementary). Any "% valid" from this literature must be checked for largest-fragment filtering and resampling-until-quota before being placed beside our 8.51 %.
5. **The frontier's geometric validity is 0.4 %-11.1 % RAW** (GenBench3D), rising to 53 %-78 % after a force-field relaxation, and the CrossDocked *training* set itself scores 48.2 %. A 92.7-94 % *chemical*-validity band and a <=11 % *geometric*-validity reality coexist in the same field; quoting the band without naming the axis is the error the band critique exists to prevent.
6. **Validity and connectivity can be 99 % and 0 % in adjacent rows of the same table** (TABASCO at 10-20 steps) -- frontier-internal evidence that the two quantities are not interchangeable.
7. **A frontier paper's metric-definition sentence contradicts its own table** (EF-TALFM `:183` vs `:54`/`:184`, §3.2) -- a new instance of the defect class this repo keeps cataloguing, and a reason to quote connectivity numbers with their stage attached.

---

## 7. Honest framing / what may and may not be said

* **MAY** be said, with these citations: our raw connectivity is 8.509630565203663 % and our largest-fragment post-processed connectivity is 100 % over 12,668 parsed rows (`metrics/by_metric/cfm_fragmentation_postproc_arm.json`), i.e. a raw->post ratio of **11.75139146567718x**; the frontier's largest *printed* connectivity raw->post ratio is GeoLDM's **1.9732620320855614x** (Buttenschoen 2025, arXiv 2505.00518v1), so ours is **5.955312x** larger.
* **MAY** be said: the frontier's largest raw->post ratio on *any* post-processing axis is **133.0x** (GenBench3D, DiffSBDD Validity3D) and ours is a different metric.
* **MUST NOT** be said: that the 92.7-94 % band is ours (it is a frontier composite, §4), that any frontier connectivity number is comparable to ours without stating the stage (§3.2, §6.4), or that our post-processing is comparable to the frontier's without naming the 22.47 % of raw heavy-atom mass it discards (`mass_retention.fraction_of_raw_mass_discarded = 0.22474803194442472`) and the fact that **21.02 % of the survivors are <=2-heavy-atom specks** (`survivor_shape.n_survivors_le_2_heavy = 2663` of 12,668).
* **MUST NOT** be said: that our RAW 8.51 % is competitive with the frontier. It is not -- §3.1.

---

## 8. What this branch could NOT establish (disclosed)

1. **GenBench3D has no LaTeX source** -- its arXiv e-print is a PDF (`%PDF-1.3`, 4,032,287 B); text was extracted with `pdftotext -layout`, so §1/§2.3 cite **PDF page numbers**, not `.tex:line`.
2. **PoseBusters' per-method PB-valid numbers are bar charts**, not text; only the values printed in prose (DiffDock 72 %, TankBind 59 %, Gold 67 %/48 %, Vina 58 %/51 %, Uni-Mol 45 %/23 %) are quoted.
3. **Several classic 3D-generation papers are absent from this host's arXiv index**: title searches for END, Silo, GeoBFN, 3D-EDiffMG, MARS, the original FlowMol and GCDM returned **0 hits** (or unrelated physics/ML papers), and direct abs-page fetches of remembered IDs resolved to unrelated titles. **No primary record was reachable for them**, so no claim is made about them here -- they are neither counted for nor against the frontier ratio. The arXiv API also returned HTTP 400 on boolean queries combining several `%22`-phrases, so candidate discovery was partial by construction.
4. **The frontier's largest connectivity ratio could in principle be higher than 1.9733x** for papers whose RAW and POST connectivity are not both printed -- DiffSBDD's implied 1.57803x is the largest such case found, and its true RAW-to-POST pair is unrecoverable because the POST value is only implied by the pipeline.
5. **First-appearance chronology of the band inside the working tree is not recoverable** from file mtimes alone (mtime = last write); what *is* measured is 0 committed renderings, 0 renderings in `paper/`, and the identity + mtime of the 38 carrying files (`band_census.json`).
6. **No GPU, no decode, no re-run.** Every "ours" number is read from a JSON on disk and re-derived arithmetically; every frontier number is read from fetched primary bytes and pinned by sha256.

---

## 9. Artefacts (all inside `molmetal/reports/wfa175_frontier_comparability/`)

| path | content |
|---|---|
| `report.md` | this file |
| `sources.json` | per-paper records: title, year, venue, url, e-print sha256 + bytes, dataset, metric, value, `raw_or_post`, `evidence_quote`, `locator`, denominator, raw/post values, ratio |
| `band_census.json` | the UTF-8-safe band census: 38 working-tree files, `paper/` 0 of 134, HEAD 0 of 10,060, with per-file line numbers |
| `tools/fetch.py` | live primary-record fetch (proxy urllib, sha256, in-memory tar extraction) |
| `tools/mine.py` | batch fetch + gzipped text cache + per-file grep |
| `tools/scan_all.py` | multi-paper phrase scan over the cache |
| `tools/build.py` | computes the ratios and writes `sources.json` (atomic) |
| `tools/band_census.py` | re-measures the band census incl. the HEAD-blob pass (atomic) |
| `tools/ratios.json` | the computed ratio numbers |
| `cache/` | pinned e-print bytes (sha256 in the JSON side-cars) + extracted text, so every quote above is re-checkable without a re-fetch |

**Nothing outside this directory was written. No file under `paper/`, `docs/`, `AGENTS.md`, `TODO/`, `metrics/` or `molmetal/scripts/` was modified. No git mutation was performed.**

## §C — CORRECTIONS (2026-11-01, Wf-A176)

**Scope and method.** Two defects found by the adversarial branch of this wave (`refutation/refutation.md`) and **independently re-verified by Wf-A176 on 2026-11-01 (CPU-only, GPU 0)**. Per AGENTS.md §5.5 rule #6, **nothing above this line was edited**: each superseded rendering is quoted **VERBATIM** below and corrected additively + dated. Every number in §C is MEASURED from a file on disk or from a live re-run whose command is given inline.

### C.1 — A1: our post-processing is **2 of** Buttenschoen's 3 steps, not 3 of 3

**Superseded rendering — `report.md` -> SECTION HEADING `### 2.1 Connectivity axis (fragment selection)` (the "how POST is obtained" column), VERBATIM:**

> | **OURS** `metrics/by_metric/cfm_fragmentation_postproc_arm.json` | c7 pool, 12,668 parsed rows | **0.08509630565203663** (1078/12668) | **1.0** | **11.75139x** | measured by our own arm; pipeline = `LargestFragmentChooser -> Chem.AddHs -> UFFOptimizeMolecule`, i.e. **the same 3-step PP as Buttenschoen** |

**Superseded rendering — `report.md` -> SECTION HEADING `## 6. New comparability findings this wave adds (all MEASURED)` -> numbered item 2, VERBATIM:**

> 2. **A frontier paper prints RAW and POST connectivity side by side for five models** (Buttenschoen's `chemical_validity` table) -- and our own post-processing pipeline is the same 3-step recipe, so our ratio is directly comparable to that column.

**Correction (MEASURED).** Steps 1-2 of our pipeline *are* Buttenschoen's; **step 3 (UFF) was executed on 0 of 12,668 molecules**. So "the same 3-step PP as Buttenschoen" (line 79) and "the same 3-step recipe, so our ratio is directly comparable to that column" (line 186) are **false as written**.

| evidence | measured value | source |
|---|---|---|
| UFF ok rate over the post-processed pool | **0.0** | `metrics/by_metric/cfm_fragmentation_postproc_arm.json` → `/postprocessed_axis/uff_ok_rate_over_postprocessed` |
| UFF failure census | **`{"ValueError: Bad Conformer Id": 12668}`** | same file → `/uff_reasons` |
| the denominator that 12668 refers to | `n_parsed = 12668`, `n_postprocessed = 12668` | same file → `/denominators` |
| the artifact's own note, verbatim | "the published pipeline's third step (UFF) has NO parameters for the metals in this corpus, so it cannot be applied verbatim to metal complexes; see uff_reasons" | same file → `/uff_note` |
| second artifact, same pool | `uff_ok = 0`, `uff_failed = 12668`, `uff_ok_rate_over_postprocessed = 0.0`, note: "the pool is SMILES-derived and carries no conformer, and UFF has no parameters for the metals; **the arm therefore measures steps 1-2 only**" | `molmetal/reports/wfa168_falsevalid_axis/falsevalid_axis.json` → `/uff` |
| a repo test that ASSERTS this | `uff_ok_rate_over_postprocessed == 0.0`; `uff_failed == n_parsed`; `uff_reasons` keys == `["ValueError: Bad Conformer Id"]` | `molmetal/tests/test_wfa168_falsevalid_axis.py:275-279` (`test_uff_step_is_inapplicable_on_this_pool`) |
| repo code that already says so | "so 'we applied the frontier recipe' is true for steps 1-2 and FALSE for step 3" | `molmetal/scripts/wfa158_frontier_postproc_matrix.py:426-428` |
| repo code that already says so | "so `uff_ok_rate = ...` -- the arm measures steps 1-2 only" | `molmetal/scripts/wfa168_falsevalid_axis.py:780-781` |

**Independent re-run by Wf-A176 (this wave, CPU-only, no GPU call).** The three steps were re-executed on the first 300 rows of the arm's own source pool `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv` (12,669 rows in the file; the arm's `/denominators/n_rows_in_csv = 12669`):

```
CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python - <<'EOF'
import csv, collections
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
rows = list(csv.reader(open('/home/hugo/codes/mol-metal/molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv')))
chooser = rdMolStandardize.LargestFragmentChooser(); reasons = collections.Counter(); ok = 0; n = 0
for r in rows[1:301]:
    m = Chem.MolFromSmiles(r[1])
    if m is None: continue
    try:
        m2 = Chem.AddHs(chooser.choose(m)); n += 1
        AllChem.UFFOptimizeMolecule(m2, maxIters=200); ok += 1
    except Exception as e:
        reasons[f'{type(e).__name__}: {e}'] += 1
print('uff_attempted', n, 'uff_ok', ok, dict(reasons))
EOF
```

**MEASURED result: `uff_attempted 300 uff_ok 0 {'ValueError: Bad Conformer Id': 300}`.** Root cause re-measured on the same run: `m2.GetNumConformers() == 0` after `Chem.AddHs` — the pool is SMILES-derived and carries no conformer, and UFF has no parameters for the metal centres (`[Pt]`, `[Ru]`, ...). The failure mode reproduces the artifact's 12,668-row census exactly, on the artifact's own pool.

**What is NOT affected, and why.** The connectivity rate is fixed by **step 1**: `LargestFragmentChooser` keeps exactly one fragment by construction. Measured on the artifact: `postprocessed_axis.connected_rate_over_postprocessed = 1.0`, `n_connected = 12668`, `n_multicomponent_after_pp = 0`, `n_pp_failed = 0`; and in the Wf-A176 re-run, **300/300** post-PP molecules had exactly one fragment. Steps 2 (`Chem.AddHs`) and 3 (UFF) are graph-preserving — AddHs adds hydrogens, UFF is a geometry refinement of an existing conformer — and neither can change connectivity. Therefore:

* the RAW→POST connectivity ratio **11.75139146567718x** is **UNAFFECTED** (already carried by `tools/ratios.json` → `/OUR_RATIO` and `sources.json` → `/computed_ratios/our_ratio`);
* the frontier comparison is **UNAFFECTED**: `11.75139146567718 / 1.9732620320855614 = 5.955312206183287` (GeoLDM 37.4 % → 73.8 %, Buttenschoen 2505.00518v1, `tables/chemical_validity.tex:17` and `:23`) — neither number in that quotient is in dispute.

**What IS affected.** Only the *pipeline-identity* claim. Our POST column is **not** the same pipeline as Buttenschoen's "+PP" column: it is that pipeline **minus its third step**. The *connectivity* comparison survives for the reason given above, but this report may not assert pipeline identity and may not describe our arm as "the same 3-step PP". No comparison in this report depends on step 3 (§2.1 uses the connectivity column only; §2.2 and §2.3 are other papers' own printed pairs). The §7 disclosures — **22.474803194442472 %** of the raw heavy-atom mass discarded (`/mass_retention/fraction_of_raw_mass_discarded`) and **2,663 of 12,668** survivors (2663/12668 = 21.021471424060625 %) being ≤ 2-heavy-atom specks (`/survivor_shape/n_survivors_le_2_heavy`) — are unaffected and remain mandatory beside any use of the 100 % figure.

### C.2 — A6: five cells of §2.1's ratio column contradict the report's own `tools/ratios.json`

**Superseded renderings — `report.md` §2.1, VERBATIM (5 rows, each with its line number):**

> | Buttenschoen 2505.00518 | EQGAT-diff | 84.4 % | 87.1 % | 1.03175x | idem |
> | DiffSBDD 2210.13695v3 | CrossDocked / DiffSBDD-inpaint | 67.52 % | 100 % | 1.48071x | idem |
> | DiffSBDD 2210.13695v3 | Binding MOAD / DiffSBDD-cond (C-alpha) | 77.38 % | 100 % | 1.29233x | idem |
> | DiffSBDD 2210.13695v3 | CrossDocked / DiffSBDD-cond | 78.27 % | 100 % | 1.27766x | idem |
> | EQGAT 2309.17296v2 | Connect. Comp., GEOM-Drugs (w_u) | 0.8657 | 1.0 | 1.15501x | implied by fragment selection |

**Correction (MEASURED — each ratio recomputed from the RAW and POST values printed in the same row):**

| line | cell | printed | recomputed from the row's own RAW / POST | correct 5-dp value | abs. delta | `tools/ratios.json` value |
|---|---|---|---|---|---|---|
| 66 | Buttenschoen / EQGAT-diff | 1.03175x | 87.1 / 84.4 = **1.0319905213270142** | **1.03199x** | 2.405e-04 | `/BB_RATIOS/EQGAT-diff` = 1.0319905213270142 |
| 69 | DiffSBDD / CrossDocked / inpaint | 1.48071x | 100 / 67.52 = **1.481042654028436** | **1.48104x** | 3.327e-04 | `/DSBDD_RATIOS/CrossDocked / DiffSBDD-inpaint (full pocket)` = 1.481042654028436 |
| 71 | DiffSBDD / Binding MOAD / cond (C-alpha) | 1.29233x | 100 / 77.38 = **1.2923235978288965** | **1.29232x** | 6.402e-06 | `/DSBDD_RATIOS/Binding MOAD / DiffSBDD-cond (C_alpha)` = 1.2923235978288965 |
| 72 | DiffSBDD / CrossDocked / cond | 1.27766x | 100 / 78.27 = **1.2776287210936503** | **1.27763x** | 3.128e-05 | `/DSBDD_RATIOS/CrossDocked / DiffSBDD-cond (full pocket)` = 1.2776287210936503 |
| 76 | EQGAT / Connect. Comp., GEOM-Drugs (w_u) | 1.15501x | 1.0 / 0.8657 = **1.1551345731777751** | **1.15513x** | 1.246e-04 | **no ratio key** in `tools/ratios.json` (only the raw map `/EQGAT_CONN`) — the recomputation above is the reference |

**Everything else in §2.1 re-derives EXACTLY.** The other **12** cells of §2.1 (lines 63, 64, 65, 67, 68, 70, 73, 74, 75, 77, 78, 79) re-derive from their own RAW / POST values; the 9 of them that carry a ratio key in `tools/ratios.json` also agree with it, and `/OUR_RATIO = 11.75139146567718` matches line 79. §2.2 and §2.3 re-derive exactly too: 0.998/0.916 = 1.0895196506550218; 0.532/0.004 = 133.0; 0.551/0.023 = 23.956521739130437; 0.783/0.094 = 8.329787234042554; 0.703/0.099 = 7.1010101010101; 0.687/0.111 = 6.1891891891891895; 0.544/0.106 = 5.132075471698114.

**Agreement / disagreement with the refutation branch.** The refutation named **4** cells (lines 66, 69, 72, 76). **Wf-A176's independent recomputation AGREES with the refutation on all 4**, to the identical full-precision values (1.0319905213270142, 1.481042654028436, 1.2776287210936503, 1.1551345731777751). **The refutation's list is INCOMPLETE: line 71 is a 5th mismatching cell that it did not flag** (printed 1.29233x vs the recomputed 1.2923235978288965 → 1.29232x). All 5 mismatches are in the 4th-5th decimal.

**Nothing load-bearing changes.** The frontier maximum used in §3 and §7 is GeoLDM's **1.9732620320855614x** (line 63 — correct as printed), and the next-largest implied ratio is DiffSBDD-cond / Binding MOAD's **1.5780337699226763x** (line 68 — correct as printed). Neither is among the 5. The §3 / §7 headline quotient `11.75139146567718 / 1.9732620320855614 = 5.955312206183287` is therefore **UNCHANGED**. These are typographic defects in a derived column, not defects in any claim the report makes.

---

**§C close-out.** A1 and A6 are the two defects Wf-A176 was asked to correct; both are corrected above, additively and dated, with every superseded rendering preserved verbatim. **Deliberately NOT changed:** the §2.1 table in place (rule #6 forbids rewriting a superseded rendering), §3 / §7's headline numbers (unaffected by both defects), `tools/ratios.json`, `band_census.json`, `sources.json`, and every file outside `molmetal/reports/wfa175_frontier_comparability/`. The refutation's other findings (claims 2b, 3, 4, 5 and the DecompDiff `n/a` verdict) are outside this wave's scope and are neither adopted nor contradicted here.

## Wf-A176 CORRECTION 3 (2026-11-01) — the §C.1 root-cause clause "UFF has no parameters for the metal centres" is FALSE

**This section is APPEND-ONLY. Everything above it is preserved VERBATIM per AGENTS.md §5.5 rule #6 and is NOT edited.**

### What is superseded

§C.1 above presents, under the heading *"Root cause re-measured on the same run"*:

> UFF has no parameters for the metal centres (`[Pt]`, `[Ru]`, …)

**That clause was NOT re-measured, and it is FALSE as stated.** It was carried over from the artefact's own `uff_note` (quoted there as verbatim), which is where the error entered.

### MEASURED refutation

* The ONLY observed error is `ValueError: Bad Conformer Id` — i.e. a MISSING CONFORMER, not a missing force-field parameter.
* `AllChem.UFFOptimizeMolecule` SUCCEEDS on metal-bearing molecules once a conformer exists: measured on `[Pt](Cl)(Cl)N` and `[Ir](Cl)(Cl)N`.
* The repo's own independent verification records the same: `molmetal/reports/wfa168_falsevalid_axis/falsevalid_axis.json` → `/uff` (uff_ok 0, uff_failed 12668) and the catalogue's own defect **D2** (`molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md`, defects table): *"UFF has no parameters for the metals" is **false**; measured cause is the missing conformer (UFF succeeds 60/60 metal-bearing + 4/4 metal-free after ETKDGv3 embedding)*.
* The measured root cause: after `Chem.AddHs`, `m2.GetNumConformers() == 0` — the pool is SMILES-derived and carries no conformers, so step 3 of the frontier recipe has nothing to minimise.

### The corrected clause

> **Root cause (re-measured 2026-11-01 on the same pool).** UFF ran on 0 of 12,668 molecules with the single error `ValueError: Bad Conformer Id` (12,668). The cause is a **missing conformer**, not a missing force-field parameter: the pool is SMILES-derived, so after `Chem.AddHs` the molecule carries 0 conformers and there is nothing to minimise; UFF itself succeeds on metal-bearing molecules once a conformer exists (`[Pt](Cl)(Cl)N`, `[Ir](Cl)(Cl)N` — MEASURED). Our post-processing therefore executes **steps 1-2 of the frontier recipe only**.

### What is NOT changed by this correction

* The §C.1 section above stays VERBATIM.
* No measured number moves: `uff_attempted 300`, `uff_ok 0`, `uff_failed 12668`, `uff_ok_rate_over_postprocessed = 0.0`, and the connectivity rate `connected_rate_over_postprocessed = 1.0` all stand exactly as measured.
* The headline ratios `11.75139146567718x` and `5.955312206183287x` are UNAFFECTED.
* `paper/**`, `metrics/**`, `AGENTS.md`, `TODO/**`, `docs/HONEST_NEGATIVE_CATALOG.md` are NOT touched.
* CPU-only, 0 GPU.

---

## §D — Wf-A180 CORRECTION (2026-11-01) — the band's ENDPOINTS are NOT settled on disk

**Date 2026-11-01.  Wave Wf-A180.  CPU-only, 0 GPU.  This section is ADDITIVE: no
line above (lines 1-350) was deleted or rewritten (AGENTS.md §5.5 rule #6).**

### §D.1 SUPERSEDED (2026-11-01, Wf-A180)

The three fenced blocks below are §4.1's heading, the "So the band is minted here"
passage, and §4's Conclusion (c) — verbatim, exactly as they stood before this
correction.  Each is still present, unmodified, at the lines named; they are quoted
here so the superseded renderings are preserved in a dated block, and they are NOT
deleted.

```
### 4.1 Source layer -- the band's endpoints are two cells of one table in HLTF
```

```
So the band is minted here:

* **lower endpoint 92.7 %** = **GeoLDM's Val&Uniq** (a *chemical* QM9 graph metric);
* **upper endpoint 94 %** = **HLTF's own PB-Valid** (a *3D PoseBusters* metric, a different test set).

They are **different models, different metrics, different axes, in one table**. That is the whole defect in a single screenshot. The other members the repo swept -- 92.98 (EF-TALFM connectivity), 94.33 (EF-TALFM yield), 93.13 (EF-TALFM all-PoseBusters-checks), 93.1 (GeoLDM PB-Valid), 93.3 (GCDM Val&Uniq), 92.9 (HLTF Val&Uniq) -- are additional cells of the same heterogeneous family, and the EF-TALFM ones are confirmed in its primary record at `src/appendix.tex:82`, `:89`, `:92`, `:115`, `:127`.
```

```
**Conclusion (c):** the band is a frontier composite whose two endpoints are different metrics of different models in one table (HLTF Table 1(b)); it entered this repository only in the 2026-11-01 working tree (0 committed renderings), was labelled as the frontier's from its first metric artifact, was briefly mis-labelled "our band" in the Wf-A161 report, and has never reached `paper/` in any rendering.
```

**Why they are superseded.**  Each of the three names ONE provenance reading as if it
were settled.  The published band's provenance is NOT settled on disk: the frontier
post-processing matrix record (`frontier_matrix.json` =
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`) holds TWO
competing readings of which frontier cells the band names — the first under
`band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both
`post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a
third candidate for the same numeral) and the second under `sources`.`EF-TALFM`
(EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted
BEFORE the largest-component retention, with the post-processed
`N. & U. & S. = 94.33 %` companion cell).  They do not even agree on the numerals
(92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.
Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT
on our scale — OUR 8.51 % IS a connected-component count on the RAW emitted SMILES,
and nothing else.

### §D.2 The corrected statement

> **The band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale —
> OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing
> else.  Which cells it names is NOT settled on disk.**  The repo holds TWO competing
> provenance readings: `band_provenance` (GeoLDM Validity & Uniq 92.7 % + HLTF
> PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type
> classification F1 92.7 % as a third candidate for the same numeral) and
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a) RAW connectivity
> `C. = 92.98 %`, counted BEFORE the largest-component retention, with the
> post-processed `N. & U. & S. = 94.33 %` companion cell).  The two do not even agree
> on the numerals (92.7/94.0 vs 92.98/94.33), so neither may be published as the
> settled provenance of the band.  The HLTF Table 1(b) transcription at §4.1 lines
> 129-138 is reproduced from the primary source and stands as printed; what is
> withdrawn is only the claim that its two cells ARE the band's endpoints.

### §D.3 What this correction does and does not change

* **UNCHANGED — §4.1's transcription of HLTF Table 1(b) (lines 127-138).**  It is a
  verbatim reading of the primary record and is unaffected by which provenance
  reading is right.
* **UNCHANGED — §4.2 (repository layer).**  The 10,060-file UTF-8-safe census, the
  0-of-10,060 HEAD-blob count, the 0-of-134 `paper/` count, the earliest-working-tree
  mtimes and the Wf-A161 mis-attribution record all stand as measured.
* **UNCHANGED — §4's headline.**  "It is a FRONTIER number. It is not ours."  That is
  invariant (i) and survives either reading.  So does the §7 "MUST NOT" list.
* **WITHDRAWN as unsettled — the ENDPOINT claim.**  That the band's lower endpoint is
  92.7 % (GeoLDM's Val&Uniq), that its upper endpoint is 94 % (HLTF's PB-Valid), that
  the band's endpoints "are two cells of one table in HLTF", that 92.98 / 94.33 /
  93.13 / 93.1 / 93.3 / 92.9 "are additional cells of the same heterogeneous family"
  *of the band*, and that the band's endpoints are "different metrics of different
  models in one table (HLTF Table 1(b))".
* **DELIBERATELY NOT ASSERTED.**  (a) That "neither endpoint is a connected-component
  count" / "the band is not a connectivity band" — TRUE under the first reading, FALSE
  under the second.  (b) That EF-TALFM's connectivity is computed AFTER retaining the
  largest component — REFUTED on disk: its 92.98 % connectivity is a RAW
  single-component rate counted BEFORE the retention (see
  `molmetal/reports/wfa168_eftalfm_axis/report.md`, and
  `molmetal/reports/wfa171_frontier_postproc/report.md` §1, which measures 92.98 % on
  the **pre-retention** graph).  (c) That the band is measured on a published
  POST-PROCESSED stream — under the first reading both band endpoints carry
  `post_processed: false`; the second reading is MIXED.
* **Method / scope.**  Both readings are read from `frontier_matrix.json` on disk (JSON
  paths `band_provenance` and `sources`.`EF-TALFM`; NOT line numbers, and NOT the token
  `frontier_axis_mapping`, which is not a key of that record).  No fetch, no GPU, no
  decode, no re-measurement.


---

## §E — Wf-A194 CORRECTION (2026-11-02) — the LIVE §4.1 body and Conclusion (c) no longer assert ONE reading as settled

**Date 2026-11-02.  Wave Wf-A194.  CPU-only, 0 GPU.  This section is ADDITIVE: no
line above was deleted, and no earlier dated section (§C, §D) was rewritten in place
(AGENTS.md §5.5 rule #6).**

### §E.1 What changed

* **§4.1's heading** (live) is now the two-reading heading — it no longer names the
  HLTF table as the source of the band's endpoints.
* **§4.1's body** (live) is now the reading-safe two-reading rendering, reproduced
  byte-for-byte from
  `molmetal/reports/wfa175_frontier_comparability/sources.json` →
  `live_section_4_1_reading_safe_2026_11_02_wfa194` → `live_section_4_1_rendering`.
* **§4's Conclusion (c)** (live) is now the reading-safe conclusion, reproduced
  byte-for-byte from the same record → `live_conclusion_c_rendering`.
* **UNCHANGED — §4.1's transcription of HLTF Table 1(b)** (the table itself, still at
  lines 127-138).  It is a verbatim reading of the primary record and is unaffected by
  which provenance reading is right.  No numeral moved: the live body still carries
  92.7, 94, 92.98, 94.33, 93.13, 93.1, 93.3, 92.9 and 8.51, every one of them already
  on disk in this report or in the frontier post-processing matrix record.
* **UNCHANGED — §D (2026-11-01, Wf-A180) in its entirety.**  §D.1's three fenced
  blocks remain the VERBATIM preservation of the three superseded renderings, and
  §D.2 / §D.3 remain as written.

### §E.2 The one sentence in §D.1 that this wave supersedes

§D.1 opens: *"Each is still present, unmodified, at the lines named; they are quoted
here so the superseded renderings are preserved in a dated block, and they are NOT
deleted."*  As of 2026-11-02 that sentence is **SUPERSEDED for the three live
positions it describes**: §4.1's heading, §4.1's body and §4's Conclusion (c) now
carry the reading-safe renderings and are no longer "present, unmodified" at those
lines.  **The preservation requirement is still met**: the three superseded
renderings survive VERBATIM in §D.1's three fenced blocks, which are unchanged, and
the live replacements are recorded in
`molmetal/reports/wfa175_frontier_comparability/sources.json` →
`live_section_4_1_reading_safe_2026_11_02_wfa194` →
`superseded_rendering_preserved_verbatim_at`.  Per AGENTS.md §5.5 rule #6, §D.1
itself was **not** rewritten in place — this §E records the supersession additively.

### §E.3 Producer parity

`report.md` in this branch is **hand-authored**: no script in the repo writes it
(MEASURED 2026-11-02 — `git grep --text "wfa175_frontier_comparability/report.md"`
over `*.py` returns no hit, and `tools/build.py` writes `sources.json` and
`tools/ratios.json` only).  Its producer-parity anchor is therefore the machine
field: the live §4.1 heading / body and the live Conclusion (c) are checked for
byte-equality against `live_section_4_1_heading` / `live_section_4_1_rendering` /
`live_conclusion_c_rendering` in `molmetal/tests/test_wfa192_live_x_closure.py`.
`sources.json` itself IS producer-emitted: `tools/build.py` gained the dated record
`live_section_4_1_reading_safe_2026_11_02_wfa194`, and a TEMP-dir regeneration
(2026-11-02) reproduces the published `sources.json` byte-for-byte apart from that
one added key.

### §E.4 Not asserted

Unchanged from §D.3(c): the three readings §D.3 deliberately declined to assert are
still deliberately not asserted, because each is TRUE under one of the two competing
readings and FALSE under the other.  §D.3(c) is the verbatim record; it is neither
restated nor rewritten here.
---

## §F -- Wf-A200 CROSS-WAVE LOCATOR REPAIR (2026-11-02) -- every LIVE `<path>:<line>` locator in this branch is demoted to a durable locator

**The defect class.**  A document in one wave's tree cited a LINE NUMBER inside
another wave's artefact.  The cited tree was re-rendered, so the line number
silently started pointing at a DIFFERENT sentence while the citation still
looked plausible.  This branch's own producer already said so -- the Wf-A161
generator's `X3` clause, read on disk 2026-11-02 at
`molmetal/scripts/wfa161_frontier_labelling.py`:

> * **X3 — FORBIDDEN as a locator:** line-number locators (`:51-89`, `:360, :378-379`).  Line numbers rot as the record gains keys; cite the JSON PATHS `band_provenance` and `sources`.`EF-TALFM`.

Four such locators were LIVE in this branch on 2026-11-02.  All four are
repaired in the LIVE body, and every superseded rendering survives VERBATIM in
this DATED section, per AGENTS.md §5.5 rule #6 (append-only + dated: nothing was
deleted and no line number was silently rewritten).

### §F.1 The four LIVE defects and the durable locator that replaced each

| # | live site | the perishable locator | what the cited line NOW reads | the durable locator |
|---|---|---|---|---|
| D1 | `report.md` live §4.2 | `TODO/README.md:3` | the **Wf-A172 dead-gate history** trailer -- a DIFFERENT sentence | `TODO/README.md` -> the "Last-updated trailer (Wf-A153 census-republish + artifact-path + guard-blind-spot + frontier-comparability wave -- 2026-11-01 -- ...)" line (a TEXT anchor) |
| D2 | `report.md` live §4.2 | `TODO/INDEX.md:3` | the **Wf-A172 dead-gate history** trailer -- a DIFFERENT sentence | `TODO/INDEX.md` -> the same Wf-A153 trailer line (a TEXT anchor) |
| D3 | `report.md` live §4.2 | `molmetal/reports/wfa161_frontier_labelling/report.md:139` | the REPAIRED "Comparability breaks" item 2 (the two-reading text) -- a DIFFERENT sentence | `molmetal/reports/wfa161_frontier_labelling/report.md` -> SECTION HEADING `## Wf-A176 CORRECTION (2026-11-01) -- §"Comparability breaks" item 2 is WRONG` -> SUB-HEADING `### The superseded sentence (verbatim, line 139 of this file)` |
| D4 | `refutation/refutation.md` claim-3 row + `refutation/refutation.json` | `molmetal/scripts/wfa161_frontier_labelling.py:934` | an unrelated sentence; the mis-attribution string is at line **913** | `molmetal/scripts/wfa161_frontier_labelling.py` -> PYTHON CONSTANT `WFA176_CORRECTION_SECTION` -> its `> 2. **Post-processing.** ...` line |

### §F.2 What each cited line NOW reads -- MEASURED 2026-11-02 on disk

* `TODO/README.md` line 3 and `TODO/INDEX.md` line 3 both open:

  ```text
  **Last-updated trailer (Wf-A172 dead-gate history CORRECTION + pre-flight live-state + config-gate-requirement wave — 2026-11-01 — the 24 pre-training gates were removed **ON PURPOSE** by th ...
  ```

  The "NOT our band" rendering lives on the NEXT line (line 4), which opens:

  ```text
  **Last-updated trailer (Wf-A153 census-republish + artifact-path + guard-blind-spot + frontier-comparability wave — 2026-11-01 — the census summary was republished a **THIRD** time, every ar ...
  ```

* `molmetal/reports/wfa161_frontier_labelling/report.md` line 139 is item 2 of
  the LIVE `## Comparability breaks against the frontier convention` section:

  ```text
  2. **Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDKit sanitization: EF-TALFM counts connectivity BEFORE the largest-component retention (its  ...
  ```

  The mis-attribution survives at line 158, inside the DATED `## Wf-A176
  CORRECTION (2026-11-01)` section whose heading is at line 151 and whose
  sub-heading is at line 156:

  ```text
  ## Wf-A176 CORRECTION (2026-11-01) — §"Comparability breaks" item 2 is WRONG
  ### The superseded sentence (verbatim, line 139 of this file)
  > 2. **Post-processing.** The frontier's connectivity rates are computed on the atom-count largest connected component after RDKit sanitization (EF-TALFM) or with no stated fragment handling at all (GEOM-DD, GGFlow). Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale.
  ```

* `molmetal/scripts/wfa161_frontier_labelling.py` line 934 is:

  ```text
       validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
  ```

  and the mis-attribution string is at line **913**, inside the python constant
  `WFA176_CORRECTION_SECTION` (declared at line 905):

  ```text
  WFA176_CORRECTION_SECTION = """\
  > 2. **Post-processing.** The frontier's connectivity rates are computed on the atom-count largest connected component after RDKit sanitization (EF-TALFM) or with no stated fragment handling at all (GEOM-DD, GGFlow). Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale.
  ```

### §F.3 The superseded LIVE renderings, preserved VERBATIM (§5.5 rule #6)

**`report.md` live §4.2, the bullet that carried the two `TODO/` line-number
locators (superseded on 2026-11-02; the file mtime stamps it carried are
themselves superseded -- both TODO files were re-written later on 2026-11-01):**

```text
* **Earliest working-tree renderings, by last-write mtime** (mtime is a LAST-WRITE stamp, so it bounds first introduction from above, not below): `TODO/README.md:3` (2026-11-01T00:27:56) and `TODO/INDEX.md:3` (01:09:29) -- both already say the band is **"NOT our band"**; then `metrics/by_metric/wfa155_metal_coordination_sphere.json` (00:56:17), whose `comparability_vs_frontier_band.not_a_frontier_comparison` reads *"This is NOT comparable to the 92.7-94 % frontier band -- see comparability_statement for the four independent breaks."* and whose `summary` reads *"This metric is NOT comparable to the published 92.7-94 % frontier band, and no number in this artifact may be presented as comparable to it."*
```

**`report.md` live §4.2, the mis-attribution pointer (line 139 of the Wf-A161
report, and that file's last-write mtime at the time):**

```text
* **The single mis-attribution, preserved verbatim (§5.5 rule 6).** `molmetal/reports/wfa161_frontier_labelling/report.md:139` (mtime 2026-11-01T01:59:40):
```

**`report.md` live §4.2, the mis-attribution quote.  The rendering below is
preserved VERBATIM as it stood; note that it inserted `**` emphasis markers
that the cited source line does NOT carry, and wrapped the excerpt in a pair
of `"` quotes -- the LIVE bullet now quotes the source excerpt byte-exactly.
The cited source line itself (the Wf-A161 report's line 158) is:

```text
  > "**Our 92.7-94 % band** comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale."
```

**`report.md` section C.1, the two self-locators (both pointed at this same
file's own live body, so they rot on every edit above them):**

```text
**Superseded rendering — `report.md:79` (§2.1, "how POST is obtained" column), VERBATIM:**
**Superseded rendering — `report.md:186` (§6, finding 2), VERBATIM:**
```

### §F.4 The LAST-WRITE mtime stamps, as measured

| file | mtime measured 2026-11-02 | the stamp this bullet carried before |
|---|---|---|
| `TODO/README.md` | 2026-11-01T19:13:41 | 2026-11-01T00:27:56 |
| `TODO/INDEX.md` | 2026-11-01T19:13:41 | 2026-11-01T01:09:29 |
| `metrics/by_metric/wfa155_metal_coordination_sphere.json` | 2026-11-01T21:58:52 | 2026-11-01T00:56:17 |

The two TODO files are now TIED to the second, so their original sub-ordering
(README before INDEX) is no longer resolvable from the stamps; the ordering
that still holds is both TODO files before the metric artefact.

**MEASURED 2026-11-02T03:07 -- ALL THREE artefacts have since been re-rendered,
so every stamp in the table above is now STALE BY CONSTRUCTION.**  Their
last-write stamps at this measurement are:

| file | last-write stamp measured 2026-11-02T03:07 |
|---|---|
| `TODO/README.md` | 2026-11-02T02:33:14.442940463 +0800 |
| `TODO/INDEX.md` | 2026-11-02T02:33:14.120165947 +0800 |
| `metrics/by_metric/wfa155_metal_coordination_sphere.json` | 2026-11-02T02:44:17.622802867 +0800 |

The two TODO files are STILL tied to the same second (02:33:14), but their
sub-second stamps now put `TODO/INDEX.md` **BEFORE** `TODO/README.md` -- the
OPPOSITE of the order the table records ("README before INDEX").  So the
sub-ordering this section called unresolvable is now resolved, and in the
other direction.

This is the defect class this wave repairs, observed on the record itself: a
stamp inside a dated section is STALE BY CONSTRUCTION, and the durable locator
is the JSON PATH or the SECTION ANCHOR, never the stamp.  Nothing above this
note was edited -- the stale rendering is preserved VERBATIM.

### §F.5 What this section does NOT do

* It does NOT re-run the band census.  The "38 working-tree files" and
  "10,060 files" figures in §4.2 are Wf-A175's own 2026-11-01 measurements;
  the refutation in `refutation/refutation.md` already records that the census
  excluded gitignored paths repo-wide and that a re-run now yields 39 files.
  That is a NUMER staleness, not a locator defect, and it is left for the
  branch that owns the census.
* It does NOT touch `paper/**`, `docs/**`, `TODO/**`, `AGENTS.md`, `registry/**`,
  `metrics/**`, or any sibling wave's tree.  The defects found in those trees
  are REPORTED, not fixed.
* It does NOT change any NUMERAL of the band, of the census, or of the
  comparability readings.

## §G -- Wf-A200f (2026-11-02) -- the numeral 94.0 is NOT unique to HLTF, and
the branch's re-measured live-drift count

### §G.1 A MEASURED counter-example to this branch's own primary-source record

The dated record `sources.json` ->
`crosswave_locator_repair_2026_11_02_wfa200` ->
`primary_source_reverification_2026_11_02` -> `cells_measured`["94.0"] says:

```text
the numeral 94.0 appears LITERALLY only in HLTF's abstract (iclr2025_conference.tex:55, 'improving PoseBusters validity to 94.0%'); the table cell prints '94\pm0.5' (qm9_benchmarks_og.tex:66, row 'PB-Valid (%)', HLTF column).  Same cell, two renderings.
```

Read over the frontier corpus this branch pins, "LITERALLY only" is **FALSE**.
MEASURED on disk on 2026-11-02 with the repo's own reader
(`refutation/tools/show.py` over the pinned `cache/2609.05097v1.txt.gz`;
CPU-only, no network call):

```text
NEAT-POCKET arXiv 2609.05097v1  sections/results.tex:23   CrossDocked & 94.0 $\pm$ 4.7
NEAT-POCKET arXiv 2609.05097v1  sections/results.tex:126  Overall, 94.0\% of the CrossDocked test-set ligands are PB valid.
```

e-print sha256 `bd978841f62a7d75e8635714e7336a6e3abec39b854ec3f2c90974890f2b7fdb`,
10,632,429 bytes; tarball member `sections/results.tex` 40,202 bytes; ON-DISK
extracted text 40,208 bytes, sha256
`d9d65a1db56b9c9ed5a8ad2bd48c0623a5a281377f381299b36d79129ef96e5d`.

**The corrected rendering is SCOPED** -- WITHIN HLTF the numeral 94.0 appears
literally only in the abstract; OVER THE FRONTIER CORPUS it is not unique to
HLTF.  The superseded sentence is preserved VERBATIM in `sources.json` ->
`primary_source_reverification_correction_2026_11_02_wfa200f` ->
`superseded_rendering_verbatim` and was NOT rewritten (AGENTS.md 5.5 rule #6).

### §G.2 What this does NOT change

* NO reading of the band.  NEAT-POCKET's 94.0 is a DIFFERENT quantity -- the
  PB-validity of the CrossDocked TEST-SET REFERENCE LIGANDS (real docked
  molecules), on a DIFFERENT test set -- so it is a THIRD candidate for the
  numeral 94.0, listed next to the branch's existing third candidate for 92.7
  (YuelBond's per-bond-type classification F1 92.7 %).
* Readings A (92.7 / 94.0) and B (92.98 / 94.33) still disagree on the
  numerals; the band's endpoint cells are still NOT settled on disk.
* The verdicts: REFUTED 0 / SURVIVED 3 / IMPRECISE 3, unchanged.
* No NUMERAL of this report changed.  The live §4.1 body and Conclusion (c)
  above are untouched.

### §G.3 The branch's live-drift count, re-measured after this correction

| class | count |
|---|---|
| `RESOLVES-BUT-DIFFERS` (live drift) | **0** |
| `RESOLVES-NO-CLAIM` | 2 |
| `TARGET-IS-SELF` | 2 |
| `DATED-SUPERSESSION` | 100 |
| `SNAPSHOT` | 152 |
| `RESOLVES-AND-MATCHES` / `TARGET-ABSENT` / `TARGET-AMBIGUOUS` | 0 |

64 files in the census, 38 scanned, 256 locator hits, wall 1.62 s
(`wfa200_crosswave_locator_audit.py --paths
molmetal/reports/wfa175_frontier_comparability`).  The counts moved from the
Wf-A200e measurement (226 hits, 70 DATED-SUPERSESSION) because the Wf-A200f
dated key and the DATED sections that carry it add 30 line-number
locators, ALL of which classify as
`DATED-SUPERSESSION`.  The live drift is still **0**.
