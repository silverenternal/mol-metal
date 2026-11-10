# Wf-A161 — Frontier labelling convention for connectivity / validity percentages

Generated: 2026-11-02T01:20:17+0800

**Question.** When a frontier generative-molecular paper prints a connectivity / validity percentage, what qualifier does it print *beside* the number — and is the population (N), the post-processing (largest-component retention? relaxation?) and the definition (per-molecule? per-pool?) stated AT the number, or only in a distant table caption / appendix subsection?

**Method.** arXiv e-print LaTeX source fetched per paper (URL, HTTP status and sha256 recorded in `sources.json`), split into LaTeX float environments (caption included) and running-text paragraphs, and each metric-bearing site classified for four qualifier classes (population / post-processing / definition / hydrogen convention). The verdict is DERIVED from the machine-measured evidence, not hand-written.

## Headline finding

**The frontier convention is NOT 'population + post-processing + definition AT the number'.** Of 5 frontier papers fetched, 1 print all three at the number, 3 qualify it in a DIFFERENT place or a DIFFERENT form, and 1 print no such metric at all.

Concretely: the population (N) is usually a TABLE-CAPTION fact; the post-processing is usually an APPENDIX fact; and the DEFINITION is usually a METRICS-SUBSECTION fact. The one paper that prints all three together does it in the caption of the table that carries the number (EF-TALFM) — never in the running text. A reader who copies '92.98 %' out of the body text of EF-TALFM gets a number with no population, no post-processing and no definition attached to it.

**Therefore our house format string must be STRICTER than the frontier convention, not merely equal to it.** The frontier will not disambiguate our numbers for us; we have to print the population, the post-processing, the denominator and the definition next to the number ourselves.

## Comparison table

| Paper | Metric reported | N stated AT the number? | Post-processing AT the number? | Definition AT the number? | H convention AT the number? | Where the qualifier sits | Verdict |
|---|---|---|---|---|---|---|---|
| EF-TALFM | Connectivity (C.) = fraction of molecules with exactly one connected component | generated-sample count, rates-computed-over-N, over-N-generated | largest-connected-component-retained, after-sanitization | fraction-of, fraction-of, denotes-the-fraction | — | `src/appendix.tex:85` | **AGREES** |
| FlowMol3 | % Valid = percent of molecules that can be sanitized by RDKit; % PB-Valid adds the PoseBusters suite (incl. one-component check) | — | — | — | explicit-hydrogens, all-atom | `main.tex:141` | **DISTINGUISHES** |
| GGFlow | Validity = fraction of valid molecules AFTER valency correction or edge resampling; Val. w/o corr. = the same fraction BEFORE correction | — | — | — | — | `neurips_2024.tex:318` | **DISTINGUISHES** |
| PAFlow | (none reported) — the paper reports Vina affinity, high-affinity rate, QED, SA and Diversity; no connectivity / validity percentage | — | — | — | — | — | **NOT-STATED** |
| GEOM-DD / GFMDiff | Mol. Stability (%) = molecules whose atoms are all stable (single fragment); Validity (%) = molecules that are theoretically correct | results-of-N-generated-samples, N-generated-samples-reported | — | — | — | `paper_aaai24_cr.tex:331` | **DISTINGUISHES** |

## Per-paper detail

### EF-TALFM — arXiv:2609.08333

- title: Fixed-Dimensional Latent Flow for Generating Variable-Size 3D Molecules
- venue: arXiv preprint (EF-TALFM); the paper our bond rule came from
- URL: https://arxiv.org/e-print/2609.08333
- retrieved: yes (sha256 `e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224`)
- verdict: **AGREES** — population + post-processing + definition are all printed AT the number (in the caption of the table that carries the number).
- headline site: `src/appendix.tex:85` (float:table, percent-value=True, header-declares-percent=True)
  - > For fragmented outputs, SMILES-based and PoseBusters metrics are computed on the atom-count largest connected component after RDKit sanitization. 0.35em 0.98 [t] 0.4 0pt (a) End-to-end generation results (%) 0.25em 2.5pt @ lrrr@ Metric ( ) & UAE-3D & FlowMol & EF-TALFM C. & 99.60 & 100.00 & 92.98 S. & 83.84 & 98.72 & 98.25 U.\,\&\,S. & 83.81 & 98.27 & 98.21 N.\,\&\,U.\,\&\,S. & 76.03 & 69.83 & 94.33 [t] 0.55 0pt (b) PoseBusters results among N.\,\&\,U.\,\&\,S. molecules (%) 0.25em 2.5pt @ lrrr@ Metric ( ) & UAE-3D & FlowMol & EF-TALFM Bond lengths & 99.96 & 100.00 & 99.79 Bond angles & 99.93 & 100.00 & 99.32 Internal steric clash & 99.64 & 99.90 & 97.22 Aromatic-ring flatness & 100.00 & 100.00 & 99.93 Double-bond flatness & 99.91 & 100.00 & 98.09 Internal energy & 99.39 & 99.83 & 97.69 All PoseBusters checks & 98.91 & 99.73 & 93.13
  - caption: > EF-TALFM has the highest yield of sanitized, unique, and training-set-novel molecules, while remaining competitive on validity without generating bond labels. N.\,\&\,U.\,\&\,S. counts generated candidates that are sanitized, nonduplicate and absent from the training set, before requiring the joint PoseBusters pass used in the primary verified novel yield. EF-TALFM achieves 94.33%, versus 76.03% for UAE-3D and 69.83% for FlowMol. UAE-3D is the closest latent counterpart but uses variable-length 
- runner-up 1 site: `src/appendix.tex:305` (paragraph, percent-value=True, header-declares-percent=True)
  - > Among the 92 , 221 connected molecules, 90 , 807 reconstructed graphs pass strict sanitization before geometry refinement.
- runner-up 2 site: `src/appendix.tex:16` (float:table, percent-value=True, header-declares-percent=True)
  - > Including generated bond types raises UAE-3D exact SMILES reconstruction from 76.8% to 99.5% while leaving sanitization nearly unchanged. 0.9 @ lccc@ & Connectivity & Sanitization & SMILES match Atom type + coordinates + bond & 100.0% & 99.5% & 99.5% Atom type + coordinates & 99.0%& 99.2% & 76.8%
  - caption: > table
- bare restatements (metric site with NO qualifier AT the number): 2
  - `src/appendix.tex:16` (float:table)
  - `src/appendix.tex:80` (paragraph)
- population outside the headline site: `src/appendix.tex:93` — Panel (a) rates are computed over 10,000 generated molecules. C. denotes the fraction of connected molecules; S. the fra
- postproc outside the headline site: `src/introduction.tex:31` — om types and Cartesian coordinates, leaving connectivity to postprocessing .
- definition outside the headline site: `src/appendix.tex:60` — Connectivity: % of molecules with exactly one connected component, i.e., one fragment.
- hydrogens outside the headline site: `src/appendix.tex:9` — ally annotated molecular states, particularly for larger or heavy-atom-rich molecules where bond connectivity, bond orders, aromat

### FlowMol3 — arXiv:2508.12629

- title: FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation
- venue: Digital Discovery (2026)
- URL: https://arxiv.org/e-print/2508.12629
- retrieved: yes (sha256 `2392a2148caa120c4dc828f636a8d1326a5716fe6d44601113c32007ed8244e0`)
- verdict: **DISTINGUISHES** — the qualifier exists but in a different place or a different form: AT the number population=none, post-processing=none, definition=none.
- headline site: `main.tex:141` (float:tabular, percent-value=True, header-declares-percent=True)
  - > FlowMol3 achieves nearly 100% molecular validity for drug-like molecules with explicit hydrogens, more accurately reproduces the functional group composition and geometry of its training data, and does so with an order of magnitude fewer learnable parameters than comparable methods.
- runner-up 1 site: `main.tex:585` (paragraph, percent-value=True, header-declares-percent=True)
  - > Molecules produced by FlowMol3 are valid 99.9% of the time; a significant milestone for all-atom generative models.
  - > Moreover, FlowMol3 pushes the PB-validity rate of generated molecules nearly to that of the training data (92% vs 93%).
- runner-up 2 site: `main.tex:596` (float:table*, percent-value=True, header-declares-percent=True)
  - > To estimate % PB-Valid on the training data, due to computational constraints, we performed 10 repeated random samplings of 15 thousand molecules from a version of the dataset containing 5 conformers per unique molecule; the value presented is the sample mean with a 95% confidence interval. ! lrrrrrrr Model & % Valid ( ) & % PB-Valid ( ) & FG Dev.
  - caption: > FlowMol3 Comparison with Existing Models: Mean value of metrics computed on molecules sampled from each model, along with 95% confidence on the mean metric values as estimated from 5 repeat samplings. All models were trained on GEOM-Drugs with explicit hydrogens. RMSD is measured in Angstroms. Detailed descriptions of all metrics are available in Section . The first row of the table is the corresponding metric value computed on the training data. To estimate % PB-Valid on the training data, due 
- bare restatements (metric site with NO qualifier AT the number): 7
  - `main.tex:141` (float:tabular)
  - `main.tex:585` (paragraph)
  - `main.tex:589` (paragraph)
  - `main.tex:591` (paragraph)
  - `main.tex:596` (float:table*)
- population outside the headline site: `main.tex:579` — For all baseline models except ADiT we sampled 5000 molecules from the trained models using default settings provided by
- postproc outside the headline site: `main.tex:624` — e) were most impacted by removing fake atoms. The energy of relaxation was increased the most by removing distortion. Validity and
- definition outside the headline site: `main.tex:548` — We report ``% Valid'' which is the percent of molecules that can be sanitized by rdkit. We additionally report ``%
- hydrogens outside the headline site: `main.tex:150` — nearly 100% molecular validity for drug-like molecules with explicit hydrogens, more accurately reproduces the functional group compositi

### GGFlow — arXiv:2411.05676

- title: Improving Molecular Graph Generation with Flow Matching and Optimal Transport
- venue: TMLR (retitled 2026); originally NeurIPS 2024
- URL: https://arxiv.org/e-print/2411.05676
- retrieved: yes (sha256 `873061ea90f11391ce130414435ffb9194b10d1f85a49bdbfb7ba2cb9c2b7c38`)
- verdict: **DISTINGUISHES** — the qualifier exists but in a different place or a different form: AT the number population=none, post-processing=none, definition=none.
- NOTE: no site in the paper prints a number WITH a percent sign; the metric is printed as bare numbers under a unit-less column header.
- headline site: `neurips_2024.tex:318` (float:table, percent-value=False, header-declares-percent=False)
  - > The best results and the second-best results are marked bold and bold . p 1.8cm p 0.4cm < p 1.2cm < p 0.8cm < cp 0.4cm < p 1.2cm < p 0.8cm < cp 0.9cm < 2 * Method & 4 c QM9 & 4 c ZINC250k & 2 * Step (r) 2-5 (r) 6-9 & Val. & Val. w/o corr. & NSPDK & FCD & Val. & Val. w/o corr. & NSPDK & FCD Training Set & 100 & 100 & 0.0001 & 0.040 & 100 & 100 & 0.0001 & 0.062 & - GraphAF & 100 & 67.14 & 0.0218 & 5.246 & 100 & 67.92 & 0.0432 & 16.128 & - GraphDF & 100 & 83.14 & 0.0647 & 10.451 & 100 & 89.72 & 0.1737 & 33.899 & - MolFlow & 100 & 92.03 & 0.0169 & 4.536 & 100 & 63.76 & 0.0468 & 20.875 & - GraphEBM & 100 & 8.78 & 0.0287 & 6.402 & 100 & 5.29 & 0.2089 & 35.467 & - PS-VAE & - & - & 0.0077 & 1.259 & 100 & - & 0.0112 & 6.320 & - MolHF & - & - & - & - & 100 & 93.62 & 0.0387 & 23.940 & - EDP-GNN & 100 & 47.69 & 0.0052 & 2.683 & 100 & 83.16 & 0.0483 & 16.819 & 1000 GDSS & 100 & 96.17 & 0.0033 & 2.565 & 100 & 97.12 & 0.0192 & 14.032 & 1000 GSDM & 100 & 99.90 & 0.0034 & 2.614 & 100 & 92.57 & 0.0168 & 12.435 & 1000 GruM & 100 & 99.69 & 0.0002 & 0.108 & 100 & 98.32 & 0.0023 & 2.235 & 1000 SwinGNN & 100 & 99.66 & 0.0003 & 0.118 & 100 & 86.16 & 0.0047 & 4.398 & 500 DiGress & 100 & 98.29 & 0.0003 & 0.095 & 100 & 94.98 & 0.0021 & 3.482 & 500 GGFlow & 100 & 99.91 & 0.0002 & 0.148 & 100 & 99.63 & 0.0010 & 1.455 & 500
  - caption: > Generation results on the QM9 and ZINC250k datasets. Results are the means of 3 different runs. The best results and the second-best results are marked bold and bold .
- runner-up 1 site: `neurips_2024.tex:406` (float:table, percent-value=False, header-declares-percent=False)
  - > [!ht] Mean absolute error of molecular property on conditional generation on the QM9 dataset. p 1.2cm cccccc 2 * Methods & 2 c DiGress & 4 c GGFlow (r) 2-3 (r) 4-7 & Uncondition & +Guidance & Unconditition & Supervised Training & +SFT & +RL Mean & 1.562 & 1.092 & 1.569 & 1.184 & 1.223 & 0.672 Variance & 1.641 & 0.894 & 1.987 & 1.579 & 1.893 & 0.647 Val. w/o & 98.29 & 74.2 & 99.91 & 86.1 & 87.0 & 92.2
  - caption: > Mean absolute error of molecular property on conditional generation on the QM9 dataset.
- bare restatements (metric site with NO qualifier AT the number): 2
  - `neurips_2024.tex:318` (float:table)
  - `neurips_2024.tex:406` (float:table)
- population outside the headline site: `neurips_2024.tex:314` — chet ChemNet Distance (FCD). To calculate these metrics, we sampled 10,000 molecules. We compared GGFlow against various molecule generation mod
- postproc outside the headline site: `neurips_2024.tex:990` — the fraction of valid molecules after valency correction or edge resampling.
- definition outside the headline site: `neurips_2024.tex:990` — ence of formal charges in the training molecules. It is the fraction of valid molecules after valency correction or edge resampling
- hydrogens outside the headline site: `neurips_2024.tex:937` — f 134,000 stable organic molecules, each containing up to 9 heavy atoms: carbon, oxygen, nitrogen, and fluorine . The dataset incl

### PAFlow — arXiv:2509.01486

- title: Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number
- venue: NeurIPS 2025
- URL: https://arxiv.org/e-print/2509.01486
- retrieved: yes (sha256 `ac4fa2096d15a0eb94aec5efe2fc7154f1c48fbf352f03f0e628c2583ed51868`)
- verdict: **NOT-STATED** — No site in the retrieved source reports a connectivity/validity percentage; the metric is absent from the paper, so no qualifier can be present.

### GEOM-DD / GFMDiff — arXiv:2401.02683

- title: Geometric-Facilitated Denoising Diffusion Model for 3D Molecule Generation
- venue: arXiv preprint (v2 title) — AAAI-style source (paper_aaai24_cr.tex)
- URL: https://arxiv.org/e-print/2401.02683
- retrieved: yes (sha256 `cb0aac177d087b27fd1b00952b11e3c0d20ce94ab66f316589f5e5a0c7098c3c`)
- verdict: **DISTINGUISHES** — the qualifier exists but in a different place or a different form: AT the number population=['results-of-N-generated-samples', 'N-generated-samples-reported'], post-processing=none, definition=none.
- headline site: `paper_aaai24_cr.tex:331` (float:table*, percent-value=True, header-declares-percent=True)
  - > [t] lllllll Type & Method & NLL & Atom\ (%) & Mol\ (%) & Validity (%) & Uniqueness Validity (%) NF & E-NF & -59.7 & 85.0 & 4.9 & 40.2 & 39.4 AR & G-SchNet & N/A & 95.7 & 68.1 & 85.5 & 80.3 2 * DDPM & EDM & -110.7 1.5 & 98.7 0.1 & 82.0 0.4 & 91.9 0.5 & 90.7 0.6 & Bridge+Force & N/A & 98.8 0.1 & 84.6 0.3 & N/A & N/A & GCDM & -171.0 0.2 & 98.7 0.0 & 85.7 0.4 & 94.8 0.2 & 93.3 0.0 & GeoLDM & N/A & 98.9 0.1 & 89.4 0.5 & 93.8 0.4 & 92.7 0.5 2 * Ours & GFMDiff\ /o tri & -123.1 0.4 & 98.7 0.1 & 85.9 0.2 & 94.9 0.2 & 94.2 0.2 & GFMDiff\ /o GFLoss & -127.5 0.4 & 98.7 0.0 & 86.5 0.1 & 95.2 0.0 & 94.5 0.0 & GFMDiff & -128.0 0.2 & 98.9 0.0 & 87.7 0.2 & 96.3 0.3 & 95.1 0.2 Data & & & 99.0 & 95.2 & 97.7 & 97.7 Performance comparison on GEOM-QM9.
  - caption: > Performance comparison on GEOM-QM9. Results of 10000 generated samples are reported with standard deviations across 3 runs using different seeds.
- runner-up 1 site: `paper_aaai24_cr.tex:491` (paragraph, percent-value=True, header-declares-percent=True)
  - > The performance lead of GFMDiff over the SOTA method in validity and validity times uniqueness is 1.1% and 1.3%, respectively.
- bare restatements (metric site with NO qualifier AT the number): 3
  - `paper_aaai24_cr.tex:187` (paragraph)
  - `paper_aaai24_cr.tex:491` (paragraph)
  - `paper_aaai24_cr.tex:506` (paragraph)
- population outside the headline site: `paper_aaai24_cr.tex:353` — Performance comparison on GEOM-QM9. Results of 10000 generated samples are reported with standard deviations across 3 runs using d
- definition outside the headline site: `paper_aaai24_cr.tex:486` — h, which are stability, validity, and uniqueness. Stability measures the proportion of atoms with correct valencies and molecules whose atoms a
- hydrogens outside the headline site: `paper_aaai24_cr.tex:169` — lecule generation face two major challenges. Since majority heavy atoms in molecules allow connections to multiple atoms through s

## Proposed house format string (PROPOSAL ONLY — not applied)

```
connectivity {v:.2f}% (single-fragment rate over N={n} decoded samples; post-processing: {postproc}; denominator: {denominator}; hydrogens: {hydrogens}; definition: exactly one connected component)
```

Every slot must be filled from a value that is measured on disk. The four slots that the frontier papers most often leave to a distant caption — population, post-processing, denominator and definition — become mandatory text next to the number itself.

## Comparability breaks against the frontier convention

1. **Population.** Our headline 8.51 % renders identically for SEVEN different populations. The frontier papers state N in a caption (EF-TALFM: 10,000; GEOM-DD: 10,000) or not at all (FlowMol3 states no N for its own models). Our house rule must therefore print N *at* the number, because the frontier does not reliably do it for us.
2. **Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDKit sanitization: EF-TALFM counts connectivity BEFORE the largest-component retention (its `C.` cell is the reconstructed graph's own fragment count), while its COMPANION cells (S., U. & S., N. & U. & S.) are counted AFTER the retention + strict sanitisation; GEOM-DD and GGFlow state no fragment handling at all. The published 92.7-94 % band is **NOT ours** — it is a FRONTIER composite, and the repo holds **TWO competing provenance readings** for it, neither of which makes it a like-for-like comparison to our number. Reading 1 (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> JSON PATH `band_provenance`): its endpoints are GeoLDM's Validity & Uniq (92.7 %) and HLTF's PB-Valid (94.0 %) on QM9 — a sanitisation-and-duplication rate and a six-check pose verdict — both carrying `post_processed: false`. Reading 2 (the same file -> JSON PATH `sources`.`EF-TALFM`, EF-TALFM arXiv:2609.08333v4): it mixes EF-TALFM's OWN Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the retention) with its post-processed N. & U. & S. companion cell (94.33 %). Under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale — while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. (The exact paper the band was assembled from is NOT established on disk: it came from a literature scan whose selection rule was not recorded.)
3. **Pre- vs post-processing.** GGFlow answers the ambiguity by printing TWO columns — the post-correction validity (100 for every method, hence uninformative) and the pre-correction validity side by side. That is the single most transferable idea found in this audit: a headline post-processing number alone hides the raw rate.
4. **Definition.** EF-TALFM splits the denominator rule: connectivity, sanitization, uniqueness and novelty are computed over ALL generated samples, while the PoseBusters check rates are computed over the novel/unique/sanitized subset. A bare percentage without its denominator is ambiguous across the whole frontier.

## Honest framing

MEASURED: every excerpt above is verbatim from the retrieved LaTeX source whose sha256 is recorded in `sources.json`. NO GPU was used and no molecule was decoded. The only numbers quoted are the ones printed by the fetched papers.
PROJECTED: the house format string above is a proposal. It has NOT been applied to any metric, JSON or document in this repo.


---

## Wf-A176 CORRECTION (2026-11-01) — §"Comparability breaks" item 2 is WRONG

**This section is APPEND-ONLY. The superseded rendering above is preserved
VERBATIM per AGENTS.md §5.5 rule #6 and is NOT edited.**

### The superseded sentence (verbatim, line 139 of this file)

> 2. **Post-processing.** The frontier's connectivity rates are computed on the atom-count largest connected component after RDKit sanitization (EF-TALFM) or with no stated fragment handling at all (GEOM-DD, GGFlow). Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale.

### Why it is wrong — three independent defects

1. **OWNERSHIP.** The 92.7-94 % band is **NOT ours**. It is a **frontier
   composite**. The repo's own metric artefact
   (`metrics/by_metric/wfa155_metal_coordination_sphere.json`,
   `comparability_statement.summary`) already says: *"This metric is NOT
   comparable to the published 92.7-94 % frontier band ... Four independent
   breaks separate the two."*
2. **METRIC TYPE.** The sentence compares the band to "the frontier's
   connectivity rates". The band's endpoints are **not connectivity rates
   at all**. MEASURED from
   `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`
   (`band_provenance.endpoint_candidates`):
   - **92.7 %** = GeoLDM's **Validity & Uniq** on QM9, tabulated by HLTF
     (arXiv:2603.27113v1, Table 1(b) row `Val&Uniq (%)`, column `GeoLDM`) —
     *"a per-molecule RDKit-sanitisable AND non-duplicate rate"* over
     10,000 unconditional QM9 samples, `post_processed: false`.
   - **94 %** = HLTF's **PB-Valid** on QM9 (same arXiv:2603.27113v1,
     Table 1(b) row `PB-Valid (%)`, column `HLTF (ours)`) — *"PoseBusters
     validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
     steric clash, aromatic and double-bond flatness, internal energy)"*.
   - A **third** paper in the same scan (YuelBond, bioRxiv
     10.1101/2025.05.06.652517v1) quotes **92.7 %** as a *"multi-class
     BOND-TYPE classification F1"* on 1,000 coordinate-noised molecules.
3. **PROVENANCE.** *"comes from a DIFFERENT perception branch (Open
   Babel)"* is not supported by any artefact on disk. Every candidate
   endpoint above is a **frontier table cell**; no Open Babel branch is
   named anywhere in the `band_provenance` block.

### The authoritative replacement (verbatim from the repo's own artefact)

`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`,
`band_provenance.headline_finding`:

> THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a
> VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %);
> a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE
> CLASSIFICATION F1. None of the three is a connected-component count. Our
> 8.51 % is a connected-component count on the RAW emitted SMILES. The
> comparison the band invites is therefore not merely apples-to-oranges --
> it is a comparison of two different observables, and the post-processing
> axis is a THIRD, separate axis on top of it.

So the corrected reading of item 2 is:

> **2. Post-processing.** The frontier's connectivity rates are computed on
> the atom-count largest connected component after RDKit sanitization
> (EF-TALFM) or with no stated fragment handling at all (GEOM-DD, GGFlow).
> The published 92.7-94 % band is **NOT ours** and is **not a connectivity
> band**: its candidate endpoints are GeoLDM's Validity & Uniq (92.7 %) and
> HLTF's PB-Valid (94.0 %) on QM9 — a sanitisation-and-duplication rate and
> a six-check pose verdict — while a third paper in the same scan quotes
> 92.7 % as a per-bond-type classification F1. None of the three is a
> connected-component count
> (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`,
> `band_provenance`), so the band cannot be placed on the frontier's
> connectivity scale — while OUR 8.51 % IS a connected-component count on
> the RAW emitted SMILES, and nothing else.

### What is STILL not established (honest-negative, unchanged)

The **exact paper** the repo's band was assembled from is **NOT
established on disk**. Verbatim from the same artefact
(`band_provenance.what_this_does_not_settle`):

> This branch did NOT establish which paper the repo's band was assembled
> from, because the band was assembled by a literature scan whose selection
> rule is not recorded on disk.

### The repair

* **Generator** (`molmetal/scripts/wfa161_frontier_labelling.py`) — repaired
  2026-11-01 (Wf-A176): the module docstring and the §"Comparability
  breaks" item-2 emission now carry the corrected text, with the
  superseded sentence preserved VERBATIM in dated comments. The report is
  **not** re-run, precisely so this file's superseded rendering survives.
* **Reach**: MEASURED 2026-11-01 — the superseded sentence occurs in **0**
  of 9,551 HEAD blobs and **0** of 134 `paper/` files. It has **not**
  reached the paper.
* **CPU-only**, 0 GPU. No metric JSON, no `paper/**` write.


---

## Wf-A176 CORRECTION 2 (2026-11-01) — the "none of the endpoints is a connected-component count" claim is NOT ESTABLISHED

**This section is APPEND-ONLY. Everything above it is preserved VERBATIM per AGENTS.md §5.5 rule #6 and is NOT edited.**

### What is superseded

The "authoritative replacement" quoted in the section above states:

> None of the three is a connected-component count ... so the band cannot be placed on the frontier's connectivity scale

That claim is TRUE under ONE provenance reading on disk and FALSE under the OTHER, so it cannot be published as settled. The repo holds TWO competing readings of what the 92.7-94 % band is, both in the SAME artefact, `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`:

**Reading 1 — `band_provenance` (lines 51-89).** The band's endpoint candidates are GeoLDM's Validity & Uniq (92.7 %, `post_processed: false`) and HLTF's PB-Valid (94.0 %, `post_processed: false`), with YuelBond's per-bond-type F1 (92.7 %) as a third candidate for the same number. Under this reading neither endpoint is a connected-component count.

**Reading 2 — `sources.EF-TALFM` in the same artefact (lines 360, 378-379), corroborated by the `wfa168_frontier_axis_mapping` branch; EF-TALFM (arXiv:2609.08333v4).** The band mixes EF-TALFM's OWN Table 3(a) cells: the RAW connectivity cell `C. = 92.98 %` (counted BEFORE the largest-component retention) and the post-processed `N. & U. & S. = 94.33 %` companion cell. Verbatim, line 378: *“The '92.7-94 %' band the repo quotes mixes the LOWEST connectivity in that table with a post-processed companion cell (N. & U. & S. = 94.33).”* Under THIS reading the lower endpoint IS a connectivity count.

**Why this matters for the correction above.** The sentence it quoted — *"None of the three is a connected-component count"* — is Reading 1's claim, stated as if it were settled. It is not: the repo's own axis-mapping branch records Reading 2, and under Reading 2 the lower endpoint is a connectivity count. A published sentence must not assert a contested provenance as fact.

### The corrected, reading-safe replacement

> The published 92.7-94 % band is **NOT ours** — it is a FRONTIER composite, and the repo holds **TWO competing provenance readings** for it, neither of which makes it a like-for-like comparison to our number. Reading 1: its endpoints are GeoLDM's Validity & Uniq (92.7 %) and HLTF's PB-Valid (94.0 %) on QM9 — a sanitisation-and-duplication rate and a six-check pose verdict — both carrying `post_processed: false`. Reading 2: it mixes EF-TALFM's OWN Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component retention) with its post-processed N. & U. & S. companion cell (94.33 %). Under **BOTH** readings the band is external, is **not one homogeneous quantity**, and is **not on our scale** — while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. (The exact paper the band was assembled from is NOT established on disk: it came from a literature scan whose selection rule was not recorded.)

### What is NOT changed by this correction

* The superseded section above stays VERBATIM.
* No number moves: the 92.7 %, 94.0 %, 92.98 %, 94.33 %, 8.51 % are all as measured on disk.
* The claim that the band is external, heterogeneous, and not on our scale is unchanged — it holds under both readings.
* `paper/**`, `metrics/**`, `AGENTS.md`, `TODO/**`, `docs/HONEST_NEGATIVE_CATALOG.md` are NOT touched.
* CPU-only, 0 GPU.

---

## Wf-A183 CORRECTION (2026-11-01) — the band's locators are JSON PATHS, and its provenance is NOT settled on disk

**This section is APPEND-ONLY. Every superseded rendering it replaces is preserved
VERBATIM below per AGENTS.md §5.5 rule #6. Nothing above this line is edited.**

### The canonical reading-safe band statement (VERBATIM from the record)

Verbatim from `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, JSON PATH
`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement`:

> The published band's provenance is NOT settled on disk: the frontier post-processing matrix record
> (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under
> `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus
> YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE
> the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even
> agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.  Under BOTH
> readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a
> connected-component count on the RAW emitted SMILES, and nothing else.

### The four invariants (VERBATIM from the record)

Verbatim from the same JSON PATH, `the_only_claims_that_may_be_published`:

* **(i)** The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a FRONTIER composite, not our measurement.
* **(ii)** It is NOT one homogeneous quantity: even under Reading A its two endpoints are two different observables (a sanitisation-and-duplication rate and a six-check pose verdict), and under Reading B one endpoint is a raw connectivity count while the other is a post-processed composite.
* **(iii)** It is NOT on our scale.  OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.
* **(iv)** The source cells of the band are NOT settled on disk.

### The corrected locators — JSON PATHS, not line numbers

| Reading | ROBUST locator (JSON PATH of `frontier_matrix.json`) | What it is NOT |
|---|---|---|
| Reading 1 | `band_provenance` | NOT a line range |
| Reading 2 | `sources`.`EF-TALFM` | NOT `frontier_axis_mapping` — that string is NOT a key of the record; it survives only as a STRING VALUE |

### The X1-X4 forbidden-claim list

* **X1 — FORBIDDEN as a settled claim:** "THE BAND IS NOT A CONNECTIVITY BAND" / "None of the three is a connected-component count".  TRUE under Reading 1, FALSE under Reading 2 (whose RAW connectivity cell is `C. = 92.98 %`).  The record's LIVE `band_provenance`.`headline_finding` no longer carries this wording — it is preserved VERBATIM at `band_provenance`.`headline_finding_superseded_2026_11_01`.
* **X2 — FORBIDDEN as a locator:** citing `frontier_axis_mapping` as a JSON key of `frontier_matrix.json`.  It is not a key; use the JSON PATH `sources`.`EF-TALFM`.
* **X3 — FORBIDDEN as a locator:** line-number locators (`:51-89`, `:360, :378-379`).  Line numbers rot as the record gains keys; cite the JSON PATHS `band_provenance` and `sources`.`EF-TALFM`.
* **X4 — FORBIDDEN:** attributing the band to Mol-Metal, or to a "DIFFERENT perception branch (Open Babel)".  The band is EXTERNAL, and no Open Babel branch is named anywhere in the band-provenance record.

### The superseded renderings, preserved VERBATIM

**The producer's module docstring, pre-Wf-A183 (the Reading 1 / Reading 2 locator lines):**

>         Reading 1  `band_provenance` (:51-89): GeoLDM's Validity & Uniq
>                    92.7 % and HLTF's PB-Valid 94.0 %, both with
>                    `post_processed: false`; a third paper quotes 92.7 %
>                    as a per-bond-type classification F1.  Under this
>                    reading no endpoint is a connected-component count.
>         Reading 2  `frontier_axis_mapping` (:360, :378-379), EF-TALFM
>                    arXiv:2609.08333v4: the band mixes EF-TALFM's OWN
>                    Table 3(a) raw connectivity cell (C. = 92.98 %, counted
>                    BEFORE the largest-component retention) with its
>                    post-processed N. & U. & S. companion cell (94.33 %).
>                    Under THIS reading the lower endpoint IS a
>                    connectivity count.

**The producer's emitted §"Comparability breaks" item 2, pre-Wf-A183:**

> Reading 2 (the same file -> `frontier_axis_mapping`, EF-TALFM arXiv:2609.08333v4): it mixes EF-TALFM's OWN Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the retention) with its post-processed N. & U. & S. companion cell (94.33 %).

**The record's superseded `band_provenance`.`headline_finding` — the X1 wording — preserved VERBATIM at `band_provenance`.`headline_finding_superseded_2026_11_01`:**

> THE BAND IS NOT A CONNECTIVITY BAND.  Its two endpoints are a VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %); a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE CLASSIFICATION F1.  None of the three is a connected-component count.  Our 8.51 % is a connected-component count on the RAW emitted SMILES.  The comparison the band invites is therefore not merely apples-to-oranges -- it is a comparison of two different observables, and the post-processing axis is a THIRD, separate axis on top of it.

### What this correction does NOT change

* No number moves: 92.7 %, 94.0 %, 92.98 %, 94.33 %, 8.51 % are the values measured on disk.
* The four invariants above are unchanged — they hold under BOTH readings.
* `paper/**`, `metrics/**`, `registry/**`, `AGENTS.md`, `TODO/**`, `docs/**` are NOT touched.
* CPU-only, 0 GPU.

---

## Wf-A187 CORRECTION (2026-11-01) — the "authoritative replacement" subsection's X1 rendering is SUPERSEDED

**This section is APPEND-ONLY. Every superseded rendering it names is preserved
VERBATIM below per AGENTS.md §5.5 rule #6. Nothing above this line is edited.**

### The superseded subsection, named

The subsection this correction supersedes is the one headed

> ### The authoritative replacement (verbatim from the repo's own artefact)

which sits inside `## Wf-A176 CORRECTION (2026-11-01) — §"Comparability breaks" item 2 is WRONG`.
It presented itself as *the authoritative replacement* and quoted the record's then-live
`band_provenance`.`headline_finding` VERBATIM:

> THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a
> VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %);
> a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE
> CLASSIFICATION F1. None of the three is a connected-component count. Our
> 8.51 % is a connected-component count on the RAW emitted SMILES. The
> comparison the band invites is therefore not merely apples-to-oranges --
> it is a comparison of two different observables, and the post-processing
> axis is a THIRD, separate axis on top of it.

### It is superseded — by what

That wording is **X1**, and it is **SUPERSEDED**:

* by `## Wf-A176 CORRECTION 2 (2026-11-01)` in this same file;
* by `## Wf-A183 CORRECTION (2026-11-01)` in this same file;
* and by the owning record itself — `band_provenance`.`headline_finding` no longer carries
  this wording (it is preserved VERBATIM at `band_provenance`.`headline_finding_superseded_2026_11_01`).

X1 is TRUE under Reading 1 and FALSE under Reading 2 (whose RAW connectivity cell is
`C. = 92.98 %`), so it must not be published as a settled claim. What this correction
withdraws is the superseded subsection's self-description — the heading that called its
content "the authoritative replacement" — and nothing else.

### The reading-safe replacement (the only form that may be quoted)

Verbatim from `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, JSON PATH
`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement`:

> The published band's provenance is NOT settled on disk: the frontier post-processing matrix record
> (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under
> `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus
> YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE
> the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even
> agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.  Under BOTH
> readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a
> connected-component count on the RAW emitted SMILES, and nothing else.

### The four invariants (VERBATIM from the record)

Verbatim from the same JSON PATH, `the_only_claims_that_may_be_published`:

* **(i)** The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a FRONTIER composite, not our measurement.
* **(ii)** It is NOT one homogeneous quantity: even under Reading A its two endpoints are two different observables (a sanitisation-and-duplication rate and a six-check pose verdict), and under Reading B one endpoint is a raw connectivity count while the other is a post-processed composite.
* **(iii)** It is NOT on our scale.  OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.
* **(iv)** The source cells of the band are NOT settled on disk.

(The record's Reading A / Reading B are this file's Reading 1 / Reading 2.)

### The band's provenance is NOT settled on disk

Which frontier cells the "92.7-94 %" band names is NOT settled on disk: the record holds TWO
competing readings of them and they do not even agree on the numerals (92.7/94.0 vs
92.98/94.33). The four invariants above are the only publishable claims.

### What this correction does NOT change

* No number moves: 92.7 %, 94.0 %, 92.98 %, 94.33 %, 8.51 % are the values measured on disk.
* The superseded subsection above stays VERBATIM — this section ADDS a supersession marker for it
  and nothing else (AGENTS.md §5.5 rule #6 — append-only).
* `paper/**`, `metrics/**`, `registry/**`, `AGENTS.md`, `TODO/**`, `docs/**` are NOT touched.
* CPU-only, 0 GPU.

---

## Wf-A194 CORRECTION (2026-11-02) — every dated block is now emitted by the producer, and the live item 2 no longer carries the refuted X3 / X4 rendering

**This section is APPEND-ONLY. Every superseded rendering it names is preserved
VERBATIM elsewhere in this file per AGENTS.md §5.5 rule #6. Nothing above this
line is edited.**

### 1. What this producer changed (MEASURED, 2026-11-02)

Before this wave this producer emitted ONLY the Wf-A183 and Wf-A187 dated
CORRECTION sections. The Wf-A176 and Wf-A176-2 sections existed ONLY in the
published `report.md` and were copied forward by `_carry_dated_sections`, which
returns "" when the target `report.md` does not exist. A run with a FRESH
`--out-dir` therefore DROPPED both dated sections:

* fresh `--out-dir` (2026-11-02, `--offline`, cache `/tmp/wfa161_arxiv_cache`):
  **31,087 B / 304 lines**;
* published `report.md` before this wave: **38,505 B / 429 lines**;
* `diff -u` between the two: **3 hunks**, of which **127 lines are
  published-only** — the two Wf-A176 sections, and nothing else.

A re-run would therefore have ERASED dated history (AGENTS.md §5.5 rule #6).
`WFA176_CORRECTION_SECTION` and `WFA176_CORRECTION2_SECTION` below hold those
two sections VERBATIM and are EMITTED, so a fresh `--out-dir` reproduces every
dated block.

### 2. The live rendering this closes

The `## Comparability breaks` item 2 of the PREVIOUS published `report.md`
asserted, as a current fact, that the frontier's connectivity rates are
computed on the atom-count largest connected component after RDKit
sanitisation (EF-TALFM), and that the published band comes from a DIFFERENT
perception branch (Open Babel) and a DIFFERENT post-processing. BOTH
renderings are REFUTED on disk:

* **the retention claim** — EF-TALFM counts connectivity BEFORE the
  largest-component retention; its `C.` cell is the reconstructed graph's own
  fragment count;
* **the Open-Babel / "DIFFERENT post-processing" claim** — the band is
  EXTERNAL, it is a FRONTIER composite, and no Open Babel branch is named
  anywhere in the band-provenance record.

This producer's item 2 has emitted the corrected, reading-safe form since
Wf-A176 CORRECTION 1; this wave makes the PUBLISHED file carry it too.

**The superseded item-2 sentence survives VERBATIM** in this file, in the
`## Wf-A176 CORRECTION (2026-11-01)` section, under the sub-heading
`### The superseded sentence (verbatim, line 139 of this file)`. It is NOT
deleted (AGENTS.md §5.5 rule #6).

**LOCATOR STALENESS (MEASURED, 2026-11-02).** That sub-heading's phrase
"line 139 of this file" is a LINE-NUMBER locator and it is now STALE: line 139
of this file carries the CORRECTED item 2, and the superseded sentence stands
in the Wf-A176 CORRECTION section instead. Cite JSON PATHS, never line
numbers: `band_provenance` and `sources`.`EF-TALFM` in
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`. This
dated section does NOT rewrite the older dated section (AGENTS.md §5.5
rule #6 — append-only).

**CROSS-WAVE CONSEQUENCE (MEASURED, 2026-11-02).** Two published artefacts
outside this producer's scope cite the OLD line 139 by line number and quote
the old sentence:
`molmetal/reports/wfa175_frontier_comparability/report.md` and
`molmetal/reports/wfa175_frontier_comparability/sources.json`. They are stale
after this wave and their owners must re-point them.

### 3. The canonical reading-safe band statement (VERBATIM from the record)

Verbatim from `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, JSON PATH
`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement`:

> The published band's provenance is NOT settled on disk: the frontier post-processing matrix record
> (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under
> `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus
> YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE
> the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even
> agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.  Under BOTH
> readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a
> connected-component count on the RAW emitted SMILES, and nothing else.

### 4. The four invariants — cited BY JSON PATH, not re-quoted

`band_provenance`.`competing_reading_and_what_this_does_not_settle`.`the_only_claims_that_may_be_published`
holds the four publishable invariants. They are quoted VERBATIM in the Wf-A183
and Wf-A187 CORRECTION sections above; they are cited HERE BY PATH so that this
section does not duplicate the band numeral.

### 5. What this correction does NOT change

* No number moves: 92.7 %, 94.0 %, 92.98 %, 94.33 %, 8.51 % are the values measured on disk.
* Every dated block that was in the published file is still in it, byte-identical.
* `paper/**`, `metrics/**`, `registry/**`, `AGENTS.md`, `TODO/**`, `docs/**` are NOT touched.
* CPU-only, 0 GPU.
