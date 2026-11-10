# Wf-A158 BRANCH FRONT -- frontier post-processing comparability matrix

Retrieved: **2026-11-01**. Built by `molmetal/scripts/wfa158_frontier_postproc_matrix.py` (CPU-only, 0 GPU).

> Comparability matrix for the published frontier connectivity / validity bands. One row per source, one column per break. Each cell records what the SOURCE says, with a verbatim quote and a locator. THIS IS NOT A SCOREBOARD: a BREAKS verdict means the two protocols are not the same measurement, not that either side is wrong.

## The matrix

| source | DENOMINATOR | METRIC | POST-PROCESSING | HYDROGENS |
|---|---|---|---|---|
| **EF-TALFM** | BREAKS | BREAKS | BREAKS | DOES-NOT-BREAK |
| **FlowMol3** | BREAKS | BREAKS | BREAKS | BREAKS |
| **GGFlow** | BREAKS | BREAKS | BREAKS | DOES-NOT-BREAK |
| **GeoLDM** | BREAKS | BREAKS | DOES-NOT-BREAK | NOT-STATED |
| **HLTF** | BREAKS | BREAKS | BREAKS | BREAKS |
| **PAFlow** | NOT-STATED | NOT-STATED | NOT-STATED | NOT-STATED |
| **YuelBond** | BREAKS | BREAKS | NOT-STATED | NOT-STATED |

## EF-TALFM

EF-TALFM -- Equivariant-Free Transformer-Autoencoded Latent Flow Matching

* arXiv **2609.08333** (v4 (latest; published 2026-09-08, updated 2026-10-21)) -- Fixed-Dimensional Latent Flow for Generating Variable-Size 3D Molecules
* retrieved 2026-11-01 from `https://arxiv.org/html/2609.08333v4`
* venue: arXiv preprint (no journal-ref in the arXiv API record)
* representation: 3D heavy-atom scaffold; unconditional generation on PCQM4Mv2
* reported band: **C. (connected) 92.98 %** (Table 3 (a) (identical numbers in Table 4, Appendix I of the v1 layout)); sample nature: training-distribution (unconditional PCQM4Mv2), NOT held-out

### DENOMINATOR -- BREAKS

> Panel (a) rates are computed over 10,000 generated molecules.

*locator*: arXiv:2609.08333v4, Table 3 caption; also Table 4 caption ('All rates are computed over 10,000 generated molecules.') and Section A.1.4

10,000 molecules drawn from the TRAINING-DISTRIBUTION model (unconditional PCQM4Mv2, 'we model only the heavy-atom scaffold'), over a '90%/10% train/validation split after preprocessing'. Their denominator is 'all generated samples'; ours is one decode. Their evaluation is NOT held-out -- it is unconditional generation from the training distribution.

### METRIC -- BREAKS

> Connectivity: % of molecules with exactly one connected component, i.e., one fragment.

*locator*: arXiv:2609.08333v4, Section A.1.4 'Evaluation metrics' (same bullet in the v1 layout: Appendix G.3)

Their 'Connectivity' is a property of the paper's OWN rule-based graph reconstruction (Appendix D, Algorithm 4) from the decoded coordinates plus the decoder's predicted chemical attributes -- 'a deterministic chemistry-aware reconstruction procedure that is separate from the learned generator'. Ours is an independent RDKit connected-component count on the raw emitted SMILES. Two different instruments, not two estimates of one quantity.

### POST-PROCESSING -- BREAKS

> We report connectivity of the reconstructed molecule and sanitization, uniqueness, and novelty rates computed after retaining the atom-count largest connected component of each sampled molecule and applying strict RDKit sanitization.

*locator*: arXiv:2609.08333v4, Table 4 caption (Appendix A.3); corroborated by the Table 3 caption sentence 'For fragmented outputs, SMILES-based and PoseBusters metrics are computed on the atom-count largest connected component after RDKit sanitization.' and by Appendix D ('we retain the largest connected component and discard the rest, since the reconstructed graph may contain multiple components.')

*recipe verdict*: **STATED** -- largest-connected-component retention (atom-count) -> strict RDKit sanitisation -> (for the geometry rows) a hydrogen-assisted restrained force-field refinement in which temporary explicit hydrogens are added, all heavy atoms are restrained near their decoded positions during minimisation, and the temporary hydrogens are discarded afterwards. Connectivity (C.) is counted BEFORE the retention; every companion cell in the same table is counted AFTER it.

MIXED, and the nuance matters. The C. = 92.98 % cell is measured on the reconstructed graph BEFORE the largest-component retention and BEFORE strict sanitisation -- two independent pieces of evidence: (i) the pipeline always retains the largest component, so a post-retention count would be 100 % by construction, and the paper reports 92.98 %; (ii) the property-conditioned population in Appendix B.1.2 counts '92,221 connected molecules' whose reconstructed graphs 'pass strict sanitization before geometry refinement', i.e. connected is counted first. BUT every companion cell in the same table (S., U. & S., N. & U. & S.) IS post-processed -- largest-component retention + strict RDKit sanitisation -- and the PoseBusters rows run on the N. & U. & S. subset with a different denominator, after a hydrogen-assisted restrained force-field refinement stage. So the '92.7-94 % band' that mixes 92.98 (raw connectivity) with 94.33 (post-processed novel+unique+sanitised) is NOT one homogeneous quantity. ON OUR SIDE the same operation is worth 8.51 % -> 100 % (see ANCHOR.postprocessed_counterpart), which is why the post-processing axis is the whole gap.

### HYDROGENS -- DOES-NOT-BREAK

> Following common practice in 3D small-molecule generation, we model only the heavy-atom scaffold.

*locator*: arXiv:2609.08333v4, Section 3.1.1 'Benchmark dataset'; corroborated by Appendix D ('the heavy-atom branch, which returns an implicit-hydrogen molecule') and by 'For heavy-atom generation, RDKit first adds temporary explicit hydrogens ... the temporary hydrogens are discarded'

Their connectivity count is on an IMPLICIT-hydrogen heavy-atom graph, same as ours. Explicit hydrogens are added only temporarily for the restrained force-field refinement and are discarded again. This is a case where the main thread's break-4 does NOT apply to this source.

Caveats the source states about itself:
* The Table 3 caption reports the per-cell denominators explicitly (10,000 for panel (a); the N. & U. & S. counts 7,603 / 6,983 / 9,433 for panel (b)).
* Table 3 caption: 'Missing internal-energy evaluations (45 for UAE-3D, 12 for FlowMol, and 191 for EF-TALFM) are counted as failures in both the internal-energy and joint pass rates.'
* The illustrated molecules are selected through a separate chemical screen, not sampled blindly.

Other observations:
* THE BAND IS NOT HOMOGENEOUS EVEN INSIDE ONE TABLE. In Table 3 (a) the connectivity (C.) column reads UAE-3D 99.60 | FlowMol 100.00 | EF-TALFM 92.98 -- so the same 10,000-sample protocol yields connectivity between 92.98 % and 100 % depending on the model, and the frontier's OWN best-practice baseline (FlowMol, the model whose numbers EF-TALFM is compared against) sits at 100.00. The '92.7-94 %' band the repo quotes mixes the LOWEST connectivity in that table with a post-processed companion cell (N. & U. & S. = 94.33).
* EF-TALFM is the paper's own contribution and it reports the LOWEST raw connectivity of the three models it compares -- so 'the frontier reports 92.7-94 %' is not a statement about what the frontier achieves, it is a statement about which cells of which tables were selected.

## FlowMol3

FlowMol3 -- Flow Matching for 3D De Novo Small-Molecule Generation

* arXiv **2508.12629** (v1 (only version on arXiv; published 2025-08-18)) -- FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation
* retrieved 2026-11-01 from `https://arxiv.org/html/2508.12629v1`
* venue: arXiv:2508.12629; the branch brief also cites Digital Discovery 2026, 5(5):2052-2066 (NOT independently verified in this session -- the arXiv API record carries no journal-ref)
* representation: 3D all-atom (explicit H) conformer; unconditional generation on GEOM-Drugs
* reported band: **% Valid 99.9 +/- 0.1 | % PB-Valid 91.9 +/- 0.7** (Table 1 (row 'FlowMol3')); sample nature: training-distribution (unconditional GEOM-Drugs), NOT held-out

### DENOMINATOR -- BREAKS

> For all baseline models except ADiT we sampled 5000 molecules from the trained models using default settings provided by the authors.

*locator*: arXiv:2508.12629v1, Section 3.5; plus github.com/dunni3/FlowMol :: fm3_evals/readme.md for the FlowMol3 row's own --n_mols=5000 --n_subsets=5

5,000 molecules from the TRAINING-DISTRIBUTION model (unconditional GEOM-Drugs), split into 5 random subsets of 1,000 for the confidence interval. The paper states the baseline denominators but NOT the FlowMol3 row's own denominator -- that number comes from the authors' public evaluation harness, and the code makes the denominator exact: metrics['frac_valid_mols'] = metrics['frac_valid_mols'] * len(sampled_mols) / n_raw_samples, i.e. EVERY raw sample counts, including those that failed to build a molecule at all. Ours is one decode.

### METRIC -- BREAKS

> We report “% Valid” which is the percent of molecules that can be sanitized by rdkit.

*locator*: arXiv:2508.12629v1, Section 3.2.1 'Validity'; the connectivity criterion is the embedded PoseBusters check 'whether the molecule forms one component (not fragmented into separate molecules)'

Their headline pair is a TOOLKIT-LEVEL sanitisation rate and a POSE-GEOMETRY verdict (six PoseBusters checks including an internal-energy comparison against an ETKDGv3+UFF ensemble). The 'one component' criterion inside % PB-Valid is one check among six, on a 3D all-atom conformer. Ours is an RDKit connected-component count on a SMILES string -- there is no conformer and no geometry in our measurement at all. The paper's own caveat: 'We argue that producing “valid” molecules is a necessary but insufficient condition for a practically useful molecular generative model.'

### POST-PROCESSING -- BREAKS

> Chem.SanitizeMol(largest_mol)

*locator*: github.com/dunni3/FlowMol :: flowmol/analysis/metrics.py :: SampleAnalyzer.compute_validity (the script the paper names: 'Metrics on the sampled molecules were then computed using the same script in the FlowMol repository.')

*recipe verdict*: **STATED** -- Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False) -> pick the fragment with the most atoms -> Chem.SanitizeMol(largest_mol) -> Chem.MolToSmiles(largest_mol); the SDF loader is Chem.SDMolSupplier(..., sanitize=False, removeHs=False). NO deduplication. The PoseBusters run is on the RAW molecule, not on the largest fragment.

THE CENTRAL FINDING FOR THIS CELL, and it is MEASURED FROM THE AUTHORS' OWN CODE. In compute_validity the connectedness counter and the validity counter are different measurements of the same samples: 'if len(mol_frags) > 1: error_message['disconnected'] += 1 else: n_connected += 1' increments n_connected ONLY for single-fragment molecules, while n_valid is incremented AFTER 'largest_mol = max(mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms())' and 'Chem.SanitizeMol(largest_mol)'. So a FRAGMENTED molecule whose largest fragment sanitises counts as VALID but NOT connected. '% Valid' (99.9 +/- 0.1) IS the post-processed number, with denominator 'n_valid / len(sampled_molecules)' = ALL raw samples. AND THE PAPER NEVER REPORTS THE RAW COUNTER: exhaustive grep of the extracted paper text shows the word 'connect' appears only in 'fully-connected graph' descriptions and in the PoseBusters check list; there is no connectivity or fragmentation RATE in Table 1, even though the authors' harness computes 'frac_connected', 'avg_frag_frac' and 'avg_num_components'. The fragments are taken with 'sanitizeFrags=False', and the SDF loader is 'Chem.SDMolSupplier(..., sanitize=False, removeHs=False)'. NO deduplication happens before the count (unlike EF-TALFM's U. & S. / N. & U. & S.). The PoseBusters run is on the RAW molecule ('rdmols = [sample.rdkit_mol for sample in sampled_molecules]'), so the 'one component' check inside % PB-Valid is NOT largest-fragment post-processed -- it is a raw check. Training-side kekulisation is also in the pipeline: 'we kekulize all molecules in the dataset and do not explicitly model aromatic bonds.'

### HYDROGENS -- BREAKS

> We train models on the GEOM-Drugs dataset 57 using explicit hydrogens.

*locator*: arXiv:2508.12629v1, Section 3.1 'Training Data'; Table 1 caption ('All models were trained on GEOM-Drugs with explicit hydrogens.'); github.com/dunni3/FlowMol :: flowmol/analysis/molecule_builder.py :: build_molecule (every sampled atom, hydrogens included, becomes an RDKit atom node) and fm3_evals/baselines/compute_baseline_comparison.py ('Chem.SDMolSupplier(..., sanitize=False, removeHs=False)')

Their molecules carry EXPLICIT hydrogen atoms as graph nodes, so the fragment count, the sanitisation and the PoseBusters checks are all computed on an explicit-H graph. Ours is implicit-H. (Their PoseBusters geometry module separately sets 'ignore_hydrogens: True' and the loader 'add_hs: False', so no hydrogen is added or removed by the evaluation harness -- the explicit hydrogens are the model's own output.)

Caveats the source states about itself:
* Section 3.2.2: 'We argue that producing “valid” molecules is a necessary but insufficient condition for a practically useful molecular generative model.'
* Section 4.1: 'the difference in chemistry between generated molecules and “real” molecules is still practically significant.'
* The Table 1 caption flags that the training-data PB-Valid reference is itself a sampled estimate (10 repeated random samplings of 15 thousand molecules) because of computational constraints.

Other observations:
* THE FRONTIER'S OWN HARNESS COMPUTES THE RAW CONNECTIVITY NUMBER AND THE PAPER DOES NOT PRINT IT. 'frac_connected' (raw, single-fragment only) and 'avg_frag_frac' (mean fraction of atoms retained by the largest-fragment step) are both computed by SampleAnalyzer.compute_validity, and the FlowMol3 PAPER reports neither: exhaustive grep of the extracted paper text finds 'connect' only in 'fully-connected graph' (the model's graph representation) and in the PoseBusters check list, and 'fragment' only in a reference title and in the same PoseBusters sentence. The published table carries % Valid and % PB-Valid -- the first post-processed, the second pose-based.
* 'avg_frag_frac' IS THE ONE LIKE-FOR-LIKE NUMBER ACROSS BOTH SIDES, and the frontier does not publish its value either: our own counterpart is the postproc-arm file's /mass_retention/heavy_atom_retention_mean = 0.7395022201805989 (73.95 % of heavy atoms kept, mean; median 0.926829268292683). The FlowMol3 VALUE of avg_frag_frac was NOT retrieved: it is not in the paper, and this branch did not run their sampler.
* The paper's own framing of the headline metric is a caveat, not a claim: 'We argue that producing "valid" molecules is a necessary but insufficient condition for a practically useful molecular generative model.'

NOT-RETRIEVED (version-level):
* **the journal version the branch brief cites -- 'FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation', Digital Discovery 2026, 5(5):2052-2066** -- the DOI supplied in the branch brief; NOT fetched in this session (HTTP None). Reason: not fetched: the branch brief supplied the citation, and the arXiv API record for 2508.12629 carries NO journal-ref, so the published version could not be located from the metadata that WAS retrieved Consequence: every FlowMol3 cell above is stated for arXiv v1 plus the authors' public evaluation harness at github.com/dunni3/FlowMol @ main. The Digital Discovery published protocol was NOT verified in this session.

## GGFlow

GGFlow -- graph flow matching with optimal transport

* arXiv **2411.05676** (v1 (only version on arXiv; published 2024-11-08)) -- Improving Molecular Graph Generation with Flow Matching and Optimal Transport
* retrieved 2026-11-01 from `https://arxiv.org/html/2411.05676v1`
* venue: arXiv:2411.05676. The branch brief describes the retitled TMLR 2026 version 'GGFlow: A Graph Flow Matching Method with Efficient Optimal Transport' (OpenReview K8RlXtMgzo) -- THAT VERSION WAS NOT RETRIEVED (OpenReview returned a browser-verification challenge / HTTP 403 ChallengeRequiredError on both the forum page and api2.openreview.net). Everything below is from the arXiv v1.
* representation: 2D molecular GRAPHS (no 3D coordinates at all); QM9 and ZINC250k
* reported band: **QM9 Val. 100 / Val. w/o corr. 99.91 ; ZINC250k Val. 100 / Val. w/o corr. 99.63** (Table 1 (means of 3 runs)); sample nature: training-distribution (unconditional graph generation), NOT held-out

### DENOMINATOR -- BREAKS

> To calculate these metrics, we sampled 10,000 molecules.

*locator*: arXiv:2411.05676v1, Section 4.1 'Molecular Graph Generation'; Table 1 caption ('Results are the means of 3 different runs.')

10,000 sampled 2D graphs, reported as the mean of 3 independent runs, from a model trained on the TRAINING split (QM9: 'We follow the train/test split from GDSS, using 12,000 molecules for training and the remaining 1,000 for testing.'). Unconditional, not held-out. Ours is one decode of one pool.

### METRIC -- BREAKS

> Validity We permit atoms to exhibit formal charges during valency checks because of the presence of formal charges in the training molecules. It is the fraction of valid molecules after valency correction or edge resampling.

*locator*: arXiv:2411.05676v1, Appendix E 'Details of Molecule Datasets' -- the metric-definition list (Validity / Validity w/o correction / FCD / NSPDK)

Their 'Validity' is a GRAPH-LEVEL valency check on a 2D molecular graph (with formal charges permitted), and the paper says so itself: 'It is the fraction of valid molecules after valency correction or edge resampling.' There is no 3D conformer and no fragment count anywhere in it. Ours counts RDKit connected components on a 3D decode. The two are not the same observable at all -- this is a bigger break than the four the main thread listed, and it is not one of the four columns.

### POST-PROCESSING -- BREAKS

> Validity w/o correction This metric explicitly evaluates the quality of molecule generation before any correction phase, providing a baseline for raw generation performance.

*locator*: arXiv:2411.05676v1, Appendix E metric definitions; Section 4.1 commentary

*recipe verdict*: **NOT-STATED** -- The paper names a 'correction phase' -- 'after valency correction or edge resampling' -- and PUBLISHES its effect (Val. 100 vs Val. w/o corr. 99.91 / 99.63), but MEASURED: an exhaustive grep of the extracted full text for 'correct' / 'resampl' / 'repair' hits exactly FOUR lines -- the Section 4.1 metric list, the Section 4.1 commentary sentence, and the two Appendix-E metric definitions ('charge' appears in ONE of them). The recipe itself is NEVER DESCRIBED.

THE CLEANEST EVIDENCE IN THE WHOLE MATRIX, AND IT IS SELF-REPORTED BY THE FRONTIER. GGFlow publishes BOTH the post-processed number and its RAW counterpart: Val. 100 vs Val. w/o corr. 99.91 (QM9) and 100 vs 99.63 (ZINC250k). So the published 'Validity 100' IS post-processed by the paper's own metric definition. HONEST MAGNITUDE: for GGFlow itself the correction is worth only +0.09 pp (QM9) / +0.37 pp (ZINC250k); the axis is nonetheless load-bearing for the other rows in the SAME table (GraphEBM 8.78 -> 100, GraphAF 67.14 -> 100, GDSS 96.17 -> 100). AND THE RECIPE ITSELF IS NOT-STATED: an exhaustive grep of the extracted full text for 'correct'/'resampl'/'repair' hits exactly FOUR lines -- the Section 4.1 metric list, the Section 4.1 commentary sentence, and the two Appendix-E metric definitions ('charge' appears in ONE of them) -- the paper never describes what 'valency correction or edge resampling' does. A frontier number whose recipe is unreadable cannot be compared to ours at the recipe level, only at the number level.

### HYDROGENS -- DOES-NOT-BREAK

> each containing up to 9 heavy atoms: carbon, oxygen, nitrogen, and fluorine

*locator*: arXiv:2411.05676v1, Appendix E.3.1 'Dataset'; corroborated by Table S1 ('QM9 | Real | 133,885 | [1, 9] | 4 | 3') -- 4 node types, no hydrogen

GGFlow generates HEAVY-ATOM graphs only (QM9: C/O/N/F; ZINC250k: 9 heavy-atom types), so its molecules are implicit-hydrogen, same as ours. The word 'hydrogen' appears NOWHERE in the paper's text (verified by exhaustive grep of the extracted full text) -- the convention is implicit in the node-type table, not stated as a protocol.

Caveats the source states about itself:
* The paper publishes the raw (uncorrected) validity next to the corrected one, which is what makes the post-processing axis visible at all.
* No caveat about the validity metric itself is stated beyond the metric definition.

Other observations:
* The TMLR retitle could not be verified: OpenReview K8RlXtMgzo returned a browser-verification challenge (HTTP 403 ChallengeRequiredError) on both the forum page and the api2 notes endpoint. The arXiv v1 title is 'Improving Molecular Graph Generation with Flow Matching and Optimal Transport'.

NOT-RETRIEVED (version-level):
* **the retitled TMLR version of this paper -- 'GGFlow: A Graph Flow Matching Method with Efficient Optimal Transport', OpenReview K8RlXtMgzo -- which the branch brief names** -- https://openreview.net/forum?id=K8RlXtMgzo (HTTP 403). Reason: the forum page returned a browser-verification challenge ('Verifying your browser') and api2.openreview.net returned {"name":"ChallengeRequiredError"} with HTTP 403 Consequence: any change of protocol between arXiv v1 and the TMLR version is UNKNOWN. Every GGFlow cell above is stated for arXiv v1 only, and the retitle is NOT verified in this session.

## GeoLDM

GeoLDM -- Geometric Latent Diffusion Models (3D molecule generation; bond types INFERRED from the generated geometry)

* arXiv **2305.01140** (v1 (only version on arXiv; submitted 2023-05-02)) -- Geometric Latent Diffusion Models for 3D Molecule Generation
* retrieved 2026-11-01 from `https://arxiv.org/e-print/2305.01140v1`
* venue: ICML 2023 -- the paper's own source uses the ICML 2023 class with the [accepted] option (icml2023.tex:26) and declares '\icmlkeywords{Machine Learning, ICML}' (icml2023.tex:131). The arXiv record carries NO journal-ref, NO comment and NO DOI, so the venue is taken from the paper's own source file, not from an arXiv metadata field.
* representation: 3D coordinates + bond types INFERRED post-hoc ('Given molecular geometries, we first predict bond types (single, double, triple, or none) by pair-wise atomic distances and atom types'); unconditional; QM9 and DRUG
* reported band: **QM9 Atom-Sta 98.9 +/- 0.1 ; Mol-Sta 89.4 +/- 0.5 ; Valid 93.8 +/- 0.4 ; Valid & Unique 92.7 +/- 0.5 (10,000 samples)** (tab:qm9_results (icml2023.tex:491)); sample nature: training-distribution (unconditional generation from the trained model, evaluated on QM9), NOT held-out -- 'Metrics are calculated with 10000 samples generated from each model. On QM9, we run the evaluation for 3 times and report the derivation.' (icml2023.tex:468) and 'We generate $10,000$ samples from each method to calculate the above metrics' (icml2023.tex:549).  The QM9 cells are therefore the model's OWN unconditional output over 10,000 generated samples and the paper reports NO held-out evaluation split for them; the table's own 'Data' row reports the QM9 DATASET's statistics (Valid 97.7 %, Valid & Unique 97.7 %), i.e. the reference population is QM9 itself.

### DENOMINATOR -- BREAKS

> We generate $10,000$ samples from each method to calculate the above metrics, and the results are reported in \cref{tab:qm9_results}.

*locator*: arXiv:2305.01140v1, icml2023.tex:549 and the tab:qm9_results caption at icml2023.tex:468; source file sha256 b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, e-print sha256 b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23

10,000 unconditional QM9 samples, the evaluation run 3 times with the DERIVATION reported. Ours is one decode of one pool of 2,112 or 12,669 rows. The protocol is the same shape as the other frontier papers' (sampling from the model trained on the dataset), so the break is a magnitude break, not a kind break -- the SIZES differ by 5x to 6x.

### METRIC -- BREAKS

> \rowcolor{lightgray} \textbf{\method} & \textbf{98.9} $\pm$ 0.1 & \textbf{89.4} $\pm$ 0.5 & \textbf{93.8} $\pm$ 0.4 & \textbf{92.7} $\pm$ 0.5 & \textbf{84.4} & \textbf{99.3} \\

*locator*: arXiv:2305.01140v1, icml2023.tex:491 (label tab:qm9_results, row GeoLDM) and the metric definitions at icml2023.tex:532; source file sha256 b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, e-print sha256 b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23

GeoLDM's QM9 row reports FOUR observables -- atom stability, molecule stability, validity and validity x uniqueness -- and the cell this record's Reading A names as the band's lower endpoint is the 'validity x uniqueness' one, i.e. a per-molecule RDKit-sanitisable AND non-duplicate rate ('we report validity and uniqueness metrics, which are the percentages of valid (measured by RDKit) and unique molecules among all the generated compounds'). The paper states NO connectivity or fragment criterion for the QM9 metrics, so the cell is not a connected-component count. THIS IS A STATEMENT ABOUT GeoLDM's OWN CELL; it does not settle which cell the repo's band names -- that contest is still open and is recorded under `band_provenance`.

### POST-PROCESSING -- DOES-NOT-BREAK

> the model occasionally generates disconnected components, as shown in the rightest column of \cref{app:fig:vis-drug}, which happens more often when trained on the large molecule DRUG dataset. However, this phenomenon actually is not a problem and is common in all non-autoregressive molecule generative models~\cite{zang2020moflow,jo2022score}, and can be easily fixed by just filtering the smaller components.

*locator*: arXiv:2305.01140v1, icml2023.tex:917; source file sha256 b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, e-print sha256 b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23

*recipe verdict*: **NOT-APPLICABLE** -- NOT APPLICABLE -- GeoLDM's QM9 pipeline has NO post-processing step. The paper's only mention of filtering is the DRUG-side remark 'can be easily fixed by just filtering the smaller components', which is a REMEDY PROPOSED for the reader, not a procedure applied to the reported numbers. The paper's source contains the word 'filter' exactly once, in that sentence.

ON THE POST-PROCESSING AXIS, GeoLDM's OWN reported QM9 numbers are on the same footing as ours: no largest-fragment retention, no explicit hydrogens, no UFF step. THE READING THAT THEY ARE RAW IS AN ARGUMENT FROM SILENCE plus ONE sentence, and it is labelled as such here: the paper never says 'no post-processing', it simply never describes a post-processing step, and the single 'filtering' sentence is a DRUG-side observation ('happens more often when trained on the large molecule DRUG dataset') whose remedy is proposed, not applied. THIS VERDICT IS ABOUT GeoLDM's OWN PROTOCOL. It says nothing about the band's provenance, and it says nothing about the band's other endpoint.

### HYDROGENS -- NOT-STATED

**NOT-STATED: the paper's ONLY mention of hydrogens is the QM9 DATASET description ('limited to 9 heavy atoms (29 atoms including hydrogens)', icml2023.tex:517); it states NO hydrogen convention for the molecules it GENERATES, and it says nothing about an explicit-H step.**

> QM9 contains 3D structures together with several quantum properties for 130k small molecules, limited to 9 heavy atoms (29 atoms including hydrogens).

*locator*: arXiv:2305.01140v1, icml2023.tex:517 (the QM9 dataset paragraph); source file sha256 b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b, e-print sha256 b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23

The paper does not state a hydrogen convention for its generated molecules, so the cell cannot be compared against ours on that axis. Recorded as NOT-STATED rather than guessed.

Caveats the source states about itself:
* tab:qm9_results caption: 'Note that, for DRUG dataset, molecule stability and uniqueness metric are omitted since they are nearly $0\%$ and $100\%$ respectively for all the methods.' (icml2023.tex:468)
* tab:qm9_results caption: 'Compared with previous methods, the latent space with both invariant and equivariant variables enables \method to achieve up to 7\% improvement for the validity of large molecule generation.' (icml2023.tex:468)

Other observations:
* GEO-LDM'S OWN PAPER NAMES THE ARTEFACT. 'the model occasionally generates disconnected components ... this phenomenon actually is not a problem and is common in all non-autoregressive molecule generative models, and can be easily fixed by just filtering the smaller components.' (icml2023.tex:917) The remedy is PROPOSED, not applied: the paper's source contains the word 'filter' exactly once, in that sentence, and the numbers in tab:qm9_results are the unfiltered ones. The same sentence says the disconnection 'happens more often when trained on the large molecule DRUG dataset', i.e. it is a DRUG-side observation, not a QM9-side correction.
* THE TWO SOURCE TABLES DISAGREE ON THE SAME OBSERVABLE, AND THE DISAGREEMENT IS ANNOTATED HERE. GeoLDM's OWN paper prints Valid = 93.8 +/- 0.4 for its own row (icml2023.tex:491, sha256 b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b); HLTF prints the GeoLDM column's Val as 93.6 +/- 0.2 in its Table 1(a) (tables/qm9_benchmarks_og.tex:38, sha256 1c81ad499093f9d20d512e29787e934ab68c6d32dc5bb3b791970830c05c8f39) and again in its Table 1(b) (:63), and its Table 1 caption states 'Baselines reproduced from GCDM Table 1', i.e. the column is a TRANSCRIPT. NEITHER PAPER FLAGS THE DIFFERENCE. BOTH NUMERALS ARE ON DISK IN THIS REPO: the Wf-A175 frontier comparability record carries GeoLDM's own 93.8 +- 0.4 at the JSON PATH /sources/4/metrics/0/value (and in /sources/4/metrics/0/evidence_quote) and HLTF's GeoLDM-column Val 93.6 +- 0.2 at the JSON PATH /sources/11/metrics/2/evidence_quote, both in molmetal/reports/wfa175_frontier_comparability/sources.json (sha256 2e665cb217759a4099c83716a8f27829797f2401c6c630adaeb1b4cfbd6e3c78, as measured on 2026-11-02). What the Wf-A175 record does NOT do is FLAG the two as a disagreement -- its report.md (sha256 170f642846192837c334d45bb6a006414bf1b53a7089c8907f4061ff0d0f10e0, as measured on 2026-11-02) calls them 'the same GeoLDM row' without noticing the differing Val, in the sentence '(Table 1(a), line 38, carries the same GeoLDM row: `GeoLDM & 98.9+-0.1* & 89.4+-0.5* & 93.6+-0.2 & 92.7+-0.5`.)'. THE LINE NUMBERS THIS REPLACES ARE NOT CITED ANY MORE: a line number is a PERISHABLE locator in an artefact that is regenerated and grows, so the JSON PATH plus the sha256 above is the durable locator. NEITHER NUMERAL IS A BAND ENDPOINT: under Reading A the endpoints are 92.7 / 94.0 and under Reading B they are 92.98 / 94.33, so 93.8 and 93.6 must never be published as band endpoints.
* THE RECORD'S OWN READING OF THE LOWER ENDPOINT IS UNCHANGED BY THIS ENTRY. Adding GeoLDM as a FIRST-CLASS source means the 92.7 % cell is now substantiated by the originating paper's OWN e-print (arXiv:2305.01140v1), not only by a transcript of HLTF's table. It does NOT settle which cell the repo's band names: the two competing readings (and the catalogue's third) remain exactly as they were.

NOT-RETRIEVED (version-level):
* **the arXiv abs-page HTML rendering of the paper** -- https://arxiv.org/abs/2305.01140v1 (HTTP None). Reason: the METADATA was taken from the arXiv Atom API record (title, authors, published date) and the CONTENT from the e-print source tarball; the abs-page HTML was not fetched in this session Consequence: every GeoLDM cell above is stated for the e-print SOURCE of v1 (the .tex read verbatim out of the tarball), which is the more precise artefact than a rendered page
* **the ICML 2023 proceedings version (PMLR)** -- not fetched in this session (HTTP None). Reason: the arXiv record carries NO journal-ref, NO comment and NO DOI, so no proceedings URL was pinned; the venue is taken from the paper's own ICML 2023 class file Consequence: the numbers above are stated for the arXiv e-print of v1; the proceedings version was not compared

## HLTF

HLTF -- Hierarchy-Guided Latent Topology Flow (explicit bond topology + 3D coordinates)

* arXiv **2603.27113** (v1 (only version on arXiv; published 2026-03-28)) -- Hierarchy-Guided Topology Latent Flow for Molecular Graph Generation
* retrieved 2026-11-01 from `https://arxiv.org/html/2603.27113v1`
* venue: ICLR 2026 AI4Mat Workshop (accepted) -- verified from the arXiv abs-page comments field: '22 pages, 2 figures, 6 tables. Accepted to ICLR 2026 AI4Mat Workshop'
* representation: 3D explicit bond topology (bond types GENERATED, not inferred); unconditional; QM9 and GEOM-DRUGS
* reported band: **QM9 Val&Uniq 92.9 +/- 0.2 ; PB-Valid 94 +/- 0.5 ; GEOM-DRUGS Valid 85.5 % RAW -> 92.2 % after post-processing** (Table 1(a), Table 1(b) (QM9) and Table 2 (GEOM-DRUGS)); sample nature: training-distribution (unconditional), NOT held-out

### DENOMINATOR -- BREAKS

> On QM9 we report atom stability (AS), molecule stability (MS), RDKit validity (Val), and Val&Uniq on 10,000 unconditional samples.

*locator*: arXiv:2603.27113v1, Section 4.1 'Benchmarks, protocols, and metrics'; Table 2's denominator is 100,000 samples on GEOM-DRUGS (same paragraph)

10,000 unconditional QM9 samples (Table 1) and 100,000 unconditional GEOM-DRUGS samples (Table 2), from a model trained on the TRAINING split. Ours is one decode of one pool of 2,112 or 12,669 rows. The protocol is the same shape as the other frontier papers' (training-distribution sampling), which is why the break is a magnitude break, not a kind break -- the SIZES differ by 5x to 6x.

### METRIC -- BREAKS

> PB-Valid (%) | 93.1 ± 0.4 | 91.9 ± 0.5 | 94 ± 0.5 *

*locator*: arXiv:2603.27113v1, Table 1(b) metric list (AS / MS / Val / Val&Uniq / Novel / PB-Valid) and the Section 4.3 'Why PoseBusters matters here' paragraph

UNDER BOTH PROVENANCE READINGS the band's UPPER endpoint is NOT a connectivity number: under Reading A it is HLTF's own PB-Valid figure of 94 +/- 0.5 %, a POSE VERDICT -- arXiv:2603.27113v1 Table 1(b) reports SIX observables (atom stability (AS), molecule stability (MS), RDKit validity (Val), Validity&Uniq, Novelty and PoseBusters validity (PB-Valid)), which is a TABLE-ROW COUNT and NOT a count of PoseBusters checks; HLTF states NO CHECK COUNT for its PB-Valid cell (NOT-STATED-IN-SOURCE): it cites PoseBusters (Buttenschoen et al. 2024) and says only that PoseBusters 'applies stricter structural checks than RDKit'. The SIX-check PoseBusters set is EF-TALFM's, not HLTF's: arXiv:2609.08333v4 Section A.1.4 counts 'all six PoseBusters sanity checks used here' and lists them (Reasonable bond lengths, Reasonable bond angles, Aromatic ring flatness, Double bond flatness, Reasonable internal energy, No internal steric clash), and its Table 3(b) reports the joint row 'All PoseBusters checks'; under Reading B the band's upper endpoint is EF-TALFM's post-processed `N. & U. & S. = 94.33 %` companion cell. The LOWER endpoint is the CONTESTED one: Reading A names GeoLDM's Validity&Uniq 92.7 % (a sanitisation-and-duplication rate, `post_processed: false`), Reading B names EF-TALFM's RAW connectivity cell `C. = 92.98 %` (counted BEFORE the largest-component retention). WHICH reading the repo's band follows is NOT settled on disk -- the two readings do not even agree on the numerals (92.7/94.0 vs 92.98/94.33) -- so it may NOT be published that 'neither endpoint is a connected-component count': that is true of the UPPER endpoint under both readings, but the LOWER endpoint IS a connected-component count under Reading B. Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. The two readings are recorded under `band_provenance`.`competing_reading_and_what_this_does_not_settle`.

### POST-PROCESSING -- BREAKS

> On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation Rappe et al. (1992) ).

*locator*: arXiv:2603.27113v1, Section 4.1; Table 2 (the '+PP' rows); Section 4.4 'What changes with post-processing'

*recipe verdict*: **STATED** -- standardized post-processing = largest fragment + explicit hydrogens + UFF relaxation, cited as Buttenschoen et al. (2025) -- the SAME protocol family as our own post-processing arm (which cites arXiv:2505.00518 by the same authors).

THE FRONTIER MEASURES THE POST-PROCESSING AXIS ITSELF, IN ONE TABLE, FOR EVERY MODEL. HLTF Table 2 reports the raw and the post-processed number side by side for six models: EQGAT-diff 59.7 -> 84.2, FlowMol 59.8 -> 84.2, GCDM 0.2 -> 95.2, GeoLDM 2.9 -> 69.6, SemlaFlow 87.5 -> 93.1, HLTF 85.5 -> 92.2 (Valid, GEOM-DRUGS, 100,000 samples). The post-processing is worth +6.7 pp on the best model and +95.0 pp on the worst. HLTF also states the recovery fraction it defines -- 'converting ~ 46 % of the pre-PP invalid samples under the same pipeline', computed as (92.2 - 85.5) / (100 - 85.5) -- which is exactly the quantity our own post-processing arm can be reported on. ON OUR SIDE the same recipe family moves 8.51 % -> 100 % for connectivity alone, with the UFF third step INAPPLICABLE to our metal corpus (uff_ok_rate_over_postprocessed = 0.0). So the frontier's own table shows that a published '92-95 %' validity number can sit 6.7 to 95 points above the same model's raw number: the post-processing axis is not a rounding detail, it IS most of the number.

### HYDROGENS -- BREAKS

> On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation Rappe et al. (1992) ).

*locator*: arXiv:2603.27113v1, Section 4.1 (the post-processing adds explicit hydrogens) and Section 4.2

The post-processing step adds EXPLICIT hydrogens ('explicit hydrogens' in the recipe sentence), so the post-processed numbers are computed on explicit-H molecules. Ours is implicit-H. The paper's raw numbers are on whatever its generator emits; the paper does not state a hydrogen convention for the raw arm, so the raw arm is implicit-H by construction (it generates a graph, and RDKit sanitisation implies implicit H). VERDICT IS 'BREAKS' because the published headline pair (85.5 % raw / 92.2 % PP) straddles the two conventions -- the two numbers we would have to compare against are not on the same convention as each other, let alone ours.

Caveats the source states about itself:
* Section 4.4 'What changes with post-processing': 'Standardized post-processing mainly relieves geometric strain (e.g., unrealistic bond lengths/angles and local steric clashes) and cannot repair infeasible topology (e.g., invalid valence or disconnected graphs).'
* Section 4.4: 'we do not claim state-of-the-art headline success on this benchmark'.
* Section 4.2 'Explicit-topology strictness': HLTF is evaluated on the bonds it generates, a stricter requirement than inferred-bond evaluation; the matched-protocol study is left to future work.
* Section 4.4: 'Within each target...' is not applicable -- the paper notes the comparison to SemlaFlow is 0.9 points 'under the same PP pipeline'.

Other observations:
* HLTF IS THE BEST SINGLE SOURCE FOR THIS BRANCH: it is the only retrieved frontier paper that publishes the RAW and the POST-PROCESSED number for the SAME model under the SAME pipeline, in one table, and that defines a recovery fraction ((PP - raw) / (100 - raw)) that our own post-processing arm can be reported on with the identical formula.
* Its post-processing is the SAME protocol family as ours: largest-fragment + explicit hydrogens + UFF, cited to Buttenschoen et al. (2025) -- the same reference our own post-processing arm cites. So the two recipes are comparable in the one case where the frontier published both sides of it.
* HLTF's Table 2 also shows that the largest post-processing effect in the frontier is on the WORST raw model (GCDM 0.2 -> 95.2, i.e. 95.2 % of its invalid samples recovered). A published 'validity' number therefore says very little about the raw generator: the same table contains a 0.2 % and a 95.2 % validity rate for models whose post-processed numbers differ by 1.0 point.

NOT-RETRIEVED (version-level):
* **the workshop proceedings PDF / the supplementary tables (Table S1-S4) that the paper's numbers refer to** -- not fetched in this session (HTTP None). Reason: the arXiv HTML carries the main tables only; the supplementary material was not fetched Consequence: every HLTF cell above is stated for the arXiv v1 HTML. The per-dataset per-metric detail behind the tables is not quoted here.

## PAFlow

PAFlow -- Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number

* arXiv **2509.01486** (v2 (published 2025-09-01, updated 2025-11-04; comment: 'Accepted by NeurIPS 2025')) -- Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number
* retrieved 2026-11-01 from `https://arxiv.org/html/2509.01486v2`
* venue: NeurIPS 2025 (per the arXiv comment field)
* representation: 3D all-atom, TARGET-CONDITIONED (SBDD) generation on CrossDocked2020
* reported band: **NO connectivity or validity band is reported** (n/a -- Tables 1/3/4 report Vina Score, Vina Min, Vina Dock, High Affinity, QED, SA, Diversity); sample nature: held-out TEST PROTEINS (CrossDocked2020), 100 molecules per test protein

### DENOMINATOR -- NOT-STATED

**NOT-STATED: this paper reports no connectivity/validity rate, so it has no denominator for one. The denominator below is for its OWN (affinity) metrics, recorded for context only.**

*locator*: arXiv:2509.01486v2 -- no connectivity/validity denominator exists; Section 4.1 states '100 molecules are sampled for each test protein.' for its AFFINITY metrics

There is no connectivity/validity band in this paper, so there is no denominator for one. For the record, the paper's own metrics are sampled 100 molecules per test protein (and 1,000 across all test proteins in the ablation), on HELD-OUT test proteins -- a genuinely different protocol from the unconditional training-distribution bands of the other three sources.

### METRIC -- NOT-STATED

**NOT-STATED: the paper reports NO connectivity metric, NO validity rate and NO sanitisation rate. The quote above is the evidence -- the evaluation axes are affinity and properties, and the quoted metric list names QED / SA / diversity, none of which is a connectivity count.**

> The generated molecules are evaluated from two aspects: binding affinity with the target protein and molecular properties.

*locator*: arXiv:2509.01486v2, Section 4.1 'Evaluation metrics'

The paper's evaluation metrics are AutoDock Vina affinity measures, High Affinity, QED, SA and diversity -- no connectivity, no sanitisation rate, no validity rate. MEASURED absence (exhaustive grep of the extracted full text): 'sanitiz' 0 hits; 'valid*' 6 hits, ALL of them 'validation loss' / 'validate' in the TRAINING sense; 'connect' 1 hit, in the knn-graph construction sentence, not a molecular count; 'fragment' 3 hits, all describing fragment-BASED generation as a PARADIGM ('fragment by fragment', 'molecular fragments', 'unrealistic fragments'), not a fragmentation metric; 'valency'/'valence' 0 hits; 'Vina' 59 hits. This is the answer to the branch's fourth-source question: a 4th graph flow-matching paper exists and does NOT report a connectivity/validity band at all.

### POST-PROCESSING -- NOT-STATED

**NOT-STATED: no post-processing recipe is described for any reported metric, because no molecular count is reported at all.**

*locator*: arXiv:2509.01486v2 -- no post-processing is described for any reported metric

*recipe verdict*: **NOT-APPLICABLE** -- NOT-APPLICABLE: there is no pre-count recipe to state, because the paper counts no molecules -- it reports docking scores, QED, SA and diversity.

Nothing in the paper describes a post-processing recipe applied before a molecular count, because no molecular count is reported. There is no 'largest fragment' step, no sanitisation step and no hydrogen-manipulation step to compare against ours.

### HYDROGENS -- NOT-STATED

**NOT-STATED: the paper calls itself 'a 3D all-atom Flow matching model' but states no hydrogen convention for any reported metric, and reports no connectivity/validity metric that a convention could apply to.**

*locator*: arXiv:2509.01486v2 -- the paper describes itself as 'a 3D all-atom Flow matching model' but states no hydrogen convention for any reported metric

The paper calls itself 'a 3D all-atom Flow matching model' in the abstract, which implies explicit hydrogens as nodes, but it reports no connectivity/validity metric, so the hydrogen convention of the counted molecule is not specified and cannot be compared.

Caveats the source states about itself:
* The paper's evaluation is on HELD-OUT test proteins, unlike the unconditional training-distribution bands of the other three sources -- so it is not a like-for-like frontier comparator either.

## YuelBond

YuelBond -- a learned bond-type decoder for 3D molecules (NOT a molecule generator)

* DOI **10.1101/2025.05.06.652517 (preprint)** (no arXiv id) (bioRxiv v1 preprint of J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052) -- YuelBond: a bond-type decoder for reconstructing molecular bonds from atomic coordinates
* retrieved 2026-11-01 from `https://www.biorxiv.org/content/10.1101/2025.05.06.652517v1.full`
* venue: bioRxiv preprint (DOI 10.1101/2025.05.06.652517); published in J. Chem. Inf. Model. 2026, 66(2), 1003-1012 (doi:10.1021/acs.jcim.5c03052) per the repo's molmetal/reports/wf_cfm_frontier_research/research_phase1b.md -- the published version was NOT fetched in this session
* representation: a BOND-TYPE CLASSIFIER over given 3D coordinates -- it does not generate molecules and does not generate coordinates
* reported band: **F1 98.4 % on standard 3-D structures ; F1 92.7 % on distorted CDGs** (the abstract and the CDG bonds-reconstruction results paragraph); sample nature: a held-out TEST SET of 1,000 molecules with synthetic coordinate noise (sigma = 0.2 A) -- the perturbation is applied BY THE PAPER to a real test set, not produced by a generative model

### DENOMINATOR -- BREAKS

> Out of 1,000 test molecules with noisy coordinates, RDKit was able to process only 217, while it failed on the remaining 783 molecules.

*locator*: bioRxiv 10.1101/2025.05.06.652517v1, the RDKit comparison paragraph and the 'Performance Evaluation ... on the CDG Bonds Reconstruction Task' section

1,000 held-out test-set molecules with SYNTHETIC coordinate noise (sigma = 0.2 A) applied by the paper itself. This is not a generative-model sampling population at all: the paper perturbs REAL test molecules. Ours is one decode of one pool. Note the repo's own correction (Wf-A146): the earlier wording '78.3 % failure rate' was wrong -- the paper prints 783 failures out of 1,000.

### METRIC -- BREAKS

> The model’s performance is evaluated using standard multi-class classification metrics: accuracy, precision, recall, and F1-score. They are computed on a per-molecule basis and aggregated across the test set.

*locator*: bioRxiv 10.1101/2025.05.06.652517v1, Methods -- 'The model's performance is evaluated using standard multi-class classification metrics...'

THE BAND'S 92.7 % IS A BOND-TYPE CLASSIFICATION F1, NOT A PER-MOLECULE RATE. The model's task is to label each EXISTING bond as single / double / triple / aromatic, given coordinates. The per-bond-type breakdown in the paper (single-bond F1 83.3 %, double 56.3 %, triple 59.9 % on the noisy cohort) makes this explicit. There is no per-molecule connectivity count anywhere in the paper. A per-bond F1 and a per-molecule connected-component rate are not two estimates of one quantity -- they are two different observables, and placing our 8.51 % beside a bond-type F1 asserts a comparability that does not exist.

### POST-PROCESSING -- NOT-STATED

**NOT-STATED: YuelBond is a DECODER, not a counter. It has no 'before the count' pipeline of its own, so there is no post-processing recipe to state. (Its 'post-processing' in the repo's usage is the DECODER ITSELF replacing RDKit's rule-based xyz2mol.)**

*locator*: bioRxiv 10.1101/2025.05.06.652517v1 -- the paper describes a bond-reconstruction model with no molecular-counting pipeline

*recipe verdict*: **NOT-APPLICABLE** -- NOT-APPLICABLE: there is no pre-count recipe because there is no per-molecule count to pre-process. The paper's own post-processing IS the learned decoder.

Recorded as NOT-STATED rather than BREAKS because the axis does not exist for this source, not because the paper declined to describe a recipe it has.

### HYDROGENS -- NOT-STATED

**NOT-STATED: the paper reports no per-molecule metric, so no hydrogen convention is stated or implied for any counted molecule.**

*locator*: bioRxiv 10.1101/2025.05.06.652517v1 -- no per-molecule metric is reported

The paper's inputs are real 3-D structures (which do carry hydrogens) and its outputs are bond labels, not molecules, so there is no counted molecule whose hydrogen convention could differ from ours.

Caveats the source states about itself:
* The paper's own per-bond-type breakdown shows the aggregate F1 is a mixture of very different per-type scores (single 83.3 %, double 56.3 %, triple 59.9 % on the noisy cohort).
* The paper states the RDKit baseline is handicapped by the distortion it introduces, not by a chemistry failure: 'This limitation arises because RDKit relies on chemically valid input geometries and rule-based heuristics that become unreliable when the structure is distorted.'

Other observations:
* TWO DIFFERENT PAPERS BOTH PRODUCE A '92.7 %', AND NEITHER IS A CONNECTIVITY RATE. (i) GeoLDM's Validity&Uniq on QM9 = 92.7 % +/- 0.5 (HLTF Table 1) is a sanitisation-and-duplication rate; (ii) YuelBond's bond-type F1 on noisy CDGs = 92.7 % is a classification score. The repo's band was assembled from a literature scan; this branch did NOT establish which of the two was the intended endpoint, and does not claim to. What IS established: both candidates are NOT connectivity rates.

NOT-RETRIEVED (version-level):
* **the published J. Chem. Inf. Model. version (doi:10.1021/acs.jcim.5c03052) and its supplementary Tables S1-S5** -- not fetched in this session (HTTP None). Reason: the branch brief supplied the DOI; the bioRxiv preprint was fetched instead Consequence: every YuelBond cell above is stated for bioRxiv v1. If the published version changed the protocol, these cells are stale with respect to it.

## Our side (the anchor)

* value: **8.51 % (0.08509630565203663)** at `metrics/by_metric/cfm_fragmentation_audit.json` jsonpath `/blind_validity_signature/3/single_fragment_ratio`
* definition: n_single / samples drawn, where n_single = the pool rows whose decoded molecule has exactly ONE RDKit fragment, measured at the RAW emitted SMILES after RDKit MolFromSmiles parse (sanitisation) and Chem.GetMolFrags -- NO post-processing.
* post-processing: none (raw)
* hydrogen convention: implicit (RDKit default after MolFromSmiles; no AddHs step)
* provenance: this branch did NOT re-measure it; read from disk in this session. the population reconciliation and the definition family (F1_pool_e2e / F2_pool_cond) are documented in molmetal/reports/wfa158_population_reconciliation/census.md (a SIBLING Wf-A158 branch artifact, read-only here). This branch did not re-derive it and does not claim to.

## Where the repo's band comes from

* band as quoted in the repo: **92.7-94 %**
* finding: READING-QUALIFIED: the published 92.7-94 % band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. INDEPENDENTLY OF WHICH READING HOLDS, the frontier publishes the post-processing axis itself -- six models, raw and post-processed side by side, deltas from 5.6 to 95.0 pp (see `band_provenance`.`post_processing_evidence_from_the_frontier_itself`).

Where the repo states it (read-only for this branch):
* metrics/by_metric/wfa155_ligand_connected_rate.json -> /comparability/frontier_band
* metrics/by_metric/wfa155_metal_coordination_sphere.json -> /comparability_statement
* docs/HONEST_NEGATIVE_CATALOG.md §6.264 (read-only for this branch)
* molmetal/reports/wf_cfm_frontier_research/research_phase1b.md (read-only for this branch)

### The band's source cells are CONTESTED on disk (reading-qualified)

* status: READING-QUALIFIED -- the two readings below are BOTH on disk and NEITHER is settled.  The band as quoted is EXTERNAL to Mol-Metal, is NOT one homogeneous quantity, is NOT on our scale, and its source cells are NOT settled on disk.
* canonical statement: The published band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.

* reading A -- locator: molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `band_provenance`
* reading B -- locator: molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `sources`.`EF-TALFM`; EF-TALFM arXiv:2609.08333v4, Table 3(a)

### endpoint 92.7 %

* **candidate_a** -- GeoLDM, as tabulated by HLTF (arXiv:2603.27113v1)
  * what it is: Validity & Uniqueness on QM9 -- a per-molecule RDKit-sanitisable AND non-duplicate rate
  * denominator: 10,000 unconditional QM9 samples
  * post-processed: False
  * locator: arXiv:2603.27113v1, Table 1(b) row 'Val&Uniq (%)', column 'GeoLDM'
  * quote: > Val&Uniq (%) | 92.7 ± 0.5 | 93.3 ± 0.0 * | 92.9 ± 0.2
  * status: VERIFIED against the primary source in this session
* **candidate_b** -- YuelBond (bioRxiv 10.1101/2025.05.06.652517v1)
  * what it is: multi-class BOND-TYPE classification F1 on 1,000 coordinate-noised molecules (sigma = 0.2 A)
  * denominator: 1,000 held-out test molecules with synthetic noise applied by the paper
  * post-processed: False
  * locator: bioRxiv 10.1101/2025.05.06.652517v1, CDG bonds-reconstruction results
  * quote: > achieving average accuracy, precision, recall, and F1-score of 92.7%, 93.6%, 92.7%, and 92.7%, respectively
  * status: VERIFIED against the primary source in this session
* note: TWO PAPERS, TWO COMPLETELY DIFFERENT OBSERVABLES, THE SAME NUMBER. This branch did NOT establish which one the repo's band intended, and does not claim to. It established that BOTH are not connectivity rates.

### endpoint 94 %

* **candidate_a** -- HLTF (arXiv:2603.27113v1)
  * what it is: PoseBusters validity on QM9 -- a pose verdict over the PoseBusters check set.  HLTF states NO CHECK COUNT for this cell (NOT-STATED-IN-SOURCE): it cites PoseBusters (Buttenschoen et al. 2024) and says only that PoseBusters 'applies stricter structural checks than RDKit'.  The SIX-check PoseBusters set is EF-TALFM's (arXiv:2609.08333v4), not HLTF's.
  * denominator: 10,000 unconditional QM9 samples
  * post-processed: False
  * locator: arXiv:2603.27113v1, Table 1(b) row 'PB-Valid (%)', column 'HLTF (ours)'
  * quote: > PB-Valid (%) | 93.1 ± 0.4 | 91.9 ± 0.5 | 94 ± 0.5 *
  * status: VERIFIED against the primary source in this session
  * **candidate_a_what_it_is_superseded_2026_11_01** -- DATED, SUPERSEDED RENDERING (PRESERVED VERBATIM, not a live claim): PoseBusters validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles, steric clash, aromatic and double-bond flatness, internal energy)
  * **candidate_a_what_it_is_superseded_note_2026_11_01** -- DATED, SUPERSEDED RENDERING (PRESERVED VERBATIM, not a live claim): 2026-11-01 residual repair: the rendering above is SUPERSEDED.  It attributed a SIX-check PoseBusters set to HLTF.  HLTF states NO check count (NOT-STATED-IN-SOURCE); the six-check set is EF-TALFM's (arXiv:2609.08333v4, Section A.1.4).  PRESERVED VERBATIM per AGENTS.md §5.5 rule #6.
* note: A pose-verdict rate, NOT a connectivity rate.  The check COUNT is NOT-STATED-IN-SOURCE for HLTF.  The SIX-check PoseBusters set that this record previously attributed to HLTF is EF-TALFM's: arXiv:2609.08333v4 Section A.1.4 counts 'all six PoseBusters sanity checks used here' and lists them (Reasonable bond lengths, Reasonable bond angles, Aromatic ring flatness, Double bond flatness, Reasonable internal energy, No internal steric clash), with the joint row 'All PoseBusters checks' in its Table 3(b).  In EF-TALFM's OWN list, Connectivity is a SEPARATE component metric and NOT one of the six -- so the 94 % is not a connectivity rate; but that list is EF-TALFM's, not HLTF's.

### The frontier publishes the post-processing axis itself

* source: arXiv:2603.27113v1, Table 2 (GEOM-DRUGS, 100,000 samples)
* recipe: largest fragment + explicit hydrogens + UFF relaxation, cited to Buttenschoen et al. (2025)

| model | raw (%) | post-processed (%) | delta (pp) |
|---|---|---|---|
| EQGAT-diff | 59.7 | 84.2 | 24.5 |
| FlowMol | 59.8 | 84.2 | 24.4 |
| GCDM | 0.2 | 95.2 | 95.0 |
| GeoLDM | 2.9 | 69.6 | 66.7 |
| SemlaFlow | 87.5 | 93.1 | 5.6 |
| HLTF | 85.5 | 92.2 | 6.7 |

THE FRONTIER PUBLISHES THE POST-PROCESSING AXIS ITSELF. Six models, one table, raw and post-processed side by side, under one pipeline. The published 'validity' number differs from the raw number by 5.6 to 95.0 points depending on the model. HLTF also defines the recovery fraction (PP - raw) / (100 - raw) and reports ~ 46 % for itself. The SAME formula applied to our own post-processing arm gives (100 - 8.50963) / (100 - 8.50963) = 100 % (i.e. (1 - 0.0851) / (1 - 0.0851) in fractions), because LargestFragmentChooser keeps exactly one fragment by construction -- and the SAME file records uff_ok_rate_over_postprocessed = 0.0, so our 100 % is reached WITHOUT the UFF third step that the frontier recipe uses.

* the same recovery formula on our side: `(post_processed - raw) / (1 - raw)  [both terms as FRACTIONS] == (pp% - raw%) / (100 - raw%)  [the same formula in PERCENT]`
  * raw (fraction) = 0.08509630565203663; post-processed (fraction) = 1.0; recovery fraction = **1.0** (fractions: raw = 0.0851..., post_processed = 1.0 (= 100 %))
* caveat: 1.0 BY CONSTRUCTION -- LargestFragmentChooser cannot leave a multi-fragment molecule. It is a property of the operation, not a quality claim, and the same file records fraction_of_raw_mass_discarded = 0.22474803194442472 and a BIMODAL survivor shape (a drug-sized molecule or a <= 2-heavy-atom speck).

* what this does not settle: This branch did NOT establish which paper the repo's band was assembled from, because the band was assembled by a literature scan whose selection rule is not recorded on disk. It DID establish, from the primary sources, that the candidate endpoints listed in this block are validity / uniqueness / pose / bond-classification numbers rather than connectivity rates -- and that the frontier's own literature publishes the post-processing effect as being worth up to 95 percentage points. It did NOT establish WHICH cells the band's endpoints name: the band's source cells are NOT settled on disk (see `band_provenance`.`competing_reading_and_what_this_does_not_settle`), and under the EF-TALFM reading the band's lower endpoint IS a raw connectivity count (`C. = 92.98 %`).

## What this branch did NOT establish

* The retitled TMLR version of GGFlow (OpenReview K8RlXtMgzo) was NOT retrieved (HTTP 403 browser challenge); only arXiv v1 was read.
* The Digital Discovery 2026 5(5):2052-2066 publication of FlowMol3 was NOT independently verified; only arXiv v1 was read.
* Author lists for GGFlow and PAFlow were not extracted.
* The FlowMol3 paper does not state its own row's denominator; the 5,000 figure comes from the authors' public evaluation harness, not from the paper.
* GGFlow's 'valency correction or edge resampling' phase is named but never described in the paper; the recipe is NOT-RETRIEVED beyond the metric definition.
* The raw HTML/PDF of these papers was read on the retrieval date; a later revision of any source could change a cell.

## Retrieval log

| url | retrieved | status | what was extracted |
|---|---|---|---|
| `https://arxiv.org/html/2609.08333v4` | 2026-11-01 | 200 | EF-TALFM metrics definitions (A.1.4), Table 3/Table 4 + captions, the 92.98/98.25/98.21/94.33 cells, the Appendix-D reconstruction and largest-component-retention text, the heavy-atom-scaffold statement, the 10,000-sample denominators, the 92,221-connected sentence, and the bond-threshold rule in Appendix D Algorithm 4. |
| `https://arxiv.org/abs/2609.08333` | 2026-11-01 | 200 | title, authors, abstract (87.9 % unique-and-novel wording, PCQM4Mv2). |
| `http://export.arxiv.org/api/query?id_list=2609.08333,2508.12629,2411.05676,2509.01486` | 2026-11-01 | 200 | canonical ids, titles, published/updated dates, and the 'Accepted by NeurIPS 2025' comment for 2509.01486. |
| `https://arxiv.org/html/2508.12629v1` | 2026-11-01 | 200 | FlowMol3 Section 3.2.1 validity definitions, Table 1 (99.9+/-0.1 % Valid, 91.9+/-0.7 % PB-Valid), the Table 1 caption, the explicit-hydrogen training statements, the 5000-sample / 5-subset sampling protocol, the kekulization statement, and the caveat that validity is 'a necessary but insufficient condition'. |
| `https://arxiv.org/html/2411.05676v1` | 2026-11-01 | 200 | GGFlow Table 1 (Val. 100 / Val. w/o corr. 99.91 on QM9; 100 / 99.63 on ZINC250k), the Appendix-E validity metric definitions (including 'after valency correction or edge resampling' and the 'Validity w/o correction' baseline), the 10,000-sample statement, the QM9/ZINC250k dataset description, and the table-statistics row (4 node types). |
| `https://openreview.net/forum?id=K8RlXtMgzo` | 2026-11-01 | 403 | NOTHING -- the page returned a browser-verification challenge ('Verifying your browser'). The api2.openreview.net notes endpoint returned 403 ChallengeRequiredError. The retitled TMLR version of GGFlow was therefore NOT retrieved; the arXiv v1 (title 'Improving Molecular Graph Generation with Flow Matching and Optimal Transport') is used instead and is flagged as such. |
| `https://arxiv.org/html/2509.01486v2` | 2026-11-01 | 200 | PAFlow title/abstract, Section 4.1 evaluation metrics, the 100-molecules-per-test-protein sampling statement, and the CrossDocked2020 dataset statement -- and, by exhaustive grep, the ABSENCE of any connectivity/validity band. |
| `https://raw.githubusercontent.com/dunni3/FlowMol/main/flowmol/analysis/metrics.py` | 2026-11-01 | 200 | SampleAnalyzer.compute_validity -- the exact largest-fragment + sanitisation sequence behind '% Valid', the frac_connected counter, and the PoseBusters bust() call on the RAW molecule. |
| `https://raw.githubusercontent.com/dunni3/FlowMol/main/fm3_evals/baselines/compute_baseline_comparison.py` | 2026-11-01 | 200 | the SDF loader flags (sanitize=False, removeHs=False), the n_subsets split arithmetic, n_raw_samples, and the frac_valid_mols denominator rescaling. |
| `https://raw.githubusercontent.com/dunni3/FlowMol/main/flowmol/analysis/molecule_builder.py` | 2026-11-01 | 200 | build_molecule -- every sampled atom (hydrogens included) becomes an RDKit atom node. |
| `https://raw.githubusercontent.com/dunni3/FlowMol/main/flowmol/analysis/pb_config.yaml` | 2026-11-01 | 200 | the PoseBusters module list, ignore_hydrogens: True, and loading.add_hs: False. |
| `https://raw.githubusercontent.com/dunni3/FlowMol/main/fm3_evals/readme.md` | 2026-11-01 | 200 | the author's own evaluation command line (--n_mols=5000 --n_subsets=5). |
| `https://arxiv.org/html/2603.27113v1` | 2026-11-01 | 200 | HLTF Table 1(a) (QM9: AS 98.8, Val 93.8, Val&Uniq 92.9), Table 1(b) (GeoLDM Val&Uniq 92.7 +/- 0.5; HLTF PB-Valid 94 +/- 0.5), Section 4.1 (the QM9 10,000-sample and GEOM-DRUGS 100,000-sample denominators and the post-processing recipe 'largest fragment, explicit hydrogens, UFF relaxation'), Table 2 (the GEOM-DRUGS raw -> +PP pairs for EVERY model), and the 'What changes with post-processing' paragraph. |
| `https://arxiv.org/abs/2603.27113` | 2026-11-01 | 200 | authors (Urvi Awasthi, Alexander Arjun Lobo, Leonid Zhukov), the comments field ('22 pages, 2 figures, 6 tables. Accepted to ICLR 2026 AI4Mat Workshop') and the abstract, which restates 98.8 % AS / 92.9 % Val&Uniq / 94.0 % PB-Valid on QM9 and 85.5 % -> 92.2 % on GEOM-DRUGS. |
| `https://www.biorxiv.org/content/10.1101/2025.05.06.652517v1.full` | 2026-11-01 | 200 | the abstract (98.4 % F1 on standard 3-D structures; 92.7 % F1 on distorted CDGs), the CDG bonds-reconstruction results (accuracy 92.7 %, precision 93.6 %, recall 92.7 %, F1 92.7 % over 1,000 noisy molecules), the per-bond-type breakdown, the RDKit comparison (217 of 1,000 processed), and the Methods statement that the metrics are multi-class bond-type classification metrics. |
| `https://arxiv.org/e-print/2305.01140v1` | 2026-11-01 | 200 | icml2023.tex (sha256 b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b): the tab:qm9_results caption (:468), the tab:qm9_results GeoLDM row (:491 -- 98.9 +/- 0.1 / 89.4 +/- 0.5 / 93.8 +/- 0.4 / 92.7 +/- 0.5 / 84.4 / 99.3), the metric definitions (:532), the sample-size sentence (:549), the QM9 dataset paragraph (:517) and the disconnected-components remark (:917); the \method macro definition (:70) and the ICML 2023 class/keyword lines (:26, :131) for the venue. |
| `http://export.arxiv.org/api/query?id_list=2305.01140` | 2026-11-01 | 200 | title 'Geometric Latent Diffusion Models for 3D Molecule Generation', authors (Minkai Xu, Alexander Powers, Ron Dror, Stefano Ermon, Jure Leskovec), v1 published 2023-05-02, and the ABSENCE of a journal-ref, a comment field and a DOI. |

## Wf-A177 CORRECTION (2026-11-01) — the band reading is CONTESTED on disk

This section is APPENDED.  Per AGENTS.md §5.5 rule #6 the superseded renderings
above are PRESERVED VERBATIM and this correction is ADDITIVE + DATED: nothing
above this line was edited, removed, re-rendered or re-ordered.

**Superseded rendering 1 -- `frontier_matrix.md` (section "Where the repo's band
comes from" -> `band_provenance` -> `finding`), VERBATIM:**

> * finding: THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %); a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE CLASSIFICATION F1. None of the three is a connected-component count. Our 8.51 % is a connected-component count on the RAW emitted SMILES. The comparison the band invites is therefore not merely apples-to-oranges -- it is a comparison of two different observables, and the post-processing axis is a THIRD, separate axis on top of it.

**Superseded rendering 2 -- `frontier_matrix.md` (section "HLTF" -> "METRIC --
BREAKS" -> note), VERBATIM:**

> THE BAND'S UPPER ENDPOINT IS NOT A CONNECTIVITY NUMBER AT ALL. Table 1(b) reports SIX observables -- atom stability (AS), molecule stability (MS), RDKit validity (Val), Validity&Uniq, Novelty and PoseBusters validity (PB-Valid) -- and the 94 +/- 0.5 % figure is PB-Valid, a SIX-CHECK POSE VERDICT (bond lengths/angles, steric clash, aromatic and double-bond flatness, internal energy). The LOWER endpoint (92.7 %) is GeoLDM's Validity&Uniq -- a sanitisation-and-duplication rate. NEITHER is a connected-component count. Ours is a connected-component count and nothing else. This is a total METRIC break, not a partial one.

**THE CORRECTION (Wf-A177, 2026-11-01).** The published 92.7-94 % band is EXTERNAL to
Mol-Metal (it is a FRONTIER composite) and it is NOT one homogeneous quantity.
The repo holds TWO COMPETING provenance readings for which cells it names, and
they do not even agree on the numerals: (A) `frontier_matrix.json` ->
`band_provenance` (:51-89): GeoLDM's Validity & Uniq 92.7 % and HLTF's PB-Valid
94.0 % on QM9, both `post_processed: false` (plus YuelBond's per-bond-type
classification F1 92.7 % as a third candidate for the same numeral) -- this
reading reproduces the quoted numerals exactly but makes NO endpoint a
connected-component count; (B) the same file -> `frontier_axis_mapping` (:360,
:378-379), EF-TALFM arXiv:2609.08333v4: the band mixes EF-TALFM's OWN Table 3(a)
raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component
retention) with its post-processed N. & U. & S. companion cell (94.33 %) -- this
reading has the connectivity+post shape but does NOT reproduce the quoted
numerals (92.98 / 94.33). Under BOTH readings the band is external, is not one
homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a
connected-component count on the RAW emitted SMILES, and nothing else. The exact
source cells of the band are NOT settled on disk.

**The only claims that may be published (they hold under BOTH readings):**

1. The published 92.7-94 % band is **EXTERNAL** to Mol-Metal -- it is a FRONTIER
   composite, not our measurement.
2. It is **NOT one homogeneous quantity**.
3. It is **NOT on our scale** -- OUR 8.51 % IS a connected-component count on
   the RAW emitted SMILES, and nothing else.
4. The **source cells of the band are NOT settled on disk**.

**What this correction does NOT change.** The per-source protocol breaks
recorded above (DENOMINATOR / METRIC / POST-PROCESSING / HYDROGENS per paper)
are unaffected: they compare the frontier's STATED protocol against ours, not
the band's provenance. The definition of OUR anchor (8.51 %, RDKit
connected-component count on the RAW emitted SMILES, no post-processing) is
unaffected. What is superseded is the reading that the band's source cells are
SETTLED, and the rendering that asserted "NONE of the three is a
connected-component count" / "NEITHER is a connected-component count" as fact
rather than as one reading among two.

**Machine-readable record:** the same contest is recorded, ADDITIVELY, under
`band_provenance` -> `competing_reading_and_what_this_does_not_settle` in
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, and in
the SOURCE of the two producers
(`molmetal/scripts/wfa158_frontier_postproc_matrix.py`).

**Locator drift note (added by Wf-A177, 2026-11-01).** The line ranges cited in the
canonical statement above -- `:51-89` (Reading A) and `:360, :378-379` (Reading B) --
are the PRE-Wf-A177 line numbers of `frontier_matrix.json`.  The Wf-A177 key added at
`:52` of that file shifted every line at or after the old `:52` by +48.  Current
positions on disk: Reading A -> `band_provenance` `:50`, `band_as_quoted_in_the_repo`
`:51`, `endpoint_candidates` `:100-`, `headline_finding` `:137`; Reading B -> `sources`
`:380` -> `EF-TALFM` `:381`, the EF-TALFM `POST-PROCESSING` note `:408` (the
`'C. = 92.98 %'` cell), `extra_observations` `:426-427` (the
`'92.7-94 % band ... mixes the LOWEST connectivity'` sentence).  The ROBUST locators
are the JSON PATHS -- `band_provenance` and `sources`.`EF-TALFM` -- not the line
numbers.

## Wf-A180 CORRECTION (2026-11-01) — the contested renderings are SUPERSEDED (reading-safe)

This section is APPENDED.  Per AGENTS.md §5.5 rule #6 the superseded
renderings above are PRESERVED VERBATIM and this correction is ADDITIVE +
DATED: nothing above this line was edited, removed, re-rendered or re-ordered.
The pre-Wf-A180 bytes of this file are a byte-identical PREFIX of the file
that now carries this section.

**Locator defect, corrected (additively).** The Wf-A177 section above points
Reading B at `frontier_axis_mapping`.  That token is **NOT a key of
`frontier_matrix.json`** — it survives only inside a string value, so
pointing a reader at it is a DEFECT, not a locator.  The ROBUST locators are
the JSON PATHS `band_provenance` and `sources`.`EF-TALFM` — as this file's
own "Locator drift note" above already said.  The two copies of that pointer
inside `frontier_matrix.json` are corrected in place, each with the
superseded string preserved VERBATIM in a dated sibling key.

**Superseded rendering 1 — `band_provenance`.`headline_finding`, VERBATIM:**

> THE BAND IS NOT A CONNECTIVITY BAND. Its two endpoints are a VALIDITY-AND-UNIQUENESS rate (92.7 %) and a POSE-VALIDITY rate (94.0 %); a third paper in the same repo's scan quotes 92.7 % as a PER-BOND-TYPE CLASSIFICATION F1. None of the three is a connected-component count. Our 8.51 % is a connected-component count on the RAW emitted SMILES. The comparison the band invites is therefore not merely apples-to-oranges -- it is a comparison of two different observables, and the post-processing axis is a THIRD, separate axis on top of it.

**Superseded rendering 2 — `sources`.`HLTF`.`cells`.`METRIC`.`note`, VERBATIM:**

> THE BAND'S UPPER ENDPOINT IS NOT A CONNECTIVITY NUMBER AT ALL. Table 1(b) reports SIX observables -- atom stability (AS), molecule stability (MS), RDKit validity (Val), Validity&Uniq, Novelty and PoseBusters validity (PB-Valid) -- and the 94 +/- 0.5 % figure is PB-Valid, a SIX-CHECK POSE VERDICT (bond lengths/angles, steric clash, aromatic and double-bond flatness, internal energy). The LOWER endpoint (92.7 %) is GeoLDM's Validity&Uniq -- a sanitisation-and-duplication rate. NEITHER is a connected-component count. Ours is a connected-component count and nothing else. This is a total METRIC break, not a partial one.

**Superseded rendering 3 — `band_provenance`.`what_this_does_not_settle`, VERBATIM:**

> This branch did NOT establish which paper the repo's band was assembled from, because the band was assembled by a literature scan whose selection rule is not recorded on disk. It DID establish, from the primary sources, that every candidate endpoint of the band is a validity / uniqueness / pose / bond-classification number rather than a connectivity rate -- and that the frontier's own literature publishes the post-processing effect as being worth up to 95 percentage points.

**Why these three are superseded.** Rendering 1 asserts "THE BAND IS NOT A
CONNECTIVITY BAND" / "None of the three is a connected-component count";
rendering 2 asserts "NEITHER is a connected-component count" / "This is a
total METRIC break, not a partial one"; rendering 3 asserts that EVERY
candidate endpoint of the band is "rather than a connectivity rate".  All
three are TRUE under Reading A and FALSE under Reading B — under Reading B the
band's LOWER endpoint IS EF-TALFM's RAW connectivity cell
`C. = 92.98 %` — so none of them may be published as settled.

**THE READING-SAFE STATEMENT (replaces all three):**

> The published band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.

**The only claims that may be published (they hold under BOTH readings):**

1. The published 92.7-94 % band is **EXTERNAL** to Mol-Metal — it is a
   FRONTIER composite, not our measurement.
2. It is **NOT one homogeneous quantity**.
3. It is **NOT on our scale** — OUR 8.51 % IS a connected-component count on
   the RAW emitted SMILES, and nothing else.
4. The **source cells of the band are NOT settled on disk**.

**What this correction does NOT change.** The per-source protocol breaks
recorded above (DENOMINATOR / METRIC / POST-PROCESSING / HYDROGENS per paper)
are unaffected: they compare the frontier's STATED protocol against ours, not
the band's provenance.  The definition of OUR anchor (8.51 %, RDKit
connected-component count on the RAW emitted SMILES, no post-processing) is
unaffected.  The Wf-A177 section above is itself unaffected — it is the
record of the contest, and its "Locator drift note" already named the JSON
PATHS as the robust locators.

**Producer state.** The producer
(`molmetal/scripts/wfa158_frontier_postproc_matrix.py`) now EMITS the
reading-safe rendering for all three sites AND emits
`band_provenance`.`competing_reading_and_what_this_does_not_settle` (the
canonical statement, the two readings and the four invariants, addressed by
JSON PATH and never by line number), so a re-run reproduces this correction
instead of dropping it.

## Producer/record reconciliation (2026-11-01) -- the producer now REPRODUCES this record byte-for-byte

This section is APPENDED.  Per AGENTS.md §5.5 rule #6 nothing above this line was deleted: the two dated sections above are PRESERVED VERBATIM, and every superseded rendering they quote is still the record of what the body said before this repair.

**The body above was re-rendered by this repair.**  The statements in the dated sections above that nothing above them was edited, removed, re-rendered or re-ordered were TRUE at their date and are SUPERSEDED by this section: the body was re-rendered from the record, which (a) replaced the three contested renderings with the reading-safe ones, (b) aligned the source order with the JSON's source order, and (c) added the band's contested-source section.  The superseded renderings themselves are preserved verbatim -- the three claim renderings quoted in the Wf-A177 and Wf-A180 sections above, and the eight live strings this repair changed, in the record's dated key `band_provenance`.`producer_reproduces_the_record_2026_11_01`.

**What changed.**  Eight live strings changed relative to the producer as it stood before this repair -- five renderings, two cross-reference descriptors and one provenance marker -- and nothing else changed: 0 numeric leaf moved and no leaf was deleted.  Seven of the eight differ from the artefact as it stood before this repair; the `wave` marker is the exception, because it was made to agree with the artefact.  The repair writes the producer's own values, after which the parity check reports 0 string leaves moved and 0 leaves only on disk.  Leaves were ADDED on both sides -- the 29 the artefact already carried and the producer did not emit, and the leaves of this dated key.

**Why.**  A producer that does not reproduce its own artefact silently deletes the history it cannot reproduce, and the deletion happens on the next ordinary re-run, with no signal.  Nothing compared the producer's output against the bytes on disk, so the divergence survived two waves.  The parity check the producer now runs in `--check-only` mode is the early warning; this key is the record.

**What changed in the record.**

* `band_provenance`.`headline_finding` -- now names the band explicitly ('92.7-94 %'); it read 'the published band's provenance'.  The superseded rendering is PRESERVED VERBATIM in this key's `superseded_renderings`.
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`status` -- now states the four invariants (external / not one homogeneous quantity / not on our scale / source cells not settled); it stated none of them.  The superseded rendering is PRESERVED VERBATIM in this key's `superseded_renderings`.
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`why_this_key_exists` -- re-worded: the previous wording asserted that the sibling keys 'state ONE provenance reading as if it were settled', which is no longer true after the reading-safe repair.  The superseded rendering is PRESERVED VERBATIM in this key's `superseded_renderings`.
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings[0]`.`locator` -- the stale line-number parenthetical is dropped; the live locator is the JSON PATH `band_provenance`, the same rule the reading-B locator already followed.  The superseded rendering is PRESERVED VERBATIM in this key's `superseded_renderings`.
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings[1]`.`note` -- now quotes the band as '92.7-94 %'; it read '92.7-94'.  The superseded rendering is PRESERVED VERBATIM in this key's `superseded_renderings`.
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`wave` -- now 'Wf-A177'; it read 'Wf-A180'.  The marker names the wave that CREATED this key, and the record on disk carries 'Wf-A177'.  The previous value ('Wf-A180') is quoted verbatim here so that nothing is silently rewritten.
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`cross_references[0]` -- the stale assertion 'the emitted string is UNCHANGED' is corrected: Wf-A180 DID replace the string it names.  The superseded string is PRESERVED VERBATIM in this key's `superseded_renderings`.
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`cross_references[1]` -- the same stale-assertion correction, and the quoted contested wording ('NEITHER is a connected-component count') is removed from the LIVE rendering: a live rendering may not assert it, only a dated one may preserve it.  The superseded string is PRESERVED VERBATIM in this key's `superseded_renderings`.

**The superseded renderings (PRESERVED VERBATIM).**

* `band_provenance/headline_finding` -- superseded:

> READING-QUALIFIED: the published band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. INDEPENDENTLY OF WHICH READING HOLDS, the frontier publishes the post-processing axis itself -- six models, raw and post-processed side by side, deltas from 5.6 to 95.0 pp (see `band_provenance`.`post_processing_evidence_from_the_frontier_itself`).

  The live rendering names the band ('92.7-94 %'); this one did not.

* `band_provenance/competing_reading_and_what_this_does_not_settle/status` -- superseded:

> READING-QUALIFIED -- the two readings below are BOTH on disk and NEITHER is settled.  This key does not replace, contradict or amend any sibling key; it records the contest (AGENTS.md §5.5 rule #6 -- the superseded renderings are PRESERVED VERBATIM and this record is ADDITIVE).

  The live status states the four invariants (external / not one homogeneous quantity / not on our scale / source cells not settled on disk); this one stated none of them.

* `band_provenance/competing_reading_and_what_this_does_not_settle/why_this_key_exists` -- superseded:

> The sibling keys of this object ('headline_finding', 'endpoint_candidates') state ONE provenance reading as if it were settled.  The repo holds TWO competing readings, and they do not even agree on the numerals.  A future reader of this file must therefore read every band claim through the lens of the two readings recorded here.

  This wording asserted that the sibling keys ('headline_finding', 'endpoint_candidates') 'state ONE provenance reading as if it were settled'.  That is no longer true after the reading-safe repair, which is why the live rendering was re-worded.

* `band_provenance/competing_reading_and_what_this_does_not_settle/readings[0]/locator` -- superseded:

> molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `band_provenance` (:51-89)

  The live locator is the JSON PATH `band_provenance` alone.  The line-number parenthetical above is STALE -- this object's own `line_number_note` says so and names the JSON PATHS as the robust locators.

* `band_provenance/competing_reading_and_what_this_does_not_settle/readings[1]/note` -- superseded:

> 92.98 / 94.33 would round to '93.0-94.3', NOT the quoted '92.7-94'.  This reading has the connectivity+post SHAPE but does NOT reproduce the quoted numerals.

  The live note quotes the band as '92.7-94 %'; this one read '92.7-94'.

* `band_provenance/competing_reading_and_what_this_does_not_settle/cross_references[0]` -- superseded:

> molmetal/scripts/wfa158_frontier_postproc_matrix.py -> BAND_PROVENANCE['headline_finding'] (the producer of the contested rendering; annotated in place by Wf-A177, the emitted string is UNCHANGED)

  The live cross-reference no longer asserts that 'the emitted string is UNCHANGED': Wf-A180 REPLACED the string it names with the reading-safe rendering, so that assertion had become false.  The superseded rendering it named is preserved verbatim in `band_provenance`.`headline_finding_superseded_2026_11_01`.

* `band_provenance/competing_reading_and_what_this_does_not_settle/cross_references[1]` -- superseded:

> molmetal/scripts/wfa158_frontier_postproc_matrix.py -> sources['HLTF'].cells['METRIC'].note (the sibling 'NEITHER is a connected-component count' rendering; annotated in place by Wf-A177, the emitted string is UNCHANGED)

  The live cross-reference no longer asserts that 'the emitted string is UNCHANGED' -- Wf-A180 REPLACED the string it names -- and it no longer quotes the contested 'NEITHER is a connected-component count' wording in a LIVE rendering: P4 forbids that, and only a DATED rendering may preserve it.  The superseded rendering it named is preserved verbatim in `sources`.`HLTF`.`cells`.`METRIC`.`note_superseded_2026_11_01`.

**The byte evidence.**

* markdown BODY (everything before the three dated sections): 67563 characters
* the Wf-A177 dated section: 5215 characters
* the Wf-A180 dated section: 5842 characters
* the record carries 1081 leaves, of which 108 are numeric
* numeric leaves moved by this repair: 0 -- the repair is TEXT-ONLY plus NEW dated keys, which is the publish gate for this repair.  The parity check measures it against the artefact on disk and prints it as 'numeric leaves moved'.
* the record JSON on disk: 180306 bytes, md5 `1758b35f6378f12cb8e66e3eb9181bf8`
* the markdown's OWN size and md5 are NOT quoted here: a file cannot contain its own hash.  They are recorded in the dated sibling file `frontier_matrix.parity_2026_11_01.json`.

**What this section does NOT change.**  The four invariants that hold under BOTH provenance readings are unchanged: the published 92.7-94 % band is EXTERNAL to Mol-Metal, it is NOT one homogeneous quantity, it is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else -- and the source cells of the band are NOT settled on disk.  The per-source protocol breaks, the quotes, the anchor and the retrieval log are untouched.

**The producer now reproduces this artefact.**  `molmetal/scripts/wfa158_frontier_postproc_matrix.py` now emits `frontier_matrix.json` and `frontier_matrix.md` byte-for-byte, and its `--check-only` mode prints the PARITY report -- byte-identical or not, numeric leaves moved, leaves only on disk, leaves only in the producer -- so the next divergence is caught before it can delete anything.

## Locator class + GeoLDM source entry (2026-11-02) -- the LIVE locator is a JSON PATH

This section is APPENDED.  Per AGENTS.md §5.5 rule #6 nothing above this line was deleted, edited, removed, re-rendered or re-ordered: the dated sections above are PRESERVED VERBATIM, and every superseded rendering they quote is still the record of what the body said before this repair.

**This DATED key records TWO closures.  (1) LOCATOR CLASS: the LIVE locators of this record are JSON PATHS; the line numbers that were presented as current positions are DEMOTED into dated sibling keys and are a RECORD OF DRIFT, not locators.  (2) The GeoLDM source entry: Reading A's lower endpoint (92.7 %) was present in this record only as a TRANSCRIPT of HLTF's table, with the ORIGINATING paper absent from `sources`; `sources`.`GeoLDM` is now a first-class source, built the same way as the other six.**

**The locator-class rule.**  A LINE NUMBER is a PERISHABLE locator: this record keeps growing, so every line number cited into it rots, and a note that asserts a set of CURRENT line numbers is STALE BY CONSTRUCTION.  The LIVE locator for any part of this record is its JSON PATH -- e.g. `band_provenance`, `sources`.`EF-TALFM`, `sources`.`GeoLDM` -- never a line number.  A line number may appear ONLY inside a DATED historical record that says, in the same place, that it is a dated historical snapshot.

**The demoted line numbers are PRESERVED VERBATIM in:**

* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`line_number_note_superseded_2026_11_01`
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings[0]`.`historical_line_numbers_measured_2026_11_01`
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings[1]`.`historical_line_numbers_measured_2026_11_01`

**The drift, measured on 2026-11-02.**

A DATED HISTORICAL MEASUREMENT, taken on 2026-11-02 against the artefact as published on 2026-11-01 (the pre-repair bytes).  It is a RECORD OF DRIFT and is STALE BY CONSTRUCTION: the ACTUAL positions it names moved again when this repair was published.  It is NOT a locator -- the LIVE locators are the JSON PATHS above.  Every position below was MEASURED by reading the file on disk, not copied from another document.

| cited position | old claim | actual position (measured 2026-11-02) | verdict |
|---|---|---|---|
| `band_provenance` | :50 | :50 | OK |
| `band_as_quoted_in_the_repo` | :51 | :51 | OK |
| `endpoint_candidates` | :100- | :107- | STALE |
| `headline_finding` | :137 | :144 | STALE |
| `sources` | :380 | :472 | STALE |
| `sources`.`EF-TALFM` | :381 | :473 | STALE |
| `sources`.`EF-TALFM`.`cells`.`POST-PROCESSING`.`note` | :408 | :500 | STALE |
| `sources`.`EF-TALFM`.`extra_observations` | :426-427 | :518-519 | STALE |

*6 of the 8 cited positions were WRONG when the drift was measured on 2026-11-02.  This is why the line numbers are DEMOTED rather than corrected: correcting them would only move the rot.*

**The GeoLDM source entry.**

* what it is: `sources`.`GeoLDM` -- arXiv:2305.01140v1, 'Geometric Latent Diffusion Models for 3D Molecule Generation'.  It is the ORIGINATING paper of Reading A's lower endpoint (92.7 %); before this entry the 92.7 % was present in this record only as a TRANSCRIPT of HLTF's table, so the record could not say where the 92.7 % came from.
* locator class: JSON PATH (`sources`.`GeoLDM`) -- never a line number.  The locators that point INTO GeoLDM's own source file (`icml2023.tex:<line>`) name the EXTERNAL primary source and are pinned by that source file's sha256, not by this record's line numbers.
* F1: ESTABLISHED (Wf-A186, `findings`.`F1`): GeoLDM's OWN paper reports its own QM9 row as Valid & Unique = 92.7 +/- 0.5 over 10,000 samples, so 92.7 % is GeoLDM's OWN published cell and HLTF merely reproduces it identically.
* F2: ESTABLISHED (Wf-A186, `findings`.`F2`): HLTF's OWN wording scopes the standardized post-processing recipe to GEOM-DRUGS ONLY; the QM9 sentence in Section 4.1 (sections/experiments.tex:8) carries NO post-processing.  Both of Reading A's endpoints are QM9 Table 1 cells, so neither carries a post-processing clause.
* F3: ESTABLISHED (Wf-A186, `findings`.`F3`): GeoLDM's OWN paper DOES name the artefact -- 'the model occasionally generates disconnected components ... this phenomenon actually is not a problem'.  The remedy ('can be easily fixed by just filtering the smaller components') is PROPOSED, not applied, so GeoLDM's QM9 numbers are RAW.  THIS ENTRY THEREFORE DOES NOT RESTATE THE REFUTED 'not a connectivity band' FRAMING: a bare statement that the band is not a connectivity band would be false in the way X1 is false, because under Reading B the band's LOWER endpoint IS EF-TALFM's RAW connectivity cell C. = 92.98 %.
* F4: ESTABLISHED (Wf-A186, `findings`.`F4`): the two source tables DISAGREE on the same observable and no document on disk records it.  GeoLDM's OWN paper reports its QM9 Valid (%) as 93.8 +/- 0.4; HLTF's Table 1(a) / 1(b) tabulate the GeoLDM column's Val (%) as 93.6 +/- 0.2.  BOTH numerals are recorded in `sources`.`GeoLDM` and NEITHER is silently picked.  They are NOT band endpoints under either reading (under Reading A the endpoints are 92.7 / 94.0, under Reading B they are 92.98 / 94.33), so 93.8 and 93.6 must never be published as band endpoints.
* F5: ESTABLISHED (Wf-A186, `findings`.`F5`): EF-TALFM Table 3(a) is CONFIRMED VERBATIM -- its columns are UAE-3D / FlowMol / EF-TALFM, its caption fixes the 10,000 denominator, and connectivity is reported OF THE RECONSTRUCTED MOLECULE while the companion rates are computed AFTER the largest-component retention.
* why the band is still not settled: Adding the originating paper does NOT settle the band's provenance: the two competing readings recorded under `band_provenance`.`competing_reading_and_what_this_does_not_settle` are UNCHANGED, and neither may be published as the settled provenance of the band.

**The retrieval receipt for arXiv:2305.01140v1.**

* e-print url: https://arxiv.org/e-print/2305.01140v1
* e-print sha256: `b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23`
* e-print bytes: 655684
* retrieved (UTC): 2026-11-01T13:05:30Z
* title locator: icml2023.tex:88
* source file: `icml2023.tex` sha256 `b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b`
* source of the receipt: `molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json -> `retrieval``

**The same statements are APPENDED to `frontier_matrix.md` as a dated section, so the human-readable rendering carries the SAME reading-safe statement as this record.**

**What this repair does NOT change.**  The four invariants that hold under BOTH provenance readings are unchanged: the published 92.7-94 % band is EXTERNAL to Mol-Metal, it is NOT one homogeneous quantity, it is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else -- and the source cells of the band are NOT settled on disk.  The five pre-existing source entries are UNCHANGED; `sources`.`GeoLDM` was ADDED beside them.  The numeric leaves that existed before this repair did NOT MOVE; the repair ADDED new numeric leaves under `sources`.`GeoLDM`, so the 'TEXT-ONLY plus NEW dated keys' wording of the 2026-11-01 reconciliation section above describes THAT repair and not this one.

## Record residuals closed (2026-11-02) -- six defects, six repairs

This section is APPENDED.  Per AGENTS.md §5.5 rule #6 the DATED SECTIONS above are PRESERVED VERBATIM, and nothing above this line was deleted, removed or re-ordered.  The BODY above WAS re-rendered by Wf-A194 -- the passages that were re-rendered are listed below -- so the 'nothing above this line was ... re-rendered' assertion in the dated 2026-11-02 section above is PRESERVED VERBATIM there and is SUPERSEDED by this section.  This repair re-rendered exactly ONE live body string of its own -- `sources`.`GeoLDM`.`extra_observations` -- whose superseded rendering is preserved verbatim below.

**The dated key that records the SIX residual defects an adversarial verifier found in this record and in the producer that emits it, and the repair for each.  Every superseded rendering that a repair replaced is PRESERVED VERBATIM in a dated sibling key, so that a re-run of the producer reproduces both the correction and the record of what it replaced (AGENTS.md §5.5 rule #6).**

### D1 -- a wrong numeral and a false 'UNCHANGED' claim in the sources census

* **the defect:** The dated 2026-11-02 locator-class key emits, inside a LIVE string: 'disk.  The five pre-existing source entries are UNCHANGED; `sources`.`GeoLDM` was ADDED beside them.'  MEASURED on 2026-11-02 by diffing the pre-Wf-A194 published bytes of this record (2026-11-01 21:16) against the published bytes (2026-11-02 01:23:34), leaf by leaf and source entry by source entry: SIX source entries pre-existed the Wf-A194 repair, not five, and `sources`.`HLTF` was NOT unchanged -- its `cells`.`METRIC`.`note` was REWRITTEN and TWO leaves were ADDED under it (`cells`.`METRIC`.`note_superseded_2026_11_01_residual` and `cells`.`METRIC`.`note_superseded_2026_11_01_residual_note`).  The other five -- `EF-TALFM`, `FlowMol3`, `GGFlow`, `PAFlow`, `YuelBond` -- are byte-identical.  So the sentence is wrong on the numeral AND false on the claim.
* **the repair:** The sentence is LEFT VERBATIM -- it is rendered into the dated 2026-11-02 markdown section above, and a dated section is not rewritten in place (AGENTS.md §5.5 rule #6).  The CORRECTED sentence is emitted in `band_provenance`.`locator_class_and_geo_ldm_source_2026_11_02`.`the_sources_census_correction_2026_11_02`.`corrected_sentence`: 'The SIX pre-existing source entries are UNCHANGED except `sources`.`HLTF`, whose `cells`.`METRIC`.`note` was rewritten and to which TWO leaves were added; the other five are byte-identical.  `sources`.`GeoLDM` was ADDED beside them.'
* **the superseded rendering is preserved VERBATIM at:** `band_provenance`.`locator_class_and_geo_ldm_source_2026_11_02`.`the_sources_census_correction_2026_11_02`.`superseded_sentence_preserved_verbatim` (NAMED DATED CONSTANT `WFA197_D1_SUPERSEDED_SENTENCE_2026_11_02` in the producer)

### D2 -- a dated 2026-11-01 block edited IN PLACE, with its prior rendering preserved nowhere

* **the defect:** The dated key `band_provenance`.`producer_and_record_reconciled_2026_11_01` had its `divergence_from_the_producer` sentence EDITED IN PLACE by the 2026-11-01 residual repair, and the PRIOR rendering was preserved NOWHERE on disk -- an AGENTS.md §5.5 rule #6 violation.  MEASURED on 2026-11-02, the CURRENT text was ALSO inaccurate on its own terms: it said the producer emits a 'Wf-A180-SHAPED' object and that this record keeps a 'Wf-A177-SHAPED' object, but the producer's emitted object and this record's object under `band_provenance`.`competing_reading_and_what_this_does_not_settle` are BYTE-IDENTICAL.
* **the repair:** The sentence now states that the producer emits, under this key, the SAME object this record keeps, and that there is therefore NO divergence between the producer and this record on this key (MEASURED on 2026-11-02).  BOTH prior renderings are preserved VERBATIM: the 2026-11-01 rendering in `band_provenance`.`producer_and_record_reconciled_2026_11_01`.`divergence_from_the_producer_superseded_2026_11_02` (NAMED DATED CONSTANT `WFA197_D2_SUPERSEDED_DIVERGENCE_2026_11_02`), and the PRIOR pre-Wf-A194 rendering in `band_provenance`.`producer_and_record_reconciled_2026_11_01_prior_rendering_2026_11_02`.`divergence_from_the_producer` (NAMED DATED CONSTANT `WFA197_D2_PRIOR_RENDERING_2026_11_02`), recovered from the pre-Wf-A194 published bytes of this record (2026-11-01 21:16).
* **the superseded rendering is preserved VERBATIM at:** the two dated sibling keys named in the repair above

### D3 -- a false 'nothing above this line was re-rendered' assertion

* **the defect:** The dated 2026-11-02 markdown section asserts, verbatim: 'This section is APPENDED.  Per AGENTS.md §5.5 rule #6 nothing above this line was deleted, edited, removed, re-rendered or re-ordered: the dated sections above are PRESERVED VERBATIM, and every superseded rendering they quote is still the record of what the body said before this repair.'  THAT ASSERTION IS FALSE on its own terms: MEASURED by diffing the pre-Wf-A194 published bytes of this record (2026-11-01 21:16) against the published bytes (2026-11-02 01:23:34), the BODY was re-rendered -- seven live leaves carry a different value, two keys were RENAMED (their values PRESERVED VERBATIM), and the three self-referential byte-evidence counters were recomputed.  The dated 2026-11-01 section above it carries the WEAKER, TRUE assertion ('nothing above this line was deleted: the two dated sections above are PRESERVED VERBATIM').
* **the repair:** The FALSE assertion is LEFT VERBATIM in the dated 2026-11-02 section (a dated section is not rewritten in place), and the corrected assertion -- the one this section carries -- says exactly what is true: the DATED SECTIONS were preserved verbatim; the BODY above was re-rendered by Wf-A194; the re-rendered passages are listed below.
* **the superseded rendering is preserved VERBATIM at:** the dated 2026-11-02 markdown section above (left VERBATIM)

### R1 -- live cross-artefact line-number locators in `sources`.`GeoLDM`

* **the defect:** The live string at `sources`.`GeoLDM`.`extra_observations` cited the Wf-A175 frontier comparability record by LINE NUMBER -- report.md:40, report.md:133, report.md:138, sources.json:170 and sources.json:417.  MEASURED on 2026-11-02: all five are ACCURATE right now, but that artefact is regenerated and grows, so a LINE NUMBER is a PERISHABLE locator there -- the same locator class Wf-A194 closed inside THIS record.
* **the repair:** The live string now cites that artefact by JSON PATH plus its sha256, as measured on 2026-11-02: GeoLDM's own 93.8 +- 0.4 at `/sources/4/metrics/0/value` and `/sources/4/metrics/0/evidence_quote`, and HLTF's GeoLDM-column Val 93.6 +- 0.2 at `/sources/11/metrics/2/evidence_quote`, all in molmetal/reports/wfa175_frontier_comparability/sources.json (sha256 2e665cb217759a4099c83716a8f27829797f2401c6c630adaeb1b4cfbd6e3c78, as measured on 2026-11-02); and the Wf-A175 report.md (sha256 170f642846192837c334d45bb6a006414bf1b53a7089c8907f4061ff0d0f10e0, as measured on 2026-11-02) is cited by its sha256 and by the verbatim sentence it carries.  The superseded rendering is preserved VERBATIM in `sources`.`GeoLDM`.`extra_observations_superseded_2026_11_02` (NAMED DATED CONSTANT `WFA197_R1_SUPERSEDED_OBSERVATION_2026_11_02` in the producer).
* **the superseded rendering is preserved VERBATIM at:** `sources`.`GeoLDM`.`extra_observations_superseded_2026_11_02`

### R2 -- a date in a key's identifier that is not provable from disk

* **the defect:** `band_provenance`.`frontier_band_residuals_closed_2026_11_01` says its date 'is the identifier', yet MEASURED on 2026-11-02 the key is ABSENT from the pre-Wf-A194 published bytes of this record (2026-11-01 21:16) and was FIRST PUBLISHED in the 2026-11-02 01:23:34 write.  A parallel block, `band_provenance`.`locator_class_and_geo_ldm_source_2026_11_02`, covers the SAME TWO closures.
* **the repair:** The key is NOT RENAMED -- other artefacts cite it by name.  A dated sibling key `band_provenance`.`frontier_band_residuals_closed_2026_11_01_identifier_not_provable_2026_11_02` records the measurement, the limitation and the overlap with the dated 2026-11-02 block, and the date in the original key's identifier is NOT asserted as a first-publication date anywhere in this record.
* **the superseded rendering is preserved VERBATIM at:** n/a -- the key is not renamed, so nothing is superseded

### X5 -- a settled count claim about the number of competing readings

* **the defect:** `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`why_this_key_exists` said, verbatim: 'The repo holds TWO competing readings, and they do not even agree on the numerals.'  That is a claim about THE REPO, not about this record's own `readings` array (which really does hold 2 entries, and whose `status` and `canonical_reading_safe_band_statement` are locally true).  MEASURED on 2026-11-02: the repo's own primary-source verification (molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json -> `invariants`.`competing_readings`) records a FURTHER reading that likewise does not reproduce the quoted numerals, and no document on disk settles how many there are.
* **the repair:** The live string now asserts NO settled count: the record enumerates two; the repo's primary-source verification records a further reading that likewise does not reproduce the quoted numerals; the NUMBER of competing readings is itself NOT settled on disk.  The superseded rendering is preserved VERBATIM in `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`why_this_key_exists_superseded_2026_11_02` (NAMED DATED CONSTANT `WFA197_X5_SUPERSEDED_WHY_2026_11_02` in the producer).  The INVARIANT-USE of 'Under BOTH readings' is NOT churned -- the four invariants hold under EVERY reading, including the third -- and the four invariants themselves are NOT touched.
* **the superseded rendering is preserved VERBATIM at:** `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`why_this_key_exists_superseded_2026_11_02`

**The passages that were re-rendered** (measured by diffing the pre-Wf-A194 published bytes of this record against the published bytes).

* `sources`.`HLTF`.`cells`.`METRIC`.`note`
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`line_number_note`
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings[0]`.`endpoints`.`upper`
* `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`the_only_claims_that_may_be_published`.`ii_it_is_not_one_homogeneous_quantity`
* `band_provenance`.`endpoint_candidates[1]`.`candidate_a`.`what_it_is`
* `band_provenance`.`endpoint_candidates[1]`.`note`
* `band_provenance`.`producer_and_record_reconciled_2026_11_01`.`divergence_from_the_producer`
* RENAMED, with the values PRESERVED VERBATIM: `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings[0]`.`current_line_numbers_on_disk` -> `readings[0]`.`historical_line_numbers_measured_2026_11_01`, and `readings[1]`.`current_line_numbers_on_disk` -> `readings[1]`.`historical_line_numbers_measured_2026_11_01`
* RECOMPUTED (the self-referential counters only): `band_provenance`.`producer_reproduces_the_record_2026_11_01`.`byte_evidence`.`md_body_chars`, `.`record_leaf_count` and `.`record_numeric_leaf_count`
* ADDED, not re-rendered: the `sources`.`GeoLDM` entry itself, its table row, its section, its two retrieval-log rows, and the dated 2026-11-02 markdown section

**What this repair did NOT do.**

* It did NOT change the FOUR publishable invariants -- the band is EXTERNAL to Mol-Metal, it is NOT one homogeneous quantity, it is NOT on our scale (OUR 8.51 % IS a connected-component count on the RAW emitted SMILES), and the source cells of the band are NOT settled on disk.  They are the only publishable claims and they are not touched here.
* It did NOT churn the INVARIANT-USE of 'Under BOTH readings': the four invariants hold under EVERY reading, including the third that the repo's primary-source verification records.
* It did NOT write a settled COUNT of the competing readings anywhere: the naive repair ('THREE') is the same defect.
* It did NOT rename `band_provenance`.`frontier_band_residuals_closed_2026_11_01`; other artefacts cite it by name.
* It did NOT edit any dated markdown section in place: the false 2026-11-02 assertion and the wrong-numeral census are LEFT VERBATIM and are superseded by dated siblings.
* It did NOT touch any file outside the six this wave owns: molmetal/scripts/wfa158_frontier_postproc_matrix.py, the three artefacts under molmetal/reports/wfa158_frontier_postproc_matrix/, and the two test files.
* It did NOT touch the Wf-A175 / Wf-A186 / Wf-A195 trees: they belong to another wave running at the same time.
* It did NOT move any numeric leaf: no numeric leaf was added, removed or given a new value -- the only leaves that changed value are the three self-referential counters the producer itself recomputes.
