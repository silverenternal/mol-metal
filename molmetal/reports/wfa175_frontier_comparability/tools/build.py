#!/usr/bin/env python3
"""Wf-A175 FRONTIER — build sources.json + report.md (CPU-only, atomic writes)."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = "/home/hugo/codes/mol-metal"
sys.path.insert(0, os.path.join(REPO, "molmetal", "scripts"))
from wf386a3_atomic_write import atomic_write_file, safe_overwrite_check, detect_truncation  # noqa: E402

E = "https://arxiv.org/e-print/"

# ---------------------------------------------------------------- our numbers
RAW = 1078 / 12668          # cfm_fragmentation_postproc_arm.json -> raw_axis.single_fragment_rate_over_parsed
POST = 1.0
OUR_RATIO = POST / RAW

def r(a, b):
    return b / a

# ------------------------------------------------------------ connectivity axis
# Buttenschoen et al. 2505.00518, tables/chemical_validity.tex (re-verified by this wave)
BB = {
    "EQGAT-diff": (84.4, 87.1),
    "FlowMol": (68.6, 87.2),
    "GCDM": (86.6, 95.5),
    "GeoLDM": (37.4, 73.8),
    "SemlaFlow": (92.3, 94.9),
}
BB_RATIOS = {k: r(v[0], v[1]) for k, v in BB.items()}

# DiffSBDD 2210.13695v3, main.tex tab:basic_metrics (raw connectivity, %)
DSBDD = {
    "CrossDocked / DiffSBDD-inpaint (full pocket)": 67.52,
    "CrossDocked / DiffSBDD-cond (C_alpha)": 79.52,
    "CrossDocked / DiffSBDD-cond (full pocket)": 78.27,
    "CrossDocked / DiffSBDD-inpaint (C_alpha)": 98.25,
    "Binding MOAD / DiffSBDD-cond (full pocket)": 63.37,
    "Binding MOAD / DiffSBDD-inpaint (full pocket)": 75.60,
    "Binding MOAD / DiffSBDD-cond (C_alpha)": 77.38,
    "Binding MOAD / DiffSBDD-inpaint (C_alpha)": 91.60,
}
DSBDD_RATIOS = {k: r(v, 100.0) for k, v in DSBDD.items()}

# geometry axis: GenBench3D 2407.04424v1 (PDF), Table 3 (raw) vs Table S3 (relaxed)
GB3D = {
    "DiffSBDD": (0.004, 0.532),
    "LiGAN": (0.023, 0.551),
    "Pocket2Mol": (0.094, 0.783),
    "3D-SBDD": (0.099, 0.703),
    "ResGen": (0.111, 0.687),
    "TargetDiff": (0.106, 0.544),
}
GB3D_RATIOS = {k: r(v[0], v[1]) for k, v in GB3D.items()}

EQGAT_VALIDITY = (0.916, 0.998)          # raw (no MolDiff PP) -> with MolDiff PP
EQGAT_CONN = {"QM9 w_s(t)": 0.9994, "QM9 w_u": 0.9970,
              "GEOM-Drugs w_s(t)": 0.9508, "GEOM-Drugs w_u": 0.8657}
TABASCO_CONN = {"30 steps": 0.99, "40 steps": 0.99, "50 steps": 1.00,
                 "100 steps": 1.00, "200 steps": 1.00, "500 steps": 1.00}

# ------------------------------------------------------------------- sources
S = []


def add(**kw):
    S.append(kw)


add(
    key="EQGAT",
    title="Navigating the Design Space of Equivariant Diffusion-Based Generative Models for 3D Molecule Generation",
    authors="Najdenov et al. (EQGAT-diff)",
    year=2023, venue="ICLR 2024 (arXiv v2)",
    arxiv="2309.17296v2", url=E + "2309.17296v2",
    eprint_sha256="9f88c403651a6414fc86c2656e35a30fb5a386152698595ed38831c4b2fc78d4",
    eprint_bytes=3242988, retrieved_utc="2026-11-01 (this wave)",
    datasets=["QM9", "GEOM-Drugs"],
    metrics=[
        {"metric": "Validity (RDKit sanitization success rate, 10,000 molecules)",
         "value": "0.998 (with MolDiff post-processing pipeline); 0.916 (without it)",
         "raw_or_post": "BOTH reported in the same paper: 0.916 = RAW (no post-processing), 0.998 = POST (MolDiff pipeline)",
         "raw_value": 0.916, "post_value": 0.998, "ratio": r(*EQGAT_VALIDITY),
         "denominator": "5 x 10,000 generated samples (best model EQGAT_disc^{x0,af,ft})",
         "evidence_quote": "We measure validity using the success rate of RDKit sanitization over 10,000 molecules (pre-selecting connected components only)  [main.tex:303] | Suppose we do not employ the post-processing scheme from MolDiff and determine the validity by parsing the generated molecule into RDKit's sanitization pipeline. In that case, EQGAT-diff obtains a mean validity of 0.916 and a mean success rate of 0.887.  [e3moldiffusion.tex:843] | EQGAT-diff ... Validity 0.998 ... Connectivity 0.968 ... Succ. Rate 0.966  [tab:appendix table-moldiff, e3moldiffusion.tex:853-855]",
         "locator": "e3moldiffusion.tex:303, :317-320, :843, :853-855"},
        {"metric": "Connect. Comp. (percent of single-connected-component molecules)",
         "value": "QM9 99.94 / 99.70 ; GEOM-Drugs 95.08 / 86.57",
         "raw_or_post": "RAW (pre-selection; reported alongside a post-selection validity)",
         "raw_value": 0.8657, "post_value": 1.0,
         "ratio": r(0.8657, 1.0),
         "denominator": "10,000 molecules per model",
         "evidence_quote": "Validity & Connect. Comp. ... w_u & 97.99 & 99.70 (QM9) & 71.44 & 86.57 (GEOM-Drugs) ; w_s(t) & 98.96 & 99.94 & 84.02 & 95.08  [e3moldiffusion.tex:317-320]",
         "locator": "e3moldiffusion.tex:317-320"},
    ],
    note="The only paper found that publishes a same-model, same-paper RAW-vs-POST validity pair.",
)

add(
    key="DiffSBDD",
    title="Structure-based Drug Design with Equivariant Diffusion Models",
    authors="Schneuing et al.",
    year=2024, venue="ICML 2024 (arXiv v3)",
    arxiv="2210.13695v3", url=E + "2210.13695v3",
    eprint_sha256="4fa02b871e61183f35f0b7a3a08e4442c0357fa8720917af0d4c401fdcb3c7ab",
    eprint_bytes=33236821, retrieved_utc="2026-11-01 (this wave)",
    datasets=["CrossDocked test set", "Binding MOAD test set"],
    metrics=[
        {"metric": "Connectivity (percent of VALID molecules with no disconnected fragment)",
         "value": "63.37 - 98.25 (8 cells)",
         "raw_or_post": "RAW - measured on the decoded graph; the same paper then applies largest-fragment selection as its post-processing step",
         "raw_value": 63.37, "post_value": 100.0, "ratio": r(63.37, 100.0),
         "denominator": "all generated molecules for the pocket set (table tab:basic_metrics)",
         "evidence_quote": "Connectivity is the proportion of valid molecules that do not contain any disconnected fragments.  [main.tex:929] | For postprocessing of generated molecules ... We then use RDKit to sanitise molecules and filter for the largest molecular fragment.  [main.tex:594-595] | A much larger percentage of molecules is fragmented but, since we can simply select and process the largest fragments in these cases, low connectivity does not necessarily affect the efficiency of the generative process.  [main.tex:932]",
         "locator": "main.tex:594-595, :929, :932, table tab:basic_metrics main.tex:940-954"},
    ],
    note="Post value 100 % is IMPLIED by the paper's own stated pipeline (largest-fragment filter), not printed as a connectivity cell.",
)

add(
    key="TABASCO",
    title="TABASCO: A Fast, Simplified Model for Molecular Generation with Improved Physical Quality",
    authors="(TABASCO authors)",
    year=2025, venue="arXiv (ICLR-submission format)",
    arxiv="2507.00899v1", url=E + "2507.00899v1",
    eprint_sha256="f91e660918b23fb27cc92fa3943bba7907b5e8ee6cd3047f877f46eb6da9a086",
    eprint_bytes=5558907, retrieved_utc="2026-11-01 (this wave)",
    datasets=["GEOM-Drugs"],
    metrics=[
        {"metric": "Connectivity (fraction of fully connected molecules)",
         "value": "0.99-1.00 at 30-500 sampling steps; 0.00 at 10 and 20 steps",
         "raw_or_post": "RAW - the authors state they saved the samples 'without any post-processing'",
         "raw_value": 0.99, "post_value": 1.0, "ratio": r(0.99, 1.0),
         "denominator": "sampled molecules per setting (3 seeds x 1000 for the baselines comparison)",
         "evidence_quote": "We additionally evaluate connectivity, which denotes the fraction of fully connected molecules.  [example_paper.tex:629] | We used the example evaluation script, which we edited to save molecules as outputted from reverse sampling, without any post-processing.  [example_paper.tex:526] | table tab:num-sample-steps: #steps 10 -> Validity 0.99, Connectivity 0.00 ; 30 -> 0.98 / 0.99 ; 50 -> 0.99 / 1.00  [example_paper.tex:632-641]",
         "locator": "example_paper.tex:526, :629, :632-641"},
    ],
    note="The frontier's own ablation shows connectivity collapsing to 0.00 while Validity stays 0.99-1.00 - the same RAW-vs-POST trap on the connectivity axis.",
)

add(
    key="GenBench3D",
    title="Benchmarking structure-based three-dimensional molecular generative models using GenBench3D",
    authors="Baillif et al.",
    year=2024, venue="arXiv (PDF-only submission)",
    arxiv="2407.04424v1", url=E + "2407.04424v1",
    eprint_sha256="330a8cc2face85120aa057c8b580ec5734df1618df01d797e0a8dc46bf08a8e3",
    eprint_bytes=4032287, retrieved_utc="2026-11-01 (this wave)",
    datasets=["CrossDocked2020", "CSD Drug Subset", "SPINDR (via NEAT-POCKET)"],
    metrics=[
        {"metric": "Validity3D (fraction of molecules with a valid 3D conformation) - RAW vs POST-relaxation",
         "value": "DiffSBDD 0.004 -> 0.532 ; LiGAN 0.023 -> 0.551 ; Pocket2Mol 0.094 -> 0.783 ; 3D-SBDD 0.099 -> 0.703 ; ResGen 0.111 -> 0.687 ; TargetDiff 0.106 -> 0.544",
         "raw_or_post": "BOTH reported: RAW = before MMFF94s relaxation (Table 3), POST = after relaxation (Table S3). NOTE: this post-processing is a GEOMETRY relaxation, not fragment selection",
         "raw_value": 0.004, "post_value": 0.532, "ratio": r(0.004, 0.532),
         "denominator": "generated molecules per model (up to 10,000; 100 per target x 100 targets)",
         "evidence_quote": "Evaluation of conformation quality metrics on raw generated molecules ... Validity3D: LiGAN 0.023, 3D_SBDD 0.099, Pocket2Mol 0.094, TargetDiff 0.106, DiffSBDD 0.004, ResGen 0.111  [Table 3] | For all models, the relaxation drastically increased the Validity3D: for instance, LiGAN increases from 0.02 to 0.55, Pocket2Mol increases from 0.09 to 0.78, and ResGen from 0.11 to 0.69.  [p.46 text]",
         "locator": "PDF p.22 (Table 3), p.46 (relaxation text), p.48 (Table S3)"},
        {"metric": "ValidityG (fraction of molecules with an RDKit-sanitisable graph)",
         "value": "LiGAN 0.753 ; 3D-SBDD 0.815 ; Pocket2Mol 0.982 ; TargetDiff 1.000 ; DiffSBDD 1.000 ; ResGen 1.000",
         "raw_or_post": "RAW (no fragment-selection step reported for the generated set; the LargestFragmentChooser is used only to load the CSD reference subset)",
         "raw_value": 0.753, "post_value": None, "ratio": None,
         "denominator": "generated molecules per model",
         "evidence_quote": "ValidityG (1) Fraction of molecules with valid graphs. A molecular graph is valid if it successfully sanitized by RDKit  [Table 1] | The structures were retrieved using the CSD Python API, read into RDKit using a .mol file intermediate, and only the largest fragment was selected for each structure using the LargestFragmentChooser in RDKit.  [CSD-reference paragraph, p.21]",
         "locator": "PDF p.21 (pipeline), p.48 (Table S1)"},
    ],
    note="PDF-only arXiv submission - no LaTeX source exists; text extracted with pdftotext -layout.",
)

# --------------------------------------------------------------- Wf-A186 REPAIR (2026-11-01)
#
# The published band's provenance is NOT settled on disk, so this generator must NOT emit a
# single settled reading of it.  The frontier post-processing matrix record
# (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`) holds TWO competing
# readings of which frontier cells the band names -- `band_provenance` and `sources`.`EF-TALFM` --
# and they do not even agree on the numerals.
#
# The note that stood here before this repair is preserved VERBATIM (AGENTS.md 5.5 rule #6):
#
#   "CRITICAL for the band: the band endpoints 92.7 and 93.8 are GeoLDM's OWN QM9 cells from ONE table row, and 89.4 is GeoLDM's own QM9 molecule-stability cell."
#
# WHY IT WAS SUPERSEDED: it names ONE settled provenance for the published band ("the band
# endpoints 92.7 and 93.8"), which the record above does not support -- and 93.8 is NOT a band
# endpoint under either competing reading.  The superseded string also survives verbatim in the
# generated `sources.json`, under the dated key `superseded_notes_2026_11_01_wfa186_frontier`.
#
# Substance of the replacement is the canonical reading-safe band statement carried by that
# record under `band_provenance`.`competing_reading_and_what_this_does_not_settle`.
#`canonical_reading_safe_band_statement` (read from disk on 2026-11-01).  It introduces no numeral
# that is not already on disk in this branch.  ONE note text is shared by the GeoLDM and the HLTF
# source records, so the generator and the artefact cannot drift apart on this claim.
BAND_PROVENANCE_NOTE = "The published band's provenance is NOT settled on disk: the frontier post-processing matrix record (molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json) holds TWO competing readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk and neither reading may be published as its settled provenance.  Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.  QUOTED FRONTIER CELLS, not band endpoints: 92.7 +/- 0.5 and 93.8 +/- 0.4 are GeoLDM's OWN QM9 cells from ONE table row (its Valid&Unique (%) and Valid (%) cells) and 89.4 +/- 0.5 is GeoLDM's OWN QM9 Mol. Stability cell; 93.8 is NOT a band endpoint under either reading, and the branch's own text does not settle 93.8 to a single cell (this report's section 4.1 also carries 93.8 +- 0.6 as HLTF's own Val (%))."

add(
    key="GeoLDM",
    title="Geometric Latent Diffusion Models for 3D Molecule Generation",
    authors="Simm et al.",
    year=2023, venue="ICML 2023",
    arxiv="2305.01140v1", url=E + "2305.01140v1",
    eprint_sha256="b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23",
    eprint_bytes=655684, retrieved_utc="2026-11-01 (this wave)",
    datasets=["QM9", "DRUG (drug-like 3D)"],
    metrics=[
        {"metric": "QM9 Valid (%) / Valid&Unique (%) / Mol. Stability (%)",
         "value": "93.8 +/- 0.4 / 92.7 +/- 0.5 / 89.4 +/- 0.5",
         "raw_or_post": "PROTOCOL-UNSPECIFIED in the paper (no fragment-selection statement); the same paper notes the model 'occasionally generates disconnected components'",
         "raw_value": None, "post_value": None, "ratio": None,
         "denominator": "10,000 samples per model",
         "evidence_quote": "\\rowcolor{lightgray} \\textbf{\\textsc{GraphLDM}} ... GeoLDM & \\textbf{98.9} $\\pm$ 0.1 & \\textbf{89.4} $\\pm$ 0.5 & \\textbf{93.8} $\\pm$ 0.4 & \\textbf{92.7} $\\pm$ 0.5 & \\textbf{84.4} & \\textbf{99.3}  [icml2023.tex:491] | the model occasionally generates disconnected components  [icml2023.tex:917]",
         "locator": "icml2023.tex:491, :917"},
    ],
    note=BAND_PROVENANCE_NOTE,
    note_supplement_2026_11_01_wfa175_frontier="APPEND-ONLY supplement (Wf-A175 FRONTIER, 2026-11-01): the earlier sentence above is correct about 92.7 and 93.8 being GeoLDM's OWN QM9 cells, but the BAND '92.7-94 %' as published is minted in a DIFFERENT table -- HLTF arXiv:2603.27113v1 tables/qm9_benchmarks_og.tex:56-66 (Table 1(b) tab:qm9_pb), whose lower endpoint 92.7 % is GeoLDM's Val&Uniq (line 64) and whose upper endpoint 94 % is HLTF's OWN PB-Valid (line 66). Different models, different metrics, same table.",
)

add(
    key="NEAT-POCKET",
    title="NEAT-POCKET: Pocket-Conditioned Autoregressive 3D Molecular Generation with a Neighborhood-Guided Sequence",
    authors="(NEAT-POCKET authors)",
    year=2026, venue="arXiv (NeurIPS-2026 submission format)",
    arxiv="2609.05097v1", url=E + "2609.05097v1",
    eprint_sha256="bd978841f62a7d75e8635714e7336a6e3abec39b854ec3f2c90974890f2b7fdb",
    eprint_bytes=10632429, retrieved_utc="2026-11-01 (this wave)",
    datasets=["CrossDocked2020", "SPINDR"],
    metrics=[
        {"metric": "PoseBusters validity (PB-valid) of de novo generations / baselines",
         "value": "NEAT-POCKET 69.2 (CrossDocked) / 80.3 (SPINDR); FLOWR 75.8 (SPINDR); CrossDocked reference ligands 94.0",
         "raw_or_post": "RAW-composite (graph + geometry); explicit H added via RDKit AddHs is the only declared post-step",
         "raw_value": 0.692, "post_value": None, "ratio": None,
         "denominator": "generated molecules per pocket (100 per target)",
         "evidence_quote": "NEAT-POCKET achieved the second-highest PB validity, with 69.2\\% PB valid molecules  [sections/results.tex:132] | NEAT-POCKET achieves higher PB validity, 80.3\\% versus 75.8\\%  [sections/results.tex:170]",
         "locator": "sections/results.tex:126, :132, :170"},
        {"metric": "INDEPENDENT PROOF that DiffSBDD's headline validity is POST-processed",
         "value": "n/a (protocol evidence)",
         "raw_or_post": "n/a",
         "raw_value": None, "post_value": None, "ratio": None,
         "denominator": "n/a",
         "evidence_quote": "In the original inference procedure, generated molecules are explicitly filtered according to internal validity criteria before being returned. Specifically, a molecule is considered valid if its largest fragment, identified using RDKit's Chem.GetMolFrags() function, can be successfully sanitized by RDKit. ... Thus, the original implementation effectively continues sampling until 100 valid molecules have been obtained.  [sections/supplementary.tex:99]",
         "locator": "sections/supplementary.tex:99"},
    ],
    note="Also documents that DiffSBDD's original protocol is a REJECTION SAMPLER that resamples until 100 valid molecules are collected - a selection bias on top of the fragment filter.",
)

add(
    key="PoseBusters",
    title="PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences",
    authors="Buttenschoen et al.",
    year=2023, venue="Chemical Science 2024 (arXiv v3)",
    arxiv="2308.05777v3", url=E + "2308.05777v3",
    eprint_sha256="e5bcb1ec75104a327385b13470ae38816d42adbff6ffb318774d0eb9e253068b",
    eprint_bytes=9900693, retrieved_utc="2026-11-01 (this wave)",
    datasets=["Astex Diverse", "PoseBusters Benchmark set"],
    metrics=[
        {"metric": "PB-valid (passes all PoseBusters tests) for docking methods",
         "value": "Gold 48 % ; AutoDock Vina 51 % ; DiffDock best of the DL methods ; Uni-Mol 23 % (RMSD<=2A: DiffDock 72 %, TankBind 59 %, Gold 67 %, Vina 58 %, Uni-Mol 45 %)",
         "raw_or_post": "RAW for connectivity (no fragment selection); the only post-processing is a post-docking energy minimisation",
         "raw_value": 0.48, "post_value": None, "ratio": None,
         "denominator": "complexes in the benchmark set",
         "evidence_quote": "Molecule poses which pass all tests in PoseBusters are `PB-valid'.  [01_main.tex:247] | we performed an additional post-docking energy minimisation of the ligand structures in the binding pocket for the PoseBusters Benchmark set  [01_main.tex:378]",
         "locator": "01_main.tex:247, :328-378"},
    ],
    note="Docking, not generation - included because the repo's PoseBusters-validity metric JSON cites it.",
)

add(
    key="Buttenschoen2025",
    title="An evaluation of unconditional 3D molecular generation methods",
    authors="Buttenschoen et al.",
    year=2025, venue="ICLR 2025 (arXiv v1)",
    arxiv="2505.00518v1", url=E + "2505.00518v1",
    eprint_sha256="52ca160b7734259eda3bfd7f981920379e144661dfd9228d70447cc6c4ee7e57",
    eprint_bytes=1059489, retrieved_utc="2026-11-01 (this wave; re-fetched, hash identical to the Wf-A171 pin)",
    datasets=["QM9", "GEOM-Drugs"],
    metrics=[
        {"metric": "Connected (%) BEFORE and AFTER the 3-step post-processing (largest fragment -> explicit H -> UFF)",
         "value": "GeoLDM 37.4 -> 73.8 ; FlowMol 68.6 -> 87.2 ; GCDM 86.6 -> 95.5 ; EQGAT-diff 84.4 -> 87.1 ; SemlaFlow 92.3 -> 94.9",
         "raw_or_post": "BOTH reported side by side in one table (RAW row + '+ PP' row) - the ONLY frontier table that measures the same models with and without fragment selection",
         "raw_value": 37.4, "post_value": 73.8, "ratio": r(37.4, 73.8),
         "denominator": "generated molecules per model (10,000-class protocol)",
         "evidence_quote": "The generated molecules were post-processed in three steps. First, the largest fragment was picked using the RDKit's LargestFragmentChooser. Second, missing explicit hydrogens were filled in ... Third, the molecular conformation was refined by minimising energy using the Universal Force Field implemented in the RDKit.  [iclr2025_conference.tex:129-132] | table chemical_validity.tex: GeoLDM 72.5 | 5.3 | 37.4 | 3.2 ... GeoLDM + PP 73.8 | 73.8 | 73.8 | 73.8",
         "locator": "iclr2025_conference.tex:129-132; tables/chemical_validity.tex"},
    ],
    note="Frontier MAXIMUM measured raw->post connectivity ratio = 1.9733x (GeoLDM).",
)

add(
    key="DecompDiff",
    title="DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design",
    authors="Guan et al.",
    year=2024, venue="NeurIPS 2024 (arXiv v1)",
    arxiv="2403.07902v1", url=E + "2403.07902v1",
    eprint_sha256="f510ac0224de07c3a6daaeb9cb3fdf0a67933213eed6d4e1fa189d87387a7c08",
    eprint_bytes=26329369, retrieved_utc="2026-11-01 (this wave)",
    datasets=["CrossDocked2020"],
    metrics=[
        {"metric": "none extracted by this wave (protocol paper for this axis)",
         "value": None, "raw_or_post": "n/a",
         "raw_value": None, "post_value": None, "ratio": None,
         "denominator": None,
         "evidence_quote": "We also introduce a diffusion process on bonds to incorporate the bond generation in an end-to-end fashion instead of post-processing.  [sections/10-intro.tex:51]",
         "locator": "sections/10-intro.tex:51, sections/40-method.tex:189"},
    ],
    note="Included as protocol evidence: the frontier is actively REMOVING bond post-processing, so its connectivity numbers are not obtained by the same route as ours.",
)

add(
    key="SemlaFlow",
    title="SemlaFlow -- Efficient 3D Molecular Generation with Latent Attention",
    authors="Irwin et al.",
    year=2024, venue="arXiv v3",
    arxiv="2406.07266v3", url=E + "2406.07266v3",
    eprint_sha256="72dc18f51a1d2361d13bb7224edaefe1bcb71833cfff52153e88e50d3a43c3c5",
    eprint_bytes=2034012, retrieved_utc="2026-11-01 (this wave)",
    datasets=["QM9", "GEOM-Drugs"],
    metrics=[
        {"metric": "self-declared comparability caveat on Validity / Mol. Stability",
         "value": None, "raw_or_post": "n/a",
         "raw_value": None, "post_value": None, "ratio": None,
         "denominator": None,
         "evidence_quote": "The validity and stability metrics presented here, however, only measure the topological structure of the molecule (i.e. the molecular graph); they provide no information on the quality of the generated conformations.  [semlaflow.tex:397]",
         "locator": "semlaflow.tex:397"},
    ],
    note="A frontier author stating in print that 'validity' is a TOPOLOGICAL metric only - the same distinction our band critique rests on.",
)

# ---------------------------------------------------------------- our source
add(
    key="OURS",
    title="mol-metal CFM fragmentation post-processing arm (this repository)",
    authors="mol-metal",
    year=2026, venue="internal artifact",
    arxiv=None, url="metrics/by_metric/cfm_fragmentation_postproc_arm.json",
    eprint_sha256=None, eprint_bytes=None, retrieved_utc="read from disk 2026-11-01",
    datasets=["c7 post-hoc decode pool"],
    metrics=[
        {"metric": "single-fragment rate over parsed pool (RAW) -> connected rate after LargestFragmentChooser + AddHs + UFF (POST)",
         "value": "1078 / 12668 = 0.08509630565203663 -> 1.0",
         "raw_or_post": "both measured in our own arm",
         "raw_value": RAW, "post_value": POST, "ratio": OUR_RATIO,
         "denominator": "12,668 parsed rows",
         "evidence_quote": "metrics/by_metric/cfm_fragmentation_postproc_arm.json -> raw_axis.single_fragment_rate_over_parsed = 0.08509630565203663 ; postprocessed_axis.connected_rate_over_parsed = 1.0",
         "locator": "metrics/by_metric/cfm_fragmentation_postproc_arm.json"},
    ],
    note="Cost of our own post-processing: 22.47 % of raw heavy mass discarded; 21.02 % of survivors are <=2-heavy-atom specks (same artifact).",
)

# --- Wf-A186 (2026-11-01): the two frontier source records this generator was NOT emitting ----
# Producer drift, MEASURED: before this repair `build.py` emitted 11 source records, while the
# published `sources.json` carried 13 (HLTF and EF-TALFM were missing) plus the top-level
# `band_census` and `computed_ratios` sections.  Re-running the generator therefore REWROTE the
# published artefact: the producer could not reproduce its own output.  The records below are
# reproduced from the published artefact verbatim, so that the generator now reproduces the
# artefact byte for byte.  The HLTF note is the ONE reading-safe band note (see above); the old
# HLTF note is preserved VERBATIM in the dated comment below and in the dated key
# `superseded_notes_2026_11_01_wfa186_frontier` of the generated `sources.json`.
#
#   "The 92.7-94 % band is minted here: two endpoints, two different metrics, two different models, one table. The same table also carries GeoLDM's MS 89.4 %, the adjacent cell that migrated onto EF-TALFM as '89.4'."
#
# The two HLTF metric names that asserted the withdrawn endpoint mapping are preserved VERBATIM:
#   "GeoLDM Val&Uniq (%) -- the LOWER endpoint of the published 92.7-94 % band"
#   "HLTF (ours) PB-Valid (%) -- the UPPER endpoint of the published 92.7-94 % band"
# The EF-TALFM note below is reproduced VERBATIM from the artefact this generator used to be
# unable to produce at all (see the Wf-A186 producer-drift note above).  It is not a band-
# provenance claim -- it is the '89.4' vs '87.9' string census.
EF_TALFM_NOTE = "String census in the re-fetched e-print AND on the abs page: '89.4' = 0 occurrences, '87.9' = 2 (e-print) / 3 (abs). The '89.4' that migrated onto this paper is GeoLDM's MS cell in HLTF Table 1(a):38 / 1(b):62 -- an adjacent-cell migration. EF-TALFM publishes NO pre-PP/post-PP connectivity pair, so no ratio can be computed from it."

# ---- Wf-A186: the two frontier sources the generator was NOT emitting --------------
add(
    key="HLTF",
    title="HLTF (arXiv 2603.27113v1) -- latent flow-matching 3D molecular generator, QM9 benchmark table",
    authors="(as printed in the e-print)",
    year=2026,
    venue="arXiv preprint",
    arxiv="2603.27113v1",
    url="https://arxiv.org/e-print/2603.27113v1",
    eprint_sha256="87c1788e9e9826cfbdf4e0788feed1a796c8a50218eecc6905df771f10c9c329",
    eprint_bytes=504186,
    retrieved_utc="2026-11-01 (this wave)",
    datasets=["QM9"],
    metrics=[
        {"metric": "GeoLDM Val&Uniq (%) -- a CANDIDATE lower endpoint of the published 92.7-94 % band (reading A of TWO competing readings; NOT settled on disk)",
         "value": "92.7 +/- 0.5",
         "raw_or_post": "PROTOCOL-UNSPECIFIED (no fragment-selection statement)",
         "raw_value": None,
         "post_value": None,
         "ratio": None,
         "denominator": "QM9 samples; the paper's own sampling protocol",
         "evidence_quote": "Val\\&Uniq (\\%) $\\uparrow$ & $92.7\\pm0.5$ & $93.3\\pm0.0$* & \\underline{$92.9\\pm0.2$} [tables/qm9_benchmarks_og.tex:64]",
         "locator": "tables/qm9_benchmarks_og.tex:64 (Table 1(b) tab:qm9_pb)"},
        {"metric": "HLTF (ours) PB-Valid (%) -- a CANDIDATE upper endpoint of the published 92.7-94 % band (reading A of TWO competing readings; NOT settled on disk)",
         "value": "94 +/- 0.5",
         "raw_or_post": "POST (PoseBusters test suite includes explicit-H / sanitization checks)",
         "raw_value": None,
         "post_value": None,
         "ratio": None,
         "denominator": "QM9 samples; PB-Valid is a DIFFERENT metric from Val&Uniq and a different model",
         "evidence_quote": "PB-Valid (\\%) $\\uparrow$ & $93.1\\pm0.4$ & $91.9\\pm0.5$ & \\textbf{$94\\pm0.5$*} [tables/qm9_benchmarks_og.tex:66]",
         "locator": "tables/qm9_benchmarks_og.tex:66"},
        {"metric": "GeoLDM MS (molecule stability) (%) -- the origin of the stray '89.4'",
         "value": "89.4 +/- 0.5",
         "raw_or_post": "PROTOCOL-UNSPECIFIED",
         "raw_value": None,
         "post_value": None,
         "ratio": None,
         "denominator": "QM9 samples",
         "evidence_quote": "GeoLDM & $98.9\\!\\pm\\!0.1$* & $89.4\\!\\pm\\!0.5$* & $93.6\\!\\pm\\!0.2$ & $92.7\\!\\pm\\!0.5$ [tables/qm9_benchmarks_og.tex:38] ... MS (\\%) $\\uparrow$ & $89.4\\pm0.5$* [tables/qm9_benchmarks_og.tex:62]",
         "locator": "tables/qm9_benchmarks_og.tex:38 and :62"},
    ],
    note=BAND_PROVENANCE_NOTE,
)

add(
    key="EF-TALFM",
    title="EF-TALFM (arXiv 2609.08333v4) -- Equivariant-Free Transformer-Autoencoded Latent Flow Matching",
    authors="(as printed in the e-print)",
    year=2026,
    venue="arXiv preprint",
    arxiv="2609.08333v4",
    url="https://arxiv.org/e-print/2609.08333v4",
    eprint_sha256="e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224",
    eprint_bytes=8508562,
    retrieved_utc="2026-11-01 (this wave; re-fetch -- hash identical to the Wf-A171 and Wf-A168 pins)",
    datasets=["PCQM4Mv2"],
    metrics=[
        {"metric": "Connectivity (%, PCQM4Mv2 generation)",
         "value": "92.98",
         "raw_or_post": "RAW by necessity (pre largest-component), but the paper's own metric sentence says POST -- internally inconsistent; see evidence",
         "raw_value": None,
         "post_value": None,
         "ratio": None,
         "denominator": "10,000 generated molecules",
         "evidence_quote": "We report connectivity of the reconstructed molecule and sanitization, uniqueness, and novelty rates computed after retaining the atom-count largest connected component of each sampled molecule and applying strict RDKit sanitization. [src/appendix.tex:183] || For fragmented outputs, SMILES-based and PoseBusters metrics use the atom-count largest connected component after sanitization. [src/appendix.tex:54] || C. denotes the fraction of connected molecules [src/appendix.tex:184] || its connectivity is moderately lower (92.98% versus 98.72%) [src/appendix.tex:82]",
         "locator": "src/appendix.tex:183 vs :54 / :184 (conflict); value at src/appendix.tex:115 and :82"},
        {"metric": "PoseBusters-verified novel yield (%) -- the figure the '87.9 vs 89.4' dispute is about",
         "value": "87.9",
         "raw_or_post": "POST (PoseBusters joint pass over the sanitized, unique, training-set-novel subset)",
         "raw_value": None,
         "post_value": None,
         "ratio": None,
         "denominator": "generated candidates; N&U&S population",
         "evidence_quote": "EF-TALFM achieves the highest PoseBusters-verified novel yield among the evaluated methods, $87.9\\%$, compared with $75.2\\%$ for UAE-3D and $69.6\\%$ for FlowMol [sections/results_unconditional.tex:35]",
         "locator": "sections/results_unconditional.tex:35"},
        {"metric": "sanitized+unique+novel yield (%) / all-PoseBusters-checks pass (%) -- other cells the band census sweeps",
         "value": "94.33 / 93.13",
         "raw_or_post": "POST",
         "raw_value": None,
         "post_value": None,
         "ratio": None,
         "denominator": "10,000 generated molecules (yield); N&U&S population (PB checks)",
         "evidence_quote": "EF-TALFM achieves 94.33\\%, versus 76.03\\% for UAE-3D and 69.83\\% for FlowMol [src/appendix.tex:89] || 93.13\\% pass all checks jointly [src/appendix.tex:92]",
         "locator": "src/appendix.tex:89, :92, :127, :223"},
    ],
    note=EF_TALFM_NOTE,
)

# --- Wf-A186 (2026-11-01): the payload sections the generator was NOT emitting ---------------
# `band_census` is the summary of `tools/band_census.py`'s census (the full per-file census is
# the sibling `band_census.json`); `computed_ratios` is derived from this module's own constants
# so the generator is a real generator for them, not a transcription.
BAND_CENSUS_SUMMARY = {
    "artifact": "molmetal/reports/wfa175_frontier_comparability/band_census.json",
    "pattern": "92.7\\s*(?:-|–|—|to)\\s*94(?:.0)?\\s*%?",
    "working_tree_files_with_band": 38,
    "working_tree_files_scanned": 10060,
    "paper_files_with_band": 0,
    "paper_files_scanned": 134,
    "paper_files_scanned_detail": "123 tracked + 11 gitignored LaTeX build artifacts",
    "head_files_with_band": 0,
    "head_files_scanned": 10060,
    "bare_scalar_92.7_in_paper_files": 8,
    "bare_scalar_note": "every substantive hit in paper/ is a DIFFERENT quantity: the YuelBond decoder's F1 (paper/sections/04_evaluation.tex -> \\paragraph{YuelBond decoder swap: \\textsc{available} parallel path (WF-CFM-Rescue, 2026-09-16).}; paper/sections/06_limitations.tex -> \\section{Limitations}\\label{sec:limitations} -> the \\item 'YuelBond decoder swap not wired; CPU + GPU 500-step smokes bit-exact with BondAwareDecoder baseline') plus build logs and SVG path data; 92.98 / 94.0 / 94.33 are 0 of 134 in paper/",
    "misattribution_site": (
        "molmetal/reports/wfa161_frontier_labelling/report.md -> SECTION HEADING "
        "'## Wf-A176 CORRECTION (2026-11-01) — §\"Comparability breaks\" item 2 is "
        "WRONG' -> SUB-HEADING '### The superseded sentence (verbatim, line 139 of "
        "this file)'.  The live 'Comparability breaks' item 2 of that file now "
        "carries the REPAIRED two-reading text (the published band is NOT ours; "
        "the repo holds TWO competing provenance readings), so the mis-attribution "
        "survives ONLY inside that dated section -- a perishable line number is "
        "NOT a locator.  The band's endpoint cells are durably located in "
        "molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json"
        " -> JSON PATH cells[\"2603.27113v1\"][57] (92.7 +- 0.5, GeoLDM Val&Uniq, "
        "tab:qm9_pb), cells[\"2603.27113v1\"][65] (94 +- 0.5, HLTF PB-Valid, "
        "tab:qm9_pb), cells[\"2609.08333v4\"][2] (92.98, EF-TALFM C., "
        "tbl:pcqm4mv2_generation (a)) and cells[\"2609.08333v4\"][11] (94.33, "
        "EF-TALFM N.&U.&S., same table)."
    ),
    "scanner": "molmetal/scripts/wfa156_text_scan.py (UTF-8-safe, git ls-files enumeration, no os.walk)"
}

# ---------------------------------------------------------------------------
# Wf-A200 (2026-11-02) -- the CROSS-WAVE LOCATOR repair.
#
# The defect class: a document in one tree cites a LINE NUMBER inside another
# tree's artefact.  When the cited tree is re-rendered, the line number
# silently starts pointing at a DIFFERENT sentence while the citation still
# looks plausible.  Wf-A194 re-rendered
# `molmetal/reports/wfa161_frontier_labelling/report.md` (mtime
# 2026-11-02 01:20:56) and moved the mis-attribution into its dated
# `## Wf-A176 CORRECTION (2026-11-01)` section, so this tree's live
# `misattribution_site` pointed at the REPAIRED item 2 instead.
#
# THE REPAIR (AGENTS.md 5.5 rule #6 -- APPEND-ONLY, DATED, nothing deleted):
#   * the LIVE field now carries a DURABLE locator -- a SECTION HEADING into
#     the Wf-A161 record plus JSON PATHS into the Wf-A186 primary-source
#     record -- instead of a line number;
#   * the superseded string survives VERBATIM in this dated record.
# ---------------------------------------------------------------------------
CROSSWAVE_LOCATOR_REPAIR_2026_11_02_WFA200 = {
    "dated": "2026-11-02",
    "wave": "Wf-A200",
    "scope": (
        "AGENTS.md section 5.5 rule #6 -- APPEND-ONLY.  This record is ADDITIVE: "
        "the superseded rendering is preserved VERBATIM below and in "
        "`report.md` section F; nothing was deleted and no line number was "
        "silently rewritten."
    ),
    "defect_class": (
        "the CROSS-WAVE LOCATOR class: a live <path>:<line> citation into another "
        "wave's artefact, where the cited artefact was re-rendered and the line "
        "number now points at a different sentence while the citation still "
        "looks plausible."
    ),
    "superseded_misattribution_site_verbatim": (
        "molmetal/reports/wfa161_frontier_labelling/report.md:139 -- 'Our 92.7-94 % "
        "band comes from a DIFFERENT perception branch (Open Babel) ...' (mtime "
        "2026-11-01T01:59:40), superseded by docs/HONEST_NEGATIVE_CATALOG.md §6.264"
    ),
    "superseded_misattribution_site_line_number": 139,
    "superseded_misattribution_site_mtime": "2026-11-01T01:59:40",
    "where_the_site_moved_to": (
        "molmetal/reports/wfa161_frontier_labelling/report.md -> section "
        "'## Wf-A176 CORRECTION (2026-11-01) — §\"Comparability breaks\" item 2 is "
        "WRONG', whose sub-heading is '### The superseded sentence (verbatim, line "
        "139 of this file)'.  MEASURED 2026-11-02 on disk: that section's heading "
        "is at line 151 and the quoted mis-attribution at line 158; the live line "
        "139 holds the REPAIRED item 2."
    ),
    "durable_locator_now": (
        "SECTION HEADING + JSON PATH: the Wf-A161 dated section above, and "
        "molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json"
        " -> JSON PATH cells[\"2603.27113v1\"][57] / [65] and "
        "cells[\"2609.08333v4\"][2] / [11] for the four endpoint cells."
    ),
    "stale_by_construction": (
        "EVERY line number that appears in THIS RECORD is STALE BY CONSTRUCTION: it "
        "was measured on 2026-11-02 and it describes a file that is still being "
        "re-rendered.  A LIVE locator must be a JSON PATH or a SECTION ANCHOR."
    ),
    "superseded_live_locators_2026_11_02_wfa200": [
        {
            "site": "molmetal/reports/wfa175_frontier_comparability/report.md -> LIVE section 4.2",
            "superseded_locator_verbatim": "TODO/README.md:3",
            "superseded_line_number": 3,
            "the_cited_line_now_reads_verbatim_prefix": '**Last-updated trailer (Wf-A172 dead-gate history CORRECTION + pre-flight live-state + config-gate-requirement wave — 2026-11-01 — ...',
            "durable_locator_now": (
                "TODO/README.md -> the \"Last-updated trailer (Wf-A153 census-republish + "
                "artifact-path + guard-blind-spot + frontier-comparability wave -- "
                "2026-11-01 -- ...)\" line -- a TEXT anchor, not a line number"
            ),
        },
        {
            "site": "molmetal/reports/wfa175_frontier_comparability/report.md -> LIVE section 4.2",
            "superseded_locator_verbatim": "TODO/INDEX.md:3",
            "superseded_line_number": 3,
            "the_cited_line_now_reads_verbatim_prefix": '**Last-updated trailer (Wf-A172 dead-gate history CORRECTION + pre-flight live-state + config-gate-requirement wave — 2026-11-01 — ...',
            "durable_locator_now": (
                "TODO/INDEX.md -> the \"Last-updated trailer (Wf-A153 census-republish + "
                "artifact-path + guard-blind-spot + frontier-comparability wave -- "
                "2026-11-01 -- ...)\" line -- a TEXT anchor, not a line number"
            ),
        },
        {
            "site": "molmetal/reports/wfa175_frontier_comparability/report.md -> LIVE section 4.2",
            "superseded_locator_verbatim": "molmetal/reports/wfa161_frontier_labelling/report.md:139",
            "superseded_line_number": 139,
            "the_cited_line_now_reads_verbatim_prefix": "2. **Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDK ...",
            "durable_locator_now": (
                "molmetal/reports/wfa161_frontier_labelling/report.md -> SECTION HEADING "
                "'## Wf-A176 CORRECTION (2026-11-01) -- \u00a7\"Comparability breaks\" "
                "item 2 is WRONG' -> SUB-HEADING '### The superseded sentence "
                "(verbatim, line 139 of this file)'"
            ),
        },
        {
            "site": "molmetal/reports/wfa175_frontier_comparability/report.md -> LIVE section 4.2",
            "superseded_locator_verbatim": (
                "paper/sections/04_evaluation.tex:3179 and "
                "paper/sections/06_limitations.tex:387"
            ),
            "superseded_line_numbers": [3179, 387],
            "durable_locator_now": (
                "paper/sections/04_evaluation.tex -> \\paragraph{YuelBond decoder swap: "
                "\\textsc{available} parallel path (WF-CFM-Rescue, 2026-09-16).}; "
                "paper/sections/06_limitations.tex -> \\section{Limitations} "
                "\\label{sec:limitations} -> the \\item 'YuelBond decoder swap not "
                "wired; CPU + GPU 500-step smokes bit-exact with BondAwareDecoder "
                "baseline' -- LaTeX anchors, not line numbers"
            ),
        },
        {
            "site": "molmetal/reports/wfa175_frontier_comparability/report.md -> section C.1 (SELF-locators)",
            "superseded_locator_verbatim": "report.md:79 and report.md:186",
            "superseded_line_numbers": [79, 186],
            "why_perishable": (
                "both pointed at THIS FILE'S OWN live body, so they rot on every edit "
                "made above them"
            ),
            "durable_locator_now": (
                "report.md -> SECTION HEADING '### 2.1 Connectivity axis (fragment "
                "selection)' (the \"how POST is obtained\" column) and SECTION HEADING "
                "'## 6. New comparability findings this wave adds (all MEASURED)' -> "
                "numbered item 2"
            ),
        },
        {
            "site": (
                "molmetal/reports/wfa175_frontier_comparability/refutation/refutation.md "
                "-> the claim-3 table row, and refutation/refutation.json -> "
                "verdicts.CLAIM_3.what_i_checked[3] and .counter_evidence[3]"
            ),
            "superseded_locator_verbatim": "molmetal/scripts/wfa161_frontier_labelling.py:934",
            "superseded_line_number": 934,
            "the_cited_line_now_reads_verbatim_prefix": '     validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,',
            "durable_locator_now": (
                "molmetal/scripts/wfa161_frontier_labelling.py -> PYTHON CONSTANT "
                "'WFA176_CORRECTION_SECTION' -> its '> 2. **Post-processing.** ...' "
                "line; the string now lives at line 913 inside that DATED constant"
            ),
        },
        {
            "site": "molmetal/reports/wfa175_frontier_comparability/tools/build.py -> BAND_CENSUS_SUMMARY.bare_scalar_note",
            "superseded_locator_verbatim": (
                "paper/sections/04_evaluation.tex:3179 and "
                "paper/sections/06_limitations.tex:387"
            ),
            "superseded_line_numbers": [3179, 387],
            "durable_locator_now": (
                "the same LaTeX anchors as the report.md row above -- LaTeX anchors, "
                "not line numbers"
            ),
        },
    ],
    "primary_source_reverification_2026_11_02": {
        "method": (
            "arXiv e-prints re-fetched live on 2026-11-02 through the host HTTPS "
            "proxy (https://arxiv.org/e-print/<id>), every tarball and every file "
            "pinned by sha256; the four per-file sha256 below are BYTE-IDENTICAL "
            "to the pins already recorded in "
            "molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json"
            " -> JSON PATH retrieval[0..2].files[]."
        ),
        "tarball_sha256": {
            "2305.01140v1": "b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23",
            "2603.27113v1": "87c1788e9e9826cfbdf4e0788feed1a796c8a50218eecc6905df771f10c9c329",
            "2609.08333v4": "e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224",
        },
        "file_sha256_read": {
            "2305.01140v1/icml2023.tex": "b469275af5cd693297147aad811a8b68c20bc93d356894c9409a46a891283b0b",
            "2603.27113v1/tables/qm9_benchmarks_og.tex": "1c81ad499093f9d20d512e29787e934ab68c6d32dc5bb3b791970830c05c8f39",
            "2603.27113v1/iclr2025_conference.tex": "60122a972df76e9185d9f22b01c93d1535f5a7e7fe821abd16f5920cfc77d857",
            "2609.08333v4/src/appendix.tex": "bf82347838d0d2cd1cf21505d12b10c4bced8cb602a628aa2318f086759433dc",
        },
        "cells_measured": {
            "92.7": "HLTF arXiv 2603.27113v1 tables/qm9_benchmarks_og.tex sub-table tab:qm9_pb row 'Val&Uniq (%)' GeoLDM column = $92.7\\pm0.5$; the SAME cell is GeoLDM's own 'Valid & Unique (%)' 92.7 +- 0.5 at icml2023.tex:491 -- the two primary sources AGREE.",
            "93.8": "GeoLDM's OWN paper prints Valid (%) = 93.8 +- 0.4 (icml2023.tex:491); HLTF's tab:qm9_pb prints GeoLDM's Val (%) = 93.6 +- 0.2 and HLTF's OWN Val (%) = 93.8 +- 0.6.  The numeral 93.8 therefore names THREE different cells across the two papers -- it is NOT a band endpoint under either reading.",
            "94.0": "the numeral 94.0 appears LITERALLY only in HLTF's abstract (iclr2025_conference.tex:55, 'improving PoseBusters validity to 94.0%'); the table cell prints '94\\pm0.5' (qm9_benchmarks_og.tex:66, row 'PB-Valid (%)', HLTF column).  Same cell, two renderings.",
            "92.98": "EF-TALFM arXiv 2609.08333v4 src/appendix.tex table tbl:pcqm4mv2_generation (a), row 'C.' (the fraction of connected molecules), EF-TALFM column = 92.98.",
            "94.33": "the same table, row 'N.&U.&S.' (sanitized, unique, training-set-novel, before the joint PoseBusters pass), EF-TALFM column = 94.33.",
            "98.9": "GeoLDM's OWN 'Atom Sta (%)' = 98.9 +- 0.1 (icml2023.tex:491) = HLTF's tab:qm9_pb row 'AS (%)' GeoLDM column (98.9\\pm0.1).  The two primary sources AGREE.",
            "10000": "the denominator of EF-TALFM's panel (a): 'Panel (a) rates are computed over 10,000 generated molecules' (src/appendix.tex:93; the additional table repeats it at :184).",
        },
        "reading_of_the_cells": (
            "the captions decide RAW vs POST: src/appendix.tex:183 reports "
            "'connectivity of the reconstructed molecule' SEPARATELY from the "
            "sanitization/uniqueness/novelty rates, which are 'computed after "
            "retaining the atom-count largest connected component'.  A retained "
            "largest component would force connectivity to 100 %, so EF-TALFM's "
            "C. = 92.98 is a RAW count and its N.&U.&S. = 94.33 is a POST "
            "count -- the repo's reading B survives the re-verification."
        ),
        "disagreements_found": [
            "GeoLDM's OWN paper prints its QM9 Valid (%) as 93.8 +- 0.4 "
            "(icml2023.tex:491); HLTF's tab:qm9_pb prints GeoLDM's Val (%) as "
            "93.6 +- 0.2 (qm9_benchmarks_og.tex:63).  TWO PRIMARY SOURCES "
            "DISAGREE about the SAME cell.  Both numerals are already recorded "
            "in the Wf-A186 record (cells[\"2305.01140v1\"][56] and "
            "cells[\"2603.27113v1\"][54]); the repo's own `note_on_93_8` names "
            "only the 93.8 pair.",
            "'94.0' is an ABSTRACT rendering of a table cell that prints "
            "'94\\pm0.5'; quoting 94.0 as a table cell value is not what the "
            "table says.",
        ],
    },
}

PRIMARY_SOURCE_REVERIFICATION_CORRECTION_2026_11_02_WFA200F = {
    "dated": "2026-11-02",
    "wave": "Wf-A200f",
    "scope": (
        "AGENTS.md section 5.5 rule #6 -- APPEND-ONLY + DATED.  This key is "
        "ADDITIVE: the string it corrects is PRESERVED VERBATIM where it "
        "stands (crosswave_locator_repair_2026_11_02_wfa200 -> "
        "primary_source_reverification_2026_11_02 -> cells_measured['94.0']) "
        "and is NOT rewritten."
    ),
    "what_was_wrong": (
        "the cells_measured['94.0'] string says the numeral 94.0 'appears "
        "LITERALLY only in HLTF's abstract'.  That is UNQUALIFIED, and read "
        "over the frontier corpus this file pins it is FALSE: MEASURED on "
        "disk, the numeral 94.0 appears LITERALLY in a SECOND pinned "
        "e-print -- NEAT-POCKET arXiv 2609.05097v1."
    ),
    "superseded_rendering_verbatim": (
        "the numeral 94.0 appears LITERALLY only in HLTF's abstract "
        "(iclr2025_conference.tex:55, 'improving PoseBusters validity to "
        "94.0%'); the table cell prints '94\\pm0.5' "
        "(qm9_benchmarks_og.tex:66, row 'PB-Valid (%)', HLTF column).  Same "
        "cell, two renderings."
    ),
    "measured_counter_example": {
        "eprint": "NEAT-POCKET arXiv 2609.05097v1",
        "eprint_sha256": (
            "bd978841f62a7d75e8635714e7336a6e3abec39b854ec3f2c90974890f2b7fdb"
        ),
        "eprint_bytes": 10632429,
        "file": "sections/results.tex",
        "file_bytes_in_the_pinned_tarball": 40202,
        "sha256_of_the_on_disk_extracted_text": (
            "d9d65a1db56b9c9ed5a8ad2bd48c0623a5a281377f381299b36d79129ef96e5d"
        ),
        "bytes_of_the_on_disk_extracted_text": 40208,
        "why_two_byte_counts": (
            "the cached .txt.gz is the whole tarball FLATTENED to text and is "
            "re-encoded, so it is 6 bytes above the tarball member's recorded "
            "40202; the sha256 above is of the ON-DISK EXTRACTED TEXT, not of "
            "the tarball member.  The tarball itself is not retained in this "
            "tree (only the 3 e-prints re-fetched by Wf-A200 are), so the "
            "member's own digest is not re-verifiable here."
        ),
        "verbatim_line_23": "CrossDocked & 94.0 $\\pm$ 4.7",
        "verbatim_line_23_note": "leading indentation of the table cell trimmed",
        "verbatim_line_126": (
            "Overall, 94.0\\% of the CrossDocked test-set ligands are PB valid."
        ),
        "the_branchs_own_record_already_carried_the_number": (
            "sources[5].metrics[0].value names 'CrossDocked reference "
            "ligands 94.0' and sources[5].metrics[0].locator is "
            "'sections/results.tex:126, :132, :170' -- so the number was "
            "already in this file; only the 'LITERALLY only' scope sentence "
            "was wrong."
        ),
    },
    "corrected_rendering": (
        "SCOPED: WITHIN HLTF arXiv 2603.27113v1 the numeral 94.0 appears "
        "literally only in the abstract (iclr2025_conference.tex:55, "
        "'improving PoseBusters validity to 94.0%'), while the table cell "
        "prints '94\\pm0.5' (tables/qm9_benchmarks_og.tex:66, row "
        "'PB-Valid (%)', HLTF column) -- same cell, two renderings.  OVER "
        "THE FRONTIER CORPUS the numeral 94.0 is NOT unique to HLTF: "
        "NEAT-POCKET arXiv 2609.05097v1 prints '94.0 \\pm 4.7' "
        "(sections/results.tex:23) and 'Overall, 94.0\\% of the "
        "CrossDocked test-set ligands are PB valid' "
        "(sections/results.tex:126)."
    ),
    "what_the_counter_example_does_and_does_not_change": (
        "It changes NO reading of the band.  NEAT-POCKET's 94.0 is a "
        "DIFFERENT quantity -- the PB-validity of the CrossDocked TEST-SET "
        "REFERENCE LIGANDS (real docked molecules), not a generated-molecule "
        "endpoint, and not HLTF's cell -- on a DIFFERENT test set.  It is a "
        "THIRD candidate for the numeral 94.0, listed next to the branch's "
        "existing third candidate for 92.7 (YuelBond's per-bond-type "
        "classification F1 92.7 %).  The band remains a frontier composite "
        "whose endpoint cells are NOT settled on disk: reading A (92.7 / "
        "94.0) and reading B (92.98 / 94.33) still disagree on the numerals."
    ),
    "no_numeral_changed": (
        "92.7, 93.8, 94.0, 92.98, 94.33, 98.9 and 10000 all keep their "
        "measured values; only the SCOPE of one sentence about the numeral "
        "94.0 is corrected.  No source record, no evidence quote and no "
        "verdict was edited."
    ),
    "verified_with": (
        "molmetal/reports/wfa175_frontier_comparability/refutation/tools/"
        "show.py over the pinned cache/2609.05097v1.txt.gz, on 2026-11-02, "
        "CPU-only.  NO network call was made for this correction: the "
        "counter-example was read from the sha256-pinned extraction already "
        "on disk."
    ),
}

COMPUTED_RATIOS = {
    "our_ratio": OUR_RATIO,
    "our_raw": RAW,
    "our_post": POST,
    "frontier_max_printed_connectivity_ratio": {
        "model": "GeoLDM",
        "source": "arXiv:2505.00518v1 tables/chemical_validity.tex:17 and :23",
        "raw": 37.4,
        "post": 73.8,
        "ratio": BB_RATIOS["GeoLDM"]
    },
    "our_over_frontier_max": OUR_RATIO / max(BB_RATIOS.values()),
    "next_largest_connectivity_ratio_implied": {
        "model": "DiffSBDD-cond (Binding MOAD)",
        "raw": 63.37,
        "post": 100.0,
        "ratio": DSBDD_RATIOS["Binding MOAD / DiffSBDD-cond (full pocket)"],
        "note": "POST value implied by the paper's own largest-fragment filter, not printed"
    },
    "largest_raw_to_post_ratio_on_OTHER_axes": {
        "model": "DiffSBDD Validity3D",
        "raw": 0.004,
        "post": 0.532,
        "ratio": GB3D_RATIOS["DiffSBDD"],
        "source": "arXiv:2407.04424v1 PDF Table 3 p.23 -> Table S3 p.49 (MMFF94s relaxation)"
    },
    "eqgat_repair_axis_ratio": {
        "raw": 0.916,
        "post": 0.998,
        "ratio": r(*EQGAT_VALIDITY)
    }
}

# The dated supersession record: every string below is preserved VERBATIM (AGENTS.md 5.5 rule #6).
SUPERSEDED_NOTES_2026_11_01 = {
    "dated": "2026-11-01",
    "wave": "Wf-A186",
    "scope": "AGENTS.md section 5.5 rule #6 -- APPEND-ONLY. Every string below is preserved VERBATIM; nothing was deleted and no superseded rendering was silently rewritten. They are superseded by the reading-safe band note now carried in `sources`.GeoLDM.note and `sources`.HLTF.note.",
    "why_superseded": "The published band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk.  Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.",
    "superseded_GeoLDM_note_verbatim": "CRITICAL for the band: the band endpoints 92.7 and 93.8 are GeoLDM's OWN QM9 cells from ONE table row, and 89.4 is GeoLDM's own QM9 molecule-stability cell.",
    "superseded_HLTF_note_verbatim": "The 92.7-94 % band is minted here: two endpoints, two different metrics, two different models, one table. The same table also carries GeoLDM's MS 89.4 %, the adjacent cell that migrated onto EF-TALFM as '89.4'.",
    "superseded_HLTF_metric_names_verbatim": {
        "GeoLDM Val&Uniq (%) -- the LOWER endpoint of the published 92.7-94 % band": "GeoLDM Val&Uniq (%) -- the LOWER endpoint of the published 92.7-94 % band",
        "HLTF (ours) PB-Valid (%) -- the UPPER endpoint of the published 92.7-94 % band": "HLTF (ours) PB-Valid (%) -- the UPPER endpoint of the published 92.7-94 % band"
    },
    "superseded_GeoLDM_note_supplement_2026_11_01_wfa175_frontier_verbatim": "APPEND-ONLY supplement (Wf-A175 FRONTIER, 2026-11-01): the earlier sentence above is correct about 92.7 and 93.8 being GeoLDM's OWN QM9 cells, but the BAND '92.7-94 %' as published is minted in a DIFFERENT table -- HLTF arXiv:2603.27113v1 tables/qm9_benchmarks_og.tex:56-66 (Table 1(b) tab:qm9_pb), whose lower endpoint 92.7 % is GeoLDM's Val&Uniq (line 64) and whose upper endpoint 94 % is HLTF's OWN PB-Valid (line 66). Different models, different metrics, same table.",
    "why_the_GeoLDM_supplement_is_also_superseded": "That dated supplement names ONE settled provenance for the band ('the BAND 92.7-94 % as published is minted in a DIFFERENT table ... whose lower endpoint 92.7 % is GeoLDM's Val&Uniq ... whose upper endpoint 94 % is HLTF's OWN PB-Valid').  The same reading that `report.md` section D.3 withdraws.  It is preserved VERBATIM in place and is NOT deleted; this entry records that it is superseded by the reading-safe note.",
    "note_on_93_8": "93.8 is NOT a band endpoint under either reading.  On disk it is a QUOTED FRONTIER CELL: GeoLDM's OWN QM9 Valid (%) = 93.8 +/- 0.4 (its own e-print, icml2023.tex:491, per the GeoLDM source record in this file), and, in `report.md` section 4.1's transcription of HLTF Table 1(b), HLTF's OWN Val (%) = 93.8 +- 0.6.  The branch's own text does not settle 93.8 to a single cell.  89.4 +/- 0.5 is GeoLDM's OWN QM9 Mol. Stability cell, not a validity endpoint.",
    "locators": {
        "competing_readings_record": "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json",
        "reading_A_json_path": "band_provenance",
        "reading_B_json_path": "sources`.`EF-TALFM",
        "canonical_reading_safe_band_statement_json_path": "band_provenance`.`competing_reading_and_what_this_does_not_settle`.`canonical_reading_safe_band_statement",
        "locator_warning": "Cite JSON PATHS, not line numbers.  `frontier_axis_mapping` is NOT a key of that record -- the token survives only inside a string value, so pointing a reader at it is a DEFECT."
    }
}

# ---------------------------------------------------------------------------
# Wf-A194 (2026-11-02) -- the LIVE reading of the band's endpoint cells.
#
# `report.md` is HAND-AUTHORED: no script in this branch emits it (build.py
# writes `sources.json` and `tools/ratios.json` only -- MEASURED 2026-11-02).
# Its section 4.1 body nevertheless ASSERTED ONE provenance reading as settled
# ("So the band is minted here: * lower endpoint 92.7 % = GeoLDM's Val&Uniq
# ... * upper endpoint 94 % = HLTF's own PB-Valid"), while this producer has
# carried the reading-safe statement since Wf-A186.  The producer knew; the
# live body did not.
#
# THE REPAIR: the live body now carries the reading-safe rendering, and this
# generator emits it -- as a MACHINE field -- so the two cannot drift apart
# again, and so a re-run reproduces the repair instead of erasing it.
#
# The SUPERSEDED rendering is preserved VERBATIM in two dated places and is
# NOT deleted:
#   * `report.md` section D.1's three fenced blocks (unchanged by Wf-A194), and
#   * this file's `superseded_notes_2026_11_01_wfa186_frontier` ->
#     `superseded_HLTF_note_verbatim` (unchanged by Wf-A194).
#
# No numeral is introduced below that is not already on disk in this branch.
# ---------------------------------------------------------------------------

LIVE_SECTION_4_1_HEADING = (
    "### 4.1 Source layer -- TWO competing readings of the band's endpoint "
    "cells; neither is settled on disk"
)

LIVE_SECTION_4_1_RENDERING = (
    "Which frontier cells the band names is **NOT settled on disk**.  The repo "
    "holds TWO competing provenance readings, and they do not even agree on the "
    "numerals:\n"
    "\n"
    "* **reading A** (`band_provenance` in the frontier post-processing matrix "
    "record `molmetal/reports/wfa158_frontier_postproc_matrix/"
    "frontier_matrix.json`): **lower endpoint 92.7 %** = **GeoLDM's Val&Uniq** "
    "(a *chemical* QM9 graph metric) and **upper endpoint 94 %** = **HLTF's own "
    "PB-Valid** (a *3D PoseBusters* metric, a different test set) -- two cells "
    "of HLTF's Table 1(b), **both `post_processed: false`**; YuelBond's "
    "per-bond-type "
    "classification F1 92.7 % is a third candidate for the same numeral;\n"
    "* **reading B** (`sources`.`EF-TALFM`): EF-TALFM arXiv:2609.08333v4 "
    "Table 3(a) -- RAW connectivity `C. = 92.98 %`, counted **BEFORE** the "
    "largest-component retention, with the post-processed `N. & U. & S. = "
    "94.33 %` companion cell -- so the frontier arm is **MIXED** under this "
    "reading.\n"
    "\n"
    "Under **BOTH** readings the band is **EXTERNAL**, is **NOT one homogeneous "
    "quantity**, and is **NOT on our scale** -- OUR 8.51 % IS a "
    "connected-component count on the RAW emitted SMILES, and nothing else.  "
    "Neither reading may be published as the band's settled provenance.  The "
    "other members the repo swept -- 92.98 (EF-TALFM connectivity), 94.33 "
    "(EF-TALFM yield), 93.13 (EF-TALFM all-PoseBusters-checks), 93.1 (GeoLDM "
    "PB-Valid), 93.3 (GCDM Val&Uniq), 92.9 (HLTF Val&Uniq) -- are additional "
    "**QUOTED FRONTIER CELLS** of the same heterogeneous family, not settled "
    "band endpoints; the EF-TALFM ones are confirmed in its primary record at "
    "`src/appendix.tex:82`, `:89`, `:92`, `:115`, `:127`."
)

LIVE_CONCLUSION_C_RENDERING = (
    "**Conclusion (c):** the band is a frontier composite whose endpoint cells "
    "are NOT settled on disk -- the repo holds TWO competing provenance "
    "readings (92.7/94.0 under `band_provenance` vs 92.98/94.33 under "
    "`sources`.`EF-TALFM`) whose numerals disagree, and under BOTH readings "
    "the band is external, is not one homogeneous quantity, and is not on our "
    "scale; it entered this repository only in the 2026-11-01 working tree (0 "
    "committed renderings), was labelled as the frontier's from its first "
    "metric artifact, was briefly mis-labelled \"our band\" in the Wf-A161 "
    "report, and has never reached `paper/` in any rendering."
)

LIVE_SECTION_4_1_READING_SAFE_2026_11_02_WFA194 = {
    "dated": "2026-11-02",
    "wave": "Wf-A194",
    "scope": (
        "AGENTS.md section 5.5 rule #6 -- APPEND-ONLY.  This record is ADDITIVE: "
        "the superseded rendering is preserved VERBATIM in `report.md` section "
        "D.1's fenced blocks and in this file's "
        "`superseded_notes_2026_11_01_wfa186_frontier` -> "
        "`superseded_HLTF_note_verbatim`; nothing was deleted."
    ),
    "what": (
        "the LIVE rendering of `report.md` section 4.1 and of section 4's "
        "Conclusion (c) -- the two places where the report asserted ONE "
        "provenance reading as settled"
    ),
    "reading_safe_band_statement": BAND_PROVENANCE_NOTE,
    "live_section_4_1_heading": LIVE_SECTION_4_1_HEADING,
    "live_section_4_1_rendering": LIVE_SECTION_4_1_RENDERING,
    "live_conclusion_c_rendering": LIVE_CONCLUSION_C_RENDERING,
    "superseded_rendering_preserved_verbatim_at": [
        "molmetal/reports/wfa175_frontier_comparability/report.md section D.1 "
        "-- three fenced blocks (heading, 'So the band is minted here:' "
        "passage, Conclusion (c)); unchanged by Wf-A194",
        "molmetal/reports/wfa175_frontier_comparability/sources.json -> "
        "superseded_notes_2026_11_01_wfa186_frontier -> "
        "superseded_HLTF_note_verbatim",
    ],
    "why_superseded": (
        "the report's section 4.1 body and Conclusion (c) named ONE provenance "
        "reading (GeoLDM Val&Uniq 92.7 % as the lower endpoint, HLTF PB-Valid "
        "94 % as the upper endpoint, 'two cells of one table in HLTF') as if it "
        "were settled.  The published band's provenance is NOT settled on disk: "
        "the frontier post-processing matrix record holds TWO competing "
        "readings -- `band_provenance` and `sources`.`EF-TALFM` -- and they do "
        "not even agree on the numerals (92.7/94.0 vs 92.98/94.33).  The "
        "corrected rendering states both readings, keeps the invariants that "
        "hold under BOTH (external, not one homogeneous quantity, not on our "
        "scale), and introduces NO numeral that is not already on disk."
    ),
    "number_preserving": (
        "TEXT-ONLY: the report's numeric leaves are UNCHANGED -- 92.7, 94, "
        "94.0, 92.98, 94.33, 93.13, 93.1, 93.3, 92.9 and 8.51 all already "
        "appear in the branch's own records; the repair moves NO number."
    ),
    "producer_parity": (
        "this record is EMITTED by this generator, so a re-run reproduces it "
        "instead of dropping it.  `report.md` itself is hand-authored (no "
        "script in this branch emits it -- MEASURED 2026-11-02), so its live "
        "body is checked against `live_section_4_1_rendering` and "
        "`live_conclusion_c_rendering` below by "
        "molmetal/tests/test_wfa192_live_x_closure.py."
    ),
}

# ------------------------------------------------------------------ write out
payload = {
    "artifact": "wfa175_frontier_comparability",
    "wave": "Wf-A175",
    "branch": "FRONTIER",
    "generated_utc": "2026-11-01",
    "gpu": "0 (CPU-only; no torch import, no CUDA call, no decode, no checkpoint read)",
    "retrieval_method": "urllib over the host HTTPS proxy http://127.0.0.1:7890; arXiv e-print redirect /e-print/<id> -> /src/<id> followed by urllib; arXiv API via export.arxiv.org/api/query with %22-phrase encoding",
    "primary_record_rule": "every number below was read from bytes fetched live on 2026-11-01 and pinned by sha256; no number comes from a search-result snippet",
    "sources": S,
    "band_census": BAND_CENSUS_SUMMARY,
    "computed_ratios": COMPUTED_RATIOS,
    "superseded_notes_2026_11_01_wfa186_frontier": SUPERSEDED_NOTES_2026_11_01,
    "live_section_4_1_reading_safe_2026_11_02_wfa194": (
        LIVE_SECTION_4_1_READING_SAFE_2026_11_02_WFA194
    ),
    "crosswave_locator_repair_2026_11_02_wfa200": (
        CROSSWAVE_LOCATOR_REPAIR_2026_11_02_WFA200
    ),
    "primary_source_reverification_correction_2026_11_02_wfa200f": (
        PRIMARY_SOURCE_REVERIFICATION_CORRECTION_2026_11_02_WFA200F
    ),
}
blob = json.dumps(payload, indent=1, ensure_ascii=False)
p1 = os.path.join(ROOT, "sources.json")
if os.path.exists(p1):
    print("overwrite risk:", safe_overwrite_check(p1, blob))
print("truncation:", detect_truncation(blob))
atomic_write_file(p1, blob)
print("wrote", p1, len(blob), "bytes")
json.dump({"BB": BB, "BB_RATIOS": BB_RATIOS, "DSBDD": DSBDD, "DSBDD_RATIOS": DSBDD_RATIOS,
           "GB3D": GB3D, "GB3D_RATIOS": GB3D_RATIOS, "EQGAT_VALIDITY": EQGAT_VALIDITY,
           "EQGAT_CONN": EQGAT_CONN, "TABASCO_CONN": TABASCO_CONN,
           "RAW": RAW, "OUR_RATIO": OUR_RATIO},
          open(os.path.join(ROOT, "tools", "ratios.json"), "w"), indent=1)
print(json.dumps({"our_ratio": OUR_RATIO,
                  "frontier_max_conn": max(BB_RATIOS.items(), key=lambda kv: kv[1]),
                  "our_over_frontier": OUR_RATIO / max(BB_RATIOS.values()),
                  "dsbdd_max": max(DSBDD_RATIOS.items(), key=lambda kv: kv[1]),
                  "gb3d_max": max(GB3D_RATIOS.items(), key=lambda kv: kv[1]),
                  "eqgat_validity_ratio": r(*EQGAT_VALIDITY)}, indent=1))
