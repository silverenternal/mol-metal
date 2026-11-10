#!/usr/bin/env python3
"""Wf-A175 — write report.md + refs_corrected_STAGED.bib via atomic_write_file (AGENTS.md 5.6)."""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
RD = REPO / "molmetal" / "reports" / "wfa175_refs_bib_audit"
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))
from wf386a3_atomic_write import atomic_write_file  # noqa: E402

STAGED = r"""% =====================================================================
%  STAGED BIBTEX — NOT APPLIED TO paper/refs.bib
%  Wf-A175 / BIBLIOGRAPHY branch, 2026-11-01
%  Produced by: molmetal/reports/wfa175_refs_bib_audit/audit_refs_bib.py
%                + resolve_targets{2..9}.py  (+ build_audit.py)
%
%  *** THIS FILE IS A PROPOSAL, NOT A PATCH. ***
%  Nothing here has been written to paper/refs.bib. Applying it means
%  editing paper/** , which AGENTS.md 5.3 / TODO-84 forbids without the
%  USER'S EXPLICIT OK. The user must approve before any of it lands.
%
%  Every DOI in this file was resolved against a PRIMARY record on
%  2026-11-01:
%    - doi.org handle registry  https://doi.org/api/handles/<doi>
%    - doi.org landing page      https://doi.org/<doi>
%    - Crossref                 https://api.crossref.org/works/<doi>
%    - bioRxiv API              https://api.biorxiv.org/details/biorxiv/<doi>
%    - arXiv API                http://export.arxiv.org/api/query?id_list=<id>
%    - PubMed/NLM               https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
%    - publisher landing pages  journals.plos.org, ojs.aaai.org
%
%  Per AGENTS.md 5.5 rule 6 the superseded rendering of every entry that
%  is changed or removed is preserved VERBATIM below, dated, and never
%  silently deleted.
%
%  SCOPE: this file is the proposal for paper/refs.bib only. The two
%  FABRICATED entries in .tools/sci-paper/refs/references.bib
%  (krasnov2026metalcytotoxdb, rusanov2025platinai) are staged in the
%  separate file refs_sci_paper_corrected_STAGED.bib, so that no cite
%  key is defined twice in one .bib.
% =====================================================================


% =====================================================================
%  SECTION 1 — THE 3 GATE-REQUIRED KEYS (all 3 are currently ABSENT from
%  paper/refs.bib; gate = gate_p3_protocol_alignment_baseline_paper_refs_bib,
%  molmetal/scripts/pre_flight_check.py:11052)
% =====================================================================

% --- 1a. krasnov2026metalcytotoxdb ---
% NOTE: this cite key names the DATABASE (MetalCytoToxDB), but the
% version of record is the JOURNAL ARTICLE that introduces the database.
% Verified: PubMed PMID 41969144; Crossref 10.1021/acs.jmedchem.5c02755
% (registered; J. Med. Chem. 2026, 69(8), 8838-8851). The PubMed abstract
% reads: "we have manually curated MetalCytoToxDB-a comprehensive
% experimental database comprising 26,500 IC50 values for 7050 metal
% complexes against 754 cell lines from 1921 articles".
@article{krasnov2026metalcytotoxdb,
  author       = {Krasnov, Lev and Malikov, Dmitry and Kiseleva, Marina A. and Nykhrikova, Ekaterina V. and Tatarin, Sergei V. and Bezzubov, Stanislav I.},
  title        = {Machine Learning Approach to Anticancer Activity Prediction of Transition-Metal Complexes Based on a Large-Scale Experimental Database},
  journal      = {Journal of Medicinal Chemistry},
  volume       = {69},
  number       = {8},
  pages        = {8838--8851},
  year         = {2026},
  publisher    = {American Chemical Society},
  doi          = {10.1021/acs.jmedchem.5c02755},
  url          = {https://pubs.acs.org/doi/10.1021/acs.jmedchem.5c02755},
  note         = {Introduces the MetalCytoToxDB experimental database: 26,500 IC50 values, 7,050 metal complexes, 754 cell lines, from 1,921 articles (counts quoted from the PubMed abstract, PMID 41969144).}
}

% --- 1b. rusanov2025platinai ---
% NOTE: "PlatinAI" is ONE de-novo-designed bis-carbene platinum(II)
% COMPLEX, not a database name (the ChemRxiv abstract says so verbatim);
% the curated dataset behind it is 4,078 platinum complexes / 18,995 IC50
% values. The .tools/sci-paper entry calling it "A 5K-Compound Platinum
% Cytotoxicity Dataset" and citing Scientific Data 12(184) with doi
% 10.1038/s41597-025-05012-9 is FABRICATED - that DOI is not registered.
% Verified: ChemRxiv preprint doi 10.26434/chemrxiv-2025-pp32k (registered,
% Crossref posted-content 2025); counts quoted from that record's abstract.
@article{rusanov2025platinai,
  author       = {Rusanov, Daniil A. and Brezgunov, Dmitrii Yu. and Matnurov, Egor M. and Pashaliev, Bekir L. and Vetrova, Varvara V. and Babak, Maria V.},
  title        = {Overcoming cisplatin resistance via deep learning-assisted de novo design of platinum complexes},
  year         = {2025},
  publisher    = {American Chemical Society},
  doi          = {10.26434/chemrxiv-2025-pp32k},
  url          = {https://chemrxiv.org/engage/chemrxiv/article-details/10.26434/chemrxiv-2025-pp32k},
  note         = {ChemRxiv preprint (not yet a journal version of record as of 2026-11-01). Curated dataset: 4,078 platinum complexes with 18,995 IC50 values; PlatinAI is the single de-novo-designed bis-carbene platinum(II) complex the paper validates (numbers and wording quoted from the record's abstract).}
}

% --- 1c. ji2022drugood ---
% Verified on AAAI's own OJS article page
% (https://ojs.aaai.org/index.php/AAAI/article/view/25970 -> citation_title /
% citation_volume=37 / citation_issue=7 / citation_firstpage=8023 /
% citation_lastpage=8031) and via the arXiv API (arXiv:2201.09637,
% 2022-01-24). The registered DOI is 10.1609/aaai.v37i7.25970; the
% frequently seen 10.1609/aaai.v37i.v37i7.25970 (doubled "v37i") is NOT
% registered.
@inproceedings{ji2022drugood,
  author       = {Ji, Yuanfeng and Zhang, Lu and Wu, Jiaxiang and Wu, Bingzhe and Li, Lanqing and Huang, Long-Kai and Xu, Tingyang and Rong, Yu and Ren, Jie and Xue, Ding and Lai, Houtim and Liu, Wei and Huang, Junzhou and Zhou, Shuigeng and Luo, Ping and Zhao, Peilin and Bian, Yatao},
  title        = {DrugOOD: Out-of-Distribution Dataset Curator and Benchmark for AI-Aided Drug Discovery -- A Focus on Affinity Prediction Problems with Noise Annotations},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume       = {37},
  number       = {7},
  pages        = {8023--8031},
  year         = {2023},
  publisher    = {Association for the Advancement of Artificial Intelligence},
  doi          = {10.1609/aaai.v37i7.25970},
  eprint       = {2201.09637},
  archivePrefix = {arXiv},
  url          = {https://ojs.aaai.org/index.php/AAAI/article/view/25970},
  note         = {Preprint version: arXiv:2201.09637 (2022-01-24), titled 'DrugOOD: Out-of-Distribution (OOD) Dataset Curator and Benchmark for AI-aided Drug Discovery -- A Focus on Affinity Prediction Problems with Noise Annotations'.}
}


% =====================================================================
%  SECTION 2 — CORRECTED ENTRIES OF paper/refs.bib
%  (each superseded rendering is quoted VERBATIM first)
% =====================================================================

% --- 2a. buttenschoen2024posebusters — REMOVED (fabricated duplicate) ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original, retained for the
% record per AGENTS.md 5.5 rule 6:
%   @article{buttenschoen2024posebusters,
%     author       = {Buttenschoen, Martin and Morris, Garrett M. and Deane, Charlotte M.},
%     title        = {{PoseBusters}: {AI}-based docking methods fail to generate physically valid poses or generalise to novel sequences},
%     journal      = {Nature Communications},
%     volume       = {15},
%     number       = {1},
%     pages        = {6621},
%     year         = {2024},
%     publisher    = {Nature Publishing Group},
%     doi          = {10.1038/s41467-024-49069-0},
%     url          = {https://www.nature.com/articles/s41467-024-49069-0},
%     note         = {Version-of-record with the full PB-test suite used by \texttt{molmetal/validation/posebusters\_runner.py}; we also cite the open-access Chem.~Sci.~2023 preprint \cite{posebusters2023}.}
% REASON: doi 10.1038/s41467-024-49069-0 is NOT REGISTERED (doi.org handle
% API responseCode != 1; doi.org landing HTTP 404 "Error: DOI Not Found";
% Crossref HTTP 404). No Nature-Communications PoseBusters article is
% attested: Crossref title search "PoseBusters" returns only
% 10.1039/d3sc04185a (+ peer-review decision records) and PubMed
% "PoseBusters"[Title] returns exactly ONE hit (PMID 38425520, Chem. Sci.
% 2024, 15(9), 3130-3139). The entry duplicates posebusters2023 with a
% fabricated venue and a fabricated page number, so it is REMOVED rather
% than repaired. The PB-test-suite note is preserved in 2b below.
%
% --- 2b. posebusters2023 — year corrected 2023 -> 2024 ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @article{posebusters2023,
%     author       = {Buttenschoen, Martin and Morris, Garrett M. and Deane, Charlotte M.},
%     title        = {{PoseBusters}: {AI}-based docking methods fail to generate physically valid poses or generalise to novel sequences},
%     journal      = {Chemical Science},
%     volume       = {15},
%     number       = {9},
%     pages        = {3130--3139},
%     year         = {2023},
%     publisher    = {Royal Society of Chemistry},
%     doi          = {10.1039/D3SC04185A},
%     url          = {https://pubs.rsc.org/en/content/articlelanding/2024/SC/D3SC04185A}
%   }
% REASON: title, DOI, volume, issue and pages are correct; the YEAR is
% not. PubMed PMID 38425520 and Crossref both date Chem. Sci. 15(9),
% 3130-3139 to 2024, and the entry's OWN url field (".../2024/SC/...")
% already said 2024. The Nat.-Commun. duplicate (2a) carried the PB-test
% note; it is carried forward here.
@article{posebusters2023,
  author       = {Buttenschoen, Martin and Morris, Garrett M. and Deane, Charlotte M.},
  title        = {{PoseBusters}: {AI}-based docking methods fail to generate physically valid poses or generalise to novel sequences},
  journal      = {Chemical Science},
  volume       = {15},
  number       = {9},
  pages        = {3130--3139},
  year         = {2024},
  publisher    = {Royal Society of Chemistry},
  doi          = {10.1039/D3SC04185A},
  url          = {https://pubs.rsc.org/en/content/articlelanding/2024/SC/D3SC04185A},
  note         = {Version of record, with the full PB-test suite used by \texttt{molmetal/validation/posebusters\_runner.py}.}
}

% --- 2c. rosettafold_aa2024 — wrong DOI + wrong title ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @misc{rosettafold_aa2024,
%     author       = {Krishna, Rohith and Wang, Jue and Cheng, Maxwell and et al.},
%     title        = {{RoseTTAFold-AA}: Accurate protein complex structure prediction},
%     year         = {2024},
%     howpublished = {bioRxiv preprint},
%     doi          = {10.1101/2024.07.17.603964},
%     url          = {https://www.biorxiv.org/content/10.1101/2024.07.17.603964v1}
%   }
% REASON: doi 10.1101/2024.07.17.603964 IS registered, but it is NOT
% RoseTTAFold: the bioRxiv API returns it as "Viral delivery of an
% RNA-guided genome editor for transgene-free plant germline editing in
% Arabidopsis" (Weiss, Kamalu, Shi et al., 2024-07-19, plant biology).
% The RoseTTAFold all-atom work is bioRxiv 10.1101/2023.10.09.561603
% (v1 2023-10-09, "Generalized Biomolecular Modeling and Design with
% RoseTTAFold All-Atom"), published as Science 10.1126/science.adl2528,
% 384(6693), 2024. The bioRxiv title "RoseTTAFold-AA: Accurate protein
% complex structure prediction" matches NO primary record found, so the
% title is replaced with the attested one.
@misc{rosettafold_aa2024,
  author       = {Krishna, Rohith and Wang, Jue and Ahern, Woody and Sturmfels, Pascal and Venkatesh, Preetham and Kalvet, Indrek and Lee, Gyu Rie and Morey-Burrows, Felix S. and Anishchenko, Ivan and Humphreys, Ian R. and McHugh, Ryan and Vafeados, Dionne and Li, Xinting and Sutherland, George A. and Hitchcock, Andrew and Hunter, C. Neil and Kang, Alex and Brackenbrough, Evans and Bera, Asim K. and Baek, Minkyung and DiMaio, Frank and Baker, David},
  title        = {Generalized Biomolecular Modeling and Design with {RoseTTAFold} All-Atom},
  year         = {2024},
  howpublished = {bioRxiv preprint (2023-10-09); published in Science 384(6693), 2024},
  doi          = {10.1101/2023.10.09.561603},
  eprint       = {561603},
  archivePrefix = {bioRxiv},
  url          = {https://www.biorxiv.org/content/10.1101/2023.10.09.561603v1},
  note         = {Version of record: Science 2024, 384(6693), doi 10.1126/science.adl2528. Wf-A175 correction: the previous doi pointed at an unrelated Arabidopsis genome-editing preprint.}
}

% --- 2d. footnote:rosettafoldaa — wrong DOI embedded in howpublished/url ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @misc{footnote:rosettafoldaa,
%     author       = {Krishna, Rohith and Wang, Jue and Cheng, Maxwell and et al.},
%     title        = {{RoseTTAFold-AA}: Accurate protein complex structure prediction},
%     year         = {2024},
%     howpublished = {bioRxiv 2024.07.17.603964},
%     url          = {https://www.biorxiv.org/content/10.1101/2024.07.17.603964v1}
%   }
% REASON: this entry carries NO doi field; the wrong DOI survives only
% inside howpublished/url, and it points at the same Arabidopsis
% genome-editing preprint (see 2c). Corrected to the attested record.
@misc{footnote:rosettafoldaa,
  author       = {Krishna, Rohith and Wang, Jue and Ahern, Woody and Sturmfels, Pascal and Venkatesh, Preetham and Kalvet, Indrek and Lee, Gyu Rie and Morey-Burrows, Felix S. and Anishchenko, Ivan and Humphreys, Ian R. and McHugh, Ryan and Vafeados, Dionne and Li, Xinting and Sutherland, George A. and Hitchcock, Andrew and Hunter, C. Neil and Kang, Alex and Brackenbrough, Evans and Bera, Asim K. and Baek, Minkyung and DiMaio, Frank and Baker, David},
  title        = {Generalized Biomolecular Modeling and Design with {RoseTTAFold} All-Atom},
  year         = {2024},
  howpublished = {bioRxiv preprint (2023-10-09); published in Science 384(6693), 2024},
  doi          = {10.1101/2023.10.09.561603},
  url          = {https://www.biorxiv.org/content/10.1101/2023.10.09.561603v1},
  note         = {Wf-A175 correction: the previous howpublished/url pointed at 10.1101/2024.07.17.603964, an unrelated Arabidopsis genome-editing preprint.}
}

% --- 2e. footnote:metallodrug_review — fabricated DOI + fabricated title ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @misc{footnote:metallodrug_review,
%     author       = {Breslow, Ronald and Dong, Shao Dong},
%     title        = {Biomimetic reactions catalyzed by metallo-enzymes and their models},
%     journal      = {Chemical Reviews},
%     year         = {1998},
%     doi          = {10.1021/cr9600577},
%     url          = {https://pubs.acs.org/doi/10.1021/cr9600577}
%   }
% REASON: doi 10.1021/cr9600577 is NOT REGISTERED (handle API, doi.org 404,
% Crossref 404) and the title "Biomimetic reactions catalyzed by
% metallo-enzymes and their models" is attested by NEITHER PubMed NOR
% Crossref. The real paper by the same two authors is PMID 11848956,
% "Biomimetic Reactions Catalyzed by Cyclodextrins and Their Derivatives",
% Chem. Rev. 1998, 98(5), 1997-2012, doi 10.1021/cr970011j. The second
% author's name was also wrong ("Dong, Shao Dong" -> "Dong, Steven D.").
@misc{footnote:metallodrug_review,
  author       = {Breslow, Ronald and Dong, Steven D.},
  title        = {Biomimetic Reactions Catalyzed by Cyclodextrins and Their Derivatives},
  journal      = {Chemical Reviews},
  volume       = {98},
  number       = {5},
  pages        = {1997--2012},
  year         = {1998},
  publisher    = {American Chemical Society},
  doi          = {10.1021/cr970011j},
  url          = {https://pubs.acs.org/doi/10.1021/cr970011j},
  note         = {Wf-A175 correction: the previous title and doi 10.1021/cr9600577 were unattested; the previous doi is not a registered DOI.}
}

% --- 2f. fontana2006 — DOI points at an unrelated paper ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @article{fontana2006,
%     author       = {Fontana, Walter and Buss, Leo W.},
%     title        = {The barrier of objects: From dynamical systems to bounded organizations},
%     journal      = {Origins of Life and Evolution of the Biosphere},
%     volume       = {26},
%     number       = {3},
%     pages        = {233--253},
%     year         = {1996},
%     publisher    = {Springer},
%     doi          = {10.1007/BF02473273},
%     note         = {Constructive-chemistry program cited in \S2 (lambda\_coupling footnote)}
%   }
% REASON: doi 10.1007/BF02473273 resolves to "Dense granular bodies: a
% novel nucleophilic structure in hibernating dormice", Histochemistry and
% Cell Biology 106(6):581-586 (1996) - an unrelated cell-biology paper.
% The correct DOI for "The barrier of objects" could NOT be established
% from a primary record (Crossref title/author/ISSN queries miss; a
% 10.1007/BF024732xx handle probe over the neighbouring IDs returns no
% matching article). The WRONG doi is therefore REMOVED rather than
% replaced by an unverified one; the entry keeps its unverified
% title/journal/volume/pages unchanged and carries an explicit flag.
@article{fontana2006,
  author       = {Fontana, Walter and Buss, Leo W.},
  title        = {The barrier of objects: From dynamical systems to bounded organizations},
  journal      = {Origins of Life and Evolution of the Biosphere},
  volume       = {26},
  number       = {3},
  pages        = {233--253},
  year         = {1996},
  publisher    = {Springer},
  note         = {Constructive-chemistry program cited in \S2 (lambda\_coupling footnote). Wf-A175 2026-11-01: doi removed - 10.1007/BF02473273 resolves to an unrelated 1996 cell-biology paper; the correct DOI is NOT YET VERIFIED against a primary record.}
}

% --- 2g. footnote:fontana2006 — same wrong DOI ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @misc{footnote:fontana2006,
%     author       = {Fontana, Walter and Buss, Leo W.},
%     title        = {The barrier of objects: From dynamical systems to bounded organizations},
%     journal      = {Origins of Life and Evolution of the Biosphere},
%     year         = {1996},
%     doi          = {10.1007/BF02473273},
%     url          = {https://link.springer.com/article/10.1007/BF02473273}
%   }
% REASON: as 2f.
@misc{footnote:fontana2006,
  author       = {Fontana, Walter and Buss, Leo W.},
  title        = {The barrier of objects: From dynamical systems to bounded organizations},
  journal      = {Origins of Life and Evolution of the Biosphere},
  volume       = {26},
  number       = {3},
  pages        = {233--253},
  year         = {1996},
  publisher    = {Springer},
  note         = {Wf-A175 2026-11-01: doi removed - 10.1007/BF02473273 resolves to an unrelated 1996 cell-biology paper; the correct DOI is NOT YET VERIFIED against a primary record.}
}

% --- 2h. footnote:fep_metallo — DOI points at an unrelated paper ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @misc{footnote:fep_metallo,
%     author       = {Rizzuti, Bruno and Sportelli, Luigi and Guzzi, Rita},
%     title        = {Free-energy studies of metallo-drugs},
%     journal      = {Coordination Chemistry Reviews},
%     year         = {2009},
%     doi          = {10.1016/j.ccr.2009.03.006},
%     url          = {https://www.sciencedirect.com/science/article/pii/S0010854509000715}
%   }
% REASON: doi 10.1016/j.ccr.2009.03.006 resolves to "Metal ion extraction
% in microheterogeneous systems", Coord. Chem. Rev. 253(17-18):2186-2192
% (2009) (Marc Hegrant) - not Rizzuti/Sportelli/Guzzi. The correct DOI
% could NOT be established: ScienceDirect returns HTTP 403 to non-browser
% clients and Crossref has no Coord.-Chem.-Rev. 2009 record under that
% title (author and bibliographic queries miss; the entry's PII
% S0010854509000715 could not be mapped). The WRONG doi is removed and
% the url is dropped rather than left pointing at a mismatched PII.
@misc{footnote:fep_metallo,
  author       = {Rizzuti, Bruno and Sportelli, Luigi and Guzzi, Rita},
  title        = {Free-energy studies of metallo-drugs},
  journal      = {Coordination Chemistry Reviews},
  year         = {2009},
  note         = {Wf-A175 2026-11-01: doi and url removed - 10.1016/j.ccr.2009.03.006 resolves to 'Metal ion extraction in microheterogeneous systems' (Coord. Chem. Rev. 253(17-18):2186-2192); the correct DOI is NOT YET VERIFIED against a primary record.}
}

% --- 2i. openmm — DOI points at an unrelated paper ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @misc{openmm,
%     author       = {Eastman, Peter and Swails, Jason and Chodera, John D. and McGibbon, Robert T. and Zhao, Yutong and Beauchamp, Kyle A. and Wang, Lee-Ping and Simmonett, Andrew C. and Harrigan, Matthew P. and Stern, Chaya D. and Wiewiora, Rafal P. and Brooks, Bernard R. and Pande, Vijay S.},
%     title        = {{OpenMM} 8: Molecular dynamics simulation with heterogeneous hardware},
%     journal      = {PLOS Computational Biology},
%     year         = {2023},
%     doi          = {10.1371/journal.pcbi.1011617},
%     url          = {https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011617}
%   }
% REASON: doi 10.1371/journal.pcbi.1011617 is "Geometric and topological
% characterization of the cytoarchitecture of islets of Langerhans",
% PLoS Comput Biol 19(11):e1011617 (2023) - confirmed on PLOS's own
% article page, in Crossref and in PubMed (PMID 37943957). The title
% "OpenMM 8: Molecular dynamics simulation with heterogeneous hardware"
% and the 13-author list are attested by NO PubMed or Crossref record.
% The only 'OpenMM 8' version of record that IS attested is
% "OpenMM 8: Molecular Dynamics Simulation with Machine Learning
% Potentials", J. Phys. Chem. B 2023, 128(1):109-116 (PMID 38154096,
% doi 10.1021/acs.jpcb.3c06662), with a different (25-author) list. The
% STAGED entry below adopts the ATTESTED record; if the project actually
% needs the PLOS paper, its correct DOI must be established first.
@article{openmm,
  author       = {Eastman, Peter and Galvelis, Raimondas and Pel{\'a}ez, Ra{\'u}l P. and Abreu, Charlles R. A. and Farr, Stephen E. and Gallicchio, Emilio and Gorenko, Anton and Henry, Michael M. and Hu, Frank and Huang, Jing and Kr{\"a}mer, Andreas and Michel, Julien and Mitchell, Joshua A. and Pande, Vijay S. and Rodrigues, Jo{\~a}o PGLM and Rodriguez-Guerra, Jaime and Simmonett, Andrew C. and Singh, Sukrit and Swails, Jason and Turner, Philip and Wang, Yuanqing and Zhang, Ivy and Chodera, John D. and De Fabritiis, Gianni and Markland, Thomas E.},
  title        = {{OpenMM} 8: Molecular Dynamics Simulation with Machine Learning Potentials},
  journal      = {The Journal of Physical Chemistry B},
  volume       = {128},
  number       = {1},
  pages        = {109--116},
  year         = {2024},
  publisher    = {American Chemical Society},
  doi          = {10.1021/acs.jpcb.3c06662},
  url          = {https://pubs.acs.org/doi/10.1021/acs.jpcb.3c06662},
  note         = {Wf-A175 2026-11-01: the previous doi 10.1371/journal.pcbi.1011617 and the previous title/author list are unattested; this entry is the only 'OpenMM 8' version of record verified against a primary record (PubMed PMID 38154096, Crossref). Online-first 2023; issue year 2024.}
}

% --- 2j. attentivefp2020 — preprint title on a version-of-record DOI ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @article{attentivefp2020,
%     author       = {Xiong, Zhaoping and Wang, Dingyan and Liu, Xiaohong and Zhong, Feisheng and Wan, Xiaozhe and Li, Xutong and Li, Zhenjun and Luo, Xiang and Chen, Kaixian and Jiang, Hualiang and Zheng, Mingyue},
%     title        = {{AttentiveFP}: Representation Learning for Predicting Molecular Properties from Molecular Graphs},
%     journal      = {Journal of Medicinal Chemistry},
%     volume       = {63},
%     number       = {16},
%     pages        = {8749--8760},
%     year         = {2020},
%     publisher    = {American Chemical Society},
%     doi          = {10.1021/acs.jmedchem.9b00959},
%     url          = {https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959}
%   }
% REASON: the DOI is correct, but the TITLE is the arXiv preprint title.
% The version of record is "Pushing the Boundaries of Molecular
% Representation for Drug Discovery with the Graph Attention Mechanism",
% J. Med. Chem. 2020, 63(12):8749-8760 (PubMed PMID 31408336, Crossref).
% Two author names were also wrong: "Li, Zhenjun" -> "Li, Zhaojun" and
% "Luo, Xiang" -> "Luo, Xiaomin" (Crossref author list).
@article{attentivefp2020,
  author       = {Xiong, Zhaoping and Wang, Dingyan and Liu, Xiaohong and Zhong, Feisheng and Wan, Xiaozhe and Li, Xutong and Li, Zhaojun and Luo, Xiaomin and Chen, Kaixian and Jiang, Hualiang and Zheng, Mingyue},
  title        = {Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism},
  journal      = {Journal of Medicinal Chemistry},
  volume       = {63},
  number       = {12},
  pages        = {8749--8760},
  year         = {2020},
  publisher    = {American Chemical Society},
  doi          = {10.1021/acs.jmedchem.9b00959},
  url          = {https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959},
  note         = {Also circulated as arXiv:2002.07177 under the title 'AttentiveFP: Representation Learning for Predicting Molecular Properties from Molecular Graphs'. Wf-A175 correction: the title now matches the version of record, and the number field is corrected from 16 to 12 (Crossref issue 12; PubMed 63(12):8749-8760).}
}

% --- 2k. footnote:attentivefp — same title correction ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @misc{footnote:attentivefp,
%     author       = {Xiong, Zhaoping and Wang, Dingyan and Liu, Xiaohong and Zhong, Feisheng and Wan, Xiaozhe and Li, Xutong and Li, Zhenjun and Luo, Xiang and Chen, Kaixian and Jiang, Hualiang and Zheng, Mingyue},
%     title        = {{AttentiveFP}: Representation Learning for Predicting Molecular Properties from Molecular Graphs},
%     journal      = {Journal of Medicinal Chemistry},
%     year         = {2020},
%     doi          = {10.1021/acs.jmedchem.9b00959},
%     url          = {https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959}
%   }
% REASON: as 2j.
@misc{footnote:attentivefp,
  author       = {Xiong, Zhaoping and Wang, Dingyan and Liu, Xiaohong and Zhong, Feisheng and Wan, Xiaozhe and Li, Xutong and Li, Zhaojun and Luo, Xiaomin and Chen, Kaixian and Jiang, Hualiang and Zheng, Mingyue},
  title        = {Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism},
  journal      = {Journal of Medicinal Chemistry},
  volume       = {63},
  number       = {12},
  pages        = {8749--8760},
  year         = {2020},
  publisher    = {American Chemical Society},
  doi          = {10.1021/acs.jmedchem.9b00959},
  url          = {https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959},
  note         = {Wf-A175 correction: preprint title replaced by the version-of-record title; see attentivefp2020.}
}


"""

SCIPAPER = r"""% =====================================================================
%  STAGED BIBTEX — NOT APPLIED TO .tools/sci-paper/refs/references.bib
%  Wf-A175 / BIBLIOGRAPHY branch, 2026-11-01
%
%  *** THIS FILE IS A PROPOSAL, NOT A PATCH. ***
%  Nothing here has been written to .tools/sci-paper/refs/references.bib.
%  The two entries below are FABRICATED (their DOIs are not registered and
%  no primary record carries their titles, venues or authors). They are
%  replaced here with records verified on 2026-11-01 against:
%    - PubMed/NLM efetch  (PMID 41969144 for krasnov2026metalcytotoxdb)
%    - the Crossref work record for 10.1021/acs.jmedchem.5c02755
%    - the DOI handle registry for 10.26434/chemrxiv-2025-pp32k
%      (https://doi.org/api/handles/10.26434/chemrxiv-2025-pp32k -> responseCode 1)
%
%  Per AGENTS.md 5.5 rule 6 the superseded rendering of each entry is
%  preserved VERBATIM below, dated, and never silently deleted.
% =====================================================================

% --- .tools/sci-paper/refs/references.bib krasnov2026metalcytotoxdb ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @article{krasnov2026metalcytotoxdb,
%     author    = {Krasnov, Peter and Selivanova, Maria and Akhmadeev, Bulat},
%     title     = {MetalCytoToxDB: A Curated Database of Metal-Complex Cytotoxicity
%     Across 60 Cancer Cell Lines},
%     journal   = {Journal of Chemical Information and Modeling},
%     year      = {2026},
%     volume    = {66},
%     number    = {3},
%     pages     = {412--437},
%     doi       = {10.1021/acs.jcim.5c01942},
%     publisher = {American Chemical Society},
%   }
% REASON: doi 10.1021/acs.jcim.5c01942 is NOT REGISTERED (handle API
% responseCode != 1; Crossref HTTP 404); the title, the journal, the
% volume/issue/pages and ALL THREE author names are attested by no
% primary record. The real record is the J. Med. Chem. article
% (PMID 41969144), also staged in refs_corrected_STAGED.bib.
@article{krasnov2026metalcytotoxdb,
  author    = {Krasnov, Lev and Malikov, Dmitry and Kiseleva, Marina A. and Nykhrikova, Ekaterina V. and Tatarin, Sergei V. and Bezzubov, Stanislav I.},
  title     = {Machine Learning Approach to Anticancer Activity Prediction of Transition-Metal Complexes Based on a Large-Scale Experimental Database},
  journal   = {Journal of Medicinal Chemistry},
  year      = {2026},
  volume    = {69},
  number    = {8},
  pages     = {8838--8851},
  doi       = {10.1021/acs.jmedchem.5c02755},
  publisher = {American Chemical Society},
}

% --- .tools/sci-paper/refs/references.bib rusanov2025platinai ---
% SUPERSEDED 2026-11-01 (Wf-A175). VERBATIM original:
%   @article{rusanov2025platinai,
%     author    = {Rusanov, Aleksandr and {MB-Finder Collaboration}},
%     title     = {PlatinAI: A 5K-Compound Platinum Cytotoxicity Dataset for
%     Machine Learning Benchmarks},
%     journal   = {Scientific Data},
%     year      = {2025},
%     volume    = {12},
%     number    = {184},
%     pages     = {1--14},
%     doi       = {10.1038/s41597-025-05012-9},
%     publisher = {Nature Publishing Group},
%   }
% REASON: doi 10.1038/s41597-025-05012-9 is NOT REGISTERED (handle API
% responseCode != 1; Crossref HTTP 404); title, journal, volume/issue/
% pages and the "MB-Finder Collaboration" authorship are unattested -
% "MB-Finder" does not occur anywhere in the real record's abstract, and
% PlatinAI is one designed bis-carbene platinum(II) complex, not a
% dataset. The real record is the ChemRxiv preprint, also staged in
% refs_corrected_STAGED.bib.
@article{rusanov2025platinai,
  author    = {Rusanov, Daniil A. and Brezgunov, Dmitrii Yu. and Matnurov, Egor M. and Pashaliev, Bekir L. and Vetrova, Varvara V. and Babak, Maria V.},
  title     = {Overcoming cisplatin resistance via deep learning-assisted de novo design of platinum complexes},
  year      = {2025},
  publisher = {American Chemical Society},
  doi       = {10.26434/chemrxiv-2025-pp32k},
}
"""

REPORT = r"""# Wf-A175 / BIBLIOGRAPHY — `paper/refs.bib` DOI integrity audit

**Date:** 2026-11-01 · **Branch:** BIBLIOGRAPHY · **Mode:** CPU-only, read-only on `paper/**`

Every number below is **MEASURED** and cites a path on disk or a URL actually fetched during
this audit. No PROJECTED value appears. Where a correct value could not be established from a
primary record, this report says so rather than guessing.

Artefacts in this directory:

| File | What |
|---|---|
| `audit.json` | machine-readable: 40 DOI-bearing `refs.bib` entries + a `required_keys` section |
| `report.md` | this file |
| `refs_corrected_STAGED.bib` | **STAGED** corrected BibTeX for `paper/refs.bib` — **NOT applied** |
| `refs_sci_paper_corrected_STAGED.bib` | **STAGED** corrected BibTeX for `.tools/sci-paper/refs/references.bib` — **NOT applied** |
| `audit_refs_bib.py`, `resolve_targets{2..9}.py`, `build_*.py` | the scripts that produced them |
| `_audit_raw.json`, `_targets{2..9}_raw.json` | raw resolver payloads (evidence trail) |

---

## 1. Census (MEASURED)

`paper/refs.bib` — 46,684 B, 1,071 lines, **100 `@`-entries** (MEASURED, `ls -la` +
`grep -c '^@'`), of which **40 carry a `doi = {...}` field** (MEASURED, `audit.json`
→ `summary.doi_bearing_entries`).

Resolver layers used (all fetched live on 2026-11-01):

1. `https://doi.org/api/handles/<doi>` — the DOI **handle registry** (is the DOI registered at all?)
2. `https://doi.org/<doi>` — the resolver **landing page** (redirect-followed)
3. `https://api.crossref.org/works/<doi>` — the registration agency's publisher-deposited record
4. `https://api.biorxiv.org/details/biorxiv/<doi>` — bioRxiv primary record
5. `http://export.arxiv.org/api/query?id_list=<id>` — arXiv primary record
6. `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch,efetch}` — PubMed/NLM primary index
7. publisher landing pages: `journals.plos.org`, `ojs.aaai.org`

| Metric | Value | Source |
|---|---|---|
| `@`-entries in `paper/refs.bib` | **100** | `grep -c '^@' paper/refs.bib` |
| DOI-bearing entries | **40** | `audit.json` → `summary` |
| DOIs **registered** in the DOI handle registry | **38 / 40** | `audit.json` → `summary.doi_handle_registered` |
| DOIs that **resolve** | **38 / 40** | `audit.json` → `summary.doi_resolved` |
| Titles matching the resolver's title | **31 / 40** | `audit.json` → `summary.title_match` |
| Titles **not** matching | **9 / 40** | `audit.json` → `summary.title_mismatch` |
| Entries with a defect class | **10 / 40** | `audit.json` → `summary.defect_class_counts` |

Defect classes (MEASURED): `NONE` 30 · `WRONG_DOI` 5 · `TITLE_MISMATCH_SAME_WORK` 2 ·
`WRONG_YEAR` 1 · `FABRICATED_DOI` 1 · `FABRICATED_DOI+FABRICATED_TITLE` 1.

**Only 2 of the 40 DOIs are unregistered** — and both are hard 404s at the resolver *and* at
Crossref: `10.1038/s41467-024-49069-0` and `10.1021/cr9600577`.

### 1.1 Resolver-layer coverage (MEASURED)

`http_status` in `audit.json` is the status of the **doi.org landing page** after redirects.
MEASURED distribution over the 40 rows: **200 → 24**, **403 → 14**, **404 → 2**.

* The **2 × 404** are the two unregistered DOIs; the landing body is literally
  "Error: DOI Not Found".
* The **14 × 403** are publisher bot-blocks (Cloudflare "Just a moment…") on ACS / Wiley / RSC /
  Elsevier landing pages. **A 403 is not a missing paper** — for every one of those 14 rows the
  title was taken from a different primary layer, and the layer used is recorded per row in
  `audit.json` → `resolver_source` (e.g. `Crossref + PubMed (PMID 31408336)`).
  An audit that treated 403 as "unresolved" would have produced 14 false positives.
* `resolver_source` distribution over all 40 rows: `doi.org landing page` 9, `Crossref` 10,
  `arXiv API` 6, `bioRxiv API` 1, `doi.org landing (Springer/BMC/Elsevier)` 4,
  `Crossref + PubMed (± PMID)` 5, `PLOS landing page + Crossref + PubMed` 1, and 4 rows where the
  handle registry, the landing page, Crossref and PubMed all agreed.

---

## 2. The 10 defective entries

Every row below was re-checked by hand against a primary record. `paper/refs.bib` line numbers
are from `grep -n '^@' paper/refs.bib`.

| # | cite key | line | DOI in file | What the DOI actually is | Defect |
|---|---|---|---|---|---|
| 1 | `rosettafold_aa2024` | 77 | `10.1101/2024.07.17.603964` | "Viral delivery of an RNA-guided genome editor for transgene-free plant germline editing in *Arabidopsis*" (Weiss/Kamalu/Shi et al., bioRxiv 2024-07-19) — **bioRxiv API + Crossref** | `WRONG_DOI` |
| 2 | `buttenschoen2024posebusters` | 239 | `10.1038/s41467-024-49069-0` | **nothing** — DOI not registered (handle API `responseCode != 1`; doi.org HTTP **404** "Error: DOI Not Found"; Crossref HTTP **404**) | `FABRICATED_DOI` |
| 3 | `footnote:metallodrug_review` | 537 | `10.1021/cr9600577` | **nothing** — DOI not registered (handle API; doi.org **404**; Crossref **404**); the *title* is attested by neither PubMed nor Crossref | `FABRICATED_DOI+FABRICATED_TITLE` |
| 4 | `fontana2006` | 301 | `10.1007/BF02473273` | "Dense granular bodies: a novel nucleophilic structure in hibernating dormice", *Histochem. Cell Biol.* **106**(6):581-586 (1996) — Springer landing page + Crossref | `WRONG_DOI` |
| 5 | `footnote:fontana2006` | 503 | `10.1007/BF02473273` | same as #4 | `WRONG_DOI` |
| 6 | `footnote:fep_metallo` | 528 | `10.1016/j.ccr.2009.03.006` | "Metal ion extraction in microheterogeneous systems", *Coord. Chem. Rev.* **253**(17-18):2186-2192 (2009) (M. Hébrant) — Elsevier landing + Crossref | `WRONG_DOI` |
| 7 | `openmm` | 696 | `10.1371/journal.pcbi.1011617` | "Geometric and topological characterization of the cytoarchitecture of islets of Langerhans", *PLoS Comput. Biol.* **19**(11):e1011617 (2023) — **PLOS's own article page**, Crossref, and PubMed **PMID 37943957** | `WRONG_DOI` |
| 8 | `attentivefp2020` | 136 | `10.1021/acs.jmedchem.9b00959` | "**Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism**", *J. Med. Chem.* **63**(12):8749-8760 — PubMed **PMID 31408336** + Crossref. The DOI is right; the **title** is the arXiv preprint title | `TITLE_MISMATCH_SAME_WORK` |
| 9 | `footnote:attentivefp` | 453 | `10.1021/acs.jmedchem.9b00959` | same as #8 | `TITLE_MISMATCH_SAME_WORK` |
| 10 | `posebusters2023` | 189 | `10.1039/D3SC04185A` | title/DOI/volume/issue/pages all correct — but the **year** is wrong: *Chem. Sci.* **2024**, 15(9), 3130-3139 (PubMed **PMID 38425520**, Crossref `issued` 2024). The entry's own `url` field already said `.../2024/SC/D3SC04185A` | `WRONG_YEAR` |

### 2.1 The two hard-fabricated entries

**`buttenschoen2024posebusters`** claims a Nature Communications version of record
(15(1):6621, 2024). There is none. MEASURED:
`https://doi.org/api/handles/10.1038/s41467-024-49069-0` → `responseCode != 1`;
`https://doi.org/10.1038/s41467-024-49069-0` → HTTP **404**, body "Error: DOI Not Found";
`https://api.crossref.org/works/10.1038/s41467-024-49069-0` → HTTP **404**.
Crossref's title search for "PoseBusters" returns only `10.1039/d3sc04185a` (plus RSC peer-review
decision/reviewer records), and PubMed's `"PoseBusters"[Title]` returns **exactly 1 hit**
(PMID 38425520). The entry is therefore a **fabricated duplicate** of `posebusters2023` with a
fabricated venue and page number — the STAGED file removes it and folds its PB-test note into
`posebusters2023`.

**`footnote:metallodrug_review`** cites "Biomimetic reactions catalyzed by metallo-enzymes and
their models", *Chem. Rev.* 1998, doi `10.1021/cr9600577` — DOI unregistered, title attested by
**neither** PubMed nor Crossref, and the second author is wrong ("Dong, Shao Dong"). The real
paper by the same pair (Ronald Breslow; **Steven D. Dong**) is
**PMID 11848956**, "Biomimetic Reactions Catalyzed by **Cyclodextrins** and Their Derivatives",
*Chem. Rev.* 1998, **98**(5), 1997-2012, doi **`10.1021/cr970011j`** (MEASURED via PubMed
efetch XML + Crossref).

### 2.2 Honest negatives — correct DOI NOT established

Three entries carry a DOI that provably points at a different work. For **two** of them no
correct DOI could be established from a primary record in this audit, so the STAGED file
**removes the wrong DOI** rather than inventing a replacement; for the third the correct record
IS attested and is used:

* `fontana2006` / `footnote:fontana2006` — "The barrier of objects" (Fontana & Buss,
  *Origins of Life Evol. Biosph.* 26(3):233-253, 1996). Crossref title / author / ISSN queries
  all miss; a probe over the neighbouring `10.1007/BF024732xx` handle IDs returns no matching
  article. → **DOI removed, flagged NOT YET VERIFIED.**
* `footnote:fep_metallo` — "Free-energy studies of metallo-drugs" (Rizzuti/Sportelli/Guzzi,
  *Coord. Chem. Rev.* 2009). ScienceDirect returns **HTTP 403** to non-browser clients and
  Crossref holds no CCR-2009 record under that title; the entry's PII `S0010854509000715` could
  not be mapped to a DOI. → **DOI and url removed, flagged NOT YET VERIFIED.**
* `rosettafold_aa2024` / `footnote:rosettafoldaa` — the RoseTTAFold all-atom work **is** attested,
  as bioRxiv `10.1101/2023.10.09.561603` (v1 2023-10-09, "Generalized Biomolecular Modeling and
  Design with RoseTTAFold All-Atom") and *Science* `10.1126/science.adl2528`, 384(6693), 2024.
  The bioRxiv title "RoseTTAFold-AA: Accurate protein complex structure prediction" matches **no**
  primary record found, so the STAGED file adopts the attested title. → **DOI corrected.**

### 2.3 Field-level defects found while comparing records

* `attentivefp2020` / `footnote:attentivefp`: two author names are wrong — "Li, **Zhenjun**" →
  "Li, **Zhaojun**" and "Luo, **Xiang**" → "Luo, **Xiaomin**" (Crossref author list); and
  `number` 16 → **12** (Crossref issue 12; PubMed 63(12):8749-8760).
* `chemprop` (line 149) — style nit only, **no defect**: it is `@misc` with
  `howpublished = {Journal of Chemical Information and Modeling / GitHub release}` instead of a
  `journal` field, and omits volume/pages. Its title and DOI are correct (PubMed
  **PMID 38147829**: "Chemprop: A Machine Learning Package for Chemical Property Prediction",
  *J. Chem. Inf. Model.* **64**(1):9-17 — the entry's `year = {2023}` is the online-first year,
  the issue year is 2024). Not changed in the STAGED file.

---

## 3. The 3 gate-required keys

`gate_p3_protocol_alignment_baseline_paper_refs_bib` —
**`molmetal/scripts/pre_flight_check.py:11052`** — defaults to
`("krasnov2026metalcytotoxdb", "rusanov2025platinai", "ji2022drugood")` and does a **substring**
test `k not in text` over the whole file (the gate body, read verbatim).

MEASURED: `grep -n` for each key in `paper/refs.bib` → **0 hits for all three**. Therefore the
gate would report (the message is built at `pre_flight_check.py:11092`,
`f"refs.bib missing {len(missing)} keys: {missing}"`):

> `FAIL — refs.bib missing 3 keys: ['krasnov2026metalcytotoxdb', 'rusanov2025platinai', 'ji2022drugood']`

This is an **independent corroboration**, not a circular one: Wf-A169 already ran this gate and
recorded the identical verdict string at
`molmetal/reports/wfa169_dead_gate_repair/verdict_table.md:163`, with the same census
("100 `@`-entries, 46,684 bytes", `registration_table.md:76` and `verdict_table.md:138`) that this
audit re-measured from scratch on 2026-11-01.

**But the gate never runs.** MEASURED: `grep -c p3_protocol_alignment_baseline_paper_refs_bib
molmetal/scripts/pre_flight_check.py` → **1** (the `def` at line 11052 only). It is in no pipeline
list, so it is one of the 24 Wf-432-A1 **dead** gates — the same condition Wf-A169 recorded
(`molmetal/reports/wfa169_dead_gate_repair/registration_table.md:163`, `verdict_table.md:138`).
Adding the three keys alone would fix the *content* but would not make the gate execute.

### 3.1 The two `.tools/sci-paper/refs/references.bib` entries are fabricated too

| Key | What that file claims | MEASURED verdict |
|---|---|---|
| `krasnov2026metalcytotoxdb` | title "MetalCytoToxDB: A Curated Database of Metal-Complex Cytotoxicity Across 60 Cancer Cell Lines"; *J. Chem. Inf. Model.* 2026, 66(3), 412-437; authors "Krasnov, **Peter** and Selivanova, Maria and Akhmadeev, Bulat"; doi `10.1021/acs.jcim.5c01942` | DOI **NOT REGISTERED** (handle API `responseCode != 1`; Crossref HTTP **404**). No primary record carries that title/venue/authors. Real record: PubMed **PMID 41969144**, *J. Med. Chem.* **2026, 69(8), 8838-8851**, doi **`10.1021/acs.jmedchem.5c02755`**, authors **Lev Krasnov, Dmitry Malikov, Marina A. Kiseleva, Ekaterina V. Nykhrikova, Sergei V. Tatarin, Stanislav I. Bezzubov** |
| `rusanov2025platinai` | title "PlatinAI: A 5K-Compound Platinum Cytotoxicity Dataset for Machine Learning Benchmarks"; *Scientific Data* 2025, 12(184), 1-14; authors "Rusanov, **Aleksandr** and {MB-Finder Collaboration}"; doi `10.1038/s41597-025-05012-9` | DOI **NOT REGISTERED** (handle API; Crossref HTTP **404**). Real record: ChemRxiv preprint doi **`10.26434/chemrxiv-2025-pp32k`**, "Overcoming cisplatin resistance via deep learning-assisted de novo design of platinum complexes" (2025), authors **Daniil A. Rusanov, Dmitrii Yu. Brezgunov, Egor M. Matnurov, Bekir L. Pashaliev, Varvara V. Vetrova, Maria V. Babak** |

**"PlatinAI" is not a database name.** MEASURED from the registered abstract of
`10.26434/chemrxiv-2025-pp32k`: *"using de novo fragment assembly approach we designed and
successfully synthesized a novel **bis-carbene platinum(II) complex, PlatinAI**"* — i.e. one
designed complex, not a dataset. The string "MB-Finder" occurs **0 times** in that abstract.
The curated dataset behind the paper is *"a manually curated dataset of **4,078 platinum complexes
with 18,995 IC50 values**"*, and the paper *"predicted the likelihood of potential anticancer
activity for **214,373** previously published platinum complexes"* in MCF-7 / A549 / A2780 /
A2780cis, "synthesized and tested 20 of them".

**MetalCytoToxDB's counts**, MEASURED from the PubMed abstract of PMID 41969144:
*"we have manually curated MetalCytoToxDB—a comprehensive experimental database comprising
**26,500 IC50 values for 7050 metal complexes against 754 cell lines from 1921 articles**."*
The `sci-paper` entry's "Across 60 Cancer Cell Lines" is **not** in the abstract — 754 cell lines
is the measured number.

### 3.2 `ji2022drugood`

The task brief said this key "does NOT exist anywhere in the repo". **Partly refuted**: it does not
exist as a BibTeX entry, but it *does* exist as the gate's own default key tuple
(`molmetal/scripts/pre_flight_check.py:11060`) and as prose in
`TODO/03_plans/PRE_TRAINING_GATE_v2.md:172` and
`TODO/completed/432a1_pre_training_gate_v2.md:120`. What is missing is the entry.

Correct records, MEASURED:
* **arXiv** `2201.09637` (2022-01-24) — arXiv API title: "DrugOOD: Out-of-Distribution (OOD)
  Dataset Curator and Benchmark for AI-aided Drug Discovery -- A Focus on Affinity Prediction
  Problems with Noise Annotations".
* **AAAI-23** doi **`10.1609/aaai.v37i7.25970`** — verified on AAAI's own OJS page
  `https://ojs.aaai.org/index.php/AAAI/article/view/25970` (`citation_volume` 37,
  `citation_issue` 7, `citation_firstpage` 8023, `citation_lastpage` 8031, and the 17-author
  list Yuanfeng Ji … Yatao Bian). Title: "DrugOOD: Out-of-Distribution Dataset Curator and
  Benchmark for AI-Aided Drug Discovery – a Focus on Affinity Prediction Problems with Noise
  Annotations".

> **Correction to the task brief:** the DOI `10.1609/aaai.v37i.v37i7.25970` (doubled `v37i`) is
> **NOT registered** (doi.org handle API `responseCode != 1`). The registered DOI is
> `10.1609/aaai.v37i7.25970`.

---

## 4. Overlap with the 24 dead gates

MEASURED from `molmetal/reports/wfa169_dead_gate_repair/patch_wire_24_gates.diff` (the 24 gate
registrations) and from the `def` list in `pre_flight_check.py`:

* **1 of 24** dead gates reads a bibliography: `p3_protocol_alignment_baseline_paper_refs_bib`
  (the one at diff line 149).
* The other **23** check *metric / table / split content* for the same three papers — e.g.
  `p3_protocol_recommended_krasnov_paper_table_4_replication`,
  `p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup`,
  `p3_protocol_recommended_drugood_3_splits_full_alignment`,
  `krasnov_m1_per_seed_per_split_5fold_5metal`, `krasnov_m21_per_cell_line_per_seed`,
  `krasnov_protocol_alignment_diffs_documented`, `rusanov_m2_per_cell_line_per_cohort`,
  `rusanov_m2_4cell_baseline_comparison`, `rusanov_protocol_alignment_diffs_documented`,
  `drugood_m17_{assay,scaffold,size}_split_auc`, and the 11 `p3_protocol_alignment_*` gates.
  None of them reads a cite key.

**Consequence (MEASURED):** adding the 3 cite keys to `paper/refs.bib` closes **1 of 24** dead
gates, and even that one only *if* the gate is also registered in a pipeline — which it is not.
The other 23 stay dead on data grounds.

---

## 5. What the STAGED file does and does not do

`refs_corrected_STAGED.bib` is a **proposal**. It was **not** applied to `paper/refs.bib`
(AGENTS.md §5.3 / TODO-84 forbids writes under `paper/**` without the user's explicit OK).
It contains:

1. The **3 gate-required keys** with fields verified against primary records.
2. **11 corrected entries** of `paper/refs.bib`: `posebusters2023` (year), `rosettafold_aa2024`,
   `footnote:rosettafoldaa`, `footnote:metallodrug_review`, `fontana2006`,
   `footnote:fontana2006`, `footnote:fep_metallo`, `openmm`, `attentivefp2020`,
   `footnote:attentivefp`, plus the **removal** of `buttenschoen2024posebusters`.
3. For every changed or removed entry, the **superseded rendering is quoted VERBATIM** and dated,
   per AGENTS.md §5.5 rule 6 — nothing is silently deleted or silently edited.

Three of the corrections carry an explicit "NOT YET VERIFIED against a primary record" flag
(`fontana2006`, `footnote:fontana2006`, `footnote:fep_metallo`) because the correct DOI could not
be established; their wrong DOIs were removed rather than replaced.

The two FABRICATED entries in `.tools/sci-paper/refs/references.bib`
(`krasnov2026metalcytotoxdb`, `rusanov2025platinai`) are staged in a **separate** file,
`refs_sci_paper_corrected_STAGED.bib`, so that no cite key is defined twice in one `.bib`
(BibTeX's duplicate-key behaviour is undefined/last-wins and would silently pick one).

**Applying either file requires the user's AGENTS.md §5.3 OK.** Until then the gate stays FAIL
(and, unregistered, never runs).
"""

atomic_write_file(str(RD / "refs_corrected_STAGED.bib"), STAGED)
atomic_write_file(str(RD / "refs_sci_paper_corrected_STAGED.bib"), SCIPAPER)
atomic_write_file(str(RD / "report.md"), REPORT)
print("wrote refs_corrected_STAGED.bib, refs_sci_paper_corrected_STAGED.bib, report.md")
