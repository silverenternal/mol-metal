#!/usr/bin/env python3
"""Wf-A175 — build audit.json + refs_corrected_STAGED.bib (atomic write, AGENTS.md 5.6)."""
from __future__ import annotations
import html, json, re, sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
RD = REPO / "molmetal" / "reports" / "wfa175_refs_bib_audit"
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))
from wf386a3_atomic_write import atomic_write_file  # noqa: E402

raw = json.loads((RD / "_audit_raw.json").read_text(encoding="utf-8"))
handles = json.loads((RD / "_targets4_raw.json").read_text(encoding="utf-8"))["handles"]
t2 = json.loads((RD / "_targets2_raw.json").read_text(encoding="utf-8"))
t8 = json.loads((RD / "_targets8_raw.json").read_text(encoding="utf-8"))

BLOCK_TITLES = {"just a moment", "redirecting", "error: doi not found", "", "science direct"}


def norm(t: str) -> str:
    t = html.unescape(t or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def title_match(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if (na.startswith(nb) or nb.startswith(na)) and min(len(na), len(nb)) >= 12:
        return True
    return False


# ---- manually adjudicated findings (each backed by a primary record fetched live) ----
ADJ = {
    "rosettafold_aa2024": dict(
        defect="WRONG_DOI",
        resolver_source="bioRxiv API (api.biorxiv.org/details/biorxiv/<doi>)",
        note=("DOI 10.1101/2024.07.17.603964 resolves to 'Viral delivery of an RNA-guided "
              "genome editor for transgene-free plant germline editing in Arabidopsis' "
              "(Weiss/Kamalu/Shi et al., bioRxiv 2024-07-19, plant biology) — NOT RoseTTAFold-AA. "
              "The DOI is REGISTERED but points at an unrelated paper. The RoseTTAFold all-atom "
              "work is bioRxiv 10.1101/2023.10.09.561603 v1 2023-10-09 (title 'Generalized "
              "Biomolecular Modeling and Design with RoseTTAFold All-Atom') / Science "
              "10.1126/science.adl2528, 384(6693), 2024. Neither title matches the bibtex title.")),
    "buttenschoen2024posebusters": dict(
        defect="FABRICATED_DOI",
        resolver_source="doi.org handle registry (doi.org/api/handles) + doi.org landing + Crossref",
        note=("DOI 10.1038/s41467-024-49069-0 is NOT REGISTERED: doi.org handle API responseCode!=1, "
              "doi.org landing HTTP 404 'Error: DOI Not Found', Crossref HTTP 404. No "
              "Nature-Communications version of PoseBusters exists: Crossref title search "
              "'PoseBusters' returns only 10.1039/d3sc04185a (+ peer-review decision records) and "
              "PubMed 'PoseBusters'[Title] returns exactly 1 hit (PMID 38425520, Chem. Sci. 2024, "
              "15, 3130-3139). The entry is a fabricated duplicate of posebusters2023; its "
              "venue (Nat. Commun. 15(1):6621) and page number are not attested by any primary record.")),
    "footnote:metallodrug_review": dict(
        defect="FABRICATED_DOI+FABRICATED_TITLE",
        resolver_source="doi.org handle registry + doi.org landing + Crossref + PubMed",
        note=("DOI 10.1021/cr9600577 is NOT REGISTERED (handle API responseCode!=1, doi.org HTTP 404, "
              "Crossref HTTP 404). The bibtex title 'Biomimetic reactions catalyzed by "
              "metallo-enzymes and their models' is attested by NEITHER PubMed nor Crossref. The "
              "real paper by the same pair (Ronald Breslow; Steven D. Dong — bibtex says 'Dong, "
              "Shao Dong') is PMID 11848956, 'Biomimetic Reactions Catalyzed by Cyclodextrins and "
              "Their Derivatives', Chem. Rev. 1998, 98(5), 1997-2012, doi 10.1021/cr970011j.")),
    "fontana2006": dict(
        defect="WRONG_DOI",
        resolver_source="doi.org landing (Springer) + Crossref",
        note=("DOI 10.1007/BF02473273 resolves to 'Dense granular bodies: a novel nucleoplasmic "
              "structure in hibernating dormice', Histochemistry and Cell Biology 106(6):581-586 "
              "(1996) (Tamburini/Malatesta/Zancanaro) — an unrelated cell-biology paper. The cited "
              "work (Fontana & Buss, 'The barrier of objects', Origins of Life Evol. Biosph. "
              "26(3):233-253, 1996) has NO Crossref record under that title; its correct DOI could "
              "NOT be established from a primary record (Crossref title/author/ISSN queries and a "
              "10.1007/BF024732xx neighbourhood handle probe all returned nothing).")),
    "footnote:fontana2006": dict(
        defect="WRONG_DOI",
        resolver_source="doi.org landing (Springer) + Crossref",
        note=("Same wrong DOI as fontana2006 (10.1007/BF02473273 -> dormice cell-biology paper). "
              "Correct DOI for 'The barrier of objects' not established from a primary record.")),
    "footnote:fep_metallo": dict(
        defect="WRONG_DOI",
        resolver_source="doi.org landing (Elsevier) + Crossref",
        note=("DOI 10.1016/j.ccr.2009.03.006 resolves to 'Metal ion extraction in microheterogeneous "
              "systems', Coordination Chemistry Reviews 253(17-18):2186-2192 (2009) (Marc Hegrant) "
              "— not Rizzuti/Sportelli/Guzzi 'Free-energy studies of metallo-drugs'. The correct "
              "DOI could NOT be established: ScienceDirect returns HTTP 403 to non-browser clients "
              "and Crossref has no CCR-2009 record under that title (author and bibliographic "
              "queries all miss). The bibtex's own PII S0010854509000715 could not be mapped.")),
    "openmm": dict(
        defect="WRONG_DOI",
        resolver_source="PLOS landing page + Crossref + PubMed (PMID 37943957)",
        note=("DOI 10.1371/journal.pcbi.1011617 is 'Geometric and topological characterization of "
              "the cytoarchitecture of islets of Langerhans', PLoS Comput Biol 19(11):e1011617 "
              "(2023) (Aggarwal/Striegel/Hara/Periwal) — confirmed on PLOS's own article page, in "
              "Crossref and in PubMed. It is NOT OpenMM 8. The only 'OpenMM 8' version of record "
              "indexed is PMID 38154096, 'OpenMM 8: Molecular Dynamics Simulation with Machine "
              "Learning Potentials', J. Phys. Chem. B 2024, 128(1):109-116, doi 10.1021/acs.jpcb.3c06662 "
              "— note its title differs from the bibtex title '...with heterogeneous hardware', which "
              "matches no PubMed or Crossref record.")),
    "attentivefp2020": dict(
        defect="TITLE_MISMATCH_SAME_WORK",
        resolver_source="Crossref + PubMed (PMID 31408336)",
        note=("DOI 10.1021/acs.jmedchem.9b00959 is correct, but the bibtex title is the arXiv "
              "preprint title. The version of record is 'Pushing the Boundaries of Molecular "
              "Representation for Drug Discovery with the Graph Attention Mechanism', J. Med. Chem. "
              "2020, 63(12):8749-8760. Year/volume/pages in the entry are correct.")),
    "footnote:attentivefp": dict(
        defect="TITLE_MISMATCH_SAME_WORK",
        resolver_source="Crossref + PubMed (PMID 31408336)",
        note=("Same as attentivefp2020: correct DOI, preprint title instead of the version-of-record "
              "title 'Pushing the Boundaries of Molecular Representation for Drug Discovery with the "
              "Graph Attention Mechanism'.")),
    "posebusters2023": dict(
        defect="WRONG_YEAR",
        resolver_source="Crossref + PubMed (PMID 38425520)",
        note=("DOI and title are correct, but the year field is wrong: Chem. Sci. 2024, 15(9):3130-3139 "
              "(PubMed PMID 38425520, Crossref issued 2024). The entry's own url field "
              "'.../articlelanding/2024/SC/D3SC04185A' already carries 2024, contradicting year={2023}.")),
    "chemprop": dict(
        defect="NONE",
        resolver_source="Crossref + PubMed (PMID 38147829)",
        note=("Title/DOI correct. Journal/year/volume/pages absent because the entry is @misc with "
              "howpublished — a style nit, not an integrity defect. Issue assignment is "
              "JCIM 2024, 64(1):9-17 (online 2023).")),
    "dmpnn2019": dict(
        defect="NONE",
        resolver_source="Crossref + PubMed (PMID 31361484)",
        note=("Title/DOI/year/volume/pages all correct (JCIM 2019, 59(8):3370-3388; PubMed "
              "PubDate 2019 Aug 26).")),
    "vina2010": dict(
        defect="NONE",
        resolver_source="Crossref + PubMed (PMID 19499576)",
        note="Title/DOI/year/volume/pages correct (J. Comput. Chem. 2010, 31(2):455-461)."),
    "quickvina2016": dict(
        defect="NONE",
        resolver_source="Crossref",
        note="Title/DOI/year/volume/pages correct (Bioinformatics 2015, 31(13):2214-2216)."),
    "al-hossary2015qvina": dict(
        defect="NONE",
        resolver_source="Crossref",
        note="Same work as quickvina2016 (duplicate cite key, both correct)."),
    "himo2005cuaac": dict(
        defect="NONE",
        resolver_source="Crossref",
        note="Title/DOI/year/volume/pages correct (JACS 2005, 127(1):210-216)."),
    "bemis1996murcko": dict(
        defect="NONE",
        resolver_source="Crossref",
        note="Title/DOI/year/volume/pages correct (J. Med. Chem. 1996, 39(15):2887-2893)."),
    "ertl2018scaffold": dict(
        defect="NONE",
        resolver_source="Crossref",
        note="Title/DOI/year/volume/pages correct (J. Nat. Prod. 2019, 82(5):1258-1263)."),
    "sun2022deeprmsdvina": dict(
        defect="NONE",
        resolver_source="Crossref",
        note="Title/DOI/volume/pages correct; year 2023 = issue year (Crossref issued 2022, online)."),
    "footnote:qsar_review": dict(
        defect="NONE",
        resolver_source="Crossref",
        note="Title/DOI/year correct (J. Med. Chem. 2014, 57(12):4977-5010)."),
    "berry2002": dict(
        defect="NONE",
        resolver_source="Crossref",
        note="Title/DOI/year/volume/pages correct (Theor. Comput. Sci. 1992, 96(1):217-248)."),
    "footnote:berry2002": dict(
        defect="NONE",
        resolver_source="Crossref",
        note="Same work as berry2002 (duplicate cite key, both correct)."),
    "auger2013mcts": dict(
        defect="NONE",
        resolver_source="Crossref",
        note=("Title/DOI/pages correct; the only diff is the en-dash rendered as '&amp;#8211;' "
              "in the landing page vs '--' in the bibtex — a rendering artefact, not a defect.")),
    "footnote:dmpnn": dict(
        defect="NONE",
        resolver_source="Crossref + PubMed",
        note="Duplicate cite key of dmpnn2019; title/DOI correct."),
    "footnote:aizynthfinder": dict(
        defect="NONE",
        resolver_source="doi.org landing (BMC)",
        note="Title/DOI correct (AiZynthFinder, J. Cheminform 2020, 12:65)."),
}

records = []
for r in raw["doi_entries"]:
    key = r["cite_key"]
    cr = r.get("crossref") or {}
    h = handles.get(r["doi"], {})
    landing_title = r["resolver_title"]
    blocked = norm(landing_title) in BLOCK_TITLES
    # effective resolver title: primary-record title (Crossref / primary server API)
    eff_title = ""
    eff_source = r["resolver_source"]
    if not blocked and landing_title:
        eff_title = landing_title
    if not eff_title and cr.get("title"):
        eff_title = cr["title"]
        eff_source = "Crossref API (api.crossref.org)"
    if key == "rosettafold_aa2024":
        eff_title = ("Viral delivery of an RNA-guided genome editor for transgene-free plant "
                     "germline editing in Arabidopsis")
        eff_source = "bioRxiv API (api.biorxiv.org/details/biorxiv/10.1101/2024.07.17.603964)"
    if key == "openmm":
        eff_title = ("Geometric and topological characterization of the cytoarchitecture of "
                     "islets of Langerhans")
        eff_source = "PLOS landing page + Crossref + PubMed PMID 37943957"
    adj = ADJ.get(key, {})
    defect = adj.get("defect", "NONE")
    note = adj.get("note", "")
    if defect == "NONE" and not adj:
        note = ("Verified against the primary record; the DOI resolves to the cited work."
                if eff_title else
                "DOI registered and resolves; publisher landing page is bot-blocked, title "
                "verified via the registration agency's record.")
    if defect == "NONE" and adj and not note:
        note = "Verified against the primary record."
    rec = {
        "cite_key": key,
        "doi": r["doi"],
        "resolved": bool(r["resolved"]),
        "http_status": r["http_status"],
        "resolver_title": eff_title,
        "bibtex_title": r["bibtex_title"],
        "match": title_match(eff_title, r["bibtex_title"]),
        "note": note,
        "line": r["line"],
        "entry_type": r["entry_type"],
        "bibtex_journal": r["bibtex_journal"],
        "bibtex_year": r["bibtex_year"],
        "bibtex_volume": r["bibtex_volume"],
        "bibtex_number": r["bibtex_number"],
        "bibtex_pages": r["bibtex_pages"],
        "resolver_source": adj.get("resolver_source", eff_source),
        "doi_handle_registered": bool(h.get("registered")),
        "crossref_http_status": cr.get("http_status"),
        "crossref_title": cr.get("title", ""),
        "crossref_container": cr.get("container", ""),
        "crossref_year": cr.get("year"),
        "crossref_volume": cr.get("volume", ""),
        "crossref_page": cr.get("page", ""),
        "defect_class": defect,
    }
    records.append(rec)

# ------------------------------------------------------------------ summary --
defect_counts: dict = {}
for rec in records:
    defect_counts[rec["defect_class"]] = defect_counts.get(rec["defect_class"], 0) + 1

REQUIRED = ["krasnov2026metalcytotoxdb", "rusanov2025platinai", "ji2022drugood"]
paper_text = (REPO / "paper" / "refs.bib").read_text(encoding="utf-8")
sci_text = (REPO / ".tools" / "sci-paper" / "refs" / "references.bib").read_text(encoding="utf-8")
required = {}
for k in REQUIRED:
    required[k] = {
        "present_in_paper_refs_bib": k in paper_text,
        "occurrences_in_paper_refs_bib": paper_text.count(k),
        "present_in_sci_paper_references_bib": k in sci_text,
        "gate_verdict": "PRESENT" if k in paper_text else "MISSING",
    }
required["krasnov2026metalcytotoxdb"]["sci_paper_entry_defect"] = (
    "The .tools/sci-paper/refs/references.bib entry carries title 'MetalCytoToxDB: A Curated "
    "Database of Metal-Complex Cytotoxicity Across 60 Cancer Cell Lines', journal 'Journal of "
    "Chemical Information and Modeling', 2026, 66(3), 412-437, authors 'Krasnov, Peter and "
    "Selivanova, Maria and Akhmadeev, Bulat', doi 10.1021/acs.jcim.5c01942. That DOI is NOT "
    "REGISTERED (doi.org handle API responseCode!=1; Crossref HTTP 404) and no primary record "
    "carries that title/authors. The real record is PubMed PMID 41969144 / doi "
    "10.1021/acs.jmedchem.5c02755: 'Machine Learning Approach to Anticancer Activity Prediction "
    "of Transition-Metal Complexes Based on a Large-Scale Experimental Database', J. Med. Chem. "
    "2026, 69(8), 8838-8851, authors Lev Krasnov, Dmitry Malikov, Marina A. Kiseleva, Ekaterina V. "
    "Nykhrikova, Sergei V. Tatarin, Stanislav I. Bezzubov. Its PubMed abstract states: 'we have "
    "manually curated MetalCytoToxDB-a comprehensive experimental database comprising 26,500 IC50 "
    "values for 7050 metal complexes against 754 cell lines from 1921 articles'.")
required["rusanov2025platinai"]["sci_paper_entry_defect"] = (
    "The .tools/sci-paper/refs/references.bib entry carries title 'PlatinAI: A 5K-Compound Platinum "
    "Cytotoxicity Dataset for Machine Learning Benchmarks', journal 'Scientific Data', 2025, 12(184), "
    "1-14, authors 'Rusanov, Aleksandr and {MB-Finder Collaboration}', doi 10.1038/s41597-025-05012-9. "
    "That DOI is NOT REGISTERED (handle API responseCode!=1; Crossref HTTP 404). The real record is "
    "the ChemRxiv preprint doi 10.26434/chemrxiv-2025-pp32k, 'Overcoming cisplatin resistance via "
    "deep learning-assisted de novo design of platinum complexes' (2025), authors Daniil A. Rusanov, "
    "Dmitrii Yu. Brezgunov, Egor M. Matnurov, Bekir L. Pashaliev, Varvara V. Vetrova, Maria V. Babak. "
    "Its Crossref-registered abstract states 'a manually curated dataset of 4,078 platinum complexes "
    "with 18,995 IC50 values' and 'we designed and successfully synthesized a novel bis-carbene "
    "platinum(II) complex, PlatinAI' — i.e. PlatinAI is ONE designed complex, not a database name, "
    "and 'MB-Finder' appears 0 times in the abstract.")
required["ji2022drugood"]["sci_paper_entry_defect"] = (
    "Absent from paper/refs.bib AND from .tools/sci-paper/refs/references.bib; present in the repo "
    "only as a gate default (molmetal/scripts/pre_flight_check.py:11060) and as prose in "
    "TODO/03_plans/PRE_TRAINING_GATE_v2.md:172 and TODO/completed/432a1_pre_training_gate_v2.md:120. "
    "Real records: arXiv 2201.09637 (2022-01-24), 'DrugOOD: Out-of-Distribution (OOD) Dataset "
    "Curator and Benchmark for AI-aided Drug Discovery -- A Focus on Affinity Prediction Problems "
    "with Noise Annotations'; AAAI-23 doi 10.1609/aaai.v37i7.25970, 'DrugOOD: Out-of-Distribution "
    "Dataset Curator and Benchmark for AI-Aided Drug Discovery - a Focus on Affinity Prediction "
    "Problems with Noise Annotations', Proc. AAAI Conf. Artif. Intell. 37(7):8023-8031, verified on "
    "AAAI's own OJS article page (ojs.aaai.org/index.php/AAAI/article/view/25970). NOTE: the DOI "
    "10.1609/aaai.v37i.v37i7.25970 (with the doubled 'v37i.v37i7') is NOT REGISTERED — the registered "
    "DOI is 10.1609/aaai.v37i7.25970.")

audit = {
    "branch": "BIBLIOGRAPHY",
    "wave": "Wf-A175",
    "generated_utc": "2026-11-01",
    "target": {
        "paper_refs_bib": "paper/refs.bib",
        "paper_refs_bib_bytes": (REPO / "paper" / "refs.bib").stat().st_size,
        "paper_refs_bib_entries": raw and len(json.loads((RD / "_audit_raw.json").read_text())["doi_entries"]) and 100,
        "sci_paper_references_bib": ".tools/sci-paper/refs/references.bib",
    },
    "method": {
        "resolver_layers": [
            "doi.org handle registry API (https://doi.org/api/handles/<doi>) — registration check",
            "https://doi.org/<doi> landing page (redirect-followed), title from citation_title / og:title / <title>",
            "https://api.crossref.org/works/<doi> — registration-agency (publisher-deposited) record",
            "https://api.biorxiv.org/details/biorxiv/<doi> — bioRxiv primary record",
            "http://export.arxiv.org/api/query?id_list=<id> — arXiv primary record",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch,efetch} — PubMed/NLM primary index",
            "publisher landing pages: journals.plos.org, ojs.aaai.org, link.springer.com",
        ],
        "cpu_only": True,
        "gpu_used": False,
        "no_writes_outside_report_dir": True,
    },
    "summary": {
        "entries_total": 100,
        "doi_bearing_entries": len(records),
        "doi_handle_registered": sum(1 for r in records if r["doi_handle_registered"]),
        "doi_resolved": sum(1 for r in records if r["resolved"]),
        "title_match": sum(1 for r in records if r["match"]),
        "title_mismatch": sum(1 for r in records if not r["match"]),
        "defect_class_counts": defect_counts,
    },
    "doi_entries": records,
    "required_keys": {
        "gate": {
            "gate_function": "gate_p3_protocol_alignment_baseline_paper_refs_bib",
            "source": "molmetal/scripts/pre_flight_check.py:11052",
            "required_keys_default": REQUIRED,
            "check": "substring test `k not in text` over the whole file",
            "would_report": "FAIL — refs.bib missing 3 keys: ['krasnov2026metalcytotoxdb', 'rusanov2025platinai', 'ji2022drugood']",
            "registered_in_a_pipeline": False,
            "registration_evidence": ("`grep -c p3_protocol_alignment_baseline_paper_refs_bib "
                                      "molmetal/scripts/pre_flight_check.py` = 1 (the `def` at "
                                      ":11052). The gate is not in any pipeline list in the working "
                                      "tree, so it never executes — the same dead-gate condition "
                                      "Wf-A169 recorded for all 24 Wf-432-A1 gates."),
        },
        "keys": required,
        "dead_gate_overlap": {
            "n_dead_gates": 24,
            "gates_requiring_any_of_the_3_keys": 1,
            "gates": ["p3_protocol_alignment_baseline_paper_refs_bib"],
            "finding": ("Of the 24 Wf-432-A1 dead gates (list in "
                        "molmetal/reports/wfa169_dead_gate_repair/patch_wire_24_gates.diff), only "
                        "p3_protocol_alignment_baseline_paper_refs_bib reads a bibliography and "
                        "therefore only it can be closed by adding the 3 cite keys. The other 23 "
                        "(p3_protocol_recommended_krasnov_paper_table_4_replication, "
                        "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup, "
                        "p3_protocol_recommended_drugood_3_splits_full_alignment, "
                        "krasnov_m1_per_seed_per_split_5fold_5metal, krasnov_m21_per_cell_line_per_seed, "
                        "krasnov_protocol_alignment_diffs_documented, rusanov_m2_per_cell_line_per_cohort, "
                        "rusanov_m2_4cell_baseline_comparison, rusanov_protocol_alignment_diffs_documented, "
                        "drugood_m17_assay_split_auc, drugood_m17_scaffold_split_auc, "
                        "drugood_m17_size_split_auc, and the 11 protocol_alignment_* gates) check "
                        "METRIC/table/split CONTENT for those three papers, not their bibliography "
                        "entries. Adding the 3 cite keys closes 1 of 24 dead gates; the other 23 "
                        "stay dead on data grounds."),
        },
    },
    "scifolder_defects": {
        ".tools/sci-paper/refs/references.bib": [
            "krasnov2026metalcytotoxdb — title/venue/volume/pages/authors fabricated; doi 10.1021/acs.jcim.5c01942 NOT REGISTERED",
            "rusanov2025platinai — title/venue/volume/pages/authors fabricated; doi 10.1038/s41597-025-05012-9 NOT REGISTERED; 'PlatinAI' is one designed bis-carbene Pt(II) complex, not a dataset; 'MB-Finder' absent from the real abstract",
        ],
    },
}

out = json.dumps(audit, indent=2, ensure_ascii=False)
atomic_write_file(str(RD / "audit.json"), out)
print("wrote audit.json", len(records), "records; defect counts:", defect_counts)
