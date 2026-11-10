# Wf-A175 / BIBLIOGRAPHY — `paper/refs.bib` DOI integrity audit

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
