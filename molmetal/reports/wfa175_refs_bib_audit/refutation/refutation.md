# Wf-A175 — adversarial refutation of the BIBLIOGRAPHY branch audit

**Date:** 2026-11-01 · **Branch:** BIBLIOGRAPHY · **Mode:** CPU-only, read-only on `paper/**`,
`.tools/**`, `docs/HONEST_NEGATIVE_CATALOG.md`, `AGENTS.md`, `TODO/**`, `README.md`,
`molmetal/scripts/pre_flight_check.py`, `metrics/**`.

**REFUTED: 1 | SURVIVED: 16 | IMPRECISE: 0**

Per-claim verdicts with the counter-evidence: [`refutation.json`](refutation.json).

---

## 1. The one REFUTED claim — CLAIM 6, the AttentiveFP issue number

The claim as written (VERBATIM, quoted per AGENTS.md §5.5 rule 6):

> CLAIM 6: `attentivefp2020` + `footnote:attentivefp` carry the arXiv preprint title on a correct
> version-of-record DOI; the version of record is "Pushing the Boundaries of Molecular
> Representation for Drug Discovery with the Graph Attention Mechanism", J. Med. Chem.
> 63(12):8749-8760; also 2 wrong author names (Zhenjun→Zhaojun, Xiang→Xiaomin) and issue 16→12

**REFUTED on the issue number. SURVIVED on the title and on both author names.**

Four independent indexes, all fetched by me during this refutation, say the issue is **16**:

| Source | Field | Value |
|---|---|---|
| `api.crossref.org/works/10.1021/acs.jmedchem.9b00959` | `issue` | `16` |
| same response | `journal-issue.issue` | `16` |
| `esummary.fcgi?db=pubmed&id=31408336` | `issue` | `16` |
| `efetch.fcgi?db=pubmed&id=31408336&rettype=medline` | `IP` | `16` (with `VI - 63`, `DP - 2020 Aug 27`) |
| `api.openalex.org/works/doi:10.1021/acs.jmedchem.9b00959` | `biblio.issue` | `16` (volume 63, 8749–8760) |

`paper/refs.bib:136` already carries `number = {16}` — i.e. **the field the audit wants to
"correct" is already correct.** The audit's `audit.json` row for this DOI has **no
`crossref_issue` field at all** (its 24 keys include `crossref_volume` and `crossref_page`
but no issue), so `12` was never measured — it was asserted.

### Consequence: the STAGED deliverable would introduce a NEW error

`molmetal/reports/wfa175_refs_bib_audit/refs_corrected_STAGED.bib:399` (`attentivefp2020`)
and `:424` (`footnote:attentivefp`) both write:

```
  number       = {12},
```

and the note at `:405` reads:

> Wf-A175 correction: the title now matches the version of record, and the number field is
> corrected from 16 to 12 (Crossref issue 12; PubMed 63(12):8749-8760).

Both attributions are false. Landing this file would corrupt a field that is currently right.
**Fix:** keep `number = {16}`, delete the issue change, and drop the misattributed sentence
from the note.

**What survives in CLAIM 6:** the version-of-record title ("Pushing the Boundaries of
Molecular Representation for Drug Discovery with the Graph Attention Mechanism" — Crossref and
PubMed agree) and both author-name corrections. Crossref's full author list is
`Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li,
Xiaomin Luo, Kaixian Chen, Hualiang Jiang, Mingyue Zheng`; `paper/refs.bib:136` has
`Li, Zhenjun` (7th) and `Luo, Xiang` (8th) — both wrong, and the audit's Zhaojun/Xiaomin
substitutions are correct.

---

## 2. What survived, and what I actually did to check it

Every figure in CLAIM 1 was recomputed from `audit.json` (not from the report prose) and
reproduced exactly: 100 `@`-entries / 40 DOI-bearing / 38 registered / 38 resolved /
31 title-match / 9 mismatch, defect histogram `NONE 30 · WRONG_DOI 5 · TITLE_MISMATCH_SAME_WORK 2 ·
WRONG_YEAR 1 · FABRICATED_DOI 1 · FABRICATED_DOI+FABRICATED_TITLE 1` = 10 defective of 40.

Independent primary-record verification I performed myself (not copied from the audit):

- **CLAIM 3** — bioRxiv API for `10.1101/2024.07.17.603964` → "Viral delivery of an RNA-guided
  genome editor for transgene-free plant germline editing in Arabidopsis", Weiss/Kamalu/Shi,
  2024-07-19. The audit's replacement `10.1101/2023.10.09.561603` → "Generalized Biomolecular
  Modeling and Design with RoseTTAFold All-Atom".
- **CLAIM 5** — Crossref for all three DOIs reproduces the claimed titles/venues/pages exactly
  (`Histochem. Cell Biol. 106(6):581-586`; `Coord. Chem. Rev. 253(17-18):2186-2192`;
  `PLoS Comput Biol 19(11):e1011617`).
- **CLAIM 7** — Crossref `issued` 2024 for `10.1039/D3SC04185A`; `refs.bib` says `year={2023}`
  while its own `url` says `/2024/SC/`.
- **CLAIM 11 / 12** — both `.tools/sci-paper` DOIs return `responseCode:100`; the REAL Krasnov
  record (PMID 41969144) is `J. Med. Chem. 2026, 69(8):8838-8851`,
  `10.1021/acs.jmedchem.5c02755`, 6 authors, and its abstract says verbatim "26,500 IC50 values
  for 7050 metal complexes against 754 cell lines from 1921 articles" — no "60 cancer cell lines"
  anywhere. The REAL Rusanov record is the ChemRxiv preprint
  `10.26434/chemrxiv-2025-pp32k` "Overcoming cisplatin resistance via deep learning-assisted de
  novo design of platinum complexes"; in that abstract `'MB-Finder'` occurs **0** times and
  "PlatinAI" is **one** novel bis-carbene platinum(II) complex.
- **CLAIM 13** — Crossref `10.1609/aaai.v37i7.25970` = AAAI 37(7):8023-8031 (2023), 17 authors;
  the doubled `10.1609/aaai.v37i.v37i7.25970` → `responseCode:100`; arXiv:2201.09637 published
  2022-01-24.

### Two scope caveats that do not falsify a claim but change what to do

- **CLAIM 9** — "never executes" is true of the **working tree only**. The patch the branch
  itself cites as evidence (`wfa169_dead_gate_repair/patch_wire_24_gates.diff`, +295/−1, NOT
  applied) registers this gate at diff:149–155 into `all` + `launch`. Wf-A169's FAIL string
  therefore came from a *patched scratch copy* (`apply_probe.json`), not from the working tree.
  Also checked: **no** script anywhere passes `--gates`, so the
  `directives.pre_flight_gates_required` lists in 5 `configs/training/*.yaml` are themselves
  inert. Operative conclusion unchanged — the gate is dead now, and the bibliography repair and
  the gate repair are the **same** write, still BLOCKED on `paper/refs.bib` (AGENTS.md §5.3).
- **CLAIM 17** — all components re-verified offline (11/11 verbatim originals found in the
  STAGED file by my own matcher; 3/3 required keys present; cite keys unique;
  `detect_truncation` → `None` on all three deliverables; `atomic_write_file` called 4× in
  `build_deliverables.py`). The git-cleanliness assertion is **not vacuous**: `paper/refs.bib`
  is tracked (`git ls-files --error-unmatch` succeeds) and `.tools/` is **not** gitignored
  (`git check-ignore` rc=1). Precision caveat: the STAGED bib holds **13 active entries vs 100**
  in `paper/refs.bib` — it is a replacement-block overlay, not a drop-in full file; copying it
  over would delete 87 entries.

---

## 3. The branch's own blind spot, stated for the record

The audit's census is sound; its **defect-recording layer** is what let the issue number through.
`audit.json` captures `crossref_volume`, `crossref_page`, `crossref_year`, `crossref_title`,
`crossref_container` — but **no `crossref_issue`**. A defect class cannot be assigned to an
issue field that is never captured, so `number` was audited by eyeball and "corrected" against a
number that four primary records contradict. The same gap class let the STAGED file carry a
`number = {12}` with a note that *cites* Crossref and PubMed as its authority.

Two smaller recording gaps, same shape: the audit stores no final landing URL (so the 403 host
attribution in CLAIM 16 rests on the raw payloads rather than on a stored field), and the
STAGED bib keeps the wrong DOIs inside its `%`-commented verbatim blocks (inert for BibTeX, but
a naive `grep -c 'doi *='` on the STAGED file returns 14, not the count of live DOIs).
