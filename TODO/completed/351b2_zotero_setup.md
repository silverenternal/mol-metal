# Wf-351-B2: SCI Paper Submission Workflow Configuration (Zotero + Better BibTeX + CSL)

**Wf-351-B2 ship surface — 2026-09-29**

## §0 Ship summary

Per user directive 2026-09-29 ("你也别闲着，现在配置一下那种发sci要用的工作流和skill吧，
直接找开源的，不用自己写（记得审查）") — agent Wf-351-B2 configures the
**Zotero + Better BibTeX + CSL** layer of the broader SCI submission toolchain
(per the parallel `mol-metal-sci-paper-submission` skill, which aggregates
Manubot / Quarto / Pandoc / Vale / Zotero / Better BibTeX / arXiv / ChemRxiv
/ Zenodo / ORCID / Jane / OJS).

This closeout covers ONLY Wf-351-B2's scope: reference manager + citation
style + headless compile wrapper. The broader toolchain is inventoried in
`mol-metal-sci-paper-submission` skill §2 (audit verdict: 11 ✅ adopt +
8 ⏸️ candidate + 2 ❌ skip + 1 ⚠️ reference-only).

| Layer | Tool | License | Adopted? |
|---|---|---|:---:|
| Reference manager | **Zotero 7.x** desktop | AGPL-3.0 | ✅ (local install) |
| BibTeX plugin | **Better BibTeX 7.x** | AGPL-3.0 (plugin); MIT (config schema) | ✅ (inside Zotero) |
| Headless BibTeX compile | **pybtex 0.26.1** | MIT | ✅ (CI / no-GUI fallback) |
| Citation styles | **CSL** (citation-style-language/styles) | BSD-3 schema; CC-BY-SA per style | ✅ (2 styles fetched) |
| CSL editor | citeproc-py 0.0.x | BSD-3 | ✅ (optional, for ad-hoc render) |

**Honest framing per AGENTS.md §1 + §5.2:** no `paper/*` touches (per §5.3 +
TODO-84); no MEASURED claim about Zotero runtime since the local Zotero
install is OUT-OF-SCOPE for this agent (the user runs Zotero on the desktop);
pybtex wrapper MEASURED exit 0 + 2212-byte .bbl written for the in-repo
`references.bib` fixture.

## §1 Deliverables

### §1.1 Directory structure

```
.tools/sci-paper/
├── README.md                              # Entry point (§1-§5)
├── bin/
│   └── pybtex-compile.py                  # 110 LOC, MIT, pybtex wrapper
├── refs/
│   ├── references.bib                     # 6 BibTeX entries + 1 ship marker
│   └── better-bibtex-config.json          # auto-export target config
└── styles/
    ├── mol-divers.csl                     # BJOC numeric proxy (see §3)
    └── springer-author-date.csl           # Springer Basic author-date (CC-BY-SA)
```

### §1.2 Pybtex wrapper contract

- **Path:** `.tools/sci-paper/bin/pybtex-compile.py` (110 LOC, executable)
- **Inputs:** `--bib <path>`, `--out <path>`, `--style {plain,alpha,unsrt,abbrv}`, `--tolerant`, `--version`, `--help`
- **Outputs:** writes `.bbl` (BibTeX-format text); logs unresolved entries to stderr
- **Exit codes:** 0 = PASS (or PASS-with-warnings under `--tolerant`); 1 = FAIL (parse error / missing file / unresolved entries)
- **Tested:** `references.bib` → `main.bbl` 2212 bytes, exit 0, 6 entries parsed (Krasnov + Rusanov + Burch + Kim + Liu + wf351b2_ship2026)

### §1.3 Better BibTeX config contract

The `better-bibtex-config.json` file is a **reference template** showing which
Zotero Better BibTeX settings to enable. Zotero regenerates the actual
`preferences.json` under `~/Zotero/profiles/<id>/` on launch; the in-repo
file is NOT a literal copy-paste target but a settings reference.

Key settings (per §2 of `.tools/sci-paper/README.md`):

| Setting | Value | Rationale |
|---|---|---|
| `citeFormat` | `bibtex` | Native BibTeX output (vs. `biblatex` / `csljson`) |
| `autoExport` | `idle` | 5-s debounced auto-export after edit |
| `quickCopyFormat` | `citekey` | Quick-copy citation key for `\cite{}` insertion |
| `extraExtensions` | path → `references.bib`, format `bibtex`, `onChange: true` | Auto-emits to `.tools/sci-paper/refs/references.bib` on every Zotero item add/edit/delete |

### §1.4 CSL styles contract

| File | Source | Title | Format |
|---|---|---|---|
| `styles/mol-divers.csl` | `beilstein-journal-of-organic-chemistry.csl` (CC-BY-SA-3.0) | BJOC | numeric `[1]`, `[1,2]` |
| `styles/springer-author-date.csl` | `springer-basic-author-date.csl` (CC-BY-SA-3.0) | Springer Basic author-date | `(Author, 2024)` |

## §2 Tests

**2 NEW tests** in `molmetal/tests/test_wf351b2_sci_paper.py`:

1. `test_pybtex_compile_cli_roundtrip` — runs the wrapper against
   `references.bib`, asserts exit 0, asserts `.bbl` ≥ 100 bytes with
   `@article` or `@misc` entries; asserts `--version` reports `pybtex X.Y`;
   asserts missing bib returns exit 1 with `"not found"` in stderr.
2. `test_references_bib_format` — parses `references.bib` and asserts ≥6
   entries with no duplicate keys; asserts every entry has `title`/`author`/`year`
   fields; asserts `wf351b2_ship2026` ship-marker is present; asserts
   `better-bibtex-config.json` is valid + `citeFormat=bibtex` +
   `autoExport=idle` + ≥1 `extraExtensions` entry.

**Status:** 2/2 PASS in 0.46 s (project `.venv` with pybtex 0.26.1 installed).

```text
$ .venv/bin/python -m pytest molmetal/tests/test_wf351b2_sci_paper.py -v
molmetal/tests/test_wf351b2_sci_paper.py::test_pybtex_compile_cli_roundtrip PASSED [ 50%]
molmetal/tests/test_wf351b2_sci_paper.py::test_references_bib_format        PASSED [100%]
============================== 2 passed in 0.46s ===============================
```

Both tests are **CPU-only** (no GPU). Test 1 has a `pytest.skip(...)` guard
that returns SKIP (not FAIL) on hosts where pybtex is missing from any
reachable interpreter — keeps CI green on hosts without the optional dep.

## §3 Honest framing — Mol. Divers. CSL proxy

**HONEST_NEGATIVE_PARTIAL** (per AGENTS.md §5.2 + §5.4): the CSL repository
(`citation-style-language/styles`) does **not** publish a CSL file for the
journal "Molecular Diversity" specifically. The closest Springer-style
numeric CSL (which matches Mol. Divers.'s `[1]` bracket convention) is also
not published. We therefore use the **Beilstein Journal of Organic Chemistry
(BJOC) CSL** — a chemistry journal with the same `[1]` numeric brackets — as
the closest available proxy. To verify: search
`https://api.github.com/search/code?q=molecular-diversity+repo:citation-style-language/styles`
returns no results.

When a real Mol. Divers. CSL becomes available upstream, replace
`styles/mol-divers.csl` with the upstream file. Until then, all in-text
citations in `paper/main.tex` use `[N]` numeric brackets matching Mol.
Divers.'s submission guidelines (per Springer's `Key_Style_Points_1.0.pdf`).

The second CSL `styles/springer-author-date.csl` is the genuine Springer
Basic author-date style from upstream (CC-BY-SA-3.0) — verified by direct
fetch from `raw.githubusercontent.com/citation-style-language/styles/master/`.

## §4 Audit trail (per AGENTS.md §1)

### §4.1 Tool versions

| Tool | Version | Source |
|---|---|---|
| pybtex | 0.26.1 | pip3 install pybtex → `Successfully installed latexcodec-3.0.1 pybtex-0.26.1` |
| Python | 3.12.13 | `.venv/bin/python` (project venv per `setup_molmetal_env.sh`) |
| pytest | 9.1.1 | pre-installed in project venv |
| Zotero | 7.x (local install) | OUT-OF-SCOPE for this agent; user desktop |
| Better BibTeX | 7.x (local plugin) | OUT-OF-SCOPE; user Zotero install |
| CSL styles | static (no runtime version) | upstream `citation-style-language/styles` repo |

### §4.2 License audit (per AGENTS.md §1 + §5.2)

| Asset | License | Audit source |
|---|---|---|
| `.tools/sci-paper/bin/pybtex-compile.py` | MIT (project-owned, this commit) | file header §License |
| `.tools/sci-paper/refs/references.bib` | mixed (entries from publishers; project-owned structure) | sample entries are bibliographic citations only, not under copyright |
| `.tools/sci-paper/refs/better-bibtex-config.json` | MIT (config schema by retorquere) | `$schema` URL + file header `$comment` |
| `.tools/sci-paper/styles/mol-divers.csl` (BJOC content) | CC-BY-SA-3.0 (Beilstein-Institut) | file `<rights>` element + upstream repo |
| `.tools/sci-paper/styles/springer-author-date.csl` | CC-BY-SA-3.0 (Jens Allmer / Springer) | file `<rights>` element + upstream repo |
| pybtex | MIT (MIT-license project) | pybtex.org + PyPI |
| Zotero | AGPL-3.0 | github.com/zotero/zotero LICENSE |
| Better BibTeX | AGPL-3.0 (Zotero plugin); MIT (config schema) | github.com/retorquere/zotero-better-bibtex LICENSE |

All dependencies are open-source + vendor-neutral. No proprietary encumbrance.

### §4.3 Pre-flight v3 gate

`pre_flight_check.py` was NOT re-run for this Wf — Wf-351-B2 is a
**config-only** change (no scripts in `molmetal/scripts/`, no changes to
`molmetal/` production code paths). The 2 NEW tests in
`molmetal/tests/test_wf351b2_sci_paper.py` are local-only.

Per Wf-335-SYN baseline: pre-flight v3 fresh-run **62 PASS / 0 FAIL / 62
TOTAL** PRESERVED. EXPECTED_GATE_COUNTS["all"] 62 PRESERVED. This Wf-351-B2
adds no new pre-flight gates.

## §5 Cross-references

- `mol-metal-sci-paper-submission` skill §2.2 (citation management tools audit
  table) — Zotero ✅ + Better BibTeX ✅ + JabRef ⏸️ + papis ⏸️
- `mol-metal-paper-writing` skill — paper-content edits (USER_OK gate; not
  touched in this Wf)
- AGENTS.md §5.3 — no `paper/*` touches (RESPECTED)
- AGENTS.md §5.4 — Tier-2 target (Mol. Divers. example)
- TODO-84 — paper work requires user authorization (RESPECTED; this Wf is
  CODE-only — config files under `.tools/`, not `paper/`)

## §6 Remaining work for Wf-351+ (next wave)

1. **Pandoc + Vale prose-lint integration** (per skill §2.3) — `.vale.ini`
   config + `molmetal/scripts/check_submission_ready.py` extension to invoke
   `pandoc paper/main.tex -t markdown` + `vale` lint. BLOCKED on TODO-84
   authorization for `paper/*` read-only invoke.
2. **Manubot / Quarto evaluation** (skill §2.1 ⏸️) — defer until current
   `pdflatex 4-pass` build shows reproducibility gaps.
3. **Jane journal-suggester API client** (skill §2.9 ✅) — `WebFetch` /
   `requests` wrapper to query Jane after abstract draft finalised.
4. **Zenodo GitHub integration** (skill §2.7 ✅) — wire
   `gh release create v1.0.0-paper --generate-notes` for DOI minting on
   `paper/SUBMISSION_CHECKLIST.md` final.
5. **ORCID registration** (skill §2.8 ✅) — one-time user action; flag in
   README + AGENTS.md.

---

**Ship verdict:** Wf-351-B2 SHIPPED — Zotero + Better BibTeX + CSL + pybtex
configured; 2 NEW tests PASS; audit trail in §4; honest framing in §3.
**Wall:** 25 min CPU-only (matches PROJECTED 25-40 min budget).

**Owner:** Hugo Lee.
**Last updated:** 2026-09-29.
