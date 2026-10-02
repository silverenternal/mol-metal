# SCI Paper Workflow — Mol-Metal

**Toolchain layer for Tier-2 journal submission** (Mol. Divers. / JCAMD / J. Mol. Graph. Model. / Comput. Biol. Chem. — per AGENTS.md §5.4).

> **Scope:** READ-only on `paper/*` per AGENTS.md §5.3. This directory
> configures the **build pipeline** (Zotero refs + CSL citation styles + pybtex
> compile wrapper); it never edits the paper itself. Use
> `mol-metal-paper-writing` + USER_OK gate for paper-content edits.

## §1 Setup

### §1.1 Install Zotero + Better BibTeX (one-time per machine)

```bash
# 1. Install Zotero (AGPL-3.0, https://www.zotero.org/download/)
#    macOS:   brew install --cask zotero
#    Linux:   flatpak install flathub org.zotero.Zotero
#    Windows: download .exe from zotero.org

# 2. Inside Zotero: Tools → Add-ons → Install Add-on From File
#    Download Better BibTeX (MIT, https://retorquere.github.io/zotero-better-bibtex/)
#    Restart Zotero when prompted.

# 3. Configure Better BibTeX auto-export (see §2):
#    Edit → Settings → Better BibTeX → Automatic Export
#    Add:  Format = BibTeX
#          Collection = Mol-Metal (root collection)
#          Path = $HOME/codes/mol-metal/.tools/sci-paper/refs/references.bib
#          On change = yes
```

### §1.2 Install Python tools

```bash
pip install pybtex          # MIT, programmatic .bib → .bbl compile fallback
pip install citeproc-py     # BSD, CSL → formatted citations
```

### §1.3 Verify CSL files

```bash
ls -la .tools/sci-paper/styles/
# mol-divers.csl          — proxy for Mol. Divers. (Springer numeric [1])
# springer-author-date.csl — Springer Basic author-date reference
```

## §2 Zotero Better BibTeX config

The JSON in `refs/better-bibtex-config.json` is the auto-export configuration
template. Copy to `~/.zotero/zotero-BgDBwRUq/` (Linux) or
`~/Library/Application Support/Zotero/profiles/<id>/` (macOS) — Zotero will
regenerate this on first launch; the file is provided as a **reference** for
which settings to enable, not as a literal file to import.

| Setting | Value | Why |
|---|---|---|
| `citeFormat` | `bibtex` | Native BibTeX output (vs. `biblatex` / `csljson`) |
| `autoExport` | `idle` | Export on idle after change (debounced 5 s) |
| `quickCopyFormat` | `citekey` | Quick-copy citation key for `\cite{}` insertion in LaTeX |
| `parseJabrefAbbreviations` | `true` | Parse JabRef abbreviation files |

The BibTeX auto-exporter (`On Change`) emits to `refs/references.bib` — this
is the file `tectonic` / `biber` consumes during compile.

## §3 pybtex programmatic fallback

For **CI / headless builds** without Zotero running, use the Python wrapper:

```bash
python .tools/sci-paper/bin/pybtex-compile.py \
    --bib  .tools/sci-paper/refs/references.bib \
    --out  paper/main.bbl \
    --style plain
```

Features:
- Reads `.bib` via `pybtex.database.input.bibtex.Parser`
- Compiles to `.bbl` via `pybtex.database.output.bibtex`
- Logs unresolved references to stderr (citation-key → empty entry warning)
- Exits 0 if all entries resolved, 1 if any unresolved (unless `--tolerant`)

This wrapper is **deterministic + testable** (no GUI dependency) — used by
the Tier-2 submission workflow's Stage 2 Validation step (see
`mol-metal-sci-paper-submission` skill §1).

## §4 CSL styles for Mol. Divers. (Springer)

The `styles/` directory contains two CSL files:

| File | Upstream | Format | Citation marker |
|---|---|---|---|
| `mol-divers.csl` | `beilstein-journal-of-organic-chemistry.csl` (CC-BY-SA-3.0) | numeric | `[1]`, `[1,2]` |
| `springer-author-date.csl` | `springer-basic-author-date.csl` (CC-BY-SA-3.0) | author-date | `(Author, 2024)` |

### §4.1 Honest framing — Mol. Divers. CSL proxy

**HONEST_NEGATIVE_PARTIAL** (per AGENTS.md §5.2): the CSL repository
(`citation-style-language/styles`) does **not** publish a CSL file for the
journal "Molecular Diversity" specifically. The closest Springer-style
numeric CSL (which matches Mol. Divers.'s `[1]` bracket convention) is also
not published. We therefore use the Beilstein Journal of Organic Chemistry
(BJOC) CSL — a chemistry journal with the same `[1]` numeric brackets — as
the closest available proxy. To verify: search
`https://api.github.com/search/code?q=molecular-diversity+repo:citation-style-language/styles`
returns no results.

When a real Mol. Divers. CSL becomes available, replace
`styles/mol-divers.csl` with the upstream file. Until then, all in-text
citations in `paper/main.tex` use `[N]` numeric brackets matching Mol.
Divers.'s submission guidelines (per Springer's `Key_Style_Points_1.0.pdf`).

### §4.2 License audit (per AGENTS.md §1 + Wf-351-A4)

- `mol-divers.csl` (BJOC): CC-BY-SA-3.0 (Beilstein-Institut)
- `springer-author-date.csl`: CC-BY-SA-3.0 (Jens Allmer)
- CSL schema itself: BSD-3 (citation-style-language/styles)

All three are open-source + vendor-neutral; no proprietary encumbrance.

## §5 Tool version pinning (reproducibility)

```bash
# tectonic --version        # 0.15.x or later (per Wf-351-A2)
# zotero --version          # 7.x or later (Better BibTeX 7.x compat)
# python -c "import pybtex; print(pybtex.__version__)"   # 0.24.x
# python -c "import citeproc_py; print(citeproc_py.__version__)"  # 0.0.x
```

Versions recorded in `TODO/completed/351b2_zotero_setup.md` at ship time.

---

**Last updated:** 2026-09-29 (Wf-351-B2 ship surface).
**Owner:** Hugo Lee.
**Cross-ref:** `mol-metal-sci-paper-submission` skill §1 Stage 1-4 build pipeline.