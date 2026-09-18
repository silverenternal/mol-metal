# WF-Paper-Compile — PDF well-formed audit

**Date:** 2026-09-14
**Auditor:** workflow-orchestrator subagent
**Scope:** `paper/main.pdf` post-pdflatex+bibtex

## Honest summary up-front

The PDF on disk (`paper/main.pdf`, 4.3 MB, 45 pages, mtime 2026-09-14 13:07:06)
**was** produced by an earlier successful pdflatex pass.  The most recent
pdflatex invocation (log mtime 2026-09-14 13:07:25) terminated with a
**Fatal error** and **did NOT regenerate the PDF** — the file on disk is the
stale artefact.  Any compile-clean claim must therefore be qualified: the
*contents* of the PDF are valid (sections, figures, references all present),
but the *source* currently does not build.

## 1. PDF file & metadata

```
$ pdfinfo paper/main.pdf
Title:           <empty>            <-- see note below
Subject:         <empty>
Author:          <empty>            <-- TODO placeholder in body, not in metadata
Creator:         LaTeX with hyperref
Producer:        pdfTeX-1.40.29
CreationDate:    Mon Sep 14 13:07:01 2026 HKT
ModDate:         Mon Sep 14 13:07:01 2026 HKT
Pages:           45
Page size:       595.276 x 841.89 pts (A4)
File size:       4321726 bytes (4.3 MB)
PDF version:     1.7
```

**File size:** 4 321 726 B (4.3 MB).
**First-page content** (pdftotext):

> Mol-Metal: a hybrid molecular lambda calculus + flow-matching framework
> for precious-metal drug design
> TODO: author list to be populated by the corresponding-author workflow.
> September 14, 2026

### Title check
`pdfinfo` reports `Title: <empty>`, but the **first page text** clearly
starts with `Mol-Metal: a hybrid molecular lambda calculus + flow-matching
framework for precious-metal drug design`, so the title string **does
contain "Mol-Metal"** even though no `\title` metadata hook is wired into
`hyperref`.  This is cosmetic — PDF viewers will fall back to the visible
title on page 1.

### Author check
No metadata `Author` field.  The visible author block on page 1 reads
`TODO: author list to be populated by the corresponding-author workflow.`,
so the **TODO placeholder is present** as expected.

### Page-count check
`Pages: 45` — passes the > 4 gate by an order of magnitude.

## 2. Figure references

```
$ pdftotext paper/main.pdf - | grep -c 'Figure [123]'
10
```

10 textual references to Figure 1 / Figure 2 / Figure 3 across the body,
plus 3 caption-style occurrences (`Figure 1:`, `Figure 2:`, `Figure 3:`) =
**3 figure environments rendered**.  Pass.

## 3. Section presence

`pdftotext` locates the seven top-level section headers at the following
lines (one per top-level section, plus a re-use of "Method" in the appendix
at line 1431 which is expected — it is the closure-theorem proof section):

| line | section header        | status |
| ---- | --------------------- | ------ |
| 34   | Introduction          | OK     |
| 155  | Related Work          | OK     |
| 354  | Method                | OK     |
| 1179 | Evaluation            | OK     |
| 1235 | Ablation Studies      | OK     |
| 1280 | Limitations           | OK     |
| 1359 | Future Work           | OK     |
| 2727 | References            | OK     |

**All 7 numbered sections + References are present.** Pass.

(The 1431 "Method" is `paper/appendix_betanf_semantics.tex` heading; it is
not a body section — the body §3 "Method" lives at line 354.)

## 4. Bibliography rendering

```
$ pdftotext paper/main.pdf - | grep -c '^References$'
1
```

The `References` heading is present exactly once.  `paper/main.bbl` (12 KB)
defines 36 bib items; `paper/refs.bib` contains 73 `@` entries; the
`paper/main.aux` records 36 `\bibcite` bindings.  The bibliography is
**rendered** in the PDF (lines 2728-end of extracted text).  Pass.

## 5. Compilation log (most recent run)

```
$ grep -E '^!' paper/main.log
! Package natbib Error: Bibliography not compatible with author-year citations.
!  ==> Fatal error occurred, no output PDF file produced!
```

```
$ grep -cE '^!'   paper/main.log  # hard errors
2
$ grep -cE 'Warning' paper/main.log
5
$ grep -cE 'Error' paper/main.log
1
```

### Error inventory

| severity | count | source |
| -------- | ----- | ------ |
| Hard LaTeX error (`!`) | 2 | natbib author-year/numerical mismatch |
| LaTeX Warning: multiply-defined label | 2 | `eq:cuaac-reduction`, `fig:click-reactions` |
| natbib Warning: multiply-defined citation | 3 | `Barendregt1984`, `Girard1989`, `Pierce2002` |
| bibtex warning | 1 | missing `appendix:closure-theorem` bibkey |
| Source: ambiguous label as cite | 1 | `paper/sections/03_1_mlc_formalism.tex:24,322` uses `\cite{appendix:closure-theorem}` which is a **label**, not a bibkey |

### Root causes

1. **natbib fatal** — `paper/main.tex` line 39 loads
   `\usepackage[round,authoryear]{natbib}` (author-year mode) but line 40
   uses `\bibliographystyle{unsrtnat}` (numerical style).  Natbib's
   author-year hooks cannot cooperate with a numerical .bst; it errors
   out.  This is what killed the most recent run.
2. **Duplicate `\label` for `eq:cuaac-reduction`** —
   `paper/sections/03_1_mlc_formalism.tex:260` and
   `paper/sections/03_2_click_chemistry.tex:110` both define the label.
3. **Duplicate `\label` for `fig:click-reactions`** —
   `paper/main.tex:259` and `paper/sections/03_2_click_chemistry.tex:172`
   both define the label.
4. **Triple-defined `Barendregt1984`/`Girard1989`/`Pierce2002`** —
   `paper/appendix_betanf_semantics.tex` defines them inline with
   `\bibitem`, and `paper/refs.bib` also contains them.
5. **Bogus cite of label** —
   `paper/sections/03_1_mlc_formalism.tex:24` and :322 use
   `\cite{appendix:closure-theorem}` which is a `\label` not a bibkey.
   bibtex emits "I didn't find a database entry for
   appendix:closure-theorem".

### Fix log (recommended; pure-LaTeX, not applied yet)

The task says "Pure-LaTeX compilation workflow: NO code modifications to
molmetal/ or paper/sections/."  The cleanest fix is therefore in
`paper/main.tex`:

* Line 39 — change `natbib` options to either:
  - keep `[round,authoryear]` and switch `\bibliographystyle{unsrtnat}`
    → `\bibliographystyle{plainnat}` (author-year), **OR**
  - drop `authoryear` → `\usepackage[round]{natbib}` and keep
    `unsrtnat` (numerical).
* Line 40 — match the bst to the natbib options.

Without removing the duplicate `\label`s and the bogus `\cite{...}` of a
label, the build will still emit warnings even after the natbib fix.  To
get an **all_clean=true** audit, the following follow-ups are needed:

* `paper/sections/03_1_mlc_formalism.tex:260` — drop or rename the
  duplicate `\label{eq:cuaac-reduction}` (keep the one in
  `03_2_click_chemistry.tex:110`).
* `paper/main.tex:259` — drop or rename the duplicate
  `\label{fig:click-reactions}` (keep the one in
  `03_2_click_chemistry.tex:172`).
* `paper/appendix_betanf_semantics.tex` — remove the inline `\bibitem`
  definitions for `Barendregt1984`/`Girard1989`/`Pierce2002` (the same
  keys live in `paper/refs.bib`).
* `paper/sections/03_1_mlc_formalism.tex:24,322` — replace
  `\cite{appendix:closure-theorem}` with `\ref{appendix:closure-theorem}`.

None of the above touches `molmetal/`.

## 6. PDF-content sanity (against the stale but valid PDF)

```
$ pdftotext paper/main.pdf - | grep -E '^(Introduction|Related Work|Method|Evaluation|Ablation Studies|Limitations|Future Work|References)$'
Introduction
Related Work
Method
Evaluation
Ablation Studies
Limitations
Future Work
Method            <-- appendix_betanf_semantics.tex heading (expected)
References
```

The bibliography block is rendered (lines 2728-end); the per-figure
captions appear; section cross-references resolve (no `??` placeholders
visible in the extracted text, only the body-text footnote `Appendix ??`
which is intentional TODO markup).

## 7. Conclusion

| metric                    | value | gate      | result |
| ------------------------- | ----- | --------- | ------ |
| pdf_pages                 | 45    | > 4       | PASS   |
| title contains Mol-Metal  | YES (body) | required | PASS |
| author field / TODO       | TODO placeholder present | required | PASS |
| n_figures_in_pdf          | 3     | >= 3      | PASS   |
| n_sections_in_pdf         | 7 + References | >= 7 | PASS |
| bib_resolved              | YES (36 bbl entries) | required | PASS |
| hard LaTeX errors         | 2 (natbib) | 0 expected | **FAIL** |
| all_clean                 | FALSE (latest run Fatal, PDF stale) | TRUE expected | **FAIL** |

`all_clean` is **false**: the on-disk PDF is well-formed (PASS on content
checks), but the source as it stands today does NOT compile cleanly — the
most recent pdflatex invocation died on a natbib author-year/numerical
mismatch, leaving the PDF as a stale 13:07:06 artefact.  A re-run is
blocked until the natbib options / bst pairing is reconciled and the
duplicate labels are pruned.

## 8. Suggested next steps (for the human / a follow-up wf)

1. `paper/main.tex:39` — pick one of the two natbib / bst combinations
   above.  Recommend `\usepackage[round]{natbib}` + `unsrtnat` (matches
   the existing numerical citation style already used in the body).
2. Re-run `pdflatex main && bibtex main && pdflatex main && pdflatex main`
   from `paper/`.
3. De-duplicate labels (see fix list above) to clear the 2 LaTeX warnings
   and 3 natbib warnings.
4. Re-run this audit; expect `all_clean=true`.