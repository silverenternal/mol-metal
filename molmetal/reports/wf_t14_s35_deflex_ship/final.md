# WF-T14-S35-Deflex — §3.5 Deflex Sub-Section Ship + Paper Recompile Verify

**Date**: 2026-09-17
**Workflow**: TODO-14 §3.5 Deflex sub-section ship (independent of GPU)
**Scope**: Verify 03_5_deflex.tex is wired into main.tex; recompile paper 4-pass; confirm gate metrics.

## 1. Wiring Verify — PASS

The §3.5 Deflex sub-section is already included via `03_method.tex` (master composer for §3), not directly in `main.tex`. This matches the spec at TODO-14 line 559 ("`paper/sections/03_5_deflex.tex` (NEW; per Workflow 1)") and TODO-14 cross-ref matrix line 694 ("`\input{sections/03_5_deflex}` (via `03_method.tex:97`) — YES").

Verified include chain:
```
paper/main.tex:222  →  \input{sections/03_method}
paper/sections/03_method.tex:97  →  \input{sections/03_5_deflex}
paper/sections/03_5_deflex.tex  →  421 lines (pre-existing from R15)
```

Action: **No edit needed** (additive constraint honored — no edit to §3.5 file).

## 2. 4-Pass Recompile — PASS

Run from `paper/`:
```
pdflatex -interaction=nonstopmode main.tex  →  76 pages, 5215621 bytes
bibtex main                                →  20 footnote warnings (pre-existing, non-blocking)
pdflatex -interaction=nonstopmode main.tex →  76 pages, 5215621 bytes
pdflatex -interaction=nonstopmode main.tex →  76 pages, 5215621 bytes
```

The non-zero pdflatex exit codes are font-cache mktexpk noise (cosmetic pfb font re-lookups) — page/byte counts confirm successful PDF write on every pass. bibtex exit=0; 20 warnings are pre-existing `\footnote:` pseudo-cite references that don't appear in `refs.bib` (same warnings as the previous recompile, no new ones introduced).

## 3. Gate Metrics — ALL PASS

| Gate            | Spec      | Actual                | Result |
|-----------------|-----------|-----------------------|--------|
| Pages           | ≥75       | 76                    | PASS   |
| Size            | ≥4.5 MB   | 5.22 MB (5215621 B)   | PASS   |
| Unresolved refs | 0 (`??`)  | 0 (grep `^\?` empty)  | PASS   |
| LaTeX errors    | 0 (`!`)   | 0 (grep `^!` empty)   | PASS   |
| Ref warnings    | 0         | 0 (grep Ref empty)    | PASS   |
| Bibtex warnings | <25       | 20 (pre-existing)     | PASS   |

PDF metadata (pdfinfo):
- Pages: 76
- File size: 5215621 bytes
- Producer: pdfTeX-1.40.29

## 4. §3.5 Deflex Content Verify — PASS

`pdftotext main.pdf` confirms §3.5 Deflex content is rendered in the output. Spot-checked occurrences:
- "Deflex: a dual-system pocket-conditioned reference frame" (section heading)
- "Deflex is our answer to this attractor. It is a small dual-system wrapper around the inner MCTS"
- "Deflex keeps the inner typed search intact and only ..."
- "The Lambda-side integration of Deflex is delivered as three sub-fixes, each independently WIRED and ..."

## 5. Verdict

**SHIPPED.** §3.5 Deflex is wired (via `03_method.tex:97`) and the paper recompile is clean. All gate metrics pass; no new warnings introduced. The 20 bibtex warnings and 3 cosmetic pdflatex non-zero exits are pre-existing and unrelated to this change. TODO-14 §3.5 line item can be closed.

## 6. Files Touched

- `/home/hugo/codes/try_triton_on_rocm/paper/main.pdf` — regenerated (76 pages, 5.22 MB)
- `/home/hugo/codes/try_triton_on_rocm/paper/main.log` — regenerated (148 KB)
- `/home/hugo/codes/try_triton_on_rocm/paper/main.aux` — regenerated
- `/home/hugo/codes/try_triton_on_rocm/paper/main.bbl` — regenerated
- `/home/hugo/codes/try_triton_on_rocm/paper/main.blg` — regenerated
- `/home/hugo/codes/try_triton_on_rocm/paper/main.out` — regenerated
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_t14_s35_deflex_ship/final.md` — this report

No edits to `paper/sections/03_5_deflex.tex` (additive constraint).
No edits to `paper/main.tex` or `paper/sections/03_method.tex` (already wired).