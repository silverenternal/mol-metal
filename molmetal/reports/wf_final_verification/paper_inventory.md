# Paper inventory — R18 ship

Generated: `2026-10-06T20:50:26.075754+00:00`  
Tool: `verify_paper_build.py`  
Repo root: `/home/hugo/codes/mol-metal`

## Summary

| Metric | Value |
|---|---:|
| `paper/main.tex` present | True |
| `paper/main.pdf` present | True |
| main.pdf size (bytes) | 5,833,041 |
| pages | (see main.pdf) |
| sections `\input`-ed | 7 |
| appendices `\input`-ed | 3 |
| `refs.bib` entries (@-keys) | 100 |
| `math_theorems.tex` appendix wired | False |
| all Wf-3 ship sections wired | False |

## §1-§7 Sections (R18 ship body)

Every numbered section is `\input`-ed from `paper/sections/`.

| Order | File | Notes |
|---:|---|---|
| 1 | `sections/01_intro.tex` | §1 Introduction |
| 2 | `sections/02_related.tex` | §2 Related Work |
| 3 | `sections/03_method.tex` | §3 Method (placeholder assembled by WF-Lambda-3) |
| 4 | `sections/04_evaluation.tex` | §4 Evaluation (placeholder) |
| 5 | `sections/05_ablation.tex` | §5 Ablation Studies (placeholder) |
| 6 | `sections/06_limitations.tex` | §6 Limitations (hand-written) |
| 7 | `sections/07_future.tex` | §7 Future Work |

## Appendices

All appendices are `\input`-ed from `paper/appendices/` and `paper/supplementary.tex`.

| Order | File | Notes |
|---:|---|---|
| 1 | `paper/appendices/closure_theorem.tex` | Appendix A: Closure under β-NF (placeholder, WF-Lambda-4) |
| 2 | `paper/appendices/betanf_semantics.tex` | Appendix B: Operational Semantics of β-NF (pre-existing) |
| 3 | `paper/supplementary.tex` | Supplementary material (S1-S10 + data table) |

## Wf-3 ship — four new §3.4-3.7 sections (TODO-25 Round-14)

The four math-grounded method-reference subsections ship the 9 R18 theorems + three adapters (W2.1 / W2.2 / W2.3).

| Wf-3 ship file | Wired in `main.tex` |
|---|:---:|
| `sections/03_4_se3_schrodinger_bridge.tex` | NO |
| `sections/03_5_transport_relaxation.tex` | NO |
| `sections/03_6_multi_parameter_ph.tex` | NO |
| `sections/03_7_monge_ampere_wasserstein_gf.tex` | NO |

## References

`paper/refs.bib` contains **100** `@`-keyed entries (counted by line-leading `@`; matches the bibliography rendered by `\bibliography{refs}`).

## Verification mode

* Read-only: this script does **not** invoke `pdflatex` or `bibtex`.
* Page count is **not** parsed from `main.log` (per task instruction); the `main.pdf` artefact is reported by size only.
* All `\input{...}` tokens in `paper/main.tex` are extracted via regex; the order matches document order.
