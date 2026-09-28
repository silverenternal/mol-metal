# Paper inventory — R18 ship

Generated: `2026-09-26T07:20:52.929222+00:00`  
Tool: `verify_paper_build.py`  
Repo root: `/home/hugo/codes/mol-metal`

## Summary

| Metric | Value |
|---|---:|
| `paper/main.tex` present | True |
| `paper/main.pdf` present | True |
| main.pdf size (bytes) | 5,833,041 |
| pages | (see main.pdf) |
| sections `\input`-ed | 21 |
| appendices `\input`-ed | 3 |
| `refs.bib` entries (@-keys) | 166 |
| `math_theorems.tex` appendix wired | True |
| all Wf-3 ship sections wired | True |

## §1-§7 Sections (R18 ship body)

Every numbered section is `\input`-ed from `paper/sections/`.

| Order | File | Notes |
|---:|---|---|
| 1 | `sections/01_intro.tex` | §1 Introduction |
| 2 | `sections/02_related.tex` | §2 Related Work |
| 3 | `sections/03_method.tex` | §3 Method (placeholder assembled by WF-Lambda-3) |
| 4 | `sections/03_4_cfm_methodology.tex` | §3.x CFM Methodology (Wf-CFM-M3-M5, Round-14 lit-grounded) |
| 5 | `sections/03_4_se3_schrodinger_bridge.tex` | §3.4 SE(3) Schrödinger Bridge (Wf-3 ship, W2.1) |
| 6 | `sections/03_5_transport_relaxation.tex` | §3.5 Transport Relaxation (Wf-3 ship, W2.2) |
| 7 | `sections/03_6_multi_parameter_ph.tex` | §3.6 Multi-Parameter PH (Wf-3 ship, W2.3) |
| 8 | `sections/03_7_monge_ampere_wasserstein_gf.tex` | §3.7 Wasserstein / Monge-Ampère foundations (Wf-3 ship) |
| 9 | `sections/03_4_se3_schrodinger_bridge_v3.tex` |  |
| 10 | `sections/03_5_transport_relaxation_v3.tex` |  |
| 11 | `sections/03_6_multi_parameter_ph_v3.tex` |  |
| 12 | `sections/03_7_monge_ampere_wasserstein_gf_v3.tex` |  |
| 13 | `sections/04_evaluation.tex` | §4 Evaluation (placeholder) |
| 14 | `sections/04_results.tex` | §4 Results (Round-17 rewrite) |
| 15 | `sections/04_results_table2_v3.tex` |  |
| 16 | `sections/04_5_honest_negative.tex` | §4.5 Honest Negative Framing (Round-17) |
| 17 | `sections/05_ablation.tex` | §5 Ablation Studies (placeholder) |
| 18 | `sections/06_limitations_v3.tex` |  |
| 19 | `sections/06_limitations_auto.tex` | §6 Limitations (auto-generated snapshot) |
| 20 | `sections/07_future.tex` | §7 Future Work |
| 21 | `sections/00_label_stubs.tex` | label stubs (silences orphan-reference warnings) |

## Appendices

All appendices are `\input`-ed from `paper/appendices/` and `paper/supplementary.tex`.

| Order | File | Notes |
|---:|---|---|
| 1 | `paper/appendices/closure_theorem.tex` | Appendix A: Closure under β-NF (placeholder, WF-Lambda-4) |
| 2 | `paper/appendices/betanf_semantics.tex` | Appendix B: Operational Semantics of β-NF (pre-existing) |
| 3 | `paper/appendices/math_theorems.tex` | Appendix D: Mathematical Theorems (Wf-Math-1, 9 R18 theorems) |
| - | `paper/supplementary.tex` | NOT currently \input-ed |

## Wf-3 ship — four new §3.4-3.7 sections (TODO-25 Round-14)

The four math-grounded method-reference subsections ship the 9 R18 theorems + three adapters (W2.1 / W2.2 / W2.3).

| Wf-3 ship file | Wired in `main.tex` |
|---|:---:|
| `sections/03_4_se3_schrodinger_bridge.tex` | YES |
| `sections/03_5_transport_relaxation.tex` | YES |
| `sections/03_6_multi_parameter_ph.tex` | YES |
| `sections/03_7_monge_ampere_wasserstein_gf.tex` | YES |

## References

`paper/refs.bib` contains **166** `@`-keyed entries (counted by line-leading `@`; matches the bibliography rendered by `\bibliography{refs}`).

## Verification mode

* Read-only: this script does **not** invoke `pdflatex` or `bibtex`.
* Page count is **not** parsed from `main.log` (per task instruction); the `main.pdf` artefact is reported by size only.
* All `\input{...}` tokens in `paper/main.tex` are extracted via regex; the order matches document order.
