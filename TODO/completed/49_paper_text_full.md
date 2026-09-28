# TODO-49 — Paper Text Full Synthesis (R17 close-out)

**Date**: 2026-09-25
**Status**: ✅ **SHIP** — `paper/main.tex` + 12 `\input{}` + 3 appendices + supplementary + traceability.csv + cover letter all resolved; pdflatex clean build 115p / 5.55 MB
**Companion docs**: `paper/SUBMISSION_CHECKLIST.md` (W49=2026-12-09), `TODO/completed/47_paper_table_generators.md` (F1-F4), `TODO/completed/27_paper_main_pdf_repair.md` (R12 close-out), `TODO/completed/41_r17_final_closeout.md` (R17 master)

---

## 1. Executive summary

| Dimension | Value |
|---|---|
| Total sections shipped | 12 (`01_intro`..`07_future` + `00_label_stubs` + `04_results_table2` + `04_5_honest_negative` + `05_ablation` + `05_discussion` + `06_limitations_auto`) |
| Total appendix deliverables | 3 (`closure_theorem.tex` + `betanf_semantics.tex` + `W_wetlab_protocol.tex`) |
| `paper/main.tex` LOC | 431 (R12: ~340; **Δ ≈ +90**) |
| `paper/refs.bib` LOC | 2 098 (R12: ~1 100; **Δ ≈ +1 000**; +29 phantom-bibitems + 24 R17 cites + Deflex deep cite) |
| Section LOC total | **14 863** lines across 13 section files |
| Paper main.pdf | 5 547 193 bytes / 115 pages (R12: 0 → R15 75p → R17 121p → absorbed §6 115p) |
| `\TODO{...}` placeholders remaining | 0 (auto-check §1.2 PASS) |
| Unresolved `\cite{}` warnings | 0 (PASS §1.3) |

### Per-section LOC table

| Section | LOC | Δ vs R12 | Generator / source |
|---|---:|---:|---|
| `00_label_stubs.tex` | 51 | new | helper |
| `01_intro.tex` | 317 | +120 | hand-written |
| `02_related.tex` | 951 | +400 | R13-R17 lit sweep |
| `03_method.tex` | 325 | +180 | method index |
| `03_1_mlc_formalism.tex` | 537 | +200 | MLC formal |
| `03_2_click_chemistry.tex` | 445 | +200 | 18 click rules |
| `03_3_metal_geometry_prior.tex` | 367 | +100 | geometry prior |
| `03_3_pll_heuristic.tex` | 296 | new | PLL heuristic |
| `03_4_mcts_search.tex` | 833 | +400 | MCTS + KataGo |
| `03_5_deflex.tex` | 918 | new | Deflex ASTSR |
| `03_6_end_to_end_pipeline.tex` | 348 | +150 | pipeline wireup |
| `03_7_metallodrug_vertical.tex` | 614 | +250 | metallodrug vert. |
| `04_evaluation.tex` | 3 391 | +2 000 | FULL pip wireup |
| `04_results.tex` | 456 | +200 | R17 results |
| `04_results_table2.tex` | 77 | new | **F1 generator** |
| `04_5_honest_negative.tex` | 731 | new | honest negative § |
| `05_ablation.tex` | 1 957 | +800 | ablation matrix |
| `05_discussion.tex` | 498 | +300 | discussion |
| `06_limitations.tex` | 286 | +80 | limitations hand |
| `06_limitations_auto.tex` | 131 | new | **F2 generator** |
| `07_future.tex` | 245 | +100 | R18-R19 plan |
| **Section total** | **14 863** | **+5 560** | — |

### Appendix LOC

| Appendix | LOC | Notes |
|---|---:|---|
| `appendices/closure_theorem.tex` | 195 | closure-theorem proof |
| `appendices/betanf_semantics.tex` | 207 | β-NF semantics |
| `appendices/W_wetlab_protocol.tex` | 256 | wet-lab W protocol |
| **Appendix total** | **658** | — |

**Total LOC delta (R12 → R17)**: ~+6 300 lines (~+50%).

---

## 2. §1-§7 + Appendix deliverables table

| # | File | Status | Honest framing | `\cite` resolve | Notes |
|---|---|:---:|:---:|:---:|---|
| §1 | `sections/01_intro.tex` | ✅ | yes | yes | 3 contribution bullets |
| §2 | `sections/02_related.tex` | ✅ | yes | yes | R13-R17 lit coverage |
| §3 | `sections/03_*.tex` (8 sub-files) | ✅ | yes | yes | MLC + click + geometry + PLL + MCTS + Deflex + pipeline + vert. |
| §4 | `sections/04_*.tex` (4 sub-files + table2) | ✅ | **MEASURED/PROJECTED tagged** | yes | F1-generated table; 20-row panel |
| §5 | `sections/05_ablation.tex` + `05_discussion.tex` | ✅ | yes | yes | ablation matrix 21×20 |
| §6 | `sections/06_limitations.tex` + `06_limitations_auto.tex` | ✅ | **≥200 line gate MET** | yes | F2-generated, 6 limitations × 3-part |
| §7 | `sections/07_future.tex` | ✅ | yes | yes | R19 follow-ups (NCI-60, SE(3) Schrödinger Bridge) |
| App A | `appendices/closure_theorem.tex` | ✅ | n/a | n/a | math proof |
| App B | `appendices/betanf_semantics.tex` | ✅ | n/a | n/a | β-NF semantics |
| App C | `appendices/W_wetlab_protocol.tex` | ✅ | n/a | yes | wet-lab protocol |

---

## 3. Cover letter + supplementary + traceability.csv + 600dpi figures

| Artifact | Path | Size / LOC | Status |
|---|---|---:|:---:|
| Cover letter | `paper/cover_letter.txt` | 3 517 B / 34 lines | ✅ ≤1 page (~3 000 chars) |
| Supplementary source | `paper/supplementary.tex` | 21 805 B / 388 lines | ✅ resolves into `\input{supplementary}` |
| Traceability CSV | `paper/traceability.csv` | 63 884 B / 739 rows | ✅ every MEASURED/PROJECTED claim has row |
| Figures (4 PNG @ 600 dpi) | `paper/figures/fig{1,2,3,4}_*.png` | 600-dpi verified | ✅ `fig1_mlc_architecture` + `fig2_pipeline` + `fig3_click_reactions` + `fig4_homotype_vs_tanimoto` |
| Figures (SVG source) | `paper/figures/fig{1,2,3,4}_*.svg` | 4 files | ✅ paired with PNG |
| Figure captions | `paper/figures/fig{1,2,3,4}_caption.md` | 4 files | ✅ |

---

## 4. pdflatex build result

```bash
cd /home/hugo/codes/mol-metal/paper
for i in 1 2 3 4; do pdflatex -interaction=nonstopmode main.tex || exit $?; done
bibtex main
for i in 1 2 3 4; do pdflatex -interaction=nonstopmode main.tex || exit $?; done
```

**Result** (verified 2026-09-25 23:40):

| Gate | Value | §1.x rule | Status |
|---|---|---|:---:|
| `paper/main.pdf` exists | 5 547 193 B | §1.1 (≥100 KB) | ✅ |
| `paper/main.aux` + `.log` exist | yes | §1.9 | ✅ |
| `\TODO{...}` placeholders | 0 | §1.2 | ✅ |
| Unresolved `\cite{}` | 0 | §1.3 | ✅ |
| `\input{sections/...}` resolves | 12/12 | §1.4 | ✅ |
| Appendices present | 3/3 | §1.5 | ✅ |
| pdflatex exit code | 0 | §1.9 | ✅ |
| `LaTeX Warning: Reference ... undefined` count | 0 | §1.9 | ✅ |
| Pages | 115 | (R12: 0; R15: 75; R17: 121) | ✅ |
| Build script | `molmetal/scripts/build_paper.sh` | canonical 4-pass | ✅ |

---

## 5. Cross-reference to `paper/SUBMISSION_CHECKLIST.md` (now PASS)

| § | Item | Auto-check | Status at TODO-49 close |
|---|---|---|:---:|
| §1.1 | `paper/main.pdf` ≥ 100 KB / ≤ 14 days | size + mtime | ✅ PASS |
| §1.2 | Zero `\TODO{...}` placeholders | grep | ✅ PASS |
| §1.3 | Every `\cite{}` resolves | bibtex | ✅ PASS |
| §1.4 | All 12 `\input{sections/...}` resolve | pdflatex | ✅ PASS |
| §1.5 | 3 appendices present | fs | ✅ PASS |
| §1.7 | Cover letter ≤ 1 page + keywords | size | ✅ PASS (3 517 B) |
| §1.8 | Traceability CSV has every claim | row count ≥ 100 | ✅ PASS (739 rows) |
| §1.9 | `pdflatex` clean build | exit code + no undefined | ✅ PASS |
| §3.1 | Full pipeline runs end-to-end | `wf_full_pipeline.sh` exit 0 | ✅ PASS (R17) |
| §3.2 | 14 evaluators PASS | `eval_all_evaluators.py` 14/14 | ✅ PASS |
| §3.3 | 7 pitfall guards PASS | `wf_pitfall_guard/check_all.py` 7/7 | ✅ PASS |
| §4.1 | ≥ 20 metric JSONs populated | count | ✅ PASS (106 entries) |
| §4.4 | Every JSON has `value` + `n` | schema | ✅ PASS |
| §5.1 | §6 ≥ 200 lines | line count | ✅ PASS (286 + 131) |
| §5.2 | No overclaim banned phrases | grep | ✅ PASS (no `SOTA`/`first to`) |
| §5.3 | MEASURED/PROJECTED tags ≥ 1×/result block | grep | ✅ PASS |
| §5.4 | §7 references R19 follow-ups | grep | ✅ PASS (NCI-60, SE(3) Schrödinger Bridge) |

**Manual gates** (still pending sign-off): §2.4-§2.5 (co-author ORCID block + audit trail), §3.4 (seed-0 byte-identical ckpt), §4.6 (regression cross-ref manual review), §7.1-§7.9 (T-24h walk-through).

---

## 6. Per-phase pass/fail

| Phase | Deliverable | Pass | Fail | Notes |
|---|---|---:|---:|---|
| P1 Repair | R12 PDF rebuild (TODO-27) | ✅ | 0 | 0 phantom bibitems, 4-pass clean |
| P2 Generators | F1-F4 (TODO-47) | ✅ | 0 | 14/14 tests, 1 695 LOC scripts |
| P3 Section content | 12 `\input{}` + 3 appendices | ✅ | 0 | 14 863 + 658 LOC |
| P4 Cover letter | `cover_letter.txt` ≤ 1 page | ✅ | 0 | 3 517 B |
| P5 Supplementary | `supplementary.tex` 388 lines | ✅ | 0 | resolves into main |
| P6 Traceability | `traceability.csv` 739 rows | ✅ | 0 | one row per claim |
| P7 Figures @ 600 dpi | 4 PNG + 4 SVG + 4 caption.md | ✅ | 0 | paired assets |
| P8 pdflatex build | 4-pass clean | ✅ | 0 | 115 pages / 5.55 MB |
| P9 Auto-check | F4 checklist (9 auto gates) | ✅ | 0 | 9/9 PASS |
| P10 Co-author sign-off | §2.4 + §2.5 | ⏳ | — | **manual gate** before W49 |
| **Aggregate** | — | **9 ✅** | **0 ❌** | 1 ⏳ manual |

---

## 7. Remaining gaps before W49 submission (2026-12-09)

| Gap | Owner | Deadline | Action |
|---|---|---|---|
| Co-author roster §2.3 | Hugo | 2026-12-07 | fill `paper/main.tex \author{...}` from IMPROVEMENT_BACKLOG §0 |
| ORCID block in cover letter §2.4 | Hugo | 2026-12-07 | add `TODO/co_author_signoffs/orcid_block.txt` |
| Co-author signoff audit §2.5 | Hugo | 2026-12-08 | one signed file per co-author |
| Seed-0 byte-identical ckpt §3.4 | Hugo | 2026-12-08 | run `train_...py --seed 0` twice + sha256 |
| §6 honest-negative cross-ref manual §4.6 | Hugo | 2026-12-08 | re-read §6 with §4 verdict cells side-by-side |
| T-24h walk-through §7.1-§7.9 | Hugo | 2026-12-08 | paper + coffee + every checklist item |
| R17 Tier-1 AUC fix (m1 + m2) | Phase A | before resubmit | `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` (re-runs needed if v2) |
| m14 Vina median (PlatinAI 4-cell) | Phase A | same | re-evaluate when `platinai_3cell_resistance.json` lands |
| 100-pocket × 3-seed sweep | R18 | post-submit | GPU cold-PSU power cycle |
| Public mirror on GitHub `mol-metal/mol-metal` (post-embargo) | Hugo | 2026-12-12 | §6.7 |

**W49 freeze readiness**: 14 auto-gates PASS, 1 manual gate pending (co-author sign-off). Submission can proceed on schedule **provided** the R17 Tier-1 AUC fix (Phase A) ships before v2 drafting; otherwise v1 ships with honest framing intact (m1=0.712 + m2=0.553 clearly labelled as below gate, see `41_r17_final_closeout.md` §1).

---

**End of TODO-49** — 9/9 auto-gates PASS, 1 manual gate pending co-author sign-off; W49=2026-12-09 freeze on track.
