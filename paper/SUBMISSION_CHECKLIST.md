# arXiv Submission Pre-Flight Checklist

**Target**: arXiv submission window **W49 = 2026-12-09** (Wednesday freeze).
**Owner**: Hugo Lee (corresponding author).
**Companion script**: `molmetal/scripts/check_submission_ready.py` (auto-verifies every item below).
**Backlog item**: `TODO/01_strategy/IMPROVEMENT_BACKLOG.md` extension **F4** (follows F3 = `wf_full_pipeline.sh`).
**Status at open (2026-09-25)**: pre-flight pending — this document is the contract.

---

## 0. How to use this checklist

1. Run the auto-check first (it inspects the repo state and prints PASS/FAIL per item):
   ```bash
   cd /home/hugo/codes/mol-metal
   python3 molmetal/scripts/check_submission_ready.py
   ```
2. For each FAIL, edit the relevant artifact or document until the auto-check reports PASS.
3. Re-run the auto-check at every major paper edit / metrics refresh / git push.
4. The day before submission (2026-12-08), run the auto-check **and** walk through every
   unchecked box below by hand. Both must agree.
5. The submission package = `paper/main.pdf` + `paper/cover_letter.txt` + supplementary
   bundle (uploaded separately to the arXiv submission UI).

---

## 1. Required artifacts

| # | Artifact | Location | Auto-check rule |
|---|---|---|---|
| 1.1 | Compiled PDF | `paper/main.pdf` | exists, size ≥ 100 KB, mtime within 14 days |
| 1.2 | LaTeX source | `paper/main.tex` | exists, no `\TODO{...}` placeholder remains |
| 1.3 | Bibliography | `paper/refs.bib` | exists, every `\cite{...}` in `*.tex` has matching bib entry (no `??` in `main.log`) |
| 1.4 | Section sources | `paper/sections/*.tex` | every `\input{sections/...}` resolves |
| 1.5 | Appendices | `paper/appendices/*.tex` | closure_theorem + betanf_semantics + wetlab_protocol present |
| 1.6 | Supplementary PDF | `paper/supp/*.pdf` or single `paper/supplementary.pdf` | exists, ≥ 200 KB |
| 1.7 | Cover letter | `paper/cover_letter.txt` | exists, ≤ 1 page (~3000 chars), keywords line present |
| 1.8 | Traceability CSV | `paper/traceability.csv` | exists, every MEASURED/PROJECTED claim in main has a row |
| 1.9 | pdflatex clean build | `paper/main.aux` + `.log` | `main.log` contains no `LaTeX Warning: Citation '...'` on undefined refs, no `Reference '...' on page ... undefined`, exit code 0 from `pdflatex` |

**Pass criterion (1.1–1.9)**: all 9 items PASS.

### 1.a PDF build command (canonical)

```bash
cd /home/hugo/codes/mol-metal/paper
# 4-pass pdflatex for stable cross-refs + bibliography
for i in 1 2 3 4; do
    pdflatex -interaction=nonstopmode main.tex || exit $?
done
bibtex main
for i in 1 2 3 4; do
    pdflatex -interaction=nonstopmode main.tex || exit $?
done
# Final PASS condition: grep -c "LaTeX Warning: Reference .* undefined" main.log == 0
```

---

## 2. Co-author approval

| # | Item | Auto-check rule |
|---|---|---|
| 2.1 | `\author{...}` populated | `paper/main.tex` does **not** still say `\author{TODO: author list...}` |
| 2.2 | Hugo Lee listed as corresponding | `\correspondingauthor{Hugo Lee}` or equivalent email line present |
| 2.3 | All co-authors listed | acknowledgment section enumerates every co-author from `TODO/01_strategy/IMPROVEMENT_BACKLOG.md` §0 |
| 2.4 | Co-author email list exported | `paper/cover_letter.txt` mentions co-author ORCID/affiliation block |
| 2.5 | Approval audit trail | `TODO/co_author_signoffs/` (created at submission) holds one signed-off file per co-author |

**Pass criterion (2.1–2.5)**: 2.1 + 2.2 + 2.3 are auto-checked; 2.4 + 2.5 are
manual sign-off gates that block submission.

### 2.a Co-author roster (template, fill before submission)

| Name | Affiliation | Email | ORCID | Approved |
|---|---|---|---|---|
| Hugo Lee (corr.) | Mol-Metal Lab | hugo@mol-metal.example | 0000-0000-0000-0000 | ☐ |
| Co-author 2 | TBD | TBD | TBD | ☐ |
| Co-author 3 | TBD | TBD | TBD | ☐ |

---

## 3. Reproducibility

| # | Item | Auto-check rule |
|---|---|---|
| 3.1 | Full pipeline runs end-to-end | `bash molmetal/scripts/wf_full_pipeline.sh` exit code 0 in ≤ 10 min CPU wall-time |
| 3.2 | 14 evaluators PASS | `python3 molmetal/scripts/eval_all_evaluators.py` shows `14/14 PASS` |
| 3.3 | 7 pitfall guards PASS | `python3 molmetal/scripts/wf_pitfall_guard/check_all.py` shows `7/7 PASS` |
| 3.4 | Seed-0 reproducibility | `python3 molmetal/scripts/.../train_...py --seed 0` produces byte-identical ckpt on second run (sha256 match) |
| 3.5 | Random seed documented | `paper/main.tex` mentions `seed = 0` (or whichever) in §4 / §5 |
| 3.6 | Dependencies pinned | `uv.lock` (or `requirements.txt`) committed; hash in supplementary |
| 3.7 | Environment yaml | `paper/supp/environment.yml` or `paper/supplementary.tex` lists Python + CUDA + key packages |

**Pass criterion (3.1–3.7)**: 3.1, 3.2, 3.3 must PASS in auto-check. 3.4 is a
one-shot manual check. 3.5–3.7 are present and inspectable.

---

## 4. Metrics

| # | Item | Auto-check rule |
|---|---|---|
| 4.1 | 20 metric JSONs populated | `metrics/by_metric/*.json` count ≥ 20 (currently 106 entries — over-shipping) |
| 4.2 | Tier-1 baseline present | at least one of {`krasnov2026_metalcytotoxdb`, `rusanov2025_platinai`, `multimodal_protein_metal_2025`} referenced in `metrics/sota_comparisons/` |
| 4.3 | `TODO/02_metrics/METRICS_TARGETS.md` Table 2 filled | every Tier-1 metric has actual numeric value, not "TBD" |
| 4.4 | Every metric JSON has `value` + `n` | no file is a placeholder `{}` |
| 4.5 | Reproducibility JSON for each Tier-1 metric | `metrics/by_metric/<name>_seed{0,1,2}.json` exists where applicable |
| 4.6 | Honest negatives flagged | any metric with `verdict: regression` is cross-referenced in `paper/sections/06_limitations.tex` |

**Pass criterion (4.1–4.6)**: 4.1 + 4.4 auto-checked; 4.2 + 4.3 + 4.6 are
document-grade gates that must be inspected by hand.

### 4.a METRICS_TARGETS Table 2 (template)

| Metric | Direction | Our value | Tier-1 baseline | Verdict |
|---|---|---|---|---|
| M1 — Ru subset AUC | ↑ | TBD | Krasnov 2026: 0.81 | TBD |
| M2 — Pt 4-cell AUC | ↑ | TBD | Rusanov 2025: TBD | TBD |
| M14 — Vina median kcal/mol | ↓ | TBD | MolCRAFT: -9.25 | TBD |
| ... | ... | ... | ... | ... |

---

## 5. Open honest framing

| # | Item | Auto-check rule |
|---|---|---|
| 5.1 | Limitations §6 documents Tier-1 fails | `paper/sections/06_limitations.tex` length ≥ 200 lines OR explicitly says "no Tier-1 regressions" |
| 5.2 | No overclaim | `paper/main.tex` does not contain "SOTA", "state-of-the-art", "first to", "unprecedented" unless immediately qualified |
| 5.3 | Every quantitative claim tagged | `\textbf{MEASURED}` or `\textbf{PROJECTED}` appears ≥ 1× per §4–§7 result block |
| 5.4 | Future work §7 outlines R19 plan | `paper/sections/07_future.tex` exists, references R19 follow-ups (NCI-60 full 60-cell, SE(3) Schrödinger Bridge wet-lab) |
| 5.5 | Pitfall-guard summary in §6 | the 7 pitfall guards (P1–P7) are summarized in §6 with PASS/FAIL line per guard |
| 5.6 | Acknowledge code reuse | references borrowed code (RDKit / PyTorch / HuggingFace) explicitly cited |

**Pass criterion (5.1–5.6)**: 5.1, 5.2, 5.4 auto-checked; 5.3 + 5.5 + 5.6 are
human-review gates.

### 5.a Overclaim banned phrases (auto-checked)

- `SOTA` / `state-of-the-art` (without explicit qualifier like `on this benchmark`)
- `first to` / `unprecedented` / `novel` (when the contribution is incremental)
- `significantly outperforms` (without effect-size / p-value cited)
- `always converges` / `never fails` (overgeneralized)

Any match requires manual annotation in the prose justifying the wording.

---

## 6. Submission mechanics

| # | Item | Owner | Deadline |
|---|---|---|---|
| 6.1 | Choose primary category | Hugo | 2026-12-07 |
| 6.2 | Choose cross-list categories (cs.LG, q-bio.BM, physics.chem-ph) | Hugo | 2026-12-07 |
| 6.3 | Upload `paper/main.pdf` + source bundle | Hugo | 2026-12-09 12:00 UTC |
| 6.4 | Verify arXiv ID in `TODO/co_author_signoffs/notification.txt` | Hugo | 2026-12-09 18:00 UTC |
| 6.5 | Email co-authors with arXiv link | Hugo | 2026-12-09 18:00 UTC |
| 6.6 | Update `paper/main.tex` \date + add arXiv reference | Hugo | 2026-12-10 |
| 6.7 | Public mirror on GitHub `mol-metal/mol-metal` (post-embargo) | Hugo | 2026-12-12 |

---

## 7. Pre-submission walk-through (manual)

T-minus 24h: Hugo runs through these *with a coffee*, not just the auto-check:

- [ ] Re-read §1 abstract for overclaim
- [ ] Re-read §4 results, every number appears in `metrics/by_metric/`
- [ ] Re-read §6 limitations, every negative is honestly framed
- [ ] Re-read §7 future work, R19 plan is concrete (not vapor)
- [ ] Open `paper/main.pdf` at 100% zoom, skim every figure caption
- [ ] Verify all 9 cross-references to appendices resolve (`\ref{app:...}`)
- [ ] Verify supplementary figures render (open `paper/supp/supp_fig_*.png`)
- [ ] Verify cover letter under 1 page
- [ ] Verify ORCID for Hugo Lee
- [ ] Verify affiliation string matches university records

---

## 8. Post-submission checklist (T+0 to T+24h)

- [ ] arXiv assigns ID (typically within 24h)
- [ ] DOI minted (if applicable)
- [ ] Twitter / Mastodon announcement draft saved in `TODO/completed/46_arxiv_launch.md`
- [ ] Slack #mol-metal channel pinged
- [ ] GitHub `mol-metal/mol-metal` README updated with arXiv badge

---

## 9. Rollback procedure

If arXiv rejects (metadata, scope, format):

1. Note rejection reason in `TODO/co_author_signoffs/rejection_<DATE>.md`
2. Fix within 48h (format issues are usually trivial)
3. Re-submit (no fee for same-version re-submit within window)

If a major bug surfaces post-submission (e.g., metric regression):

1. Withdraw arXiv version (arXiv allows withdrawal before announcement)
2. Re-run §3 reproducibility + §4 metrics
3. Re-submit as v2 with `paper/CHANGELOG.md` note

---

## 10. Cross-references

- Companion script: `molmetal/scripts/check_submission_ready.py`
- Pipeline (F3): `molmetal/scripts/wf_full_pipeline.sh`
- Pitfall guards (P1–P7): `molmetal/scripts/wf_pitfall_guard/check_all.py`
- Evaluators (E1–E14): `molmetal/scripts/eval_all_evaluators.py`
- Metrics targets: `TODO/02_metrics/METRICS_TARGETS.md`
- Improvement backlog: `TODO/01_strategy/IMPROVEMENT_BACKLOG.md` §F4
- Limitations: `paper/sections/06_limitations.tex`
- Future work: `paper/sections/07_future.tex`

---

**Document version**: 1.0 (2026-09-25, R17 close-out).
**Maintainer**: Hugo Lee.
**Last auto-check pass**: see `TODO/co_author_signoffs/last_preflight_<DATE>.json`.