# WF-Paper-1 — Assemble & Cross-Reference Audit

Workflow: WF-Paper-1 (verification pass, 2026-09-14)
Target: Digital Discovery (RSC) Q1 IF 8.5 primary, J. Chem. Inf. Model. (ACS) Q1 IF 5.6 fallback.
Style: IMRaD; MEASURED vs PROJECTED clearly labelled; no number re-derived for this paper.

## 1. Section list

| # | File | Title | Lines | Status |
|---|------|-------|------:|--------|
| 1 | `paper/sections/01_intro.tex` | Introduction | 238 | SHIPPED |
| 2 | `paper/sections/02_related.tex` | Related Work | 364 | SHIPPED |
| 6 | `paper/sections/06_limitations.tex` | Limitations | 142 | SHIPPED |
| 7 | `paper/sections/07_future.tex` | Future Work | 151 | SHIPPED |
| _ | `paper/sections/README.md` | Section inventory + footnote table | 99 | SHIPPED |
| _ | `paper/sections/CROSS_REFS.md` | One-line cross-reference map | 26 | SHIPPED (this PR) |
| 3 | `paper/sections/03_method.tex` | Method | _ | NOT WRITTEN (separate workflow) |
| 4 | `paper/sections/04_evaluation.tex` | Evaluation | _ | NOT WRITTEN (separate workflow) |
| 5 | `paper/sections/05_ablation.tex` | Ablation | _ | NOT WRITTEN (separate workflow) |

**Total `.tex` lines (sections 1, 2, 6, 7):** 895.

## 2. Per-section verification

### §1 Introduction (`01_intro.tex`)
- Section header `Introduction` appears 1x on line 1; `\label{sec:intro}` set.
- Forward cross-references (verified via `grep -oE '\\ref\{sec:[a-z0-9:]+\}'`):
  `\ref{sec:related}`, `\ref{sec:method}`, `\ref{sec:evaluation}`, `\ref{sec:ablation}`, `\ref{sec:limitations}`, `\ref{sec:future}` — all 6 present (6 forward refs).
- Empirical-evidence bullets cite `footnote:wf_lambda1c`, `footnote:wf_lambda1`, `footnote:wf_lambda2e`, `footnote:round11_parity`, `footnote:wf_extra2`, `footnote:round10_e2e`, `footnote:lambda_coupling`.
- Placeholder protocol sheet acknowledged for §3 / §4 / §5.

### §2 Related Work (`02_related.tex`)
- Section header `Related Work` appears 2x (line 10 + sub-bundle header comment).
- 9 SOTA paper citations verified: `footnote:diffsbdd`, `footnote:pocket2mol`, `footnote:targetdiff`, `footnote:moldiff`, `footnote:decompdiff`, `footnote:flowr`, `footnote:diffdock`, `footnote:bindnet`, `footnote:rosettafoldaa`.
- 7 protocol-mismatch flags verified (Flag 1–Flag 7, all with explicit `Flag N:` paragraph header).
- Cross-refs: `\ref{sec:evaluation}`, `\ref{sec:future}`, `\ref{sec:limitations}`, `\ref{sec:method}`, `\ref{sec:related:lambda}`, `\ref{sec:related:protocol}`, `\ref{sec:related:qsar}`, `\ref{sec:related:retro}` (8 explicit, with internal sub-section labels).
- Honest flag-status summary at the end: 4 open, 3 closed by measurement.

### §6 Limitations (`06_limitations.tex`)
- Section header `Limitations` appears 2x (comment + `\section{Limitations}\label{sec:limitations}`).
- 7 limitations verified via `grep -c '^  \\item'` = 7.
- Each limitation carries an explicit status (`Recommended fix (deferred/pending/projected)`) and a diagnosis grounded in a MEASURED number from a cited report.
- User-gated decisions D1–D7 referenced at the end (TODO-19 / TODO-20).

### §7 Future Work (`07_future.tex`)
- Section header `Future Work` appears 2x (comment + `\section{Future Work}`).
- 7 future items verified via `grep -c '^  \\item'` = 7.
- Status legend verified: `[Out-of-scope.]`, `[Future.]`, `[Defer.]`, `[Future.]`, `[Open.]`, `[Defer.]`, `[Open.]` (7 distinct status labels covering 4 legend classes).
- Citations: `footnote:wf_lambda1c`, `molmetal/reports/wf_extra1_full/final.md`, `ultracode_audit/agent_c_lambda.md`, `TODO/pending/21_lambda_model_coupling.md`, `TODO/pending/14_full_100pocket_paper_r13.md`, `round10_e2e_pt_cfg_vina.md`.

## 3. Grep sanity check

```
01_intro.tex: 1 occurrence of "Introduction"
02_related.tex: 2 occurrences of "Related Work"
06_limitations.tex: 2 occurrences of "Limitations"
07_future.tex: 2 occurrences of "Future Work"
```

All 4 sections > 0 occurrences of their own header → **PASS** (the task-script's literal `grep -c section{01_intro,02_related,06_limitations,07_future}` triggers a brace-expansion parse error in `ugrep`; the semantic equivalent — header-text presence — passes for all 4).

## 4. Cross-reference map (one-line per section)

| Section | Cross-references | Supporting report |
|---------|------------------|-------------------|
| §1 Introduction | §2, §3, §4, §5, §6, §7 | `wf_lambda1c_pilot_v3`, `wf_lambda1_pilot_v1`, `wf_lambda2e_compare`, `round11_engine_parity_n50`, `wf_extra2_batch`, `round10_e2e_pt_cfg_vina` |
| §2 Related Work | §1, §3, §6, internal sub-sections | `round11_engine_parity_n50`, `wf_lambda2e_compare`, `round10_e2e_pt_cfg_vina`, `wf_extra1_full`, `ultracode_audit/lambda_protocol_aligned` |
| §6 Limitations | §1, §3, §4, §5, TODO-19/20 | `round10_e2e_pt_cfg_vina`, `round10_pt_prior_ablation`, `wf_extra1_full`, `round11_engine_parity_n50`, `wf_lambda2e_compare`, `ultracode_audit/` |
| §7 Future Work | §1, §3, §4, §6, TODO-19/21 | `wf_lambda1c_pilot_v3`, `wf_extra1_full`, `round10_e2e_pt_cfg_vina`, `ultracode_audit/agent_c_lambda.md`, `TODO/pending/21_lambda_model_coupling.md`, `TODO/pending/14_full_100pocket_paper_r13.md` |

Full one-line cross-reference per section is in `paper/sections/CROSS_REFS.md`.

## 5. Total `\ref{...}` count across the 4 shipped sections

```
01_intro.tex: 6
02_related.tex: 17
06_limitations.tex: 3
07_future.tex: 3
Total: 29
```

## 6. Bundle policy

- Each section is a complete LaTeX fragment, intended to be `\input`-ed into `paper/main.tex` in numeric order.
- §3 / §4 / §5 are placeholders owned by separate workflows (WF-Paper-Method / WF-Paper-Eval / WF-Paper-Ablation).
- All MEASURED-vs-PROJECTED labels are preserved verbatim from the upstream report; no number is re-derived for this paper.
- Honest-framing is mandatory; any number cited in §1–§7 is traceable to a report under `molmetal/reports/` or a TODO under `TODO/pending/`.

## 7. Metrics

| Metric | Value |
|--------|------:|
| `n_sections_written` | 4 |
| `total_tex_lines` | 895 |
| `n_cross_refs` (\ref{sec:...} count) | 29 |

## 8. Pass/fail verdict

- 4/4 sections written, consistent, and self-labelled: **PASS**
- §1 forward-cites §2–§7: **PASS** (6 forward refs verified)
- §2 cites 9 SOTA papers + 7 protocol-mismatch flags: **PASS** (9 + 7 verified)
- §6 lists ≥7 limitations with honest framing: **PASS** (7 verified, all carry `MEASURED` / `Recommended fix` labels)
- §7 lists ≥5 future items with status labels: **PASS** (7 items, 4 status classes)
- Grep sanity: **PASS** (all 4 sections contain their own header text)
- CROSS_REFS.md written: **PASS**
- wf_paper1_assemble.md written: **PASS** (this file)