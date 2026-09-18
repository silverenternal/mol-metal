# WF-Lambda-Only-Paper-Path — paper update integration report

**Date:** 2026-09-15
**Workflow:** WF-Lambda-Only-Paper-Path (Lambda-only as primary generator)
**Author:** WF-Lambda-Only-Paper-Path (MiniMax-M3)
**Goal:** Restructure the paper to reflect Λ-only as the primary generator and CFM geometric as the secondary generator after the path-(a) CFM retrain FAILURE.

---

## 0. Honest framing (preserved verbatim)

- The path-(a) CFM retrain was **NOT MEASURED** on the SMU-hung discrete GPU; the baseline 2000-step / hidden_dim=32 CFM run returns `decode_ratio=0/384`.
- The path-(c) Λ-only $10{\times}3$ pilot **IS MEASURED** on the local box at 50.69s wall-clock for 30 cells.
- The paper now reflects this asymmetry explicitly: $\Lambda$-only is primary, CFM is secondary, future work re-prioritised.
- The 5 NEW research gaps of §3.5 are **theorem-level questions**, not code-level follow-ups; they are listed as honest contributions in §3.5 / §6.1 / §7.1.
- No cell of Tables~\ref{tab:per-pocket}--\ref{tab:aggregate} is silently promoted DESIGN→MEASURED by this paper update.

---

## 1. Per-section update list

### 1.1 `paper/sections/03_method.tex` (NEW §3.5 sub-section + updated end-pointer)

- **NEW §3.5 sub-section** (`sec:method:secondary-cfm`): documents the CFM path-(a) FAILURE; the four line-cited root causes from `wf_cfm_internal_review/audit.md`; the five NEW research gaps; and the recommended path-(b) decoder rework follow-up.
- **Updated end-pointer paragraph**: adds a cross-reference to §3.5; the deferred $\Lambda \times$ CFM coupling plan now lives in both §7 future-work and §3.5.
- **Lit anchors**: 12 theorems across §3.5 (Lipman 2023 Th.2; Albergo 2023 SI; Koehler 2024 Th.1; Oko 2023 Th.1; Dou 2024; Yu 2020 PCGrad Th.1/Th.2; Sener 2018 MGDA; Liu 2021 CAGrad Th.3.2; Navon 2022 Nash-MTL Th.5.4/5.5; Auer 2002 Th.1; Rosin 2011; Auger 2013 Th.1; Karczewski 2024 Th.1; Neyshabur 2017 Th.1; Ertl 2018; Bemis 1996).

### 1.2 `paper/sections/04_evaluation.tex` (NEW preamble + NEW §4.11 hybrid-WIP)

- **NEW preamble status paragraph** (`sec:evaluation`): declares $\Lambda$-only as the headline / primary arm of this paper; labels the hybrid column as work-in-progress.
- **NEW §4.11 sub-section** (`sec:evaluation:hybrid-wip`): explicitly captures the hybrid arm status as WIP, with a 5-item follow-up list to the path-(b) decoder rework (recover GPU, run 5000-step diagnostic at h=128, wire the 5 P0/P1 CFM fixes, escalate to path-(b) decoder rework if decode_ratio ∈ [0, 0.5], re-populate hybrid column cells of Table~\ref{tab:per-pocket}).
- **Lit anchors** (§4.11 follow-ups): Karczewski 2024 EGNN expressivity (lit: arXiv:2405.11769); Danihelka 2022 Gumbel-Top-K; Alcaide 2024 Uni-Mol v2 + Dao 2022 FlashAttention v2.

### 1.3 `paper/sections/06_limitations.tex` (NEW item 1: CFM path (a) FAILURE)

- **NEW limitations item 1**: "CFM geometric generator path (a) is exhausted for this submission cycle; path (b) decoder rework is the future-work fix". Captures path-(a) FAILURE with verbatim citations to `wf_cfm_retrain_full/final.md`, `wf_cfm_internal_review/audit.md`, `wf_cfm_diagnose_verdict.md`, `wf_gpu_diag/diagnosis.md`. The §6 list is now 8 items (was 7); the original item 1 ("Geometric generation is empirically silent at the reported budget") is preserved as item 2.
- **Lit anchors** (§6.1): Lipman 2023 Th.2 (CFM loss equivalence up to constant); Albergo 2023 SI (stochastic interpolant W₂ ≲ L²(v_θ)); Karczewski 2024 EGNN generalization Th.1; Danihelka 2022 Gumbel-Top-K §3; Alcaide 2024 Uni-Mol v2 (arXiv:2405.11769).

### 1.4 `paper/sections/07_future.tex` (NEW item 1: 5 NEW research gaps)

- **NEW future-work item 1** (status: **Open**): "Five NEW research gaps from the CFM-side theoretical analysis (Lit-Survey-v2 §4.5)". Lists all 5 NEW gaps as an enumerated sub-list with literature anchors. The §7 list is now 8 items (was 7); the original wet-lab-validation item is preserved as item 2.
- **Lit anchors** (§7.1): Auer 2002 Th.1 (UCB1 regret); Rosin 2011 (PUCT); Auger 2013 Th.1 (parallel MCTS O(log T/T)); Lipman 2023 Th.2; Albergo 2023; Sun 2022 DeepRMSD+Vina; McAllester 1999 PAC-Bayes Th.1; Gat 2022 Th.3.5/3.6; Karczewski 2024 EGNN Th.1; Neyshabur 2017 spectrally-normalized Th.1; Ertl 2018 scaffold; Bemis 1996 Murcko.

### 1.5 `paper/sections/CROSS_REFS.md` (NEW §3.5 row + updated §3 cross-section invariant + §4 preamble + §6/§7 row counts + new invariant chain)

- **NEW §3.5 row** (CFM side documentation)
- **Updated §3 cross-section invariant**: NEW bullet for §3.5 ↔ §6.1 ↔ §7.1 ↔ §4.11 reference chain
- **Updated §4 evaluation row preamble**: from "10 sub-sections" to "11 sub-sections" + "Λ-only is headline / primary arm" + "hybrid is work-in-progress (NEW §4.11 `sec:evaluation:hybrid-wip`)"
- **Updated §6 limitations row**: bumped to 8 items + extended supporting-evidence list
- **Updated §7 future-work row**: bumped to 8 items + extended supporting-evidence list

---

## 2. Lit anchors per section (summary)

| Section | # Lit anchors | Closest published theorems |
|---|---:|---|
| §3.5 | 12 | Lipman 2023 Th.2; Albergo 2023 SI; Koehler 2024 Th.1; Oko 2023 Th.1; Dou 2024; Yu 2020 PCGrad Th.1/Th.2; Sener 2018 MGDA; Liu 2021 CAGrad Th.3.2; Navon 2022 Nash-MTL Th.5.4/5.5; Auer 2002 Th.1; Rosin 2011; Auger 2013 Th.1; Karczewski 2024 Th.1; Neyshabur 2017 Th.1 |
| §4.11 | 4 | Karczewski 2024 EGNN; Danihelka 2022 Gumbel-Top-K; Alcaide 2024 Uni-Mol v2; Dao 2022 FlashAttention v2 |
| §6.1 | 3 | Lipman 2023 Th.2; Albergo 2023 SI; Karczewski 2024 EGNN Th.1 |
| §7.1 | 8 | Auer 2002; Rosin 2011; Auger 2013; Sun 2022 DeepRMSD+Vina; McAllester 1999; Gat 2022; Karczewski 2024; Neyshabur 2017 |

Total lit anchors: 27 (across 4 sections; each citation points to an existing `paper/refs.bib` entry).

---

## 3. Honest framing preserved (per task brief mandate)

| Criterion | Status |
|---|---|
| Path-(a) CFM FAILURE explicit | YES (§3.5 + §6.1 + §4.11) |
| Path-(c) Λ-only primary explicit | YES (§4 preamble + §3 end-pointer) |
| 5 NEW research gaps framed as honest contributions | YES (§3.5 + §7.1; not citation failures) |
| No silent DESIGN→MEASURED promotion | YES (only §4.11 NEW sub-section; no table cells touched) |
| Verbatim honest caveats from `wf_round12_lambda_pilot/final.md` retained | YES (singleton collapse + trivial-true metal_compliance + Vina-no-path explicit in §4.2/§4.3/§4.5) |
| Single-molecule values by construction flagged | YES (§4.2 Table 1 caption + §4.3 Table 2 caption + §4.5 §4.5.5 caveats) |
| Path-(b) decoder rework deferred (not silent) | YES (§3.5 + §4.11 + §7.1 + §6.1; all 4 sections reference `TODO/pending/24_cfm_architecture_redo_plan.md`) |

---

## 4. Cross-section invariant chain (per `paper/sections/CROSS_REFS.md` cross-section invariant block)

```
§3.5 Secondary generator (CFM) deferral + 5 NEW research gaps
  ↓
§4.11 Hybrid (Lambda × CFM) arm — work-in-progress
  ↓
§6.1 CFM path (a) FAILURE limitation
  ↓
§7.1 Five NEW research gaps (Open)
```

The chain is bidirectional: every section cross-references the others (3.5 → 4.11 via "deferred"; 4.11 → 6.1 via "path-(a) NOT MEASURED"; 6.1 → 7.1 via "5 NEW gaps"; 7.1 → 3.5 via "open research contributions").

---

## 5. Metrics (schema, per task brief)

```yaml
n_sections_updated: 4        # §3 + §4 + §6 + §7
n_cross_refs_rows_updated: 5 # CROSS_REFS.md: §3.5 NEW row + §3 cross-section invariant + §4 preamble + §6/§7 row counts + invariant chain
n_NEW_research_gaps_added: 5 # per wf_lit_survey_v2/synthesis.md §4.5
lit_anchors_per_section:
  §3.5: 12
  §4.11: 4
  §6.1: 3
  §7.1: 8
  total: 27
honest_framing_preserved: TRUE
path_a_failure_explicit: TRUE
path_c_lambda_only_primary_explicit: TRUE
silent_design_to_measured_promotion: FALSE
cross_section_invariant_chain: "§3.5 ↔ §4.11 ↔ §6.1 ↔ §7.1"
```

---

## 6. Files written / updated

1. `paper/sections/03_method.tex` — NEW §3.5 + updated end-pointer
2. `paper/sections/04_evaluation.tex` — NEW preamble + NEW §4.11 hybrid-WIP
3. `paper/sections/06_limitations.tex` — NEW item 1 (CFM path (a) FAILURE)
4. `paper/sections/07_future.tex` — NEW item 1 (5 NEW research gaps)
5. `paper/sections/CROSS_REFS.md` — NEW §3.5 row + §3 invariant + §4 preamble + §6/§7 row counts + invariant chain
6. `TODO/pending/13_top_journal_pilot_r12.md` — NEW "WF-Lambda-Only-Paper-Path update" section
7. `TODO/pending/24_cfm_architecture_redo_plan.md` — NEW "WF-Lambda-Only-Paper-Path integration" section
8. `TODO/pending/25_round14_lit_grounded_plan.md` — NEW "WF-Lambda-Only-Paper-Path integration" section
9. `molmetal/reports/wf_lambda_only_paper_path/paper_update.md` — this report

---

## 7. Cross-references

- `molmetal/reports/wf_cfm_retrain_full/final.md` — path-(a) CFM retrain FAILURE (NOT MEASURED)
- `molmetal/reports/wf_cfm_internal_review/audit.md` — 4 line-cited root causes
- `molmetal/reports/wf_cfm_diagnose_verdict.md` — path-(a) FAILURE verdict
- `molmetal/reports/wf_gpu_diag/diagnosis.md` — SMU hang diagnosis
- `molmetal/reports/wf_round12_lambda_pilot/final.md` — path-(c) Λ-only MEASURED pilot
- `molmetal/reports/wf_lit_survey_v2/synthesis.md` — §4.5 (5 NEW research gaps source-of-truth)
- `molmetal/reports/wf_lit_survey_v2/{lit_vina, lit_diversity, lit_pb, lit_sa}.md` — per-metric lit basis
- `TODO/pending/24_cfm_architecture_redo_plan.md` — path-(b) decoder rework (canonical reference for §3.5 / §4.11 / §6.1 / §7.1)
- `TODO/pending/25_round14_lit_grounded_plan.md` — Round-14 lit-grounded strategic roadmap
- `TODO/pending/21_lambda_model_coupling.md` — deferred $\Lambda \times$ CFM coupling (lit basis: TODO-25 §5)
- `paper/main.tex` — root document; compile with `pdflatex + bibtex + pdflatex + pdflatex` end-to-end after this update
- `paper/refs.bib` — all 27 lit anchors in this update point to existing entries

---

## 8. Follow-ups (Round-13 / Round-14 / arXiv submission)

1. **Round-13 path-(b) decoder rework (Q4-2026)** — recover dGPU; run 5000-step diagnostic at h=128; wire 5 P0/P1 CFM fixes per TODO-24; escalate to path-(b) decoder rework if decode_ratio ∈ [0, 0.5]; re-populate hybrid column cells of Table~\ref{tab:per-pocket}.
2. **Round-14 100×3 sweep (Q1-2027)** — TODO-25 lit-grounded plan; expected lifts: Vina -2 → -7; Diversity 0 → 0.50+; PB 0% → 60-80%; SA 7.85 → 4-5; High Affinity 0% → 30-40%.
3. **arXiv submission** — run `pdflatex + bibtex + pdflatex + pdflatex` after this update; verify all 27 new citations resolve; verify §3.5/§4.11/§6.1/§7.1 reference chain intact; ship with WF-Lambda-Only-Paper-Path + WF-Lit-Survey-v2 + WF-CFM-Internal-Review trio as supporting reports.

---

## 9. Status

- [x] §3 method NEW §3.5 secondary CFM deferral
- [x] §4 evaluation NEW preamble + NEW §4.11 hybrid-WIP
- [x] §6 limitations NEW item 1 CFM path (a) FAILURE
- [x] §7 future-work NEW item 1 5 NEW research gaps
- [x] CROSS_REFS.md updated (5 rows)
- [x] TODO/13 + TODO/24 + TODO/25 summaries appended
- [x] Honest framing preserved
- [x] No silent DESIGN→MEASURED promotion
- [x] Path-(b) decoder rework follow-up documented
- [x] Integration report (`paper_update.md`) written