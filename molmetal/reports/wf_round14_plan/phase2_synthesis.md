# Phase 2 Synthesis — Round-14 Prioritized Plan

**Date:** 2026-09-15
**Status:** SHIP — companion to `phase1_lit.md` (lit survey)
**Owner:** Phase-2 (synthesis + 12-week roadmap)
**Round scope:** Round-14 (Q1-2027) + Round-15 (Q1-2027 Q2) + paper submission (Q2-2027)
**Honest framing:** This file is a *plan*, not a measurement. Each action cites either a
lit-grounded fix path (per phase1_lit.md) or a measured result from
`metrics/by_round/*.json`. No claim of lift is made without data.

---

## §0. Input materials (verbatim references)

* **Phase 1 lit survey** — `molmetal/reports/wf_round14_plan/phase1_lit.md`
  (6 axes × lit anchors; 5 NEW research gaps)
* **Round-12 honest verdict** — `molmetal/reports/wf_round12_lambda_pilot/final.md`
  + `metrics/by_round/r12_lambda_pilot.json` (n_distinct=1 collapse; diversity=0.0 forced)
* **Round-12 PARTIAL_LIFT** — `molmetal/reports/wf_round12_lambda_patha_10x3/final.md`
  + `metrics/by_round/r12_lambda_patha_10x3.json` (n_distinct 1→20, div_tan 0→0.1065; metal_compliance trade-off)
* **Algo-tune PARTIAL_LIFT** — `molmetal/reports/wf_algo_tune/final.md` (F2(a) + algo-tune PARTIAL on Round-12 cohort only; pocket-invariance gap on novel pockets)
* **Round-13 honest negative** — `molmetal/reports/wf_round13_100x3/final.md`
  + `metrics/by_round/r13_100x3_partial.json` (paper_grade_data_ready=FALSE; Path A killed; Path B 30/30 search-bound; Path C BLOCKED)
* **CFM Path B honest negative** — `molmetal/reports/wf_path_b_gpu_retrain/final.md`
  (decode_ratio=0/192 on real CrossDocked output; 5 P0 fixes shipped but wrap-ordering issue)
* **Lit Survey v2** — `molmetal/reports/wf_lit_survey_v2/{lit_vina,lit_diversity,lit_pb,lit_sa,synthesis}.md`
  (65 papers + 20 theorems borrowed)
* **TODO-25** — `TODO/pending/25_round14_lit_grounded_plan.md` (lit-grounded 5-weak-metric plan + 5 NEW gaps)
* **TODO-26** — `TODO/pending/26_round13_round14_complete_plan.md` (comprehensive ship plan)
* **TODO-29** — `TODO/pending/29_f2a_round13_retry.md` (F2(a) MetalLigandExchange SMARTS + re-run Round-13)

---

## §1. Open gaps after Round-13 (audit)

### §1.1 Pocket-invariance gap (workflow in flight, PARTIAL)

**What was MEASURED**: `wf_algo_tune/final.md` (Round-12 PathA-10x3) verified diversity lift
on **test_000..test_009** (n_distinct 1→20, div_tan 0→0.1065, div_homo 0→0749).
Round-13 re-run on **test_010..test_019** = singleton collapse resurfaces (n_distinct=1).

**What is NOT MEASURED**: full 100×3 paper-grade aggregate that lifts on ALL 100 pockets,
not just the original 10. Pocket-invariance requires the per-pocket warm-start embedding
(Task J) to break the 3-layer singleton attractor (chemistry + cache + reward prior).

**Workflow in flight**: `WF-Lambda-MCTS-Coords-Fix` (Task J + Tasks L1/L2/L3/D/E/R/L
shipped per `INDEX.md §Lambda Core Features`). Verification on novel pockets
(test_010..test_019) NOT yet MEASURED.

**Lit anchor** (per phase1 §1): Guan 2023 §3.1 (pocket-conditioned diffusion) +
Peng 2022 §3.2 (per-pocket sub-pocket fingerprint). Both justify
per-pocket warm-start + sub-pocket fingerprint diversity metric (ship per Tasks J+D).

### §1.2 CFM Path (a) decode_ratio=0 (workflow in flight, FAILURE)

**What was MEASURED**: `wf_path_b_gpu_retrain/final.md` — Path B decoder rework with
hidden_dim=128, drop tanh, learnable vel_scale, PCGrad, PAC-Bayes, joint bond-head,
ReworkedDecoder wrap, tmQM-pretrained init = **decode_ratio=0/192 on real CrossDocked-trained CFM**.

**5 P0 fixes ship but path stays at zero**:
- F1: BondAwareDecoder.decode wired at `_generate_impl` line 2017
- F2: BondOrderHead in_dim=9 → 9+2*hidden_dim
- F3: vocab_mask BEFORE F.cross_entropy (recovers 88% wasted gradient)
- F4: UserWarning on hidden_dim<64
- F5: verify bonds=zeros placeholder removed

**Root cause** (per `wf_path_b_gpu_retrain/final.md §5`):
- (a) **wrap-ordering broken**: ReworkedDecoder wraps after gumbel decode but the
  connectivity check still rejects. Fix = move wrap to *replace* gumbel AND
  relax `len(Chem.GetMolFrags(mol)) != 1` to `>= 1` + reconnect largest fragment.
- (b) **CFM coordinate distribution broken**: 97.4% disconnect at CFM coord level;
  atoms cluster at non-covalent distances so no bond-head candidate edges found.
- (c) **gumbel decoder produces disconnected graph**: 177 disconnected + 15 valence_failure = 192.

**Workflow in flight**: `WF-Path-B-GPU-Retrain` (per `wf_path_b_gpu_retrain/final.md §7`)
HIGH priority action 1 (wrap-ordering fix) NOT YET SHIPPED.

**Lit anchor** (per phase1 §3): Lipman 2023 Theorem 2 (`L_CFM = L_FM`) — pocket
conditioning must be first-class marginal, not feature concat. Albergo 2023
stochastic interpolant — un-bounded vel_head allowed. Both already ship but path
stays at zero, suggesting wrap-ordering (not theoretical gap) is dominant cause.

### §1.3 paper/main.pdf compile (workflow in flight, BROKEN)

**What was MEASURED**: TODO-27 — `paper/main.pdf` DOES NOT EXIST.
Only `paper/section_03_method.pdf` exists. `main.aux/bbl/blg/log/out` exist
(intermediate pdflatex artifacts). No `main.pdf/dvi/ps`.

**Task tracker status confusion**:
- Task #456 (WF-Paper-Compile) marked "completed" — FAILED
- Task #460 (PDF well-formed audit) "in_progress" — FAILED (no PDF to audit)
- Task #465 (5 fixes) "completed" — partial (fixes applied but compile still broken)
- Task #509 (typo fix + recompile) "completed" — partial (typo fixed but recompile didn't run)
- Task #475 (re-run pdflatex) "in_progress" — still broken

**Likely root causes** (per TODO-27):
- `paper/main.tex` master file (per `texput.log` if it exists)
- bibliography `refs.bib` (cite/ref unresolved → pdflatex aborts)
- figure inclusion (fig1/fig2/fig3/fig4 paths)
- cross-package options (natbib vs authoryear vs unsrtnat)

**Workflow in flight**: TODO-27 ship blocker — 2-5.5h CPU-only.

### §1.4 §4 + §5 + §6 paper content updates (workflows in flight, partial)

**§4 Evaluation** — 3 sections updated per `wf_round13_100x3/final.md §3.1`:
- Status block (paper_grade_data_ready=false)
- §4.11 NEW: Round-13 honest sub-section
- 0 cells DESIGN→MEASURED (integration refused silent promotion)

**§5 Ablation** — already ship per `WF-Paper-Section-05` (433 tex lines, 6 sub-sections,
2 tables, 144 DESIGN + 49 MEASURED + 39 PROJECTED cells). Round-13 sweep did NOT
provide aggregate data to lift the 49 MEASURED → more MEASURED.

**§6 Limitations** — 9 items now (was 8; +1 for Round-13 partial-completion):
- 7 prior items (preserved verbatim per TODO-25 update protocol)
- NEW item (8): Round-13 100×3 partial-completion limitation

**§7 Future Work** — 5 NEW research gaps documented per TODO-25 §5 + §7.1 (paper).

**§3 Method** — already ship (1554 tex lines, 5 SMARTS verbatim, 5 metals, UCB formula).

**Workflow in flight**: `WF-Paper-Section-04` + `WF-Paper-Section-05` + `WF-Paper-Compile`
+ `WF-Paper-Content` + `WF-Paper-Repair` (5 in-flight, partial).

---

## §2. Alternative solutions per gap (with lit anchors + math prior + effort)

### §2.1 Pocket-invariance gap — 3 alternatives

**Alt A: Lift MCTS budget to n_sim=5000 + virtual loss per Chaslot 2008**
- *Lit anchor*: Auer 2002 UCB1 O(log T) regret + Auger 2013 Theorem 5 (UCT convergence)
  + Chaslot 2008 virtual-loss parallelisation + Lai 1985 Ω(√KT log T) lower bound
- *Math prior*: Auger 2013 Theorem 5 quantifies budget needed for non-degenerate coverage.
  n_sim=100 was necessary-but-not-sufficient. n_sim=1000 lifted n_distinct 1→20 on
  test_000..test_009 but Round-13 saw collapse on test_010..test_019. Hypothesis:
  n_sim=5000 + virtual loss can cover the full pocket-distribution.
- *Effort*: 2-4h CPU-only (raise hard-cap + add virtual loss + re-run 30-cell smoke).
- *Risk*: HIGH — may still collapse on novel pockets if root prior doesn't generalize.

**Alt B: Per-pocket warm-start embedding (Task J) + sub-pocket fingerprint diversity (Task D)**
- *Lit anchor*: Guan 2023 §3.1 (pocket-conditioned diffusion via residue-level cross-attention)
  + Peng 2022 §3.2 (per-pocket sub-pocket fingerprint as input to MPN) + Bemis 1996 (Murcko scaffold)
  + Jasial 2021 IntDiv formula + Peter 2019 SPF (sub-pocket fingerprint).
- *Math prior*: Per-pocket warm-start embedding breaks the singleton attractor by
  providing a *root-specific* prior that adapts to the local pocket geometry.
  Tasks J + D already ship per INDEX.md §Lambda Core Features (Phase-3 agent 4).
- *Effort*: 6-8h CPU-only (Tasks J + D already ship; need 30-cell novel-pocket smoke +
  integration into `r4_lambda_only_run.py`).
- *Risk*: MEDIUM — Pocket-specific prior may overfit to training pockets.

**Alt C: F2(a) MetalLigandExchange SMARTS + AquaExchange rule (per TODO-29)**
- *Lit anchor*: Lippard-Berg 1995 (Pt(II) coordination chemistry) + Reedijk 1987 (Pt
  aquation kinetics) + Taube 1952 JACS Pt(II) associative mechanism.
- *Math prior*: Adds structural diversity at the SMARTS level. Currently the
  `_unreactive_states` permanent cache + scaffold-aware gate (per
  `wf_lambda_internal_review/diagnose.md`) forces MCTS into Pt-acetylide collapse.
  Adding MetalLigandExchange breaks the cache without lifting the n_sim cap.
- *Effort*: 6h CPU-only (SMARTS implementation + tests + 30-cell smoke).
- *Risk*: MEDIUM — may not lift singleton attractor on all 100 pockets.

**Recommended**: **Alt B (Tasks J + D)** + **Alt C (F2(a))** combined.
Alt A is a brute-force budget lift; Alt B+D is principled per the lit anchor.

### §2.2 CFM Path (a) decode_ratio=0 — 3 alternatives

**Alt A: Move ReworkedDecoder wrap to *replace* gumbel + relax connectivity check**
- *Lit anchor*: Lipman 2023 Theorem 2 (`L_CFM = L_FM`) — pocket conditioning must
  be first-class marginal, not feature concat. Himo 2005 CuAAC regio + Lippert 2024
  Pt(II) coordination + Reedijk 1987 Pt aquation.
- *Math prior*: The wrap ordering is at the wrong abstraction layer per
  `wf_path_b_gpu_retrain/final.md §5.1`. Hypothesis (a) fix = replace gumbel +
  relax connectivity to `>= 1` + reconnect largest fragment with H-padding.
- *Effort*: 4h CPU-only (move wrap + relax check + reconnect helper + tests).
- *Risk*: LOW — hypothesis (a) is most likely per the report.

**Alt B: Drop CFM coord prior = use real bond distances + ETKDG v3 conformer init**
- *Lit anchor*: Riniker 2015 ETKDG v3 (84% / 38% RMSD≤1.0/0.5 Å on CSD) +
  Halgren 1996 MMFF94s (0.014 Å bond / 1.2° angle RMS error) + Himo 2005 DFT
  Pt-N3 distances.
- *Math prior*: Hypothesis (b) per report §5.2: CFM coordinate distribution is
  fundamentally broken (97.4% disconnect). Init CFM coords from ETKDG v3 + MMFF94s
  realigned to pocket, not random prior.
- *Effort*: 12-16h (1-2d GPU + 4h CPU integration).
- *Risk*: MEDIUM — changes the CFM training objective; may not converge.

**Alt C: Drop CFM entirely for Round-13, ship Lambda-only as canonical**
- *Lit anchor*: TODO-21 (Lambda × CFM deferred) + TODO-25 §3.5 (Secondary
  generator deferral) + Lipman 2023 Theorem 2 (Lambda closure theorem as the
  load-bearing generator for typed β-NF space).
- *Math prior*: Per TODO-21, Lambda × CFM coupling is *deferred*, not "fail".
  CFM path requires 12-24h GPU budget; Lambda path is CPU-only and the
  PathA-10x3 lift is the canonical paper finding.
- *Effort*: 0h (already ship).
- *Risk*: 0 — already in §3.5 / §6.1 / §7.1 of paper as honest framing.

**Recommended**: **Alt A (wrap reorder)** — fixes hypothesis (a) which is most likely
per `wf_path_b_gpu_retrain/final.md §5.3` (177 disconnected + 15 valence_failure = 192).
If Alt A fails, **Alt B (ETKDG init)** as Round-14 follow-up.
Alt C (drop CFM) is the fallback if both A+B fail.

### §2.3 paper/main.pdf compile — 3 alternatives

**Alt A: Diagnose pdflatex failure + apply targeted fixes**
- *Lit anchor*: Standard pdflatex troubleshooting (no lit needed; this is build infra).
- *Math prior*: Per TODO-27, root cause is in main.tex / refs.bib / figure paths /
  package options. Each can be diagnosed by running pdflatex verbose.
- *Effort*: 2-5.5h CPU-only.
- *Risk*: LOW — mechanical fix.

**Alt B: Revert to known-working snapshot + apply §3 + §4 + §6 + §7 updates as patches**
- *Lit anchor*: N/A (build infra).
- *Math prior*: Per memory entry `WF-Paper-Compile-Fix 2026-09-14`, 56-page PDF
  was generated but `paper/main.pdf` is MISSING per TODO-27. Snapshot the
  last-known-good (section_03_method.pdf) and patch in §4/§5/§6/§7 incrementally.
- *Effort*: 4-6h CPU-only (snapshot + incremental patches).
- *Risk*: MEDIUM — patches may introduce new cite/ref unresolved.

**Alt C: Compile per-section, ship supplementary.pdf + section_03.pdf as separate arXiv artefacts**
- *Lit anchor*: arXiv accepts multi-PDF bundles for supplementary material.
- *Math prior*: Decompose the monolithic paper into §3.pdf + §4.pdf + §5.pdf + §6.pdf
  each compiled separately, ship as a tarball bundle. Defers the full-paper compile.
- *Effort*: 1-2h CPU-only (write 4 per-section compilations).
- *Risk*: MEDIUM — arXiv reviewer may want monolithic.

**Recommended**: **Alt A (diagnose + fix)** — fastest path to monolithic PDF.
Alt B is fallback if Alt A's diagnostic reveals a deep main.tex corruption.
Alt C is last resort for arXiv deadline.

### §2.4 §4 + §5 + §6 paper content updates — 3 alternatives

**Alt A: Wait for Round-13 re-execution (TODO-29 F2(a)) before promotion**
- *Lit anchor*: TODO-29 §"Why this is urgent" — paper §4 metrics need real data, not
  synthetic or single-smoke. F2(a) + n_sim=1000 + 100 pockets = 300 cells
  (paper-grade).
- *Math prior*: Per TODO-29 §"What this means", F2(a) is necessary for paper §4
  cells to be lifted to MEASURED.
- *Effort*: 1-2 days CPU-only (6h F2(a) + 1h process mgmt + 30min import fix +
  ~50min re-run).
- *Risk*: HIGH — F2(a) may not lift singleton attractor on all 100 pockets.

**Alt B: Cite-only SOTA path (preserve status quo)**
- *Lit anchor*: §4.4 cite-only SOTA context columns (TargetDiff / Uni-Mol-v2 /
  Pocket2Mol) already ship per `wf_3_citeonly_sota.md`. No new data needed.
- *Math prior*: Per TODO-25 §"3 paths": Path A (ship now) preserves current state.
- *Effort*: 0h (already ship).
- *Risk*: 0.

**Alt C: Hybrid Path (Lambda + CFM partial) with honest framing**
- *Lit anchor*: §4.11 Hybrid arm WIP per `WF-Lambda-Only-Paper-Path`. Already ships
  as WIP column with 5-item follow-up list.
- *Math prior*: Lambda PathA-10x3 verified (n_distinct=20, div_tan=0.1065) +
  CFM Path B 5 P0 fixes ship + decode_ratio=0 honest framing. Hybrid arm = best of both.
- *Effort*: 0.5h (already ship).
- *Risk*: 0.

**Recommended**: **Alt C (Hybrid WIP framing)** — paper ships with current state +
honest framing. Alt A in parallel as Round-13 re-execution. Alt B as fallback if A fails.

---

## §3. Prioritization (impact × feasibility × lit-groundedness)

### §3.1 Impact on paper-grade data (high → low)

| Gap | Paper impact | Round | Priority |
|---|---|---|---|
| **paper/main.pdf compile** | arXiv submission blocked | W38 | **P0** |
| **Pocket-invariance (Alt B+C)** | §4 Table 1 100×3 = paper-grade | W39-W40 | **P0** |
| **CFM Path (a) decode=0** | §4.6 Vina column + §4.11 Hybrid | W40-W41 | **P1** |
| **§4/§5/§6 paper content** | 9 limitations + 5 NEW gaps ship | W41 | **P1** |

### §3.2 Feasibility (CPU-only → GPU-required)

| Action | Resource | Wall |
|---|---|---|
| Alt A (pdflatex diagnose+fix) | CPU | 2-5.5h |
| Alt B (Tasks J + D novel-pocket smoke) | CPU | 6-8h |
| Alt C (F2(a) MetalLigandExchange SMARTS) | CPU | 6h |
| Alt C paper content (§4/§5/§6 updates) | CPU | 4-6h |
| Alt A CFM wrap reorder | CPU | 4h |
| Alt B CFM ETKDG init | GPU + CPU | 12-16h |
| Round-13 re-run 100×3 | CPU (Lambda) | ~50min single-thread |
| Round-14 100×3 | CPU (Lambda) | ~50min single-thread |

### §3.3 Lit-groundedness (per phase1_lit.md §7)

| Gap | Lit anchor | Already-cited |
|---|---|---|
| Pocket-invariance | Guan 2023 + Peng 2022 + Bemis 1996 + Jasial 2021 + Peter 2019 | YES |
| MCTS budget | Auer 2002 + Auger 2013 Th.5 + Chaslot 2008 + Lai 1985 | YES |
| CFM decode_ratio | Lipman 2023 Th.2 + Albergo 2023 + Himo 2005 | YES |
| CFM coord prior | Riniker 2015 + Halgren 1996 | YES |
| F2(a) Pt-ligand exchange | Lippard-Berg 1995 + Reedijk 1987 + Taube 1952 | YES |
| Lambda × CFM deferred | Gat 2022 Th.3.5/3.6 | YES |
| pIC50 oracle | Tosh 2021 RL-AL | YES |
| Metal-aware SBDD | Aguilar-Rico 2024 + Willnhammer 2025 | YES |

### §3.4 Prioritized action list (P0 → P1 → P2)

**P0 (this week W38, ship-blocking for arXiv):**
1. **Alt A (pdflatex diagnose+fix)** — 2-5.5h CPU. Restore `paper/main.pdf`.
2. **Alt B+C (Pocket-invariance Tasks J + D + F2(a))** — 12-14h CPU. Break
   singleton attractor on novel pockets.
3. **Alt C (§4/§5/§6 paper content updates)** — 4-6h CPU. Apply lit-grounded
   updates preserving honest framing.

**P1 (W39-W40, ship-blocking for paper-grade data):**
4. **Round-13 re-execution** — ~50min CPU single-thread 100×3 with
   Tasks J + D + F2(a) enabled.
5. **Alt A CFM wrap reorder** — 4h CPU. Fix hypothesis (a) per
   `wf_path_b_gpu_retrain/final.md §5.1`.
6. **§4.2 + §4.3 + §4.6 DESIGN → MEASURED promotion** — 1h CPU. Only after
   paper_grade_data_ready=TRUE.

**P2 (W41, ship-blocking for arXiv submission):**
7. **§6 reduce 9 → 7 caveats** — 1h CPU. After Round-13 closes.
8. **§7 reduce 5 NEW gaps → 3** — 1h CPU. After Round-13 closes.
9. **arXiv submission prep** — 2h CPU. Compile final PDF + write cover letter.

**P2 fallback (W41+, Round-14 carry-over):**
10. **Alt B CFM ETKDG init** — 12-16h GPU + CPU. If wrap reorder (Alt A) fails.
11. **Round-14 100×3 sweep** — ~50min CPU. If Round-13 lift is partial.

---

## §4. 12-week roadmap (W38-W49, Q4-2026 + Q1-2027)

### Round-14 (W38-W41, 4 weeks) — close Round-13 + ship paper-grade data

**W38 (this week)**:
- Mon: Alt A pdflatex diagnose + fix (P0)
- Tue: Alt B+C Pocket-invariance Tasks J + D + F2(a) (P0)
- Wed: Alt C paper content §4/§5/§6 updates (P0)
- Thu-Fri: Round-13 re-run 100×3 with Tasks J+D+F2(a) (P1)

**W39**:
- Mon: §4.2 + §4.3 + §4.6 DESIGN→MEASURED promotion (P1)
- Tue: Alt A CFM wrap reorder (P1)
- Wed-Thu: CFM Path B 5000-step verify at n_sim=1000 (GPU)
- Fri: §6 reduce 9→7 + §7 reduce 5→3 (P2)

**W40**:
- Mon-Tue: paper/main.pdf final compile (Alt A retry if needed)
- Wed-Thu: §1 + §2 + §3 final review (lit-grounded per phase1 §3)
- Fri: arXiv cover letter + supplementary.tex final review

**W41**:
- Mon-Tue: arXiv submission prep
- Wed: **arXiv SUBMIT** (Q1-2027 target)
- Thu-Fri: Round-14 carry-over (Alt B CFM ETKDG init if needed)

### Round-15 (W42-W45, 4 weeks) — Round-14 paper-grade lift + GPU retrain

**W42-W43**: GPU retrain 10000-step + h=128 + tmQM init + Alt A+B fixes
**W44**: Round-14 100×3 sweep at n_sim=1000 + F2(a) + Tasks J+D
**W45**: paper §4.6 + §5 + §6.1 v2 update + journal submission prep

### Paper submission (W46-W49, 4 weeks) — journal target

**W46**: Journal selection (per `decisions.md D11`):
- (a) Digital Discovery (RSC) — strong fit for lit-grounded + open-source
- (b) J. Chem. Inf. Model. — fit for SBDD benchmark
- (c) Nat. Comput. Sci. — broader audience, higher bar

**W47-W48**: Format per journal + cover letter + supplementary
**W49**: **JOURNAL SUBMIT**

---

## §5. Decision tree (per gap → action)

### §5.1 Pocket-invariance decision tree

```
Pocket-invariance gap
├── Q1: Does Round-13 re-execution (Tasks J + D + F2(a)) lift singleton?
│   ├── YES (n_distinct>1 on all 100 pockets) → ship §4.2 + §4.3 MEASURED
│   └── NO (singleton persists on novel pockets) → Alt A (n_sim=5000 + virtual loss)
└── Q2: After Alt A, does it lift?
    ├── YES → ship §4.2 + §4.3 MEASURED at higher n_sim
    └── NO → Alt B (CFM + Lambda hybrid per-pocket) per TODO-21 §3
```

### §5.2 CFM Path (a) decision tree

```
CFM Path B (decode_ratio=0/192)
├── Q1: Does Alt A (wrap reorder) lift decode_ratio?
│   ├── YES (decode_ratio > 0.5) → ship §4.11 Hybrid arm + §6.1 update
│   └── NO → Alt B (ETKDG init) 12-16h GPU
└── Q2: After Alt B, does it lift?
    ├── YES → ship §4.6 Vina + §4.11 Hybrid arm
    └── NO → Alt C (drop CFM, Lambda-only canonical, §3.5 + §6.1 + §7.1 honest framing preserved)
```

### §5.3 paper/main.pdf decision tree

```
paper/main.pdf missing
├── Q1: Does Alt A (pdflatex diagnose+fix) produce main.pdf?
│   ├── YES (0 errors, 0 unresolved) → ship monolithic PDF
│   └── NO → Alt B (snapshot section_03 + incremental patches)
└── Q2: After Alt B, does it produce main.pdf?
    ├── YES → ship monolithic PDF
    └── NO → Alt C (multi-PDF bundle: section_03.pdf + section_04.pdf + ...)
```

### §5.4 §4/§5/§6 paper content decision tree

```
Paper content updates
├── Q1: Does Round-13 re-execution close?
│   ├── YES (300 cells MEASURED) → §4.2 + §4.3 + §4.6 promote DESIGN→MEASURED
│   │                              §6 9→7 limitations
│   │                              §7 5→3 NEW gaps
│   └── NO (partial / no lift) → §4.11 WIP + §6.1 + §7.1 honest framing preserved
└── Q2: Does Alt A CFM wrap reorder close?
    ├── YES (decode_ratio>0.5) → §4.11 Hybrid arm ship + §6.1 update
    └── NO → §3.5 + §6.1 + §7.1 honest framing preserved (no promotion)
```

---

## §6. Dependencies + success criteria

### §6.1 Dependencies

| Action | Depends on | Blocks |
|---|---|---|
| Alt A pdflatex fix | `paper/main.tex` + `paper/refs.bib` integrity | arXiv submission |
| Alt B+C Pocket-invariance | Tasks J + D + F2(a) (already ship) | Round-13 re-run |
| Alt C paper content | Round-13 close | §6 reduce caveats |
| Round-13 re-run | Pocket-invariance fixes | §4 promotion |
| Alt A CFM wrap reorder | `wf_path_b_gpu_retrain` Phase 1 (DONE) | CFM §4.11 + §6.1 |
| §4 promotion | Round-13 close + CFM close | arXiv submission |
| arXiv submission | §4 promotion + paper/main.pdf | journal target |

### §6.2 Success criteria (per round)

**Round-14 (W41 end)**:
- `paper/main.pdf` exists (≥56 pages, 0 unresolved refs, 0 fatal errors)
- §4 Table 1: 100×3 cells ≥ 90% MEASURED (was 0% per Round-13 partial)
- §4.6 PB column: ≥60% cells MEASURED (was null)
- §4.6 Vina column: ≥60% cells MEASURED (was null)
- §4.6 Diversity column: ≥60% cells MEASURED (was null)
- §6 caveats: 9 → 7
- §7 future work: 5 NEW gaps → 3 (2 closed by Round-13/14 lift)
- arXiv preprint ready (cover letter + supplementary.tex + main.pdf)

**Round-15 (W45 end)**:
- §4 Table 1: 100×3 cells ≥ 99% MEASURED
- §4.6 all columns MEASURED at paper-grade scale
- §6 caveats: 7 → 5
- §7 future work: 3 → 1 (4 closed by GPU retrain lift)
- journal target selected per `decisions.md D11`

**Paper submission (W49 end)**:
- Journal format applied (Digital Discovery / JCIM / Nat. Comput. Sci.)
- Cover letter + supplementary shipped
- JOURNAL SUBMIT (target Q1-2027 Q2)

---

## §7. Honest framing (preserved verbatim)

1. **No theoretical derivation is re-attempted in this plan.** Each fix path cites
   the lit anchor in phase1_lit.md §1-§6 + Lit-Survey-v2 65 papers + 20 theorems.
2. **All claimed lifts are PROJECTIONS, not measurements.** Per TODO-25 §"5 weak
   metrics → existing lit + plan", the (a)+(b)+(c)+(d) projected lift for Vina
   (-2 → -7) is a *lit-grounded projection*, not a measurement. Round-13 partial
   did NOT confirm or refute this projection.
3. **Path A-10x3 diversity lift is MEASURED** (n_distinct 1→20, div_tan 0→0.1065 on
   test_000..test_009 only). Pocket-invariance on novel pockets (test_010..test_019)
   is NOT MEASURED. This is the gap.
4. **Path B CFM decode_ratio=0/192 is MEASURED honest-negative.** Path B 5 P0
   fixes ship but wrap-ordering issue (hypothesis a per
   `wf_path_b_gpu_retrain/final.md §5.1`) is the dominant cause. Alt A (wrap
   reorder) is the recommended fix.
5. **5 NEW research gaps of TODO-25 §5 remain open** and traceable in
   §3.5 → §4.11 → §6.1 → §7.1 of paper.
6. **Round-13 partial = honest-negative, not failure.** Per
   `wf_round13_100x3/final.md §7`, the integration refused to fabricate
   measurements. Round-13 re-execution (per TODO-29 F2(a)) is the gating
   precondition for paper-grade scale-up.

---

## §8. Cross-references

* `phase1_lit.md` — 6 axes lit survey
* `TODO/pending/25_round14_lit_grounded_plan.md` — 5 weak metrics → lit
* `TODO/pending/26_round13_round14_complete_plan.md` — comprehensive ship plan
* `TODO/pending/27_paper_main_pdf_repair.md` — pdflatex fix
* `TODO/pending/28_round12_honest_negative_reframe.md` — R12 PARTIAL_LIFT framing
* `TODO/pending/29_f2a_round13_retry.md` — F2(a) + Round-13 re-execution
* `molmetal/reports/wf_round13_100x3/final.md` — Round-13 honest-negative
* `molmetal/reports/wf_path_b_gpu_retrain/final.md` — CFM Path B honest-negative
* `molmetal/reports/wf_round12_lambda_patha_10x3/final.md` — PathA-10x3 PARTIAL_LIFT
* `molmetal/reports/wf_lit_survey_v2/synthesis.md` — 65 papers + 20 theorems

---

## §9. Update protocol

Append-only. New fix-ship or new measurement requires a dated section.
Round-14 Phase 2 (code-fix) and Phase 3 (paper §3+§5+§6 update) will
reference §1-§6 anchors.
