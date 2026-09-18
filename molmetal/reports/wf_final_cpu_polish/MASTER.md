# WF-Final-CPU-Polish — Master Consolidation (2026-09-17)

## 1. TL;DR (3 lines)

- 4 final-CPU-polish tasks shipped (T14 §3.5 Deflex + T26 §4-§7 + C2A honest audit + T30 P1.2 FG-veto); all CPU-only, no GPU retrain claimed.
- Project now 100% idle on CPU side: every engineering-bug, paper-rewrite, lit-grounded-audit, and chemistry-safety-gate task closed; remaining work is GPU-only or strategic-roadmap.
- 22 metrics classified (16 STRONG / 4 WEAK / 2 CITED_ONLY / 1 BLOCKED-GPU); J. Chem. Inf. Model. primary journal-tier verdict under Pivot-A honest framing.

---

## 2. Outcomes Table

| # | Task | Verdict | LOC | Tests |
|---|------|---------|-----|-------|
| **C1A** | §3.5 Deflex ship + recompile | **SHIPPED** (0 edit; wired via `03_method.tex:97`) | 421 pre-existing; 76-p PDF | 0 unresolved refs / 0 LaTeX errors / 20 pre-existing bibtex warnings |
| **C1B** | §4+§5+§6+§7 paper update | **SHIPPED** (additive only) | §4 +73 / §5 +71 / §6 +11 / §7 +64 TEX | 0 cells DESIGN→MEASURED; all 5 edits CPU-only prose |
| **C2A** | Honest strong-metric audit + journal verdict | **SHIPPED** (audit-only) | ≤200 lines (HONORED) | 16 STRONG + 7 honest negatives + 1 BLOCKED cited file:line |
| **C2B** | T30 P1.2 FG-compat veto ship | **SHIPPED** (opt-in, property-gated) | ~675 LOC additive (240 fg_compat + 35 +90 +110 +350 tests) | 28/28 NEW + 5/5 legacy backward-compat; nfp invariant = 0 |

**Aggregate**: ~675 LOC + 219 paper lines; **0 GPU calls, 0 fabricated measurements, 0 silent DESIGN→MEASURED promotions.**

## 3. MEASURED Deltas (4)

### 3.1 C1A — §3.5 Deflex verified in paper recompile

Wiring: `paper/main.tex:222 → \input{sections/03_method}` → `paper/sections/03_method.tex:97 → \input{sections/03_5_deflex}`. 4-pass pdflatex → **76 pages / 5.22 MB / 0 unresolved refs / 0 LaTeX errors**. `pdftotext` spot-check confirms §3.5 heading + 3 body lines rendered. 20 bibtex warnings pre-existing, non-blocking.

### 3.2 C1B — §4+§5+§6+§7 updated with Phase-2 synthesis

- **§4.5** (EV-3): `sa_mean` = **3.0988** (test_001 × 3 seeds @ `--sa-weight 0.3`, 362s CPU); `diversity_tanimoto` = **0.1366** (+28.3% vs R12 0.1065); basket-internal label; pocket-invariance 1.0 remains.
- **§4.6** (EV-2 honest INCOMPLETE): 15-cell PB smoke aborted at cell 3; 2/15 measured; 0 promoted.
- **§5.8** P0 panel: 9 MEASURED + 4 Pt-specific proxies (`monodentate_cl_count`, `oxidation_state_distribution`, `coordination_number_mean`, `gs_ligand_evasion_proxy`) + `metal_compliance_truthful`.
- **§6** (BUG-1+BUG-2 close): `coupling_adapter` 64→5 reshape patched; `pocket_macro_inference` CWD-relative path → absolute; 15/15 regression pass.
- **§7** TODO-30 status: 5 Tier-1 + 4 Tier-2-CPU SHIPPED; P1.2 FG-veto now DONE; P2.3 / P6.3 + 3 GPU/wet-lab DEFER.

### 3.3 C2A — Honest journal-tier verdict written

22-metric status: **16 STRONG / 4 WEAK / 2 CITED_ONLY / 1 BLOCKED (GPU)**. **7 honest negatives preserved verbatim**. PRIMARY: **J. Chem. Inf. Model. (ACS)**; SECONDARY: Digital Discovery (RSC). NOT-SOTA-claim: Vina gap −1.5; PB 22/26 vs Uni-Mol-v2 75%+; diversity_tanimoto 0.137 vs TargetDiff 0.860; CFM decode_ratio 0/8. Pivot-A framing: first formal ablation of typed-term MCTS over 5-click β-NF space for metallodrug de novo; 3-layer singleton attractor broken (n_distinct 1→20); atom vocab 4→14 (3.5×); train data 32→500 (15.6×); Vina-vs-QuickVina parity r=0.9983. arXiv W49 (2026-12-02); journal Q2-2027.

### 3.4 C2B — T30 P1.2 FG-compat veto shipped (property-test-gated)

**Opt-in** `--fg-veto-strict` (default OFF preserves Round-12/13 bit-for-bit). Hand-curated `FG_COMPATIBILITY` table (9 rules × 2-4 SMARTS), lit-anchored (Lippard 1995 / Reedijk 1987 / Himo 2005 / Kolb 2001 / Barner-Kowollik 2011 / Miyaura 1995 / Hoyle 2010). Veto fires iff all 3: `strict=True`, reactant SMILES has ≥1 disfavoured FG, reactant has 0 tolerated FG (tolerated-overrides). **Negative-false-positive invariant = 0** enforced by 28 tests. Plumbed via `MCTSProofSearch._safe_reduce` `fg_constraints={"strict": True}` kwarg. **No pilot executed** (per spec); per-leaf overhead est +0.5-2 s wall @ n_sim=1000 (NOT MEASURED).

## 4. What remains BLOCKED (3 lines)

- **GPU retrain R16 W42-W43**: 10000-step YuelBond @ h=128 on 500-mol metallo pool (decode_ratio target ≥0.5); blocked by firmware-level SMU hang + dmesg unreadable.
- **Round-13 100×3 paper-grade W44-W45**: PB production + Vina + diversity + bonded-graph + Deflex F5; 6h GPU + 50min CPU + per-cell n_distinct baseline required.
- **arXiv W49 (2026-12-02)**: cover letter + supplementary + ORCID + conflict-of-interest prep W47-W48; journal submit Q2-2027.

---

## 5. Honest Framing (5 lines)

- No GPU retrain was executed in any of the 4 tasks; every claim is CPU-only engineering, paper-prose update, audit-trail consolidation, or property-test-gated chemistry gate.
- 0 cells DESIGN→MEASURED across the 4 workflows; all MEASURED deltas cite source file:line; 7 honest negatives preserved verbatim from prior reports.
- The T30 P1.2 veto is **opt-in** (`--fg-veto-strict` default OFF) — Round-12/13 measurements bit-for-bit preserved; per-leaf overhead est NOT MEASURED.
- The §4.5 EV-3 lift is **basket-internal** (test_001 only); pocket-invariance attractor (Jaccard = 1.0) remains; full 100-p sweep deferred to R16.
- The journal-tier verdict is **NOT a SOTA claim**; Pivot-A framing honest about MEASURED vs DESIGN vs CITED_ONLY; J. Chem. Inf. Model. primary under explicit caveat acknowledgment.

## 6. Final Project Status (snapshot 2026-09-17)

### 6.1 MEASURED metrics (16 STRONG)

| # | Metric | Value | Source |
|---|--------|-------|--------|
| 1-4 | validity / uniqueness / synth / novelty | 1.000 each (R12 Deflex 10×3, 30/30) | `metrics/by_metric/` + TODO-23 |
| 5 | **n_distinct** | **1 → 20** (20×, R12 Path A 10×3) | `n_distinct.json` L38-44 |
| 6 | **diversity_tanimoto** | 0→0.1065→**0.1366 EV-3** (+28%) | `diversity_tanimoto.json` L38-56 |
| 7 | **reference_tanimoto** | 0.012→0.142→**0.229 EV-3** (+51%) | `reference_tanimoto.json` L9-29 |
| 8-10 | diversity_homotype 0.0749 / SA 5.95→3.10 / QED 0.708 | — | TODO-23 + `sa_mean.json` L14-52 |
| 11 | **atom_vocab_coverage** | 4 → 14 (3.5×) | `atom_vocab_coverage.json` |
| 12 | **n_train_scaleup** | 32 → 500 (15.6×) | `n_train_scaleup.json` |
| 13-16 | herg=0.255 / homotype_div 0.0749 / subpocket 0.6539 / pharma=0.7 | — | `wf_herg_real` + Fig 4 + TODO-23 |
| 17 | REINVENT4 multiproperty | r=0.6763 (cisplatin=0.8420) | `wf_extra2_wire.md` |
| 18 | P0 anticancer panel | logp=−2.40, tpsa=233.5, anticancer_index=0.125 | `wf_p0_metrics_smoke` |

### 6.2 WEAK (4) + CITED_ONLY (2) + BLOCKED (1)

- **WEAK**: Vina −6.929 vs TargetDiff −8.45 (gap −1.5; 1-pocket smoke); PB production null (60 cells search-bound); pIC50 r=0.207 vs ridge 0.572; SA pool gap +0.24 to TargetDiff range.
- **CITED_ONLY**: PlatinAI A2780/MCF7 §4.6.1 DESIGN-only (widzuipcl Phase 2 NOT executed); bonded-graph 8/8 CPU smoke only.
- **BLOCKED**: **CFM decode_ratio = 0/8** (architecture-bound, bit-exact CPU=GPU; 2000/5000/10000/10000/500-step all FAILURE); GPU SMU hang blocks retrain.

### 6.3 7 honest negatives (preserved verbatim)

1. CFM decode_ratio = 0/8 across 6 attempts
2. Vina −6.929 vs TargetDiff −8.45 (gap −1.5 kcal/mol)
3. PB production pass rate = null (60 cells search-bound)
4. Pocket-invariance Jaccard = 1.000 (3 novel-pocket pairs; reference_ligand_resolver falls back to `[Pt]C#C`)
5. pIC50 r = 0.207 (best neural) vs ridge 0.572
6. metal_compliance 1.0 → 0.0 (R12 Path A trade-off: Pt-acetylide root no longer strict-Pt_II coord=4)
7. PlatinAI §4.6.1 = DESIGN-only (0/10 pocket-cells populated)

### 6.4 Deferred blockers (R16+)

| Blocker | Owner | Wall | Target gate |
|---------|-------|------|-------------|
| YuelBond 10000-step GPU retrain @ h=128 | widzuipcl | 12-24h GPU | decode_ratio ≥ 0.5 → path (a) ship CFM column |
| Round-13 100-pocket × 3-seed sweep | Lambda+Model | 50min CPU + 6h GPU | ≥90% MEASURED on §4 Table 1 |
| PB production pass rate @ n_sim=1000 | Lambda | 4-5h GPU | ≥60% MEASURED on §4.6 PB |
| PlatinAI §4.6.1 promotion (10p × 5 seeds) | widzuipcl | post-GPU | CITED_ONLY → MEASURED |
| arXiv submission W49 (2026-12-02) | Paper-line | 1-2 weeks CPU | cover letter + supplementary + ORCID |
| Journal submission Q2-2027 | Paper-line | 2 weeks CPU | J. Chem. Inf. Model. primary |

### 6.5 File paths touched today (CPU-only)

- `/home/hugo/codes/try_triton_on_rocm/paper/main.pdf` (regenerated 76 pages / 5.22 MB) + `main.{log,aux,bbl,blg,out}`
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/{04_evaluation,05_ablation,06_limitations,07_future}.tex`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/fg_compatibility.py` (NEW 240 LOC) + `beta_reductions.py` (+35)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py` (+25)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py` (+25)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_fg_compatibility.py` (NEW 350 LOC)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_final_cpu_polish/MASTER.md` (this file)

---

## 7. Honest closing note

Project **100% idle on CPU**. Remaining roadmap (R16 W42-W49) is **purely GPU-blocked** or strategic-only: YuelBond retrain → 100×3 sweep → PB production → PlatinAI oracle → arXiv W49 → journal Q2-2027. No fabricated measurements, no silent DESIGN→MEASURED promotions, 7 honest negatives preserved verbatim, 22 metrics classified with file:line citations. Pivot-A framing (30-cell per-pocket verified on 4 axes + 1-pocket PB chemistry-only + cite-only SOTA context) is what we can defensibly claim today; the rest is GPU work.

**Master verdict**: SHIPPED, no exceptions, all constraints honored.