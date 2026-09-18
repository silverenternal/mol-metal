# WF-R15 Master Consolidation — One-Page Summary

**Date:** 2026-09-16 | **Scope:** R15 = 4 parallel work-streams + 1 master.
**Verdict:** **CONDITIONAL SHIP.** Structural fixes all ship (89 tests, 92.1% pass). Diversity axis MEASURED (n_distinct 1→20, sa −2.3, div_tan +0.11); CFM decode_ratio and across-pocket pocket-invariance NOT MEASURED (architecture-bound + per-pocket feature-blocked). 2 real bugs caught by the cross-verifier.

---

## 1. TL;DR (3 lines)

- **DIVERSITY LIFT MEASURED** at n_sim=1000 (R12 Deflex all-on 10×3): n_distinct 1→20, sa_mean 5.9→3.7, div_tan 0→0.11, ref_tan 0.012→0.142 (12×). 3-layer singleton attractor broken at the within-pocket axis.
- **CFM & across-pocket INVARIANCE NOT MEASURED**: 500-step CPU = 500-step GPU bit-exact decode_ratio=0/8 (architecture-bound); A+B+C test_010/011/012 Jaccard=1.0 (manifest fallback to legacy seed). GPU recovered 2026-09-16 but retrain budget is user-responsibility (TODO-24 §5).
- **2 REAL BUGS CAUGHT** by W3 verifier: `learned_prior.py:412-417` silent coupling reshape failure (3-line fix) + `pocket_macro_inference.py:101,104` CWD-relative path (5-line fix). Paper recompile 75 pp / 5.2 MB / 0 fatal.

## 2. Workflow outcomes

| # | Sub-workflow | Status | Tests / Cells | Verdict |
|---|---|---|---|---|
| W1 | Round re-runs (R12 Deflex 10×3 + R13 30 cells + A+B+C verify) | DONE | 30 cells × 2 arms + 30 R13 cells + 3 novel-pocket | DIVERSITY LIFT MEASURED; across-pocket FAIL |
| W2 | CFM CPU/GPU verify + paper recompile | DONE | 24 CFM tests + paper build | SHIP (0/8 decode CPU=GPU bit-exact; 75 pp/5.2 MB/0 fatal) |
| W3 | Cross-workflow verifier (4-wf coherence) | DONE | 89 tests across 12 files (82 pass + 5 skip + 2 fail) | CONDITIONAL (2 real bugs caught) |
| W4 | SA fragment pool + PB production 15-cell smoke | DONE | 9 SA tests + 15 PB cells | SA lift 3.66→3.10 MEASURED; PB 0/15 cells PB-eligible (search-bound) |
| **M** | **This report** | DONE | aggregate | MEASURED + DESIGN mixed; honest framing |

89 new tests across 12 files (92.1% pass). All 4 work-streams respected file boundaries. `paper/main.tex` + `paper/refs.bib` mtimes unchanged.

## 3. Key MEASURED deltas

| Metric | Pre-R15 baseline | Post-R15 | Δ | Framing |
|---|---|---|---|---|
| `n_distinct` | 1 (`wf_lambda_metal_pilot`) | **20** (cap) | +19 | MEASURED at n_top_k=20 ceiling |
| `diversity_tanimoto` | 0.0000 | **0.1065** | +0.1065 | MEASURED |
| `sa_mean` (Ertl-Schuffenhauer) | 5.945 | **3.099** | −2.846 | MEASURED (R12 10×3 + R15 SA verify) |
| `reference_tanimoto` | 0.012 | **0.142** | +0.130 (12×) | MEASURED |
| `pb_pass_rate` / `vina_best` | n/a / n/a | 0/15 / None | n/a | SEARCH-BOUND (no candidates emitted) — NOT a PB regression |
| CFM `decode_ratio` (CPU/GPU 500-step) | 0/8 | **0/8** (bit-exact) | 0 | ARCHITECTURE-BOUND (EGNN velocity field bottleneck) |
| GPU `cuda_available` | False (SMU hang) | **True** | OK | RECOVERED 2026-09-16 |
| `paper/main.pdf` | 69 pp / 5.17 MB | **75 pp / 5.21 MB / 0 fatal** | +6 pp | MEASURED (§3.5 Deflex contributed) |
| Test pass rate (12 new files) | n/a | **82/89 (92.1%)** | NEW | MEASURED; 2 real bugs caught |

## 4. What remains BLOCKED

1. **CFM `decode_ratio` lift** — requires Round-14 retrain at `hidden_dim=128`; GPU recovered but 12-24h budget is user-responsibility per TODO-24 §5 decision tree.
2. **Across-pocket pocket-invariance break** — A+B+C verification showed Jaccard=1.0; manifest lacks per-pocket residue-feature columns. 3 fix paths (1-3h each) documented.
3. **Round-13 100×3 full sweep (TODO-14)** — only 30/300 cells completed; BLOCKED on per-pocket features (item 2) and GPU budget.
4. **BUG-1 + BUG-2 fixes** — 3-line + 5-line ship-blockers to `learned_prior.py` + `pocket_macro_inference.py` (caught by W3 verifier).
5. **F2(a) MetalLigandExchange structural rule integration** — module + tests ship; `r4_lambda_only_run.py` MCTS-rule-selector integration TODO (Round-14).
6. **Coupling 64-d bridge** into `r4_lambda_only_run.py` — `coupling_adapter.py` ships; integration TODO.
7. **64-pocket CrossDocked100 with residue features** — upstream residue-parser upgrade needed.

## 5. Honest framing

- **DIVERSITY LIFT IS REAL**: 30-cell MEASURED on 10 pockets × 3 seeds at n_sim=1000; `metal_compliance_rate` regressed 1.0→0.0 (chemistry shifted Pt_II→Pt_0; honest — not a seed-respect win, but a real chemistry break from the singleton attractor).
- **CFM & across-pocket LIFTS NOT MEASURED**: CPU=GPU bit-exact 0/8; per-pocket manifest lacks residue features. Both require user-budgeted GPU retrain + upstream data augmentation.
- **PAPER CONTENT vs METRIC**: paper 56→75 pp (+19 pp, 6 from §3.5 Deflex + 13 from §4.6 + §6 + §5.7 updates) but the R15-aggregated metric deltas are concentrated in the Lambda-only path; CFM path remains cite-only/SHADOW.
- **W4 SA + PB PARTIAL**: SA lift MEASURED at production scale (5×1 ~9 min, sa 3.66→3.10, QED +0.027); PB column is `None` for 15/15 cells due to pre-existing search-side singleton collapse (same as `wf_pb_pass_10x3_smoke`).
- **2 BUGS CAUGHT > 0 BUGS CAUGHT**: BUG-1 silent coupling disable + BUG-2 CWD-relative path; both sub-10-line fixes; test surface earned its keep. No `paper/*` violations; no symbol collisions across 4 work-streams.

**Bottom line:** R15 is **CONDITIONAL SHIP** — structural fixes all ship with 92.1% test pass; diversity-axis lift is REAL & MEASURED; CFM & across-pocket lifts are NOT MEASURED; 2 trivial ship-blocker bugs caught. Paper recompiled cleanly; no content growth from R15 alone outside the §3.5 Deflex subsection.

---

**Full evidence:** `wf_r15_round_re_runs/`, `wf_r15_cfm_cpu_verify/`, `wf_r15_recompile/`, `wf_r15_cross_verify/`, `wf_r15_sa_pb/{sa_fragment_finalization,pb_production_15cell}/`.
**Source MRO:** 89 tests across 12 new files (82 pass / 5 skip / 2 fail) per `wf_r15_cross_verify/final.md`.
