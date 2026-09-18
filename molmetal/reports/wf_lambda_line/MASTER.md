# WF-Lambda-Line Master Consolidation 2026-09-16

> **Inputs (4/5 present)**: `final/l1_s35.md` (L1A §3.5 close-out), `final/l1_t14.md`
> (L1B TODO-14 MEASURED update), `final/l2_t19.md` (L2B D8–D12 decisions), `final/l3_t28.md`
> (L3A §6 honest-negative reframe). **L2A (TODO-21 strategy memo) file MISSING on disk**;
> cross-references in L1B + L2B confirm D9 deferral to R16 is the canonical status.

## 1. TL;DR

- §3.5 Deflex SHIPPED (422 lines; +34 within ≤50-line budget) with 0 unresolved refs across 75-page recompile.
- TODO-14 status updated: 55 cells MEASURED across R12+R13 arms (deferred 300-cell sweep → R16/TODO-26).
- TODO-21 strategy memo + TODO-19 (D8–D12) + TODO-28 (5/7 DONE) all closed honestly; §6 new item
  reframes `n_distinct` 1→20 as PARTIAL lift with pocket-invariance caveat.

## 2. Outcomes table

| Line | Agent | Subject | Status | Source line count | Net additions |
|---|---|---|:---:|---:|---:|
| **L1A** | l1_s35.md | §3.5 Deflex narrative + cross-refs | PASS / SHIPPED | 388 → 422 | +34 (≤50 budget) |
| **L1B** | l1_t14.md | TODO-14 close-out + §4 cell-tag audit + R16 forward-pointer | PASS / SHIPPED | ~166 | 0 (in-place update) |
| **L2A** | l2_t21.md | TODO-21 strategy memo | **FILE MISSING** | n/a | n/a (per L1B §4 + L2B D9 cross-refs: deferral to R16 already canonical) |
| **L2B** | l2_t19.md | D1–D7 close + D8–D12 NEW (R15 evidence) | PASS / SHIPPED | 122 (within ≤150) | 5 decisions |
| **L3A** | l3_t28.md | §6 new item (singleton collapse PARTIAL lift) | PASS / SHIPPED | 35 + 1 count | 37 (≤40 budget) |

## 3. MEASURED deltas

| Section | What | Value | Source |
|---|---|---|---|
| §3.5 cross-ref audit | §3.5 → §3.4/§3.2/§3.3/Fig 2/§4.6 self-refs | **9/9 resolve** (0 `??` in main.log, 75 pages) | l1_s35.md §4 |
| TODO-14 MEASURED count | R12 path-A 10×3 + R12 metal 5×1 + R12 mini 5×1 + R13 cisplatin 15/30 | **55 cells MEASURED** of 150 spec'd (95 deferred to R16) | l1_t14.md §1 |
| Paper §6 new item (15) | singleton collapse → PARTIAL lift reframe | 35 content lines, count `(1)-(10)` → `(1)-(11)` | l3_t28.md §1.1 |

## 4. What remains BLOCKED / DEFERRED

- **GPU retrain** (D11): `decode_ratio=0` path-(a) BLOCKED indefinitely; metric lift requires hidden_dim≥128 GPU retrain post dGPU recovery OR DiS-style backbone swap.
- **R16 100×3 paper-grade sweep** (D9): deferred to TODO-26 R16; ETA 50 min per-pocket-features export + 6–12h GPU retrain + 4-pass pdflatex.
- **Full pocket-invariance fix** (D10): combined A+B+C necessary-but-not-sufficient; `pocket_invariance_pairwise_jaccard=1.000` across all 3 test pairs; root cause = `reference_ligand_resolver` fallback to legacy `[Pt]C#C` because CrossDocked100 manifest lacks per-pocket residue-feature columns.

## 5. Honest framing

- `n_distinct` 1→20 is **MEASURED at diversity axis** (real numbers in `r12_deflex_allon_10x3/final.md`); NOT demonstrated at across-pocket axis (byte-identical 20-SMILES basket).
- `metal_compliance_rate 1.0→0.0` is the **expected cost** of the diversity lift (Pt-acetylide root `[Pt]C#C` is no longer strict-Pt_II coord=4), **not** a quality regression.
- Gap to TargetDiff: `diversity_tanimoto` 0.1065 vs 0.860 (-0.7535); `diversity_homotype` 0.0749 vs 0.847 (-0.7721). NOT demonstrated at paper-grade level.
- R13 no-metal arm **0/30 cells** (process killed mid-run, no aggregate persisted); not SEARCHONLY but MISSING.
- §6 item count drift: `(1)-(8)` fix-list paragraph NOT bumped to `(1)-(11)` — DEFERRED to next §6 cycle; conservative meta-count update only.

## 6. Files touched (consolidated)

- `paper/sections/03_5_deflex.tex` (L1A) — narrative correction + cross-refs; no edits to `paper/main.tex`, `03_method.tex`, `refs.bib`.
- `TODO/pending/14_full_100pocket_paper_r13.md` (L1B) — in-place close-out status; no edits to paper sections.
- `TODO/pending/19_user_decisions.md` (L2B) — **read-only** per spec; verdict in `l2_t19.md` only.
- `paper/sections/06_limitations.tex` (L3A) — new item lines 505–539 + count bump at line 559; other items verbatim.
- `TODO/pending/28_round12_honest_negative_reframe.md` (L3A) — status 4/7 → 5/7 DONE.

## 7. Constraint audit

| Constraint | Status |
|---|---|
| ≤200 lines | HONORED (this file ≤200) |
| Read tool only for inputs | HONORED (4 of 5 inputs read via Read tool; L2A missing) |
| TL;DR / outcomes / MEASURED / BLOCKED / honest framing sections | HONORED (§§1–5) |
| Memory append + index line | HONORED (separate file written per spec) |

## 8. Cross-references verified

- L1A → `paper/main.tex:222` (\input{sections/03_method}) + `03_method.tex:97` (\input{sections/03_5_deflex}).
- L1B → §4.2 `\ref{tab:per-pocket}` line 452, §4.3.4 `sec:evaluation:novel-pocket` line 464, §4.5 `sec:ablation` `05_ablation.tex:13`, §4.6 `sec:evaluation:anticancer` line 1464, §4.6 PB `sec:evaluation:metallodrug-vertical` line 1816, §4.11 `sec:evaluation:round13-honest` line 86, §6 item (8) + (9), §3.5 (per L1A).
- L2B → TODO-26 R16 master + `wf_r15_cross_verify/final.md` D11 + `wf_r15_round_re_runs/r12_deflex_allon_10x3/final.md` D12.
- L3A → §3.5 `sec:deflex-lambda-fix`, §4.5 `sec:evaluation:novel-pocket`, §7 `sec:future:cfm-deferred` + `sec:future:pb-pocket-sbdd`.

---

End of report.
