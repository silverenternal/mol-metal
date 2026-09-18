# WF-Lambda-Fix-FullPath-v2 — Paper Integration Notes

**Date**: 2026-09-15
**Status**: WIRED ✓ | HONEST NEGATIVE — diversity lift NOT achieved
**Spec**: WF-Lambda-Fix-FullPath-v2 (4 algorithmic fixes + per-click scaffold-aware gate)

## TL;DR

Both arms of the WF-Lambda-Fix-FullPath-v2 re-verification (Arm A `auto-pt-strict` CuAAC+SPAAC, Arm B `all-5 --allow-incompatible-click`) ran cleanly on the same $10 \times 3$ panel (30 cells each, ~50 s each). The 4 algorithmic fixes (Fix 1 soft prior + Fix 2 reward rebalance + Fix 3 compliance truthfulness + Fix 4 click-rules alias) and the new **Fix 2-B per-click scaffold-aware gate** (5×5 compatibility matrix, 5 scaffolds × 5 click reactions) are mechanically in place. The scaffold-aware gate is the substantive architectural contribution; the diversity-lift hypothesis is **rejected** at `n_simulations=1000`.

**n_cells_promoted: 0** (honest-negative result; no DESIGN→MEASURED promotion in §4).
**n_resolved_caveats: 1** (caveat 3 trivially-true metal_compliance=1.0 RESOLVED at metric-definition level; caveats 1+2 RETAINED).
**n_sections_updated: 5** (3 LaTeX + 1 cross-ref + 1 integrate.md = 5 file edits).
**div_tan_lift_pp: 0.000** (both arms tied Round-12 baseline at 0.0000; mini-pilot 0.005 is the only non-degenerate baseline; expected +0.15-0.25 pp did not materialise).
**scaffold_aware_gate_documented: TRUE** (5×5 matrix + Table `tab:scaffold-aware-gate` + 5 tests in `test_lambda_mcts_singleton.py`).

## 1. What the paper now claims (per section)

### §3.2 Five click reactions (NEW per-click scaffold-aware gate)

The selection-criterion sub-section (`sec:click-chem-selection`) at `paper/sections/03_2_click_chemistry.tex` now carries:

- A new paragraph documenting the 5×5 per-click × scaffold compatibility matrix (CuAAC/SPAAC/ThiolEne/Suzuki/AmideCoupling × strict_Pt_II/Pt_II_chelating/Pt_IV/labile_metal/unknown).
- A new Table `tab:scaffold-aware-gate` with the per-cell verdicts (OK / MARG / X).
- The default behaviour: strict_Pt_II (cisplatin/nedaplatin) → CuAAC + SPAAC always-on, ThiolEne + AmideCoupling gated out (chemically defensible: thiolate attacks Pt-Cl, no carboxylate to couple), Suzuki MARG.
- The CLI integration: 5 new `--click-rules auto-*` aliases + `--allow-incompatible-click` flag (default OFF).
- The implementation reference: `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` (250 LOC, 5×5 matrix).
- 5 new tests in `test_lambda_mcts_singleton.py` — all pass; full 17-test file green at 3.50s.
- Honest framing: the matrix is **architectural and chemically defensible**, but it does **NOT** lift diversity at `n_simulations=1000`. The bottleneck is the MCTS exploration / top-k cap / reward-aggregator weighting, not the click vocabulary.

### §4 Evaluation (NEW status-paragraph bullet + §4.5 paragraph)

The §4 status paragraph (`paper/sections/04_evaluation.tex` line ~30) now carries a new bullet documenting the WF-Lambda-Fix-FullPath-v2 re-verification:

- Both arms produced 30/30 cells in ~50 s each.
- The scaffold-detection helper fires as designed (Arm A log line: `auto_scaffold_detected=strict_Pt_II (metal_seed='cisplatin', smi='[Pt]C#C')`).
- Honest framing: the algorithmic fixes are mechanically in place but not sufficient to lift diversity at `n_simulations=1000`. Both arms report `n_distinct=1`, `div_tan=0.0000`, `div_hom=0.0000`, `metal_compliance_non_seed=0.0000` across all 30 cells. The expected +0.15-0.25 pp div_tan lift did NOT materialise; both arms tied the Round-12 baseline at singleton collapse. **0 cells promoted DESIGN→MEASURED** in this round.

The §4.5 hybrid-vs-Λ-only ablation block now carries a NEW paragraph at the end of the Round-12 Λ-only panel with:

- A two-arm per-pocket × per-seed n_distinct panel (both arms identical: n_distinct mean=1, max=1, div_tan=div_hom=0.0000, mc_non_seed=0.0000).
- A cross-arm uplift table (Round-12 baseline vs WF-v2 Arm A vs WF-v2 Arm B vs mini 5×1).
- A headline finding: "diversity-lift hypothesis REJECTED, scaffold-aware gate SHIPPED" — the 4 algorithmic fixes + scaffold-aware gate did not lift diversity; the substantive contribution is the 5×5 per-click compatibility matrix.
- A 3-caveat resolution status:
  1. Singleton collapse (n_distinct=1): **RETAINED**.
  2. Diversity regression (0.0050→0.0000, 0.0020→0.0000): **RETAINED**.
  3. Trivially-true metal_compliance=1.0: **RESOLVED at metric-definition level** (Fix 3 introduced `metal_compliance_rate_non_seed` truthful view; both arms report 0.0 because the only candidate per cell is the seed).
- A "0 cells promoted DESIGN→MEASURED in this paragraph" honest disclosure.

### §CROSS_REFS (NEW §3.2 row + §4.5 row)

`paper/sections/CROSS_REFS.md` now carries:

- **§3.2 row**: extended with a new bullet documenting the per-click scaffold-aware gate (5×5 matrix, source file, CLI flags, 5 new tests, honest-negative framing).
- **§4.5 row**: extended with a new "WF-Lambda-Fix-FullPath-v2 (2026-09-15) update" bullet documenting the two-arm scaffold-aware diversity panel + headline finding + 3-caveat resolution status + 0 promotions + cross-link to §3.2.

## 2. Cells promoted DESIGN → MEASURED

**Total: 0.** No cell of Tables 1/2 in `paper/sections/04_evaluation.tex` is re-tagged. The 4 algorithmic fixes plus the scaffold-aware gate are mechanically in place; the diversity-lift hypothesis is rejected at this search budget; the substantive contribution is the 5×5 per-click compatibility matrix documented at §3.2.

## 3. Honest caveats: 1 RESOLVED, 2 RETAINED

| # | Caveat | Status | Why |
|---|---|---|---|
| 1 | Singleton collapse (n_distinct=1) | **RETAINED** | Both WF-v2 arms return n_distinct=1 across all 30 cells; the 4 algorithmic fixes did not lift the collapse. The recommendation in §4.5 to investigate UCT exploration constant / max depth / top-k cap / reward-aggregator weighting is now the load-bearing follow-up. |
| 2 | Diversity regression (0.0050→0.0000) | **RETAINED** | Both WF-v2 arms report div_tan=div_hom=0.0000; the mini-pilot no-metal-seed baseline (0.0050, 0.0020) remains the only honest non-degenerate diversity panel. |
| 3 | Trivially-true metal_compliance_rate=1.0 | **RESOLVED at metric-definition level** | Fix 3 introduced `metal_compliance_rate_non_seed` (truthful view, excludes the seed from the numerator). Both WF-v2 arms report 0.0 because the only candidate per cell IS the seed; the non-seed numerator is empty. The headline 1.0 is now labelled "trivially true (single-molecule collapse)"; the truthful view is 0.0. |

## 4. Task metrics (per task brief)

- `n_cells_promoted`: **0** (honest-negative result; no DESIGN→MEASURED promotion in §4.2/§4.3/§4.5)
- `n_resolved_caveats`: **1** (caveat 3 trivially-true metal_compliance RESOLVED at metric-definition level; caveats 1+2 RETAINED)
- `n_sections_updated`: **5** (§4 status paragraph + §4.5 new paragraph + §3.2 new sub-paragraph + CROSS_REFS §3.2 row + CROSS_REFS §4.5 row = 5 file edits, of which 3 are LaTeX source files + 1 markdown cross-ref + 1 integrate.md = 5 file edits in this round)
- `div_tan_lift_pp`: **0.000** (both WF-v2 arms tied Round-12 baseline at 0.0000; mini-pilot 0.005 is the only non-degenerate baseline; expected +0.15-0.25 pp did not materialise)
- `scaffold_aware_gate_documented`: **TRUE** (5×5 matrix + Table `tab:scaffold-aware-gate` + 5 tests in `test_lambda_mcts_singleton.py` + integration into `r4_lambda_only_run.py` + `--click-rules auto-*` aliases + `--allow-incompatible-click` flag)
- `wall_clock_per_arm_s`: **~50** (30 cells × 2 arms, pure CPU, well under 30 min budget)
- `all_4_fixes_complete`: **TRUE** (Fix 1 + Fix 2 + Fix 3 + Fix 4 + Fix 2-B scaffold-aware gate all wired)
- `lifted_from_baseline`: **FALSE** (n_distinct=1, div_tan=0.0000, div_hom=0.0000, metal_compliance_non_seed=0.0000 on both arms)

## 5. File locations

- **Paper §3.2 new sub-paragraph**: `paper/sections/03_2_click_chemistry.tex` (selection-criterion `sec:click-chem-selection` extended with 5×5 matrix documentation)
- **Paper §4 status-paragraph bullet**: `paper/sections/04_evaluation.tex` (line ~30, WF-v2 re-verification block)
- **Paper §4.5 new paragraph**: `paper/sections/04_evaluation.tex` (end of Round-12 Λ-only panel, 2-arm diversity panel + 3-caveat resolution)
- **Paper CROSS_REFS §3.2 row**: `paper/sections/CROSS_REFS.md` (extended with scaffold-aware gate bullet)
- **Paper CROSS_REFS §4.5 row**: `paper/sections/CROSS_REFS.md` (extended with WF-v2 update bullet)
- **TODO-13 summary**: `TODO/pending/13_top_journal_pilot_r12.md` (new "WF-Lambda-Fix-FullPath-v2 integration" section appended)
- **TODO-24 summary**: `TODO/pending/24_cfm_architecture_redo_plan.md` (new "Update 2026-09-15: WF-Lambda-Fix-FullPath-v2" section appended)
- **This integration report**: `molmetal/reports/wf_lambda_fix_full_path_v2/integrate.md`
- **Source code**: `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` (250 LOC, 5×5 matrix)
- **CLI integration**: `molmetal/scripts/r4_lambda_only_run.py` (5 new auto-* aliases + `--allow-incompatible-click` flag)
- **Tests**: `molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py` (5 new tests, 17/17 file pass at 3.50s)
- **Spec doc**: `molmetal/reports/wf_lambda_fix_full_path_v2/fix2_scaffold_aware.md` (Fix 2-B specification)
- **Verdict doc**: `molmetal/reports/wf_lambda_fix_full_path_v2/final.md` (HONEST NEGATIVE for diversity, SHIPPED for scaffold-aware gate)

## 6. Honest framing summary

The WF-Lambda-Fix-FullPath-v2 round is a **mixed result**: the 5×5 per-click scaffold-aware gate is the substantive architectural contribution (chemically defensible, hand-curated, 5 new tests, full CLI integration), but the diversity-lift hypothesis is rejected at `n_simulations=1000` (both arms report n_distinct=1, div_tan=div_hom=0.0000 across all 30 cells). The expected +0.15-0.25 pp div_tan lift did NOT materialise. The bottleneck is now isolable to the MCTS exploration / top-k cap / reward-aggregator weighting, not the click vocabulary. **0 cells promoted DESIGN→MEASURED** in §4. The §3.2 5×5 matrix is the load-bearing contribution; the diversity metric is the rejected side-result. The follow-up is the C5 diversity bonus channel (`--div-weight` in `RewardAggregator`), which is the next planned workflow.
