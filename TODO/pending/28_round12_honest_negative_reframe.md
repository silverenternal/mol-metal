# TODO-28 — Reframe Round-12 as Honest (PARTIAL diversity lift verified)

**Status:** ✅ **5/7 DONE** (2026-09-16) — §4 + §6 NEW item + §3.5 + §4.5 + §7 cross-references ship per l3_t28.md verdict. Items #5 (CROSS_REFS link update) and #6 (new `\HONESTNEGATIVE{}` macro) DEFERRED — cosmetic, can defer to v2.
**Original priority:** highest (paper §4 framing)
**Owner:** (unset)
**Depends on:** TODO-13 R12 plan + TODO-23 weak-to-strong plan
**Created:** 2026-09-15
**Last updated:** 2026-09-16 (L3-T28 ship; CROSS_REFS + macro DEFERRED)

**Completion tracker:**

| # | Item | Status |
|---|---|---|
| 1 | Promote diversity_tanimoto=0.1065 + diversity_homotype=0.0749 + n_distinct=20 to MEASURED in §4.3 Table 2 + §4.6 + §5.7 | ✅ DONE (we5qbl16b) |
| 2 | Add honest trade-off note: metal_compliance 1.0→0.0 is EXPECTED cost | ✅ DONE (we5qbl16b §3.4 honest-framing paragraph) |
| 3 | Add to §6 limitations: gap to TargetDiff diversity 0.7535 still open | ✅ DONE (we5qbl16b: §6 expanded 8→12 caveats) |
| 4 | Add to §7 future work: F2(a) MetalLigandExchange SMARTS rule | ✅ DONE (TODO-29 in flight) |
| 5 | Update CROSS_REFS.md to link §4.3 / §4.6 / §5.7 to `metrics/by_round/r12_lambda_patha_10x3.json` | ⏳ TODO (low priority; manual edit) |
| 6 | Add new `\HONESTNEGATIVE{}` or `\HONESTTRADEOFF{}` macro to main.tex | ⏳ TODO (cosmetic; can defer to v2) |
| 7 | Update TODO-13 R12 plan status line | ✅ DONE (this update) |

## ⚠️ UPDATE 2026-09-15 15:35:03 — `wf_round12_lambda_patha_10x3` SHIPPED

**The diversity lift IS REAL now** (not just negative result):
- n_distinct: 1 → **20** on all 30 cells (10×3 @ n_sim=1000)
- diversity_tanimoto: 0.000 → **0.1065** (+0.1065)
- diversity_homotype: 0.000 → **0.0749** (+0.0749)
- validity / synth / uniq: 1.000 (unchanged)
- metal_compliance: 1.000 → **0.000** (EXPECTED trade-off — Pt-acetylide root not strict-Pt_II)
- Wall: 98.27 s/cell ± 2.85 (under 90-min budget)

The 4-fix bundle breaks the singleton collapse:
1. Rule-symmetry (CuAAC/SPAAC/Suzuki can fire either direction)
2. Decoder rework (chem-aware soft bond prior)
3. Scaffold-aware gate (auto-pt-strict → 3 strict-Pt_II compatible rules)
4. Partner tiles (8 new + FRAGMENT_LIBRARY_200_TILES runtime expansion)

## Problem (original framing 2026-09-15 morning)

`metrics/by_round/r12_lambda_pilot.json` and `r12_lambda_pathb.json` show Round-12 Lambda Pilot 10×3 = **honest negative result** (n_distinct=1 collapse, diversity=0.0 forced, metal_compliance=1.0 trivially true). Memory + TODO-26 + paper §4 narrative still treats this as "lift".

## What's wrong with current framing (updated 2026-09-15)

The PathA-10x3 (wf_round12_lambda_patha_10x3) is **THE** diversity lift, but:
1. **diversity_tanimoto=0.1065 IS a lift off the zero floor** — but still 0.7535 below TargetDiff 0.860
2. **metal_compliance=0.000 is an EXPECTED trade-off** — not a regression, but must be explained
3. **The metric framing must be**: "diversity lift verified; metal_compliance regressed as documented cost"
4. **The honest narrative**: "singleton attractor broken, but Pt_II strict geometry not yet compatible with click-rule chemistry — F2(a) MetalLigandExchange SMARTS rule is the structural fix"

## Status confusion (audit 2026-09-15)

| Metric | Pre-PATHA claim | Post-PATHA actual |
|---|---|---|
| diversity_tanimoto | 0.0 (forced) | **0.1065 (lifted, still below TargetDiff)** |
| diversity_homotype | 0.0 (forced) | **0.0749 (lifted)** |
| n_distinct | 1 (collapse) | **20 (n_top_k cap saturated)** |
| metal_compliance | 1.0 (trivially true) | **0.0 (trade-off, documented)** |
| Vina | UNMEASURED | UNMEASURED (still) |
| PB | UNDEFINED | UNDEFINED (still search-bound) |

## What needs to happen (updated)

1. **Paper §4.3 Table 2 + §4.6 + §5.7** — promote diversity_tanimoto=0.1065 + diversity_homotype=0.0749 + n_distinct=20 to MEASURED with the 30-cell sample
2. **Add honest trade-off note**: metal_compliance 1.0→0.0 is EXPECTED cost of diversity lift (not regression)
3. **Add to §6 limitations**: gap to TargetDiff diversity 0.7535 still open; Pt_II strict geometry not yet compatible with click chemistry
4. **Add to §7 future work**: F2(a) MetalLigandExchange SMARTS rule = structural fix for BOTH diversity AND metal compliance simultaneously
5. **Update CROSS_REFS.md** to link §4.3 / §4.6 / §5.7 cells to `metrics/by_round/r12_lambda_patha_10x3.json` `verdict.status = PARTIAL_LIFT_VERIFIED`
6. **Add new `\HONESTNEGATIVE{}` or `\HONESTTRADEOFF{}` macro** to main.tex
7. **Update TODO-13 R12 plan status line**: R12 done; diversity lift verified; see TODO-28 for honest trade-off framing

## Why this is high priority

- **diversity_tanimoto=0.1065 IS publishable** — first real lift in 4 pilots
- **metal_compliance=0.000 needs honest explanation** — without it, reviewer will think this is "worse than baseline"
- **Vina/PB still missing** — paper §4 still has 55+ DESIGN cells in §4 Table 1
- **arXiv preprint ready for partial submission** — can ship §4.3 / §4.6 / §5.7 with 30-cell Lambda path now; §4.2 / §4.4 / §4.5 still DESIGN until R13 100×3 sweep

## Effort

- 0.5h editing §4.3 / §4.6 / §5.7 cells to add new MEASURED values + trade-off footnote
- 0.5h writing the honest-trade-off footnote text
- 0.5h updating CROSS_REFS.md + TODO-13 + INDEX.md
- **Total: 1.5h CPU-only**
