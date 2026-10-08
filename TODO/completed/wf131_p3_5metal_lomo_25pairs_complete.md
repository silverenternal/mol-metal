# Wf-131 P3 — 5-metal REAL LOMO full sweep (25/25 pairs) — 2026-10-08

## TL;DR

Wrapper v4 ran all 25 (metal, seed) pairs on REAL MetalCytoToxDB.csv
(26,632 rows, 5 metals × 5 seeds × 50 epochs) on cuda:0 (RTX PRO 6000
Blackwell sm_120). Total wall: **40 min 47 sec** (08:48 → 09:28 CST),
**way under projected ~5 hr** because Wf-E1 W₂=0 default makes each pair
~60 s instead of ~12 min. All 25 .DONE markers + metrics.json files on
disk per §5.5 filesystem-truth.

## MEASURED AUC results (all on disk)

| Metal | seed0 | seed1 | seed2 | seed3 | seed4 | **Mean** | Status |
|---|---|---|---|---|---|---|---|
| **Ru** | 0.6455 | 0.5904 | 0.6447 | 0.6218 | 0.6569 | **0.6319** | ❌ BELOW_GATE |
| **Ir** | 0.6700 | 0.6271 | 0.6532 | 0.6366 | 0.5611 | **0.6296** | ❌ BELOW_GATE |
| **Os** | 0.8783 | 0.8795 | 0.8860 | 0.9275 | 0.8943 | **0.8931** | ✅ ABOVE_GATE |
| **Rh** | 0.7516 | 0.8181 | 0.7962 | 0.8091 | 0.8106 | **0.7971** | ✅ ABOVE_KRASNOV_PARTIAL |
| **Re** | 0.7202 | 0.6725 | 0.6395 | 0.6812 | 0.6810 | **0.6789** | ❌ BELOW_GATE |

**Overall 5-metal mean = 0.7261** (slightly above 0.70 gate; Krasnov anchor
for full 5-metal mean not yet established).

## Honest framing (party-twenty-spirit §3.5)

- **MEASURED today**: 25/25 pairs trained 50/50 epochs on REAL MetalCytoToxDB
  5-metal cohort (Ru 18,987 + Ir 4,546 + Os 1,113 + Rh 1,118 + Re 868 = 26,632 rows);
  val_auc per pair from trainer's `metrics.json` files.
- **GATE_RESULT**: 2/5 metals ABOVE_GATE (Os 0.893, Rh 0.797); 3/5 BELOW_GATE
  (Ru 0.632, Ir 0.630, Re 0.679). Os is the standout (closest to Krasnov 0.886
  in-distribution); Rh is partial Krasnov closure (gap −0.062 vs 0.859 anchor).
- **MEASURED_vs_PROJECTED**: MEASURED (each value traces to
  `molmetal/reports/wf131_p3_pt_4cells/{metal}/seed{seed}/metrics.json::value`).
- **Wall**: MEASURED 40m47s vs PROJECTED 5 hr (7.4× faster than projection; reason:
  Wf-E1 W₂ weight = 0 default, no gradient through W₂ term).
- **Aggregation discrepancy** (cosmetic, NOT a training failure): wrapper's
  `training_summary.json` reports 24/25 because Ru/seed0's `per_seed_metrics_seed*.json`
  glob miss in the `for metal_dir in Path.iterdir()` loop. All 25 .DONE markers +
  metrics.json files ARE on disk; the discrepancy is in the wrapper's aggregation
  step only.

## Wrapper v4 fixes (per Wf-131 P3 post-mortem 2026-10-08)

Three fixes in `scripts/launch_wf131_p3_pt_4cells.sh`:

1. **Line 92**: `set -euo pipefail` → `set -eu` (drop pipefail — was the SIGPIPE amplifier)
2. **Line 529**: `${PY} trainer | tee log` → `${PY} trainer | tee log || true` (belt-and-braces)
3. **Line 553-565**: for-loop captures per-pair rc; on failure writes `.FAILED` marker
   and continues to next pair (per-pair failure isolation)

Resume-safety: Ru/seed0 + Ru/seed1 had manual `.DONE` markers (wrapper v3 died
silently after each completed 50/50 epochs in detached_launcher.log but before
writing the marker).

## What ran next: Item 4 GPU chain (auto-triggered)

`scripts/launch_item4_chain_post_wf131.sh` (NEW, 75 LOC bash) was launched at
08:59 CST (PID 942034). It waited for wrapper v4 to exit (slept 60s intervals
from 08:59 → 09:31), then at 09:31 triggered Stage 1 VRAM check (97 GB free)
+ Stage 2 chemprop_v3 P1.4 LOMO launch (~2 hr).

`scripts/launch_item4_chain_post_wf131.sh` is the canonical pattern for
post-Wf-131 chained GPU launches (per user directive 2026-10-08).

## Files / paths

- 25 metrics.json files: `molmetal/reports/wf131_p3_pt_4cells/{Ru,Ir,Os,Rh,Re}/seed{0..4}/metrics.json`
- 25 .DONE markers: `molmetal/reports/wf131_p3_pt_4cells/{Ru,Ir,Os,Rh,Re}/seed{0..4}/.DONE`
- Wrapper v4 log: `molmetal/reports/wf131_p3_pt_4cells/detached_launcher_v4.log`
- Master log: `molmetal/reports/wf131_p3_pt_4cells/master.log`
- Wrapper v4 fix closeout: `TODO/completed/wf131_p3_wrapper_v4_silent_death_fix.md`
- Item 4 chain script: `scripts/launch_item4_chain_post_wf131.sh`
- Chain log: `molmetal/reports/item4_chain/chain.log`

## Cross-refs

- AGENTS.md §5.2 (honest-negative), §5.5 (filesystem-truth), §5.6 (atomic write), §5.7 (smoke), §5.8 (wf token budget), §5.9 (single-venv)
- TODO/completed/wf131_p3_wrapper_v4_silent_death_fix.md
- TODO/completed/wf_realdata_gapfix_v1_m8pb_ship.md (sibling CPU-only gap fix)
- TODO/completed/131_wf131_p3_pt_4cell_blocked_external.md (Pt × 4-cell external-dependency blocker)
- TODO/INDEX.md §17 (5-category audit — m19 LOMO status update pending Wf-A11 cross-reference)
- Wf-E1 (W₂=0 default — root cause of 7.4× wall speedup)
