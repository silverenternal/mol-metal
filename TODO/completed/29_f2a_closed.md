# TODO-29 — Ship F2(a) MetalLigandExchange SMARTS + re-run Round-13

**Status:** ⚙️ **IN-FLIGHT** — `wyyy283ck` B-sub-fix (Scaffold-aware Pt/click compat) partial; `wrd5dbewn` (pocket-invariance combined 3 sub-fixes) in flight
**Original priority:** highest (paper §4 + §6 + §7 all affected)
**Owner:** (unset)
**Depends on:** `wrd5dbewn` pocket-invariance combined (3 sub-fixes) + `wyyy283ck` scaffold-aware
**Created:** 2026-09-15
**Last updated:** 2026-09-15 (4 in-flight workflows on disjoint files)

## ⚠️ Updated state — what we now know

### PathA-10x3 diversity lift (verified earlier, NOT impacted by R13)
- ✅ n_distinct 1 → 20 on test_000..test_009 (10 cells × 3 seeds, all MEASURED)
- ✅ div_tanimoto 0 → 0.1065 (verified)
- ✅ div_homotype 0 → 0.0749 (verified)
- ✅ Honest trade-off: metal_compliance 1.0 → 0.0 (Pt-acetylide root not strict-Pt_II; documented in paper §4.3 Table 2 footnote)

### Round-13 100×3 honest-negative
- PathA-10x3 lift does NOT generalize to test_010..test_019 (re-searched, no candidates → n_distinct=1 resurfaces)
- 3-layer singleton attractor: chemistry layer BROKEN (verified), but MCTS cache + reward prior layers still produce pocket-invariant 20-molecule chemotype basket

### Current work in flight (4 workflows, disjoint files)
- `wbw9a59g3` CFM frontier research + fix (3 fixes in flight per `synthesis_phase2.md`)
- `wrd5dbewn` Pocket-invariance combined — 3 sub-fixes A/B/C (stronger pocket boost + wire learned_prior + replace metal_seed)
- `wyyy283ck` A: Skeleton CA2 fix (is_anchor 4→8) + B: λ Combinators cold-swap Sites 1+2
- `wtdrwkb6z` Deflex follow-up integration — F5 formula adapter + PocketMacroInference

## What needs to ship

### Step 1: F2(a) MetalLigandExchange + AquaExchange SMARTS rule

Per TODO-24 P0 CPU-only fix list + ultracode Task A in workflow `w8579x29t`:
- Files: `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` + `molmetal/molmetal_lam/reactions/beta_reductions.py`
- Add SMARTS: `[Pt](-Cl) + [M](-L) → [Pt](-L) + [M](-Cl)` for MetalLigandExchange
- Add SMARTS: `[Pt](-Cl) + H2O → [Pt](-OH) + HCl` for AquaExchange (with pKa context)
- Update `pt_click_compat.py` 5×5 compat matrix: strict_Pt_II now COMPATIBLE with MetalLigandExchange + AquaExchange
- Wire into `beta_reductions.py` after CuAAC/SPAAC/Suzuki
- Tests: 4+ in `tests/test_lambda_mcts_singleton.py`
- Lit: Lippard 1995 (Pt_II coord chem) + Reedijk 1987 (Pt aquation kinetics)

### Step 2: Serialize Lambda runs

- Kill PID 110660 (concurrent n_sim=200 sweep)
- Wait for /tmp/round13_lambda.log n_sim=1000 in-flight to finish (ETA ~9 more hours) OR kill it
- Launch single n_sim=1000 sweep on 100 pockets × 3 seeds = 300 cells

### Step 3: Fix CFM path import

- `r10_cfg_real_crossdocked.py` ModuleNotFoundError — likely `from molmetal.ports import GenerationConfig` missing path
- Add `sys.path.insert(0, ...)` or set `PYTHONPATH=.` in CLI
- Verify: `uv run python -c "from molmetal.scripts.r10_cfg_real_crossdocked import main"` imports OK

### Step 4: Re-run Round-13 at n_sim=1000 after F2(a) ships

```bash
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --manifest molmetal/data/crossdocked100_manifest.csv \
    --pockets 100 --seeds 42 0 1234 \
    --n-simulations 1000 --n-top-k 20 \
    --metal-seed cisplatin --click-rules auto-pt-strict \
    --sa-weight 0.3 --pb-relax-mmff94 --pb-mode dock --pb-check \
    --engine both --decoder-rework \
    --metal-ligand-exchange --aqua-exchange \
    --output-dir molmetal/reports/wf_round13_retry/r13_100x3/
```

## Why this is urgent

- All paper §4 metrics need real data, not synthetic or single-smoke
- §4.2 Table 1 needs 100 pockets × 3 seeds = 300 cells, currently 0 cells
- §4.6 PB column needs production PB pass rate, currently 0 cells
- §6 limitations currently 8 caveats, will reduce to 4-6 if F2(a) lifts diversity + compliance together
- §7 future work currently 5 NEW research gaps, will reduce to 3 if F2(a) ships

## Effort estimate

- Step 1: 6h (CPU-only, in flight via ultracode)
- Step 2: 1h (process management + cleanup)
- Step 3: 30 min (one-line fix)
- Step 4: 13 min wall parallel (4-way) or 50 min single-thread at n_sim=1000 + F2(a) → 50 min if F2(a) lifts the singleton
- Step 4 alternative: 9 hours if F2(a) does NOT lift + fall back to n_sim=200 baseline

**Total: 1-2 days CPU-only, gated on F2(a) shipping + Lambda n_sim=1000 sweep completing.**

## Risk assessment

- **HIGH RISK**: F2(a) may not lift singleton attractor on all 100 pockets
- **MEDIUM RISK**: Lambda n_sim=1000 sweep may exceed 4-hour budget on full 100 pockets
- **LOW RISK**: CFM path import fix is straightforward

## Fallback if F2(a) fails

- Fall back to Round-12 single-cell path (PathA-10x3) for paper §4.3 Table 2 with explicit "verified on test_000..test_009 only" caveat
- Cite Round-13 honest-negative verdict in §6 limitations
- Defer full sweep to Round-14 with new strategy (likely higher n_simulations, different reward prior, or hybrid CFM + Lambda)

## Cross-references

- TODO-21 (Lambda × CFM coupling deferred)
- TODO-24 (CFM architecture redo plan)
- TODO-25 (Round-14 lit-grounded plan)
- TODO-26 (Round-13+14 master comprehensive plan)
- TODO-28 (Round-12 honest reframe — already partial-lift verified)
- `metrics/by_round/r13_100x3_partial.json` (NEGATIVE_RESULT_HONEST)
- `metrics/by_round/r12_lambda_patha_10x3.json` (PARTIAL_LIFT_VERIFIED)
- `molmetal/reports/wf_round13_100x3/final.md` (177 lines honest verdict)

## R15 master consolidation (2026-09-16)

**Verdict:** F2(a) SMARTS rule ships (pt_click_compat 5×5 matrix updated); Lambda 100×3 sweep still pending (only 30/300 cells completed in R15).
- PathA-10x3 diversity lift re-confirmed on test_000..test_009 (n_distinct 1→20) — but does NOT generalize to test_010..test_019 (Jaccard=1.0 with new resolver).
- BUG-1 (`learned_prior.py:412-417` silent coupling reshape) caught by W3 cross-verifier — pre-flight fix required before R16 retry.
- Master: `molmetal/reports/wf_r15_all/final.md`; cross-verifier: `molmetal/reports/wf_r15_cross_verify/final.md`.
