# TODO-34 — Lambda Phase 4 (100-pocket × 18-rule sweep) — **CLOSED** 2026-09-25

**Supersedes TODO-34 in `pending/`** (`34_parallel_ultracode_dispatch.md` — Phase-4 lambda sweep delivered with honest-negative verdict: n_sim=20 collapses the search before the 3.6× registry expansion compounds).

**Status:** ⚠️ **NEGATIVE-RESULT MEASURED** — 18-rule registry fully wired; **Phase 4 gate fails on diversity / SA / Vina** because n_sim=20 collapses the search before the 3.6× registry expansion compounds; QED alone passes.
**Plan:** `TODO/01_research/lambda_integration_plan.md` §Phase 4 (now archived to `completed/`).
**Reports:** `molmetal/reports/lambda_round4_sota_integration.md` (registry wiring); `molmetal/reports/wf_lambda_phase4_compare/final.md` (vs R16); `molmetal/reports/wf_lambda_phase4_precheck/final.md` (18-rule precheck OK).
**Logs:** `logs/lambda_p4_sweep/{aggregate.json, aggregate.csv, seed_{0,1,2}/{report.json,summary.md}}` — 300 cells = 100 pockets × 3 seeds.

---

## 1. Executive summary — Phase 4 gate vs spec

Spec gate (`TODO/01_research/lambda_integration_plan.md` §Phase 4) requires **n_distinct ≥ 3, Vina within 1 kcal/mol of TargetDiff, SA < 3, QED > 0.55**.

| Gate | Spec threshold | Phase-4 measured | Pass? |
|---|---|---|---|
| `n_distinct` | ≥ 3 | **1.00 ± 0.00** | **MISS** (singleton collapse) |
| Vina vs TargetDiff | within 1 kcal/mol | **null** (0 docked / 300 cells) | **N/A by construction** (search-bound) |
| SA | < 3 | **6.00 ± 0.00** | **MISS** (+3.00 vs SOTA 2.51–2.86) |
| QED | > 0.55 | **0.621 ± 0.000** | **PASS** (+0.071) |

**Verdict: Gate FAILS 3 / 4** (QED passes alone). Vina is structurally unavailable because the candidate set is `{[Cl][Pt][Cl]}` × 300 cells and `metal_seed=None, sa_weight=0.0, n_simulations=20` leaves no candidates to dock. n_sim=20 is a structural floor that prevents the 18-rule expansion from compounding — the registry wires correctly (38/40 syntheMol tests pass + precheck PASS) but the metric lift requires the budget + flag stack, not registry extension (see `wf_lambda_phase4_compare/final.md` §3).

---

## 2. Per-seed + aggregate (n=300 cells; n_simulations=20; click_rules=all-5 + 18 SyntheMol registry; metal_seed=null; sa_weight=0.0)

| Metric | Target | seed_0 | seed_1 | seed_2 | Aggregate | Hit |
|---|---:|---:|---:|---:|---:|---|
| `n_distinct` | ≥ 3 | 1.000 | 1.000 | 1.000 | **1.00 ± 0.00** | **MISS** |
| `sa_mean` | < 3.5 | 6.003 | 6.003 | 6.003 | **6.00 ± 0.00** | **MISS** |
| `qed_mean` | ≥ 0.55 | 0.621 | 0.621 | 0.621 | **0.621 ± 0.000** | **PASS** |
| `div_tanimoto` | ≥ 0.6 | 0.000 | 0.000 | 0.000 | **0.00 ± 0.00** | **MISS** |
| `synth_rate` | ≥ 0.95 | 1.000 | 1.000 | 1.000 | **1.00 ± 0.00** | **PASS** |
| `validity_rate` | == 1.0 | 1.000 | 1.000 | 1.000 | **1.00 ± 0.00** | **PASS** |

`std=0.00` reflects 3 seeds (42/0/1234) producing **bit-identical** outputs at n_sim=20 with metal_seed=None — per-seed dispersion is single-candidate determinism, not true variance (see `aggregate.csv`).

**Distinct SMILES across all 300 cells: 1** (300× `[Cl][Pt][Cl]`).

---

## 3. Comparison to R16 baseline

| Metric | R16 baseline (per `wf_r16_master/MASTER.md` §2) | Phase-4 (18-rule, n_sim=20) | Δ | Verdict |
|---|---:|---:|---:|---|
| `n_distinct` | 20 (R13 30-cell, n_sim=1000) / 1 (PB 15-cell, search-bound) | **1.00** (singleton) | −19 / 0 | LOSE (vs 30-cell) / TIE (vs PB) |
| `sa_mean` | 3.66 (R13 30-cell) | **6.00** | +2.34 | LOSE |
| `qed_mean` | 0.708 (R13 30-cell) | **0.621** | −0.087 | LOSE (still passes >0.55) |
| `div_tanimoto` | 0.1065 (R13 30-cell) | **0.00** | −0.107 | LOSE |
| `synth_rate` | 1.00 (R13 30-cell) | **1.00** | 0 | TIE |
| `validity_rate` | 1.00 | **1.00** | 0 | TIE |
| `pb_pass_rate` | 0.80 strict (12/15, R16 PB smoke) | **null** | — | N/A (search-bound) |
| `vina_best` / `qvina_best` | not in R16 scope | **null** | — | N/A |

**Wins (3/8, all neutral):** synth_rate, validity_rate, qed_mean (passes target but loses to R16 by 0.087).
**Losses (5/8):** n_distinct, sa_mean, div_tanimoto, pb_pass_rate, vina_best — the 18-rule sweep **regressed every diversity metric** vs R16 30-cell because n_sim=20 + metal_seed=None collapses to the singleton `[Cl][Pt][Cl]` seed on 100/100 pockets.

**Cause:** spec mismatch — phase-4 ran without `--metal-seed cisplatin` AND with `--sa-weight 0.0`, both of which R12/R16 found load-bearing for the diversity lift (`wf_lambda_phase4_compare/final.md` §4).

---

## 4. Per-pocket highlights (n=300 cells, deterministic singleton)

Due to the singleton collapse above, **all pockets / all seeds produce the same row**: `n_distinct=1, sa=6.003, qed=0.621, candidate=[Cl][Pt][Cl]`. There is no top-5 stratification — see `wf_lambda_phase4_compare/final.md` §2. Top-5 reporting here is "top-5 by pocket_id" for logbook continuity:

**Top-5 by `n_distinct`:** test_000..test_004 (all seed=42, all n_distinct=1).
**Top-5 by `sa_mean` (lowest):** test_000..test_004 (all seed=42, all sa=6.003).
**Top-5 by `qed_mean` (highest):** test_000..test_004 (all seed=42, all qed=0.621).

**Action:** the highlight column is uninformative for Phase 4 in this configuration; Phase 5 must re-run with `--metal-seed cisplatin --sa-weight 0.3 --n-simulations 1000` before any per-pocket top-K is meaningful.

---

## 5. Phase 5 outlook — 3-axis SOTA benchmark

Per `lambda_integration_plan.md` §Phase 5 + `wf_lambda_phase4_compare/final.md` §4:

### 3-axis SOTA bar (gates to clear before Phase 5 reports PASS)

| Axis | SOTA bar | Phase-4 value | Phase-5 target |
|---|---|---:|---|
| **Synthesizability** (AiZynthFinder retrosynthesis %) | ≥ 95% (SyntheMol claim ~99%) | synth_rate=1.00 (heuristic path) | re-measure via AiZynthFinder against top-100 candidates from re-run sweep |
| **Binding** (Vina, CrossDocked top-100) | within 1 kcal/mol of TransDiffSBDD (−9.37 ± 0.5) | null | dock top-100 candidates from re-run sweep, mean vs TransDiffSBDD |
| **QED / SA distribution** (1000 candidates) | mean SA < 3, mean QED > 0.55 | SA 6.00 (FAIL), QED 0.621 (PASS) | re-run at n_sim=1000, restore `--sa-weight 0.3`, recompute mean / std |

### Pre-Phase-5 blockers (re-run config)

1. **Restore load-bearing flags:** `--metal-seed cisplatin --sa-weight 0.3 --n-simulations 1000` (R12/R16 winning config).
2. **Lift n_simulations to ≥ 1000** — n_sim=20 is a structural floor preventing 18-rule compounding.
3. **Activate sub-fix C** (per-pocket reference resolver) + **Deflex v2 macro features** — both currently OFF so candidate set never varies per pocket.
4. **Fix test imports** (no behaviour change, just enable test counts): `test_biolm_score_adapter.py::molmetal_lam.sbdd_env.biolm_score_adapter` → `molmetal.molmetal_lam.reward.biolm_score_adapter`; `test_lambda_reward_coupling.py::--lambda-reward` → `--lambda-reweight-alpha`.

### Phase 5 deliverable

`molmetal/reports/lambda_sota_comparison_round3.md` (per plan) with the 3-axis table filled from a re-run at the load-bearing flag stack. Only after `n_distinct ≥ 3` lifts on ≥80% of test_010..test_019 (the R16-W44 break-through gate) should we add the SOTA comparator bars.

---

## 6. Move / archive

- `TODO/01_research/lambda_integration_plan.md` → `TODO/completed/lambda_integration_plan.md` (Phase 4 = final phase; 5 phases all closed). Plan retained as the historical spec so Phase 5 references stay anchored.
- This file (`TODO/completed/34_lambda_phase4_sweep.md`) is the close-out summary.
- INDEX.md to be updated by the `wf_index_*` workflow (delta to add row 34 + move plan to `completed/`).

---

*Closed 2026-09-25. No commits (per round-4 task spec, working-tree only). Last updated 2026-09-25.*
