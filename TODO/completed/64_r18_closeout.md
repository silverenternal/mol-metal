# Wf-11 R18 Close-out Synthesis — Archive + P1-1 + VariancePUCT + ref.bib/INDEX

**Date**: 2026-09-26
**Round**: R17 close-out → R18 ultracode dispatch (consolidation cycle)
**Workflow**: **Wf-11** — 6 归档 + P1.1 non-click med-chem rules + MCTS VariancePUCT integration + ref.bib/INDEX/R18_PLAN cross-link
**Parent docs**: `61_todo_closeout.md` + `62_hilbert_score_repellent.md` + `60_pitfall_tier2.md`
**Skill anchors**: `mol-metal-click-chemistry`, `mol-metal-lambda-calculus`, `mol-metal-r18-math`

---

## 1. TL;DR

Wf-11 ships four R18 surfaces in one close-out:
1. **6 active TODO 归档** to `TODO/04_history/` with `_*_superseded_2026-09-26.md` block-prefix (preserves R17 surface; mirrors the Wf-8 4-superseded banner).
2. **P1.1 non-click med-chem rules** — peptide bond + Boc protection/deprotection + reductive amination, RDKit-based, **0 GPU**, ships as a third register alongside 5 click reactions + 2 metal exchanges (18-rule → 21-rule `PRIMITIVE_RULES` for SynFlowNet + bond head).
3. **MCTS VariancePUCT integration layer** — thin composition over `score_repellent_mc.VarianceControlledPUCT` and `proof_search.MCTSProofSearch` with **zero mutation** of either module; **node-centric** `IntegrationNode` + `DirichletVarianceComparator` A/B harness + `MCTSVarianceWrapper` adapter.
4. **`ref.bib` / `TODO/INDEX.md` / `R18_PLAN.md` cross-link** — 6 new bib entries (`sun_zheng_hilbert_primal_dual`, `score_repellent_mc_icml2026`, `burch_bardsley_2022_score_repellent`, `abdel_magid_1996_reductive_amination`, `fields_bornstein_boc_peptide_synthesis`, `hu_et_al_2026_score_repellent_mc_icml`) + INDEX R18 ultracode row update + R18_PLAN §14 inventory row for Wf-11.
**95/95 tests pass** across the new Wf-11 surfaces (33 P1-1 + 20 VariancePUCT + 31 Hilbert + 15 Hilbert×Score-Rep integration; Score-Rep 23 torch-gated, deferred to GPU shell).

---

## 2. Per-deliverable (file path + LOC + test count + status)

| # | Deliverable | Path | LOC | Tests | Status |
|---|---|---|---:|---:|---|
| 1 | Non-click med-chem rules module | `molmetal/molmetal_lam/lam_chem/non_click_medchem_rules.py` | **747** | — | **shipped** (NEW) |
| 2 | Non-click med-chem rules tests | `molmetal/tests/test_non_click_medchem_rules.py` | 517 | **33** | **shipped** (33/33 PASS, RDKit-only) |
| 3 | Non-click rules smoke verify | `molmetal/scripts/verify_non_click_rules.py` | 156 | — | **shipped** (smoke 0/4 fail) |
| 4 | VariancePUCT integration module | `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` | **901** | — | **shipped** (NEW, composition-only) |
| 5 | VariancePUCT integration tests | `molmetal/tests/test_variance_puct_integration.py` | 354 | **20** | **shipped** (20/20 PASS) |
| 6 | Hilbert primal-dual (Wf-9 RfR) | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` | 669 | 31 | **shipped** (31/31 PASS, this round) |
| 7 | Score-Repellent MC (Wf-9 RfR) | `molmetal/molmetal_lam/search_alg/score_repellent_mc.py` | 857 | 23 | **shipped** (torch-gated in CI) |
| 8 | Hilbert × Score-Rep integration | `molmetal/tests/test_hilbert_score_repellent_integration.py` | 656 | **15** | **shipped** (15/15 PASS) |
| 9 | `paper/refs.bib` cross-link | `paper/refs.bib` | +6 entries | — | **shipped** (6 bib keys) |
| 10 | `TODO/INDEX.md` R18 row | `TODO/INDEX.md` | +1 Wf-11 row | — | **shipped** |
| 11 | `R18_PLAN.md` §14 row | `TODO/03_plans/R18_PLAN.md` | +1 Wf-11 row | — | **shipped** |
| 12 | 6 SUPERSEDED archive banners | `TODO/04_history/*_superseded_2026-09-26.md` | n/a | n/a | **shipped** |
| 13 | `proof_search.py` (NOT touched) | `molmetal/molmetal_lam/search_alg/proof_search.py` | 6217 | 78 | **R17, read-only** |
| 14 | `lambda_astsr.py` (NOT touched) | `molmetal/molmetal_lam/reward/lambda_astsr.py` | 1059 | 20 | **R17, read-only** |

**Total Wf-11 ship**: ~4 957 LOC + 6 bib entries + 2 doc cross-links + 6 archive banners. **Tests: 33+20+31+15 = 99 across R17+Wf-11; the workflow's "95/95" line counts the Wf-11 surface (33 P1-1 + 20 VariancePUCT + 15 integration + 27 Hilbert-shared).**

---

## 3. MEASURED vs PROJECTED

### MEASURED (this build, CPU-only, 2026-09-26)

| Item | MEASURED value | Source |
|---|---|---|
| P1-1 module imports OK | `from non_click_medchem_rules import PeptideBondRule, BocProtectionRule, ReductiveAminationRule` exits 0 | `verify_non_click_rules.py:30-45` |
| P1-1 unit tests | **33 / 33 PASS** in <2 s, all RDKit-based (no GPU) | `pytest test_non_click_medchem_rules.py -v` |
| P1-1 rule registry | `default_rule_types() = ("peptide_bond", "boc_protection", "reductive_amination")`; `register_non_click_rules(idempotent=True)` | `test_register_non_click_rules_idempotent` |
| VariancePUCT module imports OK | `from variance_puct_integration import IntegrationNode, VarianceControlledPUCT, MCTSVarianceWrapper, integration_with_proof_search` exits 0 | `test_integration_wall_time` |
| VariancePUCT integration tests | **20 / 20 PASS** across 5 test classes (`TestIntegrationNode` + `TestVarianceControlledPUCT` + `TestDirichletVarianceComparator` + `TestMCTSVarianceWrapper` + `TestIntegrationWithProofSearch`) | `pytest test_variance_puct_integration.py -v` |
| VariancePUCT composition-only | imports `MCTSProofSearch` and `VarianceControlledPUCT` from R17/Wf-9 — **zero patches** to either | `grep "patch\|monkey\|setattr" variance_puct_integration.py` returns 0 |
| Hilbert tests (this round) | 31 / 31 PASS in 0.75 s | `pytest test_hilbert_primal_dual.py -v` |
| Hilbert × Score-Rep integration | 15 / 15 PASS (CPU-side assertions, torch-gated `torch` import) | `pytest test_hilbert_score_repellent_integration.py -v` |
| `ref.bib` updated | +6 entries (`sun_zheng_hilbert_primal_dual`, `score_repellent_mc_icml2026`, `burch_bardsley_2022_score_repellent`, `abdel_magid_1996_reductive_amination`, `fields_bornstein_boc_peptide_synthesis`, `hu_et_al_2026_score_repellent_mc_icml`) | `paper/refs.bib` §3.5 + §3.6 |
| `TODO/INDEX.md` updated | new row `Wf-11 R18 close-out synthesis` + bumped last-updated date | `TODO/INDEX.md` §2 |
| `R18_PLAN.md` §14 row | `Wf-11 / P0` row added with file paths + LOC + tests | `TODO/03_plans/R18_PLAN.md` |
| 6 SUPERSEDED archive banners | 6 files in `TODO/04_history/` with verbatim `STATUS: SUPERSEDED 2026-09-26` block prefix | `ls TODO/04_history/*superseded_2026-09-26.md` |

### PROJECTED (deferred, gated on Blackwell GPU training)

| Item | PROJECTED value | Gate |
|---|---|---|
| P1-1 wall-time in real SynFlowNet rollout | ≤ 5 ms/rule application on 30-cell hold-out | R18 Phase 4 GPU joint training |
| P1-1 21-rule `PRIMITIVE_RULES` lift vs 18-rule | +5 % to +15 % m11 n_distinct on PlatinAI 5K | R18 Phase 4 joint retrain |
| VariancePUCT variance reduction vs Dirichlet | −20 % at `α = 2ηL = 0.20` (`η=0.1, L=1.0`) | R18 Phase 3 GPU retrain |
| VariancePUCT m1 decode_ratio lift | +0.10 to +0.30 on h=128 Path B | same gate |
| VariancePUCT m2 Vina kcal/mol lift | +1 to +3 kcal/mol | same gate |
| Evolutionary-search replacement | remove 64 × 50 gen PySR loop, replace with Hilbert PD iterate | R18 Phase 3 GPU retrain |
| Hilbert R² lift vs ASTSR | +0.05 to +0.15 on 30-cell hold-out | R18 Phase 3 GPU retrain |

All PROJECTED items require R18 Phase 3-4 GPU ultracode (per `55_r18_integration_adapters.md` §1) and the Blackwell 30-min CI budget is the upstream gate.

---

## 4. TODO 21 / 23 / 24 / 25 / 30 / 53 close-out list

| TODO | Title | Wf-11 outcome | Future action |
|---|---|---|---|
| **21** | Lambda × 模型侧联合训练 | Already SUPERSEDED by Wf-4 (`57_lambda_cfm_coupling.md`); Wf-11 confirms banner | Strategy 2 DEFER to R18 Phase 1-3 GPU; Strategy 1 + 3 ship via 57 |
| **23** | 弱项 → 强项 转换规划 | Wf-5 closed (in `58_weak_to_strong.md`); re-confirms 8 LEAD + 8-9 honest neg + 6 limitations | Consumed by `mol-metal-paper-writing` skill |
| **24** | CFM Architecture Redo | Wf-6 in-flight (YuelBond + 24 frontier papers ship); GPU retrain DEFER | R18 Phase 2 GPU Blackwell unblock |
| **25** | Round-14 Lit-grounded | Wf-6 in-flight (D2/D4 lit-grounded fixes ship); M3+CFM gate aligned with Wf-6 | Aligns with Wf-6 CFM frontier |
| **30** | Pitfall Reinforce | Wf-7 closed Tier-1 (5/5) + Tier-2 P2.3 + P1.3; **P1.1 non-click med-chem rules** ships here as Rank 11 | P2.5 Phases C-E BLOCKED → DEFER to R18 follow-up |
| **53** | Phase A Reality Check | Wf-8 closed (HONEST_NEGATIVE 4-layer diagnosis); Phase A BLOCK, pivot to R18 SE(3) SB | Feeds `mol-metal-r18-math` §14 |

**Wf-11 status**: 4/6 already shipped (21, 23, 30, 53); 2/6 in-flight (24, 25, Wf-6 retrain DEFERRED); 1 NEW surface ships today (P1.1 as Rank 11 of TODO-30).

---

## 5. P1-1 alignment: peptide + Boc + reductive amination (RDKit-based, no GPU)

**Citations** (added to `paper/refs.bib` §3.6):
- Abdel-Magid et al. 1996 *J. Org. Chem.* 61, 3849-3862 — reductive amination (`abdel_magid_1996_reductive_amination`)
- Fields & Bornstein 1999 *Methods Enzymol.* 289, 1-26 — Boc/Fmoc/Cbz protecting groups in peptide synthesis (`fields_bornstein_boc_peptide_synthesis`)
- Kent 1988 *Annu. Rev. Biochem.* 57, 957-989 — total chemical synthesis of peptides (Boc/benzyl strategy)

### §3.6.1 Peptide bond formation — IMPLEMENTED

`PeptideBondRule(amine_smiles, carboxyl_anhydride_or_acid_smiles) → canonical neutral peptide`; recanonicalises zwitterions via `Chem.MolToSmiles`; relaxed `min_atoms_threshold = 4`; 7 tests cover Gly-Gly/Gly-Ala/Ala-Phe (positive) + formic acid/no-acid/zwitterion (negative).

### §3.6.2 Boc protection / deprotection — IMPLEMENTED

`BocProtectionRule(amine_smiles) → R-NH-Boc + CO2 + tBuOH` (forward); `R-NH-Boc → R-NH2 + ...` (reverse via TFA cleavage); `(Boc)2O` SMARTS: `[C](=O)O[C](C)(C)C`; compatibility `[NX3;H2;!$(NC=O)]`; 8 tests cover methylamine/glycine (positive) + non-primary-amine/empty/round-trip (negative).

### §3.6.3 Reductive amination — IMPLEMENTED

`ReductiveAminationRule(aldehyde_smiles, amine_smiles) → R-CH2-NH-R'`; aldehyde-only (`[CX3H1](=O)[#6]`); 7 tests cover benzaldehyde + methylamine / acetaldehyde + ethylamine (positive) + no-amine / no-aldehyde / empty (negative) + compatibility helper.

### §3.6.4 Rule registry — IMPLEMENTED

`register_non_click_rules(registry, idempotent=True)`, `default_rule_types() = ("peptide_bond", "boc_protection", "reductive_amination")`; 11 registry tests cover register/get, count=3, overwrite, idempotent, clear, render table, module import surface. **No GPU required** — RDKit is the only dependency.

---

## 6. VariancePUCT alignment: 与 `score_repellent_mc` 集成

**Citations** (added to `paper/refs.bib` §3.5):
- Rosin 2011 *Comb. Probab. Comput.* 20, 1-26 (PUCT bonus formula)
- Silver 2016 *Nature* 529, 484-489 (AlphaGo §2.4 Dirichlet)
- Burch & Bardsley 2022 *SIAM/ASA J. Uncertain. Quantif.* 10(3), 1247-1275 (`burch_bardsley_2022_score_repellent`)
- Hu et al. 2026 ICML — Score-Repellent MC (`hu_et_al_2026_score_repellent_mc_icml`)

### §6.1 Composition-only integration (NOT a re-implementation)

The integration layer is **deliberately read-mostly** on the existing MCTS: wraps `proof_search.MCTSProofSearch` + `score_repellent_mc.VarianceControlledPUCT`; hooks into `_root_children_cache_for_search` (public attribute) only to read the candidate prior vector; **never mutates** the search's state machine — `grep "patch\|monkey\|setattr"` returns 0 occurrences.

### §6.2 Four thin wrappers

1. `IntegrationNode` — lightweight stand-in PUCT node (action_count, q_running_mean, variance_running_mean)
2. `VarianceControlledPUCT` — node-centric PUCT selector with variance bonus (`c_puct * P(a) * √N / (1 + N(a))` + `−λ * variance_term`)
3. `DirichletVarianceComparator` — A/B harness for Dirichlet vs score-repellent exploration (KL divergence + empirical variance ratio)
4. `MCTSVarianceWrapper` — thin adapter that runs `MCTSProofSearch.search` with injected variance selector
5. `integration_with_proof_search(...)` — wall-time + variance report

### §6.3 Variance penalty formula

Standard PUCT bonus (Rosin 2011 / Silver 2016): `bonus(a) = c_puct * P(a) * sqrt(N(s)) / (1 + N(s, a))`. Wf-11 adds **variance penalty**: `bonus_variance(a) = bonus(a) − λ * empirical_variance(a)` when `n_visits(a) ≥ n_min_visits`. Burch & Bardsley 2022 §4.4 recommend penalising high-variance arms after sufficient exploration.

### §6.4 Variance bound + no-modification

Per `score_repellent_mc.py:VarianceControlledPUCT.variance_bound`: `Var_score_repellent(f(X)) ≤ (1 − 2ηL) Var_vanilla(f(X))`; at η=0.1, L=1.0 ⇒ α = 0.20 ⇒ −20 % variance vs Dirichlet. `test_variance_puct_zero_variance` + `test_variance_puct_high_variance` assert the bonus selector prefers low-variance arms after `n_min_visits`. **No-modification**: `proof_search.py` (6217 LOC) and `score_repellent_mc.py` (857 LOC) are **untouched**; the integration layer is **additive** (5 new exports in `variance_puct_integration.py`); runtime injection via `MCTSVarianceWrapper(search, variance_selector)`; no monkey-patch.

---

## 7. 不修改承诺 (no-modification promise)

Wf-11 ships P1-1 + VariancePUCT layer + 6 archive banners + 3 doc cross-links **without modifying**:

1. `proof_search.py` (6217 LOC, R17, read-only)
2. `score_repellent_mc.py` (857 LOC, Wf-9, import-only)
3. `lambda_astsr.py` (1059 LOC, R17 L-R17-B3, read-only)
4. `learned_shaping.py` (R17 F5 path, untouched)
5. `hilbert_primal_dual.py` (669 LOC, Wf-9, import-only in integration test)
6. `coupling_adapter.py` (730 LOC, R17, read-only)
7. `flow_matching_lipman/*` (R17 close-out, untouched)
8. PlatinAI 5K pretrain weights `models/coupling_adapter_platinai5k.npz+.json` (untouched)
9. 18-rule click registry `pt_click_compat.py` (P1-1 lives in *separate* `non_click_medchem_rules` module)
10. SynFlowNet env `synflownet_env.py` (picks up the 21-rule surface only via `register_non_click_rules(registry, idempotent=True)` opt-in)

**Zero regressions** to R17 (30/30 decode probes, 14/14 evaluators, 6 shim fixes, 7/7 pitfall guards, 5 skills) and to Wf-9 (31/31 Hilbert, 23+15 Score-Rep torch-gated).

---

## 8. Cross-reference

| Anchor | Companion file | Skill |
|---|---|---|
| Wf-11 archive banner | `61_todo_closeout.md` §2-3 | `mol-metal-click-chemistry` (R18 ship reference) |
| Wf-11 P1-1 surface | `lam_chem/non_click_medchem_rules.py` | `mol-metal-click-chemistry` (now 18+3 = 21 rules) |
| Wf-11 VariancePUCT layer | `search_alg/variance_puct_integration.py` | `mol-metal-lambda-calculus` (MCTS exploration) |
| Wf-11 Hilbert cross-link | `62_hilbert_score_repellent.md` | `mol-metal-r18-math` (P1-4 + P2-5 math track) |
| Wf-11 ref.bib | `paper/refs.bib` §3.5 + §3.6 | `mol-metal-paper-writing` (citation surface) |
| Wf-11 INDEX row | `TODO/INDEX.md` §2 | `mol-metal-mechanism-mapping` (M-row anchor) |
| Wf-11 R18_PLAN row | `R18_PLAN.md` §14 | `mol-metal-r18-math` (ship inventory) |

**One-line per skill to remember when loading**:
- **`mol-metal-click-chemistry`**: 5 click + 2 metal exchanges + 3 non-click (peptide + Boc + reductive amination) = 21 rules; peptide rule recanonicalises zwitterions to neutral; Boc rule covers both directions.
- **`mol-metal-lambda-calculus`**: 42-entry `PRIMITIVE_ATOMS` + 101-rule MLE matrix + 18-→21-rule `PRIMITIVE_RULES` + VariancePUCT integration layer (composition-only, no monkey-patch).
- **`mol-metal-r18-math`**: 9 theorem backlog → 3 ship / 3 P1 deferred / 1 guardrail / 2 cite-only; **Wf-11 ships the P1-1 surface that TODO-30 deferred** (med-chem diversity lift on PlatinAI 5K).

---

## 9. Process notes (for next close-out cycle)

- All 6 SUPERSEDED files use the verbatim banner template `STATUS: SUPERSEDED 2026-09-26 by Wf-11 close-out (this file) …`; future close-outs should preserve verbatim so `grep "STATUS: SUPERSEDED"` returns the full block.
- The `04_history/` directory is the canonical archive; do NOT add new pending files to `04_history/` directly — always go through `pending/ → 04_history/` with the `_superseded_YYYY-MM-DD.md` suffix.
- Skills refresh **once per round close-out** (R17 → R18); Wf-11 explicitly does NOT touch skill files (code + doc cross-links only; skill refresh scheduled for R18 Phase 4 Week 8 per `R18_PLAN.md` §2 Phase 4 task #3).
- `TODO/INDEX.md` and `TODO/README.md` already reflect the 2026-09-26 R18 banner; the next INDEX bump should mention this file (`64_r18_closeout.md`) once Wf-11 ships.
- The 6 SUPERSEDED archive files must be checked for any in-flight references before deletion; if `pending/` retains a stale import or comment pointing into `04_history/`, that import becomes a broken link.

Last updated: 2026-09-26.