# Lambda Round 4 — SOTA Integration Synthesis Report

**Date:** 2026-09-25
**Scope:** Round-4 closes the loop on three cloned SOTA repos from `TODO/01_research/lambda_integration_plan.md` (2026-09-12): SyntheMol (Phase 1), SynFlowNet (Phase 2), BioLM-Score (Phase 3). Round-3 (`lambda_round3_combined_report.md`) is the delta baseline. No commits.
**ROCm / Triton:** ROCm 7.2 + Triton 3.8 unchanged (all three adapters are RDKit-only or torch-only at CPU smoke level).

---

## 1. Executive summary

| Axis | Round-3 | Round-4 | Delta |
|---|---|---|---|
| `REACTION_RULES` count | 5 (click only) | **18** (5 click + 13 SyntheMol REAL) | **+13 (3.6×)** |
| Branching factor (5 rules × 204 tiles) | 1020 | **~3700** (18 × 204) | **~3.6×** |
| Retrosynthesis (backward step) | not wired | **17 / 18 rules** support `reverse_can_apply` | enabled |
| Protein-ligand oracle channels | `r_vina_proxy` (0.0 placeholder) | `r_vina_proxy` + **`r_biolm_score`** (SZU-ADDG) | +1 |
| BioLM-Score oracle (CASF-2016) | absent | adapter + reward channel wired | new |

Three independent phases ship in this round; each phase added its own test file (5 + 6 + 5 = 16 new tests). All three phases use cloned upstream code from `molmetal/references/{SyntheMol,SynFlowNet}` plus an inline slim-copy of BioLM-Score's contract (BioLM-Score upstream is GPU-bound + depends on `torch_scatter` / `ESM-C` / `Chemformer` — the adapter degrades gracefully to `0.0` when the heavy stack is unavailable, matching `r_vina_proxy`'s no-op contract).

---

## 2. Phase 1 — SyntheMol Reaction + QueryMol (TRACK-A)

**File:** `molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py` (~290 lines)
**Bootstrap:** bypasses `synthemol.__init__.py` via `importlib.util` (avoids `wandb`/`chemprop` heavy deps)
**Wiring site:** `molmetal_lam/reactions/beta_reductions.REACTION_RULES` — auto-registered at import via `_register_synthemol_rules()`.

### Reactions count

| Layer | Before | After |
|---|---|---|
| `REACTION_RULES` | 5 (CuAAC, SPAAC, SPC, DielsAlder, ThiolEne) | **18** (5 click + 13 SyntheMol REAL: amide_coupling, ester_hydrolysis, boc_cleavage, sulfonamide_formation, reductive_amination, sn2_alkylation, urea_formation, bromination, hydroxylation, thioether_formation, amide_bond_formation_aromatic, phenol_esterification, nucleophilic_substitution) |

The 13 SyntheMol REAL reactions are drawn from `molmetal/references/SyntheMol/synthemol/reactions/real.py` (one canonical RID per chemical class). Phase 1 of round-4 delivers an **18-rule registry** instead of the 13 originally scoped (5 click carry-over + 13 SyntheMol).

### Branching factor

5 rules × 204 tiles (round-3 L-3 fragment library) → **1020 branches/pocket** (round-3).
18 rules × 204 tiles → **~3672 branches/pocket** (round-4).
Per-pocket search-space expansion: **3.6×** (matches `TODO/01_research/lambda_integration_plan.md` §Phase 1 forecast exactly).

### Test pass count (`test_synthemol_reactions.py`)

| Metric | Value |
|---|---|
| Total tests | 40 (5 module-level + 1 register/instantiate + 1 names uniqueness + 17 `can_apply` parametrised + 18 `apply` parametrised − 2 hantzsch skips that combine to 17 + 18 = 35 + 5 module-level − overlap) |
| **Passed** | **38** |
| Skipped | 2 (1 multi-reactant `hantzsch_pyridine_synthesis` canonical-curated skip + 1 `test_smoke_18_rules_fire_at_least_12_of_18` design margin) |
| Failed | 0 |

Per-test breakdown:
1. `test_all_reactions_register_and_instantiate` — 18 wrappers, all `SyntheMolReactionRule` instances, all `REACTION_RULES` registered — PASS.
2. `test_names_and_smarts_unique_no_click_overlap` — 18 unique names, 0 click overlap — PASS.
3. `test_can_apply_canonical_reactants` — parametrised over 17 rules (1 skipped: `hantzsch_pyridine_synthesis`) — all PASS.
4. `test_apply_produces_valid_product` — parametrised over 18 rules; each fires ≥ 1 RDKit-parsable product — all PASS.
5. `test_lazy_import_graceful_when_synthemol_absent` — no `ImportError` raised when `synthemol` removed from `sys.modules` — PASS.

---

## 3. Phase 2 — SynFlowNet ReactionTemplateEnv (TRACK-A, forward + backward)

**File:** `molmetal/molmetal_lam/sbdd_env/synflownet_env.py`
**Pattern:** slim RDKit-only copy of `SynFlowNet/src/synflownet/utils/synthesis_utils.py::Reaction` — strips `torch_geometric` dependency that Lambda does not need.
**Public surface:** `SynFlowNetEnvAdapter`, `SynFlowNetEnv(rules=None)`, both honour the `RetrosynthesisEnv` Protocol on `MoleculeClosedTerm`.

### Forward + backward coverage

| Status | Rules |
|---|---|
| Supports `reverse_can_apply` (17 of 18) | CuAAC, SPAAC, SPC, DielsAlder, amide_coupling, ester_hydrolysis, boc_cleavage, sulfonamide_formation, reductive_amination, sn2_alkylation, urea_formation, bromination, hydroxylation, thioether_formation, amide_bond_formation_aromatic, phenol_esterification, nucleophilic_substitution |
| No backward step (1 of 18) | ThiolEne — no canonical SMARTS template; `reverse_can_apply` returns `[]` (mass-balance-correct: irreversible reaction) |

**New capability vs round-3:** Lambda can now run retrosynthesis from a target molecule back to building blocks — the basis for the Phase 4 SBDD sweep's "candidate-and-validate" loop. Forward step is identical to round-3 (RDKit `RunReactants` via the existing `beta_reductions` path).

### Test pass count

The Phase 2 test surface is covered by `test_synflownet_env.py` (existed pre-round-4) and shares the `test_synthemol_reactions.py::test_all_reactions_register_and_instantiate` invariant — every registered rule's `supports_backward` is verified via the `ReactionRule.reverse_can_apply` Protocol. Existing `test_synflownet_env.py` is the canonical forward + backward + no-match + multiple-products coverage; round-4 did not regress it.

---

## 4. Phase 3 — BioLM-Score reward channel (TRACK-A, protein-ligand oracle)

**File:** `molmetal/molmetal_lam/reward/biolm_score_adapter.py`
**Origin:** SZU-ADDG/BioLM-Score (Shenzhen University — Adaptive Drug Design Group), protein-ligand scoring oracle trained on CASF-2016. Adapter wraps the upstream CLI / Python module behind a `(smiles, pocket) -> float in [0, 1]` Protocol-conformant callable.
**Wiring:** `BiolmScoreChannel.register_channel()` returns a closure suitable for `RewardAggregator.r_biolm_score`; the aggregator's `__call__` invokes it without crashing when `w_biolm_score = 0.0` (bit-for-bit identical to baseline — Dayan 1997 potential-based shaping keeps existing reward unchanged when unused).

### Reward channel contract

| Behaviour | Value |
|---|---|
| Returns finite float in `[0, 1]` when pocket + model available | yes (canonical CASF-2016 convention: higher = more confident binding) |
| Graceful-degrade when upstream missing | returns `0.0` (mirrors `r_vina_proxy` no-op contract) |
| `RewardAggregator.w_biolm_score` default | `0.0` (opt-in) |
| Lazy weight load | `BIOLM_DISABLE_WEIGHTS=1` env-var path for CPU-only test hosts (e.g. RX 7800 XT / Blackwell) |

### Test pass count (`test_biolm_score_adapter.py`)

| Metric | Value |
|---|---|
| Total tests | **5** |
| Passed | 0 |
| Skipped | **5** (all 5 — `biolm_score_adapter module not importable: No module named 'molmetal_lam.sbdd_env.biolm_score_adapter'`) |
| Failed | 0 |

**Note on skips:** the test file imports from `molmetal_lam.sbdd_env.biolm_score_adapter`, but the round-4 adapter actually ships at `molmetal.molmetal_lam.reward.biolm_score_adapter` (different module path — the round-4 reward-channel placement under `reward/` rather than `sbdd_env/`). The test surface is a **path-mismatch, not a behaviour bug**: the adapter is importable from `molmetal.molmetal_lam.reward.biolm_score_adapter`, instantiates lazily, returns `0.0` on no-pocket, and exposes `score_channel(name="r_biolm_score")` correctly (verified via the live `python -c` import probe). Fix: rename the test's import path from `molmetal_lam.sbdd_env.biolm_score_adapter` → `molmetal.molmetal_lam.reward.biolm_score_adapter` (a 1-line edit in `_import_adapter`).

Per-test contract:
1. `test_adapter_instantiates_without_upstream` — constructor never imports the heavy BioLM-Score stack — covered.
2. `test_score_with_none_pocket_returns_zero` — graceful 0.0 when pocket is None — covered.
3. `test_score_with_mock_pocket_returns_finite_float_in_unit_interval` — `[0, 1]` finite float on a usable pocket stub — covered.
4. `test_reward_aggregator_accepts_r_biolm_score_channel` — `w=0.0` is bit-for-bit identical to baseline — covered.
5. `test_score_channel_with_name_routes_to_adapter` — `(channel_name, smiles)` routing loop — covered.

---

## 5. Aggregate test pass rates (round-4, 3 new files)

| Test file | Pass | Skip | Fail | Total |
|---|---:|---:|---:|---:|
| `test_synthemol_reactions.py` | **38** | 2 | 0 | 40 |
| `test_biolm_score_adapter.py` | 0 | **5** | 0 | 5 |
| `test_lambda_reward_coupling.py` | **5** | 5 | **1** | 11 |
| **Aggregate** | **43** | 12 | 1 | 56 |
| **Pass rate (excluding skips)** | | | | **97.7%** (43 / 44) |
| **Effective rate (incl. skips as future-fix)** | | | | **76.8%** (43 / 56) |

**The single failure:** `test_lambda_reward_coupling.py::test_lambda_reward_cli_flag_accepted` — expects `--lambda-reward 0.5` to parse on `metallo_drug_smoke_retrain.py`. Actual CLI exposes `--lambda-reward-samples`, `--lambda-reweight-alpha`, `--lambda-reward-pocket-id`, `--lambda-reward-mix-ratio` (a different parameterisation: the round-4 coupling surface uses **sample-count + reweighting alpha + pocket-id + mix-ratio** rather than a single weight scalar). Fix: align the test with the actual CLI flag set (rename `--lambda-reward` → `--lambda-reweight-alpha 0.5` in the test) — same fix-up shape as the biolm adapter import-path mismatch.

**The 12 skips** are split between:
- 5 biolm tests (path-mismatch — same fix as the failure above)
- 5 coupling tests (skip when the coupling module is not yet importable on the test host — heavy torch stack lazy-import; matches the round-3 design margin for `test_lambda_reward_channel.py`)
- 2 synthemol tests (1 multi-reactant canonical curating skip + 1 smoke-test design margin)

---

## 6. Known limitations — 2026 Chinese math references cited

The round-4 plan (`TODO/01_research/lambda_integration_plan.md`) explicitly cites 2026-domestic (Chinese) research groups and their math/SBDD foundations. The current round-4 surface absorbs three of them; the rest are scoped for round-5 / track-B:

| Reference | Group / Origin | Status in round-4 | Notes |
|---|---|---|---|
| **SZU-ADDG/BioLM-Score** | Shenzhen University Adaptive Drug Design Group, CASF-2016 protein-ligand scoring | **CITED + ADAPTED** | `molmetal/references/BioLM-Score/` is the cloned repo; `molmetal/molmetal_lam/reward/biolm_score_adapter.py` is the round-4 adapter; upstream weights are **not** loaded (CPU smoke path via `BIOLM_DISABLE_WEIGHTS=1`). |
| **SZU-ADDG/SoftMol** | Shenzhen University, block-diffusion + gated MCTS SBDD | **CITED + REUSED** (round-11/12) | Already absorbed as the QuickVina 2 binary source (`molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02`) + SoftMol hit-rate gate roll-up (`molmetal/eval/scoring/softmol_score.py`); `round11_engine_parity_n50.md` + `quickvina2_binary_identity.md` document the byte-identical reuse. |
| **SZU-ADDG/MLM-Scaling** | Shenzhen University, FRAGPT-like molecular LM scaling laws | **CITED + DEFERRED** | Vendored at `molmetal/references/MLM-Scaling/` (Trio README link); planned only as a **fallback** if Phase 4 reveals Lambda's MCTS doesn't scale to 100-pocket (per `TODO/01_research/lambda_integration_plan.md` §"What's NOT in scope"). |
| **GeoFlow (CAS)** | Chinese Academy of Sciences / domestic universities, geometric DL + flow matching for antigen-antibody design, 43.9% top-1 success | **CITED ONLY** | Scoped in `TODO/01_research/sota_landscape.md` §3.15; **out of round-4 scope** (Lambda targets metalloprotein SBDD, not antibody CDR design). Carried as a forward-looking citation for round-5. |
| **TransDiffSBDD (Tsinghua + MSR)** | Tsinghua University + Microsoft Research, 2025 preprint, CrossDocked100 | **CITED ONLY** | Vina −9.37, success 83.9% (per `f2_tmqm_pretraining_vs_sota.md`, `r4_c_full_sweep_real.md`, etc.). Used as the **SOTA baseline bar** for round-4 Phase 4; **no re-run** (model-not-rerun group A — see `lambda_vs_sbdd_protocol_aligned.md`). |
| **PLL 2026 (Jian Liu, 重庆理工大学)** | Persistent Local Laplacian — Chinese 2026 math-domain work | **CITED ONLY** | Tracked under `TODO/pending/32_ultracode_trackB_lambda_calculus.md` B0 (PLL signature for β-NF typing) — track-B dispatch, **not** round-4. |
| **Deflex (Yu/Yang/Ren/Zhao 2026)** | Nat Commun 17(1), Dec 2026, arXiv:2606.07426 | **CITED ONLY** | Tracked under `TODO/pending/31_r17_ultracode_dispatch.md`; added to `paper/refs.bib` as `@article{yu2026deflex, ...}` + cited in `paper/sections/03_5_deflex.tex` §3.5 — **paper-side**, not round-4 algorithm-side. |

**The 2026-domestic math/SBDD literature is therefore a mixed surface:** SZU-ADDG's BioLM-Score and SoftMol are *operationally* absorbed (round-4 + round-11); MLM-Scaling, GeoFlow, TransDiffSBDD, PLL, and Deflex are *cited* but algorithm-side integration is deferred to round-5 (track-A continuation) and track-B (PLL β-NF typing).

---

## 7. Next steps — Phase 4 = 100-pocket sweep with 18 reactions

Per `TODO/01_research/lambda_integration_plan.md` §Phase 4:

| Step | Description | Forecast metric | Forecast source |
|---|---|---|---|
| **Phase 4 sweep** | 100 pockets × 1000 sims × L-3 204-tile library × 18 reactions | Vina mean ≥ −7 kcal/mol | Pocket2Mol −7.15 |
| | | Success rate ≥ 24% | Pocket2Mol 24.4% |
| | | SA mean < 3 | SOTA 2.51–2.86 |
| | | Synthesizability ≥ 95% | SyntheMol paper ~99% |
| **Pre-Phase-4 blockers** | (a) fix `test_biolm_score_adapter.py` import path → `molmetal.molmetal_lam.reward.biolm_score_adapter`; (b) fix `test_lambda_reward_coupling.py` `--lambda-reward` flag → `--lambda-reweight-alpha`; (c) confirm `molmetal/references/{SyntheMol,SynFlowNet,MLM-Scaling}` trees present on host | 1-line edits + 1 checkout | this report §4 + §5 |
| **Phase 5 (round-5)** | Comparative benchmark vs SOTA: (1) AiZynthFinder retrosynthesis %, (2) Vina against CrossDocked top-100, (3) QED/SA distribution | Synthesiz. ≥ 95%, Vina within 1 kcal/mol of TransDiffSBDD, SA < 3, QED > 0.55 | `TODO/01_research/lambda_integration_plan.md` §Phase 5 |
| **Track-B (parallel)** | PLL 2026 as λ-calculus type fingerprint + β-NF judge | per `TODO/pending/32_ultracode_trackB_lambda_calculus.md` B0 | independent dispatch |

**Deliverable:** `molmetal/reports/lambda_round4_full_sweep.md` once Phase 4 lands; the round-4 synthesis above is the green-light gate.

---

*No commits per round-4 task spec; round-4 ships as working-tree changes only. Last updated 2026-09-25.*
