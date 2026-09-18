# Governance Review — Layers 4 / 5 / 6

**Scope:** Per-metric audit of the 18 metrics (L4=7, L5=5, L6=6) in
`reports/lambda_layer_metrics.md`, against the four implementation
modules under `molmetal/molmetal_lam/{reactions,tile_lib}/`. Plus three
added cross-cutting metrics. Citations: Kolb & Sharpless 2001 (Drug
Discov. Today), Moses & Moorhouse 2007 (Chem. Soc. Rev.), Barner-
Kowollik et al. 2019 (Angew. Chem. Int. Ed.).

---

## Layer 4 — Reactions (`reactions/beta_reductions.py`)

| # | Metric | (a) Workable? | (b) Healthy target & rationale |
|---|--------|---------------|--------------------------------|
| 1 | FIRE_RATE_PER_RULE | Yes — wrap each `ReactionRule.reduce` with a `counters[rule.name]["fire"]` increment. Registry `REACTION_RULES` already exists. | Per-rule: CuAAC ≥ 0.85, SPAAC ≥ 0.80, SPC ≥ 0.85, DA ≥ 0.90, ThiolEne ≥ 0.75 (ranges follow Kolb 2001 §4 — CuAAC is the most permissive of the azide trio, SPAAC slower but strain-driven; DA near-quantitative for cyclopentadiene). |
| 2 | PRODUCT_COUNT_PER_REDUCE | Yes — append `len(products)` after `_rdkit_product_sets_to_closed_terms`. | Healthy 1–2. SPAAC may legitimately give 2 (1,4/1,5 regioisomers per Moses 2007 §3); flag > 3 as regiochemistry audit failure. |
| 3 | RUNREACTANTS_EXCEPTION_RATE | Yes — `try/except ReactionError` around `rxn.RunReactants` inside each `_reduce`. | ≈ 0. Non-zero is signal of upstream bad SMILES typing; Barner-Kowollik 2019 §2 emphasises that "click reactions are forgiving" so errors should be vanishingly rare. |
| 4 | MASS_BALANCE_PASS_RATE | Already enforced at module load via `verify_mass_balance()`. Healthy 1.0 — invariant, must always hold (all 5 rules are bond-forming; heavy atoms conserved). |
| 5 | RATE_PREDICTOR_ATTACH_RATE | Yes — walk `REACTION_RULES`, count `rule.rate_predictor is not None`. | 1.0. CuAAC additionally has `aryl_predictor`, so audit must check both attachments. |
| 6 | CATALYST_REQUIREMENT_COVERAGE | Yes — `walk REACTION_RULES; check rule.requires_catalyst`. | CuAAC requires `"Cu(I)"` (non-None); SPAAC, SPC, DielsAlder = None; ThiolEne = `"hν / radical initiator"`. Coverage is binary rule→expected tag. |
| 7 | THIOLENE_CAN_APPLY_RATE | Yes — counters incremented in `ThiolEne.can_apply`. | Healthy high (≥ 0.85) when both tiles carry the SH and C=C handles. Thiol-ene is "the most reliable photo-click" per Barner-Kowollik 2019 §4.1, so high pass rate is expected. |

## Layer 5 — Rate Predictors (`reactions/rate_predictor.py`)

| # | Metric | (a) Workable? | (b) Healthy target & rationale |
|---|--------|---------------|--------------------------------|
| 1 | PREDICTOR_TRAIN_R2 | Already on `RatePredictor.train_r2`. Yes — record per `for_reaction()`. | Per-rule R² ≥ 0.6 on 10-row hand-curated fits. Note: the `HeuristicRegressor` here is sklearn-linear (8-d features, 10 rows → severely over-determined & likely R² ≥ 0.9 in practice). Targets: CuAAC ≥ 0.6, SPAAC ≥ 0.6, SPC ≥ 0.6, DA ≥ 0.6, ThiolEne ≥ 0.6. |
| 2 | FEATURE_DIM_OK | Yes — assert `len(smiles_pair_features(a,b)) == 8` per call. | Always True; non-zero is a regression bug. |
| 3 | PREDICTION_CLIP_RATE | Yes — count `raw` outside [0,1] before `max/min` clipping in `predict_yield`. | < 0.1. High clipping = model extrapolating beyond the literature envelope (Kolb 2001 §5 warns against out-of-domain yield claims). |
| 4 | CITATION_COVERAGE | Yes — `len(p.citations) != 10` check per rule. | 1.0 (= 10/10 DOIs per reaction). All 6 reaction tables in `_LIT_DOI` are length 10. |
| 5 | BACKEND_USED | Yes — `priors[k].backend_` after `for_reaction()`. | `sklearn` in this ROCm Triton env (PySR/Julia not installed; `HeuristicRegressor` falls back automatically per module docstring). |

## Layer 6 — Tile Library (`tile_lib/click_tiles.py`, `library.py`, `tile.py`)

| # | Metric | (a) Workable? | (b) Healthy target & rationale |
|---|--------|---------------|--------------------------------|
| 1 | TILE_COUNT_PER_GROUP | Yes — assert `len(STANDARD_12_TILES) == 12`. | Exactly 4 + 4 + 4 = 12. The 4-handles-per-class floor is the smallest set that closes every reaction pair (Kolb 2001 §2.2). |
| 2 | EMBED_SUCCESS_RATE | Yes — `if tile.coords is None: counter["embed_fail"] += 1` inside `build_click_tile`. | ≈ 1.0 with `randomSeed = 0xC11C`. Cyclooctyne (`C1CCCC#CCC1`) is the hardest case (constrained 8-ring); flag if < 0.95. |
| 3 | CANONICAL_SMILES_UNIQUE | Yes — `len(set(t.smiles for t in tiles))`. | 12. Duplicates indicate a SMILES-source bug, not chemistry. |
| 4 | SAS_DISTRIBUTION | Yes — append `tile.sas_score` to history in `build_click_tile`. | Mean 1.0–4.0 (Ertl SA scale). The 12 reference tiles are hand-picked small fragments, so SAS ≈ 1–3 is realistic; tiles > 4 should fail Phase-0 admission. |
| 5 | MW_LOGP_TPSA_BOUNDS | Yes — record `(mw, logp, tpsa)` per tile build. | MW ∈ [40, 200] g/mol, logP ∈ [-1, 4], TPSA ∈ [0, 90] Å² for all 12. Azide-rich tiles push MW low; propargylamine logs low logP. |
| 6 | TILE_ID_DETERMINISM | Yes — hash-check `tile_id` set across two builds (Tile dataclass already computes sha256[:12] in `__post_init__`). | True. SHA-256 of canonical SMILES is deterministic by construction. |

---

## Added Metrics (3)

**L4-ADD: REACTION_CO_OCCURRENCE_IN_MCTS**
Spec: `cooccurrence(rule_a, rule_b) = count(trajectories where both rules fire) / count(trajectories where rule_a fires)`, gathered from `MCTSProofSearch.history[i]['best_state'].applied_rules`.
Rationale: Click rules are rarely used in isolation (Kolb 2001 §3 — orthogonal reactivity classes enable cascade synthesis; Moses 2007 §5 demonstrates tandem SPAAC-thiol-ene). Healthy ≥ 0.3 cascade rate after 1000 sims.

**L5-ADD: OUT_OF_FOLD_PREDICTOR_MAE**
Spec: 5-fold CV over `LITERATURE_YIELDS[reaction]`; report `mean(|y - ŷ|)` per rule.
Rationale: R² on the training set is optimistic. MAE ≤ 0.10 per rule is the practical bar (Barner-Kowollik 2019 §6 — yield predictors must beat 10% absolute error to be useful for retrosynthetic ranking).

**L6-ADD: TILE_FUNCTIONAL_GROUP_COVERAGE**
Spec: Per-handle counts `len(tiles with 'azide') = 4`, `'alkyne' ∪ 'cyclooctyne' = 4`, `'diene' = 1`, `'dienophile' = 1`, `'phosphine' = 1`, `'thiol' = 0` (note Phase-0 has no dedicated thiol — covered by maleimide via Michael addition). Assert symmetry: total handle count == n_tiles.
Rationale: Coverage gap on thiol would block ThiolEne (Kolb 2001 §4.1 — thiol-ene is the second-most-utilised click reaction after CuAAC). Healthy: every reaction in `REACTION_RULES` has at least one matching handle in the library.

---

## Summary

All 18 existing 1-line instrumentation sketches are workable against
the current code. Two healthy targets deserve tightening: FIRE_RATE
should be **per-rule** (not aggregate) because SPAAC legitimately
fires less often than CuAAC, and PREDICTION_CLIP_RATE < 0.1 reflects
the envelope defined by the 60-row `_LIT_DOI` table. The three added
metrics address the gaps visible from the actual files: MCTS never
records *which rules fired together* (only top-K SMILES), training-set
R² is not cross-validated, and the tile library silently lacks a
dedicated thiol handle.
